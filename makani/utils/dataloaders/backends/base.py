# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The contract between a dataset layout and the sample assembly above it.

A backend answers three questions and nothing else:

* **what is there** -- which files, how many samples in each, at what times
  (:meth:`DatasetBackend.discover`);
* **what do I emit** -- the grid of the chunk this rank reads, and the
  coordinate of every point in it (:attr:`DatasetBackend.chunk`);
* **give me these values** -- read one slab of one file, over a range of
  timesteps, into a destination array (:meth:`DatasetBackend.read`).

Everything else -- which sample index maps to which file, how windows are built
from slabs, shuffling, epoch cycling, normalization -- belongs to the caller.
That split is what lets one implementation serve both the DALI sample source and
the multifiles torch dataset.

Grids, and what a backend does *not* do
---------------------------------------
A backend emits data on its **native** grid and says which grid that is. It does
not resample. A dataset on an unstructured mesh hands out cells, labelled as
such, and converting those onto whatever grid the model wants is the job of a
downstream layer -- :class:`makani.utils.grids.GridConverter`, which already runs
per batch on the GPU.

Keeping the resampling out here buys two things: it can run on the GPU rather
than in a data loader worker, and a model that wants the native mesh stays
possible.

The chunk a rank emits is fixed at construction: the crop, the spatial
decomposition and any subsampling are settled once, so :meth:`read` takes no
spatial argument and :attr:`chunk` can carry the coordinate of every point it
will ever emit. Consumers that need to build a resampling operator, or the solar
zenith angle, read those coordinates rather than reconstructing them from an
assumed grid.

Two aspects of the contract are easy to get wrong and are handled here rather
than in each backend:

**Reading into a destination.** :meth:`read` takes an ``out`` array and fills
it. Both storage libraries support this natively -- h5py through
``read_direct``, zarr through ``get_basic_selection(out=...)`` -- and it is what
lets the DALI path reuse preallocated buffers instead of allocating per sample.
A caller that does not care passes ``out=None`` and gets a fresh array.

**File handles cannot be pickled.** DALI runs the source in worker processes and
torch does the same for its dataloader, so handles have to be dropped on the way
out and reopened on the way in. :meth:`__getstate__` and :meth:`__setstate__`
here do that for every backend; a subclass only declares which attributes hold
handles, through :attr:`_handle_attributes`.
"""

import abc
import os
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..data_helpers import get_date_from_timestamp, get_timedelta_from_timestamp


def timestamp_converter(relative_timestamp: bool = False):
    """The function turning stored times into python objects.

    Absolute times become timezone aware datetimes. A dataset whose times are
    offsets rather than dates -- a climatology indexed by time of year, say --
    asks for ``relative_timestamp`` and gets timedeltas instead.
    """
    return np.vectorize(get_timedelta_from_timestamp if relative_timestamp else get_date_from_timestamp)


def order_files_by_time(files, labels, samples_per_file, timestamps):
    """Put files in the order the sample index runs through them.

    Filenames are not a reliable order: they are the year for the makani
    layouts, but a dataset can just as well be a directory of arbitrarily named
    files. What always orders them is the data itself, so the files are sorted
    by the time of their first sample.

    Sorting by first sample only says where each file starts. Sorting by last
    sample as well and requiring the same permutation is what rules out files
    that overlap -- if one file's range straddles another's, the two orders
    disagree. Overlapping files would make a global sample index ambiguous, so
    this is refused rather than resolved.

    Returns the four inputs reordered, with the timestamps concatenated.
    """
    starts = [times[0] for times in timestamps]
    ends = [times[-1] for times in timestamps]

    by_start = np.argsort(starts, kind="stable")
    by_end = np.argsort(ends, kind="stable")
    if not np.array_equal(by_start, by_end):
        overlapping = ", ".join(os.path.basename(files[idx]) for idx in by_start[:4])
        raise RuntimeError(
            "The files have overlapping time ranges, which makes a sample index ambiguous. "
            f"Please provide files with disjoint ranges (files in start order: {overlapping}...)."
        )

    order = by_start.tolist()
    return (
        [files[idx] for idx in order],
        [labels[idx] for idx in order],
        [samples_per_file[idx] for idx in order],
        np.concatenate([timestamps[idx] for idx in order], axis=0),
    )


class GridSpec(NamedTuple):
    """A grid, and the coordinate of every point on it.

    Attributes
    ----------
    kind : str
        Grid type in makani's vocabulary -- ``"equiangular"``,
        ``"legendre-gauss"``, ``"clenshaw-curtiss"`` -- or ``"unstructured"``
        for a mesh with no tensor product structure. This is what a run means by
        ``data_grid_type``, and what tells the downstream converter what it has
        been handed.
    shape : tuple of int
        Spatial shape: ``(nlat, nlon)`` for a raster, ``(ncells,)`` for a mesh.
        The number of entries is the number of spatial dimensions the data has.
    lat, lon : numpy.ndarray
        Coordinates in degrees. For a raster these are the two 1-D axes, of
        lengths matching ``shape``. For a mesh they are per cell, both of length
        ``ncells``. Either way, indexing them alongside the data gives the
        position of every emitted value, which is all a resampler needs.
    """

    kind: str
    shape: Tuple[int, ...]
    lat: np.ndarray
    lon: np.ndarray

    @property
    def is_structured(self) -> bool:
        """Whether the grid is a lat/lon raster rather than a mesh."""
        return len(self.shape) == 2


class BackendMetadata(NamedTuple):
    """What a backend reports about a dataset once it has looked at it.

    Attributes
    ----------
    files : list of str
        One entry per unit the sample index runs through, in order. For most
        layouts a unit is a file. It need not be: a dataset that splits its
        variables across files has a sample spanning several of them, and the
        entry is then one representative path, with the backend holding the rest
        internally. Nothing outside the backend opens these.
    labels : list of int
        A label per file, used for reporting and for deriving timestamps when a
        file carries none. For the makani layouts this is the year.
    samples_per_file : list of int
        Number of samples in each unit. The caller turns this into global
        offsets; a backend never sees a global sample index.
    timestamps : numpy.ndarray
        One timezone aware datetime per sample, concatenated across files.
    grid : GridSpec
        The full grid of the dataset, before cropping or decomposition. What a
        given rank actually emits is :attr:`DatasetBackend.chunk`.
    total_channels : int
        Channels available in the dataset, not the number requested.
    """

    files: List[str]
    labels: List[int]
    samples_per_file: List[int]
    timestamps: np.ndarray
    grid: GridSpec
    total_channels: int


class DatasetBackend(abc.ABC):
    """Locates and reads samples of one dataset layout.

    Parameters
    ----------
    location : str or list of str
        Directory, directories or file the dataset lives in.
    dataset_name : str
        Name of the array holding the fields, where the layout has one.
    timestamp_name : str
        Name of the coordinate holding sample times, where the layout has one.
    latitude_name, longitude_name : str
        Names of the coordinates holding the grid. Read from the file when it
        carries them, so that a dataset on a grid other than the assumed
        equiangular one is described by its own coordinates rather than by
        fabricated ones.
    file_pattern : str, optional
        Glob for the file names, without the extension. Defaults to
        :attr:`default_file_pattern`, which is whatever the layout names its
        files: the makani layouts use ``"????"``, a file per year, while ICON
        names its files after the variable and the date. A dataset of
        arbitrarily named files in a makani layout passes ``"*"`` and is ordered
        by time instead of by name.
    relative_timestamp : bool
        Whether the stored times are offsets rather than dates, as in a
        climatology indexed by time of year.
    dhours : int
        Hours between consecutive samples, used to synthesize timestamps for
        datasets that were never annotated with them.
    channel_names : list of str, optional
        Channel names the run asks for. Only layouts that address variables by
        name rather than by index need these.
    crop_anchor, crop_size : sequence, optional
        Region of the full grid to restrict to, before decomposition. Meaningful
        only for structured grids; a mesh backend may reject them.
    io_grid, io_rank : sequence of int, optional
        Spatial decomposition and this rank's place in it, as ``[lat, lon]``.
    subsampling_factor : int
        Take every n-th point of the chunk.

        This is decimation, not resampling: it consults no neighbours, which is
        what lets it happen during the read and save the bytes it discards. The
        price is aliasing, and that it has no meaning on an unstructured mesh,
        where index order is not a spatial pattern -- a mesh backend should
        reject anything but 1 and point the caller at choosing a coarser target
        grid instead. A proper downsampling layer, where the method is the
        user's choice (average pooling, spectral truncation), is intended to
        replace this.
    lat_lon : tuple, optional
        Coordinates to use instead of those found in the files.
    target_grid : GridSpec, optional
        The grid the run will ultimately work on, when it differs from the one
        the data is stored on.

        A backend whose own grid *is* the target ignores this. A mesh backend
        needs it to turn ``io_grid``/``io_rank`` into a geographic region and
        so decide which of its cells to read: the point is that a rank should
        read the cells covering the lat/lon block it is responsible for, so that
        the resampling downstream is a local gather with nothing to exchange.
        Selecting cells that way needs a margin around the block, since an
        interpolation stencil reaches beyond it.

        Note that this tells the backend *where* it will be needed, not what to
        do about it: resampling still happens downstream.
    """

    #: how this layout names its files, without the extension. Permissive here
    #: because a layout that has no convention should not impose one; the makani
    #: layouts override it with the year they are named after.
    default_file_pattern: str = "*"

    #: attributes holding open file handles, dropped when pickled
    _handle_attributes: Tuple[str, ...] = ()

    #: attributes holding anything else that cannot be pickled -- sessions,
    #: clients, connections -- set to None on the way out and rebuilt by
    #: :meth:`_restore` on the way in
    _transient_attributes: Tuple[str, ...] = ()

    def __init__(
        self,
        location,
        dataset_name: str = "fields",
        timestamp_name: str = "timestamp",
        latitude_name: str = "lat",
        longitude_name: str = "lon",
        file_pattern: Optional[str] = None,
        relative_timestamp: bool = False,
        dhours: int = 6,
        channel_names: Optional[Sequence[str]] = None,
        crop_anchor: Optional[Sequence[int]] = None,
        crop_size: Optional[Sequence[Optional[int]]] = None,
        io_grid: Optional[Sequence[int]] = None,
        io_rank: Optional[Sequence[int]] = None,
        subsampling_factor: int = 1,
        lat_lon=None,
        target_grid: Optional[GridSpec] = None,
    ):
        self.location = location if isinstance(location, list) else [location]
        self.dataset_name = dataset_name
        self.timestamp_name = timestamp_name
        self.latitude_name = latitude_name
        self.longitude_name = longitude_name
        self.file_pattern = file_pattern if file_pattern is not None else self.default_file_pattern
        self.relative_timestamp = relative_timestamp
        self.dhours = dhours
        self.channel_names = channel_names

        self.crop_anchor = list(crop_anchor) if crop_anchor is not None else [0, 0]
        self.crop_size = list(crop_size) if crop_size is not None else [None, None]
        self.io_grid = list(io_grid) if io_grid is not None else [1, 1]
        self.io_rank = list(io_rank) if io_rank is not None else [0, 0]
        self.subsampling_factor = subsampling_factor
        self.lat_lon = lat_lon
        self.target_grid = target_grid

        self.metadata: Optional[BackendMetadata] = None
        self.chunk: Optional[GridSpec] = None

    # ---- discovery ---------------------------------------------------------

    @abc.abstractmethod
    def discover(self, enable_logging: bool = False) -> BackendMetadata:
        """Inspect the dataset and settle what this rank will emit.

        Called once, in the parent process, before any worker exists. Sets both
        :attr:`metadata` and :attr:`chunk`, since the caller needs the first to
        build its sample index and the second to size its buffers.
        """

    @property
    def num_files(self) -> int:
        if self.metadata is None:
            raise RuntimeError("discover() has to be called before the dataset can be used.")
        return len(self.metadata.files)

    # ---- handles -----------------------------------------------------------

    @abc.abstractmethod
    def open(self, file_idx: int) -> None:
        """Open one file and cache its handle.

        Called lazily from the worker that will read it. Implementations are
        expected to be idempotent: opening an already open file is a no-op.
        """

    def close(self) -> None:
        """Release every open handle."""
        for attribute in self._handle_attributes:
            handles = getattr(self, attribute, None)
            if not handles:
                continue
            for handle in handles:
                # zarr groups have no close(), only HDF5 handles do
                if handle is not None and hasattr(handle, "close"):
                    handle.close()
            setattr(self, attribute, [None] * len(handles))

    # ---- reading -----------------------------------------------------------

    @abc.abstractmethod
    def read(
        self,
        file_idx: int,
        time_slice: slice,
        channels: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Read one time range of this rank's chunk.

        Parameters
        ----------
        file_idx : int
            Index into :attr:`BackendMetadata.files`.
        time_slice : slice
            Samples to read, *local to this file*, possibly strided. Windows
            spanning files are the caller's problem: it issues one read per file.
        channels : numpy.ndarray
            Channel indices to read, **sorted ascending**. Sorted because most
            layouts can only read monotonic selections efficiently; the caller
            restores the requested order afterwards.
        out : numpy.ndarray, optional
            Destination of shape ``(n_times, len(channels), *chunk.shape)``.
            When given it is filled in place and returned, which is what lets the
            caller reuse buffers rather than allocate per sample.

        Returns
        -------
        numpy.ndarray
            ``out`` when it was given, otherwise a freshly allocated array.
        """

    # ---- pickling ----------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Drop what cannot cross a process boundary: handles, and connections."""
        state = self.__dict__.copy()
        for attribute in self._handle_attributes:
            handles = state.get(attribute)
            if handles is not None:
                state[attribute] = [None] * len(handles)
        for attribute in self._transient_attributes:
            state[attribute] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._restore()

    def _restore(self) -> None:
        """Rebuild whatever could not be pickled. Overridden where needed."""
        return

    def __del__(self):
        try:
            self.close()
        except Exception:
            # interpreter shutdown can pull the libraries out from under us, and
            # failing to close a file at that point is not worth an exception
            pass
