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

"""makani's own layout in zarr: one ``????.zarr`` store per year.

The same ``(sample, channel, lat, lon)`` array as the HDF5 layout, in a zarr
store. The differences that matter here are that zarr groups are lazy views
rather than open handles, so there is nothing to close, and that filling a
caller's buffer goes through ``get_basic_selection(out=...)`` with an
``NDBuffer`` rather than ``read_direct``.
"""

import glob
import logging
import os
from typing import List, Optional, Sequence

import numpy as np
import zarr

# zarr v3 requires the `out` target of get_basic_selection to be an NDBuffer rather than a
# raw numpy array. from_ndarray_like wraps the (possibly strided) view without copying, so
# decoded data still lands directly in the preallocated buffer (zero-copy read).
from zarr.core.buffer.cpu import NDBuffer as _ZarrNDBuffer

from ..data_helpers import get_lat_lon_grid, get_timestamp
from .base import (
    BackendMetadata,
    DatasetBackend,
    GridSpec,
    order_files_by_time,
    timestamp_converter,
)
from .makani_hdf5 import StructuredChunkMixin, contiguous_slices


def zarr_out(array):
    """Wrap a numpy view as the buffer type zarr v3 writes into."""
    return _ZarrNDBuffer.from_ndarray_like(array)


def zarr_open(path, mode="r"):
    """Open a store, preferring consolidated metadata.

    Consolidated metadata is one round trip for every array's metadata rather
    than one per array, which matters on object storage; stores that were never
    consolidated still open, just more slowly.
    """
    try:
        return zarr.open_consolidated(path, mode=mode)
    except (KeyError, ValueError):
        # the two ways a store says it was never consolidated: no metadata
        # document at all (KeyError), or one that carries no consolidated block
        # (ValueError, which is what zarr v3 raises). Either way the group is
        # still there to be opened the slow way.
        return zarr.open_group(path, mode=mode)


class MakaniZarrBackend(StructuredChunkMixin, DatasetBackend):
    """Reads the makani layout from zarr, one store per year."""

    #: one store per year, named after it
    default_file_pattern = "????"

    _handle_attributes = ("files", "dsets")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files: List = []
        self.dsets: List = []

    # ---- discovery ---------------------------------------------------------

    def _find_files(self) -> List[str]:
        paths = []
        for location in self.location:
            paths += glob.glob(os.path.join(location, f"{self.file_pattern}.zarr"))
        return sorted(paths)

    @staticmethod
    def _label_from_name(path) -> Optional[int]:
        """The year in a store name, where the name is one. None otherwise."""
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem) if stem.isdigit() else None

    def _read_coordinates(self, group):
        """Grid coordinates carried by the store, or None if it carries none."""
        for latitude_name, longitude_name in (
            (self.latitude_name, self.longitude_name),
            # the xarray convention, which is what a store written by xarray uses
            ("latitude", "longitude"),
        ):
            if latitude_name in group and longitude_name in group:
                return np.asarray(group[latitude_name]), np.asarray(group[longitude_name])
        return None

    def _probe(self, group):
        """Return the array whose shape and time axis describe this store."""
        return group[self.dataset_name]

    def _read_timestamps(self, group, dset, label, timezone_fn) -> np.ndarray:
        """Read the time coordinate, or synthesize it from the store's label."""
        # the makani name first, then the xarray convention
        candidates = [self.timestamp_name]
        if "time" != self.timestamp_name:
            candidates.append("time")

        for key in candidates:
            if key in group:
                raw = np.asarray(group[key])
                # WB2 stores use datetime64[ns]; convert to float unix seconds
                if np.issubdtype(raw.dtype, np.datetime64):
                    raw = raw.astype("datetime64[s]").astype(np.float64)
                return timezone_fn(raw)

        if label is None:
            raise ValueError(
                f"The store carries no '{self.timestamp_name}' coordinate, and its name is not a year to derive "
                "the sample times from."
            )

        synthesized = np.asarray(
            # one timestamp per sample, dhours apart -- the step belongs in the
            # hour, not in the range
            [get_timestamp(label, hour=(idx * self.dhours)).timestamp() for idx in range(dset.shape[0])]
        )
        return timezone_fn(synthesized)

    def _describe(self, group, probe):
        """Spatial shape and channel count of a store. Overridden per layout."""
        return (probe.shape[-2], probe.shape[-1]), probe.shape[1]

    def discover(self, enable_logging: bool = False) -> BackendMetadata:
        files = self._find_files()
        if not files:
            locations = ", ".join(self.location)
            raise IOError(f"Error, the specified file path(s) {locations} do not contain zarr stores.")

        labels = [self._label_from_name(path) for path in files]
        timezone_fn = timestamp_converter(self.relative_timestamp)

        samples_per_file = []
        timestamps = []
        img_shape = None
        total_channels = None
        coordinates = None

        for file_idx, path in enumerate(files):
            group = zarr_open(path)
            if enable_logging and file_idx == 0:
                logging.info("Getting file stats from {}".format(path))

            probe = self._probe(group)
            if img_shape is None:
                img_shape, total_channels = self._describe(group, probe)
                coordinates = self._read_coordinates(group)
            samples_per_file.append(probe.shape[0])
            timestamps.append(self._read_timestamps(group, probe, labels[file_idx], timezone_fn))

        files, labels, samples_per_file, timestamps = order_files_by_time(files, labels, samples_per_file, timestamps)

        self.files = [None] * len(files)
        self.dsets = [None] * len(files)

        # what the caller asked for, else what the store says, else the assumption
        if self.lat_lon is not None:
            latitude, longitude = np.asarray(self.lat_lon[0]), np.asarray(self.lat_lon[1])
        elif coordinates is not None:
            latitude, longitude = coordinates
        else:
            latitude, longitude = get_lat_lon_grid(img_shape)

        grid = GridSpec("equiangular", tuple(img_shape), np.asarray(latitude), np.asarray(longitude))
        self.chunk = self._resolve_chunk(grid)
        self.metadata = BackendMetadata(
            files=files,
            labels=labels,
            samples_per_file=samples_per_file,
            timestamps=timestamps,
            grid=grid,
            total_channels=total_channels,
        )
        return self.metadata

    # ---- handles -----------------------------------------------------------

    def open(self, file_idx: int) -> None:
        if self.files[file_idx] is not None:
            return
        self.files[file_idx] = zarr_open(self.metadata.files[file_idx])
        self.dsets[file_idx] = self.files[file_idx][self.dataset_name]

    # ---- reading -----------------------------------------------------------

    def read(self, file_idx, time_slice, channels, out=None):
        self.open(file_idx)
        dset = self.dsets[file_idx]

        if out is None:
            n_times = len(range(*time_slice.indices(dset.shape[0])))
            out = np.empty((n_times, len(channels), *self.chunk.shape), dtype=dset.dtype)

        offset = 0
        for channel_slice in contiguous_slices(channels):
            width = channel_slice.stop - channel_slice.start
            dset.get_basic_selection(
                np.s_[time_slice, channel_slice, self.lat_slice, self.lon_slice],
                out=zarr_out(out[:, offset : offset + width, ...]),
            )
            offset += width

        return out


class ArcoWB2Backend(MakaniZarrBackend):
    """Reads a WeatherBench2 style store: one array per variable.

    Same container as :class:`MakaniZarrBackend` and the same discovery, but the
    channels are not an axis: each variable is its own array, and pressure levels
    are a dimension within the atmospheric ones. A channel therefore resolves to
    a (variable, level index) pair, and a read is one selection per channel
    rather than one per run of adjacent channels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.channel_names is None:
            raise ValueError("The WeatherBench2 layout addresses variables by name, so channel_names is required.")
        self.channel_map: Optional[Sequence] = None

    def _probe(self, group):
        # build the channel map from the first store, then probe any variable it names
        if self.channel_map is None:
            from ..wb2_helpers import build_wb2_channel_map

            level_values = np.asarray(group["level"]) if "level" in group else None
            self.channel_map = build_wb2_channel_map(self.channel_names, level_values)
        return group[self.channel_map[0][0]]

    def _describe(self, group, probe):
        # channels are variables here, so the count comes from the request
        return (probe.shape[-2], probe.shape[-1]), len(self.channel_names)

    def open(self, file_idx: int) -> None:
        if self.files[file_idx] is not None:
            return
        # the group itself is the handle: reads address individual variables
        self.files[file_idx] = zarr_open(self.metadata.files[file_idx])
        self.dsets[file_idx] = self.files[file_idx]

    def read(self, file_idx, time_slice, channels, out=None):
        self.open(file_idx)
        group = self.dsets[file_idx]

        if out is None:
            probe = group[self.channel_map[channels[0]][0]]
            n_times = len(range(*time_slice.indices(probe.shape[0])))
            out = np.empty((n_times, len(channels), *self.chunk.shape), dtype=probe.dtype)

        for position, channel in enumerate(channels):
            variable, level_idx = self.channel_map[channel]
            if level_idx is None:
                selection = np.s_[time_slice, self.lat_slice, self.lon_slice]
            else:
                selection = np.s_[time_slice, level_idx, self.lat_slice, self.lon_slice]
            group[variable].get_basic_selection(selection, out=zarr_out(out[:, position, ...]))

        return out
