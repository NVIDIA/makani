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

"""makani's own HDF5 layout: one ``????.h5`` file per year.

Each file holds a single 4-D array, ``(sample, channel, lat, lon)``, optionally
annotated with a ``timestamp`` dimension scale. Files whose name is the year and
which carry no timestamps have theirs synthesized from that year and ``dhours``.

This backend also owns the two access modes that only apply to HDF5: O_DIRECT,
which bypasses the page cache for large sequential reads, and the ROS3 driver
for reading from S3 without staging.
"""

import glob
import logging
import os
from functools import partial
from itertools import groupby
from typing import List, Sequence

import h5py
import numpy as np
from torch_harmonics.distributed import compute_split_shapes

from ..aws_connector import get_default_aws_connector
from ..data_helpers import get_date_from_timestamp, get_lat_lon_grid, get_timestamp
from .base import BackendMetadata, DatasetBackend, GridSpec


def contiguous_slices(indices: Sequence[int]):
    """Group ascending indices into the fewest contiguous slices.

    HDF5 reads a monotonic slice far faster than a fancy index, so a channel
    selection like ``[0, 1, 2, 7, 8]`` is read as two slabs rather than five
    scattered elements.
    """
    for _, group in groupby(enumerate(indices), lambda pair: pair[1] - pair[0]):
        group = list(group)
        yield slice(group[0][1], group[-1][1] + 1)


class StructuredChunkMixin:
    """Turns a crop and a spatial decomposition into this rank's slice of a raster.

    Shared by every backend whose storage is already a lat/lon raster: the chunk
    is a rectangle of the full grid, and its coordinates are the corresponding
    stretches of the two axes.
    """

    def _resolve_chunk(self, grid: GridSpec) -> GridSpec:
        img_shape = grid.shape
        crop_size = list(self.crop_size)
        if crop_size[0] is None:
            crop_size[0] = img_shape[0]
        if crop_size[1] is None:
            crop_size[1] = img_shape[1]

        for dim in (0, 1):
            if self.crop_anchor[dim] + crop_size[dim] > img_shape[dim]:
                raise ValueError(
                    f"crop in dimension {dim} (anchor {self.crop_anchor[dim]} + size {crop_size[dim]}) "
                    f"exceeds image shape {img_shape[dim]}"
                )
        self.crop_size = crop_size

        # split the crop over the decomposition and take this rank's piece
        anchors, shapes = [], []
        for dim in (0, 1):
            splits = compute_split_shapes(crop_size[dim], self.io_grid[dim])
            shapes.append(splits[self.io_rank[dim]])
            anchors.append(self.crop_anchor[dim] + sum(splits[: self.io_rank[dim]]))

        self.read_anchor = anchors
        self.read_shape = shapes
        self.lat_slice = slice(anchors[0], anchors[0] + shapes[0], self.subsampling_factor)
        self.lon_slice = slice(anchors[1], anchors[1] + shapes[1], self.subsampling_factor)

        return GridSpec(
            kind=grid.kind,
            shape=(
                len(range(*self.lat_slice.indices(img_shape[0]))),
                len(range(*self.lon_slice.indices(img_shape[1]))),
            ),
            lat=grid.lat[self.lat_slice],
            lon=grid.lon[self.lon_slice],
        )


class MakaniHDF5Backend(StructuredChunkMixin, DatasetBackend):
    """Reads the makani HDF5 layout, one file per year.

    Parameters
    ----------
    enable_odirect : bool
        Read through the O_DIRECT driver, bypassing the page cache.
    odirect_alignment : int
        Alignment and block size for O_DIRECT, when the default does not suit
        the filesystem.
    enable_s3 : bool
        Read through the ROS3 driver instead of the local filesystem. Mutually
        exclusive with O_DIRECT, and disables ``read_direct``, which the ROS3
        driver does not support.
    """

    _handle_attributes = ("files", "dsets")
    # the S3 connector holds a session that does not survive pickling
    _transient_attributes = ("aws_connector",)

    def __init__(
        self,
        *args,
        enable_odirect: bool = False,
        odirect_alignment: int = 0,
        enable_s3: bool = False,
        **kwargs,
    ):
        # only the arguments this layout adds are named here; everything the
        # contract defines is forwarded, so adding a parameter to the base does
        # not silently become an unexpected keyword here
        super().__init__(*args, **kwargs)

        if enable_odirect and enable_s3:
            raise NotImplementedError("The settings enable_odirect and enable_s3 are mutually exclusive.")

        self.enable_s3 = enable_s3
        self.file_driver = None
        self.file_driver_kwargs = {}
        self.aws_connector = None

        if enable_odirect:
            self.file_driver = "direct"
            if odirect_alignment > 0:
                self.file_driver_kwargs = dict(alignment=odirect_alignment, block_size=odirect_alignment)

        if enable_s3:
            self.file_driver = "ros3"
            self.aws_connector = get_default_aws_connector(None)
            self.file_driver_kwargs = dict(
                aws_region=bytes(self.aws_connector.aws_region_name, "utf-8"),
                secret_id=bytes(self.aws_connector.aws_access_key_id, "utf-8"),
                secret_key=bytes(self.aws_connector.aws_secret_access_key, "utf-8"),
            )

        # read_direct fills a preallocated buffer without an intermediate copy,
        # but the ROS3 driver does not implement it
        self.read_direct = not self.enable_s3

        self.files: List = []
        self.dsets: List = []

    # ---- discovery ---------------------------------------------------------

    def _find_files(self) -> List[str]:
        if not self.enable_s3:
            paths = []
            for location in self.location:
                paths += glob.glob(os.path.join(location, "????.h5"))
            return sorted(paths)

        paths = [
            self.aws_connector.aws_endpoint_url + "/" + path
            for path in self.aws_connector.list_bucket(self.location)
            if path.endswith(".h5")
        ]
        return sorted(paths)

    def _open_for_stats(self):
        if not self.enable_s3:
            return partial(h5py.File, mode="r")
        return partial(h5py.File, mode="r", driver=self.file_driver, **self.file_driver_kwargs)

    def discover(self, enable_logging: bool = False) -> BackendMetadata:
        files = self._find_files()
        if not files:
            locations = ", ".join(self.location)
            raise IOError(f"Error, the specified file path(s) {locations} do not contain h5 files.")

        labels = [int(os.path.splitext(os.path.basename(path))[0]) for path in files]

        timezone_fn = np.vectorize(get_date_from_timestamp)
        fopen = self._open_for_stats()

        samples_per_file = []
        timestamps = []
        img_shape = None
        total_channels = None

        for file_idx, path in enumerate(files):
            with fopen(path) as handle:
                if enable_logging and file_idx == 0:
                    logging.info("Getting file stats from {}".format(path))
                dset = handle[self.dataset_name]
                if img_shape is None:
                    img_shape = dset.shape[2:4]
                    total_channels = dset.shape[1]
                samples_per_file.append(dset.shape[0])

                if self.timestamp_name in dset.dims[0]:
                    timestamps.append(timezone_fn(dset.dims[0][self.timestamp_name][...]))
                else:
                    # unannotated file: derive the times from the year in its name.
                    # one timestamp per sample, dhours apart -- the step belongs
                    # in the hour, not in the range
                    synthesized = np.asarray(
                        [
                            get_timestamp(labels[file_idx], hour=(idx * self.dhours)).timestamp()
                            for idx in range(dset.shape[0])
                        ]
                    )
                    timestamps.append(timezone_fn(synthesized))

        self.files = [None] * len(files)
        self.dsets = [None] * len(files)

        if self.lat_lon is None:
            latitude, longitude = get_lat_lon_grid(img_shape)
        else:
            latitude, longitude = np.asarray(self.lat_lon[0]), np.asarray(self.lat_lon[1])

        grid = GridSpec("equiangular", tuple(img_shape), np.asarray(latitude), np.asarray(longitude))
        self.chunk = self._resolve_chunk(grid)
        self.metadata = BackendMetadata(
            files=files,
            labels=labels,
            samples_per_file=samples_per_file,
            timestamps=np.concatenate(timestamps, axis=0),
            grid=grid,
            total_channels=total_channels,
        )
        return self.metadata

    # ---- handles -----------------------------------------------------------

    def open(self, file_idx: int) -> None:
        if self.files[file_idx] is not None:
            return
        self.files[file_idx] = h5py.File(
            self.metadata.files[file_idx], "r", driver=self.file_driver, **self.file_driver_kwargs
        )
        self.dsets[file_idx] = self.files[file_idx][self.dataset_name]

    def _restore(self) -> None:
        # the connector holds a session that does not survive pickling
        if self.enable_s3 and self.aws_connector is None:
            self.aws_connector = get_default_aws_connector(None)

    # ---- reading -----------------------------------------------------------

    def read(self, file_idx, time_slice, channels, out=None):
        self.open(file_idx)
        dset = self.dsets[file_idx]

        if out is None:
            n_times = len(range(*time_slice.indices(dset.shape[0])))
            out = np.empty((n_times, len(channels), *self.chunk.shape), dtype=dset.dtype)

        # read each run of adjacent channels as one slab, writing it to the
        # matching span of the destination
        offset = 0
        for channel_slice in contiguous_slices(channels):
            width = channel_slice.stop - channel_slice.start
            source = np.s_[time_slice, channel_slice, self.lat_slice, self.lon_slice]

            if self.read_direct:
                dset.read_direct(out, source, np.s_[:, offset : offset + width, ...])
            else:
                out[:, offset : offset + width, ...] = dset[source]

            offset += width

        return out
