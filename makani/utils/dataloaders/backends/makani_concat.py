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

"""The makani layout concatenated into a single HDF5 file.

What ``data_process/concatenate_dataset.py`` produces: every year in one file,
usually as a virtual dataset over the per-year files. The array is identical to
the one the per-year layout uses, so only discovery differs -- there is one file
instead of many, and its timestamps have to be read rather than derived, since
the name no longer says which year it holds.

The one other difference is S3: a virtual dataset resolves its sources as local
paths, which no object store can satisfy, so reading this layout over S3 is
refused rather than left to half-work. Reading, opening, O_DIRECT and the
pickling discipline are inherited unchanged.
"""

import os
from typing import List

import h5py
import numpy as np

from ..data_helpers import get_lat_lon_grid
from .base import BackendMetadata, GridSpec, timestamp_converter
from .makani_hdf5 import MakaniHDF5Backend


class MakaniConcatBackend(MakaniHDF5Backend):
    """Reads the makani layout from one concatenated HDF5 file.

    Takes the same arguments as :class:`.MakaniHDF5Backend`, except that
    ``location`` is the file rather than a directory to search, and ``enable_s3``
    is rejected.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("enable_s3", False):
            # the file is usually a virtual dataset whose sources are resolved
            # as local paths, which no object store can satisfy
            raise NotImplementedError("Reading a concatenated dataset from S3 is not supported.")
        super().__init__(*args, **kwargs)

    def _find_files(self) -> List[str]:
        # location is the file itself here, not a directory to search
        paths = [path for path in self.location if os.path.isfile(path)]
        if not paths:
            locations = ", ".join(self.location)
            raise IOError(f"Error, the specified file path {locations} does not contain an h5 file.")
        return paths

    def discover(self, enable_logging: bool = False) -> BackendMetadata:
        files = self._find_files()
        timezone_fn = timestamp_converter(self.relative_timestamp)

        with h5py.File(files[0], "r") as handle:
            dset = handle[self.dataset_name]

            # a concatenated file has to carry its timestamps: unlike the per-year
            # layout there is no year in the name to derive them from
            if self.timestamp_name not in dset.dims[0]:
                raise ValueError(
                    f"{files[0]} has no '{self.timestamp_name}' dimension scale. A concatenated dataset "
                    "cannot have its sample times inferred, so it has to be annotated; see "
                    "data_process/annotate_dataset.py."
                )

            timestamps = timezone_fn(dset.dims[0][self.timestamp_name][...])
            coordinates = self._read_coordinates(dset)
            img_shape = dset.shape[2:4]
            total_channels = dset.shape[1]
            n_samples = dset.shape[0]

        self.files = [None]
        self.dsets = [None]

        # what the caller asked for, else what the file says, else the assumption
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
            # the label is only used for reporting and for deriving timestamps,
            # and this layout needs neither; the first year is the useful answer
            labels=[getattr(timestamps[0], "year", None)],
            samples_per_file=[n_samples],
            timestamps=timestamps,
            grid=grid,
            total_channels=total_channels,
        )
        return self.metadata
