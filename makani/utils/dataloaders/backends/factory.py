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

"""Choosing a backend for a location.

Detection is by inspection, the way the loaders have always done it: a path
that is a file is a concatenated dataset, a directory holding ``????.h5`` is the
per-year HDF5 layout, one holding ``????.zarr`` is zarr -- and a zarr store that
has no ``fields`` array is a WeatherBench2 store, whose variables are named
individually.

``file_pattern`` is the name to glob for, without the extension. It defaults to
the makani convention of a file per year; a dataset of arbitrarily named files
passes ``"*"``.

A run that knows what it has can name it outright and skip the guessing.
"""

import glob
import os
from typing import Optional

from .arco_wb2 import ArcoWB2Backend
from .base import DatasetBackend
from .makani_concat import MakaniConcatBackend
from .makani_hdf5 import MakaniHDF5Backend
from .makani_zarr import MakaniZarrBackend, zarr_open

BACKENDS = {
    "makani_hdf5": MakaniHDF5Backend,
    "makani_zarr": MakaniZarrBackend,
    "makani_concat": MakaniConcatBackend,
    "arco_wb2": ArcoWB2Backend,
}


def detect_backend(location, dataset_name: str = "fields", enable_s3: bool = False, file_pattern: str = "????") -> str:
    """Name the backend that fits what is at ``location``."""
    locations = location if isinstance(location, list) else [location]

    if any(os.path.isfile(path) for path in locations):
        # a local file is a concatenated dataset whether or not S3 is on; the
        # concat backend is the one that gets to say S3 is not supported for it
        return "makani_concat"

    if enable_s3:
        # object storage holds the per-year HDF5 layout; there is nothing local
        # to stat, so this is the only layout reachable that way
        return "makani_hdf5"

    for path in locations:
        if glob.glob(os.path.join(path, f"{file_pattern}.h5")):
            return "makani_hdf5"

    for path in locations:
        stores = sorted(glob.glob(os.path.join(path, f"{file_pattern}.zarr")))
        if stores:
            # the layouts share a container, so the store itself has to be asked
            return "makani_zarr" if dataset_name in zarr_open(stores[0]) else "arco_wb2"

    locations = ", ".join(locations)
    raise IOError(f"Error, the specified file path(s) {locations} contain neither h5 nor zarr datasets.")


def get_backend(location, backend: Optional[str] = None, **kwargs) -> DatasetBackend:
    """Build the backend for ``location``, detecting the layout unless told.

    Parameters
    ----------
    location : str or list of str
        Where the dataset lives.
    backend : str, optional
        Name from :data:`BACKENDS`, to bypass detection.
    **kwargs
        Passed to the backend, see :class:`.base.DatasetBackend`.
    """
    if backend is None:
        backend = detect_backend(
            location,
            dataset_name=kwargs.get("dataset_name", "fields"),
            enable_s3=kwargs.get("enable_s3", False),
            file_pattern=kwargs.get("file_pattern", "????"),
        )

    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Known backends: {sorted(BACKENDS)}.")

    backend_type = BACKENDS[backend]

    # only the HDF5 layouts know what to do with these
    if backend_type not in (MakaniHDF5Backend, MakaniConcatBackend):
        for name in ("enable_odirect", "odirect_alignment", "enable_s3"):
            kwargs.pop(name, None)

    return backend_type(location, **kwargs)
