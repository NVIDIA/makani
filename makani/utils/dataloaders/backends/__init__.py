# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Storage backends for the sample sources.

Each backend knows how to find, open and read one dataset layout. They are
consumed by :class:`makani.utils.dataloaders.sample_source.SampleSource`, which
owns everything the layouts have in common: index arithmetic, shuffling, epoch
cycling, window assembly and spatial sharding.
"""

from .base import DatasetBackend, BackendMetadata
from .makani_hdf5 import MakaniHDF5Backend
from .makani_zarr import MakaniZarrBackend
from .arco_wb2 import ArcoWB2Backend
from .makani_concat import MakaniConcatBackend
from .factory import get_backend

__all__ = [
    "DatasetBackend",
    "BackendMetadata",
    "MakaniHDF5Backend",
    "MakaniZarrBackend",
    "ArcoWB2Backend",
    "MakaniConcatBackend",
    "get_backend",
]
