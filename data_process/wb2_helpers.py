# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
import mpi4py
from mpi4py.util import dtlib
import progressbar
import numpy as np

from makani.utils.features import get_channel_groups

# variable translation
surface_variables = {
    "u10m" : "10m_u_component_of_wind",
    "v10m" : "10m_v_component_of_wind",
    "t2m" : "2m_temperature",
    "d2": "2m_dewpoint_temperature",
    "u100m" : "100m_u_component_of_wind",
    "v100m" : "100m_v_component_of_wind",
    "tp" : "total_precipitation_6hr",
    "sp" : "surface_pressure",
    "msl" : "mean_sea_level_pressure",
    "tcwv": "total_column_water_vapour",
    "sst": "sea_surface_temperature",
}

atmospheric_variables = {
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "r": "relative_humidity",
    "q": "specific_humidity",
}

def split_convert_channel_names(makani_channel_names):
    
    # split in surface and atmospheric channels
    atmospheric_channel_indices, surface_channel_indices, _, atmospheric_levels = get_channel_groups(makani_channel_names)

    # atmo
    atmospheric_channel_names = [makani_channel_names[k] for k in atmospheric_channel_indices]
    pat = re.compile(r"^(.*?)\d{1,}$")
    atmospheric_channel_names = sorted(list(set([pat.match(c).groups()[0] for c in atmospheric_channel_names])))
    atmospheric_channel_names_wb2 = [atmospheric_variables[c] for c in atmospheric_channel_names]
    
    # surface
    surface_channel_names = sorted([makani_channel_names[k] for k in surface_channel_indices])
    surface_channel_names_wb2 = [surface_variables[c] for c in surface_channel_names]
    
    # levels
    atmospheric_levels = sorted(list(atmospheric_levels))

    return atmospheric_channel_names, atmospheric_channel_names_wb2, surface_channel_names, surface_channel_names_wb2, atmospheric_levels


class DistributedProgressBar(object):

    def __init__(self, num_entries: int, comm: mpi4py.MPI.Comm):
        # store comm
        self.comm = comm

        # set up progressbar
        datatype = mpi4py.MPI.INT64_T
        np_dtype = dtlib.to_numpy_dtype(datatype)
        itemsize = datatype.Get_size()
        self.win = mpi4py.MPI.Win.Allocate(itemsize, comm=comm)
        self.counts = np.zeros([1], dtype=np_dtype)
        self.comm.Barrier()

        if self.comm.Get_rank() == 0:
            # set up pbar
            self.pbar = progressbar.ProgressBar(maxval=num_entries)
        self.reset()

    def __del__(self):
        self.comm.Barrier()
        self.win.Free()
        if self.comm.Get_rank() == 0:
            self.pbar.finish()

    def reset(self):
        if self.comm.Get_rank() == 0:
            self.pbar.update(0)

    def update_counter(self, count: int):
        self.counts[0] = count
        self.win.Lock(rank=0)
        self.win.Accumulate(self.counts, target_rank=0)
        self.win.Flush_local(rank=0)
        self.win.Unlock(rank=0)
        return

    def get_counter(self)-> int:
        self.win.Lock(rank=0, lock_type=mpi4py.MPI.LOCK_SHARED)
        self.win.Get(self.counts, target_rank=0)
        self.win.Flush(rank=0)
        self.win.Unlock(rank=0)
        return int(self.counts[0])

    def update_progress(self):
        if self.comm.Get_rank() == 0:
            count = self.get_counter()
            self.pbar.update(count)
