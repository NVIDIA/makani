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

import logging
from typing import Optional, List, Tuple
import glob
import torch
import random
import numpy as np
import h5py
import torch
import math

# distributed stuff
from makani.utils import comm

# we need this
from physicsnemo.distributed.utils import compute_split_shapes

# for grid conversion
from makani.utils.grids import GridConverter


class DummyLoader(object):
    def __init__(self,
                 location: str,
                 batch_size: int,
                 dt: int,
                 dhours: int,
                 in_channels: List[int],
                 out_channels: List[int],
                 img_shape: Optional[Tuple[int,int]]=None,
                 max_samples: Optional[int]=None,
                 n_samples_per_epoch: Optional[int]=None,
                 n_history: Optional[int]=0,
                 n_future: Optional[int]=0,
                 add_zenith: Optional[bool]=False,
                 latitudes: Optional[np.array]=None,
                 longitudes: Optional[np.array]=None,
                 data_grid_type: Optional[str]="equiangular",
                 model_grid_type: Optional[str]="equiangular",
                 return_timestamp: Optional[bool]=False,
                 return_target: Optional[bool]=True,
                 dataset_path: Optional[str]="fields",
                 crop_size: Optional[Tuple[int, int]]=(None, None),
                 crop_anchor: Optional[Tuple[int, int]]=(0, 0),
                 io_grid: Optional[List[int]]=[1, 1, 1],
                 io_rank: Optional[List[int]]=[0, 0, 0],
                 device: Optional[torch.device]=torch.device("cpu"),
                 enable_logging: Optional[bool]=True,
                 **kwargs
    ):
        
        self.location = location
        self.dt = dt
        self.dhours = dhours
        self.max_samples = max_samples
        self.n_samples_per_epoch = n_samples_per_epoch
        self.batch_size = batch_size
        self.n_history = n_history
        self.n_future = n_future
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_in_channels = len(in_channels)
        self.n_out_channels = len(out_channels)
        self.img_shape = img_shape
        self.device = device
        self.return_timestamp = return_timestamp
        self.return_target = return_target
        self.io_grid = io_grid[1:]
        self.io_rank = io_rank[1:]
        if (latitudes is not None) and (longitudes is not None):
            self.lat_lon = (latitudes, longitudes)
        else:
            self.lat_lon = None

        # get cropping:
        self.crop_shape = crop_size
        self.crop_anchor = crop_anchor

        self._get_files_stats()

        # set lat_lon
        if self.lat_lon is None:
            resolution = 360.0 / float(self.img_shape[1])
            longitude = np.arange(0, 360, resolution)
            latitude = np.arange(-90, 90 + resolution, resolution)
            latitude = latitude[::-1]
            self.lat_lon = (latitude.tolist(), longitude.tolist())

        # zenith angle yes or no?
        self.add_zenith = add_zenith
        if self.add_zenith:
            self.zen_dummy = torch.zeros((self.batch_size, self.n_history + 1, 1, self.img_local_shape_x, self.img_local_shape_y), dtype=torch.float32, device=self.device)

        # grid types
        self.grid_converter = GridConverter(
            data_grid_type,
            model_grid_type,
            torch.deg2rad(torch.tensor(self.lat_lon[0][self.crop_anchor[0] : self.crop_anchor[0] + self.crop_shape[0]])).to(torch.float32),
            torch.deg2rad(torch.tensor(self.lat_lon[1][self.crop_anchor[1] : self.crop_anchor[1] + self.crop_shape[1]])).to(torch.float32),
        )

    def _get_files_stats(self):

        if self.img_shape is None:
            self.files_paths = glob.glob(self.location + "/*.h5")

            if not self.files_paths:
                raise RuntimeError(f"You have to specify img_shape if you do not provide a data path from which shapes can be deferred.")

            self.files_paths.sort()

            n_samples_total = 0
            for fname in self.files_paths:
                with h5py.File(fname, "r") as _f:
                    n_samples_total += _f["fields"].shape[0]
                    self.img_shape = _f["fields"].shape[2:4]

            if self.n_samples_per_epoch is None:
                self.n_samples_per_epoch = n_samples_total

            if self.max_samples is None:
                self.max_samples = n_samples_total

            # the user can regulate the number of samples using either variables
            self.n_samples_per_epoch = min(self.n_samples_per_epoch, self.max_samples)

        else:
            if (self.n_samples_per_epoch is not None) and (self.max_samples is None):
                self.max_samples = self.n_samples_per_epoch
            elif (self.n_samples_per_epoch is None) and (self.max_samples is not None):
                self.n_samples_per_epoch = self.max_samples

        # perform a sanity check here
        if (self.n_samples_per_epoch == 0) or (self.n_samples_per_epoch is None):
            raise RuntimeError(f"You have noit specified a valid number of samples per epoch.")

        # determine local read size:
        # sanitize the crops first
        if self.crop_shape[0] is None:
            self.crop_shape_x = self.img_shape[0]
        else:
            self.crop_shape_x = self.crop_shape[0]
        if self.crop_shape[1] is None:
            self.crop_shape_y = self.img_shape[1]
        else:
            self.crop_shape_y =	self.crop_shape[1]
        self.crop_shape = (self.crop_shape_x, self.crop_shape_y)
            
        assert self.crop_anchor[0] + self.crop_shape[0] <= self.img_shape[0]
        assert self.crop_anchor[1] + self.crop_shape[1] <= self.img_shape[1]

        # for x
        split_shapes_x = compute_split_shapes(self.crop_shape[0], self.io_grid[0])
        read_shape_x = split_shapes_x[self.io_rank[0]]
        read_anchor_x = sum(split_shapes_x[: self.io_rank[0]])

        # for y
        split_shapes_y = compute_split_shapes(self.crop_shape[1], self.io_grid[1])
        read_shape_y = split_shapes_y[self.io_rank[1]]
        read_anchor_y = sum(split_shapes_y[: self.io_rank[1]])

        # store exposed variables
        self.read_anchor = (read_anchor_x, read_anchor_y)
        self.read_shape = (read_shape_x, read_shape_y)

        # set properties for compatibility
        self.img_shape_x = self.img_shape[0]
        self.img_shape_y = self.img_shape[1]

        self.img_crop_shape_x = self.crop_shape[0]
        self.img_crop_shape_y = self.crop_shape[1]
        self.img_crop_offset_x = self.crop_anchor[0]
        self.img_crop_offset_y = self.crop_anchor[1]

        self.img_local_shape_x = self.read_shape[0]
        self.img_local_shape_y = self.read_shape[1]
        self.img_local_offset_x = self.read_anchor[0]
        self.img_local_offset_y = self.read_anchor[1]

        # sharding
        self.n_samples_total = self.n_samples_per_epoch
        self.n_samples_shard = self.n_samples_total // comm.get_size("data")

        # channels
        self.n_in_channels_local = self.n_in_channels
        self.n_out_channels_local = self.n_out_channels

        logging.info(f"Number of examples: {self.n_samples_per_epoch}. Image Shape: {self.img_shape[0]} x {self.img_shape[1]} x {self.n_in_channels_local}")
        logging.info(f"Including {self.dhours*self.dt*self.n_history} hours of past history in training at a frequency of {self.dhours*self.dt} hours")
        logging.info("WARNING: using dummy data")

        # create tensors for dummy data
        self.device = torch.device(f"cuda:{comm.get_local_rank()}")
        self.inp = torch.zeros((self.batch_size, self.n_history + 1, self.n_in_channels, self.read_shape[0], self.read_shape[1]), dtype=torch.float32, device=self.device)
        self.tar = torch.zeros(
            (self.batch_size, self.n_future + 1, self.n_out_channels_local, self.read_shape[0], self.read_shape[1]), dtype=torch.float32, device=self.device
        )

        # initialize output
        self.inp.uniform_()
        self.tar.uniform_()

        if self.return_timestamp:
            self.inp_time = torch.zeros((self.batch_size, self.n_history + 1), dtype=torch.float64)
            if self.return_target:
                self.tar_time = torch.ones((self.batch_size, self.n_future + 1), dtype=torch.float64)

        self.in_bias = np.zeros((1, self.n_in_channels, 1, 1)).astype(np.float32)
        self.in_scale = np.ones((1, self.n_in_channels, 1, 1)).astype(np.float32)
        self.out_bias = np.zeros((1, self.n_out_channels_local, 1, 1)).astype(np.float32)
        self.out_scale = np.ones((1, self.n_out_channels_local, 1, 1)).astype(np.float32)

    def get_input_normalization(self):
        return self.in_bias, self.in_scale

    def get_output_normalization(self):
        return self.out_bias, self.out_scale

    def __len__(self):
        return self.n_samples_shard

    def __iter__(self):
        self.sample_idx = 0
        return self

    def __next__(self):
        if self.sample_idx < self.n_samples_shard:
            self.sample_idx += 1

            result = (self.inp,)
            if self.return_target:
                result += (self.tar,)

            if self.add_zenith:
                result += (self.zen_dummy,)
                if self.return_target:
                    result += (self.zen_dummy,)

            if self.return_timestamp:
                result += (self.inp_time,)
                if self.return_target:
                    result += (self.tar_time,)

            return result
        else:
            raise StopIteration()
