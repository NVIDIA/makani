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

import os
import logging
from typing import Optional, List, Tuple, Union
import glob
from itertools import groupby, accumulate
import operator
from bisect import bisect_right
import datetime as dt

import torch
import numpy as np
from torch.utils.data import Dataset
import h5py

# for data normalization
from makani.utils.dataloaders.data_helpers import get_data_normalization, get_timestamp, get_date_from_timestamp, get_timedelta_from_timestamp, get_default_aws_connector

# for grid conversion
from makani.utils.grids import GridConverter

# import splitting logic
from physicsnemo.distributed.utils import compute_split_shapes


class MultifilesDataset(Dataset):
    def __init__(self,
                 location: Union[str, List[str]],
                 dt: int,
                 in_channels: List[int],
                 out_channels: List[int],
                 n_history: Optional[int]=0,
                 n_future: Optional[int]=0,
                 add_zenith: Optional[bool]=False,
                 data_grid_type: Optional[str]="equiangular",
                 model_grid_type: Optional[str]="equiangular",
                 bias: Optional[np.array]=None,
                 scale: Optional[np.array]=None,
                 return_timestamp: Optional[bool]=False,
                 relative_timestamp: Optional[bool]=False,
                 return_target: Optional[bool]=True,
                 file_suffix: Optional[str]="h5",
                 dataset_path: Optional[str]="fields",
                 enable_s3: Optional[bool]=False,
                 crop_size: Optional[Tuple[int, int]]=(None, None),
                 crop_anchor: Optional[Tuple[int, int]]=(0, 0),
                 io_grid: Optional[List[int]]=[1, 1, 1],
                 io_rank: Optional[List[int]]=[0, 0, 0],
                 enable_logging: Optional[bool]=True,
                 **kwargs):

        self.location = location
        self.dt = dt
        self.n_history = n_history
        self.n_future = n_future
        self.in_channels = np.array(in_channels)
        self.out_channels = np.array(out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.add_zenith = add_zenith
        self.return_timestamp = return_timestamp
        self.relative_timestamp = relative_timestamp
        self.return_target = return_target
        self.file_suffix = file_suffix
        self.dataset_path = dataset_path
        self.enable_s3 = enable_s3

        self.file_driver = None
        self.file_driver_kwargs = {}
        self.aws_connector = None
        if self.enable_s3:
            self.file_driver = "ros3"
            self.aws_connector = get_default_aws_connector(None)
            self.file_driver_kwargs = dict(
                aws_region=bytes(self.aws_connector.aws_region_name, "utf-8"),
                secret_id=bytes(self.aws_connector.aws_access_key_id, "utf-8"),
                secret_key=bytes(self.aws_connector.aws_secret_access_key, "utf-8"),
            )

        # also obtain an ordered in_channels list, required for h5py:
        self.in_channels_sorted = np.sort(self.in_channels)
        self.in_channels_unsort = np.argsort(np.argsort(self.in_channels))
        self.in_channels_is_sorted = np.all(self.in_channels_sorted == self.in_channels)

        # multifiles dataloader doesn't support channel parallelism yet
        # set the read slices
        assert io_grid[0] == 1
        self.io_grid = io_grid[1:]
        self.io_rank = io_rank[1:]

        # crop info
        self.crop_size = crop_size
        self.crop_anchor = crop_anchor

        # datetime logic
        if self.relative_timestamp:
            self.date_fn = np.vectorize(get_timedelta_from_timestamp)
        else:
            self.date_fn = np.vectorize(get_date_from_timestamp)

        # get more info
        self._get_files_stats(enable_logging)

        # for normalization load the statistics
        self.normalize = True

        if bias is not None:
            self.in_bias = bias[:, self.in_channels]
            self.out_bias = bias[:, self.out_channels]
        else:
            self.in_bias = np.zeros((1, len(self.in_channels), 1, 1))
            self.out_bias = np.zeros((1, len(self.out_channels), 1, 1))

        if scale is not None:
            self.in_scale = scale[:, self.in_channels]
            self.out_scale = scale[:, self.out_channels]
        else:
            self.in_scale = np.ones((1, len(self.in_channels), 1, 1))
            self.out_scale = np.ones((1, len(self.out_channels), 1, 1))

        # store local grid
        latitude = np.array(self.lat_lon[0])
        longitude = np.array(self.lat_lon[1])
        self.lon_grid, self.lat_grid = np.meshgrid(longitude, latitude)
        self.lat_grid_local = self.lat_grid[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0], self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]]
        self.lon_grid_local = self.lon_grid[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0], self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]]
        self.lat_lon_local = (
            latitude[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0]].tolist(),
            longitude[self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]].tolist(),
        )

        # grid types
        self.grid_converter = GridConverter(
            data_grid_type,
            model_grid_type,
            torch.deg2rad(torch.tensor(self.lat_lon_local[0])).to(torch.float32),
            torch.deg2rad(torch.tensor(self.lat_lon_local[1])).to(torch.float32),
        )

        # delete the aws connector since it cannot be pickled:
        del self.aws_connector
        self.aws_connector = None

    # HDF5 routines
    def _get_stats_h5(self, enable_logging):

        # ensure we do the right conversion
        fn_handle = get_timedelta_from_timestamp if self.relative_timestamp else get_date_from_timestamp

        self.n_samples_file = []
        self.date_ranges = []
        timestamps = []
        with h5py.File(self.files_paths[0], "r", driver=self.file_driver, **self.file_driver_kwargs) as _f:
            if enable_logging:
                logging.info("Getting file stats from {}".format(self.files_paths[0]))
            # original image shape (before padding)
            self.img_shape = _f[self.dataset_path].shape[2:4]
            self.total_channels = _f[self.dataset_path].shape[1]
            self.n_samples_file.append(_f[self.dataset_path].shape[0])
            lat = _f[self.dataset_path].dims[2]["lat"][...]
            lon = _f[self.dataset_path].dims[3]["lon"][...]
            self.lat_lon = (lat.tolist(), lon.tolist())
            tstamps = _f[self.dataset_path].dims[0]["timestamp"][...]
            self.date_ranges.append((fn_handle(tstamps[0]), fn_handle(tstamps[-1])))
            timestamps.append(tstamps)

        # get all remaining sample counts
        for filename in self.files_paths[1:]:
            with h5py.File(filename, "r", driver=self.file_driver, **self.file_driver_kwargs) as _f:
                self.n_samples_file.append(_f[self.dataset_path].shape[0])
                tstamps = _f[self.dataset_path].dims[0]["timestamp"][...]
                self.date_ranges.append((fn_handle(tstamps[0]), fn_handle(tstamps[-1])))
                timestamps.append(tstamps)

        # now, order the lists according to the date ranges:
        # we sort the lower and upper ends, and if the permutations match, then we can proceed
        lower_order = np.argsort([x[0] for x in self.date_ranges])
        upper_order = np.argsort([x[1] for x in self.date_ranges])
        if not np.all(lower_order == upper_order):
            raise RuntimeError("The files might have overlapping date ranges. Please make sure the individual files have disjoint ranges")
        lower_order = lower_order.tolist()

        # sort all files according to time stamps
        self.files_paths = [self.files_paths[idx] for idx in lower_order]
        self.n_samples_file = [self.n_samples_file[idx] for idx in lower_order]
        timestamps = [timestamps[idx] for idx in lower_order]
        self.timestamps = np.concatenate(timestamps, axis=0)
        self.datestamps = self.date_fn(self.timestamps)
        self.date_ranges = [self.date_ranges[idx] for idx in lower_order]

        # perform a sanity check: is the deltaT between entries consistent
        dhours_list = [int(d.total_seconds() // 3600.) for d in (self.datestamps[1:] - self.datestamps[:-1]).tolist()]

        if min(dhours_list) != max(dhours_list):
            raise RuntimeError("The time difference between steps is not constant, provide a dataset where this is the case")

        self.dhours = dhours_list[0]

        return

    def _get_files_stats(self, enable_logging):

        if isinstance(self.location, str):
            self.location = [self.location]

        # check if we specified a single file or a path with more than one file
        if not self.enable_s3:
            if os.path.isfile(self.location[0]):
                self.files_paths = self.location

            else:
                self.files_paths = []
                for location in self.location:
                    if not os.path.isdir(location):
                        raise IOError(f"Location {location} is neither a path nor a directory.")
                    self.files_paths = self.files_paths + glob.glob(os.path.join(location, f"*.{self.file_suffix}"))

        else:
            files_paths = self.aws_connector.list_bucket(self.location)

            for fpath in files_paths:
                if fpath.endswith(f".{self.file_suffix}"):
                    # prepend the endpoint
                    fpathp = self.aws_connector.aws_endpoint_url + "/" + fpath
                    self.files_paths.append(fpathp)

        if self.files_paths:
            self.file_format = "h5"
        else:
            raise IOError(f"Error, the specified file path {self.location} does not contain hdf5 files.")

        # get stats from files
        self.files_paths.sort()
        self._get_stats_h5(enable_logging)

        # extract the years from filenames
        if not self.relative_timestamp:
            self.years = sorted(list({date.year for date in self.datestamps.tolist()}))
            self.n_years = len(self.years)

        # create handles for datasets
        self.files = [None for x in self.files_paths]

	# store earliest and latest timestamp in dataset
        # we expect that all months are available per year
        self.start_date = self.datestamps[0]
        self.end_date =	self.datestamps[-1]

        # determine local read size:
        # sanitize the crops first
        crop_size_x, crop_size_y = self.crop_size
        if crop_size_x is None:
            crop_size_x = self.img_shape[0]
        if crop_size_y is None:
            crop_size_y = self.img_shape[1]
        self.crop_size = (crop_size_x, crop_size_y)
        assert self.crop_anchor[0] + self.crop_size[0] <= self.img_shape[0]
        assert self.crop_anchor[1] + self.crop_size[1] <= self.img_shape[1]

        # for x
        split_shapes_x = compute_split_shapes(self.crop_size[0], self.io_grid[0])
        read_shape_x = split_shapes_x[self.io_rank[0]]
        read_anchor_x = self.crop_anchor[0] + sum(split_shapes_x[: self.io_rank[0]])

        # for y
        split_shapes_y = compute_split_shapes(self.crop_size[1], self.io_grid[1])
        read_shape_y = split_shapes_y[self.io_rank[1]]
        read_anchor_y = self.crop_anchor[1] + sum(split_shapes_y[: self.io_rank[1]])

        # store the variables
        self.read_anchor = (read_anchor_x, read_anchor_y)
        self.read_shape = (read_shape_x, read_shape_y)

        # do some sample indexing gymnastics
        self.file_offsets = list(accumulate(self.n_samples_file, operator.add))[:-1]
        self.file_offsets.insert(0, 0)
        self.n_samples_available = sum(self.n_samples_file)
        self.n_samples_total = self.n_samples_available

        if enable_logging:
            logging.info("Average number of samples per file: {:.1f}".format(float(self.n_samples_total) / float(len(self.files))))
            logging.info(
                "Found data at path {}. Number of examples: {}. Full image Shape: {} x {} x {}. Read Shape: {} x {} x {}".format(
                    self.location, self.n_samples_available, self.img_shape[0], self.img_shape[1], self.total_channels, self.read_shape[0], self.read_shape[1], self.n_in_channels
                )
            )
            logging.info(f"Dataset covers a timespan from {self.start_date} to {self.end_date} with a resolution of {self.dhours} hour(s).")
            logging.info(f"Using a step size of {self.dhours*self.dt} hour(s) for inference.")
            logging.info("Including {} hours of past history in training at a frequency of {} hours".format(self.dhours * self.dt * (self.n_history + 1), self.dhours * self.dt))
            logging.info("Including {} hours of future targets in training at a frequency of {} hours".format(self.dhours * self.dt * (self.n_future + 1), self.dhours * self.dt))

        # set properties for compatibility
        self.img_shape_x = self.img_shape[0]
        self.img_shape_y = self.img_shape[1]

        self.img_crop_shape_x = self.crop_size[0]
        self.img_crop_shape_y = self.crop_size[1]
        self.img_crop_offset_x = self.crop_anchor[0]
        self.img_crop_offset_y = self.crop_anchor[1]

        self.img_local_shape_x = self.read_shape[0]
        self.img_local_shape_y = self.read_shape[1]
        self.img_local_offset_x = self.read_anchor[0]
        self.img_local_offset_y = self.read_anchor[1]

    def _compute_timestamps(self, global_idx, offset_start, offset_end):
        times = self.timestamps[global_idx+offset_start:global_idx+offset_end]

        return times

    def _compute_zenith_angle(self, times):
        # import
        from makani.third_party.climt.zenith_angle import cos_zenith_angle

        # convert to datetimes:
        times_dt = self.date_fn(times)

        # compute the corresponding zenith angles
        cos_zenith = np.expand_dims(cos_zenith_angle(times_dt, self.lon_grid_local, self.lat_grid_local).astype(np.float32), axis=1)

        return cos_zenith

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], "r", driver=self.file_driver, **self.file_driver_kwargs)
        self.files[file_idx] = _file[self.dataset_path]

    def _get_indices(self, global_idx):
        file_idx = bisect_right(self.file_offsets, global_idx) - 1
        local_idx = global_idx - self.file_offsets[file_idx]
        return file_idx, local_idx

    def _get_data(self, global_idx, offset_start, offset_end, target=False):
        # load slice of data:
        start_x = self.read_anchor[0]
        end_x = start_x + self.read_shape[0]

        start_y = self.read_anchor[1]
        end_y = start_y + self.read_shape[1]

        data_list = []
        for offset_idx in range(offset_start, offset_end):
            file_idx, local_idx = self._get_indices(global_idx + self.dt * offset_idx)

            # open image file
            if self.files[file_idx] is None:
                self._open_file(file_idx)

            data = self.files[file_idx][local_idx : local_idx + 1, self.in_channels_sorted, start_x:end_x, start_y:end_y]

            if not self.in_channels_is_sorted:
                data = data[:, self.in_channels_unsort, :, :]
            
            data_list.append(data)

        data = np.concatenate(data_list, axis=0)
        if self.normalize:
            if target:
                data = (data - self.out_bias) / self.out_scale
            else:
                data = (data - self.in_bias) / self.in_scale

        return data

    def __len__(self):
        toff = 1 if self.return_target else 0
        return self.n_samples_total - self.dt * (self.n_history + self.n_future + toff)

    def get_sample_at_index(self, global_idx, return_target=True):

        # load the input
        inp = self._get_data(global_idx, 0, self.n_history + 1, target=False)

        # load the target
        if return_target:
            tar = self._get_data(global_idx, self.n_history + 1, self.n_history + self.n_future + 2, target=True)

        # compute time stamps
        if self.add_zenith or self.return_timestamp:
            inp_time = self._compute_timestamps(global_idx, 0, self.n_history + 1)

            if return_target:
                tar_time = self._compute_timestamps(global_idx, self.n_history + 1, self.n_history + self.n_future + 2)

        # construct result tuple
        result = (inp,)
        if return_target:
            result += (tar,)

        if self.add_zenith:
            zen_inp = self._compute_zenith_angle(inp_time)
            result += (zen_inp,)
            if return_target:
                zen_tar = self._compute_zenith_angle(tar_time)
                result += (zen_tar,)

        # convert to tensor and convert grid
        result = tuple(torch.as_tensor(arr, dtype=torch.float32) for arr in result)
        result = tuple(map(lambda x: self.grid_converter(x), result))

        # append timestamp if requested
        if self.return_timestamp:
            result += (torch.as_tensor(inp_time, dtype=torch.float64),)
            if return_target:
                result += (torch.as_tensor(tar_time, dtype=torch.float64),)

        return result

    # this is just for the torch dataloader
    def __getitem__(self, global_idx):

        result = self.get_sample_at_index(global_idx, return_target=self.return_target)

        return result

    def get_index_at_time(self, tstamp):
        # return the sample which is equal or smaller than timestamp:
        if self.relative_timestamp:
            if not isinstance(tstamp, dt.timedelta):
                tstamp = get_timedelta_from_timestamp(tstamp)
        else:
            if not isinstance(tstamp, dt.datetime):
                tstamp = get_date_from_timestamp(tstamp)

        if (tstamp < self.start_date) or (tstamp > self.end_date):
            return None

        # this returns the position in the sorted list. We need to find it in the original list then
        gidx = bisect_right(self.datestamps, tstamp) - 1

        return gidx

    def get_time_at_index(self, global_idx):
        return self.datestamps[global_idx]

    def get_sample_at_time(self, timestamp):
        global_idx = self.get_index_at_time(timestamp)
        if global_idx is None:
            raise IndexError(f"Time stamp {timestamp} is out of range of the dataset.")
        return self.get_sample_at_index(global_idx, return_target=self.return_target)

    def get_output_normalization(self):
        return self.out_bias, self.out_scale

    def get_input_normalization(self):
        return self.in_bias, self.in_scale
