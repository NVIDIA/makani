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

import time
import sys
import os
import glob
from functools import partial
import numpy as np
import h5py
import zarr
import logging
from itertools import groupby, accumulate
import operator
from bisect import bisect_right

# for nvtx annotation
import torch

# import splitting logic
from physicsnemo.distributed.utils import compute_split_shapes

# data helpers
from .data_helpers import get_timestamp, get_date_from_timestamp, get_default_aws_connector


class GeneralES(object):
    def _get_slices(self, lst):
        for a, b in groupby(enumerate(lst), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield slice(b[0][1], b[-1][1] + 1)

    # very important: the seed has to be constant across the workers, or otherwise mayhem:
    def __init__(
        self,
        location,
        max_samples,
        samples_per_epoch,
        train,
        batch_size,
        dt,
        dhours,
        n_history,
        n_future,
        in_channels,
        out_channels,
        crop_size,
        crop_anchor,
        num_shards,
        shard_id,
        io_grid,
        io_rank,
        device_id=0,
        truncate_old=True,
        enable_logging=True,
        zenith_angle=True,
        return_timestamp=False,
        lat_lon=None,
        dataset_path="fields",
        enable_odirect=False,
        enable_s3=False,
        seed=333,
        is_parallel=True,
    ):
        self.batch_size = batch_size
        self.location = location
        self.max_samples = max_samples
        self.n_samples_per_epoch = samples_per_epoch
        self.truncate_old = truncate_old
        self.train = train
        self.dt = dt
        self.dhours = dhours
        self.n_history = n_history
        self.n_future = n_future
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_in_channels = len(in_channels)
        self.n_out_channels = len(out_channels)
        self.crop_size = crop_size
        self.crop_anchor = crop_anchor
        self.base_seed = seed
        self.num_shards = num_shards
        self.device_id = device_id
        self.shard_id = shard_id
        self.is_parallel = is_parallel
        self.zenith_angle = zenith_angle
        self.return_timestamp = return_timestamp
        self.dataset_path = dataset_path
        self.lat_lon = lat_lon

        # also obtain an ordered channels list, required for h5py:
        # in_channels
        self.in_channels_sorted = np.sort(self.in_channels)
        self.in_channels_unsort = np.argsort(np.argsort(self.in_channels))
        self.in_channels_is_sorted = np.all(self.in_channels_sorted == self.in_channels)
        # out_channels
        self.out_channels_sorted = np.sort(self.out_channels)
        self.out_channels_unsort = np.argsort(np.argsort(self.out_channels))
        self.out_channels_is_sorted = np.all(self.out_channels_sorted == self.out_channels)

        # sanity checks
        if enable_odirect and enable_s3:
            raise NotImplementedError("The setting enable_odirect and enable_s3 are mutually exclusive.")

        # O_DIRECT and S3 specific stuff
        self.enable_s3 = enable_s3
        self.file_driver = None
        self.file_driver_kwargs = {}
        self.aws_connector = None
        if enable_odirect:
            self.file_driver = "direct"

        if enable_s3:
            self.file_driver = "ros3"
            self.aws_connector = get_default_aws_connector(None)
            self.file_driver_kwargs = dict(
                aws_region=bytes(self.aws_connector.aws_region_name, "utf-8"),
                secret_id=bytes(self.aws_connector.aws_access_key_id, "utf-8"),
                secret_key=bytes(self.aws_connector.aws_secret_access_key, "utf-8"),
            )

        self.read_direct = True if not self.enable_s3 else False
        self.num_retries = 5

        # set the read slices
        # we do not support channel parallelism yet
        assert io_grid[0] == 1
        self.io_grid = io_grid[1:]
        self.io_rank = io_rank[1:]

        # parse the files
        self._get_files_stats(enable_logging)

        # set shuffling to true or false
        self.shuffle = True if train else False

        # convert in_channels to list of slices:
        self.in_channels_slices = list(self._get_slices(self.in_channels_sorted))
        self.out_channels_slices = list(self._get_slices(self.out_channels_sorted))

        # we need some additional static fields in this case
        if self.lat_lon is None:
            latitude = np.linspace(90, -90, self.img_shape[0], endpoint=True)
            longitude = np.linspace(0, 360, self.img_shape[1], endpoint=False)
            self.lat_lon = (latitude.tolist(), longitude.tolist())

        # compute local grid
        latitude = np.array(self.lat_lon[0])
        longitude = np.array(self.lat_lon[1])
        self.lon_grid, self.lat_grid = np.meshgrid(longitude, latitude)
        self.lat_grid_local = self.lat_grid[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0], self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]]
        self.lon_grid_local = self.lon_grid[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0], self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]]
        self.lat_lon_local = (
            latitude[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0]].tolist(),
            longitude[self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]].tolist(),
        )

        # datetime logic
        self.date_fn = np.vectorize(get_date_from_timestamp)

        return

    def _reorder_channels(self, inp, tar):
        # reorder data if requested:
	# inp
        if not self.in_channels_is_sorted:
            inp_re = inp[:, self.in_channels_unsort, ...].copy()
        else:
            inp_re = inp.copy()

        # tar
        if not self.out_channels_is_sorted:
            tar_re = tar[:, self.out_channels_unsort, ...].copy()
        else:
            tar_re = tar.copy()

        return inp_re, tar_re

    # HDF5 routines
    def _get_stats_h5(self, enable_logging):

        if not self.enable_s3:
            fopen_handle = partial(h5py.File, mode="r")
        else:
            fopen_handle = partial(h5py.File, mode="r", driver=self.file_driver, **self.file_driver_kwargs)

        self.n_samples_year = []
        with fopen_handle(self.files_paths[0]) as _f:
            if enable_logging:
                logging.info("Getting file stats from {}".format(self.files_paths[0]))
            # original image shape (before padding)
            self.img_shape = _f[self.dataset_path].shape[2:4]
            self.total_channels = _f[self.dataset_path].shape[1]
            self.n_samples_year.append(_f[self.dataset_path].shape[0])

        # get all sample counts
        for filename in self.files_paths[1:]:
            with fopen_handle(filename) as _f:
                self.n_samples_year.append(_f[self.dataset_path].shape[0])

        return

    def _get_year_h5(self, year_idx):
        # here we want to use the specific file driver
        self.files[year_idx] = h5py.File(self.files_paths[year_idx], "r", driver=self.file_driver, **self.file_driver_kwargs)
        self.dsets[year_idx] = self.files[year_idx][self.dataset_path]
        return

    def _get_data_h5(self, dset, local_idx, start_x, end_x, start_y, end_y):

        # input
        off = 0
        for slice_in in self.in_channels_slices:
            start = off
            end = start + (slice_in.stop - slice_in.start)

            # read the data
            if self.read_direct:
                dset.read_direct(
                    self.inp_buff, np.s_[(local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt, slice_in, start_x:end_x, start_y:end_y], np.s_[:, start:end, ...])
            else:
                self.inp_buff[:, start:end, ...] = dset[(local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt, slice_in, start_x:end_x, start_y:end_y]

            # update offset
            off = end

        # target
        off = 0
        for slice_out in self.out_channels_slices:
            start = off
            end = start + (slice_out.stop - slice_out.start)

            # read the data
            if self.read_direct:
                dset.read_direct(
                    self.tar_buff, np.s_[(local_idx + self.dt) : (local_idx + self.dt * (self.n_future + 1) + 1) : self.dt, slice_out, start_x:end_x, start_y:end_y], np.s_[:, start:end, ...]
                )
            else:
                 self.tar_buff[:, start:end, ...] = dset[(local_idx + self.dt) : (local_idx + self.dt * (self.n_future + 1) + 1) : self.dt, slice_out, start_x:end_x, start_y:end_y]

            # update offset
            off = end

        # reorder data if requested:
        inp, tar = self._reorder_channels(self.inp_buff, self.tar_buff)

        return inp, tar

    # zarr functions
    def _get_stats_zarr(self, enable_logging):
        with zarr.convenience.open(self.files_paths[0], "r") as _f:
            if enable_logging:
                logging.info("Getting file stats from {}".format(self.files_paths[0]))
            # original image shape (before padding)
            self.img_shape = _f[f"/{self.dataset_path}"].shape[2:4]
            self.total_channels = _f[f"/{self.dataset_path}"].shape[1]

        self.n_samples_year = []
        for filename in self.files_paths:
            with zarr.convenience.open(filename, "r") as _f:
                self.n_samples_year.append(_f[f"/{self.dataset_path}"].shape[0])

        return

    def _get_year_zarr(self, year_idx):
        self.files[year_idx] = zarr.convenience.open(self.files_paths[year_idx], "r")
        self.dsets[year_idx] = self.files[year_idx][f"/{self.dataset_path}"]
        return

    def _get_data_zarr(self, dset, local_idx, start_x, end_x, start_y, end_y):
        off = 0
        for slice_in in self.in_channels_slices:
            start = off
            end = start + (slice_in.stop - slice_in.start)
            self.inp_buff[:, start:end, ...] = dset[(local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt, slice_in, start_x:end_x, start_y:end_y]
            off = end

        off = 0
        for slice_out in self.out_channels_slices:
            start = off
            end = start + (slice_out.stop - slice_out.start)
            self.tar_buff[:, start:end, ...] = dset[(local_idx + self.dt) : (local_idx + self.dt * (self.n_future + 1) + 1) : self.dt, slice_out, start_x:end_x, start_y:end_y]
            off = end

        # reorder data if requested:
        inp, tar = self._reorder_channels(self.inp_buff, self.tar_buff)

        return inp, tar

    def _get_files_stats(self, enable_logging):
        # check for hdf5 files
        self.files_paths = []
        self.location = [self.location] if not isinstance(self.location, list) else self.location

        if not self.enable_s3:
            # check if we are dealing with hdf5 or zarr files
            for location in self.location:
                self.files_paths = self.files_paths + glob.glob(os.path.join(location, "????.h5"))

            # check for zarr files if no hdf5 files are found
            if self.files_paths:
                self.file_format = "h5"
            else:
                for location in self.location:
                    self.files_paths = self.files_paths + glob.glob(os.path.join(location, "????.zarr"))
                if self.files_paths:
                    self.file_format = "zarr"
        else:
            files_paths = self.aws_connector.list_bucket(self.location)

            for fpath in files_paths:
                if fpath.endswith(".h5"):
                    # prepend the endpoint
                    fpathp = self.aws_connector.aws_endpoint_url + "/" + fpath
                    self.files_paths.append(fpathp)

            if self.files_paths:
                self.file_format = "h5"

        # check if some files have been found
        if not self.files_paths:
            locstring = ", ".join(self.location)
            raise IOError(f"Error, the specified file path(s) {locstring} do neither container h5 nor zarr files.")

        # sort the files
        self.files_paths.sort()

        # extract the years from filenames
        self.years = [int(os.path.splitext(os.path.basename(x))[0]) for x in self.files_paths]

        # get stats
        self.n_years = len(self.files_paths)

        # get stats from first file
        if self.file_format == "h5":
            self._get_stats_h5(enable_logging)
        else:
            self._get_stats_zarr(enable_logging)

        # determine local read size:
        # sanitize the crops first
        if self.crop_size[0] is None:
            self.crop_size[0] = self.img_shape[0]
        if self.crop_size[1] is None:
            self.crop_size[1] = self.img_shape[1]
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
        self.read_anchor = [read_anchor_x, read_anchor_y]
        self.read_shape = [read_shape_x, read_shape_y]

        # do some sample indexing gymnastics
        self.year_offsets = list(accumulate(self.n_samples_year, operator.add))[:-1]
        self.year_offsets.insert(0, 0)
        self.n_samples_available = sum(self.n_samples_year)
        if self.max_samples is not None:
            self.n_samples_total = min(self.n_samples_available, self.max_samples)
        else:
            self.n_samples_total = self.n_samples_available

        # do the sharding
        self.n_samples_shard = self.n_samples_total // self.num_shards
        if self.truncate_old:
            self.n_samples_offset = max(self.dt * self.n_history, self.n_samples_available - self.n_samples_total - self.dt * (self.n_future + 1) - 1)
        else:
            self.n_samples_offset = self.dt * self.n_history

        # number of steps per epoch
        self.num_steps_per_cycle = self.n_samples_shard // self.batch_size
        if self.n_samples_per_epoch is None:
            self.n_samples_per_epoch = self.n_samples_total
        self.num_steps_per_epoch = self.n_samples_per_epoch // (self.batch_size * self.num_shards)

        # we need those here
        self.num_samples_per_cycle_shard = self.num_steps_per_cycle * self.batch_size
        self.num_samples_per_epoch_shard = self.num_steps_per_epoch * self.batch_size
        # prepare file lists
        self.files = [None for _ in range(self.n_years)]
        self.dsets = [None for _ in range(self.n_years)]
        if enable_logging:
            logging.info("Average number of samples per year: {:.1f}".format(float(self.n_samples_total) / float(self.n_years)))
            logging.info(
                "Found data at path {}. Number of examples: {}. Full image Shape: {} x {} x {}. Read Shape: {} x {} x {}".format(
                    self.location, self.n_samples_available, self.img_shape[0], self.img_shape[1], self.total_channels, self.read_shape[0], self.read_shape[1], self.n_in_channels
                )
            )
            logging.info(
                "Using {} from the total number of available samples with {} samples per epoch (corresponds to {} steps for {} shards with local batch size {})".format(
                    self.n_samples_total, self.n_samples_per_epoch, self.num_steps_per_epoch, self.num_shards, self.batch_size
                )
            )
            start_lidx, start_yidx = self._get_local_year_index_from_global_index(self.n_samples_offset)
            end_lidx, end_yidx = self._get_local_year_index_from_global_index(self.n_samples_available-1)
            start_date = get_timestamp(self.years[start_yidx], hour=(start_lidx * self.dhours))
            end_date = get_timestamp(self.years[end_yidx], hour=(end_lidx * self.dhours))
            logging.info(f"Date range for data set: {start_date} to {end_date}.")
            logging.info("Delta t: {} hours".format(self.dhours * self.dt))
            logging.info("Including {} hours of past history in training at a frequency of {} hours".format(self.dhours * self.dt * (self.n_history + 1), self.dhours * self.dt))
            logging.info("Including {} hours of future targets in training at a frequency of {} hours".format(self.dhours * self.dt * (self.n_future + 1), self.dhours * self.dt))

        # some state variables
        self.last_cycle_epoch = None
        self.index_permutation = None

        # prepare buffers for double buffering
        if not self.is_parallel:
            self._init_buffers()

    def _init_buffers(self):
        self.inp_buff = np.zeros((self.n_history + 1, self.n_in_channels, self.read_shape[0], self.read_shape[1]), dtype=np.float32)
        self.tar_buff = np.zeros((self.n_future + 1, self.n_out_channels, self.read_shape[0], self.read_shape[1]), dtype=np.float32)

    def _compute_timestamps(self, local_idx, year_idx):
        # compute hours into the year
        year = self.years[year_idx]

        inp_time = np.asarray([get_timestamp(year, hour=(idx * self.dhours)).timestamp() for idx in range(local_idx - self.dt * self.n_history, local_idx + 1, self.dt)])

        tar_time = np.asarray([get_timestamp(year, hour=(idx * self.dhours)).timestamp() for idx in range(local_idx + self.dt, local_idx + self.dt * (self.n_future + 1) + 1, self.dt)])

        return inp_time, tar_time

    def _compute_zenith_angle(self, inp_times, tar_times):
        # nvtx range
        torch.cuda.nvtx.range_push("GeneralES:_compute_zenith_angle")

        # import
        from makani.third_party.climt.zenith_angle import cos_zenith_angle

        # convert to datetimes:
        inp_times_dt = self.date_fn(inp_times)
        tar_times_dt = self.date_fn(tar_times)

        # zenith angle for input
        cos_zenith_inp = np.expand_dims(cos_zenith_angle(inp_times_dt, self.lon_grid_local, self.lat_grid_local).astype(np.float32), axis=1)

        # zenith angle for target:
        cos_zenith_tar = np.expand_dims(cos_zenith_angle(tar_times_dt, self.lon_grid_local, self.lat_grid_local).astype(np.float32), axis=1)

        # nvtx range
        torch.cuda.nvtx.range_pop()

        return cos_zenith_inp, cos_zenith_tar 

    def __getstate__(self):
        del self.aws_connector
        self.aws_connector = None
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.file_format == "h5":
            self.get_year_handle = self._get_year_h5
            self.get_data_handle = self._get_data_h5
        else:
            self.get_year_handle = self._get_year_zarr
            self.get_data_handle = self._get_data_zarr

        # re-create members which are needed
        if self.enable_s3 and (self.aws_connector is None):
            self.aws_connector = get_default_aws_connector(None)

        if self.is_parallel:
            self._init_buffers()

    def __len__(self):
        return self.n_samples_shard

    def __del__(self):
        # close files
        if hasattr(self, "files"):
            for f in self.files:
                if f is not None:
                    f.close()

    def _get_local_year_index_from_global_index(self, sample_idx):
        year_idx = bisect_right(self.year_offsets, sample_idx) - 1  # subtract 1 because we do 0-based indexing
        local_idx = sample_idx - self.year_offsets[year_idx]
        return local_idx, year_idx

    def __call__(self, sample_info):
        # compute global iteration index:
        global_sample_idx = sample_info.idx_in_epoch + sample_info.epoch_idx * self.num_samples_per_epoch_shard
        cycle_sample_idx = global_sample_idx % self.num_samples_per_cycle_shard
        cycle_epoch_idx = global_sample_idx // self.num_samples_per_cycle_shard

        # check if epoch is done
        if sample_info.iteration >= self.num_steps_per_epoch:
            raise StopIteration

        torch.cuda.nvtx.range_push("GeneralES:__call__")

        if cycle_epoch_idx != self.last_cycle_epoch:
            self.last_cycle_epoch = cycle_epoch_idx
            # generate a unique seed and permutation:
            rng = np.random.default_rng(seed=self.base_seed + cycle_epoch_idx)

            # shufle if requested
            if self.shuffle:
                self.index_permutation = self.n_samples_offset + rng.permutation(self.n_samples_total)
            else:
                self.index_permutation = self.n_samples_offset + np.arange(self.n_samples_total)

            # shard the data
            start = self.n_samples_shard * self.shard_id
            end = start + self.n_samples_shard
            self.index_permutation = self.index_permutation[start:end]

        # determine local and sample idx
        sample_idx = self.index_permutation[cycle_sample_idx]
        local_idx, year_idx = self._get_local_year_index_from_global_index(sample_idx)

        # if we are not at least self.dt*n_history timesteps into the prediction
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        if local_idx >= (self.n_samples_year[year_idx] - self.dt * (self.n_future + 1)):
            local_idx = self.n_samples_year[year_idx] - self.dt * (self.n_future + 1) - 1

        if self.files[year_idx] is None:

            for _ in range(self.num_retries):
                try:
                    self.get_year_handle(year_idx)
                    break
                except:
                    print(f"Cannot get year handle {year_idx}, retrying.", flush=True)
                    time.sleep(5)
            else:
                raise OSError(f"Unable to retrieve year handle {year_idx}, aborting.")

        # do the read
        dset = self.dsets[year_idx]

        # load slice of data:
        start_x = self.read_anchor[0]
        end_x = start_x + self.read_shape[0]

        start_y = self.read_anchor[1]
        end_y = start_y + self.read_shape[1]

        # read data
        inp, tar = self.get_data_handle(dset, local_idx, start_x, end_x, start_y, end_y)

        # compute time stamps
        if self.zenith_angle or self.return_timestamp:
            inp_time, tar_time = self._compute_timestamps(local_idx, year_idx)

        # construct result tuple
        result = (inp, tar)

        if self.zenith_angle:
            zen_inp, zen_tar = self._compute_zenith_angle(inp_time, tar_time)
            result = result + (zen_inp, zen_tar)

        if self.return_timestamp:
            result = result + (inp_time, tar_time)

        torch.cuda.nvtx.range_pop()

        return result
