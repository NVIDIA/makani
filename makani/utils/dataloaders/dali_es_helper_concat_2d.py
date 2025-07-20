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
import numpy as np
import h5py
import zarr
import logging
from itertools import groupby, accumulate
import operator
from bisect import bisect_right

# for nvtx annotation
import torch

# we need this for the zenith angle feature
from .data_helpers import get_timestamp, get_date_from_timestamp

# import splitting logic
from physicsnemo.distributed.utils import compute_split_shapes

# coszen
from makani.third_party.climt.zenith_angle import cos_zenith_angle


class GeneralConcatES(object):
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

        # O_DIRECT specific stuff
        self.file_driver = "direct" if enable_odirect else None
        self.read_direct = True  # if enable_odirect else True
        self.num_retries = 5

        # also obtain an ordered channels list, required for h5py
        self.in_channels_sorted = np.sort(self.in_channels)
        self.in_channels_unsort = np.argsort(np.argsort(self.in_channels))
        self.in_channels_is_sorted = np.all(self.in_channels_sorted == self.in_channels)
        # out_channels
        self.out_channels_sorted = np.sort(self.out_channels)
        self.out_channels_unsort = np.argsort(np.argsort(self.out_channels))
        self.out_channels_is_sorted = np.all(self.out_channels_sorted == self.out_channels)

        # sanity checks
        if enable_s3:
            raise NotImplementedError(f"s3 support currently not implemented for concatenated files.")

        # set the read slices
        # we do not support channel parallelism yet
        assert io_grid[0] == 1
        self.io_grid = io_grid[1:]
        self.io_rank = io_rank[1:]

        # datetime logic
        self.timezone_fn = np.vectorize(get_date_from_timestamp)

        # parse the files
        self._get_files_stats(enable_logging)
        self.shuffle = True if train else False

        # convert in_channels to list of slices:
        self.in_channels_slices = list(self._get_slices(self.in_channels_sorted))
        self.out_channels_slices = list(self._get_slices(self.out_channels_sorted))

        # we need some additional static fields in this case
        if self.lat_lon is None:
            resolution = 360.0 / float(self.img_shape[1])
            longitude = np.arange(0, 360, resolution)
            latitude = np.arange(-90, 90 + resolution, resolution)
            latitude = latitude[::-1]
            self.lat_lon = (latitude.tolist(), longitude.tolist())

        latitude = np.array(self.lat_lon[0])
        longitude = np.array(self.lat_lon[1])
        self.lon_grid, self.lat_grid = np.meshgrid(longitude, latitude)
        self.lat_grid_local = self.lat_grid[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0], self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]]
        self.lon_grid_local = self.lon_grid[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0], self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]]
        self.lat_lon_local = (
            latitude[self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0]].tolist(),
            longitude[self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1]].tolist(),
        )

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

    def _get_data_h5(self, dset, sample_idx, start_x, end_x, start_y, end_y):
        off = 0
        for slice_in in self.in_channels_slices:
            start = off
            end = start + (slice_in.stop - slice_in.start)

            # read the data
            if self.read_direct:
                dset.read_direct(
                    self.inp_buff,
                    np.s_[(sample_idx - self.dt * self.n_history) : (sample_idx + 1) : self.dt,
                          slice_in, start_x:end_x, start_y:end_y],
                    np.s_[:, start:end, ...])
            else:
                self.inp_buff[:, start:end, ...] = dset[(local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                                                        slice_in, start_x:end_x, start_y:end_y]

            # update offset
            off = end

        off = 0
        for slice_out in self.out_channels_slices:
            start = off
            end = start + (slice_out.stop - slice_out.start)

            # read the data
            if self.read_direct:
                dset.read_direct(
                    self.tar_buff,
                    np.s_[(sample_idx + self.dt) : (sample_idx + self.dt * (self.n_future + 1) + 1) : self.dt,
                          slice_out, start_x:end_x, start_y:end_y],
                    np.s_[:, start:end, ...],
                )
            else:
                self.tar_buff[:, start:end, ...] = dset[(local_idx + self.dt) : (local_idx + self.dt * (self.n_future + 1) + 1) : self.dt, slice_out, start_x:end_x, start_y:end_y]

            # update offset
            off = end

        # reorder data if requested:
        inp, tar = self._reorder_channels(self.inp_buff, self.tar_buff)

        return inp, tar

    def _get_files_stats(self, enable_logging):
        # check for h5v file
        self.file_path = self.location
        self.file_format = "h5"

        # throw error if file could not be found
        if not os.path.isfile(self.file_path):
            raise IOError(f"Error, the specified file path {self.location} does not contain an h5 file.")

        # open file:
        self.vfile = None
        with h5py.File(self.file_path, "r") as f:
            dset = f[self.dataset_path]

            # extract timestamps and convert them to datetime objects
            self.timestamps = self.timezone_fn(f[self.dataset_path].dims[0]["timestamp"][...])

            # extract number of years
            self.years = sorted(list(set([d.year for d in self.timestamps.tolist()])))

            # get stats
            self.n_years = len(self.years)

            # get stats from first file
            self.img_shape = dset.shape[2:4]
            self.total_channels = dset.shape[1]
            self.n_samples_available = dset.shape[0]

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
            start_date = self.timestamps[self.n_samples_offset]
            end_date = self.timestamps[self.n_samples_available-1]
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

    def _compute_timestamps_and_zenith_angle(self, sample_idx, compute_zenith_angle):
        # nvtx range
        torch.cuda.nvtx.range_push("GeneralES:_compute_timestamps_and_zenith_angle")

        # zenith angle for input
        inp_time = self.timestamps[sample_idx - self.dt * self.n_history : sample_idx + 1 : self.dt]
        if compute_zenith_angle:
            cos_zenith_inp = np.expand_dims(cos_zenith_angle(inp_time, self.lon_grid_local, self.lat_grid_local).astype(np.float32), axis=1)
        else:
            cos_zenith_inp = None

        # zenith angle for target:
        tar_time = self.timestamps[sample_idx + self.dt : sample_idx + self.dt * (self.n_future + 1) + 1 : self.dt]
        if compute_zenith_angle:
            cos_zenith_tar = np.expand_dims(cos_zenith_angle(tar_time, self.lon_grid_local, self.lat_grid_local).astype(np.float32), axis=1)
        else:
            cos_zenith_tar = None

        # nvtx range
        torch.cuda.nvtx.range_pop()

        return cos_zenith_inp, cos_zenith_tar, inp_time, tar_time

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.get_data_handle = self._get_data_h5

        # open file:
        for _ in range(self.num_retries):
            try:
                self.vfile = h5py.File(self.file_path, "r", driver=self.file_driver)
                break
            except Exception as err:
                print(f"Cannot open file {self.file_path}. Reason {err}, retrying.", flush=True)
                time.sleep(5)
            else:
                raise OSError(f"Unable to retrieve year handle {year_idx}, aborting.")

        # get dataset handle
        self.dset = self.vfile[self.dataset_path]

        if self.is_parallel:
            self._init_buffers()

    def __len__(self):
        return self.n_samples_shard

    def __del__(self):
        # close files
        if self.vfile is not None:
            self.vfile.close()

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

        # compute sample idx
        sample_idx = self.index_permutation[cycle_sample_idx]

        # if we are not at least self.dt*n_history timesteps into the prediction
        if sample_idx < self.dt * self.n_history:
            sample_idx += self.dt * self.n_history

        if sample_idx >= (self.n_samples_available - self.dt * (self.n_future + 1)):
            sample_idx = self.n_samples_available - self.dt * (self.n_future + 1) - 1

        # load slice of data:
        start_x = self.read_anchor[0]
        end_x = start_x + self.read_shape[0]

        start_y = self.read_anchor[1]
        end_y = start_y + self.read_shape[1]

        # read data
        inp, tar = self.get_data_handle(self.dset, sample_idx, start_x, end_x, start_y, end_y)

        # start constructing result
        result = (inp, tar)

        # get time grid
        if self.zenith_angle or self.return_timestamp:
            zen_inp, zen_tar, inp_time, tar_time = self._compute_timestamps_and_zenith_angle(sample_idx, self.zenith_angle)

            if self.zenith_angle:
                result = result + (zen_inp, zen_tar)

            if self.return_timestamp:
                result = result + (inp_time, tar_time)

        torch.cuda.nvtx.range_pop()

        return result
