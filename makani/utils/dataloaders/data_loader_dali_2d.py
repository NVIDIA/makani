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
import glob
import torch
import random
import numpy as np
import torch
from torch import Tensor
import math

# import cv2

# DALI stuff
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as dali_types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# distributed stuff
import torch.distributed as dist
from makani.utils import comm

# es helper
from makani.utils.dataloaders.data_helpers import get_data_normalization
from makani.utils.grids import GridConverter


class ERA5DaliESDataloader(object):

    def get_pipeline(self):
        pipeline = Pipeline(
            batch_size=self.batchsize,
            num_threads=2,
            device_id=self.device_index,
            py_num_workers=self.num_data_workers,
            py_start_method="spawn",
            seed=self.global_seed,
            prefetch_queue_depth=2,
        )

        img_shape_x = self.img_shape_x
        img_shape_y = self.img_shape_y
        in_channels = self.in_channels
        out_channels = self.out_channels

        num_outputs = 2
        layout = ["FCHW", "FCHW"]
        if self.add_zenith:
            num_outputs += 2
            layout += ["FCHW", "FCHW"]
        if self.return_timestamp:
            num_outputs += 2
            layout += ["F"]

        with pipeline:
            # get input and target
            data = fn.external_source(
                source=self.extsource,
                num_outputs=num_outputs,
                layout=layout,
                batch=False,
                no_copy=True,
                parallel=True,
                prefetch_queue_depth=self.num_data_workers,
            )

            inp = data[0]
            tar = data[1]
            off = 2
            if self.add_zenith:
                izen = data[2]
                tzen = data[3]
                off += 2

            if self.return_timestamp:
                itime = data[0 + off]
                ttime = data[1 + off]

            # upload to GPU
            if self.dali_device == "gpu":
                inp = inp.gpu()
                tar = tar.gpu()
                if self.add_zenith:
                    izen = izen.gpu()
                    tzen = tzen.gpu()

            # normalize if requested
            if self.normalize:
                inp = fn.normalize(inp, device=self.dali_device, axis_names=self.norm_channels, batch=self.norm_batch, mean=self.in_bias, stddev=self.in_scale)

                tar = fn.normalize(tar, device=self.dali_device, axis_names=self.norm_channels, batch=self.norm_batch, mean=self.out_bias, stddev=self.out_scale)

            # add zenith angle if requested
            pout = (inp, tar)

            if self.add_zenith:
                pout = pout + (izen, tzen)

            if self.return_timestamp:
                pout = pout + (itime, ttime)

            pipeline.set_outputs(*pout)

        return pipeline

    def __init__(self, params, location, train, seed=333, dali_device="gpu"):
        # set up workers and devices
        self.num_data_workers = params.num_data_workers
        self.dali_device = dali_device
        if self.dali_device == "gpu":
            self.device_index = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{self.device_index}")
        elif self.dali_device == "cpu":
            self.device_index = None
            self.device = torch.device("cpu")
        else:
            raise NotImplementedError(f"Makani currently does not support dali_device {self.dali_device}")

        # batch size
        self.batchsize = int(params.batch_size)

        # set up seeds
        # this one is the same on all ranks
        self.global_seed = seed
        # this one is the same for all ranks of the same model
        model_id = comm.get_world_rank() // comm.get_size("model")
        self.model_seed = self.global_seed + model_id
        # this seed is supposed to be diffferent for every rank
        self.local_seed = self.global_seed + comm.get_world_rank()

        # we need to copy those
        self.location = location
        self.train = train
        self.dt = params.dt
        self.dhours = params.dhours
        self.n_history = params.n_history
        self.n_future = params.n_future if train else params.valid_autoreg_steps
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.add_zenith = params.get("add_zenith", False)
        self.return_timestamp = params.get("return_timestamp", False)
        if hasattr(params, "lat") and hasattr(params, "lon"):
            self.lat_lon = (params.lat, params.lon)
        else:
            self.lat_lon = None
        self.dataset_path = params.h5_path
        if train:
            self.n_samples = params.get("n_train_samples", None)
            self.n_samples_per_epoch = params.get("n_train_samples_per_epoch", None)
        else:
            self.n_samples = params.get("n_eval_samples", None)
            self.n_samples_per_epoch = params.get("n_eval_samples_per_epoch", None)

        # by default we normalize over space
        self.norm_channels = "FHW"
        self.norm_batch = False
        if hasattr(params, "normalization_mode"):
            split = params.data_normalization_mode.split("-")
            self.norm_mode = split[0]
            if len(split) > 1:
                self.norm_channels = split[1]
                if "B" in self.norm_channels:
                    self.norm_batch = True
                    self.norm_channels.replace("B", "")
        else:
            self.norm_mode = "offline"

        # set sharding
        self.num_shards = params.data_num_shards
        self.shard_id = params.data_shard_id

        # get cropping:
        crop_size = [params.get("crop_size_x", None), params.get("crop_size_y", None)]
        crop_anchor = [params.get("crop_anchor_x", 0), params.get("crop_anchor_y", 0)]

        if os.path.isfile(self.location):
            from makani.utils.dataloaders.dali_es_helper_concat_2d import GeneralConcatES as GeneralES
        elif os.path.isdir(self.location):
            from makani.utils.dataloaders.dali_es_helper_2d import GeneralES
        else:
            raise IOError(f"Path {self.location} does not exist.")

        # get the image sizes
        self.extsource = GeneralES(
            self.location,
            max_samples=self.n_samples,
            samples_per_epoch=self.n_samples_per_epoch,
            train=self.train,
            batch_size=self.batchsize,
            dt=self.dt,
            dhours=self.dhours,
            n_history=self.n_history,
            n_future=self.n_future,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            crop_size=crop_size,
            crop_anchor=crop_anchor,
            num_shards=self.num_shards,
            shard_id=self.shard_id,
            io_grid=params.get("io_grid", [1, 1, 1]),
            io_rank=params.get("io_rank", [0, 0, 0]),
            device_id=self.device_index,
            truncate_old=True,
            zenith_angle=self.add_zenith,
            return_timestamp=self.return_timestamp,
            lat_lon=self.lat_lon,
            dataset_path=self.dataset_path,
            enable_odirect=params.enable_odirect,
            enable_s3=params.enable_s3,
            enable_logging=params.log_to_screen,
            seed=self.global_seed,
            is_parallel=True,
        )

        # grid types
        self.grid_converter = GridConverter(
            params.data_grid_type,
            params.model_grid_type,
            torch.deg2rad(torch.tensor(self.extsource.lat_lon_local[0])).to(torch.float32).to(self.device),
            torch.deg2rad(torch.tensor(self.extsource.lat_lon_local[1])).to(torch.float32).to(self.device),
        )

        # some image properties
        self.img_shape_x = self.extsource.img_shape[0]
        self.img_shape_y = self.extsource.img_shape[1]

        self.img_crop_shape_x = self.extsource.crop_size[0]
        self.img_crop_shape_y = self.extsource.crop_size[1]
        self.img_crop_offset_x = self.extsource.crop_anchor[0]
        self.img_crop_offset_y = self.extsource.crop_anchor[1]

        self.img_local_shape_x = self.extsource.read_shape[0]
        self.img_local_shape_y = self.extsource.read_shape[1]
        self.img_local_offset_x = self.extsource.read_anchor[0]
        self.img_local_offset_y = self.extsource.read_anchor[1]

        # num steps
        self.num_steps_per_epoch = self.extsource.num_steps_per_epoch

        # load stats
        self.normalize = True

        # in
        if self.norm_mode == "offline":
            bias, scale = get_data_normalization(params)

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

            # reformat the biases
            if self.norm_channels == "FHW":
                in_shape = (1, len(self.in_channels), 1, 1)
                out_shape = (1, len(self.out_channels), 1, 1)
            else:
                in_shape = (1, *self.in_bias.shape)
                out_shape = (1, *self.out_bias.shape)

            self.in_bias = np.reshape(self.in_bias, in_shape)
            self.in_scale = np.reshape(self.in_scale, in_shape)
            self.out_bias = np.reshape(self.out_bias, out_shape)
            self.out_scale = np.reshape(self.out_scale, out_shape)
        else:
            # in case of online normalization,
            # we do not need to set it here
            self.in_bias = None
            self.in_scale = None
            self.out_bias = None
            self.out_scale = None

        # create pipeline
        self.pipeline = self.get_pipeline()
        self.pipeline.start_py_workers()
        self.pipeline.build()

        # create iterator
        outnames = ["inp", "tar"]
        if self.add_zenith:
            outnames += ["izen", "tzen"]
        if self.return_timestamp:
            outnames += ["itime", "ttime"]

        self.iterator = DALIGenericIterator([self.pipeline], outnames, auto_reset=True, size=-1, last_batch_policy=LastBatchPolicy.DROP, prepare_first_batch=True)

    def get_input_normalization(self):
        if self.norm_mode == "offline":
            return self.in_bias, self.in_scale
        else:
            return 0.0, 1.0

    def get_output_normalization(self):
        if self.norm_mode == "offline":
            return self.out_bias, self.out_scale
        else:
            return 0.0, 1.0

    def reset_pipeline(self):
        self.pipeline.reset()
        self.iterator.reset()

    def __len__(self):
        return self.num_steps_per_epoch

    def __iter__(self):
        # self.iterator.reset()
        for token in self.iterator:
            inp = token[0]["inp"]
            tar = token[0]["tar"]

            # construct result
            result = (inp, tar)

            if self.add_zenith:
                izen = token[0]["izen"]
                tzen = token[0]["tzen"]
                result = result + (izen, tzen)

            # convert grid
            with torch.no_grad():
                result = tuple(map(lambda x: self.grid_converter(x), result))

            if self.return_timestamp:
                itime = token[0]["itime"]
                ttime = token[0]["ttime"]
                result = result + (itime, ttime)

            yield result
