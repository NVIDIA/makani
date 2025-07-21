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

import types
import math

import torch
from torch.utils.data import DataLoader

# distributed stuff
import torch.distributed as dist
from makani.utils import comm

from makani.utils.dataloaders.data_helpers import get_data_normalization


def init_distributed_io(params):
    # set up sharding
    if dist.is_initialized():
        # this should always be safe now that data comm is orthogonal to
        # model comms
        params.data_num_shards = comm.get_size("batch")
        params.data_shard_id = comm.get_rank("batch")
        params.io_grid = [1, comm.get_size("h"), comm.get_size("w")]
        params.io_rank = [0, comm.get_rank("h"), comm.get_rank("w")]
    else:
        params.data_num_shards = 1
        params.data_shard_id = 0
        params.io_grid = [1, 1, 1]
        params.io_rank = [0, 0, 0]
        return

    # define IO grid:
    params.io_grid = [1, 1, 1] if not hasattr(params, "io_grid") else params.io_grid

    # to simplify, the number of total IO ranks has to be 1 or equal to the model parallel size
    num_io_ranks = math.prod(params.io_grid)
    assert (num_io_ranks == 1) or (num_io_ranks == comm.get_size("spatial"))
    assert (params.io_grid[1] == comm.get_size("h")) or (params.io_grid[1] == 1)
    assert (params.io_grid[2] == comm.get_size("w")) or (params.io_grid[2] == 1)

    # get io ranks: mp_rank = x_coord + params.io_grid[0] * (ycoord + params.io_grid[1] * zcoord)
    mp_rank = comm.get_rank("model")
    params.io_rank = [0, 0, 0]
    if params.io_grid[1] > 1:
        params.io_rank[1] = comm.get_rank("h")
    if params.io_grid[2] > 1:
        params.io_rank[2] = comm.get_rank("w")

    return


def get_dataloader(params, files_pattern, device, mode="train"):
    init_distributed_io(params)

    if (mode == "inference") and (not params.get("multifiles", False)):
        raise NotImplementedError("Error, only multifiles dataloader is supported in inference mode.")
    
    # get data normalization
    bias, scale = get_data_normalization(params)

    # sanity check
    if not params.get("multifiles", False):
        try:
            import nvidia.dali
        except:
            raise ImportError("Setting multifiles to False requires nvidia-dali, but module was not found.")

    if params.get("multifiles", False):
        from makani.utils.dataloaders.data_loader_multifiles import MultifilesDataset as MultifilesDataset2D
        from torch.utils.data.distributed import DistributedSampler

        dataset = MultifilesDataset2D(location=files_pattern,
                                      dt=params.get("dt"),
                                      in_channels=params.get("in_channels"),
                                      out_channels=params.get("out_channels"),
                                      n_history=params.get("n_history", 0),
                                      n_future=(params.get("valid_autoreg_steps") if (mode == "eval") else params.get("n_future", 0)),
                                      add_zenith=params.get("add_zenith", False),
                                      data_grid_type=params.get("data_grid_type", "equiangular"),
                                      model_grid_type=params.get("model_grid_type", "equiangular"),
                                      bias=bias,
                                      scale=scale,
                                      crop_size=(params.get("crop_size_x", None), params.get("crop_size_y", None)),
                                      crop_anchor=(params.get("crop_anchor_x", 0), params.get("crop_anchor_y", 0)),
                                      return_timestamp=(True if (mode == "inference") else False),
                                      return_target=(False if (mode == "inference") else True),
                                      file_suffix=params.get("dataset_file_suffix", "h5"),
                                      enable_s3=params.get("enable_s3", False),
                                      io_grid=params.get("io_grid", [1,1,1]),
                                      io_rank=params.get("io_rank", [0,0,0]),
        )
        
        if mode in ["train", "eval"]:
            sampler = DistributedSampler(dataset, shuffle=(mode == "train"), num_replicas=params.data_num_shards, rank=params.data_shard_id) if (params.data_num_shards > 1) else None
            dataloader = DataLoader(
                dataset,
                batch_size=int(params.batch_size),
                num_workers=params.num_data_workers,
                shuffle=((sampler is None) and (mode == "train")),
                sampler=sampler,
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            # this will all be handled by the inferencer
            sampler = None
            dataloader = types.SimpleNamespace()

        # for compatibility with the DALI dataloader
        dataloader.lat_lon = dataset.lat_lon
        dataloader.get_output_normalization = dataset.get_output_normalization
        dataloader.get_input_normalization = dataset.get_input_normalization

    elif params.enable_synthetic_data:
        from makani.utils.dataloaders.data_loader_dummy import DummyLoader

        # use for true dummy loading
        img_shape_x = params.get("img_shape_x", None)
        img_shape_y = params.get("img_shape_y", None)
        img_shape = None
        if (img_shape_x is not None) and (img_shape_y is not None):
            img_shape = (img_shape_x, img_shape_y)
        
        dataloader = DummyLoader(
            location=files_pattern,
            device=device,
            batch_size=params.get("batch_size"),
            dt=params.get("dt"),
            dhours=params.get("dhours"),
            in_channels=params.get("in_channels"),
            out_channels=params.get("out_channels"),
            img_shape=img_shape,
            max_samples=(params.get("n_train_samples", None) if (mode == "train") else params.get("n_eval_samples", None)),
            n_samples_per_epoch=(params.get("n_train_samples_per_epoch", None) if (mode == "train") else params.get("n_eval_samples_per_epoch", None)),
            n_history=params.get("n_history", 0),
            n_future=(params.get("valid_autoreg_steps") if (mode == "eval") else params.get("n_future", 0)),
            add_zenith=params.get("add_zenith", False),
            latitudes=params.get("lat", None),
            longitudes=params.get("lon", None),
            data_grid_type=params.get("data_grid_type", "equiangular"),
            model_grid_type=params.get("model_grid_type", "equiangular"),
            crop_size=(params.get("crop_size_x", None), params.get("crop_size_y", None)),
            crop_anchor=(params.get("crop_anchor_x", 0), params.get("crop_anchor_y", 0)),
            return_timestamp=(True if (mode == "inference") else False),
            return_target=(False if (mode == "inference") else True),
            io_grid=params.get("io_grid", [1,1,1]),
            io_rank=params.get("io_rank", [0,0,0]),
        )

        dataset = types.SimpleNamespace(
            in_channels=dataloader.in_channels,
            out_channels=dataloader.out_channels,
            grid_converter=dataloader.grid_converter,
            img_shape_x=dataloader.img_shape[0],
            img_shape_y=dataloader.img_shape[1],
            img_crop_shape_x=dataloader.crop_shape[0],
            img_crop_shape_y=dataloader.crop_shape[1],
            img_crop_offset_x=dataloader.crop_anchor[0],
            img_crop_offset_y=dataloader.crop_anchor[1],
            img_local_shape_x=dataloader.read_shape[0],
            img_local_shape_y=dataloader.read_shape[1],
            img_local_offset_x=dataloader.read_anchor[0],
            img_local_offset_y=dataloader.read_anchor[1],
        )

        # not needed for the no multifiles case
        sampler = None

    else:
        from makani.utils.dataloaders.data_loader_dali_2d import ERA5DaliESDataloader as ERA5DaliESDataloader2D

        # dali loader
        dali_device = "gpu" if torch.cuda.is_available() else "cpu"
        dataloader = ERA5DaliESDataloader2D(params, files_pattern, (mode == "train"), dali_device=dali_device)

        dataset = types.SimpleNamespace(
            in_channels=dataloader.in_channels,
            out_channels=dataloader.out_channels,
            grid_converter=dataloader.grid_converter,
            img_shape_x=dataloader.img_shape_x,
            img_shape_y=dataloader.img_shape_y,
            img_crop_shape_x=dataloader.img_crop_shape_x,
            img_crop_shape_y=dataloader.img_crop_shape_y,
            img_crop_offset_x=dataloader.img_crop_offset_x,
            img_crop_offset_y=dataloader.img_crop_offset_y,
            img_local_shape_x=dataloader.img_local_shape_x,
            img_local_shape_y=dataloader.img_local_shape_y,
            img_local_offset_x=dataloader.img_local_offset_x,
            img_local_offset_y=dataloader.img_local_offset_y,
        )

        if params.enable_benchy and (mode == "train"):
            from benchy.torch import BenchmarkGenericIteratorWrapper

            dataloader = BenchmarkGenericIteratorWrapper(dataloader, params.batch_size)

        # not needed for the no multifiles case
        sampler = None

    return dataloader, dataset, sampler
