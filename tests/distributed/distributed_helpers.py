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

import os

import torch
import torch.distributed as dist

from makani.utils import comm

import torch_harmonics.distributed as thd
from torch_harmonics.distributed import split_tensor_along_dim

from makani.utils.YParams import ParamsBase

H5_PATH = "fields"
NUM_CHANNELS = 5
IMG_SIZE_H = 64
IMG_SIZE_W = 128
CHANNEL_NAMES = ["u10m", "t2m", "u500", "z500", "t500"]

def get_default_parameters():

    # instantiate parameters
    params = ParamsBase()

    # dataset related
    params.dt = 1
    params.n_history = 0
    params.n_future = 0
    params.normalization = "none"
    params.data_grid_type = "equiangular"
    params.model_grid_type = "equiangular"
    params.sht_grid_type = "legendre-gauss"

    params.resuming = False
    params.amp_mode = "none"
    params.jit_mode = "none"
    params.disable_ddp = False
    params.checkpointing_level = 0
    params.enable_synthetic_data = False
    params.split_data_channels = False

    # dataloader related
    params.in_channels = list(range(NUM_CHANNELS))
    params.out_channels = list(range(NUM_CHANNELS))
    params.channel_names = [CHANNEL_NAMES[i] for i in range(NUM_CHANNELS)]

    # number of channels
    params.N_in_channels = len(params.in_channels)
    params.N_out_channels = len(params.out_channels)

    params.batch_size = 1
    params.valid_autoreg_steps = 0
    params.num_data_workers = 1
    params.multifiles = True
    params.io_grid = [1, 1, 1]
    params.io_rank = [0, 0, 0]

    # extra channels
    params.add_grid = False
    params.add_zenith = False
    params.add_orography = False
    params.add_landmask = False
    params.add_soiltype = False

    # logging stuff, needed for higher level tests
    params.log_to_screen = False
    params.log_to_wandb = False

    return params


def set_image_shape(params, h, w, *, h_resampled=None, w_resampled=None):
    """
    Set all image-shape-related fields on ``params`` consistently.

    Sets ``img_shape_x/y``, the ``img_local_shape_*`` and ``img_crop_shape_*``
    pair (which mirror the full shape until distributed sharding rewrites
    them), all related offsets to zero, and the resampled shapes (defaulting
    to the unsampled values, i.e. no resampling).

    Centralising this prevents tests from forgetting fields that downstream
    code requires — most recently the ``img_shape_*_resampled`` pair added
    during the resampling refactor (consumed by ``model_registry.get_model``,
    ``MetricsHandler``, ``LossHandler``).
    """
    params.img_shape_x = h
    params.img_shape_y = w
    params.img_local_shape_x = params.img_crop_shape_x = h
    params.img_local_shape_y = params.img_crop_shape_y = w
    params.img_local_offset_x = params.img_crop_offset_x = 0
    params.img_local_offset_y = params.img_crop_offset_y = 0
    params.img_shape_x_resampled = h if h_resampled is None else h_resampled
    params.img_shape_y_resampled = w if w_resampled is None else w_resampled


def _init_grid(cls):
    # set up distributed
    cls.grid_size_h = int(os.getenv("GRID_H", 1))
    cls.grid_size_w = int(os.getenv("GRID_W", 1))
    cls.grid_size_e = int(os.getenv("GRID_E", 1))
    cls.world_size = cls.grid_size_h * cls.grid_size_w * cls.grid_size_e

    # init groups
    comm.init(
        model_parallel_sizes=[cls.grid_size_h, cls.grid_size_w, 1, 1],
        model_parallel_names=["h", "w", "fin", "fout"],
        data_parallel_sizes=[cls.grid_size_e, -1],
        data_parallel_names=["ensemble", "batch"],
    )
    cls.world_rank = comm.get_world_rank()

    if torch.cuda.is_available():
        if cls.world_rank == 0:
            print("Running test on GPU")
        local_rank = comm.get_local_rank()
        cls.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(cls.device)
        torch.cuda.manual_seed(333)
    else:
        if cls.world_rank == 0:
            print("Running test on CPU")
        cls.device = torch.device("cpu")
    torch.manual_seed(333)

    # store comm group parameters
    cls.wrank = comm.get_rank("w")
    cls.hrank = comm.get_rank("h")
    cls.erank = comm.get_rank("ensemble")
    cls.w_group = comm.get_group("w")
    cls.h_group = comm.get_group("h")
    cls.e_group = comm.get_group("ensemble")

    # initializing sht process groups just to be sure
    thd.init(cls.h_group, cls.w_group)

    if cls.world_rank == 0:
        print(f"Running distributed tests on grid H x W x E = {cls.grid_size_h} x {cls.grid_size_w} x {cls.grid_size_e}")

    # make sure every rank finishes setup together
    sync_and_barrier()

    return


def _split_helper(tensor, dim=None, group=None):
    with torch.no_grad():
        if (dim is not None) and dist.get_world_size(group=group):
            gsize = dist.get_world_size(group=group)
            grank = dist.get_rank(group=group)
            # split in dim
            tensor_list_local = split_tensor_along_dim(tensor, dim=dim, num_chunks=gsize)
            tensor_local = tensor_list_local[grank]
        else:
            tensor_local = tensor.clone()

    return tensor_local


def _gather_helper(tensor, dim=None, group=None):
    # get shapes
    if (dim is not None) and (dist.get_world_size(group=group) > 1):
        gsize = dist.get_world_size(group=group)
        grank = dist.get_rank(group=group)
        shape_loc = torch.tensor([tensor.shape[dim]], dtype=torch.long, device=tensor.device)
        shape_list = [torch.empty_like(shape_loc) for _ in range(dist.get_world_size(group=group))]
        shape_list[grank] = shape_loc
        dist.all_gather(shape_list, shape_loc, group=group)
        tshapes = []
        for ids in range(gsize):
            tshape = list(tensor.shape)
            tshape[dim] = shape_list[ids].item()
            tshapes.append(tuple(tshape))
        tens_gather = [torch.empty(tshapes[ids], dtype=tensor.dtype, device=tensor.device) for ids in range(gsize)]
        tens_gather[grank] = tensor
        dist.all_gather(tens_gather, tensor, group=group)
        tensor_gather = torch.cat(tens_gather, dim=dim)
    else:
        tensor_gather = tensor.clone()

    return tensor_gather


def reduce_success(success, device, group=None):
    """All-reduce a per-rank boolean check result with logical AND (via MIN).

    Returns the *global* verdict on every rank so all ranks assert consistently: a
    failure on any single rank fails the test on all ranks. This avoids one rank
    raising (and bailing out of the test) while the others are still blocked on a
    subsequent collective -- which would otherwise hang the job at the next
    all-gather / teardown barrier. Combined with the rank-0-only reporting in
    conftest, it also ensures a silenced rank can never hide a failure.

    Usage (assert the reduced verdict on every rank)::

        ok = compare_tensors("output", out_full, out_gather, verbose=verbose)
        self.assertTrue(reduce_success(ok, self.device), "output")
    """
    if (not dist.is_initialized()) or (dist.get_world_size(group) == 1):
        return bool(success)
    flag = torch.tensor([1 if success else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def sync_and_barrier():
    """Synchronize the device and barrier across all ranks.

    Called at class setup/teardown boundaries so every rank reaches the boundary
    together -- no rank races ahead into the next test class (or out of one) while
    another is still running its kernels / collectives. No-ops when CUDA or
    torch.distributed are unavailable.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
