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


# Module-level grid state, populated on first call to ``_init_grid_module``.
# Subsequent calls are no-ops, and ``_copy_grid_state`` is used to shadow the
# state onto individual TestCase classes (so tests can keep using ``cls.x``
# / ``self.x`` while comm is initialised exactly once per process).
_GRID_STATE = None


def _init_grid_module():
    """Initialise the test comm groups exactly once per Python process.

    Reads ``GRID_H``/``GRID_W``/``GRID_FIN``/``GRID_FOUT``/``GRID_E`` env vars,
    calls ``comm.init`` and ``thd.init``, and captures every derived handle
    (group, rank, size, device) into the module-level ``_GRID_STATE`` dict.
    Idempotent: a second call is a no-op.
    """
    global _GRID_STATE
    if _GRID_STATE is not None:
        return

    grid_size_h = int(os.getenv("GRID_H", 1))
    grid_size_w = int(os.getenv("GRID_W", 1))
    grid_size_fin = int(os.getenv("GRID_FIN", 1))
    grid_size_fout = int(os.getenv("GRID_FOUT", 1))
    grid_size_e = int(os.getenv("GRID_E", 1))
    # GRID_B is optional. Most tests leave batch as -1 (auto-sized to fill the
    # remaining world); the metrics test sets it explicitly.
    grid_size_b = int(os.getenv("GRID_B", -1))
    world_size = grid_size_h * grid_size_w * grid_size_fin * grid_size_fout * grid_size_e
    if grid_size_b > 0:
        world_size *= grid_size_b

    comm.init(
        model_parallel_sizes=[grid_size_h, grid_size_w, grid_size_fin, grid_size_fout],
        model_parallel_names=["h", "w", "fin", "fout"],
        data_parallel_sizes=[grid_size_e, grid_size_b],
        data_parallel_names=["ensemble", "batch"],
    )

    world_rank = comm.get_world_rank()

    if torch.cuda.is_available():
        if world_rank == 0:
            print("Running test on GPU")
        local_rank = comm.get_local_rank()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(333)
    else:
        if world_rank == 0:
            print("Running test on CPU")
        device = torch.device("cpu")
    torch.manual_seed(333)

    state = dict(
        grid_size_h=grid_size_h,
        grid_size_w=grid_size_w,
        grid_size_fin=grid_size_fin,
        grid_size_fout=grid_size_fout,
        grid_size_e=grid_size_e,
        grid_size_b=comm.get_size("batch"),
        world_size=world_size,
        world_rank=world_rank,
        device=device,
        wrank=comm.get_rank("w"),
        hrank=comm.get_rank("h"),
        finrank=comm.get_rank("fin"),
        foutrank=comm.get_rank("fout"),
        erank=comm.get_rank("ensemble"),
        batchrank=comm.get_rank("batch"),
        w_group=comm.get_group("w"),
        h_group=comm.get_group("h"),
        fin_group=comm.get_group("fin"),
        fout_group=comm.get_group("fout"),
        e_group=comm.get_group("ensemble"),
        b_group=comm.get_group("batch"),
    )

    thd.init(state["h_group"], state["w_group"])

    if world_rank == 0:
        print(
            f"Running distributed tests on grid H x W x Fin x Fout x E = "
            f"{grid_size_h} x {grid_size_w} x "
            f"{grid_size_fin} x {grid_size_fout} x {grid_size_e}"
        )

    _GRID_STATE = state


def _copy_grid_state(cls):
    """Shadow the module-level grid state onto a TestCase class.

    ``_init_grid_module`` must have been called first (typically from
    ``setUpModule``). After this call every ``cls.<name>`` / ``self.<name>``
    access for the standard handle names (``device``, ``world_rank``,
    ``h_group``, …) resolves to the module-level value.
    """
    if _GRID_STATE is None:
        raise RuntimeError("_init_grid_module() must be called before _copy_grid_state()")
    for k, v in _GRID_STATE.items():
        setattr(cls, k, v)


def _init_grid(cls):
    """Backwards-compatible helper: init the module-level grid (idempotent)
    and shadow the handles onto ``cls`` in a single call. Use this from
    ``setUpClass`` when the test file has only one TestCase class; otherwise
    prefer ``setUpModule`` → ``_init_grid_module`` plus ``_copy_grid_state``
    in each ``setUpClass``."""
    _init_grid_module()
    _copy_grid_state(cls)
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
