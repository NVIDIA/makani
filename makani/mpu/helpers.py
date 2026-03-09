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

import torch
import torch.distributed as dist

from torch_harmonics.distributed import split_tensor_along_dim
from makani.utils import comm


class _DistributedConfig:
    """
    Module-level configuration for makani.mpu.
    Env vars are used as defaults but can be overridden programmatically, e.g.:

        from makani.mpu import config
        config.debug = True
    """

    def __init__(self):
        self._debug = None

    @property
    def debug(self):
        if self._debug is None:
            return os.getenv("MAKANI_DISTRIBUTED_DEBUG", "0") == "1"
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = bool(value)

    def __repr__(self):
        return f"_DistributedConfig(debug={self.debug})"

config = _DistributedConfig()


def _check_shapes(msg, shapes_gather, shapes_expected):
    for idx, (size_gather, size_expected) in enumerate(zip(shapes_gather, shapes_expected)):
        if size_gather != size_expected:
            raise ValueError(f"{msg} shapes are not correct. Expected {size_expected}, got {size_gather} for index {idx}. Please check that the number of chunks is correct.")


def _transpose(tensor, dim0, dim1, dim1_split_sizes, group=None, async_op=False, verify_shapes=None):

    if verify_shapes is None:
        verify_shapes = config.debug

    # get comm params
    comm_size = dist.get_world_size(group=group)
    comm_rank = dist.get_rank(group=group)

    # verify_shapes: check that dim1_split_sizes are correct:
    if verify_shapes:
        dim0_size = tensor.size(dim0)
        stens = torch.as_tensor([tensor.size(dim1)], dtype=torch.int64, device=tensor.device)
        stens_gather = [torch.empty_like(stens) for _ in range(comm_size)]
        stens_gather[comm_rank] = stens
        dist.all_gather(stens_gather, stens, group=group)
        sizes_gather = [stens.item() for stens in stens_gather]
        _check_shapes("_transpose: error, dim1_split_sizes", sizes_gather, dim1_split_sizes)

    # split and local transposition
    tsplit = split_tensor_along_dim(tensor, dim=dim0, num_chunks=comm_size)
    x_send = [y.contiguous() for y in tsplit]
    x_send_shapes = [x.shape for x in x_send]
    x_recv = []
    x_shape = list(x_send_shapes[comm_rank])
    for dim1_len in dim1_split_sizes:
        x_shape[dim1] = dim1_len
        x_recv.append(torch.empty(x_shape, dtype=tensor.dtype, device=tensor.device))

    # global transposition
    req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)

    # get dim0 split sizes
    dim0_split_sizes = [x[dim0] for x in x_send_shapes]

    if verify_shapes:
        stens = torch.as_tensor([tensor.size(dim0)], dtype=torch.int64, device=tensor.device)
        stens_gather = [torch.empty_like(stens) for _ in range(comm_size)]
        stens_gather[comm_rank] = stens
        dist.all_gather(stens_gather, stens, group=group)
        sizes_gather = [stens.item() for stens in stens_gather]
        _check_shapes("_transpose: error, dim0_split_sizes", sizes_gather, dim0_split_sizes)
        if sum(sizes_gather) != dim0_size:
            raise ValueError(f"_transpose: error, dim0_split_sizes do not sum to the correct size. Expected {dim0_size}, got {torch.sum(sizes_gather)}")

    return x_recv, dim0_split_sizes, req


def gather_uneven(tensor, dim, comm_name):
    if comm.get_size(comm_name) == 1:
        return tensor

    # make tensor contiguous, just in case:
    tensor = tensor.contiguous()

    # gather dims
    dim_tensor = torch.tensor([tensor.shape[dim]], dtype=torch.int, device=tensor.device)
    dim_list = [torch.empty_like(dim_tensor) for _ in range(comm.get_size(comm_name))]
    dim_list[comm.get_rank(comm_name)] = dim_tensor
    dist.all_gather(dim_list, dim_tensor, group=comm.get_group(comm_name))

    # gather tensor
    gathered_shape = list(tensor.shape)
    tensor_list = []
    for rshape in dim_list:
        gathered_shape[dim] = rshape.item()
        tensor_list.append(torch.empty(gathered_shape, dtype=tensor.dtype, device=tensor.device))

    tensor_list[comm.get_rank(comm_name)] = tensor
    dist.all_gather(tensor_list, tensor, group=comm.get_group(comm_name))

    # concatenate
    result = torch.cat(tensor_list, dim=dim)

    return result


def sync_params(model, mode="broadcast"):
    """Helper routine to ensure shared weights are the same after initialization"""

    def _sync_param(param, comm_group, mode):
        if comm.get_size(comm_group) > 1:
            if mode == "broadcast":
                is_complex = param.is_complex()
                if is_complex:
                    param_real = torch.view_as_real(param).clone()
                else:
                    param_real = param.clone()
                # tlist = [torch.empty_like(param_real) for x in range(comm.get_size(comm_group))]
                # tlist[comm.get_rank(comm_group)] = param_real
                # gather all weights in the comm group
                dist.broadcast(param_real, src=comm.get_root(comm_group), group=comm.get_group(comm_group), async_op=False)
                # use weight of rank 0
                # important to use copy here otherwise the handle gets detaches from the optimizer
                if is_complex:
                    param.copy_(torch.view_as_complex(param_real))
                else:
                    param.copy_(param_real)
            elif mode == "mean":
                is_complex = param.is_complex()
                if is_complex:
                    dist.all_reduce(torch.view_as_real(param), op=dist.ReduceOp.AVG, group=comm.get_group(comm_group), async_op=False)
                else:
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=comm.get_group(comm_group), async_op=False)
            else:
                raise ValueError(f"Unknown weight synchronization mode {mode}")

        return

    with torch.no_grad():
        # distributed sync step
        for param in model.parameters():
            # share along data dim
            _sync_param(param, "data", mode)

            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = ["model"]

            for comm_group in param.is_shared_mp:
                _sync_param(param, comm_group, mode)

    # synchronize the device to make sure all copies have finished
    if dist.is_initialized():
        device = next(model.parameters()).device
        dist.barrier(device_ids=[device.index])

    return
