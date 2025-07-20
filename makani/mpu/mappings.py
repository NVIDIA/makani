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
from typing import Any

import torch
from torch.amp import custom_fwd, custom_bwd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from makani.utils import comm

# torch utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# we need those
from makani.mpu.helpers import _transpose

# we need the parameter counter
from makani.models.helpers import count_parameters


class distributed_transpose(torch.autograd.Function):

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, dims, dim1_split_sizes, comm_id):
        # WAR for a potential contig check torch bug for channels last contig tensors
        x = x.contiguous()
        xlist, dim0_split_sizes, _ = _transpose(x, dims[0], dims[1], dim1_split_sizes, group=comm.get_group(comm_id))
        x = torch.cat(xlist, dim=dims[1]).contiguous()
        ctx.dims = dims
        ctx.dim0_split_sizes = dim0_split_sizes
        ctx.comm_id = comm_id
        return x

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, go):
        dims = ctx.dims
        dim0_split_sizes = ctx.dim0_split_sizes
        # WAR for a potential contig check torch bug for channels last contig tensors
        go = go.contiguous()
        gilist, _, _ = _transpose(go, dims[1], dims[0], dim0_split_sizes, group=comm.get_group(ctx.comm_id))
        gi = torch.cat(gilist, dim=dims[0]).contiguous()
        return gi, None, None, None


# handler for additional gradient reductions
# helper for gradient reduction across channel parallel ranks
def init_gradient_reduction_hooks(model, device, reduction_buffer_count=1, broadcast_buffers=True, find_unused_parameters=False, gradient_as_bucket_view=True, static_graph=False, verbose=False):
    # early exit if we are not in a distributed setting:
    if not dist.is_initialized():
        return model

    # set this to false in init and then find out if we can use it:
    need_hooks = False
    ddp_group = comm.get_group("data")

    # this is the trivial case
    if comm.get_size("model") == 1:
        # the simple case, we can just continue then
        ddp_group = None
    else:
        # count parameters and reduction groups
        num_parameters_total = 0
        num_parameters_shared_model = 0
        for param in model.parameters():
            # if it does not have any annotation, we assume it is shared between all model ranks
            if not hasattr(param, "is_shared_mp"):
                if verbose:
                    print(f"Parameter {param.name} has no sharing mode specified, settting to globally shared.")
                param.is_shared_mp = ["model"]

            # add the sharing type to the dict
            num_parameters_total += 1
            if "model" in param.is_shared_mp:
                num_parameters_shared_model += 1

        # if all parameters are shared between all model ranks, then the situation is easy
        if num_parameters_shared_model == num_parameters_total:
            # we can always use DDP
            ddp_group = None

            # register some pre-multiply reduction hooks
            if verbose:
                print("Setting up gradient hooks to account for shared parameter multiplicity")
            for param in model.parameters():
                param.register_hook(lambda grad: grad * float(comm.get_size("model")))
        else:
            ddp_group = comm.get_group("data")
            broadcast_buffers = False
            need_hooks = True

    # compute bucket cap in MB:
    if need_hooks:
        # if we need hooks, we can only use a single reduction buffer:
        reduction_buffer_count = 1

    # determine size of model. Only local number of parameters is relevant:
    _, _, local_parameter_size_bytes = count_parameters(model, device)

    # compute reduction buffer size
    reduction_size_bytes = (local_parameter_size_bytes + reduction_buffer_count - 1) // reduction_buffer_count
    reduction_size_mb = (reduction_size_bytes + 1048575) // 1048576

    # we should fuse the first bucket with the others
    dist._DEFAULT_FIRST_BUCKET_BYTES = reduction_size_mb * 1048576

    # we can set up DDP and exit here
    if verbose:
        print("Setting up DDP communication hooks")
    model = DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
        bucket_cap_mb=reduction_size_mb,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        process_group=ddp_group,
    )

    if not need_hooks:
        return model

    if verbose:
        print("Setting up custom communication hooks")

    # define comm hook:
    def reduction_comm_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        # allreduce everything first:
        buff = bucket.buffer()
        params = bucket.parameters()

        # define the grad reduction function
        def grad_reduction(fut, grads, group, reduction="sum"):
            # check if grads are complex
            is_complex = [g.is_complex() for g in grads]
            grads_real = [torch.view_as_real(g) if g.is_complex() else g for g in grads]

            # flatten
            coalesced = _flatten_dense_tensors(grads_real)

            # reduce
            if reduction == "sum":
                dist.all_reduce(coalesced, op=dist.ReduceOp.SUM, group=comm.get_group(group), async_op=False)
            elif reduction == "mean":
                dist.all_reduce(coalesced, op=dist.ReduceOp.AVG, group=comm.get_group(group), async_op=False)
            else:
                raise NotImplementedError(f"Error, reduction {reduction} not supported.")

            # copy back
            for buf, synced_real, is_comp in zip(grads, _unflatten_dense_tensors(coalesced, grads_real), is_complex):
                if is_comp:
                    synced = torch.view_as_complex(synced_real)
                else:
                    synced = synced_real
                buf.copy_(synced)

            return bucket.buffer()

        # WAR: we need to add a workaround for complex gradients here, therefore we need to hack the allreduce step a little bit.
        # once this is fixed, the below line can be uncommented and we can remove the hack
        # get future for allreduce
        # fut = dist.all_reduce(buff, op=dist.ReduceOp.AVG, group=comm.get_group("data"), async_op=True).get_future()

        # get future
        fut = torch.futures.Future()
        fut.set_result(bucket.buffer())

        # get the data gradients first:
        grads = []
        for p in params:
            if p.grad is not None:
                grads.append(p.grad.data)

        if grads:
            fut = fut.then(lambda x: grad_reduction(x, grads=grads, group="data", reduction="mean"))

        # now go through the groups
        for group in comm.get_comm_names():
            if group == "data":
                continue

            # real first
            grads = []
            for p in params:
                if (p.grad is not None) and (group in p.is_shared_mp):
                    grads.append(p.grad.data)

            # append the new reduction functions
            if grads:
                fut = fut.then(lambda x: grad_reduction(x, grads=grads, group=group, reduction="sum"))

        return fut

    # register model comm hook
    model.register_comm_hook(state=None, hook=reduction_comm_hook)

    return model
