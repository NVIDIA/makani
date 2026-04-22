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

import torch
from torch import nn
import torch.distributed as dist
from makani.utils import comm


def get_memory_usage(device):
    free_mem, total_mem = torch.cuda.mem_get_info(device=device)
    allocated_mem_gb = (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0)
    torch_mem_gb = torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0 * 1024.0)

    return allocated_mem_gb, torch_mem_gb


def normalize_weights(model, eps=1e-5):
    for param in model.parameters():
        # numel = torch.tensor(param.numel(), dtype=torch.long, device=param.device)

        # compute norm: compute abs first to support complex weights
        norm = torch.sum(torch.square(torch.abs(param)))

        # compute local norm
        if hasattr(param, "sharded_dims_mp"):

            for d, group in enumerate(param.sharded_dims_mp):
                # continue if there is nothing to do
                if (group is None) or (comm.get_size(group) == 1):
                    continue

                dist.all_reduce(norm, group=comm.get_group(group))
                # dist.all_reduce(numel, group=comm.get_group(group))

        norm = torch.sqrt(norm)

        # update weights
        param.mul_(1.0 / (norm + eps))

    return


def _compute_total_grad_norm(model, norm_type=2.0, verbose=False):
    # iterate over parameters
    gnorms = []
    for name, param in model.named_parameters():

        if param.grad is None:
            continue

        # compute local norm: compute abs first to support complex grads
        if norm_type == 2.0:
            gnorm = torch.sum(torch.square(torch.abs(param.grad)))
        else:
            gnorm = torch.sum(torch.abs(param.grad))

        # compute global norm
        if hasattr(param, "sharded_dims_mp"):

            for group in param.sharded_dims_mp:
                # continue if there is nothing to do
                if (group is None) or (comm.get_size(group) == 1):
                    continue

                dist.all_reduce(gnorm, group=comm.get_group(group))

        gnorms.append(gnorm)

    if verbose:
        for gnorm in gnorms:
            if torch.any(torch.isnan(gnorm)):
                print(f"Gradient norm is NaN for parameter {name}")

    # compute total norm
    if gnorms:
        total_gnorm = torch.sum(torch.stack(gnorms))
    else:
        total_gnorm = torch.tensor(0.0, device=model.device)

    # post-process norm
    if norm_type == 2.0:
        total_gnorm = torch.sqrt(total_gnorm)

    return total_gnorm


def clip_grads(model, max_grad_norm, norm_type=2.0, verbose=False):

    # iterate over parameters
    with torch.no_grad():
        total_gnorm = _compute_total_grad_norm(model, norm_type=norm_type, verbose=verbose)

        clip_factor = max_grad_norm / (total_gnorm + 1e-6)  # add small epsilon to avoid division by zero
        clip_factor = torch.clamp(clip_factor, max=1.0)

        for param in model.parameters():
            if param.grad is None:
                continue

            param.grad.mul_(clip_factor)

    return total_gnorm


def wandb_register_activations_monitor(model: nn.Module, step: int):

    def check_eligibility(module: nn.Module) -> bool:
        is_activation = False
        is_activation = is_activation or isinstance(module, nn.ReLU)
        is_activation = is_activation or isinstance(module, nn.LeakyReLU)
        is_activation = is_activation or isinstance(module, nn.GELU)
        is_activation = is_activation or isinstance(module, nn.Sigmoid)
        is_activation = is_activation or isinstance(module, nn.SiLU)

        return is_activation

    for submodule in model.modules():
        if check_eligibility(submodule):

            def log_activation(module, step, output):
                name = module.name
                if output.is_complex:
                    normsq = output * output.conj()
                else:
                    normsq = torch.square(output)
                norm = torch.sqrt(torch.sum(normsq))
                wandb.log(f"activation {name}", norm, step=step)

            submodule.register_forward_hook(log_activation, step)

    return
