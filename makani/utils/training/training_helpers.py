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
    with torch.no_grad():
        # Group params by their MP sharding signature — same pattern as clip_grads.
        groups: dict[tuple, list] = {}
        for param in model.parameters():
            key = tuple(
                g for g in getattr(param, "sharded_dims_mp", [])
                if g is not None and comm.get_size(g) > 1
            )
            groups.setdefault(key, []).append(param)

        for mp_groups, params in groups.items():
            # _foreach_norm: one fused kernel for all local norms in the group
            norms = torch._foreach_norm(params, ord=2)

            if mp_groups:
                # Sharded params: we must all_reduce the *squared* partial norms
                # (sum-of-squares is additive across shards) then re-sqrt.
                for norm, param in zip(norms, params):
                    norm.pow_(2)
                    for g in mp_groups:
                        dist.all_reduce(norm, group=comm.get_group(g))
                    norm.sqrt_()

            # per-param scale factors: 1/(norm + eps) — then one fused weight update
            scale_factors = torch._foreach_reciprocal(torch._foreach_add(norms, eps))
            torch._foreach_mul_(params, scale_factors)

    return


def _compute_total_grad_norm(model, norm_type=2.0, verbose=False):
    # Group parameters by their model-parallel sharding signature so we can
    # issue one all_reduce per process group instead of one per parameter,
    # and use _foreach_norm to collapse N small reduce kernels into one.
    groups: dict[tuple, list] = {}
    param_map: dict[tuple, list] = {}  # for verbose NaN reporting
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = tuple(
            g for g in getattr(param, "sharded_dims_mp", [])
            if g is not None and comm.get_size(g) > 1
        )
        groups.setdefault(key, []).append(param.grad)
        param_map.setdefault(key, []).append(name)

    ord = 2 if norm_type == 2.0 else 1
    partials = []
    for mp_groups, grads in groups.items():
        # _foreach_norm: one fused kernel instead of one per tensor
        per_tensor_norms = torch._foreach_norm(grads, ord=ord)
        if norm_type == 2.0:
            partial = torch.stack(per_tensor_norms).pow(2).sum()
        else:
            partial = torch.stack(per_tensor_norms).sum()

        # one all_reduce per process group, not per parameter
        for g in mp_groups:
            dist.all_reduce(partial, group=comm.get_group(g))

        if verbose and torch.any(torch.isnan(partial)):
            for name, norm in zip(param_map[mp_groups], per_tensor_norms):
                if torch.isnan(norm):
                    print(f"Gradient norm is NaN for parameter {name}")

        partials.append(partial)

    if partials:
        total_gnorm = torch.stack(partials).sum()
    else:
        total_gnorm = torch.tensor(0.0)

    if norm_type == 2.0:
        total_gnorm = total_gnorm.sqrt()

    return total_gnorm


def clip_grads(model, max_grad_norm, norm_type=2.0, verbose=False):
    with torch.no_grad():
        total_gnorm = _compute_total_grad_norm(model, norm_type=norm_type, verbose=verbose)

        clip_factor = torch.clamp(max_grad_norm / (total_gnorm + 1e-6), max=1.0)

        # skip scaling entirely when the norm is already within bounds
        if clip_factor < 1.0:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            # _foreach_mul_: one fused kernel instead of one per tensor
            torch._foreach_mul_(grads, clip_factor)

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
