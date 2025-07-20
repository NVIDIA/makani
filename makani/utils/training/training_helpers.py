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


def clip_grads(model, max_grad_norm):

    # iterate over parameters
    with torch.no_grad():
        for param in model.parameters():

            if param.grad is None:
                continue

            # compute local norm: compute abs first to support complex grads
            gnorm = torch.sum(torch.square(torch.abs(param.grad)))

            # compute global norm
            if hasattr(param, "sharded_dims_mp"):

                for d, group in enumerate(param.sharded_dims_mp):
                    # continue if there is nothing to do
                    if (group is None) or (comm.get_size(group) == 1):
                        continue

                    dist.all_reduce(gnorm, group=comm.get_group(group))

            # compute square root
            gnorm = torch.sqrt(gnorm)

            # update grads
            if gnorm > max_grad_norm:
                param.grad.mul_(max_grad_norm / gnorm)

    return


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
