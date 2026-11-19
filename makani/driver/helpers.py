# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Helpers the driver shares with the trainers and the inferencer: optimizer parameter groups and memory usage."""

import torch
from torch import nn


# standard torch normalization module base classes whose affine parameters
# (gain / shift) should be excluded from weight decay
_NORM_MODULE_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.modules.batchnorm._BatchNorm,
    nn.modules.instancenorm._InstanceNorm,
)


def _is_norm_module(module: nn.Module) -> bool:
    """True for standard torch norms and makani's custom normalization layers.

    The class-name fallback catches the custom S2 / distributed norms
    (GeometricInstanceNormS2, DistributedInstanceNorm2d, DistributedGeometricInstanceNormS2,
    DistributedLayerNorm) without importing them (which would risk circular imports).
    """
    return isinstance(module, _NORM_MODULE_TYPES) or ("norm" in type(module).__name__.lower())


def get_parameter_groups(model: nn.Module, weight_decay: float, mode: str = "full") -> list:
    """Build optimizer parameter groups according to the weight-decay mode.

    - ``"full"``: a single group with ``weight_decay`` applied to every trainable
      parameter (the historical behavior).
    - ``"transformer"``: two groups -- ``weight_decay`` on the weight matrices and
      ``weight_decay = 0`` on biases and normalization-layer affine parameters, the
      standard modern-transformer convention. Biases are matched by name (robust to
      expanded ``(1, C, 1, 1)`` bias shapes), norm affine params by their owning
      module type, plus a 1-D safety net for any stray scale/gain.

    Any other ``mode`` raises ``ValueError``.
    """
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    if mode == "full":
        return [{"params": [p for _, p in trainable], "weight_decay": weight_decay}]

    if mode != "transformer":
        raise ValueError(f"Unknown weight_decay_mode '{mode}', expected 'full' or 'transformer'.")

    # parameters owned by normalization modules (their affine gain / shift)
    norm_param_ids = set()
    for module in model.modules():
        if _is_norm_module(module):
            for p in module.parameters(recurse=False):
                norm_param_ids.add(id(p))

    decay, no_decay = [], []
    for name, p in trainable:
        leaf = name.rsplit(".", 1)[-1]
        if ("bias" in leaf) or (id(p) in norm_param_ids) or (p.ndim < 2):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def get_memory_usage(device):
    # memory statistics are CUDA only; on the CPU there is nothing to report
    if torch.device(device).type != "cuda":
        return 0.0, 0.0

    free_mem, total_mem = torch.cuda.mem_get_info(device=device)
    allocated_mem_gb = (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0)
    torch_mem_gb = torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0 * 1024.0)

    return allocated_mem_gb, torch_mem_gb
