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


def get_no_weight_decay_parameters(model: nn.Module) -> dict:
    """The parameters the layers of ``model`` declare not to be decayed, by their names in ``model``.

    A module declares them through an optional method ``no_weight_decay()``, which returns the names of
    parameters relative to the module, as its ``named_parameters()`` gives them: normally parameters it owns
    directly, or those of a container without methods of its own, such as an ``nn.ParameterList``. A module
    without the method declares none, and the weight-decay mode decides on its parameters, see
    :func:`get_parameter_groups`.

    Parameters
    ----------
    model : nn.Module
        The model, whose modules are searched for declarations.

    Returns
    -------
    dict of str to nn.Parameter
        The declared parameters by their names in ``model``, each once, also when it is shared.

    Raises
    ------
    ValueError
        If a module declares a name that is not one of its parameters.
    """
    declared = {}
    seen = set()
    for prefix, module in model.named_modules():
        method = getattr(module, "no_weight_decay", None)
        if not callable(method):
            continue
        names = set(method() or ())
        if not names:
            continue
        parameters = dict(module.named_parameters())
        unknown = sorted(names - set(parameters))
        if unknown:
            raise ValueError(
                f"{type(module).__name__} at '{prefix or '<model>'}' declares {unknown} not to be weight decayed, but "
                f"has no such parameters"
            )
        for name in sorted(names):
            p = parameters[name]
            if id(p) not in seen:
                seen.add(id(p))
                declared[f"{prefix}.{name}" if prefix else name] = p
    return declared


def get_parameter_groups(model: nn.Module, weight_decay: float, mode: str = "full") -> list:
    """Build optimizer parameter groups according to the weight-decay mode.

    A parameter is decayed unless its layer declares it not to be, see :func:`get_no_weight_decay_parameters`, or
    the mode exempts it:

    - ``"full"``: no exemptions, every other trainable parameter is decayed (the historical behavior).
    - ``"transformer"``: biases and normalization-layer affine parameters are exempt as well, the standard
      modern-transformer convention. Biases are matched by name (robust to expanded ``(1, C, 1, 1)`` bias shapes),
      norm affine params by their owning module type, plus a 1-D safety net for any stray scale/gain.

    A declaration only exempts: it never makes the mode decay a parameter it would exempt. Any other ``mode``
    raises ``ValueError``.

    Returns
    -------
    list of dict
        A group with ``weight_decay`` and one with ``weight_decay = 0``, each only if it has parameters.
    """
    if mode not in ("full", "transformer"):
        raise ValueError(f"Unknown weight_decay_mode '{mode}', expected 'full' or 'transformer'.")

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    exempt_ids = {id(p) for p in get_no_weight_decay_parameters(model).values()}

    if mode == "transformer":
        # parameters owned by normalization modules (their affine gain / shift)
        for module in model.modules():
            if _is_norm_module(module):
                for p in module.parameters(recurse=False):
                    exempt_ids.add(id(p))

    decay, no_decay = [], []
    for name, p in trainable:
        leaf = name.rsplit(".", 1)[-1]
        if (id(p) in exempt_ids) or ((mode == "transformer") and (("bias" in leaf) or (p.ndim < 2))):
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
