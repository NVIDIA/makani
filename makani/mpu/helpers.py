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

from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch_harmonics.distributed.primitives import _gather
from makani.utils import comm


# How the gradient of a parameter is reduced over a model-parallel group it is shared over, see get_sharing.
# SUM: the ranks split the computation, e.g. over the grid, and each holds the contribution of its part.
# AVG: the ranks replicate the computation, e.g. a layer that is not split over matmul, and each holds all of it;
# averaging rather than leaving it alone keeps the replicas identical, whatever rounding a low precision brings.
SUM = "sum"
AVG = "avg"

# the model-parallel groups, and the names that stand for several of them
MODEL_GROUPS = ("h", "w", "matmul")
_COMPOSITE_GROUPS = {"spatial": ("h", "w"), "model": MODEL_GROUPS}

# a parameter without an annotation is shared over the whole model: split over the grid, replicated over matmul
DEFAULT_SHARING = {"spatial": SUM, "matmul": AVG}

# the reduction a group of the earlier list form of the annotation stands for
_LIST_REDUCTIONS = {"h": SUM, "w": SUM, "matmul": AVG}


def _expand_group(group: str) -> Tuple[str, ...]:
    if group in _COMPOSITE_GROUPS:
        return _COMPOSITE_GROUPS[group]
    if group in MODEL_GROUPS:
        return (group,)
    raise ValueError(
        f"Unknown model-parallel group {group!r}, expected one of {MODEL_GROUPS + tuple(_COMPOSITE_GROUPS)}"
    )


def get_sharing(param) -> Dict[str, str]:
    r"""
    The model-parallel groups a parameter is shared over, and how its gradient is reduced over each.

    A parameter is annotated with ``is_shared_mp``, a dict from group to reduction, e.g.
    ``{"spatial": "sum", "matmul": "avg"}`` for a pointwise layer, which every spatial rank
    applies to its part of the grid and every matmul rank to the same input. The groups are
    ``"h"``, ``"w"`` and ``"matmul"``, or ``"spatial"`` for ``h`` and ``w`` and ``"model"`` for
    all three. The reductions are :data:`SUM`, over ranks that split the computation, and
    :data:`AVG`, over ranks that replicate it. A group the parameter is sharded over, see
    ``sharded_dims_mp``, is not one it is shared over.

    A parameter without an annotation is shared as :data:`DEFAULT_SHARING`. The list form of
    earlier versions, e.g. ``["spatial"]``, is read with a sum over ``h`` and ``w`` and an
    average over ``matmul``.

    Parameters
    ----------
    param : torch.nn.Parameter
        The parameter.

    Returns
    -------
    dict
        The reduction over each of ``"h"``, ``"w"`` and ``"matmul"`` the parameter is shared over.

    Raises
    ------
    ValueError
        If the annotation names an unknown group or reduction, two reductions for one group, or a
        group the parameter is sharded over.
    """
    annotation = getattr(param, "is_shared_mp", DEFAULT_SHARING)
    if not isinstance(annotation, dict):
        annotation = {group: None for group in annotation}

    sharing = {}
    for group, reduction in annotation.items():
        for atomic in _expand_group(group):
            op = _LIST_REDUCTIONS[atomic] if reduction is None else reduction
            if op not in (SUM, AVG):
                raise ValueError(f"Unknown reduction {op!r} over {group!r}, expected {SUM!r} or {AVG!r}")
            if sharing.get(atomic, op) != op:
                raise ValueError(f"Two reductions over {atomic!r} in the annotation {annotation}")
            sharing[atomic] = op

    sharded = {group for group in getattr(param, "sharded_dims_mp", []) if group is not None}
    overlap = set(sharing) & sharded
    if overlap:
        raise ValueError(f"The parameter is sharded over {sorted(overlap)}, so it cannot be shared over them as well")

    return sharing


def get_reductions(param) -> List[Tuple[str, str]]:
    r"""
    The reductions of the gradient of a parameter, as ``(group, reduction)`` collectives.

    Those of :func:`get_sharing`, with ``h`` and ``w`` reduced together over ``"spatial"`` and
    all three over ``"model"`` where they have the same reduction.
    """
    sharing = get_sharing(param)
    if (len(sharing) == len(MODEL_GROUPS)) and (len(set(sharing.values())) == 1):
        return [("model", sharing["h"])]

    reductions = []
    if ("h" in sharing) and (sharing.get("w") == sharing["h"]):
        reductions.append(("spatial", sharing["h"]))
    else:
        reductions += [(group, sharing[group]) for group in ("h", "w") if group in sharing]
    if "matmul" in sharing:
        reductions.append(("matmul", sharing["matmul"]))
    return reductions


@torch.compiler.disable
def gather_uneven(tensor, dim, comm_name):
    # nothing to gather, and inside comm.serial() the group is None, which would gather over the world
    if comm.get_size(comm_name) == 1:
        return tensor

    # the shards may differ in size along dim; torch-harmonics' gather exchanges the sizes first
    # and pads the shards to the largest on gloo, which only gathers tensors of equal shape
    return _gather(tensor.contiguous(), dim, None, group=comm.get_group(comm_name))


def all_reduce_mean(tensor, group):
    """In-place all-reduce to the mean over ``group``. gloo has no ``ReduceOp.AVG``, so it sums and divides there."""
    if dist.get_backend(group) == dist.Backend.GLOO:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=False)
        tensor.div_(dist.get_world_size(group=group))
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=group, async_op=False)
    return tensor


def barrier(device):
    """Barrier across all ranks, if torch.distributed is initialized.

    NCCL is given the device explicitly rather than inferring it from the ambient CUDA state;
    on CPU there is no device index to give.
    """
    if not dist.is_initialized():
        return
    device_ids = [device.index] if device.type == "cuda" else None
    dist.barrier(device_ids=device_ids)


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
                dist.broadcast(
                    param_real, src=comm.get_root(comm_group), group=comm.get_group(comm_group), async_op=False
                )
                # use weight of rank 0
                # important to use copy here otherwise the handle gets detaches from the optimizer
                if is_complex:
                    param.copy_(torch.view_as_complex(param_real))
                else:
                    param.copy_(param_real)
            elif mode == "mean":
                param_real = torch.view_as_real(param) if param.is_complex() else param
                all_reduce_mean(param_real, comm.get_group(comm_group))
            else:
                raise ValueError(f"Unknown weight synchronization mode {mode}")

        return

    with torch.no_grad():
        # distributed sync step
        for param in model.parameters():
            # share along data dim
            _sync_param(param, "data", mode)

            # every group the parameter is shared over, whichever its gradient is reduced with
            for comm_group in get_sharing(param):
                _sync_param(param, comm_group, mode)

    # synchronize the device to make sure all copies have finished
    barrier(next(model.parameters()).device)

    return
