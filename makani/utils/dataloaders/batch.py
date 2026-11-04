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

"""
The batch every dataloader emits: a dict instead of a positional tuple.

A tuple leaves its meaning to the position of each tensor, and that position
shifted with the options a loader was built with, so every consumer had to
repeat the unpacking rule. The dict names every tensor instead:

==========================  ==========================================  =================================
key                         content                                     shape
==========================  ==========================================  =================================
``input``                   predicted state at the input steps          ``(B, n_history+1, C_in, *grid)``
``target``                  predicted state at the target steps         ``(B, n_future+1, C_out, *grid)``
``input_dynamic_features``  dict of time dependent features that are    ``(B, n_history+1, C_f, *grid)``
                            not predicted, e.g. ``zenith``              per feature
``target_dynamic_features`` the same at the target steps                ``(B, n_future+1, C_f, *grid)``
``input_timestamp``         time of each input step, seconds            ``(B, n_history+1)``
``target_timestamp``        time of each target step, seconds           ``(B, n_future+1)``
==========================  ==========================================  =================================

``grid`` is ``(H, W)`` for a lat/lon grid and ``(N_points,)`` for a mesh: the
batch, time and channel dimensions always lead, the grid always trails, and
the grid itself is described by the loader's ``DataShapes``, not by the shape.
The mesh layout is the contract code consuming a batch is written against, but
no loader emits it yet: they still assume a two dimensional raster.

``grid`` is always this rank's part of the grid, after cropping, spatial
sharding and subsampling: ``(H_local, W_local)``, or the local points of a mesh
including the halo. Every tensor of a batch covers the same part, the zenith
angle included. Where that part sits in the global grid is described by
``DataShapes`` as well.

Every key is present, except that a loader built without targets, as for
inference, omits the three target keys. The dynamic features are a dict even
when empty. ``static_features`` (``(C_static, *grid)``) is reserved and not emitted
yet: the static fields are the same for every sample, so the preprocessor
builds them from the configuration and holds them on the device rather than
every batch carrying a copy. The key is kept for a structured model input that
takes the state window, the dynamic and the static features apart.

A :class:`~makani.utils.dataloaders.rollout.Rollout` consumes the dict.
"""

from typing import Any, Dict, Optional

import torch

INPUT = "input"
TARGET = "target"
INPUT_DYNAMIC_FEATURES = "input_dynamic_features"
TARGET_DYNAMIC_FEATURES = "target_dynamic_features"
# reserved, no loader emits it and nothing reads it yet: the static fields (orography, land mask, ...) are the same
# for every sample, so the preprocessor builds them from params (get_static_features) and holds them on the device.
# Kept for the structured model input, which takes the state window, the dynamic and the static features apart.
STATIC_FEATURES = "static_features"
INPUT_TIMESTAMP = "input_timestamp"
TARGET_TIMESTAMP = "target_timestamp"

# names of the dynamic features
ZENITH = "zenith"

Batch = Dict[str, Any]


def make_batch(
    inp: torch.Tensor,
    inp_timestamp: torch.Tensor,
    tar: Optional[torch.Tensor] = None,
    tar_timestamp: Optional[torch.Tensor] = None,
    inp_dynamic_features: Optional[Dict[str, torch.Tensor]] = None,
    tar_dynamic_features: Optional[Dict[str, torch.Tensor]] = None,
) -> Batch:
    """
    Assemble a batch, or a single sample, from its parts.

    Parameters
    ----------
    inp : torch.Tensor
        Input state.
    inp_timestamp : torch.Tensor
        Time of each input step.
    tar : torch.Tensor, optional
        Target state. Without it, the batch has no target keys.
    tar_timestamp : torch.Tensor, optional
        Time of each target step, required with ``tar``.
    inp_dynamic_features, tar_dynamic_features : dict of str to torch.Tensor, optional
        The dynamic features at the input and target steps by name, e.g.
        :data:`ZENITH`, each with the time axis of its state. None without
        features; the target features need a target.

    Returns
    -------
    dict
        The batch, see the module docstring.
    """

    if (tar is None) != (tar_timestamp is None):
        raise ValueError("A target and its timestamps go together")
    if (tar is None) and tar_dynamic_features:
        raise ValueError("Target dynamic features need a target")

    batch = {INPUT: inp, INPUT_DYNAMIC_FEATURES: dict(inp_dynamic_features or {}), INPUT_TIMESTAMP: inp_timestamp}
    if tar is not None:
        batch[TARGET] = tar
        batch[TARGET_DYNAMIC_FEATURES] = dict(tar_dynamic_features or {})
        batch[TARGET_TIMESTAMP] = tar_timestamp

    return batch


def without_target(batch: Batch) -> Batch:
    """
    The batch without its target keys, e.g. so that they are not moved to the device.

    Parameters
    ----------
    batch : dict
        The batch.

    Returns
    -------
    dict
        A new batch with the input keys only; the one passed in is left as it is.
    """
    return {
        key: value for key, value in batch.items() if key not in (TARGET, TARGET_DYNAMIC_FEATURES, TARGET_TIMESTAMP)
    }


def batch_nbytes(batch: Batch) -> int:
    """
    The bytes of every tensor of a batch, including those of nested dicts.

    Parameters
    ----------
    batch : dict
        The batch.

    Returns
    -------
    int
        The sum of ``Tensor.nbytes`` over the batch.
    """

    def _nbytes(value):
        if isinstance(value, torch.Tensor):
            return value.nbytes
        if isinstance(value, dict):
            return sum(_nbytes(item) for item in value.values())
        return 0

    return _nbytes(batch)


def batch_to_device(batch: Batch, device: torch.device, non_blocking: bool = False) -> Batch:
    """
    Move every tensor of a batch to ``device``, including those of nested dicts.

    Parameters
    ----------
    batch : dict
        The batch.
    device : torch.device
        Where to move it.
    non_blocking : bool, optional
        Passed on to ``Tensor.to``.

    Returns
    -------
    dict
        A new batch; the one passed in is left as it is.
    """

    def _move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=non_blocking)
        if isinstance(value, dict):
            return {key: _move(item) for key, item in value.items()}
        return value

    return _move(batch)
