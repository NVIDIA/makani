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

"""Activation checkpointing that keeps the FP8 layers in FP8 in their recompute."""

from torch.utils.checkpoint import checkpoint as torch_checkpoint

from makani.utils.precision import fp8_active
from makani.utils.te_helpers import get_te


def checkpoint(function, *args, use_reentrant: bool = False, **kwargs):
    r"""
    ``torch.utils.checkpoint.checkpoint``, or transformer_engine's inside an enabled FP8 context.

    The recompute runs in the backward pass, outside the ``te.fp8_autocast`` of the forward, so with torch's
    checkpoint the FP8 layers would recompute in bf16 and save other tensors than in the forward, which torch
    refuses. transformer_engine's checkpoint recomputes in the FP8 context of the forward, and under delayed scaling
    with the scales of the forward and without recording the amaxes a second time: the gradients and the scaling
    state are those without checkpointing. It always restores the RNG states of the forward, so
    ``preserve_rng_state`` is dropped on that path; ``context_fn`` is honored by both. A checkpoint nested in one of
    transformer_engine's runs its function directly, see below.
    """
    if fp8_active():
        kwargs.pop("preserve_rng_state", None)
        if _inside_te_checkpoint() and "context_fn" not in kwargs:
            # nested in a checkpointed region, whose recompute covers this one already: transformer_engine keeps the
            # scales of the forward for the recompute in a queue per layer, which a nested checkpoint pushes and
            # pops out of order, so that the recompute would quantize with other scales than the forward
            return function(*args, **kwargs)
        return get_te().checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
    return torch_checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)


def _inside_te_checkpoint() -> bool:
    """Whether the code runs in the forward or the recompute of a transformer_engine checkpoint."""
    from transformer_engine.pytorch.distributed import (
        in_fp8_activation_recompute_phase,
        is_fp8_activation_recompute_enabled,
    )

    return is_fp8_activation_recompute_enabled() or in_fp8_activation_recompute_phase()
