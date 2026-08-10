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
#
# Workaround for pytorch/pytorch#132388: `torch.amp.custom_fwd` /
# `torch.amp.custom_bwd` assume the legacy autograd.Function API where
# ``forward(ctx, *args)`` -- the decorators set ``_fwd_used_autocast`` /
# ``_dtype`` on ``args[0]`` (= ctx) in fwd and read them off ctx in bwd.
#
# Our autograd Functions use the new-style API with a separate
# ``setup_context(ctx, inputs, output)`` staticmethod. In that style ``args[0]``
# in fwd is the first input tensor, so the decorator's state lands on a tensor
# instead of ctx, and the bwd decorator AttributeErrors when it tries to read
# ``ctx._fwd_used_autocast``.
#
# This module provides ``_custom_setup_context`` -- the missing decorator the
# PyTorch maintainers (soulitzer in #132388) said they'd accept as the fix:
# it sets the same attributes on ctx at setup_context time, bridging the gap.
#
# Adapted from torch_harmonics.distributed._amp_utils. Remove this module and
# switch back to the stdlib once upstream lands ``torch.amp.custom_setup_context``.

import functools

import torch


def _custom_setup_context(setup_context_fn=None, *, device_type: str):
    """Bridge for new-style autograd.Function + @torch.amp.custom_fwd/custom_bwd.

    Records the current autocast state on ``ctx`` at setup_context time, so
    that the ``custom_bwd`` decorator on ``backward`` can find
    ``ctx._fwd_used_autocast`` / ``ctx._dtype`` and re-enter the same autocast
    context as ``forward`` was in.

    Usage::

        class MyFn(torch.autograd.Function):
            @staticmethod
            @torch.amp.custom_fwd(device_type="cuda")
            def forward(x, ...):
                ...

            @staticmethod
            @_custom_setup_context(device_type="cuda")
            def setup_context(ctx, inputs, output):
                ...

            @staticmethod
            @torch.amp.custom_bwd(device_type="cuda")
            def backward(ctx, grad_out):
                ...
    """
    if setup_context_fn is None:
        return functools.partial(_custom_setup_context, device_type=device_type)

    @functools.wraps(setup_context_fn)
    def decorate_setup_context(ctx, *args, **kwargs):
        ctx._dtype = torch.get_autocast_dtype(device_type)
        ctx._fwd_used_autocast = torch.is_autocast_enabled(device_type)
        return setup_context_fn(ctx, *args, **kwargs)

    return decorate_setup_context


def _cast_to_autocast_dtype(*tensors, device_type: str = "cuda"):
    """Cast floating-point Tensors to the active autocast dtype.

    Mirrors what PyTorch's built-in autocast-eligible ops (``F.linear``,
    ``F.conv2d``, ...) do internally: when autocast is enabled, transient
    copies of inputs are cast to the autocast dtype before the op runs. Use
    this in an orchestrator function immediately before calling ``.apply()``
    on a custom autograd.Function, so ``setup_context`` saves the cast
    tensors and the op runs in the autocast dtype throughout fwd and bwd.

    Non-Tensor arguments and non-floating-point Tensors are passed through
    unchanged. When autocast is not enabled, all arguments are passed
    through unchanged.

    Casting only creates transient copies -- parameter storage on the calling
    module is not mutated. Gradients flowing back to fp32 parameters get
    cast to the parameter's dtype by autograd's ``AccumulateGrad`` node, so
    optimizer state stays fp32 even when the op's computation ran in bf16.

    Usage::

        x, weight = _cast_to_autocast_dtype(x, weight)
        out = _MyFn.apply(x, weight, ...)

    Returns a tuple in the same order the arguments were passed.
    """
    if not torch.is_autocast_enabled(device_type):
        return tensors
    cast_dtype = torch.get_autocast_dtype(device_type)
    return tuple(t.to(cast_dtype) if (isinstance(t, torch.Tensor) and t.is_floating_point()) else t for t in tensors)
