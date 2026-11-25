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

"""1x1 convolutions computed as matmuls over the channels."""

from typing import Optional

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable


def _gemm_out_dtype_kwargs(a: torch.Tensor) -> dict:
    """``out_dtype=float32`` for half precision GEMMs on the GPU, so that the product is reduced and kept in fp32."""
    if a.is_cuda and a.dtype in (torch.float16, torch.bfloat16):
        return {"out_dtype": torch.float32}
    return {}


def _grouped_product(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``(G, M, K) @ (B, G, K, N) -> (B, G, M, N)``, contiguous, the weight broadcast over the batch."""
    B, G = x.shape[0], x.shape[1]
    if G == 1:
        if B == 1:
            return torch.mm(w[0], x[0, 0]).reshape(1, 1, *w.shape[1:2], x.shape[-1])
        # a stride-0 batch of the weight: one strided batched GEMM, without copying it
        return torch.bmm(w.expand(B, *w.shape[1:]), x[:, 0]).unsqueeze(1)
    return torch.matmul(w, x)


class _PointwiseConv2dFunction(torch.autograd.Function):
    """1x1 convolution as a GEMM, with the weight gradient reduced over batch and grid in fp32, as conv2d does."""

    @staticmethod
    def forward(ctx, x, weight, groups):
        B, C, H, W = x.shape
        O, G = weight.shape[0], groups
        w = weight.reshape(G, O // G, C // G)
        xg = x.reshape(B, G, C // G, H * W)
        ctx.save_for_backward(x, weight)
        ctx.groups = groups
        return _grouped_product(w, xg).reshape(B, O, H, W)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        B, C, H, W = x.shape
        O, G = weight.shape[0], ctx.groups
        w = weight.reshape(G, O // G, C // G)
        gg = grad.reshape(B, G, O // G, H * W)
        grad_x = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_x = _grouped_product(w.transpose(-1, -2), gg).reshape(B, C, H, W)

        if ctx.needs_input_grad[1]:
            # one GEMM per sample and group, each over the whole grid: a single GEMM with K = H * W in half
            # precision splits K and reduces the parts in that precision, so the products are kept in fp32, summed
            # over the batch in fp32 and rounded once
            xg = x.reshape(B * G, C // G, H * W)
            dw = torch.bmm(gg.reshape(B * G, O // G, H * W), xg.transpose(-1, -2), **_gemm_out_dtype_kwargs(xg))
            grad_weight = dw.reshape(B, G, O // G, C // G).sum(dim=0).reshape(weight.shape).to(weight.dtype)

        return grad_x, grad_weight, None


def pointwise_conv2d(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, groups: int = 1
) -> torch.Tensor:
    r"""
    ``F.conv2d(x, weight, bias, groups=groups)`` for a 1x1 ``weight``, computed as a matmul over the channels.

    Under ``torch.compile``, a convolution makes inductor's layout optimization switch the graph to channels-last,
    and the gradient of the input then leaves the compiled backward channels-last; every compiled region that
    receives it has to copy it back into the layout it traced, a full transposing copy of the activation. A matmul
    keeps the gradient in the layout of the input.

    The forward is one GEMM on the ``(B, C, H * W)`` view, ``mm`` for a single sample and ``bmm`` with the weight
    broadcast over the batch otherwise, whose output is contiguous NCHW. The weight gradient is reduced over the
    batch and the grid in fp32 and rounded once, as conv2d does: a half precision GEMM over a grid of a few hundred
    thousand points otherwise splits the reduction and adds up the parts in half precision, up to four times less
    accurate for few input channels. Under autocast the input and the weight are cast to the autocast dtype, as
    conv2d does. Not twice differentiable.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape ``(B, C, H, W)``, or ``(C, H, W)`` without a batch, as conv2d takes.
    weight : torch.Tensor
        Weight of shape ``(O, C / groups)`` or ``(O, C / groups, 1, 1)``.
    bias : torch.Tensor, optional
        Bias of shape ``(O,)``, added in the dtype of the product, as conv2d does under autocast.
    groups : int, optional
        Number of groups, each a matmul of its own channels.

    Returns
    -------
    torch.Tensor
        Output of shape ``(B, O, H, W)``, or ``(O, H, W)`` without a batch.
    """
    if x.dim() == 3:
        return pointwise_conv2d(x.unsqueeze(0), weight, bias, groups).squeeze(0)

    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
        x, weight = x.to(dtype), weight.to(dtype)
    with torch.autocast(device_type=device_type, enabled=False):
        y = _PointwiseConv2dFunction.apply(x, weight, groups)
    if bias is not None:
        # in the dtype of the product, as conv2d does under autocast: an fp32 bias would promote the output
        y = y + bias.to(y.dtype).reshape(-1, 1, 1)
    return y


class PointwiseConv2d(nn.Conv2d):
    r"""
    A 1x1 convolution on ``(B, C, H, W)`` computed as a matmul over the channels, see :func:`pointwise_conv2d`.

    Same parameters, initialization and state dict as ``nn.Conv2d(in_channels, out_channels, 1, groups=groups)``,
    and an instance of it, so initializations that look for ``nn.Conv2d`` find it.
    """

    def __init__(self, in_channels, out_channels, bias=True, groups=1, device=None, dtype=None):
        super().__init__(in_channels, out_channels, 1, bias=bias, groups=groups, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pointwise_conv2d(x, self.weight, self.bias, self.groups)
