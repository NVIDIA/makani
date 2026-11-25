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

r"""
FP8 GEMMs of the 1x1 convolutions with transformer_engine, on the NCHW layout.

``te.Linear`` computes ``x W^T`` with the channels last, so a 1x1 convolution on NCHW would have to be permuted to
channels-last and back around it, a transposing copy of the activation in each direction. On the ``(C, H * W)`` view
of a sample, the three GEMMs of the layer are instead the ones transformer_engine runs in its layouts, with the roles
of the operands swapped:

=========  ======================  =======================================================
GEMM       product                 ``general_gemm``
=========  ======================  =======================================================
forward    ``y = W x``             ``A = x, B = W``, layout ``"NN"``
dgrad      ``dx = W^T dy``         ``A = dy, B = W``, layout ``"NT"``
wgrad      ``dW = dy x^T``         ``A = x, B = dy``, layout ``"TN"``, accumulated in fp32
=========  ======================  =======================================================

The quantizers of the active recipe (delayed scaling, current scaling, MXFP8) make whichever copy of an operand a
GEMM needs, the transposed one fused into the cast, so FP8 costs no layout copy. Under delayed scaling the layer
matches ``te.Linear`` on the same data: the outputs and input gradients bit for bit, and the amax histories and
scales of both directions.

transformer_engine is optional: the engine class is defined on first use.
"""

import torch

from makani.utils.te_helpers import TE_AVAILABLE, get_te
from makani.utils.precision import _fp8_global_state_manager


def fp8_gemm_eligible(in_channels: int, out_channels: int, groups: int = 1) -> bool:
    """Whether a 1x1 convolution of this shape can run in FP8: no groups, and channels divisible by 16."""
    return groups == 1 and in_channels % 16 == 0 and out_channels % 16 == 0


def fp8_input_eligible(x: torch.Tensor) -> bool:
    """Whether the input of a 1x1 convolution can be quantized: an NCHW CUDA tensor whose grid is divisible by 16."""
    return x.is_cuda and x.dim() == 4 and (x.shape[-2] * x.shape[-1]) % 16 == 0


_ENGINE_CLASS = None


def make_fp8_gemm() -> torch.nn.Module:
    """A new :class:`TEPointwiseGemm`, the TE module that holds the FP8 state of one 1x1 convolution."""
    global _ENGINE_CLASS
    if not TE_AVAILABLE:
        raise RuntimeError("FP8 layers were requested, but transformer_engine is not installed")
    if _ENGINE_CLASS is None:
        _ENGINE_CLASS = _build_engine_class()
    return _ENGINE_CLASS()


def _import_tensor_indices():
    """The indices of the forward and backward tensors in a TE module's quantizers, wherever this TE keeps them."""
    for module in ["transformer_engine.pytorch.constants", "transformer_engine.pytorch.module.linear"]:
        try:
            mod = __import__(module, fromlist=["FP8FwdTensorIdx", "FP8BwdTensorIdx"])
            return mod.FP8FwdTensorIdx, mod.FP8BwdTensorIdx
        except (ImportError, AttributeError):
            continue
    import transformer_engine_torch as tex

    return tex.FP8FwdTensors, tex.FP8BwdTensors


def _build_engine_class():
    te = get_te()
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    FP8FwdTensorIdx, FP8BwdTensorIdx = _import_tensor_indices()
    FP8GlobalStateManager = _fp8_global_state_manager()

    try:
        from transformer_engine.pytorch.module.linear import _check_fp8_reduce_and_update
    except ImportError:
        _check_fp8_reduce_and_update = FP8GlobalStateManager.is_first_fp8_module

    class _PointwiseGemmFunction(torch.autograd.Function):
        """The three GEMMs of a 1x1 convolution on NCHW in FP8, one per sample; see the module docstring."""

        @staticmethod
        def forward(ctx, x, weight, input_quantizer, weight_quantizer, grad_output_quantizer, split_acc, dtype):
            B, C, H, W = x.shape
            O, N = weight.shape[0], H * W

            weight_quantizer.set_usage(rowwise=True, columnwise=True)
            input_quantizer.set_usage(rowwise=True, columnwise=True)
            weight_q = weight_quantizer(weight)
            xc = x.reshape(B, C, N)
            # one quantization per sample, each the (C, H * W) operand of its GEMM; under delayed scaling they share
            # the scale of the step and their amaxes are combined
            xs = [input_quantizer(xc[b]) for b in range(B)]

            y = torch.empty(B, O, N, dtype=dtype, device=x.device)
            for b in range(B):
                general_gemm(xs[b], weight_q, out_dtype=dtype, layout="NN", out=y[b])

            ctx.xs, ctx.weight_q, ctx.grad_output_quantizer = xs, weight_q, grad_output_quantizer
            ctx.shape, ctx.out_channels, ctx.dtype, ctx.weight_dtype = (B, C, H, W), O, dtype, weight.dtype
            ctx.split_acc = split_acc
            ctx.reduce_and_update_bwd = _check_fp8_reduce_and_update()
            return y.reshape(B, O, H, W)

        @staticmethod
        def backward(ctx, grad):
            B, C, H, W = ctx.shape
            O, N = ctx.out_channels, H * W
            dgrad_split_acc, wgrad_split_acc = ctx.split_acc

            ctx.grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
            gc = grad.reshape(B, O, N)
            gs = [ctx.grad_output_quantizer(gc[b]) for b in range(B)]

            grad_x = grad_weight = None
            if ctx.needs_input_grad[0]:
                grad_x = torch.empty(B, C, N, dtype=ctx.dtype, device=grad.device)
                for b in range(B):
                    general_gemm(
                        gs[b],
                        ctx.weight_q,
                        out_dtype=ctx.dtype,
                        layout="NT",
                        out=grad_x[b],
                        grad=True,
                        use_split_accumulator=dgrad_split_acc,
                    )
                grad_x = grad_x.reshape(B, C, H, W)
            if ctx.needs_input_grad[1]:
                # summed over the batch in fp32 and rounded once, as the bf16 layer does
                grad_weight = torch.zeros(O, C, dtype=torch.float32, device=grad.device)
                for b in range(B):
                    general_gemm(
                        ctx.xs[b],
                        gs[b],
                        out_dtype=torch.float32,
                        layout="TN",
                        out=grad_weight,
                        accumulate=True,
                        grad=True,
                        use_split_accumulator=wgrad_split_acc,
                    )
                grad_weight = grad_weight.to(ctx.weight_dtype)

            ctx.xs = ctx.weight_q = None
            if ctx.reduce_and_update_bwd:
                FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

            return grad_x, grad_weight, None, None, None, None, None

    class TEPointwiseGemm(TransformerEngineBaseModule):
        """
        The FP8 state of one 1x1 convolution: the scaling metadata of its input, weight and output gradient, and the
        GEMMs on NCHW. Holds no parameter; the layer that owns it passes its weight, so that parameters, their
        initialization and sharing stay those of the layer.
        """

        def __init__(self):
            super().__init__()
            self._weight = None

        def _get_weight_tensors(self):
            return [self._weight] if self._weight is not None else []

        def _get_weight_quantizers(self):
            if not self.fp8:
                return [None]
            quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
            quantizer.internal = True
            return [quantizer]

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, *args, **kwargs):
            # a checkpoint without FP8 layers has no scaling state for this one, which then starts afresh
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, *args, **kwargs)
            extra = prefix + "_extra_state"
            if extra in missing_keys:
                missing_keys.remove(extra)

        def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            """``conv2d(x, weight)`` for a 1x1 ``weight`` of shape ``(O, C)`` or ``(O, C, 1, 1)``, without bias."""
            # quantized from the input and the weight as they are, as te.Linear does; the output in the activation dtype,
            # that of the autocast
            weight = weight.reshape(weight.shape[0], -1)
            device_type = x.device.type

            self._weight = weight
            x = self.prepare_forward(x)
            try:
                input_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_INPUT]
                input_quantizer.internal = True
                (weight_quantizer,) = self._get_weight_quantizers()
                grad_output_quantizer = self.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_OUTPUT1]
                grad_output_quantizer.internal = True

                recipe = FP8GlobalStateManager.get_fp8_recipe()
                split_acc = (
                    getattr(getattr(recipe, "fp8_gemm_dgrad", None), "use_split_accumulator", True),
                    getattr(getattr(recipe, "fp8_gemm_wgrad", None), "use_split_accumulator", True),
                )
                with torch.autocast(device_type=device_type, enabled=False):
                    return _PointwiseGemmFunction.apply(
                        x,
                        weight,
                        input_quantizer,
                        weight_quantizer,
                        grad_output_quantizer,
                        split_acc,
                        self.activation_dtype,
                    )
            finally:
                self._weight = None
                self.end_forward()

    return TEPointwiseGemm
