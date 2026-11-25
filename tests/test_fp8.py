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
FP8 1x1 convolutions with transformer_engine. The tests that run FP8 are skipped without transformer_engine or
without a GPU that supports FP8; those that only build the layers need transformer_engine.
"""

import os
import sys
import unittest
from parameterized import parameterized

import torch

from makani.models import model_registry
from makani.models.common.mlp import EncoderDecoder, MLP
from makani.models.common.pointwise_conv import PointwiseConv2d, pointwise_conv2d
from makani.utils.precision import AutocastManager, fp8_active, fp8_disabled, _fp8_global_state_manager
from makani.utils.te_helpers import TE_AVAILABLE, get_te

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, get_default_parameters, single_step_rollout


def _fp8_support():
    """``(available, reason)`` of FP8 compute on this machine."""
    if not TE_AVAILABLE:
        return False, "transformer_engine is not installed"
    if not torch.cuda.is_available():
        return False, "FP8 needs a GPU"
    available = _fp8_global_state_manager().is_fp8_available()
    return available if isinstance(available, tuple) else (available, "FP8 is not available on this GPU")


def _mxfp8_support():
    available, reason = _fp8_support()
    if not available:
        return available, reason
    manager = _fp8_global_state_manager()
    if not hasattr(manager, "is_mxfp8_available"):
        return False, "this transformer_engine has no MXFP8"
    available = manager.is_mxfp8_available()
    return available if isinstance(available, tuple) else (available, "MXFP8 is not available on this GPU")


_FP8, _FP8_REASON = _fp8_support()
_MXFP8, _MXFP8_REASON = _mxfp8_support()
requires_te = unittest.skipUnless(TE_AVAILABLE, "transformer_engine is not installed")
requires_fp8 = unittest.skipUnless(_FP8, _FP8_REASON)


def _model_params(amp_mode, nettype="SFNO"):
    params = get_default_parameters()
    params.history_normalization_mode = "none"
    params.img_shape_x, params.img_shape_y = 36, 72
    params.img_local_shape_x = params.img_crop_shape_x = params.img_shape_x
    params.img_local_shape_y = params.img_crop_shape_y = params.img_shape_y
    params.img_shape_x_resampled, params.img_shape_y_resampled = params.img_shape_x, params.img_shape_y
    params.img_local_offset_x = params.img_local_offset_y = 0
    params.img_crop_offset_x = params.img_crop_offset_y = 0
    params.batch_size = 2
    params.nettype = nettype
    # the internal grid on the input grid, whose 36 x 72 points are divisible by 16 as FP8 needs
    params.scale_factor = 1
    params.amp_mode = amp_mode
    return params


def _engines(model):
    """The names of the layers that carry an FP8 engine."""
    return [name for name, module in model.named_modules() if getattr(module, "fp8_gemm", None) is not None]


class TestFP8WithoutTE(unittest.TestCase):

    @unittest.skipIf(TE_AVAILABLE, "transformer_engine is installed")
    def test_fp8_mode_needs_transformer_engine(self):
        """An FP8 precision mode refuses to build the model without transformer_engine, rather than run in bf16."""
        with self.assertRaises(RuntimeError):
            model_registry.get_model(_model_params("bf16-fp8_delayed"))


@requires_te
class TestFP8Layers(unittest.TestCase):
    """Building the layers needs transformer_engine, not FP8 hardware."""

    def test_amp_mode_selects_the_fp8_layers(self):
        """An FP8 amp_mode gives the blocks FP8 engines and leaves the encoder and decoder without; bf16 gives none."""
        self.assertEqual(_engines(model_registry.get_model(_model_params("bf16"))), [])

        engines = _engines(model_registry.get_model(_model_params("bf16-fp8_delayed")))
        self.assertTrue(any(".mlp.fc1" in name for name in engines), engines)
        self.assertTrue(any(".mlp.fc2" in name for name in engines), engines)
        self.assertFalse(any("encoder" in name or "decoder" in name for name in engines), engines)

    def test_ineligible_shapes_have_no_engine(self):
        """Groups and channels that FP8 GEMMs do not take keep the layer in the precision of the autocast."""
        self.assertIsNotNone(PointwiseConv2d(32, 64, fp8=True).fp8_gemm)
        self.assertIsNone(PointwiseConv2d(32, 73, fp8=True).fp8_gemm)
        self.assertIsNone(PointwiseConv2d(32, 64, groups=2, fp8=True).fp8_gemm)
        self.assertIsNone(PointwiseConv2d(32, 64, fp8=False).fp8_gemm)

    def test_checkpoints_load_across_modes(self):
        """A checkpoint with FP8 scaling state loads into a model without, and the other way round, strictly."""
        set_seed(333)
        fp8 = model_registry.get_model(_model_params("bf16-fp8_delayed"))
        plain = model_registry.get_model(_model_params("bf16"))
        self.assertTrue(any(k.endswith("fp8_gemm._extra_state") for k in fp8.state_dict()))

        plain.load_state_dict(fp8.state_dict(), strict=True)
        fp8.load_state_dict(plain.state_dict(), strict=True)
        for (name, a), (_, b) in zip(fp8.named_parameters(), plain.named_parameters()):
            self.assertTrue(torch.equal(a, b), name)


@requires_fp8
class TestFP8Compute(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.device = torch.device("cuda")

    def _compare_with_te_linear(self, amp_mode, batch_size, steps=3):
        """
        Run the NCHW layer and te.Linear on the same data and weights, each in its own FP8 context, over a few steps
        with growing inputs, so that the delayed scales move. Returns the largest differences and the two engines.
        """
        te = get_te()
        C, O, H, W = 64, 128, 16, 24
        layer = PointwiseConv2d(C, O, bias=False, fp8=True).to(self.device)
        reference = te.Linear(C, O, bias=False, params_dtype=torch.float32)
        with torch.no_grad():
            reference.weight.copy_(layer.weight.reshape(O, C))

        diffs = {"output": 0.0, "input grad": 0.0, "weight grad": 0.0}
        for step in range(steps):
            x = torch.randn(batch_size, C, H, W, device=self.device) * (1 + step)
            g = torch.randn(batch_size, O, H, W, device=self.device)
            xa = x.clone().requires_grad_(True)
            xb = x.permute(0, 2, 3, 1).contiguous().requires_grad_(True)
            layer.weight.grad = None
            reference.weight.grad = None

            with AutocastManager(amp_mode)():
                ya = layer(xa)
            ya.backward(g.to(ya.dtype))
            with AutocastManager(amp_mode)():
                yb = reference(xb)
            yb.backward(g.permute(0, 2, 3, 1).to(yb.dtype))

            pairs = {
                "output": (ya, yb.permute(0, 3, 1, 2)),
                "input grad": (xa.grad, xb.grad.permute(0, 3, 1, 2)),
                "weight grad": (layer.weight.grad.reshape(O, C), reference.weight.grad),
            }
            for name, (a, b) in pairs.items():
                rel = ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
                diffs[name] = max(diffs[name], rel)
        return diffs, layer.fp8_gemm, reference

    @parameterized.expand([[1], [2]], skip_on_empty=True)
    def test_delayed_scaling_matches_te_linear(self, batch_size):
        """Under delayed scaling, outputs and input gradients are those of te.Linear, and so are the amax histories
        and scales of both directions; the weight gradient, accumulated in fp32, within a bf16 rounding."""
        diffs, engine, reference = self._compare_with_te_linear("bf16-fp8_delayed", batch_size)
        self.assertEqual(diffs["output"], 0.0)
        self.assertEqual(diffs["input grad"], 0.0)
        self.assertLess(diffs["weight grad"], 1e-2)
        for direction in ["scaling_fwd", "scaling_bwd"]:
            with self.subTest(direction=direction):
                mine, theirs = engine.fp8_meta[direction], reference.fp8_meta[direction]
                self.assertTrue(torch.equal(mine.amax_history, theirs.amax_history))
                self.assertTrue(torch.equal(mine.scale, theirs.scale))
                self.assertGreater(mine.amax_history.abs().max().item(), 0.0)

    def test_current_scaling_matches_te_linear(self):
        """Under current scaling, one sample is quantized as te.Linear quantizes it."""
        diffs, _, _ = self._compare_with_te_linear("bf16-fp8_current", 1)
        self.assertEqual(diffs["output"], 0.0)
        self.assertEqual(diffs["input grad"], 0.0)
        self.assertLess(diffs["weight grad"], 1e-2)

    @unittest.skipUnless(_MXFP8, _MXFP8_REASON)
    def test_mxfp8_matches_te_linear(self):
        """Under MXFP8 block scaling, the layer computes what te.Linear computes."""
        diffs, _, _ = self._compare_with_te_linear("bf16-mxfp8", 1)
        self.assertLess(diffs["output"], 1e-2)
        self.assertLess(diffs["input grad"], 1e-2)
        self.assertLess(diffs["weight grad"], 1e-2)

    def test_fp8_disabled_computes_in_bf16(self):
        """Inside fp8_disabled, nested in the FP8 context, a layer with an engine computes what the bf16 layer does
        and leaves its scaling state alone."""
        layer = PointwiseConv2d(64, 128, fp8=True).to(self.device)
        x = torch.randn(2, 64, 16, 24, device=self.device)
        with AutocastManager("bf16-fp8_delayed")():
            self.assertTrue(fp8_active())
            with fp8_disabled():
                self.assertFalse(fp8_active())
                y = layer(x)
            self.assertTrue(fp8_active())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = pointwise_conv2d(x, layer.weight, layer.bias)
        self.assertTrue(torch.equal(y, ref))
        self.assertFalse(getattr(layer.fp8_gemm, "forwarded_at_least_once", False))

    def test_encoder_decoder_run_without_fp8(self):
        """The encoder and decoder compute without FP8 inside the FP8 context, whatever their layers are."""
        seen = []
        encoder = EncoderDecoder(num_layers=1, input_dim=32, output_dim=64, hidden_dim=64, act_layer=torch.nn.GELU).to(
            self.device
        )
        encoder.fwd[0].register_forward_pre_hook(lambda module, args: seen.append(fp8_active()))
        mlp = MLP(in_features=64, hidden_features=128).to(self.device)
        mlp.fc1.register_forward_pre_hook(lambda module, args: seen.append(fp8_active()))
        x = torch.randn(2, 32, 16, 24, device=self.device)
        with AutocastManager("bf16-fp8_delayed")():
            mlp(encoder(x))
        self.assertEqual(seen, [False, True])

    @parameterized.expand([["bf16-fp8_delayed"], ["bf16-fp8_current"]], skip_on_empty=True)
    def test_model_trains_in_fp8(self, amp_mode):
        """A training step of SFNO in FP8 runs its blocks in FP8, with finite gradients, and stays close to bf16."""
        set_seed(333)
        model = model_registry.get_model(_model_params(amp_mode)).to(self.device)
        set_seed(333)
        reference = model_registry.get_model(_model_params("bf16")).to(self.device)
        reference.load_state_dict(model.state_dict(), strict=True)

        params = _model_params(amp_mode)
        x = torch.randn(2, params.N_in_channels, 36, 72, device=self.device)
        outputs = []
        for net, mode in [(model, amp_mode), (reference, "bf16")]:
            with AutocastManager(mode)():
                out = net(single_step_rollout(x.unsqueeze(1)))
            out.float().square().mean().backward()
            outputs.append(out.float())

        self.assertTrue(torch.isfinite(outputs[0]).all())
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), name)
        # every GEMM of the blocks quantizes to E4M3, a few percent each; a broken FP8 path is off by order one
        rel = ((outputs[0] - outputs[1]).norm() / outputs[1].norm()).item()
        self.assertLess(rel, 0.25)

        # the blocks did compute in FP8: their engines ran
        used = [
            name
            for name, module in model.named_modules()
            if getattr(getattr(module, "fp8_gemm", None), "forwarded_at_least_once", False)
        ]
        self.assertTrue(any(".mlp.fc1" in name for name in used), used)

    @parameterized.expand(
        [["bf16-fp8_delayed", 2], ["bf16-fp8_delayed", 3], ["bf16-fp8_current", 2], ["bf16-fp8_current", 3]],
        skip_on_empty=True,
    )
    def test_activation_checkpointing_keeps_fp8(self, amp_mode, level):
        """Activation checkpointing recomputes the FP8 layers in FP8, with the scales of the forward: over a few
        steps, the gradients and the scaling state are those without checkpointing. Level 3 checkpoints the MLP
        inside the checkpointed block, a nested checkpoint."""
        results = []
        for checkpointing_level in [0, level]:
            params = _model_params(amp_mode)
            params.checkpointing_level = checkpointing_level
            set_seed(333)
            model = model_registry.get_model(params).to(self.device)
            set_seed(334)
            x = torch.randn(2, params.N_in_channels, 36, 72, device=self.device)
            for step in range(3):
                model.zero_grad()
                with AutocastManager(amp_mode)():
                    out = model(single_step_rollout((x * (1 + step)).unsqueeze(1)))
                out.float().square().mean().backward()
            grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            engines = [m.fp8_gemm for m in model.modules() if getattr(m, "fp8_gemm", None) is not None]
            results.append((grads, engines))

        (grads, engines), (grads_ckpt, engines_ckpt) = results
        self.assertEqual(set(grads), set(grads_ckpt))
        for name in grads:
            with self.subTest(param=name):
                self.assertTrue(torch.equal(grads[name], grads_ckpt[name]))
        if amp_mode == "bf16-fp8_delayed":
            for engine, engine_ckpt in zip(engines, engines_ckpt):
                for direction in ["scaling_fwd", "scaling_bwd"]:
                    meta, meta_ckpt = engine.fp8_meta[direction], engine_ckpt.fp8_meta[direction]
                    self.assertTrue(torch.equal(meta.scale, meta_ckpt.scale))
                    self.assertTrue(torch.equal(meta.amax_history, meta_ckpt.amax_history))

    def test_compiled_model_runs_fp8_layers(self):
        """A compiled model breaks its graph at the FP8 layers and computes what it computes eagerly."""
        torch._dynamo.reset()
        # without bias: inductor fuses the bias add of the bf16 output without rounding the bias to bf16 first, which
        # is its numerics for any bf16 add, not the FP8 layers'
        layers = torch.nn.Sequential(
            PointwiseConv2d(64, 128, bias=False, fp8=True),
            torch.nn.GELU(),
            PointwiseConv2d(128, 64, bias=False, fp8=True),
        ).to(self.device)
        x = torch.randn(2, 64, 16, 24, device=self.device)
        results = []
        for func in [layers, torch.compile(layers)]:
            # fresh scaling state for each run, so that both quantize alike
            for module in layers:
                if isinstance(module, PointwiseConv2d):
                    module.fp8_gemm = type(module.fp8_gemm)()
            with AutocastManager("bf16-fp8_current")():
                results.append(func(x))
            self.assertTrue(all(m.fp8_gemm.forwarded_at_least_once for m in layers if isinstance(m, PointwiseConv2d)))
        torch._dynamo.reset()
        self.assertTrue(torch.equal(results[0], results[1]))


if __name__ == "__main__":
    unittest.main()
