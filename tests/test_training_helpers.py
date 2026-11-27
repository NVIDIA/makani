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

import unittest
import torch
import torch.nn as nn

from makani.training.training_helpers import clip_grads, _compute_total_grad_norm, normalize_weights, NonFiniteLossCheck

from makani.driver.helpers import get_parameter_groups, get_no_weight_decay_parameters

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import set_seed, disable_tf32


def _simple_model(*shapes):
    """Build a model whose parameters have the given shapes and synthetic grads."""
    params = nn.ParameterList([nn.Parameter(torch.zeros(*s)) for s in shapes])
    model = nn.Module()
    model.params = params
    return model


def _assign_grads(model, grads):
    for param, g in zip(model.parameters(), grads):
        param.grad = g.clone()


def _global_l2(grads):
    """Reference L2 norm of a flat list of grad tensors."""
    return torch.sqrt(sum(g.float().norm() ** 2 for g in grads))


class TestComputeTotalGradNorm(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_matches_reference_single_param(self):
        g = torch.randn(4, 8)
        model = _simple_model((4, 8))
        _assign_grads(model, [g])
        got = _compute_total_grad_norm(model)
        self.assertAlmostEqual(got.item(), _global_l2([g]).item(), places=5)

    def test_matches_reference_multiple_params(self):
        grads = [torch.randn(3, 5), torch.randn(7), torch.randn(2, 2, 2)]
        model = _simple_model((3, 5), (7,), (2, 2, 2))
        _assign_grads(model, grads)
        got = _compute_total_grad_norm(model)
        self.assertAlmostEqual(got.item(), _global_l2(grads).item(), places=5)

    def test_skips_params_with_no_grad(self):
        model = _simple_model((4,), (4,))
        params = list(model.parameters())
        params[0].grad = torch.ones(4)
        # params[1].grad left as None
        got = _compute_total_grad_norm(model)
        ref = _global_l2([params[0].grad])
        self.assertAlmostEqual(got.item(), ref.item(), places=5)

    def test_all_zero_grads(self):
        grads = [torch.zeros(3, 3), torch.zeros(5)]
        model = _simple_model((3, 3), (5,))
        _assign_grads(model, grads)
        got = _compute_total_grad_norm(model)
        self.assertAlmostEqual(got.item(), 0.0, places=6)

    def test_no_grads_returns_zero(self):
        model = _simple_model((4,), (4,))
        # no grads assigned at all
        got = _compute_total_grad_norm(model)
        self.assertAlmostEqual(got.item(), 0.0, places=6)

    def test_complex_grads_supported(self):
        g = torch.randn(4, 4, dtype=torch.complex64)
        model = nn.Module()
        model.p = nn.Parameter(torch.zeros(4, 4, dtype=torch.complex64))
        model.p.grad = g.clone()
        got = _compute_total_grad_norm(model)
        ref = _global_l2([g.abs()])
        self.assertAlmostEqual(got.item(), ref.item(), places=4)


class TestClipGrads(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_returned_norm_matches_compute(self):
        grads = [torch.randn(4, 4), torch.randn(8)]
        model = _simple_model((4, 4), (8,))
        _assign_grads(model, grads)
        returned = clip_grads(model, max_grad_norm=100.0)
        self.assertAlmostEqual(returned.item(), _global_l2(grads).item(), places=5)

    def test_no_clip_when_norm_within_bound(self):
        grads = [torch.randn(3, 3)]
        model = _simple_model((3, 3))
        _assign_grads(model, grads)
        original_grad = list(model.parameters())[0].grad.clone()
        clip_grads(model, max_grad_norm=1e6)
        # grads should be unchanged
        self.assertTrue(torch.allclose(list(model.parameters())[0].grad, original_grad))

    def test_clip_reduces_norm_to_max(self):
        grads = [torch.ones(10) * 10.0]  # large norm
        model = _simple_model((10,))
        _assign_grads(model, grads)
        max_norm = 1.0
        clip_grads(model, max_grad_norm=max_norm)
        clipped_norm = _compute_total_grad_norm(model)
        self.assertLessEqual(clipped_norm.item(), max_norm + 1e-5)

    def test_clip_preserves_direction(self):
        g = torch.randn(5, 5)
        model = _simple_model((5, 5))
        _assign_grads(model, [g])
        clip_grads(model, max_grad_norm=0.1)
        clipped = list(model.parameters())[0].grad
        # direction preserved: clipped grad is a positive scalar multiple of original
        ratio = clipped / g
        self.assertTrue(torch.allclose(ratio, ratio[0, 0].expand_as(ratio), atol=1e-5))
        self.assertGreater(ratio[0, 0].item(), 0.0)

    def test_grads_without_grad_unaffected(self):
        model = _simple_model((4,), (4,))
        params = list(model.parameters())
        params[0].grad = torch.ones(4) * 100.0
        # params[1].grad is None — should not raise
        clip_grads(model, max_grad_norm=0.01)
        self.assertIsNone(params[1].grad)

    def test_clip_with_multiple_params(self):
        grads = [torch.randn(4, 4) * 5, torch.randn(8) * 5]
        model = _simple_model((4, 4), (8,))
        _assign_grads(model, grads)
        max_norm = 1.0
        clip_grads(model, max_grad_norm=max_norm)
        clipped_norm = _compute_total_grad_norm(model)
        self.assertLessEqual(clipped_norm.item(), max_norm + 1e-5)


class TestNormalizeWeights(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _param_l2(self, param):
        return param.data.float().norm().item()

    def test_single_param_has_unit_norm_after(self):
        model = _simple_model((4, 8))
        with torch.no_grad():
            list(model.parameters())[0].fill_(2.0)
        normalize_weights(model)
        self.assertAlmostEqual(self._param_l2(list(model.parameters())[0]), 1.0, places=5)

    def test_multiple_params_each_have_unit_norm(self):
        model = _simple_model((3, 4), (5,), (2, 2))
        with torch.no_grad():
            for i, p in enumerate(model.parameters()):
                p.fill_(float(i + 1))
        normalize_weights(model)
        for p in model.parameters():
            self.assertAlmostEqual(self._param_l2(p), 1.0, places=5)

    def test_zero_param_not_nan(self):
        model = _simple_model((4,))
        with torch.no_grad():
            list(model.parameters())[0].zero_()
        normalize_weights(model, eps=1e-5)
        p = list(model.parameters())[0]
        self.assertFalse(torch.isnan(p).any())

    def test_direction_preserved(self):
        model = _simple_model((4,))
        p = list(model.parameters())[0]
        with torch.no_grad():
            p.copy_(torch.tensor([3.0, 4.0, 1.0, 2.0]))
        original = p.data.clone()
        normalize_weights(model)
        # after normalization the parameter should be a positive scalar multiple
        # of the original — compare unit vectors to avoid dividing by zero
        unit_before = original / original.norm()
        unit_after = p.data / p.data.norm()
        self.assertTrue(torch.allclose(unit_after, unit_before, atol=1e-5))

    def test_complex_params_supported(self):
        model = nn.Module()
        model.p = nn.Parameter(torch.randn(4, 4, dtype=torch.complex64) * 3.0)
        normalize_weights(model)
        norm = model.p.data.abs().norm().item()
        self.assertAlmostEqual(norm, 1.0, places=4)


class _WeirdNorm(nn.Module):
    """A norm-like module whose gain is 2-D and named 'weight'. Only module-type
    detection (not the name or ndim rules) can classify its gain as no-decay -- this
    proves the norm detection works beyond the simple heuristics."""

    def __init__(self, n=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n, n))
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return x


class _ParamGroupNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)  # weight(2D)->decay, bias(1D)->no_decay
        self.conv = nn.Conv2d(4, 4, 1)  # weight(4D)->decay, bias(1D)->no_decay
        self.ln = nn.LayerNorm(8)  # weight/bias -> no_decay (module type)
        self.wnorm = _WeirdNorm(4)  # 2D gain -> no_decay only via module type
        self.expanded_bias = nn.Parameter(torch.zeros(1, 4, 1, 1))  # 4D bias -> no_decay BY NAME
        self.plain_weight = nn.Parameter(torch.randn(8, 8))  # 2D weight -> decay

    def forward(self, x):
        return x


class TestParameterGroups(unittest.TestCase):
    def setUp(self):
        set_seed(0)
        self.model = _ParamGroupNet()
        self.wd = 0.1

    def _named(self):
        return dict(self.model.named_parameters())

    def test_full_mode_is_single_group(self):
        groups = get_parameter_groups(self.model, self.wd, "full")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["weight_decay"], self.wd)
        self.assertEqual(len(groups[0]["params"]), sum(1 for _ in self.model.parameters()))

    def test_transformer_mode_splits_norms_and_biases(self):
        groups = get_parameter_groups(self.model, self.wd, "transformer")
        self.assertEqual(len(groups), 2)

        decay = next(g for g in groups if g["weight_decay"] == self.wd)
        no_decay = next(g for g in groups if g["weight_decay"] == 0.0)
        decay_ids = {id(p) for p in decay["params"]}
        no_decay_ids = {id(p) for p in no_decay["params"]}

        named = self._named()

        # weight matrices keep weight decay
        for nm in ("lin.weight", "conv.weight", "plain_weight"):
            self.assertIn(id(named[nm]), decay_ids, f"{nm} should be in the decay group")

        # biases (incl. the 4-D expanded one) and all norm affine params (incl. the
        # 2-D weird-norm gain) are excluded from weight decay
        for nm in ("lin.bias", "conv.bias", "expanded_bias", "ln.weight", "ln.bias", "wnorm.weight", "wnorm.bias"):
            self.assertIn(id(named[nm]), no_decay_ids, f"{nm} should be in the no-decay group")

        # partition is exact: every param classified once, no overlap
        total = sum(1 for _ in self.model.parameters())
        self.assertEqual(len(decay_ids) + len(no_decay_ids), total)
        self.assertEqual(decay_ids & no_decay_ids, set())

    def test_unknown_mode_raises(self):
        for bad in ("selective", "Transformer", "none", ""):
            with self.assertRaises(ValueError):
                get_parameter_groups(self.model, self.wd, bad)


class _Embedding(nn.Module):
    """A layer that declares its embedding, and leaves its weight to the mode."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, 4, 8, 1))
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.bias = nn.Parameter(torch.zeros(4))

    def no_weight_decay(self):
        return {"embedding"}


class _Coefficients(nn.Module):
    """A layer whose declared parameters sit in a container without methods of its own."""

    def __init__(self):
        super().__init__()
        self.coeffs = nn.ParameterList([nn.Parameter(torch.zeros(2, 3)), nn.Parameter(torch.zeros(2, 3))])

    def no_weight_decay(self):
        return {"coeffs.0", "coeffs.1"}


class _DeclaringNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(_Embedding(), nn.Linear(4, 4))
        self.spectral = _Coefficients()
        self.plain = nn.Linear(4, 4)


class TestWeightDecayDeclarations(unittest.TestCase):
    """Layers declare the parameters not to decay through no_weight_decay(), by names relative to them."""

    def setUp(self):
        set_seed(0)
        self.model = _DeclaringNet()
        self.wd = 0.1

    def _groups(self, mode):
        return self._groups_at(self.wd, mode)

    def _groups_at(self, wd, mode):
        # the first group decays, the second does not, by the order get_parameter_groups builds them in
        groups = get_parameter_groups(self.model, wd, mode)
        if len(groups) == 1:
            return sorted(n for n, _ in self.model.named_parameters()), []
        decay, no_decay = groups
        named = {id(p): name for name, p in self.model.named_parameters()}
        return sorted(named[id(p)] for p in decay["params"]), sorted(named[id(p)] for p in no_decay["params"])

    def test_declared_names_are_prefixed_with_their_layer(self):
        declared = get_no_weight_decay_parameters(self.model)
        self.assertEqual(sorted(declared), ["block.0.embedding", "spectral.coeffs.0", "spectral.coeffs.1"])
        parameters = dict(self.model.named_parameters())
        for name, param in declared.items():
            self.assertIs(param, parameters[name])

    def test_full_mode_decays_all_but_the_declared(self):
        decay, no_decay = self._groups("full")
        self.assertEqual(no_decay, ["block.0.embedding", "spectral.coeffs.0", "spectral.coeffs.1"])
        self.assertEqual(
            decay, ["block.0.bias", "block.0.weight", "block.1.bias", "block.1.weight", "plain.bias", "plain.weight"]
        )

    def test_transformer_mode_adds_the_declared_to_its_exemptions(self):
        decay, no_decay = self._groups("transformer")
        self.assertEqual(decay, ["block.0.weight", "block.1.weight", "plain.weight"])
        self.assertEqual(
            no_decay,
            [
                "block.0.bias",
                "block.0.embedding",
                "block.1.bias",
                "plain.bias",
                "spectral.coeffs.0",
                "spectral.coeffs.1",
            ],
        )

    def test_without_declarations_full_mode_is_one_group(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        groups = get_parameter_groups(model, self.wd, "full")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["weight_decay"], self.wd)

    def test_without_weight_decay_the_groups_are_those_of_the_mode(self):
        """Without weight decay, the declarations are left out, so that the optimizer state of earlier runs loads."""
        groups = get_parameter_groups(self.model, 0.0, "full")
        self.assertEqual(len(groups), 1)

        # the groups of the mode alone: the declared embedding and coefficients, 4-D and 2-D, in the first group
        decay, no_decay = self._groups_at(0.0, "transformer")
        self.assertEqual(
            decay,
            [
                "block.0.embedding",
                "block.0.weight",
                "block.1.weight",
                "plain.weight",
                "spectral.coeffs.0",
                "spectral.coeffs.1",
            ],
        )
        self.assertEqual(no_decay, ["block.0.bias", "block.1.bias", "plain.bias"])

    def test_a_shared_parameter_counts_once(self):
        """A parameter declared by a layer that is used twice is exempt once, and in no other group."""
        embedding = _Embedding()
        model = nn.Sequential(embedding, embedding)
        declared = get_no_weight_decay_parameters(model)
        self.assertEqual(list(declared), ["0.embedding"])
        groups = get_parameter_groups(model, self.wd, "full")
        counts = [len(g["params"]) for g in groups]
        self.assertEqual(sum(counts), len(list(model.parameters())))

    def test_an_unknown_name_raises(self):
        class _Stale(nn.Module):
            def __init__(self):
                super().__init__()
                self.pos_embed = nn.Parameter(torch.zeros(2, 2))

            def no_weight_decay(self):
                return {"pos_embed", "cls_token"}

        with self.assertRaisesRegex(ValueError, "cls_token"):
            get_parameter_groups(nn.Sequential(_Stale()), self.wd, "full")


class TestNonFiniteLossCheck(unittest.TestCase):
    """The check runs single process here; the all-reduce of the flag is skipped without torch.distributed."""

    def test_finite_loss_passes(self):
        check = NonFiniteLossCheck()
        for _ in range(3):
            check(torch.tensor(1.5), step=1)

    def test_non_finite_loss_raises(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(FloatingPointError, "step 7 of epoch 2"):
                    NonFiniteLossCheck()(torch.tensor(value), step=7, epoch=2)

    def test_tolerance_with_grad_scaler(self):
        check = NonFiniteLossCheck(max_consecutive=2, grad_scaler_enabled=True)
        nan = torch.tensor(float("nan"))
        check(nan, step=1)
        check(nan, step=2)
        # a finite step resets the count
        check(torch.tensor(1.0), step=3)
        check(nan, step=4)
        check(nan, step=5)
        with self.assertRaises(FloatingPointError):
            check(nan, step=6)

    def test_no_tolerance_without_grad_scaler(self):
        # without the scaler a tolerated step would be applied to the weights
        check = NonFiniteLossCheck(max_consecutive=2, grad_scaler_enabled=False)
        with self.assertRaises(FloatingPointError):
            check(torch.tensor(float("nan")), step=1)


if __name__ == "__main__":
    unittest.main()
