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

from makani.utils.training.training_helpers import clip_grads, _compute_total_grad_norm, normalize_weights

import sys, os
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
        unit_after  = p.data / p.data.norm()
        self.assertTrue(torch.allclose(unit_after, unit_before, atol=1e-5))

    def test_complex_params_supported(self):
        model = nn.Module()
        model.p = nn.Parameter(torch.randn(4, 4, dtype=torch.complex64) * 3.0)
        normalize_weights(model)
        norm = model.p.data.abs().norm().item()
        self.assertAlmostEqual(norm, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
