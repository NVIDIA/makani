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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import unittest
from parameterized import parameterized

import torch
import torch.nn as nn

from makani.models.stepper import SingleStepWrapper, MultiStepWrapper

from .testutils import set_seed, get_default_parameters, compare_tensors, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W


class _ScaleModel(nn.Module):
    """
    Dummy model that scales the most-recent timestep slice by a learnable factor.

    The wrapper feeds (B, (n_history+1)*C, H, W) and expects (B, C, H, W) back, so
    we slice the trailing C channels (the latest timestep in the flattened-history
    layout) and multiply by ``scale``. With scale=2 the rollout produces the
    geometric sequence  pred_k = 2^(k+1) * (last C of initial input), which makes
    every assertion in this file an exact equality check.
    """

    def __init__(self, n_out_chans: int, scale: float = 2.0):
        super().__init__()
        self.n_out_chans = n_out_chans
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return self.scale * x[..., -self.n_out_chans:, :, :]


class TestStepper(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.B = 1
        self.C = NUM_CHANNELS
        self.H = IMG_SIZE_H
        self.W = IMG_SIZE_W

    def _make_params(self, n_history=0, n_future=0, push_forward=False):
        params = get_default_parameters()
        params.n_history = n_history
        params.n_future = n_future
        params.multistep = {"push_forward": push_forward}
        return params

    def _make_handle(self):
        return lambda: _ScaleModel(n_out_chans=self.C, scale=2.0)

    # ------------------------------------------------------------------
    # SingleStepWrapper
    # ------------------------------------------------------------------

    def test_single_step_no_history(self):
        params = self._make_params(n_history=0)
        wrapper = SingleStepWrapper(params, self._make_handle())
        wrapper.train()
        inp = torch.randn(self.B, self.C, self.H, self.W)
        out = wrapper(inp)
        self.assertEqual(out.shape, inp.shape)
        self.assertTrue(compare_tensors("single_step_no_history", out, 2.0 * inp, verbose=True))

    def test_single_step_with_history(self):
        n_history = 2
        params = self._make_params(n_history=n_history)
        wrapper = SingleStepWrapper(params, self._make_handle())
        wrapper.train()
        # flattened-history input layout: (n_history+1)*C channels, oldest first
        inp = torch.randn(self.B, (n_history + 1) * self.C, self.H, self.W)
        out = wrapper(inp)
        self.assertEqual(out.shape, (self.B, self.C, self.H, self.W))
        # only the most-recent timestep slice is consumed by the dummy model
        self.assertTrue(compare_tensors("single_step_with_history", out, 2.0 * inp[:, -self.C:], verbose=True))

    # ------------------------------------------------------------------
    # MultiStepWrapper — train mode produces the full rollout
    # ------------------------------------------------------------------

    @parameterized.expand([(0,), (1,)])
    def test_multistep_train_geometric_sequence(self, n_history):
        n_future = 3
        params = self._make_params(n_history=n_history, n_future=n_future)
        wrapper = MultiStepWrapper(params, self._make_handle())
        wrapper.train()

        in_chans = (n_history + 1) * self.C
        inp = torch.randn(self.B, in_chans, self.H, self.W)
        out = wrapper(inp)

        # rollout output: (B, (n_future+1)*C, H, W) — predictions concatenated along channel dim
        self.assertEqual(out.shape, (self.B, (n_future + 1) * self.C, self.H, self.W))

        # k-th block must equal 2^(k+1) * (last-C slice of inp): each step scales
        # the previous prediction by 2, and append_history places that prediction
        # at the most-recent position for the next call
        last = inp[:, -self.C:]
        for k in range(n_future + 1):
            block = out[:, k * self.C:(k + 1) * self.C]
            self.assertTrue(
                compare_tensors(
                    f"multistep_train_step_{k}_h{n_history}",
                    block,
                    (2.0 ** (k + 1)) * last,
                    verbose=True,
                )
            )

    # ------------------------------------------------------------------
    # MultiStepWrapper — eval mode collapses to a single forward
    # ------------------------------------------------------------------

    def test_multistep_eval_is_single_step(self):
        n_future = 2
        params = self._make_params(n_history=0, n_future=n_future)
        wrapper = MultiStepWrapper(params, self._make_handle())
        wrapper.eval()
        inp = torch.randn(self.B, self.C, self.H, self.W)
        out = wrapper(inp)
        # _forward_eval returns one step regardless of n_future
        self.assertEqual(out.shape, (self.B, self.C, self.H, self.W))
        self.assertTrue(compare_tensors("multistep_eval", out, 2.0 * inp, verbose=True))

    def test_train_eval_dispatch(self):
        params = self._make_params(n_history=0, n_future=2)
        wrapper = MultiStepWrapper(params, self._make_handle())
        inp = torch.randn(self.B, self.C, self.H, self.W)

        wrapper.train()
        out_train = wrapper(inp)
        self.assertEqual(out_train.shape[1], (params.n_future + 1) * self.C)

        wrapper.eval()
        out_eval = wrapper(inp)
        self.assertEqual(out_eval.shape[1], self.C)

    # ------------------------------------------------------------------
    # push_forward: same numerics, but a strictly truncated gradient through
    # the rollout (each step's input is detached, so gradients only flow one
    # step at a time)
    # ------------------------------------------------------------------

    def test_push_forward_matches_no_push(self):
        n_future = 2
        inp = torch.randn(self.B, self.C, self.H, self.W)

        wrapper_off = MultiStepWrapper(self._make_params(n_future=n_future, push_forward=False), self._make_handle())
        wrapper_on = MultiStepWrapper(self._make_params(n_future=n_future, push_forward=True), self._make_handle())
        wrapper_off.train()
        wrapper_on.train()

        out_off = wrapper_off(inp)
        out_on = wrapper_on(inp)
        self.assertTrue(compare_tensors("push_forward_values", out_off, out_on, verbose=True))

    def test_push_forward_truncates_gradient(self):
        # use ones() so the gradient takes a known closed form and we can
        # check exact values rather than just an inequality
        n_future = 2
        inp = torch.ones(self.B, self.C, self.H, self.W)

        wrapper_off = MultiStepWrapper(self._make_params(n_future=n_future, push_forward=False), self._make_handle())
        wrapper_on = MultiStepWrapper(self._make_params(n_future=n_future, push_forward=True), self._make_handle())
        wrapper_off.train()
        wrapper_on.train()

        # loss = sum of all rollout outputs. With dummy model `pred = scale * last_C(inp)`
        # and inp=ones, loss reduces to (scale + scale^2 + scale^3) * B*C*H*W.
        # d/d(scale) without push_forward = (1 + 2*scale + 3*scale^2) * B*C*H*W
        # d/d(scale) with push_forward    = (1 +   scale +   scale^2) * B*C*H*W
        # (push_forward detaches each step's input, so dpred_k/dscale only sees
        #  the direct multiplication, not the chain through earlier scales)
        wrapper_off(inp).sum().backward()
        wrapper_on(inp).sum().backward()

        bchw = self.B * self.C * self.H * self.W
        scale = 2.0
        expected_off = (1.0 + 2.0 * scale + 3.0 * scale * scale) * bchw
        expected_on  = (1.0 + scale + scale * scale) * bchw

        g_off = wrapper_off.model.scale.grad.item()
        g_on = wrapper_on.model.scale.grad.item()

        self.assertAlmostEqual(g_off, expected_off, places=3)
        self.assertAlmostEqual(g_on, expected_on, places=3)
        # sanity: truncating the rollout strictly reduces gradient magnitude
        self.assertGreater(g_off, g_on)


if __name__ == "__main__":
    unittest.main()
