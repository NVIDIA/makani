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

import unittest
from parameterized import parameterized_class

import torch

from makani.models.stochastic_physics import StochasticPhysics

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))

IMG_SHAPE = (32, 64)
BATCH_SIZE = 2


class _Params:
    """Minimal params stub supporting attribute and .get() access, like ParamsBase."""

    def __init__(self, d):
        self.__dict__.update(d)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def _make_params(n_history=0, n_out=4, sp=None, in_channels=None, out_channels=None):
    if sp is None:
        sp = {"type": "diffusion", "n_channels": 1, "sigma": 0.5, "lmax": 20, "kT": [3.15e-2], "clip": 0.8}
    c_base = n_out
    return _Params(
        dict(
            stochastic_physics=sp,
            img_shape_x_resampled=IMG_SHAPE[0],
            img_shape_y_resampled=IMG_SHAPE[1],
            batch_size=BATCH_SIZE,
            model_grid_type="equiangular",
            dt=6,
            dhours=1,
            n_history=n_history,
            N_in_predicted_channels=c_base * (n_history + 1),
            N_out_channels=n_out,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    )


@parameterized_class(("device",), _devices)
class TestStochasticPhysics(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.B, self.H, self.W = BATCH_SIZE, IMG_SHAPE[0], IMG_SHAPE[1]

    def _build(self, **kw):
        sp = StochasticPhysics(_make_params(**kw)).to(self.device)
        sp.update(replace_state=True)
        return sp

    def test_output_shape(self):
        sp = self._build(n_out=4)
        inp = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        pred = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        self.assertEqual(sp(inp, pred).shape, pred.shape)

    def test_zero_tendency_no_perturbation(self):
        """The defining SPPT property: where pred == baseline, output == pred exactly."""
        sp = self._build(n_out=4)
        inp = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        pred = inp.clone()  # persistence => zero tendency
        out = sp(inp, pred)
        self.assertTrue(compare_tensors("zero_tendency", out, pred, atol=1e-6, verbose=True))

    def test_nonzero_tendency_is_perturbed(self):
        sp = self._build(n_out=4)
        inp = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        pred = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        self.assertFalse(compare_tensors("nonzero_tendency", sp(inp, pred), pred, atol=1e-6))

    def test_history_offset_baseline(self):
        """With history, the baseline is the current (last) timestep block, not a past one."""
        sp = self._build(n_history=1, n_out=3)
        self.assertEqual(sp.baseline_index.tolist(), [3, 4, 5])
        inp = torch.randn(self.B, 6, self.H, self.W, device=self.device)
        pred = inp[:, 3:6].clone()  # equal to current block => zero tendency
        self.assertTrue(compare_tensors("history_baseline", sp(inp, pred), pred, atol=1e-6, verbose=True))

    def test_clip_prevents_sign_flip(self):
        """With clip < 1, (1 + r) stays positive, so the perturbed tendency never flips sign."""
        sp = self._build(
            n_out=4, sp={"type": "diffusion", "n_channels": 1, "sigma": 2.0, "lmax": 20, "kT": [3.15e-2], "clip": 0.8}
        )
        inp = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        pred = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        out = sp(inp, pred)
        baseline = inp  # n_history=0, in==out
        self.assertTrue(
            compare_tensors(
                "sign_preserved", torch.sign(out - baseline), torch.sign(pred - baseline), atol=0.0, rtol=0.0
            )
        )

    def test_update_advances_pattern(self):
        sp = self._build(n_out=4)
        inp = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        pred = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        out1 = sp(inp, pred)
        sp.update(replace_state=False)  # AR(1) step
        out2 = sp(inp, pred)
        self.assertFalse(compare_tensors("update_advances", out1, out2, atol=1e-6))

    def test_requires_predicted_channels_in_input(self):
        """A predicted channel with no input counterpart has no persistence baseline -> error."""
        with self.assertRaises(ValueError):
            StochasticPhysics(_make_params(n_out=3, in_channels=[0, 1, 2], out_channels=[0, 1, 9]))

    def test_channel_count_validation(self):
        """n_channels must be 1 (shared) or N_out (per-channel)."""
        sp = self._build(
            n_out=4, sp={"type": "diffusion", "n_channels": 3, "sigma": 0.5, "lmax": 20, "kT": [1e-2, 2e-2, 3e-2]}
        )
        inp = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        pred = torch.randn(self.B, 4, self.H, self.W, device=self.device)
        with self.assertRaises(ValueError):
            sp(inp, pred)


if __name__ == "__main__":
    unittest.main()
