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
The stochastic interpolant on spatially split ranks.

A distributed interpolant and a serial one get the same pointwise network and the same noise of the path, the serial
one the global field and every rank its shard, so the distributed forecast and drift pairs have to reproduce the
serial ones: the window statistics reduce over the split grid. The draws themselves are checked across the ranks:
the interpolation times are those of the sample on every model rank, and the matmul ranks draw the same noise.
"""

import os
import sys
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn

from makani.models.stochastic_interpolant import StochasticInterpolantWrapper, interpolant_input_channels
from makani.utils import comm
from makani.utils.dataloaders.batch import INPUT, TARGET
from makani.utils.dataloaders.rollout import Rollout

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import (
    _gather_helper,
    _init_grid,
    _split_helper,
    get_default_parameters,
    reduce_success,
    set_image_shape,
    teardown_comms,
)
from ..testutils import compare_tensors, disable_tf32, set_seed
from ..test_rollout import _batch

# an odd number of latitudes, so that the grid splits unevenly over two ranks
_GRID = (17, 32)
_B = 2
_C = 3
_N_STEPS = 3


class _TinyNet(nn.Module):
    """A pointwise drift network, the same on any part of the grid."""

    def __init__(self, in_chans, out_chans, hidden=8):
        super().__init__()
        self.fc1 = nn.Conv2d(in_chans, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, out_chans, 1)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def _params():
    params = get_default_parameters()
    set_image_shape(params, *_GRID)
    params.n_history = 1
    params.n_future = 1
    params.add_zenith = True
    params.history_normalization_mode = "mean"
    params.dhours = 6
    params.batch_size = _B
    params.in_channels = params.out_channels = list(range(_C))
    params.channel_names = params.channel_names[:_C]
    params.N_in_channels = (params.n_history + 1) * (_C + 1)
    params.N_out_channels = _C
    params.stochastic_interpolant = {"n_steps": _N_STEPS, "n_samples": 2, "antithetic": True}
    return params


class _Replay:
    """Hands the given fields out as the noise of the path, in order, whatever the sub-step."""

    def __init__(self, fields):
        self.fields = list(fields)

    def __call__(self, substep):
        return self.fields.pop(0)


class TestDistributedInterpolant(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    @classmethod
    def tearDownClass(cls):
        teardown_comms()

    def setUp(self):
        disable_tf32()

    def _shard(self, x):
        x = _split_helper(x, dim=-2, group=self.h_group)
        return _split_helper(x, dim=-1, group=self.w_group)

    def _gather(self, x):
        x = _gather_helper(x, dim=-2, group=self.h_group)
        return _gather_helper(x, dim=-1, group=self.w_group)

    def _shard_batch(self, batch):
        return {
            key: (
                {name: self._shard(feature) for name, feature in value.items()}
                if isinstance(value, dict)
                else (self._shard(value) if key in (INPUT, TARGET) else value)
            )
            for key, value in batch.items()
        }

    def _pair(self):
        """The interpolant on the grid and the same one built serially, with the same network."""
        params = _params()
        n_in = interpolant_input_channels(params)
        set_seed(333)
        wrapper = StochasticInterpolantWrapper(params, lambda: _TinyNet(n_in, _C)).to(self.device)
        with comm.serial():
            set_seed(333)
            reference = StochasticInterpolantWrapper(params, lambda: _TinyNet(n_in, _C)).to(self.device)
        return wrapper, reference

    def _batches(self):
        set_seed(1)
        batch = _batch(n_history=1, n_future=1, grid=_GRID)
        batch = {
            key: (
                {n: f.to(self.device) for n, f in value.items()} if isinstance(value, dict) else value.to(self.device)
            )
            for key, value in batch.items()
        }
        return self._shard_batch(batch), batch

    def _fields(self, n):
        generator = torch.Generator().manual_seed(5)
        return [torch.randn((_B, _C) + _GRID, generator=generator).to(self.device) for _ in range(n)]

    def _check(self, label, local, full, atol=1e-5, rtol=1e-5):
        with self.subTest(desc=label):
            self.assertTrue(reduce_success(compare_tensors(label, self._gather(local), full, atol, rtol), self.device))

    def test_forecast_matches_serial(self):
        wrapper, reference = self._pair()
        wrapper.eval()
        reference.eval()
        local_batch, batch = self._batches()
        fields = self._fields(2 * _N_STEPS)

        with torch.no_grad():
            with mock.patch.object(wrapper, "path_noise", _Replay(self._shard(f) for f in fields)):
                pred = wrapper(Rollout(local_batch, noise=wrapper.noise))
            with mock.patch.object(reference, "path_noise", _Replay(fields)):
                expected = reference(Rollout(batch, noise=reference.noise))

        self._check("forecast", pred, expected)

    def test_drift_pairs_match_serial(self):
        wrapper, reference = self._pair()
        wrapper.train()
        reference.train()
        local_batch, batch = self._batches()
        fields = self._fields(2)
        generator = torch.Generator().manual_seed(6)
        times = [torch.rand(_B, 2, generator=generator).to(self.device) for _ in range(2)]

        def replay_times(t):
            t = list(t)
            return lambda device=None, dtype=torch.float32: torch.cat([t[0], 1.0 - t.pop(0)], dim=1)

        with (
            mock.patch.object(wrapper, "path_noise", _Replay(self._shard(f) for f in fields)),
            mock.patch.object(wrapper, "interpolation_times", replay_times(times)),
        ):
            predicted, drift = wrapper(Rollout(local_batch, noise=wrapper.noise), drift=True)
        with (
            mock.patch.object(reference, "path_noise", _Replay(fields)),
            mock.patch.object(reference, "interpolation_times", replay_times(times)),
        ):
            predicted_ref, drift_ref = reference(Rollout(batch, noise=reference.noise), drift=True)

        self._check("path drift", drift, drift_ref)
        self._check("network drift", predicted.detach(), predicted_ref.detach())

    def test_times_are_shared_by_the_model_ranks(self):
        wrapper, _ = self._pair()
        wrapper.noise.advance(replace_state=True, batch_size=_B)
        times = wrapper.interpolation_times(self.device)
        if comm.get_size("model") > 1:
            gathered = [torch.empty_like(times) for _ in range(comm.get_size("model"))]
            dist.all_gather(gathered, times, group=comm.get_group("model"))
            self.assertTrue(reduce_success(all(torch.equal(t, times) for t in gathered), self.device))

    def test_matmul_ranks_draw_the_same_noise(self):
        if comm.get_size("matmul") < 2:
            self.skipTest("needs at least two matmul ranks")
        wrapper, _ = self._pair()
        wrapper.noise.advance(replace_state=True, batch_size=_B)
        noise = wrapper.path_noise(0).contiguous()
        gathered = [torch.empty_like(noise) for _ in range(comm.get_size("matmul"))]
        dist.all_gather(gathered, noise, group=comm.get_group("matmul"))
        self.assertTrue(reduce_success(all(torch.equal(n, noise) for n in gathered), self.device))


if __name__ == "__main__":
    unittest.main()
