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
The stochastic interpolant as a model over rollouts.

The random draws of the interpolant, the noise of the path and the interpolation times, are replaced by recorded
ones, so that the sampler and the drift pairs can be checked against the same computations written out by hand.
"""

import os
import shutil
import sys
import unittest
from unittest import mock

import torch
import torch.nn as nn
from parameterized import parameterized

from makani.models.noise_controller import NoiseController
from makani.models.stochastic_interpolant import (
    PATH_NOISE,
    PathKeys,
    StochasticInterpolantWrapper,
    interpolant_config,
    interpolant_input_channels,
)
from makani.utils.dataloaders.batch import INPUT, TARGET, INPUT_DYNAMIC_FEATURES, TARGET_DYNAMIC_FEATURES, ZENITH
from makani.utils.dataloaders.rollout import Rollout

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import get_default_parameters, set_seed, compare_tensors, disable_tf32
from .test_rollout import _B, _C, _H, _W, _batch

_GRID = (_H * 4, _W * 4)


class _TinyNet(nn.Module):
    """A small drift network, taking the interpolation time as a channel."""

    def __init__(self, in_chans, out_chans, hidden=8):
        super().__init__()
        self.fc1 = nn.Conv2d(in_chans, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, out_chans, 1)
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach())
        return self.fc2(torch.tanh(self.fc1(x)))


class _TimeAwareNet(_TinyNet):
    """A drift network taking the interpolation time as an argument."""

    def __init__(self, in_chans, out_chans, hidden=8):
        super().__init__(in_chans, out_chans, hidden)
        self.times = []

    def forward(self, x, s):
        self.times.append(s.detach())
        return super().forward(x) * (1.0 + s.reshape(-1, 1, 1, 1))


def _params(n_history=1, n_future=1, block=None, input_noise=False):
    params = get_default_parameters()
    params.img_shape_x = params.img_shape_x_resampled = _GRID[0]
    params.img_shape_y = params.img_shape_y_resampled = _GRID[1]
    params.n_history = n_history
    params.n_future = n_future
    params.add_zenith = True
    params.history_normalization_mode = "mean"
    params.dhours = 6
    params.batch_size = _B
    params.in_channels = params.out_channels = list(range(_C))
    params.channel_names = params.channel_names[:_C]
    n_noise = 0
    if input_noise:
        params.input_noise = {"type": "white", "mode": "concatenate", "n_channels": 1}
        n_noise = 1
    # as the driver derives them: the window with its dynamic channels, no static ones
    params.N_in_channels = (n_history + 1) * (_C + 1 + n_noise)
    params.N_out_channels = _C
    params.stochastic_interpolant = dict(block or {})
    return params


def _wrapper(params, net=_TinyNet):
    set_seed(333)
    return StochasticInterpolantWrapper(params, lambda: net(interpolant_input_channels(params), _C))


class _Draws:
    """Records the draws of a wrapper, or replays recorded ones."""

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.noise = []
        self.times = []

    def record(self):
        draw_noise, draw_times = self.wrapper.path_noise, self.wrapper.interpolation_times

        def noise(substep):
            self.noise.append(draw_noise(substep))
            return self.noise[-1]

        def times(device=None, dtype=torch.float32):
            self.times.append(draw_times(device, dtype))
            return self.times[-1]

        return mock.patch.multiple(self.wrapper, path_noise=noise, interpolation_times=times)


def _window(wrapper, inp, zenith):
    """The packed window, its statistics and the start of the path, written out."""
    pp = wrapper.preprocessor
    # the state normalized with its statistics, the zenith angle with its own, then the time folded into the channels
    stats = pp.compute_history_stats(inp)
    zenith = pp.history_normalize(zenith, pp.compute_history_stats(zenith))
    inpans = torch.cat([pp.history_normalize(inp, stats), zenith], dim=2).flatten(1, 2)
    x0 = pp.history_normalize(inp[:, -1:], stats, target=True).squeeze(1)
    return inpans, stats, x0


def _drift(wrapper, inpans, x, s):
    s_channel = s.reshape(-1, 1, 1, 1).expand(-1, 1, *x.shape[-2:])
    return wrapper.model(torch.cat([inpans, x, s_channel], dim=1))


class TestConfig(unittest.TestCase):

    def test_defaults(self):
        params = get_default_parameters()
        self.assertEqual(
            interpolant_config(params),
            {
                "epsilon": 1.0,
                "foellmer": False,
                "antithetic": False,
                "n_samples": 1,
                "n_steps": 1,
                "time_aware": False,
                "seed": 333,
            },
        )

    def test_block_over_legacy_keys(self):
        params = get_default_parameters()
        params.stochastic_size = 4
        params.stochastic_interpolation_steps = 7
        params.noise_epsilon = 0.5
        params.stochastic_interpolant = {"n_steps": 3}
        config = interpolant_config(params)
        self.assertEqual((config["n_samples"], config["n_steps"], config["epsilon"]), (4, 3, 0.5))

    def test_unknown_setting_refused(self):
        params = get_default_parameters()
        params.stochastic_interpolant = {"n_step": 3}
        with self.assertRaisesRegex(ValueError, "n_step"):
            interpolant_config(params)

    def test_input_channels(self):
        params = _params()
        self.assertEqual(interpolant_input_channels(params), params.N_in_channels + _C + 1)
        params.stochastic_interpolant = {"time_aware": True}
        self.assertEqual(interpolant_input_channels(params), params.N_in_channels + _C)

    def test_other_output_channels_refused(self):
        params = _params()
        params.out_channels = list(range(1, _C + 1))
        with self.assertRaisesRegex(ValueError, "same"):
            _wrapper(params)

    def test_constraints_refused(self):
        params = _params()
        params.constraints = []
        with self.assertRaisesRegex(ValueError, "Constraints"):
            _wrapper(params)


class TestSchedule(unittest.TestCase):

    def setUp(self):
        self.wrapper = _wrapper(_params(block={"epsilon": 0.7}))
        set_seed(1)
        self.x0, self.x1, self.z = (torch.randn(_B, _C, *_GRID) for _ in range(3))

    def test_endpoints(self):
        for s, expected in ((0.0, self.x0), (1.0, self.x1)):
            point, _ = self.wrapper.path(self.x0, self.x1, self.z, torch.tensor(s))
            self.assertTrue(compare_tensors(f"path at s={s}", point, expected, 1e-6, 1e-6))

    def test_drift_is_the_time_derivative_without_noise(self):
        s = torch.tensor(0.3, requires_grad=True)
        point, drift = self.wrapper.path(self.x0, self.x1, torch.zeros_like(self.z), s)
        (grad,) = torch.autograd.grad((point * self.x0).sum(), s)
        self.assertAlmostEqual(grad.item(), (drift * self.x0).sum().item(), places=3)

    def test_noise_term_of_the_drift(self):
        # sigma W_s with W_s = sqrt(s) z: only sigma is differentiated, see the module
        s = torch.tensor(0.3)
        _, drift = self.wrapper.path(torch.zeros_like(self.x0), torch.zeros_like(self.x1), self.z, s)
        self.assertTrue(compare_tensors("noise term", drift, torch.sqrt(s) * (-0.7) * self.z, 1e-6, 1e-6))

    def test_foellmer_diffusion_is_finite(self):
        self.wrapper.foellmer = True
        gsq = self.wrapper.gsq(torch.linspace(0.0, 1.0, 11))
        self.assertTrue(torch.isfinite(gsq).all())
        self.assertTrue((gsq >= 0).all())


class TestSampler(unittest.TestCase):

    def setUp(self):
        disable_tf32()

    @parameterized.expand([("euler", {"n_steps": 3}), ("foellmer", {"n_steps": 3, "foellmer": True})])
    def test_rollout_matches_euler_maruyama(self, _, block):
        """A rollout of two steps is the SDE integrated from each window, the second from the first forecast."""
        wrapper = _wrapper(_params(block=block)).eval()
        set_seed(2)
        batch = _batch(n_history=1, n_future=1, grid=_GRID)

        draws = _Draws(wrapper)
        with torch.no_grad(), draws.record():
            pred = wrapper(Rollout(batch, noise=wrapper.noise))

        inp = batch[INPUT]
        zenith = batch[INPUT_DYNAMIC_FEATURES][ZENITH]
        noise = iter(draws.noise)
        times = torch.linspace(0.0, 1.0, block["n_steps"] + 1)
        expected = []
        with torch.no_grad():
            for k in range(2):
                inpans, stats, x0 = _window(wrapper, inp, zenith)
                x = x0
                for i in range(block["n_steps"]):
                    s = times[i].expand(_B)
                    ds = times[i + 1] - times[i]
                    b = _drift(wrapper, inpans, x, s)
                    sr = s.reshape(-1, 1, 1, 1)
                    if i == 0:
                        diffusion = wrapper.sigma(sr) * torch.sqrt(ds)
                    else:
                        if wrapper.foellmer:
                            b = b + 0.5 * (wrapper.gsq(sr) - wrapper.sigma(sr) ** 2) * wrapper.score(x, x0, b, sr)
                        diffusion = torch.sqrt(wrapper.gsq(sr) * ds)
                    x = x + b * ds + diffusion * next(noise)
                step = wrapper.preprocessor.history_denormalize(x.unsqueeze(1), stats, target=True)
                expected.append(step)
                inp = torch.cat([inp[:, 1:], step], dim=1)
                zenith = torch.cat([zenith[:, 1:], batch[TARGET_DYNAMIC_FEATURES][ZENITH][:, k : k + 1]], dim=1)

        self.assertEqual(len(draws.noise), 2 * block["n_steps"])
        self.assertTrue(compare_tensors("forecast", pred, torch.cat(expected, dim=1), 1e-5, 1e-5))

    def test_time_aware_network_gets_the_times(self):
        params = _params(block={"n_steps": 2, "time_aware": True})
        wrapper = _wrapper(params, net=_TimeAwareNet).eval()
        batch = _batch(n_history=1, n_future=0, grid=_GRID)
        with torch.no_grad():
            pred = wrapper(Rollout(batch, noise=wrapper.noise))
        self.assertEqual(tuple(pred.shape), (_B, 1, _C, *_GRID))
        self.assertEqual([tuple(s.shape) for s in wrapper.model.times], [(_B,), (_B,)])
        self.assertEqual([s[0].item() for s in wrapper.model.times], [0.0, 0.5])
        self.assertEqual(wrapper.model.inputs[0].shape[1], params.N_in_channels + _C)

    def test_time_channel(self):
        wrapper = _wrapper(_params(block={"n_steps": 2})).eval()
        batch = _batch(n_history=1, n_future=0, grid=_GRID)
        with torch.no_grad():
            wrapper(Rollout(batch, noise=wrapper.noise))
        self.assertEqual([x[:, -1].unique().tolist() for x in wrapper.model.inputs], [[0.0], [0.5]])

    def test_input_noise_is_used(self):
        """The input noise of the preprocessor is concatenated to the window, as for any model."""
        params = _params(block={"n_steps": 1}, input_noise=True)
        wrapper = _wrapper(params).eval()
        self.assertTrue(wrapper.noise.is_stochastic)
        batch = _batch(n_history=1, n_future=0, grid=_GRID)
        with torch.no_grad():
            wrapper(Rollout(batch, noise=wrapper.noise))
        self.assertEqual(wrapper.model.inputs[0].shape[1], interpolant_input_channels(params))

    def test_not_encodable(self):
        with self.assertRaises(NotImplementedError):
            _wrapper(_params()).check_encodable()


class TestDriftPairs(unittest.TestCase):

    def setUp(self):
        disable_tf32()

    @parameterized.expand([("plain", False), ("antithetic", True)])
    def test_pairs_match_the_path(self, _, antithetic):
        """The drifts of the network and of the path, from the latest state of the window to the target."""
        n_samples = 2
        wrapper = _wrapper(_params(n_future=0, block={"n_samples": n_samples, "antithetic": antithetic})).train()
        set_seed(3)
        batch = _batch(n_history=1, n_future=0, grid=_GRID)

        draws = _Draws(wrapper)
        with draws.record():
            predicted, expected = wrapper(Rollout(batch, noise=wrapper.noise), drift=True)

        n = n_samples * (2 if antithetic else 1)
        self.assertEqual(tuple(predicted.shape), (_B * n, _C, *_GRID))
        self.assertEqual(predicted.shape, expected.shape)
        (s,), (z,) = draws.times, draws.noise
        if antithetic:
            self.assertTrue(compare_tensors("antithetic", s[:, n_samples:], 1.0 - s[:, :n_samples], 1e-6, 1e-6))

        with torch.no_grad():
            inpans, stats, x0 = _window(wrapper, batch[INPUT], batch[INPUT_DYNAMIC_FEATURES][ZENITH])
            x1 = wrapper.preprocessor.history_normalize(batch[TARGET], stats, target=True).squeeze(1)
            want_pred, want_drift = [], []
            for j in range(n):
                point, drift = wrapper.path(x0, x1, z, s[:, j].reshape(-1, 1, 1, 1))
                want_drift.append(drift)
                want_pred.append(_drift(wrapper, inpans, point, s[:, j]))
            # batch-major, as the pairs are flattened
            want_drift = torch.stack(want_drift, dim=1).flatten(0, 1)
            want_pred = torch.stack(want_pred, dim=1).flatten(0, 1)

        self.assertTrue(compare_tensors("path drift", expected, want_drift, 1e-5, 1e-5))
        self.assertTrue(compare_tensors("network drift", predicted.detach(), want_pred, 1e-5, 1e-5))

    def test_multistep_refused(self):
        """A later step from the true state is one more single-step pair, not a multistep forecast."""
        wrapper = _wrapper(_params()).train()
        batch = _batch(n_history=1, n_future=1, grid=_GRID)
        with self.assertRaisesRegex(ValueError, "single steps"):
            wrapper(Rollout(batch, noise=wrapper.noise), drift=True)

    def test_gradients_reach_the_network(self):
        wrapper = _wrapper(_params(block={"n_samples": 2})).train()
        batch = _batch(n_history=1, n_future=0, grid=_GRID)
        predicted, expected = wrapper(Rollout(batch, noise=wrapper.noise), drift=True)
        self.assertFalse(expected.requires_grad)
        torch.square(predicted - expected).mean().backward()
        for name, param in wrapper.model.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(param.grad)
                self.assertGreater(param.grad.abs().sum().item(), 0.0)

    def test_ensemble_refused(self):
        wrapper = _wrapper(_params()).train()
        batch = _batch(n_history=1, n_future=0, grid=_GRID)
        with self.assertRaisesRegex(ValueError, "ensemble"):
            wrapper(Rollout(batch, noise=wrapper.noise, ensemble_size=2), drift=True)


class TestPathNoise(unittest.TestCase):
    """The draws of the interpolant, derived from the keys the noise controller drives."""

    def setUp(self):
        disable_tf32()

    def test_keys_are_driven_by_the_controller(self):
        wrapper = _wrapper(_params())
        self.assertIn(PATH_NOISE, wrapper.noise)
        self.assertTrue(wrapper.noise.is_stochastic)
        wrapper.noise.advance(replace_state=True, batch_size=5)
        self.assertEqual(tuple(wrapper.path_keys.state.shape), (5,))

    def test_restored_snapshot_redraws_the_forecast(self):
        wrapper = _wrapper(_params(block={"n_steps": 3})).eval()
        batch = _batch(n_history=1, n_future=1, grid=_GRID)
        snapshot = wrapper.noise.snapshot()
        with torch.no_grad():
            first = wrapper(Rollout(batch, noise=wrapper.noise))
            wrapper.noise.restore(snapshot)
            second = wrapper(Rollout(batch, noise=wrapper.noise))
            third = wrapper(Rollout(batch, noise=wrapper.noise))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))

    def test_draws_of_a_sample_do_not_depend_on_its_batch(self):
        wrapper = _wrapper(_params(block={"n_samples": 3}))
        wrapper.noise.advance(replace_state=True, batch_size=_B)
        noise, times = wrapper.path_noise(1), wrapper.interpolation_times()
        coefficients = wrapper.path_field.get_tensor_state()
        keys = wrapper.path_keys.get_tensor_state()

        wrapper.noise.restore(_snapshot_of(wrapper, keys[1:]))
        # the draws exactly; the transform of a smaller batch may round differently
        sub_noise = wrapper.path_noise(1)
        self.assertTrue(torch.equal(wrapper.path_field.get_tensor_state(), coefficients[1:]))
        self.assertTrue(compare_tensors("noise", sub_noise, noise[1:], 1e-6, 1e-5))
        self.assertTrue(torch.equal(wrapper.interpolation_times(), times[1:]))

    def test_substeps_draw_different_noise(self):
        wrapper = _wrapper(_params())
        wrapper.noise.advance(replace_state=True, batch_size=_B)
        self.assertFalse(torch.equal(wrapper.path_noise(0), wrapper.path_noise(1)))
        self.assertTrue(torch.equal(wrapper.path_noise(0), wrapper.path_noise(0)))

    def test_spatial_ranks_share_the_times_not_the_noise(self):
        # what the ranks of one sample draw, with the shard they draw for set by hand
        a, b = _wrapper(_params()), _wrapper(_params())
        b._spatial_rank = 1
        for w in (a, b):
            w.noise.advance(replace_state=True, batch_size=_B)
        self.assertTrue(torch.equal(a.path_keys.state, b.path_keys.state))
        self.assertTrue(torch.equal(a.interpolation_times(), b.interpolation_times()))
        self.assertFalse(torch.equal(a.path_noise(0), b.path_noise(0)))

    def test_data_parallel_ranks_draw_other_keys(self):
        # when built, and when re-seeded through the controller
        for reseed in (False, True):
            keys = []
            for rank in (0, 1):
                with mock.patch("makani.models.stochastic_interpolant.comm.get_rank", return_value=rank):
                    process = PathKeys(_B, seed=7)
                    if reseed:
                        NoiseController({PATH_NOISE: process}).set_rng(seed=7)
                process.update()
                keys.append(process.state)
            with self.subTest(reseed=reseed):
                self.assertFalse(torch.equal(keys[0], keys[1]))

    def test_reseeding_reproduces_the_construction(self):
        process = PathKeys(_B, seed=7)
        process.update()
        built = process.state
        NoiseController({PATH_NOISE: process}).set_rng(seed=7)
        process.update()
        self.assertTrue(torch.equal(process.state, built))

    def test_keys_refuse_another_layout(self):
        with self.assertRaises(ValueError):
            PathKeys(_B).set_tensor_state(torch.zeros(_B, 2, dtype=torch.int64))


def _snapshot_of(wrapper, keys):
    """A snapshot of the noise of the wrapper, with other keys."""
    snapshot = wrapper.noise.snapshot()
    state = snapshot.states[PATH_NOISE]
    snapshot.states[PATH_NOISE] = type(state)(tensor=keys, rng_cpu=state.rng_cpu, rng_gpu=state.rng_gpu)
    return snapshot


class _ReplayDraws:
    """Hands out recorded draws in order, so that an eager and a compiled run see the same ones."""

    def __init__(self, noise, times):
        self.noise, self.times = list(noise), list(times)

    def patch(self, wrapper):
        noise, times = iter(self.noise), iter(self.times)
        return mock.patch.multiple(
            wrapper,
            path_noise=lambda substep: next(noise),
            interpolation_times=lambda device=None, dtype=torch.float32: next(times),
        )


@unittest.skipUnless(shutil.which("c++") or shutil.which("g++"), "inductor needs a C++ compiler on the CPU")
class TestCompiledStep(unittest.TestCase):
    """compile_step compiles the drift network inside the wrapper; the module and its methods stay as they are."""

    def setUp(self):
        disable_tf32()

    def tearDown(self):
        torch._dynamo.reset()

    def test_compiled_matches_eager(self):
        params = _params(n_future=0, block={"n_steps": 3, "n_samples": 2})
        batch = _batch(n_history=1, n_future=0, grid=_GRID)
        set_seed(4)
        draws = _ReplayDraws([torch.randn(_B, _C, *_GRID) for _ in range(4)], [torch.rand(_B, 2)])

        results = {}
        for compiled in (False, True):
            wrapper = _wrapper(params)
            if compiled:
                wrapper.compile_step()
            with draws.patch(wrapper):
                wrapper.eval()
                with torch.no_grad():
                    forecast = wrapper(Rollout(batch, noise=wrapper.noise))
                wrapper.train()
                predicted, expected = wrapper(Rollout(batch, noise=wrapper.noise), drift=True)
            torch.square(predicted - expected).mean().backward()
            grads = {name: p.grad.clone() for name, p in wrapper.model.named_parameters()}
            results[compiled] = (forecast, predicted.detach(), grads)
            # the module is not wrapped: its methods and its state dict are those of the eager one
            self.assertIsInstance(wrapper, StochasticInterpolantWrapper)
            self.assertTrue(all(not key.startswith("_orig_mod") for key in wrapper.state_dict()))

        (forecast_ref, predicted_ref, grads_ref), (forecast, predicted, grads) = results[False], results[True]
        self.assertTrue(compare_tensors("forecast", forecast, forecast_ref, 1e-5, 1e-5))
        self.assertTrue(compare_tensors("drift", predicted, predicted_ref, 1e-5, 1e-5))
        for name in grads_ref:
            with self.subTest(gradient=name):
                self.assertTrue(compare_tensors(name, grads[name], grads_ref[name], 1e-5, 1e-5))


class _RegisteredNet(nn.Module):
    """A drift network as the registry builds one, from the shapes and channel counts of the run."""

    def __init__(self, inp_shape, out_shape, inp_chans, out_chans, **kwargs):
        super().__init__()
        self.inp_chans = inp_chans
        self.conv = nn.Conv2d(inp_chans, out_chans, 1)

    def forward(self, x):
        return self.conv(x)


class TestRegistry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from makani.models import model_registry

        if "_InterpolantTestNet" not in model_registry.list_models():
            model_registry.register_model(_RegisteredNet, "_InterpolantTestNet")

    def test_block_selects_the_interpolant(self):
        """The block in the params alone builds the interpolant, as when a model package is loaded."""
        from makani.models import model_registry

        params = _params(block={"n_steps": 2})
        params.nettype = "_InterpolantTestNet"
        params.normalization = "none"
        model = model_registry.get_model(params)
        self.assertIsInstance(model, StochasticInterpolantWrapper)
        self.assertEqual(model.n_steps, 2)
        self.assertEqual(model.model.inp_chans, interpolant_input_channels(params))

        # without the block, a plain StepWrapper
        params.stochastic_interpolant = None
        self.assertNotIsInstance(model_registry.get_model(params), StochasticInterpolantWrapper)


if __name__ == "__main__":
    unittest.main()
