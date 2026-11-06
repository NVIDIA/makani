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

"""The input of a network, ModelInput, and its packing for flat networks, FlatInputAdapter."""

import os
import sys
import unittest
from collections import OrderedDict
import tempfile

import torch
from parameterized import parameterized
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.models.model_input import INPUT_NOISE, FlatInputAdapter, InputNoise, ModelInput
from makani.models.noise_controller import NoiseController
from makani.models.preprocessor import HistoryStats, Preprocessor2D
from makani.models.stepper import StepWrapper
from makani.utils.checkpoint_helpers import upgrade_state_dict
from makani.utils.driver import Driver
from makani.utils.dataloaders.batch import ZENITH
from makani.utils.functions import flatten_history

from .testutils import compare_tensors, get_default_parameters, set_seed

_CONCATENATE = {"type": "dummy", "mode": "concatenate", "n_channels": 2}
_PERTURB = {"type": "dummy", "mode": "perturb", "perturb_channels": ["u10m"]}


def _params(n_history=0, add_zenith=False, **overrides):
    params = get_default_parameters()
    params.n_history = n_history
    params.add_zenith = add_zenith
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def _advance(adapter, batch_size):
    """Advance the input noise of the adapter to a fresh state, as a rollout does at its first step."""
    noise = NoiseController(adapter.noise_processes())
    noise.advance(replace_state=True, batch_size=batch_size)


def _constant_noise(adapter, params, channels):
    """Replace the process of the input noise with a non-zero constant one, so that where its draw lands is visible."""
    from makani.models.noise import DummyNoiseS2

    adapter.input_noise.process = DummyNoiseS2(
        img_shape=[params.img_shape_x, params.img_shape_y],
        batch_size=params.batch_size,
        num_channels=channels,
        num_time_steps=params.n_history + 1,
        mode="constant_random",
    )
    adapter.input_noise.process.update()
    return adapter.input_noise.process()


class _Inputs:
    """Fields and inputs on the grid of the default parameters."""

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.B = self.params.batch_size
        self.C = self.params.N_in_channels
        self.H = self.params.img_shape_x
        self.W = self.params.img_shape_y

    def _field(self, T=1, C=None, B=None):
        return torch.randn(B or self.B, T, C or self.C, self.H, self.W)

    def _input(self, params, state, dynamic_features=None, static=None):
        pp = Preprocessor2D(params)
        stats = pp.compute_history_stats(state)
        return ModelInput(
            state=pp.history_normalize(state, stats),
            stats=stats,
            dynamic_features=dynamic_features or {},
            static=static,
        )

    def _extra_normalized(self, params, x):
        # the dynamic features and the noise, normalized with statistics of their own
        pp = Preprocessor2D(params)
        return pp.history_normalize(x, pp.compute_history_stats(x))


class TestFlatInputAdapter(_Inputs, unittest.TestCase):

    # ------------------------------------------------------------ the channel stack

    def test_without_extras_only_folds_the_time_steps(self, verbose=False):
        params = _params(n_history=1)
        model_input = self._input(params, self._field(2))
        (x,) = FlatInputAdapter(params).pack(model_input)
        self.assertTrue(compare_tensors("flattened", x, flatten_history(model_input.state), verbose=verbose))

    def test_static_features_follow_all_time_steps(self, verbose=False):
        params = _params(n_history=1, add_grid=True, gridtype="raw")
        static = Preprocessor2D(params).get_static_features()
        model_input = self._input(params, self._field(2), static=static)
        (x,) = FlatInputAdapter(params).pack(model_input)
        self.assertEqual(tuple(x.shape), (self.B, 2 * self.C + 2, self.H, self.W))
        self.assertTrue(
            compare_tensors("steps", x[:, : 2 * self.C], flatten_history(model_input.state), verbose=verbose)
        )
        self.assertTrue(
            compare_tensors("static", x[:, 2 * self.C :], static.expand(self.B, -1, -1, -1), verbose=verbose)
        )

    def test_needs_the_time_axis(self):
        params = _params()
        model_input = self._input(params, self._field(1))
        with self.assertRaises(RuntimeError):
            FlatInputAdapter(params).pack(model_input.replace(state=model_input.state.squeeze(1)))

    def test_order_is_state_features_noise_per_step(self, verbose=False):
        """[state | dynamic features | noise] per time step, oldest first, the order the networks were trained with."""
        params = _params(n_history=1, add_zenith=True, history_normalization_mode="mean", input_noise=_CONCATENATE)
        adapter = FlatInputAdapter(params)
        zenith = self._field(2, C=1)
        noise = _constant_noise(adapter, params, 2)
        model_input = self._input(params, self._field(2), {ZENITH: zenith})

        (x,) = adapter.pack(model_input)

        extra = self._extra_normalized(params, torch.cat([zenith, noise], dim=2))
        expected = flatten_history(torch.cat([model_input.state, extra], dim=2))
        self.assertTrue(compare_tensors("packed", x, expected, verbose=verbose))
        self.assertEqual(x.shape[1], 2 * (self.C + 1 + 2))

    def test_the_state_is_passed_on_as_it_is(self):
        """The state keeps its own normalization, whatever is appended to it."""
        params = _params(add_zenith=True, history_normalization_mode="mean", input_noise=_CONCATENATE)
        adapter = FlatInputAdapter(params)
        _advance(adapter, self.B)
        model_input = self._input(params, self._field(1), {ZENITH: self._field(1, C=1)})
        (x,) = adapter.pack(model_input)
        self.assertTrue(torch.equal(x[:, : self.C], model_input.state[:, 0]))

    def test_a_dynamic_feature_is_normalized_with_its_own_statistics(self, verbose=False):
        params = _params(add_zenith=True, history_normalization_mode="mean")
        zenith = 3.0 + 2.0 * self._field(1, C=1)
        (x,) = FlatInputAdapter(params).pack(self._input(params, self._field(1), {ZENITH: zenith}))
        expected = self._extra_normalized(params, zenith)[:, 0]
        self.assertTrue(compare_tensors("zenith", x[:, self.C :], expected, verbose=verbose))

    def test_concatenated_noise_adds_its_channels(self):
        params = _params(input_noise=_CONCATENATE)
        adapter = FlatInputAdapter(params)
        _advance(adapter, self.B)
        (x,) = adapter.pack(self._input(params, self._field(1)))
        self.assertEqual(x.shape[1], self.C + 2)

    def test_the_features_have_to_be_those_of_the_model(self):
        params = _params(add_zenith=True)
        adapter = FlatInputAdapter(params)
        with self.assertRaisesRegex(ValueError, "zenith"):
            adapter.pack(self._input(params, self._field(1)))
        with self.assertRaisesRegex(ValueError, "zenith"):
            adapter.pack(
                self._input(params, self._field(1), {ZENITH: self._field(1, C=1), "other": self._field(1, C=1)})
            )
        with self.assertRaisesRegex(ValueError, "zenith"):
            FlatInputAdapter(_params()).pack(self._input(params, self._field(1), {ZENITH: self._field(1, C=1)}))

    def test_batch_mismatches_are_refused(self):
        params = _params(add_zenith=True)
        with self.assertRaisesRegex(RuntimeError, "different batch size"):
            FlatInputAdapter(params).pack(
                self._input(params, self._field(1, B=self.B + 1), {ZENITH: self._field(1, C=1)})
            )
        # the noise is sized for the batch it was advanced at
        params = _params(input_noise=_CONCATENATE)
        adapter = FlatInputAdapter(params)
        _advance(adapter, self.B)
        with self.assertRaisesRegex(RuntimeError, "the noise has batch"):
            adapter.pack(self._input(params, self._field(1, B=self.B + 2)))

    def test_the_gradient_flows_to_the_state(self):
        params = _params(add_zenith=True, input_noise=_CONCATENATE)
        adapter = FlatInputAdapter(params)
        _advance(adapter, self.B)
        state = self._field(1).requires_grad_(True)
        model_input = ModelInput(state=state, stats=HistoryStats(), dynamic_features={ZENITH: self._field(1, C=1)})
        adapter.pack(model_input)[0].sum().backward()
        self.assertTrue(torch.all(state.grad == 1.0).item())

    # ------------------------------------------------------------ the path of a stochastic interpolant

    def test_the_path_follows_the_window_with_a_time_channel(self):
        params = _params()
        model_input = self._input(params, self._field(1))
        path_state, time = self._field(1, C=3)[:, 0], torch.rand(self.B)
        (x,) = FlatInputAdapter(params).pack(model_input.replace(path_state=path_state, time=time))
        self.assertEqual(x.shape[1], self.C + 3 + 1)
        self.assertTrue(torch.equal(x[:, self.C : self.C + 3], path_state))
        self.assertTrue(torch.equal(x[:, -1], time.reshape(-1, 1, 1).expand(-1, self.H, self.W)))

    def test_a_time_aware_network_takes_the_time_as_an_argument(self):
        params = _params()
        model_input = self._input(params, self._field(1))
        path_state, time = self._field(1, C=3)[:, 0], torch.rand(self.B)
        x, s = FlatInputAdapter(params, time_aware=True).pack(model_input.replace(path_state=path_state, time=time))
        self.assertEqual(x.shape[1], self.C + 3)
        self.assertIs(s, time)

    # ------------------------------------------------------------ the input noise

    def test_without_input_noise_there_is_no_process(self):
        adapter = FlatInputAdapter(_params())
        self.assertIsNone(adapter.input_noise)
        self.assertEqual(adapter.noise_processes(), {})

    def test_the_input_noise_is_its_one_process(self):
        adapter = FlatInputAdapter(_params(input_noise=_CONCATENATE))
        self.assertEqual(list(adapter.noise_processes()), [INPUT_NOISE])
        self.assertIs(adapter.noise_processes()[INPUT_NOISE], adapter.input_noise.process)
        self.assertIsNone(FlatInputAdapter(_params(input_noise=_CONCATENATE), with_input_noise=False).input_noise)


class TestInputNoise(_Inputs, unittest.TestCase):

    def test_concatenated_noise_returns_the_draw(self):
        params = _params(input_noise=_CONCATENATE)
        adapter = FlatInputAdapter(params)
        noise = _constant_noise(adapter, params, 2)
        model_input = self._input(params, self._field(1))
        state, draw = adapter.input_noise(model_input)
        self.assertIs(state, model_input.state)
        self.assertTrue(torch.equal(draw, noise))

    def test_perturbing_noise_adds_no_channels(self):
        params = _params(input_noise=_PERTURB)
        adapter = FlatInputAdapter(params)
        _advance(adapter, self.B)
        (x,) = adapter.pack(self._input(params, self._field(1)))
        self.assertEqual(x.shape[1], self.C)

    def test_zero_perturbation_leaves_the_state(self):
        params = _params(history_normalization_mode="mean", input_noise=_PERTURB)
        adapter = FlatInputAdapter(params)
        _advance(adapter, self.B)
        model_input = self._input(params, self._field(1))
        (x,) = adapter.pack(model_input)
        self.assertTrue(torch.equal(x, flatten_history(model_input.state)))

    def test_perturbation_is_in_data_space(self, verbose=False):
        """The noise perturbs the normalized state as it would the state before its normalization."""
        params = _params(history_normalization_mode="mean", input_noise=_PERTURB)
        adapter = FlatInputAdapter(params)
        noise = _constant_noise(adapter, params, 1)
        state = self._field(1)
        model_input = self._input(params, state)

        (x,) = adapter.pack(model_input)

        channel = params.channel_names.index("u10m")
        perturbed = state.clone()
        perturbed[:, :, channel] += noise[:, :, 0]
        # normalized with the statistics of the state without the noise
        expected = flatten_history(Preprocessor2D(params).history_normalize(perturbed, model_input.stats))
        self.assertTrue(compare_tensors("perturbed", x, expected, atol=1e-6, rtol=1e-5, verbose=verbose))
        others = [c for c in range(self.C) if c != channel]
        self.assertTrue(torch.equal(x[:, others], model_input.state[:, 0, others]))

    def test_the_draw_is_repeated_with_its_samples(self):
        """Each sample of a repeated input gets the noise of the sample it repeats."""
        params = _params(input_noise=_CONCATENATE)
        adapter = FlatInputAdapter(params)
        noise = _constant_noise(adapter, params, 2)
        model_input = self._input(params, self._field(1)).repeat_interleave(3)
        _, draw = adapter.input_noise(model_input)
        self.assertTrue(torch.equal(draw, noise.repeat_interleave(3, dim=0)))

    def test_the_process_is_seeded_with_the_seed_of_this_rank(self):
        from makani.models.noise import IsotropicGaussianRandomFieldS2, noise_base_seed

        layer = InputNoise.from_params(_params(input_noise={"type": "white", "mode": "concatenate", "n_channels": 1}))
        seed, reflect = noise_base_seed()
        reference = IsotropicGaussianRandomFieldS2(
            img_shape=(self.H, self.W), batch_size=1, num_channels=1, num_time_steps=1, seed=seed, reflect=reflect
        )
        samples = []
        for process in (layer.process, reference):
            noise = NoiseController({INPUT_NOISE: process})
            noise.advance(replace_state=True)
            samples.append(noise.sample(INPUT_NOISE))
        self.assertTrue(torch.equal(*samples))

    def test_without_configuration_there_is_none(self):
        self.assertIsNone(InputNoise.from_params(_params()))


class TestModelInput(unittest.TestCase):

    def test_repeat_interleave_repeats_the_samples(self):
        B, T, C = 2, 1, 3
        stats = HistoryStats(mean=torch.randn(B, 1, C, 1, 1), std=torch.rand(B, 1, C, 1, 1))
        static = torch.randn(1, 2, 4, 8)
        model_input = ModelInput(
            state=torch.randn(B, T, C, 4, 8),
            stats=stats,
            dynamic_features={ZENITH: torch.randn(B, T, 1, 4, 8)},
            static=static,
        )
        repeated = model_input.repeat_interleave(3)
        self.assertTrue(torch.equal(repeated.state, model_input.state.repeat_interleave(3, dim=0)))
        self.assertTrue(torch.equal(repeated.stats.std, stats.std.repeat_interleave(3, dim=0)))
        self.assertTrue(
            torch.equal(
                repeated.dynamic_features[ZENITH], model_input.dynamic_features[ZENITH].repeat_interleave(3, dim=0)
            )
        )
        self.assertIs(repeated.static, static)
        self.assertIsNone(repeated.path_state)
        # the noise of a sample is repeated with it
        self.assertEqual(repeated.repeats, 3)
        self.assertEqual(repeated.repeat_interleave(2).repeats, 6)
        noise = torch.randn(B, 1)
        self.assertTrue(torch.equal(repeated.repeat_noise(noise), noise.repeat_interleave(3, dim=0)))
        self.assertIs(model_input.repeat_noise(noise), noise)

    def test_statistics_without_a_batch_are_kept(self):
        stats = HistoryStats(mean=torch.zeros(1, 1, 1, 1, 1), std=torch.ones(1, 1, 1, 1, 1))
        model_input = ModelInput(state=torch.randn(2, 1, 3, 4, 8), stats=stats, dynamic_features={})
        self.assertIs(model_input.repeat_interleave(3).stats.mean, stats.mean)


class _StructuredNet(nn.Module):
    """A network that takes the ModelInput itself, and records it with the draw of its own input noise."""

    takes_model_input = True

    def __init__(self, params, with_noise=True):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.out_chans = params.N_out_channels
        self.input_noise = InputNoise.from_params(params) if with_noise else None
        self.seen = []

    def noise_processes(self):
        return {"input_noise": self.input_noise.process} if self.input_noise is not None else {}

    def forward(self, model_input):
        draw = self.input_noise(model_input)[1] if self.input_noise is not None else None
        self.seen.append((model_input, draw))
        return self.scale * model_input.state[:, -1, : self.out_chans]


class _FlatNet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.fc = nn.Conv2d(in_chans, out_chans, 1)

    def forward(self, x):
        return self.fc(x)


class TestStepWrapperInput(unittest.TestCase):

    def setUp(self):
        set_seed(333)

    def _window(self, params):
        x = torch.randn(params.batch_size, 1, params.N_in_channels, params.img_shape_x, params.img_shape_y)
        zenith = torch.randn(params.batch_size, 1, 1, params.img_shape_x, params.img_shape_y)
        return x, zenith

    def test_a_structured_network_gets_the_model_input_and_owns_its_noise(self):
        params = _params(add_zenith=True, input_noise=_CONCATENATE)
        wrapper = StepWrapper(params, lambda: _StructuredNet(params))
        # the noise is the network's, the adapter has none
        self.assertIsNone(wrapper.input_adapter.input_noise)
        self.assertEqual(list(wrapper.noise), ["model.input_noise"])
        self.assertIs(wrapper.noise.process("model.input_noise"), wrapper.model.input_noise.process)

        wrapper.noise.advance(replace_state=True, batch_size=params.batch_size)
        x, zenith = self._window(params)
        pred = wrapper.predict(x, {ZENITH: zenith})

        ((model_input, draw),) = wrapper.model.seen
        # the dynamic features as the data holds them, and the draw at the state the controller advanced to
        self.assertIs(model_input.dynamic_features[ZENITH], zenith)
        self.assertTrue(torch.equal(draw, wrapper.noise.sample("model.input_noise")))
        self.assertEqual(
            tuple(pred.shape), (params.batch_size, 1, params.N_out_channels, params.img_shape_x, params.img_shape_y)
        )

    def test_constraints_pass_the_model_input_on(self):
        # get_model wraps the network in a ConstraintsWrapper when constraints are configured
        from makani.models.parametrizations import ConstraintsWrapper

        params = _params(add_zenith=True)

        def constrained():
            return ConstraintsWrapper(
                [], params.channel_names, None, None, model_handle=lambda: _StructuredNet(params, with_noise=False)
            )

        wrapper = StepWrapper(params, constrained)
        self.assertTrue(wrapper.model.takes_model_input)
        x, zenith = self._window(params)
        wrapper.predict(x, {ZENITH: zenith})
        ((model_input, _),) = wrapper.model.model.seen
        self.assertIsInstance(model_input, ModelInput)
        self.assertIs(model_input.dynamic_features[ZENITH], zenith)

    def test_a_structured_network_has_to_take_the_configured_noise(self):
        params = _params(input_noise=_CONCATENATE)
        with self.assertRaisesRegex(ValueError, "lists no noise process"):
            StepWrapper(params, lambda: _StructuredNet(params, with_noise=False))

    def test_the_input_noise_lives_in_the_input_adapter(self):
        params = _params(input_noise={"type": "white", "mode": "concatenate", "n_channels": 1, "learnable": True})
        wrapper = StepWrapper(params, lambda: _FlatNet(params.N_in_channels + 1, params.N_out_channels))
        self.assertIs(wrapper.noise.process(INPUT_NOISE), wrapper.input_adapter.input_noise.process)
        names = [name for name, _ in wrapper.named_parameters()]
        # the parameters of the input noise lead those of the network, as they did in the preprocessor
        self.assertTrue(names[0].startswith("input_adapter.input_noise.process."))
        self.assertTrue(names[-1].startswith("model."))

    @parameterized.expand([("untagged", None), ("version_2", 2)])
    def test_a_checkpoint_of_an_earlier_version_restores(self, _, version):
        """The input noise was kept in the preprocessor, preprocessor.input_noise.*, before format version 3."""
        params = _params(input_noise={"type": "white", "mode": "concatenate", "n_channels": 1, "learnable": True})

        def make():
            return _FlatNet(params.N_in_channels + 1, params.N_out_channels)

        wrapper = StepWrapper(params, make)
        with torch.no_grad():
            for p in wrapper.parameters():
                p.add_(torch.randn_like(p))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt_mp{mp_rank}.tar")
            Driver.save_checkpoint(checkpoint_path=path, model=wrapper, checkpoint_mode="flexible")
            checkpoint = torch.load(path.format(mp_rank=0), weights_only=False)
            checkpoint["model_state"] = OrderedDict(
                (key.replace("input_adapter.input_noise.process.", "preprocessor.input_noise."), value)
                for key, value in checkpoint["model_state"].items()
            )
            self.assertTrue(any(key.startswith("preprocessor.input_noise.") for key in checkpoint["model_state"]))
            if version is None:
                del checkpoint["checkpoint_format"]
            else:
                checkpoint["checkpoint_format"]["version"] = version
            torch.save(checkpoint, path.format(mp_rank=0))

            restored = StepWrapper(params, make)
            Driver.restore_from_checkpoint(
                checkpoint_path=path, model=restored, checkpoint_mode="flexible", strict=True
            )
        for (name, p), q in zip(wrapper.named_parameters(), restored.parameters()):
            self.assertTrue(torch.equal(p, q), name)

    def test_the_upgrade_follows_a_wrapper_prefix(self):
        """Inside a wrapper, e.g. DDP, the keys carry its prefix."""
        params = _params(input_noise={"type": "white", "mode": "concatenate", "n_channels": 1, "learnable": True})
        wrapper = StepWrapper(params, lambda: _FlatNet(params.N_in_channels + 1, params.N_out_channels))
        old = {
            "module." + key.replace("input_adapter.input_noise.process.", "preprocessor.input_noise."): value
            for key, value in wrapper.state_dict().items()
        }
        holder = torch.nn.Module()
        holder.module = wrapper
        upgrade_state_dict(holder, old)
        self.assertEqual(set(old), {"module." + key for key in wrapper.state_dict()})


if __name__ == "__main__":
    unittest.main()
