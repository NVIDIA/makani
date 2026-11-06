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

import copy
import os
import sys
import contextlib
import unittest
from unittest import mock

import torch
import torch.nn as nn

from makani.models.noise import DiffusionNoiseS2, IsotropicGaussianRandomFieldS2
from makani.models.noise_controller import NoiseController, NoiseSnapshot, run_eager
from makani.models.parametrizations import ConstraintsWrapper
from makani.models.model_input import INPUT_NOISE
from makani.models.stepper import StepWrapper
from makani.utils.dataloaders.batch import (
    INPUT,
    INPUT_DYNAMIC_FEATURES,
    INPUT_TIMESTAMP,
    TARGET,
    TARGET_DYNAMIC_FEATURES,
    TARGET_TIMESTAMP,
    ZENITH,
)
from makani.utils.dataloaders.rollout import Rollout

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed
from .test_rollout import _batch, _NOISE, _params, _TinyNet

_H, _W = 16, 32
_B, _C, _T = 2, 1, 1


def _white(batch_size=_B, seed=333, learnable=False):
    return IsotropicGaussianRandomFieldS2(
        img_shape=(_H, _W), batch_size=batch_size, num_channels=_C, num_time_steps=_T, seed=seed, learnable=learnable
    )


def _diffusion(batch_size=_B, seed=333):
    return DiffusionNoiseS2(img_shape=(_H, _W), batch_size=batch_size, num_channels=_C, num_time_steps=_T, seed=seed)


def _samples(controller):
    return {name: controller.sample(name).clone() for name in controller}


class TestNoiseController(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(1)

    def test_without_processes_nothing_happens(self):
        noise = NoiseController()
        self.assertFalse(noise.is_stochastic)
        self.assertIsNone(noise.batch_size)
        noise.advance(replace_state=True, batch_size=4)
        noise.set_rng(seed=1)
        snapshot = noise.snapshot()
        self.assertEqual(dict(snapshot.states), {})
        self.assertIsNone(snapshot.batch_size)
        noise.restore(snapshot, batch_size=4)
        # nothing changed, so nothing for a rollout to detect either
        self.assertEqual(noise.version, 0)

    def test_advance_draws_a_new_state_and_counts_it(self):
        noise = NoiseController({"white": _white(), "diffusion": _diffusion()})
        self.assertTrue(noise.is_stochastic)
        self.assertEqual(list(noise), ["white", "diffusion"])

        noise.advance(replace_state=True)
        first = _samples(noise)
        first_version = noise.version
        noise.advance()
        second = _samples(noise)

        self.assertNotIn(noise.version, (0, first_version))
        for name in noise:
            with self.subTest(process=name):
                self.assertEqual(tuple(first[name].shape), (_B, _T, _C, _H, _W))
                self.assertFalse(torch.equal(first[name], second[name]))

    def test_the_sample_is_read_not_drawn(self):
        # the prediction may read the sample more than once, e.g. in a recompute under activation checkpointing
        noise = NoiseController({"diffusion": _diffusion()})
        noise.advance(replace_state=True)
        version = noise.version
        self.assertTrue(torch.equal(noise.sample("diffusion"), noise.sample("diffusion")))
        self.assertEqual(noise.version, version)

    def test_resizing_stateful_noise_needs_a_fresh_state(self):
        noise = NoiseController({"white": _white(), "diffusion": _diffusion()})
        with self.assertRaisesRegex(RuntimeError, "diffusion"):
            noise.advance(batch_size=4)
        # refused before any process was touched
        self.assertEqual(noise.batch_size, _B)
        self.assertEqual(noise.version, 0)

        noise.advance(replace_state=True, batch_size=4)
        self.assertEqual(noise.batch_size, 4)
        noise.advance(batch_size=4)

    def test_memoryless_noise_may_be_resized(self):
        noise = NoiseController({"white": _white()})
        noise.advance(batch_size=4)
        self.assertEqual(noise.batch_size, 4)

    def test_restore_rewinds_state_and_generators(self):
        noise = NoiseController({"white": _white(), "diffusion": _diffusion()})
        noise.advance(replace_state=True)
        snapshot = noise.snapshot()
        expected = []
        for _ in range(3):
            noise.advance()
            expected.append(_samples(noise))

        noise.restore(snapshot, batch_size=_B)
        for k in range(3):
            noise.advance()
            for name, sample in _samples(noise).items():
                with self.subTest(step=k, process=name):
                    self.assertTrue(torch.equal(sample, expected[k][name]))

    def test_snapshot_is_detached_and_does_not_follow_the_state(self):
        noise = NoiseController({"diffusion": _diffusion()})
        noise.advance(replace_state=True)
        snapshot = noise.snapshot()
        self.assertIsInstance(snapshot, NoiseSnapshot)
        before = snapshot.states["diffusion"].tensor.clone()
        noise.advance()
        self.assertTrue(torch.equal(snapshot.states["diffusion"].tensor, before))
        self.assertFalse(snapshot.states["diffusion"].tensor.requires_grad)
        self.assertEqual(snapshot.batch_size, _B)

    def test_restore_refuses_another_model_or_batch(self):
        noise = NoiseController({"white": _white(), "diffusion": _diffusion()})
        snapshot = noise.snapshot()
        with self.subTest("other processes"), self.assertRaisesRegex(ValueError, "processes"):
            NoiseController({"white": _white()}).restore(snapshot)
        with self.subTest("other batch"), self.assertRaisesRegex(ValueError, "batch"):
            noise.restore(snapshot, batch_size=_B + 1)

    def test_a_failed_restore_leaves_the_noise_as_it_was(self):
        noise = NoiseController({"white": _white(), "diffusion": _diffusion()})
        noise.advance(replace_state=True)
        # same names and batch, but the diffusion state has another layout; white is written before it fails
        other = NoiseController(
            {
                "white": _white(seed=7),
                "diffusion": DiffusionNoiseS2(
                    img_shape=(_H, _W), batch_size=_B, num_channels=_C + 1, num_time_steps=_T
                ),
            }
        )
        other.advance(replace_state=True)
        reference = noise.snapshot()
        version = noise.version

        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            noise.restore(other.snapshot(), batch_size=_B)

        # the state, the generators and the version are those before the restore
        self.assertEqual(noise.version, version)
        noise.advance()
        after_failure = _samples(noise)
        noise.restore(reference)
        noise.advance()
        for name, sample in _samples(noise).items():
            with self.subTest(process=name):
                self.assertTrue(torch.equal(sample, after_failure[name]))

    def test_set_rng_reproduces_a_sequence(self):
        noise = NoiseController({"white": _white(), "diffusion": _diffusion()})
        runs = []
        for _ in range(2):
            noise.set_rng(seed=7)
            noise.advance(replace_state=True)
            noise.advance()
            runs.append(_samples(noise))
        for name in noise:
            with self.subTest(process=name):
                self.assertTrue(torch.equal(runs[0][name], runs[1][name]))

    def test_set_rng_seeds_a_single_process_with_the_seed(self):
        # a model with only the input noise is seeded exactly as before the controller existed
        process, reference = _white(), _white()
        NoiseController({"white": process}).set_rng(seed=7)
        reference.set_rng(seed=7)
        reference.reset()
        process.update(replace_state=True)
        reference.update(replace_state=True)
        self.assertTrue(torch.equal(process(), reference()))

    def test_set_rng_without_reset_keeps_the_state(self):
        # continue a sequence under a new seed: the state stays, only the generators are re-seeded
        noise = NoiseController({"diffusion": _diffusion()})
        noise.advance(replace_state=True)
        noise.advance()
        state = noise.snapshot().states["diffusion"].tensor
        self.assertTrue(state.abs().sum() > 0)

        noise.set_rng(seed=7, reset=False)
        self.assertTrue(torch.equal(noise.snapshot().states["diffusion"].tensor, state))
        noise.set_rng(seed=7)
        self.assertEqual(noise.snapshot().states["diffusion"].tensor.abs().sum(), 0)

    def test_set_rng_takes_keywords_only(self):
        # the package and the controller once took (seed, reset) in different orders
        with self.assertRaises(TypeError):
            NoiseController({"white": _white()}).set_rng(7)

    def test_set_rng_gives_the_processes_their_own_streams(self):
        noise = NoiseController({"a": _white(), "b": _white()})
        noise.set_rng(seed=7)
        noise.advance(replace_state=True)
        self.assertFalse(torch.equal(noise.sample("a"), noise.sample("b")))


@contextlib.contextmanager
def _Rank(spatial=0, ensemble=0, batch=0, spatial_size=2, ensemble_size=2, batch_size=1):
    """Mocks the comm of the noise module with the ranks of one grid point."""
    ranks = {"spatial": spatial, "ensemble": ensemble, "batch": batch, "data": batch * ensemble_size + ensemble}
    sizes = {"spatial": spatial_size, "ensemble": ensemble_size, "batch": batch_size}
    sizes["data"] = ensemble_size * batch_size
    with (
        mock.patch("makani.models.noise.comm.get_rank", side_effect=ranks.get),
        mock.patch("makani.models.noise.comm.get_size", side_effect=sizes.get),
    ):
        yield


class TestSeedingOverRanks(unittest.TestCase):
    """set_rng lays the seed out over the ranks as the processes were built, so that the ranks keep their streams."""

    def _sample_after_set_rng(self, process, **rank):
        noise = NoiseController({"white": process})
        with _Rank(**rank):
            noise.set_rng(seed=7)
        noise.advance(replace_state=True)
        return noise.sample("white")

    def test_spatial_shards_draw_their_own_streams(self):
        a = self._sample_after_set_rng(_white(), spatial=0)
        b = self._sample_after_set_rng(_white(), spatial=1)
        self.assertFalse(torch.equal(a, b))

    def test_data_parallel_ranks_draw_their_own_streams(self):
        a = self._sample_after_set_rng(_white(), ensemble=0)
        b = self._sample_after_set_rng(_white(), ensemble=1)
        self.assertFalse(torch.equal(a, b))

    def test_centered_pairs_stay_opposite(self):
        # the members 0 and 1 of an ensemble pair, built as the preprocessor builds them
        samples = []
        for ensemble in (0, 1):
            process = IsotropicGaussianRandomFieldS2(
                img_shape=(_H, _W), batch_size=_B, num_channels=_C, reflect=(ensemble == 0), centered=True
            )
            samples.append(self._sample_after_set_rng(process, ensemble=ensemble))
        self.assertTrue(torch.equal(samples[0], -samples[1]))

    def test_serial_seed_is_the_seed(self):
        process = _white()
        NoiseController({"white": process}).set_rng(seed=7)
        self.assertEqual(process.rng_cpu.initial_seed(), 7)


class TestGeneratorState(unittest.TestCase):
    """The generators alone, as the checkpoint keeps them."""

    def test_restored_generators_continue_the_draws(self):
        noise = NoiseController({"white": _white(), "diffusion": _diffusion()})
        noise.advance(replace_state=True)
        state = noise.get_rng_state()
        noise.advance(replace_state=True)
        expected = _samples(noise)

        noise.advance(replace_state=True)
        noise.set_rng_state(state)
        noise.advance(replace_state=True)
        for name, sample in _samples(noise).items():
            with self.subTest(process=name):
                self.assertTrue(torch.equal(sample, expected[name]))

    def test_restore_moves_the_version_on(self):
        noise = NoiseController({"white": _white()})
        version = noise.version
        noise.set_rng_state(noise.get_rng_state())
        self.assertNotEqual(noise.version, version)

    def test_other_processes_refused(self):
        noise = NoiseController({"white": _white()})
        with self.assertRaisesRegex(ValueError, "processes"):
            noise.set_rng_state({"other": _white().get_rng_state()})


@contextlib.contextmanager
def _layout(**sizes):
    """Mocks the comm of the checkpoint helpers with a decomposition, every rank 0."""
    with (
        mock.patch("makani.utils.checkpoint_helpers.comm.get_comm_names", return_value=list(sizes)),
        mock.patch("makani.utils.checkpoint_helpers.comm.get_size", side_effect=lambda name: sizes.get(name, 1)),
        mock.patch("makani.utils.checkpoint_helpers.comm.get_rank", return_value=0),
    ):
        yield


class TestGeneratorCheckpoint(unittest.TestCase):
    """The generators as the checkpoint stores them, restored only into the decomposition they were taken in."""

    _SPATIAL = {"model": 2, "spatial": 2, "h": 2, "w": 1, "matmul": 1, "data": 1, "ensemble": 1, "batch": 1}
    _MATMUL = {"model": 2, "spatial": 1, "h": 1, "w": 1, "matmul": 2, "data": 1, "ensemble": 1, "batch": 1}

    def test_same_decomposition_restores(self):
        from makani.utils.checkpoint_helpers import gather_noise_rng_state, scatter_noise_rng_state

        noise = NoiseController({"white": _white()})
        with _layout(**self._SPATIAL):
            stored = gather_noise_rng_state(noise)
            restored = scatter_noise_rng_state(stored)
        self.assertTrue(torch.equal(restored["white"][0], noise.get_rng_state()["white"][0]))

    def test_same_totals_other_split_refused(self):
        # two spatial ranks draw different streams, two matmul ranks the same: the totals alone do not tell them apart
        from makani.utils.checkpoint_helpers import gather_noise_rng_state, scatter_noise_rng_state

        noise = NoiseController({"white": _white()})
        with _layout(**self._SPATIAL):
            stored = gather_noise_rng_state(noise)
        with _layout(**self._MATMUL), self.assertRaisesRegex(ValueError, "decomposition"):
            scatter_noise_rng_state(stored)

    def test_driver_keeps_fresh_streams_on_another_decomposition(self):
        from makani.utils.checkpoint_helpers import gather_noise_rng_state
        from makani.utils.driver import Driver

        noise = NoiseController({"white": _white()})
        with _layout(**self._SPATIAL):
            stored = gather_noise_rng_state(noise)
        noise.advance(replace_state=True)
        current = noise.get_rng_state()["white"][0]
        with _layout(**self._MATMUL):
            Driver._restore_noise_rng_state({"noise_rng_state": stored}, noise)
        self.assertTrue(torch.equal(noise.get_rng_state()["white"][0], current))


class _LatentNoiseNet(nn.Module):
    """A network that draws noise of its own: a learnable field added to its hidden features."""

    def __init__(self, in_chans, out_chans, hidden=_C):
        super().__init__()
        self.net = _TinyNet(in_chans, out_chans, hidden=hidden)
        self.latent_noise = IsotropicGaussianRandomFieldS2(
            img_shape=(_H, _W), batch_size=_B, num_channels=hidden, num_time_steps=1, seed=4242, learnable=True
        )

    def noise_processes(self):
        return {"latent": self.latent_noise}

    def forward(self, x):
        # read the current state, the model never advances its noise itself
        eta = run_eager(self.latent_noise).squeeze(1)
        return self.net.fc2(torch.tanh(self.net.fc1(x) + eta))


def _latent_wrapper(input_noise=None, constrained=False):
    params = _params(0, False, input_noise)
    params.img_shape_x = params.img_shape_x_resampled = _H
    params.img_shape_y = params.img_shape_y_resampled = _W
    n_noise = input_noise["n_channels"] if input_noise is not None else 0
    set_seed(333)

    def net():
        return _LatentNoiseNet(3 + 1 + n_noise, 3)

    def constrained_net():
        # as the model registry wraps a network for params.constraints
        return ConstraintsWrapper([], ["u", "v", "t"], None, None, model_handle=net)

    return StepWrapper(params, constrained_net if constrained else net)


class TestModelNoise(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(1)

    def test_the_controller_refers_to_the_processes_the_model_owns(self):
        noise = _NOISE["white_learnable"]
        params = _params(1, False, noise)
        wrapper = StepWrapper(params, lambda: _TinyNet(2 * (3 + 1 + 1), 3))

        self.assertEqual(list(wrapper.noise), [INPUT_NOISE])
        self.assertIs(wrapper.noise.process(INPUT_NOISE), wrapper.input_adapter.input_noise.process)
        # the controller is no submodule, so the processes are registered once, where they live
        self.assertNotIn("noise", dict(wrapper.named_children()))
        keys = set(wrapper.state_dict())
        self.assertTrue(any(key.startswith("input_adapter.input_noise.") for key in keys))
        self.assertFalse(any(key.startswith("noise.") for key in keys))
        names = [name for name, _ in wrapper.named_parameters()]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(any(name.startswith("input_adapter.input_noise.") for name in names))

    def test_a_copy_of_the_model_drives_its_own_noise(self):
        noise = _NOISE["diffusion"]
        wrapper = StepWrapper(_params(1, False, noise), lambda: _TinyNet(2 * (3 + 1 + 1), 3))
        twin = copy.deepcopy(wrapper)
        self.assertIs(twin.noise.process(INPUT_NOISE), twin.input_adapter.input_noise.process)
        self.assertIsNot(twin.noise.process(INPUT_NOISE), wrapper.input_adapter.input_noise.process)

    def test_the_network_lists_its_own_noise(self):
        wrapper = _latent_wrapper(_NOISE["white"])
        self.assertEqual(list(wrapper.noise), [INPUT_NOISE, "model.latent"])
        self.assertIs(wrapper.noise.process("model.latent"), wrapper.model.latent_noise)

    def test_a_constraints_wrapper_without_network_noise_has_none(self):
        bare = ConstraintsWrapper([], ["u", "v", "t"], None, None)
        self.assertEqual(bare.noise_processes(), {})
        around = ConstraintsWrapper([], ["u", "v", "t"], None, None, model_handle=lambda: _TinyNet(3, 3))
        self.assertEqual(around.noise_processes(), {})

    def test_a_constrained_network_lists_its_own_noise(self):
        wrapper = _latent_wrapper(constrained=True)
        self.assertEqual(list(wrapper.noise), ["model.latent"])
        self.assertIs(wrapper.noise.process("model.latent"), wrapper.model.model.latent_noise)

        # the rollout drives it, as for the network alone
        wrapper.eval()
        batch = _batch(n_history=0, n_future=2, grid=(_H, _W))
        with torch.no_grad():
            wrapper.noise.set_rng(seed=5)
            first = wrapper(Rollout(batch, noise=wrapper.noise))
            wrapper.noise.set_rng(seed=5)
            again = wrapper(Rollout(batch, noise=wrapper.noise))
        self.assertTrue(torch.equal(first, again))

    def test_a_step_of_another_model_is_refused(self):
        # two models whose noise was changed equally often: the step of one is no step of the other
        wrapper, other = _latent_wrapper(), _latent_wrapper()
        wrapper.eval()
        other.eval()
        batch = _batch(n_history=0, n_future=1, grid=(_H, _W))
        wrapper.noise.set_rng(seed=5)
        other.noise.set_rng(seed=5)
        step = next(iter(Rollout(batch, noise=other.noise)))
        wrapper.noise.advance(replace_state=True)
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "does not drive the noise of this model"):
            wrapper.predict_step(step)

    def test_a_rollout_drives_latent_noise_and_trains_it(self):
        wrapper = _latent_wrapper()
        wrapper.train()
        batch = _batch(n_history=0, n_future=2, grid=(_H, _W))

        rollout = Rollout(batch, noise=wrapper.noise)
        seen = []
        versions = []
        preds = []
        for step in rollout:
            versions.append(step.noise_version)
            seen.append(wrapper.noise.sample("model.latent").detach().clone())
            pred = wrapper.predict_step(step)
            preds.append(pred)
            rollout.advance(pred)
        loss = torch.square(torch.cat(preds, dim=1) - batch[TARGET]).mean()
        loss.backward()

        # every step its own noise, and the gradient reaches the parameters of the latent noise
        self.assertEqual(len(set(versions)), 3)
        self.assertFalse(torch.equal(seen[0], seen[1]))
        grads = {name: p.grad for name, p in wrapper.named_parameters() if p.grad is not None}
        self.assertTrue(any(name.startswith("model.latent_noise.") for name in grads))

    def test_checkpointing_is_refused_for_latent_noise(self):
        # the recompute of a checkpointed step would read the latent noise after the rollout advanced it
        wrapper = _latent_wrapper()
        wrapper.multistep_checkpoint = True
        wrapper.train()
        batch = _batch(n_history=0, n_future=2, grid=(_H, _W))
        with self.assertRaisesRegex(RuntimeError, "noise processes of their own"):
            wrapper(Rollout(batch, noise=wrapper.noise))

        # nor when the caller steps the rollout itself
        rollout = Rollout(batch, noise=wrapper.noise)
        with self.assertRaisesRegex(RuntimeError, "noise processes of their own"):
            wrapper.predict_step(next(iter(rollout)), use_checkpoint=True)

        # without checkpointing, or for a single step, it runs
        wrapper.multistep_checkpoint = False
        wrapper(Rollout(batch, noise=wrapper.noise))

    def test_a_split_rollout_continues_the_latent_noise(self):
        wrapper = _latent_wrapper()
        wrapper.eval()
        batch = _batch(n_history=0, n_future=3, grid=(_H, _W))

        with torch.no_grad():
            wrapper.noise.set_rng(seed=5)
            whole = wrapper(Rollout(batch, noise=wrapper.noise))

            wrapper.noise.set_rng(seed=5)
            first = Rollout(batch, n_steps=2, noise=wrapper.noise)
            head = wrapper(first)
            # the window after the second step: its prediction, with the features and time of its target
            rest = {
                INPUT: head[:, -1:],
                INPUT_DYNAMIC_FEATURES: {ZENITH: batch[TARGET_DYNAMIC_FEATURES][ZENITH][:, 1:2]},
                INPUT_TIMESTAMP: batch[TARGET_TIMESTAMP][:, 1:2],
                TARGET: batch[TARGET][:, 2:],
                TARGET_DYNAMIC_FEATURES: {ZENITH: batch[TARGET_DYNAMIC_FEATURES][ZENITH][:, 2:]},
                TARGET_TIMESTAMP: batch[TARGET_TIMESTAMP][:, 2:],
            }
            tail = wrapper(Rollout(rest, noise=wrapper.noise, noise_state=first.noise_state()))

        self.assertTrue(torch.equal(torch.cat([head, tail], dim=1), whole))


if __name__ == "__main__":
    unittest.main()
