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
Noise processes on spatially split ranks.

A noise process holds its spectral state split over the ranks, the degrees over h and the orders over w, and
transforms it to its part of the grid with the distributed inverse SHT. Each spatial shard draws its part of the state
from a stream of its own, seeded by :func:`~makani.models.noise.noise_base_seed`, so a distributed process does not draw
what a serial one does. The tests therefore feed both the same draws: a global draw, which the serial process gets
whole and every rank its shard of. Then the distributed process has to reproduce the serial one exactly, its state,
its samples and, for learnable noise, the gradients of its parameters once reduced as the trainers reduce them.

The seeding itself is checked across all ranks, and the snapshot and restore of a
:class:`~makani.models.noise_controller.NoiseController` on the shards.
"""

import contextlib
import os
import sys
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from parameterized import parameterized

from makani.utils import comm
from makani.models.noise import DiffusionNoiseS2, IsotropicGaussianRandomFieldS2, noise_base_seed
from makani.models.noise_controller import NoiseController

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import _gather_helper, _init_grid, _split_helper, reduce_success, teardown_comms
from ..testutils import compare_tensors, disable_tf32

# an odd number of latitudes, so that the degrees, orders and latitudes split unevenly over two ranks
_IMG_SHAPE = (17, 32)
_BATCH = 2
_CHANNELS = 3

# (name, class, keyword arguments); lmax=7 truncates the spectrum below what the grid resolves
_CASES = [
    ("isotropic", IsotropicGaussianRandomFieldS2, dict(alpha=1.0)),
    ("isotropic_lmax", IsotropicGaussianRandomFieldS2, dict(alpha=1.0, lmax=7)),
    ("isotropic_reflect", IsotropicGaussianRandomFieldS2, dict(reflect=True)),
    ("isotropic_learnable", IsotropicGaussianRandomFieldS2, dict(learnable=True)),
    ("diffusion", DiffusionNoiseS2, dict()),
    ("diffusion_lmax", DiffusionNoiseS2, dict(lmax=7)),
    ("diffusion_history", DiffusionNoiseS2, dict(num_time_steps=2)),
    ("diffusion_reflect", DiffusionNoiseS2, dict(reflect=True)),
    ("diffusion_learnable", DiffusionNoiseS2, dict(learnable=True)),
]
_LEARNABLE = [case for case in _CASES if case[2].get("learnable", False)]


class _Draws:
    """
    Replaces the draws of the noise processes by given tensors, in order.

    Inside the context, ``Tensor.normal_`` fills the tensor with the next one given, whatever the generator. A draw
    of another shape than the next tensor is recorded in ``ok`` rather than raised, so that a rank never leaves the
    others waiting in a collective; check it with :func:`reduce_success`.
    """

    def __init__(self):
        self.ok = True

    @contextlib.contextmanager
    def __call__(self, *tensors):
        queue = list(tensors)

        def normal_(tensor, mean=0.0, std=1.0, *, generator=None):
            source = queue.pop(0) if queue else None
            with torch.no_grad():
                if (source is None) or (source.shape != tensor.shape):
                    self.ok = False
                    tensor.zero_()
                else:
                    tensor.copy_(source)
            return tensor

        with mock.patch.object(torch.Tensor, "normal_", normal_):
            yield
        # every draw used
        self.ok = self.ok and not queue


class TestDistributedNoise(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    @classmethod
    def tearDownClass(cls):
        teardown_comms()

    def setUp(self):
        disable_tf32()

    # ------------------------------------------------------------------ helpers

    def _pair(self, cls, kwargs):
        """The process on the grid and the same process built serially."""
        process = cls(_IMG_SHAPE, _BATCH, _CHANNELS, **kwargs).to(self.device)
        with comm.serial():
            reference = cls(_IMG_SHAPE, _BATCH, _CHANNELS, **kwargs).to(self.device)
        return process, reference

    def _spectral(self, shape, seed):
        """A global draw in spectral layout, the same on every rank."""
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(shape, generator=generator).to(self.device)

    def _shard_spectral(self, x):
        # (..., L, M, 2): the degrees over h, the orders over w
        x = _split_helper(x, dim=-3, group=self.h_group)
        return _split_helper(x, dim=-2, group=self.w_group)

    def _gather_spectral(self, x):
        x = _gather_helper(x, dim=-3, group=self.h_group)
        return _gather_helper(x, dim=-2, group=self.w_group)

    def _shard_grid(self, x):
        x = _split_helper(x, dim=-2, group=self.h_group)
        return _split_helper(x, dim=-1, group=self.w_group)

    def _gather_grid(self, x):
        x = _gather_helper(x, dim=-2, group=self.h_group)
        return _gather_helper(x, dim=-1, group=self.w_group)

    def _draw_shapes(self, process, replace_state):
        """The global shapes of the draws of an update."""
        shape = (_BATCH, process.num_time_steps, _CHANNELS, process.lmax, process.mmax, 2)
        if isinstance(process, DiffusionNoiseS2) and not replace_state:
            # an autoregressive step draws the new time step only
            shape = (_BATCH, 1) + shape[2:]
        return shape

    def _step(self, process, reference, draws, replace_state, seed):
        """One update of both processes, from the same global draw."""
        eta = self._spectral(self._draw_shapes(reference, replace_state), seed)
        with draws(self._shard_spectral(eta)):
            process.update(replace_state=replace_state)
        with draws(eta):
            reference.update(replace_state=replace_state)

    def _check(self, label, local_gathered, full, atol=1e-5, rtol=1e-5):
        with self.subTest(desc=label):
            self.assertTrue(reduce_success(compare_tensors(label, local_gathered, full, atol, rtol), self.device))

    def _reduce_gradient(self, param):
        """
        The gradient of a parameter as the trainers reduce it, gathered over the groups it is sharded over.

        ``None`` if the parameter counts as shared over a group it is sharded over: the reduction would add up shards
        of different coefficients, or fail on shards of different sizes. Unannotated, it counts as shared over the
        whole model group, as in :func:`~makani.mpu.mappings.init_gradient_reduction_hooks`.
        """
        shared = getattr(param, "is_shared_mp", ["model"])
        expanded = set()
        for group in shared:
            expanded |= {"model": {"h", "w", "matmul"}, "spatial": {"h", "w"}}.get(group, {group})
        sharded = {group for group in param.sharded_dims_mp if group is not None}
        if any(comm.get_size(group) > 1 for group in expanded & sharded):
            return None

        grad = param.grad.clone()
        for group in shared:
            if comm.get_size(group) > 1:
                dist.all_reduce(grad, group=comm.get_group(group))
        for dim, group in enumerate(param.sharded_dims_mp):
            if (group is not None) and (comm.get_size(group) > 1):
                grad = _gather_helper(grad, dim=dim, group=comm.get_group(group))
        return grad

    # ------------------------------------------------------------------ tests

    @parameterized.expand(_CASES)
    def test_sample_matches_serial(self, _, cls, kwargs):
        """The distributed inverse SHT, with the spectral scaling sharded, reproduces the serial sample."""
        process, reference = self._pair(cls, kwargs)
        state = self._spectral(tuple(reference.state.shape), seed=11)
        process.set_tensor_state(self._shard_spectral(state))
        reference.set_tensor_state(state)

        with torch.no_grad():
            sample = process()
            expected = reference()

        with self.subTest(desc="local grid"):
            local = (_BATCH, process.num_time_steps, _CHANNELS, process.nlat_local, process.nlon_local)
            self.assertTrue(reduce_success(tuple(sample.shape) == local, self.device))
        self._check("sample", self._gather_grid(sample), expected)

    @parameterized.expand(_CASES)
    def test_updates_match_serial(self, _, cls, kwargs):
        """A fresh draw and two steps after it, from the same draws, give the serial state and samples."""
        process, reference = self._pair(cls, kwargs)
        draws = _Draws()

        for step, replace_state in enumerate((True, False, False)):
            self._step(process, reference, draws, replace_state, seed=100 + step)
            with self.subTest(desc=f"step {step}: draws"):
                self.assertTrue(reduce_success(draws.ok, self.device))
            self._check(f"step {step}: state", self._gather_spectral(process.get_tensor_state()), reference.state)
            with torch.no_grad():
                self._check(f"step {step}: sample", self._gather_grid(process()), reference())

    @parameterized.expand(_LEARNABLE)
    def test_learnable_gradients_match_serial(self, _, cls, kwargs):
        """The gradients of learnable noise, reduced over the groups the parameters are shared over, are the serial ones."""
        process, reference = self._pair(cls, kwargs)
        draws = _Draws()

        weights = [self._spectral((_BATCH, 1, _CHANNELS) + _IMG_SHAPE, seed=200 + step) for step in range(3)]
        loss = 0.0
        loss_ref = 0.0
        for step, replace_state in enumerate((True, False, False)):
            self._step(process, reference, draws, replace_state, seed=100 + step)
            # the matmul ranks hold the same noise and compute the same loss, and the gradients are summed over them:
            # each counts its share. In training, the network splits the work over matmul instead
            loss = loss + (self._shard_grid(weights[step]) * process()).sum() / comm.get_size("matmul")
            loss_ref = loss_ref + (weights[step] * reference()).sum()
        loss.backward()
        loss_ref.backward()

        with self.subTest(desc="draws"):
            self.assertTrue(reduce_success(draws.ok, self.device))
        params = dict(reference.named_parameters())
        for name, param in process.named_parameters():
            grad = self._reduce_gradient(param)
            with self.subTest(desc=f"gradient {name}: annotations"):
                self.assertIsNotNone(grad, f"{name} is shared over a group it is sharded over")
            if grad is not None:
                self._check(f"gradient {name}", grad, params[name].grad)

    def test_controller_restores_its_shards(self):
        """A snapshot of the shards, restored, continues as the processes did after it, on every rank."""
        processes = {
            "input_noise": IsotropicGaussianRandomFieldS2(_IMG_SHAPE, _BATCH, _CHANNELS, seed=noise_base_seed()[0]),
            "latent": DiffusionNoiseS2(_IMG_SHAPE, _BATCH, _CHANNELS, num_time_steps=2, seed=noise_base_seed()[0] + 1),
        }
        controller = NoiseController({name: p.to(self.device) for name, p in processes.items()})
        controller.advance(replace_state=True, batch_size=_BATCH)
        snapshot = controller.snapshot()

        with self.subTest(desc="local shards"):
            p = processes["latent"]
            local = (_BATCH, 2, _CHANNELS, p.lmax_local, p.mmax_local, 2)
            self.assertTrue(reduce_success(tuple(snapshot.states["latent"].tensor.shape) == local, self.device))

        def run():
            samples = []
            for _ in range(2):
                controller.advance()
                samples.append({name: controller.sample(name).clone() for name in controller})
            return samples

        first = run()
        controller.restore(snapshot, batch_size=_BATCH)
        second = run()
        for step, (a, b) in enumerate(zip(first, second)):
            for name in a:
                with self.subTest(desc=f"step {step}: {name}"):
                    self.assertTrue(reduce_success(torch.equal(a[name], b[name]), self.device))

    def test_base_seed_over_ranks(self):
        """
        The seeds of the ranks: one per spatial shard and data-parallel rank, the same over matmul.

        The matmul ranks hold the same input, the network splits its features over them, so they draw the same
        noise. Otherwise the seeds are all distinct, except that centered noise pairs neighbouring ensemble ranks,
        with opposite reflections.
        """
        coords = (comm.get_rank("spatial"), comm.get_rank("ensemble"), comm.get_rank("batch"))
        for centered in (False, True):
            entries = [None] * dist.get_world_size() if dist.is_initialized() else [None]
            mine = (coords, noise_base_seed(centered=centered))
            if dist.is_initialized():
                dist.all_gather_object(entries, mine)
            else:
                entries = [mine]
            for i, ((spatial_i, ens_i, batch_i), (seed_i, reflect_i)) in enumerate(entries):
                for (spatial_j, ens_j, batch_j), (seed_j, reflect_j) in entries[i + 1 :]:
                    same = (spatial_i, ens_i, batch_i) == (spatial_j, ens_j, batch_j)
                    paired = (
                        centered and not same and (spatial_i, ens_i // 2, batch_i) == (spatial_j, ens_j // 2, batch_j)
                    )
                    with self.subTest(
                        centered=centered, ranks=((spatial_i, ens_i, batch_i), (spatial_j, ens_j, batch_j))
                    ):
                        self.assertEqual(seed_i == seed_j, same or paired)
                        if same:
                            self.assertEqual(reflect_i, reflect_j)
                        if paired:
                            self.assertNotEqual(reflect_i, reflect_j)

    def test_matmul_ranks_draw_the_same_noise(self):
        """The ranks of the matmul group hold the same input, so they draw the same noise."""
        if comm.get_size("matmul") < 2:
            self.skipTest("needs at least two matmul ranks")
        seed, reflect = noise_base_seed()
        process = IsotropicGaussianRandomFieldS2(_IMG_SHAPE, _BATCH, _CHANNELS, seed=seed, reflect=reflect)
        process = process.to(self.device)
        process.update()
        with torch.no_grad():
            sample = process()

        samples = [torch.empty_like(sample) for _ in range(comm.get_size("matmul"))]
        dist.all_gather(samples, sample.contiguous(), group=comm.get_group("matmul"))
        self.assertTrue(reduce_success(all(torch.equal(s, samples[0]) for s in samples), self.device))

    def test_centered_pairs_draw_opposite_noise(self):
        """With centered noise, the two ranks of an ensemble pair draw the same field with opposite signs."""
        if comm.get_size("ensemble") < 2:
            self.skipTest("needs at least two ensemble ranks")
        seed, reflect = noise_base_seed(centered=True)
        process = DiffusionNoiseS2(_IMG_SHAPE, _BATCH, _CHANNELS, seed=seed, reflect=reflect).to(self.device)
        process.update(replace_state=True)
        process.update()
        with torch.no_grad():
            sample = process()

        group = comm.get_group("ensemble")
        samples = [torch.empty_like(sample) for _ in range(comm.get_size("ensemble"))]
        dist.all_gather(samples, sample.contiguous(), group=group)
        ok = all(torch.equal(samples[k], -samples[k + 1]) for k in range(0, len(samples) - 1, 2))
        # and the pairs differ from each other
        ok = ok and all(not torch.equal(samples[k], samples[k + 2]) for k in range(0, len(samples) - 2, 2))
        self.assertTrue(reduce_success(ok, self.device))


if __name__ == "__main__":
    unittest.main()
