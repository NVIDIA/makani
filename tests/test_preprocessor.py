# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch

from makani.models.preprocessor import Preprocessor2D
from makani.utils.dataloaders.batch import ZENITH
from makani.utils.functions import expand_history, flatten_history

from .testutils import set_seed, get_default_parameters, compare_tensors, IMG_SIZE_H, IMG_SIZE_W


def _make_pp(n_history=0, add_zenith=False, **overrides):
    params = get_default_parameters()
    params.n_history = n_history
    params.add_zenith = add_zenith
    for key, value in overrides.items():
        setattr(params, key, value)
    return Preprocessor2D(params)


class TestPreprocessor2DBasic(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.pp = Preprocessor2D(self.params)
        self.B = self.params.batch_size  # 1
        self.C = self.params.N_in_channels  # NUM_CHANNELS
        self.H = self.params.img_shape_x  # IMG_SIZE_H
        self.W = self.params.img_shape_y  # IMG_SIZE_W

    def _rand(self, *shape):
        return torch.randn(*shape)

    def _field(self, T=1, C=None, B=None):
        return self._rand(B or self.B, T, C or self.C, self.H, self.W)

    def _zenith(self, T=1, B=None):
        return self._rand(B or self.B, T, 1, self.H, self.W)

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def test_construction_does_not_raise(self):
        """Minimal params construct a Preprocessor2D without error."""
        pp = Preprocessor2D(get_default_parameters())
        self.assertIsInstance(pp, Preprocessor2D)

    def test_dynamic_features_follow_the_config(self):
        """The model takes the zenith angle exactly when the config adds it."""
        self.assertEqual(_make_pp(add_zenith=False).dynamic_features, [])
        self.assertEqual(_make_pp(add_zenith=True).dynamic_features, [ZENITH])

    # -----------------------------------------------------------------------
    # flatten_history / expand_history, the boundary to flat channel stacks
    # -----------------------------------------------------------------------

    def test_flatten_history_collapses_time(self):
        """flatten_history on (B,T,C,H,W) gives (B, T*C, H, W)."""
        T = 3
        out = flatten_history(self._field(T))
        self.assertEqual(tuple(out.shape), (self.B, T * self.C, self.H, self.W))

    def test_flatten_history_works_on_any_grid(self):
        """A mesh has a single spatial axis, (B, T, C, N)."""
        x = self._rand(self.B, 2, self.C, 7)
        self.assertEqual(tuple(flatten_history(x).shape), (self.B, 2 * self.C, 7))

    def test_flatten_history_needs_the_time_axis(self):
        """A tensor without a time axis is refused rather than passed through."""
        with self.assertRaises(RuntimeError):
            flatten_history(self._rand(self.B, self.C, self.H))

    def test_expand_history_adds_time(self):
        """expand_history on (B, T*C, H, W) gives (B, T, C, H, W)."""
        T = 3
        x = self._rand(self.B, T * self.C, self.H, self.W)
        out = expand_history(x, nhist=T)
        self.assertEqual(tuple(out.shape), (self.B, T, self.C, self.H, self.W))

    def test_flatten_expand_roundtrip(self, verbose=False):
        """flatten_history(expand_history(x, T)) is bit-identical to x."""
        T = 2
        x = self._rand(self.B, T * self.C, self.H, self.W)
        out = flatten_history(expand_history(x, nhist=T))
        self.assertTrue(compare_tensors("flatten/expand roundtrip", out, x, verbose=verbose))

    # -----------------------------------------------------------------------
    # static features
    # -----------------------------------------------------------------------

    def test_get_static_features_returns_tensor_when_enabled(self):
        """get_static_features returns a non-None tensor when grid is enabled."""
        pp = _make_pp(add_grid=True, gridtype="raw")
        sf = pp.get_static_features()
        self.assertIsNotNone(sf)
        self.assertIsInstance(sf, torch.Tensor)

    def test_get_static_features_returns_none_when_disabled(self):
        """get_static_features returns None when no features are configured."""
        self.assertIsNone(self.pp.get_static_features())

    def test_static_feature_spatial_shape_matches_grid(self):
        """The (H, W) of the stored static feature matches the image shape."""
        pp = _make_pp(add_grid=True, gridtype="raw")
        sf = pp.get_static_features()
        self.assertEqual(sf.shape[-2], IMG_SIZE_H)
        self.assertEqual(sf.shape[-1], IMG_SIZE_W)

    # -----------------------------------------------------------------------
    # correct_bias
    # -----------------------------------------------------------------------

    def test_correct_bias_noop_without_bias(self, verbose=False):
        """correct_bias returns input unchanged when no bias correction is registered."""
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.correct_bias(x)
        self.assertTrue(compare_tensors("correct_bias noop", out, x, verbose=verbose))

    def test_correct_bias_subtracts_registered_bias(self, verbose=False):
        """correct_bias subtracts the bias_correction buffer from the input."""
        pp = Preprocessor2D(get_default_parameters())
        bias = torch.ones(1, self.C, 1, 1)
        pp.register_buffer("bias_correction", bias, persistent=False)
        x = self._rand(self.B, self.C, self.H, self.W)
        out = pp.correct_bias(x)
        self.assertTrue(compare_tensors("correct_bias subtract", out, x - bias, verbose=verbose))

    def test_correct_bias_broadcasts_over_batch_and_spatial(self, verbose=False):
        """A (1, C, 1, 1) bias broadcasts correctly over (B, C, H, W)."""
        pp = Preprocessor2D(get_default_parameters())
        bias = torch.arange(self.C, dtype=torch.float32).reshape(1, self.C, 1, 1)
        pp.register_buffer("bias_correction", bias, persistent=False)
        x = torch.zeros(self.B, self.C, self.H, self.W)
        out = pp.correct_bias(x)
        expected = -bias.expand(self.B, self.C, self.H, self.W)
        self.assertTrue(compare_tensors("correct_bias broadcast", out, expected, verbose=verbose))

    # -----------------------------------------------------------------------
    # compute_history_stats / history_normalize / history_denormalize
    # -----------------------------------------------------------------------

    def test_history_normalize_none_mode_returns_x_unchanged(self, verbose=False):
        """history_normalization_mode='none': normalize is a no-op."""
        # default params have mode="none"
        x = self._field(1)
        out = self.pp.history_normalize(x, self.pp.compute_history_stats(x))
        self.assertTrue(compare_tensors("normalize noop", out, x, verbose=verbose))

    def test_history_denormalize_none_mode_returns_x_unchanged(self, verbose=False):
        """history_normalization_mode='none': denormalize is a no-op."""
        x = self._field(1)
        out = self.pp.history_denormalize(x, self.pp.compute_history_stats(x))
        self.assertTrue(compare_tensors("denormalize noop", out, x, verbose=verbose))

    def _make_pp_mean(self, n_history=1):
        """Build a Preprocessor2D in 'mean' normalization mode."""
        return _make_pp(n_history=n_history, history_normalization_mode="mean")

    def _make_pp_exponential(self, n_history=2):
        return _make_pp(n_history=n_history, history_normalization_mode="exponential", history_normalization_decay=0.5)

    def test_history_normalization_needs_the_time_axis(self):
        """Statistics and normalization refuse flat tensors instead of unfolding them."""
        pp = self._make_pp_mean(n_history=1)
        flat = self._rand(self.B, 2 * self.C, self.H, self.W)
        with self.assertRaises(RuntimeError):
            pp.compute_history_stats(flat)
        stats = pp.compute_history_stats(self._field(2))
        with self.assertRaises(RuntimeError):
            pp.history_normalize(flat, stats)
        with self.assertRaises(RuntimeError):
            pp.history_denormalize(flat, stats)

    def test_history_normalize_denormalize_roundtrip_mean_mode(self, verbose=False):
        """normalize then denormalize recovers the original tensor (mean mode, n_history=1)."""
        pp = self._make_pp_mean(n_history=1)
        x = self._field(pp.n_history + 1)
        stats = pp.compute_history_stats(x)
        xr = pp.history_denormalize(pp.history_normalize(x, stats), stats)
        self.assertTrue(compare_tensors("roundtrip", xr, x, atol=1e-5, rtol=1e-4, verbose=verbose))

    def test_history_normalize_denormalize_roundtrip_target_mean_mode(self, verbose=False):
        """normalize(target=True) then denormalize(target=True) recovers the original (mean mode)."""
        pp = self._make_pp_mean(n_history=1)
        # the input carries one more channel than the target, as with the zenith angle
        stats = pp.compute_history_stats(self._field(pp.n_history + 1, C=self.C + 1))
        y = self._field(1)
        yr = pp.history_denormalize(pp.history_normalize(y, stats, target=True), stats, target=True)
        self.assertTrue(compare_tensors("roundtrip target", yr, y, atol=1e-5, rtol=1e-4, verbose=verbose))

    def test_history_normalize_target_uses_the_leading_statistics(self, verbose=False):
        """A target holds the predicted channels, which lead the input channels."""
        pp = self._make_pp_mean(n_history=1)
        stats = pp.compute_history_stats(self._field(2, C=self.C + 1))
        y = self._field(1)
        yn = pp.history_normalize(y, stats, target=True)
        expected = (y - stats.mean[:, :, : self.C]) / stats.std[:, :, : self.C]
        self.assertTrue(compare_tensors("target statistics", yn, expected, verbose=verbose))

    def test_history_normalize_uses_the_same_statistics_for_every_step(self, verbose=False):
        """The statistics are per channel and broadcast over the time steps."""
        pp = self._make_pp_mean(n_history=1)
        x = self._field(2)
        stats = pp.compute_history_stats(x)
        xn = pp.history_normalize(x, stats)
        for t in range(2):
            expected = (x[:, t] - stats.mean[:, 0]) / stats.std[:, 0]
            self.assertTrue(compare_tensors(f"step {t}", xn[:, t], expected, verbose=verbose))

    def test_history_normalize_denormalize_roundtrip_exponential_mode(self, verbose=False):
        """Roundtrip holds for exponential normalization mode."""
        pp = self._make_pp_exponential(n_history=2)
        x = self._field(pp.n_history + 1)
        stats = pp.compute_history_stats(x)
        xr = pp.history_denormalize(pp.history_normalize(x, stats), stats)
        self.assertTrue(compare_tensors("roundtrip exponential", xr, x, atol=1e-5, rtol=1e-4, verbose=verbose))

    def test_compute_history_stats_mean_is_finite(self):
        """The mean contains no NaN or Inf."""
        pp = self._make_pp_mean(n_history=1)
        stats = pp.compute_history_stats(self._field(pp.n_history + 1))
        self.assertIsNotNone(stats.mean)
        self.assertFalse(torch.any(torch.isnan(stats.mean)).item())
        self.assertFalse(torch.any(torch.isinf(stats.mean)).item())

    def test_compute_history_stats_std_is_finite_and_nonneg(self):
        """The std is finite and non-negative."""
        pp = self._make_pp_mean(n_history=1)
        stats = pp.compute_history_stats(self._field(pp.n_history + 1))
        self.assertIsNotNone(stats.std)
        self.assertFalse(torch.any(torch.isnan(stats.std)).item())
        self.assertFalse(torch.any(torch.isinf(stats.std)).item())
        self.assertTrue((stats.std >= 0).all().item())

    def test_compute_history_stats_shapes_mean_mode(self):
        """Mean and std are (B, 1, C, 1, 1), broadcasting over time and grid."""
        pp = self._make_pp_mean(n_history=1)
        stats = pp.compute_history_stats(self._field(pp.n_history + 1))
        expected = (self.B, 1, self.C, 1, 1)
        self.assertEqual(tuple(stats.mean.shape), expected)
        self.assertEqual(tuple(stats.std.shape), expected)

    def test_history_normalize_refuses_statistics_of_another_batch(self):
        """Statistics of one window do not apply to a window of another batch."""
        pp = self._make_pp_mean(n_history=1)
        stats = pp.compute_history_stats(self._field(2))
        with self.assertRaisesRegex(RuntimeError, "batch mismatch"):
            pp.history_normalize(self._field(2, B=self.B + 1), stats)

    # -----------------------------------------------------------------------
    # compute_history_stats — known-answer tests
    # -----------------------------------------------------------------------

    def _check_constant_field(self, pp, value, verbose=False):
        T = pp.n_history + 1
        x = torch.full((self.B, T, self.C, self.H, self.W), value, dtype=torch.float32)
        stats = pp.compute_history_stats(x)
        self.assertTrue(
            compare_tensors(
                "constant-field mean",
                stats.mean,
                torch.full_like(stats.mean, value),
                atol=1e-5,
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_tensors("constant-field std", stats.std, torch.zeros_like(stats.std), atol=1e-5, verbose=verbose)
        )

    def test_compute_history_stats_constant_field_mean_mode(self, verbose=False):
        """Constant field value c -> mean == c, std == 0 (mean mode)."""
        self._check_constant_field(self._make_pp_mean(n_history=1), 3.7, verbose=verbose)

    def test_compute_history_stats_constant_field_exponential_mode(self, verbose=False):
        """Constant field -> mean == value, std == 0 (exponential mode)."""
        self._check_constant_field(self._make_pp_exponential(n_history=2), -1.25, verbose=verbose)

    def test_compute_history_stats_per_channel_constants_mean_mode(self, verbose=False):
        """Per-channel constants: the mean of channel c equals channel c's constant value."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        # every spatial/time value of channel c is (c + 1.0)
        x = torch.zeros(self.B, T, self.C, self.H, self.W, dtype=torch.float32)
        for c in range(self.C):
            x[:, :, c, :, :] = float(c + 1)
        stats = pp.compute_history_stats(x)
        expected_mean = torch.arange(1, self.C + 1, dtype=torch.float32).reshape(1, 1, self.C, 1, 1)
        self.assertTrue(compare_tensors("per-channel mean", stats.mean, expected_mean, atol=1e-5, verbose=verbose))
        self.assertTrue(
            compare_tensors("per-channel std", stats.std, torch.zeros_like(stats.std), atol=1e-5, verbose=verbose)
        )

    def test_compute_history_stats_matches_analytical_weighted_mean_exponential(self, verbose=False):
        """Spatially-constant-per-step fields: the mean recovers the analytical weighted temporal average."""
        pp = self._make_pp_exponential(n_history=2)
        T = pp.n_history + 1
        # spatially constant per (time, channel) but distinct across time & channel
        values = (torch.arange(T * self.C, dtype=torch.float32) + 1.0).reshape(T, self.C)
        x = values.reshape(1, T, self.C, 1, 1).expand(self.B, T, self.C, self.H, self.W).contiguous()
        stats = pp.compute_history_stats(x)
        # reconstruct weights exactly as the preprocessor does: first element is oldest
        w = torch.exp(-0.5 * torch.arange(start=T - 1, end=-1, step=-1, dtype=torch.float32))
        w = w / w.sum()
        expected_mean = (w.unsqueeze(-1) * values).sum(dim=0).reshape(1, 1, self.C, 1, 1)
        self.assertTrue(
            compare_tensors(
                "analytical weighted mean", stats.mean, expected_mean, atol=1e-5, rtol=1e-5, verbose=verbose
            )
        )

    def _linear_ramp(self, T):
        a = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float32)[: self.C]
        b = torch.tensor([1.0, -0.5, 2.0, 0.25, -1.0], dtype=torch.float32)[: self.C]
        x = torch.zeros(self.B, T, self.C, self.H, self.W, dtype=torch.float32)
        for t in range(T):
            x[:, t, :, :, :] = (a + b * t).reshape(1, self.C, 1, 1)
        return a, b, x

    def test_compute_history_stats_linear_ramp_mean_mode_mean(self, verbose=False):
        """Linear ramp v_t = a_c + b_c * t (spatially constant): the mean matches a_c + b_c * t_bar."""
        pp = self._make_pp_mean(n_history=2)
        T = pp.n_history + 1
        a, b, x = self._linear_ramp(T)
        stats = pp.compute_history_stats(x)
        t_bar = torch.arange(T, dtype=torch.float32).mean()
        expected_mean = (a + b * t_bar).reshape(1, 1, self.C, 1, 1)
        self.assertTrue(
            compare_tensors("linear ramp mean", stats.mean, expected_mean, atol=1e-5, rtol=1e-5, verbose=verbose)
        )

    def test_compute_history_stats_linear_ramp_mean_mode_std(self, verbose=False):
        """Linear ramp: the std = |b_c| * sqrt(Var_t) (uniform-weight variance of 0..T-1)."""
        pp = self._make_pp_mean(n_history=2)
        T = pp.n_history + 1
        _, b, x = self._linear_ramp(T)
        stats = pp.compute_history_stats(x)
        t_vec = torch.arange(T, dtype=torch.float32)
        var_t = ((t_vec - t_vec.mean()) ** 2).mean()
        expected_std = (b.abs() * torch.sqrt(var_t)).reshape(1, 1, self.C, 1, 1)
        self.assertTrue(
            compare_tensors("linear ramp std", stats.std, expected_std, atol=1e-5, rtol=1e-5, verbose=verbose)
        )

    # -----------------------------------------------------------------------
    # history_normalize post-condition: weighted temporal mean ≈ 0, weighted L2 ≈ 1
    # -----------------------------------------------------------------------

    def _check_post_condition(self, pp, weights, verbose=False):
        x = self._field(pp.n_history + 1)
        xn = pp.history_normalize(x, pp.compute_history_stats(x))
        quad_xn = pp.history_normalization.quadrature(xn)  # (B, T, C)
        quad_xn2 = pp.history_normalization.quadrature(xn**2)  # (B, T, C)
        weighted_mean = (weights * quad_xn).sum(dim=1)  # (B, C)
        weighted_l2 = (weights * quad_xn2).sum(dim=1)  # (B, C)
        self.assertTrue(
            compare_tensors(
                "weighted mean ≈ 0",
                weighted_mean,
                torch.zeros_like(weighted_mean),
                atol=1e-5,
                rtol=1e-4,
                verbose=verbose,
            )
        )
        self.assertTrue(
            compare_tensors(
                "weighted L2 ≈ 1", weighted_l2, torch.ones_like(weighted_l2), atol=1e-4, rtol=1e-4, verbose=verbose
            )
        )

    def test_history_normalize_post_condition_mean_mode(self, verbose=False):
        """After normalize (mean mode): weighted temporal mean ≈ 0, weighted L2 ≈ 1, per channel."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        # mean-mode weights are uniform 1/T
        self._check_post_condition(pp, torch.full((1, T, 1), 1.0 / T), verbose=verbose)

    def test_history_normalize_post_condition_exponential_mode(self, verbose=False):
        """After normalize (exponential mode): weighted temporal mean ≈ 0, weighted L2 ≈ 1, per channel."""
        pp = self._make_pp_exponential(n_history=2)
        T = pp.n_history + 1
        # exponential weights: buffer shape is (1, T, 1, 1, 1)
        self._check_post_condition(
            pp, pp.history_normalization.history_normalization_weights.reshape(1, T, 1), verbose=verbose
        )

    # -----------------------------------------------------------------------
    # Gradient flow through train-path ops
    # -----------------------------------------------------------------------

    def test_history_normalize_grad_flows_mean_mode(self):
        """history_normalize is differentiable w.r.t. its input (mean mode)."""
        pp = self._make_pp_mean(n_history=1)
        T = pp.n_history + 1
        stats = pp.compute_history_stats(self._field(T))
        x = self._field(T).requires_grad_(True)
        out = pp.history_normalize(x, stats)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        # d/dx of (x - m)/s summed = 1/s broadcast; grad must be non-zero everywhere
        self.assertTrue((x.grad.abs() > 0).all().item())

    # -----------------------------------------------------------------------
    # CPU ↔ GPU numerical consistency
    # -----------------------------------------------------------------------

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_compute_history_stats_cpu_gpu_match(self, verbose=False):
        """compute_history_stats produces matching mean/std on CPU and GPU (mean mode)."""
        pp_cpu = self._make_pp_mean(n_history=1)
        pp_gpu = self._make_pp_mean(n_history=1).to("cuda")
        x = self._field(pp_cpu.n_history + 1)
        stats_cpu = pp_cpu.compute_history_stats(x)
        stats_gpu = pp_gpu.compute_history_stats(x.to("cuda"))
        for name in ("mean", "std"):
            self.assertTrue(
                compare_tensors(
                    f"{name} cpu vs gpu",
                    getattr(stats_cpu, name),
                    getattr(stats_gpu, name).cpu(),
                    atol=1e-5,
                    rtol=1e-5,
                    verbose=verbose,
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_history_normalize_cpu_gpu_match(self, verbose=False):
        """history_normalize produces matching output on CPU and GPU (mean mode)."""
        pp_cpu = self._make_pp_mean(n_history=1)
        pp_gpu = self._make_pp_mean(n_history=1).to("cuda")
        x = self._field(pp_cpu.n_history + 1)
        xn_cpu = pp_cpu.history_normalize(x, pp_cpu.compute_history_stats(x))
        xg = x.to("cuda")
        xn_gpu = pp_gpu.history_normalize(xg, pp_gpu.compute_history_stats(xg)).cpu()
        self.assertTrue(
            compare_tensors("history_normalize cpu vs gpu", xn_cpu, xn_gpu, atol=1e-5, rtol=1e-4, verbose=verbose)
        )

    # -----------------------------------------------------------------------
    # Hardening: controlled-fail shape / batch checks
    # -----------------------------------------------------------------------

    def test_expand_history_raises_on_non_divisible_channels(self):
        """expand_history raises when the channel dim isn't divisible by nhist.

        The check is a torch._check (compile-safe), so it surfaces as RuntimeError.
        """
        # channels = 5, not divisible by nhist = 3
        x_bad = torch.randn(self.B, 5, self.H, self.W)
        with self.assertRaises(RuntimeError) as cm:
            expand_history(x_bad, nhist=3)
        self.assertIn("not divisible", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
