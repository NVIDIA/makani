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

import math
import unittest

import numpy as np
import torch

from makani.models.noise import (
    toep,
    IsotropicGaussianRandomFieldS2,
    DiffusionNoiseS2,
    DummyNoiseS2,
)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed

# -------------------------------------------------------------------------
# Common test dimensions (small enough to be fast on CPU)
# -------------------------------------------------------------------------
IMG_SHAPE      = (32, 64)   # nlat, nlon
BATCH_SIZE     = 2
NUM_CHANNELS   = 3
NUM_TIME_STEPS = 1


# ===========================================================================
# 1. toep()
# ===========================================================================
class TestToep(unittest.TestCase):
    """Tests for the Toeplitz-matrix helper copied from scipy."""

    def test_default_shape(self):
        """toep(c) returns an (n, n) matrix for a length-n vector c."""
        T = toep([1, 2, 3])
        self.assertEqual(T.shape, (3, 3))

    def test_asymmetric_shape(self):
        """toep(c, r) returns a (len(c), len(r)) matrix."""
        T = toep([1, 2, 3], [1, 4, 5, 6])
        self.assertEqual(T.shape, (3, 4))

    def test_first_column_equals_c(self):
        """First column of toep(c) equals c."""
        c = [10, 20, 30]
        T = toep(c)
        np.testing.assert_array_equal(T[:, 0], c)

    def test_first_row_matches_r(self):
        """First row of toep(c, r) is [c[0], r[1], r[2], ...]."""
        c = [1, 2, 3]
        r = [1, 9, 8, 7]
        T = toep(c, r)
        np.testing.assert_array_equal(T[0, :], r)

    def test_symmetric_when_real_no_r(self):
        """Without an explicit r, a real-valued c produces a symmetric matrix."""
        T = toep([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(T, T.T)

    def test_scalar(self):
        """A single-element c returns a 1×1 matrix with that value."""
        T = toep([7])
        self.assertEqual(T.shape, (1, 1))
        self.assertEqual(T[0, 0], 7)


# ===========================================================================
# 2. DummyNoiseS2
# ===========================================================================
class TestDummyNoiseS2(unittest.TestCase):
    """DummyNoiseS2 contract: finite output with the correct shape."""

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = BATCH_SIZE
        self.T = NUM_TIME_STEPS
        self.C = NUM_CHANNELS
        self.noise = DummyNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=self.B,
            num_channels=self.C,
            num_time_steps=self.T,
        ).to(self.device)

    def test_is_stateful(self):
        self.assertFalse(self.noise.is_stateful())

    def test_initial_state_shape(self):
        expected = (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1])
        self.assertEqual(tuple(self.noise.state.shape), expected)

    def test_initial_state_finite(self):
        self.assertFalse(torch.isnan(self.noise.state).any())
        self.assertFalse(torch.isinf(self.noise.state).any())

    def test_forward_shape(self):
        out = self.noise.forward()
        expected = (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1])
        self.assertEqual(tuple(out.shape), expected)

    def test_forward_finite(self):
        out = self.noise.forward()
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_update_preserves_shape(self):
        self.noise.update()
        expected = (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1])
        self.assertEqual(tuple(self.noise.state.shape), expected)

    def test_update_preserves_finiteness(self):
        self.noise.update()
        self.assertFalse(torch.isnan(self.noise.state).any())
        self.assertFalse(torch.isinf(self.noise.state).any())

    def test_update_new_batch_size(self):
        """update(batch_size=N) resizes the state and keeps it finite."""
        new_B = 5
        self.noise.update(batch_size=new_B)
        self.assertEqual(self.noise.state.shape[0], new_B)
        self.assertFalse(torch.isnan(self.noise.state).any())
        self.assertFalse(torch.isinf(self.noise.state).any())

    def test_forward_after_update_finite(self):
        self.noise.update()
        out = self.noise.forward()
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())


# ===========================================================================
# 3. IsotropicGaussianRandomFieldS2
# ===========================================================================
class TestIsotropicGRF(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = BATCH_SIZE
        self.T = NUM_TIME_STEPS
        self.C = NUM_CHANNELS
        self.sigma = 2.0

    def _make_noise(self, seed=333, reflect=False, sigma=None, batch_size=None):
        if sigma is None:
            sigma = self.sigma
        if batch_size is None:
            batch_size = self.B
        return IsotropicGaussianRandomFieldS2(
            img_shape=IMG_SHAPE,
            batch_size=batch_size,
            num_channels=self.C,
            num_time_steps=self.T,
            sigma=sigma,
            seed=seed,
            reflect=reflect,
        ).to(self.device)

    def test_is_stateful(self):
        self.assertFalse(self._make_noise().is_stateful())

    def test_initial_state_shape(self):
        noise = self._make_noise()
        # state: (B, T, C, lmax_local, mmax_local, 2)
        self.assertEqual(noise.state.shape[0], self.B)
        self.assertEqual(noise.state.shape[1], self.T)
        self.assertEqual(noise.state.shape[2], self.C)
        self.assertEqual(noise.state.shape[-1], 2)   # real + imag packed

    def test_state_shape_after_update(self):
        noise = self._make_noise()
        noise.update()
        self.assertEqual(noise.state.shape[0], self.B)
        self.assertEqual(noise.state.shape[1], self.T)
        self.assertEqual(noise.state.shape[2], self.C)
        self.assertEqual(noise.state.shape[-1], 2)

    def test_forward_shape(self):
        noise = self._make_noise()
        noise.update()
        out = noise.forward()
        self.assertEqual(tuple(out.shape),
                         (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1]))

    def test_forward_finite(self):
        noise = self._make_noise()
        noise.update()
        out = noise.forward()
        self.assertFalse(torch.isnan(out).any(), "NaN in forward output")
        self.assertFalse(torch.isinf(out).any(), "Inf in forward output")

    def test_forward_no_grad(self):
        """forward() inside torch.no_grad() does not produce grad-requiring tensors."""
        noise = self._make_noise()
        noise.update()
        with torch.no_grad():
            out = noise.forward()
        self.assertFalse(out.requires_grad)

    def test_reproducibility_same_seed(self):
        """Two instances with the same constructor seed produce identical output."""
        n1 = self._make_noise(seed=42)
        n2 = self._make_noise(seed=42)
        n1.update()
        n2.update()
        self.assertTrue(torch.equal(n1.forward(), n2.forward()))

    def test_different_seeds_differ(self):
        """Two instances with different seeds produce different outputs."""
        n1 = self._make_noise(seed=1)
        n2 = self._make_noise(seed=2)
        n1.update()
        n2.update()
        self.assertFalse(torch.equal(n1.forward(), n2.forward()))

    def test_rng_state_save_restore(self):
        """Saving and restoring the RNG state reproduces the same output."""
        noise = self._make_noise()
        cpu_state, gpu_state = noise.get_rng_state()
        noise.update()
        out1 = noise.forward().clone()
        # Restore and replay
        noise.set_rng_state(cpu_state, gpu_state)
        noise.update()
        out2 = noise.forward()
        self.assertTrue(torch.equal(out1, out2))

    def test_successive_updates_change_state(self):
        """Successive update() calls produce different states."""
        noise = self._make_noise()
        noise.update()
        state1 = noise.state.clone()
        noise.update()
        self.assertFalse(torch.equal(state1, noise.state))

    def test_reflect_negates_state(self):
        """reflect=True produces the negation of reflect=False (same seed)."""
        n_normal  = self._make_noise(seed=42, reflect=False)
        n_reflect = self._make_noise(seed=42, reflect=True)
        n_normal.update()
        n_reflect.update()
        self.assertTrue(torch.allclose(n_normal.state, -n_reflect.state))

    # -----------------------------------------------------------------------
    # Statistical tests — require a large batch for reliable estimates
    # -----------------------------------------------------------------------
    def test_spatial_variance_matches_sigma(self):
        """Empirical variance of the GRF output ≈ sigma² over a large batch.

        By construction, sum_l (2l+1)*sigma_l²/(4π) = sigma², so the expected
        spatial variance at any point equals sigma².
        """
        B_stat = 500
        sigma  = 2.0
        noise  = self._make_noise(seed=333, sigma=sigma, batch_size=B_stat)
        noise.update()
        out = noise.forward()   # (B_stat, T, C, H, W)

        empirical_var = out.var().item()
        self.assertAlmostEqual(empirical_var, sigma ** 2, delta=0.5,
                               msg=f"Expected variance ≈ {sigma**2:.1f}, got {empirical_var:.4f}")

    def test_zero_mean(self):
        """Empirical mean of the GRF output ≈ 0 over a large batch."""
        B_stat = 500
        noise  = self._make_noise(seed=333, batch_size=B_stat)
        noise.update()
        out = noise.forward()   # (B_stat, T, C, H, W)

        empirical_mean = out.mean().item()
        self.assertAlmostEqual(empirical_mean, 0.0, delta=0.2,
                               msg=f"Expected mean ≈ 0, got {empirical_mean:.4f}")


# ===========================================================================
# 4. DiffusionNoiseS2
# ===========================================================================
class TestDiffusionNoiseS2(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B     = BATCH_SIZE
        self.T     = NUM_TIME_STEPS
        self.C     = 2          # use 2 channels to exercise per-channel phi/sigma
        self.lambd = 0.5
        self.phi   = math.exp(-self.lambd)

    def _make_noise(self, seed=333, reflect=False, lambd=None,
                    batch_size=None, num_channels=None):
        if lambd is None:
            lambd = self.lambd
        if batch_size is None:
            batch_size = self.B
        if num_channels is None:
            num_channels = self.C
        return DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=batch_size,
            num_channels=num_channels,
            num_time_steps=self.T,
            lambd=lambd,
            seed=seed,
            reflect=reflect,
        ).to(self.device)

    def test_is_stateful(self):
        self.assertTrue(self._make_noise().is_stateful())

    def test_initial_state_shape(self):
        noise = self._make_noise()
        # state: (B, T, C, lmax_local, mmax_local, 2)
        self.assertEqual(noise.state.shape[0], self.B)
        self.assertEqual(noise.state.shape[1], self.T)
        self.assertEqual(noise.state.shape[2], self.C)
        self.assertEqual(noise.state.shape[-1], 2)

    def test_state_shape_after_replace(self):
        noise = self._make_noise()
        noise.update(replace_state=True)
        self.assertEqual(noise.state.shape[0], self.B)
        self.assertEqual(noise.state.shape[1], self.T)
        self.assertEqual(noise.state.shape[2], self.C)
        self.assertEqual(noise.state.shape[-1], 2)

    def test_forward_shape(self):
        noise = self._make_noise()
        noise.update(replace_state=True)
        out = noise.forward()
        self.assertEqual(tuple(out.shape),
                         (self.B, self.T, self.C, IMG_SHAPE[0], IMG_SHAPE[1]))

    def test_forward_finite(self):
        noise = self._make_noise()
        noise.update(replace_state=True)
        out = noise.forward()
        self.assertFalse(torch.isnan(out).any(), "NaN in forward output")
        self.assertFalse(torch.isinf(out).any(), "Inf in forward output")

    def test_replace_state_non_trivial(self):
        """update(replace_state=True) samples from the stationary distribution
        and must produce a non-zero state."""
        noise = self._make_noise()
        noise.update(replace_state=True)
        self.assertFalse(torch.all(noise.state == 0),
                         "State is all-zero after replace_state update")

    def test_advance_update_changes_state(self):
        """update(replace_state=False) advances the AR(1) state."""
        noise = self._make_noise()
        noise.update(replace_state=True)
        state_before = noise.state.clone()
        noise.update(replace_state=False)
        self.assertFalse(torch.equal(state_before, noise.state))

    def test_reproducibility_same_seed(self):
        """Two instances with the same seed produce identical stationary samples."""
        n1 = self._make_noise(seed=42)
        n2 = self._make_noise(seed=42)
        n1.update(replace_state=True)
        n2.update(replace_state=True)
        self.assertTrue(torch.equal(n1.state, n2.state))

    def test_different_seeds_differ(self):
        """Different seeds give different stationary samples."""
        n1 = self._make_noise(seed=1)
        n2 = self._make_noise(seed=2)
        n1.update(replace_state=True)
        n2.update(replace_state=True)
        self.assertFalse(torch.equal(n1.state, n2.state))

    def test_rng_state_save_restore(self):
        """Restoring RNG and tensor state reproduces an AR(1) advance exactly."""
        noise = self._make_noise()
        noise.update(replace_state=True)
        tensor_state = noise.get_tensor_state()
        cpu_state, gpu_state = noise.get_rng_state()

        # Advance once, record state
        noise.update(replace_state=False)
        out1 = noise.state.clone()

        # Restore and replay
        noise.set_tensor_state(tensor_state)
        noise.set_rng_state(cpu_state, gpu_state)
        noise.update(replace_state=False)
        self.assertTrue(torch.equal(out1, noise.state))

    def test_reflect_negates_state(self):
        """reflect=True flips the sign of the stationary state (same seed)."""
        n_normal  = self._make_noise(seed=42, reflect=False)
        n_reflect = self._make_noise(seed=42, reflect=True)
        n_normal.update(replace_state=True)
        n_reflect.update(replace_state=True)
        self.assertTrue(torch.allclose(n_normal.state, -n_reflect.state))

    # -----------------------------------------------------------------------
    # Statistical test — requires a large batch for reliable estimates
    # -----------------------------------------------------------------------
    def test_temporal_correlation_matches_phi(self):
        """AR(1) correlation between consecutive spectral states ≈ phi = exp(-lambd).

        For each spectral coefficient x_{l,m} the AR(1) recurrence gives
        Corr(x_t, x_{t+1}) = phi, regardless of sigma_l.  Flattening over
        (batch, l, m, real/imag) yields many i.i.d. pairs, so the Pearson
        correlation of the flattened vectors should converge tightly to phi.
        """
        B_stat = 500
        lambd  = 0.5
        phi_expected = math.exp(-lambd)

        noise = DiffusionNoiseS2(
            img_shape=IMG_SHAPE,
            batch_size=B_stat,
            num_channels=1,
            num_time_steps=1,
            lambd=lambd,
            seed=333,
        ).to(self.device)

        # Sample from the stationary distribution
        noise.update(replace_state=True)
        state_t    = noise.state.clone()   # (B, 1, 1, L, M, 2)

        # Advance one AR(1) step
        noise.update(replace_state=False)
        state_next = noise.state.clone()   # (B, 1, 1, L, M, 2)

        # Pearson correlation over all entries (correlation is phi for every one)
        x = state_t.reshape(-1).float()
        y = state_next.reshape(-1).float()
        corr = torch.corrcoef(torch.stack([x, y]))[0, 1].item()

        self.assertAlmostEqual(corr, phi_expected, delta=0.05,
                               msg=f"Expected temporal correlation ≈ {phi_expected:.4f}, got {corr:.4f}")


if __name__ == "__main__":
    unittest.main()
