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

import sys
import os
import math
import tempfile
from typing import Optional
from parameterized import parameterized

import unittest
import numpy as np
import torch

from makani.utils import LossHandler
from makani.utils.losses import (
    CRPSLoss,
    SpectralCRPSLoss,
    GradientCRPSLoss,
    VortDivCRPSLoss,
    GaussianMMDLoss,
    GeometricLpLoss,
    SpectralLpLoss,
    SpectralH1Loss,
    DriftRegularization,
    SpectralRegularization,
    EnsembleNLLLoss,
    LpEnergyScoreLoss,
    SpectralL2EnergyScoreLoss,
)
from makani.utils.losses.energy_score import SobolevEnergyScoreLoss

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, get_default_parameters, compare_tensors, compare_arrays

from properscoring import crps_ensemble, crps_gaussian

_loss_params = [
    ([{"type": "l1"}], False),
    ([{"type": "l1", "parameters": {"relative": True}}], False),
    ([{"type": "l2", "parameters": {"squared": True}}], False),
    ([{"type": "l2", "channel_weights": "constant"}], True),
    ([{"type": "l2", "channel_weights": "constant"}, {"type": "l2", "channel_weights": "auto"}], True),
    ([{"type": "h1", "channel_weights": "constant"}], True),
    ([{"type": "l2", "channel_weights": "constant", "temp_diff_normalization": True}], True),
    ([{"type": "l2", "channel_weights": "constant"}, {"type": "h1", "channel_weights": "constant"}], True),
    ([{"type": "l2", "channel_weights": "constant"}, {"type": "l1", "channel_weights": "constant"}], True),
    ([{"type": "drift_regularization"}], False),
]

_loss_weighted_params = [
    ([{"type": "l1"}], False),
    ([{"type": "l1", "parameters": {"relative": True}}], False),
    ([{"type": "l2", "parameters": {"squared": True}}], False),
    ([{"type": "l2", "channel_weights": "constant"}], False),
    ([{"type": "l2", "channel_weights": "constant"}, {"type": "l2", "channel_weights": "auto"}], False),
    ([{"type": "l2", "channel_weights": "constant", "temp_diff_normalization": True}], False),
    ([{"type": "l2", "channel_weights": "constant"}, {"type": "l1", "channel_weights": "constant"}], False),
    ([{"type": "drift_regularization"}], False),
]

_loss_zero_params = [
    ([{"type": "l1"}], False),
    ([{"type": "l2"}], False),
    ([{"type": "l2", "channel_weights": "constant"}], False),
    ([{"type": "l1", "channel_weights": "constant"}], False),
    ([{"type": "h1", "channel_weights": "constant"}], False),
    ([{"type": "drift_regularization"}], False),
]

# ---------------------------------------------------------------------------
# Shared constants for direct loss instantiation tests
# ---------------------------------------------------------------------------
_IMG_H = 32
_IMG_W = 64
_BATCH = 4
_NUM_CH = 5
_CHANNEL_NAMES = ["u10m", "t2m", "u500", "z500", "t500"]

_WIND_CHANNEL_NAMES = ["u500", "v500", "u850", "v850", "t500"]
_NUM_WIND_CH = len(_WIND_CHANNEL_NAMES)

_GEOM_KWARGS = dict(
    img_shape=(_IMG_H, _IMG_W),
    crop_shape=(_IMG_H, _IMG_W),
    crop_offset=(0, 0),
    channel_names=_CHANNEL_NAMES,
    grid_type="equiangular",
    pole_mask=0,
)

_WIND_GEOM_KWARGS = dict(
    img_shape=(_IMG_H, _IMG_W),
    crop_shape=(_IMG_H, _IMG_W),
    crop_offset=(0, 0),
    channel_names=_WIND_CHANNEL_NAMES,
    grid_type="equiangular",
    pole_mask=0,
)

_SPEC_KWARGS = dict(
    img_shape=(_IMG_H, _IMG_W),
    crop_shape=(_IMG_H, _IMG_W),
    crop_offset=(0, 0),
    channel_names=_CHANNEL_NAMES,
    grid_type="equiangular",
)


def _rand(batch=_BATCH, channels=_NUM_CH, requires_grad=False):
    t = torch.randn(batch, channels, _IMG_H, _IMG_W)
    t.requires_grad_(requires_grad)
    return t


def _rand_ensemble(ensemble=5, batch=_BATCH, channels=_NUM_CH, requires_grad=False):
    t = torch.randn(batch, ensemble, channels, _IMG_H, _IMG_W)
    t.requires_grad_(requires_grad)
    return t


# ---------------------------------------------------------------------------
# Parameter lists for TestLossCommon
# ---------------------------------------------------------------------------

# Losses expected to be elementwise non-negative.
# EnsembleNLLLoss excluded: proper scoring rule that can be negative.
# GaussianMMDLoss excluded: the unbiased U-statistic MMD² can be negative
#   (e.g., all E members = obs with E=5 gives mmd² = (3-E)/(E-1) = -0.5).
_COMMON_NONNEG = [
    ("geometric_l2",), ("geometric_l1",),
    ("spectral_l2",), ("spectral_h1",),
    ("drift_regularization",),
    ("crps_cdf",), ("crps_gauss",),
]

# Losses expected to be (near) zero when prd perfectly matches tar.
# EnsembleNLLLoss excluded: with a degenerate ensemble sigma is clipped to eps,
#   leaving a residual log(eps^2)/2 term that is large and negative.
# GaussianMMDLoss excluded: perfect prediction gives mmd² = (3-E)/(E-1) ≠ 0.
# crps_gauss included: with sigma clamped to eps the residual ≈ eps * 0.23 ≈ 2e-6,
#   well within atol=1e-4.
_COMMON_ZERO_PERFECT = [
    ("geometric_l2",), ("geometric_l1",),
    ("spectral_l2",), ("spectral_h1",),
    ("drift_regularization",),
    ("crps_cdf",), ("crps_gauss",),
]

# All losses participate in the batch-size independence test.
# GaussianMMDLoss is tested with squared=True to avoid sqrt of potentially negative mmd².
_COMMON_BATCHSIZE = [
    ("geometric_l2",), ("geometric_l1",),
    ("spectral_l2",), ("spectral_h1",),
    ("drift_regularization",),
    ("crps_cdf",), ("crps_gauss",),
    ("nll",),
    ("mmd",),
]


# ===========================================================================
class TestLossCommon(unittest.TestCase):
    """Common property tests executed directly against every loss class.

    Three properties are verified:
      1. ``test_nonneg``                — loss >= 0 elementwise
      2. ``test_zero_on_perfect_prediction`` — loss ≈ 0 when prd == tar
      3. ``test_batchsize_independence``    — loss[i] is unaffected by other samples in the batch
    """

    _E = 5  # ensemble size used for probabilistic losses

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @staticmethod
    def _make(name: str):
        """Return a freshly constructed loss instance for *name*."""
        if name == "geometric_l2":
            return GeometricLpLoss(**_GEOM_KWARGS, p=2.0)
        if name == "geometric_l1":
            return GeometricLpLoss(**_GEOM_KWARGS, p=1.0)
        if name == "spectral_l2":
            return SpectralLpLoss(**_SPEC_KWARGS)
        if name == "spectral_h1":
            return SpectralH1Loss(**_SPEC_KWARGS)
        if name == "drift_regularization":
            return DriftRegularization(**_GEOM_KWARGS)
        if name == "crps_cdf":
            return CRPSLoss(**_GEOM_KWARGS, crps_type="cdf",
                                    spatial_distributed=False, ensemble_distributed=False)
        if name == "crps_gauss":
            return CRPSLoss(**_GEOM_KWARGS, crps_type="gauss",
                                    spatial_distributed=False, ensemble_distributed=False, eps=1e-5)
        if name == "nll":
            return EnsembleNLLLoss(**_GEOM_KWARGS)
        if name == "mmd":
            # squared=True: avoids sqrt of potentially negative mmd² values in common tests
            return GaussianMMDLoss(**_GEOM_KWARGS, squared=True)
        raise ValueError(f"Unknown loss name: {name!r}")

    @classmethod
    def _make_prd_tar(cls, name: str, perfect: bool = False):
        """Return *(prd, tar)*.  Ensemble losses get a 5-D prd; others 4-D."""
        E = cls._E
        tar = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        if name in ("crps_cdf", "crps_gauss", "nll", "mmd"):
            if perfect:
                # all E members equal the observation
                prd = tar.unsqueeze(1).expand(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W).clone()
            else:
                prd = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        else:
            prd = tar.clone() if perfect else torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        return prd, tar

    # ------------------------------------------------------------------

    @parameterized.expand(_COMMON_NONNEG)
    def test_nonneg(self, name):
        """Loss output must be elementwise non-negative."""
        fn = self._make(name)
        prd, tar = self._make_prd_tar(name)
        loss = fn(prd, tar)
        self.assertTrue(
            (loss >= -1e-6).all(),
            f"{name}: found negative values, min={loss.min().item():.4e}",
        )

    @parameterized.expand(_COMMON_ZERO_PERFECT)
    def test_zero_on_perfect_prediction(self, name, verbose=False):
        """Loss must be (near) zero when the prediction perfectly matches the target."""
        fn = self._make(name)
        prd, tar = self._make_prd_tar(name, perfect=True)
        loss = fn(prd, tar)
        self.assertTrue(
            compare_tensors(f"{name} zero", loss, torch.zeros_like(loss), atol=1e-4, verbose=verbose),
        )

    @parameterized.expand(_COMMON_BATCHSIZE)
    def test_batchsize_independence(self, name, verbose=False):
        """The loss for sample [0] computed alone must equal loss[0] in a full batch."""
        fn = self._make(name)
        prd, tar = self._make_prd_tar(name)

        loss_single = fn(prd[:1], tar[:1])   # (1, C)
        loss_batch  = fn(prd,     tar)        # (B, C)

        self.assertTrue(
            compare_tensors(f"{name} batchsize", loss_single[0], loss_batch[0], verbose=verbose),
            f"{name}: loss[0] differs between single-sample and full-batch evaluation",
        )


# ===========================================================================
class TestGeometricLpLoss(unittest.TestCase):
    """Specific tests for GeometricLpLoss beyond what TestLosses covers via LossHandler."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_squared_flag_consistency(self, verbose=False):
        """For p=2, loss(squared=False)^2 must equal loss(squared=True)."""
        fn_unsq = GeometricLpLoss(**_GEOM_KWARGS, p=2.0, squared=False)
        fn_sq   = GeometricLpLoss(**_GEOM_KWARGS, p=2.0, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors("squared flag", fn_unsq(prd, tar) ** 2, fn_sq(prd, tar), atol=1e-5, rtol=1e-4, verbose=verbose),
        )

    @parameterized.expand([(1.0,), (2.0,), (4.0,)])
    def test_analytic_constant_difference(self, p, verbose=False):
        """Lp loss (squared=False) of a spatially constant difference c over a normalised
        grid equals c, because the quadrature integrates a constant field to 1."""
        c = 2.5
        fn = GeometricLpLoss(**_GEOM_KWARGS, p=p, squared=False)
        prd = torch.full((_BATCH, _NUM_CH, _IMG_H, _IMG_W), c)
        tar = torch.zeros_like(prd)
        loss = fn(prd, tar)
        self.assertTrue(
            compare_tensors(f"analytic L{p}", loss, torch.full_like(loss, c), atol=1e-4, rtol=1e-4, verbose=verbose),
        )

    def test_p_parameter_differentiated(self):
        """L1 and L2 norms must differ for a sparse large-value input."""
        fn_l1 = GeometricLpLoss(**_GEOM_KWARGS, p=1.0)
        fn_l2 = GeometricLpLoss(**_GEOM_KWARGS, p=2.0)
        prd = torch.zeros(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        prd[:, :, _IMG_H // 2, _IMG_W // 2] = 10.0  # single bright pixel per sample
        tar = torch.zeros_like(prd)
        diff = abs(fn_l1(prd, tar).mean().item() - fn_l2(prd, tar).mean().item())
        self.assertGreater(diff, 1e-3, "L1 and L2 norms should differ for a sparse input")

    @parameterized.expand([(1.0,), (4.0,)])
    def test_squared_flag_general(self, p, verbose=False):
        """For arbitrary p: loss(squared=False)^p must equal loss(squared=True)."""
        fn_unsq = GeometricLpLoss(**_GEOM_KWARGS, p=p, squared=False)
        fn_sq   = GeometricLpLoss(**_GEOM_KWARGS, p=p, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors(f"squared flag p={p}", fn_unsq(prd, tar) ** p, fn_sq(prd, tar),
                            atol=1e-4, rtol=1e-3, verbose=verbose),
        )

    @parameterized.expand([(1.0,), (2.0,), (4.0,)])
    def test_relative_loss_double_target(self, p, verbose=False):
        """relative=True: prd = 2*tar gives loss = 1 for any p.

        Proof: relative loss = (∫|2t-t|^p / ∫|t|^p)^(1/p) = (∫|t|^p / ∫|t|^p)^(1/p) = 1.
        """
        fn = GeometricLpLoss(**_GEOM_KWARGS, p=p, relative=True, squared=False)
        set_seed(333)
        tar = _rand() + 2.0   # shift away from zero to keep denominator well-conditioned
        prd = 2.0 * tar
        loss = fn(prd, tar)
        self.assertTrue(
            compare_tensors(f"relative L{p} double-target", loss, torch.ones_like(loss),
                            atol=1e-4, rtol=1e-4, verbose=verbose),
        )

    @parameterized.expand([(1.0,), (4.0,)])
    def test_gradient_flow(self, p):
        """abs() mode must produce finite, non-NaN gradients for p=1 and p=4."""
        fn  = GeometricLpLoss(**_GEOM_KWARGS, p=p, squared=False)
        prd = _rand(requires_grad=True)
        tar = _rand()
        fn(prd, tar).sum().backward()
        self.assertFalse(torch.isnan(prd.grad).any(), f"p={p}: NaN in gradient")
        self.assertFalse(torch.isinf(prd.grad).any(), f"p={p}: Inf in gradient")


# ===========================================================================
class TestSpectralLpLoss(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_squared_flag_consistency(self, verbose=False):
        """loss(squared=False)^2 must equal loss(squared=True)."""
        fn_unsq = SpectralLpLoss(**_SPEC_KWARGS, squared=False)
        fn_sq   = SpectralLpLoss(**_SPEC_KWARGS, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors("squared", fn_unsq(prd, tar) ** 2, fn_sq(prd, tar), atol=1e-5, rtol=1e-4, verbose=verbose),
        )

    def test_parseval_consistency_with_geometric_l2(self, verbose=False):
        """SpectralLpLoss and GeometricLpLoss(p=2) both approximate the normalised L2 norm
        on the sphere and should agree within 5 % for a smooth single-mode field."""
        fn_spec = SpectralLpLoss(**_SPEC_KWARGS, squared=True)
        fn_geom = GeometricLpLoss(**_GEOM_KWARGS, p=2.0, squared=True)

        lat = torch.linspace(0, math.pi, _IMG_H)
        lon = torch.linspace(0, 2.0 * math.pi, _IMG_W)
        LAT, LON = torch.meshgrid(lat, lon, indexing="ij")
        smooth = (torch.sin(LAT) * torch.cos(LON)).expand(_BATCH, _NUM_CH, -1, -1).clone()
        zeros = torch.zeros_like(smooth)

        loss_spec = fn_spec(smooth, zeros)
        loss_geom = fn_geom(smooth, zeros)
        self.assertTrue(
            compare_tensors("parseval", loss_spec, loss_geom, atol=1e-3, rtol=0.05, verbose=verbose),
            f"Spectral and geometric L2 should agree within 5 %: "
            f"spec={loss_spec.mean().item():.4f}, geom={loss_geom.mean().item():.4f}",
        )


# ===========================================================================
class TestSpectralH1Loss(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_squared_flag_consistency(self, verbose=False):
        fn_unsq = SpectralH1Loss(**_SPEC_KWARGS, squared=False)
        fn_sq   = SpectralH1Loss(**_SPEC_KWARGS, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors("squared", fn_unsq(prd, tar) ** 2, fn_sq(prd, tar), atol=1e-5, rtol=1e-4, verbose=verbose),
        )

    def test_constant_difference_has_zero_h1_seminorm(self, verbose=False):
        """A spatially constant field lives entirely in the l=0 SHT mode.
        h1_weights[0] = 0*(0+1) = 0, so the H1 seminorm must be exactly zero."""
        fn = SpectralH1Loss(**_SPEC_KWARGS, squared=True)
        prd = torch.full((_BATCH, _NUM_CH, _IMG_H, _IMG_W), 3.0)
        tar = torch.zeros_like(prd)
        loss = fn(prd, tar)
        self.assertTrue(
            compare_tensors("constant diff h1", loss, torch.zeros_like(loss), atol=1e-4, verbose=verbose),
        )

    def test_high_frequency_penalized_more_than_smooth(self):
        """After L2-normalisation, a high-frequency field (l≈8) must score higher H1 than a
        smooth field (l≈1) because h1_weights = l*(l+1) amplifies high modes."""
        fn = SpectralH1Loss(**_SPEC_KWARGS, squared=True)

        lat = torch.linspace(0, math.pi, _IMG_H)
        lon = torch.linspace(0, 2.0 * math.pi, _IMG_W)
        LAT, LON = torch.meshgrid(lat, lon, indexing="ij")

        smooth = torch.sin(LAT).expand(_BATCH, _NUM_CH, -1, -1).clone()
        rough  = (torch.sin(8 * LAT) * torch.cos(8 * LON)).expand(_BATCH, _NUM_CH, -1, -1).clone()

        # normalise to same Frobenius norm so only frequency content differs
        rough = rough * smooth.norm() / rough.norm().clamp(min=1e-6)

        tar = torch.zeros_like(smooth)
        h1_smooth = fn(smooth, tar).mean().item()
        h1_rough  = fn(rough,  tar).mean().item()
        self.assertGreater(
            h1_rough, h1_smooth,
            f"Rough H1 ({h1_rough:.4f}) should exceed smooth H1 ({h1_smooth:.4f})",
        )


# ===========================================================================
class TestSpectralRelativeLoss(unittest.TestCase):
    """Tests for the relative mode of SpectralLpLoss and SpectralH1Loss.

    The relative loss is defined as  ||SHT(prd - tar)|| / ||SHT(tar)||
    (with the H1 weighting for SpectralH1Loss).  The tests below verify
    mathematical properties that hold regardless of the spherical geometry.
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @staticmethod
    def _make(cls, squared=False):
        return cls(**_SPEC_KWARGS, relative=True, squared=squared)

    # --- zero on perfect prediction ---

    def test_l2_zero_on_perfect_prediction(self, verbose=False):
        fn  = self._make(SpectralLpLoss)
        tar = _rand()
        loss = fn(tar.clone(), tar)
        self.assertTrue(
            compare_tensors("l2 rel perfect", loss, torch.zeros_like(loss), atol=1e-5, verbose=verbose),
        )

    def test_h1_zero_on_perfect_prediction(self, verbose=False):
        fn  = self._make(SpectralH1Loss)
        tar = _rand()
        loss = fn(tar.clone(), tar)
        self.assertTrue(
            compare_tensors("h1 rel perfect", loss, torch.zeros_like(loss), atol=1e-5, verbose=verbose),
        )

    # --- unity when prd = 2 * tar  (||2t - t|| = ||t||, so ratio = 1) ---

    def test_l2_unity_when_prd_equals_twice_tar(self, verbose=False):
        fn  = self._make(SpectralLpLoss)
        tar = _rand()
        loss = fn(2.0 * tar, tar)
        self.assertTrue(
            compare_tensors("l2 rel 2x", loss, torch.ones_like(loss), atol=1e-4, rtol=1e-3, verbose=verbose),
        )

    def test_h1_unity_when_prd_equals_twice_tar(self, verbose=False):
        fn  = self._make(SpectralH1Loss)
        tar = _rand()
        loss = fn(2.0 * tar, tar)
        self.assertTrue(
            compare_tensors("h1 rel 2x", loss, torch.ones_like(loss), atol=1e-4, rtol=1e-3, verbose=verbose),
        )

    # --- squared flag consistency in relative mode ---

    def test_l2_squared_flag_consistency(self, verbose=False):
        """rel(squared=False)² must equal rel(squared=True)."""
        fn_unsq = self._make(SpectralLpLoss, squared=False)
        fn_sq   = self._make(SpectralLpLoss, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors("l2 rel squared", fn_unsq(prd, tar) ** 2, fn_sq(prd, tar), atol=1e-5, rtol=1e-4, verbose=verbose),
        )

    def test_h1_squared_flag_consistency(self, verbose=False):
        fn_unsq = self._make(SpectralH1Loss, squared=False)
        fn_sq   = self._make(SpectralH1Loss, squared=True)
        prd, tar = _rand(), _rand()
        self.assertTrue(
            compare_tensors("h1 rel squared", fn_unsq(prd, tar) ** 2, fn_sq(prd, tar), atol=1e-5, rtol=1e-4, verbose=verbose),
        )

    # --- larger error → larger relative loss ---

    def test_l2_monotone_in_error(self):
        """A prediction farther from the target must have a larger relative loss."""
        fn  = self._make(SpectralLpLoss)
        set_seed(333)
        tar = _rand()
        noise = torch.randn_like(tar)
        loss_small = fn(tar + 0.1 * noise, tar).mean().item()
        loss_large = fn(tar + 2.0 * noise, tar).mean().item()
        self.assertLess(loss_small, loss_large,
                        f"L2 rel: small-error loss {loss_small:.4f} should be < large-error {loss_large:.4f}")

    def test_h1_monotone_in_error(self):
        fn  = self._make(SpectralH1Loss)
        set_seed(333)
        tar = _rand()
        noise = torch.randn_like(tar)
        loss_small = fn(tar + 0.1 * noise, tar).mean().item()
        loss_large = fn(tar + 2.0 * noise, tar).mean().item()
        self.assertLess(loss_small, loss_large,
                        f"H1 rel: small-error loss {loss_small:.4f} should be < large-error {loss_large:.4f}")



# ===========================================================================
class TestDriftRegularization(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, p=1.0):
        return DriftRegularization(**_GEOM_KWARGS, p=p)

    def test_spatial_structure_insensitive(self, verbose=False):
        """Drift measures only the global spatial mean.  Two different spatial patterns
        with equal quadrature integrals must give zero loss."""
        fn = self._fn()
        prd = _rand()
        tar = _rand()

        # shift tar so that quadrature(tar_adjusted) == quadrature(prd) exactly
        prd_mean = fn.quadrature(prd).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        tar_mean = fn.quadrature(tar).unsqueeze(-1).unsqueeze(-1)
        tar_adjusted = tar - tar_mean + prd_mean

        loss = fn(prd, tar_adjusted)
        self.assertTrue(
            compare_tensors("zero drift", loss, torch.zeros_like(loss), atol=1e-6, verbose=verbose),
        )

    @parameterized.expand([(1.0,), (2.0,)])
    def test_scales_with_constant_bias(self, p, verbose=False):
        """For prd = tar + c (uniform bias), drift = c^p.
        The normalised quadrature maps a constant field to itself, so the bias is preserved exactly."""
        c = 2.0
        fn = self._fn(p=p)
        tar = _rand()
        prd = tar + c
        loss = fn(prd, tar)
        self.assertTrue(
            compare_tensors(f"drift scale p={p}", loss, torch.full_like(loss, c ** p), atol=1e-5, rtol=1e-4, verbose=verbose),
        )

    def test_ensemble_dim_handled(self, verbose=False):
        """5-D input (B, E, C, H, W): loss must equal the mean per-member drift.
        Each member e has constant offset biases[e]; mean drift = mean(biases)."""
        fn = self._fn(p=1.0)
        E = 4
        tar = torch.zeros(_BATCH, _NUM_CH, _IMG_H, _IMG_W)

        # biases = [1, 2, 3, 4] → mean = 2.5
        biases = torch.arange(1, E + 1, dtype=torch.float32)
        prd = biases.reshape(1, E, 1, 1, 1).expand(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W).clone()

        loss = fn(prd, tar)
        expected_mean = biases.mean().item()
        self.assertTrue(
            compare_tensors("ensemble drift", loss, torch.full_like(loss, expected_mean), atol=1e-5, rtol=1e-4, verbose=verbose),
        )


# ===========================================================================
class TestEnsembleNLLLoss(unittest.TestCase):
    """EnsembleNLLLoss requires 5-D (B, E, C, H, W) input and is tested directly."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, eps=1e-5):
        return EnsembleNLLLoss(**_GEOM_KWARGS, eps=eps)

    def test_backward(self):
        """Gradients through the multi-member NLL must be finite and free of NaNs."""
        fn = self._fn()
        E = 5
        forecasts = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W, requires_grad=True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fn(forecasts, obs).sum().backward()
        self.assertIsNotNone(forecasts.grad)
        self.assertFalse(torch.isnan(forecasts.grad).any(), "NaN in forecasts.grad")
        self.assertFalse(torch.isinf(forecasts.grad).any(), "Inf in forecasts.grad")

    def test_batch_independence(self, verbose=False):
        fn = self._fn()
        E = 5
        fc1  = torch.randn(1, E, _NUM_CH, _IMG_H, _IMG_W)
        obs1 = torch.randn(1, _NUM_CH, _IMG_H, _IMG_W)
        loss1 = fn(fc1, obs1)
        loss4 = fn(fc1.repeat(4, 1, 1, 1, 1), obs1.repeat(4, 1, 1, 1))
        self.assertTrue(compare_tensors("nll batch", loss1.repeat(4, 1), loss4, verbose=verbose))

    def test_single_member_is_finite(self):
        """E=1 forces sigma=0, clamped to eps; result must not be NaN/Inf."""
        fn = self._fn(eps=1e-5)
        forecasts = torch.randn(_BATCH, 1, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        loss = fn(forecasts, obs)
        self.assertTrue(torch.isfinite(loss).all(), "NLL must be finite for a single-member ensemble")

    def test_well_calibrated_lower_nll_than_biased(self):
        """Ensemble centred on the observation has strictly lower NLL than a biased ensemble."""
        fn = self._fn()
        E, sigma = 10, 0.5
        obs = torch.ones(_BATCH, _NUM_CH, _IMG_H, _IMG_W)

        fc_good = obs.unsqueeze(1) + sigma * torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        fc_bad  = (2.0 * obs).unsqueeze(1) + sigma * torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)

        nll_good = fn(fc_good, obs).mean().item()
        nll_bad  = fn(fc_bad,  obs).mean().item()
        self.assertLess(
            nll_good, nll_bad,
            f"Centred NLL ({nll_good:.4f}) should be < biased NLL ({nll_bad:.4f})",
        )

    def test_larger_spread_higher_nll_near_truth(self):
        """Same noise pattern scaled by σ=0.1 vs σ=2.0: the (obs-mu)^2/sigma^2 terms cancel,
        leaving only the log(sigma^2) difference, which is larger for the looser ensemble."""
        fn = self._fn()
        E = 20
        obs = torch.ones(_BATCH, _NUM_CH, _IMG_H, _IMG_W)

        set_seed(333)
        noise = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        fc_tight = obs.unsqueeze(1) + 0.1 * noise
        fc_loose = obs.unsqueeze(1) + 2.0 * noise

        nll_tight = fn(fc_tight, obs).mean().item()
        nll_loose = fn(fc_loose, obs).mean().item()
        self.assertGreater(
            nll_loose, nll_tight,
            f"Loose NLL ({nll_loose:.4f}) should exceed tight NLL ({nll_tight:.4f})",
        )


# ===========================================================================
class TestGaussianMMDLoss(unittest.TestCase):
    """Specific tests for GaussianMMDLoss.

    Note: the unbiased U-statistic MMD² can be negative (e.g., perfect prediction
    with E=5 gives mmd² = (3-E)/(E-1) = -0.5), so nonneg and zero-on-perfect tests
    do not apply.  Tested properties: squared-flag consistency, spread ordering,
    backward pass, and the E=1 code-path.
    """

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_squared_flag_consistency(self, verbose=False):
        """For a wide ensemble where mmd² > 0, sqrt(mmd²) must equal the unsquared loss."""
        fn_sq   = GaussianMMDLoss(**_GEOM_KWARGS, squared=True)
        fn_unsq = GaussianMMDLoss(**_GEOM_KWARGS, squared=False)
        # use a large-offset ensemble so that k(y_m, obs) ≈ 0 → mmd² > 0
        obs = torch.zeros(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc  = obs.unsqueeze(1) + 10.0 * torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        mmd2   = fn_sq(fc, obs)
        mmd    = fn_unsq(fc, obs)
        self.assertTrue(
            compare_tensors("squared flag", torch.sqrt(mmd2), mmd, atol=1e-5, rtol=1e-4, verbose=verbose),
        )

    def test_spread_increases_mmd(self):
        """A tight ensemble near the observation must have a higher kernel score
        than a wide ensemble far from it."""
        fn = GaussianMMDLoss(**_GEOM_KWARGS, squared=True)
        obs = torch.zeros(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        set_seed(333)
        noise = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        fc_tight = obs.unsqueeze(1) + 0.01 * noise
        fc_wide  = obs.unsqueeze(1) + 10.0 * noise
        score_tight = fn(fc_tight, obs).mean().item()
        score_wide  = fn(fc_wide,  obs).mean().item()
        self.assertGreater(
            score_tight, score_wide,
            f"Tight score ({score_tight:.4f}) should be > wide score ({score_wide:.4f})",
        )

    def test_backward(self):
        """Gradients through the double-loop MMD kernel must be finite and free of NaNs."""
        fn = GaussianMMDLoss(**_GEOM_KWARGS, squared=True)
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W, requires_grad=True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    def test_e1_special_case(self, verbose=False):
        """With E=1 the code takes a direct kernel path: mmd = k(obs, fc).
        When obs == fc the RBF kernel equals 1, so the spatially-averaged loss
        must equal 1 (squared=True) or 1 (squared=False, sqrt(1)=1)."""
        fn = GaussianMMDLoss(**_GEOM_KWARGS, squared=True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc  = obs.unsqueeze(1)  # E=1, all members = obs
        loss = fn(fc, obs)
        self.assertTrue(
            compare_tensors("e1 perfect", loss, torch.ones_like(loss), atol=1e-5, verbose=verbose),
        )


# ===========================================================================
class TestCRPSLoss(unittest.TestCase):
    """Verifies CRPSLoss against the properscoring reference implementation
    for the CDF and Gaussian kernels."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_cdf_matches_properscoring(self, verbose=False):
        crps_func = CRPSLoss(
            **_GEOM_KWARGS,
            crps_type="cdf",
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
        )

        for ensemble_size in [1, 10]:
            with self.subTest(ensemble_size=ensemble_size):
                inp = torch.empty((_BATCH, ensemble_size, _NUM_CH, _IMG_H, _IMG_W), dtype=torch.float32)
                inp.normal_(1.0, 1.0)
                tar = torch.ones(_BATCH, _NUM_CH, _IMG_H, _IMG_W, dtype=torch.float32)

                result = crps_func(inp, tar).cpu().numpy()

                tar_arr = tar.cpu().numpy()
                inp_arr = inp.cpu().numpy()

                # properscoring uses a different axis convention for the degenerate E=1 case
                if ensemble_size == 1:
                    axis = -1
                    inp_arr = np.squeeze(inp_arr, axis=1)
                else:
                    axis = 1

                result_proper = crps_ensemble(tar_arr, inp_arr, weights=None, issorted=False, axis=axis)
                quad_weight_arr = crps_func.quadrature.quad_weight.cpu().numpy()
                result_proper = np.sum(result_proper * quad_weight_arr, axis=(2, 3))

                self.assertTrue(compare_arrays("output", result, result_proper, verbose=verbose))

    def test_gauss_matches_properscoring(self, verbose=False):
        eps = 1.0e-5
        crps_func = CRPSLoss(
            **_GEOM_KWARGS,
            crps_type="gauss",
            spatial_distributed=False,
            ensemble_distributed=False,
            eps=eps,
        )

        for ensemble_size in [1, 10]:
            with self.subTest(ensemble_size=ensemble_size):
                inp = torch.empty((_BATCH, ensemble_size, _NUM_CH, _IMG_H, _IMG_W), dtype=torch.float32)
                inp.normal_(1.0, 1.0)
                tar = torch.ones(_BATCH, _NUM_CH, _IMG_H, _IMG_W, dtype=torch.float32)

                result = crps_func(inp, tar).cpu().numpy()

                tar_arr = tar.cpu().numpy()
                inp_arr = inp.cpu().numpy()

                # compute mu, sigma; guard against underflows
                mu = np.mean(inp_arr, axis=1)
                sigma = np.maximum(np.sqrt(np.var(inp_arr, axis=1)), eps)

                result_proper = crps_gaussian(tar_arr, mu, sigma, grad=False)
                quad_weight_arr = crps_func.quadrature.quad_weight.cpu().numpy()
                result_proper = np.sum(result_proper * quad_weight_arr, axis=(2, 3))

                self.assertTrue(compare_arrays("output", result, result_proper, verbose=verbose))


# ===========================================================================
class TestSpectralLossWeighted(unittest.TestCase):
    """Spectral losses (SpectralLpLoss, SpectralH1Loss) expect per-mode weights
    shaped (*, lmax, mmax) — not spatial (H, W) weights.
    These tests verify the weighting path using spectral-space weights constructed
    from fn.sht.lmax / fn.sht.mmax."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @staticmethod
    def _make(loss_type):
        return {
            "l2":   SpectralLpLoss(**_SPEC_KWARGS, squared=True),
            "h1":   SpectralH1Loss(**_SPEC_KWARGS, squared=True),
        }[loss_type]

    @parameterized.expand([("l2",), ("h1",)])
    def test_ones_weight_unchanged(self, loss_type, verbose=False):
        """All-ones spectral weight must leave the loss identical to no weight."""
        fn = self._make(loss_type)
        prd, tar = _rand(), _rand()
        wgt = torch.ones(1, 1, fn.sht.lmax, fn.sht.mmax)
        self.assertTrue(
            compare_tensors(f"{loss_type} ones wgt", fn(prd, tar, wgt), fn(prd, tar), atol=1e-5, verbose=verbose),
        )

    @parameterized.expand([("l2",), ("h1",)])
    def test_zero_weight_kills_loss(self, loss_type, verbose=False):
        """All-zeros spectral weight must produce a zero loss (no modes contribute)."""
        fn = self._make(loss_type)
        prd, tar = _rand(), _rand()
        wgt = torch.zeros(1, 1, fn.sht.lmax, fn.sht.mmax)
        loss = fn(prd, tar, wgt)
        self.assertTrue(
            compare_tensors(f"{loss_type} zero wgt", loss, torch.zeros_like(loss), atol=1e-5, verbose=verbose),
        )

    def test_l2_dc_only_weight_reduces_loss(self):
        """Keeping only the DC (l=0, m=0) mode gives strictly less L2 loss than
        the full-spectrum loss for a field that has non-trivial spatial structure."""
        fn = SpectralLpLoss(**_SPEC_KWARGS, squared=True)
        lat = torch.linspace(0, math.pi, _IMG_H)
        lon = torch.linspace(0, 2.0 * math.pi, _IMG_W)
        LAT, LON = torch.meshgrid(lat, lon, indexing="ij")
        field = (1.0 + torch.sin(LAT) * torch.cos(LON)).expand(_BATCH, _NUM_CH, -1, -1).clone()
        tar = torch.zeros_like(field)

        loss_full = fn(field, tar).mean().item()

        wgt = torch.zeros(1, 1, fn.sht.lmax, fn.sht.mmax)
        wgt[0, 0, 0, 0] = 1.0  # DC mode only
        loss_dc = fn(field, tar, wgt).mean().item()

        self.assertLess(
            loss_dc, loss_full,
            f"DC-only loss ({loss_dc:.4f}) should be < full-spectrum loss ({loss_full:.4f})",
        )

# ===========================================================================
class TestLossHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.img_shape_x = 32
        cls.img_shape_y = 64

        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name

        params = get_default_parameters()

        cls.time_diff_stds_path = os.path.join(tmp_path, "time_diff_stds.npy")
        np.save(cls.time_diff_stds_path, np.ones((1, params.N_out_channels, 1, 1), dtype=np.float64))


    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


    def setUp(self):

        disable_tf32()

        set_seed(333)

        self.params = get_default_parameters()

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = self.img_shape_x
        self.params.img_shape_y = self.img_shape_y
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_shape_x_resampled = self.params.img_shape_x
        self.params.img_shape_y_resampled = self.params.img_shape_y
        self.params.img_local_offset_x = self.params.img_crop_offset_x = 0
        self.params.img_local_offset_y = self.params.img_crop_offset_y = 0

        # also set the batch size for testing
        self.params.batch_size = 4

        # set paths
        self.params.time_diff_stds_path = self.time_diff_stds_path


    @parameterized.expand(_loss_params)
    def test_loss(self, losses, uncertainty_weighting=False):
        """
        Tests initialization of loss, as well as the forward and backward pass
        """

        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting

        # test initialization of loss object
        loss_obj = LossHandler(self.params)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        inp = torch.randn(*shape)
        inp.requires_grad = True
        tar = torch.randn(*shape)
        tar.requires_grad = True

        # forward pass and check shapes
        out = loss_obj(tar, inp)
        self.assertEqual(torch.numel(out), 1)
        self.assertTrue(out.item() >= 0.0)

        # backward pass and check gradients are not None
        out.backward()


    @parameterized.expand(_loss_params)
    def test_loss_batchsize_independence(self, losses, uncertainty_weighting=False, verbose=False):
        """
        Tests if losses are independent on batch size, in the sense that proper averaging over batch size
        is performed
        """

        self.params.losses = losses
        # not supported for bs independence:
        self.params.uncertainty_weighting = False

        # test initialization of loss object
        loss_obj = LossHandler(self.params)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        inp = torch.randn(*shape)
        tar = torch.randn(*shape)
        out = loss_obj(tar, inp)

        inp2 = torch.cat([inp, inp], dim=0)
        tar2 = torch.cat([tar, tar], dim=0)
        out2 = loss_obj(tar2, inp2)

        self.assertTrue(compare_tensors("loss", out, out2, verbose=verbose))


    @parameterized.expand(_loss_weighted_params)
    def test_loss_weighted(self, losses, uncertainty_weighting=False, verbose=False):
        """
        Tests initialization of loss, as well as the forward and backward pass
        """

        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting

        # test initialization of loss object
        loss_obj = LossHandler(self.params)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        inp = torch.randn(*shape).clone()
        inp.requires_grad = True
        tar = torch.randn(*shape).clone()
        tar.requires_grad = True
        wgt = torch.ones_like(tar)

        # forward pass and check shapes
        out = loss_obj(tar, inp)
        self.assertEqual(torch.numel(out), 1)
        self.assertTrue(out.item() >= 0.0)

        # compute weighted loss
        out_weighted = loss_obj(tar, inp, wgt)

        self.assertTrue(compare_tensors("loss", out, out_weighted, verbose=verbose))


    @parameterized.expand(_loss_weighted_params)
    def test_loss_multistep(self, losses, uncertainty_weighting=False, verbose=False):
        """
        Tests initialization of loss, as well as the forward and backward pass
        """

        self.params.n_future = 2
        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting

        # test initialization of loss object
        loss_obj = LossHandler(self.params)

        shape = (self.params.batch_size, (self.params.n_future + 1) * self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        inp = torch.randn(*shape).clone()
        inp.requires_grad = True
        tar = torch.randn(*shape).clone()
        tar.requires_grad = True
        wgt = torch.ones_like(tar)

        # forward pass and check shapes
        out = loss_obj(tar, inp)
        self.assertEqual(torch.numel(out), 1)
        self.assertTrue(out.item() >= 0.0)

        # compute weighted loss
        out_weighted = loss_obj(tar, inp, wgt)

        self.assertTrue(compare_tensors("loss", out, out_weighted, verbose=verbose))

    @parameterized.expand(_loss_zero_params)
    def test_zero_on_perfect_prediction(self, losses, uncertainty_weighting=False, verbose=False):
        """Loss must be exactly zero when prediction equals target."""
        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting
        loss_obj = LossHandler(self.params)
        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
        # use a non-zero random field so spectral losses avoid 0/0 in per-mode coherence
        prd = torch.randn(*shape)
        out = loss_obj(prd, prd)
        self.assertTrue(compare_tensors("zero loss", out, torch.zeros_like(out), atol=1e-5, verbose=verbose))

    def test_running_stats(self, verbose=False):
        """
        Tests computation of the running stats
        """

        self.params.losses = [{"type": "l2"}]

        # test initialization of loss object
        loss_obj = LossHandler(self.params, track_running_stats=True)
        loss_obj.train()

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        # this needs to be sufficiently large to mitigarte the bias due to the initialization of the running stats
        num_samples = 100
        for i in range(num_samples):

            inp = i * torch.ones(*shape)
            inp.requires_grad = True
            tar = torch.zeros(*shape)
            tar.requires_grad = True

            # forward pass and check shapes
            out = loss_obj(tar, inp)

        # generate simulated dataset
        data = torch.arange(num_samples).float().reshape(1, 1, -1).repeat(self.params.batch_size, self.params.N_out_channels, 1)
        expected_var, expected_mean = torch.var_mean(data, correction=0, dim=(0, -1))

        var, mean = loss_obj.get_running_stats()

        with self.subTest(desc="mean"):
            self.assertTrue(compare_tensors("mean", mean, expected_mean, verbose=verbose))
        with self.subTest(desc="var"):
            self.assertTrue(compare_tensors("var", var, expected_var, verbose=verbose))



# ===========================================================================
class TestCRPSLossExtended(unittest.TestCase):
    """Additional coverage for CRPSLoss: error paths, weight branches,
    and the skillspread kernel validation against properscoring."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, crps_type="cdf", **kw):
        return CRPSLoss(
            **_GEOM_KWARGS,
            crps_type=crps_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            **kw,
        )

    # ------ alpha < 1.0 raises for non-skillspread types (line 210) ------

    def test_alpha_lt1_raises_for_cdf(self):
        with self.assertRaises(NotImplementedError):
            self._fn("cdf", alpha=0.5)

    def test_alpha_lt1_raises_for_gauss(self):
        with self.assertRaises(NotImplementedError):
            self._fn("gauss", alpha=0.5)

    # ------ ensemble_weights registered as buffer (lines 220-221) ------

    def test_ensemble_weights_registered_as_buffer(self):
        """Supplying ensemble_weights must register it as a named buffer."""
        ew = torch.ones(self._E)
        fn = self._fn("cdf", ensemble_weights=ew)
        self.assertIn("ensemble_weights", dict(fn.named_buffers()))

    # ------ dim validation in forward (lines 232, 236-238) ------

    def test_wrong_forecast_dims_raises(self):
        """4-D forecasts tensor must raise ValueError (5-D expected)."""
        fn  = self._fn()
        fc  = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)   # missing ensemble dim
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(ValueError):
            fn(fc, obs)

    def test_spatial_weight_dim_mismatch_raises(self):
        """spatial_weights with fewer dims than observations must raise ValueError."""
        fn  = self._fn()
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        bad_wgt = torch.ones(_NUM_CH, _IMG_H, _IMG_W)   # 3-D; observations are 4-D
        with self.assertRaises(ValueError):
            fn(fc, obs, spatial_weights=bad_wgt)

    # ------ CDF with custom ensemble_weights (line 276) ------

    def test_cdf_with_custom_ensemble_weights_produces_finite_output(self):
        """CDF kernel must execute without error when ensemble_weights is provided."""
        E  = self._E
        ew = torch.ones(E)
        fn = self._fn("cdf", ensemble_weights=ew)
        fc  = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))
        self.assertTrue(torch.isfinite(out).all())

    # ------ skillspread + ensemble_weights raises (line 284) ------

    def test_skillspread_with_ensemble_weights_raises(self):
        """skillspread kernel does not support custom ensemble_weights."""
        E  = self._E
        ew = torch.ones(E)
        fn = self._fn("skillspread", ensemble_weights=ew)
        fc  = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(NotImplementedError):
            fn(fc, obs)

    # ------ gauss + ensemble_weights → NameError (line 292, known bug) ------

    def test_gauss_with_ensemble_weights_raises_nameerror(self):
        """Known bug: gauss branch references undefined `idx` when ensemble_weights is set."""
        E  = self._E
        ew = torch.ones(E)
        fn = self._fn("gauss", ensemble_weights=ew)
        fc  = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(NameError):
            fn(fc, obs)

    # ------ unknown crps_type raises ValueError (line 299) ------

    def test_unknown_crps_type_raises_in_forward(self):
        """Unknown crps_type must raise ValueError in forward."""
        fn = self._fn("cdf")
        fn.crps_type = "bogus"   # bypass __init__ guard; trigger the forward-time check
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(ValueError):
            fn(fc, obs)

    # ------ skillspread(alpha=0) matches properscoring.crps_ensemble ------

    def test_skillspread_alpha0_matches_properscoring(self, verbose=False):
        """crps_skillspread(alpha=0.0) is the biased CRPS and must match properscoring."""
        E  = self._E
        fn = self._fn("skillspread", alpha=0.0)
        set_seed(333)
        fc  = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        result = fn(fc, obs).cpu().numpy()

        fc_np  = fc.cpu().numpy()
        obs_np = obs.cpu().numpy()
        result_proper = crps_ensemble(obs_np, fc_np, weights=None, issorted=False, axis=1)
        quad_weight_arr = fn.quadrature.quad_weight.cpu().numpy()
        result_proper = np.sum(result_proper * quad_weight_arr, axis=(2, 3))

        self.assertTrue(compare_arrays("skillspread vs properscoring", result, result_proper, atol=1e-5, verbose=verbose))

    # ------ CDF == skillspread(alpha=0) exactly for all ensemble sizes ------

    @parameterized.expand([(2,), (5,), (10,)])
    def test_cdf_equals_skillspread_alpha0(self, ensemble_size, verbose=False):
        """CDF CRPS and skillspread(alpha=0) are the same formula; they must agree
        up to float32 rounding for every ensemble size, including E=2."""
        fn_cdf   = self._fn("cdf")
        fn_skill = self._fn("skillspread", alpha=0.0)
        set_seed(333)
        fc  = torch.randn(_BATCH, ensemble_size, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        result_cdf   = fn_cdf(fc, obs)
        result_skill = fn_skill(fc, obs)
        self.assertTrue(
            compare_tensors("cdf vs skillspread(alpha=0)", result_cdf, result_skill, atol=1e-5, rtol=1e-4, verbose=verbose),
            f"E={ensemble_size}: CDF and skillspread(alpha=0) diverged beyond float32 rounding",
        )

    # ------ CDF == skillspread(alpha=0) gradients ------

    @parameterized.expand([(2,), (5,), (10,)])
    def test_cdf_equals_skillspread_alpha0_gradients(self, ensemble_size, verbose=False):
        """CDF and skillspread(alpha=0) compute the same function; their gradients
        w.r.t. the ensemble forecasts must agree up to float32 rounding."""
        fn_cdf   = self._fn("cdf")
        fn_skill = self._fn("skillspread", alpha=0.0)
        set_seed(333)
        fc_cdf   = torch.randn(_BATCH, ensemble_size, _NUM_CH, _IMG_H, _IMG_W, requires_grad=True)
        fc_skill = fc_cdf.detach().clone().requires_grad_(True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)

        fn_cdf(fc_cdf, obs).sum().backward()
        fn_skill(fc_skill, obs).sum().backward()

        self.assertIsNotNone(fc_cdf.grad)
        self.assertIsNotNone(fc_skill.grad)
        self.assertTrue(
            compare_tensors(
                f"cdf vs skillspread(alpha=0) gradients E={ensemble_size}",
                fc_cdf.grad, fc_skill.grad, atol=1e-4, rtol=1e-3, verbose=verbose,
            ),
            f"E={ensemble_size}: CDF and skillspread(alpha=0) gradients diverged",
        )

    @parameterized.expand([("cdf",), ("skillspread",)])
    def test_gradient_sum_zero_on_perfect_prediction(self, crps_type, verbose=False):
        """Gradients summed over the ensemble dim must be zero at every pixel for a
        perfect forecast (all members == observation).  For the CDF kernel this
        requires the tail-line fix; for skillspread the antisymmetric rank
        coefficients already guarantee a zero sum, so this serves as a regression
        test for both."""
        fn  = self._fn(crps_type, alpha=0.0) if crps_type == "skillspread" else self._fn(crps_type)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc  = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        fc.requires_grad_(True)
        fn(fc, obs).sum().backward()
        self.assertFalse(torch.isnan(fc.grad).any(), f"NaN in {crps_type} gradient at perfect forecast")
        self.assertFalse(torch.isinf(fc.grad).any(), f"Inf in {crps_type} gradient at perfect forecast")
        grad_sum = fc.grad.sum(dim=1)  # sum over ensemble dim → (B, C, H, W)
        self.assertTrue(
            compare_tensors(f"{crps_type} grad ensemble sum at perfect forecast", grad_sum, torch.zeros_like(grad_sum), atol=1e-3, verbose=verbose),
        )

    # ------ fair CRPS (alpha=1) < biased CRPS (alpha=0) for spread ensemble ------

    def test_fair_crps_less_than_biased_for_spread_ensemble(self):
        """Fair CRPS (alpha=1) penalises ensemble spread less than biased CRPS (alpha=0)."""
        E = self._E
        fn_fair   = self._fn("skillspread", alpha=1.0)
        fn_biased = self._fn("skillspread", alpha=0.0)
        set_seed(333)
        obs = torch.zeros(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc  = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        self.assertLess(
            fn_fair(fc, obs).mean().item(),
            fn_biased(fc, obs).mean().item(),
        )


# ===========================================================================
class TestSpectralCRPSLoss(unittest.TestCase):
    """Full coverage for SpectralCRPSLoss."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, crps_type="skillspread", absolute=True, **kw):
        return SpectralCRPSLoss(
            **_SPEC_KWARGS,
            crps_type=crps_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            absolute=absolute,
            **kw,
        )

    # ------ type property (line 393) ------

    def test_type_property(self):
        from makani.utils.losses.base_loss import LossType
        fn = self._fn()
        self.assertEqual(fn.type, LossType.Probabilistic)

    # ------ output shape: (B, C) for all three kernels ------

    @parameterized.expand([("cdf",), ("skillspread",), ("gauss",)])
    def test_output_shape(self, crps_type):
        fn  = self._fn(crps_type)
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    # ------ non-negative output ------

    @parameterized.expand([("cdf",), ("skillspread",), ("gauss",)])
    def test_nonneg(self, crps_type):
        fn  = self._fn(crps_type)
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertTrue(
            (out >= -1e-6).all(),
            f"{crps_type}: found negative values, min={out.min().item():.4e}",
        )

    # ------ zero on perfect prediction for cdf and skillspread ------

    @parameterized.expand([("cdf",), ("skillspread",)])
    def test_zero_on_perfect_prediction(self, crps_type, verbose=False):
        """Perfect ensemble (all members = observation) must give zero loss."""
        fn  = self._fn(crps_type)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc  = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(f"spectral {crps_type} zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    # ------ absolute=False path (lines 419-426) ------

    def test_absolute_false_shape(self):
        """absolute=False folds real/imag into channels; output must still be (B, C)."""
        fn  = self._fn("skillspread", absolute=False)
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_absolute_false_nonneg(self):
        fn  = self._fn("skillspread", absolute=False)
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        out = fn(fc, obs)
        self.assertTrue((out >= -1e-6).all())

    def test_absolute_true_and_false_differ(self):
        """absolute=True and absolute=False must give different numerical results.
        Note: absolute=False only works with the skillspread kernel."""
        fn_abs  = self._fn("skillspread", absolute=True)
        fn_real = self._fn("skillspread", absolute=False)
        set_seed(333)
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        self.assertFalse(compare_tensors("abs vs real skillspread", fn_abs(fc, obs), fn_real(fc, obs)))

    # ------ dim validation in forward (lines 398-403) ------

    def test_wrong_forecast_dims_raises(self):
        fn  = self._fn()
        fc  = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)   # 4-D, not 5-D
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(ValueError):
            fn(fc, obs)

    def test_spectral_weight_dim_mismatch_raises(self):
        fn  = self._fn()
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        bad_wgt = torch.ones(_NUM_CH, fn.sht.lmax, fn.sht.mmax)   # 3-D; obs are 4-D
        with self.assertRaises(ValueError):
            fn(fc, obs, spectral_weights=bad_wgt)

    # ------ error paths inside forward ------

    def test_alpha_lt1_raises_for_cdf(self):
        with self.assertRaises(NotImplementedError):
            self._fn("cdf", alpha=0.5)

    def test_skillspread_with_ensemble_weights_raises(self):
        E  = self._E
        ew = torch.ones(E)
        fn = SpectralCRPSLoss(
            **_SPEC_KWARGS,
            crps_type="skillspread",
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=ew,
            absolute=True,
        )
        fc  = torch.randn(_BATCH, E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(NotImplementedError):
            fn(fc, obs)

    def test_unknown_crps_type_raises_in_forward(self):
        fn = self._fn("cdf")
        fn.crps_type = "bogus"
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        with self.assertRaises(ValueError):
            fn(fc, obs)

    # ------ E=1 shortcut (lines 440-443) ------

    def test_e1_gives_zero_for_perfect_prediction(self, verbose=False):
        """With E=1, spectral CRPS = |SHT(obs - fc)| which is 0 when obs == fc."""
        fn  = self._fn("skillspread")
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fc  = obs.unsqueeze(1).clone()   # (B, 1, C, H, W)
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors("spectral e1 zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    # ------ backward pass produces finite gradients ------

    def test_backward_finite(self):
        fn  = self._fn("skillspread")
        fc  = torch.randn(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W, requires_grad=True)
        obs = torch.randn(_BATCH, _NUM_CH, _IMG_H, _IMG_W)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")


# ===========================================================================
class TestSobolevEnergyScoreLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name
        params = get_default_parameters()
        cls.time_diff_stds_path = os.path.join(tmp_path, "time_diff_stds.npy")
        np.save(cls.time_diff_stds_path, np.ones((1, params.N_out_channels, 1, 1), dtype=np.float64))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.params = get_default_parameters()
        self.params.img_shape_x = 32
        self.params.img_shape_y = 64
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_shape_x_resampled = self.params.img_shape_x
        self.params.img_shape_y_resampled = self.params.img_shape_y
        self.params.img_local_offset_x = self.params.img_crop_offset_x = 0
        self.params.img_local_offset_y = self.params.img_crop_offset_y = 0
        self.params.batch_size = 4
        self.params.time_diff_stds_path = self.time_diff_stds_path

    @parameterized.expand([
        # (beta, alpha, offset, fraction, channel_reduction)
        (0.5, 1.0, 1.0, 1.0, True),
        (1.0, 1.0, 1.0, 1.0, True),
        (2.0, 1.0, 1.0, 1.0, True),
        (1.0, 0.5, 1.0, 1.0, True),
        (1.0, 2.0, 1.0, 1.0, True),
        (1.0, 1.0, 0.5, 1.0, True),
        (1.0, 1.0, 2.0, 1.0, True),
        (1.0, 1.0, 1.0, 0.5, True),
        (1.0, 1.0, 1.0, 2.0, True),
        (1.0, 1.0, 1.0, 1.0, False),
        (0.5, 0.5, 0.5, 0.5, True),
        (2.0, 2.0, 2.0, 2.0, True),
    ])
    def test_sobolev_energy_score(self, beta, alpha, offset, fraction, channel_reduction):
        """
        Tests SobolevEnergyScoreLoss for different parameter combinations,
        verifying that output and gradients are not NaN or inf.
        """
        sobolev_loss = SobolevEnergyScoreLoss(
            img_shape=(self.params.img_shape_x, self.params.img_shape_y),
            crop_shape=(self.params.img_shape_x, self.params.img_shape_y),
            crop_offset=(0, 0),
            channel_names=self.params.channel_names,
            grid_type=self.params.model_grid_type,
            lmax=None,
            spatial_distributed=False,
            ensemble_distributed=False,
            channel_reduction=channel_reduction,
            alpha=alpha,
            beta=beta,
            offset=offset,
            fraction=fraction,
        ).to(self.device)

        for ensemble_size in [2, 6]:
            with self.subTest(desc=f"beta={beta}, alpha={alpha}, offset={offset}, fraction={fraction}, channel_reduction={channel_reduction}, ensemble_size={ensemble_size}"):
                # Generate forecast tensor: (batch, ensemble, channels, lat, lon)
                forecasts = torch.randn(
                    self.params.batch_size,
                    ensemble_size,
                    self.params.N_in_channels,
                    self.params.img_shape_x,
                    self.params.img_shape_y,
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=True,
                )

                # Generate observation tensor: (batch, channels, lat, lon)
                observations = torch.randn(
                    self.params.batch_size,
                    self.params.N_in_channels,
                    self.params.img_shape_x,
                    self.params.img_shape_y,
                    device=self.device,
                    dtype=torch.float32,
                )

                # Forward pass
                result = sobolev_loss(forecasts, observations)

                # Check output is not NaN or inf
                self.assertFalse(torch.isnan(result).any(), f"Output contains NaN values")
                self.assertFalse(torch.isinf(result).any(), f"Output contains inf values")

                # Backward pass
                loss = result.sum()
                loss.backward()

                # Check gradients are not NaN or inf
                self.assertIsNotNone(forecasts.grad, "Gradients are None")
                self.assertFalse(torch.isnan(forecasts.grad).any(), f"Gradients contain NaN values")
                self.assertFalse(torch.isinf(forecasts.grad).any(), f"Gradients contain inf values")


# ===========================================================================
class TestLpEnergyScoreLoss(unittest.TestCase):
    """Tests for LpEnergyScoreLoss (and the L2EnergyScoreLoss alias)."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, channel_reduction=True, p=2.0, **kw):
        return LpEnergyScoreLoss(
            **_GEOM_KWARGS,
            spatial_distributed=False,
            ensemble_distributed=False,
            channel_reduction=channel_reduction,
            p=p,
            **kw,
        )

    def test_output_shape_channel_reduction(self):
        fn = self._fn(channel_reduction=True)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, 1))

    def test_output_shape_no_channel_reduction(self):
        fn = self._fn(channel_reduction=False)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    def test_zero_on_perfect_prediction(self, verbose=False):
        fn = self._fn()
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors("lp_es zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    @parameterized.expand([(1.0,), (2.0,), (4.0,)])
    def test_p_parameter_changes_output(self, p):
        """Different p values must produce different loss values for a spread ensemble."""
        fn_ref = self._fn(p=2.0)
        fn_p = self._fn(p=p)
        set_seed(333)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        if p == 2.0:
            self.assertTrue(compare_tensors("same p", fn_ref(fc, obs), fn_p(fc, obs)))
        else:
            self.assertFalse(compare_tensors("diff p", fn_ref(fc, obs), fn_p(fc, obs)))

    def test_batch_independence(self, verbose=False):
        fn = self._fn()
        fc = _rand_ensemble(self._E)
        obs = _rand()
        loss_single = fn(fc[:1], obs[:1])
        loss_batch = fn(fc, obs)
        self.assertTrue(
            compare_tensors("lp_es batch", loss_single[0], loss_batch[0], verbose=verbose),
        )


# ===========================================================================
class TestSpectralL2EnergyScoreLoss(unittest.TestCase):
    """Tests for SpectralL2EnergyScoreLoss."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, channel_reduction=True, **kw):
        return SpectralL2EnergyScoreLoss(
            **_SPEC_KWARGS,
            spatial_distributed=False,
            ensemble_distributed=False,
            channel_reduction=channel_reduction,
            **kw,
        )

    def test_output_shape_channel_reduction(self):
        fn = self._fn(channel_reduction=True)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, 1))

    def test_output_shape_no_channel_reduction(self):
        fn = self._fn(channel_reduction=False)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    def test_zero_on_perfect_prediction(self, verbose=False):
        fn = self._fn()
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors("spec_l2_es zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    def test_batch_independence(self, verbose=False):
        fn = self._fn()
        fc = _rand_ensemble(self._E)
        obs = _rand()
        loss_single = fn(fc[:1], obs[:1])
        loss_batch = fn(fc, obs)
        self.assertTrue(
            compare_tensors("spec_l2_es batch", loss_single[0], loss_batch[0], verbose=verbose),
        )


# ===========================================================================
class TestSpectralRegularization(unittest.TestCase):
    """Tests for SpectralRegularization."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, **kw):
        return SpectralRegularization(
            **_SPEC_KWARGS,
            spatial_distributed=False,
            ensemble_distributed=False,
            **kw,
        )

    def test_output_shape_5d(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_output_shape_4d(self):
        """4-D input (no ensemble dim) must also work."""
        fn = self._fn()
        fc = _rand()
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    def test_zero_on_perfect_prediction(self, verbose=False):
        fn = self._fn()
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors("spec_reg zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    def test_nonneg(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertTrue((out >= -1e-6).all(), f"found negative values, min={out.min().item():.4e}")

    def test_logarithmic_mode(self):
        """logarithmic=True must produce finite output."""
        fn = self._fn(logarithmic=True)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertTrue(torch.isfinite(out).all())

    def test_wrong_forecast_dims_raises(self):
        fn = self._fn()
        fc = torch.randn(1, 2, 3)
        obs = torch.randn(1, 2, 3)
        with self.assertRaises(ValueError):
            fn(fc, obs)


# ===========================================================================
class TestGradientCRPSLoss(unittest.TestCase):
    """Tests for GradientCRPSLoss."""

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, crps_type="skillspread", absolute=True, **kw):
        return GradientCRPSLoss(
            **_GEOM_KWARGS,
            crps_type=crps_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            absolute=absolute,
            **kw,
        )

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_output_shape(self, crps_type):
        fn = self._fn(crps_type)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, _NUM_CH))

    def test_output_shape_absolute_false(self):
        fn = self._fn("skillspread", absolute=False)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertEqual(tuple(out.shape), (_BATCH, 2 * _NUM_CH))

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_nonneg(self, crps_type):
        fn = self._fn(crps_type)
        fc = _rand_ensemble(self._E)
        obs = _rand()
        out = fn(fc, obs)
        self.assertTrue(
            (out >= -1e-6).all(),
            f"{crps_type}: found negative values, min={out.min().item():.4e}",
        )

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, requires_grad=True)
        obs = _rand()
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_zero_on_perfect_prediction(self, crps_type, verbose=False):
        fn = self._fn(crps_type)
        obs = _rand()
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(f"grad_crps {crps_type} zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    def test_wrong_forecast_dims_raises(self):
        fn = self._fn()
        fc = _rand()
        obs = _rand()
        with self.assertRaises(ValueError):
            fn(fc, obs)


# ===========================================================================
class TestVortDivCRPSLoss(unittest.TestCase):
    """Tests for VortDivCRPSLoss.

    Requires channel_names with u/v wind pairs. Uses _WIND_GEOM_KWARGS
    which has ["u500", "v500", "u850", "v850", "t500"].
    """

    _E = 5

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _fn(self, crps_type="skillspread", **kw):
        return VortDivCRPSLoss(
            **_WIND_GEOM_KWARGS,
            crps_type=crps_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            **kw,
        )

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_output_shape(self, crps_type):
        fn = self._fn(crps_type)
        fc = _rand_ensemble(self._E, channels=_NUM_WIND_CH)
        obs = _rand(channels=_NUM_WIND_CH)
        out = fn(fc, obs)
        n_wind = fn.wind_chans.shape[0]
        self.assertEqual(tuple(out.shape), (_BATCH, n_wind))

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_nonneg(self, crps_type):
        fn = self._fn(crps_type)
        fc = _rand_ensemble(self._E, channels=_NUM_WIND_CH)
        obs = _rand(channels=_NUM_WIND_CH)
        out = fn(fc, obs)
        self.assertTrue(
            (out >= -1e-6).all(),
            f"{crps_type}: found negative values, min={out.min().item():.4e}",
        )

    def test_backward_finite(self):
        fn = self._fn()
        fc = _rand_ensemble(self._E, channels=_NUM_WIND_CH, requires_grad=True)
        obs = _rand(channels=_NUM_WIND_CH)
        fn(fc, obs).sum().backward()
        self.assertIsNotNone(fc.grad)
        self.assertFalse(torch.isnan(fc.grad).any(), "NaN in fc.grad")
        self.assertFalse(torch.isinf(fc.grad).any(), "Inf in fc.grad")

    @parameterized.expand([("skillspread",), ("cdf",)])
    def test_zero_on_perfect_prediction(self, crps_type, verbose=False):
        fn = self._fn(crps_type)
        obs = _rand(channels=_NUM_WIND_CH)
        fc = obs.unsqueeze(1).expand(_BATCH, self._E, _NUM_WIND_CH, _IMG_H, _IMG_W).clone()
        out = fn(fc, obs)
        self.assertTrue(
            compare_tensors(f"vortdiv {crps_type} zero", out, torch.zeros_like(out), atol=1e-4, verbose=verbose),
        )

    def test_wrong_forecast_dims_raises(self):
        fn = self._fn()
        fc = _rand(channels=_NUM_WIND_CH)
        obs = _rand(channels=_NUM_WIND_CH)
        with self.assertRaises(ValueError):
            fn(fc, obs)


if __name__ == "__main__":
    disable_tf32()
    unittest.main()
