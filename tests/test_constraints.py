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

import os
import sys
import unittest

import numpy as np

import torch

import makani.utils.constants as const
from makani.utils.losses.hydrostatic_loss import HydrostaticBalanceLoss
from makani.models.parametrizations import ConstraintsWrapper
from makani.utils.constraints import NonNegativeConstraint, HydrostaticBalanceProjection, get_matching_channels_pl

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

class TestConstraints(unittest.TestCase):

    def setUp(self):

        disable_tf32()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(333)

        # load the data:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        data = np.load(os.path.join(data_dir, "sample_30km_equator.npz"))

        # fields
        self.data = torch.from_numpy(data["data"].astype(np.float32))
        self.bias = torch.from_numpy(data["bias"].astype(np.float32))
        self.scale = torch.from_numpy(data["scale"].astype(np.float32))
        self.data = ((self.data - self.bias) / self.scale).to(self.device)
        # metadata
        self.channel_names = data["channel_names"].tolist()
        self.img_shape = data["img_shape"]
        self.crop_shape = data["crop_shape"]
        self.crop_offset = data["crop_offset"]

    
    def test_hydrostatic_balance_loss(self):
        for use_moist_air_formula in [False, True]:
            with self.subTest(f"moist air formula: {use_moist_air_formula}"):
                # loss
                hbloss = HydrostaticBalanceLoss(img_shape=self.img_shape,
                                                crop_shape=self.crop_shape,
                                                crop_offset=self.crop_offset,
                                                channel_names=self.channel_names,
                                                grid_type="equiangular",
                                                bias=self.bias,
                                                scale=self.scale,
                                                p_min=50,
                                                p_max=900,
                                                use_moist_air_formula=use_moist_air_formula).to(self.device)
                loss_tens = hbloss(self.data, None)
                
                # average over batch and sum over channels
                loss_val = torch.mean(torch.sum(loss_tens, dim=1)).item()
                
                self.assertTrue(loss_val <= 1e-4)
                
    def test_hydrostatic_balance_constraint_wrapper_era5(self):
        for use_moist_air_formula in [False, True]:
            with self.subTest(f"moist air formula: {use_moist_air_formula}"):
                # loss
                hbloss = HydrostaticBalanceLoss(img_shape=self.img_shape,
                                                crop_shape=self.crop_shape,
                                                crop_offset=self.crop_offset,
                                                channel_names=self.channel_names,
                                                grid_type="equiangular",
                                                bias=self.bias,
                                                scale=self.scale,
                                                p_min=50,
                                                p_max=900,
                                                use_moist_air_formula=use_moist_air_formula).to(self.device)

                # constraints wrapper
                constraint_dict = {"type": "hydrostatic_balance",
                                   "options": dict(p_min=50, p_max=900,
                                                   use_moist_air_formula=use_moist_air_formula)}
                cwrap = ConstraintsWrapper(constraints=[constraint_dict],
                                           channel_names=self.channel_names,
                                           bias=self.bias, scale=self.scale,
                                           model_handle=None).to(self.device)

                # create a short vector:
                B, C, H, W = self.data.shape
                data_short = torch.empty((B, cwrap.N_in_channels, H, W), dtype=torch.float32, device=self.device)
                # t_idx
                data_short[:, 0, ...] = self.data[:, cwrap.constraint_list[0].t_idx[0], ...]
                # z_idx
                data_short[:, 1:len(cwrap.constraint_list[0].z_idx)+1, ...] = self.data[:, cwrap.constraint_list[0].z_idx, ...]
                # q_idx
                off_idx = len(cwrap.constraint_list[0].z_idx)+1
                if use_moist_air_formula:
                    data_short[:, off_idx:off_idx+len(cwrap.constraint_list[0].q_idx), ...] = self.data[:, cwrap.constraint_list[0].q_idx, ...]
                    off_idx += len(cwrap.constraint_list[0].q_idx)
                # remaining channels
                data_short[:, off_idx:, ...] = self.data[:, cwrap.constraint_list[0].aux_idx, ...]
                data_map = cwrap(data_short)
                
                # check the hb loss
                hb_loss_tens = hbloss(data_map, None)

                # average over batch and sum over channels
                hb_loss_val = torch.mean(torch.sum(hb_loss_tens, dim=1)).item()
                
                self.assertTrue(hb_loss_val <= 1e-6)

                # now check that the loss on the non-hb components is zero too
                aux_loss_val = torch.nn.functional.mse_loss(data_map[:, cwrap.constraint_list[0].aux_idx, ...],
                                                            self.data[:, cwrap.constraint_list[0].aux_idx, ...]).item()
                self.assertTrue(aux_loss_val <= 1e-6)

    def test_hydrostatic_balance_constraint_wrapper_random(self):
        for use_moist_air_formula in [False, True]:
            with self.subTest(f"moist air formula: {use_moist_air_formula}"):
                # loss
                hbloss = HydrostaticBalanceLoss(img_shape=self.img_shape,
                                                crop_shape=self.crop_shape,
                                                crop_offset=self.crop_offset,
                                                channel_names=self.channel_names,
                                                grid_type="equiangular",
                                                bias=self.bias,
                                                scale=self.scale,
                                                p_min=50,
                                                p_max=900,
                                                use_moist_air_formula=use_moist_air_formula).to(self.device)

                # constraints wrapper
                constraint_dict = {"type": "hydrostatic_balance",
                                   "options": dict(p_min=50, p_max=900,
                                                   use_moist_air_formula=use_moist_air_formula)}
                cwrap = ConstraintsWrapper(constraints=[constraint_dict],
                                           channel_names=self.channel_names,
                                           bias=self.bias, scale=self.scale,
                                           model_handle=None).to(self.device)

                # create a short vector:
                B, C, H, W = self.data.shape
                data_short = torch.empty((B, cwrap.N_in_channels, H, W), dtype=torch.float32, device=self.device)
                data_short.normal_(0., 1.)
                data_map = cwrap(data_short)
                
                # check the hb loss
                hb_loss_tens = hbloss(data_map, None)

                # average over batch and sum over channels 
                hb_loss_val = torch.mean(torch.sum(hb_loss_tens, dim=1)).item()

                self.assertTrue(hb_loss_val <= 1e-6)
                
                # now check that the loss on the non-hb components is zero too
                off_idx = len(cwrap.constraint_list[0].z_idx)+1
                if use_moist_air_formula:
                    off_idx += len(cwrap.constraint_list[0].q_idx)
                aux_loss_val = torch.nn.functional.mse_loss(data_map[:, cwrap.constraint_list[0].aux_idx, ...],
                                                            data_short[:, off_idx:, ...]).item()
                self.assertTrue(aux_loss_val <= 1e-6)

    def test_hydrostatic_balance_matches_independent_integration(self):
        """Independent validation of the hydrostatic-balance formula.

        Instead of checking the wrapper output against HydrostaticBalanceLoss (which
        derives from the same formula -> circular), we build a *known* temperature
        profile, integrate the hypsometric equation forward to obtain geopotentials,
        feed [T0, Z...] through the wrapper, and confirm it reconstructs the original
        temperatures (and passes geopotential / humidity / aux channels through).

            Phi_i - Phi_{i-1} = R_d * 0.5 * (Tv_i + Tv_{i-1}) * ln(p_{i-1}/p_i)
        """
        import makani.utils.constants as const

        R = const.R_DRY_AIR
        qc = const.Q_CORRECTION_MOIST_AIR

        for use_moist_air_formula in [False, True]:
            with self.subTest(f"moist air formula: {use_moist_air_formula}"):
                # synthetic channel set: matching t/z (and q) pressure levels + aux channels
                levels = [925, 850, 700, 500, 300, 100, 50]
                channel_names = [f"t{p}" for p in levels] + [f"z{p}" for p in levels]
                if use_moist_air_formula:
                    channel_names += [f"q{p}" for p in levels]
                channel_names += ["u10m", "v10m", "t2m"]  # aux passthrough channels

                constraint_dict = {"type": "hydrostatic_balance",
                                   "options": dict(p_min=0, p_max=2000,
                                                   use_moist_air_formula=use_moist_air_formula)}
                cwrap = ConstraintsWrapper(constraints=[constraint_dict],
                                           channel_names=channel_names,
                                           bias=None, scale=None,
                                           model_handle=None).to(self.device)
                con = cwrap.constraint_list[0]

                # the wrapper orders pressures descending (bottom -> top)
                pressures = con.pressures
                n = len(pressures)
                self.assertEqual(n, len(levels))
                self.assertEqual(len(con.t_idx), n)

                B, H, W = 2, 3, 4
                # known temperature profile (physical units, varying over space)
                T = 200.0 + 60.0 * torch.rand(B, n, H, W, device=self.device, dtype=torch.float32)
                if use_moist_air_formula:
                    q = 0.02 * torch.rand(B, n, H, W, device=self.device, dtype=torch.float32)
                    Tv = T * (1.0 + qc * q)
                else:
                    Tv = T

                # integrate the hypsometric equation forward to get geopotential per level
                Z = torch.zeros(B, n, H, W, device=self.device, dtype=torch.float32)
                Z[:, 0, ...] = 1000.0  # arbitrary reference geopotential at the bottom level
                for i in range(1, n):
                    plog = float(np.log(pressures[i - 1] / pressures[i]))
                    Z[:, i, ...] = Z[:, i - 1, ...] + R * 0.5 * (Tv[:, i, ...] + Tv[:, i - 1, ...]) * plog

                # assemble the reduced input in the wrapper's layout: [T0, Z0..Z_{n-1}, (q...), aux...]
                inp = torch.zeros(B, cwrap.N_in_channels, H, W, device=self.device, dtype=torch.float32)
                inp[:, 0, ...] = T[:, 0, ...]
                inp[:, 1:n + 1, ...] = Z
                off = n + 1
                if use_moist_air_formula:
                    inp[:, off:off + n, ...] = q
                    off += n
                n_aux = len(con.aux_idx)
                aux_vals = torch.randn(B, n_aux, H, W, device=self.device, dtype=torch.float32)
                inp[:, off:off + n_aux, ...] = aux_vals

                out = cwrap(inp)

                # the reconstructed temperatures must match the known profile
                self.assertTrue(compare_tensors("reconstructed temperature", out[:, con.t_idx, ...], T,
                                                atol=1e-1, rtol=1e-3, verbose=True))
                # geopotentials, aux (and humidity) pass through unchanged
                self.assertTrue(compare_tensors("geopotential passthrough", out[:, con.z_idx, ...], Z,
                                                atol=1e-2, rtol=1e-5, verbose=True))
                self.assertTrue(compare_tensors("aux passthrough", out[:, con.aux_idx, ...], aux_vals,
                                                atol=1e-4, rtol=1e-5, verbose=True))
                if use_moist_air_formula:
                    self.assertTrue(compare_tensors("humidity passthrough", out[:, con.q_idx, ...], q,
                                                    atol=1e-4, rtol=1e-5, verbose=True))

    def test_hydrostatic_balance_loss_on_balanced_profile(self):
        """Independent check of the soft-constraint loss: a profile built to satisfy
        hydrostatic balance exactly yields ~0 loss, and perturbing a single geopotential
        level makes the loss clearly positive. Uses identity normalization so a synthetic
        physical-unit field is fed straight through (no dependence on the data sample)."""
        import makani.utils.constants as const

        R = const.R_DRY_AIR
        qc = const.Q_CORRECTION_MOIST_AIR

        C = len(self.channel_names)
        # Use a full (uncropped) equiangular grid. The data sample is a 2-row band at
        # the pole (crop_offset=[719,0], crop_shape=[2,720]); the area-weighted spherical
        # quadrature gives that crop a normalized weight ~1e-5 (cos(lat) -> 0 at the pole),
        # which scales the *integrated* loss down by ~1e-5 and makes absolute thresholds
        # meaningless. On a full grid the quadrature is O(1).
        nlat, nlon = 32, 64
        img_shape = crop_shape = (nlat, nlon)
        crop_offset = (0, 0)
        H, W = nlat, nlon
        ident_bias = torch.zeros(1, C, 1, 1, dtype=torch.float32)
        ident_scale = torch.ones(1, C, 1, 1, dtype=torch.float32)

        for use_moist_air_formula in [False, True]:
            with self.subTest(f"moist air formula: {use_moist_air_formula}"):
                hbloss = HydrostaticBalanceLoss(img_shape=img_shape, crop_shape=crop_shape,
                                                crop_offset=crop_offset, channel_names=self.channel_names,
                                                grid_type="equiangular", bias=ident_bias, scale=ident_scale,
                                                p_min=50, p_max=900,
                                                use_moist_air_formula=use_moist_air_formula).to(self.device)
                pressures = hbloss.pressures
                n = len(pressures)
                B = 2

                # known temperature profile (physical units); aux channels arbitrary
                field = torch.randn(B, C, H, W, device=self.device)
                T = 200.0 + 60.0 * torch.rand(B, n, H, W, device=self.device)
                if use_moist_air_formula:
                    q = 0.02 * torch.rand(B, n, H, W, device=self.device)
                    Tv = T * (1.0 + qc * q)
                    field[:, hbloss.q_idx, ...] = q
                else:
                    Tv = T

                # integrate the hypsometric equation to get balanced geopotentials
                Z = torch.zeros(B, n, H, W, device=self.device)
                Z[:, 0, ...] = 1000.0
                for i in range(1, n):
                    plog = float(np.log(pressures[i - 1] / pressures[i]))
                    Z[:, i, ...] = Z[:, i - 1, ...] + R * 0.5 * (Tv[:, i, ...] + Tv[:, i - 1, ...]) * plog
                field[:, hbloss.t_idx, ...] = T
                field[:, hbloss.z_idx, ...] = Z

                # a balanced profile must give (numerically) zero loss
                loss_bal = torch.mean(torch.sum(hbloss(field, None), dim=1)).item()
                self.assertTrue(loss_bal <= 1e-4, f"balanced HB loss too large: {loss_bal}")

                # perturbing one geopotential level breaks balance -> loss clearly positive
                field_pert = field.clone()
                field_pert[:, hbloss.z_idx[1], ...] += 50.0
                loss_pert = torch.mean(torch.sum(hbloss(field_pert, None), dim=1)).item()
                self.assertTrue(loss_pert >= 1e-2, f"perturbed HB loss unexpectedly small: {loss_pert}")
                # and it must be vastly larger than the balanced residual
                self.assertGreater(loss_pert, 1e3 * loss_bal)


class TestNonNegativeConstraint(unittest.TestCase):
    """Tests for NonNegativeConstraint in makani/utils/constraints.py.

    Convention: x_norm = (x_raw - bias) / scale, so physical zero sits at
    x_norm = -bias/scale.  In eval mode the constraint hard-clamps to guarantee
    x_raw >= 0; in training mode it uses a smooth multiplicative approximation
    so gradients flow for slightly negative values.
    """

    # synthetic channel set used across all subtests
    ALL_CHANNELS = ["u10m", "q850", "t850", "q500", "t500"]
    CLAMP_NAMES  = ["q850", "q500"]
    CLAMP_IDX    = [1, 3]  # positions of CLAMP_NAMES in ALL_CHANNELS

    def setUp(self):
        disable_tf32()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(333)

    def _make(self, names_to_clamp=None, means=None, stds=None, **kwargs):
        """Build a NonNegativeConstraint using channel names.

        means/stds are full-channel tensors (len(ALL_CHANNELS),); the
        constructor slices out the constrained channels itself, mirroring
        how _HydrostaticBalanceWrapper receives the full bias/scale.
        """
        names = names_to_clamp if names_to_clamp is not None else self.CLAMP_NAMES
        bias  = means.view(1, -1, 1, 1) if means is not None else None
        scale = stds.view(1, -1, 1, 1)  if stds  is not None else None
        c = NonNegativeConstraint(self.ALL_CHANNELS, names, bias=bias, scale=scale, **kwargs)
        return c.to(self.device)

    # --- eval / hard clamp ---
    def test_eval_hard_clamp_no_normalization(self):
        """Eval mode: constrained channels are >= 0; unconstrained channels unchanged."""
        B, C, H, W = 2, len(self.ALL_CHANNELS), 8, 8
        c = self._make()
        c.eval()
        x = torch.randn(B, C, H, W, device=self.device)
        y = c(x)
        self.assertTrue((y[:, self.CLAMP_IDX, :, :] >= 0).all().item())
        unconstrained = [i for i in range(C) if i not in self.CLAMP_IDX]
        self.assertTrue(compare_tensors("unconstrained channels", y[:, unconstrained, :, :], x[:, unconstrained, :, :]))

    def test_eval_hard_clamp_with_normalization(self):
        """Eval mode: x_raw = y_norm * scale + bias >= 0 after clamping."""
        B, C, H, W = 2, len(self.ALL_CHANNELS), 6, 6
        means = torch.tensor([0.0, 5.0, 270.0, 3.0, 250.0])
        stds  = torch.tensor([1.0, 2.0,  10.0, 1.5,   8.0])
        c = self._make(means=means, stds=stds)
        c.eval()
        x = torch.randn(B, C, H, W, device=self.device) * 3.0
        y = c(x)
        for i, ci in enumerate(self.CLAMP_IDX):
            x_raw = y[:, ci, :, :] * stds[ci].item() + means[ci].item()
            self.assertTrue((x_raw >= -1e-6).all().item(), f"channel {self.ALL_CHANNELS[ci]} has negative physical values")

    def test_eval_positive_input_unchanged(self):
        """Eval mode: values already above physical zero are not modified."""
        B, C, H, W = 2, len(self.ALL_CHANNELS), 4, 4
        means = torch.tensor([0.0, 1.0, 270.0, 2.0, 250.0])
        stds  = torch.ones(len(self.ALL_CHANNELS))
        c = self._make(means=means, stds=stds)
        c.eval()
        x = torch.ones(B, C, H, W, device=self.device) * 5.0
        y = c(x)
        self.assertTrue(compare_tensors("positive inputs unchanged", y, x))

    # --- training / soft clamp ---
    def test_train_slightly_negative_not_zeroed(self):
        """Training mode: slightly negative values are not exactly zeroed (gradient path open)."""
        B, C, H, W = 1, len(self.ALL_CHANNELS), 4, 4
        c = self._make(names_to_clamp=["q850"])
        c.train()
        x = torch.full((B, C, H, W), -0.05, device=self.device)
        y = c(x)
        self.assertFalse((y[:, [1], :, :] == 0).all().item())

    def test_train_large_positive_identity(self):
        """Training mode: large positive values pass through essentially unchanged."""
        B, C, H, W = 2, len(self.ALL_CHANNELS), 4, 4
        c = self._make(eps=0.1)
        c.train()
        x = torch.ones(B, C, H, W, device=self.device) * 5.0
        y = c(x)
        self.assertTrue(compare_tensors("large positive passthrough", y[:, self.CLAMP_IDX, :, :], x[:, self.CLAMP_IDX, :, :], atol=1e-3))

    def test_train_gradient_flows(self):
        """Training mode: gradient is nonzero for slightly negative inputs."""
        B, C, H, W = 1, len(self.ALL_CHANNELS), 4, 4
        c = self._make()
        c.train()
        x = torch.full((B, C, H, W), -0.2, device=self.device, requires_grad=True)
        c(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse((x.grad[:, self.CLAMP_IDX, :, :] == 0).all().item())

    def test_train_normalization_offset(self):
        """Training mode: with normalization, the clamp boundary is at physical zero."""
        B, C, H, W = 1, len(self.ALL_CHANNELS), 4, 4
        means = torch.tensor([0.0, 4.0, 270.0, 6.0, 250.0])
        stds  = torch.tensor([1.0, 2.0,  10.0, 3.0,   8.0])
        c = self._make(means=means, stds=stds, eps=0.01)
        c.train()
        # set constrained channels to physical zero in normalized space
        x = torch.zeros(B, C, H, W, device=self.device)
        for ci, mi, si in zip(self.CLAMP_IDX, means[self.CLAMP_IDX], stds[self.CLAMP_IDX]):
            x[:, ci, :, :] = -mi / si
        y = c(x)
        for ci, mi, si in zip(self.CLAMP_IDX, means[self.CLAMP_IDX], stds[self.CLAMP_IDX]):
            x_raw = y[:, ci, :, :] * si.item() + mi.item()
            self.assertTrue(compare_tensors(f"{self.ALL_CHANNELS[ci]} at boundary",
                                            x_raw, torch.zeros_like(x_raw), atol=0.1))

    # --- mode switching ---
    def test_train_eval_switch(self):
        """Switching train/eval changes hard vs soft clamping on the same instance."""
        B, C, H, W = 1, len(self.ALL_CHANNELS), 4, 4
        c = self._make(names_to_clamp=["q850"])
        x = torch.full((B, C, H, W), -1.0, device=self.device)
        c.train()
        y_train = c(x)
        c.eval()
        y_eval = c(x)
        self.assertFalse((y_train[:, [1], :, :] == 0).all().item())
        self.assertTrue(compare_tensors("hard clamp to zero", y_eval[:, [1], :, :],
                                        torch.zeros_like(y_eval[:, [1], :, :])))


class TestHydrostaticBalanceProjection(unittest.TestCase):
    """Tests for HydrostaticBalanceProjection in makani/utils/constraints.py.

    The layer takes a freely-predicted (T, Z) state and applies the minimum-norm
    correction onto the hydrostatic-balance manifold

        Z_i - Z_{i-1} = c_i (Tv_i + Tv_{i-1}),   c_i = 0.5 * R_dry * ln(p_{i-1}/p_i),

    with Tv = T (dry) or Tv = T(1 + eps q) (moist, q held fixed). At strength 1
    the output satisfies balance exactly; the residual scales as (1 - strength).
    """

    LEVELS = [925, 850, 700, 500, 300, 100, 50]
    P_MIN, P_MAX = 0, 2000  # window that admits all of LEVELS

    def setUp(self):
        disable_tf32()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(333)

    # --- helpers ---
    def _channels(self, moist):
        names = [f"t{p}" for p in self.LEVELS] + [f"z{p}" for p in self.LEVELS]
        if moist:
            names += [f"q{p}" for p in self.LEVELS]
        names += ["u10m", "v10m", "t2m"]  # aux passthrough channels
        return names

    def _stats(self, channel_names, identity):
        """Per-channel (bias, scale) with realistic magnitudes, or None for identity."""
        if identity:
            return None, None
        C = len(channel_names)
        bias = torch.zeros(1, C, 1, 1)
        scale = torch.ones(1, C, 1, 1)
        for i, nm in enumerate(channel_names):
            if nm.startswith("z"):
                bias[0, i, 0, 0], scale[0, i, 0, 0] = 5.0e4, 3.0e3
            elif nm.startswith("q"):
                bias[0, i, 0, 0], scale[0, i, 0, 0] = 5.0e-3, 5.0e-3
            elif nm.startswith("t"):
                bias[0, i, 0, 0], scale[0, i, 0, 0] = 250.0, 30.0
        return bias.to(self.device), scale.to(self.device)

    def _split(self, channel_names, moist):
        z_idx, t_idx, pressures = get_matching_channels_pl(channel_names, "z", "t", self.P_MIN, self.P_MAX)
        q_idx = None
        if moist:
            q_idx, _, _ = get_matching_channels_pl(channel_names, "q", "t", self.P_MIN, self.P_MAX)
        return t_idx, z_idx, q_idx, pressures

    def _residual(self, out, channel_names, moist, bias, scale):
        """Relative hydrostatic-balance residual per interior level, in physical units."""
        t_idx, z_idx, q_idx, pressures = self._split(channel_names, moist)

        def unorm(idx):
            v = out[:, idx, ...].float()
            if bias is not None:
                v = v * scale[:, idx, ...] + bias[:, idx, ...]
            return v

        T, Z = unorm(t_idx), unorm(z_idx)
        Tv = T * (1.0 + const.Q_CORRECTION_MOIST_AIR * unorm(q_idx)) if moist else T

        res, denom = [], []
        for i in range(1, len(pressures)):
            c = 0.5 * const.R_DRY_AIR * float(np.log(pressures[i - 1] / pressures[i]))
            res.append((Z[:, i] - Z[:, i - 1]) - c * (Tv[:, i] + Tv[:, i - 1]))
            denom.append((Z[:, i] - Z[:, i - 1]).abs() + c * (Tv[:, i].abs() + Tv[:, i - 1].abs()))
        return torch.stack(res, dim=1), torch.stack(denom, dim=1)

    def _balanced_field(self, channel_names, moist, B, H, W):
        """A full normalized model-output field whose (T, Z[, q]) satisfy balance exactly
        before normalization is undone again by the constraint (used for fixed-point tests)."""
        t_idx, z_idx, q_idx, pressures = self._split(channel_names, moist)
        n = len(pressures)
        T = 200.0 + 60.0 * torch.rand(B, n, H, W, device=self.device)
        if moist:
            q = 0.02 * torch.rand(B, n, H, W, device=self.device)
            Tv = T * (1.0 + const.Q_CORRECTION_MOIST_AIR * q)
        else:
            Tv = T
        Z = torch.zeros(B, n, H, W, device=self.device)
        Z[:, 0, ...] = 1000.0
        for i in range(1, n):
            plog = float(np.log(pressures[i - 1] / pressures[i]))
            Z[:, i, ...] = Z[:, i - 1, ...] + const.R_DRY_AIR * 0.5 * (Tv[:, i, ...] + Tv[:, i - 1, ...]) * plog

        C = len(channel_names)
        field = torch.randn(B, C, H, W, device=self.device)  # aux channels arbitrary
        field[:, t_idx, ...] = T
        field[:, z_idx, ...] = Z
        if moist:
            field[:, q_idx, ...] = q
        return field

    # --- strict tests at strength = 1 ---
    def test_projection_enforces_balance_strict(self):
        """strength=1: an arbitrary (unbalanced) input is projected to satisfy balance
        to fp32 precision, for dry/moist and identity/realistic normalization."""
        B, H, W = 2, 5, 6
        for moist in [False, True]:
            for identity in [True, False]:
                with self.subTest(moist=moist, identity=identity):
                    names = self._channels(moist)
                    bias, scale = self._stats(names, identity)
                    proj = HydrostaticBalanceProjection(names, bias=bias, scale=scale,
                                                        p_min=self.P_MIN, p_max=self.P_MAX,
                                                        strength=1.0,
                                                        use_moist_air_formula=moist).to(self.device)
                    x = torch.randn(B, len(names), H, W, device=self.device)
                    y = proj(x)
                    res, denom = self._residual(y, names, moist, bias, scale)
                    rel = (res.abs() / denom.clamp(min=1e-6)).max().item()
                    self.assertLess(rel, 1e-4, f"residual not eliminated (rel={rel})")

    def test_balanced_input_is_fixed_point(self):
        """strength=1: a balanced input is returned essentially unchanged."""
        B, H, W = 2, 4, 5
        for moist in [False, True]:
            for identity in [True, False]:
                with self.subTest(moist=moist, identity=identity):
                    names = self._channels(moist)
                    bias, scale = self._stats(names, identity)
                    field = self._balanced_field(names, moist, B, H, W)
                    x = field if identity else (field - bias) / scale
                    proj = HydrostaticBalanceProjection(names, bias=bias, scale=scale,
                                                        p_min=self.P_MIN, p_max=self.P_MAX,
                                                        strength=1.0,
                                                        use_moist_air_formula=moist).to(self.device)
                    y = proj(x)
                    self.assertTrue(compare_tensors("balanced fixed point", y, x,
                                                    atol=1e-2, rtol=1e-4, verbose=True))

    def test_passthrough_channels_untouched(self):
        """strength=1: channels outside (T, Z) -- aux, and humidity in the moist case --
        are passed through bit-for-bit."""
        B, H, W = 2, 4, 4
        for moist in [False, True]:
            with self.subTest(moist=moist):
                names = self._channels(moist)
                t_idx, z_idx, q_idx, _ = self._split(names, moist)
                touched = set(t_idx) | set(z_idx)
                aux_idx = [i for i in range(len(names)) if i not in touched]
                proj = HydrostaticBalanceProjection(names, bias=None, scale=None,
                                                    p_min=self.P_MIN, p_max=self.P_MAX,
                                                    strength=1.0,
                                                    use_moist_air_formula=moist).to(self.device)
                x = torch.randn(B, len(names), H, W, device=self.device)
                y = proj(x)
                # all non-(T,Z) channels, which includes humidity, are identical
                self.assertTrue(compare_tensors("aux/humidity passthrough",
                                                y[:, aux_idx, ...], x[:, aux_idx, ...]))
                if moist:
                    self.assertTrue(compare_tensors("humidity held fixed",
                                                    y[:, q_idx, ...], x[:, q_idx, ...]))

    # --- soft tests: strength scales the correction linearly ---
    def test_strength_scales_correction(self):
        """The correction (x - out) is linear in strength, and the leftover balance
        residual scales as (1 - strength). strength=0 is the identity."""
        B, H, W = 2, 5, 6
        for moist in [False, True]:
            with self.subTest(moist=moist):
                names = self._channels(moist)
                bias, scale = self._stats(names, identity=False)
                proj = HydrostaticBalanceProjection(names, bias=bias, scale=scale,
                                                    p_min=self.P_MIN, p_max=self.P_MAX,
                                                    strength=1.0,
                                                    use_moist_air_formula=moist).to(self.device)
                x = torch.randn(B, len(names), H, W, device=self.device)

                proj.strength = 0.0
                self.assertTrue(compare_tensors("strength=0 identity", proj(x), x, atol=1e-5, rtol=1e-5))

                proj.strength = 1.0
                y1 = proj(x)
                proj.strength = 0.5
                yh = proj(x)

                # deviation from the input is exactly half of the full correction
                self.assertTrue(compare_tensors("half correction", (x - yh), 0.5 * (x - y1),
                                                atol=1e-4, rtol=1e-3, verbose=True))

                # the residual that remains at strength 0.5 is half the input's residual
                res_in, _ = self._residual(x, names, moist, bias, scale)
                res_half, _ = self._residual(yh, names, moist, bias, scale)
                self.assertTrue(compare_tensors("residual halved", res_half, 0.5 * res_in,
                                                atol=1e-1, rtol=1e-3, verbose=True))

    # --- gradients ---
    def test_gradients_flow(self):
        """Gradients reach T and Z (and q in the moist case, which modulates the split)."""
        B, H, W = 1, 4, 4
        for moist in [False, True]:
            with self.subTest(moist=moist):
                names = self._channels(moist)
                bias, scale = self._stats(names, identity=False)
                t_idx, z_idx, q_idx, _ = self._split(names, moist)
                proj = HydrostaticBalanceProjection(names, bias=bias, scale=scale,
                                                    p_min=self.P_MIN, p_max=self.P_MAX,
                                                    strength=1.0,
                                                    use_moist_air_formula=moist).to(self.device)
                x = torch.randn(B, len(names), H, W, device=self.device, requires_grad=True)
                proj(x).pow(2).sum().backward()
                self.assertIsNotNone(x.grad)
                self.assertFalse((x.grad[:, t_idx, ...] == 0).all().item())
                self.assertFalse((x.grad[:, z_idx, ...] == 0).all().item())
                if moist:
                    self.assertFalse((x.grad[:, q_idx, ...] == 0).all().item())


if __name__ == '__main__':
    unittest.main()
