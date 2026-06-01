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

from makani.utils.losses.hydrostatic_loss import HydrostaticBalanceLoss
from makani.models.parametrizations import ConstraintsWrapper

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


if __name__ == '__main__':
    unittest.main()
