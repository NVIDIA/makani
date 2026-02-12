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
import unittest

import numpy as np

import torch

from makani.utils.losses.hydrostatic_loss import HydrostaticBalanceLoss
from makani.models.parametrizations import ConstraintsWrapper, TotalWaterPathWrapper
import makani.utils.constants as const

class TestConstraints(unittest.TestCase):

    def setUp(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        # load the data:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        data = np.load(os.path.join(data_dir, "sample_30km_equator.npz"))

        # fields
        self.data = torch.as_tensor(data["data"].astype(np.float32))
        self.bias = torch.as_tensor(data["bias"].astype(np.float32))
        self.scale = torch.as_tensor(data["scale"].astype(np.float32))
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
                                                pole_mask=0,
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
                                                pole_mask=0,
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
                                                pole_mask=0,
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


    def test_total_water_path_wrapper(self):
        """Total water path from random positive q and sp on 16x32 grid; print value at center."""
        
        # data shape
        B, C, H, W = self.data.shape

        print(self.data.shape, self.bias.shape, self.scale.shape)

        # use real input data
        inp = self.data.clone().to(self.device)

        twp_wrapper = TotalWaterPathWrapper(channel_names=self.channel_names, bias=self.bias, scale=self.scale).to(self.device)
        twp = twp_wrapper(inp)

        print(twp_wrapper.pressures.shape, twp_wrapper.q_idx, twp_wrapper.sp_idx, self.channel_names)

        self.assertEqual(twp.shape, (B, H, W))
        center_twp_val = twp[0, H // 2, W // 2].item()

        # now compute the value manually:
        qvals = inp[0, twp_wrapper.q_idx, H//2, W//2].cpu().numpy()
        spval = inp[0, twp_wrapper.sp_idx, H//2, W//2].item()

        # rescale q-vals:
        qvals_rs = qvals * twp_wrapper.q_scale[0, :, 0, 0].cpu().numpy() + twp_wrapper.q_bias[0, :, 0, 0].cpu().numpy()

        # rescale SP:
        spval_rs = spval * twp_wrapper.sp_scale[0, 0, 0, 0].item() + twp_wrapper.sp_bias[0, 0, 0, 0].item()

        # concatenate pressures: everything should be in units Pa now:
        pressures = sorted(twp_wrapper.pressures[0, :, 0, 0].cpu().numpy().tolist() + [spval_rs])

        # now do a cumsum over q vals and pressure differences, stopping at spval:
        integral = 0.0
        for i in range(len(pressures)-1):

            # stop if we reached the upper integration boundary
            if pressures[i] >= spval_rs:
                break

            # perform riemann sum
            dp = pressures[i+1] - pressures[i]
            qv = qvals_rs[i]
            integral += qv * dp
        
        # normalize by gravitational acceleration
        center_twp_val_manual = integral / const.GRAVITATIONAL_ACCELERATION

        self.assertTrue(np.abs(center_twp_val - center_twp_val_manual) < 1e-6)


if __name__ == '__main__':
    unittest.main()
