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

import unittest

import numpy as np

import torch

from makani.utils.losses.hydrostatic_loss import HydrostaticBalanceLoss
from makani.models.parametrizations import ConstraintsWrapper

class TestConstraints(unittest.TestCase):

    def setUp(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        # load the data:
        data = np.load("tests/data/sample_30km_equator.npz")

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


if __name__ == '__main__':
    unittest.main()
