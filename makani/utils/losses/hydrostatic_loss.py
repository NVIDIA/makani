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

from typing import Tuple, List, Optional

import numpy as np

import torch
import torch.nn as nn

from makani.utils.losses.base_loss import GeometricBaseLoss
import makani.utils.constants as const


class HydrostaticBalanceLoss(GeometricBaseLoss):
    """Computes a loss term constraining relationship between
    pressure, geopotential and temperature based on  hydrostatic balance"""

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        bias: torch.Tensor,
        scale: torch.Tensor,
        p_min: Optional[int] = 0,
        p_max: Optional[int] = 1000,
        pole_mask: Optional[int] = 0,
        use_moist_air_formula: Optional[bool] = False,
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):

        super().__init__(img_shape, crop_shape, crop_offset, channel_names, grid_type, pole_mask, spatial_distributed)

        # store some variables
        self.use_moist_air_formula = use_moist_air_formula
        self.spatial_distributed = spatial_distributed

        # get matching pl between z and t
        from makani.utils.constraints import get_matching_channels_pl

        self.z_idx, self.t_idx, self.pressures = get_matching_channels_pl(channel_names, "z", "t", p_min, p_max)
        if self.use_moist_air_formula:
            self.q_idx, _, p_tmp = get_matching_channels_pl(channel_names, "q", "t", p_min, p_max)

        if len(self.pressures) < 2:
            raise ValueError("Warning, we could not find at least two pressure levels which are common among z and t and inside the specified limits")

        if self.use_moist_air_formula:
            for p1, p2 in zip(self.pressures, p_tmp):
                if p1 != p2:
                    raise ValueError("Error, make sure that you have the same pressure levels for t,z and q channels")

        # for the cmat method store everything
        self.register_buffer("bias", bias, persistent=False)
        self.register_buffer("scale", scale, persistent=False)

        # compute prefact
        # prefactor, units [K * kg / J] = [K * kg * s^2 / (m^2 * kg)] = [K * s^2 / m^2]
        self.prefact = 1.0 / const.R_DRY_AIR

        if self.use_moist_air_formula:
            # the factor 1000. arises from converting q-units from kg / kg to g / kg
            self.q_prefact = const.Q_CORRECTION_MOIST_AIR

        # we need to interpolate in p:
        ptens = torch.as_tensor(self.pressures, dtype=torch.float32)
        ptens = ptens.reshape(1, -1, 1, 1)
        self.register_buffer("p", ptens, persistent=False)

        # create conserved quantity matrix
        row_indices = []
        col_indices = []
        values = []
        # every interval in its own equation
        for idx in range(0, len(self.t_idx) - 1):
            # z_idx
            row_indices.append(idx)
            col_indices.append(self.z_idx[idx])
            values.append(-self.prefact)
            # z_idx+1
            row_indices.append(idx)
            col_indices.append(self.z_idx[idx + 1])
            values.append(self.prefact)
            # t_idx:
            row_indices.append(idx)
            col_indices.append(self.t_idx[idx])
            values.append(0.5 * torch.log(ptens[0, idx + 1, 0, 0] / ptens[0, idx, 0, 0]))
            # t_idx+1:
            row_indices.append(idx)
            col_indices.append(self.t_idx[idx + 1])
            values.append(0.5 * torch.log(ptens[0, idx + 1, 0, 0] / ptens[0, idx, 0, 0]))

        # get sparse tensor
        indices = torch.as_tensor([row_indices, col_indices], dtype=torch.long)
        values = torch.as_tensor(values, dtype=torch.float32)
        cmat = torch.sparse_coo_tensor(indices, values, size=(len(self.t_idx) - 1, len(channel_names))).coalesce().to_dense()

        # register buffer
        self.register_buffer("cmat", cmat, persistent=False)

    @property
    def n_channels(self):
        return self.cmat.shape[0]

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str) -> torch.Tensor:
        """
        auxiliary routine for predetermining channel weighting
        """

        # initialize empty tensor
        channel_weights = torch.ones(self.n_channels, dtype=torch.float32)

        if not channel_weight_type == "constant":
            raise NotImplementedError(f"Error, channel_weight_type {channel_weight_type} not supported for hydrostatic loss")

        # normalize
        channel_weights = channel_weights / torch.sum(channel_weights)

        return channel_weights

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:

        # undo normalization
        prdun = prd * self.scale + self.bias

        if self.use_moist_air_formula:
            prdun[:, self.t_idx, ...] = prdun[:, self.t_idx, ...] * (1.0 + self.q_prefact * prdun[:, self.q_idx, ...])

        # use sparse matmul
        prdf = prdun.permute([1, 0, 2, 3])
        C, B, H, W = prdf.shape
        prdf = prdf.reshape(C, B * H * W)

        # we need to disable autocast here
        res = torch.square(torch.matmul(self.cmat, prdf).reshape(-1, B, H, W).permute([1, 0, 2, 3])).contiguous()

        if wgt is not None:
            res = res * wgt

        loss = self.quadrature(res)

        return loss
