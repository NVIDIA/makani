# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import makani.utils.constants as const
from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature


class HydrostaticBalanceWrapper(nn.Module):
    def __init__(self, channel_names, bias, scale, p_min=50, p_max=900, use_moist_air_formula=False):
        super().__init__()

        self.use_moist_air_formula = use_moist_air_formula

        # we need the matching routine:
        from makani.utils.constraints import get_matching_channels_pl

        # set up indices
        self.z_idx, self.t_idx, self.pressures = get_matching_channels_pl(channel_names, "z", "t", p_min, p_max)

        if self.use_moist_air_formula:
            self.q_idx, _, p_tmp = get_matching_channels_pl(channel_names, "q", "t", p_min, p_max)

        # sanity checks
        if len(self.pressures) == 0:
            raise ValueError("Error, make sure that you have overlapping pressure levels for geopotential and temperature")

        if self.use_moist_air_formula:
            for p1, p2 in zip(self.pressures, p_tmp):
                if p1 != p2:
                    raise ValueError("Error, make sure that you have the same pressure levels for t,z and q channels")

        # create a mapping for all channels except for z and t:
        self.aux_idx = [i for i in range(len(channel_names)) if (i not in self.t_idx + self.z_idx)]
        if self.use_moist_air_formula:
            self.aux_idx = [i for i in self.aux_idx if (i not in self.q_idx)]

        # compute prefact
        # prefactor, units [K * kg / J] = [K * kg * s^2 / (m^2 * kg)] = [K * s^2 / m^2]
        self.prefact = 1.0 / const.R_DRY_AIR

        if self.use_moist_air_formula:
            self.q_prefact = const.Q_CORRECTION_MOIST_AIR

        # assume that the first len(self.z_idx + 1) channels of the model output tensor are the geopotentials
        # and the reference temperature tensor (first component). Assume then that the remaining components correspond to the other output
        # channel.
        # create a matrix which generates a mapping from the original output to the actual output:
        # create conserved quantity matrix
        row_indices = []
        col_indices = []
        values = []

        # extract bias and scale
        if bias is not None:
            z_bias = bias[:, self.z_idx, ...]
            t_bias = bias[:, self.t_idx, ...]
            if self.use_moist_air_formula:
                q_bias = bias[:, self.q_idx, ...]
        else:
            z_bias = torch.zeros([1, len(self.z_idx), 1, 1], dtype=torch.float32)
            t_bias = torch.zeros([1, len(self.t_idx), 1, 1], dtype=torch.float32)
            if self.use_moist_air_formula:
                q_bias = torch.zeros([1, len(self.q_idx), 1, 1], dtype=torch.float32)

        if scale is not None:
            z_scale = scale[:, self.z_idx, ...]
            t_scale = scale[:, self.t_idx, ...]
            if self.use_moist_air_formula:
                q_scale = scale[:, self.q_idx, ...]
        else:
            z_scale = torch.ones([1, len(self.z_idx), 1, 1], dtype=torch.float32)
            t_scale = torch.ones([1, len(self.t_idx), 1, 1], dtype=torch.float32)
            if self.use_moist_air_formula:
                q_scale = torch.ones([1, len(self.q_idx), 1, 1], dtype=torch.float32)

        # initial t:
        row_indices.append(self.t_idx[0])
        col_indices.append(0)
        values.append(1.0)

        # do the z_indices first:
        for idx in range(1, len(self.z_idx) + 1):
            # z_idx
            row_indices.append(self.z_idx[idx - 1])
            col_indices.append(idx)
            values.append(1.0)

        # (Z_i - Z_{i-1}) / R = 0.5 * log(p_{i-1}/p_i) * (T_i + T_{i-1})
        # -> T_i = 2 * (Z_i - Z_{i-1}) / (R * log(p_{i-1} / p_i)) - T_{i-1}
        # re-insert T{i-1} recursively
        for oidx in range(1, len(self.z_idx)):
            for iidx in range(0, oidx):
                # sign
                sign = (-1.0) ** (oidx + iidx - 1)

                # t_idx
                plog = np.log(self.pressures[iidx] / self.pressures[iidx + 1])

                # 2. * Z_i / (R * log(p_{i-1} / p_i))
                row_indices.append(self.t_idx[oidx])
                col_indices.append(iidx + 2)
                values.append(sign * 2.0 * self.prefact / plog)

                # - 2. * Z_{i-1} / (R * log(p_{i-1} / p_i))
                row_indices.append(self.t_idx[oidx])
                col_indices.append(iidx + 1)
                values.append(-sign * 2.0 * self.prefact / plog)

            # - T_0
            row_indices.append(self.t_idx[oidx])
            col_indices.append(0)
            values.append((-1.0) ** oidx)

        # q-values when moist air is used:
        off_idx = len(self.z_idx) + 1
        if self.use_moist_air_formula:
            for idx in range(off_idx, off_idx + len(self.q_idx)):
                row_indices.append(self.q_idx[idx - off_idx])
                col_indices.append(idx)
                values.append(1.0)
            # shift the offset accordingly
            off_idx += len(self.q_idx)

        # now deal with the remaining channels:
        for idx in range(off_idx, off_idx + len(self.aux_idx)):
            row_indices.append(self.aux_idx[idx - off_idx])
            col_indices.append(idx)
            values.append(1.0)

        # create matrix:
        ncols = len(channel_names) - len(self.t_idx) + 1
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
        values = torch.tensor(values, dtype=torch.float32)
        mapping = torch.sparse_coo_tensor(indices, values, size=(len(channel_names), ncols)).coalesce().to_dense()

        # register
        self.register_buffer("mapping", mapping, persistent=False)

        # bias and scale terms:
        # input
        # scale
        inp_scale = torch.ones((1, mapping.shape[1], 1, 1), dtype=torch.float32)
        inp_scale[0, 0, 0, 0] = t_scale[0, 0, 0, 0]
        inp_scale[0, 1 : len(self.z_idx) + 1, 0, 0] = z_scale[0, :, 0, 0]
        # bias
        inp_bias = torch.zeros((1, mapping.shape[1], 1, 1), dtype=torch.float32)
        inp_bias[0, 0, 0, 0] = t_bias[0, 0, 0, 0]
        inp_bias[0, 1 : len(self.z_idx) + 1, 0, 0] = z_bias[0, :, 0, 0]
        # moist air
        if self.use_moist_air_formula:
            # assume the next entries are relative humidities:
            inp_bias[0, len(self.z_idx) + 1 : len(self.z_idx) + len(self.q_idx) + 1, 0, 0] = q_bias[0, :, 0, 0]
            inp_scale[0, len(self.z_idx) + 1 : len(self.z_idx) + len(self.q_idx) + 1, 0, 0] = q_scale[0, :, 0, 0]
        self.register_buffer("inp_scale", inp_scale, persistent=True)
        self.register_buffer("inp_bias", inp_bias, persistent=True)

        # output
        # scale
        out_scale = torch.ones((1, len(channel_names), 1, 1), dtype=torch.float32)
        out_scale[:, self.t_idx, ...] = t_scale[...]
        out_scale[:, self.z_idx, ...] = z_scale[...]
        # bias
        out_bias = torch.zeros((1, len(channel_names), 1, 1), dtype=torch.float32)
        out_bias[:, self.t_idx, ...] = t_bias[...]
        out_bias[:, self.z_idx, ...] = z_bias[...]
        # moist air
        if self.use_moist_air_formula:
            out_bias[:, self.q_idx, ...] = q_bias[...]
            out_scale[:, self.q_idx, ...] = q_scale[...]
        self.register_buffer("out_scale", out_scale, persistent=True)
        self.register_buffer("out_bias", out_bias, persistent=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:

        # if input tensor is 5D, fold first two dimensions into one:
        # T can be ensemble or time dim, but it is expected to be a time dim for other constraints
        if inp.ndim == 5:
            B, T, C, H, W = inp.shape
            inp = inp.reshape(B*T, C, H, W)
            reshape = True
        else:
            B, C, H, W = inp.shape
            reshape = False

        # undo normalization
        inp_un = inp * self.inp_scale + self.inp_bias

        # convert temperature to virtual temperature
        if self.use_moist_air_formula:
            inp_un_scale = inp_un.clone()
            inp_un_scale[:, 0, ...] = inp_un[:, 0, ...] * (1.0 + self.q_prefact * inp_un[:, len(self.z_idx) + 1, ...])
        else:
            inp_un_scale = inp_un

        # expand:
        out_un = F.conv2d(inp_un_scale, self.mapping.unsqueeze(-1).unsqueeze(-1))

        # unscale temperatures if specific humidity is used
        if self.use_moist_air_formula:
            out_un[:, self.t_idx, ...] = out_un[:, self.t_idx, ...] / (1.0 + self.q_prefact * out_un[:, self.q_idx, ...])

        # undo normalization
        out = (out_un - self.out_bias) / self.out_scale

        # reshape if necessary
        if reshape:
            out = out.reshape(B, T, -1, H, W)

        return out


class TotalWaterPath(nn.Module):
    def __init__(self, channel_names, bias, scale):
        super().__init__()

        # we need the matching routine:
        from makani.utils.constraints import get_channels_pl

        # get q-channels
        self.q_idx, p_tmp = get_channels_pl(channel_names, "q", 0, np.inf, revert=False)
        pressures = torch.as_tensor(p_tmp, dtype=torch.float32).reshape(1, -1, 1, 1)
        # we need to convert pressures from hPa to Pa:
        pressures = pressures * 100.0
        self.register_buffer("pressures", pressures, persistent=False)

        # get surface pressure channel
        self.sp_idx = channel_names.index("sp")

        if bias is not None:
            q_bias = bias[:, self.q_idx, ...]
            sp_bias = bias[:, self.sp_idx, ...].unsqueeze(1)
        else:
            q_bias = torch.zeros([1, len(self.q_idx), 1, 1], dtype=torch.float32)
            sp_bias = torch.zeros([1, 1, 1, 1], dtype=torch.float32)

        if scale is not None:
            q_scale = scale[:, self.q_idx, ...]
            sp_scale = scale[:, self.sp_idx, ...].unsqueeze(1)
        else:
            q_scale = torch.ones([1, len(self.q_idx), 1, 1], dtype=torch.float32)
            sp_scale = torch.ones([1, len(self.sp_idx), 1, 1], dtype=torch.float32)

        self.register_buffer("q_scale", q_scale, persistent=False)
        self.register_buffer("q_bias", q_bias, persistent=True)
        self.register_buffer("sp_scale", sp_scale, persistent=True)
        self.register_buffer("sp_bias", sp_bias, persistent=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Compute total water path from input tensor: TWP = ∫ q dp / g
        Args:
            inp: Input tensor of shape (B, C, H, W)
        Returns:
            twp: Total water path tensor of shape (B, H, W)
        """
        qvals =  inp[:, self.q_idx, ...]
        spvals = inp[:, self.sp_idx, ...].unsqueeze(1)

        # convert to physical units:
        qvals = qvals * self.q_scale + self.q_bias
        spvals = spvals * self.sp_scale + self.sp_bias

        # concatenate with pressures:
        with torch.no_grad():
            pressures_expand = torch.tile(self.pressures, (spvals.shape[0], 1, spvals.shape[-2], spvals.shape[-1]))

        # now we need to sort according to sp:
        pressures_sorted, _ = torch.sort(torch.cat([pressures_expand, spvals], dim=1), dim=1, stable=True)
        pdiff_sorted = torch.diff(pressures_sorted, dim=1)
        # now we mask out the q-values that are above the surface pressure
        qvals_sorted = torch.where(pressures_expand <= spvals, qvals, 0.0)

        # compute total water path integral:
        # humidity is in g/kg and pressure in hPa. so the units are 
        # Pa / (m / s^2) = kg / (m s^2) / (m / s^2) = kg / m^2
        twp = torch.sum(qvals_sorted * pdiff_sorted, dim=1) / const.GRAVITATIONAL_ACCELERATION

        return twp


class DryAirSurfacePressure(nn.Module):
    def __init__(self, channel_names, bias, scale):
        super().__init__()
        self.twp = TotalWaterPath(channel_names, bias, scale)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Computes dry air pressure defined as:
        sp_dry = sp - g * TWP
        Args:
            inp: Input tensor of shape (B, C, H, W), containing q and sp channels
        Returns:
            out: Total water path tensor of shape (B, H, W) with corrected sp channel
        """
        gtwp = self.twp(inp).unsqueeze(1) * const.GRAVITATIONAL_ACCELERATION

        # normalize the correction using sp normalizations
        gtwp_normalized = (gtwp - self.twp.sp_bias) / self.twp.sp_scale
            
        out = inp[:, self.sp_idx, ...] - gtwp_normalized[:, 0, ...]

        return out.unsqueeze(1)


class SurfacePressureBalanceWrapper(nn.Module):
    def __init__(self, img_shape, grid_type, channel_names, bias, scale, distributed=False):
        super().__init__()

        # we need the dry air calculator
        self.dasp = DryAirSurfacePressure(channel_names, bias, scale)

        # we need the quadrature rule
        self.quadrature = GridQuadrature(
            quadrature_rule=grid_to_quadrature_rule(grid_type), 
            img_shape=img_shape, 
            normalize=True,
            distributed=distributed
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:

        if inp.ndim != 5:
            raise ValueError("Input tensor must be 5D with dimensions B, T, C, H, W")

        B, T, C, H, W = inp.shape

        # compute dry air pressure
        sp_dry = self.dasp(inp.reshape(B*T, C, H, W)).reshape(B, T, 1, H, W)
        # compute differences: we DO NOT negate the difference to compute P(t-delta t) - P(t)
        # since P(t-delta t) vomes before P(t) in the buffer
        dsp_dry = torch.diff(sp_dry, dim=1).reshape(B*(T-1), 1, H, W).contiguous()

        # compute the balance:
        balance = self.quadrature(dsp_dry).reshape(B, T-1, 1, H, W)

        # compute balanced surface pressure:
        # note we add sp_balance since we need to subtract <P(t) - P(t-delta t)> from P(t)
        # but we computed the one with the opposite sign.
        # we can only return the last T-1 elements since this is computed with a derivative
        balanced_sp = inp[:, 1:, self.sp_idx, ...] + balance

        return balanced_sp.unsqueeze(2)


# constraints is a list of dicts with variables, e.g.:
# constraints = [{type: hydrostatic_balance,
#                 options: {specific options}]
class ConstraintsWrapper(nn.Module):

    def __init__(self, constraints, channel_names, bias, scale, model_handle=None):
        super().__init__()

        if model_handle is not None:
            self.model = model_handle()
        else:
            self.model = None

        # sanity checks
        self.constraint_list = nn.ModuleList()
        for constraint in constraints:
            if constraint["type"] == "hydrostatic_balance":
                self.constraint_list.append(HydrostaticBalanceWrapper(**constraint["options"], channel_names=channel_names, bias=bias, scale=scale))
                self.N_in_channels = self.constraint_list[-1].mapping.shape[1]
            else:
                raise NotImplementedError(f"Error, constraints different from hydrostatic balance are not yet implemented.")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            # get model output
            inp = self.model(inp)

        # only support single constraint atm, we deal with stacking later
        out = self.constraint_list[0](inp)

        return out
