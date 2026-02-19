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

from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from physicsnemo.distributed.mappings import reduce_from_parallel_region

from makani.utils.losses.base_loss import GeometricBaseLoss, SpectralBaseLoss

from makani.utils import comm

from physicsnemo.distributed.mappings import reduce_from_parallel_region


class GeometricLpLoss(GeometricBaseLoss):
    """
    Computes the Lp loss on the sphere.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        p: Optional[float] = 2.0,
        relative: Optional[bool] = False,
        squared: Optional[bool] = False,
        pole_mask: Optional[int] = 0,
        jacobian: Optional[str] = "s2",
        grid_type: Optional[str] = "equiangular",
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            pole_mask=pole_mask,
            spatial_distributed=spatial_distributed,
        )

        self.p = p
        self.relative = relative
        self.squared = squared
        self.spatial_distributed = spatial_distributed

    def abs(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        num_examples = prd.shape[0]

        diff = torch.abs(prd - tar) ** self.p

        if wgt is not None:
            diff = diff * wgt

        all_norms = self.quadrature(diff)
        all_norms = all_norms.reshape(num_examples, -1)

        if not self.squared:
            all_norms = all_norms ** (1.0 / self.p)

        return all_norms

    def rel(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        num_examples = prd.shape[0]

        # numerator
        diff = torch.abs(prd - tar) ** self.p

        if wgt is not None:
            diff = diff * wgt

        diff_norms = self.quadrature(diff)
        diff_norms = diff_norms.reshape(num_examples, -1)

        # denominator
        tarr = torch.abs(tar) ** self.p

        if wgt is not None:
            tarr = tarr * wgt

        tar_norms = self.quadrature(tarr)
        tar_norms = tar_norms.reshape(num_examples, -1)

        # divide the ratios
        all_norms = diff_norms / tar_norms

        if not self.squared:
            all_norms = all_norms ** (1.0 / self.p)

        return all_norms

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        if self.relative:
            loss = self.rel(prd, tar, wgt)
        else:
            loss = self.abs(prd, tar, wgt)

        return loss


class SpectralLpLoss(SpectralBaseLoss):
    """
    Computes the Lp loss in spectral (SH coefficients) space
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        p: Optional[float] = 2.0,
        relative: Optional[bool] = False,
        squared: Optional[bool] = False,
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            spatial_distributed=spatial_distributed,
        )

        self.p = p
        self.relative = relative
        self.squared = squared
        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed

    def abs(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        B, C, H, W = prd.shape

        # compute SH coefficients of the difference
        coeffs = self.sht(prd - tar)

        # compute |coeffs|^p (orthonormal convention)
        coeffsp = torch.abs(coeffs) ** self.p

        if wgt is not None:
            coeffsp = coeffsp * wgt

        # sum over m: m=0 contributes once, m!=0 contribute twice (due to conjugate symmetry)
        if comm.get_rank("w") == 0:
            normp = coeffsp[..., 0] + 2 * torch.sum(coeffsp[..., 1:], dim=-1)
        else:
            normp = 2 * torch.sum(coeffsp, dim=-1)

        if self.spatial_distributed and (comm.get_size("w") > 1):
            normp = reduce_from_parallel_region(normp, "w")

        # sum over l (degrees)
        normp = normp.reshape(B, C, -1)
        normp = torch.sum(normp, dim=-1)

        if self.spatial_distributed and (comm.get_size("h") > 1):
            normp = reduce_from_parallel_region(normp, "h")

        # take p-th root unless squared is True
        if not self.squared:
            normp = normp ** (1.0 / self.p)

        return normp

    def rel(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        B, C, H, W = prd.shape

        # compute SH coefficients of the difference
        coeffs = self.sht(prd - tar)
        coeffsp = torch.abs(coeffs) ** self.p

        if wgt is not None:
            coeffsp = coeffsp * wgt

        # sum m != 0 coeffs for numerator
        if comm.get_rank("w") == 0:
            normp = coeffsp[..., 0] + 2 * torch.sum(coeffsp[..., 1:], dim=-1)
        else:
            normp = 2 * torch.sum(coeffsp, dim=-1)

        if self.spatial_distributed and (comm.get_size("w") > 1):
            normp = reduce_from_parallel_region(normp, "w")

        # sum over l
        normp = normp.reshape(B, C, -1)
        normp = torch.sum(normp, dim=-1)

        if self.spatial_distributed and (comm.get_size("h") > 1):
            normp = reduce_from_parallel_region(normp, "h")

        # compute target norm
        tar_coeffs = self.sht(tar)
        tar_coeffsp = torch.abs(tar_coeffs) ** self.p

        if wgt is not None:
            tar_coeffsp = tar_coeffsp * wgt

        # sum m != 0 coeffs for denominator
        if comm.get_rank("w") == 0:
            tar_normp = tar_coeffsp[..., 0] + 2 * torch.sum(tar_coeffsp[..., 1:], dim=-1)
        else:
            tar_normp = 2 * torch.sum(tar_coeffsp, dim=-1)

        if self.spatial_distributed and (comm.get_size("w") > 1):
            tar_normp = reduce_from_parallel_region(tar_normp, "w")

        # sum over l
        tar_normp = tar_normp.reshape(B, C, -1)
        tar_normp = torch.sum(tar_normp, dim=-1)

        if self.spatial_distributed and (comm.get_size("h") > 1):
            tar_normp = reduce_from_parallel_region(tar_normp, "h")

        # take p-th root unless squared is True
        if not self.squared:
            diff_norms = normp ** (1.0 / self.p)
            tar_norms = tar_normp ** (1.0 / self.p)
        else:
            diff_norms = normp
            tar_norms = tar_normp

        # compute relative error
        retval = diff_norms / tar_norms

        return retval

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.relative:
            loss = self.rel(prd, tar, wgt)
        else:
            loss = self.abs(prd, tar, wgt)

        return loss

