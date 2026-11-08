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

from makani.utils.losses.base_loss import GeometricBaseLoss, SpectralBaseLoss


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
        jacobian: Optional[str] = "s2",
        grid_type: Optional[str] = "equiangular",
        spatial_distributed: Optional[bool] = False,
        eps: Optional[float] = 1.0e-6,
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
        self.eps = eps

    def abs(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        num_examples = prd.shape[0]

        diff = torch.abs(prd - tar).pow(self.p)

        if wgt is not None:
            diff = diff * wgt

        all_norms = self.quadrature(diff)
        all_norms = all_norms.reshape(num_examples, -1)

        if not self.squared:
            all_norms = all_norms.pow(1.0 / self.p)

        return all_norms

    def rel(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        num_examples = prd.shape[0]

        # numerator
        diff = torch.abs(prd - tar).pow(self.p)

        if wgt is not None:
            diff = diff * wgt

        diff_norms = self.quadrature(diff)
        diff_norms = diff_norms.reshape(num_examples, -1)

        # denominator
        tarr = torch.abs(tar).pow(self.p)

        if wgt is not None:
            tarr = tarr * wgt

        tar_norms = self.quadrature(tarr)
        tar_norms = tar_norms.reshape(num_examples, -1)

        # divide the ratios (eps-guard: avoids NaN on zero-power targets)
        all_norms = diff_norms / (tar_norms + self.eps)

        if not self.squared:
            all_norms = all_norms.pow(1.0 / self.p)

        return all_norms

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs):
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
        eps: Optional[float] = 1.0e-6,
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
        self.eps = eps

    def _normp(self, field: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        # |coeffs|^p (orthonormal convention), summed over m and l
        coeffsp = torch.abs(self.sht(field)).pow(self.p)
        if wgt is not None:
            coeffsp = coeffsp * wgt
        return self.sum_over_l(self.sum_over_m(coeffsp))

    def abs(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        normp = self._normp(prd - tar, wgt)

        # take p-th root unless squared is True
        if not self.squared:
            normp = normp.pow(1.0 / self.p)

        return normp

    def rel(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        normp = self._normp(prd - tar, wgt)
        tar_normp = self._normp(tar, wgt)

        # take p-th root unless squared is True
        if not self.squared:
            diff_norms = normp.pow(1.0 / self.p)
            tar_norms = tar_normp.pow(1.0 / self.p)
        else:
            diff_norms = normp
            tar_norms = tar_normp

        # compute relative error (eps-guard: avoids NaN on zero-power targets)
        retval = diff_norms / (tar_norms + self.eps)

        return retval

    def forward(
        self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:

        if self.relative:
            loss = self.rel(prd, tar, wgt)
        else:
            loss = self.abs(prd, tar, wgt)

        return loss
