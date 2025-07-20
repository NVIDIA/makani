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

from makani.utils.losses.base_loss import GeometricBaseLoss, SpectralBaseLoss

from makani.utils import comm


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


class SpectralL2Loss(SpectralBaseLoss):
    """
    Computes the geometric L2 loss but using the spherical Harmonic transform
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
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

        self.relative = relative
        self.squared = squared
        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed

    def abs(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        B, C, H, W = prd.shape

        coeffssq = torch.square(torch.abs(self.sht(prd - tar))) / torch.pi / 4.0

        if wgt is not None:
            coeffssq = coeffssq * wgt

        if comm.get_rank("w") == 0:
            norm2 = coeffssq[..., 0] + 2 * torch.sum(coeffssq[..., 1:], dim=-1)
        else:
            norm2 = 2 * torch.sum(coeffssq, dim=-1)
        if self.spatial_distributed and (comm.get_size("w") > 1):
            norm2 = reduce_from_parallel_region(norm2, "w")

        # compute norms
        norm2 = norm2.reshape(B, C, -1)
        norm2 = torch.sum(norm2, dim=-1)

        if self.spatial_distributed and (comm.get_size("h") > 1):
            norm2 = reduce_from_parallel_region(norm2, "h")

        if not self.squared:
            norm2 = torch.sqrt(norm2)

        return norm2

    def rel(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        B, C, H, W = prd.shape

        coeffssq = torch.square(torch.abs(self.sht(prd - tar))) / torch.pi / 4.0

        if wgt is not None:
            coeffssq = coeffssq * wgt

        # sum m != 0 coeffs:
        if comm.get_rank("w") == 0:
            norm2 = coeffssq[..., 0] + 2 * torch.sum(coeffssq[..., 1:], dim=-1)
        else:
            norm2 = 2 * torch.sum(coeffssq, dim=-1)
        if self.spatial_distributed and (comm.get_size("w") > 1):
            norm2 = reduce_from_parallel_region(norm2, "w")

        # compute norms
        norm2 = norm2.reshape(B, C, -1)
        norm2 = torch.sum(norm2, dim=-1)
        if self.spatial_distributed and (comm.get_size("h") > 1):
            norm2 = reduce_from_parallel_region(norm2, "h")

        # target
        tar_coeffssq = torch.square(torch.abs(self.sht(tar))) / torch.pi / 4.0

        if wgt is not None:
            tar_coeffssq = tar_coeffssq * wgt

        # sum m != 0 coeffs:
        if comm.get_rank("w") == 0:
            tar_norm2 = tar_coeffssq[..., 0] + 2 * torch.sum(tar_coeffssq[..., 1:], dim=-1)
        else:
            tar_norm2 = 2 * torch.sum(tar_coeffssq, dim=-1)
        if self.spatial_distributed and (comm.get_size("w") > 1):
            tar_norm2 = reduce_from_parallel_region(tar_norm2, "w")

        # compute target norms
        tar_norm2 = tar_norm2.reshape(B, C, -1)
        tar_norm2 = torch.sum(tar_norm2, dim=-1)
        if self.spatial_distributed and (comm.get_size("h") > 1):
            tar_norm2 = reduce_from_parallel_region(tar_norm2, "h")

        if not self.squared:
            diff_norms = torch.sqrt(norm2)
            tar_norms = torch.sqrt(tar_norm2)
        else:
            diff_norms = norm2
            tar_norms = tar_norm2

        # setup return value
        retval = diff_norms / tar_norms

        return retval

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.relative:
            loss = self.rel(prd, tar, wgt)
        else:
            loss = self.abs(prd, tar, wgt)

        return loss
