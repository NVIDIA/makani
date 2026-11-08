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

from makani.utils.losses.base_loss import SpectralBaseLoss

# distributed stuff
from makani.utils import comm
from torch_harmonics.distributed import split_tensor_along_dim


# TODO: convert this to the seminorm
# double check if polar optimization has an effect - we use 5 here by defaul
class SpectralH1Loss(SpectralBaseLoss):
    """
    Computes the geometric H1 seminorm loss on the sphere. We do not compute the L2 part,
    this can be done by adding a GeometricLp loss with p=2 to the training
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

        self.relative = relative
        self.squared = squared
        self.eps = eps

        # store weights
        h1_weights = torch.arange(self.sht.lmax).float()
        h1_weights = h1_weights * (h1_weights + 1)

        # split up if distributed
        if self.spatial_distributed and (comm.get_size("h") > 1):
            h1_weights = split_tensor_along_dim(h1_weights, 0, comm.get_size("h"))[comm.get_rank("h")]

        self.register_buffer("h1_weights", h1_weights, persistent=False)

    def _h1_norm2(self, field: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        # |coeffs|^2 summed over m, weighted by l(l+1) and summed over l
        coeffssq = torch.square(torch.abs(self.sht(field)))
        if wgt is not None:
            coeffssq = coeffssq * wgt
        return self.sum_over_l(self.sum_over_m(coeffssq) * self.h1_weights)

    def abs(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        h1_norm2 = self._h1_norm2(prd - tar, wgt)

        if not self.squared:
            h1_norm2 = torch.sqrt(h1_norm2)

        return h1_norm2

    def rel(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        h1_norm2 = self._h1_norm2(prd - tar, wgt)
        tar_h1_norm2 = self._h1_norm2(tar, wgt)

        if not self.squared:
            diff_norms = torch.sqrt(h1_norm2)
            tar_norms = torch.sqrt(tar_h1_norm2)
        else:
            diff_norms = h1_norm2
            tar_norms = tar_h1_norm2

        # setup return value (eps-guard: avoids NaN on constant targets where l(l+1) weighting zeros the H¹ norm)
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
