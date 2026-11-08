# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from torch import amp

from makani.utils.losses.base_loss import SpectralBaseLoss


# Adjusted Mean Squared Error Loss
class SpectralAMSELoss(SpectralBaseLoss):
    """
    Computes the Adjusted MSE Loss as described in arXiv:2501.19374
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
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
        self.eps = eps

    def forward(
        self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:

        # compute the sht
        ptype = prd.dtype
        with amp.autocast(device_type=prd.device.type, enabled=False):
            prd = prd.to(torch.float32)
            tar = tar.to(torch.float32)
            xcoeffs = self.sht(prd)
            ycoeffs = self.sht(tar)

        # compute the SHT:
        xcoeffssq = torch.square(torch.abs(xcoeffs))
        ycoeffssq = torch.square(torch.abs(ycoeffs))
        xycoh_prod = torch.real(xcoeffs * ycoeffs.conj())

        # convert back
        xcoeffssq = xcoeffssq.to(dtype=ptype)
        ycoeffssq = ycoeffssq.to(dtype=ptype)
        xycoh_prod = xycoh_prod.to(dtype=ptype)

        if wgt is not None:
            xcoeffssq = xcoeffssq * wgt
            ycoeffssq = ycoeffssq * wgt
            xycoh_prod = xycoh_prod * wgt

        # Parseval sums over m
        xnorm2 = self.sum_over_m(xcoeffssq)
        ynorm2 = self.sum_over_m(ycoeffssq)
        xycoh_sum = self.sum_over_m(xycoh_prod)

        # compute sqrt
        xnorm = torch.sqrt(xnorm2)
        ynorm = torch.sqrt(ynorm2)
        # eps-guard: avoids NaN at degrees where either field has zero power
        xycoh = xycoh_sum / torch.sqrt(xnorm2 * ynorm2 + self.eps)

        # compute equation (6) from the paper
        loss = torch.square(xnorm - ynorm) + 2 * torch.maximum(xnorm2, ynorm2) * (1 - xycoh)

        # sum over l
        loss = self.sum_over_l(loss)

        return loss
