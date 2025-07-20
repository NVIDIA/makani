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
import math

import torch
import torch.nn as nn
from torch import amp

from makani.utils.losses.base_loss import SpectralBaseLoss

# distributed stuff
from makani.utils import comm
from physicsnemo.distributed.utils import split_tensor_along_dim
from physicsnemo.distributed.mappings import reduce_from_parallel_region


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

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:

        # compute the sht
        prd = prd.float()
        tar = tar.float()
        with amp.autocast(device_type="cuda", enabled=False):
            xcoeffs = self.sht(prd)
            ycoeffs = self.sht(tar)
        
        # compute the SHT:
        xcoeffssq = torch.square(torch.abs(xcoeffs))
        ycoeffssq = torch.square(torch.abs(ycoeffs))
        xycohcoeffssq = torch.real(xcoeffs * ycoeffs.conj())

        if wgt is not None:
            xcoeffssq = xcoeffssq * wgt
            ycoeffssq = ycoeffssq * wgt
            xycohcoeffssq = xycohcoeffssq * wgt

        # reduce over m
        if comm.get_rank("w") == 0:
            xnorm2 = xcoeffssq[..., 0] + 2 * torch.sum(xcoeffssq[..., 1:], dim=-1)
            ynorm2 = ycoeffssq[..., 0] + 2 * torch.sum(ycoeffssq[..., 1:], dim=-1)
            xycoh = xycohcoeffssq[..., 0] + 2 * torch.sum(xycohcoeffssq[..., 1:], dim=-1)
        else:
            xnorm2 = 2 * torch.sum(xcoeffssq, dim=-1)
            ynorm2 = 2 * torch.sum(ycoeffssq, dim=-1)
            xycoh = 2 * torch.sum(xycohcoeffssq, dim=-1)

        # distributed reduction
        if self.spatial_distributed and (comm.get_size("w") > 1):
            xnorm2 = reduce_from_parallel_region(xnorm2, "w")
            ynorm2 = reduce_from_parallel_region(ynorm2, "w")
            xycoh = reduce_from_parallel_region(xycoh, "w")

        # compute sqrt
        xnorm = torch.sqrt(xnorm2)
        ynorm = torch.sqrt(ynorm2)
        xycoh = xycoh / (xnorm * ynorm)

        # compute equation (6) from the paper
        loss = torch.square(xnorm - ynorm) + 2 * torch.maximum(xnorm2, ynorm2) * (1 - xycoh)

        # sum over l
        loss = torch.sum(loss, dim=-1)
        if self.spatial_distributed and (comm.get_size("h") > 1):
            loss = reduce_from_parallel_region(loss, "h")
        
        return loss
