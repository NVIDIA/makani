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
import math

import torch
import torch.nn as nn

from makani.utils.losses.base_loss import SpectralBaseLoss

# distributed stuff
from makani.utils import comm
from physicsnemo.distributed.utils import split_tensor_along_dim
from physicsnemo.distributed.mappings import reduce_from_parallel_region


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

        # store weights
        h1_weights = torch.arange(self.sht.lmax).float()
        h1_weights = h1_weights * (h1_weights + 1)

        # split up if distributed
        if self.spatial_distributed and (comm.get_size("h") > 1):
            h1_weights = split_tensor_along_dim(h1_weights, 0, comm.get_size("h"))

        h1_weights = h1_weights.reshape(1, 1, -1)
        self.register_buffer("h1_weights", h1_weights, persistent=False)

    def abs(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        B, C, H, W = prd.shape

        coeffssq = torch.square(torch.abs(self.sht(prd - tar)))

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
        h1_norm2 = torch.sum(norm2 * self.h1_weights, dim=-1)

        if self.spatial_distributed and (comm.get_size("h") > 1):
            h1_norm2 = reduce_from_parallel_region(h1_norm2, "h")

        if not self.squared:
            h1_norm2 = torch.sqrt(h1_norm2)

        return h1_norm2

    def rel(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        B, C, H, W = prd.shape

        coeffssq = torch.square(torch.abs(self.sht(prd - tar)))

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
        h1_norm2 = torch.sum(norm2 * self.h1_weights, dim=-1)
        if self.spatial_distributed and (comm.get_size("h") > 1):
            h1_norm2 = reduce_from_parallel_region(h1_norm2, "h")

        # target
        tar_coeffssq = torch.square(torch.abs(self.sht(tar)))

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
        tar_h1_norm2 = torch.sum(tar_norm2 * self.h1_weights, dim=-1)
        if self.spatial_distributed and (comm.get_size("h") > 1):
            tar_h1_norm2 = reduce_from_parallel_region(tar_h1_norm2, "h")

        if not self.squared:
            diff_norms = torch.sqrt(h1_norm2)
            tar_norms = torch.sqrt(tar_h1_norm2)
        else:
            diff_norms = h1_norm2
            tar_norms = tar_h1_norm2

        # setup return value
        retval = diff_norms / tar_norms

        return retval

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.relative:
            loss = self.rel(prd, tar, wgt)
        else:
            loss = self.abs(prd, tar, wgt)

        return loss
