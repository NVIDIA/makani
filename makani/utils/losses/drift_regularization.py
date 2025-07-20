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

from makani.utils.losses.base_loss import GeometricBaseLoss, LossType

from makani.utils import comm
from physicsnemo.distributed.mappings import reduce_from_parallel_region


class DriftRegularization(GeometricBaseLoss):
    """
    Computes the Lp loss on the sphere.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        p: Optional[float] = 1.0,
        pole_mask: Optional[int] = 0,
        grid_type: Optional[str] = "equiangular",
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
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
        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.ensemble_distributed = ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):

        if prd.dim() > tar.dim():
            tar = tar.unsqueeze(1)

        # compute difference between the means output has dims
        loss = torch.abs(self.quadrature(prd) -  self.quadrature(tar)).pow(self.p)

        # if ensemble
        if prd.dim() == 5:
            loss = torch.mean(loss, dim=1)
            if self.ensemble_distributed:
                loss = reduce_from_parallel_region(loss, "ensemble") / float(comm.get_size("ensemble"))

        return loss
