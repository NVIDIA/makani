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

import numpy as np

import torch
import torch.nn as nn
from torch.cuda import amp

from makani.utils.losses.base_loss import GeometricBaseLoss, SpectralBaseLoss, LossType
from makani.utils import comm

# distributed stuff
from physicsnemo.distributed.utils import compute_split_shapes, split_tensor_along_dim
from physicsnemo.distributed.mappings import scatter_to_parallel_region, reduce_from_parallel_region
from makani.mpu.mappings import distributed_transpose

# @torch.compile
# def _mmd_rbf_kernel(x: torch.Tensor, y: torch.Tensor):
#     return torch.abs(x - y)


@torch.compile
def _mmd_rbf_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0):
    return torch.exp(-0.5 * torch.square(torch.abs(x - y)) / bandwidth)


# Computes the squared maximum mean discrepancy
# @torch.compile
def _mmd2_ensemble_kernel(observation: torch.Tensor, forecasts: torch.Tensor) -> torch.Tensor:

    # initial values
    spread_term = torch.zeros_like(observation)
    disc_term = torch.zeros_like(observation)

    num_forecasts = forecasts.shape[0]

    for m in range(num_forecasts):

        # get the forecast
        ym = forecasts[m]

        # account for contributions on the off-diasgonal assuming that the kernel is symmetric
        spread_term = spread_term + 2.0 * torch.sum(_mmd_rbf_kernel(ym, forecasts[m:]), dim=0)

        # contributions to the discrepancy term
        disc_term = disc_term + _mmd_rbf_kernel(ym, observation)

    # compute the squared mmd
    mmd2 = spread_term / (num_forecasts - 1) / num_forecasts - 2.0 * disc_term / num_forecasts

    return mmd2


class EnsembleMMDLoss(GeometricBaseLoss):
    r"""
    Computes the maximum mean discrepancy loss for a specific kernel. For details see [1]

    [1] Dziugaite, Gintare Karolina; Roy, Daniel M.; Ghahramani, Zhoubin; Training generative neural networks via Maximum Mean Discrepancy optimization; arXiv:1505.03906
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        squared: Optional[bool] = False,
        pole_mask: Optional[int] = 0,
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

        self.squared = squared

        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.ensemble_distributed = comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1) and ensemble_distributed

        # we also need a variant of the weights split in ensemble direction:
        quad_weight_split = self.quadrature.quad_weight.reshape(1, 1, -1)
        if self.ensemble_distributed:
            quad_weight_split = split_tensor_along_dim(quad_weight_split, dim=-1, num_chunks=comm.get_size("ensemble"))[comm.get_rank("ensemble")]
        quad_weight_split = quad_weight_split.contiguous()
        self.register_buffer("quad_weight_split", quad_weight_split, persistent=False)

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, spatial_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that spatial_weights have NO ensemble dim
        if (spatial_weights is not None) and (spatial_weights.dim() != observations.dim()):
            raise ValueError("the weights have to have the same number of dimensions as observations")

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, lat, lon
        # observations: batch, channels, lat, lon
        B, E, C, H, W = forecasts.shape

        # if ensemble dim is one dimensional then computing the score is quick:
        if (not self.ensemble_distributed) and (forecasts.shape[1] == 1):
            # in this case, CRPS is straightforward
            mmd = _mmd_rbf_kernel(observations, forecasts.squeeze(1)).reshape(B, C, H * W)
        else:
            # transpose forecasts: ensemble, batch, channels, lat, lon
            forecasts = torch.moveaxis(forecasts, 1, 0)

            # now we need to transpose the forecasts into ensemble direction.
            # ideally we split spatial dims
            forecasts = forecasts.reshape(E, B, C, H * W)
            if self.ensemble_distributed:
                ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
                forecasts = distributed_transpose.apply(forecasts, (-1, 0), ensemble_shapes, "ensemble")
            # observations does not need a transpose, but just a split
            observations = observations.reshape(B, C, H * W)
            if self.ensemble_distributed:
                observations = scatter_to_parallel_region(observations, -1, "ensemble")
            if spatial_weights is not None:
                spatial_weights_split = spatial_weights.flatten(-2, -1)
                spatial_weights_split = scatter_to_parallel_region(spatial_weights_split, -1, "ensemble")

            # now, E dimension is local and spatial dim is split further. Compute the mmd
            mmd = _mmd2_ensemble_kernel(observations, forecasts)

        # perform spatial average of crps score
        if spatial_weights is not None:
            mmd = torch.sum(mmd * self.quad_weight_split * spatial_weights_split, dim=-1)
        else:
            mmd = torch.sum(mmd * self.quad_weight_split, dim=-1)
        if self.ensemble_distributed:
            mmd = reduce_from_parallel_region(mmd, "ensemble")

        if self.spatial_distributed:
            mmd = reduce_from_parallel_region(mmd, "spatial")

        if not self.squared:
            mmd = torch.sqrt(mmd)

        # the resulting tensor should have dimension B, C, which is what we return
        return mmd
