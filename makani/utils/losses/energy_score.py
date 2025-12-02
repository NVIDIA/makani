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

from makani.utils.losses.base_loss import GeometricBaseLoss, LossType
from makani.utils import comm

import torch_harmonics as th
import torch_harmonics.distributed as thd

# distributed stuff
from physicsnemo.distributed.utils import compute_split_shapes, split_tensor_along_dim
from physicsnemo.distributed.mappings import scatter_to_parallel_region, reduce_from_parallel_region
from makani.mpu.mappings import distributed_transpose


class EnergyScoreLoss(GeometricBaseLoss):

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        pole_mask: int,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        alpha: Optional[float] = 1.0,
        beta: Optional[float] = 2.0,
        eps: Optional[float] = 1.0e-5,
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

        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.ensemble_distributed = comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1) and ensemble_distributed
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        # we also need a variant of the weights split in ensemble direction:
        quad_weight_split = self.quadrature.quad_weight.reshape(1, 1, -1)
        if self.ensemble_distributed:
            quad_weight_split = split_tensor_along_dim(quad_weight_split, dim=-1, num_chunks=comm.get_size("ensemble"))[comm.get_rank("ensemble")]
        quad_weight_split = quad_weight_split.contiguous()
        self.register_buffer("quad_weight_split", quad_weight_split, persistent=False)

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

    @property
    def type(self):
        return LossType.Probabilistic

    @property
    def n_channels(self):
        return 1

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str, time_diff_scale: str) -> torch.Tensor:
        return torch.ones(1)

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, spatial_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that spatial_weights have NO ensemble dim
        if (spatial_weights is not None) and (spatial_weights.dim() != observations.dim()):
            spdim = spatial_weights.dim()
            odim = observations.dim()
            raise ValueError(f"the weights have to have the same number of dimensions (found {spdim}) as observations (found {odim}).")

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, lat, lon
        # observations: batch, channels, lat, lon
        B, E, C, H, W = forecasts.shape

        # transpose the forecasts to ensemble, batch, channels, lat, lon and then do distributed transpose into ensemble direction.
        # ideally we split spatial dims
        forecasts = torch.moveaxis(forecasts, 1, 0)
        forecasts = forecasts.reshape(E, B, C, H * W)
        if self.ensemble_distributed:
            ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
            forecasts = distributed_transpose.apply(forecasts, (-1, 0), ensemble_shapes, "ensemble")

        # observations does not need a transpose, but just a split
        observations = observations.reshape(1, B, C, H * W)
        if self.ensemble_distributed:
            observations = scatter_to_parallel_region(observations, -1, "ensemble")

        # for correct spatial reduction we need to do the same with spatial weights
        if spatial_weights is not None:
            spatial_weights_split = spatial_weights.flatten(start_dim=-2, end_dim=-1)
            spatial_weights_split = scatter_to_parallel_region(spatial_weights_split, -1, "ensemble")

        if self.ensemble_weights is not None:
            raise NotImplementedError("currently only constant ensemble weights are supported")
        else:
            ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

        #  ensemble size
        num_ensemble = forecasts.shape[0]

        # get nanmask from the observarions
        nanmasks = torch.logical_or(torch.isnan(observations), torch.isnan(ensemble_weights))

        # use broadcasting semantics to compute spread and skill and sum over channels (vector norm)
        espread = (forecasts.unsqueeze(1) - forecasts.unsqueeze(0)).abs().pow(self.beta)
        eskill = (observations - forecasts).abs().pow(self.beta)

        # perform masking before any reduction
        espread = torch.where(nanmasks.sum(dim=0) != 0, 0.0, espread)
        eskill = torch.where(nanmasks.sum(dim=0) != 0, 0.0, eskill)

        # do the spatial reduction
        if spatial_weights is not None:
            espread = torch.sum(espread * self.quad_weight_split * spatial_weights_split, dim=-1)
            eskill = torch.sum(eskill * self.quad_weight_split * spatial_weights_split, dim=-1)
        else:
            espread = torch.sum(espread * self.quad_weight_split, dim=-1)
            eskill = torch.sum(eskill * self.quad_weight_split, dim=-1)

        # since we split spatial dim into ensemble dim, we need to do an ensemble sum as well
        if self.ensemble_distributed:
            espread = reduce_from_parallel_region(espread, "ensemble")
            eskill = reduce_from_parallel_region(eskill, "ensemble")

        # we need to do the spatial averaging manually since
        # we are not calling the quadrature forward function
        if self.spatial_distributed:
            espread = reduce_from_parallel_region(espread, "spatial")
            eskill = reduce_from_parallel_region(eskill, "spatial")

        # do the channel reduction while ignoring NaNs
        # if channel weights are required they should be added here to the reduction
        espread = espread.sum(dim=-1, keepdim=True)
        eskill = eskill.sum(dim=-1, keepdim=True)

        # now we have reduced everything and need to sum appropriately
        espread = espread.sum(dim=(0,1)).pow(1/self.beta) * (float(num_ensemble) - 1.0 + self.alpha) / float(num_ensemble * num_ensemble * (num_ensemble - 1))
        eskill = eskill.sum(dim=0).pow(1/self.beta) / float(num_ensemble)

        # the resulting tensor should have dimension B, C which is what we return
        return eskill - 0.5 * espread