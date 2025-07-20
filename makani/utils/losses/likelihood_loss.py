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

from makani.utils.losses.base_loss import GeometricBaseLoss, LossType
from makani.utils import comm

# distributed stuff
from physicsnemo.distributed.utils import compute_split_shapes, split_tensor_along_dim
from physicsnemo.distributed.mappings import scatter_to_parallel_region, reduce_from_parallel_region
from makani.mpu.mappings import distributed_transpose


# negative log likelihood score, assuming the input ensemble is gaussian distributed
def _log_likelihood_kernel(observation: torch.Tensor, forecasts: torch.Tensor, eps: float) -> torch.Tensor:

    # compute mean var over observations
    sigmasq, mu = torch.var_mean(forecasts, dim=1, correction=0)

    # protect against too small standard deviations
    sigmasq = torch.clamp(sigmasq, min=eps**2)

    # compute normalized observation
    obs_norm = torch.square(observation - mu) / sigmasq

    # compute the negative log likelihood under the gaussian assumption
    log_likelihood = 0.5 * (torch.log(sigmasq) + obs_norm)

    return log_likelihood


class EnsembleNLLLoss(GeometricBaseLoss):
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

        self.spatial_distributed = spatial_distributed and comm.is_distributed("spatial")
        self.ensemble_distributed = ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        self.eps = eps

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

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, lat, lon
        # observations: batch, channels, lat, lon
        B, E, C, H, W = forecasts.shape

        # we assume that spatial_weights have NO ensemble dim
        if (spatial_weights is not None) and (spatial_weights.dim() != observations.dim()):
            raise ValueError("the weights have to have the same number of dimensions as observations")

        # now we need to transpose the forecasts into ensemble direction.
        # ideally we split spatial dims
        forecasts = forecasts.reshape(B, E, C, H * W)
        if self.ensemble_distributed:
            ensemble_shapes = [forecasts.shape[1] for _ in range(comm.get_size("ensemble"))]
            forecasts = distributed_transpose.apply(forecasts, (-1, 1), ensemble_shapes, "ensemble")
        # observations does not need a transpose, but just a split
        observations = observations.reshape(B, C, H * W)
        if self.ensemble_distributed:
            observations = scatter_to_parallel_region(observations, -1, "ensemble")
        if spatial_weights is not None:
            spatial_weights_split = spatial_weights.flatten(-2, -1)
            spatial_weights_split = scatter_to_parallel_region(spatial_weights_split, -1, "ensemble")

        likelihood = _log_likelihood_kernel(observations, forecasts, self.eps)

        # perform spatial average of log likelihood score
        if spatial_weights is not None:
            likelihood = torch.sum(likelihood * self.quad_weight_split * spatial_weights_split, dim=-1)
        else:
            likelihood = torch.sum(likelihood * self.quad_weight_split, dim=-1)
        if self.ensemble_distributed:
            likelihood = reduce_from_parallel_region(likelihood, "ensemble")

        if self.spatial_distributed:
            likelihood = reduce_from_parallel_region(likelihood, "spatial")

        # the resulting tensor should have dimension B, C, which is what we return
        return likelihood
