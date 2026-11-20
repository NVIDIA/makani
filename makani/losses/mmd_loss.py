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

from makani.losses.base_loss import GeometricBaseLoss, LossType, abs_pow, sum_channels_per_step
from makani.distributed import comm


# distributed stuff
from torch_harmonics.distributed import split_tensor_along_dim
from makani.distributed.mappings import scatter_to_parallel_region, reduce_from_parallel_region
from makani.distributed.mappings import distributed_transpose


class GaussianMMDLoss(GeometricBaseLoss):
    r"""
    The squared maximum mean discrepancy between the ensemble and the observation, with a Gaussian kernel, see [1].

    .. math::
        \mathrm{MMD}^2 = k(y, y) + \mathbb{E}\, k(X, X') - 2\, \mathbb{E}\, k(X, y),
        \qquad k(x, y) = \exp\left(-\frac{d(x, y)^2}{2 \sigma^2}\right),

    where :math:`d` is the :math:`L^\beta` distance over the grid, with the quadrature weights (and over the channels
    with ``channel_reduction``), so that :math:`k(y, y) = 1`. A perfect forecast scores 0, and the score grows as the
    members move away from the observation. ``alpha`` blends the fair estimate of :math:`\mathbb{E}\, k(X, X')`, over
    the pairs of distinct members (``alpha = 1``), with the average over all pairs, including each member with itself
    (``alpha = 0``); with a single member, only the latter exists.

    [1] Dziugaite, Gintare Karolina; Roy, Daniel M.; Ghahramani, Zhoubin; Training generative neural networks via Maximum Mean Discrepancy optimization; arXiv:1505.03906
    """

    def __init__(
        self,
        grid,
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        sigma: Optional[float] = 1.0,
        alpha: Optional[float] = 1.0,
        beta: Optional[float] = 2.0,
        channel_reduction: Optional[bool] = False,
        **kwargs,
    ):

        super().__init__(
            grid=grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            spatial_distributed=spatial_distributed,
        )

        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.ensemble_distributed = (
            comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1) and ensemble_distributed
        )
        self.alpha = alpha
        self.beta = beta
        self.channel_reduction = channel_reduction
        self.sigma = sigma

        # we also need a variant of the weights split in ensemble direction:
        quad_weight_split = self.quadrature.quad_weight.reshape(1, 1, -1)
        if self.ensemble_distributed:
            quad_weight_split = split_tensor_along_dim(quad_weight_split, dim=-1, num_chunks=comm.get_size("ensemble"))[
                comm.get_rank("ensemble")
            ]
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
        if self.channel_reduction:
            return 1
        else:
            return len(self.channel_names)

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str, time_diff_scale: str) -> torch.Tensor:
        if self.channel_reduction:
            return torch.ones(1)
        return super().compute_channel_weighting(channel_weight_type, time_diff_scale)

    def forward(
        self,
        forecasts: torch.Tensor,
        observations: torch.Tensor,
        spatial_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that spatial_weights have NO ensemble dim
        if (spatial_weights is not None) and (spatial_weights.dim() != observations.dim()):
            spdim = spatial_weights.dim()
            odim = observations.dim()
            raise ValueError(
                f"the weights have to have the same number of dimensions (found {spdim}) as observations (found {odim})."
            )

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, lat, lon
        # observations: batch, channels, lat, lon
        B, E, C, H, W = forecasts.shape

        # get the data type before stripping amp types
        dtype = forecasts.dtype

        # transpose the forecasts to ensemble, batch, channels, lat, lon and then do distributed transpose into ensemble direction.
        # ideally we split spatial dims
        forecasts = torch.moveaxis(forecasts, 1, 0)
        forecasts = forecasts.reshape(E, B, C, H * W)
        if self.ensemble_distributed:
            ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
            forecasts = distributed_transpose(forecasts, (-1, 0), ensemble_shapes, "ensemble")

        # observations does not need a transpose, but just a split
        observations = observations.reshape(1, B, C, H * W)
        if self.ensemble_distributed:
            observations = scatter_to_parallel_region(observations, -1, "ensemble")

        # for correct spatial reduction we need to do the same with spatial weights
        if spatial_weights is not None:
            spatial_weights_split = spatial_weights.flatten(start_dim=-2, end_dim=-1)
            if self.ensemble_distributed:
                spatial_weights_split = scatter_to_parallel_region(spatial_weights_split, -1, "ensemble")

        if self.ensemble_weights is not None:
            raise NotImplementedError("currently only constant ensemble weights are supported")

        #  ensemble size
        num_ensemble = forecasts.shape[0]

        # use broadcasting semantics to compute spread and skill and sum over channels (vector norm)
        espread = abs_pow(forecasts.unsqueeze(1) - forecasts.unsqueeze(0), self.beta)
        eskill = abs_pow(observations - forecasts, self.beta)

        # do the spatial reduction, with the weights in the dtype of the score, float32 at leasts
        quad_weight_split = self.quad_weight_split.to(dtype=torch.promote_types(eskill.dtype, torch.float32))
        if spatial_weights is not None:
            espread = torch.sum(espread * quad_weight_split * spatial_weights_split, dim=-1)
            eskill = torch.sum(eskill * quad_weight_split * spatial_weights_split, dim=-1)
        else:
            espread = torch.sum(espread * quad_weight_split, dim=-1)
            eskill = torch.sum(eskill * quad_weight_split, dim=-1)

        # since we split spatial dim into ensemble dim, we need to do an ensemble sum as well
        if self.ensemble_distributed:
            espread = reduce_from_parallel_region(espread, "ensemble")
            eskill = reduce_from_parallel_region(eskill, "ensemble")

        # we need to do the spatial averaging manually since
        # we are not calling the quadrature forward function
        if self.spatial_distributed:
            espread = reduce_from_parallel_region(espread, "spatial")
            eskill = reduce_from_parallel_region(eskill, "spatial")

        # do the channel reduction, per lead step: espread is (E, E, B, C), eskill (E, B, C)
        if self.channel_reduction:
            espread = sum_channels_per_step(espread, -1, len(self.channel_names))
            eskill = sum_channels_per_step(eskill, -1, len(self.channel_names))

        # apply the Gaussian kernel to the squared distances, d^2 = (sum |x - y|^beta)^(2 / beta)
        two_sigma_sq = 2.0 * self.sigma * self.sigma
        kspread = torch.exp(-abs_pow(espread, 2.0 / self.beta) / two_sigma_sq)
        kskill = torch.exp(-abs_pow(eskill, 2.0 / self.beta) / two_sigma_sq)

        # E k(X, y)
        kskill = kskill.sum(dim=0) / float(num_ensemble)

        # E k(X, X'): the fair estimate over the pairs of distinct members, blended by alpha with the average over
        # all pairs, which includes each member with itself, where k = 1
        offdiagonal = torch.where(
            torch.eye(num_ensemble, device=kspread.device, dtype=torch.bool).reshape(num_ensemble, num_ensemble, 1, 1),
            0.0,
            kspread,
        ).sum(dim=(0, 1))
        all_pairs = (offdiagonal + float(num_ensemble)) / float(num_ensemble * num_ensemble)
        if num_ensemble > 1:
            fair = offdiagonal / float(num_ensemble * (num_ensemble - 1))
            kspread = self.alpha * fair + (1.0 - self.alpha) * all_pairs
        else:
            kspread = all_pairs

        # the result has the dimensions B, C (one per lead step with channel_reduction); k(y, y) = 1
        return 1.0 + kspread - 2.0 * kskill
