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
import math

import torch
import torch.nn as nn
from torch import amp

from makani.utils.losses.base_loss import GeometricBaseLoss, SpectralBaseLoss, LossType
from makani.utils import comm

# distributed stuff
from physicsnemo.distributed.utils import compute_split_shapes, split_tensor_along_dim
from physicsnemo.distributed.mappings import scatter_to_parallel_region, reduce_from_parallel_region
from makani.mpu.mappings import distributed_transpose


def rankdata(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    ordinal ranking along dimension dim
    """
    ndim = x.dim()
    perm = torch.argsort(x, dim=dim, descending=False, stable=True)

    idx = torch.arange(x.shape[dim], device=x.device).reshape([-1 if i == dim else 1 for i in range(ndim)])
    rank = torch.empty_like(x, dtype=torch.long).scatter_(dim=dim, index=perm, src=idx.expand_as(perm)) + 1
    return rank


# @torch.compile
def _crps_ensemble_kernel(observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    CRPS ensemble score from integrating the PDF piecewise
    compare https://github.com/properscoring/properscoring/blob/master/properscoring/_gufuncs.py#L7
    disabling torch compile for the moment due to very long startup times when training large ensembles with ensemble parallelism
    """

    # beware: forecasts are assumed sorted in sorted order
    # get nanmask
    nanmasks = torch.logical_or(torch.isnan(forecasts), torch.isnan(weights))

    # compute total weights
    nweights = torch.where(nanmasks, 0.0, weights)
    total_weight = torch.sum(nweights, dim=0)

    # initial values
    obs_cdf = torch.zeros_like(observation)
    forecast_cdf = torch.zeros_like(observation)
    prev_forecast = torch.zeros_like(observation)
    integral = torch.zeros_like(observation)
    nanmask = torch.zeros_like(observation, dtype=torch.bool)

    # split lists
    nanmasklist = torch.split(nanmasks, 1, dim=0)
    weightslist = torch.split(weights, 1, dim=0)
    forecastlist = torch.split(forecasts, 1, dim=0)
    for n, token in enumerate(zip(forecastlist, weightslist, nanmasklist)):

        # extract variables
        tmpforecast, weight, tmpnanmask = token

        # update nanmask
        nanmask = torch.logical_or(tmpnanmask, nanmask)

        forecast = torch.where(tmpnanmask, prev_forecast, tmpforecast)

        # compute condition
        condition = torch.logical_and(observation < forecast, torch.abs(obs_cdf) < 1.0e-7)

        # compute terms
        term_true = (observation - prev_forecast) * torch.square(forecast_cdf) + (forecast - observation) * torch.square(forecast_cdf - 1)
        term_false = (forecast - prev_forecast) * torch.square(forecast_cdf - obs_cdf)
        increment = torch.where(condition, term_true, term_false)

        # compute integral
        integral = integral + torch.where(nanmask, 0.0, increment)

        # update cdf
        # this only gets updated for values which are not nan
        obs_cdf_new = torch.where(condition, 1.0, obs_cdf)
        obs_cdf = torch.where(nanmask, obs_cdf, obs_cdf_new)
        forecast_cdf = forecast_cdf + weight / total_weight

        # update forcast
        prev_forecast = forecast

    integral = integral + torch.where(torch.abs(obs_cdf) < 1.0e-7, observation - forecast, 0.0)

    # set to nan for first forecasts nan
    integral = torch.where(nanmasklist[0], torch.nan, integral)

    return torch.squeeze(integral, dim=0)


def _crps_skillspread_kernel(observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    alternative CRPS variant that uses spread and skill
    """

    observation = observation.unsqueeze(0)

    # get nanmask
    nanmasks = torch.logical_or(torch.isnan(forecasts), torch.isnan(weights))
    nanmask = torch.sum(nanmasks, dim=0).bool()

    # compute total weights
    nweights = torch.where(nanmasks, 0.0, weights)
    total_weight = torch.sum(nweights, dim=0, keepdim=True)

    # get the ranks for the spread computation
    rank = rankdata(forecasts, dim=0)

    #  ensemble size
    num_ensemble = forecasts.shape[0]

    # get the ensemble spread (total_weight is ensemble size here)
    espread = 2 * torch.mean((2 * rank - num_ensemble - 1) * forecasts, dim=0) * (float(num_ensemble) - 1.0 + alpha) / float(num_ensemble * (num_ensemble - 1))
    eskill = (observation - forecasts).abs().mean(dim=0)

    # crps = torch.where(nanmasks.sum(dim=0) != 0, torch.nan, eskill - 0.5 * espread)
    crps = eskill - 0.5 * espread

    # set to nan for first forecasts nan
    crps = torch.where(nanmask, torch.nan, crps)

    return crps


# @torch.compile
def _crps_gauss_kernel(observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    """
    CRPS Gauss score, assuming the input ensemble is gaussian distributed
    disabling torch compile for the moment due to very long startup times when training large ensembles with ensemble parallelism
    """

    # compute mean var over observations
    mu = torch.mean(forecasts * weights, dim=0)
    sigma = torch.sqrt(torch.mean(torch.square(forecasts - mu.unsqueeze(0)) * weights, dim=0))

    # protect against too small standard deviations
    sigma = torch.clamp(sigma, min=eps)

    # compute normalized observation
    obs_norm = (observation - mu) / sigma

    # compute normalization
    sqrtpi_inv = 1.0 / np.sqrt(np.pi)
    sqrt2_inv = 1.0 / np.sqrt(2.0)

    # compute PDF and CDF
    pdf = sqrtpi_inv * sqrt2_inv * torch.exp(-0.5 * torch.square(obs_norm))
    cdf2m1 = torch.erf(obs_norm * sqrt2_inv)

    # compute score
    crps = sigma * (obs_norm * cdf2m1 + 2.0 * pdf - sqrtpi_inv)

    return crps


class EnsembleCRPSLoss(GeometricBaseLoss):

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        pole_mask: int,
        crps_type: str = "skillspread",
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        alpha: Optional[float] = 1.0,
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
        self.crps_type = crps_type
        self.alpha = alpha
        self.eps = eps

        if (self.crps_type != "skillspread") and (self.alpha < 1.0):
            raise NotImplementedError("The alpha parameter (almost fair CRPS factor) is only supported for the skillspread kernel.")

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

        # if ensemble dim is one dimensional then computing the score is quick:
        if (not self.ensemble_distributed) and (E == 1):
            # in this case, CRPS is straightforward
            crps = torch.abs(observations - forecasts.squeeze(1)).reshape(B, C, H * W)
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
                spatial_weights_split = spatial_weights.flatten(start_dim=-2, end_dim=-1)
                spatial_weights_split = scatter_to_parallel_region(spatial_weights_split, -1, "ensemble")

            # run appropriate crps kernel to compute it pointwise
            if self.crps_type == "cdf":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)  # how does the sorting work out if it is batched
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_ensemble_kernel(observations, forecasts, ensemble_weights)
            elif self.crps_type == "skillspread":
                if self.ensemble_weights is not None:
                    raise NotImplementedError("currently only constant ensemble weights are supported")
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
            elif self.crps_type == "gauss":
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_gauss_kernel(observations, forecasts, ensemble_weights, self.eps)
            else:
                raise ValueError(f"Unknown CRPS crps_type {self.crps_type}")

        # perform ensemble and spatial average of crps score
        if spatial_weights is not None:
            crps = torch.sum(crps * self.quad_weight_split * spatial_weights_split, dim=-1)
        else:
            crps = torch.sum(crps * self.quad_weight_split, dim=-1)

        # since we split spatial dim into ensemble dim, we need to do an ensemble sum as well
        if self.ensemble_distributed:
            crps = reduce_from_parallel_region(crps, "ensemble")

        # we need to do the spatial averaging manually since
        # we are not calling he quadrature forward function
        if self.spatial_distributed:
            crps = reduce_from_parallel_region(crps, "spatial")

        # the resulting tensor should have dimension B, C, which is what we return
        return crps


class EnsembleSpectralCRPSLoss(SpectralBaseLoss):

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        crps_type: str = "skillspread",
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        absolute: Optional[bool] = True,
        alpha: Optional[float] = 1.0,
        eps: Optional[float] = 1.0e-5,
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

        self.spatial_distributed = spatial_distributed and comm.is_distributed("spatial")
        self.ensemble_distributed = ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        self.crps_type = crps_type
        self.alpha = alpha
        self.eps = eps

        if (self.crps_type != "skillspread") and (self.alpha < 1.0):
            raise NotImplementedError("The alpha parameter (almost fair CRPS factor) is only supported for the skillspread kernel.")

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

        # if absolute is true, the loss is computed only on the absolute value of the spectral coefficient
        self.absolute = absolute

        # get the local l weights
        lmax = self.sht.lmax
        ls = torch.arange(lmax).reshape(-1, 1)
        # l_weights = 1 / (2*ls+1)
        l_weights = torch.ones(lmax).reshape(-1, 1)
        if comm.get_size("h") > 1:
            l_weights = split_tensor_along_dim(l_weights, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
        self.register_buffer("l_weights", l_weights, persistent=False)

        # get the local m weights
        mmax = self.sht.mmax
        m_weights = 2 * torch.ones(mmax).reshape(1, -1)
        m_weights[0, 0] = 1.0
        if comm.get_size("w") > 1:
            m_weights = split_tensor_along_dim(m_weights, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]
        self.register_buffer("m_weights", m_weights, persistent=False)

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, spectral_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that spatial_weights have NO ensemble dim
        if (spectral_weights is not None) and (spectral_weights.dim() != observations.dim()):
            raise ValueError("the weights have to have the same number of dimensions as observations")

        # get the data type before stripping amp types
        dtype = forecasts.dtype

        # before anything else compute the transform
        # as the CDF definition doesn't generalize well to more than one-dimensional variables, we treat complex and imaginary part as the same
        with amp.autocast(device_type="cuda", enabled=False):
            forecasts = self.sht(forecasts.float()) / 4.0 / math.pi
            observations = self.sht(observations.float()) / 4.0 / math.pi

        if self.absolute:
            forecasts = torch.abs(forecasts).to(dtype)
            observations = torch.abs(observations).to(dtype)
        else:
            forecasts = torch.view_as_real(forecasts).to(dtype)
            observations = torch.view_as_real(observations).to(dtype)

            # merge complex dimension after channel dimension and flatten
            # this needs to be undone at the end
            forecasts = torch.movedim(forecasts, 5, 3).flatten(2, 3)
            observations = torch.movedim(observations, 4, 2).flatten(1, 2)

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, mmax, lmax
        # observations: batch, channels, mmax, lmax
        B, E, C, H, W = forecasts.shape

        if spectral_weights is None:
            spectral_weights = self.m_weights * self.l_weights
        else:
            spectral_weights = spectral_weights * self.m_weights * self.l_weights

        # if ensemble dim is one dimensional then computing the score is quick:
        if (not self.ensemble_distributed) and (E == 1):
            # in this case, CRPS is straightforward
            crps = torch.abs(observations - forecasts.squeeze(1)).reshape(B, C, H * W)
            spectral_weights_split = spectral_weights.reshape(1, 1, H * W)
        else:
            # transpose forecasts: ensemble, batch, channels, lat, lon
            forecasts = torch.movedim(forecasts, 1, 0)

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

            # tile in complex dim, then flatten last 3 dims
            spectral_weights_split = spectral_weights.reshape(1, 1, H * W)
            if self.ensemble_distributed:
                spectral_weights_split = scatter_to_parallel_region(spectral_weights_split, -1, "ensemble")

            # run appropriate crps kernel to compute it pointwise
            if self.crps_type == "cdf":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_ensemble_kernel(observations, forecasts, ensemble_weights)
            elif self.crps_type == "skillspread":
                if self.ensemble_weights is not None:
                    raise NotImplementedError("currently only constant ensemble weights are supported")
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
            elif self.crps_type == "gauss":
                if self.ensemble_weights is not None:
                    ensemble_weights = self.ensemble_weights[idx]
                else:
                    ensemble_weights = torch.ones_like(forecasts, device=forecasts.device)

                # compute score
                crps = _crps_gauss_kernel(observations, forecasts, ensemble_weights, self.eps)
            else:
                raise ValueError(f"Unknown CRPS crps_type {self.crps_type}")

        # perform spatial average of crps score
        crps = torch.sum(crps * spectral_weights_split, dim=-1)

        if self.spatial_distributed:
            crps = reduce_from_parallel_region(crps, "spatial")

        if self.ensemble_distributed:
            crps = reduce_from_parallel_region(crps, "ensemble")

        # finally undo the folding of the complex dimension into the channel dimension
        if not self.absolute:
            crps = crps.reshape(B, -1, 2).sum(dim=-1)

        # the resulting tensor should have dimension B, C, which is what we return
        return crps
