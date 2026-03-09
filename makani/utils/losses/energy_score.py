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

from makani.utils.losses.base_loss import GeometricBaseLoss, SpectralBaseLoss, LossType
from makani.utils import comm

# distributed stuff
from physicsnemo.distributed.utils import compute_split_shapes, split_tensor_along_dim
from physicsnemo.distributed.mappings import scatter_to_parallel_region, reduce_from_parallel_region
from makani.mpu.mappings import distributed_transpose


class LpEnergyScoreLoss(GeometricBaseLoss):
    """
    Lp Energy Score (Gneiting et al. 2005). Uses Lebesgue norm ||x||_p = (sum |x_i|^p)^(1/p).
    p=2 gives the standard L2/Euclidean energy score.
    """

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
        channel_reduction: Optional[bool] = True,
        alpha: Optional[float] = 1.0,
        beta: Optional[float] = 1.0,
        p: Optional[float] = 2.0,
        eps: Optional[float] = 1.0e-5,
        spread_temper_steps: Optional[int] = 0,
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
        self.channel_reduction = channel_reduction
        self.alpha = alpha
        self.beta = beta
        self.p = float(p)
        self.eps = eps
        self.spread_temper_steps = spread_temper_steps

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
        return 1 if self.channel_reduction else len(self.channel_names)

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str, time_diff_scale: str) -> torch.Tensor:
        if self.channel_reduction:
            chw = torch.ones(1)
        else:
            chw = super().compute_channel_weighting(channel_weight_type, time_diff_scale)
        return chw

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, spatial_weights: Optional[torch.Tensor] = None, lead_time_step: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:

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

        # get the data type before stripping amp types
        dtype = forecasts.dtype

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

        # get nanmask from observations and forecasts
        nanmasks = torch.logical_or(torch.isnan(observations), torch.isnan(ensemble_weights))
        nanmask_bool = nanmasks.sum(dim=0) != 0

        # impute NaN before computation to avoid 0 * NaN = NaN in backward pass
        observations = torch.where(torch.isnan(observations), 0.0, observations)
        forecasts = torch.where(torch.isnan(forecasts), 0.0, forecasts)

        # use broadcasting semantics: Lp norm components (sum |diff|^p, then ^(1/p) later)
        espread = (forecasts.unsqueeze(1) - forecasts.unsqueeze(0)).abs().pow(self.p)
        eskill = (observations - forecasts).abs().pow(self.p)

        # zero out masked positions
        espread = torch.where(nanmask_bool, 0.0, espread)
        eskill = torch.where(nanmask_bool, 0.0, eskill)

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
        if self.channel_reduction:
            espread = espread.sum(dim=-1, keepdim=True)
            eskill = eskill.sum(dim=-1, keepdim=True)

        # just to be sure, mask the diagonal of espread with self.eps
        #espread = torch.where(torch.eye(num_ensemble, device=espread.device).bool().reshape(num_ensemble, num_ensemble, 1, 1), self.eps, espread)
        # get the masks
        espread_mask = torch.where(espread < self.eps, True, False)
        eskill_mask = torch.where(eskill < self.eps, True, False)

        # mask the data
        espread = torch.where(espread_mask, self.eps, espread)
        eskill = torch.where(eskill_mask, self.eps, eskill)

        with amp.autocast(device_type="cuda", enabled=False):

            espread = espread.float()
            eskill = eskill.float()

            # Lp norm = (sum |x|^p)^(1/p); then optional beta exponent (Gneiting et al. 2005)
            espread = espread.pow(1.0 / self.p).pow(self.beta)
            eskill = eskill.pow(1.0 / self.p).pow(self.beta)

        # mask espread and sum
        espread = torch.where(espread_mask, 0.0, espread)
        eskill = torch.where(eskill_mask, 0.0, eskill)
        #espread = torch.where(torch.eye(num_ensemble, device=espread.device).bool().reshape(num_ensemble, num_ensemble, 1, 1), 0.0, espread)
        espread = espread.sum(dim=(0,1)) * (float(num_ensemble) - 1.0 + self.alpha) / float(num_ensemble * num_ensemble * (num_ensemble - 1))

        # sum over ensemble
        eskill = eskill.sum(dim=0) / float(num_ensemble)

        # apply spread tempering (lead_time_step shape (C,) for channel-wise weighting)
        if self.training and self.spread_temper_steps > 0 and lead_time_step is not None:
            scale = lead_time_step.float().to(espread.device) / self.spread_temper_steps
            espread = espread * torch.maximum(scale, torch.ones_like(scale))

        # the resulting tensor should have dimension B, C which is what we return
        loss =  eskill - 0.5 * espread

        return loss


# backward-compatibility alias
L2EnergyScoreLoss = LpEnergyScoreLoss


class SobolevEnergyScoreLoss(SpectralBaseLoss):

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        lmax: Optional[int] = None,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        channel_reduction: Optional[bool] = True,
        alpha: Optional[float] = 1.0,
        beta: Optional[float] = 1.0,
        offset: Optional[float] = 1.0,
        fraction: Optional[float] = 1.0,
        relative_weight: Optional[float] = 1.0,
        eps: Optional[float] = 1.0e-6,
        **kwargs,
    ):

        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            lmax=lmax,
            spatial_distributed=spatial_distributed,
        )

        self.spatial_distributed = spatial_distributed and comm.is_distributed("spatial")
        self.ensemble_distributed = ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        self.channel_reduction = channel_reduction
        self.alpha = alpha
        self.beta = beta
        self.fraction = fraction
        self.offset = offset
        self.relative_weight = relative_weight
        self.eps = eps

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

        # get the local l weights
        l_weights = torch.arange(self.sht.lmax, dtype=torch.float32)
        m_weights = 2 * torch.ones(self.sht.mmax, dtype=torch.float32)
        m_weights[0] = 1.0
        # get meshgrid of weights:
        l_weights, m_weights = torch.meshgrid(l_weights, m_weights, indexing="ij")

        # use the product weights
        lm_weights = (self.offset + self.relative_weight * l_weights * (l_weights + 1)).pow(self.fraction) * m_weights

        # split the tensors along all dimensions:
        if self.spatial_distributed and comm.get_size("h") > 1:
            lm_weights = split_tensor_along_dim(lm_weights, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
        if self.spatial_distributed and comm.get_size("w") > 1:
            lm_weights = split_tensor_along_dim(lm_weights, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]
        lm_weights = lm_weights.contiguous()

        self.register_buffer("lm_weights", lm_weights, persistent=False)

    @property
    def type(self):
        return LossType.Probabilistic

    @property
    def n_channels(self):
        return 1 if self.channel_reduction else len(self.channel_names)

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str, time_diff_scale: str) -> torch.Tensor:
        if self.channel_reduction:
            chw = torch.ones(1)
        else:
            chw = super().compute_channel_weighting(channel_weight_type, time_diff_scale)
        return chw

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, ensemble_weights: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # get the data type before stripping amp types
        dtype = forecasts.dtype

        # before anything else compute the transform
        # as the CDF definition doesn't generalize well to more than one-dimensional variables, we treat complex and imaginary part as the same
        with amp.autocast(device_type="cuda", enabled=False):
            # TODO: check 4 pi normalization
            forecasts = self.sht(forecasts.float()) / math.sqrt(4 * math.pi)
            observations = self.sht(observations.float()) / math.sqrt(4 * math.pi)

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, mmax, lmax
        # observations: batch, channels, mmax, lmax
        B, E, C, H, W = forecasts.shape

        # transpose the forecasts to ensemble, batch, channels, lat, lon and then do distributed transpose into ensemble direction.
        # ideally we split spatial dims
        forecasts = torch.moveaxis(forecasts, 1, 0)
        forecasts = forecasts.reshape(E, B, C, H * W)
        if self.ensemble_distributed:
            ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
            forecasts = distributed_transpose.apply(forecasts, (-1, 0), ensemble_shapes, "ensemble")        # for correct spatial reduction we need to do the same with spatial weights

        lm_weights_split = self.lm_weights.flatten(start_dim=-2, end_dim=-1)
        if self.ensemble_distributed:
            lm_weights_split = scatter_to_parallel_region(lm_weights_split, -1, "ensemble")

        # observations does not need a transpose, but just a split
        observations = observations.reshape(1, B, C, H * W)
        if self.ensemble_distributed:
            observations = scatter_to_parallel_region(observations, -1, "ensemble")

        num_ensemble = forecasts.shape[0]

        # get nanmask from observations and forecasts
        nanmasks = torch.logical_or(torch.isnan(observations), torch.isnan(forecasts))
        nanmask_bool = nanmasks.sum(dim=0) != 0

        # impute NaN before computation to avoid 0 * NaN = NaN in backward pass
        observations = torch.where(torch.isnan(observations), 0.0, observations)
        forecasts = torch.where(torch.isnan(forecasts), 0.0, forecasts)

        # compute the individual distances
        espread = lm_weights_split * (forecasts.unsqueeze(1) - forecasts.unsqueeze(0)).abs().square()
        eskill = lm_weights_split * (observations - forecasts).abs().square()

        # zero out masked positions
        espread = torch.where(nanmask_bool, 0.0, espread)
        eskill = torch.where(nanmask_bool, 0.0, eskill)

        # do the channel reduction first
        if self.channel_reduction:
            espread = espread.sum(dim=-2, keepdim=True)
            eskill = eskill.sum(dim=-2, keepdim=True)

        # do the spatial reduction
        espread = espread.sum(dim=-1, keepdim=False)
        eskill = eskill.sum(dim=-1, keepdim=False)

        # since we split spatial dim into ensemble dim, we need to do an ensemble sum as well
        if self.ensemble_distributed:
            espread = reduce_from_parallel_region(espread, "ensemble")
            eskill = reduce_from_parallel_region(eskill, "ensemble")

        # we need to do the spatial averaging manually since
        if self.spatial_distributed:
            espread = reduce_from_parallel_region(espread, "spatial")
            eskill = reduce_from_parallel_region(eskill, "spatial")

        # just to be sure, mask the diagonal of espread with self.eps
        #espread = torch.where(torch.eye(num_ensemble, device=espread.device).bool().reshape(num_ensemble, num_ensemble, 1, 1), self.eps, espread)
        # get the masks
        espread_mask = torch.where(espread < self.eps, True, False)
        eskill_mask = torch.where(eskill < self.eps, True, False)

        # mask the data
        espread = torch.where(espread_mask, self.eps, espread)
        eskill = torch.where(eskill_mask, self.eps, eskill)

        with amp.autocast(device_type="cuda", enabled=False):

            espread = espread.float()
            eskill = eskill.float()

            # This is according to the definition in Gneiting et al. 2005
            espread = torch.sqrt(espread).pow(self.beta)
            eskill = torch.sqrt(eskill).pow(self.beta)

        # mask espread and sum
        espread = torch.where(espread_mask, 0.0, espread)
        eskill = torch.where(eskill_mask, 0.0, eskill)
        #espread = torch.where(torch.eye(num_ensemble, device=espread.device).bool().reshape(num_ensemble, num_ensemble, 1, 1), 0.0, espread)
        espread = espread.sum(dim=(0,1)) * (float(num_ensemble) - 1.0 + self.alpha) / float(num_ensemble * num_ensemble * (num_ensemble - 1))

        # compute the skill term
        eskill = eskill.sum(dim=0) / float(num_ensemble)

        return (eskill - 0.5 * espread)


class SpectralL2EnergyScoreLoss(SpectralBaseLoss):

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        lmax: Optional[int] = None,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        channel_reduction: Optional[bool] = True,
        alpha: Optional[float] = 1.0,
        beta: Optional[float] = 1.0,
        eps: Optional[float] = 1.0e-3,
        **kwargs,
    ):

        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            lmax=lmax,
            spatial_distributed=spatial_distributed,
        )

        self.spatial_distributed = spatial_distributed and comm.is_distributed("spatial")
        self.ensemble_distributed = ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        self.channel_reduction = channel_reduction
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

    @property
    def type(self):
        return LossType.Probabilistic

    @property
    def n_channels(self):
        return 1 if self.channel_reduction else len(self.channel_names)

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str, time_diff_scale: str) -> torch.Tensor:
        if self.channel_reduction:
            chw = torch.ones(1)
        else:
            chw = super().compute_channel_weighting(channel_weight_type, time_diff_scale)
        return chw

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, ensemble_weights: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # get the data type before stripping amp types
        dtype = forecasts.dtype

        # before anything else compute the transform
        # as the CDF definition doesn't generalize well to more than one-dimensional variables, we treat complex and imaginary part as the same
        forecasts = forecasts.float()
        observations = observations.float()
        with amp.autocast(device_type="cuda", enabled=False):
            # TODO: check 4 pi normalization
            forecasts = self.sht(forecasts) / math.sqrt(4.0 * math.pi)
            observations = self.sht(observations) / math.sqrt(4.0 * math.pi)

        forecasts = forecasts.to(dtype)
        observations = observations.to(dtype)

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, mmax, lmax
        # observations: batch, channels, mmax, lmax
        B, E, C, H, W = forecasts.shape

        # transpose the forecasts to ensemble, batch, channels, lat, lon and then do distributed transpose into ensemble direction.
        # ideally we split spatial dims
        forecasts = torch.moveaxis(forecasts, 1, 0)
        if self.ensemble_distributed:
            ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
            forecasts = distributed_transpose.apply(forecasts, (-1, 0), ensemble_shapes, "ensemble")        # for correct spatial reduction we need to do the same with spatial weights

        lm_weights_split = self.lm_weights
        if self.ensemble_distributed:
            lm_weights_split = scatter_to_parallel_region(lm_weights_split, -1, "ensemble")

        # observations does not need a transpose, but just a split
        if self.ensemble_distributed:
            observations = scatter_to_parallel_region(observations, -1, "ensemble")

        num_ensemble = forecasts.shape[0]

        # get nanmask from observations and forecasts
        nanmasks = torch.logical_or(torch.isnan(observations), torch.isnan(forecasts))
        nanmask_bool = nanmasks.sum(dim=0) != 0

        # impute NaN before computation to avoid 0 * NaN = NaN in backward pass
        observations = torch.where(torch.isnan(observations), 0.0, observations)
        forecasts = torch.where(torch.isnan(forecasts), 0.0, forecasts)

        espread = lm_weights_split * (forecasts.unsqueeze(1) - forecasts.unsqueeze(0)).abs().square()
        eskill = lm_weights_split * (observations - forecasts).abs().square()

        # zero out masked positions
        espread = torch.where(nanmask_bool, 0.0, espread)
        eskill = torch.where(nanmask_bool, 0.0, eskill)

        # do the channel reduction first
        if self.channel_reduction:
            espread = espread.sum(dim=-3, keepdim=True)
            eskill = eskill.sum(dim=-3, keepdim=True)

        # do the spatial m reduction
        espread = espread.sum(dim=-1, keepdim=False)
        eskill = eskill.sum(dim=-1, keepdim=False)

        # since we split m dim into ensemble dim, we need to do an ensemble sum as well
        if self.ensemble_distributed:
            espread = reduce_from_parallel_region(espread, "ensemble")
            eskill = reduce_from_parallel_region(eskill, "ensemble")

        # we need to do the spatial averaging manually since
        if self.spatial_distributed:
            espread = reduce_from_parallel_region(espread, "w")
            eskill = reduce_from_parallel_region(eskill, "w")

        # get the masks
        espread_mask = torch.where(espread < self.eps, True, False)
        eskill_mask = torch.where(eskill < self.eps, True, False)

        # mask the data
        espread = torch.where(espread_mask, self.eps, espread)
        eskill = torch.where(eskill_mask, self.eps, eskill)

        with amp.autocast(device_type="cuda", enabled=False):

            espread = espread.float()
            eskill = eskill.float()

            # This is according to the definition in Gneiting et al. 2005
            espread = torch.sqrt(espread).pow(self.beta)
            eskill = torch.sqrt(eskill).pow(self.beta)

        # mask espread and sum
        espread = torch.where(espread_mask, 0.0, espread)
        eskill = torch.where(eskill_mask, 0.0, eskill)

        # now we have reduced everything and need to sum appropriately (B, C, H)
        espread = espread.sum(dim=(0,1)) * (float(num_ensemble) - 1.0 + self.alpha) / float(num_ensemble * num_ensemble * (num_ensemble - 1))
        eskill = eskill.sum(dim=0) / float(num_ensemble)

        # we now have the loss per wavenumber, which we can normalize
        loss = (eskill - 0.5 * espread)

        # we need to do the spatial averaging manually since
        loss = loss.sum(dim=-1)
        if self.spatial_distributed:
            loss = reduce_from_parallel_region(loss, "h")

        return loss


class SpectralCoherenceLoss(SpectralBaseLoss):

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        lmax: Optional[int] = None,
        relative: Optional[bool] = False,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        alpha: Optional[float] = 1.0,
        eps: Optional[float] = 1.0e-6,
        **kwargs,
    ):

        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            lmax=lmax,
            spatial_distributed=spatial_distributed,
        )

        self.relative = relative
        self.spatial_distributed = spatial_distributed and comm.is_distributed("spatial")
        self.ensemble_distributed = ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        self.eps = eps

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

        # prep ls and ms for broadcasting
        ls = torch.arange(self.sht.lmax).reshape(-1, 1)
        ms = torch.arange(self.sht.mmax).reshape(1, -1)

        lm_weights = torch.ones((self.sht.lmax, self.sht.mmax))
        lm_weights[:, 1:] *= 2.0
        lm_weights = torch.where(ms > ls, 0.0, lm_weights)
        if comm.get_size("h") > 1:
            lm_weights = split_tensor_along_dim(lm_weights, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
        if comm.get_size("w") > 1:
            lm_weights = split_tensor_along_dim(lm_weights, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]
        lm_weights = lm_weights.contiguous()

        self.register_buffer("lm_weights", lm_weights, persistent=False)

    @property
    def type(self):
        return LossType.Probabilistic

    @property
    def n_channels(self):
        return 1 if self.channel_reduction else len(self.channel_names)

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str, time_diff_scale: str) -> torch.Tensor:
        if self.channel_reduction:
            chw = torch.ones(1)
        else:
            chw = super().compute_channel_weighting(channel_weight_type, time_diff_scale)
        return chw

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, ensemble_weights: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # get the data type before stripping amp types
        dtype = forecasts.dtype


        # before anything else compute the transform
        # as the CDF definition doesn't generalize well to more than one-dimensional variables, we treat complex and imaginary part as the same
        with amp.autocast(device_type="cuda", enabled=False):
            # TODO: check 4 pi normalization
            forecasts = self.sht(forecasts.float()) / math.sqrt(4.0 * math.pi)
            observations = self.sht(observations.float()) / math.sqrt(4.0 * math.pi)

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, mmax, lmax
        # observations: batch, channels, mmax, lmax
        B, E, C, H, W = forecasts.shape

        # transpose the forecasts to ensemble, batch, channels, lat, lon and then do distributed transpose into ensemble direction.
        # ideally we split spatial dims
        forecasts = torch.moveaxis(forecasts, 1, 0)
        if self.ensemble_distributed:
            ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
            forecasts = distributed_transpose.apply(forecasts, (-1, 0), ensemble_shapes, "ensemble")        # for correct spatial reduction we need to do the same with spatial weights

        lm_weights_split = self.lm_weights
        if self.ensemble_distributed:
            lm_weights_split = scatter_to_parallel_region(lm_weights_split, -1, "ensemble")

        # observations does not need a transpose, but just a split and broadcast to ensemble dimension
        observations = observations.unsqueeze(0)
        if self.ensemble_distributed:
            observations = scatter_to_parallel_region(observations, -1, "ensemble")

        num_ensemble = forecasts.shape[0]

        # compute power spectral densities of forecasts and observations
        psd_forecasts = (lm_weights_split * forecasts.abs().square()).sum(dim=-1)
        psd_observations = (lm_weights_split * observations.abs().square()).sum(dim=-1)

        # reduce over ensemble parallel region and m spatial dimensions
        if self.ensemble_distributed:
            psd_forecasts = reduce_from_parallel_region(psd_forecasts, "ensemble")
            psd_observations = reduce_from_parallel_region(psd_observations, "ensemble")

        if self.spatial_distributed:
            psd_forecasts = reduce_from_parallel_region(psd_forecasts, "w")
            psd_observations = reduce_from_parallel_region(psd_observations, "w")


        # compute coherence between forecasts and observations
        coherence_forecasts = (lm_weights_split * (forecasts.unsqueeze(0).conj() * forecasts.unsqueeze(1)).real).sum(dim=-1)
        coherence_observations = (lm_weights_split * (forecasts.conj() * observations).real).sum(dim=-1)

        # reduce over ensemble parallel region and m spatial dimensions
        if self.ensemble_distributed:
            coherence_forecasts = reduce_from_parallel_region(coherence_forecasts, "ensemble")
            coherence_observations = reduce_from_parallel_region(coherence_observations, "ensemble")

        if self.spatial_distributed:
            coherence_forecasts = reduce_from_parallel_region(coherence_forecasts, "w")
            coherence_observations = reduce_from_parallel_region(coherence_observations, "w")

        # divide the coherence by the product of the norms (with epsilon for numerical stability)
        coherence_observations = coherence_observations / torch.sqrt(psd_forecasts * psd_observations + self.eps)
        coherence_forecasts = coherence_forecasts / torch.sqrt(psd_forecasts.unsqueeze(0) * psd_forecasts.unsqueeze(1) + self.eps)

        # compute the error in the power spectral density
        psd_skill = (psd_forecasts - psd_observations).square()
        if self.relative:
            psd_skill = psd_skill / (psd_observations + self.eps)
        psd_skill = psd_skill.sum(dim=0) / float(num_ensemble)

        # compute the coherence skill and spread
        coherence_skill = (1.0 - coherence_observations).sum(dim=0) / float(num_ensemble)

        # mask the diagonal of coherence_spread with 0.0
        coherence_spread = torch.where(torch.eye(num_ensemble, device=coherence_forecasts.device).bool().reshape(num_ensemble, num_ensemble, 1, 1, 1), 0.0, 1.0 - coherence_forecasts)
        coherence_spread = coherence_spread.sum(dim=(0, 1)) / float(num_ensemble * (num_ensemble - 1))

        # compute the loss
        if self.relative:
            loss = psd_skill + 2.0 * (coherence_skill - 0.5 * coherence_spread)
        else:
            loss = psd_skill + 2.0 * psd_observations.squeeze(0) * (coherence_skill - 0.5 * coherence_spread)

        # reduce the loss over the l dimensions
        loss = loss.sum(dim=-1)
        if self.spatial_distributed:
            loss = reduce_from_parallel_region(loss, "h")

        # reduce over the channel dimension
        loss = loss.sum(dim=-1)

        return loss

class CorrectedSpectralL2EnergyScoreLoss(SpectralBaseLoss):
    """
    Spectral L2 Energy Score with spread term capped by truth PSD (Option 2).

    Standard ES: spread reward at wavenumber k is P_k * (1 - mean coherence between
    ensemble members). The model can reduce the score by inflating P_k and
    decorrelating phases ("cheap spread").

    This variant uses P_k* (truth PSD) instead of P_k as the coefficient, so the
    spread reward is bounded by the true signal level. Same accuracy term as
    standard ES; only the spread term is modified.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        lmax: Optional[int] = None,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        channel_reduction: Optional[bool] = True,
        alpha: Optional[float] = 1.0,
        beta: Optional[float] = 1.0,
        eps: Optional[float] = 1.0e-3,
        **kwargs,
    ):
        super().__init__(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=channel_names,
            grid_type=grid_type,
            lmax=lmax,
            spatial_distributed=spatial_distributed,
        )

        self.spatial_distributed = spatial_distributed and comm.is_distributed("spatial")
        self.ensemble_distributed = ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        self.channel_reduction = channel_reduction
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

    @property
    def type(self):
        return LossType.Probabilistic

    @property
    def n_channels(self):
        return 1 if self.channel_reduction else len(self.channel_names)

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str, time_diff_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.channel_reduction:
            return torch.ones(1, dtype=torch.float32)
        return super().compute_channel_weighting(channel_weight_type, time_diff_scale)

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, ensemble_weights: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        dtype = forecasts.dtype
        forecasts = forecasts.float()
        observations = observations.float()
        with amp.autocast(device_type="cuda", enabled=False):
            forecasts = self.sht(forecasts) / math.sqrt(4.0 * math.pi)
            observations = self.sht(observations) / math.sqrt(4.0 * math.pi)

        forecasts = forecasts.to(dtype)
        observations = observations.to(dtype)

        B, E, C, H, W = forecasts.shape

        forecasts = torch.moveaxis(forecasts, 1, 0)
        if self.ensemble_distributed:
            ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
            forecasts = distributed_transpose.apply(forecasts, (-1, 0), ensemble_shapes, "ensemble")

        lm_weights_split = self.lm_weights
        if self.ensemble_distributed:
            lm_weights_split = scatter_to_parallel_region(lm_weights_split, -1, "ensemble")

        if self.ensemble_distributed:
            observations = scatter_to_parallel_region(observations, -1, "ensemble")

        num_ensemble = forecasts.shape[0]

        nanmasks = torch.logical_or(torch.isnan(observations), torch.isnan(forecasts))
        nanmask_bool = nanmasks.sum(dim=0) != 0
        observations = torch.where(torch.isnan(observations), 0.0, observations)
        forecasts = torch.where(torch.isnan(forecasts), 0.0, forecasts)

        # PSD per (b, c, l): for Option 2 we need P_pred and P_true
        # P_pred = (1/E) * sum_e sum_m w_m |forecasts[e,b,c,l,m]|^2
        # P_true = sum_m w_m |observations[b,c,l,m]|^2
        # forecasts (E, B, C, L, M), observations (B, C, L, M) after scatter
        psd_pred = (lm_weights_split * forecasts.abs().square()).sum(dim=-1).mean(dim=0)
        psd_true = (lm_weights_split * observations.abs().square()).sum(dim=-1)

        if self.ensemble_distributed:
            psd_pred = reduce_from_parallel_region(psd_pred, "ensemble")
            psd_true = reduce_from_parallel_region(psd_true, "ensemble")
        if self.spatial_distributed:
            psd_pred = reduce_from_parallel_region(psd_pred, "w")
            psd_true = reduce_from_parallel_region(psd_true, "w")

        espread = lm_weights_split * (forecasts.unsqueeze(1) - forecasts.unsqueeze(0)).abs().square()
        eskill = lm_weights_split * (observations - forecasts).abs().square()

        espread = torch.where(nanmask_bool, 0.0, espread)
        eskill = torch.where(nanmask_bool, 0.0, eskill)

        if self.channel_reduction:
            espread = espread.sum(dim=-3, keepdim=True)
            eskill = eskill.sum(dim=-3, keepdim=True)
            psd_pred = psd_pred.sum(dim=-2, keepdim=True)
            psd_true = psd_true.sum(dim=-2, keepdim=True)

        espread = espread.sum(dim=-1, keepdim=False)
        eskill = eskill.sum(dim=-1, keepdim=False)

        if self.ensemble_distributed:
            espread = reduce_from_parallel_region(espread, "ensemble")
            eskill = reduce_from_parallel_region(eskill, "ensemble")
        if self.spatial_distributed:
            espread = reduce_from_parallel_region(espread, "w")
            eskill = reduce_from_parallel_region(eskill, "w")

        espread_mask = torch.where(espread < self.eps, True, False)
        eskill_mask = torch.where(eskill < self.eps, True, False)
        espread = torch.where(espread_mask, self.eps, espread)
        eskill = torch.where(eskill_mask, self.eps, eskill)

        with amp.autocast(device_type="cuda", enabled=False):
            espread = espread.float()
            eskill = eskill.float()
            espread = torch.sqrt(espread).pow(self.beta)
            eskill = torch.sqrt(eskill).pow(self.beta)

        espread = torch.where(espread_mask, 0.0, espread)
        eskill = torch.where(eskill_mask, 0.0, eskill)

        espread = espread.sum(dim=(0, 1)) * (float(num_ensemble) - 1.0 + self.alpha) / float(num_ensemble * num_ensemble * (num_ensemble - 1))
        eskill = eskill.sum(dim=0) / float(num_ensemble)

        # Option 2: spread term = P_true * (1 - Coh^ens) = (P_true / P_pred) * [P_pred * (1 - Coh^ens)]
        # Standard ES has spread contribution 0.5 * espread = P_pred * (1 - Coh^ens) in the decomposition.
        # So we scale the spread term by (P_true / (P_pred + eps)).
        ratio = psd_true / (psd_pred + self.eps)
        loss = eskill - 0.5 * espread * ratio

        loss = loss.sum(dim=-1)
        if self.spatial_distributed:
            loss = reduce_from_parallel_region(loss, "h")

        return loss
