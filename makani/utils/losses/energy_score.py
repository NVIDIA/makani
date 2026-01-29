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


class L2EnergyScoreLoss(GeometricBaseLoss):

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
        self.channel_reduction = channel_reduction
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

        # get nanmask from the observarions
        nanmasks = torch.logical_or(torch.isnan(observations), torch.isnan(ensemble_weights))

        # use broadcasting semantics to compute spread and skill and sum over channels (vector norm)
        espread = (forecasts.unsqueeze(1) - forecasts.unsqueeze(0)).abs().square()
        eskill = (observations - forecasts).abs().square()

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
        if self.channel_reduction:
            espread = espread.sum(dim=-1, keepdim=True)
            eskill = eskill.sum(dim=-1, keepdim=True)

        # just to be sure, mask the diagonal of espread with self.eps
        espread = torch.where(torch.eye(num_ensemble, device=espread.device).bool().reshape(num_ensemble, num_ensemble, 1, 1), self.eps, espread)

        with amp.autocast(device_type="cuda", enabled=False):

            espread = espread.float()
            eskill = eskill.float()

            # This is according to the definition in Gneiting et al. 2005
            espread = torch.sqrt(espread).pow(self.beta)
            eskill = torch.sqrt(eskill).pow(self.beta)

        # mask the diagonal of espread and sum
        espread = torch.where(torch.eye(num_ensemble, device=espread.device).bool().reshape(num_ensemble, num_ensemble, 1, 1), 0.0, espread)
        espread = espread.sum(dim=(0,1)) * (float(num_ensemble) - 1.0 + self.alpha) / float(num_ensemble * num_ensemble * (num_ensemble - 1))

        # sum over ensemble
        eskill = eskill.sum(dim=0) / float(num_ensemble)

        # the resulting tensor should have dimension B, C which is what we return
        loss =  eskill - 0.5 * espread

        return loss

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
        lm_weights = (self.offset + l_weights * (l_weights + 1)).pow(self.fraction) * m_weights

        # split the tensors along all dimensions:
        lm_weights = l_weights * m_weights
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
        return torch.ones(1)

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, ensemble_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

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

        # get nanmask from the observarions
        nanmasks = torch.logical_or(torch.isnan(observations), torch.isnan(forecasts))

        # compute the individual distances
        espread = lm_weights_split * (forecasts.unsqueeze(1) - forecasts.unsqueeze(0)).abs().square()
        eskill = lm_weights_split * (observations - forecasts).abs().square()

        # perform masking before any reduction
        espread = torch.where(nanmasks.sum(dim=0) != 0, 0.0, espread)
        eskill = torch.where(nanmasks.sum(dim=0) != 0, 0.0, eskill)

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
        espread = torch.where(torch.eye(num_ensemble, device=espread.device).bool().reshape(num_ensemble, num_ensemble, 1, 1), self.eps, espread)

        with amp.autocast(device_type="cuda", enabled=False):

            espread = espread.float()
            eskill = eskill.float()

            # This is according to the definition in Gneiting et al. 2005
            espread = torch.sqrt(espread).pow(self.beta)
            eskill = torch.sqrt(eskill).pow(self.beta)

        # mask the diagonal of espread and sum
        espread = torch.where(torch.eye(num_ensemble, device=espread.device).bool().reshape(num_ensemble, num_ensemble, 1, 1), 0.0, espread)
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
        return torch.ones(1)

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, ensemble_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

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

        # get nanmask from the observarions
        nanmasks = torch.logical_or(torch.isnan(observations), torch.isnan(forecasts))

        espread = lm_weights_split * (forecasts.unsqueeze(1) - forecasts.unsqueeze(0)).abs().square()
        eskill = lm_weights_split * (observations - forecasts).abs().square()

        # perform masking before any reduction
        espread = torch.where(nanmasks.sum(dim=0) != 0, 0.0, espread)
        eskill = torch.where(nanmasks.sum(dim=0) != 0, 0.0, eskill)

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
        return 1

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str, time_diff_scale: str) -> torch.Tensor:
        return torch.ones(1)

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, ensemble_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

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