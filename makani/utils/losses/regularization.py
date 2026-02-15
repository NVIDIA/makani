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
from torch import amp

from makani.utils.losses.base_loss import GeometricBaseLoss, SpectralBaseLoss, LossType
from makani.utils import comm

# distributed stuff
from physicsnemo.distributed.utils import compute_split_shapes, split_tensor_along_dim
from physicsnemo.distributed.mappings import scatter_to_parallel_region, reduce_from_parallel_region


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

class SpectralRegularization(SpectralBaseLoss):

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
        eps: Optional[float] = 1.0e-10,
        logarithmic: Optional[bool] = False,
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
        self.eps = eps
        self.logarithmic = logarithmic

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

        self.register_buffer("lm_weights", lm_weights, persistent=False)

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor, ensemble_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() == 5:
            B, E = forecasts.shape[0:2]
            observations = observations.unsqueeze(1)
        elif forecasts.dim() == 4:
            B = forecasts.shape[0]
            E = -1
            forecasts = forecasts.unsqueeze(1)
            observations = observations.unsqueeze(1)
        else:
            raise ValueError(f"Error, forecasts tensor expected to have 4 or 5 dimensions but found {forecasts.dim()}.")

        # get the data type before stripping amp types
        dtype = forecasts.dtype

        # before anything else compute the transform
        # as the CDF definition doesn't generalize well to more than one-dimensional variables, we treat complex and imaginary part as the same
        with amp.autocast(device_type="cuda", enabled=False):
            # TODO: check 4 pi normalization
            forecasts = self.sht(forecasts.float()).abs().pow(2) / (4.0 * math.pi)
            observations = self.sht(observations.float()).abs().pow(2) / (4.0 * math.pi)

        # we assume the following shapes:
        # B, E, C, H, W (where H, W are spectral dims now)
        C, H, W = forecasts.shape[-3:]

        # get nanmask from the observarions
        nanmasks = torch.logical_or(torch.isnan(observations), torch.isnan(forecasts))

        # do the summation over the ms first to obtain the PSDs
        forecasts = (self.lm_weights * forecasts).sum(dim=-1)
        observations = (self.lm_weights * observations).sum(dim=-1)

        if self.spatial_distributed:
            forecasts = reduce_from_parallel_region(forecasts, "w")
            observations = reduce_from_parallel_region(observations, "w")

        if self.logarithmic:
            forecasts = torch.log(forecasts)
            observations = torch.log(observations)

        diff = (forecasts - observations).abs()

        if E > 0:
            diff = diff.sum(dim=1) / float(E)
            if self.ensemble_distributed:
                diff = reduce_from_parallel_region(diff, "ensemble") / float(comm.get_size("ensemble"))

        # do the l reduction
        diff = diff.sum(dim=-1)
        if self.spatial_distributed:
            diff = reduce_from_parallel_region(diff, "h")

        return diff / float(self.sht.lmax)