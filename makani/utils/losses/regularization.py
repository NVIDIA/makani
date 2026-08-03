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
from torch import amp

from makani.utils.losses.base_loss import GeometricBaseLoss, SpectralBaseLoss, LossType
from makani.utils import comm

# distributed stuff
from torch_harmonics.distributed import split_tensor_along_dim
from makani.mpu.mappings import scatter_to_parallel_region, reduce_from_parallel_region, distributed_transpose


class DriftRegularization(GeometricBaseLoss):
    """
    Computes the difference between the means of the forecasts and the observations.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        p: Optional[float] = 1.0,
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
            spatial_distributed=spatial_distributed,
        )

        self.p = p
        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.ensemble_distributed = (
            ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        )

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None, **kwargs):

        if prd.dim() > tar.dim():
            tar = tar.unsqueeze(1)

        # compute difference between the means output has dims
        loss = torch.abs(self.quadrature(prd) - self.quadrature(tar)).pow(self.p)

        # if ensemble
        if prd.dim() == 5:
            loss = torch.mean(loss, dim=1)
            if self.ensemble_distributed:
                loss = reduce_from_parallel_region(loss, "ensemble") / float(comm.get_size("ensemble"))

        return loss


class SpectralRegularization(SpectralBaseLoss):
    """
    Spectral regularization for ensemble forecasts.

    Penalizes the difference between the spectral power of the forecasts and the observations.
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
        # eps is applied inside sqrt(psd_a * psd_b + eps); the output-scale floor is sqrt(eps) ≈ 1e-5,
        # matching the 1e-5/1e-6 guards used by sibling losses
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
        self.ensemble_distributed = (
            ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        )
        self.eps = eps
        self.logarithmic = logarithmic

        # prep ls and ms for broadcasting
        ls = torch.arange(self.sht.lmax).reshape(-1, 1)
        ms = torch.arange(self.sht.mmax).reshape(1, -1)

        lm_weights = torch.ones((self.sht.lmax, self.sht.mmax))
        lm_weights[:, 1:] *= 2.0
        lm_weights = torch.where(ms > ls, 0.0, lm_weights)

        # Guard on spatial_distributed, not just the group size: the SHT above is only
        # the distributed one when spatial_distributed is set, so slicing the weights
        # whenever an h/w group happens to exist would pair local weights with
        # full-size coefficients. forward() reduces under the same condition.
        if self.spatial_distributed and comm.get_size("h") > 1:
            lm_weights = split_tensor_along_dim(lm_weights, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]

        if self.spatial_distributed and comm.get_size("w") > 1:
            lm_weights = split_tensor_along_dim(lm_weights, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]

        self.register_buffer("lm_weights", lm_weights, persistent=False)

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(
        self,
        forecasts: torch.Tensor,
        observations: torch.Tensor,
        ensemble_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

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
        else:
            diff = diff.squeeze(1)

        # do the l reduction
        diff = diff.sum(dim=-1)
        if self.spatial_distributed:
            diff = reduce_from_parallel_region(diff, "h")

        return diff / float(self.sht.lmax)


class CoherenceRegularization(SpectralBaseLoss):
    """
    Mesoscale spectral coherence regularization for ensemble forecasts.

    Penalizes low coherence between each ensemble member and the observation in
    a configurable wavenumber band [lmin, lmax], targeting the mesoscale range
    where decorrelated noise tends to appear.

    For each wavenumber l in the band, the coherence is:

        CrossPSD_l  = sum_{m} w_m * Re( f^(e)_{c,l,m} * conj(y_{c,l,m}) )
        PSD^f_l     = sum_{m} w_m * |f^(e)_{c,l,m}|^2
        PSD^y_l     = sum_{m} w_m * |y_{c,l,m}|^2
        Coh_l       = CrossPSD_l / sqrt(PSD^f_l * PSD^y_l + eps)

    Loss = (1 / |band|) * sum_{l in band} (1 - mean_{e} Coh_l^(e))

    The coherence is signed and lies in [-1, 1], so the loss is 0 for a member
    that matches the observation, 1 for an uncorrelated one, and 2 for a
    sign-flipped one. This deliberately follows SpectralCoherenceLoss rather
    than the sign-blind magnitude-squared convention: squaring would score a
    perfectly anti-correlated forecast as well as a perfect one.

    Sums run over m only, not over channels: the loss framework expects one
    value per channel (see ``SpectralBaseLoss.n_channels``) so that channel
    weighting can be applied downstream.

    When ``ensemble_coherence_weight`` is non-zero, an inter-member term
    ``w * mean_{e != e'} (1 - Coh^(e,e'))`` is added to discourage fully
    independent phases between members. It requires at least two members and is
    skipped otherwise.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        lmin: Optional[int] = None,
        lmax: Optional[int] = None,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        ensemble_weights: Optional[torch.Tensor] = None,
        ensemble_coherence_weight: Optional[float] = 0.0,
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
        self.ensemble_distributed = (
            ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        )
        self.ensemble_coherence_weight = ensemble_coherence_weight
        self.eps = eps

        if ensemble_weights is not None:
            self.register_buffer("ensemble_weights", ensemble_weights, persistent=False)
        else:
            self.ensemble_weights = ensemble_weights

        # clamp band to available wavenumbers. The upper edge is enforced by the SHT
        # itself: lmax is passed to super().__init__, so self.sht.lmax is already the
        # truncation. Only the lower edge needs an explicit mask.
        self.lmin = lmin if lmin is not None else 0
        if self.lmin >= self.sht.lmax:
            raise ValueError(
                f"Error, lmin ({self.lmin}) must be smaller than the SHT truncation lmax ({self.sht.lmax}), otherwise the band is empty."
            )

        # m-summation weights: 1 for m=0, 2 for m>0, 0 for m>l
        ls = torch.arange(self.sht.lmax).reshape(-1, 1)
        ms = torch.arange(self.sht.mmax).reshape(1, -1)

        m_weights = torch.ones((self.sht.lmax, self.sht.mmax))
        m_weights[:, 1:] *= 2.0
        m_weights = torch.where(ms > ls, 0.0, m_weights)

        # wavenumber band mask (1 inside band, 0 outside)
        l_band = torch.zeros(self.sht.lmax)
        l_band[self.lmin : self.sht.lmax] = 1.0

        # Number of wavenumbers in the band, captured BEFORE the spatial split so it is
        # the global width. Normalizing by this makes the loss a mean over the band and
        # therefore independent of both the band width and the h-decomposition.
        band_size = l_band.sum().clamp(min=1.0)

        # split for spatial distribution (l -> h, m -> w)
        if self.spatial_distributed and comm.get_size("h") > 1:
            m_weights = split_tensor_along_dim(m_weights, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
            l_band = split_tensor_along_dim(l_band, dim=0, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
        if self.spatial_distributed and comm.get_size("w") > 1:
            m_weights = split_tensor_along_dim(m_weights, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]

        self.register_buffer("m_weights", m_weights.contiguous(), persistent=False)
        self.register_buffer("l_band", l_band.contiguous(), persistent=False)
        self.register_buffer("band_size", band_size, persistent=False)

    @property
    def type(self):
        return LossType.Probabilistic

    def forward(
        self,
        forecasts: torch.Tensor,
        observations: torch.Tensor,
        spatial_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts expected 5 dims, got {forecasts.dim()}.")

        with amp.autocast(device_type="cuda", enabled=False):
            forecasts = self.sht(forecasts.float()) / math.sqrt(4.0 * math.pi)
            observations = self.sht(observations.float()) / math.sqrt(4.0 * math.pi)

        B, E, C, L, M = forecasts.shape

        # transpose the forecasts to ensemble, batch, channels, lat, lon and then do distributed transpose into ensemble direction.
        # ideally we split spatial dims
        forecasts = torch.moveaxis(forecasts, 1, 0)
        if self.ensemble_distributed:
            ensemble_shapes = [forecasts.shape[0] for _ in range(comm.get_size("ensemble"))]
            forecasts = distributed_transpose(
                forecasts, (-1, 0), ensemble_shapes, "ensemble"
            )  # for correct spatial reduction we need to do the same with spatial weights

        # also split observations along m for the ensemble-distributed transpose
        observations = observations.unsqueeze(0)
        if self.ensemble_distributed:
            observations = scatter_to_parallel_region(observations, -1, "ensemble")

        m_weights_local = self.m_weights
        if self.ensemble_distributed:
            m_weights_local = scatter_to_parallel_region(m_weights_local, -1, "ensemble")

        num_ensemble = forecasts.shape[0]

        # per-member PSD, observation PSD and the forecast-observation cross-PSD.
        # Summing over m only leaves (E, B, C, L); channels are kept so that the
        # framework can weight them downstream.
        psd_fcasts = (m_weights_local * forecasts.abs().square()).sum(dim=-1)
        psd_obs = (m_weights_local * observations.abs().square()).sum(dim=-1)
        cross_psd = (m_weights_local * (forecasts.conj() * observations).real).sum(dim=-1)

        # the inter-member term is optional and undefined for a single member
        compute_inter = (self.ensemble_coherence_weight != 0.0) and (num_ensemble > 1)
        if compute_inter:
            coh_inter = (m_weights_local * (forecasts.unsqueeze(0).conj() * forecasts.unsqueeze(1)).real).sum(dim=-1)

        # do the reduction over the distributed m dimensions. This has to happen before
        # the normalization below, since a ratio of partial sums is not the partial sum
        # of ratios.
        if self.spatial_distributed and comm.get_size("w") > 1:
            psd_fcasts = reduce_from_parallel_region(psd_fcasts, "w")
            psd_obs = reduce_from_parallel_region(psd_obs, "w")
            cross_psd = reduce_from_parallel_region(cross_psd, "w")
            if compute_inter:
                coh_inter = reduce_from_parallel_region(coh_inter, "w")

        if self.ensemble_distributed:
            psd_fcasts = reduce_from_parallel_region(psd_fcasts, "ensemble")
            psd_obs = reduce_from_parallel_region(psd_obs, "ensemble")
            cross_psd = reduce_from_parallel_region(cross_psd, "ensemble")
            if compute_inter:
                coh_inter = reduce_from_parallel_region(coh_inter, "ensemble")

        # signed coherence in [-1, 1] against the observation
        coh_obs = cross_psd / torch.sqrt(psd_fcasts * psd_obs + self.eps)

        # (1 - Coh) averaged over the ensemble -> (B, C, L)
        loss = (1.0 - coh_obs).sum(dim=0) / float(num_ensemble)

        # optional inter-member decoherence penalty, diagonal excluded
        if compute_inter:
            coh_inter = coh_inter / torch.sqrt(psd_fcasts.unsqueeze(0) * psd_fcasts.unsqueeze(1) + self.eps)
            eye = torch.eye(num_ensemble, device=coh_inter.device).bool().reshape(num_ensemble, num_ensemble, 1, 1, 1)
            coh_inter = torch.where(eye, 0.0, 1.0 - coh_inter)
            coh_inter = coh_inter.sum(dim=(0, 1)) / float(num_ensemble * (num_ensemble - 1))
            loss = loss + self.ensemble_coherence_weight * coh_inter

        # restrict to the configured wavenumber band and reduce over l
        loss = (self.l_band * loss).sum(dim=-1)

        # reduce over the spatial dimensions
        if self.spatial_distributed and comm.get_size("h") > 1:
            loss = reduce_from_parallel_region(loss, "h")

        # normalize to a mean over the band
        loss = loss / self.band_size

        return loss
