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

import math
from typing import Optional, Sequence, Tuple

import torch

import torch_harmonics as th
import torch_harmonics.distributed as thd

from makani.distributed import comm
from makani.distributed.mappings import scatter_to_parallel_region, reduce_from_parallel_region, distributed_transpose
from torch_harmonics.distributed import split_tensor_along_dim

from makani.losses import CRPSLoss, LossType
from makani.utils.grids import GridQuadrature
from makani.metrics.base_metric import _sanitize_shapes, _welford_reduction_helper, GeometricBaseMetric


class GeometricL1(GeometricBaseMetric):
    def __init__(
        self,
        grid,
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid=grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        if weight is not None:
            diff = self.quadrature(torch.abs(x - y) * weight)
        else:
            diff = self.quadrature(torch.abs(x - y))

        # reduce:
        if self.channel_reduction == "mean":
            diff = torch.mean(diff, dim=1)
        elif self.channel_reduction == "sum":
            diff = torch.sum(diff, dim=1)

        if self.batch_reduction == "mean":
            diff = torch.mean(diff, dim=0)
        elif self.batch_reduction == "sum":
            diff = torch.sum(diff, dim=0)

        return diff


class GeometricRMSE(GeometricBaseMetric):
    def __init__(
        self,
        grid,
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid=grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

    def combine(self, vals, counts, dim=0):
        vals, counts = _sanitize_shapes(vals, counts, dim=dim)
        vals_res, counts_res = _welford_reduction_helper(torch.square(vals), counts, self.batch_reduction, dim=dim)
        vals_res = torch.sqrt(vals_res)
        return vals_res, counts_res

    def finalize(self, vals, counts):
        if self.batch_reduction == "mean":
            return vals
        else:
            return vals / torch.sqrt(counts)

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        if weight is not None:
            diff = self.quadrature(torch.square(x - y) * weight)
        else:
            diff = self.quadrature(torch.square(x - y))

        # reduce channels:
        if self.channel_reduction == "mean":
            diff = torch.mean(diff, dim=1)
        elif self.channel_reduction == "sum":
            diff = torch.sum(diff, dim=1)

        # reduce batch:
        if self.batch_reduction == "mean":
            diff = torch.mean(diff, dim=0)
        elif self.batch_reduction == "sum":
            diff = torch.sum(diff, dim=0)

        # compute square root:
        result = torch.sqrt(diff)

        return result


class GeometricACC(GeometricBaseMetric):
    def __init__(
        self,
        grid,
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        method: Optional[str] = "macro",
        bias: Optional[torch.Tensor] = None,
        eps: Optional[float] = 1e-8,
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid=grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

        self.method = method
        self.eps = eps

        if bias is not None:
            if comm.get_size("w") > 1:
                bias = split_tensor_along_dim(bias, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]
            if comm.get_size("h") > 1:
                bias = split_tensor_along_dim(bias, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
            self.register_buffer("bias", bias, persistent=False)

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        counts = super().compute_counts(inp, weight)
        if self.method == "micro":
            counts = counts.unsqueeze(-1)
        return counts

    def finalize(self, vals, counts):
        if self.method == "micro":
            return vals[..., 0] / torch.sqrt(vals[..., 1] * vals[..., 2])
        else:
            return super().finalize(vals, counts)

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        if hasattr(self, "bias"):
            x = x - self.bias
            y = y - self.bias

        if weight is not None:
            cov_xy = self.quadrature(x * y * weight)
            var_x = self.quadrature(torch.square(x) * weight)
            var_y = self.quadrature(torch.square(y) * weight)
        else:
            cov_xy = self.quadrature(x * y)
            var_x = self.quadrature(torch.square(x))
            var_y = self.quadrature(torch.square(y))

        # compute ratio
        if self.method == "macro":
            acc = cov_xy / (torch.sqrt(var_x * var_y) + self.eps)
            if weight is not None:
                # weight the ACC of every sample by its valid area, compute_counts counts it with that area
                acc = acc * self.quadrature(weight.to(dtype=acc.dtype))
        else:
            # stack along dim -1:
            # we form the ratio in the finalization step
            acc = torch.stack([cov_xy, var_x, var_y], dim=-1)

        # reduce
        if self.channel_reduction == "mean":
            acc = torch.mean(acc, dim=1)
        elif self.channel_reduction == "sum":
            acc = torch.sum(acc, dim=1)

        if self.batch_reduction == "mean":
            acc = torch.mean(acc, dim=0)
        elif self.batch_reduction == "sum":
            acc = torch.sum(acc, dim=0)

        return acc


class GeometricSpread(GeometricBaseMetric):
    def __init__(
        self,
        grid,
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):

        super().__init__(
            grid=grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

    @property
    def type(self):
        return LossType.Probabilistic

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            if self.batch_reduction == "mean":
                raise ValueError(
                    "Batch reduction mode 'mean' is not supported when weights are provided. Use 'sum' instead."
                )
            elif self.batch_reduction == "sum":
                counts = torch.sum(self.quadrature(weight.to(dtype=inp.dtype)), dim=0)
            else:
                raise ValueError(f"Batch reduction mode '{self.batch_reduction}' is not supported")
        else:
            if self.batch_reduction == "mean":
                counts = torch.ones(size=(inp.shape[2],), device=inp.device, dtype=inp.dtype)
            elif self.batch_reduction == "sum":
                counts = torch.full(size=(inp.shape[2],), fill_value=inp.shape[0], device=inp.device, dtype=inp.dtype)

        if self.channel_reduction == "mean":
            counts = torch.mean(counts, dim=0)
        elif self.channel_reduction == "sum":
            counts = torch.sum(counts, dim=0)

        return counts

    def forward(
        self, forecasts: torch.Tensor, observations: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that the input is 5 dimensional, where dim=1 is the ensemble dim
        # ensemble mean size:
        # by construction, the ensemble members are evenly distributed among the ranks
        ens_size = forecasts.shape[1] * comm.get_size("ensemble")

        # compute ensemble mean:
        predictions = torch.sum(forecasts, dim=1)
        if comm.get_size("ensemble") > 1:
            predictions = reduce_from_parallel_region(predictions, "ensemble")
        predictions = predictions / float(ens_size)

        # compute the local portion of skill and spread
        spread = torch.sum(torch.square(predictions.unsqueeze(1) - forecasts), dim=1)

        # compute quadrature, the spread over the valid area of every sample
        if weight is not None:
            weight = weight.to(dtype=spread.dtype)
            area = self.quadrature(weight)
            spread = self.quadrature(spread * weight) / torch.clamp(area, min=torch.finfo(spread.dtype).tiny)
        else:
            spread = self.quadrature(spread)
            area = None

        # average spread over ensemble
        if comm.get_size("ensemble") > 1:
            spread = reduce_from_parallel_region(spread, "ensemble")
        spread = torch.sqrt(spread / float(ens_size - 1))

        # weight the spread of every sample by its valid area, compute_counts counts it with that area
        if area is not None:
            spread = spread * area

        # final reduction
        if self.channel_reduction == "mean":
            spread = torch.mean(spread, dim=1)
        elif self.channel_reduction == "sum":
            spread = torch.sum(spread, dim=1)

        if self.batch_reduction == "mean":
            spread = torch.mean(spread, dim=0)
        elif self.batch_reduction == "sum":
            spread = torch.sum(spread, dim=0)

        return spread


class GeometricSSR(GeometricBaseMetric):
    def __init__(
        self,
        grid,
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        eps: Optional[float] = 1e-6,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid=grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

        self.ensemble_distributed = (
            ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        )

        # regularizer
        self.eps = eps

    @property
    def type(self):
        return LossType.Probabilistic

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            if self.batch_reduction == "mean":
                raise ValueError(
                    "Batch reduction mode 'mean' is not supported when weights are provided. Use 'sum' instead."
                )
            elif self.batch_reduction == "sum":
                counts = torch.sum(self.quadrature(weight.to(dtype=inp.dtype)), dim=0)
            else:
                raise ValueError(f"Batch reduction mode '{self.batch_reduction}' is not supported")
        else:
            if self.batch_reduction == "mean":
                counts = torch.ones(size=(inp.shape[2],), device=inp.device, dtype=inp.dtype)
            elif self.batch_reduction == "sum":
                counts = torch.full(size=(inp.shape[2],), fill_value=inp.shape[0], device=inp.device, dtype=inp.dtype)

        if self.channel_reduction == "mean":
            counts = torch.mean(counts, dim=0)
        elif self.channel_reduction == "sum":
            counts = torch.sum(counts, dim=0)

        return counts

    def forward(
        self, forecasts: torch.Tensor, observations: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that the input is 5 dimensional, where dim=1 is the ensemble dim
        # ensemble mean size:
        # by construction, the ensemble members are evenly distributed among the ranks
        ens_size = forecasts.shape[1] * comm.get_size("ensemble")

        # compute ensemble mean:
        predictions = torch.sum(forecasts, dim=1)
        if self.ensemble_distributed:
            predictions = reduce_from_parallel_region(predictions, "ensemble")
        predictions = predictions / float(ens_size)

        # compute the local portion of skill and spread
        skill = torch.square(predictions - observations)
        spread = torch.sum(torch.square(predictions.unsqueeze(1) - forecasts), dim=1)

        # average spread over ensemble
        if self.ensemble_distributed:
            spread = reduce_from_parallel_region(spread, "ensemble")
        spread = spread / float(ens_size - 1)

        # compute SSR for each node
        if weight is not None:
            skill = self.quadrature(skill * weight)
            spread = self.quadrature(spread * weight)
        else:
            skill = self.quadrature(skill)
            spread = self.quadrature(spread)

        # compute the SSR including weighting factor
        ssr = torch.sqrt(spread / torch.clamp(skill - spread / float(ens_size), min=self.eps))
        if weight is not None:
            # weight the SSR of every sample by its valid area, compute_counts counts it with that area
            ssr = ssr * self.quadrature(weight.to(dtype=ssr.dtype))

        # reduce
        if self.channel_reduction == "mean":
            ssr = torch.mean(ssr, dim=1)
        elif self.channel_reduction == "sum":
            ssr = torch.sum(ssr, dim=1)

        if self.batch_reduction == "mean":
            ssr = torch.mean(ssr, dim=0)
        elif self.batch_reduction == "sum":
            ssr = torch.sum(ssr, dim=0)

        return ssr


def resolved_degree(grid) -> int:
    r"""
    The exclusive degree up to which the forward transform of torch-harmonics recovers a band-limited field on
    ``grid``: the smaller of what its latitudes and quadrature resolve and what its longitudes resolve. On the
    721 x 1440 equiangular grid of ERA5 that is 720, above the truncation :func:`makani.utils.grids.spherical_bandlimit`
    gives the spectral layers.
    """
    return min(grid.max_recoverable_degree, grid.max_azimuthal_order)


def kinetic_energy_bands(lmax: int, edges: Optional[Sequence[int]] = None) -> Tuple[int, ...]:
    r"""
    The edges of the degree bands of :class:`GeometricKineticEnergyRatio` on a transform of exclusive degree
    ``lmax``: band ``i`` holds the degrees ``edges[i] <= l < edges[i + 1]``. By default the synoptic scales
    (``l < 60``), the mesoscale up to the usual truncation of the 0.25 degree grids (``60 <= l < 360``) and what
    the grid resolves above that (``360 <= l < lmax``), with the edges at or above ``lmax`` dropped, so that the
    bands are the same in absolute degrees for models of different resolution.
    """
    if edges is None:
        edges = [e for e in (0, 60, 360) if e < lmax] + [lmax]
    edges = tuple(int(e) for e in edges)
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f"The band edges have to be at least two increasing degrees, got {edges}")
    if edges[0] < 0 or edges[-1] > lmax:
        raise ValueError(f"The band edges have to lie in [0, {lmax}] for a transform of degree {lmax}, got {edges}")
    return edges


class GeometricKineticEnergyRatio(GeometricBaseMetric):
    r"""
    Kinetic energy of the forecast relative to that of the observation, in bands of spherical harmonic degree.

    The channels are pairs of wind components, interleaved as ``(u_0, v_0, u_1, v_1, ...)``. The vector SHT of a
    pair gives the coefficients of its velocity potential and streamfunction, and the kinetic energy per unit mass
    in degree :math:`\ell` is

    .. math::
        E(\ell) = \frac{\ell (\ell + 1)}{2} \sum_m w_m \left( |\chi_{\ell m}|^2 + |\psi_{\ell m}|^2 \right) / (4 \pi),

    with :math:`w_0 = 1` and :math:`w_m = 2` for :math:`m > 0`, as the real transform keeps the orders
    :math:`m \geq 0` only: the sum over all degrees is the mean of :math:`|u|^2 / 2` over the sphere (Parseval).
    The components enter the transform as ``(u, v)`` in place of the ``(u_theta, u_phi)`` of torch-harmonics, a
    rotation of the wind by 90 degrees at every point, which swaps the divergent and rotational parts but keeps the
    energy of every degree.

    For every sample and member, the energy of the forecast summed over a band is divided by that of the
    observation; the ratio is averaged over the members, and over the samples as :class:`GeometricSSR` averages its
    ratio, with the counts of the base class. A ratio of 1 means the forecast carries the energy of the
    observation at those scales; below 1 it is smoother, above 1 it holds more (noise, or a mode that grows in the
    rollout). The fields are denormalized with ``bias`` and ``scale`` first, so that the wind is in physical units
    and the energy of a pair does not depend on how its components were normalized.

    A spectrum is a property of the whole field, so with weights, which mark missing values with zeros, a pair
    counts for a sample only where both of its components are valid at every point of the grid; otherwise it is
    left out of the ratio and of the counts. Unlike the area weighting of the SSR, a wind field with holes, or one
    whose v component is missing and filled with zeros, is not scored at all.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2
        The grid of the fields.
    bands : Sequence[int], optional
        Edges of the degree bands, see :func:`kinetic_energy_bands`.
    lmax : int, optional
        Exclusive degree of the transform, by default :func:`resolved_degree` of the grid.
    bias, scale : torch.Tensor, optional
        Per-channel bias and scale of the normalization, of shape ``(num_channels,)``.
    eps : float, optional
        Lower bound of the energy of the observation in a band, by default ``1e-12``.
    """

    def __init__(
        self,
        grid,
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = True,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        bands: Optional[Sequence[int]] = None,
        lmax: Optional[int] = None,
        bias: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        eps: Optional[float] = 1e-12,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid=grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

        # a spectrum is a property of the whole sphere
        if (crop_shape is not None) and (tuple(crop_shape) != tuple(grid.shape) or tuple(crop_offset) != (0, 0)):
            raise ValueError(
                f"The kinetic energy spectrum needs the whole grid {tuple(grid.shape)}, not a crop {crop_shape}"
            )

        self.ensemble_distributed = (
            ensemble_distributed and comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)
        )
        self.eps = eps

        resolved = resolved_degree(grid)
        lmax = resolved if lmax is None else min(lmax, resolved)
        self.bands = kinetic_energy_bands(lmax, bands)

        # vector SHT, and the degrees and orders of the local block of coefficients
        if self.spatial_distributed and (comm.get_size("spatial") > 1):
            if not thd.is_initialized():
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
            self.vsht = thd.DistributedRealVectorSHT(grid, lmax=lmax, mmax=lmax)
            l_offset = sum(self.vsht.l_shapes[: comm.get_rank("h")])
            l_local = self.vsht.l_shapes[comm.get_rank("h")]
            m_offset = sum(self.vsht.m_shapes[: comm.get_rank("w")])
            m_local = self.vsht.m_shapes[comm.get_rank("w")]
        else:
            self.vsht = th.RealVectorSHT(grid, lmax=lmax, mmax=lmax).float()
            l_offset, l_local, m_offset, m_local = 0, lmax, 0, lmax
        self.reduce_degrees = self.spatial_distributed and (comm.get_size("h") > 1)
        self.reduce_orders = self.spatial_distributed and (comm.get_size("w") > 1)

        # energy per coefficient power: l (l + 1) / 2, the multiplicity of the order, and 1 / (4 pi) for the mean
        ell = torch.arange(l_offset, l_offset + l_local, dtype=torch.float64)
        m = torch.arange(m_offset, m_offset + m_local, dtype=torch.float64)
        multiplicity = torch.where(m == 0, 1.0, 2.0)
        lm_weights = 0.5 * (ell * (ell + 1.0)).reshape(-1, 1) * multiplicity.reshape(1, -1) / (4.0 * math.pi)
        self.register_buffer("lm_weights", lm_weights.to(torch.float32), persistent=False)

        # band membership of the local degrees, (nbands, l_local)
        edges = torch.tensor(self.bands, dtype=torch.float64)
        band_matrix = (ell.reshape(1, -1) >= edges[:-1].reshape(-1, 1)) & (
            ell.reshape(1, -1) < edges[1:].reshape(-1, 1)
        )
        self.register_buffer("band_matrix", band_matrix.to(torch.float32), persistent=False)

        # denormalization
        self.register_buffer(
            "bias", None if bias is None else bias.reshape(-1, 1, 1).to(torch.float32), persistent=False
        )
        self.register_buffer(
            "scale", None if scale is None else scale.reshape(-1, 1, 1).to(torch.float32), persistent=False
        )

    @property
    def type(self):
        return LossType.Probabilistic

    @property
    def num_bands(self) -> int:
        return len(self.bands) - 1

    @property
    def band_labels(self) -> Tuple[str, ...]:
        return tuple(f"[{a},{b})" for a, b in zip(self.bands[:-1], self.bands[1:]))

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # one count per pair, and a singleton for the bands
        num_pairs = inp.shape[2] // 2
        if weight is not None:
            if self.batch_reduction != "sum":
                raise ValueError(
                    "Batch reduction mode 'mean' is not supported when weights are provided. Use 'sum' instead."
                )
            counts = torch.sum(self.complete_pairs(weight).to(dtype=inp.dtype), dim=0)
        elif self.batch_reduction == "mean":
            counts = torch.ones(size=(num_pairs,), device=inp.device, dtype=inp.dtype)
        else:
            counts = torch.full(size=(num_pairs,), fill_value=inp.shape[0], device=inp.device, dtype=inp.dtype)

        if self.channel_reduction == "mean":
            counts = torch.mean(counts, dim=0)
        elif self.channel_reduction == "sum":
            counts = torch.sum(counts, dim=0)

        return counts.unsqueeze(-1)

    def complete_pairs(self, weight: torch.Tensor) -> torch.Tensor:
        r"""
        Whether both components of a pair are valid (nonzero weight) at every point of the grid, for the weights
        ``(B, 2 * num_pairs, H, W)`` of the pairs; returns ``(B, num_pairs)`` booleans.
        """
        invalid = (weight[:, 0::2] <= 0) | (weight[:, 1::2] <= 0)
        num_invalid = torch.sum(invalid.to(torch.float32), dim=(-2, -1))
        if self.spatial_distributed and (comm.get_size("spatial") > 1):
            num_invalid = reduce_from_parallel_region(num_invalid.contiguous(), "spatial")
        return num_invalid == 0

    def band_energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Kinetic energy per unit mass in each band, of the wind pairs ``x`` of shape ``(..., 2 * num_pairs, H, W)``
        in normalized units; returns ``(..., num_pairs, num_bands)``.
        """
        x = x.to(torch.float32)
        if self.scale is not None:
            x = x * self.scale
        if self.bias is not None:
            x = x + self.bias
        x = x.reshape(*x.shape[:-3], x.shape[-3] // 2, 2, *x.shape[-2:])
        coeffs = self.vsht(x)
        power = torch.sum(coeffs.real.square() + coeffs.imag.square(), dim=-3)
        energy = torch.sum(power * self.lm_weights, dim=-1)
        if self.reduce_orders:
            energy = reduce_from_parallel_region(energy, "w")
        energy = torch.matmul(energy, self.band_matrix.t())
        if self.reduce_degrees:
            energy = reduce_from_parallel_region(energy, "h")
        return energy

    def forward(
        self, forecasts: torch.Tensor, observations: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")
        if forecasts.shape[2] % 2 != 0:
            raise ValueError(f"The channels have to be pairs of wind components (u, v), found {forecasts.shape[2]}")

        ens_size = forecasts.shape[1] * (comm.get_size("ensemble") if self.ensemble_distributed else 1)

        # (B, E, pairs, bands) over (B, 1, pairs, bands), averaged over the members
        energy_observations = self.band_energy(observations).unsqueeze(1)
        energy_forecasts = self.band_energy(forecasts)
        ratio = torch.sum(energy_forecasts / torch.clamp(energy_observations, min=self.eps), dim=1)
        if self.ensemble_distributed:
            ratio = reduce_from_parallel_region(ratio, "ensemble")
        ratio = ratio / float(ens_size)

        if weight is not None:
            # a pair with a missing value in either component is left out, compute_counts does not count it either
            ratio = torch.where(self.complete_pairs(weight).unsqueeze(-1), ratio, 0.0)

        # reduce
        if self.channel_reduction == "mean":
            ratio = torch.mean(ratio, dim=1)
        elif self.channel_reduction == "sum":
            ratio = torch.sum(ratio, dim=1)

        if self.batch_reduction == "mean":
            ratio = torch.mean(ratio, dim=0)
        elif self.batch_reduction == "sum":
            ratio = torch.sum(ratio, dim=0)

        return ratio


class GeometricCRPS(torch.nn.Module):
    # The skill-spread score from member pairs does E (E - 1) / 2 differences of whole fields. Compiled, they fuse
    # into one pass and beat the sort of the ranks many times over, but the traced graph, and the compile time,
    # grow with the number of pairs: about a minute at 64 members. Eager, they are only on par with the ranks at
    # 16 members and fall behind beyond. Above these ensemble sizes the metric uses the ranks.
    PAIRWISE_MAX_ENSEMBLE_COMPILED = 64
    PAIRWISE_MAX_ENSEMBLE_EAGER = 16

    def __init__(
        self,
        grid,
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        crps_type: Optional[str] = "skillspread",
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        ensemble_weights: Optional[torch.Tensor] = None,
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__()

        self.metric_func = CRPSLoss(
            grid=grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=[],
            crps_type=crps_type,
            spatial_distributed=spatial_distributed,
            ensemble_distributed=ensemble_distributed,
            ensemble_weights=ensemble_weights,
        )

        self.channel_reduction = channel_reduction
        self.batch_reduction = batch_reduction

        # the skill-spread score with members of equal weight, which is what the metric handler uses, is
        # computed from member pairs, up to an ensemble size (see uses_pairs); any other variant goes through the
        # loss
        self.pairwise = (crps_type == "skillspread") and (ensemble_weights is None)

        self.quadrature = GridQuadrature(
            grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=True,
            distributed=spatial_distributed,
        )

    @property
    def type(self):
        return self.metric_func.type

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            if self.batch_reduction == "mean":
                raise ValueError(
                    "Batch reduction mode 'mean' is not supported when weights are provided. Use 'sum' instead."
                )
            elif self.batch_reduction == "sum":
                counts = torch.sum(self.quadrature(weight.to(dtype=inp.dtype)), dim=0)
            else:
                raise ValueError(f"Batch reduction mode '{self.batch_reduction}' is not supported")
        else:
            if self.batch_reduction == "mean":
                counts = torch.ones(size=(inp.shape[2],), device=inp.device, dtype=inp.dtype)
            elif self.batch_reduction == "sum":
                counts = torch.full(size=(inp.shape[2],), fill_value=inp.shape[0], device=inp.device, dtype=inp.dtype)

        if self.channel_reduction == "mean":
            counts = torch.mean(counts, dim=0)
        elif self.channel_reduction == "sum":
            counts = torch.sum(counts, dim=0)

        return counts

    def combine(self, vals, counts, dim=0):
        vals, counts = _sanitize_shapes(vals, counts, dim=dim)
        vals_res, counts_res = _welford_reduction_helper(vals, counts, self.batch_reduction, dim=dim)
        return vals_res, counts_res

    def finalize(self, vals, counts):
        if self.batch_reduction == "mean":
            return vals
        else:
            return vals / counts

    def uses_pairs(self, local_ensemble_size: int) -> bool:
        """
        Whether the score is computed from member pairs, for the given number of members on this rank. The bound
        is on the whole ensemble, and higher when the metric is being compiled, which is decided while tracing.
        """
        if not self.pairwise:
            return False
        ensemble_size = local_ensemble_size
        if self.metric_func.ensemble_distributed:
            ensemble_size = ensemble_size * comm.get_size("ensemble")
        if torch.compiler.is_compiling():
            return ensemble_size <= self.PAIRWISE_MAX_ENSEMBLE_COMPILED
        return ensemble_size <= self.PAIRWISE_MAX_ENSEMBLE_EAGER

    def _pairwise_skillspread(
        self, forecasts: torch.Tensor, observations: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        The skill-spread CRPS integrated over the grid, (B, C), as the loss computes it, from member pairs
        instead of ranks:

            sum_k w_k [ mean_i |x_ik - y_k| - (E - 1 + alpha) / (E^2 (E - 1)) sum_{i<j} |x_ik - x_jk| ].

        The ranks need the members sorted at every point, which is slow. The sum over pairs, accumulated
        point by point, keeps a single field in memory; compiled, as the metric handler does it, it becomes
        one pass over the members. The pairwise differences are not differentiable where members tie, which
        is why the loss keeps the ranks; the metric does not need gradients.
        """
        loss = self.metric_func
        B, E, C, H, W = forecasts.shape

        # members first, then the same layout as the loss: under ensemble parallelism, all members on a part
        # of the grid
        forecasts = torch.reshape(torch.moveaxis(forecasts, 1, 0), (E, B, C, H * W))
        observations = torch.reshape(observations, (B, C, H * W))
        if weight is not None:
            weight = torch.flatten(weight, start_dim=-2)
        if loss.ensemble_distributed:
            ensemble_shapes = [E for _ in range(comm.get_size("ensemble"))]
            forecasts = distributed_transpose(forecasts, (-1, 0), ensemble_shapes, "ensemble")
            observations = scatter_to_parallel_region(observations, -1, "ensemble")
            if weight is not None:
                weight = scatter_to_parallel_region(weight, -1, "ensemble")
        num_ensemble = forecasts.shape[0]

        # in float32 at least
        work_dtype = torch.promote_types(forecasts.dtype, torch.float32)
        forecasts = forecasts.to(dtype=work_dtype)
        observations = observations.to(dtype=work_dtype)

        skill = torch.abs(forecasts[0] - observations)
        for i in range(1, num_ensemble):
            skill = skill + torch.abs(forecasts[i] - observations)
        crps = skill / float(num_ensemble)

        if num_ensemble > 1:
            spread = torch.zeros_like(observations)
            for i in range(num_ensemble):
                for j in range(i + 1, num_ensemble):
                    spread = spread + torch.abs(forecasts[i] - forecasts[j])
            crps = crps - spread * (num_ensemble - 1.0 + loss.alpha) / float(
                num_ensemble * num_ensemble * (num_ensemble - 1)
            )

        # integrate over the local grid points
        point_weight = loss.quad_weight_split.to(dtype=work_dtype)
        if weight is not None:
            point_weight = point_weight * weight.to(dtype=work_dtype)
        crps = torch.sum(crps * point_weight, dim=-1)

        # the integrals over the local part of the grid add up over the ranks that split it
        if loss.ensemble_distributed:
            crps = reduce_from_parallel_region(crps, "ensemble")
        if loss.spatial_distributed:
            crps = reduce_from_parallel_region(crps, "spatial")

        return crps

    def forward(
        self, forecasts: torch.Tensor, observations: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.uses_pairs(forecasts.shape[1]):
            crps = self._pairwise_skillspread(forecasts, observations, weight)
        else:
            crps = self.metric_func(forecasts, observations, weight)

        if self.channel_reduction == "mean":
            crps = torch.mean(crps, dim=1)
        elif self.channel_reduction == "sum":
            crps = torch.sum(crps, dim=1)

        if self.batch_reduction == "mean":
            crps = torch.mean(crps, dim=0)
        elif self.batch_reduction == "sum":
            crps = torch.sum(crps, dim=0)

        return crps


class GeometricRankHistogram(GeometricBaseMetric):
    def __init__(
        self,
        grid,
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        ensemble_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid=grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=normalize,
            channel_reduction=channel_reduction,
            batch_reduction=batch_reduction,
            spatial_distributed=spatial_distributed,
        )

        self.ensemble_distributed = (
            comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1) and ensemble_distributed
        )

        # the quadrature weights of the local grid points, flattened to match the forecasts
        quad_weight_flat = self.quadrature.quad_weight.reshape(1, 1, -1).contiguous()
        self.register_buffer("quad_weight_flat", quad_weight_flat, persistent=False)

    @property
    def type(self):
        return LossType.Probabilistic

    def _ensemble_size(self, forecasts: torch.Tensor) -> int:
        """The size of the whole ensemble; every ensemble rank holds the same number of members."""
        ensemble_size = forecasts.shape[1]
        if self.ensemble_distributed:
            ensemble_size = ensemble_size * comm.get_size("ensemble")
        return ensemble_size

    def compute_counts(self, inp: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            if self.batch_reduction == "mean":
                raise ValueError(
                    "Batch reduction mode 'mean' is not supported when weights are provided. Use 'sum' instead."
                )
            elif self.batch_reduction == "sum":
                counts = torch.sum(self.quadrature(weight.to(dtype=inp.dtype)), dim=0)
            else:
                raise ValueError(f"Batch reduction mode '{self.batch_reduction}' is not supported")
        else:
            if self.batch_reduction == "mean":
                counts = torch.ones(size=(inp.shape[2],), device=inp.device, dtype=inp.dtype)
            elif self.batch_reduction == "sum":
                counts = torch.full(size=(inp.shape[2],), fill_value=inp.shape[0], device=inp.device, dtype=inp.dtype)

        if self.channel_reduction == "mean":
            counts = torch.mean(counts, dim=0)
        elif self.channel_reduction == "sum":
            counts = torch.sum(counts, dim=0)

        # we need to add another dim:
        counts = counts.unsqueeze(-1)

        return counts

    def forward(
        self, forecasts: torch.Tensor, observations: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # sanity checks
        if forecasts.dim() != 5:
            raise ValueError(f"Error, forecasts tensor expected to have 5 dimensions but found {forecasts.dim()}.")

        # we assume that weight have NO ensemble dim
        if (weight is not None) and (weight.dim() != observations.dim()):
            spdim = weight.dim()
            odim = observations.dim()
            raise ValueError(
                f"the weights have to have the same number of dimensions (found {spdim}) as observations (found {odim})."
            )

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, lat, lon
        # observations: batch, channels, lat, lon
        B, E, C, H, W = forecasts.shape
        ensemble_size = self._ensemble_size(forecasts)

        # the rank of the observation needs no sorting: it follows from the number of members below it and
        # equal to it. Both add up over the members, so each rank counts its own members, and one sum over
        # the ensemble ranks gives the counts of the whole ensemble. The counts are exact in float32, the
        # dtype of the quadrature weights they are combined with
        wdtype = self.quad_weight_flat.dtype
        below = torch.sum(forecasts < observations.unsqueeze(1), dim=1, dtype=wdtype)
        equal = torch.sum(forecasts == observations.unsqueeze(1), dim=1, dtype=wdtype)

        # a NaN member makes the point NaN, and carries through the sum over the ensemble ranks
        nan_members = torch.any(torch.isnan(forecasts), dim=1)

        if self.ensemble_distributed:
            # pack both counts into one value for a single all-reduce; below + base * equal stays exact in
            # float32 as long as base**2 does not exceed 2**24
            base = ensemble_size + 1
            if base * base > 2**24:
                raise ValueError(f"The rank histogram supports ensembles of up to 4095 members, got {ensemble_size}.")
            packed = torch.where(nan_members, torch.nan, below + base * equal)
            packed = reduce_from_parallel_region(packed, "ensemble")
            nan_members = torch.isnan(packed)
            # the packed counts are whole numbers, so the integer division recovers them exactly
            packed = torch.where(nan_members, 0.0, packed).to(dtype=torch.int64)
            equal = torch.div(packed, base, rounding_mode="floor")
            below = (packed - base * equal).to(dtype=wdtype)
            equal = equal.to(dtype=wdtype)

        # the weight of each point, flattened over the grid. Missing observations are masked through the weight
        # by the metric handler; one that reaches the histogram, or a NaN member, makes it NaN, as with the
        # other metrics, rather than landing in a bin
        point_weight = self.quad_weight_flat.expand(B, C, -1)
        if weight is not None:
            point_weight = point_weight * torch.flatten(weight, start_dim=-2).to(dtype=wdtype)
        nan_points = torch.logical_or(torch.isnan(observations), nan_members)
        point_weight = torch.where(torch.flatten(nan_points, start_dim=-2), torch.nan, point_weight)

        # an observation tied with members takes each of the ranks from below to below + equal with the
        # same probability, as xskillscore's random_for_tied; the histogram takes that expectation, which
        # spreads the weight of the point evenly over those ranks
        lowest = torch.flatten(below, start_dim=-2)
        highest = lowest + torch.flatten(equal, start_dim=-2)
        share = point_weight / (torch.flatten(equal, start_dim=-2) + 1).to(dtype=wdtype)

        # one bin at a time, rather than a one-hot of (E + 1) times the size of the fields. Outside its ranks a
        # point adds zero times its share, which carries a NaN into every bin
        outside = 0.0 * share
        bins = []
        for rank in range(ensemble_size + 1):
            in_bin = torch.logical_and(lowest <= rank, rank <= highest)
            bins.append(torch.sum(torch.where(in_bin, share, outside), dim=-1))
        rankhist = torch.stack(bins, dim=-1)

        # we need to do the spatial averaging manually since
        # we are not calling he quadrature forward function
        if self.spatial_distributed:
            rankhist = reduce_from_parallel_region(rankhist, "spatial")

        # reduction over channels
        if self.channel_reduction == "mean":
            rankhist = torch.mean(rankhist, dim=1)
        elif self.channel_reduction == "sum":
            rankhist = torch.sum(rankhist, dim=1)

        if self.batch_reduction == "mean":
            rankhist = torch.mean(rankhist, dim=0)
        elif self.batch_reduction == "sum":
            rankhist = torch.sum(rankhist, dim=0)

        return rankhist


class SimpsonQuadrature(torch.nn.Module):
    def __init__(self, num_intervals, interval_width, device):
        super(SimpsonQuadrature, self).__init__()

        # set up integration weights
        weights = [0.0 for _ in range(num_intervals + 1)]
        if num_intervals % 2 == 0:
            # Simpsons 1/3
            for j in range(1, (num_intervals // 2 + 1)):
                weights[2 * j - 2] += 1.0
                weights[2 * j - 1] += 4.0
                weights[2 * j] += 1.0
            self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
            self.weights *= interval_width / 3.0
        else:
            raise NotImplementedError("Error, please specify an even number of intervals")

    def forward(self, x, dim=1):
        # reshape weights to handle channels
        shape = [1 for _ in range(x.dim())]
        shape[dim] = -1
        weights = torch.reshape(self.weights, shape)

        return torch.sum(x * weights, dim=dim)


class TrapezoidQuadrature(torch.nn.Module):
    def __init__(self, num_intervals, interval_width, device):
        super(TrapezoidQuadrature, self).__init__()

        # set up integration weights
        weights = [interval_width for _ in range(num_intervals + 1)]
        weights[0] *= 0.5
        weights[-1] *= 0.5
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def forward(self, x, dim=1):
        shape = [1 for _ in range(x.dim())]
        shape[dim] = -1
        weights = torch.reshape(self.weights, shape)

        return torch.sum(x * weights, dim=dim)


class Quadrature(torch.nn.Module):
    def __init__(self, num_intervals, interval_width, device):
        super(Quadrature, self).__init__()
        if num_intervals % 2 == 0:
            self.quad = SimpsonQuadrature(num_intervals, interval_width, device)
        else:
            self.quad = TrapezoidQuadrature(num_intervals, interval_width, device)

    def forward(self, x, dim=1):
        return self.quad(x, dim)
