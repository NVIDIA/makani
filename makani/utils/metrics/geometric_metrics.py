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

from typing import Optional, Tuple

import torch

from makani.utils import comm
from makani.mpu.mappings import scatter_to_parallel_region, reduce_from_parallel_region, distributed_transpose
from torch_harmonics.distributed import split_tensor_along_dim

from makani.utils.losses import CRPSLoss, LossType
from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature
from makani.utils.metrics.base_metric import _sanitize_shapes, _welford_reduction_helper, GeometricBaseMetric


class GeometricL1(GeometricBaseMetric):
    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid_type=grid_type,
            img_shape=img_shape,
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
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            grid_type=grid_type,
            img_shape=img_shape,
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
        grid_type: str,
        img_shape: Tuple[int, int],
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
            grid_type=grid_type,
            img_shape=img_shape,
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
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = False,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
        **kwargs,
    ):

        super().__init__(
            grid_type=grid_type,
            img_shape=img_shape,
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
        grid_type: str,
        img_shape: Tuple[int, int],
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
            grid_type=grid_type,
            img_shape=img_shape,
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


class GeometricCRPS(torch.nn.Module):
    # The skill-spread score from member pairs does E (E - 1) / 2 differences of whole fields. Compiled, they fuse
    # into one pass and beat the sort of the ranks many times over, but the traced graph, and the compile time,
    # grow with the number of pairs: about a minute at 64 members. Eager, they are only on par with the ranks at
    # 16 members and fall behind beyond. Above these ensemble sizes the metric uses the ranks.
    PAIRWISE_MAX_ENSEMBLE_COMPILED = 64
    PAIRWISE_MAX_ENSEMBLE_EAGER = 16

    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
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
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            channel_names=[],
            grid_type=grid_type,
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

        # get the quadrature
        quadrature_rule = grid_to_quadrature_rule(grid_type)
        self.quadrature = GridQuadrature(
            quadrature_rule,
            img_shape=img_shape,
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
        grid_type: str,
        img_shape: Tuple[int, int],
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
            grid_type=grid_type,
            img_shape=img_shape,
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
