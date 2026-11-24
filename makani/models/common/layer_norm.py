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

import torch
import torch.nn as nn

from torch import amp
from torch.autograd.function import once_differentiable

from typing import Tuple, Optional

# quadrature stuff
from makani.utils.grids import GridQuadrature


# for spatial model-parallelism
from makani.distributed import comm
from makani.distributed.mappings import gather_from_parallel_region, reduce_from_parallel_region


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """fp32 for half precision inputs, the dtype of ``x`` otherwise."""
    return torch.promote_types(x.dtype, torch.float32)


@torch.compile
def _local_stats_kernel(x: torch.Tensor, weight: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and biased variance of ``x`` over its last two dims, weighted by ``weight`` or uniform if it is ``None``.

    Casts to fp32 (fp64 for fp64 input) inside the kernel, so the fp32 copy of ``x`` is never materialized. Returns
    ``(mean, var)`` of shape ``(B, C)``, normalized by the local sum of the weights.
    """
    xf = x.to(_compute_dtype(x))
    if weight is None:
        var, mean = torch.var_mean(xf, dim=(-2, -1), unbiased=False, keepdim=False)
    else:
        w = weight.to(xf.dtype) / torch.sum(weight.to(xf.dtype))
        mean = torch.sum(xf * w, dim=(-2, -1), keepdim=False)
        var = torch.sum(torch.square(xf - mean[..., None, None]) * w, dim=(-2, -1), keepdim=False)
    return mean, var


@torch.compile
def _welford_kernel(
    vars: torch.Tensor, means: torch.Tensor, counts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combines the per-rank statistics ``[P, ...]`` with weights ``counts`` of shape ``[P]``, after Chan et al.

    ``N = sum_k n_k``, ``mean = sum_k n_k mean_k / N`` and ``M2 = sum_k n_k (var_k + (mean_k - mean)^2)``. The counts
    may be sums of quadrature weights instead of numbers of grid points.
    """
    counts = counts.reshape(-1, *([1] * (means.dim() - 1)))
    count = torch.sum(counts, dim=0)
    mean = torch.sum(counts * means, dim=0) / count
    m2 = torch.sum(counts * (vars + torch.square(means - mean)), dim=0)
    var = m2 / count

    return var, mean, count.reshape(())


def distributed_welford_variance(
    var: torch.Tensor, mean: torch.Tensor, count: torch.Tensor, comm_id: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduces local statistics over the ranks of ``comm_id``.

    ``var`` and ``mean`` hold the local statistics, ``count`` is the local number of points (or sum of weights), a
    tensor with one element. Not differentiable. Returns the global ``var`` and ``mean`` and the global count as a
    0-d tensor.
    """
    shape = mean.shape
    numel = mean.numel()
    count = count.reshape(1).to(dtype=mean.dtype, device=mean.device)

    # one buffer [1, 2 * numel + 1] per rank: the stats and the count
    buf = torch.cat([var.reshape(-1), mean.reshape(-1), count], dim=0).unsqueeze(0).contiguous()

    # every rank contributes exactly one row. Passing the shapes keeps the gather from all-gathering them first and
    # reading them back with one blocking .item() per rank.
    with torch.no_grad():
        bufs = gather_from_parallel_region(buf, dim=0, shapes=[1] * comm.get_size(comm_id), comm_id=comm_id)

    P = bufs.shape[0]
    vars = bufs[:, :numel].reshape(P, *shape)
    means = bufs[:, numel : 2 * numel].reshape(P, *shape)
    counts = bufs[:, 2 * numel]

    return _welford_kernel(vars, means, counts)


@torch.compile
def _normalize_fwd_kernel(
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Normalizes ``x`` in fp32, applies the affine map if given and casts back to the dtype of ``x``, in one kernel."""
    y = (x.to(mean.dtype) - mean[..., None, None]) * rstd[..., None, None]
    if weight is not None:
        y = y * weight.to(mean.dtype)[:, None, None] + bias.to(mean.dtype)[:, None, None]
    return y.to(x.dtype)


@torch.compile
def _normalize_bwd_reduce_kernel(
    g: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Local partial sums ``(sum_ij g, sum_ij g * xhat)`` of shape ``(B, C)`` in the dtype of ``mean``."""
    gf = g.to(mean.dtype)
    xhat = (x.to(mean.dtype) - mean[..., None, None]) * rstd[..., None, None]
    return torch.sum(gf, dim=(-2, -1)), torch.sum(gf * xhat, dim=(-2, -1))


@torch.compile
def _normalize_bwd_dx_kernel(
    g: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: Optional[torch.Tensor],
    sums: torch.Tensor,
    quad_weight: Optional[torch.Tensor],
    count: torch.Tensor,
) -> torch.Tensor:
    """``dx = rstd * (ghat - q * (S1 + xhat * S2))`` in fp32, cast to the dtype of ``x`` in the same kernel.

    ``q`` is ``quad_weight / count``, or ``1 / count`` for uniform weights, summing to one over the global grid.
    """
    if quad_weight is None:
        q = 1.0 / count
    else:
        q = quad_weight.to(mean.dtype) / count
    gf = g.to(mean.dtype)
    xhat = (x.to(mean.dtype) - mean[..., None, None]) * rstd[..., None, None]
    s1 = sums[0]
    s2 = sums[1]
    if weight is not None:
        gamma = weight.to(mean.dtype)
        gf = gf * gamma[:, None, None]
        s1 = s1 * gamma
        s2 = s2 * gamma
    dx = rstd[..., None, None] * (gf - q * (s1[..., None, None] + xhat * s2[..., None, None]))
    return dx.to(x.dtype)


class _InstanceNormFunction(torch.autograd.Function):
    r"""
    Weighted instance norm with an analytic backward, shared by the serial and the distributed norms.

    With weights :math:`q_{ij}` that sum to one over the global grid, :math:`\mu = \sum_{ij} q_{ij} x_{ij}`,
    :math:`\sigma^2 = \sum_{ij} q_{ij} (x_{ij} - \mu)^2`, :math:`r = (\sigma^2 + \varepsilon)^{-1/2}` and
    :math:`\hat{x} = (x - \mu) r`, the output is :math:`y = \gamma \hat{x} + \beta`. Plain instance norm is the case
    :math:`q = 1/N`. With :math:`\hat{g} = \gamma g`, the backward is

    .. math::

        dx = r \left(\hat{g} - q (S_1 + \hat{x} S_2)\right), \quad
        S_1 = \gamma \sum_{ij} g, \quad S_2 = \gamma \sum_{ij} g \hat{x},

    with the sums taken over the global grid, so one allreduce of a ``[2, B, C]`` tensor under spatial parallelism.
    :math:`d\gamma = \sum_{b,ij} g \hat{x}` and :math:`d\beta = \sum_{b,ij} g` are the local partial sums; the
    gradient reduction hooks of the shared parameters sum them over the ranks.

    Only ``x`` in its own dtype, the ``(B, C)`` fp32 ``mean`` and ``rstd`` and the global count are saved for the
    backward, besides the parameters and the quadrature weights. The backward treats the saved statistics as
    constants, so it is not differentiable itself: double backward raises.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, quad_weight, eps, comm_id):
        B, C, H, W = x.shape
        distributed = (comm_id is not None) and (comm.get_size(comm_id) > 1)

        # weights over the full local grid, so that their sum counts every point; a view, not a copy
        if quad_weight is not None:
            quad_weight = quad_weight.expand(1, 1, H, W)

        with amp.autocast(device_type=x.device.type, enabled=False):
            # local statistics
            mean, var = _local_stats_kernel(x, quad_weight)

            # local count: the number of points or the sum of the weights
            if quad_weight is None:
                count = torch.full((), float(H * W), dtype=mean.dtype, device=x.device)
            else:
                count = torch.sum(quad_weight.to(mean.dtype))

            # global statistics
            if distributed:
                var, mean, count = distributed_welford_variance(var, mean, count, comm_id)

            rstd = torch.rsqrt(var + eps)
            y = _normalize_fwd_kernel(x, mean, rstd, weight, bias)

        # weight and quad_weight are a parameter and a buffer, saving them takes no memory
        ctx.save_for_backward(x, mean, rstd, weight, quad_weight, count)
        ctx.distributed = distributed
        ctx.comm_id = comm_id

        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        x, mean, rstd, weight, quad_weight, count = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        with amp.autocast(device_type=x.device.type, enabled=False):
            sum_g, sum_gxhat = _normalize_bwd_reduce_kernel(g, x, mean, rstd)

            # local partial sums, reduced over the spatial ranks by the gradient hooks of the shared parameters
            if ctx.needs_input_grad[1]:
                grad_weight = torch.sum(sum_gxhat, dim=0).to(weight.dtype)
            if ctx.needs_input_grad[2]:
                grad_bias = torch.sum(sum_g, dim=0).to(weight.dtype)

            if ctx.needs_input_grad[0]:
                sums = torch.stack([sum_g, sum_gxhat], dim=0)
                if ctx.distributed:
                    sums = reduce_from_parallel_region(sums, ctx.comm_id)
                grad_x = _normalize_bwd_dx_kernel(g, x, mean, rstd, weight, sums, quad_weight, count)

        return grad_x, grad_weight, grad_bias, None, None, None


def instance_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    quad_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
    comm_id: Optional[str] = None,
) -> torch.Tensor:
    r"""
    Instance norm of ``x`` of shape ``(B, C, H, W)`` over its last two dims, optionally weighted and distributed.

    Statistics are taken in fp32 and the output has the dtype of ``x``; only ``x`` and two ``(B, C)`` fp32 tensors are
    saved for the backward.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape ``(B, C, H, W)``, this rank's part of the grid under spatial parallelism.
    weight, bias : torch.Tensor, optional
        Per-channel affine parameters of shape ``(C,)``, both or neither.
    quad_weight : torch.Tensor, optional
        Weights of the local grid points, broadcastable to ``(H, W)``. They are normalized by their sum over the
        global grid. Uniform if ``None``.
    eps : float, optional
        Added to the variance.
    comm_id : str, optional
        The group the grid is split over, ``None`` if it is not.
    """
    if (weight is None) != (bias is None):
        raise ValueError("instance_norm takes both weight and bias or neither")
    return _InstanceNormFunction.apply(x, weight, bias, quad_weight, eps, comm_id)


class GeometricInstanceNormS2(nn.Module):
    r"""
    Instance normalization with quadrature weights on the sphere :math:`S^2`.

    Ordinary instance norm averages every grid point equally, which on a
    lat-lon grid over-weights the poles: cells there cover far less area than
    cells at the equator, so a uniform mean is not the mean of the underlying
    field. This module instead uses the quadrature weights :math:`q_{ij}` of the
    grid, so that the statistics approximate true spherical integrals,

    .. math::

        \mu_{bc} = \sum_{ij} q_{ij}\, x_{bcij},
        \qquad
        \sigma^2_{bc} = \sum_{ij} q_{ij}\, (x_{bcij} - \mu_{bc})^2

    with the weights rescaled so that :math:`\sum_{ij} q_{ij} = 1` over the
    crop. Mean and variance are taken per sample
    and per channel, then applied as
    :math:`(x - \mu)/\sqrt{\sigma^2 + \varepsilon}`, optionally followed by a
    learned per-channel affine map.

    Statistics are accumulated in fp32 regardless of the surrounding autocast
    context and the result is cast back to the input dtype, matching how
    PyTorch's native norm layers behave under mixed precision. The backward is
    analytic, see :func:`instance_norm`, and keeps only the input and two
    ``(B, C)`` tensors.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2 or makani.utils.grids.PlanarGrid
        The full grid the input lives on, whose quadrature weights the statistics take, see
        :func:`~makani.utils.grids.make_grid`.
    num_features : int
        Number of channels. Only used to size the affine parameters.
    crop_shape : (int, int), optional
        Extent of the sub-region actually normalized over, the whole grid by default.
    crop_offset : (int, int), optional
        Offset of that sub-region within the full grid, by default ``(0, 0)``.
    eps : float, optional
        Constant added to the variance for numerical stability, by default ``1e-05``.
    affine : bool, optional
        If ``True``, apply a learned per-channel scale and shift after
        normalizing, by default ``False``.

    Notes
    -----
    The quadrature is constructed with ``distributed=False``, so the statistics
    are computed over each rank's local shard. This layer is intended for the
    non-spatially-decomposed case; under spatial model parallelism use
    :class:`~makani.models.common.distributed.layer_norm.DistributedGeometricInstanceNormS2`.
    """

    def __init__(
        self,
        grid,
        num_features: int,
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Tuple[int, int] = (0, 0),
        eps: Optional[float] = 1e-05,
        affine: Optional[bool] = False,
    ):
        super().__init__()

        # set up weights
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        # we only need the weights
        self.quadrature = GridQuadrature(
            grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=True,
            distributed=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Normalize each sample and channel by its quadrature-weighted statistics.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, C, H, W)``, where ``(H, W)`` matches the
            ``crop_shape`` the quadrature was constructed for.

        Returns
        -------
        torch.Tensor
            Normalized tensor of shape ``(B, C, H, W)``, cast back to the dtype
            of ``x``.
        """

        if self.affine:
            return instance_norm(x, self.weight, self.bias, self.quadrature.quad_weight, self.eps)
        return instance_norm(x, quad_weight=self.quadrature.quad_weight, eps=self.eps)
