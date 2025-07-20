# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Tuple, List, Optional

# for spatial model-parallelism
from makani.utils import comm
from physicsnemo.distributed.mappings import gather_from_parallel_region, copy_to_parallel_region

# quadrature stuff
from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature


@torch.compile
def _normalize_transform_kernel(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:

    # normalization
    x = (x - mean) / torch.sqrt(var + eps)

    # affine transformation
    x = weight * x + bias

    return x


@torch.compile
def _normalize_kernel(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, eps: float) -> torch.Tensor:

    # normalization
    x = (x - mean) / torch.sqrt(var + eps)

    return x


@torch.compile
def _welford_kernel(vars: torch.Tensor, means: torch.Tensor, counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # for weighted welford, replace counts by
    # omega = sum_i w_i, where w_i are the individual weights

    # get m2s
    m2s = vars * counts

    # do welford update
    mean = means[0, ...]
    m2 = m2s[0, ...]
    count = counts[0, ...]

    # use Welford's algorithm to accumulate them into a single mean and variance
    for i in range(1, means.shape[0]):
        delta = means[i, ...] - mean
        m2 = m2 + m2s[i, ...] + delta**2 * count * counts[i, ...] / (count + counts[i, ...])
        if i == 1:
            mean = (mean * count + means[i, ...] * counts[i, ...]) / (count + counts[i, ...])
        else:
            mean = mean + delta * counts[i, ...] / (count + counts[i, ...])

        # update the current count
        count = count + counts[i, ...]

    var = m2 / count

    return var, mean, count


def distributed_welford_variance(var: torch.Tensor, mean: torch.Tensor, count: torch.Tensor, group: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the statistics locally, then uses the Welford online algorithm to reduce them"""

    # concatenate:
    # this has the shape [3, 1, ...]
    var_mean_count = torch.stack([var, mean, count], dim=0).unsqueeze(1)

    # gather
    # this has the shape [3, spatial_size, ...], we split it up directly into individual tensors again
    vars_means_counts = gather_from_parallel_region(var_mean_count, dim=1, shapes=None, group=group)

    # split up
    vars = vars_means_counts[0, ...]
    means = vars_means_counts[1, ...]
    counts = vars_means_counts[2, ...]

    # do welford update
    var, mean, count = _welford_kernel(vars, means, counts)

    return var, mean, count


class DistributedInstanceNorm2d(nn.Module):
    """
    Computes a distributed instance norm using Welford's online algorithm
    """

    def __init__(self, num_features, eps=1e-05, affine=False):
        super().__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
            self.weight.is_shared_mp = ["spatial"]
            self.bias.is_shared_mp = ["spatial"]

    def _stats_welford(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the statistics locally, then uses the Welford online algorithm to reduce them"""

        # extract shapes
        B, C, H, W = x.shape

        # those have the shapes [B, C]
        var, mean = torch.var_mean(x, dim=(-2, -1), unbiased=False, keepdim=False)

        # workaround to not use shapes, as otherwise cuda graphs won't work
        # those have the shapes [B, C]
        count = torch.ones_like(x, requires_grad=False)
        count = torch.sum(count, dim=(-2, -1), keepdim=False)
        var, mean, _ = distributed_welford_variance(var, mean, count, "spatial")

        # reshape
        var = var.reshape(B, C, 1, 1)
        mean = mean.reshape(B, C, 1, 1)

        return var, mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        with amp.autocast(device_type="cuda", enabled=False):
            dtype = x.dtype
            x = x.float()

            # start by computing std and mean
            var, mean = self._stats_welford(x)

            # this is absolutely necessary to get the correct graph in the backward pass
            mean = copy_to_parallel_region(mean, "spatial")
            var = copy_to_parallel_region(var, "spatial")

        x = x.to(dtype)
        mean = mean.to(dtype)
        var = var.to(dtype)

        # apply the normalization
        if self.affine:
            x = _normalize_transform_kernel(x, mean, var, self.weight.reshape(-1, 1, 1), self.bias.reshape(-1, 1, 1), self.eps)
        else:
            x = _normalize_kernel(x, mean, var, self.eps)

        return x


class DistributedGeometricInstanceNormS2(DistributedInstanceNorm2d):
    """
    Computes a distributed S2 weighted instance norm using Welford's online algorithm
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        grid_type: str,
        pole_mask: int,
        num_features: int,
        eps: Optional[float] = 1e-05,
        affine: Optional[bool] = False,
    ):
        super().__init__(num_features, eps, affine)

        # set up quadrature rule:
        quadrature_rule = grid_to_quadrature_rule(grid_type)

        # we only need the weights
        quad_weight = GridQuadrature(
            quadrature_rule, img_shape=img_shape, crop_shape=crop_shape, crop_offset=crop_offset, normalize=True, pole_mask=pole_mask, distributed=True
        ).quad_weight

        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def _stats_welford(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the statistics locally, then uses the Welford online algorithm to reduce them"""

        # extract shapes
        B, C, H, W = x.shape

        # compute var, mean locally: those have the shapes [B, C]
        mean = torch.sum(x * self.quad_weight, dim=(-2, -1), keepdim=False)
        var = torch.sum(torch.square(x - mean.reshape(B, C, 1, 1)) * self.quad_weight, dim=(-2, -1), keepdim=False)
        count = torch.tile(torch.sum(self.quad_weight, dim=(-2, -1), keepdim=False), (B, C))

        # compute welford variance
        var, mean, _ = distributed_welford_variance(var, mean, count, "spatial")

        # reshape
        var = var.reshape(B, C, 1, 1)
        mean = mean.reshape(B, C, 1, 1)

        return var, mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        with amp.autocast(device_type="cuda", enabled=False):
            dtype = x.dtype
            x = x.float()

            # start by computing std and mean
            var, mean = self._stats_welford(x)

            # this is absolutely necessary to get the correct graph in the backward pass
            mean = copy_to_parallel_region(mean, "spatial")
            var = copy_to_parallel_region(var, "spatial")

        x = x.to(dtype)
        mean = mean.to(dtype)
        var = var.to(dtype)

        # apply the normalization
        if self.affine:
            x = _normalize_transform_kernel(x, mean, var, self.weight.reshape(-1, 1, 1), self.bias.reshape(-1, 1, 1), self.eps)
        else:
            x = _normalize_kernel(x, mean, var, self.eps)

        return x


class DistributedLayerNorm(nn.Module):
    """
    This is a lightweight wrapper which only computed norm across channels.
    This norm breaks equivariance since the norm across channels is different per grid
    point.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None):
        super().__init__()

        assert comm.get_size("matmul") == 1

        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=dtype)

        if elementwise_affine:
            # set up weight sharing and sharding
            self.norm.weight.is_shared_mp = ["model"]
            self.norm.weight.sharded_dims_mp = [None]
            if bias:
                self.norm.bias.is_shared_mp = ["model"]
                self.norm.bias.sharded_dims_mp = [None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # assume input is NCHW, so we transpose
        xt = torch.transpose(x, 1, 3)
        xn = self.norm(xt)
        x = torch.transpose(xn, 1, 3).contiguous()

        return x
