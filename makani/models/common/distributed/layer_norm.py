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

from typing import Tuple, Optional

# for spatial model-parallelism
from makani.distributed import comm

# quadrature stuff
from makani.utils.grids import GridQuadrature

# the instance norm shared with the serial layers; distributed_welford_variance is re-exported from here
from makani.models.common.layer_norm import instance_norm, distributed_welford_variance  # noqa: F401


class DistributedInstanceNorm2d(nn.Module):
    """
    Instance norm over a grid split across the ``"spatial"`` ranks.

    Each rank computes the statistics of its part of the grid, which a Welford (Chan et al.) combine reduces to the
    global ones. The backward is analytic, see :func:`~makani.models.common.layer_norm.instance_norm`, and keeps only
    the input and two ``(B, C)`` tensors.
    """

    def __init__(self, num_features, eps=1e-05, affine=False):
        super().__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
            self.weight.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
            self.bias.is_shared_mp = {"spatial": "sum", "matmul": "avg"}

    def _quad_weight(self) -> Optional[torch.Tensor]:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            return instance_norm(x, self.weight, self.bias, self._quad_weight(), self.eps, "spatial")
        return instance_norm(x, quad_weight=self._quad_weight(), eps=self.eps, comm_id="spatial")


class DistributedGeometricInstanceNormS2(DistributedInstanceNorm2d):
    """
    Computes a distributed S2 weighted instance norm using Welford's online algorithm.

    Takes the arguments of :class:`~makani.models.common.layer_norm.GeometricInstanceNormS2`; ``grid``
    and ``crop_shape`` are global, each rank normalizes its own part of them.
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
        super().__init__(num_features, eps, affine)

        # we only need the weights
        quad_weight = GridQuadrature(
            grid,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            normalize=True,
            distributed=True,
        ).quad_weight

        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def _quad_weight(self) -> Optional[torch.Tensor]:
        return self.quad_weight


class DistributedLayerNorm(nn.Module):
    """
    This is a lightweight wrapper which only computed norm across channels.
    This norm breaks equivariance since the norm across channels is different per grid
    point.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None):
        super().__init__()

        if comm.get_size("matmul") != 1:
            raise ValueError(
                f"this layer norm does not support matmul parallelism, but got matmul comm size {comm.get_size('matmul')}"
            )

        self.norm = nn.LayerNorm(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=dtype
        )

        if elementwise_affine:
            # set up weight sharing and sharding
            self.norm.weight.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
            self.norm.weight.sharded_dims_mp = [None]
            if bias:
                self.norm.bias.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
                self.norm.bias.sharded_dims_mp = [None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # assume input is NCHW, so we transpose
        xt = torch.transpose(x, 1, 3)
        xn = self.norm(xt)
        x = torch.transpose(xn, 1, 3).contiguous()

        return x
