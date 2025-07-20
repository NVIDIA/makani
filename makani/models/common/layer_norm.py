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

from typing import Tuple, List, Optional

# quadrature stuff
from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature

# we need the kernels
from makani.mpu.layer_norm import _normalize_kernel, _normalize_transform_kernel


# instance norm with S2 weights
class GeometricInstanceNormS2(nn.Module):
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
        super().__init__()

        # set up weights
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        # set up quadrature rule:
        quadrature_rule = grid_to_quadrature_rule(grid_type)

        # we only need the weights
        self.quadrature = GridQuadrature(
            quadrature_rule, img_shape=img_shape, crop_shape=crop_shape, crop_offset=crop_offset, normalize=True, pole_mask=pole_mask, distributed=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # extract shapes
        B, C, H, W = x.shape

        with amp.autocast(device_type="cuda", enabled=False):
            dtype = x.dtype
            x = x.float()

            # compute var and mean
            mean = self.quadrature(x)
            var = self.quadrature(torch.square(x - mean.reshape(B, C, 1, 1)))

        # reshape
        var = var.reshape(B, C, 1, 1)
        mean = mean.reshape(B, C, 1, 1)

        # convert types
        x = x.to(dtype)
        mean = mean.to(dtype)
        var = var.to(dtype)

        # apply the normalization
        if self.affine:
            x = _normalize_transform_kernel(x, mean, var, self.weight.reshape(-1, 1, 1), self.bias.reshape(-1, 1, 1), self.eps)
        else:
            x = _normalize_kernel(x, mean, var, self.eps)

        return x
