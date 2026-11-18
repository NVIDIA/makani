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

"""The normalization layer of a network, picked by name, serial or distributed."""

from functools import partial

import torch
import torch.nn as nn

from makani.distributed import comm
from makani.models.common.layer_norm import GeometricInstanceNormS2
from makani.models.common.distributed.layer_norm import (
    DistributedLayerNorm,
    DistributedInstanceNorm2d,
    DistributedGeometricInstanceNormS2,
)


@torch.compiler.disable(recursive=True)
def get_norm_layer_handle(h, w, embed_dim, normalization_layer="none", sht_grid_type="legendre-gauss"):
    r"""
    Constructor of the normalization layer ``normalization_layer`` for ``embed_dim`` channels on an
    ``h`` x ``w`` grid: ``"layer_norm"``, ``"instance_norm"``, ``"instance_norm_s2"`` (with the quadrature
    of ``sht_grid_type``) or ``"none"``. The instance norms are the distributed ones under spatial
    parallelism.
    """
    if normalization_layer == "layer_norm":
        norm_layer_handle = partial(
            DistributedLayerNorm, normalized_shape=(embed_dim), elementwise_affine=True, eps=1e-6
        )
    elif normalization_layer == "instance_norm":
        if comm.get_size("spatial") > 1:
            norm_layer_handle = partial(DistributedInstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True)
        else:
            norm_layer_handle = partial(
                nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False
            )
    elif normalization_layer == "instance_norm_s2":
        norm_layer_handle = (
            DistributedGeometricInstanceNormS2 if comm.get_size("spatial") > 1 else GeometricInstanceNormS2
        )
        norm_layer_handle = partial(
            norm_layer_handle,
            img_shape=(h, w),
            crop_shape=(h, w),
            crop_offset=(0, 0),
            grid_type=sht_grid_type,
            num_features=embed_dim,
            eps=1e-6,
            affine=True,
        )
    elif normalization_layer == "none":
        norm_layer_handle = nn.Identity
    else:
        raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.")

    return norm_layer_handle
