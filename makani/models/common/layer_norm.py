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

from typing import Tuple, Optional

# quadrature stuff
from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature

# we need the kernels
from makani.mpu.layer_norm import _normalize_kernel, _normalize_transform_kernel


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

    with :math:`\sum_{ij} q_{ij} = 1`. Mean and variance are taken per sample
    and per channel, then applied as
    :math:`(x - \mu)/\sqrt{\sigma^2 + \varepsilon}`, optionally followed by a
    learned per-channel affine map.

    Statistics are accumulated in fp32 regardless of the surrounding autocast
    context and the result is cast back to the input dtype, matching how
    PyTorch's native norm layers behave under mixed precision.

    Parameters
    ----------
    img_shape : (int, int)
        Latitude and longitude extent of the full grid the quadrature rule is
        built for.
    crop_shape : (int, int)
        Extent of the sub-region actually normalized over.
    crop_offset : (int, int)
        Offset of that sub-region within the full grid.
    grid_type : str
        Grid the input lives on (e.g. ``"equiangular"``, ``"legendre-gauss"``);
        determines the quadrature rule via
        :func:`~makani.utils.grids.grid_to_quadrature_rule`.
    num_features : int
        Number of channels. Only used to size the affine parameters.
    eps : float, optional
        Constant added to the variance for numerical stability, by default ``1e-05``.
    affine : bool, optional
        If ``True``, apply a learned per-channel scale and shift after
        normalizing, by default ``False``.

    Notes
    -----
    The quadrature is constructed with ``distributed=False``, so the statistics
    are computed over each rank's local shard. This layer is intended for the
    non-spatially-decomposed case; under spatial model parallelism the reduction
    would need to span the ``"spatial"`` group.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        grid_type: str,
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
            quadrature_rule,
            img_shape=img_shape,
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

        # extract shapes
        B, C, H, W = x.shape

        xtype = x.dtype
        with amp.autocast(device_type=x.device.type, enabled=False):
            xf = x.to(torch.float32)

            # compute var and mean
            mean = self.quadrature(xf)
            var = self.quadrature(torch.square(xf - mean.reshape(B, C, 1, 1)))

            # reshape
            var = var.reshape(B, C, 1, 1)
            mean = mean.reshape(B, C, 1, 1)

            # normalize (and affine) in fp32 for numerical stability, matching the
            # behaviour of PyTorch's native (autocast-fp32) norm ops
            if self.affine:
                xf = _normalize_transform_kernel(
                    xf, mean, var, self.weight.reshape(-1, 1, 1), self.bias.reshape(-1, 1, 1), self.eps
                )
            else:
                xf = _normalize_kernel(xf, mean, var, self.eps)

        # cast back to the input dtype so the layer is faithful to its input
        x = xf.to(xtype)

        return x
