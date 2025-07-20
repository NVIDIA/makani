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
from dataclasses import dataclass

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

import torch_harmonics as th
import torch_harmonics.distributed as thd

from makani.utils.losses.base_loss import LossType
from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature
from makani.utils import comm


def _sanitize_shapes(vals, counts, dim):
    """
    Helper routine to ensure that counts is correctly broadcasted to vals.
    """
    if counts.dim() != 1:
        raise ValueError("The shape of counts has to be exactly 1")
    cshape = [1 for _ in range(vals.dim())]
    cshape[dim] = -1
    counts = counts.reshape(cshape)
    return vals, counts
    

def _welford_reduction_helper(vals, counts, batch_reduction, dim):
    counts_res = torch.sum(counts, dim=dim)
    if batch_reduction == "mean":
        # results were mean reduced
        vals_res = torch.sum(vals * counts, dim=dim) / counts_res
    elif batch_reduction == "sum":
        # results were sum reduced
        vals_res = torch.sum(vals, dim=dim)
    else:
        # results were not reduced
        vals_res = vals
        counts_res = counts

    return vals_res, counts_res


# geometric base loss class
class GeometricBaseMetric(nn.Module, metaclass=ABCMeta):
    """
    Geometric base loss class used by all geometric losses
    """

    def __init__(
        self,
        grid_type: str,
        img_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        crop_offset: Optional[Tuple[int, int]] = (0, 0),
        normalize: Optional[bool] = True,
        channel_reduction: Optional[str] = "mean",
        batch_reduction: Optional[str] = "mean",
        spatial_distributed: Optional[bool] = False,
    ):
        super().__init__()

        self.img_shape = img_shape
        self.crop_shape = crop_shape
        self.crop_offset = crop_offset
        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed
        self.channel_reduction = channel_reduction
        self.batch_reduction = batch_reduction

        quadrature_rule = grid_to_quadrature_rule(grid_type)

        # get the quadrature
        self.quadrature = GridQuadrature(
            quadrature_rule,
            img_shape=self.img_shape,
            crop_shape=self.crop_shape,
            crop_offset=self.crop_offset,
            normalize=normalize,
            distributed=self.spatial_distributed,
        )

    @property
    def type(self):
        return LossType.Deterministic

    def combine(self, vals: torch.Tensor, counts: torch.Tensor, dim: Optional[int]=0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines how to combine multiple metrics result using Welford.

        Parameters
        ----------
        vals : torch.Tensor
            Tensor containing metric values, with different measurements stored along dimension dim.

        counts : torch.Tensor
            One-dimensional tensor containing counts, from which vals where computed. The size of counts has
            to match the size of vals in dimension dim.

        dim : int
            Which dimension of vals to aggregate over.

        Returns
        -------
        vals_res : torch.Tensor
            Tensor containing aggretated values. Same shape as vals except for aggregation dimension dim.

        counts_res : torch.Tensor
            Singleton tensor containing the updated count.
        """
        vals, counts = _sanitize_shapes(vals, counts, dim=dim)
        vals_res, counts_res = _welford_reduction_helper(vals, counts, self.batch_reduction, dim=dim)
        counts_res = counts_res.squeeze()
        return vals_res, counts_res

    def finalize(self, vals: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Defines how to compute a final average from vals and counts.

        Parameters
        ----------
        vals : torch.Tensor
            Tensor containing values.
        counts : torch.Tensor
            Singleton tensor containing counts.

        Returns
        -------
        vals_res : torch.Tensor 
            Values with correct averaging over counts.
        """
        if self.batch_reduction == "mean":
            return vals
        else:
            return vals / counts

    @abstractmethod
    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

