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

import numpy as np
import torch

from torch_harmonics.quadrature import legendre_gauss_weights, clenshaw_curtiss_weights

from makani.utils import comm
from physicsnemo.distributed.utils import compute_split_shapes
from physicsnemo.distributed.mappings import reduce_from_parallel_region


def grid_to_quadrature_rule(grid_type):

    grid_to_quad_dict = {"euclidean" : "uniform", "equiangular" : "naive", "legendre-gauss" : "legendre-gauss", "clenshaw-curtiss" : "clenshaw-curtiss", "weatherbench2" : "weatherbench2"}

    if grid_type not in grid_to_quad_dict.keys():
        raise NotImplementedError(f"Grid type {grid_type} does not have a quadrature rule")
    else:
        return grid_to_quad_dict[grid_type]


class GridConverter(torch.nn.Module):
    def __init__(self, src_grid, dst_grid, lat_rad, lon_rad):
        super(GridConverter, self).__init__()
        self.src = src_grid
        self.dst = dst_grid
        self.src_lat = lat_rad
        self.src_lon = lon_rad

        if self.src != self.dst:
            if self.dst == "legendre-gauss":
                cost_lg, _ = legendre_gauss_weights(lat_rad.shape[0], -1, 1)
                tq = torch.arccos(torch.from_numpy(cost_lg)) - torch.pi / 2.0
                self.dst_lat = tq.to(lat_rad.device)
                self.dst_lon = lon_rad

                # compute indices
                permutation = torch.arange(lat_rad.shape[0] - 1, -1, -1).to(torch.long).to(lat_rad.device)
                jj = torch.searchsorted(lat_rad, self.dst_lat, sorter=permutation) - 1
                self.indices = jj[permutation]

                # compute weights
                self.interp_weights = ((self.dst_lat - lat_rad[self.indices]) / torch.diff(lat_rad)[self.indices]).reshape(-1, 1)
            else:
                raise NotImplementedError(f"Error, destination grid type {self.dst} not implemented.")
        else:
            self.dst_lat = self.src_lat
            self.dst_lon = self.src_lon

    def get_src_coords(self):
        return self.src_lat, self.src_lon

    def get_dst_coords(self):
        return self.dst_lat, self.dst_lon

    def forward(self, data):
        if self.src == self.dst:
            return data
        else:
            return torch.lerp(data[..., self.indices, :], data[..., self.indices + 1, :], self.interp_weights.to(dtype=data.dtype))


class GridQuadrature(torch.nn.Module):
    def __init__(self, quadrature_rule, img_shape, crop_shape=None, crop_offset=(0, 0), normalize=False, pole_mask=None, distributed=False):
        super().__init__()

        self.distributed = comm.is_distributed("spatial") and distributed
        crop_shape = img_shape if crop_shape is None else crop_shape

        if quadrature_rule == "naive":
            jacobian = torch.clamp(torch.sin(torch.linspace(0, torch.pi, img_shape[0])), min=0.0)
            dtheta = torch.pi / img_shape[0]
            dlambda = 2 * torch.pi / img_shape[1]
            dA = dlambda * dtheta
            quad_weight = dA * jacobian.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
            # numerical precision can be an issue here, make sure it sums to 4pi:
            quad_weight = quad_weight * (4.0 * torch.pi) / torch.sum(quad_weight)
        elif quadrature_rule == "clenshaw-curtiss":
            cost, weights = clenshaw_curtiss_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif quadrature_rule == "legendre-gauss":
            cost, weights = legendre_gauss_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif quadrature_rule == "weatherbench2":
            # compute the 'integrated area weight' that weatherbench2 uses
            lats = torch.linspace(0, torch.pi, img_shape[0])
            cell_bounds = torch.cat([torch.as_tensor([0]), (lats[:-1] + lats[1:]) / 2, torch.as_tensor([torch.pi])])
            jacobian = torch.cos(cell_bounds[:-1]) - torch.cos(cell_bounds[1:])
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * jacobian.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif quadrature_rule == "uniform":
            quad_weight = torch.ones((img_shape[0],)).unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
            # normalize to 4pi
            quad_weight = 4.0 * torch.pi * quad_weight / torch.sum(quad_weight)
        else:
            raise ValueError(f"Unknown quadrature rule {quadrature_rule}")

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.0 * torch.pi)

        # apply pole mask
        if (pole_mask is not None) and (pole_mask > 0):
            quad_weight[:pole_mask, :] = 0.0
            quad_weight[sizes[0] - pole_mask :, :] = 0.0

        # if distributed, make sure to split correctly across ranks:
        # in case of model parallelism, we need to make sure that we use the correct shapes per rank
        # for h
        if self.distributed and (comm.get_size("h") > 1):
            shapes_h = compute_split_shapes(crop_shape[0], comm.get_size("h"))
            local_shape_h = shapes_h[comm.get_rank("h")]
            local_offset_h = crop_offset[0] + sum(shapes_h[: comm.get_rank("h")])
        else:
            local_shape_h = crop_shape[0]
            local_offset_h = crop_offset[0]

        # for w
        if self.distributed and (comm.get_size("w") > 1):
            shapes_w = compute_split_shapes(crop_shape[1], comm.get_size("w"))
            local_shape_w = shapes_w[comm.get_rank("w")]
            local_offset_w = crop_offset[1] + sum(shapes_w[: comm.get_rank("w")])
        else:
            local_shape_w = crop_shape[1]
            local_offset_w = crop_offset[1]

        # crop globally if requested
        if crop_shape is not None:
            quad_weight = quad_weight[local_offset_h : local_offset_h + local_shape_h, local_offset_w : local_offset_w + local_shape_w]

        # make it contiguous
        quad_weight = quad_weight.contiguous()

        # reshape
        H, W = quad_weight.shape
        quad_weight = quad_weight.reshape(1, 1, H, W)

        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # integrate over last two axes only:
        quad = torch.sum(x * self.quad_weight, dim=(-2, -1))
        if self.distributed and (comm.get_size("spatial") > 1):
            quad = reduce_from_parallel_region(quad.contiguous(), "spatial")

        return quad
