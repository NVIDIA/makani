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
from dataclasses import dataclass
from typing import Tuple

import torch

import torch_harmonics as th

if not hasattr(th, "as_grid"):
    raise ImportError(
        f"makani needs the grid-descriptor API of torch-harmonics 1.0, which torch-harmonics "
        f"{getattr(th, '__version__', '?')} does not have; install the version pinned in pyproject.toml"
    )


from makani.distributed import comm
from torch_harmonics.distributed import compute_split_shapes
from makani.distributed.mappings import reduce_from_parallel_region


#: The grid type of a field in the plane rather than on the sphere.
PLANAR_GRID_TYPE = "euclidean"


@dataclass(frozen=True)
class PlanarGrid:
    r"""
    A regular grid in the plane, the ``euclidean`` grid type.

    Not a grid on the sphere, so no spherical layer takes it. Its quadrature weights are uniform,
    normalized to :math:`4\pi` like the weights of a grid on the sphere, so that a loss or a metric
    reads the same on either.
    """

    nlat: int
    nlon: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nlat, self.nlon)

    @property
    def quad_weights(self) -> torch.Tensor:
        return torch.full((self.nlat * self.nlon,), 4.0 * math.pi / (self.nlat * self.nlon), dtype=torch.float64)


def make_grid(img_shape: Tuple[int, int], grid_type: str):
    """
    The grid of a field of this shape and grid type.

    A grid type names a regular grid of torch-harmonics, whose latitude rings all carry the same
    number of longitudes (``equiangular``, ``legendre-gauss``, ``lobatto``, ``fejer1``,
    ``trapezoidal``), or ``euclidean`` for a field in the plane. Other grids, such as HEALPix, are
    refused, as makani supports none of them yet.

    Parameters
    ----------
    img_shape : (int, int)
        Global ``(nlat, nlon)`` of the field.
    grid_type : str
        The grid type.

    Returns
    -------
    torch_harmonics.RegularGridS2 or PlanarGrid
    """
    nlat, nlon = (int(n) for n in img_shape)
    if grid_type == PLANAR_GRID_TYPE:
        return PlanarGrid(nlat, nlon)
    if grid_type not in regular_grid_types():
        raise ValueError(
            f"Grid type {grid_type} is not supported; makani supports the regular grids of torch-harmonics, "
            f"{', '.join(regular_grid_types())}, and {PLANAR_GRID_TYPE} for a field in the plane"
        )
    return th.as_grid(grid_type, nlat=nlat, nlon=nlon)


def regular_grid_types() -> Tuple[str, ...]:
    """The grid types of the regular grids of torch-harmonics, the grids on the sphere makani supports."""
    types = []
    pending = [th.RegularGridS2]
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        grid_type = getattr(cls, "grid_type", None)
        if isinstance(grid_type, str) and grid_type in th.grid_types():
            types.append(grid_type)
    return tuple(t for t in th.grid_types() if t in types)


def sht_grid(img_shape, grid_type):
    """
    The grid of a field on the sphere, for a spherical harmonic transform.

    Like :func:`make_grid`, and refuses the plane, where there are no spherical harmonics.

    Returns
    -------
    torch_harmonics.RegularGridS2
    """
    if grid_type == PLANAR_GRID_TYPE:
        raise ValueError(f"A field on a {PLANAR_GRID_TYPE} grid lies in the plane and has no spherical harmonics")
    return make_grid(img_shape, grid_type)


def spherical_bandlimit(grid) -> int:
    r"""
    The spherical bandlimit makani truncates to on a grid on the sphere, as an exclusive ``lmax``.

    One degree below :attr:`torch_harmonics.GridS2.default_sht_degree`, the degrees the quadrature
    of the grid integrates exactly, and at most ``nlon // 2``. The spectral layers and losses keep
    it as their ``lmax`` and ``mmax``, so it sets the shapes of spectral weights: it is the value
    makani has always used on equiangular and Legendre-Gauss grids, so that trained models keep
    their shapes.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2
        The grid, see :func:`sht_grid`.

    Returns
    -------
    int
    """
    if not isinstance(grid, th.RegularGridS2):
        raise ValueError(f"A spherical bandlimit needs a regular grid on the sphere, not {grid}")
    return min(grid.default_sht_degree - 1, grid.nlon // 2)


class GridConverter(torch.nn.Module):
    r"""
    Moves fields from the latitudes of the data to those of the model's grid.

    Datasets live on the grid their metadata names, models on ``model_grid_type``. Where the two
    differ, each target latitude, those of the torch-harmonics grid with as many latitudes as the
    data, is interpolated linearly between the two data latitudes around it. Longitudes are kept.

    Parameters
    ----------
    src_grid_type : str
        The grid type of the data.
    dst_grid_type : str
        The grid type of the model, a regular grid on the sphere, see :func:`sht_grid`.
    lat_rad : torch.Tensor
        The latitudes of the data in radians, from north to south.
    lon_rad : torch.Tensor
        The longitudes of the data in radians.

    Raises
    ------
    ValueError
        If the model's grid is not a regular grid on the sphere, or has latitudes outside of those
        of the data, as a grid with poles has where the data has none.
    """

    def __init__(self, src_grid_type, dst_grid_type, lat_rad, lon_rad):
        super().__init__()
        self.src = src_grid_type
        self.dst = dst_grid_type
        self.src_lat = lat_rad
        self.src_lon = lon_rad

        if self.src == self.dst:
            self.dst_lat = self.src_lat
            self.dst_lon = self.src_lon
            return

        nlat, nlon = lat_rad.shape[0], lon_rad.shape[0]
        dst_lat = sht_grid((nlat, nlon), self.dst).lats.to(device=lat_rad.device)
        src_lat = lat_rad.to(dtype=dst_lat.dtype)

        # the data latitudes around each target latitude: the upper one at indices, the lower at indices + 1
        tol = 1e-6 * math.pi
        if (dst_lat > src_lat[0] + tol).any() or (dst_lat < src_lat[-1] - tol).any():
            raise ValueError(
                f"The {self.dst} grid has latitudes beyond those of the {self.src} data, which span "
                f"{math.degrees(src_lat[0]):.4f} to {math.degrees(src_lat[-1]):.4f} degrees; "
                "values there would be extrapolated"
            )
        # the latitudes descend: count the data latitudes north of each target latitude
        indices = torch.searchsorted(-src_lat, -dst_lat, right=True) - 1
        indices = torch.clamp(indices, 0, nlat - 2)

        self.dst_lat = dst_lat.to(dtype=lat_rad.dtype)
        self.dst_lon = lon_rad
        self.indices = indices
        self.interp_weights = (
            ((dst_lat - src_lat[indices]) / (src_lat[indices + 1] - src_lat[indices])).to(dtype=lat_rad.dtype)
        ).reshape(-1, 1)

    def get_src_coords(self):
        return self.src_lat, self.src_lon

    def get_dst_coords(self):
        return self.dst_lat, self.dst_lon

    def forward(self, data):
        if self.src == self.dst:
            return data
        else:
            return torch.lerp(
                data[..., self.indices, :], data[..., self.indices + 1, :], self.interp_weights.to(dtype=data.dtype)
            )


class GridQuadrature(torch.nn.Module):
    r"""
    Integrates fields over (a crop of) a grid, with the quadrature weights of the grid.

    The weights are those of the grid, :attr:`torch_harmonics.GridS2.quad_weights` on the sphere:
    Clenshaw-Curtis on an equiangular grid, Gauss-Legendre on a Gaussian one, uniform in the plane.
    They sum to :math:`4\pi`, or to one with ``normalize``.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2 or PlanarGrid
        The global grid, see :func:`make_grid`.
    crop_shape, crop_offset : (int, int), optional
        The crop of the grid to integrate over; the whole grid by default.
    normalize : bool, optional
        Divide the weights by :math:`4\pi`.
    distributed : bool, optional
        Hold this rank's piece of the crop under spatial parallelism, and reduce over the ranks.
    """

    def __init__(self, grid, crop_shape=None, crop_offset=(0, 0), normalize=False, distributed=False):
        super().__init__()

        self.distributed = comm.is_distributed("spatial") and distributed
        img_shape = grid.shape
        crop_shape = img_shape if crop_shape is None else crop_shape

        quad_weight = grid.quad_weights.reshape(img_shape)

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.0 * torch.pi)

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
            quad_weight = quad_weight[
                local_offset_h : local_offset_h + local_shape_h, local_offset_w : local_offset_w + local_shape_w
            ]

        # make it contiguous
        quad_weight = quad_weight.contiguous()

        # reshape
        H, W = quad_weight.shape
        quad_weight = quad_weight.reshape(1, 1, H, W)

        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # integrate over last two axes only:
        quad = torch.sum(x * self.quad_weight.to(x.dtype), dim=(-2, -1))
        if self.distributed and (comm.get_size("spatial") > 1):
            quad = reduce_from_parallel_region(quad.contiguous(), "spatial")

        return quad
