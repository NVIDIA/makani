# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Moving a field onto a lat/lon grid without inventing or destroying anything.

Two properties decide whether a remapping is usable, and they are not the same
property:

**Consistency** -- a constant field stays constant. The weights reaching each
target point sum to one.

**Conservation** -- the integral over the sphere is unchanged. What each source
cell contributes, weighted by target areas, adds back up to its own area.

Interpolation gives the first and not the second. Inverse distance weights sum
to one by construction, but nothing makes them add up correctly per source cell,
so a remapped field slowly gains or loses mass. Rescaling the columns buys
conservation and loses consistency; iterating between the two converges to
something that is no longer an interpolation at all.

First-order conservative remapping has both, because its weights are the
*overlap areas* of the cells rather than a function of distance. It also does
the right thing in both directions without being told which it is in: where the
target is coarser than the source it averages, and where the target is finer a
target cell falls inside one source cell and simply takes its value. Nothing
here has to ask which regime it is in.

Rasters and meshes
------------------
For a lat/lon source the overlap factorises: the latitude overlap does not
depend on longitude and vice versa, so the operator is two small matrices, one
per axis, and remapping is a contraction along each. That is **exact** -- no
search, no approximation, refinement included.

For an unstructured mesh it cannot be, not without clipping spherical triangles
against lat/lon boxes, which needs the vertex connectivity that a data file does
not carry. Each cell is instead assigned whole to the target cell holding its
centre, which is the standard first-order treatment: exact in the limit where
cells tile a target cell, which is the coarsening case, and degrading where they
do not. Target cells that catch no cell at all take the nearest one, and
:attr:`ConservativeRemap.fallback_fraction` says how often that happened --
a large number means the target grid is finer than the data can support.

Everything works in the sine of latitude, because that is the coordinate area is
linear in: ``dA = d(sin phi) dlambda``. Latitude overlaps computed in degrees
would weight the poles as heavily as the equator.
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def cell_bounds_sin(latitudes: np.ndarray) -> np.ndarray:
    """Cell edges of a latitude axis, in the sine of latitude.

    Nodes are the centres and the edges sit halfway between them, with the outer
    two closing on the poles -- the same convention the WeatherBench2 quadrature
    uses, which makes the first and last cells half width for a grid whose nodes
    include the poles.
    """
    latitudes = np.asarray(latitudes, dtype=np.float64)
    interior = (latitudes[:-1] + latitudes[1:]) / 2.0
    edges = np.concatenate(([latitudes[0]], interior, [latitudes[-1]]))

    # the outer edges close on the poles rather than on the first node, so that
    # the cells tile the sphere exactly
    edges[0] = math.copysign(90.0, latitudes[0])
    edges[-1] = math.copysign(90.0, latitudes[-1])

    return np.sin(np.radians(edges))


def cell_bounds_longitude(longitudes: np.ndarray) -> np.ndarray:
    """Cell edges of a longitude axis, in degrees, spanning a full turn."""
    longitudes = np.asarray(longitudes, dtype=np.float64)
    if len(longitudes) == 1:
        return np.array([longitudes[0] - 180.0, longitudes[0] + 180.0])

    spacing = float(np.mean(np.diff(longitudes)))
    edges = np.concatenate(
        ([longitudes[0] - spacing / 2.0], (longitudes[:-1] + longitudes[1:]) / 2.0, [longitudes[-1] + spacing / 2.0])
    )
    return edges


def overlap_matrix(source_edges: np.ndarray, target_edges: np.ndarray, periodic: float = 0.0) -> np.ndarray:
    """How much of each source cell falls inside each target cell.

    Returns ``(n_target, n_source)`` of overlap lengths in whatever coordinate
    the edges are given in. ``periodic`` is the length of one full turn for an
    axis that wraps, and zero for one that does not.
    """
    # a latitude axis may run either way, and in the sine of latitude so do its
    # edges; taking the min and max per cell keeps cell i as row i regardless
    source_low = np.minimum(source_edges[:-1], source_edges[1:])[:, None]
    source_high = np.maximum(source_edges[:-1], source_edges[1:])[:, None]
    target_low = np.minimum(target_edges[:-1], target_edges[1:])
    target_high = np.maximum(target_edges[:-1], target_edges[1:])

    def straight(shift: float) -> np.ndarray:
        low = np.maximum(source_low + shift, target_low)
        high = np.minimum(source_high + shift, target_high)
        return np.clip(high - low, 0.0, None).T

    overlap = straight(0.0)
    if periodic:
        # a source cell can also reach a target cell a whole turn away
        overlap = overlap + straight(periodic) + straight(-periodic)

    return overlap


class ConservativeRemap(nn.Module):
    """Remaps a field onto a lat/lon grid, conserving its integral.

    Build with :meth:`from_raster` or :meth:`from_mesh` rather than directly.
    The operator is settled once, at construction; :meth:`forward` is a
    contraction or a scatter.

    Attributes
    ----------
    fallback_fraction : float
        Share of target cells that caught no source cell and took the nearest
        one instead. Zero for a raster source, where the overlap is exact.
    """

    def __init__(self, target_shape: Tuple[int, int]):
        super().__init__()
        self.target_shape = tuple(target_shape)
        self.fallback_fraction = 0.0
        self.source_is_mesh = False

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_raster(cls, source_lat, source_lon, target_lat, target_lon, dtype=torch.float32) -> "ConservativeRemap":
        """Build the exact operator between two lat/lon grids.

        The overlap of two lat/lon cells is the latitude overlap times the
        longitude overlap, so the operator is one matrix per axis and never has
        to be formed in full. Both are normalised by the target cell extent, so
        each row sums to one and a constant field stays constant.
        """
        remap = cls((len(target_lat), len(target_lon)))

        latitude = overlap_matrix(cell_bounds_sin(source_lat), cell_bounds_sin(target_lat))
        longitude = overlap_matrix(cell_bounds_longitude(source_lon), cell_bounds_longitude(target_lon), periodic=360.0)

        # dividing by what the target cell spans is what makes this an average
        # rather than a sum, and what makes the rows sum to one
        latitude = latitude / np.clip(latitude.sum(axis=1, keepdims=True), 1e-30, None)
        longitude = longitude / np.clip(longitude.sum(axis=1, keepdims=True), 1e-30, None)

        remap.register_buffer("latitude_weights", torch.as_tensor(latitude, dtype=dtype))
        remap.register_buffer("longitude_weights", torch.as_tensor(longitude, dtype=dtype))
        return remap

    @classmethod
    def from_mesh(
        cls,
        cell_lat,
        cell_lon,
        cell_area,
        target_lat,
        target_lon,
        dtype=torch.float32,
    ) -> "ConservativeRemap":
        """Build the operator from an unstructured mesh onto a lat/lon grid.

        Each cell is assigned whole to the target cell holding its centre and
        contributes in proportion to its area. Target cells that catch nothing
        take the value of the nearest cell, which is what the exact operator
        reduces to where the target is finer than the mesh.
        """
        remap = cls((len(target_lat), len(target_lon)))
        remap.source_is_mesh = True

        cell_lat = np.asarray(cell_lat, dtype=np.float64)
        cell_lon = np.mod(np.asarray(cell_lon, dtype=np.float64), 360.0)
        cell_area = np.asarray(cell_area, dtype=np.float64)

        target = _TargetGrid(target_lat, target_lon)
        flat = target.cell_of(cell_lat, cell_lon)

        n_target = target.n_lat * target.n_lon
        caught = np.bincount(flat, weights=cell_area, minlength=n_target)

        empty = np.flatnonzero(caught <= 0.0)
        remap.fallback_fraction = float(len(empty)) / float(n_target)

        remap.register_buffer("cell_target", torch.as_tensor(flat, dtype=torch.int64))
        remap.register_buffer("cell_weight", torch.as_tensor(cell_area, dtype=dtype))
        remap.register_buffer("target_weight", torch.as_tensor(np.clip(caught, 1e-30, None), dtype=dtype))
        remap.register_buffer("empty_target", torch.as_tensor(empty, dtype=torch.int64))
        remap.register_buffer(
            "empty_source", torch.as_tensor(target.nearest_cell(empty, cell_lat, cell_lon), dtype=torch.int64)
        )
        return remap

    # ---- application -------------------------------------------------------

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Remap a field whose trailing dimensions are the source's.

        A raster source is indexed ``(..., nlat, nlon)`` and a mesh source
        ``(..., ncells)``; the result is ``(..., target_nlat, target_nlon)``
        either way.
        """
        if self.source_is_mesh:
            return self._from_mesh(data)
        return self._from_raster(data)

    def _from_raster(self, data: torch.Tensor) -> torch.Tensor:
        latitude = self.latitude_weights.to(data.dtype)
        longitude = self.longitude_weights.to(data.dtype)

        # one contraction per axis: (..., s_lat, s_lon) -> (..., t_lat, t_lon)
        out = torch.einsum("...ij,ki->...kj", data, latitude)
        return torch.einsum("...kj,lj->...kl", out, longitude)

    def _from_mesh(self, data: torch.Tensor) -> torch.Tensor:
        weight = self.cell_weight.to(data.dtype)
        leading = data.shape[:-1]
        flat = data.reshape(-1, data.shape[-1])

        totals = torch.zeros(flat.shape[0], self.target_weight.numel(), dtype=flat.dtype, device=flat.device)
        totals.index_add_(1, self.cell_target, flat * weight)
        totals = totals / self.target_weight.to(flat.dtype)

        if self.empty_target.numel():
            totals[:, self.empty_target] = flat[:, self.empty_source]

        return totals.reshape(*leading, *self.target_shape)


class _TargetGrid:
    """The lat/lon grid a remap writes into, and where a point lands in it."""

    def __init__(self, latitudes, longitudes):
        self.latitudes = np.asarray(latitudes, dtype=np.float64)
        self.longitudes = np.mod(np.asarray(longitudes, dtype=np.float64), 360.0)
        self.n_lat = len(self.latitudes)
        self.n_lon = len(self.longitudes)

        # edges in the sine of latitude, ascending, so that searchsorted works
        # regardless of which way the axis runs
        self.sin_edges = cell_bounds_sin(self.latitudes)
        self.ascending = self.sin_edges[0] < self.sin_edges[-1]
        self.sorted_sin_edges = self.sin_edges if self.ascending else self.sin_edges[::-1]

        self.lon_edges = cell_bounds_longitude(self.longitudes)

    def cell_of(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Flat index of the target cell each point falls in."""
        row = np.searchsorted(self.sorted_sin_edges, np.sin(np.radians(lat)), side="right") - 1
        row = np.clip(row, 0, self.n_lat - 1)
        if not self.ascending:
            row = self.n_lat - 1 - row

        shifted = np.mod(lon - self.lon_edges[0], 360.0)
        column = np.clip((shifted / (360.0 / self.n_lon)).astype(np.int64), 0, self.n_lon - 1)

        return row * self.n_lon + column

    def nearest_cell(self, flat_targets: np.ndarray, cell_lat: np.ndarray, cell_lon: np.ndarray) -> np.ndarray:
        """The nearest source cell to each of these target cells.

        Only the empty ones need this, and there are few of them when the target
        is coarser than the mesh, so a direct search is affordable; where the
        target is much finer it is most of the grid, which is itself the signal
        that this is the wrong target for this data.
        """
        if len(flat_targets) == 0:
            return np.zeros(0, dtype=np.int64)

        rows, columns = np.divmod(flat_targets, self.n_lon)
        wanted_lat = np.radians(self.latitudes[rows])
        wanted_lon = np.radians(self.longitudes[columns])

        source_lat = np.radians(cell_lat)
        source_lon = np.radians(cell_lon)

        nearest = np.empty(len(flat_targets), dtype=np.int64)
        # chunked so the pairwise distances never form a single huge array
        chunk = max(1, int(4e7 // max(len(cell_lat), 1)))
        for begin in range(0, len(flat_targets), chunk):
            end = min(begin + chunk, len(flat_targets))
            cosine = np.sin(wanted_lat[begin:end, None]) * np.sin(source_lat) + np.cos(
                wanted_lat[begin:end, None]
            ) * np.cos(source_lat) * np.cos(wanted_lon[begin:end, None] - source_lon)
            nearest[begin:end] = np.argmax(cosine, axis=1)

        return nearest
