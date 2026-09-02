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

"""How much a mesh-to-grid remapping costs you, per method.

Four ways of moving a field off an unstructured mesh onto a lat/lon grid, on a
synthetic sphere where the right answer is known analytically:

``knn k=1``       the value of the closest cell, which is what "nearest" is
``knn k>1``       the k closest cells, inverse distance weighted
``barycentric``   linear on the spherical triangle containing the point
``conservative``  area weighted mean of the cells inside each target cell,
                  falling back to nearest where a target cell caught none

The first three are one family: they all evaluate the field *at a point*, and k
is how much they smooth on the way. The fourth is not -- it evaluates an
*average over an area*, which is a different question with a different answer.

They are judged on four things, and no method wins all four:

* **accuracy**, against two different right answers, because there are two.
  ``rmse@pt`` is against the field at the target point, which is what an
  interpolant is trying to reproduce. ``rmse@cell`` is against its mean over
  the target cell, which is what the grid can actually represent. Judging an
  averaging operator by the first penalises it for doing its job, and judging
  an interpolant by the second flatters it; the two columns keep that visible
  rather than deciding it here;
* **conservation**, the area weighted integral before and after, as a fraction
  of what a field of this size could carry. Normalising by the integral itself
  is useless here: a field whose integral is near zero makes any error look
  enormous;
* **smoothing**, the standard deviation of the result against the input's. All
  four methods are convex combinations, so none can overshoot -- what separates
  them is how much of the unresolvable part they average away, which is the
  point when the target is coarser and a loss when it is finer;
* **round trip**, coming back to the mesh through a common return leg, which
  measures what the forward step threw away rather than what either leg does.

The distinction that matters is between *sampling* and *averaging*. Where the
target is coarser than the mesh, several cells fall in each target cell and an
interpolation looks at one of them, which is aliasing however accurate the
interpolant is. Where the target is finer, there is nothing to average and
interpolation is the only thing that can help. The sweep over resolutions is
there to show where that crossover sits.

Noise is the point of the exercise: a smooth field flatters interpolation, and
real fields are not smooth at the grid scale.

Usage
-----
::

    python scripts/compare_remap_accuracy.py
    python scripts/compare_remap_accuracy.py --cells 20000 --noise 0.5
"""

import argparse
import math
import time
from functools import lru_cache, partial

import numpy as np
import torch
from scipy.spatial import ConvexHull, cKDTree

from makani.utils.remap import ConservativeRemap


# ---------------------------------------------------------------------------
# the sphere, the field, and the grid
# ---------------------------------------------------------------------------


def fibonacci_mesh(n_cells, seed=0):
    """Near uniform cell centres, as an icosahedral mesh would give."""
    index = np.arange(n_cells, dtype=np.float64)
    lat = np.degrees(np.arcsin(1.0 - 2.0 * (index + 0.5) / n_cells))
    lon = np.mod(np.degrees(index * np.pi * (3.0 - np.sqrt(5.0))), 360.0)
    return lat, lon


def unit_vectors(lat, lon):
    phi, lam = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)


@lru_cache(maxsize=None)
def _s2_reference_field(seed, lmax, alpha, ref_nlat):
    """A smooth field's values on a fine reference grid.

    Built from makani's ``IsotropicGaussianRandomFieldS2``: real spherical
    harmonic coefficients, scaled by a per-degree power law with exponent
    ``alpha`` and transformed to the grid with an inverse SHT. ``alpha`` is
    what sets the correlation length here: a steep falloff concentrates power
    in the lowest few degrees regardless of ``lmax`` and reads as
    large-wavelength structure, while a flatter one spreads it toward
    ``lmax`` and gives visibly finer texture -- ``lmax`` only bounds the
    *shortest* wavelength that can appear, ``alpha`` decides how much of the
    permitted range is actually used.

    A formula written directly in (lat, lon), like the trig expression this
    replaced, is generally *not* well defined at the poles -- it keeps
    depending on longitude there even though every (90, lon) is the same
    point. Spherical harmonics don't have that problem: every m != 0 mode
    vanishes at phi = +-90 by construction, so this field comes out exactly
    constant across longitude at both poles, which is what a physically
    meaningful "truth" for the pole-consistency fix needs to be. Cached on
    its arguments so every call in a sweep (mesh cells, several target grids)
    samples the same underlying field rather than drawing a fresh one.
    """
    from makani.models.noise import IsotropicGaussianRandomFieldS2

    field = IsotropicGaussianRandomFieldS2(
        img_shape=(ref_nlat, 2 * ref_nlat),
        batch_size=1,
        num_channels=1,
        sigma=1.0,
        alpha=alpha,
        lmax=lmax,
        seed=seed,
    )
    field.update()
    values = field().detach().numpy().reshape(-1)
    ref_lat, ref_lon = equiangular_grid(ref_nlat)
    return ref_lat, ref_lon, values


def analytic_field(lat, lon, noise=0.0, seed=0, lmax=10, alpha=1.0, ref_nlat=181):
    """A smooth field, optionally roughened at the grid scale.

    The smooth part is resolvable by any of the grids here (its spectrum is
    truncated to ``lmax``, comfortably below the coarsest target used in the
    sweep); the noise is not, which is what separates a method that averages
    from one that samples. ``seed`` fixes both parts, so results are
    reproducible run to run.
    """
    ref_lat, ref_lon, ref_values = _s2_reference_field(seed, lmax, alpha, ref_nlat)
    smooth = bilinear_to_points(ref_values, ref_lat, ref_lon, np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64))
    if noise <= 0.0:
        return smooth
    rng = np.random.default_rng(seed)
    return smooth + noise * rng.standard_normal(len(lat))


def equiangular_grid(n_lat):
    lat = np.linspace(90.0, -90.0, n_lat)
    lon = np.linspace(0.0, 360.0, 2 * n_lat, endpoint=False)
    return lat, lon


def cell_areas(lat, lon):
    """Area of every cell of a lat/lon grid, as a flat array."""
    edges = np.concatenate(([lat[0]], (lat[:-1] + lat[1:]) / 2.0, [lat[-1]]))
    edges[0], edges[-1] = math.copysign(90.0, lat[0]), math.copysign(90.0, lat[-1])
    d_sin = np.abs(np.diff(np.sin(np.radians(edges))))
    d_lon = 2.0 * np.pi / len(lon)
    return (d_sin[:, None] * d_lon * np.ones(len(lon))).ravel()


# ---------------------------------------------------------------------------
# the four methods
# ---------------------------------------------------------------------------


def remap_knn(tree, source_values, target_xyz, k=3, **_):
    """The k closest cells, inverse distance weighted. k=1 is nearest."""
    distance, index = tree.query(target_xyz, k=k)
    if k == 1:
        return source_values[index]

    weight = 1.0 / np.clip(distance, 1e-12, None)
    weight /= weight.sum(axis=1, keepdims=True)
    return (source_values[index] * weight).sum(axis=1)


def build_triangulation(source_xyz):
    """Spherical Delaunay: the convex hull of points on a sphere is exactly it."""
    hull = ConvexHull(source_xyz)
    triangles = hull.simplices

    incident = [[] for _ in range(len(source_xyz))]
    for number, triangle in enumerate(triangles):
        for vertex in triangle:
            incident[vertex].append(number)

    # the weights solve V^T w = p, with the vertices as the *columns*; indexing
    # gives them as rows, so the transpose is what has to be inverted
    corners = source_xyz[triangles]  # (n_tri, 3, 3), vertices in rows
    inverse = np.linalg.inv(np.swapaxes(corners, 1, 2))
    return triangles, incident, inverse


def remap_barycentric(tree, source_values, target_xyz, triangulation=None, **_):
    """Linear on the triangle containing each target point.

    The containing triangle is found among those incident to the nearest few
    vertices, which is where it has to be; a point that lands in none of them
    keeps its nearest value, which happens only at numerical edges.
    """
    triangles, incident, inverse = triangulation
    _, candidates = tree.query(target_xyz, k=3)

    out = np.empty(len(target_xyz))
    for position, point in enumerate(target_xyz):
        found = False
        seen = set()
        for vertex in candidates[position]:
            for number in incident[vertex]:
                if number in seen:
                    continue
                seen.add(number)
                weight = inverse[number] @ point
                if np.all(weight >= -1e-9):
                    weight = np.clip(weight, 0.0, None)
                    out[position] = float(source_values[triangles[number]] @ (weight / weight.sum()))
                    found = True
                    break
            if found:
                break
        if not found:
            out[position] = source_values[candidates[position][0]]
    return out


def build_conservative(source_lat, source_lon, source_area, target_lat, target_lon):
    """The real ``ConservativeRemap.from_mesh`` operator for one target grid.

    This script used to carry its own reimplementation of cell assignment and
    area-weighted averaging, parallel to ``makani.utils.remap``. The two drifted
    -- the reimplementation had no pole handling -- so both this method and the
    cell-mean reference below now go through the actual production operator
    instead. Built once per target grid and shared by both, rather than each
    pulling in its own copy of the assignment logic.
    """
    return ConservativeRemap.from_mesh(source_lat, source_lon, source_area, target_lat, target_lon)


def apply_conservative(operator, values):
    """Run the operator forward on a numpy field, numpy in and out."""
    result = operator(torch.as_tensor(values, dtype=torch.float32))
    return result.numpy().reshape(-1)


def remap_conservative(tree, source_values, target_xyz, operator=None, **_):
    """Area weighted mean via the real ``ConservativeRemap.from_mesh`` operator."""
    return apply_conservative(operator, source_values), operator.fallback_fraction


def bilinear_to_points(values, target_lat, target_lon, lat, lon):
    """Return leg: off the lat/lon grid back onto arbitrary points."""
    grid = values.reshape(len(target_lat), len(target_lon))

    ascending_lat = target_lat[::-1]
    row = np.clip(np.searchsorted(ascending_lat, lat) - 1, 0, len(target_lat) - 2)
    lo, hi = ascending_lat[row], ascending_lat[row + 1]
    t = np.clip((lat - lo) / np.clip(hi - lo, 1e-12, None), 0.0, 1.0)
    row_lo, row_hi = len(target_lat) - 1 - row, len(target_lat) - 2 - row

    spacing = 360.0 / len(target_lon)
    shifted = np.mod(lon - target_lon[0], 360.0) / spacing
    column = np.floor(shifted).astype(np.int64) % len(target_lon)
    s = shifted - np.floor(shifted)
    column_next = (column + 1) % len(target_lon)

    bottom = grid[row_lo, column] * (1 - s) + grid[row_lo, column_next] * s
    top = grid[row_hi, column] * (1 - s) + grid[row_hi, column_next] * s
    return bottom * (1 - t) + top * t


# ---------------------------------------------------------------------------


def main(args):
    source_lat, source_lon = fibonacci_mesh(args.cells)
    source_xyz = unit_vectors(source_lat, source_lon)
    source_values = analytic_field(source_lat, source_lon, noise=args.noise, seed=args.seed)
    source_area = np.full(args.cells, 4.0 * np.pi / args.cells)
    source_integral = float(source_area @ source_values)

    tree = cKDTree(source_xyz)
    started = time.perf_counter()
    triangulation = build_triangulation(source_xyz)
    hull_seconds = time.perf_counter() - started

    print(f"mesh: {args.cells} cells, noise amplitude {args.noise}")
    print(f"      spherical Delaunay: {len(triangulation[0])} triangles in {hull_seconds:.2f} s")
    print(f"      field range [{source_values.min():.3f}, {source_values.max():.3f}]\n")

    methods = {f"knn k={k}": partial(remap_knn, k=k) for k in args.neighbours}
    methods["barycentric"] = remap_barycentric
    methods["conservative"] = remap_conservative

    for n_lat in args.resolutions:
        target_lat, target_lon = equiangular_grid(n_lat)
        n_target = n_lat * 2 * n_lat
        grid_lat, grid_lon = np.meshgrid(target_lat, target_lon, indexing="ij")
        grid_lat, grid_lon = grid_lat.ravel(), grid_lon.ravel()

        target_xyz = unit_vectors(grid_lat, grid_lon)
        operator = build_conservative(source_lat, source_lon, source_area, target_lat, target_lon)

        # two references: the field at the point, and its mean over the cell.
        # The cell mean comes from feeding the noiseless field through the same
        # conservative operator being evaluated as a method, so the reference is
        # pole-consistent by the same construction and never a separate estimate
        # that could disagree with it
        truth_point = analytic_field(grid_lat, grid_lon, noise=0.0, seed=args.seed)
        target_area = cell_areas(target_lat, target_lon)

        smooth_source = analytic_field(source_lat, source_lon, noise=0.0, seed=args.seed)
        truth_cell = apply_conservative(operator, smooth_source)

        ratio = args.cells / n_target
        print(f"target {n_lat} x {2 * n_lat} = {n_target} points   ({ratio:.2f} mesh cells per target cell)")
        print(
            f"  {'method':14s} {'rmse@pt':>9s} {'rmse@cell':>10s} {'conserv':>10s} "
            f"{'smoothing':>10s} {'roundtrip':>10s} {'empty':>7s}"
        )

        for name, method in methods.items():
            begin = time.perf_counter()
            result = method(
                tree,
                source_values,
                target_xyz,
                triangulation=triangulation,
                operator=operator,
            )
            empty = None
            if isinstance(result, tuple):
                result, empty = result
            elapsed = time.perf_counter() - begin

            rmse_point = float(np.sqrt(np.mean((result - truth_point) ** 2)))
            rmse_cell = float(np.sqrt(np.mean((result - truth_cell) ** 2)))
            integral = float(target_area @ result)
            # against what a field of this magnitude spread over the sphere
            # carries, rather than against an integral that is nearly zero
            conservation = abs(integral - source_integral) / (4.0 * np.pi * float(np.std(source_values)))
            # every method here is a convex combination, so none can leave the
            # input range; what differs is how much variance survives
            smoothing = float(np.std(result)) / float(np.std(source_values))

            returned = bilinear_to_points(result, target_lat, target_lon, source_lat, source_lon)
            roundtrip = float(np.sqrt(np.mean((returned - source_values) ** 2)))

            share = "-" if empty is None else f"{empty:6.1%}"
            print(
                f"  {name:14s} {rmse_point:9.4f} {rmse_cell:10.4f} {conservation:10.2e} "
                f"{smoothing:10.3f} {roundtrip:10.4f} {share:>7s}   ({elapsed:.2f}s)"
            )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=int, default=10000, help="Cells in the synthetic mesh.")
    parser.add_argument("--noise", type=float, default=0.3, help="Amplitude of the unresolvable part of the field.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the synthetic field, mesh noise included.")
    parser.add_argument(
        "--neighbours", type=int, nargs="+", default=[1, 3, 6, 12], help="Values of k to try; k=1 is nearest."
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[24, 48, 96],
        help="Target grids to try, given as the number of latitudes (longitudes are twice that).",
    )
    main(parser.parse_args())
