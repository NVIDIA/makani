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

"""Naming a grid, and checking that a dataset is on the one it says.

Kept apart from :mod:`makani.utils.grids` because it needs none of what that
module needs -- no torch, no model parallel machinery, and above all not
``makani.utils`` itself, whose package import reaches the trainers and through
them the dataloaders. The storage backends verify a grid type while they are
being imported, so anything they depend on has to be free of that loop.

Everything that only needs to *name* or *check* a grid imports from here;
:mod:`makani.utils.grids` keeps the parts that build quadrature and convert
between grids, which do need torch.
"""

import logging

import numpy as np
import torch_harmonics as th


GRID_TYPE_TOLERANCE_DEGREES = 1e-3

DEFAULT_GRID_TYPE = "equiangular"

#: Grid types that datasets written before makani 0.4 may declare, and the grid they are on. They
#: named a quadrature rule on equally spaced latitudes rather than a grid; the quadrature now
#: follows from the grid.
LEGACY_GRID_TYPES = {"clenshaw-curtiss": "equiangular", "weatherbench2": "equiangular"}


def canonical_grid_type(grid_type, source="the dataset"):
    """The grid type to use for one a dataset declares, translating the legacy names."""
    if grid_type in LEGACY_GRID_TYPES:
        logging.getLogger(__name__).warning(
            f"{source} declares grid_type '{grid_type}', which names a quadrature rule rather than a grid; "
            f"reading it as '{LEGACY_GRID_TYPES[grid_type]}', whose quadrature weights the losses and metrics use."
        )
        return LEGACY_GRID_TYPES[grid_type]
    return grid_type


def _latitudes_of(grid_type, nlat):
    """The latitudes of a regular grid of torch-harmonics, in degrees, ascending, or None for another grid type."""
    try:
        grid = th.as_grid(grid_type, nlat=nlat, nlon=1)
    except (TypeError, ValueError):
        return None
    if not isinstance(grid, th.RegularGridS2):
        return None
    return np.sort(90.0 - np.degrees(np.asarray(grid.colats, dtype=np.float64)))


def expected_latitudes(grid_type, nlat):
    """The latitudes a grid of this type and size has, in degrees, ascending.

    Returns None for a grid that is not on the sphere, where there is nothing to
    check.
    """
    if grid_type == "euclidean":
        return None

    latitudes = _latitudes_of(grid_type, nlat)
    if latitudes is None:
        raise NotImplementedError(f"Grid type {grid_type} has no known latitudes")
    return latitudes


def matching_grid_types(latitudes, tolerance=GRID_TYPE_TOLERANCE_DEGREES):
    """Every known grid type whose nodes sit where these latitudes do.

    More than one is a possible answer: grids that place their nodes alike and
    differ only in their weights cannot be told apart by coordinates. What
    coordinates *can* separate is, for example, a Gauss grid from an equiangular
    one, which is the confusion worth catching, since it moves every node.

    Returns an empty list when the latitudes are no known grid at all.
    """
    actual = np.sort(np.asarray(latitudes, dtype=np.float64))
    matches = []
    for candidate in th.grid_types():
        expected = _latitudes_of(candidate, len(actual))
        if expected is not None and np.allclose(actual, expected, atol=tolerance, rtol=0.0):
            matches.append(candidate)
    return matches


def verify_grid_type(grid_type, latitudes, source="the dataset", tolerance=GRID_TYPE_TOLERANCE_DEGREES):
    """Check that a declared grid type matches the latitudes that came with it.

    The declaration decides the quadrature, and quadrature weights that do not
    belong to the grid are silently wrong: every area weighted loss and metric
    is then computed against the wrong measure, and nothing downstream can
    notice. Since the coordinates are right there, the claim is checked rather
    than trusted.

    Orientation is deliberately not part of the check. makani stores latitudes
    north to south and the quadrature routines produce them south to north, so
    what is compared is the distribution of nodes.

    What this cannot catch is a run declaring one grid where another with the
    same nodes was meant, as the coordinates hold no evidence either way. A
    Gauss grid moves every node, which is the confusion that is both catchable
    and worth catching.
    """
    expected = expected_latitudes(grid_type, len(latitudes))
    if expected is None:
        return

    actual = np.sort(np.asarray(latitudes, dtype=np.float64))
    if np.allclose(actual, expected, atol=tolerance, rtol=0.0):
        return

    matches = matching_grid_types(latitudes, tolerance)
    instead = (
        f"they are the nodes of {' or '.join(repr(name) for name in matches)}"
        if matches
        else "they match no grid type makani knows"
    )
    worst = float(np.max(np.abs(actual - expected)))
    raise ValueError(
        f"{source} declares grid_type '{grid_type}', but its {len(actual)} latitudes are not that grid: "
        f"{instead} (worst disagreement {worst:.4f} degrees). Quadrature weights follow from the grid type, "
        "so a wrong declaration silently mis-weights every area averaged loss and metric."
    )
