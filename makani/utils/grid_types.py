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

import numpy as np
from torch_harmonics.quadrature import legendre_gauss_weights


#: how close a stored latitude has to be to the one the grid type implies, in
#: degrees. Coordinates are usually float32 and metadata is sometimes rounded,
#: while the grids differ by a whole node spacing, so this is loose enough to
#: accept honest data and far tighter than the difference it has to catch.
GRID_TYPE_TOLERANCE_DEGREES = 1e-3

#: the grid a dataset is assumed to be on when nothing declares one. Only the
#: synthetic loader gets here: a real dataset has to say, and is checked.
DEFAULT_GRID_TYPE = "equiangular"

#: grid types whose nodes are equally spaced in latitude, poles included. They
#: differ from each other only in how they *weight* those nodes, so coordinates
#: cannot tell them apart -- only a Gauss grid has different nodes.
EQUALLY_SPACED_GRIDS = ("equiangular", "clenshaw-curtiss", "weatherbench2")


def expected_latitudes(grid_type, nlat):
    """The latitudes a grid of this type and size has, in degrees, ascending.

    Returns None for a grid that is not on the sphere, where there is nothing to
    check.
    """
    if grid_type in EQUALLY_SPACED_GRIDS:
        return np.linspace(-90.0, 90.0, nlat, endpoint=True)

    if grid_type == "legendre-gauss":
        cost, _ = legendre_gauss_weights(nlat, -1, 1)
        return np.sort(np.degrees(np.arccos(np.asarray(cost)) - np.pi / 2.0))

    if grid_type == "euclidean":
        return None

    raise NotImplementedError(f"Grid type {grid_type} has no known latitudes")


def matching_grid_types(latitudes, tolerance=GRID_TYPE_TOLERANCE_DEGREES):
    """Every known grid type whose nodes sit where these latitudes do.

    More than one is the normal answer, not a failure to decide: equiangular,
    Clenshaw-Curtiss and WeatherBench2 all use equally spaced nodes and differ
    only in their weights, so no coordinate can separate them. What coordinates
    *can* separate is a Gauss grid from the rest, which is the confusion worth
    catching, since it moves every node.

    Returns an empty list when the latitudes are no known grid at all.
    """
    actual = np.sort(np.asarray(latitudes, dtype=np.float64))
    matches = []
    for candidate in EQUALLY_SPACED_GRIDS + ("legendre-gauss",):
        expected = expected_latitudes(candidate, len(actual))
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

    What this cannot catch is a run declaring one equally spaced grid where
    another was meant -- equiangular, Clenshaw-Curtiss and WeatherBench2 share
    their nodes and differ only in weights, so the coordinates hold no evidence
    either way. A Gauss grid moves every node, which is the confusion that is
    both catchable and worth catching.
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
