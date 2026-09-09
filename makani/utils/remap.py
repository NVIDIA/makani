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
import torch_harmonics.distributed as thd
from torch_harmonics.distributed import compute_split_shapes

from makani.mpu.halo import (
    azimuthal_halo_exchange,
    exchange_counts,
    neighbor_ranks,
    owner_rank,
    redistribute_indexed,
)
from makani.mpu.mappings import copy_to_parallel_region, reduce_from_parallel_region
from makani.utils import comm


def cell_bounds_sin(latitudes: np.ndarray) -> np.ndarray:
    """Cell edges of a latitude axis, in the sine of latitude.

    Nodes are the centres and the edges sit halfway between them, with the outer
    two closing on the poles -- the same convention the WeatherBench2 quadrature
    uses, which makes the first and last cells half width for a grid whose nodes
    include the poles.

    ``latitudes`` must be the *global* axis -- the outer edges are always
    snapped to the true poles, which is only correct if the array's own ends
    really are the poles. A local shard of a distributed axis is not
    self-describing that way; see :class:`ConservativeRemap`'s
    ``spatial_distributed`` handling, which builds this from the global axis
    and slices the result, rather than calling this on a local shard directly.
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


# ---------------------------------------------------------------------------
# distributed construction helpers
# ---------------------------------------------------------------------------


def _split_range(n: int, size: int, rank: int) -> Tuple[int, int]:
    """Start/stop indices of ``rank``'s contiguous slice of a length-``n`` axis."""
    splits = compute_split_shapes(n, size)
    start = sum(splits[:rank])
    return start, start + splits[rank]


def _rank_halo_radius(weights_full, target_start, target_stop, n_source, size, rank, periodic) -> int:
    """How wide a symmetric halo ``rank``'s local target rows/columns need.

    Looks at which source entries the (global) weight matrix says this rank's
    local target rows/columns actually overlap, and how far outside its own
    source block those are. Raises ``ValueError`` if any needed entry is
    reachable from neither immediate neighbour -- see the module note on
    :class:`ConservativeRemap`.
    """
    local_start, local_stop = _split_range(n_source, size, rank)
    needed = np.flatnonzero(weights_full[target_start:target_stop].any(axis=0))
    needed = needed[(needed < local_start) | (needed >= local_stop)]
    if needed.size == 0:
        return 0

    prev_rank = (rank - 1) % size if (periodic or rank > 0) else None
    next_rank = (rank + 1) % size if (periodic or rank < size - 1) else None
    prev_start, prev_stop = _split_range(n_source, size, prev_rank) if prev_rank is not None else (0, 0)
    next_start, next_stop = _split_range(n_source, size, next_rank) if next_rank is not None else (0, 0)

    radius = 0
    for c in needed.tolist():
        if (prev_rank is not None) and (prev_start <= c < prev_stop):
            radius = max(radius, prev_stop - c)
        elif (next_rank is not None) and (next_start <= c < next_stop):
            radius = max(radius, c - next_start + 1)
        else:
            raise ValueError(
                f"rank {rank}/{size} needs source index {c}, reachable from neither immediate "
                "neighbour -- the h/w decomposition is too fine (or the interpolation reaches too "
                "far) for a nearest-neighbour-only halo"
            )

    for start, stop, label in ((prev_start, prev_stop, "prev"), (next_start, next_stop, "next")):
        if (stop - start) and (radius > stop - start):
            raise ValueError(
                f"rank {rank}/{size} needs a halo of {radius}, wider than its {label} neighbour's own "
                f"block ({stop - start}) -- the decomposition is too fine for a nearest-neighbour-only halo"
            )
    if radius > (local_stop - local_start):
        raise ValueError(
            f"rank {rank}/{size}'s own block ({local_stop - local_start}) is narrower than the halo "
            f"({radius}) it would need to supply a neighbour -- the decomposition is too fine"
        )
    return radius


def _group_halo_radius(weights_full, n_target, n_source, size, periodic) -> int:
    """The halo radius every rank in the group must agree on: the max any single rank needs.

    Computed identically (and independently) by every rank from the same
    global weight matrix, so the two ends of a P2P exchange always post
    matching buffer sizes without negotiating.
    """
    target_splits = compute_split_shapes(n_target, size)
    radius = 0
    start = 0
    for rank in range(size):
        stop = start + target_splits[rank]
        radius = max(radius, _rank_halo_radius(weights_full, start, stop, n_source, size, rank, periodic))
        start = stop
    return radius


def _ensure_thd_initialized():
    """Register makani's "h"/"w" groups with torch_harmonics.distributed, once.

    ``polar_halo_exchange`` (reused here for the raster source's latitude
    halo) is keyed off torch_harmonics' own polar/azimuth group state rather
    than makani's ``comm`` groups directly; this is the same lazy-init
    pattern :class:`makani.models.noise.BaseNoiseS2` already uses.
    """
    if not thd.is_initialized():
        polar_group = comm.get_group("h") if comm.get_size("h") > 1 else None
        azimuth_group = comm.get_group("w") if comm.get_size("w") > 1 else None
        thd.init(polar_group, azimuth_group)


def _redistribute_mesh_cells(
    remap,
    cell_lat,
    cell_lon,
    cell_area,
    row_global,
    col_global,
    lat_splits,
    h_size,
    h_rank,
    lon_splits,
    w_size,
    w_rank,
    device,
):
    """Move each mesh cell to the rank that truly owns its target cell.

    A mesh cell's contribution is a crisp point-in-box test
    (:meth:`_TargetGrid.cell_of`), not a distance-based interpolation, so
    there is no "halo margin" to reason about here the way there is for a
    raster source: a rank's exactly correct, complete set of cells is exactly
    the ones whose target assignment falls in its own block. Whatever
    geographic criterion the caller used to hand this rank its cells (see the
    class docstring), this redistributes them to their exact owner.

    Two sequential stages -- h then w -- rather than one, so a cell whose
    owner differs from its current holder in *both* directions (a diagonal
    neighbour in the 2D rank grid) still arrives correctly while only ever
    talking to immediate neighbours: stage 1 gets every cell's h-ownership
    right using only the immediate h-neighbours; stage 2, run on stage 1's
    result, does the same for w. A cell that started on a diagonal neighbour
    rides stage 1 to the intermediate rank that already shares its true w,
    then stage 2 carries it the rest of the way.

    Raises ``ValueError`` (via the stage helper) if any cell's true owner is
    more than one hop away in either direction -- see the module note on
    :class:`ConservativeRemap`.

    Called lazily, from :meth:`ConservativeRemap._from_mesh` on its first
    invocation rather than from ``from_mesh`` itself, so that construction
    issues no communication at all: the receive counts below are inherently
    data-dependent (real meshes are not uniformly dense in index space the
    way a raster's rows/columns are, so "how many cells does my neighbour
    have for me" cannot be derived from the splits alone) and so need an
    actual P2P round-trip, which only happens once real input -- and with it,
    a real device to allocate on -- exists. The schedule (which local entries
    are kept vs. handed to which neighbour, and how many are received back)
    is fixed here and stored as buffers/attrs on ``remap`` so subsequent
    forward calls just replay it on the field values, no further negotiation.
    """

    def _stage(values, owner, size, rank, prev_rank, next_rank, group, label):
        # index tensors end up on `values`'s device for free this way (owner
        # already lives there): index_select/index_add_ inside
        # redistribute_indexed require it, and it's also what NCCL (if that
        # is what `group` uses) needs for P2P in the first place
        empty = torch.zeros(0, dtype=torch.int64, device=owner.device)
        keep_t = torch.nonzero(owner == rank, as_tuple=True)[0]
        next_owner = (rank + 1) % size
        send_next_t = torch.nonzero(owner == next_owner, as_tuple=True)[0] if next_rank is not None else empty
        prev_owner = (rank - 1) % size
        prev_mask = owner == prev_owner
        # a periodic group (only "w" can be) of size 2 makes prev and next
        # the same single peer -- exclude cells already claimed via "next",
        # or they would be selected (and counted) twice. This only applies
        # when next_rank is actually active: a non-periodic ("h") boundary
        # rank's blindly-modulo'd next_owner can coincide with prev_owner
        # even though next_rank is None (declined), and excluding on that
        # would wrongly drop every legitimate prev-bound cell
        if next_rank is not None:
            prev_mask = prev_mask & (owner != next_owner)
        send_prev_t = torch.nonzero(prev_mask, as_tuple=True)[0] if prev_rank is not None else empty
        if keep_t.numel() + send_prev_t.numel() + send_next_t.numel() != values.shape[1]:
            raise ValueError(
                f"mesh redistribution ({label}): rank {rank}/{size} holds cells whose true owner is "
                "more than one rank away -- the decomposition is too fine (or the caller's initial "
                "partition too coarse) for a nearest-neighbour-only redistribution"
            )

        recv_prev, recv_next = exchange_counts(
            send_prev_t.numel(), send_next_t.numel(), prev_rank, next_rank, group, values.device
        )

        result = redistribute_indexed(
            values, keep_t, prev_rank, next_rank, send_prev_t, send_next_t, recv_prev, recv_next, group
        )
        return result, (keep_t, send_prev_t, send_next_t, recv_prev, recv_next)

    h_group = comm.get_group("h") if h_size > 1 else None
    h_prev, h_next = neighbor_ranks(h_group)
    if h_rank == 0:
        h_prev = None
    if h_rank == h_size - 1:
        h_next = None

    # this rank's own coordinate metadata, placed on the caller's device --
    # following the same convention as the rest of the distributed stack
    # (e.g. makani.mpu.helpers.gather_uneven's size tensor): derive device
    # from a real tensor already in hand, never inspect the backend. row/col
    # travel as float64 columns of the same stacked tensor (simplest way to
    # keep one P2P schedule for all five coordinate arrays at once) and get
    # cast back to int64 only where an index is actually needed
    stacked = torch.stack(
        [
            torch.as_tensor(a, dtype=torch.float64).requires_grad_(False)
            for a in (cell_lat, cell_lon, cell_area, row_global, col_global)
        ],
        dim=0,
    ).to(device)

    owner_h = owner_rank(stacked[3].to(torch.int64), lat_splits)
    stage1, (keep1, send1_prev, send1_next, recv1_prev, recv1_next) = _stage(
        stacked, owner_h, h_size, h_rank, h_prev, h_next, h_group, "h"
    )

    w_group = comm.get_group("w") if w_size > 1 else None
    w_prev, w_next = neighbor_ranks(w_group)
    owner_w = owner_rank(stage1[4].to(torch.int64), lon_splits)
    final, (keep2, send2_prev, send2_next, recv2_prev, recv2_next) = _stage(
        stage1, owner_w, w_size, w_rank, w_prev, w_next, w_group, "w"
    )

    remap.register_buffer("_mesh_h_keep", keep1)
    remap.register_buffer("_mesh_h_send_prev", send1_prev)
    remap.register_buffer("_mesh_h_send_next", send1_next)
    remap._mesh_h_prev, remap._mesh_h_next = h_prev, h_next
    remap._mesh_h_recv_prev, remap._mesh_h_recv_next = recv1_prev, recv1_next
    remap._mesh_h_group = h_group

    remap.register_buffer("_mesh_w_keep", keep2)
    remap.register_buffer("_mesh_w_send_prev", send2_prev)
    remap.register_buffer("_mesh_w_send_next", send2_next)
    remap._mesh_w_prev, remap._mesh_w_next = w_prev, w_next
    remap._mesh_w_recv_prev, remap._mesh_w_recv_next = recv2_prev, recv2_next
    remap._mesh_w_group = w_group

    final = final.cpu()
    return (
        final[0].numpy(),
        final[1].numpy(),
        final[2].numpy(),
        final[3].numpy().astype(np.int64),
        final[4].numpy().astype(np.int64),
    )


def _finalize_mesh(remap, target, flat, cell_area, cell_lat, cell_lon, dtype, spatial_distributed, device=None):
    """Build the operator's index/weight buffers from a (possibly redistributed) cell set.

    Shared by the eager (non-distributed) and lazy (distributed, first
    forward call) paths of :meth:`ConservativeRemap.from_mesh` -- everything
    from here on is identical between them once ``flat`` (each cell's local
    target-cell index) and ``cell_area`` are in hand. ``device`` places the
    freshly built buffers directly where the caller's data already lives
    (only meaningful for the lazy path, where that is known); ``None`` keeps
    the CPU default, matching every other registered buffer until something
    moves the module.
    """
    # a target row sitting exactly on a pole is one physical point, so a
    # cell landing anywhere in that row's cap belongs to the pole's value
    # regardless of the longitude it happened to carry -- collapse the
    # whole row onto its first column so the polar cap is integrated as a
    # unit rather than sliced into n_lon independent (and differently
    # populated) longitude wedges
    pole_rows = target.pole_rows
    if pole_rows.size:
        row = flat // target.n_lon
        is_pole = np.isin(row, pole_rows)
        flat = np.where(is_pole, row * target.n_lon, flat)

    n_target = target.n_lat * target.n_lon
    caught = np.bincount(flat, weights=cell_area, minlength=n_target)

    empty = np.flatnonzero(caught <= 0.0)
    if pole_rows.size:
        if spatial_distributed:
            # whether the primary column is really empty can only be known
            # after the cross-"w" reduce at forward time -- this rank's
            # local view of it is not authoritative, so none of a pole
            # row's columns are judged empty from local data alone
            pole_columns = np.concatenate([np.arange(r * target.n_lon, (r + 1) * target.n_lon) for r in pole_rows])
            empty = np.setdiff1d(empty, pole_columns, assume_unique=True)
        else:
            # the other n_lon - 1 columns of a pole row never catch anything
            # by construction now; they are filled by broadcast, not fallback
            replicas = np.concatenate([np.arange(r * target.n_lon + 1, (r + 1) * target.n_lon) for r in pole_rows])
            empty = np.setdiff1d(empty, replicas, assume_unique=True)
    remap.fallback_fraction = float(len(empty)) / float(n_target)

    remap.register_buffer("cell_target", torch.as_tensor(flat, dtype=torch.int64, device=device))
    remap.register_buffer("cell_weight", torch.as_tensor(cell_area, dtype=dtype, device=device).requires_grad_(False))
    remap.register_buffer(
        "target_weight",
        torch.as_tensor(np.clip(caught, 1e-30, None), dtype=dtype, device=device).requires_grad_(False),
    )
    remap.register_buffer("empty_target", torch.as_tensor(empty, dtype=torch.int64, device=device))
    remap.register_buffer(
        "empty_source",
        torch.as_tensor(target.nearest_cell(empty, cell_lat, cell_lon), dtype=torch.int64, device=device),
    )
    remap.register_buffer("pole_rows", torch.as_tensor(pole_rows, dtype=torch.int64, device=device))


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

    Spatial model parallelism
    --------------------------
    ``spatial_distributed=True`` says every rank calls ``from_raster``/``from_mesh``
    with the *global* ``target_lat``/``target_lon`` (and, for a raster source,
    the global ``source_lat``/``source_lon``) -- cheap coordinate metadata every
    rank can afford in full, even though the field *data* is genuinely sharded
    across the "h"/"w" groups. Construction works out this rank's local "h"/"w"
    block from that, and with it, exactly what communication (if any) each
    forward call needs -- no heuristic margin, no per-call negotiation.

    For a **raster** source, an ordinary (non-pole) target row or column needs
    a small, exactly-computed number of boundary source rows/columns from the
    immediate "h"/"w" neighbour, fetched via a point-to-point halo exchange
    (:func:`torch_harmonics.distributed.polar_halo_exchange` for "h",
    :func:`makani.mpu.halo.azimuthal_halo_exchange` for "w") each forward call.

    For a **mesh** source, a cell's contribution is a crisp point-in-box test,
    not an interpolation with reach, so there is nothing to fetch as a halo:
    construction instead redistributes each rank's cells to whichever rank's
    target block their centre truly falls in
    (see :func:`_redistribute_mesh_cells`), and every forward call replays
    that same redistribution on the field values. One consequence: the
    fallback for a target cell that catches nothing (see
    :attr:`fallback_fraction`) can only search cells this rank actually
    holds, so it may pick a farther cell than the true global nearest one --
    a pre-existing approximation (the module docstring already treats a large
    ``fallback_fraction`` as "this target is too fine for this data") that
    distribution makes slightly less precise, not a new failure mode.

    Both mechanisms assume a halo/redistribution never needs to reach past
    the immediate neighbour; if it would, construction raises ``ValueError``
    rather than silently routing through a second hop -- that means the
    decomposition is too fine (or the interpolation reaches too far) for this
    scheme.

    The pole is the one place *neither* mechanism is enough on its own, raster
    or mesh: every longitude converges there, so the true polar value needs
    contributions from the entire latitude circle, not a neighbourhood. The
    rank(s) whose block touches a pole handle it with an all-reduce over the
    "w" group -- exact for a raster source, since the undistributed "w" split
    already partitions the full circle with no gaps; exact for a mesh source
    too, since redistribution is now exact rather than a heuristic margin.
    Everywhere off the pole, on a rank that needs no halo/redistribution at
    all, this is unaffected: no communication, same computation as the
    single-process case.

    The reduce is :func:`~makani.mpu.mappings.reduce_from_parallel_region`
    immediately followed by :func:`~makani.mpu.mappings.copy_to_parallel_region`,
    not `reduce_from_parallel_region` alone. That pairing matters: each "w"
    rank applies a *different* downstream computation to the shared reduced
    value (a different longitude weight matrix, or a different division, and
    a different upstream gradient from whatever loss uses this rank's own
    slice of the target), not an identically replicated one -- unlike
    Megatron-style tensor parallelism, which `reduce_from_parallel_region`
    alone is built for and whose pass-through backward assumes it. Following
    it with `copy_to_parallel_region` (whose own backward is a sum-all-reduce)
    makes the composition sum in *both* directions, which is what a value
    combined via true distinct-per-rank contributions and then consumed
    differently by every rank actually needs -- the same composition
    :mod:`makani.mpu.layer_norm` relies on for its distributed mean/var.
    """

    def __init__(self, target_shape: Tuple[int, int]):
        super().__init__()
        self.target_shape = tuple(target_shape)
        self.fallback_fraction = 0.0
        self.source_is_mesh = False
        self.spatial_distributed = False
        self.halo_lat = 0
        self.halo_lon = 0

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_raster(
        cls, source_lat, source_lon, target_lat, target_lon, dtype=torch.float32, spatial_distributed=False
    ) -> "ConservativeRemap":
        """Build the exact operator between two lat/lon grids.

        The overlap of two lat/lon cells is the latitude overlap times the
        longitude overlap, so the operator is one matrix per axis and never has
        to be formed in full. Both are normalised by the target cell extent, so
        each row sums to one and a constant field stays constant.

        With ``spatial_distributed=True``, all four coordinate arrays are the
        *global* axes (see the class docstring); the trailing dimensions of
        the tensor passed to :meth:`forward` are this rank's local "h"/"w"
        block of the source, not the global source.
        """
        source_lat_full = np.asarray(source_lat, dtype=np.float64)
        source_lon_full = np.asarray(source_lon, dtype=np.float64)
        target_lat_full = np.asarray(target_lat, dtype=np.float64)
        target_lon_full = np.asarray(target_lon, dtype=np.float64)
        n_source_lat, n_source_lon = len(source_lat_full), len(source_lon_full)
        n_target_lat, n_target_lon = len(target_lat_full), len(target_lon_full)

        h_size, h_rank = (comm.get_size("h"), comm.get_rank("h")) if spatial_distributed else (1, 0)
        w_size, w_rank = (comm.get_size("w"), comm.get_rank("w")) if spatial_distributed else (1, 0)

        t_lat_start, t_lat_stop = _split_range(n_target_lat, h_size, h_rank)
        t_lon_start, t_lon_stop = _split_range(n_target_lon, w_size, w_rank)
        local_target_lat = target_lat_full[t_lat_start:t_lat_stop]
        local_target_lon = target_lon_full[t_lon_start:t_lon_stop]

        remap = cls((len(local_target_lat), len(local_target_lon)))
        remap.spatial_distributed = spatial_distributed

        # built from the GLOBAL arrays so edges are correct everywhere,
        # including for an interior h-rank whose own shard does not touch a
        # pole -- cell_bounds_sin snaps a local shard's own ends to the poles
        # unconditionally, which is only right for a rank that actually owns
        # one
        latitude_full = overlap_matrix(cell_bounds_sin(source_lat_full), cell_bounds_sin(target_lat_full))
        longitude_full = overlap_matrix(
            cell_bounds_longitude(source_lon_full), cell_bounds_longitude(target_lon_full), periodic=360.0
        )
        latitude_full = latitude_full / np.clip(latitude_full.sum(axis=1, keepdims=True), 1e-30, None)
        longitude_full = longitude_full / np.clip(longitude_full.sum(axis=1, keepdims=True), 1e-30, None)

        latitude_local = latitude_full[t_lat_start:t_lat_stop]
        longitude_local = longitude_full[t_lon_start:t_lon_stop]

        halo_lat = (
            _group_halo_radius(latitude_full, n_target_lat, n_source_lat, h_size, periodic=False) if h_size > 1 else 0
        )
        halo_lon = (
            _group_halo_radius(longitude_full, n_target_lon, n_source_lon, w_size, periodic=True) if w_size > 1 else 0
        )

        # the source axis of the weight matrix must always match this rank's
        # local *source* block (plus halo) -- leaving it spanning the full
        # global source axis when no halo is needed no longer matches a
        # spatially-distributed data tensor, which only ever holds its own block
        s_lat_start, s_lat_stop = _split_range(n_source_lat, h_size, h_rank)
        s_lon_start, s_lon_stop = _split_range(n_source_lon, w_size, w_rank)

        if halo_lat > 0:
            padded = 2 * halo_lat + (s_lat_stop - s_lat_start)
            latitude = np.zeros((latitude_local.shape[0], padded), dtype=latitude_local.dtype)
            g_lo, g_hi = max(0, s_lat_start - halo_lat), min(n_source_lat, s_lat_stop + halo_lat)
            p_lo = g_lo - (s_lat_start - halo_lat)
            # positions the exchange zero-pads at a true polar boundary are
            # correctly left at 0 here too: nothing real overlaps them
            latitude[:, p_lo : p_lo + (g_hi - g_lo)] = latitude_local[:, g_lo:g_hi]
        else:
            latitude = latitude_local[:, s_lat_start:s_lat_stop]

        if halo_lon > 0:
            lon_window = np.arange(s_lon_start - halo_lon, s_lon_stop + halo_lon) % n_source_lon
            longitude = longitude_local[:, lon_window]
        else:
            longitude = longitude_local[:, s_lon_start:s_lon_stop]

        # a target row sitting exactly on a pole is one physical point with
        # n_lon coordinate labels, not n_lon distinct cells -- it is forced
        # constant across longitude in _from_raster rather than here, because
        # the constant has to be the zonal mean of the *latitude-averaged*
        # row, which only exists once the latitude contraction has run
        pole_mask = np.isclose(np.abs(local_target_lat), 90.0)

        remap.register_buffer("latitude_weights", torch.as_tensor(latitude, dtype=dtype).requires_grad_(False))
        remap.register_buffer("longitude_weights", torch.as_tensor(longitude, dtype=dtype).requires_grad_(False))
        remap.register_buffer("pole_mask", torch.as_tensor(pole_mask, dtype=torch.bool))
        remap.halo_lat = int(halo_lat)
        remap.halo_lon = int(halo_lon)
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
        spatial_distributed=False,
    ) -> "ConservativeRemap":
        """Build the operator from an unstructured mesh onto a lat/lon grid.

        Each cell is assigned whole to the target cell holding its centre and
        contributes in proportion to its area. Target cells that catch nothing
        take the value of the nearest cell, which is what the exact operator
        reduces to where the target is finer than the mesh.

        With ``spatial_distributed=True``, ``target_lat``/``target_lon`` are
        the *global* target axes (see the class docstring); ``cell_lat``/
        ``cell_lon``/``cell_area`` are this rank's mesh cells under whatever
        geographic criterion the caller used to select them (e.g. the ICON
        backend's block-plus-margin) -- redistributing them to their exact
        owner needs real communication (see :func:`_redistribute_mesh_cells`),
        so unlike the rest of construction it is deferred to the first
        :meth:`forward` call rather than done here: this method itself issues
        no communication, and the deferred step gets a real device to work
        with (from the first input tensor) instead of guessing one.
        """
        cell_lat = np.asarray(cell_lat, dtype=np.float64)
        cell_lon = np.mod(np.asarray(cell_lon, dtype=np.float64), 360.0)
        cell_area = np.asarray(cell_area, dtype=np.float64)

        target_global = _TargetGrid(target_lat, target_lon)

        if spatial_distributed:
            h_size, h_rank = comm.get_size("h"), comm.get_rank("h")
            w_size, w_rank = comm.get_size("w"), comm.get_rank("w")
            lat_splits = compute_split_shapes(target_global.n_lat, h_size)
            lon_splits = compute_split_shapes(target_global.n_lon, w_size)
            t_lat_start, t_lat_stop = _split_range(target_global.n_lat, h_size, h_rank)
            t_lon_start, t_lon_stop = _split_range(target_global.n_lon, w_size, w_rank)
            local_target_lat = target_global.latitudes[t_lat_start:t_lat_stop]
            local_target_lon = target_global.longitudes[t_lon_start:t_lon_stop]
            # only .n_lat/.n_lon/.pole_rows/.nearest_cell are used from this --
            # none depend on cell_bounds_sin, so a local shard's edges being
            # wrong (see from_raster) does not matter here
            target = _TargetGrid(local_target_lat, local_target_lon)

            remap = cls((target.n_lat, target.n_lon))
            remap.source_is_mesh = True
            remap.spatial_distributed = True
            remap._mesh_ready = False
            remap._mesh_pending = dict(
                cell_lat=cell_lat,
                cell_lon=cell_lon,
                cell_area=cell_area,
                target_global=target_global,
                target=target,
                dtype=dtype,
                t_lat_start=t_lat_start,
                t_lon_start=t_lon_start,
                lat_splits=lat_splits,
                lon_splits=lon_splits,
                h_size=h_size,
                h_rank=h_rank,
                w_size=w_size,
                w_rank=w_rank,
            )
            return remap

        target = target_global
        remap = cls((target.n_lat, target.n_lon))
        remap.source_is_mesh = True
        flat = target.cell_of(cell_lat, cell_lon)
        _finalize_mesh(remap, target, flat, cell_area, cell_lat, cell_lon, dtype, spatial_distributed=False)
        return remap

    def _lazy_init_mesh(self, device: torch.device) -> None:
        """Finish mesh construction: redistribute cells to their exact owner, then build the operator.

        Runs once, on the first :meth:`forward` call, so that ``from_mesh``
        itself can issue no communication (see its docstring) -- the receive
        counts the redistribution needs are data-dependent and only knowable
        via an actual round-trip, which needs a real device to run on.
        Subsequent calls are no-ops; the resulting buffers do not change
        shape from one call to the next, matching how e.g. ``LazyLinear``
        defers its own parameter creation to first use.
        """
        pending = self._mesh_pending
        target_global, target = pending["target_global"], pending["target"]
        cell_lat, cell_lon, cell_area = pending["cell_lat"], pending["cell_lon"], pending["cell_area"]

        with torch.no_grad():
            row_global, col_global = np.divmod(target_global.cell_of(cell_lat, cell_lon), target_global.n_lon)
            cell_lat, cell_lon, cell_area, row_global, col_global = _redistribute_mesh_cells(
                self,
                cell_lat,
                cell_lon,
                cell_area,
                row_global,
                col_global,
                pending["lat_splits"],
                pending["h_size"],
                pending["h_rank"],
                pending["lon_splits"],
                pending["w_size"],
                pending["w_rank"],
                device,
            )
        flat = (row_global - pending["t_lat_start"]) * target.n_lon + (col_global - pending["t_lon_start"])

        _finalize_mesh(
            self, target, flat, cell_area, cell_lat, cell_lon, pending["dtype"], spatial_distributed=True, device=device
        )
        self._mesh_ready = True
        self._mesh_pending = None

    # ---- application -------------------------------------------------------

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Remap a field whose trailing dimensions are the source's.

        A raster source is indexed ``(..., nlat, nlon)`` and a mesh source
        ``(..., ncells)``; the result is ``(..., target_nlat, target_nlon)``
        either way. For a distributed operator, all of these are this rank's
        local shard, not the global tensor.
        """
        if self.source_is_mesh:
            return self._from_mesh(data)
        return self._from_raster(data)

    def _from_raster(self, data: torch.Tensor) -> torch.Tensor:
        # registered buffers, so a caller's remap.to(device) already moved
        # these; only the dtype cast (e.g. for AMP) is this module's job
        latitude = self.latitude_weights.to(data.dtype)
        longitude = self.longitude_weights.to(data.dtype)
        pole_mask = self.pole_mask

        if self.spatial_distributed and ((self.halo_lat > 0) or (self.halo_lon > 0)):
            leading = data.shape[:-2]
            s_lat, s_lon = data.shape[-2], data.shape[-1]
            padded = data.reshape(-1, 1, s_lat, s_lon)
            if self.halo_lat > 0:
                _ensure_thd_initialized()
                padded = thd.polar_halo_exchange(padded, self.halo_lat)
            if self.halo_lon > 0:
                padded = azimuthal_halo_exchange(padded, self.halo_lon, comm.get_group("w"))
            data = padded.reshape(*leading, padded.shape[-2], padded.shape[-1])

        # one contraction per axis: (..., s_lat, s_lon) -> (..., t_lat, t_lon)
        out = torch.einsum("...ij,ki->...kj", data, latitude)
        if pole_mask.any():
            # a row on a pole is one point relabelled n_lon times; averaging
            # a constant row over longitude leaves it that same constant, so
            # this alone makes the pole rows come out uniform after the next
            # contraction
            if self.spatial_distributed and comm.get_size("w") > 1:
                # this rank only holds a slice of the row's longitude; the
                # undistributed "w" split tiles the full circle with no gaps
                # or overlap, so summing the partial sums and counts across it
                # is exact, not an approximation. makani's differentiable
                # collectives (not raw torch.distributed) so gradients still
                # flow back through the reduce during training.
                # out's longitude axis is still padded with this rank's own
                # halo columns (duplicates of a neighbour's real data, added
                # above for the *ordinary* target rows/columns) -- summing
                # those in too would double count them against the neighbour
                # that owns them, so only the true local slice goes into the
                # pole's zonal sum/count
                # reduce_from_parallel_region alone is the wrong primitive
                # here: its backward is a pass-through, correct only when the
                # reduced value feeds an *identically replicated* downstream
                # computation on every rank (Megatron-style tensor
                # parallelism, where grad_output is already the same value on
                # every rank). Here each "w" rank applies a *different*
                # longitude weight matrix (and gets a different upstream
                # gradient) to the same zonal_mean, so the true gradient needs
                # summing grad_output across ranks too -- which is exactly
                # what following the reduce with copy_to_parallel_region does
                # (its backward is itself a sum-all-reduce), the same
                # reduce-then-copy composition makani.mpu.layer_norm already
                # relies on for its distributed mean/var
                local = out[..., self.halo_lon : out.shape[-1] - self.halo_lon] if self.halo_lon > 0 else out
                local_sum = local.sum(dim=-1, keepdim=True)
                local_count = torch.full_like(local_sum, local.shape[-1])
                total_sum = copy_to_parallel_region(reduce_from_parallel_region(local_sum, "w"), "w")
                total_count = copy_to_parallel_region(reduce_from_parallel_region(local_count, "w"), "w")
                zonal_mean = total_sum / total_count
            else:
                zonal_mean = out.mean(dim=-1, keepdim=True)
            out = torch.where(pole_mask.view(-1, 1), zonal_mean, out)
        return torch.einsum("...kj,lj->...kl", out, longitude)

    def _from_mesh(self, data: torch.Tensor) -> torch.Tensor:
        # registered buffers, so a caller's remap.to(device) already moved
        # these, including the index tensors -- index_select/index_add_
        # below need them on the same device as data, same as any module
        if self.spatial_distributed:
            if not self._mesh_ready:
                self._lazy_init_mesh(data.device)
            # replay the exact same redistribution construction fixed on the
            # coordinates onto the field values, so data ends up ordered
            # exactly like self.cell_target/self.cell_weight expect
            data = redistribute_indexed(
                data,
                self._mesh_h_keep,
                self._mesh_h_prev,
                self._mesh_h_next,
                self._mesh_h_send_prev,
                self._mesh_h_send_next,
                self._mesh_h_recv_prev,
                self._mesh_h_recv_next,
                self._mesh_h_group,
            )
            data = redistribute_indexed(
                data,
                self._mesh_w_keep,
                self._mesh_w_prev,
                self._mesh_w_next,
                self._mesh_w_send_prev,
                self._mesh_w_send_next,
                self._mesh_w_recv_prev,
                self._mesh_w_recv_next,
                self._mesh_w_group,
            )

        weight = self.cell_weight.to(data.dtype)
        leading = data.shape[:-1]
        flat = data.reshape(-1, data.shape[-1])

        totals = torch.zeros(flat.shape[0], self.target_weight.numel(), dtype=flat.dtype, device=flat.device)
        totals.index_add_(1, self.cell_target, flat * weight)
        target_weight = self.target_weight.to(flat.dtype)

        if self.pole_rows.numel() and self.spatial_distributed and comm.get_size("w") > 1:
            # every column of a pole row was collapsed onto its row's first
            # (primary) column at construction time; this rank's redistributed
            # cells only ever cover its own block, so the primary column's
            # total and weight are partial sums that need combining across
            # every rank that shares this pole -- exact, since redistribution
            # is exact (unlike the raster case there is no "did the halo reach
            # far enough" caveat here)
            # reduce_from_parallel_region alone is the wrong primitive here:
            # its backward is a pass-through, correct only when the reduced
            # value feeds an *identically replicated* downstream computation
            # on every rank. Here each "w" rank divides by its own totals and
            # gets its own upstream gradient, so the true gradient needs
            # summing across ranks too -- following the reduce with
            # copy_to_parallel_region (whose backward is itself a
            # sum-all-reduce) does that, the same composition
            # makani.mpu.layer_norm relies on for its distributed mean/var
            n_lon = self.target_shape[1]
            primary = torch.as_tensor(
                [row * n_lon for row in self.pole_rows.tolist()], dtype=torch.int64, device=flat.device
            )
            pole_totals = copy_to_parallel_region(
                reduce_from_parallel_region(totals.index_select(1, primary), "w"), "w"
            )
            pole_weight = copy_to_parallel_region(
                reduce_from_parallel_region(target_weight.index_select(0, primary), "w"), "w"
            )
            totals = totals.index_copy(1, primary, pole_totals)
            target_weight = target_weight.index_copy(0, primary, pole_weight)

        totals = totals / torch.clamp(target_weight, min=1e-30)

        if self.empty_target.numel():
            totals[:, self.empty_target] = flat[:, self.empty_source]

        if self.pole_rows.numel():
            n_lon = self.target_shape[1]
            for row in self.pole_rows.tolist():
                start = row * n_lon
                totals[:, start : start + n_lon] = totals[:, start : start + 1]

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

        # a row exactly on a pole is one physical point carrying n_lon
        # coordinate labels, not n_lon distinct cells
        self.pole_rows = np.flatnonzero(np.isclose(np.abs(self.latitudes), 90.0))

    def cell_of(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Flat index of the target cell each point falls in.

        Only correct when ``self`` was built from the *global* axis --
        ``self.sin_edges`` inherits ``cell_bounds_sin``'s pole-snapping, which
        is wrong for a local shard that does not itself touch a pole.
        """
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
