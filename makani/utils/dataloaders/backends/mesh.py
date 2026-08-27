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

"""Deciding which cells of a mesh a rank reads.

The counterpart of :class:`~.makani_hdf5.StructuredChunkMixin` for data that has
no raster structure. A raster chunk is a rectangle, and saying which one takes
four numbers. A mesh has a flat list of cells in whatever order the model wrote
them, so the question "which cells belong to this rank" has to be answered
geometrically, against the grid the run will *use* rather than the one the data
is stored on.

That is what ``target_grid`` is for. The rank's share of the target raster is a
lat/lon block; the cells it needs are the ones inside that block, plus a margin,
because interpolating onto a point near the edge reaches for cells beyond it.

Reading the selection
---------------------
Cells that are wanted are not contiguous in storage, and how badly they are
scattered is a property of the mesh ordering. ICON's is good: it follows the
icosahedral root triangles, so a compact region on the sphere lands in a handful
of long runs rather than millions of isolated cells. That matters more than it
sounds, because a strided or element-wise selection through HDF5 falls back to a
gather and measures about ten times slower than reading the same bytes in one
piece.

So the selection is turned into slices, and short gaps between them are bridged
rather than split: a read costs a fixed amount before it moves a byte, so
reading *through* a gap is cheaper than paying for another read. ``merge_gap``
is where that trade sits, in cells.
"""

from typing import List, Sequence, Tuple

import numpy as np
from torch_harmonics.distributed import compute_split_shapes

from .base import GridSpec


def coalesce_runs(indices: np.ndarray, merge_gap: int = 0) -> List[slice]:
    """Turn sorted cell indices into slices, bridging gaps up to ``merge_gap``.

    A gap shorter than ``merge_gap`` is read through rather than skipped, since
    one larger read beats two smaller ones once the fixed cost of a read exceeds
    the cost of the bytes bridged.
    """
    if len(indices) == 0:
        return []

    indices = np.asarray(indices)
    # a new run starts wherever the step from the previous index is bigger than
    # the gap we are willing to read through
    breaks = np.flatnonzero(np.diff(indices) > merge_gap + 1) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(indices)]))

    return [slice(int(indices[start]), int(indices[stop - 1]) + 1) for start, stop in zip(starts, stops)]


def block_of_rank(axis: np.ndarray, io_grid: Sequence[int], io_rank: Sequence[int], dim: int) -> Tuple[float, float]:
    """The span of a target axis this rank is responsible for."""
    splits = compute_split_shapes(len(axis), io_grid[dim])
    start = sum(splits[: io_rank[dim]])
    stop = start + splits[io_rank[dim]]
    return float(axis[start]), float(axis[stop - 1])


def within_longitudes(lon: np.ndarray, low: float, high: float) -> np.ndarray:
    """Mask of longitudes inside ``[low, high]``, the short way round the sphere.

    Longitude wraps, so a block running from 350 to 10 degrees is two intervals
    in the coordinate but one region on the sphere. Comparing against a shifted
    origin keeps it one test either way.
    """
    if high - low >= 360.0:
        return np.ones_like(lon, dtype=bool)
    return np.mod(lon - low, 360.0) <= np.mod(high - low, 360.0)


class MeshChunkMixin:
    """Selects this rank's cells from an unstructured grid.

    Expects the backend to carry ``target_grid``, ``io_grid``, ``io_rank`` and
    ``halo_degrees``, and sets ``cell_index`` (the selected cells, ascending),
    ``cell_runs`` (how they will be read) and ``read_shape``.
    """

    #: cells to read through rather than break a read for; see the module note
    merge_gap: int = 1 << 20

    #: the read buffer is scratch space, rebuilt on demand rather than shipped
    #: to a worker process
    _transient_attributes = ("_buffer",)

    def _reject_raster_options(self) -> None:
        """Refuse the options that only mean something on a raster."""
        if self.subsampling_factor != 1:
            raise ValueError(
                "subsampling_factor has no meaning on an unstructured grid, where index order is not a "
                "spatial pattern. Choose a coarser target_grid instead."
            )
        if any(size is not None for size in self.crop_size) or any(anchor != 0 for anchor in self.crop_anchor):
            raise ValueError(
                "crop_size and crop_anchor describe a rectangle of a raster, which an unstructured grid "
                "does not have. Restrict the target_grid instead."
            )

    def _resolve_chunk(self, grid: GridSpec) -> GridSpec:
        self._reject_raster_options()

        decomposed = any(size > 1 for size in self.io_grid)
        if decomposed and self.target_grid is None:
            raise ValueError(
                "An unstructured dataset needs target_grid to decompose: which cells a rank reads is "
                "decided by the region of the target grid it is responsible for, and a mesh has no rows "
                "and columns to split instead."
            )

        if not decomposed:
            # one rank, so the chunk is the whole mesh and nothing has to be selected
            self.cell_index = np.arange(grid.shape[0])
            self.cell_runs = [slice(0, grid.shape[0])]
            self._plan_compaction()
            self.read_anchor = [0]
            self.read_shape = [int(grid.shape[0])]
            return grid

        halo = self.halo_degrees
        lat_low, lat_high = block_of_rank(self.target_grid.lat, self.io_grid, self.io_rank, 0)
        lon_low, lon_high = block_of_rank(self.target_grid.lon, self.io_grid, self.io_rank, 1)

        # latitudes may run either way; the block is the interval between them
        lat_low, lat_high = min(lat_low, lat_high), max(lat_low, lat_high)

        inside = (grid.lat >= lat_low - halo) & (grid.lat <= lat_high + halo)
        if self.io_grid[1] > 1:
            inside &= within_longitudes(grid.lon, lon_low - halo, lon_high + halo)

        self.cell_index = np.flatnonzero(inside)
        if len(self.cell_index) == 0:
            raise ValueError(
                f"No cells fall in this rank's block (lat {lat_low:.2f} to {lat_high:.2f}, "
                f"lon {lon_low:.2f} to {lon_high:.2f}, halo {halo:.3f} degrees). The target grid and the "
                "mesh may not describe the same sphere."
            )

        self.cell_runs = coalesce_runs(self.cell_index, self.merge_gap)
        self._plan_compaction()
        self.read_anchor = [int(self.cell_index[0])]
        self.read_shape = [int(self.cell_index[-1] - self.cell_index[0] + 1)]

        return GridSpec(
            kind=grid.kind,
            shape=(len(self.cell_index),),
            lat=grid.lat[self.cell_index],
            lon=grid.lon[self.cell_index],
        )

    def _plan_compaction(self) -> None:
        """Work out once how the runs map onto the cells that are kept.

        The runs are read whole because that is what storage is fast at, so the
        cells bridged by ``merge_gap`` come along and have to be dropped. Where
        they land in the concatenated runs depends only on the selection, so it
        is settled here rather than recomputed on every read.
        """
        self.run_offsets = np.cumsum([0] + [run.stop - run.start for run in self.cell_runs])
        self.run_span = int(self.run_offsets[-1])
        # nothing was bridged, so the runs already are the cells, in order
        self.runs_are_exact = self.run_span == len(self.cell_index)

        if self.runs_are_exact:
            self.compact_index = None
            return

        offsets = []
        for position, run in zip(self.run_offsets[:-1], self.cell_runs):
            take = self.cell_index[(self.cell_index >= run.start) & (self.cell_index < run.stop)]
            offsets.append(position + (take - run.start))
        self.compact_index = np.concatenate(offsets)

    def _run_buffer(self, dtype) -> np.ndarray:
        """A buffer holding one field over the runs, reused across reads.

        Its size is fixed by the selection, and a read fills it completely, so
        there is nothing to gain from allocating it again per channel and per
        timestep -- which at a few tens of megabytes a time is worth avoiding.
        """
        buffer = getattr(self, "_buffer", None)
        if buffer is None or buffer.dtype != dtype or buffer.shape[0] != self.run_span:
            buffer = np.empty(self.run_span, dtype=dtype)
            self._buffer = buffer
        return buffer

    def _gather_runs(self, read_run, dtype) -> np.ndarray:
        """Read every run into the shared buffer and return the cells kept.

        The result aliases the buffer when nothing was bridged, so a caller that
        keeps it has to copy; every caller here writes it straight into an
        output array.
        """
        buffer = self._run_buffer(dtype)
        for start, stop, run in zip(self.run_offsets[:-1], self.run_offsets[1:], self.cell_runs):
            read_run(run, buffer[start:stop])

        return buffer if self.runs_are_exact else buffer[self.compact_index]

    @staticmethod
    def default_halo_degrees(lat: np.ndarray, factor: float = 3.0) -> float:
        """A margin a few cells wide, from the mesh's own spacing.

        The spacing is estimated from the cell count rather than from the
        neighbour tables, which a data file does not carry: a mesh of ``n`` cells
        covering the sphere has cells of about ``sqrt(4 pi / n)`` radians across.
        """
        spacing = np.degrees(np.sqrt(4.0 * np.pi / max(len(lat), 1)))
        return float(factor * spacing)
