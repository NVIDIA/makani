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

"""What a dataloader tells the rest of the run about the data it emits.

Every loader answers the same questions -- how big is the field, which part of
it does this run use, which part of *that* does this rank hold, how many
channels come out -- and each used to answer them in its own spelling. The DALI
loader carried ``img_local_shape_x``, the torch dataset carried ``read_shape``
*and* a block of ``img_*`` aliases written "for compatibility", the synthetic
loader carried ``read_shape`` alone, and ``get_dataloader`` reconciled them by
hand into a ``types.SimpleNamespace`` per branch. Three copies of one mapping,
and nothing anywhere stating what the mapping was.

:class:`DataShapes` is that statement. A loader reports one, ``get_dataloader``
returns it, and :meth:`makani.utils.driver.Driver._set_data_shapes` reads it.

Shape of the type
-----------------
The split is the one ``torch_harmonics`` uses for its grid descriptors: a global
grid, and a :class:`Shard` describing one rank's piece of it. When
``GridS2``/``GridShardS2`` reach a tagged release, ``grid_shape`` and ``shard``
can become those types without anything else moving, and makani inherits their
quadrature weights and spectral bounds. Until then this stays dependency free
and deliberately small.

The ``img_shape_x`` style names survive as properties. They are what ``params``
carries, what configs and checkpoints are written in, and what ``loss.py``,
``metric.py`` and the preprocessor read. They are a compatibility surface over a
structured core, not the core itself.
"""

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Shard:
    """One rank's piece of a decomposed field.

    Named after ``torch_harmonics.GridShardS2``, whose ``shape``, ``lat_offset``
    and ``lon_offset`` mean the same things, so that adopting it later is a
    rename rather than a redesign.

    Attributes
    ----------
    shape : tuple of int
        Rows and columns this rank holds.
    lat_offset, lon_offset : int
        Where they start in the full field.
    lats, lons : list of float
        Coordinates of this rank's points, in degrees.
    """

    shape: Tuple[int, int]
    lat_offset: int
    lon_offset: int
    lats: List[float] = field(default_factory=list)
    lons: List[float] = field(default_factory=list)

    @property
    def nlat(self) -> int:
        return self.shape[0]

    @property
    def nlon(self) -> int:
        return self.shape[1]


@dataclass(frozen=True)
class DataShapes:
    """The geometry and channel counts of what a loader emits.

    Attributes
    ----------
    in_channels, out_channels : sequence of int
        Channel indices the loader was asked for, in the order it emits them.
    grid_shape : tuple of int
        The dataset's own field, before cropping or decomposition.
    crop_shape, crop_offset : tuple of int
        The region of it this run uses.
    shard : Shard
        The part of that region this rank holds.
    shard_shape_resampled : tuple of int
        What the rank emits after subsampling. Reported rather than recomputed:
        the loader takes a strided slice of a region that may itself start at an
        offset, which is not always ``ceil(shard.shape / subsampling_factor)``.
    subsampling_factor : int
        Decimation applied during the read. When the downsampling layer replaces
        it this becomes 1 and the ``_resampled`` names go with it.
    grid_converter : optional
        Converts the data grid to the model grid. Carried here because the
        trainers need it and it is settled at the same time as the rest.
    grid : optional
        The backend's ``GridSpec``, where the loader has one. An unstructured
        dataset has no rows and columns, so the flattened accessors raise for it
        rather than inventing a shape.
    """

    in_channels: Sequence[int]
    out_channels: Sequence[int]

    grid_shape: Tuple[int, int]
    crop_shape: Tuple[int, int]
    crop_offset: Tuple[int, int]
    shard: Shard
    shard_shape_resampled: Tuple[int, int]

    subsampling_factor: int = 1
    grid_converter: Any = None
    grid: Any = None

    # ---- derived -----------------------------------------------------------

    @property
    def is_structured(self) -> bool:
        """Whether this describes a raster rather than a mesh."""
        return self.grid is None or getattr(self.grid, "is_structured", True)

    @property
    def grid_shape_resampled(self) -> Tuple[int, int]:
        """The full field after subsampling."""
        return (
            math.ceil(self.grid_shape[0] / self.subsampling_factor),
            math.ceil(self.grid_shape[1] / self.subsampling_factor),
        )

    @property
    def lat_lon_local(self) -> Tuple[List[float], List[float]]:
        """This rank's coordinates, under the name the trainers use."""
        return (self.shard.lats, self.shard.lons)

    # ---- the flattened names params is written in --------------------------

    def _row_column(self, name: str, values, index: int):
        """One component of a raster quantity, or a refusal for a mesh."""
        if not self.is_structured:
            raise AttributeError(
                f"'{name}' is a row or column of a raster, which this dataset does not have: it is stored on "
                f"a {getattr(self.grid, 'kind', 'non raster')} grid. Use grid_shape, shard, or the grid itself."
            )
        return values[index]

    @property
    def img_shape_x(self) -> int:
        return self._row_column("img_shape_x", self.grid_shape, 0)

    @property
    def img_shape_y(self) -> int:
        return self._row_column("img_shape_y", self.grid_shape, 1)

    @property
    def img_crop_shape_x(self) -> int:
        return self._row_column("img_crop_shape_x", self.crop_shape, 0)

    @property
    def img_crop_shape_y(self) -> int:
        return self._row_column("img_crop_shape_y", self.crop_shape, 1)

    @property
    def img_crop_offset_x(self) -> int:
        return self._row_column("img_crop_offset_x", self.crop_offset, 0)

    @property
    def img_crop_offset_y(self) -> int:
        return self._row_column("img_crop_offset_y", self.crop_offset, 1)

    @property
    def img_local_shape_x(self) -> int:
        return self._row_column("img_local_shape_x", self.shard.shape, 0)

    @property
    def img_local_shape_y(self) -> int:
        return self._row_column("img_local_shape_y", self.shard.shape, 1)

    @property
    def img_local_offset_x(self) -> int:
        return self._row_column("img_local_offset_x", (self.shard.lat_offset, self.shard.lon_offset), 0)

    @property
    def img_local_offset_y(self) -> int:
        return self._row_column("img_local_offset_y", (self.shard.lat_offset, self.shard.lon_offset), 1)

    @property
    def img_local_shape_x_resampled(self) -> int:
        return self._row_column("img_local_shape_x_resampled", self.shard_shape_resampled, 0)

    @property
    def img_local_shape_y_resampled(self) -> int:
        return self._row_column("img_local_shape_y_resampled", self.shard_shape_resampled, 1)

    @property
    def img_shape_x_resampled(self) -> int:
        return self._row_column("img_shape_x_resampled", self.grid_shape_resampled, 0)

    @property
    def img_shape_y_resampled(self) -> int:
        return self._row_column("img_shape_y_resampled", self.grid_shape_resampled, 1)

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_loader(cls, loader, grid: Optional[Any] = None, grid_converter: Optional[Any] = None) -> "DataShapes":
        """Read the geometry off a loader that reports it in tuples.

        The loaders settle the same quantities and used to name them
        differently; they all carry the tuple form now, so one reader serves
        every one of them and the spellings have nowhere to diverge again.

        The loader side says ``crop_size``/``crop_anchor``, which is what the
        backends and both sample sources call them; this type says
        ``crop_shape``/``crop_offset``, to pair with the shard. That one
        translation lives here rather than in each loader.
        """
        if grid is None:
            grid = getattr(getattr(loader, "backend", None), "chunk", None)
        if grid_converter is None:
            grid_converter = getattr(loader, "grid_converter", None)

        return cls(
            in_channels=loader.in_channels,
            out_channels=loader.out_channels,
            grid_shape=tuple(loader.img_shape),
            crop_shape=tuple(loader.crop_size),
            crop_offset=tuple(loader.crop_anchor),
            shard=Shard(
                shape=tuple(loader.read_shape),
                lat_offset=loader.read_anchor[0],
                lon_offset=loader.read_anchor[1],
                lats=loader.lat_lon_local[0],
                lons=loader.lat_lon_local[1],
            ),
            shard_shape_resampled=tuple(loader.return_shape),
            subsampling_factor=loader.subsampling_factor,
            grid_converter=grid_converter,
            grid=grid,
        )
