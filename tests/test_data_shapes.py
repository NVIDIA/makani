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

"""
Unit tests for the shape descriptor the loaders report.

What used to be three hand-built ``types.SimpleNamespace`` blocks is one type,
so the mapping from a loader's geometry to the names ``params`` carries is now
somewhere it can be checked. The failure this guards against is quiet: a field
crossed with its neighbour -- a crop offset read as a shape, a local extent read
as a global one -- leaves every shape plausible and the run subtly wrong.

Needs no DALI, no GPU and no dataset.
"""

import os
import sys
import types
import unittest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.dataloaders.data_shapes import DataShapes, Shard


def make_loader(**overrides):
    """A stand-in reporting the tuples every real loader reports."""
    loader = types.SimpleNamespace(
        in_channels=[0, 1, 2],
        out_channels=[0, 1],
        img_shape=(64, 128),
        crop_size=(32, 96),
        crop_anchor=(4, 8),
        read_shape=(16, 96),
        read_anchor=(20, 8),
        return_shape=(8, 48),
        subsampling_factor=2,
        lat_lon_local=([1.0, 2.0], [3.0, 4.0]),
        grid_converter="converter",
    )
    for name, value in overrides.items():
        setattr(loader, name, value)
    return loader


class TestFromLoader(unittest.TestCase):
    """Reading a loader's geometry into the descriptor."""

    def setUp(self):
        self.shapes = DataShapes.from_loader(make_loader())

    def test_every_field_comes_from_the_right_place(self):
        """The whole mapping, asserted field by field.

        Each of these was a line in a namespace literal, and a namespace literal
        is exactly where a crossed pair hides -- the shapes stay plausible and
        nothing downstream can tell.
        """
        self.assertEqual(self.shapes.grid_shape, (64, 128))
        self.assertEqual(self.shapes.crop_shape, (32, 96))
        self.assertEqual(self.shapes.crop_offset, (4, 8))
        self.assertEqual(self.shapes.shard.shape, (16, 96))
        self.assertEqual(self.shapes.shard.lat_offset, 20)
        self.assertEqual(self.shapes.shard.lon_offset, 8)
        self.assertEqual(self.shapes.shard_shape_resampled, (8, 48))
        self.assertEqual(self.shapes.subsampling_factor, 2)

    def test_channels_are_carried_in_order(self):
        self.assertEqual(list(self.shapes.in_channels), [0, 1, 2])
        self.assertEqual(list(self.shapes.out_channels), [0, 1])

    def test_the_grid_converter_is_carried(self):
        self.assertEqual(self.shapes.grid_converter, "converter")

    def test_an_override_wins_over_the_loader(self):
        # the DALI loader holds the converter while its sample source holds the
        # geometry, so the two come from different objects
        shapes = DataShapes.from_loader(make_loader(), grid_converter="other")
        self.assertEqual(shapes.grid_converter, "other")

    def test_the_backend_grid_is_picked_up(self):
        grid = types.SimpleNamespace(is_structured=True, kind="equiangular")
        loader = make_loader(backend=types.SimpleNamespace(chunk=grid))

        self.assertIs(DataShapes.from_loader(loader).grid, grid)

    def test_a_loader_without_a_backend_has_no_grid(self):
        # the synthetic loader has none, and reports a raster all the same
        shapes = DataShapes.from_loader(make_loader())

        self.assertIsNone(shapes.grid)
        self.assertTrue(shapes.is_structured)


class TestFlattenedNames(unittest.TestCase):
    """The ``img_shape_x`` surface ``params`` is written in.

    These names are the ones configs and checkpoints carry, so each has to keep
    meaning what it meant. Asserted individually rather than in a loop, since
    what is being checked is precisely which component each one names.
    """

    def setUp(self):
        self.shapes = DataShapes.from_loader(make_loader())

    def test_global_shape(self):
        self.assertEqual(self.shapes.img_shape_x, 64)
        self.assertEqual(self.shapes.img_shape_y, 128)

    def test_crop(self):
        self.assertEqual(self.shapes.img_crop_shape_x, 32)
        self.assertEqual(self.shapes.img_crop_shape_y, 96)
        self.assertEqual(self.shapes.img_crop_offset_x, 4)
        self.assertEqual(self.shapes.img_crop_offset_y, 8)

    def test_this_rank(self):
        self.assertEqual(self.shapes.img_local_shape_x, 16)
        self.assertEqual(self.shapes.img_local_shape_y, 96)
        self.assertEqual(self.shapes.img_local_offset_x, 20)
        self.assertEqual(self.shapes.img_local_offset_y, 8)

    def test_resampled(self):
        self.assertEqual(self.shapes.img_local_shape_x_resampled, 8)
        self.assertEqual(self.shapes.img_local_shape_y_resampled, 48)
        self.assertEqual(self.shapes.img_shape_x_resampled, 32)
        self.assertEqual(self.shapes.img_shape_y_resampled, 64)

    def test_the_global_resampled_shape_rounds_up(self):
        # a field that does not divide evenly still emits the partial row
        shapes = DataShapes.from_loader(make_loader(img_shape=(65, 128), subsampling_factor=2))
        self.assertEqual(shapes.grid_shape_resampled, (33, 64))

    def test_the_local_resampled_shape_is_reported_not_derived(self):
        """A strided read of an offset region is not ceil(shape / factor).

        Where the rank's block starts decides whether it catches the first
        sample of the stride, so the loader reports what it will emit rather
        than having it recomputed from a rule that is right only sometimes.
        """
        shapes = DataShapes.from_loader(make_loader(read_shape=(17, 96), return_shape=(8, 48)))
        self.assertEqual(shapes.img_local_shape_x_resampled, 8)

    def test_lat_lon_local_is_the_shard_coordinates(self):
        self.assertEqual(self.shapes.lat_lon_local, ([1.0, 2.0], [3.0, 4.0]))


class TestUnstructured(unittest.TestCase):
    """A mesh has no rows and columns, and says so.

    ICON emits an unstructured grid, where ``img_shape_x`` has no meaning. The
    accessors refuse rather than return the first element of something, because
    a plausible number here becomes a silently wrong model input.
    """

    def setUp(self):
        grid = types.SimpleNamespace(is_structured=False, kind="unstructured")
        self.shapes = DataShapes.from_loader(make_loader(backend=types.SimpleNamespace(chunk=grid)))

    def test_it_is_not_structured(self):
        self.assertFalse(self.shapes.is_structured)

    def test_the_flattened_accessors_refuse(self):
        for name in (
            "img_shape_x",
            "img_shape_y",
            "img_crop_shape_x",
            "img_local_shape_x",
            "img_local_offset_y",
            "img_shape_x_resampled",
            "img_local_shape_y_resampled",
        ):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError) as ctx:
                    getattr(self.shapes, name)
                self.assertIn("unstructured", str(ctx.exception))

    def test_the_structured_free_parts_still_work(self):
        # what a mesh can answer, it still answers
        self.assertEqual(list(self.shapes.in_channels), [0, 1, 2])
        self.assertEqual(self.shapes.lat_lon_local, ([1.0, 2.0], [3.0, 4.0]))


class TestShard(unittest.TestCase):
    """Named after ``torch_harmonics.GridShardS2``, and meaning the same."""

    def test_nlat_and_nlon_are_the_shape(self):
        shard = Shard(shape=(12, 34), lat_offset=1, lon_offset=2)

        self.assertEqual(shard.nlat, 12)
        self.assertEqual(shard.nlon, 34)

    def test_coordinates_default_to_empty(self):
        shard = Shard(shape=(2, 2), lat_offset=0, lon_offset=0)

        self.assertEqual(shard.lats, [])
        self.assertEqual(shard.lons, [])


class TestImmutability(unittest.TestCase):

    def test_the_descriptor_cannot_be_edited_in_place(self):
        """Settled once, at construction.

        The namespace it replaces was writable, so anything holding it could
        change what a later reader saw. Freezing it means the shapes a run is
        configured with are the shapes the loader reported.
        """
        shapes = DataShapes.from_loader(make_loader())

        with self.assertRaises(Exception):
            shapes.grid_shape = (1, 1)


if __name__ == "__main__":
    unittest.main()
