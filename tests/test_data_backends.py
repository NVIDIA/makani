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
Unit tests for the storage backends.

``test_sample_source.py`` drives all four backends end to end through
``SampleSource``, which covers the read path thoroughly. What it cannot see are
the properties that hold *between* the pieces, and those are what this file
pins:

* **coordinates match the data.** A backend promises that ``chunk.lat`` and
  ``chunk.lon`` give the position of every value it emits. Nothing downstream
  checks that, and getting it wrong stays invisible until something resamples
  with those coordinates and produces a subtly displaced field.
* **layout detection.** New logic replacing checks that used to be scattered
  across the loader; the end to end tests exercise one happy path each and none
  of the failure modes.
* **contiguous_slices.** A broken version still returns correct data, just in
  many small reads instead of a few large ones, so no behavioural test can see
  it -- only a profile can.
* **pickling.** ``test_pickle_roundtrip`` covers handles, but only for local
  files; the connection a remote backend holds is dropped by different
  machinery.

These need neither DALI nor a GPU, so unlike the end to end suite they run in
CI.
"""

import datetime as dt
import os
import pickle
import sys
import tempfile
import unittest

import h5py as h5
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.dataloaders.backends import (
    ArcoWB2Backend,
    BackendMetadata,
    MakaniConcatBackend,
    MakaniHDF5Backend,
    MakaniZarrBackend,
    get_backend,
)
from makani.utils.dataloaders.backends.base import GridSpec
from makani.utils.dataloaders.backends.factory import detect_backend
from makani.utils.dataloaders.backends.makani_hdf5 import contiguous_slices

from .testutils import (
    CHANNEL_NAMES,
    DHOURS,
    NUM_SAMPLES_PER_YEAR,
    TRAIN_YEARS,
    H5_PATH,
    IMG_SIZE_H,
    IMG_SIZE_W,
    NUM_CHANNELS,
    init_hdf5_dataset,
    init_wb2_zarr_dataset,
    init_zarr_dataset,
)


class TestContiguousSlices(unittest.TestCase):
    """
    Channel selections are read as runs of adjacent indices rather than one
    element at a time. A regression here is silent: the data stays correct and
    only the number of reads changes, so it is asserted directly.
    """

    def test_single_run(self):
        self.assertEqual(list(contiguous_slices([0, 1, 2])), [slice(0, 3)])

    def test_two_runs(self):
        self.assertEqual(list(contiguous_slices([0, 1, 2, 7, 8])), [slice(0, 3), slice(7, 9)])

    def test_isolated_indices_do_not_merge(self):
        self.assertEqual(list(contiguous_slices([0, 2, 4])), [slice(0, 1), slice(2, 3), slice(4, 5)])

    def test_single_index(self):
        self.assertEqual(list(contiguous_slices([5])), [slice(5, 6)])

    def test_empty(self):
        self.assertEqual(list(contiguous_slices([])), [])

    def test_slices_cover_exactly_the_input(self):
        indices = [1, 2, 3, 9, 10, 40]
        covered = [i for s in contiguous_slices(indices) for i in range(s.start, s.stop)]
        self.assertEqual(covered, indices)


class TestGridSpec(unittest.TestCase):

    def test_raster_is_structured(self):
        grid = GridSpec("equiangular", (4, 8), np.zeros(4), np.zeros(8))
        self.assertTrue(grid.is_structured)

    def test_mesh_is_not_structured(self):
        # what an unstructured backend would report: one spatial dimension, and
        # a coordinate per cell rather than per axis
        grid = GridSpec("unstructured", (32,), np.zeros(32), np.zeros(32))
        self.assertFalse(grid.is_structured)


class _BackendFixture(unittest.TestCase):
    """A small on-disk dataset shared by the tests below."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        (
            cls.train_path,
            cls.num_train,
            cls.valid_path,
            cls.num_valid,
            cls.stats_path,
            cls.metadata_path,
            cls.concat_path,
        ) = init_hdf5_dataset(cls.tmpdir.name, create_concat=True)

        cls.zarr_dir = tempfile.TemporaryDirectory()
        cls.zarr_train_path = init_zarr_dataset(cls.zarr_dir.name)[0]

        cls.wb2_dir = tempfile.TemporaryDirectory()
        cls.wb2_train_path = init_wb2_zarr_dataset(cls.wb2_dir.name)[0]

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()
        cls.zarr_dir.cleanup()
        cls.wb2_dir.cleanup()


class TestDetectBackend(_BackendFixture):
    """
    Which layout is at a path used to be decided in two places, by an
    ``os.path.isfile`` in the loader and by the order of two globs in the
    reader. It is one function now, so its decisions are worth stating.
    """

    def test_directory_of_h5_files(self):
        self.assertEqual(detect_backend(self.train_path), "makani_hdf5")

    def test_single_file_is_concatenated(self):
        self.assertEqual(detect_backend(self.concat_path), "makani_concat")

    def test_directory_of_zarr_stores(self):
        self.assertEqual(detect_backend(self.zarr_train_path), "makani_zarr")

    def test_s3_implies_hdf5(self):
        # nothing local to stat, and the per-year HDF5 layout is the only one
        # reachable over the ROS3 driver
        self.assertEqual(detect_backend("s3://bucket/prefix", enable_s3=True), "makani_hdf5")

    def test_hdf5_wins_over_zarr_in_a_mixed_directory(self):
        # a directory holding both is ambiguous; the order is a decision, so it
        # is pinned rather than left to whichever glob happens to run first
        mixed = os.path.join(self.tmpdir.name, "mixed")
        os.makedirs(mixed, exist_ok=True)
        for name in ("2017.h5", "2018.h5"):
            with h5.File(os.path.join(mixed, name), "w") as handle:
                handle.create_dataset(H5_PATH, data=np.zeros((2, NUM_CHANNELS, 4, 4), dtype=np.float32))
        os.makedirs(os.path.join(mixed, "2019.zarr"), exist_ok=True)

        self.assertEqual(detect_backend(mixed), "makani_hdf5")

    def test_wb2_store_is_told_apart_from_the_makani_one(self):
        # the two share a container and a naming convention, so this is the one
        # branch that has to open a store to decide: a WB2 store has no single
        # fields array, its variables are named individually
        self.assertEqual(detect_backend(self.wb2_train_path), "arco_wb2")

    def test_wb2_backend_is_built_for_a_wb2_store(self):
        backend = get_backend(self.wb2_train_path, channel_names=list(CHANNEL_NAMES))
        self.assertIsInstance(backend, ArcoWB2Backend)

    def test_wb2_store_without_channel_names_raises(self):
        # detection succeeds and construction then fails, which is the right
        # order: the message names what is missing rather than what was not found
        with self.assertRaises(ValueError) as ctx:
            get_backend(self.wb2_train_path)
        self.assertIn("channel_names", str(ctx.exception))

    def test_hdf5_only_arguments_are_not_passed_to_other_backends(self):
        # O_DIRECT and S3 are HDF5 driver settings; a zarr backend has no use for
        # them and would reject them as unexpected keywords
        backend = get_backend(self.zarr_train_path, enable_odirect=True, odirect_alignment=4096, enable_s3=False)
        self.assertIsInstance(backend, MakaniZarrBackend)

    def test_odirect_and_s3_together_are_rejected(self):
        with self.assertRaises(NotImplementedError):
            get_backend(self.train_path, enable_odirect=True, enable_s3=True)

    def test_empty_directory_raises(self):
        empty = os.path.join(self.tmpdir.name, "empty")
        os.makedirs(empty, exist_ok=True)
        with self.assertRaises(IOError):
            detect_backend(empty)

    def test_explicit_backend_skips_detection(self):
        backend = get_backend(self.train_path, backend="makani_hdf5")
        self.assertIsInstance(backend, MakaniHDF5Backend)

    def test_unknown_backend_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_backend(self.train_path, backend="not_a_backend")
        self.assertIn("not_a_backend", str(ctx.exception))

    def test_factory_detects_each_layout(self):
        for location, expected in [
            (self.train_path, MakaniHDF5Backend),
            (self.concat_path, MakaniConcatBackend),
            (self.zarr_train_path, MakaniZarrBackend),
        ]:
            with self.subTest(location=os.path.basename(location)):
                self.assertIsInstance(get_backend(location), expected)


class TestDiscovery(_BackendFixture):

    def test_metadata_describes_the_dataset(self):
        backend = get_backend(self.train_path)
        metadata = backend.discover()

        self.assertIsInstance(metadata, BackendMetadata)
        self.assertEqual(len(metadata.files), len(metadata.samples_per_file))
        self.assertEqual(len(metadata.files), len(metadata.labels))
        self.assertEqual(sum(metadata.samples_per_file), len(metadata.timestamps))
        self.assertEqual(metadata.grid.shape, (IMG_SIZE_H, IMG_SIZE_W))
        self.assertEqual(metadata.total_channels, NUM_CHANNELS)

    def test_timestamps_are_utc_and_ordered(self):
        metadata = get_backend(self.train_path).discover()

        self.assertIsNotNone(metadata.timestamps[0].tzinfo)
        self.assertTrue(all(a < b for a, b in zip(metadata.timestamps, metadata.timestamps[1:])))

    def test_labels_are_the_years(self):
        metadata = get_backend(self.train_path).discover()

        self.assertEqual(metadata.labels, sorted(metadata.labels))

        # timestamps run over every sample, not every file, so a label has to be
        # checked against the first sample of its own file
        offset = 0
        for label, count in zip(metadata.labels, metadata.samples_per_file):
            with self.subTest(label=label):
                self.assertEqual(label, metadata.timestamps[offset].year)
                self.assertEqual(label, metadata.timestamps[offset + count - 1].year)
            offset += count

    def test_concatenated_dataset_reports_one_file(self):
        metadata = get_backend(self.concat_path).discover()

        self.assertEqual(len(metadata.files), 1)
        self.assertEqual(metadata.samples_per_file[0], len(metadata.timestamps))


class TestChunkCoordinates(_BackendFixture):
    """
    The promise the whole contract rests on: a backend says where every value it
    emits belongs. A resampler consumes those coordinates, so an off-by-one
    between them and the data would not fail here, it would displace a field
    later.
    """

    def _backend(self, **kwargs):
        backend = get_backend(self.train_path, **kwargs)
        backend.discover()
        return backend

    def test_full_grid_chunk_matches_the_dataset(self):
        backend = self._backend()

        self.assertEqual(backend.chunk.shape, (IMG_SIZE_H, IMG_SIZE_W))
        np.testing.assert_allclose(backend.chunk.lat, backend.metadata.grid.lat)
        np.testing.assert_allclose(backend.chunk.lon, backend.metadata.grid.lon)

    def test_coordinate_lengths_match_the_chunk_shape(self):
        for io_grid, io_rank in [([1, 1], [0, 0]), ([2, 1], [1, 0]), ([2, 2], [1, 1]), ([3, 1], [2, 0])]:
            with self.subTest(io_grid=io_grid, io_rank=io_rank):
                backend = self._backend(io_grid=io_grid, io_rank=io_rank)

                self.assertEqual(backend.chunk.lat.shape[0], backend.chunk.shape[0])
                self.assertEqual(backend.chunk.lon.shape[0], backend.chunk.shape[1])

    def test_coordinates_are_the_slice_of_the_global_ones(self):
        # the emitted block has to carry exactly the coordinates of the region
        # it was cut from, which is what makes them usable for resampling
        backend = self._backend(io_grid=[2, 2], io_rank=[1, 1])
        grid = backend.metadata.grid

        lat_start = backend.read_anchor[0]
        lon_start = backend.read_anchor[1]
        np.testing.assert_allclose(backend.chunk.lat, grid.lat[lat_start : lat_start + backend.read_shape[0]])
        np.testing.assert_allclose(backend.chunk.lon, grid.lon[lon_start : lon_start + backend.read_shape[1]])

    def test_chunks_tile_the_grid_without_gaps_or_overlap(self):
        # every latitude of the global grid is claimed by exactly one rank
        claimed = []
        for rank in range(3):
            backend = self._backend(io_grid=[3, 1], io_rank=[rank, 0])
            claimed.append(backend.chunk.lat)

        np.testing.assert_allclose(np.concatenate(claimed), backend.metadata.grid.lat)

    def test_data_at_a_coordinate_is_the_data_at_that_coordinate(self):
        """Read a decomposed chunk and check it against the same region read whole.

        This is the assertion the contract needs: the values a rank emits, and
        the coordinates it reports for them, describe the same points as the
        global grid does.
        """
        whole = self._backend()
        part = self._backend(io_grid=[2, 2], io_rank=[1, 0])

        channels = np.arange(NUM_CHANNELS)
        reference = whole.read(0, slice(0, 1), channels)
        chunk = part.read(0, slice(0, 1), channels)

        # locate the chunk's coordinates within the global axes, then compare
        lat_offset = int(np.argmin(np.abs(whole.chunk.lat - part.chunk.lat[0])))
        lon_offset = int(np.argmin(np.abs(whole.chunk.lon - part.chunk.lon[0])))
        expected = reference[
            :,
            :,
            lat_offset : lat_offset + part.chunk.shape[0],
            lon_offset : lon_offset + part.chunk.shape[1],
        ]

        np.testing.assert_allclose(chunk, expected)

    def test_subsampling_takes_every_nth_coordinate(self):
        backend = self._backend(subsampling_factor=2)

        np.testing.assert_allclose(backend.chunk.lat, backend.metadata.grid.lat[::2])
        np.testing.assert_allclose(backend.chunk.lon, backend.metadata.grid.lon[::2])

    def test_crop_decomposition_and_subsampling_compose(self):
        """All three narrow the chunk, and they are applied in one place.

        Tested separately elsewhere; here they act together, which is where an
        interaction bug would live -- an anchor that ignores the subsampling
        step, say, or a split taken before the crop rather than after.
        """
        crop_anchor, crop_size = [2, 4], [32, 32]
        backend = self._backend(
            crop_anchor=crop_anchor,
            crop_size=crop_size,
            io_grid=[2, 1],
            io_rank=[1, 0],
            subsampling_factor=2,
        )
        grid = backend.metadata.grid

        # the crop is split over two ranks and this is the second, so it starts
        # half a crop in; subsampling then takes every other row of that
        expected_start = crop_anchor[0] + crop_size[0] // 2
        expected_lat = grid.lat[expected_start : crop_anchor[0] + crop_size[0] : 2]
        expected_lon = grid.lon[crop_anchor[1] : crop_anchor[1] + crop_size[1] : 2]

        np.testing.assert_allclose(backend.chunk.lat, expected_lat)
        np.testing.assert_allclose(backend.chunk.lon, expected_lon)
        self.assertEqual(backend.chunk.shape, (len(expected_lat), len(expected_lon)))

    def test_crop_beyond_the_grid_raises(self):
        with self.assertRaises(ValueError):
            self._backend(crop_anchor=[0, 0], crop_size=[IMG_SIZE_H + 1, IMG_SIZE_W])


class TestLifecycle(_BackendFixture):

    def test_num_files_before_discovery_raises(self):
        # asking what is in a dataset before looking is a caller error, and a
        # clear one beats an AttributeError on None
        backend = get_backend(self.train_path)
        with self.assertRaises(RuntimeError) as ctx:
            backend.num_files
        self.assertIn("discover", str(ctx.exception))

    def test_num_files_after_discovery(self):
        backend = get_backend(self.train_path)
        metadata = backend.discover()
        self.assertEqual(backend.num_files, len(metadata.files))

    def test_close_releases_handles(self):
        backend = get_backend(self.train_path)
        backend.discover()
        backend.open(0)
        self.assertIsNotNone(backend.files[0])

        backend.close()

        self.assertTrue(all(handle is None for handle in backend.files))
        self.assertTrue(all(handle is None for handle in backend.dsets))

    def test_close_is_idempotent(self):
        # __del__ calls it, so a second call during interpreter shutdown must
        # not raise
        backend = get_backend(self.train_path)
        backend.discover()
        backend.close()
        backend.close()

    def test_reopening_after_close_works(self):
        backend = get_backend(self.train_path)
        backend.discover()
        channels = np.arange(NUM_CHANNELS)

        before = backend.read(0, slice(0, 1), channels)
        backend.close()
        after = backend.read(0, slice(0, 1), channels)

        np.testing.assert_allclose(before, after)


class TestConcatBackend(_BackendFixture):

    def test_unannotated_file_is_rejected(self):
        """A concatenated file has to carry its timestamps.

        The per-year layout can fall back on the year in the filename; one file
        holding every year cannot, so the times have to be read. Failing at
        discovery with a message naming the fix beats a KeyError from h5py deep
        in a run.
        """
        path = os.path.join(self.tmpdir.name, "unannotated.h5")
        with h5.File(path, "w") as handle:
            handle.create_dataset(H5_PATH, data=np.zeros((4, NUM_CHANNELS, 8, 16), dtype=np.float32))

        backend = get_backend(path)
        with self.assertRaises(ValueError) as ctx:
            backend.discover()
        self.assertIn("annotate", str(ctx.exception))


class TestPickling(_BackendFixture):
    """
    A backend crosses into a worker process by being pickled, so anything it
    holds that cannot cross has to be dropped and rebuilt.
    """

    def test_handles_are_dropped(self):
        backend = get_backend(self.train_path)
        backend.discover()
        backend.open(0)
        self.assertIsNotNone(backend.files[0])

        state = backend.__getstate__()

        self.assertTrue(all(handle is None for handle in state["files"]))
        self.assertTrue(all(handle is None for handle in state["dsets"]))

    def test_transient_attributes_are_dropped(self):
        # the S3 connector holds a session; unlike a file handle it is not in a
        # list, so it needs its own declaration and would otherwise be pickled
        backend = get_backend(self.train_path)
        backend.discover()
        backend.aws_connector = object()

        self.assertIsNone(backend.__getstate__()["aws_connector"])

    def test_roundtrip_reads_the_same_data(self):
        backend = get_backend(self.train_path)
        backend.discover()
        channels = np.arange(NUM_CHANNELS)

        before = backend.read(0, slice(0, 2), channels)
        restored = pickle.loads(pickle.dumps(backend))
        after = restored.read(0, slice(0, 2), channels)

        np.testing.assert_allclose(before, after)

    def test_roundtrip_preserves_the_chunk(self):
        backend = get_backend(self.train_path, io_grid=[2, 2], io_rank=[0, 1])
        backend.discover()

        restored = pickle.loads(pickle.dumps(backend))

        self.assertEqual(restored.chunk.shape, backend.chunk.shape)
        np.testing.assert_allclose(restored.chunk.lat, backend.chunk.lat)
        np.testing.assert_allclose(restored.chunk.lon, backend.chunk.lon)


class TestReading(_BackendFixture):

    def test_out_is_filled_in_place(self):
        backend = get_backend(self.train_path)
        backend.discover()
        channels = np.arange(NUM_CHANNELS)

        out = np.zeros((2, NUM_CHANNELS, *backend.chunk.shape), dtype=np.float32)
        returned = backend.read(0, slice(0, 2), channels, out=out)

        self.assertIs(returned, out)
        self.assertTrue(np.any(out != 0.0))

    def test_allocating_and_filling_agree(self):
        backend = get_backend(self.train_path)
        backend.discover()
        channels = np.arange(NUM_CHANNELS)

        allocated = backend.read(0, slice(0, 2), channels)
        out = np.zeros_like(allocated)
        backend.read(0, slice(0, 2), channels, out=out)

        np.testing.assert_allclose(allocated, out)

    def test_channel_subset_is_read_in_order(self):
        backend = get_backend(self.train_path)
        backend.discover()

        every = backend.read(0, slice(0, 1), np.arange(NUM_CHANNELS))
        subset = backend.read(0, slice(0, 1), np.array([0, 2]))

        np.testing.assert_allclose(subset[:, 0], every[:, 0])
        np.testing.assert_allclose(subset[:, 1], every[:, 2])

    def test_strided_time_slice(self):
        backend = get_backend(self.train_path)
        backend.discover()
        channels = np.arange(NUM_CHANNELS)

        strided = backend.read(0, slice(0, 4, 2), channels)
        first = backend.read(0, slice(0, 1), channels)
        third = backend.read(0, slice(2, 3), channels)

        self.assertEqual(strided.shape[0], 2)
        np.testing.assert_allclose(strided[0], first[0])
        np.testing.assert_allclose(strided[1], third[0])


class TestTimestampSynthesis(unittest.TestCase):
    """Timestamps for a dataset that does not carry any.

    An unannotated file has only the year in its name to go on, so the sample
    times are derived from that year plus the ``dhours`` cadence. This is the
    one place in the backends that does date arithmetic, it runs before anything
    else can notice it went wrong, and a mistake here misdates every sample --
    which matters, because the timestamps drive both the zenith angle and the
    boundary exclusions.

    The strongest statement available is that synthesis and annotation agree:
    the same dataset written both ways has to produce the same times.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()

        cls.h5_annotated = init_hdf5_dataset(os.path.join(cls.tmpdir.name, "h5_annotated"))[0]
        cls.h5_bare = init_hdf5_dataset(os.path.join(cls.tmpdir.name, "h5_bare"), annotate=False)[0]
        cls.zarr_annotated = init_zarr_dataset(os.path.join(cls.tmpdir.name, "zarr_annotated"))[0]
        cls.zarr_bare = init_zarr_dataset(os.path.join(cls.tmpdir.name, "zarr_bare"), annotate=False)[0]

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _discover(self, path):
        return get_backend(path, dhours=DHOURS).discover()

    def test_hdf5_synthesizes_one_timestamp_per_sample(self):
        metadata = self._discover(self.h5_bare)
        self.assertEqual(len(metadata.timestamps), sum(metadata.samples_per_file))

    def test_zarr_synthesizes_one_timestamp_per_sample(self):
        metadata = self._discover(self.zarr_bare)
        self.assertEqual(len(metadata.timestamps), sum(metadata.samples_per_file))

    def test_hdf5_synthesized_times_match_the_annotated_ones(self):
        expected = self._discover(self.h5_annotated).timestamps
        synthesized = self._discover(self.h5_bare).timestamps

        self.assertEqual(len(synthesized), len(expected))
        for idx, (got, want) in enumerate(zip(synthesized, expected)):
            with self.subTest(sample=idx):
                self.assertEqual(got, want)

    def test_zarr_synthesized_times_match_the_annotated_ones(self):
        expected = self._discover(self.zarr_annotated).timestamps
        synthesized = self._discover(self.zarr_bare).timestamps

        self.assertEqual(len(synthesized), len(expected))
        for idx, (got, want) in enumerate(zip(synthesized, expected)):
            with self.subTest(sample=idx):
                self.assertEqual(got, want)

    def test_synthesis_starts_at_new_year_utc(self):
        # the year in the filename means midnight on the first of January, in
        # UTC and not in whatever zone the machine happens to sit in
        metadata = self._discover(self.h5_bare)
        first = metadata.timestamps[0]

        self.assertEqual((first.year, first.month, first.day), (TRAIN_YEARS[0], 1, 1))
        self.assertEqual((first.hour, first.minute, first.second), (0, 0, 0))
        self.assertEqual(first.utcoffset(), dt.timedelta(0))

    def test_synthesis_advances_by_dhours(self):
        metadata = self._discover(self.h5_bare)
        step = dt.timedelta(hours=DHOURS)

        for idx in range(1, 5):
            with self.subTest(sample=idx):
                self.assertEqual(metadata.timestamps[idx] - metadata.timestamps[idx - 1], step)

    def test_synthesis_restarts_each_file(self):
        # every file is dated from its own year, so the cadence does not carry
        # over a file boundary -- the next file starts at its own new year
        metadata = self._discover(self.h5_bare)
        boundary = metadata.timestamps[NUM_SAMPLES_PER_YEAR]

        self.assertEqual((boundary.year, boundary.month, boundary.day), (TRAIN_YEARS[1], 1, 1))
        self.assertEqual(boundary.hour, 0)


class TestUnconsolidatedZarr(unittest.TestCase):
    """A store whose metadata was never consolidated.

    Consolidation is an optimisation, not a requirement, and a store written
    without it has no ``zarr.json`` at the root to open. The backend falls back
    to opening the group directly; without that, a perfectly valid store is
    unreadable.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.consolidated = init_zarr_dataset(os.path.join(cls.tmpdir.name, "consolidated"))[0]
        cls.plain = init_zarr_dataset(os.path.join(cls.tmpdir.name, "plain"), consolidate=False)[0]

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_discovery_agrees_with_the_consolidated_store(self):
        expected = get_backend(self.consolidated).discover()
        metadata = get_backend(self.plain).discover()

        self.assertEqual(metadata.samples_per_file, expected.samples_per_file)
        self.assertEqual(metadata.labels, expected.labels)
        self.assertEqual(metadata.total_channels, expected.total_channels)
        self.assertEqual(metadata.grid.shape, expected.grid.shape)

    def test_reads_agree_with_the_consolidated_store(self):
        channels = np.arange(NUM_CHANNELS)

        reference = get_backend(self.consolidated)
        reference.discover()
        backend = get_backend(self.plain)
        backend.discover()

        np.testing.assert_allclose(backend.read(0, slice(0, 3), channels), reference.read(0, slice(0, 3), channels))


def _write_h5(path, data, timestamps, latitude=None, longitude=None):
    """A minimal makani-layout file, annotated as much as asked for."""
    with h5.File(path, "w") as handle:
        dset = handle.create_dataset(H5_PATH, data=data)
        if timestamps is not None:
            scale = handle.create_dataset("timestamp", data=np.asarray(timestamps, dtype=np.float64))
            scale.make_scale("timestamp")
            dset.dims[0].attach_scale(scale)
        if latitude is not None:
            handle.create_dataset("lat", data=np.asarray(latitude, dtype=np.float32))
            handle["lat"].make_scale("lat")
            dset.dims[2].attach_scale(handle["lat"])
            handle.create_dataset("lon", data=np.asarray(longitude, dtype=np.float32))
            handle["lon"].make_scale("lon")
            dset.dims[3].attach_scale(handle["lon"])
    return path


class TestTimeOrderedDiscovery(unittest.TestCase):
    """Files are ordered by the data, not by the name.

    The makani layouts name a file after its year, so sorting by name orders
    them by time as well. That is a property of those layouts, not of datasets
    in general -- an inference dataset can be a directory of arbitrarily named
    files -- so the order comes from the first sample of each file. Getting it
    wrong silently misdates every sample after the first file.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_part(self, name, start_day, n_samples=4):
        base = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=start_day)
        timestamps = [(base + dt.timedelta(days=idx)).timestamp() for idx in range(n_samples)]
        data = np.full((n_samples, NUM_CHANNELS, 4, 8), float(start_day), dtype=np.float32)
        return _write_h5(os.path.join(self.tmpdir.name, name), data, timestamps)

    def test_files_are_ordered_by_time_not_by_name(self):
        # names deliberately sort the other way round from the timestamps
        self._write_part("charlie.h5", start_day=0)
        self._write_part("bravo.h5", start_day=4)
        self._write_part("alpha.h5", start_day=8)

        metadata = get_backend(self.tmpdir.name, file_pattern="*").discover()

        self.assertEqual([os.path.basename(path) for path in metadata.files], ["charlie.h5", "bravo.h5", "alpha.h5"])
        self.assertTrue(
            all(metadata.timestamps[i] < metadata.timestamps[i + 1] for i in range(len(metadata.timestamps) - 1))
        )

    def test_samples_per_file_follows_the_same_order(self):
        # the sample index walks files in this order, so the counts have to be
        # permuted alongside them or every index past the first file is wrong
        self._write_part("charlie.h5", start_day=0, n_samples=2)
        self._write_part("alpha.h5", start_day=8, n_samples=5)

        metadata = get_backend(self.tmpdir.name, file_pattern="*").discover()

        self.assertEqual(metadata.samples_per_file, [2, 5])

    def test_overlapping_files_are_refused(self):
        # one file's range inside another's makes a global sample index
        # ambiguous, so it is refused rather than silently resolved
        self._write_part("first.h5", start_day=0, n_samples=10)
        self._write_part("inside.h5", start_day=2, n_samples=2)

        with self.assertRaises(RuntimeError) as ctx:
            get_backend(self.tmpdir.name, file_pattern="*").discover()
        self.assertIn("overlapping", str(ctx.exception))

    def test_a_file_that_is_not_a_year_needs_timestamps(self):
        # synthesis derives the times from the year in the name; without one
        # there is nothing to derive them from
        data = np.zeros((4, NUM_CHANNELS, 4, 8), dtype=np.float32)
        _write_h5(os.path.join(self.tmpdir.name, "part_one.h5"), data, timestamps=None)

        with self.assertRaises(ValueError) as ctx:
            get_backend(self.tmpdir.name, file_pattern="*").discover()
        self.assertIn("not a year", str(ctx.exception))

    def test_the_year_layout_is_unaffected(self):
        # name order and time order agree here, which is why the training path
        # sees no change from ordering by time
        for year in (2018, 2017):
            base = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
            timestamps = [(base + dt.timedelta(days=idx)).timestamp() for idx in range(3)]
            data = np.zeros((3, NUM_CHANNELS, 4, 8), dtype=np.float32)
            _write_h5(os.path.join(self.tmpdir.name, f"{year}.h5"), data, timestamps)

        metadata = get_backend(self.tmpdir.name).discover()

        self.assertEqual([os.path.basename(path) for path in metadata.files], ["2017.h5", "2018.h5"])
        self.assertEqual(metadata.labels, [2017, 2018])


class TestFileCoordinates(unittest.TestCase):
    """Where the grid comes from.

    A dataset that is not equiangular still has to describe itself correctly:
    the coordinates go to the resampler and to the solar zenith angle, and both
    give quietly wrong answers on a grid they were not told about. So the file
    is asked first, and the equiangular assumption is only the fallback for a
    file that carries no coordinates at all.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.shape = (8, 16)
        # not linspace(90, -90): a grid the fallback could not have guessed
        self.latitude = np.linspace(-89.0, 89.0, self.shape[0])
        self.longitude = np.linspace(0.0, 350.0, self.shape[1])

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, name="2017.h5", with_coordinates=True):
        base = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc)
        timestamps = [(base + dt.timedelta(days=idx)).timestamp() for idx in range(3)]
        data = np.zeros((3, NUM_CHANNELS, *self.shape), dtype=np.float32)
        return _write_h5(
            os.path.join(self.tmpdir.name, name),
            data,
            timestamps,
            latitude=self.latitude if with_coordinates else None,
            longitude=self.longitude if with_coordinates else None,
        )

    def test_coordinates_come_from_the_file(self):
        self._write()
        metadata = get_backend(self.tmpdir.name).discover()

        np.testing.assert_allclose(metadata.grid.lat, self.latitude, rtol=1e-6)
        np.testing.assert_allclose(metadata.grid.lon, self.longitude, rtol=1e-6)

    def test_a_file_without_coordinates_falls_back(self):
        self._write(with_coordinates=False)
        metadata = get_backend(self.tmpdir.name).discover()

        np.testing.assert_allclose(metadata.grid.lat, np.linspace(90, -90, self.shape[0], endpoint=True))
        np.testing.assert_allclose(metadata.grid.lon, np.linspace(0, 360, self.shape[1], endpoint=False))

    def test_an_explicit_grid_wins_over_the_file(self):
        # the caller knows something the file does not, e.g. a dataset written
        # with the wrong coordinates
        self._write()
        override = (np.linspace(1, 8, self.shape[0]), np.linspace(1, 16, self.shape[1]))
        metadata = get_backend(self.tmpdir.name, lat_lon=override).discover()

        np.testing.assert_allclose(metadata.grid.lat, override[0])

    def test_the_chunk_carries_the_file_coordinates(self):
        # the chunk is what a consumer actually reads coordinates from, so the
        # decomposition has to slice the file's grid, not a fabricated one
        self._write()
        backend = get_backend(self.tmpdir.name, io_grid=[2, 1], io_rank=[1, 0])
        backend.discover()

        np.testing.assert_allclose(backend.chunk.lat, self.latitude[self.shape[0] // 2 :], rtol=1e-6)


class TestWb2Backend(unittest.TestCase):

    def test_channel_names_are_required(self):
        # the layout addresses variables by name, so there is nothing to read
        # without them; failing at construction beats failing mid epoch
        with self.assertRaises(ValueError) as ctx:
            ArcoWB2Backend("/nonexistent", channel_names=None)
        self.assertIn("channel_names", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
