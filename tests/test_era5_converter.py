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
Tests for ``data_process/convert_era5_to_makani_input.py`` and its sources.

Everything but the last suite runs without MPI and without network access: the
output file helpers write serial files, the NCAR source reads from a stand-in
for its S3 store and the WB2 source from a small local zarr store. The full
conversion needs mpi4py and parallel HDF5 and is skipped where they are missing.
"""

import os
import sys
import tempfile
import unittest
import importlib.util
import datetime as dt
from unittest import mock

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_process.convert_era5_to_makani_input import _batched, _create_output_file, _write_missing
from data_process.sources import Source
from .testutils import compare_arrays

_LAT = [90.0, 0.0, -90.0]
_LON = [0.0, 120.0, 240.0]
_DHOURS = 6


def _utc(year, month, day, hour=0):
    return dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc)


def _metadata(channels):
    return {"dhours": _DHOURS, "coords": {"channel": channels, "lat": _LAT, "lon": _LON}}


def _samples(start, num):
    return [(idx, start + dt.timedelta(hours=_DHOURS * idx)) for idx in range(num)]


def _have_mpi_hdf5():
    return importlib.util.find_spec("mpi4py") is not None and h5py.get_config().mpi


class _OutputFileCase(unittest.TestCase):
    """Provides a serial output file laid out like the converter's."""

    channels = ["a", "b", "c"]

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.times = [t for _, t in _samples(_utc(2023, 6, 15), 8)]
        timestamps = np.array([t.timestamp() for t in self.times], dtype=np.float64)
        self.out = _create_output_file(
            os.path.join(self._tmpdir.name, "2023.h5"), None, "fields", timestamps, self.channels, _LAT, _LON
        )

    def tearDown(self):
        self.out.close()
        self._tmpdir.cleanup()


class TestBatched(unittest.TestCase):
    def test_consecutive_batches_with_remainder(self):
        self.assertEqual(list(_batched(range(7), 3)), [(0, 1, 2), (3, 4, 5), (6,)])

    def test_empty_input_gives_no_batches(self):
        self.assertEqual(list(_batched([], 3)), [])


class TestOutputFile(_OutputFileCase):
    def test_fill_value_is_declared_but_never_written(self):
        dset = self.out["fields"]
        self.assertTrue(np.isnan(dset.fillvalue))
        self.assertEqual(dset.id.get_create_plist().get_fill_time(), h5py.h5d.FILL_TIME_NEVER)

    def test_layout(self):
        self.assertEqual(self.out["fields"].shape, (8, 3, len(_LAT), len(_LON)))
        self.assertTrue(np.all(self.out["valid_data"][...] == 1))
        self.assertEqual([c.decode() for c in self.out["channel"][...]], self.channels)
        expected = np.array([t.timestamp() for t in self.times])
        self.assertTrue(
            compare_arrays("timestamps", self.out["timestamp"][...], expected, atol=0.0, rtol=0.0, shape_check=True)
        )

    def test_write_missing_touches_only_the_given_channels_and_samples(self):
        self.out["fields"][...] = 1.0
        _write_missing(self.out, "fields", _samples(_utc(2023, 6, 15), 8)[2:5], [0, 2])

        fields = self.out["fields"][...]
        valid = self.out["valid_data"][...]
        for cidx in [0, 2]:
            self.assertTrue(np.all(np.isnan(fields[2:5, cidx])))
            self.assertTrue(np.all(valid[2:5, cidx] == 0))
            self.assertTrue(np.all(fields[[0, 1, 5, 6, 7], cidx] == 1.0))
            self.assertTrue(np.all(valid[[0, 1, 5, 6, 7], cidx] == 1))
        self.assertTrue(np.all(fields[:, 1] == 1.0))
        self.assertTrue(np.all(valid[:, 1] == 1))


class _FakeNcarStore(object):
    """Stand-in for NcarStore, serving local files for the keys it knows and raising for all others."""

    def __init__(self, files):
        self.files = files
        self.handles = {}
        self.released = []

    def _path(self, key):
        for marker, path in self.files.items():
            if marker in key:
                return path
        raise FileNotFoundError(key)

    def set_read_plan(self, keys):
        pass

    def open(self, key):
        if key not in self.handles:
            self.handles[key] = h5py.File(self._path(key), "r")
        return self.handles[key]

    def coord(self, key, name):
        return self.open(key)[name][:]

    def release(self, key):
        self.released.append(key)
        handle = self.handles.pop(key, None)
        if handle is not None:
            handle.close()

    def close(self):
        for handle in self.handles.values():
            handle.close()


class TestNcarSource(_OutputFileCase):
    """Filling and imputation of the NCAR source against local stand-ins for the bucket objects."""

    channels = ["z500", "t2m", "tp", "not_a_variable"]

    def setUp(self):
        super().setUp()
        from makani.utils.dataloaders.ncar_helpers import to_ncar_hours

        # a surface file for t2m that only holds the first of the two days, so
        # that the second day exercises the missing timestep path
        self.t2m = np.arange(4 * len(_LAT) * len(_LON), dtype=np.float32).reshape(4, len(_LAT), len(_LON))
        path = os.path.join(self._tmpdir.name, "t2m.nc")
        with h5py.File(path, "w") as f:
            f["latitude"] = np.array(_LAT)
            f["longitude"] = np.array(_LON)
            f["time"] = np.array([to_ncar_hours(t) for t in self.times[:4]])
            f["VAR_2T"] = self.t2m
        self.store = _FakeNcarStore({"128_167_2t": path})

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def _make(self, **options):
        from data_process.sources.ncar import NcarSource

        with mock.patch("data_process.sources.ncar.NcarStore", lambda *args, **kwargs: self.store):
            return NcarSource(_metadata(self.channels), 0, skip_missing_channels=True, **options)

    def _fill(self, source):
        units = source.split_units(_samples(_utc(2023, 6, 15), 8))
        self.assertEqual([len(unit) for unit in units], [4, 4])
        source.begin_year(units)
        source.fill(self.out, "fields", units)

    def test_unsupported_channels_are_reported_as_skipped(self):
        self.assertEqual(self._make().skipped_channel_indices(), [3])

    def test_missing_data_fails_without_imputation(self):
        with self.assertRaises(FileNotFoundError):
            self._fill(self._make())

    def test_missing_data_is_imputed(self):
        self._fill(self._make(impute_missing_timestamps=True))
        fields = self.out["fields"][...]
        valid = self.out["valid_data"][...]

        # t2m is read for the first day, and imputed for the second, missing from its file
        self.assertTrue(compare_arrays("t2m", fields[:4, 1], self.t2m, atol=0.0, rtol=0.0, shape_check=True))
        self.assertTrue(np.all(valid[:4, 1] == 1))
        self.assertTrue(np.all(np.isnan(fields[4:, 1])))
        self.assertTrue(np.all(valid[4:, 1] == 0))

        # z500 and tp have no objects at all
        for cidx in [0, 2]:
            self.assertTrue(np.all(np.isnan(fields[:, cidx])))
            self.assertTrue(np.all(valid[:, cidx] == 0))

        # the skipped channel is the converter's business, not the source's
        self.assertTrue(np.all(valid[:, 3] == 1))

    def test_pressure_level_file_is_released_after_imputation(self):
        self._fill(self._make(impute_missing_timestamps=True))
        self.assertEqual(sum("128_129_z" in key for key in self.store.released), 2)


@unittest.skipUnless(
    importlib.util.find_spec("xarray") is not None and importlib.util.find_spec("zarr") is not None,
    "needs xarray and zarr",
)
class TestWb2Source(_OutputFileCase):
    """Filling, skipping and imputation of the WB2 source against a small local zarr store."""

    channels = ["z500", "t2m", "u10m"]

    def setUp(self):
        super().setUp()
        import xarray as xr

        # u10m is absent from the store, and the last sample is missing from its time axis
        times = np.array([np.datetime64(t.replace(tzinfo=None), "ns") for t in self.times[:-1]])
        shape = (len(times), len(_LAT), len(_LON))
        self.t2m = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        self.z = -np.arange(2 * np.prod(shape), dtype=np.float32).reshape(len(times), 2, len(_LAT), len(_LON))
        data = xr.Dataset(
            {
                "2m_temperature": (("time", "latitude", "longitude"), self.t2m),
                "geopotential": (("time", "level", "latitude", "longitude"), self.z),
            },
            coords={"time": times, "level": [500, 850], "latitude": _LAT, "longitude": _LON},
        )
        self.store = os.path.join(self._tmpdir.name, "wb2.zarr")
        data.to_zarr(self.store)

    def _make(self, **options):
        from data_process.sources.wb2 import Wb2Source

        return Wb2Source(_metadata(self.channels), 0, input_file=self.store, **options)

    def test_missing_variable_fails_without_skipping(self):
        with self.assertRaises(IndexError):
            self._make()

    def test_missing_variables_are_reported_as_skipped(self):
        self.assertEqual(self._make(skip_missing_channels=True).skipped_channel_indices(), [2])

    def test_missing_timestamp_fails_without_imputation(self):
        source = self._make(skip_missing_channels=True, batch_size=8)
        with self.assertRaises(IndexError):
            source.fill(self.out, "fields", source.split_units(_samples(_utc(2023, 6, 15), 8)))

    def test_fill_and_impute(self):
        source = self._make(skip_missing_channels=True, impute_missing_timestamps=True, batch_size=3)
        units = source.split_units(_samples(_utc(2023, 6, 15), 8))
        for batch in _batched(units, source.units_per_fill):
            source.fill(self.out, "fields", list(batch))
        fields = self.out["fields"][...]
        valid = self.out["valid_data"][...]

        self.assertTrue(compare_arrays("t2m", fields[:7, 1], self.t2m, atol=0.0, rtol=0.0, shape_check=True))
        self.assertTrue(compare_arrays("z500", fields[:7, 0], self.z[:, 0], atol=0.0, rtol=0.0, shape_check=True))
        self.assertTrue(np.all(valid[:7, :2] == 1))
        # the last sample is not in the store
        self.assertTrue(np.all(np.isnan(fields[7, :2])))
        self.assertTrue(np.all(valid[7, :2] == 0))


class _FakeSource(Source):
    """Writes the sample index into every channel it provides and skips the last one."""

    units_per_fill = 2

    def fill(self, out, entry_key, units):
        for unit in units:
            for sample_index, _ in unit:
                for cidx in range(len(self.channel_names) - 1):
                    out[entry_key][sample_index, cidx, ...] = np.full(
                        (len(self.lat), len(self.lon)), sample_index, dtype=np.float32
                    )

    def skipped_channel_indices(self):
        return [len(self.channel_names) - 1]


@unittest.skipUnless(_have_mpi_hdf5(), "needs mpi4py and h5py built with MPI")
class TestConvert(unittest.TestCase):
    """A full conversion on a single rank across a year boundary, with a source that skips a channel."""

    def setUp(self):
        import json

        self._tmpdir = tempfile.TemporaryDirectory()
        self.metadata_file = os.path.join(self._tmpdir.name, "metadata.json")
        with open(self.metadata_file, "w") as f:
            json.dump(_metadata(["a", "b", "c"]), f)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_yearly_files(self):
        from data_process.convert_era5_to_makani_input import convert

        convert(_FakeSource, self._tmpdir.name, self.metadata_file, _utc(2022, 12, 31, 6), _utc(2023, 1, 1, 12))

        expected = {
            "2022.h5": [_utc(2022, 12, 31, 6), _utc(2022, 12, 31, 12), _utc(2022, 12, 31, 18)],
            "2023.h5": [_utc(2023, 1, 1, 0), _utc(2023, 1, 1, 6), _utc(2023, 1, 1, 12)],
        }
        for name, times in expected.items():
            with self.subTest(file=name), h5py.File(os.path.join(self._tmpdir.name, name), "r") as f:
                stamps = np.array([t.timestamp() for t in times])
                self.assertTrue(
                    compare_arrays("timestamps", f["timestamp"][...], stamps, atol=0.0, rtol=0.0, shape_check=True)
                )
                fields = f["fields"][...]
                valid = f["valid_data"][...]
                for idx in range(len(times)):
                    self.assertTrue(np.all(fields[idx, :2] == idx))
                self.assertTrue(np.all(valid[:, :2] == 1))
                # the skipped channel is written as missing rather than left unwritten
                self.assertTrue(np.all(np.isnan(fields[:, 2])))
                self.assertTrue(np.all(valid[:, 2] == 0))


if __name__ == "__main__":
    unittest.main()
