# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import tempfile
import unittest

import numpy as np

from makani.utils.parse_dataset_metada import parse_dataset_metadata

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import get_default_parameters, init_dataset_metadata

_CH_NAMES = ["u10m", "t2m", "u500", "z500", "t500"]
_IMG_H    = 32
_IMG_W    = 64
_DHOURS   = 6
_H5_PATH  = "fields"


class TestParseDatasetMetadata(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir  = self._tmpdir.name

        self.params = get_default_parameters()
        # ParamsBase: __setitem__ updates both the internal dict AND the
        # attribute, so parse_dataset_metadata can reach them via params["key"].
        # Plain attribute assignment (params.x = v) only sets the attribute and
        # leaves params["x"] broken, so we always use dict-style here.
        self.params["img_shape_x"]   = _IMG_H
        self.params["img_shape_y"]   = _IMG_W
        self.params["channel_names"] = _CH_NAMES

        # Standard fixture used by most tests.
        self.metadata_path = init_dataset_metadata(
            path=os.path.join(self.tmpdir, "meta"),
            channel_names=_CH_NAMES,
            dhours=_DHOURS,
            h5_path=_H5_PATH,
            lat=np.linspace(90, -90, _IMG_H, endpoint=True).tolist(),
            lon=np.linspace(0, 360, _IMG_W, endpoint=False).tolist(),
        )
        self.params["metadata_json_path"] = self.metadata_path

    def tearDown(self):
        self._tmpdir.cleanup()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _write_json(self, subdir, metadata):
        """Persist a custom metadata dict and return its path."""
        d = os.path.join(self.tmpdir, subdir)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "data.json")
        with open(path, "w") as f:
            json.dump(metadata, f)
        return path

    def _base_metadata(self, **overrides):
        """Minimal valid metadata dict; keyword args override top-level keys."""
        m = dict(
            dataset_name="testing",
            h5_path=_H5_PATH,
            dhours=_DHOURS,
            attrs={"description": "test"},
            coords=dict(
                grid_type="equiangular",
                channel=_CH_NAMES,
                lat=np.linspace(90, -90, _IMG_H, endpoint=True).tolist(),
                lon=np.linspace(0, 360, _IMG_W, endpoint=False).tolist(),
            ),
        )
        m.update(overrides)
        return m

    # -----------------------------------------------------------------------
    # 1. Basic field wiring
    # -----------------------------------------------------------------------

    def test_h5_path_set(self):
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["h5_path"], _H5_PATH)

    def test_dhours_set(self):
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["dhours"], _DHOURS)

    # -----------------------------------------------------------------------
    # 2. lat / lon present in metadata
    # -----------------------------------------------------------------------

    def test_lat_read_from_metadata(self):
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["lat"], np.linspace(90, -90, _IMG_H, endpoint=True).tolist())

    def test_lon_read_from_metadata(self):
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["lon"], np.linspace(0, 360, _IMG_W, endpoint=False).tolist())

    def test_data_grid_type_from_metadata(self):
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["data_grid_type"], "equiangular")

    # -----------------------------------------------------------------------
    # 3. Dummy lat / lon generated when absent from metadata
    # -----------------------------------------------------------------------

    def _path_without_latlon(self, subdir):
        meta = self._base_metadata()
        del meta["coords"]["lat"]
        del meta["coords"]["lon"]
        path = self._write_json(subdir, meta)
        self.params["metadata_json_path"] = path
        return path

    def test_dummy_lat_length_matches_img_shape_x(self):
        path = self._path_without_latlon("no_latlon_a")
        params, _ = parse_dataset_metadata(path, self.params)
        self.assertEqual(len(params["lat"]), _IMG_H)

    def test_dummy_lat_starts_at_north_pole(self):
        path = self._path_without_latlon("no_latlon_b")
        params, _ = parse_dataset_metadata(path, self.params)
        self.assertAlmostEqual(params["lat"][0], 90.0)

    def test_dummy_lat_ends_at_south_pole(self):
        path = self._path_without_latlon("no_latlon_c")
        params, _ = parse_dataset_metadata(path, self.params)
        self.assertAlmostEqual(params["lat"][-1], -90.0)

    def test_dummy_lat_monotonically_decreasing(self):
        path = self._path_without_latlon("no_latlon_d")
        params, _ = parse_dataset_metadata(path, self.params)
        lat = params["lat"]
        self.assertTrue(all(lat[i] > lat[i + 1] for i in range(len(lat) - 1)))

    def test_dummy_lon_length_matches_img_shape_y(self):
        path = self._path_without_latlon("no_latlon_e")
        params, _ = parse_dataset_metadata(path, self.params)
        self.assertEqual(len(params["lon"]), _IMG_W)

    def test_dummy_lon_starts_at_zero(self):
        path = self._path_without_latlon("no_latlon_f")
        params, _ = parse_dataset_metadata(path, self.params)
        self.assertAlmostEqual(params["lon"][0], 0.0)

    def test_dummy_lon_does_not_reach_360(self):
        path = self._path_without_latlon("no_latlon_g")
        params, _ = parse_dataset_metadata(path, self.params)
        self.assertLess(params["lon"][-1], 360.0)

    def test_dummy_lon_monotonically_increasing(self):
        path = self._path_without_latlon("no_latlon_h")
        params, _ = parse_dataset_metadata(path, self.params)
        lon = params["lon"]
        self.assertTrue(all(lon[i] < lon[i + 1] for i in range(len(lon) - 1)))

    def test_dummy_grid_type_is_equiangular(self):
        path = self._path_without_latlon("no_latlon_i")
        params, _ = parse_dataset_metadata(path, self.params)
        self.assertEqual(params["data_grid_type"], "equiangular")

    # -----------------------------------------------------------------------
    # 4. Channel sanitization
    # -----------------------------------------------------------------------

    def test_channel_subset_produces_correct_indices(self):
        """["u10m", "z500"] → indices [0, 3] in the dataset."""
        self.params["channel_names"] = ["u10m", "z500"]
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["in_channels"],  [0, 3])
        self.assertEqual(params["out_channels"], [0, 3])

    def test_in_channels_always_equals_out_channels(self):
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["in_channels"], params["out_channels"])

    def test_data_channel_names_is_full_dataset_list(self):
        """data_channel_names must be the complete dataset list, not the requested subset."""
        self.params["channel_names"] = ["u10m"]
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["data_channel_names"], _CH_NAMES)

    def test_all_channels_used_when_channel_names_absent_from_params(self):
        """Without channel_names in params, all dataset channels are accepted."""
        del self.params.channel_names
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["channel_names"], _CH_NAMES)
        self.assertEqual(params["in_channels"], list(range(len(_CH_NAMES))))

    # -----------------------------------------------------------------------
    # 5. analysis_epoch_start_dates
    # -----------------------------------------------------------------------

    def test_analysis_epoch_start_dates_read_when_present(self):
        dates = ["2020-01-01T00:00:00", "2021-06-15T00:00:00"]
        path = self._write_json("with_dates", self._base_metadata(analysis_epoch_start_dates=dates))
        self.params["metadata_json_path"] = path
        params, _ = parse_dataset_metadata(path, self.params)
        self.assertEqual(params["analysis_epoch_start_dates"], dates)

    def test_analysis_epoch_start_dates_defaults_to_empty_list(self):
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["analysis_epoch_start_dates"], [])

    # -----------------------------------------------------------------------
    # 6. dataset dict
    # -----------------------------------------------------------------------

    def test_dataset_name(self):
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["dataset"]["name"], "testing")

    def test_dataset_description(self):
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["dataset"]["description"], "A synthetic test dataset.")

    def test_dataset_metadata_file_is_json_path(self):
        params, _ = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(params["dataset"]["metadata_file"], self.metadata_path)

    # -----------------------------------------------------------------------
    # 7. Return value
    # -----------------------------------------------------------------------

    def test_returns_params_and_raw_metadata(self):
        result = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertEqual(len(result), 2)

    def test_raw_metadata_is_dict_with_expected_keys(self):
        _, metadata = parse_dataset_metadata(self.metadata_path, self.params)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata["dataset_name"], "testing")
        self.assertEqual(metadata["h5_path"], _H5_PATH)
        self.assertEqual(metadata["dhours"], _DHOURS)

    # -----------------------------------------------------------------------
    # 8. Error paths
    # -----------------------------------------------------------------------

    def test_missing_file_raises(self):
        with self.assertRaises(Exception):
            parse_dataset_metadata("/nonexistent/data.json", self.params)

    def test_unknown_channel_raises_value_error(self):
        self.params["channel_names"] = ["u10m", "pressure_500"]   # pressure_500 not in dataset
        with self.assertRaises(ValueError):
            parse_dataset_metadata(self.metadata_path, self.params)


if __name__ == "__main__":
    unittest.main()
