# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

from makani.utils.dataloaders.wb2_helpers import (
    surface_variables,
    atmospheric_variables,
    surface_variables_inv,
    atmospheric_variables_inv,
    split_convert_channel_names,
    build_wb2_channel_map,
)


# ===========================================================================
# 1. Mapping table sanity checks
# ===========================================================================

class TestMappingTables(unittest.TestCase):

    def test_surface_variables_non_empty(self):
        self.assertGreater(len(surface_variables), 0)

    def test_atmospheric_variables_non_empty(self):
        self.assertGreater(len(atmospheric_variables), 0)

    def test_inverse_surface_round_trips(self):
        for era5, wb2 in surface_variables.items():
            self.assertEqual(surface_variables_inv[wb2], era5)

    def test_inverse_atmospheric_round_trips(self):
        for era5, wb2 in atmospheric_variables.items():
            self.assertEqual(atmospheric_variables_inv[wb2], era5)

    def test_known_surface_mappings(self):
        self.assertEqual(surface_variables["u10m"], "10m_u_component_of_wind")
        self.assertEqual(surface_variables["t2m"],  "2m_temperature")
        self.assertEqual(surface_variables["msl"],  "mean_sea_level_pressure")

    def test_known_atmospheric_mappings(self):
        self.assertEqual(atmospheric_variables["z"], "geopotential")
        self.assertEqual(atmospheric_variables["u"], "u_component_of_wind")
        self.assertEqual(atmospheric_variables["t"], "temperature")


# ===========================================================================
# 2. build_wb2_channel_map
# ===========================================================================

class TestBuildWb2ChannelMap(unittest.TestCase):

    # ---- correct mappings --------------------------------------------------

    def test_single_surface_channel(self):
        result = build_wb2_channel_map(["u10m"])
        self.assertEqual(result, [("10m_u_component_of_wind", None)])

    def test_multiple_surface_channels(self):
        result = build_wb2_channel_map(["u10m", "t2m", "msl"])
        self.assertEqual(result, [
            ("10m_u_component_of_wind", None),
            ("2m_temperature", None),
            ("mean_sea_level_pressure", None),
        ])

    def test_single_atmospheric_channel(self):
        result = build_wb2_channel_map(["z500"], level_values=[100, 500, 850])
        self.assertEqual(result, [("geopotential", 1)])   # 500 is at index 1

    def test_atmospheric_level_index_matches_position(self):
        levels = [50, 100, 200, 500, 850, 1000]
        result = build_wb2_channel_map(["u500"], level_values=levels)
        self.assertEqual(result[0], ("u_component_of_wind", 3))  # 500 at index 3

    def test_mixed_surface_and_atmospheric(self):
        result = build_wb2_channel_map(
            ["u10m", "t2m", "z500", "u850"],
            level_values=[500, 850],
        )
        self.assertEqual(result, [
            ("10m_u_component_of_wind", None),
            ("2m_temperature", None),
            ("geopotential", 0),          # 500 at index 0
            ("u_component_of_wind", 1),   # 850 at index 1
        ])

    def test_d2_treated_as_surface_not_atmospheric(self):
        # "d2" ends with a digit but must be treated as a surface variable
        result = build_wb2_channel_map(["d2"])
        self.assertEqual(result, [("2m_dewpoint_temperature", None)])

    def test_multiple_levels_same_variable(self):
        result = build_wb2_channel_map(["z500", "z850"], level_values=[500, 850])
        self.assertEqual(result[0], ("geopotential", 0))
        self.assertEqual(result[1], ("geopotential", 1))

    def test_length_matches_input(self):
        channels = ["u10m", "t2m", "z500", "u500", "t500"]
        result = build_wb2_channel_map(channels, level_values=[500])
        self.assertEqual(len(result), len(channels))

    # ---- graceful error handling -------------------------------------------

    def test_unknown_surface_variable_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_wb2_channel_map(["totally_unknown"])
        self.assertIn("totally_unknown", str(ctx.exception))

    def test_unknown_atmospheric_prefix_raises(self):
        # "xyz500": ends in digits, not "d2", but "xyz" is not a known prefix
        with self.assertRaises(ValueError) as ctx:
            build_wb2_channel_map(["xyz500"], level_values=[500])
        self.assertIn("xyz", str(ctx.exception))

    def test_atmospheric_level_absent_from_store_raises(self):
        # z500 requested but store only has levels [100, 850]
        with self.assertRaises(ValueError) as ctx:
            build_wb2_channel_map(["z500"], level_values=[100, 850])
        self.assertIn("500", str(ctx.exception))

    def test_atmospheric_channel_with_no_level_values_raises(self):
        # no level_values provided at all — level_to_idx is empty
        with self.assertRaises(ValueError) as ctx:
            build_wb2_channel_map(["z500"])
        self.assertIn("500", str(ctx.exception))

    def test_error_message_lists_available_levels(self):
        with self.assertRaises(ValueError) as ctx:
            build_wb2_channel_map(["z500"], level_values=[100, 850])
        msg = str(ctx.exception)
        self.assertIn("100", msg)
        self.assertIn("850", msg)

    def test_error_message_lists_known_surface_names(self):
        with self.assertRaises(ValueError) as ctx:
            build_wb2_channel_map(["bad_surf"])
        self.assertIn("Known names", str(ctx.exception))

    def test_error_message_lists_known_atmospheric_prefixes(self):
        with self.assertRaises(ValueError) as ctx:
            build_wb2_channel_map(["bad500"], level_values=[500])
        self.assertIn("Known prefixes", str(ctx.exception))


# ===========================================================================
# 3. split_convert_channel_names
# ===========================================================================

class TestSplitConvertChannelNames(unittest.TestCase):

    def test_mixed_channels_split_correctly(self):
        channels = ["u10m", "t2m", "z500", "u500", "t850"]
        atm_names, atm_wb2, surf_names, surf_wb2, levels = split_convert_channel_names(channels)
        self.assertIn("z",   atm_names)
        self.assertIn("u",   atm_names)
        self.assertNotIn("t2m", atm_names)
        self.assertIn("geopotential",       atm_wb2)
        self.assertIn("u_component_of_wind", atm_wb2)
        self.assertIn("u10m", surf_names)
        self.assertIn("t2m",  surf_names)
        self.assertIn(500,  levels)
        self.assertIn(850,  levels)

    def test_surface_only(self):
        atm_names, atm_wb2, surf_names, surf_wb2, levels = split_convert_channel_names(["u10m", "t2m"])
        self.assertEqual(atm_names, [])
        self.assertEqual(atm_wb2,   [])
        self.assertEqual(levels,    [])
        self.assertIn("u10m", surf_names)
        self.assertIn("t2m",  surf_names)
        self.assertIn("10m_u_component_of_wind", surf_wb2)

    def test_atmospheric_only(self):
        atm_names, atm_wb2, surf_names, surf_wb2, levels = split_convert_channel_names(["z500", "t500", "u500"])
        self.assertEqual(surf_names, [])
        self.assertEqual(surf_wb2,   [])
        self.assertIn("z", atm_names)
        self.assertEqual(levels, [500])

    def test_levels_are_sorted(self):
        _, _, _, _, levels = split_convert_channel_names(["z850", "z500", "z200"])
        self.assertEqual(levels, sorted(levels))

    def test_atmospheric_prefixes_are_deduplicated(self):
        # z500 and z850 both use prefix "z" — should appear once
        atm_names, _, _, _, _ = split_convert_channel_names(["z500", "z850"])
        self.assertEqual(atm_names.count("z"), 1)

    def test_output_lengths_match(self):
        atm_names, atm_wb2, surf_names, surf_wb2, _ = split_convert_channel_names(
            ["u10m", "t2m", "z500", "u500"]
        )
        self.assertEqual(len(atm_names), len(atm_wb2))
        self.assertEqual(len(surf_names), len(surf_wb2))


if __name__ == "__main__":
    unittest.main()
