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
Unit tests for ``makani.utils.dataloaders.icon_helpers``: the decoding of ICON
netCDF conventions and the mapping from ICON variable names onto makani
channels.

The candidate resolution and channel grouping themselves are shared with the
other readers and are covered in ``test_channel_helpers.py``; what is pinned
here is which ICON names each makani channel maps to.

Everything here works on plain arrays and attribute values, so no ICON file, no
h5py and no grid is needed. The reads and the regridding live in the converter
and are not covered.
"""

import os
import sys
import unittest
import datetime as dt

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.dataloaders.icon_helpers import (
    GRAVITY,
    IconVariable,
    accumulated_variables,
    ICON_TIME_UNITS,
    build_icon_channel_groups,
    check_grid_uuid,
    decode_time,
    decode_values,
    grid_coordinates_in_degrees,
    pressure_level_index,
    pressure_levels_in_hpa,
)


def _utc(year, month, day, hour=0, minute=0, second=0):
    return dt.datetime(year, month, day, hour, minute, second, tzinfo=dt.timezone.utc)


class TestBuildIconChannelGroups(unittest.TestCase):
    """
    Grouping mirrors the NCAR reader: all levels of a variable are read from one
    ICON variable, and every group remembers which makani channel indices it
    fills.
    """

    def test_levels_of_one_variable_share_a_group(self):
        groups = build_icon_channel_groups(["z500", "z850", "z1000"])

        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group.kind, "pl")
        self.assertEqual(group.name, "z")
        self.assertEqual(group.levels, [500, 850, 1000])
        self.assertEqual(group.channel_indices, [0, 1, 2])

    def test_channel_indices_track_positions_in_the_original_list(self):
        groups = build_icon_channel_groups(["z500", "t850", "z1000", "t2m"])
        by_name = {group.name: group for group in groups}

        self.assertEqual(by_name["z"].channel_indices, [0, 2])
        self.assertEqual(by_name["z"].levels, [500, 1000])
        self.assertEqual(by_name["t"].channel_indices, [1])
        self.assertEqual(by_name["t2m"].channel_indices, [3])

    def test_pressure_level_groups_come_first(self):
        groups = build_icon_channel_groups(["t2m", "z500", "tp", "u850"])
        kinds = [group.kind for group in groups]

        self.assertEqual(kinds[:2], ["pl", "pl"])
        self.assertEqual(sorted(kinds[2:]), ["accum", "sfc"])

    def test_resolution_follows_the_naming_of_the_file(self):
        nwp = build_icon_channel_groups(["t850", "t2m", "msl"], available=["temp", "t_2m", "pres_msl"])
        aes = build_icon_channel_groups(["t850", "t2m", "msl"], available=["ta", "tas", "psl"])

        self.assertEqual([group.variables[0].name for group in nwp], ["temp", "t_2m", "pres_msl"])
        self.assertEqual([group.variables[0].name for group in aes], ["ta", "tas", "psl"])

    def test_accumulation_semantics_come_from_the_resolved_variable(self):
        # a running total has to be differenced, a flux has to be integrated;
        # which one applies is a property of the file, not of the channel
        total = build_icon_channel_groups(["tp"], available=["tot_prec"])[0]
        flux = build_icon_channel_groups(["tp"], available=["pr"])[0]

        self.assertEqual(total.variables[0].accumulation, "since_start")
        self.assertEqual(flux.variables[0].accumulation, "rate")

    def test_geopotential_and_geopotential_height_are_distinguishable(self):
        # z is geopotential; a file offering only zg gives metres and needs a
        # factor of GRAVITY, so the units have to survive resolution
        geopotential = build_icon_channel_groups(["z500"], available=["geopot"])[0]
        height = build_icon_channel_groups(["z500"], available=["zg"])[0]

        self.assertEqual(geopotential.variables[0].units, "m2 s-2")
        self.assertEqual(height.variables[0].units, "m")
        self.assertAlmostEqual(GRAVITY, 9.80665)

    def test_hydrometeors_map_onto_era5_channel_names(self):
        # ERA5 carries these as specific contents in kg kg-1, the same quantity
        # and unit ICON writes, so the channels keep the ERA5 vocabulary
        # no qg in this file, so cswc falls back to qs alone; the sum is covered
        # in test_snow_is_summed_with_graupel_to_match_era5
        channels = ["clwc500", "ciwc500", "crwc500", "cswc500"]
        groups = build_icon_channel_groups(channels, available=["qc", "qi", "qr", "qs"])

        self.assertEqual([group.name for group in groups], ["clwc", "ciwc", "crwc", "cswc"])
        self.assertEqual([group.variables[0].name for group in groups], ["qc", "qi", "qr", "qs"])
        for group in groups:
            with self.subTest(channel=group.name):
                self.assertEqual(group.variables[0].units, "kg kg-1")
                self.assertEqual(group.levels, [500])

    def test_snow_is_summed_with_graupel_to_match_era5(self):
        """
        ERA5's snow content includes graupel, ICON's qs does not, so the
        ERA5-named channel is the sum. At convection-resolving resolution
        graupel dominates in deep convective cores, so reading qs alone would
        bias the field exactly where the run is most informative.
        """
        group = build_icon_channel_groups(["cswc500"], available=["qs", "qg"])[0]

        self.assertEqual(group.name, "cswc")
        self.assertEqual([variable.name for variable in group.variables], ["qs", "qg"])

    def test_snow_falls_back_to_qs_without_graupel(self):
        # a microphysics scheme with no graupel category still provides cswc
        group = build_icon_channel_groups(["cswc500"], available=["qs"])[0]

        self.assertEqual([variable.name for variable in group.variables], ["qs"])

    def test_graupel_is_still_available_on_its_own(self):
        # for runs that prefer ICON's species split over the ERA5 vocabulary
        group = build_icon_channel_groups(["qg500"], available=["qs", "qg"])[0]

        self.assertEqual(group.name, "qg")
        self.assertEqual([variable.name for variable in group.variables], ["qg"])

    def test_hydrometeor_channel_names_parse_despite_four_letters(self):
        # the channel name splitter is built around 1-3 letter prefixes; these
        # names are longer, so pin down that the level still separates correctly
        group = build_icon_channel_groups(["clwc850", "clwc1000"], available=["qc"])[0]

        self.assertEqual(group.name, "clwc")
        self.assertEqual(group.levels, [850, 1000])

    def test_graupel_keeps_the_icon_name(self):
        # ERA5 has no graupel parameter, the IFS folds it into snow, so there is
        # no ERA5 channel to map onto
        group = build_icon_channel_groups(["qg500"], available=["qg"])[0]

        self.assertEqual(group.name, "qg")
        self.assertEqual(group.variables[0].name, "qg")

    def test_cmip_style_cloud_names_resolve(self):
        groups = build_icon_channel_groups(["clwc500", "ciwc500"], available=["clw", "cli"])
        self.assertEqual([group.variables[0].name for group in groups], ["clw", "cli"])

    def test_precipitation_offers_alternatives_not_components(self):
        """
        The mirror image of the NCAR tp entry, and the reason the nesting is
        explicit.

        tot_prec and pr are two ways a run may report precipitation, not two
        quantities to add up: a file carries one or the other. Nested one level
        deeper they would read as components, resolution would demand both, and
        every file would fail to provide tp at all.
        """
        candidates = accumulated_variables["tp"]

        self.assertGreater(len(candidates), 1, "tp must offer more than one possible source")
        for candidate in candidates:
            with self.subTest(variable=candidate.name):
                # a bare descriptor is a candidate of a single component
                self.assertIsInstance(candidate, IconVariable)

    def test_unresolvable_channels_raise(self):
        with self.subTest(desc="unknown atmospheric prefix"):
            with self.assertRaises(ValueError):
                build_icon_channel_groups(["xyz500"])

        with self.subTest(desc="unknown surface name"):
            with self.assertRaises(ValueError):
                build_icon_channel_groups(["not_a_variable"])

        with self.subTest(desc="known channel the file does not provide"):
            with self.assertRaises(ValueError):
                build_icon_channel_groups(["t850"], available=["qv"])

    def test_unresolvable_channels_can_be_skipped(self):
        groups = build_icon_channel_groups(["z500", "xyz500", "not_a_variable", "t2m"], skip_missing_channels=True)

        self.assertEqual(sorted(group.name for group in groups), ["t2m", "z"])


class TestDecodeTime(unittest.TestCase):
    """
    ICON's native encoding packs the date into the integer part and the time of
    day into the fraction. Reading it as a CF offset would produce wrong but
    plausible dates, which is the whole reason this function exists.
    """

    def test_icon_float_encoding(self):
        # 0.333333 of a day is 08:00
        times = decode_time([20170821.333333], ICON_TIME_UNITS)
        self.assertEqual(times, [_utc(2017, 8, 21, 8)])

    def test_icon_float_encoding_midnight_and_noon(self):
        times = decode_time([20170821.0, 20170821.5, 20171231.75], ICON_TIME_UNITS)
        self.assertEqual(
            times,
            [_utc(2017, 8, 21, 0), _utc(2017, 8, 21, 12), _utc(2017, 12, 31, 18)],
        )

    def test_icon_float_encoding_rounds_to_the_nearest_second(self):
        # the fraction is a float, so an exact hour is not exactly representable
        times = decode_time([20170821.25], ICON_TIME_UNITS)
        self.assertEqual(times[0].second, 0)
        self.assertEqual(times[0].microsecond, 0)
        self.assertEqual(times[0], _utc(2017, 8, 21, 6))

    def test_cf_encoding_is_also_accepted(self):
        times = decode_time([0, 6, 24], b"hours since 2017-01-01 00:00:00")
        self.assertEqual(times, [_utc(2017, 1, 1), _utc(2017, 1, 1, 6), _utc(2017, 1, 2)])

    def test_cf_encoding_variants(self):
        for units in [
            "days since 2017-01-01",
            "days since 2017-01-01 00:00:00",
            "days since 2017-01-01T00:00:00Z",
        ]:
            with self.subTest(units=units):
                self.assertEqual(decode_time([1.5], units), [_utc(2017, 1, 2, 12)])

    def test_cf_reference_without_padding(self):
        # udunits does not require zero padding and ICON writes exactly this
        times = decode_time([0, 180], "minutes since 2020-1-1 00:00:00")
        self.assertEqual(times, [_utc(2020, 1, 1, 0), _utc(2020, 1, 1, 3)])

    def test_cf_reference_with_offset_is_converted(self):
        # a reference naming a local instant is moved to UTC, matching
        # makani.utils.dataloaders.data_helpers.get_date_from_string
        self.assertEqual(decode_time([0], "hours since 2020-01-01 00:00:00 +2:00"), [_utc(2019, 12, 31, 22)])
        self.assertEqual(decode_time([0], "hours since 2020-01-01 00:00:00 -05:00"), [_utc(2020, 1, 1, 5)])
        self.assertEqual(decode_time([0], "hours since 2020-01-01 00:00:00 +0200"), [_utc(2019, 12, 31, 22)])
        self.assertEqual(decode_time([0], "hours since 2020-01-01T00:00:00+02:00"), [_utc(2019, 12, 31, 22)])

    def test_cf_reference_designators_mean_utc(self):
        for reference in ("2020-01-01 00:00:00 UTC", "2020-01-01 00:00:00 GMT", "2020-01-01T00:00:00Z"):
            with self.subTest(reference=reference):
                self.assertEqual(decode_time([0], f"hours since {reference}"), [_utc(2020, 1, 1)])

    def test_bare_date_is_not_read_as_an_offset(self):
        # "2020-01-01" ends in something shaped like "-01"; taking it for an
        # offset would silently move the epoch by an hour
        self.assertEqual(decode_time([0], "days since 2020-01-01"), [_utc(2020, 1, 1)])

    def test_cf_seconds_and_minutes(self):
        self.assertEqual(decode_time([90], "seconds since 2017-01-01"), [_utc(2017, 1, 1, 0, 1, 30)])
        self.assertEqual(decode_time([90], "minutes since 2017-01-01"), [_utc(2017, 1, 1, 1, 30)])

    def test_scalar_input_is_accepted(self):
        self.assertEqual(decode_time(20170821.5, ICON_TIME_UNITS), [_utc(2017, 8, 21, 12)])

    def test_unknown_encoding_raises(self):
        with self.assertRaises(ValueError):
            decode_time([0.0], "elapsed model seconds")

    def test_invalid_date_raises(self):
        # month 13 is not a date, and must not be silently accepted
        with self.assertRaises(ValueError):
            decode_time([20171321.0], ICON_TIME_UNITS)


class TestDecodeValues(unittest.TestCase):
    """
    Reading netCDF through h5py bypasses the library's unpacking, so packing and
    fill values have to be applied here or the fields come out as raw integers.
    """

    def test_unpacks_scale_and_offset(self):
        raw = np.array([0, 100, 200], dtype=np.int16)
        values = decode_values(raw, scale_factor=0.5, add_offset=250.0)

        np.testing.assert_allclose(values, [250.0, 300.0, 350.0])
        self.assertEqual(values.dtype, np.float32)

    def test_passes_unpacked_data_through(self):
        raw = np.array([1.5, 2.5], dtype=np.float64)
        np.testing.assert_allclose(decode_values(raw), [1.5, 2.5])

    def test_fill_values_become_nan(self):
        raw = np.array([1, -9999, 3], dtype=np.int32)
        values = decode_values(raw, fill_value=-9999)

        self.assertTrue(np.isnan(values[1]))
        np.testing.assert_allclose(values[[0, 2]], [1.0, 3.0])

    def test_fill_value_is_matched_before_unpacking(self):
        # CF matches the sentinel against the stored value; matching after
        # scaling would either miss it or catch valid data instead
        raw = np.array([0, -32767, 100], dtype=np.int16)
        values = decode_values(raw, fill_value=-32767, scale_factor=0.1, add_offset=273.15)

        self.assertTrue(np.isnan(values[1]))
        np.testing.assert_allclose(values[[0, 2]], [273.15, 283.15], rtol=1e-6)

    def test_integer_output_dtype_is_rejected(self):
        # NaN cannot be represented, so missing data would turn into a number
        with self.assertRaises(ValueError):
            decode_values(np.array([1, 2]), fill_value=1, dtype=np.int32)


class TestGridCoordinates(unittest.TestCase):
    """ICON stores cell centers in radians with longitude in [-pi, pi]; makani
    works in degrees with longitude in [0, 360)."""

    def test_converts_radians_to_degrees(self):
        lon, lat = grid_coordinates_in_degrees([0.0, np.pi / 2], [0.0, np.pi / 4])

        np.testing.assert_allclose(lon, [0.0, 90.0])
        np.testing.assert_allclose(lat, [0.0, 45.0])

    def test_wraps_negative_longitudes(self):
        lon, _ = grid_coordinates_in_degrees([-np.pi / 2, -np.pi], [0.0, 0.0])
        np.testing.assert_allclose(lon, [270.0, 180.0])

    def test_poles_are_accepted(self):
        _, lat = grid_coordinates_in_degrees([0.0, 0.0], [np.pi / 2, -np.pi / 2])
        np.testing.assert_allclose(lat, [90.0, -90.0])

    def test_degrees_input_is_rejected(self):
        # a file already in degrees would otherwise collapse into a few degrees
        # around the prime meridian without any error
        with self.assertRaises(ValueError):
            grid_coordinates_in_degrees([0.0, 90.0], [0.0, 45.0])

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            grid_coordinates_in_degrees([0.0, 1.0], [0.0])


class TestPressureLevels(unittest.TestCase):
    """ICON writes levels in Pa, makani channel names carry hPa; selecting the
    wrong one picks a completely different level without failing."""

    def test_pascals_are_converted(self):
        np.testing.assert_allclose(pressure_levels_in_hpa([100000.0, 50000.0, 5000.0]), [1000.0, 500.0, 50.0])

    def test_hectopascals_are_left_alone(self):
        np.testing.assert_allclose(pressure_levels_in_hpa([1000.0, 500.0, 50.0]), [1000.0, 500.0, 50.0])

    def test_index_lookup_in_pascals(self):
        levels = [100000.0, 85000.0, 50000.0, 25000.0]
        self.assertEqual(pressure_level_index(levels, 500), 2)
        self.assertEqual(pressure_level_index(levels, 1000), 0)

    def test_index_lookup_in_hectopascals(self):
        levels = [1000.0, 850.0, 500.0, 250.0]
        self.assertEqual(pressure_level_index(levels, 850), 1)

    def test_missing_level_raises(self):
        with self.assertRaises(ValueError):
            pressure_level_index([100000.0, 85000.0], 500)

    def test_empty_level_coordinate_raises(self):
        with self.assertRaises(ValueError):
            pressure_levels_in_hpa([])


class TestCheckGridUuid(unittest.TestCase):
    """The data file names the grid it was run on; regridding against a
    different grid file scrambles the field silently."""

    def test_matching_uuids_pass(self):
        check_grid_uuid("A1B2-C3D4", b"a1b2-c3d4")

    def test_mismatched_uuids_raise(self):
        with self.assertRaises(ValueError):
            check_grid_uuid("a1b2-c3d4", "ffff-0000")

    def test_absent_uuids_are_tolerated(self):
        # not every setup writes the attribute, so this cannot be fatal
        check_grid_uuid(None, "a1b2-c3d4")
        check_grid_uuid("a1b2-c3d4", None)
        check_grid_uuid("", "a1b2-c3d4")

    def test_numpy_string_attributes_are_handled(self):
        # h5py hands back bytes or 1-element arrays for netCDF string attributes
        check_grid_uuid(np.array([b"a1b2-c3d4"]), "a1b2-c3d4")


if __name__ == "__main__":
    unittest.main()
