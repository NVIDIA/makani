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
Unit tests for ``makani.utils.dataloaders.ncar_helpers``, the channel mapping,
object key and accumulation window arithmetic behind the NSF NCAR ERA5 (RDA
d633000) converter. The channel name splitter it builds on is shared with the
other readers and is covered in ``test_features.py``.

Everything here is pure computation on names and datetimes, so no S3 access, no
MPI and no data fixtures are involved. The reads themselves live in
``data_process/convert_ncar_era5_to_makani_input.py`` and are not covered.
"""

import os
import sys
import unittest
import datetime as dt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.dataloaders.ncar_helpers import (
    ACCUM_INIT_HOURS,
    accumulated_variables,
    ACCUM_MAX_FORECAST_HOUR,
    NCAR_EPOCH,
    NcarVariable,
    accumulation_key,
    analysis_pl_key,
    analysis_sfc_key,
    build_ncar_channel_groups,
    latest_forecast_init,
    resolve_accumulation_segments,
    to_ncar_hours,
)


def _utc(year, month, day, hour=0):
    return dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc)


class TestBuildNcarChannelGroups(unittest.TestCase):
    """
    Grouping decides how many objects get fetched: all levels of one variable
    live in a single chunk, so they must end up in one group, and each group has
    to remember which channel index of the makani array it fills.
    """

    def test_levels_of_one_variable_share_a_group(self):
        groups = build_ncar_channel_groups(["z500", "z850", "z1000"])

        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group.kind, "pl")
        self.assertEqual(group.name, "z")
        self.assertEqual(group.levels, [500, 850, 1000])
        self.assertEqual(group.channel_indices, [0, 1, 2])

    def test_channel_indices_track_positions_in_the_original_list(self):
        # interleaved variables: each group has to pick up its own positions
        groups = build_ncar_channel_groups(["z500", "t850", "z1000", "t2m"])
        by_name = {group.name: group for group in groups}

        self.assertEqual(by_name["z"].channel_indices, [0, 2])
        self.assertEqual(by_name["z"].levels, [500, 1000])
        self.assertEqual(by_name["t"].channel_indices, [1])
        self.assertEqual(by_name["t2m"].channel_indices, [3])

    def test_pressure_level_groups_come_first(self):
        groups = build_ncar_channel_groups(["t2m", "z500", "tp", "u850"])
        kinds = [group.kind for group in groups]

        self.assertEqual(kinds[:2], ["pl", "pl"])
        self.assertEqual(sorted(kinds[2:]), ["accum", "sfc"])

    def test_surface_and_accumulated_channels_are_classified(self):
        groups = {group.name: group for group in build_ncar_channel_groups(["t2m", "tp"])}

        self.assertEqual(groups["t2m"].kind, "sfc")
        self.assertIsNone(groups["t2m"].levels)
        self.assertEqual(len(groups["t2m"].variables), 1)

        # tp is not shipped directly, it is reconstructed as lsp + cp
        self.assertEqual(groups["tp"].kind, "accum")
        self.assertEqual([var.short_name for var in groups["tp"].variables], ["lsp", "cp"])

    def test_precipitation_is_summed_not_alternatives(self):
        """
        The nesting of the tp entry carries meaning that is easy to lose.

        d633000 has no total precipitation, so tp is lsp PLUS cp: one candidate
        made of two components. Written one level flatter it would read as two
        *alternatives*, the reader would take whichever it saw first, and the
        dataset would silently carry roughly half its precipitation.
        """
        candidates = accumulated_variables["tp"]

        self.assertEqual(len(candidates), 1, "tp must offer exactly one way to be built")
        self.assertEqual([variable.short_name for variable in candidates[0]], ["lsp", "cp"])

    def test_unknown_channels_raise(self):
        with self.subTest(desc="unknown atmospheric prefix"):
            with self.assertRaises(ValueError):
                build_ncar_channel_groups(["xyz500"])

        with self.subTest(desc="unknown surface name"):
            with self.assertRaises(ValueError):
                build_ncar_channel_groups(["not_a_variable"])

    def test_unknown_channels_can_be_skipped(self):
        groups = build_ncar_channel_groups(["z500", "xyz500", "not_a_variable", "t2m"], skip_missing_channels=True)

        self.assertEqual(sorted(group.name for group in groups), ["t2m", "z"])
        # the surviving channels keep their original indices, holes and all
        by_name = {group.name: group for group in groups}
        self.assertEqual(by_name["z"].channel_indices, [0])
        self.assertEqual(by_name["t2m"].channel_indices, [3])


class TestObjectKeys(unittest.TestCase):
    """
    The three streams are laid out differently on S3: pressure levels one file
    per day, surface analysis one per calendar month, accumulations one per half
    month. The keys are built from the variable descriptor and a date.
    """

    pl_variable = NcarVariable("e5.oper.an.pl", "128_129", "z", "sc", "Z")
    sfc_variable = NcarVariable("e5.oper.an.sfc", "128_167", "2t", "sc", "VAR_2T")
    accum_variable = NcarVariable("e5.oper.fc.sfc.accumu", "128_142", "lsp", "sc", "LSP")

    def test_pressure_level_key_spans_one_day(self):
        self.assertEqual(
            analysis_pl_key(self.pl_variable, dt.date(2017, 1, 5)),
            "e5.oper.an.pl/201701/e5.oper.an.pl.128_129_z.ll025sc.2017010500_2017010523.nc",
        )

    def test_surface_key_spans_the_calendar_month(self):
        self.assertEqual(
            analysis_sfc_key(self.sfc_variable, dt.date(2017, 1, 5)),
            "e5.oper.an.sfc/201701/e5.oper.an.sfc.128_167_2t.ll025sc.2017010100_2017013123.nc",
        )

    def test_surface_key_handles_leap_february(self):
        # the end stamp is the last day of the month, which 2016 makes 29
        self.assertTrue(analysis_sfc_key(self.sfc_variable, dt.date(2016, 2, 10)).endswith("2016020100_2016022923.nc"))
        self.assertTrue(analysis_sfc_key(self.sfc_variable, dt.date(2017, 2, 10)).endswith("2017020100_2017022823.nc"))

    def test_accumulation_key_splits_the_month_in_half(self):
        first_half = accumulation_key(self.accum_variable, _utc(2017, 1, 5, 6))
        second_half = accumulation_key(self.accum_variable, _utc(2017, 1, 20, 18))

        self.assertTrue(first_half.endswith("2017010106_2017011606.nc"))
        self.assertTrue(second_half.endswith("2017011606_2017020106.nc"))

    def test_accumulation_key_rolls_over_the_year(self):
        # the second half of December ends in the next January, not month 13
        key = accumulation_key(self.accum_variable, _utc(2017, 12, 20, 18))
        self.assertTrue(key.endswith("2017121606_2018010106.nc"))
        self.assertIn("/201712/", key)


class TestLatestForecastInit(unittest.TestCase):
    """The accumulated stream is initialized at 06Z and 18Z; a time before 06Z
    belongs to the previous day's 18Z run."""

    def test_after_the_evening_run(self):
        for hour in [18, 21, 23]:
            with self.subTest(hour=hour):
                self.assertEqual(latest_forecast_init(_utc(2017, 1, 5, hour)), _utc(2017, 1, 5, 18))

    def test_after_the_morning_run(self):
        for hour in [6, 12, 17]:
            with self.subTest(hour=hour):
                self.assertEqual(latest_forecast_init(_utc(2017, 1, 5, hour)), _utc(2017, 1, 5, 6))

    def test_before_the_morning_run_falls_back_to_the_previous_day(self):
        for hour in [0, 3, 5]:
            with self.subTest(hour=hour):
                self.assertEqual(latest_forecast_init(_utc(2017, 1, 5, hour)), _utc(2017, 1, 4, 18))

    def test_falls_back_across_a_year_boundary(self):
        self.assertEqual(latest_forecast_init(_utc(2018, 1, 1, 0)), _utc(2017, 12, 31, 18))


class TestResolveAccumulationSegments(unittest.TestCase):
    """
    A run only reaches forecast hour 12 while runs start 12 hours apart, so an
    accumulation window may straddle two (or three) runs and has to be cut at the
    run boundaries. The segments are half open forecast hour ranges that must
    tile the window exactly.
    """

    def test_window_inside_a_single_run(self):
        # 06Z .. 12Z is covered by the 06Z run, forecast hours 0..6
        segments = resolve_accumulation_segments(_utc(2017, 1, 5, 12), 6)
        self.assertEqual(segments, [(_utc(2017, 1, 5, 6), 0, 6)])

    def test_window_split_across_two_runs(self):
        # 12Z .. 00Z starts between the 06Z and 18Z runs, so it is cut at 18Z
        segments = resolve_accumulation_segments(_utc(2017, 1, 5, 0), 12)
        self.assertEqual(
            segments,
            [(_utc(2017, 1, 4, 6), 6, 12), (_utc(2017, 1, 4, 18), 0, 6)],
        )

    def test_single_hour_window(self):
        segments = resolve_accumulation_segments(_utc(2017, 1, 5, 0), 1)
        self.assertEqual(segments, [(_utc(2017, 1, 4, 18), 5, 6)])

    def test_segments_tile_the_window(self):
        """
        The property that matters for correctness: whatever the split, the
        segments have to start at the beginning of the window, end at the valid
        time, be contiguous in wall clock, and stay within the forecast range of
        a run. Checked over a full day of valid times and several window lengths.
        """
        for window_hours in [1, 3, 6, 12, 24]:
            for hour in range(24):
                valid_time = _utc(2017, 1, 5, hour)
                segments = resolve_accumulation_segments(valid_time, window_hours)

                with self.subTest(window=window_hours, hour=hour):
                    self.assertTrue(segments)

                    # forecast hour ranges are non-empty and within a run
                    for init_time, start, end in segments:
                        self.assertLess(start, end)
                        self.assertGreaterEqual(start, 0)
                        self.assertLessEqual(end, ACCUM_MAX_FORECAST_HOUR)
                        self.assertIn(init_time.hour, ACCUM_INIT_HOURS)

                    # the hours add up to the requested window
                    self.assertEqual(sum(end - start for _, start, end in segments), window_hours)

                    # and they are contiguous, from window start to valid time
                    bounds = [
                        (init_time + dt.timedelta(hours=start), init_time + dt.timedelta(hours=end))
                        for init_time, start, end in segments
                    ]
                    self.assertEqual(bounds[0][0], valid_time - dt.timedelta(hours=window_hours))
                    self.assertEqual(bounds[-1][1], valid_time)
                    for (_, end), (start, _) in zip(bounds, bounds[1:]):
                        self.assertEqual(end, start)

    def test_non_positive_window_raises(self):
        for window_hours in [0, -1]:
            with self.subTest(window=window_hours):
                with self.assertRaises(ValueError):
                    resolve_accumulation_segments(_utc(2017, 1, 5, 0), window_hours)


class TestToNcarHours(unittest.TestCase):
    """The netCDF time coordinate of d633000 is hours since 1900-01-01."""

    def test_epoch_is_zero(self):
        self.assertEqual(to_ncar_hours(NCAR_EPOCH), 0)

    def test_counts_whole_hours_from_the_epoch(self):
        self.assertEqual(to_ncar_hours(NCAR_EPOCH + dt.timedelta(hours=1)), 1)
        self.assertEqual(to_ncar_hours(NCAR_EPOCH + dt.timedelta(days=1)), 24)
        self.assertEqual(to_ncar_hours(NCAR_EPOCH + dt.timedelta(days=365, hours=7)), 365 * 24 + 7)

    def test_truncates_sub_hour_offsets(self):
        self.assertEqual(to_ncar_hours(NCAR_EPOCH + dt.timedelta(minutes=90)), 1)


if __name__ == "__main__":
    unittest.main()
