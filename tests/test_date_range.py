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
Unit tests for ``data_process.date_range``, which decides which samples go into
each yearly file written by the Weatherbench and NCAR converters.

Pure datetime arithmetic, so no MPI and no data are involved. What happens to
an existing file on disk is up to the converters and is not covered here.
"""

import os
import sys
import unittest
import argparse as ap
import datetime as dt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_process.date_range import add_date_range_arguments, date_range_from_args, parse_date, yearly_sample_times


def _utc(year, month, day, hour=0):
    return dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc)


class TestParseDate(unittest.TestCase):
    def test_bare_start_date_is_midnight(self):
        self.assertEqual(parse_date("2026-06-30"), _utc(2026, 6, 30, 0))

    def test_bare_end_date_includes_whole_day(self):
        self.assertEqual(parse_date("2026-06-30", end_of_day=True), _utc(2026, 6, 30, 23))

    def test_explicit_hour_is_kept_for_end_date(self):
        # only a bare date is widened to the end of the day
        self.assertEqual(parse_date("2026-06-30T06", end_of_day=True), _utc(2026, 6, 30, 6))

    def test_naive_values_are_utc(self):
        self.assertEqual(parse_date("2026-01-01T06:00"), _utc(2026, 1, 1, 6))

    def test_aware_values_are_converted_to_utc(self):
        self.assertEqual(parse_date("2026-01-01T06:00+02:00"), _utc(2026, 1, 1, 4))

    def test_command_line_round_trip(self):
        parser = ap.ArgumentParser()
        add_date_range_arguments(parser)
        args = parser.parse_args(["--start_date", "2018-01-01", "--end_date", "2019-12-31"])
        self.assertEqual(date_range_from_args(args), (_utc(2018, 1, 1, 0), _utc(2019, 12, 31, 23)))


class TestYearlySampleTimes(unittest.TestCase):
    def test_full_years_match_the_yearly_grid(self):
        # 2023 is a common year, 2024 a leap year
        result = yearly_sample_times(_utc(2023, 1, 1), _utc(2024, 12, 31, 23), 6)
        self.assertEqual([year for year, _ in result], [2023, 2024])
        self.assertEqual([len(times) for _, times in result], [365 * 4, 366 * 4])
        self.assertEqual(result[0][1][0], _utc(2023, 1, 1, 0))
        self.assertEqual(result[1][1][-1], _utc(2024, 12, 31, 18))

    def test_partial_year_ends_inclusively(self):
        [(year, times)] = yearly_sample_times(_utc(2026, 1, 1), _utc(2026, 6, 30, 23), 6)
        self.assertEqual(year, 2026)
        self.assertEqual(times[0], _utc(2026, 1, 1, 0))
        self.assertEqual(times[-1], _utc(2026, 6, 30, 18))
        self.assertEqual(len(times), 181 * 4)

    def test_end_on_a_sample_is_included(self):
        [(_, times)] = yearly_sample_times(_utc(2026, 1, 1), _utc(2026, 1, 2, 12), 6)
        self.assertEqual(times[-1], _utc(2026, 1, 2, 12))

    def test_off_grid_start_rounds_up(self):
        [(_, times)] = yearly_sample_times(_utc(2026, 3, 1, 7), _utc(2026, 3, 1, 23), 6)
        self.assertEqual(times, [_utc(2026, 3, 1, 12), _utc(2026, 3, 1, 18)])

    def test_grid_is_anchored_at_the_start_of_each_year(self):
        # a start inside the year must not shift the grid of the following year
        result = yearly_sample_times(_utc(2025, 12, 31, 7), _utc(2026, 1, 1, 23), 6)
        self.assertEqual(result[0], (2025, [_utc(2025, 12, 31, 12), _utc(2025, 12, 31, 18)]))
        self.assertEqual(result[1][1][0], _utc(2026, 1, 1, 0))

    def test_range_across_years_is_split(self):
        result = yearly_sample_times(_utc(2025, 7, 1), _utc(2027, 3, 31, 23), 24)
        self.assertEqual([year for year, _ in result], [2025, 2026, 2027])
        for year, times in result:
            self.assertTrue(all(t.year == year for t in times))
        self.assertEqual(result[0][1][0], _utc(2025, 7, 1))
        self.assertEqual(len(result[1][1]), 365)
        self.assertEqual(result[2][1][-1], _utc(2027, 3, 31))

    def test_year_without_samples_is_dropped(self):
        # nothing in 2025 lies on the 6h grid after 19Z on December 31st
        result = yearly_sample_times(_utc(2025, 12, 31, 19), _utc(2026, 1, 1, 0), 6)
        self.assertEqual(result, [(2026, [_utc(2026, 1, 1, 0)])])

    def test_range_without_samples_raises(self):
        with self.assertRaises(ValueError):
            yearly_sample_times(_utc(2026, 1, 1, 1), _utc(2026, 1, 1, 5), 6)

    def test_end_before_start_raises(self):
        with self.assertRaises(ValueError):
            yearly_sample_times(_utc(2026, 2, 1), _utc(2026, 1, 1), 6)


if __name__ == "__main__":
    unittest.main()
