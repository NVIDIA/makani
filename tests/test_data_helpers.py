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

import datetime as dt
import unittest

import numpy as np

from makani.utils.dataloaders.data_helpers import (
    get_lat_lon_grid,
    get_data_normalization,
    get_climatology,
    get_timestamp,
    get_date_from_string,
    get_date_from_timestamp,
    get_timedelta_from_timestamp,
    get_date_ranges,
)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import set_seed, get_default_parameters

UTC = dt.timezone.utc


# ===========================================================================
# 1. get_lat_lon_grid
# ===========================================================================
class TestGetLatLonGrid(unittest.TestCase):
    """
    Contracts:
      - latitude  = linspace(90, -90, nlat, endpoint=True)   → includes both poles
      - longitude = linspace(0, 360, nlon, endpoint=False)   → excludes 360 (wraps)
    """

    def test_lat_shape(self):
        lat, _ = get_lat_lon_grid((36, 72))
        self.assertEqual(lat.shape, (36,))

    def test_lon_shape(self):
        _, lon = get_lat_lon_grid((36, 72))
        self.assertEqual(lon.shape, (72,))

    def test_lat_starts_at_north_pole(self):
        lat, _ = get_lat_lon_grid((36, 72))
        self.assertAlmostEqual(lat[0], 90.0)

    def test_lat_ends_at_south_pole(self):
        """endpoint=True means -90 is included."""
        lat, _ = get_lat_lon_grid((36, 72))
        self.assertAlmostEqual(lat[-1], -90.0)

    def test_lat_monotonically_decreasing(self):
        """Latitude goes north-to-south."""
        lat, _ = get_lat_lon_grid((36, 72))
        self.assertTrue(np.all(np.diff(lat) < 0))

    def test_lon_starts_at_zero(self):
        _, lon = get_lat_lon_grid((36, 72))
        self.assertAlmostEqual(lon[0], 0.0)

    def test_lon_does_not_reach_360(self):
        """endpoint=False means 360 is excluded."""
        _, lon = get_lat_lon_grid((36, 72))
        self.assertLess(lon[-1], 360.0)

    def test_lon_monotonically_increasing(self):
        _, lon = get_lat_lon_grid((36, 72))
        self.assertTrue(np.all(np.diff(lon) > 0))

    def test_single_point(self):
        """Edge case: 1×1 grid returns scalar arrays."""
        lat, lon = get_lat_lon_grid((1, 1))
        self.assertEqual(lat.shape, (1,))
        self.assertEqual(lon.shape, (1,))
        self.assertAlmostEqual(lat[0],  90.0)
        self.assertAlmostEqual(lon[0],   0.0)


# ===========================================================================
# 2. get_data_normalization
# ===========================================================================
class TestGetDataNormalization(unittest.TestCase):

    def setUp(self):
        self.params = get_default_parameters()  # normalization = "none"

    def test_normalization_none_string_returns_none(self):
        """'none' does not match any branch → (None, None)."""
        bias, scale = get_data_normalization(self.params)
        self.assertIsNone(bias)
        self.assertIsNone(scale)

    def test_no_normalization_attr_returns_none(self):
        """Without a normalization attribute at all → (None, None)."""
        params = get_default_parameters()
        del params.normalization          # remove the attribute
        bias, scale = get_data_normalization(params)
        self.assertIsNone(bias)
        self.assertIsNone(scale)

    def test_minmax_missing_file_raises(self):
        """minmax mode raises when min_path does not exist."""
        self.params.normalization = "minmax"
        self.params.min_path = "/nonexistent/mins.npy"
        self.params.max_path = "/nonexistent/maxs.npy"
        with self.assertRaises(FileNotFoundError):
            get_data_normalization(self.params)

    def test_zscore_missing_file_raises(self):
        """zscore mode raises when global_means_path does not exist."""
        self.params.normalization = "zscore"
        self.params.global_means_path = "/nonexistent/means.npy"
        self.params.global_stds_path  = "/nonexistent/stds.npy"
        with self.assertRaises(FileNotFoundError):
            get_data_normalization(self.params)


# ===========================================================================
# 3. get_climatology (synthetic-data path only — no file I/O)
# ===========================================================================
class TestGetClimatology(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.params.enable_synthetic_data = True
        self.params.N_out_channels  = 5
        self.params.img_crop_shape_x = 32
        self.params.img_crop_shape_y = 64
        self.params.subsampling_factor = 1

    def test_shape(self):
        """Synthetic climatology has shape (1, C, H, W)."""
        clim = get_climatology(self.params)
        self.assertEqual(
            clim.shape,
            (1, self.params.N_out_channels,
             self.params.img_crop_shape_x,
             self.params.img_crop_shape_y),
        )

    def test_all_zeros(self):
        """Synthetic climatology is all zeros."""
        clim = get_climatology(self.params)
        self.assertTrue(np.all(clim == 0.0))

    def test_dtype_float32(self):
        clim = get_climatology(self.params)
        self.assertEqual(clim.dtype, np.float32)

    def test_subsampling_halves_spatial_dims(self):
        """subsampling_factor=2 halves both spatial dimensions."""
        self.params.subsampling_factor = 2
        clim = get_climatology(self.params)
        self.assertEqual(clim.shape[-2], self.params.img_crop_shape_x // 2)
        self.assertEqual(clim.shape[-1], self.params.img_crop_shape_y // 2)

    def test_subsampling_preserves_zeros(self):
        """Subsampled synthetic climatology is still all zeros."""
        self.params.subsampling_factor = 2
        clim = get_climatology(self.params)
        self.assertTrue(np.all(clim == 0.0))


# ===========================================================================
# 4. get_timestamp
# ===========================================================================
class TestGetTimestamp(unittest.TestCase):
    """
    get_timestamp(year, hour) → Jan 1 <year> 00:00 UTC + <hour> hours.
    """

    def test_hour_zero_is_jan_1_midnight(self):
        ts = get_timestamp(2020, 0)
        self.assertEqual(ts, dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC))

    def test_hour_24_is_jan_2(self):
        ts = get_timestamp(2020, 24)
        self.assertEqual(ts, dt.datetime(2020, 1, 2, 0, 0, 0, tzinfo=UTC))

    def test_hour_offset(self):
        ts = get_timestamp(2020, 6)
        self.assertEqual(ts, dt.datetime(2020, 1, 1, 6, 0, 0, tzinfo=UTC))

    def test_is_utc(self):
        ts = get_timestamp(2020, 0)
        self.assertEqual(ts.tzinfo, UTC)

    def test_year_boundary(self):
        """hour = 8760 (365 days) should land at the start of the next year."""
        ts = get_timestamp(2019, 365 * 24)
        self.assertEqual(ts.year, 2020)
        self.assertEqual(ts.month, 1)
        self.assertEqual(ts.day, 1)

    def test_different_years(self):
        ts_a = get_timestamp(2019, 0)
        ts_b = get_timestamp(2020, 0)
        self.assertLess(ts_a, ts_b)


# ===========================================================================
# 5. get_date_from_string
# ===========================================================================
class TestGetDateFromString(unittest.TestCase):

    def test_utc_aware_string_preserved(self):
        """An already-UTC ISO string should parse to the same moment."""
        ts = get_date_from_string("2020-06-15T12:00:00+00:00")
        self.assertEqual(ts, dt.datetime(2020, 6, 15, 12, 0, 0, tzinfo=UTC))

    def test_offset_aware_string_converted_to_utc(self):
        """A +05:00 string should be converted: 17:00+05 → 12:00 UTC."""
        ts = get_date_from_string("2020-06-15T17:00:00+05:00")
        self.assertEqual(ts.tzinfo, UTC)
        self.assertEqual(ts.hour, 12)
        self.assertEqual(ts.date(), dt.date(2020, 6, 15))

    def test_naive_string_returns_timezone_aware(self):
        """A naive ISO string must produce a timezone-aware result."""
        ts = get_date_from_string("2020-06-15T12:00:00")
        self.assertIsNotNone(ts.tzinfo)

    def test_result_is_datetime(self):
        ts = get_date_from_string("2020-01-01")
        self.assertIsInstance(ts, dt.datetime)


# ===========================================================================
# 6. get_date_from_timestamp
# ===========================================================================
class TestGetDateFromTimestamp(unittest.TestCase):

    def test_unix_epoch(self):
        """Timestamp 0 is 1970-01-01 00:00:00 UTC."""
        ts = get_date_from_timestamp(0)
        self.assertEqual(ts, dt.datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC))

    def test_one_hour(self):
        ts = get_date_from_timestamp(3600)
        self.assertEqual(ts, dt.datetime(1970, 1, 1, 1, 0, 0, tzinfo=UTC))

    def test_one_day(self):
        ts = get_date_from_timestamp(86400)
        self.assertEqual(ts, dt.datetime(1970, 1, 2, 0, 0, 0, tzinfo=UTC))

    def test_is_utc(self):
        ts = get_date_from_timestamp(0)
        self.assertEqual(ts.tzinfo, UTC)


# ===========================================================================
# 7. get_timedelta_from_timestamp
# ===========================================================================
class TestGetTimedeltaFromTimestamp(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(get_timedelta_from_timestamp(0), dt.timedelta(0))

    def test_one_hour(self):
        self.assertEqual(get_timedelta_from_timestamp(3600), dt.timedelta(hours=1))

    def test_one_day(self):
        self.assertEqual(get_timedelta_from_timestamp(86400), dt.timedelta(days=1))

    def test_fractional_seconds(self):
        self.assertEqual(
            get_timedelta_from_timestamp(90),
            dt.timedelta(minutes=1, seconds=30),
        )


# ===========================================================================
# 8. get_date_ranges
# ===========================================================================
class TestGetDateRanges(unittest.TestCase):

    def _date(self, y=2020, m=1, d=1, h=12):
        return dt.datetime(y, m, d, h, 0, 0, tzinfo=UTC)

    def test_single_date_produces_one_range(self):
        ranges = get_date_ranges([self._date()], lookback_hours=6, lookahead_hours=24)
        self.assertEqual(len(ranges), 1)

    def test_multiple_dates_produce_matching_count(self):
        dates = [self._date(d=1), self._date(d=2), self._date(d=3)]
        ranges = get_date_ranges(dates, lookback_hours=6, lookahead_hours=6)
        self.assertEqual(len(ranges), 3)

    def test_range_start_is_date_minus_lookback(self):
        date = self._date()
        ranges = get_date_ranges([date], lookback_hours=6, lookahead_hours=0)
        start, _ = ranges[0]
        self.assertEqual(start, date - dt.timedelta(hours=6))

    def test_range_end_is_date_plus_lookahead(self):
        date = self._date()
        ranges = get_date_ranges([date], lookback_hours=0, lookahead_hours=24)
        _, end = ranges[0]
        self.assertEqual(end, date + dt.timedelta(hours=24))

    def test_zero_offsets_give_point_range(self):
        date = self._date()
        ranges = get_date_ranges([date], lookback_hours=0, lookahead_hours=0)
        start, end = ranges[0]
        self.assertEqual(start, date)
        self.assertEqual(end, date)

    def test_start_before_end_when_offsets_nonzero(self):
        date = self._date()
        ranges = get_date_ranges([date], lookback_hours=6, lookahead_hours=24)
        start, end = ranges[0]
        self.assertLess(start, end)

    def test_each_range_is_independent(self):
        """Ranges for different dates should not overlap when dates are far apart."""
        d1 = self._date(d=1)
        d2 = self._date(d=10)
        ranges = get_date_ranges([d1, d2], lookback_hours=6, lookahead_hours=6)
        _, end1  = ranges[0]
        start2, _ = ranges[1]
        self.assertLess(end1, start2)


if __name__ == "__main__":
    unittest.main()
