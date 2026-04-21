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

import datetime as dt
import unittest

import numpy as np

from makani.third_party.climt import zenith_angle as v1
from makani.third_party.climt import zenith_angle_v2 as v2

from .testutils import compare_arrays


def _sample_times():
    """A mix of epochs across year, season and time-of-day."""
    return np.asarray([
        dt.datetime(2000, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2002, 6, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2003, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2018, 3, 21, 6, 30, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2025, 9, 22, 18, 45, 0, tzinfo=dt.timezone.utc),
    ])


def _sample_grid(step=5.0):
    """Regular (lon, lat) degree grid covering the globe."""
    lon = np.arange(0.0, 360.0, step)
    lat = np.arange(-90.0, 90.0 + step / 2, step)[::-1]
    return np.meshgrid(lon, lat)  # lon_grid, lat_grid, both [H, W]


def _jc_from_time(mod, time):
    """Julian centuries since 2000 computed via the given module's `_days_from_2000`.

    Mirrors v2's internal recipe (float32 divisor) so we feed the jc-taking v2
    helpers an input consistent with how they would normally be called.
    """
    return mod._days_from_2000(time) / np.float32(36525.0)


def _wrapped_angle_diff(a, b):
    """|a - b| wrapped into [-π, π] for angular quantities."""
    return np.angle(np.exp(1j * (a.astype(np.float64) - b.astype(np.float64))))


class TestDaysFrom2000(unittest.TestCase):
    """Both versions expose the same API and should agree exactly."""

    def test_matches(self, atol=0.0, rtol=0.0, verbose=False):
        t = _sample_times()
        a = v1._days_from_2000(t)
        b = v2._days_from_2000(t)
        self.assertTrue(compare_arrays("days_from_2000", a, b, atol=atol, rtol=rtol, verbose=verbose))
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(b.dtype, np.float32)


class TestObliquityStar(unittest.TestCase):
    """Both versions take julian centuries and should agree exactly."""

    def test_matches(self, atol=0.0, rtol=0.0, verbose=False):
        t = _sample_times()
        jc = _jc_from_time(v2, t)  # either module works; they're equivalent here
        a = v1._obliquity_star(jc)
        b = v2._obliquity_star(jc)
        self.assertTrue(compare_arrays("obliquity_star", a, b, atol=atol, rtol=rtol, verbose=verbose))


class TestGreenwichMeanSiderealTime(unittest.TestCase):
    """v1 takes `model_time`, v2 takes `jc`. Same math — compare with tolerance
    because v1 computes jc in float64 internally while v2 uses float32."""

    def test_matches(self, atol=1e-3, rtol=0.0, verbose=False):
        t = _sample_times()
        jc = _jc_from_time(v2, t)
        a = v1._greenwich_mean_sidereal_time(t)
        b = v2._greenwich_mean_sidereal_time(jc)
        # GMST is an angle in [0, 2π); wrap the difference before comparing.
        diff = _wrapped_angle_diff(a, b)
        self.assertTrue(compare_arrays("gmst (wrapped)", diff, np.zeros_like(diff), atol=atol, rtol=rtol, verbose=verbose))


class TestSunEclipticLongitude(unittest.TestCase):
    """v1 takes `model_time`, v2 takes `jc`. Angular quantity — compare wrapped."""

    def test_matches(self, atol=1e-4, rtol=0.0, verbose=False):
        t = _sample_times()
        jc = _jc_from_time(v2, t)
        a = v1._sun_ecliptic_longitude(t)
        b = v2._sun_ecliptic_longitude(jc)
        diff = _wrapped_angle_diff(a, b)
        self.assertTrue(compare_arrays("sun_ecliptic_longitude (wrapped)", diff, np.zeros_like(diff), atol=atol, rtol=rtol, verbose=verbose))


class TestCosZenithAngle(unittest.TestCase):
    """Full public API: both versions accept (time, lon, lat) in degrees."""

    def test_grid_matches(self, atol=1e-4, rtol=1e-4, verbose=False):
        t = _sample_times()
        lon_grid, lat_grid = _sample_grid(step=5.0)
        a = v1.cos_zenith_angle(t, lon_grid, lat_grid)
        b = v2.cos_zenith_angle(t, lon_grid, lat_grid)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(compare_arrays("cos_zenith_angle grid", a, b, atol=atol, rtol=rtol, verbose=verbose))

    def test_grid_values_in_range(self):
        """cos(zenith) must lie in [-1, 1]."""
        t = _sample_times()
        lon_grid, lat_grid = _sample_grid(step=10.0)
        b = v2.cos_zenith_angle(t, lon_grid, lat_grid)
        self.assertTrue(np.all(np.abs(b) <= 1.0 + 1e-5))

    def test_cache_hit_gives_same_result(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Second call with the same grid uses the identity cache — result
        must still match the first call and v1."""
        t = _sample_times()
        lon_grid, lat_grid = _sample_grid(step=10.0)
        b1 = v2.cos_zenith_angle(t, lon_grid, lat_grid)
        b2 = v2.cos_zenith_angle(t, lon_grid, lat_grid)  # cache hit
        # Cache hit must be bit-exact; this is not a tolerance-tunable check.
        self.assertTrue(compare_arrays("cache hit (bit-exact)", b1, b2, atol=0.0, rtol=0.0, verbose=verbose))
        a = v1.cos_zenith_angle(t, lon_grid, lat_grid)
        self.assertTrue(compare_arrays("cache hit vs v1", a, b2, atol=atol, rtol=rtol, verbose=verbose))

    def test_cache_invalidates_on_new_grid(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Swapping in a different grid must not return stale cached values."""
        t = _sample_times()
        lon_a, lat_a = _sample_grid(step=10.0)
        lon_b, lat_b = _sample_grid(step=20.0)
        ra = v2.cos_zenith_angle(t, lon_a, lat_a)
        rb = v2.cos_zenith_angle(t, lon_b, lat_b)
        self.assertEqual(ra.shape, (len(t),) + lon_a.shape)
        self.assertEqual(rb.shape, (len(t),) + lon_b.shape)
        # And the new call should match v1 on the same new grid.
        rb_ref = v1.cos_zenith_angle(t, lon_b, lat_b)
        self.assertTrue(compare_arrays("cache invalidated", rb, rb_ref, atol=atol, rtol=rtol, verbose=verbose))

    def test_single_time(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Single timestamp (0-d array) should still produce a [1, H, W] result."""
        t = np.asarray(dt.datetime(2020, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc))
        lon_grid, lat_grid = _sample_grid(step=20.0)
        a = v1.cos_zenith_angle(t, lon_grid, lat_grid)
        b = v2.cos_zenith_angle(t, lon_grid, lat_grid)
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(compare_arrays("single time", a, b, atol=atol, rtol=rtol, verbose=verbose))


class TestCosZenithAngleEdgeCases(unittest.TestCase):
    """Edge cases around calendar boundaries and astronomical reference points."""

    def _cmp(self, msg, t, lon_grid, lat_grid, atol=1e-4, rtol=1e-4, verbose=False):
        a = v1.cos_zenith_angle(t, lon_grid, lat_grid)
        b = v2.cos_zenith_angle(t, lon_grid, lat_grid)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(compare_arrays(msg, a, b, atol=atol, rtol=rtol, verbose=verbose))
        self.assertTrue(np.all(np.abs(b) <= 1.0 + 1e-5))
        return b

    def test_reference_epoch(self, atol=1e-4, rtol=1e-4, verbose=False):
        """2000-01-01 12:00 UTC is the reference epoch — days_from_2000 must be 0."""
        t = np.asarray([dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)])
        self.assertEqual(float(v1._days_from_2000(t)[0]), 0.0)
        self.assertEqual(float(v2._days_from_2000(t)[0]), 0.0)
        lon_grid, lat_grid = _sample_grid(step=20.0)
        self._cmp("reference_epoch", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_leap_day_transition_2024(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Span Feb 28 → Feb 29 → Mar 1 across 2024 (leap year) at 1h cadence."""
        start = dt.datetime(2024, 2, 28, 0, 0, 0, tzinfo=dt.timezone.utc)
        t = np.asarray([start + dt.timedelta(hours=h) for h in range(72)])
        lon_grid, lat_grid = _sample_grid(step=10.0)
        b = self._cmp("leap_day_2024", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)
        # Consecutive hours must differ smoothly — no jumps > 0.5 in cos(zenith).
        self.assertLess(float(np.max(np.abs(np.diff(b, axis=0)))), 0.5)

    def test_leap_day_transition_2000(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Year 2000 is a century leap year — Feb 29 exists."""
        t = np.asarray([
            dt.datetime(2000, 2, 28, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2000, 2, 29, 0, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2000, 2, 29, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2000, 2, 29, 23, 59, 59, tzinfo=dt.timezone.utc),
            dt.datetime(2000, 3, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2000, 3, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
        ])
        lon_grid, lat_grid = _sample_grid(step=15.0)
        self._cmp("leap_day_2000", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_non_leap_year_feb_march(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Non-leap year: Feb 28 → Mar 1 (no Feb 29)."""
        t = np.asarray([
            dt.datetime(2023, 2, 28, 23, 59, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2023, 3, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2023, 3, 1, 0, 1, 0, tzinfo=dt.timezone.utc),
        ])
        lon_grid, lat_grid = _sample_grid(step=15.0)
        self._cmp("non_leap_feb_march", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_year_boundary_leap_year_end(self, atol=1e-4, rtol=1e-4, verbose=False):
        """End of leap year (day 366) → start of next year."""
        t = np.asarray([
            dt.datetime(2024, 12, 31, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc),
            dt.datetime(2025, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
        ])
        lon_grid, lat_grid = _sample_grid(step=15.0)
        self._cmp("leap_year_end_boundary", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_year_boundary_non_leap(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Rollover from a regular year."""
        t = np.asarray([
            dt.datetime(2023, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
        ])
        lon_grid, lat_grid = _sample_grid(step=15.0)
        self._cmp("non_leap_year_boundary", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_millennium_boundary(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Cross the 2000 epoch from below (pre-epoch → post-epoch)."""
        t = np.asarray([
            dt.datetime(1999, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc),
            dt.datetime(2000, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),  # epoch exactly
            dt.datetime(2000, 1, 1, 12, 0, 1, tzinfo=dt.timezone.utc),
        ])
        lon_grid, lat_grid = _sample_grid(step=20.0)
        self._cmp("millennium_boundary", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_pre_2000_dates(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Dates before the epoch → negative days_from_2000."""
        t = np.asarray([
            dt.datetime(1950, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(1980, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(1985, 12, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(1999, 3, 20, 6, 0, 0, tzinfo=dt.timezone.utc),
        ])
        lon_grid, lat_grid = _sample_grid(step=20.0)
        self._cmp("pre_2000_dates", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_far_future(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Keep the expansions numerically stable a century out."""
        t = np.asarray([
            dt.datetime(2099, 12, 31, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2100, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2100, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
        ])
        lon_grid, lat_grid = _sample_grid(step=20.0)
        self._cmp("far_future", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_longitude_wrap(self, atol=1e-5, rtol=0.0, verbose=False):
        """cos(zenith) at lon=0 and lon=360 must match (same physical point)."""
        t = np.asarray([dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)])
        lat = np.array([[0.0, 30.0, -30.0, 60.0]])  # [1, 4]
        lon0 = np.zeros_like(lat)
        lon360 = np.full_like(lat, 360.0)
        r0 = v2.cos_zenith_angle(t, lon0, lat)
        r360 = v2.cos_zenith_angle(t, lon360, lat)
        self.assertTrue(compare_arrays("longitude wrap", r0, r360, atol=atol, rtol=rtol, verbose=verbose))

    def test_solstice_polar_day_night(self):
        """North pole at June solstice noon → sun up; at December solstice → sun down.
        South pole is the mirror. This is a physics sanity check independent of v1."""
        pole_lat = np.array([[89.9, -89.9]])  # [1, 2]
        lon = np.zeros_like(pole_lat)

        june = np.asarray([dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)])
        dec = np.asarray([dt.datetime(2024, 12, 21, 12, 0, 0, tzinfo=dt.timezone.utc)])

        cz_june = v2.cos_zenith_angle(june, lon, pole_lat)[0, 0]  # [T, 1, 2]
        cz_dec = v2.cos_zenith_angle(dec, lon, pole_lat)[0, 0]

        # North pole in June → sun above horizon; in December → below.
        self.assertGreater(float(cz_june[0]), 0.0)    # N pole, June
        self.assertLess(float(cz_june[1]), 0.0)        # S pole, June
        self.assertLess(float(cz_dec[0]), 0.0)         # N pole, December
        self.assertGreater(float(cz_dec[1]), 0.0)      # S pole, December

    def test_equator_solar_noon_near_equinox(self):
        """At the vernal equinox, the sun is near the equator; at the sub-solar
        longitude at noon UTC the sun should be nearly overhead (cos_z ≈ 1)."""
        # Vernal equinox 2024 is ~Mar 20 03:06 UTC. Pick a nearby time and a lon
        # where local solar noon coincides with UTC noon-ish.
        t = np.asarray([dt.datetime(2024, 3, 20, 12, 0, 0, tzinfo=dt.timezone.utc)])
        # Scan longitudes at equator; find the peak.
        lon = np.linspace(-180.0, 180.0, 361)[None, :]
        lat = np.zeros_like(lon)
        cz = v2.cos_zenith_angle(t, lon, lat)[0, 0]
        self.assertGreater(float(cz.max()), 0.995)

    def test_day_cycle_smoothness(self, atol=1e-4, rtol=1e-4, verbose=False):
        """24 evenly-spaced times in a single day should give a smooth curve at
        any given location."""
        base = dt.datetime(2024, 5, 15, 0, 0, 0, tzinfo=dt.timezone.utc)
        t = np.asarray([base + dt.timedelta(hours=h) for h in range(24)])
        lon = np.array([[0.0]])
        lat = np.array([[45.0]])
        cz_v2 = v2.cos_zenith_angle(t, lon, lat).reshape(-1)
        cz_v1 = v1.cos_zenith_angle(t, lon, lat).reshape(-1)
        self.assertTrue(compare_arrays("day_cycle", cz_v1, cz_v2, atol=atol, rtol=rtol, verbose=verbose))
        # At 45°N the curve has one maximum per day; d/dh should change sign once
        # or twice (sunrise, sunset). Bound the largest hourly step to rule out
        # discontinuities from a calendar-boundary bug.
        self.assertLess(float(np.max(np.abs(np.diff(cz_v2)))), 0.3)


if __name__ == "__main__":
    unittest.main()
