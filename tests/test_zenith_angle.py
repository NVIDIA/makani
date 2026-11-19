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

from makani.third_party.climt import zenith_angle

from .testutils import compare_arrays


def _sample_times():
    """A mix of epochs across year, season and time-of-day."""
    return np.asarray(
        [
            dt.datetime(2000, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2002, 6, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2003, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2018, 3, 21, 6, 30, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2025, 9, 22, 18, 45, 0, tzinfo=dt.timezone.utc),
        ]
    )


def _sample_grid(step=5.0):
    """Regular (lon, lat) degree grid covering the globe."""
    lon = np.arange(0.0, 360.0, step)
    lat = np.arange(-90.0, 90.0 + step / 2, step)[::-1]
    return np.meshgrid(lon, lat)  # lon_grid, lat_grid, both [H, W]


class TestDaysFrom2000(unittest.TestCase):
    """Days since 2000-01-01 12:00 UTC, the epoch of the expansions."""

    def test_days(self):
        t = np.asarray(
            [
                dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2000, 1, 2, 12, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(1999, 12, 31, 0, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2001, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
            ]
        )
        days = zenith_angle._days_from_2000(t)
        self.assertEqual(days.dtype, np.float32)
        # 2000 is a leap year: 366 days to 2001-01-01 12:00
        np.testing.assert_array_equal(days, np.array([0.0, 1.0, -1.5, 366.0], dtype=np.float32))


class TestCosZenithAngle(unittest.TestCase):
    """The public API: (time, lon, lat), the coordinates in degrees, give [T, H, W]."""

    def test_grid_shape(self):
        t = _sample_times()
        lon_grid, lat_grid = _sample_grid(step=5.0)
        self.assertEqual(zenith_angle.cos_zenith_angle(t, lon_grid, lat_grid).shape, (len(t),) + lon_grid.shape)

    def test_grid_values_in_range(self):
        """cos(zenith) must lie in [-1, 1]."""
        t = _sample_times()
        lon_grid, lat_grid = _sample_grid(step=10.0)
        b = zenith_angle.cos_zenith_angle(t, lon_grid, lat_grid)
        self.assertTrue(np.all(np.abs(b) <= 1.0 + 1e-5))

    def test_cache_hit_gives_same_result(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Second call with the same grid uses the identity cache — result
        must match the first call bit for bit."""
        t = _sample_times()
        lon_grid, lat_grid = _sample_grid(step=10.0)
        b1 = zenith_angle.cos_zenith_angle(t, lon_grid, lat_grid)
        b2 = zenith_angle.cos_zenith_angle(t, lon_grid, lat_grid)  # cache hit
        # Cache hit must be bit-exact; this is not a tolerance-tunable check.
        self.assertTrue(compare_arrays("cache hit (bit-exact)", b1, b2, atol=0.0, rtol=0.0, verbose=verbose))

    def test_cache_invalidates_on_new_grid(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Swapping in a different grid must not return stale cached values."""
        t = _sample_times()
        lon_a, lat_a = _sample_grid(step=10.0)
        lon_b, lat_b = _sample_grid(step=20.0)
        ra = zenith_angle.cos_zenith_angle(t, lon_a, lat_a)
        rb = zenith_angle.cos_zenith_angle(t, lon_b, lat_b)
        self.assertEqual(ra.shape, (len(t),) + lon_a.shape)
        self.assertEqual(rb.shape, (len(t),) + lon_b.shape)
        # And the new call should match an uncached one on copies of the new grid.
        rb_ref = zenith_angle.cos_zenith_angle(t, lon_b.copy(), lat_b.copy())
        self.assertTrue(compare_arrays("cache invalidated", rb, rb_ref, atol=atol, rtol=rtol, verbose=verbose))

    def test_single_time(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Single timestamp (0-d array) should still produce a [1, H, W] result."""
        t = np.asarray(dt.datetime(2020, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc))
        lon_grid, lat_grid = _sample_grid(step=20.0)
        b = zenith_angle.cos_zenith_angle(t, lon_grid, lat_grid)
        self.assertEqual(b.shape, (1,) + lon_grid.shape)
        reference = zenith_angle.cos_zenith_angle(np.atleast_1d(t), lon_grid.copy(), lat_grid.copy())
        self.assertTrue(compare_arrays("single time", b, reference, atol=atol, rtol=rtol, verbose=verbose))


class TestCosZenithAngleEdgeCases(unittest.TestCase):
    """
    Edge cases around calendar boundaries and astronomical reference points: the result is
    well formed there. Its values at these dates are anchored by the pvlib reference points below.
    """

    def _cmp(self, msg, t, lon_grid, lat_grid, atol=1e-4, rtol=1e-4, verbose=False):
        b = zenith_angle.cos_zenith_angle(t, lon_grid, lat_grid)
        self.assertEqual(b.shape, (len(t),) + lon_grid.shape, msg)
        self.assertTrue(np.all(np.isfinite(b)), msg)
        self.assertTrue(np.all(np.abs(b) <= 1.0 + 1e-5), msg)
        return b

    def test_reference_epoch(self, atol=1e-4, rtol=1e-4, verbose=False):
        """2000-01-01 12:00 UTC is the reference epoch — days_from_2000 must be 0."""
        t = np.asarray([dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)])
        self.assertEqual(float(zenith_angle._days_from_2000(t)[0]), 0.0)
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
        t = np.asarray(
            [
                dt.datetime(2000, 2, 28, 12, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2000, 2, 29, 0, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2000, 2, 29, 12, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2000, 2, 29, 23, 59, 59, tzinfo=dt.timezone.utc),
                dt.datetime(2000, 3, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2000, 3, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
            ]
        )
        lon_grid, lat_grid = _sample_grid(step=15.0)
        self._cmp("leap_day_2000", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_non_leap_year_feb_march(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Non-leap year: Feb 28 → Mar 1 (no Feb 29)."""
        t = np.asarray(
            [
                dt.datetime(2023, 2, 28, 23, 59, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2023, 3, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2023, 3, 1, 0, 1, 0, tzinfo=dt.timezone.utc),
            ]
        )
        lon_grid, lat_grid = _sample_grid(step=15.0)
        self._cmp("non_leap_feb_march", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_year_boundary_leap_year_end(self, atol=1e-4, rtol=1e-4, verbose=False):
        """End of leap year (day 366) → start of next year."""
        t = np.asarray(
            [
                dt.datetime(2024, 12, 31, 12, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2024, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc),
                dt.datetime(2025, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
            ]
        )
        lon_grid, lat_grid = _sample_grid(step=15.0)
        self._cmp("leap_year_end_boundary", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_year_boundary_non_leap(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Rollover from a regular year."""
        t = np.asarray(
            [
                dt.datetime(2023, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc),
                dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            ]
        )
        lon_grid, lat_grid = _sample_grid(step=15.0)
        self._cmp("non_leap_year_boundary", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_millennium_boundary(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Cross the 2000 epoch from below (pre-epoch → post-epoch)."""
        t = np.asarray(
            [
                dt.datetime(1999, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc),
                dt.datetime(2000, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),  # epoch exactly
                dt.datetime(2000, 1, 1, 12, 0, 1, tzinfo=dt.timezone.utc),
            ]
        )
        lon_grid, lat_grid = _sample_grid(step=20.0)
        self._cmp("millennium_boundary", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_pre_2000_dates(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Dates before the epoch → negative days_from_2000."""
        t = np.asarray(
            [
                dt.datetime(1950, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(1980, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(1985, 12, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(1999, 3, 20, 6, 0, 0, tzinfo=dt.timezone.utc),
            ]
        )
        lon_grid, lat_grid = _sample_grid(step=20.0)
        self._cmp("pre_2000_dates", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_far_future(self, atol=1e-4, rtol=1e-4, verbose=False):
        """Keep the expansions numerically stable a century out."""
        t = np.asarray(
            [
                dt.datetime(2099, 12, 31, 12, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2100, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2100, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
            ]
        )
        lon_grid, lat_grid = _sample_grid(step=20.0)
        self._cmp("far_future", t, lon_grid, lat_grid, atol=atol, rtol=rtol, verbose=verbose)

    def test_longitude_wrap(self, atol=1e-5, rtol=0.0, verbose=False):
        """cos(zenith) at lon=0 and lon=360 must match (same physical point)."""
        t = np.asarray([dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)])
        lat = np.array([[0.0, 30.0, -30.0, 60.0]])  # [1, 4]
        lon0 = np.zeros_like(lat)
        lon360 = np.full_like(lat, 360.0)
        r0 = zenith_angle.cos_zenith_angle(t, lon0, lat)
        r360 = zenith_angle.cos_zenith_angle(t, lon360, lat)
        self.assertTrue(compare_arrays("longitude wrap", r0, r360, atol=atol, rtol=rtol, verbose=verbose))

    def test_solstice_polar_day_night(self):
        """North pole at June solstice noon → sun up; at December solstice → sun down.
        South pole is the mirror. This is a physics sanity check."""
        pole_lat = np.array([[89.9, -89.9]])  # [1, 2]
        lon = np.zeros_like(pole_lat)

        june = np.asarray([dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)])
        dec = np.asarray([dt.datetime(2024, 12, 21, 12, 0, 0, tzinfo=dt.timezone.utc)])

        cz_june = zenith_angle.cos_zenith_angle(june, lon, pole_lat)[0, 0]  # [T, 1, 2]
        cz_dec = zenith_angle.cos_zenith_angle(dec, lon, pole_lat)[0, 0]

        # North pole in June → sun above horizon; in December → below.
        self.assertGreater(float(cz_june[0]), 0.0)  # N pole, June
        self.assertLess(float(cz_june[1]), 0.0)  # S pole, June
        self.assertLess(float(cz_dec[0]), 0.0)  # N pole, December
        self.assertGreater(float(cz_dec[1]), 0.0)  # S pole, December

    def test_equator_solar_noon_near_equinox(self):
        """At the vernal equinox, the sun is near the equator; at the sub-solar
        longitude at noon UTC the sun should be nearly overhead (cos_z ≈ 1)."""
        # Vernal equinox 2024 is ~Mar 20 03:06 UTC. Pick a nearby time and a lon
        # where local solar noon coincides with UTC noon-ish.
        t = np.asarray([dt.datetime(2024, 3, 20, 12, 0, 0, tzinfo=dt.timezone.utc)])
        # Scan longitudes at equator; find the peak.
        lon = np.linspace(-180.0, 180.0, 361)[None, :]
        lat = np.zeros_like(lon)
        cz = zenith_angle.cos_zenith_angle(t, lon, lat)[0, 0]
        self.assertGreater(float(cz.max()), 0.995)

    def test_day_cycle_smoothness(self, atol=1e-4, rtol=1e-4, verbose=False):
        """24 evenly-spaced times in a single day should give a smooth curve at
        any given location."""
        base = dt.datetime(2024, 5, 15, 0, 0, 0, tzinfo=dt.timezone.utc)
        t = np.asarray([base + dt.timedelta(hours=h) for h in range(24)])
        lon = np.array([[0.0]])
        lat = np.array([[45.0]])
        cz = zenith_angle.cos_zenith_angle(t, lon, lat).reshape(-1)
        # At 45°N the curve has one maximum per day; d/dh should change sign once
        # or twice (sunrise, sunset). Bound the largest hourly step to rule out
        # discontinuities from a calendar-boundary bug.
        self.assertLess(float(np.max(np.abs(np.diff(cz)))), 0.3)


# External reference values: the cosine of the solar zenith angle from pvlib's NREL SPA
# (no refraction), an ephemeris independent of makani's. scripts/generate_zenith_reference.py
# prints them for the points below.
#
# The tolerance below (2e-3) covers float32 rounding and SPA-variant noise.
# makani uses Spencer-style polynomial reductions; pvlib uses full NREL SPA with
# more terms. The largest disagreements show up at low solar elevation,
# where trig is most sensitive to tiny angular differences (~0.1° → ~1.5e-3
# in cos_z). 2e-3 leaves headroom there while still catching any real bug:
# a date off by a day shifts cos_z by ~6e-3 at the moderate elevations of the
# calendar-edge points, which is why those sit at 40° of latitude.
_EDT = dt.timezone(dt.timedelta(hours=-4))  # US east coast, summer DST
_JST = dt.timezone(dt.timedelta(hours=9))  # Japan, no DST
_AEDT = dt.timezone(dt.timedelta(hours=11))  # Sydney, southern DST
_REFERENCE_POINTS = [
    # (datetime, lat, lon, expected_cos_z, label); datetimes in a local zone keep that zone
    (
        dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
        23.44,
        0.0,
        0.999970,
        "june solstice, subsolar lat, GMT",
    ),
    (
        dt.datetime(2024, 12, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
        -23.44,
        0.0,
        0.999977,
        "dec solstice, subsolar lat, GMT",
    ),
    (dt.datetime(2024, 3, 20, 12, 0, 0, tzinfo=dt.timezone.utc), 0.0, 0.0, 0.999488, "march equinox, equator, GMT"),
    (dt.datetime(2024, 9, 22, 12, 0, 0, tzinfo=dt.timezone.utc), 0.0, 0.0, 0.999468, "sept equinox, equator, GMT"),
    (dt.datetime(2024, 7, 4, 14, 0, 0, tzinfo=_EDT), 40.0, -75.0, 0.934621, "summer afternoon, eastern US"),
    (dt.datetime(2024, 1, 15, 15, 0, 0, tzinfo=_JST), 35.0, 139.0, 0.316050, "winter afternoon, tokyo"),
    (dt.datetime(2024, 11, 10, 11, 0, 0, tzinfo=_AEDT), -33.9, 151.2, 0.884919, "spring morning, sydney"),
    # calendar edges, at moderate solar elevation
    (dt.datetime(1950, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc), 40.0, 0.0, 0.453386, "1950, before the epoch"),
    (dt.datetime(1980, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc), -40.0, 0.0, 0.447088, "june 1980, before the epoch"),
    (dt.datetime(1999, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc), 40.0, 180.0, 0.452786, "last second before 2000"),
    (dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc), 40.0, 0.0, 0.453379, "the epoch, 2000-01-01 12:00"),
    (
        dt.datetime(2000, 2, 29, 15, 0, 0, tzinfo=dt.timezone.utc),
        40.0,
        0.0,
        0.479044,
        "feb 29 of the century leap year 2000",
    ),
    (dt.datetime(2023, 3, 1, 0, 0, 0, tzinfo=dt.timezone.utc), 40.0, 180.0, 0.670959, "march 1 after feb 28 of 2023"),
    (dt.datetime(2024, 2, 29, 6, 0, 0, tzinfo=dt.timezone.utc), 40.0, 90.0, 0.670998, "feb 29 of the leap year 2024"),
    (
        dt.datetime(2024, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc),
        40.0,
        180.0,
        0.453904,
        "day 366 of the leap year 2024",
    ),
    (dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc), 40.0, 0.0, 0.454551, "new year 2025"),
    (dt.datetime(2100, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc), 40.0, 180.0, 0.453807, "new year 2100"),
    (dt.datetime(2100, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc), -40.0, 0.0, 0.447263, "june solstice 2100"),
]


class TestExternalReferenceAnchors(unittest.TestCase):
    """Anchor the zenith angle against an independent ephemeris, pvlib's NREL SPA."""

    def test_reference_values(self, atol=2e-3, rtol=0.0):
        for t, lat, lon, expected, label in _REFERENCE_POINTS:
            with self.subTest(label=label):
                t_arr = np.asarray([t])
                lon_arr = np.array([[float(lon)]])
                lat_arr = np.array([[float(lat)]])
                got = float(zenith_angle.cos_zenith_angle(t_arr, lon_arr, lat_arr).reshape(-1)[0])
                self.assertTrue(
                    abs(got - expected) <= atol + rtol * abs(expected),
                    f"{label}: cos_z={got}, expected {expected} (|Δ|={abs(got-expected):.2e}, atol={atol})",
                )


class TestAnalyticalAnchors(unittest.TestCase):
    """Physics-only anchors that hold by construction — no external ephemeris
    needed. These catch bugs that would slip past the reference-value
    checks (e.g. a sign-flipping bug that happens to preserve
    one specific reference value)."""

    def test_antipodal_symmetry(self):
        """For any (t, lon, lat): cos_z(t, lon+180, -lat) = -cos_z(t, lon, lat).

        The antipode of any point sees the sun at the negated elevation — sin
        is odd, so cos(zenith) flips sign exactly. This is a topological fact
        about the sphere, independent of the sun's position."""
        t = np.asarray(
            [
                dt.datetime(2024, 1, 15, 6, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2024, 12, 21, 18, 0, 0, tzinfo=dt.timezone.utc),
            ]
        )
        lat = np.array([[10.0, -30.0, 45.0, -60.0]])
        lon = np.array([[0.0, 90.0, -45.0, 170.0]])
        a = zenith_angle.cos_zenith_angle(t, lon, lat)
        b = zenith_angle.cos_zenith_angle(t, lon + 180.0, -lat)
        self.assertTrue(compare_arrays("antipodal symmetry", a, -b, atol=1e-5, rtol=0.0))

    def test_subsolar_point_exists(self):
        """At any instant, the sun is directly overhead at exactly one point
        on Earth between lat ±23.44° (the subsolar point). A fine global scan
        must contain a point with cos_z ≈ 1."""
        for t_single in [
            dt.datetime(2024, 1, 15, 6, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 4, 7, 14, 30, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 8, 21, 21, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 11, 30, 3, 15, 0, tzinfo=dt.timezone.utc),
        ]:
            with self.subTest(t=t_single.isoformat()):
                t = np.asarray([t_single])
                # 0.25° resolution in the tropical band where the subsolar point lives
                lon = np.linspace(-180.0, 180.0, 1441)[None, :]
                lat = np.linspace(-25.0, 25.0, 201)[:, None]
                lon_grid, lat_grid = np.broadcast_arrays(lon, lat)
                cz = zenith_angle.cos_zenith_angle(t, lon_grid.copy(), lat_grid.copy())
                self.assertGreater(float(cz.max()), 0.9995)

    def test_terminator_crossing(self):
        """A great-circle scan starting from the subsolar point must cross
        cos_z = 0 (the terminator / day-night boundary). We don't need to
        know where the subsolar point is — just that the max and min over
        a global grid straddle zero with the boundary crossed somewhere."""
        t = np.asarray([dt.datetime(2024, 5, 15, 12, 0, 0, tzinfo=dt.timezone.utc)])
        lon = np.linspace(-180.0, 180.0, 361)[None, :]
        lat = np.linspace(-90.0, 90.0, 181)[:, None]
        lon_grid, lat_grid = np.broadcast_arrays(lon, lat)
        cz = zenith_angle.cos_zenith_angle(t, lon_grid.copy(), lat_grid.copy())[0]
        self.assertGreater(float(cz.max()), 0.99)  # subsolar nearby
        self.assertLess(float(cz.min()), -0.99)  # antisolar nearby
        # Continuity: at least one zero-crossing per latitude band away from poles
        mid_band = cz[40:-40]  # exclude polar caps where day/night may be 24h
        sign_changes = np.diff(np.sign(mid_band), axis=1)
        self.assertTrue(
            np.all(np.any(sign_changes != 0, axis=1)), "every mid-latitude band should cross the terminator"
        )


class TestTimezoneContract(unittest.TestCase):
    """Pin down how cos_zenith_angle handles naive datetimes, alternate
    timezones, and numpy.datetime64."""

    def test_alternate_timezone_matches_utc(self):
        """Same instant expressed in UTC vs +09:00 must give identical cos_z.
        Python's datetime subtraction works on instants, not wall-clock time,
        so a tz-aware comparison here is purely a smoke test on the semantics."""
        utc = dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)
        tokyo = utc.astimezone(dt.timezone(dt.timedelta(hours=9)))
        self.assertEqual(utc, tokyo)
        lon_grid, lat_grid = _sample_grid(step=20.0)
        a = zenith_angle.cos_zenith_angle(np.asarray([utc]), lon_grid, lat_grid)
        b = zenith_angle.cos_zenith_angle(np.asarray([tokyo]), lon_grid, lat_grid)
        self.assertTrue(compare_arrays("tz-aware UTC vs +09:00", a, b, atol=0.0, rtol=0.0))

    def test_naive_datetime_contract(self):
        """A naive datetime input must either raise (preferred, explicit) or
        be treated as UTC. We accept either contract; failing here means the
        behavior is undefined and a silent timezone bug is possible."""
        aware = np.asarray([dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)])
        naive = np.asarray([dt.datetime(2024, 6, 21, 12, 0, 0)])
        lon_grid, lat_grid = _sample_grid(step=20.0)
        aware_result = zenith_angle.cos_zenith_angle(aware, lon_grid, lat_grid)
        try:
            naive_result = zenith_angle.cos_zenith_angle(naive, lon_grid, lat_grid)
        except TypeError:
            return  # clean rejection is acceptable
        self.assertTrue(compare_arrays("naive == UTC", aware_result, naive_result, atol=1e-6, rtol=0.0))

    def test_numpy_datetime64_treated_as_utc(self):
        """np.datetime64 has no tzinfo and is conventionally UTC. Production
        dataloaders may produce datetime64 arrays internally, so this needs
        to either work or fail with a clear error."""
        aware = np.asarray([dt.datetime(2024, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)])
        dt64 = np.array(["2024-06-21T12:00:00"], dtype="datetime64[s]")
        lon_grid, lat_grid = _sample_grid(step=20.0)
        a = zenith_angle.cos_zenith_angle(aware, lon_grid, lat_grid)
        try:
            b = zenith_angle.cos_zenith_angle(dt64, lon_grid, lat_grid)
        except TypeError:
            self.skipTest("datetime64 input not supported")
            return
        self.assertTrue(compare_arrays("datetime64 == aware UTC", a, b, atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
