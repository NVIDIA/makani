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
Unit tests for mapping makani channel names onto the data archives.

Covers the channel name splitter in ``makani.utils.features``, the grouping in
``makani.utils.dataloaders.channel_helpers`` that all readers share, and the
per archive tables and conventions on top of it: NSF NCAR ERA5, WeatherBench2
and ICON. Everything here is pure computation on names, attributes and
datetimes, so no files, no network and no MPI are involved.
"""

import os
import sys
import unittest
from typing import NamedTuple, Optional
import datetime as dt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.features import get_channel_groups, split_channel_name
from makani.utils.dataloaders.channel_helpers import (
    ChannelGroup,
    build_channel_groups,
    resolve_variable,
)
from makani.utils.dataloaders.ncar_helpers import (
    ACCUM_INIT_HOURS,
    accumulated_variables as ncar_accumulated_variables,
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
from makani.utils.dataloaders.wb2_helpers import (
    Wb2Variable,
    surface_variables,
    atmospheric_variables,
    surface_variables_inv,
    atmospheric_variables_inv,
    surface_wb2_name,
    atmospheric_wb2_name,
    split_convert_channel_names,
    build_wb2_channel_map,
)
from makani.utils.dataloaders.icon_helpers import (
    GRAVITY,
    IconVariable,
    accumulated_variables as icon_accumulated_variables,
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


# ---------------------------------------------------------------------------
# Channel name splitting: makani.utils.features
# ---------------------------------------------------------------------------
#
# Unit tests for the channel name classification in ``makani.utils.features``.
#
# ``split_channel_name`` is the single definition of how a channel name is split
# into a variable prefix and a pressure level. Every data source reader goes
# through it, so the cases pinned down here are the contract the NCAR, WB2 and
# ICON helpers all rely on; they used to be reimplemented per source and had
# started to drift apart.


class TestSplitChannelName(unittest.TestCase):
    """
    Channel names are classified by a trailing number: ``z500`` is the variable
    ``z`` on 500 hPa, while ``u10m`` and ``t2m`` end in a letter and are surface
    fields. ``d2`` is the one name that looks like a level but is not.
    """

    def test_pressure_level_channels(self):
        self.assertEqual(split_channel_name("z500"), ("z", 500))
        self.assertEqual(split_channel_name("t850"), ("t", 850))
        self.assertEqual(split_channel_name("u1000"), ("u", 1000))
        self.assertEqual(split_channel_name("q50"), ("q", 50))

    def test_surface_channels_have_no_level(self):
        for name in ["t2m", "u10m", "v10m", "u100m", "sp", "msl", "tcwv", "sst", "tp"]:
            with self.subTest(channel=name):
                self.assertEqual(split_channel_name(name), (name, None))

    def test_d2_is_not_read_as_a_level(self):
        # "d2" would otherwise parse as variable "d" on 2 hPa
        self.assertEqual(split_channel_name("d2"), ("d2", None))

    def test_prefixes_longer_than_three_characters(self):
        # the pattern only requires letters before the digits; the prefix itself
        # is everything ahead of them, which the hydrometeor channels rely on
        self.assertEqual(split_channel_name("clwc500"), ("clwc", 500))
        self.assertEqual(split_channel_name("ciwc1000"), ("ciwc", 1000))
        self.assertEqual(split_channel_name("cswc250"), ("cswc", 250))

    def test_digits_without_a_letter_prefix_are_not_a_level(self):
        # this is where the per-source copies used to disagree: without the
        # letter gate, names like these parsed as an atmospheric variable
        for name in ["1000", "x12345"]:
            with self.subTest(channel=name):
                self.assertIsNone(split_channel_name(name)[1])


class TestGetChannelGroupsUsesTheSplitter(unittest.TestCase):
    """``get_channel_groups`` classifies through the same function, so the two
    cannot disagree about what is atmospheric."""

    def test_groups_match_the_splitter(self):
        names = ["u10m", "t2m", "z500", "t500", "z850", "t850", "d2"]
        atmo, surf, _, _, levels = get_channel_groups(names)

        expected_atmo = {idx for idx, name in enumerate(names) if split_channel_name(name)[1] is not None}
        expected_surf = set(range(len(names))) - expected_atmo

        self.assertEqual(set(atmo), expected_atmo)
        self.assertEqual(set(surf), expected_surf)
        self.assertEqual(sorted(levels), [500, 850])

    def test_dewpoint_is_grouped_as_surface(self):
        atmo, surf, _, _, _ = get_channel_groups(["d2", "z500", "t500"])
        self.assertEqual(sorted(surf), [0])
        self.assertEqual(sorted(atmo), [1, 2])


# ---------------------------------------------------------------------------
# Channel grouping: makani.utils.dataloaders.channel_helpers
# ---------------------------------------------------------------------------
#
# Unit tests for ``makani.utils.dataloaders.channel_helpers``, the channel
# grouping shared by the NCAR, WB2 and ICON readers.
#
# The distinction the tests care about most is alternatives versus components: a
# table entry listing several variables can mean "pick whichever the file has"
# (ICON's tot_prec or pr) or "sum all of them" (NCAR's tp = lsp + cp). The two
# look identical in the data and mean the opposite, so both are pinned here with
# a fake descriptor rather than through any one reader's tables.


class FakeVariable(NamedTuple):
    """Stand-in for a reader's descriptor, with the fields the contract names."""

    name: str
    kind: str
    units: Optional[str] = None
    accumulation: str = "none"


Z = FakeVariable("geopot", "pl", units="m2 s-2")
ZG = FakeVariable("zg", "pl", units="m")
T = FakeVariable("temp", "pl", units="K")
T2M = FakeVariable("t_2m", "sfc", units="K")
LSP = FakeVariable("lsp", "accum", units="m", accumulation="since_start")
CP = FakeVariable("cp", "accum", units="m", accumulation="since_start")
TOT_PREC = FakeVariable("tot_prec", "accum", accumulation="since_start")
PR = FakeVariable("pr", "accum", accumulation="rate")

ATMOSPHERIC = {"z": (Z, ZG), "t": (T,)}
SURFACE = {"t2m": (T2M,)}


class TestResolveVariable(unittest.TestCase):
    """Resolution returns the *components* of the chosen candidate, always a tuple."""

    def test_prefers_the_first_available_candidate(self):
        self.assertEqual(resolve_variable((Z, ZG), ["zg", "geopot"]), (Z,))

    def test_falls_through_to_a_later_candidate(self):
        self.assertEqual(resolve_variable((Z, ZG), ["zg", "temp"]), (ZG,))

    def test_returns_none_when_nothing_matches(self):
        self.assertIsNone(resolve_variable((Z, ZG), ["temp"]))

    def test_without_a_file_the_first_candidate_wins(self):
        self.assertEqual(resolve_variable((Z, ZG), None), (Z,))

    def test_a_bare_descriptor_is_a_valid_entry(self):
        # readers whose channels map one to one (NCAR, WB2) write the descriptor
        # directly rather than wrapping it in a one element tuple
        self.assertEqual(resolve_variable(Z, ["geopot"]), (Z,))
        self.assertEqual(resolve_variable(Z, None), (Z,))

    def test_empty_entry_resolves_to_none(self):
        self.assertIsNone(resolve_variable(None, None))
        self.assertIsNone(resolve_variable((), None))

    # ---- alternatives versus components ------------------------------------

    def test_summed_components_are_returned_together(self):
        # one candidate made of two components: both are needed
        entry = ((LSP, CP),)
        self.assertEqual(resolve_variable(entry, ["lsp", "cp"]), (LSP, CP))

    def test_summed_components_require_every_part(self):
        # a partial match is not a match: summing lsp alone would be wrong
        entry = ((LSP, CP),)
        self.assertIsNone(resolve_variable(entry, ["lsp"]))

    def test_alternatives_need_only_one(self):
        # the same shape of data, opposite meaning: either one suffices
        entry = (TOT_PREC, PR)
        self.assertEqual(resolve_variable(entry, ["pr"]), (PR,))
        self.assertEqual(resolve_variable(entry, ["tot_prec"]), (TOT_PREC,))

    def test_a_summed_candidate_can_have_alternatives(self):
        # prefer the sum, fall back to the single variable when the file lacks
        # the second component
        entry = ((LSP, CP), LSP)
        self.assertEqual(resolve_variable(entry, ["lsp", "cp"]), (LSP, CP))
        self.assertEqual(resolve_variable(entry, ["lsp"]), (LSP,))


class TestBuildChannelGroups(unittest.TestCase):

    def test_levels_of_one_variable_share_a_group(self):
        groups = build_channel_groups(["z500", "z850", "z1000"], ATMOSPHERIC)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], ChannelGroup("pl", "z", [Z], [0, 1, 2], [500, 850, 1000]))

    def test_channel_indices_track_positions_in_the_original_list(self):
        groups = build_channel_groups(["z500", "t850", "z1000", "t2m"], ATMOSPHERIC, SURFACE)
        by_name = {group.name: group for group in groups}

        self.assertEqual(by_name["z"].channel_indices, [0, 2])
        self.assertEqual(by_name["z"].levels, [500, 1000])
        self.assertEqual(by_name["t"].channel_indices, [1])
        self.assertEqual(by_name["t2m"].channel_indices, [3])

    def test_pressure_level_groups_come_first(self):
        groups = build_channel_groups(["t2m", "z500", "tp", "t850"], ATMOSPHERIC, SURFACE, {"tp": ((LSP, CP),)})
        kinds = [group.kind for group in groups]

        self.assertEqual(kinds[:2], ["pl", "pl"])
        self.assertEqual(sorted(kinds[2:]), ["accum", "sfc"])

    def test_surface_groups_have_no_levels(self):
        group = build_channel_groups(["t2m"], ATMOSPHERIC, SURFACE)[0]

        self.assertEqual(group.kind, "sfc")
        self.assertIsNone(group.levels)
        self.assertEqual(group.variables, [T2M])

    def test_accumulated_group_carries_every_component(self):
        group = build_channel_groups(["tp"], ATMOSPHERIC, SURFACE, {"tp": ((LSP, CP),)})[0]

        self.assertEqual(group.kind, "accum")
        self.assertEqual(group.variables, [LSP, CP])

    def test_resolution_follows_the_file(self):
        nwp = build_channel_groups(["z500"], ATMOSPHERIC, available=["geopot"])[0]
        cmip = build_channel_groups(["z500"], ATMOSPHERIC, available=["zg"])[0]

        self.assertEqual(nwp.variables, [Z])
        self.assertEqual(cmip.variables, [ZG])

    def test_unresolvable_channels_raise(self):
        with self.subTest(desc="unknown atmospheric prefix"):
            with self.assertRaises(ValueError):
                build_channel_groups(["xyz500"], ATMOSPHERIC, SURFACE)

        with self.subTest(desc="unknown surface name"):
            with self.assertRaises(ValueError):
                build_channel_groups(["not_a_variable"], ATMOSPHERIC, SURFACE)

        with self.subTest(desc="known channel the file does not provide"):
            with self.assertRaises(ValueError):
                build_channel_groups(["z500"], ATMOSPHERIC, available=["temp"])

    def test_error_message_names_the_source(self):
        with self.assertRaises(ValueError) as ctx:
            build_channel_groups(["xyz500"], ATMOSPHERIC, source="ICON")
        self.assertIn("ICON", str(ctx.exception))

    def test_unresolvable_channels_can_be_skipped(self):
        groups = build_channel_groups(
            ["z500", "xyz500", "not_a_variable", "t2m"], ATMOSPHERIC, SURFACE, skip_missing_channels=True
        )

        self.assertEqual(sorted(group.name for group in groups), ["t2m", "z"])
        # the surviving channels keep their original indices, holes and all
        by_name = {group.name: group for group in groups}
        self.assertEqual(by_name["z"].channel_indices, [0])
        self.assertEqual(by_name["t2m"].channel_indices, [3])

    def test_empty_channel_list(self):
        self.assertEqual(build_channel_groups([], ATMOSPHERIC, SURFACE), [])


# ---------------------------------------------------------------------------
# NSF NCAR ERA5: makani.utils.dataloaders.ncar_helpers
# ---------------------------------------------------------------------------
#
# Unit tests for ``makani.utils.dataloaders.ncar_helpers``, the channel mapping,
# object key and accumulation window arithmetic behind the NSF NCAR ERA5 (RDA
# d633000) converter. The channel name splitter it builds on is shared with the
# other readers and is covered in the channel name splitting section above.
#
# Everything here is pure computation on names and datetimes, so no S3 access, no
# MPI and no data fixtures are involved. The reads themselves live in
# ``data_process/sources/ncar.py`` and are not covered.


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
        candidates = ncar_accumulated_variables["tp"]

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


# ---------------------------------------------------------------------------
# WeatherBench2: makani.utils.dataloaders.wb2_helpers
# ---------------------------------------------------------------------------


class TestVariableSemantics(unittest.TestCase):
    """
    The tables carry more than the WB2 name: the kind, the units and how the
    variable relates to the channel in time. Those fields are what a converter
    reasons about, and unlike the name they are never exercised by simply
    reading a store, so they are pinned here.
    """

    def test_every_entry_is_a_descriptor(self):
        for table in (surface_variables, atmospheric_variables):
            for channel, variable in table.items():
                with self.subTest(channel=channel):
                    self.assertIsInstance(variable, Wb2Variable)
                    self.assertTrue(variable.name)
                    self.assertIn(variable.kind, ("pl", "sfc", "accum"))
                    self.assertTrue(variable.units)

    def test_atmospheric_entries_are_pressure_level(self):
        for prefix, variable in atmospheric_variables.items():
            with self.subTest(prefix=prefix):
                self.assertEqual(variable.kind, "pl")

    def test_instantaneous_surface_fields_do_not_accumulate(self):
        for channel, variable in surface_variables.items():
            if variable.kind == "sfc":
                with self.subTest(channel=channel):
                    self.assertEqual(variable.accumulation, "none")

    def test_precipitation_carries_a_fixed_window(self):
        # the window is baked into the WB2 name ("..._6hr"), so a store fixes it
        # and the reader neither differences nor integrates anything
        tp = surface_variables["tp"]

        self.assertEqual(tp.name, "total_precipitation_6hr")
        self.assertEqual(tp.kind, "accum")
        self.assertEqual(tp.accumulation, "window")


class TestNameLookups(unittest.TestCase):
    """
    The two accessors are what the converters use, so that the mapping tables
    stay an implementation detail of wb2_helpers. They also turn an unknown
    name into a readable ValueError rather than a bare KeyError coming out of
    the middle of a conversion run.
    """

    def test_surface_lookup(self):
        self.assertEqual(surface_wb2_name("t2m"), "2m_temperature")
        self.assertEqual(surface_wb2_name("msl"), "mean_sea_level_pressure")

    def test_atmospheric_lookup(self):
        self.assertEqual(atmospheric_wb2_name("z"), "geopotential")
        self.assertEqual(atmospheric_wb2_name("q"), "specific_humidity")

    def test_lookups_agree_with_the_tables(self):
        for name in surface_variables:
            with self.subTest(channel=name):
                self.assertEqual(surface_wb2_name(name), surface_variables[name].name)
        for prefix in atmospheric_variables:
            with self.subTest(prefix=prefix):
                self.assertEqual(atmospheric_wb2_name(prefix), atmospheric_variables[prefix].name)

    def test_unknown_surface_name_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            surface_wb2_name("not_a_variable")
        self.assertIn("not_a_variable", str(ctx.exception))
        self.assertIn("Known names", str(ctx.exception))

    def test_unknown_atmospheric_prefix_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            atmospheric_wb2_name("xyz")
        self.assertIn("xyz", str(ctx.exception))
        self.assertIn("Known prefixes", str(ctx.exception))

    def test_atmospheric_prefix_is_not_a_full_channel_name(self):
        # the accessor takes the prefix, not "z500"; passing the channel name is
        # a mistake that has to fail rather than silently miss
        with self.assertRaises(ValueError):
            atmospheric_wb2_name("z500")


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
            self.assertEqual(surface_variables_inv[wb2.name], era5)

    def test_inverse_atmospheric_round_trips(self):
        for era5, wb2 in atmospheric_variables.items():
            self.assertEqual(atmospheric_variables_inv[wb2.name], era5)

    def test_known_surface_mappings(self):
        self.assertEqual(surface_variables["u10m"].name, "10m_u_component_of_wind")
        self.assertEqual(surface_variables["t2m"].name, "2m_temperature")
        self.assertEqual(surface_variables["msl"].name, "mean_sea_level_pressure")

    def test_known_atmospheric_mappings(self):
        self.assertEqual(atmospheric_variables["z"].name, "geopotential")
        self.assertEqual(atmospheric_variables["u"].name, "u_component_of_wind")
        self.assertEqual(atmospheric_variables["t"].name, "temperature")


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
        self.assertEqual(
            result,
            [
                ("10m_u_component_of_wind", None),
                ("2m_temperature", None),
                ("mean_sea_level_pressure", None),
            ],
        )

    def test_single_atmospheric_channel(self):
        result = build_wb2_channel_map(["z500"], level_values=[100, 500, 850])
        self.assertEqual(result, [("geopotential", 1)])  # 500 is at index 1

    def test_atmospheric_level_index_matches_position(self):
        levels = [50, 100, 200, 500, 850, 1000]
        result = build_wb2_channel_map(["u500"], level_values=levels)
        self.assertEqual(result[0], ("u_component_of_wind", 3))  # 500 at index 3

    def test_mixed_surface_and_atmospheric(self):
        result = build_wb2_channel_map(
            ["u10m", "t2m", "z500", "u850"],
            level_values=[500, 850],
        )
        self.assertEqual(
            result,
            [
                ("10m_u_component_of_wind", None),
                ("2m_temperature", None),
                ("geopotential", 0),  # 500 at index 0
                ("u_component_of_wind", 1),  # 850 at index 1
            ],
        )

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
        # balanced grid: same variables (z, u) at each level (500, 850)
        channels = ["u10m", "t2m", "z500", "u500", "z850", "u850"]
        atm_names, atm_wb2, surf_names, surf_wb2, levels = split_convert_channel_names(channels)
        self.assertIn("z", atm_names)
        self.assertIn("u", atm_names)
        self.assertNotIn("t2m", atm_names)
        self.assertIn("geopotential", atm_wb2)
        self.assertIn("u_component_of_wind", atm_wb2)
        self.assertIn("u10m", surf_names)
        self.assertIn("t2m", surf_names)
        self.assertIn(500, levels)
        self.assertIn(850, levels)

    def test_surface_only(self):
        atm_names, atm_wb2, surf_names, surf_wb2, levels = split_convert_channel_names(["u10m", "t2m"])
        self.assertEqual(atm_names, [])
        self.assertEqual(atm_wb2, [])
        self.assertEqual(levels, [])
        self.assertIn("u10m", surf_names)
        self.assertIn("t2m", surf_names)
        self.assertIn("10m_u_component_of_wind", surf_wb2)

    def test_atmospheric_only(self):
        atm_names, atm_wb2, surf_names, surf_wb2, levels = split_convert_channel_names(["z500", "t500", "u500"])
        self.assertEqual(surf_names, [])
        self.assertEqual(surf_wb2, [])
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
        atm_names, atm_wb2, surf_names, surf_wb2, _ = split_convert_channel_names(["u10m", "t2m", "z500", "u500"])
        self.assertEqual(len(atm_names), len(atm_wb2))
        self.assertEqual(len(surf_names), len(surf_wb2))


# ---------------------------------------------------------------------------
# ICON: makani.utils.dataloaders.icon_helpers
# ---------------------------------------------------------------------------
#
# Unit tests for ``makani.utils.dataloaders.icon_helpers``: the decoding of ICON
# netCDF conventions and the mapping from ICON variable names onto makani
# channels.
#
# The candidate resolution and channel grouping themselves are shared with the
# other readers and are covered in the channel grouping section above; what is pinned
# here is which ICON names each makani channel maps to.
#
# Everything here works on plain arrays and attribute values, so no ICON file, no
# h5py and no grid is needed. The reads and the regridding live in the converter
# and are not covered.


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
        candidates = icon_accumulated_variables["tp"]

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
