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
Unit tests for ``makani.utils.dataloaders.channel_helpers``, the channel
grouping shared by the NCAR, WB2 and ICON readers.

The distinction the tests care about most is alternatives versus components: a
table entry listing several variables can mean "pick whichever the file has"
(ICON's tot_prec or pr) or "sum all of them" (NCAR's tp = lsp + cp). The two
look identical in the data and mean the opposite, so both are pinned here with
a fake descriptor rather than through any one reader's tables.
"""

import os
import sys
import unittest
from typing import NamedTuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.dataloaders.channel_helpers import (
    ChannelGroup,
    build_channel_groups,
    resolve_variable,
)


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


if __name__ == "__main__":
    unittest.main()
