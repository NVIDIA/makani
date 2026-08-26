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
Unit tests for the channel name classification in ``makani.utils.features``.

``split_channel_name`` is the single definition of how a channel name is split
into a variable prefix and a pressure level. Every data source reader goes
through it, so the cases pinned down here are the contract the NCAR, WB2 and
ICON helpers all rely on; they used to be reimplemented per source and had
started to drift apart.
"""

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.features import get_channel_groups, split_channel_name


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


if __name__ == "__main__":
    unittest.main()
