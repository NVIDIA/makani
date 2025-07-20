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


# this routine computes the matching pressure levels between two pl variables
# with prefix1 and prefix 2 respectively. pmin and pmax are the minimum and maximum pressure levels considered
def get_matching_channels_pl(channel_names, prefix1, prefix2, p_min, p_max, revert=True):
    # we better use regexp
    import re

    # analyse list of channel names, extract geopotential and temperatures:
    p1_pat = re.compile(r"^" + prefix1 + r"\d{1,}$")
    p2_pat = re.compile(r"^" + prefix2 + r"\d{1,}$")
    p1_chans = [x for x in channel_names if (p1_pat.match(x) is not None)]
    p2_chans = [x for x in channel_names if (p2_pat.match(x) is not None)]

    # extract common pressure levels
    p1_pressures = [int(x.replace(prefix1, "")) for x in p1_chans]
    p2_pressures = [int(x.replace(prefix2, "")) for x in p2_chans]

    # check which are the common pressure levels:
    pressures = sorted([x for x in p1_pressures if ((x in p2_pressures) and (x >= p_min) and (x <= p_max))], reverse=revert)

    # create an indexlist for z-channels
    p1_idx = [channel_names.index(f"{prefix1}{p}") for p in pressures]
    p2_idx = [channel_names.index(f"{prefix2}{p}") for p in pressures]

    return p1_idx, p2_idx, pressures
