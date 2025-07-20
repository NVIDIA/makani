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

import re
from collections import OrderedDict


def get_auxiliary_channels(
    add_zenith=False,
    add_grid=False,
    grid_type=None,
    grid_num_frequencies=0,
    add_orography=False,
    add_landmask=False,
    landmask_preprocessing='floor',
    add_soiltype=False,
    add_copernicus_emb=False,
    n_noise_chan=0,
    **kwargs,
):
    """
    Auxiliary routine to return the list of appended channel names. Must match behavior of preprocessor and dataloader
    """
    channel_names = []

    if add_zenith:
        channel_names.append("xzen")

    if n_noise_chan > 0:
        for c in range(n_noise_chan):
            channel_names.append(f"xnoise{c}")

    if add_grid:
        if grid_type == "sinusoidal":
            for f in range(1, grid_num_frequencies + 1):
                channel_names += [f"xsgrlat{f}", f"xsgrlon{f}"]
        else:
            channel_names += ["xgrlat", "xgrlon"]

    if add_orography:
        channel_names.append("xoro")

    if add_landmask:
        if landmask_preprocessing in ["floor", "round"]:
            channel_names += ["xlsml", "xlsms"]
        elif landmask_preprocessing == "raw":
            channel_names += ["xlsm"]

    if add_soiltype:
        channel_names += [f"xst{i}" for i in range(8)]

    if add_copernicus_emb:
        channel_names += [f"xcop{i}" for i in range(8)]


    return channel_names


def get_water_channels(channel_names):
    """
    Helper routine to extract water channels from channel names
    """

    water_chans = []
    for c, ch in enumerate(channel_names):
        if ch[0] in {"q", "r"} or ch == "tcwv":
            water_chans.append(c)

    return water_chans


def get_wind_channels(channel_names):
    """
    Helper routine to extract water channels from channel names
    """

    wind_chans = []
    for c, ch in enumerate(channel_names):
        if ch[0] == "u":
            vc = channel_names.index("v" + ch[1:])
            wind_chans = wind_chans + [c, vc]

    return wind_chans


def get_channel_groups(channel_names, aux_channel_names=[]):
    """
    Helper routine to extract indices of atmospheric, surface and auxiliary variables and group them into their respective groups
    """

    atmo_groups = OrderedDict()
    atmo_chans = []
    surf_chans = []
    aux_chans = []

    # parse channel names and group variables by pressure level/surface variables
    for idx, chn in enumerate(channel_names):
        # check if pattern matches an atmospheric variable
        if re.search("[a-z]{1,3}[0-9]{1,4}$", chn) is not None:
            pressure_level = int(re.search("[0-9]{1,4}$", chn).group())
            if pressure_level not in atmo_groups.keys():
                atmo_groups[pressure_level] = []
            atmo_groups[pressure_level].append(idx)
        else:
            surf_chans.append(idx)

    # check the correctness of the groups (they should all come in the same order and same number of vars)
    n_atmo_chans = None
    for plvl, idx in atmo_groups.items():
        if n_atmo_chans is None:
            n_atmo_chans = len(idx)
        else:
            assert n_atmo_chans == len(idx)

        atmo_chans += idx

    # append the auxiliary variable to the surface channels
    aux_chans = [idx + len(channel_names) for idx in range(len(aux_channel_names))]

    return atmo_chans, surf_chans, aux_chans, atmo_groups.keys()
