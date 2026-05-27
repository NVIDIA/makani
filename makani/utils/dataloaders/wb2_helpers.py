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
import numpy as np


# ---------------------------------------------------------------------------
# ERA5 short name <-> WeatherBench2 long name mappings
# ---------------------------------------------------------------------------

surface_variables = {
    "u10m":  "10m_u_component_of_wind",
    "v10m":  "10m_v_component_of_wind",
    "t2m":   "2m_temperature",
    "d2":    "2m_dewpoint_temperature",
    "u100m": "100m_u_component_of_wind",
    "v100m": "100m_v_component_of_wind",
    "tp":    "total_precipitation_6hr",
    "sp":    "surface_pressure",
    "msl":   "mean_sea_level_pressure",
    "tcwv":  "total_column_water_vapour",
    "sst":   "sea_surface_temperature",
}

atmospheric_variables = {
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "r": "relative_humidity",
    "q": "specific_humidity",
}

# reverse lookups
surface_variables_inv = {v: k for k, v in surface_variables.items()}
atmospheric_variables_inv = {v: k for k, v in atmospheric_variables.items()}


# ---------------------------------------------------------------------------
# Channel name helpers
# ---------------------------------------------------------------------------

def split_convert_channel_names(makani_channel_names):
    """Split ERA5/makani channel names into surface and atmospheric groups with WB2 names.

    Returns
    -------
    atmospheric_channel_names : list[str]
        ERA5 short prefixes for atmospheric variables (e.g. ["z", "u"]).
    atmospheric_channel_names_wb2 : list[str]
        Corresponding WB2 long names (e.g. ["geopotential", "u_component_of_wind"]).
    surface_channel_names : list[str]
        ERA5 short names for surface variables (e.g. ["t2m", "u10m"]).
    surface_channel_names_wb2 : list[str]
        Corresponding WB2 long names.
    atmospheric_levels : list[int]
        Sorted list of distinct pressure levels found in the channel list.
    """
    from makani.utils.features import get_channel_groups

    atmospheric_channel_indices, surface_channel_indices, _, _, atmospheric_levels = get_channel_groups(makani_channel_names)

    pat = re.compile(r"^(.*?)\d{1,}$")
    atmospheric_channel_names = sorted(list(set(
        pat.match(makani_channel_names[k]).group(1)
        for k in atmospheric_channel_indices
    )))
    atmospheric_channel_names_wb2 = [atmospheric_variables[c] for c in atmospheric_channel_names]

    surface_channel_names = sorted([makani_channel_names[k] for k in surface_channel_indices])
    surface_channel_names_wb2 = [surface_variables[c] for c in surface_channel_names]

    atmospheric_levels = sorted(list(atmospheric_levels))

    return (
        atmospheric_channel_names,
        atmospheric_channel_names_wb2,
        surface_channel_names,
        surface_channel_names_wb2,
        atmospheric_levels,
    )


def build_wb2_channel_map(channel_names, level_values=None):
    """Build a per-channel WB2 conversion table for online zarr reading.

    This is the online equivalent of what the offline conversion scripts do:
    it maps each ERA5/makani channel name to the zarr variable name and, for
    atmospheric variables, the integer index into the store's ``level``
    coordinate array.

    Parameters
    ----------
    channel_names : list[str]
        ERA5/makani channel names in channel-index order (e.g. ["u10m", "z500"]).
    level_values : array-like of int, optional
        Ordered pressure levels available in the zarr store
        (e.g. [50, 100, 150, ..., 1000]).  Required when the channel list
        contains any atmospheric variable.

    Returns
    -------
    list of (zarr_variable_name: str, level_array_idx: int | None)
        One entry per channel.  ``level_array_idx`` is the integer position
        in the store's ``level`` coordinate array, *not* the pressure value.
    """
    level_to_idx = {}
    if level_values is not None:
        level_to_idx = {int(lv): i for i, lv in enumerate(level_values)}

    channel_map = []
    for ch_name in channel_names:
        m = re.search(r"[0-9]{1,4}$", ch_name)
        if m is not None and ch_name != "d2":
            pressure = int(m.group())
            prefix = ch_name[: m.start()]
            if prefix not in atmospheric_variables:
                raise ValueError(
                    f"Unknown atmospheric variable prefix '{prefix}' for channel '{ch_name}'. "
                    f"Known prefixes: {list(atmospheric_variables)}"
                )
            zarr_name = atmospheric_variables[prefix]
            if pressure not in level_to_idx:
                raise ValueError(
                    f"Pressure level {pressure} hPa (channel '{ch_name}') not found in zarr store. "
                    f"Available levels: {sorted(level_to_idx)}"
                )
            channel_map.append((zarr_name, level_to_idx[pressure]))
        else:
            if ch_name not in surface_variables:
                raise ValueError(
                    f"Unknown surface variable '{ch_name}'. "
                    f"Known names: {list(surface_variables)}"
                )
            channel_map.append((surface_variables[ch_name], None))

    return channel_map


# ---------------------------------------------------------------------------
# GCS storage helper (useful for online reads from Google Cloud Storage)
# ---------------------------------------------------------------------------

def gcs_storage_options():
    """Return gcsfs storage options, falling back to anonymous access if no ADC found."""
    try:
        import google.auth
        google.auth.default()
        return {}
    except Exception:
        return {"token": "anon"}
