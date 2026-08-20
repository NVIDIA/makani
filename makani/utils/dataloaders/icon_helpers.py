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

"""Helpers for reading ICON model output in netCDF form.

ICON writes netCDF4, which is HDF5 underneath, so the files open with ``h5py``
and no netCDF library is needed. What ``h5py`` does *not* do is the decoding
that ``netCDF4``/``xarray`` would normally hide, and ICON adds conventions of
its own on top. This module isolates that decoding and the mapping from ICON
variable names onto makani channels, so that the converter in
``data_process/`` is left with I/O and regridding only, and so that the fiddly
parts can be tested without any ICON data.

Three things here are worth knowing before touching the converter:

* **Time is not CF.** ICON writes ``time:units = "day as %Y%m%d.%f"``, i.e. the
  float ``20170821.333333`` means 2017-08-21 08:00. Interpreting that as an
  offset from an epoch, the way CF ``days since ...`` works, yields dates that
  are wrong but look plausible. See :func:`decode_time`.
* **Values may be packed.** netCDF ``scale_factor``/``add_offset`` packing and
  ``_FillValue`` are applied by the netCDF library, not by HDF5, so reading
  through ``h5py`` returns the raw (possibly integer) values. See
  :func:`decode_values`.
* **Variable names are not standardized.** ICON is run with different physics
  packages and output namelists: NWP setups write ``temp``, ``pres_sfc``,
  ``t_2m``, while AES/CMIP style setups write ``ta``, ``ps``, ``tas`` for the
  same fields. Each makani channel therefore maps to a *list* of candidate
  names, resolved against the variables a given file actually contains. See
  :func:`resolve_variable`.

..note::
    The name tables below are assembled from the common ICON output namelists
    and have not yet been checked against a real file from the producer. Expect
    to extend them; the resolution machinery is what matters and is what the
    tests pin down.
"""

import re
import datetime as dt
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# channel name parsing is shared with the NCAR reader so that "z500" is split the
# same way everywhere; both follow makani.utils.features.get_channel_groups
from makani.utils.dataloaders.ncar_helpers import split_channel_name


# ICON's own time encoding: the integer part is a YYYYMMDD date, the fraction is
# the elapsed part of that day
ICON_TIME_UNITS = "day as %Y%m%d.%f"

# standard gravity, for converting between geopotential (m2/s2, ICON "geopot")
# and geopotential height (m, CMIP "zg")
GRAVITY = 9.80665

_CF_UNITS_PATTERN = re.compile(r"^\s*(day|hour|minute|second)s?\s+since\s+(.+?)\s*$", re.IGNORECASE)

_CF_UNIT_SECONDS = {"day": 86400, "hour": 3600, "minute": 60, "second": 1}

_CF_REFERENCE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)


class IconVariable(NamedTuple):
    """One candidate ICON variable for a makani channel.

    Attributes
    ----------
    name : str
        Variable name as it appears in the netCDF file.
    kind : str
        ``"pl"`` for fields on pressure levels, ``"sfc"`` for single level
        fields, ``"accum"`` for fields that need temporal post-processing.
    accumulation : str
        How the variable relates to the makani channel in time. ``"none"`` for
        instantaneous fields; ``"since_start"`` for totals accumulated from the
        beginning of the run, which have to be differenced between consecutive
        outputs; ``"rate"`` for fluxes, which have to be multiplied by the
        length of the window.
    units : str, optional
        Units the variable is expected to carry, for the converter to check
        against the file. Mismatches usually mean a different variable was
        resolved than intended, e.g. geopotential against geopotential height.
    """

    name: str
    kind: str
    accumulation: str = "none"
    units: Optional[str] = None


# ---------------------------------------------------------------------------
# ERA5-style makani channel -> candidate ICON variables
#
# Candidates are listed in preference order: NWP style names first, since that
# is what the operational and most limited area setups write, then AES/CMIP
# style, then plain ERA5 short names for files that were converted elsewhere.
# ---------------------------------------------------------------------------

atmospheric_variables: Dict[str, Tuple[IconVariable, ...]] = {
    # NOTE: makani's "z" is geopotential (m2/s2). "zg" is geopotential *height*
    # (m) and needs multiplying by GRAVITY; the units field flags which is which.
    "z": (
        IconVariable("geopot", "pl", units="m2 s-2"),
        IconVariable("zg", "pl", units="m"),
        IconVariable("z", "pl", units="m2 s-2"),
    ),
    "t": (
        IconVariable("temp", "pl", units="K"),
        IconVariable("ta", "pl", units="K"),
        IconVariable("t", "pl", units="K"),
    ),
    "u": (
        IconVariable("u", "pl", units="m s-1"),
        IconVariable("ua", "pl", units="m s-1"),
    ),
    "v": (
        IconVariable("v", "pl", units="m s-1"),
        IconVariable("va", "pl", units="m s-1"),
    ),
    "w": (
        IconVariable("omega", "pl", units="Pa s-1"),
        IconVariable("wap", "pl", units="Pa s-1"),
    ),
    "q": (
        IconVariable("qv", "pl", units="kg kg-1"),
        IconVariable("hus", "pl", units="kg kg-1"),
    ),
    "r": (
        IconVariable("rh", "pl", units="%"),
        IconVariable("hur", "pl", units="%"),
    ),
}

surface_variables: Dict[str, Tuple[IconVariable, ...]] = {
    "u10m": (IconVariable("u_10m", "sfc", units="m s-1"), IconVariable("uas", "sfc", units="m s-1")),
    "v10m": (IconVariable("v_10m", "sfc", units="m s-1"), IconVariable("vas", "sfc", units="m s-1")),
    "t2m": (IconVariable("t_2m", "sfc", units="K"), IconVariable("tas", "sfc", units="K")),
    "d2": (IconVariable("td_2m", "sfc", units="K"), IconVariable("tdps", "sfc", units="K")),
    "sp": (IconVariable("pres_sfc", "sfc", units="Pa"), IconVariable("ps", "sfc", units="Pa")),
    "msl": (IconVariable("pres_msl", "sfc", units="Pa"), IconVariable("psl", "sfc", units="Pa")),
    "tcwv": (IconVariable("tqv", "sfc", units="kg m-2"), IconVariable("prw", "sfc", units="kg m-2")),
    "sst": (IconVariable("t_seasfc", "sfc", units="K"), IconVariable("tos", "sfc", units="K")),
}

accumulated_variables: Dict[str, Tuple[IconVariable, ...]] = {
    # tot_prec is a running total since the start of the run and has to be
    # differenced; pr is an instantaneous flux and has to be integrated over the
    # window instead. Which one a file carries changes the arithmetic, hence the
    # accumulation field rather than a single code path.
    "tp": (
        IconVariable("tot_prec", "accum", accumulation="since_start", units="kg m-2"),
        IconVariable("pr", "accum", accumulation="rate", units="kg m-2 s-1"),
    ),
}


class ChannelGroup(NamedTuple):
    """A set of makani channels served by one ICON variable.

    Attributes
    ----------
    kind : str
        ``"pl"``, ``"sfc"`` or ``"accum"``, taken from the resolved variable.
    name : str
        The makani side name: the channel prefix for pressure level groups
        (``"z"``), the channel name otherwise (``"t2m"``).
    variable : IconVariable
        The ICON variable the group reads from.
    channel_indices : list of int
        Index of each member channel along the channel axis of the makani
        ``fields`` array.
    levels : list of int or None
        Pressure level in hPa for each member channel, ``None`` for single
        level and accumulated groups.
    """

    kind: str
    name: str
    variable: IconVariable
    channel_indices: List[int]
    levels: Optional[List[int]]


def _as_text(value) -> str:
    """Normalize an HDF5 attribute to ``str``.

    ``h5py`` hands back ``bytes`` for the fixed length strings netCDF writes,
    and 0-d/1-element arrays for scalar attributes, so attributes have to be
    unwrapped before they can be compared against anything.
    """
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected a scalar attribute, got an array of size {value.size}.")
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def resolve_variable(candidates: Sequence[IconVariable], available: Optional[Sequence[str]] = None):
    """Pick the ICON variable a file actually provides for a makani channel.

    Parameters
    ----------
    candidates : sequence of IconVariable
        Candidates in preference order, as listed in the tables above.
    available : sequence of str, optional
        Variable names present in the file. When omitted the first candidate is
        returned, which is what to do when the file has not been opened yet.

    Returns
    -------
    IconVariable or None
        The first candidate the file provides, or None if it provides none.
    """
    if not candidates:
        return None

    if available is None:
        return candidates[0]

    names = set(available)
    for candidate in candidates:
        if candidate.name in names:
            return candidate

    return None


def build_icon_channel_groups(
    channel_names: List[str],
    available: Optional[Sequence[str]] = None,
    skip_missing_channels: bool = False,
) -> List[ChannelGroup]:
    """Group makani channel names by the ICON variable that provides them.

    All levels of a variable come from the same ICON variable, which is stored
    as ``(time, plev, ncells)``, so they are collected into one group and read
    together.

    Parameters
    ----------
    channel_names : list of str
        Channel names in channel index order, i.e. ``coords["channel"]`` from
        the dataset metadata.
    available : sequence of str, optional
        Variable names present in the file, used to resolve between the
        alternative ICON namings. When omitted the preferred name is assumed.
    skip_missing_channels : bool, optional
        If True, channels that no candidate variable covers are dropped instead
        of raising.

    Returns
    -------
    list of ChannelGroup
        One entry per source variable, pressure level groups first.

    Raises
    ------
    ValueError
        If a channel cannot be resolved and ``skip_missing_channels`` is False.
    """
    pl_groups: Dict[str, ChannelGroup] = {}
    other_groups: List[ChannelGroup] = []

    for cidx, channel_name in enumerate(channel_names):
        prefix, level = split_channel_name(channel_name)

        if level is not None:
            variable = resolve_variable(atmospheric_variables.get(prefix, ()), available)
            if variable is None:
                if skip_missing_channels:
                    continue
                raise ValueError(
                    f"No ICON variable found for atmospheric channel '{channel_name}'. "
                    f"Known prefixes: {list(atmospheric_variables)}."
                )
            group = pl_groups.get(prefix)
            if group is None:
                group = ChannelGroup("pl", prefix, variable, [], [])
                pl_groups[prefix] = group
            group.channel_indices.append(cidx)
            group.levels.append(level)
            continue

        variable = resolve_variable(surface_variables.get(channel_name, ()), available)
        if variable is not None:
            other_groups.append(ChannelGroup("sfc", channel_name, variable, [cidx], None))
            continue

        variable = resolve_variable(accumulated_variables.get(channel_name, ()), available)
        if variable is not None:
            other_groups.append(ChannelGroup("accum", channel_name, variable, [cidx], None))
            continue

        if not skip_missing_channels:
            raise ValueError(
                f"No ICON variable found for surface channel '{channel_name}'. "
                f"Known names: {list(surface_variables) + list(accumulated_variables)}."
            )

    return list(pl_groups.values()) + other_groups


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode_time(values, units) -> List[dt.datetime]:
    """Decode an ICON time coordinate into timezone aware UTC datetimes.

    Two encodings are accepted: ICON's own ``"day as %Y%m%d.%f"``, where the
    integer part is the calendar date and the fraction is the elapsed part of
    the day, and the CF ``"<unit> since <reference>"`` form that ICON writes
    when configured for CF output.

    Times are rounded to the nearest second, because the fractional day is
    stored as a float and an exact 08:00 comes back as 07:59:59.99 otherwise.

    Parameters
    ----------
    values : array_like
        Raw values of the time variable.
    units : str or bytes
        The ``units`` attribute of the time variable.

    Returns
    -------
    list of datetime.datetime
        One timezone aware datetime per input value.

    Raises
    ------
    ValueError
        If the units string is neither of the two supported encodings, or if a
        value is not a valid date under it.
    """
    text = _as_text(units).strip()
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))

    if text == ICON_TIME_UNITS:
        return [_decode_icon_time_value(value) for value in values]

    match = _CF_UNITS_PATTERN.match(text)
    if match is not None:
        unit, reference = match.group(1).lower(), match.group(2)
        seconds_per_unit = _CF_UNIT_SECONDS[unit]
        origin = _parse_cf_reference(reference)
        return [origin + dt.timedelta(seconds=round(value * seconds_per_unit)) for value in values]

    raise ValueError(
        f"Unsupported time encoding '{text}'. Expected ICON's '{ICON_TIME_UNITS}' "
        "or a CF style '<unit> since <reference>'."
    )


def _decode_icon_time_value(value: float) -> dt.datetime:
    """Decode a single ``YYYYMMDD.fraction`` value."""
    stamp = int(np.floor(value))
    fraction = float(value) - stamp

    try:
        day = dt.datetime.strptime(str(stamp), "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    except ValueError as err:
        raise ValueError(f"Time value {value} does not start with a valid YYYYMMDD date.") from err

    return day + dt.timedelta(seconds=round(fraction * 86400))


def _parse_cf_reference(reference: str) -> dt.datetime:
    """Parse the reference date of a CF ``<unit> since <reference>`` string."""
    text = reference.strip()

    # trailing timezone designators, which strptime does not take in this position
    for suffix in ("Z", "UTC", "+00:00", "+0000"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()

    for fmt in _CF_REFERENCE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse CF reference date '{reference}'.")


def decode_values(
    raw: np.ndarray,
    fill_value=None,
    scale_factor=None,
    add_offset=None,
    dtype=np.float32,
) -> np.ndarray:
    """Apply netCDF packing and missing value conventions to a raw HDF5 read.

    The netCDF library normally does this on the way out, but reading a netCDF4
    file through ``h5py`` bypasses it: a packed variable comes back as the
    stored integers, and fill values come back as whatever sentinel was written.

    Fill values are matched against the *raw* data, before unpacking, as CF
    requires, and become NaN. Unpacking is ``raw * scale_factor + add_offset``,
    with either factor optional.

    Parameters
    ----------
    raw : numpy.ndarray
        Values as read from the file.
    fill_value : scalar, optional
        Value denoting missing data, from ``_FillValue`` or ``missing_value``.
    scale_factor, add_offset : scalar, optional
        Packing parameters from the variable attributes.
    dtype : numpy dtype, optional
        Floating point type of the result, float32 by default.

    Returns
    -------
    numpy.ndarray
        Decoded values, with missing data as NaN.
    """
    if not np.issubdtype(np.dtype(dtype), np.floating):
        raise ValueError(f"Decoded values need a floating point dtype to hold NaN, got {dtype}.")

    raw = np.asarray(raw)
    missing = None
    if fill_value is not None:
        missing = raw == np.asarray(fill_value).reshape(()).astype(raw.dtype, copy=False)

    values = raw.astype(dtype, copy=True)

    if scale_factor is not None:
        values *= dtype(scale_factor)
    if add_offset is not None:
        values += dtype(add_offset)

    if missing is not None and missing.any():
        values[missing] = np.nan

    return values


# ---------------------------------------------------------------------------
# Grid and level metadata
# ---------------------------------------------------------------------------


def grid_coordinates_in_degrees(clon, clat) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ICON cell center coordinates from radians to degrees.

    ICON stores ``clon``/``clat`` in radians with longitude in [-pi, pi]. The
    result uses degrees with longitude in [0, 360), matching makani's own
    convention and the ERA5 datasets it is trained on.

    Parameters
    ----------
    clon, clat : array_like
        Cell center longitude and latitude in radians.

    Returns
    -------
    lon, lat : numpy.ndarray
        Coordinates in degrees, longitude wrapped into [0, 360).

    Raises
    ------
    ValueError
        If the inputs do not look like radians, which almost always means the
        file stores degrees already and the caller would otherwise silently
        collapse the whole globe into a few degrees around the prime meridian.
    """
    clon = np.asarray(clon, dtype=np.float64)
    clat = np.asarray(clat, dtype=np.float64)

    if clon.shape != clat.shape:
        raise ValueError(f"clon and clat must have the same shape, got {clon.shape} and {clat.shape}.")

    if np.nanmax(np.abs(clat)) > np.pi / 2 + 1e-6:
        raise ValueError(
            "Cell latitudes exceed pi/2, so they are not radians. ICON writes radians; "
            "if this file stores degrees, pass them through unchanged instead."
        )

    lat = np.rad2deg(clat)
    lon = np.mod(np.rad2deg(clon), 360.0)

    return lon, lat


def pressure_levels_in_hpa(levels) -> np.ndarray:
    """Normalize a pressure level coordinate to hPa.

    ICON writes pressure levels in Pa while makani channel names carry hPa
    (``z500``), so one of the two has to be converted, and getting it backwards
    silently selects the wrong level. Pa is detected by magnitude: no
    meteorologically useful level exceeds 2000 hPa, and none is below 0.01 hPa.
    """
    levels = np.asarray(levels, dtype=np.float64)

    if levels.size == 0:
        raise ValueError("Empty pressure level coordinate.")

    if np.nanmax(levels) > 2000.0:
        return levels / 100.0

    return levels


def pressure_level_index(levels, wanted_hpa: int, tolerance: float = 0.5) -> int:
    """Return the index of ``wanted_hpa`` in a pressure level coordinate.

    The coordinate may be in Pa or hPa, see :func:`pressure_levels_in_hpa`.

    Raises
    ------
    ValueError
        If no level matches within ``tolerance`` hPa.
    """
    levels_hpa = pressure_levels_in_hpa(levels)
    distance = np.abs(levels_hpa - wanted_hpa)
    closest = int(np.argmin(distance))

    if distance[closest] > tolerance:
        raise ValueError(
            f"No pressure level within {tolerance} hPa of {wanted_hpa} hPa. "
            f"Available levels (hPa): {np.array2string(levels_hpa, threshold=20)}."
        )

    return closest


def check_grid_uuid(data_uuid, grid_uuid) -> None:
    """Check that a data file and a grid file describe the same horizontal grid.

    ICON stamps output files with the ``uuidOfHGrid`` of the grid they were run
    on. Regridding against the wrong grid file produces a plausible looking but
    scrambled field, so the UUIDs are compared rather than assumed to match.
    Comparison ignores case and surrounding whitespace; either side being absent
    is accepted, since not every setup writes the attribute.

    Raises
    ------
    ValueError
        If both UUIDs are present and differ.
    """
    if data_uuid is None or grid_uuid is None:
        return

    data_text = _as_text(data_uuid).strip().lower()
    grid_text = _as_text(grid_uuid).strip().lower()

    if not data_text or not grid_text:
        return

    if data_text != grid_text:
        raise ValueError(
            f"The data file was written on grid {data_text} but the grid file describes {grid_text}. "
            "Regridding with a mismatched grid file silently scrambles the field."
        )
