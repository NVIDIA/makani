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

"""Shared channel grouping for the data source readers.

Every reader (NCAR, WeatherBench2, ICON) has to answer the same question: given
the makani channel names a run asks for, which source variables provide them,
and which channel index does each fill. This module holds that logic once, so
the readers only contribute their name tables.

Source variable descriptors
---------------------------
Each reader defines its own descriptor type, because *locating* a variable
differs completely between sources: NCAR needs an S3 stream, parameter code and
grid suffix, ICON and WB2 need only a variable name. Descriptors are NamedTuples
and are expected to expose:

``kind``
    ``"pl"``, ``"sfc"`` or ``"accum"``. Informational; the group kind comes from
    which table the entry was found in.
``units`` (optional)
    Units the source is expected to carry, for the converter to check against
    the file. Catches quantity confusions such as geopotential in m2/s2 versus
    geopotential height in m.
``accumulation`` (optional)
    ``"none"`` for instantaneous fields, ``"since_start"`` for running totals
    that must be differenced, ``"rate"`` for fluxes that must be integrated over
    the window, ``"window"`` for totals already accumulated over a fixed window
    that the source, not the reader, decided.
``name``
    Required only by readers that resolve against the variables a file actually
    contains, see ``available`` in :func:`build_channel_groups`.

Table format: alternatives versus components
--------------------------------------------
A makani channel can map to a source in two different ways, and they look alike
but mean the opposite:

* **alternatives** -- several possible sources, of which one is chosen. ICON is
  run with different physics packages, so ``tp`` may appear as ``tot_prec`` *or*
  as ``pr``.
* **components** -- several variables that are summed to form the channel. NCAR
  does not ship total precipitation, so ``tp`` is ``lsp`` *plus* ``cp``.

A table therefore maps a channel to a sequence of *candidates*, where each
candidate is either a single descriptor or a sequence of descriptors that are
summed::

    atmospheric = {"z": (geopot_variable,)}                 # one candidate
    accumulated = {"tp": ((lsp_variable, cp_variable),)}    # one candidate, summed
    accumulated = {"tp": (tot_prec_variable, pr_variable)}  # two alternatives

A bare descriptor is accepted wherever a candidate is expected. Descriptors are
told apart from plain sequences by ``_fields``, since a NamedTuple is itself a
tuple and ``isinstance(x, tuple)`` cannot distinguish the two.

Getting this wrong is silent rather than loud -- summing two alternatives, or
picking one of two components, produces a field that is wrong by a factor or by
a missing term but still looks like weather. Hence the explicit nesting, and
hence a candidate matches only if the file provides *every* one of its
components.

See also
--------
makani.utils.dataloaders.ncar_helpers : NSF NCAR ERA5 on S3, one canonical name
    per channel, ``tp`` summed from two components.
makani.utils.dataloaders.wb2_helpers : WeatherBench2 zarr, one long name per
    channel, levels addressed by index rather than by name.
makani.utils.dataloaders.icon_helpers : ICON netCDF, several candidate names per
    channel depending on the physics package the run used.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from makani.utils.features import split_channel_name


class ChannelGroup(NamedTuple):
    """A set of makani channels served by the same source variable(s).

    Attributes
    ----------
    kind : str
        ``"pl"``, ``"sfc"`` or ``"accum"``, from the table the entry came from.
    name : str
        The makani side name: the channel prefix for pressure level groups
        (``"z"``), the channel name otherwise (``"t2m"``).
    variables : list
        Source descriptors for this group. More than one entry means the values
        are summed to form the channel, see the module docstring.
    channel_indices : list of int
        Index of each member channel along the channel axis of the makani
        ``fields`` array.
    levels : list of int or None
        Pressure level in hPa for each member channel, ``None`` for single level
        and accumulated groups.
    """

    kind: str
    name: str
    variables: List[Any]
    channel_indices: List[int]
    levels: Optional[List[int]]


def _components(candidate) -> tuple:
    """Normalize a candidate into the tuple of descriptors it is made of."""
    if hasattr(candidate, "_fields"):
        # a descriptor: a NamedTuple, which is also a tuple, hence the _fields check
        return (candidate,)
    return tuple(candidate)


def _candidates(entry) -> tuple:
    """Normalize a table entry into a tuple of candidates."""
    if entry is None:
        return ()
    if hasattr(entry, "_fields"):
        return (entry,)
    return tuple(entry)


def resolve_variable(entry, available: Optional[Sequence[str]] = None):
    """Pick the candidate a file actually provides.

    Parameters
    ----------
    entry : descriptor, or sequence of candidates
        A table entry, see the module docstring.
    available : sequence of str, optional
        Variable names present in the file. When omitted the first candidate is
        returned, which is what to do before a file has been opened. A candidate
        made of several summed components matches only if the file provides all
        of them.

    Returns
    -------
    tuple or None
        The components of the first matching candidate, or None if the file
        provides none of them.
    """
    candidates = _candidates(entry)
    if not candidates:
        return None

    if available is None:
        return _components(candidates[0])

    names = set(available)
    for candidate in candidates:
        components = _components(candidate)
        if all(component.name in names for component in components):
            return components

    return None


def build_channel_groups(
    channel_names: List[str],
    atmospheric: Dict[str, Any],
    surface: Optional[Dict[str, Any]] = None,
    accumulated: Optional[Dict[str, Any]] = None,
    available: Optional[Sequence[str]] = None,
    skip_missing_channels: bool = False,
    source: str = "source",
) -> List[ChannelGroup]:
    """Group makani channel names by the source variable that provides them.

    All levels of a variable are collected into a single group, because every
    source stores them together: one chunk per (time, level) slab in ICON, one
    file holding all levels in NCAR, one array with a level axis in WB2. Reading
    them as a group is what keeps the number of reads down.

    Parameters
    ----------
    channel_names : list of str
        Channel names in channel index order, i.e. ``coords["channel"]`` from
        the dataset metadata.
    atmospheric, surface, accumulated : dict
        Tables mapping a channel prefix (atmospheric) or channel name (surface,
        accumulated) to a table entry, see the module docstring.
    available : sequence of str, optional
        Variable names present in the file, used to choose between alternatives.
    skip_missing_channels : bool, optional
        If True, channels no candidate covers are dropped instead of raising.
    source : str, optional
        Name of the data source, used in error messages only.

    Returns
    -------
    list of ChannelGroup
        One entry per source variable, pressure level groups first.

    Raises
    ------
    ValueError
        If a channel cannot be resolved and ``skip_missing_channels`` is False.
    """
    surface = surface or {}
    accumulated = accumulated or {}

    pl_groups: Dict[str, ChannelGroup] = {}
    other_groups: List[ChannelGroup] = []

    for cidx, channel_name in enumerate(channel_names):
        prefix, level = split_channel_name(channel_name)

        if level is not None:
            variables = resolve_variable(atmospheric.get(prefix), available)
            if variables is None:
                if skip_missing_channels:
                    continue
                raise ValueError(
                    f"No {source} variable found for atmospheric channel '{channel_name}' "
                    f"(prefix '{prefix}'). Known prefixes: {list(atmospheric)}."
                )
            group = pl_groups.get(prefix)
            if group is None:
                group = ChannelGroup("pl", prefix, list(variables), [], [])
                pl_groups[prefix] = group
            group.channel_indices.append(cidx)
            group.levels.append(level)
            continue

        variables = resolve_variable(surface.get(channel_name), available)
        if variables is not None:
            other_groups.append(ChannelGroup("sfc", channel_name, list(variables), [cidx], None))
            continue

        variables = resolve_variable(accumulated.get(channel_name), available)
        if variables is not None:
            other_groups.append(ChannelGroup("accum", channel_name, list(variables), [cidx], None))
            continue

        if not skip_missing_channels:
            raise ValueError(
                f"No {source} variable found for surface channel '{channel_name}'. "
                f"Known names: {list(surface) + list(accumulated)}."
            )

    return list(pl_groups.values()) + other_groups
