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

"""ICON output, read on its native icosahedral mesh.

Netcdf4 files of ``(time, level, ncells)``, one variable per file, on a grid
whose coordinates live in a separate grid file. The mesh is emitted as it is
stored: this backend does not resample, and the cells it hands out are labelled
with their own latitude and longitude so that a layer downstream can put them
wherever the model wants them.

Three things make this layout different from the makani ones, and between them
they account for most of what is here.

**A sample spans several files.** Each variable is its own file, so the 72
channels of a run come from a dozen of them. What the sample index runs through
is therefore a *unit* rather than a file, and a unit is a stretch of the common
time axis over which no variable changes file. The rest is bookkeeping: for
every sample and every variable, which file and which index within it.

**The variables do not share a time axis.** In the EXCLAIM DYAMOND output
``temp`` is three hourly in daily files, ``u`` on altitude levels is hourly in
daily files, and ``t_2m`` is hourly in monthly ones. The dataset's samples are
the times that *all* the required variables have, which is their intersection.

**Names alone do not identify a variable.** The same run writes two variables
called ``u``, one on pressure levels and one on altitude levels. A makani
channel like ``u500`` means the pressure level one, so candidates are resolved
by the level coordinate they carry rather than by name.

Reading
-------
Cells are selected geometrically -- see :mod:`.mesh` -- and read as slices. The
files carry no compression filter, so a contiguous range costs about what its
length suggests, while a strided or element-wise selection falls back to a
gather that measures an order of magnitude slower. Everything here is therefore
built to issue few, long, contiguous reads.

Reads are issued one variable after another. Nothing is gained by threading
them: h5py serializes calls into HDF5 behind a global lock, so concurrency has
to come from running several workers, which is what the DALI pipeline already
does.
"""

import datetime as dt
import glob
import logging
import os
import re
from collections import Counter, OrderedDict
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import h5py
import numpy as np

from ..channel_helpers import split_channel_name
from ..icon_helpers import (
    accumulated_variables,
    atmospheric_variables,
    build_icon_channel_groups,
    surface_variables,
    check_grid_uuid,
    decode_time,
    decode_values,
    grid_coordinates_in_degrees,
    pressure_level_index,
    pressure_levels_in_hpa,
)
from .base import BackendMetadata, DatasetBackend, GridSpec
from .mesh import MeshChunkMixin

#: level coordinates ICON writes, and what they mean for a makani channel
PRESSURE_LEVEL_NAMES = ("plev", "pressure", "lev")
SURFACE_LEVEL_NAMES = ("height", "height_2", "heightAboveGround", "alt", "altitude", "depth")


def _candidate_names(candidates) -> List[str]:
    """Every variable name in a table entry, flattening summed components."""
    names = []
    for candidate in candidates or ():
        if hasattr(candidate, "name"):
            names.append(candidate.name)
        else:
            # a tuple of components that are summed to form the channel
            names.extend(component.name for component in candidate)
    return names


def candidate_variable_names(channel_names: Sequence[str]) -> List[str]:
    """Every ICON variable that could serve one of these channels.

    The alternatives, not the resolved choice: a run asking for ``u500`` may be
    served by ``u`` or ``ua``, and which one is present is exactly what has not
    been established yet.
    """
    names = set()
    for channel in channel_names:
        prefix, level = split_channel_name(channel)
        if level is not None:
            names.update(_candidate_names(atmospheric_variables.get(prefix)))
        else:
            names.update(_candidate_names(surface_variables.get(channel)))
            names.update(_candidate_names(accumulated_variables.get(channel)))
    return sorted(names)


def known_variable_names() -> List[str]:
    """Every ICON variable the channel tables know about.

    Classification uses this rather than the wanted set, so that a file holding
    a variable no channel asked for is recognised *and skipped*, while a file
    whose name matches nothing stays a candidate for opening.
    """
    names = set()
    for table in (atmospheric_variables, surface_variables, accumulated_variables):
        for candidates in table.values():
            names.update(_candidate_names(candidates))
    return sorted(names)


def variable_of_file(path: str, candidates: Sequence[str]) -> Optional[str]:
    """Which variable a file holds, from its name, or None if it says nothing.

    ICON names a file after the variable it carries and then decorates it, so
    ``temp_EXCLAIM_..._20210601T000000Z.nc`` is ``temp``. The catch is that the
    names nest: ``u_10m_dyamond_...`` starts with ``u_`` as well as ``u_10m_``,
    and reading ten metre wind as upper air ``u`` would be silent. The longest
    match is therefore the answer, and only names we are actually looking for
    are candidates at all.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    matches = [name for name in candidates if stem == name or stem.startswith(name + "_")]
    return max(matches, key=len) if matches else None


#: how ICON stamps a file with the period it covers: a full timestamp for the
#: daily files, a bare year and month for the aggregated ones
_DATE_PATTERNS = (
    (re.compile(r"(\d{8})T(\d{6})Z"), "%Y%m%d%H%M%S"),
    (re.compile(r"_(\d{6})(?=[._]|$)"), "%Y%m"),
)


def date_of_file(path: str) -> Optional[dt.datetime]:
    """When the period a file covers begins, from its name.

    ``..._20210601T000000Z.nc`` starts at midnight on the first of June;
    ``..._202106.nc`` is the monthly aggregate and starts at the same place.
    Returns None when the name carries no date, which is a reason to open the
    file rather than to guess.
    """
    stem = os.path.basename(path)
    for pattern, fmt in _DATE_PATTERNS:
        match = pattern.search(stem)
        if match:
            return dt.datetime.strptime("".join(match.groups()), fmt).replace(tzinfo=dt.timezone.utc)
    return None


def survey_files(paths: Sequence[str], candidates: Sequence[str]) -> Dict[Optional[str], List[str]]:
    """Group files by the variable their name claims, before opening any of them."""
    survey: Dict[Optional[str], List[str]] = {}
    for path in paths:
        survey.setdefault(variable_of_file(path, candidates), []).append(path)
    return survey


class VariableSource(NamedTuple):
    """Where one ICON variable's samples live, and on what levels.

    Attributes
    ----------
    name : str
        Variable name inside the files.
    paths : list of str
        Files carrying it, in time order.
    times : numpy.ndarray
        Every sample time it has, concatenated across those files.
    path_of_sample, index_of_sample : numpy.ndarray
        For each entry of ``times``, which file it is in and where in that file.
    levels_hpa : numpy.ndarray, optional
        Pressure levels in hPa, for a variable that has them.
    level_name : str, optional
        Name of the level coordinate, which is what tells a pressure level
        variable from an altitude one of the same name.
    """

    name: str
    paths: List[str]
    times: np.ndarray
    path_of_sample: np.ndarray
    index_of_sample: np.ndarray
    levels_hpa: Optional[np.ndarray]
    level_name: Optional[str]

    @property
    def is_pressure(self) -> bool:
        return self.level_name in PRESSURE_LEVEL_NAMES


class Unit(NamedTuple):
    """A stretch of the sample axis over which no variable changes file."""

    start: int
    n_samples: int
    paths: Dict[str, str]


class ChannelSource(NamedTuple):
    """What one makani channel is read from."""

    variable: str
    level_index: int


class IconBackend(MeshChunkMixin, DatasetBackend):
    """Reads ICON output on its native mesh.

    Parameters
    ----------
    grid_file : str
        The ICON grid, which carries ``clon``/``clat``. Output files reference a
        grid rather than containing one, so this cannot be inferred.
    halo_degrees : float, optional
        Margin around this rank's block, in degrees. Defaults to about three
        cells, from the mesh's own spacing -- enough for an interpolation
        stencil to reach past the block edge without a neighbour exchange.
    max_open_files : int, optional
        Handles to keep. A long run touches one file per variable per day, which
        would otherwise accumulate.
    enable_odirect : bool
        Read through the O_DIRECT driver, bypassing the page cache. Worth more
        here than for the makani layouts: a training run reads tens of gigabytes
        per sample and never reads any of it twice, so everything it pulls
        through the cache is eviction pressure on something else. The cost is
        that a read shorter than the alignment is rounded up, which is why the
        selection is coalesced into long runs first.
    odirect_alignment : int
        Alignment and block size for O_DIRECT, when the default does not suit
        the filesystem.
    """

    #: named after the variable and the date, never after the year
    default_file_pattern = "*"

    _handle_attributes = ("files",)
    _transient_attributes = MeshChunkMixin._transient_attributes

    def __init__(
        self,
        *args,
        grid_file: Optional[str] = None,
        halo_degrees: Optional[float] = None,
        max_open_files: int = 64,
        enable_odirect: bool = False,
        odirect_alignment: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if grid_file is None:
            raise ValueError(
                "ICON output references its grid rather than carrying it, so grid_file is required: it is "
                "the file holding clon/clat, named by the grid_file_uri attribute of the data."
            )
        if self.channel_names is None:
            raise ValueError("ICON addresses variables by name, so channel_names is required.")

        self.grid_file = grid_file
        self.halo_degrees = halo_degrees
        self.max_open_files = max_open_files

        self.file_driver = "direct" if enable_odirect else None
        self.file_driver_kwargs = {}
        if enable_odirect and odirect_alignment > 0:
            self.file_driver_kwargs = dict(alignment=odirect_alignment, block_size=odirect_alignment)

        self.files: List = []
        self.source_paths: List[str] = []
        self.sources: Dict[str, VariableSource] = {}
        self.units: List[Unit] = []
        self.channel_plan: Dict[int, ChannelSource] = {}
        self._open_order: "OrderedDict[str, int]" = OrderedDict()

    # ---- discovery ---------------------------------------------------------

    def _find_files(self) -> List[str]:
        paths = []
        for location in self.location:
            paths += glob.glob(os.path.join(location, f"{self.file_pattern}.nc"))
        return sorted(paths)

    @staticmethod
    def _data_variables(handle) -> List[str]:
        """Variables that hold fields, as opposed to coordinates."""
        return [
            name
            for name, obj in handle.items()
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and "coordinates" not in name
        ]

    def _level_coordinate(self, handle, dset) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Name and values of the level axis, or None where there is none."""
        for name in PRESSURE_LEVEL_NAMES + SURFACE_LEVEL_NAMES:
            if name in handle and handle[name].shape == (dset.shape[1],):
                values = np.asarray(handle[name])
                if name in PRESSURE_LEVEL_NAMES:
                    return name, pressure_levels_in_hpa(handle[name])
                return name, values
        return None, None

    def _files_worth_opening(self, files: List[str], enable_logging: bool) -> List[str]:
        """Drop the files whose name says they hold a variable nothing asked for.

        A run reads a dozen variables out of a directory that may hold many
        more, and every file it opens costs a round trip to the metadata server
        before it says anything. Names are used only to *skip*: which variable a
        file really holds, and whether it is on pressure levels, is still
        decided by looking inside the ones that survive.

        A file whose name matches nothing known is kept rather than dropped,
        and if no wanted variable is named at all the survey is ignored
        entirely -- so a dataset that does not follow the convention costs what
        it always did rather than quietly losing data.
        """
        wanted = set(candidate_variable_names(self.channel_names))
        self.survey = survey_files(files, known_variable_names())

        named = {name for name in self.survey if name is not None}
        if not named & wanted:
            # the names say nothing we recognise, so they are no guide at all
            return files

        keep = [path for name in named & wanted for path in self.survey[name]]
        # a name that matches nothing known could still hold a wanted variable
        keep += self.survey.get(None, [])

        skipped = len(files) - len(keep)
        if skipped and enable_logging:
            logging.info(f"Skipping {skipped} of {len(files)} ICON files: no channel asks for what they hold")

        return sorted(keep)

    def _inspect(self, path: str, grid_uuid: Optional[str]) -> Dict[str, dict]:
        """What one file contributes: its variables, their times and levels."""
        found = {}
        with h5py.File(path, "r") as handle:
            check_grid_uuid(handle.attrs.get("uuidOfHGrid"), grid_uuid)

            if self.timestamp_name not in handle and "time" not in handle:
                raise ValueError(f"{path} carries no time coordinate, so its samples cannot be placed.")
            time_key = self.timestamp_name if self.timestamp_name in handle else "time"
            times = np.asarray(decode_time(handle[time_key][...], handle[time_key].attrs.get("units")))

            for name in self._data_variables(handle):
                dset = handle[name]
                level_name, levels = self._level_coordinate(handle, dset)
                found[name] = {
                    "times": times,
                    "n_cells": int(dset.shape[-1]),
                    "level_name": level_name,
                    "levels": levels,
                }
        return found

    def _file_groups(self, files: List[str]) -> List[List[str]]:
        """Files grouped by the variable their name claims.

        A group shares a level coordinate, a cadence and a file length, which is
        what lets most of it be described from one member. Files whose name says
        nothing are each their own group, so nothing is assumed about them.
        """
        keep = set(files)
        groups = []
        for name, paths in self.survey.items():
            present = sorted(path for path in paths if path in keep)
            if not present:
                continue
            if name is None:
                groups.extend([path] for path in present)
            else:
                groups.append(present)
        return groups

    def _inspect_group(self, paths: List[str], grid_uuid: Optional[str]) -> Dict[str, Dict[str, dict]]:
        """Describe every file of one variable, opening as few as possible.

        The level coordinate, the cadence and the length are properties of the
        variable rather than of the file, so one member describes them all. Only
        the sample times differ, and a file stamped with the period it covers
        says those too -- given that it is the same shape as its neighbours.

        Three things keep that from being a guess:

        * the first and last file are always read, because truncation happens at
          the ends of a run;
        * a file whose size differs from the rest of the group is read, since a
          short file is exactly the one whose times cannot be derived;
        * what is derived for the last file is checked against what was read
          from it, and a mismatch abandons the shortcut for the whole group.

        Getting this wrong would be silent -- plausible times attached to the
        wrong samples -- so the fallback is to open everything, which is what
        this did before.
        """
        opened = {path: self._inspect(path, grid_uuid) for path in {paths[0], paths[-1]}}
        if len(paths) <= 2:
            return {path: opened[path] for path in paths}

        dates = {path: date_of_file(path) for path in paths}
        sizes = {path: os.path.getsize(path) for path in paths}
        usual_size = Counter(sizes.values()).most_common(1)[0][0]

        first, last = paths[0], paths[-1]
        reference = opened[first]

        def derived(path) -> Optional[Dict[str, dict]]:
            """The description a file's name implies, or None if it does not."""
            if dates[path] is None or dates[first] is None or sizes[path] != usual_size:
                return None
            shift = dates[path] - dates[first]
            return {name: dict(info, times=info["times"] + shift) for name, info in reference.items()}

        # the check: the last file was read, so what its name implies can be
        # compared against what it actually says
        predicted = derived(last)
        trustworthy = predicted is not None and all(
            np.array_equal(predicted[name]["times"], opened[last][name]["times"]) for name in opened[last]
        )
        if not trustworthy:
            logging.debug("ICON file names do not predict their contents for this group; reading every file")
            return {path: self._inspect(path, grid_uuid) for path in paths}

        described = {}
        for path in paths:
            if path in opened:
                described[path] = opened[path]
                continue
            implied = derived(path)
            described[path] = implied if implied is not None else self._inspect(path, grid_uuid)
        return described

    def _build_sources(self, files: List[str], grid_uuid: Optional[str]) -> Dict[str, VariableSource]:
        """Collect every variable across every file, in time order.

        A variable name can appear with two different level coordinates -- the
        pressure level ``u`` and the altitude level one -- so sources are keyed
        by name *and* level coordinate, and resolved to a channel later.
        """
        collected: Dict[Tuple[str, Optional[str]], dict] = {}

        described = {}
        for group in self._file_groups(files):
            described.update(self._inspect_group(group, grid_uuid))

        for path in files:
            for name, info in described[path].items():
                key = (name, info["level_name"])
                entry = collected.setdefault(
                    key, {"paths": [], "times": [], "levels": info["levels"], "n_cells": info["n_cells"]}
                )
                entry["paths"].append(path)
                entry["times"].append(info["times"])

        sources = {}
        for (name, level_name), entry in collected.items():
            order = np.argsort([times[0] for times in entry["times"]])
            paths = [entry["paths"][idx] for idx in order]
            per_file = [entry["times"][idx] for idx in order]

            times = np.concatenate(per_file)
            path_of_sample = np.concatenate([np.full(len(part), idx) for idx, part in enumerate(per_file)])
            index_of_sample = np.concatenate([np.arange(len(part)) for part in per_file])

            sources[(name, level_name)] = VariableSource(
                name=name,
                paths=paths,
                times=times,
                path_of_sample=path_of_sample,
                index_of_sample=index_of_sample,
                levels_hpa=entry["levels"] if level_name in PRESSURE_LEVEL_NAMES else None,
                level_name=level_name,
            )
        return sources

    def _resolve_channels(self, sources: Dict[Tuple[str, Optional[str]], VariableSource]) -> Dict[int, ChannelSource]:
        """Decide which variable and level each makani channel is read from."""
        available = sorted({name for name, _ in sources})
        groups = build_icon_channel_groups(list(self.channel_names), available=available)

        plan: Dict[int, ChannelSource] = {}
        for group in groups:
            if group.kind == "accum":
                raise NotImplementedError(
                    f"Channel group '{group.name}' is accumulated, which needs differencing between "
                    "consecutive outputs. That is a converter's job, not a reader's."
                )
            if len(group.variables) > 1:
                raise NotImplementedError(
                    f"Channel group '{group.name}' is the sum of several ICON variables, which this "
                    "backend does not assemble."
                )

            wanted = group.variables[0].name
            candidates = [source for (name, _), source in sources.items() if name == wanted]
            if not candidates:
                raise ValueError(f"Variable '{wanted}' is named by the channel table but not in the files.")

            if group.kind == "pl":
                # the disambiguation: two variables may share a name, and only
                # the one on pressure levels can serve a pressure level channel
                pressure = [source for source in candidates if source.is_pressure]
                if not pressure:
                    raise ValueError(
                        f"Channel group '{group.name}' needs '{wanted}' on pressure levels, but the files "
                        f"carry it on {[source.level_name for source in candidates]} only."
                    )
                source = pressure[0]
                for channel_index, level in zip(group.channel_indices, group.levels):
                    plan[channel_index] = ChannelSource(wanted, pressure_level_index(source.levels_hpa, level))
            else:
                source = next((c for c in candidates if not c.is_pressure), candidates[0])
                plan[group.channel_indices[0]] = ChannelSource(wanted, 0)

            self.sources[wanted] = source

        missing = set(range(len(self.channel_names))) - set(plan)
        if missing:
            names = [self.channel_names[idx] for idx in sorted(missing)]
            raise ValueError(f"No ICON variable resolved for channels {names}.")
        return plan

    def _common_times(self) -> np.ndarray:
        """The times every required variable has."""
        times = None
        for source in self.sources.values():
            available = set(source.times.tolist())
            times = available if times is None else (times & available)
        if not times:
            raise ValueError(
                "The required variables share no sample times. They are written at different cadences, "
                "so the dataset is their intersection -- and here that is empty."
            )
        return np.array(sorted(times))

    def _build_units(self, times: np.ndarray) -> List[Unit]:
        """Split the sample axis wherever any variable changes file."""
        locate = {}
        for name, source in self.sources.items():
            positions = np.searchsorted(source.times, times)
            locate[name] = (source.path_of_sample[positions], source.index_of_sample[positions])

        boundaries = np.zeros(len(times), dtype=bool)
        boundaries[0] = True
        for path_of_sample, _ in locate.values():
            boundaries[1:] |= path_of_sample[1:] != path_of_sample[:-1]

        starts = np.flatnonzero(boundaries)
        stops = np.append(starts[1:], len(times))

        units = []
        for start, stop in zip(starts, stops):
            paths = {name: self.sources[name].paths[locate[name][0][start]] for name in self.sources}
            units.append(Unit(int(start), int(stop - start), paths))

        self._sample_position = {name: locate[name][1] for name in self.sources}
        return units

    def _read_grid(self) -> GridSpec:
        with h5py.File(self.grid_file, "r") as handle:
            if "clon" not in handle or "clat" not in handle:
                raise ValueError(f"{self.grid_file} has no clon/clat, so it is not an ICON grid file.")
            lon, lat = grid_coordinates_in_degrees(handle["clon"][...], handle["clat"][...])
            self.grid_uuid = handle.attrs.get("uuidOfHGrid")
        return GridSpec("unstructured", (len(lat),), lat, lon)

    def discover(self, enable_logging: bool = False) -> BackendMetadata:
        files = self._find_files()
        if not files:
            locations = ", ".join(self.location)
            raise IOError(f"Error, the specified file path(s) {locations} contain no netCDF files.")

        grid = self._read_grid()
        if self.halo_degrees is None:
            self.halo_degrees = self.default_halo_degrees(grid.lat)

        if enable_logging:
            logging.info(f"Getting file stats from {len(files)} ICON files, grid {os.path.basename(self.grid_file)}")

        files = self._files_worth_opening(files, enable_logging)
        sources = self._build_sources(files, getattr(self, "grid_uuid", None))
        self.channel_plan = self._resolve_channels(sources)

        times = self._common_times()
        self.units = self._build_units(times)

        self.source_paths = sorted({path for source in self.sources.values() for path in source.paths})
        self.path_index = {path: idx for idx, path in enumerate(self.source_paths)}
        self.files = [None] * len(self.source_paths)

        self.chunk = self._resolve_chunk(grid)
        self.metadata = BackendMetadata(
            files=[unit.paths[next(iter(unit.paths))] for unit in self.units],
            labels=[time.year for time in times[[unit.start for unit in self.units]]],
            samples_per_file=[unit.n_samples for unit in self.units],
            timestamps=times,
            grid=grid,
            total_channels=len(self.channel_names),
        )
        return self.metadata

    # ---- handles -----------------------------------------------------------

    def open(self, file_idx: int) -> None:
        """Open every file the unit needs, dropping the least recently used."""
        for path in self.units[file_idx].paths.values():
            self._handle(path)

    def _handle(self, path: str):
        index = self.path_index[path]
        if self.files[index] is None:
            self.files[index] = h5py.File(path, "r", driver=self.file_driver, **self.file_driver_kwargs)
        self._open_order[path] = index
        self._open_order.move_to_end(path)

        while len(self._open_order) > self.max_open_files:
            stale, stale_index = self._open_order.popitem(last=False)
            handle, self.files[stale_index] = self.files[stale_index], None
            if handle is not None:
                handle.close()

        return self.files[index]

    def _restore(self) -> None:
        # handles and the read buffer are both rebuilt lazily in the worker
        self._open_order = OrderedDict()
        self._buffer = None

    # ---- reading -----------------------------------------------------------

    @staticmethod
    def _packing(attributes) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Fill value and packing parameters of one variable."""
        fill = attributes.get("_FillValue", attributes.get("missing_value"))
        return fill, attributes.get("scale_factor"), attributes.get("add_offset")

    def _decode_into(self, values, destination, fill, scale_factor, add_offset) -> None:
        """Write decoded values into ``destination``, without a copy where possible.

        The general path goes through :func:`~..icon_helpers.decode_values`,
        which matches fill values against the *raw* data before unpacking, as CF
        requires. That matters when a variable is packed, because the sentinel
        is an integer and comparing it after conversion to float is not the same
        test. Unpacked data -- which is what ICON writes here -- has no such
        ordering constraint, so the values can be copied straight in and the
        fills replaced in place.
        """
        if scale_factor is None and add_offset is None:
            np.copyto(destination, values, casting="same_kind")
            if fill is not None:
                destination[destination == np.asarray(fill).reshape(()).astype(destination.dtype)] = np.nan
            return

        destination[...] = decode_values(values, fill_value=fill, scale_factor=scale_factor, add_offset=add_offset)

    def read(self, file_idx, time_slice, channels, out=None):
        unit = self.units[file_idx]
        local_indices = list(range(*time_slice.indices(unit.n_samples)))

        if out is None:
            out = np.empty((len(local_indices), len(channels), *self.chunk.shape), dtype=np.float32)

        for position, channel in enumerate(channels):
            plan = self.channel_plan[int(channel)]
            dset = self._handle(unit.paths[plan.variable])[plan.variable]
            fill, scale_factor, add_offset = self._packing(dset.attrs)

            for step, local_idx in enumerate(local_indices):
                time_idx = int(self._sample_position[plan.variable][unit.start + local_idx])

                # bound explicitly: the reader is called once per run, and reading
                # a loop variable through a closure is how that goes subtly wrong
                def read_run(run, destination, dset=dset, time_idx=time_idx, level=plan.level_index):
                    dset.read_direct(destination, np.s_[time_idx, level, run], np.s_[:])

                values = self._gather_runs(read_run, dset.dtype)
                self._decode_into(values, out[step, position, :], fill, scale_factor, add_offset)

        return out
