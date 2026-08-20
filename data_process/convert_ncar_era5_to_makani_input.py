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

from typing import Dict, List, Optional
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
import threading
import os
import sys
import json
import time
import numpy as np
import h5py as h5
import datetime as dt
import argparse as ap
import warnings

# MPI
from mpi4py import MPI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from makani.utils.dataloaders.ncar_helpers import (
    NCAR_ERA5_BUCKET,
    accumulation_key,
    analysis_pl_key,
    analysis_sfc_key,
    build_ncar_channel_groups,
    resolve_accumulation_segments,
    to_ncar_hours,
)
from data_process.data_process_helpers import DistributedProgressBar


class NcarStore(object):
    """Reader for NCAR ERA5 netCDF4 objects on S3, with optional local caching.

    The objects are chunked such that a single chunk holds one timestep of all
    pressure levels, so h5py byte range reads over S3 subset in time for free
    while subsetting in level is not possible. Open file handles are kept in a
    small LRU cache because a rank walks days in order and therefore revisits
    the same monthly surface file many times.

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket to read from.
    cache_dir : str, optional
        If given, objects are downloaded here on first use and reopened from
        local disk afterwards. Downloads are staged through a temporary file and
        renamed into place, so concurrent ranks cannot observe a partial file.
    max_open : int, optional
        Number of file handles to keep open. Wants to comfortably exceed the
        number of monthly surface and accumulation files a day touches, since
        those are revisited on every day of the month; pressure level files are
        released explicitly and do not compete for slots.
    prefetch_workers : int, optional
        Number of background threads fetching upcoming objects. Zero disables
        prefetching and every object is read lazily on demand.

        Concurrency has to happen at this level rather than around the h5py
        reads: h5py serializes every HDF5 call behind a global lock, and because
        the file object is a Python fsspec stream the network waits happen
        inside that lock too, so threading the reads themselves buys nothing.
        Fetching whole objects in the background instead gives genuinely
        concurrent S3 streams and overlaps transfer with decompression.

        Each in flight object is held whole, in memory when streaming, so peak
        memory grows by roughly ``prefetch_workers`` times the object size, and
        pressure level files are on the order of a gigabyte. Size this against
        the ranks you place per node. With ``cache_dir`` set the prefetch lands
        on disk instead and costs no memory.
    """

    def __init__(
        self,
        bucket: str,
        cache_dir: Optional[str] = None,
        max_open: Optional[int] = 16,
        prefetch_workers: Optional[int] = 0,
    ):
        import s3fs

        self.bucket = bucket
        self.cache_dir = cache_dir
        self.max_open = max_open
        self.prefetch_workers = prefetch_workers
        self.fs = s3fs.S3FileSystem(anon=True)
        self._handles = OrderedDict()
        self._coords = {}
        self._pool = ThreadPoolExecutor(prefetch_workers) if prefetch_workers > 0 else None
        self._pending = OrderedDict()
        self._plan = []
        self._plan_position = 0

    def _fetch(self, key: str) -> str:
        """Download ``key`` into the cache directory if not already present, return its local path."""
        path = os.path.join(self.cache_dir, key)
        if os.path.isfile(path):
            return path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # unique per rank and per prefetch thread, so two fetches can never
        # stage through the same temporary path
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.part"
        self.fs.get(f"{self.bucket}/{key}", tmp)
        os.replace(tmp, path)
        return path

    def set_read_plan(self, keys: List[str]):
        """Declare the objects this rank will read, in order, and start filling the pipe.

        The plan is what makes prefetching useful: the store can keep exactly
        ``prefetch_workers`` fetches in flight at all times, topping the queue up
        as each object is consumed, rather than draining and refilling in bursts.
        """
        self._plan = list(keys)
        self._plan_position = 0
        self._top_up()

    def _top_up(self):
        """Queue background fetches until ``prefetch_workers`` objects are in flight."""
        if self._pool is None:
            return
        while len(self._pending) < self.prefetch_workers and self._plan_position < len(self._plan):
            key = self._plan[self._plan_position]
            self._plan_position += 1
            if key in self._pending or key in self._handles:
                continue
            if self.cache_dir is not None:
                self._pending[key] = self._pool.submit(self._fetch, key)
            else:
                self._pending[key] = self._pool.submit(self.fs.cat_file, f"{self.bucket}/{key}")

    def open(self, key: str) -> h5.File:
        """Return an open :class:`h5py.File` for ``key``, reusing a cached handle if possible."""
        if key in self._handles:
            self._handles.move_to_end(key)
            return self._handles[key][0]

        while len(self._handles) >= self.max_open:
            stale_key, (stale_file, stale_raw) = self._handles.popitem(last=False)
            stale_file.close()
            if stale_raw is not None:
                stale_raw.close()
            for cached in [k for k in self._coords if k[0] == stale_key]:
                del self._coords[cached]

        pending = self._pending.pop(key, None)
        if pending is not None:
            # blocks only if the background fetch has not finished yet
            fetched = pending.result()
            self._top_up()
            if self.cache_dir is not None:
                handle, raw = h5.File(fetched, "r"), None
            else:
                raw = io.BytesIO(fetched)
                handle = h5.File(raw, "r")
        elif self.cache_dir is not None:
            handle, raw = h5.File(self._fetch(key), "r"), None
        else:
            raw = self.fs.open(f"{self.bucket}/{key}", "rb")
            handle = h5.File(raw, "r")

        self._handles[key] = (handle, raw)
        return handle

    def release(self, key: str):
        """Close ``key`` and drop its cached coordinates.

        Called once a file is known to be finished with. This matters for the
        pressure level files: they are consumed entirely within a single day, and
        under prefetching they are resident in memory whole, so holding them in
        the LRU until something evicts them would pin gigabytes per rank for no
        benefit.
        """
        entry = self._handles.pop(key, None)
        if entry is not None:
            handle, raw = entry
            handle.close()
            if raw is not None:
                raw.close()
        for cached in [k for k in self._coords if k[0] == key]:
            del self._coords[cached]

    def coord(self, key: str, name: str) -> np.ndarray:
        """Return a one dimensional coordinate of ``key``, cached in memory.

        Coordinates are tiny but are consulted once per sample per variable to
        locate a timestep, and over S3 each of those lookups is a round trip.
        Entries are dropped when the owning handle is evicted, so a reopened
        file is always read afresh.
        """
        if (key, name) not in self._coords:
            self._coords[(key, name)] = self.open(key)[name][:]
        return self._coords[(key, name)]

    def close(self):
        """Close all open handles and drop anything still being prefetched."""
        for pending in self._pending.values():
            pending.cancel()
        self._pending.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for handle, raw in self._handles.values():
            handle.close()
            if raw is not None:
                raw.close()
        self._handles.clear()
        self._coords.clear()


def _mask_fill(data: np.ndarray, dset: h5.Dataset) -> np.ndarray:
    """Replace the netCDF ``_FillValue`` with NaN, in place.

    Fields that are only defined over part of the globe, most notably sea
    surface temperature, carry a large sentinel over the undefined region. The
    makani dataloaders understand NaN, so translate it here.

    ``data`` must be a freshly read array that the caller owns, since it is
    modified in place: the pressure level blocks are 150 MB apiece and neither a
    copy nor the float64 promotion that ``np.where(..., np.nan, ...)`` would
    introduce is worth paying for on this path.
    """
    fill = dset.attrs.get("_FillValue", None)
    if fill is None:
        return data
    mask = data == np.asarray(fill).astype(data.dtype).item()
    if mask.any():
        data[mask] = np.nan
    return data


def _coord_index(values: np.ndarray, wanted: int, name: str) -> int:
    """Return the position of ``wanted`` (hours since 1900-01-01) in a time coordinate."""
    matches = np.nonzero(values == wanted)[0]
    if matches.size == 0:
        raise IndexError(f"Time {wanted} not found in coordinate '{name}'.")
    return int(matches[0])


def _check_grid(handle: h5.File, lat: List[float], lon: List[float]):
    """Verify that the source grid matches the grid declared in the metadata."""
    file_lat, file_lon = handle["latitude"][:], handle["longitude"][:]
    if file_lat.shape[0] != len(lat) or file_lon.shape[0] != len(lon):
        raise ValueError(
            f"Grid mismatch: NCAR ERA5 is {file_lat.shape[0]}x{file_lon.shape[0]} but the metadata declares "
            f"{len(lat)}x{len(lon)}. This converter does not regrid; use a metadata file on the native "
            f"0.25 degree grid."
        )
    if not (np.allclose(file_lat, lat) and np.allclose(file_lon, lon)):
        raise ValueError(
            "Grid mismatch: NCAR ERA5 coordinates differ from the metadata coordinates. Expected latitude "
            "descending from 90 to -90 and longitude ascending from 0 to 359.75."
        )


def _keys_for_day(groups, day, day_times, window_hours) -> List[str]:
    """List every object needed to fill one day, in the order the fill loop reads them."""
    keys = []
    for group in groups:
        if group.kind == "pl":
            keys.append(analysis_pl_key(group.variables[0], day))
        elif group.kind == "sfc":
            keys.append(analysis_sfc_key(group.variables[0], day))
        else:
            for _, valid_time in day_times:
                for init_time, _, _ in resolve_accumulation_segments(valid_time, window_hours):
                    keys += [accumulation_key(variable, init_time) for variable in group.variables]
    # preserve order while dropping repeats, since accumulation runs are shared between samples
    return list(dict.fromkeys(keys))


def _fill_pressure_levels(store, group, out, entry_key, day, day_times, lat, lon, grid_checked):
    """Fill all channels of one pressure level variable for a single day."""
    key = analysis_pl_key(group.variables[0], day)
    handle = store.open(key)
    if key not in grid_checked:
        _check_grid(handle, lat, lon)
        grid_checked.add(key)

    dset = handle[group.variables[0].h5_name]
    levels = list(store.coord(key, "level").astype(int))
    level_positions = [levels.index(level) for level in group.levels]
    times = store.coord(key, "time")

    for sample_index, valid_time in day_times:
        # one read pulls the whole chunk, which holds every level of this timestep
        block = _mask_fill(dset[_coord_index(times, to_ncar_hours(valid_time), "time")], dset)
        for cidx, position in zip(group.channel_indices, level_positions):
            out[entry_key][sample_index, cidx, ...] = block[position]

    # a pressure level file covers exactly one day, so it is finished with here
    store.release(key)


def _fill_surface(store, group, out, entry_key, day, day_times, lat, lon, grid_checked):
    """Fill one surface analysis channel for a single day."""
    variable = group.variables[0]
    key = analysis_sfc_key(variable, day)
    handle = store.open(key)
    if key not in grid_checked:
        _check_grid(handle, lat, lon)
        grid_checked.add(key)

    dset = handle[variable.h5_name]
    times = store.coord(key, "time")
    cidx = group.channel_indices[0]
    for sample_index, valid_time in day_times:
        index = _coord_index(times, to_ncar_hours(valid_time), "time")
        out[entry_key][sample_index, cidx, ...] = _mask_fill(dset[index], dset)


def _fill_accumulated(store, group, out, entry_key, day, day_times, window_hours, lat, lon, grid_checked):
    """Fill one accumulated channel for a single day.

    NCAR stores the accumulated fields already de-accumulated, one value per
    forecast hour, so the value over the window is the sum of the hours it
    spans, taken across every run the window touches. Each read returns all 12
    forecast hours of a run, and consecutive samples usually share a run, so the
    runs are memoized for the duration of the day.
    """
    cidx = group.channel_indices[0]
    runs: Dict[tuple, np.ndarray] = {}

    def read_run(variable, init_time):
        key = accumulation_key(variable, init_time)
        if (key, variable.h5_name) not in runs:
            handle = store.open(key)
            if key not in grid_checked:
                _check_grid(handle, lat, lon)
                grid_checked.add(key)
            dset = handle[variable.h5_name]
            inits = store.coord(key, "forecast_initial_time")
            index = _coord_index(inits, to_ncar_hours(init_time), "forecast_initial_time")
            runs[(key, variable.h5_name)] = _mask_fill(dset[index], dset)
        return runs[(key, variable.h5_name)]

    for sample_index, valid_time in day_times:
        total = np.zeros((len(lat), len(lon)), dtype=np.float32)
        for init_time, hour_start, hour_end in resolve_accumulation_segments(valid_time, window_hours):
            for variable in group.variables:
                run = read_run(variable, init_time)
                # forecast hour h holds the accumulation over the hour ending at h
                total += run[hour_start:hour_end].sum(axis=0)

        out[entry_key][sample_index, cidx, ...] = total


def convert(
    output_dir: str,
    metadata_file: str,
    years: List[int],
    bucket: Optional[str] = NCAR_ERA5_BUCKET,
    entry_key: Optional[str] = "fields",
    cache_dir: Optional[str] = None,
    accumulation_hours: Optional[int] = None,
    prefetch_workers: Optional[int] = 0,
    force_overwrite: Optional[bool] = False,
    skip_missing_channels: Optional[bool] = False,
    verbose: Optional[bool] = False,
):
    """Convert NSF NCAR ERA5 (RDA d633000) data on S3 to makani format.

    Data is streamed from the public bucket straight into one makani HDF5 file
    per year, without staging the source netCDF files, unless ``cache_dir`` is
    given. The NCAR grid is already the makani grid, so no regridding or
    latitude flipping takes place; a mismatch against the metadata grid is an
    error rather than something this routine tries to fix.

    This routine supports distributed processing via mpi4py. Work is split over
    whole days rather than individual timestamps, because the pressure level
    files are chunked as one chunk per timestep across all levels and splitting
    a day across ranks would make several ranks fetch and decompress the same
    chunk.

    Parameters
    ----------
    output_dir : str
        Directory to where output files will be written to (makani format). One file per year will be written.
    metadata_file : str
        name of the file to read metadata from. The metadata is a json file, and after reading it should be a
        dictionary containing metadata describing the dataset. Most important entries are:
        dhours: distance between subsequent samples in hours
        coords: this is a dictionary which contains two lists, latitude and longitude coordinates in degrees as well as channel names.
        Example: coords = dict(lat=[-90.0, ..., 90.], lon=[0, ..., 360], channel=["t2m", "u500", "v500", ...])
        Note that the number of entries in coords["lat"] has to match dimension -2 of the dataset, and coords["lon"] dimension -1.
        The length of the channel names has to match dimension -3 (or dimension 1, which is the same) of the dataset.
    years : List[int]
        List of years to extract from the cloud dataset
    bucket : str
        Name of the S3 bucket holding the NCAR ERA5 data.
    entry_key : str
        This is the HDF5 dataset name of the data in the files. Defaults to "fields".
    cache_dir : str, optional
        Directory used to cache the raw NCAR files. Without it nothing is written
        to local disk, but an interrupted run has to refetch everything. Note that
        the raw files are considerably larger than the converted output.
    accumulation_hours : int, optional
        Length in hours of the window used for accumulated channels such as tp.
        Defaults to dhours. Windows longer than a forecast run are stitched
        together from consecutive runs.
    prefetch_workers : int, optional
        Background threads per rank fetching upcoming objects, overlapping
        transfer with decompression and giving each rank several concurrent S3
        streams. Zero, the default, reads lazily on demand. This only helps if
        the reads are latency bound rather than capped by aggregate bandwidth;
        when the link is already saturated it adds memory pressure for nothing.
        While streaming, peak memory grows by roughly this many object sizes,
        and pressure level objects are on the order of a gigabyte, so weigh it
        against the ranks placed per node.
    force_overwrite : bool
        Setting this flag to True will overwrite existing files.
    skip_missing_channels : bool
        Setting this flag to True will skip channels without an NCAR counterpart instead of failing.
    verbose : bool
        Enable for more printing.
    """

    # get comm ranks and size
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # timer
    start_time = time.perf_counter()

    # get metadata info
    metadata = None
    if comm_rank == 0:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    metadata = comm.bcast(metadata, root=0)
    dhours = metadata["dhours"]
    channel_names = metadata["coords"]["channel"]
    chanlen = max([len(v) for v in channel_names])
    lat = metadata["coords"]["lat"]
    lon = metadata["coords"]["lon"]

    # group channels by the source file that provides them
    groups = build_ncar_channel_groups(channel_names, skip_missing_channels=skip_missing_channels)
    if accumulation_hours is None:
        accumulation_hours = dhours
    if comm_rank == 0:
        covered = sum(len(g.channel_indices) for g in groups)
        if covered != len(channel_names):
            warnings.warn(f"Skipping {len(channel_names) - covered} channels without an NCAR counterpart.")
        if any(g.kind == "accum" for g in groups):
            print(f"Accumulated channels use a {accumulation_hours}h window ending at the sample time.")

    store = NcarStore(bucket, cache_dir=cache_dir, prefetch_workers=prefetch_workers)
    grid_checked = set()

    # check total number of entries:
    num_entries_total = 0
    timelist = []
    for year in years:
        start_date = dt.datetime(year=year, day=1, month=1, tzinfo=dt.timezone.utc)
        end_date = dt.datetime(year=year, day=31, month=12, hour=23, tzinfo=dt.timezone.utc)
        hours_in_year = int((end_date - start_date).total_seconds() // 3600)
        times = [start_date + h * dt.timedelta(hours=1) for h in range(0, hours_in_year + 1, dhours)]
        timelist.append(times)
        num_entries_total += len(times)

    # set up distributed progressbar
    pbar = DistributedProgressBar(num_entries_total, comm)

    # do loop over years
    for idy, year in enumerate(years):
        times = timelist[idy]
        dataset_shape = (len(times), len(channel_names), len(lat), len(lon))

        # bucket the samples by calendar day, then hand out contiguous runs of
        # days so that a rank keeps hitting the same monthly surface file
        days = OrderedDict()
        for sample_index, valid_time in enumerate(times):
            days.setdefault(valid_time.date(), []).append((sample_index, valid_time))
        days = list(days.items())

        num_days_local = (len(days) + comm_size - 1) // comm_size
        start_days = min(comm_rank * num_days_local, len(days))
        end_days = min(start_days + num_days_local, len(days))
        days_local = days[start_days:end_days]
        num_samples_local = sum(len(v) for _, v in days_local)

        if verbose:
            print(f"Rank {comm_rank}: number of local days: {len(days_local)} ({num_samples_local} samples)")

        # helper arrays:
        timestamps = np.array([t.timestamp() for t in times], dtype=np.float64)

        comm.Barrier()
        ofile = os.path.join(output_dir, f"{year}.h5")
        file_exists = False
        if comm_rank == 0:
            file_exists = os.path.isfile(ofile)
        file_exists = comm.bcast(file_exists, root=0)
        if file_exists and not force_overwrite:
            if comm_rank == 0:
                print(f"File {ofile} already exists, skipping.")
            pbar.update_counter(num_samples_local)
            pbar.update_progress()
            continue

        f = h5.File(ofile, "w", driver="mpio", comm=comm)
        f.create_dataset(entry_key, dataset_shape, dtype=np.float32)

        # create dimension scales
        # datasets
        f.create_dataset("valid_data", data=np.ones((len(timestamps), len(channel_names)), dtype=np.int32))
        f.create_dataset("timestamp", data=timestamps)
        f.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
        f["channel"][...] = channel_names
        f.create_dataset("lat", data=lat)
        f.create_dataset("lon", data=lon)
        # scales
        f["timestamp"].make_scale("timestamp")
        f["channel"].make_scale("channel")
        f["lat"].make_scale("lat")
        f["lon"].make_scale("lon")
        # label
        f[entry_key].dims[0].label = "Timestamp in seconds in UTC time zone"
        f[entry_key].dims[1].label = "Channel name"
        f[entry_key].dims[2].label = "Latitude in degrees"
        f[entry_key].dims[3].label = "Longitude in degrees"
        # attach
        f[entry_key].dims[0].attach_scale(f["timestamp"])
        f[entry_key].dims[1].attach_scale(f["channel"])
        f[entry_key].dims[2].attach_scale(f["lat"])
        f[entry_key].dims[3].attach_scale(f["lon"])

        # Declare the objects to prefetch, so the store can keep the queue topped
        # up across day boundaries rather than in bursts. Only the pressure level
        # files qualify: they are read in full within one day, which is exactly
        # what a whole-object fetch suits, and they dominate the byte volume. The
        # surface and accumulation files are read in slivers spread over a month
        # and are better left on the lazy block-cached path, which also keeps
        # them from sitting in memory whole for weeks of simulated time.
        pressure_level_groups = [group for group in groups if group.kind == "pl"]
        store.set_read_plan(
            [
                key
                for day, day_times in days_local
                for key in _keys_for_day(pressure_level_groups, day, day_times, accumulation_hours)
            ]
        )

        # populate fields
        for day, day_times in days_local:
            for group in groups:
                if group.kind == "pl":
                    _fill_pressure_levels(store, group, f, entry_key, day, day_times, lat, lon, grid_checked)
                elif group.kind == "sfc":
                    _fill_surface(store, group, f, entry_key, day, day_times, lat, lon, grid_checked)
                else:
                    _fill_accumulated(
                        store, group, f, entry_key, day, day_times, accumulation_hours, lat, lon, grid_checked
                    )

            # update progressbar
            pbar.update_counter(len(day_times))
            pbar.update_progress()

        # we need to wait here
        if verbose:
            print(f"Rank {comm_rank}: waiting for barrier on file {ofile}.")
        comm.Barrier()

        # close file
        f.close()

    store.close()

    # do a final pbar update
    comm.Barrier()
    pbar.update_progress()

    # end time
    end_time = time.perf_counter()
    run_time = str(dt.timedelta(seconds=end_time - start_time))

    if comm_rank == 0:
        print(f"All done. Run time {run_time}.")

    comm.Barrier()

    return


def main(args):
    convert(
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
        years=args.years,
        bucket=args.bucket,
        cache_dir=args.cache_dir,
        accumulation_hours=args.accumulation_hours,
        prefetch_workers=args.prefetch_workers,
        force_overwrite=args.force_overwrite,
        skip_missing_channels=args.skip_missing_channels,
        verbose=args.verbose,
    )


if __name__ == "__main__":

    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Local directory for output files.", required=True)
    parser.add_argument("--metadata_file", type=str, help="Local file with metadata.", required=True)
    parser.add_argument("--years", type=int, nargs="+", help="Which years to convert", required=True)
    parser.add_argument("--bucket", type=str, default=NCAR_ERA5_BUCKET, help="S3 bucket with NCAR ERA5 data")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional directory to cache raw NCAR files in")
    parser.add_argument(
        "--accumulation_hours",
        type=int,
        default=None,
        help="Window in hours for accumulated channels such as tp. Defaults to dhours.",
    )
    parser.add_argument(
        "--prefetch_workers",
        type=int,
        default=0,
        help="Background fetch threads per rank. Helps when reads are latency bound; costs roughly "
        "this many object sizes of memory per rank while streaming.",
    )
    parser.add_argument("--skip_missing_channels", action="store_true", help="Skip missing channels and do not fail")
    parser.add_argument("--force_overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
