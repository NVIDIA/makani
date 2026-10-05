# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Callable, Dict, Optional
from itertools import islice
import os
import sys
import json
import time
import functools
import numpy as np
import h5py as h5
import datetime as dt
import argparse as ap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from makani.utils.dataloaders.ncar_helpers import NCAR_ERA5_BUCKET
from data_process.date_range import add_date_range_arguments, date_range_from_args, yearly_sample_times
from data_process.sources import Source


def _batched(items, size):
    """Consecutive tuples of up to ``size`` items; ``itertools.batched`` needs Python 3.12."""
    iterator = iter(items)
    while batch := tuple(islice(iterator, size)):
        yield batch


def _create_output_file(ofile, comm, entry_key, timestamps, channel_names, lat, lon) -> h5.File:
    """Create a yearly makani file with its datasets and dimension scales.

    The file is opened for parallel writing on ``comm``, or serially if ``comm`` is None.
    """
    chanlen = max([len(v) for v in channel_names])
    dataset_shape = (len(timestamps), len(channel_names), len(lat), len(lon))

    f = h5.File(ofile, "w", driver="mpio", comm=comm) if comm is not None else h5.File(ofile, "w")
    # Declare NaN as the fill value, so that missing data is self describing to
    # any HDF5 reader. The fill time has to stay "never": parallel HDF5
    # allocates storage at creation, so any other fill time would write the
    # whole dataset once up front just to prefill it. As a consequence, missing
    # data has to be written as NaN explicitly. The fill time is set on a
    # property list rather than through create_dataset, whose fill_time keyword
    # is newer than the h5py versions supported.
    dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
    dcpl.set_fill_time(h5.h5d.FILL_TIME_NEVER)
    f.create_dataset(entry_key, dataset_shape, dtype=np.float32, fillvalue=np.nan, dcpl=dcpl)

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

    return f


def _write_missing(out, entry_key, samples, channel_indices):
    """Write NaN and clear ``valid_data`` for ``channel_indices`` at a contiguous run of samples."""
    tstart, tend = samples[0][0], samples[-1][0] + 1
    nan = np.full((tend - tstart, *out[entry_key].shape[2:]), np.nan, dtype=np.float32)
    for cidx in channel_indices:
        out[entry_key][tstart:tend, cidx, ...] = nan
        out["valid_data"][tstart:tend, cidx] = 0


def convert(
    make_source: Callable[[Dict, int], Source],
    output_dir: str,
    metadata_file: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    entry_key: Optional[str] = "fields",
    force_overwrite: Optional[bool] = False,
    verbose: Optional[bool] = False,
):
    """Convert ERA5 from one of the supported archives to makani format, one file per year.

    The archive specific reading is delegated to a :class:`data_process.sources.Source`;
    this routine owns the output files, the date range and the distribution of
    work. It supports distributed processing via mpi4py: the source groups the
    samples of each year into units that must stay on one rank, and the units are
    handed out to the ranks in contiguous runs.

    Parameters
    ----------
    make_source : Callable[[Dict, int], Source]
        Called with the metadata dictionary and the MPI rank, returns the source to read from.
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
    start_date : datetime.datetime
        First time to extract, inclusive. Samples stay on the ``dhours`` grid
        anchored at 00Z on January 1st, so a start date off that grid is
        rounded up to the next sample.
    end_date : datetime.datetime
        Last time to extract, inclusive. One file is written for every year
        touched by the range; the first and last of them may be partial years.
    entry_key : str
        This is the HDF5 dataset name of the data in the files. Defaults to "fields".
    force_overwrite : bool
        Setting this flag to True will overwrite existing files.
    verbose : bool
        Enable for more printing.
    """

    # imported here so that the helpers above can be used and tested without MPI
    from mpi4py import MPI
    from data_process.data_process_helpers import DistributedProgressBar

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
    channel_names = metadata["coords"]["channel"]
    lat = metadata["coords"]["lat"]
    lon = metadata["coords"]["lon"]

    source = make_source(metadata, comm_rank)

    # check total number of entries:
    years, timelist = zip(*yearly_sample_times(start_date, end_date, metadata["dhours"]))
    num_entries_total = sum(len(times) for times in timelist)
    if comm_rank == 0:
        for year, times in zip(years, timelist):
            print(f"{year}: {len(times)} samples from {times[0]:%Y-%m-%dT%H} to {times[-1]:%Y-%m-%dT%H}")

    # set up distributed progressbar
    pbar = DistributedProgressBar(num_entries_total, comm)

    # do loop over years
    for year, times in zip(years, timelist):

        # hand out contiguous runs of units
        units = source.split_units(list(enumerate(times)))
        num_units_local = (len(units) + comm_size - 1) // comm_size
        start_units = min(comm_rank * num_units_local, len(units))
        end_units = min(start_units + num_units_local, len(units))
        units_local = units[start_units:end_units]
        num_samples_local = sum(len(unit) for unit in units_local)

        if verbose:
            print(f"Rank {comm_rank}: number of local units: {len(units_local)} ({num_samples_local} samples)")

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

        timestamps = np.array([t.timestamp() for t in times], dtype=np.float64)
        f = _create_output_file(ofile, comm, entry_key, timestamps, channel_names, lat, lon)

        # populate fields; channels the source cannot provide are written as
        # missing, since the fill value is declared but never written
        skipped_channels = source.skipped_channel_indices()
        source.begin_year(units_local)
        for unit_batch in _batched(units_local, source.units_per_fill):
            source.fill(f, entry_key, list(unit_batch))
            if skipped_channels:
                _write_missing(f, entry_key, [sample for unit in unit_batch for sample in unit], skipped_channels)

            # update progressbar
            pbar.update_counter(sum(len(unit) for unit in unit_batch))
            pbar.update_progress()

        # we need to wait here
        if verbose:
            print(f"Rank {comm_rank}: waiting for barrier on file {ofile}.")
        comm.Barrier()

        # close file
        f.close()

    summary = source.summary()
    source.close()

    # do a final pbar update
    comm.Barrier()
    pbar.update_progress()

    # end time
    end_time = time.perf_counter()
    run_time = str(dt.timedelta(seconds=end_time - start_time))

    if comm_rank == 0:
        print(f"All done. Run time {run_time}." + (f" {summary}" if summary else ""))

    comm.Barrier()

    return


def main(args):
    # backends are imported lazily, so that each only needs its own dependencies
    if args.source == "wb2":
        from data_process.sources.wb2 import Wb2Source

        make_source = functools.partial(
            Wb2Source,
            input_file=args.input_file,
            batch_size=args.batch_size,
            coord_mode=args.coord_mode,
            skip_missing_channels=args.skip_missing_channels,
            impute_missing_timestamps=args.impute_missing_timestamps,
        )
    elif args.source == "ncar":
        from data_process.sources.ncar import NcarSource

        make_source = functools.partial(
            NcarSource,
            bucket=args.bucket,
            cache_dir=args.cache_dir,
            accumulation_hours=args.accumulation_hours,
            prefetch_workers=args.prefetch_workers,
            skip_missing_channels=args.skip_missing_channels,
            impute_missing_timestamps=args.impute_missing_timestamps,
        )
    else:
        raise ValueError(f"Unknown source {args.source}.")

    start_date, end_date = date_range_from_args(args)
    convert(
        make_source=make_source,
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
        start_date=start_date,
        end_date=end_date,
        force_overwrite=args.force_overwrite,
        verbose=args.verbose,
    )


def build_parser() -> ap.ArgumentParser:
    """Command line parser with one subcommand per source, sharing the output related options."""
    common = ap.ArgumentParser(add_help=False)
    common.add_argument("--output_dir", type=str, help="Local directory for output files.", required=True)
    common.add_argument("--metadata_file", type=str, help="Local file with metadata.", required=True)
    add_date_range_arguments(common)
    common.add_argument("--skip_missing_channels", action="store_true", help="Skip missing channels and do not fail")
    common.add_argument(
        "--impute_missing_timestamps",
        action="store_true",
        help="Write NaN and clear valid_data for data missing from the source, instead of failing",
    )
    common.add_argument("--force_overwrite", action="store_true", help="Overwrite existing files")
    common.add_argument("--verbose", action="store_true")

    parser = ap.ArgumentParser(description="Convert ERA5 to makani format, one HDF5 file per year.")
    sources = parser.add_subparsers(dest="source", required=True, metavar="source")

    wb2 = sources.add_parser("wb2", parents=[common], help="WeatherBench2 / ARCO-ERA5 zarr store, e.g. on GCS")
    wb2.add_argument("--input_file", type=str, help="WB2 input file", required=True)
    wb2.add_argument("--batch_size", type=int, default=32, help="Batch size for writing chunks")
    wb2.add_argument(
        "--coord_mode",
        type=str,
        default="match",
        choices=["match", "force-flip-lat", "force"],
        help="How to align input lat/lon to metadata: match (default), force-flip-lat, force",
    )

    ncar = sources.add_parser("ncar", parents=[common], help="NSF NCAR ERA5 (RDA d633000) on S3")
    ncar.add_argument("--bucket", type=str, default=NCAR_ERA5_BUCKET, help="S3 bucket with NCAR ERA5 data")
    ncar.add_argument("--cache_dir", type=str, default=None, help="Optional directory to cache raw NCAR files in")
    ncar.add_argument(
        "--accumulation_hours",
        type=int,
        default=None,
        help="Window in hours for accumulated channels such as tp. Defaults to dhours.",
    )
    ncar.add_argument(
        "--prefetch_workers",
        type=int,
        default=0,
        help="Background fetch threads per rank. Helps when reads are latency bound; costs roughly "
        "this many object sizes of memory per rank while streaming.",
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
