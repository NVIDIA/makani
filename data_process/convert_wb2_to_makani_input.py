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

from typing import Optional, List
from itertools import batched
import progressbar
import os
import re
import json
import time
import pickle
import numpy as np
import h5py as h5
import datetime as dt
import argparse as ap
from glob import glob
import xarray as xr

# MPI
from mpi4py import MPI

from makani.utils.features import get_channel_groups
from makani.utils.dataloaders.data_helpers import get_date_from_timestamp

from wb2_helpers import surface_variables, atmospheric_variables, split_convert_channel_names


def convert(input_file: str, output_dir: str, metadata_file: str, years: List[int],
            batch_size: Optional[int]=32, entry_key: Optional[str]='fields',
            skip_missing: Optional[bool]=False, verbose: Optional[bool]=False):

    """Function to convert ARCO-ERA5 data (used by Weatherbench 2) to makani format.

    This function reads all files from the input_path and generates a WB2 compatible output file which
    is stored as specified in output_file.

    This routine supports distributed processing via mpi4py.
    ...

    Parameters
    ----------
    input_file : str
        GCS specifier for the ARCO-ERA5 dataset
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
    batch_size : int
        Batch size in which the samples are processed. This does not have any effect on the statistics (besides small numerical changes because of order of operations), but
        is merely a performance setting. Bigger batches are more efficient but require more memory.
    entry_key: str
        This is the HDF5 dataset name of the data in the files. Defaults to "fields".
    skip_missing: bool
        Setting this flag to True will skip missing data instead of failing.
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
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    metadata = comm.bcast(metadata, root=0)
    dhours = metadata["dhours"]
    channel_names = metadata['coords']['channel']
    chanlen = max([len(v) for v in channel_names])
    lat = metadata['coords']["lat"]
    lon = metadata['coords']["lon"]

    # split in surface and atmospheric channels
    atmospheric_channel_names, atmospheric_channel_names_wb2, surface_channel_names, surface_channel_names_wb2, atmospheric_levels = split_convert_channel_names(channel_names)

    # open cloud dataset
    wb2_data = xr.open_dataset(input_file, engine="zarr")

    # check total number of entries:
    num_entries_total = 0
    timelist = []
    for year in years:
        start_date = dt.datetime(year=year, day=1, month=1, tzinfo=dt.timezone.utc)
        end_date = dt.datetime(year=year, day=31, month=12, hour=23, tzinfo=dt.timezone.utc)
        hours_in_year = int((end_date - start_date).total_seconds() // 3600)
        times = [start_date + h * dt.timedelta(hours=1) for h in range(0,hours_in_year+1,dhours)]
        timelist.append(times)
        num_entries_total += len(times)

    # set up progressbar
    if comm_rank == 0:
        pbar = progressbar.ProgressBar(maxval=num_entries_total)
        pbar.update(0)

    # do loop over years
    skipped_channels = set()
    num_entries_current = 0
    for idy, year in enumerate(years):
        times = timelist[idy]
        dataset_shape = (len(times), len(channel_names), len(lat), len(lon))

        # local lists to work on
        num_dates_local = (len(times) + comm_size - 1) // comm_size
        start_dates = comm_rank * num_dates_local
        end_dates = min(start_dates + num_dates_local, len(times))
        times_local = times[start_dates:end_dates]

        # helper arrays:
        timestamps = np.array([t.timestamp() for t in times], dtype=np.float64)

        with h5.File(os.path.join(output_dir, f"{year}.h5"), "w", driver="mpio", comm=comm) as f:
            f.create_dataset(entry_key, dataset_shape, dtype=np.float32)

            # create dimension scales
            # datasets
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

            # populate fields
            for timebatch in batched(times_local, batch_size):
                tstart = times.index(timebatch[0])
                tend = tstart + len(timebatch)

                # surface channel variables
                for sc,scwb2 in zip(surface_channel_names,surface_channel_names_wb2):
                    cidx = channel_names.index(sc)
                    if scwb2 not in wb2_data:
                        if skip_missing:
                            if (comm_rank == 0) and not (scwb2 in skipped_channels):
                                print(f"Key {scwb2} not found in dataset, skipping")
                            skipped_channels.add(scwb2)
                            continue
                        else:
                            raise IndexError(f"Key {scwb2} not found in dataset.")
                    timebatch = [np.datetime64(t) for t in list(timebatch)]
                    wb2_sel = wb2_data[scwb2]
                    data = wb2_sel[wb2_sel["time"].isin(timebatch)].values

                    # checks:
                    if data.shape[0] != len(timebatch):
                        raise IndexError(f"Dates {timebatch} not all found in dataset for {scwb2}.")

                    f[entry_key][tstart:tend, cidx, ...] = data[...]

                # atmospheric level variables
                for ac, acwb2 in zip(atmospheric_channel_names, atmospheric_channel_names_wb2):
                    for idl, alevel in enumerate(atmospheric_levels):
                        cidx = channel_names.index(ac + str(alevel))
                        if acwb2 not in wb2_data:
                            if skip_missing:
                                if (comm_rank == 0) and not (acwb2 in skipped_channels):
                                     print(f"Key {acwb2} not found in dataset, skipping")
                                skipped_channels.add(acwb2)
                                continue
                            else:
                                raise IndexError(f"Key {acwb2} not found in dataset.")
                        wb2_sel = wb2_data[acwb2].sel(level=alevel)
                        data = wb2_sel[wb2_sel["time"].isin(timebatch)].values

                        if data.shape[0] != len(timebatch):
                            raise IndexError(f"Dates {timebatch} not all found in dataset for {acwb2}.")

                        f[entry_key][tstart:tend, cidx, ...] = data[...]

                # update progressbar
                num_entries_current += len(timebatch) * comm_size
                if comm_rank == 0:
                    pbar.update(num_entries_current)

            # we need to wait here
            comm.Barrier()

    # end time
    end_time = time.perf_counter()
    run_time = str(dt.timedelta(seconds=end_time-start_time))

    if comm_rank == 0:
        pbar.finish()
        print(f"All done. Run time {run_time}. Skipped channels: {list(skipped_channels)}")

    comm.Barrier()

    return


def main(args):
    # concatenate files with timestamp information
    convert(input_file=args.input_file,
            output_dir=args.output_dir,
            metadata_file=args.metadata_file,
            years=args.years,
            batch_size=args.batch_size,
            skip_missing=args.skip_missing,
            verbose=args.verbose)


if __name__ == '__main__':

    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="WB2 input file", required=True)
    parser.add_argument("--output_dir", type=str, help="Local directory for output files.", required=True)
    parser.add_argument("--metadata_file", type=str, help="Local file with metadata.", required=True)
    parser.add_argument("--years", type=int, nargs='+', help="Which years to convert", required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for writing chunks")
    parser.add_argument("--skip_missing", action="store_true", help="Skip missing channels and do not fail")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
