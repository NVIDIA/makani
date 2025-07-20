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

import os
from typing import List, Optional
import json
import time
import pickle
import numpy as np
import h5py as h5
import datetime as dt
import argparse as ap
from glob import glob


def concatenate(input_output_dir: str, output_file: str, metadata: dict,
                file_names_to_concatenate: List[str], years: List[int], dhoursrel: Optional[int]=1,
                entry_key: Optional[str]="fields"):
    """Function to concatenate multiple HDF5 files of an existing makani compatible HDF5 dataset.

    The concatenation is performed virtually, not physically. Which means the disk overhead is small, since only a file
    with references to the original file will be created. The file will be written to os.path.join(input_output_dir, output_file),
    so it will reside in the same directory as the input files. The references inside the virtual file will be with respect to the local folder.
    While virtual HDF5 datasets support specifying any location, we chose to co-locate them in the same directory. The reason is that the files can be found,
    even if the full path is not available, for example when the volume is mounted into a container under another mount point than it has on the host system.

    ...

    Parameters
    ----------
    input_output_dir: str
        directory to where the dataset files are located which are to be concatenated. The concatenated dataset will be written to this same directory.
    output_file: str
        file name of the concatenated dataset, has to be different from all the other file names in the directory. To distinguish the file from the others,
        it is recommended to give it a different suffix, e.g. .h5v instead of .h5.
    metadata : dict
        dictionary containing metadata describing the dataset. Most important entries are:
        dhours: distance between subsequent samples in hours
        coords: this is a dictionary which contains two lists, latitude and longitude coordinates in degrees as well as channel names.
        Example: coords = dict(lat=[-90.0, ..., 90.], lon=[0, ..., 360], channel=["t2m", "u500", "v500", ...])
        Note that the number of entries in coords["lat"] has to match dimension -2 of the dataset, and coords["lon"] dimension -1.
        The length of the channel names has to match dimension -3 (or dimension 1, which is the same) of the dataset.
    file_names_to_annotate : List[str]
        List of filenames to annotate. Has to be the same length as years. File name paths have to be relative to input_output_dir.
    years : int
        List of years, one for each file. For example if the kth file in file_names_to_annotate stores data from year 1990, then
        years[k] = 1990. The datestampts for each entry in the files is computed based on this information and the dhours stamp in the metadata
        dictionary.
    dhoursrel : int
        Relative dhours (distance between samples in hours) of existing and newly concatenated dataset. A value of 1 means every timestamps of the original dataset is used, a value of n > 1 means that every nth timestamp is used.
    entry_key: str
        This is the HDF5 dataset name of the data in the files. Defaults to "fields".
    """

    # ensure that files in file_names_to_concatenate are contained in input_output_dir:
    for fname in file_names_to_concatenate:
        ifname = os.path.join(input_output_dir, fname)
        if not os.path.isfile(ifname):
            raise FileNotFoundError(f"File {ifname} could not be found. Make sure that file {fname} is contained in the input_output_dir {input_output_dir}")

    # scan all files:
    entries_per_year = []
    entries_per_year_red = []
    for idx, fname in enumerate(file_names_to_concatenate):
        ifname = os.path.join(input_output_dir, fname)
        with h5.File(ifname, 'r') as f:
            if idx == 0:
                dataset_shape = f[entry_key].shape
                dataset_dtype = f[entry_key].dtype
            entries_per_year.append(f[entry_key].shape[0])
            entries_per_year_red.append(f[entry_key].shape[0] // dhoursrel)

    # total hours:
    total_entries = sum(entries_per_year)
    total_entries_red = sum(entries_per_year_red)

    # create a list of timestamps
    # base
    dhours = metadata["dhours"]
    jan_01_epoch = dt.datetime(years[0], 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    timestamps = []
    for idx, fname in enumerate(file_names_to_concatenate):
        ifname = os.path.join(input_output_dir, fname)
        try:
            with h5.File(ifname, 'r') as f:
                ts = f[entry_key].dims[0]["timestamp"][...]
                ts = ts[::dhoursrel]
            timestamps.append(ts.astype(np.float64))
        except:
            print(f"File {fname} is not annotated, deriving timestamps")
            year = years[idx]
            ne_red = entries_per_year_red[idx]
            jan_01_epoch = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
            ts = [jan_01_epoch + dt.timedelta(hours=h * dhours * dhoursrel) for h in range(ne_red)]
            timestamps.append(np.asarray([x.timestamp() for x in ts]).astype(np.float64))

    # convert to array
    timestamps = np.concatenate(timestamps, axis=0)

    # sanity checks:
    if timestamps.shape[0] != total_entries_red:
        raise IndexError(f"Timestamp vector has different size than number of entries in dataset: {timestamps.shape[0]} != {total_entries_red}.")

    # add lon and lat coords
    latitudes = np.array(metadata["coords"]["lat"], dtype=np.float32)
    longitudes = np.array(metadata["coords"]["lon"], dtype=np.float32)

    # add channel names
    channel_names = metadata["coords"]["channel"]
    chanlen = max([len(v) for v in channel_names])

    print( f"Combining dataset of size {total_entries} with dhours = {dhours} into single virtual file of size: {(total_entries_red,) + dataset_shape[1:]} for years {years[0]}-{years[-1]} with dhours = {dhours * dhoursrel}." )

    # create virtual layout
    layout = h5.VirtualLayout(shape=(total_entries_red,) + dataset_shape[1:],
                              dtype=dataset_dtype)

    with h5.File(os.path.join(input_output_dir, output_file), 'w', libver='latest') as f:
        # save timestamps first
        f.create_dataset("timestamp", data=timestamps.astype(np.float64))
        f.create_dataset("lat", data=latitudes)
        f.create_dataset("lon", data=longitudes)

        # channels dataset
        f.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
        f["channel"][...] = channel_names

        # create scales
        f["timestamp"].make_scale("timestamp")
        f["channel"].make_scale("channel")
        f["lat"].make_scale("lat")
        f["lon"].make_scale("lon")

        off = 0
        for filename, ne, ne_red in zip(file_names_to_concatenate, entries_per_year, entries_per_year_red):
            # shape
            shape = (ne,) + dataset_shape[1:]
            vsource = h5.VirtualSource(filename, entry_key, shape=shape)
            start = off
            end = off + ne_red

            if dhoursrel > 1:
                layout[start:end, ...] = vsource[::dhoursrel, ...]
            else:
                layout[start:end, ...] = vsource

            # increase offset in concatenated file
            off += ne_red

        f.create_virtual_dataset(entry_key, layout, fillvalue=0)

        # label dimensions
        f[entry_key].dims[0].label = "Timestamp in UTC time zone"
        f[entry_key].dims[1].label = "Channel name"
        f[entry_key].dims[2].label = "Latitude in degrees"
        f[entry_key].dims[3].label = "Longitude in degrees"

        # attach scales
        f[entry_key].dims[0].attach_scale(f["timestamp"])
        f[entry_key].dims[1].attach_scale(f["channel"])
        f[entry_key].dims[2].attach_scale(f["lat"])
        f[entry_key].dims[3].attach_scale(f["lon"])

    print("All done.")

    return


def main(args):
    # get files
    files = sorted(glob(os.path.join(args.input_output_dir, "*.h5")))

    # make sure that years are consecutive
    years = sorted([int(os.path.splitext(os.path.basename(pname))[0]) for pname in files])

    start = years[0]
    for y in range(start, start + len(years)):
        if not y in years:
            raise ValueError(f"Error, data for year {y} not found! Cannot concatenate dataset.")

    # create sorted and curated file list
    files = [str(y) + ".h5" for y in years]

    # load metadata:
    with open(args.dataset_metadata, "r") as f:
        metadata = json.load(f)

    # concatenate files with timestamp information
    concatenate(args.input_output_dir, args.output_file, metadata, files, years, args.dhours_rel)


if __name__ == '__main__':

    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--dataset_metadata", type=str, help="Input file containing metadata.", required=True)
    parser.add_argument("--input_output_dir", type=str, help="Directory with input files.", required=True)
    parser.add_argument("--output_file", type=str, help="Filename for saving virtual file.", required=True)
    parser.add_argument("--dhours_rel", type=int, default=1, help="dhours of the output dataset, relative to the input dataset. A value of 1 means the datasets are simply being concatenated, while a value of n > 1 means that every nth sample is taken from the input datasets.")
    args = parser.parse_args()

    main(args)
