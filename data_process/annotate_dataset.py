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


def annotate(metadata: dict, file_names_to_annotate: List[str], years: List[int], entry_key: Optional[str]="fields"):
    """Function to annotate the dimensions of an existing makani compatible HDF5 dataset.

    Modifies the files in-place. If a file is already annotated, the annotation is skipped.

    ...

    Parameters
    ----------
    metadata : dict
        dictionary containing metadata describing the dataset. Most important entries are:
        dhours: distance between subsequent samples in hours
        coords: this is a dictionary which contains two lists, latitude and longitude coordinates in degrees as well as channel names.
        Example: coords = dict(lat=[-90.0, ..., 90.], lon=[0, ..., 360], channel=["t2m", "u500", "v500", ...])
        Note that the number of entries in coords["lat"] has to match dimension -2 of the dataset, and coords["lon"] dimension -1.
        The length of the channel names has to match dimension -3 (or dimension 1, which is the same) of the dataset.
    file_names_to_annotate : List[str]
        List of filenames to annotate. Has to be the same length as years.
    years : int
        List of years, one for each file. For example if the kth file in file_names_to_annotate stores data from year 1990, then
        years[k] = 1990. The datestampts for each entry in the files is computed based on this information and the dhours stamp in the metadata
        dictionary.
    entry_key: str
        This is the HDF5 dataset name of the data in the files. Defaults to "fields".
    """

    # get dhours
    dhours = metadata["dhours"]

    # get lon and lat coords
    latitudes = np.array(metadata["coords"]["lat"], dtype=np.float32)
    longitudes = np.array(metadata["coords"]["lon"], dtype=np.float32)

    # get channel names
    channel_names = metadata["coords"]["channel"]
    chanlen = max([len(v) for v in channel_names])

    print( f"Annotating files: {file_names_to_annotate}")

    for filename, year in zip(file_names_to_annotate, years):

        # get year offset
        year_start = dt.datetime(year=year, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc).timestamp()

        with h5.File(filename, 'a', libver='latest') as f:
            # save timestamps first
            num_samples = f[entry_key].shape[0]
            timestamps = year_start + np.arange(0, num_samples * dhours * 3600, dhours * 3600, dtype=np.float64)

            # create datasets for scales
            try:
                f.create_dataset("timestamp", data=timestamps)
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
            except ValueError as err:
                print(f"Could not annotate {filename}. Reason: {err}")
                continue

    print("All done.")

    return


def main(args):
    # get files
    files = glob(os.path.join(args.dataset_dir, "*.h5"))

    # make sure that years are consecutive
    years = sorted([int(os.path.splitext(os.path.basename(pname))[0]) for pname in files])

    # create sorted and curated file list
    files = [os.path.join(args.dataset_dir, str(y) + ".h5") for y in years]

    # load metadata:
    with open(args.dataset_metadata, "r") as f:
        metadata = json.load(f)

    # concatenate files with timestamp information
    annotate(metadata, files, years)


if __name__ == '__main__':

    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--dataset_metadata", type=str, help="Input file containing metadata.", required=True)
    parser.add_argument("--dataset_dir", type=str, help="Directory with files to annotate.", required=True)
    args = parser.parse_args()

    main(args)
