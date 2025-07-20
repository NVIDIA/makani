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

from typings import Optional
import re
import argparse as ap
import xarray as xr
import h5py as h5
import json
import datetime as dt
import numpy as np
from tqdm import tqdm

from mpi4py import MPI

from makani.utils.dataloaders.data_helpers import get_timestamp
from makani.utils.features import get_channel_groups
from wb2_helpers import split_convert_channel_names


def generate_wb2_climatology(metadata_file: str, input_climatology: str, mask_output_file: str, climatology_output_file: str,
                             verbose: Optional[bool]=False):
    
    """Function to generate a ground profile mask and climatology compatible with Weatherbench 2.

    Weatherbench 2 uses a climatology computed on sliding windows. Additionally, it creates a mask labeling points where the
    geopotential is lower than the actual elevation of the ground as invalid.
    This function can extract that masks from the corresponding WB2 climatology file stored on GCS and brings it into a format which can be used in
    makani.

    This routine supports distributed processing via mpi4py.

    ...

    Parameters
    ----------
    metadata_file : str
        name of the file to read metadata from. The metadata is a json file, and after reading it should be a
        dictionary containing metadata describing the dataset. Most important entries are:
        dhours: distance between subsequent samples in hours
        coords: this is a dictionary which contains two lists, latitude and longitude coordinates in degrees as well as channel names.
        Example: coords = dict(lat=[-90.0, ..., 90.], lon=[0, ..., 360], channel=["t2m", "u500", "v500", ...])
        Note that the number of entries in coords["lat"] has to match dimension -2 of the dataset, and coords["lon"] dimension -1.
        The length of the channel names has to match dimension -3 (or dimension 1, which is the same) of the dataset. 
    input_climatology : str
        Fully qualified name of the gcs location of the climatology input file. For example:
        gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr
    mask_output_file : str
        This is the name of the file the mask data will be written to in HDF5 format.
    climatology_output_file : str
        Name of the file the climatology data will be written to. Only channels present from the metadatas channels list will be
        used and written in the correct order. If a channel which is present in the metadata file is not found in the climatology dataset,
        default values for that channel will be written. The translation of channel names between Weatherbench 2 and Makani convention is automatically
        performed by the code. The details for this can be found in wb2_helpers.py
    verbose : bool
        Flag to print verbose output.
    """

    # get comm ranks and size
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    metadata = None
    if comm_rank == 0:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    metadata = comm.bcast(metadata, root=0)

    # read channel names
    channel_names = metadata['coords']['channel']

    # split in surface and atmospheric channels
    atmospheric_channel_names, atmospheric_channel_names_wb2, surface_channel_names, surface_channel_names_wb2, atmospheric_levels = split_convert_channel_names(channel_names)
    
    # open zarr file and load the above_ground mask:
    clim = xr.open_zarr(input_climatology)

    # above ground data, only relevant levels
    above_ground = clim["above_ground"]

    if comm_rank == 0:
        print("Performing sanity checks")
    # make sure levels are present
    mask_levels_list = above_ground.level.values.tolist()
    for level in atmospheric_levels:
        if not level in mask_levels_list:
            raise RuntimeError(f"Error, level {level} is not in the climatology levels. Available levels {mask_levels_list}")
    mask_levels = above_ground.loc[:, :, atmospheric_levels, :, :]

    # check if the grid matches
    metadata_lat = metadata["coords"]["lat"]
    metadata_lon = metadata["coords"]["lon"]
    mask_lat = mask_levels.latitude.values
    mask_lon = mask_levels.longitude.values

    if not np.allclose(metadata_lat, mask_lat):
        raise RuntimeError(f"Error, the latitudes from the metadata file and from the climatology do not match. Climatology grid: {mask_lat}, data grid: {metadata_lat}")

    if not np.allclose(metadata_lon, mask_lon):
        raise RuntimeError(f"Error, the longitudes from the metadata file and from the climatology do not match. Climatology grid: {mask_lon}, data grid: {metadata_lon}")
    
    # reshape mask array and prepare input
    T,D,L,H,W = mask_levels.shape
    DT = D*T

    # split time steps by rank
    Dloc = (D + comm_size - 1) // comm_size
    Dstart = Dloc * comm_rank
    Dend = min(Dstart + Dloc, D)
    # update Dloc
    Dloc = Dend - Dstart
    DTstart = Dstart * T
    DTend = Dend * T

    # datasets for dimension annotation
    # get list of timestamps
    days = clim["dayofyear"].values - 1
    hours = clim["hour"].values
    timestamps = []
    for hour in hours.tolist():
        timestamps.append(days * 24 + hour)
    timestamps = np.stack(timestamps, axis=1).flatten()
    timestamps = np.array([dt.timedelta(hours=t).total_seconds() for t in timestamps.tolist()], dtype=np.float64)
    # channel names
    chanlen = max([len(v) for v in channel_names])
    

    if verbose:
        print(f"{comm_rank}: file={input_climatology}, DT={D*T}, DTstart={DTstart}, DTend={DTend}, Dstart={Dstart}, Dend={Dend}, Dloc={Dloc}")
    
    # store mask data
    if mask_output_file is not None:

        if comm_rank == 0:
            print("Storing masks")
        
        with h5.File(mask_output_file, "w", driver="mpio", comm=comm) as f:

            # create dataset
            dset = f.create_dataset("fields", [DT, len(channel_names), H, W], dtype="f4")

            # initialize everything with 1 so that all surface variables have mask 1
            dset[DTstart:DTend, ...] = 1.0

            # create dimension scales
            # datasets
            f.create_dataset("timestamp", data=timestamps)
            f.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
            f["channel"][...] = channel_names
            f.create_dataset("lat", data=metadata_lat)
            f.create_dataset("lon", data=metadata_lon)
            # scales
            f["timestamp"].make_scale("timestamp")
            f["channel"].make_scale("channel")
            f["lat"].make_scale("lat")
            f["lon"].make_scale("lon")
            # label
            f["fields"].dims[0].label = "Time offset in seconds relative to start of year"
            f["fields"].dims[1].label = "Channel name"
            f["fields"].dims[2].label = "Latitude in degrees"
            f["fields"].dims[3].label = "Longitude in degrees"
            # attach
            f["fields"].dims[0].attach_scale(f["timestamp"])
            f["fields"].dims[1].attach_scale(f["channel"])
            f["fields"].dims[2].attach_scale(f["lat"])
            f["fields"].dims[3].attach_scale(f["lon"])
            
            # now iterate over all levels and then write the corresponding variables
            for alevel in tqdm(atmospheric_levels):
                data = mask_levels.sel(level=alevel).isel(dayofyear=slice(Dstart,Dend)).values
                # transpose and reshape
                data = data.transpose(1,0,2,3).reshape(Dloc*T,H,W)
                for prefix in atmospheric_channel_names:
                    varname = prefix + str(alevel)
                    cidx = channel_names.index(varname)
                    dset[DTstart:DTend, cidx, ...] = data[...]
    else:
        if comm_rank == 0:
            print(f"No mask file specified, skipping")

    # store climatology
    if climatology_output_file is not None:
        
        if comm_rank == 0:
            print("Storing climatology")
            
        with h5.File(climatology_output_file, "w", driver="mpio", comm=comm) as f:
        
            # create dataset
            dset = f.create_dataset("fields", [DT, len(channel_names), H, W], dtype="f4")

            # initialize with zero to be safe
            dset[DTstart:DTend, ...] = 0.0

            # create dimension scales
            # datasets
            f.create_dataset("timestamp", data=timestamps)
            f.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
            f["channel"][...] = channel_names
            f.create_dataset("lat", data=metadata_lat)
            f.create_dataset("lon", data=metadata_lon)
            # scales
            f["timestamp"].make_scale("timestamp")
            f["channel"].make_scale("channel")
            f["lat"].make_scale("lat")
            f["lon"].make_scale("lon")
            # label
            f["fields"].dims[0].label = "Time offset in seconds relative to start of year"
            f["fields"].dims[1].label = "Channel name"
            f["fields"].dims[2].label = "Latitude in degrees"
            f["fields"].dims[3].label = "Longitude in degrees"
            # attach
            f["fields"].dims[0].attach_scale(f["timestamp"])
            f["fields"].dims[1].attach_scale(f["channel"])
            f["fields"].dims[2].attach_scale(f["lat"])
            f["fields"].dims[3].attach_scale(f["lon"])

            # surface channels
            for sc,scwb2 in zip(surface_channel_names,surface_channel_names_wb2):
                cidx = channel_names.index(sc)
                if scwb2 not in clim:
                    if comm_rank == 0:
                        print(f"Key {scwb2} not found in dataset, skipping")
                    continue
                data = clim[scwb2].isel(dayofyear=slice(Dstart,Dend)).values
                # transpose and reshape
                data = data.transpose(1,0,2,3).reshape(Dloc*T,H,W)
                dset[DTstart:DTend, cidx, ...] = data[...]

            # create atmospheric channels
            for ac, acwb2 in zip(atmospheric_channel_names, atmospheric_channel_names_wb2):
                for idl, alevel in enumerate(atmospheric_levels):
                    cidx = channel_names.index(ac + str(alevel))
                    if acwb2 not in clim:
                        if comm_rank == 0:
                            print(f"Key {acwb2} not found in dataset, skipping")
                        continue
                    data = clim[acwb2].sel(level=alevel).isel(dayofyear=slice(Dstart,Dend)).values
                    # transpose and reshape
                    data = data.transpose(1,0,2,3).reshape(Dloc*T,H,W)
                    dset[DTstart:DTend, cidx, ...] = data[...]
    else:
        if comm_rank == 0:
            print(f"No climatology file specified, skipping")
                    

    # wait for everybody to finish
    comm.Barrier()
    
    # close xarray
    clim.close()

    print("All done")


def main(args):
    generate_wb2_climatology(args.metadata_file, args.input_climatology, args.mask_output_file, args.climatology_output_file, verbose=args.verbose))

    return


if __name__ == "__main__":
    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_climatology", type=str, default="gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr",
                        help="Input climatology in zarr format.")
    parser.add_argument("--metadata_file", type=str, help="File containing dataset metadata used during training.", required=True)
    parser.add_argument("--mask_output_file", type=str, default=None, help="Filename for the mask file in HDF5 format, including full path.")
    parser.add_argument("--climatology_output_file", type=str, default=None, help="Filename for the climatology file in HDF5 format, including full path.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
