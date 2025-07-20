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


def transfer_channels(input_file: str, output_file: str, channels: List[str],
                      batch_size: Optional[int]=32, entry_key: Optional[str]='fields',
                      verbose: Optional[bool]=False):
    
    """Function to transfer channels from one input file to the other. 

    This function reads corresponding channels from the input_file and copies them to the corresponding slot
    in the output file.

    This routine supports distributed processing via mpi4py.
    ...

    Parameters
    ----------
    input_file : str
        Input file in makani HDF5 format. Must be fully annotated.
    output_file : str
        Output file in makani HDF5 format. Must be fully annotated.
    channels : List[str]
        Channels to transfer.
    batch_size : int
        Batch size in which the samples are processed. This does not have any effect on the statistics (besides small numerical changes because of order of operations), but
        is merely a performance setting. Bigger batches are more efficient but require more memory.
    entry_key: str
        This is the HDF5 dataset name of the data in the files. Defaults to "fields".  
    verbose : bool
        Enable for more printing.
    """

    # get comm ranks and size
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # timer
    start_time = time.perf_counter()

    # check total number of entries:
    num_entries_total_in = 0
    num_entries_total_out = 0
    channels_in = None
    channels_out = None
    if comm_rank == 0:
        with h5.File(input_file, 'r') as f:
            num_entries_total_in = f[entry_key].shape[0]
            channels_in = [x.decode() for x in f["channel"][...].tolist()]

        with h5.File(output_file, 'r') as f:
            num_entries_total_out = f[entry_key].shape[0]
            channels_out = [x.decode() for x in f["channel"][...].tolist()]

    num_entries_total_in = comm.bcast(num_entries_total_in, root=0)
    num_entries_total_out = comm.bcast(num_entries_total_out, root=0)
    channels_in = comm.bcast(channels_in)
    channels_out = comm.bcast(channels_out)

    if num_entries_total_in != num_entries_total_out:
        raise IndexError(f"Files {input_file} and {output_file} are using a different number of samples ({num_entries_total_in} vs {num_entries_total_out}).")

    num_entries_total = num_entries_total_in
            
    # set up progressbar
    if comm_rank == 0:
        pbar = progressbar.ProgressBar(maxval=num_entries_total)
        pbar.update(0)

    # get offsets
    num_entries_local = (num_entries_total + comm_size - 1) // comm_size
    num_entries_start = num_entries_local * comm_rank
    num_entries_end = min(num_entries_start + num_entries_local, num_entries_total)
    num_entries_list = list(range(num_entries_start, num_entries_end))

    if comm_rank == 0:
        print(f"Transferring {channels} from {input_file} to {output_file}.")

    # open files
    fin = h5.File(input_file, "r", driver="mpio", comm=comm)
    fout = h5.File(output_file, "a", driver="mpio", comm=comm)
    
    # do loop over channels
    num_entries_current = 0
    for idc, channel in enumerate(channels):

        # find channel index in files
        cidx_in = channels_in.index(channel)
        cidx_out = channels_out.index(channel)
        
        # populate fields
        for entries in batched(num_entries_list, batch_size):
            entries = list(entries)

            data = fin[entry_key][entries, cidx_in, ...]
            fout[entry_key][entries, cidx_out, ...] = data[...]

            # update progressbar
            num_entries_current += len(entries) * comm_size
            if comm_rank == 0:
                num_entries_current = min(num_entries_current, num_entries_total)
                pbar.update(num_entries_current)

    # we need to wait here
    comm.Barrier()
    
    # close file
    fin.close()
    fout.close()
        
    # end time
    end_time = time.perf_counter()
    run_time = str(dt.timedelta(seconds=end_time-start_time))

    if comm_rank == 0:
        pbar.finish()
        print(f"All done. Run time {run_time}.")

    comm.Barrier()

    return


def main(args):
    # concatenate files with timestamp information
    transfer_channels(input_file=args.input_file,
                      output_file=args.output_file,
                      channels=args.channels,
                      batch_size=args.batch_size,
                      verbose=args.verbose)


if __name__ == '__main__':

    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="makani input file", required=True)
    parser.add_argument("--output_file", type=str, help="File to which channels will be written", required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for writing chunks")
    parser.add_argument("--channels", default=[], nargs="+", type=str, help="Channels to be copied from input to output. Must be specified as a list.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    main(args)
