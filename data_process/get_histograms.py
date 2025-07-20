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

import sys
import os
from typing import Optional
import time
import pickle
import numpy as np
import h5py as h5
from math import ceil
import argparse as ap
from itertools import groupby, accumulate
import operator
from bisect import bisect_right
from tqdm import tqdm

# MPI
from mpi4py import MPI
from mpi4py.util import dtlib

# we need the parser
import json

# we need that for quadrature
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from makani.utils.grids import GridQuadrature


def allgather_safe(comm, obj):
    
    # serialize the stuff
    fdata = pickle.dumps(obj, protocol = pickle.HIGHEST_PROTOCOL)
    
    #total size
    comm_size = comm.Get_size()
    num_bytes = len(fdata)
    total_bytes = num_bytes * comm_size

    #chunk by ~1GB:
    gigabyte = 1024*1024*1024
    
    # determine number of chunks
    num_chunks = (total_bytes + gigabyte - 1) // gigabyte
    
    # determine local chunksize
    chunksize = (num_bytes + num_chunks - 1) // num_chunks
    
    # datatype stuff
    datatype = MPI.BYTE
    np_dtype = dtlib.to_numpy_dtype(datatype)
    
    # gather stuff
    # prepare buffers:
    sendbuff = np.frombuffer(memoryview(fdata), dtype=np_dtype, count=num_bytes)
    recvbuff = np.empty((comm_size * chunksize), dtype=np_dtype)
    resultbuffs = np.split(np.empty(num_bytes * comm_size, dtype=np_dtype), comm_size)
    
    # do subsequent gathers
    for i in range(0, num_chunks):
        # create buffer views
        start = i * chunksize
        end = min(start + chunksize, num_bytes)
        eff_bytes = end - start
        sendbuffv = sendbuff[start:end]
        recvbuffv = recvbuff[0:eff_bytes*comm_size]
        
        # perform allgather on views
        comm.Allgather([sendbuffv, datatype], [recvbuffv, datatype])
        
        # split result buffer for easier processing
        recvbuff_split = np.split(recvbuffv, comm_size)
        for j in range(comm_size):
            resultbuffs[j][start:end] = recvbuff_split[j][...]
    results = [x.tobytes() for x in resultbuffs]

    # unpickle:
    results = [pickle.loads(x) for x in results]
    
    return results
            

def get_file_stats(filename,
                   file_slice,
                   bias=None,
                   norm=None,
                   batch_size=8,
                   progress=None):

    count = 0
    mins = []
    maxs = []
    with h5.File(filename, 'r') as f:

        # create batch
        slc_start = file_slice.start
        slc_stop = file_slice.stop
        for batch_start in range(slc_start, slc_stop, batch_size):
            batch_stop = min(batch_start+batch_size, slc_stop)
            sub_slc = slice(batch_start, batch_stop)
            
            data = f['fields'][sub_slc, ...]

            if bias is not None:
                data = data - bias

            if norm is not None:
                data = data / norm

            # counts
            count += data.shape[0] * data.shape[2] * data.shape[3]
            
            # min/max
            mins.append(np.min(data, axis=(0,2,3)))
            maxs.append(np.max(data, axis=(0,2,3)))

            if progress is not None:
                progress.update(batch_stop-batch_start)
    
    # concat and take min/max
    mins = np.min(np.stack(mins, axis=1), axis=1)
    maxs = np.max(np.stack(maxs, axis=1), axis=1)

    return count, mins, maxs


def get_file_histograms(filename, file_slice,
                        minvals, maxvals, nbins,
                        quadrature_weights,
                        bias=None,
                        norm=None,
                        batch_size=8,
                        progress=None):

    histograms = None
    with h5.File(filename, 'r') as f:

        # create batch
        slc_start = file_slice.start
        slc_stop = file_slice.stop
        for batch_start in range(slc_start, slc_stop, batch_size):
            batch_stop = min(batch_start+batch_size, slc_stop)
            sub_slc = slice(batch_start, batch_stop)
            
            data = f['fields'][sub_slc, ...]
            
            if bias is not None:
                data = data - bias

            if norm is not None:
                data = data / norm
            
            # get histograms along channel axis
            datalist = np.split(data, data.shape[1], axis=1)

            # tile the weights
            quadrature_weights = np.tile(quadrature_weights, (batch_stop-batch_start, 1, 1, 1))
            
            print(datalist[0].shape, quadrature_weights.shape)
            
            # generate histograms
            tmphistograms = [np.histogram(x, bins=nbins, range=(minval, maxval), weights=quadrature_weights)
                             for x,minval,maxval in zip(datalist, minvals, maxvals)]
            
            if histograms is None:
                histograms = tmphistograms
            else:
                histograms = [(x[0]+y[0], x[1]) for x,y in zip(histograms, tmphistograms)]

            if progress is not None:
                progress.update(batch_stop-batch_start)
                
    return histograms


def get_wind_channels(channel_names):
    # find the pairs in the channel names and alter the stats accordingly
    channel_dict = { channel_names[ch] : ch for ch in set(range(len(channel_names)))}

    uchannels = []
    vchannels = []
    for chn, ch in channel_dict.items():
        if chn[0] == 'u':
            vchn = 'v' + chn[1:]
            if vchn in channel_dict.keys():
                vch = channel_dict[vchn]

                uchannels.append(ch)
                vchannels.append(vch)
    
    return uchannels, vchannels


def get_histograms(input_dir: str, output_dir: str, stats_dir: str, metadata_file: str,
                   quadrature_rule: str, nbins: Optional[int]=100, batch_size: Optional[int]=16):

    """Function to compute histograms for all variables of a makani HDF5 dataset. 

    This function reads data from input_path and computes histograms based on number of bins specified.
    The results are stored in histograms.h5 located in the output_dir.

    All histograms are weighted by spherical quadrature weights.

    This routine supports distributed processing via mpi4py.
    ...

    Parameters
    ----------
    input_path : str
        Path which hosts the HDF5 files to compute the statistics on. Note, this routine supports virtual datasets genrated using concatenate_dataset.py.
        If you want to use a concatenated dataset, please specify the full path including the filename, e.g. <path-to-data>/train.h5v. In this case,
        the routine will ignore all the other files in the same folder.
    output_path : str
        Output path to specify where to store the computed statistics.
    stats_dir : str
        Path which contains the stats computed by get_stats.py.
    metadata_file : str
        name of the file to read metadata from. The metadata is a json file, and after reading it should be a
        dictionary containing metadata describing the dataset. Most important entries are:
        dhours: distance between subsequent samples in hours
        coords: this is a dictionary which contains two lists, latitude and longitude coordinates in degrees as well as channel names.
        Example: coords = dict(lat=[-90.0, ..., 90.], lon=[0, ..., 360], channel=["t2m", "u500", "v500", ...])
        Note that the number of entries in coords["lat"] has to match dimension -2 of the dataset, and coords["lon"] dimension -1.
        The length of the channel names has to match dimension -3 (or dimension 1, which is the same) of the dataset. 
    quadrature_rule : str
        Which spherical quadrature rule to use for the spatial averages. Supported are "naive", "clenshaw-curtiss" and "gauss-legendre".
    nbins : int
        Number of bins used for the histogram. If a number < 1 is specified, the number of bins is automatically computed based on the data.
    batch_size : int
        Batch size in which the samples are processed. This does not have any effect on the statistics (besides small numerical changes because of order of operations), but
        is merely a performance setting. Bigger batches are more efficient but require more memory.
    """

    # get comm
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # get files
    filelist = None
    data_shape = None
    num_samples = None
    wind_channels = None
    norm = None
    bias = None
    if comm_rank == 0:
        filelist = sorted([os.path.join(input_dir, x) for x in os.listdir(input_dir)])
        if not filelist:
            raise FileNotFoundError(f"Error, directory {input_dir} is empty.")

        # open the first file to check for stats
        num_samples = []
        for filename in filelist:
            with h5.File(filename, 'r') as f:
                data_shape = f['fields'].shape
                num_samples.append(data_shape[0])

        # open metadata file
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        channel_names = metadata['coords']['channel']
        wind_channels = get_wind_channels(channel_names)

        if stats_dir is not None:
            norm = np.load(os.path.join(stats_dir, "global_stds.npy"))
            bias = np.load(os.path.join(stats_dir, "global_means.npy"))

    # communicate the files
    filelist = comm.bcast(filelist, root=0)
    num_samples = comm.bcast(num_samples, root=0)
    data_shape = comm.bcast(data_shape, root=0)
    wind_channels = comm.bcast(wind_channels, root=0)
    if norm is not None:
        norm = comm.bcast(norm, root=0)
    if bias is not None:
        bias = comm.bcast(bias, root=0)

    # DEBUG
    filelist = filelist[:2]
    num_samples = num_samples[:2]
    # DEBUG
    
    # get file offsets
    num_samples_total = sum(num_samples)
    num_channels = data_shape[1]
    height, width = (data_shape[2], data_shape[3])

    # quadrature:
    quadrature_weights = GridQuadrature(quadrature_rule, (height, width),
                                        crop_shape=None, crop_offset=(0, 0),
                                        normalize=True, pole_mask=None).quad_weight.cpu().numpy()

    if comm_rank == 0:
        print(f"Found {len(filelist)} files with a total of {num_samples_total} samples. Each sample has the shape {num_channels}x{height}x{width} (CxHxW).")
    
    # do the sharding:
    num_samples_chunk = (num_samples_total + comm_size - 1) // comm_size
    samples_start = num_samples_chunk * comm_rank
    samples_end = min([samples_start + num_samples_chunk, num_samples_total])
    sample_offsets = list(accumulate(num_samples, operator.add))[:-1]
    sample_offsets.insert(0, 0)

    if comm_rank == 0:
        print("Loading data with the following chunking:")
    for rank in range(comm_size):
        if comm_rank == rank:
            print("Rank = ", comm_rank, " samples start = ", samples_start, " samples end = ", samples_end, flush=True)
        comm.Barrier()

    # convert list of indices to files and ranges in files:
    mapping = {}
    for idx in range(samples_start, samples_end):
	# compute indices
        file_idx = bisect_right(sample_offsets, idx) - 1
        local_idx = idx - sample_offsets[file_idx]

        # lookup
        filename = filelist[file_idx]
        if filename in mapping:
            # update upper and lower bounds
            mapping[filename] = ( min(local_idx, mapping[filename][0]),
                                  max(local_idx, mapping[filename][1]) )
        else:
            mapping[filename] = (local_idx, local_idx)

    # offsets has order like:
    #[0, 1460, 2920, ...]
    
    # compute local stats
    if comm_rank == 0:
        progress = tqdm(desc="Preprocessing bounds", total=num_samples_local)
    else:
        progress = None
    start = time.time()
    mins = []
    maxs = []
    count = 0
    for filename, index_bounds in mapping.items():
        tmpcount, tmpmins, tmpmaxs = get_file_stats(filename,
                                                    file_slice=slice(index_bounds[0], index_bounds[1]+1),
                                                    batch_size=batch_size,
                                                    bias=bias,
                                                    norm=norm,
                                                    progress=progress)
        mins.append(tmpmins)
        maxs.append(tmpmaxs)
        count += tmpcount
    mins = np.min(np.stack(mins, axis=1), axis=1)
    maxs = np.max(np.stack(maxs, axis=1), axis=1)
    duration = time.time() - start
    if comm_rank == 0:
        progress.close()
        
    # wait for everybody else
    print(f"Rank {comm_rank} stats done. Duration for {(samples_end - samples_start)} samples: {duration:.2f}s", flush=True)
    comm.Barrier()

    # now gather the stats from all nodes: we need to do that safely
    countlist = allgather_safe(comm, count)
    minmaxlist = allgather_safe(comm, [mins, maxs])

    # compute global min and max and count
    count = sum(countlist)
    mins = np.min(np.stack([x[0] for x in minmaxlist], axis=1), axis=1).tolist()
    maxs = np.max(np.stack([x[1] for x in minmaxlist], axis=1), axis=1).tolist()
    
    if comm_rank == 0:
        print(f"Data range overview on {count} datapoints:")
        for c,mi,ma in zip(channel_names, mins, maxs):
            print(f"{c}: min = {mi}, max = {ma}")

    # set nbins to sqrt(count) if smaller than one
    if nbins <= 0:
        nbins = np.sqrt(count)
    else:
        nbins = nbins
            
    # wait for rank 0 to finish
    comm.Barrier()

    # now create histograms:
    if comm_rank == 0:
        progress = tqdm(desc="Computing histograms", total=num_samples_local)
    else:
        progress = None
    start = time.time()
    histograms = None
    for filename, index_bounds in mapping.items():
        tmphistograms = get_file_histograms(filename,
                                            file_slice=slice(index_bounds[0], index_bounds[1]+1),
                                            minvals=mins,
                                            maxvals=maxs,
                                            nbins=nbins,
                                            quadrature_weights=quadrature_weights,
                                            bias=bias,
                                            norm=norm,
                                            batch_size=batch_size,
                                            progress=progress)
        
        if histograms is None:
            histograms = tmphistograms
        else:
            histograms = [(x[0]+y[0], x[1]) for x,y in zip(histograms, tmphistograms)]
            
    duration = time.time() - start

    # wait for everybody else
    print(f"Rank {comm_rank} histograms done. Duration for {(samples_end - samples_start)} samples: {duration:.2f}s", flush=True)
    comm.Barrier()
    
    # now gather the stats from all nodes: we need to do that safely
    histogramlist = allgather_safe(comm, histograms)
    
    # combine the results
    histograms = histogramlist[0]
    for tmphistograms in histogramlist[1:]:
        histograms = [(x[0]+y[0], x[1]) for x,y in zip(histograms, tmphistograms)]
    
    if comm_rank == 0:
        edges = np.stack([e for _,e in histograms], axis=0)
        data = np.stack([h for h,_ in histograms], axis=0)

        outfilename = os.path.join(output_dir, "histograms.h5")
        with h5.File(outfilename, "w") as f:
            f["edges"] = edges
            f["data"] = data

    # wait for everybody to finish
    comm.Barrier()


def main(args):
    get_histograms(input_dir=args.input_dir,
                   output_dir=args.output_dir,
                   stats_dir=args.stats_dir,
                   metadata_file=args.metadata_file,
                   quadrature_rule=args.quadrature_rule,
                   nbins=args.nbins,
                   batch_size=args.batch_size)

    return
    

if __name__ == "__main__":
    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory with input files.", required=True)
    parser.add_argument("--metadata_file", type=str, help="File containing dataset metadata.", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory for saving stats files.", required=True)
    parser.add_argument("--stats_dir", type=str, default=None, help="Directory with stats normalization files.")
    parser.add_argument("--quadrature_rule", type=str, default="naive", choices=["naive", "clenshaw-curtiss", "gauss-legendre"], help="Specify quadrature_rule for spatial averages.")
    parser.add_argument("--nbins", type=int, default=100, help="Number of bins for histograms")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size used for reading chunks from a file at a time to avoid OOM errors.")
    args = parser.parse_args()
    
    main(args)




