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
from typing import Optional
import sys
import time
import pickle
import json
import numpy as np
import h5py as h5
import math
import argparse as ap
from itertools import groupby, accumulate
import operator
from bisect import bisect_right
from glob import glob
from tqdm import tqdm

# MPI
from mpi4py import MPI
from mpi4py.util import dtlib

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


def welford_combine(stats1, stats2):
    # update time means first:
    stats = {}

    for k in stats1.keys():
        s_a = stats1[k]
        s_b = stats2[k]

        # update stats
        n_a = s_a["counts"]
        n_b = s_b["counts"]
        n_ab = n_a + n_b

        if s_a["type"] == "min":
            if n_a == 0:
                values = s_b["values"]
            elif n_b == 0:
                values = s_a["values"]
            else:
                values = np.minimum(s_a["values"], s_b["values"])
        elif s_a["type"] == "max":
            if n_a == 0:
                values = s_b["values"]
            elif n_b ==	0:
                values = s_a["values"]
            else:
                values = np.maximum(s_a["values"], s_b["values"])
        elif s_a["type"] == "mean":
            mean_a = s_a["values"]
            mean_b = s_b["values"]
            values = (mean_a * float(n_a) + mean_b * float(n_b)) / float(n_ab)
        elif s_a["type"] == "meanvar":
            mean_a = s_a["values"][0]
            mean_b = s_b["values"][0]
            m2_a = s_a["values"][1]
            m2_b = s_b["values"][1]
            delta = mean_b - mean_a

            values = [(mean_a * float(n_a) + mean_b * float(n_b)) / float(n_ab),
                      m2_a + m2_b + delta * delta * float(n_a * n_b) / float(n_ab)]

        stats[k] = {"counts": n_ab,
                    "type": s_a["type"],
                    "values": values}

    return stats


def get_file_stats(filename,
                   file_slice,
                   wind_indices,
                   quadrature,
                   dt=1,
                   batch_size=16,
                   progress=None):

    stats = None
    with h5.File(filename, 'r') as f:

        # get dataset
        dset= f['fields']

        # create batch
        slc_start = file_slice.start
        slc_stop = file_slice.stop
        if slc_stop is None:
            slc_stop = dset.shape[0]

        if batch_size is None:
            batch_size = slc_stop - slc_start
        
        for batch_start in range(slc_start, slc_stop, batch_size):
            batch_stop = min(batch_start+batch_size, slc_stop)
            sub_slc = slice(batch_start, batch_stop)

            # get slice
            data = dset[sub_slc, ...].astype(np.float64)

            # define counts
            counts_time = data.shape[0]
            counts_planar = counts_time * data.shape[2] * data.shape[3]

            # compute mean and variance
            tdata = torch.from_numpy(data)
            tmean = torch.mean(quadrature(tdata), keepdims=False, dim=0).reshape(1, -1, 1, 1)
            tvar = torch.mean(quadrature(torch.square(tdata - tmean)), keepdims=False, dim=0).reshape(1, -1, 1, 1)

            # time diffs: read one more sample for these, if possible
            # TODO: tile it for dt < batch_size
            if batch_start >= dt:
                sub_slc_m_dt = slice(batch_start-dt, batch_stop)
                data_m_dt = dset[sub_slc_m_dt, ...].astype(np.float64)
                tdata_m_dt = torch.from_numpy(data_m_dt)
                tdiff = tdata_m_dt[dt:, ...] - tdata_m_dt[:-dt, ...]
                counts_timediff = tdiff.shape[0]
                tdiffmean = torch.mean(quadrature(tdiff), keepdims=False, dim=0).reshape(1, -1, 1, 1)
                tdiffvar = torch.mean(quadrature(torch.square(tdiff - tdiffmean)), keepdims=False, dim=0).reshape(1, -1, 1, 1)
            else:
                # skip those for tdiff
                counts_timediff = 0

            # fill the dict
            tmpstats = dict(maxs = {"values": np.max(data, keepdims=True, axis = (0, 2, 3)),
                                    "type": "max",
                                    "counts": counts_planar},
                            mins = {"values": np.min(data, keepdims=True, axis = (0, 2, 3)),
                                    "type": "min",
                                    "counts": counts_planar},
                            time_means = {"values": np.mean(data, keepdims=True, axis = 0),
                                          "type": "mean",
                                          "counts": counts_time},
                            global_meanvar = {"values": [tmean.numpy(), float(counts_time) * tvar.numpy()],
                                              "type": "meanvar",
                                              "counts": counts_time})
            if counts_timediff != 0:
                tmpstats["time_diff_meanvar"] = {"values": [tdiffmean.numpy(),
                                                            float(counts_timediff) * tdiffvar.numpy()],
                                                 "type": "meanvar",
                                                 "counts": counts_timediff}
            else:
                # we need the shapes
                tshape = tmean.shape
                tmpstats["time_diff_meanvar"] = {"values": [np.zeros(tshape, dtype=np.float64), np.zeros(tshape, dtype=np.float64)], "type": "meanvar", "counts": 0}

            if wind_indices is not None:
                u_tens = tdata[:, wind_indices[0]]
                v_tens = tdata[:, wind_indices[1]]
                wind_magnitude = torch.sqrt(torch.square(u_tens) + torch.square(v_tens))
                wind_mean = torch.mean(quadrature(wind_magnitude), keepdims=False, dim=0).reshape(1, -1, 1, 1)
                wind_var = torch.mean(quadrature(torch.square(wind_magnitude - wind_mean)), keepdims=False, dim=0).reshape(1, -1, 1, 1)
                tmpstats["wind_meanvar"] = {"values": [wind_mean.numpy(), float(counts_time) * wind_var.numpy()],
                                            "type": "meanvar",
                                            "counts": counts_time}

                if counts_timediff != 0:
                    udiff_tens = tdiff[:, wind_indices[0]]
                    vdiff_tens = tdiff[:, wind_indices[1]]
                    winddiff_magnitude = torch.sqrt(torch.square(udiff_tens) + torch.square(vdiff_tens))
                    winddiff_mean = torch.mean(quadrature(winddiff_magnitude), keepdims=False, dim=0).reshape(1, -1, 1, 1)
                    winddiff_var = torch.mean(quadrature(torch.square(winddiff_magnitude - winddiff_mean)), keepdims=False, dim=0).reshape(1, -1, 1, 1)
                    tmpstats["winddiff_meanvar"] = {"values": [winddiff_mean.numpy(), float(counts_timediff) * winddiff_var.numpy()],
                                                    "type": "meanvar",
                                                    "counts": counts_timediff}
                else:
                    wdiffshape = wind_mean.shape
                    tmpstats["winddiff_meanvar"] = {"values": [np.zeros(wdiffshape, dtype=np.float64), np.zeros(wdiffshape, dtype=np.float64)],
                                                    "type": "meanvar",
                                                    "counts": 0}

            if stats is not None:
                stats = welford_combine(stats, tmpstats)
            else:
                stats = tmpstats

            if progress is not None:
                progress.update(batch_stop-batch_start)

    return stats

def get_wind_channels(channel_names):
    # find the pairs in the channel names and alter the stats accordingly
    u_variables = sorted([x for x in channel_names if x.startswith("u")])
    v_variables = sorted([x for x in channel_names if x.startswith("v")])

    # some sanity checks
    error = False
    if len(u_variables) != len(v_variables):
        error = True
    for u, v in zip(u_variables, v_variables):
        if u.replace("u", "") != v.replace("v", ""):
            error = True

    if error:
        raise ValueError("Error, cannot group wind channels together because not all pairs of wind channels are in the dataset.")

    # find the indices of the channels in the original channel names:
    uchannels = [channel_names.index(u) for u in u_variables]
    vchannels = [channel_names.index(v) for v in v_variables]

    return (uchannels, vchannels), (u_variables, v_variables)


def collective_reduce(comm, stats):
    statslist = allgather_safe(comm, stats)
    stats = statslist[0]
    for tmpstats in statslist[1:]:
        stats = welford_combine(stats, tmpstats)

    return stats


def binary_reduce(comm, stats):
    csize = comm.Get_size()
    crank = comm.Get_rank()

    # check for power of two
    assert((csize & (csize-1) == 0) and csize != 0)

    # how many steps do we need:
    nsteps = int(math.log(csize,2))

    # init step 1
    recv_ranks = range(0,csize,2)
    send_ranks = range(1,csize,2)

    for step in range(nsteps):
        for rrank,srank in zip(recv_ranks, send_ranks):
            if crank == rrank:
                rstats = comm.recv(source=srank, tag=srank)
                stats = welford_combine(stats, rstats)
            elif crank == srank:
                comm.send(stats, dest=rrank, tag=srank)

        # wait for everyone being ready before doing the next epoch
        comm.Barrier()

        # shrink the list
        if (step < nsteps-1):
            recv_ranks = recv_ranks[0::2]
            send_ranks = recv_ranks[1::2]

    return stats


def get_stats(input_path: str, output_path: str, metadata_file: str,
              dt: int, quadrature_rule: str, wind_angle_aware: bool,
              batch_size: Optional[int]=16, reduction_group_size: Optional[int]=8):

    """Function to compute various statistics of all variables of a makani HDF5 dataset. 

    This function reads data from input_path and computes minimum, maximum, mean and standard deviation
    for all variables in the dataset. This is done globally, meaning averaged over space and time.
    Those will be stored in files mins.npy, maxs.npy, global_means.npy and global_stds.npy respectively.
    Additionally, it creates a climatology, i.e. a temporal average of all spatial variables (no windowing).
    This data is stored in  time_means.npy.
    Finally, it computes the means and standard deviations for all variables for a fixed time difference dt.
    This data is stored in files time_diff_means_dt<chosen dt>.npy and time_diff_stds_dt<chosen dt>.npy respectively.

    All spatial averages are performed using spherical quadrature weights. The type of weights to be used can be specified by the user.

    This routine supports distributed processing via mpi4py. For numerically safe reductions, it uses parallel Welford variance computation.

    ...

    Parameters
    ----------
    input_path : str
        Path which hosts the HDF5 files to compute the statistics on. Note, this routine supports virtual datasets genrated using concatenate_dataset.py.
        If you want to use a concatenated dataset, please specify the full path including the filename, e.g. <path-to-data>/train.h5v. In this case,
        the routine will ignore all the other files in the same folder.
    output_path : str
        Output path to specify where to store the computed statistics.
    metadata_file : str
        name of the file to read metadata from. The metadata is a json file, and after reading it should be a
        dictionary containing metadata describing the dataset. Most important entries are:
        dhours: distance between subsequent samples in hours
        coords: this is a dictionary which contains two lists, latitude and longitude coordinates in degrees as well as channel names.
        Example: coords = dict(lat=[-90.0, ..., 90.], lon=[0, ..., 360], channel=["t2m", "u500", "v500", ...])
        Note that the number of entries in coords["lat"] has to match dimension -2 of the dataset, and coords["lon"] dimension -1.
        The length of the channel names has to match dimension -3 (or dimension 1, which is the same) of the dataset. 
    dt : int
        The temporal difference for which the temporal means and standard deviations should be computed. Note that this is in units of dhours (see metadata file),
    quadrature_rule : str
        Which spherical quadrature rule to use for the spatial averages. Supported are "naive", "clenshaw-curtiss" and "gauss-legendre".
    wind_angle_aware : bool
        If this flag is set to true, then wind channels will be grouped together (all u and v channels, e.g. u500 and v500, u10m and v10m, etc) and
        instead of computing stadard deviation component-wise, the standard deviation will be computed for the magnitude. This ensures that the direction of the
        wind vectors will not change when normalized by the standard deviation during training.
    batch_size : int
        Batch size in which the samples are processed. This does not have any effect on the statistics (besides small numerical changes because of order of operations), but
        is merely a performance setting. Bigger batches are more efficient but require more memory.
    reduction_group_size : int
        Reduction group size for the parallel Welford reduction. Th MPI world communicator is partitioned accordingly. Changing this value impacts performance but not numerical accuracy.
    """

    # disable gradients globally
    torch.set_grad_enabled(False)

    # get comm
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # create group comm
    group_id = comm_rank % reduction_group_size
    group_rank = comm_rank // reduction_group_size
    group_comm = comm.Split(color=group_id, key=group_rank)

    # create intergroup comm
    intergroup_comm = comm.Split(color=group_rank, key=group_id)

    # get files
    filelist = None
    data_shape = None
    num_samples = None
    wind_channels = None
    channel_names = None
    combined_file = None
    if comm_rank == 0:
        if os.path.isdir(input_path):
            combined_file = False
            filelist = sorted(glob(os.path.join(input_path, "*.h5")))
            if not filelist:
                raise FileNotFoundError(f"Error, directory {input_path} is empty.")

            # open the first file to check for stats
            num_samples = []
            for filename in filelist:
                with h5.File(filename, 'r') as f:
                    data_shape = f['fields'].shape
                    num_samples.append(data_shape[0])

        else:
            combined_file = True
            filelist = [input_path]
            with h5.File(filelist[0], 'r') as f:
                data_shape = f['fields'].shape
                num_samples = [data_shape[0]]

        # open metadata file
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # read channel names
        channel_names = metadata['coords']['channel']


    # communicate important information
    combined_file = comm.bcast(combined_file, root=0)
    channel_names = comm.bcast(channel_names, root=0)
    filelist = comm.bcast(filelist, root=0)
    num_samples = comm.bcast(num_samples, root=0)
    data_shape = comm.bcast(data_shape, root=0)

    # identify the wind channels
    if wind_angle_aware:
        wind_channels, wind_variables = get_wind_channels(channel_names)

    # get file offsets
    num_samples_total = sum(num_samples)
    num_channels = data_shape[1]
    height, width = (data_shape[2], data_shape[3])

    # quadrature:
    quadrature = GridQuadrature(quadrature_rule, (height, width),
                                crop_shape=None, crop_offset=(0, 0),
                                normalize=True, pole_mask=None)

    if comm_rank == 0:
        print(f"Found {len(filelist)} files with a total of {num_samples_total} samples. Each sample has the shape {num_channels}x{height}x{width} (CxHxW).")

    # do the sharding:
    num_samples_chunk = (num_samples_total + comm_size - 1) // comm_size
    samples_start = num_samples_chunk * comm_rank
    samples_end = min([samples_start + num_samples_chunk, num_samples_total])
    sample_offsets = list(accumulate(num_samples, operator.add))[:-1]
    sample_offsets.insert(0, 0)
    num_samples_local = samples_end - samples_start

    if comm_rank == 0:
        print("Loading data with the following chunking:")
    for	rank in	range(comm_size):
        if comm_rank ==	rank:
            print("Rank = ", comm_rank, " samples start = ", samples_start, " samples end = ", samples_end, flush=True)
        comm.Barrier()

    # convert list of indices to files and ranges in files:
    if combined_file:
        mapping = {filelist[0]: (samples_start, samples_end)}
    else:
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

    # initialize arrays
    stats = dict(global_meanvar = {"type": "meanvar", "counts": 0, "values": [np.zeros((1, num_channels, 1, 1)), np.zeros((1, num_channels, 1, 1))]},
                 mins = {"type": "min", "counts": 0, "values": np.zeros((1, num_channels, 1, 1))},
                 maxs = {"type": "max", "counts": 0, "values": np.zeros((1, num_channels, 1, 1))},
                 time_means = {"type": "mean", "counts": 0, "values": np.zeros((1, num_channels, height, width))},
                 time_diff_meanvar = {"type": "meanvar", "counts": 0, "values": [np.zeros((1, num_channels, 1, 1)), np.zeros((1, num_channels, 1, 1))]})

    if wind_channels is not None:
        num_wind_channels = len(wind_channels[0])
        stats["wind_meanvar"] = {"type": "meanvar", "counts": 0, "values": [np.zeros((1, num_wind_channels, 1, 1)), np.zeros((1, num_wind_channels, 1, 1))]}
        stats["winddiff_meanvar"] = {"type": "meanvar", "counts": 0, "values": [np.zeros((1, num_wind_channels, 1, 1)), np.zeros((1, num_wind_channels, 1, 1))]}

    # compute local stats
    if comm_rank == 0:
        progress = tqdm(desc="Computing stats", total=num_samples_local)
    else:
        progress = None
    start = time.time()
    for filename, index_bounds in mapping.items():
        tmpstats = get_file_stats(filename, slice(index_bounds[0], index_bounds[1]+1), wind_channels, quadrature, dt, batch_size, progress)
        stats = welford_combine(stats, tmpstats)
    duration = time.time() - start
    if comm_rank == 0:
        progress.close()

    # wait for everybody else
    print(f"Rank {comm_rank} done. Duration for {num_samples_local} samples: {duration:.2f}s", flush=True)
    group_comm.Barrier()

    # now gather the stats across group:
    stats = collective_reduce(group_comm, stats)
    intergroup_comm.Barrier()
    if group_rank == 0:
        print(f"Group {group_id} done.", flush=True)

    # now, do binary reduction orthogonal to groups
    stats = binary_reduce(intergroup_comm, stats)

    # wait for everybody
    comm.Barrier()

    if comm_rank == 0:
        # compute global stds:
        stats["global_meanvar"]["values"][1] = np.sqrt(stats["global_meanvar"]["values"][1] / float(stats["global_meanvar"]["counts"]))
        stats["time_diff_meanvar"]["values"][1] = np.sqrt(stats["time_diff_meanvar"]["values"][1] / float(stats["time_diff_meanvar"]["counts"]))

        # overwrite the wind channels
        if wind_channels is not None:
            stats["wind_meanvar"]["values"][1] = np.sqrt(stats["wind_meanvar"]["values"][1] / float(stats["wind_meanvar"]["counts"]))
            # overwrite stds but do not overwrite means
            stats["global_meanvar"]["values"][1][: , wind_channels[0]] = stats["wind_meanvar"]["values"][1]
            stats["global_meanvar"]["values"][1][: , wind_channels[1]] = stats["wind_meanvar"]["values"][1]

            # same for wind diffs
            stats["winddiff_meanvar"]["values"][1] = np.sqrt(stats["winddiff_meanvar"]["values"][1] / float(stats["winddiff_meanvar"]["counts"]))
            # again, only overwrite stds:
            stats["time_diff_meanvar"]["values"][1][:, wind_channels[0]] = stats["winddiff_meanvar"]["values"][1]
            stats["time_diff_meanvar"]["values"][1][:, wind_channels[1]] = stats["winddiff_meanvar"]["values"][1]


        # save the stats
        np.save(os.path.join(output_path, 'global_means.npy'), stats["global_meanvar"]["values"][0].astype(np.float32))
        np.save(os.path.join(output_path, 'global_stds.npy'), stats["global_meanvar"]["values"][1].astype(np.float32))
        np.save(os.path.join(output_path, 'mins.npy'), stats["mins"]["values"].astype(np.float32))
        np.save(os.path.join(output_path, 'maxs.npy'), stats["maxs"]["values"].astype(np.float32))
        np.save(os.path.join(output_path, 'time_means.npy'), stats["time_means"]["values"].astype(np.float32))
        np.save(os.path.join(output_path, f'time_diff_means_dt{dt}.npy'), stats["time_diff_meanvar"]["values"][0].astype(np.float32))
        np.save(os.path.join(output_path, f'time_diff_stds_dt{dt}.npy'), stats["time_diff_meanvar"]["values"][1].astype(np.float32))

        print("means: ", stats["global_meanvar"]["values"][0])
        print("stds: ", stats["global_meanvar"]["values"][1])
        print(f"time_diff_means (dt={dt}): ", stats["time_diff_meanvar"]["values"][0])
        print(f"time_diff_stds (dt={dt}): ", stats["time_diff_meanvar"]["values"][1])


    # wait for rank 0 to finish
    comm.Barrier()

    
def main(args):
    get_stats(input_path=args.input_path,
              output_path=args.output_path,
              metadata_file=args.metadata_file,
              dt=args.dt,
              quadrature_rule=args.quadrature_rule,
              wind_angle_aware=args.wind_angle_aware,
              batch_size=args.batch_size,
              reduction_group_size=args.reduction_group_size,
    )

    return


if __name__ == "__main__":
    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Directory with input files or a virtual hdf5 file with the combined input.", required=True)
    parser.add_argument("--metadata_file", type=str, help="File containing dataset metadata.", required=True)
    parser.add_argument("--output_path", type=str, help="Directory for saving stats files.", required=True)
    parser.add_argument("--reduction_group_size", type=int, default=8, help="Size of collective reduction groups.")
    parser.add_argument("--quadrature_rule", type=str, default="naive", choices=["naive", "clenshaw-curtiss", "gauss-legendre"], help="Specify quadrature_rule for spatial averages.")
    parser.add_argument("--dt", type=int, default=1, help="Step size for which time difference stats will be computed.")
    parser.add_argument('--wind_angle_aware', action='store_true', help="Just compute mean and magnitude of wind vectors and not componentwise stats")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size used for reading chunks from a file at a time to avoid OOM errors.")
    args = parser.parse_args()

    main(args)




