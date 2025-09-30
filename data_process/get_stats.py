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
import time
import socket
import json
import numpy as np
import h5py as h5
import math
import argparse as ap
from itertools import accumulate
import operator
from bisect import bisect_right
from glob import glob

# MPI
from mpi4py import MPI

import torch
import torch.distributed as dist
from makani.utils.grids import GridQuadrature

from wb2_helpers import DistributedProgressBar

def mask_data(data):
    data_masked = data.clone()
    nan_mask = torch.isnan(data)
    data_masked[nan_mask] = 0.0
    valid_mask = torch.logical_not(nan_mask).to(torch.float64)
    return data_masked, valid_mask


def allgather_dict(stats, group):
    # initialize a list of empty dictionaries
    stats_gather = []
    for _ in range(dist.get_world_size(group)):
        substats = {varname: {} for varname in stats.keys()}
        stats_gather.append(substats)
    
    # iterate over full dict
    for varname, substats in stats.items():
        for k,v in substats.items():
            if isinstance(v, torch.Tensor):
                vcont = v.contiguous()
                v_gather = [torch.empty_like(vcont) for _ in range(dist.get_world_size(group))]
                v_gather[dist.get_rank(group)] = vcont
                dist.all_gather(v_gather, vcont, group=group)
                for ivg, vg in enumerate(v_gather):
                    stats_gather[ivg][varname][k] = vg
            else:
                for ivg in range(dist.get_world_size(group)):
                    stats_gather[ivg][varname][k] = v

    return stats_gather


def send_recv_dict(stats, src_rank, dst_rank, group):
    group_rank = dist.get_rank(group)
    group_size = dist.get_world_size(group)
    stats_recv = {varname: {} for varname in stats.keys()}
    count = 0
    for varname, substats in stats.items():
        for k,v in substats.items():
            if isinstance(v, torch.Tensor):
                # send/recv
                tag = src_rank + group_size * count
                if group_rank == dst_rank:
                    # we need to convert group rank to global rank
                    src_rank_global = dist.get_global_rank(group, src_rank)
                    recv_handle = dist.irecv(v, src=src_rank_global, tag=tag, group=group)
                    recv_handle.wait()
                elif group_rank == src_rank:
                    # we need to convert group rank to global rank
                    dst_rank_global = dist.get_global_rank(group, dst_rank)
                    send_handle = dist.isend(v, dst=dst_rank_global, tag=tag, group=group)
                    send_handle.wait()
                count += 1

            # update dictionary
            stats_recv[varname][k] = v

    return stats_recv


def collective_reduce(stats, group):
    # get stats from all ranks
    statslist = allgather_dict(stats, group)

    # perform welford reduction
    stats_reduced = statslist[0]
    for tmpstats in statslist[1:]:
        stats_reduced = welford_combine(stats_reduced, tmpstats)

    return stats_reduced


def binary_reduce(stats, group, device):
    csize = dist.get_world_size(group)
    crank = dist.get_rank(group)

    # check for power of two
    assert((csize & (csize-1) == 0) and csize != 0)

    # how many steps do we need:
    nsteps = int(math.log(csize,2))

    # init step 1
    recv_ranks = range(0,csize,2)
    send_ranks = range(1,csize,2)

    for step in range(nsteps):
        for rrank, srank in zip(recv_ranks, send_ranks):
            rstats = send_recv_dict(stats, srank, rrank, group)
            if crank == rrank:
                stats = welford_combine(stats, rstats)

        # wait for everyone being ready before doing the next epoch
        dist.barrier(group=group, device_ids=[device.index])

        # shrink the list
        if (step < nsteps-1):
            recv_ranks = recv_ranks[0::2]
            send_ranks = recv_ranks[1::2]

    return stats


def welford_combine(stats1, stats2):
    # update time means first:
    stats = {}

    for k in stats1.keys():
        s_a = stats1[k]
        s_b = stats2[k]

        # update stats, but unexpand to match shapes
        n_a = s_a["counts"][None, :, None, None]
        n_b = s_b["counts"][None, :, None, None]
        n_ab = n_a + n_b

        if s_a["type"] == "min":
            values = torch.minimum(s_a["values"], s_b["values"])
        elif s_a["type"] == "max":
            values = torch.maximum(s_a["values"], s_b["values"])
        elif s_a["type"] == "mean":
            mean_a = s_a["values"]
            mean_b = s_b["values"]
            values = (mean_a * n_a + mean_b * n_b) / n_ab
        elif s_a["type"] == "meanvar":
            mean_a, m2_a = s_a["values"].unbind(0)
            mean_b, m2_b = s_b["values"].unbind(0)
            delta = mean_b - mean_a

            values = torch.stack(
                [
                    (mean_a * n_a + mean_b * n_b) / n_ab,
                    m2_a + m2_b + delta * delta * n_a * n_b / n_ab
                ], dim=0
            ).contiguous()

        stats[k] = {"counts": n_ab.reshape(-1),
                    "type": s_a["type"],
                    "values": values}

    return stats


def get_file_stats(filename,
                   file_slice,
                   wind_indices,
                   quadrature,
                   fail_on_nan=False,
                   dt=1,
                   batch_size=16,
                   device=torch.device("cpu"),
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
            data = dset[sub_slc, ...]
            tdata = torch.from_numpy(data).to(device=device, dtype=torch.float64)

            # check for NaNs
            if torch.isnan(tdata).any():
                if fail_on_nan:
                    raise ValueError(f"NaN values encountered in {filename}.")
                else:
                    # create mask of NaNs and mask data
                    tdata_masked, valid_mask = mask_data(tdata)
            else:
                tdata_masked = tdata
                valid_mask = torch.ones_like(tdata)

            # define counts
            counts_time = tdata.shape[0]
            valid_count = torch.sum(quadrature(valid_mask), dim=0)
            counts_time_space = valid_count
            
            # Basic observables
            # compute mean and variance
            # the mean needs to be divided by number of valid samples:
            tmean = torch.sum(quadrature(tdata_masked), dim=0, keepdim=False).reshape(1, -1, 1, 1) / valid_count[None, :, None, None]
            # we compute m2 directly, so we do not need to divide by number of valid samples:
            tm2 = torch.sum(quadrature(torch.square(tdata_masked - tmean)), dim=0, keepdim=False).reshape(1, -1, 1, 1)

            # fill the dict
            tmpstats = dict(
                maxs = {
                    "type": "max",
                    "counts": counts_time_space.clone(),
                    # apparently, torch.max does not support multiple dimensions, so we need to do it in steps
                    "values": torch.max(torch.max(torch.max(tdata, dim=0, keepdim=True).values, dim=2, keepdim=True).values, dim=3, keepdim=True).values,
                },
                mins = {
                    "type": "min",
                    "counts": counts_time_space.clone(),
                    # same for torch.min
                    "values": torch.min(torch.min(torch.min(tdata, dim=0, keepdim=True).values, dim=2, keepdim=True).values, dim=3, keepdim=True).values,
                },
                time_means = {
                    "type": "mean",
                    "counts": float(counts_time) * torch.ones((data.shape[1]), dtype=torch.float64, device=device),
                    "values": torch.mean(tdata, dim=0, keepdim=True),
                },
                global_meanvar = {
                    "type": "meanvar",
                    "counts": valid_count.clone(),
                    "values": torch.stack([tmean, tm2], dim=0).contiguous(),
                }
            )

            # time diffs: read one more sample for these, if possible
            # TODO: tile it for dt < batch_size
            if batch_start >= dt:
                sub_slc_m_dt = slice(batch_start-dt, batch_stop)
                data_m_dt = dset[sub_slc_m_dt, ...]
                tdata_m_dt = torch.from_numpy(data_m_dt).to(device=device, dtype=torch.float64)
                tdiff = tdata_m_dt[dt:, ...] - tdata_m_dt[:-dt, ...]
                counts_timediff = tdiff.shape[0]
                tdiff_masked, tdiff_valid_mask = mask_data(tdiff)
                tdiff_valid_count = torch.sum(quadrature(tdiff_valid_mask), dim=0)
                tdiffmean = torch.sum(quadrature(tdiff_masked), dim=0, keepdim=False).reshape(1, -1, 1, 1) / tdiff_valid_count[None, :, None, None]
                tdiffm2 = torch.sum(quadrature(torch.square(tdiff_masked - tdiffmean)), dim=0, keepdim=False).reshape(1, -1, 1, 1)
            else:
                # skip those for tdiff
                counts_timediff = 0
                tdiff_valid_count = torch.zeros((data.shape[1]), dtype=torch.float64, device=device)

            if counts_timediff != 0:
                tmpstats["time_diff_meanvar"] = {
                    "type": "meanvar",
                    "counts": tdiff_valid_count.clone(),
                    "values": torch.stack([tdiffmean, tdiffm2], dim=0).contiguous(),
                }
            else:
                # we need the shapes
                tshape = tmean.shape
                tmpstats["time_diff_meanvar"] = {
                    "type": "meanvar", 
                    "counts": torch.zeros(data.shape[1], dtype=torch.float64, device=device),
                    "values": torch.stack(
                        [
                            torch.zeros(tshape, dtype=torch.float64, device=device), 
                            torch.zeros(tshape, dtype=torch.float64, device=device)
                        ], 
                        dim=0
                    ).contiguous(),
                }

            if wind_indices is not None:
                u_tens = tdata[:, wind_indices[0]]
                v_tens = tdata[:, wind_indices[1]]
                wind_magnitude = torch.sqrt(torch.square(u_tens) + torch.square(v_tens))
                wind_magnitude_masked, wind_valid_mask = mask_data(wind_magnitude)
                wind_valid_count = torch.sum(quadrature(wind_valid_mask), dim=0)
                wind_mean = torch.sum(quadrature(wind_magnitude_masked), dim=0, keepdim=False).reshape(1, -1, 1, 1) / wind_valid_count[None, :, None, None]
                wind_m2 = torch.sum(quadrature(torch.square(wind_magnitude_masked - wind_mean)), dim=0, keepdim=False).reshape(1, -1, 1, 1)
                tmpstats["wind_meanvar"] = {
                    "type": "meanvar",
                    "counts": wind_valid_count.clone(),
                    "values": torch.stack([wind_mean, wind_m2], dim=0).contiguous(),
                }

                if counts_timediff != 0:
                    udiff_tens = tdiff[:, wind_indices[0]]
                    vdiff_tens = tdiff[:, wind_indices[1]]
                    winddiff_magnitude = torch.sqrt(torch.square(udiff_tens) + torch.square(vdiff_tens))
                    winddiff_magnitude_masked, winddiff_valid_mask = mask_data(winddiff_magnitude)
                    winddiff_valid_count = torch.sum(quadrature(winddiff_valid_mask), dim=0)
                    winddiff_mean = torch.sum(quadrature(winddiff_magnitude_masked), dim=0, keepdim=False).reshape(1, -1, 1, 1) / winddiff_valid_count[None, :, None, None]
                    winddiff_m2 = torch.sum(quadrature(torch.square(winddiff_magnitude_masked - winddiff_mean)), dim=0, keepdim=False).reshape(1, -1, 1, 1)
                    tmpstats["winddiff_meanvar"] = {
                        "type": "meanvar",
                        "counts": winddiff_valid_count.clone(),
                        "values": torch.stack([winddiff_mean, winddiff_m2], dim=0).contiguous(),
                    }
                else:
                    wdiffshape = wind_mean.shape
                    tmpstats["winddiff_meanvar"] = {
                        "type": "meanvar",
                        "counts": torch.zeros(wdiffshape[1], dtype=torch.float64, device=device),
                        "values": torch.stack(
                            [
                                torch.zeros(wdiffshape, dtype=torch.float64, device=device), 
                                torch.zeros(wdiffshape, dtype=torch.float64, device=device)
                            ],
                            dim=0
                        ).contiguous(),
                    }

            if stats is not None:
                stats = welford_combine(stats, tmpstats)
            else:
                stats = tmpstats

            if progress is not None:
                progress.update_counter(batch_stop-batch_start)
                progress.update_progress()

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


def get_stats(input_path: str, output_path: str, metadata_file: str,
              dt: int, quadrature_rule: str, wind_angle_aware: bool, fail_on_nan: bool=False,
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
    fail_on_nan : bool
        If this flag is set to true, then the code will fail if NaN values are encountered.
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
    comm_local_rank = comm_rank % torch.cuda.device_count()

    # create group comm
    #group_id = comm_rank % reduction_group_size
    #group_rank = comm_rank // reduction_group_size
    #group_comm = comm.Split(color=group_id, key=group_rank)

    # create intergroup comm
    #intergroup_comm = comm.Split(color=group_rank, key=group_id)
    # set wireup parameters
    hostname = socket.gethostname()
    hostname = comm.bcast(hostname, root=0)
    os.environ["MASTER_ADDR"] = hostname
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(comm_rank)
    os.environ["WORLD_SIZE"] = str(comm_size)
    os.environ["LOCAL_RANK"] = str(comm_local_rank)

    # init torch distributed
    device = torch.device(f"cuda:{comm_local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo", 
        init_method="env://",
        world_size=comm_size,
        rank=comm_rank,
        device_id=device,
    )
    mesh = dist.init_device_mesh(
        device_type=device.type, 
        mesh_shape=[reduction_group_size, comm_size // reduction_group_size],
        mesh_dim_names=["reduction", "tree"],
    )

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
        wind_channels, _ = get_wind_channels(channel_names)

    # get file offsets
    num_samples_total = sum(num_samples)
    num_channels = data_shape[1]
    height, width = (data_shape[2], data_shape[3])

    # quadrature:
    quadrature = GridQuadrature(quadrature_rule, (height, width),
                                crop_shape=None, crop_offset=(0, 0),
                                normalize=False, pole_mask=None).to(device)

    if comm_rank == 0:
        print(f"Found {len(filelist)} files with a total of {num_samples_total} samples. Each sample has the shape {num_channels}x{height}x{width} (CxHxW).")

    # do the sharding:
    num_samples_chunk = (num_samples_total + comm_size - 1) // comm_size
    samples_start = num_samples_chunk * comm_rank
    samples_end = min([samples_start + num_samples_chunk, num_samples_total])
    sample_offsets = list(accumulate(num_samples, operator.add))[:-1]
    sample_offsets.insert(0, 0)
    #num_samples_local = samples_end - samples_start

    if comm_rank == 0:
        print("Loading data with the following chunking:")
    for	rank in	range(comm_size):
        if comm_rank ==	rank:
            print(f"Rank {comm_rank}, working on samples [{samples_start}, {samples_end})", flush=True)
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
    stats = dict(
        global_meanvar = {
            "type": "meanvar", 
            "counts": torch.zeros((num_channels), dtype=torch.float64, device=device), 
            "values": torch.zeros((2, 1, num_channels, 1, 1), dtype=torch.float64, device=device),
        },
        mins = {
            "type": "min", 
            "counts": torch.zeros((num_channels), dtype=torch.float64, device=device), 
            "values": torch.full((1, num_channels, 1, 1), torch.inf, dtype=torch.float64, device=device)
        },
        maxs = {
            "type": "max", 
            "counts": torch.zeros((num_channels), dtype=torch.float64, device=device), 
            "values": torch.full((1, num_channels, 1, 1), -torch.inf, dtype=torch.float64, device=device)
        },
        time_means = {
            "type": "mean", 
            "counts": torch.zeros((num_channels), dtype=torch.float64, device=device), 
            "values": torch.zeros((1, num_channels, height, width), dtype=torch.float64, device=device)
        },
        time_diff_meanvar = {
            "type": "meanvar", 
            "counts": torch.zeros((num_channels), dtype=torch.float64, device=device), 
            "values": torch.zeros((2, 1, num_channels, 1, 1), dtype=torch.float64, device=device), 
        }
    )

    if wind_channels is not None:
        num_wind_channels = len(wind_channels[0])
        stats["wind_meanvar"] = {
            "type": "meanvar", 
            "counts": torch.zeros((num_wind_channels), dtype=torch.float64, device=device), 
            "values": torch.zeros((2, 1, num_wind_channels, 1, 1), dtype=torch.float64, device=device), 
        }
        stats["winddiff_meanvar"] = {
            "type": "meanvar", 
            "counts": torch.zeros((num_wind_channels), dtype=torch.float64, device=device),
            "values": torch.zeros((2, 1, num_wind_channels, 1, 1), dtype=torch.float64, device=device), 
        }

    # compute local stats
    progress = DistributedProgressBar(num_samples_total, comm)
    start = time.time()
    for filename, index_bounds in mapping.items():
        tmpstats = get_file_stats(filename, slice(index_bounds[0], index_bounds[1]+1), wind_channels, quadrature, fail_on_nan, dt, batch_size, device, progress)
        stats = welford_combine(stats, tmpstats)

    # wait for everybody else
    comm.Barrier()
    duration = time.time() - start
    if comm_rank == 0:
        print(f"Duration for {num_samples_total} samples: {duration:.2f}s", flush=True)
    del progress

    # do reductions within groups
    start = time.time()
    stats = collective_reduce(stats, mesh.get_group("reduction"))
    comm.Barrier()
    duration = time.time() - start
    if comm_rank == 0:
        print(f"Reduction within groups done. Duration: {duration:.2f}s", flush=True)

    # now, do binary reduction orthogonal to groups
    start = time.time()
    # only rank 0 of the allreduce group will do the binary reduction
    if dist.get_rank(mesh.get_group("reduction")) == 0:
        stats = binary_reduce(stats, mesh.get_group("tree"), device)
    comm.Barrier()
    duration = time.time() - start
    if comm_rank == 0:
        print(f"Reduction across groups done. Duration: {duration:.2f}s", flush=True)

    # write the data to disk
    if comm_rank == 0:
        start = time.time()

        # move stats to cpu and convert to numpy
        for varname, substats in stats.items():
            for k,v in substats.items():
                if isinstance(v, torch.Tensor):
                    stats[varname][k] = v.cpu().numpy()

        # compute global stds:
        stats["global_meanvar"]["values"][1, ...] = np.sqrt(stats["global_meanvar"]["values"][1, ...] / stats["global_meanvar"]["counts"][None, :, None, None])
        stats["time_diff_meanvar"]["values"][1, ...] = np.sqrt(stats["time_diff_meanvar"]["values"][1, ...] / stats["time_diff_meanvar"]["counts"][None, :, None, None])

        # overwrite the wind channels
        if wind_channels is not None:
            stats["wind_meanvar"]["values"][1, ...] = np.sqrt(stats["wind_meanvar"]["values"][1, ...] / stats["wind_meanvar"]["counts"][None, :, None, None])
            # there is a numpy bug here: if the leading dim is singleton and the second dim gets selected, the
            # dims are swapped afterwards. Working around this by making use of the fact that batch dim is singleton:
            stats["global_meanvar"]["values"][1, 0, wind_channels[0], ...] = stats["wind_meanvar"]["values"][1, 0, ...]
            stats["global_meanvar"]["values"][1, 0, wind_channels[1], ...] = stats["wind_meanvar"]["values"][1, 0, ...]

            # same for wind diffs
            stats["winddiff_meanvar"]["values"][1, ...] = np.sqrt(stats["winddiff_meanvar"]["values"][1, ...] / stats["winddiff_meanvar"]["counts"][None, :, None, None])
            # again, only overwrite stds:
            stats["time_diff_meanvar"]["values"][1, 0, wind_channels[0]] = stats["winddiff_meanvar"]["values"][1, 0, ...]
            stats["time_diff_meanvar"]["values"][1, 0, wind_channels[1]] = stats["winddiff_meanvar"]["values"][1, 0, ...]


        # save the stats
        np.save(os.path.join(output_path, 'global_means.npy'), stats["global_meanvar"]["values"][0, ...].astype(np.float32))
        np.save(os.path.join(output_path, 'global_stds.npy'), stats["global_meanvar"]["values"][1, ...].astype(np.float32))
        np.save(os.path.join(output_path, 'mins.npy'), stats["mins"]["values"].astype(np.float32))
        np.save(os.path.join(output_path, 'maxs.npy'), stats["maxs"]["values"].astype(np.float32))
        np.save(os.path.join(output_path, 'time_means.npy'), stats["time_means"]["values"].astype(np.float32))
        np.save(os.path.join(output_path, f'time_diff_means_dt{dt}.npy'), stats["time_diff_meanvar"]["values"][0, ...].astype(np.float32))
        np.save(os.path.join(output_path, f'time_diff_stds_dt{dt}.npy'), stats["time_diff_meanvar"]["values"][1, ...].astype(np.float32))

        duration = time.time() - start
        print(f"Saving stats done. Duration: {duration:.2f}s", flush=True)

        print("means: ", stats["global_meanvar"]["values"][0, ...])
        print("stds: ", stats["global_meanvar"]["values"][1, ...])
        print(f"time_diff_means (dt={dt}): ", stats["time_diff_meanvar"]["values"][0, ...])
        print(f"time_diff_stds (dt={dt}): ", stats["time_diff_meanvar"]["values"][1, ...])


    # wait for rank 0 to finish
    comm.Barrier()

    # shut down pytorch comms
    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()

    # close MPI
    MPI.Finalize()


def main(args):
    get_stats(input_path=args.input_path,
              output_path=args.output_path,
              metadata_file=args.metadata_file,
              dt=args.dt,
              quadrature_rule=args.quadrature_rule,
              wind_angle_aware=args.wind_angle_aware,
              fail_on_nan=args.fail_on_nan,
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
    parser.add_argument('--fail_on_nan', action='store_true', help="When computing stats, code will fail if NaN values are encountered.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size used for reading chunks from a file at a time to avoid OOM errors.")
    args = parser.parse_args()

    main(args)




