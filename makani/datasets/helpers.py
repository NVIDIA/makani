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


from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist

from makani.datasets.parallel import MPI, is_parallel

if MPI is not None:
    from mpi4py.util import dtlib


def mask_data(data):
    # check for NaNs and return a FP valued mask where
    # valid_mask = 1 for data which are valid (not NaN) and 0 otherwise.

    if torch.isnan(data).any():
        nan_mask = torch.isnan(data)
        valid_mask = torch.logical_not(nan_mask).to(torch.float64)
        data_masked = torch.where(valid_mask > 0.0, data, 0.0)
    else:
        valid_mask = torch.ones_like(data)
        data_masked = data

    return data_masked, valid_mask


def file_sample_ranges(filelist, num_samples, samples_start, samples_end):
    """
    The samples ``[samples_start, samples_end)`` of the files ``filelist``, taken one after the other.

    ``num_samples`` holds the number of samples of each file. Returns ``{filename: slice}`` with the
    range of these samples within each file, in file order, leaving out the files with none of them.
    A single combined file is the case of one file.
    """
    ranges = {}
    offset = 0
    for filename, count in zip(filelist, num_samples):
        start = max(samples_start - offset, 0)
        stop = min(samples_end - offset, count)
        if start < stop:
            ranges[filename] = slice(start, stop)
        offset += count
    return ranges


def allgather_dict(stats, group):
    # initialize a list of empty dictionaries
    stats_gather = []
    for _ in range(dist.get_world_size(group)):
        substats = {varname: {} for varname in stats.keys()}
        stats_gather.append(substats)

    # iterate over full dict
    for varname, substats in stats.items():
        for k, v in substats.items():
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
    """
    Send the tensors of ``stats`` from ``src_rank`` to ``dst_rank`` of ``group``.

    On ``dst_rank``, returns what ``src_rank`` sent, received into new tensors, so that its own
    ``stats`` stay as they are; on every other rank, returns ``stats``.
    """
    group_rank = dist.get_rank(group)
    group_size = dist.get_world_size(group)
    stats_recv = {varname: {} for varname in stats.keys()}
    count = 0
    for varname, substats in stats.items():
        for k, v in substats.items():
            if isinstance(v, torch.Tensor):
                tag = src_rank + group_size * count
                if group_rank == dst_rank:
                    buffer = torch.empty_like(v)
                    dist.recv(buffer, src=dist.get_global_rank(group, src_rank), tag=tag, group=group)
                    v = buffer
                elif group_rank == src_rank:
                    dist.send(v.contiguous(), dst=dist.get_global_rank(group, dst_rank), tag=tag, group=group)
                count += 1

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
    """
    Reduce the stats of all ranks of ``group`` onto its rank 0, in pairs.

    At distance 1, 2, 4, ..., each rank ``r`` that is a multiple of twice the distance combines what
    rank ``r + distance`` sends, if there is one, so any group size works in ``ceil(log2(size))``
    steps. Only rank 0 holds the result.
    """
    csize = dist.get_world_size(group)
    crank = dist.get_rank(group)

    distance = 1
    while distance < csize:
        if (crank % (2 * distance) == 0) and (crank + distance < csize):
            stats = welford_combine(stats, send_recv_dict(stats, crank + distance, crank, group))
        elif crank % (2 * distance) == distance:
            send_recv_dict(stats, crank, crank - distance, group)

        # everyone is done with this step before the next one
        dist.barrier(group=group, device_ids=[device.index] if device.type == "cuda" else None)
        distance *= 2

    return stats


def welford_combine(stats1, stats2):
    # update time means first:
    stats = {}

    for k in stats1.keys():
        s_a = stats1[k]
        s_b = stats2[k]

        # update stats, but unexpand to match shapes
        if (s_b["counts"].ndim != 0) and (s_a["counts"].ndim != s_a["values"].ndim):
            n_a = s_a["counts"][None, :, None, None]
            n_b = s_b["counts"][None, :, None, None]
            reshape = True
        else:
            n_a = s_a["counts"]
            n_b = s_b["counts"]
            reshape = False

        # combined counts
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
                [(mean_a * n_a + mean_b * n_b) / n_ab, m2_a + m2_b + delta * delta * n_a * n_b / n_ab], dim=0
            ).contiguous()

        if reshape:
            n_ab = n_ab.reshape(-1)

        stats[k] = {"counts": n_ab, "type": s_a["type"], "values": values}

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
        raise ValueError(
            "Error, cannot group wind channels together because not all pairs of wind channels are in the dataset."
        )

    # find the indices of the channels in the original channel names:
    uchannels = [channel_names.index(u) for u in u_variables]
    vchannels = [channel_names.index(v) for v in v_variables]

    return (uchannels, vchannels), (u_variables, v_variables)


class DistributedProgressBar(object):
    """
    A progress bar on rank 0 for the work of all ranks.

    The ranks add what they did to a counter on rank 0, an MPI window; a single rank, also one
    without MPI (:class:`~makani.datasets.parallel.SerialComm`), counts locally.
    """

    def __init__(self, num_entries: int, comm):
        self.comm = comm

        if is_parallel(comm):
            datatype = MPI.INT64_T
            np_dtype = dtlib.to_numpy_dtype(datatype)
            self.win = MPI.Win.Allocate(datatype.Get_size(), comm=comm)
        else:
            np_dtype = np.int64
            self.win = None
        self.counts = np.zeros([1], dtype=np_dtype)
        self.local_count = 0
        self.comm.Barrier()

        if self.comm.Get_rank() == 0:
            self.pbar = tqdm(total=num_entries)
        self.reset()

    def __del__(self):
        self.comm.Barrier()
        if self.win is not None:
            self.win.Free()
        if self.comm.Get_rank() == 0:
            self.pbar.close()

    def reset(self):
        if self.comm.Get_rank() == 0:
            self.pbar.n = 0
            self.pbar.refresh()

    def update_counter(self, count: int):
        if self.win is None:
            self.local_count += count
            return
        self.counts[0] = count
        self.win.Lock(rank=0)
        self.win.Accumulate(self.counts, target_rank=0)
        self.win.Flush_local(rank=0)
        self.win.Unlock(rank=0)

    def get_counter(self) -> int:
        if self.win is None:
            return self.local_count
        self.win.Lock(rank=0, lock_type=MPI.LOCK_SHARED)
        self.win.Get(self.counts, target_rank=0)
        self.win.Flush(rank=0)
        self.win.Unlock(rank=0)
        return int(self.counts[0])

    def update_progress(self):
        if self.comm.Get_rank() == 0:
            count = self.get_counter()
            self.pbar.n = min(count, self.pbar.total)
            self.pbar.refresh()
