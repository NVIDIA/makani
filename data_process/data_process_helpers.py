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

import torch
import torch.distributed as dist
import math

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
        if (s_b["counts"].ndim != 0) and (s_a["counts"].ndim != s_a["values"].ndim):
            n_a = s_a["counts"][None, :, None, None]
            n_b = s_b["counts"][None, :, None, None]
        else:
            n_a = s_a["counts"]
            n_b = s_b["counts"]
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
