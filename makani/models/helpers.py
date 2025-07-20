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

import torch
import torch.distributed as dist

from makani.utils import comm


def count_parameters(model, device):
    """Counts model parameters"""

    with torch.no_grad():
        total_stats = torch.zeros(2, dtype=torch.long, device=device)
        local_bytes = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue

            # make sure complex weight tensors are accounted for correctly
            pview = torch.view_as_real(p) if p.is_complex() else p
            pstats = torch.tensor([pview.numel(), pview.nbytes], dtype=torch.long, device=device)
            local_bytes += pview.nbytes

            # if the weight is split, then we need to reduce
            if hasattr(p, "sharded_dims_mp"):
                for group in p.sharded_dims_mp:
                    if (group is not None) and (comm.get_size(group) > 1):
                        dist.all_reduce(pstats, group=comm.get_group(group))

            # sum the total stats
            total_stats += pstats

    # transfer to cpu
    total_stats_arr = total_stats.cpu().numpy()
    total_count = total_stats_arr[0]
    total_bytes = total_stats_arr[1]

    return total_count, total_bytes, local_bytes


def compare_model_parameters(model1, model2):
    """Checks whether both models have the same parameters"""

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).any():
            return False
    return True


def check_parameters(model):
    """Prints shapes, strides and whether parameters are contiguous"""
    for p in model.parameters():
        if p.requires_grad:
            print(p.shape, p.stride(), p.is_contiguous())

    return
