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
import time
import logging
import math
from typing import Union
import numpy as np

# we are using the distributed manager from physicsnemo
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.distributed.config import ProcessGroupNode, ProcessGroupConfig

# we need this
_DM = None
_COMM_ROOTS = {}


def get_size(name: str) -> int:
    global _DM
    if (_DM is not None) and (_DM.world_size > 1):
        return _DM.group_size(name)
    else:
        return 1


def get_rank(name: str) -> int:
    global _DM
    if (_DM is not None) and (_DM.world_size > 1):
        return _DM.group_rank(name)
    else:
        return 0


def get_group(name: str):
    global _DM
    if _DM is not None:
        return _DM.group(name)
    else:
        return None


def get_root(name: str) -> int:
    global _DM
    global _COMM_ROOTS
    if (name in _COMM_ROOTS) and (_DM.world_size > 1):
        return _COMM_ROOTS[name]
    else:
        return 0


# specialized routines for world comms
def get_world_size():
    global _DM
    if _DM is not None:
        return _DM.world_size
    else:
        return 1


def get_world_rank():
    global _DM
    if _DM is not None:
        return _DM.rank
    else:
        return 0


def get_local_rank():
    global _DM
    if _DM is not None:
        return _DM.local_rank
    else:
        return 0


def get_comm_names():
    global _DM
    if _DM is not None:
        return [name for name in _DM.group_names if (not name.startswith("__orthogonal_to"))]
    else:
        return []


def get_model_comm_names():
    return [x for x in get_comm_names() if x not in ["world", "data", "ensemble", "batch"]]


def is_distributed(name: str):
    global _DM
    if _DM is not None:
        return name in _DM.group_names
    else:
        return False


def cleanup():
    global _DM
    if _DM is not None:
        _DM.cleanup()
        _DM = None
    return
    

# initialization routine
def init(model_parallel_sizes=[1, 1, 1, 1], model_parallel_names=["h", "w", "fin", "fout"], data_parallel_sizes=[1, -1], data_parallel_names=["ensemble", "batch"], verbose=False):

    # call basic init first
    DistributedManager.initialize()

    # extract manager object
    global _DM
    _DM = DistributedManager()

    # create process group config:
    world = ProcessGroupNode("world", size=_DM.world_size)
    pconfig = ProcessGroupConfig(world)

    # add nodes:
    # model
    pconfig.add_node(ProcessGroupNode("model"), parent="world")
    # spatial and matmul
    pconfig.add_node(ProcessGroupNode("spatial"), parent="model")
    pconfig.add_node(ProcessGroupNode("matmul"), parent="model")
    # subgroups for spatial
    pconfig.add_node(ProcessGroupNode("h"), parent="spatial")
    pconfig.add_node(ProcessGroupNode("w"), parent="spatial")
    # subgroups for matmul:
    pconfig.add_node(ProcessGroupNode("fin"), parent="matmul")
    pconfig.add_node(ProcessGroupNode("fout"), parent="matmul")
    # add data node last
    pconfig.add_node(ProcessGroupNode("data"), parent="world")
    # other data parallel dims
    for dgname in data_parallel_names:
        pconfig.add_node(ProcessGroupNode(dgname), parent="data")

    # set up leaf sizes
    # model
    model_leaf_config = {}
    for k, v in zip(model_parallel_names, model_parallel_sizes):
        model_leaf_config[k] = v
    # data
    data_group_size = _DM.world_size // math.prod(model_leaf_config.values())
    data_leaf_config = {}
    for k, v in zip(data_parallel_names, data_parallel_sizes):
        data_leaf_config[k] = v
    # determine some automatic shapes: only one is supported, the others will
    # default to 1
    ndata = 1
    for k in data_leaf_config:
        v = data_leaf_config[k]
        if v > 0:
            ndata *= v
    for k in data_leaf_config:
        v = data_leaf_config[k]
        if v <= 0:
            data_leaf_config[k] = data_group_size // ndata
            # the others will automatically be sized 1
            ndata = data_group_size
    # fuse leaf configs
    leaf_config = model_leaf_config
    for k in data_leaf_config:
        leaf_config[k] = data_leaf_config[k]
    # update sizes
    pconfig.set_leaf_group_sizes(leaf_config, update_parent_sizes=True)

    # create remaining process groups
    _DM.create_groups_from_config(pconfig, verbose=(verbose and (_DM.rank == 0)))

    # get comm roots:
    global _COMM_ROOTS
    for gname in get_comm_names():
        rank = _DM.rank
        for grp in _DM._group_ranks[gname]:
            if rank in grp:
                _COMM_ROOTS[gname] = min(grp)

    if verbose:
        import torch

        for rank in range(_DM.world_size):
            if rank == _DM.rank:
                print(f"{rank}: groups:")
                for gname in get_comm_names():
                    print(f"\t{gname}: {_DM._group_ranks[gname]}, root={_COMM_ROOTS[gname]}")
            torch.distributed.barrier(device_ids=[_DM.local_rank])

    return get_size("model")
