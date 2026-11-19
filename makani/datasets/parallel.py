# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
MPI for the dataset scripts, where it is available.

The scripts run in parallel under ``mpirun`` when mpi4py is installed (``pip install makani[datasets-mpi]``),
and serially with plain ``python`` otherwise: :func:`get_comm` then returns a :class:`SerialComm`, a
communicator of one rank. Shared HDF5 files are written with the ``mpio`` driver only with more than
one rank, see :func:`h5_kwargs`, so a serial run works with any h5py, including the pip wheels, which
are built without parallel HDF5.
"""

import os
import socket
from typing import Any, Dict, List, Optional, Tuple

from makani.distributed.launch import detect_launch

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class SerialComm:
    """
    The communicator of a run without MPI: a single rank.

    Provides the part of the ``mpi4py.MPI.Comm`` interface the dataset scripts use, so that they
    run the same code with and without MPI.
    """

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def Barrier(self):
        pass

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def allgather(self, obj: Any) -> list:
        return [obj]

    def Dup(self) -> "SerialComm":
        return self

    def Free(self):
        pass


def get_comm():
    """
    A duplicate of ``MPI.COMM_WORLD`` if mpi4py is installed, a :class:`SerialComm` otherwise.

    Refuses a launch of several processes, see :func:`makani.distributed.launch.detect_launch`, that
    MPI does not connect: without mpi4py, or with an mpi4py built against another MPI than the
    launcher's, every process would take itself for the only rank, read all of the data and write the
    same files.
    """
    launch = detect_launch()
    comm = SerialComm() if MPI is None else MPI.COMM_WORLD.Dup()
    if (launch.size > 1) and (comm.Get_size() != launch.size):
        if MPI is None:
            reason = "mpi4py is not installed; install it with `pip install makani[datasets-mpi]`"
        else:
            reason = (
                f"MPI connects {comm.Get_size()} of them; mpi4py may be built against another MPI than the launcher's"
            )
        raise RuntimeError(f"The script was launched by {launch.launcher} as {launch.size} processes, but {reason}.")
    return comm


def is_parallel(comm) -> bool:
    """Whether the run has more than one rank."""
    return (comm is not None) and (comm.Get_size() > 1)


def h5_kwargs(comm) -> Dict[str, Optional[Any]]:
    """
    The keyword arguments of ``h5py.File`` for a file that all ranks of ``comm`` open together.

    The ``mpio`` driver with more than one rank, which needs h5py built against parallel HDF5;
    nothing otherwise, so that a serial run works with any h5py.
    """
    if not is_parallel(comm):
        return {}

    import h5py

    if not h5py.get_config().mpi:
        raise RuntimeError(
            f"Writing a file from {comm.Get_size()} ranks needs h5py built against parallel HDF5, "
            "which this h5py is not. Run with a single rank, or install h5py with MPI support."
        )
    return {"driver": "mpio", "comm": comm}


def local_device_index(comm) -> int:
    """
    The GPU of this rank on its node, or 0 without GPUs.

    The rank on its node, if the launcher says, see :func:`makani.distributed.launch.detect_launch`;
    the rank in ``comm`` otherwise, round robin over the GPUs.
    """
    import torch

    local_rank = detect_launch().local_rank
    if local_rank is None:
        local_rank = comm.Get_rank()
    return local_rank % max(torch.cuda.device_count(), 1)


def reduction_mesh_shape(reduction_group_size: int, world_size: int) -> List[int]:
    """
    The device mesh ``[reduction, tree]`` of the parallel reductions.

    The reduction groups take the largest size that divides the world size and does not exceed
    ``reduction_group_size``, so that any number of ranks works, also a single one.
    """
    group = max(d for d in range(1, min(reduction_group_size, world_size) + 1) if world_size % d == 0)
    return [group, world_size // group]


def init_torch_distributed(comm, reduction_group_size: int) -> Tuple[Any, Any]:
    """
    Wire torch.distributed up over the ranks of ``comm``, for the parallel reductions of the
    statistics scripts.

    The ranks meet at rank 0's host, on ``MASTER_PORT``, 29500 unless it is set, and reduce over
    NCCL with one GPU per rank where there are GPUs, over gloo on the CPU otherwise. Also on a
    single rank without MPI, so that the scripts run the same code either way.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm or SerialComm
        The ranks, see :func:`get_comm`.
    reduction_group_size : int
        The requested size of the reduction groups, see :func:`reduction_mesh_shape`.

    Returns
    -------
    device : torch.device
        The device of this rank.
    mesh : torch.distributed.DeviceMesh
        The ``[reduction, tree]`` mesh of the reductions.
    """
    import torch
    import torch.distributed as dist

    rank, size = comm.Get_rank(), comm.Get_size()
    local_rank = local_device_index(comm)

    os.environ["MASTER_ADDR"] = comm.bcast(socket.gethostname(), root=0)
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    dist.init_process_group(
        backend="nccl" if device.type == "cuda" else "gloo",
        init_method="env://",
        world_size=size,
        rank=rank,
        device_id=device if device.type == "cuda" else None,
    )
    mesh = dist.init_device_mesh(
        device_type=device.type,
        mesh_shape=reduction_mesh_shape(reduction_group_size, size),
        mesh_dim_names=["reduction", "tree"],
    )
    return device, mesh


def shutdown_torch_distributed(device):
    """Wait for all ranks, and take torch.distributed down again, see :func:`init_torch_distributed`."""
    import torch.distributed as dist

    dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
    dist.destroy_process_group()
