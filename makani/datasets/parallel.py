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

from typing import Any, Dict, List, Optional

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
    """A duplicate of ``MPI.COMM_WORLD`` if mpi4py is installed, a :class:`SerialComm` otherwise."""
    if MPI is None:
        return SerialComm()
    return MPI.COMM_WORLD.Dup()


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
    """The GPU of this rank on its node, round robin, or 0 without GPUs."""
    import torch

    return comm.Get_rank() % max(torch.cuda.device_count(), 1)


def reduction_mesh_shape(reduction_group_size: int, world_size: int) -> List[int]:
    """
    The device mesh ``[reduction, tree]`` of the parallel reductions.

    The reduction groups take the largest size that divides the world size and does not exceed
    ``reduction_group_size``, so that any number of ranks works, also a single one.
    """
    group = max(d for d in range(1, min(reduction_group_size, world_size) + 1) if world_size % d == 0)
    return [group, world_size // group]
