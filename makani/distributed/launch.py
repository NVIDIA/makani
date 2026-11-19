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
Where this process sits in its launch, from the environment its launcher set.

Launchers export the number of processes they started, the rank of each and its rank on its node,
each under its own names. :func:`detect_launch` reads them in a fixed order, so that every part of
makani agrees on one answer whatever started it:

1. Open MPI's ``mpirun``: ``OMPI_COMM_WORLD_SIZE``, ``OMPI_COMM_WORLD_RANK``,
   ``OMPI_COMM_WORLD_LOCAL_RANK``.
2. A Slurm step, as ``srun`` starts it: ``SLURM_STEP_NUM_TASKS`` (or ``SLURM_NTASKS``),
   ``SLURM_PROCID``, ``SLURM_LOCALID``. Only inside a step, which ``SLURM_STEP_ID`` marks: in the
   batch script itself, Slurm's task variables describe the job, not the one process running there.
3. PMI, which MPICH's and Intel MPI's launchers set: ``PMI_SIZE``, ``PMI_RANK``, and
   ``MPI_LOCALRANKID`` or ``PMI_LOCAL_RANK``.
4. Set by hand or by torchrun: ``WORLD_SIZE``, ``RANK`` (or ``WORLD_RANK``), ``LOCAL_RANK``.

Without any of them, the process is a launch of its own: one process, rank 0.
"""

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class Launch:
    """
    The launch a process belongs to.

    Attributes
    ----------
    size : int
        The number of processes the launcher started.
    rank : int
        The rank of this process among them.
    local_rank : int or None
        Its rank among the processes on its node; None if the launcher started several processes
        and does not say.
    launcher : str
        What started it: ``"openmpi"``, ``"slurm"``, ``"pmi"``, ``"env"`` or ``"none"``.
    """

    size: int = 1
    rank: int = 0
    local_rank: Optional[int] = 0
    launcher: str = "none"


def _first(environ: Mapping[str, str], names: Sequence[str]) -> Optional[int]:
    """The first of the variables that is set, as an integer."""
    for name in names:
        value = environ.get(name, "").strip()
        if value:
            try:
                return int(value)
            except ValueError as error:
                raise ValueError(f"{name}={value!r} is not an integer") from error
    return None


#: The variables of each launcher, in the order they are queried: size, rank, local rank.
_LAUNCHERS = (
    ("openmpi", ("OMPI_COMM_WORLD_SIZE",), ("OMPI_COMM_WORLD_RANK",), ("OMPI_COMM_WORLD_LOCAL_RANK",)),
    ("slurm", ("SLURM_STEP_NUM_TASKS", "SLURM_NTASKS"), ("SLURM_PROCID",), ("SLURM_LOCALID",)),
    ("pmi", ("PMI_SIZE",), ("PMI_RANK",), ("MPI_LOCALRANKID", "PMI_LOCAL_RANK")),
    ("env", ("WORLD_SIZE",), ("RANK", "WORLD_RANK"), ("LOCAL_RANK",)),
)


def detect_launch(environ: Optional[Mapping[str, str]] = None) -> Launch:
    """
    The launch this process belongs to, from the variables of the first launcher that set them.

    Parameters
    ----------
    environ : mapping, optional
        The environment to read; ``os.environ`` by default.

    Returns
    -------
    Launch
        The size, rank and local rank, and which launcher they came from.

    Raises
    ------
    ValueError
        If a launcher sets a size but no rank, or a rank outside the size.
    """
    environ = os.environ if environ is None else environ

    for launcher, size_names, rank_names, local_names in _LAUNCHERS:
        # a Slurm batch script is no step, see the module docstring
        if launcher == "slurm" and not environ.get("SLURM_STEP_ID", "").strip():
            continue
        size = _first(environ, size_names)
        if size is None:
            continue

        rank = _first(environ, rank_names)
        if rank is None:
            if size > 1:
                raise ValueError(f"{size_names[0]}={size} is set, but none of {', '.join(rank_names)}")
            rank = 0
        if not 0 <= rank < size:
            raise ValueError(f"{launcher}: rank {rank} is outside of the {size} processes launched")

        local_rank = _first(environ, local_names)
        if (local_rank is None) and (size == 1):
            local_rank = 0

        return Launch(size=size, rank=rank, local_rank=local_rank, launcher=launcher)

    return Launch()
