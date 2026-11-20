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

import unittest

import torch
import torch.distributed as dist
from parameterized import parameterized

from makani.datasets.helpers import binary_reduce, collective_reduce
from makani.datasets.parallel import reduction_mesh_shape

from .distributed_helpers import _init_grid, reduce_success, sync_and_barrier, teardown_comms


def _data(rank):
    """The samples (samples, channels) of a rank: a different number on every rank, with different statistics."""
    generator = torch.Generator().manual_seed(100 + rank)
    return torch.randn(5 + rank, 3, dtype=torch.float64, generator=generator) + rank


def _stats(x, device):
    """Statistics of samples ``x`` (samples, channels) in the layout of the statistics scripts."""
    x = x.to(device)
    counts = torch.full((x.shape[1],), float(x.shape[0]), dtype=torch.float64, device=device)
    mean = x.mean(0)
    m2 = ((x - mean) ** 2).sum(0)
    return {
        "meanvar": {"type": "meanvar", "counts": counts, "values": torch.stack([mean, m2]).reshape(2, 1, -1, 1, 1)},
        "mins": {"type": "min", "counts": counts.clone(), "values": x.min(0).values.reshape(1, -1, 1, 1)},
        "maxs": {"type": "max", "counts": counts.clone(), "values": x.max(0).values.reshape(1, -1, 1, 1)},
    }


def _matches(stats, expected):
    return all(
        torch.allclose(stats[name]["values"], entry["values"], rtol=1e-12, atol=1e-12)
        and torch.equal(stats[name]["counts"], entry["counts"])
        for name, entry in expected.items()
    )


class TestDistributedDatasetReduction(unittest.TestCase):
    """
    The parallel reduction of the statistics scripts (``compute_stats``, ``compute_spectra``,
    ``compute_hydrostatic_balance_climatology``) over all ranks of the run: a Welford reduction within
    the reduction groups, then a binary tree across them onto rank 0. Every rank holds different
    samples, and the result has to be the statistics of all of them, for any number of ranks and any
    requested group size.

    Run with e.g.::

        GRID_H=2 GRID_W=2 mpirun -n 4 pytest tests/distributed/test_distributed_dataset_reduction.py
    """

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    @classmethod
    def tearDownClass(cls):
        sync_and_barrier()
        teardown_comms()

    @parameterized.expand([(1,), (2,), (3,), (4,), (8,)])
    def test_reduction_matches_the_statistics_of_all_data(self, reduction_group_size):
        # the mesh of the statistics scripts, over all ranks
        mesh = dist.init_device_mesh(
            self.device.type,
            mesh_shape=reduction_mesh_shape(reduction_group_size, self.world_size),
            mesh_dim_names=["reduction", "tree"],
        )
        reduction_group = mesh.get_group("reduction")

        # within the groups: every rank of a group holds the statistics of the group
        stats = collective_reduce(_stats(_data(self.world_rank), self.device), reduction_group)
        group_data = torch.cat([_data(rank) for rank in dist.get_process_group_ranks(reduction_group)])
        ok = _matches(stats, _stats(group_data, self.device))

        # across the groups: rank 0 holds the statistics of all data
        if dist.get_rank(reduction_group) == 0:
            stats = binary_reduce(stats, mesh.get_group("tree"), self.device)
        if self.world_rank == 0:
            all_data = torch.cat([_data(rank) for rank in range(self.world_size)])
            ok = ok and _matches(stats, _stats(all_data, self.device))

        self.assertTrue(reduce_success(ok, self.device), f"reduction_group_size={reduction_group_size}")

    @parameterized.expand([(3, 1), (6, 1), (6, 2)])
    def test_reduction_on_ranks_without_a_partner(self, num_ranks, reduction_group_size):
        # the CI grids have a power of two ranks, so the tree reduction across the groups always
        # pairs them up; on the first 3 or 6 ranks of the run, a tree of 3 groups leaves one without
        if self.world_size < num_ranks:
            self.skipTest(f"needs {num_ranks} ranks")

        # the groups of the mesh [reduction, tree] over the first ranks, as init_device_mesh lays them
        # out; every rank creates every group, also those it is not in
        group, tree = reduction_mesh_shape(reduction_group_size, num_ranks)
        layout = torch.arange(num_ranks).reshape(group, tree)
        reduction_groups = [dist.new_group(layout[:, j].tolist()) for j in range(tree)]
        tree_group = dist.new_group(layout[0, :].tolist())

        ok = True
        if self.world_rank < num_ranks:
            i, j = (layout == self.world_rank).nonzero()[0].tolist()
            stats = collective_reduce(_stats(_data(self.world_rank), self.device), reduction_groups[j])
            if i == 0:
                stats = binary_reduce(stats, tree_group, self.device)
            if self.world_rank == 0:
                all_data = torch.cat([_data(rank) for rank in range(num_ranks)])
                ok = _matches(stats, _stats(all_data, self.device))

        self.assertTrue(
            reduce_success(ok, self.device), f"num_ranks={num_ranks}, reduction_group_size={reduction_group_size}"
        )


if __name__ == "__main__":
    unittest.main()
