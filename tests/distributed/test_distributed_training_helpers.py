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

import os
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.distributed as dist

from makani.distributed import comm
from makani.distributed.helpers import AVG, SUM, get_sharing, sync_params
from makani.distributed.mappings import init_gradient_reduction_hooks
from makani.training.training_helpers import normalize_weights, save_as_best_checkpoint

from .distributed_helpers import (
    _init_grid,
    make_shared_tmpdir,
    reduce_success,
    remove_shared_tmpdir,
    sync_and_barrier,
    teardown_comms,
)

_NUM_FEATURES = 6


class _TinyModel(nn.Module):
    """
    Linear model with a parameter of each kind of sharing. The loss is linear in the weights, so the
    local gradient of each weight is its input.

    * ``shared``: without annotation, shared over the whole model group, summed over the spatial
      ranks and averaged over the matmul ranks.
    * ``local``: sharded over h, like a spectral weight, summed over w and averaged over matmul.
    * ``summed``: summed over matmul, like the norms of the heads of an attention split over matmul.
    * ``listed``: annotated with the list form of earlier versions, ``["spatial"]``.

    With ``all_shared``, no parameter is annotated.
    """

    def __init__(self, all_shared=False):
        super().__init__()
        self.shared = nn.Parameter(torch.zeros(_NUM_FEATURES, dtype=torch.float64))
        self.local = nn.Parameter(torch.zeros(_NUM_FEATURES, dtype=torch.float64))
        self.summed = nn.Parameter(torch.zeros(_NUM_FEATURES, dtype=torch.float64))
        self.listed = nn.Parameter(torch.zeros(_NUM_FEATURES, dtype=torch.float64))
        if not all_shared:
            self.local.is_shared_mp = {"w": "sum", "matmul": "avg"}
            self.local.sharded_dims_mp = ["h"]
            self.summed.is_shared_mp = {"matmul": "sum"}
            self.listed.is_shared_mp = ["spatial"]

    def forward(self, xs):
        return sum((p * x).sum() for p, x in zip(self.parameters(), xs))


class _ShardedModel(nn.Module):
    """
    A parameter of each kind of sharding, for the weight normalization: each rank holds a shard, and
    the norm of a parameter is that of the whole tensor, over the shards of all the groups it is split
    across.

    * ``replicated``: not sharded, each rank normalizes its own tensor.
    * ``h``: sharded over h, like a spectral weight.
    * ``spatial``: sharded over h and w, like a position embedding.
    * ``matmul``: sharded over the matmul ranks, like the weights of a distributed MLP.
    """

    _SHARDING = {"replicated": [], "h": ["h"], "spatial": [None, "h", "w"], "matmul": ["matmul", None]}

    def __init__(self):
        super().__init__()
        for name, sharded_dims in self._SHARDING.items():
            param = nn.Parameter(torch.zeros(_NUM_FEATURES, dtype=torch.float64))
            param.sharded_dims_mp = sharded_dims
            self.register_parameter(name, param)


class TestDistributedTrainingHelpers(unittest.TestCase):
    """
    Weight synchronization (``sync_params``) and gradient reduction (``init_gradient_reduction_hooks``)
    across the data and model groups. Every rank starts from different values, and the expected
    result is replayed from the values of all ranks and the group layout, so the tests do not
    depend on how the groups are laid out.

    Run with e.g.::

        GRID_H=2 GRID_W=2 GRID_E=2 mpirun -n 8 pytest tests/distributed/test_distributed_training_helpers.py
    """

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    @classmethod
    def tearDownClass(cls):
        sync_and_barrier()
        teardown_comms()

    def _rank_values(self, offset):
        # distinct per rank and per element
        return torch.arange(_NUM_FEATURES, dtype=torch.float64, device=self.device) + 10.0 * self.world_rank + offset

    def _all_gather(self, tensor):
        if not dist.is_initialized():
            return tensor.unsqueeze(0).clone()
        tlist = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tlist, tensor.contiguous())
        return torch.stack(tlist, dim=0)

    def _group_keys(self, name):
        # world rank of each rank's group root; a rank in a group of size 1 is its own root
        root = comm.get_root(name) if comm.get_size(name) > 1 else self.world_rank
        return self._all_gather(torch.tensor([root], dtype=torch.long, device=self.device)).flatten()

    def _replay(self, values, name, op):
        """Apply a broadcast, mean or sum over the group ``name`` to the values of all ranks."""
        keys = self._group_keys(name)
        result = values.clone()
        for root in keys.unique().tolist():
            members = keys == root
            if op == "broadcast":
                result[members] = values[root]
            elif op == "mean":
                result[members] = values[members].mean(dim=0)
            elif op == "sum":
                result[members] = values[members].sum(dim=0)
            else:
                raise ValueError(f"Unknown op {op}")
        return result

    def _check(self, name, actual, expected, verbose=False):
        ok = torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        if verbose and not ok:
            print(f"rank {self.world_rank}: {name} {actual.tolist()} vs expected {expected.tolist()}")
        self.assertTrue(reduce_success(ok, self.device), name)

    def _test_sync_params(self, mode, verbose=False):
        model = _TinyModel().to(self.device)
        with torch.no_grad():
            for i, p in enumerate(model.parameters()):
                p.copy_(self._rank_values(0.5 * i))

        initial = {pname: self._all_gather(p.detach()) for pname, p in model.named_parameters()}

        sync_params(model, mode=mode)

        for pname, param in model.named_parameters():
            # sync_params goes over the data group first, then over each group the parameter is shared across,
            # whichever its gradient is reduced with
            expected = self._replay(initial[pname], "data", mode)
            for group in get_sharing(param):
                expected = self._replay(expected, group, mode)
            self._check(f"{mode} {pname}", param.detach(), expected[self.world_rank], verbose=verbose)

    def test_sync_params_broadcast(self, verbose=False):
        self._test_sync_params("broadcast", verbose=verbose)

    def test_sync_params_mean(self, verbose=False):
        self._test_sync_params("mean", verbose=verbose)

    def _test_gradient_reduction(self, all_shared, verbose=False):
        model = _TinyModel(all_shared=all_shared).to(self.device)
        model_ddp = init_gradient_reduction_hooks(model, self.device, verbose=False)

        xs = [self._rank_values(1.0 + i) for i, _ in enumerate(model.parameters())]
        model_ddp(xs).backward()

        # the local gradient of each weight is its input; it is averaged over the data group, then summed or
        # averaged over each group the parameter is shared over, as its annotation says
        for (pname, param), x in zip(model.named_parameters(), xs):
            expected = self._replay(self._all_gather(x), "data", "mean")
            for group, reduction in get_sharing(param).items():
                expected = self._replay(expected, group, "sum" if reduction == SUM else "mean")
            self._check(f"gradient {pname}", param.grad, expected[self.world_rank], verbose=verbose)

    def test_normalize_weights(self, verbose=False):
        model = _ShardedModel().to(self.device)
        with torch.no_grad():
            for i, p in enumerate(model.parameters()):
                p.copy_(self._rank_values(0.5 * i))

        initial = {pname: p.detach().clone() for pname, p in model.named_parameters()}

        eps = 1e-5
        normalize_weights(model, eps=eps)

        # each parameter is divided by the norm of the whole tensor: the sum of squares of the shards, summed
        # over every group the parameter is split across
        for pname, param in model.named_parameters():
            sumsq = self._all_gather(initial[pname].square().sum().reshape(1))
            for group in model._SHARDING[pname]:
                if group is not None:
                    sumsq = self._replay(sumsq, group, "sum")
            expected = initial[pname] / (sumsq[self.world_rank].sqrt() + eps)
            self._check(f"normalized {pname}", param.detach(), expected, verbose=verbose)

    def test_the_sharing_of_the_model(self):
        model = _TinyModel()
        self.assertEqual(get_sharing(model.shared), {"h": SUM, "w": SUM, "matmul": AVG})
        self.assertEqual(get_sharing(model.local), {"w": SUM, "matmul": AVG})
        self.assertEqual(get_sharing(model.summed), {"matmul": SUM})
        # the list form of earlier versions: a sum over the spatial ranks
        self.assertEqual(get_sharing(model.listed), {"h": SUM, "w": SUM})

    def test_gradient_reduction_all_shared(self, verbose=False):
        # no parameter annotated: each is shared over the whole model group
        self._test_gradient_reduction(all_shared=True, verbose=verbose)

    def test_gradient_reduction_mixed(self, verbose=False):
        self._test_gradient_reduction(all_shared=False, verbose=verbose)

    def _decisions(self, *args):
        # the decision of every rank
        save = save_as_best_checkpoint(*args, self.device)
        return self._all_gather(torch.tensor([int(save)], dtype=torch.int64, device=self.device)).flatten().tolist()

    def test_best_checkpoint_decision_is_the_same_on_every_rank(self):
        """
        Saving the best checkpoint takes collectives over the model ranks, so every rank has to decide alike. A flexible
        checkpoint is written as the file of model rank 0 only: a rank that looked for the file of its own model rank
        would find none and save each epoch, while the others save only when the loss improves, and the ranks hang in
        different collectives at the first epoch whose loss does not.
        """
        tmpdir = make_shared_tmpdir("makani_best_checkpoint_", self.device)
        try:
            path = os.path.join(tmpdir, "best_ckpt_mp{mp_rank}.tar")
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            # no best checkpoint yet: save, whatever the loss
            self.assertEqual(self._decisions(path, 2.0, 1.0), [1] * world_size)

            # the best checkpoint as flexible mode writes it, the file of model rank 0 only
            if self.world_rank == 0:
                with open(path.format(mp_rank=0), "w") as f:
                    f.write("checkpoint")
            sync_and_barrier()
            self.assertEqual(self._decisions(path, 2.0, 1.0), [0] * world_size, "a worse loss saves nothing")
            self.assertEqual(self._decisions(path, 1.0, 1.0), [1] * world_size, "a loss as good saves everywhere")

            # a file system that shows the file to some ranks only: they follow rank 0
            with mock.patch("os.path.isfile", return_value=(self.world_rank == 0)):
                self.assertEqual(self._decisions(path, 2.0, 1.0), [0] * world_size)
            with mock.patch("os.path.isfile", return_value=(self.world_rank != 0)):
                self.assertEqual(self._decisions(path, 2.0, 1.0), [1] * world_size)
        finally:
            remove_shared_tmpdir(tmpdir)


if __name__ == "__main__":
    unittest.main()
