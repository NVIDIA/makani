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

import torch
import torch.distributed as dist
from parameterized import parameterized

from makani.distributed import comm
from makani.training import Trainer, EnsembleTrainer, StochasticTrainer, AutoencoderTrainer
from makani.utils.argument_parser import get_default_argument_parser

from ..test_trainers import init_params
from ..testutils import disable_tf32, set_seed, init_hdf5_dataset
from .distributed_helpers import (
    _init_grid,
    make_shared_tmpdir,
    reduce_success,
    remove_shared_tmpdir,
    sync_and_barrier,
    teardown_comms,
)

# a small dataset, so that an epoch is a few steps: 8 training samples over two years, 4 validation samples
_SAMPLES_PER_YEAR = 4


class TestDistributedTrainerCheckpoints(unittest.TestCase):
    """
    The checkpoints a trainer writes at the end of its epochs, on all ranks of a distributed run.

    Run with e.g.::

        GRID_H=2 torchrun --nproc_per_node=4 -m pytest tests/distributed/test_distributed_trainer_checkpoints.py

    which gives two model ranks and two data-parallel ranks.
    """

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

        # one dataset and one experiment directory for all ranks, as on a shared file system
        cls.tmpdir = make_shared_tmpdir("makani_trainer_checkpoints_", cls.device)
        paths = [None]
        if cls.world_rank == 0:
            paths = [init_hdf5_dataset(os.path.join(cls.tmpdir, "data"), num_samples_per_year=_SAMPLES_PER_YEAR)]
        if dist.is_initialized():
            dist.broadcast_object_list(paths, src=0)
        cls.train_path, cls.n_train_samples, cls.valid_path, cls.n_eval_samples, cls.stats_path, _, _ = paths[0]

        sync_and_barrier()

    @classmethod
    def tearDownClass(cls):
        remove_shared_tmpdir(cls.tmpdir)
        teardown_comms()

    def _params(self, name, checkpoint_mode):
        exp_path = os.path.join(self.tmpdir, name)
        if self.world_rank == 0:
            os.makedirs(os.path.join(exp_path, "training_checkpoints"))
        sync_and_barrier()

        params = init_params(
            exp_path,
            self.train_path,
            self.valid_path,
            self.stats_path,
            batch_size=1,
            ensemble_size=2,
            stochastic_size=2,
            n_history=0,
            n_future=0,
            normalization="zscore",
            num_data_workers=0,
        )
        params.multifiles = True
        params.dhours = 365 * 24 // _SAMPLES_PER_YEAR
        params.n_train_samples = self.n_train_samples
        params.n_eval_samples = self.n_eval_samples
        params.print_timings_frequency = 0
        params["save_checkpoint"] = checkpoint_mode
        params["max_epochs"] = 2
        # the options of the training script the trainers read when distributed, at their defaults
        parser = get_default_argument_parser()
        for name in ("parameters_reduction_buffer_count", "disable_ddp", "enable_grad_anomaly_detection"):
            params[name] = parser.get_default(name)
        return params

    @parameterized.expand(
        [
            (trainer_handle, checkpoint_mode)
            for trainer_handle in (Trainer, EnsembleTrainer, StochasticTrainer, AutoencoderTrainer)
            for checkpoint_mode in ("flexible", "legacy")
        ],
        skip_on_empty=True,
    )
    def test_best_checkpoint_only_where_the_loss_improves(self, trainer_handle, checkpoint_mode):
        """
        Two epochs, the second with a worse validation loss: every rank has to skip the best checkpoint then, since
        saving takes collectives over the model ranks and only the first data-parallel rank of each model group writes.
        A decision that differs between ranks, or a collective some of them skip, hangs the run at the end of the second
        epoch; the best checkpoint is that of the first.
        """
        disable_tf32()
        set_seed(333)

        params = self._params(f"{trainer_handle.__name__}_{checkpoint_mode}", checkpoint_mode)
        trainer = trainer_handle(params, self.world_rank, device=self.device)

        # the validation loss of the two epochs, the same on every rank
        validate = trainer.validate_one_epoch
        losses = iter([1.0, 2.0])

        def validate_with_loss(*args, **kwargs):
            valid_time, viz_time, logs = validate(*args, **kwargs)
            logs["base"]["validation loss"] = next(losses)
            return valid_time, viz_time, logs

        trainer.validate_one_epoch = validate_with_loss
        trainer.train()

        # the writers of the checkpoints check them, and every rank asserts the verdict of all of them, so that a
        # failure fails the test on all ranks instead of leaving the others waiting in a collective
        sync_and_barrier()
        success = trainer.epoch == 2
        if (comm.get_rank("data") == 0) and (checkpoint_mode == "legacy" or comm.get_rank("model") == 0):
            mp_rank = comm.get_rank("model") if checkpoint_mode == "legacy" else 0
            try:
                epochs = [
                    torch.load(path.format(mp_rank=mp_rank), map_location="cpu", weights_only=False)["epoch"]
                    for path in (params.best_checkpoint_path, params.checkpoint_path)
                ]
            except (OSError, KeyError) as error:
                epochs = [None, None]
                print(f"rank {self.world_rank}: cannot read the checkpoints: {error}")
            success = success and (epochs == [1, 2])
            if not success:
                print(f"rank {self.world_rank}: epoch {trainer.epoch}, best and latest checkpoints of epochs {epochs}")
        self.assertTrue(reduce_success(success, self.device))


if __name__ == "__main__":
    unittest.main()
