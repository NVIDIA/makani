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


import sys
import os
import tempfile
import unittest
from parameterized import parameterized

import h5py as h5
import numpy as np

import torch

import torch_harmonics.distributed as thd

from makani.utils import comm
from torch_harmonics.distributed import compute_split_shapes

from makani.utils import MetricsHandler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .distributed_helpers import (
    _init_grid_module, _copy_grid_state,
    _split_helper, get_default_parameters, set_image_shape,
)
from ..testutils import disable_tf32, set_seed, compare_arrays


# Comm groups to override to size 1 around the serial reference build/run.
_SERIAL_OVERRIDE = dict(
    model=1, spatial=1, matmul=1,
    h=1, w=1, fin=1, fout=1,
    ensemble=1, batch=1,
)

# Module-level MPI communicator (duplicated from MPI.COMM_WORLD once per
# process, freed in tearDownModule).
_MPI_COMM = None


def setUpModule():
    """Initialise MPI + comm groups once for the whole module (see
    tests_distributed_model.setUpModule for the rationale). Reads ``GRID_B``
    in addition to the usual H/W/FIN/FOUT/E."""
    global _MPI_COMM
    from mpi4py import MPI
    _MPI_COMM = MPI.COMM_WORLD.Dup()
    _init_grid_module()


def tearDownModule():
    comm.cleanup()
    if _MPI_COMM is not None:
        # mpi_comm is an mpi4py Intracomm (returned by MPI.COMM_WORLD.Dup());
        # its disposal API is Free(), called explicitly rather than relying on
        # interpreter shutdown.
        _MPI_COMM.Free()


_metric_handler_params = [
    ("equiangular", 4, 16, 3, "mean"),
    #("equiangular", 4, 16, 3, "sum"),
]

class TestDistributedMetricHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls, path="/tmp"):
        # Shadow module-level state (MPI handle + grid state) onto the class
        # so tests can keep using ``self.mpi_comm`` / ``self.h_group`` etc.
        cls.mpi_comm = _MPI_COMM
        cls.mpi_comm_rank = _MPI_COMM.Get_rank()
        cls.mpi_comm_size = _MPI_COMM.Get_size()
        _copy_grid_state(cls)

        set_seed(333)

        # create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)

        return

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _split_helper(self, tensor):
        with torch.no_grad():
            # split in W
            tensor_local = _split_helper(tensor, dim=-1, group=self.w_group)

            # split in H
            tensor_local = _split_helper(tensor_local, dim=-2, group=self.h_group)

            # split in E
            if tensor.dim() == 6:
                tensor_local = _split_helper(tensor_local, dim=2, group=self.e_group)

            # split in B
            tensor_local = _split_helper(tensor_local, dim=1, group=self.b_group)

        return tensor_local

    def setUp(self):

        disable_tf32()

        self.params = get_default_parameters()
        self.params["dhours"] = 1

        # set image shape, local/crop shapes, offsets, and resampled shapes
        # consistently — see set_image_shape for the rationale.
        set_image_shape(self.params, h=36, w=72)

        return

    @parameterized.expand(_metric_handler_params, skip_on_empty=True)
    def test_metric_handler_aggregation(self, grid_type, batch_size, ensemble_size, num_rollout_steps, bred, verbose=False):
        # create dummy climatology
        num_steps = 4
        num_channels = len(self.params.channel_names)
        clim = torch.zeros(1, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)

        # local — every comm-size-based branch inside MetricsHandler resolves
        # to the serial path while this override scope is active.
        with comm.override_sizes(**_SERIAL_OVERRIDE):
            self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
            self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
            self.params.img_local_offset_x = 0
            self.params.img_local_offset_y = 0

            # set batch size and ensemble size:
            self.params.batch_size = batch_size
            self.params.ensemble_size = ensemble_size

            metric_handler_local = MetricsHandler(self.params,
                                                  clim,
                                                  num_rollout_steps,
                                                  self.device,
                                                  l1_var_names=self.params.channel_names,
                                                  rmse_var_names=self.params.channel_names,
                                                  acc_var_names=self.params.channel_names,
                                                  crps_var_names=self.params.channel_names,
                                                  spread_var_names=self.params.channel_names,
                                                  ssr_var_names=self.params.channel_names,
                                                  rh_var_names=self.params.channel_names,
                                                  wb2_compatible=False)
            metric_handler_local.initialize_buffers()
            metric_handler_local.zero_buffers()

            inplist = [torch.randn((num_rollout_steps, batch_size, ensemble_size, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y),
                                   dtype=torch.float32, device=self.device) for _ in range(num_steps)]
            tarlist = [torch.randn((num_rollout_steps, batch_size, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y),
                                   dtype=torch.float32, device=self.device) for _ in range(num_steps)]

            for inp, tar in zip(inplist, tarlist):
                for idt in range(num_rollout_steps):
                    inpp = inp[idt, ...]
                    tarp = tar[idt, ...]

                    # dummy loss
                    loss = torch.tensor(1., dtype=torch.float32, device=self.device)

                    # update metric handler
                    metric_handler_local.update(inpp, tarp, loss, idt)

            # finalize
            logs_local = metric_handler_local.finalize()

        # wait for everybody
        self.mpi_comm.Barrier()

        # distributed
        #set up shapes
        h_shapes = compute_split_shapes(self.params.img_shape_x, comm.get_size("h"))
        h_off = [0] + np.cumsum(h_shapes).tolist()[:-1]
        w_shapes = compute_split_shapes(self.params.img_shape_y, comm.get_size("w"))
        w_off =	[0] + np.cumsum(w_shapes).tolist()[:-1]

        self.params.img_local_shape_x = h_shapes[comm.get_rank("h")]
        self.params.img_local_offset_x = h_off[comm.get_rank("h")]
        self.params.img_local_shape_y = w_shapes[comm.get_rank("w")]
        self.params.img_local_offset_y = w_off[comm.get_rank("w")]

        # split tensors
        inplist_split = [self._split_helper(tensor) for tensor in inplist]
        tarlist_split = [self._split_helper(tensor) for tensor in tarlist]

        # init metric handler
        metric_handler_dist = MetricsHandler(self.params,
                                             clim,
                                             num_rollout_steps,
                                             self.device,
                                             l1_var_names=self.params.channel_names,
                                             rmse_var_names=self.params.channel_names,
                                             acc_var_names=self.params.channel_names,
                                             crps_var_names=self.params.channel_names,
                                             spread_var_names=self.params.channel_names,
                                             ssr_var_names=self.params.channel_names,
                                             rh_var_names=self.params.channel_names,
                                             wb2_compatible=False)
        metric_handler_dist.initialize_buffers()
        metric_handler_dist.zero_buffers()

        for inp, tar in zip(inplist_split, tarlist_split):
            for idt in range(num_rollout_steps):
                inpp = inp[idt, ...]
                tarp = tar[idt, ...]

                # dummy loss
                loss = torch.tensor(1., dtype=torch.float32, device=self.device)

                # update metric handler
                metric_handler_dist.update(inpp, tarp, loss, idt)

        # finalize
        logs_dist = metric_handler_dist.finalize()

        # extract dicts
        metrics_local = logs_local["metrics"]
        metrics_dist = logs_dist["metrics"]

        # compare scalar metrics
        for key in  metrics_local.keys():
            if key == "rollouts":
                continue
            val_local = metrics_local[key]
            val_dist = metrics_dist[key]
            if verbose:
                print(f"log metric {key}: local={val_local}, dist={val_dist}")
            self.assertTrue(compare_arrays(f"log metric {key}", np.asarray(val_local), np.asarray(val_dist), verbose=verbose))

        # compare rollouts
        rollouts_local = logs_local["metrics"]["rollouts"]
        rollouts_dist = logs_dist["metrics"]["rollouts"]

        # aggregate table into data
        data_local = []
        for row in rollouts_local.data:
            data_local.append(row[-1])
        data_local = np.array(data_local)

        data_dist = []
        for row in rollouts_dist.data:
            data_dist.append(row[-1])
        data_dist = np.array(data_dist)

        with self.subTest(desc="rollouts"):
            self.assertTrue(compare_arrays("rollouts", data_dist, data_local, verbose=verbose))

        # save output files and compare
        if comm.get_world_rank() == 0:
            metric_handler_local.save(os.path.join(self.tmpdir.name, "metrics_local.h5"))
            metric_handler_dist.save(os.path.join(self.tmpdir.name, "metrics_dist.h5"))

            file_local = h5.File(os.path.join(self.tmpdir.name, "metrics_local.h5"), "r")
            file_dist = h5.File(os.path.join(self.tmpdir.name, "metrics_dist.h5"), "r")

            for key in file_local.keys():
                data_local = file_local[key]["metric_data"][...]
                data_dist = file_dist[key]["metric_data"][...]
                with self.subTest(desc=f"file metric {key}"):
                    self.assertTrue(compare_arrays(f"file metric {key}", data_dist, data_local, verbose=verbose))

            # close files
            file_local.close()
            file_dist.close()

        # wait for everything to finish
        self.mpi_comm.Barrier()


if __name__ == "__main__":
    unittest.main()
