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


import os
import tempfile
import unittest
from parameterized import parameterized

import h5py as h5
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist

import torch_harmonics.distributed as thd

from makani.utils import comm
from makani.utils import functions as fn
from physicsnemo.distributed.utils import split_tensor_along_dim, compute_split_shapes
from physicsnemo.distributed.mappings import gather_from_parallel_region, scatter_to_parallel_region, reduce_from_parallel_region

from makani.utils import MetricsHandler

from distributed_helpers import split_helper, gather_helper, get_default_parameters

# because of physicsnemo/NCCL tear down issues, we can only run one test at a time
_metric_handler_params = [
    #("equiangular", 4, 16, 3, "mean"),
    ("equiangular", 4, 16, 3, "sum"),
]

class TestDistributedMetricHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls, path="/tmp"):
        from mpi4py import MPI
        cls.mpi_comm = MPI.COMM_WORLD.Dup()
        cls.mpi_comm_rank = cls.mpi_comm.Get_rank()
        cls.mpi_comm_size = cls.mpi_comm.Get_size()

        if torch.cuda.is_available():
            if cls.mpi_comm_rank == 0:
                print("Running test on GPU")
            local_rank = cls.mpi_comm_rank % torch.cuda.device_count()
            cls.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(cls.device)
            torch.cuda.manual_seed(333)
        else:
            if self.mpi_comm_rank == 0:
                print("Running test on CPU")
            cls.device = torch.device("cpu")
        torch.manual_seed(333)

        # create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        
        return

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _init_comms(self):
        
        # set up distributed
        self.grid_size_h = int(os.getenv("GRID_H", 1))
        self.grid_size_w = int(os.getenv("GRID_W", 1))
        self.grid_size_e = int(os.getenv("GRID_E", 1))
        self.grid_size_b = int(os.getenv("GRID_B", 1))
        self.world_size = self.grid_size_h * self.grid_size_w * self.grid_size_e * self.grid_size_b

        # init groups
        comm.init(
            model_parallel_sizes=[self.grid_size_h, self.grid_size_w, 1, 1],
            model_parallel_names=["h", "w", "fin", "fout"],
            data_parallel_sizes=[self.grid_size_e, self.grid_size_b],
            data_parallel_names=["ensemble", "batch"],
        )
        self.world_rank = comm.get_world_rank()

        # store comm group parameters
        self.wrank = comm.get_rank("w")
        self.hrank = comm.get_rank("h")
        self.erank = comm.get_rank("ensemble")
        self.w_group = comm.get_group("w")
        self.h_group = comm.get_group("h")
        self.e_group = comm.get_group("ensemble")
        self.b_group = comm.get_group("batch")

        # initializing sht process groups just to be sure
        thd.init(self.h_group, self.w_group)

        if self.world_rank == 0:
            print(f"Running distributed tests on grid H x W x E x B = {self.grid_size_h} x {self.grid_size_w} x {self.grid_size_e} x {self.grid_size_b}")

        return

    def _destroy_comms(self):
        comm.cleanup()
        return

    def _split_helper(self, tensor):
        with torch.no_grad():
            # split in W
            tensor_local = split_helper(tensor, dim=-1, group=self.w_group)

            # split in H
            tensor_local = split_helper(tensor_local, dim=-2, group=self.h_group)

            # split in E
            if tensor.dim() == 6:
                tensor_local = split_helper(tensor_local, dim=2, group=self.e_group)

            # split in B
            tensor_local = split_helper(tensor_local, dim=1, group=self.b_group)

        return tensor_local
        
    def setUp(self):
        self.params = get_default_parameters()
        self.params["dhours"] = 1

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = 36
        self.params.img_shape_y = 72
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_local_offset_x = 0
        self.params.img_local_offset_y = 0
        self.params.img_crop_offset_x = 0
        self.params.img_crop_offset_y = 0

        return
    
    @parameterized.expand(_metric_handler_params, skip_on_empty=True)
    def test_metric_handler_aggregation(self, grid_type, batch_size, ensemble_size, num_rollout_steps, bred, verbose=False):
        # create dummy climatology
        num_steps = 4
        num_channels = len(self.params.channel_names)
        clim = torch.zeros(1, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)

        # local
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

        loss_acc = 0.
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

        # init comms
        self._init_comms()

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
            self.assertTrue(np.allclose(val_local, val_dist))

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

        self.assertTrue(np.allclose(data_local, data_dist))

        # save output files and compare
        if comm.get_world_rank() == 0:
            metric_handler_local.save(os.path.join(self.tmpdir.name, "metrics_local.h5"))
            metric_handler_dist.save(os.path.join(self.tmpdir.name, "metrics_dist.h5"))

            file_local = h5.File(os.path.join(self.tmpdir.name, "metrics_local.h5"), "r")
            file_dist = h5.File(os.path.join(self.tmpdir.name, "metrics_dist.h5"), "r")

            for key in file_local.keys():
                data_local = file_local[key]["metric_data"][...]
                data_dist = file_dist[key]["metric_data"][...]
                if verbose:
                    print(f"file metric {key}: local={data_local}, dist={data_dist}")
                self.assertTrue(np.allclose(data_local, data_dist))

            # close files
            file_local.close()
            file_dist.close()

        # wait for everything to finish
        self.mpi_comm.Barrier()

        self._destroy_comms()
        
        

if __name__ == "__main__":
    unittest.main()
