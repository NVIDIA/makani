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

import torch
import torch.nn.functional as F
import torch.distributed as dist

import torch_harmonics.distributed as thd

from makani.utils import comm
from makani.utils import functions as fn
from physicsnemo.distributed.utils import split_tensor_along_dim, compute_split_shapes
from physicsnemo.distributed.mappings import gather_from_parallel_region, scatter_to_parallel_region, reduce_from_parallel_region

from makani.utils.grids import GridQuadrature
from makani.utils.losses import EnsembleCRPSLoss, EnsembleNLLLoss

from distributed_helpers import split_helper, gather_helper

class TestDistributedLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # set up distributed
        cls.grid_size_h = int(os.getenv("GRID_H", 1))
        cls.grid_size_w = int(os.getenv("GRID_W", 1))
        cls.grid_size_e = int(os.getenv("GRID_E", 1))
        cls.world_size = cls.grid_size_h * cls.grid_size_w * cls.grid_size_e

        # init groups
        comm.init(
            model_parallel_sizes=[cls.grid_size_h, cls.grid_size_w, 1, 1],
            model_parallel_names=["h", "w", "fin", "fout"],
            data_parallel_sizes=[cls.grid_size_e, -1],
            data_parallel_names=["ensemble", "batch"],
        )
        cls.world_rank = comm.get_world_rank()

        if torch.cuda.is_available():
            if cls.world_rank == 0:
                print("Running test on GPU")
            local_rank = comm.get_local_rank()
            cls.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(cls.device)
            torch.cuda.manual_seed(333)
        else:
            if cls.world_rank == 0:
                print("Running test on CPU")
            cls.device = torch.device("cpu")
        torch.manual_seed(333)

        # store comm group parameters
        cls.wrank = comm.get_rank("w")
        cls.hrank = comm.get_rank("h")
        cls.erank = comm.get_rank("ensemble")
        cls.w_group = comm.get_group("w")
        cls.h_group = comm.get_group("h")
        cls.e_group = comm.get_group("ensemble")

        # initializing sht process groups just to be sure
        thd.init(cls.h_group, cls.w_group)

        if cls.world_rank == 0:
            print(f"Running distributed tests on grid H x W x E = {cls.grid_size_h} x {cls.grid_size_w} x {cls.grid_size_e}")

    def _split_helper(self, tensor):
        with torch.no_grad():
            # split in W
            tensor_local = split_helper(tensor, dim=-1, group=self.w_group)

            # split in H
            tensor_local = split_helper(tensor_local, dim=-2, group=self.h_group)

            # split in E
            if tensor.dim() == 5:
                tensor_local = split_helper(tensor_local, dim=1, group=self.e_group)

        return tensor_local

    def _gather_helper_fwd(self, tensor):
        # gather in world
        if self.world_size > 1:
            olist = [torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device) for _ in range(self.world_size)]
            olist[self.world_rank] = tensor
            dist.all_gather(olist, tensor)
            tensor_gather = torch.stack(olist, dim=-1)
        else:
            tensor_gather = tensor.unsqueeze(-1)

        return tensor_gather

    def _gather_helper_bwd(self, tensor, ensemble=False):
        tensor_gather = gather_helper(tensor, dim=-1, group=self.w_group)
        tensor_gather = gather_helper(tensor_gather, dim=-2, group=self.h_group)
        if ensemble:
            tensor_gather = gather_helper(tensor_gather, dim=1, group=self.e_group)

        return tensor_gather

    
    @parameterized.expand(
        [
            [128, 256, 32, 8, 1e-6, "naive", False],
            [128, 256, 32, 8, 1e-6, "naive", True],
            [129, 256, 32, 8, 1e-6, "naive", False],
            [129, 256, 32, 8, 1e-6, "naive", True],
            [129, 256, 32, 8, 1e-6, "clenshaw-curtiss", False],
            [129, 256, 32, 8, 1e-6, "clenshaw-curtiss", True],
            [129, 256, 32, 8, 1e-6, "legendre-gauss", False],
            [129, 256, 32, 8, 1e-6, "legendre-gauss", True],
            [129, 256, 32, 8, 1e-6, "weatherbench2", False],
            [129, 256, 32, 8, 1e-6, "weatherbench2", True],
        ], skip_on_empty=True
    )
    def test_distributed_quadrature(self, nlat, nlon, batch_size, num_chan, tol, quad_rule, normalize, verbose=False):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        quad_local = GridQuadrature(quadrature_rule=quad_rule, img_shape=(H, W), normalize=normalize, distributed=False).to(self.device)
        quad_dist = GridQuadrature(quadrature_rule=quad_rule, img_shape=(H, W), normalize=normalize, distributed=True).to(self.device)

        # input
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)
        inp_local = self._split_helper(inp_full)
        inp_full.requires_grad = True
        inp_local.requires_grad = True

        # local
        out_full = quad_local(inp_full)
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        
        # distributed
        out_local = quad_dist(inp_local)
        out_local.backward(ograd_full)
        igrad_local = inp_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            out_gather_full = self._gather_helper_fwd(out_local)

            # compute errors
            err = fn.relative_error(out_gather_full, out_full.unsqueeze(1))
            if verbose and (self.world_rank == 0):
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
	# evaluate BWD pass
        #############################################################
        with torch.no_grad():
            igrad_gather_full = self._gather_helper_bwd(igrad_local, False)
            
            # compute errors
            err = fn.relative_error(igrad_gather_full, igrad_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)


    @parameterized.expand(
        [
            [128, 256, 32, 8, 4, 1e-6, "ensemble_crps"],
            [129, 256, 1, 10, 4, 1e-6, "ensemble_crps"],
            [128, 256, 32, 8, 4, 1e-6, "ensemble_crps"],
            [129, 256, 1, 10, 4, 1e-6, "ensemble_crps"],
            [128, 256, 32, 8, 4, 1e-6, "skillspread_crps"],
            [129, 256, 1, 10, 4, 1e-6, "skillspread_crps"],
            [128, 256, 32, 8, 4, 1e-6, "gauss_crps"],
            [129, 256, 1, 10, 4, 1e-6, "gauss_crps"],
            [128, 256, 32, 8, 4, 1e-6, "ensemble_nll"],
            [129, 256, 1, 10, 4, 1e-6, "ensemble_nll"],
        ], skip_on_empty=True
    )
    def test_distributed_crps(self, nlat, nlon, batch_size, num_chan, ens_size, tol, loss_type, verbose=False):
        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        # generate gauss random distributed around 1, with sigma=2
        mean, sigma = (1.0, 2.0)
        inp_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.zeros((B, C, H, W), dtype=torch.float32, device=self.device) + mean

        if loss_type == "ensemble_crps":
            # local loss
            loss_fn_local = EnsembleCRPSLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                pole_mask=0,
                crps_type="cdf",
                spatial_distributed=False,
                ensemble_distributed=False,
                ensemble_weights=None,
            ).to(self.device)

            # distributed loss
            loss_fn_dist = EnsembleCRPSLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                pole_mask=0,
                crps_type="cdf",
                spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
                ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
                ensemble_weights=None,
            ).to(self.device)
        elif loss_type == "gauss_crps":
            # local loss
            loss_fn_local = EnsembleCRPSLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                pole_mask=0,
                crps_type="gauss",
                spatial_distributed=False,
                ensemble_distributed=False,
                eps=1.0e-5,
            ).to(self.device)

            # distributed loss
            loss_fn_dist = EnsembleCRPSLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                pole_mask=0,
                crps_type="gauss",
                spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
                ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
                eps=1.0e-5,
            ).to(self.device)
        elif loss_type == "skillspread_crps":
            # local loss
            loss_fn_local = EnsembleCRPSLoss(
                img_shape=(H, W),
		crop_shape=None,
		crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                pole_mask=0,
                crps_type="skillspread",
                spatial_distributed=False,
                ensemble_distributed=False,
                eps=1.0e-5,
            ).to(self.device)

            # distributed loss
            loss_fn_dist = EnsembleCRPSLoss(
                img_shape=(H, W),
		crop_shape=None,
                crop_offset=(0, 0),
	        channel_names=(),
                grid_type="equiangular",
	        pole_mask=0,
	        crps_type="skillspread",
		spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
                ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
                eps=1.0e-5,
            ).to(self.device)
        elif loss_type == "ensemble_nll":
            # local loss
            loss_fn_local = EnsembleNLLLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                pole_mask=0,
                spatial_distributed=False,
                ensemble_distributed=False,
                eps=1.0e-5,
            ).to(self.device)

            # distributed loss
            loss_fn_dist = EnsembleNLLLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                pole_mask=0,
                spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
                ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
                eps=1.0e-5,
            ).to(self.device)

        #############################################################
        # local loss
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(inp_full, obs_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = loss_full.clone() * 10000.0

        # BWD pass
        loss_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        obs_local = self._split_helper(obs_full)
        inp_local.requires_grad = True
        obs_local.requires_grad = True
        loss_local = loss_fn_dist(inp_local, obs_local)

        # BWD pass
        loss_local = loss_fn_dist(inp_local, obs_local)
        loss_local.backward(ograd_full)
        igrad_local = inp_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            loss_gather_full = self._gather_helper_fwd(loss_local)

            # compute errors
            err = fn.relative_error(loss_gather_full, loss_full.unsqueeze(1))
            if verbose and (self.world_rank == 0):
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        with torch.no_grad():
            # foreacst grads
            igrad_gather_full = self._gather_helper_bwd(igrad_local, True)

            # compute errors
            err = fn.relative_error(igrad_gather_full, igrad_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of forecast gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)

        with torch.no_grad():
            # observation grads
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)

            # compute errors
            err = fn.relative_error(obsgrad_gather_full, obsgrad_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of observation gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)


if __name__ == "__main__":
    unittest.main()
