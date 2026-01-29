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
import sys
import unittest
from parameterized import parameterized

import torch
import torch.nn.functional as F
import torch.distributed as dist

import torch_harmonics.distributed as thd

from makani.utils import comm
from makani.utils import functions as fn

from makani.utils.grids import GridQuadrature
from makani.utils.losses import CRPSLoss, EnsembleNLLLoss, SpectralCRPSLoss

# Add parent directory to path for testutils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import split_helper, gather_helper
from ..testutils import compare_tensors, disable_tf32

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
            [128, 256, 32, 8, "naive", False, 1e-6],
            [128, 256, 32, 8, "naive", True, 1e-6],
            [129, 256, 32, 8, "naive", True, 1e-6],
            [129, 256, 32, 8, "clenshaw-curtiss", False, 1e-6],
            [129, 256, 32, 8, "clenshaw-curtiss", True, 1e-6],
            [129, 256, 32, 8, "legendre-gauss", False, 1e-6],
            [129, 256, 32, 8, "legendre-gauss", True, 1e-6],
            [129, 256, 32, 8, "weatherbench2", False, 1e-6],
            [129, 256, 32, 8, "weatherbench2", True, 1e-6],
        ], skip_on_empty=True
    )
    def test_distributed_quadrature(self, nlat, nlon, batch_size, num_chan, quad_rule, normalize, tol, verbose=False):
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
        with self.subTest(desc="outputs"):
            self.assertTrue(compare_tensors("outputs", out_local, out_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate BWD pass
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper_bwd(igrad_local, False)
            self.assertTrue(compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose))


    @parameterized.expand(
        [
            [128, 256, 32, 8, 4, "cdf", 1e-5],
            [129, 256, 1, 10, 4, "cdf", 1e-5],
            [128, 256, 32, 8, 4, "cdf", 1e-5],
            [129, 256, 1, 10, 4, "cdf", 1e-5],
            [128, 256, 32, 8, 4, "skillspread", 1e-5],
            [129, 256, 1, 10, 4, "skillspread", 1e-5],
            [128, 256, 32, 8, 4, "gauss", 1e-5],
            [129, 256, 1, 10, 4, "gauss", 1e-5],
            [128, 256, 32, 8, 4, "nll", 1e-5],
            [129, 256, 1, 10, 4, "nll", 1e-5],
        ], skip_on_empty=True
    )
    def test_distributed_crps(self, nlat, nlon, batch_size, num_chan, ens_size, loss_type, tol, verbose=False):
        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        # generate gauss random distributed around 1, with sigma=2
        mean, sigma = (1.0, 2.0)
        inp_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.full((B, C, H, W), fill_value=mean, dtype=torch.float32, device=self.device)

        if loss_type != "nll":
            # local loss
            loss_fn_local = CRPSLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                pole_mask=0,
                crps_type=loss_type,
                eps=1.0e-5,
                spatial_distributed=False,
                ensemble_distributed=False,
                ensemble_weights=None,
            ).to(self.device)

            # distributed loss
            loss_fn_dist = CRPSLoss(
                img_shape=(H, W),
                crop_shape=None,
                crop_offset=(0, 0),
                channel_names=(),
                grid_type="equiangular",
                pole_mask=0,
                crps_type=loss_type,
                eps=1.0e-5,
                spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
                ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
                ensemble_weights=None,
            ).to(self.device)
        else:
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
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

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

        # BWD pass
        loss_local = loss_fn_dist(inp_local, obs_local)
        loss_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate BWD pass
        #############################################################
        # foreacst grads
        with self.subTest(desc="forecast gradients"):
            igrad_gather_full = self._gather_helper_bwd(igrad_local, True)
            self.assertTrue(compare_tensors("forecast gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose))

        # observation grads
        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(compare_tensors("observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose))


    @parameterized.expand(
        [
            [128, 256, 32, 8, 4, "cdf", True, 1e-4],
            [129, 256, 1, 10, 4, "cdf", True, 1e-4],
            [128, 256, 32, 8, 4, "skillspread", False, 1e-4],
            [129, 256, 1, 10, 4, "skillspread", False, 1e-4],
            [128, 256, 32, 8, 4, "skillspread", True, 1e-4],
            [129, 256, 1, 10, 4, "skillspread", True, 1e-4],
        ], skip_on_empty=True
    )
    def test_distributed_spectral_crps(self, nlat, nlon, batch_size, num_chan, ens_size, loss_type, absolute, tol, verbose=True):

        # disable tf32
        disable_tf32()

        # extract shapes
        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        # generate gauss random distributed around 1, with sigma=2
        mean, sigma = (1.0, 2.0)
        inp_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.01 + mean

        # local loss
        loss_fn_local = SpectralCRPSLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            crps_type=loss_type,
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
            eps=1.0e-5,
            absolute=absolute,
        ).to(self.device)

        # distributed loss
        loss_fn_dist = SpectralCRPSLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            crps_type=loss_type,
            spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
            ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
            ensemble_weights=None,
            eps=1.0e-5,
            absolute=absolute,
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
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        # BWD pass
        loss_full = loss_fn_local(inp_full, obs_full)
        loss_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full.clone())
        obs_local = self._split_helper(obs_full.clone())
        inp_local.requires_grad = True
        obs_local.requires_grad = True

        # BWD pass
        loss_local = loss_fn_dist(inp_local, obs_local)
        loss_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate BWD pass
        #############################################################
        # foreacst grads
        with self.subTest(desc="forecast gradients"):
            igrad_gather_full = self._gather_helper_bwd(igrad_local, True)
            self.assertTrue(compare_tensors("forecast gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose))

        # observation grads
        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            if self.world_rank == 0:
                print("obsgrad_gather_full", obsgrad_gather_full[0, 0, ...], "obsgrad_full", obsgrad_full[0, 0, ...])
            self.assertTrue(compare_tensors("observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose))


if __name__ == "__main__":
    unittest.main()
