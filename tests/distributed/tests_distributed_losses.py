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
import torch.distributed as dist

from makani.utils import comm

from makani.utils.grids import GridQuadrature
from makani.utils.losses import (
    CRPSLoss,
    EnsembleNLLLoss,
    SpectralCRPSLoss,
    L2EnergyScoreLoss,
    SpectralL2EnergyScoreLoss,
    SobolevEnergyScoreLoss,
)

# Add parent directory to path for testutils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tests.distributed.distributed_helpers import init_grid, split_helper, gather_helper
from tests.testutils import compare_tensors, disable_tf32

class TestDistributedLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_grid(cls)

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

        # disable tf32 for deterministic comparison# disable tf32# disable tf32 for deterministic comparison# disable tf32
        disable_tf32()

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

        # disable tf32 for deterministic comparison# disable tf32
        disable_tf32()

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


    @parameterized.expand(
        [
            [128, 256, 8, 3, 4, 1e-4],
            [129, 256, 2, 5, 4, 1e-4],
        ], skip_on_empty=True
    )
    def test_distributed_l2_energy_score(self, nlat, nlon, batch_size, num_chan, ens_size, tol, verbose=False):

        # disable tf32 for deterministic comparison# disable tf32
        disable_tf32()

        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        # inputs
        mean, sigma = (1.0, 2.0)
        forecasts_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.01 + mean

        # local loss
        loss_fn_local = L2EnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            pole_mask=0,
            alpha=1.0,
            beta=1.0,
            eps=1.0e-5,
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
        ).to(self.device)

        # distributed loss
        loss_fn_dist = L2EnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            pole_mask=0,
            alpha=1.0,
            beta=1.0,
            eps=1.0e-5,
            spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
            ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
            ensemble_weights=None,
        ).to(self.device)

        #############################################################
        # local loss
        #############################################################
        forecasts_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(forecasts_full, obs_full)

        with torch.no_grad():
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        loss_full.backward(ograd_full)
        fgrad_full = forecasts_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        forecasts_local = self._split_helper(forecasts_full.clone())
        obs_local = self._split_helper(obs_full.clone())
        forecasts_local.requires_grad = True
        obs_local.requires_grad = True

        loss_local = loss_fn_dist(forecasts_local, obs_local)
        loss_local.backward(ograd_local)
        fgrad_local = forecasts_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate BWD pass
        #############################################################
        with self.subTest(desc="forecast gradients"):
            fgrad_gather_full = self._gather_helper_bwd(fgrad_local, True)
            self.assertTrue(compare_tensors("forecast gradients", fgrad_gather_full, fgrad_full, tol, tol, verbose=verbose))

        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(compare_tensors("observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose))


    @parameterized.expand(
        [
            [128, 256, 8, 3, 4, 1e-4],
            [129, 256, 2, 5, 4, 1e-4],
        ], skip_on_empty=True
    )
    def test_distributed_spectral_l2_energy_score(self, nlat, nlon, batch_size, num_chan, ens_size, tol, verbose=True):

        # disable tf32 for deterministic comparison
        disable_tf32()

        # shapes
        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        mean, sigma = (1.0, 2.0)
        forecasts_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.01 + mean

        # local loss
        loss_fn_local = SpectralL2EnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            alpha=1.0,
            eps=1.0e-3,
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
        ).to(self.device)

        # distributed loss
        loss_fn_dist = SpectralL2EnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            alpha=1.0,
            eps=1.0e-3,
            spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
            ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
            ensemble_weights=None,
        ).to(self.device)

        #############################################################
        # local loss
        #############################################################
        forecasts_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(forecasts_full, obs_full)

        with torch.no_grad():
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        loss_full.backward(ograd_full)
        fgrad_full = forecasts_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        forecasts_local = self._split_helper(forecasts_full.clone())
        obs_local = self._split_helper(obs_full.clone())
        forecasts_local.requires_grad = True
        obs_local.requires_grad = True

        loss_local = loss_fn_dist(forecasts_local, obs_local)
        loss_local.backward(ograd_local)
        fgrad_local = forecasts_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate BWD pass
        #############################################################
        with self.subTest(desc="forecast gradients"):
            fgrad_gather_full = self._gather_helper_bwd(fgrad_local, True)
            self.assertTrue(compare_tensors("forecast gradients", fgrad_gather_full, fgrad_full, tol, tol, verbose=verbose))

        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(compare_tensors("observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose))


    @parameterized.expand(
        [
            [128, 256, 8, 3, 4, 1.0, 1.0, 1.0, 1.0, 1e-5],
            [129, 256, 2, 2, 4, 0.8, 1.2, 0.5, 0.7, 1e-5],
        ], skip_on_empty=True
    )
    def test_distributed_sobolev_energy_score(self, nlat, nlon, batch_size, num_chan, ens_size, alpha, beta, offset, fraction, tol, verbose=True):

        disable_tf32()

        B, E, C, H, W = batch_size, ens_size, num_chan, nlat, nlon

        mean, sigma = (0.3, 1.1)
        forecasts_full = torch.randn((B, E, C, H, W), dtype=torch.float32, device=self.device) * sigma + mean
        obs_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device) * sigma * 0.05 + mean

        # local loss
        loss_fn_local = SobolevEnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
            alpha=alpha,
            beta=beta,
            offset=offset,
            fraction=fraction,
            eps=1.0e-6,
        ).to(self.device)

        # distributed loss
        loss_fn_dist = SobolevEnergyScoreLoss(
            img_shape=(H, W),
            crop_shape=None,
            crop_offset=(0, 0),
            channel_names=(),
            grid_type="equiangular",
            spatial_distributed=(comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)),
            ensemble_distributed=(comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)),
            ensemble_weights=None,
            alpha=alpha,
            beta=beta,
            offset=offset,
            fraction=fraction,
            eps=1.0e-6,
        ).to(self.device)

        #############################################################
        # local loss
        #############################################################
        forecasts_full.requires_grad = True
        obs_full.requires_grad = True
        loss_full = loss_fn_local(forecasts_full, obs_full)

        with torch.no_grad():
            ograd_full = torch.randn_like(loss_full)
            ograd_local = ograd_full.clone()

        loss_full.backward(ograd_full)
        fgrad_full = forecasts_full.grad.clone()
        obsgrad_full = obs_full.grad.clone()

        #############################################################
        # distributed loss
        #############################################################
        forecasts_local = self._split_helper(forecasts_full.clone())
        obs_local = self._split_helper(obs_full.clone())
        forecasts_local.requires_grad = True
        obs_local.requires_grad = True

        loss_local = loss_fn_dist(forecasts_local, obs_local)
        loss_local.backward(ograd_local)
        fgrad_local = forecasts_local.grad.clone()
        obsgrad_local = obs_local.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="outputs"):
            self.assertTrue(compare_tensors("outputs", loss_local, loss_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate BWD pass
        #############################################################
        with self.subTest(desc="forecast gradients"):
            fgrad_gather_full = self._gather_helper_bwd(fgrad_local, True)
            self.assertTrue(compare_tensors("forecast gradients", fgrad_gather_full, fgrad_full, tol, tol, verbose=verbose))

        with self.subTest(desc="observation gradients"):
            obsgrad_gather_full = self._gather_helper_bwd(obsgrad_local, False)
            self.assertTrue(compare_tensors("observation gradients", obsgrad_gather_full, obsgrad_full, tol, tol, verbose=verbose))


if __name__ == "__main__":
    disable_tf32()
    unittest.main()
