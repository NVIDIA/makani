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
import unittest
from parameterized import parameterized

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import torch_harmonics as th
import torch_harmonics.distributed as thd

from makani.utils import comm
from makani.utils import functions as fn

from makani.mpu.mappings import init_gradient_reduction_hooks

# layer norm imports
from makani.models.common.layer_norm import GeometricInstanceNormS2
from makani.mpu.layer_norm import DistributedGeometricInstanceNormS2, DistributedInstanceNorm2d

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import split_helper, gather_helper
from ..testutils import disable_tf32, compare_tensors

class TestDistributedLayers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # set up distributed
        cls.grid_size_h = int(os.getenv('GRID_H', 1))
        cls.grid_size_w = int(os.getenv('GRID_W', 1))
        cls.world_size = cls.grid_size_h * cls.grid_size_w

        # init groups
        comm.init(model_parallel_sizes=[cls.grid_size_h, cls.grid_size_w, 1, 1, 1],
                  model_parallel_names=["h", "w", "fin", "fout", "batch"])
        cls.world_rank = comm.get_world_rank()

        torch.manual_seed(333)
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
            cls.device = torch.device('cpu')

        # store comm group parameters
        cls.wrank = comm.get_rank("w")
        cls.hrank = comm.get_rank("h")
        cls.w_group = comm.get_group("w")
        cls.h_group = comm.get_group("h")

        # initializing sht process groups
        thd.init(cls.h_group, cls.w_group)

        if cls.world_rank == 0:
            print(f"Running distributed tests on grid H x W = {cls.grid_size_h} x {cls.grid_size_w}")

    def setUp(self):
        disable_tf32()


    def _init_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        return

        
    def _split_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_local = split_helper(tensor, dim=hdim, group=self.h_group)
        tensor_local = split_helper(tensor_local, dim=wdim, group=self.w_group)
        return tensor_local
        
        
    def _gather_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_gather = gather_helper(tensor, dim=hdim, group=self.h_group)
        tensor_gather = gather_helper(tensor_gather, dim=wdim, group=self.w_group)

        return tensor_gather


    @parameterized.expand(
        [
            [180, 360, 256, 512, 32,  8, 5e-4],
            [181, 360, 181, 360, 1, 10, 5e-5],
            [180, 360, 128, 256, 32,  8, 5e-4],
            [181, 360,  91, 180, 1, 10, 5e-5],
            [128, 256, 256, 512, 32,  8, 5e-5],
            [ 91, 180, 181, 360, 1, 10, 5e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectral_conv(self, nlat_in, nlon_in, nlat_out, nlon_out, batch_size, num_chan, tol, verbose=True):
        B, C, Hi, Wi, Ho, Wo = batch_size, num_chan, nlat_in, nlon_in, nlat_out, nlon_out

        from makani.models.common import SpectralConv
        
        # set up handles
        forward_transform_local = th.RealSHT(nlat=Hi, nlon=Wi).to(self.device)
        inverse_transform_local = th.InverseRealSHT(nlat=Ho, nlon=Wo, lmax=forward_transform_local.lmax, mmax=forward_transform_local.mmax).to(self.device)
        forward_transform_dist = thd.DistributedRealSHT(nlat=Hi, nlon=Wi).to(self.device)
        inverse_transform_dist = thd.DistributedInverseRealSHT(nlat=Ho, nlon=Wo, lmax=forward_transform_dist.lmax, mmax=forward_transform_dist.mmax).to(self.device)

        self._init_seed(333)

        spect_conv_local = SpectralConv(
            forward_transform_local,
            inverse_transform_local,
            C,
            C,
            operator_type="dhconv",
            num_groups=1,
            bias=True,
            gain=1.0,
        ).to(self.device)

        spect_conv_dist = SpectralConv(
	    forward_transform_dist,
            inverse_transform_dist,
            C,
            C,
            operator_type="dhconv",
            num_groups=1,
            bias=True,
            gain=1.0,
        ).to(self.device)

        # set up wgrad reductions
        spect_conv_dist = init_gradient_reduction_hooks(
            spect_conv_dist,
            device=self.device,
            reduction_buffer_count=1,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            verbose=False,
        )

        # make sure weights are the same:
        with torch.no_grad():
            weight = self._split_helper(spect_conv_local.weight, hdim=-1, wdim=None)
            spect_conv_dist.module.weight.copy_(weight)
            spect_conv_dist.module.bias.copy_(spect_conv_local.bias)
        
        # input
        inp_full = torch.randn((B, C, Hi, Wi), dtype=torch.float32, device=self.device)
        
        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full, _ = spect_conv_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)
            
        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        wgrad_full = spect_conv_local.weight.grad.clone()
        bgrad_full = spect_conv_local.bias.grad.clone()
        
        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local, _ = spect_conv_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local, _ = spect_conv_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        wgrad_local = spect_conv_dist.module.weight.grad.clone()
        bgrad_local = spect_conv_dist.module.bias.grad.clone()
        
        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate input grads
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate Weight grads
        #############################################################
        with self.subTest(desc="weight gradients"):
            wgrad_gather_full = self._gather_helper(wgrad_local, hdim=-1, wdim=None)
            self.assertTrue(compare_tensors("weight gradients", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose))

        with self.subTest(desc="bias gradients"):
            bgrad_gather_list = [torch.empty_like(bgrad_local) for _ in range(self.world_size)]
            bgrad_gather_list[self.world_rank] = bgrad_local
            dist.all_gather(bgrad_gather_list, bgrad_local, group=None)
            for idb, bgrad_gather_full in enumerate(bgrad_gather_list):
                self.assertTrue(compare_tensors(f"bias gradient {idb}", bgrad_gather_full, bgrad_full, tol, tol, verbose=verbose))

    @parameterized.expand(
        [
            [256, 512, 32, 8, True, 1e-5],
            [181, 360, 1, 10, True, 1e-5],
            [256, 512, 32, 8, False, 1e-5],
            [181, 360, 1, 10, False, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_instance_norm_2d(self, nlat, nlon, batch_size, num_chan, affine, tol, verbose=True):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        self._init_seed(333)

        # create local (serial) instance norm - using PyTorch's standard InstanceNorm2d
        norm_local = nn.InstanceNorm2d(
            num_features=C,
            eps=1e-5,
            affine=affine,
            track_running_stats=False,
        ).to(self.device)

        # create distributed instance norm
        norm_dist = DistributedInstanceNorm2d(
            num_features=C,
            eps=1e-5,
            affine=affine,
        ).to(self.device)

        # set up gradient reduction hooks for distributed version if affine=True
        if affine:
            norm_dist = init_gradient_reduction_hooks(
                norm_dist,
                device=self.device,
                reduction_buffer_count=1,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
                verbose=False,
            )
            norm_dist_handle = norm_dist.module
        else:
            norm_dist_handle = norm_dist

        # make sure weights are the same if affine=True
        if affine:
            with torch.no_grad():
                norm_dist_handle.weight.copy_(norm_local.weight)
                norm_dist_handle.bias.copy_(norm_local.bias)
        
        # input
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)
        
        #############################################################
        # local (serial) transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = norm_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        if affine:
            wgrad_full = norm_local.weight.grad.clone()
            bgrad_full = norm_local.bias.grad.clone()
        
        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = norm_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        if affine:
            wgrad_local = norm_dist_handle.weight.grad.clone()
            bgrad_local = norm_dist_handle.bias.grad.clone()
        
        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate input grads
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate weight and bias grads
        #############################################################
        # weight gradients should be the same across all processes
        if affine:
            with self.subTest(desc="weight gradients"):
                wgrad_gather_list = [torch.empty_like(wgrad_local) for _ in range(self.world_size)]
                wgrad_gather_list[self.world_rank] = wgrad_local
                dist.all_gather(wgrad_gather_list, wgrad_local, group=None)
                for idw, wgrad_gather_full in enumerate(wgrad_gather_list):
                    self.assertTrue(compare_tensors(f"weight gradient {idw}", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose))


        # bias gradients should be the same across all processes
        if affine:
            with self.subTest(desc="bias gradients"):
                bgrad_gather_list = [torch.empty_like(bgrad_local) for _ in range(self.world_size)]
                bgrad_gather_list[self.world_rank] = bgrad_local
                dist.all_gather(bgrad_gather_list, bgrad_local, group=None)
                for idb, bgrad_gather_full in enumerate(bgrad_gather_list):
                    self.assertTrue(compare_tensors(f"bias gradient {idb}", bgrad_gather_full, bgrad_full, tol, tol, verbose=verbose))


    @parameterized.expand(
        [
            [181, 360, 1, 4, "equiangular", True, 1e-5],
            [181, 360, 1, 4, "equiangular", False, 1e-5],
            [180, 360, 1, 10, "legendre-gauss", True, 1e-5],
            [180, 360, 1, 10, "legendre-gauss", False, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_geometric_instance_norm_s2(self, nlat, nlon, batch_size, num_chan, grid_type, affine, tol, verbose=True):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        # set up layer norm parameters
        img_shape = (H, W)
        crop_shape = (H, W)
        crop_offset = (0, 0)
        pole_mask = 0
        eps = 1e-5

        self._init_seed(333)

        # create local (serial) layer norm
        norm_local = GeometricInstanceNormS2(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            grid_type=grid_type,
            pole_mask=pole_mask,
            num_features=C,
            eps=eps,
            affine=affine,
        ).to(self.device)

        # create distributed layer norm
        norm_dist = DistributedGeometricInstanceNormS2(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            grid_type=grid_type,
            pole_mask=pole_mask,
            num_features=C,
            eps=eps,
            affine=affine,
        ).to(self.device)

        # set up gradient reduction hooks for distributed version
        if affine:
            norm_dist = init_gradient_reduction_hooks(
                norm_dist,
                device=self.device,
                reduction_buffer_count=1,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
                verbose=False,
            )

        #make sure weights are the same if affine=True
        if affine:
            with torch.no_grad():
                norm_dist.module.weight.copy_(norm_local.weight)
                norm_dist.module.bias.copy_(norm_local.bias)
        
        # input
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)
        
        #############################################################
        # local (serial) transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = norm_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        if affine:
            wgrad_full = norm_local.weight.grad.clone()
            bgrad_full = norm_local.bias.grad.clone()
        
        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = norm_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        if affine:
            wgrad_local = norm_dist.module.weight.grad.clone()
            bgrad_local = norm_dist.module.bias.grad.clone()
        
        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate input grads
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate weight and bias grads
        #############################################################
        # weight gradients should be the same across all processes
        if affine:
            with self.subTest(desc="weight gradients"):
                wgrad_gather_list = [torch.empty_like(wgrad_local) for _ in range(self.world_size)]
                wgrad_gather_list[self.world_rank] = wgrad_local
                dist.all_gather(wgrad_gather_list, wgrad_local, group=None)
                for idw, wgrad_gather_full in enumerate(wgrad_gather_list):
                    self.assertTrue(compare_tensors(f"weight gradient {idw}", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose))

        # bias gradients should be the same across all processes
        if affine:
            with self.subTest(desc="bias gradients"):
                bgrad_gather_list = [torch.empty_like(bgrad_local) for _ in range(self.world_size)]
                bgrad_gather_list[self.world_rank] = bgrad_local
                dist.all_gather(bgrad_gather_list, bgrad_local, group=None)
                for idb, bgrad_gather_full in enumerate(bgrad_gather_list):
                    self.assertTrue(compare_tensors(f"bias gradient {idb}", bgrad_gather_full, bgrad_full, tol, tol, verbose=verbose))


if __name__ == '__main__':    
    unittest.main()
