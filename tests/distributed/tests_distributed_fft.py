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
import unittest
from parameterized import parameterized

import torch
import torch.nn.functional as F
import torch.distributed as dist

import torch_harmonics.distributed as thd

from makani.models.common import RealFFT1, InverseRealFFT1, RealFFT2, InverseRealFFT2, RealFFT3, InverseRealFFT3

from makani.utils import comm
from makani.utils import functions as fn
from physicsnemo.distributed.utils import split_tensor_along_dim
from physicsnemo.distributed.mappings import gather_from_parallel_region, scatter_to_parallel_region, \
                                         reduce_from_parallel_region
from makani.mpu.fft import DistributedRealFFT1, DistributedInverseRealFFT1, DistributedRealFFT2, DistributedInverseRealFFT2, DistributedRealFFT3, DistributedInverseRealFFT3

from distributed_helpers import split_helper, gather_helper

class TestDistributedRealFFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # set up distributed
        cls.grid_size_h = int(os.getenv('GRID_H', 1))
        cls.grid_size_w = int(os.getenv('GRID_W', 1))
        cls.world_size = cls.grid_size_h * cls.grid_size_w

        # init groups
        comm.init(model_parallel_sizes=[cls.grid_size_h, cls.grid_size_w, 1, 1],
                  model_parallel_names=["h", "w", "fin", "fout"])
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
            cls.device = torch.device('cpu')
        torch.manual_seed(333)

        # store comm group parameters
        cls.wrank = comm.get_rank("w")
        cls.hrank = comm.get_rank("h")
        cls.w_group = comm.get_group("w")
        cls.h_group = comm.get_group("h")

        # initializing sht process groups
        thd.init(cls.h_group, cls.w_group)

        if cls.world_rank == 0:
            print(f"Running distributed tests on grid H x W = {cls.grid_size_h} x {cls.grid_size_w}")

        
    def _split_helper(self, tensor):
        tensor_local = split_helper(tensor, dim=-1, group=self.w_group)
        tensor_local = split_helper(tensor_local, dim=-2, group=self.h_group)

        return tensor_local
        
        
    def _gather_helper(self, tensor):
        tensor_gather = gather_helper(tensor, dim=-2, group=self.h_group)
        tensor_gather =	gather_helper(tensor_gather, dim=-1, group=self.w_group)
        
        return tensor_gather

    @parameterized.expand([
        [256, 512, 32,  8, 1e-6],
        [361, 720,  1, 10, 1e-6],
    ])
    def test_distributed_fft1(self, nlat, nlon, batch_size, num_chan, tol, verbose=False):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        # set up handles
        forward_transform_local = RealFFT1(nlon=W).to(self.device)
        forward_transform_dist = DistributedRealFFT1(nlon=W).to(self.device)

        # create tensors
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = forward_transform_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)
            
        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        
        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        inp_local.requires_grad = True
        out_local = forward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = forward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        
        # set eval dims
        dims = (-1,-2)
        
        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            out_gather_full = self._gather_helper(out_local)
            err = fn.relative_error(out_gather_full, out_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        with torch.no_grad():
            igrad_gather_full = self._gather_helper(igrad_local)
            err = fn.relative_error(igrad_gather_full, igrad_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)

    @parameterized.expand([
        [256, 512, 32,  8, 1e-6],
        [361, 720,  1, 10, 1e-6],
    ])
    def test_distributed_ifft1(self, nlat, nlon, batch_size, num_chan, tol, verbose=False):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        forward_transform_local = RealFFT1(nlon=W).to(self.device)
        backward_transform_local = InverseRealFFT1(nlon=W).to(self.device)
        backward_transform_dist = DistributedInverseRealFFT1(nlon=W).to(self.device)

        # create tensors
        dummy_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)
        inp_full = forward_transform_local(dummy_full)

        #############################################################
        # local transform
	    #############################################################
	    # FWD pass
        inp_full.requires_grad = True
        out_full = backward_transform_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)

        # repeat once due to known irfft bug
        inp_full.grad = None
        out_full = backward_transform_local(inp_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        inp_local.requires_grad = True
        out_local = backward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = backward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # set eval dims
        dims = (-1,-2)
        
        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            out_gather_full = self._gather_helper(out_local)
            err = fn.relative_error(out_gather_full, out_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        with torch.no_grad():
            igrad_gather_full = self._gather_helper(igrad_local)
            err = fn.relative_error(igrad_gather_full, igrad_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)


    @parameterized.expand([
        [256, 512, 0, 32,  8, 1e-6],
        [361, 720, 0,  1, 10, 1e-6],
        [256, 512, 4, 32,  8, 1e-6],
        [361, 720, 4,  1, 10, 1e-6],
    ])
    def test_distributed_fft2_3(self, nlat, nlon, nalt, batch_size, num_chan, tol, verbose=False):
        B, C, D, H, W = batch_size, num_chan, nalt, nlat, nlon

        # set up handles
        if D > 0:
            forward_transform_local = RealFFT3(nd=D, nh=H, nw=W).to(self.device)
            forward_transform_dist = DistributedRealFFT3(nd=D, nh=H, nw=W).to(self.device)
        else:
            forward_transform_local = RealFFT2(nlat=H, nlon=W).to(self.device)
            forward_transform_dist = DistributedRealFFT2(nlat=H, nlon=W).to(self.device)

        # create tensors
        if D > 0:
            inp_full = torch.randn((B, C, D, H, W), dtype=torch.float32, device=self.device)
        else:
            inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = forward_transform_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)
            
        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        
        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        inp_local.requires_grad = True
        out_local = forward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = forward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        
        # set eval dims
        dims = (-1,-2,-3) if D > 0 else (-1,-2)
        
        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            out_gather_full = self._gather_helper(out_local)
            err = fn.relative_error(out_gather_full, out_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        with torch.no_grad():
            igrad_gather_full = self._gather_helper(igrad_local)
            err = fn.relative_error(igrad_gather_full, igrad_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)


    @parameterized.expand([
        [256, 512, 0, 32,  8, 1e-6],
        [361, 720, 0,  1, 10, 1e-6],
        [256, 512, 4, 32,  8, 1e-6],
        [361, 720, 4,  1, 10, 1e-6],
    ])
    def test_distributed_ifft2_3(self, nlat, nlon, nalt, batch_size, num_chan, tol, verbose=False):
        B, C, D, H, W = batch_size, num_chan, nalt, nlat, nlon

        if D > 0:
            forward_transform_local = RealFFT3(nd=D, nh=H, nw=W).to(self.device)
            backward_transform_local = InverseRealFFT3(nd=D, nh=H, nw=W).to(self.device)
            backward_transform_dist = DistributedInverseRealFFT3(nd=D, nh=H, nw=W).to(self.device)
        else:
            forward_transform_local = RealFFT2(nlat=H, nlon=W).to(self.device)
            backward_transform_local = InverseRealFFT2(nlat=H, nlon=W).to(self.device)
            backward_transform_dist = DistributedInverseRealFFT2(nlat=H, nlon=W).to(self.device)

        # create tensors
        if D > 0:
            dummy_full = torch.randn((B, C, D, H, W), dtype=torch.float32, device=self.device)
        else:
            dummy_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)
        inp_full = forward_transform_local(dummy_full)

        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = backward_transform_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)

        # repeat once due to known irfft bug
        inp_full.grad = None
        out_full = backward_transform_local(inp_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full)
        inp_local.requires_grad = True
        out_local = backward_transform_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full)
        out_local = backward_transform_dist(inp_local)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        # set eval dims
        dims = (-1,-2,-3) if D > 0 else	(-1,-2)
        
        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            out_gather_full = self._gather_helper(out_local)
            err = fn.relative_error(out_gather_full, out_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= tol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        with torch.no_grad():
            igrad_gather_full = self._gather_helper(igrad_local)
            err = fn.relative_error(igrad_gather_full, igrad_full)
            if verbose and (self.world_rank == 0):
                print(f"final relative error of gradients: {err.item()}")
        self.assertTrue(err.item() <= tol)

if __name__ == '__main__':    
    unittest.main()
