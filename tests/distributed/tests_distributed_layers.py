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
import torch.distributed as dist

import torch_harmonics as th
import torch_harmonics.distributed as thd

from makani.utils import comm

from makani.mpu.mappings import init_gradient_reduction_hooks

# layer norm imports
from makani.models.common.layer_norm import GeometricInstanceNormS2
from makani.mpu.layer_norm import DistributedGeometricInstanceNormS2, DistributedInstanceNorm2d

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import _init_grid, _split_helper, _gather_helper
from ..testutils import disable_tf32, set_seed, compare_tensors

class TestDistributedLayers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    def setUp(self):
        disable_tf32()


    def _split_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_local = _split_helper(tensor, dim=hdim, group=self.h_group)
        tensor_local = _split_helper(tensor_local, dim=wdim, group=self.w_group)
        return tensor_local


    def _gather_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_gather = _gather_helper(tensor, dim=hdim, group=self.h_group)
        tensor_gather = _gather_helper(tensor_gather, dim=wdim, group=self.w_group)

        return tensor_gather


    @parameterized.expand(
        [
            [180, 360, 256, 512, 32,  8, 1e-3],
            [181, 360, 181, 360, 1, 10, 1e-3],
            [180, 360, 128, 256, 32,  8, 1e-4],
            [181, 360,  91, 180, 1, 10, 1e-4],
            [128, 256, 256, 512, 32,  8, 1e-4],
            [ 91, 180, 181, 360, 1, 10, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectral_conv(self, nlat_in, nlon_in, nlat_out, nlon_out, batch_size, num_chan, tol, verbose=False):
        B, C, Hi, Wi, Ho, Wo = batch_size, num_chan, nlat_in, nlon_in, nlat_out, nlon_out

        from makani.models.common import SpectralConv

        # set up handles
        forward_transform_local = th.RealSHT(nlat=Hi, nlon=Wi).to(self.device)
        inverse_transform_local = th.InverseRealSHT(nlat=Ho, nlon=Wo, lmax=forward_transform_local.lmax, mmax=forward_transform_local.mmax).to(self.device)
        forward_transform_dist = thd.DistributedRealSHT(nlat=Hi, nlon=Wi).to(self.device)
        inverse_transform_dist = thd.DistributedInverseRealSHT(nlat=Ho, nlon=Wo, lmax=forward_transform_dist.lmax, mmax=forward_transform_dist.mmax).to(self.device)

        set_seed(333)

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
            # B, H, W, C_in, C_out, bias, input_format,    freeze_mode, tol
            [2, 8, 16, 16, 24, True,  "nchw",        "none",   1e-5],
            [2, 8, 16, 16, 24, True,  "nchw",        "input",  1e-5],
            [2, 8, 16, 16, 24, True,  "nchw",        "weight", 1e-5],
            [2, 8, 16, 16, 24, True,  "nchw",        "bias",   1e-5],
            [2, 8, 16, 16, 24, False, "nchw",        "none",   1e-5],
            [2, 1, 32, 16, 24, True,  "traditional", "none",   1e-5],
            [2, 1, 32, 16, 24, True,  "traditional", "input",  1e-5],
            [2, 1, 32, 16, 24, True,  "traditional", "weight", 1e-5],
            [2, 1, 32, 16, 24, True,  "traditional", "bias",   1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_matmul(self, B, H, W, C_in, C_out, bias, input_format, freeze_mode, tol, verbose=False):
        """Compare DistributedMatmul to a serial F.conv2d / F.linear reference.

        Uses the ``fin``/``fout`` feature-parallel comm groups (sized via the
        ``GRID_FIN`` / ``GRID_FOUT`` env vars in ``_init_grid``). ``freeze_mode``
        picks one tensor to mark as not requiring grad and asserts that its
        gradient is None while the remaining gradients still match the serial
        reference.
        """
        from makani.mpu.layers import DistributedMatmul

        comm_inp_name = "fin"
        comm_out_name = "fout"
        inp_size = comm.get_size(comm_inp_name)
        out_size = comm.get_size(comm_out_name)

        if C_in % inp_size != 0 or C_out % out_size != 0:
            self.skipTest(
                f"channel dims ({C_in}, {C_out}) not divisible by comm sizes "
                f"({inp_size}, {out_size})"
            )

        set_seed(333)

        # build full reference inputs / params, then derive layouts
        if input_format == "nchw":
            inp_full = torch.randn((B, C_in, H, W), dtype=torch.float32, device=self.device)
            weight_full = torch.randn((C_out, C_in, 1, 1), dtype=torch.float32, device=self.device)
            bias_full = torch.randn((1, C_out, 1, 1), dtype=torch.float32, device=self.device) if bias else None
            chan_dim_inp = 1
            chan_dim_out = 1
            bias_chan_dim = 1
        else:  # traditional: (..., C)
            inp_full = torch.randn((B, H * W, C_in), dtype=torch.float32, device=self.device)
            weight_full = torch.randn((C_out, C_in), dtype=torch.float32, device=self.device)
            bias_full = torch.randn((C_out,), dtype=torch.float32, device=self.device) if bias else None
            chan_dim_inp = -1
            chan_dim_out = -1
            bias_chan_dim = 0

        ################################################################
        # serial reference
        ################################################################
        inp_ref = inp_full.detach().clone()
        weight_ref = weight_full.detach().clone()
        bias_ref = bias_full.detach().clone() if bias else None

        inp_ref.requires_grad = (freeze_mode != "input")
        weight_ref.requires_grad = (freeze_mode != "weight")
        if bias_ref is not None:
            bias_ref.requires_grad = (freeze_mode != "bias")

        if input_format == "nchw":
            out_ref = nn.functional.conv2d(inp_ref, weight_ref, bias=None)
        else:
            out_ref = nn.functional.linear(inp_ref, weight_ref)
        if bias_ref is not None:
            out_ref = out_ref + bias_ref

        with torch.no_grad():
            ograd_full = torch.randn_like(out_ref)
        out_ref.backward(ograd_full)

        igrad_ref = inp_ref.grad.clone() if inp_ref.grad is not None else None
        wgrad_ref = weight_ref.grad.clone() if weight_ref.grad is not None else None
        bgrad_ref = bias_ref.grad.clone() if (bias_ref is not None and bias_ref.grad is not None) else None

        ################################################################
        # distributed version
        ################################################################
        matmul_dist = DistributedMatmul(
            C_in, C_out,
            input_format=input_format,
            comm_inp_name=comm_inp_name,
            comm_out_name=comm_out_name,
            bias=bias,
        ).to(self.device)

        # copy the matching shard of the reference weight/bias into the dist module
        with torch.no_grad():
            w_local = _split_helper(weight_full, dim=0, group=comm.get_group(comm_out_name))
            w_local = _split_helper(w_local, dim=1, group=comm.get_group(comm_inp_name))
            matmul_dist.weight.copy_(w_local)
            if bias:
                b_local = _split_helper(bias_full, dim=bias_chan_dim, group=comm.get_group(comm_out_name))
                matmul_dist.bias.copy_(b_local)

        # freeze parameters according to mode
        matmul_dist.weight.requires_grad_(freeze_mode != "weight")
        if bias:
            matmul_dist.bias.requires_grad_(freeze_mode != "bias")

        # split input along its channel dim
        inp_dist = _split_helper(inp_full, dim=chan_dim_inp, group=comm.get_group(comm_inp_name))
        inp_dist = inp_dist.detach().clone()
        inp_dist.requires_grad = (freeze_mode != "input")

        out_dist = matmul_dist(inp_dist)
        ograd_dist = _split_helper(ograd_full, dim=chan_dim_out, group=comm.get_group(comm_out_name))
        out_dist.backward(ograd_dist)

        ################################################################
        # compare
        ################################################################
        with self.subTest(desc="output"):
            out_gathered = _gather_helper(out_dist, dim=chan_dim_out, group=comm.get_group(comm_out_name))
            self.assertTrue(compare_tensors("output", out_gathered, out_ref.detach(), tol, tol, verbose=verbose))

        with self.subTest(desc="input grad"):
            if freeze_mode == "input":
                self.assertIsNone(inp_dist.grad)
            else:
                igrad_gathered = _gather_helper(inp_dist.grad, dim=chan_dim_inp, group=comm.get_group(comm_inp_name))
                self.assertTrue(compare_tensors("input grad", igrad_gathered, igrad_ref, tol, tol, verbose=verbose))

        with self.subTest(desc="weight grad"):
            if freeze_mode == "weight":
                self.assertIsNone(matmul_dist.weight.grad)
            else:
                wg = _gather_helper(matmul_dist.weight.grad, dim=0, group=comm.get_group(comm_out_name))
                wg = _gather_helper(wg, dim=1, group=comm.get_group(comm_inp_name))
                self.assertTrue(compare_tensors("weight grad", wg, wgrad_ref, tol, tol, verbose=verbose))

        if bias:
            with self.subTest(desc="bias grad"):
                if freeze_mode == "bias":
                    self.assertIsNone(matmul_dist.bias.grad)
                else:
                    bg = _gather_helper(matmul_dist.bias.grad, dim=bias_chan_dim, group=comm.get_group(comm_out_name))
                    self.assertTrue(compare_tensors("bias grad", bg, bgrad_ref, tol, tol, verbose=verbose))


    @parameterized.expand(
        [
            [256, 512, 32, 8, True, 1e-4],
            [181, 360, 1, 10, True, 1e-4],
            [256, 512, 32, 8, False, 1e-4],
            [181, 360, 1, 10, False, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_instance_norm_2d(self, nlat, nlon, batch_size, num_chan, affine, tol, verbose=False):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        set_seed(333)

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
            [181, 360, 1, 4, "equiangular", True, 1e-4],
            [181, 360, 1, 4, "equiangular", False, 1e-4],
            [180, 360, 1, 10, "legendre-gauss", True, 1e-4],
            [180, 360, 1, 10, "legendre-gauss", False, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_geometric_instance_norm_s2(self, nlat, nlon, batch_size, num_chan, grid_type, affine, tol, verbose=False):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        # set up layer norm parameters
        img_shape = (H, W)
        crop_shape = (H, W)
        crop_offset = (0, 0)
        eps = 1e-5

        set_seed(333)

        # create local (serial) layer norm
        norm_local = GeometricInstanceNormS2(
            img_shape=img_shape,
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            grid_type=grid_type,
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
