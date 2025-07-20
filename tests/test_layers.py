# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
from parameterized import parameterized

import torch

from makani.models.networks.pangu import EarthAttention3D

from makani.utils import functions as fn
from testutils import get_default_parameters

class TestLayers(unittest.TestCase):

    def setUp(self):
        self.params = get_default_parameters()

        self.params.history_normalization_mode = "none"

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = 36
        self.params.img_shape_y = 72
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_local_offset_x = 0
        self.params.img_local_offset_y = 0
        self.params.img_crop_offset_x = 0
        self.params.img_crop_offset_y = 0

        # also set the batch size for testing
        self.params.batch_size = 4

        # set device and seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        return

    @parameterized.expand(
        [
            (1, 16, 1, 1e-7, 1e-5),
            (4, 16, 1, 1e-7, 1e-5),
            (1, 16, 2, 1e-7, 1e-5),
        ], skip_on_empty=True
    )
    def test_earth_attention_3d(self, batch_size, num_channels, num_heads, atol, rtol, verbose=True):
        """
        Tests initialization of all the models and the forward and backward pass
        """

        # some parameters
        pressure_levels = 11
        
        ea_naive = EarthAttention3D(dim=num_channels,
                                    input_resolution=(2, 6, 12),
                                    window_size=(2, 6, 12),
                                    num_heads=num_heads,
                                    qkv_bias=True,
	                            qk_scale=None,
                                    attn_drop=0.0,
                                    proj_drop=0.0,
                                    use_sdpa=False).to(self.device)

        ea_sdpa = EarthAttention3D(dim=num_channels,
                                    input_resolution=(2, 6, 12),
                                    window_size=(2, 6, 12),
                                    num_heads=num_heads,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    attn_drop=0.0,
                                    proj_drop=0.0,
                                    use_sdpa=True).to(self.device)

        # copy weights
        with torch.no_grad():
            # earth position bias
            ea_sdpa.earth_position_bias_table.copy_(ea_naive.earth_position_bias_table)
            # linear inputs
            ea_sdpa.qkv.weight.copy_(ea_naive.qkv.weight)
            ea_sdpa.qkv.bias.copy_(ea_naive.qkv.bias)
            # linear outputs
            ea_sdpa.proj.weight.copy_(ea_naive.proj.weight)
            ea_sdpa.proj.bias.copy_(ea_naive.proj.bias)

        # prepare some dummy data
        #[8, 1, 144, 8]
        inp_shape = (8 * batch_size, 1, 144, num_channels)
        inp = torch.randn(*inp_shape, dtype=torch.float32, device=self.device)
        inp.requires_grad = True
        
        # forward/backward pass naive
        inp.grad = None
        out_naive = ea_naive(inp)
        loss = torch.sum(out_naive)
        loss.backward()
        igrad_naive = inp.grad.clone()

        # forward/backward pass sdpa
        inp.grad = None
        out_sdpa = ea_sdpa(inp)
        loss = torch.sum(out_sdpa)
        loss.backward()
        igrad_sdpa = inp.grad.clone()
        
        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            err = fn.relative_error(out_sdpa, out_naive)
            if verbose:
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= rtol)
        
        #############################################################
        # evaluate BWD pass
        #############################################################
        # igrads
        with torch.no_grad():
            err = fn.relative_error(igrad_sdpa, igrad_naive)
            if verbose:
                print(f"final relative error of input gradients: {err.item()}")
        self.assertTrue(err.item() <= rtol)
        
        # wgrads
        with torch.no_grad():
            good = True
            errs = []
            for ngrad, sgrad in zip(ea_naive.parameters(), ea_sdpa.parameters()):
                err = fn.relative_error(sgrad, ngrad)
                if err > rtol:
                    # in some cases, the gradient itself can be small, check absolute tolerance then
                    aerr = fn.absolute_error(sgrad, ngrad)
                    if aerr < atol:
                        continue
                    if verbose:
                        print(f"final relative error of weight gradient {key}: {err.item()}, absolute error: {aerr.item()}")
                    good = False
                errs.append(err)
        merr = torch.stack(errs, dim=0).mean().item()
        if verbose:
            print(f"final relative average error of weight gradients: {merr}")
        self.assertTrue(good)
        
        
        

if __name__ == "__main__":
    unittest.main()
