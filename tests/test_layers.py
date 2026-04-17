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

import sys
import os
import unittest
from parameterized import parameterized

import torch

from makani.models.networks.pangu import EarthAttention3D
from makani.models.common.layers import (
    SeededDropout2d,
    LayerScale,
    UpSample2D,
    DownSample2D,
    UpSample3D,
    DownSample3D,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, get_default_parameters, compare_tensors

class TestLayers(unittest.TestCase):

    def setUp(self):

        disable_tf32()

        self.params = get_default_parameters()

        self.params.history_normalization_mode = "none"

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = 36
        self.params.img_shape_y = 72
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_shape_x_resampled = self.params.img_shape_x
        self.params.img_shape_y_resampled = self.params.img_shape_y
        self.params.img_local_offset_x = 0
        self.params.img_local_offset_y = 0
        self.params.img_crop_offset_x = 0
        self.params.img_crop_offset_y = 0

        # also set the batch size for testing
        self.params.batch_size = 4

        # set device and seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(333)

        return

    @parameterized.expand(
        [
            (1, 16, 1, 1e-7, 1e-5),
            (4, 16, 1, 1e-7, 1e-5),
            (1, 16, 2, 1e-7, 1e-5),
        ], skip_on_empty=True
    )
    def test_earth_attention_3d(self, batch_size, num_channels, num_heads, atol, rtol, verbose=False):
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
        with self.subTest(desc="output"):
            self.assertTrue(compare_tensors("output", out_sdpa, out_naive, atol=atol, rtol=rtol, verbose=verbose))
        
        #############################################################
        # evaluate BWD pass
        #############################################################
        # igrads
        with self.subTest(desc="input gradient"):
            self.assertTrue(compare_tensors("input gradient", igrad_sdpa, igrad_naive, atol=atol, rtol=rtol, verbose=verbose))
        
        # wgrads
        with torch.no_grad():
            for (_, ngrad), (skey, sgrad) in zip(ea_naive.named_parameters(), ea_sdpa.named_parameters()):
                with self.subTest(desc=f"weight gradient {skey}"):
                    self.assertTrue(compare_tensors(f"weight gradient {skey}", sgrad, ngrad, atol=atol, rtol=rtol, verbose=verbose))


    def test_seeded_dropout2d_deterministic_mask(self, atol=1e-8, rtol=1e-8, verbose=False):
        """Two dropout layers with the same seed should produce identical masks."""
        set_seed(333)
        x = torch.randn(2, 3, 4, 4, device=self.device)

        drop1 = SeededDropout2d(drop_prob=0.5, seed=999).to(self.device).train()
        drop2 = SeededDropout2d(drop_prob=0.5, seed=999).to(self.device).train()

        out1 = drop1(x)
        out2 = drop2(x)

        self.assertTrue(compare_tensors("output", out1, out2, atol=atol, rtol=rtol, verbose=verbose))

    # ---------------------------------------------------------------------
    # LayerScale
    # ---------------------------------------------------------------------

    def test_layerscale_shape_preserved(self):
        """LayerScale is (B, C, H, W) → (B, C, H, W)."""
        C, H, W = 4, 8, 12
        layer = LayerScale(num_chans=C, init_value=0.1).to(self.device)
        x = torch.randn(2, C, H, W, device=self.device)
        out = layer(x)
        self.assertEqual(tuple(out.shape), (2, C, H, W))

    def test_layerscale_init_value_is_constant_multiplier(self, verbose=False):
        """At init, LayerScale(init_value=v)(x) equals v * x elementwise."""
        C, H, W = 4, 5, 7
        v = 0.25
        layer = LayerScale(num_chans=C, init_value=v).to(self.device)
        x = torch.randn(3, C, H, W, device=self.device)
        out = layer(x)
        self.assertTrue(compare_tensors("layerscale init v*x", out, v * x,
                                        atol=1e-6, rtol=1e-5, verbose=verbose))

    def test_layerscale_per_channel_weight(self, verbose=False):
        """Setting a distinct weight per channel scales each channel independently."""
        C, H, W = 3, 4, 6
        layer = LayerScale(num_chans=C, init_value=0.0).to(self.device)
        # inject distinct values [1, 2, 3]
        with torch.no_grad():
            layer.weight.copy_(torch.arange(1, C + 1, dtype=torch.float32,
                                            device=self.device).reshape(C, 1, 1, 1))
        x = torch.randn(2, C, H, W, device=self.device)
        out = layer(x)
        scale = torch.arange(1, C + 1, dtype=torch.float32,
                             device=self.device).reshape(1, C, 1, 1)
        self.assertTrue(compare_tensors("layerscale per-channel", out, x * scale,
                                        atol=1e-6, rtol=1e-5, verbose=verbose))

    def test_layerscale_gradients_flow(self):
        """LayerScale is differentiable w.r.t. both input and weight."""
        C, H, W = 3, 5, 5
        layer = LayerScale(num_chans=C, init_value=0.5).to(self.device)
        x = torch.randn(2, C, H, W, device=self.device, requires_grad=True)
        layer(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertTrue(torch.isfinite(layer.weight.grad).all().item())

    # ---------------------------------------------------------------------
    # UpSample2D  (in_dim must equal 2 * out_dim)
    # ---------------------------------------------------------------------

    def test_upsample2d_shape_3d_input(self):
        """3-D input (B, N, in_dim) → (B, out_lat, out_lon, out_dim)."""
        in_dim, out_dim = 16, 8
        in_res, out_res = (4, 6), (6, 10)
        layer = UpSample2D(in_dim, out_dim, in_res, out_res).to(self.device)
        x = torch.randn(2, in_res[0] * in_res[1], in_dim, device=self.device)
        out = layer(x)
        self.assertEqual(tuple(out.shape), (2, out_res[0], out_res[1], out_dim))

    def test_upsample2d_shape_4d_input(self):
        """4-D input (B, in_lat, in_lon, in_dim) → (B, out_lat, out_lon, out_dim)."""
        in_dim, out_dim = 16, 8
        in_res, out_res = (4, 6), (6, 10)
        layer = UpSample2D(in_dim, out_dim, in_res, out_res).to(self.device)
        x = torch.randn(2, in_res[0], in_res[1], in_dim, device=self.device)
        out = layer(x)
        self.assertEqual(tuple(out.shape), (2, out_res[0], out_res[1], out_dim))

    def test_upsample2d_4d_wrong_resolution_raises(self):
        """4-D input whose (lat, lon) doesn't match input_resolution triggers assertion."""
        in_dim, out_dim = 16, 8
        in_res, out_res = (4, 6), (6, 10)
        layer = UpSample2D(in_dim, out_dim, in_res, out_res).to(self.device)
        bad = torch.randn(2, 5, 6, in_dim, device=self.device)
        with self.assertRaises(AssertionError):
            layer(bad)

    def test_upsample2d_gradients_flow(self):
        """UpSample2D passes gradients back to the input."""
        in_dim, out_dim = 16, 8
        in_res, out_res = (4, 6), (6, 10)
        layer = UpSample2D(in_dim, out_dim, in_res, out_res).to(self.device)
        x = torch.randn(2, in_res[0] * in_res[1], in_dim,
                        device=self.device, requires_grad=True)
        layer(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    # ---------------------------------------------------------------------
    # DownSample2D  (output channel count is 2 * in_dim)
    # ---------------------------------------------------------------------

    def test_downsample2d_shape_3d_input(self):
        """3-D input (B, N, in_dim) → (B, out_lat, out_lon, 2*in_dim)."""
        in_dim = 8
        in_res, out_res = (6, 10), (4, 8)
        layer = DownSample2D(in_dim, in_res, out_res).to(self.device)
        x = torch.randn(2, in_res[0] * in_res[1], in_dim, device=self.device)
        out = layer(x)
        self.assertEqual(tuple(out.shape), (2, out_res[0], out_res[1], 2 * in_dim))

    def test_downsample2d_shape_4d_input(self):
        """4-D input (B, in_lat, in_lon, in_dim) → (B, out_lat, out_lon, 2*in_dim)."""
        in_dim = 8
        in_res, out_res = (6, 10), (4, 8)
        layer = DownSample2D(in_dim, in_res, out_res).to(self.device)
        x = torch.randn(2, in_res[0], in_res[1], in_dim, device=self.device)
        out = layer(x)
        self.assertEqual(tuple(out.shape), (2, out_res[0], out_res[1], 2 * in_dim))

    def test_downsample2d_4d_wrong_resolution_raises(self):
        """4-D input whose (lat, lon) doesn't match input_resolution triggers assertion."""
        in_dim = 8
        in_res, out_res = (6, 10), (4, 8)
        layer = DownSample2D(in_dim, in_res, out_res).to(self.device)
        bad = torch.randn(2, 7, 10, in_dim, device=self.device)
        with self.assertRaises(AssertionError):
            layer(bad)

    def test_downsample2d_gradients_flow(self):
        """DownSample2D passes gradients back to the input."""
        in_dim = 8
        in_res, out_res = (6, 10), (4, 8)
        layer = DownSample2D(in_dim, in_res, out_res).to(self.device)
        x = torch.randn(2, in_res[0] * in_res[1], in_dim,
                        device=self.device, requires_grad=True)
        layer(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    # ---------------------------------------------------------------------
    # UpSample3D  (in_dim must equal 2 * out_dim; out_pl ≤ in_pl)
    # ---------------------------------------------------------------------

    def test_upsample3d_shape(self):
        """3-D output shape: (B, out_pl*out_lat*out_lon, out_dim)."""
        in_dim, out_dim = 16, 8
        in_res  = (3, 4, 6)
        out_res = (3, 6, 10)
        layer = UpSample3D(in_dim, out_dim, in_res, out_res).to(self.device)
        N = in_res[0] * in_res[1] * in_res[2]
        x = torch.randn(2, N, in_dim, device=self.device)
        out = layer(x)
        M = out_res[0] * out_res[1] * out_res[2]
        self.assertEqual(tuple(out.shape), (2, M, out_dim))

    def test_upsample3d_gradients_flow(self):
        """UpSample3D passes gradients back to the input."""
        in_dim, out_dim = 16, 8
        in_res, out_res = (3, 4, 6), (3, 6, 10)
        layer = UpSample3D(in_dim, out_dim, in_res, out_res).to(self.device)
        N = in_res[0] * in_res[1] * in_res[2]
        x = torch.randn(2, N, in_dim, device=self.device, requires_grad=True)
        layer(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    # ---------------------------------------------------------------------
    # DownSample3D  (in_pl == out_pl; output channel count = 2 * in_dim)
    # ---------------------------------------------------------------------

    def test_downsample3d_shape(self):
        """3-D output shape: (B, out_pl*out_lat*out_lon, 2*in_dim)."""
        in_dim = 8
        in_res  = (3, 6, 10)
        out_res = (3, 4, 8)
        layer = DownSample3D(in_dim, in_res, out_res).to(self.device)
        N = in_res[0] * in_res[1] * in_res[2]
        x = torch.randn(2, N, in_dim, device=self.device)
        out = layer(x)
        M = out_res[0] * out_res[1] * out_res[2]
        self.assertEqual(tuple(out.shape), (2, M, 2 * in_dim))

    def test_downsample3d_gradients_flow(self):
        """DownSample3D passes gradients back to the input."""
        in_dim = 8
        in_res, out_res = (3, 6, 10), (3, 4, 8)
        layer = DownSample3D(in_dim, in_res, out_res).to(self.device)
        N = in_res[0] * in_res[1] * in_res[2]
        x = torch.randn(2, N, in_dim, device=self.device, requires_grad=True)
        layer(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())


if __name__ == "__main__":
    unittest.main()
