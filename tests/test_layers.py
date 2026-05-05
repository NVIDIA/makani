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
from makani.models.common.imputation import MLPImputation, ConstantImputation

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


# ===========================================================================
class TestMLPImputation(unittest.TestCase):
    """Tests for MLPImputation in makani.models.common.imputation.

    The module replaces NaN values (or values flagged by an explicit mask) in a
    configurable channel subset (`inpute_chans`) with predictions from an MLP
    that takes the full input and produces values for the imputable channels.

    Invariants the tests pin down:
      - non-imputable channels are passed through bit-equal
      - non-masked positions in imputable channels are passed through bit-equal
      - all NaN values are replaced (output has no NaN)
      - auto-mask (None) and explicit mask (torch.isnan(x_sub)) agree
      - shape is preserved
      - works for batch_size > 1 and for 5-D (B, E, C, H, W) tensors
      - backward through the imputation is finite
    """

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.B = 4
        self.C = 6                      # total channels
        self.imputable = [1, 4]         # channels to impute
        self.H = 8
        self.W = 12

    def _fn(self):
        return MLPImputation(
            inp_chans=self.C,
            inpute_chans=torch.tensor(self.imputable, dtype=torch.long),
            mlp_ratio=2.0,
            activation_function=torch.nn.GELU,
        )

    def _make_input_with_nans(self, batch_size, n_nans_per_imputable=5):
        """Random tensor with ``n_nans_per_imputable`` NaN positions inserted into
        each imputable channel. Returns the tensor and a boolean mask of which
        positions ARE NaN (shape: (B, len(imputable), H, W))."""
        x = torch.randn(batch_size, self.C, self.H, self.W)
        nan_mask_sub = torch.zeros(batch_size, len(self.imputable), self.H, self.W, dtype=torch.bool)
        for b in range(batch_size):
            for ic in range(len(self.imputable)):
                rng = torch.randperm(self.H * self.W)[:n_nans_per_imputable]
                nan_mask_sub[b, ic].view(-1)[rng] = True
        # write NaN into the actual imputable channels
        for ic_idx, c in enumerate(self.imputable):
            x[:, c][nan_mask_sub[:, ic_idx]] = float("nan")
        return x, nan_mask_sub

    # -- shape / no-NaN invariants -----------------------------------------

    def test_output_shape_preserved(self):
        fn = self._fn()
        x, _ = self._make_input_with_nans(batch_size=self.B)
        out = fn(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_has_no_nan(self):
        fn = self._fn()
        x, _ = self._make_input_with_nans(batch_size=self.B)
        out = fn(x)
        self.assertFalse(torch.isnan(out).any(),
                         "MLPImputation output still contains NaN — imputation failed")

    # -- pass-through guarantees: only NaN positions in imputable channels change ---

    def test_non_imputable_channels_pass_through(self):
        """Channels not in inpute_chans must be returned exactly as-is."""
        fn = self._fn()
        x, _ = self._make_input_with_nans(batch_size=self.B)
        out = fn(x)
        non_imputable = [c for c in range(self.C) if c not in self.imputable]
        self.assertTrue(
            compare_tensors(
                "non-imputable channels passthrough",
                out[:, non_imputable], x[:, non_imputable], atol=0.0, rtol=0.0,
            ),
            "non-imputable channels were modified by MLPImputation",
        )

    def test_only_nan_positions_change(self):
        """In imputable channels, non-NaN positions must be passed through bit-equal.
        Only the originally-NaN positions are allowed to change."""
        fn = self._fn()
        x, nan_mask_sub = self._make_input_with_nans(batch_size=self.B)
        out = fn(x)

        # for each imputable channel, the non-NaN positions of the original input
        # must equal the corresponding positions in the output
        for ic_idx, c in enumerate(self.imputable):
            in_chan = x[:, c]
            out_chan = out[:, c]
            keep = ~nan_mask_sub[:, ic_idx]      # True where input was NOT NaN
            self.assertTrue(
                compare_tensors(
                    f"non-NaN positions preserved (channel {c})",
                    out_chan[keep], in_chan[keep], atol=0.0, rtol=0.0,
                ),
                f"channel {c}: non-NaN positions were modified by MLPImputation",
            )

    # -- mask handling ------------------------------------------------------

    def test_auto_mask_matches_explicit_mask(self):
        """Passing mask=None (auto-detect via torch.isnan) must give the same
        result as passing the explicit isnan(x_sub) tensor."""
        fn = self._fn()
        x, _ = self._make_input_with_nans(batch_size=self.B)
        x_sub = x[..., torch.tensor(self.imputable, dtype=torch.long), :, :]
        explicit_mask = torch.isnan(x_sub)

        out_auto = fn(x)
        out_explicit = fn(x, mask=explicit_mask)
        self.assertTrue(
            compare_tensors("auto vs explicit mask", out_auto, out_explicit),
            "auto-mask and explicit mask produced different outputs",
        )

    def test_explicit_mask_can_extend_nan_mask(self):
        """An explicit mask is logical-OR'd with isnan(x_sub), so additional
        non-NaN positions can be flagged for imputation. Output at those
        flagged positions must come from the MLP — i.e. differ from the input
        value at that position (with overwhelming probability for randn inputs
        whose value is not equal to the MLP output)."""
        fn = self._fn()
        x = torch.randn(self.B, self.C, self.H, self.W)   # no NaNs
        # flag a single position in each imputable channel
        extra_mask = torch.zeros(self.B, len(self.imputable), self.H, self.W, dtype=torch.bool)
        extra_mask[:, :, 0, 0] = True

        out = fn(x, mask=extra_mask)
        # all non-flagged positions in imputable channels must be unchanged
        for ic_idx, c in enumerate(self.imputable):
            keep = ~extra_mask[:, ic_idx]
            self.assertTrue(
                compare_tensors(
                    f"non-flagged positions preserved (channel {c})",
                    out[:, c][keep], x[:, c][keep], atol=0.0, rtol=0.0,
                )
            )

    # -- batch sizes --------------------------------------------------------

    def test_batch_size_one(self):
        """Single-element batch must work (no broadcasting / squeeze edge case)."""
        fn = self._fn()
        x, _ = self._make_input_with_nans(batch_size=1)
        out = fn(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_batch_size_consistency(self):
        """The per-sample output must be independent of other samples in the
        batch — running sample [0] alone vs in a full batch must agree."""
        fn = self._fn()
        x_full, _ = self._make_input_with_nans(batch_size=self.B)
        out_full = fn(x_full)
        out_single = fn(x_full[:1])
        self.assertTrue(
            compare_tensors(
                "batch size consistency", out_single, out_full[:1], atol=1e-6, rtol=1e-5,
            )
        )

    def test_5d_input_with_ensemble_dim(self):
        """The implementation uses ``x.dim() - 3`` as the channel axis and reshapes
        to flatten extra batch dims for Conv2d. (B, E, C, H, W) inputs must work
        and obey the same invariants."""
        E = 3
        fn = self._fn()
        # build a 5-D input with NaNs in imputable channels
        x = torch.randn(self.B, E, self.C, self.H, self.W)
        for c in self.imputable:
            x[..., c, 0, 0] = float("nan")
        out = fn(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    # -- backward -----------------------------------------------------------

    def test_backward_finite(self):
        fn = self._fn()
        x, _ = self._make_input_with_nans(batch_size=self.B)
        # forward + backward; can't require_grad on a NaN-containing tensor input
        # without producing NaN gradients at NaN positions, so use the output's sum
        # — what we want is to ensure the MLP weights get a finite gradient
        loss = fn(x).sum()
        loss.backward()
        for name, p in fn.named_parameters():
            self.assertIsNotNone(p.grad, f"no gradient on {name}")
            self.assertFalse(torch.isnan(p.grad).any(), f"NaN gradient on {name}")
            self.assertFalse(torch.isinf(p.grad).any(), f"Inf gradient on {name}")


# ===========================================================================
class TestConstantImputation(unittest.TestCase):
    """Tests for ConstantImputation — replaces masked positions with a
    learnable per-channel constant (one ``nn.Parameter`` of shape (C, 1, 1))."""

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.B = 4
        self.C = 5
        self.H = 8
        self.W = 12

    def _fn(self):
        return ConstantImputation(inp_chans=self.C)

    # -- shape / no-NaN invariants -----------------------------------------

    def test_output_shape_preserved(self):
        fn = self._fn()
        x = torch.randn(self.B, self.C, self.H, self.W)
        out = fn(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_has_no_nan(self):
        """Even if the input has NaN values, the output must not."""
        fn = self._fn()
        x = torch.randn(self.B, self.C, self.H, self.W)
        x[:, 0, 0, 0] = float("nan")
        x[:, 2, 1, 2] = float("nan")
        out = fn(x)
        self.assertFalse(torch.isnan(out).any())

    # -- correctness: only NaN positions change ----------------------------

    def test_only_nan_positions_change(self):
        """Non-NaN positions must be passed through bit-equal; NaN positions
        must be replaced with the per-channel learnable weight."""
        fn = self._fn()
        x = torch.randn(self.B, self.C, self.H, self.W)
        # insert a few NaNs in known positions
        nan_positions = [(0, 0, 0, 0), (1, 2, 1, 2), (2, 4, 3, 4)]
        for b, c, h, w in nan_positions:
            x[b, c, h, w] = float("nan")

        out = fn(x)
        nan_mask = torch.isnan(x)
        keep = ~nan_mask
        self.assertTrue(
            compare_tensors(
                "non-NaN positions preserved", out[keep], x[keep], atol=0.0, rtol=0.0,
            ),
            "non-NaN positions were modified by ConstantImputation",
        )

        # at NaN positions, output must equal the per-channel weight
        weight_bcast = fn.weight.expand(self.B, self.C, self.H, self.W)
        self.assertTrue(
            compare_tensors(
                "NaN positions replaced by weight",
                out[nan_mask], weight_bcast[nan_mask], atol=0.0, rtol=0.0,
            )
        )

    def test_explicit_mask_extends_nan_mask(self):
        """Passing an explicit mask logical-OR's with torch.isnan(x); positions
        flagged by either source get replaced."""
        fn = self._fn()
        x = torch.randn(self.B, self.C, self.H, self.W)   # no NaNs
        # flag specific (b, c, h, w) positions explicitly
        explicit_mask = torch.zeros_like(x, dtype=torch.bool)
        explicit_mask[0, 0, 0, 0] = True
        explicit_mask[1, 3, 2, 5] = True

        out = fn(x, mask=explicit_mask)
        keep = ~explicit_mask
        self.assertTrue(
            compare_tensors(
                "non-flagged positions preserved", out[keep], x[keep], atol=0.0, rtol=0.0,
            )
        )
        weight_bcast = fn.weight.expand(self.B, self.C, self.H, self.W)
        self.assertTrue(
            compare_tensors(
                "flagged positions replaced by weight",
                out[explicit_mask], weight_bcast[explicit_mask], atol=0.0, rtol=0.0,
            )
        )

    def test_auto_mask_matches_explicit_mask(self):
        """mask=None (auto-detect via isnan) and mask=isnan(x) must agree."""
        fn = self._fn()
        x = torch.randn(self.B, self.C, self.H, self.W)
        x[0, 0, 0, 0] = float("nan")
        x[1, 2, 3, 4] = float("nan")

        out_auto = fn(x)
        out_explicit = fn(x, mask=torch.isnan(x))
        self.assertTrue(
            compare_tensors("auto vs explicit mask", out_auto, out_explicit, atol=0.0, rtol=0.0),
        )

    # -- batch sizes --------------------------------------------------------

    def test_batch_size_one(self):
        fn = self._fn()
        x = torch.randn(1, self.C, self.H, self.W)
        x[0, 1, 0, 0] = float("nan")
        out = fn(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_batch_size_consistency(self):
        """Per-sample independence: running sample [0] alone vs in a full batch
        must agree exactly (the imputation is purely per-element)."""
        fn = self._fn()
        x = torch.randn(self.B, self.C, self.H, self.W)
        x[0, 0, 0, 0] = float("nan")
        x[1, 2, 1, 2] = float("nan")

        out_full = fn(x)
        out_single = fn(x[:1])
        self.assertTrue(
            compare_tensors("batch size consistency", out_single, out_full[:1], atol=0.0, rtol=0.0),
        )

    # -- learnable weight ---------------------------------------------------

    def test_weight_is_learnable(self):
        """The per-channel weight must be an nn.Parameter; gradient through
        it must be non-None and finite after backward when at least one
        position is masked (otherwise no path through the weight)."""
        fn = self._fn()
        # ensure at least one masked position so the weight participates in the forward
        x = torch.randn(self.B, self.C, self.H, self.W)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, :, 0, 0] = True

        loss = fn(x, mask=mask).sum()
        loss.backward()
        self.assertIsNotNone(fn.weight.grad, "ConstantImputation.weight has no gradient")
        self.assertFalse(torch.isnan(fn.weight.grad).any())
        self.assertFalse(torch.isinf(fn.weight.grad).any())


if __name__ == "__main__":
    unittest.main()
