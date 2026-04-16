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
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import unittest
import torch

from makani.models.preprocessor import Preprocessor2D

from .testutils import set_seed, get_default_parameters, compare_tensors, IMG_SIZE_H, IMG_SIZE_W, NUM_CHANNELS


class TestPreprocessor2DBasic(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.pp = Preprocessor2D(self.params)
        self.B = self.params.batch_size       # 1
        self.C = self.params.N_in_channels    # NUM_CHANNELS
        self.H = self.params.img_shape_x      # IMG_SIZE_H
        self.W = self.params.img_shape_y      # IMG_SIZE_W

    def _rand(self, *shape):
        return torch.randn(*shape)

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def test_construction_does_not_raise(self):
        """Minimal params construct a Preprocessor2D without error."""
        pp = Preprocessor2D(get_default_parameters())
        self.assertIsInstance(pp, Preprocessor2D)

    def test_construction_residual_target(self):
        """target='residual' with normalize_residual=False constructs correctly."""
        params = get_default_parameters()
        params.target = "residual"
        params.normalize_residual = False
        pp = Preprocessor2D(params)
        self.assertTrue(pp.learn_residual)

    def test_construction_default_target_not_residual(self):
        """target='default' sets learn_residual=False."""
        self.assertFalse(self.pp.learn_residual)

    # -----------------------------------------------------------------------
    # flatten_history / expand_history
    # -----------------------------------------------------------------------

    def test_flatten_history_4d_noop(self, verbose=False):
        """flatten_history on a 4-D tensor returns it unchanged."""
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.flatten_history(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(compare_tensors("flatten_history 4D noop", out, x, verbose=verbose))

    def test_flatten_history_5d_collapses_time(self):
        """flatten_history on (B,T,C,H,W) gives (B, T*C, H, W)."""
        T = 3
        x = self._rand(self.B, T, self.C, self.H, self.W)
        out = self.pp.flatten_history(x)
        self.assertEqual(tuple(out.shape), (self.B, T * self.C, self.H, self.W))

    def test_expand_history_5d_noop(self):
        """expand_history on a 5-D tensor returns it unchanged."""
        T = 2
        x = self._rand(self.B, T, self.C, self.H, self.W)
        out = self.pp.expand_history(x, nhist=T)
        self.assertEqual(out.shape, x.shape)

    def test_expand_history_4d_adds_time(self):
        """expand_history on (B, T*C, H, W) gives (B, T, C, H, W)."""
        T = 3
        x = self._rand(self.B, T * self.C, self.H, self.W)
        out = self.pp.expand_history(x, nhist=T)
        self.assertEqual(tuple(out.shape), (self.B, T, self.C, self.H, self.W))

    def test_flatten_expand_roundtrip(self, verbose=False):
        """flatten_history(expand_history(x, T)) is bit-identical to x."""
        T = 2
        x = self._rand(self.B, T * self.C, self.H, self.W)
        out = self.pp.flatten_history(self.pp.expand_history(x, nhist=T))
        self.assertTrue(compare_tensors("flatten/expand roundtrip", out, x, verbose=verbose))

    # -----------------------------------------------------------------------
    # add_residual
    # -----------------------------------------------------------------------

    def test_add_residual_default_mode_returns_dx(self, verbose=False):
        """In non-residual mode, add_residual(x, dx) returns dx unchanged."""
        x  = self._rand(self.B, self.C, self.H, self.W)
        dx = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.add_residual(x, dx)
        self.assertTrue(compare_tensors("add_residual default", out, dx, verbose=verbose))

    def test_add_residual_residual_mode_adds_to_last_step(self, verbose=True):
        """In residual mode, add_residual adds dx to the last time-step of x."""
        params = get_default_parameters()
        params.target = "residual"
        params.n_history = 1
        pp = Preprocessor2D(params)
        T = 2  # n_history + 1
        x  = self._rand(self.B, T * self.C, self.H, self.W)
        dx = self._rand(self.B, self.C, self.H, self.W)

        out = pp.add_residual(x, dx)

        out_5d = pp.expand_history(out, nhist=T)
        x_5d   = pp.expand_history(x, nhist=T)
        self.assertTrue(compare_tensors("add_residual last step", out_5d[:, -1], x_5d[:, -1] + dx, verbose=verbose))
        self.assertTrue(compare_tensors("add_residual early step unchanged", out_5d[:, 0], x_5d[:, 0], verbose=verbose))

    # -----------------------------------------------------------------------
    # Static features
    # -----------------------------------------------------------------------

    def test_add_static_features_noop_when_none(self, verbose=False):
        """Without static features, add_static_features is identity."""
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.add_static_features(x)
        self.assertTrue(compare_tensors("add_static noop", out, x, verbose=verbose))

    def test_remove_static_features_noop_when_none(self, verbose=False):
        """Without static features, remove_static_features is identity."""
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.remove_static_features(x)
        self.assertTrue(compare_tensors("remove_static noop", out, x, verbose=verbose))

    def test_add_static_features_increases_channels(self):
        """Raw 2-channel grid appends 2 channels to the input."""
        params = get_default_parameters()
        params.add_grid = True
        params.gridtype = "raw"
        pp = Preprocessor2D(params)
        x = self._rand(self.B, self.C, self.H, self.W)
        out = pp.add_static_features(x)
        self.assertEqual(out.shape[1], self.C + 2)

    def test_add_remove_static_features_roundtrip_shape(self):
        """remove_static_features restores original channel count."""
        params = get_default_parameters()
        params.add_grid = True
        params.gridtype = "raw"
        pp = Preprocessor2D(params)
        x = self._rand(self.B, self.C, self.H, self.W)
        out = pp.remove_static_features(pp.add_static_features(x))
        self.assertEqual(out.shape, x.shape)

    def test_add_remove_static_features_roundtrip_values(self, verbose=False):
        """remove_static_features leaves the original channels bit-identical."""
        params = get_default_parameters()
        params.add_grid = True
        params.gridtype = "raw"
        pp = Preprocessor2D(params)
        x = self._rand(self.B, self.C, self.H, self.W)
        out = pp.remove_static_features(pp.add_static_features(x))
        self.assertTrue(compare_tensors("add/remove static roundtrip", out, x, verbose=verbose))

    def test_get_static_features_returns_tensor_when_enabled(self):
        """get_static_features returns a non-None tensor when grid is enabled."""
        params = get_default_parameters()
        params.add_grid = True
        params.gridtype = "raw"
        pp = Preprocessor2D(params)
        sf = pp.get_static_features()
        self.assertIsNotNone(sf)
        self.assertIsInstance(sf, torch.Tensor)

    def test_get_static_features_returns_none_when_disabled(self):
        """get_static_features returns None when no features are configured."""
        self.assertIsNone(self.pp.get_static_features())

    def test_static_feature_spatial_shape_matches_grid(self):
        """The (H, W) of the stored static feature matches the image shape."""
        params = get_default_parameters()
        params.add_grid = True
        params.gridtype = "raw"
        pp = Preprocessor2D(params)
        sf = pp.get_static_features()
        self.assertEqual(sf.shape[-2], IMG_SIZE_H)
        self.assertEqual(sf.shape[-1], IMG_SIZE_W)

    # -----------------------------------------------------------------------
    # append_history
    # -----------------------------------------------------------------------

    def test_append_history_n_history_0_returns_x2(self, verbose=False):
        """When n_history=0, append_history returns x2 unchanged."""
        x1 = self._rand(self.B, self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.append_history(x1, x2, step=0)
        self.assertTrue(compare_tensors("append_history n_history=0", out, x2, verbose=verbose))

    def test_append_history_n_history_1_shape(self):
        """When n_history=1, append_history returns (B, 2*C, H, W)."""
        params = get_default_parameters()
        params.n_history = 1
        pp = Preprocessor2D(params)
        T = 2
        x1 = self._rand(self.B, T * self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)
        out = pp.append_history(x1, x2, step=0)
        self.assertEqual(tuple(out.shape), (self.B, T * self.C, self.H, self.W))

    def test_append_history_n_history_1_rolls_content(self, verbose=False):
        """append_history drops the oldest time-step and appends x2 as newest."""
        params = get_default_parameters()
        params.n_history = 1
        pp = Preprocessor2D(params)
        T = 2
        x1 = self._rand(self.B, T * self.C, self.H, self.W)
        x2 = self._rand(self.B, self.C, self.H, self.W)

        out = pp.append_history(x1, x2, step=0)
        out_5d = pp.expand_history(out, nhist=T)
        x1_5d = pp.expand_history(x1, nhist=T)

        self.assertTrue(compare_tensors("append_history newest slot", out_5d[:, -1], x2, verbose=verbose))
        self.assertTrue(compare_tensors("append_history oldest slot rolled", out_5d[:, 0], x1_5d[:, 1], verbose=verbose))

    # -----------------------------------------------------------------------
    # cache_unpredicted_features / get_unpredicted_features
    # -----------------------------------------------------------------------

    def test_get_unpredicted_features_none_without_cache(self):
        """Before caching, get_unpredicted_features returns (None, None)."""
        self.pp.eval()
        inp, tar = self.pp.get_unpredicted_features()
        self.assertIsNone(inp)
        self.assertIsNone(tar)

    def test_cache_unpredicted_features_stores_and_retrieves_train(self, verbose=False):
        """Cached xz/yz are retrievable from get_unpredicted_features (train mode)."""
        self.pp.train()
        x  = self._rand(self.B, self.C, self.H, self.W)
        y  = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, y, xz=xz, yz=yz)
        inp, tar = self.pp.get_unpredicted_features()
        self.assertIsNotNone(inp)
        self.assertIsNotNone(tar)
        self.assertTrue(compare_tensors("cache unpredicted inp (train)", inp, xz, verbose=verbose))
        self.assertTrue(compare_tensors("cache unpredicted tar (train)", tar, yz, verbose=verbose))

    def test_cache_unpredicted_features_stores_and_retrieves_eval(self, verbose=False):
        """Cached xz/yz are retrievable in eval mode."""
        self.pp.eval()
        x  = self._rand(self.B, self.C, self.H, self.W)
        y  = self._rand(self.B, self.C, self.H, self.W)
        xz = self._rand(self.B, 1, 1, self.H, self.W)
        yz = self._rand(self.B, 1, 1, self.H, self.W)
        self.pp.cache_unpredicted_features(x, y, xz=xz, yz=yz)
        inp, tar = self.pp.get_unpredicted_features()
        self.assertIsNotNone(inp)
        self.assertIsNotNone(tar)
        self.assertTrue(compare_tensors("cache unpredicted inp (eval)", inp, xz, verbose=verbose))
        self.assertTrue(compare_tensors("cache unpredicted tar (eval)", tar, yz, verbose=verbose))

    def test_cache_unpredicted_features_returns_x_y_unchanged(self, verbose=False):
        """cache_unpredicted_features returns x and y bit-identically."""
        self.pp.train()
        x = self._rand(self.B, self.C, self.H, self.W)
        y = self._rand(self.B, self.C, self.H, self.W)
        xr, yr = self.pp.cache_unpredicted_features(x, y)
        self.assertTrue(compare_tensors("cache returns x unchanged", xr, x, verbose=verbose))
        self.assertTrue(compare_tensors("cache returns y unchanged", yr, y, verbose=verbose))

    # -----------------------------------------------------------------------
    # correct_bias
    # -----------------------------------------------------------------------

    def test_correct_bias_noop_without_bias(self, verbose=False):
        """correct_bias returns input unchanged when no bias correction is registered."""
        x = self._rand(self.B, self.C, self.H, self.W)
        out = self.pp.correct_bias(x)
        self.assertTrue(compare_tensors("correct_bias noop", out, x, verbose=verbose))

    def test_correct_bias_subtracts_registered_bias(self, verbose=False):
        """correct_bias subtracts the bias_correction buffer from the input."""
        pp = Preprocessor2D(get_default_parameters())
        bias = torch.ones(1, self.C, 1, 1)
        pp.register_buffer("bias_correction", bias, persistent=False)
        x = self._rand(self.B, self.C, self.H, self.W)
        out = pp.correct_bias(x)
        self.assertTrue(compare_tensors("correct_bias subtract", out, x - bias, verbose=verbose))

    def test_correct_bias_broadcasts_over_batch_and_spatial(self, verbose=False):
        """A (1, C, 1, 1) bias broadcasts correctly over (B, C, H, W)."""
        pp = Preprocessor2D(get_default_parameters())
        bias = torch.arange(self.C, dtype=torch.float32).reshape(1, self.C, 1, 1)
        pp.register_buffer("bias_correction", bias, persistent=False)
        x = torch.zeros(self.B, self.C, self.H, self.W)
        out = pp.correct_bias(x)
        expected = -bias.expand(self.B, self.C, self.H, self.W)
        self.assertTrue(compare_tensors("correct_bias broadcast", out, expected, verbose=verbose))

    # -----------------------------------------------------------------------
    # Internal state accessors — no noise
    # -----------------------------------------------------------------------

    def test_get_base_seed_default_no_noise(self):
        """Without noise, get_base_seed returns the supplied default value."""
        self.assertEqual(self.pp.get_base_seed(default=42), 42)

    def test_get_internal_rng_no_noise_returns_none(self):
        """Without noise, both get_internal_rng(gpu=True/False) return None."""
        self.assertIsNone(self.pp.get_internal_rng(gpu=True))
        self.assertIsNone(self.pp.get_internal_rng(gpu=False))

    def test_get_internal_state_no_noise_rng(self):
        """Without noise, get_internal_state(tensor=False) returns (None, None)."""
        state = self.pp.get_internal_state(tensor=False)
        self.assertEqual(state, (None, None))

    def test_get_internal_state_no_noise_tensor(self):
        """Without noise, get_internal_state(tensor=True) returns None."""
        self.assertIsNone(self.pp.get_internal_state(tensor=True))

    def test_set_internal_state_no_noise_noop(self):
        """Without noise, set_internal_state(None) does not raise."""
        self.pp.set_internal_state(None)

    def test_update_internal_state_no_noise_noop(self):
        """Without noise, update_internal_state() does not raise."""
        self.pp.update_internal_state()

    # -----------------------------------------------------------------------
    # Internal state accessors — with DummyNoiseS2
    # -----------------------------------------------------------------------

    def _make_pp_with_noise(self, input_noise_mode="concatenate", n_channels=1):
        params = get_default_parameters()
        params.input_noise = {"type": "dummy", "mode": input_noise_mode, "n_channels": n_channels}
        return Preprocessor2D(params)

    def test_construction_with_dummy_noise(self):
        """Preprocessor2D with dummy noise constructs without error."""
        pp = self._make_pp_with_noise()
        self.assertTrue(hasattr(pp, "input_noise"))

    def test_get_base_seed_with_noise_is_int(self):
        """With noise, get_base_seed returns an int (the noise_base_seed)."""
        pp = self._make_pp_with_noise()
        self.assertIsInstance(pp.get_base_seed(), int)

    def test_get_internal_rng_with_noise_cpu(self):
        """With noise, get_internal_rng(gpu=False) returns a CPU Generator."""
        pp = self._make_pp_with_noise()
        rng = pp.get_internal_rng(gpu=False)
        self.assertIsNotNone(rng)
        self.assertIsInstance(rng, torch.Generator)

    def test_get_internal_state_with_noise_rng_is_tuple(self):
        """With noise, get_internal_state(tensor=False) returns a 2-tuple."""
        pp = self._make_pp_with_noise()
        state = pp.get_internal_state(tensor=False)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 2)

    def test_get_internal_state_with_noise_tensor_is_tensor(self):
        """With noise, get_internal_state(tensor=True) returns a Tensor."""
        pp = self._make_pp_with_noise()
        state = pp.get_internal_state(tensor=True)
        self.assertIsInstance(state, torch.Tensor)

    def test_set_rng_with_noise_does_not_raise(self):
        """set_rng resets the noise RNG without raising."""
        pp = self._make_pp_with_noise()
        pp.set_rng(reset=True, seed=42)

    def test_set_internal_state_tensor_roundtrip(self, verbose=False):
        """Saving and restoring tensor state restores the stored noise field."""
        pp = self._make_pp_with_noise()
        pp.update_internal_state()
        tensor_state = pp.get_internal_state(tensor=True)
        # overwrite with zeros, then restore
        pp.input_noise.state.zero_()
        pp.set_internal_state(tensor_state)
        restored = pp.get_internal_state(tensor=True)
        self.assertTrue(compare_tensors("tensor state roundtrip", restored, tensor_state, verbose=verbose))

    def test_update_internal_state_with_noise_does_not_raise(self):
        """update_internal_state() with noise does not raise."""
        pp = self._make_pp_with_noise()
        pp.update_internal_state()


if __name__ == "__main__":
    unittest.main()
