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

import unittest

import torch

from makani.models.preprocessor_helpers import get_bias_correction, get_static_features

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import set_seed, get_default_parameters

# Grid dimensions used across all tests — small enough to be fast on CPU.
IMG_H = 8
IMG_W = 16


# ===========================================================================
# 1. get_bias_correction
# ===========================================================================
class TestGetBiasCorrection(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.params.img_shape_x = IMG_H
        self.params.img_shape_y = IMG_W

    def test_no_bias_correction_returns_none(self):
        """Returns None when 'bias_correction' is absent from params."""
        self.assertIsNone(get_bias_correction(self.params))

    def test_missing_file_raises_ioerror(self):
        """Raises IOError when bias_correction points to a non-existent file."""
        self.params.bias_correction = "/nonexistent/bias.npy"
        with self.assertRaises(IOError):
            get_bias_correction(self.params)


# ===========================================================================
# 2. get_static_features
# ===========================================================================
class TestGetStaticFeatures(unittest.TestCase):

    def setUp(self):
        set_seed(333)
        self.params = get_default_parameters()
        self.params.img_shape_x = IMG_H
        self.params.img_shape_y = IMG_W

    # -----------------------------------------------------------------------
    # 2a. Trivial / None case
    # -----------------------------------------------------------------------
    def test_no_features_returns_none(self):
        """Returns None when all feature flags are False (default state)."""
        self.assertIsNone(get_static_features(self.params))

    # -----------------------------------------------------------------------
    # 2b. add_grid — raw gridtype
    # -----------------------------------------------------------------------
    def test_add_grid_raw_shape(self):
        """Raw grid produces a (1, 2, H, W) tensor: one channel per spatial dim."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        out = get_static_features(self.params)
        self.assertIsNotNone(out)
        self.assertEqual(tuple(out.shape), (1, 2, IMG_H, IMG_W))

    def test_add_grid_raw_values_in_unit_interval(self):
        """Raw grid coordinates come from linspace, so all values are in [0, 1)."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        out = get_static_features(self.params)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLess(out.max().item(), 1.0)

    # -----------------------------------------------------------------------
    # 2c. add_grid — sinusoidal gridtype, channel-count arithmetic
    # -----------------------------------------------------------------------
    def test_add_grid_sinusoidal_1freq_with_cos_channels(self):
        """num_freq=1, add_cos=True: {sin,cos}(grid) × 2 spatial dims → 4 channels."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 1
        self.params.add_cos_to_grid = True
        out = get_static_features(self.params)
        self.assertEqual(tuple(out.shape), (1, 4, IMG_H, IMG_W))

    def test_add_grid_sinusoidal_1freq_no_cos_channels(self):
        """num_freq=1, add_cos=False: sin(grid) × 2 spatial dims → 2 channels."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 1
        self.params.add_cos_to_grid = False
        out = get_static_features(self.params)
        self.assertEqual(tuple(out.shape), (1, 2, IMG_H, IMG_W))

    def test_add_grid_sinusoidal_2freq_with_cos_channels(self):
        """num_freq=2, add_cos=True: {sin,cos}(k*grid) k=1,2 × 2 dims → 8 channels."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 2
        self.params.add_cos_to_grid = True
        out = get_static_features(self.params)
        self.assertEqual(tuple(out.shape), (1, 8, IMG_H, IMG_W))

    def test_add_grid_sinusoidal_2freq_no_cos_channels(self):
        """num_freq=2, add_cos=False: sin(k*grid) k=1,2 × 2 dims → 4 channels."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 2
        self.params.add_cos_to_grid = False
        out = get_static_features(self.params)
        self.assertEqual(tuple(out.shape), (1, 4, IMG_H, IMG_W))

    def test_add_grid_sinusoidal_values_bounded(self):
        """All sinusoidal-encoded values must lie in [-1, 1]."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 2
        self.params.add_cos_to_grid = True
        out = get_static_features(self.params)
        self.assertGreaterEqual(out.min().item(), -1.0)
        self.assertLessEqual(out.max().item(),  1.0)

    def test_add_grid_sinusoidal_finite(self):
        """Sinusoidal grid features must be finite (no NaN or Inf)."""
        self.params.add_grid = True
        self.params.gridtype = "sinusoidal"
        self.params.grid_num_frequencies = 1
        self.params.add_cos_to_grid = True
        out = get_static_features(self.params)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    # -----------------------------------------------------------------------
    # 2d. Spatial sharding
    # -----------------------------------------------------------------------
    def test_add_grid_spatial_sharding_height(self):
        """img_local_offset_x / img_local_shape_x slice the H dimension."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.img_local_offset_x = 2
        self.params.img_local_shape_x  = 4    # rows 2..5 → 4 rows
        out = get_static_features(self.params)
        self.assertEqual(out.shape[-2], 4)
        self.assertEqual(out.shape[-1], IMG_W)

    def test_add_grid_spatial_sharding_width(self):
        """img_local_offset_y / img_local_shape_y slice the W dimension."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.img_local_offset_y = 4
        self.params.img_local_shape_y  = 8    # cols 4..11 → 8 cols
        out = get_static_features(self.params)
        self.assertEqual(out.shape[-2], IMG_H)
        self.assertEqual(out.shape[-1], 8)

    def test_add_grid_sharding_clamped_to_grid_boundary(self):
        """offset + local_shape > grid_size is clamped: only the remaining rows."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.img_local_offset_x = 6
        self.params.img_local_shape_x  = 8    # would overshoot → clamped to 2 rows
        out = get_static_features(self.params)
        self.assertEqual(out.shape[-2], IMG_H - 6)

    # -----------------------------------------------------------------------
    # 2e. Subsampling
    # -----------------------------------------------------------------------
    def test_add_grid_subsampling_halves_spatial_dims(self):
        """subsampling_factor=2 halves both spatial dimensions."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.subsampling_factor = 2
        out = get_static_features(self.params)
        self.assertEqual(out.shape[-2], IMG_H // 2)
        self.assertEqual(out.shape[-1], IMG_W // 2)

    def test_add_grid_subsampling_factor_1_unchanged(self):
        """subsampling_factor=1 leaves both spatial dimensions unchanged."""
        self.params.add_grid = True
        self.params.gridtype = "raw"
        self.params.subsampling_factor = 1
        out = get_static_features(self.params)
        self.assertEqual(out.shape[-2], IMG_H)
        self.assertEqual(out.shape[-1], IMG_W)

    # -----------------------------------------------------------------------
    # 2f. IOError guards for file-dependent features
    # -----------------------------------------------------------------------
    def test_add_orography_missing_file_raises_ioerror(self):
        """Raises IOError when orography_path does not exist."""
        self.params.add_orography = True
        self.params.orography_path = "/nonexistent/orography.npy"
        with self.assertRaises(IOError):
            get_static_features(self.params)

    def test_add_landmask_missing_file_raises_ioerror(self):
        """Raises IOError when landmask_path does not exist."""
        self.params.add_landmask = True
        self.params.landmask_path = "/nonexistent/landmask.npy"
        with self.assertRaises(IOError):
            get_static_features(self.params)

    def test_add_soiltype_missing_file_raises_ioerror(self):
        """Raises IOError when soiltype_path does not exist."""
        self.params.add_soiltype = True
        self.params.soiltype_path = "/nonexistent/soiltype.npy"
        with self.assertRaises(IOError):
            get_static_features(self.params)

    def test_add_copernicus_emb_missing_file_raises_ioerror(self):
        """Raises IOError when copernicus_emb_path does not exist."""
        self.params.add_copernicus_emb = True
        self.params.copernicus_emb_path = "/nonexistent/copernicus.npy"
        with self.assertRaises(IOError):
            get_static_features(self.params)


if __name__ == "__main__":
    unittest.main()
