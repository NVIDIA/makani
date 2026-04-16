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

import math
import os
import sys
import unittest

import torch
from parameterized import parameterized

from makani.utils.grids import grid_to_quadrature_rule, GridConverter, GridQuadrature

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

# Small grid that runs fast on CPU
_H, _W = 64, 128


# ---------------------------------------------------------------------------
# grid_to_quadrature_rule
# ---------------------------------------------------------------------------

class TestGridToQuadratureRule(unittest.TestCase):

    @parameterized.expand([
        ("euclidean",        "uniform"),
        ("equiangular",      "naive"),
        ("legendre-gauss",   "legendre-gauss"),
        ("clenshaw-curtiss", "clenshaw-curtiss"),
        ("weatherbench2",    "weatherbench2"),
    ])
    def test_known_grid_types(self, grid_type, expected_rule):
        self.assertEqual(grid_to_quadrature_rule(grid_type), expected_rule)

    def test_unknown_grid_type_raises(self):
        with self.assertRaises(NotImplementedError):
            grid_to_quadrature_rule("unknown_grid")


# ---------------------------------------------------------------------------
# GridConverter
# ---------------------------------------------------------------------------

class TestGridConverter(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)
        self.H, self.W = _H, _W
        # equiangular source grid: north (+π/2) to south (-π/2), longitudes [0, 2π)
        self.lat_rad = torch.linspace(math.pi / 2, -math.pi / 2, self.H)
        self.lon_rad = torch.arange(self.W) * (2.0 * math.pi / self.W)

    def test_passthrough_same_grid_returns_input(self):
        """forward() is the identity (same object) when src == dst."""
        converter = GridConverter("equiangular", "equiangular", self.lat_rad, self.lon_rad)
        x = torch.randn(2, 3, self.H, self.W)
        out = converter(x)
        self.assertIs(out, x)

    def test_passthrough_coords_unchanged(self):
        """src == dst: dst coords are equal to src coords."""
        converter = GridConverter("equiangular", "equiangular", self.lat_rad, self.lon_rad)
        src_lat, src_lon = converter.get_src_coords()
        dst_lat, dst_lon = converter.get_dst_coords()
        self.assertTrue(torch.equal(src_lat, dst_lat))
        self.assertTrue(torch.equal(src_lon, dst_lon))

    def test_equiangular_to_legendre_gauss_output_shape(self):
        """equiangular → legendre-gauss output has the same spatial shape as input."""
        converter = GridConverter("equiangular", "legendre-gauss", self.lat_rad, self.lon_rad)
        x = torch.randn(2, 3, self.H, self.W)
        out = converter(x)
        self.assertEqual(out.shape, (2, 3, self.H, self.W))

    def test_equiangular_to_legendre_gauss_dst_lats_within_src_range(self):
        """Legendre-Gauss destination nodes must lie strictly inside the source lat range."""
        converter = GridConverter("equiangular", "legendre-gauss", self.lat_rad, self.lon_rad)
        dst_lat, _ = converter.get_dst_coords()
        self.assertTrue((dst_lat > self.lat_rad[-1]).all(),
                        "dst_lat has values at or below the south pole")
        self.assertTrue((dst_lat < self.lat_rad[0]).all(),
                        "dst_lat has values at or above the north pole")

    def test_equiangular_to_legendre_gauss_linear_function_exact(self, verbose=False):
        """Linear interpolation recovers a latitude-linear field exactly."""
        converter = GridConverter("equiangular", "legendre-gauss", self.lat_rad, self.lon_rad)

        # Field whose value at row i is lat_rad[i], constant across lon
        data = self.lat_rad.unsqueeze(-1).expand(self.H, self.W)
        data = data.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        out = converter(data)                   # (1, 1, H_dst, W)

        dst_lat, _ = converter.get_dst_coords()
        expected = dst_lat.unsqueeze(-1).expand(-1, self.W).unsqueeze(0).unsqueeze(0).to(out.dtype)

        self.assertTrue(
            compare_tensors("linear function interpolation", out, expected,
                            atol=1e-5, rtol=1e-5, verbose=verbose)
        )

    def test_unsupported_dst_raises(self):
        """An unsupported destination grid raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            GridConverter("equiangular", "unsupported_dst", self.lat_rad, self.lon_rad)


# ---------------------------------------------------------------------------
# GridQuadrature
# ---------------------------------------------------------------------------

class TestGridQuadrature(unittest.TestCase):

    def setUp(self):
        disable_tf32()

    # --- weight correctness --------------------------------------------------

    @parameterized.expand([
        ("naive",),
        ("clenshaw-curtiss",),
        ("legendre-gauss",),
        ("weatherbench2",),
        ("uniform",),
    ])
    def test_weights_sum_to_4pi(self, quadrature_rule):
        """Quadrature weights must integrate the constant function 1 to 4π (unit-sphere area)."""
        gq = GridQuadrature(quadrature_rule, img_shape=(_H, _W))
        total = gq.quad_weight.sum().item()
        self.assertAlmostEqual(
            total, 4.0 * math.pi, delta=1e-4,
            msg=f"{quadrature_rule}: weights sum to {total:.6f}, expected 4π≈{4*math.pi:.6f}",
        )

    @parameterized.expand([
        ("naive",),
        ("clenshaw-curtiss",),
        ("legendre-gauss",),
        ("weatherbench2",),
        ("uniform",),
    ])
    def test_weights_sum_to_one_when_normalized(self, quadrature_rule):
        """With normalize=True the weights must sum to 1."""
        gq = GridQuadrature(quadrature_rule, img_shape=(_H, _W), normalize=True)
        total = gq.quad_weight.sum().item()
        self.assertAlmostEqual(
            total, 1.0, delta=1e-5,
            msg=f"{quadrature_rule} normalized: weights sum to {total:.8f}, expected 1.0",
        )

    @parameterized.expand([
        ("naive",),
        ("clenshaw-curtiss",),
        ("legendre-gauss",),
        ("weatherbench2",),
        ("uniform",),
    ])
    def test_weights_all_non_negative(self, quadrature_rule):
        """All quadrature weights must be non-negative."""
        gq = GridQuadrature(quadrature_rule, img_shape=(_H, _W))
        self.assertTrue(
            (gq.quad_weight >= 0).all(),
            msg=f"{quadrature_rule}: negative weights found",
        )

    # --- forward pass --------------------------------------------------------

    @parameterized.expand([
        ("naive",),
        ("clenshaw-curtiss",),
        ("legendre-gauss",),
        ("weatherbench2",),
        ("uniform",),
    ])
    def test_forward_constant_one_integrates_to_4pi(self, quadrature_rule, verbose=False):
        """Integrating f≡1 over the sphere yields 4π for every quadrature rule."""
        gq = GridQuadrature(quadrature_rule, img_shape=(_H, _W))
        B, C = 2, 3
        x = torch.ones(B, C, _H, _W)
        out = gq(x)
        expected = torch.full((B, C), 4.0 * math.pi)
        self.assertTrue(
            compare_tensors(
                f"constant integral ({quadrature_rule})",
                out, expected, atol=1e-3, rtol=1e-4, verbose=verbose,
            )
        )

    def test_forward_output_shape(self):
        """forward() reduces the spatial (H, W) dims and returns (B, C)."""
        gq = GridQuadrature("legendre-gauss", img_shape=(_H, _W))
        x = torch.randn(4, 5, _H, _W)
        out = gq(x)
        self.assertEqual(out.shape, (4, 5))

    def test_forward_linearity(self, verbose=False):
        """forward() must be linear: Q(a*x + b*y) == a*Q(x) + b*Q(y)."""
        gq = GridQuadrature("legendre-gauss", img_shape=(_H, _W))
        set_seed(333)
        x = torch.randn(2, 3, _H, _W)
        y = torch.randn(2, 3, _H, _W)
        a, b = 2.5, -1.3
        lhs = gq(a * x + b * y)
        rhs = a * gq(x) + b * gq(y)
        self.assertTrue(
            compare_tensors("linearity", lhs, rhs, atol=1e-4, rtol=1e-5, verbose=verbose)
        )

    # --- construction edge cases ---------------------------------------------

    def test_unknown_rule_raises(self):
        """An unknown quadrature rule raises ValueError."""
        with self.assertRaises(ValueError):
            GridQuadrature("unknown_rule", img_shape=(_H, _W))

    def test_crop_reduces_total_weight(self):
        """A spatially cropped quadrature has strictly less total weight than the full sphere."""
        gq_full = GridQuadrature("legendre-gauss", img_shape=(_H, _W))
        gq_crop = GridQuadrature(
            "legendre-gauss", img_shape=(_H, _W),
            crop_shape=(_H // 2, _W), crop_offset=(0, 0),
        )
        total_full = gq_full.quad_weight.sum().item()
        total_crop = gq_crop.quad_weight.sum().item()
        self.assertGreater(total_crop, 0.0)
        self.assertLess(total_crop, total_full)

    def test_crop_shape_equals_img_shape_matches_full(self, verbose=False):
        """crop_shape == img_shape (with zero offset) is equivalent to no crop."""
        gq_full = GridQuadrature("clenshaw-curtiss", img_shape=(_H, _W))
        gq_crop = GridQuadrature(
            "clenshaw-curtiss", img_shape=(_H, _W),
            crop_shape=(_H, _W), crop_offset=(0, 0),
        )
        self.assertTrue(
            compare_tensors("full crop vs no crop", gq_full.quad_weight, gq_crop.quad_weight,
                            atol=0.0, rtol=0.0, verbose=verbose)
        )


if __name__ == "__main__":
    unittest.main()
