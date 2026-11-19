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

import numpy as np
import torch
from parameterized import parameterized

import torch_harmonics as th

from makani.data.grid_types import canonical_grid_type, expected_latitudes, matching_grid_types, verify_grid_type
from makani.utils.grids import GridConverter, GridQuadrature, PlanarGrid, make_grid, regular_grid_types, sht_grid
from makani.models.common.bandlimit import BandLimitMask

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

# Small grid that runs fast on CPU
_H, _W = 64, 128


# ---------------------------------------------------------------------------
# make_grid
# ---------------------------------------------------------------------------

#: every grid type makani builds a grid for
_GRID_TYPES = ["equiangular", "legendre-gauss", "lobatto", "fejer1", "trapezoidal", "euclidean"]


class TestMakeGrid(unittest.TestCase):
    """The grid types are the regular grids of torch-harmonics, and the plane."""

    def test_the_regular_grids_of_torch_harmonics(self):
        self.assertEqual(
            set(regular_grid_types()), {"equiangular", "legendre-gauss", "lobatto", "fejer1", "trapezoidal"}
        )

    @parameterized.expand([(t,) for t in _GRID_TYPES if t != "euclidean"])
    def test_a_regular_grid_on_the_sphere(self, grid_type):
        grid = make_grid((_H, _W), grid_type)
        self.assertIsInstance(grid, th.RegularGridS2)
        self.assertEqual(grid.grid_type, grid_type)
        self.assertEqual(grid.shape, (_H, _W))
        sgrid = sht_grid((_H, _W), grid_type)
        self.assertEqual((type(sgrid), sgrid.shape), (type(grid), grid.shape))

    def test_the_plane(self):
        grid = make_grid((_H, _W), "euclidean")
        self.assertEqual(grid, PlanarGrid(_H, _W))
        self.assertEqual(grid.shape, (_H, _W))
        with self.assertRaisesRegex(ValueError, "no spherical harmonics"):
            sht_grid((_H, _W), "euclidean")

    @parameterized.expand([("healpix",), ("clenshaw-curtiss",), ("weatherbench2",), ("cubed-sphere",)])
    def test_other_grid_types_are_refused(self, grid_type):
        # healpix is a grid of torch-harmonics, but not a regular one; the others name no grid
        with self.assertRaisesRegex(ValueError, "not supported"):
            make_grid((_H, _W), grid_type)


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
        self.assertTrue((dst_lat > self.lat_rad[-1]).all(), "dst_lat has values at or below the south pole")
        self.assertTrue((dst_lat < self.lat_rad[0]).all(), "dst_lat has values at or above the north pole")

    def test_equiangular_to_legendre_gauss_linear_function_exact(self, verbose=False):
        """Linear interpolation recovers a latitude-linear field exactly."""
        converter = GridConverter("equiangular", "legendre-gauss", self.lat_rad, self.lon_rad)

        # Field whose value at row i is lat_rad[i], constant across lon
        data = self.lat_rad.unsqueeze(-1).expand(self.H, self.W)
        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        out = converter(data)  # (1, 1, H_dst, W)

        dst_lat, _ = converter.get_dst_coords()
        expected = dst_lat.unsqueeze(-1).expand(-1, self.W).unsqueeze(0).unsqueeze(0).to(out.dtype)

        self.assertTrue(
            compare_tensors("linear function interpolation", out, expected, atol=1e-5, rtol=1e-5, verbose=verbose)
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

    @parameterized.expand([(t,) for t in _GRID_TYPES])
    def test_weights_sum_to_4pi(self, grid_type):
        """Quadrature weights must integrate the constant function 1 to 4π (unit-sphere area)."""
        gq = GridQuadrature(make_grid((_H, _W), grid_type))
        total = gq.quad_weight.sum().item()
        self.assertAlmostEqual(
            total,
            4.0 * math.pi,
            delta=1e-4,
            msg=f"{grid_type}: weights sum to {total:.6f}, expected 4π≈{4*math.pi:.6f}",
        )

    @parameterized.expand([(t,) for t in _GRID_TYPES])
    def test_weights_sum_to_one_when_normalized(self, grid_type):
        """With normalize=True the weights must sum to 1."""
        gq = GridQuadrature(make_grid((_H, _W), grid_type), normalize=True)
        total = gq.quad_weight.sum().item()
        self.assertAlmostEqual(
            total,
            1.0,
            delta=1e-5,
            msg=f"{grid_type} normalized: weights sum to {total:.8f}, expected 1.0",
        )

    @parameterized.expand([(t,) for t in _GRID_TYPES])
    def test_weights_all_non_negative(self, grid_type):
        """All quadrature weights must be non-negative."""
        gq = GridQuadrature(make_grid((_H, _W), grid_type))
        self.assertTrue(
            (gq.quad_weight >= 0).all(),
            msg=f"{grid_type}: negative weights found",
        )

    # --- forward pass --------------------------------------------------------

    @parameterized.expand([(t,) for t in _GRID_TYPES])
    def test_forward_constant_one_integrates_to_4pi(self, grid_type, verbose=False):
        """Integrating f≡1 yields 4π on every grid."""
        gq = GridQuadrature(make_grid((_H, _W), grid_type))
        B, C = 2, 3
        x = torch.ones(B, C, _H, _W)
        out = gq(x)
        expected = torch.full((B, C), 4.0 * math.pi)
        self.assertTrue(
            compare_tensors(
                f"constant integral ({grid_type})",
                out,
                expected,
                atol=1e-3,
                rtol=1e-4,
                verbose=verbose,
            )
        )

    def test_forward_output_shape(self):
        """forward() reduces the spatial (H, W) dims and returns (B, C)."""
        gq = GridQuadrature(make_grid((_H, _W), "legendre-gauss"))
        x = torch.randn(4, 5, _H, _W)
        out = gq(x)
        self.assertEqual(out.shape, (4, 5))

    def test_forward_linearity(self, verbose=False):
        """forward() must be linear: Q(a*x + b*y) == a*Q(x) + b*Q(y)."""
        gq = GridQuadrature(make_grid((_H, _W), "legendre-gauss"))
        set_seed(333)
        x = torch.randn(2, 3, _H, _W)
        y = torch.randn(2, 3, _H, _W)
        a, b = 2.5, -1.3
        lhs = gq(a * x + b * y)
        rhs = a * gq(x) + b * gq(y)
        self.assertTrue(compare_tensors("linearity", lhs, rhs, atol=1e-4, rtol=1e-5, verbose=verbose))

    # --- construction edge cases ---------------------------------------------

    @parameterized.expand([(t,) for t in _GRID_TYPES])
    def test_the_weights_are_those_of_the_grid(self, grid_type):
        grid = make_grid((_H, _W), grid_type)
        gq = GridQuadrature(grid)
        self.assertTrue(torch.equal(gq.quad_weight.reshape(-1), grid.quad_weights.reshape(-1).to(gq.quad_weight.dtype)))

    @parameterized.expand([("equiangular",), ("legendre-gauss",)])
    def test_smooth_fields_integrate_accurately(self, grid_type):
        # the mean of cos^2 of the latitude over the sphere is 2/3; Clenshaw-Curtis and Gauss-Legendre get it to round-off
        grid = make_grid((_H, _W), grid_type)
        gq = GridQuadrature(grid, normalize=True)
        lat = torch.pi / 2 - torch.as_tensor(grid.colats, dtype=torch.float64)
        field = torch.cos(lat).square().reshape(1, 1, -1, 1).expand(1, 1, _H, _W)
        self.assertAlmostEqual(gq(field).item(), 2.0 / 3.0, places=10)

    def test_crop_reduces_total_weight(self):
        """A spatially cropped quadrature has strictly less total weight than the full sphere."""
        gq_full = GridQuadrature(make_grid((_H, _W), "legendre-gauss"))
        gq_crop = GridQuadrature(
            make_grid((_H, _W), "legendre-gauss"),
            crop_shape=(_H // 2, _W),
            crop_offset=(0, 0),
        )
        total_full = gq_full.quad_weight.sum().item()
        total_crop = gq_crop.quad_weight.sum().item()
        self.assertGreater(total_crop, 0.0)
        self.assertLess(total_crop, total_full)

    def test_crop_shape_equals_img_shape_matches_full(self, verbose=False):
        """crop_shape == img_shape (with zero offset) is equivalent to no crop."""
        gq_full = GridQuadrature(make_grid((_H, _W), "equiangular"))
        gq_crop = GridQuadrature(
            make_grid((_H, _W), "equiangular"),
            crop_shape=(_H, _W),
            crop_offset=(0, 0),
        )
        self.assertTrue(
            compare_tensors(
                "full crop vs no crop", gq_full.quad_weight, gq_crop.quad_weight, atol=0.0, rtol=0.0, verbose=verbose
            )
        )


class TestGridTypeVerification(unittest.TestCase):
    """Checking that a declared grid type matches the coordinates it came with.

    The declaration decides the quadrature. Weights that do not belong to the
    grid mis-weight every area averaged loss and metric, by an amount that looks
    like a modelling difference rather than a bug -- and since the coordinates
    are always available, the claim never has to be taken on trust.
    """

    def test_equiangular_latitudes_span_pole_to_pole(self):
        latitudes = expected_latitudes("equiangular", 5)
        self.assertTrue(np.allclose(latitudes, [-90.0, -45.0, 0.0, 45.0, 90.0]))

    def test_legacy_grid_types_are_read_as_equiangular(self):
        # datasets written before makani 0.4 may declare a quadrature rule on equally spaced latitudes
        for grid_type in ("clenshaw-curtiss", "weatherbench2"):
            with self.subTest(grid_type=grid_type), self.assertLogs("makani.data.grid_types", level="WARNING"):
                self.assertEqual(canonical_grid_type(grid_type, source="data.json"), "equiangular")
        self.assertEqual(canonical_grid_type("legendre-gauss"), "legendre-gauss")

    def test_gauss_nodes_are_not_the_equiangular_ones(self):
        # the premise of the whole check: the grids are distinguishable
        gauss = expected_latitudes("legendre-gauss", 32)
        equiangular = expected_latitudes("equiangular", 32)

        self.assertGreater(np.max(np.abs(gauss - equiangular)), 1.0)

    def test_gauss_nodes_stay_inside_the_poles(self):
        gauss = expected_latitudes("legendre-gauss", 16)

        self.assertGreater(gauss[0], -90.0)
        self.assertLess(gauss[-1], 90.0)

    def test_a_grid_off_the_sphere_has_no_latitudes(self):
        self.assertIsNone(expected_latitudes("euclidean", 8))

    def test_an_unknown_grid_type_is_refused(self):
        with self.assertRaises(NotImplementedError):
            expected_latitudes("cubed-sphere", 8)

    @parameterized.expand([(t,) for t in _GRID_TYPES if t != "euclidean"])
    def test_each_grid_matches_only_its_own_nodes(self, grid_type):
        self.assertEqual(matching_grid_types(expected_latitudes(grid_type, 24)), [grid_type])

    def test_gauss_nodes_match_only_the_gauss_grid(self):
        # the one separation coordinates can make, and the one worth making
        self.assertEqual(matching_grid_types(expected_latitudes("legendre-gauss", 24)), ["legendre-gauss"])

    def test_latitudes_that_are_no_known_grid_match_nothing(self):
        self.assertEqual(matching_grid_types(np.linspace(-30.0, 30.0, 12)), [])

    def test_a_truthful_declaration_passes(self):
        for grid_type in regular_grid_types():
            with self.subTest(grid_type=grid_type):
                verify_grid_type(grid_type, expected_latitudes(grid_type, 24))

    def test_orientation_is_not_part_of_the_claim(self):
        # makani stores latitudes north to south, the quadrature routines
        # produce them south to north; what is compared is where the nodes are
        verify_grid_type("legendre-gauss", np.flip(expected_latitudes("legendre-gauss", 24)))

    def test_a_false_declaration_is_refused_and_says_what_it_looks_like(self):
        with self.assertRaises(ValueError) as ctx:
            verify_grid_type("legendre-gauss", expected_latitudes("equiangular", 24), source="data.json")

        message = str(ctx.exception)
        self.assertIn("data.json", message)
        self.assertIn("legendre-gauss", message)
        self.assertIn("equiangular", message)

    def test_float32_coordinates_still_verify(self):
        # coordinates come out of files as float32, so the tolerance has to
        # absorb that while still separating the grids
        latitudes = expected_latitudes("equiangular", 721).astype(np.float32).astype(np.float64)
        verify_grid_type("equiangular", latitudes)

    def test_a_grid_off_the_sphere_is_not_checked(self):
        verify_grid_type("euclidean", np.linspace(0.0, 1.0, 8))


# ---------------------------------------------------------------------------
# BandLimitMask
# ---------------------------------------------------------------------------


class TestBandLimitMask(unittest.TestCase):
    """BandLimitMask keeps the modes below the band limit and removes the ones above."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_sht_removes_modes_above_lmax(self, verbose=False):
        import torch_harmonics as th

        nlat, nlon, lmax = 16, 32, 6
        isht = th.InverseRealSHT(sht_grid((nlat, nlon), "legendre-gauss"))

        # random real field with modes up to the grid's full band limit
        coeffs = torch.randn(2, 3, isht.lmax, isht.mmax, dtype=torch.complex64)
        ls = torch.arange(isht.lmax).reshape(-1, 1)
        ms = torch.arange(isht.mmax).reshape(1, -1)
        coeffs = torch.where(ms <= ls, coeffs, torch.zeros_like(coeffs))
        coeffs[..., 0] = coeffs[..., 0].real.to(coeffs.dtype)
        low = torch.where(ls < lmax, coeffs, torch.zeros_like(coeffs))

        field = isht(coeffs)
        expected = isht(low)

        mask = BandLimitMask((nlat, nlon), "legendre-gauss", lmax=lmax, type="sht")
        self.assertTrue(compare_tensors("band limited", mask(field), expected, atol=1e-5, rtol=1e-5, verbose=verbose))
        # a field that is already band limited passes unchanged
        self.assertTrue(compare_tensors("unchanged", mask(expected), expected, atol=1e-5, rtol=1e-5, verbose=verbose))

    def test_fft_cutoff_shrinks_towards_the_poles(self, verbose=False):
        from torch_harmonics.quadrature import precompute_latitudes

        nlat, nlon, lmax, m = 16, 32, 8, 3
        lon = torch.arange(nlon, dtype=torch.float32) * 2 * math.pi / nlon
        wave = torch.cos(m * lon).reshape(1, 1, 1, nlon).expand(1, 1, nlat, nlon)
        constant = torch.ones(1, 1, nlat, nlon)

        mask = BandLimitMask((nlat, nlon), "equiangular", lmax=lmax, type="fft")

        # the zonal mean is kept on every latitude
        self.assertTrue(compare_tensors("m=0", mask(constant), constant, atol=1e-5, verbose=verbose))

        # wavenumber m is kept where the latitude circle resolves it at the equator's grid spacing
        colats, _ = precompute_latitudes(nlat, grid="equiangular")
        keep = m <= torch.ceil((lmax - 1) * torch.sin(colats.float()))
        self.assertTrue(keep.any() and not keep.all(), "the test should cover kept and removed latitudes")
        expected = torch.where(keep.reshape(1, 1, nlat, 1), wave, torch.zeros_like(wave))
        self.assertTrue(compare_tensors(f"m={m}", mask(wave), expected, atol=1e-5, verbose=verbose))

    @parameterized.expand([("sht", "legendre-gauss"), ("sht", "equiangular"), ("fft", "equiangular")])
    def test_default_lmax_is_the_grid_bandlimit(self, mask_type, grid_type, verbose=False):
        # without lmax, the transforms and the mask have to use the grid's band limit
        from makani.utils.grids import compute_spherical_bandlimit

        img_shape = (17, 32) if grid_type == "equiangular" else (16, 32)
        field = torch.randn(2, 3, *img_shape)

        default = BandLimitMask(img_shape, grid_type, type=mask_type)
        explicit = BandLimitMask(
            img_shape, grid_type, lmax=compute_spherical_bandlimit(img_shape, grid_type), type=mask_type
        )
        self.assertEqual(default.lmax, explicit.lmax)
        self.assertTrue(compare_tensors("default lmax", default(field), explicit(field), verbose=verbose))

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            BandLimitMask((16, 32), "equiangular", lmax=8, type="wavelet")


if __name__ == "__main__":
    unittest.main()
