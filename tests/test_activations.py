# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest
from parameterized import parameterized

import torch
import torch.nn as nn

from makani.models.common.activations import ComplexReLU, ComplexActivation, MagnitudePreservingSiLU

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

# ---------------------------------------------------------------------------
# Shared input shape
# ---------------------------------------------------------------------------
SHAPE = (2, 4, 6, 5)   # (batch, channels, H, W) — all complex


def _cx_randn(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _cx(real, imag):
    """Build a complex64 tensor from explicit real/imag float tensors."""
    return torch.complex(real.float(), imag.float())


# All four supported modes
_ALL_MODES = [("cartesian",), ("modulus",), ("halfplane",), ("real",)]


# ===========================================================================
# TestComplexReLU
# ===========================================================================
class TestComplexReLU(unittest.TestCase):
    """Tests for ComplexReLU covering all four modes:
    'real', 'cartesian', 'modulus', 'halfplane'."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    # --- helpers ---

    def _make_modulus_fn(self, bias_val=0.0):
        fn = ComplexReLU(mode="modulus", bias_shape=(1,), scale=bias_val)
        with torch.no_grad():
            fn.bias.fill_(bias_val)
        return fn

    def _make_halfplane_fn(self, bias_angle=0.0, negative_slope=0.0):
        fn = ComplexReLU(mode="halfplane", negative_slope=negative_slope, bias_shape=(1,))
        with torch.no_grad():
            fn.bias.fill_(bias_angle)
        return fn

    def _z_at_angle(self, angle_rad, magnitude=1.0):
        r = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)], dtype=torch.float32) * magnitude
        return torch.complex(r[0:1], r[1:2])

    # --- shape / dtype ---

    @parameterized.expand(_ALL_MODES)
    def test_shape_dtype_preserved(self, mode):
        fn  = ComplexReLU(mode=mode)
        z   = _cx_randn(*SHAPE)
        out = fn(z)
        self.assertEqual(out.shape, z.shape, f"{mode}: shape changed")
        self.assertEqual(out.dtype, z.dtype, f"{mode}: dtype changed")

    # --- "real" mode ---

    def test_real_real_part_clipped(self, verbose=False):
        fn   = ComplexReLU(mode="real", negative_slope=0.0)
        real = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        imag = torch.tensor([ 3.0,  4.0, 5.0, 6.0, 7.0])
        out  = fn(_cx(real, imag))
        self.assertTrue(compare_tensors("real real part", out.real, real.clamp(min=0.0), atol=1e-6, verbose=verbose))

    def test_real_imaginary_part_unchanged(self, verbose=False):
        """Imaginary part must pass through unmodified regardless of sign."""
        fn   = ComplexReLU(mode="real", negative_slope=0.0)
        real = torch.tensor([-2.0, -1.0, 0.0,  1.0, 2.0])
        imag = torch.tensor([-3.0,  4.0, -5.0, 6.0, -7.0])
        out  = fn(_cx(real, imag))
        self.assertTrue(compare_tensors("real imag part", out.imag, imag, atol=1e-6, verbose=verbose))

    # --- "cartesian" mode ---

    def test_cartesian_real_part_clipped(self, verbose=False):
        fn   = ComplexReLU(mode="cartesian", negative_slope=0.0)
        real = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        imag = torch.tensor([ 3.0,  4.0, 5.0, 6.0, 7.0])
        out  = fn(_cx(real, imag))
        self.assertTrue(compare_tensors("cartesian real part", out.real, real.clamp(min=0.0), atol=1e-6, verbose=verbose))

    def test_cartesian_imaginary_part_clipped(self, verbose=False):
        """Unlike 'real' mode, negative imaginary parts are zeroed."""
        fn   = ComplexReLU(mode="cartesian", negative_slope=0.0)
        real = torch.tensor([1.0,  1.0,  1.0, 1.0, 1.0])
        imag = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        out  = fn(_cx(real, imag))
        self.assertTrue(compare_tensors("cartesian imag part", out.imag, imag.clamp(min=0.0), atol=1e-6, verbose=verbose))

    def test_cartesian_differs_from_real_on_negative_imag(self):
        """'cartesian' zeros negative imaginary parts; 'real' preserves them."""
        fn_cx   = ComplexReLU(mode="cartesian", negative_slope=0.0)
        fn_real = ComplexReLU(mode="real",      negative_slope=0.0)
        z = _cx(torch.tensor([1.0]), torch.tensor([-1.0]))
        self.assertFalse(
            torch.allclose(fn_cx(z).imag, fn_real(z).imag),
            "cartesian and real modes should differ on negative imaginary input",
        )

    # --- "modulus" mode ---

    def test_modulus_phase_preserved_when_nonzero(self, verbose=False):
        """When |z| + bias > 0 the output phase must equal the input phase."""
        fn  = self._make_modulus_fn(bias_val=0.0)
        z   = _cx_randn(10) * 5.0 + (2.0 + 2.0j)
        out = fn(z)
        nonzero = out.abs() > 1e-6
        self.assertTrue(
            compare_tensors("modulus phase", torch.angle(out)[nonzero], torch.angle(z)[nonzero], atol=1e-4, verbose=verbose),
        )

    def test_modulus_output_magnitude(self, verbose=False):
        """Output modulus equals max(|z| + bias, 0)."""
        fn  = self._make_modulus_fn(bias_val=1.0)
        z   = _cx_randn(20) * 3.0
        out = fn(z)
        self.assertTrue(
            compare_tensors("modulus abs", out.abs(), (z.abs() + 1.0).clamp(min=0.0), atol=1e-5, verbose=verbose),
        )

    def test_modulus_zero_when_nonpositive(self, verbose=False):
        """When |z| + bias ≤ 0 the output must be zero."""
        fn  = self._make_modulus_fn(bias_val=-10.0)
        out = fn(_cx_randn(20))
        self.assertTrue(compare_tensors("modulus zero", out.abs(), torch.zeros_like(out.abs()), atol=1e-6, verbose=verbose))

    # --- "halfplane" mode ---

    def test_halfplane_passthrough_in_window(self, verbose=False):
        """Inputs whose angle falls in [bias, bias + π/2) pass through unchanged."""
        fn  = self._make_halfplane_fn(bias_angle=0.0)
        z   = self._z_at_angle(math.pi / 4)   # well inside [0, π/2)
        self.assertTrue(compare_tensors("halfplane passthrough", fn(z), z, atol=1e-6, verbose=verbose))

    def test_halfplane_scaled_outside_window(self, verbose=False):
        """Inputs outside the window are multiplied by negative_slope."""
        slope = 0.1
        fn  = self._make_halfplane_fn(bias_angle=0.0, negative_slope=slope)
        z   = self._z_at_angle(math.pi)        # well outside [0, π/2)
        self.assertTrue(compare_tensors("halfplane scaled", fn(z), slope * z, atol=1e-6, verbose=verbose))

    def test_halfplane_zero_slope_kills_outside_window(self, verbose=False):
        """With negative_slope=0, inputs outside the window collapse to zero."""
        fn  = self._make_halfplane_fn(bias_angle=0.0, negative_slope=0.0)
        z   = self._z_at_angle(math.pi)
        self.assertTrue(compare_tensors("halfplane zero", fn(z), torch.zeros_like(z), atol=1e-6, verbose=verbose))

    def test_halfplane_bias_shifts_window(self, verbose=False):
        """Shifting bias by π/2 puts angle=π/4 outside the window."""
        fn  = self._make_halfplane_fn(bias_angle=math.pi / 2, negative_slope=0.0)
        z   = self._z_at_angle(math.pi / 4)    # now below bias → outside
        self.assertTrue(compare_tensors("halfplane bias shift", fn(z), torch.zeros_like(z), atol=1e-6, verbose=verbose))

    # --- negative_slope on "real" / "cartesian" ---

    @parameterized.expand([("real",), ("cartesian",)])
    def test_negative_slope_scales_negative_real(self, mode, verbose=False):
        slope = 0.2
        fn  = ComplexReLU(mode=mode, negative_slope=slope)
        out = fn(_cx(torch.tensor([-3.0]), torch.tensor([0.0])))
        self.assertTrue(
            compare_tensors(f"{mode} leaky slope", out.real, torch.tensor([slope * -3.0]), atol=1e-6, verbose=verbose),
        )

    @parameterized.expand([("real",), ("cartesian",)])
    def test_negative_slope_leaves_positive_real_unchanged(self, mode, verbose=False):
        fn  = ComplexReLU(mode=mode, negative_slope=0.2)
        out = fn(_cx(torch.tensor([3.0]), torch.tensor([0.0])))
        self.assertTrue(
            compare_tensors(f"{mode} positive unaffected", out.real, torch.tensor([3.0]), atol=1e-6, verbose=verbose),
        )

    # --- unknown mode ---

    def test_unknown_mode_raises(self):
        fn = ComplexReLU(mode="unknown_mode")
        with self.assertRaises(NotImplementedError):
            fn(_cx_randn(4))

    # --- backward ---

    @parameterized.expand(_ALL_MODES)
    def test_backward(self, mode):
        fn = ComplexReLU(mode=mode)
        z  = _cx_randn(*SHAPE) * 2.0 + (1.0 + 1.0j)
        zr = torch.view_as_real(z).requires_grad_(True)
        torch.view_as_real(fn(torch.view_as_complex(zr))).sum().backward()
        self.assertIsNotNone(zr.grad,                f"{mode}: grad is None")
        self.assertFalse(torch.isnan(zr.grad).any(), f"{mode}: NaN in grad")
        self.assertFalse(torch.isinf(zr.grad).any(), f"{mode}: Inf in grad")

    def test_modulus_backward_near_zero(self):
        """The modulus mode divides by |z|; gradients must stay finite near zero."""
        fn = ComplexReLU(mode="modulus")
        with torch.no_grad():
            fn.bias.fill_(1.0)
        zr = torch.view_as_real(_cx_randn(16) * 1e-3).requires_grad_(True)
        torch.view_as_real(fn(torch.view_as_complex(zr))).sum().backward()
        self.assertFalse(torch.isnan(zr.grad).any(), "NaN in grad near zero (modulus mode)")
        self.assertFalse(torch.isinf(zr.grad).any(), "Inf in grad near zero (modulus mode)")


# ===========================================================================
# TestComplexActivation
# ===========================================================================
class TestComplexActivation(unittest.TestCase):
    """Tests for ComplexActivation covering all three modes:
    'cartesian', 'modulus', and the identity fall-through."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    # --- shape / dtype ---

    @parameterized.expand([("cartesian",), ("modulus",), ("identity",)])
    def test_shape_dtype_preserved(self, mode):
        fn  = ComplexActivation(nn.ReLU(), mode=mode)
        z   = _cx_randn(*SHAPE)
        out = fn(z)
        self.assertEqual(out.shape, z.shape, f"{mode}: shape changed")
        self.assertEqual(out.dtype, z.dtype, f"{mode}: dtype changed")

    # --- cartesian mode ---

    def test_cartesian_real_part_activated(self, verbose=False):
        fn   = ComplexActivation(nn.ReLU(), mode="cartesian")
        real = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        imag = torch.tensor([ 3.0,  4.0, 5.0, 6.0, 7.0])
        out  = fn(_cx(real, imag))
        self.assertTrue(
            compare_tensors("cartesian real", out.real, real.clamp(min=0.0), atol=1e-6, verbose=verbose),
        )

    def test_cartesian_imaginary_part_activated(self, verbose=False):
        fn   = ComplexActivation(nn.ReLU(), mode="cartesian")
        real = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        imag = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        out  = fn(_cx(real, imag))
        self.assertTrue(
            compare_tensors("cartesian imag", out.imag, imag.clamp(min=0.0), atol=1e-6, verbose=verbose),
        )

    # --- modulus mode ---

    def test_modulus_phase_preserved(self, verbose=False):
        """When the activated modulus is non-zero, the output phase equals the input phase."""
        fn = ComplexActivation(nn.ReLU(), mode="modulus")
        with torch.no_grad():
            fn.bias.fill_(0.0)
        z   = _cx_randn(20) * 3.0 + (2.0 + 2.0j)
        out = fn(z)
        nonzero = out.abs() > 1e-6
        self.assertTrue(
            compare_tensors("modulus phase", torch.angle(out)[nonzero], torch.angle(z)[nonzero], atol=1e-4, verbose=verbose),
        )

    def test_modulus_output_magnitude(self, verbose=False):
        """Output modulus equals act(|z| + bias)."""
        fn = ComplexActivation(nn.ReLU(), mode="modulus")
        with torch.no_grad():
            fn.bias.fill_(1.0)
        z   = _cx_randn(20) * 2.0
        out = fn(z)
        self.assertTrue(
            compare_tensors("modulus abs", out.abs(), torch.relu(z.abs() + 1.0), atol=1e-5, verbose=verbose),
        )

    def test_modulus_bias_shape_none(self):
        fn = ComplexActivation(nn.ReLU(), mode="modulus", bias_shape=None)
        self.assertEqual(fn.bias.shape, torch.Size([1]))

    def test_modulus_bias_shape_respected(self):
        fn = ComplexActivation(nn.ReLU(), mode="modulus", bias_shape=(4,))
        self.assertEqual(fn.bias.shape, torch.Size([4]))

    # --- identity fall-through ---

    def test_identity_passthrough(self, verbose=False):
        fn  = ComplexActivation(nn.ReLU(), mode="identity")
        z   = _cx_randn(*SHAPE)
        self.assertTrue(compare_tensors("identity", fn(z), z, atol=1e-7, verbose=verbose))

    # --- backward ---

    @parameterized.expand([("cartesian",), ("modulus",), ("identity",)])
    def test_backward(self, mode):
        fn = ComplexActivation(nn.ReLU(), mode=mode)
        z  = _cx_randn(*SHAPE) * 2.0 + (1.0 + 1.0j)
        zr = torch.view_as_real(z).requires_grad_(True)
        torch.view_as_real(fn(torch.view_as_complex(zr))).sum().backward()
        self.assertIsNotNone(zr.grad,                f"{mode}: grad is None")
        self.assertFalse(torch.isnan(zr.grad).any(), f"{mode}: NaN in grad")
        self.assertFalse(torch.isinf(zr.grad).any(), f"{mode}: Inf in grad")


# ===========================================================================
# TestMagnitudePreservingSiLU
# ===========================================================================
class TestMagnitudePreservingSiLU(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_shape_dtype_preserved(self):
        fn  = MagnitudePreservingSiLU()
        x   = torch.randn(2, 4, 6, 5)
        out = fn(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_output_equals_scaled_silu(self, verbose=False):
        """Output must equal silu(x) / normalization_factor."""
        norm = 0.596
        fn   = MagnitudePreservingSiLU(normalization_factor=norm)
        x    = torch.randn(8, 16)
        self.assertTrue(
            compare_tensors("mp_silu value", fn(x), torch.nn.functional.silu(x) / norm, atol=1e-6, verbose=verbose),
        )

    def test_custom_normalization_factor(self, verbose=False):
        norm = 0.75
        fn   = MagnitudePreservingSiLU(normalization_factor=norm)
        x    = torch.randn(8)
        self.assertTrue(
            compare_tensors("mp_silu custom", fn(x), torch.nn.functional.silu(x) / norm, atol=1e-6, verbose=verbose),
        )

    def test_backward(self):
        fn = MagnitudePreservingSiLU()
        x  = torch.randn(4, 8, requires_grad=True)
        fn(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any(), "NaN in grad")
        self.assertFalse(torch.isinf(x.grad).any(), "Inf in grad")


if __name__ == "__main__":
    unittest.main()
