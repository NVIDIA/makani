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

import math
import unittest
from parameterized import parameterized

import torch

from makani.models.common.activations import ComplexReLU

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, compare_tensors

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
# 1. Shape and dtype preservation
# ===========================================================================
class TestComplexReLUShapeDtype(unittest.TestCase):
    """Output shape and dtype must match the input for every mode."""

    def setUp(self):
        disable_tf32()
        torch.manual_seed(0)

    @parameterized.expand(_ALL_MODES)
    def test_shape_dtype_preserved(self, mode):
        fn = ComplexReLU(mode=mode)
        z  = _cx_randn(*SHAPE)
        out = fn(z)
        self.assertEqual(out.shape, z.shape,   f"{mode}: shape changed")
        self.assertEqual(out.dtype, z.dtype,   f"{mode}: dtype changed")


# ===========================================================================
# 2. "real" mode: imaginary part is unchanged, real part is ReLU-clipped
# ===========================================================================
class TestComplexReLUReal(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        torch.manual_seed(1)

    def test_real_part_clipped(self):
        fn = ComplexReLU(mode="real", negative_slope=0.0)
        # mix of positive and negative real parts
        real = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        imag = torch.tensor([ 3.0,  4.0, 5.0, 6.0, 7.0])
        z   = _cx(real, imag)
        out = fn(z)
        expected_real = real.clamp(min=0.0)
        self.assertTrue(
            compare_tensors("real mode real part", out.real, expected_real, atol=1e-6),
        )

    def test_imaginary_part_unchanged(self):
        """The imaginary part must pass through unmodified regardless of sign."""
        fn = ComplexReLU(mode="real", negative_slope=0.0)
        real = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        imag = torch.tensor([-3.0,  4.0, -5.0, 6.0, -7.0])
        z   = _cx(real, imag)
        out = fn(z)
        self.assertTrue(
            compare_tensors("real mode imag part", out.imag, imag, atol=1e-6),
            "imaginary part was modified in 'real' mode",
        )


# ===========================================================================
# 3. "cartesian" mode: both real and imaginary parts independently clipped
# ===========================================================================
class TestComplexReLUCartesian(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        torch.manual_seed(2)

    def test_real_part_clipped(self):
        fn = ComplexReLU(mode="cartesian", negative_slope=0.0)
        real = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        imag = torch.tensor([ 3.0,  4.0, 5.0, 6.0, 7.0])
        z   = _cx(real, imag)
        out = fn(z)
        self.assertTrue(
            compare_tensors("cartesian real part", out.real, real.clamp(min=0.0), atol=1e-6),
        )

    def test_imaginary_part_clipped(self):
        """Unlike 'real' mode, negative imaginary parts are zeroed."""
        fn = ComplexReLU(mode="cartesian", negative_slope=0.0)
        real = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        imag = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        z   = _cx(real, imag)
        out = fn(z)
        self.assertTrue(
            compare_tensors("cartesian imag part", out.imag, imag.clamp(min=0.0), atol=1e-6),
        )

    def test_differs_from_real_mode_on_negative_imag(self):
        """'cartesian' zeros negative imaginary parts; 'real' preserves them."""
        fn_cx   = ComplexReLU(mode="cartesian", negative_slope=0.0)
        fn_real = ComplexReLU(mode="real",      negative_slope=0.0)
        real = torch.tensor([1.0])
        imag = torch.tensor([-1.0])   # negative imag — the two modes must differ
        z = _cx(real, imag)
        out_cx   = fn_cx(z)
        out_real = fn_real(z)
        self.assertFalse(
            torch.allclose(out_cx.imag, out_real.imag),
            "cartesian and real modes should differ on negative imaginary input",
        )


# ===========================================================================
# 4. "modulus" mode: phase preserved, modulus activated
# ===========================================================================
class TestComplexReLUModulus(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        torch.manual_seed(3)

    def _make_fn(self, bias_val=0.0):
        fn = ComplexReLU(mode="modulus", bias_shape=(1,), scale=bias_val)
        # override bias to an exact value for reproducible tests
        with torch.no_grad():
            fn.bias.fill_(bias_val)
        return fn

    def test_phase_preserved_when_nonzero(self):
        """When |z| + bias > 0 the output phase must equal the input phase."""
        fn = self._make_fn(bias_val=0.0)
        # use inputs well away from zero so phase is numerically stable
        z = _cx_randn(10) * 5.0 + (2.0 + 2.0j)
        out = fn(z)
        nonzero = out.abs() > 1e-6
        self.assertTrue(
            compare_tensors(
                "modulus phase",
                torch.angle(out)[nonzero],
                torch.angle(z)[nonzero],
                atol=1e-4,
            ),
            "phase changed in 'modulus' mode",
        )

    def test_modulus_activated(self):
        """Output modulus equals max(|z| + bias, 0)."""
        fn = self._make_fn(bias_val=1.0)
        z   = _cx_randn(20) * 3.0
        out = fn(z)
        zabs = z.abs()
        expected_abs = (zabs + 1.0).clamp(min=0.0)
        self.assertTrue(
            compare_tensors("modulus abs", out.abs(), expected_abs, atol=1e-5),
        )

    def test_zero_when_modulus_plus_bias_nonpositive(self):
        """When |z| + bias ≤ 0 the output must be zero."""
        fn = self._make_fn(bias_val=-10.0)   # bias so negative that all outputs should be 0
        z = _cx_randn(20)                     # |z| ~ 1, so |z| + (-10) < 0
        out = fn(z)
        self.assertTrue(
            compare_tensors("modulus zero", out.abs(), torch.zeros_like(out.abs()), atol=1e-6),
        )


# ===========================================================================
# 5. "halfplane" mode: passthrough vs scale based on angle
# ===========================================================================
class TestComplexReLUHalfplane(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        torch.manual_seed(4)

    def _make_fn(self, bias_angle=0.0, negative_slope=0.0):
        fn = ComplexReLU(mode="halfplane", negative_slope=negative_slope, bias_shape=(1,))
        with torch.no_grad():
            fn.bias.fill_(bias_angle)
        return fn

    def _z_at_angle(self, angle_rad, magnitude=1.0):
        """Return a complex scalar at the given angle."""
        r = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)], dtype=torch.float32) * magnitude
        return torch.complex(r[0:1], r[1:2])

    def test_passthrough_in_window(self):
        """Inputs whose angle falls in [bias, bias + π/2) must pass through unchanged."""
        fn = self._make_fn(bias_angle=0.0)
        # angle = π/4 is well inside [0, π/2)
        z   = self._z_at_angle(math.pi / 4)
        out = fn(z)
        self.assertTrue(compare_tensors("halfplane passthrough", out, z, atol=1e-6))

    def test_scaled_outside_window(self):
        """Inputs outside the window are multiplied by negative_slope."""
        slope = 0.1
        fn = self._make_fn(bias_angle=0.0, negative_slope=slope)
        # angle = π (well outside [0, π/2))
        z   = self._z_at_angle(math.pi)
        out = fn(z)
        self.assertTrue(compare_tensors("halfplane scaled", out, slope * z, atol=1e-6))

    def test_zero_slope_kills_outside_window(self):
        """With negative_slope=0, inputs outside the window collapse to zero."""
        fn = self._make_fn(bias_angle=0.0, negative_slope=0.0)
        z   = self._z_at_angle(math.pi)   # angle = π, outside [0, π/2)
        out = fn(z)
        self.assertTrue(compare_tensors("halfplane zero", out, torch.zeros_like(z), atol=1e-6))

    def test_bias_shifts_window(self):
        """Shifting bias by π/4 makes angle=π/4 fall at the window boundary (excluded)."""
        fn = self._make_fn(bias_angle=math.pi / 2)
        # angle = π/4 is now below bias → outside window
        z   = self._z_at_angle(math.pi / 4)
        out = fn(z)
        # negative_slope=0 → output should be zero
        self.assertTrue(compare_tensors("halfplane bias shift", out, torch.zeros_like(z), atol=1e-6))


# ===========================================================================
# 6. negative_slope effect on "real" and "cartesian"
# ===========================================================================
class TestComplexReLUNegativeSlope(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        torch.manual_seed(5)

    @parameterized.expand([("real",), ("cartesian",)])
    def test_negative_real_scaled_by_slope(self, mode):
        slope = 0.2
        fn = ComplexReLU(mode=mode, negative_slope=slope)
        real = torch.tensor([-3.0])
        imag = torch.tensor([ 0.0])
        z   = _cx(real, imag)
        out = fn(z)
        expected_real = torch.tensor([slope * -3.0])
        self.assertTrue(
            compare_tensors(f"{mode} leaky slope", out.real, expected_real, atol=1e-6),
        )

    @parameterized.expand([("real",), ("cartesian",)])
    def test_positive_real_unaffected_by_slope(self, mode):
        fn = ComplexReLU(mode=mode, negative_slope=0.2)
        real = torch.tensor([3.0])
        imag = torch.tensor([0.0])
        z   = _cx(real, imag)
        out = fn(z)
        self.assertTrue(
            compare_tensors(f"{mode} positive unaffected", out.real, real, atol=1e-6),
        )


# ===========================================================================
# 7. Unknown mode raises NotImplementedError
# ===========================================================================
class TestComplexReLUUnknownMode(unittest.TestCase):

    def test_unknown_mode_raises(self):
        fn = ComplexReLU(mode="unknown_mode")
        z  = _cx_randn(4)
        with self.assertRaises(NotImplementedError):
            fn(z)


# ===========================================================================
# 8. Backward pass — finite, NaN-free gradients for all modes
# ===========================================================================
class TestComplexReLUBackward(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        torch.manual_seed(6)

    @parameterized.expand(_ALL_MODES)
    def test_backward(self, mode):
        fn = ComplexReLU(mode=mode)
        # keep inputs away from zero to avoid gradient singularities in modulus mode
        z  = _cx_randn(*SHAPE) * 2.0 + (1.0 + 1.0j)
        # ComplexReLU operates on complex tensors; autograd works through view_as_real
        zr = torch.view_as_real(z).requires_grad_(True)
        zc = torch.view_as_complex(zr)
        out = fn(zc)
        torch.view_as_real(out).sum().backward()
        self.assertIsNotNone(zr.grad,                           f"{mode}: grad is None")
        self.assertFalse(torch.isnan(zr.grad).any(),            f"{mode}: NaN in grad")
        self.assertFalse(torch.isinf(zr.grad).any(),            f"{mode}: Inf in grad")

    def test_modulus_backward_near_zero(self):
        """The modulus mode divides by |z|; verify gradients remain finite near zero."""
        fn = ComplexReLU(mode="modulus")
        with torch.no_grad():
            fn.bias.fill_(1.0)   # ensure |z| + bias > 0 even near z=0
        z_small = _cx_randn(16) * 1e-3   # very small magnitude
        zr = torch.view_as_real(z_small).requires_grad_(True)
        zc = torch.view_as_complex(zr)
        torch.view_as_real(fn(zc)).sum().backward()
        self.assertFalse(torch.isnan(zr.grad).any(), "NaN in grad near zero (modulus mode)")
        self.assertFalse(torch.isinf(zr.grad).any(), "Inf in grad near zero (modulus mode)")


if __name__ == "__main__":
    unittest.main()
