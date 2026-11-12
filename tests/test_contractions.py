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

import unittest
from parameterized import parameterized

import torch

from makani.models.common.contractions import _contract_lmwise, _contract_dense_pytorch

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors
from makani.utils.functions import contiguous_complex_safe

# ---------------------------------------------------------------------------
# Fixed small dimensions — tests run on CPU
# ---------------------------------------------------------------------------
B = 2  # batch
G = 2  # groups
I = 4  # in-channels per group
O = 3  # out-channels per group
H = 6  # modes along the first spectral dimension
W = 5  # modes along the second spectral dimension


def _cx(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _ref_lmwise(x, w):
    """Reference contraction, an explicit loop over the modes."""
    out = torch.zeros(x.shape[0], w.shape[0], w.shape[2], x.shape[-2], x.shape[-1], dtype=x.dtype)
    for k in range(x.shape[-2]):
        for m in range(x.shape[-1]):
            out[..., k, m] = torch.einsum("bgi,gio->bgo", x[..., k, m], w[..., k, m])
    return out


def _mk_lmwise():
    return _cx(B, G, I, H, W), _cx(G, I, O, H, W)


_KERNELS = [("kernel", _contract_lmwise), ("dense", _contract_dense_pytorch)]


class TestContraction(unittest.TestCase):
    """The mode-wise contraction of the planar spectral convolution."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @parameterized.expand(_KERNELS)
    def test_matches_reference(self, name, fn, verbose=False):
        x, w = _mk_lmwise()
        out = fn(x, w)
        self.assertEqual(tuple(out.shape), (B, G, O, H, W))
        self.assertTrue(
            compare_tensors(name, out, _ref_lmwise(x, w), atol=1e-5, rtol=1e-4, verbose=verbose),
            f"{name}: differs from the loop over modes",
        )

    def test_dense_takes_non_contiguous_input(self, verbose=False):
        """The spectral convolution hands over a reshaped transform output; the result has to be contiguous."""
        x, w = _mk_lmwise()
        x_t = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        self.assertFalse(x_t.is_contiguous())
        out = _contract_dense_pytorch(x_t, w)
        self.assertTrue(out.is_contiguous())
        self.assertTrue(
            compare_tensors("non-contiguous", out, _ref_lmwise(x, w), atol=1e-5, rtol=1e-4, verbose=verbose)
        )

    @parameterized.expand(_KERNELS)
    def test_backward(self, name, fn):
        x, w = _mk_lmwise()
        x = x.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        fn(x, w).sum().abs().backward()
        for tensor, label in ((x, "x"), (w, "w")):
            self.assertIsNotNone(tensor.grad, f"{name}: {label}.grad is None")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"{name}: non-finite {label}.grad")


class TestContiguousComplexSafe(unittest.TestCase):
    """The complex-safe contiguous() has to be a pure re-view, not a numerical change.

    It exists because inductor cannot codegen a copy over a complex buffer (triton has no
    complex type: KeyError: 'complex64'), so the copy is taken on the real (..., 2) view
    instead. That is only a legitimate substitution if the values and the resulting layout
    match what plain .contiguous() would have produced.
    """

    def setUp(self):
        set_seed(333)

    def test_complex_values_preserved(self):
        # transpose to make it non-contiguous, which is the case that actually copies
        x = torch.randn(4, 8, 6, dtype=torch.complex64).transpose(-1, -2)
        self.assertFalse(x.is_contiguous())

        out = contiguous_complex_safe(x)

        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.dtype, x.dtype)
        self.assertTrue(torch.equal(out, x.contiguous()))

    def test_already_contiguous_complex(self):
        x = torch.randn(2, 3, 4, dtype=torch.complex64)

        out = contiguous_complex_safe(x)

        self.assertTrue(out.is_contiguous())
        self.assertTrue(torch.equal(out, x))

    def test_real_tensor_takes_plain_path(self):
        x = torch.randn(4, 8, 6).transpose(-1, -2)

        out = contiguous_complex_safe(x)

        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(out, x.contiguous()))


if __name__ == "__main__":
    unittest.main()
