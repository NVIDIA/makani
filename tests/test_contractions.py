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

from makani.models.common.contractions import (
    _contract_lmwise,
    _contract_lwise,
    _contract_sep_lmwise,
    _contract_sep_lwise,
    _contract_dense_pytorch,
)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

# ---------------------------------------------------------------------------
# Fixed small dimensions — tests run on CPU
# ---------------------------------------------------------------------------
B = 2   # batch
G = 2   # groups
I = 4   # in-channels per group  (== O for separable)
O = 3   # out-channels per group (non-separable only)
H = 6   # lmax  (l-modes)
W = 5   # mmax  (m-modes)


def _cx(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


# ---------------------------------------------------------------------------
# Reference implementations (plain torch.einsum, no @torch.compile).
# ---------------------------------------------------------------------------

def _ref_lmwise(x, w):     return torch.einsum("bgixy,gioxy->bgoxy", x, w)
def _ref_lwise(x, w):      return torch.einsum("bgixy,giox->bgoxy",  x, w)
def _ref_sep_lmwise(x, w): return torch.einsum("bgixy,gixy->bgixy",  x, w)
def _ref_sep_lwise(x, w):  return torch.einsum("bgixy,gix->bgixy",   x, w)

# ---------------------------------------------------------------------------
# Input / weight factories
# ---------------------------------------------------------------------------

def _mk_lmwise():     return _cx(B, G, I, H, W), _cx(G, I, O, H, W)
def _mk_lwise():      return _cx(B, G, I, H, W), _cx(G, I, O, H)
def _mk_sep_lmwise(): return _cx(B, G, I, H, W), _cx(G, I, H, W)
def _mk_sep_lwise():  return _cx(B, G, I, H, W), _cx(G, I, H)

# ---------------------------------------------------------------------------
# Master table: (name, compiled_fn, ref_fn, input_factory, expected_output_shape)
# ---------------------------------------------------------------------------
_KERNEL_CASES = [
    ("lmwise",     _contract_lmwise,     _ref_lmwise,     _mk_lmwise,     (B, G, O, H, W)),
    ("lwise",      _contract_lwise,      _ref_lwise,      _mk_lwise,      (B, G, O, H, W)),
    ("sep_lmwise", _contract_sep_lmwise, _ref_sep_lmwise, _mk_sep_lmwise, (B, G, I, H, W)),
    ("sep_lwise",  _contract_sep_lwise,  _ref_sep_lwise,  _mk_sep_lwise,  (B, G, I, H, W)),
]

# Lookup helpers
_MAKER = {name: mk  for name, _, _, mk,  _ in _KERNEL_CASES}
_REF   = {name: ref for name, _, ref, _, _ in _KERNEL_CASES}

# Dispatcher: (separable, operator_type) → kernel name
_DISPATCHER_CASES = [
    (False, "diagonal", "lmwise"),
    (False, "dhconv",   "lwise"),
    (True,  "diagonal", "sep_lmwise"),
    (True,  "dhconv",   "sep_lwise"),
]


# ===========================================================================
# 1. Output shape
# ===========================================================================
class TestContractionShape(unittest.TestCase):
    """Every kernel produces an output tensor with the expected shape."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @parameterized.expand([(name, fn, mk, shape) for name, fn, _, mk, shape in _KERNEL_CASES])
    def test_output_shape(self, name, fn, make_inputs, expected_shape):
        x, w = make_inputs()
        out = fn(x, w)
        self.assertEqual(
            tuple(out.shape), expected_shape,
            f"{name}: got {tuple(out.shape)}, expected {expected_shape}",
        )


# ===========================================================================
# 2. Compile correctness (@torch.compile == plain torch.einsum)
# ===========================================================================
class TestContractionCorrectness(unittest.TestCase):
    """Compiled kernels are numerically identical to the reference einsums."""

    def setUp(self):
        disable_tf32()

    @parameterized.expand([(name, fn, ref, mk) for name, fn, ref, mk, _ in _KERNEL_CASES])
    def test_compile_correctness(self, name, fn, ref_fn, make_inputs, verbose=False):
        set_seed(333)
        x, w = make_inputs()
        self.assertTrue(
            compare_tensors(f"{name} compile", fn(x, w), ref_fn(x, w), atol=1e-5, rtol=1e-4, verbose=verbose),
            f"{name}: compiled result differs from reference einsum",
        )


# ===========================================================================
# 3. Dispatcher routing
# ===========================================================================
class TestContractionDispatcher(unittest.TestCase):
    """_contract_dense_pytorch routes every (separable, operator_type)
    combination to the correct kernel."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    @parameterized.expand(
        [(f"sep{s}_op{op}", s, op, kname) for s, op, kname in _DISPATCHER_CASES]
    )
    def test_routing(self, _label, separable, operator_type, kernel_name, verbose=False):
        set_seed(333)
        x, w = _MAKER[kernel_name]()
        expected = _REF[kernel_name](x, w)
        got = _contract_dense_pytorch(x, w, separable=separable, operator_type=operator_type)
        self.assertTrue(
            compare_tensors(f"dispatcher {kernel_name}", got, expected, atol=1e-5, rtol=1e-4, verbose=verbose),
            f"dispatcher mismatch: separable={separable}, op={operator_type}",
        )

    def test_unknown_operator_nonsep_raises(self):
        x, w = _mk_lmwise()
        with self.assertRaises(ValueError):
            _contract_dense_pytorch(x, w, separable=False, operator_type="unknown")

    def test_unknown_operator_sep_raises(self):
        x, w = _mk_sep_lmwise()
        with self.assertRaises(ValueError):
            _contract_dense_pytorch(x, w, separable=True, operator_type="unknown")


# ===========================================================================
# 4. Separable == diagonal-weight non-separable
# ===========================================================================
class TestContractionConsistency(unittest.TestCase):
    """Mathematical relationships between separable and non-separable kernels."""

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def test_lmwise_diagonal_weight_equals_sep_lmwise(self, verbose=False):
        """Non-separable lmwise with diagonal (I==O) weight must equal sep_lmwise."""
        x     = _cx(B, G, I, H, W)
        w_sep = _cx(G, I, H, W)
        w_full = torch.zeros(G, I, I, H, W, dtype=torch.complex64)
        for i in range(I):
            w_full[:, i, i, :, :] = w_sep[:, i, :, :]
        self.assertTrue(
            compare_tensors(
                "lmwise diag == sep_lmwise",
                _contract_lmwise(x, w_full),
                _contract_sep_lmwise(x, w_sep),
                atol=1e-5, rtol=1e-4, verbose=verbose,
            )
        )

    def test_lwise_diagonal_weight_equals_sep_lwise(self, verbose=False):
        """Non-separable lwise with diagonal weight must equal sep_lwise."""
        x     = _cx(B, G, I, H, W)
        w_sep = _cx(G, I, H)
        w_full = torch.zeros(G, I, I, H, dtype=torch.complex64)
        for i in range(I):
            w_full[:, i, i, :] = w_sep[:, i, :]
        self.assertTrue(
            compare_tensors(
                "lwise diag == sep_lwise",
                _contract_lwise(x, w_full),
                _contract_sep_lwise(x, w_sep),
                atol=1e-5, rtol=1e-4, verbose=verbose,
            )
        )

    def test_lwise_equals_lmwise_mconst_weight(self, verbose=False):
        """lwise (no m-dim in weight) must equal lmwise when the weight is
        constant across m."""
        x    = _cx(B, G, I, H, W)
        w_l  = _cx(G, I, O, H)
        w_lm = w_l.unsqueeze(-1).expand(G, I, O, H, W).contiguous()
        self.assertTrue(
            compare_tensors(
                "lwise == lmwise mconst",
                _contract_lwise(x, w_l),
                _contract_lmwise(x, w_lm),
                atol=1e-5, rtol=1e-4, verbose=verbose,
            )
        )


# ===========================================================================
# 5. Backward pass
# ===========================================================================
class TestContractionBackward(unittest.TestCase):
    """Gradients through every kernel are finite and NaN-free."""

    def setUp(self):
        disable_tf32()

    @parameterized.expand([(name, fn, mk) for name, fn, _, mk, _ in _KERNEL_CASES])
    def test_backward(self, name, fn, make_inputs):
        set_seed(333)
        x, w = make_inputs()
        x = x.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        fn(x, w).sum().abs().backward()
        for tensor, label in ((x, "x"), (w, "w")):
            self.assertIsNotNone(tensor.grad,                f"{name}: {label}.grad is None")
            self.assertFalse(torch.isnan(tensor.grad).any(), f"{name}: NaN in {label}.grad")
            self.assertFalse(torch.isinf(tensor.grad).any(), f"{name}: Inf in {label}.grad")


if __name__ == "__main__":
    unittest.main()
