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

import os
import sys
import unittest
from parameterized import parameterized, parameterized_class

import torch
import torch.nn as nn

from makani.utils.grids import make_grid
from makani.models.common.layer_norm import GeometricInstanceNormS2, instance_norm
from makani.models.common.distributed.layer_norm import DistributedInstanceNorm2d

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def _reference_instance_norm(x, weight, bias, quad_weight, eps):
    """The instance norm with autograd through every op, in fp32, as the layers computed it before the analytic
    backward: weighted mean and variance, normalize, affine, cast back."""
    xf = x.to(torch.float32)
    if quad_weight is None:
        var, mean = torch.var_mean(xf, dim=(-2, -1), unbiased=False, keepdim=True)
    else:
        q = quad_weight.to(torch.float32) / torch.sum(quad_weight.to(torch.float32))
        mean = torch.sum(xf * q, dim=(-2, -1), keepdim=True)
        var = torch.sum(torch.square(xf - mean) * q, dim=(-2, -1), keepdim=True)
    y = (xf - mean) / torch.sqrt(var + eps)
    if weight is not None:
        y = weight.reshape(-1, 1, 1) * y + bias.reshape(-1, 1, 1)
    return y.to(x.dtype)


@parameterized_class(("device",), _devices)
class TestInstanceNorm(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _quad_weight(self, H, W, dtype=torch.float32):
        # positive weights that vary over the grid, not normalized
        return torch.rand(1, 1, H, W, dtype=dtype, device=self.device) + 0.1

    @parameterized.expand(
        [
            ["uniform", False, False],
            ["uniform affine", False, True],
            ["weighted", True, False],
            ["weighted affine", True, True],
        ],
        skip_on_empty=True,
    )
    def test_gradcheck(self, _, weighted, affine):
        """The analytic backward against finite differences, in fp64."""
        B, C, H, W = 2, 3, 5, 7
        x = torch.randn(B, C, H, W, dtype=torch.float64, device=self.device, requires_grad=True)
        quad_weight = self._quad_weight(H, W, dtype=torch.float64) if weighted else None
        if affine:
            weight = torch.randn(C, dtype=torch.float64, device=self.device, requires_grad=True)
            bias = torch.randn(C, dtype=torch.float64, device=self.device, requires_grad=True)
            inputs = (x, weight, bias)

            def func(x, w, b):
                return instance_norm(x, w, b, quad_weight, 1e-5)

        else:
            inputs = (x,)

            def func(x):
                return instance_norm(x, quad_weight=quad_weight, eps=1e-5)

        self.assertTrue(torch.autograd.gradcheck(func, inputs))

    @parameterized.expand(
        [
            [torch.float32, False],
            [torch.float32, True],
            [torch.bfloat16, False],
            [torch.bfloat16, True],
        ],
        skip_on_empty=True,
    )
    def test_matches_reference(self, dtype, weighted, verbose=False):
        """Outputs and gradients agree with differentiating through the plain fp32 formula, within fp32 rounding."""
        B, C, H, W = 2, 6, 17, 32
        inp = (3.0 + 2.0 * torch.randn(B, C, H, W, device=self.device)).to(dtype)
        ograd = torch.randn(B, C, H, W, device=self.device).to(dtype)
        quad_weight = self._quad_weight(H, W) if weighted else None
        weight = torch.randn(C, device=self.device)
        bias = torch.randn(C, device=self.device)

        results = []
        for func in [instance_norm, _reference_instance_norm]:
            x = inp.clone().requires_grad_(True)
            w = weight.clone().requires_grad_(True)
            b = bias.clone().requires_grad_(True)
            y = func(x, w, b, quad_weight, 1e-5)
            y.backward(ograd)
            results.append((y.detach(), x.grad, w.grad, b.grad))

        for name, new, ref in zip(["output", "input grad", "weight grad", "bias grad"], *results):
            with self.subTest(desc=name):
                self.assertEqual(new.dtype, ref.dtype)
                # fp32 rounding; the half precision output and input grad may round to the neighbouring value
                atol, rtol = (1e-4, 1e-5) if new.dtype == torch.float32 else (1e-2, 1e-2)
                self.assertTrue(compare_tensors(name, new.float(), ref.float(), atol=atol, rtol=rtol, verbose=verbose))

    def test_matches_torch_instance_norm(self):
        """The plain distributed layer on a single rank is torch's instance norm."""
        B, C, H, W = 3, 5, 16, 24
        ref = nn.InstanceNorm2d(C, eps=1e-5, affine=True, track_running_stats=False).to(self.device)
        with torch.no_grad():
            ref.weight.normal_()
            ref.bias.normal_()
        layer = DistributedInstanceNorm2d(C, eps=1e-5, affine=True).to(self.device)
        layer.load_state_dict(ref.state_dict())

        inp = torch.randn(B, C, H, W, device=self.device)
        ograd = torch.randn(B, C, H, W, device=self.device)
        results = []
        for module in [layer, ref]:
            x = inp.clone().requires_grad_(True)
            module(x).backward(ograd)
            results.append((x.grad, module.weight.grad, module.bias.grad))

        for name, new, ref in zip(["input grad", "weight grad", "bias grad"], *results):
            with self.subTest(desc=name):
                self.assertTrue(compare_tensors(name, new, ref, atol=1e-4, rtol=1e-5))

    @parameterized.expand(
        [
            [torch.bfloat16, "plain", False],
            [torch.bfloat16, "plain", True],
            [torch.bfloat16, "s2", False],
            [torch.bfloat16, "s2", True],
            [torch.float32, "plain", False],
            [torch.float32, "plain", True],
            [torch.float32, "s2", False],
            [torch.float32, "s2", True],
        ],
        skip_on_empty=True,
    )
    def test_saved_tensors(self, dtype, kind, compiled):
        """The only full-size tensor kept for the backward is the input itself, everything else is O(B C), or a
        parameter or buffer of the layer."""
        B, C, H, W = 2, 8, 32, 64
        if kind == "plain":
            layer = DistributedInstanceNorm2d(C, affine=True)
        else:
            layer = GeometricInstanceNormS2(make_grid((H, W), "equiangular"), C, affine=True)
        layer = layer.to(self.device)
        func = torch.compile(layer) if compiled else layer
        owned = {t.data_ptr() for t in list(layer.parameters()) + list(layer.buffers())}

        x = torch.randn(B, C, H, W, device=self.device, dtype=dtype, requires_grad=True)
        saved = []

        def pack(t):
            saved.append(t)
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                y = func(x)
        # the backward runs with what was saved
        y.backward(torch.ones_like(y))

        full_size = [t for t in saved if t.numel() == x.numel()]
        with self.subTest(desc="full-size saved tensors"):
            self.assertEqual(len(full_size), 1, f"expected only x, got {[(t.shape, t.dtype) for t in full_size]}")
            self.assertEqual(full_size[0].dtype, x.dtype)
            self.assertEqual(full_size[0].data_ptr(), x.data_ptr())

        with self.subTest(desc="saved bytes"):
            extra = sum(
                t.numel() * t.element_size() for t in saved if t.data_ptr() not in owned and t.numel() != x.numel()
            )
            self.assertLessEqual(extra, 8 * (2 * B * C + 1))


if __name__ == "__main__":
    unittest.main()
