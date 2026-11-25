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

from makani.models.common.pointwise_conv import PointwiseConv2d

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


@parameterized_class(("device",), _devices)
class TestPointwiseConv2d(unittest.TestCase):

    def setUp(self):
        disable_tf32()
        set_seed(333)

    def _pair(self, in_channels, out_channels, bias, groups):
        reference = nn.Conv2d(in_channels, out_channels, 1, bias=bias, groups=groups).to(self.device)
        layer = PointwiseConv2d(in_channels, out_channels, bias=bias, groups=groups).to(self.device)
        layer.load_state_dict(reference.state_dict(), strict=True)
        return reference, layer

    @parameterized.expand(
        [
            [1, 1, True, False],
            [4, 1, True, False],
            [1, 4, True, False],
            [4, 4, False, False],
            [1, 1, True, True],
            [4, 1, True, True],
            [1, 4, True, True],
            [4, 4, False, True],
        ],
        skip_on_empty=True,
    )
    def test_matches_conv2d(self, batch_size, groups, bias, autocast, verbose=False):
        """Output, input, weight and bias gradients of conv2d, in its dtype and in contiguous NCHW."""
        if autocast and self.device.type == "cpu":
            self.skipTest("bf16 autocast compared on the GPU only")
        reference, layer = self._pair(32, 64, bias, groups)
        inp = torch.randn(batch_size, 32, 9, 16, device=self.device)
        ograd = torch.randn(batch_size, 64, 9, 16, device=self.device)

        results = []
        for module in [layer, reference]:
            x = inp.clone().requires_grad_(True)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=autocast):
                y = module(x)
            y.backward(ograd.to(y.dtype))
            grads = [module.weight.grad] + ([module.bias.grad] if bias else [])
            results.append((y.detach(), x.grad, *grads))

        with self.subTest(desc="layout"):
            self.assertEqual(results[0][0].dtype, results[1][0].dtype)
            self.assertTrue(results[0][0].is_contiguous())

        # rounding relative to the size of the values, as the weight gradient sums over batch and grid; in bf16 the
        # GEMMs accumulate in fp32 and round once, as conv2d does
        tol = 1e-6 if not autocast else 1e-2
        names = ["output", "input grad", "weight grad", "bias grad"]
        for name, new, ref in zip(names, *results):
            with self.subTest(desc=name):
                atol = tol * ref.float().abs().max().item()
                self.assertTrue(compare_tensors(name, new.float(), ref.float(), atol=atol, rtol=tol, verbose=verbose))

    def test_is_a_conv2d(self):
        """Initializations that look for nn.Conv2d find the layer, and its state dict is that of nn.Conv2d."""
        reference, layer = self._pair(8, 16, True, 2)
        self.assertIsInstance(layer, nn.Conv2d)
        self.assertEqual(
            {k: v.shape for k, v in layer.state_dict().items()},
            {k: v.shape for k, v in reference.state_dict().items()},
        )


if __name__ == "__main__":
    unittest.main()
