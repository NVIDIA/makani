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

from torch._functorch.aot_autograd import aot_module_simplified

from makani.models import model_registry
from makani.models.common.pointwise_conv import PointwiseConv2d

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, compare_tensors, get_default_parameters, single_step_rollout

_devices = [(torch.device("cpu"),)]
if torch.cuda.is_available():
    _devices.append((torch.device("cuda"),))


def _count_convolutions(func, *args):
    """Compiles ``func``, runs its forward and backward on ``args`` and counts the convolutions in the forward and
    backward graphs that AOT autograd hands to the compiler. A convolution in them makes inductor switch the graph to
    channels-last, which is what the 1x1 convolutions as matmuls avoid."""
    count = 0

    def compiler(gm, _):
        nonlocal count
        count += sum(1 for n in gm.graph.nodes if n.op == "call_function" and "convolution" in str(n.target))
        return gm.forward

    def backend(gm, example_inputs):
        return aot_module_simplified(gm, example_inputs, fw_compiler=compiler, bw_compiler=compiler)

    torch._dynamo.reset()
    out = torch.compile(func, backend=backend)(*args)
    out.float().sum().backward()
    torch._dynamo.reset()
    return count


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

    @parameterized.expand([[1, 1], [4, 1], [1, 4], [4, 4]], skip_on_empty=True)
    def test_compiled_matches_eager(self, batch_size, groups, verbose=False):
        """Under torch.compile the layer gives what it gives eagerly, and the input gradient stays NCHW."""
        _, layer = self._pair(32, 64, True, groups)
        inp = torch.randn(batch_size, 32, 9, 16, device=self.device)
        ograd = torch.randn(batch_size, 64, 9, 16, device=self.device)

        results = []
        torch._dynamo.reset()
        for func in [layer, torch.compile(layer)]:
            layer.zero_grad()
            x = inp.clone().requires_grad_(True)
            y = func(x)
            y.backward(ograd)
            results.append((y.detach(), x.grad, layer.weight.grad.clone(), layer.bias.grad.clone()))
        torch._dynamo.reset()

        with self.subTest(desc="input grad layout"):
            self.assertTrue(results[1][1].is_contiguous())
        for name, compiled, eager in zip(["output", "input grad", "weight grad", "bias grad"], results[1], results[0]):
            with self.subTest(desc=name):
                atol = 1e-6 * eager.abs().max().item()
                self.assertTrue(compare_tensors(name, compiled, eager, atol=atol, rtol=1e-6, verbose=verbose))

    def test_compiled_graph_has_no_convolution(self):
        """The compiled forward and backward of the layer contain no convolution."""
        layer = torch.nn.Sequential(PointwiseConv2d(16, 32), torch.nn.GELU(), PointwiseConv2d(32, 16, groups=4))
        layer = layer.to(self.device)
        x = torch.randn(2, 16, 8, 12, device=self.device, requires_grad=True)
        self.assertEqual(_count_convolutions(layer, x), 0)

    @parameterized.expand([["SFNO"], ["FCN3"], ["FCN3.1"], ["SNO"]], skip_on_empty=True)
    def test_compiled_model_has_no_convolution(self, nettype):
        """The compiled forward and backward of the spherical models contain no convolution, so inductor keeps them
        and the gradients they pass on in NCHW."""
        params = get_default_parameters()
        params.history_normalization_mode = "none"
        params.img_shape_x, params.img_shape_y = 36, 72
        params.img_local_shape_x = params.img_crop_shape_x = params.img_shape_x
        params.img_local_shape_y = params.img_crop_shape_y = params.img_shape_y
        params.img_shape_x_resampled, params.img_shape_y_resampled = params.img_shape_x, params.img_shape_y
        params.img_local_offset_x = params.img_local_offset_y = 0
        params.img_crop_offset_x = params.img_crop_offset_y = 0
        params.batch_size = 2
        params.nettype = nettype
        model = model_registry.get_model(params).to(self.device)

        x = torch.randn(2, params.N_in_channels, 36, 72, device=self.device, requires_grad=True)
        self.assertEqual(_count_convolutions(lambda x: model(single_step_rollout(x.unsqueeze(1))), x), 0)

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
