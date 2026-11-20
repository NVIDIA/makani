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

import torch
from parameterized import parameterized

from makani.losses import LossHandler
from makani.metrics.handler import MetricsHandler
from makani.utils.jit import compiles_metrics, compiles_model, get_jit_mode
from makani.utils.params import ParamsBase

from .testutils import get_default_parameters


def _params(**kwargs):
    params = ParamsBase()
    for key, value in kwargs.items():
        params[key] = value
    return params


class TestJitMode(unittest.TestCase):
    """What each jit_mode compiles."""

    @parameterized.expand(
        [
            ("none", False, False),
            ("metrics", False, True),
            ("all", True, True),
        ]
    )
    def test_what_each_mode_compiles(self, mode, model, metrics):
        params = _params(jit_mode=mode)
        self.assertEqual(compiles_model(params), model)
        self.assertEqual(compiles_metrics(params), metrics)

    def test_compiles_nothing_without_a_mode(self):
        params = _params()
        self.assertEqual(get_jit_mode(params), "none")
        self.assertFalse(compiles_model(params))
        self.assertFalse(compiles_metrics(params))

    def test_inductor_is_all_with_a_warning(self):
        with self.assertWarnsRegex(FutureWarning, "deprecated"):
            self.assertEqual(get_jit_mode(_params(jit_mode="inductor")), "all")

    def test_refuses_an_unknown_mode(self):
        with self.assertRaises(ValueError):
            get_jit_mode(_params(jit_mode="script"))


class TestHandlersFollowJitMode(unittest.TestCase):
    """The loss and metric handlers compile as jit_mode says, unless told explicitly."""

    def setUp(self):
        torch._dynamo.reset()

        self.params = get_default_parameters()
        self.params.img_shape_x = self.params.img_shape_x_resampled = self.params.img_crop_shape_x = 16
        self.params.img_shape_y = self.params.img_shape_y_resampled = self.params.img_crop_shape_y = 32
        self.params.img_local_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_shape_y
        self.params.img_local_offset_x = self.params.img_crop_offset_x = 0
        self.params.img_local_offset_y = self.params.img_crop_offset_y = 0
        self.params.losses = [{"type": "l2"}]
        self.params.dhours = 1

    def _loss_compiled(self, **kwargs):
        handler = LossHandler(self.params, **kwargs)
        return isinstance(handler.loss_fn[0], torch._dynamo.eval_frame.OptimizedModule)

    def _metrics_compiled(self, **kwargs):
        names = self.params.channel_names
        clim = torch.zeros(1, len(names), self.params.img_shape_x, self.params.img_shape_y)
        handler = MetricsHandler(self.params, clim, 1, torch.device("cpu"), rmse_var_names=names, **kwargs)
        return handler.metric_handles[0].metric_forward is not handler.metric_handles[0].metric_func

    @parameterized.expand([("none", False), ("metrics", True), ("all", True)])
    def test_handlers_follow_the_mode(self, mode, compiled):
        self.params.jit_mode = mode
        self.assertEqual(self._loss_compiled(), compiled)
        self.assertEqual(self._metrics_compiled(), compiled)

    @parameterized.expand([("none", True), ("metrics", False)])
    def test_an_explicit_compile_overrides_the_mode(self, mode, compile):
        self.params.jit_mode = mode
        self.assertEqual(self._loss_compiled(compile=compile), compile)
        self.assertEqual(self._metrics_compiled(compile=compile), compile)


if __name__ == "__main__":
    unittest.main()
