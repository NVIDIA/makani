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

import logging
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn
from parameterized import parameterized

from makani.driver.base import Driver
from makani.utils.params import ParamsBase


def _params(optimizer_type, **kwargs):
    params = ParamsBase()
    params["optimizer_type"] = optimizer_type
    params["optimizer_beta1"] = 0.8
    params["optimizer_beta2"] = 0.95
    params["lr"] = 0.25
    for key, value in kwargs.items():
        params[key] = value
    return params


def _model():
    return nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2))


class TestGetOptimizer(unittest.TestCase):
    """The optimizers ``Driver.get_optimizer`` builds from the parameters."""

    def setUp(self):
        # get_optimizer only reads the logger of the driver, so the expensive constructor is skipped
        self.driver = Driver.__new__(Driver)
        self.driver.log_to_screen = True
        self.driver.logger = logging.getLogger("test_optimizers")

    def _build(self, params, model=None):
        with self.assertLogs("test_optimizers", level="INFO"):
            return self.driver.get_optimizer(model if model is not None else _model(), params)

    @parameterized.expand([("Adam", torch.optim.Adam), ("AdamW", torch.optim.AdamW)])
    def test_adam(self, name, cls):
        optimizer = self._build(_params(name, optimizer_eps=1e-6))

        self.assertIs(type(optimizer), cls)
        group = optimizer.param_groups[0]
        self.assertEqual(group["lr"], 0.25)
        self.assertEqual(group["betas"], (0.8, 0.95))
        self.assertEqual(group["eps"], 1e-6)
        self.assertTrue(group["foreach"])

    def test_sgd(self):
        optimizer = self._build(_params("SGD", momentum=0.9, nesterov=True))

        self.assertIs(type(optimizer), torch.optim.SGD)
        group = optimizer.param_groups[0]
        self.assertEqual(group["lr"], 0.25)
        self.assertEqual(group["momentum"], 0.9)
        self.assertTrue(group["nesterov"])

    def test_sgd_defaults_to_plain_gradient_descent(self):
        group = self._build(_params("SGD")).param_groups[0]
        self.assertEqual(group["momentum"], 0)
        self.assertFalse(group["nesterov"])

    @parameterized.expand(["Adam", "AdamW", "SGD"])
    def test_weight_decay_modes(self, name):
        # "full" decays every parameter; "transformer" spares the biases and the norm parameters
        model = _model()
        num_params = len(list(model.parameters()))
        for mode, num_decayed in [("full", num_params), ("transformer", 2)]:
            with self.subTest(mode=mode):
                optimizer = self._build(_params(name, weight_decay=0.1, weight_decay_mode=mode), model)
                decayed = sum(len(g["params"]) for g in optimizer.param_groups if g["weight_decay"] == 0.1)
                total = sum(len(g["params"]) for g in optimizer.param_groups)
                self.assertEqual(total, num_params)
                self.assertEqual(decayed, num_decayed)

    def test_sirfshampoo(self):
        # sirfshampoo is optional; a stand-in checks what the driver hands it
        calls = []

        class SIRFShampoo:
            def __init__(self, model, lr):
                calls.append((model, lr))

        model = _model()
        with mock.patch.dict(sys.modules, {"sirfshampoo": types.SimpleNamespace(SIRFShampoo=SIRFShampoo)}):
            optimizer = self._build(_params("SIRFShampoo"), model)

        self.assertIsInstance(optimizer, SIRFShampoo)
        self.assertEqual(calls, [(model, 0.25)])

    def test_unknown_optimizer_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown optimizer type"):
            self.driver.get_optimizer(_model(), _params("Adagrad"))


if __name__ == "__main__":
    unittest.main()
