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

import importlib
import os
import sys
import tempfile
import textwrap
import unittest

import numpy as np
import torch
import torch.nn as nn

from makani.models import model_registry
from makani.models.parametrizations import ConstraintsWrapper

from .testutils import get_default_parameters, NUM_CHANNELS

# a network outside of makani, which a config names by its file: nettype: "path/to/file.py:RecordingNet"
_NETWORK_SOURCE = textwrap.dedent(
    """
    import torch.nn as nn


    class RecordingNet(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.out_chans = kwargs["out_chans"]

        def forward(self, x):
            return x[:, : self.out_chans]


    class NotAModule:
        pass
    """
)


class TestModelRegistry(unittest.TestCase):
    """
    Registering networks, by class or by the file that defines them, and building them with
    ``get_model`` from the parameters, with the normalization statistics, the hydrostatic
    balance climatology and the constraints the parameters ask for.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.network_file = os.path.join(self.tmpdir.name, "external_net.py")
        with open(self.network_file, "w") as f:
            f.write(_NETWORK_SOURCE)
        self.registered_before = set(model_registry.list_models())

        self.params = get_default_parameters()
        self.params.nettype = f"{self.network_file}:RecordingNet"

    def tearDown(self):
        for name in set(model_registry.list_models()) - self.registered_before:
            del model_registry._model_registry[name]
        sys.modules.pop("makani.models._dynamic.external_net", None)
        self.tmpdir.cleanup()

    def _save(self, name, array):
        path = os.path.join(self.tmpdir.name, name)
        np.save(path, array)
        return path

    def _set_zscore(self):
        means = np.arange(NUM_CHANNELS, dtype=np.float64).reshape(1, NUM_CHANNELS, 1, 1)
        stds = 1.0 + means
        self.params.normalization = "zscore"
        self.params.global_means_path = self._save("means.npy", means)
        self.params.global_stds_path = self._save("stds.npy", stds)
        return means, stds

    def test_nettype_names_a_file(self):
        with self.assertLogs(level="WARNING") as logs:
            model = model_registry.get_model(self.params)

        self.assertIn("Trying to register it", "\n".join(logs.output))
        self.assertIn(self.params.nettype, model_registry.list_models())
        net = model.model
        self.assertEqual(type(net).__name__, "RecordingNet")
        # the network's module is importable under a dotted name, which torch.compile needs
        self.assertIs(importlib.import_module(type(net).__module__), sys.modules["makani.models._dynamic.external_net"])
        self.assertEqual(net.kwargs["inp_shape"], (self.params.img_shape_x, self.params.img_shape_y))
        self.assertEqual(net.kwargs["out_shape"], net.kwargs["inp_shape"])
        self.assertEqual(net.kwargs["inp_chans"], self.params.N_in_channels)
        self.assertEqual(net.kwargs["out_chans"], self.params.N_out_channels)
        # the parameters are passed on as keyword arguments too
        self.assertEqual(net.kwargs["channel_names"], self.params.channel_names)

    def test_register_a_file_under_a_name(self):
        model_registry.register_model(f"{self.network_file}:RecordingNet", "external")
        self.assertIn("external", model_registry.list_models())

    def test_register_a_class(self):
        class LocalNet(nn.Module):
            pass

        model_registry.register_model(LocalNet)
        self.assertIn("LocalNet", model_registry.list_models())

        with self.assertRaisesRegex(ValueError, "already in use"):
            model_registry.register_model(LocalNet)

    def test_register_refuses_what_is_not_a_module(self):
        with self.assertRaisesRegex(ValueError, "Only subclasses of torch.nn.Module"):
            model_registry.register_model(f"{self.network_file}:NotAModule", "not_a_module")

    def test_register_refuses_a_malformed_file_reference(self):
        with self.assertRaisesRegex(ValueError, "Expected model string of format"):
            model_registry.register_model(self.network_file)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            model_registry.register_model("/nonexistent/net.py:Net")

    def test_normalization_statistics_of_the_output_channels(self):
        means, stds = self._set_zscore()
        self.params.out_channels = [1, 3]
        self.params.N_out_channels = 2

        net = model_registry.get_model(self.params).model

        np.testing.assert_array_equal(net.kwargs["normalization_means"], means.flatten()[[1, 3]])
        np.testing.assert_array_equal(net.kwargs["normalization_stds"], stds.flatten()[[1, 3]])

    def test_normalization_needs_the_output_channels(self):
        self._set_zscore()
        del self.params.out_channels
        with self.assertRaisesRegex(ValueError, "requires params.out_channels"):
            model_registry.get_model(self.params)

    def test_hydrostatic_balance_climatology(self):
        clim = np.arange(4.0).reshape(1, 4, 1, 1)
        self.params.hydrostatic_balance_means_path = self._save("hb.npy", clim)

        net = model_registry.get_model(self.params).model

        np.testing.assert_array_equal(net.kwargs["hydrostatic_balance_means"], clim)

    def test_constraints_wrap_the_network(self):
        means, stds = self._set_zscore()
        clamped = self.params.channel_names[1]
        self.params.constraints = [{"type": "nonnegative", "options": {"mode": "silu", "names_to_clamp": [clamped]}}]

        cwrap = model_registry.get_model(self.params).model

        self.assertIsInstance(cwrap, ConstraintsWrapper)
        self.assertEqual(type(cwrap.model).__name__, "RecordingNet")
        self.assertEqual(cwrap.model.kwargs["out_chans"], NUM_CHANNELS)

        # in physical units, the clamped channel comes out nonnegative, and the others as they went in
        cwrap.eval()
        x = torch.full((1, NUM_CHANNELS, 4, 8), -10.0)
        with torch.no_grad():
            out = cwrap(x)
        physical = out * torch.from_numpy(stds).float() + torch.from_numpy(means).float()
        self.assertTrue(torch.all(physical[:, 1] >= 0.0))
        others = [c for c in range(NUM_CHANNELS) if c != 1]
        self.assertTrue(torch.equal(out[:, others], x[:, others]))


if __name__ == "__main__":
    unittest.main()
