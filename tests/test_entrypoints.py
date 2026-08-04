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

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import warnings
from parameterized import parameterized
from importlib.metadata import entry_points


class TestEntryPoints(unittest.TestCase):

    def setUp(self):
        self.model_entry_points = {
            entry_point.name: entry_point
            for entry_point in entry_points(group="physicsnemo.models")
            if not entry_point.value.startswith("physicsnemo.experimental.models")
        }

    @parameterized.expand(["SFNO"])
    def test_model_entry_points(self, model_name):
        """Test model entry points"""

        # Check the model entry point.
        model_ep = self.model_entry_points.get(model_name)
        with self.subTest(desc="model entry point is not None"):
            self.assertIsNotNone(model_ep)

        # Try loading the model type.
        model_type = model_ep.load()
        with self.subTest(desc="model type is not None"):
            self.assertIsNotNone(model_type)

        # Create the model.
        model = model_type()
        with self.subTest(desc="model is not None"):
            self.assertIsNotNone(model)


class TestPhysicsNeMoCompat(unittest.TestCase):
    """Guards the PhysicsNeMo 1.x/2.x compatibility contract.

    PhysicsNeMo 2.0 made ``Module.from_torch`` registration opt-in and changed
    the generated class name. Both changes are silent -- the old call still
    succeeds, it just stops registering -- so nothing else in the suite would
    catch a regression here. These tests assert the behavior makani relies on,
    which :mod:`makani.models.physicsnemo_compat` normalizes across versions.
    """

    @parameterized.expand(
        [
            ("SFNO", "makani.models.networks.sfnonet", "SphericalFourierNeuralOperatorNet"),
            ("FNO", "makani.models.networks.sfnonet", "FourierNeuralOperatorNet"),
            ("FCN3", "makani.models.networks.fourcastnet3", "AtmoSphericNeuralOperatorNet"),
            ("FCN3", "makani.models.networks.fourcastnet3_1", "AtmoSphericNeuralOperatorNet31"),
        ]
    )
    def test_registered_under_legacy_name(self, attr, module_name, torch_class_name):
        """The wrapped class keeps its 1.x name and stays in the model registry."""
        import importlib

        from makani.models.physicsnemo_compat import get_model_registry, legacy_registered_name

        ModelRegistry = get_model_registry()

        module = importlib.import_module(module_name)
        wrapped = getattr(module, attr)
        expected = legacy_registered_name(getattr(module, torch_class_name))

        with self.subTest(desc="class name matches the PhysicsNeMo 1.x name"):
            self.assertEqual(wrapped.__name__, expected)

        # Registration is what from_checkpoint resolves against; on 2.x it only
        # happens because the compat helper passes register=True.
        with self.subTest(desc="class is registered"):
            self.assertIn(expected, ModelRegistry().list_models())

    def test_metadata_does_not_set_deprecated_name(self):
        """Constructing the metadata must not emit a DeprecationWarning.

        ``ModelMetaData.name`` is deprecated and inert on 2.x. makani keeps it
        off the dataclasses and applies it via the compat helper instead.
        """
        from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNetMetaData

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SphericalFourierNeuralOperatorNetMetaData()

        offenders = [w for w in caught if issubclass(w.category, DeprecationWarning) and "name" in str(w.message)]
        self.assertEqual(offenders, [], f"metadata set a deprecated field: {[str(w.message) for w in offenders]}")


if __name__ == "__main__":
    unittest.main()
