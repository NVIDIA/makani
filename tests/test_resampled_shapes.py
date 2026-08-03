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

import unittest
from parameterized import parameterized

import torch

from makani.utils.YParams import ParamsBase, ensure_resampled_shapes
from makani.models import model_registry

from .testutils import set_seed, get_default_parameters


class TestResampledShapeFallback(unittest.TestCase):
    """``img_shape_{x,y}_resampled`` must be optional outside training.

    The resampled shapes are populated at runtime from the dataset (Driver copies
    them from the dataloader), so they are absent from model packages written
    before resampling existed and from params assembled by external callers such
    as earth2studio, which do not pass input shapes at all.

    Consumers read them unconditionally -- ``model_registry.get_model``,
    ``Preprocessor2D`` and ``ModelWrapper`` -- so without a fallback loading an
    older SFNO package fails with

        AttributeError: 'ParamsBase' object has no attribute 'img_shape_x_resampled'

    Falling back to the unresampled shape is correct in exactly these cases,
    since no resampling took place.
    """

    def setUp(self):
        set_seed(333)

    def _params_without_resampled(self, nettype):
        """Params as an older package / an external caller would supply them."""
        params = get_default_parameters()
        params.nettype = nettype
        params.img_shape_x = 36
        params.img_shape_y = 72
        params.img_local_shape_x = params.img_crop_shape_x = params.img_shape_x
        params.img_local_shape_y = params.img_crop_shape_y = params.img_shape_y
        # deliberately NOT set: img_shape_x_resampled / img_shape_y_resampled
        for key in ("img_shape_x_resampled", "img_shape_y_resampled"):
            if hasattr(params, key):
                delattr(params, key)
            params.params.pop(key, None)
        return params

    # -- the helper itself ---------------------------------------------------

    def test_fills_from_unresampled(self):
        params = self._params_without_resampled("SFNO")
        ensure_resampled_shapes(params)
        self.assertEqual(params.img_shape_x_resampled, 36)
        self.assertEqual(params.img_shape_y_resampled, 72)

    def test_does_not_overwrite_explicit_values(self):
        """A genuinely resampled config must survive untouched."""
        params = self._params_without_resampled("SFNO")
        params.img_shape_x_resampled = 18
        params.img_shape_y_resampled = 36
        ensure_resampled_shapes(params)
        self.assertEqual(params.img_shape_x_resampled, 18)
        self.assertEqual(params.img_shape_y_resampled, 36)

    def test_is_idempotent(self):
        params = self._params_without_resampled("SFNO")
        ensure_resampled_shapes(params)
        ensure_resampled_shapes(params)
        self.assertEqual(params.img_shape_x_resampled, 36)

    def test_none_is_treated_as_absent(self):
        """Driver leaves these as None when no dataset is attached."""
        params = self._params_without_resampled("SFNO")
        params.img_shape_x_resampled = None
        params.img_shape_y_resampled = None
        ensure_resampled_shapes(params)
        self.assertEqual(params.img_shape_x_resampled, 36)
        self.assertEqual(params.img_shape_y_resampled, 72)

    def test_missing_both_raises_clearly(self):
        """With no shape at all, fail with an actionable message rather than an
        AttributeError from deep inside model construction."""
        params = ParamsBase()
        params.update_params({"nettype": "SFNO"})
        with self.assertRaises(AttributeError) as cm:
            ensure_resampled_shapes(params)
        self.assertIn("img_shape_x", str(cm.exception))

    # -- the reported failure ------------------------------------------------

    @parameterized.expand([("SFNO",), ("FNO",), ("FCN3",)])
    def test_get_model_without_resampled_shapes(self, nettype):
        """Regression: this is the exact path that raised for older packages."""
        params = self._params_without_resampled(nettype)
        model = model_registry.get_model(params, multistep=False)

        # the fallback must have populated params for the downstream consumers
        # (Preprocessor2D reads them from this same object)
        self.assertEqual(params.img_shape_x_resampled, params.img_shape_x)
        self.assertEqual(params.img_shape_y_resampled, params.img_shape_y)

        inp = torch.randn(1, params.N_in_channels, params.img_shape_x, params.img_shape_y)
        out = model(inp)
        self.assertEqual(out.shape, (1, params.N_out_channels, params.img_shape_x, params.img_shape_y))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
