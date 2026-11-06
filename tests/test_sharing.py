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

"""The model-parallel annotations of parameters, is_shared_mp, and the reductions of their gradients."""

import os
import sys
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.mpu.helpers import AVG, DEFAULT_SHARING, MODEL_GROUPS, SUM, get_reductions, get_sharing
from makani.mpu.layer_norm import DistributedGeometricInstanceNormS2, DistributedInstanceNorm2d
from makani.mpu.layers import (
    DistributedEncoderDecoder,
    DistributedMatmul,
    DistributedMLP,
    DistributedPatchEmbed,
    MultiHeadAttention,
    StochasticMLP,
)


def _param(sharing=None, sharded=None):
    param = nn.Parameter(torch.zeros(4))
    if sharing is not None:
        param.is_shared_mp = sharing
    if sharded is not None:
        param.sharded_dims_mp = sharded
    return param


class TestGetSharing(unittest.TestCase):

    def test_composite_groups_expand(self):
        self.assertEqual(get_sharing(_param({"spatial": SUM, "matmul": AVG})), {"h": SUM, "w": SUM, "matmul": AVG})
        self.assertEqual(get_sharing(_param({"model": AVG})), {"h": AVG, "w": AVG, "matmul": AVG})
        self.assertEqual(get_sharing(_param({"w": SUM})), {"w": SUM})
        self.assertEqual(get_sharing(_param({})), {})

    def test_without_annotation_the_parameter_is_shared_over_the_model(self):
        self.assertEqual(get_sharing(_param()), get_sharing(_param(DEFAULT_SHARING)))
        self.assertEqual(get_sharing(_param()), {"h": SUM, "w": SUM, "matmul": AVG})

    def test_the_list_form_sums_over_space_and_averages_over_matmul(self):
        self.assertEqual(get_sharing(_param(["spatial"])), {"h": SUM, "w": SUM})
        self.assertEqual(get_sharing(_param(["model"])), {"h": SUM, "w": SUM, "matmul": AVG})
        self.assertEqual(get_sharing(_param(["matmul", "w"])), {"w": SUM, "matmul": AVG})
        self.assertEqual(get_sharing(_param([])), {})

    def test_invalid_annotations_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown model-parallel group"):
            get_sharing(_param({"data": SUM}))
        with self.assertRaisesRegex(ValueError, "Unknown reduction"):
            get_sharing(_param({"matmul": "max"}))
        with self.assertRaisesRegex(ValueError, "Two reductions"):
            get_sharing(_param({"spatial": SUM, "w": AVG}))
        with self.assertRaisesRegex(ValueError, "sharded over"):
            get_sharing(_param({"spatial": SUM}, sharded=["h"]))

    def test_reductions_are_merged_where_they_agree(self):
        self.assertEqual(get_reductions(_param({"model": SUM})), [("model", SUM)])
        self.assertEqual(get_reductions(_param({"spatial": SUM, "matmul": AVG})), [("spatial", SUM), ("matmul", AVG)])
        self.assertEqual(get_reductions(_param({"h": SUM, "w": AVG})), [("h", SUM), ("w", AVG)])
        self.assertEqual(get_reductions(_param({"w": SUM, "matmul": AVG})), [("w", SUM), ("matmul", AVG)])
        self.assertEqual(get_reductions(_param({})), [])


class TestDistributedLayerSharing(unittest.TestCase):
    """
    The annotations of the distributed layers, built here on a single rank, where they are annotated as on many.

    Over each model-parallel group, a parameter is either sharded or shared, never both and never neither; see
    test_models.py for the networks.
    """

    @parameterized.expand(
        [
            ("matmul_column", lambda: DistributedMatmul(8, 16, parallel_mode="column")),
            ("matmul_row", lambda: DistributedMatmul(16, 8, parallel_mode="row")),
            ("mlp", lambda: DistributedMLP(8, 16, 8)),
            ("encoder", lambda: DistributedEncoderDecoder(1, 8, 4, 16, nn.GELU)),
            ("attention", lambda: MultiHeadAttention(16, num_heads=4, qkv_bias=True, qk_norm=True)),
            (
                "patch_embed_split",
                lambda: DistributedPatchEmbed((16, 16), (4, 4), 3, 8, output_is_matmul_parallel=True),
            ),
            ("patch_embed", lambda: DistributedPatchEmbed((16, 16), (4, 4), 3, 8, output_is_matmul_parallel=False)),
            ("stochastic_mlp", lambda: StochasticMLP(8, 16, 8)),
            ("instance_norm", lambda: DistributedInstanceNorm2d(8, affine=True)),
            (
                "geometric_instance_norm",
                lambda: DistributedGeometricInstanceNormS2((16, 32), (16, 32), (0, 0), "equiangular", 8, affine=True),
            ),
        ]
    )
    def test_every_parameter_is_shared_or_sharded_over_each_model_group(self, _, build):
        layer = build()
        params = list(layer.named_parameters())
        self.assertTrue(params)
        for name, param in params:
            with self.subTest(param=name):
                shared = set(get_sharing(param))
                sharded = {group for group in getattr(param, "sharded_dims_mp", []) if group is not None}
                self.assertEqual(shared & sharded, set())
                self.assertEqual(shared | sharded, set(MODEL_GROUPS))

    def test_row_and_column_biases(self):
        # a column bias is split over matmul with the output, a row bias added after the reduction, on every rank alike
        self.assertEqual(get_sharing(DistributedMatmul(8, 16, parallel_mode="column").bias), {"h": SUM, "w": SUM})
        self.assertEqual(
            get_sharing(DistributedMatmul(16, 8, parallel_mode="row").bias), {"h": SUM, "w": SUM, "matmul": AVG}
        )

    def test_attention_norms_are_summed_over_the_heads(self):
        attention = MultiHeadAttention(16, num_heads=4, qk_norm=True)
        for norm in (attention.q_norm, attention.k_norm):
            for param in norm.parameters():
                self.assertEqual(get_sharing(param)["matmul"], SUM)


if __name__ == "__main__":
    unittest.main()
