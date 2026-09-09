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

import os
import sys
import json
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "benchmark"))

from makani.utils import benchmark as benchmark_utils
from makani.utils.parse_dataset_metada import parse_dataset_metadata

from .testutils import get_default_parameters

import sweep  # noqa: E402  (scripts/benchmark is appended to sys.path above)


class TestStepTimer(unittest.TestCase):
    """The timer is on the training hot path, so its disabled state has to be free and exact."""

    def test_disabled_records_nothing(self):
        timer = benchmark_utils.StepTimer(enabled=False)
        for _ in range(4):
            timer.begin_step()
            timer.end_step()

        self.assertEqual(timer.num_steps, 0)
        self.assertEqual(timer.timings_ms(), [])

    def test_records_steps_and_drops_warmup(self):
        timer = benchmark_utils.StepTimer(enabled=True, warmup_steps=2)
        for _ in range(5):
            timer.begin_step()
            timer.end_step()

        self.assertEqual(timer.num_steps, 5)
        self.assertEqual(len(timer.timings_ms()), 3)
        self.assertEqual(len(timer.timings_ms(include_warmup=True)), 5)

    def test_unbalanced_calls_raise(self):
        timer = benchmark_utils.StepTimer(enabled=True)

        timer.begin_step()
        with self.assertRaises(RuntimeError):
            timer.begin_step()

        timer.end_step()
        with self.assertRaises(RuntimeError):
            timer.end_step()

    def test_reset_clears_timings(self):
        timer = benchmark_utils.StepTimer(enabled=True)
        timer.begin_step()
        timer.end_step()
        timer.reset()

        self.assertEqual(timer.num_steps, 0)


class TestSummarize(unittest.TestCase):
    def test_summary_statistics(self):
        stats = benchmark_utils.summarize_timings([10.0, 20.0, 30.0, 40.0])

        self.assertEqual(stats["num_steps"], 4)
        self.assertAlmostEqual(stats["median_ms"], 25.0)
        self.assertAlmostEqual(stats["mean_ms"], 25.0)
        self.assertAlmostEqual(stats["min_ms"], 10.0)
        self.assertAlmostEqual(stats["max_ms"], 40.0)

    def test_empty(self):
        self.assertEqual(benchmark_utils.summarize_timings([]), {"num_steps": 0})


class TestComparabilityHash(unittest.TestCase):
    """The hash decides which runs may share a table, so what it ignores matters as much
    as what it covers."""

    def _params(self):
        params = get_default_parameters()
        params.nettype = "SFNO"
        params.num_layers = 2
        params.img_shape_x = 32
        params.img_shape_y = 64
        params.global_batch_size = 4
        return params

    def test_decomposition_does_not_change_the_hash(self):
        params = self._params()
        params.h_parallel_size = 1
        params.batch_size = 4
        params.world_size = 1
        first = benchmark_utils.hash_dict(benchmark_utils.get_comparability_keys(params))

        params.h_parallel_size = 4
        params.batch_size = 1
        params.world_size = 4
        second = benchmark_utils.hash_dict(benchmark_utils.get_comparability_keys(params))

        self.assertEqual(first, second)

    def test_architecture_changes_the_hash(self):
        params = self._params()
        first = benchmark_utils.hash_dict(benchmark_utils.get_comparability_keys(params))

        params.num_layers = 4
        second = benchmark_utils.hash_dict(benchmark_utils.get_comparability_keys(params))

        self.assertNotEqual(first, second)

    def test_data_source_changes_the_hash(self):
        params = self._params()
        synthetic = benchmark_utils.hash_dict(
            benchmark_utils.get_comparability_keys(params, extra={"__metadata_source": "generated"})
        )
        real = benchmark_utils.hash_dict(
            benchmark_utils.get_comparability_keys(params, extra={"__metadata_source": "provided"})
        )

        self.assertNotEqual(synthetic, real)


class TestGeneratedMetadata(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

        self.params = get_default_parameters()
        self.params.img_shape_x = 32
        self.params.img_shape_y = 64

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_generated_descriptor_parses(self):
        """The generated descriptor has to satisfy the same parser a real dataset does."""
        path = os.path.join(self.tmpdir.name, "metadata.json")
        benchmark_utils.generate_metadata_json(self.params, path, dhours=6)

        with open(path, "r") as f:
            metadata = json.load(f)

        self.assertEqual(metadata["coords"]["channel"], list(self.params.channel_names))
        self.assertEqual(metadata["dhours"], 6)

        self.params.metadata_json_path = path
        params, _ = parse_dataset_metadata(path, params=self.params)

        # coordinates are absent from the descriptor and reconstructed from the grid type
        self.assertEqual(len(params["lat"]), self.params.img_shape_x)
        self.assertEqual(len(params["lon"]), self.params.img_shape_y)
        self.assertEqual(len(params["in_channels"]), len(self.params.channel_names))

    def test_missing_shapes_raise(self):
        params = get_default_parameters()
        with self.assertRaises(ValueError):
            benchmark_utils.generate_metadata_json(params, os.path.join(self.tmpdir.name, "m.json"))


class TestSweepEnumeration(unittest.TestCase):
    def test_all_decompositions_tile_the_world(self):
        decompositions = sweep.enumerate_decompositions(8, ensemble_size=4, global_batch_size=8)

        self.assertTrue(len(decompositions) > 0)
        for dec in decompositions:
            self.assertEqual(dec["h"] * dec["w"] * dec["matmul"] * dec["ensemble"] * dec["batch"], 8)
            self.assertEqual(4 % dec["ensemble"], 0)

    def test_matmul_excluded_by_default(self):
        decompositions = sweep.enumerate_decompositions(8, ensemble_size=4, global_batch_size=8)
        self.assertTrue(all(dec["matmul"] == 1 for dec in decompositions))

        with_matmul = sweep.enumerate_decompositions(8, ensemble_size=4, global_batch_size=8, include_matmul=True)
        self.assertTrue(any(dec["matmul"] > 1 for dec in with_matmul))

    def test_ensemble_must_divide(self):
        # an ensemble of 3 cannot be split across 2 ranks
        decompositions = sweep.enumerate_decompositions(8, ensemble_size=3, global_batch_size=8)
        self.assertTrue(all(dec["ensemble"] in (1, 3) for dec in decompositions))


if __name__ == "__main__":
    unittest.main()
