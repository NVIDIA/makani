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
from parameterized import parameterized

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "benchmark"))

from makani.benchmark import helpers as benchmark_utils
from makani.utils.profiling import StepTimer
from makani.data.parse_dataset_metadata import parse_dataset_metadata

from makani.models import model_registry
from makani.models.common.pointwise_conv import PointwiseConv2d
from makani.utils.precision import AutocastManager, fp8_disabled

from .testutils import get_default_parameters, single_step_rollout
from .test_fp8 import requires_fp8, _model_params

import sweep  # noqa: E402  (scripts/benchmark is appended to sys.path above)
import report  # noqa: E402


class TestStepTimer(unittest.TestCase):
    """The timer is on the training hot path, so its disabled state has to be free and exact."""

    def test_disabled_records_nothing(self):
        timer = StepTimer(enabled=False)
        for _ in range(4):
            timer.begin_step()
            timer.end_step()

        self.assertEqual(timer.num_steps, 0)
        self.assertEqual(timer.timings_ms(), [])

    def test_records_steps_and_drops_warmup(self):
        timer = StepTimer(enabled=True, warmup_steps=2)
        for _ in range(5):
            timer.begin_step()
            timer.end_step()

        self.assertEqual(timer.num_steps, 5)
        self.assertEqual(len(timer.timings_ms()), 3)
        self.assertEqual(len(timer.timings_ms(include_warmup=True)), 5)

    def test_unbalanced_calls_raise(self):
        timer = StepTimer(enabled=True)

        timer.begin_step()
        with self.assertRaises(RuntimeError):
            timer.begin_step()

        timer.end_step()
        with self.assertRaises(RuntimeError):
            timer.end_step()

    def test_reset_clears_timings(self):
        timer = StepTimer(enabled=True)
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

        # item assignment, not attribute assignment: ParamsBase keeps the two apart and
        # parse_dataset_metadata indexes params by item, which is also how a config read
        # from yaml populates it. get_default_parameters() sets these as attributes only.
        self.params = get_default_parameters()
        self.params["img_shape_x"] = 32
        self.params["img_shape_y"] = 64
        self.params["channel_names"] = list(self.params.channel_names)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_generated_descriptor_parses(self):
        """The generated descriptor has to satisfy the same parser a real dataset does."""
        path = os.path.join(self.tmpdir.name, "metadata.json")
        benchmark_utils.generate_metadata_json(self.params, path, dhours=6)

        with open(path, "r") as f:
            metadata = json.load(f)

        self.assertEqual(metadata["coords"]["channel"], list(self.params["channel_names"]))
        self.assertEqual(metadata["dhours"], 6)

        self.params["metadata_json_path"] = path
        params, _ = parse_dataset_metadata(path, params=self.params)

        # coordinates are absent from the descriptor and reconstructed from the grid type
        self.assertEqual(len(params["lat"]), self.params["img_shape_x"])
        self.assertEqual(len(params["lon"]), self.params["img_shape_y"])
        self.assertEqual(len(params["in_channels"]), len(self.params["channel_names"]))

    def test_missing_shapes_raise(self):
        """The shape check has to fire on the access path the parser uses.

        get_default_parameters() sets img_shape_x as an attribute, so a check written against
        the attribute path would pass here and then fail with a KeyError inside the parser.
        """
        params = get_default_parameters()
        params["channel_names"] = list(params.channel_names)

        with self.assertRaises(ValueError):
            benchmark_utils.generate_metadata_json(params, os.path.join(self.tmpdir.name, "m.json"))

    def test_missing_channel_names_raise(self):
        params = get_default_parameters()
        params["img_shape_x"] = 32
        params["img_shape_y"] = 64

        with self.assertRaises(ValueError):
            benchmark_utils.generate_metadata_json(params, os.path.join(self.tmpdir.name, "m.json"))


class _Engine(nn.Module):
    def forward(self, x, weight):
        return x


class _Layer(nn.Module):
    """A pointwise layer as the benchmark sees it: an fp8_gemm attribute, an engine or None."""

    def __init__(self, engine, run_engine=True):
        super().__init__()
        self.fp8_gemm = engine
        self.run_engine = run_engine

    def forward(self, x):
        # an engine that does not run, as in an fp8_disabled region or on an ineligible input
        return self.fp8_gemm(x, None) if (self.fp8_gemm is not None and self.run_engine) else x


class TestPrecision(unittest.TestCase):
    def test_fp8_layer_usage_counts_the_engines_that_run(self):
        model = nn.Sequential(_Layer(_Engine()), _Layer(_Engine(), run_engine=False), _Layer(None), nn.Identity())
        usage = benchmark_utils.FP8LayerUsage(model)

        x = torch.zeros(1)
        for _ in range(3):
            model(x)

        self.assertEqual(usage.record(), {"pointwise_layers": 3, "fp8_layers": 2, "fp8_layers_run": 1})
        # the hooks remove themselves
        self.assertTrue(all(len(layer.fp8_gemm._forward_hooks) == 0 for layer in model[:2]))

    def test_precision_record_names_the_recipe(self):
        record = benchmark_utils.get_precision_record("bf16-fp8_delayed", {"fp8_layers_run": 4})
        self.assertEqual(record, {"amp_mode": "bf16-fp8_delayed", "fp8_recipe": "fp8_delayed", "fp8_layers_run": 4})

        self.assertIsNone(benchmark_utils.get_precision_record("bf16")["fp8_recipe"])
        self.assertEqual(benchmark_utils.get_precision_record(None)["amp_mode"], "none")

    def test_precision_changes_the_hash(self):
        params = get_default_parameters()
        params.amp_mode = "bf16"
        bf16 = benchmark_utils.hash_dict(benchmark_utils.get_comparability_keys(params))
        params.amp_mode = "bf16-fp8_delayed"
        fp8 = benchmark_utils.hash_dict(benchmark_utils.get_comparability_keys(params))

        self.assertNotEqual(bf16, fp8)


@requires_fp8
class TestPrecisionInFP8(unittest.TestCase):
    """The counts on FP8 layers of transformer_engine, which need FP8 hardware."""

    def setUp(self):
        self.device = torch.device("cuda")

    @parameterized.expand([["bf16-fp8_delayed"], ["bf16-fp8_current"]], skip_on_empty=True)
    def test_fp8_layer_usage_counts_the_layers_in_fp8(self, amp_mode):
        """Of three layers with an engine, one runs in FP8: one runs in fp8_disabled, one on an input on the CPU,
        which is not eligible; a layer with 24 channels has no engine."""
        fp8, disabled, cpu = [PointwiseConv2d(32, 32, 1, fp8=True) for _ in range(3)]
        no_engine = PointwiseConv2d(24, 32, 1, fp8=True)
        model = nn.ModuleList([fp8, disabled, cpu, no_engine])
        fp8.to(self.device), disabled.to(self.device), no_engine.to(self.device)
        self.assertIsNone(no_engine.fp8_gemm)

        usage = benchmark_utils.FP8LayerUsage(model)
        x = torch.randn(2, 32, 16, 16, device=self.device)
        with AutocastManager(amp_mode)():
            fp8(x)
            with fp8_disabled():
                disabled(x)
            cpu(x.cpu())
            no_engine(torch.randn(2, 24, 16, 16, device=self.device))

        self.assertEqual(usage.record(), {"pointwise_layers": 4, "fp8_layers": 3, "fp8_layers_run": 1})

    @parameterized.expand([["bf16-fp8_delayed"], ["bf16-fp8_current"]], skip_on_empty=True)
    def test_model_runs_all_its_engines_in_fp8(self, amp_mode):
        """A training step of SFNO runs every engine in FP8; the encoder and decoder layers have none."""
        params = _model_params(amp_mode)
        model = model_registry.get_model(params).to(self.device)
        usage = benchmark_utils.FP8LayerUsage(model)

        x = torch.randn(2, params.N_in_channels, 36, 72, device=self.device)
        with AutocastManager(amp_mode)():
            out = model(single_step_rollout(x.unsqueeze(1)))
        out.float().square().mean().backward()

        record = benchmark_utils.get_precision_record(amp_mode, usage.record())
        self.assertGreater(record["fp8_layers"], 0)
        self.assertEqual(record["fp8_layers_run"], record["fp8_layers"])
        self.assertGreater(record["pointwise_layers"], record["fp8_layers"])
        self.assertEqual(record["fp8_recipe"], amp_mode.split("-")[1])


class TestReportAcrossPrecision(unittest.TestCase):
    @staticmethod
    def _record(amp_mode, spec, median_ms):
        return {
            "config": {"comparability_keys": {"nettype": "SFNO", "amp_mode": amp_mode}},
            "model": {"amp_mode": amp_mode},
            "precision": {"amp_mode": amp_mode, "fp8_recipe": "fp8_delayed" if "fp8" in amp_mode else None},
            "decomposition": {"spec": spec, "h": 1, "w": 1, "matmul": 1, "ensemble": 1, "batch": 1},
            "batching": {"local_batch_size": 1, "local_ensemble_size": 1},
            "timing": {"median_ms": median_ms, "p10_ms": median_ms, "p90_ms": median_ms},
            "throughput": {"samples_per_second": 1000.0 / median_ms},
        }

    def test_runs_differing_in_precision_share_a_group(self):
        bf16, fp8 = self._record("bf16", "a", 10.0), self._record("bf16-fp8_delayed", "a", 8.0)
        self.assertEqual(report.precision_free_hash(bf16), report.precision_free_hash(fp8))

        other = self._record("bf16", "a", 10.0)
        other["config"]["comparability_keys"]["nettype"] = "FCN3"
        self.assertNotEqual(report.precision_free_hash(bf16), report.precision_free_hash(other))

    def test_speedup_over_the_reference_precision(self):
        records = [self._record("bf16", "a", 10.0), self._record("bf16-fp8_delayed", "a", 8.0)]
        references = report.reference_times(records, "bf16")

        rows = [report.record_row(r, reference_ms=references.get(r["decomposition"]["spec"])) for r in records]
        self.assertEqual([row["vs_reference"] for row in rows], [1.0, 1.25])
        self.assertEqual([row["amp"] for row in rows], ["bf16", "bf16-fp8_delayed"])


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
