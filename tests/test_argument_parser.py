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

from makani.utils.argument_parser import arguments_to_params, get_default_argument_parser
from makani.utils.YParams import ParamsBase


class TestArgumentsToParams(unittest.TestCase):
    """Every option of the default parser reaches the parameters."""

    def _params(self, argv, training=True, ensemble=False):
        parser = get_default_argument_parser(training=training)
        if ensemble:
            parser.add_argument("--ensemble_parallel_size", default=1, type=int)
        return arguments_to_params(ParamsBase(), parser.parse_args(argv))

    def test_performance_and_data_options(self):
        params = self._params(
            [
                "--amp_mode=bf16",
                "--jit_mode=inductor",
                "--checkpointing_level=2",
                "--print_timings_frequency=10",
                "--enable_synthetic_data",
                "--enable_s3",
                "--split_data_channels",
                "--odirect_config=4K",
            ]
        )
        self.assertEqual(params.amp_mode, "bf16")
        self.assertEqual(params.jit_mode, "inductor")
        self.assertEqual(params.checkpointing_level, 2)
        self.assertEqual(params.print_timings_frequency, 10)
        self.assertTrue(params.enable_synthetic_data)
        self.assertTrue(params.enable_s3)
        self.assertTrue(params.split_data_channels)
        self.assertTrue(params.enable_odirect)
        self.assertEqual(params.odirect_alignment, 4096)

    def test_training_options(self):
        params = self._params(
            [
                "--save_checkpoint=flexible",
                "--load_checkpoint=flexible",
                "--multistep_count=3",
                "--multistep_checkpoint",
                "--skip_validation",
                "--disable_ddp",
                "--parameters_reduction_buffer_count=4",
            ]
        )
        self.assertEqual(params.save_checkpoint, "flexible")
        self.assertEqual(params.load_checkpoint, "flexible")
        self.assertEqual(params.multistep_count, 3)
        self.assertEqual(params.n_future, 2)
        self.assertTrue(params.multistep_checkpoint)
        self.assertTrue(params.skip_validation)
        self.assertFalse(params.skip_training)
        self.assertTrue(params.disable_ddp)
        self.assertFalse(params.enable_grad_anomaly_detection)
        self.assertEqual(params.parameters_reduction_buffer_count, 4)

    def test_decomposition(self):
        params = self._params(["--h_parallel_size=2", "--w_parallel_size=3", "--matmul_parallel_size=4"])
        self.assertEqual(params.model_parallel_sizes, [2, 3, 4])
        self.assertEqual(params.model_parallel_names, ["h", "w", "matmul"])
        self.assertFalse(params.is_set("data_parallel_sizes"))

        params = self._params(["--ensemble_parallel_size=2"], ensemble=True)
        self.assertEqual(params.ensemble_parallel_size, 2)
        self.assertEqual(params.data_parallel_sizes, [2, -1])
        self.assertEqual(params.data_parallel_names, ["ensemble", "batch"])

    def test_parser_without_training_options(self):
        params = self._params(["--jit_mode=inductor"], training=False)
        self.assertEqual(params.jit_mode, "inductor")
        self.assertFalse(params.is_set("save_checkpoint"))
        self.assertFalse(params.is_set("n_future"))


if __name__ == "__main__":
    unittest.main()
