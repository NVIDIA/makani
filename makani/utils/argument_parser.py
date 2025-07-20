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

import argparse


def get_default_argument_parser(training=True):

    # parser instance
    parser = argparse.ArgumentParser()

    # configuration options
    parser.add_argument("--yaml_config", default="./config/sfnonet.yaml", type=str)
    parser.add_argument("--config", default="base_73chq", type=str)
    parser.add_argument("--run_num", default="00", type=str)

    # hyperparameters override
    parser.add_argument("--batch_size", default=-1, type=int, help="Switch for overriding batch size in the configuration file.")

    # model parallelization options
    parser.add_argument("--fin_parallel_size", default=1, type=int, help="Input feature parallelization")
    parser.add_argument("--fout_parallel_size", default=1, type=int, help="Output feature parallelization")
    parser.add_argument("--h_parallel_size", default=1, type=int, help="Spatial parallelism dimension in h")
    parser.add_argument("--w_parallel_size", default=1, type=int, help="Spatial parallelism dimension in w")

    # performance options
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "fp16", "bf16"], help="Specify the mixed precision mode which should be used.")
    parser.add_argument("--jit_mode", default="none", type=str, choices=["none", "inductor"], help="Specify if and how to use torch compile.")
    parser.add_argument("--checkpointing_level", default=0, type=int, help="How aggressively checkpointing is used")
    parser.add_argument("--print_timings_frequency", default=-1, type=int, help="Frequency at which to print timing information")
    if training:
        parser.add_argument("--skip_validation", action="store_true", help="Flag to allow validation skipping, useful for profiling and debugging")
        parser.add_argument("--skip_training", action="store_true", help="Flag to skip training, useful for debugging")
        parser.add_argument("--parameters_reduction_buffer_count", default=1, type=int, help="How many buffers will be used (approximately) for weight gradient reductions.")

    # data options
    parser.add_argument("--enable_synthetic_data", action="store_true", help="Enable to use synthetic data.")
    parser.add_argument("--enable_odirect", action="store_true", help="Enable to use O_DIRECT for Data I/O.")
    parser.add_argument(
        "--enable_s3",
        action="store_true",
        help="Enable data loading from AWS S3 buckets. Requires the environment variables AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, AWS_ENDPOINT_URL to be set.",
    )
    parser.add_argument("--split_data_channels", action="store_true")

    # checkpoint format
    if training:
        parser.add_argument("--save_checkpoint", default="legacy", choices=["none", "flexible", "legacy"], type=str, help="Format in which to save checkpoints.")
    parser.add_argument("--load_checkpoint", default="legacy", choices=["flexible", "legacy"], type=str, help="Format in which to load checkpoints.")

    # multistep stuff
    if training:
        parser.add_argument("--multistep_count", default=1, type=int, help="Number of autoregressive training steps. A value of 1 denotes conventional training")
    
    # debug parameters
    parser.add_argument("--enable_benchy", action="store_true")
    if training:
        parser.add_argument("--disable_ddp", action="store_true")
        parser.add_argument("--enable_grad_anomaly_detection", action="store_true")

    # profiling parameters
    parser.add_argument("--capture_range_start", default=1, type=int, help="Profile range start step")
    parser.add_argument("--capture_range_stop", default=1, type=int, help="Profile range stop step")
    parser.add_argument("--capture_ranks", default=[], type=int, nargs="+", help="Profile ranks from that list")
    parser.add_argument("--capture_mode", default="training", type=str, choices=["training", "validation"], help="Specify which phase to capture")
    parser.add_argument("--capture_prefix", default=None, type=str, help="Prefix including full path for profiling files")
    parser.add_argument("--capture_type", default="torch", type=str, choices=["torch", "cupti"], help="Type for capturing, either torch internal profiler or cupti API.")

    return parser
