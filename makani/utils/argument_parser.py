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


def _nonempty_str(value):
    if not value:
        raise argparse.ArgumentTypeError("--run_num must be a non-empty value (e.g. --run_num=0)")
    return value


def get_default_argument_parser(training=True):

    # parser instance
    parser = argparse.ArgumentParser()

    # configuration options
    parser.add_argument("--yaml_config", default="./config/sfnonet.yaml", type=str)
    parser.add_argument("--config", default="base_73chq", type=str)
    parser.add_argument("--run_num", default="00", type=_nonempty_str)

    # hyperparameters override
    parser.add_argument(
        "--batch_size", default=-1, type=int, help="Switch for overriding batch size in the configuration file."
    )

    # model parallelization options
    parser.add_argument("--matmul_parallel_size", default=1, type=int, help="Feature (tensor) parallelism dimension")
    parser.add_argument("--h_parallel_size", default=1, type=int, help="Spatial parallelism dimension in h")
    parser.add_argument("--w_parallel_size", default=1, type=int, help="Spatial parallelism dimension in w")

    # performance options
    parser.add_argument(
        "--amp_mode",
        default="none",
        type=str,
        help="Mixed precision mode: 'none', 'fp16', 'bf16', or a combined '<amp>-<fp8recipe>' such as 'bf16-fp8_delayed' (requires transformer_engine). See makani.utils.precision.",
    )
    parser.add_argument(
        "--jit_mode",
        default="none",
        type=str,
        choices=["none", "inductor"],
        help="Specify if and how to use torch compile.",
    )
    parser.add_argument("--checkpointing_level", default=0, type=int, help="How aggressively checkpointing is used")
    parser.add_argument(
        "--print_timings_frequency", default=-1, type=int, help="Frequency at which to print timing information"
    )
    if training:
        parser.add_argument(
            "--skip_validation",
            action="store_true",
            help="Flag to allow validation skipping, useful for profiling and debugging",
        )
        parser.add_argument("--skip_training", action="store_true", help="Flag to skip training, useful for debugging")
        parser.add_argument(
            "--parameters_reduction_buffer_count",
            default=1,
            type=int,
            help="How many buffers will be used (approximately) for weight gradient reductions.",
        )

    # data options
    parser.add_argument("--enable_synthetic_data", action="store_true", help="Enable to use synthetic data.")
    parser.add_argument(
        "--odirect_config",
        type=str,
        default=None,
        help=(
            "Enable O_DIRECT for Data I/O and set the alignment size. "
            "Accepts a byte count (e.g. '4096'), a kilobyte shorthand (e.g. '4K'), "
            "or a megabyte shorthand (e.g. '1M'). Omit or pass nothing to use the POSIX driver."
        ),
    )
    parser.add_argument(
        "--enable_s3",
        action="store_true",
        help="Enable data loading from AWS S3 buckets. Requires the environment variables AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, AWS_ENDPOINT_URL to be set.",
    )
    parser.add_argument("--split_data_channels", action="store_true")

    # checkpoint format
    if training:
        parser.add_argument(
            "--save_checkpoint",
            default="legacy",
            choices=["none", "flexible", "legacy"],
            type=str,
            help="Format in which to save checkpoints.",
        )
    parser.add_argument(
        "--load_checkpoint",
        default="legacy",
        choices=["flexible", "legacy"],
        type=str,
        help="Format in which to load checkpoints.",
    )

    # multistep stuff
    if training:
        parser.add_argument(
            "--multistep_count",
            default=1,
            type=int,
            help="Number of autoregressive training steps. A value of 1 denotes conventional training",
        )
        parser.add_argument(
            "--multistep_checkpoint",
            action="store_true",
            help="Activation-checkpoint each autoregressive step's model forward to cut the O(n_future) activation memory of backprop-through-time at the cost of one extra forward per step.",
        )

    # debug parameters
    if training:
        parser.add_argument("--disable_ddp", action="store_true")
        parser.add_argument("--enable_grad_anomaly_detection", action="store_true")

    # profiling parameters
    parser.add_argument("--capture_range_start", default=1, type=int, help="Profile range start step")
    parser.add_argument("--capture_range_stop", default=1, type=int, help="Profile range stop step")
    parser.add_argument("--capture_ranks", default=[], type=int, nargs="+", help="Profile ranks from that list")
    parser.add_argument(
        "--capture_mode",
        default="training",
        type=str,
        choices=["training", "validation"],
        help="Specify which phase to capture",
    )
    parser.add_argument(
        "--capture_prefix", default=None, type=str, help="Prefix including full path for profiling files"
    )
    parser.add_argument(
        "--capture_type",
        default="torch",
        type=str,
        choices=["torch", "cupti"],
        help="Type for capturing, either torch internal profiler or cupti API.",
    )

    return parser


def parse_odirect_config(config_str):
    """Parse --odirect_config into (enable_odirect: bool, alignment: int in bytes).

    Accepted formats:
      None / omitted  -> (False, 0)      POSIX driver, no O_DIRECT
      "4096"          -> (True, 4096)    bare byte count
      "4K" / "4k"     -> (True, 4096)   kibibytes
      "1M" / "1m"     -> (True, 1048576) mebibytes
    """
    if config_str is None:
        return False, 0
    s = config_str.strip()
    if s[-1].lower() == "k":
        return True, int(s[:-1]) * 1024
    elif s[-1].lower() == "m":
        return True, int(s[:-1]) * 1024 * 1024
    else:
        try:
            return True, int(s)
        except ValueError:
            raise ValueError(
                f"Invalid --odirect_config value '{config_str}'. "
                "Expected a byte count (e.g. '4096'), a kibibyte value (e.g. '4K'), "
                "or a mebibyte value (e.g. '1M')."
            )


# options of the default parser copied into the parameters as they are; those of training only are
# copied when the parser has them
_COPIED_ARGUMENTS = (
    "matmul_parallel_size",
    "h_parallel_size",
    "w_parallel_size",
    "amp_mode",
    "jit_mode",
    "checkpointing_level",
    "print_timings_frequency",
    "enable_synthetic_data",
    "enable_s3",
    "split_data_channels",
    "load_checkpoint",
)
_COPIED_TRAINING_ARGUMENTS = (
    "skip_validation",
    "skip_training",
    "parameters_reduction_buffer_count",
    "save_checkpoint",
    "multistep_count",
    "multistep_checkpoint",
    "disable_ddp",
    "enable_grad_anomaly_detection",
)


def arguments_to_params(params, args):
    """Copies the options of :func:`get_default_argument_parser` from ``args`` into ``params``.

    Besides the options themselves, sets the decomposition (``model_parallel_sizes`` and names, and
    ``data_parallel_sizes`` and names when the parser has ``--ensemble_parallel_size``), the O_DIRECT
    settings from ``--odirect_config``, and ``n_future`` from ``--multistep_count``. The batch size is
    left to the caller, which splits it over the data-parallel ranks.
    """
    for name in _COPIED_ARGUMENTS:
        params[name] = getattr(args, name)
    for name in _COPIED_TRAINING_ARGUMENTS:
        if hasattr(args, name):
            params[name] = getattr(args, name)

    params["model_parallel_sizes"] = [args.h_parallel_size, args.w_parallel_size, args.matmul_parallel_size]
    params["model_parallel_names"] = ["h", "w", "matmul"]
    if hasattr(args, "ensemble_parallel_size"):
        params["ensemble_parallel_size"] = args.ensemble_parallel_size
        params["data_parallel_sizes"] = [args.ensemble_parallel_size, -1]
        params["data_parallel_names"] = ["ensemble", "batch"]

    params["enable_odirect"], params["odirect_alignment"] = parse_odirect_config(args.odirect_config)

    # n_future counts only the steps after the first
    if hasattr(args, "multistep_count"):
        params["n_future"] = args.multistep_count - 1

    return params
