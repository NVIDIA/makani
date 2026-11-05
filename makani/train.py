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
import torch
import logging
from functools import partial

# utilities
from makani.utils.profiling import Timer
from makani.utils.checkpoint_helpers import setup_training_run
from makani.utils import logging_utils
from makani.utils.YParams import YParams

# distributed computing stuff
from makani.utils import comm
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani.utils import argument_parser
from makani.utils.argument_parser import arguments_to_params
from makani.utils import profiling

# import trainer
from makani import Trainer

if __name__ == "__main__":
    parser = argument_parser.get_default_argument_parser()
    parser.add_argument(
        "--mode", default="train", type=str, choices=["train", "test"], help="Run training or perform a test"
    )
    # parse
    args = parser.parse_args()

    # parse parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # copy the command line options into the parameters
    arguments_to_params(params, args)

    # make sure to reconfigure logger after the pytorch distributed init
    with Timer() as timer:
        comm.init(
            model_parallel_sizes=params["model_parallel_sizes"],
            model_parallel_names=params["model_parallel_names"],
            verbose=False,
        )
    world_rank = comm.get_world_rank()
    if world_rank == 0:
        print(f"Communicators wireup time: {timer.time:.2f}s")

    # update parameters
    params["world_size"] = comm.get_world_size()
    if args.batch_size > 0:
        params.batch_size = args.batch_size
    params["global_batch_size"] = params.batch_size
    if params["global_batch_size"] % comm.get_size("data") != 0:
        raise ValueError(f"cannot evenly distribute {params['global_batch_size']} across {comm.get_size('data')} GPU.")
    params["batch_size"] = int(params["global_batch_size"] // comm.get_size("data"))

    # optimizer params
    if "optimizer_max_grad_norm" not in params:
        params["optimizer_max_grad_norm"] = 1.0

    # set device
    if torch.cuda.is_available():
        torch.cuda.set_device(comm.get_local_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # debug parameters
    if args.enable_grad_anomaly_detection:
        torch.autograd.set_detect_anomaly(True)

    # set up the run directory, the checkpoint paths, and whether the run resumes
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    setup_training_run(params, expDir)

    # set default wandb dir
    if not hasattr(params, "wandb_dir") or params["wandb_dir"] is None:
        params["wandb_dir"] = expDir

    if world_rank == 0:
        logging_utils.config_logger()
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, "out.log"))
        logging_utils.log_versions()
        params.log(logging.getLogger())

    # determine logging
    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    # parse dataset metadata
    if "metadata_json_path" in params:
        params, _ = parse_dataset_metadata(params["metadata_json_path"], params=params)
    else:
        raise RuntimeError("Error, please specify a dataset descriptor file in json format")

    # instantiate trainer
    trainer = Trainer(params, world_rank)
    # if profiling is enabled, use context manager here:
    if world_rank in args.capture_ranks:
        if args.capture_type == "torch":
            capture_prefix = f"{args.capture_prefix}_rank{world_rank}" if args.capture_prefix is not None else None
            trace_handler = partial(profiling.trace_handler, print_stats=True, export_trace_prefix=capture_prefix)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=args.capture_range_start - 1,
                    warmup=1,
                    active=args.capture_range_stop - args.capture_range_start,
                    repeat=1,
                ),
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                if args.capture_mode == "training":
                    trainer.train(training_profiler=profiler)
                elif args.capture_mode == "validation":
                    trainer.train(validation_profiler=profiler)

        elif args.capture_type == "cupti":
            with profiling.CUDAProfiler(
                capture_range_start=args.capture_range_start, capture_range_stop=args.capture_range_stop, enabled=True
            ) as profiler:
                with torch.autograd.profiler.emit_nvtx(enabled=True, record_shapes=False):
                    if args.capture_mode == "training":
                        trainer.train(training_profiler=profiler)
                    elif args.capture_mode == "validation":
                        trainer.train(validation_profiler=profiler)
    else:
        trainer.train()

    # cleanup
    comm.cleanup()
