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

"""Step-time benchmark driver for training and inference on synthetic data.

The model comes from a config file and config line, exactly as for ``train.py``
and ``ensemble.py`` -- so anything in ``config/`` can be benchmarked without
touching this file:

    mpirun -np 8 python -u -m makani.benchmark \\
        --yaml_config=config/fourcastnet3.yaml \\
        --config=fcn3_sc2_edim45_layers10_pretrain1 \\
        --mode=train --h_parallel_size=2 --ensemble_parallel_size=2 \\
        --benchmark_steps=20 --benchmark_warmup_steps=5

The decomposition is always an *input*: it is taken from the CLI (or the
matching ``MAKANI_*`` environment variables, so a job script can prescribe it
without rewriting the command line) and never chosen here. What ``comm``
actually built is then recorded alongside the timings, so results from different
launchers remain comparable.

Data is synthetic by default. The dataset descriptor that the synthetic loader
needs is generated from the resolved config rather than checked in, since the
metadata parser validates the config's channel names against the descriptor's
channel list -- see :func:`makani.utils.benchmark.generate_metadata_json`.
"""

import os
import time
import logging
import tempfile

import torch
import torch.distributed as dist

from makani.utils import benchmark as benchmark_utils
from makani.utils import comm
from makani.utils import argument_parser
from makani.utils import logging_utils
from makani.utils.YParams import YParams
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani.utils.profiling import Timer
from makani.models.helpers import count_parameters

from makani import Trainer, EnsembleTrainer


def _env_override(cli_value: int, env_name: str, default: int = 1) -> int:
    """CLI wins when it was given, otherwise fall back to the environment.

    A job script that prescribes the decomposition through the environment
    should not have to also rewrite the command line, but an explicit flag must
    still beat a stale exported variable -- hence the "CLI wins when it differs
    from the default" rule.
    """
    if cli_value != default:
        return cli_value

    env_value = os.environ.get(env_name, None)

    return int(env_value) if env_value is not None else default


def get_benchmark_argument_parser():
    parser = argument_parser.get_default_argument_parser()

    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        choices=["train", "inference"],
        help="Benchmark the training step or the inference (autoregressive rollout) step.",
    )
    parser.add_argument("--ensemble_parallel_size", default=1, type=int, help="Ensemble parallelization")
    parser.add_argument(
        "--ensemble_size",
        default=-1,
        type=int,
        help="Switch for overriding the ensemble size in the configuration file.",
    )
    parser.add_argument("--benchmark_steps", default=20, type=int, help="Number of steps to measure (after warmup).")
    parser.add_argument(
        "--benchmark_warmup_steps",
        default=5,
        type=int,
        help="Number of leading steps to run but discard, covering autotuning, NCCL buffer "
        "allocation, allocator warmup and (with --jit_mode=inductor) compilation.",
    )
    parser.add_argument(
        "--rollout_steps",
        default=-1,
        type=int,
        help="Autoregressive steps per inference step. Defaults to the config's valid_autoreg_steps.",
    )
    parser.add_argument(
        "--output_dir",
        default="./benchmark_results",
        type=str,
        help="Directory for the result file and the per-run provenance files.",
    )
    parser.add_argument(
        "--results_file", default="results.jsonl", type=str, help="JSON-lines file results are appended to."
    )
    parser.add_argument("--run_tag", default=None, type=str, help="Free-form label stored with the run.")
    parser.add_argument(
        "--metadata_json_path",
        default=None,
        type=str,
        help="Dataset descriptor to use. Generated from the config when omitted.",
    )
    parser.add_argument(
        "--dhours",
        default=1,
        type=int,
        help="Sampling frequency declared in the generated dataset descriptor.",
    )
    parser.add_argument(
        "--use_real_data",
        action="store_true",
        help="Benchmark against the dataset in the config instead of synthetic data. Requires "
        "--metadata_json_path and valid data paths in the config.",
    )

    return parser


def setup_params(args):
    """Resolve the config, the decomposition and the benchmark overrides."""
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # decomposition: prescribed externally, never inferred here
    h_parallel_size = _env_override(args.h_parallel_size, "MAKANI_H_PARALLEL_SIZE")
    w_parallel_size = _env_override(args.w_parallel_size, "MAKANI_W_PARALLEL_SIZE")
    matmul_parallel_size = _env_override(args.matmul_parallel_size, "MAKANI_MATMUL_PARALLEL_SIZE")
    ensemble_parallel_size = _env_override(args.ensemble_parallel_size, "MAKANI_ENSEMBLE_PARALLEL_SIZE")

    params["h_parallel_size"] = h_parallel_size
    params["w_parallel_size"] = w_parallel_size
    params["matmul_parallel_size"] = matmul_parallel_size
    params["ensemble_parallel_size"] = ensemble_parallel_size
    params["model_parallel_sizes"] = [h_parallel_size, w_parallel_size, matmul_parallel_size]
    params["model_parallel_names"] = ["h", "w", "matmul"]
    params["data_parallel_sizes"] = [ensemble_parallel_size, -1]
    params["data_parallel_names"] = ["ensemble", "batch"]
    params["parameters_reduction_buffer_count"] = args.parameters_reduction_buffer_count

    # performance knobs, taken straight from the CLI so they show up in the record
    params["amp_mode"] = args.amp_mode
    params["jit_mode"] = args.jit_mode
    params["checkpointing_level"] = args.checkpointing_level
    params["multistep_count"] = args.multistep_count
    params["n_future"] = args.multistep_count - 1
    params["multistep_checkpoint"] = args.multistep_checkpoint
    params["split_data_channels"] = args.split_data_channels
    params["disable_ddp"] = args.disable_ddp
    params["enable_odirect"], params["odirect_alignment"] = argument_parser.parse_odirect_config(args.odirect_config)
    params["enable_s3"] = args.enable_s3

    if args.ensemble_size > 0:
        params["ensemble_size"] = args.ensemble_size

    # overrides the benchmark applies on top of the config. Recorded verbatim in the result,
    # so it is never ambiguous what the harness changed underneath the model config.
    overrides = {
        "enable_synthetic_data": not args.use_real_data,
        "log_to_wandb": False,
        "save_checkpoint": "none",
        "load_checkpoint": "legacy",
        "resuming": False,
        "pretrained": False,
        "skip_validation": True,
        "skip_training": False,
        "print_timings_frequency": -1,
        "dump_weights_and_grads": 0,
        "log_video": 0,
        "max_epochs": 1,
        "benchmark_mode": True,
        "benchmark_warmup_steps": args.benchmark_warmup_steps,
        "benchmark_steps": args.benchmark_steps,
    }

    if args.rollout_steps > 0:
        overrides["valid_autoreg_steps"] = args.rollout_steps

    for key, value in overrides.items():
        params[key] = value

    return params, overrides


def build_trainer(params, world_rank):
    """Pick the trainer that matches the config, rather than assuming one."""
    if params.is_set("ensemble_size") and (params["ensemble_size"] > 1):
        return EnsembleTrainer(params, world_rank), "EnsembleTrainer"

    return Trainer(params, world_rank), "Trainer"


def main():
    parser = get_benchmark_argument_parser()
    args = parser.parse_args()

    params, overrides = setup_params(args)

    # wireup
    with Timer() as timer:
        comm.init(
            model_parallel_sizes=params["model_parallel_sizes"],
            model_parallel_names=params["model_parallel_names"],
            data_parallel_sizes=params["data_parallel_sizes"],
            data_parallel_names=params["data_parallel_names"],
            verbose=False,
        )
    wireup_time = timer.time

    world_rank = comm.get_world_rank()

    # a decomposition that does not tile the world would otherwise produce a healthy-looking
    # run that reports a number for a configuration nobody asked for
    decomposition = benchmark_utils.check_decomposition()

    if torch.cuda.is_available():
        torch.cuda.set_device(comm.get_local_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # batching
    params["world_size"] = comm.get_world_size()
    if args.batch_size > 0:
        params["batch_size"] = args.batch_size
    params["global_batch_size"] = params["batch_size"]
    if params["global_batch_size"] % comm.get_size("batch") != 0:
        raise ValueError(
            f"cannot evenly distribute a global batch size of {params['global_batch_size']} across "
            f"{comm.get_size('batch')} batch-parallel ranks."
        )
    params["batch_size"] = int(params["global_batch_size"] // comm.get_size("batch"))

    # ensemble
    ensemble_size = params["ensemble_size"] if params.is_set("ensemble_size") else 1
    if ensemble_size % comm.get_size("ensemble") != 0:
        raise ValueError(
            f"cannot evenly distribute an ensemble of {ensemble_size} across "
            f"{comm.get_size('ensemble')} ensemble-parallel ranks."
        )
    if (ensemble_size <= 1) and (comm.get_size("ensemble") > 1):
        raise ValueError(
            f"--ensemble_parallel_size={comm.get_size('ensemble')} was requested, but config "
            f"'{args.config}' has no ensemble (ensemble_size={ensemble_size}). Pick an ensemble config, "
            "or set --ensemble_parallel_size=1."
        )
    params["ensemble_size"] = ensemble_size
    params["local_ensemble_size"] = ensemble_size // comm.get_size("ensemble")

    if "optimizer_max_grad_norm" not in params:
        params["optimizer_max_grad_norm"] = 1.0

    # output directory. Kept separate from the training exp_dir: a benchmark writes results,
    # not experiments, and should not land in the middle of a training run's output tree.
    # The run id is broadcast rather than computed per rank: two ranks that call strftime on
    # opposite sides of a second boundary would otherwise disagree about the directory name.
    run_id = "{spec}_{mode}_{stamp}".format(
        spec=decomposition["spec"], mode=args.mode, stamp=time.strftime("%Y%m%d-%H%M%S")
    )
    if dist.is_initialized():
        run_id_buffer = [run_id]
        dist.broadcast_object_list(run_id_buffer, src=0)
        run_id = run_id_buffer[0]

    output_dir = os.path.abspath(args.output_dir)
    exp_dir = os.path.join(output_dir, "runs", run_id)

    # every rank creates it: with a container-local output directory this path exists on one
    # node but not on the others, and a rank that assumes rank 0 made it fails on the far node
    os.makedirs(os.path.join(exp_dir, "training_checkpoints"), exist_ok=True)

    params["exp_dir"] = output_dir
    params["experiment_dir"] = exp_dir
    params["checkpoint_path"] = os.path.join(exp_dir, "training_checkpoints/ckpt_mp{mp_rank}_v{checkpoint_version}.tar")
    params["best_checkpoint_path"] = os.path.join(exp_dir, "training_checkpoints/best_ckpt_mp{mp_rank}.tar")
    params["wandb_dir"] = exp_dir

    # synthetic data lives nowhere, but the loader still wants paths
    if not args.use_real_data:
        params["train_data_path"] = exp_dir
        params["valid_data_path"] = exp_dir
        params["inf_data_path"] = exp_dir

    params["log_to_screen"] = (world_rank == 0) and params.get("log_to_screen", True)
    params["log_to_wandb"] = False

    if world_rank == 0:
        logging_utils.config_logger()
        logging.info(f"communicator wireup took {wireup_time:.2f}s")
        logging.info(f"writing benchmark output to {exp_dir}")

    # dataset descriptor: generated from the config unless one was prescribed
    if args.metadata_json_path is not None:
        metadata_json_path = os.path.abspath(args.metadata_json_path)
        metadata_read_path = metadata_json_path
        metadata_source = "provided"
    else:
        if args.use_real_data:
            raise ValueError("--use_real_data requires --metadata_json_path pointing at the dataset descriptor.")

        metadata_source = "generated"
        metadata_json_path = os.path.join(exp_dir, "benchmark_metadata.json")

        # Every rank generates its own copy, into a node-local path, and reads that. Writing it
        # once on rank 0 and having the others read it back requires the output directory to be
        # on a shared filesystem -- which it is not when the job runs with a container-local
        # workdir, so the ranks on every other node would fail to find it. The content is a pure
        # function of the resolved config, so the copies are identical by construction and no
        # barrier or broadcast is needed. The rank-0 copy in exp_dir is kept for provenance.
        metadata_read_path = os.path.join(
            tempfile.gettempdir(), f"makani-benchmark-metadata-{run_id}-rank{world_rank}.json"
        )
        benchmark_utils.generate_metadata_json(params, metadata_read_path, dhours=args.dhours)

        if world_rank == 0:
            benchmark_utils.generate_metadata_json(params, metadata_json_path, dhours=args.dhours)

    params["metadata_json_path"] = metadata_read_path
    params, _ = parse_dataset_metadata(metadata_read_path, params=params)

    # size the epoch so that a single pass over the loader is exactly the benchmark. The dummy
    # loader yields one batch per "sample" and shards by the full data group, so the step count
    # is n_samples_per_epoch // comm.get_size("data").
    total_steps = args.benchmark_warmup_steps + args.benchmark_steps
    epoch_samples = total_steps * comm.get_size("data")
    params["n_train_samples"] = epoch_samples
    params["n_train_samples_per_epoch"] = epoch_samples
    params["n_eval_samples"] = epoch_samples
    params["n_eval_samples_per_epoch"] = epoch_samples

    # build the trainer and run the measured pass
    if world_rank == 0:
        logging.info(
            "setting up the trainer: dataloaders, then the model, then the optimizer and the "
            f"gradient reduction hooks. At {params['img_shape_x']}x{params['img_shape_y']} the model "
            "step precomputes its spherical harmonic transforms on the host, which dominates setup "
            "and looks like a stall; the phase breakdown below says where the time went."
        )

    trainer, trainer_name = build_trainer(params, world_rank)

    # the trainers time every setup phase but only print the breakdown from train(), which the
    # benchmark bypasses. Printing it here is what tells a long model init apart from a hang.
    trainer._log_timers()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if dist.is_initialized():
        dist.barrier()

    if world_rank == 0:
        logging.info(
            f"running {args.benchmark_warmup_steps} warmup + {args.benchmark_steps} measured " f"{args.mode} steps"
        )

    wall_start = time.perf_counter_ns()
    if args.mode == "train":
        trainer.train_one_epoch()
    else:
        trainer.validate_one_epoch(epoch=0)
    if dist.is_initialized():
        dist.barrier()
    wall_time = (time.perf_counter_ns() - wall_start) * 1e-9

    if world_rank == 0:
        logging.info(f"measured pass finished in {wall_time:.1f}s, collecting timings")

    # collect timings. The step time of the run is the max over ranks, taken per step: a
    # collective step is only done when its slowest participant is done.
    local_timings = trainer.step_timer.timings_ms()
    global_timings = benchmark_utils.reduce_step_timings(local_timings)

    if len(global_timings) == 0:
        raise RuntimeError(
            f"no steps were measured: the loader produced at most {args.benchmark_warmup_steps} steps. "
            "Lower --benchmark_warmup_steps or raise --benchmark_steps."
        )

    timing = benchmark_utils.summarize_timings(global_timings)
    timing_local = benchmark_utils.summarize_timings(local_timings)

    # memory, max over ranks
    memory = benchmark_utils.get_memory_record()
    if memory and dist.is_initialized():
        buf = torch.tensor(
            [memory["max_allocated_gb"], memory["max_reserved_gb"]],
            dtype=torch.float64,
            device=torch.device(f"cuda:{comm.get_local_rank()}"),
        )
        dist.all_reduce(buf, op=dist.ReduceOp.MAX)
        memory = {"max_allocated_gb": buf[0].item(), "max_reserved_gb": buf[1].item()}

    num_parameters, param_bytes, _ = count_parameters(trainer.model, trainer.device)

    # throughput. For inference one step is a full rollout episode, so the per-rollout-step
    # figure is reported alongside it.
    rollout_steps = int(params["valid_autoreg_steps"]) + 1 if args.mode == "inference" else 1
    step_seconds = timing["median_ms"] * 1e-3
    throughput = {
        "samples_per_second": params["global_batch_size"] / step_seconds,
        "ensemble_members_per_second": params["global_batch_size"] * ensemble_size / step_seconds,
        "rollout_steps_per_step": rollout_steps,
        "median_ms_per_rollout_step": timing["median_ms"] / rollout_steps,
    }

    record = {
        "run_id": run_id,
        "run_tag": args.run_tag,
        "mode": args.mode,
        "trainer": trainer_name,
        "benchmark": {
            "warmup_steps": args.benchmark_warmup_steps,
            "requested_steps": args.benchmark_steps,
            "wireup_seconds": wireup_time,
            "wall_seconds": wall_time,
        },
        "config": {
            "yaml_path": os.path.relpath(os.path.abspath(args.yaml_config), os.getcwd()),
            "yaml_abspath": os.path.abspath(args.yaml_config),
            "config_name": args.config,
            "yaml_file_sha256": benchmark_utils.hash_file(args.yaml_config),
            "comparability_hash": None,  # filled in below
            "comparability_keys": None,
            "benchmark_overrides": overrides,
        },
        "model": {
            "nettype": params["nettype"],
            "num_parameters": int(num_parameters),
            "parameter_bytes": int(param_bytes),
            "img_shape_x": params["img_shape_x"],
            "img_shape_y": params["img_shape_y"],
            "n_in_channels": params["N_in_channels"],
            "n_out_channels": params["N_out_channels"],
            "n_history": params["n_history"],
            "n_future": params["n_future"],
            "amp_mode": params["amp_mode"],
            "jit_mode": params["jit_mode"],
            "checkpointing_level": params["checkpointing_level"],
        },
        "decomposition": decomposition,
        "batching": {
            "global_batch_size": params["global_batch_size"],
            "local_batch_size": params["batch_size"],
            "ensemble_size": ensemble_size,
            "local_ensemble_size": params["local_ensemble_size"],
            "effective_samples_per_step": params["batch_size"] * params["local_ensemble_size"],
            "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 1),
            "multistep_count": params["multistep_count"],
        },
        "metadata": {
            "source": metadata_source,
            "path": metadata_json_path,
            "sha256": benchmark_utils.hash_file(metadata_json_path),
            "grid_type": params["data_grid_type"],
            "dhours": params["dhours"],
            "num_channels": len(params["data_channel_names"]),
        },
        "timing": timing,
        "timing_local": timing_local,
        "throughput": throughput,
        "memory": memory,
        "environment": benchmark_utils.get_environment_record(),
    }

    # comparability: the decomposition and the world size are deliberately not part of this,
    # since those are the axes the suite varies. The data source is, since a synthetic-loader
    # step time and a real-IO step time are not the same measurement.
    comparability_keys = benchmark_utils.get_comparability_keys(
        params, extra={"__metadata_source": metadata_source, "__mode": args.mode}
    )
    record["config"]["comparability_keys"] = comparability_keys
    record["config"]["comparability_hash"] = benchmark_utils.hash_dict(comparability_keys)
    record["config"]["resolved_config_sha256"] = benchmark_utils.hash_dict(params.to_dict())

    if world_rank == 0:
        results_path = os.path.join(output_dir, args.results_file)
        benchmark_utils.append_record(results_path, record)

        # full resolved config next to the record, for exact reproduction. Best effort: the
        # measurement is already done and recorded at this point, so a config value the yaml
        # dumper cannot represent must not take the result down with it.
        try:
            params.to_yaml(os.path.join(exp_dir, "params.yaml"), overwrite=True)
        except Exception as err:
            logging.warning(f"could not write the resolved config to {exp_dir}/params.yaml: {err}")

        print(benchmark_utils.format_record(record))
        logging.info(f"appended benchmark record to {results_path}")

    # drop the node-local descriptor copy; the rank-0 copy in exp_dir is the one kept
    if (metadata_source == "generated") and (metadata_read_path != metadata_json_path):
        try:
            os.remove(metadata_read_path)
        except OSError:
            pass

    if dist.is_initialized():
        dist.barrier()

    comm.cleanup()


if __name__ == "__main__":
    main()
