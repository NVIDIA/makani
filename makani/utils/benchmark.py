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

"""Measurement and bookkeeping helpers for the benchmark driver (``makani/benchmark.py``).

Three separate concerns live here:

1. :class:`StepTimer` -- per-step wall time, recorded with CUDA events so the
   measurement is not distorted by the asynchronous launch queue. It is
   constructed (disabled) by every :class:`~makani.utils.driver.Driver`, so the
   trainers can call ``begin_step``/``end_step`` unconditionally and pay nothing
   outside a benchmark run.

2. Provenance -- what was actually run: the decomposition ``comm`` built, the
   config file and config line, the software/hardware environment.

3. Comparability -- a hash over the parts of the resolved configuration that
   have to match before two step times mean the same thing. Deliberately
   *excludes* the decomposition and the world size, since those are the axes a
   benchmark suite varies.
"""

import os
import json
import time
import datetime
import socket
import hashlib
import getpass
import subprocess
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import torch
import torch.distributed as dist

from makani.utils import comm


class StepTimer:
    """Records the wall time of individual training/inference steps.

    On CUDA the timing is taken with events recorded into the stream rather than
    with a host-side clock: a host clock around an asynchronous step measures
    launch time, not execution time, and would only become correct by inserting
    a synchronization per step -- which itself perturbs what is being measured.
    Events are read out once, at collection time, after a single synchronize.

    ``warmup_steps`` leading steps are recorded but dropped from the statistics.
    They cover cudnn/cublas autotuning, NCCL buffer allocation, the allocator
    reaching steady state, and (with ``--jit_mode=inductor``) compilation.

    When ``enabled`` is False every method is a no-op, so the calls can sit
    unconditionally in the training loops.
    """

    def __init__(self, enabled: bool = False, warmup_steps: int = 0, label: str = "step"):
        self.enabled = enabled
        self.warmup_steps = warmup_steps
        self.label = label
        self.use_cuda_events = enabled and torch.cuda.is_available()
        self.reset()

    def reset(self):
        self._events = []
        self._host_times_ms = []
        self._open = None

    @property
    def num_steps(self) -> int:
        return len(self._events) if self.use_cuda_events else len(self._host_times_ms)

    def begin_step(self):
        if not self.enabled:
            return
        if self._open is not None:
            raise RuntimeError("StepTimer.begin_step() called twice without an intervening end_step()")
        if self.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._open = start
        else:
            self._open = time.perf_counter_ns()

    def end_step(self):
        if not self.enabled:
            return
        if self._open is None:
            raise RuntimeError("StepTimer.end_step() called without a matching begin_step()")
        if self.use_cuda_events:
            stop = torch.cuda.Event(enable_timing=True)
            stop.record()
            self._events.append((self._open, stop))
        else:
            self._host_times_ms.append((time.perf_counter_ns() - self._open) * 1e-6)
        self._open = None

    def timings_ms(self, include_warmup: bool = False) -> List[float]:
        """All recorded step times in milliseconds, warmup dropped by default."""
        if self.use_cuda_events:
            if self._events:
                torch.cuda.synchronize()
            times = [start.elapsed_time(stop) for start, stop in self._events]
        else:
            times = list(self._host_times_ms)

        if not include_warmup:
            times = times[self.warmup_steps :]

        return times


def summarize_timings(times_ms: Sequence[float]) -> Dict[str, Any]:
    """Distribution summary of a list of step times.

    The median is the headline number rather than the mean: a single step
    delayed by a stray allocation or a checkpoint of the host process should not
    move the reported figure.
    """
    if len(times_ms) == 0:
        return {"num_steps": 0}

    arr = np.asarray(times_ms, dtype=np.float64)

    return {
        "num_steps": int(arr.size),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
    }


def reduce_step_timings(times_ms: Sequence[float]) -> List[float]:
    """Element-wise max of the per-step times across all ranks.

    A collective step is only finished when its slowest participant is finished,
    so the step time of the run is the max over ranks -- taken per step, not on
    the already-summarized statistics, since which rank straggles can change
    from step to step.

    Ranks are truncated to the smallest common step count first. They should
    agree (the dummy loader shards evenly and the trainers ``drop_last``), but a
    mismatch here would otherwise deadlock rather than report.
    """
    if not dist.is_initialized() or comm.get_world_size() == 1:
        return list(times_ms)

    device = torch.device(f"cuda:{comm.get_local_rank()}") if torch.cuda.is_available() else torch.device("cpu")

    count = torch.tensor([len(times_ms)], dtype=torch.long, device=device)
    dist.all_reduce(count, op=dist.ReduceOp.MIN)
    num_steps = int(count.item())

    if num_steps == 0:
        return []

    local = torch.tensor(list(times_ms)[:num_steps], dtype=torch.float64, device=device)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)

    return local.cpu().tolist()


def get_decomposition_record() -> Dict[str, Any]:
    """The decomposition ``comm`` actually built, not the one that was requested.

    The distinction matters: the data dimensions are auto-sized from ``-1`` in
    :func:`makani.utils.comm.init`, so what a launcher asked for and what the run
    used are not the same thing. Reporting the resolved sizes makes it
    impossible for the logged decomposition to disagree with the measured one.
    """
    sizes = {name: comm.get_size(name) for name in ["h", "w", "matmul", "ensemble", "batch"]}

    spec = "w{world}_h{h}w{w}m{matmul}e{ensemble}b{batch}".format(world=comm.get_world_size(), **sizes)

    record = dict(
        world_size=comm.get_world_size(),
        model=comm.get_size("model"),
        spatial=comm.get_size("spatial"),
        data=comm.get_size("data"),
        spec=spec,
        **sizes,
    )

    return record


def check_decomposition():
    """Fail loudly on a decomposition that does not tile the world.

    A mis-prescribed decomposition otherwise produces a run that looks healthy
    and reports a number for a configuration nobody asked for.
    """
    dec = get_decomposition_record()
    product = dec["h"] * dec["w"] * dec["matmul"] * dec["ensemble"] * dec["batch"]

    if product != dec["world_size"]:
        raise ValueError(
            f"decomposition h={dec['h']} w={dec['w']} matmul={dec['matmul']} ensemble={dec['ensemble']} "
            f"batch={dec['batch']} multiplies out to {product} ranks but the world has {dec['world_size']}."
        )

    return dec


def _run_git(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git"] + args,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if out.returncode != 0:
        return None

    return out.stdout.strip()


def get_environment_record() -> Dict[str, Any]:
    """Software and hardware identity of the run.

    Carried as annotation rather than as part of the comparability hash: a run
    on a different machine is still comparable, it just has to be visible that
    it was one.
    """
    record = {
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        # UTC, so runs from machines in different zones sort and compare directly
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_run_git(["status", "--porcelain", "--untracked-files=no"])),
        # CPU affinity, because a run launched under a pinning wrapper (bindpcie and friends)
        # and one launched without it differ by a real margin in step time -- host-side launch
        # overhead, dataloader workers and NUMA locality all move. Without this, the two are
        # indistinguishable in the results file.
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        record.update(
            {
                "gpu_name": props.name,
                "gpu_memory_gb": round(props.total_memory / 1024**3, 1),
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpus_per_node": torch.cuda.device_count(),
            }
        )
        try:
            nccl_version = torch.cuda.nccl.version()
            record["nccl_version"] = ".".join(str(v) for v in nccl_version)
        except (AttributeError, RuntimeError):
            record["nccl_version"] = None
    else:
        record["gpu_name"] = None

    return record


def get_memory_record() -> Dict[str, Any]:
    """Peak device memory over the run, in GiB."""
    if not torch.cuda.is_available():
        return {}

    return {
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }


# Keys dropped before hashing a configuration for comparability. These are the
# knobs that a benchmark suite is expected to vary between runs (decomposition,
# world size, per-rank batch), plus bookkeeping that says nothing about what was
# computed (paths, logging, checkpoint format, worker counts). Everything else
# -- crucially every architecture key, whatever the nettype calls them -- is
# hashed, so this is a blocklist rather than an allowlist: a new model
# hyperparameter is part of the identity of a run by default.
_COMPARABILITY_EXCLUDED_KEYS = frozenset(
    {
        # decomposition and run geometry
        "batch_size",
        "local_ensemble_size",
        "world_size",
        "h_parallel_size",
        "w_parallel_size",
        "matmul_parallel_size",
        "ensemble_parallel_size",
        "model_parallel_sizes",
        "model_parallel_names",
        "data_parallel_sizes",
        "data_parallel_names",
        "parameters_reduction_buffer_count",
        "io_grid",
        "io_rank",
        "data_shard_id",
        "data_num_shards",
        # run length -- a benchmark caps it, it is not a property of the model
        "max_epochs",
        "n_train_samples",
        "n_train_samples_per_epoch",
        "n_eval_samples",
        "n_eval_samples_per_epoch",
        "num_samples_per_epoch",
        "n_years",
        # paths and dataset bookkeeping
        "exp_dir",
        "experiment_dir",
        "checkpoint_path",
        "best_checkpoint_path",
        "metadata_json_path",
        "train_data_path",
        "valid_data_path",
        "inf_data_path",
        "min_path",
        "max_path",
        "time_means_path",
        "global_means_path",
        "global_stds_path",
        "time_diff_means_path",
        "time_diff_stds_path",
        "orography_path",
        "landmask_path",
        "maskpath",
        "h5_path",
        "dataset",
        "lat",
        "lon",
        "dhours",
        # io and runtime plumbing
        "enable_synthetic_data",
        "enable_s3",
        "enable_odirect",
        "odirect_alignment",
        "multifiles",
        "dataset_backend",
        "num_data_workers",
        "num_visualization_workers",
        "print_timings_frequency",
        "verbose",
        "resuming",
        "pretrained",
        "load_checkpoint",
        "save_checkpoint",
        "checkpoint_num_versions",
        "skip_validation",
        "skip_training",
        "disable_ddp",
        "enable_grad_anomaly_detection",
        "dump_weights_and_grads",
        "split_data_channels",
        "run_num",
        "save_raw_forecasts",
        "save_channel",
        "load_loss",
    }
)

_COMPARABILITY_EXCLUDED_PREFIXES = ("wandb", "log_", "benchmark_", "capture_", "img_crop_")


def _is_comparability_key(key: str) -> bool:
    if key.startswith("_"):
        return False
    if key in _COMPARABILITY_EXCLUDED_KEYS:
        return False
    if key.startswith(_COMPARABILITY_EXCLUDED_PREFIXES):
        return False
    return True


def get_comparability_keys(params, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The subset of the resolved config that two runs must share to be comparable."""
    keys = {k: v for k, v in params.to_dict().items() if _is_comparability_key(k)}

    if extra is not None:
        keys.update(extra)

    return keys


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def hash_dict(obj: Dict[str, Any], length: int = 16) -> str:
    return hashlib.sha256(_stable_json(obj).encode("utf-8")).hexdigest()[:length]


def hash_file(path: str, length: int = 16) -> Optional[str]:
    if (path is None) or (not os.path.isfile(path)):
        return None

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()[:length]


def generate_metadata_json(params, path: str, dhours: int = 1, grid_type: Optional[str] = None) -> str:
    """Write a dataset descriptor for the synthetic loader, derived from the config.

    A checked-in descriptor would have to pin a channel list, and
    :func:`makani.utils.parse_dataset_metada.parse_dataset_metadata` validates
    every name in the config's ``channel_names`` against ``coords.channel`` --
    so a static file would silently restrict the suite to the models whose
    channel naming it happens to match. Deriving it from the resolved config
    instead means any config benchmarks out of the box.

    Coordinates are deliberately omitted: the parser synthesizes them from
    ``coords.grid_type`` and the config's ``img_shape_x``/``img_shape_y``, which
    is the documented path for dummy-data experiments and keeps the resolution
    defined in exactly one place (the model config).
    """
    # The checks below use ``in`` (item access) rather than ``is_set`` (attribute access) on
    # purpose: ``ParamsBase`` does not mirror attribute assignment into its dict, and
    # ``parse_dataset_metadata`` -- the consumer of this descriptor -- indexes ``params`` by
    # item. Validating through the attribute path would let a params object that satisfies this
    # function still fail with a bare KeyError from inside the parser.
    if ("img_shape_x" not in params) or ("img_shape_y" not in params):
        raise ValueError(
            "Generating a synthetic dataset descriptor requires img_shape_x and img_shape_y in the model "
            "config: without coordinates in the descriptor, the grid is reconstructed from the declared "
            "grid type and those shapes. Add them to the config, or pass --metadata_json_path explicitly."
        )

    if ("channel_names" not in params) or (not params["channel_names"]):
        raise ValueError(
            "Generating a synthetic dataset descriptor requires channel_names in the model config, since "
            "the descriptor's channel list is derived from it. Pass --metadata_json_path explicitly to use "
            "a real dataset descriptor instead."
        )

    channel_names = params["channel_names"]

    if grid_type is None:
        grid_type = params.get("data_grid_type", None) or "equiangular"

    metadata = {
        "dataset_name": "makani-benchmark-synthetic",
        "attrs": {"description": "synthetic dataset descriptor generated by makani/benchmark.py"},
        "h5_path": "fields",
        "dims": ["time", "channel", "lat", "lon"],
        "dhours": dhours,
        "coords": {
            "grid_type": grid_type,
            "channel": list(channel_names),
        },
    }

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

    return path


def append_record(path: str, record: Dict[str, Any]):
    """Append one run to a JSON-lines result file, creating it if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def format_record(record: Dict[str, Any]) -> str:
    """Human-readable one-run summary for the log."""
    dec = record["decomposition"]
    batching = record["batching"]
    timing = record["timing"]
    memory = record.get("memory", {})

    lines = [
        "",
        "=" * 72,
        f"benchmark: {record['mode']}  {record['config']['yaml_path']}:{record['config']['config_name']}",
        "=" * 72,
        f"  model               : {record['model']['nettype']} "
        f"({record['model']['num_parameters'] / 1e6:.1f}M parameters)",
        f"  shape               : {record['model']['img_shape_x']} x {record['model']['img_shape_y']} x "
        f"{record['model']['n_in_channels']} in / {record['model']['n_out_channels']} out",
        f"  decomposition       : h={dec['h']} w={dec['w']} matmul={dec['matmul']} "
        f"ensemble={dec['ensemble']} batch={dec['batch']} (world={dec['world_size']})",
        f"  global batch size   : {batching['global_batch_size']} " f"(local {batching['local_batch_size']})",
        f"  ensemble size       : {batching['ensemble_size']} (local {batching['local_ensemble_size']})",
        f"  steps               : {timing['num_steps']} measured, {record['benchmark']['warmup_steps']} warmup",
        f"  step time (median)  : {timing['median_ms']:.1f} ms",
        f"  step time (p10/p90) : {timing['p10_ms']:.1f} / {timing['p90_ms']:.1f} ms",
        f"  throughput          : {record['throughput']['samples_per_second']:.2f} samples/s",
    ]

    if memory:
        lines.append(
            f"  peak memory         : {memory['max_allocated_gb']:.1f} GiB allocated, "
            f"{memory['max_reserved_gb']:.1f} GiB reserved"
        )

    lines += [
        f"  spec                : {dec['spec']}",
        f"  comparability hash  : {record['config']['comparability_hash']}",
        "=" * 72,
        "",
    ]

    return "\n".join(lines)
