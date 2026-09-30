#!/usr/bin/env python3

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

"""Print benchmark records as the summary block each run prints when it finishes.

A record carries everything needed to reproduce and compare a run, which makes it long to read
directly. This renders the same block ``makani/benchmark.py`` prints at the end of a run, for
records already written to a results file:

    python scripts/benchmark/show.py benchmark_results/results.jsonl
    python scripts/benchmark/show.py results.jsonl --last 3
    python scripts/benchmark/show.py results.jsonl --config ICON_R02B09approx --mode train
    python scripts/benchmark/show.py results.jsonl --oneline

Use ``report.py`` instead to compare runs against each other in a table; this one is for looking
at runs one at a time.

..note::
    The block layout is duplicated from ``makani.utils.benchmark.format_record`` rather than
    imported, for the same reason ``report.py`` duplicates its device-name handling: importing
    makani pulls in torch, and reading a results file should work on a login node. Keep the two
    in sync when either changes.
"""

import sys
import json
import argparse


def _get(record, *path, default=None):
    """Walk nested keys, tolerating records written by older versions of the driver."""
    node = record
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]

    return default if node is None else node


def _memory_lines(record):
    """Per-GPU memory, in whichever shape the record carries it."""
    memory = _get(record, "memory", default={})
    if not memory:
        return []

    world_size = _get(record, "decomposition", "world_size", default=0)

    # records predating the per-GPU spread carried a single max
    if "max_allocated_gb" in memory:
        line = f"  peak memory         : {memory['max_allocated_gb']:.1f} GiB allocated"
        if memory.get("max_reserved_gb") is not None:
            line += f", {memory['max_reserved_gb']:.1f} GiB reserved"
        return [line]

    lines = []
    for field, label in (
        ("peak_allocated_gb", "peak allocated"),
        ("peak_reserved_gb", "peak reserved"),
        ("device_used_gb", "device used"),
    ):
        stats = memory.get(field)
        if stats:
            lines.append(
                f"  {label:<20}: {stats['max']:.1f} GiB max, {stats['mean']:.1f} GiB mean, "
                f"{stats['min']:.1f} GiB min  (over {world_size} GPUs)"
            )

    return lines


def _device_name(record):
    """What to call the GPU: the label if one was given, else the driver name, else the fingerprint."""
    label = _get(record, "environment", "gpu_label")
    if label:
        return label

    name = _get(record, "environment", "gpu_name")
    if _get(record, "environment", "gpu_name_is_placeholder", default=False):
        return _get(record, "environment", "gpu_descriptor", default=name)

    return name


def format_record(record):
    """The post-run summary block, rendered from a stored record."""
    timing = _get(record, "timing", default={})
    model = _get(record, "model", default={})
    dec = _get(record, "decomposition", default={})
    batching = _get(record, "batching", default={})
    output = _get(record, "output", default={})

    dataset = _get(record, "metadata", "dataset_name")
    parameters = model.get("num_parameters")

    lines = [
        "",
        "=" * 72,
        f"benchmark: {record.get('mode')}  {_get(record, 'config', 'yaml_path')}:"
        f"{_get(record, 'config', 'config_name')}",
        "=" * 72,
        f"  run id              : {record.get('run_id')}",
        f"  when (UTC)          : {_get(record, 'environment', 'timestamp')}",
    ]

    if record.get("run_tag"):
        lines.append(f"  tag                 : {record['run_tag']}")
    if dataset:
        lines.append(f"  dataset             : {dataset}")

    lines += [
        f"  model               : {model.get('nettype')}"
        + (f" ({parameters / 1e6:.1f}M parameters)" if parameters else ""),
        f"  shape               : {model.get('img_shape_x')} x {model.get('img_shape_y')} x "
        f"{model.get('n_in_channels')} in / {model.get('n_out_channels')} out",
        f"  precision           : amp={model.get('amp_mode')}, jit={model.get('jit_mode')}, "
        f"checkpointing={model.get('checkpointing_level')}",
        f"  decomposition       : h={dec.get('h')} w={dec.get('w')} matmul={dec.get('matmul')} "
        f"ensemble={dec.get('ensemble')} batch={dec.get('batch')} (world={dec.get('world_size')})",
        f"  global batch size   : {batching.get('global_batch_size')} (local {batching.get('local_batch_size')})",
        f"  ensemble size       : {batching.get('ensemble_size')} (local {batching.get('local_ensemble_size')})",
        f"  steps               : {timing.get('num_steps')} measured, "
        f"{_get(record, 'benchmark', 'warmup_steps')} warmup",
        f"  step time (median)  : {timing.get('median_ms', float('nan')):.1f} ms",
        f"  step time (p10/p90) : {timing.get('p10_ms', float('nan')):.1f} / "
        f"{timing.get('p90_ms', float('nan')):.1f} ms",
        f"  throughput          : {_get(record, 'throughput', 'samples_per_second', default=float('nan')):.2f}"
        " samples/s",
    ]

    lines += _memory_lines(record)

    if output.get("enabled"):
        written = output.get("bytes_written")
        lines.append(
            f"  output              : {output.get('channels')} channels, "
            + (f"{written / 1024**3:.1f} GiB written, " if written else "")
            + f"finalize {output.get('finalize_seconds', float('nan')):.1f}s"
        )

    lines += [
        f"  hardware            : {_device_name(record)} x{dec.get('world_size')}, "
        f"{_get(record, 'environment', 'cpu_name')}",
        f"  spec                : {dec.get('spec')}",
        f"  comparability hash  : {_get(record, 'config', 'comparability_hash')}",
        "=" * 72,
        "",
    ]

    return "\n".join(lines)


def format_oneline(record):
    timing = _get(record, "timing", default={})

    return (
        f"{_get(record, 'environment', 'timestamp', default=''):<21} "
        f"{_get(record, 'config', 'config_name', default=''):<38} "
        f"{record.get('mode', ''):<10} "
        f"{_get(record, 'decomposition', 'spec', default=''):<22} "
        f"{timing.get('median_ms', float('nan')):9.1f} ms  "
        f"{_device_name(record) or ''}"
    )


def load_records(paths):
    records = []
    for path in paths:
        with open(path, "r") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"warning: skipping {path}:{lineno}: {e}", file=sys.stderr)

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="JSON-lines result file(s) written by makani/benchmark.py")
    parser.add_argument("--last", type=int, default=None, metavar="N", help="Show only the last N records.")
    parser.add_argument("--tag", default=None, help="Only records with this run_tag.")
    parser.add_argument("--config", default=None, help="Only records whose config name contains this string.")
    parser.add_argument("--mode", default=None, choices=["train", "validate", "inference"], help="Only this mode.")
    parser.add_argument("--spec", default=None, help="Only records whose decomposition spec contains this string.")
    parser.add_argument("--oneline", action="store_true", help="One line per record instead of a block.")
    args = parser.parse_args()

    records = load_records(args.results)

    if args.tag is not None:
        records = [r for r in records if r.get("run_tag") == args.tag]
    if args.config is not None:
        records = [r for r in records if args.config in (_get(r, "config", "config_name", default="") or "")]
    if args.mode is not None:
        records = [r for r in records if r.get("mode") == args.mode]
    if args.spec is not None:
        records = [r for r in records if args.spec in (_get(r, "decomposition", "spec", default="") or "")]

    if args.last is not None:
        records = records[-args.last :]

    if not records:
        print("no benchmark records matched", file=sys.stderr)
        return 1

    for record in records:
        print(format_oneline(record) if args.oneline else format_record(record))

    return 0


if __name__ == "__main__":
    sys.exit(main())
