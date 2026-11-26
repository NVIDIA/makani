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

"""Aggregate benchmark records into a comparison table.

Runs are grouped by the comparability hash written by ``makani.benchmark``:
two runs land in the same table only when everything that has to match for their
step times to mean the same thing does match. Runs that differ are reported
separately, with the diverging keys named, rather than silently averaged
together -- ``--force`` puts them in one table anyway.

    python scripts/benchmark/report.py benchmark_results/results.jsonl
    python scripts/benchmark/report.py benchmark_results/results.jsonl --csv table.csv

The precision is part of the comparability hash, so runs in bf16 and in FP8 land in separate tables.
``--across-precision`` puts them in one, with a column of the speedup of each run over the run of the same
decomposition in the reference precision, ``--reference-amp`` (bf16 by default):

    python scripts/benchmark/report.py benchmark_results/results.jsonl --across-precision
"""

import os
import csv
import sys
import json
import hashlib
import argparse
from collections import defaultdict

COLUMNS = [
    ("spec", "decomposition"),
    ("h", "h"),
    ("w", "w"),
    ("m", "matmul"),
    ("e", "ens"),
    ("b", "batch"),
    ("local_batch", "local bs"),
    ("local_ens", "local E"),
    ("median_ms", "step [ms]"),
    ("p10_ms", "p10 [ms]"),
    ("p90_ms", "p90 [ms]"),
    ("amp", "amp"),
    ("fp8", "fp8 layers"),
    ("samples_per_s", "samples/s"),
    ("mem_max_gb", "peak max [GiB]"),
    ("mem_mean_gb", "peak mean [GiB]"),
    ("dev_max_gb", "device max [GiB]"),
    ("speedup", "speedup"),
    ("vs_reference", "vs ref amp"),
    ("gpu", "gpu"),
    ("cpu", "cpu"),
]


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


#: Vendor prefixes stripped for display, matching torch-harmonics' benchmarks/run.py::_fmt_arch
#: so a device is named the same in both suites' tables. Duplicated rather than imported: this
#: script reads result files and should stay runnable without makani (and therefore torch).
_ARCH_STRIP = ("NVIDIA ", "AMD ", "Intel ")

#: Mirrors makani.benchmark.helpers._PLACEHOLDER_GPU_NAMES, for records written before
#: gpu_name_is_placeholder existed.
_PLACEHOLDER_GPU_NAMES = ("NVIDIA Graphics Device", "Graphics Device", "NVIDIA Device")


def _fmt_arch(name):
    if not name:
        return name

    for prefix in _ARCH_STRIP:
        if name.startswith(prefix):
            return name[len(prefix) :]

    return name


def _device_columns(environment):
    """Name the hardware the way torch-harmonics' benchmarks do.

    ``gpu_label`` wins when set: pre-release parts report a placeholder through the driver, and
    the label is then the only accurate name for the part.
    """
    gpu = environment.get("gpu_label")

    if not gpu:
        name = environment.get("gpu_name")
        # a driver placeholder names nothing; fall back to the capability/SM/memory fingerprint
        if environment.get("gpu_name_is_placeholder") or name in _PLACEHOLDER_GPU_NAMES:
            gpu = environment.get("gpu_descriptor") or _fmt_arch(name)
        else:
            gpu = _fmt_arch(name)

    return {"gpu": gpu, "cpu": _fmt_arch(environment.get("cpu_name"))}


def _memory_columns(memory):
    """Flatten the per-GPU memory spread into table columns.

    Records written before the memory block gained its min/max/mean shape carried a single
    ``max_allocated_gb``; those are read as the max so old result files still tabulate.
    """
    if not memory:
        return {"mem_max_gb": None, "mem_mean_gb": None, "dev_max_gb": None}

    if "max_allocated_gb" in memory:
        return {"mem_max_gb": round(memory["max_allocated_gb"], 1), "mem_mean_gb": None, "dev_max_gb": None}

    peak = memory.get("peak_allocated_gb") or {}
    device = memory.get("device_used_gb") or {}

    return {
        "mem_max_gb": round(peak["max"], 1) if "max" in peak else None,
        "mem_mean_gb": round(peak["mean"], 1) if "mean" in peak else None,
        "dev_max_gb": round(device["max"], 1) if "max" in device else None,
    }


def _amp_mode(record):
    return (record.get("model") or {}).get("amp_mode") or "none"


def _fp8_column(precision):
    """The pointwise layers that ran in FP8, of all of them, for a run with an FP8 recipe."""
    if not precision or precision.get("fp8_recipe") is None:
        return None

    return f"{precision.get('fp8_layers_run')}/{precision.get('pointwise_layers')}"


def precision_free_hash(record):
    """The comparability hash without the precision, to group runs that differ only in their amp mode."""
    keys = dict(record["config"].get("comparability_keys") or {})
    keys.pop("amp_mode", None)

    return hashlib.sha256(json.dumps(keys, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def record_row(record, baseline_ms=None, reference_ms=None):
    dec = record["decomposition"]
    batching = record["batching"]
    timing = record["timing"]

    row = {
        "spec": dec["spec"],
        "h": dec["h"],
        "w": dec["w"],
        "m": dec["matmul"],
        "e": dec["ensemble"],
        "b": dec["batch"],
        "local_batch": batching["local_batch_size"],
        "local_ens": batching["local_ensemble_size"],
        "median_ms": round(timing["median_ms"], 1),
        "p10_ms": round(timing["p10_ms"], 1),
        "p90_ms": round(timing["p90_ms"], 1),
        "amp": _amp_mode(record),
        "fp8": _fp8_column(record.get("precision")),
        "samples_per_s": round(record["throughput"]["samples_per_second"], 2),
        **_memory_columns(record.get("memory") or {}),
        **_device_columns(record.get("environment") or {}),
    }

    # step time relative to the least-decomposed run in the group. Above 1 means the
    # decomposition made the step faster than the baseline layout did, below 1 that the
    # added communication cost more than the split saved.
    row["speedup"] = round(baseline_ms / timing["median_ms"], 3) if baseline_ms else None

    # step time of the same decomposition in the reference precision over this one, with --across-precision
    row["vs_reference"] = round(reference_ms / timing["median_ms"], 3) if reference_ms else None

    return row


def pick_baseline(records):
    """The least-decomposed run in a group: the reference every other row is scaled against."""

    def parallelism(record):
        dec = record["decomposition"]
        return (dec["h"] * dec["w"] * dec["matmul"] * dec["ensemble"], dec["world_size"])

    return min(records, key=parallelism)


def diverging_keys(records):
    """Which comparability keys differ across a set of records."""
    keys = defaultdict(set)
    for record in records:
        for key, value in (record["config"].get("comparability_keys") or {}).items():
            keys[key].add(json.dumps(value, default=str, sort_keys=True))

    return sorted(k for k, v in keys.items() if len(v) > 1)


def format_markdown(rows):
    headers = [label for _, label in COLUMNS]
    keys = [key for key, _ in COLUMNS]

    widths = [
        max(len(headers[i]), max((len(str(row.get(keys[i], ""))) for row in rows), default=0)) for i in range(len(keys))
    ]

    def fmt(values):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    lines = [fmt(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        lines.append(fmt(["" if row.get(k) is None else row.get(k) for k in keys]))

    return "\n".join(lines)


def describe_group(record, amp=None):
    """The configuration the runs of a group share; ``amp`` names the amp modes of a group across precisions."""
    config = record["config"]
    model = record["model"]

    dataset = (record.get("metadata") or {}).get("dataset_name")
    dataset = f"{dataset}  " if dataset else ""

    return (
        f"{config['yaml_path']}:{config['config_name']}  [{record['mode']}]  {dataset}"
        f"{model['nettype']} {model['img_shape_x']}x{model['img_shape_y']}x{model['n_in_channels']}, "
        f"global batch {record['batching']['global_batch_size']}, ensemble {record['batching']['ensemble_size']}, "
        f"amp={amp or model['amp_mode']}, jit={model['jit_mode']}"
    )


def reference_times(records, reference_amp):
    """The median step time of each decomposition in the reference precision, that of the last run in the result
    files if there are several."""
    times = {}
    for record in records:
        if _amp_mode(record) == reference_amp:
            times[record["decomposition"]["spec"]] = record["timing"]["median_ms"]

    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="JSON-lines result file(s) written by makani.benchmark")
    parser.add_argument("--csv", default=None, help="Also write the rows to this CSV file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Put runs with differing comparability hashes in one table anyway.",
    )
    parser.add_argument(
        "--across-precision",
        action="store_true",
        help="Put runs that differ only in their amp mode in one table, with their speedup over the reference "
        "precision.",
    )
    parser.add_argument(
        "--reference-amp",
        default="bf16",
        help="The amp mode the speedups of --across-precision are taken over, by default bf16.",
    )
    parser.add_argument("--tag", default=None, help="Only include runs with this run_tag.")
    args = parser.parse_args()

    records = load_records(args.results)

    if args.tag is not None:
        records = [r for r in records if r.get("run_tag") == args.tag]

    if not records:
        print("no benchmark records found", file=sys.stderr)
        return 1

    if args.force:
        groups = {"all runs (--force)": records}
    else:
        group_hash = precision_free_hash if args.across_precision else (lambda r: r["config"]["comparability_hash"])
        grouped = defaultdict(list)
        for record in records:
            grouped[group_hash(record)].append(record)
        groups = {h: rs for h, rs in sorted(grouped.items())}

    all_rows = []
    for group_key, group_records in groups.items():
        # the decomposition speedup within each precision, so that it stays a speedup of the decomposition
        baselines = {}
        for amp in {_amp_mode(r) for r in group_records}:
            baselines[amp] = pick_baseline([r for r in group_records if _amp_mode(r) == amp])["timing"]["median_ms"]
        references = reference_times(group_records, args.reference_amp) if args.across_precision else {}

        rows = sorted(
            (record_row(r, baselines[_amp_mode(r)], references.get(r["decomposition"]["spec"])) for r in group_records),
            key=lambda row: (row["h"] * row["w"] * row["m"] * row["e"], row["spec"], row["amp"]),
        )

        print()
        amp = ", ".join(sorted({_amp_mode(r) for r in group_records})) if args.across_precision else None
        print(f"## {describe_group(group_records[0], amp)}")
        print(f"   comparability hash: {group_key}  ({len(group_records)} runs)")
        if args.force:
            divergent = diverging_keys(group_records)
            if divergent:
                print(f"   WARNING: runs are not comparable, these keys differ: {', '.join(divergent)}")
        print()
        print(format_markdown(rows))

        for row in rows:
            row["group"] = group_key
        all_rows += rows

    if len(groups) > 1 and not args.force:
        print()
        print(
            f"note: {len(groups)} groups of mutually comparable runs. Pass --force to tabulate them "
            "together (the diverging keys will be listed)."
        )

    if args.csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group"] + [key for key, _ in COLUMNS])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
