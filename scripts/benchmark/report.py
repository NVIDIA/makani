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

Runs are grouped by the comparability hash written by ``makani/benchmark.py``:
two runs land in the same table only when everything that has to match for their
step times to mean the same thing does match. Runs that differ are reported
separately, with the diverging keys named, rather than silently averaged
together -- ``--force`` puts them in one table anyway.

    python scripts/benchmark/report.py benchmark_results/results.jsonl
    python scripts/benchmark/report.py benchmark_results/results.jsonl --csv table.csv
"""

import os
import csv
import sys
import json
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
    ("samples_per_s", "samples/s"),
    ("memory_gb", "peak [GiB]"),
    ("speedup", "speedup"),
    ("gpu", "gpu"),
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


def record_row(record, baseline_ms=None):
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
        "samples_per_s": round(record["throughput"]["samples_per_second"], 2),
        "memory_gb": round(record["memory"]["max_allocated_gb"], 1) if record.get("memory") else None,
        "gpu": record["environment"].get("gpu_name"),
    }

    # step time relative to the least-decomposed run in the group. Above 1 means the
    # decomposition made the step faster than the baseline layout did, below 1 that the
    # added communication cost more than the split saved.
    row["speedup"] = round(baseline_ms / timing["median_ms"], 3) if baseline_ms else None

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


def describe_group(record):
    config = record["config"]
    model = record["model"]

    dataset = (record.get("metadata") or {}).get("dataset_name")
    dataset = f"{dataset}  " if dataset else ""

    return (
        f"{config['yaml_path']}:{config['config_name']}  [{record['mode']}]  {dataset}"
        f"{model['nettype']} {model['img_shape_x']}x{model['img_shape_y']}x{model['n_in_channels']}, "
        f"global batch {record['batching']['global_batch_size']}, ensemble {record['batching']['ensemble_size']}, "
        f"amp={model['amp_mode']}, jit={model['jit_mode']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="JSON-lines result file(s) written by makani/benchmark.py")
    parser.add_argument("--csv", default=None, help="Also write the rows to this CSV file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Put runs with differing comparability hashes in one table anyway.",
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
        grouped = defaultdict(list)
        for record in records:
            grouped[record["config"]["comparability_hash"]].append(record)
        groups = {h: rs for h, rs in sorted(grouped.items())}

    all_rows = []
    for group_key, group_records in groups.items():
        baseline = pick_baseline(group_records)
        baseline_ms = baseline["timing"]["median_ms"]

        rows = sorted(
            (record_row(r, baseline_ms) for r in group_records),
            key=lambda row: (row["h"] * row["w"] * row["m"] * row["e"], row["spec"]),
        )

        print()
        print(f"## {describe_group(group_records[0])}")
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
