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

"""Emit benchmark launch commands for a set of decompositions.

This is a convenience generator, not a runner: it prints commands and exits, so
the decomposition still comes from whatever launches the runs. Pipe it to a
shell, paste the lines into a job script, or ignore it entirely and prescribe
the decomposition yourself -- ``makani/benchmark.py`` takes it from the CLI or
from the ``MAKANI_*_PARALLEL_SIZE`` environment variables either way.

    python scripts/benchmark/sweep.py --gpus 8 --ensemble-size 4 \\
        --yaml-config config/fourcastnet3.yaml --config fcn3_sc2_edim45_layers10_pretrain1
"""

import sys
import shlex
import argparse


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def enumerate_decompositions(gpus, ensemble_size, global_batch_size, include_matmul=False, max_spatial=None):
    """All decompositions of ``gpus`` ranks that the model and the batch can actually take."""
    decompositions = []

    matmul_choices = divisors(gpus) if include_matmul else [1]

    for h in divisors(gpus):
        if (max_spatial is not None) and (h > max_spatial):
            continue
        for w in divisors(gpus // h):
            if (max_spatial is not None) and (w > max_spatial):
                continue
            for m in matmul_choices:
                if (gpus // (h * w)) % m != 0:
                    continue
                for e in divisors(gpus // (h * w * m)):
                    # the ensemble has to split evenly across the ensemble-parallel ranks
                    if ensemble_size % e != 0:
                        continue
                    batch = gpus // (h * w * m * e)
                    # and the global batch across the batch-parallel ranks
                    if global_batch_size % batch != 0:
                        continue
                    decompositions.append(dict(h=h, w=w, matmul=m, ensemble=e, batch=batch))

    return decompositions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", type=int, required=True, help="Number of ranks the runs will use.")
    parser.add_argument("--yaml-config", required=True, help="Model config file.")
    parser.add_argument("--config", required=True, help="Config line within the config file.")
    parser.add_argument("--mode", default="train", choices=["train", "inference"])
    parser.add_argument("--ensemble-size", type=int, default=4, help="Total ensemble members.")
    parser.add_argument("--global-batch-size", type=int, default=None, help="Overrides the config's batch size.")
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--benchmark-warmup-steps", type=int, default=5)
    parser.add_argument("--amp-mode", default=None)
    parser.add_argument("--jit-mode", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--max-spatial",
        type=int,
        default=None,
        help="Cap on the h and w parallel sizes, e.g. to stay within a node.",
    )
    parser.add_argument(
        "--include-matmul",
        action="store_true",
        help="Include matmul (feature) parallelism. Off by default: the gradient reduction hooks "
        "still mix up SUM and AVG for matmul-parallel weights, so those runs would be timing a "
        "configuration that does not train correctly.",
    )
    parser.add_argument(
        "--launcher",
        default="mpirun -np {gpus} --allow-run-as-root",
        help="Launcher prefix. '{gpus}' is substituted.",
    )
    args = parser.parse_args()

    # the global batch has to be known to filter decompositions the batch cannot fill
    global_batch_size = args.global_batch_size if args.global_batch_size is not None else args.gpus

    decompositions = enumerate_decompositions(
        args.gpus,
        args.ensemble_size,
        global_batch_size,
        include_matmul=args.include_matmul,
        max_spatial=args.max_spatial,
    )

    if not decompositions:
        print(
            f"no valid decomposition of {args.gpus} ranks for ensemble_size={args.ensemble_size} "
            f"and global batch size {global_batch_size}",
            file=sys.stderr,
        )
        return 1

    launcher = args.launcher.format(gpus=args.gpus)

    for dec in decompositions:
        cmd = [
            *shlex.split(launcher),
            "python",
            "-u",
            "makani/benchmark.py",
            f"--yaml_config={args.yaml_config}",
            f"--config={args.config}",
            f"--mode={args.mode}",
            f"--h_parallel_size={dec['h']}",
            f"--w_parallel_size={dec['w']}",
            f"--matmul_parallel_size={dec['matmul']}",
            f"--ensemble_parallel_size={dec['ensemble']}",
            f"--ensemble_size={args.ensemble_size}",
            f"--batch_size={global_batch_size}",
            f"--benchmark_steps={args.benchmark_steps}",
            f"--benchmark_warmup_steps={args.benchmark_warmup_steps}",
        ]

        if args.amp_mode is not None:
            cmd.append(f"--amp_mode={args.amp_mode}")
        if args.jit_mode is not None:
            cmd.append(f"--jit_mode={args.jit_mode}")
        if args.run_tag is not None:
            cmd.append(f"--run_tag={args.run_tag}")
        if args.output_dir is not None:
            cmd.append(f"--output_dir={args.output_dir}")

        print(shlex.join(cmd))

    return 0


if __name__ == "__main__":
    sys.exit(main())
