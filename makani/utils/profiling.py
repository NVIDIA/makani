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

import sys
import time
import torch


def trace_handler(prof, print_stats=True, export_trace_prefix=None):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    if export_trace_prefix is not None:
        prof.export_chrome_trace(export_trace_prefix + "_" + str(prof.step_num) + ".json")

    return


class Timer(object):
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = time.perf_counter() - self.start


class CUDAProfiler(object):

    def __init__(self, capture_range_start, capture_range_stop, exit_on_stop=True, enabled=True):
        self.enabled = enabled
        self.entered = False
        self.profiling_started = False
        self.exit_on_stop = exit_on_stop
        self.start = capture_range_start
        self.stop = capture_range_stop
        if self.stop <= self.start:
            raise ValueError("The capture range has to satisfy capture_range_start < capture_range_stop.")
        self._step = 0

        if self.enabled:
            from ctypes import cdll

            self.libcudart = cdll.LoadLibrary("libcudart.so")

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("CUDA context manager is not reentrant")
        self.entered = True
        torch.cuda.synchronize()
        return self

    def step(self):
        if not self.enabled:
            return

        self._step += 1
        if (self._step >= self.start) and (not self.profiling_started):
            torch.cuda.synchronize()
            self.libcudart.cudaProfilerStart()
            self.profiling_started = True

        if (self._step >= self.stop) and (self.profiling_started):
            torch.cuda.synchronize()
            self.libcudart.cudaProfilerStop()
            self.profiling_started = False
            if self.exit_on_stop:
                sys.exit(0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        torch.cuda.synchronize()
        if self.profiling_started:
            libcudart.cudaProfilerStop()
            self.profiling_started = False

        return False
