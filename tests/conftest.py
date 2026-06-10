# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Top-level pytest config for the makani test suite.

Forces the multiprocessing start method to ``forkserver`` for the whole session.
The default ``fork`` start method is not safe in our runtime image: the parent
process has the GPU stack (CUDA/XPU, RAPIDS, DALI) loaded, and forking a
``torch.utils.data.DataLoader`` worker out of that state aborts the child with a
fatal ``Aborted`` (e.g. ``torch.xpu.random.manual_seed_all`` on the seed, or
DALI's ``isinstance`` import hook during collate).

``forkserver`` launches a clean, lightweight server process (via a fresh exec,
so it never inherits the parent's CUDA/XPU/DALI state) and forks workers from
*that*. Workers therefore start from a clean state -- the XPU lazy-init is a
from-scratch no-op instead of a bad-fork abort -- while keeping fork's cheap
worker startup. We preload ``torch`` into the server to warm it (importing torch
does not initialize any GPU backend, so the server stays clean). This is in the
same spirit as the DALI dataloader, which already runs its workers with
``py_start_method="spawn"``, and lets us exercise the ``num_data_workers > 0``
case.

Override with ``MAKANI_TEST_START_METHOD`` (e.g. ``spawn`` or ``fork``) to debug.
"""

import os
import multiprocessing as mp


def pytest_configure(config):
    # must run before any worker process is created (DataLoader workers are
    # started during test execution, well after pytest_configure)
    method = os.environ.get("MAKANI_TEST_START_METHOD", "forkserver")
    try:
        mp.set_start_method(method, force=True)
    except (RuntimeError, ValueError):
        # set_start_method raises if the method is unknown on this platform;
        # leave the default in place rather than failing collection
        return

    # warm the forkserver so workers fork from a server that already has torch
    # imported (cheaper worker startup); importing torch does not init CUDA/XPU,
    # so the server stays clean for fork purposes
    if method == "forkserver":
        try:
            mp.set_forkserver_preload(["torch"])
        except (RuntimeError, ValueError, ImportError):
            pass
