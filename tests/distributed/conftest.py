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
Distributed-test pytest config.

Suppresses pytest terminal output on non-zero ranks so that the combined
mpirun/srun stdout shows a single coherent test summary instead of N
interleaved copies. The hook fires before each test (after ``setUpClass``
has run, so ``comm`` is already initialized via ``_init_grid``), with a
guard flag so the silencing happens at most once per session.

Hooking at ``pytest_runtest_setup`` (rather than ``pytest_configure``) is
intentional: it leaves the terminal reporter active during ``setUpClass``,
so any error in distributed bootstrap (e.g. a comm-init failure) surfaces
on every rank where it happens, not just rank 0.

Escape hatch: setting ``MAKANI_TEST_NO_SILENCE=1`` in the environment
disables the silencing entirely — useful when debugging a per-rank
failure that rank 0 doesn't reproduce.
"""

import os

# Module-level flag: ensures we only attempt the silencing once.
_silenced = False


def _get_world_rank():
    """Return the world rank, preferring ``comm`` if initialized, else env vars."""
    try:
        from makani.utils import comm
        return comm.get_world_rank()
    except Exception:
        # Fallback: read from the launcher's env vars before any Python
        # comm setup has happened. This branch should rarely fire in
        # practice (distributed tests use _init_grid in setUpClass).
        for key in ("OMPI_COMM_WORLD_RANK", "RANK", "SLURM_PROCID", "PMI_RANK"):
            val = os.environ.get(key)
            if val is not None:
                return int(val)
        return 0


def pytest_runtest_setup(item):
    """Silence pytest output on non-rank-0 processes.

    Runs once per session: the first test's setup triggers this hook, by
    which point ``setUpClass`` has already initialized ``comm``. Subsequent
    tests are no-ops thanks to the ``_silenced`` flag.
    """
    global _silenced
    if _silenced:
        return

    # Honor an env-var escape hatch: forces full output on every rank.
    if os.environ.get("MAKANI_TEST_NO_SILENCE"):
        _silenced = True   # mark so we don't re-check on every test
        return

    rank = _get_world_rank()
    if rank != 0:
        terminal_reporter = item.config.pluginmanager.get_plugin("terminalreporter")
        if terminal_reporter is not None:
            item.config.pluginmanager.unregister(terminal_reporter)

    _silenced = True


def _grid():
    """The distributed decomposition from the launcher env (default 1 x 1 x 1)."""
    return (
        int(os.environ.get("GRID_H", 1)),
        int(os.environ.get("GRID_W", 1)),
        int(os.environ.get("GRID_E", 1)),
    )


def pytest_report_header(config):
    """Surface the GRID_H x GRID_W x GRID_E decomposition at the top of the session.

    Gated on rank 0: unlike a configure-time silencing, the reporter on non-zero
    ranks is only unregistered later (in ``pytest_runtest_setup``), so without this
    gate every rank would print the header. Returns ``None`` for a plain 1x1x1 run.
    """
    h, w, e = _grid()
    if h == 1 and w == 1 and e == 1:
        return None
    if _get_world_rank() != 0:
        return None
    return f"distributed grid: GRID_H x GRID_W x GRID_E = {h} x {w} x {e}"


def pytest_collection_modifyitems(config, items):
    """Append the grid to each distributed test's node id (e.g. ``...[h=2,w=4]``) so
    the decomposition is visible alongside every test name in ``-v`` output. The
    ``e`` component is only shown when ensemble parallelism is enabled (GRID_E > 1).
    """
    h, w, e = _grid()
    if h == 1 and w == 1 and e == 1:
        return
    suffix = f"[h={h},w={w}]" if e == 1 else f"[h={h},w={w},e={e}]"
    for item in items:
        if item.fspath.basename.startswith(("test_distributed_", "tests_distributed_")):
            item._nodeid = item.nodeid + suffix
