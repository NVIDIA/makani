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

Suppresses pytest's console output on every rank except rank 0, so the combined
mpirun/srun stdout shows a single coherent test summary instead of N interleaved
copies. The silencing unregisters the terminal reporter in ``pytest_configure``
with ``@pytest.hookimpl(trylast=True)`` -- see that hook for why ``trylast`` is
required and why it must happen at configure time.

It also surfaces the GRID_H x GRID_W x GRID_E decomposition: ``pytest_report_header``
prints it once at the top of the session and ``pytest_collection_modifyitems``
appends ``[h=H,w=W]`` to each distributed test's node id.

Escape hatch: setting ``MAKANI_TEST_NO_SILENCE=1`` in the environment disables the
silencing on every rank -- useful when debugging a per-rank failure (e.g. a
distributed-bootstrap error in ``setUpClass``) that rank 0 doesn't reproduce.
"""

import os

import pytest


def _get_world_rank():
    """Return the world rank from the env: ``WORLD_RANK`` if set, else ``RANK``.

    Read directly from the env, not via ``comm``: ``pytest_configure`` runs before
    ``comm.init()`` (which happens in ``setUpClass`` / ``setUpModule``), and
    ``comm.get_world_rank()`` returns 0 until then -- so trusting comm here would make
    every rank look like rank 0 and nothing would get silenced. ``WORLD_RANK`` is
    checked first as the explicit convention; ``RANK`` is the torch.distributed /
    torchrun (and mpirun) variable that is actually set in our launches.
    """
    for key in ("WORLD_RANK", "RANK"):
        val = os.environ.get(key)
        if val is not None:
            return int(val)
    return 0


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Silence pytest's console output on every rank except rank 0.

    ``trylast=True`` is REQUIRED: conftest ``pytest_configure`` hooks otherwise run
    *before* pytest's builtin terminal plugin registers the ``terminalreporter``
    (LIFO plugin order), so ``get_plugin("terminalreporter")`` returns ``None`` and
    the unregister silently no-ops -- the duplicate-output bug. Running last ensures
    the reporter exists by the time we unregister it.

    Silencing at configure time (rather than in ``pytest_runtest_setup``) is also what
    suppresses the session header, the collected-items line and the summary on non-zero
    ranks: a per-test hook fires too late, after those have already been printed.
    """
    # Honor an env-var escape hatch: forces full output on every rank.
    if os.environ.get("MAKANI_TEST_NO_SILENCE"):
        return

    # rank comes from the env (comm isn't initialized at configure time)
    if _get_world_rank() == 0:
        return

    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        config.pluginmanager.unregister(reporter)


def _grid():
    """The distributed decomposition from the launcher env (default 1 x 1 x 1)."""
    return (
        int(os.environ.get("GRID_H", 1)),
        int(os.environ.get("GRID_W", 1)),
        int(os.environ.get("GRID_E", 1)),
    )


def pytest_report_header(config):
    """Surface the GRID_H x GRID_W x GRID_E decomposition at the top of the session.

    Non-zero ranks have already had their terminal reporter unregistered in
    ``pytest_configure``, so this prints on rank 0 only. Returns ``None`` for a plain
    1x1x1 run.
    """
    h, w, e = _grid()
    if h == 1 and w == 1 and e == 1:
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
