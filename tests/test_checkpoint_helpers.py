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
Non-distributed unit tests for the three pure-Python helpers in
``makani.utils.checkpoint_helpers``:

  * ``get_latest_checkpoint_version``
  * ``get_model_state_dict_prefix``
  * ``prepend_prefix_to_state_dict``

The distributed gather/scatter round-trip is covered separately by
``tests/distributed/tests_distributed_checkpoint.py``.
"""

import os
import sys
import time
import unittest
import tempfile
from collections import OrderedDict

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.checkpoint_helpers import (
    get_latest_checkpoint_version,
    get_model_state_dict_prefix,
    prepend_prefix_to_state_dict,
)


class TestGetLatestCheckpointVersion(unittest.TestCase):
    """
    ``get_latest_checkpoint_version`` formats the path template with
    ``mp_rank=0, checkpoint_version="*"``, globs the result, and parses
    the version from the basename via ``_v(\\d+)`` at end-of-name.
    """

    def setUp(self):
        self.tmpdir_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_ctx.name
        # Standard makani checkpoint pattern: mp_rank in name, version after _v.
        self.path_template = os.path.join(
            self.tmpdir, "ckpt_mp{mp_rank}_v{checkpoint_version}.tar"
        )

    def tearDown(self):
        self.tmpdir_ctx.cleanup()

    def _touch(self, name, mtime=None):
        """Create an empty file at name, optionally overriding its mtime."""
        path = os.path.join(self.tmpdir, name)
        with open(path, "w"):
            pass
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def test_returns_zero_when_no_files_match(self):
        # No files exist → bare-except branch returns 0.
        self.assertEqual(get_latest_checkpoint_version(self.path_template), 0)

    def test_extracts_version_from_single_file(self):
        self._touch("ckpt_mp0_v3.tar")
        self.assertEqual(get_latest_checkpoint_version(self.path_template), 3)

    def test_picks_latest_by_mtime_not_version_number(self):
        # The implementation uses os.path.getmtime to pick the "latest" — NOT the
        # numeric max. Pin this behavior: write v5 at t=100, v2 at t=200, expect 2.
        self._touch("ckpt_mp0_v5.tar", mtime=100)
        self._touch("ckpt_mp0_v2.tar", mtime=200)
        self.assertEqual(get_latest_checkpoint_version(self.path_template), 2)

    def test_picks_highest_version_when_mtimes_strictly_increase(self):
        # Common case: each successive checkpoint has a later mtime AND a larger
        # version. The function returns the last-written one.
        self._touch("ckpt_mp0_v0.tar", mtime=100)
        self._touch("ckpt_mp0_v1.tar", mtime=200)
        self._touch("ckpt_mp0_v7.tar", mtime=300)
        self.assertEqual(get_latest_checkpoint_version(self.path_template), 7)

    def test_only_inspects_mp_rank_0_files(self):
        # The template is formatted with mp_rank=0, so files at other ranks
        # are not seen. With only an mp_rank=1 file present, function returns 0.
        self._touch("ckpt_mp1_v9.tar")
        self.assertEqual(get_latest_checkpoint_version(self.path_template), 0)

    def test_returns_zero_on_unparseable_filename(self):
        # File matches the glob (mp_rank=0) but lacks ``_v<digits>``: the regex
        # match fails, the bare except swallows it, function returns 0.
        # We construct a glob-matching file by faking the wildcard portion.
        weird_path = os.path.join(self.tmpdir, "ckpt_mp0_vNOTANUMBER.tar")
        with open(weird_path, "w"):
            pass
        self.assertEqual(get_latest_checkpoint_version(self.path_template), 0)


class _OrigModWrapper:
    """Bare wrapper exposing ``_orig_mod`` — mimics torch.compile's OptimizedModule
    for prefix-walk testing without invoking the actual compiler."""

    def __init__(self, inner):
        self._orig_mod = inner


class _FakeDDP(nn.parallel.DistributedDataParallel):
    """Subclass of DDP that bypasses ``__init__`` so it doesn't require an
    initialized process group. ``isinstance`` still returns True, which is
    all ``get_model_state_dict_prefix`` checks."""

    def __init__(self, inner):
        nn.Module.__init__(self)
        self.module = inner


class TestGetModelStateDictPrefix(unittest.TestCase):
    """
    Walks the wrapper chain accumulating ``_orig_mod.`` for torch.compile and
    ``module.`` for DDP. Order of accumulation reflects the wrapper nesting
    order from outside in.
    """

    def test_plain_module_no_prefix(self):
        m = nn.Linear(2, 2)
        self.assertEqual(get_model_state_dict_prefix(m), "")

    def test_compile_wrapper_only(self):
        m = _OrigModWrapper(nn.Linear(2, 2))
        self.assertEqual(get_model_state_dict_prefix(m), "_orig_mod.")

    def test_ddp_wrapper_only(self):
        m = _FakeDDP(nn.Linear(2, 2))
        self.assertEqual(get_model_state_dict_prefix(m), "module.")

    def test_ddp_wraps_compile_outer_first(self):
        # Outer wrapper is DDP, inner is compile → prefix is "module._orig_mod."
        m = _FakeDDP(_OrigModWrapper(nn.Linear(2, 2)))
        self.assertEqual(get_model_state_dict_prefix(m), "module._orig_mod.")

    def test_compile_wraps_ddp_outer_first(self):
        # Outer wrapper is compile, inner is DDP → prefix is "_orig_mod.module."
        m = _OrigModWrapper(_FakeDDP(nn.Linear(2, 2)))
        self.assertEqual(get_model_state_dict_prefix(m), "_orig_mod.module.")

    def test_three_level_nesting(self):
        # compile(DDP(compile(model))) → "_orig_mod.module._orig_mod."
        m = _OrigModWrapper(_FakeDDP(_OrigModWrapper(nn.Linear(2, 2))))
        self.assertEqual(get_model_state_dict_prefix(m), "_orig_mod.module._orig_mod.")


class TestPrependPrefixToStateDict(unittest.TestCase):
    """
    In-place key-rename: every key gets ``prefix`` prepended. ``_metadata``
    (when present) gets the same treatment. Pinning the slightly surprising
    behavior of the metadata loop's empty-key handling.
    """

    def test_empty_dict_no_op(self):
        d = OrderedDict()
        prepend_prefix_to_state_dict(d, "module.")
        self.assertEqual(dict(d), {})

    def test_empty_prefix_keys_unchanged(self):
        # Empty prefix means newkey == key. The pop-then-reinsert sequence
        # iterates in original order and reinserts in the same order, so the
        # mapping is unchanged.
        d = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
        prepend_prefix_to_state_dict(d, "")
        self.assertEqual(list(d.keys()), ["a", "b", "c"])
        self.assertEqual(list(d.values()), [1, 2, 3])

    def test_keys_get_prefix_prepended(self):
        d = OrderedDict([("layer.weight", 1), ("layer.bias", 2)])
        prepend_prefix_to_state_dict(d, "module.")
        self.assertEqual(set(d.keys()), {"module.layer.weight", "module.layer.bias"})
        self.assertEqual(d["module.layer.weight"], 1)
        self.assertEqual(d["module.layer.bias"], 2)

    def test_in_place_same_object(self):
        # The function mutates the passed-in dict; doesn't return a new one.
        d = OrderedDict([("a", 1)])
        before_id = id(d)
        result = prepend_prefix_to_state_dict(d, "x.")
        self.assertIsNone(result)             # function returns None
        self.assertEqual(id(d), before_id)    # same object
        self.assertEqual(list(d.keys()), ["x.a"])

    def test_metadata_keys_also_get_prefix(self):
        # state_dict from torch normally has a _metadata attribute (an OrderedDict);
        # the function also rewrites those keys. Construct a state-dict-like dict
        # and attach _metadata via a subclass (plain dict can't take new attrs).
        class _DictWithMetadata(OrderedDict):
            pass

        d = _DictWithMetadata([("a", 1), ("b.c", 2)])
        d._metadata = OrderedDict([("", {"version": 1}), ("b", {"version": 1})])

        prepend_prefix_to_state_dict(d, "module.")

        with self.subTest(desc="data keys"):
            self.assertEqual(set(d.keys()), {"module.a", "module.b.c"})
        with self.subTest(desc="metadata keys"):
            # The empty-string metadata key becomes the prefix verbatim ("module.");
            # the "b" key becomes "module.b".
            self.assertEqual(set(d._metadata.keys()), {"module.", "module.b"})
        with self.subTest(desc="metadata values preserved"):
            self.assertEqual(d._metadata["module."], {"version": 1})
            self.assertEqual(d._metadata["module.b"], {"version": 1})


if __name__ == "__main__":
    unittest.main()
