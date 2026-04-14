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

"""
Unit tests for the DALI external-source helper classes.

GeneralES       – directory-based loader (one HDF5 file per year)
GeneralConcatES – single-file loader (all years concatenated)

These classes implement the DALI external-source protocol but are plain Python
objects that can be exercised without DALI.  We mock the DALI SampleInfo
namedtuple and call __call__ directly.

Dataset fixture
---------------
TestGeneralES reuses init_dataset() from testutils, which creates two training
years (2017, 2018) of 365 samples each (dhours=24) under a temp directory.

TestGeneralConcatES uses the same dimensions but builds a single concatenated
HDF5 file.

Boundary exclusion test
-----------------------
With dhours=24 and the default n_history=0/n_future=0/dt=1 config:
  lookback_hours = 24 * (n_future+1) = 24
  boundary "2017-01-25T00:00:00+00:00" → exclusion range [2017-01-24, 2017-01-25)
  sample 23 of 2017 (= 2017-01-01 + 23*24h = 2017-01-24) falls inside → excluded.
"""

import math
import os
import sys
import datetime as dt
import tempfile
import unittest

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import (
    H5_PATH,
    NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W,
    NUM_SAMPLES_PER_YEAR, TRAIN_YEARS, DHOURS,
    init_dataset,
    compare_arrays,
)

# ---------------------------------------------------------------------------
# Dataset parameters — derived from testutils constants, not hardcoded
# ---------------------------------------------------------------------------
_N_CH       = NUM_CHANNELS
_IMG_H      = IMG_SIZE_H
_IMG_W      = IMG_SIZE_W
_N_PER_YEAR = NUM_SAMPLES_PER_YEAR
_DHOURS     = DHOURS
_YEARS      = TRAIN_YEARS
_N_TOTAL    = len(_YEARS) * _N_PER_YEAR

# Default config: n_history=0, n_future=0, dt=1, truncate_old=False
# samples_end = N_TOTAL - dt*(n_future+1) = N_TOTAL - 1
_N_VALID = _N_TOTAL - 1

# Boundary timestamp = exactly sample 24 of the first training year.
# lookback_hours = DHOURS*(n_future+1) = DHOURS*1, so the exclusion window
# is [sample_24_time - DHOURS, sample_24_time), which contains sample 23.
_BOUNDARY_DT = (
    dt.datetime(_YEARS[0], 1, 1, tzinfo=dt.timezone.utc)
    + dt.timedelta(hours=24 * _DHOURS)   # sample 24
)
_BOUNDARY_TS = _BOUNDARY_DT.isoformat()
_N_VALID_WITH_BOUNDARY = _N_VALID - 1   # sample 23 excluded


# ---------------------------------------------------------------------------
# Mock DALI SampleInfo
# ---------------------------------------------------------------------------
class _SampleInfo:
    def __init__(self, idx_in_epoch=0, epoch_idx=0, iteration=0):
        self.idx_in_epoch = idx_in_epoch
        self.epoch_idx    = epoch_idx
        self.iteration    = iteration


# ---------------------------------------------------------------------------
# Distinctive-channel datasets (for channel-reordering test only)
# Channel c has the constant value (c+1.0) so reordering can be verified.
# ---------------------------------------------------------------------------

def _distinctive_year_arrays(year):
    """Return (data, timestamps) arrays for one year of distinctive-channel data."""
    year_start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc).timestamp()
    timestamps = year_start + np.arange(_N_PER_YEAR, dtype=np.float64) * _DHOURS * 3600
    data = np.zeros((_N_PER_YEAR, _N_CH, _IMG_H, _IMG_W), dtype=np.float32)
    for c in range(_N_CH):
        data[:, c, :, :] = float(c + 1)
    return data, timestamps


def _make_distinctive_dir(root):
    """Per-year directory with distinctive-channel data (for GeneralES)."""
    os.makedirs(root, exist_ok=True)
    for year in _YEARS:
        data, timestamps = _distinctive_year_arrays(year)
        with h5py.File(os.path.join(root, f"{year}.h5"), "w") as f:
            ds = f.create_dataset(H5_PATH, data=data)
            ts = f.create_dataset("timestamp", data=timestamps)
            ts.make_scale("timestamp")
            ds.dims[0].attach_scale(ts)
    return root


def _make_distinctive_concat(root):
    """Single concatenated file with distinctive-channel data (for GeneralConcatES)."""
    os.makedirs(root, exist_ok=True)
    all_data, all_ts = [], []
    for year in _YEARS:
        data, timestamps = _distinctive_year_arrays(year)
        all_data.append(data)
        all_ts.append(timestamps)
    path = os.path.join(root, "train_concat.h5")
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(H5_PATH, data=np.concatenate(all_data, axis=0))
        ts = f.create_dataset("timestamp", data=np.concatenate(all_ts))
        ts.make_scale("timestamp")
        ds.dims[0].attach_scale(ts)
    return path


# ---------------------------------------------------------------------------
# ES factory helpers
# ---------------------------------------------------------------------------

def _default_kwargs(location):
    return dict(
        location=location,
        max_samples=None,
        samples_per_epoch=None,
        train=False,            # no shuffle → deterministic
        batch_size=1,
        dt=1,
        dhours=_DHOURS,
        n_history=0,
        n_future=0,
        in_channels=np.arange(_N_CH),
        out_channels=np.arange(_N_CH),
        crop_size=[None, None],
        crop_anchor=[0, 0],
        subsampling_factor=1,
        num_shards=1,
        shard_id=0,
        io_grid=[1, 1, 1],
        io_rank=[0, 0, 0],
        device_id=0,
        truncate_old=False,
        enable_logging=False,
        zenith_angle=False,
        return_timestamp=False,
        is_parallel=False,
        timestamp_boundary_list=[],
    )


def _make_general_es(location, **overrides):
    """Build a GeneralES and simulate __setstate__ (set file-handle methods)."""
    from makani.utils.dataloaders.dali_es_helper_2d import GeneralES
    kw = _default_kwargs(location)
    kw.update(overrides)
    es = GeneralES(**kw)
    # replicate what __setstate__ does: install the per-format file handles
    if es.file_format == "h5":
        es.get_year_handle = es._get_year_h5
        es.get_data_handle = es._get_data_h5
    else:
        es.get_year_handle = es._get_year_zarr
        es.get_data_handle = es._get_data_zarr
    return es


def _make_concat_es(file_path, **overrides):
    """Build a GeneralConcatES and simulate __setstate__ (open file handles)."""
    from makani.utils.dataloaders.dali_es_helper_concat_2d import GeneralConcatES
    kw = _default_kwargs(file_path)
    kw.update(overrides)
    es = GeneralConcatES(**kw)
    # replicate what __setstate__ does: open the file and install data handle
    es.vfile = h5py.File(es.file_path, "r", driver=es.file_driver)
    es.dset  = es.vfile[es.dataset_path]
    es.get_data_handle = es._get_data_h5
    return es


def _ts_to_posix(t):
    """Normalise a timestamp to a POSIX float.

    GeneralES._compute_timestamps() returns numpy float64 (seconds since epoch).
    GeneralConcatES._compute_timestamps_and_zenith_angle() returns datetime objects
    (slices of self.timestamps).  This helper handles both.
    """
    try:
        return float(t.timestamp())   # datetime path
    except AttributeError:
        return float(t)               # float / np.float64 path


# ===========================================================================
# Shared test logic (mixin)
# ===========================================================================

class _BaseESTests:
    """
    Tests shared between TestGeneralES and TestGeneralConcatES.

    Subclasses must implement:
        _make(**overrides)             → ES backed by standard random dataset
        _make_distinctive(**overrides) → ES backed by distinctive-channel dataset
    """

    def _call0(self, es):
        """Call es at the very first sample of epoch 0."""
        return es(_SampleInfo(idx_in_epoch=0, epoch_idx=0, iteration=0))

    # ---- 1. basic shapes ------------------------------------------------

    def test_basic_output_shapes(self):
        es = self._make()
        inp, tar = self._call0(es)
        self.assertEqual(inp.shape, (1, _N_CH, _IMG_H, _IMG_W))
        self.assertEqual(tar.shape, (1, _N_CH, _IMG_H, _IMG_W))

    # ---- 2. n_history ---------------------------------------------------

    def test_n_history_adds_input_timesteps(self):
        es = self._make(n_history=2)
        inp, _tar = self._call0(es)
        self.assertEqual(inp.shape[0], 3)   # n_history + 1

    # ---- 3. n_future ----------------------------------------------------

    def test_n_future_adds_target_timesteps(self):
        es = self._make(n_future=2)
        _inp, tar = self._call0(es)
        self.assertEqual(tar.shape[0], 3)   # n_future + 1

    # ---- 4. subsampling -------------------------------------------------

    def test_subsampling_halves_spatial_dims(self):
        es = self._make(subsampling_factor=2)
        inp, tar = self._call0(es)
        self.assertEqual(inp.shape, (1, _N_CH, math.ceil(_IMG_H / 2), math.ceil(_IMG_W / 2)))
        self.assertEqual(tar.shape, (1, _N_CH, math.ceil(_IMG_H / 2), math.ceil(_IMG_W / 2)))

    def test_subsampling_quarter_spatial_dims(self):
        es = self._make(subsampling_factor=4)
        inp, tar = self._call0(es)
        self.assertEqual(inp.shape, (1, _N_CH, math.ceil(_IMG_H / 4), math.ceil(_IMG_W / 4)))
        self.assertEqual(tar.shape, (1, _N_CH, math.ceil(_IMG_H / 4), math.ceil(_IMG_W / 4)))

    def test_subsampling_matches_manual_subsample(self):
        """ES subsampling == slicing the full-res sample every S pixels."""
        S = 2
        es_full = self._make()
        es_sub  = self._make(subsampling_factor=S)
        inp_full, tar_full = self._call0(es_full)
        inp_sub,  tar_sub  = self._call0(es_sub)
        self.assertTrue(compare_arrays(
            "subsampled inp", inp_sub, inp_full[:, :, ::S, ::S], verbose=True))
        self.assertTrue(compare_arrays(
            "subsampled tar", tar_sub, tar_full[:, :, ::S, ::S], verbose=True))

    # ---- 5. spatial crop ------------------------------------------------

    def test_spatial_crop_limits_dims(self):
        crop_h, crop_w = _IMG_H // 2, _IMG_W // 2
        es = self._make(crop_size=[crop_h, crop_w], crop_anchor=[0, 0])
        inp, tar = self._call0(es)
        self.assertEqual(inp.shape, (1, _N_CH, crop_h, crop_w))
        self.assertEqual(tar.shape, (1, _N_CH, crop_h, crop_w))

    def test_spatial_crop_with_nonzero_anchor(self):
        crop_h, crop_w     = _IMG_H // 2, _IMG_W // 2
        anchor_h, anchor_w = 4, 8
        es = self._make(crop_size=[crop_h, crop_w], crop_anchor=[anchor_h, anchor_w])
        inp, _tar = self._call0(es)
        self.assertEqual(inp.shape, (1, _N_CH, crop_h, crop_w))

    def test_crop_matches_manual_crop(self):
        """ES crop (zero anchor) == slicing the full-res sample."""
        crop_h, crop_w = _IMG_H // 2, _IMG_W // 2
        es_full = self._make()
        es_crop = self._make(crop_size=[crop_h, crop_w], crop_anchor=[0, 0])
        inp_full, tar_full = self._call0(es_full)
        inp_crop, tar_crop = self._call0(es_crop)
        self.assertTrue(compare_arrays(
            "cropped inp", inp_crop, inp_full[:, :, :crop_h, :crop_w], verbose=True))
        self.assertTrue(compare_arrays(
            "cropped tar", tar_crop, tar_full[:, :, :crop_h, :crop_w], verbose=True))

    def test_crop_with_anchor_matches_manual_crop(self):
        """ES crop with nonzero anchor == slicing full-res at the anchor offset."""
        crop_h, crop_w     = _IMG_H // 2, _IMG_W // 2
        anchor_h, anchor_w = 4, 8
        es_full = self._make()
        es_crop = self._make(crop_size=[crop_h, crop_w], crop_anchor=[anchor_h, anchor_w])
        inp_full, tar_full = self._call0(es_full)
        inp_crop, tar_crop = self._call0(es_crop)
        self.assertTrue(compare_arrays(
            "anchored crop inp",
            inp_crop,
            inp_full[:, :, anchor_h:anchor_h + crop_h, anchor_w:anchor_w + crop_w],
            verbose=True,
        ))
        self.assertTrue(compare_arrays(
            "anchored crop tar",
            tar_crop,
            tar_full[:, :, anchor_h:anchor_h + crop_h, anchor_w:anchor_w + crop_w],
            verbose=True,
        ))

    # ---- 6. temporal window consistency ---------------------------------

    def test_window_matches_individual_samples(self):
        """
        Load one windowed sample at index i with n_history=1, n_future=3.
        The resulting (inp, tar) should equal 6 individually loaded frames
        stitched together:
          inp[0..1]  ←  single samples at global indices i-1, i
          tar[0..3]  ←  single samples at global indices i+1, i+2, i+3, i+4

        With train=False and truncate_old=False, the windowed ES starts at
        samples_start = dt * n_history = 1, so i = 1 and the window spans
        global indices 0..5.  The single-sample ES (n_history=0, n_future=0)
        starts at 0, so idx_in_epoch=k deterministically selects sample k.
        """
        N_HIST, N_FUT = 1, 3

        es_window = self._make(n_history=N_HIST, n_future=N_FUT)
        es_single = self._make()   # n_history=0, n_future=0

        inp_win, tar_win = self._call0(es_window)

        # collect (N_HIST+1) + (N_FUT+1) = 6 individual frames
        n_frames = (N_HIST + 1) + (N_FUT + 1)
        singles = []
        for k in range(n_frames):
            inp_k, _ = es_single(_SampleInfo(idx_in_epoch=k, epoch_idx=0, iteration=0))
            singles.append(inp_k[0])   # shape (n_ch, H, W) — copy returned by _reorder_channels

        for t in range(N_HIST + 1):
            self.assertTrue(compare_arrays(
                f"inp t={t}", inp_win[t], singles[t], verbose=True,
            ))
        for t in range(N_FUT + 1):
            self.assertTrue(compare_arrays(
                f"tar t={t}", tar_win[t], singles[N_HIST + 1 + t], verbose=True,
            ))

    # ---- 7. channel subset ----------------------------------------------

    def test_channel_subset_shapes(self):
        es = self._make(
            in_channels=np.array([0, 1, 2]),
            out_channels=np.array([0, 1]),
        )
        inp, tar = self._call0(es)
        self.assertEqual(inp.shape[1], 3)
        self.assertEqual(tar.shape[1], 2)

    # ---- 7. unsorted channels reordered ---------------------------------

    def test_unsorted_channels_reordered(self):
        """
        With in_channels=[3,1,0,2] and channel c having constant value (c+1):
          output position 0 → channel 3 → value 4
          output position 1 → channel 1 → value 2
          output position 2 → channel 0 → value 1
          output position 3 → channel 2 → value 3
        """
        sel = [3, 1, 0, 2]
        es = self._make_distinctive(in_channels=np.array(sel))
        inp, _tar = self._call0(es)
        # expected shape matches the number of selected channels (4), not _N_CH (5)
        vals = [4.0, 2.0, 1.0, 3.0]
        expected = np.empty((len(sel), _IMG_H, _IMG_W), dtype=np.float32)
        for pos, val in enumerate(vals):
            expected[pos] = val
        self.assertTrue(
            compare_arrays("unsorted channel reorder", inp[0], expected, verbose=True)
        )

    # ---- 7b. unsorted out_channels reordered ----------------------------

    def test_unsorted_out_channels_reordered(self):
        """
        With out_channels=[3,1,0,2] and channel c having constant value (c+1):
          tar output position 0 → channel 3 → value 4
          tar output position 1 → channel 1 → value 2
          tar output position 2 → channel 0 → value 1
          tar output position 3 → channel 2 → value 3
        """
        sel = [3, 1, 0, 2]
        es = self._make_distinctive(out_channels=np.array(sel))
        _inp, tar = self._call0(es)
        vals = [4.0, 2.0, 1.0, 3.0]
        expected = np.empty((len(sel), _IMG_H, _IMG_W), dtype=np.float32)
        for pos, val in enumerate(vals):
            expected[pos] = val
        self.assertTrue(
            compare_arrays("unsorted out_channel reorder", tar[0], expected, verbose=True)
        )

    # ---- 8. zenith angle ------------------------------------------------

    def test_zenith_angle_appended(self):
        es = self._make(zenith_angle=True)
        result = self._call0(es)
        self.assertEqual(len(result), 4)          # inp, tar, zen_inp, zen_tar
        zen_inp, zen_tar = result[2], result[3]
        self.assertEqual(zen_inp.shape, (1, 1, _IMG_H, _IMG_W))
        self.assertEqual(zen_tar.shape, (1, 1, _IMG_H, _IMG_W))
        # cosine zenith angle is bounded in [-1, 1]
        self.assertTrue(np.all(zen_inp >= -1.0) and np.all(zen_inp <= 1.0))
        self.assertTrue(np.all(zen_tar >= -1.0) and np.all(zen_tar <= 1.0))

    def test_zenith_with_subsampling(self):
        es = self._make(zenith_angle=True, subsampling_factor=2)
        result = self._call0(es)
        self.assertEqual(result[2].shape, (1, 1, math.ceil(_IMG_H / 2), math.ceil(_IMG_W / 2)))

    # ---- 9. return timestamp --------------------------------------------

    def test_return_timestamp_appended(self):
        es = self._make(return_timestamp=True)
        result = self._call0(es)
        self.assertEqual(len(result), 4)    # inp, tar, inp_time, tar_time
        self.assertEqual(len(result[2]), 1) # n_history + 1
        self.assertEqual(len(result[3]), 1) # n_future + 1

    def test_return_timestamp_n_history(self):
        es = self._make(return_timestamp=True, n_history=2)
        result = self._call0(es)
        self.assertEqual(len(result[2]), 3) # n_history + 1 = 3

    # ---- 10. zenith + timestamp combined --------------------------------

    def test_zenith_and_timestamp_both_appended(self):
        es = self._make(zenith_angle=True, return_timestamp=True)
        result = self._call0(es)
        self.assertEqual(len(result), 6)   # inp, tar, zen_inp, zen_tar, inp_t, tar_t

    # ---- 11. pickle round-trip ------------------------------------------

    def test_pickle_roundtrip(self):
        """
        Simulates the DALI main-process → worker-process transfer.

        Main-process state: __init__ has run but no file handles are open and
        no buffers are allocated (is_parallel=True).  After pickle.loads the
        worker calls __setstate__, which opens the file, installs the data
        handle, and allocates buffers.  The restored ES must produce the same
        output as a freshly constructed (worker-state) ES.
        """
        import pickle
        es_ref      = self._make()             # worker-process state (our simulation)
        es_restored = pickle.loads(pickle.dumps(self._make_for_pickle()))

        inp_ref, tar_ref = self._call0(es_ref)
        inp_res, tar_res = self._call0(es_restored)

        with self.subTest(desc="inp"):
            self.assertTrue(compare_arrays("pickled inp", inp_res, inp_ref, verbose=True))
        with self.subTest(desc="tar"):
            self.assertTrue(compare_arrays("pickled tar", tar_res, tar_ref, verbose=True))

    # ---- 12. StopIteration at epoch end ---------------------------------

    def test_stop_iteration_at_epoch_end(self):
        es = self._make()
        with self.assertRaises(StopIteration):
            es(_SampleInfo(iteration=es.num_steps_per_epoch))

    # ---- 12. n_samples_total / max_samples ------------------------------

    def test_max_samples_limits_dataset(self):
        """max_samples caps the usable sample count."""
        max_s = 50
        es_full   = self._make()
        es_capped = self._make(max_samples=max_s)
        self.assertLess(es_capped.n_samples_total, es_full.n_samples_total)
        self.assertLessEqual(es_capped.n_samples_total, max_s)

    def test_n_samples_total(self):
        es = self._make()
        self.assertEqual(es.n_samples_total, _N_VALID)

    def test_shuffle_epoch_cycle(self):
        """
        Exercises train=True (shuffled) with a short epoch and max_samples cap.

        Setup
        -----
        max_samples = N_SAMPLES + 1  →  n_samples_total = N_SAMPLES = 50
        samples_per_epoch = EPOCH_LEN = 10, batch_size = 1
        → num_steps_per_epoch = 10, num_steps_per_cycle = 50

        DALI epoch model
        ----------------
        iteration resets to 0 at the start of each epoch.
        idx_in_epoch and epoch_idx together form global_sample_idx, which
        walks through the shuffle permutation across epochs.
        5 epochs × 10 steps = 50 total steps = 1 full cycle  →  every sample
        visited exactly once.

        Assertions
        ----------
        1. StopIteration is raised after exactly EPOCH_LEN steps in each epoch.
        2. Across all epochs, exactly N_SAMPLES samples are loaded.
        3. Every sample is visited exactly once (no repeats, no omissions):
           the collected timestamp set equals es.timestamps[es.indices_select].
        """
        N_SAMPLES = 50
        EPOCH_LEN = 10
        N_EPOCHS  = N_SAMPLES // EPOCH_LEN   # 5 full epochs = 1 cycle

        es = self._make(
            train=True,
            max_samples=N_SAMPLES + 1,       # +1: samples_end = (N+1) - 1 = N
            samples_per_epoch=EPOCH_LEN,
            batch_size=1,
            return_timestamp=True,
        )
        self.assertEqual(es.n_samples_total,     N_SAMPLES)
        self.assertEqual(es.num_steps_per_epoch, EPOCH_LEN)

        loaded_ts = []

        for epoch_idx in range(N_EPOCHS):
            # --- normal steps for this epoch ---
            for step in range(EPOCH_LEN):
                result = es(_SampleInfo(
                    idx_in_epoch=step,
                    epoch_idx=epoch_idx,
                    iteration=step,          # iteration resets to 0 each epoch
                ))
                inp_time = result[2]         # (inp, tar, inp_time, tar_time)
                loaded_ts.append(_ts_to_posix(inp_time[0]))

            # --- StopIteration fires when iteration reaches EPOCH_LEN ---
            with self.subTest(desc=f"StopIteration epoch {epoch_idx}"):
                with self.assertRaises(StopIteration):
                    es(_SampleInfo(
                        idx_in_epoch=0,
                        epoch_idx=epoch_idx,
                        iteration=EPOCH_LEN,
                    ))

        # 1. Correct total count
        with self.subTest(desc="total samples loaded"):
            self.assertEqual(len(loaded_ts), N_SAMPLES)

        # 2. No sample repeated
        with self.subTest(desc="no repeated samples"):
            self.assertEqual(len(set(loaded_ts)), N_SAMPLES)

        # 3. Loaded set == complete valid-sample set (all indices_select covered)
        expected_ts = {_ts_to_posix(ts) for ts in es.timestamps[es.indices_select]}
        with self.subTest(desc="complete coverage"):
            self.assertEqual(set(loaded_ts), expected_ts)

    # ---- 13. timestamp boundary list ------------------------------------

    def test_timestamp_boundary_excludes_one_sample(self):
        """
        Boundary at 2017-01-25T00:00:00Z puts sample-23 of 2017 inside the
        lookback window [2017-01-24, 2017-01-25), excluding exactly 1 index.
        """
        es_plain    = self._make()
        es_boundary = self._make(timestamp_boundary_list=[_BOUNDARY_TS])
        self.assertEqual(es_plain.n_samples_total,    _N_VALID)
        self.assertEqual(es_boundary.n_samples_total, _N_VALID_WITH_BOUNDARY)

    def test_empty_boundary_list_unchanged(self):
        es = self._make(timestamp_boundary_list=[])
        self.assertEqual(es.n_samples_total, _N_VALID)


# ===========================================================================
# Concrete test classes
# ===========================================================================

class TestGeneralES(_BaseESTests, unittest.TestCase):
    """Tests for GeneralES (directory of per-year HDF5 files).

    The main dataset is created by init_dataset() from testutils, which
    produces the same two-year training layout used by the other dataloader tests.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._train_path, *_ = init_dataset(self._tmpdir.name)
        # small distinctive-channel dataset for the reordering test
        self._dc_path = _make_distinctive_dir(
            os.path.join(self._tmpdir.name, "dc"))

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make(self, **overrides):
        return _make_general_es(self._train_path, **overrides)

    def _make_distinctive(self, **overrides):
        return _make_general_es(self._dc_path, **overrides)

    def _make_for_pickle(self, **overrides):
        """Main-process state: __init__ done, no file handles, no buffers."""
        from makani.utils.dataloaders.dali_es_helper_2d import GeneralES
        kw = _default_kwargs(self._train_path)
        kw["is_parallel"] = True   # buffers not yet allocated; __setstate__ will do it
        kw.update(overrides)
        return GeneralES(**kw)

    def test_missing_directory_raises(self):
        with self.assertRaises(IOError):
            _make_general_es(os.path.join(self._tmpdir.name, "does_not_exist"))


class TestGeneralConcatES(_BaseESTests, unittest.TestCase):
    """Tests for GeneralConcatES (single concatenated HDF5 file)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        # init_dataset with create_concat=True writes train_concat.h5 alongside
        # the per-year files and returns its path as the 7th element.
        *_, self._file = init_dataset(self._tmpdir.name, create_concat=True)
        # distinctive-channel concat file for the reordering test
        dc_root = os.path.join(self._tmpdir.name, "dc")
        self._file_dc = _make_distinctive_concat(dc_root)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make(self, **overrides):
        return _make_concat_es(self._file, **overrides)

    def _make_distinctive(self, **overrides):
        return _make_concat_es(self._file_dc, **overrides)

    def _make_for_pickle(self, **overrides):
        """Main-process state: __init__ done, vfile=None, no buffers."""
        from makani.utils.dataloaders.dali_es_helper_concat_2d import GeneralConcatES
        kw = _default_kwargs(self._file)
        kw["is_parallel"] = True   # buffers not yet allocated; __setstate__ will do it
        kw.update(overrides)
        return GeneralConcatES(**kw)

    def test_missing_file_raises(self):
        with self.assertRaises(IOError):
            _make_concat_es(os.path.join(self._tmpdir.name, "no_such.h5"))

    def test_s3_raises_not_implemented(self):
        from makani.utils.dataloaders.dali_es_helper_concat_2d import GeneralConcatES
        kw = _default_kwargs(self._file)
        kw["enable_s3"] = True
        with self.assertRaises(NotImplementedError):
            GeneralConcatES(**kw)


if __name__ == "__main__":
    unittest.main()
