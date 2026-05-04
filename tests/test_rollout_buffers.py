# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
from parameterized import parameterized
import tempfile
import os
import sys
import numpy as np
import torch
import h5py as h5
from typing import Optional

from makani.utils.inference.rollout_buffer import MeanStdBuffer, TemporalAverageBuffer
from makani.utils.dataloaders.data_helpers import get_lat_lon_grid

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, set_seed, init_dataset, get_default_parameters, compare_arrays, compare_tensors, H5_PATH, IMG_SIZE_H, IMG_SIZE_W


def init_dataset_params(
    train_path: str,
    valid_path: str,
    stats_path: str,
    batch_size: int,
    n_history: int,
    n_future: int,
    normalization: str,
    num_data_workers: int,
):
    """Initialize dataset parameters using the same approach as test_dataloader.py"""

    # instantiate params base
    params = get_default_parameters()

    # init paths
    params.train_data_path = train_path
    params.valid_data_path = valid_path
    params.min_path = os.path.join(stats_path, "mins.npy")
    params.max_path = os.path.join(stats_path, "maxs.npy")
    params.time_means_path = os.path.join(stats_path, "time_means.npy")
    params.global_means_path = os.path.join(stats_path, "global_means.npy")
    params.global_stds_path = os.path.join(stats_path, "global_stds.npy")
    params.time_diff_means_path = os.path.join(stats_path, "time_diff_means.npy")
    params.time_diff_stds_path = os.path.join(stats_path, "time_diff_stds.npy")

    # general parameters
    params.dhours = 24
    params.h5_path = H5_PATH
    params.n_history = n_history
    params.n_future = n_future
    params.batch_size = batch_size
    params.normalization = normalization

    # performance parameters
    params.num_data_workers = num_data_workers
    params.enable_odirect = False
    params.enable_s3 = False

    return params


class TestRolloutBuffers(unittest.TestCase):
    """
    Test class for TemporalAverageBuffer using dataset initialization from test_dataloader.py
    """

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        """Set up test environment using class-level setup like test_dataloader.py"""
        
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        set_seed(333)

        # create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name

        # init datasets and stats using the same approach as test_dataloader.py
        cls.train_path, cls.num_train, cls.valid_path, cls.num_valid, cls.stats_path, cls.metadata_path, _ = init_dataset(tmp_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up after test using class-level teardown"""
        cls.tmpdir.cleanup()

    def setUp(self):
        """Set up test parameters for each test method"""
        
        disable_tf32()

        # Initialize parameters using the same approach as test_dataloader.py
        self.params = init_dataset_params(
            self.train_path, 
            self.valid_path, 
            self.stats_path, 
            batch_size=2, 
            n_history=0, 
            n_future=0, 
            normalization="zscore", 
            num_data_workers=1
        )

        self.params.multifiles = True
        self.params.num_train = self.num_train
        self.params.num_valid = self.num_valid

        # this is also correct for most cases:
        self.params.io_grid = [1, 1, 1]
        self.params.io_rank = [0, 0, 0]

        # Set rollout parameters
        self.rollout_dt = 6  # 6 hours
        self.ensemble_size = 4
        
        # Channel configuration - use the same channels as testutils
        self.channel_names = ["u10m", "t2m", "u500", "z500", "t500"]
        self.output_channels = ["u10m", "t2m", "u500"]  # Only track some channels
        self.num_channels = len(self.channel_names)
        self.num_output_channels = len(self.output_channels)
        
        # Set image dimensions - use the same as testutils
        self.img_shape = (IMG_SIZE_H, IMG_SIZE_W)  # (lat, lon)
        self.local_shape = (IMG_SIZE_H, IMG_SIZE_W)  # Same as img_shape for non-distributed test
        self.local_offset = (0, 0)
        
        # Create lat/lon grid
        self.latitude, self.longitude = get_lat_lon_grid(self.img_shape)
        self.lat_lon = (self.latitude.tolist(), self.longitude.tolist())

    @parameterized.expand(
        [
            (1, 1, False), (2, 1, False), (4, 1, False), (1, 2, False), (2, 2, False), (4, 2, False),
            (1, 1, True), (2, 1, True), (4, 1, True), (1, 2, True), (2, 2, True), (4, 2, True),
        ],
        skip_on_empty=True,
    )
    def test_temporal_averaging_buffer(self, batch_size, num_rollout_steps, scale_bias, verbose=False):
        """
        Test TemporalAverageBuffer by feeding data one tensor at a time and comparing
        with manual mean and variance calculations
        """
        # Create output file path
        output_file = os.path.join(self.tmpdir.name, "temporal_average_output.h5")

        if not scale_bias:
            scale = None
            bias = None
        else:
            scale = torch.ones((self.num_channels,), dtype=torch.float32)
            bias = torch.zeros((self.num_channels,), dtype=torch.float32)
            
        # Initialize TemporalAverageBuffer
        buffer = TemporalAverageBuffer(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=6,
            img_shape=self.img_shape,
            local_shape=self.local_shape,
            local_offset=self.local_offset,
            channel_names=self.channel_names,
            lat_lon=self.lat_lon,
            device=self.device,
            output_channels=self.output_channels,
            output_file=output_file,
            scale=scale,
            bias=bias,
        )
        
        # Load test data from the dummy dataset
        test_file = os.path.join(self.valid_path, "2019.h5")
        with h5.File(test_file, "r") as hf:
            # Load all data from the test file
            data = hf[H5_PATH][:]  # Shape: (num_samples, num_channels, lat, lon)
        
        # Convert to torch tensor
        data_tensor = torch.from_numpy(data).to(self.device)
        
        # Get channel indices for output channels
        output_channel_indices = [self.channel_names.index(ch) for ch in self.output_channels]
        
        # Prepare manual calculation arrays
        manual_data = []  # Will store all data for manual calculation
        
        # Feed data one tensor at a time to the buffer
        num_samples = (data_tensor.shape[0] // batch_size // num_rollout_steps) * num_rollout_steps * batch_size
        
        for idt, step in enumerate(range(0, num_samples, batch_size)):
            # Extract single sample and reshape for buffer
            # Buffer expects: (batch_size, num_channels, lat, lon)
            sample = data_tensor[step:step+batch_size, ...]  # Add batch and ensemble dimension

            # Update buffer
            idte = idt % num_rollout_steps
            buffer.update(sample, idte)
            
            # Store for manual calculation
            manual_data.append(sample[:, output_channel_indices, :, :].cpu().numpy())
        
        # Finalize buffer
        buffer.finalize()
        
        # Manual calculation
        manual_data = np.stack(manual_data, axis=0)
        manual_data = manual_data.reshape(-1, num_rollout_steps, batch_size, self.num_output_channels, *self.img_shape)
        manual_data = np.transpose(manual_data, axes=(0, 2, 1, 3, 4, 5)).reshape(-1, num_rollout_steps, self.num_output_channels, *self.img_shape)
        
        # Calculate manual mean and std for output channels only
        manual_mean = np.mean(manual_data, axis=0)
        manual_std = np.std(manual_data, axis=0, ddof=1)  # ddof=1 for sample std
        
        # Read results from HDF5 file
        with h5.File(output_file, "r") as hf:
            buffer_mean = hf["mean"][:]  # Shape: (num_rollout_steps, num_channels, lat, lon)
            buffer_std = hf["std"][:]     # Shape: (num_rollout_steps, num_channels, lat, lon)
            lead_time = hf["lead_time"][:]
            channels = [x.decode() for x in hf["channel"][:]]
            lats = hf["lat"][:]
            lons = hf["lon"][:]
        
        # Verify file structure
        with self.subTest(desc="buffer shapes"):
            self.assertEqual(buffer_mean.shape, (num_rollout_steps, len(self.output_channels), *self.img_shape))
            self.assertEqual(buffer_std.shape, (num_rollout_steps, len(self.output_channels), *self.img_shape))
            self.assertEqual(len(lead_time), num_rollout_steps)
            self.assertEqual(len(channels), len(self.output_channels))
            self.assertEqual(len(lats), self.img_shape[0])
            self.assertEqual(len(lons), self.img_shape[1])

        # Verify channel names
        with self.subTest(desc="channel names"):
            self.assertEqual(channels, self.output_channels)

        # Verify lead times
        expected_lead_times = np.arange(self.rollout_dt, (num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64)
        with self.subTest(desc="lead times"):
            self.assertTrue(compare_arrays("lead times", lead_time, expected_lead_times, atol=0.0, rtol=1e-6, verbose=verbose))

        # Verify lat/lon coordinates
        with self.subTest(desc="latitudes"):
            self.assertTrue(compare_arrays("latitudes", lats, self.latitude, atol=0.0, rtol=1e-6, verbose=verbose))
        with self.subTest(desc="longitudes"):
            self.assertTrue(compare_arrays("longitudes", lons, self.longitude, atol=0.0, rtol=1e-6, verbose=verbose))

        # Compare with buffer output
        with self.subTest(desc="mean"):
            self.assertTrue(compare_arrays("mean", buffer_mean, manual_mean, atol=0.0, rtol=1e-5, verbose=verbose))
        with self.subTest(desc="std"):
            self.assertTrue(compare_arrays("std", buffer_std, manual_std, atol=0.0, rtol=1e-5, verbose=verbose))


class TestMeanStdBuffer(unittest.TestCase):
    """
    Tests the Welford online accumulator that all averaging buffers inherit from.
    Bugs in this math produce silent skill-metric regressions, so we check it
    against torch's offline mean/m2 on the concatenated batch.

    ``MeanStdBuffer`` is abstract (inherits the ``DataBuffer`` ABC), so we use
    a thin concrete subclass that exposes the protected math methods through
    ``update``.
    """

    @staticmethod
    def _make_buffer(num_rollout_steps, num_channels, variable_shape, scale=None, bias=None):
        class _Concrete(MeanStdBuffer):
            def update(self, data, idt):
                mean, m2, count = self._compute_stats(data, dim=0)
                self._welford_combine(mean, m2, count, idt)

            def finalize(self):
                pass

        channel_names = [f"ch_{i}" for i in range(num_channels)]
        return _Concrete(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=1,
            variable_shape=variable_shape,
            channel_names=channel_names,
            device="cpu",
            scale=scale,
            bias=bias,
            output_channels=channel_names,   # output all channels
            output_file=None,
        )

    def test_compute_stats_matches_torch(self, verbose=False):
        buf = self._make_buffer(num_rollout_steps=1, num_channels=2, variable_shape=(4, 4))
        data = torch.randn(10, 2, 4, 4)
        mean, m2, count = buf._compute_stats(data, dim=0)
        expected_mean = data.mean(dim=0)
        expected_m2 = (data - expected_mean).pow(2).sum(dim=0)

        with self.subTest(desc="mean"):
            self.assertTrue(compare_tensors("compute_stats mean", mean, expected_mean, atol=1e-6, verbose=verbose))
        with self.subTest(desc="m2"):
            self.assertTrue(compare_tensors("compute_stats m2", m2, expected_m2, atol=1e-4, verbose=verbose))
        with self.subTest(desc="count"):
            self.assertEqual(count.item(), 10)

    def test_single_batch_welford(self, verbose=False):
        # Welford from one batch should reproduce the batch's mean and m2 directly.
        buf = self._make_buffer(num_rollout_steps=1, num_channels=3, variable_shape=(4, 4))
        torch.manual_seed(0)
        data = torch.randn(8, 3, 4, 4)
        buf.update(data, idt=0)

        expected_mean = data.mean(dim=0)
        expected_m2 = (data - expected_mean).pow(2).sum(dim=0)

        with self.subTest(desc="mean"):
            self.assertTrue(compare_tensors("running_mean", buf.running_mean[0], expected_mean, atol=1e-5, verbose=verbose))
        with self.subTest(desc="m2"):
            self.assertTrue(compare_tensors("running_var", buf.running_var[0], expected_m2, atol=1e-4, verbose=verbose))
        with self.subTest(desc="count"):
            self.assertEqual(buf.num_samples_tracked[0].item(), 8)

    def test_two_batch_welford_matches_concatenated(self, verbose=False):
        # Online Welford on two unequal batches must match offline mean/m2 on
        # their concatenation. This is the core invariant.
        buf = self._make_buffer(num_rollout_steps=1, num_channels=2, variable_shape=(3, 5))
        torch.manual_seed(7)
        data1 = torch.randn(5, 2, 3, 5)
        data2 = torch.randn(7, 2, 3, 5)
        buf.update(data1, idt=0)
        buf.update(data2, idt=0)

        joined = torch.cat([data1, data2], dim=0)
        expected_mean = joined.mean(dim=0)
        expected_m2 = (joined - expected_mean).pow(2).sum(dim=0)

        with self.subTest(desc="mean"):
            self.assertTrue(compare_tensors("two-batch mean", buf.running_mean[0], expected_mean, atol=1e-5, verbose=verbose))
        with self.subTest(desc="m2"):
            self.assertTrue(compare_tensors("two-batch m2", buf.running_var[0], expected_m2, atol=1e-3, verbose=verbose))
        with self.subTest(desc="count"):
            self.assertEqual(buf.num_samples_tracked[0].item(), 12)

    def test_three_batch_welford(self, verbose=False):
        # More batches to confirm the recurrence holds beyond the easy 2-batch case.
        buf = self._make_buffer(num_rollout_steps=1, num_channels=1, variable_shape=(3, 3))
        torch.manual_seed(42)
        batches = [torch.randn(b, 1, 3, 3) for b in [3, 5, 7]]
        for b in batches:
            buf.update(b, idt=0)

        joined = torch.cat(batches, dim=0)
        expected_mean = joined.mean(dim=0)
        expected_m2 = (joined - expected_mean).pow(2).sum(dim=0)

        with self.subTest(desc="mean"):
            self.assertTrue(compare_tensors("three-batch mean", buf.running_mean[0], expected_mean, atol=1e-5, verbose=verbose))
        with self.subTest(desc="m2"):
            self.assertTrue(compare_tensors("three-batch m2", buf.running_var[0], expected_m2, atol=1e-3, verbose=verbose))
        with self.subTest(desc="count"):
            self.assertEqual(buf.num_samples_tracked[0].item(), 15)

    def test_rollout_steps_track_independently(self, verbose=False):
        # Each rollout step has its own running stats — updating one must
        # leave the others zero.
        buf = self._make_buffer(num_rollout_steps=3, num_channels=2, variable_shape=(4, 4))
        buf.update(torch.randn(4, 2, 4, 4), idt=1)

        with self.subTest(desc="counts"):
            self.assertEqual(buf.num_samples_tracked[0].item(), 0)
            self.assertEqual(buf.num_samples_tracked[1].item(), 4)
            self.assertEqual(buf.num_samples_tracked[2].item(), 0)
        with self.subTest(desc="mean step 0 untouched"):
            self.assertTrue(compare_tensors("step 0 mean", buf.running_mean[0], torch.zeros_like(buf.running_mean[0]), verbose=verbose))
        with self.subTest(desc="mean step 2 untouched"):
            self.assertTrue(compare_tensors("step 2 mean", buf.running_mean[2], torch.zeros_like(buf.running_mean[2]), verbose=verbose))

    def test_zero_buffers_resets_all_state(self, verbose=False):
        buf = self._make_buffer(num_rollout_steps=2, num_channels=2, variable_shape=(4, 4))
        buf.update(torch.randn(5, 2, 4, 4), idt=0)
        buf.update(torch.randn(5, 2, 4, 4), idt=1)
        self.assertGreater(buf.num_samples_tracked.sum().item(), 0)

        buf.zero_buffers()

        with self.subTest(desc="counts"):
            self.assertEqual(buf.num_samples_tracked.sum().item(), 0)
        with self.subTest(desc="running_mean reset"):
            self.assertTrue(compare_tensors("running_mean", buf.running_mean, torch.zeros_like(buf.running_mean), verbose=verbose))
        with self.subTest(desc="running_var reset"):
            self.assertTrue(compare_tensors("running_var", buf.running_var, torch.zeros_like(buf.running_var), verbose=verbose))


class TestTemporalAverageBufferUnit(unittest.TestCase):
    """
    In-memory unit tests for TemporalAverageBuffer that don't touch disk and
    don't depend on the dataset fixture. Complements ``TestRolloutBuffers``,
    which is the integration-style test that loads real HDF5 data and writes
    to an output file.
    """

    @staticmethod
    def _make_buffer(num_rollout_steps, channel_names, output_channels, img_shape, scale=None, bias=None):
        H, W = img_shape
        return TemporalAverageBuffer(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=1,
            img_shape=(H, W),
            local_shape=(H, W),
            local_offset=(0, 0),
            channel_names=channel_names,
            lat_lon=(list(range(H)), list(range(W))),
            device="cpu",
            scale=scale,
            bias=bias,
            output_channels=output_channels,
            output_file=None,
        )

    def test_update_subset_of_channels(self, verbose=False):
        # output_channels selects a subset; only those channels are accumulated.
        # 'a' (index 0) and 'c' (index 2) but not 'b' (index 1).
        channel_names = ["a", "b", "c"]
        output_channels = ["a", "c"]
        buf = self._make_buffer(
            num_rollout_steps=2,
            channel_names=channel_names,
            output_channels=output_channels,
            img_shape=(4, 8),
        )

        torch.manual_seed(1)
        data1 = torch.randn(3, 3, 4, 8)
        data2 = torch.randn(2, 3, 4, 8)
        buf.update(data1, idt=0)
        buf.update(data2, idt=0)

        joined = torch.cat([data1, data2], dim=0)
        sub = joined[:, [0, 2], :, :]
        expected_mean = sub.mean(dim=0)
        expected_m2 = (sub - expected_mean).pow(2).sum(dim=0)

        with self.subTest(desc="mean"):
            self.assertTrue(compare_tensors("subset mean", buf.running_mean[0], expected_mean, atol=1e-5, verbose=verbose))
        with self.subTest(desc="m2"):
            self.assertTrue(compare_tensors("subset m2", buf.running_var[0], expected_m2, atol=1e-3, verbose=verbose))
        with self.subTest(desc="count"):
            self.assertEqual(buf.num_samples_tracked[0].item(), 5)

    def test_scale_and_bias_applied_before_averaging(self, verbose=False):
        # scale and bias are applied per-channel before the Welford update.
        # With constant data == 1, the per-channel mean must be scale*1 + bias.
        channel_names = ["a", "b"]
        output_channels = ["a", "b"]
        scale = torch.tensor([2.0, 0.5])
        bias = torch.tensor([1.0, -1.0])
        buf = self._make_buffer(
            num_rollout_steps=1,
            channel_names=channel_names,
            output_channels=output_channels,
            img_shape=(2, 2),
            scale=scale,
            bias=bias,
        )

        data = torch.ones(4, 2, 2, 2)
        buf.update(data, idt=0)

        # mean per channel = scale*1 + bias = [3.0, -0.5], spatially constant.
        with self.subTest(desc="ch_a mean = scale*1 + bias"):
            self.assertTrue(compare_tensors("ch_a", buf.running_mean[0, 0], 3.0 * torch.ones(2, 2), atol=1e-6, verbose=verbose))
        with self.subTest(desc="ch_b mean = scale*1 + bias"):
            self.assertTrue(compare_tensors("ch_b", buf.running_mean[0, 1], -0.5 * torch.ones(2, 2), atol=1e-6, verbose=verbose))
        with self.subTest(desc="m2 zero for constant input"):
            # constant input → zero variance → m2 = 0
            self.assertTrue(compare_tensors("zero m2", buf.running_var[0], torch.zeros_like(buf.running_var[0]), verbose=verbose))

    def test_rollout_steps_independent(self, verbose=False):
        # Updates to different rollout steps must not contaminate each other.
        channel_names = ["a"]
        buf = self._make_buffer(
            num_rollout_steps=3,
            channel_names=channel_names,
            output_channels=channel_names,
            img_shape=(2, 2),
        )

        # Step 0: known constant 1.0
        buf.update(torch.ones(4, 1, 2, 2), idt=0)
        # Step 2: known constant 5.0
        buf.update(5.0 * torch.ones(4, 1, 2, 2), idt=2)

        with self.subTest(desc="step 0"):
            self.assertTrue(compare_tensors("step 0", buf.running_mean[0], torch.ones_like(buf.running_mean[0]), verbose=verbose))
        with self.subTest(desc="step 1 untouched"):
            self.assertTrue(compare_tensors("step 1", buf.running_mean[1], torch.zeros_like(buf.running_mean[1]), verbose=verbose))
        with self.subTest(desc="step 2"):
            self.assertTrue(compare_tensors("step 2", buf.running_mean[2], 5.0 * torch.ones_like(buf.running_mean[2]), verbose=verbose))
        with self.subTest(desc="step 1 count"):
            self.assertEqual(buf.num_samples_tracked[1].item(), 0)

    def test_finalize_no_output_file_runs_without_error(self, verbose=False):
        # finalize() with no output_file should compute std and return cleanly.
        # std = sqrt(running_var / (n - 1)) — sample-corrected.
        channel_names = ["a"]
        buf = self._make_buffer(
            num_rollout_steps=1,
            channel_names=channel_names,
            output_channels=channel_names,
            img_shape=(2, 2),
        )

        # Two-sample batch with values [-1, +1] → mean=0, m2 = 2 (two unit deltas).
        data = torch.tensor([[[-1.0, -1.0], [-1.0, -1.0]],
                             [[ 1.0,  1.0], [ 1.0,  1.0]]]).reshape(2, 1, 2, 2)
        buf.update(data, idt=0)

        with self.subTest(desc="post-update mean"):
            self.assertTrue(compare_tensors("mean", buf.running_mean[0], torch.zeros(2, 2), atol=1e-6, verbose=verbose))
        with self.subTest(desc="post-update m2"):
            self.assertTrue(compare_tensors("m2", buf.running_var[0], 2.0 * torch.ones(2, 2), atol=1e-6, verbose=verbose))

        # finalize is in-place math + (no-op write) — should not raise
        buf.finalize()


if __name__ == "__main__":
    unittest.main()
