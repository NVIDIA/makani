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
import shutil
import numpy as np
import torch
import h5py as h5
from typing import Optional

from makani.utils.inference.rollout_buffer import TemporalAverageBuffer
from testutils import init_dataset, get_default_parameters, H5_PATH, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W


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
        torch.manual_seed(333)
        np.random.seed(333)

        # create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name

        # init datasets and stats using the same approach as test_dataloader.py
        cls.train_path, cls.num_train, cls.valid_path, cls.num_valid, cls.stats_path, cls.metadata_path = init_dataset(tmp_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up after test using class-level teardown"""
        cls.tmpdir.cleanup()

    def setUp(self):
        """Set up test parameters for each test method"""
        
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
        self.longitude = np.linspace(0, 360, IMG_SIZE_W, endpoint=False)
        self.latitude = np.linspace(90, -90, IMG_SIZE_H, endpoint=True)

        self.lat_lon = (self.latitude.tolist(), self.longitude.tolist())

    @parameterized.expand(
        [
            (1, 1, False), (2, 1, False), (4, 1, False), (1, 2, False), (2, 2, False), (4, 2, False),
            (1, 1, True), (2, 1, True), (4, 1, True), (1, 2, True), (2, 2, True), (4, 2, True),
        ],
        skip_on_empty=True,
    )
    def test_temporal_averaging_buffer(self, batch_size, num_rollout_steps, scale_bias):
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
        self.assertEqual(buffer_mean.shape, (num_rollout_steps, len(self.output_channels), *self.img_shape))
        self.assertEqual(buffer_std.shape, (num_rollout_steps, len(self.output_channels), *self.img_shape))
        self.assertEqual(len(lead_time), num_rollout_steps)
        self.assertEqual(len(channels), len(self.output_channels))
        self.assertEqual(len(lats), self.img_shape[0])
        self.assertEqual(len(lons), self.img_shape[1])

        # Verify channel names
        self.assertEqual(channels, self.output_channels)

        # Verify lead times
        expected_lead_times = np.arange(self.rollout_dt, (num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64)
        self.assertTrue(np.allclose(lead_time, expected_lead_times))

        # Verify lat/lon coordinates
        self.assertTrue(np.allclose(lats, self.latitude, atol=0.0, rtol=1e-6))
        self.assertTrue(np.allclose(lons, self.longitude, atol=0.0, rtol=1e-6))

        # Compare with buffer output
        self.assertTrue(np.allclose(buffer_mean, manual_mean, atol=0.0, rtol=1e-5))
        self.assertTrue(np.allclose(buffer_std, manual_std, atol=0.0, rtol=1e-5))


if __name__ == "__main__":
    unittest.main() 
