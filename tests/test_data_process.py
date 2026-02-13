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

import importlib.util
import sys
import os
import json
import tempfile
import unittest
import numpy as np
import h5py as h5
import datetime as dt
from parameterized import parameterized

import torch

from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import disable_tf32, init_dataset, H5_PATH, IMG_SIZE_H, IMG_SIZE_W, compare_arrays


class TestAnnotateDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = cls.tmpdir.name

        # Create unannotated dataset
        path = os.path.join(tmp_path, "data")
        os.makedirs(path, exist_ok=True)
        cls.train_path, cls.num_train, cls.test_path, cls.num_test, _, cls.metadata_path = init_dataset(path, nan_fraction=0.0, annotate=False)

        # Create reference dataset with annotations
        ref_path = os.path.join(tmp_path, "ref_data")
        os.makedirs(ref_path, exist_ok=True)
        cls.ref_train_path, cls.ref_num_train, cls.ref_test_path, cls.ref_num_test, _, _ = init_dataset(ref_path, annotate=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        disable_tf32()

    def test_annotate_dataset(self, verbose=False):
        # import necessary modules
        from data_process.annotate_dataset import annotate

        # Load metadata
        with open(os.path.join(self.metadata_path, "data.json"), "r") as f:
            metadata = json.load(f)

        # Get list of files to annotate
        train_files = sorted([os.path.join(self.train_path, f) for f in os.listdir(self.train_path) if f.endswith(".h5")])
        test_files = sorted([os.path.join(self.test_path, f) for f in os.listdir(self.test_path) if f.endswith(".h5")])
        all_files = train_files + test_files
        years = [2017, 2018, 2019]  # Corresponding years for the files

        # Run annotation
        annotate(metadata, all_files, years)

        # reference files:
        train_files_ref = sorted([os.path.join(self.ref_train_path, f) for f in os.listdir(self.ref_train_path) if f.endswith(".h5")])
        test_files_ref = sorted([os.path.join(self.ref_test_path, f) for f in os.listdir(self.ref_test_path) if f.endswith(".h5")])
        all_files_ref = train_files_ref + test_files_ref

        # Compare with reference dataset
        for file_path, ref_file_path in zip(all_files, all_files_ref):
            with h5.File(file_path, "r") as f, h5.File(ref_file_path, "r") as ref_f:
                # Check data content
                with self.subTest(desc="data"):
                    self.assertTrue(np.allclose(f[H5_PATH][...], ref_f[H5_PATH][...]))

                # Check annotations
                with self.subTest(desc="timestamp"):
                    self.assertTrue(compare_arrays("timestamp", f["timestamp"][...], ref_f["timestamp"][...], verbose=verbose))
                with self.subTest(desc="lat"):
                    self.assertTrue(compare_arrays("lat", f["lat"][...], ref_f["lat"][...], verbose=verbose))
                with self.subTest(desc="lon"):
                    self.assertTrue(compare_arrays("lon", f["lon"][...], ref_f["lon"][...], verbose=verbose))
                with self.subTest(desc="channel"):
                    self.assertEqual(f["channel"][...].tolist(), ref_f["channel"][...].tolist())

                # Check dimension labels
                with self.subTest(desc="timestamp label"):
                    self.assertEqual(f[H5_PATH].dims[0].label, "Timestamp in UTC time zone")
                with self.subTest(desc="channel label"):
                    self.assertEqual(f[H5_PATH].dims[1].label, "Channel name")
                with self.subTest(desc="latitude label"):
                    self.assertEqual(f[H5_PATH].dims[2].label, "Latitude in degrees")
                with self.subTest(desc="longitude label"):
                    self.assertEqual(f[H5_PATH].dims[3].label, "Longitude in degrees")

                # Check scales
                with self.subTest(desc="timestamp scale"):
                    self.assertTrue(compare_arrays("timestamp scale", f[H5_PATH].dims[0]["timestamp"][...], ref_f[H5_PATH].dims[0]["timestamp"][...], verbose=verbose))
                with self.subTest(desc="channel scale"):
                    self.assertTrue(compare_arrays("channel scale", f[H5_PATH].dims[2]["lat"][...], ref_f[H5_PATH].dims[2]["lat"][...], verbose=verbose))
                with self.subTest(desc="longitude scale"):
                    self.assertTrue(compare_arrays("longitude scale", f[H5_PATH].dims[3]["lon"][...], ref_f[H5_PATH].dims[3]["lon"][...], verbose=verbose))


class TestConcatenateDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = cls.tmpdir.name

        # Create dataset
        path = os.path.join(tmp_path, "data")
        os.makedirs(path, exist_ok=True)
        cls.train_path, cls.num_train, cls.test_path, cls.num_test, _, cls.metadata_path = init_dataset(path, annotate=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        disable_tf32()

    @parameterized.expand(
        [1, 5],
        skip_on_empty=False,
    )
    def test_concatenate_dataset(self, dhoursrel):
        # import necessary modules
        from data_process.concatenate_dataset import concatenate

        # Load metadata
        with open(os.path.join(self.metadata_path, "data.json"), "r") as f:
            metadata = json.load(f)
        channel_names = metadata["coords"]["channel"]

        # Get list of files to concatenate
        train_files = sorted([f for f in os.listdir(self.train_path) if f.endswith(".h5")])
        years = [2017, 2018]  # Corresponding years for the files

        # Create output directory
        input_dirs = [self.train_path]

        # Run concatenation
        output_file = os.path.join(input_dirs[0], "concatenated.h5v")
        concatenate(input_dirs, output_file, metadata, [channel_names], train_files, years, dhoursrel=dhoursrel)

        # Compare concatenated file with original files
        with h5.File(output_file, "r") as f_conc:
            # Get total number of samples
            total_samples = f_conc[H5_PATH].shape[0]
            
            # Track current position in concatenated file
            current_pos = 0
            
            # Compare each original file's data with corresponding section in concatenated file
            for file_path in train_files:
                ifile_path = os.path.join(self.train_path, file_path)
                with h5.File(ifile_path, "r") as f_orig:
                    num_samples = f_orig[H5_PATH].shape[0] // dhoursrel
                    
                    # Compare data
                    self.assertTrue(np.allclose(
                        f_conc[H5_PATH][current_pos:current_pos + num_samples, ...],
                        f_orig[H5_PATH][::dhoursrel, ...]
                    ))
                    
                    # Compare timestamps
                    self.assertTrue(np.allclose(
                        f_conc["timestamp"][current_pos:current_pos + num_samples, ...],
                        f_orig["timestamp"][::dhoursrel, ...]
                    ))
                    
                    # Update position
                    current_pos += num_samples

            # Verify total number of samples
            with self.subTest(desc="total number of samples"):
                self.assertEqual(current_pos, total_samples)

            # Verify metadata
            with self.subTest(desc="lat"):
                self.assertTrue(np.allclose(f_conc["lat"][...], metadata["coords"]["lat"]))
            with self.subTest(desc="lon"):
                self.assertTrue(np.allclose(f_conc["lon"][...], metadata["coords"]["lon"]))
            with self.subTest(desc="channel"):
                self.assertEqual([c.decode() for c in f_conc["channel"][...].tolist()], metadata["coords"]["channel"])

            # Verify dimension labels
            with self.subTest(desc="timestamp label"):
                self.assertEqual(f_conc[H5_PATH].dims[0].label, "Timestamp in UTC time zone")
            with self.subTest(desc="channel label"):
                self.assertEqual(f_conc[H5_PATH].dims[1].label, "Channel name")
            with self.subTest(desc="latitude label"):
                self.assertEqual(f_conc[H5_PATH].dims[2].label, "Latitude in degrees")
            with self.subTest(desc="longitude label"):
                self.assertEqual(f_conc[H5_PATH].dims[3].label, "Longitude in degrees")


class TestGetStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = cls.tmpdir.name

        # Create dataset
        path = os.path.join(tmp_path, "data")
        os.makedirs(path, exist_ok=True)
        cls.train_path, cls.num_train, cls.test_path, cls.num_test, _, cls.metadata_path = init_dataset(path, annotate=True)

        # Create dataset with annotations and NaNs:
        nan_path = os.path.join(tmp_path, "nan_data")
        os.makedirs(nan_path, exist_ok=True)
        cls.nan_train_path, cls.nan_num_train, cls.nan_test_path, cls.nan_num_test, _, _ = init_dataset(nan_path, nan_fraction=0.1, annotate=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    @parameterized.expand(
        [
            (8, False),
            (16, False),
            (8, True),
            (16, True),
        ], skip_on_empty=False
    )
    @unittest.skipUnless(importlib.util.find_spec("mpi4py") is not None, "mpi4py needs to be installed for this test")
    def test_get_stats(self, batch_size, allow_nan, verbose=True):
        # import necessary modules
        from data_process.get_stats import welford_combine, get_file_stats, mask_data

        # Get list of files to process
        if allow_nan:
            train_files = sorted([os.path.join(self.nan_train_path, f) for f in os.listdir(self.nan_train_path) if f.endswith(".h5")])
        else:
            train_files = sorted([os.path.join(self.train_path, f) for f in os.listdir(self.train_path) if f.endswith(".h5")])
        
        # Create quadrature rule
        quadrature_rule = grid_to_quadrature_rule("equiangular")
        quadrature = GridQuadrature(quadrature_rule, (IMG_SIZE_H, IMG_SIZE_W), normalize=False)

        # Get stats using get_file_stats
        stats = None
        for file_path in train_files:
            file_stats = get_file_stats(
                filename=file_path,
                file_slice=slice(0, None),  # Process entire file
                wind_indices=None,  # No wind indices
                quadrature=quadrature,
                fail_on_nan=not allow_nan,
                dt=1,
                batch_size=batch_size,
            )
            if stats is None:
                stats = file_stats
            else:
                stats = welford_combine(stats, file_stats)

        # Compute stats naively by loading entire dataset
        all_data = []
        for file_path in train_files:
            with h5.File(file_path, 'r') as f:
                data = f[H5_PATH][...].astype(np.float64)
                all_data.append(data)
        all_data = np.concatenate(all_data, axis=0)
        
        # Convert to torch tensor for quadrature
        tdata = torch.as_tensor(all_data)
        tdata_masked, valid_mask = mask_data(tdata)
        valid_count = torch.sum(quadrature(valid_mask), dim=0).reshape(1, -1, 1, 1)

        # Compute means and variances using quadrature
        tmean = torch.sum(quadrature(tdata_masked * valid_mask), keepdims=False, dim=0).reshape(1, -1, 1, 1) / valid_count
        tm2 = torch.sum(quadrature(torch.square(tdata_masked - tmean) * valid_mask), keepdims=False, dim=0).reshape(1, -1, 1, 1)

        # Compute time differences
        tdiff = tdata[1:] - tdata[:-1]
        tdiff_masked, tdiff_valid_mask = mask_data(tdiff)
        tdiff_valid_count = torch.sum(quadrature(tdiff_valid_mask), dim=0).reshape(1, -1, 1, 1)
        tdiffmean = torch.sum(quadrature(tdiff_masked * tdiff_valid_mask), keepdims=False, dim=0).reshape(1, -1, 1, 1) / tdiff_valid_count
        tdiffvar = torch.sum(quadrature(torch.square(tdiff_masked - tdiffmean) * tdiff_valid_mask), keepdims=False, dim=0).reshape(1, -1, 1, 1) / tdiff_valid_count

        # Compare results
        with self.subTest(desc="mean"):
            self.assertTrue(compare_arrays("mean", stats["global_meanvar"]["values"][0].numpy(), tmean.numpy(), verbose=verbose))
        with self.subTest(desc="m2"):
            self.assertTrue(compare_arrays("m2", stats["global_meanvar"]["values"][1].numpy(), tm2.numpy(), verbose=verbose))

        # Compare min/max
        with self.subTest(desc="max"):
            self.assertTrue(compare_arrays("max", stats["maxs"]["values"].numpy(), np.nanmax(all_data, keepdims=True, axis=(0, 2, 3)), verbose=verbose))
        with self.subTest(desc="min"):
            self.assertTrue(compare_arrays("min", stats["mins"]["values"].numpy(), np.nanmin(all_data, keepdims=True, axis=(0, 2, 3)), verbose=verbose))


if __name__ == "__main__":
    unittest.main() 
