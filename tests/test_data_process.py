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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature
from testutils import init_dataset, H5_PATH, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W

class TestAnnotateDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = cls.tmpdir.name

        # Create unannotated dataset
        path = os.path.join(tmp_path, "data")
        os.makedirs(path, exist_ok=True)
        cls.train_path, cls.num_train, cls.test_path, cls.num_test, _, cls.metadata_path = init_dataset(path, annotate=False)

        # Create reference dataset with annotations
        ref_path = os.path.join(tmp_path, "ref_data")
        os.makedirs(ref_path, exist_ok=True)
        cls.ref_train_path, cls.ref_num_train, cls.ref_test_path, cls.ref_num_test, _, _ = init_dataset(ref_path, annotate=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_annotate_dataset(self):
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
                self.assertTrue(np.allclose(f[H5_PATH][...], ref_f[H5_PATH][...]))

                # Check annotations
                self.assertTrue(np.allclose(f["timestamp"][...], ref_f["timestamp"][...]))
                self.assertTrue(np.allclose(f["lat"][...], ref_f["lat"][...]))
                self.assertTrue(np.allclose(f["lon"][...], ref_f["lon"][...]))
                self.assertEqual(f["channel"][...].tolist(), ref_f["channel"][...].tolist())

                # Check dimension labels
                self.assertEqual(f[H5_PATH].dims[0].label, "Timestamp in UTC time zone")
                self.assertEqual(f[H5_PATH].dims[1].label, "Channel name")
                self.assertEqual(f[H5_PATH].dims[2].label, "Latitude in degrees")
                self.assertEqual(f[H5_PATH].dims[3].label, "Longitude in degrees")

                # Check scales
                self.assertTrue(np.allclose(f[H5_PATH].dims[0]["timestamp"], ref_f[H5_PATH].dims[0]["timestamp"]))
                self.assertTrue(np.allclose(f[H5_PATH].dims[2]["lat"],ref_f[H5_PATH].dims[2]["lat"]))
                self.assertTrue(np.allclose(f[H5_PATH].dims[3]["lon"], ref_f[H5_PATH].dims[3]["lon"]))


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

        # Get list of files to concatenate
        train_files = sorted([f for f in os.listdir(self.train_path) if f.endswith(".h5")])
        years = [2017, 2018]  # Corresponding years for the files

        # Create output directory
        output_dir = self.train_path

        # Run concatenation
        output_file = "concatenated.h5v"
        concatenate(output_dir, output_file, metadata, train_files, years, dhoursrel=dhoursrel)

        # Compare concatenated file with original files
        with h5.File(os.path.join(output_dir, output_file), "r") as f_conc:
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
            self.assertEqual(current_pos, total_samples)

            # Verify metadata
            self.assertTrue(np.allclose(f_conc["lat"][...], metadata["coords"]["lat"]))
            self.assertTrue(np.allclose(f_conc["lon"][...], metadata["coords"]["lon"]))
            self.assertEqual([c.decode() for c in f_conc["channel"][...].tolist()], metadata["coords"]["channel"])

            # Verify dimension labels
            self.assertEqual(f_conc[H5_PATH].dims[0].label, "Timestamp in UTC time zone")
            self.assertEqual(f_conc[H5_PATH].dims[1].label, "Channel name")
            self.assertEqual(f_conc[H5_PATH].dims[2].label, "Latitude in degrees")
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

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    @parameterized.expand([8, 16], skip_on_empty=False)
    @unittest.skipUnless(importlib.util.find_spec("mpi4py") is not None, "mpi4py needs to be installed for this test")
    def test_get_stats(self, batch_size):
        # import necessary modules
        from data_process.get_stats import welford_combine, get_file_stats

        # Load metadata
        with open(os.path.join(self.metadata_path, "data.json"), "r") as f:
            metadata = json.load(f)

        # Get list of files to process
        train_files = sorted([os.path.join(self.train_path, f) for f in os.listdir(self.train_path) if f.endswith(".h5")])
        
        # Create quadrature rule
        quadrature_rule = grid_to_quadrature_rule("equiangular")
        quadrature = GridQuadrature(quadrature_rule, (IMG_SIZE_H, IMG_SIZE_W))

        # Get stats using get_file_stats
        stats = None
        for file_path in train_files:
            file_stats = get_file_stats(
                file_path,
                slice(0, None),  # Process entire file
                None,  # No wind indices
                quadrature,
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
        tdata = torch.from_numpy(all_data)
        
        # Compute means and variances using quadrature
        tmean = torch.mean(quadrature(tdata), keepdims=False, dim=0).reshape(1, -1, 1, 1)
        tvar = torch.mean(quadrature(torch.square(tdata - tmean)), keepdims=False, dim=0).reshape(1, -1, 1, 1)
        
        # Compute time differences
        tdiff = tdata[1:] - tdata[:-1]
        tdiffmean = torch.mean(quadrature(tdiff), keepdims=False, dim=0).reshape(1, -1, 1, 1)
        tdiffvar = torch.mean(quadrature(torch.square(tdiff - tdiffmean)), keepdims=False, dim=0).reshape(1, -1, 1, 1)

        # Compare results        
        self.assertTrue(np.allclose(stats["global_meanvar"]["values"][0], tmean.numpy()))
        self.assertTrue(np.allclose(stats["global_meanvar"]["values"][1], float(all_data.shape[0]) * tvar.numpy(), atol=0.0, rtol=1e-3))
        
        # this test is more tricky since it crosses file boundaries
        #self.assertTrue(np.allclose(stats["time_diff_meanvar"]["values"][0], tdiffmean.numpy()))
        #self.assertTrue(np.allclose(stats["time_diff_meanvar"]["values"][1], float(tdiff.shape[0]) * tdiffvar.numpy()))
        
        # Compare min/max
        self.assertTrue(np.allclose(stats["maxs"]["values"], np.max(all_data, keepdims=True, axis=(0, 2, 3))))
        self.assertTrue(np.allclose(stats["mins"]["values"], np.min(all_data, keepdims=True, axis=(0, 2, 3))))


if __name__ == "__main__":
    unittest.main() 
