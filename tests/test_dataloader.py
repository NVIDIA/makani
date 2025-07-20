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

import os
import glob
import copy
import tempfile
import datetime as dt
from typing import List, Optional
from parameterized import parameterized

import unittest
import torch
import numpy as np
import h5py as h5

from makani.utils.dataloader import get_dataloader
from makani.utils.dataloaders.dali_es_helper_2d import GeneralES

from testutils import get_default_parameters, init_dataset
from testutils import H5_PATH, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W

_multifiles_params = [True]
_have_dali = True
try:
    import nvidia.dali
except:
    _have_dali = False

if _have_dali:
    _multifiles_params.append(False)


def get_sample(path: str, idx):
    files = sorted(glob.glob(os.path.join(path, "*.h5")))
    h5file = h5.File(files[0], "r")
    num_samples_per_file = h5file[H5_PATH].shape[0]
    h5file.close()
    file_id = idx // num_samples_per_file
    file_index = idx % num_samples_per_file

    with h5.File(files[file_id], "r") as f:
        data = f[H5_PATH][file_index, ...]

    return data


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


class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):

        cls.device = torch.device("cpu")

        # create temporary directory
        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name

        # init datasets and stats
        cls.train_path, cls.num_train, cls.valid_path, cls.num_valid, cls.stats_path, cls.metadata_path = init_dataset(tmp_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


    def setUp(self):

        self.params = init_dataset_params(self.train_path, self.valid_path, self.stats_path, batch_size=2, n_history=0, n_future=0, normalization="zscore", num_data_workers=1)

        self.params.multifiles = True
        self.params.num_train = self.num_train
        self.params.num_valid = self.num_valid

        # this is also correct for most cases:
        self.params.io_grid = [1, 1, 1]
        self.params.io_rank = [0, 0, 0]

        self.num_steps = 5


    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_shapes_and_sample_counts(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # create dataloaders
        valid_loader, valid_dataset, _ = get_dataloader(self.params, self.params.valid_data_path, mode="eval", device=self.device)

        # do tests
        num_valid_steps = self.params.num_valid // self.params.batch_size
        for idt, token in enumerate(valid_loader):
            self.assertEqual(len(token), 2)
            inp, tar = token

            self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
            self.assertEqual(tuple(tar.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))

        self.assertEqual((idt + 1), num_valid_steps)


    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_content(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # create dataloaders
        valid_loader, valid_dataset, _ = get_dataloader(self.params, self.params.valid_data_path, mode="eval", device=self.device)

        # do tests
        for idt, token in enumerate(valid_loader):
            # get loader samples
            inp, tar = token

            # get test samples
            off = self.params.batch_size * idt
            inp_res = []
            tar_res = []
            for b in range(self.params.batch_size):
                inp_res.append(get_sample(self.params.valid_data_path, off + b))
                tar_res.append(get_sample(self.params.valid_data_path, off + b + 1))
            test_inp = np.squeeze(np.stack(inp_res, axis=0))
            test_tar = np.squeeze(np.stack(tar_res, axis=0))

            inp = np.squeeze(inp.cpu().numpy())
            tar = np.squeeze(tar.cpu().numpy())

            self.assertTrue(np.allclose(inp, test_inp))
            self.assertTrue(np.allclose(tar, test_tar))

            if idt > self.num_steps:
                break

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_channel_ordering(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # create dataloaders
        valid_loader, valid_dataset, _ = get_dataloader(self.params, self.params.valid_data_path, mode="eval", device=self.device)

        # create flipped dataloader
        params = copy.deepcopy(self.params)
        params.in_channels = params.in_channels[::-1]
        params.out_channels = params.out_channels[::-1]
        valid_loader_flip, valid_dataset_flip, _ = get_dataloader(params, self.params.valid_data_path, mode="eval", device=self.device)

        for idt, (token, token_flip) in enumerate(zip(valid_loader, valid_loader_flip)):
            inp, tar = token
            inp_flip, tar_flip = token_flip

            self.assertFalse(torch.allclose(inp, inp_flip))
            inp_flip_flip = torch.flip(inp_flip, dims=(2,))
            self.assertTrue(torch.allclose(inp, inp_flip_flip))

            self.assertFalse(torch.allclose(tar, tar_flip))
            tar_flip_flip = torch.flip(tar_flip, dims=(2,))
            self.assertTrue(torch.allclose(tar, tar_flip_flip))

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_history(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # set history:
        self.params.n_history = 3

        # create dataloaders
        valid_loader, valid_dataset, _ = get_dataloader(self.params, self.params.valid_data_path, mode="eval", device=self.device)

        # do tests
        for idt, token in enumerate(valid_loader):
            self.assertEqual(len(token), 2)
            inp, tar = token

            self.assertEqual(tuple(inp.shape), (self.params.batch_size, self.params.n_history + 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
            self.assertEqual(tuple(tar.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))

            if idt > self.num_steps:
                break

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_future(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # set future:
        self.params.n_future = 3

        # create dataloaders
        train_loader, train_dataset, _ = get_dataloader(self.params, self.params.train_data_path, mode="train", device=self.device)

        # do tests
        for idt, token in enumerate(train_loader):
            self.assertEqual(len(token), 2)
            inp, tar = token

            self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
            self.assertEqual(tuple(tar.shape), (self.params.batch_size, self.params.n_future + 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))

            if idt > self.num_steps:
                break

    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_autoreg(self, multifiles):

        # set mutltifiles
        self.params.multifiles = multifiles

        # set autoreg
        self.params.valid_autoreg_steps = 3

        # create dataloaders
        valid_loader, valid_dataset, _ = get_dataloader(self.params, self.params.valid_data_path, mode="eval", device=self.device)

        # do tests
        for idt, token in enumerate(valid_loader):
            self.assertEqual(len(token), 2)
            inp, tar = token

            self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
            self.assertEqual(tuple(tar.shape), (self.params.batch_size, self.params.valid_autoreg_steps + 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))

            if idt > self.num_steps:
                break


    def test_date_retrieval(self):

        # this only works with the multifiles loader since we cannot access certain samples with the dali loader
        self.params.multifiles = True

        # create dataloaders
        train_loader, train_dataset, _ = get_dataloader(self.params, self.params.train_data_path, mode="train", device=self.device)

        # we know we have self.params.num_train per year, so we can perfectly estimate the date
        # we do not use leap years in that test so we always have 365 days
        dhours = 24

        # this is just a date in the first file
        time1 = train_loader.dataset.get_time_at_index(10)
        # since the stuff is 0 indexed, idx = 10 actually corresponds to day 11 in january
        time1_comp = dt.datetime(year=2017, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc) + dt.timedelta(hours = dhours * 10)
        self.assertEqual(time1, time1_comp)

        # this date should be in the second file
        time2 = train_loader.dataset.get_time_at_index(365)
        time2_comp = dt.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc)
        self.assertEqual(time2, time2_comp)


    def test_index_retrieval(self):

        # this only works with the multifiles loader since we cannot access certain samples with the dali loader
        self.params.multifiles = True

        # create dataloaders
        train_loader, train_dataset, _ = get_dataloader(self.params, self.params.train_data_path, mode="train", device=self.device)

        # we know we have self.params.num_train per year, so we can perfectly estimate the date
        # we do not use leap years in that test so we always have 365 days
        dhours = 24

        # this is just a date in the first file
        tstamp = dt.datetime(year=2017, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc) + dt.timedelta(hours = dhours * 10)
        time1_idx = train_loader.dataset.get_index_at_time(tstamp)
        # since the stuff is 0 indexed, idx = 10 actually corresponds to day 11 in january
        self.assertEqual(time1_idx, 10)

        # this date should be in the second file
        tstamp = dt.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc)
        time2_idx = train_loader.dataset.get_index_at_time(tstamp)
        self.assertEqual(time2_idx, 365)


    @parameterized.expand(_multifiles_params, skip_on_empty=False)
    def test_distributed(self, multifiles):

        # set multifiles
        self.params.multifiles = multifiles

        # set IO grid
        self.params.io_grid = [1, 2, 1]
        self.params.io_rank = [0, 1, 0]

        valid_loader, valid_dataset, _ = get_dataloader(self.params, self.params.valid_data_path, mode="eval", device=self.device)

        off_x = valid_dataset.img_local_offset_x
        off_y = valid_dataset.img_local_offset_y
        range_x = valid_dataset.img_local_shape_x
        range_y = valid_dataset.img_local_shape_y

        # do tests
        num_steps = 3
        for idt, token in enumerate(valid_loader):
            # get loader samples
            inp, tar = token

            self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, range_x, range_y))
            self.assertEqual(tuple(tar.shape), (self.params.batch_size, 1, NUM_CHANNELS, range_x, range_y))

            # get test samples
            off = self.params.batch_size * idt
            inp_res = []
            tar_res = []
            for b in range(self.params.batch_size):
                tmp = get_sample(self.params.valid_data_path, off + b)
                inp_res.append(tmp[:, off_x : off_x + range_x, off_y : off_y + range_y])
                tmp = get_sample(self.params.valid_data_path, off + b + 1)
                tar_res.append(tmp[:, off_x : off_x + range_x, off_y : off_y + range_y])

            # stack
            test_inp = np.squeeze(np.stack(inp_res, axis=0))
            test_tar = np.squeeze(np.stack(tar_res, axis=0))

            inp = np.squeeze(inp.cpu().numpy())
            tar = np.squeeze(tar.cpu().numpy())

            self.assertTrue(np.allclose(inp, test_inp))
            self.assertTrue(np.allclose(tar, test_tar))

            if idt > self.num_steps:
                break

if __name__ == "__main__":
    unittest.main()
