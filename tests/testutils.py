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
import json
import datetime as dt
from typing import List, Optional
from packaging import version

import numpy as np
import h5py as h5

import torch

from makani.utils.YParams import ParamsBase

H5_PATH = "fields"
NUM_CHANNELS = 5
IMG_SIZE_H = 64
IMG_SIZE_W = 128
CHANNEL_NAMES = ["u10m", "t2m", "u500", "z500", "t500"]


def get_default_parameters():

    # instantiate parameters
    params = ParamsBase()

    # dataset related
    params.dt = 1
    params.n_history = 0
    params.n_future = 0
    params.normalization = "none"
    params.data_grid_type = "equiangular"
    params.model_grid_type = "equiangular"
    params.sht_grid_type = "legendre-gauss"

    params.resuming = False
    params.amp_mode = "none"
    params.jit_mode = "none"
    params.disable_ddp = False
    params.checkpointing_level = 0
    params.enable_synthetic_data = False
    params.split_data_channels = False

    # dataloader related
    params.in_channels = list(range(NUM_CHANNELS))
    params.out_channels = list(range(NUM_CHANNELS))
    params.channel_names = [CHANNEL_NAMES[i] for i in range(NUM_CHANNELS)]

    # number of channels
    params.N_in_channels = len(params.in_channels)
    params.N_out_channels = len(params.out_channels)

    params.batch_size = 1
    params.valid_autoreg_steps = 0
    params.num_data_workers = 1
    params.multifiles = True
    params.io_grid = [1, 1, 1]
    params.io_rank = [0, 0, 0]

    # extra channels
    params.add_grid = False
    params.add_zenith = False
    params.add_orography = False
    params.add_landmask = False
    params.add_soiltype = False

    # logging stuff, needed for higher level tests
    params.log_to_screen = False
    params.log_to_wandb = False

    return params


def init_dataset(
    path: str, 
    num_samples_per_year: Optional[int] = 365, 
    num_channels: Optional[int] = NUM_CHANNELS, 
    img_size_h: Optional[int] = IMG_SIZE_H, 
    img_size_w: Optional[int] = IMG_SIZE_W, 
    nan_fraction: Optional[float] = 0.0,
    annotate: Optional[bool] = True
):

    test_path = os.path.join(path, "test")
    os.makedirs(test_path)

    train_path = os.path.join(path, "train")
    os.makedirs(train_path)

    stats_path = os.path.join(path, "stats")
    os.makedirs(stats_path)

    metadata_path = os.path.join(path, "metadata")
    os.makedirs(metadata_path)

    # rng:
    rng = np.random.default_rng(seed=333)

    # create lon lat grid
    longitude = np.linspace(0, 360, img_size_w, endpoint=False)
    latitude = np.linspace(-90, 90, img_size_h, endpoint=True)
    latitude = latitude[::-1]

    # channels names
    channel_names = [f"chan_{idx}" for idx in range(num_channels)]
    chanlen = max([len(x) for x in channel_names])

    # set dhours:
    hours_per_year = 365 * 24
    dhours = hours_per_year // num_samples_per_year

    # create training files
    num_train = 0
    for y in [2017, 2018]:
        data_path = os.path.join(train_path, f"{y}.h5")
        with h5.File(data_path, "w") as hf:
            hf.create_dataset(H5_PATH, shape=(num_samples_per_year, num_channels, img_size_h, img_size_w))

            num_dof = num_samples_per_year * num_channels * img_size_h * img_size_w
            data = rng.random((num_dof,), dtype=np.float32)

            # add NaNs
            if nan_fraction > 0.0:
                indices = np.arange(num_samples_per_year * num_channels * img_size_h * img_size_w, dtype=np.int32)
                nan_count = int(nan_fraction * num_dof)
                rng.shuffle(indices)
                nan_indices = indices[0:nan_count]
                data[nan_indices] = np.nan

            # reshape to correct shape
            data = data.reshape(num_samples_per_year, num_channels, img_size_h, img_size_w)

            # store in file
            hf[H5_PATH][...] = data[...]

            # annotations
            if annotate:
                # create datasets
                year_start = dt.datetime(year=y, month=1, day=1, hour=0, tzinfo=dt.timezone.utc).timestamp()
                timestamps = year_start + np.arange(0, hours_per_year * 3600, dhours * 3600, dtype=np.float64)
                hf.create_dataset("timestamp", data=timestamps)
                hf.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
                hf["channel"][...] = channel_names
                hf.create_dataset("lat", data=latitude)
                hf.create_dataset("lon", data=longitude)
                # create scales
                hf["timestamp"].make_scale("timestamp")
                hf["channel"].make_scale("channel")
                hf["lat"].make_scale("lat")
                hf["lon"].make_scale("lon")
                # attach scales
                hf[H5_PATH].dims[0].attach_scale(hf["timestamp"])
                hf[H5_PATH].dims[1].attach_scale(hf["channel"])
                hf[H5_PATH].dims[2].attach_scale(hf["lat"])
                hf[H5_PATH].dims[3].attach_scale(hf["lon"])

        num_train += num_samples_per_year

    # create validation files
    num_test = 0
    for y in [2019]:
        data_path = os.path.join(test_path, f"{y}.h5")
        with h5.File(data_path, "w") as hf:
            hf.create_dataset(H5_PATH, shape=(num_samples_per_year, num_channels, img_size_h, img_size_w))
            hf[H5_PATH][...] = rng.random((num_samples_per_year, num_channels, img_size_h, img_size_w), dtype=np.float32)

            # annotations
            if annotate:
                # create datasets
                year_start = dt.datetime(year=y, month=1, day=1, hour=0, tzinfo=dt.timezone.utc).timestamp()
                timestamps = year_start + np.arange(0, hours_per_year * 3600, dhours * 3600, dtype=np.float64)
                hf.create_dataset("timestamp", data=timestamps)
                hf.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
                hf["channel"][...] = channel_names
                hf.create_dataset("lat", data=latitude)
                hf.create_dataset("lon", data=longitude)
                # create scales
                hf["timestamp"].make_scale("timestamp")
                hf["channel"].make_scale("channel")
                hf["lat"].make_scale("lat")
                hf["lon"].make_scale("lon")
                # attach scales
                hf[H5_PATH].dims[0].attach_scale(hf["timestamp"])
                hf[H5_PATH].dims[1].attach_scale(hf["channel"])
                hf[H5_PATH].dims[2].attach_scale(hf["lat"])
                hf[H5_PATH].dims[3].attach_scale(hf["lon"])

        num_test += num_samples_per_year

    # create stats files
    np.save(os.path.join(stats_path, "mins.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))

    np.save(os.path.join(stats_path, "maxs.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))

    np.save(os.path.join(stats_path, "time_means.npy"), np.zeros((1, num_channels, img_size_h, img_size_w), dtype=np.float64))

    np.save(os.path.join(stats_path, "global_means.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))

    np.save(os.path.join(stats_path, "global_stds.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))

    np.save(os.path.join(stats_path, "time_diff_means.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))

    np.save(os.path.join(stats_path, "time_diff_stds.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))

    # create metadata file:
    metadata = dict(dataset_name="testing",
                    h5_path=H5_PATH,
                    dims=["time", "channel", "lat", "lon"],
                    dhours=dhours,
                    coords=dict(
                        grid_type="equiangular",
                        lat=latitude.tolist(),
                        lon=longitude.tolist(),
                        channel=channel_names,
                    )
    )
    with open(os.path.join(metadata_path, "data.json"), "w") as f:
        json.dump(metadata, f)

    return train_path, num_train, test_path, num_test, stats_path, metadata_path

def compare_tensors(msg, tensor1, tensor2, atol=1e-8, rtol=1e-5, verbose=False):

    # some None checks
    if tensor1 is None and tensor2 is None:
        allclose = True
    elif tensor1 is None and tensor2 is not None:
        allclose = False
        if verbose:
            print(f"tensor1 is None and tensor2 is not None")
    elif tensor1 is not None and tensor2 is None:
        allclose = False
        if verbose:
            print(f"tensor1 is not None and tensor2 is None")
    else:
        diff = torch.abs(tensor1 - tensor2)
        abs_diff = torch.mean(diff, dim=0)
        rel_diff = torch.mean(diff / torch.clamp(torch.abs(tensor2), min=1e-6), dim=0)
        allclose = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
        if not allclose and verbose:
            print(f"Absolute difference on {msg}: min = {abs_diff.min()}, mean = {abs_diff.mean()}, max = {abs_diff.max()}")
            print(f"Relative difference on {msg}: min = {rel_diff.min()}, mean = {rel_diff.mean()}, max = {rel_diff.max()}")
            print(f"Element values with max difference on {msg}: {tensor1.flatten()[diff.argmax()]} and {tensor2.flatten()[diff.argmax()]}")
            # find violating entry
            worst_diff = torch.argmax(diff - (atol + rtol * torch.abs(tensor2)))
            diff_bad = diff.flatten()[worst_diff].item()
            tensor2_abs_bad = torch.abs(tensor2).flatten()[worst_diff].item()
            print(f"Worst allclose condition violation: {diff_bad} <= {atol} + {rtol} * {tensor2_abs_bad} = {atol + rtol * tensor2_abs_bad}")

    return allclose


def compare_arrays(msg, array1, array2, atol=1e-8, rtol=1e-5, verbose=False):
    # some None checks
    if array1 is None and array2 is None:
        allclose = True
    elif array1 is None and array2 is not None:
        allclose = False
        if verbose:
            print(f"array1 is None and array2 is not None")
    elif array1 is not None and array2 is None:
        allclose = False
        if verbose:
            print(f"array1 is not None and array2 is None")
    else:
        # some sanitization
        if array1.ndim == 0:
            array1 = array1.reshape(1)
        if array2.ndim == 0:
            array2 = array2.reshape(1)
        # compute error
        diff = np.abs(array1 - array2)
        abs_diff = np.mean(diff, axis=0)
        rel_diff = np.mean(diff / np.clip(np.abs(array2), a_min=1e-6, a_max=None), axis=0)
        allclose = np.allclose(array1, array2, atol=atol, rtol=rtol)
        if not allclose and verbose:
            print(f"Absolute difference on {msg}: min = {abs_diff.min()}, mean = {abs_diff.mean()}, max = {abs_diff.max()}")
            print(f"Relative difference on {msg}: min = {rel_diff.min()}, mean = {rel_diff.mean()}, max = {rel_diff.max()}")
            print(f"Element values with max difference on {msg}: {array1.flatten()[diff.argmax()]} and {array2.flatten()[diff.argmax()]}")
            # find violating entry
            worst_diff = np.argmax(diff - (atol + rtol * np.abs(array2)))
            diff_bad = diff.flatten()[worst_diff].item()
            array2_abs_bad = np.abs(array2).flatten()[worst_diff].item()
            print(f"Worst allclose condition violation: {diff_bad} <= {atol} + {rtol} * {array2_abs_bad} = {atol + rtol * array2_abs_bad}")

    return allclose

def disable_tf32():
    # the api for this was changed lately in pytorch
    if torch.cuda.is_available():
        if version.parse(torch.__version__) >= version.parse("2.9.0"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.rnn.fp32_precision = "ieee"
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    return