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

from packaging import version
import os
import re
import json
import datetime as dt
import random
from typing import List, Optional

import numpy as np
import h5py as h5
import zarr

import torch

from makani.utils.YParams import ParamsBase

H5_PATH = "fields"
NUM_CHANNELS = 5
IMG_SIZE_H = 64
IMG_SIZE_W = 128
CHANNEL_NAMES = ["u10m", "t2m", "u500", "z500", "t500"]

# Dataset layout used by init_hdf5_dataset
NUM_SAMPLES_PER_YEAR = 365
TRAIN_YEARS = [2017, 2018]
TEST_YEARS  = [2019]
DHOURS = (365 * 24) // NUM_SAMPLES_PER_YEAR  # 24


def set_seed(seed=333):
    """Seed torch, torch.cuda, and numpy for reproducible tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return


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

    # preprocessor fields — required by Preprocessor2D; tests that need different
    # values override these in their own setUp after calling get_default_parameters()
    params.img_shape_x = IMG_SIZE_H
    params.img_shape_y = IMG_SIZE_W
    params.img_shape_x_resampled = IMG_SIZE_H
    params.img_shape_y_resampled = IMG_SIZE_W
    params.history_normalization_mode = "none"
    params.history_normalization_decay = 0.5
    params.normalize_residual = False

    return params


def init_dataset_metadata(
    path: str,
    dataset_name: str = "testing",
    h5_path: str = H5_PATH,
    dhours: int = DHOURS,
    channel_names: Optional[List[str]] = None,
    lat: Optional[List[float]] = None,
    lon: Optional[List[float]] = None,
    grid_type: str = "equiangular",
    attrs: Optional[dict] = None,
    analysis_epoch_start_dates: Optional[List[str]] = None,
) -> str:
    """Write a dataset metadata JSON to <path>/data.json and return the file path.

    This is the canonical way to create a metadata fixture for tests.  It is
    also called internally by ``init_hdf5_dataset`` so that both produce identical
    JSON layouts (including the ``attrs`` key required by
    ``parse_dataset_metadata``).
    """
    if channel_names is None:
        channel_names = [f"chan_{i}" for i in range(NUM_CHANNELS)]
    if lat is None:
        lat = np.linspace(90, -90, IMG_SIZE_H, endpoint=True).tolist()
    if lon is None:
        lon = np.linspace(0, 360, IMG_SIZE_W, endpoint=False).tolist()
    if attrs is None:
        attrs = {"description": "A synthetic test dataset."}

    metadata = dict(
        dataset_name=dataset_name,
        h5_path=h5_path,
        dims=["time", "channel", "lat", "lon"],
        dhours=dhours,
        attrs=attrs,
        coords=dict(
            grid_type=grid_type,
            lat=lat,
            lon=lon,
            channel=channel_names,
        ),
    )
    if analysis_epoch_start_dates is not None:
        metadata["analysis_epoch_start_dates"] = analysis_epoch_start_dates

    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "data.json")
    with open(file_path, "w") as f:
        json.dump(metadata, f)
    return file_path


def init_hdf5_dataset(
    path: str,
    num_samples_per_year: Optional[int] = 365,
    num_channels: Optional[int] = NUM_CHANNELS,
    img_size_h: Optional[int] = IMG_SIZE_H,
    img_size_w: Optional[int] = IMG_SIZE_W,
    nan_fraction: Optional[float] = 0.0,
    annotate: Optional[bool] = True,
    create_concat: Optional[bool] = False,
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
    latitude = np.linspace(90, -90, img_size_h, endpoint=True)

    # channels names
    channel_names = [f"chan_{idx}" for idx in range(num_channels)]
    chanlen = max([len(x) for x in channel_names])

    # set dhours:
    hours_per_year = 365 * 24
    dhours = hours_per_year // num_samples_per_year

    # storage for the concatenated training file (populated when create_concat=True)
    concat_data_list = []
    concat_ts_list   = []

    # create training files
    num_train = 0
    for y in [2017, 2018]:
        data_path = os.path.join(train_path, f"{y}.h5")
        with h5.File(data_path, "w") as hf:
            hf.create_dataset(H5_PATH, shape=(num_samples_per_year, num_channels, img_size_h, img_size_w), dtype="f4")

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

            # compute timestamps for this year (used for both annotation and concat)
            year_start = dt.datetime(year=y, month=1, day=1, hour=0, tzinfo=dt.timezone.utc).timestamp()
            timestamps = year_start + np.arange(0, hours_per_year * 3600, dhours * 3600, dtype=np.float64)

            # annotations
            if annotate:
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

            # collect for the concatenated file
            if create_concat:
                concat_data_list.append(data.copy())
                concat_ts_list.append(timestamps)

        num_train += num_samples_per_year

    # create validation files
    num_test = 0
    for y in [2019]:
        data_path = os.path.join(test_path, f"{y}.h5")
        with h5.File(data_path, "w") as hf:
            hf.create_dataset(H5_PATH, shape=(num_samples_per_year, num_channels, img_size_h, img_size_w), dtype="f4")
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

    # create concatenated training file (all years in one HDF5)
    if create_concat:
        concat_train_path = os.path.join(path, "train_concat.h5")
        concat_data = np.concatenate(concat_data_list, axis=0)
        concat_ts   = np.concatenate(concat_ts_list)
        with h5.File(concat_train_path, "w") as hf:
            hf.create_dataset(H5_PATH, data=concat_data)
            hf.create_dataset("timestamp", data=concat_ts)
            hf.create_dataset("channel", len(channel_names), dtype=h5.string_dtype(length=chanlen))
            hf["channel"][...] = channel_names
            hf.create_dataset("lat", data=latitude)
            hf.create_dataset("lon", data=longitude)
            hf["timestamp"].make_scale("timestamp")
            hf["channel"].make_scale("channel")
            hf["lat"].make_scale("lat")
            hf["lon"].make_scale("lon")
            hf[H5_PATH].dims[0].attach_scale(hf["timestamp"])
            hf[H5_PATH].dims[1].attach_scale(hf["channel"])
            hf[H5_PATH].dims[2].attach_scale(hf["lat"])
            hf[H5_PATH].dims[3].attach_scale(hf["lon"])
    else:
        concat_train_path = None

    # create metadata file:
    init_dataset_metadata(
        path=metadata_path,
        dataset_name="testing",
        h5_path=H5_PATH,
        dhours=dhours,
        channel_names=channel_names,
        lat=latitude.tolist(),
        lon=longitude.tolist(),
    )

    return train_path, num_train, test_path, num_test, stats_path, metadata_path, concat_train_path


def init_zarr_dataset(
    path: str,
    num_samples_per_year: Optional[int] = 365,
    num_channels: Optional[int] = NUM_CHANNELS,
    img_size_h: Optional[int] = IMG_SIZE_H,
    img_size_w: Optional[int] = IMG_SIZE_W,
    nan_fraction: Optional[float] = 0.0,
    annotate: Optional[bool] = True,
    consolidate: Optional[bool] = True,
):
    """Create a zarr-backed dummy dataset with the same layout as init_hdf5_dataset.

    Files are written as ``<year>.zarr`` directories (matching the glob pattern
    the dataloader uses).  When ``annotate=True`` a ``"time"`` coordinate array
    is written at the group level so that ``_zarr_read_timestamps`` picks it up
    via the primary lookup path.  ``consolidate=True`` calls
    ``zarr.consolidate_metadata`` so the ``open_consolidated`` fast path is
    exercised in the dataloader.

    Returns the same tuple as ``init_hdf5_dataset`` (with ``concat_train_path=None``
    since the concat path is HDF5-only).
    """
    test_path = os.path.join(path, "test_zarr")
    os.makedirs(test_path)

    train_path = os.path.join(path, "train_zarr")
    os.makedirs(train_path)

    stats_path = os.path.join(path, "stats_zarr")
    os.makedirs(stats_path)

    metadata_path = os.path.join(path, "metadata_zarr")
    os.makedirs(metadata_path)

    rng = np.random.default_rng(seed=333)

    longitude = np.linspace(0, 360, img_size_w, endpoint=False)
    latitude = np.linspace(90, -90, img_size_h, endpoint=True)
    channel_names = [f"chan_{idx}" for idx in range(num_channels)]

    hours_per_year = 365 * 24
    dhours = hours_per_year // num_samples_per_year

    def _write_year(store_path, year, data):
        g = zarr.open_group(store_path, mode="w")
        g.create_dataset(H5_PATH, data=data, chunks=(1, num_channels, img_size_h, img_size_w))
        if annotate:
            year_start = dt.datetime(year=year, month=1, day=1, hour=0, tzinfo=dt.timezone.utc).timestamp()
            timestamps = year_start + np.arange(0, hours_per_year * 3600, dhours * 3600, dtype=np.float64)
            g.create_dataset("time", data=timestamps)
            g.create_dataset("channel", data=np.array(channel_names))
            g.create_dataset("lat", data=latitude)
            g.create_dataset("lon", data=longitude)
        if consolidate:
            zarr.consolidate_metadata(store_path)

    num_train = 0
    for y in [2017, 2018]:
        num_dof = num_samples_per_year * num_channels * img_size_h * img_size_w
        data = rng.random((num_dof,), dtype=np.float32).reshape(num_samples_per_year, num_channels, img_size_h, img_size_w)
        if nan_fraction > 0.0:
            n_nan = int(nan_fraction * num_dof)
            idx = rng.choice(num_dof, size=n_nan, replace=False)
            data.flat[idx] = np.nan
        _write_year(os.path.join(train_path, f"{y}.zarr"), y, data)
        num_train += num_samples_per_year

    num_test = 0
    for y in [2019]:
        data = rng.random((num_samples_per_year, num_channels, img_size_h, img_size_w), dtype=np.float32)
        _write_year(os.path.join(test_path, f"{y}.zarr"), y, data)
        num_test += num_samples_per_year

    # stats files are format-agnostic — reuse the same .npy layout
    np.save(os.path.join(stats_path, "mins.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "maxs.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_means.npy"), np.zeros((1, num_channels, img_size_h, img_size_w), dtype=np.float64))
    np.save(os.path.join(stats_path, "global_means.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "global_stds.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_diff_means.npy"), np.zeros((1, num_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_diff_stds.npy"), np.ones((1, num_channels, 1, 1), dtype=np.float64))

    init_dataset_metadata(
        path=metadata_path,
        dataset_name="testing_zarr",
        h5_path=H5_PATH,
        dhours=dhours,
        channel_names=channel_names,
        lat=latitude.tolist(),
        lon=longitude.tolist(),
    )

    return train_path, num_train, test_path, num_test, stats_path, metadata_path, None


def init_wb2_zarr_dataset(
    path: str,
    channel_names: Optional[List[str]] = None,
    num_samples_per_year: Optional[int] = 365,
    img_size_h: Optional[int] = IMG_SIZE_H,
    img_size_w: Optional[int] = IMG_SIZE_W,
    annotate: Optional[bool] = True,
    consolidate: Optional[bool] = True,
):
    """Create a WB2-layout zarr dataset for testing.

    Mirrors the structure of real WeatherBench2 zarr stores:
      - one zarr array per variable (no single stacked ``fields`` array)
      - surface variables: shape ``(time, latitude, longitude)``
      - atmospheric variables: shape ``(time, level, latitude, longitude)``
      - ``time`` coordinate stored as ``datetime64[ns]``
      - ``latitude``, ``longitude``, ``level`` coordinates at group level

    ``channel_names`` defaults to ``CHANNEL_NAMES`` which contains both surface
    (u10m, t2m) and atmospheric (u500, z500, t500) variables so both code paths
    are exercised.

    Returns the same tuple as ``init_hdf5_dataset``/``init_zarr_dataset``
    (with ``concat_path=None``).
    """
    from makani.utils.dataloaders.wb2_helpers import surface_variables, atmospheric_variables

    if channel_names is None:
        channel_names = list(CHANNEL_NAMES)

    test_path = os.path.join(path, "test_zarr_wb2")
    os.makedirs(test_path)
    train_path = os.path.join(path, "train_zarr_wb2")
    os.makedirs(train_path)
    stats_path = os.path.join(path, "stats_zarr_wb2")
    os.makedirs(stats_path)
    metadata_path = os.path.join(path, "metadata_zarr_wb2")
    os.makedirs(metadata_path)

    rng = np.random.default_rng(seed=333)
    longitude = np.linspace(0, 360, img_size_w, endpoint=False).astype(np.float32)
    latitude = np.linspace(90, -90, img_size_h, endpoint=True).astype(np.float32)

    hours_per_year = 365 * 24
    dhours = hours_per_year // num_samples_per_year

    # parse channel_names to determine which WB2 variables are needed and at what levels
    atm_vars = {}    # wb2_name -> set of required pressure levels
    surf_vars = set()

    for ch_name in channel_names:
        m = re.search(r"[0-9]{1,4}$", ch_name)
        if m is not None and ch_name != "d2":
            pressure = int(m.group())
            prefix = ch_name[: m.start()]
            wb2_name = atmospheric_variables[prefix]
            atm_vars.setdefault(wb2_name, set()).add(pressure)
        else:
            surf_vars.add(surface_variables[ch_name])

    # unified sorted level array covering all atmospheric variables in the fixture
    all_levels = sorted(set().union(*atm_vars.values())) if atm_vars else []
    levels = np.array(all_levels, dtype=np.int64)
    n_levels = len(levels)

    def _write_year(store_path, year):
        g = zarr.open_group(store_path, mode="w")

        if annotate:
            year_start = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
            times = np.array(
                [np.datetime64(int((year_start + dt.timedelta(hours=i * dhours)).timestamp() * 1e9), "ns")
                 for i in range(num_samples_per_year)]
            )
            g.create_dataset("time", data=times)
            g.create_dataset("latitude", data=latitude)
            g.create_dataset("longitude", data=longitude)
            if n_levels > 0:
                g.create_dataset("level", data=levels)

        for wb2_name in surf_vars:
            data = rng.random((num_samples_per_year, img_size_h, img_size_w), dtype=np.float32)
            g.create_dataset(wb2_name, data=data, chunks=(1, img_size_h, img_size_w))

        for wb2_name in atm_vars:
            data = rng.random((num_samples_per_year, n_levels, img_size_h, img_size_w), dtype=np.float32)
            g.create_dataset(wb2_name, data=data, chunks=(1, n_levels, img_size_h, img_size_w))

        if consolidate:
            zarr.consolidate_metadata(store_path)

    num_train = 0
    for y in [2017, 2018]:
        _write_year(os.path.join(train_path, f"{y}.zarr"), y)
        num_train += num_samples_per_year

    num_test = 0
    for y in [2019]:
        _write_year(os.path.join(test_path, f"{y}.zarr"), y)
        num_test += num_samples_per_year

    n_channels = len(channel_names)
    np.save(os.path.join(stats_path, "mins.npy"), np.zeros((1, n_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "maxs.npy"), np.ones((1, n_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_means.npy"), np.zeros((1, n_channels, img_size_h, img_size_w), dtype=np.float64))
    np.save(os.path.join(stats_path, "global_means.npy"), np.zeros((1, n_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "global_stds.npy"), np.ones((1, n_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_diff_means.npy"), np.zeros((1, n_channels, 1, 1), dtype=np.float64))
    np.save(os.path.join(stats_path, "time_diff_stds.npy"), np.ones((1, n_channels, 1, 1), dtype=np.float64))

    init_dataset_metadata(
        path=metadata_path,
        dataset_name="testing_wb2",
        h5_path=H5_PATH,
        dhours=dhours,
        channel_names=channel_names,
        lat=latitude.tolist(),
        lon=longitude.tolist(),
    )

    return train_path, num_train, test_path, num_test, stats_path, metadata_path, None


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