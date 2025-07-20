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
import datetime as dt
import numpy as np

from .aws_connector import AWSConnector


def get_data_normalization(params):

    bias = None
    scale = None

    if hasattr(params, "normalization"):
        if params.normalization == "minmax":
            mins = np.load(params.min_path)
            maxes = np.load(params.max_path)
            bias = mins
            scale = maxes - mins
        elif params.normalization == "zscore":
            means = np.load(params.global_means_path)
            stds = np.load(params.global_stds_path)
            bias = means
            scale = stds

        elif isinstance(params.normalization, dict):
            mins = np.load(params.min_path)
            maxes = np.load(params.max_path)
            means = np.load(params.global_means_path)
            stds = np.load(params.global_stds_path)

            bias = means
            scale = stds

            for ch in params.normalization.keys():
                c = params.data_channel_names.index(ch)
                if params.normalization[ch] == "minmax":
                    bias[:, c] = mins[:, c]
                    scale[:, c] = maxes[:, c] - mins[:, c]
                elif params.normalization[ch] == "zscore":
                    pass
                else:
                    raise ValueError(f"Unknown normalization mode {params.normalization[ch]}")

    return bias, scale


def get_climatology(params):
    """
    routine for fetching climatology and normalization factors
    """

    # compute climatology
    if params.enable_synthetic_data:
        clim = np.zeros([1, params.N_out_channels, params.img_crop_shape_x, params.img_crop_shape_y], dtype=np.float32)
    else:
        # full bias and scale
        bias, scale = get_data_normalization(params)
        bias = bias[:, params.out_channels, ...]
        scale = scale[:, params.out_channels, ...]

        # we need this window
        start_x = params.img_crop_offset_x
        end_x = start_x + params.img_crop_shape_x
        start_y = params.img_crop_offset_y
        end_y = start_y + params.img_crop_shape_y

        # now we crop the time means
        time_means = np.load(params.time_means_path)
        time_means = time_means[..., params.out_channels, start_x:end_x, start_y:end_y]
        clim = (time_means - bias) / scale

    return clim


# compute UTC timestamp fom year and hour into year.
def get_timestamp(year, hour):

    # compute timestamp for january 1st in UTC. Important, we need to specify a timezone here
    jan_01_epoch = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)

    # compute offset
    return jan_01_epoch + dt.timedelta(hours=hour)


def get_date_from_timestamp(timestamp):
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)

def get_timedelta_from_timestamp(timestamp):
    return dt.timedelta(seconds=timestamp)

def get_default_aws_connector(aws_session_token):
    return AWSConnector(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_region_name=os.getenv("AWS_DEFAULT_REGION"),
        aws_endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        aws_session_token=aws_session_token,
    )
