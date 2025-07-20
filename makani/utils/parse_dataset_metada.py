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

import json
import numpy as np


def parse_dataset_metadata(metadata_json_path, params):
    """Helper routine for parsing the metadata file data.json in the datasets."""

    try:
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)

        params["h5_path"] = metadata["h5_path"]
        params["dhours"] = metadata["dhours"]

        # read grid information: if not present, assume equiangular
        if ("lat" in metadata["coords"]) and ("lon" in metadata["coords"]):
            params["lat"] = metadata["coords"]["lat"]
            params["lon"] = metadata["coords"]["lon"]
            params["data_grid_type"] = metadata["coords"]["grid_type"]
        else:
            # create a dummy lat grid, useful for dummy data experiments
            params["lat"] = np.linspace(start=90.0, stop=-90.0, endpoint=True, num=params["img_shape_x"]).tolist()
            params["lon"] = np.linspace(start=0.0, stop=360.0, endpoint=False, num=params["img_shape_y"]).tolist()
            params["data_grid_type"] = "equiangular"

        # channel name sanitization step
        channel_names = metadata["coords"]["channel"]
        channels_idx = []
        if hasattr(params, "channel_names"):
            for pchn in params["channel_names"]:
                if pchn not in channel_names:
                    raise ValueError(f"Error, requested channel {pchn} not found in dataset.")
                else:
                    idx = channel_names.index(pchn)
                    channels_idx.append(idx)
        else:
            params["channel_names"] = channel_names
            channels_idx = list(range(len(channel_names)))

        # set number of channels
        params["in_channels"] = channels_idx
        params["out_channels"] = channels_idx

        # remember the channel names within the dataset if needed later
        params["data_channel_names"] = channel_names

        # get other metadata:
        params["dataset"] = dict(name=metadata["dataset_name"], description=metadata["attrs"]["description"], metadata_file=params["metadata_json_path"])

    except Exception as e:
        raise

    return params, metadata
