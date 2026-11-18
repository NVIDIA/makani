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

from makani.utils.grid_types import expected_latitudes, verify_grid_type


def parse_dataset_metadata(metadata_json_path, params):
    """Helper routine for parsing the metadata file data.json in the datasets."""

    try:
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)

        params["h5_path"] = metadata["h5_path"]
        params["dhours"] = metadata["dhours"]

        # load excluded list of timestamps if available
        params["analysis_epoch_start_dates"] = metadata.get("analysis_epoch_start_dates", [])

        # the grid type is required: it decides the quadrature, and a dataset
        # that does not say what grid it is on cannot be integrated over
        # correctly. Defaulting to equiangular is what this used to do, and it
        # is wrong silently for every dataset that is not.
        if "grid_type" not in metadata["coords"]:
            raise ValueError(
                f"{metadata_json_path} does not declare coords.grid_type. It is required: the quadrature "
                "weights follow from it, so a dataset that does not say what grid it is on cannot be area "
                "averaged correctly. Add one of 'equiangular', 'legendre-gauss', 'clenshaw-curtiss', "
                "'weatherbench2' or 'euclidean'; see data_process/examples/metadata.json."
            )
        params["data_grid_type"] = metadata["coords"]["grid_type"]

        if ("lat" in metadata["coords"]) and ("lon" in metadata["coords"]):
            params["lat"] = metadata["coords"]["lat"]
            params["lon"] = metadata["coords"]["lon"]

            # the coordinates are right here, so the declaration is checked
            # rather than taken on trust
            verify_grid_type(params["data_grid_type"], params["lat"], source=metadata_json_path)
        else:
            # no coordinates given, which is useful for dummy data experiments:
            # build them from the grid type the dataset does declare
            latitudes = expected_latitudes(params["data_grid_type"], params["img_shape_x"])
            if latitudes is None:
                raise ValueError(
                    f"{metadata_json_path} declares grid_type '{params['data_grid_type']}' and no "
                    "coordinates, so there is nothing to place the data on. Give coords.lat and coords.lon."
                )
            # stored north to south, as everything downstream expects
            params["lat"] = np.flip(latitudes).tolist()
            params["lon"] = np.linspace(start=0.0, stop=360.0, endpoint=False, num=params["img_shape_y"]).tolist()

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
        params["dataset"] = dict(
            name=metadata["dataset_name"],
            description=metadata["attrs"]["description"],
            metadata_file=params["metadata_json_path"],
        )

    except Exception as e:
        raise

    return params, metadata
