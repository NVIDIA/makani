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

import os
import shutil
import numpy as np
import argparse as ap
import json
import glob

def main(desc_path, input_path, output_path):

    # open metadata file
    with open(args.metadata_file, 'r') as f:
        metadata = json.load(f)

    # read channel names
    channel_names = metadata['coords']['channel']

    # copy all stats files first
    for f in glob.iglob(os.path.join(input_path, "*.npy")):
        shutil.copyfile(f, os.path.join(output_path, os.path.basename(f)))

    print("Postprocessing minima:")
    # correct water channel minima to be exactly 0.0
    mins_file = os.path.join(output_path, "mins.npy")
    mins = np.load(mins_file)

    for c, chn in enumerate(channel_names):
        if chn.startswith("q") or chn == "tcwv":
            mins[0, c, 0, 0] = 0.0
    np.save(mins_file, mins)

    print("Clamping stds")
    # global
    stds_file = os.path.join(output_path, "global_stds.npy")
    stds = np.load(stds_file)
    stds = np.maximum(stds, 1e-4)
    np.save(stds_file, stds)
    # time diff
    stds_file =	os.path.join(output_path, "time_diff_stds.npy")
    stds = np.load(stds_file)
    stds = np.maximum(stds, 1e-4)
    np.save(stds_file, stds)


if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Directory with input stats files.", required=True)
    parser.add_argument("--metadata_file", type=str, help="File containing dataset metadata.", required=True)
    parser.add_argument("--output_path", type=str, help="Directory for saving stats files.", required=True)
    args = parser.parse_args()

    main(desc_path=args.metadata_file, input_path=args.input_path, output_path=args.output_path)
