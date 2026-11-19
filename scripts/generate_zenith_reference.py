# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Print the external reference values of ``tests/test_zenith_angle.py``.

The cosine of the solar zenith angle from pvlib's NREL SPA (``method="nrel_numpy"``, no
refraction), an ephemeris independent of makani's, for the points of ``_REFERENCE_POINTS``
in that file. Run it in a makani environment with pvlib and pandas, which makani does not depend on::

    pip install pvlib pandas
    python scripts/generate_zenith_reference.py
"""

import datetime as dt
import math
import os
import sys

import pandas as pd
import pvlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tests.test_zenith_angle import _REFERENCE_POINTS  # noqa: E402


def cos_zenith(time: dt.datetime, lat: float, lon: float) -> float:
    position = pvlib.solarposition.get_solarposition(pd.DatetimeIndex([time]), lat, lon, method="nrel_numpy")
    return math.cos(math.radians(position["zenith"].iloc[0]))


if __name__ == "__main__":
    for time, lat, lon, _, label in _REFERENCE_POINTS:
        print(f"{cos_zenith(time, lat, lon):.6f}  {label}")
