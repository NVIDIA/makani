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

"""The WeatherBench2 / ARCO layout.

Defined alongside the makani zarr backend it derives from, since the two share
discovery and differ only in how a channel resolves to storage. Re-exported here
so the module layout matches the one backend per layout convention.
"""

from .makani_zarr import ArcoWB2Backend

__all__ = ["ArcoWB2Backend"]
