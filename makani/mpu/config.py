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

import os


class _DistributedConfig:
    """
    Module-level configuration for makani.mpu.
    Env vars are used as defaults but can be overridden programmatically, e.g.:

        from makani.mpu.config import config
        config.debug = True
    """

    def __init__(self):
        self._debug = None

    @property
    def debug(self):
        if self._debug is None:
            return os.getenv("MAKANI_DISTRIBUTED_DEBUG", "0") == "1"
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = bool(value)

    def __repr__(self):
        return f"_DistributedConfig(debug={self.debug})"

config = _DistributedConfig()
