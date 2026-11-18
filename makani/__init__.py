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

__version__ = "0.4.0rc1"

# The trainers and the inferencer are imported on first use, so that importing any module of makani,
# e.g. a layer in makani.models.common, does not load the training and inference stack with it.
_LAZY_EXPORTS = {
    "Trainer": "makani.training",
    "AutoencoderTrainer": "makani.training",
    "EnsembleTrainer": "makani.training",
    "StochasticTrainer": "makani.training",
    "Inferencer": "makani.inference",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY_EXPORTS))
