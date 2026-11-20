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

"""What ``jit_mode`` compiles with ``torch.compile``.

- ``"none"``: nothing, everything runs eagerly.
- ``"metrics"``: the loss terms and the metrics, the model runs eagerly.
- ``"all"``: the model step too.

``"inductor"``, the name of ``"all"`` before the losses and metrics followed the mode, is still accepted
for now, with a warning.
"""

import warnings

JIT_MODES = ("none", "metrics", "all")

# old names, still accepted
_JIT_MODE_ALIASES = {"inductor": "all"}


def get_jit_mode(params) -> str:
    """The ``jit_mode`` of the parameters, ``"none"`` when they do not set it."""
    mode = params.get("jit_mode", "none")
    if mode in _JIT_MODE_ALIASES:
        warnings.warn(f"jit_mode={mode} is deprecated, use jit_mode={_JIT_MODE_ALIASES[mode]}", FutureWarning)
        mode = _JIT_MODE_ALIASES[mode]
    if mode not in JIT_MODES:
        raise ValueError(f"Unknown jit_mode {mode}, expected one of {', '.join(JIT_MODES)}")
    return mode


def compiles_model(params) -> bool:
    """Whether ``jit_mode`` compiles the model step."""
    return get_jit_mode(params) == "all"


def compiles_metrics(params) -> bool:
    """Whether ``jit_mode`` compiles the loss terms and the metrics."""
    return get_jit_mode(params) in ("metrics", "all")
