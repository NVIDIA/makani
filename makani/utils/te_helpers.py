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
transformer-engine availability helper.

``find_spec`` locates the package without executing it, so we can set the
availability flag at import time without paying the cost (or CUDA-context side
effects) of importing transformer_engine just to learn whether it exists. The
actual module is imported lazily, and cached, the first time it is needed.
"""

import importlib.util

# cheap, side-effect-free check: is the package installed? (does not import it)
TE_AVAILABLE = importlib.util.find_spec("transformer_engine") is not None

_te = None


def get_te():
    """Return the ``transformer_engine.pytorch`` module, importing it lazily on first
    use and caching it. Returns ``None`` if transformer_engine is not installed."""
    global _te
    if _te is None and TE_AVAILABLE:
        import transformer_engine.pytorch as te

        _te = te
    return _te
