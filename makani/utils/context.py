# SPDX-FileCopyrightText: Copyright (c) 20245 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from torch import nn
from typing import Optional
from contextlib import contextmanager

@contextmanager
def rng_context(cpu_rng: torch.Generator, device_rng: Optional[torch.Generator] = None):
    """
    Context manager for temporarily setting CPU and device RNG states.

    This context manager allows you to temporarily set specific RNG states
    for reproducibility, then automatically restore the original global states.

    Parameters
    ----------
    cpu_rng_state : torch.Tensor
        CPU RNG state to set temporarily
    device_rng_state : torch.Tensor, optional
        Device (CUDA) RNG state to set temporarily. Uses current device.

    Examples
    --------
    >>> # Save current states
    >>> cpu_state = torch.get_rng_state()
    >>> device_state = torch.cuda.get_rng_state()
    >>>
    >>> # Later, temporarily use those states
    >>> with rng_context(cpu_state, device_state):
    >>>     # Code here uses the provided RNG states
    >>>     x = torch.randn(10)
    >>> # Original RNG states are restored here
    """

    # Backup and set CPU RNG state
    cpu_backup = torch.get_rng_state()
    torch.set_rng_state(cpu_rng.get_state())

    # Backup and set device RNG state if provided
    device_backup = None
    if device_rng is not None and torch.cuda.is_available():
        device_backup = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(device_rng.get_state())
    try:
        yield

    finally:
        # Restore states
        cpu_rng.set_state(torch.get_rng_state())
        torch.set_rng_state(cpu_backup)
        if device_backup is not None:
            device_rng.set_state(torch.cuda.get_rng_state())
            torch.cuda.set_rng_state(device_backup)
