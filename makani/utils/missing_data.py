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

"""Handling of missing values in targets, shared by the losses and the metrics.

NaN is the single marker for missing data throughout makani: the converters
write it for imputed samples and skipped channels, the yearly HDF5 files
declare it as their fill value, and fields that are only defined over part of
the globe, sea surface temperature over land say, carry it as well.

How missing points are then kept out of a computation depends on its consumer:

- the metrics work on the grid with weighted means, so a missing point gets
  zero weight and the result is renormalized to the valid area, see
  :func:`missing_weights`;
- the losses include spectral ones, which need complete fields and cannot
  take spatial weights, so the missing points are filled in identically in
  prediction and target instead, see :func:`fill_missing`.
"""

from typing import Optional, Union

import torch


def missing_values(x: torch.Tensor) -> torch.Tensor:
    """Boolean tensor marking the missing values of ``x``."""
    return torch.isnan(x)


def fill_missing(
    x: torch.Tensor, fill: Union[torch.Tensor, float], missing: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Replace the missing values of ``x`` by ``fill``, which is broadcast against ``x``.

    ``missing`` defaults to :func:`missing_values` of ``x``; pass it to apply the
    mask of one tensor to another.
    """
    if missing is None:
        missing = missing_values(x)
    return torch.where(missing, fill, x)


def missing_weights(missing: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weights that are zero at missing points, combined with optional existing weights ``wgt``."""
    valid = torch.logical_not(missing).to(torch.float32)
    if wgt is not None:
        valid = wgt * valid
    return valid
