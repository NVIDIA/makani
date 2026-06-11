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


import torch
import torch.nn as nn


class NonNegativeConstraint(nn.Module):
    """Enforce nonnegativity on a named subset of channels (dim -3, nchw layout).

    Normalization convention: x_norm = (x_raw - bias) / scale, so physical
    zero sits at x_norm = -bias/scale. The offset = bias/scale is precomputed
    in the constructor so the forward pass is cheap.

    Training mode: smooth multiplicative approximation x*sigmoid(x/eps) applied
    in the shifted (physical-zero-centered) space so gradients flow for slightly
    negative values.

    Eval/inference mode: hard clamp, guaranteeing x_raw >= 0 before any
    downstream conservation corrections.

    Args:
        channel_names:  full list of channel names for the model output tensor.
        names_to_clamp: list of channel names to enforce nonnegativity on.
                        Names not found in channel_names are silently skipped.
        bias:           normalization bias tensor, shape (1, C, 1, 1) or None.
        scale:          normalization scale tensor, shape (1, C, 1, 1) or None.
        eps:            transition width for the soft clamp (normalized units).
    """

    def __init__(self, channel_names, names_to_clamp, bias=None, scale=None, eps=0.1):
        super().__init__()
        self.eps = eps

        # resolve names to indices, skipping any not present in channel_names
        chan_idx = [channel_names.index(n) for n in names_to_clamp if n in channel_names]
        if not chan_idx:
            raise ValueError(f"None of the requested channel names {names_to_clamp} were found in channel_names.")
        self.register_buffer("channel_indices", torch.tensor(chan_idx, dtype=torch.long), persistent=False)

        if bias is not None and scale is not None:
            means = bias[0, chan_idx, 0, 0].float()
            stds  = scale[0, chan_idx, 0, 0].float()
            # offset = bias/scale; physical zero is at x_norm = -offset
            offset = (means / stds).view(1, -1, 1, 1)
            self.register_buffer("offset", offset, persistent=False)
        else:
            self.offset = None

    def forward(self, x):
        w = x[..., self.channel_indices, :, :]
        offset = self.offset.to(x.dtype) if self.offset is not None else None

        if self.training:
            # shift so physical zero maps to 0, apply smooth clamp, shift back
            w_shifted = w + offset if offset is not None else w
            w = w_shifted * torch.sigmoid(w_shifted / self.eps)
            if offset is not None:
                w = w - offset
        else:
            # hard clamp: x_norm >= -offset  <=>  x_raw >= 0
            lo = -offset if offset is not None else x.new_zeros(1)
            w = torch.clamp(w, min=lo)

        return x.index_copy(-3, self.channel_indices, w.to(x.dtype))


# this routine computes the matching pressure levels between two pl variables
# with prefix1 and prefix 2 respectively. pmin and pmax are the minimum and maximum pressure levels considered
def get_matching_channels_pl(channel_names, prefix1, prefix2, p_min, p_max, revert=True):
    # we better use regexp
    import re

    # analyse list of channel names, extract geopotential and temperatures:
    p1_pat = re.compile(r"^" + prefix1 + r"\d{1,}$")
    p2_pat = re.compile(r"^" + prefix2 + r"\d{1,}$")
    p1_chans = [x for x in channel_names if (p1_pat.match(x) is not None)]
    p2_chans = [x for x in channel_names if (p2_pat.match(x) is not None)]

    # extract common pressure levels
    p1_pressures = [int(x.replace(prefix1, "")) for x in p1_chans]
    p2_pressures = [int(x.replace(prefix2, "")) for x in p2_chans]

    # check which are the common pressure levels:
    pressures = sorted([x for x in p1_pressures if ((x in p2_pressures) and (x >= p_min) and (x <= p_max))], reverse=revert)

    # create an indexlist for z-channels
    p1_idx = [channel_names.index(f"{prefix1}{p}") for p in pressures]
    p2_idx = [channel_names.index(f"{prefix2}{p}") for p in pressures]

    return p1_idx, p2_idx, pressures
