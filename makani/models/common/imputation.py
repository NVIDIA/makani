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

import torch
import torch.nn as nn
from typing import Optional

from makani.utils import comm
from .layers import EncoderDecoder

# helper module to handle imputation of SST
class MLPImputation(nn.Module):
    def __init__(
        self,
        inp_chans: int = 2,
        inpute_chans: torch.Tensor = torch.tensor([0]),
        mlp_ratio: float = 2.0,
        activation_function: nn.Module = nn.GELU,
    ):
        super().__init__()

        self.inp_chans = inp_chans
        self.inpute_chans = inpute_chans
        self.out_chans = inpute_chans.shape[0]

        self.mlp = EncoderDecoder(
            num_layers=1,
            input_dim=self.inp_chans,
            output_dim=self.out_chans,
            hidden_dim=int(mlp_ratio * self.out_chans),
            act_layer=activation_function,
            input_format="nchw",
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is None:
            mask = torch.isnan(x[..., self.inpute_chans, :, :])
        else:
            mask = torch.logical_or(mask, torch.isnan(x[..., self.inpute_chans, :, :]))

        x[..., self.inpute_chans, :, :] = torch.where(mask, 0.0, x[..., self.inpute_chans, :, :])

        # flatten extra batch dims for Conv2d compatibility
        batch_shape = x.shape[:-3]
        x_flat = x.reshape(-1, *x.shape[-3:])
        mlp_out = self.mlp(x_flat).reshape(*batch_shape, self.out_chans, *x_flat.shape[-2:])

        x[..., self.inpute_chans, :, :] = torch.where(mask, mlp_out, x[..., self.inpute_chans, :, :])

        return x

class ConstantImputation(nn.Module):
    def __init__(
        self,
        inp_chans: int = 2,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(inp_chans, 1, 1))

        if comm.get_size("spatial") > 1:
            self.weight.is_shared_mp = ["spatial"]
            self.weight.sharded_dims_mp = [None, None, None]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is None:
            mask = torch.isnan(x)
        else:
            mask = torch.logical_or(mask, torch.isnan(x))
        return torch.where(mask, self.weight, x)