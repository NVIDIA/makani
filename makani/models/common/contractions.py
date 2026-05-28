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


def _contract_lmwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgixy,gioxy->bgoxy", ac, bc)


def _contract_lwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgixy,giox->bgoxy", ac, bc)


def _contract_sep_lmwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgixy,gixy->bgixy", ac, bc)


def _contract_sep_lwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgixy,gix->bgixy", ac, bc)


def _contract_dense_pytorch(x, weight, separable=False, operator_type="diagonal"):
    """Dense spectral convolution contraction dispatching to the appropriate compiled einsum kernel."""
    x = x.contiguous()

    if separable:
        if operator_type == "diagonal":
            x = _contract_sep_lmwise(x, weight)
        elif operator_type == "dhconv":
            x = _contract_sep_lwise(x, weight)
        else:
            raise ValueError(f"Unknown operator type {operator_type}")
    else:
        if operator_type == "diagonal":
            x = _contract_lmwise(x, weight)
        elif operator_type == "dhconv":
            x = _contract_lwise(x, weight)
        else:
            raise ValueError(f"Unknown operator type {operator_type}")

    return x.contiguous()
