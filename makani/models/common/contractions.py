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


@torch.compile
def _contract_lmwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    resc = torch.einsum("bgixy,gioxy->bgoxy", ac, bc)
    return resc


@torch.compile
def _contract_lwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    resc = torch.einsum("bgixy,giox->bgoxy", ac, bc)
    return resc


@torch.compile
def _contract_sep_lmwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    resc = torch.einsum("bgixy,gixy->bgixy", ac, bc)
    return resc


@torch.compile
def _contract_sep_lwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    resc = torch.einsum("bgixy,gix->bgixy", ac, bc)
    return resc


@torch.compile
def _contract_lmwise_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bgixys,gioxy->bgoxys", a, b).contiguous()
    return res


@torch.compile
def _contract_lwise_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bgixys,giox->bgoxys", a, b).contiguous()
    return res


@torch.compile
def _contract_sep_lmwise_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bgixys,gixy->bgixys", a, b).contiguous()
    return res


@torch.compile
def _contract_sep_lwise_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bgixys,gix->bgixys", a, b).contiguous()
    return res


def _contract_dense_pytorch(x, weight, separable=False, operator_type="diagonal", complex=True):
    """Dense spectral convolution contraction dispatching to the appropriate compiled einsum kernel."""
    x = x.contiguous()

    if separable:
        if operator_type == "diagonal":
            if complex:
                x = _contract_sep_lmwise(x, weight)
            else:
                x = _contract_sep_lmwise_real(x, weight)
        elif operator_type == "dhconv":
            if complex:
                x = _contract_sep_lwise(x, weight)
            else:
                x = _contract_sep_lwise_real(x, weight)
        else:
            raise ValueError(f"Unknown operator type {operator_type}")
    else:
        if operator_type == "diagonal":
            if complex:
                x = _contract_lmwise(x, weight)
            else:
                x = _contract_lmwise_real(x, weight)
        elif operator_type == "dhconv":
            if complex:
                x = _contract_lwise(x, weight)
            else:
                x = _contract_lwise_real(x, weight)
        else:
            raise ValueError(f"Unknown operator type {operator_type}")

    return x.contiguous()
