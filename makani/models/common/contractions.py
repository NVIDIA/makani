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


# for the factorized spectral convolution
@torch.compile
def _contract_rank(xc: torch.Tensor, wc: torch.Tensor, ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    resc = torch.einsum("bixy,ior,xr,yr->boxy", xc, wc, ac, bc)
    return resc


# new contractions set to replace older ones. We use complex


@torch.compile
def _contract_lmwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    resc = torch.einsum("bgixy,gioxy->bgoxy", ac, bc)
    return resc


@torch.compile
def _contract_lwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    resc = torch.einsum("bgixy,giox->bgoxy", ac, bc)
    return resc


@torch.compile
def _contract_mwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    resc = torch.einsum("bgixy,gioy->bgoxy", ac, bc)
    return resc


@torch.compile
def _contract_sep_lmwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    resc = torch.einsum("bgixy,gixy->bgoxy", ac, bc)
    return resc


@torch.compile
def _contract_sep_lwise(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    resc = torch.einsum("bgixy,gix->bgoxy", ac, bc)
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
    res = torch.einsum("bgixys,gixy->bgoxys", a, b).contiguous()
    return res


@torch.compile
def _contract_sep_lwise_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bgixys,gix->bgoxys", a, b).contiguous()
    return res
