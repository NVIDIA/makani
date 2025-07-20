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

from functools import partial

from makani.models.common.contractions import _contract_lmwise, _contract_lwise, _contract_sep_lmwise, _contract_sep_lwise
from makani.models.common.contractions import _contract_lmwise_real, _contract_lwise_real, _contract_sep_lmwise_real, _contract_sep_lwise_real

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False, operator_type="diagonal"):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    if operator_type == "diagonal":
        pass
    elif operator_type == "block-diagonal":
        weight_syms.insert(-1, einsum_symbols[order + 1])
        out_syms[-1] = weight_syms[-2]
    elif operator_type == "dhconv":
        weight_syms.pop()
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    eq = "".join(x_syms) + "," + "".join(weight_syms) + "->" + "".join(out_syms)

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    res = tl.einsum(eq, x, weight).contiguous()

    return res


# jitted PyTorch contractions:
def _contract_dense_pytorch(x, weight, separable=False, operator_type="diagonal", complex=True):
    # make sure input is contig
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
            raise ValueError(f"Unkonw operator type {operator_type}")
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
            raise ValueError(f"Unkonw operator type {operator_type}")

    # make contiguous
    x = x.contiguous()
    return x


def _contract_dense_reconstruct(x, weight, separable=False, operator_type="diagonal", complex=True):
    """Contraction for dense tensors, factorized or not"""
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
        # weight = torch.view_as_real(weight)

    return _contract_dense_pytorch(x, weight, separable=separable, operator_type=operator_type, complex=complex)


def get_contract_fun(weight, implementation="reconstructed", separable=False, operator_type="diagonal", complex=True):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : torch.Tensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)

    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        handle = partial(_contract_dense_reconstruct, separable=separable, complex=complex, operator_type=operator_type)
        return handle
    elif implementation == "factorized":
        handle = partial(_contract_dense_pytorch, separable=separable, complex=complex, operator_type=operator_type)
        return handle
    else:
        raise ValueError(f'Got {implementation=}, expected "reconstructed" or "factorized"')
