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


# Dense channel-mixing contractions used by SpectralAttention. These were dropped
# in e59c2f4 while the call sites in spectral_convolution.py kept referencing
# them, leaving SpectralAttention unconstructible (NameError) for both of its
# operator types. Restored verbatim from the pre-e59c2f4 networks/contractions.py.
# Note these are ungrouped, unlike the _contract_* helpers above.
def compl_mul2d_fwd(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    r"""
    Mix channels with a single weight matrix shared across all spectral modes.

    .. math::

        y_{b,o,l,m} = \sum_i x_{b,i,l,m}\, w_{i,o}

    Parameters
    ----------
    ac : torch.Tensor
        Input coefficients of shape ``(batch, in_channels, l, m)``.
    bc : torch.Tensor
        Weight of shape ``(in_channels, out_channels)``, shared over ``l`` and ``m``.

    Returns
    -------
    torch.Tensor
        Output of shape ``(batch, out_channels, l, m)``.
    """
    return torch.einsum("bixy,io->boxy", ac, bc)


def compl_muladd2d_fwd(ac: torch.Tensor, bc: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
    r"""
    :func:`compl_mul2d_fwd` followed by a bias add.

    Parameters
    ----------
    ac : torch.Tensor
        Input coefficients of shape ``(batch, in_channels, l, m)``.
    bc : torch.Tensor
        Weight of shape ``(in_channels, out_channels)``.
    cc : torch.Tensor
        Bias, broadcastable to ``(batch, out_channels, l, m)``.

    Returns
    -------
    torch.Tensor
        Output of shape ``(batch, out_channels, l, m)``.
    """
    return compl_mul2d_fwd(ac, bc) + cc


def compl_exp_mul2d_fwd(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    r"""
    Mix channels with a separate weight matrix per spherical degree :math:`l`.

    .. math::

        y_{b,o,l,m} = \sum_i x_{b,i,l,m}\, w_{l,i,o}

    The extra leading weight axis makes the channel mixing depend on the degree,
    which lets the operator learn a scale-dependent response instead of applying
    one matrix to every mode.

    Parameters
    ----------
    ac : torch.Tensor
        Input coefficients of shape ``(batch, in_channels, l, m)``.
    bc : torch.Tensor
        Weight of shape ``(l, in_channels, out_channels)``.

    Returns
    -------
    torch.Tensor
        Output of shape ``(batch, out_channels, l, m)``.
    """
    return torch.einsum("bixy,xio->boxy", ac, bc)


def compl_exp_muladd2d_fwd(ac: torch.Tensor, bc: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
    r"""
    :func:`compl_exp_mul2d_fwd` followed by a bias add.

    Parameters
    ----------
    ac : torch.Tensor
        Input coefficients of shape ``(batch, in_channels, l, m)``.
    bc : torch.Tensor
        Weight of shape ``(l, in_channels, out_channels)``.
    cc : torch.Tensor
        Bias, broadcastable to ``(batch, out_channels, l, m)``.

    Returns
    -------
    torch.Tensor
        Output of shape ``(batch, out_channels, l, m)``.
    """
    return compl_exp_mul2d_fwd(ac, bc) + cc
