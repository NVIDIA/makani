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


class MLPImputation(nn.Module):
    r"""
    Learned imputation of missing values from the other input channels.

    Fields such as sea surface temperature are undefined over parts of the
    globe and arrive as ``NaN``. Feeding those straight into a network poisons
    every downstream activation, so they must be filled first. This module
    predicts the fill values with a small MLP that sees all ``inp_chans``
    channels, letting it infer a plausible value from correlated fields rather
    than substituting a constant.

    Only masked positions are replaced; valid data passes through untouched.
    Positions that are ``NaN`` in the input are always treated as missing, in
    addition to anything flagged by an explicit ``mask``.

    Parameters
    ----------
    inp_chans : int, optional
        Total number of input channels the MLP conditions on, by default ``2``.
    inpute_chans : torch.Tensor, optional
        1D integer tensor of channel indices to impute, by default
        ``tensor([0])``. Its length sets the number of predicted channels.
    mlp_ratio : float, optional
        Hidden width of the MLP as a multiple of the number of imputed
        channels, by default ``2.0``.
    activation_function : torch.nn.Module, optional
        Activation used inside the MLP, by default :class:`torch.nn.GELU`.

    See Also
    --------
    ConstantImputation : cheaper alternative that fills with a learned constant.
    """

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

    def _scatter_channels(self, x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Out-of-place scatter of `values` into `x` at `self.inpute_chans` along the channel dim."""
        idx = self.inpute_chans.to(x.device)
        # build an index tensor broadcastable to x's full shape
        c_dim = x.dim() - 3  # channel axis
        shape = [1] * x.dim()
        shape[c_dim] = idx.shape[0]
        idx_expanded = idx.view(shape).expand_as(values)
        return x.scatter(c_dim, idx_expanded, values)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        r"""
        Fill masked entries of the imputed channels with MLP predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., inp_chans, nlat, nlon)``. Leading batch
            dimensions are flattened internally, so any number of them is fine.
        mask : torch.Tensor, optional
            Boolean mask of shape ``(..., len(inpute_chans), nlat, nlon)``, with
            ``True`` marking positions to impute. Combined by logical OR with
            the ``NaN`` positions of ``x``. If omitted, only ``NaN`` positions
            are imputed.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``, with masked entries of the
            imputed channels replaced and everything else unchanged.
        """
        x_sub = x[..., self.inpute_chans, :, :]

        if mask is None:
            mask = torch.isnan(x_sub)
        else:
            mask = torch.logical_or(mask, torch.isnan(x_sub))

        # zero out masked channels for the MLP input (out-of-place)
        x_zeroed = torch.where(mask, torch.zeros_like(x_sub), x_sub)
        x_clean = self._scatter_channels(x, x_zeroed)

        # flatten extra batch dims for Conv2d compatibility
        batch_shape = x_clean.shape[:-3]
        x_flat = x_clean.reshape(-1, *x_clean.shape[-3:])
        mlp_out = self.mlp(x_flat).reshape(*batch_shape, self.out_chans, *x_flat.shape[-2:])

        # replace only masked positions with MLP predictions (out-of-place)
        imputed_sub = torch.where(mask, mlp_out, x_zeroed)

        return self._scatter_channels(x_clean, imputed_sub)


class ConstantImputation(nn.Module):
    r"""
    Imputation of missing values with a learned per-channel constant.

    The cheap counterpart to :class:`MLPImputation`: instead of predicting fill
    values from the other channels, each channel gets a single scalar that is
    learned jointly with the rest of the model. Masked positions are replaced by
    that scalar and valid data passes through untouched.

    Under spatial model parallelism the fill values are replicated rather than
    sharded (the parameter is marked shared across the ``"spatial"`` group), so
    every rank imputes with identical constants.

    Parameters
    ----------
    inp_chans : int, optional
        Number of input channels, by default ``2``. One fill value is learned
        per channel, initialized from a standard normal.

    See Also
    --------
    MLPImputation : learned imputation conditioned on the other channels.
    """

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
        r"""
        Replace masked entries with the learned per-channel constants.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., inp_chans, nlat, nlon)``.
        mask : torch.Tensor, optional
            Boolean mask broadcastable to ``x``, with ``True`` marking positions
            to impute. Combined by logical OR with the ``NaN`` positions of
            ``x``. If omitted, only ``NaN`` positions are imputed.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x`` with masked entries filled.
        """
        if mask is None:
            mask = torch.isnan(x)
        else:
            mask = torch.logical_or(mask, torch.isnan(x))
        return torch.where(mask, self.weight, x)
