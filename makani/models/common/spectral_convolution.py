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
import math

from functools import partial
from torch import amp

# import convenience functions for factorized tensors
from makani.utils import comm
from makani.models.common import ComplexReLU
from makani.models.common.contractions import (
    _contract_dense_pytorch,
    compl_mul2d_fwd,
    compl_muladd2d_fwd,
    compl_exp_mul2d_fwd,
    compl_exp_muladd2d_fwd,
)

import torch_harmonics as th
import torch_harmonics.distributed as thd


def spectral_conv_s2(
    forward_transform,
    inverse_transform,
    in_channels,
    out_channels,
    num_groups=1,
    bias=False,
    gain=1.0,
):
    r"""
    Build the spherical convolution of torch-harmonics on the grids and band of these transforms.

    Returns :class:`torch_harmonics.SpectralConvS2`, or
    :class:`torch_harmonics.distributed.DistributedSpectralConvS2` when the
    transforms are distributed. The layer builds its own transforms; torch-harmonics
    shares their tables with the ones passed here, so this costs no memory. Its
    weight holds one entry per degree :math:`l`, shared over the orders :math:`m`
    (``"dhconv"``), which is what makes it a convolution on :math:`S^2`.

    The bias is the spectral bias of torch-harmonics: a learned coefficient per input
    channel and mode, scaled by the integral of that channel over the sphere and added
    before the contraction. It replaces the per-channel bias in grid space of makani's
    earlier ``SpectralConv``.

    Parameters
    ----------
    forward_transform : torch.nn.Module
        Grid-to-spectral transform whose grid is the input grid.
    inverse_transform : torch.nn.Module
        Spectral-to-grid transform whose grid is the output grid. Both transforms
        have to keep the same degrees and orders.
    in_channels : int
        Number of input channels. Must be divisible by ``num_groups``.
    out_channels : int
        Number of output channels. Must be divisible by ``num_groups``.
    num_groups : int, optional
        Number of channel groups mixed independently, by default ``1``.
    bias : bool, optional
        If ``True``, learn the spectral bias described above, by default ``False``.
    gain : float, optional
        Scales the variance of the weight initialization, by default ``1.0``.

    Returns
    -------
    torch.nn.Module
        A module mapping ``(B, in_channels, *grid_in.shape)`` to
        ``(B, out_channels, *grid_out.shape)``.

    Raises
    ------
    ValueError
        If the two transforms keep different degrees or orders.

    Notes
    -----
    Under spatial parallelism the weight is split over its degrees on the ``h``
    ranks, and each ``w`` rank computes the contribution of its own orders to the
    gradient, so the weight is shared over ``w`` with a sum. The spectral bias is
    split over degrees on ``h`` and orders on ``w``. Both are replicated over
    ``matmul``.
    """
    if (forward_transform.lmax, forward_transform.mmax) != (inverse_transform.lmax, inverse_transform.mmax):
        raise ValueError(
            f"the forward transform keeps lmax={forward_transform.lmax}, mmax={forward_transform.mmax}, "
            f"the inverse transform lmax={inverse_transform.lmax}, mmax={inverse_transform.mmax}"
        )

    distributed = isinstance(inverse_transform, (thd.DistributedInverseRealSHT, thd.DistributedRealSHT))
    conv_handle = thd.DistributedSpectralConvS2 if distributed else th.SpectralConvS2
    conv = conv_handle(
        forward_transform.grid,
        inverse_transform.grid,
        in_channels,
        out_channels,
        num_groups=num_groups,
        bias=bias,
        lmax=inverse_transform.lmax,
        mmax=inverse_transform.mmax,
    )

    with torch.no_grad():
        conv.weight.mul_(math.sqrt(gain))

    # split over the degrees on the h ranks; the w ranks each hold the contribution of their orders
    conv.weight.is_shared_mp = {"w": "sum", "matmul": "avg"}
    conv.weight.sharded_dims_mp = [None, None, None, "h"]
    if bias:
        conv.spectral_bias.is_shared_mp = {"matmul": "avg"}
        conv.spectral_bias.sharded_dims_mp = [None, None, "h", "w"]

    return conv


class SpectralConv(nn.Module):
    r"""
    Spectral convolution on the periodic plane, via the 2D FFT.

    The planar counterpart of :func:`spectral_conv_s2`, used by the Fourier
    Neural Operator. Convolution in grid space is multiplication in Fourier space,
    so this layer transforms the input, multiplies every mode by its own learned
    channel-mixing matrix, and transforms back:

    .. math::

        y = \mathcal{F}^{-1}\bigl( W \cdot \mathcal{F}(x) \bigr)

    Every wavenumber pair :math:`(k_y, k_x)` has an independent weight, as
    translation equivariance on the plane allows. On the sphere use
    :func:`spectral_conv_s2` instead.

    Transforms run in fp32 with autocast disabled; only the contraction runs in
    the surrounding precision.

    Parameters
    ----------
    forward_transform : torch.nn.Module
        Grid-to-spectral transform, :class:`~makani.models.common.fft.RealFFT2` or
        :class:`~makani.mpu.fft.DistributedRealFFT2`.
    inverse_transform : torch.nn.Module
        Spectral-to-grid transform, :class:`~makani.models.common.fft.InverseRealFFT2`
        or :class:`~makani.mpu.fft.DistributedInverseRealFFT2`. May target a
        different resolution than ``forward_transform``, in which case the residual
        is resampled accordingly.
    in_channels : int
        Number of input channels. Must be divisible by ``num_groups``.
    out_channels : int
        Number of output channels. Must be divisible by ``num_groups``.
    num_groups : int, optional
        Number of channel groups mixed independently, by default ``1``.
    bias : bool, optional
        If ``True``, add a learned per-channel bias in grid space, by default
        ``False``.
    gain : float, optional
        Scales the variance of the weight initialization, by default ``1.0``.

    Raises
    ------
    TypeError
        If the transforms are spherical harmonic transforms.
    ValueError
        If the channel counts are not divisible by ``num_groups``, or if the two
        transforms keep different modes.

    Notes
    -----
    Under spatial parallelism the weight is split along its spectral dimensions
    (``h`` for the first, ``w`` for the second), as the distributed FFTs split
    the modes.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        num_groups=1,
        bias=False,
        gain=1.0,
    ):
        super().__init__()

        if isinstance(inverse_transform, (th.InverseRealSHT, thd.DistributedInverseRealSHT)):
            raise TypeError("SpectralConv is the planar convolution; on the sphere use spectral_conv_s2")
        if in_channels % num_groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_groups ({num_groups})")
        if out_channels % num_groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by num_groups ({num_groups})")
        if (forward_transform.lmax, forward_transform.mmax) != (inverse_transform.lmax, inverse_transform.mmax):
            raise ValueError(
                f"the forward transform keeps lmax={forward_transform.lmax}, mmax={forward_transform.mmax}, "
                f"the inverse transform lmax={inverse_transform.lmax}, mmax={inverse_transform.mmax}"
            )

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (self.forward_transform.nlat != self.inverse_transform.nlat) or (
            self.forward_transform.nlon != self.inverse_transform.nlon
        )

        # the distributed FFTs split the modes like this; the serial ones keep them all
        if hasattr(self.inverse_transform, "l_shapes"):
            self.modes_lat_local = self.inverse_transform.l_shapes[comm.get_rank("h")]
            self.modes_lon_local = self.inverse_transform.m_shapes[comm.get_rank("w")]
        else:
            self.modes_lat_local = self.modes_lat
            self.modes_lon_local = self.modes_lon

        weight_shape = [
            num_groups,
            in_channels // num_groups,
            out_channels // num_groups,
            self.modes_lat_local,
            self.modes_lon_local,
        ]

        # Compute scaling factor for correct initialization
        scale = math.sqrt(gain / (in_channels // num_groups)) * torch.ones(
            self.modes_lat_local, 1, dtype=torch.complex64
        )
        # seemingly the first weight is not really complex, so we need to account for that
        scale[0] *= math.sqrt(2.0)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))
        self.weight.is_shared_mp = {"matmul": "avg"}
        self.weight.sharded_dims_mp = [None, None, None, "h", "w"]

        self._contract = partial(_contract_dense_pytorch, separable=False, operator_type="diagonal")

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))
            self.bias.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
            self.bias.sharded_dims_mp = [None, None, None, None]

    def forward(self, x):
        r"""
        Transform, apply the learned spectral weights, and transform back.

        Parameters
        ----------
        x : torch.Tensor
            Input field of shape ``(B, in_channels, nlat, nlon)`` on the grid
            the forward transform was constructed for.

        Returns
        -------
        x : torch.Tensor
            Output field of shape ``(B, out_channels, nlat_out, nlon_out)``.
        residual : torch.Tensor
            The input, resampled onto the output grid if the forward and
            inverse transforms differ in resolution, and returned unchanged
            otherwise. Returned so the caller can form its own skip connection
            at the correct resolution, which it could not do itself without
            repeating the transform.
        """
        dtype = x.dtype
        residual = x

        with amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.float32)
            # No .contiguous() on the transform output: it is contiguous by construction, and a
            # copy over a complex buffer cannot be codegen'd by inductor -- triton has no complex
            # type, so the kernel signature fails with KeyError: 'complex64'.
            x = self.forward_transform(x)
            if self.scale_residual:
                residual = self.inverse_transform(x).to(dtype=dtype)

        B, C, H, W = x.shape
        x = x.reshape(B, self.num_groups, C // self.num_groups, H, W)
        xp = self._contract(x, self.weight)
        # merge the group axes on the real view: the contraction output can be non-contiguous, in
        # which case this reshape copies, and that copy must not be over a complex buffer (above)
        x = torch.view_as_complex(torch.view_as_real(xp).reshape(B, self.out_channels, H, W, 2).contiguous())

        with amp.autocast(device_type=x.device.type, enabled=False):
            x = self.inverse_transform(x)

        x = x.to(dtype=dtype)

        if hasattr(self, "bias"):
            # cast the fp32 bias to the activation dtype before the (non-autocast-managed)
            # add, mirroring how autocast folds bias into conv/linear -- keeps the output
            # in the input dtype instead of promoting to fp32.
            x = x + self.bias.to(dtype=x.dtype)

        return x, residual


class SpectralAttention(nn.Module):
    r"""
    Nonlinear spectral layer: an MLP applied in spherical harmonic space.

    Where :func:`spectral_conv_s2` multiplies the spectrum by a fixed learned
    factor, this layer runs a small complex-valued MLP over the coefficients,

    .. math::

        y = \mathcal{F}^{-1}\bigl( \mathrm{MLP}(\mathcal{F}(x)) \bigr)

    with complex activations between the layers. Interleaving nonlinearities
    with the channel mixing lets modes interact rather than each being scaled
    independently, which is what the "attention" in the name refers to -- it is
    not dot-product attention.

    Two weight layouts are available via ``operator_type``:

    * ``"diagonal"`` -- one weight matrix shared across all modes.
    * ``"l-dependant"`` -- an independent weight matrix per spherical degree
      :math:`l`, so the channel mixing can vary with spatial scale. Costs a
      factor of ``lmax`` more parameters.

    Parameters
    ----------
    forward_transform : torch.nn.Module
        Grid-to-spectral transform.
    inverse_transform : torch.nn.Module
        Spectral-to-grid transform. May target a different resolution or grid.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    operator_type : str, optional
        ``"diagonal"`` (default) or ``"l-dependant"``; see above.
    hidden_size_factor : int, optional
        Hidden width of the spectral MLP as a multiple of ``in_channels``, by
        default ``2``.
    complex_activation : str, optional
        Mode passed to :class:`~makani.models.common.activations.ComplexReLU`,
        by default ``"real"``.
    bias : bool, optional
        If ``True``, add a learned complex bias in each spectral layer, by
        default ``False``.
    spectral_layers : int, optional
        Number of hidden layers in the spectral MLP, by default ``1``.
    drop_rate : float, optional
        Dropout probability applied after each activation, by default ``0.0``.
    gain : float, optional
        Scales the variance of the output weight's initialization, by default ``1.0``.

    Raises
    ------
    ValueError
        If the inverse transform's mode counts disagree with the forward
        transform's, or if ``operator_type`` is unknown.

    See Also
    --------
    spectral_conv_s2 : the linear, cheaper spectral layer on the sphere.
    SpectralConv : its counterpart on the plane.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        operator_type="diagonal",
        hidden_size_factor=2,
        complex_activation="real",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
        gain=1.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.operator_type = operator_type
        self.spectral_layers = spectral_layers

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.scale_residual = (
            (self.forward_transform.nlat != self.inverse_transform.nlat)
            or (self.forward_transform.nlon != self.inverse_transform.nlon)
            or (self.forward_transform.grid != self.inverse_transform.grid)
        )

        if inverse_transform.lmax != self.modes_lat:
            raise ValueError(
                f"inverse transform lmax ({inverse_transform.lmax}) must match modes_lat ({self.modes_lat})"
            )
        if inverse_transform.mmax != self.modes_lon:
            raise ValueError(
                f"inverse transform mmax ({inverse_transform.mmax}) must match modes_lon ({self.modes_lon})"
            )

        hidden_size = int(hidden_size_factor * self.in_channels)

        if operator_type == "diagonal":
            self.mul_add_handle = compl_muladd2d_fwd
            self.mul_handle = compl_mul2d_fwd

            # weights
            scale = math.sqrt(2.0 / float(in_channels))
            w = [scale * torch.randn(self.in_channels, hidden_size, dtype=torch.complex64)]
            for l in range(1, self.spectral_layers):
                scale = math.sqrt(2.0 / float(hidden_size))
                w.append(scale * torch.randn(hidden_size, hidden_size, dtype=torch.complex64))
            self.w = nn.ParameterList(w)

            scale = math.sqrt(gain / float(in_channels))
            self.wout = nn.Parameter(scale * torch.randn(hidden_size, self.out_channels, dtype=torch.complex64))

            if bias:
                self.b = nn.ParameterList(
                    [scale * torch.randn(hidden_size, 1, 1, dtype=torch.complex64) for _ in range(self.spectral_layers)]
                )

            self.activations = nn.ModuleList([])
            for l in range(0, self.spectral_layers):
                self.activations.append(
                    ComplexReLU(mode=complex_activation, bias_shape=(hidden_size, 1, 1), scale=scale)
                )

        elif operator_type == "l-dependant":
            self.mul_add_handle = compl_exp_muladd2d_fwd
            self.mul_handle = compl_exp_mul2d_fwd

            # weights
            scale = math.sqrt(2.0 / float(in_channels))
            w = [scale * torch.randn(self.modes_lat, self.in_channels, hidden_size, dtype=torch.complex64)]
            for l in range(1, self.spectral_layers):
                scale = math.sqrt(2.0 / float(hidden_size))
                w.append(scale * torch.randn(self.modes_lat, hidden_size, hidden_size, dtype=torch.complex64))
            self.w = nn.ParameterList(w)

            if bias:
                self.b = nn.ParameterList(
                    [scale * torch.randn(hidden_size, 1, 1, dtype=torch.complex64) for _ in range(self.spectral_layers)]
                )

            scale = math.sqrt(gain / float(in_channels))
            self.wout = nn.Parameter(
                scale * torch.randn(self.modes_lat, hidden_size, self.out_channels, dtype=torch.complex64)
            )

            self.activations = nn.ModuleList([])
            for l in range(0, self.spectral_layers):
                self.activations.append(
                    ComplexReLU(mode=complex_activation, bias_shape=(hidden_size, 1, 1), scale=scale)
                )

        else:
            raise ValueError("Unknown operator type")

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward_mlp(self, x):
        r"""
        Run the complex-valued MLP on spectral coefficients.

        Exposed separately from ``forward`` so the spectral MLP can be applied
        to coefficients that are already in spectral space, without paying for a
        transform round trip.

        Parameters
        ----------
        x : torch.Tensor
            Complex coefficients of shape ``(B, in_channels, lmax, mmax)``.

        Returns
        -------
        torch.Tensor
            Complex coefficients of shape ``(B, out_channels, lmax, mmax)``.
        """
        # the contractions, weights, biases and activations all work on the complex coefficients directly
        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                x = self.mul_add_handle(x, self.w[l], self.b[l])
            else:
                x = self.mul_handle(x, self.w[l])
            x = self.activations[l](x)
            x = self.drop(x)

        # final MLP
        x = self.mul_handle(x, self.wout)

        return x

    def forward(self, x):
        r"""
        Transform, apply the spectral MLP, and transform back.

        Parameters
        ----------
        x : torch.Tensor
            Input field of shape ``(B, in_channels, nlat, nlon)``.

        Returns
        -------
        x : torch.Tensor
            Output field of shape ``(B, out_channels, nlat_out, nlon_out)``.
        residual : torch.Tensor
            The input, resampled onto the output grid if the forward and
            inverse transforms differ, and unchanged otherwise. Returned so the
            caller can form a skip connection at the correct resolution.
        """
        dtype = x.dtype
        residual = x

        # FWD transform
        with amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.float32)
            x = self.forward_transform(x)
            if self.scale_residual:
                residual = self.inverse_transform(x)

        # convert back. Only the residual is cast here: x still holds the complex
        # spectral coefficients that forward_mlp consumes, so casting it to the (real)
        # input dtype at this point would discard the imaginary part. It is cast once
        # after the inverse transform below, as in SpectralConv.forward.
        if self.scale_residual:
            residual = residual.to(dtype=dtype)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform: the coefficients are complex and stay so, only the real output is cast below
        with amp.autocast(device_type=x.device.type, enabled=False):
            x = self.inverse_transform(x)

        # convert back
        x = x.to(dtype=dtype)

        return x, residual
