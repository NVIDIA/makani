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

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from makani.utils.functions import contiguous_complex_safe

# the normalization of the inverse that applies the adjoint of an rfft of the given normalization
_ADJOINT_NORM = {None: "forward", "backward": "forward", "ortho": "ortho", "forward": "backward"}


def _resize(x: torch.Tensor, n: Optional[int], dim: int) -> torch.Tensor:
    """Zero-pad or cut ``x`` to length ``n`` along ``dim``, as the ``n`` argument of :func:`torch.fft.rfft` does."""
    if n is None or n == x.shape[dim]:
        return x
    if n < x.shape[dim]:
        return x.narrow(dim, 0, n)
    dim = dim % x.dim()
    return F.pad(x, [0, 0] * (x.dim() - 1 - dim) + [0, n - x.shape[dim]])


def _single_bins(nbins: int, n: int, device: torch.device) -> torch.Tensor:
    """
    Mask of the one-sided bins ``0 .. nbins - 1`` of a length ``n`` real signal that have no mirror image: DC and, for
    even ``n``, Nyquist. A c2r inverse takes only their real parts, and every other bin twice. Built on the device:
    setting single entries from Python copies a scalar from the host and synchronizes the stream.
    """
    k = torch.arange(nbins, device=device)
    single = k == 0
    if n % 2 == 0:
        single = single | (k == n // 2)
    return single


class _RealFFT(torch.autograd.Function):
    """
    :func:`torch.fft.rfft` along ``dim``, truncated to the first ``nmodes`` bins, with a backward that keeps only the
    length of the input.

    The backward of ``torch.fft.rfft`` keeps the whole input (it reads one size of it, and autograd saves the tensor
    for that) and returns the real part of a full-length c2c inverse, a stride-2 view that a compiled consumer of
    the gradient copies; the backward of a truncating slice zero-fills and copies the full spectrum on top. The
    gradient is :math:`\\mathrm{Re} \\sum_k g_k e^{2 \\pi i j k / n}` over the kept bins: a c2r inverse of ``g`` with
    the bins it counts twice halved and the imaginary parts of the single ones zeroed, which pads the truncated
    bins itself. Half the transform size, and a contiguous real gradient.
    """

    # forward and setup_context apart, and a generated vmap rule, so that torch.func (grad, vmap, jacrev) can
    # transform it; the forward and backward see one sample, so the dimension needs no shift under vmap
    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor, nmodes: int, dim: int, norm: Optional[str]) -> torch.Tensor:
        y = torch.fft.rfft(x, dim=dim, norm=norm)
        if nmodes < y.shape[dim]:
            # the slice is over a complex tensor, so the copy goes through the real view: inductor cannot
            # codegen a copy on a complex buffer
            y = contiguous_complex_safe(y.narrow(dim, 0, nmodes))
        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, _, dim, norm = inputs
        ctx.n = x.shape[dim]
        ctx.dim = dim % x.dim()
        ctx.norm = norm

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        shape = [1] * grad.dim()
        shape[ctx.dim] = grad.shape[ctx.dim]
        single = _single_bins(grad.shape[ctx.dim], ctx.n, grad.device).reshape(shape)
        grad = torch.complex(torch.where(single, grad.real, 0.5 * grad.real), torch.where(single, 0.0, 0.5 * grad.imag))
        return torch.fft.irfft(grad, n=ctx.n, dim=ctx.dim, norm=_ADJOINT_NORM[ctx.norm]), None, None, None


def rfft(
    x: torch.Tensor, n: Optional[int] = None, nmodes: Optional[int] = None, dim: int = -1, norm: Optional[str] = None
) -> torch.Tensor:
    r"""
    Real FFT along ``dim``, keeping the first ``nmodes`` bins.

    The same as ``torch.fft.rfft(x, n=n, dim=dim, norm=norm)`` cut to ``nmodes`` bins along ``dim``, contiguous, but
    the backward keeps nothing of the input but its length and returns a contiguous gradient; see :class:`_RealFFT`.

    Parameters
    ----------
    x : torch.Tensor
        Real input.
    n : int, optional
        Length of the transform; ``x`` is zero-padded or cut to it along ``dim``. Defaults to ``x.shape[dim]``.
    nmodes : int, optional
        Number of bins kept, at most ``n // 2 + 1``. Defaults to all of them.
    dim : int, optional
        Dimension of the transform, by default ``-1``.
    norm : str, optional
        Normalization, as for :func:`torch.fft.rfft`.

    Returns
    -------
    torch.Tensor
        Complex spectrum with ``nmodes`` bins along ``dim``.
    """
    x = _resize(x, n, dim)
    nbins = x.shape[dim] // 2 + 1
    nmodes = nbins if nmodes is None else nmodes
    if not 0 < nmodes <= nbins:
        raise ValueError(f"nmodes must be in [1, {nbins}] for a transform of length {x.shape[dim]}, got {nmodes}")
    return _RealFFT.apply(x, nmodes, dim, norm)


def irfft(x: torch.Tensor, n: int, dim: int = -1, norm: Optional[str] = None) -> torch.Tensor:
    r"""
    Inverse real FFT of the one-sided spectrum ``x`` along ``dim``, to length ``n``.

    Missing bins up to ``n // 2 + 1`` are zero and extra ones are dropped, as in :func:`torch.fft.irfft`. The
    imaginary parts of the DC and (even ``n``) Nyquist bins are zeroed first, so that the result does not depend on
    how the backend treats them (cuFFT does not ignore them). ``torch_harmonics.fft.irfft`` does the same, but sets
    its mask from Python, which costs a host-to-device copy and a stream synchronization per call; here it is built
    on the device.

    Parameters
    ----------
    x : torch.Tensor
        Complex one-sided spectrum.
    n : int
        Length of the output along ``dim``.
    dim : int, optional
        Dimension of the transform, by default ``-1``.
    norm : str, optional
        Normalization, as for :func:`torch.fft.irfft`.

    Returns
    -------
    torch.Tensor
        Real signal of length ``n`` along ``dim``.
    """
    dim = dim % x.dim()
    if x.shape[dim] == 0:
        # FFT backends need at least one input bin; an empty spectrum synthesizes zero and keeps autograd
        x = F.pad(x, [0, 0] * (x.dim() - 1 - dim) + [0, 1])
    shape = [1] * x.dim()
    shape[dim] = x.shape[dim]
    single = _single_bins(x.shape[dim], n, x.device).reshape(shape)
    x = torch.complex(x.real, torch.where(single, 0.0, x.imag))
    return torch.fft.irfft(x, n=n, dim=dim, norm=norm)


class RealFFT1(nn.Module):
    r"""
    Forward real-valued FFT along the last dimension, with mode truncation.

    Wraps :func:`torch.fft.rfft` behind the same module interface the SHT
    classes use, so a spectral layer can swap a planar transform for a spherical
    one without changing its call sites. The transform is applied to the last
    dimension and the output is truncated to ``mmax`` modes.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2 or makani.utils.grids.PlanarGrid
        The grid of the field; the transform runs along its ``nlon`` longitudes.
    lmax : int, optional
        Upper bound on the retained mode count, clamped to ``nlon // 2 + 1``.
        Defaults to the full ``nlon // 2 + 1``.
    mmax : int, optional
        Number of modes actually kept, clamped to ``lmax``. Defaults to ``lmax``.

    See Also
    --------
    InverseRealFFT1 : the corresponding inverse transform.
    """

    def __init__(self, grid, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # use local FFT here
        self.grid = grid
        self.nlon = grid.nlon
        self.lmax = min(lmax or self.nlon // 2 + 1, self.nlon // 2 + 1)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.lmax)

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho") -> torch.Tensor:
        r"""
        Transform to spectral space and truncate.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input of shape ``(..., nlon)``.
        norm : str, optional
            Normalization mode forwarded to :func:`torch.fft.rfft`, by default
            ``"ortho"`` (unitary, so forward and inverse are adjoint).

        Returns
        -------
        torch.Tensor
            Complex coefficients of shape ``(..., mmax)``.
        """
        # transform with mode truncation, see rfft for its backward
        return rfft(x, n=self.nlon, nmodes=self.mmax, dim=-1, norm=norm)


class InverseRealFFT1(nn.Module):
    r"""
    Inverse real-valued FFT along the last dimension.

    Counterpart to :class:`RealFFT1`. Truncated inputs are zero-padded back up
    to ``nlon // 2 + 1`` modes implicitly by :func:`torch.fft.irfft`, so a
    signal round-tripped through ``RealFFT1`` and this module is band-limited
    rather than exactly recovered whenever ``mmax`` truncates.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2 or makani.utils.grids.PlanarGrid
        The grid of the field; the transform runs along its ``nlon`` longitudes.
    lmax : int, optional
        Upper bound on the mode count, clamped to ``nlon // 2 + 1``. Defaults to
        the full ``nlon // 2 + 1``.
    mmax : int, optional
        Number of input modes expected, clamped to ``lmax``. Defaults to ``lmax``.

    See Also
    --------
    RealFFT1 : the corresponding forward transform.
    """

    def __init__(self, grid, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        self.grid = grid
        self.nlon = grid.nlon
        self.lmax = min(lmax or self.nlon // 2 + 1, self.nlon // 2 + 1)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.lmax)

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho") -> torch.Tensor:
        r"""
        Transform back to grid space.

        Parameters
        ----------
        x : torch.Tensor
            Complex coefficients of shape ``(..., m)`` with ``m <= nlon // 2 + 1``.
            Missing high modes are zero-padded.
        norm : str, optional
            Normalization mode forwarded to :func:`torch.fft.irfft`, by default
            ``"ortho"``. Must match the mode used in the forward transform.

        Returns
        -------
        torch.Tensor
            Real-valued signal of shape ``(..., nlon)``.
        """
        # implicit padding
        return irfft(x, n=self.nlon, dim=-1, norm=norm)


class RealFFT2(nn.Module):
    r"""
    Forward real-valued 2D FFT with mode truncation, mirroring the SHT interface.

    Drop-in planar replacement for a forward spherical harmonic transform: the
    last two dimensions are treated as latitude and longitude, and the result is
    truncated to ``lmax`` latitudinal and ``mmax`` longitudinal modes.

    Truncation along the latitude axis is two-sided. The FFT along a
    non-halved axis places positive frequencies at the front and negative
    frequencies at the back, so keeping the ``lmax`` lowest frequencies means
    concatenating the leading ``ceil(lmax/2)`` and trailing ``floor(lmax/2)``
    entries; a plain ``[:lmax]`` slice would discard all negative frequencies
    instead. The longitude axis needs no such care because ``rfft2`` already
    returns only non-negative frequencies there. Truncation is skipped entirely
    when ``lmax`` and ``mmax`` are at their maxima.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2 or makani.utils.grids.PlanarGrid
        The grid of the field, ``nlat`` latitudes along the second-to-last dimension and ``nlon``
        longitudes along the last.
    lmax : int, optional
        Retained latitudinal modes, clamped to ``nlat``. Defaults to ``nlat``.
    mmax : int, optional
        Retained longitudinal modes, clamped to ``nlon // 2 + 1``. Defaults to
        ``nlon // 2 + 1``.

    See Also
    --------
    InverseRealFFT2 : the corresponding inverse transform.
    """

    def __init__(self, grid, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        self.grid = grid
        self.nlat = grid.nlat
        self.nlon = grid.nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho") -> torch.Tensor:
        r"""
        Transform to spectral space and truncate both spectral axes.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input of shape ``(..., nlat, nlon)``.
        norm : str, optional
            Normalization mode forwarded to :func:`torch.fft.rfft2`, by default
            ``"ortho"``.

        Returns
        -------
        torch.Tensor
            Complex coefficients of shape ``(..., lmax, mmax)``.
        """
        # the 2D transform as rfft2 does it, a real transform in longitude and a complex one in latitude, but with
        # the longitude truncation before the latitude transform, and the backward of rfft
        y = rfft(x, n=self.nlon, nmodes=self.mmax, dim=-1, norm=norm)
        y = torch.fft.fft(y, n=self.nlat, dim=-2, norm=norm)

        if self.lmax < self.nlat:
            y = torch.cat((y[..., : self.lmax_high, :], y[..., -self.lmax_low :, :]), dim=-2)

        return y


class InverseRealFFT2(nn.Module):
    r"""
    Inverse real-valued 2D FFT, mirroring the inverse SHT interface.

    Counterpart to :class:`RealFFT2`. Truncated coefficients are zero-padded
    back to the full grid before the inverse transform. The padding is inserted
    *between* the retained positive and negative latitudinal frequencies (not
    appended at the end), which is where the missing high frequencies belong in
    FFT layout.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2 or makani.utils.grids.PlanarGrid
        The grid of the field, ``nlat`` latitudes along the second-to-last dimension and ``nlon``
        longitudes along the last.
    lmax : int, optional
        Latitudinal modes expected on input, clamped to ``nlat``. Defaults to ``nlat``.
    mmax : int, optional
        Longitudinal modes expected on input, clamped to ``nlon // 2 + 1``.
        Defaults to ``nlon // 2 + 1``.

    See Also
    --------
    RealFFT2 : the corresponding forward transform.
    """

    def __init__(self, grid, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # the inverse in latitude first, then the real inverse in longitude (irfft), which needs the m=0 column to
        # be Hermitian. Truncation to an even lmax breaks that, and backends differ in how they treat the remaining
        # imaginary parts
        self.grid = grid
        self.nlat = grid.nlat
        self.nlon = grid.nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho") -> torch.Tensor:
        r"""
        Zero-pad the truncated spectrum and transform back to grid space.

        Parameters
        ----------
        x : torch.Tensor
            Complex coefficients of shape ``(..., lmax, m)`` with ``m >= mmax``;
            anything beyond ``mmax`` is dropped.
        norm : str, optional
            Normalization mode forwarded to the inverse FFTs, by default
            ``"ortho"``. Must match the mode used in the forward transform.

        Returns
        -------
        torch.Tensor
            Real-valued field of shape ``(..., nlat, nlon)``.
        """
        # truncation is implicit but better do it manually
        xt = x[..., : self.mmax]

        if self.truncate:
            # pad
            xth = xt[..., : self.lmax_high, :]
            xtl = xt[..., -self.lmax_low :, :]
            xthp = F.pad(xth, (0, 0, 0, self.nlat - self.lmax))
            xt = torch.cat([xthp, xtl], dim=-2)

        out = torch.fft.ifft(xt, n=self.nlat, dim=-2, norm=norm)
        out = irfft(out, n=self.nlon, dim=-1, norm=norm)

        return out


class RealFFT3(nn.Module):
    r"""
    Forward real-valued 3D FFT with truncation on all three spectral axes.

    Three-dimensional analogue of :class:`RealFFT2`, used by models that treat a
    vertical (or temporal) axis spectrally alongside the two horizontal ones.
    The transform runs over the last three dimensions ``(d, h, w)``. As in the
    2D case, truncation along ``d`` and ``h`` is two-sided (leading and trailing
    frequency blocks are concatenated) while ``w`` is a simple head slice,
    because ``rfftn`` halves only the last axis. Normalization is fixed to
    ``"ortho"``.

    Parameters
    ----------
    nd : int
        Number of grid points along the third-to-last dimension.
    nh : int
        Number of grid points along the second-to-last dimension.
    nw : int
        Number of grid points along the last dimension.
    ldmax : int, optional
        Retained modes along ``d``, clamped to ``nd``. Defaults to ``nd``.
    lhmax : int, optional
        Retained modes along ``h``, clamped to ``nh``. Defaults to ``nh``.
    lwmax : int, optional
        Retained modes along ``w``, clamped to ``nw // 2 + 1``. Defaults to
        ``nw // 2 + 1``.

    See Also
    --------
    InverseRealFFT3 : the corresponding inverse transform.
    """

    def __init__(self, nd, nh, nw, ldmax=None, lhmax=None, lwmax=None):
        super().__init__()

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)

        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

    def forward(self, x):
        r"""
        Transform to spectral space and truncate all three spectral axes.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input of shape ``(..., nd, nh, nw)``.

        Returns
        -------
        torch.Tensor
            Complex coefficients of shape ``(..., ldmax, lhmax, lwmax)``.
        """
        # the 3D transform as rfftn does it, a real transform in w and complex ones in d and h, but with the w
        # truncation before the complex transforms, and the backward of rfft
        x = rfft(x, n=self.nw, nmodes=self.lwmax, dim=-1, norm="ortho")
        x = torch.fft.fft2(x, s=(self.nd, self.nh), dim=(-3, -2), norm="ortho")

        # truncate in h
        x = torch.cat([x[..., : self.lhmax_high, :], x[..., -self.lhmax_low :, :]], dim=-2)

        # truncate in d
        x = torch.cat([x[..., : self.ldmax_high, :, :], x[..., -self.ldmax_low :, :, :]], dim=-3)

        return x


class InverseRealFFT3(nn.Module):
    r"""
    Inverse real-valued 3D FFT, counterpart to :class:`RealFFT3`.

    Zero-pads the ``d`` and ``h`` spectral axes back to full length (inserting
    the padding between the retained positive and negative frequency blocks)
    and lets the real inverse in ``w`` pad the halved ``w`` axis implicitly.
    Padding is skipped on any axis that was not truncated. Normalization is
    fixed to ``"ortho"``.

    Parameters
    ----------
    nd : int
        Number of grid points along the third-to-last dimension.
    nh : int
        Number of grid points along the second-to-last dimension.
    nw : int
        Number of grid points along the last dimension.
    ldmax : int, optional
        Modes expected along ``d``, clamped to ``nd``. Defaults to ``nd``.
    lhmax : int, optional
        Modes expected along ``h``, clamped to ``nh``. Defaults to ``nh``.
    lwmax : int, optional
        Modes expected along ``w``, clamped to ``nw // 2 + 1``. Defaults to
        ``nw // 2 + 1``.

    See Also
    --------
    RealFFT3 : the corresponding forward transform.
    """

    def __init__(self, nd, nh, nw, ldmax=None, lhmax=None, lwmax=None):
        super().__init__()

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)

        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

    def forward(self, x):
        r"""
        Zero-pad the truncated spectrum and transform back to grid space.

        Parameters
        ----------
        x : torch.Tensor
            Complex coefficients of shape ``(..., ldmax, lhmax, lwmax)``.

        Returns
        -------
        torch.Tensor
            Real-valued field of shape ``(..., nd, nh, nw)``.
        """

        # pad in d
        if self.ldmax < self.nd:
            # pad
            xh = x[..., : self.ldmax_high, :, :]
            xl = x[..., -self.ldmax_low :, :, :]
            xhp = F.pad(xh, (0, 0, 0, 0, 0, self.nd - self.ldmax))
            x = torch.cat([xhp, xl], dim=-3)

        # pad in h
        if self.lhmax < self.nh:
            # pad
            xh = x[..., : self.lhmax_high, :]
            xl = x[..., -self.lhmax_low :, :]
            xhp = F.pad(xh, (0, 0, 0, self.nh - self.lhmax))
            x = torch.cat([xhp, xl], dim=-2)

        # the complex inverse in d and h first, then the real inverse in w, as in InverseRealFFT2
        x = torch.fft.ifft2(x, s=(self.nd, self.nh), dim=(-3, -2), norm="ortho")
        x = irfft(x, n=self.nw, dim=-1, norm="ortho")

        return x
