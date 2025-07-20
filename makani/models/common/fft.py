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


class RealFFT1(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # use local FFT here
        self.fft_handle = torch.fft.rfft

        self.nlon = nlon
        self.lmax = min(lmax or self.nlon // 2 + 1, self.nlon // 2 + 1)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.lmax)

    def forward(self, x: torch.Tensor, norm: Optional[str]="ortho") -> torch.Tensor:
        y = self.fft_handle(x, n=self.nlon, dim=-1, norm=norm)

        # mode truncation
        y = y[..., : self.mmax].contiguous()

        return y


class InverseRealFFT1(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # use local FFT here
        self.ifft_handle = torch.fft.irfft

        self.nlon = nlon
        self.lmax = min(lmax or self.nlon // 2 + 1, self.nlon // 2 + 1)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.lmax)

    def forward(self, x: torch.Tensor, norm: Optional[str]="ortho") -> torch.Tensor:
        # implicit padding
        y = self.ifft_handle(x, n=self.nlon, dim=-1, norm=norm)

        return y


class RealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat: int, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # use local FFT here
        self.fft_handle = torch.fft.rfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x: torch.Tensor, norm: Optional[str]="ortho") -> torch.Tensor:
        y = self.fft_handle(x, s=(self.nlat, self.nlon), dim=(-2, -1), norm=norm)

        if self.truncate:
            y = torch.cat((y[..., : self.lmax_high, : self.mmax], y[..., -self.lmax_low :, : self.mmax]), dim=-2)

        return y


class InverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat: int, nlon: int, lmax: Optional[int] = None, mmax: Optional[int] = None):
        super().__init__()

        # use local FFT here
        self.ifft_handle = torch.fft.irfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x: torch.Tensor, norm: Optional[str]="ortho") -> torch.Tensor:
        # truncation is implicit but better do it manually
        xt = x[..., : self.mmax]

        if self.truncate:
            # pad
            xth = xt[..., : self.lmax_high, :]
            xtl = xt[..., -self.lmax_low :, :]
            xthp = F.pad(xth, (0, 0, 0, self.nlat - self.lmax))
            xt = torch.cat([xthp, xtl], dim=-2)

        out = self.ifft_handle(xt, s=(self.nlat, self.nlon), dim=(-2, -1), norm=norm)

        return out


class RealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
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
        x = torch.fft.rfftn(x, s=(self.nd, self.nh, self.nw), dim=(-3, -2, -1), norm="ortho")

        # truncate in w
        x = x[..., : self.lwmax]

        # truncate in h
        x = torch.cat([x[..., : self.lhmax_high, :], x[..., -self.lhmax_low :, :]], dim=-2)

        # truncate in d
        x = torch.cat([x[..., : self.ldmax_high, :, :], x[..., -self.ldmax_low :, :, :]], dim=-3)

        return x


class InverseRealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
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

        x = torch.fft.irfftn(x, s=(self.nd, self.nh, self.nw), dim=(-3, -2, -1), norm="ortho")

        return x