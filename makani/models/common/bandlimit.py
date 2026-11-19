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

"""Band limiting on the sphere: the modes above a degree removed with the SHT, or per latitude with the FFT."""

import torch
import torch.amp as amp

from torch_harmonics.distributed import split_tensor_along_dim

from makani.distributed import comm
from makani.utils.grids import spherical_bandlimit


class BandLimitMask(torch.nn.Module):
    r"""
    Removes the modes of a field on the sphere above a band limit.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2
        The grid of the field, see :func:`~makani.utils.grids.sht_grid`.
    lmax : int, optional
        The band limit, exclusive; :func:`~makani.utils.grids.spherical_bandlimit` of the grid by
        default.
    type : str, optional
        ``"sht"`` keeps the degrees below ``lmax``; ``"fft"`` keeps, on each latitude, the zonal
        wavenumbers its circle resolves at the grid spacing ``lmax`` gives the equator.
    """

    def __init__(self, grid, lmax=None, type="sht"):
        super().__init__()
        self.grid = grid
        self.lmax = lmax if lmax is not None else spherical_bandlimit(grid)
        self.type = type

        if self.type == "sht":
            # SHT for the computation of SH coefficients
            if comm.get_size("spatial") > 1:
                from torch_harmonics.distributed import DistributedRealSHT, DistributedInverseRealSHT
                import torch_harmonics.distributed as thd

                if not thd.is_initialized():
                    polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                    azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                    thd.init(polar_group, azimuth_group)
                self.forward_transform = DistributedRealSHT(grid, lmax=self.lmax, mmax=self.lmax).float()
                self.inverse_transform = DistributedInverseRealSHT(grid, lmax=self.lmax, mmax=self.lmax).float()
            else:
                from torch_harmonics import RealSHT, InverseRealSHT

                self.forward_transform = RealSHT(grid, lmax=self.lmax, mmax=self.lmax).float()
                self.inverse_transform = InverseRealSHT(grid, lmax=self.lmax, mmax=self.lmax).float()

        elif self.type == "fft":

            # get the cutoff frequency in m for each latitude, from the colatitudes of the grid
            colats = grid.colats
            # get the grid spacing at the equator
            delta_equator = 2 * torch.pi / (self.lmax - 1)
            mlim = torch.ceil(2 * torch.pi * torch.sin(colats) / delta_equator).reshape(grid.nlat, 1)
            ms = torch.arange(self.lmax).reshape(1, -1)
            mask = ms <= mlim
            mask = split_tensor_along_dim(mask, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]
            mask = split_tensor_along_dim(mask, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
            self.register_buffer("mask", mask, persistent=False)

            if comm.get_size("spatial") > 1:
                from makani.models.common.distributed.fft import DistributedRealFFT1, DistributedInverseRealFFT1

                self.forward_transform = DistributedRealFFT1(grid, lmax=self.lmax, mmax=self.lmax).float()
                self.inverse_transform = DistributedInverseRealFFT1(grid, lmax=self.lmax, mmax=self.lmax).float()
            else:
                from makani.models.common.fft import RealFFT1, InverseRealFFT1

                self.forward_transform = RealFFT1(grid, lmax=self.lmax, mmax=self.lmax).float()
                self.inverse_transform = InverseRealFFT1(grid, lmax=self.lmax, mmax=self.lmax).float()
        else:
            raise ValueError(f"Unknown truncation type {self.type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        with amp.autocast(device_type="cuda", enabled=False):
            dtype = x.dtype
            x = x.float()

            x = self.forward_transform(x)

            if hasattr(self, "mask"):
                x = torch.where(self.mask, x, torch.zeros_like(x))

            x = self.inverse_transform(x)

            x = x.to(dtype=dtype)

        return x
