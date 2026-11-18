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

from torch_harmonics.quadrature import precompute_latitudes
from torch_harmonics.distributed import split_tensor_along_dim

from makani.distributed import comm
from makani.utils.grids import compute_spherical_bandlimit, sht_grid


class BandLimitMask(torch.nn.Module):
    def __init__(self, img_shape, grid_type, lmax=None, type="sht"):
        super().__init__()
        self.img_shape = img_shape
        self.grid_type = grid_type
        self.lmax = lmax if lmax is not None else compute_spherical_bandlimit(img_shape, grid_type)
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
                grid = sht_grid(img_shape, grid_type)
                self.forward_transform = DistributedRealSHT(grid, lmax=self.lmax, mmax=self.lmax).float()
                self.inverse_transform = DistributedInverseRealSHT(grid, lmax=self.lmax, mmax=self.lmax).float()
            else:
                from torch_harmonics import RealSHT, InverseRealSHT

                grid = sht_grid(img_shape, grid_type)
                self.forward_transform = RealSHT(grid, lmax=self.lmax, mmax=self.lmax).float()
                self.inverse_transform = InverseRealSHT(grid, lmax=self.lmax, mmax=self.lmax).float()

        elif self.type == "fft":

            # get the cutoff frequency in m for each latitude
            lats, _ = precompute_latitudes(self.img_shape[0], grid=self.grid_type)
            # get the grid spacing at the equator
            delta_equator = 2 * torch.pi / (self.lmax - 1)
            mlim = torch.ceil(2 * torch.pi * torch.sin(lats) / delta_equator).reshape(self.img_shape[0], 1)
            ms = torch.arange(self.lmax).reshape(1, -1)
            mask = ms <= mlim
            mask = split_tensor_along_dim(mask, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]
            mask = split_tensor_along_dim(mask, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
            self.register_buffer("mask", mask, persistent=False)

            if comm.get_size("spatial") > 1:
                from makani.models.common.distributed.fft import DistributedRealFFT1, DistributedInverseRealFFT1

                self.forward_transform = DistributedRealFFT1(img_shape[1], lmax=self.lmax, mmax=self.lmax).float()
                self.inverse_transform = DistributedInverseRealFFT1(
                    img_shape[1], lmax=self.lmax, mmax=self.lmax
                ).float()
            else:
                from makani.models.common.fft import RealFFT1, InverseRealFFT1

                self.forward_transform = RealFFT1(img_shape[1], lmax=self.lmax, mmax=self.lmax).float()
                self.inverse_transform = InverseRealFFT1(img_shape[1], lmax=self.lmax, mmax=self.lmax).float()
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
