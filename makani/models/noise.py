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

from abc import ABCMeta, abstractmethod
from typing import List

import math
import numpy as np
from scipy.special import zeta

import torch
import torch.nn as nn
from torch import amp

import torch_harmonics as th
import torch_harmonics.distributed as thd

from makani.utils import comm
from physicsnemo.distributed.utils import split_tensor_along_dim, compute_split_shapes


class BaseNoiseS2(nn.Module):
    def __init__(
        self,
        img_shape,
        batch_size,
        num_channels,
        num_time_steps,
        grid_type="equiangular",
        seed=333,
        reflect=False,
        **kwargs,
    ):
        r"""
        Abstract base class for noise on the sphere. Initializes the inverse SHT needed by many of the
        noise classes.
        """
        super().__init__()

        # Number of latitudinal modes.
        self.nlat, self.nlon = img_shape
        self.num_channels = num_channels
        self.num_time_steps = num_time_steps
        self.reflect = reflect

        # Inverse SHT
        if comm.get_size("spatial") > 1:
            if not thd.is_initialized():
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
            self.isht = thd.DistributedInverseRealSHT(self.nlat, self.nlon, grid=grid_type)
            self.lmax_local = self.isht.l_shapes[comm.get_rank("h")]
            self.mmax_local = self.isht.m_shapes[comm.get_rank("w")]
            self.nlat_local = self.isht.lat_shapes[comm.get_rank("h")]
            self.nlon_local = self.isht.lon_shapes[comm.get_rank("w")]
        else:
            self.isht = th.InverseRealSHT(self.nlat, self.nlon, grid=grid_type)
            self.lmax_local = self.isht.lmax
            self.mmax_local = self.isht.mmax
            self.nlat_local = self.nlat
            self.nlon_local = self.nlon

        self.lmax = self.isht.lmax
        self.mmax = self.isht.mmax

        # generator objects:
        self.set_rng(seed=seed)

        # store the noise state: initialize to None
        self.register_buffer("state", torch.zeros((batch_size, self.num_time_steps, self.num_channels, self.lmax_local, self.mmax_local, 2), dtype=torch.float32), persistent=False)

    def set_rng(self, seed=333):
        self.rng_cpu = torch.Generator(device=torch.device("cpu"))
        self.rng_cpu.manual_seed(seed)
        if torch.cuda.is_available():
            self.rng_gpu = torch.Generator(device=torch.device(f"cuda:{comm.get_local_rank()}"))
            self.rng_gpu.manual_seed(seed)

    # Resets the internal state. Can be used to change the batch size if required.
    def reset(self, batch_size=None):
        if self.state is not None:

            if batch_size is not None:
                self.state = torch.zeros(batch_size, self.num_time_steps, self.num_channels, self.lmax_local, self.mmax_local, 2, dtype=self.state.dtype, device=self.state.device)

            with torch.no_grad():
                self.state.fill_(0.0)

    # this routine generates a noise sample for a single time step and updates the state accordingly, by appending the last time step
    def update(self, replace_state=False, batch_size=None):

         if replace_state:
            # create single occurence
            with torch.no_grad():
                if batch_size is None:
                    batch_size = self.state.shape[0]
                newstate = torch.empty((batch_size, self.num_time_steps, self.num_channels, self.lmax_local, self.mmax_local, 2), dtype=self.state.dtype, device=self.state.device)
                if self.state.is_cuda:
                    newstate.normal_(mean=0.0, std=1.0, generator=self.rng_gpu)
                else:
                    newstate.normal_(mean=0.0, std=1.0, generator=self.rng_cpu)

                if self.reflect:
                    newstate = -newstate

                if newstate.shape == self.state.shape:
                    self.state.copy_(newstate)
                else:
                    self.state = newstate

        return

    def set_rng_state(self, cpu_state, gpu_state):
        if cpu_state is not None:
            self.rng_cpu.set_state(cpu_state)
        if torch.cuda.is_available() and (gpu_state is not None):
            self.rng_gpu.set_state(gpu_state)

        return

    def get_rng_state(self):
        cpu_state = self.rng_cpu.get_state()
        gpu_state = None
        if torch.cuda.is_available():
            gpu_state = self.rng_gpu.get_state()

        return cpu_state, gpu_state

    def get_tensor_state(self):
        return self.state.detach().clone()

    def set_tensor_state(self, newstate):
        with torch.no_grad():
            self.state.copy_(newstate)
        return


class IsotropicGaussianRandomFieldS2(BaseNoiseS2):
    def __init__(
        self,
        img_shape,
        batch_size,
        num_channels,
        num_time_steps=1,
        sigma=1.0,
        alpha=0.0,
        grid_type="equiangular",
        seed=333,
        reflect=False,
        learnable =False,
        **kwargs,
    ):
        r"""
        GRF on the unit sphere. This implementation follows [1].

        References
        ============
        [1] Lang, A.; Schwab C.; ISOTROPIC GAUSSIAN RANDOM FIELDS ON THE SPHERE: REGULARITY, FAST SIMULATION AND STOCHASTIC PARTIAL DIFFERENTIAL EQUATIONS; The Annals of Applied Probability; 2015, Vol. 25, No. 6, 3047-3094; DOI: 10.1214/14-AAP1067

        Parameters
        ============
        img_shape : (int, int)
            Number of latitudinal and longitudinal modes
        batch_size: int
            Batch size for the noise
        num_channels: int
            Number of channels for the noise
        sigma : float, default is 1.0
            Scale parameter corresponding to the diagonal entry of the covariance kernel
        alpha: float, default is 0.0
            Decay factor in the angular power spectrum. White noise corresponds to alpha = 0.0
        grid_type : string, default is "equiangular"
            Grid type. Currently supports "equiangular" and
            "legendre-gauss".
        learnable : bool, default is False
            Parameter which enables learnable Gaussian noise
        """
        super().__init__(img_shape=img_shape, batch_size=batch_size, num_channels=num_channels, num_time_steps=num_time_steps, grid_type=grid_type, seed=seed, reflect=reflect)

        if not isinstance(alpha, float):
            alpha = float(alpha)

        # Compute ls, angular power spectrum and sigma_l:
        ls = torch.arange(self.lmax)
        power_spectrum = torch.pow(2 * ls + 1, -alpha)
        norm_factor = torch.sum((2 * ls + 1) * power_spectrum / 4.0 / math.pi)
        sigma_l = sigma * torch.sqrt(power_spectrum / norm_factor)

        # the new shape is B, T, C, L, M
        sigma_l = sigma_l.reshape((1, 1, 1, self.lmax, 1)).to(dtype=torch.float32)

        # split tensor
        if comm.get_size("h") > 1:
            sigma_l = split_tensor_along_dim(sigma_l, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]

        # register buffer
        if learnable:
            self.register_parameter("sigma_l", nn.Parameter(sigma_l))
        else:
            self.register_buffer("sigma_l", sigma_l, persistent=False)

    def forward(self, update_internal_state=False):

        # combine channels and time:
        cstate = torch.view_as_complex(self.state / math.sqrt(2)) * self.sigma_l
        batch_size = cstate.shape[0]

        # flatten history
        cstate = cstate.reshape(batch_size, self.num_time_steps * self.num_channels, self.lmax_local, self.mmax_local)

        # transform
        with amp.autocast(device_type="cuda", enabled=False):
            eta = self.isht(cstate)

        # expand history
        eta = eta.reshape(batch_size, self.num_time_steps, self.num_channels, self.nlat_local, self.nlon_local)

        # update the internal state if requested
        if update_internal_state:
            self.update()

        return eta


# taken from scipy: https://github.com/scipy/scipy/blob/v1.13.0/scipy/linalg/_special_matrices.py#L17-L77
def toep(c, r=None):

    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()
    # Form a 1-D array containing a reversed c followed by r[1:] that could be
    # strided to give us toeplitz matrix.
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]

    return np.lib.stride_tricks.as_strided(vals[len(c) - 1 :], shape=out_shp, strides=(-n, n)).copy()


class DiffusionNoiseS2(BaseNoiseS2):
    def __init__(
        self,
        img_shape,
        batch_size,
        num_channels,
        num_time_steps=1,
        sigma=1.0,
        kT=0.5 * (500.0 / 6370.0) ** 2,
        lambd=1.0,
        grid_type="equiangular",
        seed=333,
        reflect=False,
        learnable =False,
        **kwargs,
    ):
        r"""
        A Random Field derived from a gaussian Diffusion Process on the sphere:

        For details see https://www.ecmwf.int/sites/default/files/elibrary/2009/11577-stochastic-parametrization-and-model-uncertainty.pdf,
        appendix 8.1.
        Supports noising multiple channels at once

        img_shape : (int, int)
            Number of latitudinal and longitudinal modes
        batch_size: int
            Batch size for the noise
        num_channels: int
            Number of channels for the noise
        sigma : float, default is 1
            Stationary standard deviation
        kT : float or List, default is 0.5 * (500 km / 6370 km)^2 = 0.00308057
            Spatial correlation length. If this is a list it has to match num_channels.
        lambd : float or List, default is 1.0
            Temporal correlation length, should be set to (t / tau). If this is a list it has to match num_channels.
        grid_type : string, default is "equiangular"
            Grid type. Currently supports "equiangular" and
            "legendre-gauss".
        learnable : bool, default is False
            Parameter which enables learnable Diffusion noise
        """
        super().__init__(img_shape=img_shape, batch_size=batch_size, num_channels=num_channels, num_time_steps=num_time_steps, grid_type=grid_type, seed=seed, reflect=reflect)

        # Compute l:
        ls = torch.arange(self.lmax)

        # make sure kT is a torch.Tensor
        if isinstance(kT, list):
            kT = torch.as_tensor(kT)
            assert len(kT.shape) == 1
            assert kT.shape[0] == num_channels
        else:
            kT = torch.as_tensor([kT]).repeat(num_channels)
        kT = kT.reshape(self.num_channels, 1)

        # same for lambd
        if isinstance(lambd, list):
            lambd = torch.as_tensor(lambd)
            assert len(lambd.shape) == 1
            assert lambd.shape[0] == num_channels
        else:
            lambd = torch.as_tensor([lambd]).repeat(num_channels)
        lambd = lambd.reshape(self.num_channels, 1)

        # f-tensor:
        ektllp1 = torch.exp(-kT * ls * (ls + 1))
        F0norm = torch.sum((2 * ls[1:] + 1) * ektllp1[..., 1:], dim=-1, keepdim=True)
        # create a discount vector in time:
        phi = torch.exp(-lambd)
        F0 = sigma * torch.sqrt(0.5 * (1 - phi**2) / F0norm)
        sigma_l = F0 * torch.exp(-0.5 * kT * ls * (ls + 1))
        # we multiply by 4 pi to get the correct variance. Check ECMWF docs and their Spherical Harmonic normalization
        sigma_l = math.sqrt(4 * math.pi) * sigma_l

        # the new shape is C, L, M
        phi = phi.reshape((self.num_channels, 1, 1)).to(dtype=torch.float32)
        # the new shape is B, T, C, L, M
        sigma_l = sigma_l.reshape((1, 1, self.num_channels, self.lmax, 1)).to(dtype=torch.float32)

        # split tensor
        if comm.get_size("h") > 1:
            sigma_l = split_tensor_along_dim(sigma_l, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]

        # unsqueeze complex dim
        phi = phi.unsqueeze(-1)
        sigma_l = sigma_l.unsqueeze(-1)

        # register buffer
        if learnable:
            self.phi = nn.Parameter(phi)
            self.phi.is_shared_mp = ["matmul", "h", "w"]
            self.phi.sharded_dims_mp = [None, None, None]
            self.sigma_l = nn.Parameter(sigma_l)
            self.sigma_l.is_shared_mp = ["matmul", "w"]
            self.sigma_l.sharded_dims_mp = [None, None, None, "h", None, None]
        else:
            self.register_buffer("phi", phi, persistent=False)
            self.register_buffer("sigma_l", sigma_l, persistent=False)

        # store the noise state: initialize to None
        self.register_buffer("state", torch.zeros((batch_size, self.num_time_steps, self.num_channels, self.lmax_local, self.mmax_local, 2), dtype=torch.float32), persistent=False)

        # if num_time_steps > 1, we need the toeplitz matrix for the discounts:
        #            [    1,     0,   0, 0]
        # discount = [  phi,     1,   0, 0]
        #            [phi^2,   phi,   1, 0]
        #            [phi^3, phi^2, phi, 1]
        if self.num_time_steps > 1:
            if learnable:
                raise NotImplementedError(f"num_time_steps>1 learnable diffusion noise not supported")

            discount = []
            for phi in self.phi.reshape(-1).tolist():
                print(phi)
                phivec = np.power(self.phi, np.arange(0, self.num_time_steps))
                disc = torch.tensor(toep(phivec, np.zeros(self.num_time_steps)))
                disc = disc.to(dtype=torch.float32)
                discount.append(disc)
            discount = torch.stack(discount)
            print(discount.shape)
            self.register_buffer("discount", discount, persistent=False)

    # this routine generates a noise sample for a single time step and updates the state accordingly, by appending the last time step
    def update(self, replace_state=False, batch_size=None):

        # create single occurence
        with torch.no_grad():
            nsteps = self.num_time_steps if replace_state else 1
            if batch_size is None:
                batch_size = self.state.shape[0]
            eta_l = torch.empty((batch_size, nsteps, self.num_channels, self.lmax_local, self.mmax_local, 2), dtype=torch.float32, device=self.state.device)
            if self.state.is_cuda:
                eta_l.normal_(mean=0.0, std=1.0, generator=self.rng_gpu)
            else:
                eta_l.normal_(mean=0.0, std=1.0, generator=self.rng_cpu)

            # multiply by sigma
            eta_l = self.sigma_l * eta_l

            # reflect if required:
            if self.reflect:
                eta_l = -eta_l

            if not replace_state:
                # update previous state
                if self.num_time_steps > 1:
                    newstate = self.phi * self.state[:, -1, ...] + eta_l.squeeze(1)
                    newstate = torch.cat([self.state[:, 1:, ...], newstate.unsqueeze(1)], dim=1)
                else:
                    newstate = self.phi * self.state + eta_l
            else:
                newstate = eta_l
                # the very first element in the time history requires a different weighting to sample the stationary distribution
                newstate[:, 0, ...] = newstate[:, 0, ...] / torch.sqrt(1.0 - self.phi**2)
                # get the right history by multiplying with the discount matrix
                if self.num_time_steps > 1:
                    newstate = torch.einsum("ctr,brclmu->btclmu", self.discount, newstate)

            # update the state
            if newstate.shape == self.state.shape:
                self.state.copy_(newstate)
            else:
                self.state = newstate

        return

    def forward(self, update_internal_state=False):

        # combine channels and time:
        cstate = torch.view_as_complex(self.state)
        batch_size = cstate.shape[0]

        # flatten history
        cstate = cstate.reshape(batch_size, self.num_time_steps * self.num_channels, self.lmax_local, self.mmax_local)

        # transform
        with amp.autocast(device_type="cuda", enabled=False):
            eta = self.isht(cstate)

        # expand history
        eta = eta.reshape(batch_size, self.num_time_steps, self.num_channels, self.nlat_local, self.nlon_local)

        # update the internal state if requested
        if update_internal_state:
            self.update()

        return eta


class DummyNoiseS2(nn.Module):
    def __init__(
        self,
        img_shape,
        batch_size,
        num_channels,
        num_time_steps=1,
        **kwargs,
    ):
        r"""
        Dummy noise module for debugging purposes.

        Parameters
        ============
        img_shape : (int, int)
            Number of latitudinal and longitudinal modes
        batch_size: int
            Batch size for the noise
        num_channels: int
            Number of channels for the noise
        num_time_steps: int
            Number of time steps
        """

        super().__init__()

        # Number of latitudinal modes.
        self.nlat, self.nlon = img_shape
        self.num_channels = num_channels
        self.num_time_steps = num_time_steps

        if comm.get_size("spatial") > 1:
            lat_shapes = compute_split_shapes(self.nlat, comm.get_size("h"))
            lon_shapes = compute_split_shapes(self.nlon, comm.get_size("w"))
            self.nlat_local = lat_shapes[comm.get_rank("h")]
            self.nlon_local = lon_shapes[comm.get_rank("w")]
        else:
            self.nlat_local = self.nlat
            self.nlon_local = self.nlon

        # store the noise state: initialize to None
        self.register_buffer("state", torch.zeros((batch_size, self.num_time_steps, self.num_channels, self.nlat_local, self.nlon_local), dtype=torch.float32), persistent=False)

    def update(self, replace_state=False, batch_size=None):

        # create single occurence
        with torch.no_grad():
            if batch_size is None:
                batch_size = self.state.shape[0]
            newstate = torch.zeros((batch_size, self.num_time_steps, self.num_channels, self.nlat_local, self.nlon_local), dtype=self.state.dtype, device=self.state.device)

            if newstate.shape == self.state.shape:
                self.state.copy_(newstate)
            else:
                self.state = newstate

        return

    def forward(self, update_internal_state=False):

        # combine channels and time:
        state = self.state

        # update the internal state if requested
        if update_internal_state:
            self.update()

        return state
