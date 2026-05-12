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

import math
import numpy as np
import sys
if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import torch
import torch.nn as nn
from torch import amp

import torch_harmonics as th
import torch_harmonics.distributed as thd

from makani.utils import comm
from torch_harmonics.distributed import split_tensor_along_dim, compute_split_shapes


class BaseNoiseS2(nn.Module):
    def __init__(
        self,
        img_shape,
        batch_size,
        num_channels,
        num_time_steps,
        grid_type="equiangular",
        lmax=None,
        seed=333,
        reflect=False,
        **kwargs,
    ):
        r"""
        Abstract base class for noise on the sphere. Initializes the inverse SHT needed by many of the
        noise classes. Derived noise classes can be stateful or stateless.
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
            self.isht = thd.DistributedInverseRealSHT(self.nlat, self.nlon, lmax=lmax, mmax=lmax, grid=grid_type)
            self.lmax_local = self.isht.l_shapes[comm.get_rank("h")]
            self.mmax_local = self.isht.m_shapes[comm.get_rank("w")]
            self.nlat_local = self.isht.lat_shapes[comm.get_rank("h")]
            self.nlon_local = self.isht.lon_shapes[comm.get_rank("w")]
        else:
            self.isht = th.InverseRealSHT(self.nlat, self.nlon, lmax=lmax, mmax=lmax, grid=grid_type)
            self.lmax_local = self.isht.lmax
            self.mmax_local = self.isht.mmax
            self.nlat_local = self.nlat
            self.nlon_local = self.nlon

        self.lmax = self.isht.lmax
        self.mmax = self.isht.mmax

        # generator objects:
        self.set_rng(seed=seed)

        # allocate the state buffer via the centralized helper; subclasses customize the
        # per-batch shape by overriding _state_shape_suffix.
        self._ensure_state(batch_size, device=torch.device("cpu"), dtype=torch.float32)

    @property
    def _state_shape_suffix(self):
        """
        Shape of the state buffer beyond the batch dim. Subclasses override this to
        customize the layout (e.g. DummyNoiseS2 stores state in spatial, not spectral, form).
        """
        return (self.num_time_steps, self.num_channels, self.lmax_local, self.mmax_local, 2)

    def _ensure_state(self, batch_size, device=None, dtype=None):
        """
        Single source of truth for (re)allocating ``self.state``.

        This is the only method that (re-)registers the state buffer. Calling it with
        the same shape as the current state is a no-op; calling it with a different
        batch size re-registers the buffer so buffer semantics (``.to(device)``,
        ``state_dict`` membership via ``_buffers``, etc.) are preserved rather than
        relying on the ``__setattr__`` hook.
        """
        if device is None:
            device = self.state.device if ("state" in self._buffers) else torch.device("cpu")
        if dtype is None:
            dtype = self.state.dtype if ("state" in self._buffers) else torch.float32

        target_shape = (batch_size,) + tuple(self._state_shape_suffix)
        if ("state" not in self._buffers) or (tuple(self.state.shape) != target_shape):
            self.register_buffer(
                "state",
                torch.zeros(target_shape, dtype=dtype, device=device),
                persistent=False,
            )

    def is_stateful(self):
        raise NotImplementedError("is_stateful method not implemented for this noise class")

    def extra_repr(self):
        return (
            f"img_shape=({self.nlat}, {self.nlon}), "
            f"num_channels={self.num_channels}, num_time_steps={self.num_time_steps}, "
            f"lmax={self.lmax}, reflect={self.reflect}"
        )

    def set_rng(self, seed=333):
        self.rng_cpu = torch.Generator(device=torch.device("cpu"))
        self.rng_cpu.manual_seed(seed)
        if torch.cuda.is_available():
            self.rng_gpu = torch.Generator(device=torch.device(f"cuda:{comm.get_local_rank()}"))
            self.rng_gpu.manual_seed(seed)

    # Resets the internal state. Can be used to change the batch size if required.
    def reset(self, batch_size=None):
        if batch_size is not None:
            self._ensure_state(batch_size)
        with torch.no_grad():
            self.state.zero_()

    # this routine generates a noise sample for a single time step and updates the state accordingly, by appending the last time step
    def update(self, replace_state=False, batch_size=None):

        if batch_size is not None:
            self._ensure_state(batch_size)

        with torch.no_grad():
            newstate = torch.empty_like(self.state)
            if self.state.is_cuda:
                newstate.normal_(mean=0.0, std=1.0, generator=self.rng_gpu)
            else:
                newstate.normal_(mean=0.0, std=1.0, generator=self.rng_cpu)

            if self.reflect:
                newstate = -newstate

            self.state.copy_(newstate)

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
        # Validate the state layout (everything beyond the batch dim) matches this
        # noise module's expected suffix BEFORE touching `self.state`. Only the batch
        # dim may differ — it is auto-resized via `_ensure_state`. Any other
        # difference raises `ValueError` up-front with a useful message instead of
        # silently mutating the state into a bad shape and failing at `copy_()`.
        expected_suffix = tuple(self._state_shape_suffix)
        actual_suffix = tuple(newstate.shape[1:]) if newstate.dim() >= 1 else tuple(newstate.shape)
        if actual_suffix != expected_suffix:
            raise ValueError(
                f"set_tensor_state: shape mismatch beyond batch dim. "
                f"Expected suffix {expected_suffix}, got {actual_suffix} "
                f"(full newstate.shape={tuple(newstate.shape)}, "
                f"current state.shape={tuple(self.state.shape)})."
            )
        if tuple(newstate.shape) != tuple(self.state.shape):
            self._ensure_state(newstate.shape[0])
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
        learnable=False,
        **kwargs,
    ):
        r"""
        GRF on the unit sphere. This implementation follows [1]. This noise is stateless.

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

        # stash config for extra_repr
        self.sigma = sigma
        self.alpha = alpha
        self.learnable = learnable

        if not isinstance(alpha, float):
            alpha = float(alpha)

        # Compute ls, angular power spectrum and sigma_l:
        ls = torch.arange(self.lmax).reshape(-1 ,1)
        ms = torch.arange(self.mmax)
        power_spectrum = torch.pow(2 * ls + 1, -alpha)
        norm_factor = torch.sum((2 * ls + 1) * power_spectrum / 4.0 / math.pi)
        sigma_l = sigma * torch.sqrt(power_spectrum / norm_factor)
        sigma_l = torch.where(ms <= ls, sigma_l, 0.0)

        # the new shape is B, T, C, L, M
        sigma_l = sigma_l.reshape((1, 1, 1, self.lmax, self.mmax)).to(dtype=torch.float32)

        # split tensor
        if comm.get_size("h") > 1:
            sigma_l = split_tensor_along_dim(sigma_l, dim=-2, num_chunks=comm.get_size("h"))[comm.get_rank("h")]

        # split tensor
        if comm.get_size("w") > 1:
            sigma_l = split_tensor_along_dim(sigma_l, dim=-1, num_chunks=comm.get_size("w"))[comm.get_rank("w")]

        # register buffer
        if learnable:
            self.register_parameter("sigma_l", nn.Parameter(sigma_l))
            # sigma_l is sharded by both h and w (different shapes per spatial
            # rank). Replicated only across matmul, so only matmul reduction
            # is meaningful — defaulting to ["model"] would attempt an
            # all_reduce across mismatched shapes. Matmul-replicated → MEAN.
            self.sigma_l.is_shared_mp = ["matmul"]
            self.sigma_l.sharded_dims_mp = [None, None, None, "h", "w"]
        else:
            self.register_buffer("sigma_l", sigma_l, persistent=False)

    @override
    def is_stateful(self):
        return False

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", sigma={self.sigma}, alpha={self.alpha}, learnable={self.learnable}"
        )

    @override
    def forward(self, update_internal_state=False):

        # combine channels and time:
        cstate = torch.view_as_complex(self.state / math.sqrt(2)) * self.sigma_l
        batch_size = cstate.shape[0]

        # flatten history
        cstate = cstate.reshape(batch_size, self.num_time_steps * self.num_channels, self.lmax_local, self.mmax_local)

        # transform
        with amp.autocast(device_type=cstate.device.type, enabled=False):
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
        Supports noising multiple channels at once. This noise is stateful.

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

        # stash config for extra_repr (store originals before processing into tensors below)
        self.sigma = sigma
        self.kT = kT
        self.lambd = lambd
        self.learnable = learnable

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
            # phi is per-channel and broadcast over a state sharded by both h
            # (lmax) and w (mmax). Different spatial ranks accumulate partial
            # gradients into phi → SUM across spatial. Matmul-replicated →
            # MEAN (default).
            self.phi.is_shared_mp = ["matmul", "spatial"]
            self.phi.sharded_dims_mp = [None, None, None]
            self.phi.is_shared_mp_op = {"spatial": "sum"}
            self.sigma_l = nn.Parameter(sigma_l)
            # sigma_l has lmax sharded by h (different h-ranks already hold
            # disjoint slices, no reduction across h), but no mmax dim, so it
            # broadcasts over a w-sharded mmax → SUM across w. Matmul default
            # MEAN is correct. Cannot use "spatial" here because that would
            # imply replication across h, which is false.
            self.sigma_l.is_shared_mp = ["matmul", "w"]
            self.sigma_l.sharded_dims_mp = [None, None, None, "h", None, None]
            self.sigma_l.is_shared_mp_op = {"w": "sum"}
        else:
            self.register_buffer("phi", phi, persistent=False)
            self.register_buffer("sigma_l", sigma_l, persistent=False)

        # state buffer is already allocated by BaseNoiseS2.__init__ via _ensure_state

        # if num_time_steps > 1, we need the toeplitz matrix for the discounts:
        #            [    1,     0,   0, 0]
        # discount = [  phi,     1,   0, 0]
        #            [phi^2,   phi,   1, 0]
        #            [phi^3, phi^2, phi, 1]
        if self.num_time_steps > 1:
            if learnable:
                raise NotImplementedError(f"num_time_steps>1 learnable diffusion noise not supported")

            discount = []
            phi_flat = self.phi.reshape(-1)
            for phi_tmp in phi_flat.tolist():
                phivec = np.power(phi_tmp, np.arange(0, self.num_time_steps))
                disc = torch.as_tensor(toep(phivec, np.zeros(self.num_time_steps)))
                disc = disc.to(dtype=torch.float32)
                discount.append(disc)
            discount = torch.stack(discount, dim=0)
            self.register_buffer("discount", discount, persistent=False)

    @override
    def is_stateful(self):
        return True

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", sigma={self.sigma}, kT={self.kT}, lambd={self.lambd}, learnable={self.learnable}"
        )

    # this routine generates a noise sample for a single time step and updates the state accordingly, by appending the last time step
    @override
    def update(self, replace_state=False, batch_size=None):
        if batch_size is not None:
            self._ensure_state(batch_size)

        with torch.no_grad():
            with amp.autocast(device_type=self.state.device.type, enabled=False):
                # draw either the full T-step history (replace) or a single step (AR)
                if replace_state:
                    eta_l = torch.empty_like(self.state)
                else:
                    B = self.state.shape[0]
                    eta_l = torch.empty(
                        (B, 1, self.num_channels, self.lmax_local, self.mmax_local, 2),
                        dtype=self.state.dtype, device=self.state.device,
                    )
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
                        last_state = self.state[:, -1, ...].unsqueeze(1)
                        newstep = self.phi * last_state + eta_l
                        newstate = torch.cat([self.state[:, 1:, ...], newstep], dim=1)
                    else:
                        newstate = self.phi * self.state + eta_l
                else:
                    newstate = eta_l
                    # the very first element in the time history requires a different weighting to sample the stationary distribution
                    newstate[:, 0, ...] = newstate[:, 0, ...] / torch.sqrt(1.0 - self.phi**2)
                    # get the right history by multiplying with the discount matrix
                    if self.num_time_steps > 1:
                        newstate = torch.einsum("ctr,brclmu->btclmu", self.discount, newstate).contiguous()

                # shape matches self.state after _ensure_state above
                self.state.copy_(newstate)

        return

    @override
    def forward(self, update_internal_state=False):

        # combine channels and time:
        cstate = torch.view_as_complex(self.state)
        batch_size = cstate.shape[0]

        # flatten history
        cstate = cstate.reshape(batch_size, self.num_time_steps * self.num_channels, self.lmax_local, self.mmax_local)

        # transform
        with amp.autocast(device_type=cstate.device.type, enabled=False):
            eta = self.isht(cstate)

        # expand history
        eta = eta.reshape(batch_size, self.num_time_steps, self.num_channels, self.nlat_local, self.nlon_local)

        # update the internal state if requested
        if update_internal_state:
            self.update()

        return eta


class DummyNoiseS2(BaseNoiseS2):
    def __init__(
        self,
        img_shape,
        batch_size,
        num_channels,
        num_time_steps=1,
        mode="constant_zero",
        seed=333,
        **kwargs,
    ):
        r"""
        Dummy noise module for testing and debugging. This noise is stateless.

        The module always emits a tensor with the correct output shape (B, T, C, H, W)
        but carries no stochastic signal beyond what the chosen mode specifies.

        Supported modes
        ---------------
        constant_zero (default)
            Always emits an all-zero tensor. Useful for verifying shape consistency
            of the noise pipeline without introducing any stochastic signal. In particular,
            when the preprocessor is configured in 'perturb' mode the noise is *added* to
            the model input channels. Returning zeros guarantees that the input is not
            modified, so integration tests can check shapes and control-flow correctness
            without having to account for random perturbations.

        constant_random
            Draws a spatially-uniform Gaussian tensor once per update() call and holds it
            fixed until the next update(). Useful for verifying that the model handles a
            non-zero, reproducible noise pattern correctly without the overhead of the
            spherical-harmonic transform used by the real noise classes.

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
        mode : str, default 'constant_zero'
            Output mode. One of 'constant_zero' or 'constant_random'.
        seed : int, default 333
            Random seed used in 'constant_random' mode; ignored otherwise.
        """

        if mode not in ("constant_zero", "constant_random"):
            raise ValueError(f"DummyNoiseS2: unknown mode '{mode}'. "
                             f"Expected 'constant_zero' or 'constant_random'.")

        self.mode = mode

        # BaseNoiseS2.__init__ sets up nlat/nlon, comm splits, rng_cpu/rng_gpu, and
        # allocates the state buffer via _ensure_state. The shape is picked up through
        # our override of _state_shape_suffix below, which gives a spatial (H, W)
        # layout instead of the spectral (L, M, 2) default.
        super().__init__(
            img_shape=img_shape,
            batch_size=batch_size,
            num_channels=num_channels,
            num_time_steps=num_time_steps,
            seed=seed,
        )

    @property
    def _state_shape_suffix(self):
        # spatial (H, W) rather than spectral (L, M, 2)
        return (self.num_time_steps, self.num_channels, self.nlat_local, self.nlon_local)

    @override
    def is_stateful(self):
        return False

    def extra_repr(self):
        return super().extra_repr() + f", mode={self.mode}"

    @override
    def update(self, replace_state=False, batch_size=None):
        if batch_size is not None:
            self._ensure_state(batch_size)

        with torch.no_grad():
            newstate = torch.empty_like(self.state)

            if self.mode == "constant_zero":
                newstate.zero_()
            else:  # constant_random
                if self.state.is_cuda:
                    newstate.normal_(mean=0.0, std=1.0, generator=self.rng_gpu)
                else:
                    newstate.normal_(mean=0.0, std=1.0, generator=self.rng_cpu)

            self.state.copy_(newstate)

        return

    @override
    def forward(self, update_internal_state=False):

        state = self.state

        # update the internal state if requested
        if update_internal_state:
            self.update()

        return state
