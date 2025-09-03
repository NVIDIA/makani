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

import os
import copy
from functools import partial
from typing import Union, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from makani.utils import comm
from makani.utils.grids import GridConverter
from physicsnemo.distributed.mappings import reduce_from_parallel_region, copy_to_parallel_region

from makani.models.preprocessor_helpers import get_bias_correction, get_static_features


class Preprocessor2D(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.n_history = params.n_history
        self.history_normalization_mode = params.history_normalization_mode
        if self.history_normalization_mode == "exponential":
            self.history_normalization_decay = params.history_normalization_decay
            # inverse ordering, since first element is oldest
            history_normalization_weights = torch.exp((-self.history_normalization_decay) * torch.arange(start=self.n_history, end=-1, step=-1, dtype=torch.float32))
            history_normalization_weights = history_normalization_weights / torch.sum(history_normalization_weights)
            history_normalization_weights = torch.reshape(history_normalization_weights, (1, -1, 1, 1, 1))
        elif self.history_normalization_mode == "mean":
            history_normalization_weights = torch.as_tensor(1.0 / float(self.n_history + 1), dtype=torch.float32)
            history_normalization_weights = torch.reshape(history_normalization_weights, (1, -1, 1, 1, 1))
        else:
            history_normalization_weights = torch.ones(self.n_history + 1, dtype=torch.float32)
        self.register_buffer("history_normalization_weights", history_normalization_weights, persistent=False)
        self.history_mean = None
        self.history_std = None
        self.history_diff_mean = None
        self.history_diff_var = None
        self.history_eps = 1e-6

        # residual normalization
        self.learn_residual = params.target == "residual"
        if self.learn_residual and (params.normalize_residual):
            with torch.no_grad():
                residual_scale = torch.as_tensor(np.load(params.time_diff_stds_path)).to(torch.float32)
                self.register_buffer("residual_scale", residual_scale, persistent=False)
        else:
            self.residual_scale = None

        # image shape
        self.img_shape = [params.img_shape_x, params.img_shape_y]

        # unpredicted input channels:
        self.unpredicted_inp_train = None
        self.unpredicted_tar_train = None
        self.unpredicted_inp_eval = None
        self.unpredicted_tar_eval = None

        # get bias correction
        bias = get_bias_correction(params)

        if bias is not None:
            # register static buffer
            self.register_buffer("bias_correction", bias, persistent=False)

        # process static features
        static_features = get_static_features(params)
        self.do_add_static_features = False
        if static_features is not None:

            # remember that we need static features
            self.do_add_static_features = True

            # register static buffer
            self.register_buffer("static_features", static_features, persistent=False)

        if hasattr(params, "input_noise"):
            noise_params = params.input_noise
            centered_noise = noise_params.get("centered", False)

            # noise seed: important, this will be passed down as-is
            if not centered_noise:
                seed = 333 + comm.get_rank("model") + comm.get_size("model") * comm.get_rank("data")
                reflect = False
            else:
                # here, ranks (0,1), (2,3), ... should map to the same eff rank, since they only differ by reflection but should otherwise get the
                # same seed
                ensemble_eff_rank = comm.get_rank("ensemble") // 2
                reflect = (comm.get_rank("ensemble") % 2 == 0)
                seed = 333 + comm.get_rank("model") + comm.get_size("model") * ensemble_eff_rank + comm.get_size("model") * comm.get_size("ensemble") * comm.get_rank("batch")

            if "type" not in noise_params:
                raise ValueError("Error, please specify an input noise type")

            self.input_noise_mode = noise_params.get("mode", "concatenate")

            if self.input_noise_mode == "concatenate":
                noise_channels = noise_params.get("n_channels", 1)
            elif self.input_noise_mode == "perturb":
                self.perturb_channels = noise_params.get("perturb_channels", params.channel_names)
                self.perturb_channels = [params.channel_names.index(ch) for ch in self.perturb_channels]
                noise_channels = len(self.perturb_channels)
            else:
                raise NotImplementedError(f"Error, input noise mode {self.input_noise_mode} not supported.")

            if noise_params["type"] == "diffusion":
                from makani.models.noise import DiffusionNoiseS2

                # set the spatio-temporal correlation length
                kT = noise_params.get("kT", 0.5 * (100 / 6370) ** 2)
                lambd = noise_params.get("lambd", params.dt * params.dhours / 6.0)

                self.input_noise = DiffusionNoiseS2(
                    img_shape=self.img_shape,
                    batch_size=params.batch_size,
                    num_channels=noise_channels,
                    num_time_steps=self.n_history + 1,
                    sigma=noise_params.get("sigma", 1.0),
                    kT=kT,  # use various scales
                    lambd=lambd,  # use suggestion here: tau=6h
                    grid_type=params.model_grid_type,
                    seed=seed,
                    reflect=reflect,
                    learnable=noise_params.get("learnable", False)
                )
            elif noise_params["type"] == "white":
                from makani.models.noise import IsotropicGaussianRandomFieldS2

                self.input_noise = IsotropicGaussianRandomFieldS2(
                    img_shape=self.img_shape,
                    batch_size=params.batch_size,
                    num_channels=noise_channels,
                    num_time_steps=self.n_history + 1,
                    sigma=noise_params.get("sigma", 1.0),
                    alpha=noise_params.get("alpha", 0.0),
                    grid_type=params.model_grid_type,
                    seed=seed,
                    reflect=reflect,
                    learnable=noise_params.get("learnable", False)
                )
            elif noise_params["type"] == "dummy":
                from makani.models.noise import DummyNoiseS2

                self.input_noise = DummyNoiseS2(
                    img_shape=self.img_shape,
                    batch_size=params.batch_size,
                    num_channels=noise_channels,
                    num_time_steps=self.n_history + 1,
                )
            else:
                raise NotImplementedError(f'Error, input noise type {noise_params["type"]} not supported.')

    def flatten_history(self, x):
        # flatten input
        if x.dim() == 5:
            b_, t_, c_, h_, w_ = x.shape
            x = torch.reshape(x, (b_, t_ * c_, h_, w_))

        return x

    def expand_history(self, x, nhist):
        if x.dim() == 4:
            b_, ct_, h_, w_ = x.shape
            x = torch.reshape(x, (b_, nhist, ct_ // nhist, h_, w_))
        return x

    def add_residual(self, x, dx):
        if self.learn_residual:
            if self.residual_scale is not None:
                dx = dx * self.residual_scale

            # add residual: deal with history
            x = self.expand_history(x, nhist=self.n_history + 1)
            x[:, -1, ...] = x[:, -1, ...] + dx
            x = self.flatten_history(x)
        else:
            x = dx

        return x

    def add_static_features(self, x):
        if self.do_add_static_features:
            # we need to replicate the grid for each batch:
            static = torch.tile(self.static_features, dims=(x.shape[0], 1, 1, 1))
            x = torch.cat([x, static], dim=1)

        return x

    def remove_static_features(self, x):
        # only remove if something was added in the first place
        if self.do_add_static_features:
            nfeat = self.static_features.shape[1]
            x = x[:, : x.shape[1] - nfeat, :, :]
        return x

    def append_history(self, x1, x2, step, update_state=True):
        r"""
        Take care of unpredicted features first. This is necessary in order to copy the targets unpredicted features
        (such as zenith angle) into the inputs unpredicted features, such that they can be forward in the next
        autoregressive step. extract utar
        """

        # update the unpredicted input
        if update_state:
            if self.training:
                if (self.unpredicted_tar_train is not None) and (step < self.unpredicted_tar_train.shape[1]):
                    utar = self.unpredicted_tar_train[:, step : (step + 1), :, :, :]
                    if self.n_history == 0:
                        self.unpredicted_inp_train.copy_(utar)
                    else:
                        self.unpredicted_inp_train.copy_(torch.cat([self.unpredicted_inp_train[:, 1:, :, :, :], utar], dim=1))
            else:
                if (self.unpredicted_tar_eval is not None) and (step < self.unpredicted_tar_eval.shape[1]):
                    utar = self.unpredicted_tar_eval[:, step : (step + 1), :, :, :]
                    if self.n_history == 0:
                        self.unpredicted_inp_eval.copy_(utar)
                    else:
                        self.unpredicted_inp_eval.copy_(torch.cat([self.unpredicted_inp_eval[:, 1:, :, :, :], utar], dim=1))

        if self.n_history > 0:
            # this is more complicated
            x1 = self.expand_history(x1, nhist=self.n_history + 1)
            x2 = self.expand_history(x2, nhist=1)

            # append
            res = torch.cat([x1[:, 1:, :, :, :], x2], dim=1)

            # flatten again
            res = self.flatten_history(res)
        else:
            res = x2

        return res

    def _append_channels(self, x, xc):

        # x-dimension
        xdim = x.dim()

        # expand history
        x = self.expand_history(x, self.n_history + 1)
        xc = self.expand_history(xc, self.n_history + 1)

        # this routine also adds noise every time a channel gets appended
        if hasattr(self, "input_noise"):
            n = self.input_noise()
            if self.input_noise_mode == "concatenate":
                xc = torch.cat([xc, n], dim=2)
            elif self.input_noise_mode == "perturb":
                x[:, :, self.perturb_channels] = x[:, :, self.perturb_channels] + n

        # concatenate
        xo = torch.cat([x, xc], dim=2)

        # flatten if requested
        if xdim == 4:
            xo = self.flatten_history(xo)

        return xo

    def history_compute_stats(self, x):
        if self.history_normalization_mode == "none":
            self.history_mean = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=x.device)
            self.history_std = torch.ones((1, 1, 1, 1), dtype=torch.float32, device=x.device)
        elif self.history_normalization_mode == "timediff":
            # reshaping
            xdim = x.dim()
            if xdim == 4:
                b_, c_, h_, w_ = x.shape
                xr = torch.reshape(x, (b_, (self.n_history + 1), c_ // (self.n_history + 1), h_, w_))
            else:
                xshape = x.shape
                xr = x

            # time difference mean:
            self.history_diff_mean = torch.mean(torch.sum(xr[:, 1:, ...] - xr[:, 0:-1, ...], dim=(4, 5)), dim=(1, 2))
            # reduce across gpus
            if comm.get_size("spatial") > 1:
                self.history_diff_mean = reduce_from_parallel_region(self.history_diff_mean, "spatial")
            self.history_diff_mean = self.history_diff_mean / float(self.img_shape[0] * self.img_shape[1])

            # time difference std
            self.history_diff_var = torch.mean(torch.sum(torch.square((xr[:, 1:, ...] - xr[:, 0:-1, ...]) - self.history_diff_mean), dim=(4, 5)), dim=(1, 2))
            # reduce across gpus
            if comm.get_size("spatial") > 1:
                self.history_diff_var = reduce_from_parallel_region(self.history_diff_var, "spatial")
            self.history_diff_var = self.history_diff_var / float(self.img_shape[0] * self.img_shape[1])

            # time difference stds
            self.history_diff_mean = copy_to_parallel_region(self.history_diff_mean, "spatial")
            self.history_diff_var = copy_to_parallel_region(self.history_diff_var, "spatial")
        else:
            xdim = x.dim()
            if xdim == 4:
                b_, c_, h_, w_ = x.shape
                xr = torch.reshape(x, (b_, (self.n_history + 1), c_ // (self.n_history + 1), h_, w_))
            else:
                xshape = x.shape
                xr = x

            # mean
            # compute weighted mean over dim 1, but sum over dim=3,4
            self.history_mean = torch.sum(xr * self.history_normalization_weights, dim=(1, 3, 4), keepdim=True)
            # reduce across gpus
            if comm.get_size("spatial") > 1:
                self.history_mean = reduce_from_parallel_region(self.history_mean, "spatial")
            self.history_mean = self.history_mean / float(self.img_shape[0] * self.img_shape[1])

            # compute std
            self.history_std = torch.sum(torch.square(xr - self.history_mean) * self.history_normalization_weights, dim=(1, 3, 4), keepdim=True)
            # reduce across gpus
            if comm.get_size("spatial") > 1:
                self.history_std = reduce_from_parallel_region(self.history_std, "spatial")
            self.history_std = torch.sqrt(self.history_std / float(self.img_shape[0] * self.img_shape[1]))

            # squeeze
            self.history_mean = torch.squeeze(self.history_mean, dim=1)
            self.history_std = torch.squeeze(self.history_std, dim=1)

            # copy to parallel region
            self.history_mean = copy_to_parallel_region(self.history_mean, "spatial")
            self.history_std = copy_to_parallel_region(self.history_std, "spatial")

        return

    def history_normalize(self, x, target=False):
        if self.history_normalization_mode in ["none", "timediff"]:
            return x

        xdim = x.dim()
        if xdim == 4:
            b_, c_, h_, w_ = x.shape
            xr = torch.reshape(x, (b_, (self.n_history + 1), c_ // (self.n_history + 1), h_, w_))
        else:
            xshape = x.shape
            xr = x
            x = self.flatten_history(x)

        # normalize
        if target:
            # strip off the unpredicted channels
            xn = (x - self.history_mean[:, : x.shape[1], :, :]) / self.history_std[:, : x.shape[1], :, :]
        else:
            # tile to include history
            hm = torch.tile(self.history_mean, (1, self.n_history + 1, 1, 1))
            hs = torch.tile(self.history_std, (1, self.n_history + 1, 1, 1))
            xn = (x - hm) / hs

        if xdim == 5:
            xn = torch.reshape(xn, xshape)

        return xn

    def history_denormalize(self, xn, target=False):
        if self.history_normalization_mode in ["none", "timediff"]:
            return xn

        assert self.history_mean is not None
        assert self.history_std is not None

        xndim = xn.dim()
        if xndim == 5:
            xnshape = xn.shape
            xn = self.flatten_history(xn)

        # de-normalize
        if target:
            # strip off the unpredicted channels
            x = xn * self.history_std[:, : xn.shape[1], :, :] + self.history_mean[:, : xn.shape[1], :, :]
        else:
            # tile to include history
            hm = torch.tile(self.history_mean, (1, self.n_history + 1, 1, 1))
            hs = torch.tile(self.history_std, (1, self.n_history + 1, 1, 1))
            x = xn * hs + hm

        if xndim == 5:
            x = torch.reshape(x, xnshape)

        return x

    def cache_unpredicted_features(self, x, y, xz=None, yz=None):
        if self.training:
            if (self.unpredicted_inp_train is not None) and (xz is not None) and (self.unpredicted_inp_train.shape == xz.shape):
                self.unpredicted_inp_train.copy_(xz)
            else:
                self.unpredicted_inp_train = xz

            if (self.unpredicted_tar_train is not None) and (yz is not None) and (self.unpredicted_tar_train.shape == yz.shape):
                self.unpredicted_tar_train.copy_(yz)
            else:
                self.unpredicted_tar_train = yz
        else:
            if (self.unpredicted_inp_eval is not None) and (xz is not None) and (self.unpredicted_inp_eval.shape == xz.shape):
                self.unpredicted_inp_eval.copy_(xz)
            else:
                self.unpredicted_inp_eval = xz

            if (self.unpredicted_tar_eval is not None) and (yz is not None) and (self.unpredicted_tar_eval.shape == yz.shape):
                self.unpredicted_tar_eval.copy_(yz)
            else:
                self.unpredicted_tar_eval = yz

        return x, y

    def get_internal_rng(self, gpu=True):
        if hasattr(self, "input_noise"):
            if gpu:
                return self.input_noise.rng_gpu
            else:
                return self.input_noise.rng_cpu
        else:
            return None
    
    def set_rng(self, reset = True, seed=333):
        if hasattr(self, "input_noise"):
            self.input_noise.set_rng(reset, seed)
            if reset:
                self.input_noise.reset()
        return

    def get_internal_state(self, tensor=False):
        if hasattr(self, "input_noise"):
            if tensor:
                state = self.input_noise.get_tensor_state()
            else:
                state = self.input_noise.get_rng_state()
        else:
            if tensor:
                state = None
            else:
                state = (None, None)

        return state

    def set_internal_state(self, state: Union[Tuple, torch.Tensor]):
        if hasattr(self, "input_noise") and (state is not None):
            if isinstance(state, torch.Tensor):
                self.input_noise.set_tensor_state(state)
            else:
                self.input_noise.set_rng_state(*state)

        return

    def update_internal_state(self, replace_state=False, batch_size=None):
        if hasattr(self, "input_noise"):
            self.input_noise.update(replace_state=replace_state, batch_size=batch_size)
        return

    def append_unpredicted_features(self, inp, target=False):
        if self.training:
            if not target:
                if self.unpredicted_inp_train is not None:
                    inp = self._append_channels(inp, self.unpredicted_inp_train)
            else:
                if self.unpredicted_tar_train is not None:
                    inp = self._append_channels(inp, self.unpredicted_tar_train)
        else:
            if not target:
                if self.unpredicted_inp_eval is not None:
                    inp = self._append_channels(inp, self.unpredicted_inp_eval)
            else:
                if self.unpredicted_tar_eval is not None:
                    inp = self._append_channels(inp, self.unpredicted_tar_eval)
        return inp

    # accessors: clone returned tensors just to be safe
    def get_static_features(self):
        if self.do_add_static_features:
            return self.static_features.clone()
        else:
            return None

    def get_unpredicted_features(self):
        if self.training:
            if self.unpredicted_inp_train is not None:
                inpu = self.unpredicted_inp_train.clone()
            else:
                inpu = None
            if self.unpredicted_tar_train is not None:
                taru = self.unpredicted_tar_train.clone()
            else:
                taru = None
        else:
            if self.unpredicted_inp_eval is not None:
                inpu = self.unpredicted_inp_eval.clone()
            else:
                inpu = None
            if self.unpredicted_tar_eval is not None:
                taru = self.unpredicted_tar_eval.clone()
            else:
                taru = None

        return inpu, taru

    def correct_bias(self, inp: torch.Tensor):
        if hasattr(self, "bias_correction"):
            inp = inp - self.bias_correction
        return inp

def get_preprocessor(params):
    return Preprocessor2D(params)
