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

from typing import Optional, Tuple, List
from functools import partial
import math

import numpy as np

import torch
from torch import nn

from makani.utils import comm
from makani.utils.grids import GridQuadrature
from makani.utils.dataloaders.data_helpers import get_data_normalization
from physicsnemo.distributed.utils import compute_split_shapes
from physicsnemo.distributed.mappings import gather_from_parallel_region, reduce_from_parallel_region

import torch_harmonics as harmonics
from torch_harmonics.quadrature import clenshaw_curtiss_weights, legendre_gauss_weights

from .losses import LossType, GeometricLpLoss, SpectralH1Loss, SpectralAMSELoss, HydrostaticBalanceLoss
from .losses import EnsembleCRPSLoss, EnsembleSpectralCRPSLoss, EnsembleNLLLoss, EnsembleMMDLoss
from .losses import DriftRegularization


class LossHandler(nn.Module):
    """
    Wrapper class that will handle computing losses. Each loss term returns a vector of losses,
    which in the end gets weighted and aggregated.
    """

    def __init__(self, params, track_running_stats: bool = False, seed: int = 0, eps: float = 1e-6, **kwargs):
        super().__init__()

        self.rank = comm.get_rank("matmul")
        self.n_future = params.n_future
        self.spatial_distributed = comm.is_distributed("spatial") and (comm.get_size("spatial") > 1)
        self.ensemble_distributed = comm.is_distributed("ensemble") and (comm.get_size("ensemble") > 1)

        # get global image shape
        self.img_shape = (params.img_shape_x, params.img_shape_y)
        self.crop_shape = (params.img_crop_shape_x, params.img_crop_shape_y)
        self.crop_offset = (params.img_crop_offset_x, params.img_crop_offset_y)

        # check whether dynamic loss weighting is required
        self.uncertainty_weighting = params.get("uncertainty_weighting", False)
        self.randomized_loss_weights = params.get("randomized_loss_weights", False)
        self.random_slice_loss = params.get("random_slice_loss", False)

        # whether to keep running stats
        self.track_running_stats = track_running_stats or self.uncertainty_weighting
        self.eps = eps

        n_channels = len(params.channel_names)

        # determine channel weighting
        if hasattr(params, "losses"):
            losses = params.losses
        elif hasattr(params, "loss"):
            losses = [{"type": params.loss}]
        else:
            raise ValueError("No loss function specified.")

        # load normalization term:
        bias, scale = get_data_normalization(params)
        if bias is not None:
            bias = torch.from_numpy(bias)[:, params.out_channels, ...].to(torch.float32)
        else:
            bias = torch.zeros((1, len(params.out_channels), 1, 1), dtype=torch.float32)

        if scale is not None:
            scale = torch.from_numpy(scale)[:, params.out_channels, ...].to(torch.float32)
        else:
            scale = torch.ones((1, len(params.out_channels), 1, 1), dtype=torch.float32)

        # create module list
        self.loss_fn = nn.ModuleList([])

        channel_weights = []

        for loss in losses:
            loss_type = loss["type"]

            # get pole mask if it was specified
            pole_mask = loss.get("pole_mask", 0)

            # get extra loss arguments if specified
            loss_params = loss.get("parameters", {})

            # get the loss function object
            loss_handle = self._parse_loss_type(loss_type)
            loss_fn = loss_handle(
                img_shape=self.img_shape,
                crop_shape=self.crop_shape,
                crop_offset=self.crop_offset,
                channel_names=params.channel_names,
                pole_mask=pole_mask,
                bias=bias,
                scale=scale,
                grid_type=params.model_grid_type,
                spatial_distributed=self.spatial_distributed,
                ensemble_distributed=self.ensemble_distributed,
                **loss_params,
            )

            # append to dict and compile before:
            # TODO: fix the compile issue
            # self.loss_fn[loss_type] = torch.compile(loss_fn)
            self.loss_fn.append(loss_fn)

            # determine channel weighting
            if "channel_weights" not in loss.keys():
                channel_weight_type = "constant"
            else:
                channel_weight_type = loss["channel_weights"]

            if isinstance(channel_weight_type, List):
                chw = torch.tensor(channel_weight_type, dtype=torch.float32).reshape(1, -1)
                assert chw.shape[1] == loss_fn.n_channels
            else:
                chw = loss_fn.compute_channel_weighting(channel_weight_type)

            # the option to normalize outputs with stds of the time difference rather than th
            if ("temp_diff_normalization" in loss.keys()) and loss["temp_diff_normalization"]:

                # extract relevant stds
                time_diff_stds = torch.from_numpy(np.load(params.time_diff_stds_path)).reshape(1, -1)[:, params.out_channels]
                # the time differences are computed between two consecutive datapoints,
                # so we need to account for the number of timesteps used in the prediction
                # this is now commebnted out as we expect the stats to be computed with the correct dt
                # time_diff_stds *= np.sqrt(params.dt)

                # to avoid division by  very small numbers, we clamp the time differences from below
                time_diff_stds = torch.clamp(time_diff_stds, min=1e-4)

                time_var_weights = scale.reshape(1, -1) / time_diff_stds

                if hasattr(loss_fn, "squared") and loss_fn.squared:
                    time_var_weights = time_var_weights**2

                chw = chw * time_var_weights

            chw = chw.reshape(1, -1)

            # check for a relative weight that weights the loss relative to other losses
            if "relative_weight" in loss.keys():
                chw *= loss["relative_weight"]

            channel_weights.append(chw)

        channel_weights = torch.cat(channel_weights, dim=1)
        ncw = channel_weights.shape[1]
        self.register_buffer("channel_weights", channel_weights)

        # set up tensor to track running stats
        # those need to have the same dimensions as the
        # the m2 buffer is filled with a very small non-zero value to avoid division by zero early on
        stats_buffer_shape = (self.n_future + 1) * channel_weights.shape[-1]
        self.register_buffer("running_mean", torch.zeros(stats_buffer_shape))
        self.register_buffer("running_var", torch.ones(stats_buffer_shape))
        self.register_buffer("num_batches_tracked", torch.LongTensor([0]))

        # weighting factor for multistep, by default a uniform weight is used
        if params.get("multistep_loss_weight", False):
            multistep_weight = 2.0 * torch.arange(1, self.n_future + 2, dtype=torch.float32) / float((self.n_future + 2) * (self.n_future + 1))
        else:
            # uniform weighting factor for the case of multistep training
            multistep_weight = torch.ones(self.n_future + 1, dtype=torch.float32) / float(self.n_future + 1)
        # tile multistep_weights in channel_dim, but channel_dim needs to be fastest dim
        multistep_weight = torch.repeat_interleave(multistep_weight.reshape(1, -1), ncw, dim=1)
        self.register_buffer("multistep_weight", multistep_weight)

        # generator objects:
        seed = seed
        self.rng_cpu = torch.Generator(device=torch.device("cpu"))
        self.rng_cpu.manual_seed(seed)
        if torch.cuda.is_available():
            self.rng_gpu = torch.Generator(device=torch.device(f"cuda:{comm.get_local_rank()}"))
            self.rng_gpu.manual_seed(seed)

    @torch.compiler.disable(recursive=False)
    def _parse_loss_type(self, loss_type: str):
        """
        auxiliary routine for parsing the loss function
        """

        loss_type = set(loss_type.split())

        relative = "relative" in loss_type
        squared = "squared" in loss_type

        jacobian = "s2" if "geometric" in loss_type else "flat"

        # decide which loss to use
        if "l2" in loss_type:
            loss_handle = partial(GeometricLpLoss, p=2, relative=relative, squared=squared, jacobian=jacobian)
        elif "l1" in loss_type:
            loss_handle = partial(GeometricLpLoss, p=1, relative=relative, squared=squared, jacobian=jacobian)
        elif "h1" in loss_type:
            assert jacobian == "s2"
            loss_handle = partial(SpectralH1Loss, relative=relative, squared=squared, jacobian=jacobian)
        elif "amse" in loss_type:
            loss_handle = SpectralAMSELoss
        elif "hydrostatic" in loss_type:
            use_moist_air_formula = "use_moist_air_formula" in loss_type
            # check if we have p_min,p_max specified
            p_min = 50
            p_max = 900
            for x in loss_type:
                if x.startswith("p_min="):
                    p_min = int(x.replace("p_min=", ""))
                elif x.startswith("p_max="):
                    p_max = int(x.replace("p_max=", ""))
            loss_handle = partial(HydrostaticBalanceLoss, p_min=p_min, p_max=p_max, use_moist_air_formula=use_moist_air_formula)
        elif "ensemble_crps" in loss_type:
            loss_handle = partial(EnsembleCRPSLoss, crps_type="cdf")
        elif "ensemble_spectral_crps" in loss_type:
            loss_handle = partial(EnsembleSpectralCRPSLoss, crps_type="cdf")
        elif "gauss_crps" in loss_type:
            loss_handle = partial(EnsembleCRPSLoss, crps_type="gauss")
        elif "ensemble_nll" in loss_type:
            loss_handle = EnsembleNLLLoss
        elif "ensemble_mmd" in loss_type:
            loss_handle = EnsembleMMDLoss
        elif "drift_regularization" in loss_type:
            loss_handle = DriftRegularization
        else:
            raise NotImplementedError(f"Unknown loss function: {loss_type}")

        return loss_handle

    @torch.compiler.disable(recursive=False)
    def _gather_batch(self, x: torch.Tensor) -> torch.Tensor:
        if comm.is_distributed("batch") and comm.get_size("batch") > 1:
            x = gather_from_parallel_region(x, 0, None, "batch")
        return x

    @torch.compiler.disable(recursive=False)
    def is_distributed(self):
        return False

    def _update_running_stats(self, x: torch.Tensor):
        """
        Uses Chan's parallel version of the Welford's algorithm [1]. For details see

        [1] Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J.; Updating Formulae and a Pairwise Algorithm for Computing Sample Variances. Technical Report STAN-CS-79-773
        [2] Algorithms for calculating variance; Wikipedia; https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """

        with torch.no_grad():
            num_batches = torch.ones_like(x[:, 0], dtype=torch.long)
            num_batches = self._gather_batch(num_batches).sum()
            x = self._gather_batch(x)

            # compute the variance and mean over the local batch dimension
            var, mean = torch.var_mean(x, dim=(0), correction=0, keepdim=False)

            m2 = var * num_batches

            # use Welford's algorithm to accumulate the batch mean and variance into the running
            delta = mean - self.running_mean
            self.running_var += m2 + delta**2 * self.num_batches_tracked * num_batches / (self.num_batches_tracked + num_batches)
            self.running_mean += delta * num_batches / (self.num_batches_tracked + num_batches)

            # update the current num_batches_tracked
            self.num_batches_tracked += num_batches

    def get_running_stats(self, correction: int = 0):
        if not self.track_running_stats:
            raise ValueError("Module does not track running stats")

        var = self.running_var / (self.num_batches_tracked - int(correction))
        mean = self.running_mean

        return var, mean

    def reset_running_stats(self):
        with torch.no_grad():
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None):
        # we assume the following:
        # if prd is 5D, we assume that the dims are
        # batch, ensemble, channel, h, w
        # otherwise we assume that the dims are
        # batch, channel, h, w

        # if random slices are enabled, we need to recombine both prediction and targets across the channel dimension
        if self.random_slice_loss:
            n_channels = prd.shape[-3]

            # generate random slice and normalize it
            rslice = torch.zeros(n_channels, n_channels, 1, 1, dtype=prd.dtype, device=prd.device)
            if rslice.is_cuda:
                rslice.normal_(0.0, 1.0, generator=self.rng_gpu)
            else:
                rslice.normal_(0.0, 1.0, generator=self.rng_cpu)
            rslice = rslice / torch.linalg.vector_norm(rslice, dim=1, keepdim=True)

            # compute randomly sliced predictions and targets
            if prd.dim() == 5:
                batch_size, ensemble_size = prd.shape[0:2]
                prd = prd.reshape(batch_size * ensemble_size, *prd.shape[2:])
                prd = nn.functional.conv2d(prd, rslice)
                prd = prd.reshape(batch_size, ensemble_size, *prd.shape[1:])
            else:
                prd = nn.functional.conv2d(prd, rslice)
            tar = nn.functional.conv2d(tar, rslice)

        # compute average over ensemble dim if requested:
        # TODO: change the behavior to instead compute the expected value of the deterministic losses
        if prd.dim() == 5:
            prdm = torch.mean(prd, dim=1)
            if self.ensemble_distributed:
                prdm = reduce_from_parallel_region(prdm, "ensemble") / float(comm.get_size("ensemble"))
        else:
            prdm = prd

        # compute loss contributions from each loss
        loss_vals = []
        for lfn in self.loss_fn:
            if lfn.type == LossType.Deterministic:
                loss_vals.append(lfn(prdm, tar, wgt))
            else:
                loss_vals.append(lfn(prd, tar, wgt))
        all_losses = torch.cat(loss_vals, dim=-1)

        if self.training and self.track_running_stats:
            self._update_running_stats(all_losses.clone())

        # process channel weights
        chw = self.channel_weights
        if self.uncertainty_weighting and self.training:
            var, _ = self.get_running_stats()
            chw = chw / (torch.sqrt(2 * var) + self.eps)

        if self.randomized_loss_weights:
            rmask = torch.zeros_like(chw)
            if rmask.is_cuda:
                rmask.uniform_(0.0, 1.0, generator=self.rng_gpu)
            else:
                rmask.uniform_(0.0, 1.0, generator=self.rng_cpu)

            rmask = rmask / rmask.sum()
            chw = chw * rmask

        # fold in multistep weight
        if self.training:
            if self.n_future > 0:
                chw = torch.tile(chw, (1, self.n_future + 1))
            chw = chw * self.multistep_weight

        # compute average over batch and weighted sum over channels
        loss = torch.mean(torch.sum(chw * all_losses, dim=1), dim=0)

        return loss
