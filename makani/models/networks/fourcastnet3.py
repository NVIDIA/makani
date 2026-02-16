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
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.amp as amp
from torch.utils.checkpoint import checkpoint

# helpers
from makani.models.common import DropPath, LayerScale, MLP, EncoderDecoder, SpectralConv, LearnablePositionEmbedding, ConstantImputation, MLPImputation
from makani.utils.features import get_water_channels, get_channel_groups
from makani.utils.grids import compute_spherical_bandlimit

# get spectral transforms and spherical convolutions from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd

# get pre-formulated layers
from makani.mpu.layers import DistributedMLP

# more distributed stuff
from makani.utils import comm

# for annotation of models
from dataclasses import dataclass
import physicsnemo
from physicsnemo.models.meta import ModelMetaData

# heuristic for finding theta_cutoff
def _compute_cutoff_radius(lmax, kernel_shape, basis_type):
    margin_factor = {"piecewise linear": 1.0, "morlet": 1.0, "harmonic": 1.0, "zernike": 1.0}
    return margin_factor[basis_type] * (kernel_shape[0] + 0.25) * math.pi / float(lmax)

# commenting out torch.compile due to long intiial compile times
# @torch.compile
def _soft_clamp(x: torch.Tensor, offset: float = 0.0):
    x = x + offset
    y = torch.where(x > 0.0, x**2, 0.0)
    y = torch.where(x >= 0.5, x - 0.25, y)
    return y

# heper function to be able to pass Sin as an activation function
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class DiscreteContinuousEncoder(nn.Module):
    def __init__(
        self,
        inp_shape=(721, 1440),
        out_shape=(480, 960),
        grid_in="equiangular",
        grid_out="equiangular",
        inp_chans=2,
        out_chans=2,
        kernel_shape=(3,3),
        basis_type="morlet",
        basis_norm_mode="nodal",
        lmax=240,
        use_mlp=False,
        mlp_ratio=2.0,
        activation_function=nn.GELU,
        groups=1,
        bias=False,
    ):
        super().__init__()

        # heuristic for finding theta_cutoff
        theta_cutoff = _compute_cutoff_radius(lmax=lmax, kernel_shape=kernel_shape, basis_type=basis_type)

        # set up local convolution
        conv_handle = thd.DistributedDiscreteContinuousConvS2 if comm.get_size("spatial") > 1 else th.DiscreteContinuousConvS2
        self.conv = conv_handle(
            inp_chans,
            out_chans,
            in_shape=inp_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            grid_in=grid_in,
            grid_out=grid_out,
            groups=groups,
            bias=bias,
            theta_cutoff=theta_cutoff,
        )
        if comm.get_size("spatial") > 1:
            self.conv.weight.is_shared_mp = ["spatial"]
            self.conv.weight.sharded_dims_mp = [None, None, None]
            if self.conv.bias is not None:
                self.conv.bias.is_shared_mp = ["spatial"]
                self.conv.bias.sharded_dims_mp = [None]

        if use_mlp:
            with torch.no_grad():
                self.conv.weight *= math.sqrt(2.0)

            self.act = activation_function()

            self.mlp = EncoderDecoder(
                num_layers=1,
                input_dim=out_chans,
                output_dim=out_chans,
                hidden_dim=int(mlp_ratio * out_chans),
                act_layer=activation_function,
                input_format="nchw",
            )

    def _conv_forward(self, x):
        dtype = x.dtype
        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.conv(x)
            x = x.to(dtype=dtype)
        return x

    def forward(self, x):
        x = self._conv_forward(x)

        if hasattr(self, "act"):
            x = self.act(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        return x


class DiscreteContinuousDecoder(nn.Module):
    def __init__(
        self,
        inp_shape=(480, 960),
        out_shape=(721, 1440),
        grid_in="equiangular",
        grid_out="equiangular",
        inp_chans=2,
        out_chans=2,
        kernel_shape=(3, 3),
        basis_type="morlet",
        basis_norm_mode="nodal",
        theta_cutoff_factor=1.0,
        lmax=240,
        use_mlp=False,
        mlp_ratio=2.0,
        activation_function=nn.GELU,
        groups=1,
        bias=False,
        upsample_sht=False,
    ):
        super().__init__()

        if use_mlp:
            self.mlp = EncoderDecoder(
                num_layers=1, input_dim=inp_chans, output_dim=inp_chans, hidden_dim=int(mlp_ratio * inp_chans), act_layer=activation_function, input_format="nchw", gain=2.0
            )

            self.act = activation_function()

        # init distributed torch-harmonics if needed
        if comm.get_size("spatial") > 1:
            polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
            azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
            thd.init(polar_group, azimuth_group)

        # spatial parallelism in the SHT
        if upsample_sht:
            # set up sht for upsampling
            sht_handle = thd.DistributedRealSHT if comm.get_size("spatial") > 1 else th.RealSHT
            isht_handle = thd.DistributedInverseRealSHT if comm.get_size("spatial") > 1 else th.InverseRealSHT

            # set upsampling module
            self.sht = sht_handle(*inp_shape, grid=grid_in).float()
            self.isht = isht_handle(*out_shape, lmax=self.sht.lmax, mmax=self.sht.mmax, grid=grid_out).float()
            self.upsample = nn.Sequential(self.sht, self.isht)
        else:
            resample_handle = thd.DistributedResampleS2 if comm.get_size("spatial") > 1 else th.ResampleS2

            self.upsample = resample_handle(*inp_shape, *out_shape, grid_in=grid_in, grid_out=grid_out, mode="bilinear")

        # heuristic for finding theta_cutoff
        # nto entirely clear if out or in shape should be used here with a non-conv method for upsampling
        theta_cutoff = theta_cutoff_factor * _compute_cutoff_radius(lmax=lmax, kernel_shape=kernel_shape, basis_type=basis_type)

        # set up DISCO convolution
        conv_handle = thd.DistributedDiscreteContinuousConvS2 if comm.get_size("spatial") > 1 else th.DiscreteContinuousConvS2
        self.conv = conv_handle(
            inp_chans,
            out_chans,
            in_shape=out_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            grid_in=grid_out,
            grid_out=grid_out,
            groups=groups,
            bias=bias,
            theta_cutoff=theta_cutoff,
        )
        if comm.get_size("spatial") > 1:
            self.conv.weight.is_shared_mp = ["spatial"]
            self.conv.weight.sharded_dims_mp = [None, None, None]
            if self.conv.bias is not None:
                self.conv.bias.is_shared_mp = ["spatial"]
                self.conv.bias.sharded_dims_mp = [None]

    def _conv_forward(self, x):
        dtype = x.dtype

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.upsample(x)
            x = self.conv(x)
            x = x.to(dtype=dtype)

        return x

    def forward(self, x):
        dtype = x.dtype

        if hasattr(self, "act"):
            x = self.act(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self._conv_forward(x)

        return x


class NeuralOperatorBlock(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        inp_chans,
        out_chans,
        conv_type="local",
        mlp_ratio=2.0,
        mlp_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.Identity,
        num_groups=1,
        skip="identity",
        layer_scale=True,
        use_mlp=False,
        kernel_shape=(3, 3),
        basis_type="morlet",
        basis_norm_mode="nodal",
        lmax=240,
        checkpointing_level=0,
        bias=False,
        stochastic_bias=False,
        seed=333,
    ):
        super().__init__()

        # generator objects:
        seed = seed + comm.get_rank("model") + comm.get_size("model") * comm.get_rank("ensemble") + comm.get_size("model") * comm.get_size("ensemble") * comm.get_rank("batch")
        self.set_rng(seed=seed)

        # determine some shapes
        self.inp_shape = (forward_transform.nlat, forward_transform.nlon)
        self.out_shape = (inverse_transform.nlat, inverse_transform.nlon)
        self.out_chans = out_chans

        # gain factor for the convolution
        gain_factor = 0.5

        # disco convolution layer
        if conv_type == "local":

            # heuristic for finding theta_cutoff
            theta_cutoff = _compute_cutoff_radius(lmax=lmax, kernel_shape=kernel_shape, basis_type=basis_type)

            conv_handle = thd.DistributedDiscreteContinuousConvS2 if comm.get_size("spatial") > 1 else th.DiscreteContinuousConvS2
            self.local_conv = conv_handle(
                inp_chans,
                inp_chans if use_mlp else out_chans,
                in_shape=self.inp_shape,
                out_shape=self.out_shape,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
                basis_norm_mode=basis_norm_mode,
                groups=num_groups,
                grid_in=forward_transform.grid,
                grid_out=inverse_transform.grid,
                bias=False,
                theta_cutoff=theta_cutoff,
            )
            if comm.get_size("spatial") > 1:
                self.local_conv.weight.is_shared_mp = ["spatial"]
                self.local_conv.weight.sharded_dims_mp = [None, None, None]
                if self.local_conv.bias is not None:
                    self.local_conv.bias.is_shared_mp = ["spatial"]
                    self.local_conv.bias.sharded_dims_mp = [None]

            with torch.no_grad():
                self.local_conv.weight *= gain_factor

        elif conv_type == "global":
            # convolution layer
            self.global_conv = SpectralConv(
                forward_transform,
                inverse_transform,
                inp_chans,
                inp_chans if use_mlp else out_chans,
                operator_type="dhconv",
                num_groups=num_groups,
                bias=False,
                gain=gain_factor,
            )
        else:
            raise ValueError(f"Unknown convolution type {conv_type}")

        # stochastic bias
        if stochastic_bias:
            self.bias_std = nn.Parameter(torch.zeros(inp_chans if use_mlp else out_chans, 1, 1))
            scale = math.sqrt(gain_factor / self.bias_std.shape[0] / 2)
            nn.init.normal_(self.bias_std, mean=0.0, std=scale)
            self.bias_std.is_shared_mp = ["spatial"]

        # norm layer
        self.norm = norm_layer()

        # MLP
        if use_mlp == True:
            MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
            mlp_hidden_dim = int(inp_chans * mlp_ratio)
            self.mlp = MLPH(
                in_features=inp_chans,
                out_features=out_chans,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=mlp_drop_rate,
                drop_type="features",
                checkpointing=(checkpointing_level >= 2),
                gain=gain_factor,
            )

        # dropout
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        if layer_scale:
            self.layer_scale = LayerScale(out_chans)
            self.layer_scale.weight.is_shared_mp = ["spatial"]
            self.layer_scale.weight.sharded_dims_mp = [None, None, None, None]
        else:
            self.layer_scale = nn.Identity()

        # skip connection
        if skip == "linear":
            gain_factor = 1.0
            self.skip = nn.Conv2d(inp_chans, out_chans, 1, 1, bias=False)
            torch.nn.init.normal_(self.skip.weight, std=math.sqrt(gain_factor / inp_chans))
            self.skip.weight.is_shared_mp = ["spatial"]
            self.skip.weight.sharded_dims_mp = [None, None, None, None]
            if self.skip.bias is not None:
                self.skip.bias.is_shared_mp = ["spatial"]
                self.skip.bias.sharded_dims_mp = [None]
        elif skip == "identity":
            self.skip = nn.Identity()
        elif skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {skip}")

    @torch.compiler.disable(recursive=False)
    def set_rng(self, seed=333):
        self.rng_cpu = torch.Generator(device=torch.device("cpu"))
        self.rng_cpu.manual_seed(seed)
        if torch.cuda.is_available():
            self.rng_gpu = torch.Generator(device=torch.device(f"cuda:{comm.get_local_rank()}"))
            self.rng_gpu.manual_seed(seed)

    def _conv_forward(self, x):
        if hasattr(self, "global_conv"):
            dx, _ = self.global_conv(x)
        elif hasattr(self, "local_conv"):
            dx = self.local_conv(x)

        return dx

    def forward(self, x):
        """
        Updated NO block
        """

        dx = self._conv_forward(x)

        if hasattr(self, "bias_std"):
            n = torch.zeros(*dx.shape[:-2], 1, 1, device=dx.device, dtype=dx.dtype)
            n.normal_(mean=0.0, std=1.0, generator=self.rng_gpu if n.is_cuda else self.rng_cpu)
            dx = dx + self.bias_std * n

        if hasattr(self, "norm"):
            dx = self.norm(dx)

        if hasattr(self, "mlp"):
            dx = self.mlp(dx)

        dx = self.drop_path(dx)

        if hasattr(self, "skip"):
            x = self.skip(x[..., : self.out_chans, :, :]) + self.layer_scale(dx)
        else:
            x = dx

        return x


class AtmoSphericNeuralOperatorNet(nn.Module):
    """
    Backbone of the FourCastNet2 architecture. Uses a Spherical Neural Operator which is derived from the
    Spherical Fourier Neural Operator and augmented with localized spherical Neural Operator Convolutions.
    Encoder and Decoder are grouped into channel groups to treat armospheric and surface variables appropriately.

    References:
    [1] Bonev et al., Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere
    [2] Ocampo et al., Scalable and Equivariant Spherical CNNs by Discrete-Continuous (DISCO) Convolutions
    [3] Liu-Schiaffini et al., Neural Operators with Localized Integral and Differential Kernels
    """

    def __init__(
        self,
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        inp_shape=(721, 1440),
        out_shape=(721, 1440),
        kernel_shape=(3, 3),
        filter_basis_type="morlet",
        filter_basis_norm_mode="mean",
        encoder_mlp=False,
        upsample_sht=False,
        channel_names=["u500", "v500"],
        aux_channel_names=[],
        n_history=0,
        embed_dim=8,
        aux_embed_dim=8,
        pos_embed_dim=0,
        num_layers=4,
        num_groups=1,
        use_mlp=True,
        mlp_ratio=2.0,
        activation_function="gelu",
        layer_scale=True,
        pos_drop_rate=0.0,
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        normalization_layer="none",
        hard_thresholding_fraction=0.25,
        scale_factor=8,
        lmax=None,
        sfno_block_frequency=2,
        big_skip=False,
        clamp_water=False,
        encoder_bias=False,
        bias=False,
        checkpointing_level=0,
        freeze_encoder=False,
        freeze_processor=False,
        normalization_means=None,
        normalization_stds=None,
        stochastic_bias=False,
        seed=333,
        **kwargs,
    ):
        super().__init__()

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.embed_dim = embed_dim
        self.aux_embed_dim = aux_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.big_skip = big_skip
        self.checkpointing_level = checkpointing_level
        self.n_history = n_history

        # compute the downscaled image size
        self.h = int(self.inp_shape[0] // scale_factor)
        self.w = int(self.inp_shape[1] // scale_factor)

        if normalization_means is not None:
            self.register_buffer("normalization_means", torch.as_tensor(normalization_means))
        if normalization_stds is not None:
            self.register_buffer("normalization_stds", torch.as_tensor(normalization_stds))

        # initialize spectral transforms
        self._init_spectral_transforms(model_grid_type, sht_grid_type, hard_thresholding_fraction, lmax)

        # compute static permutations to extract
        self._precompute_channel_groups(channel_names, aux_channel_names, n_history)

        # compute the total number of internal groups
        self.n_chans = self.n_atmo_groups * self.n_atmo_chans + self.n_surf_chans
        self.total_aux_embed_dim = self.aux_embed_dim + self.pos_embed_dim

        # convert kernel shape to tuple
        kernel_shape = tuple(kernel_shape)

        # determine activation function
        if activation_function == "relu":
            activation_function = nn.ReLU
        elif activation_function == "gelu":
            activation_function = nn.GELU
        elif activation_function == "silu":
            activation_function = nn.SiLU
        elif activation_function == "sin":
            activation_function = Sin
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # encoder for the atmospheric and surface channels
        self.encoder = DiscreteContinuousEncoder(
            inp_shape=inp_shape,
            out_shape=(self.h, self.w),
            inp_chans=self.n_chans * (self.n_history + 1),
            out_chans=self.embed_dim,
            grid_in=model_grid_type,
            grid_out=sht_grid_type,
            kernel_shape=kernel_shape,
            basis_type=filter_basis_type,
            basis_norm_mode=filter_basis_norm_mode,
            lmax=self.lmax,
            activation_function=activation_function,
            groups=math.gcd(self.n_chans, self.embed_dim),
            bias=encoder_bias,
            use_mlp=encoder_mlp,
        )

        if self.sst_channels_in.shape[0] > 0:
            self.sst_imputation = MLPImputation(
                inp_chans=self.n_chans * (self.n_history + 1),
                out_chans=self.sst_channels_in.shape[0],
                mlp_ratio=mlp_ratio,
                activation_function=activation_function,
            )

        # encoder for the auxiliary channels
        if self.n_aux_chans > 0:
            self.aux_encoder = DiscreteContinuousEncoder(
                inp_shape=inp_shape,
                out_shape=(self.h, self.w),
                inp_chans=self.n_aux_chans,
                out_chans=self.aux_embed_dim,
                grid_in=model_grid_type,
                grid_out=sht_grid_type,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                basis_norm_mode=filter_basis_norm_mode,
                lmax=self.lmax,
                activation_function=activation_function,
                groups=math.gcd(self.n_aux_chans, self.aux_embed_dim),
                bias=encoder_bias,
                use_mlp=encoder_mlp,
            )


        # decoder for the atmospheric and surface variables
        self.decoder = DiscreteContinuousDecoder(
            inp_shape=(self.h, self.w),
            out_shape=out_shape,
            inp_chans=self.embed_dim,
            out_chans=self.n_chans,
            grid_in=sht_grid_type,
            grid_out=model_grid_type,
            kernel_shape=kernel_shape,
            basis_type=filter_basis_type,
            basis_norm_mode=filter_basis_norm_mode,
            lmax=self.lmax,
            activation_function=activation_function,
            groups=math.gcd(self.n_chans, self.embed_dim),
            bias=encoder_bias,
            use_mlp=encoder_mlp,
            upsample_sht=upsample_sht,
        )

        # position embedding
        if self.pos_embed_dim > 0:
            self.pos_embed = LearnablePositionEmbedding(img_shape=(self.h, self.w), grid=sht_grid_type, num_chans=self.pos_embed_dim, embed_type="lat")

        # dropout
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]

        # get the handle for the normalization layer
        norm_layer = self._get_norm_layer_handle(self.h, self.w, self.embed_dim + self.total_aux_embed_dim, normalization_layer=normalization_layer, sht_grid_type=sht_grid_type)

        # Internal NO blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):

            # determine the convolution type
            if (sfno_block_frequency > 0) and (i % sfno_block_frequency == 0):
                conv_type = "global"
            else:
                conv_type = "local"

            block = NeuralOperatorBlock(
                self.sht,
                self.isht,
                self.embed_dim + self.total_aux_embed_dim,
                self.embed_dim,
                conv_type=conv_type,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                path_drop_rate=dpr[i],
                act_layer=activation_function,
                norm_layer=norm_layer,
                skip="identity",
                layer_scale=layer_scale,
                use_mlp=use_mlp,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                basis_norm_mode=filter_basis_norm_mode,
                lmax=self.lmax,
                checkpointing_level=checkpointing_level,
                bias=bias,
                stochastic_bias=stochastic_bias,
                seed=seed,
            )

            self.blocks.append(block)

        # controlled output normalization of q and tcwv
        if clamp_water:
            water_chans = get_water_channels(channel_names)
            if len(water_chans) > 0:
                self.register_buffer("water_channels", torch.LongTensor(water_chans), persistent=False)

        # freeze the encoder/decoder
        if freeze_encoder:
            frozen_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            if hasattr(self, "aux_encoder"):
                frozen_params += list(self.aux_encoder.parameters())
            if self.big_skip:
                frozen_params += list(self.residual_transform.parameters())
            for param in frozen_params:
                param.requires_grad = False

        # freeze the processor part
        if freeze_processor:
            frozen_params = self.blocks.parameters()
            for param in frozen_params:
                param.requires_grad = False


    @torch.compiler.disable(recursive=False)
    def _init_spectral_transforms(
        self,
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        hard_thresholding_fraction=1.0,
        lmax=None,
    ):
        """
        Initialize the spectral transforms based on the maximum number of modes to keep. Handles the computation
        of local image shapes and domain parallelism, based on the
        """

        # precompute the cutoff frequency on the sphere
        if lmax is None:
            lmax =  compute_spherical_bandlimit(self.inp_shape, model_grid_type)
            lmax = int(lmax * hard_thresholding_fraction)
        self.lmax = lmax

        # if sht_grid_type == "equiangular":
        #     self.h = self.lmax + 1
        #     self.w = 2 * self.lmax
        # elif sht_grid_type == "legendre-gauss":
        #     self.h = self.lmax + 1
        #     self.w = 2 * self.lmax
        # else:
        #     raise ValueError(f"Unknown SHT grid type {sht_grid_type}")

        sht_handle = th.RealSHT
        isht_handle = th.InverseRealSHT

        # spatial parallelism in the SHT
        if comm.get_size("spatial") > 1:
            polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
            azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
            thd.init(polar_group, azimuth_group)
            sht_handle = thd.DistributedRealSHT
            isht_handle = thd.DistributedInverseRealSHT

        # set up
        self.sht = sht_handle(self.h, self.w, lmax=self.lmax, mmax=self.lmax, grid=sht_grid_type).float()
        self.isht = isht_handle(self.h, self.w, lmax=self.lmax, mmax=self.lmax, grid=sht_grid_type).float()

    @torch.compiler.disable(recursive=True)
    def _get_norm_layer_handle(
        self,
        h,
        w,
        embed_dim,
        normalization_layer="none",
        sht_grid_type="legendre-gauss",
    ):
        """
        get the handle for ionitializing normalization layers
        """
        # pick norm layer
        if normalization_layer == "layer_norm":
            from makani.mpu.layer_norm import DistributedLayerNorm
            norm_layer_handle = partial(DistributedLayerNorm, normalized_shape=(embed_dim), elementwise_affine=True, eps=1e-6)
        elif normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                from makani.mpu.layer_norm import DistributedInstanceNorm2d
                norm_layer_handle = partial(DistributedInstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True)
            else:
                norm_layer_handle = partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif normalization_layer == "instance_norm_s2":
            if comm.get_size("spatial") > 1:
                from makani.mpu.layer_norm import DistributedGeometricInstanceNormS2
                norm_layer_handle = DistributedGeometricInstanceNormS2
            else:
                from makani.models.common import GeometricInstanceNormS2
                norm_layer_handle = GeometricInstanceNormS2
            norm_layer_handle = partial(
                norm_layer_handle,
                img_shape=(h, w),
                crop_shape=(h, w),
                crop_offset=(0, 0),
                grid_type=sht_grid_type,
                pole_mask=0,
                num_features=embed_dim,
                eps=1e-6,
                affine=True,
            )
        elif normalization_layer == "none":
            norm_layer_handle = nn.Identity
        else:
            raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.")

        return norm_layer_handle

    @torch.compiler.disable(recursive=True)
    def _precompute_channel_groups(
        self,
        channel_names=[],
        aux_channel_names=[],
        n_history=0,
    ):
        """
        group the channels appropriately into atmospheric pressure levels and surface variables
        """

        atmo_chans, surf_chans, dyn_aux_chans, stat_aux_chans, pressure_lvls = get_channel_groups(channel_names, aux_channel_names)
        sst_chans = [channel_names.index("sst")] if "sst" in channel_names else []
        lsml_chans = [len(channel_names) + aux_channel_names.index("xlsml")] if "xlsml" in aux_channel_names else []

        # compute how many channel groups will be kept internally
        self.n_atmo_groups = len(pressure_lvls)
        self.n_atmo_chans = len(atmo_chans) // self.n_atmo_groups
        self.n_surf_chans = len(surf_chans)
        self.n_dyn_aux_chans = len(dyn_aux_chans)
        self.n_stat_aux_chans= len(stat_aux_chans)
        self.n_aux_chans = self.n_dyn_aux_chans * (n_history + 1) + self.n_stat_aux_chans

        # make sure they are divisible. Attention! This does not guarantee that the grrouping is correct
        if len(atmo_chans) % self.n_atmo_groups:
            raise ValueError(f"Expected number of atmospheric variables to be divisible by number of atmospheric groups but got {len(atmo_chans)} and {self.n_atmo_groups}")

        # if history is included, adapt the channel lists to include the offsets
        self.n_atmo_groups = self.n_atmo_groups
        n_dyn_chans = len(atmo_chans) + len(surf_chans) + len(dyn_aux_chans)
        atmo_chans_in = atmo_chans.copy()
        surf_chans_in = surf_chans.copy()
        sst_chans_in = sst_chans.copy()
        for ih in range(1, n_history+1):
            atmo_chans_in += [(c + ih*n_dyn_chans) for c in atmo_chans]
            surf_chans_in += [(c + ih*n_dyn_chans) for c in surf_chans]
            sst_chans_in  += [(c + ih*n_dyn_chans) for c in sst_chans]
            dyn_aux_chans += [(c + ih*n_dyn_chans) for c in dyn_aux_chans]
        # account for the history offset in the static aux channels
        stat_aux_chans = [c + n_history*n_dyn_chans for c in stat_aux_chans]

        self.register_buffer("atmo_channels_in", torch.LongTensor(atmo_chans_in), persistent=False)
        self.register_buffer("atmo_channels_out", torch.LongTensor(atmo_chans), persistent=False)
        self.register_buffer("surf_channels_in", torch.LongTensor(surf_chans_in), persistent=False)
        self.register_buffer("surf_channels_out", torch.LongTensor(surf_chans), persistent=False)
        self.register_buffer("sst_channels_in", torch.LongTensor(sst_chans_in), persistent=False)
        self.register_buffer("sst_channels_out", torch.LongTensor(sst_chans), persistent=False)
        self.register_buffer("dyn_aux_channels", torch.LongTensor(dyn_aux_chans), persistent=False)
        self.register_buffer("stat_aux_channels", torch.LongTensor(stat_aux_chans), persistent=False)
        self.register_buffer("land_mask_channels", torch.LongTensor(lsml_chans), persistent=False)
        self.register_buffer("in_channels", torch.LongTensor(surf_chans_in + atmo_chans_in), persistent=False)
        self.register_buffer("aux_channels", torch.LongTensor(dyn_aux_chans + stat_aux_chans), persistent=False)
        self.register_buffer("pred_channels", torch.LongTensor(surf_chans + atmo_chans), persistent=False)

        return

    def encode(self, x):
        """
        forward pass for the encoder
        """
        batchdims = x.shape[:-3]

        x = x[..., self.in_channels, :, :].reshape(-1, 1 + self.n_history, self.n_chans, *x.shape[-2:])
        # move the history backwards and fold it into the channel dimension
        x = x.transpose(-3,-4).reshape(-1, self.n_chans * (1 + self.n_history), *x.shape[-2:]).contiguous()
        x = self.encoder(x)

        x = x.reshape(*batchdims, self.embed_dim, *x.shape[-2:])

        return x

    def encode_auxiliary_channels(self, x):
        """
        returns the embedded auxiliary channels
        """
        batchdims = x.shape[:-3]

        aux_tensors = []

        if hasattr(self, "aux_encoder"):
            x_aux = x[..., self.aux_channels, :, :].contiguous()
            x_aux = self.aux_encoder(x_aux)
            x_aux = x_aux.reshape(*batchdims, self.aux_embed_dim, *x_aux.shape[-2:])
            aux_tensors.append(x_aux)

        if hasattr(self, "pos_embed"):
            x_pos = self.pos_embed()
            aux_tensors.append(x_pos)

        if len(aux_tensors) > 0:
            x_aux = torch.cat(aux_tensors, dim=-3)
        else:
            x_aux = None

        return x_aux

    def decode(self, x):
        """
        forward pass for the decoder
        """

        batchdims = x.shape[:-3]

        x = x[..., : self.embed_dim, :, :]
        x = self.decoder(x)
        x_out = torch.zeros(*batchdims, self.n_chans, *x.shape[-2:], dtype=x.dtype, device=x.device)
        x_out[..., self.pred_channels, :, :] = x.reshape(*batchdims, -1, *x.shape[-2:])
        return x_out

    def processor_blocks(self, x, x_aux):
        # maybe clean the padding just in case
        x = self.pos_drop(x)

        # do the feature extraction
        for blk in self.blocks:

            # append the auxiliary channels to the input of each block
            if x_aux is not None:
                x = torch.cat([x, x_aux], dim=-3)

            if self.checkpointing_level >= 3:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        return x

    def clamp_water_channels(self, x):
        """clamp water channes with a smooth, positive activation function"""
        if hasattr(self, "water_channels"):
            if hasattr(self, "normalization_means") and hasattr(self, "normalization_stds"):
                means = self.normalization_means[self.water_channels].view(1, -1, 1, 1)
                stds = self.normalization_stds[self.water_channels].view(1, -1, 1, 1)
                offset = means / stds
                w = _soft_clamp(x[..., self.water_channels, :, :], offset=offset) - offset
            else:
                w = _soft_clamp(x[..., self.water_channels, :, :])
            # the following eventually leads to spectral instability
            # w = nn.functional.softplus(x[..., self.water_channels, :, :], beta=5, threshold=5)
            x[..., self.water_channels, :, :] = w

        return x

    def forward(self, x):

        # start by imputing the SST channels if applicable
        if hasattr(self, "sst_imputation"):
            if self.land_mask_channels.nelement() > 0:
                # get a land mask that is broadcastable to the input shape
                mask = x[..., self.land_mask_channels, :, :].unsqueeze(-3)
            else:
                mask = None
            x[..., self.sst_channels_in, :, :] = self.sst_imputation(x, mask=mask)

        # save big skip
        if self.big_skip:
            residual = x[..., self.pred_channels, :, :].contiguous()

        # extract embeddings for the auxiliary embeddings
        x_aux = self.encode_auxiliary_channels(x)

        # run the encoder
        if self.checkpointing_level >= 1:
            x = checkpoint(self.encode, x, use_reentrant=False)
        else:
            x = self.encode(x)

        # run the processor
        x = self.processor_blocks(x, x_aux)

        # run the decoder
        if self.checkpointing_level >= 1:
            x = checkpoint(self.decode, x, use_reentrant=False)
        else:
            x = self.decode(x)

        if self.big_skip:
            x[..., self.pred_channels, :, :] = x + residual

        # apply output transform
        x = self.clamp_water_channels(x)

        return x

# this part exposes the model to modulus by constructing modulus Modules
@dataclass
class AtmoSphericNeuralOperatorNetMetaData(ModelMetaData):
    name: str = "FCN3"

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True


FCN3 = physicsnemo.Module.from_torch(AtmoSphericNeuralOperatorNet, AtmoSphericNeuralOperatorNetMetaData())