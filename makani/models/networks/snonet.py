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
import torch
import torch.nn as nn
import torch.special as special
import torch.amp as amp
from torch.utils.checkpoint import checkpoint

from functools import partial
from itertools import groupby

# helpers
from makani.models.common import DropPath, LayerScale, MLP, EncoderDecoder, SpectralConv
from makani.utils.features import get_water_channels

# get spectral transforms and spherical convolutions from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd

# get pre-formulated layers
from makani.models.common import GeometricInstanceNormS2
from makani.mpu.layers import DistributedMLP, DistributedEncoderDecoder

# more distributed stuff
from makani.utils import comm
from physicsnemo.distributed.utils import split_tensor_along_dim

# layer normalization
from physicsnemo.distributed.mappings import scatter_to_parallel_region, gather_from_parallel_region
from makani.mpu.layer_norm import DistributedInstanceNorm2d, DistributedLayerNorm

# heuristic for finding theta_cutoff
def _compute_cutoff_radius(nlat, kernel_shape, basis_type):
    theta_cutoff_factor = {"piecewise linear": 0.5, "morlet": 0.5, "zernike": math.sqrt(2.0)}

    return (kernel_shape[0] + 1) * theta_cutoff_factor[basis_type] * math.pi / float(nlat - 1)


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
        basis_norm_mode="mean",
        use_mlp=False,
        mlp_ratio=2.0,
        activation_function=nn.GELU,
        groups=1,
        bias=False,
    ):
        super().__init__()

        # heuristic for finding theta_cutoff
        theta_cutoff = _compute_cutoff_radius(nlat=inp_shape[0], kernel_shape=kernel_shape, basis_type=basis_type)

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
                self.conv.bias.is_shared_mp = ["model"]
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

    def forward(self, x):
        dtype = x.dtype

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.conv(x)
            x = x.to(dtype=dtype)

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
        basis_norm_mode="mean",
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
        theta_cutoff = _compute_cutoff_radius(nlat=out_shape[0], kernel_shape=kernel_shape, basis_type=basis_type)

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
            bias=False,
            theta_cutoff=theta_cutoff,
        )
        if comm.get_size("spatial") > 1:
            self.conv.weight.is_shared_mp = ["spatial"]
            self.conv.weight.sharded_dims_mp = [None, None, None]
            if self.conv.bias is not None:
                self.conv.bias.is_shared_mp = ["model"]
                self.conv.bias.sharded_dims_mp = [None]

    def forward(self, x):
        dtype = x.dtype

        if hasattr(self, "act"):
            x = self.act(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.upsample(x)
            x = self.conv(x)
            x = x.to(dtype=dtype)

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
        basis_norm_mode="mean",
        checkpointing_level=0,
        bias=False,
    ):
        super().__init__()

        # determine some shapes
        self.input_shape = (forward_transform.nlat, forward_transform.nlon)
        self.output_shape = (inverse_transform.nlat, inverse_transform.nlon)
        self.out_chans = out_chans

        # gain factor for the convolution
        gain_factor = 1.0

        # disco convolution layer
        if conv_type == "local":

            conv_handle = thd.DistributedDiscreteContinuousConvS2 if comm.get_size("spatial") > 1 else th.DiscreteContinuousConvS2
            self.local_conv = conv_handle(
                inp_chans,
                inp_chans,
                in_shape=self.input_shape,
                out_shape=self.output_shape,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
                basis_norm_mode=basis_norm_mode,
                groups=num_groups,
                grid_in=forward_transform.grid,
                grid_out=inverse_transform.grid,
                bias=False,
                theta_cutoff=math.sqrt(2) * torch.pi / float(self.input_shape[0] - 1),
            )
            if comm.get_size("spatial") > 1:
                self.local_conv.weight.is_shared_mp = ["spatial"]
                self.local_conv.weight.sharded_dims_mp = [None, None, None]
                if self.local_conv.bias is not None:
                    self.local_conv.bias.is_shared_mp = ["model"]
                    self.local_conv.bias.sharded_dims_mp = [None]

            with torch.no_grad():
                self.local_conv.weight *= gain_factor

        elif conv_type == "global":
            # convolution layer
            self.global_conv = SpectralConv(
                forward_transform,
                inverse_transform,
                inp_chans,
                inp_chans,
                operator_type="dhconv",
                num_groups=num_groups,
                bias=bias,
                gain=gain_factor,
            )
        else:
            raise ValueError(f"Unknown convolution type {conv_type}")

        # norm layer
        self.norm = norm_layer()

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
                checkpointing=(checkpointing_level>=2),
                gain=gain_factor,
            )

        # dropout
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        if layer_scale:
            self.layer_scale = LayerScale(out_chans)
            if comm.get_size("spatial") > 1:
                self.layer_scale.weight.is_shared_mp = ["model"]
                self.layer_scale.weight.sharded_dims_mp = [None, None, None, None]
        else:
            self.layer_scale = nn.Identity()

        # skip connection
        if skip == "linear":
            gain_factor = 1.0
            self.skip = nn.Conv2d(inp_chans, out_chans, 1, 1, bias=False)
            torch.nn.init.normal_(self.skip.weight, std=math.sqrt(gain_factor / inp_chans))
            if comm.get_size("spatial") > 1:
                self.skip.weight.is_shared_mp = ["model"]
                self.skip.weight.sharded_dims_mp = [None, None, None, None]
                if self.skip.bias is not None:
                    self.skip.bias.is_shared_mp = ["model"]
                    self.skip.bias.sharded_dims_mp = [None]

        elif skip == "identity":
            self.skip = nn.Identity()
        elif skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {skip}")

    def forward(self, x):
        """
        Updated NO block
        """

        if hasattr(self, "global_conv"):
            dx, _ = self.global_conv(x)
        elif hasattr(self, "local_conv"):
            dx = self.local_conv(x)

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


class SphericalNeuralOperatorNet(nn.Module):
    """
    Backbone of the FourCastNet2 architecture. Uses a variant of the Spherical Fourier Neural Operator augmented with localized
    spherical Neural Operator Convolutions in encoder and decoder as well as processor layers.

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
        scale_factor=8,
        encoder_kernel_shape=(3, 3),
        encoder_mlp=False,
        encoder_groups=1,
        channel_names=["z500", "t850"],
        inp_chans=2,
        out_chans=2,
        embed_dim=32,
        num_layers=4,
        num_groups=1,
        use_mlp=True,
        mlp_ratio=2.0,
        activation_function="gelu",
        layer_scale=True,
        pos_drop_rate=0.0,
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        normalization_layer="instance_norm",
        max_modes=None,
        hard_thresholding_fraction=1.0,
        sfno_block_frequency=2,
        big_skip=True,
        clamp_water=False,
        bias=False,
        checkpointing_level=0,
        freeze_encoder=False,
        freeze_processor=False,
        **kwargs,
    ):
        super().__init__()

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.big_skip = big_skip
        self.checkpointing_level = checkpointing_level

        # compute the downscaled image size
        self.h = int(self.inp_shape[0] // scale_factor)
        self.w = int(self.inp_shape[1] // scale_factor)

        # initialize spectral transforms
        self._init_spectral_transforms(model_grid_type, sht_grid_type, hard_thresholding_fraction, max_modes)

        # determine activation function
        if activation_function == "relu":
            activation_function = nn.ReLU
        elif activation_function == "gelu":
            activation_function = nn.GELU
        elif activation_function == "silu":
            activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # convert kernel shape to tuple
        kernel_shape = tuple(kernel_shape)
        encoder_kernel_shape = tuple(encoder_kernel_shape)

        # set up encoder
        self.encoder = DiscreteContinuousEncoder(
            inp_shape=inp_shape,
            out_shape=(self.h, self.w),
            inp_chans=inp_chans,
            out_chans=embed_dim,
            grid_in=model_grid_type,
            grid_out=sht_grid_type,
            kernel_shape=encoder_kernel_shape,
            basis_type=filter_basis_type,
            basis_norm_mode=filter_basis_norm_mode,
            bias=bias,
            groups=encoder_groups,
            use_mlp=encoder_mlp,
        )

        # dropout
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]

        # get the handle for the normalization layer
        norm_layer = self._get_norm_layer_handle(self.h, self.w, embed_dim, normalization_layer=normalization_layer, sht_grid_type=sht_grid_type)

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            first_layer = i == 0
            last_layer = i == num_layers - 1

            skip = "identity"
            if i % sfno_block_frequency == 0:
                conv_type = "global"
            else:
                conv_type = "local"

            block = NeuralOperatorBlock(
                self.sht,
                self.isht,
                embed_dim,
                embed_dim,
                conv_type=conv_type,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                path_drop_rate=dpr[i],
                act_layer=activation_function,
                norm_layer=norm_layer,
                skip=skip,
                layer_scale=layer_scale,
                use_mlp=use_mlp,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                basis_norm_mode=filter_basis_norm_mode,
                bias=bias,
                checkpointing_level=checkpointing_level,
            )

            self.blocks.append(block)

        # set up decoder
        self.decoder = DiscreteContinuousDecoder(
            inp_shape=(self.h, self.w),
            out_shape=out_shape,
            inp_chans=embed_dim,
            out_chans=out_chans,
            grid_in=sht_grid_type,
            grid_out=model_grid_type,
            kernel_shape=encoder_kernel_shape,
            basis_type=filter_basis_type,
            basis_norm_mode=filter_basis_norm_mode,
            bias=bias,
            groups=encoder_groups,
            use_mlp=encoder_mlp,
        )

        # residual prediction
        if self.big_skip:
            self.residual_transform = nn.Conv2d(self.out_chans, self.out_chans, 1, bias=False)
            self.residual_transform.weight.is_shared_mp = ["spatial"]
            self.residual_transform.weight.sharded_dims_mp = [None, None, None, None]
            scale = math.sqrt(0.5 / self.out_chans)
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

        # controlled output normalization of q and tcwv
        if clamp_water:
            water_chans = get_water_channels(channel_names)
            if len(water_chans) > 0:
                self.register_buffer("water_channels", torch.LongTensor(water_chans), persistent=False)

        # finally, freeze part of the model if requested
        if freeze_processor:
            for param in self.blocks.parameters():
                param.requires_grad = False

        # freeze the encoder/decoder
        if freeze_encoder:
            frozen_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
            if self.big_skip:
                frozen_params += list(self.residual_transform.parameters())
            for param in frozen_params:
                param.requires_grad = False

    @torch.compiler.disable(recursive=False)
    def _init_spectral_transforms(
        self,
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        hard_thresholding_fraction=1.0,
        max_modes=None,
    ):
        """
        Initialize the spectral transforms based on the maximum number of modes to keep. Handles the computation
        of local image shapes and domain parallelism, based on the
        """

        # precompute the cutoff frequency on the sphere
        if max_modes is not None:
            modes_lat, modes_lon = max_modes
        else:
            modes_lat = int(self.h * hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * hard_thresholding_fraction)

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
        self.sht = sht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type).float()
        self.isht = isht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type).float()

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
            norm_layer_handle = partial(DistributedLayerNorm, normalized_shape=(embed_dim), elementwise_affine=True, eps=1e-6)
        elif normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                norm_layer_handle = partial(DistributedInstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True)
            else:
                norm_layer_handle = partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif normalization_layer == "instance_norm_s2":
            norm_layer_handle = DistributedGeometricInstanceNormS2 if comm.get_size("spatial") > 1 else GeometricInstanceNormS2
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

        def _get_slices(lst):
            for a, b in groupby(enumerate(lst), lambda pair: pair[1] - pair[0]):
                b = list(b)
                yield slice(b[0][1], b[-1][1] + 1)

        self.water_chans = list(_get_slices(water_chans))

    def _forward_features(self, x):
        for blk in self.blocks:
            if self.checkpointing_level >= 3:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        return x

    def clamp_water_channels(self, x):

        if hasattr(self, "water_channels"):
            w = nn.functional.relu(x[..., self.water_channels, :, :])
            x[..., self.water_channels, :, :] = w

        return x

    def forward(self, x):
        dtype = x.dtype

        # save big skip
        if self.big_skip:
            residual = x[..., : self.out_chans, :, :].contiguous()

        if self.checkpointing_level >= 1:
            x = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)

        # maybe clean the padding just in case
        x = self.pos_drop(x)

        # do the feature extraction
        x = self._forward_features(x)

        if self.checkpointing_level >= 1:
            x = checkpoint(self.decoder, x, use_reentrant=False)
        else:
            x = self.decoder(x)

        if self.big_skip:
            x = x + self.residual_transform(residual)

        # apply output transform
        x = self.clamp_water_channels(x)

        return x
