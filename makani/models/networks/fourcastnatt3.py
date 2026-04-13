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
import torch.amp as amp
from torch.utils.checkpoint import checkpoint

from functools import partial

# helpers
from makani.models.common import DropPath, LayerScale, MLP, EncoderDecoder, SpectralConv
from makani.utils.features import get_water_channels, get_channel_groups

# get spectral transforms and spherical convolutions from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd

# get pre-formulated layers
#from makani.models.common import GeometricInstanceNormS2
from makani.mpu.layers import DistributedMLP

# more distributed stuff
from makani.utils import comm

# for annotation of models
from dataclasses import dataclass
import physicsnemo
from physicsnemo import ModelMetaData

# heuristic for finding theta_cutoff
def _compute_cutoff_radius(nlat, kernel_shape, basis_type):
    theta_cutoff_factor = {"piecewise linear": 0.5, "morlet": 0.5, "zernike": math.sqrt(2.0)}

    return (kernel_shape[0] + 1) * theta_cutoff_factor[basis_type] * math.pi / float(nlat - 1)

# commenting out torch.compile due to long intiial compile times
# @torch.compile
def _soft_clamp(x: torch.Tensor, offset: float = 0.0):
    dtype = x.dtype
    x = x + offset
    y = torch.where(x > 0.0, x**2, 0.0)
    y = torch.where(x >= 0.5, x - 0.25, y)
    y = y.to(dtype=dtype)
    return y


@torch.compiler.disable(recursive=True)
def _get_norm_layer_handle(
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
        inverse_transform=None,
        use_mlp=False,
        mlp_ratio=2.0,
        activation_function=nn.GELU,
        bias=False,
    ):
        super().__init__()

        # heuristic for finding theta_cutoff
        theta_cutoff = _compute_cutoff_radius(nlat=inp_shape[0], kernel_shape=kernel_shape, basis_type=basis_type)

        # set up local convolution
        attn_handle = thd.DistributedNeighborhoodAttentionS2 if comm.get_size("spatial") > 1 else th.NeighborhoodAttentionS2
        self.attn = attn_handle(
            in_channels=inp_chans,
            out_channels=out_chans,
            in_shape=inp_shape,
            out_shape=out_shape,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=bias,
            theta_cutoff=theta_cutoff,
        )
        if comm.get_size("spatial") > 1:
            self.attn.q_weights.is_shared_mp = ["spatial"]
            self.attn.q_weights.sharded_dims_mp = [None, None, None, None]
            self.attn.k_weights.is_shared_mp = ["spatial"]
            self.attn.k_weights.sharded_dims_mp = [None, None, None, None]
            self.attn.v_weights.is_shared_mp = ["spatial"]
            self.attn.v_weights.sharded_dims_mp = [None, None, None, None]
            self.attn.proj_weights.is_shared_mp = ["spatial"]
            self.attn.proj_weights.sharded_dims_mp = [None, None, None, None]
            if self.attn.q_bias is not None:
                self.attn.q_bias.is_shared_mp = ["spatial"]
                self.attn.q_bias.sharded_dims_mp = [None]
            if self.attn.k_bias is not None:
                self.attn.k_bias.is_shared_mp = ["spatial"]
                self.attn.k_bias.sharded_dims_mp = [None]
            if self.attn.v_bias is not None:
                self.attn.v_bias.is_shared_mp = ["spatial"]
                self.attn.v_bias.sharded_dims_mp = [None]
            if self.attn.proj_bias is not None:
                self.attn.proj_bias.is_shared_mp = ["spatial"]

        # learnable query in spherical harmonic space: grid-agnostic Perceiver-style latent.
        # iSHT maps the spectral coefficients to the encoder output grid, so the query is
        # band-limited and independent of grid type or resolution.
        # InverseRealSHT has no learnable parameters (precomputed buffers only), so storing
        # the reference here does not double-count parameters from the outer model.
        if inverse_transform is not None:
            self.isht = inverse_transform

            # local spectral mode counts — mirrors the SpectralConv pattern
            if isinstance(self.isht, thd.DistributedInverseRealSHT):
                modes_lat_local = self.isht.l_shapes[comm.get_rank("h")]
                modes_lon_local = self.isht.m_shapes[comm.get_rank("w")]
            else:
                modes_lat_local = self.isht.lmax
                modes_lon_local = self.isht.mmax

            # zero-init: isht(0)=0 → uniform attention → average-pool starting point
            self.latent_query_spec = nn.Parameter(
                torch.zeros(1, inp_chans, modes_lat_local, modes_lon_local, dtype=torch.complex64)
            )
            # sharding annotation matches SpectralConv diagonal weights:
            # lat modes sharded in "h", lon modes sharded in "w"
            self.latent_query_spec.is_shared_mp = ["matmul"]
            self.latent_query_spec.sharded_dims_mp = [None, None, "h", "w"]
        else:
            # fallback: bilinear downsample as query
            resample_handle = thd.DistributedResampleS2 if comm.get_size("spatial") > 1 else th.ResampleS2
            self.downsample = resample_handle(*inp_shape, *out_shape, grid_in=grid_in, grid_out=grid_out, mode="bilinear")

        if use_mlp:
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
        with amp.autocast(device_type=x.device.type, enabled=False):
            if hasattr(self, "latent_query_spec"):
                # spectral latent: expand over batch then project to spatial grid
                query = self.latent_query_spec.to(torch.complex64).expand(x.shape[0], -1, -1, -1).contiguous()
                query = self.isht(query)
            else:
                query = x.to(torch.float32)
                query = self.downsample(query)
        query = query.to(dtype=x.dtype)

        # cross-attention: learned spectral queries attend to full-resolution input
        xd = self.attn(query=query, key=x, value=x)

        if hasattr(self, "act"):
            xd = self.act(xd)

        if hasattr(self, "mlp"):
            xd = self.mlp(xd)

        return xd


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
        inverse_transform=None,
        use_mlp=False,
        mlp_ratio=2.0,
        activation_function=nn.GELU,
        bias=False,
        upsample_sht=False,
    ):
        super().__init__()

        # init distributed torch-harmonics if needed
        if comm.get_size("spatial") > 1:
            polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
            azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
            thd.init(polar_group, azimuth_group)

        attn_handle = thd.DistributedNeighborhoodAttentionS2 if comm.get_size("spatial") > 1 else th.NeighborhoodAttentionS2

        if inverse_transform is not None:
            # Perceiver-style: learnable spectral query projected to the output grid,
            # cross-attending to the low-res latent as keys/values. Symmetric to the encoder.
            self.isht_out = inverse_transform

            # local spectral mode counts — mirrors the encoder's latent_query_spec pattern
            if isinstance(self.isht_out, thd.DistributedInverseRealSHT):
                modes_lat_local = self.isht_out.l_shapes[comm.get_rank("h")]
                modes_lon_local = self.isht_out.m_shapes[comm.get_rank("w")]
            else:
                modes_lat_local = self.isht_out.lmax
                modes_lon_local = self.isht_out.mmax

            # learnable query in spherical harmonic space: grid-agnostic Perceiver-style latent.
            # iSHT maps spectral coefficients to the decoder output grid, so the query is
            # band-limited and independent of the specific output grid type or resolution.
            # zero-init: isht(0)=0 → uniform attention → average-pool starting point
            self.latent_query_spec = nn.Parameter(
                torch.zeros(1, inp_chans, modes_lat_local, modes_lon_local, dtype=torch.complex64)
            )
            # sharding annotation: lat modes sharded in "h", lon modes sharded in "w"
            self.latent_query_spec.is_shared_mp = ["matmul"]
            self.latent_query_spec.sharded_dims_mp = [None, None, "h", "w"]

            # theta_cutoff based on the latent (kv) grid resolution
            theta_cutoff = _compute_cutoff_radius(nlat=inp_shape[0], kernel_shape=kernel_shape, basis_type=basis_type)

            # cross-attention: high-res output queries attend to low-res latent keys/values
            self.attn = attn_handle(
                in_channels=inp_chans,
                out_channels=out_chans,
                in_shape=inp_shape,
                out_shape=out_shape,
                grid_in=grid_in,
                grid_out=grid_out,
                bias=bias,
                theta_cutoff=theta_cutoff,
            )
        else:
            # classic path: optional pre-mlp on the latent, explicit spatial upsample, self-attention
            if use_mlp:
                self.mlp = EncoderDecoder(
                    num_layers=1, input_dim=inp_chans, output_dim=inp_chans, hidden_dim=int(mlp_ratio * inp_chans), act_layer=activation_function, input_format="nchw", gain=2.0
                )
                self.act = activation_function()

            if upsample_sht:
                sht_handle = thd.DistributedRealSHT if comm.get_size("spatial") > 1 else th.RealSHT
                isht_handle = thd.DistributedInverseRealSHT if comm.get_size("spatial") > 1 else th.InverseRealSHT
                self.sht = sht_handle(*inp_shape, grid=grid_in).float()
                self.isht = isht_handle(*out_shape, lmax=self.sht.lmax, mmax=self.sht.mmax, grid=grid_out).float()
                self.upsample = nn.Sequential(self.sht, self.isht)
            else:
                resample_handle = thd.DistributedResampleS2 if comm.get_size("spatial") > 1 else th.ResampleS2
                self.upsample = resample_handle(*inp_shape, *out_shape, grid_in=grid_in, grid_out=grid_out, mode="bilinear")

            theta_cutoff = _compute_cutoff_radius(nlat=out_shape[0], kernel_shape=kernel_shape, basis_type=basis_type)

            self.attn = attn_handle(
                in_channels=inp_chans,
                out_channels=out_chans,
                in_shape=out_shape,
                out_shape=out_shape,
                grid_in=grid_out,
                grid_out=grid_out,
                bias=bias,
                theta_cutoff=theta_cutoff,
            )

        if comm.get_size("spatial") > 1:
            self.attn.q_weights.is_shared_mp = ["spatial"]
            self.attn.q_weights.sharded_dims_mp = [None, None, None, None]
            self.attn.k_weights.is_shared_mp = ["spatial"]
            self.attn.k_weights.sharded_dims_mp = [None, None, None, None]
            self.attn.v_weights.is_shared_mp = ["spatial"]
            self.attn.v_weights.sharded_dims_mp = [None, None, None, None]
            self.attn.proj_weights.is_shared_mp = ["spatial"]
            self.attn.proj_weights.sharded_dims_mp = [None, None, None, None]
            if bias:
                self.attn.q_bias.is_shared_mp = ["spatial"]
                self.attn.q_bias.sharded_dims_mp = [None]
                self.attn.k_bias.is_shared_mp = ["spatial"]
                self.attn.k_bias.sharded_dims_mp = [None]
                self.attn.v_bias.is_shared_mp = ["spatial"]
                self.attn.v_bias.sharded_dims_mp = [None]
                self.attn.proj_bias.is_shared_mp = ["spatial"]
                self.attn.proj_bias.sharded_dims_mp = [None]
                self.attn.bias.sharded_dims_mp = [None]

    def forward(self, x):
        if hasattr(self, "latent_query_spec"):
            # Perceiver-style: project spectral query to output grid, cross-attend to latent
            with amp.autocast(device_type=x.device.type, enabled=False):
                query = self.latent_query_spec.to(torch.complex64).expand(x.shape[0], -1, -1, -1).contiguous()
                query = self.isht_out(query)
            query = query.to(dtype=x.dtype)
            x = self.attn(query=query, key=x, value=x)
        else:
            # classic path: optional pre-mlp, upsample, self-attention
            dtype = x.dtype
            if hasattr(self, "act"):
                x = self.act(x)
            if hasattr(self, "mlp"):
                x = self.mlp(x)
            with amp.autocast(device_type=x.device.type, enabled=False):
                x = x.to(torch.float32)
                x = self.upsample(x)
            x = x.to(dtype=dtype)
            x = self.attn(x)
        return x


class NeuralOperatorBlock(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        inp_chans,
        out_chans,
        attn_type="local",
        mlp_ratio=2.0,
        mlp_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        normalization_layer="layer_norm",
        num_groups=1,
        skip="identity",
        layer_scale=True,
        use_mlp=False,
        kernel_shape=(3, 3),
        basis_type="morlet",
        checkpointing_level=0,
        bias=False,
    ):
        super().__init__()

        # determine some shapes
        self.inp_shape = (forward_transform.nlat, forward_transform.nlon)
        self.out_shape = (inverse_transform.nlat, inverse_transform.nlon)
        self.out_chans = out_chans

        # gain factor for the convolution
        gain_factor = 1.0

        # disco convolution layer
        if attn_type == "local":
            # heuristic for finding theta_cutoff
            theta_cutoff = 2 * _compute_cutoff_radius(nlat=self.inp_shape[0], kernel_shape=kernel_shape, basis_type=basis_type)

            attn_handle = thd.DistributedNeighborhoodAttentionS2 if comm.get_size("spatial") > 1 else th.NeighborhoodAttentionS2
            self.attn = attn_handle(
                in_channels=inp_chans,
                out_channels=inp_chans,
                in_shape=self.inp_shape,
                out_shape=self.out_shape,
                grid_in=forward_transform.grid,
                grid_out=inverse_transform.grid,
                bias=bias,
                theta_cutoff=theta_cutoff,
            )
            if comm.get_size("spatial") > 1:
                self.attn.q_weights.is_shared_mp = ["spatial"]
                self.attn.q_weights.sharded_dims_mp = [None, None, None, None]
                self.attn.k_weights.is_shared_mp = ["spatial"]
                self.attn.k_weights.sharded_dims_mp = [None, None, None, None]
                self.attn.v_weights.is_shared_mp = ["spatial"]
                self.attn.v_weights.sharded_dims_mp = [None, None, None, None]
                self.attn.proj_weights.is_shared_mp = ["spatial"]
                self.attn.proj_weights.sharded_dims_mp = [None, None, None, None]
                if bias:
                    self.attn.q_bias.is_shared_mp = ["spatial"]
                    self.attn.q_bias.sharded_dims_mp = [None]
                    self.attn.k_bias.is_shared_mp = ["spatial"]
                    self.attn.k_bias.sharded_dims_mp = [None]
                    self.attn.v_bias.is_shared_mp = ["spatial"]
                    self.attn.v_bias.sharded_dims_mp = [None]
                    self.attn.proj_bias.is_shared_mp = ["spatial"]
                    self.attn.proj_bias.sharded_dims_mp = [None]

        elif attn_type == "global":
            # global spectral convolution — same as FCN3's global blocks
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
            raise ValueError(f"Unknown attention type {attn_type}")

        # get normalization layer handles and instances
        norm_layer_handle = _get_norm_layer_handle(
            self.inp_shape[0],
            self.inp_shape[1],
            inp_chans,
            normalization_layer=normalization_layer,
            sht_grid_type=forward_transform.grid,
        )
        self.norm1 = norm_layer_handle()
        self.norm2 = norm_layer_handle()

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

        # skip projection only needed when dimensions change (e.g. first block with aux channels)
        if inp_chans != out_chans:
            self.skip_projection = nn.Conv2d(inp_chans, out_chans, 1, bias=bias)
            self.skip_projection.weight.is_shared_mp = ["spatial"]
            self.skip_projection.weight.sharded_dims_mp = [None, None, None, None]
            if bias:
                self.skip_projection.bias.is_shared_mp = ["spatial"]
                self.skip_projection.bias.sharded_dims_mp = [None]

        # layer scale for stable init: residual branches start near-zero
        if layer_scale:
            self.layer_scale1 = LayerScale(inp_chans)
            self.layer_scale1.weight.is_shared_mp = ["spatial"]
            self.layer_scale1.weight.sharded_dims_mp = [None, None, None, None]
            self.layer_scale2 = LayerScale(out_chans)
            self.layer_scale2.weight.is_shared_mp = ["spatial"]
            self.layer_scale2.weight.sharded_dims_mp = [None, None, None, None]

        # dropout
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Updated NO block
        """
        attn_scale = self.layer_scale1 if hasattr(self, "layer_scale1") else nn.Identity()
        mlp_scale  = self.layer_scale2 if hasattr(self, "layer_scale2") else nn.Identity()

        # mixing sub-layer: Pre-LN residual, local attention or global spectral conv
        x_norm = self.norm1(x)
        if hasattr(self, "global_conv"):
            mix_out, _ = self.global_conv(x_norm)
        else:
            mix_out = self.attn(x_norm)
        x = x + self.drop_path(attn_scale(mix_out))

        # mlp sub-layer: use skip_projection only when dimensions change
        if hasattr(self, "skip_projection"):
            x = self.skip_projection(x) + self.drop_path(mlp_scale(self.mlp(self.norm2(x))))
        else:
            x = x + self.drop_path(mlp_scale(self.mlp(self.norm2(x))))

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
        scale_factor=8,
        encoder_mlp=False,
        upsample_sht=False,
        channel_names=["u500", "v500"],
        aux_channel_names=[],
        n_history=0,
        atmo_embed_dim=8,
        surf_embed_dim=8,
        aux_embed_dim=8,
        num_layers=4,
        use_mlp=True,
        mlp_ratio=2.0,
        activation_function="gelu",
        layer_scale=True,
        pos_drop_rate=0.0,
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        normalization_layer="none",
        max_modes=None,
        hard_thresholding_fraction=1.0,
        sfno_block_frequency=2,
        big_skip=False,
        clamp_water=False,
        bias=False,
        checkpointing_level=0,
        freeze_encoder=False,
        freeze_processor=False,
        perceiver_decoder=True,
        **kwargs,
    ):
        super().__init__()

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.atmo_embed_dim = atmo_embed_dim
        self.surf_embed_dim = surf_embed_dim
        self.aux_embed_dim = aux_embed_dim
        self.big_skip = big_skip
        self.checkpointing_level = checkpointing_level

        # currently doesn't support neither history nor future:
        assert n_history == 0

        # compute the downscaled image size
        self.h = int(self.inp_shape[0] // scale_factor)
        self.w = int(self.inp_shape[1] // scale_factor)

        # initialize spectral transforms
        self._init_spectral_transforms(model_grid_type, sht_grid_type, hard_thresholding_fraction, max_modes)

        # iSHT that maps the latent spectral modes to the output grid resolution.
        # Used by the Perceiver-style decoder to produce band-limited output queries.
        if perceiver_decoder:
            isht_out_handle = thd.DistributedInverseRealSHT if comm.get_size("spatial") > 1 else th.InverseRealSHT
            self.isht_out = isht_out_handle(
                out_shape[0], out_shape[1],
                lmax=self.modes_lat, mmax=self.modes_lon,
                grid=model_grid_type,
            ).float()
            decoder_inv_transform = self.isht_out
        else:
            decoder_inv_transform = None

        # compute static permutations to extract
        self._precompute_channel_groups(channel_names, aux_channel_names)

        # compute the total number of internal groups
        self.n_out_chans = self.n_atmo_groups * self.n_atmo_chans + self.n_surf_chans
        self.total_embed_dim = self.n_atmo_groups * self.atmo_embed_dim + self.surf_embed_dim

        # convert kernel shape to tuple
        kernel_shape = tuple(kernel_shape)

        # determine activation function
        if activation_function == "relu":
            activation_function = nn.ReLU
        elif activation_function == "gelu":
            activation_function = nn.GELU
        elif activation_function == "silu":
            activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # encoder for the atmospheric channels
        # TODO: add the groups
        self.atmo_encoder = DiscreteContinuousEncoder(
            inp_shape=inp_shape,
            out_shape=(self.h, self.w),
            inp_chans=self.n_atmo_chans,
            out_chans=self.atmo_embed_dim,
            grid_in=model_grid_type,
            grid_out=sht_grid_type,
            kernel_shape=kernel_shape,
            basis_type=filter_basis_type,
            inverse_transform=self.isht,
            activation_function=activation_function,
            bias=bias,
            use_mlp=encoder_mlp,
        )

        # encoder for the surface channels
        if self.n_surf_chans > 0:
            self.surf_encoder = DiscreteContinuousEncoder(
                inp_shape=inp_shape,
                out_shape=(self.h, self.w),
                inp_chans=self.n_surf_chans,
                out_chans=self.surf_embed_dim,
                grid_in=model_grid_type,
                grid_out=sht_grid_type,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                inverse_transform=self.isht,
                activation_function=activation_function,
                bias=bias,
                use_mlp=encoder_mlp,
            )

        # decoder for the atmospheric variables
        self.atmo_decoder = DiscreteContinuousDecoder(
            inp_shape=(self.h, self.w),
            out_shape=out_shape,
            inp_chans=self.atmo_embed_dim,
            out_chans=self.n_atmo_chans,
            grid_in=sht_grid_type,
            grid_out=model_grid_type,
            kernel_shape=kernel_shape,
            basis_type=filter_basis_type,
            inverse_transform=decoder_inv_transform,
            activation_function=activation_function,
            bias=bias,
            use_mlp=encoder_mlp,
            upsample_sht=upsample_sht,
        )

        # decoder for the surface variables
        if self.n_surf_chans > 0:
            self.surf_decoder = DiscreteContinuousDecoder(
                inp_shape=(self.h, self.w),
                out_shape=out_shape,
                inp_chans=self.surf_embed_dim,
                out_chans=self.n_surf_chans,
                grid_in=sht_grid_type,
                grid_out=model_grid_type,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                inverse_transform=decoder_inv_transform,
                activation_function=activation_function,
                bias=bias,
                use_mlp=encoder_mlp,
                upsample_sht=upsample_sht,
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
                inverse_transform=self.isht,
                activation_function=activation_function,
                bias=bias,
                use_mlp=encoder_mlp,
            )

        # dropout
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]


        # Internal NO blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            first_layer = i == 0
            last_layer = i == num_layers - 1

            if i % sfno_block_frequency == 0:
                attn_type = "global"
            else:
                attn_type = "local"

            block = NeuralOperatorBlock(
                self.sht,
                self.isht,
                self.total_embed_dim + (self.n_aux_chans > 0) * self.aux_embed_dim,
                self.total_embed_dim,
                attn_type=attn_type,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                path_drop_rate=dpr[i],
                act_layer=activation_function,
                normalization_layer=normalization_layer,
                num_groups=num_groups,
                skip="identity",
                layer_scale=layer_scale,
                use_mlp=use_mlp,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                bias=bias,
                checkpointing_level=checkpointing_level,
            )

            self.blocks.append(block)

        # residual prediction
        if self.big_skip:
            self.residual_transform = nn.Conv2d(self.n_out_chans, self.n_out_chans, 1, bias=False)
            self.residual_transform.weight.is_shared_mp = ["spatial"]
            self.residual_transform.weight.sharded_dims_mp = [None, None, None, None]
            if self.residual_transform.bias is not None:
                self.residual_transform.bias.is_shared_mp = ["spatial"]
                self.residual_transform.bias.sharded_dims_mp = [None]
            scale = math.sqrt(0.5 / self.n_out_chans)
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

        # controlled output normalization of q and tcwv
        if clamp_water:
            water_chans = get_water_channels(channel_names)
            if len(water_chans) > 0:
                self.register_buffer("water_channels", torch.LongTensor(water_chans), persistent=False)

        # freeze the encoder/decoder
        if freeze_encoder:
            frozen_params = list(self.atmo_encoder.parameters()) + list(self.atmo_decoder.parameters())
            if hasattr(self, "surf_encoder"):
                frozen_params += list(self.surf_encoder.parameters()) + list(self.surf_decoder.parameters())
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
        self.modes_lat = modes_lat
        self.modes_lon = modes_lon


    @torch.compiler.disable(recursive=True)
    def _precompute_channel_groups(
        self,
        channel_names=[],
        aux_channel_names=[],
    ):
        """
        group the channels appropriately into atmospheric pressure levels and surface variables
        """

        atmo_chans, surf_chans, aux_chans, pressure_lvls = get_channel_groups(channel_names, aux_channel_names)

        # compute how many channel groups will be kept internally
        self.n_atmo_groups = len(pressure_lvls)
        self.n_atmo_chans = len(atmo_chans) // self.n_atmo_groups

        # make sure they are divisible. Attention! This does not guarantee that the grrouping is correct
        if len(atmo_chans) % self.n_atmo_groups:
            raise ValueError(f"Expected number of atmospheric variables to be divisible by number of atmospheric groups but got {len(atmo_chans)} and {self.n_atmo_groups}")

        self.register_buffer("atmo_channels", torch.LongTensor(atmo_chans), persistent=False)
        self.register_buffer("surf_channels", torch.LongTensor(surf_chans), persistent=False)
        self.register_buffer("aux_channels", torch.LongTensor(aux_chans), persistent=False)

        self.n_surf_chans = self.surf_channels.shape[0]
        self.n_aux_chans = self.aux_channels.shape[0]

        return

    def encode(self, x):
        """
        forward pass for the encoder
        """
        batchdims = x.shape[:-3]

        # for atmospheric channels the same encoder is applied to each atmospheric level
        x_atmo = x[..., self.atmo_channels, :, :].contiguous().reshape(-1, self.n_atmo_chans, *x.shape[-2:])
        x_out = self.atmo_encoder(x_atmo)
        x_out = x_out.reshape(*batchdims, self.n_atmo_groups * self.atmo_embed_dim, *x_out.shape[-2:])

        if hasattr(self, "surf_encoder"):
            x_surf = x[..., self.surf_channels, :, :].contiguous()
            x_surf = self.surf_encoder(x_surf)
            x_out = torch.cat((x_out, x_surf), dim=-3)

        x_out = x_out.reshape(*batchdims, self.total_embed_dim, *x_out.shape[-2:])

        return x_out

    def encode_auxiliary_channels(self, x):
        """
        returns the embedded auxiliary channels
        """
        batchdims = x.shape[:-3]

        if hasattr(self, "aux_encoder"):
            x_aux = x[..., self.aux_channels, :, :]
            x_aux = self.aux_encoder(x_aux)
            x_aux = x_aux.reshape(*batchdims, self.aux_embed_dim, *x_aux.shape[-2:])
        else:
            x_aux = None

        return x_aux

    def decode(self, x):
        """
        forward pass for the decoder
        """

        batchdims = x.shape[:-3]

        x_atmo = x[..., : (self.n_atmo_groups * self.atmo_embed_dim), :, :].reshape(-1, self.atmo_embed_dim, *x.shape[-2:])
        x_atmo = self.atmo_decoder(x_atmo)
        x_out = torch.zeros(*batchdims, self.n_out_chans, *x_atmo.shape[-2:], dtype=x.dtype, device=x.device)
        x_out[..., self.atmo_channels, :, :] = x_atmo.reshape(*batchdims, -1, *x_atmo.shape[-2:])

        if hasattr(self, "surf_decoder"):
            x_surf = x[..., -self.surf_embed_dim :, :, :]
            x_surf = self.surf_decoder(x_surf)
            x_out[..., self.surf_channels, :, :] = x_surf.reshape(*batchdims, -1, *x_surf.shape[-2:])

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
            w = _soft_clamp(x[..., self.water_channels, :, :])
            # the following eventually leads to spectral instability
            # w = nn.functional.softplus(x[..., self.water_channels, :, :], beta=5, threshold=5)
            x[..., self.water_channels, :, :] = w

        return x

    def forward(self, x):

        # save big skip
        if self.big_skip:
            residual = x[..., : self.n_out_chans, :, :].contiguous()

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
            x = x + self.residual_transform(residual)

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
