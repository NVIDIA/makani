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
from makani.models.common import DropPath, LayerScale, MLP
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

def _init_low_l_spectral(spec_param, isht, std=0.1, l_cutoff_frac=0.25):
    """
    Init a complex spectral parameter with small Gaussian noise on low-l modes only.

    Breaks the iSHT(0)=0 symmetry that makes the Perceiver cross-attention degenerate
    to a uniform mean-pool over each query's KV neighborhood at init. Restricting to
    low-l keeps the initial query smooth (large-scale spatial variation, no high-freq
    noise).

    Strategy: build the full global init on every rank (identical because the global
    RNG state matches), then slice out the local (l, m) portion. This makes the init
    independent of the spatial parallelism layout — a single-GPU run and a multi-GPU
    run with the same seed produce equivalent global random fields.
    """
    with torch.no_grad():
        global_l = isht.lmax
        global_m = isht.mmax
        device = spec_param.device
        inp_chans = spec_param.shape[1]

        # global mask: valid SH modes (m <= l) inside the low-l band
        l_idx = torch.arange(global_l, device=device)
        m_idx = torch.arange(global_m, device=device)
        l_cutoff = max(1, int(global_l * l_cutoff_frac))
        mask = ((m_idx.view(1, -1) <= l_idx.view(-1, 1)) &
                (l_idx.view(-1, 1) < l_cutoff))
        mask_f = mask.view(1, 1, global_l, global_m).float()

        # global Gaussian field — identical on every rank (same RNG state)
        global_real = torch.randn(1, inp_chans, global_l, global_m, device=device) * std * mask_f
        global_imag = torch.randn(1, inp_chans, global_l, global_m, device=device) * std * mask_f

        # m=0 must be real for a real iSHT (conjugate symmetry)
        global_imag[..., :, 0] = 0

        # slice the local portion held by this rank
        if isinstance(isht, thd.DistributedInverseRealSHT):
            l_start = sum(isht.l_shapes[:comm.get_rank("h")])
            l_end = l_start + isht.l_shapes[comm.get_rank("h")]
            m_start = sum(isht.m_shapes[:comm.get_rank("w")])
            m_end = m_start + isht.m_shapes[comm.get_rank("w")]
        else:
            l_start, l_end = 0, global_l
            m_start, m_end = 0, global_m

        local_real = global_real[..., l_start:l_end, m_start:m_end]
        local_imag = global_imag[..., l_start:l_end, m_start:m_end]

        spec_param.copy_(torch.complex(local_real, local_imag))


class FiLM(nn.Module):
    """
    AdaLN-Zero style spatial FiLM modulation: ``x * (1 + gamma) + beta`` where
    ``(gamma, beta)`` are produced from a conditioning tensor by a 1x1 conv.

    The projection is zero-initialized so the modulation is identity at start —
    the surrounding network trains exactly as it would without conditioning until
    FiLM learns to use it. This is the trick from DiT that lets AdaLN train from
    scratch without instability.
    """
    def __init__(self, cond_dim, feat_dim, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(cond_dim, 2 * feat_dim, 1, bias=bias)
        nn.init.zeros_(self.proj.weight)
        self.proj.weight.is_shared_mp = ["spatial"]
        self.proj.weight.sharded_dims_mp = [None, None, None, None]
        if bias:
            nn.init.zeros_(self.proj.bias)
            self.proj.bias.is_shared_mp = ["spatial"]
            self.proj.bias.sharded_dims_mp = [None]

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        return x * (1.0 + gamma) + beta


class PreLNMLP(nn.Module):
    """
    Standard transformer Pre-LN MLP residual: ``x = x + LayerScale(MLP(norm(x)))``.

    Used to add a stabilizing residual MLP after the encoder/decoder cross-attention,
    so the latent enters/leaves the deep processor stack with a consistent magnitude
    and the model has additional non-linear capacity at the boundary.
    """
    def __init__(
        self,
        chans,
        h,
        w,
        mlp_ratio=2.0,
        normalization_layer="layer_norm",
        sht_grid_type="legendre-gauss",
        act_layer=nn.GELU,
        layer_scale=True,
        gain=0.5,
        use_te=False,
    ):
        super().__init__()
        norm_handle = _get_norm_layer_handle(h, w, chans, normalization_layer, sht_grid_type)
        self.norm = norm_handle()

        MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
        self.mlp = MLPH(
            in_features=chans,
            out_features=chans,
            hidden_features=int(chans * mlp_ratio),
            act_layer=act_layer,
            gain=gain,
            use_te=use_te,
        )

        if layer_scale:
            self.layer_scale = LayerScale(chans)
            self.layer_scale.weight.is_shared_mp = ["spatial"]
            self.layer_scale.weight.sharded_dims_mp = [None, None, None, None]

    def forward(self, x):
        scale = self.layer_scale if hasattr(self, "layer_scale") else nn.Identity()
        return x + scale(self.mlp(self.norm(x)))


@torch.compile
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
        normalization_layer="layer_norm",
        layer_scale=True,
        bias=False,
        use_te=False,
    ):
        super().__init__()

        # sanity check
        if inverse_transform is None:
            raise ValueError("Encoder requires inverse_transform (the latent iSHT) to be provided")

        # heuristic for finding theta_cutoff. Sized by the LATENT (out_shape) grid so that
        # the receptive field scales with the bottleneck cell width — one latent cell wide
        # in each direction. Using inp_shape (high-res key grid) would shrink the radius
        # as input resolution grows, which is the wrong direction for a cross-attention
        # bridging into the latent.
        theta_cutoff = _compute_cutoff_radius(nlat=out_shape[0], kernel_shape=kernel_shape, basis_type=basis_type)

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
                self.attn.proj_bias.sharded_dims_mp = [None]

        # learnable query in spherical harmonic space: grid-agnostic Perceiver-style latent.
        # iSHT maps the spectral coefficients to the encoder output grid, so the query is
        # band-limited and independent of grid type or resolution.
        # InverseRealSHT has no learnable parameters (precomputed buffers only), so storing
        # the reference here does not double-count parameters from the outer model.
        self.isht = inverse_transform

        # local spectral mode counts — mirrors the SpectralConv pattern
        if isinstance(self.isht, thd.DistributedInverseRealSHT):
            modes_lat_local = self.isht.l_shapes[comm.get_rank("h")]
            modes_lon_local = self.isht.m_shapes[comm.get_rank("w")]
        else:
            modes_lat_local = self.isht.lmax
            modes_lon_local = self.isht.mmax

        # small low-l Gaussian init: breaks the iSHT(0)=0 symmetry so the cross-attention
        # softmax is position-aware from step 0 instead of starting as a uniform mean-pool.
        self.latent_query_spec = nn.Parameter(
            torch.zeros(1, inp_chans, modes_lat_local, modes_lon_local, dtype=torch.complex64)
        )
        _init_low_l_spectral(self.latent_query_spec, self.isht)
        # sharding annotation matches SpectralConv diagonal weights:
        # lat modes sharded in "h", lon modes sharded in "w"
        self.latent_query_spec.is_shared_mp = ["matmul"]
        self.latent_query_spec.sharded_dims_mp = [None, None, "h", "w"]

        # Pre-LN MLP residual after the cross-attention. Adds non-linear capacity
        # and stabilizes the latent magnitude before it enters the deep processor.
        if use_mlp:
            self.post_mlp = PreLNMLP(
                chans=out_chans,
                h=out_shape[0],
                w=out_shape[1],
                mlp_ratio=mlp_ratio,
                normalization_layer=normalization_layer,
                sht_grid_type=grid_out,
                act_layer=activation_function,
                layer_scale=layer_scale,
                use_te=use_te,
            )

    def forward(self, x):
        # spectral latent → spatial grid via iSHT
        with amp.autocast(device_type=x.device.type, enabled=False):
            query = self.latent_query_spec.to(torch.complex64).expand(x.shape[0], -1, -1, -1).contiguous()
            query = self.isht(query)
        query = query.to(dtype=x.dtype)

        # cross-attention: learned spectral queries attend to full-resolution input
        xd = self.attn(query=query, key=x, value=x)

        if hasattr(self, "post_mlp"):
            xd = self.post_mlp(xd)

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
        normalization_layer="layer_norm",
        layer_scale=True,
        bias=False,
        upsample_sht=False,
        perceiver_decoder=False,
        use_te=False,
    ):
        super().__init__()

        # sanity checks
        if perceiver_decoder and inverse_transform is None:
            raise ValueError("Perceiver decoder requires inverse transform to be provided")

        # init distributed torch-harmonics if needed
        if comm.get_size("spatial") > 1:
            polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
            azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
            thd.init(polar_group, azimuth_group)

        attn_handle = thd.DistributedNeighborhoodAttentionS2 if comm.get_size("spatial") > 1 else th.NeighborhoodAttentionS2

        if perceiver_decoder:
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
            # Small low-l Gaussian init breaks the iSHT(0)=0 symmetry; see _init_low_l_spectral.
            self.latent_query_spec = nn.Parameter(
                torch.zeros(1, inp_chans, modes_lat_local, modes_lon_local, dtype=torch.complex64)
            )
            _init_low_l_spectral(self.latent_query_spec, self.isht_out)
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
            # classic path: optional pre-mlp residual on the latent, explicit upsample, self-attention.
            # The pre-mlp is a Pre-LN MLP residual on the low-res latent, giving the decoder some
            # non-linear capacity before upsampling.
            if use_mlp:
                self.pre_mlp = PreLNMLP(
                    chans=inp_chans,
                    h=inp_shape[0],
                    w=inp_shape[1],
                    mlp_ratio=mlp_ratio,
                    normalization_layer=normalization_layer,
                    sht_grid_type=grid_in,
                    act_layer=activation_function,
                    layer_scale=layer_scale,
                    use_te=use_te,
                )

            if upsample_sht:
                sht_handle = thd.DistributedRealSHT if comm.get_size("spatial") > 1 else th.RealSHT
                isht_handle = thd.DistributedInverseRealSHT if comm.get_size("spatial") > 1 else th.InverseRealSHT
                self.sht = sht_handle(*inp_shape, grid=grid_in).float()
                self.isht = isht_handle(*out_shape, lmax=self.sht.lmax, mmax=self.sht.mmax, grid=grid_out).float()
                self.upsample = nn.Sequential(self.sht, self.isht)
            else:
                resample_handle = thd.DistributedResampleS2 if comm.get_size("spatial") > 1 else th.ResampleS2
                self.upsample = resample_handle(*inp_shape, *out_shape, grid_in=grid_in, grid_out=grid_out, mode="bilinear")

            # Size the self-attention's neighborhood by the LATENT (pre-upsample) grid,
            # not the post-upsample high-res grid. After bilinear upsample, adjacent
            # high-res cells are highly correlated — they're interpolated from the same
            # latent neighborhood — so a radius sized by the high-res spacing only spans
            # ~1 latent cell and gives the attention nothing meaningful to mix. Sizing
            # by inp_shape (latent) gives ~1 latent cell of context (~80 high-res neighbors)
            # so the self-attention can actually refine the upsampled features.
            theta_cutoff = _compute_cutoff_radius(nlat=inp_shape[0], kernel_shape=kernel_shape, basis_type=basis_type)

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
                self.attn.proj_bias.sharded_dims_mp = [None]

        # Post-attention Pre-LN MLP residual on the decoder output (high-res, out_chans space).
        # Mirrors the encoder's post_mlp; gives the decoder non-linear capacity at the boundary
        # back to the prediction grid.
        if use_mlp:
            self.post_mlp = PreLNMLP(
                chans=out_chans,
                h=out_shape[0],
                w=out_shape[1],
                mlp_ratio=mlp_ratio,
                normalization_layer=normalization_layer,
                sht_grid_type=grid_out,
                act_layer=activation_function,
                layer_scale=layer_scale,
                use_te=use_te,
            )

    def forward(self, x):
        if hasattr(self, "latent_query_spec"):
            # Perceiver-style: project spectral query to output grid, cross-attend to latent
            with amp.autocast(device_type=x.device.type, enabled=False):
                query = self.latent_query_spec.to(torch.complex64).expand(x.shape[0], -1, -1, -1).contiguous()
                query = self.isht_out(query)
            query = query.to(dtype=x.dtype)
            x = self.attn(query=query, key=x, value=x)
        else:
            # classic path: optional pre-mlp residual on the latent, upsample, self-attention
            if hasattr(self, "pre_mlp"):
                x = self.pre_mlp(x)
            dtype = x.dtype
            with amp.autocast(device_type=x.device.type, enabled=False):
                x = x.to(torch.float32)
                x = self.upsample(x)
            x = x.to(dtype=dtype)
            x = self.attn(x)

        if hasattr(self, "post_mlp"):
            x = self.post_mlp(x)

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
        kernel_shape=(3, 3),
        basis_type="morlet",
        checkpointing_level=0,
        bias=False,
        aux_embed_dim=0,
        use_te=False,
    ):
        super().__init__()

        # determine some shapes
        self.inp_shape = (forward_transform.nlat, forward_transform.nlon)
        self.out_shape = (inverse_transform.nlat, inverse_transform.nlon)
        self.out_chans = out_chans

        # gain factor for the convolution
        gain_factor = 0.5

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
                    self.attn.proj_bias.sharded_dims_mp = [None]

            # match the gain convention used by SpectralConv/MLP: scale the output
            # projection so the residual branch's output variance is multiplied by gain_factor
            with torch.no_grad():
                self.attn.proj_weights *= math.sqrt(gain_factor)

        elif attn_type == "global":
            # global spectral convolution — torch_harmonics SpectralConvS2 (Driscoll-Healy).
            # Bias is intentionally disabled: at our spectral budget (lmax = mmax ~ modes_lat),
            # the spectral_bias parameter would add hundreds of millions of params per global
            # block. The convolution alone (per-l weights) provides the rotation-equivariant
            # mixing.
            sconv_handle = thd.DistributedSpectralConvS2 if comm.get_size("spatial") > 1 else th.SpectralConvS2
            self.global_conv = sconv_handle(
                in_shape=self.inp_shape,
                out_shape=self.out_shape,
                in_channels=inp_chans,
                out_channels=inp_chans,
                num_groups=num_groups,
                grid_in=forward_transform.grid,
                grid_out=inverse_transform.grid,
                bias=False,
            )
            # match the gain convention: scale weights so the residual branch's variance
            # is multiplied by gain_factor (default init uses sqrt(1/in_chans); we want sqrt(gain/in_chans)).
            with torch.no_grad():
                self.global_conv.weight *= math.sqrt(gain_factor)

            # sharding annotations: weight shape is (num_groups, in/groups, out/groups, lmax_local).
            # Replicated across matmul, sharded along "h" (polar) on the lmax dim.
            # (torch_harmonics deliberately leaves grad-reduction strategy to the caller.)
            self.global_conv.weight.is_shared_mp = ["matmul"]
            self.global_conv.weight.sharded_dims_mp = [None, None, None, "h"]
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
            use_te=use_te,
        )

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

        # AdaLN-Zero style FiLM modulation from x_aux. One head per Pre-LN site.
        # Zero-init means the block is identity in aux at start; the model trains
        # as if there is no conditioning until the FiLM heads learn to use it.
        if aux_embed_dim > 0:
            self.film_attn = FiLM(aux_embed_dim, inp_chans, bias=True)
            self.film_mlp = FiLM(aux_embed_dim, inp_chans, bias=True)

    def forward(self, x, x_aux=None):
        """
        Updated NO block
        """
        attn_scale = self.layer_scale1 if hasattr(self, "layer_scale1") else nn.Identity()
        mlp_scale  = self.layer_scale2 if hasattr(self, "layer_scale2") else nn.Identity()

        # mixing sub-layer: Pre-LN residual, local attention or global spectral conv
        x_norm = self.norm1(x)
        if x_aux is not None and hasattr(self, "film_attn"):
            x_norm = self.film_attn(x_norm, x_aux)
        if hasattr(self, "global_conv"):
            mix_out = self.global_conv(x_norm)
        else:
            mix_out = self.attn(x_norm)
        x = x + self.drop_path(attn_scale(mix_out))

        # mlp sub-layer
        x_norm2 = self.norm2(x)
        if x_aux is not None and hasattr(self, "film_mlp"):
            x_norm2 = self.film_mlp(x_norm2, x_aux)
        x = x + self.drop_path(mlp_scale(self.mlp(x_norm2)))

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
        embed_dim=None,
        num_layers=4,
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
        perceiver_decoder=False,
        num_groups=1,
        use_te=True,
        **kwargs,
    ):
        super().__init__()

        # Perceiver-style decoder is not yet supported under spatial model parallelism
        # because DistributedNeighborhoodAttentionS2 cannot do low-res KV → high-res Q.
        if perceiver_decoder and comm.get_size("spatial") > 1:
            raise NotImplementedError(
                "perceiver_decoder=True is not supported with spatial model parallelism yet"
            )

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.atmo_embed_dim = atmo_embed_dim
        self.surf_embed_dim = surf_embed_dim
        self.aux_embed_dim = aux_embed_dim
        self.embed_dim = embed_dim
        # unified-encoder mode: when embed_dim is set, all predicted channels share
        # a single encoder/decoder pair at that width (FCN3-style). When None, fall
        # back to the per-pressure-level (per-group) encoder/decoder design.
        self.unified_encoder = (embed_dim is not None)
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
        if self.unified_encoder:
            self.total_embed_dim = self.embed_dim
        else:
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

        if self.unified_encoder:
            # single encoder/decoder pair over all predicted channels (atmo concat surf).
            # Trades the per-level weight-sharing prior for a narrower, cheaper processor.
            self.encoder = DiscreteContinuousEncoder(
                inp_shape=inp_shape,
                out_shape=(self.h, self.w),
                inp_chans=self.n_out_chans,
                out_chans=self.embed_dim,
                grid_in=model_grid_type,
                grid_out=sht_grid_type,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                inverse_transform=self.isht,
                activation_function=activation_function,
                normalization_layer=normalization_layer,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                bias=bias,
                use_mlp=encoder_mlp,
                use_te=use_te,
            )
            self.decoder = DiscreteContinuousDecoder(
                inp_shape=(self.h, self.w),
                out_shape=out_shape,
                inp_chans=self.embed_dim,
                out_chans=self.n_out_chans,
                grid_in=sht_grid_type,
                grid_out=model_grid_type,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                inverse_transform=decoder_inv_transform,
                activation_function=activation_function,
                normalization_layer=normalization_layer,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                bias=bias,
                use_mlp=encoder_mlp,
                upsample_sht=upsample_sht,
                perceiver_decoder=perceiver_decoder,
                use_te=use_te,
            )
        else:
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
                normalization_layer=normalization_layer,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                bias=bias,
                use_mlp=encoder_mlp,
                use_te=use_te,
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
                    normalization_layer=normalization_layer,
                    layer_scale=layer_scale,
                    mlp_ratio=mlp_ratio,
                    bias=bias,
                    use_mlp=encoder_mlp,
                    use_te=use_te,
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
                normalization_layer=normalization_layer,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                bias=bias,
                use_mlp=encoder_mlp,
                upsample_sht=upsample_sht,
                perceiver_decoder=perceiver_decoder,
                use_te=use_te,
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
                    normalization_layer=normalization_layer,
                    layer_scale=layer_scale,
                    mlp_ratio=mlp_ratio,
                    bias=bias,
                    use_mlp=encoder_mlp,
                    upsample_sht=upsample_sht,
                    perceiver_decoder=perceiver_decoder,
                    use_te=use_te,
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
                normalization_layer=normalization_layer,
                layer_scale=layer_scale,
                mlp_ratio=mlp_ratio,
                bias=bias,
                use_mlp=encoder_mlp,
                use_te=use_te,
            )

        # dropout
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]


        # Internal NO blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            if i % sfno_block_frequency == 0:
                attn_type = "global"
            else:
                attn_type = "local"

            block = NeuralOperatorBlock(
                self.sht,
                self.isht,
                self.total_embed_dim,
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
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                bias=bias,
                checkpointing_level=checkpointing_level,
                aux_embed_dim=self.aux_embed_dim if self.n_aux_chans > 0 else 0,
                use_te=use_te,
            )

            self.blocks.append(block)

        # residual prediction
        if self.big_skip:
            self.residual_transform = nn.Conv2d(self.n_out_chans, self.n_out_chans, 1, bias=False)
            self.residual_transform.weight.is_shared_mp = ["spatial"]
            self.residual_transform.weight.sharded_dims_mp = [None, None, None, None]
            scale = math.sqrt(0.5 / self.n_out_chans)
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

            # SHT roundtrip filter: bandwidth-limit the residual to the model's effective
            # bandwidth so the input's high-l content (which the model cannot have predicted
            # through its own latent at lmax = self.modes_lat) does not contaminate the
            # output with frequencies the rest of the network never modeled.
            sht_handle  = thd.DistributedRealSHT        if comm.get_size("spatial") > 1 else th.RealSHT
            isht_handle = thd.DistributedInverseRealSHT if comm.get_size("spatial") > 1 else th.InverseRealSHT
            self.residual_sht = sht_handle(
                out_shape[0], out_shape[1],
                lmax=self.modes_lat, mmax=self.modes_lon,
                grid=model_grid_type,
            ).float()
            self.residual_isht = isht_handle(
                out_shape[0], out_shape[1],
                lmax=self.modes_lat, mmax=self.modes_lon,
                grid=model_grid_type,
            ).float()

        # controlled output normalization of q and tcwv
        if clamp_water:
            water_chans = get_water_channels(channel_names)
            if len(water_chans) > 0:
                self.register_buffer("water_channels", torch.LongTensor(water_chans), persistent=False)

        # freeze the encoder/decoder
        if freeze_encoder:
            if self.unified_encoder:
                frozen_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            else:
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

        atmo_chans, surf_chans, dyn_aux_chans, stat_aux_chans, pressure_lvls = get_channel_groups(channel_names, aux_channel_names)
        aux_chans = dyn_aux_chans + stat_aux_chans

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
        if self.unified_encoder:
            # gather predicted channels (atmo concat surf) and run a single encoder
            if self.n_surf_chans > 0:
                x_in = torch.cat([x[..., self.atmo_channels, :, :], x[..., self.surf_channels, :, :]], dim=-3)
            else:
                x_in = x[..., self.atmo_channels, :, :]
            return self.encoder(x_in)

        batchdims = x.shape[:-3]

        # for atmospheric channels the same encoder is applied to each atmospheric level
        x_atmo = x[..., self.atmo_channels, :, :].reshape(-1, self.n_atmo_chans, *x.shape[-2:])
        x_out = self.atmo_encoder(x_atmo)
        x_out = x_out.reshape(*batchdims, self.n_atmo_groups * self.atmo_embed_dim, *x_out.shape[-2:])

        if hasattr(self, "surf_encoder"):
            x_surf = x[..., self.surf_channels, :, :]
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

        if self.unified_encoder:
            # single decoder produces all n_out_chans at once, ordered atmo then surf;
            # scatter back into the canonical channel layout used by the rest of the model.
            y = self.decoder(x)
            x_out = torch.zeros(*batchdims, self.n_out_chans, *y.shape[-2:], dtype=y.dtype, device=y.device)
            n_atmo_out = self.n_atmo_groups * self.n_atmo_chans
            x_out[..., self.atmo_channels, :, :] = y[..., :n_atmo_out, :, :]
            if self.n_surf_chans > 0:
                x_out[..., self.surf_channels, :, :] = y[..., n_atmo_out:, :, :]
            return x_out

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

        # do the feature extraction. Each block conditions on x_aux via FiLM
        # (AdaLN-Zero); aux is no longer concatenated as channels.
        for blk in self.blocks:
            if self.checkpointing_level >= 3:
                x = checkpoint(blk, x, x_aux, use_reentrant=False)
            else:
                x = blk(x, x_aux)

        return x

    def clamp_water_channels(self, x):
        """clamp water channels with a smooth, positive activation function"""
        if hasattr(self, "water_channels"):
            w = _soft_clamp(x[..., self.water_channels, :, :])
            # the following eventually leads to spectral instability
            # w = nn.functional.softplus(x[..., self.water_channels, :, :], beta=5, threshold=5)
            x = x.index_copy(-3, self.water_channels, w.to(x.dtype))

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
            # bandwidth-limit the residual via SHT roundtrip before mixing it in
            with amp.autocast(device_type=x.device.type, enabled=False):
                residual_lp = self.residual_isht(self.residual_sht(residual.float()))
            x = x + self.residual_transform(residual_lp.to(x.dtype))

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
