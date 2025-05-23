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

from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch.cuda import amp
from typing import Optional
import math

from makani.models.common.contractions import compl_muladd2d_fwd, compl_mul2d_fwd
from makani.models.common.contractions import _contract_diagonal


@torch.jit.script
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2d ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.red_img_size = ((img_size[0] // patch_size[0]), (img_size[1] // patch_size[1]))
        num_patches = self.red_img_size[0] * self.red_img_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.proj.weight.is_shared_mp = ["spatial"]
        self.proj.bias.is_shared_mp = ["spatial"]

    def forward(self, x):
        # gather input
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, hidden_dim, act_layer, gain=1.0, input_format="nchw"):
        super(EncoderDecoder, self).__init__()

        encoder_modules = []
        current_dim = input_dim
        for i in range(num_layers):
            # fully connected layer
            if input_format == "nchw":
                encoder_modules.append(nn.Conv2d(current_dim, hidden_dim, 1, bias=True))
            elif input_format == "traditional":
                encoder_modules.append(nn.Linear(current_dim, hidden_dim, bias=True))
            else:
                raise NotImplementedError(f"Error, input format {input_format} not supported.")

            # weight sharing
            encoder_modules[-1].weight.is_shared_mp = ["spatial"]

            # proper initializaiton
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
            if encoder_modules[-1].bias is not None:
                encoder_modules[-1].bias.is_shared_mp = ["spatial"]
                nn.init.constant_(encoder_modules[-1].bias, 0.0)

            encoder_modules.append(act_layer())
            current_dim = hidden_dim

        # final output layer
        if input_format == "nchw":
            encoder_modules.append(nn.Conv2d(current_dim, output_dim, 1, bias=False))
        elif input_format == "traditional":
            encoder_modules.append(nn.Linear(current_dim, output_dim, bias=False))

        # weight sharing
        encoder_modules[-1].weight.is_shared_mp = ["spatial"]

        # proper initializaiton
        scale = math.sqrt(gain / current_dim)
        nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
        if encoder_modules[-1].bias is not None:
            encoder_modules[-1].bias.is_shared_mp = ["spatial"]
            nn.init.constant_(encoder_modules[-1].bias, 0.0)

        self.fwd = nn.Sequential(*encoder_modules)

    def forward(self, x):
        return self.fwd(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        output_bias=True,
        input_format="nchw",
        drop_rate=0.0,
        drop_type="iid",
        checkpointing=0,
        gain=1.0,
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # First fully connected layer
        if input_format == "nchw":
            fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
            fc1.weight.is_shared_mp = ["spatial"]
            fc1.bias.is_shared_mp = ["spatial"]
        elif input_format == "traditional":
            fc1 = nn.Linear(in_features, hidden_features, bias=True)
        else:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        # initialize the weights correctly
        scale = math.sqrt(2.0 / in_features)
        nn.init.normal_(fc1.weight, mean=0.0, std=scale)
        nn.init.constant_(fc1.bias, 0.0)

        # activation
        act = act_layer()

        # sanity checks
        if (input_format == "traditional") and (drop_type == "features"):
            raise NotImplementedError(f"Error, traditional input format and feature dropout cannot be selected simultaneously")

        # output layer
        if input_format == "nchw":
            fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
            fc2.weight.is_shared_mp = ["spatial"]
            if output_bias:
                fc2.bias.is_shared_mp = ["spatial"]
        elif input_format == "traditional":
            fc2 = nn.Linear(hidden_features, out_features, bias=output_bias)
        else:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features)
        nn.init.normal_(fc2.weight, mean=0.0, std=scale)
        if fc2.bias is not None:
            nn.init.constant_(fc2.bias, 0.0)

        if drop_rate > 0.0:
            if drop_type == "iid":
                drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Error, drop_type {drop_type} not supported")
        else:
            drop = nn.Identity()

        # create forward pass
        self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)

    @torch.jit.ignore
    def checkpoint_forward(self, x):
        return checkpoint(self.fwd, x, use_reentrant=False)

    def forward(self, x):
        if self.checkpointing >= 2:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)


class RealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):
        super(RealFFT2, self).__init__()

        # use local FFT here
        self.fft_handle = torch.fft.rfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x):
        y = self.fft_handle(x, s=(self.nlat, self.nlon), dim=(-2, -1), norm="ortho")

        if self.truncate:
            y = torch.cat((y[..., : self.lmax_high, : self.mmax], y[..., -self.lmax_low :, : self.mmax]), dim=-2)

        return y


class InverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):
        super(InverseRealFFT2, self).__init__()

        # use local FFT here
        self.ifft_handle = torch.fft.irfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x):
        # truncation is implicit but better do it manually
        xt = x[..., : self.mmax]

        if self.truncate:
            # pad
            xth = xt[..., : self.lmax_high, :]
            xtl = xt[..., -self.lmax_low :, :]
            xthp = F.pad(xth, (0, 0, 0, self.nlat - self.lmax))
            xt = torch.cat([xthp, xtl], dim=-2)

        out = torch.fft.irfft2(xt, s=(self.nlat, self.nlon), dim=(-2, -1), norm="ortho")

        return out
