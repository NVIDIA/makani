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

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
import warnings

from makani.utils.context import rng_context

# transformer engine is an optional dependency: it is only used for the
# (optional) FP8/FP4 MLP path and must not be required for import. availability is
# checked without importing it; the module is imported lazily where used.
from makani.utils.te_helpers import TE_AVAILABLE as _TE_AVAILABLE, get_te


@torch.compile(fullgraph=False)
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


class SeededDropout2d(nn.Module):
    def __init__(self, drop_prob=0.0, seed=333):
        super(SeededDropout2d, self).__init__()
        self.drop_prob = drop_prob
        self.seed = seed
        self.drop = nn.Dropout2d(p=self.drop_prob)

        # set RNG states
        self.rng_cpu = torch.Generator(device=torch.device("cpu"))
        self.rng_cpu.manual_seed(seed)
        self.rng_gpu = None
        if torch.cuda.is_available():
            self.rng_gpu = torch.Generator(device=torch.cuda.current_device())
            self.rng_gpu.manual_seed(seed)

    # rng_context swaps the global RNG state via (cuda.)get_rng_state/set_rng_state and
    # stateful torch.Generator objects, none of which Dynamo can trace. Mark the whole
    # forward as a compile boundary so it runs eagerly instead of erroring/graph-breaking
    # mid-trace; the surrounding graph still compiles around it.
    @torch.compiler.disable
    def forward(self, x):
        with rng_context(self.rng_cpu, self.rng_gpu):
            xdrop = self.drop(x)
        return xdrop


class LayerScale(nn.Module):
    def __init__(self, num_chans=3, init_value=0.1):
        super().__init__()
        self.num_chans = num_chans
        self.weight = nn.Parameter(torch.randn(self.num_chans, 1, 1, 1))
        torch.nn.init.constant_(self.weight, val=init_value)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, groups=self.num_chans)


class PatchEmbed2D(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768, padding=False, flatten=True, norm_layer=None):
        super().__init__()
        self.red_img_size = ((img_size[0] // patch_size[0]), (img_size[1] // patch_size[1]))
        self.num_patches = self.red_img_size[0] * self.red_img_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.proj.weight.is_shared_mp = ["spatial"]
        self.proj.bias.is_shared_mp = ["spatial"]
        self.padding=padding
        self.flatten = flatten

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        if self.padding:
            padding_left = padding_right = padding_top = padding_bottom = 0
            h_remainder = self.img_size[0] % self.patch_size[0]
            w_remainder = self.img_size[1] % self.patch_size[1]
            if h_remainder:
                h_pad = self.patch_size[0] - h_remainder
                padding_top = h_pad // 2
                padding_bottom = int(h_pad - padding_top)
            if w_remainder:
                w_pad = self.patch_size[1] - w_remainder
                padding_left = w_pad // 2
                padding_right = int(w_pad - padding_left)
            self.pad = nn.ZeroPad2d(
                (padding_left, padding_right, padding_top, padding_bottom)
            )

    def forward(self, x):
        # gather input
        B, C, H, W = x.shape
        if self.padding:
            x = self.pad(x)
        torch._check(H == self.img_size[0], lambda: f"Input image height {H} doesn't match model {self.img_size[0]}.")
        torch._check(W == self.img_size[1], lambda: f"Input image width {W} doesn't match model {self.img_size[1]}.")
        # forward pass
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # flatten: new: B, C, H*W
        if self.flatten:
            x = x.flatten(2)
        return x

class PatchEmbed3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D Image to Patch Embedding.

    Args:
        img_size (tuple[int]): Image size.
        patch_size (tuple[int]): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim(int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, padding=False, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.padding=padding
        level, height, width = img_size

        if self.padding:
            l_patch_size, h_patch_size, w_patch_size = patch_size
            padding_left = (
                padding_right
            ) = padding_top = padding_bottom = padding_front = padding_back = 0

            l_remainder = level % l_patch_size
            h_remainder = height % h_patch_size
            w_remainder = width % w_patch_size

            if l_remainder:
                l_pad = l_patch_size - l_remainder
                padding_front = l_pad // 2
                padding_back = l_pad - padding_front
            if h_remainder:
                h_pad = h_patch_size - h_remainder
                padding_top = h_pad // 2
                padding_bottom = h_pad - padding_top
            if w_remainder:
                w_pad = w_patch_size - w_remainder
                padding_left = w_pad // 2
                padding_right = w_pad - padding_left

            self.pad = nn.ZeroPad3d(
                (
                    padding_left,
                    padding_right,
                    padding_top,
                    padding_bottom,
                    padding_front,
                    padding_back,
                )
            )
        # proj
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, L, H, W = x.shape
        if self.padding:
            x = self.pad(x)
        x = self.proj(x)
        if self.norm:
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x

class PatchRecovery2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        output = self.conv(x)

        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[
            :, :, padding_top : H - padding_bottom, padding_left : W - padding_right
        ]

class PatchRecovery3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[
            :,
            :,
            padding_front : Pl - padding_back,
            padding_top : Lat - padding_bottom,
            padding_left : Lon - padding_right,
        ]

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        output_dim,
        hidden_dim,
        act_layer,
        gain=1.0,
        input_format="nchw",
        groups=1,
    ):
        super(EncoderDecoder, self).__init__()

        encoder_modules = []
        current_dim = input_dim
        for i in range(num_layers):
            # fully connected layer
            if input_format == "nchw":
                encoder_modules.append(nn.Conv2d(current_dim, hidden_dim, 1, bias=True, groups=groups))
            elif input_format == "traditional":
                encoder_modules.append(nn.Linear(current_dim, hidden_dim, bias=True))
            else:
                raise NotImplementedError(f"Error, input format {input_format} not supported.")

            # weight sharing
            encoder_modules[-1].weight.is_shared_mp = ["spatial"]

            # proper initializaiton (fan-in per group for grouped conv)
            fan_in = (current_dim // groups) if input_format == "nchw" else current_dim
            scale = math.sqrt(2.0 / fan_in)
            nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
            if encoder_modules[-1].bias is not None:
                encoder_modules[-1].bias.is_shared_mp = ["spatial"]
                nn.init.constant_(encoder_modules[-1].bias, 0.0)

            encoder_modules.append(act_layer())
            current_dim = hidden_dim

        # final output layer
        if input_format == "nchw":
            encoder_modules.append(nn.Conv2d(current_dim, output_dim, 1, bias=False, groups=groups))
        elif input_format == "traditional":
            encoder_modules.append(nn.Linear(current_dim, output_dim, bias=False))

        # weight sharing
        encoder_modules[-1].weight.is_shared_mp = ["spatial"]

        # proper initializaiton
        fan_in = (current_dim // groups) if input_format == "nchw" else current_dim
        scale = math.sqrt(gain / fan_in)
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
        checkpointing=False,
        gain=1.0,
        use_te=False,
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        self.input_format = input_format
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # only use transformer engine if it was requested and is actually available
        self.use_te = use_te and _TE_AVAILABLE
        if use_te and not _TE_AVAILABLE:
            warnings.warn("use_te=True was requested but transformer_engine is not installed; falling back to the standard MLP.")

        # sanity checks
        if (input_format == "traditional") and (drop_type == "features"):
            raise NotImplementedError(f"Error, traditional input format and feature dropout cannot be selected simultaneously")

        # transformer engine linears operate on the last (channel) dimension; for
        # nchw inputs we transpose to channels-last around the GEMMs (see forward).
        if self.use_te:
            te = get_te()
            fc1 = te.Linear(in_features, hidden_features, bias=True)
            fc2 = te.Linear(hidden_features, out_features, bias=output_bias)
        elif input_format == "nchw":
            fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
            fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
        elif input_format == "traditional":
            fc1 = nn.Linear(in_features, hidden_features, bias=True)
            fc2 = nn.Linear(hidden_features, out_features, bias=output_bias)
        else:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        # sharing settings: weights/biases are replicated across the spatial model
        # group, so the gradient reduction hook sums them over "spatial". This must
        # be stamped on every parameter (including te.Linear ones, which otherwise
        # arrive unannotated) for the comm hook to reduce them correctly.
        fc1.weight.is_shared_mp = ["spatial"]
        fc1.bias.is_shared_mp = ["spatial"]
        fc2.weight.is_shared_mp = ["spatial"]
        if fc2.bias is not None:
            fc2.bias.is_shared_mp = ["spatial"]

        # initialize the weights correctly (identical to the standard path so that
        # toggling use_te does not change initialization)
        nn.init.normal_(fc1.weight, mean=0.0, std=math.sqrt(2.0 / in_features))
        nn.init.constant_(fc1.bias, 0.0)
        # gain factor for the output determines the scaling of the output init
        nn.init.normal_(fc2.weight, mean=0.0, std=math.sqrt(gain / hidden_features))
        if fc2.bias is not None:
            nn.init.constant_(fc2.bias, 0.0)

        # activation
        act = act_layer()

        if drop_rate > 0.0:
            if drop_type == "iid":
                drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Error, drop_type {drop_type} not supported")
        else:
            drop = nn.Identity()

        if self.use_te:
            # keep the modules separate so forward can insert the channels-last
            # transposes; dropout stays in nchw layout so "features" dropout keeps
            # dropping channels (dim=1) as in the standard path.
            self.fc1 = fc1
            self.fc2 = fc2
            self.act = act
            self.drop = drop
        else:
            # create forward pass
            self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)

    def _te_forward(self, x):
        if self.input_format == "nchw":
            # nchw -> nhwc for the te GEMM, back to nchw for (channel) dropout
            x = self.fc1(x.permute(0, 2, 3, 1).contiguous())
            x = self.act(x)
            x = self.drop(x.permute(0, 3, 1, 2).contiguous())
            x = self.fc2(x.permute(0, 2, 3, 1).contiguous())
            x = self.drop(x.permute(0, 3, 1, 2).contiguous())
        else:
            # traditional format already has the channels in the last dimension
            x = self.drop(self.act(self.fc1(x)))
            x = self.drop(self.fc2(x))
        return x

    @torch.compiler.disable(recursive=False)
    def checkpoint_forward(self, x):
        fwd = self._te_forward if self.use_te else self.fwd
        return checkpoint(fwd, x, use_reentrant=False)

    def forward(self, x):
        if self.checkpointing:
            return self.checkpoint_forward(x)
        elif self.use_te:
            return self._te_forward(x)
        else:
            return self.fwd(x)

class UpSample2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D Up-sampling operation.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [latitude, longitude]
        output_resolution (tuple[int]): [latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        if len(x.shape) == 3:
            B, N, C = x.shape
        else:
            B, N_lat, N_lon, C = x.shape
            torch._check(N_lat == self.input_resolution[0], lambda: f"Input shape {x.shape} does not match expected input resolution {self.input_resolution}.")
            torch._check(N_lon == self.input_resolution[1], lambda: f"Input shape {x.shape} does not match expected input resolution {self.input_resolution}.")
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, in_lat * 2, in_lon * 2, -1)

        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[
            :, pad_top : 2 * in_lat - pad_bottom, pad_left : 2 * in_lon - pad_right, :
        ]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.norm(x)
        x = self.linear2(x)
        x = x.reshape(B, out_lat, out_lon, -1)
        return x

class DownSample2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D Down-sampling operation

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [latitude, longitude]
        output_resolution (tuple[int]): [latitude, longitude]
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution

        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        self.pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x: torch.Tensor):
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution
        # unfold input resolution
        if len(x.shape) == 3:
            B, N, C = x.shape
            x = x.reshape(B, in_lat, in_lon, C)
        else:
            B, N_lat, N_lon, C = x.shape
            torch._check(N_lat == in_lat, lambda: f"Input shape {x.shape} does not match expected input resolution {self.input_resolution}.")
            torch._check(N_lon == in_lon, lambda: f"Input shape {x.shape} does not match expected input resolution {self.input_resolution}.")

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)
        x = x.reshape(B, out_lat, 2, out_lon, 2, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        x = x.reshape(B, out_lat, out_lon, -1)
        return x

class UpSample3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D Up-sampling operation.
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(
            0, 1, 2, 4, 3, 5, 6
        )
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)

        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[
            :,
            :out_pl,
            pad_top : 2 * in_lat - pad_bottom,
            pad_left : 2 * in_lon - pad_right,
            :,
        ]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x

class DownSample3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D Down-sampling operation
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = nn.ZeroPad3d(
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        )

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)

        return x
