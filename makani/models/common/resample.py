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

"""Up- and downsampling in 2D and 3D."""

import torch
import torch.nn as nn


class UpSample2D(nn.Module):
    r"""
    Learned 2x upsampling of a token grid via channel-to-space rearrangement.

    Rather than interpolating, each token is projected to four times the output
    width and that channel block is reshaped into a 2x2 spatial neighborhood --
    so the finer detail is *learned* from the coarse token, not smoothed in.
    The doubled grid is then center-cropped to ``output_resolution``, which lets
    the layer target resolutions that are not exactly twice the input, and a
    LayerNorm plus a second projection mixes the result.

    Parameters
    ----------
    in_dim : int
        Number of input channels.
    out_dim : int
        Number of output channels.
    input_resolution : tuple of int
        Input grid as ``(latitude, longitude)``.
    output_resolution : tuple of int
        Output grid as ``(latitude, longitude)``. Must be no larger than twice
        the input resolution along each axis.

    References
    ----------
    Revised from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        r"""
        Upsample the token grid and crop to the output resolution.

        Parameters
        ----------
        x : torch.Tensor
            Tokens as either ``(B, N, in_dim)`` with ``N = in_lat * in_lon``, or
            ``(B, in_lat, in_lon, in_dim)``.

        Returns
        -------
        torch.Tensor
            Tokens of shape ``(B, out_lat, out_lon, out_dim)``.
        """
        if len(x.shape) == 3:
            B, N, C = x.shape
        else:
            B, N_lat, N_lon, C = x.shape
            torch._check(
                N_lat == self.input_resolution[0],
                lambda: f"Input shape {x.shape} does not match expected input resolution {self.input_resolution}.",
            )
            torch._check(
                N_lon == self.input_resolution[1],
                lambda: f"Input shape {x.shape} does not match expected input resolution {self.input_resolution}.",
            )
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

        x = x[:, pad_top : 2 * in_lat - pad_bottom, pad_left : 2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.norm(x)
        x = self.linear2(x)
        x = x.reshape(B, out_lat, out_lon, -1)
        return x


class DownSample2D(nn.Module):
    r"""
    Learned 2x downsampling of a token grid via space-to-channel rearrangement.

    The mirror image of :class:`UpSample2D`: the input is zero-padded to exactly
    twice the output resolution, each 2x2 spatial neighborhood is folded into
    the channel dimension (giving ``4 * in_dim`` channels), and a linear layer
    then mixes those down to ``2 * in_dim``. Nothing is discarded before the
    projection, so the layer learns which detail to keep instead of committing
    to a fixed pooling rule.

    Parameters
    ----------
    in_dim : int
        Number of input channels. The output has ``2 * in_dim`` channels.
    input_resolution : tuple of int
        Input grid as ``(latitude, longitude)``.
    output_resolution : tuple of int
        Output grid as ``(latitude, longitude)``. Twice this resolution must be
        at least the input resolution along each axis; the difference is padded.

    References
    ----------
    Revised from WeatherLearn https://github.com/lizhuoq/WeatherLearn
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
        r"""
        Pad, fold 2x2 neighborhoods into channels, and project down.

        Parameters
        ----------
        x : torch.Tensor
            Tokens as either ``(B, N, in_dim)`` with ``N = in_lat * in_lon``, or
            ``(B, in_lat, in_lon, in_dim)``.

        Returns
        -------
        torch.Tensor
            Tokens of shape ``(B, out_lat, out_lon, 2 * in_dim)``.
        """
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution
        # unfold input resolution
        if len(x.shape) == 3:
            B, N, C = x.shape
            x = x.reshape(B, in_lat, in_lon, C)
        else:
            B, N_lat, N_lon, C = x.shape
            torch._check(
                N_lat == in_lat,
                lambda: f"Input shape {x.shape} does not match expected input resolution {self.input_resolution}.",
            )
            torch._check(
                N_lon == in_lon,
                lambda: f"Input shape {x.shape} does not match expected input resolution {self.input_resolution}.",
            )

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)
        x = x.reshape(B, out_lat, 2, out_lon, 2, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        x = x.reshape(B, out_lat, out_lon, -1)
        return x


class UpSample3D(nn.Module):
    r"""
    Learned 2x horizontal upsampling of a 3D token grid.

    Volumetric counterpart of :class:`UpSample2D`. Upsampling is applied to the
    latitude and longitude axes only -- the pressure-level axis is truncated to
    ``out_pl`` rather than refined, since vertical levels are physically
    distinct surfaces and interpolating between them is not meaningful the way
    horizontal refinement is.

    Parameters
    ----------
    in_dim : int
        Number of input channels.
    out_dim : int
        Number of output channels.
    input_resolution : tuple of int
        Input grid as ``(pressure levels, latitude, longitude)``.
    output_resolution : tuple of int
        Output grid as ``(pressure levels, latitude, longitude)``. The
        horizontal extents must be no larger than twice the input; ``out_pl``
        must be no larger than ``in_pl``.

    References
    ----------
    Revised from WeatherLearn https://github.com/lizhuoq/WeatherLearn,
    implementation from https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        r"""
        Upsample horizontally, then crop levels and the horizontal extent.

        Parameters
        ----------
        x : torch.Tensor
            Tokens of shape ``(B, N, in_dim)`` with
            ``N = in_pl * in_lat * in_lon``.

        Returns
        -------
        torch.Tensor
            Tokens of shape ``(B, out_pl * out_lat * out_lon, out_dim)``.
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6)
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
    r"""
    Learned 2x horizontal downsampling of a 3D token grid.

    Volumetric counterpart of :class:`DownSample2D`. Only the latitude and
    longitude axes are folded into channels; the pressure-level axis is left
    intact, so vertical resolution is preserved while the horizontal grid is
    coarsened.

    Parameters
    ----------
    in_dim : int
        Number of input channels. The output has ``2 * in_dim`` channels.
    input_resolution : tuple of int
        Input grid as ``(pressure levels, latitude, longitude)``.
    output_resolution : tuple of int
        Output grid as ``(pressure levels, latitude, longitude)``. Twice the
        horizontal extents must be at least the input's; the difference is padded.

    References
    ----------
    Revised from WeatherLearn https://github.com/lizhuoq/WeatherLearn,
    implementation from https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py
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

        self.pad = nn.ZeroPad3d((pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))

    def forward(self, x):
        r"""
        Pad horizontally, fold 2x2 neighborhoods into channels, and project down.

        Parameters
        ----------
        x : torch.Tensor
            Tokens of shape ``(B, N, in_dim)`` with
            ``N = in_pl * in_lat * in_lon``.

        Returns
        -------
        torch.Tensor
            Tokens of shape ``(B, out_pl * out_lat * out_lon, 2 * in_dim)``.
        """
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
