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

"""Patch embeddings and recoveries in 2D and 3D."""

import torch
import torch.nn as nn


class PatchEmbed2D(nn.Module):
    r"""
    Split a 2D field into non-overlapping patches and embed each one.

    The standard vision-transformer stem: a strided convolution whose kernel and
    stride both equal ``patch_size`` computes one ``embed_dim``-dimensional
    token per patch in a single op. This reduces an ``(H, W)`` field to
    ``(H/p_h, W/p_w)`` tokens, which is what makes attention over a
    high-resolution field affordable.

    Parameters
    ----------
    img_size : (int, int), optional
        Expected input height and width, by default ``(224, 224)``. Checked at
        runtime in ``forward``.
    patch_size : (int, int), optional
        Patch height and width, by default ``(16, 16)``.
    in_chans : int, optional
        Number of input channels, by default ``3``.
    embed_dim : int, optional
        Dimension of each output token, by default ``768``.
    padding : bool, optional
        If ``True``, symmetrically zero-pad the input so both dimensions become
        divisible by ``patch_size``, by default ``False``.
    flatten : bool, optional
        If ``True`` (the default), flatten the two spatial dimensions into a
        single token axis. If ``False``, the patch grid is kept 2D.
    norm_layer : callable, optional
        Normalization applied to the embedded tokens over the channel dimension,
        called as ``norm_layer(embed_dim)``. By default no normalization.

    Notes
    -----
    The projection weight and bias are tagged
    ``is_shared_mp = {"spatial": "sum", "matmul": "avg"}``: they are replicated
    rather than sharded, and their gradients are summed over the spatial ranks,
    each holding the contribution of its part of the grid, and averaged over
    the matmul ranks, which compute the same.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        padding=False,
        flatten=True,
        norm_layer=None,
    ):
        super().__init__()
        self.red_img_size = ((img_size[0] // patch_size[0]), (img_size[1] // patch_size[1]))
        self.num_patches = self.red_img_size[0] * self.red_img_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.proj.weight.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
        self.proj.bias.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
        self.padding = padding
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
            self.pad = nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))

    def forward(self, x):
        r"""
        Embed the input field into patch tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, in_chans, H, W)``, where ``(H, W)`` must equal
            ``img_size``.

        Returns
        -------
        torch.Tensor
            Token tensor of shape ``(B, embed_dim, num_patches)`` if
            ``flatten`` is ``True``, otherwise ``(B, embed_dim, H/p_h, W/p_w)``.
        """
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
    r"""
    Split a 3D volume into non-overlapping patches and embed each one.

    Volumetric counterpart of :class:`PatchEmbed2D`, used by models that treat
    pressure levels as a third spatial axis rather than as channels: patches
    then span a block of levels as well as a lat-lon tile, so vertical structure
    is captured in the token itself. The patch grid is kept 3D (no flattening).

    Parameters
    ----------
    img_size : tuple of int
        Expected input size as ``(level, height, width)``.
    patch_size : tuple of int
        Patch size as ``(level, height, width)``.
    in_chans : int
        Number of input channels.
    embed_dim : int
        Dimension of each output token.
    padding : bool, optional
        If ``True``, symmetrically zero-pad each axis so it becomes divisible by
        the corresponding patch size, by default ``False``.
    norm_layer : callable, optional
        Normalization applied to the embedded tokens over the channel
        dimension, called as ``norm_layer(embed_dim)``. By default none.

    References
    ----------
    Revised from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, padding=False, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.padding = padding
        level, height, width = img_size

        if self.padding:
            l_patch_size, h_patch_size, w_patch_size = patch_size
            padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0

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
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        r"""
        Embed the input volume into patch tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, in_chans, L, H, W)``.

        Returns
        -------
        torch.Tensor
            Token tensor of shape ``(B, embed_dim, L/p_l, H/p_h, W/p_w)``,
            computed on the padded input when ``padding`` is enabled.
        """
        B, C, L, H, W = x.shape
        if self.padding:
            x = self.pad(x)
        x = self.proj(x)
        if self.norm:
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x


class PatchRecovery2D(nn.Module):
    r"""
    Decode patch tokens back into a full-resolution 2D field.

    Inverse of :class:`PatchEmbed2D`: a transposed convolution with kernel and
    stride equal to ``patch_size`` expands each token back into its patch. Since
    the token grid may cover more area than the original field (the encoder can
    pad up to a whole number of patches), the result is center-cropped back to
    ``img_size``.

    Parameters
    ----------
    img_size : tuple of int
        Target output size as ``(lat, lon)``.
    patch_size : tuple of int
        Patch size as ``(lat, lon)``; must match the encoder's.
    in_chans : int
        Number of input channels, i.e. the token embedding dimension.
    out_chans : int
        Number of output channels.

    References
    ----------
    Revised from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        r"""
        Expand tokens to grid space and center-crop to ``img_size``.

        Parameters
        ----------
        x : torch.Tensor
            Token tensor of shape ``(B, in_chans, H_p, W_p)`` on the patch grid.

        Returns
        -------
        torch.Tensor
            Field of shape ``(B, out_chans, lat, lon)``.
        """
        output = self.conv(x)

        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[:, :, padding_top : H - padding_bottom, padding_left : W - padding_right]


class PatchRecovery3D(nn.Module):
    r"""
    Decode patch tokens back into a full-resolution 3D volume.

    Volumetric counterpart of :class:`PatchRecovery2D` and the inverse of
    :class:`PatchEmbed3D`: a transposed convolution expands each token into its
    ``(level, lat, lon)`` patch, and the result is center-cropped back to
    ``img_size`` to undo any padding the encoder applied.

    Parameters
    ----------
    img_size : tuple of int
        Target output size as ``(pl, lat, lon)``.
    patch_size : tuple of int
        Patch size as ``(pl, lat, lon)``; must match the encoder's.
    in_chans : int
        Number of input channels, i.e. the token embedding dimension.
    out_chans : int
        Number of output channels.

    References
    ----------
    Revised from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        r"""
        Expand tokens to grid space and center-crop to ``img_size``.

        Parameters
        ----------
        x : torch.Tensor
            Token tensor of shape ``(B, in_chans, L_p, H_p, W_p)`` on the patch grid.

        Returns
        -------
        torch.Tensor
            Volume of shape ``(B, out_chans, pl, lat, lon)``.
        """
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
