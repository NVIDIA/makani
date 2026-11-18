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

"""The spatially distributed patch embedding."""

import torch
import torch.nn as nn

from torch_harmonics.distributed import compute_split_shapes

from makani.utils import comm
from makani.distributed.mappings import gather_from_parallel_region, copy_to_parallel_region


class DistributedPatchEmbed(nn.Module):
    r"""
    Patch embedding with the spatial grid sharded across ranks.

    Distributed counterpart of
    :class:`~makani.models.common.patch.PatchEmbed2D`. The image is sharded
    along its width across the spatial group and each rank embeds the slice it
    owns. Because patch embedding is a strided convolution whose stride equals
    its kernel, patches never straddle a shard boundary, so no halo exchange is
    needed. The projection weights are replicated across the spatial group and
    their gradients reduced over it.

    Parameters
    ----------
    img_size : (int, int), optional
        Global image size, by default ``(224, 224)``. The patched width must be
        divisible by the spatial group size.
    patch_size : (int, int), optional
        Patch size, by default ``(16, 16)``.
    in_chans : int, optional
        Number of input channels, by default ``3``.
    embed_dim : int, optional
        Global dimension of each output token, by default ``768``. Sharded over
        the matmul group when the output is matmul-parallel, in which case it
        must be divisible by that group's size.
    input_is_matmul_parallel : bool, optional
        Whether the input arrives sharded along channels over the matmul group,
        in which case it is gathered first. By default ``False``.
    output_is_matmul_parallel : bool, optional
        Whether to leave the output sharded along the embedding dimension, by
        default ``False``.

    Raises
    ------
    ValueError
        If the patched width is not divisible by the spatial group size, or if
        ``embed_dim`` is not divisible by the matmul group size when the output
        is matmul-parallel.

    See Also
    --------
    makani.models.common.patch.PatchEmbed2D : the single-rank version.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=True,
    ):
        super().__init__()

        # store params
        self.input_parallel = input_is_matmul_parallel
        self.output_parallel = output_is_matmul_parallel

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        spatial_comm_size = comm.get_size("spatial")

        # compute parameters
        if (img_size[1] // patch_size[1]) % spatial_comm_size != 0:
            raise ValueError(
                f"the spatial comm size ({spatial_comm_size}) must evenly divide patched W ({img_size[1] // patch_size[1]})"
            )
        num_patches = ((img_size[1] // patch_size[1]) // spatial_comm_size) * (img_size[0] // patch_size[0])
        self.img_size = (img_size[0], img_size[1] // spatial_comm_size)
        self.patch_size = patch_size
        self.num_patches = num_patches

        # get effective embedding size:
        if self.output_parallel:
            if embed_dim % matmul_comm_size != 0:
                raise ValueError(
                    f"the embed_dim ({embed_dim}) needs to be divisible by matmul_parallel_size ({matmul_comm_size})"
                )
            out_chans_local = embed_dim // matmul_comm_size
        else:
            out_chans_local = embed_dim

        # the weights of this layer are shared across the spatial ranks
        self.proj = nn.Conv2d(in_chans, out_chans_local, kernel_size=patch_size, stride=patch_size)

        # shared over the spatial ranks, each embedding its part of the grid; split over matmul if the output is,
        # else every matmul rank computes the same
        if self.output_parallel:
            sharing, sharded = {"spatial": "sum"}, "matmul"
        else:
            sharing, sharded = {"spatial": "sum", "matmul": "avg"}, None
        self.proj.weight.is_shared_mp = dict(sharing)
        self.proj.weight.sharded_dims_mp = [sharded, None, None, None]
        self.proj.bias.is_shared_mp = dict(sharing)
        self.proj.bias.sharded_dims_mp = [sharded]

        # gather shapes
        self.gather_shapes = compute_split_shapes(in_chans, comm.get_size("matmul"))

    def forward(self, x):
        r"""
        Embed this rank's slice of the image into patch tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, in_chans, H, W_local)``, where ``W_local`` is
            this rank's share of the width. Gathered along channels first if the
            input is matmul-parallel.

        Returns
        -------
        torch.Tensor
            Tokens of shape ``(B, embed_dim_local, num_patches)``, where
            ``embed_dim_local`` is the full ``embed_dim`` unless the output is
            matmul-parallel.
        """
        if self.input_parallel:
            x = gather_from_parallel_region(x, 1, self.gather_shapes, "matmul")

        if self.output_parallel:
            x = copy_to_parallel_region(x, "matmul")

        B, C, H, W = x.shape
        torch._check(H == self.img_size[0], lambda: f"Input image height {H} doesn't match model {self.img_size[0]}.")
        torch._check(W == self.img_size[1], lambda: f"Input image width {W} doesn't match model {self.img_size[1]}.")
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x
