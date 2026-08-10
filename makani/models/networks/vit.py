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


import torch.nn.functional as F
import torch
import torch.nn as nn

# mp stuff
from makani.utils import comm
from makani.models.common import DropPath, MLP, PatchEmbed2D
from makani.mpu.layers import DistributedMLP, DistributedAttention


class Attention(nn.Module):
    r"""
    Standard multi-head self-attention.

    Projects the input to queries, keys and values, splits them into
    ``num_heads`` independent subspaces, and applies scaled dot-product
    attention:

    .. math::

        \mathrm{Attention}(Q, K, V) =
        \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V

    Dispatches to :func:`torch.nn.functional.scaled_dot_product_attention`, so
    it uses a fused kernel where one is available.

    Parameters
    ----------
    dim : int
        Feature dimension. Must be divisible by ``num_heads``.
    input_format : str, optional
        Input layout, by default ``"traditional"`` (channels last).
    num_heads : int, optional
        Number of attention heads, by default ``8``.
    qkv_bias : bool, optional
        Whether the QKV projection carries a bias, by default ``False``.
    qk_norm : bool, optional
        Normalize queries and keys before the attention product, by default
        ``False``. Bounds the logits and stabilizes training at large widths.
    attn_drop_rate : float, optional
        Dropout probability on the attention weights, by default ``0.0``.
    proj_drop_rate : float, optional
        Dropout probability after the output projection, by default ``0.0``.
    norm_layer : callable, optional
        Normalization used for ``qk_norm``, by default :class:`torch.nn.LayerNorm`.

    Raises
    ------
    ValueError
        If ``dim`` is not divisible by ``num_heads``.

    See Also
    --------
    makani.mpu.layers.DistributedAttention : head-parallel version.
    """

    def __init__(
        self,
        dim,
        input_format="traditional",
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(dim, dim)

        if proj_drop_rate > 0:
            self.proj_drop = nn.Dropout(proj_drop_rate)
        else:
            self.proj_drop = nn.Identity()

    def forward(self, x):
        r"""
        Apply multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Token tensor of shape ``(B, N, dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, N, dim)``.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    r"""
    Standard pre-norm transformer block.

    Attention followed by a feed-forward MLP, each preceded by a normalization
    and wrapped in a residual connection:

    .. code-block:: text

        x = x + drop_path(attn(norm1(x)))
        x = norm2(x)
        x = x + drop_path(mlp(x))

    Both sublayers automatically switch to their tensor-parallel
    implementations when the ``comm_name`` group has more than one rank, so the
    same block definition serves single-GPU and model-parallel runs.

    Parameters
    ----------
    dim : int
        Feature dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float, optional
        Hidden width of the MLP as a multiple of ``dim``, by default ``4.0``.
    qkv_bias : bool, optional
        Whether the QKV projection carries a bias, by default ``False``.
    mlp_drop_rate : float, optional
        Dropout probability in the MLP and after the attention projection, by
        default ``0.0``.
    attn_drop_rate : float, optional
        Dropout probability on the attention weights, by default ``0.0``.
    path_drop_rate : float, optional
        Stochastic depth probability for both residual branches, by default ``0.0``.
    act_layer : callable, optional
        Activation constructor for the MLP, by default :class:`torch.nn.GELU`.
    norm_layer : callable, optional
        Normalization constructor, by default :class:`torch.nn.LayerNorm`.
    comm_name : str, optional
        Communicator group used when model parallel, by default ``"matmul"``.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        mlp_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        comm_name="matmul",
    ):
        super().__init__()

        if comm.get_size(comm_name) > 1:
            self.attn = DistributedAttention(
                dim,
                input_format="traditional",
                comm_name=comm_name,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=mlp_drop_rate,
                norm_layer=norm_layer,
            )
        else:
            self.attn = Attention(
                dim,
                input_format="traditional",
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=mlp_drop_rate,
                norm_layer=norm_layer,
            )
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        # distribute MLP for model parallelism
        if comm.get_size(comm_name) > 1:
            self.mlp = DistributedMLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                out_features=dim,
                act_layer=act_layer,
                drop_rate=mlp_drop_rate,
                input_format="traditional",
                comm_name=comm_name,
            )
        else:
            self.mlp = MLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                out_features=dim,
                act_layer=act_layer,
                drop_rate=mlp_drop_rate,
                input_format="traditional",
            )

    def forward(self, x):
        r"""
        Apply attention and MLP with residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Token tensor of shape ``(B, N, dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, N, dim)``.
        """
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))
        return x


class VisionTransformer(nn.Module):
    r"""
    Vision transformer baseline for gridded forecasting.

    Embeds the input field into patch tokens, adds a learned position
    embedding, runs a stack of :class:`Block` layers, and decodes each token
    back into its patch of the output field. Attention is global over patches,
    so every location can influence every other in a single layer -- but at
    ``O(N^2)`` cost in the number of patches, which is why the input is
    tokenized rather than processed at full resolution.

    Included as a baseline against the spectral operators; unlike them it has no
    notion of the sphere's geometry.

    Parameters
    ----------
    inp_shape : list of int, optional
        Input grid as ``[nlat, nlon]``, by default ``[72, 144]``.
    patch_size : (int, int), optional
        Patch size, by default ``(6, 6)``.
    inp_chans : int, optional
        Number of input channels, by default ``3``.
    out_chans : int, optional
        Number of output channels, by default ``3``.
    embed_dim : int, optional
        Token dimension, by default ``768``.
    depth : int, optional
        Number of transformer blocks, by default ``12``.
    num_heads : int, optional
        Number of attention heads per block, by default ``12``.
    mlp_ratio : float, optional
        Hidden width of the block MLPs as a multiple of ``embed_dim``, by
        default ``4.0``.
    qkv_bias : bool, optional
        Whether the QKV projections carry a bias, by default ``True``.
    mlp_drop_rate : float, optional
        Dropout probability in the MLPs, by default ``0.0``.
    attn_drop_rate : float, optional
        Dropout probability on the attention weights, by default ``0.0``.
    path_drop_rate : float, optional
        Maximum stochastic depth probability, by default ``0.0``. Ramped
        linearly from zero at the first block to this value at the last, the
        usual schedule: early layers are kept intact while deeper ones are
        dropped more aggressively.
    norm_layer : str, optional
        Normalization type; only ``"layer_norm"`` is supported.
    comm_name : str, optional
        Communicator group used when model parallel, by default ``"matmul"``.
    **kwargs
        Ignored; present so model configs can pass extra keys.

    Raises
    ------
    NotImplementedError
        If ``norm_layer`` is not ``"layer_norm"``.
    """

    def __init__(
        self,
        inp_shape=[72, 144],
        patch_size=(6, 6),
        inp_chans=3,
        out_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        mlp_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        norm_layer="layer_norm",
        comm_name="matmul",
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = inp_shape
        self.out_ch = out_chans
        self.comm_name = comm_name

        self.patch_embed = PatchEmbed2D(
            img_size=self.img_size, patch_size=patch_size, in_chans=inp_chans, embed_dim=self.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # annotate for distributed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_embed.is_shared_mp = []

        self.pos_drop = nn.Dropout(p=path_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, depth)]  # stochastic depth decay rule

        if norm_layer == "layer_norm":
            norm_layer_handle = nn.LayerNorm
        else:
            raise NotImplementedError(f"Error, normalization layer type {norm_layer} not implemented for ViT.")

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    mlp_drop_rate=mlp_drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    path_drop_rate=dpr[i],
                    norm_layer=norm_layer_handle,
                    comm_name=comm_name,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer_handle(embed_dim)

        self.out_size = self.out_ch * self.patch_size[0] * self.patch_size[1]

        self.head = nn.Linear(embed_dim, self.out_size, bias=False)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        r"""
        Embed the input field into position-encoded tokens.

        Exposed separately from ``forward`` so callers can obtain the token
        sequence -- for probing or feature extraction -- without running the
        blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input field of shape ``(B, inp_chans, nlat, nlon)``.

        Returns
        -------
        torch.Tensor
            Tokens of shape ``(B, num_patches, embed_dim)``, with the position
            embedding added and dropout applied.
        """
        B, C, H, W = x.shape
        x = self.patch_embed(x).transpose(1, 2)  # patch linear embedding

        # add positional encoding to each token
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_head(self, x):
        r"""
        Decode tokens back into a full-resolution output field.

        Each token is projected to ``out_chans * patch_h * patch_w`` values and
        those are unfolded into its patch, reassembling the grid.

        Parameters
        ----------
        x : torch.Tensor
            Tokens of shape ``(B, num_patches, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Field of shape ``(B, out_chans, nlat, nlon)``.
        """
        B, _, _ = x.shape  # B x N x embed_dim
        x = x.reshape(B, self.patch_embed.red_img_size[0], self.patch_embed.red_img_size[1], self.embed_dim)
        B, h, w, _ = x.shape

        # apply head
        x = self.head(x)
        x = x.reshape(shape=(B, h, w, self.patch_size[0], self.patch_size[1], self.out_ch))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, self.out_ch, self.img_size[0], self.img_size[1]))

        return x

    def forward(self, x):
        r"""
        Map an input field to the predicted output field.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, inp_chans, nlat, nlon)``.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(B, out_chans, nlat, nlon)``.
        """
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.forward_head(x)
        return x
