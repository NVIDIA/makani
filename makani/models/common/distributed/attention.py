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

"""Tensor-parallel multi-head attention."""

import torch.nn as nn
import torch.nn.functional as F

from makani.utils import comm
from makani.models.common.distributed.matmul import DistributedMatmul


class MultiHeadAttention(nn.Module):
    r"""
    Multi-head self-attention, with the heads split over the matmul ranks under model parallelism.

    Projects the tokens to queries, keys and values, normalizes queries and keys
    with ``qk_norm``, applies scaled dot-product attention per head, and
    projects the concatenated heads back:

    .. math::
        \mathrm{Attention}(Q, K, V) =
        \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V

    Heads are independent by construction, which makes them the natural unit
    to split: with a ``comm_name`` group of more than one rank, the query, key
    and value projections are column-parallel :class:`DistributedMatmul`
    layers, each rank computing its heads end to end, and the output projection
    is row-parallel, reducing the concatenated heads to a replicated result, so
    the layer costs a single all-reduce. Without, they are
    :class:`torch.nn.Linear` layers. Either way the parameters have the same
    names and full shapes, and every projection keeps its heads in order, so a
    checkpoint restores across any decomposition.

    Biases that cannot affect the result are left out. A bias of the keys adds
    :math:`q \cdot b_k` to every logit of a query, which the softmax removes, so
    the keys have no bias without ``qk_norm``; with it, the normalization of
    the keys has no bias, for the same reason, while the key projection has one
    with ``qkv_bias``, as it acts before the normalization.

    Parameters
    ----------
    dim : int
        Feature dimension. Must be divisible by ``num_heads``.
    num_heads : int, optional
        Number of attention heads, by default ``8``. Must be divisible by the
        size of the ``comm_name`` group.
    qkv_bias : bool, optional
        Whether the query, key and value projections carry biases, by default
        ``False``; see above for the keys.
    qk_norm : bool, optional
        Normalize queries and keys before the attention product, by default
        ``False``. Stabilizes training by bounding the logits.
    attn_drop_rate : float, optional
        Dropout probability on the attention weights in training, by default ``0.0``.
    proj_drop_rate : float, optional
        Dropout probability after the output projection, by default ``0.0``.
    norm_layer : callable, optional
        Normalization used for ``qk_norm``, called with the head dimension, by
        default :class:`torch.nn.LayerNorm`.
    comm_name : str, optional
        Group to split the heads over, by default ``"matmul"``.
    use_te : bool, optional
        Use TransformerEngine linears for the split projections, by default ``False``.

    Raises
    ------
    ValueError
        If ``dim`` is not divisible by ``num_heads``, or ``num_heads`` not by
        the size of the ``comm_name`` group.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        comm_name="matmul",
        use_te=False,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by num_heads {num_heads}")
        comm_size = comm.get_size(comm_name)
        if num_heads % comm_size != 0:
            raise ValueError(f"heads ({num_heads}) are not evenly split across model ranks ({comm_size})")

        self.num_heads = num_heads
        self.num_heads_local = num_heads // comm_size
        self.head_dim = dim // num_heads
        self.attn_drop_rate = attn_drop_rate

        def projection(bias, parallel_mode):
            if comm_size > 1:
                return DistributedMatmul(
                    dim, dim, "traditional", comm_name=comm_name, parallel_mode=parallel_mode, bias=bias, use_te=use_te
                )
            linear = nn.Linear(dim, dim, bias=bias)
            for param in linear.parameters():
                param.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
            return linear

        # the heads of this rank, from the queries, keys and values
        self.q = projection(qkv_bias, "column")
        self.k = projection(qkv_bias and qk_norm, "column")
        self.v = projection(qkv_bias, "column")

        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
            if getattr(self.k_norm, "bias", None) is not None:
                self.k_norm.register_parameter("bias", None)
            # the norms act on every head, and the heads are split over matmul, so each matmul rank holds the
            # contribution of its heads to their gradients; as for the other layers, the spatial ranks would each
            # hold that of their part of the grid
            for norm in (self.q_norm, self.k_norm):
                for param in norm.parameters():
                    param.is_shared_mp = {"spatial": "sum", "matmul": "sum"}
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # the heads back to the features, reduced over the ranks of the heads
        self.proj = projection(True, "row")
        self.proj_drop = nn.Dropout(proj_drop_rate) if proj_drop_rate > 0.0 else nn.Identity()

    def _heads(self, x, projection):
        # (B, N, dim) -> (B, heads, N, head_dim), the heads of this rank
        B, N, _ = x.shape
        return projection(x).reshape(B, N, self.num_heads_local, self.head_dim).transpose(1, 2)

    def forward(self, x):
        r"""
        Apply multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Token tensor of shape ``(B, N, dim)``, replicated over the ranks of
            the heads.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, N, dim)``, replicated as the input.
        """
        B, N, _ = x.shape

        q = self.q_norm(self._heads(x, self.q))
        k = self.k_norm(self._heads(x, self.k))
        v = self._heads(x, self.v)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate if self.training else 0.0)

        # the heads of this rank side by side, then reduced over the ranks by the row-parallel projection
        x = x.transpose(1, 2).reshape(B, N, self.num_heads_local * self.head_dim)
        return self.proj_drop(self.proj(x))
