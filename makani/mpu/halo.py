# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Point-to-point exchange with the immediate neighbours in an "h"/"w" group.

Everything here assumes a halo never needs to reach past the immediate
neighbour -- callers are responsible for checking that (comparing the halo
a mapping actually needs against the neighbour's own block size) and raising
if it doesn't hold, since silently fetching from a second hop would make this
a different, more expensive algorithm than the one asked for.

``exchange_indexed`` is the one primitive: given, for each side, which of
this rank's local entries (along the last dimension) a neighbour needs and
how many entries this rank is due to receive back, it moves exactly that and
nothing else, via a single batched P2P round, and is differentiable. A side
is skipped by passing ``None`` for its rank -- that is how a non-periodic
group's boundary rank (no wraparound, e.g. the poles in the "h" direction)
declines a neighbour it does not have, the same way
``torch_harmonics.distributed.primitives._PolarHaloExchangeFn`` guards its
own boundary ranks.

The output/gradient layout is ``[from_prev, local, from_next]`` -- halo
before local data, matching the convention ``_PolarHaloExchangeFn`` already
uses (``[recv_top, x, recv_bot]``), which is what makes ``azimuthal_halo_exchange``
below a drop-in longitude counterpart to ``torch_harmonics``'s (polar-only,
non-periodic) ``polar_halo_exchange``.

Two things every P2P round here has to get right:

* No message tags. ``prev_rank == next_rank`` whenever a group has exactly
  two members (the periodic "w" case; "h" never hits this, since its
  boundary ranks decline the side that would collide) -- with only one
  physical peer, a single call needs to send/receive *two* logically
  distinct payloads ("toward prev" and "toward next") to/from that one rank.
  An earlier version of this module disambiguated the two with
  ``dist.P2POp``'s ``tag=``, which works on gloo but not for real: per
  ``torch.distributed``'s own docs, "tag is not supported with the NCCL
  backend" -- on a real (NCCL) cluster the two untagged sends/recvs to the
  same peer get matched by call order alone, which silently swaps prev/next
  data. The fix used throughout this module is to never rely on tags at
  all: when ``prev_rank == next_rank``, the two payloads are concatenated
  (fixed canonical order ``[to_prev, to_next]``, agreed on by construction
  since every rank runs the same code) into a *single* send/recv pair with
  that one peer, then split back apart on arrival. When the two ranks are
  genuinely distinct (any group with more than two members), no ambiguity
  exists in the first place -- one message per peer, ordinary point-to-point.
* Every batch here posts its irecv ops before its isend ops. gloo's isend
  can block until a matching irecv is posted on the peer: if both ranks in a
  pair issue send-before-recv in the same order, each blocks on its own
  first send waiting for a recv that only gets posted after that send
  returns -- a real, timing-dependent deadlock (not every run hits it, which
  makes it easy to miss locally). Posting every recv first removes the
  ordering dependency entirely.
"""

import torch
import torch.distributed as dist

from torch_harmonics.distributed.primitives import get_group_neighbors

# bridge so new-style autograd.Function (separate setup_context) works with
# torch.amp.custom_fwd/custom_bwd; see makani/mpu/_amp_utils.py (pytorch#132388).
from makani.mpu._amp_utils import _custom_setup_context


def _p2p_exchange(send_prev, send_next, prev_rank, next_rank, recv_count_prev, recv_count_next, group):
    """Exchange payloads with the immediate prev/next neighbours, one P2P round, no tags.

    ``send_prev``/``send_next`` are ``None`` exactly when ``prev_rank``/
    ``next_rank`` are ``None`` (a declined side). See the module docstring
    for why ``prev_rank == next_rank`` (the periodic "w" group's two-member
    case) is handled as a single combined message instead of two concurrent
    ones. Returns ``(recv_prev, recv_next)``, each ``None`` if the
    corresponding rank is ``None``.
    """
    if (prev_rank is None) and (next_rank is None):
        return None, None

    if (prev_rank is not None) and (prev_rank == next_rank):
        peer = prev_rank
        send_buf = torch.cat([send_prev, send_next], dim=-1).contiguous()
        recv_buf = torch.zeros(*send_prev.shape[:-1], recv_count_prev + recv_count_next, dtype=send_prev.dtype, device=send_prev.device)
        reqs = dist.batch_isend_irecv(
            [dist.P2POp(dist.irecv, recv_buf, peer, group), dist.P2POp(dist.isend, send_buf, peer, group)]
        )
        for req in reqs:
            req.wait()
        # the peer runs the same code, so its own send_buf is also laid out
        # [to_its_prev, to_its_next] -- and since it is both my prev and my
        # next, "to_its_prev" (size = its own send-to-prev count, which is
        # exactly what I already know as recv_count_next) is what fills my
        # recv_next slot, and "to_its_next" fills my recv_prev slot
        recv_next = recv_buf[..., :recv_count_next]
        recv_prev = recv_buf[..., recv_count_next:]
        return recv_prev, recv_next

    recv_ops, send_ops = [], []
    recv_prev = recv_next = None
    if prev_rank is not None:
        recv_prev = torch.zeros(*send_prev.shape[:-1], recv_count_prev, dtype=send_prev.dtype, device=send_prev.device)
        recv_ops.append(dist.P2POp(dist.irecv, recv_prev, prev_rank, group))
        send_ops.append(dist.P2POp(dist.isend, send_prev.contiguous(), prev_rank, group))
    if next_rank is not None:
        recv_next = torch.zeros(*send_next.shape[:-1], recv_count_next, dtype=send_next.dtype, device=send_next.device)
        recv_ops.append(dist.P2POp(dist.irecv, recv_next, next_rank, group))
        send_ops.append(dist.P2POp(dist.isend, send_next.contiguous(), next_rank, group))

    reqs = dist.batch_isend_irecv(recv_ops + send_ops)
    for req in reqs:
        req.wait()
    return recv_prev, recv_next


def owner_rank(global_index: torch.Tensor, splits) -> torch.Tensor:
    """Which rank's contiguous block (as sized by ``splits``, from ``compute_split_shapes``) each global index falls in.

    ``global_index`` is a torch tensor (this feeds directly into the P2P
    index computation in ``_redistribute_mesh_cells``, so staying in torch
    here avoids a numpy round-trip for no reason); ``splits`` is the plain
    list ``compute_split_shapes`` returns.
    """
    boundaries = torch.cumsum(
        torch.as_tensor(splits, dtype=global_index.dtype, device=global_index.device), dim=0
    )[:-1]
    return torch.searchsorted(boundaries, global_index, right=True)


def exchange_counts(send_count_prev, send_count_next, prev_rank, next_rank, group, device):
    """Tiny metadata round-trip: tell each neighbour how many entries are coming, learn how many to expect back.

    Not differentiable -- this only ever carries plain counts. ``device``
    follows the same convention as the rest of the distributed stack (e.g.
    ``makani.mpu.helpers.gather_uneven``'s size tensor): derived from a real
    tensor already in hand -- there is no backend-inspection fallback here,
    so callers with no tensor yet (there are none left; mesh redistribution
    now runs lazily on first forward, once ``data`` exists) would need one.
    """
    if (prev_rank is None) and (next_rank is None):
        return 0, 0
    send_prev = torch.as_tensor([send_count_prev], dtype=torch.int64, device=device) if prev_rank is not None else None
    send_next = torch.as_tensor([send_count_next], dtype=torch.int64, device=device) if next_rank is not None else None
    recv_prev, recv_next = _p2p_exchange(send_prev, send_next, prev_rank, next_rank, 1, 1, group)
    recv_prev = int(recv_prev.item()) if recv_prev is not None else 0
    recv_next = int(recv_next.item()) if recv_next is not None else 0
    return recv_prev, recv_next


def neighbor_ranks(group):
    """``(prev_rank, next_rank)`` global ranks of the immediate neighbours in ``group``.

    ``(None, None)`` if ``group`` is ``None`` or has a single member -- nothing to
    exchange with.
    """
    if (group is None) or (dist.get_world_size(group) <= 1):
        return None, None
    return get_group_neighbors(group)


class _ExchangeIndexedFn(torch.autograd.Function):
    """Differentiable P2P exchange of selected entries along the last dimension."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(x, prev_rank, next_rank, send_index_prev, send_index_next, recv_count_prev, recv_count_next, group):
        if (prev_rank is None) and (next_rank is None):
            return x

        send_prev = x.index_select(-1, send_index_prev) if prev_rank is not None else None
        send_next = x.index_select(-1, send_index_next) if next_rank is not None else None
        recv_prev, recv_next = _p2p_exchange(send_prev, send_next, prev_rank, next_rank, recv_count_prev, recv_count_next, group)

        pieces = [piece for piece in (recv_prev, x, recv_next) if piece is not None]
        return torch.cat(pieces, dim=-1).contiguous()

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        x, prev_rank, next_rank, send_index_prev, send_index_next, recv_count_prev, recv_count_next, group = inputs
        ctx.N = x.shape[-1]
        ctx.prev_rank = prev_rank
        ctx.next_rank = next_rank
        ctx.send_index_prev = send_index_prev
        ctx.send_index_next = send_index_next
        ctx.recv_count_prev = recv_count_prev
        ctx.recv_count_next = recv_count_next
        ctx.group = group

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        if (ctx.prev_rank is None) and (ctx.next_rank is None):
            return dout, None, None, None, None, None, None, None

        prev_rank, next_rank, group = ctx.prev_rank, ctx.next_rank, ctx.group

        offset = 0
        grad_to_prev = None
        if prev_rank is not None:
            grad_to_prev = dout[..., offset : offset + ctx.recv_count_prev].contiguous()
            offset += ctx.recv_count_prev
        dx = dout[..., offset : offset + ctx.N].contiguous().clone()
        offset += ctx.N
        grad_to_next = None
        if next_rank is not None:
            grad_to_next = dout[..., offset : offset + ctx.recv_count_next].contiguous()

        recv_from_prev, recv_from_next = _p2p_exchange(
            grad_to_prev, grad_to_next, prev_rank, next_rank, ctx.send_index_prev.numel() if prev_rank is not None else 0,
            ctx.send_index_next.numel() if next_rank is not None else 0, group
        )

        # gradient owed back for the local entries this rank sent away, from
        # whichever neighbour(s) received them -- index_add_ since a stage can
        # run more than once (mesh's h-stage then w-stage) and a given local
        # entry could in principle be selected by both a prev- and next-side send
        if recv_from_prev is not None:
            dx.index_add_(-1, ctx.send_index_prev, recv_from_prev)
        if recv_from_next is not None:
            dx.index_add_(-1, ctx.send_index_next, recv_from_next)

        return dx, None, None, None, None, None, None, None


@torch.compiler.disable()
def exchange_indexed(x, prev_rank, next_rank, send_index_prev, send_index_next, recv_count_prev, recv_count_next, group):
    """Exchange selected entries of ``x`` (last dim) with immediate neighbours.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(..., N)``.
    prev_rank, next_rank : int or None
        Global rank of each neighbour, or ``None`` to decline that side (e.g. a
        non-periodic group's boundary rank).
    send_index_prev, send_index_next : torch.LongTensor or None
        Indices into ``x``'s last dimension to send to each neighbour. Required
        (non-``None``) exactly when the corresponding rank is not ``None``.
    recv_count_prev, recv_count_next : int
        How many entries this rank is due to receive from each neighbour.
    group : torch.distributed.ProcessGroup

    Returns
    -------
    torch.Tensor
        Shape ``(..., recv_count_prev + N + recv_count_next)``, in that order.
    """
    if (prev_rank is None) and (next_rank is None):
        return x
    return _ExchangeIndexedFn.apply(
        x, prev_rank, next_rank, send_index_prev, send_index_next, recv_count_prev, recv_count_next, group
    )


class _RedistributeFn(torch.autograd.Function):
    """Differentiable P2P redistribution: keep some entries, hand the rest to their owner.

    Unlike ``_ExchangeIndexedFn``, entries that are sent away are dropped from
    the local output rather than kept -- ownership moves, it isn't copied.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(x, keep_index, prev_rank, next_rank, send_index_prev, send_index_next, recv_count_prev, recv_count_next, group):
        kept = x.index_select(-1, keep_index)
        if (prev_rank is None) and (next_rank is None):
            return kept

        send_prev = x.index_select(-1, send_index_prev) if prev_rank is not None else None
        send_next = x.index_select(-1, send_index_next) if next_rank is not None else None
        recv_prev, recv_next = _p2p_exchange(send_prev, send_next, prev_rank, next_rank, recv_count_prev, recv_count_next, group)

        pieces = [kept] + [piece for piece in (recv_prev, recv_next) if piece is not None]
        return torch.cat(pieces, dim=-1).contiguous()

    @staticmethod
    @_custom_setup_context(device_type="cuda")
    def setup_context(ctx, inputs, output):
        x, keep_index, prev_rank, next_rank, send_index_prev, send_index_next, recv_count_prev, recv_count_next, group = (
            inputs
        )
        ctx.N = x.shape[-1]
        ctx.leading = x.shape[:-1]
        ctx.dtype = x.dtype
        ctx.device = x.device
        ctx.keep_index = keep_index
        ctx.prev_rank = prev_rank
        ctx.next_rank = next_rank
        ctx.send_index_prev = send_index_prev
        ctx.send_index_next = send_index_next
        ctx.recv_count_prev = recv_count_prev
        ctx.recv_count_next = recv_count_next
        ctx.group = group

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        n_keep = ctx.keep_index.numel()
        dx = torch.zeros(*ctx.leading, ctx.N, dtype=ctx.dtype, device=ctx.device)
        dx.index_copy_(-1, ctx.keep_index, dout[..., :n_keep].contiguous())

        if (ctx.prev_rank is None) and (ctx.next_rank is None):
            return dx, None, None, None, None, None, None, None, None

        prev_rank, next_rank, group = ctx.prev_rank, ctx.next_rank, ctx.group
        offset = n_keep
        grad_to_prev = None
        if prev_rank is not None:
            grad_to_prev = dout[..., offset : offset + ctx.recv_count_prev].contiguous()
            offset += ctx.recv_count_prev
        grad_to_next = None
        if next_rank is not None:
            grad_to_next = dout[..., offset : offset + ctx.recv_count_next].contiguous()

        recv_from_prev, recv_from_next = _p2p_exchange(
            grad_to_prev, grad_to_next, prev_rank, next_rank, ctx.send_index_prev.numel() if prev_rank is not None else 0,
            ctx.send_index_next.numel() if next_rank is not None else 0, group
        )

        if recv_from_prev is not None:
            dx.index_add_(-1, ctx.send_index_prev, recv_from_prev)
        if recv_from_next is not None:
            dx.index_add_(-1, ctx.send_index_next, recv_from_next)

        return dx, None, None, None, None, None, None, None, None


@torch.compiler.disable()
def redistribute_indexed(x, keep_index, prev_rank, next_rank, send_index_prev, send_index_next, recv_count_prev, recv_count_next, group):
    """Keep ``x[..., keep_index]``, hand the rest to their owning neighbour, receive what neighbours hand over.

    Unlike :func:`exchange_indexed` (which duplicates boundary data as a halo),
    entries that are sent away are dropped from the local output -- ownership
    moves, it doesn't get copied. Output order is ``[kept, from_prev, from_next]``.
    """
    return _RedistributeFn.apply(
        x, keep_index, prev_rank, next_rank, send_index_prev, send_index_next, recv_count_prev, recv_count_next, group
    )


def azimuthal_halo_exchange(x, r_lon, group):
    """Exchange ``r_lon`` boundary columns with neighbouring "w" ranks, periodic.

    The longitude counterpart of ``torch_harmonics.distributed.primitives.polar_halo_exchange``
    -- same halo-padding contract (``x`` of shape ``(..., W)`` in, ``(..., W + 2*r_lon)``
    out, differentiable), but periodic: longitude wraps, so unlike the polar
    direction there is no boundary rank that declines a side.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(..., W)``.
    r_lon : int
        Number of halo columns to exchange on each side. Must not exceed this
        rank's own local ``W`` -- callers are expected to have already checked
        the halo fits within the immediate neighbour (see module docstring).
    group : torch.distributed.ProcessGroup or None
    """
    if r_lon == 0:
        return x
    prev_rank, next_rank = neighbor_ranks(group)
    if prev_rank is None:
        return x
    W = x.shape[-1]
    send_index_prev = torch.arange(r_lon, device=x.device)
    send_index_next = torch.arange(W - r_lon, W, device=x.device)
    return exchange_indexed(x, prev_rank, next_rank, send_index_prev, send_index_next, r_lon, r_lon, group)
