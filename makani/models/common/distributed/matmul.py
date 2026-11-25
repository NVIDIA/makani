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

"""The tensor-parallel matmul."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from makani.distributed import comm
from makani.distributed.mappings import reduce_from_parallel_region, copy_to_parallel_region

from makani.models.common.pointwise_conv import pointwise_conv2d, make_fp8_gemm_if, drop_fp8_state, fp8_pointwise


class DistributedMatmul(nn.Module):
    r"""Megatron-style tensor-parallel matmul over a single feature-parallel group.

    The legacy 2D (fin x fout) decomposition has been retired in favor of a 1D
    column/row fork-join over a single comm group (``comm_name``, "matmul" by
    default):

    - ``parallel_mode="column"`` shards the OUTPUT dimension. The input is
      replicated across the group (``copy_to_parallel_region``: identity in the
      forward, all-reduce in the backward) and there is NO output reduction, so
      the result is sharded along the output dimension.
    - ``parallel_mode="row"`` shards the INPUT dimension. The input is assumed to
      be sharded (e.g. the sharded hidden produced by a preceding column layer),
      and the partial outputs are all-reduced (``reduce_from_parallel_region``)
      into a replicated full output. For a row layer the bias is replicated and
      added AFTER the reduction.

    Pairing a column layer with a row layer is what makes this cheap: the
    intermediate stays sharded and only one all-reduce is needed for the pair,
    rather than one per layer.

    Parameters
    ----------
    inp_dim : int
        Global input feature dimension. For ``"row"`` mode this is sharded
        across the group, so it must be divisible by the group size.
    out_dim : int
        Global output feature dimension. For ``"column"`` mode this is sharded
        across the group, so it must be divisible by the group size.
    input_format : str, optional
        ``"nchw"`` (default) for ``(B, C, H, W)`` inputs, or ``"traditional"``
        for channels-last inputs.
    comm_name : str, optional
        Name of the communicator group to shard over, by default ``"matmul"``.
    parallel_mode : str, optional
        ``"column"`` (default) or ``"row"``; see above.
    bias : bool, optional
        Whether to add a bias, by default ``True``.
    fp8 : bool, optional
        Give the local GEMM of the ``"nchw"`` format an FP8 engine of transformer_engine on NCHW, without permutes,
        see :class:`~makani.models.common.pointwise_conv.PointwiseConv2d`. ``None`` (default) follows
        :func:`~makani.models.common.pointwise_conv.fp8_layers`, as the model is built for an FP8 ``amp_mode``.

    Raises
    ------
    ValueError
        If ``parallel_mode`` is neither ``"column"`` nor ``"row"``, or if the
        sharded dimension is not divisible by the group size.
    """

    def __init__(
        self,
        inp_dim,
        out_dim,
        input_format="nchw",
        comm_name="matmul",
        parallel_mode="column",
        bias=True,
        fp8=None,
    ):
        super(DistributedMatmul, self).__init__()

        if parallel_mode not in ["column", "row"]:
            raise ValueError(f"Error, parallel_mode {parallel_mode} not supported (use 'column' or 'row').")

        if input_format not in ["nchw", "traditional"]:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        self.comm_name = comm_name
        self.parallel_mode = parallel_mode
        self.input_format = input_format

        comm_size = comm.get_size(comm_name)

        # column shards the output dim, row shards the input dim
        if parallel_mode == "column":
            if out_dim % comm_size != 0:
                raise ValueError(
                    f"the output feature dim ({out_dim}) has to be evenly divisible by the matmul comm dim ({comm_size})"
                )
            out_dim_local = out_dim // comm_size
            inp_dim_local = inp_dim
        else:
            if inp_dim % comm_size != 0:
                raise ValueError(
                    f"the input feature dim ({inp_dim}) has to be evenly divisible by the matmul comm dim ({comm_size})"
                )
            out_dim_local = out_dim
            inp_dim_local = inp_dim // comm_size

        # the dimension that is sharded over the matmul group (None for the
        # replicated dimension) - used for checkpoint gather/scatter only
        weight_inp_shard = None if parallel_mode == "column" else comm_name
        weight_out_shard = comm_name if parallel_mode == "column" else None

        # weight
        if input_format == "nchw":
            self.weight = nn.Parameter(torch.ones(out_dim_local, inp_dim_local, 1, 1))
            self.weight.is_shared_mp = {"spatial": "sum"}
            self.weight.sharded_dims_mp = [weight_out_shard, weight_inp_shard, None, None]
            # a 1x1 convolution as a matmul, see pointwise_conv2d for the gradient layout
            self.matmul_handle = pointwise_conv2d
        else:  # traditional
            self.weight = nn.Parameter(torch.ones(out_dim_local, inp_dim_local))
            self.weight.is_shared_mp = {"spatial": "sum"}
            self.weight.sharded_dims_mp = [weight_out_shard, weight_inp_shard]
            self.matmul_handle = F.linear

        # bias
        self.bias = None
        if bias:
            # a column bias is sharded over the (sharded) output dim, a row bias
            # is replicated over the matmul group: it is added after the all-reduce,
            # which every matmul rank computes alike, so its gradient is averaged
            bias_shard = comm_name if parallel_mode == "column" else None
            # 1-D in either format, as nn.Conv2d and nn.Linear keep it, so that a checkpoint is the same as that
            # of the serial layer
            # a row bias is added after the reduction, which every matmul rank computes alike
            bias_sharing = {"spatial": "sum"} if parallel_mode == "column" else {"spatial": "sum", "matmul": "avg"}
            self.bias = nn.Parameter(torch.zeros(out_dim_local))
            self.bias.is_shared_mp = bias_sharing
            self.bias.sharded_dims_mp = [bias_shard]

        # FP8 GEMMs on NCHW for the local shard, see PointwiseConv2d
        self.fp8_gemm = make_fp8_gemm_if(fp8, inp_dim_local, out_dim_local) if input_format == "nchw" else None

    @staticmethod
    def upgrade_state_dict(state_dict, prefix=""):
        r"""
        Upgrade a state dict of an earlier version in place: the bias was kept as ``(1, C, 1, 1)`` in the nchw format.

        Parameters
        ----------
        state_dict : dict
            A state dict holding the layer.
        prefix : str, optional
            The prefix of the keys of the layer.
        """
        key = prefix + "bias"
        if (key in state_dict) and (state_dict[key] is not None) and (state_dict[key].dim() == 4):
            state_dict[key] = state_dict[key].reshape(-1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        drop_fp8_state(self, state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _local_matmul(self, x):
        if self.fp8_gemm is not None:
            y = fp8_pointwise(self.fp8_gemm, x, self.weight)
            if y is not None:
                return y
        return self.matmul_handle(x, self.weight, bias=None)

    def forward(self, x):
        r"""
        Apply the tensor-parallel matmul.

        Parameters
        ----------
        x : torch.Tensor
            For ``"column"`` mode, a replicated tensor with the full
            ``inp_dim``. For ``"row"`` mode, a tensor already sharded along the
            input dimension, typically the output of a preceding column layer.

        Returns
        -------
        torch.Tensor
            For ``"column"`` mode, sharded along the output dimension. For
            ``"row"`` mode, replicated with the full ``out_dim``.
        """
        if self.parallel_mode == "column":
            # replicated input -> sharded output, no output reduction
            x = copy_to_parallel_region(x, self.comm_name)
            x = self._local_matmul(x)
        else:
            # sharded input -> reduced (replicated) output
            x = self._local_matmul(x)
            x = reduce_from_parallel_region(x, self.comm_name)

        if self.bias is not None:
            x = x + (self.bias.reshape(1, -1, 1, 1) if self.input_format == "nchw" else self.bias)

        return x
