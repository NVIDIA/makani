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

"""Tensor-parallel channel MLPs: the encoder/decoder, the MLP and the stochastic MLP."""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from makani.distributed import comm
from makani.models.common.mlp import upgrade_mlp_state_dict
from makani.models.common.distributed.matmul import DistributedMatmul
from makani.models.common.pointwise_conv import pointwise_conv2d

# transformer engine is an optional dependency: it is only used for the
# (optional) FP8/FP4 path and must not be required for import. availability is
# checked without importing it; the module is imported lazily where used.
from makani.utils.te_helpers import TE_AVAILABLE as _TE_AVAILABLE


# distributed encoder/decoder
class DistributedEncoderDecoder(nn.Module):
    r"""
    Tensor-parallel counterpart of
    :class:`~makani.models.common.mlp.EncoderDecoder`.

    A stack of :class:`DistributedMatmul` layers with activations in between,
    arranged so the chain consumes a replicated input and produces a replicated
    output. The parallel modes are assigned from the *end* backwards: the last
    layer is row-parallel (which reduces to a replicated output) and the modes
    alternate from there. This keeps intermediates sharded and costs one
    all-reduce per column/row pair rather than one per layer.

    Because the modes alternate and the last must be ``"row"``, the first layer
    is column-parallel only if the number of matmuls is even -- hence the
    requirement below.

    The layers are those of the serial
    :class:`~makani.models.common.mlp.EncoderDecoder`, with the same
    parameter names and shapes, so that a checkpoint of one restores into the
    other.

    Parameters
    ----------
    num_layers : int
        Number of hidden layers before the output layer, as in
        :class:`~makani.models.common.mlp.EncoderDecoder`, so ``num_layers + 1``
        matmuls. Must be odd when the communicator group is larger than one.
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output features.
    hidden_dim : int
        Width of the hidden layers.
    act_layer : callable
        Activation module constructor, called with no arguments.
    gain : float, optional
        Scales the variance of the output layer's initialization, by default ``1.0``.
    input_format : str, optional
        ``"nchw"`` (default) or ``"traditional"``.
    comm_name : str, optional
        Communicator group to shard over, by default ``"matmul"``.
    use_te : bool, optional
        Use TransformerEngine linears for the local GEMMs, by default ``False``.

    Raises
    ------
    ValueError
        If ``num_layers`` is even while matmul parallelism is active.
    """

    def __init__(
        self,
        num_layers,
        input_dim,
        output_dim,
        hidden_dim,
        act_layer,
        gain=1.0,
        input_format="nchw",
        comm_name="matmul",
        use_te=False,
    ):
        super(DistributedEncoderDecoder, self).__init__()

        self.comm_name = comm_name

        # num_layers hidden layers and the output layer, as in EncoderDecoder, so that both build the same network.
        # the chain takes a replicated input and must produce a replicated output. we therefore drive the
        # column/row fork-join from the END: the last layer is row-parallel (reduces to a replicated output) and
        # we alternate backwards. for the first layer to be column-parallel (consuming the replicated input) the
        # number of matmuls must be even when we are actually feature-parallel, i.e. num_layers odd.
        num_matmuls = num_layers + 1
        if (comm.get_size(comm_name) > 1) and (num_matmuls % 2 != 0):
            raise ValueError(
                f"DistributedEncoderDecoder needs an even number of matmuls under matmul parallelism, i.e. an odd number "
                f"of hidden layers, got {num_layers} hidden layers."
            )
        modes = ["row" if ((num_matmuls - 1 - i) % 2 == 0) else "column" for i in range(num_matmuls)]

        # get list of modules
        encoder_modules = []
        current_dim = input_dim
        for i in range(num_layers):
            encoder_modules.append(
                DistributedMatmul(
                    current_dim,
                    hidden_dim,
                    input_format=input_format,
                    comm_name=comm_name,
                    parallel_mode=modes[i],
                    bias=True,
                    use_te=use_te,
                )
            )

            # proper initialization
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
            if encoder_modules[-1].bias is not None:
                nn.init.constant_(encoder_modules[-1].bias, 0.0)

            encoder_modules.append(act_layer())
            current_dim = hidden_dim

        # final layer (row-parallel, replicated output, no bias)
        encoder_modules.append(
            DistributedMatmul(
                current_dim,
                output_dim,
                input_format=input_format,
                comm_name=comm_name,
                parallel_mode=modes[-1],
                bias=False,
                use_te=use_te,
            )
        )

        # proper initialization of final layer
        scale = math.sqrt(gain / current_dim)
        nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
        if encoder_modules[-1].bias is not None:
            nn.init.constant_(encoder_modules[-1].bias, 0.0)

        # create fwd sequence
        self.fwd = nn.Sequential(*encoder_modules)

    def forward(self, x):
        r"""
        Apply the tensor-parallel layer stack.

        Parameters
        ----------
        x : torch.Tensor
            Replicated input with ``input_dim`` features, in the layout implied
            by ``input_format``.

        Returns
        -------
        torch.Tensor
            Replicated output with ``output_dim`` features.
        """
        return self.fwd(x)


class DistributedMLP(nn.Module):
    r"""
    Tensor-parallel counterpart of :class:`~makani.models.common.mlp.MLP`.

    Two-layer feed-forward block in which the hidden dimension is sharded across
    the communicator group: the first matmul is column-parallel and the second
    row-parallel, so the wide intermediate never has to be materialized in full
    on any rank and the pair needs only a single all-reduce.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int, optional
        Width of the sharded hidden layer, defaults to ``in_features``.
    out_features : int, optional
        Number of output features, defaults to ``in_features``.
    output_bias : bool, optional
        Whether the output projection carries a bias, by default ``True``.
    input_format : str, optional
        ``"nchw"`` (default) or ``"traditional"``.
    comm_name : str, optional
        Communicator group to shard the hidden dimension over, by default
        ``"matmul"``.
    act_layer : callable, optional
        Activation constructor, by default :class:`torch.nn.GELU`.
    drop_rate : float, optional
        Dropout probability, by default ``0.0``.
    drop_type : str, optional
        ``"iid"`` (default) or ``"features"``. ``"features"`` requires
        ``input_format="nchw"``.
    checkpointing : bool, optional
        Recompute the block in the backward pass instead of storing
        activations, by default ``False``.
    gain : float, optional
        Scales the variance of the output layer's initialization, by default ``1.0``.
    use_te : bool, optional
        Use TransformerEngine linears for the GEMMs, by default ``False``. On
        this path the matmuls are built channels-last and the block permutes
        once at its boundaries rather than around every GEMM.

    Raises
    ------
    NotImplementedError
        If ``drop_type`` is unsupported, or if ``"traditional"`` is combined
        with feature dropout.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        output_bias=True,
        input_format="nchw",
        comm_name="matmul",
        act_layer=nn.GELU,
        drop_rate=0.0,
        drop_type="iid",
        checkpointing=False,
        gain=1.0,
        use_te=False,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # sanity checks:
        if (input_format == "traditional") and (drop_type == "features"):
            raise NotImplementedError(
                "Error, traditional input format and feature dropout cannot be selected simultaneously"
            )

        # TE GEMMs operate on the last (channel) dim. Rather than permuting
        # nchw<->nhwc around *each* matmul, when the TE path is active we build the
        # matmuls in "traditional" (channels-last) mode and permute once at the MLP
        # boundaries (see fwd) -- two permutes for the whole MLP instead of one pair
        # per matmul. Feature dropout (Dropout2d) drops channels at dim=1 and so
        # needs the nchw layout, so keep the standard nchw path in that case.
        self.channels_last = use_te and _TE_AVAILABLE and (input_format == "nchw") and (drop_type != "features")
        matmul_format = "traditional" if self.channels_last else input_format

        # column-parallel: replicated input -> hidden sharded over the matmul group
        self.fc1 = DistributedMatmul(
            in_features,
            hidden_features,
            input_format=matmul_format,
            comm_name=comm_name,
            parallel_mode="column",
            bias=True,
            use_te=use_te,
        )

        # initialize the weights correctly
        scale = math.sqrt(2.0 / in_features)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=scale)
        nn.init.constant_(self.fc1.bias, 0.0)

        # row-parallel: sharded hidden -> all-reduced (replicated) output
        self.fc2 = DistributedMatmul(
            hidden_features,
            out_features,
            input_format=matmul_format,
            comm_name=comm_name,
            parallel_mode="row",
            bias=output_bias,
            use_te=use_te,
        )

        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=scale)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0.0)

        self.act = act_layer()

        if drop_rate > 0.0:
            if drop_type == "iid":
                self.drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                self.drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Error, drop_type {drop_type} not supported")
        else:
            self.drop = nn.Identity()

    @staticmethod
    def upgrade_state_dict(state_dict, prefix=""):
        r"""
        Upgrade a state dict of an earlier version in place, of a serial or a distributed MLP; see
        :func:`~makani.models.common.mlp.upgrade_mlp_state_dict`.
        """
        upgrade_mlp_state_dict(state_dict, prefix)

    def fwd(self, x):
        r"""
        Run the block body, without the checkpointing wrapper.

        Split out from ``forward`` so gradient checkpointing has a plain
        callable to recompute.

        Parameters
        ----------
        x : torch.Tensor
            Replicated input with ``in_features`` features.

        Returns
        -------
        torch.Tensor
            Replicated output with ``out_features`` features.
        """
        # on the TE path the matmuls are channels-last ("traditional"): permute
        # nchw -> nhwc once here and back at the end, keeping act/iid-dropout
        # channels-last in between.
        if self.channels_last:
            x = x.permute(0, 2, 3, 1).contiguous()

        # do the mlp
        # first layer
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        # second layer
        x = self.fc2(x)
        x = self.drop(x)

        if self.channels_last:
            x = x.permute(0, 3, 1, 2).contiguous()

        return x

    @torch.compiler.disable(recursive=False)
    def _checkpoint_forward(self, x):
        return checkpoint(self.fwd, x, use_reentrant=False)

    def forward(self, x):
        r"""
        Apply the tensor-parallel feed-forward block.

        Parameters
        ----------
        x : torch.Tensor
            Replicated input with ``in_features`` features, in the layout
            implied by ``input_format``.

        Returns
        -------
        torch.Tensor
            Replicated output with ``out_features`` features.
        """
        if self.checkpointing:
            return self._checkpoint_forward(x)
        else:
            return self.fwd(x)


class StochasticMLP(nn.Module):
    r"""
    Two-layer MLP whose weights are resampled from a learned distribution.

    Instead of holding point-estimate weights, each layer stores a mean and a
    log standard deviation and draws a fresh weight on every forward pass:

    .. math::

        W = s\, e^{\log \sigma} \odot \varepsilon + \mu,
        \qquad \varepsilon \sim \mathcal{N}(0, 1)

    where :math:`s` is a constant fan-in scale. This is a variational /
    Bayesian-style layer: the model learns how uncertain each weight is, and
    the resulting forward-pass variability is a principled source of ensemble
    spread rather than an injected perturbation. Parameterizing the spread
    through :math:`\log \sigma` keeps it positive without a constraint, and
    leaving it at its zero initialization makes the effective standard
    deviation exactly :math:`s`.

    The draws come from private, explicitly seeded generators rather than the
    global RNG, so the sampled weights are reproducible and can be decorrelated
    across ensemble members.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int, optional
        Width of the hidden layer, defaults to ``in_features``.
    out_features : int, optional
        Number of output features, defaults to ``in_features``.
    act_layer : callable, optional
        Activation constructor, by default :class:`torch.nn.GELU`.
    output_bias : bool, optional
        Whether the output projection carries a bias, by default ``True``.
    input_format : str, optional
        Only ``"nchw"`` is supported; anything else raises.
    drop_rate : float, optional
        Dropout probability, by default ``0.0``.
    drop_type : str, optional
        ``"iid"`` or ``"features"``, by default ``"iid"``.
    checkpointing : bool, optional
        Recompute the block in the backward pass, by default ``False``.
    gain : float, optional
        Scales the variance of the output layer's initialization, by default ``1.0``.
    seed : int, optional
        Seed for the private weight-sampling generators. Give distinct values
        to distinct ensemble members.
    **kwargs
        Ignored; present so block configs can pass extra keys.

    Raises
    ------
    NotImplementedError
        If ``input_format`` is not ``"nchw"``, if ``drop_type`` is unsupported,
        or if ``"traditional"`` is combined with feature dropout.

    Notes
    -----
    The weight parameters are tagged ``is_shared_mp = {"spatial": "sum", "matmul": "avg"}``:
    they are replicated over the model-parallel ranks, summed over the spatial
    ranks, each holding the contribution of its part of the grid, and averaged
    over the matmul ranks, which compute the same.

    See Also
    --------
    DistributedMLP : the deterministic counterpart.
    """

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
        seed=333,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # generator objects:
        self.set_rng(seed=seed)

        # First fully connected layer
        if input_format == "nchw":
            self.fc1_weight_log_std = nn.Parameter(torch.zeros(hidden_features, in_features, 1, 1))
            self.fc1_weight_mean = nn.Parameter(torch.zeros(hidden_features, in_features, 1, 1))
            self.fc1_bias = nn.Parameter(torch.zeros(hidden_features))
        else:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        # sharing settings
        self.fc1_weight_log_std.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
        self.fc1_weight_mean.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
        self.fc1_bias.is_shared_mp = {"spatial": "sum", "matmul": "avg"}

        # The stochastic weight is  weight = scale * exp(log_std) * eps + mean  with
        # eps ~ N(0, 1). log_std keeps its zero init (so the effective std == scale and
        # is always positive thanks to exp), and the fan-in scale is a constant factor.
        scale = math.sqrt(1.0 / in_features)
        self.fc1_weight_std_scale = scale
        nn.init.normal_(self.fc1_weight_mean, mean=0.0, std=scale)

        # activation
        self.act = act_layer()

        # sanity checks
        if (input_format == "traditional") and (drop_type == "features"):
            raise NotImplementedError(
                "Error, traditional input format and feature dropout cannot be selected simultaneously"
            )

        # output layer
        if input_format == "nchw":
            self.fc2_weight_log_std = nn.Parameter(torch.zeros(out_features, hidden_features, 1, 1))
            self.fc2_weight_mean = nn.Parameter(torch.zeros(out_features, hidden_features, 1, 1))
            self.fc2_bias = nn.Parameter(torch.zeros(out_features)) if output_bias else None
        else:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        # sharing settings
        self.fc2_weight_log_std.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
        self.fc2_weight_mean.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
        if self.fc2_bias is not None:
            self.fc2_bias.is_shared_mp = {"spatial": "sum", "matmul": "avg"}

        # gain factor for the output determines the scaling of the output init
        # (same scheme as fc1: log_std stays at its zero init, scale is a constant factor)
        scale = math.sqrt(gain / hidden_features / 2)
        self.fc2_weight_std_scale = scale
        nn.init.normal_(self.fc2_weight_mean, mean=0.0, std=scale)
        if self.fc2_bias is not None:
            nn.init.constant_(self.fc2_bias, 0.0)

        if drop_rate > 0.0:
            if drop_type == "iid":
                self.drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                self.drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Error, drop_type {drop_type} not supported")
        else:
            self.drop = nn.Identity()

    @torch.compiler.disable(recursive=False)
    def set_rng(self, seed=333):
        r"""
        Re-seed the private weight-sampling generators.

        Parameters
        ----------
        seed : int, optional
            Seed applied to both the CPU and CUDA generators, by default
            ``333``. The CUDA generator is bound to this rank's local device.
        """
        self.rng_cpu = torch.Generator(device=torch.device("cpu"))
        self.rng_cpu.manual_seed(seed)
        if torch.cuda.is_available():
            self.rng_gpu = torch.Generator(device=torch.device(f"cuda:{comm.get_local_rank()}"))
            self.rng_gpu.manual_seed(seed)

    @torch.compiler.disable(recursive=False)
    def checkpoint_forward(self, x):
        r"""
        Run the block under gradient checkpointing.

        Normally reached via ``forward`` with ``checkpointing=True``. Note that
        the weights are resampled during recomputation, so the backward pass
        sees a different draw than the forward did.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, in_features, H, W)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, out_features, H, W)``.
        """
        return checkpoint(self.fwd, x, use_reentrant=False)

    def fwd(self, x):
        r"""
        Sample weights and run the block body, without the checkpointing wrapper.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, in_features, H, W)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, out_features, H, W)``. Two calls with the same
            input give different results, since the weights are redrawn.
        """

        # generate weight1: weight = scale * exp(log_std) * eps + mean
        weight1 = torch.empty_like(self.fc1_weight_mean)
        weight1.normal_(mean=0.0, std=1.0, generator=self.rng_gpu if weight1.is_cuda else self.rng_cpu)
        std1 = self.fc1_weight_std_scale * torch.exp(self.fc1_weight_log_std)
        weight1 = std1 * weight1 + self.fc1_weight_mean

        # fully connected 1
        x = pointwise_conv2d(x, weight1, bias=self.fc1_bias)

        # activation
        x = self.act(x)

        # dropout
        x = self.drop(x)

        # generate weight2: weight = scale * exp(log_std) * eps + mean
        weight2 = torch.empty_like(self.fc2_weight_mean)
        weight2.normal_(mean=0.0, std=1.0, generator=self.rng_gpu if weight2.is_cuda else self.rng_cpu)
        std2 = self.fc2_weight_std_scale * torch.exp(self.fc2_weight_log_std)
        weight2 = std2 * weight2 + self.fc2_weight_mean

        # fully connected 2
        x = pointwise_conv2d(x, weight2, bias=self.fc2_bias)

        # dropout
        x = self.drop(x)

        return x

    def forward(self, x):
        r"""
        Apply the stochastic feed-forward block.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, in_features, H, W)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, out_features, H, W)``, using a fresh weight
            draw.
        """
        if self.checkpointing:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)
