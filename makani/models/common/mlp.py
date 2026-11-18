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

"""Channel MLPs: the encoder/decoder and the MLP."""

import math
import warnings

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# transformer engine is an optional dependency: it is only used for the
# (optional) FP8/FP4 MLP path and must not be required for import. availability is
# checked without importing it; the module is imported lazily where used.
from makani.utils.te_helpers import TE_AVAILABLE as _TE_AVAILABLE, get_te


def flatten_biases(state_dict, keys) -> None:
    r"""Flatten the biases of earlier versions kept as ``(1, C, 1, 1)`` to ``(C,)``, in place, where they are so."""
    for key in keys:
        bias = state_dict.get(key, None)
        if (bias is not None) and (bias.dim() == 4):
            state_dict[key] = bias.reshape(-1)


def upgrade_mlp_state_dict(state_dict, prefix: str = "") -> None:
    r"""
    Upgrade the keys of an MLP of an earlier version in place, for ``MLP`` and ``DistributedMLP`` alike.

    The serial MLP kept its layers in a sequence, ``fwd.0`` and ``fwd.3``, now ``fc1`` and ``fc2``, and the
    distributed one kept the biases of the nchw format as ``(1, C, 1, 1)``. Either may be restored into either.
    """
    renames = {prefix + "fwd.0.": prefix + "fc1.", prefix + "fwd.3.": prefix + "fc2."}
    for key in list(state_dict.keys()):
        for old, new in renames.items():
            if key.startswith(old):
                state_dict[new + key[len(old) :]] = state_dict.pop(key)
                break
    flatten_biases(state_dict, [prefix + "fc1.bias", prefix + "fc2.bias"])


class EncoderDecoder(nn.Module):
    r"""
    Stack of pointwise layers used as an encoder or decoder head.

    A configurable number of hidden layers, each a 1x1 convolution (or a
    :class:`~torch.nn.Linear`, depending on ``input_format``) followed by an
    activation, ending in a bias-free output projection. Acting only across
    channels, it changes the feature dimension without touching spatial
    structure -- which is what makes it the right tool for lifting input
    variables into the model's latent width and projecting back out again.

    Hidden layers are initialized with :math:`\mathcal{N}(0, 2/\mathrm{fan\_in})`
    (He initialization, appropriate for the ReLU-family activations these are
    used with), while the output layer uses ``gain / fan_in``, so callers can
    scale down the final projection where a near-zero-variance initial output is
    wanted.

    Parameters
    ----------
    num_layers : int
        Number of hidden layer/activation pairs before the output projection.
        ``0`` gives a bare linear projection.
    input_dim : int
        Number of input channels/features.
    output_dim : int
        Number of output channels/features.
    hidden_dim : int
        Width of the hidden layers.
    act_layer : callable
        Activation module constructor, called with no arguments.
    gain : float, optional
        Scales the variance of the output layer's initialization, by default ``1.0``.
    input_format : str, optional
        ``"nchw"`` (default) uses 1x1 convolutions on ``(B, C, H, W)`` inputs;
        ``"traditional"`` uses linear layers acting on the last dimension.
    groups : int, optional
        Number of groups for the convolutions, by default ``1``. Only meaningful
        for ``"nchw"``; splits the channel mixing into independent blocks.

    Raises
    ------
    NotImplementedError
        If ``input_format`` is not ``"nchw"`` or ``"traditional"``.

    Notes
    -----
    All weights and biases are tagged
    ``is_shared_mp = {"spatial": "sum", "matmul": "avg"}``: the layer is
    pointwise, so every rank holds the same parameters; their gradients are
    summed over the spatial ranks, each holding the contribution of its part of
    the grid, and averaged over the matmul ranks, which compute the same.
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
            encoder_modules[-1].weight.is_shared_mp = {"spatial": "sum", "matmul": "avg"}

            # proper initializaiton (fan-in per group for grouped conv)
            fan_in = (current_dim // groups) if input_format == "nchw" else current_dim
            scale = math.sqrt(2.0 / fan_in)
            nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
            if encoder_modules[-1].bias is not None:
                encoder_modules[-1].bias.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
                nn.init.constant_(encoder_modules[-1].bias, 0.0)

            encoder_modules.append(act_layer())
            current_dim = hidden_dim

        # final output layer
        if input_format == "nchw":
            encoder_modules.append(nn.Conv2d(current_dim, output_dim, 1, bias=False, groups=groups))
        elif input_format == "traditional":
            encoder_modules.append(nn.Linear(current_dim, output_dim, bias=False))

        # weight sharing
        encoder_modules[-1].weight.is_shared_mp = {"spatial": "sum", "matmul": "avg"}

        # proper initializaiton
        fan_in = (current_dim // groups) if input_format == "nchw" else current_dim
        scale = math.sqrt(gain / fan_in)
        nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
        if encoder_modules[-1].bias is not None:
            encoder_modules[-1].bias.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
            nn.init.constant_(encoder_modules[-1].bias, 0.0)

        self.fwd = nn.Sequential(*encoder_modules)

    @staticmethod
    def upgrade_state_dict(state_dict, prefix=""):
        r"""
        Upgrade a state dict of an earlier version in place: a :class:`~makani.models.common.distributed.mlp.DistributedEncoderDecoder`
        kept the biases of the nchw format as ``(1, C, 1, 1)``.

        Parameters
        ----------
        state_dict : dict
            A state dict holding the layers.
        prefix : str, optional
            The prefix of the keys of the layers.
        """
        flatten_biases(
            state_dict,
            [
                k
                for k in state_dict
                if k.startswith(prefix + "fwd.") and k.endswith(".bias") and k.count(".") == prefix.count(".") + 2
            ],
        )

    def forward(self, x):
        r"""
        Apply the stack of pointwise layers.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, input_dim, H, W)`` for ``input_format="nchw"``,
            or ``(..., input_dim)`` for ``"traditional"``.

        Returns
        -------
        torch.Tensor
            Output with the channel/feature dimension mapped to ``output_dim``
            and all other dimensions unchanged.
        """
        return self.fwd(x)


class MLP(nn.Module):
    r"""
    Two-layer pointwise feed-forward block.

    The channel-mixing half of a transformer or neural-operator block: expand to
    ``hidden_features``, apply a nonlinearity, project back to
    ``out_features``, with dropout after each stage. All operations are
    pointwise in space, so spatial mixing is left entirely to the attention or
    spectral layer this block is paired with.

    Weights use :math:`\mathcal{N}(0, 2/\mathrm{fan\_in})` on the first layer and
    ``gain / hidden_features`` on the second, so ``gain`` controls how strongly
    the block contributes at initialization.

    Parameters
    ----------
    in_features : int
        Number of input channels/features.
    hidden_features : int, optional
        Width of the hidden layer, defaults to ``in_features``.
    out_features : int, optional
        Number of output channels/features, defaults to ``in_features``.
    act_layer : callable, optional
        Activation module constructor, by default :class:`torch.nn.GELU`.
    output_bias : bool, optional
        Whether the output projection carries a bias, by default ``True``.
    input_format : str, optional
        ``"nchw"`` (default) for ``(B, C, H, W)`` inputs, or ``"traditional"``
        for channels-last inputs.
    drop_rate : float, optional
        Dropout probability, by default ``0.0`` (dropout replaced by identity).
    drop_type : str, optional
        ``"iid"`` (default) drops individual elements; ``"features"`` drops
        whole channels via :class:`torch.nn.Dropout2d`. ``"features"`` requires
        ``input_format="nchw"``.
    checkpointing : bool, optional
        If ``True``, recompute the block during the backward pass instead of
        storing its activations, trading compute for memory. By default ``False``.
    gain : float, optional
        Scales the variance of the output layer's initialization, by default ``1.0``.
    use_te : bool, optional
        If ``True``, use TransformerEngine linear layers for the two GEMMs
        (enabling FP8/FP4 paths). Silently falls back to the standard path with
        a warning if TransformerEngine is not installed. Initialization is
        identical either way, so toggling this does not change results at step 0.
    **kwargs
        Ignored; present so model configs can pass extra keys.

    Raises
    ------
    NotImplementedError
        If ``input_format`` is unsupported, if ``drop_type`` is unsupported, or
        if ``"traditional"`` is combined with ``drop_type="features"``.

    Notes
    -----
    TransformerEngine linears operate on the last dimension, so for ``"nchw"``
    inputs the block transposes to channels-last around the GEMMs. When dropout
    is elementwise the whole block stays channels-last and pays only one permute
    in and one out; feature dropout needs ``nchw`` to drop at ``dim=1`` and
    forces additional transposes.
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
            warnings.warn(
                "use_te=True was requested but transformer_engine is not installed; falling back to the standard MLP."
            )

        # sanity checks
        if (input_format == "traditional") and (drop_type == "features"):
            raise NotImplementedError(
                "Error, traditional input format and feature dropout cannot be selected simultaneously"
            )

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

        # sharing settings: weights/biases are replicated over the model group, so the
        # gradient reduction hook sums them over "spatial", whose ranks each hold the
        # contribution of their part of the grid, and averages them over "matmul",
        # whose ranks compute the same. This must be stamped on every parameter
        # (including te.Linear ones, which otherwise arrive unannotated) for the comm
        # hook to reduce them correctly.
        fc1.weight.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
        fc1.bias.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
        fc2.weight.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
        if fc2.bias is not None:
            fc2.bias.is_shared_mp = {"spatial": "sum", "matmul": "avg"}

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

        # the layers under the names of DistributedMLP, so that a checkpoint of one restores into the other
        self.fc1 = fc1
        self.act = act
        self.drop = drop
        self.fc2 = fc2

        if self.use_te:
            # Dropout2d ("features") drops whole channels at dim=1, so it needs the
            # nchw layout and forces a bounce back from channels-last around each
            # dropout. Plain Dropout ("iid") and Identity are elementwise/no-ops, so
            # for them we can stay channels-last across the whole MLP and pay only
            # one permute in + one out.
            self.inner_transpose = isinstance(drop, nn.Dropout2d)

    @staticmethod
    def upgrade_state_dict(state_dict, prefix=""):
        r"""
        Upgrade a state dict of an earlier version in place: the layers were kept in a sequence, ``fwd.0`` and
        ``fwd.3``, and those of a :class:`~makani.models.common.distributed.mlp.DistributedMLP` kept the bias of the nchw format as
        ``(1, C, 1, 1)``.

        Parameters
        ----------
        state_dict : dict
            A state dict holding the MLP.
        prefix : str, optional
            The prefix of the keys of the MLP.
        """
        upgrade_mlp_state_dict(state_dict, prefix)

    def fwd(self, x):
        r"""The block body, without the checkpointing wrapper; see :meth:`forward`."""
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

    def _te_forward(self, x):
        if self.input_format == "nchw":
            # permute nchw -> nhwc once for the te GEMMs; act/iid-dropout are
            # elementwise so they run channels-last too. Only feature dropout
            # (Dropout2d) needs nchw to drop channels at dim=1, so the inner
            # transposes around the first dropout are guarded by inner_transpose.
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.act(self.fc1(x))
            if self.inner_transpose:
                x = x.permute(0, 3, 1, 2).contiguous()
            x = self.drop(x)
            if self.inner_transpose:
                x = x.permute(0, 2, 3, 1).contiguous()
            x = self.fc2(x)
            # final permute back to nchw is unconditional: it is the last op, and
            # dropping in nchw is correct for both feature and iid/no dropout.
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.drop(x)
        else:
            # traditional format already has the channels in the last dimension
            x = self.drop(self.act(self.fc1(x)))
            x = self.drop(self.fc2(x))
        return x

    @torch.compiler.disable(recursive=False)
    def checkpoint_forward(self, x):
        r"""
        Run the block under gradient checkpointing.

        Activations are discarded and recomputed during the backward pass.
        Normally reached via ``forward`` with ``checkpointing=True`` rather than
        called directly.

        Parameters
        ----------
        x : torch.Tensor
            Input in the layout implied by ``input_format``.

        Returns
        -------
        torch.Tensor
            Same result as ``forward``, with the intermediate activations not
            retained.
        """
        fwd = self._te_forward if self.use_te else self.fwd
        return checkpoint(fwd, x, use_reentrant=False)

    def forward(self, x):
        r"""
        Apply the feed-forward block.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, in_features, H, W)`` for
            ``input_format="nchw"``, or ``(..., in_features)`` for
            ``"traditional"``.

        Returns
        -------
        torch.Tensor
            Output with the channel/feature dimension mapped to
            ``out_features``, all other dimensions unchanged.
        """
        if self.checkpointing:
            return self.checkpoint_forward(x)
        elif self.use_te:
            return self._te_forward(x)
        else:
            return self.fwd(x)
