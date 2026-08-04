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
    r"""
    Stochastic depth: randomly zero the residual branch for whole samples.

    Module wrapper around :func:`drop_path`. Placed on the residual branch of a
    block, it drops that branch entirely for a random subset of the batch during
    training, so the network is implicitly trained as an ensemble of varying
    depth. Surviving samples are rescaled by :math:`1/(1-p)` to keep the
    expected activation unchanged, and the layer is a no-op in ``eval`` mode.

    Parameters
    ----------
    drop_prob : float, optional
        Probability :math:`p` of dropping the branch for a given sample.
        ``None`` or ``0.0`` disables the layer.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        r"""
        Drop the branch for a random subset of samples.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, ...)``. The drop mask is drawn per sample and
            broadcast over all remaining dimensions.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``. Returned unchanged in ``eval``
            mode or when ``drop_prob`` is ``0.0``.
        """
        return drop_path(x, self.drop_prob, self.training)


class SeededDropout2d(nn.Module):
    r"""
    Channel dropout drawing from a private, explicitly seeded RNG.

    Behaves like :class:`torch.nn.Dropout2d` (whole channels are zeroed, not
    individual elements), except that the mask is drawn from generators owned by
    this module instead of the global RNG. That makes the dropout pattern
    reproducible and independent of however much other randomness the
    surrounding model consumed, which is what ensemble members need when their
    perturbations must be controlled rather than incidental.

    Parameters
    ----------
    drop_prob : float, optional
        Probability of zeroing a channel, by default ``0.0``.
    seed : int, optional
        Seed for the private CPU and CUDA generators, by default ``333``. Pass
        distinct seeds across ensemble members to decorrelate them.

    Notes
    -----
    ``forward`` is marked :func:`torch.compiler.disable`, so it runs eagerly.
    Swapping global RNG state is not traceable by Dynamo; excluding this one
    method lets the surrounding graph still compile.
    """

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
        r"""
        Apply channel dropout using this module's private RNG.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``, with whole channels zeroed and
            the rest rescaled by :math:`1/(1-p)`. A no-op in ``eval`` mode.
        """
        with rng_context(self.rng_cpu, self.rng_gpu):
            xdrop = self.drop(x)
        return xdrop


class LayerScale(nn.Module):
    r"""
    Learned per-channel rescaling of a residual branch.

    Multiplies each channel by its own learned scalar, initialized to a small
    value so the branch starts out contributing almost nothing and the block
    behaves like an identity at initialization. The network then learns how much
    of each branch to admit, which is what makes very deep residual stacks train
    stably.

    Implemented as a grouped 1x1 convolution with one group per channel, which
    is equivalent to a broadcast multiply but keeps the parameter in the layout
    the surrounding conv-based blocks expect.

    Parameters
    ----------
    num_chans : int, optional
        Number of channels, by default ``3``. One scale is learned per channel.
    init_value : float, optional
        Value all scales are initialized to, by default ``0.1``.
    """

    def __init__(self, num_chans=3, init_value=0.1):
        super().__init__()
        self.num_chans = num_chans
        self.weight = nn.Parameter(torch.randn(self.num_chans, 1, 1, 1))
        torch.nn.init.constant_(self.weight, val=init_value)

    def forward(self, x):
        r"""
        Scale each channel by its learned coefficient.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, num_chans, H, W)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``.
        """
        return nn.functional.conv2d(x, self.weight, groups=self.num_chans)


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
    The projection weight and bias are tagged ``is_shared_mp = ["spatial"]``, so
    they are replicated rather than sharded across the spatial model-parallel
    group and their gradients are reduced over it.
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
        self.proj.weight.is_shared_mp = ["spatial"]
        self.proj.bias.is_shared_mp = ["spatial"]
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
            # permute back leaves channels-last strides; restore contiguous format so the
            # tag does not propagate into downstream convolutions, whose weight gradients
            # would then be tagged channels_last and defeat DDP's gradient_as_bucket_view.
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
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
            # see PatchEmbed2D.forward: restore contiguous format after the round trip so
            # the channels-last tag does not propagate into downstream convolutions.
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()
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
    All weights and biases are tagged ``is_shared_mp = ["spatial"]``: the layer
    is pointwise, so every spatial rank holds the same parameters and their
    gradients are reduced across the spatial group.
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
            # transposes around the te GEMMs.
            self.fc1 = fc1
            self.fc2 = fc2
            self.act = act
            self.drop = drop
            # Dropout2d ("features") drops whole channels at dim=1, so it needs the
            # nchw layout and forces a bounce back from channels-last around each
            # dropout. Plain Dropout ("iid") and Identity are elementwise/no-ops, so
            # for them we can stay channels-last across the whole MLP and pay only
            # one permute in + one out.
            self.inner_transpose = isinstance(drop, nn.Dropout2d)
        else:
            # create forward pass
            self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)

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

        # Padding the input to facilitate downsampling. The permute back leaves
        # channels-last strides, so make the layout explicit before the reshapes below --
        # which would otherwise force the copy implicitly anyway.
        x = self.pad(x.permute(0, -1, 1, 2)).permute(0, 2, 3, 1).contiguous()
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
