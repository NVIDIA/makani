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

"""Regularization layers: stochastic depth, seeded dropout and layer scale."""

import torch
import torch.nn as nn

from makani.utils.context import rng_context


@torch.compile(fullgraph=False)
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if (drop_prob is None) or (drop_prob == 0.0) or (not training):
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
