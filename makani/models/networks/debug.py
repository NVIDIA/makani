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
from torch import nn


class DebugNet(nn.Module):
    r"""
    Trivial pass-through network for testing the training pipeline.

    Multiplies its input by a single learned scalar, initialized to one. That
    scalar exists only so optimizer construction and the gradient reduction
    hooks have something to work with -- a network with no parameters would
    crash them. Use this to exercise the dataloader, loss, checkpointing and
    distributed plumbing without the cost or the confounding behavior of a real
    model.

    Parameters
    ----------
    **kwargs
        Ignored; accepted so the model registry can pass a full model config.
    """

    def __init__(self, **kwargs):
        super().__init__()

        # create dummy param so that it won't crash in optimizer instantiation
        self.factor = nn.Parameter(torch.ones((1), dtype=torch.float32))

    def forward(self, x):
        r"""
        Scale the input by the learned scalar.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``. Note the channel count is
            unchanged, so this only stands in for a model whose input and
            output channels match.
        """
        return self.factor * x
