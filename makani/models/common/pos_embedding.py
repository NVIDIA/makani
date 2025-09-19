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

import abc

import torch
import torch.nn as nn

from makani.utils import comm
from physicsnemo.distributed.utils import compute_split_shapes

class PositionEmbedding(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for position embeddings.

    This class defines the interface for position embedding modules
    that add positional information to input tensors.

    Parameters
    ----------
    img_shape : tuple, optional
        Image shape (height, width), by default (480, 960)
    grid : str, optional
        Grid type, by default "equiangular"
    num_chans : int, optional
        Number of channels, by default 1
    """

    def __init__(self, img_shape=(480, 960), grid="equiangular", num_chans=1):
        super().__init__()

        self.img_shape = img_shape
        self.num_chans = num_chans

    def forward(self):

        return self.position_embeddings

class LearnablePositionEmbedding(PositionEmbedding):
    """
    Learnable position embeddings for spherical transformers.

    This module provides learnable position embeddings that can be either
    latitude-only or full latitude-longitude embeddings.

    Parameters
    ----------
    img_shape : tuple, optional
        Image shape (height, width), by default (480, 960)
    grid : str, optional
        Grid type, by default "equiangular"
    num_chans : int, optional
        Number of channels, by default 1
    embed_type : str, optional
        Embedding type ("lat" or "latlon"), by default "lat"
    """

    def __init__(self, img_shape=(480, 960), grid="equiangular", num_chans=1, embed_type="lat"):
        super().__init__(img_shape=img_shape, grid=grid, num_chans=num_chans)

        # if distributed, make sure to split correctly across ranks:
        # in case of model parallelism, we need to make sure that we use the correct shapes per rank
        # for h
        if comm.get_size("h") > 1:
            self.local_shape_h = compute_split_shapes(img_shape[0], comm.get_size("h"))[comm.get_rank("h")]
        else:
            self.local_shape_h = img_shape[0]

        # for w
        if comm.get_size("w") > 1:
            self.local_shape_w = compute_split_shapes(img_shape[1], comm.get_size("w"))[comm.get_rank("w")]
        else:
            self.local_shape_w = img_shape[1]

        if embed_type == "latlon":
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_chans, self.local_shape_h, self.local_shape_w))
            self.position_embeddings.is_shared_mp = []
            self.position_embeddings.sharded_dims_mp = [None, None, "h", "w"]
        elif embed_type == "lat":
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_chans, self.local_shape_h, 1))
            self.position_embeddings.is_shared_mp = ["w"]
            self.position_embeddings.sharded_dims_mp = [None, None, "h", None]
        else:
            raise ValueError(f"Unknown learnable position embedding type {embed_type}")

    def forward(self):
        return self.position_embeddings.expand(-1,-1,self.local_shape_h, self.local_shape_w)
