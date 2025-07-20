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

from typing import Optional, Tuple, List
from dataclasses import dataclass

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

import torch_harmonics as th
import torch_harmonics.distributed as thd

from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature
from makani.utils import comm


def _compute_channel_weighting_helper(channel_names: List[str], channel_weight_type: str) -> torch.Tensor:
    """
    auxiliary routine for predetermining channel weighting
    """

    # initialize empty tensor
    channel_weights = torch.ones(len(channel_names), dtype=torch.float32)

    if channel_weight_type == "constant":
        # nothing to do here
        pass

    elif channel_weight_type == "auto":

        for c, chn in enumerate(channel_names):
            if chn in ["u10m", "v10m", "u100m", "v100m", "tp", "sp", "msl", "tcwv"]:
                channel_weights[c] = 0.1
            elif chn in ["t2m", "2d"]:
                channel_weights[c] = 1.0
            elif chn[0] in ["z", "u", "v", "t", "r", "q"]:
                pressure_level = float(chn[1:])
                channel_weights[c] = 0.001 * pressure_level
            else:
                channel_weights[c] = 0.01

    elif channel_weight_type == "custom":

        weight_dict = {
            "u10m": 0.0037454564357253584,
            "v10m": 0.0031170353648495284,
            "u100m": 0.003769711955006904,
            "v100m": 0.0031352227345726185,
            "t2m": 0.012677452364115778,
            "sp": 0.015285068386494079,
            "msl": 0.009071787341075057,
            "tcwv": 0.026559985751436477,
            "u50": 0.006971333585456881,
            "u100": 0.005758677866964381,
            "u150": 0.005307960678288598,
            "u200": 0.0054915693809966925,
            "u250": 0.005143796945970392,
            "u300": 0.004881395005330393,
            "u400": 0.004641379569852355,
            "u500": 0.004722972449376466,
            "u600": 0.0045655400343972505,
            "u700": 0.004274954866969274,
            "u850": 0.00394871467079016,
            "u925": 0.003876401915997666,
            "u1000": 0.003693958885577892,
            "v50": 0.0029775261093895116,
            "v100": 0.0030503389749466344,
            "v150": 0.0036099618876629424,
            "v200": 0.003693958885577892,
            "v250": 0.003761591950795796,
            "v300": 0.003827548631576873,
            "v400": 0.0035566579697697527,
            "v500": 0.0034787232333803753,
            "v600": 0.0033011702517144583,
            "v700": 0.0031254032450236213,
            "v850": 0.002936223729561914,
            "v925": 0.0030840071755531095,
            "v1000": 0.0031073292938735737,
            "z50": 0.022460695346070078,
            "z100": 0.02676351054646664,
            "z150": 0.023053717005372253,
            "z200": 0.02400438574786183,
            "z250": 0.021493157700393208,
            "z300": 0.028864777903420638,
            "z400": 0.02666135974285417,
            "z500": 0.022902545090582926,
            "z600": 0.02275334284243581,
            "z700": 0.017729127544740594,
            "z850": 0.014197715960625596,
            "z925": 0.011105367651236556,
            "z1000": 0.00890979113855586,
            "t50": 0.008819793248267415,
            "t100": 0.01133973417634382,
            "t150": 0.00903657988696998,
            "t200": 0.007848625002952577,
            "t250": 0.009674897856825198,
            "t300": 0.013179766514392064,
            "t400": 0.013777665192559749,
            "t500": 0.015768117951755742,
            "t600": 0.015086989746496315,
            "t700": 0.013643117680913659,
            "t850": 0.012631602626813369,
            "t925": 0.013279992875718238,
            "t1000": 0.012169470823393369,
            "q50": 0.018286063488554435,
            "q100": 0.03204255161755869,
            "q150": 0.029349900221125182,
            "q200": 0.04037731937935141,
            "q250": 0.036006578621792754,
            "q300": 0.036381647149103094,
            "q400": 0.031184268984945508,
            "q500": 0.028511331643378747,
            "q600": 0.025587092500468107,
            "q700": 0.021167503795841796,
            "q850": 0.021039988712734315,
            "q925": 0.02718006323979686,
            "q1000": 0.04157902531326068,
        }

        for c, chn in enumerate(channel_names):
            channel_weights[c] = weight_dict.get(chn, 1.0)

    elif channel_weight_type == "pangu":

        weight_dict = {
            "u10m": 0.77,
            "v10m": 0.66,
            "t2m": 3.0,
            "msl": 1.5,
            "u50": 0.77,
            "u100": 0.77,
            "u150": 0.77,
            "u200": 0.77,
            "u250": 0.77,
            "u300": 0.77,
            "u400": 0.77,
            "u500": 0.77,
            "u600": 0.77,
            "u700": 0.77,
            "u850": 0.77,
            "u925": 0.77,
            "u1000": 0.77,
            "v50": 0.54,
            "v100": 0.54,
            "v150": 0.54,
            "v200": 0.54,
            "v250": 0.54,
            "v300": 0.54,
            "v400": 0.54,
            "v500": 0.54,
            "v600": 0.54,
            "v700": 0.54,
            "v850": 0.54,
            "v925": 0.54,
            "v1000": 0.54,
            "z50": 3.0,
            "z100": 3.0,
            "z150": 3.0,
            "z200": 3.0,
            "z250": 3.0,
            "z300": 3.0,
            "z400": 3.0,
            "z500": 3.0,
            "z600": 3.0,
            "z700": 3.0,
            "z850": 3.0,
            "z925": 3.0,
            "z1000": 3.0,
            "t50": 1.5,
            "t100": 1.5,
            "t150": 1.5,
            "t200": 1.5,
            "t250": 1.5,
            "t300": 1.5,
            "t400": 1.5,
            "t500": 1.5,
            "t600": 1.5,
            "t700": 1.5,
            "t850": 1.5,
            "t925": 1.5,
            "t1000": 1.5,
            "q50": 0.6,
            "q100": 0.6,
            "q150": 0.6,
            "q200": 0.6,
            "q250": 0.6,
            "q300": 0.6,
            "q400": 0.6,
            "q500": 0.6,
            "q600": 0.6,
            "q700": 0.6,
            "q850": 0.6,
            "q925": 0.6,
            "q1000": 0.6,
        }

        for c, chn in enumerate(channel_names):
            channel_weights[c] = weight_dict.get(chn, 1.0)
    else:
        raise NotImplementedError("Unknown channel weighting type {channel_weight_type}")

    # normalize
    channel_weights = channel_weights / torch.sum(channel_weights)

    return channel_weights


@dataclass
class LossType(object):
    Deterministic = 1
    Probabilistic = 2


# geometric base loss class
class GeometricBaseLoss(nn.Module, metaclass=ABCMeta):
    """
    Geometric base loss class used by all geometric losses
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        pole_mask: int,
        spatial_distributed: Optional[bool] = False,
    ):
        super().__init__()

        self.img_shape = img_shape
        self.crop_shape = crop_shape
        self.crop_offset = crop_offset
        self.channel_names = channel_names
        self.pole_mask = pole_mask
        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed

        quadrature_rule = grid_to_quadrature_rule(grid_type)

        # get the quadrature
        self.quadrature = GridQuadrature(
            quadrature_rule,
            img_shape=self.img_shape,
            crop_shape=self.crop_shape,
            crop_offset=self.crop_offset,
            normalize=True,
            pole_mask=self.pole_mask,
            distributed=self.spatial_distributed,
        )

    @property
    def type(self):
        return LossType.Deterministic

    @property
    def n_channels(self):
        return len(self.channel_names)

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str) -> torch.Tensor:
        return _compute_channel_weighting_helper(self.channel_names, channel_weight_type)

    @abstractmethod
    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class SpectralBaseLoss(nn.Module, metaclass=ABCMeta):
    """
    Geometric base loss class used by all geometric losses
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape: Tuple[int, int],
        crop_offset: Tuple[int, int],
        channel_names: List[str],
        grid_type: str,
        spatial_distributed: Optional[bool] = False,
    ):
        super().__init__()

        self.img_shape = img_shape
        self.crop_shape = crop_shape
        self.crop_offset = crop_offset
        self.channel_names = channel_names
        self.spatial_distributed = comm.is_distributed("spatial") and spatial_distributed

        if self.spatial_distributed and (comm.get_size("spatial") > 1):
            if not thd.is_initialized():
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
            self.sht = thd.DistributedRealSHT(*img_shape, grid=grid_type)
        else:
            self.sht = th.RealSHT(*img_shape, grid=grid_type).float()

    @property
    def type(self):
        return LossType.Deterministic

    @property
    def n_channels(self):
        return len(self.n_channels)

    @torch.compiler.disable(recursive=False)
    def compute_channel_weighting(self, channel_weight_type: str) -> torch.Tensor:
        return _compute_channel_weighting_helper(self.channel_names, channel_weight_type)

    @abstractmethod
    def forward(self, prd: torch.Tensor, tar: torch.Tensor, wgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
