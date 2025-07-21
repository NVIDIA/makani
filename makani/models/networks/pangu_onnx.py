# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys
import math
import numpy as np
from collections.abc import Sequence
from copy import deepcopy

import torch
import time
import torch.nn as nn
from torch.nn import functional as F
from makani.utils.features import get_channel_groups

from makani.models.onnx_wrapper import OnnxWrapper

class PanguOnnx(OnnxWrapper):
    '''
    An ONNX Wrapper that runs inference on the Pangu model release in https://github.com/198808xc/Pangu-Weather
    Args:
        channel_names: List containing the name of the channels/variables that are inputted in the model
        channel_order_surface: List containing the names of the surface variables with the ordering that the ONNX model expects
        channel_order_atmo: List containing the names of the atmospheric variables with the ordering that the ONNX model expects
        channel_order_PL: List containing the names of the pressure levels with the ordering that the ONNX model expects
        onnx_file: Path to the ONNX file containing the model
    '''
    def __init__(self, 
        channel_names=[],
        aux_channel_names=[],
        onnx_file=None,
        **kwargs,
    ):
        super(PanguOnnx,self).__init__(onnx_file, **kwargs)

        self._precompute_channel_groups(channel_names, aux_channel_names)


    def _precompute_channel_groups(
        self,
        channel_names=[],
        aux_channel_names=[],
    ):
        """
        group the channels appropriately into atmospheric pressure levels and surface variables
        """

        atmo_chans, surf_chans, _, pressure_lvls = get_channel_groups(channel_names, aux_channel_names)

        # compute how many channel groups will be kept internally
        self.n_atmo_groups = len(pressure_lvls)
        self.n_atmo_chans = len(atmo_chans) // self.n_atmo_groups

        # make sure they are divisible. Attention! This does not guarantee that the grrouping is correct
        if len(atmo_chans) % self.n_atmo_groups:
            raise ValueError(f"Expected number of atmospheric variables to be divisible by number of atmospheric groups but got {len(atmo_chans)} and {self.n_atmo_groups}")

        self.register_buffer("atmo_channels", torch.LongTensor(atmo_chans), persistent=False)
        self.register_buffer("surf_channels", torch.LongTensor(surf_chans), persistent=False)

        return

    def prepare_input(self, input):

        B,V,Lat,Long=input.shape

        if B>1:
            raise NotImplementedError("Not implemented yet for batch size greater than 1")   

        input=input.squeeze(0)
        surface_aux_inp=input[self.surf_channels]
        atmospheric_inp=input[self.atmo_channels].reshape(self.n_atmo_groups,self.n_atmo_chans,Lat,Long).transpose(1,0)
        
        return surface_aux_inp, atmospheric_inp

    def prepare_output(self, output_surface, output_atmospheric):
        """
        The output of the Pangu model gives separate outputs for surface and atmospheric variables
        Also the atmospheric variables are restructured before fed to the network --> see self.prepare_input()
        This functions reverts the restructuring and concatenates the output to a single tensor.
        """

        _,Lat,Long=output_surface.shape

        output=torch.cat([output_surface,output_atmospheric.reshape(-1,Lat,Long)],dim=0)

        return output.unsqueeze(0)

      
    def forward(self, input):
        
        surface, atmospheric = self.prepare_input(input)


        output,output_surface=self.onnx_session_run({'input':atmospheric,'input_surface':surface})

        output = self.prepare_output(output_surface, output)

        
        return output
