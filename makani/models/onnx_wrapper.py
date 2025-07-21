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
import numpy as np
from typing import Union
import torch
import torch.nn as nn


import onnx


class OnnxWrapper(nn.Module):
    """
    A torch.nn.Module wrapper that runs inference on an ONNX model
    Args:
        onnx_file: File containing the onnx weights
    """
    _onnxruntime=None

    def __init__(self,onnx_file, **kwargs):
        super(OnnxWrapper,self).__init__()

        # Lazy import onnxruntime
        if OnnxWrapper._onnxruntime is None:
            import onnxruntime 
            OnnxWrapper._onnxruntime=onnxruntime
        self.ort=OnnxWrapper._onnxruntime

        #Initialize inference session
        
        self.options = self.ort.SessionOptions()
        self.options.enable_cpu_mem_arena=False
        self.options.enable_mem_pattern = False
        self.options.enable_mem_reuse = False
        # Increase the number for faster inference and more memory consumption
        self.options.intra_op_num_threads = 1
        self.cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

        self.load_onnx_session(onnx_file)
       
    def load_onnx_session(self,onnx_file):
        self.onnx_file=onnx_file

        # Check if cuda is available to initialize a CUDA onnxruntime        
        if  torch.cuda.is_available():
            self.gpu_session=self.ort.InferenceSession(self.onnx_file,sess_options=self.options, providers=[('CUDAExecutionProvider', self.cuda_provider_options)])
        else:
            self.gpu_session=None
        
        self.cpu_session=self.ort.InferenceSession(self.onnx_file,sess_options=self.options, providers=[('CPUExecutionProvider', self.cuda_provider_options)])


    def onnx_session_run(self,inputs):
        input_device=next(iter(inputs.values())).device
        if (self.gpu_session is not None) and input_device!=torch.device('cpu'):
            onnx_session=self.gpu_session
        else:
            onnx_session=self.cpu_session

        for key in inputs.keys():
            inputs[key]=inputs[key].cpu().detach().numpy().astype(np.float32)
                    
        outputs=onnx_session.run(None,inputs)

        output_tensor_list=[]
        for output in outputs:
            output_tensor_list.append(torch.from_numpy(output).to(input_device))

        return output_tensor_list