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


def compare_model_parameters(model1, model2):
    """Checks whether both models have the same parameters"""

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1.detach(), p2.detach()):
            return False
    return True


def check_parameters(model):
    """Prints shapes, strides and whether parameters are contiguous"""
    for p in model.parameters():
        if p.requires_grad:
            print(p.shape, p.stride(), p.is_contiguous())

    return
