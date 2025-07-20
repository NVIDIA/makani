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

import os
import glob
import re

from collections import OrderedDict
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from makani.utils import comm
from makani.mpu.helpers import gather_uneven

from physicsnemo.distributed.utils import split_tensor_along_dim


def get_latest_checkpoint_version(checkpoint_path):
    try:
        checkpoint_path = max(glob.glob(checkpoint_path.format(mp_rank=0, checkpoint_version="*")), key=os.path.getmtime)
        pathname, _ = os.path.splitext(checkpoint_path)
        latest_version = int(re.match(r"^.*?_v(\d{1,})$", pathname).groups()[0])
    except:
        print(f"Could not identify version for checkpoint {checkpoint_path}. Skipping detection.")
        latest_version = 0

    return latest_version


def gather_model_state_dict(model: nn.Module, grads: Optional[bool]=False) -> OrderedDict:
    # create empty dict to hold the state
    state_dict = OrderedDict()

    # iterate over parameters and gather them from the ranks
    for name, param in model.named_parameters():
        weight = param.clone()
        if hasattr(param, "sharded_dims_mp"):
            # gather the weight across all sharded dimensions
            for d, group in enumerate(param.sharded_dims_mp):
                if group is not None:
                    weight = gather_uneven(weight, d, group)

        state_dict[name] = weight.cpu()

        if grads:
            if param.grad is not None:
                grad = param.grad.clone()
                if hasattr(param, "sharded_dims_mp"):
                    for d, group in enumerate(param.sharded_dims_mp):
                        if group is not None:
                            grad = gather_uneven(grad, d, group)
                grad = grad.cpu()
            else:
                grad = None

            state_dict[name + ".grad"] = grad            

    return state_dict


def scatter_model_state_dict(model: nn.Module, state_dict: OrderedDict, strict: Optional[bool] = True) -> OrderedDict():

    # iterate over model parameters and split accordingly
    for name, param in model.named_parameters():

        # make sure that the parameter is in the state dict
        if name in state_dict.keys():

            # in this case, we need to distribute the weight
            if hasattr(param, "sharded_dims_mp"):

                # make a copy
                weight = state_dict[name].clone()

                # split if necessary
                for d, group in enumerate(param.sharded_dims_mp):
                    # continue if there is nothing to do
                    if (group is None) or (comm.get_size(group) == 1):
                        continue

                    weight = split_tensor_along_dim(weight, dim=d, num_chunks=comm.get_size(group))[comm.get_rank(group)]

                # update state dict
                state_dict[name] = weight

        elif strict:
            # TODO: maybe do at least a warning for non-strict mode
            raise ValueError(f"Missing key {k}")

    return state_dict


def gather_optimizer_state_dict(model: nn.Module, optimizer: Optimizer) -> OrderedDict:

    # if optimizer is SGD, we can just return the local dict:
    if isinstance(optimizer, torch.optim.SGD):
        return optimizer.state_dict()

    # do sanity checks
    if not (isinstance(optimizer, torch.optim.Adam) or isinstance(optimizer, torch.optim.AdamW)):
        raise NotImplementedError("Error, only Adam and AdamW state can be stored in flexible format at the moment.")

    # state dict:
    state_dict = optimizer.state_dict()

    # we need to copy the optimizer dict the hard way
    optimizer_dict = OrderedDict()
    optimizer_dict["param_groups"] = []
    for pgroup in state_dict["param_groups"]:
        pdict = {key: value for key, value in pgroup.items()}
        optimizer_dict["param_groups"].append(pdict)

    # check whether the corresponding model paramter is distributed.
    # if yes, we need to gather it
    optimizer_dict["state"] = {}
    for index, param in enumerate(model.parameters()):
        optimizer_dict["state"][index] = {"step": state_dict["state"][index]["step"].clone()}
        if hasattr(param, "sharded_dims_mp"):
            exp_avg = state_dict["state"][index]["exp_avg"].clone()
            exp_avg_sq = state_dict["state"][index]["exp_avg_sq"].clone()

            # gather the optimizer state across all sharded dimensions
            for d, group in enumerate(param.sharded_dims_mp):
                if group is not None:
                    exp_avg = gather_uneven(exp_avg, d, group)
                    exp_avg_sq = gather_uneven(exp_avg_sq, d, group)

            optimizer_dict["state"][index]["exp_avg"] = exp_avg
            optimizer_dict["state"][index]["exp_avg_sq"] = exp_avg_sq
        else:
            optimizer_dict["state"][index]["exp_avg"] = state_dict["state"][index]["exp_avg"].clone()
            optimizer_dict["state"][index]["exp_avg_sq"] = state_dict["state"][index]["exp_avg_sq"].clone()

    return optimizer_dict


def scatter_optimizer_state_dict(model: nn.Module, optimizer: Optimizer, optimizer_state_dict: OrderedDict) -> OrderedDict():

    # some sanity checks
    # if optimizer is SGD, we can just return the local dict:
    if isinstance(optimizer, torch.optim.SGD):
        return optimizer_state_dict

    if not (isinstance(optimizer, torch.optim.Adam) or isinstance(optimizer, torch.optim.AdamW)):
        raise NotImplementedError("Error, only Adam and AdamW state can be restored from flexible format at the moment.")

    # iterate over model parameters and split accordingly
    for idp, param in enumerate(model.parameters()):

        # in this case, we need to distribute the weight
        if hasattr(param, "sharded_dims_mp"):

            # clone the state
            exp_avg = optimizer_state_dict["state"][idp]["exp_avg"].clone()
            exp_avg_sq = optimizer_state_dict["state"][idp]["exp_avg_sq"].clone()

            for d, group in enumerate(param.sharded_dims_mp):
                # continue if there is nothing to do
                if (group is None) or (comm.get_size(group) == 1):
                    continue

                exp_avg = split_tensor_along_dim(exp_avg, dim=d, num_chunks=comm.get_size(group))[comm.get_rank(group)]
                exp_avg_sq = split_tensor_along_dim(exp_avg_sq, dim=d, num_chunks=comm.get_size(group))[comm.get_rank(group)]

            # update the state dict
            optimizer_state_dict["state"][idp]["exp_avg"] = exp_avg
            optimizer_state_dict["state"][idp]["exp_avg_sq"] = exp_avg_sq

    return optimizer_state_dict


def prepend_prefix_to_state_dict(
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    r"""Append the prefix to states in state_dict in place.

    ..note::
        Given a `state_dict` from a local model, a DP/DDP model can load it by applying
        `prepend_prefix_to_state_dict(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = list(state_dict.keys())
    for key in keys:
        newkey = prefix + key
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if hasattr(state_dict, "_metadata"):
        keys = list(state_dict._metadata.keys())
        for key in keys:
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.
            if len(key) >= 0:
                newkey = prefix + key
                state_dict._metadata[newkey] = state_dict._metadata.pop(key)
