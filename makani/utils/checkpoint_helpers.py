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
import zlib
import pickle

from collections import OrderedDict
from typing import Any, Dict, List, Optional
from warnings import warn

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer

from makani.utils import comm
from makani.mpu.helpers import gather_uneven

from torch_harmonics.distributed import split_tensor_along_dim


# escape hatch for loading legacy checkpoints which contain pickled objects that are not
# on the allowlist below. Only set this for checkpoints from a trusted source: it restores
# the unrestricted (arbitrary code execution) unpickler.
UNSAFE_LOAD_ENV_VAR = "MAKANI_ALLOW_UNSAFE_CHECKPOINT_LOAD"

_safe_globals_registered = False


def _register_checkpoint_safe_globals():
    """Allowlist the non-tensor types makani stores inside checkpoints.

    ``torch.load(weights_only=True)`` uses a restricted unpickler which only knows about
    tensors and a handful of builtin containers. Everything makani writes into a checkpoint
    beyond that has to be declared here. Note that the sharding metadata (``sharded_dims_mp``)
    attached to the checkpoint tensors needs no entry: it is a plain list of strings/None and
    is restored through ``torch._tensor._rebuild_from_type_v2``, which the restricted
    unpickler permits.
    """

    global _safe_globals_registered
    if _safe_globals_registered:
        return

    # imported lazily to avoid a circular import through makani.utils
    from makani.utils.YParams import ParamsBase, YParams

    torch.serialization.add_safe_globals([ParamsBase, YParams])

    _safe_globals_registered = True


def load_checkpoint(checkpoint_fname: str, map_location: str = "cpu", mmap: bool = False) -> Dict[str, Any]:
    """Load a makani checkpoint with the safe (``weights_only=True``) unpickler.

    Parameters
    ============
    checkpoint_fname: str
        Path of the checkpoint file to load
    map_location: str
        Device to map the storages to, forwarded to ``torch.load``
    mmap: bool
        Whether to memory-map the tensor storages instead of reading them eagerly

    Returns
    ========
    Dict[str, Any]
        The deserialized checkpoint dictionary
    """

    _register_checkpoint_safe_globals()

    if os.environ.get(UNSAFE_LOAD_ENV_VAR, "0").lower() in ("1", "true", "yes"):
        warn(
            f"{UNSAFE_LOAD_ENV_VAR} is set, loading {checkpoint_fname} with the unrestricted unpickler. "
            "This executes arbitrary code contained in the checkpoint file and should only be done for "
            "checkpoints from a trusted source.",
            RuntimeWarning,
        )
        return torch.load(checkpoint_fname, map_location=map_location, weights_only=False, mmap=mmap)

    try:
        return torch.load(checkpoint_fname, map_location=map_location, weights_only=True, mmap=mmap)
    except pickle.UnpicklingError as err:
        raise RuntimeError(
            f"Checkpoint {checkpoint_fname} contains pickled objects which are not on the allowlist in "
            f"{__name__}._register_checkpoint_safe_globals. If the type reported below is a legitimate "
            "makani type, add it to that allowlist. If the checkpoint comes from a trusted source, loading "
            f"can be forced by setting {UNSAFE_LOAD_ENV_VAR}=1, which disables the safety check entirely."
        ) from err


def get_latest_checkpoint_version(checkpoint_path):
    try:
        checkpoint_path = max(
            glob.glob(checkpoint_path.format(mp_rank=0, checkpoint_version="*")), key=os.path.getmtime
        )
        pathname, _ = os.path.splitext(checkpoint_path)
        latest_version = int(re.match(r"^.*?_v(\d{1,})$", pathname).groups()[0])
    except:
        print(f"Could not identify version for checkpoint {checkpoint_path}. Skipping detection.")
        latest_version = 0

    return latest_version


def supports_dataloader_state(dataloader: Any) -> bool:
    """Whether the dataloader can capture and restore its state (only the DALI loaders can)."""
    return (
        (dataloader is not None)
        and hasattr(dataloader, "state_dict")
        and hasattr(dataloader, "load_state_dict")
        and getattr(dataloader, "enable_checkpointing", False)
    )


def gather_dataloader_state_dict(dataloader: Any) -> Optional[List[Dict[str, Any]]]:
    """Collect the dataloader state of every data-parallel rank.

    The data pipeline is sharded over the ``data`` comm, so every rank in that comm holds a
    different state, while checkpoint files are only written by rank 0 of that comm. The states
    are therefore gathered into a list indexed by the rank within the ``data`` comm, which is
    what ``scatter_dataloader_state_dict`` reverses on restore. In contrast to the model state,
    the entries are opaque and independent of each other, so gathering is a plain collection of
    per-rank objects and scattering is a plain index lookup - nothing is split or merged.

    ..note::
        This is a collective over the ``data`` comm, so it has to be called by all ranks, not
        just by the rank which writes the checkpoint file. The states are plain (picklable)
        python objects: ``all_gather_object`` stages them through a byte tensor on the comm's
        default device itself, which for NCCL is the rank's current CUDA device.

    Returns None if the dataloader does not support checkpointing.
    """

    if not supports_dataloader_state(dataloader):
        return None

    state_dict = dataloader.state_dict()

    num_data_ranks = comm.get_size("data")
    if num_data_ranks == 1:
        return [state_dict]

    state_dicts = [None for _ in range(num_data_ranks)]
    dist.all_gather_object(state_dicts, state_dict, group=comm.get_group("data"))

    return state_dicts


def scatter_dataloader_state_dict(state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick this rank's entry out of the list stored by ``gather_dataloader_state_dict``."""

    num_data_ranks = comm.get_size("data")
    if len(state_dicts) != num_data_ranks:
        raise ValueError(
            f"Error, the checkpoint stores dataloader state for {len(state_dicts)} data-parallel ranks "
            f"but the current run has {num_data_ranks}. The data pipeline can only be restored into the "
            "same data-parallel decomposition it was stored from."
        )

    return state_dicts[comm.get_rank("data")]


def gather_model_state_dict(model: nn.Module, grads: Optional[bool] = False) -> OrderedDict:
    # precondition: every rank in the model group must issue the SAME ordered
    # sequence of gather_uneven calls, otherwise the all_gather inside
    # gather_uneven deadlocks (600s NCCL timeout instead of a readable error).
    if comm.get_size("model") > 1:
        plan = [
            f"{name}:{d}:{group}"
            for name, param in model.named_parameters()
            if hasattr(param, "sharded_dims_mp")
            for d, group in enumerate(param.sharded_dims_mp)
            if group is not None
        ]
        # deterministic hash (Python's hash() is per-process randomized)
        sig = zlib.crc32(("|".join(plan)).encode())
        device = next(model.parameters()).device
        sig_t = torch.tensor([sig], dtype=torch.int64, device=device)
        sig_list = [torch.empty_like(sig_t) for _ in range(comm.get_size("model"))]
        dist.all_gather(sig_list, sig_t, group=comm.get_group("model"))
        sigs = [int(s.item()) for s in sig_list]
        if len(set(sigs)) != 1:
            raise RuntimeError(
                f"[rank {comm.get_world_rank()}] checkpoint gather plan mismatch "
                f"across model group: per-rank crc32 = {sigs}. "
                f"This rank's plan has {len(plan)} gather ops."
            )

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

                    weight = split_tensor_along_dim(weight, dim=d, num_chunks=comm.get_size(group))[
                        comm.get_rank(group)
                    ]

                # update state dict
                state_dict[name] = weight

        elif strict:
            # TODO: maybe do at least a warning for non-strict mode
            raise ValueError(f"Missing key {name}")

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

    # The integer keys in state_dict["state"] follow torch.optim's own packing order:
    # parameters are numbered group-by-group, in optimizer.param_groups order (deduplicated).
    # We reconstruct the index -> param map from optimizer.param_groups so each state entry is
    # paired with the parameter that actually owns it. This is correct for any number of param
    # groups, unlike indexing by enumerate(model.parameters()), which only matches when there is
    # a single group built directly from model.parameters().
    optimizer_dict["state"] = {}
    seen = set()
    index = 0
    for pgroup in optimizer.param_groups:
        for param in pgroup["params"]:
            if id(param) in seen:
                continue
            seen.add(id(param))

            # params without optimizer state (e.g. never received a gradient) carry no state,
            # but still consume an index so we stay aligned with the packing
            if index not in state_dict["state"]:
                index += 1
                continue

            pstate = state_dict["state"][index]
            exp_avg = pstate["exp_avg"].clone()
            exp_avg_sq = pstate["exp_avg_sq"].clone()

            # if the parameter is sharded, gather its optimizer moments across all sharded dims
            if hasattr(param, "sharded_dims_mp"):
                for d, group in enumerate(param.sharded_dims_mp):
                    if group is not None:
                        exp_avg = gather_uneven(exp_avg, d, group)
                        exp_avg_sq = gather_uneven(exp_avg_sq, d, group)

            optimizer_dict["state"][index] = {
                "step": pstate["step"].clone(),
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
            }
            index += 1

    return optimizer_dict


def scatter_optimizer_state_dict(
    model: nn.Module, optimizer: Optimizer, optimizer_state_dict: OrderedDict
) -> OrderedDict():

    # some sanity checks
    # if optimizer is SGD, we can just return the local dict:
    if isinstance(optimizer, torch.optim.SGD):
        return optimizer_state_dict

    if not (isinstance(optimizer, torch.optim.Adam) or isinstance(optimizer, torch.optim.AdamW)):
        raise NotImplementedError(
            "Error, only Adam and AdamW state can be restored from flexible format at the moment."
        )

    # Reconstruct the index -> param map from optimizer.param_groups (matching torch.optim's
    # packing order) so each saved state entry is split with the sharding of its true owner,
    # for any number of parameter groups. See gather_optimizer_state_dict for details.
    seen = set()
    index = 0
    for pgroup in optimizer.param_groups:
        for param in pgroup["params"]:
            if id(param) in seen:
                continue
            seen.add(id(param))

            # in this case, we need to distribute the state
            if (index in optimizer_state_dict["state"]) and hasattr(param, "sharded_dims_mp"):

                # clone the state
                exp_avg = optimizer_state_dict["state"][index]["exp_avg"].clone()
                exp_avg_sq = optimizer_state_dict["state"][index]["exp_avg_sq"].clone()

                for d, group in enumerate(param.sharded_dims_mp):
                    # continue if there is nothing to do
                    if (group is None) or (comm.get_size(group) == 1):
                        continue

                    exp_avg = split_tensor_along_dim(exp_avg, dim=d, num_chunks=comm.get_size(group))[
                        comm.get_rank(group)
                    ]
                    exp_avg_sq = split_tensor_along_dim(exp_avg_sq, dim=d, num_chunks=comm.get_size(group))[
                        comm.get_rank(group)
                    ]

                # update the state dict
                optimizer_state_dict["state"][index]["exp_avg"] = exp_avg
                optimizer_state_dict["state"][index]["exp_avg_sq"] = exp_avg_sq

            index += 1

    return optimizer_state_dict


def get_model_state_dict_prefix(model: nn.Module) -> str:
    """Return the key prefix that the model's state_dict uses.

    torch.compile wraps a model in OptimizedModule (keys gain ``_orig_mod.``),
    and DDP wraps it in DistributedDataParallel (keys gain ``module.``).
    These wrappers may be nested in any combination, so we walk the wrapper
    chain and accumulate the prefix in order.
    """
    prefix = ""
    m = model
    while True:
        if hasattr(m, "_orig_mod"):
            prefix += "_orig_mod."
            m = m._orig_mod
        elif isinstance(m, nn.parallel.DistributedDataParallel):
            prefix += "module."
            m = m.module
        else:
            break
    return prefix


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
