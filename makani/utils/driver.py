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
import abc
import gc

from typing import Optional, Dict
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist

import logging
import wandb

# makani dependencies
from makani.utils.YParams import YParams
from makani.utils.features import get_auxiliary_channels
from makani.utils import comm
from makani.utils.dataloaders.data_helpers import get_data_normalization
from makani.utils.checkpoint_helpers import (
    gather_model_state_dict,
    scatter_model_state_dict,
    gather_optimizer_state_dict,
    scatter_optimizer_state_dict,
    prepend_prefix_to_state_dict,
)

# for flexible checkpoints
from physicsnemo.distributed.utils import split_tensor_along_dim


class Driver(metaclass=abc.ABCMeta):
    """
    Driver class acts as abstract base class for all derived training and inference classes

    The driver class sets up default parameters, logging infrastructure, wandb infrastructure, a single eval dataset
    """

    def _log_timers(self):
        print_prefix = "    "
        if self.timers and self.log_to_screen:
            self.logger.info("Initialization time breakdown:")
            for k,v in self.timers.items():
                self.logger.info(f"{print_prefix}{k} [s]: {v:.2f}")

    def __init__(self, params: YParams = None, world_rank: Optional[int] = 0, device: Optional[str] = None):
        # define timer dict
        self.timers = {}

        # update params
        self.params = self._set_default_parameters(params)

        # set up distributed communicators, even if it is a non-distributed instance
        self.world_rank = world_rank
        self.data_parallel_rank = comm.get_rank("data")

        # set the default device
        if device is not None:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                self.device = torch.device("cpu")

        # set the logger
        self.log_to_screen = self.params.log_to_screen if (hasattr(params, "log_to_screen") and params.log_to_screen) else False
        if self.log_to_screen:
            self.logger = logging.getLogger()

        # set wandb
        self.log_to_wandb = self.params.log_to_wandb if (hasattr(params, "log_to_wandb") and params.log_to_wandb) else False

    def __del__(self):
        if hasattr(self, "log_to_wandb") and self.log_to_wandb:
            wandb.finish()

    def _set_dataloader(self, params, path, train=False, device=None):
        # initialize data loader
        if params.log_to_screen:
            self.logger.info(f"Using channel names: {params.channel_names}")
            self.logger.info("initializing data loader")

    def _set_default_parameters(self, params):
        """
        Routine for updating parameters internally. This is intended to be the only place where the parameters are modified
        """

        if not hasattr(params, "gradient_accumulation_steps"):
            params["gradient_accumulation_steps"] = 1

        if not hasattr(params, "log_to_screen"):
            params["log_to_screen"] = False

        if not hasattr(params, "history_normalization_mode"):
            params["history_normalization_mode"] = "none"

        if not hasattr(params, "num_visualization_workers"):
            params["num_visualization_workers"] = 1

        if not hasattr(params, "log_video"):
            params["log_video"] = 0

        if not hasattr(params, "dump_weights_and_grads"):
            params["dump_weights_and_grads"] = 0

        # how to handle checkpoints
        if not hasattr(params, "load_checkpoint"):
            params["load_checkpoint"] = "legacy"

        if not hasattr(params, "save_checkpoint"):
            params["save_checkpoint"] = "legacy"

        if not hasattr(params, "load_optimizer"):
            params["load_optimizer"] = True

        if not hasattr(params, "load_scheduler"):
            params["load_scheduler"] = True

        if not hasattr(params, "load_counters"):
            params["load_counters"] = True

        if not  hasattr(params, "checkpoint_num_versions"):
            params["checkpoint_num_versions"] = 3

        return params

    def _set_data_shapes(self, params, dataset):
        """
        Routine for setting the shapes correctly
        """

        params.N_in_channels = len(dataset.in_channels)
        params.N_out_channels = len(dataset.out_channels)

        params.img_shape_x = dataset.img_shape_x
        params.img_shape_y = dataset.img_shape_y

        params.img_crop_shape_x = dataset.img_crop_shape_x
        params.img_crop_shape_y = dataset.img_crop_shape_y
        params.img_crop_offset_x = dataset.img_crop_offset_x
        params.img_crop_offset_y = dataset.img_crop_offset_y

        params.img_local_shape_x = dataset.img_local_shape_x
        params.img_local_shape_y = dataset.img_local_shape_y
        params.img_local_offset_x = dataset.img_local_offset_x
        params.img_local_offset_y = dataset.img_local_offset_y

        # derived quantities
        params["N_in_predicted_channels"] = params.N_in_channels

        # sanitization:
        params["add_zenith"] = params.get("add_zenith", False)

        # input channels
        # zenith channel is appended to all the samples, so we need to do it here
        params["N_dynamic_channels"] = 0
        if params.add_zenith:
            params.N_dynamic_channels += 1

        params.n_noise_chan = 0
        if hasattr(params, "input_noise"):
            if params.input_noise["mode"] == "concatenate":
                if "n_channels" in params.input_noise:
                    params.n_noise_chan = params.input_noise["n_channels"]
                else:
                    params.n_noise_chan = 1
        params.N_dynamic_channels += params.n_noise_chan

        # initialize static channels
        params["N_static_channels"] = 0

        # these are static and the same for all samples in the same time history
        if params.get("add_grid", False):
            n_grid_chan = 2
            gridtype = params.get("gridtype", "sinusoidal")
            if gridtype == "sinusoidal":
                n_grid_chan *= 2 * params.get("grid_num_frequencies", 1)

            params.N_static_channels += n_grid_chan

        if params.get("add_orography", False):
            params.N_static_channels += 1

        if params.get("add_landmask", False):
            landmask_preprocessing = params.get("landmask_preprocessing", "floor")
            if landmask_preprocessing == "raw":
                params.N_static_channels += 1
            elif landmask_preprocessing in ["round", "floor"]:
                params.N_static_channels += 2

        if params.get("add_soiltype", False):
            params.N_static_channels += 8

        # update input channels withj the dynamic channels
        params.N_in_channels += params.N_dynamic_channels

        # dynamic channels are replicated at each step
        if params.n_history >= 1:
            params.N_in_channels = (params.n_history + 1) * params.N_in_channels
            params.N_in_predicted_channels *= params.n_history + 1

        # update input channels with the static channels
        params.N_in_channels += params.N_static_channels

        # get names of additional channels
        params["aux_channel_names"] = get_auxiliary_channels(**params.to_dict())

        # target channels
        params.N_target_channels = (params.n_future + 1) * params.N_out_channels

    def _init_wandb(self, params, job_type):
        """
        Convenience routine for setting up wandb
        """

        # set up wandb logging
        if self.log_to_wandb:
            # login first:
            wandb.login(anonymous="never")

            # check if we want to resume or not
            if not params.resuming:
                # generate run id
                params["wandb_run_id"] = wandb.util.generate_id()

                # create a lost of tags:
                # paralellism:
                tags = [f"ngpu{comm.get_world_size()}", f'mp{comm.get_size("model")}', f'sp{comm.get_size("spatial")}']

                # initialize wandb
                self.wandb_run = wandb.init(
                    dir=params.wandb_dir,
                    job_type=job_type,
                    config=params,
                    name=params.wandb_name,
                    group=params.wandb_group,
                    project=params.wandb_project,
                    entity=params.wandb_entity,
                    tags=tags,
                    id=params["wandb_run_id"],
                )

                # store params in wandb folder
                params.to_yaml(os.path.join(params.wandb_dir, "wandb", "makani_restart.yaml"), overwrite=True)
            else:
                # retrieve run id from wandb config file:
                # wandb_config = YParams(os.path.join(params.wandb_dir, "wandb", "latest-run", "files", "config.yaml"), "params")
                # params["wandb_run_id"] = wandb_config["value"]["wandb_run_id"]
                tmpparams = YParams(os.path.join(params.wandb_dir, "wandb", "makani_restart.yaml"))
                params["wandb_run_id"] = tmpparams["wandb_run_id"]

                # initialize wandb: resume=must is super strict
                # but its better to fail than doing the wrong thing silently
                self.wandb_run = wandb.init(dir=params.wandb_dir, project=params.wandb_project, entity=params.wandb_entity, id=params["wandb_run_id"], resume="must")

            # create wandb dataset artifact
            if hasattr(params, "dataset"):
                # try using, otherwise create it:
                dataset_string = params["dataset"]["name"]
                # truncate to 128-7-1=120 characters
                if len(dataset_string) >= 120:
                    dataset_string = dataset_string[:120]
                dataset_tag = dataset_string + ":latest"

                if not wandb.run.offline:
                    api = wandb.Api()
                    if api.artifact_collection_exists(dataset_string, type="dataset"):
                        # try using existing dataset
                        self.wandb_dataset = wandb.use_artifact(dataset_tag, type="dataset")
                        print(f"Using dataset artifact {dataset_tag}")
                    else:
                        # create new one if it does not exist
                        self.wandb_dataset = wandb.Artifact(name=dataset_string, description=params["dataset"]["description"], type="dataset")
                        self.wandb_dataset.add_file(params["dataset"]["metadata_file"], name="metadata")
                        wandb.log_artifact(self.wandb_dataset)
                        print(f"Creating artifact {dataset_string}")

            # create data normalization artifact
            if hasattr(params, "normalization"):

                if hasattr(params, "dataset"):
                    norm_string = params["dataset"]["name"] + "_"
                else:
                    norm_string = ""

                # generate name string
                if isinstance(params.normalization, dict):
                    norm_string += "zscore_minmax_" + "_".join([f"{k}-{v}" for k, v in params.normalization.items()])
                else:
                    norm_string += params.normalization
                # truncate
                if len(norm_string) >= 120:
                    norm_string = norm_string[:120]
                norm_tag = norm_string + ":latest"

                if not wandb.run.offline:
                    api = wandb.Api()
                    if api.artifact_collection_exists(norm_string, type="dataset_normalization"):
                        # try using existing normalization
                        self.wandb_normalization = wandb.use_artifact(norm_tag, type="dataset_normalization")
                        print(f"Using normalization artifact {norm_tag}")
                    else:
                        # create normalization artifact
                        self.wandb_normalization = wandb.Artifact(name=norm_string, description="data normalization", type="dataset_normalization")
                        bias, scale = get_data_normalization(params)
                        # filter only used channels
                        bias = bias.flatten()[params.in_channels]
                        scale = scale.flatten()[params.in_channels]
                        data = np.stack([scale, bias], axis=0).tolist()
                        data[0].insert(0, "scale")
                        data[1].insert(0, "bias")
                        print(len(data[0]), len(data[1]))
                        # create columns
                        columns = ["type"] + params.channel_names
                        # create table
                        tab = wandb.Table(columns=columns, data=data)
                        self.wandb_normalization.add(tab, name="data")
                        wandb.log_artifact(self.wandb_normalization)
                        print(f"Creating artifact {norm_string}")

    @staticmethod
    def restore_from_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
        checkpoint_mode: str = "legacy",
        strict: bool = True,
    ):
        """
        Routine for restoring a checkpoint from a path.
        """
        with torch.no_grad():
            if checkpoint_mode == "legacy":
                # legacy mode
                Driver._restore_checkpoint_legacy(checkpoint_path, model, loss, optimizer, scheduler, counters, strict=strict)
            elif checkpoint_mode == "flexible":
                # new flexible mode allows to load models in arbitrary model-parallel configurations
                Driver._restore_checkpoint_flexible(checkpoint_path, model, loss, optimizer, scheduler, counters, strict=strict)
            else:
                raise ValueError(f"Unknown checkoint mode {checkpoint_mode}.")

        # clean up
        gc.collect()

        return

    @staticmethod
    def _restore_checkpoint_legacy(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
        strict: bool = True,
        validate_comms: bool = True,
    ):
        checkpoint_fname = checkpoint_path.format(mp_rank=comm.get_rank("model"))
        checkpoint = torch.load(checkpoint_fname, map_location="cpu", weights_only=False)

        # check compatibility of the comm grid stored inside the file
        if validate_comms:
            # load the comm dict
            if "comm_grid" not in checkpoint:
                from warnings import warn

                warn(
                    "It is highly recommended to upgrade the checkpointing format to include model parallel comm grid information, so that a correct restoring can be guaranteed. This can be achieved by loading and saving a model with a newer version of makani. In future versions of makani, this warning will become an error.",
                    DeprecationWarning,
                )
            else:
                comm_dict = checkpoint["comm_grid"]
                comm_names = comm.get_model_comm_names()
                for cname in comm_names:
                    # check comm
                    if cname not in comm_dict.keys():
                        raise RuntimeError(f"Error, communicator name {cname} not found in communicator information stored in file, but present in the current comm table.")
                    # check size
                    if comm.get_size(cname) != comm_dict[cname]["size"]:
                        raise RuntimeError(f"Error, communicator {cname} has size {comm.get_size(cname)}, but expected size {comm_dict[cname]['size']}")
                    # check rank
                    if comm.get_rank(cname) != comm_dict[cname]["rank"]:
                        raise RuntimeError(f"Error, communicator {cname} rank {comm.get_rank(cname)} is trying to load a file from rank {comm_dict[cname]['rank']}")

        # if all those test pass, we are good to go
        # this is reworked to avoid loading modules related to the SHT
        state_dict = checkpoint["model_state"]
        if isinstance(model, nn.parallel.DistributedDataParallel):
            # prepend module prefix to state dict:
            prepend_prefix_to_state_dict(state_dict, "module.")

        # load state dict
        model.load_state_dict(state_dict, strict=strict)

        # the loss is also restored in the case that it has a state
        if loss is not None:
            loss.load_state_dict(checkpoint["loss_state_dict"])

        # If finetuning, restore checkpoint does not load optimizer state, instead uses config specified lr.
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if counters is not None:
            counters["iters"] = checkpoint["iters"]
            counters["start_epoch"] = checkpoint["epoch"]

        return

    @staticmethod
    def _restore_checkpoint_flexible(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
        strict: bool = True,
    ):
        # when loading the weights in flexble mode we exclusively use mp_rank=0 and load them onto the cpu
        checkpoint_fname = checkpoint_path.format(mp_rank=0)
        checkpoint = torch.load(checkpoint_fname, map_location="cpu", weights_only=False)

        # this is reworked to avoid loading modules related to the SHT
        state_dict = checkpoint["model_state"]

        if isinstance(model, nn.parallel.DistributedDataParallel):
            # prepend module prefix to state dict:
            prepend_prefix_to_state_dict(state_dict, "module.")

        if comm.get_size("model") > 1:
            state_dict = scatter_model_state_dict(model, state_dict, strict)

        # load state dict
        model.load_state_dict(state_dict, strict=strict)

        # the loss is also restored in the case that it has a state
        if loss is not None:
            loss.load_state_dict(checkpoint["loss_state_dict"])

        # If finetuning, restore checkpoint does not load optimizer state, instead uses config specified lr.
        if optimizer is not None:
            if comm.get_size("model") > 1:
                checkpoint["optimizer_state_dict"] = scatter_optimizer_state_dict(model, optimizer, checkpoint["optimizer_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if counters is not None:
            counters["iters"] = checkpoint["iters"]
            counters["start_epoch"] = checkpoint["epoch"]

        return

    @staticmethod
    def save_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
        checkpoint_mode: str = "legacy",
    ):
        """
        Save out checkpoint
        """
        with torch.no_grad():
            # legacy mode
            if checkpoint_mode == "legacy":
                Driver._save_checkpoint_legacy(checkpoint_path, model, loss, optimizer, scheduler, counters)
            elif checkpoint_mode == "flexible":
                Driver._save_checkpoint_flexible(checkpoint_path, model, loss, optimizer, scheduler, counters)
            else:
                raise ValueError(f"Unknown checkoint mode {checkpoint_mode}.")

        # clean up
        gc.collect()

        return

    @staticmethod
    def _save_checkpoint_legacy(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
    ):
        # maybe the logic regarding the mp rank should be moved to somewhere else?
        checkpoint_fname = checkpoint_path.format(mp_rank=comm.get_rank("model"))

        # attach sharding information to model state:
        state_dict = model.state_dict()

        # drop module prefix in case if DDP is being used
        if isinstance(model, nn.parallel.DistributedDataParallel):
            nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")

        # check for model parallelism
        for param, sk in zip(model.parameters(), state_dict.keys()):
            if hasattr(param, "sharded_dims_mp"):
                state_dict[sk].sharded_dims_mp = param.sharded_dims_mp

        # add model state dict to store dict
        store_dict = {"model_state": state_dict}

        # comm infrastructure:
        comm_names = comm.get_model_comm_names()
        comm_dict = OrderedDict()
        for cname in comm_names:
            rank = comm.get_rank(cname)
            size = comm.get_size(cname)
            comm_dict[cname] = {"size": size, "rank": rank}
        store_dict["comm_grid"] = comm_dict

        if loss is not None:
            store_dict["loss_state_dict"] = loss.state_dict()

        if optimizer is not None:
            store_dict["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            store_dict["scheduler_state_dict"] = scheduler.state_dict()

        if counters is not None:
            store_dict["iters"] = counters["iters"]
            store_dict["epoch"] = counters["epoch"]

        torch.save(store_dict, checkpoint_fname)

        return

    @staticmethod
    def _save_checkpoint_flexible(
        checkpoint_path: str,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
        counters: Optional[Dict[str, int]] = None,
    ):
        # checkpoint name
        checkpoint_fname = checkpoint_path.format(mp_rank=0)

        # iterate over parameters and gather them from the ranks
        if comm.get_size("model") > 1:
            state_dict = gather_model_state_dict(model)
        else:
            state_dict = model.state_dict()

        # drop module prefix in case if DDP is being used
        if isinstance(model, nn.parallel.DistributedDataParallel):
            nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")

        store_dict = {"model_state": state_dict}

        if loss is not None:
            store_dict["loss_state_dict"] = loss.state_dict()

        if optimizer is not None:
            if comm.get_size("model") > 1:
                store_dict["optimizer_state_dict"] = gather_optimizer_state_dict(model, optimizer)
            else:
                store_dict["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            store_dict["scheduler_state_dict"] = scheduler.state_dict()

        if counters is not None:
            store_dict["iters"] = counters["iters"]
            store_dict["epoch"] = counters["epoch"]

        # in flexible mode only rank 0 needs to save the data to disk
        if comm.get_world_rank() == 0:
            torch.save(store_dict, checkpoint_fname)

        return

    @staticmethod
    def dump_weights_and_grads(weights_and_grads_path: str, model: nn.Module, step: int = 0):
        """
        Helper routine intended for debugging purposes to dump weights and grads
        """

        mp_rank = comm.get_rank("model")
        weights_and_grads_fname = os.path.join(weights_and_grads_path, f"weights_and_grads_step{step}_mp{mp_rank}.tar")

        weights_dict = {k: v for k, v in model.named_parameters()}
        grad_dict = {k: v.grad for k, v in model.named_parameters()}

        store_dict = {"step": step, "grads": grad_dict, "weights": weights_dict}
        torch.save(store_dict, weights_and_grads_fname)

    # TODO: would be nice to convert this to static methods
    def get_optimizer(self, model, params):
        """
        Convenience routine for setting up the optimizer
        """

        # optimizer setup
        betas = (params.optimizer_beta1, params.optimizer_beta2)
        all_parameters = model.parameters()
        if params.optimizer_type == "Adam":
            if self.log_to_screen:
                self.logger.info("using Adam optimizer")
            optimizer = optim.Adam(all_parameters, betas=betas, lr=params.get("lr", 1e-3), weight_decay=params.get("weight_decay", 0), foreach=True)
        elif params.optimizer_type == "AdamW":
            if self.log_to_screen:
                self.logger.info("using AdamW optimizer")
            optimizer = optim.AdamW(all_parameters, betas=betas, lr=params.get("lr", 1e-3), weight_decay=params.get("weight_decay", 0), foreach=True)
        elif params.optimizer_type == "SGD":
            if self.log_to_screen:
                self.logger.info("using SGD optimizer")
            optimizer = optim.SGD(all_parameters, lr=params.get("lr", 1e-3), weight_decay=params.get("weight_decay", 0), momentum=params.get("momentum", 0), foreach=True)
        elif params.optimizer_type == "SIRFShampoo":
            if self.log_to_screen:
                self.logger.info("using SIRFShampoo optimizer")
            from sirfshampoo import SIRFShampoo

            optimizer = SIRFShampoo(model, lr=params.get("lr", 1e-3))
        else:
            raise ValueError(f"Unknown optimizer type {params.optimizer_type}")

        return optimizer

    # TODO: would be nice to convert this to static methods
    def get_scheduler(self, optimizer, params):
        """Convenience routine for setting up the scheduler"""

        if params.scheduler == "ReduceLROnPlateau":
            if not hasattr(params, "scheduler_mode"):
                params["scheduler_mode"] = "min"
            if params.get("skip_validation", False):
                raise ValueError(f"Error, you cannot skip validation when using ReduceLROnPlateau scheduler.")
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=params.scheduler_factor, patience=params.scheduler_patience, mode=params.scheduler_mode)
        elif params.scheduler == "StepLR":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=params.scheduler_step_size, gamma=params.scheduler_gamma)
        elif params.scheduler == "CosineAnnealingLR":
            if not hasattr(params, "scheduler_min_lr"):
                params["scheduler_min_lr"] = 0.0
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.scheduler_T_max, eta_min=params.scheduler_min_lr)
        elif params.scheduler == "OneCycleLR":
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=params.lr, total_steps=params.scheduler_T_max, steps_per_epoch=1)
        else:
            scheduler = None

        # warmup scheduler
        if params.lr_warmup_steps > 0:
            if params.scheduler == "ReduceLROnPlateau":
                raise NotImplementedError("Error, warmup scheduler not implemented for ReduceLROnPlateau scheduler")
            warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=params.lr_start, end_factor=1.0, total_iters=params.lr_warmup_steps)

            scheduler = lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, scheduler], milestones=[params.lr_warmup_steps])

        return scheduler
