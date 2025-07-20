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
import sys
import gc
import time
from typing import Optional
import numpy as np
from tqdm import tqdm

# gpu info
import pynvml

# torch
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.amp as amp
import torch.distributed as dist

import logging
import wandb

# makani depenedencies
from makani.utils import LossHandler, MetricsHandler
from makani.utils.driver import Driver
from makani.utils.training import Trainer
from makani.utils.dataloader import get_dataloader
from makani.utils.dataloaders.data_helpers import get_climatology
from makani.utils.YParams import YParams

# model registry
from makani.models import model_registry

# distributed computing stuff
from makani.utils import comm
from makani.utils import visualize

from makani.mpu.mappings import init_gradient_reduction_hooks
from makani.mpu.helpers import sync_params, gather_uneven

# for counting model parameters
from makani.models.helpers import count_parameters
from physicsnemo.distributed.mappings import reduce_from_parallel_region

# checkpoint helpers
from makani.utils.checkpoint_helpers import get_latest_checkpoint_version

# weight normalizing helper
from makani.utils.training.training_helpers import normalize_weights, clip_grads


class StochasticTrainer(Driver):
    """
    Trainer class holding all the necessary information to perform training.
    """

    def __init__(self, params: Optional[YParams] = None, world_rank: Optional[int] = 0, device: Optional[str] = None):
        super().__init__(params, world_rank, device)

        if self.log_to_screen:
            self.logger.warning("using StochasticTrainer. This trainer is largely untested. Proceed with caution.")

        # init wandb
        if self.log_to_wandb:
            self._init_wandb(self.params, job_type="stochastic")

        # set checkpoint version: start at -1 so that first version which is written is 0
        self.checkpoint_version_current = -1

        # init nccl: do a single AR to make sure that SHARP locks
        # on to the right tree, and that barriers can be used etc
        if dist.is_initialized():
            tens = torch.ones(1, device=self.device)
            dist.all_reduce(tens, group=comm.get_group("data"))

        # nvml stuff
        if self.log_to_screen:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        # set amp_parameters
        if hasattr(self.params, "amp_mode") and (self.params.amp_mode != "none"):
            self.amp_enabled = True
            if self.params.amp_mode == "fp16":
                self.amp_dtype = torch.float16
            elif self.params.amp_mode == "bf16":
                self.amp_dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown amp mode {self.params.amp_mode}")

            if self.log_to_screen:
                self.logger.info(f"Enabling automatic mixed precision in {self.params.amp_mode}.")
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        # initialize data loader
        if self.log_to_screen:
            self.logger.info(f"Using channel names: {self.params.channel_names}")
            self.logger.info("initializing data loader")
        self.train_dataloader, self.train_dataset, self.train_sampler = get_dataloader(self.params, self.params.train_data_path, mode="train", device=self.device)
        self.valid_dataloader, self.valid_dataset, self.valid_sampler = get_dataloader(self.params, self.params.valid_data_path, mode="eval", device=self.device)
        self._set_data_shapes(self.params, self.valid_dataset)

        if self.log_to_screen:
            self.logger.info("data loader initialized")

        # record data required to reproduce workflow using a model package
        if self.world_rank == 0:
            from makani.models.model_package import save_model_package

            save_model_package(self.params)

        # init preprocessor and model
        self.model = model_registry.get_model(self.params, use_stochastic_interpolation=True).to(self.device)
        self.preprocessor = self.model.preprocessor

        # print aux channel names:
        if self.log_to_screen:
            self.logger.info(f"Auxiliary channel names: {self.params.aux_channel_names}")

        # if model-parallelism is enabled, we need to sure that shared weights are matching across ranks
        # as random seeds might get out of sync during initialization
        # DEBUG: this also needs to be fixed in NCCL
        # if comm.get_size("model") > 1:
        sync_params(self.model, mode="broadcast")

        # add a barrier here, just to make sure
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # define process group for DDP, we might need to override that
        if dist.is_initialized() and not self.params.disable_ddp:
            ddp_process_group = comm.get_group("data")

        # log gradients to wandb
        if self.log_to_wandb:
            wandb.watch(self.model, log="all")

        # print model
        if self.log_to_screen:
            self.logger.info(f"\n{self.model}")

        # metrics handler
        clim = get_climatology(self.params)
        clim = torch.from_numpy(clim).to(torch.float32)
        ens_var_names = ["u10m", "t2m", "u500", "z500", "q500", "sp"] if (self.params.ensemble_size > 1) else []
        rollout_length = params.get("valid_autoreg_steps", 0) + 1
        self.metrics = MetricsHandler(params=self.params, climatology=clim, num_rollout_steps=rollout_length, device=self.device, crps_var_names=ens_var_names, spread_var_names=ens_var_names, ssr_var_names=ens_var_names)
        self.metrics.initialize_buffers()

        # loss handler
        self.loss_obj = LossHandler(self.params)
        self.loss_obj = self.loss_obj.to(self.device)

        # optimizer and scheduler setup
        self.optimizer = self.get_optimizer(self.model, self.params)
        self.scheduler = self.get_scheduler(self.optimizer, self.params)

        # gradient scaler
        self.gscaler = amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

        # weight normalization
        self.normalize_weights = self.params.get("normalize_weights", False)

        # gradient clipping
        self.max_grad_norm = self.params.get("optimizer_max_grad_norm", -1.0)

        # we need this further down
        capture_stream = None
        if dist.is_initialized() and not self.params.disable_ddp:
            if self.device.type == "cuda":
                capture_stream = torch.Stream(device="cuda")

            with torch.cuda.stream(capture_stream):
                self.model = init_gradient_reduction_hooks(
                    self.model,
                    device=self.device,
                    reduction_buffer_count=self.params.parameters_reduction_buffer_count,
                    broadcast_buffers=False,
                    find_unused_parameters=self.params["enable_grad_anomaly_detection"],
                    gradient_as_bucket_view=True,
                    static_graph=False,
                    verbose=True,
                )

            # capture stream sync
            if capture_stream is not None:
                capture_stream.synchronize()

        # lets get one sample from the dataloader:
        # set to train just to be safe
        self._set_train()
        # get sample and map to gpu
        iterator = iter(self.train_dataloader)
        data = next(iterator)
        gdata = map(lambda x: x.to(self.device), data)
        # extract unpredicted features
        inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
        # flatten
        inp = self.preprocessor.flatten_history(inp)
        tar = self.preprocessor.flatten_history(tar)
        # get shapes
        inp_shape = inp.shape
        tar_shape = tar.shape

        self._compile_model(inp_shape)

        # visualization wrapper:
        # plot_list = [{"name": "windspeed_uv10", "functor": "lambda x: np.sqrt(np.square(x[0, ...]) + np.square(x[1, ...]))", "diverging": False}]
        plot_list = [{"name": "geopotential_z500", "functor": "lambda x: x[0, ...]", "diverging": False}]
        out_bias, out_scale = self.train_dataloader.get_output_normalization()
        self.visualizer = visualize.VisualizationWrapper(
            self.params.log_to_wandb,
            path=None,
            prefix=None,
            plot_list=plot_list,
            lat=np.deg2rad(np.array(self.valid_dataloader.lat_lon[0])),
            lon=np.deg2rad(np.array(self.valid_dataloader.lat_lon[1])) - np.pi,
            scale=out_scale[0, ...],
            bias=out_bias[0, ...],
            num_workers=self.params.num_visualization_workers,
        )
        # allocate pinned tensors for faster copy:
        if self.device.type == "cuda":
            self.viz_stream = torch.Stream(device="cuda")
        else:
            self.viz_stream = None

        pin_memory = self.device.type == "cuda"
        self.viz_prediction_cpu = torch.empty(
            ((self.params.N_target_channels // (self.params.n_future + 1)), self.params.img_crop_shape_x, self.params.img_crop_shape_y), device="cpu", pin_memory=pin_memory
        )
        self.viz_target_cpu = torch.empty(
            ((self.params.N_target_channels // (self.params.n_future + 1)), self.params.img_crop_shape_x, self.params.img_crop_shape_y), device="cpu", pin_memory=pin_memory
        )

        # reload checkpoints
        counters = {"iters": 0, "start_epoch": 0}
        if self.params.pretrained and not self.params.resuming:
            if not self.params.is_set("pretrained_checkpoint_path"):
                raise ValueError("Error, please specify a valid pretrained checkpoint path")

            # use specified checkpoint
            checkpoint_path = self.params.pretrained_checkpoint_path

            if self.log_to_screen:
                self.logger.info(f"Loading pretrained checkpoint {checkpoint_path} in {self.params.load_checkpoint} mode")

            self.restore_from_checkpoint(
                checkpoint_path,
                model=self.model,
                loss=self.loss_obj if self.params.get("load_loss", True) else None,
                optimizer=self.optimizer if self.params.get("load_optimizer", True) else None,
                scheduler=self.scheduler if self.params.get("load_scheduler", True) else None,
                counters=counters if self.params.get("load_counters", True) else None,
                checkpoint_mode=self.params.load_checkpoint,
                strict=self.params.get("strict_restore", True),
            )

            # override learning rate - useful when restoring optimizer but want to override the LR
            if self.params.get("override_lr", False):
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.params.get("lr", 1e-3)

        if self.params.resuming:

            # find latest checkpoint
            checkpoint_path = self.params.checkpoint_path
            self.checkpoint_version_current = get_latest_checkpoint_version(checkpoint_path)
            checkpoint_path = checkpoint_path.format(checkpoint_version=self.checkpoint_version_current, mp_rank="{mp_rank}")

            if self.log_to_screen:
                self.logger.info(f"Resuming from checkpoint {checkpoint_path} in {self.params.load_checkpoint} mode")

            self.restore_from_checkpoint(
                checkpoint_path,
                model=self.model,
                loss=self.loss_obj if self.params.get("load_loss", True) else None,
                optimizer=self.optimizer if self.params.get("load_optimizer", True) else None,
                scheduler=self.scheduler if self.params.get("load_scheduler", True) else None,
                counters=counters if self.params.get("load_counters", True) else None,
                checkpoint_mode=self.params.load_checkpoint,
                strict=self.params.get("strict_restore", True),
            )

        # read out counters correctly
        self.iters = counters["iters"]
        self.start_epoch = counters["start_epoch"]
        self.epoch = self.start_epoch

        # wait till everybody is ready
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # counting runs a reduction so we need to count on all ranks before printing on rank 0
        pcount, _, _ = count_parameters(self.model, self.device)
        if self.log_to_screen:
            self.logger.info("Number of trainable model parameters: {}".format(pcount))

    # jit stuff
    def _compile_model(self, inp_shape):

        if self.params.jit_mode == "inductor":
            self.model = torch.compile(self.model)
            self.model_train = self.model
            self.model_eval = self.model

        else:
            self.model_train = self.model
            self.model_eval = self.model

        return

    def _set_train(self):
        self.model.train()
        self.loss_obj.train()
        self.preprocessor.train()

    def _set_eval(self):
        self.model.eval()
        self.loss_obj.eval()
        self.preprocessor.eval()

    def train(self):
        # log parameters
        if self.log_to_screen:
            # log memory usage so far
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024.0 * 1024.0 * 1024.0)
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (1024.0 * 1024.0 * 1024.0)
            self.logger.info(f"Scaffolding memory high watermark: {all_mem_gb} GB ({max_mem_gb} GB for pytorch)")
            # announce training start
            self.logger.info("Starting Training Loop...")

        # perform a barrier here to make sure everybody is ready
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except ValueError:
            pass

        training_start = time.time()
        best_valid_loss = 1.0e6
        for epoch in range(self.start_epoch, self.params.max_epochs):
            if dist.is_initialized():
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                if self.valid_sampler is not None:
                    self.valid_sampler.set_epoch(epoch)

            # start timer
            epoch_start = time.time()

            # train if not to be skipped
            if not self.params.get("skip_training", False):
                train_time, train_data_gb, train_logs = self.train_one_epoch()
            else:
                train_time = 0
                train_data_gb = 0
                train_logs = {"train_steps" : 0, "loss" : 0.0}

            # validate if not to be skipped
            if not self.params.get("skip_validation", False):
                valid_time, viz_time, valid_logs = self.validate_one_epoch(epoch)
            else:
                valid_time = 0
                viz_time = 0
                valid_logs = {"base": {}, "metrics": {}}

            if self.params.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(valid_logs["base"]["validation loss"])
            elif self.scheduler is not None:
                self.scheduler.step()

            # log learning rate
            if self.log_to_wandb:
                for param_group in self.optimizer.param_groups:
                    lr = param_group["lr"]
                wandb.log({"learning rate": lr}, step=self.epoch)

            # save out checkpoints
            if (self.data_parallel_rank == 0) and (self.params.save_checkpoint != "none") and not self.params.get("skip_training", False):
                store_start = time.time()
                checkpoint_mode = self.params["save_checkpoint"]
                counters = {"iters": self.iters, "epoch": self.epoch}

                # increase checkpoint counter
                self.checkpoint_version_current = (self.checkpoint_version_current + 1) % self.params.checkpoint_num_versions
                checkpoint_path = self.params.checkpoint_path.format(checkpoint_version=self.checkpoint_version_current, mp_rank="{mp_rank}")

                # checkpoint at the end of every epoch
                self.save_checkpoint(checkpoint_path, self.model, self.loss_obj, self.optimizer, self.scheduler, counters, checkpoint_mode=checkpoint_mode)

                # save best checkpoint
                best_checkpoint_path = self.params.best_checkpoint_path.format(mp_rank=comm.get_rank("model"))
                best_checkpoint_saved = os.path.isfile(best_checkpoint_path)
                if (not self.params.get("skip_validation", False)) and ((not best_checkpoint_saved) or (valid_logs["base"]["validation loss"] <= best_valid_loss)):
                    self.save_checkpoint(self.params.best_checkpoint_path, self.model, self.loss_obj, self.optimizer, self.scheduler, counters, checkpoint_mode=checkpoint_mode)
                    best_valid_loss = valid_logs["base"]["validation loss"]

                # time how long it took
                store_stop = time.time()

                if self.log_to_screen:
                    self.logger.info(f"Saving checkpoint ({checkpoint_mode}) took: {(store_stop - store_start):.2f} sec")

            # wait for everybody
            if dist.is_initialized():
                dist.barrier(device_ids=[self.device.index])

            # end timer
            epoch_end = time.time()

            # create timing logs:
            timing_logs = {
                "epoch time [s]": epoch_end - epoch_start,
                "training time [s]": train_time,
                "validation time [s]": valid_time,
                "visualization time [s]": viz_time,
                "training step time [ms]": train_logs["train_steps"] and (train_time / train_logs["train_steps"]) * 10**3 or 0,
                "minimal IO rate [GB/s]": train_time and train_data_gb / train_time or 0,
            }

            # log metrics:
            self.log_epoch(train_logs, valid_logs, timing_logs)

            # exit here if not training:
            if self.params.get("skip_training", False):
                break

        # training done
        training_end = time.time()
        if self.log_to_screen:
            self.logger.info("Total training time is {:.2f} sec".format(training_end - training_start))

        return

    def train_one_epoch(self):
        self.epoch += 1
        total_data_bytes = 0
        self._set_train()

        # perform weight normalization if requested:
        # do it before training and then repeat after each training
        # step
        if self.normalize_weights:
            with torch.no_grad():
                normalize_weights(self.model, eps=1e-4)

        # we need this for the loss average
        accumulated_loss = torch.zeros((2), dtype=torch.float32, device=self.device)

        train_steps = 0
        train_start = time.perf_counter_ns()
        self.model_train.zero_grad(set_to_none=True)
        for data in tqdm(self.train_dataloader, desc=f"Training progress epoch {self.epoch}", disable=not self.log_to_screen):
            train_steps += 1
            self.iters += 1

            # map to device
            gdata = map(lambda x: x.to(self.device), data)

            # do preprocessing
            inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)

            # flatten the history
            inp = self.preprocessor.flatten_history(inp)
            tar = self.preprocessor.flatten_history(tar)

            # assuming float32
            total_data_bytes += inp.nbytes + tar.nbytes

            # check if we need to perform an update
            do_update = (train_steps % self.params["gradient_accumulation_steps"] == 0)
            loss_scaling_fact = 1.0
            if self.params["gradient_accumulation_steps"] > 1:
                loss_scaling_fact = 1.0 / np.float32(self.params["gradient_accumulation_steps"])

            with amp.autocast(device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                if do_update:
                    pred, tar = self.model_train(inp, tar, n_samples=self.params.stochastic_size)
                    loss = self.loss_obj(pred, tar)
                else:
                    with self.model_train.no_sync():
                        pred, tar = self.model_train(inp, tar, n_samples=self.params.stochastic_size)
                        loss = self.loss_obj(pred, tar)
                loss = loss * loss_scaling_fact

            self.gscaler.scale(loss).backward()

            # increment accumulated loss
            accumulated_loss[0] += loss.detach().clone() * inp.shape[0]
            accumulated_loss[1] += inp.shape[0]

            # gradient clipping
            if do_update:
                if self.max_grad_norm > 0.0:
                    self.gscaler.unscale_(self.optimizer)
                    clip_grads(self.model_train, self.max_grad_norm)

                # perform weight update
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
                self.model_train.zero_grad(set_to_none=True)

                # perform weight normalization if requested: only required if weights have changed
                if self.normalize_weights:
                    with torch.no_grad():
                        normalize_weights(self.model_train, eps=1e-4)

            if (self.params.print_timings_frequency > 0) and (self.iters % self.params.print_timings_frequency == 0) and self.log_to_screen:
                running_train_time = time.perf_counter_ns() - train_start
                print(f"Average step time after step {self.iters}: {running_train_time / float(train_steps) * 10**(-6):.1f} ms")
                print(
                    f"Average effective io rate after step {self.iters}: {total_data_bytes * float(comm.get_world_size()) / (float(running_train_time) * 10**(-9) * 1024. * 1024. * 1024.):.2f} GB/s"
                )
                print(f"Current loss {loss.item()}")

            # if logging of weights and grads during training is enabled, write them out at the first step of each epoch
            if (self.params.dump_weights_and_grads > 0) and ((self.iters - 1) % self.params.dump_weights_and_grads == 0):
                weights_and_grads_path = self.params["experiment_dir"]
                if self.log_to_screen:
                    self.logger.info(f"Dumping weights and gradients to {weights_and_grads_path}")
                self.dump_weights_and_grads(weights_and_grads_path, self.model, step=(self.epoch * self.params.num_samples_per_epoch + self.iters))

        # average the loss over ranks and steps
        if dist.is_initialized():
            dist.all_reduce(accumulated_loss, op=dist.ReduceOp.SUM, group=comm.get_group("data"))

        # add the train loss to logs
        train_loss = accumulated_loss[0] / (accumulated_loss[1] * loss_scaling_fact)
        logs = {"loss": train_loss.item()}

        # add train steps to log
        logs["train_steps"] = train_steps

        # global sync is in order
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # finalize timers
        train_end = time.perf_counter_ns()
        train_time = (train_end - train_start) * 10 ** (-9)
        total_data_gb = (total_data_bytes / (1024.0 * 1024.0 * 1024.0)) * float(comm.get_world_size())

        return train_time, total_data_gb, logs

    def validate_one_epoch(self, epoch):
        # set to eval
        self._set_eval()

        # clear cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # synchronize
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # initialize metrics buffers
        self.metrics.zero_buffers()

        visualize = self.params.log_video and (epoch % self.params.log_video == 0)

        # start the timer
        valid_start = time.time()

        with torch.inference_mode():
            with torch.no_grad():

                # normalize model weights if requested
                if self.normalize_weights:
                    normalize_weights(self.model, eps=1e-4)

                eval_steps = 0
                for data in tqdm(self.valid_dataloader, desc=f"Validation progress epoch {self.epoch}", disable=not self.log_to_screen):
                    eval_steps += 1

                    # map to gpu
                    gdata = map(lambda x: x.to(self.device), data)

                    # preprocess
                    inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
                    inp = self.preprocessor.flatten_history(inp)

                    # split list of targets
                    tarlist = torch.split(tar, 1, dim=1)

                    # make sure the input has the correct shape
                    # inpt = inp.unsqueeze(1).repeat(1, self.params.local_ensemble_size, 1, 1, 1)
                    # do we need to clone here?
                    inptlist = [inp.clone() for _ in range(self.params.local_ensemble_size)]
                    for idt, targ in enumerate(tarlist):
                        # flatten history of the target
                        targ = self.preprocessor.flatten_history(targ)

                        # FW pass
                        predlist = []
                        with amp.autocast(device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                            for ic in range(self.params.local_ensemble_size):
                                # retrieve input
                                inpt = inptlist[ic]

                                # forward pass
                                pred = self.model_eval(inpt, n_steps=self.params.stochastic_interpolation_steps)
                                predlist.append(pred)

                                # append input to prediction
                                inptlist[ic] = self.preprocessor.append_history(inpt, pred, idt, update_state=(ic == 0))

                            # concatenate
                            pred = torch.stack(predlist, dim=1)
                            loss = self.loss_obj(pred, targ)

                        # TODO: move all of this into the visualization handler
                        if (eval_steps <= 1) and visualize:
                            # create average prediction for deterministic metrics
                            predm = torch.mean(pred, dim=1)
                            if comm.get_size("ensemble") > 1:
                                predm = reduce_from_parallel_region(predm, "ensemble") / float(comm.get_size("ensemble"))

                            pred_gather = predm[0, ...].detach().clone()
                            targ_gather = targ[0, ...].detach().clone()

                            pred_gather = self.metrics._gather_input(pred_gather)
                            targ_gather = self.metrics._gather_input(targ_gather)

                            if self.viz_stream is not None:
                                self.viz_stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(self.viz_stream):
                                self.viz_prediction_cpu.copy_(pred_gather, non_blocking=True)
                                self.viz_target_cpu.copy_(targ_gather, non_blocking=True)
                            if self.viz_stream is not None:
                                self.viz_stream.synchronize()

                            pred_cpu = self.viz_prediction_cpu.to(torch.float32).numpy()
                            targ_cpu = self.viz_target_cpu.to(torch.float32).numpy()

                            tag = f"step{eval_steps}_time{str(idt).zfill(3)}"
                            self.visualizer.add(tag, pred_cpu, targ_cpu)

                        # update metrics
                        self.metrics.update(pred, targ, loss, idt)

                # create final logs
                logs = self.metrics.finalize()

        # finalize plotting
        viz_time = time.perf_counter_ns()
        if visualize:
            self.visualizer.finalize()
        viz_time = (time.perf_counter_ns() - viz_time) * 10 ** (-9)

        # global sync is in order
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # timer
        valid_time = time.time() - valid_start

        return valid_time, viz_time, logs

    def log_epoch(self, train_logs, valid_logs, timing_logs):
        # separator
        separator = "".join(["-" for _ in range(50)])
        print_prefix = "    "

        def get_pad(nchar):
            return "".join([" " for x in range(nchar)])

        if self.log_to_screen:
            # header:
            self.logger.info(separator)
            self.logger.info(f"Epoch {self.epoch} summary:")
            self.logger.info(f"Performance Parameters:")
            self.logger.info(print_prefix + "training steps: {}".format(train_logs["train_steps"]))
            self.logger.info(print_prefix + "validation steps: {}".format(valid_logs["base"]["validation steps"]))
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024.0 * 1024.0 * 1024.0)
            self.logger.info(print_prefix + f"memory footprint [GB]: {all_mem_gb:.2f}")
            for key in timing_logs.keys():
                self.logger.info(print_prefix + key + ": {:.2f}".format(timing_logs[key]))

            # compute padding:
            print_list = ["training loss", "validation loss"] + list(valid_logs["metrics"].keys())
            max_len = max([len(x) for x in print_list])
            pad_len = [max_len - len(x) for x in print_list]
            # validation summary
            self.logger.info("Metrics:")
            self.logger.info(print_prefix + "training loss: {}{}".format(get_pad(pad_len[0]), train_logs["loss"]))
            self.logger.info(print_prefix + "validation loss: {}{}".format(get_pad(pad_len[1]), valid_logs["base"]["validation loss"]))
            for idk, key in enumerate(print_list[3:], start=3):
                value = valid_logs["metrics"][key]
                if np.isscalar(value):
                    self.logger.info(f"{print_prefix}{key}: {get_pad(pad_len[idk])}{value}")
            self.logger.info(separator)

        if self.log_to_wandb:
            wandb.log(train_logs, step=self.epoch)
            wandb.log(valid_logs["base"], step=self.epoch)

            # log metrics
            wandb.log(valid_logs["metrics"], step=self.epoch, commit=True)

        return
