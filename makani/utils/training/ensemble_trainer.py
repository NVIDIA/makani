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
from torch import amp
import torch.distributed as dist

import logging
import wandb

# timers
from makani.utils.profiling import Timer

# for the manipulation of state dict
from collections import OrderedDict

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
from makani.utils.training.training_helpers import clip_grads

class EnsembleTrainer(Trainer):
    """
    Trainer class holding all the necessary information to perform training.
    """

    def __init__(self, params: Optional[YParams] = None, world_rank: Optional[int] = 0, device: Optional[str] = None):
        Driver.__init__(self, params, world_rank, device)

        # init wandb
        with Timer() as timer:
            if self.log_to_wandb:
                self._init_wandb(self.params, job_type="ensemble")
        self.timers["wandb init"] = timer.time

        # set checkpoint version: start at -1 so that first version which is written is 0
        self.checkpoint_version_current = -1

        # init nccl: do a single AR to make sure that SHARP locks
        # on to the right tree, and that barriers can be used etc
        with Timer() as	timer:
            if dist.is_initialized():
                tens = torch.ones(1, device=self.device)
                dist.all_reduce(tens, group=comm.get_group("data"))
        self.timers["nccl init"] = timer.time

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
        with Timer() as	timer:
            if self.log_to_screen:
                self.logger.info(f"Using channel names: {self.params.channel_names}")
                self.logger.info("initializing data loader")
            self.train_dataloader, self.train_dataset, self.train_sampler = get_dataloader(self.params, self.params.train_data_path, mode="train", device=self.device)
            self.valid_dataloader, self.valid_dataset, self.valid_sampler = get_dataloader(self.params, self.params.valid_data_path, mode="eval", device=self.device)
            self._set_data_shapes(self.params, self.valid_dataset)
        self.timers["dataloader init"] = timer.time

        if self.log_to_screen:
            self.logger.info("data loader initialized")

        # record data required to reproduce workflow using a model package
        with Timer() as timer:
            if self.world_rank == 0:
                from makani.models.model_package import save_model_package
                save_model_package(self.params)
        self.timers["save model package"] = timer.time

        # init preprocessor and model
        with Timer() as	timer:
            self.multistep = self.params.n_future > 0
            self.model = model_registry.get_model(self.params, multistep=self.multistep).to(self.device)
            self.preprocessor = self.model.preprocessor
        self.timers["model init"] = timer.time

        # print aux channel names:
        if self.log_to_screen:
            self.logger.info(f"Auxiliary channel names: {self.params.aux_channel_names}")

        # if model-parallelism is enabled, we need to sure that shared weights are matching across ranks
        # as random seeds might get out of sync during initialization
        # DEBUG: this also needs to be fixed in NCCL
        # if comm.get_size("model") > 1:
        with Timer() as timer:
            sync_params(self.model, mode="broadcast")
        self.timers["sync parameters"] = timer.time

        # add a barrier here
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
        with Timer() as timer:
            clim = get_climatology(self.params)
            clim = torch.from_numpy(clim).to(torch.float32)
            rollout_length = params.get("valid_autoreg_steps", 0) + 1
            self.metrics = MetricsHandler(params=self.params, climatology=clim, num_rollout_steps=rollout_length, device=self.device)
            self.metrics.initialize_buffers()
        self.timers["metric handler init"] = timer.time

        # loss handler
        with Timer() as timer:
            self.loss_obj = LossHandler(self.params)
            self.loss_obj = self.loss_obj.to(self.device)
        self.timers["loss handler init"] = timer.time

        # optimizer and scheduler setup
        # model
        with Timer() as timer:
            self.model_optimizer = self.get_optimizer(self.model, self.params)
            self.model_scheduler = self.get_scheduler(self.model_optimizer, self.params)
        self.timers["optimizer and scheduler init"] = timer.time

        # gradient scaler
        self.gscaler = amp.GradScaler("cuda", enabled=(self.amp_dtype == torch.float16))

        # gradient clipping
        self.max_grad_norm = self.params.get("optimizer_max_grad_norm", -1.0)

        # we need this further down
        with Timer() as timer:
            capture_stream = None
            if dist.is_initialized() and not self.params.disable_ddp:
                if self.device.type == "cuda":
                    capture_stream = torch.Stream(device="cuda")

                with torch.cuda.stream(capture_stream):
                    # model
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
        self.timers["reduction hooks init"] = timer.time

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

        with Timer() as timer:
            self._compile_model(inp_shape)
        self.timers["compile model"] = timer.time

        # visualization wrapper:
        with Timer() as timer:
            plot_list = [{"name": "windspeed_uv10", "functor": "lambda x: np.sqrt(np.square(x[0, ...]) + np.square(x[1, ...]))", "diverging": False}]
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
        self.timers["visualizer init"] = timer.time

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

            with Timer() as timer:
                self.restore_from_checkpoint(
                    checkpoint_path,
                    model=self.model,
                    loss=self.loss_obj if self.params.get("load_loss", True) else None,
                    optimizer=self.model_optimizer if self.params.get("load_optimizer", True) else None,
                    scheduler=self.model_scheduler if self.params.get("load_scheduler", True) else None,
                    counters=counters if self.params.get("load_counters", True) else None,
                    checkpoint_mode=self.params.load_checkpoint,
                    strict=self.params.get("strict_restore", True),
                )
            self.timers["loading checkpoint"] = timer.time

            # override learning rate - useful when restoring optimizer but want to override the LR
            if self.params.get("override_lr", False):
                for param_group in self.model_optimizer.param_groups:
                    param_group["lr"] = self.params.get("lr", 1e-3)

        if self.params.resuming:

            # find latest checkpoint
            checkpoint_path = self.params.checkpoint_path
            self.checkpoint_version_current = get_latest_checkpoint_version(checkpoint_path)
            checkpoint_path = checkpoint_path.format(checkpoint_version=self.checkpoint_version_current, mp_rank="{mp_rank}")

            if self.log_to_screen:
                self.logger.info(f"Resuming from checkpoint {checkpoint_path} in {self.params.load_checkpoint} mode")

            with Timer() as timer:
                self.restore_from_checkpoint(
                    checkpoint_path,
                    model=self.model,
                    loss=self.loss_obj if self.params.get("load_loss", True) else None,
                    optimizer=self.model_optimizer if self.params.get("load_optimizer", True) else None,
                    scheduler=self.model_scheduler if self.params.get("load_scheduler", True) else None,
                    counters=counters if self.params.get("load_counters", True) else None,
                    checkpoint_mode=self.params.load_checkpoint,
                    strict=self.params.get("strict_restore", True),
                )
            self.timers["loading checkpoint"] = timer.time

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

        # allow freezing of model parameters to train only the perturbation
        if self.params.is_set("freeze_model") and self.params.freeze_model:
            if self.log_to_screen:
                self.logger.info(f"Freezing model weights")
            for param in self.model.parameters():
                param.requires_grad = False

        # print setup times
        self._log_timers()

    def _set_train(self):
        self.model.train()
        self.loss_obj.train()
        self.preprocessor.train()

    def _set_eval(self):
        self.model.eval()
        self.loss_obj.eval()
        self.preprocessor.eval()

    def train(self, training_profiler=None, validation_profiler=None):
        # log parameters
        if self.log_to_screen:
            # log memory usage so far
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024.0 * 1024.0 * 1024.0)
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (1024.0 * 1024.0 * 1024.0)
            self.logger.info(f"Scaffolding memory high watermark: {all_mem_gb} GB ({max_mem_gb} GB for pytorch)")
            # announce training start
            self.logger.info("Starting Ensemble Training Loop...")

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
                train_time, train_data_gb, train_logs = self.train_one_epoch(profiler=training_profiler)
            else:
                train_time = 0
                train_data_gb = 0
                train_logs = {"train_steps" : 0, "loss" : 0.0}

            # validate if not to be skipped
            if not self.params.get("skip_validation", False):
                valid_time, viz_time, valid_logs = self.validate_one_epoch(epoch, profiler=validation_profiler)
            else:
                valid_time = 0
                viz_time = 0
                valid_logs = {"base": {}, "metrics": {}}

            if self.params.scheduler == "ReduceLROnPlateau":
                self.model_scheduler.step(valid_logs["base"]["validation loss"])
            elif self.model_scheduler is not None:
                self.model_scheduler.step()

            # log learning rate
            if self.log_to_wandb:
                for param_group in self.model_optimizer.param_groups:
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
                self.save_checkpoint(checkpoint_path, self.model, self.loss_obj, self.model_optimizer, self.model_scheduler, counters, checkpoint_mode=checkpoint_mode)

                # save best checkpoint
                best_checkpoint_path = self.params.best_checkpoint_path.format(mp_rank=comm.get_rank("model"))
                best_model_checkpoint_saved = os.path.isfile(best_checkpoint_path)
                if (not self.params.get("skip_validation", False)) and ((not best_model_checkpoint_saved) or (valid_logs["base"]["validation loss"] <= best_valid_loss)):
                    self.save_checkpoint(
                        self.params.best_checkpoint_path, self.model, self.loss_obj, self.model_optimizer, self.model_scheduler, counters, checkpoint_mode=checkpoint_mode
                    )
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

            # exit here if not training
            if self.params.get("skip_training", False):
                break

        # training done
        training_end = time.time()
        if self.log_to_screen:
            self.logger.info("Total training time is {:.2f} sec".format(training_end - training_start))

        return


    def _ensemble_step(self, inp: torch.Tensor, tar: torch.Tensor):
        predlist = []
        for _ in range(self.params.local_ensemble_size):
            # forward pass
            pred = self.model_train(inp)
            # store prediction
            predlist.append(pred)

        # stack predictions along new dim (ensemble dim):
        pred = torch.stack(predlist, dim=1)
        # compute loss
        loss = self.loss_obj(pred, tar)

        return pred, loss


    def train_one_epoch(self, profiler=None):
        self.epoch += 1
        total_data_bytes = 0
        self._set_train()

        # we need this for the loss average
        accumulated_loss = torch.zeros((2), dtype=torch.float32, device=self.device)

        train_steps = 0
        train_start = time.perf_counter_ns()
        self.model_train.zero_grad(set_to_none=True)
        for data in tqdm(self.train_dataloader, desc=f"Training progress epoch {self.epoch}", disable=not self.log_to_screen):
            train_steps += 1
            self.iters += 1

            torch.cuda.nvtx.range_push(f"train step {train_steps}")

            # map to device
            gdata = map(lambda x: x.to(self.device), data)

            # do preprocessing
            inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)

            # flatten the history
            inp = self.preprocessor.flatten_history(inp)
            tar = self.preprocessor.flatten_history(tar)

            # assuming float32
            total_data_bytes += inp.nbytes + tar.nbytes

            # check if we need to perform an gacc update
            do_update = (train_steps % self.params["gradient_accumulation_steps"] == 0)
            loss_scaling_fact = 1.0
            if self.params["gradient_accumulation_steps"] > 1:
                loss_scaling_fact = 1.0 / np.float32(self.params["gradient_accumulation_steps"])

            # accumulate loss into this tensor
            with amp.autocast(device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):

                if do_update:
                    pred, loss = self._ensemble_step(inp, tar)
                else:
                    with self.model_train.no_sync():
                        pred, loss = self._ensemble_step(inp, tar)
                loss = loss * loss_scaling_fact

            # backward pass
            self.gscaler.scale(loss).backward()

            # increment accumulated loss
            accumulated_loss[0] += loss.detach().clone() * inp.shape[0]
            accumulated_loss[1] += inp.shape[0]

            # perform weight update if requested
            if do_update:
                if self.max_grad_norm > 0.0:
                    self.gscaler.unscale_(self.model_optimizer)
                    clip_grads(self.model_train, self.max_grad_norm)

                self.gscaler.step(self.model_optimizer)
                self.gscaler.update()
                self.model_train.zero_grad(set_to_none=True)

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

            torch.cuda.nvtx.range_pop()

            # profiler step
            if profiler is not None:
                profiler.step()

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

    def validate_one_epoch(self, epoch, profiler=None):
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

        # create list of states for noised inputs:
        with torch.inference_mode():
            with torch.no_grad():

                eval_steps = 0
                for data in tqdm(self.valid_dataloader, desc=f"Validation progress epoch {self.epoch}", disable=not self.log_to_screen):
                    eval_steps += 1

                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_push(f"eval step {eval_steps}")

                    # map to gpu
                    gdata = map(lambda x: x.to(self.device), data)

                    # preprocess
                    inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
                    inp = self.preprocessor.flatten_history(inp)

                    # split list of targets
                    tarlist = torch.split(tar, 1, dim=1)

                    # do autoregression for each ensemble member individually
                    # do the rollout
                    noise_states = [None for _ in range(self.params.local_ensemble_size)]
                    inptlist = [inp.clone() for _ in range(self.params.local_ensemble_size)]
                    for idt, targ in enumerate(tarlist):

                        # flatten history of the target
                        targ = self.preprocessor.flatten_history(targ)

                        # FW pass
                        predlist = []

                        with amp.autocast(device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                            for e in range(self.params.local_ensemble_size):
                                # retrieve input
                                inpt = inptlist[e]

                                # restore noise state
                                if (self.params.local_ensemble_size > 1) and (noise_states[e] is not None):
                                    # recover correct state
                                    self.preprocessor.set_internal_state(noise_states[e])

                                # forward pass
                                pred = self.model_eval(inpt)
                                predlist.append(pred)

                                # store new state
                                if self.params.local_ensemble_size > 1:
                                    noise_states[e] = self.preprocessor.get_internal_state(tensor=True)

                                # append input to prediction
                                last_member = e == self.params.local_ensemble_size - 1
                                inptlist[e] = self.preprocessor.append_history(inpt, pred, idt, update_state=last_member)

                            # concatenate
                            pred = torch.stack(predlist, dim=1)
                            loss = self.loss_obj(pred, targ)

                        # TODO: move all of this into the visualization handler
                        if (eval_steps <= 1) and visualize:
                            # create average prediction for deterministic metrics
                            pred_gather = pred[0, 0, ...].clone()
                            targ_gather = targ[0, ...].clone()

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

                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()

                    # profiler step
                    if profiler is not None:
                        profiler.step()

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
