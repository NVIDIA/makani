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
import time
from typing import Optional, Union, List

import numpy as np
from tqdm import tqdm
import pynvml
import datetime as dt
import h5py as h5

import torch
import torch.distributed as dist
import torch.amp as amp
import torch.utils.data as tud

import logging
import wandb

from makani.utils.driver import Driver
from makani.utils import LossHandler, MetricsHandler
from makani.utils.dataloader import get_dataloader
from makani.utils.dataloaders.data_loader_multifiles import MultifilesDataset
from makani.utils.dataloaders.data_helpers import get_timestamp, get_date_from_timestamp, get_default_aws_connector, get_data_normalization, get_climatology
from makani.utils.YParams import YParams

from makani.models import model_registry

# distributed computing stuff
from makani.utils import comm
from makani.utils import visualize

# inference specific stuff
from makani.utils.inference.helpers import split_list, SortedIndexSampler, date_helper, translate_date_sampler_to_timedelta_sampler
from makani.utils.inference.rollout_buffer import RolloutBuffer, TemporalAverageBuffer, SpectrumAverageBuffer, ZonalSpectrumAverageBuffer

# checkpoint helpers
from makani.utils.checkpoint_helpers import get_latest_checkpoint_version

class Inferencer(Driver):
    """
    Inferencer class holding all the necessary information to perform inference. Design is similar to Trainer, however only keeping the necessary information.

    Parameters
    ============
    params : YParams
        Parameter object for initialization
    world_rank : int
        Rank in the world communicator group
    device : str
        Device on which to perform the inference
    """

    def __init__(self, params: Optional[YParams] = None, world_rank: Optional[int] = 0, device: Optional[str] = None):
        # init the trainer
        super().__init__(params, world_rank, device)

        # init wandb
        if self.log_to_wandb:
            self._init_wandb(self.params, job_type="inference")

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

        # resuming needs is set to False so loading checkpoints does not attempt to set the optimizer state
        self.params["resuming"] = False

        # data loader
        if self.params.log_to_screen:
            self.logger.info("initializing data loader")

        # manually overwrites the dataloader to multifiles. No other dartaloader is supported for inference at the current time
        self.params["multifiles"] = True
        if not hasattr(self.params, "enable_synthetic_data"):
            self.params["enable_synthetic_data"] = False
        if not hasattr(self.params, "amp"):
            self.params["enable_synthetic_data"] = False

        # the file path is taken from inf_data_path to perform inference on the out of sample dataset
        self.valid_dataloader, self.valid_dataset, _ = get_dataloader(self.params, self.params.inf_data_path, mode="inference", device=self.device)
        self._set_data_shapes(self.params, self.valid_dataset)

        if self.log_to_screen:
            self.logger.info("data loader initialized")

        self.mask_dataset = None
        if self.params.get("mask_file", None) is not None:
            self.mask_dataset = MultifilesDataset(
                location=self.params.get("mask_file"),
                dt=1,  # not relevant since we are looking up samples directly
                in_channels=self.params.get("in_channels"),
                out_channels=self.params.get("out_channels"),
                n_history=0,  # since we only use it for the output
                n_future=0,
                add_zenith=0,
                data_grid_type=params.get("data_grid_type", "equiangular"),
                model_grid_type=params.get("model_grid_type", "equiangular"),
                crop_size=(params.get("crop_size_x", None), params.get("crop_size_y", None)),
                crop_anchor=(params.get("crop_anchor_x", 0), params.get("crop_anchor_y", 0)),
                return_timestamp=False,
                relative_timestamp=True,
                return_target=False,
                file_suffix=params.get("dataset_file_suffix", "h5"),
                dataset_path="fields",
                enable_s3=params.get("enable_s3", False),
                io_grid=params.get("io_grid", [1, 1, 1]),
                io_rank=params.get("io_rank", [0, 0, 0]),
                enable_logging=False,
            )

            # we need grid quadrature as well in this case
            from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature

            quadrature_rule = grid_to_quadrature_rule(params.get("model_grid_type", "equiangular"))
            self.quadrature = GridQuadrature(
                quadrature_rule,
                img_shape=self.mask_dataset.img_shape,
                crop_shape=self.mask_dataset.crop_size,
                crop_offset=self.mask_dataset.crop_anchor,
                normalize=True,
                distributed=(comm.get_size("spatial") > 1),
            ).to(self.device)

        self.climatology_dataset = None
        if self.params.get("climatology_file", None) is not None:
            bias, scale = get_data_normalization(self.params)
            self.climatology_dataset = MultifilesDataset(
                location=self.params.get("climatology_file"),
                dt=1,  # not relevant since we are looking up samples directly
                in_channels=self.params.get("in_channels"),
                out_channels=self.params.get("out_channels"),
                n_history=0,  # since we only use it for the output
                n_future=0,
                add_zenith=0,
                data_grid_type=params.get("data_grid_type", "equiangular"),
                model_grid_type=params.get("model_grid_type", "equiangular"),
                bias=bias, # we subtract the bias to avoid subtracting too big numbers
                scale=scale, # we need to set that to make sure the climatology is properly scaled
                crop_size=(params.get("crop_size_x", None), params.get("crop_size_y", None)),
                crop_anchor=(params.get("crop_anchor_x", 0), params.get("crop_anchor_y", 0)),
                return_timestamp=False,
                relative_timestamp=True,
                return_target=False,
                file_suffix=params.get("dataset_file_suffix", "h5"),
                dataset_path="fields",
                enable_s3=params.get("enable_s3", False),
                io_grid=params.get("io_grid", [1, 1, 1]),
                io_rank=params.get("io_rank", [0, 0, 0]),
                enable_logging=False,
            )

        # get model
        self.multistep = False
        self.model = model_registry.get_model(self.params, multistep=self.multistep).to(self.device)
        self.preprocessor = self.model.preprocessor

        # print model
        if self.world_rank == 0:
            print(self.model)

        if self.log_to_screen:
            self.logger.info(f"Loading pretrained checkpoint {self.params.checkpoint_path} in {self.params.load_checkpoint} mode")

        # restore from checkpoint
        checkpoint_path = self.params.checkpoint_path
        self.checkpoint_version_current = get_latest_checkpoint_version(checkpoint_path)
        checkpoint_path = checkpoint_path.format(checkpoint_version=self.checkpoint_version_current, mp_rank="{mp_rank}")

        self.restore_from_checkpoint(
            checkpoint_path,
            model=self.model,
            checkpoint_mode=self.params.load_checkpoint,
            strict=self.params.get("strict_restore", True),
        )

        # loss handler
        self.loss_obj = LossHandler(self.params)
        self.loss_obj = self.loss_obj.to(self.device)

    def _set_eval(self):
        self.model.eval()
        self.loss_obj.eval()
        self.preprocessor.eval()

    # shorthand for inference range running over the full dataset
    def inference_epoch(
            self, rollout_steps: int, compute_metrics: bool = False, output_channels: List[str] = [], output_file: Optional[str] = None, output_memory_buffer_size: Optional[int] = None, bias_file: Optional[str] = None, spectrum_file: Optional[str] = None, zonal_spectrum_file: Optional[str] = None, wb2_compatible: Optional[bool] = False, profiler=None
    ):
        """
        Runs the model in autoregressive inference mode on the entire validation dataset. Computes metrics and scores the model.
        """

        # get the number of maximum samples
        num_samples = len(self.valid_dataset)

        # split the samples across ranks
        samples_local = split_list(list(range(0, num_samples)), comm.get_size("batch"))[comm.get_rank("batch")]

        # distribute samples across all ranks
        start = min(samples_local)
        end = max(samples_local) + 1

        logs = self.inference_range(
            start,
            end,
            1,
            rollout_steps=rollout_steps,
            batch_size=self.params.batch_size,
            compute_metrics=compute_metrics,
            output_channels=output_channels,
            output_file=output_file,
            output_memory_buffer_size=output_memory_buffer_size,
            bias_file=bias_file,
            spectrum_file=spectrum_file,
            zonal_spectrum_file=zonal_spectrum_file,
            wb2_compatible=wb2_compatible,
            profiler=profiler,
        )

        return logs

    # routine which allows to run over a given range
    def inference_range(
        self,
        start: int,
        end: int,
        step: int,
        rollout_steps: int,
        batch_size: int,
        compute_metrics: bool = False,
        metrics_file: Optional[str] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        output_memory_buffer_size: Optional[int] = None,
        bias_file: Optional[str] = None,
        spectrum_file: Optional[str] = None,
        zonal_spectrum_file: Optional[str] = None,
        wb2_compatible: Optional[bool] = False,
        profiler=None,
    ):

        # create index list for range
        indices = list(range(start, end, step))

        # run inference on indexlist
        logs = self.inference_indexlist(
            indices,
            rollout_steps=rollout_steps,
            batch_size=batch_size,
            compute_metrics=compute_metrics,
            metrics_file=metrics_file,
            output_channels=output_channels,
            output_file=output_file,
            output_memory_buffer_size=output_memory_buffer_size,
            bias_file=bias_file,
            spectrum_file=spectrum_file,
            zonal_spectrum_file=zonal_spectrum_file,
            wb2_compatible=wb2_compatible,
            profiler=profiler,
        )

        return logs

    def inference_indexlist(
        self,
        indices: Union[List[int], torch.Tensor],
        rollout_steps: int,
        batch_size: int,
        compute_metrics: bool = False,
        metrics_file: Optional[str] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        output_memory_buffer_size: Optional[int] = None,
        bias_file: Optional[str] = None,
        spectrum_file: Optional[str] = None,
        zonal_spectrum_file: Optional[str] = None,
        wb2_compatible: Optional[bool] = False,
        profiler=None,
    ):

        # initialize metrics handler if requested
        if compute_metrics:
            # metrics handler
            if self.climatology_dataset is None:
                clim = get_climatology(self.params)
                clim = torch.from_numpy(clim).to(torch.float32)
            else:
                clim = None
            metric_channels = self.params.channel_names
            metrics = MetricsHandler(
                params=self.params,
                climatology=clim,
                num_rollout_steps=rollout_steps,
                device=self.device,
                l1_var_names=metric_channels,
                rmse_var_names=metric_channels,
                acc_var_names=metric_channels,
                crps_var_names=metric_channels,
                spread_var_names=metric_channels,
                ssr_var_names=metric_channels,
                rh_var_names=metric_channels,
                wb2_compatible=wb2_compatible,
            )
            metrics.initialize_buffers()
        else:
            metrics = None

        # get scaling factors for the output
        bias, scale = get_data_normalization(self.params)
        bias = torch.from_numpy(bias[:, self.params.out_channels, ...]).to(dtype=torch.float32)
        scale = torch.from_numpy(scale[:, self.params.out_channels, ...]).to(dtype=torch.float32)

        # get shapes
        img_shape = [self.params.img_shape_x, self.params.img_shape_y]
        local_shape = [self.params.get("img_local_shape_x", img_shape[0]), self.params.get("img_local_shape_y", img_shape[1])]
        local_offset = [self.params.get("img_local_offset_x", 0), self.params.get("img_local_offset_y", 0)]


        if output_file is not None:
            # intiialize the rollout buffer
            rollout_buffer = RolloutBuffer(
                num_samples=len(indices),
                batch_size=batch_size,
                num_rollout_steps=rollout_steps,
                rollout_dt=self.params.dt,
                ensemble_size=self.params.local_ensemble_size,
                img_shape=img_shape,
                local_shape=local_shape,
                local_offset=local_offset,
                channel_names=self.params.channel_names,
                lat_lon=(self.params.lat, self.params.lon),
                device=self.device,
                scale=scale,
                bias=bias,
                output_file=output_file,
                output_channels=output_channels,
                output_memory_buffer_size=output_memory_buffer_size,
            )
        else:
            rollout_buffer = None

        if bias_file is not None:
            # we use all channels for the bias computation
            bias_channels = output_channels if output_channels else self.params.channel_names
            # intiialize buffer for bias computation
            # set bias to none since we only pass the differences
            bias_buffer = TemporalAverageBuffer(
                num_rollout_steps=rollout_steps,
                rollout_dt=self.params.dt,
                img_shape=img_shape,
                local_shape=local_shape,
                local_offset=local_offset,
                channel_names=self.params.channel_names,
                lat_lon=(self.params.lat, self.params.lon),
                device=self.device,
                scale=scale,
                bias=None,
                output_file=bias_file,
                output_channels=bias_channels,
            )
        else:
            bias_buffer = None

        if spectrum_file is not None:
            # we use all channels for the spectral computation
            spectrum_channels = output_channels if output_channels else self.params.channel_names
            spectrum_buffer = SpectrumAverageBuffer(
                num_rollout_steps=rollout_steps,
                rollout_dt=self.params.dt,
                img_shape=img_shape,
                ensemble_size=self.params.local_ensemble_size,
                grid_type=self.params.model_grid_type,
                channel_names=self.params.channel_names,
                device=self.device,
                scale=scale,
                bias=bias,
                output_channels=spectrum_channels,
                output_file=spectrum_file,
                spatial_distributed=True,
            )
        else:
            spectrum_buffer = None

        if zonal_spectrum_file is not None:
            # we use all channels for the zonal spectral computation
            zonal_spectrum_channels = output_channels if output_channels else self.params.channel_names
            zonal_spectrum_buffer = ZonalSpectrumAverageBuffer(
                num_rollout_steps=rollout_steps,
	            rollout_dt=self.params.dt,
                img_shape=img_shape,
                ensemble_size=self.params.local_ensemble_size,
                channel_names=self.params.channel_names,
                lat_lon=(self.params.lat, self.params.lon),
                device=self.device,
                scale=scale,
                bias=bias,
                output_channels=zonal_spectrum_channels,
                output_file=zonal_spectrum_file,
                spatial_distributed=True,
            )
        else:
            zonal_spectrum_buffer = None


        # if wb2 compatibility enabled check if a mask was specified
        if wb2_compatible and self.log_to_screen:
            if self.mask_dataset is None:
                self.logger.warning("WeatherBench compatibility enabled but no mask specified. Results may differ, please specify a mask.")
            if self.climatology_dataset is None:
                self.logger.warning("WeatherBench compatibility enabled but no climatology specified. Results may differ, please specify a wb2-compatible climatology.")

        logs = self._inference_indexlist(indices, rollout_steps, batch_size, metrics=metrics, rollout_buffer=rollout_buffer, bias_buffer=bias_buffer, spectrum_buffer=spectrum_buffer, zonal_spectrum_buffer=zonal_spectrum_buffer, profiler=profiler)

        if compute_metrics and not (metrics_file is None):
            if comm.get_rank("world") == 0:
                metrics.save(metrics_file)

        return logs

    def _inference_indexlist(
        self,
        indices: Union[List[int], torch.Tensor],
        rollout_steps: int,
        batch_size: int,
        profiler: Optional = None,
        metrics: Optional = None,
        rollout_buffer: Optional = None,
        bias_buffer: Optional = None,
        spectrum_buffer: Optional = None,
        zonal_spectrum_buffer: Optional = None,
    ):
        """
        main routine that implements autoregressive inference over a number of indices
        """

        # set to eval
        self._set_eval()

        # clear cache
        torch.cuda.empty_cache()

        # initialize metrics buffers
        if metrics is not None:
            metrics.zero_buffers()

        # initialize rollout buffers
        if rollout_buffer is not None:
            rollout_buffer.zero_buffers()

        if bias_buffer is not None:
            bias_buffer.zero_buffers()

        if spectrum_buffer is not None:
            spectrum_buffer.zero_buffers()

        if zonal_spectrum_buffer is not None:
            zonal_spectrum_buffer.zero_buffers()

        # this is the biggest index we can produce
        num_samples = self.valid_dataset.n_samples_total

        date_fn = np.vectorize(get_date_from_timestamp)

        # now we need to reorganize the data tuples so that the loader first rolls out the first batch indices, then the next,
        # etc: generate batched list first:
        # use sorted index sampler, which does the trick
        sampler = SortedIndexSampler(indices, num_samples, batch_size, rollout_steps, self.params.dt)
        subset_dataloader = tud.DataLoader(self.valid_dataset, batch_sampler=sampler, prefetch_factor=6, pin_memory=True, num_workers=self.params.num_data_workers)

        # get timedelta with respect to beginning of year
        if self.mask_dataset is not None:
            mask_sampler = translate_date_sampler_to_timedelta_sampler(sampler, self.valid_dataset, self.mask_dataset)
            self.mask_dataloader = tud.DataLoader(self.mask_dataset, batch_sampler=mask_sampler, prefetch_factor=6, pin_memory=True, num_workers=self.params.num_data_workers)
            mask_iterator = iter(self.mask_dataloader)

        if self.climatology_dataset is not None:
            climatology_sampler = translate_date_sampler_to_timedelta_sampler(sampler, self.valid_dataset, self.climatology_dataset)
            self.climatology_dataloader = tud.DataLoader(self.climatology_dataset, batch_sampler=climatology_sampler, prefetch_factor=6, pin_memory=True, num_workers=self.params.num_data_workers)
            climatology_iterator = iter(self.climatology_dataloader)

        # create loader for the full epoch
        noise_states = None
        inptlist = None
        idt = 0
        with torch.inference_mode():
            with torch.no_grad():
                for token in tqdm(subset_dataloader, desc="Inference progress", disable=not self.log_to_screen):

                    # effective idt and initial condition
                    idte = idt % (rollout_steps + 1)
                    ibatch = idt // (rollout_steps + 1)

                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_push(f"inference step {idt}, rollout step {idte}")

                    # move input to GPU
                    gtoken = map(lambda x: x.to(self.device), token)

                    # get mask
                    if self.mask_dataset is not None:
                        # we need target mask
                        (masks,) = next(mask_iterator)
                        # send to device and remove time dim
                        masks = masks.to(self.device).squeeze(1)

                        # we should normalize the masks just in case:
                        masks_norm = self.quadrature(masks).unsqueeze(-1).unsqueeze(-1)
                        masks = masks / masks_norm
                    else:
                        masks = None

                    # get climatology
                    if self.climatology_dataset is not None:
                        # we need the target clims
                        (clims,) = next(climatology_iterator)
                        # send to device and remove time dim
                        clims = clims.to(self.device).squeeze(1)
                    else:
                        clims = None

                    if idte == 0:
                        # use the input
                        if self.params.add_zenith:
                            inp, inpz, tinp = gtoken
                        else:
                            inp, tinp = gtoken
                            inpz = None

                        dates = date_fn(tinp.flatten().detach().cpu()).tolist()
                        datestring = ", ".join([d.strftime("%Y-%m-%dT%H:%M:%S") for d in dates])
                        print(f"inferencing dates: {datestring}")

                        inp = self.preprocessor.flatten_history(inp)

                        # set the batch size
                        self.preprocessor.update_internal_state(replace_state=True, batch_size=inp.shape[0])

                        # reset noise states and input list
                        noise_states = [None for _ in range(self.params.local_ensemble_size)]
                        inptlist = [inp.clone() for _ in range(self.params.local_ensemble_size)]

                        if rollout_buffer is not None:
                            inpt = torch.stack(inptlist, dim=1)
                            rollout_buffer.update(inpt, tinp[:, 0], idt=idte)

                    else:
                        # use this as target
                        if self.params.add_zenith:
                            tar, tarz, ttar = gtoken
                        else:
                            tar, ttar = gtoken
                            tarz = None
                        targ = self.preprocessor.flatten_history(tar)

                        # set unpredicted
                        self.preprocessor.cache_unpredicted_features(None, None, inpz, tarz)

                        # do predictions
                        predlist = []
                        with amp.autocast(device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                            for e in range(self.params.local_ensemble_size):

                                if torch.cuda.is_available() and (self.params.local_ensemble_size > 1):
                                    torch.cuda.nvtx.range_push(f"ensemble step {e}")

                                # retrieve input
                                inpt = inptlist[e]

                                # restore noise state
                                if (self.params.local_ensemble_size > 1) and (noise_states[e] is not None):
                                    self.preprocessor.set_internal_state(noise_states[e])

                                # forward pass
                                pred = self.model(inpt)
                                predlist.append(pred)

                                # store new state
                                if self.params.local_ensemble_size > 1:
                                    noise_states[e] = self.preprocessor.get_internal_state(tensor=True)

                                # append input to prediction and get the new unpredicted features. idt is 0 here as there is always only one target
                                last_member = e == self.params.local_ensemble_size - 1
                                inptlist[e] = self.preprocessor.append_history(inpt, pred, 0, update_state=last_member)

                                if torch.cuda.is_available() and (self.params.local_ensemble_size > 1):
                                    torch.cuda.nvtx.range_pop()

                            # concatenate
                            pred = torch.stack(predlist, dim=1)
                            # set weight to None here, since I am not sure what to do for spectral components
                            # in spectral CRPS
                            loss = self.loss_obj(pred, targ, None)

                            if rollout_buffer is not None:
                                rollout_buffer.update(pred, ttar[:, 0], idt=idte)

                        # reset zenith angles
                        inpz = tarz
                        tinp = ttar

                        # update the metrics starting from the first actual prediction
                        if (metrics is not None):

                            # subtract clim
                            if clims is not None:
                                predc = pred - clims.unsqueeze(1)
                                targc = targ - clims
                            else:
                                predc = pred
                                targc = targ

                            # update metrics
                            metrics.update(predc, targc, loss, idte - 1, masks)

                        # update the bias computation
                        if (bias_buffer is not None):
                            diff = pred - targ.unsqueeze(dim=1)
                            B, E, C, H, W = diff.shape
                            diff = diff.reshape(B*E, C, H, W).contiguous()
                            bias_buffer.update(diff, idte - 1)

                        if (spectrum_buffer is not None):
                            if pred.dim() == 4:
                                prede = pred.unsqueeze(1)
                            else:
                                prede = pred

                            spectrum_buffer.update(prede, targ.unsqueeze(1), idte - 1)

                        if (zonal_spectrum_buffer is not None):
                            if pred.dim() == 4:
                                prede = pred.unsqueeze(1)
                            else:
                                prede = pred

                            zonal_spectrum_buffer.update(prede, targ.unsqueeze(1), idte - 1)

                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()

                    # increment counter
                    idt += 1

                    if profiler is not None:
                        profiler.step()

        # barrier to ensure everyone is here before calling finalize
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # create final logs
        if metrics is not None:
            logs = metrics.finalize()
        else:
            logs = dict()

        # wait for the copying and writing to disk to finish
        if rollout_buffer is not None:
            rollout_buffer.finalize()

        # record bias
        if bias_buffer is not None:
            bias_buffer.finalize()

        # record spectrum
        if spectrum_buffer is not None:
            spectrum_buffer.finalize()

        # record zonal spectrum
        if zonal_spectrum_buffer is not None:
            zonal_spectrum_buffer.finalize()

        # global sync is in order
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        return logs

    def log_score(self, scoring_logs, scoring_time):
        # separator
        separator = "".join(["-" for _ in range(50)])
        print_prefix = "    "

        def get_pad(nchar):
            return "".join([" " for x in range(nchar)])

        if self.log_to_screen:
            # header:
            self.logger.info(separator)
            self.logger.info(f"Scoring summary:")
            self.logger.info("Total scoring time is {:.2f} sec".format(scoring_time))

            # compute padding:
            print_list = list(scoring_logs["metrics"].keys())
            max_len = max([len(x) for x in print_list])
            pad_len = [max_len - len(x) for x in print_list]
            # validation summary
            self.logger.info("Metrics:")
            for idk, key in enumerate(print_list):
                value = scoring_logs["metrics"][key]
                self.logger.info(f"{print_prefix}{key}: {get_pad(pad_len[idk])}{value}")
            self.logger.info(separator)

        return

    def score_model(
            self, metrics_file: Optional[str] = None, output_channels: List[str] = [], output_file: Optional[str] = None, output_memory_buffer_size: Optional[int]=None, bias_file: Optional[str]=None, spectrum_file: Optional[str]=None, zonal_spectrum_file: Optional[str]=None, start_date=None, end_date=None, date_step=1, wb2_compatible=False, profiler=None
    ):
        """
        main routine for scoring models. Runs the inference over the entire dataset and computes the score. Then writes them to disk
        """

        # log parameters
        if self.log_to_screen:
            # log memory usage so far
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024.0 * 1024.0 * 1024.0)
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (1024.0 * 1024.0 * 1024.0)
            self.logger.info(f"Scaffolding memory high watermark: {all_mem_gb} GB ({max_mem_gb} GB for pytorch)")
            # announce training start
            self.logger.info("Starting Scoring...")

        # perform a barrier here to make sure everybody is ready
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except ValueError:
            pass

        # check if a date range is specified:
        if start_date is not None:
            start_date = date_helper(start_date)

        if end_date is not None:
            end_date = date_helper(end_date)

        # now check if the dates are within dataset range:
        if start_date is not None:
            start_index = self.valid_dataset.get_index_at_time(start_date)
            if start_index is None:
                raise IndexError(f"Error, start date {start_date} is outside the dataset range of {self.valid_dataset.start_date} to {self.valid_dataset.end_date}")
        else:
            start_index = 0

        if end_date is not None:
            end_index = self.valid_dataset.get_index_at_time(end_date)
            if end_index is None:
                raise IndexError(f"Error, end date {end_date} is outside the dataset range of {self.valid_dataset.start_date} to {self.valid_dataset.end_date}")
        else:
            end_index = len(self.valid_dataset) - 1

        # perform sanity checks
        if end_index <= start_index:
            raise ValueError(f"Error, start date {start_date} has to be strictly smaller than end date {end_date}")

        # get the step size
        if date_step < self.valid_dataset.dhours:
            raise ValueError(f"date_step {date_step} is smaller than the dataset dhours {self.valid_dataset.dhours}")
        step = date_step // self.valid_dataset.dhours

        start_date = self.valid_dataset.get_time_at_index(start_index)
        end_date = self.valid_dataset.get_time_at_index(end_index)
        if self.log_to_screen:
            self.logger.info(f"Using date range: {start_date} to {end_date} with a step of {date_step} hours.")
            if output_channels:
                self.logger.info(f"Logging following channels: {output_channels}")

        # split the samples across ranks
        samples_local = split_list(list(range(start_index, end_index, step)), comm.get_size("batch"))[comm.get_rank("batch")]

        # distribute samples across all ranks
        start = min(samples_local)
        end = max(samples_local) + 1

        # start timer
        scoring_start = time.time()

        rollout_steps = self.params.get("valid_autoreg_steps", 0) + 1
        scoring_logs = self.inference_range(
            start,
            end,
            step,
            rollout_steps=rollout_steps,
            batch_size=self.params.batch_size,
            compute_metrics=True,
            metrics_file=metrics_file,
            output_channels=output_channels,
            output_file=output_file,
            output_memory_buffer_size=output_memory_buffer_size,
            bias_file=bias_file,
            spectrum_file=spectrum_file,
            zonal_spectrum_file=zonal_spectrum_file,
            wb2_compatible=wb2_compatible,
            profiler=profiler,
        )

        # end timer
        scoring_end = time.time()

        self.log_score(scoring_logs, scoring_end - scoring_start)

        return
