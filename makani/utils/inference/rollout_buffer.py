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

from typing import Optional, List, Tuple, Union
from abc import ABCMeta, abstractmethod

import torch
import numpy as np
import h5py as h5

# distributed computing stuff
from torch import amp
import torch.distributed as dist
from makani.utils import comm
from makani.models.common import RealFFT1
from makani.mpu.fft import DistributedRealFFT1
from torch_harmonics.distributed import compute_split_shapes, split_tensor_along_dim
from makani.mpu.mappings import reduce_from_parallel_region

# get torch_harmonics for spectra
import torch_harmonics as th
import torch_harmonics.distributed as thd


def _barrier(device):
    """
    Synchronize across ranks if a process group is initialized.

    Wraps ``dist.barrier`` with two safety guards: it's a no-op when no
    process group has been initialized, and it only passes ``device_ids``
    on CUDA (``device.index`` is ``None`` for CPU and would make
    ``dist.barrier(device_ids=[None])`` raise).
    """
    if not dist.is_initialized():
        return
    device_ids = [device.index] if device.type == "cuda" else None
    dist.barrier(device_ids=device_ids)



class DataBuffer(object, metaclass=ABCMeta):
    r"""
    DataBuffer class used as base class for online data analysis
    """

    def __init__(
        self,
	    num_rollout_steps: int,
	    rollout_dt: int,
	    channel_names: List[str],
	    device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
	    output_channels: List[str] = [],
        output_file: Optional[str] = None,
    ):

	    # store members
        self.num_rollout_steps = num_rollout_steps
        self.rollout_dt = rollout_dt
        self.output_channels = output_channels
        self.output_file = output_file

        # get mapping from channels to all channels
        self.channel_mask = [channel_names.index(c) for c in self.output_channels]
        self.num_channels = len(self.output_channels)

        if not self.num_channels > 0:
            raise ValueError(f"Empty channel lists aren't suported. Got {self.output_channels}.")

        # set device and create a stream for writing out data
        self.device = device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # deal with output scale and bias
        if scale is not None:
            scale = scale.squeeze()
            self.scale = scale[self.channel_mask].to(self.device)
        else:
            self.scale = torch.ones(self.num_channels, device=self.device)
        if bias is not None:
            bias = bias.squeeze()
            self.bias = bias[self.channel_mask].to(self.device)
        else:
            self.bias = torch.zeros(self.num_channels, device=self.device)

        # instantiate communicator:
        if self.output_file is not None:
            # set up communicator if requested
            if comm.get_world_size() > 1:
                # initialize MPI. This call is collective!
                from mpi4py import MPI
                self.mpi_comm = MPI.COMM_WORLD.Split(color=0, key=comm.get_world_rank())
            else:
                self.mpi_comm = None

        return

    @abstractmethod
    def zero_buffers(self):
        pass

    @abstractmethod
    def update(self, pred, tstamps, idt):
        pass

    @abstractmethod
    def finalize(self):
        pass


class RolloutBuffer(DataBuffer):
    r"""
    RolloutBuffer class handles the recording of selected channels during the rollout and manages associated buffers.

    Parameters
    ============
    num_samples : int
        Total number of initial conditions to be recorded
    batch_size : int
        Maximum local batch size
    num_rollout_steps : int
        Number of total rollout steps
    ensemble_size : int
        Ensemble size
    img_shape: Tuple[int]
        shape of the output
    channel_names : List[str]
        list of all the channel names in the output
    device : Union[str, torch.device]
        Device on which inference is performed
    scale : torch.Tensor
        Scale for output normalization
    bias : torch.Tensor
        Bias for output normalization
    output_channels : List[str]
        List of channels to be recorded
    output_file: str, optional
        Outputfile to write to
    output_memory_buffer_size: int, optional
        Number of samples to cache in memory before writing to disk
    streaming_mode: bool, optional
        When ``True``, every ``update()`` call writes directly to disk and no
        in-memory rollout buffer is allocated. Requires ``output_file`` to be
        set. Default is ``False``.
    buffer_device: str or torch.device, optional
        Device on which the in-memory rollout buffer is allocated. Defaults
        to ``torch.device("cpu")``, which uses pinned memory when ``device``
        is CUDA. Pass a CUDA device to keep the buffer GPU-resident and
        avoid GPU→CPU transfers on every ``update()``; the buffer is copied
        to host once at flush time. Ignored when ``streaming_mode`` is
        ``True``.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        num_rollout_steps: int,
        rollout_dt: int,
        ensemble_size: int,
        img_shape: Tuple[int, int],
        local_shape: Tuple[int, int],
        local_offset: Tuple[int, int],
        channel_names: List[str],
        lat_lon: Tuple[List[float], List[float]],
        device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        output_memory_buffer_size: Optional[int] = None,
        streaming_mode: bool = False,
        buffer_device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__(num_rollout_steps, rollout_dt, channel_names, device, scale, bias, output_channels, output_file)

        # streaming mode has no in-memory fallback — every update writes to disk.
        if streaming_mode and output_file is None:
            raise ValueError(
                "streaming_mode=True requires output_file to be set; "
                "there is no in-memory buffer in streaming mode."
            )
        self.streaming_mode = streaming_mode

        # resolve buffer device (mirrors DataBuffer's str→torch.device coercion).
        # CUDA buffers avoid GPU→CPU transfers per update() call but pay one
        # bulk copy at flush time.
        self.buffer_device = torch.device(buffer_device) if isinstance(buffer_device, str) else buffer_device

        # store additional members
        self.img_shape = img_shape
        self.local_shape = local_shape
        self.local_offset = local_offset
        self.lat_lon = lat_lon
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.output_channels = output_channels
        self.num_buffered_samples = output_memory_buffer_size if output_memory_buffer_size is not None else num_samples
        self.num_buffered_samples = max(min(self.num_buffered_samples, num_samples), batch_size)

        # little hacky but we use this to compute the range where to write the output to
        if comm.is_distributed("batch") and comm.get_size("batch") > 1:
            num_samples = torch.tensor([self.num_samples], dtype=torch.long, device=self.device)
            num_samples_list = [num_samples.clone() for _ in range(comm.get_size("batch"))]
            dist.all_gather(num_samples_list, num_samples, group=comm.get_group("batch"))
            num_samples = torch.cumsum(torch.cat(num_samples_list, dim=0), dim=0).cpu().numpy()
            self.num_samples_offsets = [
                0,
            ] + num_samples.tolist()
        else:
            self.num_samples_offsets = [0, self.num_samples]
        self.num_samples_total = self.num_samples_offsets[-1]

        # rollout buffer dimensions: initial_conditions x num_rollout_steps x ensemble_size x num_channels x nlat x nlon
        if self.streaming_mode:
            # No in-memory buffer; each update() flushes to disk directly.
            self.rollout_data = None
            self.timestamp_data = None
        else:
            # pin_memory only applies to CPU buffers, and is only useful when the
            # compute device is CUDA (it accelerates async H2D/D2H transfers).
            pin_memory = (self.buffer_device.type == "cpu") and (self.device.type == "cuda")
            local_buffer_size = (self.num_buffered_samples, self.num_rollout_steps + 1, self.ensemble_size, self.num_channels, *self.local_shape)
            self.rollout_data = torch.zeros(local_buffer_size, dtype=torch.float32, device=self.buffer_device, pin_memory=pin_memory)
            self.timestamp_data = torch.zeros((self.num_buffered_samples), dtype=torch.float64, device=self.buffer_device, pin_memory=pin_memory)

        # open output_file
        self.file_handle = None
        if self.output_file is not None:
            self._create_output_file(self.output_file)

            # set up local buffer offsets
            self.file_offset = self.num_samples_offsets[comm.get_rank("batch")]

        # initialize buffer offsets
        self.buffer_offset = 0

        # reshape bias and scale
        self.bias = self.bias.reshape(-1, 1, 1)
        self.scale = self.scale.reshape(-1, 1, 1)

    def _create_output_file(self, output_file):
        if self.mpi_comm is not None:
            # initialize MPI. This call is collective!
            self.file_handle = h5.File(output_file, "w", driver="mpio", comm=self.mpi_comm)
        else:
            self.file_handle = h5.File(output_file, "w")

        # create hdf5 dataset
        total_buffer_size = (self.num_samples_total, self.num_rollout_steps + 1, self.ensemble_size * comm.get_size("ensemble"), self.num_channels, *self.img_shape)
        self.rollout_buffer_disk = self.file_handle.create_dataset("fields", total_buffer_size, dtype=np.float32)

        # create timestamps for scale
        self.timestamp_buffer_disk = self.file_handle.create_dataset("timestamp", (self.num_samples_total), dtype=np.float64)
        self.timestamp_buffer_disk.make_scale("timestamp")
        self.rollout_buffer_disk.dims[0].attach_scale(self.timestamp_buffer_disk)

        # create leadtimes
        leadtime = self.file_handle.create_dataset("lead_time", (self.num_rollout_steps + 1), dtype=np.float64)
        leadtime.make_scale("lead_time")
        self.rollout_buffer_disk.dims[1].attach_scale(leadtime)
        if comm.get_world_rank() == 0:
            dts = np.arange(0, (self.num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64)
            leadtime[:] = dts[:]

        # create channel descriptors, no need to store the handle to that since we can populate right away:
        chanlen = max([len(v) for v in self.output_channels])
        chans = self.file_handle.create_dataset("channel", self.num_channels, dtype=h5.string_dtype(length=chanlen))
        chans.make_scale("channel")
        self.rollout_buffer_disk.dims[3].attach_scale(chans)
        if comm.get_world_rank() == 0:
            chans[...] = self.output_channels

        # create lon and lat descriptors
        lats = self.file_handle.create_dataset("lat", len(self.lat_lon[0]), dtype=np.float32)
        lats.make_scale("lat")
        self.rollout_buffer_disk.dims[4].attach_scale(lats)
        if comm.get_world_rank() == 0:
            lats[...] = np.array(self.lat_lon[0], dtype=np.float32)
        lons = self.file_handle.create_dataset("lon", len(self.lat_lon[1]), dtype=np.float32)
        lons.make_scale("lon")
        self.rollout_buffer_disk.dims[5].attach_scale(lons)
        if comm.get_world_rank() == 0:
            lons[...] = np.array(self.lat_lon[1], dtype=np.float32)

        # label dimensions for more information
        self.rollout_buffer_disk.dims[0].label = "Timestamp in UTC time zone"
        self.rollout_buffer_disk.dims[1].label = "Lead time relative to timestamp"
        self.rollout_buffer_disk.dims[2].label = "Ensemble index"
        self.rollout_buffer_disk.dims[3].label = "Channel name"
        self.rollout_buffer_disk.dims[4].label = "Latitude in degrees"
        self.rollout_buffer_disk.dims[5].label = "Longitude in degrees"

        return

    # close the output file
    def __del__(self):
        # ``__del__`` may run after a partial-init failure (e.g. constructor
        # validation raised before ``self.file_handle`` was assigned). Use
        # getattr so a missing attribute doesn't shadow the original error.
        file_handle = getattr(self, "file_handle", None)
        if file_handle is not None:
            file_handle.close()
            self.file_handle = None
        return

    def zero_buffers(self):
        """
        Reset all buffer state to a fresh post-construction state. Zeros the
        data tensors AND resets ``buffer_offset`` so callers may invoke this
        mid-rollout without leaving a stale offset pointing into newly-zeroed
        slots.

        This mirrors ``MeanStdBuffer.zero_buffers`` (which also resets its
        offset analog ``num_samples_tracked``) for consistency.

        In streaming mode there are no in-memory buffers, so this is a no-op.
        """

        if self.streaming_mode:
            return

        with torch.no_grad():
            self.timestamp_data.fill_(0.0)
            self.rollout_data.fill_(0.0)
        self.buffer_offset = 0

        return

    def _flush_buffer_to_disk(self):
        """Write the accumulated in-memory buffer to disk (non-streaming mode)."""

        # wait for everything to complete (CUDA only — synchronize is a CUDA-specific
        # primitive and would raise on CPU-only builds).
        if self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)

        if (self.file_handle is not None) and (self.buffer_offset > 0):
            # batch ranges in file
            batch_start_file = self.file_offset
            # the buffer offset represents the current filling level of the local buffer
            batch_end_file = batch_start_file + self.buffer_offset
            batch_range = slice(batch_start_file, batch_end_file)

            # batch ranges in memory buffer
            batch_range_buffer = slice(0, self.buffer_offset)

            # ensemble range
            ens_start = self.ensemble_size * comm.get_rank("ensemble")
            ens_range = slice(ens_start, ens_start + self.ensemble_size)

            # spatial range
            lat_range = slice(self.local_offset[0], self.local_offset[0] + self.local_shape[0])
            lon_range = slice(self.local_offset[1], self.local_offset[1] + self.local_shape[1])

            # bring buffer to host if it lives on a non-CPU device
            rollout_src = self.rollout_data if self.buffer_device.type == "cpu" else self.rollout_data.cpu()
            timestamp_src = self.timestamp_data if self.buffer_device.type == "cpu" else self.timestamp_data.cpu()

            # write_direct requires C-contiguous source arrays
            rollout_arr = np.ascontiguousarray(rollout_src.numpy())
            timestamp_arr = np.ascontiguousarray(timestamp_src.numpy())

            # concurrent writing
            if (comm.get_rank("model") == 0) and (comm.get_rank("ensemble") == 0):
                self.timestamp_buffer_disk.write_direct(
                    timestamp_arr,
                    source_sel=batch_range_buffer,
                    dest_sel=batch_range,
                )
            self.rollout_buffer_disk.write_direct(
                rollout_arr,
                source_sel=np.s_[batch_range_buffer, ...],
                dest_sel=np.s_[batch_range, :, ens_range, :, lat_range, lon_range],
            )

            # advance the file pointer BEFORE zero_buffers — zero_buffers now
            # resets buffer_offset, so we must capture it here while still valid.
            self.file_offset += self.buffer_offset

        # reset buffers (also resets buffer_offset)
        self.zero_buffers()

        return

    def _write_to_disk(self, tstamps, predp, idt):
        """Write a single (batch, leadtime) slice straight to disk (streaming mode)."""

        if self.file_handle is None:
            return

        current_batch_size = predp.shape[0]

        # batch range in file (uses current file_offset, advanced after the last leadtime)
        batch_range = slice(self.file_offset, self.file_offset + current_batch_size)

        # ensemble range
        ens_start = self.ensemble_size * comm.get_rank("ensemble")
        ens_range = slice(ens_start, ens_start + self.ensemble_size)

        # spatial range
        lat_range = slice(self.local_offset[0], self.local_offset[0] + self.local_shape[0])
        lon_range = slice(self.local_offset[1], self.local_offset[1] + self.local_shape[1])

        # bring data to host for h5py write; ensure C-contiguity for write_direct
        predp_arr = np.ascontiguousarray(predp.detach().cpu().numpy())

        self.rollout_buffer_disk.write_direct(
            predp_arr,
            dest_sel=np.s_[batch_range, idt, ens_range, :, lat_range, lon_range],
        )

        # write timestamps once per IC (on the first leadtime), single-writer
        if (idt == 0) and (comm.get_rank("model") == 0) and (comm.get_rank("ensemble") == 0):
            tarr = np.ascontiguousarray(tstamps.detach().cpu().numpy())
            self.timestamp_buffer_disk.write_direct(
                tarr,
                dest_sel=batch_range,
            )

        return

    def update(self, pred, tstamps, idt):
        """update local buffers"""

        # get the current batch size
        current_batch_size = pred.shape[0]

        # check if we can buffer the next element or if we need to flush now
        # (only meaningful in non-streaming mode — streaming has no buffer to fill)
        if (not self.streaming_mode) and (idt == 0) and (self.buffer_offset + current_batch_size > self.num_buffered_samples):
            self._flush_buffer_to_disk()

        with torch.no_grad():
            predp = self.scale * pred[..., self.channel_mask, :, :] + self.bias

            if self.streaming_mode:
                self._write_to_disk(tstamps, predp, idt)
                # advance file pointer when this IC's last leadtime is done
                if (idt + 1) == (self.num_rollout_steps + 1):
                    self.file_offset += current_batch_size
            else:
                batch_start = self.buffer_offset
                batch_end = batch_start + current_batch_size

                if idt == 0:
                    self.timestamp_data[batch_start:batch_end].copy_(tstamps, non_blocking=True)
                self.rollout_data[batch_start:batch_end, idt].copy_(predp, non_blocking=True)

                # increase buffer pointer if the next one is a new IC
                if (idt + 1) == (self.num_rollout_steps + 1):
                    self.buffer_offset += current_batch_size

        return

    def finalize(self):

        _barrier(self.device)

        # Flush remaining buffered data when there's both a file to write to and
        # an in-memory buffer to flush. ``_flush_buffer_to_disk`` doubles as a
        # destructive reset; in the no-file path it just zeros the buffer.
        # Calling it here without a file would silently destroy in-memory data
        # the caller may want to read. In streaming mode there's nothing to
        # flush — every ``update()`` already wrote directly to disk.
        if (not self.streaming_mode) and (self.file_handle is not None):
            self._flush_buffer_to_disk()

        _barrier(self.device)

        # Close the output file and null the handle so a subsequent finalize()
        # or __del__ doesn't try to operate on an already-closed h5py object.
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

        return


class MeanStdBuffer(DataBuffer):
    r"""
    MeanStdBuffer class handles the recording of bias during inference. Performs Welford reduction on the fly during inference

    Parameters
    ============
    num_rollout_steps : int
        Number of total rollout steps
    variable_shape: Union[Tuple[int], Tuple[int, int]],
        shape of the variable tensor to be averaged for each rollout step and channel
    channel_names : List[str]
        list of all the channel names in the output
    device : Union[str, torch.device]
        Device on which inference is performed
    scale : torch.Tensor
        Scale for output normalization
    bias : torch.Tensor
        Bias for output normalization
    output_channels : List[str]
        List of channels to be recorded
    output_file: str, optional
        Outputfile to write to
    """

    def __init__(
        self,
        num_rollout_steps: int,
        rollout_dt: int,
        variable_shape: Union[Tuple[int], Tuple[int, int]],
        channel_names: List[str],
        device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
    ):
        super().__init__(num_rollout_steps, rollout_dt, channel_names, device, scale, bias, output_channels, output_file)

        # rollout buffer on CPU has dimensions num_rollout_steps x num_channels x nlat x nlon
        pin_memory = self.device.type == "cuda"
        local_buffer_size = (self.num_rollout_steps, self.num_channels, *variable_shape)
        self.running_mean = torch.zeros(local_buffer_size, dtype=torch.float32, device=self.device)
        self.running_var = torch.zeros(local_buffer_size, dtype=torch.float32, device=self.device)

        # recall how many variable dims we have
        self.variable_dims = len(variable_shape)

        # number of samples seen
        self.num_samples_tracked = torch.zeros((self.num_rollout_steps, 1, *(1 for _ in range(self.variable_dims))), dtype=torch.int64, device=self.device)

    def zero_buffers(self):
        """
        set buffers to zero
        """
        with torch.no_grad():
            self.running_mean.fill_(0.0)
            self.running_var.fill_(0.0)
            self.num_samples_tracked.fill_(0)

        return

    def _compute_stats(self, data, dim=0):
        count = torch.tensor(data.shape[dim], dtype=torch.int64, device=self.device).reshape(1, *(1 for _ in range(self.variable_dims)))
        var, mean = torch.var_mean(data, dim=dim, correction=0, keepdim=False)
        m2 = var * count

        return mean, m2, count

    def _welford_combine(self, mean, m2, count, idt) -> torch.Tensor:
        count_new = self.num_samples_tracked[idt] + count
        delta = mean - self.running_mean[idt]
        self.running_mean[idt, ...] += delta * float(count) / float(count_new)
        self.running_var[idt, ...] += m2 + torch.square(delta) * float(self.num_samples_tracked[idt] * count) / float(count_new)
        self.num_samples_tracked[idt, ...] = count_new

        return

    def _aggregate_stats(self, group_name="data"):
        if comm.get_size(group_name) > 1:
            with torch.no_grad():
                # extract tensors: we need to be SUPER CAREFUL:
                # reduce_from_parallel-region is seemingly inplace when called from within
                # no_grad region, so we need to create copies here
                counts_old = self.num_samples_tracked.clone()
                means_old = self.running_mean.clone()
                m2s_old = self.running_var.clone()

                # counts are: n = sum_k n_k
                counts_agg = reduce_from_parallel_region(self.num_samples_tracked, group_name)
                # means are: mu = sum_i n_i * mu_i / n
                # normalize before reducing to avoid overflows
                meansc = counts_old * means_old / counts_agg.to(dtype=torch.float32)
                means_agg = reduce_from_parallel_region(meansc, group_name)

                # m2s are: sum_i m2_i + sum_i n_i * (mu_i - mu)^2
                m2s_agg = reduce_from_parallel_region(m2s_old, group_name)
                deltas = torch.square(means_old - means_agg) * counts_old
                deltas_agg = reduce_from_parallel_region(deltas, group_name)
                m2s_agg += deltas_agg

                # write back
                self.num_samples_tracked[...] = counts_agg[...]
                self.running_mean[...] = means_agg[...]
                self.running_var[...] = m2s_agg[...]

        return


class TemporalAverageBuffer(MeanStdBuffer):
    r"""
    temporalAverageBuffer class handles the recording of spatial data during inference. Performs Welford reduction on the fly during inference

    Parameters
    ============
    num_rollout_steps : int
        Number of total rollout steps
    img_shape: Tuple[int]
        shape of the output
    channel_names : List[str]
        list of all the channel names in the output
    device : Union[str, torch.device]
        Device on which inference is performed
    scale : torch.Tensor
        Scale for output normalization
    bias : torch.Tensor
        Bias for output normalization
    output_channels : List[str]
        List of channels to be recorded
    output_file: str, optional
        Outputfile to write to
    """
    def __init__(self,
        num_rollout_steps: int,
        rollout_dt: int,
	    img_shape: Tuple[int, int],
        local_shape: Tuple[int, int],
	    local_offset: Tuple[int, int],
        channel_names: List[str],
        lat_lon: Tuple[List[float], List[float]],
        device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
	    output_channels: List[str] = [],
        output_file: Optional[str] = None):

        super().__init__(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=rollout_dt,
            variable_shape=local_shape,
            channel_names=channel_names,
            device=device,
            scale=scale,
            bias=bias,
            output_channels=output_channels,
            output_file=output_file,
        )

        self.img_shape = img_shape
        self.local_shape = local_shape
        self.local_offset = local_offset
        self.lat_lon = lat_lon

        # reshape bias and scale
        self.bias = self.bias.reshape(1, -1, 1, 1)
        self.scale = self.scale.reshape(1, -1, 1, 1)

    def _write_output_data(self, mean, std):

        if self.mpi_comm is not None:
            # initialize MPI. This call is collective!
            file_handle = h5.File(self.output_file, "w", driver="mpio", comm=self.mpi_comm)
        else:
            file_handle = h5.File(self.output_file, "w")

        # create hdf5 dataset and write the data
        data_shape = (self.num_rollout_steps, self.num_channels, self.img_shape[0], self.img_shape[1])
        mean_data = file_handle.create_dataset("mean", data_shape, dtype=np.float64)
        std_data = file_handle.create_dataset("std", data_shape, dtype=np.float64)

        # spatial range
        lat_range = slice(self.local_offset[0], self.local_offset[0] + self.local_shape[0])
        lon_range = slice(self.local_offset[1], self.local_offset[1] + self.local_shape[1])

        if comm.get_rank("data") == 0:
            mean_data[..., lat_range, lon_range] = mean[...]
            std_data[..., lat_range, lon_range] = std[...]

        # create leadtimes
        leadtime = file_handle.create_dataset("lead_time", self.num_rollout_steps, dtype=np.float64)
        leadtime.make_scale("lead_time")
        mean_data.dims[0].attach_scale(leadtime)
        std_data.dims[0].attach_scale(leadtime)
        dts = np.arange(self.rollout_dt, (self.num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64)
        if comm.get_world_rank() == 0:
            leadtime[:] = dts[:]

        # create channel descriptors, no need to store the handle to that since we can populate right away:
        chanlen = max([len(v) for v in self.output_channels])
        chans = file_handle.create_dataset("channel", self.num_channels, dtype=h5.string_dtype(length=chanlen))
        chans.make_scale("channel")
        mean_data.dims[1].attach_scale(chans)
        std_data.dims[1].attach_scale(chans)
        if comm.get_world_rank() == 0:
            chans[...] = self.output_channels

        # create lon and lat descriptors
        lats = file_handle.create_dataset("lat", len(self.lat_lon[0]), dtype=np.float32)
        lats.make_scale("lat")
        mean_data.dims[2].attach_scale(lats)
        std_data.dims[2].attach_scale(lats)
        if comm.get_world_rank() == 0:
            lats[...] = np.array(self.lat_lon[0], dtype=np.float32)
        lons = file_handle.create_dataset("lon", len(self.lat_lon[1]), dtype=np.float32)
        lons.make_scale("lon")
        mean_data.dims[3].attach_scale(lons)
        std_data.dims[3].attach_scale(lons)
        if comm.get_world_rank() == 0:
            lons[...] = np.array(self.lat_lon[1], dtype=np.float32)

        # label dimensions for more information
        mean_data.dims[0].label = "Lead time"
        mean_data.dims[1].label = "Channel name"
        mean_data.dims[2].label = "Latitude in degrees"
        mean_data.dims[3].label = "Longitude in degrees"
        std_data.dims[0].label = "Lead time"
        std_data.dims[1].label = "Channel name"
        std_data.dims[2].label = "Latitude in degrees"
        std_data.dims[3].label = "Longitude in degrees"

        # close file
        file_handle.close()

        _barrier(self.device)

        return

    def update(self, data, idt):
        """update local buffers"""

        with torch.no_grad():
            data_projected = self.scale * data[..., self.channel_mask, :, :] + self.bias

            # compute the local variance and mean over the local batch dimension
            mean, m2, count = self._compute_stats(data_projected, dim=0)

            # do distributed welford
            self._welford_combine(mean, m2, count, idt)

        return


    def finalize(self):

        _barrier(self.device)

        # compute common mean
        self._aggregate_stats("data")

        std = torch.sqrt(self.running_var / (self.num_samples_tracked.to(dtype=torch.float32) - 1.0))
        mean = self.running_mean

        # write out bias/var
        if self.output_file is not None:
            self._write_output_data(mean.cpu().numpy(), std.cpu().numpy())

        _barrier(self.device)

        return


class SpectrumAverageBuffer(MeanStdBuffer):
    r"""
    SpectrumAverageBuffer class handles the recording of spatial data during inference. Performs Welford reduction on the fly during inference

    Parameters
    ============
    num_rollout_steps : int
        Number of total rollout steps
    img_shape: Tuple[int]
        shape of the output
    channel_names : List[str]
        list of all the channel names in the output
    device : Union[str, torch.device]
        Device on which inference is performed
    scale : torch.Tensor
        Scale for output normalization
    bias : torch.Tensor
        Bias for output normalization
    output_channels : List[str]
        List of channels to be recorded
    output_file: str, optional
        Outputfile to write to
    """
    def __init__(self,
        num_rollout_steps: int,
        rollout_dt: int,
        img_shape: Tuple[int, int],
        ensemble_size: int,
        grid_type: str,
        channel_names: List[str],
        device: Union[str, torch.device],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        spatial_distributed: Optional[bool] = False):

        # instantiate SHT
        self.spatial_distributed = spatial_distributed
        if self.spatial_distributed and (comm.get_size("spatial") > 1):
            if not thd.is_initialized():
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
            self.sht = thd.DistributedRealSHT(*img_shape, grid=grid_type)
            self.lmax = self.sht.lmax
            self.lmax_local = self.sht.l_shapes[comm.get_rank("h")]
            # compute loffset
            offsets = [0] + np.cumsum(self.sht.l_shapes).tolist()
            self.lmax_offset = offsets[comm.get_rank("h")]
        else:
            self.sht = th.RealSHT(*img_shape, grid=grid_type)
            self.lmax = self.sht.lmax
            self.lmax_local = self.lmax
            self.lmax_offset = 0

        # determine parameter size
        self.ensemble_size = ensemble_size
        enssize = self.ensemble_size
        if comm.get_rank("ensemble") == 0:
            enssize += 1
        variable_shape = (enssize, self.lmax_local)

        super().__init__(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=rollout_dt,
            variable_shape=variable_shape,
            channel_names=channel_names,
            device=device,
            scale=scale,
            bias=bias,
            output_channels=output_channels,
            output_file=output_file,)

        # move sht to device
        self.sht = self.sht.to(self.device)

        # reshape bias and scale
        self.bias = self.bias.reshape(1, 1, -1, 1, 1)
        self.scale = self.scale.reshape(1, 1, -1, 1, 1)

    def update(self, data, targ, idt):
        """update local buffers"""

        with torch.no_grad():
            if comm.get_rank("ensemble") == 0:
                data = torch.cat([targ, data], dim=1)

            # rescale data and project channels
            datap = self.scale * data[..., self.channel_mask, :, :] + self.bias

            # perform SHT in FP32 to be safe
            dtype = datap.dtype
            with amp.autocast(device_type=datap.device.type, enabled=False):
                sdatap = self.sht(datap.to(torch.float32))

                # compute power spectrum:
                sdatap = torch.square(torch.abs(sdatap))

                # be w-distributed aware:
                if comm.get_rank("w") == 0:
                    sdatap[..., 1:] *= 2.0
                else:
                    sdatap[...] *= 2.0

                # local sum over w
                spect = torch.sum(sdatap, dim=-1)

                # do distributed sum over w
                spect = reduce_from_parallel_region(spect, "w")

            # convert back precision
            spect = spect.to(dtype=dtype)

            # swap E and C
            spect = spect.permute(0,2,1,3).contiguous()

            # compute the local variance and mean over the local batch dimension
            mean, m2, count = self._compute_stats(spect, dim=0)

            # do distributed welford
            self._welford_combine(mean, m2, count, idt)

        return

    def finalize(self):

        _barrier(self.device)

        # compute common mean
        self._aggregate_stats("batch")

        std = torch.sqrt(self.running_var / (self.num_samples_tracked.to(dtype=torch.float32) - 1.0))
        mean = self.running_mean

        # write out bias/var
        if self.output_file is not None:
            self._write_output_data(mean.cpu().numpy(), std.cpu().numpy())

        _barrier(self.device)

        return

    def _write_output_data(self, mean, std):

        if self.mpi_comm is not None:
            # initialize MPI. This call is collective!
            file_handle = h5.File(self.output_file, "w", driver="mpio", comm=self.mpi_comm)
        else:
            file_handle = h5.File(self.output_file, "w")

        # create hdf5 dataset and write the data
        data_shape = (self.num_rollout_steps, self.num_channels, self.ensemble_size * comm.get_size("ensemble") + 1, self.lmax)
        mean_data = file_handle.create_dataset("mean", data_shape, dtype=np.float64)
        std_data = file_handle.create_dataset("std", data_shape, dtype=np.float64)

        # l range
        l_start = self.lmax_offset
        l_end = l_start + self.lmax_local
        l_range = slice(l_start, l_end)

        # ensemble range
        if comm.get_size("ensemble") > 1:
            if comm.get_rank("ensemble") == 0:
                ens_start = 0
                ens_end = self.ensemble_size + 1
            else:
                ens_start = comm.get_rank("ensemble") * self.ensemble_size + 1
                ens_end = ens_start + self.ensemble_size
            ens_range = slice(ens_start, ens_end)
        else:
            ens_range = slice(0, self.ensemble_size + 1)

        if (comm.get_rank("batch") == 0) and (comm.get_rank("w") == 0):
            mean_data[..., ens_range, l_range] = mean[...]
            std_data[..., ens_range, l_range] = std[...]

        # create leadtimes
        leadtime = file_handle.create_dataset("lead_time", self.num_rollout_steps, dtype=np.float64)
        leadtime.make_scale("lead_time")
        mean_data.dims[0].attach_scale(leadtime)
        std_data.dims[0].attach_scale(leadtime)
        dts = np.arange(self.rollout_dt, (self.num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64)
        if comm.get_world_rank() == 0:
            leadtime[:] = dts[:]

        # create channel descriptors, no need to store the handle to that since we can populate right away:
        chanlen = max([len(v) for v in self.output_channels])
        chans = file_handle.create_dataset("channel", self.num_channels, dtype=h5.string_dtype(length=chanlen))
        chans.make_scale("channel")
        mean_data.dims[1].attach_scale(chans)
        std_data.dims[1].attach_scale(chans)
        if comm.get_world_rank() == 0:
            chans[...] = self.output_channels

        # label dimensions for more information
        mean_data.dims[0].label = "Lead time"
        std_data.dims[0].label = "Lead time"
        mean_data.dims[1].label = "Channel name"
        std_data.dims[1].label = "Channel name"
        mean_data.dims[2].label = "Ensemble index"
        std_data.dims[2].label = "Ensemble index"
        mean_data.dims[3].label = "Spectral mode"
        std_data.dims[3].label = "Spectral mode"

        # close file
        file_handle.close()

        _barrier(self.device)

        return

class ZonalSpectrumAverageBuffer(MeanStdBuffer):
    r"""
    SpectrumAverageBuffer class handles the recording of spatial data during inference. Performs Welford reduction on the fly during inference

    Parameters
    ============
    num_rollout_steps : int
        Number of total rollout steps
    img_shape: Tuple[int]
        shape of the output
    channel_names : List[str]
        list of all the channel names in the output
    lat_lon: Tuple[List[float], List[float]]
        latitude and longitude coordinates
    device : Union[str, torch.device]
        Device on which inference is performed
    scale : torch.Tensor
        Scale for output normalization
    bias : torch.Tensor
        Bias for output normalization
    output_channels : List[str]
        List of channels to be recorded
    output_file: str, optional
        Outputfile to write to
    """
    def __init__(self,
        num_rollout_steps: int,
        rollout_dt: int,
        img_shape: Tuple[int, int],
        ensemble_size: int,
        channel_names: List[str],
        lat_lon: Tuple[List[float], List[float]],
        device: Union[str, torch.device],
	    scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_channels: List[str] = [],
        output_file: Optional[str] = None,
        spatial_distributed: Optional[bool] = False):

        # instantiate SHT
        self.spatial_distributed = spatial_distributed and comm.get_size("spatial") > 1
        self.nlat = img_shape[0]
        if self.spatial_distributed:
            if comm.get_size("w") > 1:
                self.rfft = DistributedRealFFT1(nlon=img_shape[1])
                m_shapes = self.rfft.m_shapes
                self.mmax_local = m_shapes[comm.get_rank("w")]
                # compute loffset
                offsets = [0] + np.cumsum(m_shapes).tolist()
                self.mmax_offset = offsets[comm.get_rank("w")]
            else:
                self.rfft = RealFFT1(nlon=img_shape[1])
                self.mmax_local = self.rfft.mmax
                self.mmax_offset = 0
            nlat_shapes = compute_split_shapes(self.nlat, comm.get_size("h"))
            self.nlat_local = nlat_shapes[comm.get_rank("h")]
            # compute loffset
            offsets = [0] + np.cumsum(nlat_shapes).tolist()
            self.nlat_offset = offsets[comm.get_rank("h")]
        else:
            self.rfft = RealFFT1(nlon=img_shape[1])
            self.mmax_local = self.rfft.mmax
            self.mmax_offset = 0
            self.nlat_local = self.nlat
            self.nlat_offset = 0

        # store lat/lon data
        self.lat_lon = lat_lon
        self.mmax = self.rfft.mmax

        # determine parameter size
        self.ensemble_size = ensemble_size
        enssize = self.ensemble_size
        if comm.get_rank("ensemble") == 0:
            enssize += 1
        variable_shape = (enssize, self.nlat_local, self.mmax_local)

        super().__init__(
            num_rollout_steps=num_rollout_steps,
            rollout_dt=rollout_dt,
            variable_shape=variable_shape,
            channel_names=channel_names,
            device=device,
            scale=scale,
            bias=bias,
            output_channels=output_channels,
            output_file=output_file,)

        # reshape bias and scale
        self.bias = self.bias.reshape(1, 1, -1, 1, 1)
        self.scale = self.scale.reshape(1, 1, -1, 1, 1)

        # we need to compute the latitude weighting factors
        lats = torch.tensor(self.lat_lon[0], dtype=torch.float32, device=self.device)
        # we need to use cos since we use latitude
        self.lat_weights = torch.cos(torch.deg2rad(lats))
        # split the tensor in lat dim
        if self.spatial_distributed and comm.get_size("h") > 1:
            self.lat_weights = split_tensor_along_dim(self.lat_weights, dim=0, num_chunks=comm.get_size("h"))[comm.get_rank("h")]
        # reshape:
        self.lat_weights = torch.reshape(self.lat_weights, (1,1,1,-1,1))

    def update(self, data, targ, idt):
        """update local buffers"""

        with torch.no_grad():
            if comm.get_rank("ensemble") == 0:
                data = torch.cat([targ, data], dim=1)

            # rescale data and project channels:
            datap = self.scale * data[..., self.channel_mask, :, :] + self.bias

            # distributed FFT
            dtype = datap.dtype
            with amp.autocast(device_type=datap.device.type, enabled=False):
                sdatap = self.rfft(datap.to(torch.float32), norm="forward")

                # compute power spectrum:
                spect = self.lat_weights * torch.square(torch.abs(sdatap))

                # be w-distributed aware:
                if comm.get_rank("w") == 0:
                    spect[..., 1:] *= 2.0
                else:
                    spect[...] *= 2.0

            # convert back precision
            spect = spect.to(dtype=dtype)

            # swap E and C
            spect = spect.permute(0,2,1,3,4).contiguous()

            # compute the local variance and mean over the local batch dimension
            mean, m2, count = self._compute_stats(spect, dim=0)

            # do distributed welford
            self._welford_combine(mean, m2, count, idt)

        return

    def finalize(self):

        _barrier(self.device)

        # compute common mean
        self._aggregate_stats("batch")

        std = torch.sqrt(self.running_var / (self.num_samples_tracked.to(dtype=torch.float32) - 1.0))
        mean = self.running_mean

        # write out bias/var
        if self.output_file is not None:
            self._write_output_data(mean.cpu().numpy(), std.cpu().numpy())

        _barrier(self.device)

        return

    def _write_output_data(self, mean, std):

        if self.mpi_comm is not None:
            # initialize MPI. This call is collective!
            file_handle = h5.File(self.output_file, "w", driver="mpio", comm=self.mpi_comm)
        else:
            file_handle = h5.File(self.output_file, "w")

        # create hdf5 dataset and write the data
        data_shape = (self.num_rollout_steps, self.num_channels, self.ensemble_size * comm.get_size("ensemble") + 1, self.nlat, self.mmax)
        mean_data = file_handle.create_dataset("mean", data_shape, dtype=np.float64)
        std_data = file_handle.create_dataset("std", data_shape, dtype=np.float64)

        # lat range
        lat_range = slice(self.nlat_offset, self.nlat_offset + self.nlat_local)
        # m-range
        m_range = slice(self.mmax_offset, self.mmax_offset + self.mmax_local)

        # ensemble range
        if comm.get_size("ensemble") > 1:
            if comm.get_rank("ensemble") == 0:
                ens_start = 0
                ens_end = self.ensemble_size + 1
            else:
                ens_start = comm.get_rank("ensemble") * self.ensemble_size + 1
                ens_end = ens_start + self.ensemble_size
            ens_range = slice(ens_start, ens_end)
        else:
            ens_range = slice(0, self.ensemble_size + 1)

        if comm.get_rank("batch") == 0:
            mean_data[..., ens_range, lat_range, m_range] = mean[...]
            std_data[..., ens_range, lat_range, m_range] = std[...]

        # create leadtimes
        leadtime = file_handle.create_dataset("lead_time", self.num_rollout_steps, dtype=np.float64)
        leadtime.make_scale("lead_time")
        mean_data.dims[0].attach_scale(leadtime)
        std_data.dims[0].attach_scale(leadtime)
        dts = np.arange(self.rollout_dt, (self.num_rollout_steps + 1) * self.rollout_dt, self.rollout_dt, dtype=np.float64)
        if comm.get_world_rank() == 0:
            leadtime[:] = dts[:]

        # create channel descriptors, no need to store the handle to that since we can populate right away:
        chanlen = max([len(v) for v in self.output_channels])
        chans = file_handle.create_dataset("channel", self.num_channels, dtype=h5.string_dtype(length=chanlen))
        chans.make_scale("channel")
        mean_data.dims[1].attach_scale(chans)
        std_data.dims[1].attach_scale(chans)
        if comm.get_world_rank() == 0:
            chans[...] = self.output_channels

         # create lon and lat descriptors
        lats = file_handle.create_dataset("lat", len(self.lat_lon[0]), dtype=np.float32)
        lats.make_scale("lat")
        mean_data.dims[3].attach_scale(lats)
        std_data.dims[3].attach_scale(lats)
        if comm.get_world_rank() == 0:
            lats[...] = np.array(self.lat_lon[0], dtype=np.float32)

        # label dimensions for more information
        mean_data.dims[0].label = "Lead time"
        std_data.dims[0].label = "Lead time"
        mean_data.dims[1].label = "Channel name"
        std_data.dims[1].label = "Channel name"
        mean_data.dims[2].label = "Ensemble index"
        std_data.dims[2].label = "Ensemble index"
        mean_data.dims[3].label = "Latitude"
        std_data.dims[3].label = "Latitude"
        mean_data.dims[4].label = "Spectral mode"
        std_data.dims[4].label = "Spectral mode"

        # close file
        file_handle.close()

        _barrier(self.device)

        return
