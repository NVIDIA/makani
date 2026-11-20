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
from typing import Optional
import time
import json
import numpy as np
import h5py as h5
import argparse as ap
from glob import glob

# MPI
from makani.datasets.parallel import get_comm, init_torch_distributed, shutdown_torch_distributed

import torch
import torch.distributed as dist
from torch_harmonics import RealSHT

from makani.utils.grids import make_grid
from makani.data.grid_types import canonical_grid_type

from makani.datasets.helpers import DistributedProgressBar, file_sample_ranges
from makani.datasets.helpers import welford_combine, collective_reduce, binary_reduce


@torch.compile(fullgraph=True)
def compute_powerspectrum(x, sht):
    coeffs = torch.square(torch.abs(sht(x)))
    coeffs[..., 1:] *= 2.0
    power_spectrum = torch.sum(coeffs, dim=-1)

    return power_spectrum


def get_file_power_spectra(
    filename,
    file_slice,
    sht,
    batch_size: Optional[int] = 16,
    device: Optional[torch.device] = torch.device("cpu"),
    progress: Optional[DistributedProgressBar] = None,
) -> dict:

    power_spectra = None
    with h5.File(filename, "r") as f:

        # get dataset
        dset = f["fields"]

        # create batch
        slc_start = file_slice.start
        slc_stop = file_slice.stop
        if slc_stop is None:
            slc_stop = dset.shape[0]

        if batch_size is None:
            batch_size = slc_stop - slc_start

        for batch_start in range(slc_start, slc_stop, batch_size):
            batch_stop = min(batch_start + batch_size, slc_stop)
            sub_slc = slice(batch_start, batch_stop)

            # get slice
            data = dset[sub_slc, ...]
            tdata = torch.from_numpy(data).to(device=device, dtype=torch.float64)

            # check for NaNs
            if torch.isnan(tdata).any():
                raise ValueError(f"NaN values encountered in {filename}.")

            # compute sht of data
            power_spectrum = compute_powerspectrum(tdata, sht)

            # # define counts
            counts_time = torch.as_tensor(tdata.shape[0], dtype=torch.float64, device=device)

            # Basic observables
            # compute mean and variance
            # the mean needs to be divided by number of valid samples:
            power_spectrum_mean = torch.sum(power_spectrum, dim=0, keepdim=False) / counts_time
            # we compute m2 directly, so we do not need to divide by number of valid samples:
            power_spectrum_m2 = torch.sum(
                torch.square(power_spectrum - power_spectrum_mean[None, ...]), dim=0, keepdim=False
            )

            # fill the dict
            tmpspectra = dict(
                global_meanvar={
                    "type": "meanvar",
                    "counts": counts_time.clone(),
                    "values": torch.stack([power_spectrum_mean, power_spectrum_m2], dim=0).contiguous(),
                }
            )

            if power_spectra is not None:
                power_spectra = welford_combine(power_spectra, tmpspectra)
            else:
                power_spectra = tmpspectra

            if progress is not None:
                progress.update_counter(batch_stop - batch_start)
                progress.update_progress()

    return power_spectra


def compute_power_spectra(
    input_path: str,
    output_path: str,
    metadata_file: str,
    batch_size: Optional[int] = 16,
    reduction_group_size: Optional[int] = 8,
):
    """Function to compute power spectra of all variables of a makani HDF5 dataset.

    This function reads data from input_path and computes power spectra
    for all variables in the dataset. This is done globally, meaning averaged over space and time.
    They are stored as power_spectra_means.npy and power_spectra_stds.npy.

    It runs serially, or in parallel under mpirun where mpi4py is installed, see :mod:`makani.datasets.parallel`. For numerically safe reductions, it uses parallel Welford variance computation.

    ...

    Parameters
    ----------
    input_path : str
        Path which hosts the HDF5 files to compute the statistics on. Note, this routine supports virtual datasets genrated using concatenate_dataset.py.
        If you want to use a concatenated dataset, please specify the full path including the filename, e.g. <path-to-data>/train.h5v. In this case,
        the routine will ignore all the other files in the same folder.
    output_path : str
        Output path to specify where to store the computed statistics.
    metadata_file : str
        name of the file to read metadata from. The metadata is a json file, and after reading it should be a
        dictionary containing metadata describing the dataset. Most important entries are:
        dhours: distance between subsequent samples in hours
        coords: this is a dictionary which contains two lists, latitude and longitude coordinates in degrees as well as channel names.
        Example: coords = dict(lat=[-90.0, ..., 90.], lon=[0, ..., 360], channel=["t2m", "u500", "v500", ...])
        Note that the number of entries in coords["lat"] has to match dimension -2 of the dataset, and coords["lon"] dimension -1.
        The length of the channel names has to match dimension -3 (or dimension 1, which is the same) of the dataset.
    batch_size : int
        Batch size in which the samples are processed. This does not have any effect on the statistics (besides small numerical changes because of order of operations), but
        is merely a performance setting. Bigger batches are more efficient but require more memory.
    reduction_group_size : int
        Reduction group size for the parallel Welford reduction, clamped to a divisor of the number of ranks. The ranks are partitioned accordingly. Changing this value impacts performance but not numerical accuracy.
    """

    # disable gradients globally
    torch.set_grad_enabled(False)

    # get comm
    comm = get_comm()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # torch.distributed for the reductions
    device, mesh = init_torch_distributed(comm, reduction_group_size)

    # get files
    filelist = None
    data_shape = None
    num_samples = None
    channel_names = None
    combined_file = None
    grid_type = None
    if comm_rank == 0:
        if os.path.isdir(input_path):
            combined_file = False
            filelist = sorted(glob(os.path.join(input_path, "*.h5")))
            if not filelist:
                raise FileNotFoundError(f"Error, directory {input_path} is empty.")

            # open the first file to check for stats
            num_samples = []
            for filename in filelist:
                with h5.File(filename, "r") as f:
                    data_shape = f["fields"].shape
                    num_samples.append(data_shape[0])

        else:
            combined_file = True
            filelist = [input_path]
            with h5.File(filelist[0], "r") as f:
                data_shape = f["fields"].shape
                num_samples = [data_shape[0]]

        # open metadata file
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # read channel names
        channel_names = metadata["coords"]["channel"]
        grid_type = canonical_grid_type(metadata["coords"]["grid_type"], source=metadata_file)

    # communicate important information
    channel_names = comm.bcast(channel_names, root=0)
    filelist = comm.bcast(filelist, root=0)
    num_samples = comm.bcast(num_samples, root=0)
    data_shape = comm.bcast(data_shape, root=0)
    grid_type = comm.bcast(grid_type, root=0)

    # get file offsets
    num_samples_total = sum(num_samples)
    num_channels = data_shape[1]
    height, width = (data_shape[2], data_shape[3])

    # SHT:
    sht = RealSHT(make_grid((height, width), grid_type)).to(device)

    if comm_rank == 0:
        print(
            f"Found {len(filelist)} files with a total of {num_samples_total} samples. Each sample has the shape {num_channels}x{height}x{width} (CxHxW)."
        )

    # do the sharding:
    num_samples_chunk = (num_samples_total + comm_size - 1) // comm_size
    samples_start = num_samples_chunk * comm_rank
    samples_end = min([samples_start + num_samples_chunk, num_samples_total])
    # num_samples_local = samples_end - samples_start

    if comm_rank == 0:
        print("Loading data with the following chunking:")
    for rank in range(comm_size):
        if comm_rank == rank:
            print(f"Rank {comm_rank}, working on samples [{samples_start}, {samples_end})", flush=True)
        comm.Barrier()

    # the samples of this rank, as a range within each file
    mapping = file_sample_ranges(filelist, num_samples, samples_start, samples_end)

    # initialize arrays
    stats = dict(
        global_meanvar={
            "type": "meanvar",
            "counts": torch.as_tensor(0.0, dtype=torch.float64, device=device),
            "values": torch.zeros((2, num_channels, sht.lmax), dtype=torch.float64, device=device),
        },
    )

    # compute local stats
    progress = DistributedProgressBar(num_samples_total, comm)
    start = time.time()
    for filename, file_slice in mapping.items():
        tmpstats = get_file_power_spectra(filename, file_slice, sht, batch_size, device, progress)
        stats = welford_combine(stats, tmpstats)

    # wait for everybody else
    comm.Barrier()
    duration = time.time() - start
    if comm_rank == 0:
        print(f"Duration for {num_samples_total} samples: {duration:.2f}s", flush=True)
    del progress

    # do reductions within groups
    start = time.time()
    stats = collective_reduce(stats, mesh.get_group("reduction"))
    comm.Barrier()
    duration = time.time() - start
    if comm_rank == 0:
        print(f"Reduction within groups done. Duration: {duration:.2f}s", flush=True)

    # now, do binary reduction orthogonal to groups
    start = time.time()
    # only rank 0 of the allreduce group will do the binary reduction
    if dist.get_rank(mesh.get_group("reduction")) == 0:
        stats = binary_reduce(stats, mesh.get_group("tree"), device)
    comm.Barrier()
    duration = time.time() - start
    if comm_rank == 0:
        print(f"Reduction across groups done. Duration: {duration:.2f}s", flush=True)

    # write the data to disk
    if comm_rank == 0:
        start = time.time()

        # move stats to cpu and convert to numpy
        for varname, substats in stats.items():
            for k, v in substats.items():
                if isinstance(v, torch.Tensor):
                    stats[varname][k] = v.cpu().numpy()

        # compute global stds:
        stats["global_meanvar"]["values"][1, ...] = np.sqrt(
            stats["global_meanvar"]["values"][1, ...] / stats["global_meanvar"]["counts"]
        )

        # save the stats
        np.save(
            os.path.join(output_path, "power_spectra_means.npy"),
            stats["global_meanvar"]["values"][0, ...].astype(np.float32),
        )
        np.save(
            os.path.join(output_path, "power_spectra_stds.npy"),
            stats["global_meanvar"]["values"][1, ...].astype(np.float32),
        )

        duration = time.time() - start
        print(f"Saving stats done. Duration: {duration:.2f}s", flush=True)

        print("means: ", stats["global_meanvar"]["values"][0, ...])
        print("stds: ", stats["global_meanvar"]["values"][1, ...])

    # wait for rank 0 to finish
    comm.Barrier()

    # shut down pytorch comms
    shutdown_torch_distributed(device)


def main(args):
    compute_power_spectra(
        input_path=args.input_path,
        output_path=args.output_path,
        metadata_file=args.metadata_file,
        batch_size=args.batch_size,
        reduction_group_size=args.reduction_group_size,
    )

    return


if __name__ == "__main__":
    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="Directory with input files or a virtual hdf5 file with the combined input.",
        required=True,
    )
    parser.add_argument("--metadata_file", type=str, help="File containing dataset metadata.", required=True)
    parser.add_argument("--output_path", type=str, help="Directory for saving stats files.", required=True)
    parser.add_argument("--reduction_group_size", type=int, default=8, help="Size of collective reduction groups.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size used for reading chunks from a file at a time to avoid OOM errors.",
    )
    args = parser.parse_args()

    main(args)
