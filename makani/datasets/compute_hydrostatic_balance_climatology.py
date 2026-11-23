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
from makani.utils.grids import GridQuadrature, make_grid
from makani.data.grid_types import canonical_grid_type
import makani.models.constants as const
from makani.models.constraints import get_matching_channels_pl

from makani.datasets.helpers import DistributedProgressBar, file_sample_ranges

from makani.datasets.helpers import (
    mask_data,
    welford_combine,
    collective_reduce,
    binary_reduce,
)


@torch.compile()
def compute_residual(tdata, z_idx, t_idx, q_idx, coeffs, q_prefact):
    """Per-point hydrostatic-balance residual for each interior level.

        r_i = (Z_i - Z_{i-1}) - c_i (Tv_i + Tv_{i-1}),    i = 1 .. L-1
        c_i = 0.5 * R_dry * ln(p_{i-1} / p_i),            Tv = T (1 + eps q) [moist] or T [dry]

    This is exactly A x evaluated on the physical fields, so its climatological mean is
    the affine offset b_clim consumed by HydrostaticBalanceProjection. Inputs are in
    physical (un-normalized) units, as stored in the dataset.

    Args:
        tdata:     (B, C, H, W) physical fields.
        z_idx/t_idx: channel indices of the matching geopotential/temperature levels
                   (descending pressure), length L.
        q_idx:     channel indices of the matching specific-humidity levels, or None (dry).
        coeffs:    (L-1,) tensor of c_i.
        q_prefact: epsilon in Tv = T(1 + eps q).

    Returns:
        (B, L-1, H, W) residual.
    """
    Z = tdata[:, z_idx, ...]
    T = tdata[:, t_idx, ...]
    if q_idx is not None:
        q = tdata[:, q_idx, ...]
        Tv = T * (1.0 + q_prefact * q)
    else:
        Tv = T

    dZ = Z[:, 1:, ...] - Z[:, :-1, ...]
    sumTv = Tv[:, 1:, ...] + Tv[:, :-1, ...]
    return dZ - coeffs.view(1, -1, 1, 1) * sumTv


def get_file_stats(
    filename,
    file_slice,
    z_idx,
    t_idx,
    q_idx,
    coeffs,
    q_prefact,
    quadrature,
    fail_on_nan=False,
    batch_size=16,
    device=torch.device("cpu"),
    progress=None,
):

    stats = None
    n_interior = len(z_idx) - 1
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
            tdata = torch.as_tensor(data).to(device=device, dtype=torch.float64)

            # check for NaNs
            if fail_on_nan and torch.isnan(tdata).any():
                raise ValueError(f"NaN values encountered in {filename}.")

            # per-point hydrostatic residual (B, n_interior, H, W), physical units
            residual = compute_residual(tdata, z_idx, t_idx, q_idx, coeffs, q_prefact)

            # get imputed valid data and valid mask
            res_masked, valid_mask = mask_data(residual)

            # counts
            counts_time = residual.shape[0]
            valid_count = torch.sum(quadrature(valid_mask), dim=0)

            # area-weighted, time-summed mean and m2 (global meanvar over time + space)
            rmean = (
                torch.sum(quadrature(res_masked * valid_mask), dim=0, keepdim=False).reshape(1, -1, 1, 1)
                / valid_count[None, :, None, None]
            )
            rm2 = torch.sum(quadrature(torch.square(res_masked - rmean) * valid_mask), dim=0, keepdim=False).reshape(
                1, -1, 1, 1
            )

            tmpstats = dict(
                # spatially-resolved temporal mean of the residual (climatology field)
                time_means={
                    "type": "mean",
                    "counts": float(counts_time) * torch.ones((n_interior), dtype=torch.float64, device=device),
                    "values": torch.mean(residual, dim=0, keepdim=True),
                },
                # global (time + space) mean and variance of the residual per interior level
                global_meanvar={
                    "type": "meanvar",
                    "counts": valid_count.clone(),
                    "values": torch.stack([rmean, rm2], dim=0).contiguous(),
                },
            )
            del res_masked, valid_mask, residual

            if stats is not None:
                stats = welford_combine(stats, tmpstats)
            else:
                stats = tmpstats

            if progress is not None:
                progress.update_counter(batch_stop - batch_start)
                progress.update_progress()

    return stats


def compute_hydrostatic_balance_climatology(
    input_path: str,
    output_path: str,
    metadata_file: str,
    p_min: Optional[float] = None,
    p_max: Optional[float] = None,
    use_moist_air_formula: bool = False,
    fail_on_nan: bool = False,
    batch_size: Optional[int] = 16,
    reduction_group_size: Optional[int] = 8,
):
    """Compute the climatology of the hydrostatic-balance residual of a makani HDF5 dataset.

    For every matching geopotential/temperature (and, if requested, specific-humidity) pressure
    level in the window [p_min, p_max], this evaluates the per-point hydrostatic residual

        r_i = (Z_i - Z_{i-1}) - c_i (Tv_i + Tv_{i-1}),   c_i = 0.5 R_dry ln(p_{i-1}/p_i),

    on the physical fields and averages it. Two artifacts are written:

      * ``hydrostatic_balance_means.npy``  -- the global (time + area-weighted space) mean residual
        per interior level, shape (1, L-1, 1, 1). This is the affine offset ``b_clim`` consumed by
        ``HydrostaticBalanceProjection(climatology_offset=...)``.
      * ``hydrostatic_balance_time_means.npy`` -- the temporal mean residual keeping spatial
        structure, shape (1, L-1, H, W), for a future latitude-dependent offset.

    The (trivial here, but cheaply available) standard deviation is also written to
    ``hydrostatic_balance_stds.npy``, and the matched pressure levels to
    ``hydrostatic_balance_pressures.npy`` so consumers can align the per-level values unambiguously.

    Mirrors ``makani.datasets.compute_stats``: spatial averages use spherical quadrature, and reductions across
    ranks use parallel Welford combination. ``use_moist_air_formula`` selects virtual temperature
    and must match the setting used by the projection that consumes the offset.
    """

    # unspecified pressure bounds -> include all available levels
    if p_min is None:
        p_min = -float("inf")
    if p_max is None:
        p_max = float("inf")

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
    grid_type = None
    combined_file = None
    if comm_rank == 0:
        if os.path.isdir(input_path):
            combined_file = False
            filelist = sorted(glob(os.path.join(input_path, "*.h5")))
            if not filelist:
                raise FileNotFoundError(f"Error, directory {input_path} is empty.")

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
    grid_type = comm.bcast(grid_type, root=0)
    filelist = comm.bcast(filelist, root=0)
    num_samples = comm.bcast(num_samples, root=0)
    data_shape = comm.bcast(data_shape, root=0)

    # resolve matching z/t (and q) pressure levels (descending pressure, surface -> top)
    z_idx, t_idx, pressures = get_matching_channels_pl(channel_names, "z", "t", p_min, p_max)
    if len(pressures) < 2:
        raise ValueError("Error, need at least two overlapping z/t pressure levels in [p_min, p_max].")
    q_idx = None
    q_prefact = const.Q_CORRECTION_MOIST_AIR
    if use_moist_air_formula:
        q_idx, _, q_pressures = get_matching_channels_pl(channel_names, "q", "t", p_min, p_max)
        for p1, p2 in zip(pressures, q_pressures):
            if p1 != p2:
                raise ValueError("Error, make sure that you have the same pressure levels for t, z and q channels")

    # hydrostatic coefficients c_i (descending pressure)
    coeffs = torch.tensor(
        [0.5 * const.R_DRY_AIR * np.log(pressures[i - 1] / pressures[i]) for i in range(1, len(pressures))],
        dtype=torch.float64,
        device=device,
    )
    n_interior = len(pressures) - 1

    # get file offsets
    num_samples_total = sum(num_samples)
    height, width = (data_shape[2], data_shape[3])

    # quadrature: normalized to 4pi (we divide the integral by the valid area to obtain a mean)
    quadrature = GridQuadrature(
        make_grid((height, width), grid_type), crop_shape=None, crop_offset=(0, 0), normalize=False
    ).to(device)

    if comm_rank == 0:
        print(
            f"Found {len(filelist)} files with a total of {num_samples_total} samples. "
            f"Computing the hydrostatic residual over {n_interior} interior levels "
            f"(pressures {pressures}, moist={use_moist_air_formula})."
        )

    # do the sharding:
    num_samples_chunk = (num_samples_total + comm_size - 1) // comm_size
    samples_start = num_samples_chunk * comm_rank
    samples_end = min([samples_start + num_samples_chunk, num_samples_total])

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
            "counts": torch.zeros((n_interior), dtype=torch.float64, device=device),
            "values": torch.zeros((2, 1, n_interior, 1, 1), dtype=torch.float64, device=device),
        },
        time_means={
            "type": "mean",
            "counts": torch.zeros((n_interior), dtype=torch.float64, device=device),
            "values": torch.zeros((1, n_interior, height, width), dtype=torch.float64, device=device),
        },
    )

    # compute local stats
    progress = DistributedProgressBar(num_samples_total, comm)
    start = time.time()
    for filename, file_slice in mapping.items():
        tmpstats = get_file_stats(
            filename,
            file_slice,
            z_idx,
            t_idx,
            q_idx,
            coeffs,
            q_prefact,
            quadrature,
            fail_on_nan,
            batch_size,
            device,
            progress,
        )
        stats = welford_combine(stats, tmpstats)

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

        # convert m2 to std
        stats["global_meanvar"]["values"][1, ...] = np.sqrt(
            stats["global_meanvar"]["values"][1, ...] / stats["global_meanvar"]["counts"][None, :, None, None]
        )

        # save the climatology offset b_clim and friends
        np.save(
            os.path.join(output_path, "hydrostatic_balance_means.npy"),
            stats["global_meanvar"]["values"][0, ...].astype(np.float32),
        )
        np.save(
            os.path.join(output_path, "hydrostatic_balance_stds.npy"),
            stats["global_meanvar"]["values"][1, ...].astype(np.float32),
        )
        np.save(
            os.path.join(output_path, "hydrostatic_balance_time_means.npy"),
            stats["time_means"]["values"].astype(np.float32),
        )
        np.save(os.path.join(output_path, "hydrostatic_balance_pressures.npy"), np.asarray(pressures, dtype=np.float32))

        duration = time.time() - start
        print(f"Saving stats done. Duration: {duration:.2f}s", flush=True)

        print("pressures (descending): ", pressures)
        print("hydrostatic residual means (b_clim): ", stats["global_meanvar"]["values"][0, 0, :, 0, 0])
        print("hydrostatic residual stds:           ", stats["global_meanvar"]["values"][1, 0, :, 0, 0])

    # wait for rank 0 to finish
    comm.Barrier()

    # shut down pytorch comms
    shutdown_torch_distributed(device)


def main(args):
    compute_hydrostatic_balance_climatology(
        input_path=args.input_path,
        output_path=args.output_path,
        metadata_file=args.metadata_file,
        p_min=args.p_min,
        p_max=args.p_max,
        use_moist_air_formula=args.use_moist_air_formula,
        fail_on_nan=args.fail_on_nan,
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
        "--p_min",
        type=float,
        default=None,
        help="Minimum pressure level (hPa) to include. If unspecified, includes all available levels.",
    )
    parser.add_argument(
        "--p_max",
        type=float,
        default=None,
        help="Maximum pressure level (hPa) to include. If unspecified, includes all available levels.",
    )
    parser.add_argument(
        "--use_moist_air_formula",
        action="store_true",
        help="Use virtual temperature Tv = T(1 + eps q); requires matching q pressure levels.",
    )
    parser.add_argument(
        "--fail_on_nan", action="store_true", help="When computing stats, code will fail if NaN values are encountered."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size used for reading chunks from a file at a time to avoid OOM errors.",
    )
    args = parser.parse_args()

    main(args)
