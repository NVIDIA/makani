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

"""
Stochastic physics: an SPPT-style multiplicative perturbation of the model tendency.

Inspired by ECMWF's Stochastically Perturbed Parametrization Tendencies (SPPT), this module
perturbs the *increment* the model produces rather than adding noise to the input:

    y = baseline + (1 + r) * (pred - baseline)

where ``baseline`` is the persistence forecast (the current state of the predicted variables)
and ``r`` is a smooth, band-limited AR(1) random field on the sphere (reusing the same noise
generators as the input noise). Because the perturbation scales the tendency ``pred - baseline``,
it vanishes where the model predicts no change and is co-located with active weather -- the
property that keeps NWP stochastic physics from imprinting grid-scale speckle.

The module is deliberately model-agnostic: it operates on the physical increment at the
time-stepping boundary (the stepper), so it works for any network regardless of whether it
predicts the full state or an internal residual.
"""

import numpy as np
import torch
import torch.nn as nn

from makani.models.noise import build_noise, noise_seed_reflect, run_eager

# Offset so the stochastic-physics noise stream is decorrelated from the input noise while
# keeping the same per-ensemble-member seeding structure.
_SPPT_SEED_OFFSET = 90001


class StochasticPhysics(nn.Module):
    """SPPT-style multiplicative tendency perturbation.

    Parameters
    ----------
    params : ParamsBase
        Global parameter object. Reads the ``stochastic_physics`` config block and the
        channel / grid metadata needed to reconstruct the persistence baseline.
    """

    def __init__(self, params):
        super().__init__()

        sp = params.stochastic_physics

        # clamp |r| so that (1 + r) stays positive (no sign flip of the tendency); None disables
        self.clip = sp.get("clip", None)

        # resolved noise scale info
        img_shape = [params.img_shape_x_resampled, params.img_shape_y_resampled]
        num_channels = sp.get("n_channels", 1)
        self.num_channels = num_channels

        # per-member seed on an independent stream from the input noise
        seed, reflect = noise_seed_reflect(sp.get("centered", False), seed_offset=_SPPT_SEED_OFFSET)

        # a single field per step, advanced as an AR(1) process over the rollout
        self.noise = build_noise(
            sp,
            img_shape=img_shape,
            batch_size=params.batch_size,
            num_channels=num_channels,
            num_time_steps=1,
            grid_type=params.model_grid_type,
            seed=seed,
            reflect=reflect,
            default_lambd=params.dt * params.dhours / 6.0,
        )

        # precompute the index that gathers the predicted variables' current-timestep state
        # out of the raw (physical) input, aligned 1:1 with the model output channels.
        self.register_buffer("baseline_index", self._build_baseline_index(params), persistent=False)

    @staticmethod
    def _build_baseline_index(params):
        r"""Map each output channel to its position in the current-timestep prognostic block.

        The raw input is laid out as ``[t-n_history, ..., t]`` with each timestep holding the
        prognostic (``in_channels``) block; static and unpredicted features are appended later
        by the preprocessor and are not present here. The persistence baseline is therefore the
        predicted variables of the *last* (current) timestep block.
        """
        n_history = params.n_history
        # channels per timestep in the raw prognostic input
        c_base = params.N_in_predicted_channels // (n_history + 1)

        in_ch = params.get("in_channels", None)
        out_ch = params.get("out_channels", None)
        if (in_ch is not None) and (out_ch is not None):
            in_ch = list(np.asarray(in_ch).ravel().tolist())
            out_ch = list(np.asarray(out_ch).ravel().tolist())
            missing = [c for c in out_ch if c not in in_ch]
            if missing:
                raise ValueError(
                    f"stochastic_physics requires every predicted channel to also be an input "
                    f"(prognostic) channel so a persistence baseline exists; missing: {missing}"
                )
            pos = [in_ch.index(c) for c in out_ch]
        else:
            # no explicit channel maps -> predicted vars are the leading channels of the block
            pos = list(range(params.N_out_channels))

        # offset into the current (last) timestep block
        idx = torch.as_tensor([n_history * c_base + p for p in pos], dtype=torch.long)
        return idx

    # --- stochastic-state management (delegated to the noise module) -----------------------

    def update(self, replace_state=False, batch_size=None):
        self.noise.update(replace_state=replace_state, batch_size=batch_size)

    def set_rng(self, seed=333, reset=True):
        self.noise.set_rng(seed)
        if reset:
            self.noise.reset()

    # --- perturbation ----------------------------------------------------------------------

    def _baseline(self, inp):
        torch._check(
            inp.shape[1] > int(self.baseline_index.max()),
            lambda: (
                f"StochasticPhysics: input has {inp.shape[1]} channels but the persistence "
                f"baseline needs index {int(self.baseline_index.max())}. The raw prognostic "
                f"input layout does not match the configured channels."
            ),
        )
        return inp.index_select(1, self.baseline_index)

    def forward(self, inp, pred):
        r"""Apply ``y = baseline + (1 + r) * (pred - baseline)``.

        ``inp`` is the raw (physical) input for the current step; ``pred`` is the denormalized
        model prediction of the target channels. Perturbation is applied in physical space
        where the increment is unambiguous (the affine normalization cancels in the difference).
        """
        baseline = self._baseline(inp).to(pred.dtype)

        # (B, T=1, C_noise, H, W) -> current step (B, C_noise, H, W)
        r = run_eager(self.noise)[:, -1].to(pred.dtype)

        if self.clip is not None:
            r = torch.clamp(r, -self.clip, self.clip)

        # broadcast a single shared pattern across all variables (classic SPPT), or apply a
        # per-channel pattern when n_channels matches the number of predicted channels.
        if self.num_channels not in (1, pred.shape[1]):
            raise ValueError(
                f"stochastic_physics.n_channels must be 1 (shared pattern) or "
                f"{pred.shape[1]} (per predicted channel), got {self.num_channels}"
            )

        return baseline + (1.0 + r) * (pred - baseline)
