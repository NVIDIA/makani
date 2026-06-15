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


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

import makani.utils.constants as const


class NonNegativeConstraint(nn.Module):
    """Enforce nonnegativity on a named subset of channels (dim -3, nchw layout).

    Normalization convention: x_norm = (x_raw - bias) / scale, so physical
    zero sits at x_norm = -bias/scale. The offset = bias/scale is precomputed
    in the constructor so the forward pass is cheap.

    Training mode: smooth multiplicative approximation x*sigmoid(x/eps) applied
    in the shifted (physical-zero-centered) space so gradients flow for slightly
    negative values.

    Eval/inference mode: hard clamp, guaranteeing x_raw >= 0 before any
    downstream conservation corrections.

    Args:
        channel_names:  full list of channel names for the model output tensor.
        names_to_clamp: list of channel names to enforce nonnegativity on.
                        Names not found in channel_names are silently skipped.
        bias:           normalization bias tensor, shape (1, C, 1, 1) or None.
        scale:          normalization scale tensor, shape (1, C, 1, 1) or None.
        eps:            transition width for the soft clamp (normalized units).
    """

    def __init__(self, channel_names, names_to_clamp, bias=None, scale=None, eps=0.1):
        super().__init__()
        self.eps = eps

        # resolve names to indices, skipping any not present in channel_names
        chan_idx = [channel_names.index(n) for n in names_to_clamp if n in channel_names]
        if not chan_idx:
            raise ValueError(f"None of the requested channel names {names_to_clamp} were found in channel_names.")
        self.register_buffer("channel_indices", torch.tensor(chan_idx, dtype=torch.long), persistent=False)

        if bias is not None and scale is not None:
            means = bias[0, chan_idx, 0, 0].to(torch.float32)
            stds  = scale[0, chan_idx, 0, 0].to(torch.float32)
            # offset = bias/scale; physical zero is at x_norm = -offset
            offset = (means / stds).view(1, -1, 1, 1)
            self.register_buffer("offset", offset, persistent=False)
        else:
            self.offset = None

    def forward(self, x):
        w = x[..., self.channel_indices, :, :]
        offset = self.offset.to(x.dtype) if self.offset is not None else None

        if self.training:
            # shift so physical zero maps to 0, apply smooth clamp, shift back
            w_shifted = w + offset if offset is not None else w
            w = w_shifted * torch.sigmoid(w_shifted / self.eps)
            if offset is not None:
                w = w - offset
        else:
            # hard clamp: x_norm >= -offset  <=>  x_raw >= 0
            lo = -offset if offset is not None else x.new_zeros(1)
            w = torch.clamp(w, min=lo)

        return x.index_copy(-3, self.channel_indices, w.to(x.dtype))


class HydrostaticBalanceProjection(nn.Module):
    r"""Softly enforce hydrostatic balance by projecting the (T, Z) sub-state onto
    the balance manifold, leaving the model free to predict T and Z independently.

    Unlike the reparametrizing ``_HydrostaticBalanceWrapper`` (which derives the
    full temperature column from the geopotentials and thereby discards L-1
    degrees of freedom), this layer takes the model's freely-predicted T and Z
    columns and applies the *minimum-norm* differentiable correction that makes
    them satisfy the discrete hydrostatic relation

        Z_i - Z_{i-1} = c_i (T_i + T_{i-1}),   c_i = 0.5 * R_dry * log(p_{i-1}/p_i).

    Stacking one column as x = [T_0..T_{L-1}, Z_0..Z_{L-1}] (physical units), the
    L-1 relations form a linear system A x = 0. The orthogonal projection onto
    ker A is x' = x - A^T (A A^T)^{-1} A x. The correction is distributed across
    both T and Z (minimum-norm) rather than dumped entirely onto T.

    The projection is carried out in *normalized* coordinates (metric W =
    diag(1/scale^2)) so a correction is measured in sigma units and shared
    fairly between variables whose physical magnitudes differ by orders of
    magnitude (geopotential ~ 1e4 vs temperature ~ 1e2). With the affine offset
    from the normalization bias folded in, the forward pass is a single 1x1
    conv plus a bias, with everything precomputed in the constructor.

    A ``strength`` (lambda) in [0, 1] damps the correction: 1.0 enforces balance
    exactly at the output, < 1.0 nudges toward the manifold while letting some
    residual imbalance through. Already-balanced states are left untouched at
    any strength.

    A ``climatology_offset`` makes the target manifold affine, ``A x = b_clim``
    instead of the homogeneous ``A x = 0``. This is useful for reanalysis data
    (e.g. ERA5 on pressure levels) which carries a systematic hydrostatic residual
    from the model-level -> pressure-level interpolation: pinning the residual to
    its climatological mean preserves that systematic imbalance (so scores against
    the reanalysis are not penalized) while still constraining the residual's
    fluctuations (so autoregressive rollouts stay on a consistent manifold). The
    offset is a per-interior-level residual b_clim in physical geopotential units
    (m^2/s^2); b_clim[i-1] is the expected value of
    ``(Z_i - Z_{i-1}) - c_i (Tv_i + Tv_{i-1})``. It is expected to be computed over
    *all* matching z/t pressure levels (as get_hydrostatic_balance_climatology does),
    so its length must be ``(#matching z/t levels) - 1``; the constructor verifies
    this and slices out the contiguous subset for the requested [p_min, p_max]
    window, relying on the metadata-consistent channel ordering rather than an
    explicit level file. ``None`` (default) recovers the homogeneous projection.

    Moist-air variant (``use_moist_air_formula=True``): the relation uses virtual
    temperature T_v = T (1 + eps q), which is bilinear in (T, q) and would break a
    linear projection. We hold the predicted specific humidity q fixed (balance
    should not rewrite moisture) and project in (T_v, Z) variables, where the
    constraint is *structurally identical* to the dry one and reuses the same
    operator: convert T -> T_v, project, then recover T = T_v / (1 + eps q).
    Because the T_v conversion is a physical-units operation, the whole projection
    is carried out in physical units (un-normalize -> project -> re-normalize);
    the weighted physical projection P = W^{-1} A^T (A W^{-1} A^T)^{-1} A with
    W = diag(1/scale^2) is algebraically identical to the normalized-space form,
    so the dry path is unchanged.

    Args:
        channel_names:         full list of channel names for the model output.
        bias:                  normalization bias tensor, shape (1, C, 1, 1) or None.
        scale:                 normalization scale tensor, shape (1, C, 1, 1) or None.
        p_min, p_max:          pressure-level window (hPa) to include.
        strength:              damping factor lambda in [0, 1] (default 1.0 = exact).
        use_moist_air_formula: use virtual temperature (requires matching q levels).
        climatology_offset:    optional per-interior-level residual b_clim as a
                               torch.Tensor (physical units; like bias/scale) over ALL
                               matching z/t levels (length (#matching z/t levels) - 1),
                               defining an affine target manifold A x = b_clim. Sliced
                               internally to the [p_min, p_max] window. None (default)
                               -> A x = 0.
    """

    def __init__(self, channel_names, bias=None, scale=None, p_min=50, p_max=900, strength=1.0,
                 use_moist_air_formula=False, climatology_offset=None):
        super().__init__()

        self.strength = float(strength)
        self.use_moist_air_formula = use_moist_air_formula

        # matching z/t pressure levels (descending pressure, surface -> top)
        z_idx, t_idx, pressures = get_matching_channels_pl(channel_names, "z", "t", p_min, p_max)
        if len(pressures) < 2:
            raise ValueError("Error, need at least two overlapping z/t pressure levels to enforce hydrostatic balance.")

        L = len(pressures)

        # channel order of the gathered sub-state: [T_0..T_{L-1}, Z_0..Z_{L-1}]
        all_idx = t_idx + z_idx
        self.register_buffer("channel_indices", torch.tensor(all_idx, dtype=torch.long), persistent=False)

        # moist air: matching q levels (must coincide with the z/t levels)
        if self.use_moist_air_formula:
            q_idx, _, q_pressures = get_matching_channels_pl(channel_names, "q", "t", p_min, p_max)
            for p1, p2 in zip(pressures, q_pressures):
                if p1 != p2:
                    raise ValueError("Error, make sure that you have the same pressure levels for t, z and q channels")
            self.q_prefact = const.Q_CORRECTION_MOIST_AIR
            self.register_buffer("q_indices", torch.as_tensor(q_idx, dtype=torch.long), persistent=False)

        # physical-units constraint matrix A: (L-1) x 2L, columns [T (or Tv)..., Z...]
        # built once at construction in float64 for an accurate (ill-conditioned) inverse;
        # the resulting operators are stored as fp32 buffers and forward() is pure torch.
        A = torch.zeros((L - 1, 2 * L), dtype=torch.float64)
        for i in range(1, L):
            c_i = 0.5 * const.R_DRY_AIR * math.log(pressures[i - 1] / pressures[i])
            r = i - 1
            A[r, L + i] = 1.0  # Z_i
            A[r, L + i - 1] = -1.0  # Z_{i-1}
            A[r, i] = -c_i  # T_i
            A[r, i - 1] = -c_i  # T_{i-1}

        # normalization of the gathered sub-state (defaults: identity), CPU float64
        def _gather(stat, idx, default):
            if stat is not None:
                return stat[0, idx, 0, 0].to(device="cpu", dtype=torch.float64)
            return torch.full((len(idx),), default, dtype=torch.float64)

        scale_sub = _gather(scale, all_idx, 1.0)
        bias_sub = _gather(bias, all_idx, 0.0)

        # weighted physical projection with metric W = diag(1/scale^2):
        #   x' = x - lambda * M (A x - b_clim),   M = W^{-1} A^T (A W^{-1} A^T)^{-1}.
        # Split into P = M A (the projection) and off = M b_clim (the affine shift):
        #   x' = x - lambda * (P x - off).
        # With b_clim = 0 this is the homogeneous projection (off = 0), algebraically
        # identical to the normalized-space form.
        Winv = scale_sub**2
        AWinv = A * Winv[None, :]
        M = (A.T * Winv[:, None]) @ torch.linalg.inv(AWinv @ A.T)  # (2L, L-1)
        Pphys = M @ A  # (2L, 2L)

        # affine offset onto the target manifold A x = b_clim
        if climatology_offset is not None:
            b_clim_full = climatology_offset.to(device="cpu", dtype=torch.float64).reshape(-1)
            # The climatology is expected to be computed over ALL matching z/t levels (the
            # convention of get_hydrostatic_balance_climatology), so its length is one per
            # interior level over the full set. Verify that, then slice out the contiguous
            # subset for this [p_min, p_max] window -- relying, like the other stats, on the
            # metadata-consistent channel ordering rather than an explicit level file.
            _, _, pressures_all = get_matching_channels_pl(channel_names, "z", "t", float("-inf"), float("inf"))
            if b_clim_full.shape[0] != len(pressures_all) - 1:
                raise ValueError(
                    f"climatology_offset must have length {len(pressures_all) - 1} (one per interior level "
                    f"over all {len(pressures_all)} matching z/t pressure levels), got {b_clim_full.shape[0]}."
                )
            start = pressures_all.index(pressures[0])
            b_clim = b_clim_full[start : start + (L - 1)]
            off = M @ b_clim  # (2L,)
        else:
            off = torch.zeros(2 * L, dtype=torch.float64)

        # store as fp32 buffers; forward runs in fp32 regardless of AMP
        self.register_buffer("proj", Pphys.float(), persistent=False)
        self.register_buffer("off", off.float().view(1, -1, 1, 1), persistent=False)
        self.register_buffer("scale_sub", scale_sub.float().view(1, -1, 1, 1), persistent=False)
        self.register_buffer("bias_sub", bias_sub.float().view(1, -1, 1, 1), persistent=False)
        if self.use_moist_air_formula:
            q_scale = _gather(scale, q_idx, 1.0)
            q_bias = _gather(bias, q_idx, 0.0)
            self.register_buffer("q_scale", q_scale.float().view(1, -1, 1, 1), persistent=False)
            self.register_buffer("q_bias", q_bias.float().view(1, -1, 1, 1), persistent=False)
        self.num_levels = L

    def forward(self, x):
        L = self.num_levels

        # gather [T..., Z...] sub-state (normalized model output)
        y = x[..., self.channel_indices, :, :]

        # The projection mixes large physical geopotentials with temperatures via
        # an ill-conditioned operator; keep everything in fp32 even under AMP.
        with amp.autocast(device_type=x.device.type, enabled=False):
            y = y.float()

            # un-normalize to physical units
            x_phys = y * self.scale_sub + self.bias_sub

            # convert temperature -> virtual temperature using the (fixed) humidity,
            # so the constraint becomes the same linear relation in (Tv, Z)
            if self.use_moist_air_formula:
                q_phys = x[..., self.q_indices, :, :].float() * self.q_scale + self.q_bias
                mult = torch.cat([1.0 + self.q_prefact * q_phys, torch.ones_like(x_phys[..., L:, :, :])], dim=-3)
                x_phys = x_phys * mult

            # minimum-(weighted-)norm correction onto the (affine) balance manifold
            corr = F.conv2d(x_phys, self.proj.unsqueeze(-1).unsqueeze(-1)) - self.off
            x_phys = x_phys - self.strength * corr

            # recover temperature from virtual temperature (q held fixed)
            if self.use_moist_air_formula:
                x_phys = x_phys / mult

            # re-normalize
            y = (x_phys - self.bias_sub) / self.scale_sub

        return x.index_copy(-3, self.channel_indices, y.to(x.dtype))


# this routine computes the matching pressure levels between two pl variables
# with prefix1 and prefix 2 respectively. pmin and pmax are the minimum and maximum pressure levels considered
def get_matching_channels_pl(channel_names, prefix1, prefix2, p_min, p_max, revert=True):
    # we better use regexp
    import re

    # analyse list of channel names, extract geopotential and temperatures:
    p1_pat = re.compile(r"^" + prefix1 + r"\d{1,}$")
    p2_pat = re.compile(r"^" + prefix2 + r"\d{1,}$")
    p1_chans = [x for x in channel_names if (p1_pat.match(x) is not None)]
    p2_chans = [x for x in channel_names if (p2_pat.match(x) is not None)]

    # extract common pressure levels
    p1_pressures = [int(x.replace(prefix1, "")) for x in p1_chans]
    p2_pressures = [int(x.replace(prefix2, "")) for x in p2_chans]

    # check which are the common pressure levels:
    pressures = sorted([x for x in p1_pressures if ((x in p2_pressures) and (x >= p_min) and (x <= p_max))], reverse=revert)

    # create an indexlist for z-channels
    p1_idx = [channel_names.index(f"{prefix1}{p}") for p in pressures]
    p2_idx = [channel_names.index(f"{prefix2}{p}") for p in pressures]

    return p1_idx, p2_idx, pressures
