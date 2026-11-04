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

from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn

from makani.utils.grids import GridQuadrature, grid_to_quadrature_rule
from makani.mpu.mappings import copy_to_parallel_region

from makani.models.preprocessor_helpers import get_bias_correction, get_static_features
from makani.models.noise_controller import noise_base_seed
from makani.utils.dataloaders.batch import ZENITH

# the name of the input noise among the noise processes of a model
INPUT_NOISE = "input"


class HistoryStats(NamedTuple):
    r"""
    Normalization statistics of one history window.

    ``mean`` and ``std`` are ``(B, 1, C, 1, 1)``, so that they broadcast over the
    time and grid dimensions, or zeros and ones of shape ``(1, 1, 1, 1, 1)`` in
    the ``"none"`` mode; the ``"timediff"`` mode fills ``diff_mean`` and
    ``diff_var`` instead.
    """

    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    diff_mean: Optional[torch.Tensor] = None
    diff_var: Optional[torch.Tensor] = None


class Preprocessor2D(nn.Module):
    r"""
    Input/output plumbing that surrounds the model in one step of a rollout.

    Sits between the data and the network and owns everything that is not the
    model itself: dynamic and static features, noise injection, bias
    correction, and the normalization of the history window.

    Everything here works on tensors with an explicit time axis,
    ``(B, T, C, H, W)``. The time steps are folded into the channels in exactly
    one place, :meth:`pack_model_input`, right before the network, which still
    takes a single flat channel stack.

    The preprocessor keeps no state between steps: the input window and its
    dynamic features come from a
    :class:`~makani.utils.dataloaders.rollout.Rollout`, which also slides the
    window forward, and the history statistics of a step are returned to the
    caller rather than stored. The one stateful part is the input noise, which
    the preprocessor owns but does not drive: the
    :class:`~makani.models.noise_controller.NoiseController` of the model
    advances it, see :meth:`noise_processes`, and the model passes its sample
    to :meth:`assemble_input`.

    The main responsibilities are:

    * **Dynamic features.** Quantities the model consumes but does not
      predict, such as the solar zenith angle. Which ones the model takes is
      fixed by the configuration (:attr:`dynamic_features`); their values come
      with the step and are appended to the input (:meth:`assemble_input`).
    * **Static features.** Time-invariant fields such as orography and the
      land-sea mask, appended to every packed input.
    * **Noise.** Optional stochastic perturbation of the input, either
      concatenated as extra channels or added to selected channels, for
      ensemble forecasting.
    * **History normalization.** Normalization of the input window with its
      own statistics (:meth:`compute_history_stats`), undone on the prediction.
    * **Bias correction.** An optional additive correction applied to the
      model output.

    Parameters
    ----------
    params : ParamsBase
        Configuration object. Relevant keys include the image shape and
        resampled shape, ``n_history``, ``history_normalization_mode``,
        ``input_noise``, and the static-feature and bias-correction settings.
    """

    def __init__(self, params):
        super().__init__()

        # image shape — must be set first; used by quadrature and noise constructors below
        self.img_shape = [params.img_shape_x, params.img_shape_y]
        self.img_shape_resampled = [params.img_shape_x_resampled, params.img_shape_y_resampled]

        self.subsampling_factor = params.get("subsampling_factor", 1)
        self.n_history = params.n_history
        self.history_normalization_mode = params.history_normalization_mode
        if self.history_normalization_mode == "exponential":
            self.history_normalization_decay = params.history_normalization_decay
            # inverse ordering, since first element is oldest
            history_normalization_weights = torch.exp(
                (-self.history_normalization_decay)
                * torch.arange(start=self.n_history, end=-1, step=-1, dtype=torch.float32)
            )
            history_normalization_weights = history_normalization_weights / torch.sum(history_normalization_weights)
            history_normalization_weights = torch.reshape(history_normalization_weights, (1, -1, 1, 1, 1))
        elif self.history_normalization_mode == "mean":
            history_normalization_weights = torch.as_tensor(1.0 / float(self.n_history + 1), dtype=torch.float32)
            history_normalization_weights = torch.reshape(history_normalization_weights, (1, -1, 1, 1, 1))
        else:
            history_normalization_weights = torch.ones(self.n_history + 1, dtype=torch.float32)
        self.register_buffer("history_normalization_weights", history_normalization_weights, persistent=False)
        if self.history_normalization_mode != "none":
            self.quadrature = GridQuadrature(
                grid_to_quadrature_rule(params.model_grid_type),
                img_shape=self.img_shape_resampled,
                crop_shape=None,
                crop_offset=(0, 0),
                normalize=True,
                distributed=True,
            )

        self.history_eps = 1e-6

        # the dynamic features the model takes, in the order their channels are appended
        self.dynamic_features = [ZENITH] if params.get("add_zenith", False) else []

        # get bias correction
        bias = get_bias_correction(params)

        if bias is not None:
            # register static buffer
            self.register_buffer("bias_correction", bias, persistent=False)

        # process static features
        static_features = get_static_features(params)
        self.do_add_static_features = False
        if static_features is not None:

            # remember that we need static features
            self.do_add_static_features = True

            # register static buffer
            self.register_buffer("static_features", static_features, persistent=False)

        if params.get("input_noise", None) is not None:
            noise_params = params.input_noise

            # the seed of this rank, passed down as-is
            self.noise_base_seed, reflect = noise_base_seed(centered=noise_params.get("centered", False))

            if "type" not in noise_params:
                raise ValueError("Error, please specify an input noise type")

            self.input_noise_mode = noise_params.get("mode", "concatenate")

            if self.input_noise_mode == "concatenate":
                noise_channels = noise_params.get("n_channels", 1)
            elif self.input_noise_mode == "perturb":
                self.perturb_channels = noise_params.get("perturb_channels", params.channel_names)
                self.perturb_channels = [params.channel_names.index(ch) for ch in self.perturb_channels]
                noise_channels = len(self.perturb_channels)
            else:
                raise NotImplementedError(f"Error, input noise mode {self.input_noise_mode} not supported.")

            noise_lmax = noise_params.get("lmax", None)

            if noise_params["type"] == "diffusion":
                from makani.models.noise import DiffusionNoiseS2

                # set the spatio-temporal correlation length
                kT = noise_params.get("kT", 0.5 * (100 / 6370) ** 2)
                lambd = noise_params.get("lambd", params.dt * params.dhours / 6.0)

                self.input_noise = DiffusionNoiseS2(
                    img_shape=self.img_shape_resampled,
                    batch_size=params.batch_size,
                    num_channels=noise_channels,
                    num_time_steps=self.n_history + 1,
                    sigma=noise_params.get("sigma", 1.0),
                    kT=kT,  # use various scales
                    lambd=lambd,  # use suggestion here: tau=6h
                    grid_type=params.model_grid_type,
                    lmax=noise_lmax,
                    seed=self.noise_base_seed,
                    reflect=reflect,
                    learnable=noise_params.get("learnable", False),
                )
            elif noise_params["type"] == "white":
                from makani.models.noise import IsotropicGaussianRandomFieldS2

                self.input_noise = IsotropicGaussianRandomFieldS2(
                    img_shape=self.img_shape_resampled,
                    batch_size=params.batch_size,
                    num_channels=noise_channels,
                    num_time_steps=self.n_history + 1,
                    sigma=noise_params.get("sigma", 1.0),
                    alpha=noise_params.get("alpha", 0.0),
                    grid_type=params.model_grid_type,
                    lmax=noise_lmax,
                    seed=self.noise_base_seed,
                    reflect=reflect,
                    learnable=noise_params.get("learnable", False),
                )
            elif noise_params["type"] == "dummy":
                from makani.models.noise import DummyNoiseS2

                self.input_noise = DummyNoiseS2(
                    img_shape=self.img_shape_resampled,
                    batch_size=params.batch_size,
                    num_channels=noise_channels,
                    num_time_steps=self.n_history + 1,
                )
            else:
                raise NotImplementedError(f'Error, input noise type {noise_params["type"]} not supported.')

    def flatten_history(self, x):
        r"""
        Fold the time dimension into the channel dimension.

        Only for the boundary to code that still takes a flat channel stack, the
        network and the losses; everything in between keeps the time axis.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, T, C, H, W)``. Tensors that are already 4D
            are returned unchanged, so this is safe to call unconditionally.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, T * C, H, W)``.
        """
        # flatten input
        if x.dim() == 5:
            b_, t_, c_, h_, w_ = x.shape
            x = torch.reshape(x, (b_, t_ * c_, h_, w_))

        return x

    def expand_history(self, x, nhist):
        r"""
        Split a flattened channel stack back into time and channel dimensions.

        Inverse of :meth:`flatten_history`.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, T * C, H, W)``. Tensors that are already 5D
            are returned unchanged.
        nhist : int
            Number of time steps ``T`` to split off. The channel dimension must
            be divisible by this; the check is a ``torch._check`` so it survives
            as a runtime assertion under ``torch.compile`` rather than becoming
            a graph-breaking data-dependent branch.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, nhist, C, H, W)``.
        """
        if x.dim() == 4:
            b_, ct_, h_, w_ = x.shape
            # torch._check (rather than `if ...: raise`) so this stays a runtime
            # assertion under torch.compile instead of becoming a data-dependent
            # branch that breaks the graph.
            torch._check(
                ct_ % nhist == 0,
                lambda: (
                    f"expand_history: channel dim {ct_} is not divisible by nhist={nhist}. "
                    f"The flattened-history input may not match the preprocessor's expected "
                    f"n_history={self.n_history} (so ct_ should be a multiple of n_history+1={nhist})."
                ),
            )
            x = torch.reshape(x, (b_, nhist, ct_ // nhist, h_, w_))
        return x

    def _check_time_axis(self, x, caller):
        torch._check(
            x.dim() == 5,
            lambda: f"{caller}: expected a tensor with an explicit time axis, (B, T, C, H, W), got shape {tuple(x.shape)}",
        )

    def pack_model_input(self, x):
        r"""
        Pack the input into the single channel stack the network takes.

        The one place where the time axis is folded into the channels: the time
        steps are stacked oldest first, and the static features, which have no
        time axis, are appended after them.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, C, H, W)``, with the dynamic features
            already appended.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, T * C + n_static, H, W)``.
        """
        self._check_time_axis(x, "pack_model_input")

        x = self.flatten_history(x)

        if self.do_add_static_features:
            # the same fields for every sample
            static = torch.tile(self.static_features, dims=(x.shape[0], 1, 1, 1))
            x = torch.cat([x, static], dim=1)

        return x

    def assemble_input(self, x, dynamic_features: Dict[str, torch.Tensor], noise: Optional[torch.Tensor] = None):
        r"""
        Append the dynamic features and the noise to the input.

        The features and the noise sample are passed in, as a
        :class:`~makani.utils.dataloaders.rollout.RolloutStep` carries them.
        The channels are appended in a fixed order,
        ``[x | dynamic features, in the order of dynamic_features | noise]``;
        in ``perturb`` mode the noise is added to the selected channels instead.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, n_history + 1, C, H, W)``.
        dynamic_features : dict
            The features of the window by name, each ``(B, n_history + 1, C_f, H, W)``.
            Has to hold exactly the features the model takes, :attr:`dynamic_features`.
        noise : torch.Tensor, optional
            The noise sample for this window. Required if, and only if, the
            preprocessor has input noise.

        Returns
        -------
        torch.Tensor
            Input with the appended channels, ``(B, n_history + 1, C', H, W)``.

        Raises
        ------
        ValueError
            If the features are not the ones the model takes, or the noise is
            missing or unexpected.
        """
        self._check_time_axis(x, "assemble_input")

        if set(dynamic_features) != set(self.dynamic_features):
            raise ValueError(
                f"assemble_input: the model takes the dynamic features {sorted(self.dynamic_features)}, "
                f"got {sorted(dynamic_features)}"
            )
        if hasattr(self, "input_noise") != (noise is not None):
            raise ValueError(
                "assemble_input: the preprocessor has input noise, but no noise sample was given"
                if noise is None
                else "assemble_input: a noise sample was given, but the preprocessor has no input noise"
            )

        appended = []
        for name in self.dynamic_features:
            feature = dynamic_features[name]
            # if these diverge, the cat below would fail with a cryptic message. torch._check
            # keeps this a runtime assertion under torch.compile instead of a graph break.
            torch._check(
                (feature.shape[0] == x.shape[0]) and (feature.shape[1] == x.shape[1]),
                lambda: (
                    f"assemble_input: '{name}' has batch and time {tuple(feature.shape[:2])}, "
                    f"the input {tuple(x.shape[:2])}. Was it made at a different batch size?"
                ),
            )
            appended.append(feature)

        if noise is not None:
            torch._check(
                noise.shape[0] == x.shape[0],
                lambda: f"assemble_input: the noise has batch {noise.shape[0]}, the input {x.shape[0]}",
            )
            if self.input_noise_mode == "concatenate":
                appended.append(noise)
            elif self.input_noise_mode == "perturb":
                # fully out-of-place: build a zero noise field and add to all channels
                noise_full = torch.zeros_like(x)
                noise_full[:, :, self.perturb_channels] = noise
                x = x + noise_full

        if not appended:
            return x

        return torch.cat([x] + appended, dim=2)

    def noise_processes(self):
        r"""
        The noise processes of the preprocessor, for the :class:`~makani.models.noise_controller.NoiseController` of the model.

        The preprocessor owns the input noise, so that its learnable parameters
        are trained and checkpointed with the model, but does not drive it: the
        controller advances, snapshots and re-seeds it, and the caller of
        :meth:`assemble_input` passes in its sample.

        Returns
        -------
        dict
            ``{"input": the input noise}``, or empty without input noise.
        """
        if not hasattr(self, "input_noise"):
            return {}

        return {INPUT_NOISE: self.input_noise}

    def compute_history_stats(self, x) -> HistoryStats:
        r"""
        Compute the normalization statistics of a history window.

        Statistics are area-weighted using the grid quadrature, so they are true
        spherical means rather than uniform grid averages, and are broadcast
        across the spatial model-parallel group so every rank normalizes
        identically.

        The behavior depends on ``history_normalization_mode``:

        * ``"none"`` -- statistics are set to zero mean and unit standard
          deviation, making normalization a no-op.
        * ``"timediff"`` -- statistics of the *differences* between consecutive
          time steps, characterizing the tendency rather than the state.
        * anything else (e.g. ``"exponential"``) -- a weighted mean and standard
          deviation over the history window, using
          ``history_normalization_weights``.

        Parameters
        ----------
        x : torch.Tensor
            History window of shape ``(B, n_history + 1, C, H, W)``.

        Returns
        -------
        HistoryStats
            The statistics, to pass to :meth:`history_normalize` and
            :meth:`history_denormalize`.
        """
        self._check_time_axis(x, "compute_history_stats")

        if self.history_normalization_mode == "none":
            return HistoryStats(
                mean=torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32, device=x.device),
                std=torch.ones((1, 1, 1, 1, 1), dtype=torch.float32, device=x.device),
            )

        if self.history_normalization_mode == "timediff":
            # time difference mean:
            diff_mean = torch.mean(self.quadrature(x[:, 1:, ...] - x[:, 0:-1, ...]), dim=(1, 2))

            # time difference std
            diff_var = torch.mean(
                self.quadrature(torch.square((x[:, 1:, ...] - x[:, 0:-1, ...]) - diff_mean)), dim=(1, 2)
            )

            return HistoryStats(
                diff_mean=copy_to_parallel_region(diff_mean, "spatial"),
                diff_var=copy_to_parallel_region(diff_var, "spatial"),
            )

        # quadrature reduces (H, W) -> (B, T, C); the weighted sum over T -> (B, 1, C)
        mean = torch.sum(self.quadrature(x * self.history_normalization_weights), dim=1, keepdim=True)
        b_, _, c_ = mean.shape
        mean = mean.reshape(b_, 1, c_, 1, 1)

        std = torch.sum(
            self.quadrature(torch.square(x - mean) * self.history_normalization_weights),
            dim=1,
            keepdim=True,
        )
        std = torch.sqrt(std.reshape(b_, 1, c_, 1, 1))

        # copy to parallel region
        return HistoryStats(mean=copy_to_parallel_region(mean, "spatial"), std=copy_to_parallel_region(std, "spatial"))

    def _check_history_stats(self, x, caller: str, stats: HistoryStats):
        # the batch is the only dimension of the statistics that is not a broadcast singleton
        if (stats.mean is None) or (stats.std is None):
            raise ValueError(
                f"{caller}: the statistics have no mean and std; compute them with compute_history_stats "
                f"in the same mode ('{self.history_normalization_mode}')"
            )
        stats_batch = stats.mean.shape[0]
        torch._check(
            stats_batch == x.shape[0],
            lambda: (
                f"{caller}: batch mismatch between input ({x.shape[0]}) and history stats ({stats_batch}). "
                f"Were they computed on another window?"
            ),
        )

    def history_normalize(self, x, stats: HistoryStats, target=False):
        r"""
        Normalize a tensor with the history statistics.

        A no-op in the ``"none"`` and ``"timediff"`` modes, which do not define
        a state-space normalization.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, T, C, H, W)``.
        stats : HistoryStats
            The statistics, from :meth:`compute_history_stats`.
        target : bool, optional
            If ``True``, ``x`` holds only the predicted channels, which lead
            the input channels the statistics were computed on, so only those
            statistics are used. By default ``False``.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as ``x``.
        """
        if self.history_normalization_mode in ["none", "timediff"]:
            return x

        self._check_time_axis(x, "history_normalize")
        self._check_history_stats(x, caller="history_normalize", stats=stats)

        mean, std = self._history_stats(x, target, stats)

        return (x - mean) / std

    def history_denormalize(self, xn, stats: HistoryStats, target=False):
        r"""
        Undo :meth:`history_normalize`.

        A no-op in the ``"none"`` and ``"timediff"`` modes.

        Parameters
        ----------
        xn : torch.Tensor
            Normalized tensor of shape ``(B, T, C, H, W)``.
        stats : HistoryStats
            The statistics, from :meth:`compute_history_stats`.
        target : bool, optional
            If ``True``, ``xn`` holds only the predicted channels, see
            :meth:`history_normalize`. By default ``False``.

        Returns
        -------
        torch.Tensor
            Denormalized tensor with the same shape as ``xn``.
        """
        if self.history_normalization_mode in ["none", "timediff"]:
            return xn

        self._check_time_axis(xn, "history_denormalize")
        self._check_history_stats(xn, caller="history_denormalize", stats=stats)

        mean, std = self._history_stats(xn, target, stats)

        return xn * std + mean

    def _history_stats(self, x, target, stats):
        # (B, 1, C, 1, 1), broadcast over time and grid
        mean, std = stats.mean, stats.std
        if target:
            num_channels = x.shape[2]
            return mean[:, :, :num_channels], std[:, :, :num_channels]
        return mean, std

    def get_static_features(self):
        r"""
        Return a copy of the static feature tensor.

        Returns
        -------
        torch.Tensor or None
            Static features of shape ``(1, n_static, H, W)``, or ``None`` if
            none are configured. Cloned so callers cannot mutate the buffer.
        """
        if self.do_add_static_features:
            return self.static_features.clone()
        else:
            return None

    def stack_dynamic_features(self, features):
        r"""
        Stack dynamic features along the channel axis, in the order the model takes them.

        For models that pack their input themselves rather than through
        :meth:`assemble_input`.

        Parameters
        ----------
        features : dict
            Features by name, as a batch or a rollout step carries them.

        Returns
        -------
        torch.Tensor or None
            ``(B, T, sum C_f, H, W)``, or ``None`` if the model takes no dynamic features.
        """
        if not self.dynamic_features:
            return None

        return torch.cat([features[name] for name in self.dynamic_features], dim=2)

    def correct_bias(self, inp: torch.Tensor):
        r"""
        Subtract the configured bias correction from the input.

        A no-op if no bias correction is configured.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor; the correction must broadcast against it.

        Returns
        -------
        torch.Tensor
            The corrected tensor, same shape as ``inp``.
        """
        if hasattr(self, "bias_correction"):
            inp = inp - self.bias_correction
        return inp
