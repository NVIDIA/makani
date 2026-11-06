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
from makani.utils.dataloaders.batch import ZENITH


def model_dynamic_features(params):
    r"""The dynamic features a model takes, by name, in the order their channels are appended."""
    return [ZENITH] if params.get("add_zenith", False) else []


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


class HistoryNormalization(nn.Module):
    r"""
    Normalization of a history window with its own statistics.

    Per channel, the statistics are area-weighted means over the grid, using
    the grid quadrature, combined over the time steps of the window, and
    broadcast across the spatial model-parallel group so that every rank
    normalizes identically. As the statistics of a channel depend on that
    channel only, a window can be normalized in parts.

    Parameters
    ----------
    params : ParamsBase
        Configuration object. Reads the resampled image shape, ``n_history``,
        ``history_normalization_mode`` (and ``history_normalization_decay`` in
        the ``"exponential"`` mode) and ``model_grid_type``.
    """

    def __init__(self, params):
        super().__init__()

        img_shape_resampled = [params.img_shape_x_resampled, params.img_shape_y_resampled]
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
                img_shape=img_shape_resampled,
                crop_shape=None,
                crop_offset=(0, 0),
                normalize=True,
                distributed=True,
            )

        self.history_eps = 1e-6

    def _check_time_axis(self, x, caller):
        torch._check(
            x.dim() == 5,
            lambda: f"{caller}: expected a tensor with an explicit time axis, (B, T, C, H, W), got shape {tuple(x.shape)}",
        )

    def compute_stats(self, x) -> HistoryStats:
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
            The statistics, to pass to :meth:`normalize` and
            :meth:`denormalize`.
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
                f"{caller}: the statistics have no mean and std; compute them with compute_stats "
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

    def normalize(self, x, stats: HistoryStats, target=False):
        r"""
        Normalize a tensor with the history statistics.

        A no-op in the ``"none"`` and ``"timediff"`` modes, which do not define
        a state-space normalization.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, T, C, H, W)``.
        stats : HistoryStats
            The statistics, from :meth:`compute_stats`.
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

    def denormalize(self, xn, stats: HistoryStats, target=False):
        r"""
        Undo :meth:`normalize`.

        A no-op in the ``"none"`` and ``"timediff"`` modes.

        Parameters
        ----------
        xn : torch.Tensor
            Normalized tensor of shape ``(B, T, C, H, W)``.
        stats : HistoryStats
            The statistics, from :meth:`compute_stats`.
        target : bool, optional
            If ``True``, ``xn`` holds only the predicted channels, see
            :meth:`normalize`. By default ``False``.

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


class Preprocessor2D(nn.Module):
    r"""
    Input/output plumbing that surrounds the network in one step of a rollout.

    Sits between the data and the network and owns what the model needs around
    it: the history normalization of the state, the dynamic and static
    features, and the bias correction. The network input itself is assembled
    from them by the :class:`~makani.models.model_input.FlatInputAdapter`, or
    by a network that takes a :class:`~makani.models.model_input.ModelInput`.

    Everything here works on tensors with an explicit time axis,
    ``(B, T, C, H, W)``.

    The preprocessor keeps no state between steps: the input window and its
    dynamic features come from a
    :class:`~makani.utils.dataloaders.rollout.Rollout`, which also slides the
    window forward, and the history statistics of a step are returned to the
    caller rather than stored.

    The main responsibilities are:

    * **History normalization.** Normalization of the state window with its
      own statistics (:meth:`compute_history_stats`), undone on the prediction.
    * **Dynamic features.** Quantities the model consumes but does not
      predict, such as the solar zenith angle. Which ones the model takes is
      fixed by the configuration (:attr:`dynamic_features`); their values come
      with the step.
    * **Static features.** Time-invariant fields such as orography and the
      land-sea mask, :meth:`get_static_features`.
    * **Bias correction.** An optional additive correction applied to the
      model output.

    Parameters
    ----------
    params : ParamsBase
        Configuration object. Relevant keys include the image shape and
        resampled shape, ``n_history``, ``history_normalization_mode``, and the
        static-feature and bias-correction settings.
    """

    def __init__(self, params):
        super().__init__()

        self.img_shape = [params.img_shape_x, params.img_shape_y]
        self.img_shape_resampled = [params.img_shape_x_resampled, params.img_shape_y_resampled]

        self.subsampling_factor = params.get("subsampling_factor", 1)
        self.n_history = params.n_history
        self.history_normalization_mode = params.history_normalization_mode
        self.history_normalization = HistoryNormalization(params)

        # the dynamic features the model takes, in the order their channels are appended
        self.dynamic_features = model_dynamic_features(params)

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

    def check_dynamic_features(self, dynamic_features: Dict[str, torch.Tensor], caller: str):
        r"""
        Check that the dynamic features of a window are the ones the model takes.

        Parameters
        ----------
        dynamic_features : dict
            The features of the window by name.
        caller : str
            Named in the error.

        Raises
        ------
        ValueError
            If the features are not the ones the model takes, :attr:`dynamic_features`.
        """
        if set(dynamic_features) != set(self.dynamic_features):
            raise ValueError(
                f"{caller}: the model takes the dynamic features {sorted(self.dynamic_features)}, "
                f"got {sorted(dynamic_features)}"
            )

    def compute_history_stats(self, x) -> HistoryStats:
        r"""
        The normalization statistics of a history window, see :meth:`HistoryNormalization.compute_stats`.

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
        return self.history_normalization.compute_stats(x)

    def history_normalize(self, x, stats: HistoryStats, target=False):
        r"""
        Normalize a tensor with the history statistics, see :meth:`HistoryNormalization.normalize`.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(B, T, C, H, W)``.
        stats : HistoryStats
            The statistics, from :meth:`compute_history_stats`.
        target : bool, optional
            If ``True``, ``x`` holds only the predicted channels, which lead
            the channels the statistics were computed on, so only those
            statistics are used. By default ``False``.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as ``x``.
        """
        return self.history_normalization.normalize(x, stats, target=target)

    def history_denormalize(self, xn, stats: HistoryStats, target=False):
        r"""
        Undo :meth:`history_normalize`.

        Parameters
        ----------
        xn : torch.Tensor
            Normalized tensor of shape ``(B, T, C, H, W)``.
        stats : HistoryStats
            The statistics, from :meth:`compute_history_stats`.
        target : bool, optional
            As in :meth:`history_normalize`. By default ``False``.

        Returns
        -------
        torch.Tensor
            Denormalized tensor with the same shape as ``xn``.
        """
        return self.history_normalization.denormalize(xn, stats, target=target)

    def get_static_features(self):
        r"""
        Return the static feature tensor.

        Returns
        -------
        torch.Tensor or None
            Static features of shape ``(1, n_static, H, W)``, or ``None`` if
            none are configured.
        """
        if self.do_add_static_features:
            return self.static_features
        else:
            return None

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
