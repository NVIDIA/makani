# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""
The input of a network in one step, and the adapter that packs it for flat networks.

A :class:`~makani.models.stepper.StepWrapper` hands its network a
:class:`ModelInput`: the history-normalized state window, and the dynamic and
static features, each in a field of its own. A network that takes it declares
``takes_model_input = True``; any other network is a flat network, which takes
one channel stack, packed by the :class:`FlatInputAdapter` of the wrapper.

The noise of a model is the model's own: the model owns its noise processes
and draws from them where it uses them, e.g. with an :class:`InputNoise` layer
for noise on the input, or in its latent space. It lists them through
``noise_processes()``, so that the
:class:`~makani.models.noise_controller.NoiseController` of the wrapper can
advance, snapshot, restore and re-seed them, consistently with a rollout.
"""

import dataclasses
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from makani.models.noise import build_noise, noise_base_seed, run_eager
from makani.models.preprocessor import HistoryNormalization, HistoryStats, model_dynamic_features
from makani.utils.functions import flatten_history

# the name of the input noise among the noise processes of a model
INPUT_NOISE = "input_noise"


def takes_model_input(network: nn.Module) -> bool:
    r"""Whether a network takes a :class:`ModelInput`, rather than the flat channel stack of a :class:`FlatInputAdapter`."""
    return bool(getattr(network, "takes_model_input", False))


@dataclasses.dataclass
class ModelInput:
    r"""
    The input of a network in one step.

    Attributes
    ----------
    state : torch.Tensor
        The state window, ``(B, n_history + 1, C, H, W)``, oldest step first,
        normalized with its history statistics :attr:`stats`.
    stats : HistoryStats
        The history statistics of the state, which the prediction is
        denormalized with.
    dynamic_features : dict
        The dynamic features of the window by name, each
        ``(B, n_history + 1, C_f, H, W)``, as the data holds them.
    static : torch.Tensor, optional
        The static features, ``(1, n_static, H, W)``, the same for every
        sample; ``None`` without static features.
    path_state : torch.Tensor, optional
        The state on the path of a stochastic interpolant, ``(B, C_out, H, W)``.
    time : torch.Tensor, optional
        The interpolation time of :attr:`path_state`, ``(B,)``.
    repeats : int
        How many times each sample of the noise is repeated in the batch, one
        after the other, see :meth:`repeat_interleave`: the noise processes
        are sized for ``B / repeats`` samples. By default 1.
    """

    state: torch.Tensor
    stats: HistoryStats
    dynamic_features: Dict[str, torch.Tensor]
    static: Optional[torch.Tensor] = None
    path_state: Optional[torch.Tensor] = None
    time: Optional[torch.Tensor] = None
    repeats: int = 1

    def replace(self, **changes) -> "ModelInput":
        r"""A copy with the given fields replaced."""
        return dataclasses.replace(self, **changes)

    def repeat_interleave(self, repeats: int) -> "ModelInput":
        r"""
        Repeat each sample ``repeats`` times along the batch, as :func:`torch.repeat_interleave` does.

        Fields the same for every sample, the static features and statistics
        without a batch, are kept. The noise of a sample is repeated with it:
        a model draws its noise for the samples, and repeats each draw
        :attr:`repeats` times, see :meth:`repeat_noise`.
        """
        batch_size = self.state.shape[0]

        def repeat(x):
            if (x is None) or (x.shape[0] != batch_size):
                return x
            return x.repeat_interleave(repeats, dim=0)

        return ModelInput(
            state=repeat(self.state),
            stats=HistoryStats(*(repeat(s) for s in self.stats)),
            dynamic_features={name: repeat(f) for name, f in self.dynamic_features.items()},
            static=self.static,
            path_state=repeat(self.path_state),
            time=repeat(self.time),
            repeats=self.repeats * repeats,
        )

    def repeat_noise(self, noise: torch.Tensor) -> torch.Tensor:
        r"""
        A draw of noise for the samples, repeated as the samples are in this input.

        Parameters
        ----------
        noise : torch.Tensor
            The draw, ``(B / repeats, ...)``.

        Returns
        -------
        torch.Tensor
            ``(B, ...)``.
        """
        if self.repeats == 1:
            return noise
        return noise.repeat_interleave(self.repeats, dim=0)


class InputNoise(nn.Module):
    r"""
    Noise on the input of a network, as a layer.

    Owns its noise process, so that its learnable parameters are trained and
    checkpointed with the model, and draws from it in its forward. It does not
    drive it: the :class:`~makani.models.noise_controller.NoiseController` of
    the model advances, snapshots and re-seeds the process, which the owner of
    the layer lists through ``noise_processes()``. A draw is a deterministic
    function of the state of the process and of its parameters, so it stays in
    their autograd graph. The noise comes in one of two modes:

    * ``concatenate``: the draw is returned, for the caller to take in as
      channels of its own.
    * ``perturb``: the draw is added to selected channels of the state. It is a
      perturbation in data space: it is scaled by the history standard
      deviation of the state, so that it perturbs the normalized state as it
      would the state before its normalization. The history statistics, and
      so the denormalization of the prediction, do not depend on it.

    Parameters
    ----------
    noise_params : dict
        The configuration of the noise, ``params.input_noise``: ``mode``,
        ``n_channels`` (concatenate), ``perturb_channels`` (perturb), and the
        settings of the process, see :func:`~makani.models.noise.build_noise`.
    channel_names : list of str
        The channels of the state, to look the perturbed channels up in.
    img_shape : tuple of int
        The grid of the model.
    batch_size : int
        The batch the process is sized for initially.
    n_history : int
        The window holds ``n_history + 1`` time steps.
    grid_type : str
        The grid type of the model.
    history_normalization_mode : str
        The history normalization of the state; a perturbation is scaled by
        the history standard deviation unless it is ``"none"`` or
        ``"timediff"``.
    dt_hours : float, optional
        The time step of the model in hours. Read only by diffusion noise,
        whose temporal correlation defaults to 6 hours.
    """

    def __init__(
        self,
        noise_params,
        channel_names,
        img_shape,
        batch_size: int,
        n_history: int,
        grid_type: str,
        history_normalization_mode: str,
        dt_hours: Optional[float] = None,
    ):
        super().__init__()

        self.mode = noise_params.get("mode", "concatenate")
        if self.mode == "concatenate":
            num_channels = noise_params.get("n_channels", 1)
            self.perturb_channels = None
        elif self.mode == "perturb":
            self.perturb_channels = [
                channel_names.index(ch) for ch in noise_params.get("perturb_channels", channel_names)
            ]
            num_channels = len(self.perturb_channels)
        else:
            raise NotImplementedError(f"Error, input noise mode {self.mode} not supported.")
        self.scale_by_std = history_normalization_mode not in ["none", "timediff"]

        # the seed of this rank; the temporal correlation of the diffusion noise defaults to tau = 6h, read from
        # the time step only for it, since the other types have no temporal correlation
        centered = noise_params.get("centered", False)
        seed, reflect = noise_base_seed(centered=centered)
        default_lambd = dt_hours / 6.0 if noise_params.get("type") == "diffusion" else 1.0
        self.process = build_noise(
            noise_params,
            img_shape=list(img_shape),
            batch_size=batch_size,
            num_channels=num_channels,
            num_time_steps=n_history + 1,
            grid_type=grid_type,
            seed=seed,
            reflect=reflect,
            centered=centered,
            default_lambd=default_lambd,
        )

    @classmethod
    def from_params(cls, params) -> Optional["InputNoise"]:
        r"""The input noise configured in ``params.input_noise``, ``None`` without one."""
        if params.get("input_noise", None) is None:
            return None
        return cls(
            params.input_noise,
            channel_names=params.channel_names,
            img_shape=(params.img_shape_x_resampled, params.img_shape_y_resampled),
            batch_size=params.batch_size,
            n_history=params.n_history,
            grid_type=params.model_grid_type,
            history_normalization_mode=params.history_normalization_mode,
            # read only for diffusion noise, the other types have no temporal correlation
            dt_hours=(params.dt * params.dhours) if params.input_noise.get("type") == "diffusion" else None,
        )

    def draw(self, model_input: ModelInput) -> torch.Tensor:
        r"""
        The draw of the noise for an input, ``(B, n_history + 1, C_noise, H, W)``.

        Raises
        ------
        RuntimeError
            If the process is not sized for the samples of the input.
        """
        # computed eagerly: the noise is complex-valued, see run_eager
        noise = model_input.repeat_noise(run_eager(self.process))
        torch._check(
            noise.shape[0] == model_input.state.shape[0],
            lambda: (
                f"InputNoise: the noise has batch {noise.shape[0]}, the state {model_input.state.shape[0]}: advance "
                f"the noise of the model at the batch it predicts, see NoiseController.advance"
            ),
        )
        return noise

    def forward(self, model_input: ModelInput) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Apply the noise to an input.

        Parameters
        ----------
        model_input : ModelInput
            The input of the step.

        Returns
        -------
        tuple
            The state, perturbed in ``perturb`` mode, and the draw to take in
            as channels in ``concatenate`` mode, else ``None``.
        """
        noise = self.draw(model_input)
        state = model_input.state
        if self.mode == "concatenate":
            return state, noise

        # fully out-of-place
        perturbation = torch.zeros_like(state)
        perturbation[:, :, self.perturb_channels] = noise
        if self.scale_by_std:
            perturbation = perturbation / model_input.stats.std
        return state + perturbation, None


class FlatInputAdapter(nn.Module):
    r"""
    The input of a flat network: its input noise, and the packing of its channel stack.

    A flat network takes one channel stack, which :meth:`pack` assembles in a
    fixed order: per time step, oldest first,
    ``[state | dynamic features, in the order of dynamic_features | noise]``,
    then the static features, then, for a stochastic interpolant, the state on
    its path and, unless the network is time-aware, a channel of the
    interpolation time. The dynamic features and the concatenated noise are
    normalized with history statistics of their own, as the state is.

    The adapter is the front of a flat network, so it owns the input noise of
    one, an :class:`InputNoise`, configured by ``params.input_noise``, and
    lists it through :meth:`noise_processes`. A network that takes a
    :class:`ModelInput` owns the noise it takes itself; the adapter then has
    none.

    Parameters
    ----------
    params : ParamsBase
        Configuration object. Reads ``input_noise``, ``add_zenith``, and,
        for the noise and the normalization, ``channel_names``,
        ``batch_size``, ``n_history``, the resampled image shape and the
        history normalization settings.
    with_input_noise : bool, optional
        Whether the adapter builds the input noise configured in ``params``.
        By default ``True``.
    time_aware : bool, optional
        Whether a stochastic interpolant passes the interpolation time to the
        network as an argument of its own, ``network(x, s)``, instead of as a
        channel. By default ``False``.
    """

    def __init__(self, params, with_input_noise: bool = True, time_aware: bool = False):
        super().__init__()

        self.time_aware = time_aware

        # the dynamic features, in the order their channels are appended
        self.dynamic_features = model_dynamic_features(params)

        # the dynamic features and the concatenated noise are normalized with statistics of their own
        self.history_normalization = HistoryNormalization(params)

        self.input_noise = InputNoise.from_params(params) if with_input_noise else None

    def noise_processes(self):
        r"""
        The noise processes of the adapter, for the :class:`~makani.models.noise_controller.NoiseController` of the model.

        Returns
        -------
        dict
            ``{"input_noise": the process of the input noise}``, or empty without input noise.
        """
        if self.input_noise is None:
            return {}

        return {INPUT_NOISE: self.input_noise.process}

    def pack(self, model_input: ModelInput) -> Tuple[torch.Tensor, ...]:
        r"""
        Pack the input for a flat network, drawing its input noise.

        Parameters
        ----------
        model_input : ModelInput
            The input of the step.

        Returns
        -------
        tuple of torch.Tensor
            The arguments of the network: the channel stack
            ``(B, T * C' + n_static [+ C_out [+ 1]], H, W)``, followed by the
            interpolation time ``(B,)`` for a time-aware network.

        Raises
        ------
        ValueError
            If the dynamic features are not the ones the model takes.
        """
        state = model_input.state
        torch._check(
            state.dim() == 5,
            lambda: f"pack: expected a state with an explicit time axis, (B, T, C, H, W), got shape {tuple(state.shape)}",
        )
        dynamic_features = model_input.dynamic_features
        if set(dynamic_features) != set(self.dynamic_features):
            raise ValueError(
                f"pack: the model takes the dynamic features {sorted(self.dynamic_features)}, "
                f"got {sorted(dynamic_features)}"
            )

        appended = []
        for name in self.dynamic_features:
            feature = dynamic_features[name]
            # if these diverge, the cat below would fail with a cryptic message. torch._check
            # keeps this a runtime assertion under torch.compile instead of a graph break.
            torch._check(
                (feature.shape[0] == state.shape[0]) and (feature.shape[1] == state.shape[1]),
                lambda: (
                    f"pack: '{name}' has batch and time {tuple(feature.shape[:2])}, "
                    f"the state {tuple(state.shape[:2])}. Was it made at a different batch size?"
                ),
            )
            appended.append(feature)

        if self.input_noise is not None:
            state, noise = self.input_noise(model_input)
            if noise is not None:
                appended.append(noise)

        x = state
        if appended:
            extra = torch.cat(appended, dim=2)
            norm = self.history_normalization
            if norm.history_normalization_mode not in ["none", "timediff"]:
                extra = norm.normalize(extra, norm.compute_stats(extra))
            x = torch.cat([x, extra], dim=2)

        # the one place where the time axis is folded into the channels
        x = flatten_history(x)

        if model_input.static is not None:
            # the same fields for every sample
            static = torch.tile(model_input.static, dims=(x.shape[0], 1, 1, 1))
            x = torch.cat([x, static], dim=1)

        if model_input.path_state is None:
            return (x,)

        # the state on the path of a stochastic interpolant, and its interpolation time
        x = torch.cat([x, model_input.path_state], dim=1)
        s = model_input.time
        if self.time_aware:
            return (x, s)
        s_channel = s.reshape(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1]).to(x.dtype)
        return (torch.cat([x, s_channel], dim=1),)
