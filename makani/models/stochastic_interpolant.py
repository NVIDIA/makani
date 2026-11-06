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

r"""
Forecasting with stochastic interpolants.

A stochastic interpolant [1] connects the current state :math:`x_0` and the next one :math:`x_1` by the path

.. math::

    I_s = \alpha(s)\, x_0 + \beta(s)\, x_1 + \gamma(s)\, z, \qquad s \in [0, 1],

with :math:`\alpha(s) = 1 - s`, :math:`\beta(s) = s^2`, :math:`\gamma(s) = \sqrt{s}\, \sigma(s)`,
:math:`\sigma(s) = \varepsilon (1 - s)` and :math:`z` standard normal. The network learns the drift
:math:`b(I_s, s)` of the path, conditioned on the window it predicts from, by regression on
:math:`\partial_s I_s`. A forecast integrates the SDE :math:`dX_s = b(X_s, s)\, ds + \sigma(s)\, dW_s` from
:math:`X_0 = x_0` to :math:`s = 1` with Euler-Maruyama.

[1] Chen et al., Probabilistic forecasting with stochastic interpolants and Föllmer processes, 2024.
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from makani.models.noise import IsotropicGaussianRandomFieldS2
from makani.models.noise_controller import NoiseController
from makani.models.stepper import StepWrapper
from makani.utils import comm
from makani.utils.dataloaders.rollout import Rollout

# the configuration of the interpolant and its defaults, see interpolant_config
_DEFAULTS = {
    "epsilon": 1.0,
    "foellmer": False,
    "antithetic": False,
    "n_samples": 1,
    "n_steps": 1,
    "time_aware": False,
    "seed": 333,
}

# the top-level keys the settings were read from before the stochastic_interpolant block
_LEGACY_KEYS = {
    "epsilon": "noise_epsilon",
    "foellmer": "use_foellmer",
    "n_samples": "stochastic_size",
    "n_steps": "stochastic_interpolation_steps",
    "time_aware": "time_aware",
}

# the name of the keys of the path in the noise controller of the model
PATH_NOISE = "path_noise"

# the seeds of the keys of the data-parallel ranks lie this far apart
_DATA_SEED_STRIDE = 7919

# the keys are drawn below this bound, so that mixing them with the sub-step and the rank stays a valid seed
_KEY_BOUND = 2**62

# the tag of the interpolation times among the draws derived from a key, which the sub-steps count up from 0
_TIMES_TAG = -1


def _derived_seed(*values: int) -> int:
    # a seed for one draw derived from a key; Python hashes tuples of ints the same in every process
    return hash(values) & (_KEY_BOUND - 1)


class PathKeys(nn.Module):
    r"""
    The keys of the random draws of the stochastic interpolant, one per sample and step.

    A forecast step integrates the SDE of the interpolant in several sub-steps,
    each with a field of noise of its own, and a training step draws the
    interpolation times and the noise of the path. The interpolant derives all
    of these from one key per sample, which this process draws at each
    :meth:`update`. So the :class:`~makani.models.noise_controller.NoiseController`
    advances, snapshots, restores and re-seeds them like any noise process, at
    the cost of one integer per sample: a restored state redraws the same
    fields and times.

    The keys are the same on all model ranks of a sample: the interpolation
    times have to be, and the noise mixes in the spatial rank, see
    :meth:`StochasticInterpolantWrapper.path_noise`. They are drawn on the CPU,
    so they do not depend on the device either.

    Parameters
    ----------
    batch_size : int
        The initial batch, resized by :meth:`update`.
    seed : int, optional
        The base seed, by default ``333``, laid out over the ranks by
        :meth:`rank_seed`.

    Attributes
    ----------
    state : torch.Tensor
        The keys, ``(B,)``, ``int64``, on the CPU.
    """

    def __init__(self, batch_size: int, seed: int = 333):
        super().__init__()
        # a plain attribute, not a buffer: the keys stay on the CPU when the model moves
        self.state = torch.zeros(batch_size, dtype=torch.int64)
        self.set_rng(self.rank_seed(seed))

    def is_stateful(self) -> bool:
        r"""``False``: each step draws keys independent of the previous ones."""
        return False

    def rank_seed(self, seed: int = 333) -> int:
        r"""
        The seed of this rank for the base seed ``seed``.

        The data-parallel ranks hold different samples, so they draw different
        keys; the model ranks of a sample draw the same, see the class.
        :meth:`~makani.models.noise_controller.NoiseController.set_rng`
        re-seeds the keys with it, as it does the other noise processes, see
        :meth:`~makani.models.noise.BaseNoiseS2.rank_seed`. Serially, it is
        ``seed`` itself.
        """
        return seed + _DATA_SEED_STRIDE * comm.get_rank("data")

    def set_rng(self, seed: int = 333):
        r"""Re-seed the generator of the keys with the seed of this rank, see :meth:`rank_seed`."""
        self.rng = torch.Generator(device=torch.device("cpu"))
        self.rng.manual_seed(seed)

    def reset(self, batch_size: Optional[int] = None):
        r"""Zero the keys, resized to ``batch_size`` if given."""
        self.state = torch.zeros(self.state.shape[0] if batch_size is None else batch_size, dtype=torch.int64)

    def update(self, replace_state: bool = False, batch_size: Optional[int] = None):
        r"""Draw new keys, for ``batch_size`` samples if given; ``replace_state`` makes no difference."""
        batch_size = self.state.shape[0] if batch_size is None else batch_size
        self.state = torch.randint(0, _KEY_BOUND, (batch_size,), dtype=torch.int64, generator=self.rng)

    def get_rng_state(self):
        r"""The state of the generator, and ``None`` for the CUDA one it does not have."""
        return self.rng.get_state(), None

    def set_rng_state(self, cpu_state, gpu_state):
        r"""Restore the generator from :meth:`get_rng_state`."""
        if cpu_state is not None:
            self.rng.set_state(cpu_state)

    def get_tensor_state(self) -> torch.Tensor:
        r"""A copy of the keys."""
        return self.state.clone()

    def set_tensor_state(self, newstate: torch.Tensor):
        r"""
        Replace the keys by a copy of ``newstate``.

        Raises
        ------
        ValueError
            If ``newstate`` is not one key per sample.
        """
        if newstate.dim() != 1:
            raise ValueError(f"The keys are one per sample, (B,), got shape {tuple(newstate.shape)}")
        self.state = newstate.detach().to(device="cpu", dtype=torch.int64).clone()

    def forward(self) -> torch.Tensor:
        r"""The keys, ``(B,)``."""
        return self.state


def interpolant_config(params) -> Dict[str, Any]:
    r"""
    The configuration of the stochastic interpolant.

    Read from the ``stochastic_interpolant`` block of ``params``; a setting it
    does not hold falls back to its earlier top-level key, then to its default.

    Parameters
    ----------
    params : ParamsBase
        The run configuration.

    Returns
    -------
    dict
        ``epsilon``
            Scale of the noise of the path, :math:`\sigma(s) = \varepsilon (1 - s)`.
        ``foellmer``
            Sample with the diffusion of the Föllmer process instead of
            :math:`\sigma`.
        ``antithetic``
            Train on each interpolation time :math:`s` and on :math:`1 - s`.
        ``n_samples``
            Interpolation times per sample in training.
        ``n_steps``
            Euler-Maruyama steps per forecast step.
        ``time_aware``
            Pass :math:`s` to the network as a second argument,
            ``network(x, s)`` with ``s`` of shape ``(B,)``, instead of as an
            input channel.
        ``seed``
            Seed of the interpolation times in training.

    Raises
    ------
    ValueError
        If the block holds an unknown setting.
    """
    block = dict(params.get("stochastic_interpolant", None) or {})
    unknown = set(block) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown stochastic_interpolant settings {sorted(unknown)}, known are {sorted(_DEFAULTS)}")

    config = {}
    for key, default in _DEFAULTS.items():
        if key in block:
            config[key] = block[key]
        elif (key in _LEGACY_KEYS) and (params.get(_LEGACY_KEYS[key], None) is not None):
            config[key] = params.get(_LEGACY_KEYS[key])
        else:
            config[key] = default
    return config


def interpolant_input_channels(params) -> int:
    r"""
    The input channels of the network of a stochastic interpolant.

    The packed window, as for a :class:`~makani.models.stepper.StepWrapper`, the
    state along the path and, unless the network is time aware, the
    interpolation time as a channel of its own.

    Parameters
    ----------
    params : ParamsBase
        The run configuration, with the channel counts the driver derives.

    Returns
    -------
    int
        The number of input channels.
    """
    return params.N_in_channels + params.N_out_channels + (0 if interpolant_config(params)["time_aware"] else 1)


class StochasticInterpolantWrapper(StepWrapper):
    r"""
    A :class:`~makani.models.stepper.StepWrapper` that forecasts with a stochastic interpolant.

    The window is prepared as for any model -- dynamic features, input noise,
    history normalization, static features -- and the prediction of a step is
    post-processed as for any model, so the interpolant runs over rollouts,
    in model packages and in the inferencer like a deterministic model. What
    differs is the step: instead of one network forward, :meth:`predict_step`
    integrates the SDE of the interpolant over ``n_steps`` steps, each
    evaluating the drift network. The path runs in the space the network
    predicts in, after history normalization.

    In training, ``forward(rollout, drift=True)`` returns the drift of the
    network and that of the path at random interpolation times, for the loss to
    compare, for one step: see :meth:`drift_pairs` for why multistep training
    is refused.

    Parameters
    ----------
    params : ParamsBase
        Configuration, see :func:`interpolant_config` for the settings of the
        interpolant. The input and the output channels have to be the same.
    model_handle : callable
        Builds the drift network, with the input channels of
        :func:`interpolant_input_channels`.

    Attributes
    ----------
    config : dict
        The settings of the interpolant, see :func:`interpolant_config`.
    path_keys : PathKeys
        The keys of the noise of the path and of the interpolation times,
        driven by :attr:`noise` under the name ``"path_noise"``: a rollout
        advances them once per step, see :meth:`path_noise` and
        :meth:`interpolation_times`.
    path_field : IsotropicGaussianRandomFieldS2
        Transforms the coefficients drawn from the keys to a field on the grid
        of the model.

    Raises
    ------
    ValueError
        If the input and the output channels differ, or constraints are
        configured.
    """

    def __init__(self, params, model_handle):
        in_channels = params.get("in_channels", None)
        out_channels = params.get("out_channels", None)
        if (in_channels is not None) and (out_channels is not None) and (list(in_channels) != list(out_channels)):
            raise ValueError(
                "The stochastic interpolant starts its path from the current state, so the input and the output "
                f"channels have to be the same, got {list(in_channels)} and {list(out_channels)}"
            )
        if hasattr(params, "constraints"):
            raise ValueError("Constraints are not supported by the stochastic interpolant: it predicts a drift")

        super().__init__(params, model_handle)

        self.config = interpolant_config(params)
        self.epsilon = float(self.config["epsilon"])
        self.foellmer = bool(self.config["foellmer"])
        self.antithetic = bool(self.config["antithetic"])
        self.n_samples = int(self.config["n_samples"])
        self.n_steps = int(self.config["n_steps"])
        self.time_aware = bool(self.config["time_aware"])
        self.input_adapter.time_aware = self.time_aware

        # the keys of the random draws, driven by the controller with the other noise processes of the model
        self.path_keys = PathKeys(params.batch_size, seed=self.config["seed"])
        self.noise = NoiseController(self._noise_processes())

        # transforms the coefficients drawn from the keys to white noise on the grid of the model; its own
        # generators are not used
        self.path_field = IsotropicGaussianRandomFieldS2(
            tuple(self.preprocessor.img_shape_resampled),
            params.batch_size,
            params.N_out_channels,
            num_time_steps=1,
            sigma=1.0,
            alpha=0.0,
            grid_type=params.model_grid_type,
        )
        # the shard of the spectral coefficients this rank draws, as the model is laid out when it is built
        self._spatial_rank = comm.get_rank("spatial")

    def _noise_processes(self):
        # the processes of a StepWrapper, then the keys of the path; called by StepWrapper.__init__ before they exist
        processes = super()._noise_processes()
        if "path_keys" in self._modules:
            processes[PATH_NOISE] = self.path_keys
        return processes

    # ------------------------------------------------------------------ schedule

    def alpha(self, s):
        return 1.0 - s

    def dalpha(self, s):
        return -torch.ones_like(s)

    def beta(self, s):
        return torch.square(s)

    def dbeta(self, s):
        return 2.0 * s

    def sigma(self, s):
        return self.epsilon * (1.0 - s)

    def dsigma(self, s):
        return -self.epsilon * torch.ones_like(s)

    def gamma(self, s):
        return torch.sqrt(s) * self.sigma(s)

    def dgamma(self, s):
        # the path draws z once, gamma(s) z = sigma(s) W_s with W_s = sqrt(s) z: its time derivative in the
        # sense of the paper differentiates sigma only, the increments of W are what the SDE noise supplies
        return torch.sqrt(s) * self.dsigma(s)

    def gsq(self, s):
        r"""
        Squared diffusion coefficient of the sampling SDE.

        :math:`\sigma^2(s)` by default; with ``foellmer``, the diffusion of the
        Föllmer process, which shares the marginals of the path.
        """
        if not self.foellmer:
            return torch.square(self.sigma(s))
        ratio = torch.where(s > 0, s * self.dbeta(s) / self.beta(s), torch.full_like(s, 2.0))
        term1 = 2.0 * torch.square(self.sigma(s)) * ratio
        term2 = 2.0 * s * self.sigma(s) * self.dsigma(s)
        return torch.abs(term1 - term2 - torch.square(self.sigma(s)))

    def score(self, x, x0, b, s):
        r"""
        The score :math:`\nabla_x \log \rho_s(x)` of the path, from its drift ``b``.

        For a Gaussian interpolant, drift and score are related in closed form,
        so one network serves both. Singular at :math:`s = 0` and :math:`s = 1`,
        where the sampler never evaluates it.
        """
        a = 1.0 / (s * self.sigma(s) * (self.dbeta(s) * self.sigma(s) - self.beta(s) * self.dsigma(s)))
        c = x * self.dbeta(s) + (self.beta(s) * self.dalpha(s) - self.dbeta(s) * self.alpha(s)) * x0
        return a * (self.beta(s) * b - c)

    def path(self, x0, x1, z, s):
        r"""
        The point of the path at time ``s`` and its time derivative, the drift to learn.

        Parameters
        ----------
        x0, x1, z : torch.Tensor
            Start, end and noise of the path, of the same shape.
        s : torch.Tensor
            Times, broadcastable against them.

        Returns
        -------
        tuple of torch.Tensor
            :math:`I_s` and :math:`\partial_s I_s`.
        """
        point = self.alpha(s) * x0 + self.beta(s) * x1 + self.gamma(s) * z
        drift = self.dalpha(s) * x0 + self.dbeta(s) * x1 + self.dgamma(s) * z
        return point, drift

    # ------------------------------------------------------------------ network

    def _drift_network(self, model_input, x, s, use_checkpoint=False):
        # the drift at (x, s), conditioned on the input of the step; s of shape (B,)
        return self._network(model_input.replace(path_state=x, time=s), use_checkpoint)

    @torch.compiler.disable
    def path_noise(self, substep: int) -> torch.Tensor:
        r"""
        The noise of one sub-step of the path, derived from the current keys.

        Each sample draws the coefficients of its shard from a generator seeded
        by its key, the sub-step and the spatial rank, so the shards of a sample
        draw different coefficients, and a sample its own noise whatever the
        batch it is in.

        Parameters
        ----------
        substep : int
            The sub-step of the SDE, from 0; the training draws use 0.

        Returns
        -------
        torch.Tensor
            White noise, ``(B, C, H, W)`` on this rank's part of the grid.
        """
        keys = self.noise.sample(PATH_NOISE).tolist()
        field = self.path_field
        device = field.sigma_l.device
        shape = (1,) + tuple(field._state_shape_suffix)
        coefficients = []
        for key in keys:
            generator = torch.Generator(device=device)
            generator.manual_seed(_derived_seed(key, substep, self._spatial_rank))
            coefficients.append(torch.randn(shape, generator=generator, device=device, dtype=torch.float32))
        field.set_tensor_state(torch.cat(coefficients, dim=0))
        return field().squeeze(1)

    @torch.compiler.disable
    def interpolation_times(self, device=None, dtype=torch.float32) -> torch.Tensor:
        r"""
        The interpolation times of a training step, derived from the current keys.

        The same on all model ranks of a sample. With ``antithetic``, the
        times :math:`s` are followed by :math:`1 - s`.

        Returns
        -------
        torch.Tensor
            ``(B, n_samples)``, or ``(B, 2 * n_samples)`` if antithetic.
        """
        times = []
        for key in self.noise.sample(PATH_NOISE).tolist():
            generator = torch.Generator(device=torch.device("cpu"))
            generator.manual_seed(_derived_seed(key, _TIMES_TAG))
            times.append(torch.rand(self.n_samples, generator=generator, dtype=torch.float64))
        s = torch.stack(times).to(device=device, dtype=dtype)
        if self.antithetic:
            s = torch.cat([s, 1.0 - s], dim=1)
        return s

    def _current_state(self, inp, stats):
        # the latest step of the window, in the space the network predicts in: the start of the path
        return self.preprocessor.history_normalize(inp[:, -1:], stats, target=True).squeeze(1)

    # ------------------------------------------------------------------ step

    def _sample(self, model_input, x0, use_checkpoint=False):
        # Euler-Maruyama from s = 0 to s = 1 in n_steps steps; the score correction of the Föllmer diffusion
        # is singular at s = 0, so the first step uses sigma, as in the paper
        batch_size = x0.shape[0]
        times = torch.linspace(0.0, 1.0, self.n_steps + 1, dtype=x0.dtype, device=x0.device)
        x = x0
        for i in range(self.n_steps):
            s = times[i].expand(batch_size)
            ds = times[i + 1] - times[i]
            b = self._drift_network(model_input, x, s, use_checkpoint)
            sr = s.reshape(-1, 1, 1, 1)
            if i == 0:
                diffusion = self.sigma(sr) * torch.sqrt(ds)
            else:
                if self.foellmer:
                    b = b + 0.5 * (self.gsq(sr) - torch.square(self.sigma(sr))) * self.score(x, x0, b, sr)
                diffusion = torch.sqrt(self.gsq(sr) * ds)
            x = x + b * ds + diffusion * self.path_noise(i)
        return x

    def _predict_window(self, inp, dynamic_features, use_checkpoint: bool = False):
        pp = self.preprocessor
        model_input = self._model_input(inp, dynamic_features)
        stats = model_input.stats
        x0 = self._current_state(inp, stats)
        predn = pp.correct_bias(self._sample(model_input, x0, use_checkpoint)).unsqueeze(1)
        return pp.history_denormalize(predn, target=True, stats=stats)

    def compile_step(self, **compile_kwargs):
        r"""
        ``torch.compile`` the drift network, the call the SDE sampler and the drift training share.

        The Euler-Maruyama loop and the draws of the path noise and the
        interpolation times stay in Python, as the rollout around a
        :class:`~makani.models.stepper.StepWrapper` step does. The module itself
        is not wrapped, so its methods, ``steps``, ``noise`` and
        ``drift_pairs``, stay reachable, and the state dict keeps its keys.

        Parameters
        ----------
        **compile_kwargs
            Passed on to ``torch.compile``.
        """
        # see StepWrapper.compile_step for the recompile limit
        torch._dynamo.config.recompile_limit = int(os.getenv("MAKANI_INDUCTOR_COMPILE_LIMIT", "32"))
        self._drift_network = torch.compile(self._drift_network, **compile_kwargs)
        # a checkpointed network with noise of its own runs outside the compiled call, see
        # StepWrapper._checkpoint_with_noise, so it is compiled on its own as well
        if self._network_noise():
            self._compiled["network"] = torch.compile(self.model, **compile_kwargs)

    def check_encodable(self):
        r"""
        The interpolant has no latent state of its own to return.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("The stochastic interpolant does not expose a latent state")

    # ------------------------------------------------------------------ training

    def drift_pairs(self, rollout: Rollout):
        r"""
        The drift of the network and that of the path, for one step.

        The path runs from the latest state of the window to the target. For
        each sample, ``n_samples`` interpolation times (twice that if
        antithetic) are drawn, and one field of the path noise.

        Only single steps are trained: the drift regression of a later step,
        started from the true state again, would be one more single-step pair,
        not a multistep forecast, and the multistep weights of the loss would
        have no lead time to weigh. Training from the model's own forecasts, the
        push-forward analogue, is not implemented.

        Parameters
        ----------
        rollout : Rollout
            A rollout of one step, not started yet, of a batch with targets
            and without an ensemble.

        Returns
        -------
        tuple of torch.Tensor
            The drift of the network and the drift of the path, both
            ``(B * S, C, H, W)``, batch-major.

        Raises
        ------
        ValueError
            If the rollout has more than one step, an ensemble or no targets,
            or does not drive the noise of the model.
        """
        if rollout.n_steps != 1:
            raise ValueError(
                f"The drift is trained on single steps, got a rollout of {rollout.n_steps}: train with n_future = 0"
            )
        if rollout.ensemble_size != 1:
            raise ValueError("The drift is trained without an ensemble")
        if (rollout.noise is not None) and (rollout.noise is not self.noise):
            raise ValueError(
                "The rollout drives the noise of another model: make it with Rollout(..., noise=model.noise)"
            )
        if self.noise.is_stochastic and (rollout.noise is None):
            raise ValueError(
                "The model is stochastic, but the rollout has no noise: make it with Rollout(..., noise=model.noise)"
            )

        step = next(rollout)
        if step.target is None:
            raise ValueError("The drift is trained on a rollout with targets")
        self._check_step_noise(step)

        pp = self.preprocessor
        model_input = self._model_input(step.input, step.input_dynamic_features)
        stats = model_input.stats
        with torch.no_grad():
            x0 = self._current_state(step.input, stats)
            x1 = pp.history_normalize(step.target, stats, target=True).squeeze(1)
            s = self.interpolation_times(x0.device, x0.dtype)
            z = self.path_noise(0)
            # (B, S, C, H, W)
            point, drift = self.path(x0.unsqueeze(1), x1.unsqueeze(1), z.unsqueeze(1), s.reshape(*s.shape, 1, 1, 1))

        n = s.shape[1]
        predicted = self._drift_network(model_input.repeat_interleave(n), point.flatten(0, 1), s.flatten(0, 1))
        # the rollout is finished with its only step
        rollout.advance(step.target)

        return predicted, drift.flatten(0, 1)

    def forward(self, rollout: Rollout, drift: bool = False):
        r"""
        Forecast over a rollout, or return the drift pairs to train on.

        Parameters
        ----------
        rollout : Rollout
            The rollout, not started yet.
        drift : bool, optional
            Return :meth:`drift_pairs` instead of the forecast. By default
            ``False``, the forecast of
            :meth:`~makani.models.stepper.StepWrapper.forward`.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            The forecast, ``(B * E, n_steps, C, H, W)``, or the drift pairs.
        """
        if drift:
            return self.drift_pairs(rollout)
        return super().forward(rollout)
