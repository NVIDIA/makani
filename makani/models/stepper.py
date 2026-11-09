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

import contextlib
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from makani.models.model_input import (
    FlatInputAdapter,
    ModelInput,
    takes_model_input,
)
from makani.models.preprocessor import Preprocessor2D
from makani.models.noise_controller import NoiseController
from makani.utils.dataloaders.rollout import Rollout, RolloutStep


def _assert_checkpoint_safe(module: nn.Module, noise: NoiseController):
    """Guard for rollout activation checkpointing.

    Checkpointing recomputes each step's model forward during backward. Global-RNG
    stochastic ops (e.g. DropPath) are covered by checkpoint's preserve_rng_state, but
    modules that advance their *own* torch.Generator (e.g. SeededDropout2d) are not:
    their recomputed mask would diverge from the original forward and corrupt gradients.
    Fail loudly rather than train silently-wrong. The noise processes of the model are
    exempt: their generators only advance them, outside the forward, and the recompute
    draws from the state of its step, see :meth:`StepWrapper._network`.
    """
    processes = {id(noise.process(name)) for name in noise}
    offenders = sorted(
        {
            type(m).__name__
            for m in module.modules()
            if (id(m) not in processes)
            and (
                isinstance(getattr(m, "rng_cpu", None), torch.Generator)
                or isinstance(getattr(m, "rng_gpu", None), torch.Generator)
            )
        }
    )
    if offenders:
        raise RuntimeError(
            f"multistep_checkpoint is incompatible with modules carrying private RNG "
            f"generators (found: {offenders}). Their dropout masks are not restored on "
            f"the checkpoint recompute, which would corrupt gradients. Disable "
            f"multistep_checkpoint or make these modules use the global RNG."
        )


class StepWrapper(nn.Module):
    r"""
    Bind a network to its preprocessor and its input.

    Wraps the network so callers can hand it data as the dataloader emits it
    and get predictions back in the same space, without having to replicate
    the preprocessing pipeline at every call site. That is the data space: the
    fields normalized by the dataset statistics, which the dataloader already
    applies and this wrapper never undoes. Each step performs: the history
    normalization of the window, the assembly of the network input, a
    :class:`~makani.models.model_input.ModelInput`, the model forward, bias
    correction, and the history denormalization of the output. A network that
    declares ``takes_model_input = True`` gets the
    :class:`~makani.models.model_input.ModelInput` itself; any other network
    gets the channel stack the
    :class:`~makani.models.model_input.FlatInputAdapter` packs from it.

    ``forward`` takes a :class:`~makani.utils.dataloaders.rollout.Rollout` and
    runs the model over all of its steps; :meth:`steps` hands out the
    predictions one at a time instead, so that an evaluation can release each
    one before the next, rather than keep all lead times. The rollout carries the input
    window and the dynamic features of each step, so the preprocessor keeps no
    state between steps, and the history statistics of a step are passed from
    its normalization to its denormalization. Outside a rollout, :meth:`predict`
    and :meth:`encode` take one window and its dynamic features directly, e.g.
    for the model package.

    The stochastic state of the model is driven through :attr:`noise`, a
    :class:`~makani.models.noise_controller.NoiseController` over the noise
    processes of the model: the input noise of the input adapter, and those the
    network lists through an optional ``noise_processes()`` method, e.g. for
    noise in its latent space. A rollout advances it before each step, see
    :class:`~makani.utils.dataloaders.rollout.Rollout`; outside a rollout, the
    caller advances it before :meth:`predict`. How the noise is used is up to
    the model, which draws from its processes where it uses them, at their
    current state, so that the draws stay in the autograd graph of learnable
    noise; the wrapper only steers the state.

    Input and output keep the time axis explicit; only the network itself sees
    the time steps folded into its channels.

    Two options control cost and stability of training through a rollout of
    several steps. Both apply only in training mode, so evaluation and
    inference are unaffected by them:

    * **Push-forward mode.** Detaches the input at each step, so gradients do
      not flow through the whole rollout. This trains the model to be robust to
      its own errors without the cost -- or the instability -- of
      backpropagating through many steps.
    * **Rollout checkpointing.** Recomputes each step's forward during the
      backward pass instead of retaining its activations, turning the
      ``O(n_steps)`` activation multiplier of backprop-through-time into
      ``O(1)`` at the cost of one extra forward per step. Only used for
      rollouts of more than one step.

    Parameters
    ----------
    params : ParamsBase
        Configuration. Reads ``multistep.push_forward`` and
        ``multistep_checkpoint``; the rest is forwarded to
        :class:`~makani.models.preprocessor.Preprocessor2D`.
    model_handle : callable
        Zero-argument factory returning the network to wrap.

    Attributes
    ----------
    preprocessor : Preprocessor2D
        Normalizes the window, and holds the dynamic and static features and
        the bias correction.
    input_adapter : FlatInputAdapter
        Owns the input noise, and packs the input of a flat network.
    model : torch.nn.Module
        The network.
    noise : NoiseController
        Drives the noise processes of the model, the input noise and those the
        network lists through an optional ``noise_processes()``; a rollout of a
        stochastic model is made with ``Rollout(..., noise=model.noise)``. See
        :mod:`~makani.models.noise_controller`.
    """

    def __init__(self, params, model_handle):
        super().__init__()
        self.preprocessor = Preprocessor2D(params)
        # before the network: the input noise draws its parameters first, and they lead the parameters of the model
        self.input_adapter = FlatInputAdapter(params)
        self.model = model_handle()
        multistep_parameters = params.get("multistep", {"push_forward": False})
        self.push_forward_mode = multistep_parameters["push_forward"]

        # rollout-level activation checkpointing, set via the --multistep_checkpoint CLI flag (top-level
        # param, like n_future). Off by default. Whether the model is safe to checkpoint is checked on
        # first use, so that loading a model for evaluation never fails on it.
        self.multistep_checkpoint = params.get("multistep_checkpoint", False)
        self._checkpoint_checked = False

        # compiled functions of the step, kept out of the module tree so that the state dict keeps its keys
        self._compiled = {}

        # a network that takes a ModelInput owns the noise it takes, e.g. an InputNoise of its own
        if takes_model_input(self.model) and (self.input_adapter.input_noise is not None):
            self.input_adapter.input_noise = None
            if not hasattr(self.model, "noise_processes") or not self.model.noise_processes():
                raise ValueError(
                    f"{type(self.model).__name__} takes a ModelInput, so it owns the noise it takes, but lists no noise "
                    f"process although input_noise is configured: build an InputNoise in the network, see "
                    f"InputNoise.from_params, and list its process through noise_processes()"
                )

        # references to the noise processes, which stay submodules of the input adapter and the network
        self.noise = NoiseController(self._noise_processes())

    @staticmethod
    def upgrade_state_dict(state_dict, prefix=""):
        r"""
        Upgrade a state dict of an earlier version in place: the input noise was kept by the preprocessor.

        ``preprocessor.input_noise.*`` becomes ``input_adapter.input_noise.process.*``; see
        :func:`~makani.utils.checkpoint_helpers.upgrade_checkpoint`.
        """
        old, new = prefix + "preprocessor.input_noise.", prefix + "input_adapter.input_noise.process."
        for key in [k for k in state_dict if k.startswith(old)]:
            state_dict[new + key[len(old) :]] = state_dict.pop(key)

    def _noise_processes(self):
        # the input noise first, so that it keeps its seed under NoiseController.set_rng
        processes = dict(self.input_adapter.noise_processes())
        if hasattr(self.model, "noise_processes"):
            for name, process in self.model.noise_processes().items():
                processes[f"model.{name}"] = process
        return processes

    def _model_input(self, inp, dynamic_features) -> ModelInput:
        # the input of the network for one window, with the history statistics its prediction is denormalized with
        pp = self.preprocessor
        pp.check_dynamic_features(dynamic_features, "StepWrapper")

        stats = pp.compute_history_stats(inp)
        return ModelInput(
            state=pp.history_normalize(inp, stats),
            stats=stats,
            dynamic_features=dynamic_features,
            static=pp.get_static_features(),
        )

    def _network(self, model_input: ModelInput, use_checkpoint: bool = False, method=None):
        # the network on its input, packed for a flat network; the packing, and the draw of the input noise in it,
        # stay outside the checkpoint below
        args = (model_input,) if takes_model_input(self.model) else self.input_adapter.pack(model_input)
        fn = self.model if method is None else method
        if use_checkpoint:
            # Only the network forward is recomputed in the backward pass; the input preparation stays
            # outside the checkpoint, so it runs once. Global-RNG ops inside the network are handled by
            # preserve_rng_state=True; private-generator modules are rejected up front by
            # _assert_checkpoint_safe.
            if (method is None) and self._network_noise():
                return self._checkpoint_with_noise(*args)
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True)
        return fn(*args)

    def _network_noise(self):
        # the noise processes the network draws from itself
        return [name for name in self.noise if name.startswith("model.")]

    @torch.compiler.disable
    def _checkpoint_with_noise(self, *args):
        # The noise the network draws itself is drawn again in the recompute, from the state of this step, which
        # the rollout has advanced by then: the recompute context of the checkpoint draws from that state itself,
        # graph included. Dynamo cannot trace a checkpoint with a context of its own, so this runs eagerly, around
        # the network compiled on its own, see compile_step.
        states = self.noise.live_states(self._network_noise())
        network = self._compiled.get("network", self.model)
        return checkpoint(
            network,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
            context_fn=lambda: (contextlib.nullcontext(), self.noise.drawing_from(states)),
        )

    def _predict_window(self, inp, dynamic_features, use_checkpoint: bool = False):
        pp = self.preprocessor

        model_input = self._model_input(inp, dynamic_features)
        predn = self._network(model_input, use_checkpoint)

        # the network predicts one step as a flat channel stack; give it its time axis back
        predn = pp.correct_bias(predn).unsqueeze(1)

        return pp.history_denormalize(predn, target=True, stats=model_input.stats)

    def compile_step(self, **compile_kwargs):
        r"""
        ``torch.compile`` the prediction of one rollout step.

        A rollout runs its loop, and draws its noise, in Python; what is worth
        compiling is the step: preprocessing, network and postprocessing. The
        compiled step does not depend on the number of steps, and the state
        dict keeps its keys.

        Parameters
        ----------
        **compile_kwargs
            Passed on to ``torch.compile``.
        """
        # layers whose forward breaks the graph, e.g. around distributed collectives, are compiled
        # as frames of their own, once per parameter shape. Models with many such layers of
        # different sizes exceed torch's default limit of 8 and fall back to eager for the rest
        torch._dynamo.config.recompile_limit = int(os.getenv("MAKANI_INDUCTOR_COMPILE_LIMIT", "32"))
        self._predict_window = torch.compile(self._predict_window, **compile_kwargs)
        # a checkpointed network with noise of its own runs outside the compiled step, see _checkpoint_with_noise,
        # so it is compiled on its own as well
        if self._network_noise():
            self._compiled["network"] = torch.compile(self.model, **compile_kwargs)

    def predict(self, inp, dynamic_features):
        r"""
        Predict one step from a window, without a rollout.

        A stochastic model predicts from the current state of its noise, see
        :attr:`noise`: advance it before each prediction, replacing it before
        the first one.

        Parameters
        ----------
        inp : torch.Tensor
            The window, ``(B, n_history + 1, C, H, W)``, in data space.
        dynamic_features : dict
            The dynamic features over the window, name to ``(B, n_history + 1, C_f, H, W)``;
            empty if the model takes none.

        Returns
        -------
        torch.Tensor
            The prediction, ``(B, 1, C_out, H, W)``, in data space.
        """
        return self._predict_window(inp, dynamic_features)

    def check_encodable(self):
        r"""
        Check that the network can return its latent state, see :meth:`encode`.

        Raises
        ------
        NotImplementedError
            If the network does not expose ``encode_process``.
        """
        if not hasattr(self.model, "encode_process"):
            raise NotImplementedError(f"{type(self.model).__name__} does not expose encode_process().")

    def encode(self, inp, dynamic_features):
        r"""
        The latent state of the network for one window, before decoding.

        The input goes through the same preparation as in :meth:`predict`; only
        the decoder, the bias correction and the denormalization are skipped.

        Parameters
        ----------
        inp, dynamic_features
            As in :meth:`predict`, which also describes the noise.

        Returns
        -------
        torch.Tensor
            The latent features, ``(B, embed_dim, h, w)`` on the internal grid of the network.

        Raises
        ------
        NotImplementedError
            If the network does not expose ``encode_process``.
        """
        self.check_encodable()

        model_input = self._model_input(inp, dynamic_features)

        return self._network(model_input, method=self.model.encode_process)

    def predict_step(self, step: RolloutStep, use_checkpoint: bool = False):
        r"""
        Predict one step of a rollout.

        Parameters
        ----------
        step : RolloutStep
            The step, as a :class:`~makani.utils.dataloaders.rollout.Rollout` hands it out.
        use_checkpoint : bool, optional
            Recompute the network in the backward pass instead of keeping its
            activations. By default ``False``.

        Returns
        -------
        torch.Tensor
            The prediction, ``(B * E, 1, C_out, H, W)``, in data space, at
            the batch of the rollout with its ``E`` ensemble members folded in.

        Raises
        ------
        RuntimeError
            If the model is stochastic and its noise is not at the state the
            rollout advanced it to for this step, e.g. because the rollout does
            not drive the noise of this model, or if ``use_checkpoint`` is set
            for a model that is not safe to checkpoint, see
            :func:`_assert_checkpoint_safe`.
        """
        self._check_step_noise(step)
        if use_checkpoint:
            self._check_checkpoint_safe()
        return self._predict_window(step.input, step.input_dynamic_features, use_checkpoint)

    def _check_step_noise(self, step: RolloutStep):
        # a stochastic model predicts from the state its rollout advanced for this step, not from a state
        # someone else has advanced since, nor from a stale one
        if self.noise.is_stochastic and (step.noise_version != self.noise.version):
            raise RuntimeError(
                f"Step {step.index} was handed out at version {step.noise_version} of the noise state, which is "
                f"at version {self.noise.version} now: the noise has been advanced since, or the rollout does "
                f"not drive the noise of this model, see Rollout(..., noise=model.noise)"
            )

    def forward(self, rollout: Rollout):
        r"""
        Run the model over all steps of a rollout.

        Each prediction is fed back into the window of the next step, see
        :class:`~makani.utils.dataloaders.rollout.Rollout`. In training mode,
        push-forward mode and rollout checkpointing apply if configured. For a
        single step outside a rollout, use :meth:`predict`; for the predictions
        one step at a time, use :meth:`steps`.

        Parameters
        ----------
        rollout : Rollout
            The rollout, not started yet.

        Returns
        -------
        torch.Tensor
            The predictions, ``(B * E, n_steps, C_out, H, W)``, in data
            space, at the batch of the rollout with its ``E`` ensemble members
            folded in; :meth:`~makani.utils.dataloaders.rollout.Rollout.unfold`
            splits them out. Step ``k`` predicts target step ``k``.

        Raises
        ------
        TypeError, ValueError, RuntimeError
            As in :meth:`steps`.
        """
        # (B * E, n_steps, C_out, H, W), step k predicts target step k
        return torch.cat(list(self.steps(rollout)), dim=1)

    def steps(self, rollout: Rollout):
        r"""
        Run the model over the steps of a rollout, one prediction at a time.

        The same rollout as :meth:`forward`, but the predictions are handed out
        as they are made, so that the caller can release each one before the
        next, e.g. in an evaluation over many lead times. Each prediction is
        fed back into the window of the next step before it is handed out, so
        the rollout has finished once the last one is. The rollout is checked
        right away, not at the first step.

        Parameters
        ----------
        rollout : Rollout
            The rollout, not started yet.

        Returns
        -------
        iterator of torch.Tensor
            The prediction of each step, ``(B * E, 1, C_out, H, W)``, in data
            space, at the batch of the rollout with its ``E`` ensemble members
            folded in.

        Raises
        ------
        TypeError
            If ``rollout`` is not a :class:`~makani.utils.dataloaders.rollout.Rollout`.
        ValueError
            If the rollout has no steps, or does not drive the noise of this
            model, see :attr:`noise`.
        RuntimeError
            If the rollout has been started already, or if rollout
            checkpointing is configured for a model that is not safe to
            checkpoint, see :func:`_assert_checkpoint_safe`.
        """
        if not isinstance(rollout, Rollout):
            raise TypeError(
                f"{type(self).__name__} takes a Rollout, got {type(rollout).__name__}; "
                "use predict(x, dynamic_features) for a single step"
            )
        if (rollout.noise is not None) and (rollout.noise is not self.noise):
            raise ValueError(
                "The rollout drives the noise of another model: make it with Rollout(..., noise=model.noise)"
            )
        if self.noise.is_stochastic and (rollout.noise is None):
            raise ValueError(
                "The model is stochastic, but the rollout has no noise: make it with Rollout(..., noise=model.noise)"
            )
        if rollout.n_steps == 0:
            raise ValueError("The rollout has no steps to predict")
        if rollout.started:
            raise RuntimeError("The rollout has been stepped already: a rollout can be run only once, make a new one")
        push_forward = self.training and self.push_forward_mode
        # push-forward already detaches between steps, so checkpointing would be redundant there, and a
        # single step has nothing to gain from it
        use_checkpoint = (
            self.training
            and self.multistep_checkpoint
            and torch.is_grad_enabled()
            and not push_forward
            and rollout.n_steps > 1
        )
        if use_checkpoint:
            self._check_checkpoint_safe()

        return self._steps(rollout, push_forward, use_checkpoint)

    def _check_checkpoint_safe(self):
        # once, on the first checkpointed step, so that loading a model for evaluation never fails on it
        if not self._checkpoint_checked:
            _assert_checkpoint_safe(self.model, self.noise)
            self._checkpoint_checked = True

    def _steps(self, rollout, push_forward, use_checkpoint):
        for step in rollout:
            pred = self.predict_step(step, use_checkpoint=use_checkpoint)
            rollout.advance(pred.detach() if push_forward else pred)
            yield pred
