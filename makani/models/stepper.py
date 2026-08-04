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

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from makani.models.preprocessor import Preprocessor2D


def _assert_checkpoint_safe(module: nn.Module):
    """Guard for rollout activation checkpointing.

    Checkpointing recomputes each step's model forward during backward. Global-RNG
    stochastic ops (e.g. DropPath) are covered by checkpoint's preserve_rng_state, but
    modules that advance their *own* torch.Generator (e.g. SeededDropout2d) are not:
    their recomputed mask would diverge from the original forward and corrupt gradients.
    Fail loudly rather than train silently-wrong.
    """
    offenders = sorted(
        {
            type(m).__name__
            for m in module.modules()
            if isinstance(getattr(m, "rng_cpu", None), torch.Generator)
            or isinstance(getattr(m, "rng_gpu", None), torch.Generator)
        }
    )
    if offenders:
        raise RuntimeError(
            f"multistep_checkpoint is incompatible with modules carrying private RNG "
            f"generators (found: {offenders}). Their dropout masks are not restored on "
            f"the checkpoint recompute, which would corrupt gradients. Disable "
            f"multistep_checkpoint or make these modules use the global RNG."
        )


class SingleStepWrapper(nn.Module):
    r"""
    Bind a model to its preprocessor for a single forecast step.

    Wraps the network so callers can hand it raw data and get a prediction back
    in physical units, without having to replicate the preprocessing pipeline at
    every call site. One ``forward`` performs: state update, appending
    unpredicted and static features, history normalization, the model forward,
    bias correction, and denormalization of the output.

    Parameters
    ----------
    params : ParamsBase
        Configuration, forwarded to :class:`~makani.models.preprocessor.Preprocessor2D`.
    model_handle : callable
        Zero-argument factory returning the network to wrap.

    See Also
    --------
    MultiStepWrapper : the autoregressive multi-step counterpart.
    """

    def __init__(self, params, model_handle):
        super().__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()

    def _preprocess(self, inp, update_state=True, replace_state=True):
        """Run the shared input pipeline for one step.

        The stochastic state is sized to THIS call's batch. The noise module is
        constructed with ``params.batch_size`` -- a training setting, and 1 for a
        packaged model -- but inference legitimately drives it with other
        batches, e.g. an ensemble pushing B*E members through as a single
        batched forward. Without this, ``append_unpredicted_features`` rejects
        the call with "batch mismatch between input_noise state (1) and input".
        ``_ensure_state`` reallocates only when the batch actually changes, so
        the steady-state cost is zero; the flip side is that changing the batch
        mid-rollout discards a stateful noise module's history.
        """
        # update internal state
        if update_state:
            self.preprocessor.update_internal_state(replace_state=replace_state, batch_size=inp.shape[0])

        # append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # now normalize
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # now add static features if requested
        return self.preprocessor.add_static_features(inpan)

    def forward(self, inp, update_state=True, replace_state=True):
        r"""
        Predict a single step from raw input.

        Parameters
        ----------
        inp : torch.Tensor
            Input of shape ``(B, (n_history + 1) * C, H, W)`` in physical units.
        update_state : bool, optional
            Advance the stochastic noise state before the forward pass, by
            default ``True``.
        replace_state : bool, optional
            Draw a fresh noise state rather than taking one autoregressive
            step, by default ``True``. Required if the batch size changed since
            the last call and the noise is stateful.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(B, C_out, H, W)``, denormalized back to
            physical units.
        """
        inpans = self._preprocess(inp, update_state=update_state, replace_state=replace_state)

        # forward pass
        yn = self.model(inpans)

        # perform bias correction if requested
        yn = self.preprocessor.correct_bias(yn)

        # undo normalization
        y = self.preprocessor.history_denormalize(yn, target=True)

        # SPPT-style multiplicative tendency perturbation (no-op unless configured)
        y = self.preprocessor.apply_stochastic_physics(inp, y)

        return y

    def encode_process(self, inp, update_state=True, replace_state=True):
        r"""Prepare one input and return backbone features before decoding.

        Shares :meth:`_preprocess` with :meth:`forward`, so the preprocessing is
        identical by construction; only the decoder, bias correction, and target
        denormalization are skipped.

        Parameters
        ----------
        inp : torch.Tensor
            Input of shape ``(B, (n_history + 1) * C, H, W)`` in physical units.
        update_state : bool, optional
            Advance the stochastic noise state first, by default ``True``.
        replace_state : bool, optional
            Draw a fresh noise state rather than stepping, by default ``True``.

        Returns
        -------
        torch.Tensor
            Latent features of shape ``(B, embed_dim, h, w)`` on the model's
            internal grid.

        Raises
        ------
        NotImplementedError
            If the wrapped model does not expose ``encode_process``.
        """
        if not hasattr(self.model, "encode_process"):
            raise NotImplementedError(f"{type(self.model).__name__} does not expose encode_process().")

        inpans = self._preprocess(inp, update_state=update_state, replace_state=replace_state)

        return self.model.encode_process(inpans)


class MultiStepWrapper(nn.Module):
    r"""
    Bind a model to its preprocessor for an autoregressive rollout.

    Runs the wrapped model for ``n_future + 1`` steps, feeding each prediction
    back in as the next input and sliding the history window forward. Returns
    all steps stacked, which is what multi-step training losses need.

    Two options control cost and stability:

    * **Push-forward mode.** Detaches the input at each step, so gradients do
      not flow through the whole rollout. This trains the model to be robust to
      its own errors without the cost -- or the instability -- of
      backpropagating through many steps.
    * **Rollout checkpointing.** Recomputes each step's forward during the
      backward pass instead of retaining its activations, turning the
      ``O(n_future)`` activation multiplier of backprop-through-time into
      ``O(1)`` at the cost of one extra forward per step.

    Parameters
    ----------
    params : ParamsBase
        Configuration. Reads ``n_future``, ``multistep.push_forward``, and
        ``multistep_checkpoint``; the rest is forwarded to the preprocessor.
    model_handle : callable
        Zero-argument factory returning the network to wrap.

    See Also
    --------
    SingleStepWrapper : the single-step counterpart.
    """

    def __init__(self, params, model_handle):
        super().__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()
        multistep_parameters = params.get("multistep", {"push_forward": False})
        self.push_forward_mode = multistep_parameters["push_forward"]

        # rollout-level activation checkpointing: recompute each step's model forward in
        # the backward pass instead of retaining its full activation graph. Turns the
        # O(n_future) activation multiplier of backprop-through-time back into O(1) at the
        # cost of one extra forward per step. Set via the --multistep_checkpoint CLI flag
        # (top-level param, like n_future). Off by default.
        self.multistep_checkpoint = params.get("multistep_checkpoint", False)
        if self.multistep_checkpoint:
            _assert_checkpoint_safe(self.model)

        # collect parameters for history
        self.n_future = params.n_future

    def _forward_train(self, inp, update_state=True, replace_state=True):
        result = []
        inpt = inp

        # initialize fresh buffer: decide whether we want to replace the state
        if update_state:
            self.preprocessor.update_internal_state(replace_state=replace_state)

        # do the rollout
        for step in range(self.n_future + 1):

            # in push-forward mode, we need to detach the tensor:
            if self.push_forward_mode:
                inpt = inpt.detach()

            # add unpredicted features
            inpa = self.preprocessor.append_unpredicted_features(inpt)

            # do history normalization
            self.preprocessor.history_compute_stats(inpa)
            inpan = self.preprocessor.history_normalize(inpa, target=False)

            # add static features
            inpans = self.preprocessor.add_static_features(inpan)

            # prediction
            # Only the pure model forward is checkpointed; the stateful preprocessor calls
            # (history/noise/state updates) stay outside the checkpoint so they run once and
            # are not re-executed during the backward recompute. Global-RNG ops inside the
            # model are handled by preserve_rng_state=True; private-generator modules are
            # rejected up-front by _assert_checkpoint_safe. Push-forward mode already detaches
            # between steps, so checkpointing would be redundant there.
            if self.multistep_checkpoint and self.training and torch.is_grad_enabled() and not self.push_forward_mode:
                predn = checkpoint(self.model, inpans, use_reentrant=False, preserve_rng_state=True)
            else:
                predn = self.model(inpans)

            # perform bias correction if requested
            predn = self.preprocessor.correct_bias(predn)

            # append the denormalized result to output list
            # important to do that here, otherwise normalization stats
            # will have been updated later:
            pred = self.preprocessor.history_denormalize(predn, target=True)

            # SPPT-style multiplicative tendency perturbation (no-op unless configured).
            # Applied before append_history so the perturbed state feeds the next rollout step,
            # letting the spread grow through the learned dynamics.
            pred = self.preprocessor.apply_stochastic_physics(inpt, pred)

            # append output
            result.append(pred)

            if step == self.n_future:
                break

            # update internal buffer
            self.preprocessor.update_internal_state(replace_state=False)

            # append history
            inpt = self.preprocessor.append_history(inpt, pred, step)

        # concat the tensors along channel dim to be compatible with flattened target
        result = torch.cat(result, dim=1)

        return result

    def _forward_eval(self, inp, update_state=True, replace_state=True):
        # update internal state; size the noise state to this call's batch so that
        # batched inference works regardless of params.batch_size. See
        # SingleStepWrapper._preprocess for the rationale. The training path
        # deliberately keeps the fixed-batch behaviour.
        if update_state:
            self.preprocessor.update_internal_state(replace_state=replace_state, batch_size=inp.shape[0])

        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # do history normalization
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # add static features
        inpans = self.preprocessor.add_static_features(inpan)

        # important, remove normalization here,
        # because otherwise normalization stats are already outdated
        yn = self.model(inpans)

        # perform bias correction if requested
        yn = self.preprocessor.correct_bias(yn)

        # important, remove normalization here,
        # because otherwise normalization stats are already outdated
        y = self.preprocessor.history_denormalize(yn, target=True)

        # SPPT-style multiplicative tendency perturbation (no-op unless configured)
        y = self.preprocessor.apply_stochastic_physics(inp, y)

        return y

    def forward(self, inp, update_state=True, replace_state=True):
        r"""
        Roll the model forward autoregressively and return all predicted steps.

        Dispatches to the training or evaluation path based on the module's
        training mode; the training path additionally honors push-forward mode
        and rollout checkpointing, which do not apply under ``no_grad``.

        Parameters
        ----------
        inp : torch.Tensor
            Initial input of shape ``(B, (n_history + 1) * C, H, W)`` in
            physical units.
        update_state : bool, optional
            Advance the stochastic noise state before the rollout, by default
            ``True``.
        replace_state : bool, optional
            Draw a fresh noise state rather than continuing an existing
            trajectory, by default ``True``.

        Returns
        -------
        torch.Tensor
            Predictions for all ``n_future + 1`` steps, concatenated along the
            channel dimension as ``(B, (n_future + 1) * C_out, H, W)``.
        """
        # decide which routine to call
        if self.training:
            y = self._forward_train(inp, update_state=update_state, replace_state=replace_state)
        else:
            y = self._forward_eval(inp, update_state=update_state, replace_state=replace_state)

        return y
