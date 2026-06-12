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
    offenders = sorted({
        type(m).__name__
        for m in module.modules()
        if isinstance(getattr(m, "rng_cpu", None), torch.Generator)
        or isinstance(getattr(m, "rng_gpu", None), torch.Generator)
    })
    if offenders:
        raise RuntimeError(
            f"multistep_checkpoint is incompatible with modules carrying private RNG "
            f"generators (found: {offenders}). Their dropout masks are not restored on "
            f"the checkpoint recompute, which would corrupt gradients. Disable "
            f"multistep_checkpoint or make these modules use the global RNG."
        )


class SingleStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super().__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()

    def forward(self, inp, update_state=True, replace_state=True):
        # update internal state
        if update_state:
            self.preprocessor.update_internal_state(replace_state=replace_state)

        # append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # now normalize
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # now add static features if requested
        inpans = self.preprocessor.add_static_features(inpan)

        # forward pass
        yn = self.model(inpans)

        # perform bias correction if requested
        yn = self.preprocessor.correct_bias(yn)

        # undo normalization
        y = self.preprocessor.history_denormalize(yn, target=True)

        return y


class MultiStepWrapper(nn.Module):
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
        # update internal state
        if update_state:
            self.preprocessor.update_internal_state(replace_state=replace_state)

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

        return y

    def forward(self, inp, update_state=True, replace_state=True):
        # decide which routine to call
        if self.training:
            y = self._forward_train(inp, update_state=update_state, replace_state=replace_state)
        else:
            y = self._forward_eval(inp, update_state=update_state, replace_state=replace_state)

        return y
