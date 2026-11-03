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

"""
Control over the stochastic state of a model.

A model may draw noise in several places: at its input, as the input noise of
the preprocessor does, or inside the network, e.g. in its latent space. Each
such place is a *noise process*, a module with a state that is advanced once
per autoregressive step, see :class:`~makani.models.noise.BaseNoiseS2`. The
model owns its processes: they are submodules, so that their learnable
parameters are trained, checkpointed and moved with the model.

What drives the processes is the :class:`NoiseController` of the model. A
:class:`~makani.utils.dataloaders.rollout.Rollout` advances it before each
step, replacing the state at the first one, and snapshots or restores it to
continue a noise sequence in another rollout; the model package does the same
for single steps. Nothing outside the model draws the samples: the model reads
them from its current state while it predicts, so that they stay in the
autograd graph of its learnable parameters::

    noise = model.noise
    noise.advance(replace_state=True, batch_size=B)   # before the first step
    pred = model.predict(x, features)                  # reads the current state
    snapshot = noise.snapshot()                        # rewind point
    ...
    noise.restore(snapshot)                            # rewind

Advancing is the only part that consumes random numbers, and it happens
outside the prediction, which only reads the state. A sample drawn outside
the network, as the input noise is, is kept for the backward pass, so a step
recomputed under rollout checkpointing sees the same noise. A network that
reads its own processes inside its forward would read them again in the
recompute, after the rollout has advanced them; rollout checkpointing is
refused for such networks, see :class:`~makani.models.stepper.StepWrapper`.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from makani.utils import comm

# the offset between the seeds of the processes of one model in set_rng, a prime well above the
# range of the per-rank offsets, so that re-seeded processes do not share a stream
_SEED_STRIDE = 104729


@torch.compiler.disable
def run_eager(fn, *args, **kwargs):
    """Run a callable outside torch.compile (forces a graph break).

    The noise field is complex-valued (torch.complex + inverse SHT) and inductor's Triton
    backend cannot codegen complex dtypes (KeyError: 'complex64'). A method-level
    @torch.compiler.disable on the noise module's forward is NOT honored when dynamo inlines
    the nn.Module call (it traces straight into it), whereas a disable on a plain function call
    like this one is respected — so we break at the call site instead.
    """
    return fn(*args, **kwargs)


def noise_base_seed(centered: bool = False) -> Tuple[int, bool]:
    r"""
    The seed of a noise process on this rank, and whether it is reflected.

    Every rank of the model-parallel group draws its own part of the field,
    and every data-parallel rank its own samples. With ``centered`` noise, the
    ensemble ranks come in pairs, ``(0, 1), (2, 3), ...``, that draw the same
    field with opposite signs, so that the ensemble mean of the noise is zero.

    Parameters
    ----------
    centered : bool, optional
        Pair up the ensemble ranks as described above. By default ``False``.

    Returns
    -------
    seed : int
        The seed of the process on this rank.
    reflect : bool
        Whether this rank draws the reflected field of its pair.
    """
    if not centered:
        seed = 333 + comm.get_rank("model") + comm.get_size("model") * comm.get_rank("data")
        return seed, False

    # ranks (0,1), (2,3), ... map to the same effective rank, since they differ only by the reflection
    ensemble_eff_rank = comm.get_rank("ensemble") // 2
    reflect = comm.get_rank("ensemble") % 2 == 0
    seed = (
        333
        + comm.get_rank("model")
        + comm.get_size("model") * ensemble_eff_rank
        + comm.get_size("model") * comm.get_size("ensemble") * comm.get_rank("batch")
    )
    return seed, reflect


@dataclass(frozen=True)
class NoiseState:
    r"""
    A snapshot of one noise process.

    Holds the state tensor of the process, detached, and the states of its
    generators, so that a restored process continues the sequence an
    uninterrupted one would have drawn.
    """

    tensor: torch.Tensor
    rng_cpu: Optional[torch.Tensor]
    rng_gpu: Optional[torch.Tensor]

    @property
    def batch_size(self) -> int:
        """The batch the state was taken at."""
        return self.tensor.shape[0]


@dataclass(frozen=True)
class NoiseSnapshot:
    r"""
    A snapshot of all noise processes of a model, see :meth:`NoiseController.snapshot`.

    Attributes
    ----------
    states : Mapping[str, NoiseState]
        The state of each process, by name.
    """

    states: Mapping[str, NoiseState]

    @property
    def batch_size(self) -> Optional[int]:
        """The batch the snapshot was taken at, ``None`` for a model without noise."""
        sizes = {state.batch_size for state in self.states.values()}
        if len(sizes) > 1:
            raise ValueError(f"The noise states were taken at different batches, {sorted(sizes)}")
        return sizes.pop() if sizes else None


class NoiseController:
    r"""
    Drive the noise processes of a model: advance, sample, snapshot, restore and re-seed them.

    The controller holds references to the processes, it does not own them:
    they are submodules of the model, which trains, checkpoints and moves them.
    For the same reason the controller is not an ``nn.Module``, which would
    register them a second time.

    A model without noise has a controller without processes, which does
    nothing; :attr:`is_stochastic` tells the two apart.

    Parameters
    ----------
    processes : Mapping[str, nn.Module], optional
        The noise processes by name, in the order they are seeded in
        :meth:`set_rng`. Each provides ``update(replace_state, batch_size)``,
        a call for its sample, ``is_stateful()``, ``set_rng(seed)``,
        ``reset()``, ``get_tensor_state()``, ``set_tensor_state(tensor)``,
        ``get_rng_state()``, ``set_rng_state(cpu, gpu)`` and a ``state``
        tensor with the batch first, as :class:`~makani.models.noise.BaseNoiseS2`
        does.
    """

    def __init__(self, processes: Optional[Mapping[str, nn.Module]] = None):
        self._processes: Dict[str, nn.Module] = dict(processes or {})
        # counts the changes of the state, by any caller, see version
        self._version = 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._processes)})"

    def __contains__(self, name: str) -> bool:
        return name in self._processes

    def __iter__(self) -> Iterator[str]:
        return iter(self._processes)

    def __len__(self) -> int:
        return len(self._processes)

    @property
    def is_stochastic(self) -> bool:
        """Whether the model has any noise process."""
        return len(self._processes) > 0

    @property
    def version(self) -> int:
        """Counts the changes of the state, so that a caller can tell whether someone else advanced it since."""
        return self._version

    @property
    def batch_size(self) -> Optional[int]:
        """The batch the state is sized for, ``None`` without noise."""
        for process in self._processes.values():
            return process.state.shape[0]
        return None

    def process(self, name: str) -> nn.Module:
        """The noise process of that name."""
        return self._processes[name]

    def advance(self, replace_state: bool = False, batch_size: Optional[int] = None):
        r"""
        Advance every process by one step.

        Parameters
        ----------
        replace_state : bool, optional
            Draw a fresh state from the stationary distribution instead of
            taking one autoregressive step. Required to change the batch of
            stateful noise. By default ``False``.
        batch_size : int, optional
            Size the state to this batch. ``None`` keeps the current size.

        Raises
        ------
        RuntimeError
            If the state of stateful noise would have to be resized while
            continuing its sequence, see below.

        Notes
        -----
        The state of stateful noise (``DiffusionNoiseS2``) carries the history
        of its autoregressive sequence. Resizing it zeroes that history, which
        would silently restart the noise from zero, with a spin-up transient
        instead of the stationary distribution. So a resize is accepted only
        together with ``replace_state``. Memoryless noise redraws its state
        every step and may be resized at any time.
        """
        if not replace_state and (batch_size is not None):
            for name, process in self._processes.items():
                current_batch = process.state.shape[0]
                if (current_batch != batch_size) and process.is_stateful():
                    raise RuntimeError(
                        f"Refusing to resize the state of the noise process '{name}' from batch {current_batch} "
                        f"to {batch_size} while continuing its autoregressive sequence. Resizing would zero the "
                        f"history the state carries and silently restart the noise from zero. Replace the state "
                        f"(replace_state=True) to start a fresh sequence at the new batch, or keep the batch "
                        f"fixed for the whole sequence."
                    )

        for process in self._processes.values():
            process.update(replace_state=replace_state, batch_size=batch_size)
        if self._processes:
            self._version += 1

    def sample(self, name: str) -> torch.Tensor:
        r"""
        The sample of a process at its current state.

        Computed eagerly, outside ``torch.compile``, see :func:`run_eager`. The
        sample is a deterministic function of the state and of the parameters
        of the process, so it stays in their autograd graph: draw it where it
        is used, inside the prediction.

        Parameters
        ----------
        name : str
            The process.

        Returns
        -------
        torch.Tensor
            The sample, ``(B, T, C, *grid)`` for the input noise.
        """
        return run_eager(self._processes[name])

    def snapshot(self) -> NoiseSnapshot:
        r"""
        A detached snapshot of the state and of the generators of every process.

        Returns
        -------
        NoiseSnapshot
            The snapshot, to pass to :meth:`restore`; it does not change when
            the processes advance.
        """
        states = {}
        for name, process in self._processes.items():
            rng_cpu, rng_gpu = process.get_rng_state()
            states[name] = NoiseState(tensor=process.get_tensor_state(), rng_cpu=rng_cpu, rng_gpu=rng_gpu)
        return NoiseSnapshot(states=states)

    def restore(self, snapshot: NoiseSnapshot, batch_size: Optional[int] = None):
        r"""
        Rewind the processes to a snapshot.

        Parameters
        ----------
        snapshot : NoiseSnapshot
            From :meth:`snapshot`, of a model with the same processes.
        batch_size : int, optional
            The batch the caller continues at. If given, a snapshot taken at
            another batch is refused.

        Raises
        ------
        ValueError
            If the snapshot holds other processes than this controller, or was
            taken at another batch than ``batch_size``.
        """
        if set(snapshot.states) != set(self._processes):
            raise ValueError(
                f"The snapshot holds the noise processes {sorted(snapshot.states)}, "
                f"the model has {sorted(self._processes)}"
            )
        if (batch_size is not None) and self._processes and (snapshot.batch_size != batch_size):
            raise ValueError(
                f"The noise state was taken at batch {snapshot.batch_size}, the caller runs at batch {batch_size}"
            )

        for name, process in self._processes.items():
            state = snapshot.states[name]
            process.set_tensor_state(state.tensor)
            process.set_rng_state(state.rng_cpu, state.rng_gpu)
        if self._processes:
            self._version += 1

    def set_rng(self, seed: int = 333, reset: bool = True):
        r"""
        Re-seed every process, optionally zeroing its state.

        Use this to make a forecast reproducible, or to decorrelate ensemble
        members by giving each a distinct seed. The processes are seeded in
        order, ``seed``, ``seed + stride``, ..., so that they draw different
        streams; a model with a single process is seeded with ``seed``.

        Parameters
        ----------
        seed : int, optional
            The seed, by default ``333``.
        reset : bool, optional
            Also zero the states, by default ``True``. Keep them only to
            continue a sequence under a new seed.
        """
        for idx, process in enumerate(self._processes.values()):
            process.set_rng(seed + idx * _SEED_STRIDE)
            if reset:
                process.reset()
        if self._processes:
            self._version += 1
