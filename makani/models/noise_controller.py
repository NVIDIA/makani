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
continue a noise sequence in another rollout; the model package advances it
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

import contextlib
import itertools
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from makani.models.noise import run_eager

# the offset between the seeds of the processes of one model in set_rng, a prime well above the
# range of the per-rank offsets, so that re-seeded processes do not share a stream
_SEED_STRIDE = 104729

# the versions of all controllers come from one counter, so that no two states, of any controllers, share one
_VERSIONS = itertools.count(1)


@dataclass(frozen=True)
class NoiseState:
    r"""
    A snapshot of one noise process.

    Holds the state tensor of the process, detached, and the states of its
    generators, so that a restored process continues the sequence an
    uninterrupted one would have drawn.

    Attributes
    ----------
    tensor : torch.Tensor
        The state of the process, a detached copy, with the batch first.
    rng_cpu : torch.Tensor or None
        The state of its CPU generator.
    rng_gpu : torch.Tensor or None
        The state of its GPU generator, ``None`` without one.
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
        # identifies the state, changed by any caller, see version; 0 until the first change
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
        """Identifies the current state, so that a caller can tell whether someone else changed it since.

        Every change (:meth:`advance`, :meth:`restore`, :meth:`set_rng`) gives the state a new version, taken
        from a counter shared by all controllers, so that the version of one controller never matches a state
        of another one.
        """
        return self._version

    @property
    def batch_size(self) -> Optional[int]:
        """The batch the state is sized for, ``None`` without noise."""
        for process in self._processes.values():
            return process.state.shape[0]
        return None

    def process(self, name: str) -> nn.Module:
        r"""
        The noise process of that name, the module the model owns.

        Parameters
        ----------
        name : str
            The process, e.g. ``"input_noise"`` or ``"model.<name>"``.

        Returns
        -------
        nn.Module
            The process.

        Raises
        ------
        KeyError
            If the model has no process of that name.
        """
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
            self._version = next(_VERSIONS)

    def sample(self, name: str) -> torch.Tensor:
        r"""
        The sample of a process at its current state.

        Computed eagerly, outside ``torch.compile``, see :func:`~makani.models.noise.run_eager`. The
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

        Raises
        ------
        KeyError
            If the model has no process of that name.
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
            If the snapshot holds other processes than this controller, was
            taken at another batch than ``batch_size``, or a process refuses
            its state, e.g. one of another layout. The processes are left as
            they were then.
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

        # all or nothing: a process that refuses its state, e.g. one of another layout, would otherwise leave the
        # processes before it rewound and the others not, at the version of the state before
        previous = self.snapshot()
        try:
            self._write(snapshot)
        except Exception:
            self._write(previous)
            raise
        if self._processes:
            self._version = next(_VERSIONS)

    def live_states(self, names: Optional[Iterator[str]] = None) -> Dict[str, torch.Tensor]:
        r"""
        The state of the processes as it is, without their generators, see :meth:`drawing_from`.

        A draw of a process is a function of its state and its parameters only, so the state is all
        it takes to draw the same noise again. The states are not copied: the graph of the state of
        learnable stateful noise, through which its parameters reach its draws, has to stay intact for
        a draw from it to be the same draw, for the backward too. A process replaces its state rather
        than writing into it, so the states stay as they are while the processes advance.

        Parameters
        ----------
        names : iterable of str, optional
            The processes, by default all of them.

        Returns
        -------
        dict
            The state of each process by name.
        """
        names = list(self._processes) if names is None else list(names)
        return {name: self._processes[name].get_live_state() for name in names}

    @contextlib.contextmanager
    def drawing_from(self, states: Mapping[str, torch.Tensor]):
        r"""
        Draw from the given states for the duration of the context, then from the current ones again.

        For the recompute of a checkpointed step in the backward pass, which has to draw the noise
        the step was run with, after the rollout has advanced it: see
        :meth:`~makani.models.stepper.StepWrapper.predict_step`. The recompute draws from the very
        state of the step, so that its graph matches the one of the step. Nothing is advanced, so the
        version of the state does not change.

        Parameters
        ----------
        states : dict
            The states to draw from by process, from :meth:`live_states`.
        """
        current = self.live_states(states.keys())
        try:
            for name, state in states.items():
                self._processes[name].set_live_state(state)
            yield
        finally:
            for name, state in current.items():
                self._processes[name].set_live_state(state)

    def _write(self, snapshot: NoiseSnapshot):
        for name, process in self._processes.items():
            state = snapshot.states[name]
            process.set_tensor_state(state.tensor)
            process.set_rng_state(state.rng_cpu, state.rng_gpu)

    def get_rng_state(self) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        r"""
        The states of the generators of every process, without the noise states.

        What a training run needs to continue its noise after a restart: its
        rollouts replace the noise state at their first step, drawn from the
        generators, so the generators alone continue the sequence of draws.
        Small, unlike the noise states, which are as large as the spectral
        coefficients.

        Returns
        -------
        dict
            Process name to the ``(cpu, gpu)`` generator states, see
            :meth:`set_rng_state`.
        """
        return {name: process.get_rng_state() for name, process in self._processes.items()}

    def set_rng_state(self, states: Mapping[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]):
        r"""
        Restore the generators of every process from :meth:`get_rng_state`.

        The noise states are left as they are; a rollout that replaces them at
        its first step then draws what the run would have drawn.

        Parameters
        ----------
        states : Mapping
            Process name to the ``(cpu, gpu)`` generator states.

        Raises
        ------
        ValueError
            If the states are of other processes than this controller drives.
        """
        if set(states) != set(self._processes):
            raise ValueError(
                f"The generator states are of the noise processes {sorted(states)}, "
                f"the model has {sorted(self._processes)}"
            )
        for name, process in self._processes.items():
            process.set_rng_state(*states[name])
        if self._processes:
            self._version = next(_VERSIONS)

    def set_rng(self, *, seed: int = 333, reset: bool = True):
        r"""
        Re-seed every process, optionally zeroing its state.

        Use this to make a forecast reproducible, or to decorrelate ensemble
        members by giving each a distinct seed. The processes are seeded in
        order, ``seed``, ``seed + stride``, ..., so that they draw different
        streams; a model with a single process is seeded with ``seed``. A
        process with a ``rank_seed`` method lays its base seed out over the
        ranks, see :meth:`~makani.models.noise.BaseNoiseS2.rank_seed`, so that
        the spatial shards and the data-parallel ranks still draw their own
        streams, as when the model was built; serially, the seeds are the base
        seeds.

        Parameters
        ----------
        seed : int, optional
            The seed, by default ``333``.
        reset : bool, optional
            Also zero the states, by default ``True``. Keep them only to
            continue a sequence under a new seed.
        """
        for idx, process in enumerate(self._processes.values()):
            base = seed + idx * _SEED_STRIDE
            process.set_rng(process.rank_seed(base) if hasattr(process, "rank_seed") else base)
            if reset:
                process.reset()
        if self._processes:
            self._version = next(_VERSIONS)
