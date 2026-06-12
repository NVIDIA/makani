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
Precision / autocast helper.

Mixed precision in makani is selected by a single ``mode`` string that combines an
AMP dtype with an optional transformer-engine FP8 recipe:

    none                 -> fp32, no autocast
    fp16                 -> torch.autocast(float16)
    bf16                 -> torch.autocast(bfloat16)
    bf16-fp8_delayed     -> torch.autocast(bfloat16) + te.fp8_autocast(DelayedScaling)
    fp16-fp8_hybrid      -> torch.autocast(float16)  + te.fp8_autocast(DelayedScaling, HYBRID)
    bf16-mxfp8           -> torch.autocast(bfloat16)  + te.fp8_autocast(MXFP8BlockScaling)
    ...

``AutocastManager`` parses the mode once and hands out a single context manager that
nests the AMP autocast and (if requested) the FP8 autocast at the right level, so call
sites stay flat:

    self.autocast = AutocastManager(params.amp_mode, fp8_group=comm.get_group("data"))
    self.gscaler  = amp.GradScaler("cuda", enabled=self.autocast.grad_scaler_enabled)
    ...
    with self.autocast():
        out = model(inp)

instead of hand-nesting ``with amp.autocast(...): with te.fp8_autocast(...): ...``.
"""

import contextlib

import torch
from torch import amp

from makani.utils.te_helpers import TE_AVAILABLE, get_te


_AMP_DTYPES = {
    "none": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _make_fp8_recipe(key: str):
    """Build a transformer-engine FP8 recipe from a short key.

    Recipe classes vary across TE versions, so each is imported lazily and a clear
    error is raised if the key (or the recipe class) is unavailable. This dict is the
    extension point for new recipes.
    """
    if not TE_AVAILABLE:
        raise RuntimeError(f"precision mode requests fp8 recipe '{key}' but transformer_engine is not installed")

    from transformer_engine.common import recipe as te_recipe

    builders = {
        # delayed scaling (amax history); HYBRID = e4m3 fwd / e5m2 bwd
        "fp8_delayed": lambda: te_recipe.DelayedScaling(fp8_format=te_recipe.Format.HYBRID),
        "fp8_hybrid": lambda: te_recipe.DelayedScaling(fp8_format=te_recipe.Format.HYBRID),
        "fp8_e4m3": lambda: te_recipe.DelayedScaling(fp8_format=te_recipe.Format.E4M3),
        # current (per-tensor, no history) and block scaling (mxfp8); newer TE only
        "fp8_current": lambda: te_recipe.Float8CurrentScaling(),
        "mxfp8": lambda: te_recipe.MXFP8BlockScaling(),
    }

    if key not in builders:
        raise ValueError(f"unknown fp8 recipe '{key}' (supported: {sorted(builders)})")

    try:
        return builders[key]()
    except AttributeError as err:
        raise RuntimeError(f"fp8 recipe '{key}' is not available in this transformer_engine version") from err


def parse_precision_mode(mode: str):
    """Split a precision ``mode`` string into ``(amp_dtype, fp8_recipe_key)``.

    ``fp8_recipe_key`` is ``None`` when no FP8 recipe is requested. Raises ``ValueError``
    on an unknown amp precision or a malformed mode.
    """
    mode = (mode or "none").lower()
    parts = mode.split("-")
    if len(parts) > 2:
        raise ValueError(f"malformed precision mode '{mode}' (expected '<amp>' or '<amp>-<fp8recipe>')")

    amp_part = parts[0]
    fp8_part = parts[1] if len(parts) == 2 else None

    if amp_part not in _AMP_DTYPES:
        raise ValueError(f"unknown amp precision '{amp_part}' in mode '{mode}' (expected one of {sorted(_AMP_DTYPES)})")

    return _AMP_DTYPES[amp_part], fp8_part


class AutocastManager:
    """Parses a precision mode once and yields the combined autocast context manager.

    Parameters
    ----------
    mode : str
        Precision mode, e.g. ``"bf16"`` or ``"bf16-fp8_delayed"`` (see module docstring).
    device_type : str
        Device type passed to ``torch.autocast`` (default ``"cuda"``).
    fp8_group : Optional[torch.distributed.ProcessGroup]
        Process group for FP8 amax / scale synchronization (delayed scaling). Pass the
        data-parallel group; ignored when no FP8 recipe is selected.

        NOTE: fp8_group must exclude the tensor-parallel/matmul group. The column/row
        ``DistributedMatmul`` shards weights across matmul, so each matmul rank owns a
        distinct weight shard with its own legitimate amax/scale -- reducing amax over
        matmul would couple independent shards and clamp them to a common (over-
        conservative) scale. The data group is orthogonal to model (spatial x matmul),
        so it already excludes matmul. Current scaling / MXFP8 derive scales locally and
        do not use this group.
    """

    def __init__(self, mode, device_type="cuda", fp8_group=None):
        self.mode = mode
        self.device_type = device_type
        self.fp8_group = fp8_group

        self.amp_dtype, self._fp8_recipe_key = parse_precision_mode(mode)
        self.amp_enabled = self.amp_dtype != torch.float32
        self.fp8_enabled = self._fp8_recipe_key is not None

        # an fp8 mode must fail loudly when transformer_engine is missing rather than
        # silently degrade to the plain amp dtype
        if self.fp8_enabled and not TE_AVAILABLE:
            raise RuntimeError(
                f"precision mode '{mode}' requests an fp8 recipe but transformer_engine is not installed; "
                f"install transformer_engine or choose a non-fp8 mode (none/fp16/bf16)"
            )

        # build the recipe eagerly so an unsupported mode fails at construction, not in forward
        self._fp8_recipe = _make_fp8_recipe(self._fp8_recipe_key) if self.fp8_enabled else None

    @property
    def grad_scaler_enabled(self) -> bool:
        """The loss scaler is only needed for fp16; bf16 and fp8 do not use it."""
        return self.amp_dtype == torch.float16

    @contextlib.contextmanager
    def __call__(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(amp.autocast(device_type=self.device_type, enabled=self.amp_enabled, dtype=self.amp_dtype))
            if self.fp8_enabled:
                te = get_te()
                stack.enter_context(te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe, fp8_group=self.fp8_group))
            yield
