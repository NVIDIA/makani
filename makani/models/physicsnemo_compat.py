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
Compatibility helpers for building PhysicsNeMo modules across 1.x and 2.x.

PhysicsNeMo 2.0 changed how :meth:`physicsnemo.Module.from_torch` behaves in two
ways that matter to makani, and both are silent -- the old call still runs, it
just does less than it used to:

* **Registration became opt-in.** In 1.x every converted class was registered in
  the ``ModelRegistry`` automatically. In 2.x ``register`` defaults to ``False``,
  and an unregistered class cannot be reloaded via
  :meth:`physicsnemo.Module.from_checkpoint`.
* **The generated class is named differently.** 1.x appended a
  ``PhysicsNeMoModel`` suffix; 2.x uses the wrapped class name verbatim unless a
  ``name`` is supplied. That name is stored in the checkpoint metadata and is
  what ``from_checkpoint`` resolves against, so letting it change would strand
  checkpoints written under the other version.

Separately, ``ModelMetaData.name`` is deprecated in 2.x: setting it raises a
``DeprecationWarning`` and has no effect, since the model name now comes from the
class. In 1.x it is still read, for the logger label and for the default
``.mdlus`` filename. makani therefore keeps ``name`` off the metadata dataclasses
and lets :func:`module_from_torch` apply it where it is still meaningful.

The version is probed by inspecting the ``from_torch`` signature rather than by
comparing ``physicsnemo.__version__``. Feature detection keeps this correct
across release candidates and backports, where a version comparison would have to
guess.
"""

import inspect

import physicsnemo

__all__ = [
    "PHYSICSNEMO_REGISTER_IS_OPT_IN",
    "get_model_registry",
    "legacy_registered_name",
    "module_from_torch",
]


# True on PhysicsNeMo >= 2.0, where from_torch gained `name`/`register` and no
# longer registers by default.
PHYSICSNEMO_REGISTER_IS_OPT_IN = "register" in inspect.signature(physicsnemo.Module.from_torch).parameters


def get_model_registry():
    r"""
    Return the PhysicsNeMo ``ModelRegistry`` class from wherever it lives.

    The registry moved from ``physicsnemo.registry`` (1.x) to
    ``physicsnemo.core`` (2.x), and the old package was removed rather than
    aliased, so importing it directly pins a caller to one major version.

    Returns
    -------
    type
        The ``ModelRegistry`` class.
    """
    try:
        # PhysicsNeMo >= 2.0
        from physicsnemo.core import ModelRegistry
    except ImportError:
        # PhysicsNeMo 1.x
        from physicsnemo.registry import ModelRegistry

    return ModelRegistry


def legacy_registered_name(torch_model_class):
    r"""
    Return the class name PhysicsNeMo 1.x would have generated.

    Reproducing the 1.x name on 2.x is what keeps checkpoints portable in both
    directions: the name is recorded in the checkpoint metadata and resolved
    again on load, so it has to agree across versions.

    Parameters
    ----------
    torch_model_class : type
        The PyTorch model class being wrapped.

    Returns
    -------
    str
        The wrapped class name with a ``PhysicsNeMoModel`` suffix, e.g.
        ``"SphericalFourierNeuralOperatorNetPhysicsNeMoModel"``.
    """
    return f"{torch_model_class.__name__}PhysicsNeMoModel"


def module_from_torch(torch_model_class, meta, name=None, registered_name=None):
    r"""
    Wrap a PyTorch model as a PhysicsNeMo module, identically on 1.x and 2.x.

    Produces a class with the same ``__name__`` and the same ``ModelRegistry``
    entry under either major version, so callers -- and any checkpoint that
    records the class name -- cannot tell which one is installed.

    Parameters
    ----------
    torch_model_class : type
        The PyTorch model class to wrap.
    meta : physicsnemo.ModelMetaData
        Metadata instance for the model. Must not set ``name``; pass it via the
        ``name`` argument instead, so 2.x does not warn about a deprecated field.
    name : str, optional
        Friendly model name, e.g. ``"SFNO"``. Applied to ``meta.name`` on 1.x,
        where it drives the logger label and the default ``.mdlus`` filename. On
        2.x it is also assigned, for makani-side introspection, but PhysicsNeMo
        itself ignores it. Assigning after construction rather than passing it to
        the dataclass is deliberate: the deprecation warning fires in
        ``__post_init__`` only.
    registered_name : str, optional
        Name to register the class under. Defaults to
        :func:`legacy_registered_name`, which matches 1.x.

    Returns
    -------
    type
        A ``physicsnemo.Module`` subclass wrapping ``torch_model_class``.
    """
    if registered_name is None:
        registered_name = legacy_registered_name(torch_model_class)

    if name is not None:
        meta.name = name

    if PHYSICSNEMO_REGISTER_IS_OPT_IN:
        return physicsnemo.Module.from_torch(
            torch_model_class,
            meta,
            name=registered_name,
            register=True,
        )

    # PhysicsNeMo 1.x: from_torch registers unconditionally, under exactly the
    # name legacy_registered_name() reproduces, so there is nothing to pass.
    return physicsnemo.Module.from_torch(torch_model_class, meta)
