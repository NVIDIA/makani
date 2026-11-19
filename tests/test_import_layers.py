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

"""The packages of makani import each other in one direction only, from a higher layer to a lower one, so
that no import cycles can form between them; and no module uses a wildcard import."""

import ast
import os
import unittest

import makani

# lowest first; a package may import from its own layer and the layers below it
LAYERS = [
    ("makani.distributed",),
    ("makani.utils", "makani.third_party"),
    ("makani.data",),
    ("makani.models",),
    ("makani.losses",),
    ("makani.metrics",),
    ("makani.driver",),
    ("makani.training", "makani.inference"),
    ("makani",),  # the package itself, its entry scripts and the benchmark
]

# the entry point that model packages written before makani 0.4 record, which Earth2MIP imports by name
ALLOWED = {("makani.models.model_package", "makani.driver.model_package")}


def _layer(module):
    best = None
    for index, prefixes in enumerate(LAYERS):
        for prefix in prefixes:
            if (module == prefix or module.startswith(prefix + ".")) and (best is None or len(prefix) > best[1]):
                best = (index, len(prefix))
    return None if best is None else best[0]


def _imports(path, module):
    """The makani modules a file imports, with their line numbers; imports for type checking only are left out."""
    tree = ast.parse(open(path).read(), filename=path)
    type_checking = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            type_checking.update(id(child) for child in ast.walk(node))

    package = module.split(".") if path.endswith("__init__.py") else module.split(".")[:-1]
    for node in ast.walk(tree):
        if id(node) in type_checking:
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            yield node.lineno, target, [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, []


def _modules():
    root = os.path.dirname(os.path.dirname(os.path.abspath(makani.__file__)))
    for directory, _, files in os.walk(os.path.join(root, "makani")):
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(directory, name)
                module = os.path.relpath(path, root)[: -len(".py")].replace(os.sep, ".").removesuffix(".__init__")
                yield path, module


class TestImportLayers(unittest.TestCase):

    def test_no_upward_imports(self):
        violations = []
        for path, module in _modules():
            source = _layer(module)
            for lineno, target, _ in _imports(path, module):
                if target == "makani" or not target.startswith("makani."):
                    continue
                if (module, target) in ALLOWED:
                    continue
                if _layer(target) > source:
                    violations.append(f"{module}:{lineno} imports {target}")
        self.assertEqual(violations, [], "imports from a higher layer:\n" + "\n".join(violations))

    def test_no_wildcard_imports(self):
        wildcards = [
            f"{module}:{lineno} from {target} import *"
            for path, module in _modules()
            for lineno, target, names in _imports(path, module)
            if "*" in names
        ]
        self.assertEqual(wildcards, [], "wildcard imports:\n" + "\n".join(wildcards))


if __name__ == "__main__":
    unittest.main()
