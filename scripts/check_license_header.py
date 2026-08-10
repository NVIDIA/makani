#!/usr/bin/env python3

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

"""Check that Python files carry an SPDX license header.

Unlike torch-harmonics, makani is not single-licensed: the package is Apache-2.0
but vendored code under makani/third_party/ keeps its upstream license (climt is
BSD-3-Clause). So this checks for the presence of both SPDX tags and that the
identifier is one we expect, rather than pinning one exact license.
"""

import sys

COPYRIGHT_MARKER = "SPDX-FileCopyrightText:"
LICENSE_MARKER = "SPDX-License-Identifier:"

# Licenses in use across the repository. Add here when vendoring code under a new
# one, so that a typo'd or unexpected identifier is still caught.
ALLOWED_LICENSES = {
    "Apache-2.0",
    "BSD-3-Clause",
}

# Number of lines to scan at the top of each file
HEADER_SCAN_LINES = 10


def check_file(path):
    """Return (ok, reason). ``reason`` is None when the header is fine."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = "".join(f.readline() for _ in range(HEADER_SCAN_LINES))
    except (OSError, UnicodeDecodeError):
        return True, None  # skip files we cannot read

    if not head.strip():
        return True, None  # skip empty files

    if COPYRIGHT_MARKER not in head:
        return False, f"missing '{COPYRIGHT_MARKER}'"

    if LICENSE_MARKER not in head:
        return False, f"missing '{LICENSE_MARKER}'"

    # pull the identifier off the line carrying the marker
    for line in head.splitlines():
        if LICENSE_MARKER in line:
            identifier = line.split(LICENSE_MARKER, 1)[1].strip()
            if identifier not in ALLOWED_LICENSES:
                allowed = ", ".join(sorted(ALLOWED_LICENSES))
                return False, f"unexpected license '{identifier}' (allowed: {allowed})"
            break

    return True, None


def main():
    failed = []
    for path in sys.argv[1:]:
        if not path.endswith(".py"):
            continue
        ok, reason = check_file(path)
        if not ok:
            failed.append((path, reason))

    if failed:
        print("SPDX license header problems:")
        for path, reason in failed:
            print(f"  {path}: {reason}")
        print()
        print("Every .py file must carry these lines near the top:")
        print(f"  # {COPYRIGHT_MARKER} Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.")
        print(f"  # {LICENSE_MARKER} Apache-2.0")
        sys.exit(1)


if __name__ == "__main__":
    main()
