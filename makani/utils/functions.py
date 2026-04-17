# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# this computes a relative error compatible with torch.allclose or np.allclose
def relative_error(tensor1, tensor2):
    return torch.sum(torch.abs(tensor1-tensor2)) / torch.sum(torch.abs(tensor2))

# this computes an absolute error compatible with torch.allclose or np.allclose
def absolute_error(tensor1, tensor2):
    return torch.max(torch.abs(tensor1-tensor2))


def expand_ensemble(x: torch.Tensor, ensemble_size: int) -> torch.Tensor:
    """
    Replicate each element of ``x`` along a new ensemble dim, then flatten that dim
    into the batch dim: ``(B, ...) -> (B*E, ...)``.

    Layout is batch-major: the E consecutive entries at positions ``b*E .. b*E+E-1``
    all correspond to the original batch element ``x[b]``. This pairs with a
    symmetric ``pred_flat.reshape(B, E, ...)`` on the model output so a single forward
    on the folded batch produces independent ensemble samples per input.

    ``ensemble_size <= 1`` returns ``x`` unchanged (no allocation).
    """
    if ensemble_size <= 1:
        return x
    return (
        x.unsqueeze(1)
         .repeat_interleave(ensemble_size, dim=1)
         .reshape(x.shape[0] * ensemble_size, *x.shape[1:])
    )
