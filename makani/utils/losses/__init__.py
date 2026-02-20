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

from .base_loss import LossType
from .h1_loss import SpectralH1Loss
from .lp_loss import GeometricLpLoss, SpectralLpLoss
from .amse_loss import SpectralAMSELoss
from .hydrostatic_loss import HydrostaticBalanceLoss
from .crps_loss import CRPSLoss, SpectralCRPSLoss, GradientCRPSLoss, VortDivCRPSLoss, KernelScoreLoss
from .energy_score import L2EnergyScoreLoss, SobolevEnergyScoreLoss, SpectralL2EnergyScoreLoss, SpectralCoherenceLoss
from .mmd_loss import GaussianMMDLoss
from .likelihood_loss import EnsembleNLLLoss
from .regularization import DriftRegularization, SpectralRegularization
