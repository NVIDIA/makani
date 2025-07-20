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

import os
import time
import unittest
from parameterized import parameterized
import tempfile

import numpy as np

import torch
from torch import nn

from makani.models.common import MLP
from makani.utils.driver import Driver
from makani.utils.checkpoint_helpers import get_latest_checkpoint_version

from testutils import get_default_parameters


class TestSaveRestore(unittest.TestCase):

    def setUp(self):
        self.params = get_default_parameters()

        self.params.history_normalization_mode = "none"

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = 36
        self.params.img_shape_y = 72
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_local_offset_x = 0
        self.params.img_local_offset_y = 0

        # also set the batch size for testing
        self.params.batch_size = 4

    def test_get_latest_checkpoint_version(self):

        def create_empty(filename):
            with open(filename, 'w') as fp:
                pass

        with tempfile.TemporaryDirectory() as tempdir:
            create_empty(os.path.join(tempdir, "checkpoint_mp1_v2.tar"))
            time.sleep(3)
            create_empty(os.path.join(tempdir, "checkpoint_mp0_v2.tar"))
            time.sleep(3)
            create_empty(os.path.join(tempdir, "checkpoint_mp1_v0.tar"))
            time.sleep(3)
            create_empty(os.path.join(tempdir, "checkpoint_mp0_v0.tar"))
            time.sleep(3)
            create_empty(os.path.join(tempdir, "checkpoint_mp1_v1.tar"))
            time.sleep(3)
            create_empty(os.path.join(tempdir, "checkpoint_mp0_v1.tar"))

            version = get_latest_checkpoint_version(os.path.join(tempdir, "checkpoint_mp0_v{checkpoint_version}.tar"))

        self.assertTrue(version == 1)

    def test_get_latest_checkpoint_version_default(self):
        version = get_latest_checkpoint_version("checkpoint_mp0.tar")
        self.assertTrue(version == 0)


    @parameterized.expand(["legacy", "flexible"])
    def test_save_restore(self, checkpoint_mode):
        """
        Tests initialization of all the models and the forward and backward pass
        """

        model = MLP(self.params.N_in_channels,
                    hidden_features=2*self.params.N_in_channels,
                    out_features=self.params.N_out_channels,
                    act_layer=nn.GELU,
                    output_bias=True,
                    input_format="nchw",
                    drop_rate=0.0,
        )

        inp_shape = (self.params.batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y)

        # prepare some dummy data
        inp = torch.randn(*inp_shape)
        inp.requires_grad = True

        # do forward pass:
        out_before = model(inp).detach().cpu().numpy()

        with tempfile.TemporaryDirectory() as tempdir:
            # checkpoint path
            checkpoint_path = os.path.join(tempdir, "ckpt.tar")

            # store checkpoint
            Driver.save_checkpoint(
                checkpoint_path,
                model=model,
                checkpoint_mode=checkpoint_mode
            )

            # scramble model
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()

            # reload checkpoint
            Driver.restore_from_checkpoint(
                checkpoint_path,
                model=model,
                loss=None,
                optimizer=None,
                scheduler=None,
                counters=None,
                checkpoint_mode=checkpoint_mode
            )

        # do forward pass
        out_after = model(inp).detach().cpu().numpy()

        # compare
        self.assertTrue(np.allclose(out_before, out_after, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
