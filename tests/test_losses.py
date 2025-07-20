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
import tempfile
from typing import Optional
from parameterized import parameterized

import unittest
import numpy as np
import torch

from makani.models import model_registry
from makani.utils import LossHandler
from makani.utils.losses import EnsembleCRPSLoss

from testutils import get_default_parameters

from properscoring import crps_ensemble, crps_gaussian

_loss_params = [
    ([{"type": "l1"}], False),
    ([{"type": "relative l1"}], False),
    ([{"type": "squared l2"}], False),
    ([{"type": "geometric l2", "channel_weights": "constant"}], True),
    ([{"type": "geometric l2", "channel_weights": "constant"}, {"type": "l2", "channel_weights": "auto"}], True),
    ([{"type": "geometric h1", "channel_weights": "constant"}], True),
    ([{"type": "geometric l2", "channel_weights": "constant", "temp_diff_normalization": True}], True),
    ([{"type": "geometric l2", "channel_weights": "constant"}, {"type": "geometric h1", "channel_weights": "constant"}], True),
    ([{"type": "geometric l2", "channel_weights": "constant"}, {"type": "geometric l1", "channel_weights": "constant"}], True),
]

_loss_weighted_params = [
    ([{"type": "l1"}], False),
    ([{"type": "relative l1"}], False),
    ([{"type": "squared l2"}], False),
    ([{"type": "geometric l2", "channel_weights": "constant"}], False),
    ([{"type": "geometric l2", "channel_weights": "constant"}, {"type": "l2", "channel_weights": "auto"}], False),
    ([{"type": "geometric l2", "channel_weights": "constant", "temp_diff_normalization": True}], False),
    ([{"type": "geometric l2", "channel_weights": "constant"}, {"type": "geometric l1", "channel_weights": "constant"}], False),
]


class TestLosses(unittest.TestCase):

    @classmethod
    def setUpClass(cls, path: Optional[str] = "/tmp"):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.img_shape_x = 32
        cls.img_shape_y = 64

        cls.tmpdir = tempfile.TemporaryDirectory(dir=path)
        tmp_path = cls.tmpdir.name

        params = get_default_parameters()

        cls.time_diff_stds_path = os.path.join(tmp_path, "time_diff_stds.npy")
        np.save(cls.time_diff_stds_path, np.ones((1, params.N_out_channels, 1, 1), dtype=np.float64))


    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


    def setUp(self):

        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        self.params = get_default_parameters()

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = self.img_shape_x
        self.params.img_shape_y = self.img_shape_y
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_local_offset_x = self.params.img_crop_offset_x = 0
        self.params.img_local_offset_y = self.params.img_crop_offset_y = 0

        # also set the batch size for testing
        self.params.batch_size = 4

        # set paths
        self.params.time_diff_stds_path = self.time_diff_stds_path


    @parameterized.expand(_loss_params)
    def test_loss(self, losses, uncertainty_weighting=False):
        """
        Tests initialization of loss, as well as the forward and backward pass
        """

        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting

        # test initialization of loss object
        loss_obj = LossHandler(self.params)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        inp = torch.randn(*shape)
        inp.requires_grad = True
        tar = torch.randn(*shape)
        tar.requires_grad = True

        # forward pass and check shapes
        out = loss_obj(tar, inp)
        self.assertEqual(torch.numel(out), 1)
        self.assertTrue(out.item() >= 0.0)

        # backward pass and check gradients are not None
        out.backward()


    @parameterized.expand(_loss_params)
    def test_loss_batchsize_independence(self, losses, uncertainty_weighting=False):
        """
        Tests if losses are independent on batch size, in the sense that proper averaging over batch size
        is performed
        """

        self.params.losses = losses
        # not supported for bs independence:
        self.params.uncertainty_weighting = False

        # test initialization of loss object
        loss_obj = LossHandler(self.params)
        
        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        inp = torch.randn(*shape)
        tar = torch.randn(*shape)
        out = loss_obj(tar, inp)

        inp2 = torch.cat([inp, inp], dim=0)
        tar2 = torch.cat([tar, tar], dim=0)
        out2 = loss_obj(tar2, inp2)

        self.assertTrue(torch.allclose(out, out2))
        

    @parameterized.expand(_loss_weighted_params)
    def test_loss_weighted(self, losses, uncertainty_weighting=False):
        """
        Tests initialization of loss, as well as the forward and backward pass
        """
        
        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting

	# test initialization of loss object
        loss_obj = LossHandler(self.params)

        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        inp = torch.randn(*shape).clone()
        inp.requires_grad = True
        tar = torch.randn(*shape).clone()
        tar.requires_grad = True
        wgt = torch.ones_like(tar)

        # forward pass and check shapes
        out = loss_obj(tar, inp)
        self.assertEqual(torch.numel(out), 1)
        self.assertTrue(out.item() >= 0.0)

        # compute weighted loss
        out_weighted = loss_obj(tar, inp, wgt)
        
        self.assertTrue(torch.allclose(out, out_weighted))

    @parameterized.expand(_loss_weighted_params)
    def test_loss_multistep(self, losses, uncertainty_weighting=False):
        """
        Tests initialization of loss, as well as the forward and backward pass
        """

        self.params.n_future = 2
        self.params.losses = losses
        self.params.uncertainty_weighting = uncertainty_weighting

        # test initialization of loss object
        loss_obj = LossHandler(self.params)

        shape = (self.params.batch_size, (self.params.n_future + 1) * self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        inp = torch.randn(*shape).clone()
        inp.requires_grad = True
        tar = torch.randn(*shape).clone()
        tar.requires_grad = True
        wgt = torch.ones_like(tar)

	# forward pass and check shapes
        out = loss_obj(tar, inp)
        self.assertEqual(torch.numel(out), 1)
        self.assertTrue(out.item() >= 0.0)

        # compute weighted loss
        out_weighted = loss_obj(tar, inp, wgt)

        self.assertTrue(torch.allclose(out, out_weighted))        

    def test_running_stats(self):
        """
        Tests computation of the running stats
        """
    
        self.params.losses = [{"type": "l2"}]
    
        # test initialization of loss object
        loss_obj = LossHandler(self.params, track_running_stats=True)
        loss_obj.train()
    
        shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
    
        # this needs to be sufficiently large to mitigarte the bias due to the initialization of the running stats
        num_samples = 100
        for i in range(num_samples):
    
            inp = i * torch.ones(*shape)
            inp.requires_grad = True
            tar = torch.zeros(*shape)
            tar.requires_grad = True
    
            # forward pass and check shapes
            out = loss_obj(tar, inp)
    
        # generate simulated dataset
        data = torch.arange(num_samples).float().reshape(1, 1, -1).repeat(self.params.batch_size, self.params.N_out_channels, 1)
        expected_var, expected_mean = torch.var_mean(data, correction=0, dim=(0, -1))
    
        var, mean = loss_obj.get_running_stats()
    
        self.assertTrue(torch.allclose(mean, expected_mean))
        self.assertTrue(torch.allclose(var, expected_var))

    def test_ensemble_crps(self):
        crps_func = EnsembleCRPSLoss(
            img_shape=(self.params.img_shape_x, self.params.img_shape_y),
            crop_shape=(self.params.img_shape_x, self.params.img_shape_y),
            crop_offset=(0, 0),
            channel_names=self.params.channel_names,
            grid_type=self.params.model_grid_type,
            pole_mask=0,
            crps_type="cdf",
            spatial_distributed=False,
            ensemble_distributed=False,
            ensemble_weights=None,
        )
    
        for ensemble_size in [1, 10]:
            with self.subTest(f"{ensemble_size}"):
                # generate input tensor
                inp = torch.empty((self.params.batch_size, ensemble_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y), dtype=torch.float32)
                with torch.no_grad():
                    inp.normal_(1.0, 1.0)
    
                # target tensor
                tar = torch.ones((self.params.batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y), dtype=torch.float32)
    
                # torch result
                result = crps_func(inp, tar).cpu().numpy()
    
                # properscoring result
                tar_arr = tar.cpu().numpy()
                inp_arr = inp.cpu().numpy()
    
                # I think this is a bug with the axis index in properscoring
                # for the degenerate case:
                if ensemble_size == 1:
                    axis = -1
                    inp_arr = np.squeeze(inp_arr, axis=1)
                else:
                    axis = 1
    
                result_proper = crps_ensemble(tar_arr, inp_arr, weights=None, issorted=False, axis=axis)
                quad_weight_arr = crps_func.quadrature.quad_weight.cpu().numpy()
                result_proper = np.sum(result_proper * quad_weight_arr, axis=(2, 3))
    
                self.assertTrue(np.allclose(result, result_proper, rtol=1e-5, atol=0))

    def test_gauss_crps(self):
    
        # protext against sigma=0
        eps = 1.0e-5
    
        crps_func = EnsembleCRPSLoss(
            img_shape=(self.params.img_shape_x, self.params.img_shape_y),
            crop_shape=(self.params.img_shape_x, self.params.img_shape_y),
            crop_offset=(0, 0),
            channel_names=self.params.channel_names,
            grid_type=self.params.model_grid_type,
            pole_mask=0,
            crps_type="gauss",
            spatial_distributed=False,
            ensemble_distributed=False,
            eps=eps,
        )
    
        for ensemble_size in [1, 10]:
            with self.subTest(f"{ensemble_size}"):
                # generate input tensor
                inp = torch.empty((self.params.batch_size, ensemble_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y), dtype=torch.float32)
                with torch.no_grad():
                    inp.normal_(1.0, 1.0)
    
                # target tensor
                tar = torch.ones((self.params.batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y), dtype=torch.float32)
    
                # torch result
                result = crps_func(inp, tar).cpu().numpy()
    
                # properscoring result
                tar_arr = tar.cpu().numpy()
                inp_arr = inp.cpu().numpy()
    
                # compute mu, sigma, guard against underflows
                mu = np.mean(inp_arr, axis=1)
                sigma = np.maximum(np.sqrt(np.var(inp_arr, axis=1)), eps)
    
                result_proper = crps_gaussian(tar_arr, mu, sigma, grad=False)
                quad_weight_arr = crps_func.quadrature.quad_weight.cpu().numpy()
                result_proper = np.sum(result_proper * quad_weight_arr, axis=(2, 3))
    
                self.assertTrue(np.allclose(result, result_proper, rtol=1e-5, atol=0))


if __name__ == "__main__":
    unittest.main()
