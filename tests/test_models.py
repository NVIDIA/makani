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

import unittest
from parameterized import parameterized

import torch

from makani.models import model_registry
from makani.utils import checkpoint_helpers
from makani.utils import functions as fn
from makani.utils import LossHandler

from testutils import get_default_parameters

class TestModels(unittest.TestCase):

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
        self.params.img_crop_offset_x = 0
        self.params.img_crop_offset_y = 0

        # also set the batch size for testing
        self.params.batch_size = 4

        # set device and seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        return

    @parameterized.expand(model_registry.list_models(), skip_on_empty=True)
    def test_model(self, nettype):
        """
        Tests initialization of all the models and the forward and backward pass
        """
        self.params.nettype = nettype
        if nettype == "DebugNet":
            return

        multistep = self.params.n_future > 0
        model = model_registry.get_model(self.params, multistep=multistep).to(self.device)

        inp_shape = (self.params.batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y)
        out_shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)

        # prepare some dummy data
        inp = torch.randn(*inp_shape, dtype=torch.float32, device=self.device)
        inp.requires_grad = True

        # forward pass and check shapes
        out = model(inp)
        self.assertEqual(out.shape, out_shape)

        # backward pass and check gradients are not None
        out = torch.sum(out)
        out.backward()
        self.assertTrue(inp.grad is not None)
        self.assertEqual(inp.grad.shape, inp_shape)


    @parameterized.expand(
        [
            ('AFNO', 1e-7, 5e-3),
            # test on the CPU vor AFNOv2 yields lower
            # agreement than on GPU, adjusting this for CI
            ('AFNOv2', 1e-7, 5e-3),
            ('FNO', 1e-7, 1e-5),
            ('ViT', 1e-7, 1e-5),
            ("SFNO", 1e-7, 1e-5),
            ("SNO", 1e-7, 1e-5),
            ("FCN3", 1e-7, 1e-5),
            ("Pangu", 1e-7, 1e-5)
        ],
        skip_on_empty=True,
    )
    def test_gradient_accumulation(self, nettype, atol, rtol, verbose=False):
        """
        Tests initialization of all the models and the forward and backward pass
        """
        self.params.nettype = nettype
        if nettype == "DebugNet":
            return

        # get loss object
        self.params.losses = [{"type": "geometric l2", "channel_weights": "constant"}]
        loss_obj = LossHandler(self.params).to(self.device)

        # get model
        model = model_registry.get_model(self.params, multistep=False).to(self.device)

        batch_size = self.params.batch_size * 2
        inp_shape = (batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y)

        # prepare some dummy data
        inp = torch.randn(*inp_shape, dtype=torch.float32, device=self.device)
        tar = torch.randn_like(inp, dtype=torch.float32, device=self.device)
        inp.requires_grad = True
        tar.requires_grad = False

        # forward pass
        model.zero_grad(set_to_none=True)

        out_single = model(inp)
        loss = loss_obj(out_single, tar)
        # backward pass
        loss.backward()
        # igrad
        igrad_single = inp.grad.clone()

        # store the gradients
        state_dict_single_step = checkpoint_helpers.gather_model_state_dict(model, grads=True)

        # split input
        inp_split = torch.split(inp, self.params.batch_size, dim=0)
        tar_split = torch.split(tar, self.params.batch_size, dim=0)

        # forward pass
        model.zero_grad(set_to_none=True)
        inp_tmp = inp_split[0].detach().clone()
        inp_tmp.requires_grad = True
        tar_tmp = tar_split[0].detach().clone()

        # step 1
        out_double = model(inp_tmp)
        loss = loss_obj(out_double, tar_tmp) / 2.
        loss.backward()
        igrad_double = inp_tmp.grad.clone()

        inp_tmp = inp_split[1].detach().clone()
        inp_tmp.requires_grad =	True
        tar_tmp = tar_split[1].detach().clone()

        # step 2
        out = model(inp_tmp)
        loss = loss_obj(out, tar_tmp) / 2.
        loss.backward()
        out_double = torch.cat([out_double, out], dim=0)
        igrad_double = torch.cat([igrad_double, inp_tmp.grad.clone()], dim=0)

        # store gradients
        state_dict_double_step = checkpoint_helpers.gather_model_state_dict(model, grads=True)

        #############################################################
        # evaluate FWD pass
        #############################################################
        with torch.no_grad():
            err = fn.relative_error(out_double, out_single)
            if verbose:
                print(f"final relative error of output: {err.item()}")
        self.assertTrue(err.item() <= rtol)

        #############################################################
        # evaluate BWD pass
        #############################################################
        # igrads
        with torch.no_grad():
            err = fn.relative_error(igrad_double, igrad_single)
            if verbose:
                print(f"final relative error of input grads: {err.item()}")
        self.assertTrue(err.item() <= rtol)
        # wgrads
        with torch.no_grad():
            good = True
            errs = []
            for key in state_dict_single_step.keys():
                if key.endswith(".grad"):
                    wgrad_single = state_dict_single_step[key]
                    wgrad_double = state_dict_double_step[key]
                    if (wgrad_single is None) and (wgrad_double is None):
                        continue
                    elif (wgrad_single is None) and (wgrad_double is not None):
                        if verbose:
                            print(f"weight gradient {key} is None in single but not None in double step model")
                        good = False
                    elif (wgrad_single is not None) and (wgrad_double is None):
                        if verbose:
                            print(f"weight gradient {key} is not None in single but None in double step model")
                        good = False
                    else:
                        err = fn.relative_error(wgrad_double, wgrad_single)
                        if err > rtol:
                            # in some cases, the gradient itself can be small, check absolute tolerance then
                            aerr = fn.absolute_error(wgrad_double, wgrad_single)
                            if aerr < atol:
                                continue
                            if verbose:
                                print(f"final relative error of weight gradient {key}: {err.item()}, absolute error: {aerr.item()}")
                            good = False
                        errs.append(err)
            merr = torch.stack(errs, dim=0).mean().item()
            if verbose:
                print(f"final relative average error of weight gradients: {merr}")
        self.assertTrue(good)



if __name__ == "__main__":
    unittest.main()
