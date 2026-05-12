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


import sys
import os
import tempfile
import unittest
from parameterized import parameterized

import torch

from makani.utils import comm
from makani.utils import driver
from makani.utils import checkpoint_helpers
from makani.utils import LossHandler
from makani.models import model_registry
from makani.mpu.mappings import init_gradient_reduction_hooks
from makani.mpu.mappings import reduce_from_parallel_region

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import (
    get_default_parameters, set_image_shape,
    _split_helper, _gather_helper,
    _init_grid_module, _copy_grid_state,
)
from ..testutils import disable_tf32, set_seed, compare_tensors


# Comm groups to override to size 1 when constructing the serial reference
# model alongside the distributed one. Covers every model-parallel group plus
# the "model" / "spatial" / "matmul" parent groups that model code branches on.
_SERIAL_OVERRIDE = dict(
    model=1, spatial=1, matmul=1,
    h=1, w=1, fin=1, fout=1,
)


# Module-level MPI communicator (duplicated from MPI.COMM_WORLD once per
# process, freed in tearDownModule). Both comm and MPI are initialised once
# per process so test cases can construct serial and distributed models
# side-by-side without re-initialising any communication state.
_MPI_COMM = None


def setUpModule():
    """Initialise MPI + comm groups once per Python process.

    The distributed model tests used to tear down and re-create comm in every
    test method so that a serial reference model could be constructed before
    comm was initialised. That pattern failed when running multiple tests in
    the same process because comm/NCCL state did not survive teardown +
    re-init. Instead we init both communicators exactly once at module load,
    and tests construct their serial reference model inside a
    ``with comm.override_sizes(**_SERIAL_OVERRIDE):`` block — comm itself
    keeps its real sizes for the distributed model.
    """
    global _MPI_COMM
    from mpi4py import MPI
    _MPI_COMM = MPI.COMM_WORLD.Dup()
    _init_grid_module()


def tearDownModule():
    comm.cleanup()
    if _MPI_COMM is not None:
        # mpi_comm is an mpi4py Intracomm (returned by MPI.COMM_WORLD.Dup());
        # its disposal API is Free(), called explicitly rather than relying on
        # interpreter shutdown.
        _MPI_COMM.Free()


class TestDistributedModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Shadow module-level state (MPI handle + grid state) onto the class
        # so tests can keep using ``self.mpi_comm`` / ``self.h_group`` etc.
        cls.mpi_comm = _MPI_COMM
        cls.mpi_comm_rank = _MPI_COMM.Get_rank()
        cls.mpi_comm_size = _MPI_COMM.Get_size()
        _copy_grid_state(cls)

        set_seed(333)

        return

    def setUp(self):

        disable_tf32()

        self.params = get_default_parameters()

        self.params.history_normalization_mode = "none"

        # set image shape, local/crop shapes, offsets, and resampled shapes
        # consistently — see set_image_shape for the rationale.
        set_image_shape(self.params, h=36, w=72)

        # also set the batch size for testing
        self.params.batch_size = 4


    def _split_helper(self, tensor, hdim=None, wdim=None):
        tensor_local = _split_helper(tensor, dim=hdim, group=self.h_group)
        tensor_local = _split_helper(tensor_local, dim=wdim, group=self.w_group)

        return tensor_local


    def _gather_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_gather = _gather_helper(tensor, dim=hdim, group=self.h_group)
        tensor_gather = _gather_helper(tensor_gather, dim=wdim, group=self.w_group)

        return tensor_gather

    @parameterized.expand(
        [
            "SFNO",
            "SNO",
            "FCN3",
        ],
        skip_on_empty=True,
    )
    def test_distributed_model_checkpoint_restore(self, nettype, verbose=False):
        """
        Tests initialization of all the models and the forward and backward pass
        """
        self.params.nettype = nettype
        if nettype == "DebugNet":
            return

        # fix seed
        set_seed(333)

        # create temporary dir
        tmp_path = None
        if self.mpi_comm_rank == 0:
            tmpdir = tempfile.TemporaryDirectory(dir="/tmp")
            tmp_path = tmpdir.name
        tmp_path = self.mpi_comm.bcast(tmp_path, root=0)

        # Construct the serial reference model with every model-parallel comm
        # group overridden to size 1 — the comm groups themselves keep their
        # real sizes for the distributed model below.
        with comm.override_sizes(**_SERIAL_OVERRIDE):
            model = model_registry.get_model(self.params, multistep=False).to(self.device)

            # get state dict
            state_dict_full = checkpoint_helpers.gather_model_state_dict(model, grads=False)

            if self.mpi_comm_rank == 0:
                driver.Driver.save_checkpoint(checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
                                              model=model,
                                              checkpoint_mode="flexible")

        self.mpi_comm.Barrier()

        model_dist = model_registry.get_model(self.params, multistep=False).to(self.device)

        # load checkpoint
        driver.Driver.restore_from_checkpoint(checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
                                              model=model_dist,
                                              loss=None,
                                              optimizer=None,
                                              scheduler=None,
                                              counters=None,
                                              checkpoint_mode="flexible",
                                              strict=False)

        # compare parameters
        state_dict_gather_full = checkpoint_helpers.gather_model_state_dict(model_dist, grads=False)
        for key in state_dict_full.keys():
            with self.subTest(desc=f"parameter {key}"):
                param_full = state_dict_full[key].cpu()
                param_gather_full = state_dict_gather_full[key]
                self.assertTrue(compare_tensors(f"parameter {key}", param_full, param_gather_full, verbose=verbose))

        self.mpi_comm.Barrier()


    @parameterized.expand(
        [
            # fp32 accumulation order differs between serial and distributed
            # paths — observed worst-case relative drift around 4e-3 on a
            # single small-magnitude weight-grad element for SFNO with h=w=2.
            # Use a looser tol that still catches a real factor-of-N regression
            # (factor-of-N errors would show up as ~25%-100% relative diff).
            ("SFNO", 1e-2),
            ("SNO", 1e-4),
            ("FCN3", 1e-4),
        ],
        skip_on_empty=True,
    )
    def test_distributed_model_fwd_bwd(self, nettype, tol, verbose=True):
        """
        Tests forward backward pass of distributed model vs serial model
        """

        self.params.nettype = nettype
        if nettype == "DebugNet":
            return

        # fix seed
        set_seed(333)

        # create temporary dir
        tmp_path = None
        if self.mpi_comm_rank == 0:
            tmpdir = tempfile.TemporaryDirectory(dir="/tmp")
            tmp_path = tmpdir.name
        tmp_path = self.mpi_comm.bcast(tmp_path, root=0)

        multistep = self.params.n_future > 0
        # Construct + run the serial reference model under the size-1 override
        # so its internal branches all pick the non-distributed path even
        # though the comm groups are live for the distributed model below.
        with comm.override_sizes(**_SERIAL_OVERRIDE):
            model = model_registry.get_model(self.params, multistep=multistep).to(self.device)

            inp_shape = (self.params.batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y)

            # prepare some dummy data
            inp_full = torch.randn(*inp_shape, dtype=torch.float32, device=self.device)
            inp_full.requires_grad = True

            # forward pass and save
            out_full = model(inp_full).clone()
            loss_full = torch.sum(out_full)

            # perform backward pass
            loss_full.backward()
            igrad_full = inp_full.grad.clone()

            # store output:
            if self.mpi_comm_rank == 0:
                torch.save(out_full, os.path.join(tmp_path, "out_full.pt"))
                torch.save(igrad_full, os.path.join(tmp_path, "igrad_full.pt"))
                driver.Driver.save_checkpoint(checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
                                              model=model,
                                              checkpoint_mode="flexible")
            self.mpi_comm.Barrier()

            # get also grad output
            state_dict_full = checkpoint_helpers.gather_model_state_dict(model, grads=True)

            # delete local model
            del model

        # create model, this times distributed
        model_dist = model_registry.get_model(self.params, multistep=multistep).to(self.device)

        # save reduction hooks
        model_dist = init_gradient_reduction_hooks(
            model_dist,
            device=self.device,
            reduction_buffer_count=1,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            verbose=False,
        )

        # load checkpoint
        driver.Driver.restore_from_checkpoint(checkpoint_path=os.path.join(tmp_path, "checkpoint.pt"),
                                              model=model_dist,
                                              loss=None,
                                              optimizer=None,
                                              scheduler=None,
                                              counters=None,
                                              checkpoint_mode="flexible",
                                              strict=False)

        # split input
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = model_dist(inp_local)
        loss_dist = reduce_from_parallel_region(torch.sum(out_local), "model")
        loss_dist.backward()
        igrad_local = inp_local.grad.clone()

        # get weights and wgrads
        state_dict_gather_full = checkpoint_helpers.gather_model_state_dict(model_dist, grads=True)

        #############################################################
        # evaluate FWD pass
        #############################################################
        # output
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose))

        # loss
        with self.subTest(desc="loss"):
            self.assertTrue(compare_tensors("loss", loss_dist, loss_full, tol, tol, verbose=verbose))

        #############################################################
        # evaluate BWD pass
        #############################################################
        # dgrad
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose))

        # wgrads
        for key in state_dict_full.keys():
            if key.endswith(".grad"):
                with self.subTest(desc=f"weight gradient {key}"):
                    wgrad_full = state_dict_full[key]
                    wgrad_gather_full = state_dict_gather_full["module." + key]
                    self.assertTrue(compare_tensors(f"weight gradient {key}", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose))


    @parameterized.expand(
        [
            ("SFNO", 1e-6, 1e-6),
            ("SNO", 1e-6, 1e-6),
            ("FCN3", 1e-6, 1e-6),
        ],
        skip_on_empty=True,
    )
    def test_distributed_model_gradient_accumulation(self, nettype, atol, rtol, verbose=False):
        """
        Tests gradient accumulation with distributed models
        """

        self.params.nettype = nettype
        if nettype == "DebugNet":
            return

        # fix seed
        set_seed(333)

        # create model, this times distributed
        model_dist = model_registry.get_model(self.params, multistep=False).to(self.device)

        # save reduction hooks
        model_dist = init_gradient_reduction_hooks(
            model_dist,
            device=self.device,
            reduction_buffer_count=1,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            verbose=False,
        )

        # get loss object
        self.params.losses = [{"type": "l2", "channel_weights": "constant"}]
        loss_obj = LossHandler(self.params).to(self.device)

        # input shape
        batch_size = self.params.batch_size * 2
        inp_shape = (batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y)

        # prepare some dummy data and split across ranks
        inp_full = torch.randn(*inp_shape, dtype=torch.float32, device=self.device)
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        tar_full = torch.randn_like(inp_full)
        tar_local = self._split_helper(tar_full, hdim=-2, wdim=-1)
        tar_local.requires_grad = False

        # perform a single forward:
        model_dist.zero_grad(set_to_none=True)
        out_single_local = model_dist(inp_local)
        loss = loss_obj(out_single_local, tar_local)
        # backward pass
        loss.backward()

        # store the gradients
        state_dict_single_step = checkpoint_helpers.gather_model_state_dict(model_dist, grads=True)

        # split input
        inp_local_split = torch.split(inp_local, self.params.batch_size, dim=0)
        tar_local_split = torch.split(tar_local, self.params.batch_size, dim=0)

        # now perform multiple steps with gradient accumulation
        model_dist.zero_grad(set_to_none=True)
        inp_local_tmp = inp_local_split[0].detach().clone()
        inp_local_tmp.requires_grad = True
        tar_local_tmp = tar_local_split[0].detach().clone()

        # step 1
        with model_dist.no_sync():
            out_double_local = model_dist(inp_local_tmp)
            loss = loss_obj(out_double_local, tar_local_tmp) / 2.
        loss.backward()

        inp_local_tmp = inp_local_split[1].detach().clone()
        inp_local_tmp.requires_grad = True
        tar_local_tmp = tar_local_split[1].detach().clone()

        # step 2
        out = model_dist(inp_local_tmp)
        loss = loss_obj(out, tar_local_tmp) / 2.
        loss.backward()
        out_double_local = torch.cat([out_double_local, out], dim=0)

        # store the gradients
        state_dict_double_step = checkpoint_helpers.gather_model_state_dict(model_dist, grads=True)

        #############################################################
        # evaluate FWD pass
        #############################################################
        # output
        with self.subTest(desc="output"):
            out_single_gather = self._gather_helper(out_single_local, hdim=-2, wdim=-1)
            out_double_gather = self._gather_helper(out_double_local, hdim=-2, wdim=-1)
            self.assertTrue(compare_tensors("output", out_double_gather, out_single_gather, atol, rtol, verbose=verbose))

        #############################################################
        # evaluate BWD pass
        #############################################################
        # wgrads
        for key in state_dict_single_step.keys():
            if key.endswith(".grad"):
                with self.subTest(desc=f"weight gradient {key}"):
                    wgrad_single = state_dict_single_step[key]
                    wgrad_double = state_dict_double_step[key]
                    self.assertTrue(compare_tensors(f"weight gradient {key}", wgrad_double, wgrad_single, atol, rtol, verbose=verbose))



if __name__ == '__main__':
    unittest.main()
