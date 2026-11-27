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
import math
import unittest
from unittest import mock
from collections import OrderedDict
from parameterized import parameterized

import torch
import torch.nn as nn
import torch.distributed as dist

import torch_harmonics as th
import torch_harmonics.distributed as thd

from makani.distributed import comm
from makani.utils.grids import make_grid, sht_grid
from makani.driver import checkpoint as checkpoint_helpers

from makani.utils.precision import AutocastManager, _fp8_global_state_manager
from makani.utils.te_helpers import TE_AVAILABLE

# mlps
from makani.models.common.mlp import MLP
from makani.models.common.distributed.mlp import DistributedMLP
from makani.models.common.pointwise_conv import fp8_layers

# layer norm imports
from makani.models.common.layer_norm import GeometricInstanceNormS2
from makani.models.common.pos_embedding import SpectralPositionEmbedding
from makani.models.common.distributed.layer_norm import DistributedGeometricInstanceNormS2, DistributedInstanceNorm2d

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .distributed_helpers import _init_grid, _split_helper, _gather_helper, reduce_success, teardown_comms
from ..testutils import disable_tf32, set_seed, compare_tensors


def _fp8_available() -> bool:
    """Whether transformer_engine is installed and the GPU computes in FP8."""
    if not (TE_AVAILABLE and torch.cuda.is_available()):
        return False
    available = _fp8_global_state_manager().is_fp8_available()
    return available[0] if isinstance(available, tuple) else bool(available)


class TestDistributedLayers(unittest.TestCase):
    """
    Distributed layers against their serial counterparts, on any decomposition.

    The weight gradients are reduced in the tests themselves, see :meth:`_sum_gradient`, rather than by the
    gradient reduction hooks of the training, so that they check the layers and nothing else: summed over the
    spatial ranks, which split the computation of a layer and each hold the contribution of their part of the
    grid. Most of these layers do not split their computation over the matmul ranks, which compute the same on
    each, so their gradients are not summed over them: every matmul rank holds the full gradient already. The
    attention splits its heads over the matmul ranks, so the gradients of its q/k norms, which act on every head,
    are summed over them.
    """

    @classmethod
    def setUpClass(cls):
        _init_grid(cls)

    @classmethod
    def tearDownClass(cls):
        teardown_comms()

    def setUp(self):
        disable_tf32()

    def _split_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_local = _split_helper(tensor, dim=hdim, group=self.h_group)
        tensor_local = _split_helper(tensor_local, dim=wdim, group=self.w_group)
        return tensor_local

    def _sum_gradient(self, grad, group):
        """The gradient of a parameter shared over the ranks of a group that split its computation, summed over them."""
        grad = grad.clone()
        if comm.get_size(group) > 1:
            dist.all_reduce(grad, group=comm.get_group(group))
        return grad

    def _gather_helper(self, tensor, hdim=-2, wdim=-1):
        tensor_gather = _gather_helper(tensor, dim=hdim, group=self.h_group)
        tensor_gather = _gather_helper(tensor_gather, dim=wdim, group=self.w_group)

        return tensor_gather

    @parameterized.expand(
        [
            # even and odd latitudes, the same grid and up- and downsampling, at most 128 longitudes
            # so that the CPU runners of the CI hold every rank's serial reference
            [48, 96, 64, 128, 32, 8, 1e-3],
            [49, 96, 49, 96, 1, 10, 1e-3],
            [48, 96, 32, 64, 32, 8, 1e-4],
            [49, 96, 25, 48, 1, 10, 1e-4],
            [32, 64, 64, 128, 32, 8, 1e-4],
            [25, 48, 49, 96, 1, 10, 1e-4],
            # the three filters of SFNO with scale_factor=8 on a 36 x 72 grid: truncated modes,
            # mixed grid types and an internal grid too small to split evenly. SFNO clamps mmax to lmax,
            # as torch-harmonics keeps no orders m >= lmax
            [36, 72, 4, 9, 4, 8, 1e-4, "equiangular", "legendre-gauss", 4, 4],
            [4, 9, 4, 9, 4, 8, 1e-4, "legendre-gauss", "legendre-gauss", 4, 4],
            [4, 9, 36, 72, 4, 8, 1e-4, "legendre-gauss", "equiangular", 4, 4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectral_conv(
        self,
        nlat_in,
        nlon_in,
        nlat_out,
        nlon_out,
        batch_size,
        num_chan,
        tol,
        grid_in="equiangular",
        grid_out="equiangular",
        lmax=None,
        mmax=None,
        verbose=False,
    ):
        B, C, Hi, Wi, Ho, Wo = batch_size, num_chan, nlat_in, nlon_in, nlat_out, nlon_out

        from makani.models.common import spectral_conv_s2

        # set up handles; without explicit modes the inverse takes those of the forward
        forward_transform_local = th.RealSHT(sht_grid((Hi, Wi), grid_in), lmax=lmax, mmax=mmax).to(self.device)
        inverse_transform_local = th.InverseRealSHT(
            sht_grid((Ho, Wo), grid_out),
            lmax=forward_transform_local.lmax,
            mmax=forward_transform_local.mmax,
        ).to(self.device)
        forward_transform_dist = thd.DistributedRealSHT(sht_grid((Hi, Wi), grid_in), lmax=lmax, mmax=mmax).to(
            self.device
        )
        inverse_transform_dist = thd.DistributedInverseRealSHT(
            sht_grid((Ho, Wo), grid_out),
            lmax=forward_transform_dist.lmax,
            mmax=forward_transform_dist.mmax,
        ).to(self.device)

        set_seed(333)

        spect_conv_local = spectral_conv_s2(
            forward_transform_local,
            inverse_transform_local,
            C,
            C,
            num_groups=1,
            bias=True,
            gain=1.0,
        ).to(self.device)

        spect_conv_dist = spectral_conv_s2(
            forward_transform_dist,
            inverse_transform_dist,
            C,
            C,
            num_groups=1,
            bias=True,
            gain=1.0,
        ).to(self.device)

        # make sure weights are the same. The spectral bias starts at zero; a random one checks that
        # every rank adds the right modes of it. Reseeded, since the distributed initialization above
        # draws a different number of values on each rank where lmax does not split evenly
        set_seed(334)
        with torch.no_grad():
            spect_conv_local.spectral_bias.copy_(torch.randn_like(spect_conv_local.spectral_bias))
            weight = self._split_helper(spect_conv_local.weight, hdim=-1, wdim=None)
            spect_conv_dist.weight.copy_(weight)
            bias = self._split_helper(spect_conv_local.spectral_bias, hdim=-2, wdim=-1)
            spect_conv_dist.spectral_bias.copy_(bias)

        # every rank draws the same input only from the same generator state: the distributed
        # weights were initialized on each rank's share of the degrees, which differ in size when
        # lmax does not split evenly. On the GPU the input comes from the CUDA generator, which the
        # initialization on the CPU leaves alone, on the CPU both come from the same one
        set_seed(333)
        inp_full = torch.randn((B, C, Hi, Wi), dtype=torch.float32, device=self.device)

        #############################################################
        # local transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = spect_conv_local(inp_full)

        # create grads for backward
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        wgrad_full = spect_conv_local.weight.grad.clone()
        bgrad_full = spect_conv_local.spectral_bias.grad.clone()

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = spect_conv_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        # the weight is split over the degrees on the h ranks, and the w ranks each hold the contribution of their
        # orders to them; the spectral bias is split over the degrees on h and the orders on w
        wgrad_local = self._sum_gradient(spect_conv_dist.weight.grad, "w")
        bgrad_local = spect_conv_dist.spectral_bias.grad.clone()

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate input grads
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        #############################################################
        # evaluate Weight grads
        #############################################################
        with self.subTest(desc="weight gradients"):
            wgrad_gather_full = self._gather_helper(wgrad_local, hdim=-1, wdim=None)
            self.assertTrue(
                reduce_success(
                    compare_tensors("weight gradients", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="spectral bias gradients"):
            bgrad_gather_full = self._gather_helper(bgrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "spectral bias gradients", bgrad_gather_full, bgrad_full, tol, tol, verbose=verbose
                    ),
                    self.device,
                )
            )

    @parameterized.expand(
        [
            # nlat_in, nlon_in, nlat_out, nlon_out, lmax, mmax, batch_size, num_chan, tol
            # all modes, lmax != mmax (32 and 33), splitting evenly over two ranks in h
            [32, 64, 32, 64, None, None, 2, 8, 1e-5],
            # odd sizes, splitting unevenly over two and four ranks (lmax = 33, mmax = 36)
            [33, 70, 33, 70, None, None, 2, 8, 1e-5],
            # truncated, lmax < mmax, both odd
            [33, 70, 33, 70, 15, 21, 2, 8, 1e-5],
            # FNO's first block: onto a coarser grid, which resamples the residual
            [36, 72, 18, 36, 18, 19, 2, 8, 1e-5],
            # lmax == mmax (64): the local weight has to be a block of the global one wherever the modes are split
            [64, 126, 64, 126, None, None, 2, 8, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_fft_spectral_conv(
        self, nlat_in, nlon_in, nlat_out, nlon_out, lmax, mmax, batch_size, num_chan, tol, verbose=False
    ):
        """The planar convolution of FNO, against its serial counterpart.

        Its weight has one entry per mode and is split over the modes like the distributed FFTs split
        them, lmax over h and mmax over w, so every rank holds its own block of it.
        """
        from makani.models.common import SpectralConv, RealFFT2, InverseRealFFT2
        from makani.models.common.distributed.fft import DistributedRealFFT2, DistributedInverseRealFFT2

        B, C = batch_size, num_chan

        forward_local = RealFFT2(make_grid((nlat_in, nlon_in), "euclidean"), lmax=lmax, mmax=mmax).to(self.device)
        inverse_local = InverseRealFFT2(
            make_grid((nlat_out, nlon_out), "euclidean"), lmax=forward_local.lmax, mmax=forward_local.mmax
        )
        forward_dist = DistributedRealFFT2(make_grid((nlat_in, nlon_in), "euclidean"), lmax=lmax, mmax=mmax).to(
            self.device
        )
        inverse_dist = DistributedInverseRealFFT2(
            make_grid((nlat_out, nlon_out), "euclidean"), lmax=forward_dist.lmax, mmax=forward_dist.mmax
        )

        # the initialization scales each row of modes; with unit draws the weights are those scales, and
        # the distributed ones have to be the serial ones, split, whatever the split
        with mock.patch.object(torch, "randn", lambda *shape, **kwargs: torch.ones(*shape, **kwargs)):
            scale_local = SpectralConv(forward_local, inverse_local, C, C).to(self.device).weight
            scale_dist = SpectralConv(forward_dist, inverse_dist, C, C).to(self.device).weight
        with self.subTest(desc="initialization"):
            self.assertTrue(
                reduce_success(
                    compare_tensors(
                        "initialization", self._gather_helper(scale_dist.detach()), scale_local.detach(), 0.0, 0.0
                    ),
                    self.device,
                )
            )

        set_seed(333)
        conv_local = SpectralConv(forward_local, inverse_local.to(self.device), C, C, bias=True).to(self.device)
        conv_dist = SpectralConv(forward_dist, inverse_dist.to(self.device), C, C, bias=True).to(self.device)

        # the local weights are a block of the global ones
        self.assertEqual(conv_dist.weight.shape[-2:], self._split_helper(conv_local.weight).shape[-2:])

        # the bias starts at zero; a random one checks that it is added once. Reseeded, since the
        # distributed initialization draws a different number of values on each rank where the modes
        # do not split evenly
        set_seed(334)
        with torch.no_grad():
            conv_local.bias.copy_(torch.randn_like(conv_local.bias))
            conv_dist.weight.copy_(self._split_helper(conv_local.weight))
            conv_dist.bias.copy_(conv_local.bias)

        inp_full = torch.randn((B, C, nlat_in, nlon_in), dtype=torch.float32, device=self.device)

        # serial
        inp_full.requires_grad = True
        out_full, res_full = conv_local(inp_full)
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
            rgrad_full = torch.randn_like(res_full)
        torch.autograd.backward([out_full, res_full], [ograd_full, rgrad_full])
        igrad_full = inp_full.grad.clone()
        wgrad_full = conv_local.weight.grad.clone()
        bgrad_full = conv_local.bias.grad.clone()

        # distributed
        inp_local = self._split_helper(inp_full.detach().clone())
        inp_local.requires_grad = True
        out_local, res_local = conv_dist(inp_local)
        torch.autograd.backward(
            [out_local, res_local], [self._split_helper(ograd_full), self._split_helper(rgrad_full)]
        )
        igrad_local = inp_local.grad.clone()
        # every mode lives on one rank, so the weight gradient of a rank is complete; the bias is
        # shared by all spatial ranks, each holding the contribution of its points
        wgrad_local = conv_dist.weight.grad.clone()
        bgrad_local = self._sum_gradient(conv_dist.bias.grad, "spatial")

        for desc, local, full in [
            ("output", out_local, out_full),
            ("residual", res_local, res_full),
            ("input gradients", igrad_local, igrad_full),
            ("weight gradients", wgrad_local, wgrad_full),
        ]:
            with self.subTest(desc=desc):
                gathered = self._gather_helper(local)
                self.assertTrue(
                    reduce_success(compare_tensors(desc, gathered, full, tol, tol, verbose=verbose), self.device)
                )

        with self.subTest(desc="bias gradients"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("bias gradients", bgrad_local, bgrad_full, tol, tol, verbose=verbose), self.device
                )
            )

    # the DISCO convolutions of SNO with scale_factor=8 on a 36 x 72 grid: the encoder onto the internal
    # 4 x 9 grid and the convolution of the decoder on the output grid. The local convolution of the
    # blocks runs on a 16 x 33 grid instead of 4 x 9: its support would cover most of the sphere there,
    # so with 4 polar ranks each holding a single latitude, the halo would reach past a neighbour
    @parameterized.expand(
        [
            [36, 72, 4, 9, "equiangular", "legendre-gauss", (3, 3), 2.0 * math.pi / 35.0, 4, 8, 1e-4],
            [16, 33, 16, 33, "legendre-gauss", "legendre-gauss", (3, 3), math.sqrt(2.0) * math.pi / 15.0, 4, 8, 1e-4],
            [36, 72, 36, 72, "equiangular", "equiangular", (3, 3), 2.0 * math.pi / 35.0, 4, 8, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_disco_conv(
        self,
        nlat_in,
        nlon_in,
        nlat_out,
        nlon_out,
        grid_in,
        grid_out,
        kernel_shape,
        theta_cutoff,
        batch_size,
        num_chan,
        tol,
        verbose=False,
    ):
        B, C = batch_size, num_chan

        conv_args = dict(
            kernel_shape=kernel_shape,
            basis_type="harmonic",
            basis_norm_mode="mean",
            grid_in=sht_grid((nlat_in, nlon_in), grid_in),
            grid_out=sht_grid((nlat_out, nlon_out), grid_out),
            bias=False,
            theta_cutoff=theta_cutoff,
        )

        set_seed(333)
        conv_local = th.DiscreteContinuousConvS2(in_channels=C, out_channels=C, **conv_args).to(self.device)
        conv_dist = thd.DistributedDiscreteContinuousConvS2(in_channels=C, out_channels=C, **conv_args).to(self.device)
        with torch.no_grad():
            conv_dist.weight.copy_(conv_local.weight)

        inp_full = torch.randn((B, C, nlat_in, nlon_in), dtype=torch.float32, device=self.device)

        # local
        inp_full.requires_grad = True
        out_full = conv_local(inp_full)
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()
        wgrad_full = conv_local.weight.grad.clone()

        # distributed
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = conv_dist(inp_local)
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()
        # the weight is replicated over the spatial ranks, each holding the contribution of its part of the grid
        wgrad_local = self._sum_gradient(conv_dist.weight.grad, "spatial")

        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="weight gradients"):
            # a weight gradient sums over the whole sphere, which the ranks split differently, so its
            # rounding error follows the gradient's scale, not the entry: e.g. 1.5e-4 at an entry of
            # 0.08 next to entries of 43. The absolute tolerance follows the largest entry
            wgrad_atol = tol * max(1.0, wgrad_full.abs().max().item())
            self.assertTrue(
                reduce_success(
                    compare_tensors("weight gradients", wgrad_local, wgrad_full, wgrad_atol, tol, verbose=verbose),
                    self.device,
                )
            )

    def test_distributed_disco_conv_after_conv_without_halo(self, verbose=False):
        """
        SNO's encoder convolution followed by its decoder convolution, as in the model.

        On a 2 x 2 grid the encoder (36 x 72 -> 4 x 9) needs no latitude halo. torch-harmonics 0.9.3 still sent
        the whole local tensor as a halo, which nobody received; on NCCL the decoder's halo exchange picked it
        up and got the first latitude row of the second polar rank wrong, from B * C_in >= 256 on.
        """
        B, C, tol = 8, 32, 1e-4

        conv_args = dict(kernel_shape=(3, 3), basis_type="harmonic", basis_norm_mode="mean", bias=False)
        encoder_args = dict(
            grid_in=sht_grid((36, 72), "equiangular"),
            grid_out=sht_grid((4, 9), "legendre-gauss"),
            theta_cutoff=2.0 * math.pi / 35.0,
            **conv_args,
        )
        decoder_args = dict(
            grid_in=sht_grid((36, 72), "equiangular"),
            grid_out=sht_grid((36, 72), "equiangular"),
            theta_cutoff=2.0 * math.pi / 35.0,
            **conv_args,
        )

        set_seed(333)
        encoder_local = th.DiscreteContinuousConvS2(in_channels=5, out_channels=C, **encoder_args).to(self.device)
        encoder_dist = thd.DistributedDiscreteContinuousConvS2(in_channels=5, out_channels=C, **encoder_args)
        encoder_dist = encoder_dist.to(self.device)
        decoder_local = th.DiscreteContinuousConvS2(in_channels=C, out_channels=5, **decoder_args).to(self.device)
        decoder_dist = thd.DistributedDiscreteContinuousConvS2(in_channels=C, out_channels=5, **decoder_args)
        decoder_dist = decoder_dist.to(self.device)
        with torch.no_grad():
            encoder_dist.weight.copy_(encoder_local.weight)
            decoder_dist.weight.copy_(decoder_local.weight)

        enc_inp_full = torch.randn((B, 5, 36, 72), dtype=torch.float32, device=self.device)
        dec_inp_full = torch.randn((B, C, 36, 72), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            enc_out_full = encoder_local(enc_inp_full)
            dec_out_full = decoder_local(dec_inp_full)
            enc_out_local = encoder_dist(self._split_helper(enc_inp_full, hdim=-2, wdim=-1))
            dec_out_local = decoder_dist(self._split_helper(dec_inp_full, hdim=-2, wdim=-1))

        for desc, out_local, out_full in [
            ("encoder output", enc_out_local, enc_out_full),
            ("decoder output", dec_out_local, dec_out_full),
        ]:
            with self.subTest(desc=desc):
                out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
                self.assertTrue(
                    reduce_success(
                        compare_tensors(desc, out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                    )
                )

    # the bilinear resampling in the decoder of SNO, from the internal 4 x 9 grid back onto 36 x 72, and the
    # opposite direction
    @parameterized.expand(
        [
            [4, 9, 36, 72, "legendre-gauss", "equiangular", 4, 8, 1e-5],
            [36, 72, 4, 9, "equiangular", "legendre-gauss", 4, 8, 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_resample(
        self, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out, batch_size, num_chan, tol, verbose=False
    ):
        B, C = batch_size, num_chan

        resample_args = dict(
            grid_in=sht_grid((nlat_in, nlon_in), grid_in),
            grid_out=sht_grid((nlat_out, nlon_out), grid_out),
            mode="bilinear",
        )
        resample_local = th.ResampleS2(**resample_args).to(self.device)
        resample_dist = thd.DistributedResampleS2(**resample_args).to(self.device)

        set_seed(333)
        inp_full = torch.randn((B, C, nlat_in, nlon_in), dtype=torch.float32, device=self.device)

        # local
        inp_full.requires_grad = True
        out_full = resample_local(inp_full)
        with torch.no_grad():
            ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        # distributed
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = resample_dist(inp_local)
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

    # SNO's distributed layers called several times in a row with changing batch sizes, as the
    # gradient accumulation test does (a batch of 8 and batches of 4): every call has to match the
    # serial layer, whatever the module saw before. The local convolution runs on a 16 latitude grid
    # with SNO's cutoff for it, sqrt(2) pi / (nlat - 1): on 4 latitudes that cutoff spans 85 degrees,
    # a halo of 2 rows, more than the single latitude each of 4 polar ranks holds; on 16 the halo is
    # 1 row against 4 latitudes per rank
    @parameterized.expand(
        [
            ["disco_encoder", 36, 72, 4, 9, "equiangular", "legendre-gauss", 2.0 * math.pi / 35.0],
            ["disco_local", 16, 33, 16, 33, "legendre-gauss", "legendre-gauss", math.sqrt(2.0) * math.pi / 15.0],
            ["disco_decoder", 36, 72, 36, 72, "equiangular", "equiangular", 2.0 * math.pi / 35.0],
            ["resample_decoder", 4, 9, 36, 72, "legendre-gauss", "equiangular", None],
            ["spectral_global", 4, 9, 4, 9, "legendre-gauss", "legendre-gauss", None],
            ["instance_norm", 4, 9, 4, 9, "legendre-gauss", "legendre-gauss", None],
        ],
        skip_on_empty=True,
    )
    def test_distributed_layer_batch_size_sequence(
        self, kind, nlat_in, nlon_in, nlat_out, nlon_out, grid_in, grid_out, theta_cutoff, verbose=False
    ):
        # the channels of SNO in the model test: 5 variables in and out, embed_dim 32 inside
        C, tol = 32, 1e-4
        C_in = 5 if kind == "disco_encoder" else C
        C_out = 5 if kind == "disco_decoder" else C

        set_seed(333)
        if kind.startswith("disco"):
            conv_args = dict(
                kernel_shape=(3, 3),
                basis_type="harmonic",
                basis_norm_mode="mean",
                grid_in=sht_grid((nlat_in, nlon_in), grid_in),
                grid_out=sht_grid((nlat_out, nlon_out), grid_out),
                bias=False,
                theta_cutoff=theta_cutoff,
            )
            layer_local = th.DiscreteContinuousConvS2(in_channels=C_in, out_channels=C_out, **conv_args)
            layer_local = layer_local.to(self.device)
            layer_dist = thd.DistributedDiscreteContinuousConvS2(in_channels=C_in, out_channels=C_out, **conv_args)
            layer_dist = layer_dist.to(self.device)
            with torch.no_grad():
                layer_dist.weight.copy_(layer_local.weight)
        elif kind.startswith("resample"):
            resample_args = dict(
                grid_in=sht_grid((nlat_in, nlon_in), grid_in),
                grid_out=sht_grid((nlat_out, nlon_out), grid_out),
                mode="bilinear",
            )
            layer_local = th.ResampleS2(**resample_args).to(self.device)
            layer_dist = thd.DistributedResampleS2(**resample_args).to(self.device)
        elif kind == "instance_norm":
            # what SNO's blocks normalize with, on its internal grid; affine as in the model, with
            # weights away from the identity
            layer_local = nn.InstanceNorm2d(C, eps=1e-6, affine=True, track_running_stats=False).to(self.device)
            layer_dist = DistributedInstanceNorm2d(C, eps=1e-6, affine=True).to(self.device)
            with torch.no_grad():
                layer_local.weight.normal_()
                layer_local.bias.normal_()
                layer_dist.weight.copy_(layer_local.weight)
                layer_dist.bias.copy_(layer_local.bias)
        else:
            from makani.models.common import spectral_conv_s2

            transforms_local = (
                th.RealSHT(sht_grid((nlat_in, nlon_in), grid_in), lmax=4, mmax=4),
                th.InverseRealSHT(sht_grid((nlat_out, nlon_out), grid_out), lmax=4, mmax=4),
            )
            transforms_dist = (
                thd.DistributedRealSHT(sht_grid((nlat_in, nlon_in), grid_in), lmax=4, mmax=4),
                thd.DistributedInverseRealSHT(sht_grid((nlat_out, nlon_out), grid_out), lmax=4, mmax=4),
            )
            conv_args = dict(num_groups=1, bias=True, gain=1.0)
            layer_local = spectral_conv_s2(*[t.to(self.device) for t in transforms_local], C, C, **conv_args)
            layer_dist = spectral_conv_s2(*[t.to(self.device) for t in transforms_dist], C, C, **conv_args)
            layer_local, layer_dist = layer_local.to(self.device), layer_dist.to(self.device)
            # the same bias on every rank, however much of the generator the distributed initialization used
            set_seed(334)
            with torch.no_grad():
                layer_local.spectral_bias.copy_(torch.randn_like(layer_local.spectral_bias))
                layer_dist.weight.copy_(self._split_helper(layer_local.weight, hdim=-1, wdim=None))
                layer_dist.spectral_bias.copy_(self._split_helper(layer_local.spectral_bias, hdim=-2, wdim=-1))

        # the same inputs on every rank, however much of the generator the distributed initialization
        # used, see test_distributed_spectral_conv
        set_seed(333)

        # B=4 first, so that a failure at B=8 is not mistaken for one of the first call
        for call, batch_size in enumerate([4, 8, 4]):
            inp_full = torch.randn((batch_size, C_in, nlat_in, nlon_in), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                out_full = layer_local(inp_full)
                out_local = layer_dist(self._split_helper(inp_full, hdim=-2, wdim=-1))
            with self.subTest(desc=f"call {call}, batch {batch_size}: output"):
                out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
                self.assertTrue(
                    reduce_success(
                        compare_tensors(
                            f"{kind} call {call} output", out_gather_full, out_full, tol, tol, verbose=verbose
                        ),
                        self.device,
                    )
                )

            # each layer against itself, batch by batch: the samples are independent, so the larger
            # batch has to give what its halves give. Tells which of the two layers is off
            if batch_size > 4:
                with torch.no_grad():
                    halves_full = torch.cat([layer_local(x) for x in inp_full.split(4, dim=0)], dim=0)
                    inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
                    halves_local = torch.cat([layer_dist(x) for x in inp_local.split(4, dim=0)], dim=0)
                with self.subTest(desc=f"call {call}, batch {batch_size}: serial layer vs its halves"):
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"{kind} call {call} serial vs halves", out_full, halves_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )
                with self.subTest(desc=f"call {call}, batch {batch_size}: distributed layer vs its halves"):
                    halves_gather_full = self._gather_helper(halves_local, hdim=-2, wdim=-1)
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"{kind} call {call} distributed vs halves",
                                out_gather_full,
                                halves_gather_full,
                                tol,
                                tol,
                                verbose=verbose,
                            ),
                            self.device,
                        )
                    )

    @parameterized.expand(
        [
            [64, 128, 32, 8, True, 1e-4],
            [65, 128, 1, 10, True, 1e-4],
            [64, 128, 32, 8, False, 1e-4],
            [65, 128, 1, 10, False, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_instance_norm_2d(self, nlat, nlon, batch_size, num_chan, affine, tol, verbose=False):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        set_seed(333)

        # create local (serial) instance norm - using PyTorch's standard InstanceNorm2d
        norm_local = nn.InstanceNorm2d(
            num_features=C,
            eps=1e-5,
            affine=affine,
            track_running_stats=False,
        ).to(self.device)

        # create distributed instance norm
        norm_dist = DistributedInstanceNorm2d(
            num_features=C,
            eps=1e-5,
            affine=affine,
        ).to(self.device)

        # make sure weights are the same if affine=True
        if affine:
            with torch.no_grad():
                norm_dist.weight.copy_(norm_local.weight)
                norm_dist.bias.copy_(norm_local.bias)

        # input
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        #############################################################
        # local (serial) transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = norm_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        if affine:
            wgrad_full = norm_local.weight.grad.clone()
            bgrad_full = norm_local.bias.grad.clone()

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = norm_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        if affine:
            # shared by the spatial ranks, each holding the contribution of its part of the grid
            wgrad_local = self._sum_gradient(norm_dist.weight.grad, "spatial")
            bgrad_local = self._sum_gradient(norm_dist.bias.grad, "spatial")

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate input grads
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        #############################################################
        # evaluate weight and bias grads
        #############################################################
        # weight gradients should be the same across all processes
        if affine:
            with self.subTest(desc="weight gradients"):
                wgrad_gather_list = [torch.empty_like(wgrad_local) for _ in range(self.world_size)]
                wgrad_gather_list[self.world_rank] = wgrad_local
                dist.all_gather(wgrad_gather_list, wgrad_local, group=None)
                for idw, wgrad_gather_full in enumerate(wgrad_gather_list):
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"weight gradient {idw}", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )

        # bias gradients should be the same across all processes
        if affine:
            with self.subTest(desc="bias gradients"):
                bgrad_gather_list = [torch.empty_like(bgrad_local) for _ in range(self.world_size)]
                bgrad_gather_list[self.world_rank] = bgrad_local
                dist.all_gather(bgrad_gather_list, bgrad_local, group=None)
                for idb, bgrad_gather_full in enumerate(bgrad_gather_list):
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"bias gradient {idb}", bgrad_gather_full, bgrad_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )

    @parameterized.expand([[2, 16, 16, 4, True, 1e-5], [2, 16, 16, 4, False, 1e-5]], skip_on_empty=True)
    def test_distributed_attention(self, batch_size, num_tokens, dim, num_heads, qk_norm, tol, verbose=False):
        """
        The attention with its heads split over the matmul ranks against the serial one.

        Each matmul rank holds the rows of its heads of the query, key and value projections, contiguous in each, and
        the matching columns of the output projection; the gradients of the q/k norms, which act on every head, are
        summed over the matmul ranks. The attention does not split the tokens, so the spatial ranks compute the same.
        """
        from makani.models.common.distributed.attention import MultiHeadAttention

        msize, mrank = comm.get_size("matmul"), comm.get_rank("matmul")
        if num_heads % msize != 0:
            self.skipTest(f"{num_heads} heads do not split over {msize} matmul ranks")
        rows = slice(mrank * dim // msize, (mrank + 1) * dim // msize)

        kwargs = dict(num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)
        set_seed(333)
        with comm.serial():
            attn_local = MultiHeadAttention(dim, **kwargs).to(self.device)
        attn_dist = MultiHeadAttention(dim, **kwargs).to(self.device)
        self.assertEqual(
            {name: tuple(p.shape) for name, p in attn_local.named_parameters()},
            {
                name: tuple(
                    s * (msize if g == "matmul" else 1)
                    for s, g in zip(p.shape, getattr(p, "sharded_dims_mp", [None] * p.dim()))
                )
                for name, p in attn_dist.named_parameters()
            },
        )

        # the layout of a sharded parameter: its rows, or for the output projection its columns
        def local(name, t):
            if name.startswith("proj.weight"):
                return t[:, rows]
            if name.split(".")[0] in ("q", "k", "v"):
                return t[rows]
            return t

        with torch.no_grad():
            for name, param in attn_local.named_parameters():
                param.normal_(std=0.3)
            for name, param in attn_dist.named_parameters():
                param.copy_(local(name, dict(attn_local.named_parameters())[name]))

        # a flexible checkpoint of the split attention is that of the serial one, and restores into it
        gathered = checkpoint_helpers.gather_model_state_dict(attn_dist)
        with comm.serial():
            serial = checkpoint_helpers.gather_model_state_dict(attn_local)
        scattered = checkpoint_helpers.scatter_model_state_dict(attn_dist, OrderedDict(serial))
        for name, param in attn_dist.named_parameters():
            with self.subTest(desc=f"checkpoint {name}"):
                self.assertTrue(reduce_success(torch.equal(gathered[name], serial[name]), self.device))
                self.assertTrue(
                    reduce_success(torch.equal(scattered[name].to(self.device), param.detach()), self.device)
                )

        set_seed(333)
        inp_full = torch.randn((batch_size, num_tokens, dim), dtype=torch.float32, device=self.device)
        ograd_full = torch.randn((batch_size, num_tokens, dim), dtype=torch.float32, device=self.device)

        inp_full.requires_grad = True
        out_full = attn_local(inp_full)
        out_full.backward(ograd_full)

        inp_local = inp_full.detach().clone().requires_grad_(True)
        out_local = attn_dist(inp_local)
        out_local.backward(ograd_full)

        checks = {"output": (out_local, out_full), "input gradients": (inp_local.grad, inp_full.grad)}
        params_local = dict(attn_local.named_parameters())
        for name, param in attn_dist.named_parameters():
            grad = param.grad
            if name.split(".")[0] in ("q_norm", "k_norm"):
                # act on every head, so each matmul rank holds the contribution of its heads
                grad = self._sum_gradient(grad, "matmul")
            # the others are split over matmul with the heads, or, the bias of the output projection, added after
            # the reduction, which every matmul rank computes alike
            checks[f"{name} gradients"] = (grad, local(name, params_local[name].grad))

        for label, (local_value, full) in checks.items():
            with self.subTest(desc=label):
                self.assertTrue(
                    reduce_success(compare_tensors(label, local_value, full, tol, tol, verbose=verbose), self.device)
                )

    def test_attention_has_no_redundant_biases(self):
        """A bias of the keys adds the same to every logit of a query, which the softmax removes."""
        from makani.models.common.distributed.attention import MultiHeadAttention

        with comm.serial():
            plain = MultiHeadAttention(16, num_heads=4, qkv_bias=True, qk_norm=False)
            normed = MultiHeadAttention(16, num_heads=4, qkv_bias=True, qk_norm=True)
        self.assertIsNone(plain.k.bias)
        self.assertIsNotNone(plain.q.bias)
        self.assertIsNotNone(plain.v.bias)
        self.assertIsNotNone(normed.k.bias)
        self.assertIsNone(normed.k_norm.bias)
        self.assertIsNotNone(normed.q_norm.bias)

    @parameterized.expand(
        [
            [33, 64, 2, 1, 3, "equiangular", "mean", 1e-5],
            [33, 64, 2, 2, 3, "equiangular", "exponential", 1e-5],
            [32, 64, 2, 1, 3, "legendre-gauss", "exponential", 1e-5],
            [33, 64, 3, 2, 3, "equiangular", "timediff", 1e-5],
        ],
        skip_on_empty=True,
    )
    def test_distributed_history_normalization(
        self, nlat, nlon, batch_size, n_history, num_chan, grid_type, mode, tol, verbose=False
    ):
        """
        The history normalization, distributed over the spatial ranks, against the serial one, with the gradient
        flowing through its statistics, as it does where the window holds predictions (multistep training).
        """
        from makani.models.preprocessor import HistoryNormalization
        from ..testutils import get_default_parameters

        params = get_default_parameters()
        params.img_shape_x_resampled, params.img_shape_y_resampled = nlat, nlon
        params.n_history = n_history
        params.history_normalization_mode = mode
        params.history_normalization_decay = 0.5
        params.model_grid_type = grid_type

        set_seed(333)
        with comm.serial():
            norm_local = HistoryNormalization(params).to(self.device)
        norm_dist = HistoryNormalization(params).to(self.device)

        # a window whose statistics differ from sample to sample and channel to channel
        shape = (batch_size, n_history + 1, num_chan, nlat, nlon)
        x_full = 2.0 * torch.randn(shape, device=self.device) + torch.randn(
            batch_size, 1, num_chan, 1, 1, device=self.device
        )
        x_full.requires_grad_(True)
        ograd_full = torch.randn_like(x_full)
        # the serial reference runs as a single process: its copies to the parallel region must not reduce
        with comm.serial():
            stats_full = norm_local.compute_stats(x_full)
            out_full = norm_local.normalize(x_full, stats_full)
        x_local = self._split_helper(x_full.detach()).requires_grad_(True)
        stats_local = norm_dist.compute_stats(x_local)
        out_local = norm_dist.normalize(x_local, stats_local)

        if mode == "timediff":
            # normalize leaves the window as it is; the statistics, one mean and variance of the time differences per
            # sample, are compared, and the gradient flows through them
            with self.subTest(desc="statistics"):
                for name in ("diff_mean", "diff_var"):
                    local, full = getattr(stats_local, name), getattr(stats_full, name)
                    self.assertEqual(tuple(full.shape), (batch_size,))
                    self.assertTrue(
                        reduce_success(compare_tensors(name, local, full, tol, tol, verbose=verbose), self.device)
                    )

            # used on each rank's own part of the grid, as normalize uses the statistics of the other modes
            def use(x, stats):
                return x * stats.diff_mean.reshape(-1, 1, 1, 1, 1) + stats.diff_var.reshape(-1, 1, 1, 1, 1)

            with comm.serial():
                use(x_full, stats_full).backward(ograd_full)
            use(x_local, stats_local).backward(self._split_helper(ograd_full))
        else:
            with comm.serial():
                out_full.backward(ograd_full)
            out_local.backward(self._split_helper(ograd_full))

        with self.subTest(desc="output"):
            out_gather = self._gather_helper(out_local)
            self.assertTrue(
                reduce_success(compare_tensors("output", out_gather, out_full, tol, tol, verbose=verbose), self.device)
            )
        with self.subTest(desc="input gradients"):
            igrad_gather = self._gather_helper(x_local.grad)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather, x_full.grad, tol, tol, verbose=verbose),
                    self.device,
                )
            )

    @parameterized.expand(
        [
            [65, 128, 1, 4, "equiangular", True, 1e-4],
            [65, 128, 1, 4, "equiangular", False, 1e-4],
            [64, 128, 1, 10, "legendre-gauss", True, 1e-4],
            [64, 128, 1, 10, "legendre-gauss", False, 1e-4],
        ],
        skip_on_empty=True,
    )
    def test_distributed_geometric_instance_norm_s2(
        self, nlat, nlon, batch_size, num_chan, grid_type, affine, tol, verbose=False
    ):
        B, C, H, W = batch_size, num_chan, nlat, nlon

        # set up layer norm parameters
        img_shape = (H, W)
        crop_shape = (H, W)
        crop_offset = (0, 0)
        eps = 1e-5

        set_seed(333)

        # create local (serial) layer norm
        norm_local = GeometricInstanceNormS2(
            grid=make_grid(img_shape, grid_type),
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            num_features=C,
            eps=eps,
            affine=affine,
        ).to(self.device)

        # create distributed layer norm
        norm_dist = DistributedGeometricInstanceNormS2(
            grid=make_grid(img_shape, grid_type),
            crop_shape=crop_shape,
            crop_offset=crop_offset,
            num_features=C,
            eps=eps,
            affine=affine,
        ).to(self.device)

        # make sure weights are the same if affine=True
        if affine:
            with torch.no_grad():
                norm_dist.weight.copy_(norm_local.weight)
                norm_dist.bias.copy_(norm_local.bias)

        # input
        inp_full = torch.randn((B, C, H, W), dtype=torch.float32, device=self.device)

        #############################################################
        # local (serial) transform
        #############################################################
        # FWD pass
        inp_full.requires_grad = True
        out_full = norm_local(inp_full)

        # create grad for backward
        with torch.no_grad():
            # create full grad
            ograd_full = torch.randn_like(out_full)

        # BWD pass
        out_full.backward(ograd_full)
        igrad_full = inp_full.grad.clone()

        if affine:
            wgrad_full = norm_local.weight.grad.clone()
            bgrad_full = norm_local.bias.grad.clone()

        #############################################################
        # distributed transform
        #############################################################
        # FWD pass
        inp_local = self._split_helper(inp_full, hdim=-2, wdim=-1)
        inp_local.requires_grad = True
        out_local = norm_dist(inp_local)

        # BWD pass
        ograd_local = self._split_helper(ograd_full, hdim=-2, wdim=-1)
        out_local.backward(ograd_local)
        igrad_local = inp_local.grad.clone()

        if affine:
            # shared by the spatial ranks, each holding the contribution of its part of the grid
            wgrad_local = self._sum_gradient(norm_dist.weight.grad, "spatial")
            bgrad_local = self._sum_gradient(norm_dist.bias.grad, "spatial")

        #############################################################
        # evaluate FWD pass
        #############################################################
        with self.subTest(desc="output"):
            out_gather_full = self._gather_helper(out_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", out_gather_full, out_full, tol, tol, verbose=verbose), self.device
                )
            )

        #############################################################
        # evaluate input grads
        #############################################################
        with self.subTest(desc="input gradients"):
            igrad_gather_full = self._gather_helper(igrad_local, hdim=-2, wdim=-1)
            self.assertTrue(
                reduce_success(
                    compare_tensors("input gradients", igrad_gather_full, igrad_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        #############################################################
        # evaluate weight and bias grads
        #############################################################
        # weight gradients should be the same across all processes
        if affine:
            with self.subTest(desc="weight gradients"):
                wgrad_gather_list = [torch.empty_like(wgrad_local) for _ in range(self.world_size)]
                wgrad_gather_list[self.world_rank] = wgrad_local
                dist.all_gather(wgrad_gather_list, wgrad_local, group=None)
                for idw, wgrad_gather_full in enumerate(wgrad_gather_list):
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"weight gradient {idw}", wgrad_gather_full, wgrad_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )

        # bias gradients should be the same across all processes
        if affine:
            with self.subTest(desc="bias gradients"):
                bgrad_gather_list = [torch.empty_like(bgrad_local) for _ in range(self.world_size)]
                bgrad_gather_list[self.world_rank] = bgrad_local
                dist.all_gather(bgrad_gather_list, bgrad_local, group=None)
                for idb, bgrad_gather_full in enumerate(bgrad_gather_list):
                    self.assertTrue(
                        reduce_success(
                            compare_tensors(
                                f"bias gradient {idb}", bgrad_gather_full, bgrad_full, tol, tol, verbose=verbose
                            ),
                            self.device,
                        )
                    )

    @parameterized.expand(
        [
            # over 2 x 2 ranks, the parts of 17 x 24 points are padded to a multiple of 16, those of 16 x 24 are not;
            # the serial grid of 1584 points is not either
            [33, 48, 2, 64, 128, "bf16-fp8_delayed"],
            [33, 48, 2, 64, 128, "bf16-fp8_current"],
            # 17 x 30 points are padded on any decomposition, a single rank as well
            [17, 30, 2, 64, 128, "bf16-fp8_delayed"],
        ],
        skip_on_empty=True,
    )
    # below the expansion, so that every generated case carries the skip, not only the original function
    @unittest.skipUnless(_fp8_available(), "transformer_engine with an FP8 GPU is not available")
    def test_distributed_mlp_fp8(self, nlat, nlon, batch_size, num_chan, hidden_dim, amp_mode, steps=3, verbose=False):
        """
        The distributed MLP in FP8 against the serial one, on grids whose parts the FP8 GEMMs pad.

        All ranks run the same layers in FP8, whether their part of the grid is padded or not, as the amax
        reduction of delayed scaling over all ranks needs: the test finishes, rather than hangs.

        FP8 rounds to a few percent, so the tensors are compared by their relative error over the whole tensor, not
        element by element: the distributed MLP in FP8 is as close to the serial one in bf16 as the serial one in FP8
        is. Under delayed scaling the scales are those of the serial layers, as the amaxes are reduced over all
        ranks, so the distributed MLP computes what the serial one does in FP8, up to the bf16 rounding of the
        partial sums over the matmul ranks. Under current scaling each rank scales its own part, the shard of a
        matmul rank and the part of the grid of a spatial rank, which rounds to other FP8 values than the serial
        layers do, as accurately.
        """
        if hidden_dim % comm.get_size("matmul") != 0:
            self.skipTest(f"hidden_dim {hidden_dim} not divisible by matmul size {comm.get_size('matmul')}")
        delayed = amp_mode == "bf16-fp8_delayed"

        set_seed(333)
        with fp8_layers(True):
            with comm.serial():
                mlp_full = MLP(num_chan, hidden_dim, num_chan, act_layer=nn.GELU, input_format="nchw").to(self.device)
            mlp_dist = DistributedMLP(
                num_chan, hidden_dim, num_chan, input_format="nchw", comm_name="matmul", act_layer=nn.GELU
            ).to(self.device)
        for mlp in [mlp_full, mlp_dist]:
            self.assertTrue(all(fc.fp8_gemm is not None for fc in [mlp.fc1, mlp.fc2]))

        # the serial weights, split over the matmul ranks
        weights = OrderedDict((name, param.detach()) for name, param in mlp_full.named_parameters())
        with torch.no_grad():
            mlp_dist.load_state_dict(checkpoint_helpers.scatter_model_state_dict(mlp_dist, weights), strict=False)

        def run(mlp, inp, ograd, mode):
            # each in its own autocast, whose first FP8 module updates the scales of all in its backward
            inp = inp.clone().requires_grad_(True)
            mlp.zero_grad()
            with AutocastManager(mode)():
                out = mlp(inp)
            out.backward(ograd)
            return out.float(), inp.grad

        def relative_error(tensor, reference):
            return ((tensor.float() - reference.float()).norm() / reference.float().norm()).item()

        # the bf16 rounding of what the ranks reduce, their partial sums over the matmul ranks and over their parts
        # of the grid: a few 1e-3, also where FP8 does not enter and the serial MLP in FP8 computes what it computes
        # in bf16, as for the gradient of the output bias, the sum of the output gradient
        rounding = 1e-2

        def check(desc, dist_fp8, full_fp8, full_bf16):
            error, error_serial = relative_error(dist_fp8, full_bf16), relative_error(full_fp8, full_bf16)
            ok = error <= 1.25 * error_serial + rounding
            if delayed:
                ok = ok and relative_error(dist_fp8, full_fp8) <= rounding
            if verbose:
                print(f"{desc}: {error:.3e} from bf16 (serial FP8 {error_serial:.3e})")
            with self.subTest(desc=desc):
                self.assertTrue(reduce_success(ok, self.device), desc)

        for step in range(steps):
            inp_full = torch.randn(batch_size, num_chan, nlat, nlon, device=self.device) * (1 + step)
            ograd_full = torch.randn(batch_size, num_chan, nlat, nlon, device=self.device)

            # the serial MLP in bf16, outside the FP8 context, and in FP8
            out_bf16, igrad_bf16 = run(mlp_full, inp_full, ograd_full, "bf16")
            grads_bf16 = {name: param.grad.clone() for name, param in mlp_full.named_parameters()}
            out_full, igrad_full = run(mlp_full, inp_full, ograd_full, amp_mode)
            out_local, igrad_local = run(
                mlp_dist, self._split_helper(inp_full), self._split_helper(ograd_full), amp_mode
            )

            with self.subTest(desc="FP8 on all ranks", step=step):
                ran = all(fc.fp8_gemm.forwarded_at_least_once for fc in [mlp_dist.fc1, mlp_dist.fc2])
                self.assertTrue(reduce_success(ran, self.device))

            check(f"output, step {step}", self._gather_helper(out_local), out_full, out_bf16)
            check(f"input gradients, step {step}", self._gather_helper(igrad_local), igrad_full, igrad_bf16)

            if delayed:
                for name in ["fc1", "fc2"]:
                    for direction in ["scaling_fwd", "scaling_bwd"]:
                        meta_dist = getattr(mlp_dist, name).fp8_gemm.fp8_meta[direction]
                        meta_full = getattr(mlp_full, name).fp8_gemm.fp8_meta[direction]
                        with self.subTest(desc=f"{name} {direction} scales", step=step):
                            self.assertTrue(
                                reduce_success(
                                    compare_tensors(
                                        f"{name} {direction} scales", meta_dist.scale, meta_full.scale, 1e-2, 1e-2
                                    ),
                                    self.device,
                                )
                            )

        # the weight gradients of the last step, summed over the spatial ranks and gathered over the matmul ranks
        for param in mlp_dist.parameters():
            param.grad = self._sum_gradient(param.grad, "spatial")
        grads = checkpoint_helpers.gather_model_state_dict(mlp_dist, grads=True)
        for name, param in mlp_full.named_parameters():
            check(f"{name} gradient", grads[name + ".grad"], param.grad.cpu(), grads_bf16[name].cpu())

    @parameterized.expand(
        [
            # even and odd latitudes, zonal and longitude-dependent, and fewer degrees than ranks of most grids, so
            # that the transform keeps more degrees and orders than the embedding
            [48, 96, 16, 1, "legendre-gauss"],
            [49, 96, 16, 4, "legendre-gauss"],
            [25, 48, 2, 1, "legendre-gauss"],
            [37, 72, 8, 8, "equiangular"],
        ],
        skip_on_empty=True,
    )
    def test_distributed_spectral_position_embedding(self, nlat, nlon, lmax, mmax, grid_type, verbose=False):
        """The embedding on each rank's part of the grid, and the gradient of its coefficients summed over the spatial
        ranks, against the inverse SHT of the coefficients on the whole grid."""
        set_seed(333)
        grid = make_grid((nlat, nlon), grid_type)
        tol = 1e-5

        emb = SpectralPositionEmbedding(grid, num_chans=8, lmax=lmax, mmax=mmax).to(self.device)
        with torch.no_grad():
            emb.coeffs.normal_()
        coeffs_full = emb.coeffs.detach().clone().requires_grad_(True)

        # the serial reference, on the whole grid
        isht = th.InverseRealSHT(grid, lmax=lmax, mmax=mmax).to(self.device)
        out_full = isht(torch.view_as_complex((coeffs_full * emb.mask).contiguous()).unsqueeze(0))
        ograd_full = torch.randn_like(out_full)
        out_full.backward(ograd_full)

        out_local = emb()
        out_local.backward(self._split_helper(ograd_full))

        with self.subTest(desc="output"):
            self.assertTrue(
                reduce_success(
                    compare_tensors("output", self._gather_helper(out_local), out_full, tol, tol, verbose=verbose),
                    self.device,
                )
            )

        with self.subTest(desc="coefficient gradients"):
            grad = self._sum_gradient(emb.coeffs.grad, "spatial")
            self.assertTrue(
                reduce_success(
                    compare_tensors("coefficient gradients", grad, coeffs_full.grad, tol, tol, verbose=verbose),
                    self.device,
                )
            )


if __name__ == "__main__":
    unittest.main()
