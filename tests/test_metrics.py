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
import unittest
from parameterized import parameterized

import itertools
import tempfile
import numpy as np
import torch
import xarray as xr
import xskillscore as xs

from makani.utils import functions as fn
from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature
from makani.utils import MetricsHandler
from makani.utils.metrics.functions import GeometricL1, GeometricRMSE, GeometricACC, GeometricPCC, GeometricCRPS, GeometricSSR, GeometricRankHistogram, Quadrature

from testutils import get_default_parameters

# check consistency with weatherbench2 if it is installed
try:
    import weatherbench2 as wb2
    from weatherbench2 import metrics
except ImportError:
    wb2 = None

# parameters for deterministic metrics
# grid_type, shape
_deterministic_metrics_params = [
    ("euclidean", *(1, 10, 17, 32)),
    ("equiangular", *(1, 10, 17, 32)),
    ("equiangular", *(4, 21, 17, 32)),
    ("equiangular", *(8, 21, 17, 32)),
    ("clenshaw-curtiss", *(1, 10, 17, 32)),
    ("clenshaw-curtiss", *(4, 21, 17, 32)),
    ("legendre-gauss", *(1, 10, 17, 32)),
    ("legendre-gauss", *(4, 21, 17, 32)),
]

# parameters for probabilistic metrics
# grid_type, shape, ensemble_size
_probabilistic_metrics_params = [
    ("equiangular", *(1, 1, 10, 17, 32)),
    ("equiangular", *(1, 21, 10, 17, 32)),
    ("equiangular", *(4, 16, 21, 17, 32)),
    ("clenshaw-curtiss", *(1, 21, 10, 17, 32)),
    ("clenshaw-curtiss", *(4, 16, 21, 17, 32)),
    ("legendre-gauss", *(1, 21, 10, 17, 32)),
    ("legendre-gauss", *(4, 16, 21, 17, 32)),
]

# parameters for weatherbench2 metrics
_wb2_metrics_params = [
    ("equiangular", *(1, 21, 10, 17, 32)),
    ("equiangular", *(4, 16, 21, 17, 32)),
    ("clenshaw-curtiss", *(1, 21, 10, 17, 32)),
    ("clenshaw-curtiss", *(4, 16, 21, 17, 32)),
    ("weatherbench2", *(1, 21, 10, 17, 32)),
    ("weatherbench2", *(4, 16, 21, 17, 32)),
]

_deterministic_metric_aggregation_params = [
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "mean", "mean"),
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "none", "mean"),
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "sum", "sum"),
    (GeometricL1, "equiangular", *(4, 21, 17, 32), "none", "sum"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "mean", "mean"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "none", "mean"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "sum", "sum"),
    (GeometricRMSE, "equiangular", *(4, 21, 17, 32), "none", "sum"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "mean", "mean"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "none", "mean"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "sum", "sum"),
    (GeometricACC, "equiangular", *(4, 21, 17, 32), "none", "sum"),
    (GeometricPCC, "equiangular", *(4, 21, 17, 32), "mean", "mean"),
    (GeometricPCC, "equiangular", *(4, 21, 17, 32), "none", "mean"),
    (GeometricPCC, "equiangular", *(4, 21, 17, 32), "sum", "sum"),
    (GeometricPCC, "equiangular", *(4, 21, 17, 32), "none", "sum"),
]

_probabilistic_metric_aggregation_params = [
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "mean", "mean"),
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "none", "mean"),
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricCRPS, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "mean", "mean"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "none", "mean"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricSSR, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "mean", "mean"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "none", "mean"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "sum", "sum"),
    (GeometricRankHistogram, "equiangular", *(4, 16, 21, 17, 32), "none", "sum"),
]

_metric_handler_params = [
    ("equiangular", 4, 16, 3, "mean"),
    ("equiangular", 4, 16, 3, "sum"),
]

class TestMetrics(unittest.TestCase):
    """
    Testsuite for makani metrics. Compares to properscoring
    """

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        return

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_weight_normalization(self, grid_type, batch_size, num_channels, nlat, nlon):
        quadrature_type = grid_to_quadrature_rule(grid_type)
        quad = GridQuadrature(quadrature_type, img_shape=(nlat, nlon), normalize=True).to(self.device)
        flat_vector = torch.ones(batch_size, num_channels, nlat, nlon, device=self.device)
        integral = torch.mean(quad(flat_vector)).item()

        self.assertTrue(np.allclose(integral, 1.0, rtol=1e-5, atol=0))

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_weighted_rmse(self, grid_type, batch_size, num_channels, nlat, nlon):

        # rmse handle
        rmse_func = GeometricRMSE(grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction="none", batch_reduction="none").to(self.device)

        # generate toy data
        A = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        B = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        rmse = torch.mean(rmse_func(A, B), dim=0).cpu().numpy()

        lwf = torch.squeeze(rmse_func.quadrature.quad_weight).cpu().numpy()
        lwf = xr.DataArray(lwf, dims=["lat", "lon"])
        A = xr.DataArray(A.cpu(), dims=["batch", "channels", "lat", "lon"])
        B = xr.DataArray(B.cpu(), dims=["batch", "channels", "lat", "lon"])
        rmse_xskillscore = xs.rmse(A, B, weights=lwf, dim=["lat", "lon"]).to_numpy().mean(axis=0)

        # relative difference
        rdiff = np.abs(rmse - rmse_xskillscore) / np.abs(rmse_xskillscore)

        self.assertTrue(rdiff.mean() <= 1e-5)
        self.assertTrue(np.allclose(rmse, rmse_xskillscore, rtol=1e-5, atol=0))

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_l1(self, grid_type, batch_size, num_channels, nlat, nlon):

        # l1 handle
        l1_func = GeometricL1(grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction="mean", batch_reduction="mean").to(self.device)

        # generate toy data
        A = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        B = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        l1 = l1_func(A, B).cpu().numpy()

        lwf = l1_func.quadrature.quad_weight.cpu().numpy()
        lwf = xr.DataArray(np.tile(lwf, (batch_size, num_channels, 1, 1)), dims=["batch", "channels", "lat", "lon"])
        A = xr.DataArray(A.cpu(), dims=["batch", "channels", "lat", "lon"])
        B = xr.DataArray(B.cpu(), dims=["batch", "channels", "lat", "lon"])
        l1_xskillscore = xs.mae(A, B, weights=lwf).to_numpy()

        rdiff = np.abs(l1 - l1_xskillscore) / np.abs(l1_xskillscore)

        self.assertTrue(rdiff <= 1e-5)

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_weighted_acc_macro(self, grid_type, batch_size, num_channels, nlat, nlon):

        # ACC handle
        acc_func = GeometricACC(grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction="none", batch_reduction="mean", method="macro").to(self.device)

        # generate toy data
        A = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        B = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)

        # compute means
        A_mean = acc_func.quadrature(A).reshape(batch_size, num_channels, 1, 1)
        B_mean = acc_func.quadrature(B).reshape(batch_size, num_channels, 1, 1)

        # compute score using makani
        acc = acc_func(A - A_mean, B - B_mean).cpu().numpy()

        lwf = torch.squeeze(acc_func.quadrature.quad_weight).cpu().numpy()
        lwf = xr.DataArray(lwf, dims=["lat", "lon"])
        A = xr.DataArray(A.cpu(), dims=["batch", "channels", "lat", "lon"])
        B = xr.DataArray(B.cpu(), dims=["batch", "channels", "lat", "lon"])

        # compute score using xskillscore
        acc_xskillscore = xs.pearson_r(A, B, weights=lwf, dim=["lat", "lon"]).to_numpy().mean(axis=0)

        # compute difference
        rdiff = np.abs(acc - acc_xskillscore) / np.abs(acc_xskillscore)

        self.assertTrue(rdiff.mean() <= 1e-5)
        self.assertTrue(np.allclose(acc, acc_xskillscore, rtol=5e-5, atol=0))

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_weighted_acc_micro(self, grid_type, batch_size, num_channels, nlat, nlon):

        # ACC handle
        acc_func = GeometricACC(grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction="none", batch_reduction="mean", method="micro").to(self.device)

        # generate toy data
        A = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        B = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)

        # compute means
        A_mean = acc_func.quadrature(A).mean(dim=0).reshape(1, num_channels, 1, 1)
        B_mean = acc_func.quadrature(B).mean(dim=0).reshape(1, num_channels, 1, 1)

        # compute score using makani
        acc = acc_func(A - A_mean, B - B_mean).cpu().numpy()
        # finalize score
        acc = acc[..., 0] / np.sqrt(acc[..., 1] * acc[..., 2])

        # we need to ensure that the weights have the correct dimensions
        lwf = torch.tile(torch.squeeze(acc_func.quadrature.quad_weight).unsqueeze(0), (batch_size, 1, 1)).cpu().numpy()
        lwf = xr.DataArray(lwf, dims=["batch", "lat", "lon"])
        A = xr.DataArray(A.cpu(), dims=["batch", "channels", "lat", "lon"])
        B = xr.DataArray(B.cpu(), dims=["batch", "channels", "lat", "lon"])

        # compute score using xskillscore   
        acc_xskillscore = xs.pearson_r(A, B, weights=lwf, dim=["batch", "lat", "lon"]).to_numpy()

        # compute difference
        rdiff = np.abs(acc - acc_xskillscore) / np.abs(acc_xskillscore)

        self.assertTrue(rdiff.mean() <= 1e-5)
        self.assertTrue(np.allclose(acc, acc_xskillscore, rtol=5e-5, atol=0))

    @parameterized.expand(_deterministic_metrics_params, skip_on_empty=True)
    def test_weighted_pcc(self, grid_type, batch_size, num_channels, nlat, nlon):

        # PCC handle
        pcc_func = GeometricPCC(grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction="none", batch_reduction="mean").to(self.device)

        # generate toy data
        A = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        B = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)

        # compute means
        A_mean = pcc_func.quadrature(A).mean(dim=0).reshape(1, num_channels, 1, 1)
        B_mean = pcc_func.quadrature(B).mean(dim=0).reshape(1, num_channels, 1, 1)

        # compute score using makani
        pcc = pcc_func(A, B).cpu().numpy()
        # finalize score
        pcc = pcc[..., 0] / np.sqrt(pcc[..., 1] * pcc[..., 2])

        # we need to ensure that the weights have the correct dimensions
        lwf = torch.tile(torch.squeeze(pcc_func.quadrature.quad_weight).unsqueeze(0), (batch_size, 1, 1)).cpu().numpy()
        lwf = xr.DataArray(lwf, dims=["batch", "lat", "lon"])
        A = xr.DataArray(A.cpu(), dims=["batch", "channels", "lat", "lon"])
        B = xr.DataArray(B.cpu(), dims=["batch", "channels", "lat", "lon"])

        # compute score using xskillscore
        pcc_xskillscore = xs.pearson_r(A, B, weights=lwf, dim=["batch", "lat", "lon"]).to_numpy()

        # compute difference
        rdiff = np.abs(pcc - pcc_xskillscore) / np.abs(pcc_xskillscore)

        self.assertTrue(rdiff.mean() <= 1e-5)
        self.assertTrue(np.allclose(pcc, pcc_xskillscore, rtol=5e-5, atol=0))

    @parameterized.expand(_probabilistic_metrics_params, skip_on_empty=True)
    def test_weighted_crps(self, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon):

        # CRPS handle
        crps_func = GeometricCRPS(
            grid_type=grid_type, img_shape=(nlat, nlon), crop_shape=(nlat, nlon), crop_offset=(0, 0), crps_type="cdf", channel_reduction="none", batch_reduction="none"
        ).to(self.device)

        # generate toy data
        obs = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        fct = torch.randn(batch_size, ensemble_size, num_channels, nlat, nlon, device=self.device)
        crps = crps_func(fct, obs).cpu().numpy()

        lwf = torch.squeeze(crps_func.metric_func.quadrature.quad_weight).cpu().numpy()
        lwf = xr.DataArray(lwf, dims=["lat", "lon"])
        obs = xr.DataArray(obs.cpu(), dims=["batch", "channels", "lat", "lon"])
        fct = xr.DataArray(fct.cpu(), dims=["batch", "ensemble", "channels", "lat", "lon"])

        crps_xskillscore = xs.crps_ensemble(obs, fct, member_dim="ensemble", weights=lwf, dim=["lat", "lon"]).to_numpy()

        # compute difference 
        rdiff = np.abs(crps - crps_xskillscore) / np.abs(crps_xskillscore)

        self.assertTrue(rdiff.mean() <= 1e-5)
        self.assertTrue(np.allclose(crps, crps_xskillscore, rtol=5e-5, atol=0))

    # we need to relax the bounds here because on CPU, for some reason, the deviations are big
    @parameterized.expand(_probabilistic_metrics_params, skip_on_empty=True)
    def test_rank_histogram(self, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon, atol=1e-3, rtol=1e-4, verbose=True):
        # Rank histogram Handle
        rankhist_handle = GeometricRankHistogram(grid_type=grid_type,
                                                 img_shape=(nlat, nlon),
                                                 crop_shape=(nlat, nlon),
                                                 crop_offset=(0, 0),
                                                 normalize=True,
                                                 channel_reduction="none",
                                                 batch_reduction="mean").to(self.device)

        # generate toy data
        obs = torch.rand(batch_size, num_channels, nlat, nlon, device=self.device)
        # shift channels by channel index
        obs = obs + torch.linspace(start=0, end=num_channels-1, steps=num_channels, device=self.device).reshape(1, -1, 1, 1)
        # assume linear ensemble forecast
        fct = torch.linspace(start=0, end=1, steps=ensemble_size, device=self.device).reshape(1,-1,1,1,1)
        fct = torch.tile(fct, (batch_size, 1, num_channels, nlat, nlon))
        # move also by channel index
        fct = fct + torch.linspace(start=0, end=num_channels-1, steps=num_channels, device=self.device).reshape(1, -1, 1, 1)

        # compute rank histogram
        rankhist = rankhist_handle(fct, obs)

        # ensure that means are as expected
        means = torch.mean(rankhist, dim=(1))

        # expected means: always 1 / (ensemble_size+1)
        means_expected = torch.ones([num_channels]) / float(ensemble_size + 1)

        self.assertTrue(np.allclose(means_expected.cpu().numpy(), means.cpu().numpy(), atol=atol, rtol=rtol))

        # compare to xskillscore:
        obs = xr.DataArray(obs.cpu().numpy(), dims=["batch", "channels", "lat", "lon"])
        fct = xr.DataArray(fct.cpu().numpy(), dims=["batch", "ensemble", "channels", "lat", "lon"])
        rankhist_xskillscore = xs.rank_histogram(obs, fct, dim="batch", member_dim="ensemble").to_numpy()

        # we need to divide by the batch size because xskillscore does not average but just sum
        rankhist_xskillscore = rankhist_xskillscore / float(batch_size)

        # for spatial averaging, flatten the spatial dims
        rankhist_xskillscore = rankhist_xskillscore.reshape(num_channels, nlat*nlon, ensemble_size+1)
        qw = rankhist_handle.quad_weight_split.squeeze(0).cpu().numpy()
        rankhist_xskillscore = np.sum(rankhist_xskillscore * qw, axis=1)

        if verbose:
            rerr = fn.relative_error(rankhist, torch.from_numpy(rankhist_xskillscore).to(self.device))
            aerr = fn.absolute_error(rankhist, torch.from_numpy(rankhist_xskillscore).to(self.device))
            if verbose:
                print(f"final relative error of output: {rerr.item()}")
                print(f"final absolute error of output: {aerr.item()}")

        self.assertTrue(np.allclose(rankhist.cpu().numpy(), rankhist_xskillscore, atol=atol, rtol=rtol))


class TestMetricsAggregation(unittest.TestCase):
    """
    A set of tests that test the aggregation
    """

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        return

    @parameterized.expand(_deterministic_metric_aggregation_params, skip_on_empty=True)
    def test_deterministic_aggregation(self, metric_handle, grid_type, batch_size, num_channels, nlat, nlon, cred, bred, verbose=False):

        # inflate batch size
        batch_size = 4 * batch_size
                
        # metric handle
        metric_func = metric_handle(grid_type, img_shape=(nlat, nlon), normalize=True, channel_reduction=cred, batch_reduction=bred).to(self.device)

        # input and target:
        inp = torch.randn((batch_size, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)
        tar = torch.randn((batch_size, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)
        
        # full metric
        res_full = metric_func(inp, tar)
        counts_full = torch.tensor(inp.shape[0], dtype=torch.float32, device=self.device)
        res_full = metric_func.finalize(res_full, counts_full)
        
        # split and compute metrics stepwise
        inp_split = torch.split(inp, batch_size // 4, dim=0)
        tar_split = torch.split(tar, batch_size // 4, dim=0)
        
        res_split = metric_func(inp_split[0], tar_split[0])
        counts_split = torch.tensor(inp_split[0].shape[0], dtype=torch.float32, device=self.device)
        for inps, tars in zip(inp_split[1:], tar_split[1:]):
            res_tmp = metric_func(inps, tars)
            res_tmp = torch.stack([res_split, res_tmp], dim=0)
            counts_tmp = torch.tensor(inps.shape[0], dtype=torch.float32, device=self.device)
            counts_tmp = torch.stack([counts_split, counts_tmp], dim=0)
            res_tmp, counts_tmp = metric_func.combine(res_tmp, counts_tmp, dim=0)
            with torch.no_grad():
                res_split.copy_(res_tmp)
                counts_split.copy_(counts_tmp)
                
        res_split = metric_func.finalize(res_split, counts_split)

        # compare
        if verbose:
            print(f"Result full: {res_full}, result split: {res_split}")
        self.assertTrue(np.allclose(res_full.cpu().numpy(), res_split.cpu().numpy(), rtol=1e-6, atol=1e-6))


    @parameterized.expand(_probabilistic_metric_aggregation_params, skip_on_empty=True)
    def test_probabilistic_aggregation(self, metric_handle, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon, cred, bred):

        # inflate batch size
        batch_size = 4 * batch_size

        # metric handle
        metric_func = metric_handle(grid_type=grid_type, img_shape=(nlat, nlon), crop_shape=(nlat, nlon), crop_offset=(0, 0), crps_type="skillspread", normalize=True, channel_reduction=cred, batch_reduction=bred).to(self.device)

        # input and target:
        inp = torch.randn((batch_size, ensemble_size, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)
        tar = torch.randn((batch_size, num_channels, nlat, nlon), dtype=torch.float32, device=self.device)

        # full metric
        res_full = metric_func(inp, tar)
        counts_full = torch.tensor(inp.shape[0], dtype=torch.float32, device=self.device)
        res_full = metric_func.finalize(res_full, counts_full)

        # split and compute metrics stepwise
        inp_split = torch.split(inp, batch_size // 4, dim=0)
        tar_split = torch.split(tar, batch_size // 4, dim=0)

        res_split = metric_func(inp_split[0], tar_split[0])
        counts_split = torch.tensor(inp_split[0].shape[0], dtype=torch.float32, device=self.device)
        for inps, tars in zip(inp_split[1:], tar_split[1:]):
            res_tmp = metric_func(inps, tars)
            res_tmp = torch.stack([res_split, res_tmp], dim=0)
            counts_tmp = torch.tensor(inps.shape[0], dtype=torch.float32, device=self.device)
            counts_tmp = torch.stack([counts_split, counts_tmp], dim=0)
            res_tmp, counts_tmp = metric_func.combine(res_tmp, counts_tmp, dim=0)
            with torch.no_grad():
                res_split.copy_(res_tmp)
                counts_split.copy_(counts_tmp)

        res_split = metric_func.finalize(res_split, counts_split)

        # compare
        self.assertTrue(np.allclose(res_full.cpu().numpy(), res_split.cpu().numpy(), rtol=1e-6, atol=1e-6))

        
class TestMetricsHandler(unittest.TestCase):
    """
    A set of tests for the metrics handler
    """
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        self.params = get_default_parameters()
        self.params["dhours"] = 1

        # generating the image logic that is typically used by the dataloader
        self.params.img_shape_x = 17
        self.params.img_shape_y = 32
        self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
        self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
        self.params.img_local_offset_x = 0
        self.params.img_local_offset_y = 0
        self.params.img_crop_offset_x = 0
        self.params.img_crop_offset_y = 0

        return

    @parameterized.expand(_metric_handler_params[:1], skip_on_empty=True)
    def test_save_metrics(self, grid_type, batch_size, ensemble_size, num_rollout_steps, bred):

        # create dummy climatology
        num_steps = 1
        num_channels = len(self.params.channel_names)
        clim = torch.zeros(1, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)

        # update parameters
        self.params.batch_size = batch_size
        self.params.ensemble_size = ensemble_size

        metric_handler = MetricsHandler(self.params,
                                        clim,
                                        num_rollout_steps,
                                        self.device,
                                        l1_var_names=self.params.channel_names,
                                        rmse_var_names=self.params.channel_names,
                                        acc_var_names=self.params.channel_names,
                                        crps_var_names=self.params.channel_names,
                                        spread_var_names=self.params.channel_names,
                                        ssr_var_names=self.params.channel_names,
                                        rh_var_names=self.params.channel_names,
                                        wb2_compatible=False)
        metric_handler.initialize_buffers()
        metric_handler.zero_buffers()

        inplist = [torch.randn((num_rollout_steps, batch_size, ensemble_size, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y),
                               dtype=torch.float32, device=self.device) for _ in range(num_steps)]
        tarlist = [torch.randn((num_rollout_steps, batch_size, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y),
                               dtype=torch.float32, device=self.device) for _ in range(num_steps)]

        for inp, tar in zip(inplist, tarlist):
            for idt in range(num_rollout_steps):
                inpp = inp[idt, ...]
                tarp = tar[idt, ...]

                # super simple l1 loss
                loss = torch.mean(torch.abs(torch.mean(inpp, dim=1)-tarp))

                # update metric handler
                metric_handler.update(inpp, tarp, loss, idt)

        # finalize
        _ = metric_handler.finalize()

        with tempfile.TemporaryDirectory() as tempdir:
            outfile = os.path.join(tempdir, "metrics.h5")
            metric_handler.save(outfile)

            # make sure file gor written
            self.assertTrue(os.path.isfile(outfile))


    @parameterized.expand(_metric_handler_params, skip_on_empty=True)
    def test_aggregation(self, grid_type, batch_size, ensemble_size, num_rollout_steps, bred):

        # create dummy climatology
        num_steps = 4
        num_channels = len(self.params.channel_names)
        clim = torch.zeros(1, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)

        # update parameters
        self.params.batch_size = batch_size
        self.params.ensemble_size = ensemble_size
        
        metric_handler = MetricsHandler(self.params,
                                        clim,
                                        num_rollout_steps,
                                        self.device,
                                        l1_var_names=self.params.channel_names,
                                        rmse_var_names=self.params.channel_names,
                                        acc_var_names=self.params.channel_names,
                                        crps_var_names=self.params.channel_names,
                                        spread_var_names=self.params.channel_names,
                                        ssr_var_names=self.params.channel_names,
                                        rh_var_names=self.params.channel_names,
                                        wb2_compatible=False)
        metric_handler.initialize_buffers()
        metric_handler.zero_buffers()

        metric_handler_split = MetricsHandler(self.params,
                                              clim,
                                              num_rollout_steps,
                                              self.device,
                                              l1_var_names=self.params.channel_names,
                                              rmse_var_names=self.params.channel_names,
                                              acc_var_names=self.params.channel_names,
                                              crps_var_names=self.params.channel_names,
                                              spread_var_names=self.params.channel_names,
                                              ssr_var_names=self.params.channel_names,
                                              rh_var_names=self.params.channel_names,
                                              wb2_compatible=False)
        metric_handler_split.initialize_buffers()
        metric_handler_split.zero_buffers()

        inplist = [torch.randn((num_rollout_steps, batch_size, ensemble_size, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y),
                               dtype=torch.float32, device=self.device) for _ in range(num_steps)]
        tarlist = [torch.randn((num_rollout_steps, batch_size, num_channels, self.params.img_local_shape_x, self.params.img_local_shape_y),
                               dtype=torch.float32, device=self.device) for _ in range(num_steps)]

        loss_acc = 0.
        for inp, tar in zip(inplist, tarlist):
            for idt in range(num_rollout_steps):
                inpp = inp[idt, ...]
                tarp = tar[idt, ...]
                
                # super simple l1 loss
                loss = torch.mean(torch.abs(torch.mean(inpp, dim=1)-tarp))
                if idt == 0:
                    loss_acc += loss.item()

                # update metric handler
                metric_handler.update(inpp, tarp, loss, idt)

        # finalize
        loss_acc /= float(num_steps)
        logs_full = metric_handler.finalize()

        # compare loss
        mloss_full = logs_full["base"]["validation loss"]
        self.assertTrue(np.allclose(mloss_full, loss_acc))

        # metric handler split:
        inplist_split = list(itertools.chain.from_iterable([torch.split(tens, batch_size // 2, dim=1) for tens in inplist]))
        tarlist_split = list(itertools.chain.from_iterable([torch.split(tens, batch_size // 2, dim=1) for tens in tarlist]))

        for inp, tar in zip(inplist_split, tarlist_split):
            for idt in range(num_rollout_steps):
                inpp = inp[idt, ...]
                tarp = tar[idt, ...]

                # super simple l1 loss
                loss = torch.mean(torch.abs(torch.mean(inpp, dim=1)-tarp))

                # update metric handler
                metric_handler_split.update(inpp, tarp, loss, idt)

        # finalize
        logs_split = metric_handler_split.finalize()
        
        # compare loss
        mloss_split = logs_split["base"]["validation loss"]
        self.assertTrue(np.allclose(mloss_split, loss_acc))

        # extract dicts
        metrics_full = logs_full["metrics"]
        metrics_split = logs_split["metrics"]

        # compare scalar metrics
        for key in  metrics_full.keys():
            if key == "rollouts":
                continue
            val_full = metrics_full[key]
            val_split = metrics_split[key]
            self.assertTrue(np.allclose(val_full, val_split))

        # compare rollouts
        rollouts_full = logs_full["metrics"]["rollouts"]
        rollouts_split = logs_split["metrics"]["rollouts"]

        # aggregate table into data
        data_full = []
        for row in rollouts_full.data:
            data_full.append(row[-1])
        data_full = np.array(data_full)

        data_split = []
        for row in rollouts_split.data:
            data_split.append(row[-1])
        data_split = np.array(data_split)

        self.assertTrue(np.allclose(data_full, data_split))


# TODO: ssr test comparing to xskillcore
    
@unittest.skipIf(wb2 is None, "test requires weatherbench2 installation")
class ComparetMetricsWB2(unittest.TestCase):
    """
    A set of tests that compare weatherbench2 metrics to makani metrics
    """

    def setUp(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

    # same as above but compare to wb2
    @parameterized.expand(_wb2_metrics_params, skip_on_empty=True)
    def test_weighted_crps(self, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon):

        # CRPS handle
        crps_func = GeometricCRPS(
            img_shape=(nlat, nlon), crop_shape=(nlat, nlon), crop_offset=(0, 0), channel_names=[], grid_type=grid_type, crps_type="skillspread", channel_reduction="none", batch_reduction="none"
        ).to(self.device)

        # generate toy data
        obs = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        fct = torch.randn(batch_size, ensemble_size, num_channels, nlat, nlon, device=self.device)

        # create xarray datasets according to wb2 specification
        latitude = np.linspace(-90, 90, nlat, endpoint=True)
        longitude = np.linspace(0, 360, nlon, endpoint=False)
        xr_obs = xr.Dataset(
            data_vars=dict(var=(["batch", "channel", "latitude", "longitude"], obs.cpu().numpy())),
            coords=dict(latitude=latitude, longitude=longitude),
        )
        xr_fct = xr.Dataset(
            data_vars=dict(var=(["batch", "ensemble", "channel", "latitude", "longitude"], fct.cpu().numpy())),
            coords=dict(latitude=latitude, longitude=longitude),
        )

        # compute and compare CRPS
        crps = crps_func(fct, obs).cpu().numpy()
        crps_wb2 = wb2.metrics.CRPS(ensemble_dim="ensemble").compute_chunk(xr_fct, xr_obs, region=None)["var"].to_numpy()

        rdiff = np.abs(crps - crps_wb2) / np.abs(crps_wb2)
        self.assertTrue(rdiff.mean() <= 5e-4)
        self.assertTrue(np.allclose(crps, crps_wb2, rtol=5e-4, atol=0))

    # same as above but compare to wb2
    @parameterized.expand(_wb2_metrics_params, skip_on_empty=True)
    def test_weighted_ssr(self, grid_type, batch_size, ensemble_size, num_channels, nlat, nlon):

        # CRPS handle
        ssr_func = GeometricSSR(
            img_shape=(nlat, nlon), crop_shape=(nlat, nlon), crop_offset=(0, 0), channel_names=[], grid_type=grid_type, channel_reduction="none", batch_reduction="none"
        ).to(self.device)

        # generate toy data
        obs = torch.randn(batch_size, num_channels, nlat, nlon, device=self.device)
        fct = torch.randn(batch_size, ensemble_size, num_channels, nlat, nlon, device=self.device)

        # create xarray datasets according to wb2 specification
        latitude = np.linspace(-90, 90, nlat, endpoint=True)
        longitude = np.linspace(0, 360, nlon, endpoint=False)
        xr_obs = xr.Dataset(
            data_vars=dict(var=(["batch", "channel", "latitude", "longitude"], obs.cpu().numpy())),
            coords=dict(latitude=latitude, longitude=longitude),
        )
        xr_fct = xr.Dataset(
            data_vars=dict(var=(["batch", "ensemble", "channel", "latitude", "longitude"], fct.cpu().numpy())),
            coords=dict(latitude=latitude, longitude=longitude),
        )

        # compute and compare ssr
        ssr = ssr_func(fct, obs).cpu().numpy()
        ssr_skill_wb2 = wb2.metrics.EnergyScoreSkill(ensemble_dim="ensemble").compute_chunk(xr_fct, xr_obs, region=None)["var"].to_numpy()
        ssr_spread_wb2 = wb2.metrics.EnergyScoreSpread(ensemble_dim="ensemble").compute_chunk(xr_fct, xr_obs, region=None)["var"].to_numpy()
        ssr_wb2 = ssr_spread_wb2 / ssr_skill_wb2

        rdiff = np.abs(ssr - ssr_wb2) / np.abs(ssr_wb2)
        self.assertTrue(rdiff.mean() <= 1e-2)
        self.assertTrue(np.allclose(ssr, ssr_wb2, rtol=2e-2, atol=0))


if __name__ == "__main__":
    unittest.main()
