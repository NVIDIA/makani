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

"""
Model package for easy inference/packaging. Model packages contain all the necessary data to
perform inference and its interface is compatible with earth2mip
"""
import os
import shutil
import json
import jsbeautifier
import numpy as np
import torch
from makani.utils.YParams import ParamsBase
from makani.utils.driver import Driver
from makani.third_party.climt.zenith_angle import cos_zenith_angle
from makani.utils.dataloaders.data_helpers import get_data_normalization
from makani.models import model_registry
import datetime
import logging


logger = logging.getLogger(__name__)



class LocalPackage:
    """
    Implements the earth2mip/modulus Package interface.
    """

    # These define the model package in terms of where makani expects the files to be located
    THIS_MODULE = "makani.models.model_package"
    MODEL_PACKAGE_CHECKPOINT_PATH = "training_checkpoints/best_ckpt_mp0.tar"
    MINS_FILE = "mins.npy"
    MAXS_FILE = "maxs.npy"
    MEANS_FILE = "global_means.npy"
    STDS_FILE = "global_stds.npy"
    OROGRAPHY_FILE = "orography.nc"
    LANDMASK_FILE = "land_mask.nc"
    SOILTYPE_FILE = "soil_type.nc"

    def __init__(self, root):
        self.root = root

    def get(self, path):
        return os.path.join(self.root, path)

    @staticmethod
    def _load_static_data(package, params):
        if params.get("add_orography", False):
            params.orography_path = package.get(LocalPackage.OROGRAPHY_FILE)
        if params.get("add_landmask", False):
            params.landmask_path = package.get(LocalPackage.LANDMASK_FILE)
        if params.get("add_soiltype", False):
            params.soiltype_path = package.get(LocalPackage.SOILTYPE_FILE)

        # alweays load all normalization files
        if params.get("global_means_path", None) is not None:
            params.global_means_path = package.get(LocalPackage.MEANS_FILE)
        if params.get("global_stds_path", None) is not None:
            params.global_stds_path = package.get(LocalPackage.STDS_FILE)
        if params.get("min_path", None) is not None:
            params.min_path = package.get(LocalPackage.MINS_FILE)
        if params.get("max_path", None) is not None:
            params.max_path = package.get(LocalPackage.MAXS_FILE)


class ModelWrapper(torch.nn.Module):
    """
    Model wrapper to make inference simple outside of makani.

    Attributes
    ----------
    model : torch.nn.Module
        ML model that is wrapped.
    params : ParamsBase
        parameter object containing information on how the model was initialized in makani

    Methods
    -------
    forward(x, time):
        performs a single prediction steps
    """

    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.params = params
        nlat = params.img_shape_x
        nlon = params.img_shape_y

        # configure lats
        if "lat" in self.params:
            self.lats = np.asarray(self.params.lat)
        else:
            self.lats = np.linspace(90, -90, nlat, endpoint=True)

        # configure lons
        if "lon" in self.params:
            self.lons =	np.asarray(self.params.lon)
        else:
            self.lons = np.linspace(0, 360, nlon, endpoint=False)

        # zenith angle
        self.add_zenith = params.get("add_zenith", False)
        if self.add_zenith:
            self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)

        # load the normalization files
        bias, scale = get_data_normalization(self.params)

        # convert them to torch
        in_bias = torch.as_tensor(bias[:, self.params.in_channels]).to(torch.float32)
        in_scale = torch.as_tensor(scale[:, self.params.in_channels]).to(torch.float32)
        out_bias = torch.as_tensor(bias[:, self.params.out_channels]).to(torch.float32)
        out_scale = torch.as_tensor(scale[:, self.params.out_channels]).to(torch.float32)

        self.register_buffer("in_bias", in_bias)
        self.register_buffer("in_scale", in_scale)
        self.register_buffer("out_bias", out_bias)
        self.register_buffer("out_scale", out_scale)

    @property
    def in_channels(self):
        return self.params.get("channel_names", None)

    @property
    def out_channels(self):
        return self.params.get("channel_names", None)

    @property
    def timestep(self):
        return self.params.dt * self.params.dhours

    def replace_state(self, replace_state=True):
        self.model.preprocessor.update_internal_state(replace_state=True)
        return
    
    def set_rng(self, reset = True, seed=333):
        self.model.preprocessor.set_rng(reset, seed)
        return
        
    def forward(self, x, time, normalized_data=True, replace_state=None):
        if not normalized_data:
            x = (x - self.in_bias) / self.in_scale

        if self.add_zenith:
            cosz = cos_zenith_angle(time, self.lon_grid, self.lat_grid)
            cosz = cosz.astype(np.float32)
            z = torch.as_tensor(cosz).to(device=x.device)
            while z.ndim != x.ndim:
                z = z[None]
            self.model.preprocessor.cache_unpredicted_features(None, None, xz=z, yz=None)

        out = self.model(x, replace_state=replace_state)

        if not normalized_data:
            out = out * self.out_scale + self.out_bias

        return out


def save_model_package(params):
    """
    Saves out a self-contained model-package.
    The idea is to save anything necessary for inference beyond the checkpoints in one location.
    """
    # save out the current state of the parameters, make it human readable
    config_path = os.path.join(params.experiment_dir, "config.json")
    jsopts = jsbeautifier.default_options()
    jsopts.indent_size = 2

    with open(config_path, "w") as f:
        msg = jsbeautifier.beautify(json.dumps(params.to_dict()), jsopts)
        f.write(msg)

    if params.get("add_orography", False):
        shutil.copy(params.orography_path, os.path.join(params.experiment_dir, os.path.basename(params.orography_path)))

    if params.get("add_landmask", False):
        shutil.copy(params.landmask_path, os.path.join(params.experiment_dir, os.path.basename(params.landmask_path)))

    if params.get("add_soiltype", False):
        shutil.copy(params.soiltype_path, os.path.join(params.experiment_dir, os.path.basename(params.soiltype_path)))

    # always save out all normalization files
    if params.get("global_means_path", None) is not None:
        shutil.copy(params.global_means_path, os.path.join(params.experiment_dir, os.path.basename(params.global_means_path)))
    if params.get("global_stds_path", None) is not None:
        shutil.copy(params.global_stds_path, os.path.join(params.experiment_dir, os.path.basename(params.global_stds_path)))
    if params.get("min_path", None) is not None:
        shutil.copy(params.min_path, os.path.join(params.experiment_dir, os.path.basename(params.min_path)))
    if params.get("max_path", None) is not None:
        shutil.copy(params.max_path, os.path.join(params.experiment_dir, os.path.basename(params.max_path)))

    # write out earth2mip metadata.json
    fcn_mip_data = {
        "entrypoint": {"name": f"{LocalPackage.THIS_MODULE}:load_time_loop"},
    }
    with open(os.path.join(params.experiment_dir, "metadata.json"), "w") as f:
        msg = jsbeautifier.beautify(json.dumps(fcn_mip_data), jsopts)
        f.write(msg)


# TODO: this is not clean and should be reworked to allow restoring from params + checkpoint file
def load_model_package(package, pretrained=True, device="cpu", multistep=False):
    """
    Loads model package and return the wrapper which can be used for inference.
    """
    path = package.get("config.json")
    params = ParamsBase.from_json(path)
    LocalPackage._load_static_data(package, params)

    # assume we are not distributed
    # distributed checkpoints might be saved with different params values
    params.img_local_offset_x = 0
    params.img_local_offset_y = 0
    params.img_local_shape_x = params.img_shape_x
    params.img_local_shape_y = params.img_shape_y

    # get the model and
    model = model_registry.get_model(params, multistep=multistep).to(device)

    if pretrained:
        best_checkpoint_path = package.get(LocalPackage.MODEL_PACKAGE_CHECKPOINT_PATH)
        Driver.restore_from_checkpoint(best_checkpoint_path, model)

    model = ModelWrapper(model, params=params)

    # by default we want to do evaluation so setting it to eval here
    model.eval()

    return model


def load_time_loop(package, device=None, time_step_hours=None):
    """This function loads an earth2mip TimeLoop object that
    can be used for inference.

    A TimeLoop encapsulates normalization, regridding, and other logic, so is a
    very minimal interface to expose to a framework like earth2mip.

    See https://github.com/NVIDIA/earth2mip/blob/main/docs/concepts.rst
    for more info on this interface.
    """

    from earth2mip.networks import Inference
    from earth2mip.grid import equiangular_lat_lon_grid
    from physicsnemo.distributed.manager import DistributedManager

    config = package.get("config.json")
    params = ParamsBase.from_json(config)

    if params.in_channels != params.out_channels:
        raise NotImplementedError("Non-equal input and output channels are not implemented yet.")

    names = [params.data_channel_names[i] for i in params.in_channels]
    params.min_path = package.get(LocalPackage.MINS_FILE)
    params.max_path = package.get(LocalPackage.MAXS_FILE)
    params.global_means_path = package.get(LocalPackage.MEANS_FILE)
    params.global_stds_path = package.get(LocalPackage.STDS_FILE)

    center, scale = get_data_normalization(params)

    model = load_model_package(package, pretrained=True, device=device)
    shape = (params.img_crop_shape_x, params.img_crop_shape_y)

    # TODO: insert a check to see if the grid e2mip computes is the same that makani uses
    grid = equiangular_lat_lon_grid(nlat=params.img_crop_shape_x, nlon=params.img_crop_shape_y, includes_south_pole=True)

    if time_step_hours is None:
        hour = datetime.timedelta(hours=1)
        time_step = hour * params.get("dt", 6)
    else:
        time_step = datetime.timedelta(hours=time_step_hours)

    # Here we use the built-in class earth2mip.networks.Inference
    # will later be extended to use the makani inferencer
    inference = Inference(
        model=model,
        channel_names=names,
        center=center[:, params.in_channels],
        scale=scale[:, params.out_channels],
        grid=grid,
        n_history=params.n_history,
        time_step=time_step,
    )
    inference.to(device)
    return inference
