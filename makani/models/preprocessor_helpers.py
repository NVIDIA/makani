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

import os
from functools import partial
import torch

def get_bias_correction(params):
    if params.get("bias_correction", None) is not None:
        from makani.utils.auxiliary_fields import get_bias_correction

        # set up sharding parameters
        start_x = params.get("img_local_offset_x", 0)
        end_x = min(start_x + params.get("img_local_shape_x", params.img_shape_x), params.img_shape_x)
        start_y = params.get("img_local_offset_y", 0)
        end_y = min(start_y + params.get("img_local_shape_y", params.img_shape_y), params.img_shape_y)

        bc_path = params.get("bias_correction", None)
        if not os.path.isfile(bc_path):
            raise IOError(f"Specify a valid bias correction path, got {bc_path}")

        bias = torch.as_tensor(get_bias_correction(bc_path, params.get("out_channels")), dtype=torch.float32)

        # shard the bias correction
        bias = bias[:, :, start_x:end_x, start_y:end_y]

    else:
        bias = None

    return bias


def get_static_features(params):

    # set up normalizer
    normalize_static_features = params.get("normalize_static_features", False)
    normalizer = None
    if normalize_static_features:
        from makani.utils.grids import grid_to_quadrature_rule, GridQuadrature

        quadrature_rule = grid_to_quadrature_rule(params.get("data_grid_type", "equiangular"))
        crop_shape = [params.get("img_crop_shape_x", params.img_shape_x), params.get("img_crop_shape_y", params.img_shape_y)]
        crop_offset = [params.get("img_crop_offset_x", 0), params.get("img_crop_offset_y", 0)]

        quadrature = GridQuadrature(quadrature_rule, img_shape=params.img_shape, crop_shape=crop_shape, crop_offset=crop_offset, normalize=True, distributed=False)

        def normalize(tensor, eps=0.0):
            mean = quadrature(tensor).reshape(1, -1, 1, 1)
            std = torch.sqrt(quadrature(torch.square(tensor - mean)).reshape(1, -1, 1, 1))
            tensor = (tensor - mean) / (std + eps)
            return tensor

        normalizer = partial(normalize, eps=0.0)

    # set up sharding parameters
    start_x = params.get("img_local_offset_x", 0)
    end_x = min(start_x + params.get("img_local_shape_x", params.img_shape_x), params.img_shape_x)
    start_y = params.get("img_local_offset_y", 0)
    end_y = min(start_y + params.get("img_local_shape_y", params.img_shape_y), params.img_shape_y)

    static_features = None
    if params.get("add_grid", False):
        with torch.no_grad():
            if hasattr(params, "lat") and hasattr(params, "lon"):
                from makani.utils.grids import GridConverter

                lat = torch.as_tensor(params.lat).to(torch.float32)
                lon = torch.as_tensor(params.lon).to(torch.float32)

                # convert grid if required
                gconv = GridConverter(params.data_grid_type, params.model_grid_type, torch.deg2rad(lat), torch.deg2rad(lon))
                tx, ty = gconv.get_dst_coords()
                tx = tx.to(torch.float32)
                ty = ty.to(torch.float32)
            else:
                tx = torch.linspace(0, 1, params.img_shape_x + 1, dtype=torch.float32)[0:-1]
                ty = torch.linspace(0, 1, params.img_shape_y + 1, dtype=torch.float32)[0:-1]

            x_grid, y_grid = torch.meshgrid(tx, ty, indexing="ij")
            x_grid, y_grid = x_grid.unsqueeze(0).unsqueeze(0), y_grid.unsqueeze(0).unsqueeze(0)
            grid = torch.cat([x_grid, y_grid], dim=1)

            # transform if requested
            gridtype = params.get("gridtype", "sinusoidal")
            if gridtype == "sinusoidal":
                num_freq = params.get("grid_num_frequencies", 1)

                singrid = None
                for freq in range(1, num_freq + 1):
                    if singrid is None:
                        singrid = [torch.sin(grid), torch.cos(grid)]
                    else:
                        singrid = singrid + [torch.sin(freq * grid), torch.cos(freq * grid)]

                static_features = torch.cat(singrid, dim=-3)
            else:
                static_features = grid

            # normalize if requested
            if normalizer is not None:
                static_features = normalizer(static_features)

            # shard spatially
            static_features = static_features[:, :, start_x:end_x, start_y:end_y]

    if params.get("add_orography", False):
        from makani.utils.auxiliary_fields import get_orography

        orography_path = params.get("orography_path", None)
        if not os.path.isfile(orography_path):
            raise IOError(f"Specify a valid orography path, got {orography_path}")

        with torch.no_grad():
            oro = torch.as_tensor(get_orography(orography_path), dtype=torch.float32)
            oro = torch.reshape(oro, (1, 1, oro.shape[0], oro.shape[1]))

            eps = 1e-6
            if normalizer is not None:
                oro = normalizer(oro, eps=eps)
            else:
                oro = (oro - torch.mean(oro)) / (torch.std(oro) + eps)

            # shard
            oro = oro[:, :, start_x:end_x, start_y:end_y]

            if static_features is None:
                static_features = oro
            else:
                static_features = torch.cat([static_features, oro], dim=1)

    if params.get("add_landmask", False):
        from makani.utils.auxiliary_fields import get_land_mask

        landmask_path = params.get("landmask_path", None)
        if not os.path.isfile(landmask_path):
            raise IOError(f"Specify a valid landmask path, got {landmask_path}")

        landmask_preprocessing = params.get("landmask_preprocessing", "floor")
        with torch.no_grad():
            if landmask_preprocessing == "floor":
                lsm = torch.as_tensor(get_land_mask(landmask_path), dtype=torch.long)
                # one hot encode and move channels to front:
                lsm = torch.permute(torch.nn.functional.one_hot(lsm), (2, 0, 1)).to(torch.float32)
                lsm = torch.reshape(lsm, (1, lsm.shape[0], lsm.shape[1], lsm.shape[2]))
            elif landmask_preprocessing == "round":
                lsm = torch.as_tensor(get_land_mask(landmask_path), dtype=torch.float32).round()
                lsm = lsm.to(torch.long)
                # one hot encode and move channels to front:
                lsm = torch.permute(torch.nn.functional.one_hot(lsm), (2, 0, 1)).to(torch.float32)
                lsm = torch.reshape(lsm, (1, lsm.shape[0], lsm.shape[1], lsm.shape[2]))
            elif landmask_preprocessing == "raw":
                lsm = torch.as_tensor(get_land_mask(landmask_path), dtype=torch.float32)
                lsm = torch.reshape(lsm, (1, 1, lsm.shape[0], lsm.shape[1]))

            # no normalization since the data is one hot encoded

            # shard
            lsm = lsm[:, :, start_x:end_x, start_y:end_y]

            if static_features is None:
                static_features = lsm
            else:
                static_features = torch.cat([static_features, lsm], dim=1)

    if params.get("add_soiltype", False):
        from makani.utils.auxiliary_fields import get_soiltype

        soiltype_path = params.get("soiltype_path", None)
        if not os.path.isfile(soiltype_path):
            raise IOError(f"Specify a valid soiltype path, got {soiltype_path}")

        with torch.no_grad():
            st = torch.as_tensor(get_soiltype(soiltype_path), dtype=torch.long)

            # one hot encode and move channels to front:
            st = torch.permute(torch.nn.functional.one_hot(st), (2, 0, 1)).to(torch.float32)
            st = torch.reshape(st, (1, st.shape[0], st.shape[1], st.shape[2]))

            # no normalization since the data is one hot encoded

            # shard
            st = st[:, :, start_x:end_x, start_y:end_y]

            if static_features is None:
                static_features = st
            else:
                static_features = torch.cat([static_features, st], dim=1)

    if params.get("add_copernicus_emb", False):
        from makani.utils.auxiliary_fields import get_copernicus_emb

        copernicus_emb_path = params.get("copernicus_emb_path", None)
        if not os.path.isfile(copernicus_emb_path):
            raise IOError(f"Specify a valid copernicus embedding path, got {copernicus_emb_path}")

        with torch.no_grad():
            emb = get_copernicus_emb(copernicus_emb_path)

            # one hot encode and move channels to front:
            emb = torch.permute(emb, (2, 0, 1))
            emb = torch.reshape(emb, (1, emb.shape[0], emb.shape[1], emb.shape[2]))

            # no normalization since the data is already in the right format
            eps = 1e-6
            if normalizer is not None:
                emb = normalizer(emb, eps=eps)
            else:
                emb = (emb - torch.mean(emb)) / (torch.std(emb) + eps)

            # shard
            emb = emb[:, :, start_x:end_x, start_y:end_y]

            if static_features is None:
                static_features = emb
            else:
                static_features = torch.cat([static_features, emb], dim=1)

    return static_features