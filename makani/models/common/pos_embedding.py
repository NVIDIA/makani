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

import abc

import torch
import torch.nn as nn
from torch import amp

import torch_harmonics as th
import torch_harmonics.distributed as thd
from torch_harmonics.sht import truncate_sht

from makani.distributed import comm
from torch_harmonics.distributed import compute_split_shapes


class PositionEmbedding(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for position embeddings.

    This class defines the interface for position embedding modules
    that add positional information to input tensors.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2 or makani.utils.grids.PlanarGrid
        The grid of the embedding, see :func:`~makani.utils.grids.make_grid`.
    num_chans : int, optional
        Number of channels, by default 1
    """

    def __init__(self, grid, num_chans=1):
        super().__init__()

        self.grid = grid
        self.img_shape = tuple(grid.shape)
        self.num_chans = num_chans

    def forward(self):
        r"""
        Return the position embedding tensor.

        Takes no input: the embedding depends only on grid position, so callers
        add the returned tensor to their features themselves. Subclasses are
        expected to override this if the embedding needs to be materialized or
        broadcast before use.

        Returns
        -------
        torch.Tensor
            Embedding of shape ``(1, num_chans, H, W)``, broadcastable against a
            feature tensor on the same grid.
        """

        return self.position_embeddings


class LearnablePositionEmbedding(PositionEmbedding):
    r"""
    Learnable position embeddings for spherical transformers.

    This module provides learnable position embeddings that can be either
    latitude-only or full latitude-longitude embeddings.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2 or makani.utils.grids.PlanarGrid
        The grid of the embedding, see :func:`~makani.utils.grids.make_grid`.
    num_chans : int, optional
        Number of channels, by default 1
    embed_type : str, optional
        Embedding type ("lat" or "latlon"), by default "lat"

    Raises
    ------
    ValueError
        If ``embed_type`` is neither ``"lat"`` nor ``"latlon"``.

    Notes
    -----
    ``"lat"`` learns one vector per latitude ring and broadcasts it along
    longitude. This is the natural choice on a sphere: the physics is
    zonally symmetric, so tying parameters along longitude cuts the parameter
    count by a factor of ``nlon`` without giving up the pole-to-equator
    structure the model actually needs. ``"latlon"`` learns an independent
    vector per grid point and is only worth it when longitude-dependent
    features (land-sea contrast, orography) must be memorized.

    Under model parallelism the embedding is allocated at each rank's *local*
    shape and tagged with its sharding, so it is sharded over ``h`` (and over
    ``w`` for ``"latlon"``) rather than replicated. For ``"lat"`` the parameter
    is marked shared across the ``w`` group, which keeps gradients consistent
    across the ranks that hold the same latitudes.
    """

    def __init__(self, grid, num_chans=1, embed_type="lat"):
        super().__init__(grid=grid, num_chans=num_chans)
        img_shape = self.img_shape

        # if distributed, make sure to split correctly across ranks:
        # in case of model parallelism, we need to make sure that we use the correct shapes per rank
        # for h
        if comm.get_size("h") > 1:
            self.local_shape_h = compute_split_shapes(img_shape[0], comm.get_size("h"))[comm.get_rank("h")]
        else:
            self.local_shape_h = img_shape[0]

        # for w
        if comm.get_size("w") > 1:
            self.local_shape_w = compute_split_shapes(img_shape[1], comm.get_size("w"))[comm.get_rank("w")]
        else:
            self.local_shape_w = img_shape[1]

        if embed_type == "latlon":
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, self.num_chans, self.local_shape_h, self.local_shape_w)
            )
            self.position_embeddings.is_shared_mp = {"matmul": "avg"}
            self.position_embeddings.sharded_dims_mp = [None, None, "h", "w"]
        elif embed_type == "lat":
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_chans, self.local_shape_h, 1))
            self.position_embeddings.is_shared_mp = {"w": "sum", "matmul": "avg"}
            self.position_embeddings.sharded_dims_mp = [None, None, "h", None]
        else:
            raise ValueError(f"Unknown learnable position embedding type {embed_type}")

    def no_weight_decay(self):
        r"""
        Parameters that should be excluded from weight decay: the embedding, which encodes location rather than a
        learned transformation, see :func:`~makani.driver.helpers.get_no_weight_decay_parameters`.

        Returns
        -------
        set of str
            Names of the parameters to exclude, relative to the layer.
        """
        return {"position_embeddings"}

    def forward(self):
        r"""
        Return the learned embedding, expanded to the local grid shape.

        For ``embed_type="lat"`` the stored parameter has a singleton longitude
        dimension; expanding it here gives callers a tensor matching the local
        feature map without materializing ``nlon`` copies of the parameter.

        Returns
        -------
        torch.Tensor
            Embedding of shape ``(1, num_chans, local_shape_h, local_shape_w)``.
            This is a view of the parameter, not a copy.
        """
        return self.position_embeddings.expand(-1, -1, self.local_shape_h, self.local_shape_w)


class SpectralPositionEmbedding(PositionEmbedding):
    r"""
    A learnable position embedding of band-limited spherical harmonics.

    The embedding is the inverse spherical harmonic transform of learned coefficients :math:`c_{lm}` of the degrees
    :math:`l < l_{max}` and orders :math:`m < m_{max}`, a vector of ``num_chans`` channels each:

    .. math::

        e(\theta, \lambda) = \sum_{l < l_{max}} \sum_{m < \min(l + 1, m_{max})} c_{lm} \, Y_l^m(\theta, \lambda).

    With ``mmax=1``, the default, only the zonal harmonics :math:`Y_l^0` remain, and the embedding depends on the
    latitude alone. Unlike an embedding of one free vector per latitude ring
    (:class:`LearnablePositionEmbedding` with ``embed_type="lat"``), it is smooth by construction, also towards the
    poles, whose rings have the least area, and so the least gradient, but which an optimizer such as Adam steps as
    far as the others. Band-limited, it changes between neighboring rings by no more than its degrees allow, at the
    poles as anywhere; choose ``lmax`` well below the degrees of the grid for it to be smooth on the grid. The zonal
    harmonics peak at the poles, :math:`|Y_l^0| = \sqrt{(2l+1)/(4\pi)}` there, so that the poles respond the most to
    the coefficients of high degrees. With ``mmax > 1``, the embedding depends on the longitude as well.

    The coefficients are zero-initialized, so that a network starts as without the embedding, and replicated over the
    ranks. Each spatial rank transforms its block of them, the degrees of its ``h`` rank and the orders of its ``w``
    rank, so that their gradient is summed over the spatial ranks. They are not weight decayed, see
    :meth:`no_weight_decay`.

    Parameters
    ----------
    grid : torch_harmonics.RegularGridS2
        The grid of the embedding, see :func:`~makani.utils.grids.make_grid`.
    num_chans : int, optional
        Number of channels, by default 1
    lmax : int, optional
        The number of degrees, :math:`l < l_{max}`, by default 32. At most the number of degrees the grid resolves.
    mmax : int, optional
        The number of orders, :math:`m < m_{max}`, by default 1, the zonal embedding. At most ``lmax``. Both are
        exclusive bounds, as in torch-harmonics: ``mmax=1`` keeps the order :math:`m = 0` only.

    Raises
    ------
    ValueError
        If ``lmax`` or ``mmax`` is not positive, ``mmax`` exceeds ``lmax``, or the grid does not resolve ``lmax``
        degrees.
    """

    def __init__(self, grid, num_chans=1, lmax=32, mmax=1):
        super().__init__(grid=grid, num_chans=num_chans)

        if (lmax < 1) or (mmax < 1) or (mmax > lmax):
            raise ValueError(f"The spectral position embedding needs 1 <= mmax <= lmax, got lmax={lmax}, mmax={mmax}")
        # the degrees the grid resolves, its default truncation: torch-harmonics builds transforms beyond it, which
        # alias
        resolved = truncate_sht(grid, None, None, None)
        if lmax > resolved.lmax:
            raise ValueError(
                f"The grid of shape {tuple(grid.shape)} resolves {resolved.lmax} degrees, fewer than the {lmax} of the "
                f"spectral position embedding"
            )
        self.lmax = lmax
        self.mmax = mmax

        # The distributed transforms of torch-harmonics split the degrees over the h ranks and the orders over the w
        # ranks unevenly if need be, but support no empty blocks: a zonal embedding has a single order, which would
        # leave all w ranks but one without. The transform keeps at least as many degrees and orders as ranks split
        # them, and as torch-harmonics keeps no more orders than degrees, the degrees cover the w ranks as well. The
        # coefficients beyond lmax and mmax are zero.
        lmax_transform = max(lmax, comm.get_size("h"), comm.get_size("w"))
        mmax_transform = max(mmax, comm.get_size("w"))
        if comm.get_size("spatial") > 1:
            if not thd.is_initialized():
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
            self.isht = thd.DistributedInverseRealSHT(grid, lmax=lmax_transform, mmax=mmax_transform)
            l_shapes, m_shapes = self.isht.l_shapes, self.isht.m_shapes
        else:
            self.isht = th.InverseRealSHT(grid, lmax=lmax_transform, mmax=mmax_transform)
            l_shapes, m_shapes = [self.isht.lmax], [self.isht.mmax]
        if (self.isht.lmax != lmax_transform) or (self.isht.mmax != mmax_transform):
            raise ValueError(
                f"The transform of the spectral position embedding keeps {self.isht.lmax} degrees and {self.isht.mmax} "
                f"orders instead of {lmax_transform} and {mmax_transform}"
            )

        # this rank's block of the coefficients of the transform
        self.l_start = sum(l_shapes[: comm.get_rank("h")])
        self.l_end = self.l_start + l_shapes[comm.get_rank("h")]
        self.m_start = sum(m_shapes[: comm.get_rank("w")])
        self.m_end = self.m_start + m_shapes[comm.get_rank("w")]
        self.lmax_transform = lmax_transform
        self.mmax_transform = mmax_transform

        # real and imaginary parts, zero so that the network starts as without the embedding
        self.coeffs = nn.Parameter(torch.zeros(self.num_chans, lmax, mmax, 2))
        # the same on every rank; each spatial rank contributes the gradient of its block
        self.coeffs.is_shared_mp = {"spatial": "sum", "matmul": "avg"}
        self.coeffs.sharded_dims_mp = [None, None, None, None]

        # only m <= l is a spherical harmonic, and m = 0 has no imaginary part
        l_idx = torch.arange(lmax).reshape(-1, 1, 1)
        m_idx = torch.arange(mmax).reshape(1, -1, 1)
        part = torch.arange(2).reshape(1, 1, -1)
        mask = (m_idx <= l_idx) & ((part == 0) | (m_idx > 0))
        self.register_buffer("mask", mask.to(torch.float32), persistent=False)

    def no_weight_decay(self):
        r"""
        Parameters that should be excluded from weight decay: the coefficients, as the embedding encodes location
        rather than a learned transformation, see :func:`~makani.driver.helpers.get_no_weight_decay_parameters`.

        Returns
        -------
        set of str
            Names of the parameters to exclude, relative to the layer.
        """
        return {"coeffs"}

    def forward(self):
        r"""
        Return the embedding on the local part of the grid.

        Returns
        -------
        torch.Tensor
            Embedding of shape ``(1, num_chans, nlat_local, nlon_local)``, in the dtype of the coefficients, also
            under autocast: callers cast it to the dtype of their features.
        """
        coeffs = self.coeffs * self.mask
        # the coefficients of the transform, zero beyond lmax and mmax, and this rank's block of them
        coeffs = nn.functional.pad(
            coeffs, (0, 0, 0, self.mmax_transform - self.mmax, 0, self.lmax_transform - self.lmax)
        )
        coeffs = coeffs[:, self.l_start : self.l_end, self.m_start : self.m_end, :]
        # in the dtype of the coefficients, which the transform follows, not in the lower precision of an autocast
        with amp.autocast(device_type=coeffs.device.type, enabled=False):
            return self.isht(torch.view_as_complex(coeffs.contiguous()).unsqueeze(0))

    def extra_repr(self) -> str:
        return f"num_chans={self.num_chans}, lmax={self.lmax}, mmax={self.mmax}"
