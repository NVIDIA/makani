# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# This file contains code from the climt project (https://github.com/CliMT/climt)
# which is licensed under BSD-3-Clause. Original copyright: (c) 2016 Rodrigo Caballero.
# Modified 2024 by NVIDIA: vectorization over coordinates and restructuring for
# dataloader performance.

import datetime as dt
import numpy as np
from typing import Tuple

dtype = np.float32

def _days_from_2000(model_time: np.ndarray) -> np.ndarray:
    """Days since 2000-01-01 12:00 UTC."""
    time_diff = model_time - dt.datetime(2000, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    return (np.asarray(time_diff).astype("timedelta64[us]") / np.timedelta64(1, "D")).astype(dtype)


def _greenwich_mean_sidereal_time(jc: np.ndarray) -> np.ndarray:
    """GMST in radians. `jc` = julian centuries since 2000."""
    theta = dtype(
        67310.54841
        + jc * (876600 * 3600 + 8640184.812866 + jc * (0.093104 - jc * 6.2e-5))
    )
    return (np.deg2rad(theta / 240.0) % (2 * np.pi)).astype(dtype)


def _sun_ecliptic_longitude(jc: np.ndarray) -> np.ndarray:
    mean_anomaly = np.deg2rad(
        357.52910 + 35999.05030 * jc - 0.0001559 * jc * jc - 0.00000048 * jc * jc * jc,
        dtype=dtype,
    )
    mean_longitude = np.deg2rad(
        280.46645 + 36000.76983 * jc + 0.0003032 * jc * jc, dtype=dtype
    )
    d_l = np.deg2rad(
        (1.914600 - 0.004817 * jc - 0.000014 * jc * jc) * np.sin(mean_anomaly)
        + (0.019993 - 0.000101 * jc) * np.sin(2 * mean_anomaly)
        + 0.000290 * np.sin(3 * mean_anomaly),
        dtype=dtype,
    )
    return mean_longitude + d_l


def _obliquity_star(jc: np.ndarray) -> np.ndarray:
    return np.deg2rad(
        23.0 + 26.0 / 60 + 21.406 / 3600.0
        - (
            46.836769 * jc
            - 0.0001831 * jc ** 2
            + 0.00200340 * jc ** 3
            - 0.576e-6 * jc ** 4
            - 4.34e-8 * jc ** 5
        ) / 3600.0,
        dtype=dtype,
    )


def _time_scalars(model_time: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-only scalar quantities. Shape [T] each.

    Returns (sin_dec, cos_dec, gmst_minus_ra). Combining gmst and ra here means
    the downstream grid expression only has to add one broadcasted offset.
    """
    jc = _days_from_2000(model_time) / dtype(36525.0)
    eps = _obliquity_star(jc)
    eclon = _sun_ecliptic_longitude(jc)
    x = np.cos(eclon)
    y = np.cos(eps) * np.sin(eclon)
    z = np.sin(eps) * np.sin(eclon)
    r = np.sqrt(1.0 - z * z)
    dec = np.arctan2(z, r)
    ra = dtype(2.0) * np.arctan2(y, x + r)
    gmst = _greenwich_mean_sidereal_time(jc)
    return (
        np.sin(dec).astype(dtype),
        np.cos(dec).astype(dtype),
        (gmst - ra).astype(dtype),
    )


# Identity-based cache: dataloaders pass the same lat/lon arrays each call.
_grid_cache = {"lon": None, "lat": None, "lon_rad": None, "sin_lat": None, "cos_lat": None}


def _prep_grid(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lon is _grid_cache["lon"] and lat is _grid_cache["lat"]:
        return _grid_cache["lon_rad"], _grid_cache["sin_lat"], _grid_cache["cos_lat"]
    lon_rad = np.deg2rad(lon, dtype=dtype)
    lat_rad = np.deg2rad(lat, dtype=dtype)
    _grid_cache["lon"] = lon
    _grid_cache["lat"] = lat
    _grid_cache["lon_rad"] = lon_rad
    _grid_cache["sin_lat"] = np.sin(lat_rad).astype(dtype)
    _grid_cache["cos_lat"] = np.cos(lat_rad).astype(dtype)
    return _grid_cache["lon_rad"], _grid_cache["sin_lat"], _grid_cache["cos_lat"]


def cos_zenith_angle(
    time: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Cosine of sun-zenith angle at each UTC `time` on the (lon, lat) grid.

    Args:
        time: datetime array, shape [T] (scalars and 0-d arrays are accepted).
        lon, lat: degrees, any broadcastable shape (typically [H, W]).
    Returns:
        float32 array, shape [T, *lon.shape].
    """
    sin_dec, cos_dec, gmst_minus_ra = _time_scalars(np.atleast_1d(time))
    lon_rad, sin_lat, cos_lat = _prep_grid(lon, lat)

    t_shape = sin_dec.shape + (1,) * lon_rad.ndim
    sin_dec = sin_dec.reshape(t_shape)
    cos_dec = cos_dec.reshape(t_shape)
    gmst_minus_ra = gmst_minus_ra.reshape(t_shape)

    cos_h = np.cos(gmst_minus_ra + lon_rad)
    return sin_lat * sin_dec + cos_lat * cos_dec * cos_h
