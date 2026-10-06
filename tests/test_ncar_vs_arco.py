# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Cross check of the NCAR source against ARCO-ERA5, read through the WB2 source.

Both archives hold the same ECMWF reanalysis, but they were produced
independently and regridded differently, so they do not agree bitwise and a
fixed tolerance would say little. Instead each channel has to match ARCO far
better than a deliberately wrong counterpart does: ARCO six hours later, and
ARCO at the neighbouring pressure level. That is what the bugs this guards
against look like -- a timestep picked off by one, a level mixed up, channels
swapped -- while regridding noise affects the right and the wrong comparison
alike.

The test reads about a gigabyte from the public S3 and GCS buckets, so it only
runs when ``MAKANI_TEST_NETWORK=1`` is set. It prints the metrics, which is what
the margins below were meant to be calibrated from.

Total precipitation is left out: ARCO stores it hourly, while the WB2 source
expects the 6 hourly accumulation of the WeatherBench2 datasets.
"""

import os
import sys
import tempfile
import unittest
import importlib.util
import datetime as dt

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# one day at 6 hours plus the next 00Z, which crosses NCAR's daily pressure level files
TIMES = [dt.datetime(2023, 6, 15, tzinfo=dt.timezone.utc) + dt.timedelta(hours=6 * idx) for idx in range(5)]

# the channels compared, each with the neighbouring level used as wrong counterpart
CHANNELS = ["z500", "t850", "t2m", "u10m", "msl"]
NEIGHBOURS = {"z500": "z550", "t850": "t875"}

# ARCO also provides the neighbours; the WB2 source reads every pressure level
# variable on every level requested, so they come as the full product
ARCO_CHANNELS = [f"{var}{level}" for var in ["z", "t"] for level in [500, 550, 850, 875]] + ["t2m", "u10m", "msl"]

# the 0.25 degree grid both archives are on, latitude descending
LAT = np.linspace(90.0, -90.0, 721).tolist()
LON = (np.arange(1440) * 0.25).tolist()

# a channel has to match ARCO this well, normalized by ARCO's spread ...
MAX_ERROR = 0.05
# ... and this many times better than its wrong counterparts
MIN_MARGIN = 5.0


def _have_dependencies():
    return all(importlib.util.find_spec(name) is not None for name in ["s3fs", "gcsfs", "xarray", "zarr"])


def _convert(source, channels, workdir, name):
    """Fill ``TIMES`` from ``source`` into a serial output file and return fields and validity."""
    from data_process.convert_era5_to_makani_input import _batched, _create_output_file

    stamps = np.array([t.timestamp() for t in TIMES], dtype=np.float64)
    path = os.path.join(workdir, f"{name}.h5")
    with _create_output_file(path, None, "fields", stamps, channels, LAT, LON) as out:
        units = source.split_units(list(enumerate(TIMES)))
        source.begin_year(units)
        for batch in _batched(units, source.units_per_fill):
            source.fill(out, "fields", list(batch))
        fields, valid = out["fields"][...], out["valid_data"][...]
    source.close()
    return fields, valid


def _normalized_error(values, reference):
    """Area weighted RMSE of ``values`` against ``reference``, relative to the spread of ``reference``."""
    weights = np.broadcast_to(np.cos(np.deg2rad(np.asarray(LAT)))[:, None], reference.shape[-2:])
    weights = np.broadcast_to(weights, reference.shape)

    def mean(x):
        return np.sum(weights * x) / np.sum(weights)

    rmse = np.sqrt(mean((values - reference) ** 2))
    spread = np.sqrt(mean((reference - mean(reference)) ** 2))
    return float(rmse / spread)


@unittest.skipUnless(
    os.environ.get("MAKANI_TEST_NETWORK") == "1", "reads from public buckets, set MAKANI_TEST_NETWORK=1"
)
@unittest.skipUnless(_have_dependencies(), "needs s3fs, gcsfs, xarray and zarr")
class TestNcarVsArco(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from data_process.sources.ncar import NcarSource
        from data_process.sources.wb2 import Wb2Source

        cls._tmpdir = tempfile.TemporaryDirectory()

        ncar = NcarSource(cls._metadata(CHANNELS), 0)
        arco = Wb2Source(cls._metadata(ARCO_CHANNELS), 0, input_file=ARCO_STORE, batch_size=len(TIMES))

        cls.ncar, cls.ncar_valid = _convert(ncar, CHANNELS, cls._tmpdir.name, "ncar")
        cls.arco, cls.arco_valid = _convert(arco, ARCO_CHANNELS, cls._tmpdir.name, "arco")

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    @staticmethod
    def _metadata(channels):
        return {"dhours": 6, "coords": {"channel": channels, "lat": LAT, "lon": LON}}

    def _arco(self, channel):
        return self.arco[:, ARCO_CHANNELS.index(channel)]

    def test_everything_was_read(self):
        self.assertTrue(np.all(self.ncar_valid == 1))
        self.assertTrue(np.all(self.arco_valid == 1))
        self.assertFalse(np.any(np.isnan(self.ncar)))
        self.assertFalse(np.any(np.isnan(self.arco)))

    def test_channels_match_arco_better_than_wrong_counterparts(self):
        for cidx, channel in enumerate(CHANNELS):
            with self.subTest(channel=channel):
                ncar = self.ncar[:, cidx]
                arco = self._arco(channel)

                error = _normalized_error(ncar, arco)
                # NCAR at t against ARCO at t + 6h
                wrong = {"6h later": _normalized_error(ncar[:-1], arco[1:])}
                if channel in NEIGHBOURS:
                    wrong[NEIGHBOURS[channel]] = _normalized_error(ncar, self._arco(NEIGHBOURS[channel]))

                print(f"{channel}: error {error:.2e}, " + ", ".join(f"{k} {v:.2e}" for k, v in wrong.items()))

                self.assertLess(error, MAX_ERROR)
                for desc, value in wrong.items():
                    self.assertGreater(value, MIN_MARGIN * error, f"{channel} is not distinguishable from {desc}")


if __name__ == "__main__":
    unittest.main()
