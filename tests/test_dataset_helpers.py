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

"""The helpers of the dataset scripts, and the per-file kernels of the statistics scripts, against values computed directly."""

import os
import sys
import tempfile
import unittest

import h5py as h5
import numpy as np
import torch
from parameterized import parameterized

from makani.datasets.helpers import file_sample_ranges, get_wind_channels, mask_data, welford_combine
from makani.utils.grids import GridQuadrature, make_grid

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .testutils import init_dataset_metadata


def _write_fields(path, data):
    with h5.File(path, "w") as f:
        f["fields"] = data


class TestFileSampleRanges(unittest.TestCase):
    """The samples of a rank, as a range within each file."""

    FILES = ["a.h5", "b.h5", "c.h5"]
    COUNTS = [4, 3, 5]

    @parameterized.expand(
        [
            ("all", 0, 12, {"a.h5": slice(0, 4), "b.h5": slice(0, 3), "c.h5": slice(0, 5)}),
            ("within_one_file", 1, 3, {"a.h5": slice(1, 3)}),
            ("across_files", 2, 9, {"a.h5": slice(2, 4), "b.h5": slice(0, 3), "c.h5": slice(0, 2)}),
            ("from_a_file_boundary", 4, 7, {"b.h5": slice(0, 3)}),
            ("empty", 5, 5, {}),
            ("past_the_end", 12, 12, {}),
        ]
    )
    def test_ranges(self, _, start, end, expected):
        self.assertEqual(file_sample_ranges(self.FILES, self.COUNTS, start, end), expected)

    @parameterized.expand([(1,), (2,), (3,), (5,), (13,)])
    def test_the_ranks_read_every_sample_once(self, num_ranks):
        # the sharding of the statistics scripts, for several files and for one combined file
        for files, counts in [(self.FILES, self.COUNTS), (["combined.h5v"], [sum(self.COUNTS)])]:
            total = sum(counts)
            chunk = (total + num_ranks - 1) // num_ranks
            seen = []
            for rank in range(num_ranks):
                start = chunk * rank
                end = min(start + chunk, total)
                offsets = dict(zip(files, np.cumsum([0] + counts[:-1])))
                for filename, slc in file_sample_ranges(files, counts, start, end).items():
                    seen.extend(range(offsets[filename] + slc.start, offsets[filename] + slc.stop))
            with self.subTest(files=files):
                self.assertEqual(seen, list(range(total)))


class TestMaskData(unittest.TestCase):
    def test_data_without_nans_is_all_valid(self):
        data = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        masked, valid = mask_data(data)
        self.assertTrue(torch.equal(masked, data))
        self.assertTrue(torch.equal(valid, torch.ones_like(data)))

    def test_nans_are_zeroed_and_masked(self):
        data = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        nans = torch.rand(data.shape) < 0.3
        data[nans] = torch.nan
        masked, valid = mask_data(data)
        self.assertFalse(torch.isnan(masked).any())
        self.assertTrue(torch.equal(valid, (~nans).to(torch.float64)))
        self.assertTrue(torch.equal(masked[~nans], data[~nans]))
        self.assertTrue(torch.all(masked[nans] == 0.0))


def _channel_stats(x):
    """Statistics of samples ``x`` (B, C, H, W), with per-channel counts, in the layout of compute_stats."""
    num_channels = x.shape[1]
    counts = torch.full((num_channels,), float(x.shape[0] * x.shape[2] * x.shape[3]), dtype=torch.float64)
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    m2 = ((x - mean) ** 2).sum(dim=(0, 2, 3), keepdim=True)
    return {
        "meanvar": {"type": "meanvar", "counts": counts, "values": torch.stack([mean, m2])},
        "mins": {"type": "min", "counts": counts.clone(), "values": x.amin(dim=(0, 2, 3), keepdim=True)},
        "maxs": {"type": "max", "counts": counts.clone(), "values": x.amax(dim=(0, 2, 3), keepdim=True)},
        "time_means": {
            "type": "mean",
            "counts": torch.full((num_channels,), float(x.shape[0]), dtype=torch.float64),
            "values": x.mean(dim=0, keepdim=True),
        },
    }


def _empty_channel_stats(shape):
    """The zero-count statistics compute_stats starts from."""
    _, num_channels, height, width = shape

    def zeros():
        return torch.zeros((num_channels,), dtype=torch.float64)

    return {
        "meanvar": {
            "type": "meanvar",
            "counts": zeros(),
            "values": torch.zeros((2, 1, num_channels, 1, 1), dtype=torch.float64),
        },
        "mins": {
            "type": "min",
            "counts": zeros(),
            "values": torch.full((1, num_channels, 1, 1), torch.inf, dtype=torch.float64),
        },
        "maxs": {
            "type": "max",
            "counts": zeros(),
            "values": torch.full((1, num_channels, 1, 1), -torch.inf, dtype=torch.float64),
        },
        "time_means": {
            "type": "mean",
            "counts": zeros(),
            "values": torch.zeros((1, num_channels, height, width), dtype=torch.float64),
        },
    }


class TestWelfordCombine(unittest.TestCase):
    """Combining the statistics of parts of the data gives the statistics of all of it."""

    @parameterized.expand([([10],), ([3, 7],), ([1, 1, 8],), ([4, 4, 2],)])
    def test_per_channel_counts(self, sizes):
        x = (
            torch.randn(sum(sizes), 3, 4, 6, dtype=torch.float64) * torch.tensor([1.0, 10.0, 0.1]).reshape(1, 3, 1, 1)
            + 5.0
        )
        stats = _empty_channel_stats(x.shape)
        for part in torch.split(x, sizes):
            stats = welford_combine(stats, _channel_stats(part))

        expected = _channel_stats(x)
        for name, entry in expected.items():
            with self.subTest(stat=name):
                torch.testing.assert_close(stats[name]["values"], entry["values"], rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(stats[name]["counts"], entry["counts"], rtol=0.0, atol=0.0)

    def test_scalar_counts(self):
        # the layout of compute_spectra: one count, the number of samples, for all channels and degrees
        x = torch.randn(9, 2, 5, dtype=torch.float64)

        def stats_of(part):
            mean = part.mean(0)
            return {
                "meanvar": {
                    "type": "meanvar",
                    "counts": torch.as_tensor(float(part.shape[0]), dtype=torch.float64),
                    "values": torch.stack([mean, ((part - mean) ** 2).sum(0)]),
                }
            }

        stats = {
            "meanvar": {
                "type": "meanvar",
                "counts": torch.as_tensor(0.0, dtype=torch.float64),
                "values": torch.zeros(2, 2, 5, dtype=torch.float64),
            }
        }
        for part in torch.split(x, [2, 4, 3]):
            stats = welford_combine(stats, stats_of(part))

        expected = stats_of(x)["meanvar"]
        torch.testing.assert_close(stats["meanvar"]["values"], expected["values"], rtol=1e-12, atol=1e-12)
        self.assertEqual(float(stats["meanvar"]["counts"]), 9.0)


class TestGetWindChannels(unittest.TestCase):
    def test_pairs_are_matched_by_level(self):
        names = ["t2m", "v500", "u10m", "u500", "v10m"]
        (uchannels, vchannels), (unames, vnames) = get_wind_channels(names)
        self.assertEqual(unames, ["u10m", "u500"])
        self.assertEqual(vnames, ["v10m", "v500"])
        self.assertEqual(uchannels, [2, 3])
        self.assertEqual(vchannels, [4, 1])

    @parameterized.expand([(["u10m", "v10m", "u500"],), (["u10m", "v100m"],)])
    def test_unpaired_wind_channels_are_refused(self, names):
        with self.assertRaises(ValueError):
            get_wind_channels(names)

    def test_histogram_script_pairs_and_skips_unpaired(self):
        from makani.datasets.compute_histograms import get_wind_channels as histogram_wind_channels

        uchannels, vchannels = histogram_wind_channels(["v10m", "u10m", "u500", "t2m"])
        self.assertEqual((uchannels, vchannels), ([1], [0]))


class TestHistograms(unittest.TestCase):
    """compute_histograms: the per-file extrema and histograms, and the whole script on one rank."""

    NUM_SAMPLES = (5, 4)
    SHAPE = (3, 6, 12)

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.input_dir = os.path.join(cls.tmpdir.name, "data")
        os.makedirs(cls.input_dir)
        generator = np.random.default_rng(1234)
        cls.data = []
        for i, num_samples in enumerate(cls.NUM_SAMPLES):
            data = generator.normal(size=(num_samples,) + cls.SHAPE) * np.array([1.0, 3.0, 0.5]).reshape(1, 3, 1, 1)
            _write_fields(os.path.join(cls.input_dir, f"{i}.h5"), data)
            cls.data.append(data)
        cls.metadata_file = init_dataset_metadata(
            os.path.join(cls.tmpdir.name, "metadata"), channel_names=["u10m", "v10m", "t2m"], grid_type="equiangular"
        )
        cls.weights = (
            GridQuadrature(make_grid(cls.SHAPE[1:], "equiangular"), crop_shape=None, crop_offset=(0, 0), normalize=True)
            .quad_weight.cpu()
            .numpy()
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    @parameterized.expand([(1,), (2,), (8,)])
    def test_file_extrema(self, batch_size):
        from makani.datasets.compute_histograms import get_file_stats

        data = self.data[0][1:4]
        bias = np.array([1.0, -2.0, 0.5]).reshape(1, 3, 1, 1)
        norm = np.array([2.0, 4.0, 0.5]).reshape(1, 3, 1, 1)
        count, mins, maxs = get_file_stats(
            os.path.join(self.input_dir, "0.h5"), slice(1, 4), bias=bias, norm=norm, batch_size=batch_size
        )

        expected = (data - bias) / norm
        self.assertEqual(count, 3 * self.SHAPE[1] * self.SHAPE[2])
        np.testing.assert_allclose(mins, expected.min(axis=(0, 2, 3)))
        np.testing.assert_allclose(maxs, expected.max(axis=(0, 2, 3)))

    @parameterized.expand([(1,), (2,), (8,)])
    def test_file_histograms(self, batch_size):
        # with several batches, the weights of each batch have to match the batch
        from makani.datasets.compute_histograms import get_file_histograms

        data = self.data[0]
        mins = data.min(axis=(0, 2, 3))
        maxs = data.max(axis=(0, 2, 3))
        histograms = get_file_histograms(
            os.path.join(self.input_dir, "0.h5"),
            slice(0, len(data)),
            mins,
            maxs,
            7,
            self.weights,
            batch_size=batch_size,
        )

        weights = np.tile(self.weights, (len(data), 1, 1, 1))
        for c, (counts, edges) in enumerate(histograms):
            expected_counts, expected_edges = np.histogram(
                data[:, c : c + 1], bins=7, range=(mins[c], maxs[c]), weights=weights
            )
            with self.subTest(channel=c):
                np.testing.assert_allclose(counts, expected_counts, rtol=1e-12)
                np.testing.assert_allclose(edges, expected_edges)
                # the weights of each sample sum to one
                self.assertAlmostEqual(counts.sum(), len(data))

    @parameterized.expand([(5, 2), (0, 3)])
    def test_script_on_one_rank(self, nbins, batch_size):
        from makani.datasets.compute_histograms import compute_histograms

        with tempfile.TemporaryDirectory() as output_dir:
            compute_histograms(self.input_dir, output_dir, None, self.metadata_file, nbins=nbins, batch_size=batch_size)
            with h5.File(os.path.join(output_dir, "histograms.h5"), "r") as f:
                counts = f["data"][...]
                edges = f["edges"][...]

        data = np.concatenate(self.data)
        if nbins <= 0:
            # the square root of the number of points
            nbins = int(np.sqrt(data.shape[0] * data.shape[2] * data.shape[3]))
        weights = np.tile(self.weights, (len(data), 1, 1, 1))
        self.assertEqual(counts.shape, (self.SHAPE[0], nbins))
        for c in range(self.SHAPE[0]):
            expected_counts, expected_edges = np.histogram(
                data[:, c : c + 1], bins=nbins, range=(data[:, c].min(), data[:, c].max()), weights=weights
            )
            with self.subTest(channel=c):
                np.testing.assert_allclose(counts[c], expected_counts, rtol=1e-12)
                np.testing.assert_allclose(edges[c], expected_edges)


class TestPowerSpectra(unittest.TestCase):
    """compute_spectra: the power spectrum of a field, and its statistics over the samples of a file."""

    NLAT, NLON = 8, 16

    def setUp(self):
        from torch_harmonics import InverseRealSHT, RealSHT

        grid = make_grid((self.NLAT, self.NLON), "legendre-gauss")
        self.sht = RealSHT(grid).to(torch.float64)
        self.isht = InverseRealSHT(grid).to(torch.float64)

    @parameterized.expand([(2, 0), (3, 2), (5, 5)])
    def test_power_of_a_spherical_harmonic(self, l, m):
        from makani.datasets.compute_spectra import compute_powerspectrum

        # the coefficients of order 0 of a real field are real
        value = 1.5 + 0.5j if m > 0 else 1.5
        coeffs = torch.zeros(1, 1, self.sht.lmax, self.sht.mmax, dtype=torch.complex128)
        coeffs[..., l, m] = value
        field = self.isht(coeffs)

        power = compute_powerspectrum(field, self.sht)
        expected = torch.zeros(1, 1, self.sht.lmax, dtype=torch.float64)
        # a real field with order m != 0 also holds the conjugate coefficient of order -m
        expected[..., l] = abs(value) ** 2 * (2.0 if m > 0 else 1.0)
        torch.testing.assert_close(power, expected, rtol=1e-10, atol=1e-10)

    @parameterized.expand([(None,), (1,), (3,)])
    def test_file_statistics(self, batch_size):
        from makani.datasets.compute_spectra import get_file_power_spectra

        generator = torch.Generator().manual_seed(42)
        data = self.isht(torch.randn(7, 2, self.sht.lmax, self.sht.mmax, dtype=torch.complex128, generator=generator))
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "data.h5")
            _write_fields(filename, data.numpy())
            stats = get_file_power_spectra(filename, slice(1, 6), self.sht, batch_size=batch_size)

        power = torch.sum(
            torch.abs(self.sht(data[1:6])) ** 2 * torch.tensor([1.0] + [2.0] * (self.sht.mmax - 1)), dim=-1
        )
        mean = power.mean(0)
        torch.testing.assert_close(stats["global_meanvar"]["values"][0], mean, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(
            stats["global_meanvar"]["values"][1], ((power - mean) ** 2).sum(0), rtol=1e-10, atol=1e-10
        )
        self.assertEqual(float(stats["global_meanvar"]["counts"]), 5.0)

    def test_script_uses_the_grid_of_the_dataset(self):
        # on a Gaussian grid, the spectra need the transform of that grid
        import subprocess

        generator = torch.Generator().manual_seed(7)
        data = [
            self.isht(torch.randn(n, 2, self.sht.lmax, self.sht.mmax, dtype=torch.complex128, generator=generator))
            for n in (3, 4)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "data")
            os.makedirs(input_dir)
            for i, d in enumerate(data):
                _write_fields(os.path.join(input_dir, f"{i}.h5"), d.numpy())
            metadata_file = init_dataset_metadata(
                os.path.join(tmpdir, "metadata"), channel_names=["a", "b"], grid_type="legendre-gauss"
            )
            output_dir = os.path.join(tmpdir, "out")
            os.makedirs(output_dir)

            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MASTER_PORT="29537")
            args = [
                "--input_path",
                input_dir,
                "--metadata_file",
                metadata_file,
                "--output_path",
                output_dir,
                "--batch_size",
                "2",
            ]
            result = subprocess.run(
                [sys.executable, "-m", "makani.datasets.compute_spectra"] + args,
                capture_output=True,
                text=True,
                env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            means = np.load(os.path.join(output_dir, "power_spectra_means.npy"))
            stds = np.load(os.path.join(output_dir, "power_spectra_stds.npy"))

        all_data = torch.cat(data)
        power = torch.sum(
            torch.abs(self.sht(all_data)) ** 2 * torch.tensor([1.0] + [2.0] * (self.sht.mmax - 1)), dim=-1
        )
        np.testing.assert_allclose(means, power.mean(0).numpy(), rtol=1e-5)
        np.testing.assert_allclose(stds, power.std(0, correction=0).numpy(), rtol=1e-5)


class TestHydrostaticResidual(unittest.TestCase):
    """compute_hydrostatic_balance_climatology: the residual of the hydrostatic balance and its statistics."""

    LEVELS = 4
    SHAPE = (6, 12)

    def _fields(self, num_samples, moist, residual):
        """Geopotential built level by level from temperature (and humidity), off balance by ``residual``."""
        generator = torch.Generator().manual_seed(3)
        height, width = self.SHAPE
        T = 250.0 + 10.0 * torch.rand(num_samples, self.LEVELS, height, width, dtype=torch.float64, generator=generator)
        q = 0.01 * torch.rand(num_samples, self.LEVELS, height, width, dtype=torch.float64, generator=generator)
        Tv = T * (1.0 + self.q_prefact * q) if moist else T
        Z = [1000.0 * torch.rand(num_samples, height, width, dtype=torch.float64, generator=generator)]
        for i in range(1, self.LEVELS):
            Z.append(Z[-1] + self.coeffs[i - 1] * (Tv[:, i] + Tv[:, i - 1]) + residual[:, i - 1])
        # channels: z of all levels, then t, then q
        return torch.cat([torch.stack(Z, dim=1), T, q], dim=1)

    def setUp(self):
        self.coeffs = torch.tensor([20.0, 35.0, 50.0], dtype=torch.float64)
        self.q_prefact = 0.608
        self.z_idx = list(range(self.LEVELS))
        self.t_idx = list(range(self.LEVELS, 2 * self.LEVELS))
        self.q_idx = list(range(2 * self.LEVELS, 3 * self.LEVELS))

    @parameterized.expand([(True,), (False,)])
    def test_residual(self, moist):
        from makani.datasets.compute_hydrostatic_balance_climatology import compute_residual

        residual = torch.randn(3, self.LEVELS - 1, *self.SHAPE, dtype=torch.float64)
        data = self._fields(3, moist, residual)
        q_idx = self.q_idx if moist else None
        result = compute_residual(data, self.z_idx, self.t_idx, q_idx, self.coeffs, self.q_prefact)
        torch.testing.assert_close(result, residual, rtol=1e-9, atol=1e-9)

    @parameterized.expand([(2,), (16,)])
    def test_file_statistics(self, batch_size):
        from makani.datasets.compute_hydrostatic_balance_climatology import get_file_stats

        num_samples = 5
        generator = torch.Generator().manual_seed(11)
        residual = torch.randn(
            num_samples, self.LEVELS - 1, *self.SHAPE, dtype=torch.float64, generator=generator
        ) + torch.tensor([1.0, -2.0, 3.0]).reshape(1, -1, 1, 1)
        data = self._fields(num_samples, True, residual)
        quadrature = GridQuadrature(
            make_grid(self.SHAPE, "equiangular"), crop_shape=None, crop_offset=(0, 0), normalize=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "data.h5")
            _write_fields(filename, data.numpy())
            stats = get_file_stats(
                filename,
                slice(0, None),
                self.z_idx,
                self.t_idx,
                self.q_idx,
                self.coeffs,
                self.q_prefact,
                quadrature,
                batch_size=batch_size,
            )

        weights = quadrature.quad_weight.to(torch.float64)
        area = weights.sum()
        mean = (residual * weights).sum(dim=(0, 2, 3)) / (num_samples * area)
        m2 = (((residual - mean.reshape(1, -1, 1, 1)) ** 2) * weights).sum(dim=(0, 2, 3))

        torch.testing.assert_close(stats["time_means"]["values"], residual.mean(0, keepdim=True), rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(stats["global_meanvar"]["values"][0].reshape(-1), mean, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(stats["global_meanvar"]["values"][1].reshape(-1), m2, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(
            stats["global_meanvar"]["counts"],
            torch.full((self.LEVELS - 1,), float(num_samples * area), dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
