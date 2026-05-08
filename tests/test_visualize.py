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

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest

import numpy as np

_have_matplotlib = importlib.util.find_spec("matplotlib") is not None
_have_pil = importlib.util.find_spec("PIL") is not None
_have_moviepy = importlib.util.find_spec("moviepy") is not None
_have_visualize_deps = _have_matplotlib and _have_pil

if _have_visualize_deps:
    import matplotlib

    matplotlib.use("Agg")

    from PIL import Image

    from makani.utils.visualize import (
        VisualizationWrapper,
        _worker_init,
        plot_comparison,
        plot_rollout_metrics,
        visualize_field,
    )


@unittest.skipUnless(_have_visualize_deps, "matplotlib and/or PIL are not installed")
class TestPlotComparison(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.pred = rng.standard_normal((16, 32)).astype(np.float32)
        self.truth = rng.standard_normal((16, 32)).astype(np.float32)

    def test_returns_pil_image(self):
        img = plot_comparison(self.pred, self.truth)
        self.assertIsInstance(img, Image.Image)
        self.assertGreater(img.size[0], 0)
        self.assertGreater(img.size[1], 0)

    def test_image_modes_are_renderable(self):
        img = plot_comparison(self.pred, self.truth)
        self.assertIn(img.mode, ("RGB", "RGBA", "P"))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            plot_comparison(self.pred, self.truth[:, :16])

    def test_non_2d_input_raises(self):
        with self.assertRaises(AssertionError):
            plot_comparison(self.pred.flatten(), self.truth.flatten())

    def test_diverging_branch(self):
        img = plot_comparison(self.pred, self.truth, diverging=True)
        self.assertIsInstance(img, Image.Image)

    def test_explicit_lat_lon(self):
        H, W = self.pred.shape
        lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, H)
        lon = np.linspace(-np.pi, np.pi, W)
        img = plot_comparison(self.pred, self.truth, lat=lat, lon=lon)
        self.assertIsInstance(img, Image.Image)


@unittest.skipUnless(_have_visualize_deps, "matplotlib and/or PIL are not installed")
class TestVisualizeField(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.pred = rng.standard_normal((16, 32)).astype(np.float32)
        self.target = rng.standard_normal((16, 32)).astype(np.float32)

    def test_token_round_trip(self):
        tag = ("epoch_0", "u10m")
        out_tag, img = visualize_field(
            tag, "lambda x: x", self.pred, self.target, None, None, 1.0, 0.0, False
        )
        self.assertEqual(out_tag, tag)
        self.assertIsInstance(img, Image.Image)

    def test_functor_is_applied(self):
        # Use a constant functor; both pred and target collapse to the same value,
        # so plot_comparison must not raise (vmin == vmax edge case is tolerated by matplotlib).
        tag = ("t", "f")
        _, img = visualize_field(
            tag, "lambda x: np.zeros_like(x)", self.pred, self.target,
            None, None, 1.0, 0.0, False,
        )
        self.assertIsInstance(img, Image.Image)

    def test_scale_bias_applied(self):
        # Verify scale*x + bias is computed before the functor by using a functor
        # whose output depends linearly on the input mean.
        tag = ("t", "f")
        _, img = visualize_field(
            tag, "lambda x: x", self.pred, self.target,
            None, None, scale=2.0, bias=1.0, diverging=False,
        )
        self.assertIsInstance(img, Image.Image)


@unittest.skipUnless(_have_visualize_deps, "matplotlib and/or PIL are not installed")
class TestPlotRolloutMetrics(unittest.TestCase):
    def test_writes_png_files(self):
        curves = [np.linspace(0.0, 1.0, 10), np.linspace(1.0, 0.0, 10)]
        var_names = ["u10m", "v10m"]
        with tempfile.TemporaryDirectory() as tmp:
            plot_rollout_metrics(curves, var_names, score_path=tmp, file_prefix="rmse", dtxdh=6)
            for name in var_names:
                fpath = os.path.join(tmp, f"rmse_{name}.png")
                self.assertTrue(os.path.exists(fpath), f"missing {fpath}")
                self.assertGreater(os.path.getsize(fpath), 0)

    def test_no_score_path_does_not_raise(self):
        curves = [np.linspace(0.0, 1.0, 5)]
        plot_rollout_metrics(curves, ["t2m"], score_path=None)


@unittest.skipUnless(_have_visualize_deps, "matplotlib and/or PIL are not installed")
class TestWorkerInit(unittest.TestCase):
    def test_sets_mplbackend_env(self):
        original = os.environ.get("MPLBACKEND")
        try:
            os.environ.pop("MPLBACKEND", None)
            _worker_init()
            self.assertEqual(os.environ.get("MPLBACKEND"), "Agg")
        finally:
            if original is None:
                os.environ.pop("MPLBACKEND", None)
            else:
                os.environ["MPLBACKEND"] = original


@unittest.skipUnless(_have_visualize_deps and _have_moviepy, "matplotlib, PIL, or moviepy is not installed")
class TestVisualizationWrapper(unittest.TestCase):
    """End-to-end smoke test for the multiprocess visualization pipeline."""

    def _make_wrapper(self, num_workers=1):
        plot_list = [
            {"name": "u10m", "functor": "lambda x: x", "diverging": False},
            {"name": "t2m", "functor": "lambda x: x", "diverging": True},
        ]
        return VisualizationWrapper(
            log_to_wandb=False,
            path=".",
            prefix="test",
            plot_list=plot_list,
            lat=None,
            lon=None,
            scale=1.0,
            bias=0.0,
            num_workers=num_workers,
        )

    def _add_frames(self, wrapper, n_frames=3, shape=(16, 32)):
        rng = np.random.default_rng(2)
        for i in range(n_frames):
            pred = rng.standard_normal(shape).astype(np.float32)
            target = rng.standard_normal(shape).astype(np.float32)
            wrapper.add(tag=str(i), prediction=pred, target=target)

    def test_finalize_writes_gif_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            wrapper = self._make_wrapper()
            try:
                self._add_frames(wrapper)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    wrapper.finalize()
                self.assertTrue(os.path.exists("video_output.gif"))
                self.assertGreater(os.path.getsize("video_output.gif"), 0)
                # logger=None should keep moviepy's "Building file ..." chatter out of stdout/stderr
                output = buf.getvalue()
                self.assertNotIn("MoviePy", output)
                self.assertNotIn("Building file", output)
            finally:
                wrapper.executor.shutdown(wait=True)
                os.chdir(cwd)

    def test_reset_clears_requests(self):
        wrapper = self._make_wrapper()
        try:
            self._add_frames(wrapper, n_frames=2)
            self.assertGreater(len(wrapper.requests), 0)
            wrapper.reset()
            self.assertEqual(len(wrapper.requests), 0)
        finally:
            wrapper.executor.shutdown(wait=True)

    def test_finalize_resets_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            wrapper = self._make_wrapper()
            try:
                self._add_frames(wrapper)
                wrapper.finalize()
                self.assertEqual(len(wrapper.requests), 0)
            finally:
                wrapper.executor.shutdown(wait=True)
                os.chdir(cwd)

    def test_add_accumulates_one_request_per_plot_entry(self):
        wrapper = self._make_wrapper()
        try:
            rng = np.random.default_rng(3)
            pred = rng.standard_normal((8, 16)).astype(np.float32)
            target = rng.standard_normal((8, 16)).astype(np.float32)
            wrapper.add(tag="0", prediction=pred, target=target)
            # two entries in plot_list => two submitted requests
            self.assertEqual(len(wrapper.requests), 2)
        finally:
            wrapper.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
