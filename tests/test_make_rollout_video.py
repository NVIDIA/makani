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

import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

_have_video_deps = all(importlib.util.find_spec(m) is not None for m in ("matplotlib", "imageio", "imageio_ffmpeg"))

if _have_video_deps:
    import imageio_ffmpeg
    from PIL import Image

    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
    # scripts is appended to sys.path above
    import make_rollout_video

# solid colors survive the GIF palette exactly and the yuv420p conversion of H.264 closely
_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 255, 255)]
_HEIGHT, _WIDTH, _FPS = 48, 64, 4


def _frames():
    return [np.full((_HEIGHT, _WIDTH, 3), color, dtype=np.uint8) for color in _COLORS]


@unittest.skipUnless(_have_video_deps, "matplotlib, imageio or imageio-ffmpeg is not installed (the vis extra)")
class TestWriteVideo(unittest.TestCase):
    """write_video writes frames that read back in order, at their size and frame rate."""

    def test_gif(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rollout.gif")
            make_rollout_video.write_video(_frames(), path, _FPS)

            with Image.open(path) as gif:
                self.assertEqual(gif.format, "GIF")
                self.assertEqual(gif.n_frames, len(_COLORS))
                self.assertEqual(gif.size, (_WIDTH, _HEIGHT))
                self.assertEqual(gif.info["loop"], 0)
                for idx, color in enumerate(_COLORS):
                    gif.seek(idx)
                    # GIF stores the duration in hundredths of a second
                    self.assertAlmostEqual(gif.info["duration"], 1000 / _FPS, delta=10)
                    self.assertEqual(gif.convert("RGB").getpixel((_WIDTH // 2, _HEIGHT // 2)), color)

    def test_mp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rollout.mp4")
            make_rollout_video.write_video(_frames(), path, _FPS)

            reader = imageio_ffmpeg.read_frames(path)
            meta = next(reader)
            self.assertEqual(tuple(meta["size"]), (_WIDTH, _HEIGHT))
            self.assertEqual(meta["codec"], "h264")
            self.assertAlmostEqual(meta["fps"], _FPS)
            frames = [np.frombuffer(buf, dtype=np.uint8).reshape(_HEIGHT, _WIDTH, 3) for buf in reader]

        self.assertEqual(len(frames), len(_COLORS))
        for frame, color in zip(frames, _COLORS):
            np.testing.assert_allclose(frame.reshape(-1, 3).mean(axis=0), color, atol=8)

    def test_frames_from_figure(self):
        """Figures of odd pixel size are cropped to the even size H.264 needs."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(1.01, 0.75), dpi=100)
        try:
            frame = make_rollout_video._to_frame(fig)
        finally:
            plt.close(fig)
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.shape[0] % 2, 0)
        self.assertEqual(frame.shape[1] % 2, 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "figure.mp4")
            make_rollout_video.write_video([frame, frame], path, _FPS)
            self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
