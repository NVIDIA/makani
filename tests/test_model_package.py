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
import shutil
import tempfile
import unittest
import datetime as dt

import numpy as np
import torch
import torch.nn as nn

from makani.models.model_package import ModelWrapper
from makani.models.stepper import SingleStepWrapper

from .testutils import set_seed, get_default_parameters, compare_tensors, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W


class _LeadingChannelsModel(nn.Module):
    """
    Dummy backbone returning ``scale`` times the leading ``n_out_chans`` channels.

    The wrapper may append unpredicted (zenith) and static features, so we slice
    from the FRONT -- those are the data channels of the oldest history step and
    are unaffected by whatever gets appended behind them.
    """

    def __init__(self, n_out_chans: int, scale: float = 2.0):
        super().__init__()
        self.n_out_chans = n_out_chans
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return self.scale * x[..., : self.n_out_chans, :, :]

    def encode_process(self, x):
        return self.forward(x)


class _ModelPackageTestBase(unittest.TestCase):
    """Builds a ModelWrapper around a dummy backbone without a real package on disk."""

    @classmethod
    def setUpClass(cls):
        # ModelWrapper needs real normalization arrays; "none" makes
        # get_data_normalization return (None, None), which the wrapper cannot index.
        cls.tmpdir = tempfile.mkdtemp()
        means = np.zeros((1, NUM_CHANNELS, 1, 1), dtype=np.float64)
        stds = np.ones((1, NUM_CHANNELS, 1, 1), dtype=np.float64)
        cls.means_path = os.path.join(cls.tmpdir, "global_means.npy")
        cls.stds_path = os.path.join(cls.tmpdir, "global_stds.npy")
        np.save(cls.means_path, means)
        np.save(cls.stds_path, stds)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        set_seed(333)
        self.C = NUM_CHANNELS
        self.H = IMG_SIZE_H
        self.W = IMG_SIZE_W

    def _make_params(self, n_history=0, add_zenith=True, input_noise=None):
        params = get_default_parameters()
        params.n_history = n_history
        params.add_zenith = add_zenith
        params.dhours = 6
        # identity normalization (mean 0, std 1) so forward stays an exact comparison
        params.normalization = "zscore"
        params.global_means_path = self.means_path
        params.global_stds_path = self.stds_path
        if input_noise is not None:
            params.input_noise = input_noise
        return params

    def _make_wrapper(self, n_history=0, add_zenith=True, input_noise=None):
        params = self._make_params(n_history=n_history, add_zenith=add_zenith, input_noise=input_noise)
        stepper = SingleStepWrapper(params, lambda: _LeadingChannelsModel(n_out_chans=NUM_CHANNELS, scale=2.0))
        wrapper = ModelWrapper(stepper, params)
        # the packaged model is always used in eval mode
        wrapper.eval()
        return wrapper

    def _times(self, n):
        base = dt.datetime(2020, 6, 21, 12, 0, 0, tzinfo=dt.timezone.utc)
        return np.array([base + dt.timedelta(hours=6 * i) for i in range(n)])


class TestZenithFeatureShapes(_ModelPackageTestBase):
    """
    The cached zenith tensor must be (B, n_history+1, H, W).

    That is what Preprocessor2D._append_channels consumes: it runs expand_history
    on the cached tensor, reshaping (B, nhist, H, W) -> (B, nhist, 1, H, W) before
    concatenating on the channel axis. Getting either of the two leading axes
    wrong trips the batch check or the history-divisibility check.
    """

    def _check(self, n_history, batch, n_times):
        wrapper = self._make_wrapper(n_history=n_history)
        x = torch.randn(batch, (n_history + 1) * self.C, self.H, self.W)
        z = wrapper._zenith_features(x, self._times(n_times))
        self.assertEqual(tuple(z.shape), (batch, n_history + 1, self.H, self.W))

    def test_single_sample_no_history(self):
        self._check(n_history=0, batch=1, n_times=1)

    def test_batched_one_time_per_member(self):
        # the case main got wrong: dim 0 must be batch, not a prepended singleton
        self._check(n_history=0, batch=4, n_times=4)

    def test_batched_shared_time(self):
        # a perturbed-IC ensemble: every member valid at the same time
        self._check(n_history=0, batch=4, n_times=1)

    def test_single_sample_with_history(self):
        # this worked before the latent-API PR and must keep working: dim 1 is history
        self._check(n_history=2, batch=1, n_times=3)

    def test_batched_with_history_per_member(self):
        # B * nhist times, ordered member-major
        self._check(n_history=2, batch=4, n_times=12)

    def test_batched_with_history_shared_window(self):
        # one history window broadcast across the batch
        self._check(n_history=2, batch=4, n_times=3)

    def test_member_major_ordering(self):
        # with B*nhist times the reshape must group by member, not interleave them:
        # member b's history window is times [b*nhist : (b+1)*nhist]
        n_history, batch = 2, 3
        nhist = n_history + 1
        wrapper = self._make_wrapper(n_history=n_history)
        x = torch.randn(batch, nhist * self.C, self.H, self.W)
        times = self._times(batch * nhist)

        z = wrapper._zenith_features(x, times)

        from makani.third_party.climt.zenith_angle_v2 import cos_zenith_angle

        for b in range(batch):
            expected = cos_zenith_angle(times[b * nhist : (b + 1) * nhist], wrapper.lon_grid, wrapper.lat_grid)
            self.assertTrue(
                compare_tensors(
                    f"member_major_b{b}",
                    z[b],
                    torch.as_tensor(expected.astype(np.float32)),
                    verbose=True,
                )
            )

    def test_shared_window_is_identical_across_members(self):
        wrapper = self._make_wrapper(n_history=1)
        x = torch.randn(5, 2 * self.C, self.H, self.W)
        z = wrapper._zenith_features(x, self._times(2))
        for b in range(1, 5):
            self.assertTrue(torch.equal(z[0], z[b]))

    def test_mismatched_time_count_raises(self):
        # 5 times for batch 4 / nhist 1 matches neither convention
        wrapper = self._make_wrapper(n_history=0)
        x = torch.randn(4, self.C, self.H, self.W)
        with self.assertRaises(ValueError) as cm:
            wrapper._zenith_features(x, self._times(5))
        # the message must name both acceptable counts (4 per-member, 1 shared)
        msg = str(cm.exception)
        self.assertIn("add_zenith", msg)
        self.assertIn("Pass either 4 times", msg)

    def test_non_4d_input_raises(self):
        wrapper = self._make_wrapper(n_history=0)
        x = torch.randn(2, 1, self.C, self.H, self.W)
        with self.assertRaises(ValueError):
            wrapper._prepare_input(x, self._times(2), normalized_data=True)


class TestModelPackageForward(_ModelPackageTestBase):
    """End-to-end: the wrapper must run for any batch size, with and without zenith."""

    def _run(self, wrapper, batch, n_history, n_times, **kwargs):
        x = torch.randn(batch, (n_history + 1) * self.C, self.H, self.W)
        out = wrapper(x, self._times(n_times), **kwargs)
        self.assertEqual(tuple(out.shape), (batch, self.C, self.H, self.W))
        # identity normalization + leading-channel dummy => exactly 2 * leading slice
        self.assertTrue(compare_tensors("model_package_forward", out, 2.0 * x[:, : self.C], verbose=True))

    def test_forward_batch_one(self):
        self._run(self._make_wrapper(), batch=1, n_history=0, n_times=1)

    def test_forward_batched(self):
        # the whole point: B > 1 through the packaged path
        self._run(self._make_wrapper(), batch=4, n_history=0, n_times=4)

    def test_forward_batched_shared_time(self):
        self._run(self._make_wrapper(), batch=4, n_history=0, n_times=1)

    def test_forward_with_history(self):
        self._run(self._make_wrapper(n_history=2), batch=1, n_history=2, n_times=3)

    def test_forward_batched_with_history(self):
        self._run(self._make_wrapper(n_history=2), batch=3, n_history=2, n_times=9)

    def test_forward_without_zenith_ignores_time(self):
        wrapper = self._make_wrapper(add_zenith=False)
        self._run(wrapper, batch=4, n_history=0, n_times=1)

    def test_forward_denormalizes_when_not_normalized(self):
        # with mean 0 / std 1 the round trip is the identity, so the result is unchanged
        wrapper = self._make_wrapper()
        x = torch.randn(2, self.C, self.H, self.W)
        out = wrapper(x, self._times(2), normalized_data=False)
        self.assertTrue(compare_tensors("denorm_roundtrip", out, 2.0 * x, verbose=True))

    def test_encode_process_matches_forward_preprocessing(self):
        # the dummy's encode_process mirrors its forward, so with an identity
        # denormalization the staged path must reproduce forward exactly
        wrapper = self._make_wrapper()
        x = torch.randn(4, self.C, self.H, self.W)
        times = self._times(4)

        features = wrapper.encode_process(x, times)
        expected = wrapper(x, times)

        self.assertTrue(compare_tensors("model_package_encode_process", features, expected, verbose=True))


class TestNoiseBatchGuard(_ModelPackageTestBase):
    """
    Resizing a stateful noise state mid-sequence must fail loudly.

    The state carries an n_history+1 AR history; reallocating zeroes it, so
    continuing the sequence would silently restart the noise from zero instead of
    the intended stationary distribution.
    """

    DIFFUSION = {"type": "diffusion", "mode": "concatenate", "n_channels": 1}
    WHITE = {"type": "white", "mode": "concatenate", "n_channels": 1}

    def test_resize_during_ar_sequence_raises(self):
        wrapper = self._make_wrapper(input_noise=self.DIFFUSION)
        pp = wrapper.model.preprocessor
        self.assertEqual(pp.input_noise.state.shape[0], 1)

        with self.assertRaises(RuntimeError) as cm:
            pp.update_internal_state(replace_state=False, batch_size=4)
        self.assertIn("replace_state=True", str(cm.exception))

    def test_resize_with_replace_state_is_allowed(self):
        wrapper = self._make_wrapper(input_noise=self.DIFFUSION)
        pp = wrapper.model.preprocessor

        pp.update_internal_state(replace_state=True, batch_size=4)

        self.assertEqual(pp.input_noise.state.shape[0], 4)
        # the history axis is config-derived and must be untouched by the resize
        self.assertEqual(pp.input_noise.state.shape[1], pp.n_history + 1)

    def test_ar_step_at_fixed_batch_is_allowed(self):
        # this is the ensemble/inference pattern: prime once, then roll forward
        wrapper = self._make_wrapper(input_noise=self.DIFFUSION)
        pp = wrapper.model.preprocessor

        pp.update_internal_state(replace_state=True, batch_size=4)
        pp.update_internal_state(replace_state=False, batch_size=4)

        self.assertEqual(pp.input_noise.state.shape[0], 4)

    def test_stateless_noise_is_not_guarded(self):
        # white noise redraws every step, so a resize destroys nothing
        wrapper = self._make_wrapper(input_noise=self.WHITE)
        pp = wrapper.model.preprocessor
        self.assertFalse(pp.input_noise.is_stateful())

        pp.update_internal_state(replace_state=False, batch_size=4)

        self.assertEqual(pp.input_noise.state.shape[0], 4)

    def test_batched_forward_after_priming(self):
        # the documented recipe: prime at the target batch, then run batched
        wrapper = self._make_wrapper(input_noise=self.DIFFUSION)
        wrapper.update_state(replace_state=True, batch_size=4)

        x = torch.randn(4, self.C, self.H, self.W)
        out = wrapper(x, self._times(4), replace_state=False)

        self.assertEqual(tuple(out.shape), (4, self.C, self.H, self.W))

    def test_batched_forward_without_priming_raises(self):
        # noise state is still at params.batch_size == 1; forward defaults to
        # replace_state=None (falsy), so this must be refused rather than silently
        # restarting the AR sequence from zero
        wrapper = self._make_wrapper(input_noise=self.DIFFUSION)
        x = torch.randn(4, self.C, self.H, self.W)

        with self.assertRaises(RuntimeError):
            wrapper(x, self._times(4))


if __name__ == "__main__":
    unittest.main()
