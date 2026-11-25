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

import unittest
from parameterized import parameterized

import torch
import torch.nn.functional as F

from makani.utils.grids import make_grid
from makani.models.common import RealFFT1, InverseRealFFT1, RealFFT2, InverseRealFFT2, RealFFT3, InverseRealFFT3
from makani.models.common.fft import rfft, irfft

_devices = [torch.device("cpu")]
if torch.cuda.is_available():
    _devices.append(torch.device("cuda"))

_NORMS = [None, "backward", "ortho", "forward"]


def _saved_bytes(fn, *args):
    """Bytes autograd keeps for the backward of fn(*args)."""
    sizes = []

    def pack(t):
        sizes.append(t.numel() * t.element_size())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(*args)
    return sum(sizes)


def _torch_irfft(x, n, dim=-1, norm=None):
    """The inverse as torch-harmonics computes it: zero the imaginary parts of DC and Nyquist, then torch.fft.irfft."""
    m = x.shape[dim]
    scale = torch.ones(m, dtype=x.real.dtype, device=x.device)
    scale[0] = 0.0
    if n % 2 == 0 and n // 2 < m:
        scale[n // 2] = 0.0
    shape = [1] * x.dim()
    shape[dim] = m
    return torch.fft.irfft(torch.complex(x.real, x.imag * scale.reshape(shape)), n=n, dim=dim, norm=norm)


class TestRealFFT(unittest.TestCase):
    """rfft and irfft against torch.fft, and what their backward keeps."""

    @parameterized.expand(
        [(n, nmodes, dim, norm) for n in (10, 11) for nmodes in (None, 4) for dim in (-1, 1) for norm in _NORMS]
    )
    def test_rfft_matches_torch(self, n, nmodes, dim, norm):
        for device in _devices:
            with self.subTest(device=device):
                x = torch.randn(3, n, 5, n, dtype=torch.float64, device=device, requires_grad=True)
                y = rfft(x, nmodes=nmodes, dim=dim, norm=norm)
                ref = torch.fft.rfft(x, dim=dim, norm=norm)
                if nmodes is not None:
                    ref = ref.narrow(dim, 0, nmodes)
                torch.testing.assert_close(y, ref)
                self.assertTrue(y.is_contiguous() or nmodes is None)

                g = torch.randn_like(y)
                (gx,) = torch.autograd.grad(y, x, g)
                (gref,) = torch.autograd.grad(ref, x, g)
                torch.testing.assert_close(gx, gref)
                if dim == -1:
                    # torch returns the real part of a complex tensor here, a stride-2 view
                    self.assertTrue(gx.is_contiguous())

    @parameterized.expand([(n, nmodes, norm) for n in (6, 7) for nmodes in (None, 2) for norm in _NORMS])
    def test_rfft_gradcheck(self, n, nmodes, norm):
        x = torch.randn(2, 3, n, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda t: rfft(t, nmodes=nmodes, norm=norm), (x,)))
        self.assertTrue(torch.autograd.gradgradcheck(lambda t: rfft(t, nmodes=nmodes, norm=norm), (x,)))

    @parameterized.expand([(8, 12), (8, 5)])
    def test_rfft_resizes_to_n(self, length, n):
        x = torch.randn(2, length, dtype=torch.float64, requires_grad=True)
        y = rfft(x, n=n)
        ref = torch.fft.rfft(x, n=n)
        torch.testing.assert_close(y, ref)
        g = torch.randn_like(y)
        torch.testing.assert_close(torch.autograd.grad(y, x, g)[0], torch.autograd.grad(ref, x, g)[0])

    def test_rfft_keeps_nothing_of_the_input(self):
        x = torch.randn(2, 16, 32, 64, requires_grad=True)
        self.assertGreaterEqual(_saved_bytes(lambda t: torch.fft.rfft(t), x), x.numel() * x.element_size())
        self.assertEqual(_saved_bytes(lambda t: rfft(t), x), 0)
        # the truncation is inside the function, so its backward does not zero-fill the full spectrum either
        self.assertEqual(_saved_bytes(lambda t: rfft(t, nmodes=10), x), 0)

    def test_rfft_rejects_too_many_modes(self):
        with self.assertRaises(ValueError):
            rfft(torch.randn(4, 10), nmodes=7)

    @parameterized.expand([(n, m, norm) for n, m in ((10, 6), (10, 4), (11, 6), (11, 3), (8, 0)) for norm in _NORMS])
    def test_irfft_matches_torch_harmonics(self, n, m, norm):
        for device in _devices:
            with self.subTest(device=device):
                # imaginary parts on DC and Nyquist, which the inverse has to ignore
                x = torch.randn(3, 4, max(m, 1), dtype=torch.complex128, device=device)[..., :m].requires_grad_(True)
                y = irfft(x, n=n, norm=norm)
                ref = _torch_irfft(F.pad(x, (0, 1)) if m == 0 else x, n=n, norm=norm)
                torch.testing.assert_close(y, ref)
                if m > 0:
                    g = torch.randn_like(y)
                    torch.testing.assert_close(torch.autograd.grad(y, x, g)[0], torch.autograd.grad(ref, x, g)[0])

    def test_roundtrip(self):
        x = torch.randn(2, 3, 16, dtype=torch.float64)
        for norm in _NORMS:
            torch.testing.assert_close(irfft(rfft(x, norm=norm), n=16, norm=norm), x)

    def test_torch_func(self):
        x = torch.randn(3, 4, 10, dtype=torch.float64)
        s = torch.randn(3, 4, 6, dtype=torch.complex128)

        def loss(t):
            return torch.view_as_real(rfft(t, nmodes=4, dim=-1, norm="ortho")).square().sum()

        xg = x.clone().requires_grad_(True)
        (gref,) = torch.autograd.grad(loss(xg), xg)
        torch.testing.assert_close(torch.func.grad(loss)(x), gref)
        # vmap over the leading dimension, the transform along the last dimension of each sample
        torch.testing.assert_close(torch.func.vmap(torch.func.grad(loss))(x), gref)
        torch.testing.assert_close(torch.func.vmap(lambda t: rfft(t, dim=1))(x), torch.fft.rfft(x, dim=2))

        # along the first dimension of each sample, not its last: a dimension saved for backward in the
        # batched rank, or not normalized, would pick the wrong axis or length (7 and 10 differ)
        x = torch.randn(3, 7, 10, dtype=torch.float64)
        for dim in (0, -2):

            def loss_first(t):
                return torch.view_as_real(rfft(t, nmodes=3, dim=dim, norm="ortho")).square().sum()

            ref = torch.fft.rfft(x, dim=1, norm="ortho")[:, :3]
            torch.testing.assert_close(torch.func.vmap(lambda t: rfft(t, nmodes=3, dim=dim, norm="ortho"))(x), ref)
            xg = x.clone().requires_grad_(True)
            (gref,) = torch.autograd.grad(
                torch.view_as_real(rfft(xg, nmodes=3, dim=1, norm="ortho")).square().sum(), xg
            )
            torch.testing.assert_close(torch.func.vmap(torch.func.grad(loss_first))(x), gref)

        sg = s.clone().requires_grad_(True)
        (sref,) = torch.autograd.grad(irfft(sg, n=10).square().sum(), sg)
        torch.testing.assert_close(torch.func.grad(lambda t: irfft(t, n=10).square().sum())(s), sref)

    def test_conjugated_spectrum(self):
        # a conjugate view, which view_as_real rejects; irfft reads real and imaginary parts instead
        s = torch.randn(3, 4, 6, dtype=torch.complex128, requires_grad=True)
        y = irfft(s.conj(), n=10)
        ref = _torch_irfft(s.conj().resolve_conj(), n=10)
        torch.testing.assert_close(y, ref)
        g = torch.randn_like(y)
        torch.testing.assert_close(torch.autograd.grad(y, s, g)[0], torch.autograd.grad(ref, s, g)[0])

    def test_compiles(self):
        def f(x):
            return irfft(rfft(x, nmodes=9, norm="ortho"), n=x.shape[-1], norm="ortho")

        x = torch.randn(2, 4, 8, 32, requires_grad=True)
        y = torch.compile(f, fullgraph=True)(x)
        (gx,) = torch.autograd.grad(y.square().sum(), x)
        y_ref = f(x)
        (gx_ref,) = torch.autograd.grad(y_ref.square().sum(), x)
        torch.testing.assert_close(y, y_ref)
        torch.testing.assert_close(gx, gx_ref)


class TestRealFFTLayers(unittest.TestCase):
    """The layers against the transforms they computed before: torch.fft.rfft, rfft2 and rfftn with slicing."""

    @parameterized.expand([(16, 30, None), (16, 30, 8), (15, 31, 5)])
    def test_fft1(self, nlat, nlon, mmax):
        grid = make_grid((nlat, nlon), "euclidean")
        fwd, inv = RealFFT1(grid, mmax=mmax), InverseRealFFT1(grid, mmax=mmax)
        x = torch.randn(2, 3, nlat, nlon, dtype=torch.float64, requires_grad=True)
        y = fwd(x)
        ref = torch.fft.rfft(x, n=nlon, dim=-1, norm="ortho")[..., : fwd.mmax]
        torch.testing.assert_close(y, ref)
        g = torch.randn_like(y)
        torch.testing.assert_close(torch.autograd.grad(y, x, g)[0], torch.autograd.grad(ref, x, g)[0])
        torch.testing.assert_close(inv(y), _torch_irfft(y, n=nlon, norm="ortho"))

    @parameterized.expand([(16, 30, None, None), (16, 30, 9, 8), (15, 31, 6, 5), (16, 30, 16, 8)])
    def test_fft2(self, nlat, nlon, lmax, mmax):
        grid = make_grid((nlat, nlon), "euclidean")
        fwd, inv = RealFFT2(grid, lmax=lmax, mmax=mmax), InverseRealFFT2(grid, lmax=lmax, mmax=mmax)
        x = torch.randn(2, 3, nlat, nlon, dtype=torch.float64, requires_grad=True)
        y = fwd(x)
        ref = torch.fft.rfft2(x, s=(nlat, nlon), dim=(-2, -1), norm="ortho")
        if fwd.truncate:
            ref = torch.cat((ref[..., : fwd.lmax_high, : fwd.mmax], ref[..., -fwd.lmax_low :, : fwd.mmax]), dim=-2)
        torch.testing.assert_close(y, ref)
        g = torch.randn_like(y)
        torch.testing.assert_close(torch.autograd.grad(y, x, g)[0], torch.autograd.grad(ref, x, g)[0])
        self.assertEqual(inv(y).shape, x.shape)

    @parameterized.expand([(6, 8, 10, None, None, None), (6, 8, 10, 4, 5, 4), (7, 9, 11, 3, 4, 5)])
    def test_fft3(self, nd, nh, nw, ldmax, lhmax, lwmax):
        fwd = RealFFT3(nd, nh, nw, ldmax, lhmax, lwmax)
        inv = InverseRealFFT3(nd, nh, nw, ldmax, lhmax, lwmax)
        x = torch.randn(2, 3, nd, nh, nw, dtype=torch.float64, requires_grad=True)
        y = fwd(x)
        ref = torch.fft.rfftn(x, s=(nd, nh, nw), dim=(-3, -2, -1), norm="ortho")[..., : fwd.lwmax]
        ref = torch.cat([ref[..., : fwd.lhmax_high, :], ref[..., -fwd.lhmax_low :, :]], dim=-2)
        ref = torch.cat([ref[..., : fwd.ldmax_high, :, :], ref[..., -fwd.ldmax_low :, :, :]], dim=-3)
        torch.testing.assert_close(y, ref)
        g = torch.randn_like(y)
        torch.testing.assert_close(torch.autograd.grad(y, x, g)[0], torch.autograd.grad(ref, x, g)[0])
        self.assertEqual(inv(y).shape, x.shape)


if __name__ == "__main__":
    unittest.main()
