"""HiFi-GAN with Multi-Receptive Field (MRF) generator.

This is the missing piece referenced by :mod:`rvc.lib.algo.synthesizers`.
The repo ships the NSF and plain HiFi-GAN generators already; this module
fills in the MRF variant used by the ``MRF-HiFi-GAN`` / ``MRF HiFi-GAN``
vocoder selection in :class:`Synthesizer`.

Structurally it mirrors :class:`HiFiGANNRFGenerator` (NSF source module +
transposed-conv upsample ladder + per-stage parallel resblocks) but allows
the caller to pick a non-zero ``harmonic_num`` so the source signal carries
richer harmonics — which is the main differentiator of the MRF variant.
"""

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import weight_norm

sys.path.append(os.getcwd())

from rvc.lib.algo.commons import init_weights
from rvc.lib.algo.residuals import ResBlock, LRELU_SLOPE
from rvc.lib.algo._weight_norm_compat import remove_weight_norm


class SineGen(nn.Module):
    """Sine generator used by the NSF source module."""

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        return torch.ones_like(f0) * (f0 > self.voiced_threshold)

    def _f02sine(self, f0, upp):
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, dtype=f0.dtype, device=f0.device)
        rad += F.pad(
            (torch.fmod(rad[:, :-1, -1:].float() + 0.5, 1.0) - 0.5).cumsum(dim=1).fmod(1.0).to(f0),
            (0, 0, 1, 0),
            mode="constant",
        )
        rad = rad.reshape(f0.shape[0], -1, 1)
        # Non-in-place multiply: in-place ``rad *= arange`` only works when
        # ``harmonic_num == 0`` (so the broadcast keeps rad's last dim = 1).
        # With ``harmonic_num > 0`` the result shape changes to
        # ``(B, T*upp, dim)`` and the in-place op raises a RuntimeError.
        rad = rad * torch.arange(1, self.dim + 1, dtype=f0.dtype, device=f0.device).reshape(1, 1, -1)
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        return torch.sin(2 * math.pi * rad)

    def forward(self, f0, upp):
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = self._f02sine(f0, upp) * self.sine_amp
            uv = F.interpolate(self._f02uv(f0).transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
            sine_waves = sine_waves * uv + (
                (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves)
            )
        return sine_waves


class SourceModuleHnNSF(nn.Module):
    """NSF source: take f0 -> sine wave -> linear projection -> tanh."""

    def __init__(self, sample_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x, upsample_factor=1):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x, upsample_factor).to(dtype=self.l_linear.weight.dtype)))


class HiFiGANMRFGenerator(nn.Module):
    """HiFi-GAN generator with Multi-Receptive Field (MRF) resblock banks."""

    def __init__(
        self,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
        gin_channels,
        sample_rate,
        harmonic_num=8,
        checkpointing=False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates) if upsample_rates else 1
        self.f0_upsamp = nn.Upsample(scale_factor=self.upp)
        self.m_source = SourceModuleHnNSF(sample_rate=sample_rate, harmonic_num=harmonic_num)

        self.conv_pre = nn.Conv1d(in_channel, upsample_initial_channel, 7, 1, padding=3)
        self.checkpointing = checkpointing

        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        channels = [upsample_initial_channel // (2 ** (i + 1)) for i in range(self.num_upsamples)]
        stride_f0s = [
            math.prod(upsample_rates[i + 1:]) if i + 1 < self.num_upsamples else 1
            for i in range(self.num_upsamples)
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        channels[i],
                        k,
                        u,
                        padding=((k - u) // 2) if u % 2 == 0 else (u // 2 + u % 2),
                        output_padding=u % 2,
                    )
                )
            )
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            self.noise_convs.append(
                nn.Conv1d(
                    1,
                    channels[i],
                    kernel_size=kernel,
                    stride=stride,
                    padding=0 if stride == 1 else (kernel - stride) // 2,
                )
            )

        # MRF: for every upsample stage, place |resblock_kernel_sizes| parallel
        # resblocks (one per (kernel_size, dilation) pair).  Layout in the
        # ModuleList is row-major: index = stage * num_kernels + j.
        self.resblocks = nn.ModuleList(
            ResBlock(channels[i], k, d)
            for i in range(self.num_upsamples)
            for k, d in zip(resblock_kernel_sizes, resblock_dilations)
        )

        self.conv_post = nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g=None):
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = F.leaky_relu(x, LRELU_SLOPE)

            stage_lo = i * self.num_kernels
            stage_hi = stage_lo + self.num_kernels
            stage_resblocks = self.resblocks[stage_lo:stage_hi]

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False) + noise_convs(har_source)
                xs = sum(checkpoint(rb, x, use_reentrant=False) for rb in stage_resblocks)
            else:
                x = ups(x) + noise_convs(har_source)
                xs = sum(rb(x) for rb in stage_resblocks)

            x = xs / self.num_kernels

        return torch.tanh(self.conv_post(F.leaky_relu(x, LRELU_SLOPE)))

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
