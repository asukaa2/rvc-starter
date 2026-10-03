"""Lightweight spectral-gating noise reducer.

The full :pypi:`noisereduce` package is overkill for the post-processing step
done by the RVC inference pipeline — we just need a self-contained spectral
gate that estimates a noise floor from the quietest frames and softly
attenuates it.  This module implements exactly that, using nothing more than
``numpy`` and ``scipy.signal`` so it works on every backend supported by the
project (CPU, CUDA, MPS, OpenCL — the ``device`` argument is accepted for
API compatibility but ignored because spectral gating is so cheap on CPU).
"""

import numpy as np
from scipy.signal import stft, istft


def reduce_noise(
    y,
    sr,
    prop_decrease=1.0,
    device=None,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
    n_noise_frames=10,
):
    """Reduce stationary background noise via spectral gating.

    Parameters
    ----------
    y : np.ndarray
        Mono 1-D (or 2-D ``(channels, samples)``) input signal.  Output is
        always returned as a 1-D ``np.ndarray`` matching ``y``'s dtype.
    sr : int
        Sample rate of ``y``.
    prop_decrease : float, default ``1.0``
        Proportion (0-1) of the estimated noise to subtract.  Values below 1
        produce a gentler attenuation.
    device : str, optional
        Device hint — ignored, kept for API compatibility with the original
        RVC pipeline that passed ``config.device`` through.
    n_fft, hop_length, win_length : int
        STFT parameters.
    n_noise_frames : int
        Number of the quietest frames used to estimate the noise spectrum.

    Returns
    -------
    np.ndarray
        Denoised 1-D waveform of the same length as ``y`` (rounded to the
        nearest STFT frame).
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y_in = y[None, :]
    elif y.ndim == 2:
        y_in = y
    else:
        raise ValueError(f"reduce_noise expects 1-D or 2-D audio, got {y.ndim}-D")

    # Skip very short signals — there's nothing to gate.
    if y_in.shape[-1] < win_length:
        return y if y.ndim == 1 else y[0]

    nperseg = win_length
    noverlap = win_length - hop_length

    f, t, Zxx = stft(
        y_in,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Per-channel frame energy; the quietest `n_noise_frames` are averaged to
    # produce the noise magnitude spectrum.
    energy_per_frame = mag.mean(axis=1)
    n_noise = min(n_noise_frames, energy_per_frame.shape[-1])
    noise_idx = np.argsort(energy_per_frame, axis=-1)[..., :n_noise]

    noise_mag = np.zeros((mag.shape[0], mag.shape[1], 1), dtype=mag.dtype)
    for c in range(mag.shape[0]):
        # Two-step indexing: numpy would otherwise move the fancy-indexed
        # axis to the front of the result when mixed with a slice, which
        # produces a (n_noise, freq) array instead of (freq, n_noise).
        noise_frames = mag[c][:, noise_idx[c]]          # (freq, n_noise)
        noise_mag[c, :, 0] = noise_frames.mean(axis=-1)  # (freq,)

    prop = float(np.clip(prop_decrease, 0.0, 1.0))
    # Soft spectral gate: y' = y - prop * noise_mag, floored at zero.
    gain = np.maximum(0.0, mag - prop * noise_mag) / np.maximum(mag, 1e-8)
    gain = np.clip(gain, 0.0, 1.0)

    Zxx_clean = gain * mag * np.exp(1j * phase)
    _, y_clean = istft(
        Zxx_clean,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        input_onesided=True,
    )

    # Match the original length so downstream code keeps working.
    target_len = y_in.shape[-1]
    if y_clean.shape[-1] > target_len:
        y_clean = y_clean[..., :target_len]
    elif y_clean.shape[-1] < target_len:
        pad = target_len - y_clean.shape[-1]
        y_clean = np.pad(y_clean, ((0, 0), (0, pad)))

    out = y_clean.squeeze(0)
    # Match the input dtype to avoid surprising upcasts (e.g. float32 → float64).
    out = out.astype(y.dtype, copy=False)
    return out
