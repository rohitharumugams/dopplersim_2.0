"""Broadband spectrum shapes and tyre emission (vendored subset for forward path)."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.signal import butter, sosfiltfilt

_SOS: dict[tuple, np.ndarray] = {}
_TIRE_SOURCE_PEAK = 1.0
_TIRE_RADIUS_M = 0.33


def _bandpass(sr: int, lo: float, hi: float) -> np.ndarray:
    key = (sr, lo, hi)
    if key not in _SOS:
        nyq = sr / 2.0
        _SOS[key] = butter(3, [lo / nyq, min(hi, nyq - 50) / nyq], btype="band", output="sos")
    return _SOS[key]


def _colored_noise(n: int, seed: int, beta: float = 0.35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n).astype(np.float64)
    spec = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if n > 1 else 1.0
    spec /= np.power(freqs, beta * 0.5)
    return np.fft.irfft(spec, n=n).astype(np.float64)


def _hz_rfftfreq(n: int, sr: int) -> np.ndarray:
    return np.fft.rfftfreq(n, d=1.0 / float(sr))


def _tire_spectrum_shape(spec: np.ndarray, freqs_hz: np.ndarray, sr: int) -> np.ndarray:
    del sr
    f = np.maximum(freqs_hz, 1.0)
    mid = np.exp(-0.5 * ((np.log10(f / 650.0)) / 0.35) ** 2)
    mid[f < 120.0] *= f[f < 120.0] / 120.0
    mid[f > 1200.0] *= np.exp(-(f[f > 1200.0] - 1200.0) / 350.0)
    lf = np.exp(-0.5 * ((np.log10(f / 95.0)) / 0.50) ** 2)
    lf[f < 40.0] *= f[f < 40.0] / 40.0
    peak = mid + 0.90 * lf
    return spec * peak


def _body_spectrum_shape(spec: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    f = np.maximum(freqs_hz, 1.0)
    peak = np.exp(-0.5 * ((np.log10(f / 520.0)) / 0.45) ** 2)
    peak[f < 80.0] *= f[f < 80.0] / 80.0
    peak[f > 1800.0] *= np.exp(-(f[f > 1800.0] - 1800.0) / 450.0)
    return spec * peak


def tire_emission_buffer(
    sr: int,
    n_samples: int,
    speed_mps: float,
    emitter_id: str,
    *,
    lo_hz: float = 40.0,
    hi_hz: float = 2200.0,
) -> np.ndarray:
    seed = int(hashlib.md5(f"tire|{emitter_id}|{sr}".encode()).hexdigest()[:8], 16)
    n = max(n_samples, 4096)
    noise = _colored_noise(n, seed=seed, beta=0.48)
    spec = np.fft.rfft(noise)
    spec = _tire_spectrum_shape(spec, _hz_rfftfreq(n, sr), sr)
    out = np.fft.irfft(spec, n=n).astype(np.float64)
    out = sosfiltfilt(_bandpass(sr, lo_hz, hi_hz), out)

    v = max(1.0, float(speed_mps))
    f_rot = v / (2.0 * np.pi * _TIRE_RADIUS_M)
    t = np.arange(n, dtype=np.float64) / float(sr)
    tread = 1.0 + 0.22 * np.sin(2.0 * np.pi * f_rot * t + seed % 7)
    tread *= 1.0 + 0.08 * np.sin(2.0 * np.pi * 2.0 * f_rot * t + 0.4)
    out = out * tread

    v_ref = 27.78
    level = float((v / v_ref) ** 1.6 * _TIRE_SOURCE_PEAK)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-8:
        out = out / peak * level
    return out[:n_samples].astype(np.float32)
