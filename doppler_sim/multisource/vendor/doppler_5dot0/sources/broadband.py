"""Broadband q(t′) shaped from VS13 recording spectrum (tyres, body bed)."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.signal import butter, sosfiltfilt

from doppler_4dot0.audio.body_source import recording_spectrum_shape, segment_broadband_buffer


def _colored_noise(n: int, seed: int, beta: float = 0.38) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n // 2 + 1).astype(np.complex128)
    freqs = np.arange(len(w), dtype=np.float64)
    freqs[0] = 1.0
    w *= 1.0 / (freqs ** (0.5 * beta))
    return np.fft.irfft(w, n=n).astype(np.float64)


def _bandpass_sos(sr: int, lo: float, hi: float) -> np.ndarray:
    nyq = sr / 2.0
    return butter(3, [lo / nyq, min(hi, nyq - 50) / nyq], btype='band', output='sos')


def tire_emission_from_recording(
    passby: np.ndarray,
    sr: int,
    n_samples: int,
    speed_mps: float,
    emitter_id: str,
    *,
    lo_hz: float = 120.0,
    hi_hz: float = 3800.0,
) -> np.ndarray:
    """Tyre patch noise shaped by real recording — extended HF vs synthetic peak."""
    n_fft = max(4096, n_samples)
    shape = recording_spectrum_shape(passby, sr, n_fft)
    seed = int(hashlib.md5(f'tire5|{emitter_id}|{sr}'.encode()).hexdigest()[:8], 16)
    return segment_broadband_buffer(
        emitter_id, sr, n_samples, speed_mps=speed_mps,
        spectrum_shape=shape, level=1.0, lo_hz=lo_hz, hi_hz=hi_hz,
        _noise_seed=seed,
    )


def recording_broadband_bed(
    passby: np.ndarray,
    sr: int,
    n_samples: int,
    speed_mps: float,
    bed_id: str = 'road_bed',
    *,
    lo_hz: float = 80.0,
    hi_hz: float = 4000.0,
    level: float = 0.35,
) -> np.ndarray:
    """Incoherent bed filling mid/HF gaps between tonal radiators."""
    n_fft = max(4096, n_samples)
    shape = recording_spectrum_shape(passby, sr, n_fft)
    seed = int(hashlib.md5(f'bed5|{bed_id}|{sr}'.encode()).hexdigest()[:8], 16)
    n = max(n_samples, 4096)
    noise = _colored_noise(n, seed=seed, beta=0.40)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    sh = np.interp(freqs, np.linspace(0, sr / 2, len(shape)), shape, left=0, right=shape[-1])
    out = np.fft.irfft(spec * sh, n=n).astype(np.float64)
    out = sosfiltfilt(_bandpass_sos(sr, lo_hz, hi_hz), out)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-8:
        out = out / peak * float(level)
    return out[:n_samples].astype(np.float32)
