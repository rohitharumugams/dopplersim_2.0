"""Independent broadband body-segment sources (incoherent panels / tire patches)."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _colored_noise(n: int, seed: int, beta: float = 0.45) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n // 2 + 1).astype(np.complex128)
    freqs = np.arange(len(w), dtype=np.float64)
    freqs[0] = 1.0
    w *= 1.0 / (freqs ** (0.5 * beta))
    return np.fft.irfft(w, n=n).astype(np.float64)


def recording_spectrum_shape(
    passby: np.ndarray,
    sr: int,
    n_fft: int,
) -> np.ndarray:
    """Normalised magnitude spectrum from VS13 approach timbre (shape target for body)."""
    from doppler_4dot0.audio.source import approach_timbre_excerpt

    x = np.asarray(approach_timbre_excerpt(passby, sr), dtype=np.float64)
    if len(x) < 256:
        x = np.pad(x, (0, max(0, 256 - len(x))))
    spec = np.abs(np.fft.rfft(x, n=n_fft))
    spec[0] = 0.0
    peak = float(np.max(spec))
    if peak < 1e-12:
        return np.ones(len(spec), dtype=np.float64)
    return (spec / peak).astype(np.float64)


def segment_band_broadband_buffer(
    segment_id: str,
    band_id: str,
    sr: int,
    n_samples: int,
    *,
    speed_mps: float,
    spectrum_shape: np.ndarray,
    lo_hz: float,
    hi_hz: float,
    level: float = 1.0,
) -> np.ndarray:
    """Independent band-limited patch noise (deterministic seed per segment+band)."""
    seed = int(
        hashlib.md5(f'vs13body|{segment_id}|{band_id}|{sr}'.encode()).hexdigest()[:8],
        16,
    )
    return segment_broadband_buffer(
        f'{segment_id}|{band_id}',
        sr,
        n_samples,
        speed_mps=speed_mps,
        spectrum_shape=spectrum_shape,
        level=level,
        lo_hz=lo_hz,
        hi_hz=hi_hz,
        _noise_seed=seed,
    )


def segment_broadband_buffer(
    segment_id: str,
    sr: int,
    n_samples: int,
    *,
    speed_mps: float,
    spectrum_shape: np.ndarray,
    level: float = 1.0,
    lo_hz: float = 40.0,
    hi_hz: float = 3500.0,
    _noise_seed: int | None = None,
) -> np.ndarray:
    """
    Spatially incoherent body patch: independent coloured noise per segment,
    shaped to VS13 recording spectrum (not phase-locked to other segments).
    """
    seed = _noise_seed if _noise_seed is not None else int(
        hashlib.md5(f'vs13body|{segment_id}|{sr}'.encode()).hexdigest()[:8], 16,
    )
    n = max(n_samples, 4096)
    noise = _colored_noise(n, seed=seed, beta=0.44)
    spec = np.fft.rfft(noise)
    n_spec = len(spec)
    shape = np.asarray(spectrum_shape, dtype=np.float64)
    if len(shape) != n_spec:
        xi = np.linspace(0, 1, len(shape))
        xo = np.linspace(0, 1, n_spec)
        shape = np.interp(xo, xi, shape)
    spec *= shape
    out = np.fft.irfft(spec, n=n).astype(np.float64)

    nyq = 0.5 * sr
    lo, hi = max(lo_hz / nyq, 1e-4), min(hi_hz / nyq, 0.999)
    if hi > lo:
        sos = butter(3, [lo, hi], btype='band', output='sos')
        out = sosfiltfilt(sos, out)

    v_ref = 27.78
    v = max(1.0, float(speed_mps))
    gain = float(level * (v / v_ref) ** 1.25)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-8:
        out = out / peak * gain
    return out[:n_samples].astype(np.float32)
