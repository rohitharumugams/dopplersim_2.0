"""STFT magnitude targets for VS13 inverse fit."""

from __future__ import annotations

import numpy as np
from scipy.signal import stft


def _stft_mag(y: np.ndarray, sr: int, *, nperseg: int = 512, hop: int = 128) -> np.ndarray:
    sig = np.asarray(y, dtype=np.float64).ravel()
    _, _, z = stft(sig, fs=sr, nperseg=nperseg, noverlap=nperseg - hop, boundary=None)
    return np.abs(z).astype(np.float64)


def stft_mag_features(
    y: np.ndarray,
    sr: int,
    *,
    f_max_hz: float = 4000.0,
    df: int = 6,
    dt: int = 3,
    nperseg: int = 512,
    hop: int = 128,
    log1p_scale: float = 12.0,
) -> np.ndarray:
    """Downsampled log-magnitude STFT vector (0 – f_max_hz)."""
    mag = _stft_mag(y, sr, nperseg=nperseg, hop=hop)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / float(sr))[: mag.shape[0]]
    f_hi = int(np.searchsorted(freqs, f_max_hz))
    mag = mag[:f_hi, :]
    mag = mag[::df, ::dt]
    peak = float(np.max(mag))
    if peak < 1e-12:
        return np.zeros(mag.size, dtype=np.float64)
    norm = np.log1p(mag / peak * log1p_scale)
    return norm.ravel()


def stft_mag_residual(
    sim: np.ndarray,
    real: np.ndarray,
    sr: int,
    *,
    weight: float = 1.0,
) -> np.ndarray:
    """Residual between sim and real STFT magnitude features."""
    a = stft_mag_features(sim, sr)
    b = stft_mag_features(real, sr)
    n = min(len(a), len(b))
    return (a[:n] - b[:n]) * float(weight)
