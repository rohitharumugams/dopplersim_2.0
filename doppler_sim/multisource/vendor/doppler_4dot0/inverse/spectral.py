"""Frequency-band helpers for VS13 inverse calibration."""

from __future__ import annotations

import numpy as np

BANDS = (
    ('sub200', 0.0, 200.0),
    ('low_200_800', 200.0, 800.0),
    ('mid_800_2k', 800.0, 2000.0),
    ('high_2k_4k', 2000.0, 4000.0),
)


def bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    sig = np.asarray(y, dtype=np.float64).ravel()
    nyq = 0.5 * sr
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.999)
    if hi_n <= lo_n:
        return np.zeros_like(sig)
    b, a = butter(2, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, sig)


def band_shares(y: np.ndarray, sr: int, t0: float, t1: float) -> dict[str, float]:
    """Normalised band energy in time window [t0, t1]."""
    i0 = max(0, int(t0 * sr))
    i1 = min(len(y), int(t1 * sr))
    if i1 <= i0:
        return {name: 0.0 for name, _, _ in BANDS}
    seg = np.asarray(y[i0:i1], dtype=np.float64)
    total = 0.0
    out: dict[str, float] = {}
    for name, lo, hi in BANDS:
        b = bandpass(seg, sr, lo, hi)
        e = float(np.sum(b * b))
        out[name] = e
        total += e
    if total < 1e-20:
        return {name: 0.0 for name, _, _ in BANDS}
    return {k: v / total for k, v in out.items()}


def apply_band_gains(
    q: np.ndarray,
    sr: int,
    gains: dict[str, float],
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Build q_i(t′) = sum_b gain_b · band_b(q)."""
    qin = np.asarray(q, dtype=np.float64).ravel()
    y = np.zeros_like(qin)
    for name, lo, hi in BANDS:
        g = float(gains.get(name, 0.0))
        if g <= 0:
            continue
        y += g * bandpass(qin, sr, lo, hi)
    if normalize:
        peak = float(np.max(np.abs(y)))
        if peak > 1e-8:
            y = y / peak
    return y.astype(np.float32)


def envelope_fwhm_s(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Peak-centred FWHM (same definition as dopplernet analyze_vs13)."""
    from scipy.ndimage import uniform_filter1d

    sig = np.asarray(y, dtype=np.float64).ravel()
    win = max(64, int(0.05 * sr))
    env = np.sqrt(uniform_filter1d(sig * sig, size=win, mode='nearest'))
    peak = float(np.max(env))
    if peak < 1e-12:
        return 0.0, 0.0
    norm = env / peak
    peak_i = int(np.argmax(norm))
    t = np.arange(len(sig)) / float(sr)
    half = 0.5

    before = norm[: peak_i + 1]
    tb = t[: peak_i + 1]
    hit_lo = np.where(before >= half)[0]
    t_lo = float(tb[hit_lo[0]]) if len(hit_lo) else float(t[0])

    after = norm[peak_i:]
    ta = t[peak_i:]
    hit_hi = np.where(after >= half)[0]
    t_hi = float(ta[hit_hi[-1]]) if len(hit_hi) else float(t[-1])

    return float(t_hi - t_lo), float(t[peak_i])
