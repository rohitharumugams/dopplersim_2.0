"""Stationary emission-time source q(t′) from VS13 field recording."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def flatten_rms(x: np.ndarray, sr: int, win_ms: float = 50.0) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).ravel()
    if len(y) < sr // 8:
        return y.astype(np.float32)
    win = max(64, int(sr * win_ms / 1000.0))
    env = np.sqrt(uniform_filter1d(y * y, size=win, mode='nearest'))
    flat = y / np.maximum(env, 1e-8)
    peak = float(np.max(np.abs(flat)))
    if peak > 1e-8:
        flat /= peak
    return flat.astype(np.float32)


def approach_timbre_excerpt(
    passby: np.ndarray,
    sr: int,
    *,
    max_excerpt_s: float = 3.5,
    swell_frac: float = 0.38,
) -> np.ndarray:
    """Use quiet pre-CPA approach only — avoids baking pass-by level into q(t′)."""
    y = np.asarray(passby, dtype=np.float64).ravel()
    if len(y) < sr // 8:
        return flatten_rms(y, sr)

    win = max(64, int(sr * 0.05))
    env = np.sqrt(uniform_filter1d(y * y, size=win, mode='nearest'))
    peak_env = float(np.max(env))
    if peak_env < 1e-8:
        return flatten_rms(y, sr)

    thresh = swell_frac * peak_env
    loud = np.where(env > thresh)[0]
    end = int(loud[0]) if len(loud) else len(y)
    end = max(sr // 4, min(end, int(max_excerpt_s * sr)))
    return flatten_rms(y[:end], sr)


def midband_emphasis(x: np.ndarray, sr: int, gain_db: float = 2.5) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).ravel()
    if len(y) < sr // 4:
        return y.astype(np.float32)
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / float(sr))
    w = np.ones_like(freqs)
    mid = (freqs >= 150.0) & (freqs <= 2000.0)
    w[mid] = 10.0 ** (gain_db / 20.0)
    w[freqs < 35.0] = 0.15
    out = np.fft.irfft(spec * w, n=len(y))
    peak = float(np.max(np.abs(out)))
    if peak > 1e-8:
        out = out / peak
    return out.astype(np.float32)


def pad_for_doppler(x: np.ndarray, n: int) -> np.ndarray:
    pad = max(2048, int(n * 0.15))
    return np.pad(np.asarray(x, dtype=np.float32).ravel(), (0, pad), mode='edge')


def emission_level_envelope(
    passby: np.ndarray,
    sr: int,
    *,
    smooth_s: float = 0.35,
    min_frac: float = 0.12,
) -> np.ndarray:
    """
    Slow emission-strength envelope E(t′) from the field recording.

    Vehicles are not stationary-level sources: engine/tire output swells toward CPA.
    Applied to q(t′) before propagation (separate from geometric 1/R).
    """
    y = np.asarray(passby, dtype=np.float64).ravel()
    win = max(64, int(smooth_s * sr))
    env = np.sqrt(uniform_filter1d(y * y, size=win, mode='nearest'))
    peak = float(np.max(env))
    if peak < 1e-12:
        return np.ones_like(y)
    env = env / peak
    floor = float(min_frac)
    return np.maximum(env, floor).astype(np.float64)


def passby_stationary_timbre(
    passby: np.ndarray,
    sr: int,
    cpa_time_s: float,
    *,
    cpa_margin_s: float = 0.55,
    min_seg_s: float = 1.0,
) -> np.ndarray:
    """
    Harmonic timbre from approach + recession (exclude CPA swell).

    Real VS13 horizontal brightness = engine orders present before and after CPA.
    """
    y = np.asarray(passby, dtype=np.float64).ravel()
    if len(y) < sr // 4:
        return flatten_rms(y, sr)

    cpa_i = int(float(cpa_time_s) * sr)
    margin = max(sr // 8, int(cpa_margin_s * sr))
    min_len = max(sr // 4, int(min_seg_s * sr))

    approach = y[: max(0, cpa_i - margin)]
    recede = y[min(len(y), cpa_i + margin):]

    parts: list[np.ndarray] = []
    for seg in (approach, recede):
        if len(seg) >= min_len:
            parts.append(flatten_rms(seg, sr))

    if not parts:
        return flatten_rms(y, sr)

    cat = np.concatenate(parts)
    return flatten_rms(cat, sr)


def tonal_harmonic_emphasis(x: np.ndarray, sr: int, gain_db: float = 4.0) -> np.ndarray:
    """Boost engine-order peaks so spectrogram shows horizontal harmonic brightness."""
    y = np.asarray(x, dtype=np.float64).ravel()
    if len(y) < sr // 4:
        return y.astype(np.float32)
    spec = np.fft.rfft(y)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / float(sr))
    # Emphasise tonal bins above local smoothed spectrum
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(mag, size=max(3, len(mag) // 128), mode='nearest')
    peak_mask = mag > (1.18 * smooth + 1e-12)
    w = np.ones_like(mag)
    boost = 10.0 ** (gain_db / 20.0)
    w[peak_mask & (freqs >= 60.0) & (freqs <= 2800.0)] = boost
    w[freqs < 35.0] = 0.35
    lf = (freqs >= 35.0) & (freqs <= 520.0)
    w[lf] = np.maximum(w[lf], boost * 0.85)
    out = np.fft.irfft(spec * w, n=len(y))
    peak = float(np.max(np.abs(out)))
    if peak > 1e-8:
        out = out / peak
    return out.astype(np.float32)


def emission_buffer_from_recording(
    passby: np.ndarray,
    sr: int,
    n_samples: int,
    *,
    retain_level_envelope: bool = True,
    cpa_time_s: float | None = None,
) -> np.ndarray:
    """q(t′): stationary harmonic timbre + optional slow emission swell."""
    import librosa

    if cpa_time_s is not None:
        x = tonal_harmonic_emphasis(
            passby_stationary_timbre(passby, sr, cpa_time_s), sr,
        )
    else:
        x = tonal_harmonic_emphasis(midband_emphasis(approach_timbre_excerpt(passby, sr), sr), sr)
    need = max(n_samples, 4096)
    if len(x) >= need:
        buf = x[:need].copy()
    else:
        # Tile stationary timbre (loop) rather than time-stretch — preserves harmonics
        reps = int(np.ceil(need / len(x)))
        buf = np.tile(x, reps)[:need].astype(np.float32)
    if retain_level_envelope and len(passby) >= n_samples:
        lev = emission_level_envelope(
            passby[:n_samples], sr, smooth_s=0.55, min_frac=0.42,
        )
        buf = buf[: len(lev)] * lev[: len(buf)]
    peak = float(np.max(np.abs(buf)))
    if peak > 1e-8:
        buf = buf / peak
    return pad_for_doppler(buf, n_samples).astype(np.float32)
