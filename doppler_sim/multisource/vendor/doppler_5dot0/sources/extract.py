"""
Per-source emission buffers q(t′) from VS13 field recording.

Each tonal source carries the **full harmonic stack** (engine orders) with only
gentle spectral tilt per radiator. Distinct q comes from different time excerpts
and circular offsets — not brick-wall bandpass or LF orthogonalization.
"""

from __future__ import annotations

import numpy as np

from doppler_4dot0.audio.source import (
    emission_level_envelope,
    flatten_rms,
    pad_for_doppler,
    passby_stationary_timbre,
)

# Full tonal range — overlapping across all radiators (no spectral holes)
_TONAL_LO_HZ = 35.0
_TONAL_HI_HZ = 3800.0


def _excerpt(
    passby: np.ndarray,
    sr: int,
    cpa_time_s: float,
    *,
    which: str = 'both',
    margin_s: float = 0.55,
    min_seg_s: float = 0.85,
) -> np.ndarray:
    y = np.asarray(passby, dtype=np.float64).ravel()
    cpa_i = int(float(cpa_time_s) * sr)
    margin = max(sr // 8, int(margin_s * sr))
    min_len = max(sr // 4, int(min_seg_s * sr))

    approach = y[: max(0, cpa_i - margin)]
    recede = y[min(len(y), cpa_i + margin):]

    parts: list[np.ndarray] = []
    if which in ('both', 'approach') and len(approach) >= min_len:
        parts.append(flatten_rms(approach, sr))
    if which in ('both', 'recede') and len(recede) >= min_len:
        parts.append(flatten_rms(recede, sr))

    if not parts:
        return flatten_rms(y, sr)
    return flatten_rms(np.concatenate(parts), sr)


def _full_harmonic_timbre(
    passby: np.ndarray,
    sr: int,
    cpa_time_s: float,
    *,
    which: str = 'both',
) -> np.ndarray:
    """Full engine-order timbre 35 Hz – 3.8 kHz from approach/recession."""
    if which == 'both':
        excerpt = passby_stationary_timbre(passby, sr, cpa_time_s, cpa_margin_s=0.55)
    else:
        excerpt = _excerpt(passby, sr, cpa_time_s, which=which, margin_s=0.60)
    y = np.asarray(excerpt, dtype=np.float64).ravel()
    if len(y) < sr // 4:
        return y.astype(np.float32)
    spec = np.fft.rfft(y)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / float(sr))
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(mag, size=max(3, len(mag) // 128), mode='nearest')
    peak_mask = mag > (1.15 * smooth + 1e-12)
    w = np.ones_like(mag)
    boost = 10.0 ** (4.5 / 20.0)
    w[peak_mask & (freqs >= 60.0) & (freqs <= _TONAL_HI_HZ)] = boost
    w[freqs < 35.0] = 0.35
    lf = (freqs >= 35.0) & (freqs <= 600.0)
    w[lf] = np.maximum(w[lf], boost * 0.85)
    out = np.fft.irfft(spec * w, n=len(y))
    peak = float(np.max(np.abs(out)))
    return (out / peak if peak > 1e-8 else out).astype(np.float32)


def _gentle_tilt_db(
    freqs_hz: np.ndarray,
    *,
    pivot_hz: float,
    slope_db_per_oct: float,
    floor_db: float = -18.0,
    ceiling_db: float = 6.0,
) -> np.ndarray:
    """Smooth log-frequency tilt (dB) — no brick-wall band edges."""
    f = np.maximum(freqs_hz, 1.0)
    oct = np.log2(f / max(pivot_hz, 1.0))
    db = np.clip(slope_db_per_oct * oct, floor_db, ceiling_db)
    return (10.0 ** (db / 20.0)).astype(np.float64)


def _apply_gentle_emphasis(
    x: np.ndarray,
    sr: int,
    *,
    pivot_hz: float,
    slope_db_per_oct: float,
    sub_boost_db: float = 0.0,
    hf_keep_db: float = 0.0,
) -> np.ndarray:
    """Multiplicative FFT emphasis — full harmonic content retained."""
    y = np.asarray(x, dtype=np.float64).ravel()
    n = len(y)
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))

    w = _gentle_tilt_db(freqs, pivot_hz=pivot_hz, slope_db_per_oct=slope_db_per_oct)
    w[freqs < _TONAL_LO_HZ] *= 0.25
    hi = freqs > _TONAL_HI_HZ
    if np.any(hi):
        hf_factor = 10.0 ** (hf_keep_db / 20.0) * np.exp(-(freqs[hi] - _TONAL_HI_HZ) / 900.0)
        w[hi] *= hf_factor

    if sub_boost_db:
        sub = (freqs >= 35.0) & (freqs <= 280.0)
        w[sub] *= 10.0 ** (sub_boost_db / 20.0)

    out = np.fft.irfft(spec * w, n=n)
    peak = float(np.max(np.abs(out)))
    return (out / peak if peak > 1e-8 else out).astype(np.float32)


def _circular_offset(x: np.ndarray, offset_samples: int) -> np.ndarray:
    """Decorrelate spatial sources without removing shared harmonics."""
    n = len(x)
    k = int(offset_samples) % max(n, 1)
    if k == 0:
        return np.asarray(x, dtype=np.float32)
    return np.roll(np.asarray(x, dtype=np.float32), k)


def _seamless_tile(x: np.ndarray, n_samples: int) -> np.ndarray:
    """Crossfade loop — avoids vertical seams from hard np.tile."""
    buf = np.asarray(x, dtype=np.float32).ravel()
    need = max(n_samples, 4096)
    if len(buf) >= need:
        return buf[:need].copy()

    fade = min(len(buf) // 4, 2048)
    out = buf.copy()
    while len(out) < need:
        chunk = buf.copy()
        if fade > 8 and len(out) >= fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            chunk[:fade] = chunk[:fade] * ramp + out[-fade:] * (1.0 - ramp)
        out = np.concatenate([out, chunk[fade:]])
    return out[:need].astype(np.float32)


def _extend_q(
    x: np.ndarray,
    n_samples: int,
    passby: np.ndarray,
    sr: int,
    *,
    retain_level: bool = True,
) -> np.ndarray:
    out = _seamless_tile(x, max(n_samples, 4096))
    if retain_level and len(passby) >= n_samples:
        lev = emission_level_envelope(passby[:n_samples], sr, smooth_s=0.55, min_frac=0.38)
        out = out[: len(lev)] * lev[: len(out)]
    peak = float(np.max(np.abs(out)))
    if peak > 1e-8:
        out = out / peak
    return pad_for_doppler(out, n_samples).astype(np.float32)


def extract_tonal_q(
    passby: np.ndarray,
    sr: int,
    n_samples: int,
    cpa_time_s: float,
    *,
    source_id: str,
) -> np.ndarray:
    """Distinct full-harmonic q(t′) per tonal radiator."""
    if source_id == 'engine':
        x = _full_harmonic_timbre(passby, sr, cpa_time_s, which='both')
        x = _apply_gentle_emphasis(
            x, sr, pivot_hz=450.0, slope_db_per_oct=-0.12,
            sub_boost_db=1.5, hf_keep_db=-1.0,
        )
        x = _circular_offset(x, 0)
    elif source_id == 'intake':
        x = _full_harmonic_timbre(passby, sr, cpa_time_s, which='approach')
        x = _apply_gentle_emphasis(
            x, sr, pivot_hz=750.0, slope_db_per_oct=0.22,
            sub_boost_db=-0.5, hf_keep_db=1.0,
        )
        x = _circular_offset(x, sr // 17 + 113)
    elif source_id == 'exhaust':
        x = _full_harmonic_timbre(passby, sr, cpa_time_s, which='recede')
        x = _apply_gentle_emphasis(
            x, sr, pivot_hz=380.0, slope_db_per_oct=-0.05,
            sub_boost_db=2.0, hf_keep_db=-0.5,
        )
        x = _circular_offset(x, sr // 11 + 307)
    else:
        x = _full_harmonic_timbre(passby, sr, cpa_time_s, which='both')

    return _extend_q(x, n_samples, passby, sr)


def extract_all_tonal_q(
    passby: np.ndarray,
    sr: int,
    n_samples: int,
    cpa_time_s: float,
) -> dict[str, np.ndarray]:
    """Engine, intake, exhaust — full harmonics, distinct excerpts + offsets."""
    return {
        sid: extract_tonal_q(passby, sr, n_samples, cpa_time_s, source_id=sid)
        for sid in ('engine', 'intake', 'exhaust')
    }
