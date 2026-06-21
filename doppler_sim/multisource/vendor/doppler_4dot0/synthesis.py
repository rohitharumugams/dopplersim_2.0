"""Retarded-time render helpers (subset vendored from doppler_4dot0.synthesis)."""

from __future__ import annotations

import numpy as np

from doppler_4dot0.physics.point_source import render_observer_pressure


def _remove_dc(y: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    sig = np.asarray(y, dtype=np.float64).ravel()
    if len(sig) < sr // 4:
        return sig.astype(np.float32)
    b, a = butter(2, 25.0 / (0.5 * sr), btype="high")
    return filtfilt(b, a, sig).astype(np.float32)


def _render_path(
    src: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    c: float,
    *,
    gain_scale: float | np.ndarray | None = None,
    near_field_m: float | None = None,
) -> np.ndarray:
    gs = None if gain_scale is None else (
        np.full(len(t), float(gain_scale)) if np.isscalar(gain_scale) else gain_scale
    )
    return render_observer_pressure(
        src, t, r, c, t, gain_scale=gs, near_field_m=near_field_m,
    )
