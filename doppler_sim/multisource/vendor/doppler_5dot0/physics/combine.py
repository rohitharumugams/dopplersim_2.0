"""Frequency-dependent combination rules for multi-source pass-by."""

from __future__ import annotations

import numpy as np

from doppler_4dot0.inverse.body_combine import combine_body_segments, combine_cluster_pair


def power_sum(pressures: list[np.ndarray]) -> np.ndarray:
    if not pressures:
        return np.array([], dtype=np.float64)
    if len(pressures) == 1:
        return np.asarray(pressures[0], dtype=np.float64).ravel()
    stack = np.stack([np.asarray(p, dtype=np.float64).ravel() for p in pressures], axis=0)
    n = min(row.shape[0] for row in stack)
    stack = stack[:, :n]
    return np.sqrt(np.sum(stack * stack, axis=0)).astype(np.float64)


def combine_front_tonal(
    p_engine: np.ndarray,
    p_intake: np.ndarray,
    sr: int,
    *,
    crossover_hz: float = 1100.0,
    gamma_low: float = 0.85,
    gamma_high: float = 0.35,
) -> np.ndarray:
    """Engine + intake: coherent LF (shared crank), incoherent HF (path decorrelation)."""
    return combine_cluster_pair(
        p_engine, p_intake, sr,
        gamma_low=gamma_low, gamma_high=gamma_high, crossover_hz=crossover_hz,
    )


def combine_tonal_cluster(
    p_front: np.ndarray,
    p_exhaust: np.ndarray,
    sr: int,
    *,
    crossover_hz: float = 850.0,
    gamma_low: float = 0.72,
    gamma_high: float = 0.28,
) -> np.ndarray:
    """Front bay + exhaust: partial LF coherence, HF power blend."""
    return combine_cluster_pair(
        p_front, p_exhaust, sr,
        gamma_low=gamma_low, gamma_high=gamma_high, crossover_hz=crossover_hz,
    )


def combine_broadband_cluster(pressures: list[np.ndarray], sr: int) -> np.ndarray:
    """Tyres, body, aero — incoherent √(Σp²)."""
    return combine_body_segments(pressures, sr, mode='incoherent')
