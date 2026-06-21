"""Combine pressures from many body segments with frequency-dependent coherence."""

from __future__ import annotations

import numpy as np


def combine_body_segments(
    pressures: list[np.ndarray],
    sr: int,
    *,
    gamma_low: float = 1.0,
    gamma_high: float = 0.25,
    crossover_hz: float = 1200.0,
    mode: str = 'partial',
) -> np.ndarray:
    """
    Many-segment body combination.

    mode='incoherent' : p = √(Σ p_i²) — broadband body panels (widens envelope)
    mode='partial'    : STFT γ(f) blend of coherent sum and incoherent magnitude
    """
    if not pressures:
        return np.array([], dtype=np.float64)
    if len(pressures) == 1:
        return np.asarray(pressures[0], dtype=np.float64).ravel()

    n = min(len(p) for p in pressures)
    stack = np.stack([np.asarray(p, dtype=np.float64).ravel()[:n] for p in pressures], axis=0)

    if mode == 'incoherent':
        return np.sqrt(np.sum(stack * stack, axis=0)).astype(np.float64)

    if mode == 'hybrid':
        from scipy.signal import butter, filtfilt

        p_coh = np.sum(stack, axis=0)
        p_inc = np.sqrt(np.sum(stack * stack, axis=0))
        wc = min(0.99, crossover_hz / (0.5 * sr))
        b_lo, a_lo = butter(2, wc, btype='low')
        b_hi, a_hi = butter(2, wc, btype='high')
        return (filtfilt(b_lo, a_lo, p_coh) + filtfilt(b_hi, a_hi, p_inc)).astype(np.float64)

    from scipy.signal import stft, istft

    nperseg = 1024
    hop = 256
    Z_list = [stft(stack[i], fs=sr, nperseg=nperseg, noverlap=nperseg - hop)[2] for i in range(len(stack))]
    f = stft(stack[0], fs=sr, nperseg=nperseg, noverlap=nperseg - hop)[0]

    Zcoh = np.sum(Z_list, axis=0)
    Zpow = np.sum(np.abs(np.stack(Z_list, axis=0)) ** 2, axis=0)
    Zinc = np.sqrt(np.maximum(Zpow, 0.0)) * np.exp(1j * np.angle(Zcoh + 1e-12))

    gamma = np.ones_like(f, dtype=np.float64)
    gamma[f >= crossover_hz] = gamma_high
    mid = (f >= crossover_hz * 0.5) & (f < crossover_hz)
    if np.any(mid):
        gamma[mid] = np.linspace(gamma_low, gamma_high, int(np.sum(mid)))

    G = gamma[:, np.newaxis]
    Zout = G * Zcoh + (1.0 - G) * Zinc
    _, out = istft(Zout, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
    return out[:n].astype(np.float64)


def combine_cluster_pair(
    p_rear: np.ndarray,
    p_front: np.ndarray,
    sr: int,
    *,
    gamma_low: float = 1.0,
    gamma_high: float = 0.35,
    crossover_hz: float = 1200.0,
) -> np.ndarray:
    """Two-cluster partial coherence (rear half vs front half of body)."""
    return combine_body_segments(
        [p_rear, p_front], sr,
        gamma_low=gamma_low, gamma_high=gamma_high, crossover_hz=crossover_hz,
    )
