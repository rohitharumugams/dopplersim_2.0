"""Atmospheric absorption from Bass et al., JASA 97, 680–683 (1995).

β_air(f) is in nepers per metre. It is not the Doppler factor α. The 1996
erratum (JASA 99, 1259) was not folded in; these coefficients reproduce the
1995 worked example (1.078 dB/100 m at 1 kHz, 20 °C, 10%, 0.5 atm).

Sound speed is left alone. Temperature, humidity, and pressure build β only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import hann

DB_PER_NEPER = 20.0 / np.log(10.0)
T0_K = 293.15
T01_K = 273.16
P_S0_ATM = 1.0
AIR_N_FFT = 4096
AIR_HOP = AIR_N_FFT // 2


def air_absorption_np_per_m(
    frequencies_hz: np.ndarray,
    temperature_c: float,
    relative_humidity_percent: float,
    pressure_atm: float,
) -> np.ndarray:
    """Bass Eq. 3. ``relative_humidity_percent`` is a percent, so 50 means 50%."""
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    temperature_k = float(temperature_c) + 273.15
    pressure = float(pressure_atm)
    if temperature_k <= 0.0 or not np.isfinite(temperature_k):
        raise ValueError("temperature_c must be above absolute zero")
    if not np.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("pressure_atm must be positive")
    humidity = float(relative_humidity_percent)
    if not np.isfinite(humidity) or humidity < 0.0:
        raise ValueError("relative_humidity_percent must be non-negative")

    p_sat = 10.0 ** (-6.8346 * (T01_K / temperature_k) ** 1.261 + 4.6151)
    h = humidity * (p_sat / P_S0_ATM) / (pressure / P_S0_ATM)
    f_r_o = 24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
    t_ratio = T0_K / temperature_k
    f_r_n = np.sqrt(t_ratio) * (
        9.0 + 280.0 * h * np.exp(-4.17 * (t_ratio ** (1.0 / 3.0) - 1.0))
    )
    scaled = freqs / pressure
    scaled_sq = scaled * scaled
    classical = 1.84e-11 * np.sqrt(temperature_k / T0_K)
    oxygen = 0.01278 * np.exp(-2239.1 / temperature_k) / (f_r_o + scaled_sq / f_r_o)
    nitrogen = 0.1068 * np.exp(-3352.0 / temperature_k) / (f_r_n + scaled_sq / f_r_n)
    relax = (temperature_k / T0_K) ** (-2.5) * (oxygen + nitrogen)
    beta = pressure * scaled_sq * (classical + relax)
    beta = np.where(freqs <= 0.0, 0.0, beta)
    return np.maximum(beta, 0.0)


def absorption_db_per_m(beta_np_per_m: np.ndarray) -> np.ndarray:
    return DB_PER_NEPER * np.asarray(beta_np_per_m, dtype=np.float64)


@dataclass(frozen=True)
class Atmosphere:
    """One clip's air. Disabled leaves propagation exactly as 1/R."""

    temperature_c: float = 20.0
    relative_humidity_percent: float = 50.0
    pressure_atm: float = 1.0
    enabled: bool = True
    max_undo_db: float = 30.0

    def beta(self, frequencies_hz: np.ndarray) -> np.ndarray:
        return air_absorption_np_per_m(
            frequencies_hz,
            self.temperature_c,
            self.relative_humidity_percent,
            self.pressure_atm,
        )

    def undo_gain(self, frequencies_hz: np.ndarray, distance_m: float) -> np.ndarray:
        """e^{+βR} used to invert absorption, capped in decibels."""
        beta = self.beta(frequencies_hz)
        cap = float(self.max_undo_db) / DB_PER_NEPER
        return np.exp(np.minimum(beta * float(distance_m), cap))

    def forward_beta(self, sr: int, n_fft: int = AIR_N_FFT) -> np.ndarray:
        freqs = np.fft.rfftfreq(int(n_fft), d=1.0 / float(sr))
        return self.beta(freqs)


def apply_retarded_air_absorption(
    waveform: np.ndarray,
    distance_m: np.ndarray,
    beta: np.ndarray,
    n_fft: int = AIR_N_FFT,
) -> np.ndarray:
    """Time-varying magnitude filter |H(f,t)| = exp(-β(f) R(t)).

    R is the retarded distance already used for 1/R. Frame centers carry R,
    so the loss follows the pass-by instead of one equalizer for the whole clip.
    """
    y = np.asarray(waveform, dtype=np.float64)
    distance = np.asarray(distance_m, dtype=np.float64)
    if y.shape != distance.shape:
        raise ValueError("waveform and distance must have the same shape")
    if y.size == 0 or float(np.max(np.abs(y))) < 1e-20:
        return y.copy()

    n_fft = int(n_fft)
    hop = n_fft // 2
    window = hann(n_fft, sym=False)
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape != (n_fft // 2 + 1,):
        raise ValueError("beta must match rfft bins of n_fft")

    pad = n_fft // 2
    y_pad = np.pad(y, (pad, pad))
    r_pad = np.pad(distance, (pad, pad), mode="edge")
    n_pad = len(y_pad)
    n_frames = 1 + max(n_pad - n_fft, 0) // hop
    acc = np.zeros(n_pad, dtype=np.float64)
    wsum = np.zeros(n_pad, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        if start + n_fft > n_pad:
            break
        r_c = float(r_pad[start + n_fft // 2])
        segment = y_pad[start : start + n_fft] * window
        spectrum = np.fft.rfft(segment)
        if np.isfinite(r_c) and r_c > 0.0:
            spectrum *= np.exp(-beta * r_c)
        restored = np.fft.irfft(spectrum, n=n_fft) * window
        acc[start : start + n_fft] += restored
        wsum[start : start + n_fft] += window * window
    out = np.zeros(n_pad, dtype=np.float64)
    valid = wsum > 1e-12
    out[valid] = acc[valid] / wsum[valid]
    return out[pad : pad + len(y)]
