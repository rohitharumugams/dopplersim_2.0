"""
Observer-frame retarded-time monopole (Verma / first-principles).

For each observer sample t_n:

  t′ = emission time solving  t_n = t′ + R(t′) / c
  p_i(t_n) = q(t′) / R_eff(t′)

Doppler, phase, and interference emerge from time-varying R(t′).
Fractional-delay sampling uses bandlimited (upsampled cubic) interpolation
to avoid linear-interp phase noise that masquerades as broadband speckle.
"""

from __future__ import annotations

import numpy as np

from doppler_4dot0.config import NEAR_FIELD_M, OBSERVER_XYZ

# Upsample emission buffer before fractional-delay read (anti-aliased interp)
SOURCE_UPSAMPLE = 8


def speed_of_sound(temp_c: float = 20.0, humidity_rh: float = 50.0) -> float:
    tk = float(temp_c) + 273.15
    rh = max(0.0, min(float(humidity_rh), 100.0)) / 100.0
    es = 611.2 * np.exp(17.67 * float(temp_c) / (float(temp_c) + 243.5))
    pv = min(rh * es, 0.49 * 101325.0)
    w = 0.62198 * pv / (101325.0 - pv + 1e-12)
    return float(np.sqrt(1.4 * 287.058 * tk * (1.0 + 0.61 * w)))


def radial_geometry(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    obs_xyz: tuple[float, float, float] = OBSERVER_XYZ,
) -> tuple[np.ndarray, np.ndarray]:
    ox, oy, oz = obs_xyz
    px = x - ox
    py = y - oy
    pz = z - oz
    r = np.sqrt(px * px + py * py + pz * pz)
    r_safe = np.maximum(r, 1e-9)
    v_r = (vx * px + vy * py + vz * pz) / r_safe
    return r, v_r


def doppler_ratio(v_r: np.ndarray, c_sound: float) -> np.ndarray:
    c = max(1.0, float(c_sound))
    v_r_clamped = np.clip(v_r, -0.95 * c, 0.95 * c)
    return (c / (c + v_r_clamped)).astype(np.float64)


def _effective_range(r: np.ndarray, near_field_m: float | None = None) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    r0 = NEAR_FIELD_M if near_field_m is None else float(near_field_m)
    return np.sqrt(r * r + r0 * r0)


def _cubic_interp(t_grid: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    from scipy.interpolate import CubicSpline

    t_grid = np.asarray(t_grid, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    t_query = np.asarray(t_query, dtype=np.float64)
    out = np.full_like(t_query, np.nan, dtype=np.float64)
    valid = np.isfinite(t_query) & (t_query >= t_grid[0]) & (t_query <= t_grid[-1])
    if not np.any(valid):
        return out
    cs = CubicSpline(t_grid, values, extrapolate=False)
    out[valid] = cs(t_query[valid])
    return out


def emission_time_at_observer(
    t_obs: np.ndarray,
    t_emit: np.ndarray,
    r_emit: np.ndarray,
    c_sound: float,
) -> np.ndarray:
    """Invert t_obs = t′ + R(t′)/c using smooth cubic inverse (monotone pass-by)."""
    t_emit = np.asarray(t_emit, dtype=np.float64)
    r_emit = np.asarray(r_emit, dtype=np.float64)
    t_arrival = t_emit + r_emit / float(c_sound)
    return _cubic_interp(t_arrival, t_emit, np.asarray(t_obs, dtype=np.float64))


def sample_stationary_source(
    source: np.ndarray,
    t_emit_grid: np.ndarray,
    t_query: np.ndarray,
    *,
    upsample: int = SOURCE_UPSAMPLE,
) -> np.ndarray:
    """
    Bandlimited fractional-delay sample of q(t′).

    Upsample source with cubic spline, then linear read on fine grid —
    equivalent to low-pass filtered fractional delay (avoids linear phase noise).
    """
    from scipy.interpolate import CubicSpline

    src = np.asarray(source, dtype=np.float64).ravel()
    t_grid = np.asarray(t_emit_grid, dtype=np.float64).ravel()
    t_q = np.asarray(t_query, dtype=np.float64)
    if len(src) != len(t_grid):
        raise ValueError('source and t_emit_grid length mismatch')

    valid = np.isfinite(t_q) & (t_q >= t_grid[0]) & (t_q <= t_grid[-1])
    out = np.zeros_like(t_q, dtype=np.float64)
    if not np.any(valid):
        return out

    up = max(1, int(upsample))
    if up > 1 and len(src) > up:
        n_fine = (len(src) - 1) * up + 1
        t_fine = np.linspace(t_grid[0], t_grid[-1], n_fine)
        cs = CubicSpline(t_grid, src, extrapolate=False)
        src_fine = cs(t_fine)
        out[valid] = np.interp(t_q[valid], t_fine, src_fine)
    else:
        cs = CubicSpline(t_grid, src, extrapolate=False)
        out[valid] = cs(t_q[valid])
    return out


def render_observer_pressure(
    source: np.ndarray,
    t_emit: np.ndarray,
    r: np.ndarray,
    c_sound: float,
    t_obs: np.ndarray | None = None,
    *,
    relative_gain: float = 1.0,
    gain_scale: np.ndarray | None = None,
    near_field_m: float | None = None,
) -> np.ndarray:
    n = min(len(source), len(r), len(t_emit))
    if n < 4:
        return np.zeros(max(len(r), len(source)), dtype=np.float32)

    t_emit = np.asarray(t_emit[:n], dtype=np.float64)
    r = np.asarray(r[:n], dtype=np.float64)
    src = np.asarray(source[:n], dtype=np.float64)
    t_obs = t_emit if t_obs is None else np.asarray(t_obs[:n], dtype=np.float64)

    t_prime = emission_time_at_observer(t_obs, t_emit, r, c_sound)
    q_at_tprime = sample_stationary_source(src, t_emit, t_prime)
    r_at_tprime = _cubic_interp(t_emit, r, t_prime)

    r_eff = _effective_range(r_at_tprime, near_field_m)
    gain = float(relative_gain) / np.maximum(r_eff, 1e-9)
    if gain_scale is not None:
        gs = _cubic_interp(t_emit, np.asarray(gain_scale[:n], dtype=np.float64), t_prime)
        gs = np.where(np.isfinite(gs), gs, 1.0)
        gain = gain * gs

    p_obs = np.where(np.isfinite(t_prime), q_at_tprime * gain, 0.0)
    return p_obs.astype(np.float32)


render_point_source = render_observer_pressure
