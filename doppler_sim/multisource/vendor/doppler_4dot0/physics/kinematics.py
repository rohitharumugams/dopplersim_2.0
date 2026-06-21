"""
Straight-line CPA pass-by — vehicle as a single point source.

Observer at (0, 0, z_obs). Source travels along x at constant speed v,
with lateral offset y = dCPA. Vehicle centre crosses x = 0 at t = t_CPA.
"""

from __future__ import annotations

import numpy as np


def passby_timeline(duration_s: float, sr: int) -> tuple[np.ndarray, float]:
    n = max(4, int(round(duration_s * sr)))
    dt = duration_s / n
    t = np.arange(n, dtype=np.float64) * dt
    return t, dt


def point_source_trajectory(
    t: np.ndarray,
    *,
    dcpa_m: float,
    speed_mps: float,
    travel_sign: float,
    cpa_time_s: float | None,
    source_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single emitter at vehicle centre (legacy helper)."""
    x, y, z, vx, vy, vz = emitter_trajectories(
        t,
        body_offsets=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        dcpa_m=dcpa_m,
        speed_mps=speed_mps,
        travel_sign=travel_sign,
        cpa_time_s=cpa_time_s,
        source_height_m=source_height_m,
    )
    return x[0], y[0], z[0], vx[0], vy[0], vz[0]


def emitter_trajectories(
    t: np.ndarray,
    *,
    body_offsets: np.ndarray,
    dcpa_m: float,
    speed_mps: float,
    travel_sign: float,
    cpa_time_s: float | None,
    source_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rigid-body monopoles: vehicle centre on CPA path, each offset fixed in body frame.

    body_offsets : (M, 3) with (dx, dy, dz); dx > 0 is toward the front of the car.
    Returns x, y, z, vx, vy, vz each shape (M, N).
    """
    t = np.asarray(t, dtype=np.float64)
    offsets = np.asarray(body_offsets, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError('body_offsets must be (M, 3)')

    sign = 1.0 if float(travel_sign) >= 0 else -1.0
    v = abs(float(speed_mps))
    dcpa = max(0.01, float(dcpa_m))
    t_cpa = float(t[-1] / 2.0) if cpa_time_s is None else float(cpa_time_s)
    z_src = max(0.01, float(source_height_m))

    x_c = sign * v * (t - t_cpa)
    y_c = np.full_like(t, dcpa)
    z_c = np.full_like(t, z_src)
    vx_c = np.full_like(t, sign * v)
    vy_c = np.zeros_like(t)
    vz_c = np.zeros_like(t)

    dx = offsets[:, 0:1] * sign
    dy = offsets[:, 1:2]
    dz = offsets[:, 2:3]

    m = offsets.shape[0]
    x = x_c[np.newaxis, :] + dx
    y = y_c[np.newaxis, :] + dy
    z = z_c[np.newaxis, :] + dz
    vx = np.broadcast_to(vx_c, (m, len(t)))
    vy = np.broadcast_to(vy_c, (m, len(t)))
    vz = np.broadcast_to(vz_c, (m, len(t)))
    return x, y, z, vx, vy, vz
