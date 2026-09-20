"""2D free-path pass-by: draw a trajectory, synthesize retarded-time audio."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

SPEED_OF_SOUND = 343.0


def _polyline_arclength(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        raise ValueError("path must be an (N, 2) array with N >= 2")
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_path_constant_speed(
    xy: np.ndarray,
    *,
    speed_mps: float,
    sr: int,
    pad_s: float = 0.25,
) -> dict[str, np.ndarray]:
    """Parameterize a polyline at constant speed; return timed state samples."""
    xy = np.asarray(xy, dtype=np.float64)
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(xy, axis=0)) > 1e-9, axis=1)
    xy = xy[keep]
    if len(xy) < 2:
        raise ValueError("path needs at least two distinct points")

    s = _polyline_arclength(xy)
    length = float(s[-1])
    if length < 1e-3:
        raise ValueError("path length is too short (draw a longer path)")

    v = float(speed_mps)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("speed_mps must be positive")

    travel_s = length / v
    duration_s = travel_s + 2.0 * float(pad_s)
    n = max(int(np.ceil(duration_s * sr)), 1)
    t = np.arange(n, dtype=np.float64) / float(sr)

    t_motion = np.clip(t - float(pad_s), 0.0, travel_s)
    s_query = v * t_motion

    x = np.interp(s_query, s, xy[:, 0])
    y = np.interp(s_query, s, xy[:, 1])

    ds = np.diff(s)
    dxy = np.diff(xy, axis=0)
    ds_safe = np.maximum(ds, 1e-12)
    tx = dxy[:, 0] / ds_safe
    ty = dxy[:, 1] / ds_safe
    seg = np.searchsorted(s, s_query, side="right") - 1
    seg = np.clip(seg, 0, len(tx) - 1)
    moving = (t_motion > 0.0) & (t_motion < travel_s)
    vx = np.where(moving, v * tx[seg], 0.0)
    vy = np.where(moving, v * ty[seg], 0.0)

    return {
        "t": t,
        "x": x.astype(np.float64),
        "y": y.astype(np.float64),
        "vx": vx.astype(np.float64),
        "vy": vy.astype(np.float64),
        "tx": np.where(moving, tx[seg], tx[seg]).astype(np.float64),
        "ty": np.where(moving, ty[seg], ty[seg]).astype(np.float64),
        "s": s_query.astype(np.float64),
        "length_m": np.array([length], dtype=np.float64),
        "duration_s": np.array([duration_s], dtype=np.float64),
        "travel_s": np.array([travel_s], dtype=np.float64),
        "pad_s": np.array([float(pad_s)], dtype=np.float64),
    }


def solve_retarded_time_path(
    t_obs: np.ndarray,
    t_src: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    *,
    mic_xy: tuple[float, float] = (0.0, 0.0),
    c: float = SPEED_OF_SOUND,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve t_r + R(t_r)/c = t_obs on a timed source path (vectorized)."""
    t_obs = np.asarray(t_obs, dtype=np.float64)
    t_src = np.asarray(t_src, dtype=np.float64)
    x_src = np.asarray(x_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    mx, my = float(mic_xy[0]), float(mic_xy[1])

    r_src = np.sqrt((x_src - mx) ** 2 + (y_src - my) ** 2)
    f_src = t_src + r_src / float(c)

    t_r = np.full(len(t_obs), np.nan, dtype=np.float64)
    r_at = np.full(len(t_obs), np.nan, dtype=np.float64)

    f_min = float(f_src[0])
    f_max = float(f_src[-1])
    inside = (t_obs >= f_min) & (t_obs <= f_max)
    if not np.any(inside):
        return t_r, r_at

    to = t_obs[inside]
    j = np.searchsorted(f_src, to, side="right") - 1
    j = np.clip(j, 0, len(f_src) - 2)
    f0 = f_src[j]
    f1 = f_src[j + 1]
    denom = f1 - f0
    alpha = np.where(np.abs(denom) < 1e-15, 0.0, (to - f0) / denom)
    alpha = np.clip(alpha, 0.0, 1.0)

    tr = t_src[j] + alpha * (t_src[j + 1] - t_src[j])
    xr = x_src[j] + alpha * (x_src[j + 1] - x_src[j])
    yr = y_src[j] + alpha * (y_src[j + 1] - y_src[j])
    rr = np.sqrt((xr - mx) ** 2 + (yr - my) ** 2)

    t_r[inside] = tr
    r_at[inside] = rr
    return t_r, r_at


def _emitter_path(
    traj: dict[str, np.ndarray],
    offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift centerline path along local tangent by longitudinal offset (m)."""
    x = traj["x"] + float(offset_m) * traj["tx"]
    y = traj["y"] + float(offset_m) * traj["ty"]
    return x, y


def synthesize_path_audio(
    xy: np.ndarray,
    *,
    speed_mps: float,
    sr: int,
    freqs: np.ndarray,
    psd: np.ndarray,
    synthesize_psd_noise: Any,
    rng: np.random.Generator | None = None,
    mic_xy: tuple[float, float] = (0.0, 0.0),
    pad_s: float = 0.25,
    c: float = SPEED_OF_SOUND,
    vehicle_length: float = 4.5,
    num_emitters: int = 1,
) -> dict[str, Any]:
    """Build observer audio for a drawn polyline (Pass-By acoustics on a free path)."""
    if rng is None:
        rng = np.random.default_rng()

    traj = resample_path_constant_speed(xy, speed_mps=speed_mps, sr=sr, pad_s=pad_s)
    t = traj["t"]
    n_obs = len(t)
    mx, my = float(mic_xy[0]), float(mic_xy[1])

    if num_emitters <= 1:
        offsets = np.array([0.0], dtype=np.float64)
    else:
        offsets = np.linspace(-float(vehicle_length) / 2.0, float(vehicle_length) / 2.0, int(num_emitters))

    weight = 1.0 / np.sqrt(len(offsets))
    output = np.zeros(n_obs, dtype=np.float64)
    reference_quantities: dict[str, np.ndarray] | None = None

    for off in offsets:
        x_e, y_e = _emitter_path(traj, float(off))
        t_r, r = solve_retarded_time_path(t, t, x_e, y_e, mic_xy=mic_xy, c=c)
        valid = np.isfinite(t_r) & np.isfinite(r) & (r > 0.0)
        if not np.any(valid):
            continue

        t_r_min = float(np.nanmin(t_r[valid])) - 1.0
        t_r_max = float(np.nanmax(t_r[valid])) + 1.0
        source_start = min(t_r_min, 0.0)
        source_end = max(t_r_max, float(t[-1]))
        source_len = max(int(np.ceil((source_end - source_start) * sr)), 1)
        emitter_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        source = synthesize_psd_noise(freqs, psd, source_len, emitter_rng)
        source_times = source_start + np.arange(len(source), dtype=np.float64) / float(sr)
        src_interp = interp1d(source_times, source, bounds_error=False, fill_value=0.0)

        contribution = np.zeros(n_obs, dtype=np.float64)
        contribution[valid] = src_interp(t_r[valid]) / np.maximum(r[valid], 1e-9)
        output += weight * contribution

        if reference_quantities is None:
            # Radial velocity / Doppler from centerline kinematics at emission.
            # Use instantaneous centerline state nearest to t_r.
            idx = np.clip(np.searchsorted(t, t_r, side="left"), 0, n_obs - 1)
            x_c = traj["x"][idx]
            y_c = traj["y"][idx]
            vx = traj["vx"][idx]
            vy = traj["vy"][idx]
            dx = x_c - mx
            dy = y_c - my
            r_safe = np.maximum(r, 1e-9)
            v_r = (dx * vx + dy * vy) / r_safe
            v_r = np.where(valid, v_r, np.nan)
            # du/dt = c/(c + v_r); v_r = (r · v)/R, receding-positive.
            alpha = float(c) / (float(c) + v_r)
            reference_quantities = {
                "t_r": t_r,
                "R": r,
                "v_r": v_r,
                "alpha": alpha,
                "x": traj["x"],
                "y": traj["y"],
            }

    if reference_quantities is None:
        raise RuntimeError("no valid retarded-time samples for this path / speed")

    peak = float(np.max(np.abs(output)))
    if peak > 1e-12:
        output *= 0.95 / peak

    range_inst = np.sqrt((traj["x"] - mx) ** 2 + (traj["y"] - my) ** 2)
    cpa_idx = int(np.argmin(range_inst))

    return {
        "audio": output.astype(np.float64),
        "sr": int(sr),
        "trajectory": traj,
        "quantities": reference_quantities,
        "range_inst": range_inst,
        "cpa_time_sec": float(t[cpa_idx]),
        "cpa_distance_m": float(range_inst[cpa_idx]),
        "mic_xy": np.array([mx, my], dtype=np.float64),
        "path_xy": np.asarray(xy, dtype=np.float64),
    }
