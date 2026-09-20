"""3D free-path pass-by: draw a trajectory in (x, y, z), synthesize retarded-time audio."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

SPEED_OF_SOUND = 343.0


def _polyline_arclength(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 2:
        raise ValueError("path must be an (N, 3) array with N >= 2")
    d = np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_path_constant_speed(
    xyz: np.ndarray,
    *,
    speed_mps: float,
    sr: int,
    pad_s: float = 0.25,
) -> dict[str, np.ndarray]:
    """Parameterize a 3D polyline at constant speed."""
    xyz = np.asarray(xyz, dtype=np.float64)
    keep = np.ones(len(xyz), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(xyz, axis=0)) > 1e-9, axis=1)
    xyz = xyz[keep]
    if len(xyz) < 2:
        raise ValueError("path needs at least two distinct points")

    s = _polyline_arclength(xyz)
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

    x = np.interp(s_query, s, xyz[:, 0])
    y = np.interp(s_query, s, xyz[:, 1])
    z = np.interp(s_query, s, xyz[:, 2])

    ds = np.diff(s)
    dxyz = np.diff(xyz, axis=0)
    ds_safe = np.maximum(ds, 1e-12)
    tx = dxyz[:, 0] / ds_safe
    ty = dxyz[:, 1] / ds_safe
    tz = dxyz[:, 2] / ds_safe
    seg = np.searchsorted(s, s_query, side="right") - 1
    seg = np.clip(seg, 0, len(tx) - 1)
    moving = (t_motion > 0.0) & (t_motion < travel_s)
    vx = np.where(moving, v * tx[seg], 0.0)
    vy = np.where(moving, v * ty[seg], 0.0)
    vz = np.where(moving, v * tz[seg], 0.0)

    return {
        "t": t,
        "x": x.astype(np.float64),
        "y": y.astype(np.float64),
        "z": z.astype(np.float64),
        "vx": vx.astype(np.float64),
        "vy": vy.astype(np.float64),
        "vz": vz.astype(np.float64),
        "tx": tx[seg].astype(np.float64),
        "ty": ty[seg].astype(np.float64),
        "tz": tz[seg].astype(np.float64),
        "s": s_query.astype(np.float64),
        "length_m": np.array([length], dtype=np.float64),
        "duration_s": np.array([duration_s], dtype=np.float64),
        "travel_s": np.array([travel_s], dtype=np.float64),
        "pad_s": np.array([float(pad_s)], dtype=np.float64),
    }


def solve_retarded_time_path3d(
    t_obs: np.ndarray,
    t_src: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    z_src: np.ndarray,
    *,
    mic_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    c: float = SPEED_OF_SOUND,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve t_r + R(t_r)/c = t_obs for a timed 3D source path."""
    t_obs = np.asarray(t_obs, dtype=np.float64)
    t_src = np.asarray(t_src, dtype=np.float64)
    x_src = np.asarray(x_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    z_src = np.asarray(z_src, dtype=np.float64)
    mx, my, mz = float(mic_xyz[0]), float(mic_xyz[1]), float(mic_xyz[2])

    r_src = np.sqrt((x_src - mx) ** 2 + (y_src - my) ** 2 + (z_src - mz) ** 2)
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
    zr = z_src[j] + alpha * (z_src[j + 1] - z_src[j])
    rr = np.sqrt((xr - mx) ** 2 + (yr - my) ** 2 + (zr - mz) ** 2)

    t_r[inside] = tr
    r_at[inside] = rr
    return t_r, r_at


def synthesize_path3d_audio(
    xyz: np.ndarray,
    *,
    speed_mps: float,
    sr: int,
    freqs: np.ndarray,
    psd: np.ndarray,
    synthesize_psd_noise: Any,
    rng: np.random.Generator | None = None,
    mic_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pad_s: float = 0.25,
    c: float = SPEED_OF_SOUND,
    vehicle_length: float = 4.5,
    num_emitters: int = 1,
) -> dict[str, Any]:
    """Build observer audio for a drawn 3D polyline."""
    if rng is None:
        rng = np.random.default_rng()

    traj = resample_path_constant_speed(xyz, speed_mps=speed_mps, sr=sr, pad_s=pad_s)
    t = traj["t"]
    n_obs = len(t)
    mx, my, mz = float(mic_xyz[0]), float(mic_xyz[1]), float(mic_xyz[2])

    if num_emitters <= 1:
        offsets = np.array([0.0], dtype=np.float64)
    else:
        offsets = np.linspace(-float(vehicle_length) / 2.0, float(vehicle_length) / 2.0, int(num_emitters))

    weight = 1.0 / np.sqrt(len(offsets))
    output = np.zeros(n_obs, dtype=np.float64)
    reference_quantities: dict[str, np.ndarray] | None = None

    for off in offsets:
        x_e = traj["x"] + float(off) * traj["tx"]
        y_e = traj["y"] + float(off) * traj["ty"]
        z_e = traj["z"] + float(off) * traj["tz"]
        t_r, r = solve_retarded_time_path3d(
            t, t, x_e, y_e, z_e, mic_xyz=mic_xyz, c=c
        )
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
            idx = np.clip(np.searchsorted(t, t_r, side="left"), 0, n_obs - 1)
            dx = traj["x"][idx] - mx
            dy = traj["y"][idx] - my
            dz = traj["z"][idx] - mz
            vx = traj["vx"][idx]
            vy = traj["vy"][idx]
            vz = traj["vz"][idx]
            r_safe = np.maximum(r, 1e-9)
            v_r = (dx * vx + dy * vy + dz * vz) / r_safe
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
                "z": traj["z"],
            }

    if reference_quantities is None:
        raise RuntimeError("no valid retarded-time samples for this path / speed")

    peak = float(np.max(np.abs(output)))
    if peak > 1e-12:
        output *= 0.95 / peak

    range_inst = np.sqrt(
        (traj["x"] - mx) ** 2 + (traj["y"] - my) ** 2 + (traj["z"] - mz) ** 2
    )
    cpa_idx = int(np.argmin(range_inst))

    return {
        "audio": output.astype(np.float64),
        "sr": int(sr),
        "trajectory": traj,
        "quantities": reference_quantities,
        "range_inst": range_inst,
        "cpa_time_sec": float(t[cpa_idx]),
        "cpa_distance_m": float(range_inst[cpa_idx]),
        "mic_xyz": np.array([mx, my, mz], dtype=np.float64),
        "path_xyz": np.asarray(xyz, dtype=np.float64),
    }
