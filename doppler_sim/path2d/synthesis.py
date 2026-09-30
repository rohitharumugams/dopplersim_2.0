"""2D free-path pass-by: draw a trajectory, synthesize retarded-time audio."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from doppler_sim.atmosphere import AIR_N_FFT, apply_retarded_air_absorption

SPEED_OF_SOUND = 343.0
# A drawn polyline changes heading in one sample. Emitters sitting off the axle
# would teleport, which ticks the audio and spikes the radial-velocity plot.
# This is the distance over which the body heading is allowed to turn.
HEADING_SMOOTH_M = 1.5


def _polyline_arclength(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        raise ValueError("path must be an (N, 2) array with N >= 2")
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(d)])


def _body_heading(
    tx_seg: np.ndarray,
    ty_seg: np.ndarray,
    seg: np.ndarray,
    moving: np.ndarray,
    speed: float,
    sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Heading for rigid placement. Axle velocity stays on the drawn tangent."""
    tx = np.asarray(tx_seg, dtype=np.float64)[seg]
    ty = np.asarray(ty_seg, dtype=np.float64)[seg]
    idx = np.flatnonzero(moving)
    if idx.size < 4 or speed <= 0.0:
        return tx, ty
    sigma = HEADING_SMOOTH_M * float(sr) / float(speed)
    if sigma < 0.75:
        return tx, ty
    theta = np.unwrap(np.arctan2(ty[idx], tx[idx]))
    theta = gaussian_filter1d(theta, sigma=float(sigma), mode="nearest")
    tx = tx.copy()
    ty = ty.copy()
    tx[idx] = np.cos(theta)
    ty[idx] = np.sin(theta)
    tx[: idx[0]] = tx[idx[0]]
    ty[: idx[0]] = ty[idx[0]]
    tx[idx[-1] + 1 :] = tx[idx[-1]]
    ty[idx[-1] + 1 :] = ty[idx[-1]]
    return tx, ty


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
    # vx, vy follow the drawn line. tx, ty yaw at a finite rate so an offset
    # emitter cannot change place in a single sample.
    htx, hty = _body_heading(tx, ty, seg, moving, v, sr)

    return {
        "t": t,
        "x": x.astype(np.float64),
        "y": y.astype(np.float64),
        "vx": vx.astype(np.float64),
        "vy": vy.astype(np.float64),
        "tx": htx.astype(np.float64),
        "ty": hty.astype(np.float64),
        "s": s_query.astype(np.float64),
        "length_m": np.array([length], dtype=np.float64),
        "duration_s": np.array([duration_s], dtype=np.float64),
        "travel_s": np.array([travel_s], dtype=np.float64),
        "pad_s": np.array([float(pad_s)], dtype=np.float64),
    }


def rigid_emitter_xy(
    x: np.ndarray,
    y: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    offsets_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place vehicle-frame offsets on a planar path.

    ``offsets_xyz`` is (N, 3) in the axle-midpoint frame: +X forward, +Y left, +Z up.
    Heading is the path tangent. Left is (-ty, tx). Height is not returned; it does
    not affect the ground track.
    """
    offsets = np.asarray(offsets_xyz, dtype=np.float64).reshape(-1, 3)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    ox = offsets[:, 0][:, None]
    oy = offsets[:, 1][:, None]
    xe = x[None, :] + ox * tx[None, :] + oy * (-ty)[None, :]
    ye = y[None, :] + ox * ty[None, :] + oy * tx[None, :]
    return xe, ye


def solve_retarded_time_path(
    t_obs: np.ndarray,
    t_src: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    *,
    mic_xy: tuple[float, float] = (0.0, 0.0),
    c: float = SPEED_OF_SOUND,
    z_src: np.ndarray | None = None,
    mic_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve t_r + R(t_r)/c = t_obs on a timed source path (vectorized)."""
    t_obs = np.asarray(t_obs, dtype=np.float64)
    t_src = np.asarray(t_src, dtype=np.float64)
    x_src = np.asarray(x_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    mx, my = float(mic_xy[0]), float(mic_xy[1])
    planar = z_src is None and float(mic_z) == 0.0

    if planar:
        r_src = np.sqrt((x_src - mx) ** 2 + (y_src - my) ** 2)
    else:
        z_arr = np.zeros_like(x_src) if z_src is None else np.asarray(z_src, dtype=np.float64)
        r_src = np.sqrt((x_src - mx) ** 2 + (y_src - my) ** 2 + (z_arr - float(mic_z)) ** 2)
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
    if planar:
        rr = np.sqrt((xr - mx) ** 2 + (yr - my) ** 2)
    else:
        zr = z_arr[j] + alpha * (z_arr[j + 1] - z_arr[j])
        rr = np.sqrt((xr - mx) ** 2 + (yr - my) ** 2 + (zr - float(mic_z)) ** 2)

    t_r[inside] = tr
    r_at[inside] = rr
    return t_r, r_at


def _with_air(
    contribution: np.ndarray,
    distance: np.ndarray,
    air_beta: np.ndarray | None,
    n_fft: int,
) -> np.ndarray:
    """Absorption after the retarded-time warp. None keeps the 1/R sample."""
    if air_beta is None:
        return contribution
    return apply_retarded_air_absorption(contribution, distance, air_beta, n_fft)


def _emitter_path(
    traj: dict[str, np.ndarray],
    offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift centerline path along local tangent by longitudinal offset (m)."""
    x = traj["x"] + float(offset_m) * traj["tx"]
    y = traj["y"] + float(offset_m) * traj["ty"]
    return x, y


def _doppler_from_state(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    t_r: np.ndarray,
    r: np.ndarray,
    *,
    mic_xy: tuple[float, float],
    c: float,
) -> dict[str, np.ndarray]:
    """Radial speed from position and velocity interpolated at emission time.

    ``r`` is the propagation distance already solved for this track, so a
    raised microphone is included there. Vertical speed is zero.
    """
    valid = np.isfinite(t_r) & np.isfinite(r) & (r > 0.0)
    tr = np.where(valid, t_r, t[0])
    x_e = np.interp(tr, t, x)
    y_e = np.interp(tr, t, y)
    vxe = np.interp(tr, t, vx)
    vye = np.interp(tr, t, vy)
    dx = x_e - float(mic_xy[0])
    dy = y_e - float(mic_xy[1])
    v_r = (dx * vxe + dy * vye) / np.maximum(r, 1e-9)
    v_r = np.where(valid, v_r, np.nan)
    alpha = float(c) / (float(c) + v_r)
    return {"t_r": t_r, "R": r, "v_r": v_r, "alpha": alpha}


def _reference_from_axle(
    traj: dict[str, np.ndarray],
    t_r: np.ndarray,
    r: np.ndarray,
    *,
    mic_xy: tuple[float, float],
    mic_z: float,
    c: float,
) -> dict[str, np.ndarray]:
    """Doppler curves for the axle midpoint, using the 3D line of sight."""
    del mic_z  # height is already inside ``r``
    curves = _doppler_from_state(
        traj["t"],
        traj["x"],
        traj["y"],
        traj["vx"],
        traj["vy"],
        t_r,
        r,
        mic_xy=mic_xy,
        c=c,
    )
    curves["x"] = traj["x"]
    curves["y"] = traj["y"]
    return curves


def _synthesize_rigid_emitters(
    traj: dict[str, np.ndarray],
    offsets_xyz: np.ndarray,
    *,
    sr: int,
    freqs: np.ndarray,
    psd: np.ndarray,
    synthesize_psd_noise: Any,
    rng: np.random.Generator,
    mic_xy: tuple[float, float],
    mic_z: float,
    c: float,
    air_beta: np.ndarray | None = None,
    air_n_fft: int = AIR_N_FFT,
) -> dict[str, Any]:
    """Incoherent sum of rigid-body emitters. Each has its own 3D retarded time."""
    offsets = np.asarray(offsets_xyz, dtype=np.float64).reshape(-1, 3)
    if offsets.shape[0] < 1:
        raise ValueError("body layout needs at least one emitter")

    t = traj["t"]
    n_obs = len(t)
    mx, my = float(mic_xy[0]), float(mic_xy[1])
    xe, ye = rigid_emitter_xy(traj["x"], traj["y"], traj["tx"], traj["ty"], offsets)
    weight = 1.0 / np.sqrt(len(offsets))
    output = np.zeros(n_obs, dtype=np.float64)
    emitter_curves: list[dict[str, np.ndarray]] = []

    for i in range(len(offsets)):
        ze = np.full(n_obs, float(offsets[i, 2]), dtype=np.float64)
        t_r, r = solve_retarded_time_path(
            t, t, xe[i], ye[i], mic_xy=mic_xy, c=c, z_src=ze, mic_z=mic_z
        )
        # Rigid-body velocity, including yaw rate on a curve.
        vx_e = np.gradient(xe[i], t)
        vy_e = np.gradient(ye[i], t)
        emitter_curves.append(
            _doppler_from_state(
                t, xe[i], ye[i], vx_e, vy_e, t_r, r, mic_xy=mic_xy, c=c
            )
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
        contribution = _with_air(contribution, r, air_beta, air_n_fft)
        output += weight * contribution

    z_axle = np.zeros(n_obs, dtype=np.float64)
    t_r_ref, r_ref = solve_retarded_time_path(
        t, t, traj["x"], traj["y"], mic_xy=mic_xy, c=c, z_src=z_axle, mic_z=mic_z
    )
    if not np.any(np.isfinite(t_r_ref)):
        raise RuntimeError("no valid retarded-time samples for this path / speed")
    reference = _reference_from_axle(
        traj, t_r_ref, r_ref, mic_xy=mic_xy, mic_z=mic_z, c=c
    )

    peak = float(np.max(np.abs(output)))
    if peak > 1e-12:
        output *= 0.95 / peak

    range_inst = np.sqrt((traj["x"] - mx) ** 2 + (traj["y"] - my) ** 2 + mic_z**2)
    cpa_idx = int(np.argmin(range_inst))
    return {
        "audio": output.astype(np.float64),
        "sr": int(sr),
        "trajectory": traj,
        "quantities": reference,
        "range_inst": range_inst,
        "cpa_time_sec": float(t[cpa_idx]),
        "cpa_distance_m": float(range_inst[cpa_idx]),
        "mic_xy": np.array([mx, my], dtype=np.float64),
        "mic_z": np.array([float(mic_z)], dtype=np.float64),
        "emitter_curves": emitter_curves,
    }


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
    emitter_offsets_xyz: np.ndarray | None = None,
    mic_z: float = 0.0,
    air_beta: np.ndarray | None = None,
    air_n_fft: int = AIR_N_FFT,
) -> dict[str, Any]:
    """Build observer audio for a drawn polyline (Pass-By acoustics on a free path)."""
    if rng is None:
        rng = np.random.default_rng()

    traj = resample_path_constant_speed(xy, speed_mps=speed_mps, sr=sr, pad_s=pad_s)
    if emitter_offsets_xyz is not None:
        result = _synthesize_rigid_emitters(
            traj,
            emitter_offsets_xyz,
            sr=sr,
            freqs=freqs,
            psd=psd,
            synthesize_psd_noise=synthesize_psd_noise,
            rng=rng,
            mic_xy=mic_xy,
            mic_z=float(mic_z),
            c=c,
            air_beta=air_beta,
            air_n_fft=air_n_fft,
        )
        result["path_xy"] = np.asarray(xy, dtype=np.float64)
        return result

    t = traj["t"]
    n_obs = len(t)
    mx, my = float(mic_xy[0]), float(mic_xy[1])

    if num_emitters <= 1:
        offsets = np.array([0.0], dtype=np.float64)
    else:
        offsets = np.linspace(-float(vehicle_length) / 2.0, float(vehicle_length) / 2.0, int(num_emitters))

    weight = 1.0 / np.sqrt(len(offsets))
    output = np.zeros(n_obs, dtype=np.float64)
    any_valid = False

    for off in offsets:
        x_e, y_e = _emitter_path(traj, float(off))
        t_r, r = solve_retarded_time_path(t, t, x_e, y_e, mic_xy=mic_xy, c=c)
        valid = np.isfinite(t_r) & np.isfinite(r) & (r > 0.0)
        if not np.any(valid):
            continue
        any_valid = True

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
        contribution = _with_air(contribution, r, air_beta, air_n_fft)
        output += weight * contribution

    # Plots follow the geometric center (offset 0), interpolated at emission
    # time. Audio above is unchanged and still uses each emitter's own range.
    t_r_ref, r_ref = solve_retarded_time_path(t, t, traj["x"], traj["y"], mic_xy=mic_xy, c=c)
    if not any_valid or not np.any(np.isfinite(t_r_ref)):
        raise RuntimeError("no valid retarded-time samples for this path / speed")
    reference_quantities = _reference_from_axle(
        traj, t_r_ref, r_ref, mic_xy=mic_xy, mic_z=0.0, c=c
    )

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
