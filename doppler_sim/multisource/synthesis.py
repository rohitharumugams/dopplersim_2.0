"""
Multi-source pass-by synthesis using Doppler 5.0's 14-source catalog and combine rules.

Extracts per-source emission q(t′) from the upload using original pass-by metadata (v₁, t_CPA,1),
propagates each catalog source at the render geometry (v₂, h₂, t_CPA,2), then combines with the
same partial-coherence tonal cluster + incoherent broadband rules as doppler_5dot0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from doppler_sim.multisource.bootstrap import ensure_vendor_on_path

ensure_vendor_on_path()

from doppler_4dot0.audio.body_source import recording_spectrum_shape, segment_broadband_buffer
from doppler_4dot0.physics.kinematics import passby_timeline
from doppler_4dot0.physics.point_source import speed_of_sound
from doppler_4dot0.synthesis import _remove_dc
from doppler_5dot0.config import (
    BODY_NEAR_FIELD_M,
    DEFAULT_RH,
    DEFAULT_TEMP_C,
    NEAR_FIELD_M,
    SAMPLE_RATE,
    SOURCE_HEIGHT_M,
)
from doppler_5dot0.inverse.synthesis import (
    Vs13MultiCalibration,
    _SourceCache,
    _component_tracks,
    _geom,
    _lf_bed_from_real,
    _match_real_level,
    _offsets_from_specs,
    _propagate_q,
    infer_vs13_calibration,
)
from doppler_5dot0.sources.broadband import recording_broadband_bed, tire_emission_from_recording
from doppler_5dot0.sources.catalog import build_source_catalog, split_by_kind
from doppler_5dot0.sources.extract import extract_all_tonal_q

# Match Pass-By Sim 2.0: emission q extends before min retarded time and after max.
PREROLL_MARGIN_S = 3.0


def _emit_grid_bounds_for_paths(
    geom: tuple,
    path_indices: list[int],
    c: float,
    *,
    margin_s: float = PREROLL_MARGIN_S,
    include_ground: bool = False,
) -> tuple[float, float]:
    """
    Emission-time span needed for causal coverage of observer window [0, T].

    Returns (t_start, t_end_emit) where t_end_emit >= T so pad_for_doppler tail samples
    are reachable at late observer times (5.0 appends ~15% edge-held q after T).
    """
    from doppler_4dot0.physics.point_source import radial_geometry
    from doppler_5dot0.config import OBSERVER_XYZ

    t, _, x, y, z, vx, vy, vz, _ = geom
    t = np.asarray(t, dtype=np.float64)
    min_arrival = np.inf
    max_arrival = -np.inf
    for idx in path_indices:
        r, _ = radial_geometry(
            x[idx], y[idx], z[idx],
            vx[idx], vy[idx], vz[idx], OBSERVER_XYZ,
        )
        arrivals = t + r / c
        min_arrival = min(min_arrival, float(np.min(arrivals)))
        max_arrival = max(max_arrival, float(np.max(arrivals)))
        if include_ground:
            z_img = -np.maximum(z[idx], 0.01)
            r_g, _ = radial_geometry(
                x[idx], y[idx], z_img,
                vx[idx], vy[idx], vz[idx], OBSERVER_XYZ,
            )
            arrivals_g = t + r_g / c
            min_arrival = min(min_arrival, float(np.min(arrivals_g)))
            max_arrival = max(max_arrival, float(np.max(arrivals_g)))
    if not np.isfinite(min_arrival):
        return 0.0, float(t[-1])
    if min_arrival <= 0.0:
        t_start = -float(margin_s)
    else:
        t_start = float(-min_arrival - margin_s)
    # Observer hears arrivals up to ~T; emissions may occur slightly before last arrival.
    t_end_emit = max(float(t[-1]), float(max_arrival) + float(margin_s))
    return t_start, t_end_emit


def _build_q_emit_series(
    q: np.ndarray,
    t: np.ndarray,
    t_start: float,
    t_end_emit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build emission buffer on uniform grid covering [t_start, t_end_emit].

    Uses the full q buffer (incl. pad_for_doppler). Prepends seamless-tiled q for t < 0
    (same crossfade loop as 5.0 extract). Appends edge-held tail when t_end_emit exceeds q.
    """
    from doppler_5dot0.sources.extract import _seamless_tile

    q = np.asarray(q, dtype=np.float64).ravel()
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.asarray(t, dtype=np.float64).ravel()
    dt = float(np.mean(np.diff(t)))
    t0 = float(t[0])
    t_q_end = t0 + (len(q) - 1) * dt

    n_pre = max(0, int(np.ceil((t0 - t_start) / dt)))
    if n_pre > 0:
        q_pre = np.nan_to_num(
            _seamless_tile(q, n_pre).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0,
        )
        q_body = np.concatenate([q_pre, q])
    else:
        q_body = q.copy()
    t_body0 = t0 - n_pre * dt
    t_body_end = t_body0 + (len(q_body) - 1) * dt

    n_post = max(0, int(np.ceil((t_end_emit - max(t_q_end, t_body_end)) / dt)))
    if n_post > 0:
        q_body = np.concatenate([q_body, np.full(n_post, q[-1], dtype=np.float64)])

    t_emit = t_body0 + np.arange(len(q_body), dtype=np.float64) * dt
    if t_emit[-1] < t_end_emit - 0.25 * dt:
        extra = int(np.ceil((t_end_emit - t_emit[-1]) / dt))
        q_body = np.concatenate([q_body, np.full(extra, q[-1], dtype=np.float64)])
        t_emit = t_body0 + np.arange(len(q_body), dtype=np.float64) * dt

    return q_body.astype(np.float64), t_emit


def _kinematics_on_emit_grid(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    path_index: int,
    t_emit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constant-velocity extrapolation of one path onto arbitrary emission times."""
    t = np.asarray(t, dtype=np.float64).ravel()
    t_emit = np.asarray(t_emit, dtype=np.float64).ravel()
    xi = np.asarray(x[path_index], dtype=np.float64)
    yi = np.asarray(y[path_index], dtype=np.float64)
    zi = np.asarray(z[path_index], dtype=np.float64)
    vxi = float(vx[path_index, 0])
    vyi = float(vy[path_index, 0])
    vzi = float(vz[path_index, 0])
    xe = xi[0] + vxi * (t_emit - t[0])
    ye = yi[0] + vyi * (t_emit - t[0])
    ze = zi[0] + vzi * (t_emit - t[0])
    return xe, ye, ze, np.full_like(t_emit, vxi), np.full_like(t_emit, vyi), np.full_like(t_emit, vzi)


def _emit_grid_start_for_paths(
    geom: tuple,
    path_indices: list[int],
    c: float,
    *,
    margin_s: float = PREROLL_MARGIN_S,
    include_ground: bool = False,
) -> float:
    """Backward-compatible helper — earliest emission-grid time."""
    t_start, _ = _emit_grid_bounds_for_paths(
        geom, path_indices, c, margin_s=margin_s, include_ground=include_ground,
    )
    return t_start


def _propagate_q_preroll(
    q: np.ndarray,
    geom: tuple,
    path_index: int,
    *,
    margin_s: float = PREROLL_MARGIN_S,
    near_field_m: float = NEAR_FIELD_M,
    include_ground: bool = True,
) -> np.ndarray:
    """
    Retarded-time propagation with full q(t′) including pad_for_doppler and causal margins.

    Stock _propagate_q uses q[:n] on [0, T] only; pad_for_doppler (~15% edge-held tail) is
    never reachable. This builds [t_start, t_end] emission grids with seamless pre-roll tiling.
    """
    from doppler_4dot0.physics.point_source import radial_geometry, render_observer_pressure
    from doppler_5dot0.config import GROUND_REFLECTION_COEFF, OBSERVER_XYZ

    t, c, x, y, z, vx, vy, vz, _ = geom
    n = len(t)
    q = np.asarray(q, dtype=np.float64).ravel()
    t_start, t_end_emit = _emit_grid_bounds_for_paths(
        geom, [path_index], c, margin_s=margin_s, include_ground=include_ground,
    )
    dt = float(np.mean(np.diff(t)))
    t_q_end = float(t[0]) + (len(q) - 1) * dt
    t_end_emit = max(t_end_emit, t_q_end)

    needs_extended = t_start < float(t[0]) - 1e-9 or len(q) > n or t_end_emit > float(t[-1]) + 1e-9
    if not needs_extended:
        return _propagate_q(
            q,
            geom,
            path_index,
            near_field_m=near_field_m,
            include_ground=include_ground,
        )

    q_emit, t_emit = _build_q_emit_series(q, t, t_start, t_end_emit)
    q_emit = np.nan_to_num(q_emit, nan=0.0, posinf=0.0, neginf=0.0)
    xe, ye, ze, vxe, vye, vze = _kinematics_on_emit_grid(
        t, x, y, z, vx, vy, vz, path_index, t_emit,
    )
    t_obs = np.asarray(t[:n], dtype=np.float64)

    r, _ = radial_geometry(xe, ye, ze, vxe, vye, vze, OBSERVER_XYZ)
    p = render_observer_pressure(
        q_emit, t_emit, r, c, t_obs=t_obs, near_field_m=near_field_m,
    )
    if include_ground:
        z_img = -np.maximum(ze, 0.01)
        r_g, _ = radial_geometry(xe, ye, z_img, vxe, vye, vze, OBSERVER_XYZ)
        p = p + render_observer_pressure(
            q_emit,
            t_emit,
            r_g,
            c,
            t_obs=t_obs,
            relative_gain=GROUND_REFLECTION_COEFF,
            near_field_m=near_field_m,
        )
    return np.asarray(p[:n], dtype=np.float64)


def _propagate_full_q_preroll(
    q: np.ndarray,
    geom: tuple,
    *,
    margin_s: float = PREROLL_MARGIN_S,
    near_field_m: float = NEAR_FIELD_M,
    include_ground: bool = True,
) -> np.ndarray:
    return _propagate_q_preroll(
        q,
        geom,
        0,
        margin_s=margin_s,
        near_field_m=near_field_m,
        include_ground=include_ground,
    )


def _estimate_render_preroll_s(
    geoms: dict[str, tuple],
    catalog_counts: dict[str, int],
    c: float,
    *,
    margin_s: float = PREROLL_MARGIN_S,
) -> float:
    """Earliest emission-grid start used across all sources (for metadata)."""
    starts: list[float] = []
    for key, geom in geoms.items():
        if key in ("engine", "intake", "exhaust", "bed"):
            starts.append(
                _emit_grid_start_for_paths(
                    geom, [0], c, margin_s=margin_s, include_ground=key != "bed",
                )
            )
        elif key in ("tires", "body", "aero"):
            n_paths = catalog_counts.get(key, 1)
            starts.append(
                _emit_grid_start_for_paths(
                    geom, list(range(n_paths)), c, margin_s=margin_s, include_ground=False,
                )
            )
    return min(starts) if starts else 0.0


@dataclass
class MultisourceRenderParams:
    """Geometry bridge from DopplerSim 2.0 RenderParams."""

    v1_mps: float
    t_cpa1_s: float
    h1_m: float
    v2_mps: float
    t_cpa2_s: float
    h2_m: float
    t_out_s: float
    vehicle_length_m: float
    travel_sign: float = 1.0


def _prepare_recording(y: np.ndarray, sr: int, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    if sr != SAMPLE_RATE:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    if len(y) >= n:
        return y[:n]
    pad = n - len(y)
    return np.pad(y, (0, pad))


def _precompute_cache_retarget(
    real: np.ndarray,
    *,
    extract_speed_mps: float,
    extract_cpa_time_s: float,
    render_speed_mps: float,
    render_cpa_time_s: float,
    render_dcpa_m: float,
    duration_s: float,
    travel_sign: float,
    sr: int,
    car_length_m: float,
) -> _SourceCache:
    """Extract q using original pass-by context; propagate at render geometry."""
    t, _ = passby_timeline(duration_s, sr)
    n = len(t)
    real = np.asarray(real, dtype=np.float64).ravel()[:n]

    catalog = build_source_catalog(car_length_m)
    tonal_specs, tire_specs, body_specs, aero_specs = split_by_kind(catalog)
    q_map = extract_all_tonal_q(real, sr, n, extract_cpa_time_s)

    geoms: dict[str, tuple] = {}
    engine_p = intake_p = exhaust_p = np.array([], dtype=np.float64)

    for spec in tonal_specs:
        off = _offsets_from_specs([spec])
        geom = _geom(
            off,
            speed_mps=render_speed_mps,
            duration_s=duration_s,
            cpa_time_s=render_cpa_time_s,
            dcpa_m=render_dcpa_m,
            travel_sign=travel_sign,
            sr=sr,
        )
        geoms[spec.id] = geom
        q = q_map[spec.id]
        p_full = _propagate_full_q_preroll(q, geom, near_field_m=NEAR_FIELD_M, include_ground=True)
        if spec.id == "engine":
            engine_p = p_full
        elif spec.id == "intake":
            intake_p = p_full
        else:
            exhaust_p = p_full

    spec_shape = recording_spectrum_shape(real, sr, max(4096, n))

    tire_off = _offsets_from_specs(tire_specs)
    tire_geom = _geom(
        tire_off,
        speed_mps=render_speed_mps,
        duration_s=duration_s,
        cpa_time_s=render_cpa_time_s,
        dcpa_m=render_dcpa_m,
        travel_sign=travel_sign,
        sr=sr,
    )
    geoms["tires"] = tire_geom
    tires: list[np.ndarray] = []
    for i, tw in enumerate(tire_specs):
        q_tire = tire_emission_from_recording(
            real,
            sr,
            n + 2048,
            extract_speed_mps,
            tw.id,
            lo_hz=120.0,
            hi_hz=3800.0,
        )
        q_tire = np.asarray(q_tire[:n], dtype=np.float64) * float(np.sqrt(tw.gain))
        tires.append(
            _propagate_q_preroll(
                q_tire,
                tire_geom,
                i,
                near_field_m=BODY_NEAR_FIELD_M,
                include_ground=False,
            )
        )

    body_off = _offsets_from_specs(body_specs)
    body_geom = _geom(
        body_off,
        speed_mps=render_speed_mps,
        duration_s=duration_s,
        cpa_time_s=render_cpa_time_s,
        dcpa_m=render_dcpa_m,
        travel_sign=travel_sign,
        sr=sr,
    )
    geoms["body"] = body_geom
    body: list[np.ndarray] = []
    for i, bs in enumerate(body_specs):
        q_patch = segment_broadband_buffer(
            bs.id,
            sr,
            n,
            speed_mps=extract_speed_mps,
            spectrum_shape=spec_shape,
            level=np.sqrt(bs.gain),
            lo_hz=80.0,
            hi_hz=4000.0,
        )
        body.append(
            _propagate_q_preroll(
                q_patch,
                body_geom,
                i,
                near_field_m=BODY_NEAR_FIELD_M,
                include_ground=False,
            )
        )

    aero_off = _offsets_from_specs(aero_specs)
    aero_geom = _geom(
        aero_off,
        speed_mps=render_speed_mps,
        duration_s=duration_s,
        cpa_time_s=render_cpa_time_s,
        dcpa_m=render_dcpa_m,
        travel_sign=travel_sign,
        sr=sr,
    )
    geoms["aero"] = aero_geom
    aero: list[np.ndarray] = []
    for i, ap in enumerate(aero_specs):
        q_patch = segment_broadband_buffer(
            ap.id,
            sr,
            n,
            speed_mps=extract_speed_mps,
            spectrum_shape=spec_shape,
            level=np.sqrt(ap.gain),
            lo_hz=700.0,
            hi_hz=5500.0,
        )
        aero.append(
            _propagate_q_preroll(
                q_patch,
                aero_geom,
                i,
                near_field_m=BODY_NEAR_FIELD_M,
                include_ground=False,
            )
        )

    center_off = np.array([[0.0, 0.0, SOURCE_HEIGHT_M]], dtype=np.float64)
    bed_geom = _geom(
        center_off,
        speed_mps=render_speed_mps,
        duration_s=duration_s,
        cpa_time_s=render_cpa_time_s,
        dcpa_m=render_dcpa_m,
        travel_sign=travel_sign,
        sr=sr,
    )
    geoms["bed"] = bed_geom
    q_bed = recording_broadband_bed(real, sr, n, extract_speed_mps, lo_hz=80.0, hi_hz=4000.0, level=1.0)
    p_bed = _propagate_q_preroll(q_bed, bed_geom, 0, near_field_m=BODY_NEAR_FIELD_M, include_ground=False)

    return _SourceCache(
        engine_p=engine_p,
        intake_p=intake_p,
        exhaust_p=exhaust_p,
        tires=tires,
        body=body,
        aero=aero,
        bed=p_bed,
        geoms=geoms,
        q_buffers=q_map,
    )


def render_params_from_doppler(params) -> MultisourceRenderParams:
    """Map DopplerSim RenderParams to multisource geometry."""
    travel_sign = 1.0 if float(params.v2) >= 0.0 else -1.0
    return MultisourceRenderParams(
        v1_mps=float(params.v1),
        t_cpa1_s=float(params.t_cpa1),
        h1_m=float(params.h1),
        v2_mps=float(abs(params.v2)),
        t_cpa2_s=float(params.t_cpa2),
        h2_m=max(0.01, float(params.h2)),
        t_out_s=float(params.t_out),
        vehicle_length_m=float(params.vehicle_length),
        travel_sign=travel_sign,
    )


def synthesize_multisource_passby(
    real: np.ndarray,
    sr: int,
    params,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """
    Run 5.0 inverse multi-source pipeline with optional geometry retarget.

    Returns (combined_audio, component_tracks, meta).
    """
    ms = render_params_from_doppler(params)
    duration_s = ms.t_out_s
    n = max(4, int(round(duration_s * SAMPLE_RATE)))
    real = _prepare_recording(real, sr, n)
    sr = SAMPLE_RATE

    cache = _precompute_cache_retarget(
        real,
        extract_speed_mps=ms.v1_mps,
        extract_cpa_time_s=ms.t_cpa1_s,
        render_speed_mps=ms.v2_mps,
        render_cpa_time_s=ms.t_cpa2_s,
        render_dcpa_m=ms.h2_m,
        duration_s=duration_s,
        travel_sign=ms.travel_sign,
        sr=sr,
        car_length_m=ms.vehicle_length_m,
    )

    cal = infer_vs13_calibration(
        real,
        speed_mps=ms.v2_mps,
        duration_s=duration_s,
        cpa_time_s=ms.t_cpa2_s,
        sr=sr,
        dcpa_m=ms.h2_m,
        car_length_m=ms.vehicle_length_m,
        travel_sign=ms.travel_sign,
        cache=cache,
    )

    tracks = _component_tracks(cache, cal, sr)
    combined, level_scale = _match_real_level(tracks["combined"], real, sr, ms.t_cpa2_s)
    edge = np.concatenate([real[: sr // 2], real[-sr // 2 :]])
    centre = real[int(0.4 * len(real)) : int(0.6 * len(real))]
    lf_gain = min(
        0.16,
        max(
            0.04,
            float(np.sqrt(np.mean(edge * edge)) / (float(np.max(np.abs(centre))) + 1e-12)),
        ),
    )
    combined = combined + lf_gain * float(np.max(np.abs(combined))) * _lf_bed_from_real(
        real, sr, n, ms.t_cpa2_s
    )
    cal.lf_bed_gain = lf_gain
    cal.level_scale = level_scale

    scale = level_scale
    out_tracks: dict[str, np.ndarray] = {k: v * scale for k, v in tracks.items()}
    out_tracks["combined"] = combined
    for key in out_tracks:
        out_tracks[key] = _remove_dc(out_tracks[key], sr)

    tonal_specs, tire_specs, body_specs, aero_specs = split_by_kind(
        build_source_catalog(ms.vehicle_length_m)
    )
    c = speed_of_sound(DEFAULT_TEMP_C, DEFAULT_RH)
    preroll_start_s = _estimate_render_preroll_s(
        cache.geoms,
        {"tires": len(tire_specs), "body": len(body_specs), "aero": len(aero_specs)},
        c,
    )
    meta: dict[str, Any] = {
        "physics": "doppler_5dot0_inverse_multisource",
        "model": "14_source_catalog",
        "emission_preroll": {
            "margin_s": PREROLL_MARGIN_S,
            "emit_grid_start_s": preroll_start_s,
            "uses_full_q_buffer": True,
        },
        "source_count": {
            "tonal": len(tonal_specs),
            "tires": len(tire_specs),
            "body": len(body_specs),
            "aero": len(aero_specs),
            "total_catalog": len(tonal_specs) + len(tire_specs) + len(body_specs) + len(aero_specs),
        },
        "combine": {
            "tonal_front": "combine_front_tonal (engine + intake)",
            "tonal_cluster": "combine_tonal_cluster (front + exhaust)",
            "broadband": "combine_broadband_cluster (√Σp²) for tyres/body/aero",
            "final": "tonal_cluster + broadband_cluster + LF bed",
        },
        "extract_geometry": {
            "v1_mps": ms.v1_mps,
            "h1_m": ms.h1_m,
            "t_cpa1_s": ms.t_cpa1_s,
        },
        "render_geometry": {
            "v2_mps": ms.v2_mps,
            "h2_m": ms.h2_m,
            "t_cpa2_s": ms.t_cpa2_s,
            "travel_sign": ms.travel_sign,
        },
        "calibration": {
            "engine_gain": cal.engine_gain,
            "intake_gain": cal.intake_gain,
            "exhaust_gain": cal.exhaust_gain,
            "global_gain": cal.global_gain,
            "body_gain": cal.body_gain,
            "tire_gain": cal.tire_gain,
            "aero_gain": cal.aero_gain,
            "bed_gain": cal.bed_gain,
            "tonal_lf_coherence": cal.tonal_lf_coherence,
            "level_scale": cal.level_scale,
            "lf_bed_gain": cal.lf_bed_gain,
            "target_band_shares": cal.target_band_shares,
            "fitted_band_shares": cal.fitted_band_shares,
        },
        "c_sound": c,
    }

    combined_out = out_tracks["combined"].astype(np.float64)
    component_out = {k: np.asarray(v, dtype=np.float64) for k, v in out_tracks.items()}
    return combined_out, component_out, meta
