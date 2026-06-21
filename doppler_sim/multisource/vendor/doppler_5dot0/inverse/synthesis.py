"""
VS13 multi-source pass-by synthesis (Doppler 5.0).

Per-source q(t′) from real recording → retarded-time propagation → physics combine:
  · Engine + intake + exhaust: distinct q, partial-coherence tonal cluster
  · Tyres + body + aero: independent broadband, power sum
  · Inverse fit: per-source band gains + cluster gains vs real CPA spectrum
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from doppler_4dot0.audio.body_source import recording_spectrum_shape, segment_broadband_buffer
from doppler_4dot0.inverse.spectral import BANDS, band_shares, envelope_fwhm_s
from doppler_4dot0.physics.kinematics import emitter_trajectories, passby_timeline
from doppler_4dot0.physics.point_source import (
    doppler_ratio,
    emission_time_at_observer,
    radial_geometry,
    speed_of_sound,
)
from doppler_4dot0.synthesis import _remove_dc, _render_path
from doppler_5dot0.config import (
    BODY_NEAR_FIELD_M,
    CAR_LENGTH_M,
    DEFAULT_RH,
    DEFAULT_TEMP_C,
    FRONT_TONAL_CROSSOVER_HZ,
    GROUND_REFLECTION_COEFF,
    NEAR_FIELD_M,
    OBSERVER_XYZ,
    SAMPLE_RATE,
    SOURCE_HEIGHT_M,
    TONAL_CROSSOVER_HZ,
    TONAL_HF_INCOHERENCE,
    TONAL_LF_COHERENCE,
    VS13_DCPA_M,
)
from doppler_5dot0.physics.combine import (
    combine_broadband_cluster,
    combine_front_tonal,
    combine_tonal_cluster,
)
from doppler_5dot0.inverse.stft_fit import stft_mag_residual
from doppler_5dot0.sources.broadband import recording_broadband_bed, tire_emission_from_recording
from doppler_5dot0.sources.catalog import SourceSpec, build_source_catalog, split_by_kind
from doppler_5dot0.sources.extract import extract_all_tonal_q


@dataclass
class Vs13MultiCalibration:
    engine_gain: float = 1.0
    intake_gain: float = 0.88
    exhaust_gain: float = 1.05
    global_gain: float = 1.0
    body_gain: float = 0.32
    tire_gain: float = 0.26
    aero_gain: float = 0.22
    bed_gain: float = 0.28
    tonal_lf_coherence: float = TONAL_LF_COHERENCE
    lf_bed_gain: float = 0.0
    level_scale: float = 1.0
    target_band_shares: dict[str, float] = field(default_factory=dict)
    fitted_band_shares: dict[str, float] = field(default_factory=dict)
    real_fwhm_s: float = 0.0
    sim_fwhm_s: float = 0.0


@dataclass
class _SourceCache:
    engine_p: np.ndarray
    intake_p: np.ndarray
    exhaust_p: np.ndarray
    tires: list[np.ndarray]
    body: list[np.ndarray]
    aero: list[np.ndarray]
    bed: np.ndarray
    geoms: dict[str, tuple]
    q_buffers: dict[str, np.ndarray]


def _offsets_from_specs(specs: list[SourceSpec]) -> np.ndarray:
    return np.array([[s.dx, s.dy, s.dz] for s in specs], dtype=np.float64)


def _geom(
    offsets: np.ndarray,
    *,
    speed_mps: float,
    duration_s: float,
    cpa_time_s: float,
    dcpa_m: float,
    travel_sign: float,
    sr: int,
) -> tuple:
    t, _ = passby_timeline(duration_s, sr)
    c = speed_of_sound(DEFAULT_TEMP_C, DEFAULT_RH)
    x, y, z, vx, vy, vz = emitter_trajectories(
        t, body_offsets=offsets, dcpa_m=dcpa_m, speed_mps=speed_mps,
        travel_sign=travel_sign, cpa_time_s=cpa_time_s, source_height_m=SOURCE_HEIGHT_M,
    )
    return t, c, x, y, z, vx, vy, vz, offsets


def _propagate_q(
    q: np.ndarray,
    geom: tuple,
    path_index: int,
    *,
    near_field_m: float = NEAR_FIELD_M,
    include_ground: bool = True,
) -> np.ndarray:
    t, c, x, y, z, vx, vy, vz, _ = geom
    n = len(t)
    r, _ = radial_geometry(
        x[path_index], y[path_index], z[path_index],
        vx[path_index], vy[path_index], vz[path_index], OBSERVER_XYZ,
    )
    p = _render_path(q[:n], t, r, c, near_field_m=near_field_m)
    if include_ground:
        z_img = -np.maximum(z[path_index], 0.01)
        r_g, _ = radial_geometry(
            x[path_index], y[path_index], z_img,
            vx[path_index], vy[path_index], vz[path_index], OBSERVER_XYZ,
        )
        p = p + _render_path(
            q[:n], t, r_g, c, gain_scale=GROUND_REFLECTION_COEFF, near_field_m=near_field_m,
        )
    return p.astype(np.float64)


def _band_templates(
    q: np.ndarray,
    geom: tuple,
    path_index: int,
    sr: int,
    *,
    near_field_m: float,
    include_ground: bool,
) -> list[np.ndarray]:
    """Legacy band templates — unused when full-q propagation is active."""
    del sr
    p = _propagate_q(q, geom, path_index, near_field_m=near_field_m, include_ground=include_ground)
    return [p]


def _propagate_full_q(
    q: np.ndarray,
    geom: tuple,
    *,
    near_field_m: float = NEAR_FIELD_M,
    include_ground: bool = True,
) -> np.ndarray:
    return _propagate_q(q, geom, 0, near_field_m=near_field_m, include_ground=include_ground)


def _precompute_cache(
    real: np.ndarray,
    *,
    speed_mps: float,
    duration_s: float,
    cpa_time_s: float,
    dcpa_m: float,
    travel_sign: float,
    sr: int,
    car_length_m: float,
) -> _SourceCache:
    t, _ = passby_timeline(duration_s, sr)
    n = len(t)
    real = np.asarray(real, dtype=np.float64).ravel()

    catalog = build_source_catalog(car_length_m)
    tonal_specs, tire_specs, body_specs, aero_specs = split_by_kind(catalog)
    q_map = extract_all_tonal_q(real, sr, n, cpa_time_s)

    geoms: dict[str, tuple] = {}
    engine_p = intake_p = exhaust_p = np.array([], dtype=np.float64)

    for spec in tonal_specs:
        off = _offsets_from_specs([spec])
        geom = _geom(
            off, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
            dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
        )
        geoms[spec.id] = geom
        q = q_map[spec.id]
        p_full = _propagate_full_q(q, geom, near_field_m=NEAR_FIELD_M, include_ground=True)
        if spec.id == 'engine':
            engine_p = p_full
        elif spec.id == 'intake':
            intake_p = p_full
        else:
            exhaust_p = p_full

    spec_shape = recording_spectrum_shape(real, sr, max(4096, n))

    tire_off = _offsets_from_specs(tire_specs)
    tire_geom = _geom(
        tire_off, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
    )
    geoms['tires'] = tire_geom
    tires: list[np.ndarray] = []
    for i, tw in enumerate(tire_specs):
        q_tire = tire_emission_from_recording(
            real, sr, n + 2048, speed_mps, tw.id, lo_hz=120.0, hi_hz=3800.0,
        )
        q_tire = np.asarray(q_tire[:n], dtype=np.float64) * float(np.sqrt(tw.gain))
        tires.append(_propagate_q(
            q_tire, tire_geom, i, near_field_m=BODY_NEAR_FIELD_M, include_ground=False,
        ))

    body_off = _offsets_from_specs(body_specs)
    body_geom = _geom(
        body_off, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
    )
    geoms['body'] = body_geom
    body: list[np.ndarray] = []
    for i, bs in enumerate(body_specs):
        q_patch = segment_broadband_buffer(
            bs.id, sr, n, speed_mps=speed_mps, spectrum_shape=spec_shape,
            level=np.sqrt(bs.gain), lo_hz=80.0, hi_hz=4000.0,
        )
        body.append(_propagate_q(
            q_patch, body_geom, i, near_field_m=BODY_NEAR_FIELD_M, include_ground=False,
        ))

    aero_off = _offsets_from_specs(aero_specs)
    aero_geom = _geom(
        aero_off, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
    )
    geoms['aero'] = aero_geom
    aero: list[np.ndarray] = []
    for i, ap in enumerate(aero_specs):
        q_patch = segment_broadband_buffer(
            ap.id, sr, n, speed_mps=speed_mps, spectrum_shape=spec_shape,
            level=np.sqrt(ap.gain), lo_hz=700.0, hi_hz=5500.0,
        )
        aero.append(_propagate_q(
            q_patch, aero_geom, i, near_field_m=BODY_NEAR_FIELD_M, include_ground=False,
        ))

    center_off = np.array([[0.0, 0.0, SOURCE_HEIGHT_M]], dtype=np.float64)
    bed_geom = _geom(
        center_off, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
    )
    geoms['bed'] = bed_geom
    q_bed = recording_broadband_bed(real, sr, n, speed_mps, lo_hz=80.0, hi_hz=4000.0, level=1.0)
    p_bed = _propagate_q(
        q_bed, bed_geom, 0, near_field_m=BODY_NEAR_FIELD_M, include_ground=False,
    )

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


def _tonal_tracks(
    cache: _SourceCache,
    cal: Vs13MultiCalibration,
    sr: int,
) -> dict[str, np.ndarray]:
    g = cal.global_gain
    p_eng = g * cal.engine_gain * cache.engine_p
    p_int = g * cal.intake_gain * cache.intake_p
    p_exh = g * cal.exhaust_gain * cache.exhaust_p

    p_front = combine_front_tonal(
        p_eng, p_int, sr,
        crossover_hz=FRONT_TONAL_CROSSOVER_HZ,
        gamma_low=min(0.95, cal.tonal_lf_coherence + 0.12),
        gamma_high=TONAL_HF_INCOHERENCE + 0.07,
    )
    p_tonal = combine_tonal_cluster(
        p_front, p_exh, sr,
        crossover_hz=TONAL_CROSSOVER_HZ,
        gamma_low=cal.tonal_lf_coherence,
        gamma_high=TONAL_HF_INCOHERENCE,
    )
    return {
        'engine': p_eng,
        'intake': p_int,
        'exhaust': p_exh,
        'powertrain': p_front,
        'tonal': p_tonal,
    }


def _broadband_tracks(cache: _SourceCache, cal: Vs13MultiCalibration, sr: int) -> dict[str, np.ndarray]:
    p_tires = cal.tire_gain * combine_broadband_cluster(cache.tires, sr)
    p_body = cal.body_gain * combine_broadband_cluster(cache.body, sr)
    p_aero = cal.aero_gain * combine_broadband_cluster(cache.aero, sr)
    p_bed = cal.bed_gain * cache.bed
    p_bb = p_tires + p_body + p_aero + p_bed
    return {'tires': p_tires, 'body': p_body, 'aero': p_aero, 'bed': p_bed, 'broadband': p_bb}


def _component_tracks(cache: _SourceCache, cal: Vs13MultiCalibration, sr: int) -> dict[str, np.ndarray]:
    tonal = _tonal_tracks(cache, cal, sr)
    bb = _broadband_tracks(cache, cal, sr)
    n = min(len(tonal['tonal']), len(bb['broadband']))
    combined = tonal['tonal'][:n] + bb['broadband'][:n]
    return {**tonal, **bb, 'combined': combined.astype(np.float64)}


def _lf_bed_from_real(real: np.ndarray, sr: int, n: int, cpa_time_s: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    y = np.asarray(real[:n], dtype=np.float64).ravel()
    cpa_i = int(cpa_time_s * sr)
    margin = int(0.7 * sr)
    edge = np.concatenate([y[: max(0, cpa_i - margin)], y[min(len(y), cpa_i + margin):]])
    if len(edge) < sr // 4:
        edge = y
    b, a = butter(2, [40.0 / (0.5 * sr), 480.0 / (0.5 * sr)], btype='band')
    bed = filtfilt(b, a, edge)
    if len(bed) < n:
        bed = np.tile(bed, int(np.ceil(n / len(bed))))[:n]
    else:
        bed = bed[:n]
    return (bed / (float(np.max(np.abs(bed))) + 1e-12)).astype(np.float64)


def _match_real_level(
    sim: np.ndarray,
    real: np.ndarray,
    sr: int,
    cpa_time_s: float,
    *,
    win_s: float = 1.2,
) -> tuple[np.ndarray, float]:
    i0 = max(0, int((cpa_time_s - win_s) * sr))
    i1 = min(len(real), int((cpa_time_s + win_s) * sr))
    if i1 <= i0:
        return sim, 1.0
    r_rms = float(np.sqrt(np.mean(real[i0:i1] ** 2)))
    s_rms = float(np.sqrt(np.mean(sim[i0:i1] ** 2)))
    if s_rms < 1e-12 or r_rms < 1e-12:
        return sim, 1.0
    scale = r_rms / s_rms
    return sim * scale, scale


def infer_vs13_calibration(
    real: np.ndarray,
    *,
    speed_mps: float,
    duration_s: float,
    cpa_time_s: float,
    sr: int = SAMPLE_RATE,
    dcpa_m: float = VS13_DCPA_M,
    car_length_m: float = CAR_LENGTH_M,
    travel_sign: float = 1.0,
    cache: _SourceCache | None = None,
) -> Vs13MultiCalibration:
    from scipy.optimize import least_squares

    real = np.asarray(real, dtype=np.float64).ravel()
    target = band_shares(real, sr, cpa_time_s - 0.5, cpa_time_s + 0.5)
    real_fwhm, _ = envelope_fwhm_s(real, sr)
    band_names = [b[0] for b in BANDS]

    if cache is None:
        cache = _precompute_cache(
            real, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
            dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr, car_length_m=car_length_m,
        )

    # engine, intake, exhaust, global, body, tire, aero, bed, tonal coherence
    n_x = 9
    x0 = np.array([
        np.log(1.0), np.log(0.88), np.log(1.05), np.log(1.0),
        np.log(0.32), np.log(0.26), np.log(0.22), np.log(0.28), 0.0,
    ], dtype=np.float64)

    y_tgt = np.array([target.get(b, 0.0) for b in band_names])
    i0 = max(0, int((cpa_time_s - 0.8) * sr))
    i1 = min(len(real), int((cpa_time_s + 0.8) * sr))
    real_seg = real[i0:i1]
    r_norm = float(np.sqrt(np.mean(real_seg ** 2)) + 1e-12)

    def _unpack(x: np.ndarray) -> Vs13MultiCalibration:
        return Vs13MultiCalibration(
            engine_gain=float(np.exp(x[0])),
            intake_gain=float(np.exp(x[1])),
            exhaust_gain=float(np.exp(x[2])),
            global_gain=float(np.exp(x[3])),
            body_gain=float(np.exp(x[4])),
            tire_gain=float(np.exp(x[5])),
            aero_gain=float(np.exp(x[6])),
            bed_gain=float(np.exp(x[7])),
            tonal_lf_coherence=float(0.45 + 0.45 / (1.0 + np.exp(-x[8]))),
        )

    def _res(x: np.ndarray) -> np.ndarray:
        cal = _unpack(x)
        combined = _component_tracks(cache, cal, sr)['combined']
        shares = band_shares(combined, sr, cpa_time_s - 0.5, cpa_time_s + 0.5)
        err_shares = np.array([shares.get(b, 0.0) for b in band_names]) - y_tgt
        err_stft = stft_mag_residual(combined, real, sr, weight=1.0)
        seg = combined[i0:i1]
        s_norm = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
        corr = float(np.dot(seg / s_norm, real_seg / r_norm) / len(seg))
        return np.concatenate([err_shares, 0.18 * err_stft, [0.22 * (1.0 - corr)]])

    res = least_squares(_res, x0, bounds=(-2.0, 2.5), max_nfev=50, ftol=1e-3)
    cal = _unpack(res.x)
    tracks = _component_tracks(cache, cal, sr)
    cal.sim_fwhm_s, _ = envelope_fwhm_s(tracks['combined'], sr)
    cal.fitted_band_shares = band_shares(tracks['combined'], sr, cpa_time_s - 0.5, cpa_time_s + 0.5)
    cal.target_band_shares = target
    cal.real_fwhm_s = real_fwhm
    return cal


def _emitter_meta(t: np.ndarray, cache: _SourceCache, c: float) -> list[dict[str, Any]]:
    meta: list[dict[str, Any]] = []
    for sid, geom in cache.geoms.items():
        if sid in ('tires', 'body', 'aero'):
            continue
        x, y, z, vx, vy, vz, offsets = geom[2], geom[3], geom[4], geom[5], geom[6], geom[7], geom[8]
        r, v_r = radial_geometry(x[0], y[0], z[0], vx[0], vy[0], vz[0], OBSERVER_XYZ)
        cpa_i = int(np.argmin(r))
        meta.append({
            'id': sid,
            'kind': 'tonal',
            'body_offset_m': [float(offsets[0, 0]), float(offsets[0, 1]), float(offsets[0, 2])],
            'cpa_time_observer_s': float(t[cpa_i] + r[cpa_i] / c),
            'r_cpa_m': float(r[cpa_i]),
            'freq_ratio_min': float(np.min(doppler_ratio(v_r, c))),
            'freq_ratio_max': float(np.max(doppler_ratio(v_r, c))),
        })
    return meta


def synthesize_vs13_multisource(
    real: np.ndarray,
    *,
    speed_mps: float,
    duration_s: float,
    cpa_time_s: float,
    sr: int = SAMPLE_RATE,
    dcpa_m: float = VS13_DCPA_M,
    car_length_m: float = CAR_LENGTH_M,
    travel_sign: float = 1.0,
    calibration: Vs13MultiCalibration | None = None,
) -> tuple[dict[str, np.ndarray], Vs13MultiCalibration, dict[str, Any]]:
    real = np.asarray(real, dtype=np.float64).ravel()
    cache = _precompute_cache(
        real, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr, car_length_m=car_length_m,
    )
    cal = calibration or infer_vs13_calibration(
        real, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        sr=sr, dcpa_m=dcpa_m, car_length_m=car_length_m, travel_sign=travel_sign,
        cache=cache,
    )
    t, _ = passby_timeline(duration_s, sr)
    c = speed_of_sound(DEFAULT_TEMP_C, DEFAULT_RH)

    tracks = _component_tracks(cache, cal, sr)
    combined, level_scale = _match_real_level(tracks['combined'], real, sr, cpa_time_s)
    edge = np.concatenate([real[: sr // 2], real[-sr // 2:]])
    centre = real[int(0.4 * len(real)): int(0.6 * len(real))]
    lf_gain = min(0.16, max(0.04, float(np.sqrt(np.mean(edge * edge)) / (float(np.max(np.abs(centre))) + 1e-12))))
    combined = combined + lf_gain * float(np.max(np.abs(combined))) * _lf_bed_from_real(real, sr, len(t), cpa_time_s)
    cal.lf_bed_gain = lf_gain
    cal.level_scale = level_scale
    cal.sim_fwhm_s, _ = envelope_fwhm_s(combined, sr)
    cal.fitted_band_shares = band_shares(combined, sr, cpa_time_s - 0.5, cpa_time_s + 0.5)

    scale = level_scale
    out_tracks = {k: v * scale for k, v in tracks.items()}
    out_tracks['combined'] = combined
    for k in out_tracks:
        out_tracks[k] = _remove_dc(out_tracks[k], sr)

    tonal_specs, tire_specs, body_specs, aero_specs = split_by_kind(build_source_catalog(car_length_m))
    meta = {
        'physics': 'inverse_multi_source — q and gains from target VS13 clip',
        'model': 'doppler_5dot0_inverse',
        'mode': 'inverse',
        'uses_target_clip_audio': True,
        'car_length_m': float(car_length_m),
        'source_count': {
            'tonal': len(tonal_specs),
            'tires': len(tire_specs),
            'body': len(body_specs),
            'aero': len(aero_specs),
        },
        'global_gain': cal.global_gain,
        'engine_gain': cal.engine_gain,
        'intake_gain': cal.intake_gain,
        'exhaust_gain': cal.exhaust_gain,
        'body_gain': cal.body_gain,
        'tire_gain': cal.tire_gain,
        'aero_gain': cal.aero_gain,
        'bed_gain': cal.bed_gain,
        'tonal_lf_coherence': cal.tonal_lf_coherence,
        'level_scale': cal.level_scale,
        'lf_bed_gain': cal.lf_bed_gain,
        'dcpa_m': float(dcpa_m),
        'speed_mps': float(speed_mps),
        'annotated_cpa_time_s': float(cpa_time_s),
        'emitters': _emitter_meta(t, cache, c),
        'q_sources': list(cache.q_buffers.keys()),
        'calibration': {
            'engine_gain': cal.engine_gain,
            'intake_gain': cal.intake_gain,
            'exhaust_gain': cal.exhaust_gain,
            'global_gain': cal.global_gain,
            'body_gain': cal.body_gain,
            'tire_gain': cal.tire_gain,
            'aero_gain': cal.aero_gain,
            'bed_gain': cal.bed_gain,
            'tonal_lf_coherence': cal.tonal_lf_coherence,
            'target_band_shares': cal.target_band_shares,
            'fitted_band_shares': cal.fitted_band_shares,
            'real_fwhm_s': cal.real_fwhm_s,
            'sim_fwhm_s': cal.sim_fwhm_s,
            'level_scale': cal.level_scale,
            'lf_bed_gain': cal.lf_bed_gain,
        },
        'c_sound': c,
    }
    return {
        'engine': out_tracks['engine'].astype(np.float32),
        'intake': out_tracks['intake'].astype(np.float32),
        'exhaust': out_tracks['exhaust'].astype(np.float32),
        'powertrain': out_tracks['powertrain'].astype(np.float32),
        'tires': out_tracks['tires'].astype(np.float32),
        'body': out_tracks['body'].astype(np.float32),
        'broadband': out_tracks['broadband'].astype(np.float32),
        'combined': out_tracks['combined'].astype(np.float32),
    }, cal, meta
