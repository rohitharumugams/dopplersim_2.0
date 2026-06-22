"""
Forward pass-by synthesis — no content from target VS13 clip.

Uses vehicle library audio + analytic broadband + physics calibration only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from doppler_4dot0.physics.kinematics import passby_timeline
from doppler_4dot0.physics.point_source import speed_of_sound
from doppler_4dot0.synthesis import _remove_dc
from doppler_5dot0.config import (
    AERO_SPEED_EXPONENT,
    BODY_SPEED_EXPONENT,
    CAR_LENGTH_M,
    DEFAULT_CAR_MODEL_ID,
    DEFAULT_RH,
    DEFAULT_TEMP_C,
    NEAR_FIELD_M,
    SAMPLE_RATE,
    SOURCE_HEIGHT_M,
    TIRE_SPEED_EXPONENT,
    VS13_DCPA_M,
    WAKE_SPEED_EXPONENT,
)
from doppler_5dot0.forward.calibration import forward_calibration
from doppler_5dot0.inverse.synthesis import (
    Vs13MultiCalibration,
    _SourceCache,
    _component_tracks,
    _emitter_meta,
    _geom,
    _offsets_from_specs,
    _propagate_full_q,
    _tire_index,
)
from doppler_5dot0.physics.passby_envelope import speed_dependent_gain
from doppler_5dot0.physics.propagation import (
    propagate_aero_patch,
    propagate_body_patch,
    propagate_tire,
    propagate_wake,
)
from doppler_5dot0.sources.broadband import turbulence_wake_emission
from doppler_5dot0.sources.catalog import build_source_catalog, split_by_kind
from doppler_5dot0.sources.forward import (
    aero_q_forward,
    bed_q_forward,
    body_q_forward,
    extract_all_tonal_q_forward,
    load_library_audio,
    tire_q_forward,
)
from dopplernet_corrected.models.car_models import get_car_model


def _car_length_m(car_model_id: str) -> float:
    try:
        return float(get_car_model(car_model_id).get('length_m', CAR_LENGTH_M))
    except KeyError:
        return CAR_LENGTH_M


def _precompute_forward_cache(
    *,
    speed_mps: float,
    duration_s: float,
    cpa_time_s: float,
    dcpa_m: float,
    travel_sign: float,
    sr: int,
    car_model_id: str,
    audio_file: str | None = None,
) -> _SourceCache:
    t, _ = passby_timeline(duration_s, sr)
    n = len(t)
    car_length_m = _car_length_m(car_model_id)
    model = get_car_model(car_model_id)
    lib_file = audio_file or str(model['audio_file'])
    library = load_library_audio(lib_file, sr)

    catalog = build_source_catalog(car_length_m)
    tonal_specs, tire_specs, body_specs, aero_specs, wake_specs = split_by_kind(catalog)
    q_map = extract_all_tonal_q_forward(
        library, sr, n,
        cpa_time_s=cpa_time_s,
    )

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

    tire_speed_gain = speed_dependent_gain(speed_mps, exponent=TIRE_SPEED_EXPONENT)
    body_speed_gain = speed_dependent_gain(speed_mps, exponent=BODY_SPEED_EXPONENT)
    aero_speed_gain = speed_dependent_gain(speed_mps, exponent=AERO_SPEED_EXPONENT)
    wake_speed_gain = speed_dependent_gain(speed_mps, exponent=WAKE_SPEED_EXPONENT)

    tire_off = _offsets_from_specs(tire_specs)
    tire_geom = _geom(
        tire_off, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
    )
    geoms['tires'] = tire_geom
    tires: list[np.ndarray] = []
    for tw in tire_specs:
        q_tire = np.asarray(
            tire_q_forward(library, sr, n + 2048, speed_mps, tw.id)[:n], dtype=np.float64,
        )
        q_tire = q_tire * float(np.sqrt(tw.gain)) * tire_speed_gain
        tires.append(propagate_tire(q_tire, tire_geom, _tire_index(tire_specs, tw.id)))

    body_off = _offsets_from_specs(body_specs)
    body_geom = _geom(
        body_off, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
    )
    geoms['body'] = body_geom
    body: list[np.ndarray] = []
    for i, bs in enumerate(body_specs):
        q_patch = body_q_forward(library, sr, n, speed_mps, bs.id)
        q_patch = np.asarray(q_patch, dtype=np.float64) * float(np.sqrt(bs.gain)) * body_speed_gain
        body.append(propagate_body_patch(q_patch, body_geom, i))

    aero_off = _offsets_from_specs(aero_specs)
    aero_geom = _geom(
        aero_off, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
    )
    geoms['aero'] = aero_geom
    aero: list[np.ndarray] = []
    for i, ap in enumerate(aero_specs):
        q_patch = aero_q_forward(library, sr, n, speed_mps, ap.id)
        q_patch = np.asarray(q_patch, dtype=np.float64) * float(np.sqrt(ap.gain)) * aero_speed_gain
        aero.append(propagate_aero_patch(q_patch, aero_geom, i))

    wake_off = _offsets_from_specs(wake_specs)
    wake_geom = _geom(
        wake_off, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
    )
    geoms['wake'] = wake_geom
    wake: list[np.ndarray] = []
    for i, ws in enumerate(wake_specs):
        q_wake = turbulence_wake_emission(
            sr, n, speed_mps, ws.id, level=float(np.sqrt(ws.gain)) * wake_speed_gain,
        )
        wake.append(propagate_wake(np.asarray(q_wake[:n], dtype=np.float64), wake_geom, i))

    center_off = np.array([[0.0, 0.0, SOURCE_HEIGHT_M]], dtype=np.float64)
    bed_geom = _geom(
        center_off, speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
    )
    geoms['bed'] = bed_geom
    q_bed = bed_q_forward(library, sr, n, speed_mps)
    p_bed = propagate_body_patch(np.asarray(q_bed[:n], dtype=np.float64), bed_geom, 0)

    return _SourceCache(
        engine_p=engine_p,
        intake_p=intake_p,
        exhaust_p=exhaust_p,
        tires=tires,
        body=body,
        aero=aero,
        wake=wake,
        bed=p_bed,
        geoms=geoms,
        q_buffers=q_map,
    )


def _norm_peak(y: np.ndarray, peak: float = 0.92) -> np.ndarray:
    sig = np.asarray(y, dtype=np.float64).ravel()
    p = float(np.max(np.abs(sig)))
    if p < 1e-12:
        return sig.astype(np.float32)
    return (sig / p * peak).astype(np.float32)


def synthesize_forward_passby(
    *,
    speed_mps: float,
    duration_s: float,
    cpa_time_s: float,
    sr: int = SAMPLE_RATE,
    dcpa_m: float = VS13_DCPA_M,
    travel_sign: float = 1.0,
    car_model_id: str = DEFAULT_CAR_MODEL_ID,
    audio_file: str | None = None,
    calibration: Vs13MultiCalibration | None = None,
    normalize_peak: bool = True,
) -> tuple[dict[str, np.ndarray], Vs13MultiCalibration, dict[str, Any]]:
    """Generic forward pass-by — library + physics only."""
    cal = calibration or forward_calibration(speed_mps)
    cache = _precompute_forward_cache(
        speed_mps=speed_mps, duration_s=duration_s, cpa_time_s=cpa_time_s,
        dcpa_m=dcpa_m, travel_sign=travel_sign, sr=sr,
        car_model_id=car_model_id, audio_file=audio_file,
    )
    t, _ = passby_timeline(duration_s, sr)
    c = speed_of_sound(DEFAULT_TEMP_C, DEFAULT_RH)

    tracks = _component_tracks(cache, cal, sr)
    combined_raw = tracks['combined'].astype(np.float64)

    if normalize_peak:
        combined = _norm_peak(combined_raw)
        out_tracks = {k: _norm_peak(v) for k, v in tracks.items()}
        out_tracks['combined'] = combined
    else:
        out_tracks = {k: np.asarray(v, dtype=np.float64) for k, v in tracks.items()}
        combined = combined_raw

    for k in out_tracks:
        out_tracks[k] = _remove_dc(out_tracks[k], sr)

    tonal_specs, tire_specs, body_specs, aero_specs, wake_specs = split_by_kind(
        build_source_catalog(_car_length_m(car_model_id)),
    )
    model = get_car_model(car_model_id)
    meta = {
        'physics': 'forward_multi_source — Cevher ES envelope physics (horn, dipole, wake)',
        'model': 'doppler_5dot0_forward',
        'mode': 'forward',
        'car_model_id': car_model_id,
        'library_audio': audio_file or model['audio_file'],
        'car_length_m': _car_length_m(car_model_id),
        'source_count': {
            'tonal': len(tonal_specs),
            'tires': len(tire_specs),
            'body': len(body_specs),
            'aero': len(aero_specs),
            'wake': len(wake_specs),
        },
        'uses_target_clip_audio': False,
        'global_gain': cal.global_gain,
        'engine_gain': cal.engine_gain,
        'intake_gain': cal.intake_gain,
        'exhaust_gain': cal.exhaust_gain,
        'body_gain': cal.body_gain,
        'tire_gain': cal.tire_gain,
        'aero_gain': cal.aero_gain,
        'wake_gain': cal.wake_gain,
        'bed_gain': cal.bed_gain,
        'tonal_lf_coherence': cal.tonal_lf_coherence,
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
            'wake_gain': cal.wake_gain,
            'bed_gain': cal.bed_gain,
            'tonal_lf_coherence': cal.tonal_lf_coherence,
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
        '_combined_raw': combined_raw.astype(np.float32),
    }, cal, meta
