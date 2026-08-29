"""Batch synthesis for 2D whiteboard paths (uses path2d audio, not straight render_pass_by)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from doppler_sim.batch.constants import resolve_project_path
from doppler_sim.batch.path2d_planner import Path2dPlannedSample
from doppler_sim.batch.pipeline import _get_or_compute_signature


def _plan_as_straight_compat(plan: Path2dPlannedSample):
    """Adapt Path2dPlannedSample for the shared signature cache in pipeline.py."""
    from doppler_sim.batch.planner import PlannedSample

    return PlannedSample(
        index=plan.index,
        vehicle=plan.vehicle,
        source_speed_mps=plan.source_speed_mps,
        source_path=plan.source_path,
        speed_mps=plan.speed_mps,
        cpa_distance_m=plan.cpa_distance_m,
        cpa_time_sec=plan.cpa_time_sec,
        h1_m=plan.h1_m,
        t_cpa1_s=plan.t_cpa1_s,
        vehicle_length_m=plan.vehicle_length_m,
        num_emitters=plan.num_emitters,
        t_out_s=plan.t_out_s,
        path_type=plan.path_type,
    )


def synthesize_path2d_planned_sample(
    plan: Path2dPlannedSample,
    *,
    base_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Run synthesize_path_audio using a cached (or freshly computed) source signature."""
    from doppler_sim.application import BASE_DIR, OUTPUT_SR, synthesize_psd_noise
    from doppler_sim.path2d.synthesis import synthesize_path_audio

    root = base_dir or BASE_DIR
    source_path = resolve_project_path(root, plan.source_path)
    compat = _plan_as_straight_compat(plan)

    _uploaded, sr, freqs, psd_observed, psd_inverted, stft, stft_times = (
        _get_or_compute_signature(source_path, compat)
    )

    xy = np.asarray(plan.path_xy, dtype=np.float64)
    result = synthesize_path_audio(
        xy,
        speed_mps=float(plan.speed_mps),
        sr=OUTPUT_SR,
        freqs=freqs,
        psd=psd_inverted,
        synthesize_psd_noise=synthesize_psd_noise,
        mic_xy=(float(plan.mic_x), float(plan.mic_y)),
        vehicle_length=float(plan.vehicle_length_m),
        num_emitters=int(plan.num_emitters),
    )

    aux: dict[str, Any] = {
        "uploaded_sr": sr,
        "output_sr": OUTPUT_SR,
        "freqs": freqs,
        "psd_observed": psd_observed,
        "psd_inverted": psd_inverted,
        "stft": stft,
        "stft_times": stft_times,
        "trajectory": result["trajectory"],
        "path_xy": xy,
        "cpa_time_sec": float(result["cpa_time_sec"]),
        "cpa_distance_m": float(result["cpa_distance_m"]),
        "t_out_s": float(result["trajectory"]["duration_s"][0]),
    }
    return result["audio"], result["quantities"], aux
