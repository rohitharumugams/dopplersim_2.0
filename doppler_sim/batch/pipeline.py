"""Invoke the existing Pass-By Sim synthesis pipeline without modifying it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

from doppler_sim.batch.constants import resolve_project_path
from doppler_sim.batch.planner import PlannedSample


def synthesize_planned_sample(
    plan: PlannedSample,
    *,
    base_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Run estimate_source_signature + render_pass_by from the live Pass-By backend."""
    from doppler_sim.application import (
        BASE_DIR,
        OUTPUT_SR,
        RenderParams,
        estimate_source_signature,
        render_pass_by,
    )

    root = base_dir or BASE_DIR
    source_path = resolve_project_path(root, plan.source_path)
    audio, sr = librosa.load(str(source_path), sr=None, mono=True)
    if audio.size == 0:
        raise ValueError(f"Empty source clip: {plan.source_path}")

    params = RenderParams(
        v1=float(plan.source_speed_mps),
        h1=float(plan.h1_m),
        t_cpa1=float(plan.t_cpa1_s),
        vehicle_length=float(plan.vehicle_length_m),
        num_emitters=int(plan.num_emitters),
        v2=float(plan.speed_mps),
        h2=float(plan.cpa_distance_m),
        t_cpa2=float(plan.cpa_time_sec),
        t_out=float(plan.t_out_s),
    )

    uploaded = np.asarray(audio, dtype=float)
    freqs, psd_observed, psd_inverted, stft, stft_times = estimate_source_signature(
        uploaded,
        sr,
        params,
    )
    generated, quantities, _ = render_pass_by(
        freqs,
        psd_inverted,
        params,
        uploaded,
        sr,
    )

    aux = {
        "params": params,
        "uploaded_sr": sr,
        "output_sr": OUTPUT_SR,
        "freqs": freqs,
        "psd_observed": psd_observed,
        "psd_inverted": psd_inverted,
        "stft": stft,
        "stft_times": stft_times,
    }
    return generated, quantities, aux
