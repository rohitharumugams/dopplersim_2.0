"""Invoke the existing Pass-By Sim synthesis pipeline without modifying it.

The source signature (STFT → PSD inversion) only depends on the source
recording file and a fixed set of recording-geometry parameters that are
identical for every clip generated from the same VehicleSelection:

    (source_path, v1, h1, t_cpa1, vehicle_length, num_emitters)

It is cheap (~0.2 s) but we still memoise it in-process so that a worker
generating many clips of the same vehicle does not redo it every time.
The dominant per-clip cost is render_pass_by, which is unique per clip.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from doppler_sim.batch.constants import resolve_project_path
from doppler_sim.batch.planner import PlannedSample

# ---------------------------------------------------------------------------
# In-process source-signature cache
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_source_cache: dict[tuple, tuple] = {}


def _sig_key(source_path: Path, plan: PlannedSample) -> tuple:
    """Stable key for source-signature memoisation."""
    return (
        str(source_path.resolve()),
        round(float(plan.source_speed_mps), 8),
        round(float(plan.h1_m), 8),
        round(float(plan.t_cpa1_s), 8),
        round(float(plan.vehicle_length_m), 8),
        int(plan.num_emitters),
    )


def _compute_signature(source_path: Path, plan: PlannedSample) -> tuple:
    """Load source WAV and run estimate_source_signature once."""
    from doppler_sim.application import RenderParams, estimate_source_signature

    audio, sr = librosa.load(str(source_path), sr=None, mono=True)
    if audio.size == 0:
        raise ValueError(f"Empty source clip: {source_path}")

    uploaded = np.asarray(audio, dtype=float)

    # v2 / h2 / t_cpa2 / t_out are NOT used by estimate_source_signature —
    # only v1, h1, t_cpa1, vehicle_length, num_emitters matter here.
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
    freqs, psd_observed, psd_inverted, stft, stft_times = estimate_source_signature(
        uploaded, sr, params
    )
    return (uploaded, int(sr), freqs, psd_observed, psd_inverted, stft, stft_times)


def _get_or_compute_signature(source_path: Path, plan: PlannedSample) -> tuple:
    """Return cached signature, computing it on first call for this key."""
    key = _sig_key(source_path, plan)

    entry = _source_cache.get(key)
    if entry is not None:
        return entry

    new_entry = _compute_signature(source_path, plan)
    with _cache_lock:
        return _source_cache.setdefault(key, new_entry)


def prewarm_source_cache(samples: list[PlannedSample], base_dir: Path) -> None:
    """Pre-compute unique source signatures (used by the single-process path)."""
    seen: set[tuple] = set()
    for sample in samples:
        source_path = resolve_project_path(base_dir, sample.source_path)
        key = _sig_key(source_path, sample)
        if key in seen or key in _source_cache:
            continue
        seen.add(key)
        _get_or_compute_signature(source_path, sample)


def clear_source_cache() -> None:
    """Release cached signature arrays (call after a batch finishes)."""
    with _cache_lock:
        _source_cache.clear()


# ---------------------------------------------------------------------------
# Per-clip synthesis (worker entry point)
# ---------------------------------------------------------------------------

def synthesize_planned_sample(
    plan: PlannedSample,
    *,
    base_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Run render_pass_by using a cached (or freshly computed) source signature."""
    from doppler_sim.application import (
        BASE_DIR,
        OUTPUT_SR,
        RenderParams,
        render_pass_by,
    )

    root = base_dir or BASE_DIR
    source_path = resolve_project_path(root, plan.source_path)

    uploaded, sr, freqs, psd_observed, psd_inverted, stft, stft_times = (
        _get_or_compute_signature(source_path, plan)
    )

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

    generated, quantities, _ = render_pass_by(
        freqs,
        psd_inverted,
        params,
        uploaded,
        sr,
    )

    aux: dict[str, Any] = {
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
