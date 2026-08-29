"""Batch artifact export for 2D whiteboard (free-path Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from doppler_sim.batch.constants import FEATURE_SR, mps_to_display, output_wav_name
from doppler_sim.batch.features import (
    _derive_spectral_features,
    _display_speed,
    _ensure_primary_stft,
    _kinematics_from_quantities,
    _meta_dir,
    _spec_dir,
    _speed_unit_label,
)
from doppler_sim.batch.path2d_planner import Path2dBatchConfig, Path2dPlannedSample
from doppler_sim.batch.phase1_state import PATH_TYPE_FREE_2D, write_free_path_phase1_metadata_dir
from doppler_sim.batch.vehicle_metadata import vehicle_display_name
from doppler_sim.specg.explorer import (
    SPECG_DEFAULT_ANALYSIS,
    SPECG_SR,
    export_batch_spectrograms,
    prepare_batch_spec_audio,
)


def _trajectory_plot_path2d(
    plan: Path2dPlannedSample,
    path_xy: np.ndarray,
    traj: dict[str, np.ndarray],
    out_path: Path,
    *,
    speed_unit: str = "mps",
) -> None:
    mx, my = float(plan.mic_x), float(plan.mic_y)
    v_label = mps_to_display(plan.speed_mps, speed_unit)
    unit = "km/h" if speed_unit == "kmph" else "m/s"

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(path_xy[:, 0], path_xy[:, 1], color="#94a3b8", linewidth=1.5, alpha=0.7, label="Planned path")
    ax.plot(traj["x"], traj["y"], color="#2563eb", linewidth=2.0, label="Timed trajectory")
    ax.scatter([path_xy[0, 0]], [path_xy[0, 1]], c="#16a34a", s=40, zorder=5, label="Start")
    ax.scatter([path_xy[-1, 0]], [path_xy[-1, 1]], c="#dc2626", s=40, zorder=5, label="End")
    ax.scatter([mx], [my], c="#111827", marker="x", s=80, zorder=6, label="Microphone")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, which="major", color="#cbd5e1", linewidth=0.8)
    ax.axhline(0.0, color="#64748b", linewidth=0.8)
    ax.axvline(0.0, color="#64748b", linewidth=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"2D whiteboard path — v={v_label:g} {unit}, "
        f"CPA h={plan.cpa_distance_m:.1f} m, t={plan.cpa_time_sec:.2f} s"
    )
    ax.legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_path2d_phase1_metadata(
    meta_out: Path,
    plan: Path2dPlannedSample,
    *,
    trajectory: dict[str, np.ndarray],
    y_wav: np.ndarray,
    y_spec: np.ndarray,
    sr_spec: int,
    analysis,
    stft_tensor: np.ndarray,
    derived_cpa_time: float,
    derived_cpa_distance: float,
) -> dict[str, Any]:
    return write_free_path_phase1_metadata_dir(
        meta_out,
        trajectory=trajectory,
        mic_position=(float(plan.mic_x), float(plan.mic_y)),
        cpa_time_sec=float(plan.cpa_time_sec),
        cpa_distance_m=float(plan.cpa_distance_m),
        speed_mps=float(plan.speed_mps),
        t_out_s=float(plan.t_out_s),
        y_wav=y_wav,
        wav_sr=FEATURE_SR,
        y_spec=y_spec,
        sr_spec=sr_spec,
        hop_length=int(analysis.stft.hop_length),
        n_fft=int(analysis.stft.n_fft),
        stft_tensor=stft_tensor,
        path_type=PATH_TYPE_FREE_2D,
        dim=2,
        stft_exported=True,
        extra_schema={
            "batch_path_xy": "metadata/path_xy.npy",
            "derived_cpa_time_sec": float(derived_cpa_time),
            "derived_cpa_distance_m": float(derived_cpa_distance),
        },
    )


def _export_simple_path2d_artifacts(
    sample_dir: Path,
    plan: Path2dPlannedSample,
    y: np.ndarray,
    config: Path2dBatchConfig,
    wav_name: str,
    *,
    trajectory: dict[str, np.ndarray],
    derived_cpa_time: float,
    derived_cpa_distance: float,
) -> dict[str, Any]:
    spec_out = _spec_dir(sample_dir)
    meta_out = _meta_dir(sample_dir)
    analysis = SPECG_DEFAULT_ANALYSIS
    fmax_hz = float(config.specg_fmax_hz)
    y_spec, sr_spec = prepare_batch_spec_audio(y, FEATURE_SR)
    speed_display = _display_speed(plan, config)
    speed_label = int(speed_display) if speed_display == int(speed_display) else speed_display
    unit_label = _speed_unit_label(config.speed_unit)
    sample_title = f"{vehicle_display_name(plan.vehicle)} — {speed_label} {unit_label} (2D path)"

    selected = _ensure_primary_stft(list(config.spectrogram_types))
    exported = export_batch_spectrograms(
        y_spec,
        sr_spec,
        fmax_hz,
        analysis,
        selected,
        spec_out,
        sample_title=sample_title,
        png_keys=config.spectrogram_png_types,
        generate_combined=config.generate_combined_png,
    )

    stft_tensor = np.load(spec_out / "stft.npy")
    phase1 = _write_path2d_phase1_metadata(
        meta_out,
        plan,
        trajectory=trajectory,
        y_wav=y,
        y_spec=y_spec,
        sr_spec=sr_spec,
        analysis=analysis,
        stft_tensor=stft_tensor,
        derived_cpa_time=derived_cpa_time,
        derived_cpa_distance=derived_cpa_distance,
    )
    derived = phase1["derived_frames"]
    path_xy = np.asarray(plan.path_xy, dtype=np.float32)
    np.save(meta_out / "path_xy.npy", path_xy)

    np.save(sample_dir / "velocity.npy", np.array([_display_speed(plan, config)], dtype=np.float32))
    np.save(sample_dir / "cpa.npy", derived["cpa_time_sec"].astype(np.float32))

    speed_key = "speed_kmph" if config.speed_unit == "kmph" else "speed_mps"
    return {
        "wav_name": wav_name,
        "wav_path": wav_name,
        "labels": {
            speed_key: _display_speed(plan, config),
            "cpa_time_sec": float(derived["cpa_time_sec"][0]),
            "cpa_time_plan_sec": float(plan.cpa_time_sec),
            "cpa_distance_m": float(derived["cpa_distance_m"][0]),
            "cpa_distance_plan_m": float(plan.cpa_distance_m),
            "direction": int(derived["direction"][0]),
        },
        "spectrogram_exports": exported,
        "combined_spectrogram": (
            "spectrograms/combined.png"
            if config.generate_combined_png
            and exported
            and config.spectrogram_png_types
            and any(k in config.spectrogram_png_types for k in exported)
            else None
        ),
        "phase1": True,
    }


def export_path2d_sample_artifacts(
    sample_dir: Path,
    plan: Path2dPlannedSample,
    audio: np.ndarray,
    quantities: dict[str, np.ndarray],
    batch_id: str,
    config: Path2dBatchConfig,
    *,
    trajectory: dict[str, np.ndarray],
    derived_cpa_time: float,
    derived_cpa_distance: float,
) -> dict[str, Any]:
    """Write WAV, spectrograms/, metadata/ for one 2D whiteboard batch sample."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(audio, dtype=np.float32)
    path_xy = np.asarray(plan.path_xy, dtype=np.float64)

    wav_name = output_wav_name(plan.vehicle, plan.speed_mps, unit=config.speed_unit)
    sf.write(sample_dir / wav_name, y, FEATURE_SR, subtype="PCM_16")

    if config.simple_generate:
        return _export_simple_path2d_artifacts(
            sample_dir,
            plan,
            y,
            config,
            wav_name,
            trajectory=trajectory,
            derived_cpa_time=derived_cpa_time,
            derived_cpa_distance=derived_cpa_distance,
        )

    spec_out = _spec_dir(sample_dir)
    meta_out = _meta_dir(sample_dir)
    analysis = SPECG_DEFAULT_ANALYSIS
    fmax_hz = float(config.specg_fmax_hz)
    y_spec, sr_spec = prepare_batch_spec_audio(y, FEATURE_SR)
    speed_display = _display_speed(plan, config)
    speed_label = int(speed_display) if speed_display == int(speed_display) else speed_display
    unit_label = _speed_unit_label(config.speed_unit)
    sample_title = f"{vehicle_display_name(plan.vehicle)} — {speed_label} {unit_label} (2D path)"

    selected = _ensure_primary_stft(list(config.spectrogram_types))
    exported = export_batch_spectrograms(
        y_spec,
        sr_spec,
        fmax_hz,
        analysis,
        selected,
        spec_out,
        sample_title=sample_title,
        png_keys=config.spectrogram_png_types,
        generate_combined=config.generate_combined_png,
    )

    stft_tensor = np.load(spec_out / "stft.npy")
    phase1 = _write_path2d_phase1_metadata(
        meta_out,
        plan,
        trajectory=trajectory,
        y_wav=y,
        y_spec=y_spec,
        sr_spec=sr_spec,
        analysis=analysis,
        stft_tensor=stft_tensor,
        derived_cpa_time=derived_cpa_time,
        derived_cpa_distance=derived_cpa_distance,
    )
    derived = phase1["derived_frames"]

    spectral = _derive_spectral_features(
        stft_tensor,
        y_spec,
        sr_spec,
        analysis.stft.hop_length,
        analysis.stft.n_fft,
    )
    for name, arr in spectral.items():
        np.save(meta_out / f"{name}.npy", arr)

    kin = _kinematics_from_quantities(quantities, FEATURE_SR)
    np.save(meta_out / "kinematics.npy", kin)
    _trajectory_plot_path2d(
        plan,
        path_xy,
        trajectory,
        sample_dir / "trajectory_plot.png",
        speed_unit=config.speed_unit,
    )
    np.save(meta_out / "path_xy.npy", path_xy.astype(np.float32))

    from doppler_sim.application import emitter_offsets

    offsets = emitter_offsets(plan.vehicle_length_m, plan.num_emitters)
    np.save(meta_out / "source_positions.npy", offsets.astype(np.float32))
    np.save(meta_out / "speed.npy", np.array([_display_speed(plan, config)], dtype=np.float32))
    np.save(meta_out / "distance.npy", derived["cpa_distance_m"].astype(np.float32))

    speed_key = "speed_kmph" if config.speed_unit == "kmph" else "speed_mps"
    source_speed_key = "source_speed_kmph" if config.speed_unit == "kmph" else "source_speed_mps"
    labels = {
        "vehicle": plan.vehicle,
        "path_type": PATH_TYPE_FREE_2D,
        "speed_unit": config.speed_unit,
        source_speed_key: mps_to_display(plan.source_speed_mps, config.speed_unit),
        speed_key: _display_speed(plan, config),
        "cpa_distance_m": float(derived["cpa_distance_m"][0]),
        "cpa_distance_plan_m": float(plan.cpa_distance_m),
        "cpa_time_sec": float(derived["cpa_time_sec"][0]),
        "cpa_time_plan_sec": float(plan.cpa_time_sec),
        "direction": int(derived["direction"][0]),
        "mic_x_m": float(plan.mic_x),
        "mic_y_m": float(plan.mic_y),
        "phase1_state_columns": ["x_m", "vx_mps", "y_m", "vy_mps"],
        "phase1_polar_state": "metadata/polar_state.npy",
    }
    np.save(meta_out / "labels.npy", labels, allow_pickle=True)

    sim_params = {
        "batch_id": batch_id,
        "sample_index": plan.index,
        "path_type": PATH_TYPE_FREE_2D,
        "speed_unit": config.speed_unit,
        "source_clip": plan.source_path,
        "path_xy": plan.path_xy,
        "mic_xy": [float(plan.mic_x), float(plan.mic_y)],
        "phase1": {
            "learning_problem": "A(1:T) -> s(1:T)",
            "state_file": "metadata/state_frames.npy",
            "polar_state_file": "metadata/polar_state.npy",
            "acoustic_primary": "spectrograms/stft.npy",
            "spec_sr_hz": SPECG_SR,
            "hop_length": int(analysis.stft.hop_length),
            "n_fft": int(analysis.stft.n_fft),
        },
        "original_pass_by": {
            ("v1_kmph" if config.speed_unit == "kmph" else "v1_mps"): mps_to_display(
                plan.source_speed_mps, config.speed_unit
            ),
            "h1_m": plan.h1_m,
            "t_cpa1_s": plan.t_cpa1_s,
        },
        "render_path2d": {
            ("v2_kmph" if config.speed_unit == "kmph" else "v2_mps"): _display_speed(plan, config),
            "cpa_distance_plan_m": plan.cpa_distance_m,
            "cpa_time_plan_sec": plan.cpa_time_sec,
            "vehicle_length_m": plan.vehicle_length_m,
            "num_emitters": plan.num_emitters,
            "t_out_s": plan.t_out_s,
        },
    }
    (meta_out / "simulation_parameters.json").write_text(
        json.dumps(sim_params, indent=2),
        encoding="utf-8",
    )

    return {
        "wav_name": wav_name,
        "wav_path": wav_name,
        "labels": labels,
        "spectrogram_exports": exported,
        "combined_spectrogram": (
            "spectrograms/combined.png"
            if config.generate_combined_png
            and exported
            and config.spectrogram_png_types
            and any(k in config.spectrogram_png_types for k in exported)
            else None
        ),
        "phase1": True,
    }
