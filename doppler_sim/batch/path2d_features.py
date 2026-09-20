"""Batch artifact export for 2D whiteboard (free-path Phase 1)."""

from __future__ import annotations

import json
import os
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
from doppler_sim.batch.phase1_state import (
    PATH_TYPE_FREE_2D,
    build_free_path_phase1_arrays,
    free_path_phase1_schema,
    write_free_path_phase1_metadata_dir,
)
from doppler_sim.batch.vehicle_metadata import vehicle_display_name
from doppler_sim.specg.explorer import (
    SPECG_DEFAULT_ANALYSIS,
    SPECG_SR,
    export_batch_spectrograms,
    prepare_batch_spec_audio,
)
import librosa

# Train-mode STFT (MVP): same Δf/Δt as 44.1k/4096/512, half Nyquist, half F bins.
TRAIN_STFT_SR = 22050
TRAIN_STFT_N_FFT = 2048
TRAIN_STFT_HOP = 256
TRAIN_STFT_WINDOW = "hann"

# Shared axis box for train-mode trajectory plots (batch path planning defaults).
_TRAIN_PLOT_X_PAD_M = 5.0
_TRAIN_PLOT_Y_PAD_M = 2.0


def _train_mode_magnitude_stft(y: np.ndarray, *, sr: int) -> np.ndarray:
    """Linear-frequency |STFT| as float32 magnitude (log compression deferred to training)."""
    y = np.asarray(y, dtype=np.float32)
    if int(sr) != TRAIN_STFT_SR:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=TRAIN_STFT_SR)
    S = np.abs(
        librosa.stft(
            y,
            n_fft=TRAIN_STFT_N_FFT,
            hop_length=TRAIN_STFT_HOP,
            window=TRAIN_STFT_WINDOW,
            center=True,
        )
    )
    return np.asarray(S, dtype=np.float32)


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


def _train_mode_axis_limits(config: Path2dBatchConfig) -> tuple[float, float, float, float]:
    """Fixed axis box from batch path ranges so plots are comparable across clips."""
    x_lo = float(min(config.path_x_start_min, config.path_x_start_max)) - _TRAIN_PLOT_X_PAD_M
    x_hi = float(max(config.path_x_end_min, config.path_x_end_max)) + _TRAIN_PLOT_X_PAD_M
    y_lo = max(0.0, float(config.path_y_min) - _TRAIN_PLOT_Y_PAD_M)
    y_hi = float(config.path_y_max) + _TRAIN_PLOT_Y_PAD_M
    return x_lo, x_hi, y_lo, y_hi


def _trajectory_plot_path2d_train(
    plan: Path2dPlannedSample,
    path_xy: np.ndarray,
    out_path: Path,
    *,
    config: Path2dBatchConfig,
) -> None:
    """Compact 320×240 train-mode path plot (~5–15 KB)."""
    mx, my = float(plan.mic_x), float(plan.mic_y)
    xy = np.asarray(path_xy, dtype=np.float64)
    x_lo, x_hi, y_lo, y_hi = _train_mode_axis_limits(config)

    # 320×240 px at 100 dpi → 3.2×2.4 in
    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=100, facecolor="white")
    ax.set_facecolor("white")
    ax.plot(xy[:, 0], xy[:, 1], color="#2563eb", linewidth=1.2)
    ax.scatter([xy[0, 0]], [xy[0, 1]], c="#16a34a", s=18, zorder=5)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], c="#dc2626", s=18, zorder=5)
    ax.scatter([mx], [my], c="#111827", marker="x", s=36, zorder=6)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.grid(True, which="major", color="#e2e8f0", linewidth=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)


def _write_train_mode_schema_once(
    batch_dir: Path,
    *,
    hop_length: int,
    n_fft: int,
    spec_sr: int,
    wav_sr: int,
    n_frames: int,
) -> None:
    """Write batch-level phase1_schema.json once (identical conventions across clips)."""
    schema_path = batch_dir / "phase1_schema.json"
    if schema_path.exists():
        return
    schema = free_path_phase1_schema(
        path_type=PATH_TYPE_FREE_2D,
        dim=2,
        hop_length=int(hop_length),
        n_fft=int(n_fft),
        spec_sr=int(spec_sr),
        wav_sr=int(wav_sr),
        n_frames=int(n_frames),
        primary_stft_relpath="spectrograms/stft.npy",
        stft_exported=True,
    )
    schema["export_mode"] = "train"
    schema["n_frames_note"] = "Per-clip; see each sample metadata/frame_times.npy"
    schema["stft"] = {
        "sr_hz": TRAIN_STFT_SR,
        "n_fft": TRAIN_STFT_N_FFT,
        "hop_length": TRAIN_STFT_HOP,
        "window": TRAIN_STFT_WINDOW,
        "scale": "magnitude_float32",
        "log_compression": "deferred_to_training",
        "freq_axis": "linear",
        "freq_spacing_hz": TRAIN_STFT_SR / TRAIN_STFT_N_FFT,
        "time_spacing_s": TRAIN_STFT_HOP / TRAIN_STFT_SR,
        "file": "spectrograms/stft.npy",
        "wav_sr_hz": FEATURE_SR,
    }
    schema["train_mode_keeps"] = [
        "wav",
        "spectrograms/stft.npy",
        "metadata/state_frames.npy",
        "metadata/frame_times.npy",
        "metadata/range_m.npy",
        "metadata/radial_velocity_mps.npy",
        "metadata/cpa_time.npy",
        "metadata/cpa_distance_m.npy",
        "metadata/polar_state.npy",
        "metadata/canonical_state_frames.npy",
        "metadata/path_xy.npy",
        "metadata/speed.npy",
        "metadata/labels.npy",
        "metadata/simulation_parameters.json",
        "trajectory_plot.png",
        "phase1_schema.json (batch root, once)",
    ]
    schema["free_path_extensions"]["ml_primary_state"] = "metadata/canonical_state_frames.npy"
    schema["free_path_extensions"]["ml_primary_note"] = (
        "Train mode: use canonical_state_frames.npy (x, y columns) as the main target; "
        "polar_state.npy is kept for alternate experiments. Audio-rate state is not exported."
    )
    payload = json.dumps(schema, indent=2)
    try:
        fd = os.open(str(schema_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
    except Exception:
        try:
            schema_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


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


def _export_train_path2d_artifacts(
    sample_dir: Path,
    plan: Path2dPlannedSample,
    y: np.ndarray,
    config: Path2dBatchConfig,
    wav_name: str,
    *,
    trajectory: dict[str, np.ndarray],
    derived_cpa_time: float,
    derived_cpa_distance: float,
    batch_id: str,
) -> dict[str, Any]:
    """MVP train-mode export: WAV + STFT + frame labels + settings (~2–3 MB/clip)."""
    spec_out = _spec_dir(sample_dir)
    meta_out = _meta_dir(sample_dir)

    # WAV stays at FEATURE_SR (44.1 kHz). STFT is computed at TRAIN_STFT_SR.
    y_wav = np.asarray(y, dtype=np.float32)
    stft_tensor = _train_mode_magnitude_stft(y_wav, sr=FEATURE_SR)
    np.save(spec_out / "stft.npy", stft_tensor)

    n_frames = int(stft_tensor.shape[1])
    # Frame labels follow STFT frame centers at the STFT analysis rate (22.05 kHz).
    bundle = build_free_path_phase1_arrays(
        trajectory,
        mic_position=(float(plan.mic_x), float(plan.mic_y)),
        n_wav_samples=len(y_wav),
        wav_sr=FEATURE_SR,
        n_spec_samples=int(round(len(y_wav) * TRAIN_STFT_SR / float(FEATURE_SR))),
        spec_sr=TRAIN_STFT_SR,
        hop_length=TRAIN_STFT_HOP,
        n_fft=TRAIN_STFT_N_FFT,
        n_frames=n_frames,
        dim=2,
    )
    derived = bundle["derived_frames"]

    np.save(meta_out / "state_frames.npy", bundle["state_frames"])
    np.save(meta_out / "frame_times.npy", bundle["frame_times"])
    np.save(meta_out / "range_m.npy", derived["range_m"])
    np.save(meta_out / "radial_velocity_mps.npy", derived["radial_velocity_mps"])
    np.save(meta_out / "cpa_time.npy", derived["cpa_time_sec"])
    np.save(meta_out / "cpa_distance_m.npy", derived["cpa_distance_m"])
    np.save(meta_out / "polar_state.npy", bundle["polar_frames"])
    np.save(meta_out / "canonical_state_frames.npy", bundle["canonical_frames"])
    path_xy = np.asarray(plan.path_xy, dtype=np.float32)
    np.save(meta_out / "path_xy.npy", path_xy)
    np.save(meta_out / "speed.npy", np.array([_display_speed(plan, config)], dtype=np.float32))

    speed_key = "speed_kmph" if config.speed_unit == "kmph" else "speed_mps"
    source_speed_key = "source_speed_kmph" if config.speed_unit == "kmph" else "source_speed_mps"
    labels = {
        "vehicle": plan.vehicle,
        "path_type": PATH_TYPE_FREE_2D,
        "export_mode": "train",
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
        "num_emitters": int(plan.num_emitters),
        "vehicle_length_m": float(plan.vehicle_length_m),
        "phase1_state_columns": ["x_m", "vx_mps", "y_m", "vy_mps"],
        "phase1_canonical_state": "metadata/canonical_state_frames.npy",
        "phase1_polar_state": "metadata/polar_state.npy",
        "derived_cpa_time_sec": float(derived_cpa_time),
        "derived_cpa_distance_m": float(derived_cpa_distance),
        "stft": {
            "sr_hz": TRAIN_STFT_SR,
            "n_fft": TRAIN_STFT_N_FFT,
            "hop_length": TRAIN_STFT_HOP,
            "window": TRAIN_STFT_WINDOW,
            "scale": "magnitude_float32",
            "log_compression": "deferred_to_training",
            "freq_axis": "linear",
            "freq_spacing_hz": TRAIN_STFT_SR / TRAIN_STFT_N_FFT,
            "time_spacing_s": TRAIN_STFT_HOP / TRAIN_STFT_SR,
            "wav_sr_hz": FEATURE_SR,
        },
    }
    np.save(meta_out / "labels.npy", labels, allow_pickle=True)

    sim_params = {
        "batch_id": batch_id,
        "sample_index": plan.index,
        "path_type": PATH_TYPE_FREE_2D,
        "export_mode": "train",
        "speed_unit": config.speed_unit,
        "source_clip": plan.source_path,
        "path_xy": plan.path_xy,
        "mic_xy": [float(plan.mic_x), float(plan.mic_y)],
        "phase1": {
            "learning_problem": "A(1:T) -> s(1:T)",
            "state_file": "metadata/canonical_state_frames.npy",
            "state_frames_file": "metadata/state_frames.npy",
            "polar_state_file": "metadata/polar_state.npy",
            "acoustic_primary": "spectrograms/stft.npy",
            "spec_sr_hz": TRAIN_STFT_SR,
            "hop_length": TRAIN_STFT_HOP,
            "n_fft": TRAIN_STFT_N_FFT,
            "window": TRAIN_STFT_WINDOW,
            "stft_scale": "magnitude_float32",
            "log_compression": "deferred_to_training",
            "freq_axis": "linear",
            "wav_sr_hz": FEATURE_SR,
            "schema_file": "phase1_schema.json",
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

    _trajectory_plot_path2d_train(plan, path_xy, sample_dir / "trajectory_plot.png", config=config)

    batch_dir = sample_dir.parent.parent
    _write_train_mode_schema_once(
        batch_dir,
        hop_length=TRAIN_STFT_HOP,
        n_fft=TRAIN_STFT_N_FFT,
        spec_sr=TRAIN_STFT_SR,
        wav_sr=FEATURE_SR,
        n_frames=n_frames,
    )

    return {
        "wav_name": wav_name,
        "wav_path": wav_name,
        "labels": labels,
        "spectrogram_exports": ["stft"],
        "combined_spectrogram": None,
        "phase1": True,
        "export_mode": "train",
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

    if config.train_mode:
        return _export_train_path2d_artifacts(
            sample_dir,
            plan,
            y,
            config,
            wav_name,
            trajectory=trajectory,
            derived_cpa_time=derived_cpa_time,
            derived_cpa_distance=derived_cpa_distance,
            batch_id=batch_id,
        )

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
