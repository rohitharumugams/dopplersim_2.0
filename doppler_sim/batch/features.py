"""Batch feature export (uses Spectrogram Explorer pipeline for spectrograms)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from doppler_sim.batch.constants import FEATURE_SR, PATH_TYPE_STRAIGHT, mps_to_display, output_wav_name
from doppler_sim.batch.planner import BatchConfig, PlannedSample
from doppler_sim.batch.vehicle_metadata import vehicle_display_name
from doppler_sim.specg.explorer import (
    SPECG_DEFAULT_ANALYSIS,
    SPECG_SR,
    export_batch_spectrograms,
    prepare_batch_spec_audio,
)


def _spec_dir(sample_dir: Path) -> Path:
    d = sample_dir / "spectrograms"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _meta_dir(sample_dir: Path) -> Path:
    d = sample_dir / "metadata"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_primary_stft(keys: list[str]) -> list[str]:
    """Phase 1 primary acoustic map A is STFT; always export it."""
    out = list(keys)
    if "stft" not in out:
        out = ["stft", *out]
    return out


def _derive_spectral_features(
    spec: np.ndarray,
    y: np.ndarray,
    sr: int,
    hop_length: int,
    n_fft: int,
) -> dict[str, np.ndarray]:
    T = spec.shape[1]
    dt = hop_length / float(sr)
    dominant_bin = np.argmax(spec, axis=0)
    n_freq = spec.shape[0]
    frequency_hz = (dominant_bin / max(n_freq - 1, 1) * (sr / 2.0)).astype(np.float32)
    frequency = (frequency_hz / (sr / 2.0)).astype(np.float32)

    dfdt_raw = np.zeros(T, dtype=np.float32)
    dfdt_raw[1:] = (frequency_hz[1:] - frequency_hz[:-1]) / max(dt, 1e-9)
    dfdt = (dfdt_raw / (np.max(np.abs(dfdt_raw)) + 1e-8)).astype(np.float32)

    rms_raw = librosa.feature.rms(
        y=y, frame_length=n_fft, hop_length=hop_length, center=True
    )[0]
    if len(rms_raw) > T:
        rms_raw = rms_raw[:T]
    elif len(rms_raw) < T:
        rms_raw = np.pad(rms_raw, (0, T - len(rms_raw)))
    rms = (rms_raw / (np.max(rms_raw) + 1e-8)).astype(np.float32)

    spec_topk = np.zeros((T, 3, 2), dtype=np.float32)
    for t in range(T):
        frame = spec[:, t]
        idx = np.argsort(frame)[-3:][::-1]
        for k, bin_idx in enumerate(idx):
            spec_topk[t, k, 0] = bin_idx / max(n_freq - 1, 1)
            spec_topk[t, k, 1] = frame[bin_idx]

    # Match librosa STFT frame centers (center=True): t_i = i * hop / sr.
    time_arr = (np.arange(T, dtype=np.float64) * hop_length / float(sr)).astype(np.float32)
    return {
        "frequency": frequency,
        "dfdt": dfdt,
        "rms": rms,
        "spec_topk": spec_topk,
        "time": time_arr,
    }


def _kinematics_from_quantities(quantities: dict[str, np.ndarray], sr: int) -> np.ndarray:
    """Propagation diagnostics at observer sample times: [t, v_r, alpha, R].

    These come from retarded-time emitter geometry used in synthesis. They are
    not the Phase 1 centerline state; radial velocity from state is separate.
    """
    t_r = quantities["t_r"]
    v_r = quantities["v_r"]
    alpha = quantities["alpha"]
    r = quantities.get("R")
    mask = np.isfinite(t_r)
    t_obs = np.arange(len(t_r), dtype=np.float32) / float(sr)
    if r is None:
        r_col = np.zeros(int(np.sum(mask)), dtype=np.float32)
    else:
        r_col = np.asarray(r, dtype=np.float32)[mask]
    return np.column_stack(
        [
            t_obs[mask],
            np.asarray(v_r, dtype=np.float32)[mask],
            np.asarray(alpha, dtype=np.float32)[mask],
            r_col,
        ]
    ).astype(np.float32)


def _trajectory_plot(plan: PlannedSample, path: Path, *, speed_unit: str = "mps") -> None:
    v_mps = float(plan.speed_mps)
    v_label = mps_to_display(plan.speed_mps, speed_unit)
    unit = "km/h" if speed_unit == "kmph" else "m/s"
    d = float(plan.cpa_distance_m)
    t_cpa = float(plan.cpa_time_sec)
    t_out = float(plan.t_out_s)

    x_start = v_mps * (0.0 - t_cpa)
    x_end = v_mps * (t_out - t_cpa)
    direction = "left_to_right" if x_end >= x_start else "right_to_left"
    arrow = "→" if direction == "left_to_right" else "←"

    x_pad = max(20.0, abs(x_end - x_start) * 0.2)
    half_span = max(abs(x_start), abs(x_end), d, 50.0) + x_pad

    fig, ax = plt.subplots(figsize=(7, 7.5), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, linestyle=":", alpha=0.6, color="#c4c4c4")

    path_label = (
        f"Path (Straight), pass-by, v={v_label:g} {unit}, d={d:.1f}m, "
        f"t_CPA={t_cpa:.2f}s, dir={direction}{arrow}"
    )
    ax.plot([x_start, x_end], [d, d], color="#2563eb", linewidth=2, label=path_label, zorder=3)
    ax.scatter([0], [0], color="#dc2626", s=45, zorder=5, label="Observer")
    ax.scatter([x_start], [d], color="#16a34a", s=45, zorder=5, label="t=0s")
    ax.scatter([x_end], [d], color="#f97316", s=45, zorder=5, label=f"t={t_out:g}s")
    ax.scatter([0], [d], color="#9333ea", marker="*", s=140, zorder=6, label=f"CPA {t_cpa:.2f}s")
    ax.plot([0, 0], [0, d], linestyle="--", color="#9ca3af", linewidth=1.2, zorder=2)

    ax.set_xlim(-half_span, half_span)
    ax.set_ylim(-half_span, half_span)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=1,
        fontsize=8,
        frameon=True,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _display_speed(plan: PlannedSample, config: BatchConfig) -> float:
    return mps_to_display(plan.speed_mps, config.speed_unit)


def _speed_unit_label(unit: str) -> str:
    return "km/h" if unit == "kmph" else "m/s"


def _write_phase1_metadata(
    meta_out: Path,
    plan: PlannedSample,
    y_wav: np.ndarray,
    y_spec: np.ndarray,
    sr_spec: int,
    analysis,
    stft_tensor: np.ndarray,
    *,
    stft_exported: bool,
) -> dict[str, Any]:
    """Write Phase 1 state, frame alignment, derived quantities, and schema."""
    from doppler_sim.batch.phase1_state import write_phase1_metadata_dir

    return write_phase1_metadata_dir(
        meta_out,
        speed_mps=float(plan.speed_mps),
        cpa_distance_m=float(plan.cpa_distance_m),
        cpa_time_sec=float(plan.cpa_time_sec),
        t_out_s=float(plan.t_out_s),
        y_wav=y_wav,
        wav_sr=FEATURE_SR,
        y_spec=y_spec,
        sr_spec=sr_spec,
        hop_length=int(analysis.stft.hop_length),
        n_fft=int(analysis.stft.n_fft),
        stft_tensor=stft_tensor,
        stft_exported=stft_exported,
    )


def _export_simple_sample_artifacts(
    sample_dir: Path,
    plan: PlannedSample,
    y: np.ndarray,
    config: BatchConfig,
    wav_name: str,
) -> dict[str, Any]:
    """WAV + spectrograms + Phase 1 A/s pairing (legacy scalars kept)."""
    spec_out = _spec_dir(sample_dir)
    meta_out = _meta_dir(sample_dir)
    analysis = SPECG_DEFAULT_ANALYSIS
    fmax_hz = float(config.specg_fmax_hz)
    y_spec, sr_spec = prepare_batch_spec_audio(y, FEATURE_SR)
    speed_display = _display_speed(plan, config)
    speed_label = int(speed_display) if speed_display == int(speed_display) else speed_display
    unit_label = _speed_unit_label(config.speed_unit)
    sample_title = f"{vehicle_display_name(plan.vehicle)} — {speed_label} {unit_label}"

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
    phase1 = _write_phase1_metadata(
        meta_out,
        plan,
        y,
        y_spec,
        sr_spec,
        analysis,
        stft_tensor,
        stft_exported=True,
    )
    derived = phase1["derived_frames"]

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


def export_sample_artifacts(
    sample_dir: Path,
    plan: PlannedSample,
    audio: np.ndarray,
    quantities: dict[str, np.ndarray],
    batch_id: str,
    config: BatchConfig,
) -> dict[str, Any]:
    """Write WAV, selected spectrograms/, and metadata/ for one sample."""
    sample_dir.mkdir(parents=True, exist_ok=True)

    y = np.asarray(audio, dtype=np.float32)

    wav_name = output_wav_name(plan.vehicle, plan.speed_mps, unit=config.speed_unit)
    wav_path = sample_dir / wav_name
    sf.write(wav_path, y, FEATURE_SR, subtype="PCM_16")

    if config.simple_generate:
        return _export_simple_sample_artifacts(sample_dir, plan, y, config, wav_name)

    spec_out = _spec_dir(sample_dir)
    meta_out = _meta_dir(sample_dir)

    analysis = SPECG_DEFAULT_ANALYSIS
    fmax_hz = float(config.specg_fmax_hz)
    y_spec, sr_spec = prepare_batch_spec_audio(y, FEATURE_SR)
    speed_display = _display_speed(plan, config)
    speed_label = int(speed_display) if speed_display == int(speed_display) else speed_display
    unit_label = _speed_unit_label(config.speed_unit)
    sample_title = f"{vehicle_display_name(plan.vehicle)} — {speed_label} {unit_label}"

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
    phase1 = _write_phase1_metadata(
        meta_out,
        plan,
        y,
        y_spec,
        sr_spec,
        analysis,
        stft_tensor,
        stft_exported=True,
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
    _trajectory_plot(plan, sample_dir / "trajectory_plot.png", speed_unit=config.speed_unit)

    from doppler_sim.application import emitter_offsets

    offsets = emitter_offsets(plan.vehicle_length_m, plan.num_emitters)
    np.save(meta_out / "source_positions.npy", offsets.astype(np.float32))

    np.save(meta_out / "speed.npy", np.array([_display_speed(plan, config)], dtype=np.float32))
    np.save(meta_out / "distance.npy", derived["cpa_distance_m"].astype(np.float32))

    speed_key = "speed_kmph" if config.speed_unit == "kmph" else "speed_mps"
    source_speed_key = "source_speed_kmph" if config.speed_unit == "kmph" else "source_speed_mps"
    labels = {
        "vehicle": plan.vehicle,
        "path_type": PATH_TYPE_STRAIGHT,
        "speed_unit": config.speed_unit,
        source_speed_key: mps_to_display(plan.source_speed_mps, config.speed_unit),
        speed_key: _display_speed(plan, config),
        "cpa_distance_m": float(derived["cpa_distance_m"][0]),
        "cpa_distance_plan_m": float(plan.cpa_distance_m),
        "cpa_time_sec": float(derived["cpa_time_sec"][0]),
        "cpa_time_plan_sec": float(plan.cpa_time_sec),
        "direction": int(derived["direction"][0]),
        "phase1_state_columns": ["x_m", "vx_mps", "y_m", "vy_mps"],
    }
    np.save(meta_out / "labels.npy", labels, allow_pickle=True)

    sim_params = {
        "batch_id": batch_id,
        "sample_index": plan.index,
        "path_type": PATH_TYPE_STRAIGHT,
        "speed_unit": config.speed_unit,
        "source_clip": plan.source_path,
        "phase1": {
            "learning_problem": "A(1:T) -> s(1:T)",
            "state_file": "metadata/state_frames.npy",
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
        "render_pass_by": {
            ("v2_kmph" if config.speed_unit == "kmph" else "v2_mps"): _display_speed(plan, config),
            "h2_m": plan.cpa_distance_m,
            "t_cpa2_s": plan.cpa_time_sec,
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
