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

from doppler_sim.batch.constants import FEATURE_SR, PATH_TYPE_STRAIGHT, output_wav_name
from doppler_sim.batch.planner import BatchConfig, PlannedSample
from doppler_sim.batch.vehicle_metadata import vehicle_display_name
from doppler_sim.specg.explorer import (
    SPECG_DEFAULT_ANALYSIS,
    SPECG_TYPE_BY_KEY,
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


def _derive_spectral_features(
    spec: np.ndarray,
    y: np.ndarray,
    sr: int,
    hop_length: int,
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

    rms_raw = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
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

    time_arr = np.linspace(0, len(y) / float(sr), T, endpoint=False, dtype=np.float32)
    return {
        "frequency": frequency,
        "dfdt": dfdt,
        "rms": rms,
        "spec_topk": spec_topk,
        "time": time_arr,
    }


def _kinematics_from_quantities(quantities: dict[str, np.ndarray], sr: int) -> np.ndarray:
    t_r = quantities["t_r"]
    v_r = quantities["v_r"]
    alpha = quantities["alpha"]
    mask = np.isfinite(t_r)
    t_obs = np.arange(len(t_r), dtype=np.float32) / float(sr)
    return np.column_stack(
        [
            t_obs[mask],
            v_r[mask].astype(np.float32),
            alpha[mask].astype(np.float32),
            np.zeros(np.sum(mask), dtype=np.float32),
        ]
    ).astype(np.float32)


def _trajectory_from_plan(plan: PlannedSample, n: int) -> np.ndarray:
    t = np.linspace(0, plan.t_out_s, n, endpoint=False, dtype=np.float32)
    x = plan.speed_mps * (t - plan.cpa_time_sec)
    y = np.full_like(t, plan.cpa_distance_m)
    return np.column_stack([t, x, y]).astype(np.float32)


def _trajectory_plot(plan: PlannedSample, path: Path) -> None:
    v = float(plan.speed_mps)
    d = float(plan.cpa_distance_m)
    t_cpa = float(plan.cpa_time_sec)
    t_out = float(plan.t_out_s)

    x_start = v * (0.0 - t_cpa)
    x_end = v * (t_out - t_cpa)
    direction = "left_to_right" if x_end >= x_start else "right_to_left"
    arrow = "→" if direction == "left_to_right" else "←"

    x_pad = max(20.0, abs(x_end - x_start) * 0.2)
    half_span = max(abs(x_start), abs(x_end), d, 50.0) + x_pad

    fig, ax = plt.subplots(figsize=(7, 7.5), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, linestyle=":", alpha=0.6, color="#c4c4c4")

    path_label = (
        f"Path (Straight), pass-by, v={v:g}m/s, d={d:.1f}m, "
        f"t_CPA={t_cpa:.2f}s, dir={direction}{arrow}"
    )
    ax.plot(
        [x_start, x_end],
        [d, d],
        color="#2563eb",
        linewidth=2,
        label=path_label,
        zorder=3,
    )
    ax.scatter([0], [0], color="#dc2626", s=45, zorder=5, label="Observer")
    ax.scatter([x_start], [d], color="#16a34a", s=45, zorder=5, label="t=0s")
    ax.scatter([x_end], [d], color="#f97316", s=45, zorder=5, label=f"t={t_out:g}s")
    ax.scatter(
        [0],
        [d],
        color="#9333ea",
        marker="*",
        s=140,
        zorder=6,
        label=f"CPA {t_cpa:.2f}s",
    )
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
    spec_out = _spec_dir(sample_dir)
    meta_out = _meta_dir(sample_dir)

    y = np.asarray(audio, dtype=np.float32)

    wav_name = output_wav_name(plan.vehicle, plan.speed_mps)
    wav_path = sample_dir / wav_name
    sf.write(wav_path, y, FEATURE_SR, subtype="PCM_16")

    analysis = SPECG_DEFAULT_ANALYSIS
    fmax_hz = float(config.specg_fmax_hz)
    y_spec, sr_spec = prepare_batch_spec_audio(y, FEATURE_SR)
    speed_label = int(plan.speed_mps) if plan.speed_mps == int(plan.speed_mps) else plan.speed_mps
    sample_title = f"{vehicle_display_name(plan.vehicle)} — {speed_label} m/s"
    exported = export_batch_spectrograms(
        y_spec,
        sr_spec,
        fmax_hz,
        analysis,
        config.spectrogram_types,
        spec_out,
        meta_out,
        sample_title=sample_title,
    )

    if "stft" in exported:
        stft_item = SPECG_TYPE_BY_KEY["stft"]
        stft_tensor = stft_item.build_tensor(y_spec, sr_spec, fmax_hz, analysis)
        spectral = _derive_spectral_features(
            stft_tensor,
            y_spec,
            sr_spec,
            analysis.stft.hop_length,
        )
        for name, arr in spectral.items():
            np.save(meta_out / f"{name}.npy", arr)

    kin = _kinematics_from_quantities(quantities, FEATURE_SR)
    np.save(meta_out / "kinematics.npy", kin)

    traj = _trajectory_from_plan(plan, len(y))
    np.save(meta_out / "trajectory.npy", traj)
    _trajectory_plot(plan, sample_dir / "trajectory_plot.png")

    from doppler_sim.application import emitter_offsets

    offsets = emitter_offsets(plan.vehicle_length_m, plan.num_emitters)
    np.save(meta_out / "source_positions.npy", offsets.astype(np.float32))

    np.save(meta_out / "speed.npy", np.array([plan.speed_mps], dtype=np.float32))
    np.save(meta_out / "distance.npy", np.array([plan.cpa_distance_m], dtype=np.float32))
    np.save(meta_out / "direction.npy", np.array([0], dtype=np.int32))
    np.save(meta_out / "cpa_time.npy", np.array([plan.cpa_time_sec], dtype=np.float32))
    labels = {
        "vehicle": plan.vehicle,
        "path_type": PATH_TYPE_STRAIGHT,
        "source_speed_mps": plan.source_speed_mps,
        "speed_mps": plan.speed_mps,
        "cpa_distance_m": plan.cpa_distance_m,
        "cpa_time_sec": plan.cpa_time_sec,
    }
    np.save(meta_out / "labels.npy", labels, allow_pickle=True)

    sim_params = {
        "batch_id": batch_id,
        "sample_index": plan.index,
        "path_type": PATH_TYPE_STRAIGHT,
        "source_clip": plan.source_path,
        "original_pass_by": {
            "v1_mps": plan.source_speed_mps,
            "h1_m": plan.h1_m,
            "t_cpa1_s": plan.t_cpa1_s,
        },
        "render_pass_by": {
            "v2_mps": plan.speed_mps,
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
        "combined_spectrogram": "spectrograms/combined.png" if exported else None,
    }
