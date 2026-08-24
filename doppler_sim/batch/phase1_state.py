"""Phase 1 physical / acoustic representation for straight pass-bys.

Ground-truth vehicle state at observer time t (microphone at origin):

    s(t) = [x(t), x_dot(t), y(t), y_dot(t)]

Straight, constant-speed kinematics (Pass-By / batch convention):

    x(t) = v * (t - t_cpa)
    y(t) = h
    x_dot(t) = v
    y_dot(t) = 0

Derived from s (not independent labels):

    speed       = sqrt(vx^2 + vy^2)
    range r     = sqrt(x^2 + y^2)
    v_radial    = (x*vx + y*vy) / r
    acceleration = (0, 0) under constant-speed Phase 1
    t_CPA       = argmin_t r(t)
    CPA distance = min_t r(t)
    direction   = sign(vx)   (+1 left→right, -1 right→left)

Learning problem:

    A(1:T) → s(1:T)

where A is the primary time–frequency map (STFT at Spectrogram Explorer SR/hop)
and s is sampled on the same frame-center times.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Column layout for state arrays (documented in phase1_schema.json).
STATE_COLUMNS = ("x_m", "vx_mps", "y_m", "vy_mps")


def straight_passby_state(
    t: np.ndarray,
    *,
    speed_mps: float,
    cpa_distance_m: float,
    cpa_time_sec: float,
) -> np.ndarray:
    """Return s(t) with shape (N, 4): [x, vx, y, vy] in SI units."""
    t = np.asarray(t, dtype=np.float64)
    v = float(speed_mps)
    h = float(cpa_distance_m)
    t_cpa = float(cpa_time_sec)

    x = v * (t - t_cpa)
    y = np.full_like(t, h, dtype=np.float64)
    vx = np.full_like(t, v, dtype=np.float64)
    vy = np.zeros_like(t, dtype=np.float64)
    return np.column_stack([x, vx, y, vy]).astype(np.float64)


def sample_times(n_samples: int, sr: int) -> np.ndarray:
    """Audio-sample times matching render: t[k] = k / sr (float64 for kinematics)."""
    return np.arange(int(n_samples), dtype=np.float64) / float(sr)


def stft_n_frames(n_samples: int, hop_length: int) -> int:
    """Frame count for librosa.stft(..., center=True) default padding."""
    return 1 + int(n_samples) // int(hop_length)


def stft_frame_times(
    n_frames: int,
    *,
    sr: int,
    hop_length: int,
    n_fft: int | None = None,
) -> np.ndarray:
    """Center times of STFT frames (seconds), matching librosa.stft(center=True).

    With centered frames, frame ``i`` is centered on sample ``i * hop_length``,
    so ``t_i = i * hop_length / sr``. ``n_fft`` is accepted for API symmetry
    with callers but does not shift the center times under this convention.
    """
    del n_fft  # documented above; kept so call sites stay explicit
    frames = np.arange(int(n_frames), dtype=np.float64)
    return frames * float(hop_length) / float(sr)


def derived_from_state(state: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    """Derive speed, range, radial velocity, CPA, direction from s(t)."""
    state = np.asarray(state, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError(f"state must have shape (N, 4), got {state.shape}")
    if times.shape[0] != state.shape[0]:
        raise ValueError("times and state length mismatch")

    x = state[:, 0]
    vx = state[:, 1]
    y = state[:, 2]
    vy = state[:, 3]

    speed = np.sqrt(vx * vx + vy * vy)
    # Keep range in float64 for CPA argmin; float32 quantization near CPA
    # can shift the discrete minimum by several samples.
    range_m64 = np.sqrt(x * x + y * y)
    denom = np.maximum(range_m64, 1e-12)
    v_radial = (x * vx + y * vy) / denom
    accel_xy = np.zeros((state.shape[0], 2), dtype=np.float32)

    cpa_idx = int(np.argmin(range_m64))
    cpa_time = float(times[cpa_idx])
    cpa_distance = float(range_m64[cpa_idx])

    speed = speed.astype(np.float32)
    range_m = range_m64.astype(np.float32)
    v_radial = v_radial.astype(np.float32)

    # Constant-speed path: direction from longitudinal velocity.
    v_mean = float(np.mean(vx))
    if abs(v_mean) < 1e-12:
        direction = 0
    else:
        direction = 1 if v_mean > 0.0 else -1

    return {
        "speed_mps": speed,
        "range_m": range_m,
        "radial_velocity_mps": v_radial,
        "acceleration_xy_mps2": accel_xy,
        "cpa_time_sec": np.array([cpa_time], dtype=np.float64),
        "cpa_distance_m": np.array([cpa_distance], dtype=np.float64),
        "cpa_index": np.array([cpa_idx], dtype=np.int32),
        "direction": np.array([direction], dtype=np.int8),
    }


def _frame_rms_centered(
    y: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    n_frames: int,
) -> np.ndarray:
    """RMS per centered frame, matching librosa.feature.rms(center=True) length."""
    y = np.asarray(y, dtype=np.float64)
    pad = int(frame_length // 2)
    y_pad = np.pad(y, (pad, pad), mode="constant")
    rms = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        frame = y_pad[start : start + frame_length]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size))
        rms[i] = np.sqrt(np.mean(frame * frame))
    return rms.astype(np.float32)


def acoustic_state_from_stft(
    stft_db: np.ndarray,
    y: np.ndarray,
    *,
    sr: int,
    hop_length: int,
    n_fft: int,
) -> dict[str, np.ndarray]:
    """Compact per-frame acoustic state A_t ≈ [f_dom, amplitude, centroid].

    This is a secondary representation; the primary A is the full STFT map.
    """
    stft_db = np.asarray(stft_db, dtype=np.float32)
    if stft_db.ndim != 2:
        raise ValueError(f"stft_db must be (F, T), got {stft_db.shape}")

    n_freq, n_frames = stft_db.shape
    freqs = np.fft.rfftfreq(int(n_fft), d=1.0 / float(sr)).astype(np.float64)
    if freqs.shape[0] != n_freq:
        freqs = freqs[:n_freq]
        if freqs.shape[0] < n_freq:
            freqs = np.pad(freqs, (0, n_freq - freqs.shape[0]), mode="edge")

    # Dominant bin frequency (Hz).
    dom_bin = np.argmax(stft_db, axis=0)
    f_dom = freqs[dom_bin].astype(np.float32)

    # Linear-ish amplitude proxy from dB map (unnormalized peak per frame).
    peak_db = np.max(stft_db, axis=0)
    amp = np.power(10.0, peak_db / 20.0).astype(np.float32)
    amp /= np.max(amp) + 1e-8

    # Spectral centroid on a positive power proxy from the dB spectrogram.
    power = np.power(10.0, stft_db / 10.0).astype(np.float64)
    power_sum = np.sum(power, axis=0) + 1e-12
    centroid = (freqs[:, None] * power) / power_sum
    centroid_hz = np.sum(centroid, axis=0).astype(np.float32)

    rms_raw = _frame_rms_centered(
        y,
        frame_length=int(n_fft),
        hop_length=int(hop_length),
        n_frames=n_frames,
    )
    rms = (rms_raw / (np.max(rms_raw) + 1e-8)).astype(np.float32)

    # Columns: dominant_freq_hz, peak_amp_norm, spectral_centroid_hz, rms_norm
    acoustic = np.column_stack([f_dom, amp, centroid_hz, rms]).astype(np.float32)
    return {
        "acoustic_state": acoustic,
        "dominant_frequency_hz": f_dom,
        "peak_amplitude_norm": amp,
        "spectral_centroid_hz": centroid_hz,
        "rms_norm": rms,
    }


def build_phase1_arrays(
    *,
    speed_mps: float,
    cpa_distance_m: float,
    cpa_time_sec: float,
    n_wav_samples: int,
    wav_sr: int,
    n_spec_samples: int,
    spec_sr: int,
    hop_length: int,
    n_fft: int,
    n_frames: int | None = None,
) -> dict[str, Any]:
    """Build sample-rate and STFT-frame-aligned Phase 1 state bundles."""
    t_wav = sample_times(n_wav_samples, wav_sr)
    state_wav = straight_passby_state(
        t_wav,
        speed_mps=speed_mps,
        cpa_distance_m=cpa_distance_m,
        cpa_time_sec=cpa_time_sec,
    )

    if n_frames is None:
        n_frames = stft_n_frames(n_spec_samples, hop_length)
    t_frames = stft_frame_times(
        n_frames,
        sr=spec_sr,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    state_frames = straight_passby_state(
        t_frames,
        speed_mps=speed_mps,
        cpa_distance_m=cpa_distance_m,
        cpa_time_sec=cpa_time_sec,
    )

    derived_wav = derived_from_state(state_wav, t_wav)
    derived_frames = derived_from_state(state_frames, t_frames)

    # Cast to float32 only after float64 derivation (avoids CPA argmin bias).
    state_wav_f32 = state_wav.astype(np.float32)
    state_frames_f32 = state_frames.astype(np.float32)
    t_wav_f32 = t_wav.astype(np.float32)
    t_frames_f32 = t_frames.astype(np.float32)

    # Geometric path companion (backward-compatible trajectory columns).
    trajectory = np.column_stack(
        [t_wav_f32, state_wav_f32[:, 0], state_wav_f32[:, 2]]
    ).astype(np.float32)

    return {
        "state": state_wav_f32,
        "state_times": t_wav_f32,
        "state_frames": state_frames_f32,
        "frame_times": t_frames_f32,
        "trajectory": trajectory,
        "derived_wav": derived_wav,
        "derived_frames": derived_frames,
        "n_frames": int(n_frames),
        "hop_length": int(hop_length),
        "n_fft": int(n_fft),
        "spec_sr": int(spec_sr),
        "wav_sr": int(wav_sr),
    }


def phase1_schema(
    *,
    hop_length: int,
    n_fft: int,
    spec_sr: int,
    wav_sr: int,
    n_frames: int,
    primary_stft_relpath: str = "spectrograms/stft.npy",
    stft_exported: bool,
) -> dict[str, Any]:
    """Machine-readable description of the A(1:T) → s(1:T) pairing."""
    return {
        "phase": 1,
        "learning_problem": "A(1:T) -> s(1:T)",
        "physical_state": {
            "sample_rate_file": "metadata/state.npy",
            "sample_rate_times_file": "metadata/state_times.npy",
            "frame_aligned_file": "metadata/state_frames.npy",
            "frame_times_file": "metadata/frame_times.npy",
            "columns": list(STATE_COLUMNS),
            "units": ["m", "m/s", "m", "m/s"],
            "kinematics": {
                "path_type": "straight",
                "x_m": "v * (t - t_cpa)",
                "y_m": "h (constant CPA lateral distance)",
                "vx_mps": "v (constant)",
                "vy_mps": "0",
                "microphone": [0.0, 0.0],
            },
            "semantics": (
                "s(t) is the vehicle *centerline* at observer time t. "
                "Rendered audio is multi-emitter + retarded-time propagation; "
                "do not treat s as the retarded emitter state that produced each sample. "
                "metadata/kinematics.npy (when present) is retarded-time diagnostics, "
                "not the same object as state-derived radial velocity."
            ),
            "duration_alignment": (
                "state.npy is truncated to the Spectrogram Explorer analysis window "
                "(same duration as spectrograms/stft.npy). Use state_frames.npy for A↔s."
            ),
        },
        "acoustic_representation": {
            "primary_file": primary_stft_relpath,
            "primary_format": "STFT magnitude in dB (librosa amplitude_to_db)",
            "primary_shape": "(F, T)",
            "sr_hz": int(spec_sr),
            "n_fft": int(n_fft),
            "hop_length": int(hop_length),
            "n_frames": int(n_frames),
            "stft_exported": bool(stft_exported),
            "secondary_acoustic_state_file": "metadata/acoustic_state.npy",
            "secondary_columns": [
                "dominant_frequency_hz",
                "peak_amplitude_norm",
                "spectral_centroid_hz",
                "rms_norm",
            ],
            "note": (
                "Pair spectrograms/stft.npy columns with metadata/state_frames.npy "
                "rows via metadata/frame_times.npy (same T)."
            ),
        },
        "derived_from_state": {
            "speed_mps": "sqrt(vx^2 + vy^2)",
            "range_m": "sqrt(x^2 + y^2)",
            "radial_velocity_mps": "(x*vx + y*vy) / range",
            "acceleration_xy_mps2": "[0, 0] (constant speed)",
            "cpa_time_sec": (
                "argmin_t range(t) on the STFT frame grid — primary ML CPA label"
            ),
            "cpa_distance_m": "min_t range(t) on the STFT frame grid",
            "direction": (
                "sign(vx): +1 left-to-right, -1 right-to-left, 0 if |v|~0. "
                "BREAKING: previously always 0 (unused stub)."
            ),
            "files": {
                "speed_series_mps": "metadata/speed_series_mps.npy",
                "range_m": "metadata/range_m.npy",
                "radial_velocity_mps": "metadata/radial_velocity_mps.npy",
                "acceleration_xy_mps2": "metadata/acceleration_xy_mps2.npy",
                "cpa_time_sec": "metadata/cpa_time.npy",
                "cpa_distance_m": "metadata/cpa_distance_m.npy",
                "direction": "metadata/direction.npy",
            },
        },
        "label_contract": {
            "ml_primary_cpa_time": "cpa_time_sec (frame-derived)",
            "plan_cpa_time": "cpa_time_plan_sec / schema.plan.cpa_time_sec",
            "ml_primary_cpa_distance": "cpa_distance_m (frame-derived)",
            "plan_cpa_distance": "cpa_distance_plan_m / schema.plan.cpa_distance_m",
            "note": (
                "Do not mix plan CPA from older dataset.csv rows with "
                "spectrogram-aligned state_frames without checking column names."
            ),
        },
        "wav_sr_hz": int(wav_sr),
    }


def write_phase1_metadata_dir(
    meta_out: Path,
    *,
    speed_mps: float,
    cpa_distance_m: float,
    cpa_time_sec: float,
    t_out_s: float,
    y_wav: np.ndarray,
    wav_sr: int,
    y_spec: np.ndarray,
    sr_spec: int,
    hop_length: int,
    n_fft: int,
    stft_tensor: np.ndarray,
    stft_exported: bool = True,
    primary_stft_relpath: str = "spectrograms/stft.npy",
    extra_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write Phase 1 metadata files (shared by batch and Pass-By single export)."""
    meta_out.mkdir(parents=True, exist_ok=True)

    # Align sample-rate state to spectrogram analysis duration (Explorer may
    # truncate long clips, e.g. SPECG_MAX_DURATION_S).
    y_wav = np.asarray(y_wav, dtype=np.float32)
    y_spec = np.asarray(y_spec, dtype=np.float32)
    spec_duration_s = len(y_spec) / float(sr_spec)
    max_wav = max(int(round(spec_duration_s * float(wav_sr))), 1)
    truncated = False
    if len(y_wav) > max_wav:
        y_wav = y_wav[:max_wav]
        truncated = True

    n_frames = int(stft_tensor.shape[1])
    bundle = build_phase1_arrays(
        speed_mps=float(speed_mps),
        cpa_distance_m=float(cpa_distance_m),
        cpa_time_sec=float(cpa_time_sec),
        n_wav_samples=len(y_wav),
        wav_sr=int(wav_sr),
        n_spec_samples=len(y_spec),
        spec_sr=int(sr_spec),
        hop_length=int(hop_length),
        n_fft=int(n_fft),
        n_frames=n_frames,
    )
    if bundle["state_frames"].shape[0] != n_frames:
        raise RuntimeError(
            f"Phase 1 frame count mismatch: state T={bundle['state_frames'].shape[0]} "
            f"vs STFT T={n_frames}"
        )

    derived = bundle["derived_frames"]
    acoustic = acoustic_state_from_stft(
        stft_tensor,
        y_spec,
        sr=int(sr_spec),
        hop_length=int(hop_length),
        n_fft=int(n_fft),
    )

    np.save(meta_out / "state.npy", bundle["state"])
    np.save(meta_out / "state_times.npy", bundle["state_times"])
    np.save(meta_out / "state_frames.npy", bundle["state_frames"])
    np.save(meta_out / "frame_times.npy", bundle["frame_times"])
    np.save(meta_out / "trajectory.npy", bundle["trajectory"])
    np.save(meta_out / "speed_series_mps.npy", derived["speed_mps"])
    np.save(meta_out / "range_m.npy", derived["range_m"])
    np.save(meta_out / "radial_velocity_mps.npy", derived["radial_velocity_mps"])
    np.save(meta_out / "acceleration_xy_mps2.npy", derived["acceleration_xy_mps2"])
    np.save(meta_out / "cpa_time.npy", derived["cpa_time_sec"])
    np.save(meta_out / "cpa_distance_m.npy", derived["cpa_distance_m"])
    np.save(meta_out / "direction.npy", derived["direction"])
    np.save(meta_out / "acoustic_state.npy", acoustic["acoustic_state"])

    schema = phase1_schema(
        hop_length=int(hop_length),
        n_fft=int(n_fft),
        spec_sr=int(sr_spec),
        wav_sr=int(wav_sr),
        n_frames=n_frames,
        primary_stft_relpath=primary_stft_relpath,
        stft_exported=stft_exported,
    )
    schema["plan"] = {
        "speed_mps": float(speed_mps),
        "cpa_distance_m": float(cpa_distance_m),
        "cpa_time_sec": float(cpa_time_sec),
        "t_out_s": float(t_out_s),
    }
    schema["derived_frame_check"] = {
        "cpa_time_sec": float(derived["cpa_time_sec"][0]),
        "cpa_distance_m": float(derived["cpa_distance_m"][0]),
        "direction": int(derived["direction"][0]),
    }
    schema["cpa_labels"] = {
        "cpa_time_plan_sec": float(cpa_time_sec),
        "cpa_time_sec": float(derived["cpa_time_sec"][0]),
        "cpa_distance_plan_m": float(cpa_distance_m),
        "cpa_distance_m": float(derived["cpa_distance_m"][0]),
        "ml_primary": "frame-derived (cpa_time_sec / cpa_distance_m)",
    }
    schema["state_truncated_to_spec_duration"] = truncated
    if extra_schema:
        schema.update(extra_schema)
    (meta_out / "phase1_schema.json").write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )
    return {
        "bundle": bundle,
        "derived_frames": derived,
        "acoustic": acoustic,
        "schema": schema,
    }


def export_phase1_package(
    root: Path,
    *,
    speed_mps: float,
    cpa_distance_m: float,
    cpa_time_sec: float,
    t_out_s: float,
    audio: np.ndarray,
    wav_sr: int,
    source: str = "pass_by",
) -> dict[str, Any]:
    """Export Phase 1 package under ``root/{spectrograms,metadata}/``.

    Used by Pass-By single-clip export (and can be reused elsewhere). Primary
    acoustic map A is STFT at Spectrogram Explorer SR/hop, matching batch.
    """
    from doppler_sim.specg.explorer import (
        SPECG_DEFAULT_ANALYSIS,
        SPECG_DEFAULT_FMAX_HZ,
        SPECG_TYPE_BY_KEY,
        prepare_batch_spec_audio,
    )

    root = Path(root)
    spec_out = root / "spectrograms"
    meta_out = root / "metadata"
    spec_out.mkdir(parents=True, exist_ok=True)

    y = np.asarray(audio, dtype=np.float32)
    y_spec, sr_spec = prepare_batch_spec_audio(y, int(wav_sr))
    analysis = SPECG_DEFAULT_ANALYSIS
    stft_item = SPECG_TYPE_BY_KEY["stft"]
    stft_tensor = stft_item.build_tensor(y_spec, sr_spec, float(SPECG_DEFAULT_FMAX_HZ), analysis)
    np.save(spec_out / "stft.npy", stft_tensor)

    return write_phase1_metadata_dir(
        meta_out,
        speed_mps=speed_mps,
        cpa_distance_m=cpa_distance_m,
        cpa_time_sec=cpa_time_sec,
        t_out_s=t_out_s,
        y_wav=y,
        wav_sr=int(wav_sr),
        y_spec=y_spec,
        sr_spec=sr_spec,
        hop_length=int(analysis.stft.hop_length),
        n_fft=int(analysis.stft.n_fft),
        stft_tensor=stft_tensor,
        stft_exported=True,
        primary_stft_relpath="spectrograms/stft.npy",
        extra_schema={"source": source},
    )
