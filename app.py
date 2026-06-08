"""Physics-based vehicle pass-by audio re-rendering.

Version 2: PSD is estimated by inverting observed Doppler (f_src = f_obs/α) and
geometric attenuation (×R) using the original pass-by parameters (v₁, h₁, t_CPA₁).
The inverted intrinsic spectrum drives synthesis; propagation re-applies exact
retarded-time physics so Doppler emerges from s_obs(t) = s_src(t_r(t)) / R(t).
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import uuid
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-doppler"))

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
import soundfile as sf
from flask import Flask, render_template, request, send_file, send_from_directory, session, url_for
from scipy.interpolate import interp1d

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PLOTS_DIR = BASE_DIR / "static" / "plots"
GENERATED_DIR = BASE_DIR / "static" / "generated"
COMPARE_DIR = BASE_DIR / "static" / "compare"
RENDERS_DIR = BASE_DIR / "renders"

for directory in (UPLOAD_DIR, PLOTS_DIR, GENERATED_DIR, COMPARE_DIR, RENDERS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

OUTPUT_SR = 44100
SPEED_OF_SOUND = 343.0
KMH_PER_MPS = 3.6
DEFAULT_FREQ_MAX = 10000.0
MAX_FREQ_LIMIT = 22050.0
N_FFT = 4096
HOP_LENGTH = 512

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "doppler-sim-dev-key")


@dataclass
class RenderParams:
    v1: float
    h1: float
    t_cpa1: float
    vehicle_length: float
    num_emitters: int
    v2: float
    h2: float
    t_cpa2: float
    t_out: float


def emitter_offsets(length: float, num_emitters: int) -> np.ndarray:
    if num_emitters <= 1:
        return np.array([0.0])
    return np.linspace(-length / 2.0, length / 2.0, num_emitters)


def compute_stft(audio_mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio_mono = audio_mono / (np.max(np.abs(audio_mono)) + 1e-12)
    stft = librosa.stft(
        audio_mono,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    return stft, freqs, times


def estimate_psd_observed(stft: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(stft) ** 2, axis=1)


def invert_stft_frame_to_source_power(
    stft_frame: np.ndarray,
    freqs: np.ndarray,
    alpha: float,
    distance: float,
) -> np.ndarray:
    """Undo pressure attenuation (×R) and Doppler (f_src = f_obs / α)."""
    magnitude_src = np.abs(stft_frame) * distance
    f_src = freqs / alpha
    return np.interp(freqs, f_src, magnitude_src**2, left=0.0, right=0.0)


def invert_stft_to_source_spectrogram(
    stft: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    v: float,
    h: float,
    t_cpa: float,
    x0: float,
) -> np.ndarray:
    quantities = compute_propagation_quantities(times, v, h, x0, t_cpa)
    valid = np.isfinite(quantities["t_r"]) & (quantities["R"] > 0.0)

    source_spectrogram = np.zeros_like(stft, dtype=float)
    for frame_idx in range(stft.shape[1]):
        if not valid[frame_idx]:
            continue
        source_spectrogram[:, frame_idx] = invert_stft_frame_to_source_power(
            stft[:, frame_idx],
            freqs,
            float(quantities["alpha"][frame_idx]),
            float(quantities["R"][frame_idx]),
        )
    return source_spectrogram


def estimate_psd_inverted(
    stft: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    params: RenderParams,
) -> np.ndarray:
    """Estimate intrinsic source PSD by inverting propagation per emitter, then averaging."""
    offsets = emitter_offsets(params.vehicle_length, params.num_emitters)
    psd_sum = np.zeros(len(freqs), dtype=float)

    for x0 in offsets:
        source_spectrogram = invert_stft_to_source_spectrogram(
            stft,
            freqs,
            times,
            params.v1,
            params.h1,
            params.t_cpa1,
            float(x0),
        )
        valid_frames = np.any(source_spectrogram > 0.0, axis=0)
        if np.any(valid_frames):
            psd_sum += np.mean(source_spectrogram[:, valid_frames], axis=1)

    if len(offsets) == 0:
        return psd_sum

    psd = psd_sum / len(offsets)
    return np.maximum(psd, 0.0)


def estimate_source_signature(
    audio_mono: np.ndarray,
    sr: int,
    params: RenderParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stft, freqs, times = compute_stft(audio_mono, sr)
    psd_observed = estimate_psd_observed(stft)
    psd_inverted = estimate_psd_inverted(stft, freqs, times, params)
    return freqs, psd_observed, psd_inverted, stft, times


def synthesize_psd_noise(
    freqs: np.ndarray,
    psd: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.standard_normal(n_samples)
    noise_fft = np.fft.rfft(noise)
    fft_freqs = np.fft.rfftfreq(n_samples, d=1.0 / OUTPUT_SR)
    psd_interp = np.exp(
        np.interp(
            np.log(fft_freqs + 1.0),
            np.log(freqs + 1.0),
            np.log(np.maximum(psd, 1e-12)),
            left=np.log(1e-12),
            right=np.log(1e-12),
        )
    )
    shaped = noise_fft * np.sqrt(np.maximum(psd_interp, 0.0))
    source = np.fft.irfft(shaped, n=n_samples)
    source /= np.max(np.abs(source)) + 1e-12
    return source


def _is_geometrically_consistent_retarded_root(
    root: float,
    t_obs: float,
    v: float,
    x0: float,
    h: float,
    c: float,
    rel_tol: float = 1e-4,
) -> bool:
    if not np.isfinite(root) or root >= t_obs:
        return False
    prop_r = c * (t_obs - root)
    if prop_r <= 0.0:
        return False
    x_at_tr = v * root + x0
    geom_r = np.sqrt(x_at_tr**2 + h**2)
    return abs(geom_r - prop_r) / (geom_r + 1e-9) < rel_tol


def _select_retarded_root(
    roots: np.ndarray,
    t_obs: float,
    v: float,
    x0: float,
    h: float,
    c: float,
) -> float:
    for root in sorted(roots, reverse=True):
        if _is_geometrically_consistent_retarded_root(root, t_obs, v, x0, h, c):
            return float(root)
    return np.nan


def solve_retarded_time(
    t_obs: np.ndarray,
    v: float,
    x0: float,
    h: float,
    c: float = SPEED_OF_SOUND,
) -> np.ndarray:
    a = v**2 - c**2
    b = 2.0 * (v * x0 + c**2 * t_obs)
    cc = x0**2 + h**2 - c**2 * t_obs**2

    t_r = np.full_like(t_obs, np.nan, dtype=float)

    if np.allclose(a, 0.0):
        for idx in range(len(t_obs)):
            if abs(b[idx]) < 1e-12:
                continue
            root = -cc[idx] / b[idx]
            chosen = _select_retarded_root(np.array([root]), float(t_obs[idx]), v, x0, h, c)
            if np.isfinite(chosen):
                t_r[idx] = chosen
        return t_r

    disc = b**2 - 4.0 * a * cc
    for idx in range(len(t_obs)):
        if disc[idx] < 0.0:
            continue
        sqrt_disc = np.sqrt(disc[idx])
        root1 = (-b[idx] + sqrt_disc) / (2.0 * a)
        root2 = (-b[idx] - sqrt_disc) / (2.0 * a)
        chosen = _select_retarded_root(
            np.array([root1, root2]),
            float(t_obs[idx]),
            v,
            x0,
            h,
            c,
        )
        if np.isfinite(chosen):
            t_r[idx] = chosen

    return t_r


def vehicle_center_position(t: np.ndarray, v: float, t_cpa: float) -> np.ndarray:
    return v * (t - t_cpa)


def compute_propagation_quantities(
    t_obs: np.ndarray,
    v: float,
    h: float,
    x0: float,
    t_cpa: float,
) -> dict[str, np.ndarray]:
    # x(t_r) = v(t_r - t_cpa) + x0  =>  v*t_r + (x0 - v*t_cpa) in the quadratic form
    x0_quadratic = x0 - v * t_cpa
    t_r = solve_retarded_time(t_obs, v, x0_quadratic, h)
    r = SPEED_OF_SOUND * (t_obs - t_r)
    x_emitter = vehicle_center_position(t_r, v, t_cpa) + x0
    v_r = v * x_emitter / np.maximum(r, 1e-9)
    alpha = SPEED_OF_SOUND / (SPEED_OF_SOUND - v_r)
    tau = r / SPEED_OF_SOUND
    return {
        "t_r": t_r,
        "R": r,
        "v_r": v_r,
        "alpha": alpha,
        "tau": tau,
        "x_emitter": x_emitter,
    }


def extract_amplitude_envelope(
    audio: np.ndarray,
    sr: int,
    smooth_ms: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (observer times, peak-normalized RMS envelope) from the recording."""
    frame_len = max(int(sr * smooth_ms / 1000.0), 1)
    hop = max(frame_len // 4, 1)
    frames = librosa.util.frame(np.abs(audio), frame_length=frame_len, hop_length=hop)
    rms = np.sqrt(np.mean(frames**2, axis=0))
    t_env = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    rms /= np.max(rms) + 1e-12
    return t_env, rms


def render_emitter(
    freqs: np.ndarray,
    psd: np.ndarray,
    v: float,
    h: float,
    t_cpa: float,
    x0: float,
    t_out: float,
    rng: np.random.Generator,
    v1: float,
    h1: float,
    t_cpa1: float,
    t_env1: np.ndarray,
    env1: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_obs = int(np.ceil(t_out * OUTPUT_SR))
    t_obs = np.arange(n_obs, dtype=float) / OUTPUT_SR

    quantities = compute_propagation_quantities(t_obs, v, h, x0, t_cpa)
    t_r = quantities["t_r"]
    r = quantities["R"]

    valid = np.isfinite(t_r) & (r > 0.0)
    if not np.any(valid):
        return np.zeros(n_obs), quantities

    min_source_time = float(np.nanmin(t_r[valid]))
    max_source_time = float(np.nanmax(t_r[valid]))
    source_start = min_source_time - 3.0
    source_end = max_source_time + 3.0
    source_len = max(int(np.ceil((source_end - source_start) * OUTPUT_SR)), 1)
    source = synthesize_psd_noise(freqs, psd, source_len, rng)
    source_times = source_start + np.arange(len(source), dtype=float) / OUTPUT_SR

    interpolator = interp1d(
        source_times,
        source,
        bounds_error=False,
        fill_value=0.0,
    )

    observed = np.zeros(n_obs, dtype=float)
    observed[valid] = interpolator(t_r[valid])

    t_obs1 = t_env1
    q1 = compute_propagation_quantities(t_obs1, v1, h1, x0, t_cpa1)
    t_r1 = q1["t_r"]
    valid1 = np.isfinite(t_r1)

    if np.any(valid1):
        sort_idx = np.argsort(t_r1[valid1])
        t_r1_sorted = t_r1[valid1][sort_idx]
        t_obs1_sorted = t_obs1[valid1][sort_idx]
        src_to_obs1 = interp1d(
            t_r1_sorted,
            t_obs1_sorted,
            bounds_error=False,
            fill_value=np.nan,
        )
        t_obs1_equiv = src_to_obs1(t_r[valid])

        env_interp = interp1d(
            t_env1,
            env1,
            bounds_error=False,
            fill_value=0.0,
        )
        amplitude = env_interp(t_obs1_equiv)
        amplitude = np.nan_to_num(amplitude, nan=0.0)
        observed[valid] *= amplitude

    return observed, quantities


def render_pass_by(
    freqs: np.ndarray,
    psd: np.ndarray,
    params: RenderParams,
    uploaded_audio: np.ndarray,
    uploaded_sr: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[np.ndarray]]:
    offsets = emitter_offsets(params.vehicle_length, params.num_emitters)
    rng = np.random.default_rng()
    n_obs = int(np.ceil(params.t_out * OUTPUT_SR))
    output = np.zeros(n_obs, dtype=float)
    weight = 1.0 / np.sqrt(params.num_emitters)

    reference_quantities: dict[str, np.ndarray] | None = None
    emitter_signals: list[np.ndarray] = []
    t_env1, env1 = extract_amplitude_envelope(uploaded_audio, uploaded_sr)

    for x0 in offsets:
        emitter_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        contribution, quantities = render_emitter(
            freqs=freqs,
            psd=psd,
            v=params.v2,
            h=params.h2,
            t_cpa=params.t_cpa2,
            x0=float(x0),
            t_out=params.t_out,
            rng=emitter_rng,
            v1=params.v1,
            h1=params.h1,
            t_cpa1=params.t_cpa1,
            t_env1=t_env1,
            env1=env1,
        )
        emitter_signals.append(contribution)
        output += weight * contribution
        if reference_quantities is None:
            reference_quantities = quantities

    if reference_quantities is None:
        t_obs = np.arange(n_obs, dtype=float) / OUTPUT_SR
        reference_quantities = compute_propagation_quantities(
            t_obs,
            params.v2,
            params.h2,
            0.0,
            params.t_cpa2,
        )

    return output, reference_quantities, emitter_signals


def save_plot(filename: str, plot_dir: Path) -> str:
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return filename


def plot_waveform(
    audio: np.ndarray,
    sr: int,
    title: str,
    filename: str,
    plot_dir: Path,
    color: str = "#2563eb",
) -> str:
    plt.figure(figsize=(10, 3))
    times = np.arange(len(audio)) / sr
    plt.plot(times, audio, color=color, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    return save_plot(filename, plot_dir)


def plot_spectrogram(
    audio: np.ndarray,
    sr: int,
    title: str,
    filename: str,
    plot_dir: Path,
    freq_max: float = DEFAULT_FREQ_MAX,
) -> str:
    plt.figure(figsize=(10, 4))
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann")
    magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    librosa.display.specshow(
        magnitude_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.ylim(0, freq_max)
    return save_plot(filename, plot_dir)


def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    filename: str,
    title: str,
    plot_dir: Path,
    color: str = "#059669",
) -> str:
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, np.maximum(psd, 1e-12), color=color)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.grid(True, which="both", alpha=0.3)
    return save_plot(filename, plot_dir)


def plot_psd_comparison(
    freqs: np.ndarray,
    psd_observed: np.ndarray,
    psd_inverted: np.ndarray,
    filename: str,
    plot_dir: Path,
) -> str:
    plt.figure(figsize=(10, 4))
    plt.semilogy(
        freqs,
        np.maximum(psd_observed, 1e-12),
        color="#64748b",
        label="Observed (raw recording)",
        alpha=0.9,
    )
    plt.semilogy(
        freqs,
        np.maximum(psd_inverted, 1e-12),
        color="#059669",
        label="Inverted (intrinsic, used for synthesis)",
        alpha=0.9,
    )
    plt.title("Observed vs Inverted Source PSD")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.legend(loc="best")
    plt.grid(True, which="both", alpha=0.3)
    return save_plot(filename, plot_dir)


def plot_inverted_spectrogram(
    stft: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    params: RenderParams,
    filename: str,
    plot_dir: Path,
    freq_max: float = DEFAULT_FREQ_MAX,
) -> str:
    offsets = emitter_offsets(params.vehicle_length, params.num_emitters)
    spectrogram_sum = np.zeros_like(stft, dtype=float)
    for x0 in offsets:
        spectrogram_sum += invert_stft_to_source_spectrogram(
            stft,
            freqs,
            times,
            params.v1,
            params.h1,
            params.t_cpa1,
            float(x0),
        )
    spectrogram = spectrogram_sum / max(len(offsets), 1)
    spectrogram_db = 10.0 * np.log10(np.maximum(spectrogram, 1e-12))

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(
        times,
        freqs,
        spectrogram_db,
        shading="gouraud",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Inverted Intrinsic Spectrogram (Doppler & attenuation removed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, min(freq_max, float(freqs[-1])))
    return save_plot(filename, plot_dir)


def plot_vehicle_geometry(
    length: float,
    num_emitters: int,
    filename: str,
    plot_dir: Path,
) -> str:
    offsets = emitter_offsets(length, num_emitters)
    plt.figure(figsize=(10, 3))
    y = np.zeros_like(offsets)
    plt.scatter(offsets, y, s=120, color="#dc2626", zorder=3)
    plt.plot([-length / 2, length / 2], [0, 0], color="#64748b", linewidth=3)
    for x0 in offsets:
        plt.axvline(x0, color="#fca5a5", linestyle="--", alpha=0.5)
    plt.title("Vehicle Emitter Geometry")
    plt.xlabel("Longitudinal position (m)")
    plt.yticks([])
    plt.grid(True, axis="x", alpha=0.3)
    return save_plot(filename, plot_dir)


def plot_observer_geometry(
    params: RenderParams,
    filename: str,
    plot_dir: Path,
) -> str:
    t = np.linspace(0.0, params.t_out, 500)
    x_vehicle = vehicle_center_position(t, params.v2, params.t_cpa2)
    plt.figure(figsize=(8, 6))
    plt.axhline(params.h2, color="#94a3b8", linestyle="--", label=f"Trajectory y = {params.h2:.2f} m")
    plt.plot(x_vehicle, np.full_like(x_vehicle, params.h2), color="#2563eb", linewidth=2, label="Vehicle path")
    plt.scatter([0.0], [0.0], s=160, color="#16a34a", zorder=4, label="Observer")
    plt.scatter([0.0], [params.h2], s=120, color="#dc2626", zorder=4, label="CPA")
    plt.title("Observer and Vehicle Geometry")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    return save_plot(filename, plot_dir)


def plot_series(
    t: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    filename: str,
    plot_dir: Path,
    color: str = "#7c3aed",
) -> str:
    plt.figure(figsize=(10, 3))
    mask = np.isfinite(values)
    plt.plot(t[mask], values[mask], color=color, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Observer time (s)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    return save_plot(filename, plot_dir)


PLOT_EXPORT_NAMES = {
    "uploaded_waveform": "01_uploaded_waveform.png",
    "uploaded_spectrogram": "02_uploaded_spectrogram.png",
    "psd_comparison": "03_psd_comparison.png",
    "observed_psd": "04_observed_psd.png",
    "intrinsic_psd": "05_intrinsic_psd.png",
    "inverted_spectrogram": "06_inverted_spectrogram.png",
    "vehicle_geometry": "07_vehicle_geometry.png",
    "observer_geometry": "08_observer_geometry.png",
    "retarded_time": "09_retarded_time.png",
    "radial_velocity": "10_radial_velocity.png",
    "doppler_factor": "11_doppler_factor.png",
    "propagation_distance": "12_propagation_distance.png",
    "generated_waveform": "13_generated_waveform.png",
    "generated_spectrogram": "14_generated_spectrogram.png",
}


def generate_all_plots(
    uploaded_audio: np.ndarray,
    uploaded_sr: int,
    generated_audio: np.ndarray,
    freqs: np.ndarray,
    psd_observed: np.ndarray,
    psd_inverted: np.ndarray,
    stft: np.ndarray,
    stft_times: np.ndarray,
    params: RenderParams,
    quantities: dict[str, np.ndarray],
    plot_dir: Path,
    freq_max: float = DEFAULT_FREQ_MAX,
) -> dict[str, str]:
    t_obs = np.arange(len(generated_audio), dtype=float) / OUTPUT_SR
    plots = {
        "uploaded_waveform": plot_waveform(
            uploaded_audio,
            uploaded_sr,
            "Uploaded Waveform",
            PLOT_EXPORT_NAMES["uploaded_waveform"],
            plot_dir,
            color="#475569",
        ),
        "uploaded_spectrogram": plot_spectrogram(
            uploaded_audio,
            uploaded_sr,
            "Uploaded Spectrogram",
            PLOT_EXPORT_NAMES["uploaded_spectrogram"],
            plot_dir,
            freq_max=freq_max,
        ),
        "observed_psd": plot_psd(
            freqs,
            psd_observed,
            PLOT_EXPORT_NAMES["observed_psd"],
            "Observed PSD (raw recording average)",
            plot_dir,
            color="#64748b",
        ),
        "intrinsic_psd": plot_psd(
            freqs,
            psd_inverted,
            PLOT_EXPORT_NAMES["intrinsic_psd"],
            "Inverted Intrinsic PSD (de-Dopplered, used for synthesis)",
            plot_dir,
            color="#059669",
        ),
        "psd_comparison": plot_psd_comparison(
            freqs,
            psd_observed,
            psd_inverted,
            PLOT_EXPORT_NAMES["psd_comparison"],
            plot_dir,
        ),
        "inverted_spectrogram": plot_inverted_spectrogram(
            stft,
            freqs,
            stft_times,
            params,
            PLOT_EXPORT_NAMES["inverted_spectrogram"],
            plot_dir,
            freq_max=freq_max,
        ),
        "vehicle_geometry": plot_vehicle_geometry(
            params.vehicle_length,
            params.num_emitters,
            PLOT_EXPORT_NAMES["vehicle_geometry"],
            plot_dir,
        ),
        "observer_geometry": plot_observer_geometry(
            params,
            PLOT_EXPORT_NAMES["observer_geometry"],
            plot_dir,
        ),
        "retarded_time": plot_series(
            t_obs,
            quantities["t_r"],
            "Retarded Time vs Observer Time",
            "Retarded time (s)",
            PLOT_EXPORT_NAMES["retarded_time"],
            plot_dir,
        ),
        "radial_velocity": plot_series(
            t_obs,
            quantities["v_r"],
            "Radial Velocity vs Observer Time",
            "Radial velocity (m/s)",
            PLOT_EXPORT_NAMES["radial_velocity"],
            plot_dir,
            color="#ea580c",
        ),
        "doppler_factor": plot_series(
            t_obs,
            quantities["alpha"],
            "Doppler Factor vs Observer Time",
            "α = c / (c − v_r)",
            PLOT_EXPORT_NAMES["doppler_factor"],
            plot_dir,
            color="#0891b2",
        ),
        "propagation_distance": plot_series(
            t_obs,
            quantities["R"],
            "Propagation Distance vs Observer Time",
            "Distance R (m)",
            PLOT_EXPORT_NAMES["propagation_distance"],
            plot_dir,
            color="#9333ea",
        ),
        "generated_waveform": plot_waveform(
            generated_audio,
            OUTPUT_SR,
            "Generated Waveform",
            PLOT_EXPORT_NAMES["generated_waveform"],
            plot_dir,
            color="#2563eb",
        ),
        "generated_spectrogram": plot_spectrogram(
            generated_audio,
            OUTPUT_SR,
            "Generated Spectrogram",
            PLOT_EXPORT_NAMES["generated_spectrogram"],
            plot_dir,
            freq_max=freq_max,
        ),
    }
    return plots


def sanitize_bundle_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-. ]", "", name.strip())
    cleaned = cleaned.replace(" ", "_").strip("._")
    return cleaned[:80] or "doppler_sim_export"


def parse_freq_max(default: float | None = None) -> float:
    fallback = DEFAULT_FREQ_MAX if default is None else default
    value = parse_float("freq_max", fallback)
    return float(np.clip(value, 500.0, MAX_FREQ_LIMIT))


def save_render_state(
    render_id: str,
    *,
    output_name: str,
    upload_filename: str | None,
    speed_unit: str,
    freq_max: float,
    params: RenderParams,
    plots: dict[str, str],
    uploaded_audio: np.ndarray,
    uploaded_sr: int,
    generated_audio: np.ndarray,
    freqs: np.ndarray,
    psd_observed: np.ndarray,
    psd_inverted: np.ndarray,
    stft: np.ndarray,
    stft_times: np.ndarray,
    quantities: dict[str, np.ndarray],
) -> None:
    render_dir = RENDERS_DIR / render_id
    render_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        render_dir / "data.npz",
        uploaded_audio=uploaded_audio,
        uploaded_sr=np.array([uploaded_sr]),
        generated_audio=generated_audio,
        freqs=freqs,
        psd_observed=psd_observed,
        psd_inverted=psd_inverted,
        stft=stft,
        stft_times=stft_times,
        t_r=quantities["t_r"],
        v_r=quantities["v_r"],
        alpha=quantities["alpha"],
        R=quantities["R"],
    )

    meta = {
        "output_name": output_name,
        "upload_filename": upload_filename,
        "speed_unit": speed_unit,
        "freq_max": freq_max,
        "params": asdict(params),
        "plots": plots,
    }
    (render_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    session["last_render_id"] = render_id


def load_render_state(render_id: str) -> tuple[dict, dict[str, np.ndarray]]:
    render_dir = RENDERS_DIR / render_id
    meta = json.loads((render_dir / "meta.json").read_text(encoding="utf-8"))
    data = np.load(render_dir / "data.npz")
    arrays = {key: data[key] for key in data.files}
    arrays["uploaded_sr"] = int(arrays["uploaded_sr"][0])
    arrays["quantities"] = {
        "t_r": arrays.pop("t_r"),
        "v_r": arrays.pop("v_r"),
        "alpha": arrays.pop("alpha"),
        "R": arrays.pop("R"),
    }
    return meta, arrays


def plot_urls_for_render(render_id: str, plots: dict[str, str], freq_max: float) -> dict[str, str]:
    cache_key = int(freq_max)
    return {
        key: f"{url_for('plot_file', render_id=render_id, filename=filename)}?v={cache_key}"
        for key, filename in plots.items()
    }


def build_success_context(
    render_id: str,
    meta: dict,
    plots: dict[str, str],
    params: RenderParams,
    speed_unit: str,
    upload_filename: str | None,
) -> dict:
    output_name = meta["output_name"]
    freq_max = float(meta["freq_max"])
    return {
        "success": True,
        "render_id": render_id,
        "audio_url": url_for("generated_file", filename=output_name),
        "freq_max": freq_max,
        "plots": plot_urls_for_render(render_id, plots, freq_max),
        **form_context(
            params,
            speed_unit,
            upload_filename=upload_filename,
            freq_max=freq_max,
        ),
    }


def regenerate_plots_from_state(render_id: str, freq_max: float) -> tuple[dict, RenderParams, dict[str, str]]:
    meta, arrays = load_render_state(render_id)
    params = RenderParams(**meta["params"])
    plot_dir = PLOTS_DIR / render_id

    plots = generate_all_plots(
        arrays["uploaded_audio"],
        arrays["uploaded_sr"],
        arrays["generated_audio"],
        arrays["freqs"],
        arrays["psd_observed"],
        arrays["psd_inverted"],
        arrays["stft"],
        arrays["stft_times"],
        params,
        arrays["quantities"],
        plot_dir,
        freq_max=freq_max,
    )

    meta["freq_max"] = freq_max
    meta["plots"] = plots
    (RENDERS_DIR / render_id / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta, params, plots


def parse_float(name: str, default: float) -> float:
    value = request.form.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(name: str, default: int) -> int:
    value = request.form.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_speed_unit(default: str = "mps") -> str:
    unit = request.form.get("speed_unit", default)
    return "kmph" if unit == "kmph" else "mps"


def to_mps(speed: float, unit: str) -> float:
    return speed / KMH_PER_MPS if unit == "kmph" else speed


def from_mps(speed_mps: float, unit: str) -> float:
    return speed_mps * KMH_PER_MPS if unit == "kmph" else speed_mps


def parse_params() -> tuple[RenderParams, str]:
    speed_unit = parse_speed_unit()
    default_v1 = 72.0 if speed_unit == "kmph" else 20.0
    default_v2 = 90.0 if speed_unit == "kmph" else 25.0
    return RenderParams(
        v1=to_mps(parse_float("v1", default_v1), speed_unit),
        h1=parse_float("h1", 10.0),
        t_cpa1=parse_float("t_cpa1", 2.0),
        vehicle_length=parse_float("vehicle_length", 4.5),
        num_emitters=max(1, parse_int("num_emitters", 3)),
        v2=to_mps(parse_float("v2", default_v2), speed_unit),
        h2=parse_float("h2", 8.0),
        t_cpa2=parse_float("t_cpa2", 2.5),
        t_out=max(0.5, parse_float("t_out", 6.0)),
    ), speed_unit


def form_context(
    params: RenderParams | None = None,
    speed_unit: str = "mps",
    freq_max: float = DEFAULT_FREQ_MAX,
    **extra,
) -> dict:
    ctx = {
        **upload_context(),
        "speed_unit": speed_unit,
        "freq_max": freq_max,
        "v1_display": from_mps(params.v1, speed_unit) if params else (72.0 if speed_unit == "kmph" else 20.0),
        "v2_display": from_mps(params.v2, speed_unit) if params else (90.0 if speed_unit == "kmph" else 25.0),
    }
    if params is not None:
        ctx["params"] = params
    ctx.update(extra)
    return ctx


def upload_context() -> dict:
    upload_id = session.get("upload_id")
    upload_filename = session.get("upload_filename")
    has_upload = bool(upload_id and (UPLOAD_DIR / f"{upload_id}.wav").exists())
    return {
        "has_upload": has_upload,
        "upload_filename": upload_filename if has_upload else None,
    }


def resolve_upload_path() -> tuple[Path | None, str | None, str | None]:
    uploaded = request.files.get("audio_file")
    if uploaded is not None and uploaded.filename:
        upload_id = uuid.uuid4().hex
        upload_path = UPLOAD_DIR / f"{upload_id}.wav"
        uploaded.save(upload_path)
        session["upload_id"] = upload_id
        session["upload_filename"] = uploaded.filename
        return upload_path, uploaded.filename, None

    upload_id = session.get("upload_id")
    if not upload_id:
        return None, None, "Please upload a WAV file."

    upload_path = UPLOAD_DIR / f"{upload_id}.wav"
    if not upload_path.exists():
        session.pop("upload_id", None)
        session.pop("upload_filename", None)
        return None, None, "Your previous upload expired. Please upload a WAV file again."

    return upload_path, session.get("upload_filename"), None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **form_context())


@app.route("/generate", methods=["POST"])
def generate():
    params, speed_unit = parse_params()
    freq_max = parse_freq_max()
    upload_path, upload_filename, upload_error = resolve_upload_path()
    if upload_error:
        return render_template(
            "index.html",
            error=upload_error,
            **form_context(params, speed_unit, freq_max=freq_max),
        )

    try:
        audio, sr = librosa.load(upload_path, sr=None, mono=True)
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Failed to load audio: {exc}",
            **form_context(params, speed_unit, freq_max=freq_max),
        )

    if audio.size == 0:
        return render_template(
            "index.html",
            error="Uploaded file is empty.",
            **form_context(params, speed_unit, freq_max=freq_max),
        )

    uploaded_plot_copy = audio.copy()
    uploaded_sr = sr

    freqs, psd_observed, psd_inverted, stft, stft_times = estimate_source_signature(
        audio,
        sr,
        params,
    )
    del audio

    generated, quantities, _ = render_pass_by(
        freqs,
        psd_inverted,
        params,
        uploaded_plot_copy,
        uploaded_sr,
    )

    output_name = f"{uuid.uuid4().hex}.wav"
    output_path = GENERATED_DIR / output_name
    sf.write(output_path, generated, OUTPUT_SR, subtype="PCM_16")

    render_id = uuid.uuid4().hex
    plot_dir = PLOTS_DIR / render_id
    plots = generate_all_plots(
        uploaded_plot_copy,
        uploaded_sr,
        generated,
        freqs,
        psd_observed,
        psd_inverted,
        stft,
        stft_times,
        params,
        quantities,
        plot_dir,
        freq_max=freq_max,
    )

    save_render_state(
        render_id,
        output_name=output_name,
        upload_filename=upload_filename,
        speed_unit=speed_unit,
        freq_max=freq_max,
        params=params,
        plots=plots,
        uploaded_audio=uploaded_plot_copy,
        uploaded_sr=uploaded_sr,
        generated_audio=generated,
        freqs=freqs,
        psd_observed=psd_observed,
        psd_inverted=psd_inverted,
        stft=stft,
        stft_times=stft_times,
        quantities=quantities,
    )

    meta = {
        "output_name": output_name,
        "freq_max": freq_max,
    }
    return render_template(
        "index.html",
        **build_success_context(render_id, meta, plots, params, speed_unit, upload_filename),
    )


@app.route("/update-freq-max", methods=["POST"])
def update_freq_max():
    render_id = session.get("last_render_id")
    if not render_id or not (RENDERS_DIR / render_id / "meta.json").exists():
        return render_template(
            "index.html",
            error="No recent render found. Generate pass-by audio first.",
            **form_context(freq_max=parse_freq_max()),
        )

    meta, arrays = load_render_state(render_id)
    freq_max = parse_freq_max(default=float(meta.get("freq_max", DEFAULT_FREQ_MAX)))
    meta, params, plots = regenerate_plots_from_state(render_id, freq_max)
    speed_unit = meta.get("speed_unit", "mps")

    return render_template(
        "index.html",
        **build_success_context(
            render_id,
            meta,
            plots,
            params,
            speed_unit,
            meta.get("upload_filename"),
        ),
    )


@app.route("/download-bundle", methods=["POST"])
def download_bundle():
    render_id = session.get("last_render_id")
    if not render_id or not (RENDERS_DIR / render_id / "meta.json").exists():
        return render_template(
            "index.html",
            error="No recent render found. Generate pass-by audio first.",
            **form_context(),
        )

    meta, _ = load_render_state(render_id)
    bundle_name = sanitize_bundle_name(request.form.get("bundle_name", "doppler_sim_export"))
    plot_dir = PLOTS_DIR / render_id
    audio_path = GENERATED_DIR / meta["output_name"]

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(audio_path, arcname=f"{bundle_name}/generated_audio.wav")
        for plot_key, plot_filename in meta["plots"].items():
            plot_path = plot_dir / plot_filename
            if plot_path.exists():
                export_name = PLOT_EXPORT_NAMES.get(plot_key, plot_filename)
                archive.write(plot_path, arcname=f"{bundle_name}/{export_name}")

    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{bundle_name}.zip",
    )


@app.route("/static/plots/<render_id>/<path:filename>")
def plot_file(render_id: str, filename: str):
    return send_from_directory(PLOTS_DIR / render_id, filename)


@app.route("/static/generated/<path:filename>")
def generated_file(filename: str):
    return send_from_directory(GENERATED_DIR, filename)


@app.route("/static/compare/<path:filename>")
def compare_file(filename: str):
    return send_from_directory(COMPARE_DIR, filename)


@app.route("/download/<path:filename>")
def download_file(filename: str):
    return send_from_directory(
        GENERATED_DIR,
        filename,
        as_attachment=True,
        download_name=filename,
    )


# ---------------------------------------------------------------------------
# Spectrogram Explorer (separate tab — does not use Doppler pipeline)
# ---------------------------------------------------------------------------

SPECG_SR = 22050
SPECG_MAX_DURATION_S = 90.0
SPECG_ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SPECG_DEFAULT_FMAX_HZ = 2000.0
SPECG_FMAX_MIN_HZ = 100.0
SPECG_N_FFT_CHOICES = [256, 512, 1024, 2048, 4096, 8192]
SPECG_HOP_CHOICES = [64, 128, 256, 512, 1024, 2048]
SPECG_WINDOW_CHOICES: Dict[str, str] = {
    "hann": "Hann",
    "hamming": "Hamming",
    "blackman": "Blackman",
    "bartlett": "Bartlett",
    "nuttall": "Nuttall",
    "flattop": "Flat-top",
    "boxcar": "Boxcar (rectangular)",
}


SPECG_WB_PRESETS: Dict[str, Tuple[str, int, int]] = {
    "256_64": ("Very short — best time resolution", 256, 64),
    "512_128": ("Short (default)", 512, 128),
    "1024_256": ("Medium-short", 1024, 256),
}
SPECG_NB_PRESETS: Dict[str, Tuple[str, int, int]] = {
    "2048_512": ("Medium-long", 2048, 512),
    "4096_1024": ("Long (default)", 4096, 1024),
    "8192_2048": ("Very long — best frequency resolution", 8192, 2048),
}
SPECG_DEFAULT_WB_PRESET = "512_128"
SPECG_DEFAULT_NB_PRESET = "4096_1024"


@dataclass
class SpecgStftParams:
    n_fft: int
    hop_length: int
    window: str


@dataclass
class SpecgAnalysisParams:
    stft: SpecgStftParams
    wideband: SpecgStftParams
    narrowband: SpecgStftParams
    wb_preset: str
    nb_preset: str
    wb_preset_label: str
    nb_preset_label: str


SPECG_DEFAULT_STFT = SpecgStftParams(2048, 512, "hann")
SPECG_DEFAULT_WIDEBAND = SpecgStftParams(512, 128, "hann")
SPECG_DEFAULT_NARROWBAND = SpecgStftParams(4096, 1024, "hann")
SPECG_DEFAULT_ANALYSIS = SpecgAnalysisParams(
    SPECG_DEFAULT_STFT,
    SPECG_DEFAULT_WIDEBAND,
    SPECG_DEFAULT_NARROWBAND,
    SPECG_DEFAULT_WB_PRESET,
    SPECG_DEFAULT_NB_PRESET,
    SPECG_WB_PRESETS[SPECG_DEFAULT_WB_PRESET][0],
    SPECG_NB_PRESETS[SPECG_DEFAULT_NB_PRESET][0],
)


def specg_parse_int_choice(raw: str | None, choices: List[int], default: int) -> int:
    try:
        value = int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default
    return value if value in choices else default


def specg_parse_window(raw: str | None, default: str = "hann") -> str:
    if raw in SPECG_WINDOW_CHOICES:
        return raw
    return default if default in SPECG_WINDOW_CHOICES else "hann"


def specg_parse_stft_group(form, default: SpecgStftParams, window: str) -> SpecgStftParams:
    n_fft = specg_parse_int_choice(form.get("stft_n_fft"), SPECG_N_FFT_CHOICES, default.n_fft)
    hop = specg_parse_int_choice(form.get("stft_hop"), SPECG_HOP_CHOICES, default.hop_length)
    if hop > n_fft:
        hop = min(h for h in SPECG_HOP_CHOICES if h <= n_fft)
    return SpecgStftParams(n_fft=n_fft, hop_length=hop, window=window)


def specg_parse_preset_key(raw: str | None, presets: Dict[str, Tuple[str, int, int]], default: str) -> str:
    return raw if raw in presets else default


def specg_resolve_wb_preset(requested: str, stft_n_fft: int) -> str:
    if requested in SPECG_WB_PRESETS and SPECG_WB_PRESETS[requested][1] < stft_n_fft:
        return requested
    valid = [k for k, v in SPECG_WB_PRESETS.items() if v[1] < stft_n_fft]
    if not valid:
        return min(SPECG_WB_PRESETS, key=lambda k: SPECG_WB_PRESETS[k][1])
    return max(valid, key=lambda k: SPECG_WB_PRESETS[k][1])


def specg_resolve_nb_preset(requested: str, stft_n_fft: int) -> str:
    if requested in SPECG_NB_PRESETS and SPECG_NB_PRESETS[requested][1] > stft_n_fft:
        return requested
    valid = [k for k, v in SPECG_NB_PRESETS.items() if v[1] > stft_n_fft]
    if not valid:
        return max(SPECG_NB_PRESETS, key=lambda k: SPECG_NB_PRESETS[k][1])
    return min(valid, key=lambda k: SPECG_NB_PRESETS[k][1])


def specg_preset_params(key: str, presets: Dict[str, Tuple[str, int, int]], window: str) -> SpecgStftParams:
    _, n_fft, hop = presets[key]
    return SpecgStftParams(n_fft=n_fft, hop_length=hop, window=window)


def specg_parse_analysis_params(form) -> SpecgAnalysisParams:
    window = specg_parse_window(form.get("stft_window"), SPECG_DEFAULT_STFT.window)
    stft = specg_parse_stft_group(form, SPECG_DEFAULT_STFT, window)
    wb_key = specg_resolve_wb_preset(
        specg_parse_preset_key(form.get("wb_preset"), SPECG_WB_PRESETS, SPECG_DEFAULT_WB_PRESET),
        stft.n_fft,
    )
    nb_key = specg_resolve_nb_preset(
        specg_parse_preset_key(form.get("nb_preset"), SPECG_NB_PRESETS, SPECG_DEFAULT_NB_PRESET),
        stft.n_fft,
    )
    return SpecgAnalysisParams(
        stft=stft,
        wideband=specg_preset_params(wb_key, SPECG_WB_PRESETS, window),
        narrowband=specg_preset_params(nb_key, SPECG_NB_PRESETS, window),
        wb_preset=wb_key,
        nb_preset=nb_key,
        wb_preset_label=SPECG_WB_PRESETS[wb_key][0],
        nb_preset_label=SPECG_NB_PRESETS[nb_key][0],
    )


def specg_window_label(window: str) -> str:
    return SPECG_WINDOW_CHOICES.get(window, window)


def specg_analysis_template_context(params: SpecgAnalysisParams) -> Dict[str, Any]:
    return {
        "stft_n_fft": params.stft.n_fft,
        "stft_hop": params.stft.hop_length,
        "wb_preset": params.wb_preset,
        "nb_preset": params.nb_preset,
        "stft_window": params.stft.window,
        "n_fft_choices": SPECG_N_FFT_CHOICES,
        "hop_choices": SPECG_HOP_CHOICES,
        "window_choices": SPECG_WINDOW_CHOICES,
        "wb_presets": SPECG_WB_PRESETS,
        "nb_presets": SPECG_NB_PRESETS,
    }


def specg_parse_fmax_hz(raw: str | None, sr: int) -> float:
    nyquist = sr / 2.0
    try:
        fmax = float(raw) if raw not in (None, "") else SPECG_DEFAULT_FMAX_HZ
    except (TypeError, ValueError):
        fmax = SPECG_DEFAULT_FMAX_HZ
    return float(np.clip(fmax, SPECG_FMAX_MIN_HZ, nyquist))


def specg_parse_fmax_from_form(form, sr: int) -> float:
    custom = (form.get("fmax_hz_custom") or "").strip()
    if custom:
        return specg_parse_fmax_hz(custom, sr)
    return specg_parse_fmax_hz(form.get("fmax_hz"), sr)


def specg_load_audio(path: str) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=SPECG_SR, mono=True)
    max_samples = int(SPECG_MAX_DURATION_S * sr)
    if y.size > max_samples:
        y = y[:max_samples]
    return y.astype(np.float32), int(sr)


def specg_fig_to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def specg_fmt_hz(hz: float) -> str:
    return f"{int(hz)} Hz" if hz == int(hz) else f"{hz:.1f} Hz"


def specg_panel(
    title: str,
    image: str,
    sr: int,
    fmax_hz: float,
    extra: List[Tuple[str, str]],
) -> Dict[str, Any]:
    labels: List[Tuple[str, str]] = [
        ("Sample rate", f"{sr} Hz"),
        ("fmax", specg_fmt_hz(fmax_hz)),
    ]
    labels.extend(extra)
    return {"title": title, "image": image, "labels": labels}


def specg_style_hz_ax(ax: plt.Axes, title: str, fmax_hz: float) -> None:
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.set_ylim(0, fmax_hz)
    ax.tick_params(colors="#aab0c0", labelsize=7)


def specg_plot_spec(
    S: np.ndarray,
    sr: int,
    hop: int,
    title: str,
    fmax_hz: float,
    *,
    y_axis: str = "hz",
    cmap: str = "magma",
    is_power: bool = False,
) -> str:
    if np.max(np.abs(S)) <= 0:
        S_db = S
    elif is_power:
        S_db = librosa.power_to_db(S, ref=np.max)
    else:
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    if y_axis == "chroma":
        librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=hop,
            x_axis="time",
            y_axis="chroma",
            ax=ax,
            cmap=cmap,
        )
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.tick_params(colors="#aab0c0", labelsize=7)
    elif y_axis == "mel":
        librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=hop,
            x_axis="time",
            y_axis="mel",
            fmax=fmax_hz,
            ax=ax,
            cmap=cmap,
        )
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.tick_params(colors="#aab0c0", labelsize=7)
    else:
        librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=hop,
            x_axis="time",
            y_axis="hz",
            fmax=fmax_hz,
            ax=ax,
            cmap=cmap,
        )
        specg_style_hz_ax(ax, title, fmax_hz)
    fig.tight_layout()
    return specg_fig_to_b64(fig)


def spec_stft(y: np.ndarray, sr: int, fmax_hz: float, p: SpecgStftParams) -> Dict[str, Any]:
    title = "STFT Spectrogram"
    S = np.abs(librosa.stft(y, n_fft=p.n_fft, hop_length=p.hop_length, window=p.window))
    img = specg_plot_spec(S, sr, p.hop_length, title, fmax_hz)
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "STFT"),
            ("n_fft", str(p.n_fft)),
            ("hop_length", str(p.hop_length)),
            ("Window", specg_window_label(p.window)),
            ("Y-axis", "Frequency (Hz)"),
            ("Scale", "Amplitude → dB"),
        ],
    )


def spec_wideband(
    y: np.ndarray, sr: int, fmax_hz: float, p: SpecgStftParams, profile: str
) -> Dict[str, Any]:
    title = "Wideband Spectrogram"
    S = np.abs(librosa.stft(y, n_fft=p.n_fft, hop_length=p.hop_length, window=p.window))
    img = specg_plot_spec(S, sr, p.hop_length, title, fmax_hz)
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "STFT (wideband)"),
            ("Profile", profile),
            ("n_fft", str(p.n_fft)),
            ("hop_length", str(p.hop_length)),
            ("Window", specg_window_label(p.window)),
            ("Trade-off", "Short window → better time resolution"),
            ("Y-axis", "Frequency (Hz)"),
            ("Scale", "Amplitude → dB"),
        ],
    )


def spec_narrowband(
    y: np.ndarray, sr: int, fmax_hz: float, p: SpecgStftParams, profile: str
) -> Dict[str, Any]:
    title = "Narrowband Spectrogram"
    S = np.abs(librosa.stft(y, n_fft=p.n_fft, hop_length=p.hop_length, window=p.window))
    img = specg_plot_spec(S, sr, p.hop_length, title, fmax_hz)
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "STFT (narrowband)"),
            ("Profile", profile),
            ("n_fft", str(p.n_fft)),
            ("hop_length", str(p.hop_length)),
            ("Window", specg_window_label(p.window)),
            ("Trade-off", "Long window → better frequency resolution"),
            ("Y-axis", "Frequency (Hz)"),
            ("Scale", "Amplitude → dB"),
        ],
    )


def spec_mel(y: np.ndarray, sr: int, fmax_hz: float, p: SpecgStftParams) -> Dict[str, Any]:
    title = "Mel Spectrogram"
    n_mels = 128
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        window=p.window,
        n_mels=n_mels,
        fmax=fmax_hz,
    )
    img = specg_plot_spec(S, sr, p.hop_length, title, fmax_hz, y_axis="mel", is_power=True)
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "Mel filterbank (STFT)"),
            ("n_fft", str(p.n_fft)),
            ("hop_length", str(p.hop_length)),
            ("Window", specg_window_label(p.window)),
            ("n_mels", str(n_mels)),
            ("Y-axis", "Mel bands"),
            ("Scale", "Power"),
        ],
    )


def spec_log_mel(y: np.ndarray, sr: int, fmax_hz: float, p: SpecgStftParams) -> Dict[str, Any]:
    title = "Log-Mel Spectrogram"
    n_mels = 128
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        window=p.window,
        n_mels=n_mels,
        fmax=fmax_hz,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=p.hop_length,
        x_axis="time",
        y_axis="mel",
        fmax=fmax_hz,
        ax=ax,
        cmap="magma",
    )
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="#aab0c0", labelsize=7)
    fig.tight_layout()
    img = specg_fig_to_b64(fig)
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "Mel filterbank (STFT)"),
            ("n_fft", str(p.n_fft)),
            ("hop_length", str(p.hop_length)),
            ("Window", specg_window_label(p.window)),
            ("n_mels", str(n_mels)),
            ("Y-axis", "Mel bands"),
            ("Scale", "log₁₀ power (dB)"),
        ],
    )


def spec_cqt(y: np.ndarray, sr: int, fmax_hz: float) -> Dict[str, Any]:
    title = "CQT Spectrogram"
    hop = 512
    bins_per_octave = 12
    fmin = librosa.note_to_hz("C1")
    n_bins = int(np.ceil(bins_per_octave * np.log2(fmax_hz / fmin)))
    C = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=hop,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin,
        )
    )
    cqt_freqs = librosa.cqt_frequencies(
        n_bins=C.shape[0], fmin=fmin, bins_per_octave=bins_per_octave
    )
    keep = cqt_freqs <= fmax_hz
    C = C[keep]
    cqt_freqs = cqt_freqs[keep]
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    times = librosa.times_like(C, hop_length=hop, sr=sr)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    ax.pcolormesh(times, cqt_freqs, C_db, shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_ylim(max(cqt_freqs.min(), 20.0), fmax_hz)
    ax.set_xlabel("Time (s)", color="#aab0c0", fontsize=7)
    ax.set_ylabel("Hz", color="#aab0c0", fontsize=7)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="#aab0c0", labelsize=7)
    fig.tight_layout()
    img = specg_fig_to_b64(fig)
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "Constant-Q (CQT)"),
            ("hop_length", str(hop)),
            ("bins_per_octave", str(bins_per_octave)),
            ("fmin", specg_fmt_hz(fmin)),
            ("n_bins", str(int(C.shape[0]))),
            ("Y-axis", "log frequency (Hz)"),
            ("Scale", "Amplitude → dB"),
        ],
    )


def spec_wavelet(y: np.ndarray, sr: int, fmax_hz: float) -> Dict[str, Any]:
    title = "Wavelet Scalogram"
    n_scales = 96
    wavelet = "morl (Morlet)"
    scales = np.geomspace(1, 128, num=n_scales)
    coef, freqs_hz = pywt.cwt(y, scales, "morl", sampling_period=1.0 / sr)
    keep = freqs_hz <= fmax_hz
    coef = coef[keep]
    freqs_hz = freqs_hz[keep]
    power_db = librosa.power_to_db(np.abs(coef) ** 2, ref=np.max)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    extent = [0, len(y) / sr, freqs_hz[-1], freqs_hz[0]]
    ax.imshow(power_db, aspect="auto", origin="upper", extent=extent, cmap="magma")
    ax.set_xlabel("Time (s)", color="#aab0c0", fontsize=7)
    ax.set_ylabel("Hz", color="#aab0c0", fontsize=7)
    specg_style_hz_ax(ax, title, fmax_hz)
    fig.tight_layout()
    img = specg_fig_to_b64(fig)
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "CWT (continuous wavelet)"),
            ("Wavelet", wavelet),
            ("n_scales", str(n_scales)),
            ("Scale range", "1–128 (geomspace)"),
            ("hop_length", "N/A (continuous)"),
            ("Y-axis", "Frequency (Hz)"),
            ("Scale", "Power → dB"),
        ],
    )


def spec_reassigned(y: np.ndarray, sr: int, fmax_hz: float, p: SpecgStftParams) -> Dict[str, Any]:
    title = "Reassigned Spectrogram"
    freqs, times, mags = librosa.reassigned_spectrogram(
        y,
        sr=sr,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        window=p.window,
        fill_nan=True,
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    mags_db = librosa.power_to_db(mags, ref=np.max)
    img_mesh = ax.pcolormesh(times, freqs, mags_db, shading="auto", cmap="magma")
    ax.set_xlabel("Time (s)", color="#aab0c0", fontsize=7)
    ax.set_ylabel("Hz", color="#aab0c0", fontsize=7)
    specg_style_hz_ax(ax, title, fmax_hz)
    fig.colorbar(img_mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    img = specg_fig_to_b64(fig)
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "Reassigned STFT"),
            ("n_fft", str(p.n_fft)),
            ("hop_length", str(p.hop_length)),
            ("Window", specg_window_label(p.window)),
            ("fill_nan", "true"),
            ("Y-axis", "Frequency (Hz)"),
            ("Scale", "Power → dB"),
        ],
    )


def spec_synchrosqueezed(y: np.ndarray, sr: int, fmax_hz: float) -> Dict[str, Any]:
    from ssqueezepy import ssq_cwt

    title = "Synchrosqueezed Spectrogram"
    target_sr = min(sr, 11025)
    if sr != target_sr:
        y_ssq = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    else:
        y_ssq = y
    n_scales = 64
    scales = np.geomspace(2, 128, n_scales)
    Tx, _, ssq_freqs, *_ = ssq_cwt(y_ssq, wavelet="morlet", fs=target_sr, scales=scales)
    keep = ssq_freqs <= fmax_hz
    Tx = Tx[keep]
    ssq_freqs = ssq_freqs[keep]

    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    Tx_db = librosa.power_to_db(np.abs(Tx) ** 2, ref=np.max)
    extent = [0, len(y_ssq) / target_sr, ssq_freqs[-1], ssq_freqs[0]]
    ax.imshow(Tx_db, aspect="auto", origin="upper", extent=extent, cmap="magma")
    ax.set_xlabel("Time (s)", color="#aab0c0", fontsize=7)
    ax.set_ylabel("Hz", color="#aab0c0", fontsize=7)
    specg_style_hz_ax(ax, title, fmax_hz)
    fig.tight_layout()
    img = specg_fig_to_b64(fig)
    resample_note = f"{target_sr} Hz" if target_sr != sr else "none"
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "Synchrosqueezed CWT"),
            ("Wavelet", "Morlet (ssqueezepy)"),
            ("n_scales", str(n_scales)),
            ("Analysis sr", f"{target_sr} Hz"),
            ("Resample", resample_note),
            ("hop_length", "N/A"),
            ("Y-axis", "Frequency (Hz)"),
            ("Scale", "Power → dB"),
        ],
    )


def spec_chroma(y: np.ndarray, sr: int, fmax_hz: float) -> Dict[str, Any]:
    title = "Chromagram"
    hop = 512
    n_chroma = 12
    bins_per_octave = 36
    fmin = librosa.note_to_hz("C1")
    n_octaves = max(1, int(np.floor(np.log2(fmax_hz / fmin))))
    C = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop,
        n_chroma=n_chroma,
        fmin=fmin,
        n_octaves=n_octaves,
        bins_per_octave=bins_per_octave,
    )
    img = specg_plot_spec(C, sr, hop, title, fmax_hz, y_axis="chroma")
    effective_fmax = fmin * (2**n_octaves)
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "Chroma-CQT"),
            ("hop_length", str(hop)),
            ("n_chroma", str(n_chroma)),
            ("fmin", specg_fmt_hz(fmin)),
            ("n_octaves", str(n_octaves)),
            ("bins_per_octave", str(bins_per_octave)),
            ("Band top", specg_fmt_hz(min(effective_fmax, fmax_hz))),
            ("Y-axis", "Pitch class (C–B)"),
            ("Scale", "Energy → dB"),
        ],
    )


def specg_error_placeholder(title: str, message: str) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor="#1a1020")
    ax.set_facecolor("#1a1020")
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        title,
        ha="center",
        va="center",
        color="#ff6b6b",
        fontsize=9,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.35,
        message[:120],
        ha="center",
        va="center",
        color="#ccc",
        fontsize=7,
        wrap=True,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    return specg_fig_to_b64(fig)


def build_all_spectrograms(
    y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams
) -> List[Dict[str, Any]]:
    builders: List[Tuple[str, Callable[[], Dict[str, Any]]]] = [
        ("STFT Spectrogram", lambda: spec_stft(y, sr, fmax_hz, analysis.stft)),
        (
            "Wideband Spectrogram",
            lambda: spec_wideband(y, sr, fmax_hz, analysis.wideband, analysis.wb_preset_label),
        ),
        (
            "Narrowband Spectrogram",
            lambda: spec_narrowband(y, sr, fmax_hz, analysis.narrowband, analysis.nb_preset_label),
        ),
        ("Mel Spectrogram", lambda: spec_mel(y, sr, fmax_hz, analysis.stft)),
        ("Log-Mel Spectrogram", lambda: spec_log_mel(y, sr, fmax_hz, analysis.stft)),
        ("CQT Spectrogram", lambda: spec_cqt(y, sr, fmax_hz)),
        ("Wavelet Scalogram", lambda: spec_wavelet(y, sr, fmax_hz)),
        ("Reassigned Spectrogram", lambda: spec_reassigned(y, sr, fmax_hz, analysis.stft)),
        ("Synchrosqueezed Spectrogram", lambda: spec_synchrosqueezed(y, sr, fmax_hz)),
        ("Chromagram", lambda: spec_chroma(y, sr, fmax_hz)),
    ]
    out: List[Dict[str, Any]] = []
    for title, fn in builders:
        try:
            out.append(fn())
        except Exception as exc:
            out.append(
                {
                    "title": title,
                    "image": specg_error_placeholder(title, str(exc)),
                    "labels": [("Error", str(exc)[:80])],
                }
            )
    return out


@app.route("/spectrograms", methods=["GET", "POST"])
def spectrograms():
    panels: List[Dict[str, str]] | None = None
    error: str | None = None
    filename: str | None = None
    duration_s: float | None = None
    fmax_hz = SPECG_DEFAULT_FMAX_HZ
    nyquist_hz = SPECG_SR / 2.0
    analysis = SPECG_DEFAULT_ANALYSIS

    if request.method == "POST":
        analysis = specg_parse_analysis_params(request.form)
        fmax_hz = specg_parse_fmax_from_form(request.form, SPECG_SR)
        f = request.files.get("audio")
        if f is None or not f.filename:
            error = "Please choose a WAV or MP3 file."
        else:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in SPECG_ALLOWED_EXT:
                error = f"Unsupported format {ext}. Use WAV, MP3, FLAC, OGG, or M4A."
            else:
                suffix = ext or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    f.save(tmp.name)
                    tmp_path = tmp.name
                try:
                    y, sr = specg_load_audio(tmp_path)
                    filename = f.filename
                    duration_s = float(len(y) / sr)
                    fmax_hz = specg_parse_fmax_from_form(request.form, sr)
                    panels = build_all_spectrograms(y, sr, fmax_hz, analysis)
                except Exception as exc:
                    error = f"Could not process audio: {exc}"
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    return render_template(
        "index.html",
        active_tab="spectrograms",
        panels=panels,
        error=error,
        filename=filename,
        duration_s=duration_s,
        fmax_hz=int(fmax_hz) if fmax_hz == int(fmax_hz) else fmax_hz,
        default_fmax_hz=int(SPECG_DEFAULT_FMAX_HZ),
        nyquist_hz=int(nyquist_hz),
        fmax_presets=[500, 1000, 2000, 4000, 6000, 8000, 11025],
        **specg_analysis_template_context(analysis),
    )


# ---------------------------------------------------------------------------
# Audio Comparison (two-clip spectrogram comparison tab)
# ---------------------------------------------------------------------------

COMPARE_SR = 22050
COMPARE_DEFAULT_MAX_Y_FREQ = 2500.0
COMPARE_ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


COMPARE_SPEC_N_FFT = 4096
COMPARE_SPEC_HOP = 64
COMPARE_SPEC_WIN = 4096
COMPARE_SPEC_WINDOW = "hann"


def _compare_style_ax(ax: plt.Axes, *, title: str = "") -> None:
    ax.set_facecolor("#0f1117")
    if title:
        ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#aab0c0", labelsize=8)
    ax.xaxis.label.set_color("#aab0c0")
    ax.yaxis.label.set_color("#aab0c0")


def _compare_plot_waveform(y: np.ndarray, sr: int, color: str, peak: float) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 2.4), facecolor="#0f1117")
    times = np.arange(len(y), dtype=np.float64) / float(sr)
    ax.plot(times, y, color=color, linewidth=0.5, alpha=0.95)
    ax.axhline(peak, color=color, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.axhline(-peak, color=color, linestyle="--", linewidth=0.6, alpha=0.45)
    _compare_style_ax(ax, title=f"Waveform (peak {peak:.4f})")
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    margin = max(peak * 1.08, 1e-6)
    ax.set_ylim(-margin, margin)
    ax.set_xlim(0.0, max(float(len(y)) / float(sr), 1e-6))
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    return specg_fig_to_b64(fig)


def _compare_plot_spectrogram(y: np.ndarray, sr: int, max_y_freq: float) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 3.6), facecolor="#0f1117")
    stft = librosa.stft(
        y,
        n_fft=COMPARE_SPEC_N_FFT,
        hop_length=COMPARE_SPEC_HOP,
        win_length=COMPARE_SPEC_WIN,
        window=COMPARE_SPEC_WINDOW,
    )
    s_power = np.abs(stft) ** 2
    d_db = librosa.power_to_db(s_power, ref=np.max)
    vmax = float(np.max(d_db))
    vmin = vmax - 80.0
    librosa.display.specshow(
        d_db,
        sr=sr,
        hop_length=COMPARE_SPEC_HOP,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
        shading="gouraud",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_ylim(0, max_y_freq)
    ax.set_yticks(np.linspace(0, max_y_freq, 6))
    _compare_style_ax(ax, title="Spectrogram")
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    fig.tight_layout()
    return specg_fig_to_b64(fig)


def _compare_plot_rms(y: np.ndarray, sr: int, color: str) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 2.0), facecolor="#0f1117")
    rms = librosa.feature.rms(
        y=y,
        frame_length=COMPARE_SPEC_N_FFT,
        hop_length=COMPARE_SPEC_HOP,
        center=True,
    )[0]
    times = librosa.times_like(rms, sr=sr, hop_length=COMPARE_SPEC_HOP)
    bar_width = (
        (float(times[1] - times[0]) * 0.9)
        if len(times) > 1
        else (float(COMPARE_SPEC_HOP) / float(sr))
    )
    ax.bar(times, rms, width=bar_width, color=color, edgecolor="none", alpha=0.9)
    _compare_style_ax(ax, title="RMS envelope")
    ax.set_ylabel("RMS", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylim(0, max(1e-6, float(np.max(rms)) * 1.15))
    ax.set_xlim(0.0, max(float(len(y)) / float(sr), 1e-6))
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    return specg_fig_to_b64(fig)


def build_compare_plots(
    y_a: np.ndarray,
    y_b: np.ndarray,
    sr: int,
    max_y_freq: float,
) -> Dict[str, str]:
    max_y_freq = float(max_y_freq) if max_y_freq else COMPARE_DEFAULT_MAX_Y_FREQ
    if max_y_freq <= 0:
        max_y_freq = COMPARE_DEFAULT_MAX_Y_FREQ
    peak_a = float(np.max(np.abs(y_a)))
    peak_b = float(np.max(np.abs(y_b)))
    return {
        "waveform_a": _compare_plot_waveform(y_a, sr, "#6c9eff", peak_a),
        "waveform_b": _compare_plot_waveform(y_b, sr, "#f0883e", peak_b),
        "spectrogram_a": _compare_plot_spectrogram(y_a, sr, max_y_freq),
        "spectrogram_b": _compare_plot_spectrogram(y_b, sr, max_y_freq),
        "rms_a": _compare_plot_rms(y_a, sr, "#6c9eff"),
        "rms_b": _compare_plot_rms(y_b, sr, "#f0883e"),
    }


def compute_audio_comparison_metrics(
    y_a: np.ndarray, y_b: np.ndarray, sr: int
) -> Dict[str, float]:
    n_fft = 2048
    hop_length = 256

    rms_a = librosa.feature.rms(y=y_a, frame_length=n_fft, hop_length=hop_length)[0]
    rms_b = librosa.feature.rms(y=y_b, frame_length=n_fft, hop_length=hop_length)[0]
    n_env = min(len(rms_a), len(rms_b))
    rms_a = rms_a[:n_env]
    rms_b = rms_b[:n_env]

    rms_a_norm = rms_a / (np.max(rms_a) + 1e-9)
    rms_b_norm = rms_b / (np.max(rms_b) + 1e-9)
    amp_overlap = float(
        np.sum(np.minimum(rms_a_norm, rms_b_norm))
        / (np.sum(np.maximum(rms_a_norm, rms_b_norm)) + 1e-9)
        * 100.0
    )

    if n_env > 1 and (np.std(rms_a_norm) > 1e-9) and (np.std(rms_b_norm) > 1e-9):
        env_corr = float(np.corrcoef(rms_a_norm, rms_b_norm)[0, 1])
    else:
        env_corr = 0.0
    env_corr_pct = float(np.clip((env_corr + 1.0) * 50.0, 0.0, 100.0))

    stft_a = np.abs(librosa.stft(y_a, n_fft=n_fft, hop_length=hop_length))
    stft_b = np.abs(librosa.stft(y_b, n_fft=n_fft, hop_length=hop_length))
    spec_a = np.mean(stft_a, axis=1)
    spec_b = np.mean(stft_b, axis=1)
    spec_a_norm = spec_a / (np.sum(spec_a) + 1e-9)
    spec_b_norm = spec_b / (np.sum(spec_b) + 1e-9)
    spectral_overlap = float(np.sum(np.minimum(spec_a_norm, spec_b_norm)) * 100.0)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    dom_freq_a = float(freqs[int(np.argmax(spec_a))]) if len(spec_a) else 0.0
    dom_freq_b = float(freqs[int(np.argmax(spec_b))]) if len(spec_b) else 0.0

    overall_similarity = float(
        np.clip((spectral_overlap * 0.55) + (amp_overlap * 0.30) + (env_corr_pct * 0.15), 0.0, 100.0)
    )

    return {
        "duration_a_sec": float(len(y_a) / sr),
        "duration_b_sec": float(len(y_b) / sr),
        "peak_a": float(np.max(np.abs(y_a))),
        "peak_b": float(np.max(np.abs(y_b))),
        "dominant_freq_a_hz": dom_freq_a,
        "dominant_freq_b_hz": dom_freq_b,
        "envelope_correlation_percent": env_corr_pct,
        "spectral_overlap_percent": spectral_overlap,
        "amplitude_overlap_percent": amp_overlap,
        "overall_similarity_percent": overall_similarity,
    }


def compare_load_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=COMPARE_SR, mono=True)
    return y.astype(np.float32)


@app.route("/compare", methods=["GET", "POST"])
def compare_audio():
    error: str | None = None
    compare_plots: Dict[str, str] | None = None
    metrics: Dict[str, float] | None = None
    filename_a: str | None = None
    filename_b: str | None = None
    max_y_freq = COMPARE_DEFAULT_MAX_Y_FREQ

    if request.method == "POST":
        try:
            max_y_freq = float(request.form.get("max_y_freq", COMPARE_DEFAULT_MAX_Y_FREQ))
        except (TypeError, ValueError):
            max_y_freq = COMPARE_DEFAULT_MAX_Y_FREQ
        if max_y_freq <= 0:
            max_y_freq = COMPARE_DEFAULT_MAX_Y_FREQ

        file_a = request.files.get("file_a")
        file_b = request.files.get("file_b")
        if file_a is None or file_b is None or not file_a.filename or not file_b.filename:
            error = "Please upload both audio clips."
        else:
            ext_a = os.path.splitext(file_a.filename)[1].lower()
            ext_b = os.path.splitext(file_b.filename)[1].lower()
            if ext_a not in COMPARE_ALLOWED_EXT or ext_b not in COMPARE_ALLOWED_EXT:
                error = "Unsupported format. Use WAV, MP3, FLAC, OGG, or M4A."
            else:
                temp_paths: List[str] = []
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext_a) as tmp_a:
                        file_a.save(tmp_a.name)
                        temp_paths.append(tmp_a.name)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext_b) as tmp_b:
                        file_b.save(tmp_b.name)
                        temp_paths.append(tmp_b.name)

                    y_a = compare_load_audio(temp_paths[0])
                    y_b = compare_load_audio(temp_paths[1])
                    if y_a.size == 0 or y_b.size == 0:
                        error = "One of the uploaded files appears empty or unreadable."
                    else:
                        filename_a = file_a.filename
                        filename_b = file_b.filename
                        metrics = compute_audio_comparison_metrics(y_a, y_b, COMPARE_SR)
                        try:
                            compare_plots = build_compare_plots(y_a, y_b, COMPARE_SR, max_y_freq)
                        except Exception as plot_exc:
                            error = f"Failed to generate comparison plots: {plot_exc}"
                except Exception as exc:
                    error = f"Could not process audio: {exc}"
                finally:
                    for p in temp_paths:
                        try:
                            os.unlink(p)
                        except OSError:
                            pass

    return render_template(
        "index.html",
        active_tab="compare",
        error=error,
        compare_plots=compare_plots,
        compare_metrics=metrics,
        filename_a=filename_a,
        filename_b=filename_b,
        max_y_freq=int(max_y_freq) if max_y_freq == int(max_y_freq) else max_y_freq,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)
