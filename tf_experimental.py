"""Experimental time–frequency transforms (Wigner–Ville).

Isolated from Pass-By synthesis and default Spectrogram Explorer POST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from tf_analysis import ReassignedAtoms, StftParams, compute_reassigned_atoms, plot_reassigned_specg_b64, plot_stft_specg_b64

__all__ = [
    "ExperimentalComparison",
    "WvdResult",
    "build_experimental_comparison",
    "compute_wvd",
    "plot_wvd_specg_b64",
]

EXPERIMENTAL_SR = 22050
EXPERIMENTAL_MAX_DURATION_S = 30.0
WVD_MAX_DURATION_S = 6.0
WVD_ANALYSIS_SR = 11025


@dataclass
class WvdResult:
    times: np.ndarray
    freqs: np.ndarray
    power: np.ndarray
    sr: int
    segment_start_s: float
    segment_duration_s: float


@dataclass
class ExperimentalComparison:
    stft_image: str
    reassigned_image: str
    wvd_image: str
    labels: list[tuple[str, str]]
    error: str | None = None


def _fig_to_b64(fig: plt.Figure) -> str:
    import base64
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _error_panel_b64(title: str, message: str) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor="#1a1020")
    ax.set_facecolor("#1a1020")
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", color="#ff6b6b", fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.35, message[:160], ha="center", va="center", color="#ccc", fontsize=8, transform=ax.transAxes)
    fig.tight_layout()
    return _fig_to_b64(fig)


def compute_wvd(
    y: np.ndarray,
    sr: int,
    *,
    max_duration_s: float = WVD_MAX_DURATION_S,
    analysis_sr: int = WVD_ANALYSIS_SR,
    fmax_hz: float,
    n_freq: int = 512,
    max_lag: int = 255,
    time_hop: int = 64,
) -> WvdResult:
    """Smoothed pseudo-Wigner–Ville on a centered segment (downsampled for cost)."""
    if len(y) == 0:
        raise ValueError("Empty audio.")

    y = np.asarray(y, dtype=float)
    duration_s = len(y) / float(sr)
    seg_dur = min(duration_s, max_duration_s)
    seg_samples = int(seg_dur * sr)
    start = max(0, (len(y) - seg_samples) // 2)
    segment = y[start : start + seg_samples]

    if sr != analysis_sr:
        segment = librosa.resample(segment, orig_sr=sr, target_sr=analysis_sr)
        sr_out = analysis_sr
    else:
        sr_out = sr

    segment = segment / (np.max(np.abs(segment)) + 1e-12)
    z = hilbert(segment)
    n = len(z)
    n_time = max(1, (n + time_hop - 1) // time_hop)
    wvd = np.zeros((n_freq, n_time), dtype=float)
    times = np.zeros(n_time, dtype=float)

    for ti, t in enumerate(range(0, n, time_hop)):
        tau_max = min(max_lag, t, n - 1 - t)
        if tau_max < 1:
            continue
        taus = np.arange(-tau_max, tau_max + 1)
        r = z[t + taus] * np.conj(z[t - taus])
        spec = np.fft.fftshift(np.fft.fft(r, n=n_freq))
        wvd[:, ti] = np.abs(spec) ** 2
        times[ti] = t / sr_out

    freqs = np.fft.fftshift(np.fft.fftfreq(n_freq, d=1.0 / sr_out))
    keep = (freqs >= 0.0) & (freqs <= fmax_hz)
    wvd = wvd[keep]
    freqs = freqs[keep]

    return WvdResult(
        times=times,
        freqs=freqs,
        power=wvd,
        sr=sr_out,
        segment_start_s=start / float(sr),
        segment_duration_s=seg_dur,
    )


def plot_wvd_specg_b64(
    wvd: WvdResult,
    *,
    fmax_hz: float,
    title: str,
    t_cpa: float | None = None,
) -> str:
    power_db = librosa.power_to_db(wvd.power, ref=np.max)
    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    mesh = ax.pcolormesh(wvd.times, wvd.freqs, power_db, shading="auto", cmap="magma")
    ax.set_xlabel("Time (s)", color="#aab0c0", fontsize=8)
    ax.set_ylabel("Hz", color="#aab0c0", fontsize=8)
    ax.set_ylim(0, fmax_hz)
    ax.set_title(
        f"{title}\n(cross-terms expected on multi-tone audio)",
        color="#fbbf24",
        fontsize=9,
        pad=6,
    )
    ax.tick_params(colors="#aab0c0", labelsize=7)
    if t_cpa is not None and np.isfinite(t_cpa):
        t_rel = t_cpa - wvd.segment_start_s
        if 0.0 <= t_rel <= wvd.segment_duration_s:
            ax.axvline(t_rel, color="#fbbf24", linestyle="--", linewidth=1.2, alpha=0.85)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _fig_to_b64(fig)


def build_experimental_comparison(
    y: np.ndarray,
    sr: int,
    *,
    fmax_hz: float,
    stft_params: StftParams,
    t_cpa: float | None = None,
) -> ExperimentalComparison:
    labels: list[tuple[str, str]] = [
        ("Mode", "Experimental TF compare"),
        ("Analysis sr", f"{sr} Hz"),
        ("STFT n_fft", str(stft_params.n_fft)),
        ("STFT hop", str(stft_params.hop_length)),
        ("fmax", f"{fmax_hz:.0f} Hz"),
        ("WVD segment", f"≤ {WVD_MAX_DURATION_S:.0f} s @ {WVD_ANALYSIS_SR} Hz"),
        ("Note", "WVD uses independent dB scale; cross-terms on harmonics"),
    ]

    try:
        stft_img = plot_stft_specg_b64(
            y,
            sr,
            stft_params,
            fmax_hz=fmax_hz,
            title="STFT",
            t_cpa=t_cpa,
        )
        atoms = compute_reassigned_atoms(
            y,
            sr,
            n_fft=stft_params.n_fft,
            hop_length=stft_params.hop_length,
            window=stft_params.window,
        )
        reassigned_img = plot_reassigned_specg_b64(
            atoms,
            fmax_hz=fmax_hz,
            title="Reassigned STFT",
            t_cpa=t_cpa,
        )
        wvd = compute_wvd(y, sr, fmax_hz=fmax_hz)
        wvd_img = plot_wvd_specg_b64(wvd, fmax_hz=fmax_hz, title="Wigner–Ville (pseudo)", t_cpa=t_cpa)
        labels.append(("WVD segment start", f"{wvd.segment_start_s:.2f} s"))
        return ExperimentalComparison(
            stft_image=stft_img,
            reassigned_image=reassigned_img,
            wvd_image=wvd_img,
            labels=labels,
        )
    except Exception as exc:
        msg = str(exc)
        err_img = _error_panel_b64("Experimental TF failed", msg)
        return ExperimentalComparison(
            stft_image=err_img,
            reassigned_image=err_img,
            wvd_image=err_img,
            labels=labels + [("Error", msg[:120])],
            error=msg,
        )
