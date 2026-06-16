"""Time–frequency reassignment helpers (Auger–Flandrin via librosa).

Visualization-only — not used by the Pass-By synthesis pipeline.

Parameter contexts:
  - Spectrogram Explorer: typically sr=22050, n_fft/hop from SpecgStftParams.
  - Pass-By diagnostic plots: sr=44100 (or upload SR), n_fft=4096, hop=512.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np

from doppler_sim.spec_panel_style import add_spec_colorbar, spec_panel_subplots, spec_panel_to_b64

__all__ = [
    "ReassignedAtoms",
    "StftParams",
    "atoms_summary",
    "atoms_to_npz_bytes",
    "compute_reassigned_atoms",
    "compute_stft_power",
    "filter_atoms",
    "plot_reassigned_passby",
    "plot_reassigned_specg_b64",
    "plot_stft_reassigned_comparison_b64",
    "plot_stft_specg_b64",
]


@dataclass
class StftParams:
    n_fft: int
    hop_length: int
    window: str = "hann"


@dataclass
class ReassignedAtoms:
    """Relocated STFT energy on the librosa reassignment grid."""

    times: np.ndarray
    freqs: np.ndarray
    mags: np.ndarray
    sr: int
    n_fft: int
    hop_length: int
    window: str

    def metadata(self) -> dict[str, Any]:
        return {
            "sr": self.sr,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "window": self.window,
        }


def compute_reassigned_atoms(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int,
    hop_length: int,
    window: str = "hann",
    fill_nan: bool = True,
) -> ReassignedAtoms:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*where.*used without.*out.*",
            category=UserWarning,
        )
        freqs, times, mags = librosa.reassigned_spectrogram(
            y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            fill_nan=fill_nan,
        )
    return ReassignedAtoms(
        times=np.asarray(times),
        freqs=np.asarray(freqs),
        mags=np.asarray(mags),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
    )


def compute_stft_power(
    y: np.ndarray,
    sr: int,
    params: StftParams,
    *,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stft = librosa.stft(
        y,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        window=params.window,
        center=center,
    )
    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=params.hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=params.n_fft)
    power = np.abs(stft) ** 2
    return freqs, times, power


def filter_atoms(
    atoms: ReassignedAtoms,
    *,
    fmax_hz: float,
    min_mag_db: float | None = None,
) -> ReassignedAtoms:
    freq_mask = atoms.freqs <= fmax_hz
    mags = atoms.mags.copy()
    mags[~freq_mask] = 0.0

    if min_mag_db is not None:
        ref = float(np.nanmax(mags)) if np.any(np.isfinite(mags)) else 1.0
        if ref <= 0.0:
            ref = 1.0
        floor = ref * (10.0 ** (min_mag_db / 10.0))
        mags[mags < floor] = 0.0

    return ReassignedAtoms(
        times=atoms.times,
        freqs=atoms.freqs,
        mags=mags,
        sr=atoms.sr,
        n_fft=atoms.n_fft,
        hop_length=atoms.hop_length,
        window=atoms.window,
    )


def atoms_summary(
    atoms: ReassignedAtoms,
    *,
    fmax_hz: float,
    min_mag_db: float = -40.0,
) -> dict[str, Any]:
    filtered = filter_atoms(atoms, fmax_hz=fmax_hz, min_mag_db=min_mag_db)
    mags = filtered.mags
    total_bins = int(mags.size)
    ref = float(np.nanmax(mags)) if np.any(np.isfinite(mags)) else 0.0
    if ref > 0.0:
        floor = ref * (10.0 ** (min_mag_db / 10.0))
        above = int(np.sum((mags >= floor) & np.isfinite(mags)))
        active = mags[(mags >= floor) & np.isfinite(mags)]
        f_min = float(np.min(active)) if active.size else 0.0
        f_max = float(np.max(active)) if active.size else 0.0
        # Relocated frequency extent among active bins (use freq axis rows)
        row_energy = np.sum(mags, axis=1)
        active_rows = row_energy >= floor
        if np.any(active_rows):
            f_axis = filtered.freqs
            if f_axis.ndim == 1:
                idx = np.where(active_rows)[0]
                freq_min_hz = float(f_axis[idx[0]])
                freq_max_hz = float(f_axis[idx[-1]])
            else:
                active_freqs = f_axis[active_rows]
                freq_min_hz = float(np.nanmin(active_freqs))
                freq_max_hz = float(np.nanmax(active_freqs))
        else:
            freq_min_hz = 0.0
            freq_max_hz = 0.0
    else:
        above = 0
        f_min = f_max = 0.0
        freq_min_hz = freq_max_hz = 0.0

    return {
        "total_bins": total_bins,
        "above_threshold": above,
        "threshold_db": min_mag_db,
        "peak_mag": ref,
        "freq_min_hz": freq_min_hz,
        "freq_max_hz": freq_max_hz,
    }


def atoms_to_npz_bytes(atoms: ReassignedAtoms) -> bytes:
    buf = BytesIO()
    np.savez_compressed(
        buf,
        times=atoms.times,
        freqs=atoms.freqs,
        mags=atoms.mags,
        sr=np.array([atoms.sr]),
        n_fft=np.array([atoms.n_fft]),
        hop_length=np.array([atoms.hop_length]),
        window=np.array([atoms.window]),
    )
    buf.seek(0)
    return buf.read()


def _add_t_cpa_line(ax: plt.Axes, t_cpa: float | None, *, style: str) -> None:
    if t_cpa is None or not np.isfinite(t_cpa):
        return
    color = "#fbbf24" if style == "explorer" else "#dc2626"
    ax.axvline(t_cpa, color=color, linestyle="--", linewidth=1.2, alpha=0.85, label="t_CPA")


def _power_to_db(power: np.ndarray, *, ref: Any = np.max) -> np.ndarray:
    """Convert power spectrogram to dB without librosa/numpy `where` warnings."""
    p = np.asarray(power, dtype=np.float64)
    if ref is None:
        ref_val = float(np.nanmax(p))
    elif callable(ref):
        ref_val = float(ref(p))
    else:
        ref_val = float(ref)
    if not np.isfinite(ref_val) or ref_val <= 0.0:
        ref_val = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.maximum(p, 0.0) / ref_val)
    db[~np.isfinite(p)] = np.nan
    return db


def _stft_grid_axes(atoms: ReassignedAtoms) -> tuple[np.ndarray, np.ndarray]:
    """Regular monotonic STFT axes for plotting reassigned magnitudes."""
    n_freq, n_time = atoms.mags.shape
    times = librosa.frames_to_time(
        np.arange(n_time),
        sr=atoms.sr,
        hop_length=atoms.hop_length,
    )
    freqs = librosa.fft_frequencies(sr=atoms.sr, n_fft=atoms.n_fft)
    if freqs.shape[0] != n_freq:
        freqs = freqs[:n_freq]
    return times, freqs


def _fig_to_b64(fig: plt.Figure, dpi: int = 100) -> str:
    return spec_panel_to_b64(fig)


def plot_stft_specg_b64(
    y: np.ndarray,
    sr: int,
    params: StftParams,
    *,
    fmax_hz: float,
    title: str,
    t_cpa: float | None = None,
    shared_vmax_db: float | None = None,
) -> str:
    freqs, times, power = compute_stft_power(y, sr, params)
    keep = freqs <= fmax_hz
    power = power[keep]
    freqs = freqs[keep]
    power_db = _power_to_db(power, ref=np.max)
    if shared_vmax_db is not None:
        vmin = shared_vmax_db - 80.0
        vmax = shared_vmax_db
    else:
        vmin = vmax = None

    fig, ax = spec_panel_subplots()
    mesh = ax.pcolormesh(
        times,
        freqs,
        power_db,
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Time (s)", color="#aab0c0", fontsize=8)
    ax.set_ylabel("Hz", color="#aab0c0", fontsize=8)
    ax.set_ylim(0, fmax_hz)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="#aab0c0", labelsize=7)
    _add_t_cpa_line(ax, t_cpa, style="explorer")
    add_spec_colorbar(fig, ax, mesh)
    return _fig_to_b64(fig)


def plot_reassigned_specg_b64(
    atoms: ReassignedAtoms,
    *,
    fmax_hz: float,
    title: str,
    t_cpa: float | None = None,
    shared_vmax_db: float | None = None,
) -> str:
    mags_db = _power_to_db(atoms.mags, ref=np.max)
    times, freqs = _stft_grid_axes(atoms)
    if shared_vmax_db is not None:
        vmin = shared_vmax_db - 80.0
        vmax = shared_vmax_db
    else:
        vmin = vmax = None

    fig, ax = spec_panel_subplots()
    mesh = ax.pcolormesh(
        times,
        freqs,
        mags_db,
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Time (s)", color="#aab0c0", fontsize=8)
    ax.set_ylabel("Hz", color="#aab0c0", fontsize=8)
    ax.set_ylim(0, fmax_hz)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="#aab0c0", labelsize=7)
    _add_t_cpa_line(ax, t_cpa, style="explorer")
    add_spec_colorbar(fig, ax, mesh)
    return _fig_to_b64(fig)


def plot_stft_reassigned_comparison_b64(
    y: np.ndarray,
    sr: int,
    params: StftParams,
    *,
    fmax_hz: float,
    t_cpa: float | None = None,
) -> tuple[str, str, float]:
    """Side-by-side STFT vs reassigned with shared dB reference."""
    freqs, times, power = compute_stft_power(y, sr, params)
    keep = freqs <= fmax_hz
    power_db = _power_to_db(power[keep], ref=np.max)
    atoms = compute_reassigned_atoms(
        y,
        sr,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        window=params.window,
    )
    reassigned_db = _power_to_db(atoms.mags, ref=np.max)
    shared_vmax = float(max(np.nanmax(power_db), np.nanmax(reassigned_db)))

    stft_img = plot_stft_specg_b64(
        y,
        sr,
        params,
        fmax_hz=fmax_hz,
        title="STFT (comparison)",
        t_cpa=t_cpa,
        shared_vmax_db=shared_vmax,
    )
    reassigned_img = plot_reassigned_specg_b64(
        atoms,
        fmax_hz=fmax_hz,
        title="Reassigned STFT (comparison)",
        t_cpa=t_cpa,
        shared_vmax_db=shared_vmax,
    )
    return stft_img, reassigned_img, shared_vmax


def plot_reassigned_passby(
    audio: np.ndarray,
    sr: int,
    title: str,
    filename: str,
    plot_dir: Path,
    *,
    n_fft: int,
    hop_length: int,
    window: str = "hann",
    freq_max: float,
    t_cpa: float | None = None,
) -> str:
    """Light-theme reassigned plot for Pass-By diagnostic exports."""
    atoms = compute_reassigned_atoms(
        audio,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
    )
    plt.figure(figsize=(10, 4))
    mags_db = _power_to_db(atoms.mags, ref=np.max)
    times, freqs = _stft_grid_axes(atoms)
    plt.pcolormesh(times, freqs, mags_db, shading="auto", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, freq_max)
    if t_cpa is not None and np.isfinite(t_cpa):
        plt.axvline(t_cpa, color="#dc2626", linestyle="--", linewidth=1.2, label=f"t_CPA = {t_cpa:.2f} s")
        plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / filename
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return filename
