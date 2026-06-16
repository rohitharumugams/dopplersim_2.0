"""Spectrogram Explorer — authoritative spectrogram implementation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pywt

from doppler_sim.spec_panel_style import (
    SPECG_PANEL_DPI,
    SPECG_PANEL_FIGSIZE,
    add_spec_colorbar,
    spec_panel_subplots,
    spec_panel_to_b64,
)
from doppler_sim.tf.analysis import (
    ReassignedAtoms,
    StftParams,
    atoms_summary,
    compute_reassigned_atoms,
    plot_reassigned_specg_b64,
    plot_stft_reassigned_comparison_b64,
)


SPECG_SR = 22050
SPECG_MAX_DURATION_S = 90.0
SPECG_ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SPECG_DEFAULT_FMAX_HZ = 1250.0
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
    return spec_panel_to_b64(fig)


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
    fig, ax = spec_panel_subplots()
    ax.set_facecolor("#0f1117")
    mesh = None
    if y_axis == "chroma":
        mesh = librosa.display.specshow(
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
        mesh = librosa.display.specshow(
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
        mesh = librosa.display.specshow(
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
    if mesh is not None:
        add_spec_colorbar(fig, ax, mesh)
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
    fig, ax = spec_panel_subplots()
    vmax = float(np.max(S))
    if vmax <= 0.0 or not np.isfinite(vmax):
        S_db = librosa.power_to_db(np.maximum(S, 1e-10), ref=1.0)
        img_mesh = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=p.hop_length,
            x_axis="time",
            y_axis="mel",
            fmax=fmax_hz,
            ax=ax,
            cmap="magma",
        )
    else:
        positive = S[S > 0]
        vmin = max(float(np.percentile(positive, 5)) if positive.size else vmax * 1e-4, vmax * 1e-5)
        img_mesh = librosa.display.specshow(
            S,
            sr=sr,
            hop_length=p.hop_length,
            x_axis="time",
            y_axis="mel",
            fmax=fmax_hz,
            ax=ax,
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="#aab0c0", labelsize=7)
    add_spec_colorbar(fig, ax, img_mesh)
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
            ("Scale", "Power (log color mapping)"),
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
    fig, ax = spec_panel_subplots()
    mesh = librosa.display.specshow(
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
    add_spec_colorbar(fig, ax, mesh)
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

    fig, ax = spec_panel_subplots()
    mesh = ax.pcolormesh(times, cqt_freqs, C_db, shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_ylim(max(cqt_freqs.min(), 20.0), fmax_hz)
    ax.set_xlabel("Time (s)", color="#aab0c0", fontsize=7)
    ax.set_ylabel("Hz", color="#aab0c0", fontsize=7)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="#aab0c0", labelsize=7)
    add_spec_colorbar(fig, ax, mesh)
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

    fig, ax = spec_panel_subplots()
    extent = [0, len(y) / sr, freqs_hz[-1], freqs_hz[0]]
    mesh = ax.imshow(power_db, aspect="auto", origin="upper", extent=extent, cmap="magma")
    ax.set_xlabel("Time (s)", color="#aab0c0", fontsize=7)
    ax.set_ylabel("Hz", color="#aab0c0", fontsize=7)
    specg_style_hz_ax(ax, title, fmax_hz)
    add_spec_colorbar(fig, ax, mesh)
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
    atoms = compute_reassigned_atoms(
        y,
        sr,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        window=p.window,
    )
    img = plot_reassigned_specg_b64(atoms, fmax_hz=fmax_hz, title=title)
    return specg_panel(
        title,
        img,
        sr,
        fmax_hz,
        [
            ("Transform", "Reassigned STFT (Auger–Flandrin)"),
            ("n_fft", str(p.n_fft)),
            ("hop_length", str(p.hop_length)),
            ("Window", specg_window_label(p.window)),
            ("fill_nan", "true"),
            ("Benefit", "Sub-bin frequency resolution"),
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

    fig, ax = spec_panel_subplots()
    Tx_db = librosa.power_to_db(np.abs(Tx) ** 2, ref=np.max)
    extent = [0, len(y_ssq) / target_sr, ssq_freqs[-1], ssq_freqs[0]]
    mesh = ax.imshow(Tx_db, aspect="auto", origin="upper", extent=extent, cmap="magma")
    ax.set_xlabel("Time (s)", color="#aab0c0", fontsize=7)
    ax.set_ylabel("Hz", color="#aab0c0", fontsize=7)
    specg_style_hz_ax(ax, title, fmax_hz)
    add_spec_colorbar(fig, ax, mesh)
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
    fig, ax = spec_panel_subplots(ax_facecolor="#1a1020")
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
    return specg_fig_to_b64(fig)


def _tensor_from_stft_mag(S: np.ndarray) -> np.ndarray:
    if np.max(np.abs(S)) <= 0:
        return np.asarray(S, dtype=np.float32)
    return librosa.amplitude_to_db(S, ref=np.max).astype(np.float32)


def _tensor_stft(y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams) -> np.ndarray:
    p = analysis.stft
    S = np.abs(librosa.stft(y, n_fft=p.n_fft, hop_length=p.hop_length, window=p.window))
    return _tensor_from_stft_mag(S)


def _tensor_wideband(y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams) -> np.ndarray:
    p = analysis.wideband
    S = np.abs(librosa.stft(y, n_fft=p.n_fft, hop_length=p.hop_length, window=p.window))
    return _tensor_from_stft_mag(S)


def _tensor_narrowband(y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams) -> np.ndarray:
    p = analysis.narrowband
    S = np.abs(librosa.stft(y, n_fft=p.n_fft, hop_length=p.hop_length, window=p.window))
    return _tensor_from_stft_mag(S)


def _tensor_mel(y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams) -> np.ndarray:
    p = analysis.stft
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        window=p.window,
        n_mels=128,
        fmax=fmax_hz,
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def _tensor_log_mel(y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams) -> np.ndarray:
    return _tensor_mel(y, sr, fmax_hz, analysis)


def _tensor_cqt(y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams) -> np.ndarray:
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
    return librosa.amplitude_to_db(C, ref=np.max).astype(np.float32)


def _tensor_wavelet(y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams) -> np.ndarray:
    scales = np.geomspace(1, 128, num=96)
    coef, freqs_hz = pywt.cwt(y, scales, "morl", sampling_period=1.0 / sr)
    keep = freqs_hz <= fmax_hz
    coef = coef[keep]
    return librosa.power_to_db(np.abs(coef) ** 2, ref=np.max).astype(np.float32)


def _tensor_reassignment(y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams) -> np.ndarray:
    p = analysis.stft
    atoms = compute_reassigned_atoms(
        y,
        sr,
        n_fft=p.n_fft,
        hop_length=p.hop_length,
        window=p.window,
    )
    mags = np.maximum(atoms.mags, 0.0)
    if np.max(mags) <= 0:
        return mags.astype(np.float32)
    return librosa.power_to_db(mags, ref=np.max).astype(np.float32)


def _tensor_synchrosqueezed(y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams) -> np.ndarray:
    from ssqueezepy import ssq_cwt

    target_sr = min(sr, 11025)
    y_ssq = librosa.resample(y, orig_sr=sr, target_sr=target_sr) if sr != target_sr else y
    scales = np.geomspace(2, 128, 64)
    Tx, _, ssq_freqs, *_ = ssq_cwt(y_ssq, wavelet="morlet", fs=target_sr, scales=scales)
    keep = ssq_freqs <= fmax_hz
    Tx = Tx[keep]
    return librosa.power_to_db(np.abs(Tx) ** 2, ref=np.max).astype(np.float32)


def _tensor_chroma(y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams) -> np.ndarray:
    hop = 512
    fmin = librosa.note_to_hz("C1")
    n_octaves = max(1, int(np.floor(np.log2(fmax_hz / fmin))))
    C = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop,
        n_chroma=12,
        fmin=fmin,
        n_octaves=n_octaves,
        bins_per_octave=36,
    )
    return librosa.amplitude_to_db(C, ref=np.max).astype(np.float32)


@dataclass(frozen=True)
class SpecgTypeDef:
    key: str
    label: str
    default_enabled: bool
    build_panel: Callable[[np.ndarray, int, float, SpecgAnalysisParams], Dict[str, Any]]
    build_tensor: Callable[[np.ndarray, int, float, SpecgAnalysisParams], np.ndarray]


SPECG_TYPE_REGISTRY: Tuple[SpecgTypeDef, ...] = (
    SpecgTypeDef(
        "stft",
        "STFT",
        True,
        lambda y, sr, fmax_hz, analysis: spec_stft(y, sr, fmax_hz, analysis.stft),
        _tensor_stft,
    ),
    SpecgTypeDef(
        "wideband",
        "Wideband Spectrogram",
        False,
        lambda y, sr, fmax_hz, analysis: spec_wideband(
            y, sr, fmax_hz, analysis.wideband, analysis.wb_preset_label
        ),
        _tensor_wideband,
    ),
    SpecgTypeDef(
        "narrowband",
        "Narrowband",
        True,
        lambda y, sr, fmax_hz, analysis: spec_narrowband(
            y, sr, fmax_hz, analysis.narrowband, analysis.nb_preset_label
        ),
        _tensor_narrowband,
    ),
    SpecgTypeDef(
        "mel",
        "Mel Spectrogram",
        False,
        lambda y, sr, fmax_hz, analysis: spec_mel(y, sr, fmax_hz, analysis.stft),
        _tensor_mel,
    ),
    SpecgTypeDef(
        "log_mel",
        "Log-Mel Spectrogram",
        True,
        lambda y, sr, fmax_hz, analysis: spec_log_mel(y, sr, fmax_hz, analysis.stft),
        _tensor_log_mel,
    ),
    SpecgTypeDef(
        "cqt",
        "CQT",
        True,
        lambda y, sr, fmax_hz, analysis: spec_cqt(y, sr, fmax_hz),
        _tensor_cqt,
    ),
    SpecgTypeDef(
        "wavelet",
        "Wavelet Scalogram",
        False,
        lambda y, sr, fmax_hz, analysis: spec_wavelet(y, sr, fmax_hz),
        _tensor_wavelet,
    ),
    SpecgTypeDef(
        "reassignment",
        "Reassigned Spectrogram",
        True,
        lambda y, sr, fmax_hz, analysis: spec_reassigned(y, sr, fmax_hz, analysis.stft),
        _tensor_reassignment,
    ),
    SpecgTypeDef(
        "synchrosqueezed",
        "Synchrosqueezed",
        False,
        lambda y, sr, fmax_hz, analysis: spec_synchrosqueezed(y, sr, fmax_hz),
        _tensor_synchrosqueezed,
    ),
    SpecgTypeDef(
        "chroma",
        "Chromagram",
        False,
        lambda y, sr, fmax_hz, analysis: spec_chroma(y, sr, fmax_hz),
        _tensor_chroma,
    ),
)

SPECG_TYPE_BY_KEY = {item.key: item for item in SPECG_TYPE_REGISTRY}
DEFAULT_BATCH_SPEC_KEYS = [item.key for item in SPECG_TYPE_REGISTRY if item.default_enabled]
SPECG_TYPE_CHOICES = [(item.key, item.label, item.default_enabled) for item in SPECG_TYPE_REGISTRY]


def parse_batch_spec_types(form, fallback: List[str] | None = None) -> List[str]:
    selected = [
        item.key
        for item in SPECG_TYPE_REGISTRY
        if form.get(f"spec_{item.key}") in ("on", "1", "true", "yes")
    ]
    if selected:
        return selected
    return list(fallback or DEFAULT_BATCH_SPEC_KEYS)


def prepare_batch_spec_audio(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Resample/limit audio the same way Spectrogram Explorer loads clips."""
    if sr != SPECG_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SPECG_SR)
        sr = SPECG_SR
    max_samples = int(SPECG_MAX_DURATION_S * sr)
    if y.size > max_samples:
        y = y[:max_samples]
    return np.asarray(y, dtype=np.float32), int(sr)


def build_all_spectrograms(
    y: np.ndarray, sr: int, fmax_hz: float, analysis: SpecgAnalysisParams
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in SPECG_TYPE_REGISTRY:
        try:
            out.append(item.build_panel(y, sr, fmax_hz, analysis))
        except Exception as exc:
            out.append(
                {
                    "title": item.label,
                    "image": specg_error_placeholder(item.label, str(exc)),
                    "labels": [("Error", str(exc)[:80])],
                }
            )
    return out


def batch_spec_grid_layout(n: int) -> tuple[int, list[int]]:
    """Return (max columns, panels per row) for batch combined spectrogram mosaics."""
    if n <= 0:
        return 1, []
    if n == 1:
        return 1, [1]
    ncols = 2 if n % 2 == 0 else (3 if n > 3 else 3)
    rows: list[int] = []
    remaining = n
    while remaining > 0:
        if remaining <= ncols:
            if remaining == 1 and rows:
                rows[-1] -= 1
                rows.append(2)
            else:
                rows.append(remaining)
            break
        take = ncols
        if remaining - take == 1:
            take = ncols - 1
        rows.append(take)
        remaining -= take
    return ncols, rows


def _panel_image_array(panel: Dict[str, Any]) -> np.ndarray:
    return plt.imread(BytesIO(base64.b64decode(panel["image"])))


def _normalize_panel_image(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a panel to a common height/width for mosaic alignment."""
    th, tw = target_shape
    h, w = arr.shape[0], arr.shape[1]
    if h == th and w == tw:
        return arr
    try:
        from PIL import Image

        if arr.dtype != np.uint8:
            scaled = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            scaled = arr
        if scaled.ndim == 2:
            mode = "L"
        elif scaled.shape[2] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"
        img = Image.fromarray(scaled, mode=mode)
        resized = img.resize((tw, th), Image.Resampling.LANCZOS)
        out = np.asarray(resized)
        if arr.dtype != np.uint8:
            return out.astype(np.float32) / 255.0
        return out
    except Exception:
        y_idx = np.linspace(0, h - 1, th).astype(int)
        x_idx = np.linspace(0, w - 1, tw).astype(int)
        return arr[np.ix_(y_idx, x_idx)]


def save_batch_spec_panel_png(panel: Dict[str, Any], path: Path) -> None:
    """Save one spectrogram PNG (plot only, no metadata caption)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(panel["image"]))


def save_batch_combined_spectrogram(
    panels: List[Tuple[str, Dict[str, Any]]],
    sample_title: str,
    path: Path,
) -> None:
    """Save all selected spectrograms in one comparison image."""
    n = len(panels)
    if n == 0:
        return

    ncols, row_sizes = batch_spec_grid_layout(n)
    nrows = len(row_sizes)
    panel_arrays = [_panel_image_array(panel) for _, panel in panels]
    target_h = max(arr.shape[0] for arr in panel_arrays)
    target_w = max(arr.shape[1] for arr in panel_arrays)
    panel_arrays = [_normalize_panel_image(arr, (target_h, target_w)) for arr in panel_arrays]

    fig = plt.figure(
        figsize=(SPECG_PANEL_FIGSIZE[0] * ncols, SPECG_PANEL_FIGSIZE[1] * nrows + 0.55),
        facecolor="#0f1117",
    )
    fig.suptitle(sample_title, color="white", fontsize=13, fontweight="bold", y=0.995)

    outer = fig.add_gridspec(
        nrows,
        ncols,
        hspace=0.22,
        wspace=0.18,
        top=0.93,
        bottom=0.03,
        left=0.04,
        right=0.98,
    )

    panel_idx = 0
    for row_i, count in enumerate(row_sizes):
        for col_i in range(count):
            panel_idx += 1
            ax = fig.add_subplot(outer[row_i, col_i])
            ax.imshow(panel_arrays[panel_idx - 1], aspect="auto")
            ax.set_axis_off()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=SPECG_PANEL_DPI, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)


def export_batch_spectrograms(
    y: np.ndarray,
    sr: int,
    fmax_hz: float,
    analysis: SpecgAnalysisParams,
    selected_keys: List[str],
    spec_dir: Path,
    meta_dir: Path,
    *,
    sample_title: str | None = None,
) -> List[str]:
    """Export PNG + NPY for selected spectrogram types using Explorer pipeline."""
    spec_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    exported: List[str] = []
    combined_panels: List[Tuple[str, Dict[str, Any]]] = []
    for key in selected_keys:
        item = SPECG_TYPE_BY_KEY.get(key)
        if item is None:
            continue
        panel = item.build_panel(y, sr, fmax_hz, analysis)
        png_path = spec_dir / f"{key}.png"
        save_batch_spec_panel_png(panel, png_path)
        tensor = item.build_tensor(y, sr, fmax_hz, analysis)
        np.save(meta_dir / f"{key}.npy", tensor)
        exported.append(key)
        combined_panels.append((item.label, panel))

    if combined_panels and sample_title:
        save_batch_combined_spectrogram(
            combined_panels,
            sample_title,
            spec_dir / "combined.png",
        )
    return exported


def specg_build_reassignment_extras(
    y: np.ndarray,
    sr: int,
    fmax_hz: float,
    analysis: SpecgAnalysisParams,
    t_cpa: float | None,
    save_atoms: Callable[[ReassignedAtoms], str | None],
) -> tuple[dict[str, str], dict[str, Any], str | None]:
    stft_params = StftParams(
        n_fft=analysis.stft.n_fft,
        hop_length=analysis.stft.hop_length,
        window=analysis.stft.window,
    )
    stft_img, reassigned_img, _ = plot_stft_reassigned_comparison_b64(
        y,
        sr,
        stft_params,
        fmax_hz=fmax_hz,
        t_cpa=t_cpa,
    )
    atoms = compute_reassigned_atoms(
        y,
        sr,
        n_fft=analysis.stft.n_fft,
        hop_length=analysis.stft.hop_length,
        window=analysis.stft.window,
    )
    summary = atoms_summary(atoms, fmax_hz=fmax_hz)
    atoms_id = save_atoms(atoms)
    comparison = {
        "stft_compare": stft_img,
        "reassigned_compare": reassigned_img,
    }
    return comparison, summary, atoms_id