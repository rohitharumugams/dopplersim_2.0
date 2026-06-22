"""
Forward (generic) source q(t′) — no VS13 target clip.

Uses vehicle library audio as stationary emission buffers (same core as doppler_4dot0),
then physics propagation produces Doppler sweeps. No synthetic sine comb, no CPA swell on q.
"""

from __future__ import annotations

import os

import numpy as np

from doppler_4dot0.audio.body_source import segment_broadband_buffer
from doppler_4dot0.audio.source import (
    emission_buffer_from_recording,
    flatten_rms,
    tonal_harmonic_emphasis,
)
from doppler_5dot0.config import VEHICLE_SOUNDS_DIR
from doppler_5dot0.sources.extract import (
    _apply_gentle_emphasis,
    _circular_offset,
    pad_for_doppler,
)


def load_library_audio(audio_file: str, sr: int) -> np.ndarray:
    path = os.path.join(VEHICLE_SOUNDS_DIR, audio_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Vehicle library audio not found: {path}')
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return np.asarray(y, dtype=np.float64).ravel()


def _library_stationary_timbre(library: np.ndarray, sr: int) -> np.ndarray:
    """Harmonic timbre from static rev clip — full spectrum, no band splitting."""
    flat = flatten_rms(library, sr)
    return tonal_harmonic_emphasis(flat, sr, gain_db=5.0)


def _tile_stationary_q(x: np.ndarray, n_samples: int) -> np.ndarray:
    """Loop stationary timbre; level stays constant in emission time."""
    buf = np.asarray(x, dtype=np.float32).ravel()
    need = max(n_samples, 4096)
    if len(buf) >= need:
        out = buf[:need].copy()
    else:
        reps = int(np.ceil(need / max(len(buf), 1)))
        out = np.tile(buf, reps)[:need].astype(np.float32)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-8:
        out = out / peak
    return pad_for_doppler(out, n_samples).astype(np.float32)


def library_emission_buffer(library: np.ndarray, sr: int, n_samples: int) -> np.ndarray:
    """
    Stationary q(t′) from vehicle library — no pass-by envelope (physics handles that).
    """
    return emission_buffer_from_recording(
        library, sr, n_samples,
        retain_level_envelope=False,
        cpa_time_s=None,
    )


def library_spectrum_shape(library: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
    flat = flatten_rms(library, sr)
    x = np.asarray(flat, dtype=np.float64).ravel()
    if len(x) < 256:
        x = np.pad(x, (0, max(0, 256 - len(x))))
    spec = np.abs(np.fft.rfft(x, n=n_fft))
    spec[0] = 0.0
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    hf = np.exp(-0.5 * ((np.log10(np.maximum(freqs, 1.0) / 900.0)) / 0.42) ** 2)
    hf[freqs < 200.0] = 0.0
    spec = spec + 0.40 * hf * float(np.max(spec) + 1e-12)
    peak = float(np.max(spec))
    return (spec / peak if peak > 1e-12 else spec).astype(np.float64)


def extract_tonal_q_forward(
    library: np.ndarray,
    sr: int,
    n_samples: int,
    *,
    source_id: str,
    harmonic_base: np.ndarray | None = None,
) -> np.ndarray:
    """
    Full-harmonic q from library; sources differ by mild tilt + circular offset only.
    """
    if harmonic_base is None:
        harmonic_base = _tile_stationary_q(_library_stationary_timbre(library, sr), n_samples)

    base = np.asarray(harmonic_base, dtype=np.float32)

    if source_id == 'engine':
        x = _apply_gentle_emphasis(
            base, sr, pivot_hz=480.0, slope_db_per_oct=-0.08,
            sub_boost_db=1.0, hf_keep_db=0.5,
        )
        x = _circular_offset(x, 0)
    elif source_id == 'intake':
        x = _apply_gentle_emphasis(
            base, sr, pivot_hz=800.0, slope_db_per_oct=0.10,
            sub_boost_db=-0.3, hf_keep_db=1.0,
        )
        x = _circular_offset(x, sr // 17 + 113)
    elif source_id == 'exhaust':
        x = _apply_gentle_emphasis(
            base, sr, pivot_hz=400.0, slope_db_per_oct=-0.04,
            sub_boost_db=1.5, hf_keep_db=0.0,
        )
        x = _circular_offset(x, sr // 11 + 307)
    else:
        x = base

    peak = float(np.max(np.abs(x)))
    if peak > 1e-8:
        x = x / peak
    return pad_for_doppler(np.asarray(x, dtype=np.float32), n_samples)


def extract_all_tonal_q_forward(
    library: np.ndarray,
    sr: int,
    n_samples: int,
    *,
    cpa_time_s: float | None = None,
) -> dict[str, np.ndarray]:
    del cpa_time_s  # forward: no CPA content in q
    base = _tile_stationary_q(_library_stationary_timbre(library, sr), n_samples)
    return {
        sid: extract_tonal_q_forward(
            library, sr, n_samples, source_id=sid, harmonic_base=base,
        )
        for sid in ('engine', 'intake', 'exhaust')
    }


def _analytic_spectrum_shape(n_fft: int, sr: int, profile: str) -> np.ndarray:
    from dopplernet_corrected.audio.filters import _body_spectrum_shape, _tire_spectrum_shape

    n = max(n_fft * 2, 4096)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    unit = np.ones(len(freqs), dtype=np.float64)
    if profile == 'tire':
        spec = _tire_spectrum_shape(unit, freqs, sr)
    elif profile == 'aero':
        f = np.maximum(freqs, 1.0)
        spec = np.exp(-0.5 * ((np.log10(f / 1600.0)) / 0.38) ** 2)
        spec[f < 250.0] *= f[f < 250.0] / 250.0
    else:
        spec = _body_spectrum_shape(unit, freqs)
    spec[0] = 0.0
    peak = float(np.max(spec))
    return (spec / peak if peak > 1e-12 else spec).astype(np.float64)


def _blend_shapes(library_shape: np.ndarray, analytic: np.ndarray) -> np.ndarray:
    ls = np.asarray(library_shape, dtype=np.float64).ravel()
    an = np.asarray(analytic, dtype=np.float64).ravel()
    n = max(len(ls), len(an))
    if len(ls) != n:
        ls = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(ls)), ls)
    if len(an) != n:
        an = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(an)), an)
    out = 0.55 * ls + 0.45 * an
    peak = float(np.max(out))
    return (out / peak if peak > 1e-12 else out).astype(np.float64)


def tire_q_forward(
    library: np.ndarray,
    sr: int,
    n: int,
    speed_mps: float,
    emitter_id: str,
) -> np.ndarray:
    from dopplernet_corrected.audio.filters import tire_emission_buffer

    syn = tire_emission_buffer(sr, n, speed_mps, emitter_id, lo_hz=80.0, hi_hz=4000.0)
    shape = _blend_shapes(
        library_spectrum_shape(library, sr, max(4096, n)),
        _analytic_spectrum_shape(max(4096, n), sr, 'tire'),
    )
    spec = np.fft.rfft(np.asarray(syn, dtype=np.float64))
    n_spec = len(spec)
    if len(shape) != n_spec:
        shape = np.interp(np.linspace(0, 1, n_spec), np.linspace(0, 1, len(shape)), shape)
    out = np.fft.irfft(spec * shape, n=n).astype(np.float64)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-8:
        out = out / peak * float(np.max(np.abs(syn)))
    return out[:n].astype(np.float32)


def body_q_forward(
    library: np.ndarray,
    sr: int,
    n: int,
    speed_mps: float,
    segment_id: str,
) -> np.ndarray:
    shape = _blend_shapes(
        library_spectrum_shape(library, sr, max(4096, n)),
        _analytic_spectrum_shape(max(4096, n), sr, 'body'),
    )
    return segment_broadband_buffer(
        segment_id, sr, n, speed_mps=speed_mps,
        spectrum_shape=shape, level=1.0, lo_hz=60.0, hi_hz=4200.0,
    )


def aero_q_forward(
    library: np.ndarray,
    sr: int,
    n: int,
    speed_mps: float,
    segment_id: str,
) -> np.ndarray:
    shape = _blend_shapes(
        library_spectrum_shape(library, sr, max(4096, n)),
        _analytic_spectrum_shape(max(4096, n), sr, 'aero'),
    )
    return segment_broadband_buffer(
        segment_id, sr, n, speed_mps=speed_mps,
        spectrum_shape=shape, level=1.0, lo_hz=350.0, hi_hz=5200.0,
    )


def bed_q_forward(library: np.ndarray, sr: int, n: int, speed_mps: float) -> np.ndarray:
    shape = library_spectrum_shape(library, sr, max(4096, n))
    return segment_broadband_buffer(
        'road_bed', sr, n, speed_mps=speed_mps,
        spectrum_shape=shape, level=0.45, lo_hz=40.0, hi_hz=3500.0,
    )
