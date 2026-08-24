#!/usr/bin/env python3
"""Verify Phase 1 kinematics, A/s frame alignment, and acoustic CPA regression.

Checks:
  1. Instantaneous centerline s(t) and derived CPA on sample / STFT grids.
  2. render_pass_by loudness peak tracks t_CPA₂ (1/R), not a synthetic upload
     envelope peak — guards against the old envelope-warp regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doppler_sim.application import (  # noqa: E402
    OUTPUT_SR,
    SPEED_OF_SOUND,
    RenderParams,
    compute_propagation_quantities,
    render_pass_by,
)
from doppler_sim.batch.phase1_state import (  # noqa: E402
    build_phase1_arrays,
    derived_from_state,
    straight_passby_state,
)
from doppler_sim.specg.explorer import SPECG_DEFAULT_ANALYSIS, SPECG_SR  # noqa: E402


def _assert_close(name: str, a: float, b: float, tol: float) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{name}: {a} vs {b} (tol={tol})")


def _rms_peak_time(y: np.ndarray, sr: int, *, hop: int = 2048) -> float:
    best_e = -1.0
    best_t = 0.0
    for i in range(0, max(len(y) - hop, 1), hop):
        seg = y[i : i + hop]
        e = float(np.mean(seg * seg))
        t = (i + hop / 2.0) / float(sr)
        if e > best_e:
            best_e = e
            best_t = t
    return best_t


def _verify_kinematics() -> dict[str, float]:
    v = 15.0
    h = 8.0
    t_cpa = 2.4
    t_out = 6.0
    wav_sr = 44100
    n_wav = int(np.ceil(t_out * wav_sr))
    n_spec = int(np.ceil(t_out * SPECG_SR))
    hop = SPECG_DEFAULT_ANALYSIS.stft.hop_length
    n_fft = SPECG_DEFAULT_ANALYSIS.stft.n_fft

    bundle = build_phase1_arrays(
        speed_mps=v,
        cpa_distance_m=h,
        cpa_time_sec=t_cpa,
        n_wav_samples=n_wav,
        wav_sr=wav_sr,
        n_spec_samples=n_spec,
        spec_sr=SPECG_SR,
        hop_length=hop,
        n_fft=n_fft,
    )

    state = bundle["state"].astype(np.float64)
    t = bundle["state_times"].astype(np.float64)
    x, vx, y, vy = state.T

    assert np.allclose(vx, v), "vx must equal constant speed"
    assert np.allclose(vy, 0.0), "vy must be 0"
    assert np.allclose(y, h), "y must equal CPA distance h"
    assert np.allclose(x, v * (t - t_cpa), atol=1e-4), "x = v*(t - t_cpa)"

    derived = bundle["derived_wav"]
    assert np.allclose(derived["speed_mps"], abs(v), atol=1e-5)

    d64 = derived_from_state(
        straight_passby_state(t, speed_mps=v, cpa_distance_m=h, cpa_time_sec=t_cpa),
        t,
    )
    pre = t < t_cpa
    post = t > t_cpa
    assert np.all(d64["radial_velocity_mps"][pre] < 0)
    assert np.all(d64["radial_velocity_mps"][post] > 0)

    _assert_close("cpa_time (sample grid)", float(derived["cpa_time_sec"][0]), t_cpa, 1e-9)
    _assert_close("cpa_distance", float(derived["cpa_distance_m"][0]), h, 1e-9)
    assert int(derived["direction"][0]) == 1

    state_neg = straight_passby_state(
        t[:10],
        speed_mps=-v,
        cpa_distance_m=h,
        cpa_time_sec=t_cpa,
    )
    d_neg = derived_from_state(state_neg, t[:10])
    assert int(d_neg["direction"][0]) == -1

    assert bundle["state_frames"].shape == (bundle["n_frames"], 4)
    assert bundle["frame_times"].shape == (bundle["n_frames"],)
    hop_s = hop / float(SPECG_SR)
    _assert_close(
        "cpa_time (frame grid)",
        float(bundle["derived_frames"]["cpa_time_sec"][0]),
        t_cpa,
        hop_s + 1e-9,
    )

    print("phase1_state OK")
    print(
        f"  wav samples={n_wav}  frames={bundle['n_frames']}  "
        f"spec_sr={SPECG_SR} hop={hop} n_fft={n_fft}"
    )
    print(
        f"  derived CPA time={float(bundle['derived_frames']['cpa_time_sec'][0]):.4f}s "
        f"(plan {t_cpa:.4f}s)  CPA dist={float(bundle['derived_frames']['cpa_distance_m'][0]):.4f}m"
    )
    return {"v": v, "h": h, "t_cpa": t_cpa, "hop_s": hop_s}


def _verify_acoustic_cpa() -> None:
    """Loudness must track t_CPA₂ under 1/R, even if analysis t_CPA₁ differs."""
    t_cpa2 = 9.0
    t_out = 10.0
    h2 = 8.0
    v2 = 15.0
    # Deliberately mismatched "upload CPA" analysis time — must not pin loudness.
    t_cpa1 = 5.2

    freqs = np.linspace(0.0, 10_000.0, 513)
    psd = np.ones_like(freqs)
    params = RenderParams(
        v1=v2,
        h1=h2,
        t_cpa1=t_cpa1,
        vehicle_length=4.5,
        num_emitters=1,
        v2=v2,
        h2=h2,
        t_cpa2=t_cpa2,
        t_out=t_out,
    )
    generated, quantities, _ = render_pass_by(freqs, psd, params)
    amp_peak = _rms_peak_time(generated, OUTPUT_SR)
    t_obs = np.arange(len(generated), dtype=np.float64) / float(OUTPUT_SR)
    min_r_t = float(t_obs[int(np.argmin(quantities["R"]))])

    # Deterministic 1/R peak == argmin R ≈ t_cpa + h/c.
    q = compute_propagation_quantities(t_obs, v2, h2, 0.0, t_cpa2)
    inv_r = np.zeros_like(q["R"])
    valid = np.isfinite(q["R"]) & (q["R"] > 0.0)
    inv_r[valid] = 1.0 / np.maximum(q["R"][valid], 1e-12)
    max_inv_r_t = float(t_obs[int(np.argmax(inv_r))])
    _assert_close("max(1/R) vs min R", max_inv_r_t, min_r_t, 1e-6)

    hop_s = 2048 / float(OUTPUT_SR)
    # Allow retarded shift + one RMS hop of stochastic carrier jitter.
    tol = (h2 / SPEED_OF_SOUND) + hop_s + 0.15
    if abs(amp_peak - t_cpa2) > tol:
        raise AssertionError(
            f"acoustic CPA regression: RMS peak at {amp_peak:.3f}s, "
            f"expected near t_cpa2={t_cpa2:.3f}s (tol={tol:.3f}s). "
            f"minR={min_r_t:.3f}s t_cpa1={t_cpa1} — loudness must not follow upload CPA."
        )
    # Also ensure we did not collapse back to the mismatched t_cpa1.
    if abs(amp_peak - t_cpa1) < 0.5 and abs(amp_peak - t_cpa2) > 1.0:
        raise AssertionError(
            f"acoustic CPA still pinned near t_cpa1={t_cpa1:.3f}s (peak={amp_peak:.3f}s)"
        )

    print("acoustic_cpa OK")
    print(
        f"  t_cpa2={t_cpa2:.3f}s  RMS peak={amp_peak:.3f}s  "
        f"minR={min_r_t:.3f}s  |peak-plan|={abs(amp_peak - t_cpa2):.3f}s  "
        f"(t_cpa1 decoy={t_cpa1:.3f}s)"
    )


def main() -> int:
    _verify_kinematics()
    _verify_acoustic_cpa()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
