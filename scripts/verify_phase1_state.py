#!/usr/bin/env python3
"""Verify Phase 1 state physics and A/s frame alignment (no batch render required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doppler_sim.batch.phase1_state import (  # noqa: E402
    build_phase1_arrays,
    derived_from_state,
    straight_passby_state,
)
from doppler_sim.specg.explorer import SPECG_DEFAULT_ANALYSIS, SPECG_SR  # noqa: E402


def _assert_close(name: str, a: float, b: float, tol: float) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{name}: {a} vs {b} (tol={tol})")


def main() -> int:
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

    # Recompute radial-velocity sign check on float64 kinematics.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
