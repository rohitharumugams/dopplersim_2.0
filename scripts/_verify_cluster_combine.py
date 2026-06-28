"""Verify tonal + broadband cluster combination in dopplersim_2.0 multisource."""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doppler_sim.application import RenderParams
from doppler_sim.multisource.bootstrap import ensure_vendor_on_path

ensure_vendor_on_path()

from doppler_5dot0.config import SAMPLE_RATE
from doppler_5dot0.inverse.synthesis import (
    _broadband_tracks,
    _component_tracks,
    infer_vs13_calibration,
)
from doppler_sim.multisource.synthesis import (
    _precompute_cache_retarget,
    _prepare_recording,
    render_params_from_doppler,
    synthesize_multisource_passby,
)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(x * x)))


def main() -> None:
    wav = (
        ROOT.parent
        / "dopplernet_corrected/data/vs13/KiaSportage/KiaSportage/KiaSportage_68.wav"
    )
    y, sr_in = librosa.load(wav, sr=None, mono=True)

    v_mps = 68 / 3.6
    params = RenderParams(
        v1=v_mps,
        h1=0.5,
        t_cpa1=4.96,
        vehicle_length=4.51,
        num_emitters=3,
        v2=v_mps,
        h2=0.5,
        t_cpa2=4.96,
        t_out=10.0,
    )
    ms = render_params_from_doppler(params)
    n = max(4, int(round(ms.t_out_s * SAMPLE_RATE)))
    real = _prepare_recording(y, sr_in, n)

    cache = _precompute_cache_retarget(
        real,
        extract_speed_mps=ms.v1_mps,
        extract_cpa_time_s=ms.t_cpa1_s,
        render_speed_mps=ms.v2_mps,
        render_cpa_time_s=ms.t_cpa2_s,
        render_dcpa_m=ms.h2_m,
        duration_s=ms.t_out_s,
        travel_sign=ms.travel_sign,
        sr=SAMPLE_RATE,
        car_length_m=ms.vehicle_length_m,
    )

    print("=== CACHE PRESSURE RMS (pre-calibration) ===")
    print(f"engine:  {rms(cache.engine_p):.6e}")
    print(f"intake:  {rms(cache.intake_p):.6e}")
    print(f"exhaust: {rms(cache.exhaust_p):.6e}")
    print(f"tires x4: {[f'{rms(t):.3e}' for t in cache.tires]}")
    print(f"body x5:  mean rms {np.mean([rms(b) for b in cache.body]):.3e}")
    print(f"aero x2:  mean rms {np.mean([rms(a) for a in cache.aero]):.3e}")
    print(f"bed:     {rms(cache.bed):.6e}")

    cal = infer_vs13_calibration(
        real,
        speed_mps=ms.v2_mps,
        duration_s=ms.t_out_s,
        cpa_time_s=ms.t_cpa2_s,
        sr=SAMPLE_RATE,
        dcpa_m=ms.h2_m,
        car_length_m=ms.vehicle_length_m,
        travel_sign=ms.travel_sign,
        cache=cache,
    )

    print("\n=== CALIBRATION GAINS ===")
    for k in (
        "engine_gain",
        "intake_gain",
        "exhaust_gain",
        "global_gain",
        "body_gain",
        "tire_gain",
        "aero_gain",
        "bed_gain",
    ):
        print(f"  {k}: {getattr(cal, k):.4f}")

    tracks = _component_tracks(cache, cal, SAMPLE_RATE)
    tonal = tracks["tonal"]
    bb = tracks["broadband"]
    comb = tracks["combined"]
    nmin = min(len(tonal), len(bb), len(comb))
    recon = tonal[:nmin] + bb[:nmin]
    err = float(np.max(np.abs(recon - comb[:nmin])))
    rel = err / (float(np.max(np.abs(comb[:nmin]))) + 1e-12)

    print("\n=== COMBINE IDENTITY: combined == tonal + broadband ===")
    print(f"max abs error: {err:.3e}")
    print(f"relative error: {rel:.3e}")
    print(f"PASS: {rel < 1e-10}")

    bb_parts = _broadband_tracks(cache, cal, SAMPLE_RATE)
    bb_recon = bb_parts["tires"] + bb_parts["body"] + bb_parts["aero"] + bb_parts["bed"]
    err_bb = float(np.max(np.abs(bb_recon[:nmin] - bb[:nmin])))
    print("\n=== BROADBAND IDENTITY: broadband == tires+body+aero+bed ===")
    print(f"max abs error: {err_bb:.3e}")
    print(f"PASS: {err_bb < 1e-10}")

    cpa_i = int(ms.t_cpa2_s * SAMPLE_RATE)
    win = slice(max(0, cpa_i - SAMPLE_RATE // 2), min(nmin, cpa_i + SAMPLE_RATE // 2))
    print("\n=== CPA WINDOW RMS (pre-level-match tracks) ===")
    for name in ("tonal", "broadband", "combined"):
        print(f"  {name}: {rms(tracks[name][win]):.6e}")
    print(
        f"  broadband/tonal RMS ratio: "
        f"{rms(tracks['broadband'][win]) / (rms(tracks['tonal'][win]) + 1e-12):.4f}"
    )

    combined_out, component_out, meta = synthesize_multisource_passby(y, sr_in, params)
    scaled_tonal = component_out["tonal"]
    scaled_bb = component_out["broadband"]
    n2 = min(len(scaled_tonal), len(scaled_bb), len(combined_out))
    scaled_sum = scaled_tonal[:n2] + scaled_bb[:n2]
    lf_residual = combined_out[:n2] - scaled_sum[:n2]

    print("\n=== FINAL WAV OUTPUT ===")
    print(f"level_scale: {cal.level_scale:.4f}")
    print(f"lf_bed_gain: {cal.lf_bed_gain:.4f}")
    print(f"scaled (tonal+bb) vs (final - LF) max err: {float(np.max(np.abs(scaled_sum[:n2] - (combined_out[:n2] - lf_residual[:n2])))):.3e}")
    print(f"LF bed RMS in final: {rms(lf_residual):.6e}")
    print(f"final combined RMS: {rms(combined_out):.6e}")
    print(f"scaled tonal+bb RMS: {rms(scaled_sum):.6e}")
    print(f"broadband share of tonal+bb energy: {(rms(scaled_bb)/rms(scaled_sum))**2*100:.1f}%")
    print("\n=== META combine ===")
    combine = meta.get("combine") or {}
    for k, v in combine.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
