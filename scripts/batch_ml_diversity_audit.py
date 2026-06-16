#!/usr/bin/env python3
"""Audit batch-generation parameter and audio diversity for VAE/diffusion training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doppler_sim.batch.catalog import scan_input_catalog
from doppler_sim.batch.constants import CSV_HEADERS
from doppler_sim.batch.planner import BatchConfig, VehicleSelection, build_batch_plan
from doppler_sim.batch.pipeline import synthesize_planned_sample


def _uniq(xs: list) -> list:
    return sorted(set(xs))


def analyze_plan(label: str, cfg: BatchConfig, catalog) -> object:
    plan, _ = build_batch_plan(cfg, catalog, ROOT)
    speeds = [s.speed_mps for s in plan.samples]
    dists = [s.cpa_distance_m for s in plan.samples]
    tcpas = [s.cpa_time_sec for s in plan.samples]
    vehs = [s.vehicle for s in plan.samples]

    print(f"\n=== PLAN: {label} (n={len(plan.samples)}) ===")
    print(f"  vehicles: {len(_uniq(vehs))} unique")
    print(f"  speed_mps: {len(_uniq(speeds))} unique, range [{min(speeds):.1f}, {max(speeds):.1f}]")
    print(f"  cpa_distance_m: {len(_uniq(dists))} unique, range [{min(dists):.1f}, {max(dists):.1f}]")
    print(f"  cpa_time_sec: {len(_uniq(tcpas))} unique")

    tuples = list(zip(speeds, dists, tcpas, vehs))
    u = len(set(tuples))
    print(f"  unique (speed, dist, tcpa, vehicle) combos: {u}/{len(tuples)} ({100 * u / len(tuples):.0f}%)")

    flags: list[str] = []
    if len(_uniq(tcpas)) == 1:
        flags.append("t_CPA2 fixed — all clips share same pass-by timing")
    if len(_uniq(speeds)) < 5:
        flags.append("low speed diversity")
    if len(_uniq(dists)) < 5:
        flags.append("low distance diversity")
    if len(_uniq(vehs)) == 1:
        flags.append("single vehicle class")
    if flags:
        print("  FLAGS:", "; ".join(flags))
    return plan


def audit_audio_diversity(plan, n: int = 5) -> None:
    print(f"\n=== AUDIO SYNTHESIS ({n} clips) ===")
    feats: list[dict] = []
    for sample in plan.samples[:n]:
        audio, _quantities, _aux = synthesize_planned_sample(sample)
        rms = float(np.sqrt(np.mean(audio**2)))
        spec = np.abs(np.fft.rfft(audio))
        spec = spec / (np.linalg.norm(spec) + 1e-12)
        centroid = float(np.sum(np.arange(len(spec)) * spec) / (np.sum(spec) + 1e-12))
        feats.append(
            {
                "idx": sample.index,
                "vehicle": sample.vehicle,
                "v2": sample.speed_mps,
                "h2": sample.cpa_distance_m,
                "tcpa2": sample.cpa_time_sec,
                "rms": rms,
                "centroid_bin": centroid,
                "spec": spec,
            }
        )
        print(
            f"  sample {sample.index}: {sample.vehicle} "
            f"v2={sample.speed_mps} h2={sample.cpa_distance_m:.1f} "
            f"t_cpa2={sample.cpa_time_sec} rms={rms:.4f}"
        )

    print("\n  pairwise spectral cosine distances (1 - cos, higher = more different):")
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            a, b = feats[i]["spec"], feats[j]["spec"]
            n_bins = min(len(a), len(b))
            cos = float(np.dot(a[:n_bins], b[:n_bins]))
            print(f"    {feats[i]['idx']} vs {feats[j]['idx']}: {1.0 - cos:.4f}")


def main() -> int:
    catalog = scan_input_catalog(ROOT)
    vehicles = catalog.sorted_vehicle_names()
    print("=== INPUT CATALOG ===")
    print(f"vehicles: {len(vehicles)}")
    if not vehicles:
        print("No input clips in static/inputs/ — add WAV files before running audio synthesis.")
        return 1

    selections = [
        VehicleSelection(vehicle=v, source_speed_mps=catalog.vehicles[v][0].speed_mps)
        for v in vehicles[: min(3, len(vehicles))]
    ]

    analyze_plan(
        "CURRENT DEFAULTS (t_CPA2 = 5 s fixed)",
        BatchConfig(
            batch_name="audit",
            total_clips=50,
            selections=selections,
            t_cpa2_min=5,
            t_cpa2_max=5,
        ),
        catalog,
    )
    plan_ml = analyze_plan(
        "ML-FRIENDLY (t_CPA2 2–8 s, 3 vehicles)",
        BatchConfig(
            batch_name="audit_ml",
            total_clips=50,
            selections=selections,
            t_cpa2_min=2,
            t_cpa2_max=8,
        ),
        catalog,
    )
    analyze_plan(
        "WORST CASE (single vehicle, fixed t_CPA2)",
        BatchConfig(
            batch_name="audit_one",
            total_clips=30,
            selections=[selections[0]],
            t_cpa2_min=5,
            t_cpa2_max=5,
        ),
        catalog,
    )

    print("\n=== ML EXPORT CHECKLIST ===")
    print("  dataset.csv columns:", ", ".join(CSV_HEADERS))
    print("  per sample: WAV, spectrograms/*, metadata/*.npy, simulation_parameters.json")

    synthesize = "--synthesize" in sys.argv
    if synthesize:
        audit_audio_diversity(plan_ml, n=min(5, len(plan_ml.samples)))
    else:
        print("\nRun with --synthesize to render a few clips and compare audio diversity.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
