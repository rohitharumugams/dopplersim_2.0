"""Batch sample planning (straight-line path only)."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from doppler_sim.batch.catalog import SourceClip, VehicleCatalog
from doppler_sim.batch.constants import DEFAULT_BATCH_OUTPUT_DIR, DEFAULT_BATCH_INPUT_DIR, PATH_TYPE_STRAIGHT, to_project_relative
from doppler_sim.batch.sampler import SamplerBank, distribute_even_counts, evenly_spaced_speeds
from doppler_sim.specg.explorer import DEFAULT_BATCH_SPEC_KEYS, SPECG_DEFAULT_FMAX_HZ


@dataclass
class VehicleSelection:
    vehicle: str
    source_speed_mps: float


@dataclass
class BatchConfig:
    batch_name: str
    total_clips: int
    selections: list[VehicleSelection]
    speed_mps_min: float = 10.0
    speed_mps_max: float = 40.0
    cpa_distance_min: float = 0.5
    cpa_distance_max: float = 100.0
    h1_m: float = 0.5
    t_cpa1_s: float = 2.0
    t_cpa2_min: float = 5.0
    t_cpa2_max: float = 5.0
    vehicle_length_m: float = 4.5
    vehicle_lengths: dict[str, float] = field(default_factory=dict)
    num_emitters: int = 2
    t_out_s: float = 10.0
    seed: int = 42
    spectrogram_types: list[str] = field(default_factory=lambda: list(DEFAULT_BATCH_SPEC_KEYS))
    specg_fmax_hz: float = SPECG_DEFAULT_FMAX_HZ
    output_dir: str = DEFAULT_BATCH_OUTPUT_DIR
    input_dir: str = DEFAULT_BATCH_INPUT_DIR

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "selections": [asdict(s) for s in self.selections],
            "path_type": PATH_TYPE_STRAIGHT,
        }


@dataclass
class PlannedSample:
    index: int
    vehicle: str
    source_speed_mps: float
    source_path: str
    speed_mps: float
    cpa_distance_m: float
    cpa_time_sec: float
    h1_m: float
    t_cpa1_s: float
    vehicle_length_m: float
    num_emitters: int
    t_out_s: float
    path_type: str = PATH_TYPE_STRAIGHT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchPlan:
    config: BatchConfig
    samples: list[PlannedSample] = field(default_factory=list)

    def save(self, path: Path) -> None:
        payload = {
            "config": self.config.to_dict(),
            "samples": [s.to_dict() for s in self.samples],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BatchPlan":
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = data["config"]
        selections = [VehicleSelection(**s) for s in cfg.pop("selections")]
        cfg.pop("path_type", None)
        spectrogram_types = cfg.pop("spectrogram_types", None)
        vehicle_lengths = cfg.pop("vehicle_lengths", None)
        config = BatchConfig(selections=selections, **cfg)
        if spectrogram_types is not None:
            config.spectrogram_types = spectrogram_types
        if vehicle_lengths is not None:
            config.vehicle_lengths = vehicle_lengths
        samples = [PlannedSample(**row) for row in data["samples"]]
        return cls(config=config, samples=samples)


def _plan_sample_assignments(
    selections: list[VehicleSelection],
    total_clips: int,
    speed_min: float,
    speed_max: float,
    rng: random.Random,
) -> list[tuple[VehicleSelection, float]]:
    """Fair vehicle counts; each vehicle gets an evenly spaced speed schedule."""
    counts = distribute_even_counts(total_clips, len(selections))
    assignments: list[tuple[VehicleSelection, float]] = []

    for sel, n_clips in zip(selections, counts):
        speeds = evenly_spaced_speeds(speed_min, speed_max, n_clips)
        for speed_mps in speeds:
            assignments.append((sel, speed_mps))

    rng.shuffle(assignments)
    return assignments


def build_batch_plan(
    config: BatchConfig,
    catalog: VehicleCatalog,
    base_dir: Path,
) -> tuple[BatchPlan, SamplerBank]:
    if not config.selections:
        raise ValueError("Select at least one vehicle with a source speed.")

    rng = random.Random(config.seed)
    selections: list[VehicleSelection] = []
    for sel in config.selections:
        clip = catalog.clip_for(sel.vehicle, sel.source_speed_mps)
        if clip is None:
            raise ValueError(
                f"No input clip for {sel.vehicle} at {sel.source_speed_mps:.3f} m/s "
                f"(expected a matching file in static/inputs/, e.g. {sel.vehicle}_59.wav or "
                f"{sel.vehicle}_{sel.source_speed_mps:g}mps.wav)"
            )
        selections.append(sel)

    assignments = _plan_sample_assignments(
        selections,
        config.total_clips,
        config.speed_mps_min,
        config.speed_mps_max,
        rng,
    )

    bank = SamplerBank()
    samples: list[PlannedSample] = []

    dist_sampler = bank.get(
        "distance",
        int(config.cpa_distance_min * 10),
        int(config.cpa_distance_max * 10),
    )
    tcpa_sampler = bank.get(
        "t_cpa2",
        int(config.t_cpa2_min * 100),
        int(config.t_cpa2_max * 100),
    )

    for idx, (sel, speed_mps) in enumerate(assignments, start=1):
        clip = catalog.clip_for(sel.vehicle, sel.source_speed_mps)
        assert clip is not None

        cpa_distance_m = dist_sampler.next() / 10.0
        cpa_time_sec = tcpa_sampler.next() / 100.0
        vehicle_length_m = config.vehicle_lengths.get(sel.vehicle, config.vehicle_length_m)
        t_cpa1_s = clip.t_cpa1_s if clip.t_cpa1_s is not None else config.t_cpa1_s

        samples.append(
            PlannedSample(
                index=idx,
                vehicle=sel.vehicle,
                source_speed_mps=sel.source_speed_mps,
                source_path=to_project_relative(base_dir, clip.path),
                speed_mps=speed_mps,
                cpa_distance_m=cpa_distance_m,
                cpa_time_sec=cpa_time_sec,
                h1_m=config.h1_m,
                t_cpa1_s=t_cpa1_s,
                vehicle_length_m=vehicle_length_m,
                num_emitters=config.num_emitters,
                t_out_s=config.t_out_s,
            )
        )

    return BatchPlan(config=config, samples=samples), bank
