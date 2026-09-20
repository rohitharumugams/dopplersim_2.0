"""Batch planning for 2D whiteboard (free-path) datasets."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from doppler_sim.batch.catalog import VehicleCatalog
from doppler_sim.batch.constants import (
    DEFAULT_BATCH_INPUT_DIR,
    DEFAULT_BATCH_OUTPUT_DIR,
    FEATURE_SR,
    to_project_relative,
)
from doppler_sim.batch.planner import (
    VehicleSelection,
    _plan_sample_assignments,
)
from doppler_sim.batch.phase1_state import PATH_TYPE_FREE_2D
from doppler_sim.batch.sampler import SamplerBank
from doppler_sim.path2d.path_generator import _path_length, random_curved_path_2d
from doppler_sim.path2d.synthesis import resample_path_constant_speed
from doppler_sim.specg.explorer import DEFAULT_BATCH_SPEC_KEYS, SPECG_DEFAULT_FMAX_HZ

# Train-mode clip duration targets (path length / speed + end pads).
TRAIN_DURATION_MIN_S = 8.0
TRAIN_DURATION_MAX_S = 12.0
TRAIN_DURATION_TYPICAL_S = 10.0
TRAIN_PATH_PAD_S = 0.5  # matches synthesize_path_audio default 2 * 0.25 s


@dataclass
class Path2dBatchConfig:
    batch_name: str
    total_clips: int
    selections: list[VehicleSelection]
    speed_mps_min: float = 30.0 / 3.6  # 30 km/h
    speed_mps_max: float = 105.0 / 3.6  # 105 km/h
    h1_m: float = 0.5
    t_cpa1_s: float = 2.0
    vehicle_length_m: float = 4.5
    vehicle_lengths: dict[str, float] = field(default_factory=dict)
    num_emitters: int = 1
    seed: int = 42
    spectrogram_types: list[str] = field(default_factory=lambda: list(DEFAULT_BATCH_SPEC_KEYS))
    spectrogram_png_types: list[str] = field(default_factory=list)
    generate_combined_png: bool = False
    simple_generate: bool = False
    train_mode: bool = False
    speed_unit: str = "kmph"
    num_workers: int = 10
    specg_fmax_hz: float = SPECG_DEFAULT_FMAX_HZ
    output_dir: str = DEFAULT_BATCH_OUTPUT_DIR
    input_dir: str = DEFAULT_BATCH_INPUT_DIR
    # Path generation (above x-axis, smooth curves)
    path_x_start_min: float = -70.0
    path_x_start_max: float = -35.0
    path_x_end_min: float = 35.0
    path_x_end_max: float = 70.0
    path_y_min: float = 5.0
    path_y_max: float = 28.0
    path_max_turn_deg: float = 40.0
    path_segments: int = 4
    path_smooth_samples: int = 20
    mic_x: float = 0.0
    mic_y: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "selections": [asdict(s) for s in self.selections],
            "path_type": PATH_TYPE_FREE_2D,
        }


def path2d_batch_config_from_dict(cfg: dict[str, Any]) -> Path2dBatchConfig:
    data = dict(cfg)
    selections = [VehicleSelection(**s) for s in data.pop("selections")]
    data.pop("path_type", None)
    spectrogram_types = data.pop("spectrogram_types", None)
    spectrogram_png_types = data.pop("spectrogram_png_types", None)
    vehicle_lengths = data.pop("vehicle_lengths", None)
    data.pop("generate_spec_png", None)
    simple_generate = data.pop("simple_generate", False)
    train_mode = data.pop("train_mode", False)
    speed_unit = data.pop("speed_unit", "mps")
    num_workers = max(1, int(data.pop("num_workers", 10)))
    # Straight-batch fields ignored on resume if present.
    for legacy in (
        "cpa_distance_min",
        "cpa_distance_max",
        "t_cpa2_min",
        "t_cpa2_max",
        "t_out_s",
    ):
        data.pop(legacy, None)
    config = Path2dBatchConfig(
        selections=selections,
        num_workers=num_workers,
        **data,
    )
    if spectrogram_types is not None:
        config.spectrogram_types = spectrogram_types
    if spectrogram_png_types is not None:
        config.spectrogram_png_types = list(spectrogram_png_types)
    if vehicle_lengths is not None:
        config.vehicle_lengths = vehicle_lengths
    config.simple_generate = bool(simple_generate)
    config.train_mode = bool(train_mode)
    if config.train_mode:
        # MVP train export: STFT only; no spectrogram PNGs / Phase-1 "simple" extras.
        config.spectrogram_types = ["stft"]
        config.spectrogram_png_types = []
        config.generate_combined_png = False
        config.simple_generate = False
    config.speed_unit = "kmph" if speed_unit == "kmph" else "mps"
    return config


@dataclass
class Path2dPlannedSample:
    index: int
    vehicle: str
    source_speed_mps: float
    source_path: str
    speed_mps: float
    path_xy: list[list[float]]
    mic_x: float
    mic_y: float
    cpa_distance_m: float
    cpa_time_sec: float
    t_out_s: float
    h1_m: float
    t_cpa1_s: float
    vehicle_length_m: float
    num_emitters: int
    path_type: str = PATH_TYPE_FREE_2D

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Path2dBatchPlan:
    config: Path2dBatchConfig
    samples: list[Path2dPlannedSample] = field(default_factory=list)

    def save(self, path: Path) -> None:
        payload = {
            "config": self.config.to_dict(),
            "samples": [s.to_dict() for s in self.samples],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Path2dBatchPlan":
        data = json.loads(path.read_text(encoding="utf-8"))
        config = path2d_batch_config_from_dict(data["config"])
        samples = [Path2dPlannedSample(**row) for row in data["samples"]]
        return cls(config=config, samples=samples)


def _estimate_path_cpa(
    path_xy: np.ndarray,
    *,
    speed_mps: float,
    mic_x: float,
    mic_y: float,
) -> tuple[float, float, float]:
    """Instantaneous CPA time/distance and output duration from path kinematics."""
    traj = resample_path_constant_speed(path_xy, speed_mps=float(speed_mps), sr=FEATURE_SR)
    mx, my = float(mic_x), float(mic_y)
    range_inst = np.sqrt((traj["x"] - mx) ** 2 + (traj["y"] - my) ** 2)
    cpa_idx = int(np.argmin(range_inst))
    return (
        float(range_inst[cpa_idx]),
        float(traj["t"][cpa_idx]),
        float(traj["duration_s"][0]),
    )


def _sample_train_target_duration_s(rng: random.Random) -> float:
    """Sample clip length in [8, 12] s with mode at 10 s."""
    return float(
        rng.triangular(TRAIN_DURATION_MIN_S, TRAIN_DURATION_MAX_S, TRAIN_DURATION_TYPICAL_S)
    )


def _retarget_train_mode_duration(
    path_xy: np.ndarray,
    speed_mps: float,
    *,
    speed_min: float,
    speed_max: float,
    mic_x: float,
    mic_y: float,
    rng: random.Random,
) -> tuple[np.ndarray, float, float, float, float]:
    """Adjust path/speed so clip duration falls in 8–12 s (≈10 s typical).

    Prefer retargeting speed; if that exits the allowed band, uniformly scale the
    path about the microphone so length matches the clamped speed.
    """
    xy = np.asarray(path_xy, dtype=np.float64)
    length = _path_length(xy)
    if length < 1e-3:
        raise ValueError("path length too short for train-mode retargeting")

    target_t = _sample_train_target_duration_s(rng)
    travel_s = max(target_t - TRAIN_PATH_PAD_S, 1e-3)
    v_needed = length / travel_s
    v_lo = float(min(speed_min, speed_max))
    v_hi = float(max(speed_min, speed_max))

    if v_lo <= v_needed <= v_hi:
        speed = float(v_needed)
    else:
        speed = float(np.clip(v_needed, v_lo, v_hi))
        length_needed = speed * travel_s
        scale = length_needed / length
        mx, my = float(mic_x), float(mic_y)
        xy = np.column_stack(
            [
                mx + (xy[:, 0] - mx) * scale,
                my + (xy[:, 1] - my) * scale,
            ]
        )

    cpa_distance_m, cpa_time_sec, t_out_s = _estimate_path_cpa(
        xy,
        speed_mps=speed,
        mic_x=mic_x,
        mic_y=mic_y,
    )
    # Numerical guard: keep within the train window (± small tolerance).
    if t_out_s < TRAIN_DURATION_MIN_S - 0.05 or t_out_s > TRAIN_DURATION_MAX_S + 0.05:
        # Final snap: scale path to exact target at current speed.
        length = _path_length(xy)
        travel_s = max(target_t - TRAIN_PATH_PAD_S, 1e-3)
        scale = (speed * travel_s) / max(length, 1e-9)
        mx, my = float(mic_x), float(mic_y)
        xy = np.column_stack(
            [
                mx + (xy[:, 0] - mx) * scale,
                my + (xy[:, 1] - my) * scale,
            ]
        )
        cpa_distance_m, cpa_time_sec, t_out_s = _estimate_path_cpa(
            xy,
            speed_mps=speed,
            mic_x=mic_x,
            mic_y=mic_y,
        )

    return xy, float(speed), float(cpa_distance_m), float(cpa_time_sec), float(t_out_s)


def build_path2d_batch_plan(
    config: Path2dBatchConfig,
    catalog: VehicleCatalog,
    base_dir: Path,
) -> tuple[Path2dBatchPlan, SamplerBank]:
    if not config.selections:
        raise ValueError("Select at least one vehicle with a source speed.")

    rng = random.Random(config.seed)
    selections: list[VehicleSelection] = []
    for sel in config.selections:
        clip = catalog.clip_for(sel.vehicle, sel.source_speed_mps)
        if clip is None:
            raise ValueError(
                f"No input clip for {sel.vehicle} at {sel.source_speed_mps:.3f} m/s "
                f"(expected a matching file in static/inputs/)"
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
    samples: list[Path2dPlannedSample] = []

    x_start_sampler = bank.get(
        "path_x_start",
        int(config.path_x_start_min),
        int(config.path_x_start_max),
    )
    x_end_sampler = bank.get(
        "path_x_end",
        int(config.path_x_end_min),
        int(config.path_x_end_max),
    )

    for idx, (sel, speed_mps) in enumerate(assignments, start=1):
        clip = catalog.clip_for(sel.vehicle, sel.source_speed_mps)
        assert clip is not None

        sample_rng = random.Random(config.seed * 100_003 + idx * 7919)
        x_start = float(x_start_sampler.next())
        x_end = float(x_end_sampler.next())
        if x_end <= x_start + 20.0:
            x_start, x_end = min(x_start, x_end - 25.0), max(x_end, x_start + 25.0)

        path_xy = random_curved_path_2d(
            sample_rng,
            x_start=x_start,
            x_end=x_end,
            y_min=float(config.path_y_min),
            y_max=float(config.path_y_max),
            n_segments=int(config.path_segments),
            points_per_segment=int(config.path_smooth_samples),
            max_turn_deg=float(config.path_max_turn_deg),
            mic_x=float(config.mic_x),
            mic_y=float(config.mic_y),
        )

        if config.train_mode:
            path_xy, speed_mps, cpa_distance_m, cpa_time_sec, t_out_s = _retarget_train_mode_duration(
                path_xy,
                float(speed_mps),
                speed_min=float(config.speed_mps_min),
                speed_max=float(config.speed_mps_max),
                mic_x=float(config.mic_x),
                mic_y=float(config.mic_y),
                rng=sample_rng,
            )
        else:
            cpa_distance_m, cpa_time_sec, t_out_s = _estimate_path_cpa(
                path_xy,
                speed_mps=float(speed_mps),
                mic_x=float(config.mic_x),
                mic_y=float(config.mic_y),
            )

        vehicle_length_m = config.vehicle_lengths.get(sel.vehicle, config.vehicle_length_m)
        t_cpa1_s = clip.t_cpa1_s if clip.t_cpa1_s is not None else config.t_cpa1_s

        samples.append(
            Path2dPlannedSample(
                index=idx,
                vehicle=sel.vehicle,
                source_speed_mps=sel.source_speed_mps,
                source_path=to_project_relative(base_dir, clip.path),
                speed_mps=float(speed_mps),
                path_xy=np.asarray(path_xy, dtype=np.float64).tolist(),
                mic_x=float(config.mic_x),
                mic_y=float(config.mic_y),
                cpa_distance_m=float(cpa_distance_m),
                cpa_time_sec=float(cpa_time_sec),
                t_out_s=float(t_out_s),
                h1_m=config.h1_m,
                t_cpa1_s=t_cpa1_s,
                vehicle_length_m=vehicle_length_m,
                num_emitters=config.num_emitters,
            )
        )

    return Path2dBatchPlan(config=config, samples=samples), bank
