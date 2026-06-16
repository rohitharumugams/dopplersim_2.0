"""Discover source recordings from the static/inputs/ folder."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from doppler_sim.batch.constants import resolve_input_root

KMH_PER_MPS = 3.6

# sedan_10mps, sedan_10_mps
CLIP_PATTERN_MPS = re.compile(r"^(.+?)_(\d+(?:\.\d+)?)(?:_)?mps$", re.IGNORECASE)
# sedan_59kmh
CLIP_PATTERN_KMPH = re.compile(r"^(.+?)_(\d+(?:\.\d+)?)(?:_)?kmh$", re.IGNORECASE)
# CitroenC4Picasso_59 — bare numeric suffix is treated as km/h (common pass-by naming)
CLIP_PATTERN_BARE = re.compile(r"^(.+?)_(\d+(?:\.\d+)?)$")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass
class SourceClip:
    vehicle: str
    speed_mps: float
    path: Path
    speed_label: str = ""
    t_cpa1_s: float | None = None
    sidecar_path: Path | None = None

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass
class VehicleCatalog:
    vehicles: dict[str, list[SourceClip]] = field(default_factory=dict)

    def sorted_vehicle_names(self) -> list[str]:
        return sorted(self.vehicles.keys())

    def speeds_for(self, vehicle: str) -> list[float]:
        clips = self.vehicles.get(vehicle, [])
        return sorted({c.speed_mps for c in clips})

    def clip_for(self, vehicle: str, speed_mps: float) -> SourceClip | None:
        for clip in self.vehicles.get(vehicle, []):
            if abs(clip.speed_mps - speed_mps) < 1e-6:
                return clip
        return None

    def median_speed(self, vehicle: str) -> float | None:
        speeds = self.speeds_for(vehicle)
        if not speeds:
            return None
        mid = len(speeds) // 2
        return speeds[mid]


def parse_clip_stem(stem: str) -> tuple[str, float, str] | None:
    """Parse a clip filename stem into vehicle name, speed (m/s), and display label."""
    match = CLIP_PATTERN_MPS.match(stem)
    if match:
        vehicle = match.group(1)
        speed_mps = float(match.group(2))
        return vehicle, speed_mps, f"{speed_mps:g} m/s"

    match = CLIP_PATTERN_KMPH.match(stem)
    if match:
        vehicle = match.group(1)
        speed_kmh = float(match.group(2))
        speed_mps = speed_kmh / KMH_PER_MPS
        return vehicle, speed_mps, f"{speed_kmh:g} km/h"

    match = CLIP_PATTERN_BARE.match(stem)
    if match:
        vehicle = match.group(1)
        speed_kmh = float(match.group(2))
        speed_mps = speed_kmh / KMH_PER_MPS
        return vehicle, speed_mps, f"{speed_kmh:g} km/h"

    return None


def parse_clip_sidecar(txt_path: Path) -> tuple[float, float] | None:
    """Parse `<stem>.txt` sidecar: `speed_kmh t_cpa1_s` (space or comma separated)."""
    if not txt_path.is_file():
        return None
    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = re.split(r"[\s,]+", text.replace(",", " "))
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    try:
        speed_kmh = float(parts[0])
        t_cpa1_s = float(parts[1])
    except ValueError:
        return None
    if speed_kmh <= 0 or t_cpa1_s < 0:
        return None
    return speed_kmh, t_cpa1_s


def scan_input_catalog(base_dir: Path, input_dir: str | None = None) -> VehicleCatalog:
    root = resolve_input_root(base_dir, input_dir)
    root.mkdir(parents=True, exist_ok=True)
    catalog = VehicleCatalog()

    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        parsed = parse_clip_stem(path.stem)
        if parsed is None:
            continue
        vehicle, speed_mps, speed_label = parsed
        sidecar_path = path.with_suffix(".txt")
        t_cpa1_s: float | None = None
        sidecar = parse_clip_sidecar(sidecar_path)
        if sidecar is not None:
            speed_kmh, t_cpa1_s = sidecar
            speed_mps = speed_kmh / KMH_PER_MPS
            speed_label = f"{speed_kmh:g} km/h (sidecar)"

        clip = SourceClip(
            vehicle=vehicle,
            speed_mps=speed_mps,
            path=path,
            speed_label=speed_label,
            t_cpa1_s=t_cpa1_s,
            sidecar_path=sidecar_path if sidecar_path.is_file() else None,
        )
        catalog.vehicles.setdefault(vehicle, []).append(clip)

    for vehicle in catalog.vehicles:
        catalog.vehicles[vehicle].sort(key=lambda c: c.speed_mps)

    return catalog
