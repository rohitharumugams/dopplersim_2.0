"""Vehicle pass-by source catalog (engine, intake, exhaust, tyres, body, aero)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from doppler_5dot0.config import (
    AERO_PATCH_COUNT,
    BODY_PATCH_COUNT,
    CAR_LENGTH_M,
    ENGINE_DX_M,
    EXHAUST_DX_M,
    EXHAUST_HEIGHT_M,
    INTAKE_DX_M,
    INTAKE_HEIGHT_M,
    SOURCE_HEIGHT_M,
    TIRE_AXLE_DX_M,
    TIRE_TRACK_DY_M,
)

SourceKind = Literal['tonal', 'tire', 'body', 'aero']


@dataclass(frozen=True)
class SourceSpec:
    id: str
    label: str
    kind: SourceKind
    dx: float
    dy: float
    dz: float
    gain: float = 1.0
    combine: str = 'coherent'  # coherent | incoherent


def build_source_catalog(
    car_length_m: float = CAR_LENGTH_M,
) -> list[SourceSpec]:
    """
    Multi-source layout from vehicle pass-by literature:
      · Engine/powertrain — tonal, front bay
      · Intake — tonal, grille / engine bay (mid band)
      · Exhaust — tonal, rear tailpipe (LF harmonics + resonances)
      · Tyres ×4 — broadband 500–3 kHz, incoherent power sum
      · Body patches — diffuse mid band, incoherent
      · Aero patches — HF rush >2 kHz, incoherent

    See doppler_5dot0/RESEARCH_SOURCES.md for paper ↔ source mapping.
    """
    specs: list[SourceSpec] = [
        SourceSpec('engine', 'Engine / powertrain', 'tonal', ENGINE_DX_M, 0.0, SOURCE_HEIGHT_M, 1.0, 'coherent'),
        SourceSpec('intake', 'Intake / induction', 'tonal', INTAKE_DX_M, 0.0, INTAKE_HEIGHT_M, 0.85, 'coherent'),
        SourceSpec('exhaust', 'Exhaust / tailpipe', 'tonal', EXHAUST_DX_M, -0.28, EXHAUST_HEIGHT_M, 1.05, 'coherent'),
    ]

    tire_layout = [
        ('tire_fl', TIRE_AXLE_DX_M, TIRE_TRACK_DY_M, 0.85),
        ('tire_fr', TIRE_AXLE_DX_M, -TIRE_TRACK_DY_M, 1.10),
        ('tire_rl', -TIRE_AXLE_DX_M, TIRE_TRACK_DY_M, 0.90),
        ('tire_rr', -TIRE_AXLE_DX_M, -TIRE_TRACK_DY_M, 1.15),
    ]
    for tid, dx, dy, g in tire_layout:
        specs.append(SourceSpec(tid, tid.replace('_', ' ').title(), 'tire', dx, dy, 0.01, g, 'incoherent'))

    half = car_length_m / 2.0
    for i in range(BODY_PATCH_COUNT):
        x = -half * 0.85 + (1.7 * half * 0.85) * i / max(BODY_PATCH_COUNT - 1, 1)
        w = 0.85 + 0.15 * (1.0 - abs(x / half))
        specs.append(SourceSpec(
            f'body_{i}', f'Body patch {i + 1}', 'body', float(x), 0.0, SOURCE_HEIGHT_M * 0.9, w, 'incoherent',
        ))

    aero_xs = [(-half * 0.5), (-half * 0.1), (half * 0.15), (half * 0.45)][:AERO_PATCH_COUNT]
    for i, x in enumerate(aero_xs):
        dy = 0.35 if i % 2 == 0 else -0.35
        specs.append(SourceSpec(
            f'aero_{i}', f'Aero patch {i + 1}', 'aero', float(x), dy, 0.95, 0.70, 'incoherent',
        ))

    return specs


def split_by_kind(specs: list[SourceSpec]) -> tuple[list[SourceSpec], list[SourceSpec], list[SourceSpec], list[SourceSpec]]:
    tonal = [s for s in specs if s.kind == 'tonal']
    tires = [s for s in specs if s.kind == 'tire']
    body = [s for s in specs if s.kind == 'body']
    aero = [s for s in specs if s.kind == 'aero']
    return tonal, tires, body, aero
