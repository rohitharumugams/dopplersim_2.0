"""VS13 working geometry for the 2D whiteboard vehicle schematic.

Loads the researched coordinate set and maps it onto catalog vehicle IDs.
Body contours are drawn in the frontend; this module only supplies measured
dimensions, tire specs, and source XYZ in the axle-midpoint frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GEOMETRY_JSON = (
    Path(__file__).resolve().parent.parent.parent
    / "misc"
    / "car_schematics"
    / "VS13_DopplerSim_geometry.json"
)

CATALOG_TO_VS13_ID: dict[str, int] = {
    "CitroenC4Picasso": 1,
    "KiaSportage": 2,
    "Mazda3": 3,
    "MercedesAMG550": 4,
    "MercedesGLA": 5,
    "NissanQashqai": 6,
    "OpelInsignia": 7,
    "Peugeot208": 8,
    "Peugeot3008": 9,
    "Peugeot307": 10,
    "RenaultCaptur": 11,
    "RenaultScenic": 12,
    "VWPassat": 13,
}

# Parametric silhouette family only — dimensions still come from the dataset.
BODY_FAMILY: dict[str, str] = {
    "CitroenC4Picasso": "mpv",
    "KiaSportage": "suv",
    "Mazda3": "hatch",
    "MercedesAMG550": "sedan",
    "MercedesGLA": "suv",
    "NissanQashqai": "suv",
    "OpelInsignia": "liftback",
    "Peugeot208": "hatch",
    "Peugeot3008": "suv",
    "Peugeot307": "hatch",
    "RenaultCaptur": "suv",
    "RenaultScenic": "mpv",
    "VWPassat": "sedan",
}

_SOURCE_KEYS = (
    ("engine", "Engine acoustic center"),
    ("intake", "Intake acoustic center"),
    ("exhaust", "Exhaust outlet acoustic center"),
    ("muffler", "Rear silencer/muffler"),
)

_cache: dict[int, dict[str, Any]] | None = None


def _xyz(block: dict[str, Any], key: str) -> list[float | None]:
    item = block.get(key) or {}
    xyz = item.get("xyz_m") or [None, None, None]
    out: list[float | None] = []
    for value in xyz[:3]:
        out.append(None if value is None else float(value))
    while len(out) < 3:
        out.append(None)
    return out


def _load_vs13() -> dict[int, dict[str, Any]]:
    global _cache
    if _cache is not None:
        return _cache
    raw = json.loads(GEOMETRY_JSON.read_text(encoding="utf-8"))
    by_id: dict[int, dict[str, Any]] = {}
    for row in raw:
        by_id[int(row["vehicle_id"])] = row
    _cache = by_id
    return by_id


def geometry_for_catalog_id(catalog_id: str) -> dict[str, Any] | None:
    vs13_id = CATALOG_TO_VS13_ID.get(catalog_id)
    if vs13_id is None or not GEOMETRY_JSON.is_file():
        return None
    row = _load_vs13().get(vs13_id)
    if row is None:
        return None
    dims = row.get("dimensions_m") or {}
    sources_raw = row.get("sources_and_geometry") or {}
    sources = {name: _xyz(sources_raw, key) for name, key in _SOURCE_KEYS}
    return {
        "vs13_id": vs13_id,
        "working_identity": row.get("working_identity") or "",
        "identity_confidence": row.get("identity_confidence") or "",
        "body_family": BODY_FAMILY.get(catalog_id, "hatch"),
        "tire": row.get("tire") or "",
        "dims": {
            "L": float(dims["L"]),
            "W": float(dims["W"]),
            "Wm": None if dims.get("Wm") is None else float(dims["Wm"]),
            "H": float(dims["H"]),
            "WB": float(dims["WB"]),
            "FT": float(dims["FT"]),
            "RT": float(dims["RT"]),
            "FO": float(dims["FO"]),
            "RO": float(dims["RO"]),
            "GC": float(dims["GC"]),
        },
        "sources": sources,
    }


def attach_geometry(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for option in options:
        geometry = geometry_for_catalog_id(str(option.get("id") or ""))
        if geometry is not None:
            option["geometry"] = geometry
    return options
