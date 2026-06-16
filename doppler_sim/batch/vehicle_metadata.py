"""Known vehicle metadata for batch generation (length, display names)."""

from __future__ import annotations

from typing import Any

VEHICLE_METADATA: dict[str, dict[str, Any]] = {
    "Peugeot208": {
        "display_name": "Peugeot 208 1.4 HDI",
        "length_m": 3.962,
        "body_style": "B hatch",
    },
    "RenaultCaptur": {
        "display_name": "Renault Captur 1.5 DCI",
        "length_m": 4.122,
        "body_style": "B-SUV",
    },
    "Peugeot307": {
        "display_name": "Peugeot 307 2.0 HDI",
        "length_m": 4.210,
        "body_style": "C hatch/wagon",
    },
    "RenaultScenic": {
        "display_name": "Renault Scenic III 1.9 DCI",
        "length_m": 4.264,
        "body_style": "C MPV",
    },
    "Peugeot3008": {
        "display_name": "Peugeot 3008 1.6 HDI",
        "length_m": 4.365,
        "body_style": "C-SUV",
    },
    "NissanQashqai": {
        "display_name": "Nissan Qashqai 1.5 DCI J11",
        "length_m": 4.380,
        "body_style": "C-SUV",
    },
    "MercedesGLA": {
        "display_name": "Mercedes GLA 200D",
        "length_m": 4.424,
        "body_style": "C-SUV",
    },
    "CitroenC4Picasso": {
        "display_name": "Citroen C4 Picasso 1.6 HDI",
        "length_m": 4.428,
        "body_style": "C MPV",
    },
    "KiaSportage": {
        "display_name": "Kia Sportage 1.6 GDI",
        "length_m": 4.440,
        "body_style": "C-SUV",
    },
    "Mazda3": {
        "display_name": "Mazda 3 Skyactiv hatch",
        "length_m": 4.465,
        "body_style": "C hatch",
    },
    "VWPassat": {
        "display_name": "VW Passat B7 1.6 TDI",
        "length_m": 4.769,
        "body_style": "D sedan",
    },
    "OpelInsignia": {
        "display_name": "Opel Insignia 2.0 CDTI",
        "length_m": 4.830,
        "body_style": "D liftback",
    },
    "MercedesAMG550": {
        "display_name": "Mercedes S550 W222",
        "length_m": 5.146,
        "body_style": "F sedan",
    },
}


def vehicle_display_name(short_name: str) -> str:
    meta = VEHICLE_METADATA.get(short_name)
    return meta["display_name"] if meta else short_name


def known_length_m(short_name: str) -> float | None:
    meta = VEHICLE_METADATA.get(short_name)
    if not meta:
        return None
    return float(meta["length_m"])


def vehicles_missing_metadata(vehicle_names: list[str]) -> list[str]:
    return [name for name in vehicle_names if name not in VEHICLE_METADATA]
