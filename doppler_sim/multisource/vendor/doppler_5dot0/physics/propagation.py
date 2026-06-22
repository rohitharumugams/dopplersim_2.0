"""
Shared retarded-time propagation helpers for Doppler 5.0 multi-source pass-by.
"""

from __future__ import annotations

import numpy as np

from doppler_4dot0.physics.point_source import radial_geometry
from doppler_4dot0.synthesis import _render_path
from doppler_5dot0.config import (
    AERO_NEAR_FIELD_M,
    BODY_NEAR_FIELD_M,
    GROUND_REFLECTION_COEFF,
    NEAR_FIELD_M,
    OBSERVER_XYZ,
    TIRE_NEAR_FIELD_M,
    TURBULENCE_NEAR_FIELD_M,
)
from doppler_5dot0.physics.passby_envelope import observer_side_gain, tire_horn_gain_linear
from doppler_5dot0.sources.catalog import SourceSpec


def propagate_monopole(
    q: np.ndarray,
    geom: tuple,
    path_index: int,
    *,
    near_field_m: float = NEAR_FIELD_M,
    include_ground: bool = False,
    gain_scale: np.ndarray | None = None,
) -> np.ndarray:
    t, c, x, y, z, vx, vy, vz, _ = geom
    n = len(t)
    r, _ = radial_geometry(
        x[path_index], y[path_index], z[path_index],
        vx[path_index], vy[path_index], vz[path_index], OBSERVER_XYZ,
    )
    gs = gain_scale
    if gs is None and len(x) > path_index:
        gs = None
    p = _render_path(q[:n], t, r, c, gain_scale=gs, near_field_m=near_field_m)
    if include_ground:
        z_img = -np.maximum(z[path_index], 0.01)
        r_g, _ = radial_geometry(
            x[path_index], y[path_index], z_img,
            vx[path_index], vy[path_index], vz[path_index], OBSERVER_XYZ,
        )
        p = p + _render_path(
            q[:n], t, r_g, c,
            gain_scale=GROUND_REFLECTION_COEFF if gs is None else gs * GROUND_REFLECTION_COEFF,
            near_field_m=near_field_m,
        )
    return p.astype(np.float64)


def propagate_tire(
    q: np.ndarray,
    geom: tuple,
    path_index: int,
) -> np.ndarray:
    t, c, x, y, z, vx, vy, vz, _ = geom
    horn = tire_horn_gain_linear(
        x[path_index], y[path_index], z[path_index],
        vx[path_index], vy[path_index],
        OBSERVER_XYZ,
    )
    return propagate_monopole(
        q, geom, path_index,
        near_field_m=TIRE_NEAR_FIELD_M,
        include_ground=True,
        gain_scale=horn,
    )


def propagate_body_patch(q: np.ndarray, geom: tuple, path_index: int) -> np.ndarray:
    t, c, x, y, z, vx, vy, vz, _ = geom
    side = observer_side_gain(x[path_index], y[path_index], OBSERVER_XYZ)
    return propagate_monopole(
        q, geom, path_index,
        near_field_m=BODY_NEAR_FIELD_M,
        include_ground=False,
        gain_scale=side,
    )


def propagate_aero_patch(q: np.ndarray, geom: tuple, path_index: int) -> np.ndarray:
    t, c, x, y, z, vx, vy, vz, _ = geom
    side = observer_side_gain(x[path_index], y[path_index], OBSERVER_XYZ, side_db=4.0)
    return propagate_monopole(
        q, geom, path_index,
        near_field_m=AERO_NEAR_FIELD_M,
        include_ground=False,
        gain_scale=side,
    )


def propagate_wake(q: np.ndarray, geom: tuple, path_index: int) -> np.ndarray:
    t, c, x, y, z, vx, vy, vz, _ = geom
    side = observer_side_gain(x[path_index], y[path_index], OBSERVER_XYZ, side_db=2.0)
    return propagate_monopole(
        q, geom, path_index,
        near_field_m=TURBULENCE_NEAR_FIELD_M,
        include_ground=False,
        gain_scale=side,
    )


def near_field_for_kind(kind: str) -> float:
    if kind == 'tire':
        return TIRE_NEAR_FIELD_M
    if kind == 'body':
        return BODY_NEAR_FIELD_M
    if kind == 'aero':
        return AERO_NEAR_FIELD_M
    if kind == 'wake':
        return TURBULENCE_NEAR_FIELD_M
    return NEAR_FIELD_M


def tire_specs_ordered(tire_specs: list[SourceSpec]) -> tuple[SourceSpec, SourceSpec, SourceSpec, SourceSpec]:
    """Return (fl, fr, rl, rr) in catalog order."""
    by_id = {s.id: s for s in tire_specs}
    return by_id['tire_fl'], by_id['tire_fr'], by_id['tire_rl'], by_id['tire_rr']
