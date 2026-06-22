"""
Pass-by envelope physics (Cevher et al. 2009, IEEE TSP 57(1)).

Models spatial acoustic pattern components that shape the drive-by power envelope
beyond a single moving monopole:

  · Tire horn effect — directional amplification of tyre–road noise
  · Axle dipole interference — coherent left/right and front/rear tyre pairs
  · Wake turbulence — trailing broadband after the vehicle passes the observer

References: envelope shape (ES) components in Sec. V.A; horn effect Sec. II.b;
dipole interference Sec. IV.
"""

from __future__ import annotations

import numpy as np

from doppler_5dot0.config import HORN_PEAK_DB, SPEED_REF_MPS


def speed_dependent_gain(
    speed_mps: float,
    *,
    exponent: float,
    ref_mps: float = SPEED_REF_MPS,
) -> float:
    """Mechanism loudness vs speed (tyre ~ v^2, aero stronger at highway speed)."""
    v = max(1.0, float(speed_mps))
    r = max(1.0, float(ref_mps))
    return float((v / r) ** float(exponent))


def tire_horn_gain_linear(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    obs_xyz: tuple[float, float, float],
    *,
    peak_db: float = HORN_PEAK_DB,
) -> np.ndarray:
    """
    Simplified horn directivity for a tyre contact patch.

    Radiation is strongest when the observer lies in the forward tyre-radiation
    quadrant relative to vehicle motion (Cevher Sec. II.b, Fig. 5).
    """
    ox, oy, _ = obs_xyz
    px = ox - x
    py = oy - y
    bearing = np.arctan2(py, px)
    heading = np.arctan2(vy, vx + 1e-12)
    rel = bearing - heading
    rel = (rel + np.pi) % (2.0 * np.pi) - np.pi
    # Forward hemisphere weighting; rear hemisphere gets baseline unity.
    forward = np.cos(np.clip(rel, -0.5 * np.pi, 0.5 * np.pi)) ** 2
    peak_lin = 10.0 ** (float(peak_db) / 20.0)
    return 1.0 + (peak_lin - 1.0) * np.maximum(forward, 0.0)


def observer_side_gain(
    x: np.ndarray,
    y: np.ndarray,
    obs_xyz: tuple[float, float, float],
    *,
    side_db: float = 3.0,
) -> np.ndarray:
    """
    Mild lateral directivity: body/aero patches on the observer side are louder.
    """
    ox, oy, _ = obs_xyz
    side = np.sign(oy - y)
    side = np.where(np.abs(side) < 0.01, 1.0, side)
    boost = 10.0 ** (float(side_db) / 20.0)
    return np.where(side > 0, boost, 1.0 / boost)
