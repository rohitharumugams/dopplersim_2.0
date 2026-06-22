"""Physics priors for forward pass-by (no VS13 clip fitting)."""

from __future__ import annotations

from doppler_5dot0.config import TONAL_LF_COHERENCE
from doppler_5dot0.inverse.synthesis import Vs13MultiCalibration

_V_REF_MPS = 27.78  # 100 km/h reference


def forward_calibration(speed_mps: float) -> Vs13MultiCalibration:
    """
    Speed-scaled gains from pass-by literature (no target recording).

    Tuned so ~100 km/h has mixed engine + tyre mid-band (not LF-only).
    """
    r = max(0.35, float(speed_mps) / _V_REF_MPS)
    # Broadband: floor speed scaling so tyre/body stay audible below ~70 km/h
    r_bb = max(0.62, r)
    # Tonal: slightly stronger at low speed (engine-dominated pass-by)
    tonal_scale = 0.85 + 0.25 * max(0.0, 1.0 - r)
    return Vs13MultiCalibration(
        engine_gain=0.28 * tonal_scale,
        intake_gain=0.22 * tonal_scale,
        exhaust_gain=0.34 * tonal_scale,
        global_gain=0.52,
        body_gain=0.95 * (r_bb ** 1.05),
        tire_gain=1.45 * (r_bb ** 1.15),
        aero_gain=0.55 * (r_bb ** 1.70),
        wake_gain=0.48 * (r_bb ** 1.85),
        bed_gain=0.30 * (r_bb ** 0.90),
        tonal_lf_coherence=TONAL_LF_COHERENCE,
    )
