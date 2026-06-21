"""VS13 point-source pass-by configuration (vendored for dopplersim_2.0 multisource)."""

import os

PKG_DIR = os.path.dirname(__file__)
ROOT_DIR = PKG_DIR

SAMPLE_RATE = 44100
MAX_DURATION_S = 30.0

# VS13 roadside geometry (Djukanović et al., TELFOR 2022)
VS13_DCPA_M = 0.5
OBSERVER_XYZ = (0.0, 0.0, 1.2)
# Vehicle as distributed body monopoles along x (front / rear), 3 m apart
CAR_LENGTH_M = 3.0
EMITTER_HALF_LENGTH_M = CAR_LENGTH_M / 2.0  # ±1.5 m from vehicle centre
SOURCE_HEIGHT_M = 0.5  # monopole height above road

# Component matched model (engine + incoherent body/tire power sum)
ENGINE_DX_M = 0.30  # powertrain monopole (coherent q₀)
COMPONENT_BODY_COUNT = 9
TIRE_AXLE_DX_M = 0.90  # |dx| from vehicle centre to axle
TIRE_TRACK_DY_M = 0.74  # half track width (nearside wheels at −dy)

# Distributed body synthesis (legacy / vs13_fit)
BODY_SEGMENT_COUNT = 15
BODY_ACOUSTIC_LENGTH_M = 6.0
BODY_NEAR_FIELD_M = 14.0
BODY_COMBINE_MODE = 'hybrid'
BODY_HYBRID_CROSSOVER_HZ = 500.0

DEFAULT_TEMP_C = 20.0
DEFAULT_RH = 50.0

# Near-field softening: amplitude ∝ 1/√(r² + R₀²)
NEAR_FIELD_M = 2.0

# Ground image (hard asphalt)
GROUND_REFLECTION_COEFF = 0.22

OUTPUT_DIR = os.path.join(PKG_DIR, "output")

VS13_DIR = ""
