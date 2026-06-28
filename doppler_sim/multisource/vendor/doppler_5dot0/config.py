"""VS13 multi-source pass-by configuration (vendored for dopplersim_2.0 multisource)."""

import os

PKG_DIR = os.path.dirname(__file__)
ROOT_DIR = PKG_DIR

SAMPLE_RATE = 44100
MAX_DURATION_S = 30.0

# VS13 roadside geometry (Djukanović et al.)
VS13_DCPA_M = 0.5
OBSERVER_XYZ = (0.0, 0.0, 1.2)

# Compact SUV body used for VS13 inverse (Kia Sportage scale, shortened acoustic length)
CAR_LENGTH_M = 3.0
DEFAULT_TEMP_C = 20.0
DEFAULT_RH = 50.0

# Tonal radiators (ISO 362 / SPC point-source layout)
ENGINE_DX_M = 0.32
INTAKE_DX_M = 0.52
EXHAUST_DX_M = -1.18
SOURCE_HEIGHT_M = 0.55
EXHAUST_HEIGHT_M = 0.28
INTAKE_HEIGHT_M = 0.42

# Broadband clusters
TIRE_AXLE_DX_M = 0.90
TIRE_TRACK_DY_M = 0.74
BODY_PATCH_COUNT = 5
AERO_PATCH_COUNT = 2

# Propagation
NEAR_FIELD_M = 2.0
TIRE_NEAR_FIELD_M = 7.0
BODY_NEAR_FIELD_M = 10.0
GROUND_REFLECTION_COEFF = 0.22

# Inverse fit: keep broadband clusters from collapsing under tonal-dominated STFT fit
MIN_BODY_GAIN = 0.52
MIN_TIRE_GAIN = 0.32
MIN_AERO_GAIN = 0.28
MIN_BED_GAIN = 0.22

# Tonal cluster: engine+intake coherent LF, exhaust partial; HF power-blend
TONAL_LF_COHERENCE = 0.72
TONAL_HF_INCOHERENCE = 0.28
TONAL_CROSSOVER_HZ = 850.0
FRONT_TONAL_CROSSOVER_HZ = 1100.0

OUTPUT_DIR = os.path.join(PKG_DIR, "output")

VEHICLE_SOUNDS_DIR = ""
DEFAULT_CAR_MODEL_ID = "kia_sportage"
VS13_DIR = ""
