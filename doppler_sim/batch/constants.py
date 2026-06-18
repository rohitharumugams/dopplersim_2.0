"""Batch generation constants (straight-line path only)."""

from __future__ import annotations

from pathlib import Path

PATH_TYPE_STRAIGHT = "straight"
SAMPLE_ID_WIDTH = 7
KMH_PER_MPS = 3.6
HOP_LENGTH = 512
FEATURE_SR = 44100
N_FFT_STFT = 2048
N_FFT_NARROWBAND = 4096
HOP_NARROWBAND = 1024
CQT_BINS = 84
CQT_BPO = 12

CSV_HEADERS = [
    "sample_id",
    "batch_id",
    "filename",
    "vehicle_class",
    "trajectory_type",
    "source_speed_mps",
    "speed_mps",
    "cpa_distance_m",
    "cpa_time_sec",
    "vehicle_length_m",
    "num_emitters",
    "pass_by_in_clip",
]


def mps_to_display(speed_mps: float, unit: str) -> float:
    if unit == "kmph":
        return round(speed_mps * KMH_PER_MPS, 1)
    return round(float(speed_mps), 1)


def display_to_mps(speed: float, unit: str) -> float:
    if unit == "kmph":
        return float(speed) / KMH_PER_MPS
    return float(speed)


def csv_speed_field_names(unit: str) -> tuple[str, str]:
    if unit == "kmph":
        return "source_speed_kmph", "speed_kmph"
    return "source_speed_mps", "speed_mps"


def csv_headers(unit: str = "mps") -> list[str]:
    source_key, speed_key = csv_speed_field_names(unit)
    return [
        "sample_id",
        "batch_id",
        "filename",
        "vehicle_class",
        "trajectory_type",
        source_key,
        speed_key,
        "cpa_distance_m",
        "cpa_time_sec",
        "vehicle_length_m",
        "num_emitters",
        "pass_by_in_clip",
    ]


def sample_dir_name(index: int) -> str:
    return f"sample_{index:0{SAMPLE_ID_WIDTH}d}"


def output_wav_name(vehicle: str, speed_mps: float, *, unit: str = "mps") -> str:
    speed_display = mps_to_display(speed_mps, unit)
    speed_label = int(speed_display) if speed_display == int(speed_display) else speed_display
    suffix = "kmh" if unit == "kmph" else "mps"
    return f"{vehicle}_{speed_label}{suffix}.wav"


DEFAULT_BATCH_OUTPUT_DIR = "static/batch_outputs"
DEFAULT_BATCH_INPUT_DIR = "static/inputs"
DEFAULT_BATCH_NAME = "my_dataset_001"


def batch_root(base_dir: Path) -> Path:
    """Default batch output root (legacy). Prefer resolve_batch_output_root."""
    return base_dir / "static" / "batch_outputs"


def _resolve_relative_project_path(
    base_dir: Path,
    raw_path: str | None,
    *,
    default: str,
    label: str,
) -> Path:
    raw = (raw_path or default).strip().replace("\\", "/")
    if not raw:
        raw = default
    if Path(raw).is_absolute() or (len(raw) > 1 and raw[1] == ":"):
        raise ValueError(f"{label} must be a relative path from the project root (e.g. {default}).")
    parts = Path(raw).parts
    if ".." in parts:
        raise ValueError(f"{label} cannot contain '..'.")

    root = base_dir.resolve()
    resolved = (root / raw).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must stay inside the project directory.")
    return resolved


def resolve_batch_output_root(base_dir: Path, output_dir: str | None) -> Path:
    """Resolve a user-provided output folder relative to the project root."""
    return _resolve_relative_project_path(
        base_dir,
        output_dir,
        default=DEFAULT_BATCH_OUTPUT_DIR,
        label="Output folder",
    )


def resolve_input_root(base_dir: Path, input_dir: str | None = None) -> Path:
    """Resolve a user-provided input folder relative to the project root."""
    return _resolve_relative_project_path(
        base_dir,
        input_dir,
        default=DEFAULT_BATCH_INPUT_DIR,
        label="Input folder",
    )


def input_root(base_dir: Path) -> Path:
    return resolve_input_root(base_dir, None)


def to_project_relative(base_dir: Path, path: Path | str) -> str:
    """Store paths relative to the repo root (portable across machines)."""
    resolved = Path(path).resolve()
    root = base_dir.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix().replace("\\", "/")


def resolve_project_path(base_dir: Path, path: Path | str) -> Path:
    """Resolve a project-relative path, or pass through legacy absolute paths."""
    raw = Path(path)
    if raw.is_absolute() or (len(str(path)) > 1 and str(path)[1] == ":"):
        return raw.resolve()
    return (base_dir.resolve() / raw).resolve()
