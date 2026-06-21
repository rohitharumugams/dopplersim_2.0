"""Ensure vendored doppler_4dot0 / doppler_5dot0 packages are importable for Multisource."""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR: Path | None = None


def ensure_vendor_on_path() -> Path:
    """Add multisource/vendor to sys.path so `doppler_4dot0` / `doppler_5dot0` resolve locally."""
    global _VENDOR_DIR
    vendor = Path(__file__).resolve().parent / "vendor"
    vendor_str = str(vendor)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
    _VENDOR_DIR = vendor
    return vendor


def ensure_monorepo_on_path() -> Path:
    """Backward-compatible alias — now loads vendored copies, not the monorepo."""
    return ensure_vendor_on_path()
