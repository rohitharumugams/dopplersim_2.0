"""Cyclic integer sampler with resume state."""

from __future__ import annotations

import json
import math
from pathlib import Path


def distribute_even_counts(total: int, n_groups: int) -> list[int]:
    """Split *total* across *n_groups* as evenly as possible (remainder +1 to first groups)."""
    if n_groups <= 0:
        return []
    total = max(0, int(total))
    base, remainder = divmod(total, n_groups)
    return [base + (1 if i < remainder else 0) for i in range(n_groups)]


def evenly_spaced_speeds(speed_min: float, speed_max: float, count: int) -> list[float]:
    """Evenly spaced speeds on [min, max] with uniform repetition when oversubscribed."""
    if count <= 0:
        return []
    lo = float(speed_min)
    hi = float(speed_max)
    if hi < lo:
        lo, hi = hi, lo

    if count == 1:
        return [_round_speed(lo)]

    width = hi - lo
    if width < 1e-12:
        return [_round_speed(lo)] * count

    lattice_size = _speed_lattice_size(lo, hi, count)
    lattice = [_round_speed(lo + i * (width / (lattice_size - 1))) for i in range(lattice_size)]

    if count == lattice_size:
        return lattice
    if count < lattice_size:
        return lattice[:count]
    return _uniform_expand_speeds(lattice, count)


def _round_speed(value: float) -> float:
    return round(float(value), 4)


def _speed_lattice_size(lo: float, hi: float, count: int) -> int:
    """Unique lattice points spanning the interval before repetition."""
    width = hi - lo
    if count <= 1:
        return 1
    # All unique when each vehicle's allocation fits on one evenly spaced lattice.
    target_step = width / (count - 1)
    # Minimum step before we cap the lattice and repeat uniformly (sub-centimeter per second).
    min_step = 1e-4
    if target_step >= min_step:
        return count
    max_unique = max(2, int(math.floor(width / min_step)) + 1)
    return min(count, max_unique)


def _uniform_expand_speeds(lattice: list[float], count: int) -> list[float]:
    """Repeat lattice values so each speed appears equally often (±1)."""
    n = len(lattice)
    if n == 0:
        return []
    if count <= n:
        return lattice[:count]
    base, remainder = divmod(count, n)
    expanded: list[float] = []
    for i, speed in enumerate(lattice):
        expanded.extend([speed] * (base + (1 if i < remainder else 0)))
    return expanded


class CyclicSampler:
    def __init__(self, low: int, high: int, step: int = 1, offset: int = 0):
        if high < low:
            low, high = high, low
        self.low = low
        self.high = high
        self.step = max(1, step)
        self.offset = offset % self._span()

    def _span(self) -> int:
        return self.high - self.low + 1

    def next(self) -> int:
        value = self.low + (self.offset % self._span())
        self.offset = (self.offset + self.step) % self._span()
        return value

    def to_dict(self) -> dict:
        return {
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CyclicSampler":
        return cls(
            int(data["low"]),
            int(data["high"]),
            int(data.get("step", 1)),
            int(data.get("offset", 0)),
        )


class SamplerBank:
    def __init__(self) -> None:
        self._samplers: dict[str, CyclicSampler] = {}

    def get(self, key: str, low: int, high: int) -> CyclicSampler:
        if key not in self._samplers:
            self._samplers[key] = CyclicSampler(low, high)
        return self._samplers[key]

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({k: s.to_dict() for k, s in self._samplers.items()}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "SamplerBank":
        bank = cls()
        if not path.exists():
            return bank
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, payload in data.items():
            bank._samplers[key] = CyclicSampler.from_dict(payload)
        return bank
