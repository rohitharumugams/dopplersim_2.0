"""Procedural 2D curved paths for batch generation (planning only, no audio physics)."""

from __future__ import annotations

import math
import random

import numpy as np
from scipy.interpolate import PchipInterpolator


def _max_turn_deg(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return 0.0
    max_turn = 0.0
    for i in range(1, len(xy) - 1):
        v1 = xy[i] - xy[i - 1]
        v2 = xy[i + 1] - xy[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        turn = math.degrees(math.acos(cos_a))
        max_turn = max(max_turn, turn)
    return max_turn


def _path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def _dedupe_polyline(xy: np.ndarray, *, min_step_m: float = 0.05) -> np.ndarray:
    if len(xy) < 2:
        return xy
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > float(min_step_m)
    out = xy[keep]
    return out if len(out) >= 2 else xy[:2]


def _chaikin_corner_cut(xy: np.ndarray, *, iterations: int = 1) -> np.ndarray:
    """Round polyline corners (planning-only); endpoints preserved."""
    pts = np.asarray(xy, dtype=np.float64)
    if len(pts) < 3 or iterations <= 0:
        return pts
    for _ in range(int(iterations)):
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            out.append(0.75 * p0 + 0.25 * p1)
            out.append(0.25 * p0 + 0.75 * p1)
        out.append(pts[-1])
        pts = np.asarray(out, dtype=np.float64)
    return pts


def _random_waypoints(
    rng: random.Random,
    *,
    x_start: float,
    x_end: float,
    y_min: float,
    y_max: float,
    n_waypoints: int,
) -> np.ndarray:
    span_x = float(x_end - x_start)
    min_seg_x = span_x / max(n_waypoints * 4, 1)

    xs = [float(x_start)]
    for _ in range(n_waypoints - 2):
        lo = xs[-1] + min_seg_x
        hi = float(x_end) - min_seg_x * (n_waypoints - len(xs) - 1)
        if lo >= hi:
            lo = xs[-1] + 1e-3
            hi = lo + 1e-3
        xs.append(float(rng.uniform(lo, hi)))
    xs.append(float(x_end))

    y_span = float(y_max - y_min)
    max_step_y = max(y_span * 0.35, 1.0)
    ys = [float(rng.uniform(y_min, y_max))]
    for _ in range(n_waypoints - 1):
        ys.append(float(np.clip(ys[-1] + rng.uniform(-max_step_y, max_step_y), y_min, y_max)))

    return np.column_stack([xs, ys]).astype(np.float64)


def _smooth_path_through_waypoints(
    waypoints: np.ndarray,
    *,
    samples_per_span: int,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Monotonic-x smooth path: PCHIP spline y(x) through rounded random knots."""
    wp = _dedupe_polyline(waypoints, min_step_m=0.05)
    if len(wp) >= 4:
        wp = _chaikin_corner_cut(wp, iterations=1)
        wp = _dedupe_polyline(wp, min_step_m=0.05)
    if len(wp) < 2:
        return wp

    xs = wp[:, 0]
    ys = wp[:, 1]
    n_spans = max(1, len(wp) - 1)
    n_dense = max(n_spans * max(6, int(samples_per_span)), 2)
    x_dense = np.linspace(float(xs[0]), float(xs[-1]), n_dense, dtype=np.float64)

    if len(wp) >= 3:
        y_dense = PchipInterpolator(xs, ys, extrapolate=False)(x_dense)
    else:
        y_dense = np.interp(x_dense, xs, ys)

    y_dense = np.clip(y_dense, float(y_min), float(y_max))
    return np.column_stack([x_dense, y_dense]).astype(np.float64)


def random_curved_path_2d(
    rng: random.Random,
    *,
    x_start: float = -60.0,
    x_end: float = 60.0,
    y_min: float = 5.0,
    y_max: float = 30.0,
    n_segments: int = 3,
    points_per_segment: int = 16,
    max_turn_deg: float = 40.0,
    min_length_m: float = 40.0,
    mic_x: float = 0.0,
    mic_y: float = 0.0,
    max_attempts: int = 128,
) -> np.ndarray:
    """Return a smooth random path (N, 2) above the x-axis.

    Random waypoints define y(x) knots; a shape-preserving PCHIP spline rounds corners
    while keeping x monotonic. Turn angle is checked on the dense output polyline.
    """
    if x_end <= x_start:
        raise ValueError("x_end must be greater than x_start")
    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min")
    if y_min < 0.0:
        raise ValueError("y_min must be >= 0 for paths above the x-axis")

    n_waypoints = max(3, int(n_segments) + 2)
    samples_per_span = max(6, int(points_per_segment))
    max_turn_deg = float(max(5.0, max_turn_deg))

    for _ in range(max_attempts):
        waypoints = _random_waypoints(
            rng,
            x_start=x_start,
            x_end=x_end,
            y_min=y_min,
            y_max=y_max,
            n_waypoints=n_waypoints,
        )

        xy = _smooth_path_through_waypoints(
            waypoints,
            samples_per_span=samples_per_span,
            y_min=y_min,
            y_max=y_max,
        )
        xy = _dedupe_polyline(xy, min_step_m=0.25)
        if len(xy) < 2:
            continue
        if np.any(xy[:, 1] < y_min - 1e-6):
            continue
        if _max_turn_deg(xy) > max_turn_deg:
            continue
        if _path_length(xy) < min_length_m:
            continue

        r = np.sqrt((xy[:, 0] - mic_x) ** 2 + (xy[:, 1] - mic_y) ** 2)
        if float(np.min(r)) > max(y_max, 5.0) * 2.0:
            continue

        return xy

    raise RuntimeError(
        "Could not sample an acceptable random path "
        f"(x={x_start:.1f}..{x_end:.1f}, y={y_min:.1f}..{y_max:.1f}, "
        f"max_turn={max_turn_deg:.1f}°)"
    )
