"""Body-layout emitters and the output figures that show them.

Vehicle frame: axle midpoint, +X forward, +Y left, +Z up.
Tire emitters sit on the contact patch (z = 0). Engine and exhaust use the
researched acoustic centers. The path point is the axle midpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from doppler_sim.path2d.synthesis import rigid_emitter_xy

TIRE_COLORS = {
    "FL": "#1e293b",
    "FR": "#64748b",
    "RL": "#0f766e",
    "RR": "#0284c7",
}
ENGINE_COLOR = "#dc2626"
EXHAUST_COLOR = "#2563eb"
LINE_COLOR = "#dc2626"
AXLE_COLOR = "#94a3b8"

PROPAGATION_PLOTS = (
    ("t_r", "09_retarded_time.png", "Retarded Time vs Observer Time", "Retarded time (s)"),
    ("v_r", "10_radial_velocity.png", "Radial Velocity vs Observer Time", "Radial velocity (m/s)"),
    ("alpha", "11_doppler_factor.png", "Doppler Factor vs Observer Time", "α = c / (c + v_r)"),
    ("R", "12_propagation_distance.png", "Propagation Distance vs Observer Time", "Distance R (m)"),
)


def linear_emitters(length_m: float, num_emitters: int) -> list[dict[str, Any]]:
    n = max(int(num_emitters), 1)
    if n == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-float(length_m) / 2.0, float(length_m) / 2.0, n)
    return [
        {
            "name": f"E{i + 1}",
            "kind": "line",
            "color": LINE_COLOR,
            "xyz": [float(x0), 0.0, 0.0],
        }
        for i, x0 in enumerate(offsets)
    ]


def body_emitters(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    dims = geometry["dims"]
    sources = geometry["sources"]
    engine = sources["engine"]
    exhaust = sources["exhaust"]
    wb = float(dims["WB"])
    return [
        {"name": "FL", "kind": "tire", "color": TIRE_COLORS["FL"], "xyz": [wb / 2.0, float(dims["FT"]) / 2.0, 0.0]},
        {"name": "FR", "kind": "tire", "color": TIRE_COLORS["FR"], "xyz": [wb / 2.0, -float(dims["FT"]) / 2.0, 0.0]},
        {"name": "RL", "kind": "tire", "color": TIRE_COLORS["RL"], "xyz": [-wb / 2.0, float(dims["RT"]) / 2.0, 0.0]},
        {"name": "RR", "kind": "tire", "color": TIRE_COLORS["RR"], "xyz": [-wb / 2.0, -float(dims["RT"]) / 2.0, 0.0]},
        {
            "name": "Engine",
            "kind": "engine",
            "color": ENGINE_COLOR,
            "xyz": [float(engine[0]), float(engine[1]), float(engine[2])],
        },
        {
            "name": "Exhaust",
            "kind": "exhaust",
            "color": EXHAUST_COLOR,
            "xyz": [float(exhaust[0]), float(exhaust[1]), float(exhaust[2])],
        },
    ]


def offsets_xyz(emitters: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([e["xyz"] for e in emitters], dtype=np.float64)


def _save(filename: str, plot_dir: Path) -> str:
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return filename


def plot_body_diagram(
    emitters: list[dict[str, Any]],
    dims: dict[str, float],
    filename: str,
    plot_dir: Path,
) -> str:
    """Plan and side view of the six body emitters, in the axle-midpoint frame."""
    wb = float(dims["WB"])
    x_front = wb / 2.0 + float(dims["FO"])
    x_rear = -(wb / 2.0 + float(dims["RO"]))
    half_w = float(dims["W"]) / 2.0
    xs = [e["xyz"][0] for e in emitters]
    ys = [e["xyz"][1] for e in emitters]
    zs = [e["xyz"][2] for e in emitters]
    colors = [e["color"] for e in emitters]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax_top, ax_side = axes
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="#e2e8f0", linewidth=0.8)
        ax.axhline(0.0, color="#cbd5e1", linewidth=0.8)
        ax.axvline(0.0, color="#cbd5e1", linewidth=0.8)

    body_x = [x_rear, x_front, x_front, x_rear, x_rear]
    body_y = [-half_w, -half_w, half_w, half_w, -half_w]
    ax_top.plot(body_x, body_y, color="#94a3b8", linewidth=1.4)
    ax_top.scatter(xs, ys, s=90, c=colors, zorder=3, edgecolors="white", linewidths=0.6)
    for emitter, x, y in zip(emitters, xs, ys):
        ax_top.annotate(
            emitter["name"],
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#334155",
        )
    ax_top.set_xlabel("Forward x (m)")
    ax_top.set_ylabel("Left y (m)")
    ax_top.set_title("Top")

    ax_side.plot([x_rear, x_front], [0.0, 0.0], color="#94a3b8", linewidth=3.0)
    side_groups: dict[tuple[float, float], list[str]] = {}
    side_color: dict[tuple[float, float], str] = {}
    for emitter, x, z in zip(emitters, xs, zs):
        key = (round(float(x), 3), round(float(z), 3))
        side_groups.setdefault(key, []).append(emitter["name"])
        side_color[key] = emitter["color"]
    ax_side.scatter(
        [key[0] for key in side_groups],
        [key[1] for key in side_groups],
        s=90,
        c=[side_color[key] for key in side_groups],
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
    )
    for key, names in side_groups.items():
        ax_side.annotate(
            " ".join(names),
            key,
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#334155",
        )
    ax_side.set_xlabel("Forward x (m)")
    ax_side.set_ylabel("Height z (m)")
    ax_side.set_title("Side")
    fig.suptitle("Body emitters", fontsize=12)
    fig.patch.set_facecolor("#ffffff")
    return _save(filename, plot_dir)


def plot_path_emitters(
    path_xy: np.ndarray,
    traj: dict[str, np.ndarray],
    mic_xy: tuple[float, float],
    emitters: list[dict[str, Any]],
    filename: str,
    plot_dir: Path,
) -> str:
    """Drawn path with each emitter's ground track and its pose at closest approach."""
    path_xy = np.asarray(path_xy, dtype=np.float64)
    mx, my = float(mic_xy[0]), float(mic_xy[1])
    x = np.asarray(traj["x"], dtype=np.float64)
    y = np.asarray(traj["y"], dtype=np.float64)
    step = max(len(x) // 250, 1)
    xe, ye = rigid_emitter_xy(x, y, traj["tx"], traj["ty"], offsets_xyz(emitters))
    cpa = int(np.argmin((x - mx) ** 2 + (y - my) ** 2))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    ax.plot(path_xy[:, 0], path_xy[:, 1], color="#94a3b8", linewidth=1.5, alpha=0.7, label="Drawn path")
    ax.plot(x, y, color="#2563eb", linewidth=1.6, label="Axle path")
    seen: set[str] = set()
    for emitter, ex, ey in zip(emitters, xe, ye):
        kind = emitter["kind"]
        label = {"tire": "Tires", "engine": "Engine", "exhaust": "Exhaust", "line": "Emitters"}.get(kind, emitter["name"])
        ax.plot(
            ex[::step],
            ey[::step],
            color=emitter["color"],
            linewidth=1.0,
            alpha=0.85,
            label=label if kind not in seen else None,
        )
        seen.add(kind)
        ax.scatter([ex[cpa]], [ey[cpa]], s=36, c=emitter["color"], zorder=5, edgecolors="white", linewidths=0.4)
    ax.scatter([path_xy[0, 0]], [path_xy[0, 1]], c="#16a34a", s=40, zorder=6, label="Start")
    ax.scatter([path_xy[-1, 0]], [path_xy[-1, 1]], c="#dc2626", s=40, zorder=6, label="End")
    ax.scatter([mx], [my], c="#111827", marker="x", s=80, zorder=6, label="Microphone")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, which="major", color="#cbd5e1", linewidth=0.8)
    ax.axhline(0.0, color="#64748b", linewidth=0.8)
    ax.axvline(0.0, color="#64748b", linewidth=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Path and emitter points")
    ax.legend(loc="best", fontsize=8)
    return _save(filename, plot_dir)


def _heading_at(t_src: np.ndarray, tx: np.ndarray, ty: np.ndarray, t_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tx_q = np.interp(t_q, t_src, tx)
    ty_q = np.interp(t_q, t_src, ty)
    norm = np.maximum(np.hypot(tx_q, ty_q), 1e-9)
    return tx_q / norm, ty_q / norm


def write_emitter_animation(
    render_dir: Path,
    *,
    traj: dict[str, np.ndarray],
    path_xy: np.ndarray,
    mic_xy: tuple[float, float],
    emitters: list[dict[str, Any]],
    fps: float = 30.0,
) -> None:
    t = np.asarray(traj["t"], dtype=np.float64)
    if len(t) < 2:
        return
    duration = float(t[-1])
    n_frames = max(int(np.ceil(duration * fps)) + 1, 2)
    t_q = np.linspace(0.0, duration, n_frames)
    x_q = np.interp(t_q, t, traj["x"])
    y_q = np.interp(t_q, t, traj["y"])
    tx_q, ty_q = _heading_at(t, traj["tx"], traj["ty"], t_q)
    xe, ye = rigid_emitter_xy(x_q, y_q, tx_q, ty_q, offsets_xyz(emitters))
    payload = {
        "fps": float(fps),
        "duration_s": duration,
        "mic": [float(mic_xy[0]), float(mic_xy[1])],
        "path": np.asarray(path_xy, dtype=np.float64).tolist(),
        "t": t_q.tolist(),
        "x": x_q.tolist(),
        "y": y_q.tolist(),
        "world": {"xmin": -50.0, "xmax": 50.0, "ymin": -35.0, "ymax": 35.0},
        "emitters": [
            {
                "name": emitter["name"],
                "kind": emitter["kind"],
                "color": emitter["color"],
                "x": xe[i].tolist(),
                "y": ye[i].tolist(),
            }
            for i, emitter in enumerate(emitters)
        ],
    }
    (render_dir / "path_animation.json").write_text(json.dumps(payload), encoding="utf-8")


def _sample_index(n: int, max_points: int) -> np.ndarray:
    step = max(int(n) // int(max_points), 1)
    idx = np.arange(0, n, step)
    if len(idx) == 0:
        return np.array([0], dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _json_series(values: np.ndarray, idx: np.ndarray) -> list[float | None]:
    sampled = np.asarray(values, dtype=np.float64)[idx]
    return [None if not np.isfinite(v) else float(v) for v in sampled]


def pack_propagation(
    t_obs: np.ndarray,
    emitters: list[dict[str, Any]],
    curves: list[dict[str, np.ndarray]],
    axle: dict[str, np.ndarray],
    max_points: int = 800,
) -> dict[str, Any]:
    """Downsample per-emitter Doppler curves so the figures can be redrawn later."""
    t = np.asarray(t_obs, dtype=np.float64)
    idx = _sample_index(len(t), max_points)
    packed = []
    for emitter, curve in zip(emitters, curves):
        packed.append(
            {
                "name": emitter["name"],
                "color": emitter["color"],
                "t_r": _json_series(curve["t_r"], idx),
                "R": _json_series(curve["R"], idx),
                "v_r": _json_series(curve["v_r"], idx),
                "alpha": _json_series(curve["alpha"], idx),
            }
        )
    return {
        "t": [float(v) for v in t[idx]],
        "emitters": packed,
        "axle": {
            "t_r": _json_series(axle["t_r"], idx),
            "R": _json_series(axle["R"], idx),
            "v_r": _json_series(axle["v_r"], idx),
            "alpha": _json_series(axle["alpha"], idx),
        },
    }


def _finite_series(values: list[float | None]) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)


def plot_propagation_curves(propagation: dict[str, Any], plot_dir: Path) -> None:
    """Replace the single-line Doppler figures with one curve per body emitter."""
    t = np.asarray(propagation["t"], dtype=np.float64)
    axle = propagation.get("axle")
    for key, filename, title, ylabel in PROPAGATION_PLOTS:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for emitter in propagation["emitters"]:
            y = _finite_series(emitter[key])
            mask = np.isfinite(y)
            ax.plot(t[mask], y[mask], color=emitter["color"], linewidth=1.15, label=emitter["name"])
        if axle is not None:
            y = _finite_series(axle[key])
            mask = np.isfinite(y)
            ax.plot(
                t[mask],
                y[mask],
                color=AXLE_COLOR,
                linewidth=1.0,
                linestyle="--",
                label="Axle",
            )
        ax.set_title(title)
        ax.set_xlabel("Observer time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=4, loc="best")
        fig.patch.set_facecolor("#ffffff")
        _save(filename, plot_dir)


def save_emitter_view(
    render_dir: Path,
    *,
    layout: str,
    mic_z: float,
    mic_xy: tuple[float, float],
    path_xy: np.ndarray,
    traj: dict[str, np.ndarray],
    emitters: list[dict[str, Any]],
    dims: dict[str, float] | None,
    propagation: dict[str, Any] | None = None,
) -> None:
    t = np.asarray(traj["t"], dtype=np.float64)
    step = max(len(t) // 400, 1)
    idx = np.arange(0, len(t), step)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)
    payload = {
        "layout": layout,
        "mic_z": float(mic_z),
        "mic_xy": [float(mic_xy[0]), float(mic_xy[1])],
        "path": np.asarray(path_xy, dtype=np.float64).tolist(),
        "t": t[idx].tolist(),
        "x": np.asarray(traj["x"], dtype=np.float64)[idx].tolist(),
        "y": np.asarray(traj["y"], dtype=np.float64)[idx].tolist(),
        "tx": np.asarray(traj["tx"], dtype=np.float64)[idx].tolist(),
        "ty": np.asarray(traj["ty"], dtype=np.float64)[idx].tolist(),
        "emitters": emitters,
        "dims": dims,
    }
    if propagation is not None:
        payload["propagation"] = propagation
    (render_dir / "emitter_view.json").write_text(json.dumps(payload), encoding="utf-8")


def replot_emitter_views(render_dir: Path, plot_dir: Path, observer_name: str, geometry_name: str) -> None:
    path = render_dir / "emitter_view.json"
    if not path.is_file():
        return
    view = json.loads(path.read_text(encoding="utf-8"))
    traj = {
        "x": np.asarray(view["x"], dtype=np.float64),
        "y": np.asarray(view["y"], dtype=np.float64),
        "tx": np.asarray(view["tx"], dtype=np.float64),
        "ty": np.asarray(view["ty"], dtype=np.float64),
    }
    plot_path_emitters(
        np.asarray(view["path"], dtype=np.float64),
        traj,
        (float(view["mic_xy"][0]), float(view["mic_xy"][1])),
        view["emitters"],
        observer_name,
        plot_dir,
    )
    if view.get("layout") == "body" and view.get("dims"):
        plot_body_diagram(view["emitters"], view["dims"], geometry_name, plot_dir)
    if view.get("propagation"):
        plot_propagation_curves(view["propagation"], plot_dir)
