"""Diagnostic plots for the multisource (5.0) pass-by tab."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from doppler_sim.multisource.bootstrap import ensure_vendor_on_path

ensure_vendor_on_path()

from doppler_5dot0.sources.catalog import build_source_catalog, split_by_kind


def plot_source_catalog_layout(
    car_length_m: float,
    filename: str,
    plot_dir: Path,
) -> str:
    """Scatter the 14 catalog sources in the vehicle body frame."""
    catalog = build_source_catalog(car_length_m)
    tonal, tires, body, aero = split_by_kind(catalog)
    groups = [
        (tonal, "#dc2626", "tonal"),
        (tires, "#2563eb", "tire"),
        (body, "#16a34a", "body"),
        (aero, "#9333ea", "aero"),
    ]

    plt.figure(figsize=(10, 4))
    half = car_length_m / 2.0
    plt.plot([-half, half], [0, 0], color="#64748b", linewidth=3, zorder=1)
    for specs, color, kind in groups:
        xs = [s.dx for s in specs]
        ys = [s.dy for s in specs]
        plt.scatter(xs, ys, s=80, color=color, zorder=3, label=f"{kind} ({len(specs)})")
        for spec in specs:
            plt.annotate(
                spec.id.replace("_", " "),
                (spec.dx, spec.dy),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=6,
                color=color,
            )
    plt.title(f"5.0 source catalog — {len(catalog)} emitters (L={car_length_m:.1f} m)")
    plt.xlabel("Longitudinal dx (m)")
    plt.ylabel("Lateral dy (m)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)
    plt.axis("equal")
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return filename


def plot_component_waveforms(
    tracks: dict[str, np.ndarray],
    sr: int,
    filename: str,
    plot_dir: Path,
) -> str:
    """Stack key component waveforms (engine, intake, exhaust, tires, broadband, combined)."""
    keys = ["engine", "intake", "exhaust", "tires", "broadband", "combined"]
    colors = ["#dc2626", "#ea580c", "#f97316", "#2563eb", "#16a34a", "#7c3aed"]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 8), sharex=True)
    for ax, key, color in zip(axes, keys, colors):
        y = tracks.get(key)
        if y is None or len(y) == 0:
            ax.set_visible(False)
            continue
        t = np.arange(len(y)) / float(sr)
        ax.plot(t, y, color=color, linewidth=0.6)
        ax.set_ylabel(key, fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Multi-source component waveforms (5.0 combine)", fontsize=11)
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return filename
