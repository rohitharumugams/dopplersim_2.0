"""Shared matplotlib layout for spectrogram panel PNG exports."""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib.pyplot as plt

SPECG_PANEL_FIGSIZE = (6.2, 3.4)
SPECG_PANEL_DPI = 110
SPECG_PANEL_FACECOLOR = "#0f1117"


def spec_panel_subplots(*, ax_facecolor: str = "#0f1117"):
    fig, ax = plt.subplots(figsize=SPECG_PANEL_FIGSIZE, facecolor=SPECG_PANEL_FACECOLOR)
    ax.set_facecolor(ax_facecolor)
    return fig, ax


def add_spec_colorbar(fig: plt.Figure, ax: plt.Axes, mappable) -> None:
    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)


def finalize_spec_panel(fig: plt.Figure) -> None:
    fig.subplots_adjust(left=0.09, right=0.88, top=0.88, bottom=0.14)


def spec_panel_to_b64(fig: plt.Figure) -> str:
    finalize_spec_panel(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=SPECG_PANEL_DPI, facecolor=SPECG_PANEL_FACECOLOR)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
