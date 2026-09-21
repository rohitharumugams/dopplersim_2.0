"""Standalone 2D Path Board UI (single-clip only).

This package is a separate Flask frontend for deployments that should expose
only the metered 2D whiteboard Pass-By flow. It reuses synthesis helpers from
``doppler_sim.application`` without modifying that module or ``templates/index.html``.

Run with::

    python whiteboard2d_app.py
"""

from __future__ import annotations

from doppler_sim.whiteboard2d.app import create_app

__all__ = ["create_app"]
