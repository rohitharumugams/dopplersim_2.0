"""Entry point for the 2D whiteboard-only website.

Does not replace ``app.py`` (full DopplerSim UI). Run this process to serve
only the single-clip 2D path board::

    python whiteboard2d_app.py
    # → http://127.0.0.1:5004
"""

from __future__ import annotations

import os

from doppler_sim.whiteboard2d import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5004"))
    debug = os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "0").lower() in ("1", "true", "yes")
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=use_reloader)
