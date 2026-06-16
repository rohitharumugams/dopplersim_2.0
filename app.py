"""DopplerSim 2.0 entry point for `python app.py` and gunicorn `app:app`."""

from __future__ import annotations

import os

from doppler_sim import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5003"))
    debug = os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    app.run(debug=debug, host="0.0.0.0", port=port)
