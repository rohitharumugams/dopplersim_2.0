"""DopplerSim 2.0 entry point for `python app.py` and gunicorn `app:app`."""

from __future__ import annotations

import os

from doppler_sim import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5003"))
    debug = os.environ.get("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    # The auto-reloader restarts the whole process on any .py change, which
    # kills in-progress batch generation (it runs in a background thread) and
    # leaves clips unsaved. Disable it by default; opt back in with
    # FLASK_USE_RELOADER=1 if you specifically want live code reloading.
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "0").lower() in ("1", "true", "yes")
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=use_reloader)
