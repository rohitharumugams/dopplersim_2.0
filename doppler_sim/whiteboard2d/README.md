# 2D Whiteboard-only frontend

Standalone site that exposes **only** the single-clip 2D Path Board.

- Does **not** change `app.py`, `doppler_sim/application.py`, or `templates/index.html`
- Full DopplerSim UI remains: `python app.py` (port 5003)
- This UI: `python whiteboard2d_app.py` (port **5004**)

```bash
python whiteboard2d_app.py
# → http://127.0.0.1:5004
```

Gunicorn: `gunicorn whiteboard2d_app:app`
