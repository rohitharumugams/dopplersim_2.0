# DopplerSim 2.0

Physics-based vehicle pass-by audio re-rendering. Upload a recorded pass-by, specify original and target geometry, and synthesize a new pass-by with retarded-time Doppler physics. **Batch Generation** builds ML-ready datasets by sweeping vehicles and speeds through the same synthesis pipeline.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Add source WAVs + .txt sidecars to static/inputs/
python app.py
# → http://127.0.0.1:5003 → Batch Generation tab
```

## Features

The web app has five tabs:

| Tab | Purpose |
|-----|---------|
| **Batch Generation** | Build datasets from a vehicle catalog — parallel workers, resume, progress, flexible export |
| **Pass-By Simulator** | Single-clip re-render with full diagnostic plots |
| **Spectrogram Explorer** | Compare STFT, mel, CQT, reassigned, CWT, SSQ, chroma, and related views |
| **Audio Comparison** | Side-by-side metrics for two clips |
| **Experimental TF** | STFT vs reassigned vs Wigner–Ville (≤30 s, research preview) |

Reassignment and Wigner–Ville are **visualization only** — they do not alter Pass-By audio synthesis.

## Requirements

- Python 3.10+
- See `requirements.txt` (Flask, NumPy, SciPy, librosa, soundfile, matplotlib, PyWavelets, ssqueezepy)

## Setup

```bash
git clone <your-repo-url>
cd DopplerSim_2.0
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data paths (relative to project root)

| Path | Purpose |
|------|---------|
| `static/inputs/` | Source pass-by WAVs for batch generation (+ `.txt` sidecars) |
| `static/batch_outputs/` | Generated batch datasets (default output root) |
| `uploads/` | Pass-By Sim uploads (runtime) |
| `renders/` | Render session state (runtime) |
| `static/generated/`, `static/plots/`, `static/compare/` | Other runtime outputs (gitignored) |

Only `.gitkeep` placeholders are tracked under `static/inputs/` and `static/batch_outputs/`; all generated content stays local.

### Batch source clips

Place recordings in `static/inputs/` (or a custom input folder set in the UI):

**Filename patterns** (vehicle name + speed):

- `VehicleName_59.wav` — bare numeric suffix = **km/h**
- `VehicleName_59kmh.wav` or `VehicleName_36kmph.wav`
- `VehicleName_10mps.wav` or `VehicleName_10_mps.wav`

**Sidecar (recommended):** `VehicleName_59.txt` with:

```text
54.0 5.65
```

Format: `speed_kmh t_cpa1_s` (space- or comma-separated). When present, the sidecar overrides the filename speed and supplies source geometry `t_CPA,1`.

Example: `CitroenC4Picasso_59.wav` + `CitroenC4Picasso_59.txt`

## Run

```bash
python app.py
```

Open [http://127.0.0.1:5003](http://127.0.0.1:5003).

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `5003` | HTTP port |
| `FLASK_DEBUG` | `1` | Flask debug mode (error pages) |
| `FLASK_USE_RELOADER` | `0` | Auto-reload on code changes. **Keep off** during batch jobs — the reloader kills in-progress generation. |

**Production:** `gunicorn app:app` — see `deploy/` for a systemd example (GCP/Ubuntu). If nginx sits in front, set `client_max_body_size 100m` (see `deploy/nginx-dopplersim.conf.example`) or uploads hit **413 Request Entity Too Large**. Flask `MAX_CONTENT_LENGTH` is 100 MB.

---

## Batch Generation

### Workflow

1. Open **Batch Generation**.
2. Set **batch name**, **total clips**, and **parallel workers** (default: CPU count − 1).
3. Optionally change **input** / **output** directories (defaults: `static/inputs`, `static/batch_outputs`).
4. Select vehicles; choose a **source speed** per vehicle when multiple clips exist.
5. Configure speed range (v₂), CPA distance (h₂), t_CPA,2, emitters, clip length, spectrograms, and export options.
6. Click **Start batch** (confirm overwrite if the folder exists).
7. Monitor progress (completed / total, active workers, ETA, wall time). Use **Resume batch** after interruption.

### Export options

| Option | Description |
|--------|-------------|
| **Phase 1 export** (checkbox; formerly “Simple generate”) | WAV + spectrograms + Phase 1 `metadata/` state pairing; skips trajectory plot and propagation `kinematics.npy`. Legacy `velocity.npy` / `cpa.npy` still written. |
| **Spectrogram NPY** | Per-type `.npy` arrays in `spectrograms/` (default on: STFT, narrowband, log-mel, CQT, reassigned). |
| **Spectrogram PNG** | Per-type panel images — independent checkboxes; requires that type's NPY to be selected. |
| **Combined PNG** | Single comparison image of all selected PNG types (off by default). |
| **Speed unit** | Enter and label speeds in **m/s** or **km/h** (physics always uses m/s internally). |
| **Parallel workers** | Process pool — one clip per worker process on its own CPU core. |

**Available spectrogram types:** STFT, wideband, narrowband, mel, log-mel, CQT, wavelet, reassigned, synchrosqueezed, chromagram.

### Sampling (deterministic, seed-controlled)

All variation is **reproducible** from `seed` (default 42) and saved in `sampler_state.json` for resume:

| Parameter | Rule |
|-----------|------|
| Vehicle allocation | Clips split **evenly** across selected vehicles (e.g. 400 clips × 4 cars → 100 each). |
| Render speed v₂ | **Evenly spaced** from min to max; uniform repetition when oversubscribed. |
| CPA distance h₂ | **Cyclic sweep** through [min, max] in 0.1 m steps. |
| t_CPA,2 | **Cyclic sweep** through [min, max] in 0.01 s steps. |

Set min = max on distance or t_CPA,2 for a fixed value. Total clips should be divisible by vehicle count for perfectly even per-vehicle counts.

### Output layout

**Full mode** (Phase 1 export checkbox off):

```
static/batch_outputs/<batch_name>/
  audio_clips/sample_0000001/
    <vehicle>_<speed>mps.wav       # or _<speed>kmh.wav
    trajectory_plot.png
    spectrograms/                  # .npy (+ optional .png); STFT always included (Phase 1 A)
    metadata/
      state.npy                    # (N,4) s(t)=[x,vx,y,vy] at WAV sample rate
      state_times.npy
      state_frames.npy             # (T,4) s aligned to STFT frame centers
      frame_times.npy              # (T,) — same T as spectrograms/stft.npy
      acoustic_state.npy           # (T,4) secondary A_t features
      range_m.npy, radial_velocity_mps.npy, speed_series_mps.npy, …
      cpa_time.npy, cpa_distance_m.npy, direction.npy  # derived from s
      phase1_schema.json           # documents A(1:T) → s(1:T)
      kinematics.npy, labels.npy, simulation_parameters.json, …
  dataset.csv
  …
```

**Phase 1 learning pair:** `spectrograms/stft.npy` (A) ↔ `metadata/state_frames.npy` (s) via `frame_times.npy`.
CPA / speed / distance / direction are derived from `s`, not independent physics labels.

**CPA labels:** `cpa_time_sec` / `cpa_distance_m` are **frame-derived from instantaneous centerline range** (ML primary). Plan geometry is stored as `cpa_time_plan_sec` / `cpa_distance_plan_m` in `dataset.csv` and `labels.npy`. Acoustic loudness peaks near **retarded** min-`R` (≈ `t_CPA₂ + h/c`, tens of ms for typical `h`), not exactly the Phase 1 derived time. `direction.npy` is `+1`/`-1`. Centerline `s` is not identical to multi-emitter retarded-time synthesis.

**Phase 1 export mode** (checkbox on — skips trajectory plot / `kinematics.npy` only):

```
audio_clips/sample_0000001/
  <vehicle>_<speed>mps.wav
  velocity.npy                     # scalar display-unit speed (legacy)
  cpa.npy                          # CPA time derived from s on the STFT grid
  spectrograms/                    # includes STFT
  metadata/                        # same Phase 1 state / schema as full mode
```

Resume skips any `sample_*` folder that already contains a `.wav` file.

---

## Pass-By Simulator

1. Upload a mono WAV pass-by recording.
2. Set **original geometry** `(v₁, h₁, t_CPA,1)` for inverting the upload.
3. Set **vehicle model** (length `L`, emitter count `N`).
4. Set **render geometry** `(v₂, h₂, t_CPA,2, T_out)`.
5. Optionally enable **Include reassigned spectrograms** (diagnostic plots only).
6. Click **Generate** — WAV, plots, and bundle download.

Outputs go under `static/` and `renders/`. Each render also writes a **Phase 1 package** at `renders/<id>/phase1/` (same `A`/`s` layout as batch). **Download bundle** includes that `phase1/` folder.

### 2D / 3D Path Board

Draw a polyline on the Path Board; synthesis follows that centerline at constant speed with retarded time and `1/R`. Phase 1 export uses the **drawn path**, not a straight `(v₂, h₂, t_CPA₂)` fit:

- `metadata/state_frames.npy` — mic-centric centerline `[x, vx, y, vy]` (3D adds `z, vz`)
- `metadata/polar_state.npy` — recommended ML primary for curves (`r, r_dot, θ, θ_dot`)
- `metadata/canonical_state_frames.npy` — CPA on +y, `vx ≥ 0` at CPA
- `metadata/phase1_schema.json` — `"path_type": "free_path_2d"` or `"free_path_3d"`

**Note:** Path Board bundles generated before this fix used straight pass-by labels; regenerate for ML use.

---

## Physics (short)

- Vehicle geometry: `x(t) = v(t − t_CPA) + x₀`, range `R = √(x² + h²)`.
- Retarded time: solve `c(t − t_r) = R(t_r)` with geometric root selection.
- **Analysis:** per STFT frame, undo spreading (`×R`) and Doppler (`f_src = f/α` with `α = c/(c+v_r)`, `v_r` receding-positive); average to an intrinsic PSD per emitter.
- **Synthesis:** colored noise from that PSD; `s_obs(t) = s_src(t_r(t)) / R(t)`; sum `N` emitters with `1/√N` scaling. Amplitude follows geometric spreading at render geometry (`t_CPA₂`, `h₂`) — **not** the upload’s recorded envelope timing.
- **Phase 1 state (batch labels):** instantaneous centerline `s(t)=[x,ẋ,y,ẏ]` with `x=v(t−t_CPA)`, `y=h`, `ẋ=v`, `ẏ=0`. Derived CPA is `argmin √(x²+y²)` on that clock. Audio uses retarded time, so the loudness peak sits near observer min-`R` (≈ `t_CPA + h/c`). `metadata/kinematics.npy` (full export) stores the retarded-time propagation view separately.

Batch and single-clip Doppler Pass-By modes share the same `render_pass_by` backend.

Sanity check: `python scripts/verify_phase1_state.py` (straight Phase 1 kinematics, acoustic CPA regression, L-shaped free-path labels).

---

## Project layout

```
app.py                    Entry point
doppler_sim/
  application.py          Flask routes + Pass-By physics
  spec_panel_style.py     Shared spectrogram panel styling
  batch/
    catalog.py            Scan inputs/, parse filenames + sidecars
    planner.py            Build sample plan from BatchConfig
    sampler.py            Cyclic samplers (distance, t_CPA,2)
    runner.py             Async job, process pool, progress, resume
    pipeline.py             Per-clip synthesis wrapper
    features.py           WAV, spectrograms, metadata export
    phase1_state.py       Phase 1 s(t)=[x,vx,y,vy] + derived quantities
    constants.py            Paths, CSV headers, speed units
    vehicle_metadata.py     Display names for vehicles
  specg/explorer.py       Spectrogram Explorer + batch spectrogram export
  tf/
    analysis.py           Reassignment atoms and plots
    experimental.py       Wigner–Ville comparison tab
templates/index.html      Unified UI (all tabs)
deploy/                   systemd service + GCP setup script
scripts/
  batch_ml_diversity_audit.py
  extract_specg.py
static/inputs/            Your source WAVs (gitignored contents)
static/batch_outputs/     Generated datasets (gitignored contents)
```

---

## What stays local (gitignored)

- LaTeX and presentations (`*.tex`, `*.pdf`, `latex/`)
- Notes and local docs (`ref_docs/`, markdown except `README.md`)
- Audio, datasets, uploads, renders, generated plots
- Virtualenv, caches, `.DS_Store`, scratch scripts (`bench*.py`, `smoke*.py`)

---

## Known limitations

- Best suited to **subsonic** pass-bys with approximately known geometry.
- **Supersonic** cases are not modeled correctly.
- Absolute loudness may differ from the upload (peak-normalized WAV; intrinsic PSD + `1/R` only). Timing of the loudness peak follows render `t_CPA₂`, not the recording envelope.
- Wrong original geometry `(v₁, h₁, t_CPA₁)` corrupts the **inverted spectrum** (timbre), not the loudness CPA time.
- Early samples can be **silent** until a valid retarded-time root exists.
- Multiple emitters **smear** the Doppler ridge; each synthesis uses new random noise.
- Phase 1 `s` is **centerline / instantaneous**; multi-emitter retarded audio is a different object (see `phase1_schema.json`).
- **Wigner–Ville** (Experimental TF) shows cross-terms on harmonic vehicle audio.
- Batch planner supports **straight-line** trajectories only.

## License

Add your license here if needed.
