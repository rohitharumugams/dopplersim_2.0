# DopplerSim 2.0

Physics-based vehicle pass-by audio re-rendering. Upload a recorded pass-by, specify original and target geometry, and synthesize a new pass-by with retarded-time Doppler physics. **Batch Generation** builds ML-ready datasets by sweeping vehicles and speeds through the same synthesis pipeline.

## Features

The web app has five tabs:

1. **Batch Generation** — orchestrate many pass-by clips from a catalog of source recordings. Select vehicles, set a speed range and clip count, and export a structured dataset (WAV, spectrograms, metadata, CSV). Uses **evenly spaced speed sampling** across the range, **balanced clip counts per vehicle**, resume/override, and progress tracking. Reuses the Pass-By Sim backend (no separate physics path).
2. **Pass-By Simulator** — invert Doppler and geometric spreading from the upload, estimate an intrinsic power spectral density, synthesize a new source, and render with exact retarded-time propagation under new speed, CPA distance, and timing. Optional **reassigned spectrogram** diagnostic plots (checkbox, default off).
3. **Spectrogram Explorer** — compare STFT, mel, reassigned, wideband/narrowband CWT, SSQ, and related spectrogram views. Includes **STFT vs reassigned** side-by-side comparison, optional CPA time overlay, and **`.npz` atom export** for relocated time–frequency data.
4. **Audio Comparison** — side-by-side waveform, spectrogram, RMS, and similarity metrics for two clips.
5. **Experimental TF** — research preview comparing STFT, reassigned STFT, and **Wigner–Ville** on short clips (≤30 s). Cross-terms expected on multi-tone audio; not for production analysis.

Reassignment and WVD are **visualization only** — they do not alter Pass-By audio synthesis.

## Requirements

- Python 3.10+
- See `requirements.txt`

## Setup

```bash
git clone <your-repo-url>
cd DopplerSim2.0
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All data folders use **paths relative to the project root** (no machine-specific absolute paths):

| Path | Purpose |
|------|---------|
| `static/inputs/` | Source pass-by WAVs for batch generation (+ `.txt` sidecars) |
| `static/batch_outputs/` | Generated batch datasets (default output root) |
| `uploads/` | Pass-By Sim uploads (created at runtime) |
| `renders/` | Render session state (created at runtime) |

### Batch source clips

Place one or more vehicle recordings in `static/inputs/`:

- **Audio:** `VehicleName_59.wav` (speed in km/h in filename) or `VehicleName_10mps.wav`
- **Sidecar (required):** `VehicleName_59.txt` with `speed_kmh t_cpa1_s` (e.g. `54.0 5.65`) for source geometry `t_CPA,1`

Example: `CitroenC4Picasso_59.wav` + `CitroenC4Picasso_59.txt`

## Run

```bash
python app.py
```

Open [http://127.0.0.1:5003](http://127.0.0.1:5003) in your browser.

Production (e.g. Heroku): `gunicorn app:app` (see `Procfile`).

### Batch Generation workflow

1. Open the **Batch Generation** tab.
2. Set **batch name** and **total clips** (default output: `static/batch_outputs/<batch_name>/`).
3. Select vehicles from the catalog; pick a **source speed** per vehicle when multiple clips exist.
4. Configure **speed range** (v₂ min/max), **CPA distance** (h₂), **t_CPA,2**, and optional spectrogram types.
5. Click **Start batch**. If the output folder already exists, confirm **overwrite** or change the batch name.
6. Watch **progress** (clips completed / total). Use **Resume batch** to continue after interruption.

**Sampling rules (deterministic):**

- Clips are split **evenly across selected vehicles** (e.g. 400 clips, 4 vehicles → 100 each).
- Within each vehicle, render speeds are **evenly spaced** from min to max across the full interval (floating-point steps; uniform repetition only when oversubscribed).

**Per-sample output** (`static/batch_outputs/<batch_name>/`):

```
audio_clips/sample_0000001/
  <vehicle>_<speed>mps.wav
  trajectory_plot.png
  spectrograms/          # STFT, CQT, reassigned, etc. + combined.png
  metadata/              # .npy tensors, simulation_parameters.json
dataset.csv              # batch-level index (appended per clip)
batch_plan_state.json    # plan + config (for resume)
progress.json            # live status
```

### Pass-By Simulator workflow

1. Upload a mono WAV pass-by recording.
2. Set **original pass-by parameters** `(v₁, h₁, t_CPA,1)` used to invert the upload.
3. Set **vehicle model** (length `L`, emitter count `N`).
4. Set **rendering parameters** `(v₂, h₂, t_CPA,2, T_out)`.
5. Optionally enable **Include reassigned spectrograms** for extra diagnostic plots.
6. Click **Generate** to produce a new WAV, plots, and diagnostics.

Generated Pass-By outputs are written under `static/` and `renders/` (gitignored).

## Physics (short)

- Vehicle geometry: `x(t) = v(t − t_CPA) + x₀`, range `R = √(x² + h²)`.
- Retarded time: solve `c(t − t_r) = R(t_r)` with geometric root selection.
- **Analysis:** per STFT frame, undo spreading (`×R`) and Doppler (`f_src = f/α`) using original parameters; average to an intrinsic power spectral density.
- **Synthesis:** colored noise from that spectrum; sample via `s_obs(t) = s_src(t_r(t))`; warp the recorded amplitude envelope; sum `N` emitters with `1/√N` scaling.

## Project layout

```
app.py                              Entry point (`python app.py`, gunicorn `app:app`)
doppler_sim/
  application.py                    Flask app (physics, routes, all tabs)
  batch/                            Batch generation (catalog, planner, runner, export)
    catalog.py                      Scan static/inputs, parse sidecars
    planner.py                      Sample plan (vehicles + speeds)
    runner.py                       Async job, progress, resume
    features.py                     WAV, spectrograms, metadata per sample
    pipeline.py                     Calls Pass-By synthesis per planned clip
  specg/explorer.py                 Spectrogram Explorer + batch spectrogram export
  tf/
    analysis.py                     Reassignment helpers (atoms, plots, export)
    experimental.py                 Experimental Wigner–Ville comparison
templates/index.html                Unified UI (all five tabs)
requirements.txt
static/inputs/                      Batch source clips (add your WAVs + .txt)
static/batch_outputs/               Batch datasets (gitignored contents)
uploads/                            Uploaded WAVs (gitignored, runtime)
renders/                            Render session state (gitignored)
```

## Known limitations

- Best suited to **subsonic** pass-bys with approximately known geometry.
- **Supersonic / sonic-boom** cases are not modeled correctly.
- Output amplitude is driven mainly by **envelope warping**, not explicit `1/R` in the render path — levels may not match the original closely.
- Early output samples can be **silent** until a valid retarded-time root exists.
- More emitters mainly **smear** the Doppler ridge; spectrograms can look similar because each run uses new random noise and plots are peak-normalized.
- **Wigner–Ville** (Experimental TF tab) can show cross-term artifacts on harmonic vehicle audio.
- Batch generation is **straight-line path only**; curved trajectories are not supported in the batch planner.

## License

Add your license here if needed.
