# DopplerSim 2.0

Physics-based vehicle pass-by audio re-rendering. Upload a recorded pass-by, specify original and target geometry, and synthesize a new pass-by with retarded-time Doppler physics.

## Features

The web app has four tabs:

1. **Pass-By Simulator** — invert Doppler and geometric spreading from the upload, estimate an intrinsic power spectral density, synthesize a new source, and render with exact retarded-time propagation under new speed, CPA distance, and timing. Optional **reassigned spectrogram** diagnostic plots (checkbox, default off).
2. **Spectrogram Explorer** — compare STFT, mel, reassigned, wideband/narrowband CWT, SSQ, and related spectrogram views. Includes **STFT vs reassigned** side-by-side comparison, optional CPA time overlay, and **`.npz` atom export** for relocated time–frequency data.
3. **Audio Comparison** — side-by-side waveform, spectrogram, RMS, and similarity metrics for two clips.
4. **Experimental TF** — research preview comparing STFT, reassigned STFT, and **Wigner–Ville** on short clips (≤30 s). Cross-terms expected on multi-tone audio; not for production analysis.

Reassignment and WVD are **visualization only** — they do not alter Pass-By audio synthesis.

## Requirements

- Python 3.10+
- See `requirements.txt`

## Setup

```bash
cd DopplerSim_2.0
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open [http://127.0.0.1:5003](http://127.0.0.1:5003) in your browser.

### Main workflow (Pass-By Simulator)

1. Upload a mono WAV pass-by recording.
2. Set **original pass-by parameters** `(v₁, h₁, t_CPA,1)` used to invert the upload.
3. Set **vehicle model** (length `L`, emitter count `N`).
4. Set **rendering parameters** `(v₂, h₂, t_CPA,2, T_out)`.
5. Optionally enable **Include reassigned spectrograms** for extra diagnostic plots.
6. Click **Generate** to produce a new WAV, plots, and diagnostics.

Generated outputs are written under `static/` and `renders/` (both gitignored).

## Physics (short)

- Vehicle geometry: `x(t) = v(t − t_CPA) + x₀`, range `R = √(x² + h²)`.
- Retarded time: solve `c(t − t_r) = R(t_r)` with geometric root selection.
- **Analysis:** per STFT frame, undo spreading (`×R`) and Doppler (`f_src = f/α`) using original parameters; average to an intrinsic power spectral density.
- **Synthesis:** colored noise from that spectrum; sample via `s_obs(t) = s_src(t_r(t))`; warp the recorded amplitude envelope; sum `N` emitters with `1/√N` scaling.

## Project layout

```
app.py                              Flask app (simulator + spectrograms + compare + experimental TF)
tf_analysis.py                      Reassignment helpers (atoms, plots, export)
tf_experimental.py                  Experimental Wigner–Ville comparison
templates/index.html                Unified UI
requirements.txt
uploads/                            Uploaded WAVs (gitignored)
static/                             Generated plots, audio, atoms (gitignored)
renders/                            Render session state (gitignored)
```

## Known limitations

- Best suited to **subsonic** pass-bys with approximately known geometry.
- **Supersonic / sonic-boom** cases are not modeled correctly.
- Output amplitude is driven mainly by **envelope warping**, not explicit `1/R` in the render path — levels may not match the original closely.
- Early output samples can be **silent** until a valid retarded-time root exists.
- More emitters mainly **smear** the Doppler ridge; spectrograms can look similar because each run uses new random noise and plots are peak-normalized.
- **Wigner–Ville** (Experimental TF tab) can show cross-term artifacts on harmonic vehicle audio.

## License

Add your license here if needed.
