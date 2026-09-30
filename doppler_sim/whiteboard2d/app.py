"""Flask app: 2D whiteboard single-clip UI only.

Imports physics / plot helpers from ``doppler_sim.application`` but does not
register routes on the main DopplerSim app. Existing UI and backend files are
left untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, send_file, send_from_directory, session

# Reuse existing helpers — do not edit application.py.
import doppler_sim.application as core
from doppler_sim.atmosphere import AIR_N_FFT, Atmosphere
from doppler_sim.batch.catalog import SourceClip, scan_input_catalog
from doppler_sim.batch.vehicle_metadata import (
    VEHICLE_METADATA,
    known_length_m,
    vehicle_display_name,
)
from doppler_sim.whiteboard2d.body_layout import (
    body_emitters,
    linear_emitters,
    offsets_xyz,
    pack_propagation,
    plot_body_diagram,
    plot_path_emitters,
    plot_propagation_curves,
    replot_emitter_views,
    save_emitter_view,
    write_emitter_animation,
)
from doppler_sim.whiteboard2d.vehicle_schematics import attach_geometry, geometry_for_catalog_id

TEMPLATE = "whiteboard2d.html"
TAB = core.PASS_BY_PATH2D
TARGET_SOURCE_KMH = 60.0
DEFAULT_V2_KMH = 60.0
KMH_PER_MPS = 3.6


def _body_frame_quantities(times, v, h, t_cpa, xyz, mic_z: float) -> dict[str, np.ndarray]:
    """Straight-pass propagation for one body emitter.

    The microphone is at (0, 0, mic_z). The axle travels on y = h, heading +x,
    and passes x = 0 at ``t_cpa``. Vehicle +Y is left, so an emitter at y_i
    sits at world y = h + y_i (a positive h puts the mic on the vehicle's right).
    With y_i = z_i = mic_z = 0 this is the same model as the 2D invert.
    """
    x_i, y_i, z_i = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    x0_quadratic = x_i - float(v) * float(t_cpa)
    h_eff = float(np.hypot(float(h) + y_i, z_i - float(mic_z)))
    t_r = core.solve_retarded_time(times, float(v), x0_quadratic, h_eff)
    r = core.SPEED_OF_SOUND * (times - t_r)
    x_emitter = core.vehicle_center_position(t_r, float(v), float(t_cpa)) + x_i
    v_r = float(v) * x_emitter / np.maximum(r, 1e-9)
    alpha = core.SPEED_OF_SOUND / (core.SPEED_OF_SOUND + v_r)
    bad = ~np.isfinite(t_r) | ~(r > 0.0)
    return {
        "t_r": np.where(bad, np.nan, t_r),
        "R": np.where(bad, np.nan, r),
        "v_r": np.where(bad, np.nan, v_r),
        "alpha": np.where(bad, np.nan, alpha),
    }


def _invert_with_quantities(stft, freqs, quantities, air_beta: np.ndarray | None = None, max_air_undo_db: float = 30.0) -> np.ndarray:
    source = np.zeros_like(stft, dtype=float)
    valid = np.isfinite(quantities["t_r"]) & (quantities["R"] > 0.0)
    undo_cap = float(max_air_undo_db) / 8.685889638065037
    for frame_idx in range(stft.shape[1]):
        if not valid[frame_idx]:
            continue
        air_undo = None
        if air_beta is not None:
            air_undo = np.exp(np.minimum(air_beta * float(quantities["R"][frame_idx]), undo_cap))
        source[:, frame_idx] = core.invert_stft_frame_to_source_power(
            stft[:, frame_idx],
            freqs,
            float(quantities["alpha"][frame_idx]),
            float(quantities["R"][frame_idx]),
            air_undo=air_undo,
        )
    return source


def _estimate_signature(
    audio,
    sr,
    params,
    emitter_xyz: np.ndarray | None = None,
    mic_z: float = 0.0,
    atmosphere: Atmosphere | None = None,
):
    """Invert the clip. Body passes each emitter's (x, y, z); Linear uses the stock line."""
    use_air = atmosphere is not None and atmosphere.enabled
    if emitter_xyz is None and not use_air:
        return core.estimate_source_signature(audio, sr, params)

    stft, freqs, times = core.compute_stft(audio, sr)
    beta = atmosphere.beta(freqs) if use_air else None
    undo_db = atmosphere.max_undo_db if use_air else 30.0
    psd_observed = core.estimate_psd_observed(stft)
    if emitter_xyz is None:
        psd_inverted = core.estimate_psd_inverted(
            stft, freqs, times, params, air_beta=beta, max_air_undo_db=undo_db
        )
        return freqs, psd_observed, psd_inverted, stft, times

    offsets = np.asarray(emitter_xyz, dtype=np.float64).reshape(-1, 3)
    if offsets.shape[0] < 1:
        raise ValueError("Body layout needs at least one emitter to invert the recording.")

    psd_sum = np.zeros(len(freqs), dtype=float)
    for xyz in offsets:
        quantities = _body_frame_quantities(
            times, params.v1, params.h1, params.t_cpa1, xyz, mic_z
        )
        source_spectrogram = _invert_with_quantities(
            stft, freqs, quantities, air_beta=beta, max_air_undo_db=undo_db
        )
        valid_frames = np.any(source_spectrogram > 0.0, axis=0)
        if np.any(valid_frames):
            psd_sum += np.mean(source_spectrogram[:, valid_frames], axis=1)
    psd_inverted = np.maximum(psd_sum / float(offsets.shape[0]), 0.0)
    return freqs, psd_observed, psd_inverted, stft, times


def _form_float(name: str, default: float) -> float:
    try:
        value = float(request.form.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _atmosphere_from_request() -> Atmosphere:
    """Read air settings from the generate form. Other posts keep the defaults."""
    if request.method == "POST" and "temperature_c" in request.form:
        return Atmosphere(
            temperature_c=_form_float("temperature_c", 20.0),
            relative_humidity_percent=_form_float("relative_humidity_percent", 50.0),
            pressure_atm=_form_float("pressure_atm", 1.0),
            enabled=request.form.get("air_absorption") in ("on", "true", "1", "yes"),
        )
    return Atmosphere()


def _atmosphere_from_meta(meta: dict) -> Atmosphere:
    saved = meta.get("atmosphere") or {}
    return Atmosphere(
        temperature_c=float(saved.get("temperature_c", 20.0)),
        relative_humidity_percent=float(saved.get("relative_humidity_percent", 50.0)),
        pressure_atm=float(saved.get("pressure_atm", 1.0)),
        enabled=bool(saved.get("enabled", True)),
    )


def _nearest_clip(clips: list[SourceClip], target_kmh: float = TARGET_SOURCE_KMH) -> SourceClip:
    target_mps = target_kmh / KMH_PER_MPS
    return min(clips, key=lambda c: abs(c.speed_mps - target_mps))


def _vehicle_options() -> list[dict[str, Any]]:
    catalog = scan_input_catalog(core.BASE_DIR)
    options: list[dict[str, Any]] = []
    for vehicle in VEHICLE_METADATA:
        clips = catalog.vehicles.get(vehicle, [])
        length = known_length_m(vehicle) or 4.5
        entry: dict[str, Any] = {
            "id": vehicle,
            "label": vehicle_display_name(vehicle),
            "length_m": float(length),
            "available": bool(clips),
        }
        if clips:
            clip = _nearest_clip(clips)
            speed_kmh = float(clip.speed_mps * KMH_PER_MPS)
            entry.update(
                {
                    "speed_kmh": round(speed_kmh, 2),
                    "speed_mps": float(clip.speed_mps),
                    "speed_label": clip.speed_label,
                    "t_cpa1": float(clip.t_cpa1_s) if clip.t_cpa1_s is not None else 5.0,
                    "filename": clip.path.name,
                    "path": str(clip.path),
                }
            )
        options.append(entry)
    return attach_geometry(options)


def _default_params_from_vehicle(vehicle: dict[str, Any] | None) -> core.RenderParams:
    v1_kmh = float(vehicle["speed_kmh"]) if vehicle and vehicle.get("available") else TARGET_SOURCE_KMH
    length = float(vehicle["length_m"]) if vehicle else 4.5
    t_cpa1 = float(vehicle["t_cpa1"]) if vehicle and vehicle.get("available") else 5.0
    return core.RenderParams(
        v1=v1_kmh / KMH_PER_MPS,
        h1=10.0,
        t_cpa1=t_cpa1,
        vehicle_length=length,
        num_emitters=3,
        v2=DEFAULT_V2_KMH / KMH_PER_MPS,
        h2=8.0,
        t_cpa2=2.5,
        t_out=6.0,
    )


def _pick_default_vehicle(options: list[dict[str, Any]]) -> dict[str, Any] | None:
    for opt in options:
        if opt.get("available"):
            return opt
    return options[0] if options else None


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(core.BASE_DIR / "templates"),
        static_folder=str(core.BASE_DIR / "static"),
        static_url_path="/assets",
    )
    app.config["MAX_CONTENT_LENGTH"] = core.app.config["MAX_CONTENT_LENGTH"]
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "doppler-sim-whiteboard2d-dev-key")
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    if os.environ.get("FORCE_HTTPS", "").lower() in ("1", "true", "yes"):
        app.config["SESSION_COOKIE_SECURE"] = True

    def _page(**extra):
        return render_template(TEMPLATE, **extra)

    def _ctx(
        params=None,
        speed_unit: str = "kmph",
        freq_max: float | None = None,
        spec_quality: str = "sd",
        *,
        selected_vehicle: str | None = None,
        emitter_layout: str | None = None,
        mic_z: float | None = None,
        atmosphere: Atmosphere | None = None,
        **extra,
    ):
        options = _vehicle_options()
        by_id = {o["id"]: o for o in options}
        selected = by_id.get(selected_vehicle) if selected_vehicle else None
        if selected is None or not selected.get("available"):
            selected = _pick_default_vehicle(options)
        if params is None:
            params = _default_params_from_vehicle(selected)
        ctx = core.form_context(
            params,
            speed_unit,
            freq_max=core.DEFAULT_FREQ_MAX if freq_max is None else freq_max,
            spec_quality=spec_quality,
            tab=TAB,
            **extra,
        )
        ctx["vehicle_options"] = options
        ctx["selected_vehicle"] = selected["id"] if selected else ""
        if emitter_layout is None:
            emitter_layout = (request.form.get("emitter_layout") or "linear").strip()
        if emitter_layout not in ("linear", "body"):
            emitter_layout = "linear"
        if mic_z is None:
            try:
                mic_z = float(request.form.get("mic_z", "1.2"))
            except (TypeError, ValueError):
                mic_z = 1.2
        if not np.isfinite(mic_z):
            mic_z = 1.2
        ctx["emitter_layout"] = emitter_layout
        ctx["mic_z"] = float(mic_z)
        if atmosphere is None:
            atmosphere = _atmosphere_from_request()
        ctx["air_enabled"] = bool(atmosphere.enabled)
        ctx["temperature_c"] = float(atmosphere.temperature_c)
        ctx["relative_humidity_percent"] = float(atmosphere.relative_humidity_percent)
        ctx["pressure_atm"] = float(atmosphere.pressure_atm)
        ctx["vehicle_catalog_json"] = json.dumps(
            [
                {
                    "id": o["id"],
                    "label": o["label"],
                    "available": o["available"],
                    "length_m": o["length_m"],
                    "speed_kmh": o.get("speed_kmh"),
                    "t_cpa1": o.get("t_cpa1"),
                    "speed_label": o.get("speed_label"),
                    "filename": o.get("filename"),
                    "geometry": o.get("geometry"),
                }
                for o in options
            ]
        )
        return ctx

    def _resolve_source(
        selected_vehicle: str | None,
    ) -> tuple[Path | None, str | None, str | None]:
        """Prefer a fresh upload; otherwise use the chosen catalog clip (~60 km/h)."""
        uploaded = request.files.get("audio_file")
        if uploaded is not None and uploaded.filename:
            return core.resolve_upload_path(TAB)

        options = _vehicle_options()
        by_id = {o["id"]: o for o in options}
        vehicle = by_id.get(selected_vehicle or "")
        if vehicle and vehicle.get("available"):
            src = Path(vehicle["path"])
            if not src.is_file():
                return None, None, f"Catalog clip missing for {vehicle['id']}."
            upload_id = uuid.uuid4().hex
            dest = core.UPLOAD_DIR / f"{upload_id}.wav"
            shutil.copy2(src, dest)
            session[TAB.upload_id_key] = upload_id
            session[TAB.upload_filename_key] = src.name
            return dest, src.name, None

        if selected_vehicle:
            return (
                None,
                None,
                f"No source clip found for {selected_vehicle} under static/inputs/. "
                "Add the WAV (+ sidecar) or upload a file.",
            )

        available = [o for o in options if o.get("available")]
        if not available:
            return (
                None,
                None,
                "No vehicle clips found under static/inputs/. "
                "Pull the latest branch with source WAVs, or upload a file.",
            )

        return core.resolve_upload_path(TAB)

    @app.route("/health")
    def health():
        return "ok", 200

    @app.route("/", methods=["GET"])
    @app.route("/path2d", methods=["GET"])
    def path2d_board():
        return _page(**_ctx())

    @app.route("/path2d/generate", methods=["POST"])
    def path2d_generate():
        from doppler_sim.path2d import synthesize_path_audio

        params, speed_unit = core.parse_params()
        # Whiteboard site defaults to km/h even if the shared parser default is m/s.
        if not request.form.get("speed_unit"):
            speed_unit = "kmph"
        freq_max = core.parse_freq_max()
        include_reassigned = core.parse_include_reassigned()
        spec_quality = core.parse_spec_quality()
        spec_hop = core.spec_hop_for_quality(spec_quality)
        selected_vehicle = (request.form.get("catalog_vehicle") or "").strip() or None
        emitter_layout = (request.form.get("emitter_layout") or "linear").strip()
        if emitter_layout not in ("linear", "body"):
            emitter_layout = "linear"
        try:
            mic_z = float(request.form.get("mic_z", "1.2"))
        except (TypeError, ValueError):
            mic_z = 1.2
        if not np.isfinite(mic_z):
            mic_z = 1.2
        atmosphere = _atmosphere_from_request()

        upload_path, upload_filename, upload_error = _resolve_source(selected_vehicle)
        if upload_error:
            return _page(
                error=upload_error,
                **_ctx(
                    params,
                    speed_unit,
                    freq_max=freq_max,
                    spec_quality=spec_quality,
                    selected_vehicle=selected_vehicle,
                ),
            )

        try:
            raw_path = request.form.get("path_json", "").strip()
            if not raw_path:
                raise ValueError("Draw a path on the whiteboard before generating.")
            points = json.loads(raw_path)
            if not isinstance(points, list) or len(points) < 2:
                raise ValueError("Path must contain at least two points.")
            xy = np.asarray([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float64)
            mic_x = float(request.form.get("mic_x", "0"))
            mic_y = float(request.form.get("mic_y", "0"))
        except Exception as exc:
            return _page(
                error=f"Invalid path / mic settings: {exc}",
                **_ctx(
                    params,
                    speed_unit,
                    freq_max=freq_max,
                    spec_quality=spec_quality,
                    selected_vehicle=selected_vehicle,
                ),
            )

        try:
            audio, sr = librosa.load(upload_path, sr=None, mono=True)
        except Exception as exc:
            return _page(
                error=f"Failed to load audio: {exc}",
                **_ctx(
                    params,
                    speed_unit,
                    freq_max=freq_max,
                    spec_quality=spec_quality,
                    selected_vehicle=selected_vehicle,
                ),
            )
        if audio.size == 0:
            return _page(
                error="Uploaded file is empty.",
                **_ctx(
                    params,
                    speed_unit,
                    freq_max=freq_max,
                    spec_quality=spec_quality,
                    selected_vehicle=selected_vehicle,
                ),
            )

        uploaded_plot_copy = audio.copy()
        uploaded_sr = sr

        try:
            geometry = geometry_for_catalog_id(selected_vehicle or "")
            body_dims = None
            if emitter_layout == "body":
                if geometry is None:
                    raise ValueError("This vehicle has no body geometry, so Body layout cannot be placed.")
                emitters = body_emitters(geometry)
                body_dims = geometry["dims"]
                body_xyz = offsets_xyz(emitters)
                synth_kwargs = {
                    "emitter_offsets_xyz": body_xyz,
                    "mic_z": float(mic_z),
                }
            else:
                emitters = linear_emitters(float(params.vehicle_length), int(params.num_emitters))
                body_xyz = None
                synth_kwargs = {}

            freqs, psd_observed, psd_inverted, stft, stft_times = _estimate_signature(
                audio, sr, params, body_xyz, mic_z=float(mic_z), atmosphere=atmosphere
            )
            del audio

            result = synthesize_path_audio(
                xy,
                speed_mps=float(params.v2),
                sr=core.OUTPUT_SR,
                freqs=freqs,
                psd=psd_inverted,
                synthesize_psd_noise=core.synthesize_psd_noise,
                mic_xy=(mic_x, mic_y),
                vehicle_length=float(params.vehicle_length),
                num_emitters=int(params.num_emitters),
                air_beta=atmosphere.forward_beta(core.OUTPUT_SR, AIR_N_FFT) if atmosphere.enabled else None,
                air_n_fft=AIR_N_FFT,
                **synth_kwargs,
            )
            generated = result["audio"]
            quantities = result["quantities"]
            traj = result["trajectory"]

            params = core.RenderParams(
                v1=params.v1,
                h1=params.h1,
                t_cpa1=params.t_cpa1,
                vehicle_length=params.vehicle_length,
                num_emitters=len(emitters) if emitter_layout == "body" else params.num_emitters,
                v2=float(params.v2),
                h2=float(result["cpa_distance_m"]),
                t_cpa2=float(result["cpa_time_sec"]),
                t_out=float(traj["duration_s"][0]),
            )

            output_name = f"{uuid.uuid4().hex}.wav"
            sf.write(core.GENERATED_DIR / output_name, generated, core.OUTPUT_SR, subtype="PCM_16")

            render_id = uuid.uuid4().hex
            plot_dir = core.PLOTS_DIR / render_id
            plots = core.generate_all_plots(
                uploaded_plot_copy,
                uploaded_sr,
                generated,
                freqs,
                psd_observed,
                psd_inverted,
                stft,
                stft_times,
                params,
                quantities,
                plot_dir,
                freq_max=freq_max,
                include_reassigned=include_reassigned,
                spec_hop=spec_hop,
            )
            plots["observer_geometry"] = plot_path_emitters(
                xy,
                traj,
                (mic_x, mic_y),
                emitters,
                core.PLOT_EXPORT_NAMES["observer_geometry"],
                plot_dir,
            )
            propagation = None
            if emitter_layout == "body" and body_dims is not None:
                plots["vehicle_geometry"] = plot_body_diagram(
                    emitters,
                    body_dims,
                    core.PLOT_EXPORT_NAMES["vehicle_geometry"],
                    plot_dir,
                )
                propagation = pack_propagation(
                    traj["t"], emitters, result["emitter_curves"], quantities
                )
                plot_propagation_curves(propagation, plot_dir)

            core.save_render_state(
                render_id,
                tab=TAB,
                output_name=output_name,
                upload_filename=upload_filename,
                speed_unit=speed_unit,
                freq_max=freq_max,
                params=params,
                plots=plots,
                uploaded_audio=uploaded_plot_copy,
                uploaded_sr=uploaded_sr,
                generated_audio=generated,
                freqs=freqs,
                psd_observed=psd_observed,
                psd_inverted=psd_inverted,
                stft=stft,
                stft_times=stft_times,
                quantities=quantities,
                include_reassigned=include_reassigned,
                spec_quality=spec_quality,
                pipeline="path2d",
                path_trajectory=traj,
                mic_position=(mic_x, mic_y),
            )
            render_dir = core.RENDERS_DIR / render_id
            write_emitter_animation(
                render_dir,
                traj=traj,
                path_xy=xy,
                mic_xy=(mic_x, mic_y),
                emitters=emitters,
            )
            save_emitter_view(
                render_dir,
                layout=emitter_layout,
                mic_z=float(mic_z) if emitter_layout == "body" else 0.0,
                mic_xy=(mic_x, mic_y),
                path_xy=xy,
                traj=traj,
                emitters=emitters,
                dims=body_dims,
                propagation=propagation,
            )
            meta_path = render_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["has_path_animation"] = True
            meta["emitter_layout"] = emitter_layout
            meta["mic_z"] = float(mic_z) if emitter_layout == "body" else 0.0
            meta["atmosphere"] = {
                "enabled": bool(atmosphere.enabled),
                "temperature_c": float(atmosphere.temperature_c),
                "relative_humidity_percent": float(atmosphere.relative_humidity_percent),
                "pressure_atm": float(atmosphere.pressure_atm),
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except Exception as exc:
            return _page(
                error=f"Generation failed: {exc}",
                **_ctx(
                    params,
                    speed_unit,
                    freq_max=freq_max,
                    spec_quality=spec_quality,
                    selected_vehicle=selected_vehicle,
                ),
            )

        saved_meta, _ = core.load_render_state(render_id)
        ctx = _ctx(
            params,
            speed_unit,
            freq_max=freq_max,
            spec_quality=spec_quality,
            selected_vehicle=selected_vehicle,
        )
        ctx.update(
            core.build_success_context(
                render_id,
                saved_meta,
                plots,
                params,
                speed_unit,
                upload_filename,
                tab=TAB,
            )
        )
        return _page(**ctx)

    @app.route("/path2d/update-freq-max", methods=["POST"])
    def path2d_update_freq_max():
        render_id = session.get(TAB.last_render_id_key)
        if not render_id or not (core.RENDERS_DIR / render_id / "meta.json").exists():
            return _page(
                error="No recent render found. Generate pass-by audio first.",
                **_ctx(freq_max=core.parse_freq_max()),
            )

        meta, _arrays = core.load_render_state(render_id)
        freq_max = core.parse_freq_max(default=float(meta.get("freq_max", core.DEFAULT_FREQ_MAX)))
        spec_quality = core.parse_spec_quality(default=meta.get("spec_quality", "sd"))
        meta, params, plots = core.regenerate_plots_from_state(
            render_id, freq_max, spec_quality=spec_quality
        )
        replot_emitter_views(
            core.RENDERS_DIR / render_id,
            core.PLOTS_DIR / render_id,
            core.PLOT_EXPORT_NAMES["observer_geometry"],
            core.PLOT_EXPORT_NAMES["vehicle_geometry"],
        )
        speed_unit = meta.get("speed_unit", "kmph")
        ctx = _ctx(
            params,
            speed_unit,
            freq_max=freq_max,
            spec_quality=spec_quality,
            emitter_layout=meta.get("emitter_layout", "linear"),
            mic_z=float(meta.get("mic_z", 1.2)),
            atmosphere=_atmosphere_from_meta(meta),
        )
        ctx.update(
            core.build_success_context(
                render_id,
                meta,
                plots,
                params,
                speed_unit,
                meta.get("upload_filename"),
                tab=TAB,
            )
        )
        return _page(**ctx)

    @app.route("/path2d/download-bundle", methods=["POST"])
    def path2d_download_bundle():
        render_id = session.get(TAB.last_render_id_key)
        if not render_id or not (core.RENDERS_DIR / render_id / "meta.json").exists():
            return _page(
                error="No recent render found. Generate pass-by audio first.",
                **_ctx(),
            )

        meta, _ = core.load_render_state(render_id)
        bundle_name = core.sanitize_bundle_name(request.form.get("bundle_name", "path2d_whiteboard_export"))
        plot_dir = core.PLOTS_DIR / render_id
        audio_path = core.GENERATED_DIR / meta["output_name"]

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(audio_path, arcname=f"{bundle_name}/generated_audio.wav")
            for plot_key, plot_filename in meta["plots"].items():
                plot_path = plot_dir / plot_filename
                if plot_path.exists():
                    export_name = core.PLOT_EXPORT_NAMES.get(plot_key, plot_filename)
                    archive.write(plot_path, arcname=f"{bundle_name}/{export_name}")

            phase1_dir = core.RENDERS_DIR / render_id / "phase1"
            if phase1_dir.is_dir():
                for path in sorted(phase1_dir.rglob("*")):
                    if path.is_file():
                        archive.write(
                            path,
                            arcname=f"{bundle_name}/phase1/{path.relative_to(phase1_dir).as_posix()}",
                        )
                archive.writestr(
                    f"{bundle_name}/phase1/README.txt",
                    "Phase 1 learning pair: phase1/spectrograms/stft.npy (A) "
                    "↔ phase1/metadata/state_frames.npy (s) via frame_times.npy.\n",
                )

        buffer.seek(0)
        return send_file(
            buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{bundle_name}.zip",
        )

    @app.route("/path2d/animation/<render_id>")
    def path2d_animation(render_id: str):
        path = core.RENDERS_DIR / render_id / "path_animation.json"
        if not path.exists():
            return {"error": "animation not found"}, 404
        return send_file(path, mimetype="application/json")

    @app.route("/media/plots/<render_id>/<path:filename>")
    def plot_file(render_id: str, filename: str):
        return send_from_directory(core.PLOTS_DIR / render_id, filename)

    @app.route("/media/generated/<path:filename>")
    def generated_file(filename: str):
        return send_from_directory(core.GENERATED_DIR, filename)

    return app


app = create_app()
