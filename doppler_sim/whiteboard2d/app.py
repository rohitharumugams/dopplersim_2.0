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
from doppler_sim.batch.catalog import SourceClip, scan_input_catalog
from doppler_sim.batch.vehicle_metadata import (
    VEHICLE_METADATA,
    known_length_m,
    vehicle_display_name,
)

TEMPLATE = "whiteboard2d.html"
TAB = core.PASS_BY_PATH2D
TARGET_SOURCE_KMH = 60.0
DEFAULT_V2_KMH = 60.0
KMH_PER_MPS = 3.6


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
    return options


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
        ctx["vehicle_catalog_json"] = json.dumps(
            [
                {
                    "id": o["id"],
                    "available": o["available"],
                    "length_m": o["length_m"],
                    "speed_kmh": o.get("speed_kmh"),
                    "t_cpa1": o.get("t_cpa1"),
                    "speed_label": o.get("speed_label"),
                    "filename": o.get("filename"),
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
            freqs, psd_observed, psd_inverted, stft, stft_times = core.estimate_source_signature(
                audio, sr, params
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
            )
            generated = result["audio"]
            quantities = result["quantities"]
            traj = result["trajectory"]

            params = core.RenderParams(
                v1=params.v1,
                h1=params.h1,
                t_cpa1=params.t_cpa1,
                vehicle_length=params.vehicle_length,
                num_emitters=params.num_emitters,
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
            plots["observer_geometry"] = core._plot_path2d_board(
                xy,
                traj,
                (mic_x, mic_y),
                plot_dir,
                core.PLOT_EXPORT_NAMES["observer_geometry"],
            )

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
            core._write_path_animation(
                render_id,
                traj=traj,
                path_xy=xy,
                mic_xy=(mic_x, mic_y),
            )
            meta_path = core.RENDERS_DIR / render_id / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["has_path_animation"] = True
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
        speed_unit = meta.get("speed_unit", "kmph")
        ctx = _ctx(params, speed_unit, freq_max=freq_max, spec_quality=spec_quality)
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
