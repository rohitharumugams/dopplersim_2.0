"""Batch generation orchestration with resume support."""

from __future__ import annotations

import csv
import json
import shutil
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from doppler_sim.batch.catalog import VehicleCatalog, scan_input_catalog
from doppler_sim.batch.constants import (
    CSV_HEADERS,
    DEFAULT_BATCH_OUTPUT_DIR,
    resolve_batch_output_root,
    sample_dir_name,
)
from doppler_sim.batch.features import export_sample_artifacts
from doppler_sim.batch.pipeline import synthesize_planned_sample
from doppler_sim.batch.planner import BatchConfig, BatchPlan, PlannedSample, build_batch_plan
from doppler_sim.batch.sampler import SamplerBank

PLAN_STATE_FILE = "batch_plan_state.json"
SAMPLER_STATE_FILE = "sampler_state.json"
PROGRESS_FILE = "progress.json"

_lock = threading.Lock()
_runner_state: dict[str, Any] = {
    "thread": None,
    "cancel": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _batch_dir(base_dir: Path, batch_name: str, output_dir: str) -> Path:
    return resolve_batch_output_root(base_dir, output_dir) / batch_name


def _audio_clips_dir(batch_dir: Path) -> Path:
    d = batch_dir / "audio_clips"
    d.mkdir(parents=True, exist_ok=True)
    return d


def scan_completed_indices(audio_dir: Path) -> set[int]:
    done: set[int] = set()
    if not audio_dir.is_dir():
        return done
    for child in audio_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("sample_"):
            continue
        try:
            done.add(int(child.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return done


_progress_lock = threading.Lock()


def _write_progress(batch_dir: Path, payload: dict[str, Any]) -> None:
    """Write progress.json (direct write — safe on Windows while status polls read)."""
    path = batch_dir / PROGRESS_FILE
    data = json.dumps(payload, indent=2)
    tmp = path.with_suffix(".json.tmp")
    with _progress_lock:
        path.write_text(data, encoding="utf-8")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _read_progress_file(path: Path, batch_name: str) -> dict[str, Any]:
    if not path.exists():
        return {"status": "idle", "batch_id": batch_name}
    for attempt in range(3):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                return {"status": "idle", "batch_id": batch_name}
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(0.02)
                continue
            return {"status": "idle", "batch_id": batch_name}
        except OSError:
            if attempt < 2:
                time.sleep(0.02)
                continue
            return {"status": "idle", "batch_id": batch_name}
    return {"status": "idle", "batch_id": batch_name}


def _append_log(batch_dir: Path, batch_id: str, line: str) -> None:
    log_path = batch_dir / f"generation_log_{batch_id}.txt"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


def _append_clips_jsonl(batch_dir: Path, record: dict[str, Any]) -> None:
    with (batch_dir / "clips_metadata.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _ensure_dataset_header(batch_dir: Path) -> None:
    csv_path = batch_dir / "dataset.csv"
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADERS)


def _append_dataset_row(batch_dir: Path, row: dict[str, Any]) -> None:
    _ensure_dataset_header(batch_dir)
    with (batch_dir / "dataset.csv").open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
        writer.writerow({k: row.get(k, "") for k in CSV_HEADERS})


def _clip_to_csv_row(batch_id: str, plan: PlannedSample, wav_name: str) -> dict[str, Any]:
    return {
        "sample_id": sample_dir_name(plan.index),
        "batch_id": batch_id,
        "filename": wav_name,
        "vehicle_class": plan.vehicle,
        "trajectory_type": plan.path_type,
        "source_speed_mps": plan.source_speed_mps,
        "speed_mps": plan.speed_mps,
        "cpa_distance_m": plan.cpa_distance_m,
        "cpa_time_sec": plan.cpa_time_sec,
        "vehicle_length_m": plan.vehicle_length_m,
        "num_emitters": plan.num_emitters,
        "pass_by_in_clip": True,
    }


def _write_batch_metadata(batch_dir: Path, batch_id: str, config: BatchConfig, stats: dict[str, Any]) -> None:
    payload = {
        "batch_id": batch_id,
        "config": config.to_dict(),
        "statistics": stats,
        "timestamp": _utc_now(),
    }
    (batch_dir / f"metadata_{batch_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"Batch: {batch_id}",
        f"Generated at: {payload['timestamp']}",
        f"Total requested: {stats.get('total', 0)}",
        f"Completed: {stats.get('completed', 0)}",
        f"Failed: {stats.get('failed', 0)}",
        f"Skipped (resume): {stats.get('skipped', 0)}",
    ]
    (batch_dir / f"statistics_{batch_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batch_job(base_dir: Path, config: BatchConfig, *, resume: bool = False, override: bool = False) -> None:
    batch_id = config.batch_name
    batch_dir = _batch_dir(base_dir, batch_id, config.output_dir)
    audio_dir = _audio_clips_dir(batch_dir)

    batch_dir.mkdir(parents=True, exist_ok=True)

    plan_path = batch_dir / PLAN_STATE_FILE
    sampler_path = batch_dir / SAMPLER_STATE_FILE

    if resume and plan_path.exists():
        plan = BatchPlan.load(plan_path)
        config = plan.config
        batch_id = config.batch_name
    else:
        catalog = scan_input_catalog(base_dir, config.input_dir)
        plan, bank = build_batch_plan(config, catalog, base_dir)
        plan.save(plan_path)
        bank.save(sampler_path)

    completed = scan_completed_indices(audio_dir) if resume else set()
    stats = {
        "total": len(plan.samples),
        "completed": len(completed),
        "failed": 0,
        "skipped": 0,
    }

    _write_progress(
        batch_dir,
        {
            "status": "running",
            "batch_id": batch_id,
            "total": len(plan.samples),
            "completed": len(completed),
            "failed": 0,
            "skipped": 0,
            "started_at": _utc_now(),
        },
    )
    _append_log(batch_dir, batch_id, f"[{_utc_now()}] Batch started (resume={resume})")

    for sample in plan.samples:
        with _lock:
            if _runner_state.get("cancel"):
                _write_progress(
                    batch_dir,
                    {
                        "status": "cancelled",
                        "batch_id": batch_id,
                        "total": len(plan.samples),
                        "completed": stats["completed"],
                        "failed": stats["failed"],
                        "skipped": stats["skipped"],
                        "message": "Cancelled by user",
                    },
                )
                _append_log(batch_dir, batch_id, f"[{_utc_now()}] Batch cancelled")
                return

        if sample.index in completed:
            stats["skipped"] += 1
            continue

        sample_dir = audio_dir / sample_dir_name(sample.index)
        try:
            _append_log(
                batch_dir,
                batch_id,
                f"[{_utc_now()}] sample {sample.index}: source={sample.source_path} v2={sample.speed_mps}",
            )
            audio, quantities, _aux = synthesize_planned_sample(sample, base_dir=base_dir)
            artifact = export_sample_artifacts(sample_dir, sample, audio, quantities, batch_id, config)
            row = _clip_to_csv_row(batch_id, sample, artifact["wav_name"])
            _append_dataset_row(batch_dir, row)
            _append_clips_jsonl(
                batch_dir,
                {
                    "sample_id": row["sample_id"],
                    "batch_id": batch_id,
                    "plan": sample.to_dict(),
                    "artifact": artifact,
                },
            )
            stats["completed"] += 1
            completed.add(sample.index)
            _write_progress(
                batch_dir,
                {
                    "status": "running",
                    "batch_id": batch_id,
                    "total": len(plan.samples),
                    "completed": stats["completed"],
                    "failed": stats["failed"],
                    "skipped": stats["skipped"],
                    "current_sample": sample.index,
                },
            )
        except Exception as exc:
            stats["failed"] += 1
            _append_log(batch_dir, batch_id, f"[{_utc_now()}] sample {sample.index} FAILED: {exc}")
            _append_log(batch_dir, batch_id, traceback.format_exc())

    _write_batch_metadata(batch_dir, batch_id, config, stats)
    _write_progress(
        batch_dir,
        {
            "status": "completed",
            "batch_id": batch_id,
            "total": len(plan.samples),
            "completed": stats["completed"],
            "failed": stats["failed"],
            "skipped": stats["skipped"],
            "finished_at": _utc_now(),
        },
    )
    _append_log(batch_dir, batch_id, f"[{_utc_now()}] Batch finished")


def prepare_batch_workspace(
    base_dir: Path,
    config: BatchConfig,
    *,
    resume: bool = False,
    override: bool = False,
) -> dict[str, Any]:
    """Reset progress on disk before the async worker starts (avoids stale UI totals)."""
    batch_dir = _batch_dir(base_dir, config.batch_name, config.output_dir)
    if resume:
        progress = _read_progress_file(batch_dir / PROGRESS_FILE, config.batch_name)
        if progress.get("status") == "idle":
            started_at = _utc_now()
            progress = {
                "status": "running",
                "batch_id": config.batch_name,
                "total": config.total_clips,
                "completed": 0,
                "failed": 0,
                "skipped": 0,
                "started_at": started_at,
            }
            _write_progress(batch_dir, progress)
        return progress

    if override and batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "status": "running",
        "batch_id": config.batch_name,
        "total": config.total_clips,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "started_at": _utc_now(),
    }
    _write_progress(batch_dir, progress)
    return progress


def start_batch_async(
    base_dir: Path,
    config: BatchConfig,
    *,
    resume: bool = False,
    override: bool = False,
) -> tuple[bool, str, dict[str, Any] | None]:
    with _lock:
        thread = _runner_state.get("thread")
        if thread is not None and thread.is_alive():
            return False, "A batch is already running.", None

        _runner_state["cancel"] = False
        initial_progress = prepare_batch_workspace(base_dir, config, resume=resume, override=override)

        def _target() -> None:
            try:
                run_batch_job(base_dir, config, resume=resume, override=override)
            finally:
                with _lock:
                    _runner_state["thread"] = None

        t = threading.Thread(target=_target, name="batch-generation", daemon=True)
        _runner_state["thread"] = t
        t.start()
    return True, "Batch started.", initial_progress


def batch_output_dir_exists(base_dir: Path, batch_name: str, output_dir: str = DEFAULT_BATCH_OUTPUT_DIR) -> bool:
    batch_dir = _batch_dir(base_dir, batch_name, output_dir)
    return batch_dir.exists()


def cancel_batch() -> None:
    with _lock:
        _runner_state["cancel"] = True


def batch_progress(base_dir: Path, batch_name: str, output_dir: str = DEFAULT_BATCH_OUTPUT_DIR) -> dict[str, Any]:
    path = _batch_dir(base_dir, batch_name, output_dir) / PROGRESS_FILE
    return _read_progress_file(path, batch_name)


def is_batch_running() -> bool:
    with _lock:
        thread = _runner_state.get("thread")
        return thread is not None and thread.is_alive()


def load_catalog(base_dir: Path, input_dir: str | None = None) -> VehicleCatalog:
    return scan_input_catalog(base_dir, input_dir)
