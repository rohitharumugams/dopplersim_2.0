"""Batch generation orchestration with resume support."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import shutil
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from doppler_sim.batch.catalog import VehicleCatalog, scan_input_catalog
from doppler_sim.batch.constants import (
    DEFAULT_BATCH_OUTPUT_DIR,
    csv_headers,
    csv_speed_field_names,
    mps_to_display,
    resolve_batch_output_root,
    sample_dir_name,
)
from doppler_sim.batch.features import export_sample_artifacts
from doppler_sim.batch.pipeline import (
    clear_source_cache,
    synthesize_planned_sample,
)
from doppler_sim.batch.planner import (
    BatchConfig,
    BatchPlan,
    PlannedSample,
    batch_config_from_dict,
    build_batch_plan,
)
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
            index = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        has_wav = any(
            item.is_file() and item.suffix.lower() == ".wav"
            for item in child.iterdir()
        )
        if has_wav:
            done.add(index)
    return done


def _configure_cpu_threads() -> None:
    """
    Limit BLAS/OpenMP threads per worker to avoid CPU oversubscription.

    On Apple Silicon the Accelerate framework ignores OMP_NUM_THREADS but
    does respect VECLIB_MAXIMUM_THREADS; set all variants so any BLAS
    backend is covered.
    """
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"


def _init_worker_process() -> None:
    """Process-pool initializer: limit BLAS threads and set matplotlib backend."""
    _configure_cpu_threads()
    import matplotlib

    matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Thread-safe I/O helpers
# ---------------------------------------------------------------------------

_progress_lock = threading.Lock()
_io_lock = threading.Lock()


def _is_cancelled() -> bool:
    with _lock:
        return bool(_runner_state.get("cancel"))


def _write_progress(batch_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically write progress.json — no pre-read, no tmp file dance."""
    data = json.dumps(payload, indent=2)
    path = batch_dir / PROGRESS_FILE
    with _progress_lock:
        path.write_text(data, encoding="utf-8")


def _read_progress_file(path: Path, batch_name: str) -> dict[str, Any]:
    if not path.exists():
        return {"status": "idle", "batch_id": batch_name}
    for attempt in range(3):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                return {"status": "idle", "batch_id": batch_name}
            return json.loads(raw)
        except (json.JSONDecodeError, OSError):
            if attempt < 2:
                time.sleep(0.02)
    return {"status": "idle", "batch_id": batch_name}


def _append_log(batch_dir: Path, batch_id: str, line: str) -> None:
    log_path = batch_dir / f"generation_log_{batch_id}.txt"
    with _io_lock:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line.rstrip() + "\n")


def _append_clips_jsonl(batch_dir: Path, record: dict[str, Any]) -> None:
    with _io_lock:
        with (batch_dir / "clips_metadata.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def _ensure_dataset_header(batch_dir: Path, headers: list[str]) -> None:
    csv_path = batch_dir / "dataset.csv"
    with _io_lock:
        if csv_path.exists():
            return
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(headers)


def _append_dataset_row(batch_dir: Path, row: dict[str, Any], headers: list[str]) -> None:
    _ensure_dataset_header(batch_dir, headers)
    with _io_lock:
        with (batch_dir / "dataset.csv").open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=headers).writerow(
                {k: row.get(k, "") for k in headers}
            )


def _clip_to_csv_row(
    batch_id: str,
    plan: PlannedSample,
    wav_name: str,
    config: BatchConfig,
) -> dict[str, Any]:
    source_key, speed_key = csv_speed_field_names(config.speed_unit)
    return {
        "sample_id": sample_dir_name(plan.index),
        "batch_id": batch_id,
        "filename": wav_name,
        "vehicle_class": plan.vehicle,
        "trajectory_type": plan.path_type,
        source_key: mps_to_display(plan.source_speed_mps, config.speed_unit),
        speed_key: mps_to_display(plan.speed_mps, config.speed_unit),
        "cpa_distance_m": plan.cpa_distance_m,
        "cpa_time_sec": plan.cpa_time_sec,
        "vehicle_length_m": plan.vehicle_length_m,
        "num_emitters": plan.num_emitters,
        "pass_by_in_clip": True,
    }


def _write_batch_metadata(
    batch_dir: Path, batch_id: str, config: BatchConfig, stats: dict[str, Any]
) -> None:
    payload = {
        "batch_id": batch_id,
        "config": config.to_dict(),
        "statistics": stats,
        "timestamp": _utc_now(),
    }
    (batch_dir / f"metadata_{batch_id}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        f"Batch: {batch_id}",
        f"Generated at: {payload['timestamp']}",
        f"Total requested: {stats.get('total', 0)}",
        f"Completed: {stats.get('completed', 0)}",
        f"Failed: {stats.get('failed', 0)}",
        f"Skipped (resume): {stats.get('skipped', 0)}",
    ]
    (batch_dir / f"statistics_{batch_id}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Per-sample worker (called by thread pool)
# ---------------------------------------------------------------------------

def _process_planned_sample(
    base_dir: Path,
    sample: PlannedSample,
    config: BatchConfig,
    batch_id: str,
) -> dict[str, Any]:
    """Synthesize and export one planned sample — thread-safe across distinct dirs."""
    sample_index = sample.index
    try:
        batch_dir = _batch_dir(base_dir, config.batch_name, config.output_dir)
        audio_dir = _audio_clips_dir(batch_dir)
        sample_dir = audio_dir / sample_dir_name(sample.index)

        audio, quantities, _aux = synthesize_planned_sample(sample, base_dir=base_dir)
        artifact = export_sample_artifacts(
            sample_dir, sample, audio, quantities, batch_id, config
        )
        row = _clip_to_csv_row(batch_id, sample, artifact["wav_name"], config)
        worker_id = os.getpid()
        return {
            "ok": True,
            "index": sample_index,
            "row": row,
            "clips_record": {
                "sample_id": row["sample_id"],
                "batch_id": batch_id,
                "plan": sample.to_dict(),
                "artifact": artifact,
            },
            "log_line": (
                f"sample {sample.index}: source={sample.source_path} "
                f"v2={sample.speed_mps} (worker pid={worker_id})"
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "index": sample_index,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _run_batch_sample_task(
    base_dir_str: str,
    sample_data: dict[str, Any],
    config_data: dict[str, Any],
    batch_id: str,
) -> dict[str, Any]:
    """Process-pool entry point (each worker is a separate Python process)."""
    return _process_planned_sample(
        Path(base_dir_str),
        PlannedSample(**sample_data),
        batch_config_from_dict(config_data),
        batch_id,
    )


# ---------------------------------------------------------------------------
# Result handling
# ---------------------------------------------------------------------------

def _process_sample_result(
    result: dict[str, Any],
    *,
    batch_dir: Path,
    batch_id: str,
    dataset_headers: list[str],
    stats: dict[str, Any],
    completed: set[int],
) -> None:
    sample_index = int(result.get("index", -1))
    if result.get("ok"):
        row = result["row"]
        _append_log(batch_dir, batch_id, f"[{_utc_now()}] {result.get('log_line', '')}")
        _append_dataset_row(batch_dir, row, dataset_headers)
        _append_clips_jsonl(batch_dir, result["clips_record"])
        stats["completed"] += 1
        completed.add(sample_index)
        return

    stats["failed"] += 1
    _append_log(
        batch_dir,
        batch_id,
        f"[{_utc_now()}] sample {sample_index} FAILED: {result.get('error')}",
    )
    tb = result.get("traceback")
    if tb:
        _append_log(batch_dir, batch_id, tb)


# ---------------------------------------------------------------------------
# Progress payload builder
# ---------------------------------------------------------------------------

def _progress_payload(
    batch_id: str,
    config: BatchConfig,
    stats: dict[str, Any],
    *,
    status: str = "running",
    current_sample: int | None = None,
    in_flight: int = 0,
    message: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "batch_id": batch_id,
        "total": stats["total"],
        "completed": stats["completed"],
        "failed": stats["failed"],
        "skipped": stats["skipped"],
        "num_workers": max(1, int(config.num_workers)),
        "in_flight": max(0, int(in_flight)),
    }
    if current_sample is not None:
        payload["current_sample"] = current_sample
    if message is not None:
        payload["message"] = message
    if started_at is not None:
        payload["started_at"] = started_at
    if finished_at is not None:
        payload["finished_at"] = finished_at
    return payload


# ---------------------------------------------------------------------------
# Sequential execution (num_workers == 1)
# ---------------------------------------------------------------------------

def _run_samples_sequential(
    *,
    base_dir: Path,
    batch_dir: Path,
    batch_id: str,
    config: BatchConfig,
    plan: BatchPlan,
    completed: set[int],
    stats: dict[str, Any],
    dataset_headers: list[str],
    started_at: str,
) -> bool:
    for sample in plan.samples:
        if _is_cancelled():
            _write_progress(
                batch_dir,
                _progress_payload(
                    batch_id,
                    config,
                    stats,
                    status="cancelled",
                    message="Cancelled by user",
                    started_at=started_at,
                ),
            )
            _append_log(batch_dir, batch_id, f"[{_utc_now()}] Batch cancelled")
            return False

        if sample.index in completed:
            continue

        sample_dir = _audio_clips_dir(batch_dir) / sample_dir_name(sample.index)
        try:
            _append_log(
                batch_dir,
                batch_id,
                f"[{_utc_now()}] sample {sample.index}: source={sample.source_path} "
                f"v2={sample.speed_mps}",
            )
            audio, quantities, _aux = synthesize_planned_sample(
                sample, base_dir=base_dir
            )
            artifact = export_sample_artifacts(
                sample_dir, sample, audio, quantities, batch_id, config
            )
            row = _clip_to_csv_row(batch_id, sample, artifact["wav_name"], config)
            _append_dataset_row(batch_dir, row, dataset_headers)
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
                _progress_payload(
                    batch_id,
                    config,
                    stats,
                    current_sample=sample.index,
                    started_at=started_at,
                ),
            )
        except Exception as exc:
            stats["failed"] += 1
            _append_log(
                batch_dir,
                batch_id,
                f"[{_utc_now()}] sample {sample.index} FAILED: {exc}",
            )
            _append_log(batch_dir, batch_id, traceback.format_exc())
    return True


# ---------------------------------------------------------------------------
# Parallel execution (num_workers > 1)
# ---------------------------------------------------------------------------

def _run_samples_parallel(
    *,
    base_dir: Path,
    batch_dir: Path,
    batch_id: str,
    config: BatchConfig,
    pending_samples: list[PlannedSample],
    completed: set[int],
    stats: dict[str, Any],
    dataset_headers: list[str],
    started_at: str,
) -> bool:
    workers = max(1, min(int(config.num_workers), len(pending_samples)))

    _append_log(
        batch_dir,
        batch_id,
        f"[{_utc_now()}] Starting {workers} worker processes for "
        f"{len(pending_samples)} clips",
    )

    # Write initial progress so the UI shows the correct state immediately
    _write_progress(
        batch_dir,
        _progress_payload(
            batch_id, config, stats, in_flight=0, started_at=started_at
        ),
    )

    if _is_cancelled():
        _write_progress(
            batch_dir,
            _progress_payload(
                batch_id,
                config,
                stats,
                status="cancelled",
                message="Cancelled by user",
                started_at=started_at,
            ),
        )
        return False

    config_payload = config.to_dict()
    mp_context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
        initializer=_init_worker_process,
    ) as executor:
        future_to_sample: dict[Any, PlannedSample] = {
            executor.submit(
                _run_batch_sample_task,
                str(base_dir),
                sample.to_dict(),
                config_payload,
                batch_id,
            ): sample
            for sample in pending_samples
        }

        in_flight_count = len(future_to_sample)

        for future in as_completed(future_to_sample):
            in_flight_count -= 1

            if _is_cancelled():
                # Cancel remaining pending futures (already-running ones finish)
                for f in future_to_sample:
                    f.cancel()
                _write_progress(
                    batch_dir,
                    _progress_payload(
                        batch_id,
                        config,
                        stats,
                        status="cancelled",
                        in_flight=in_flight_count,
                        message="Cancelled by user",
                        started_at=started_at,
                    ),
                )
                _append_log(batch_dir, batch_id, f"[{_utc_now()}] Batch cancelled")
                return False

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "ok": False,
                    "index": future_to_sample[future].index,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }

            _process_sample_result(
                result,
                batch_dir=batch_dir,
                batch_id=batch_id,
                dataset_headers=dataset_headers,
                stats=stats,
                completed=completed,
            )
            _write_progress(
                batch_dir,
                _progress_payload(
                    batch_id,
                    config,
                    stats,
                    current_sample=result.get("index"),
                    in_flight=in_flight_count,
                    started_at=started_at,
                ),
            )

    if stats["completed"] >= stats["total"]:
        _write_progress(
            batch_dir,
            _progress_payload(
                batch_id,
                config,
                stats,
                in_flight=0,
                message="All clips generated — finalizing…",
                started_at=started_at,
            ),
        )
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_batch_job(
    base_dir: Path, config: BatchConfig, *, resume: bool = False, override: bool = False
) -> None:
    _configure_cpu_threads()
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

    # Reuse the started_at that prepare_batch_workspace already wrote to disk.
    # The browser captured that timestamp as expectedRunStartedAt; if we mint
    # a new one here the isStaleCompletedProgress check will loop forever.
    existing_progress = _read_progress_file(batch_dir / PROGRESS_FILE, batch_id)
    started_at = existing_progress.get("started_at") or _utc_now()

    _write_progress(
        batch_dir,
        _progress_payload(batch_id, config, stats, started_at=started_at),
    )
    _append_log(
        batch_dir,
        batch_id,
        f"[{started_at}] Batch started "
        f"(resume={resume}, workers={max(1, int(config.num_workers))})",
    )

    dataset_headers = csv_headers(config.speed_unit)
    pending_samples = [
        sample for sample in plan.samples if sample.index not in completed
    ]

    try:
        if max(1, int(config.num_workers)) == 1:
            finished_ok = _run_samples_sequential(
                base_dir=base_dir,
                batch_dir=batch_dir,
                batch_id=batch_id,
                config=config,
                plan=plan,
                completed=completed,
                stats=stats,
                dataset_headers=dataset_headers,
                started_at=started_at,
            )
        else:
            finished_ok = _run_samples_parallel(
                base_dir=base_dir,
                batch_dir=batch_dir,
                batch_id=batch_id,
                config=config,
                pending_samples=pending_samples,
                completed=completed,
                stats=stats,
                dataset_headers=dataset_headers,
                started_at=started_at,
            )
    finally:
        clear_source_cache()

    if not finished_ok:
        return

    _write_progress(
        batch_dir,
        _progress_payload(
            batch_id,
            config,
            stats,
            status="completed",
            started_at=started_at,
            finished_at=_utc_now(),
        ),
    )
    _write_batch_metadata(batch_dir, batch_id, config, stats)
    _append_log(batch_dir, batch_id, f"[{_utc_now()}] Batch finished")


# ---------------------------------------------------------------------------
# Workspace / async helpers
# ---------------------------------------------------------------------------

def prepare_batch_workspace(
    base_dir: Path,
    config: BatchConfig,
    *,
    resume: bool = False,
    override: bool = False,
) -> dict[str, Any]:
    """Reset progress on disk before the async worker starts (avoids stale UI totals)."""
    batch_dir = _batch_dir(base_dir, config.batch_name, config.output_dir)
    workers = max(1, int(config.num_workers))
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
                "num_workers": workers,
                "in_flight": 0,
                "started_at": started_at,
            }
            _write_progress(batch_dir, progress)
        else:
            progress["num_workers"] = workers
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
        "num_workers": workers,
        "in_flight": 0,
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
        initial_progress = prepare_batch_workspace(
            base_dir, config, resume=resume, override=override
        )

        def _target() -> None:
            try:
                run_batch_job(base_dir, config, resume=resume, override=override)
            finally:
                with _lock:
                    _runner_state["thread"] = None

        t = threading.Thread(target=_target, name="batch-generation", daemon=False)
        _runner_state["thread"] = t
        t.start()
    return True, "Batch started.", initial_progress


def batch_output_dir_exists(
    base_dir: Path, batch_name: str, output_dir: str = DEFAULT_BATCH_OUTPUT_DIR
) -> bool:
    return _batch_dir(base_dir, batch_name, output_dir).exists()


def cancel_batch() -> None:
    with _lock:
        _runner_state["cancel"] = True


def batch_progress(
    base_dir: Path, batch_name: str, output_dir: str = DEFAULT_BATCH_OUTPUT_DIR
) -> dict[str, Any]:
    path = _batch_dir(base_dir, batch_name, output_dir) / PROGRESS_FILE
    return _read_progress_file(path, batch_name)


def is_batch_running() -> bool:
    with _lock:
        thread = _runner_state.get("thread")
        return thread is not None and thread.is_alive()


def load_catalog(base_dir: Path, input_dir: str | None = None) -> VehicleCatalog:
    return scan_input_catalog(base_dir, input_dir)
