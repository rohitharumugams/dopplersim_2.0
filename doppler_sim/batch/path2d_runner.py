"""Batch orchestration for 2D whiteboard datasets."""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from doppler_sim.batch.catalog import scan_input_catalog
from doppler_sim.batch.constants import (
    DEFAULT_BATCH_OUTPUT_DIR,
    csv_headers,
    csv_speed_field_names,
    mps_to_display,
    resolve_batch_output_root,
    sample_dir_name,
)
from doppler_sim.batch.path2d_features import export_path2d_sample_artifacts
from doppler_sim.batch.path2d_pipeline import synthesize_path2d_planned_sample
from doppler_sim.batch.path2d_planner import (
    Path2dBatchConfig,
    Path2dBatchPlan,
    Path2dPlannedSample,
    build_path2d_batch_plan,
    path2d_batch_config_from_dict,
)
from doppler_sim.batch.pipeline import clear_source_cache
from doppler_sim.batch.runner import (
    PLAN_STATE_FILE,
    PROGRESS_FILE,
    SAMPLER_STATE_FILE,
    _append_clips_jsonl,
    _append_dataset_row,
    _append_log,
    _audio_clips_dir,
    _batch_dir,
    _configure_cpu_threads,
    _ensure_dataset_header,
    _init_worker_process,
    _progress_payload,
    _read_progress_file,
    _write_progress,
    scan_completed_indices,
)

_lock = threading.Lock()
_runner_state: dict[str, Any] = {
    "thread": None,
    "cancel": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip_to_csv_row(
    batch_id: str,
    plan: Path2dPlannedSample,
    wav_name: str,
    config: Path2dBatchConfig,
    *,
    labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_key, speed_key = csv_speed_field_names(config.speed_unit)
    labels = labels or {}
    cpa_time = labels.get("cpa_time_sec", plan.cpa_time_sec)
    cpa_dist = labels.get("cpa_distance_m", plan.cpa_distance_m)
    return {
        "sample_id": sample_dir_name(plan.index),
        "batch_id": batch_id,
        "filename": wav_name,
        "vehicle_class": plan.vehicle,
        "trajectory_type": plan.path_type,
        source_key: mps_to_display(plan.source_speed_mps, config.speed_unit),
        speed_key: mps_to_display(plan.speed_mps, config.speed_unit),
        "cpa_distance_m": cpa_dist,
        "cpa_distance_plan_m": plan.cpa_distance_m,
        "cpa_time_sec": cpa_time,
        "cpa_time_plan_sec": plan.cpa_time_sec,
        "vehicle_length_m": plan.vehicle_length_m,
        "num_emitters": plan.num_emitters,
        "pass_by_in_clip": True,
    }


def _write_batch_metadata(
    batch_dir: Path, batch_id: str, config: Path2dBatchConfig, stats: dict[str, Any]
) -> None:
    payload = {
        "batch_id": batch_id,
        "config": config.to_dict(),
        "statistics": stats,
        "timestamp": _utc_now(),
    }
    (batch_dir / f"metadata_{batch_id}.json").write_text(
        __import__("json").dumps(payload, indent=2),
        encoding="utf-8",
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
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _is_cancelled() -> bool:
    with _lock:
        return bool(_runner_state.get("cancel"))


def _process_planned_sample(
    base_dir: Path,
    sample: Path2dPlannedSample,
    config: Path2dBatchConfig,
    batch_id: str,
) -> dict[str, Any]:
    sample_index = sample.index
    try:
        batch_dir = _batch_dir(base_dir, config.batch_name, config.output_dir)
        audio_dir = _audio_clips_dir(batch_dir)
        sample_dir = audio_dir / sample_dir_name(sample.index)

        audio, quantities, aux = synthesize_path2d_planned_sample(sample, base_dir=base_dir)
        artifact = export_path2d_sample_artifacts(
            sample_dir,
            sample,
            audio,
            quantities,
            batch_id,
            config,
            trajectory=aux["trajectory"],
            derived_cpa_time=float(aux["cpa_time_sec"]),
            derived_cpa_distance=float(aux["cpa_distance_m"]),
        )
        row = _clip_to_csv_row(
            batch_id,
            sample,
            artifact["wav_name"],
            config,
            labels=artifact.get("labels"),
        )
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
                f"v2={sample.speed_mps} path_pts={len(sample.path_xy)} "
                f"(worker pid={worker_id})"
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "index": sample_index,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _run_path2d_batch_sample_task(
    base_dir_str: str,
    sample_data: dict[str, Any],
    config_data: dict[str, Any],
    batch_id: str,
) -> dict[str, Any]:
    return _process_planned_sample(
        Path(base_dir_str),
        Path2dPlannedSample(**sample_data),
        path2d_batch_config_from_dict(config_data),
        batch_id,
    )


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


def _run_samples_sequential(
    *,
    base_dir: Path,
    batch_dir: Path,
    batch_id: str,
    config: Path2dBatchConfig,
    plan: Path2dBatchPlan,
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
            audio, quantities, aux = synthesize_path2d_planned_sample(
                sample, base_dir=base_dir
            )
            artifact = export_path2d_sample_artifacts(
                sample_dir,
                sample,
                audio,
                quantities,
                batch_id,
                config,
                trajectory=aux["trajectory"],
                derived_cpa_time=float(aux["cpa_time_sec"]),
                derived_cpa_distance=float(aux["cpa_distance_m"]),
            )
            row = _clip_to_csv_row(
                batch_id,
                sample,
                artifact["wav_name"],
                config,
                labels=artifact.get("labels"),
            )
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


def _run_samples_parallel(
    *,
    base_dir: Path,
    batch_dir: Path,
    batch_id: str,
    config: Path2dBatchConfig,
    pending_samples: list[Path2dPlannedSample],
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

    _write_progress(
        batch_dir,
        _progress_payload(batch_id, config, stats, in_flight=0, started_at=started_at),
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
        future_to_sample = {
            executor.submit(
                _run_path2d_batch_sample_task,
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


def run_path2d_batch_job(
    base_dir: Path, config: Path2dBatchConfig, *, resume: bool = False, override: bool = False
) -> None:
    _configure_cpu_threads()
    batch_id = config.batch_name
    batch_dir = _batch_dir(base_dir, batch_id, config.output_dir)
    audio_dir = _audio_clips_dir(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    plan_path = batch_dir / PLAN_STATE_FILE
    sampler_path = batch_dir / SAMPLER_STATE_FILE

    if resume and plan_path.exists():
        plan = Path2dBatchPlan.load(plan_path)
        config = plan.config
        batch_id = config.batch_name
    else:
        catalog = scan_input_catalog(base_dir, config.input_dir)
        plan, bank = build_path2d_batch_plan(config, catalog, base_dir)
        plan.save(plan_path)
        bank.save(sampler_path)

    completed = scan_completed_indices(audio_dir) if resume else set()
    stats = {
        "total": len(plan.samples),
        "completed": len(completed),
        "failed": 0,
        "skipped": 0,
    }

    existing_progress = _read_progress_file(batch_dir / PROGRESS_FILE, batch_id)
    started_at = existing_progress.get("started_at") or _utc_now()

    _write_progress(
        batch_dir,
        _progress_payload(batch_id, config, stats, started_at=started_at),
    )
    _append_log(
        batch_dir,
        batch_id,
        f"[{started_at}] 2D whiteboard batch started "
        f"(resume={resume}, workers={max(1, int(config.num_workers))})",
    )

    dataset_headers = csv_headers(config.speed_unit)
    _ensure_dataset_header(batch_dir, dataset_headers)
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


def prepare_path2d_batch_workspace(
    base_dir: Path,
    config: Path2dBatchConfig,
    *,
    resume: bool = False,
    override: bool = False,
) -> dict[str, Any]:
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


def start_path2d_batch_async(
    base_dir: Path,
    config: Path2dBatchConfig,
    *,
    resume: bool = False,
    override: bool = False,
) -> tuple[bool, str, dict[str, Any] | None]:
    with _lock:
        thread = _runner_state.get("thread")
        if thread is not None and thread.is_alive():
            return False, "A batch is already running.", None

        _runner_state["cancel"] = False
        initial_progress = prepare_path2d_batch_workspace(
            base_dir, config, resume=resume, override=override
        )

        def _target() -> None:
            try:
                run_path2d_batch_job(base_dir, config, resume=resume, override=override)
            finally:
                with _lock:
                    _runner_state["thread"] = None

        t = threading.Thread(
            target=_target, name="path2d-whiteboard-batch", daemon=False
        )
        _runner_state["thread"] = t
        t.start()
    return True, "2D whiteboard batch started.", initial_progress


def path2d_batch_output_dir_exists(
    base_dir: Path, batch_name: str, output_dir: str = DEFAULT_BATCH_OUTPUT_DIR
) -> bool:
    return _batch_dir(base_dir, batch_name, output_dir).exists()


def cancel_path2d_batch() -> None:
    with _lock:
        _runner_state["cancel"] = True


def path2d_batch_progress(
    base_dir: Path, batch_name: str, output_dir: str = DEFAULT_BATCH_OUTPUT_DIR
) -> dict[str, Any]:
    path = _batch_dir(base_dir, batch_name, output_dir) / PROGRESS_FILE
    return _read_progress_file(path, batch_name)


def is_path2d_batch_running() -> bool:
    with _lock:
        thread = _runner_state.get("thread")
        return thread is not None and thread.is_alive()
