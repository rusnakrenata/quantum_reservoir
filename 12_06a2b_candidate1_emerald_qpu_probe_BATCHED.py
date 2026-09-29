#!/usr/bin/env python
from __future__ import annotations

"""
WEEK 12.6A.2b — IQM CANDIDATE #1 BATCHED REAL-EMERALD QPU PROBE
================================================================

Fix for IQM server limit:
    max 100 circuits/job

The original probe contains:
    100 endpoints x 2 layouts x 2 settings = 400 circuits

Therefore this script submits exactly:
    4 jobs x 25 endpoints/job x 2 layouts x 2 settings = 100 circuits/job

Scientific logic is UNCHANGED:
- same two 12.6A.1 finalists
- same 100 deterministic 2022-2024 training endpoints
- same 512 shots/setting default
- same fixed layouts
- routing_method="none"
- optimization_level=1
- seed_transpiler=42
- true claim targets NOT used for layout selection
- layouts interleaved within every endpoint
- calibration_set_id must remain the exact 12.6A.1 ID
- one qrc_qpu_run row per actual IQM job
- winner frozen only after ALL FOUR jobs complete and all returned
  calibration_set_ids match the expected calibration

Default = dry run.
Use --submit to execute the four real Emerald jobs.

No full 2025 evaluation here.
2026 untouched.
"""

import argparse
import importlib.util
import json
import math
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit.providers import JobStatus

import db_objects as qpu_db
from iqm_account import get_client


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

A2_PATH = HERE / "12_06a2_candidate1_emerald_qpu_probe.py"

PREFIX = "12_06a2b_c1"
OUT_BATCH_PLAN = RESULTS / f"{PREFIX}_batch_plan.csv"
OUT_PREFLIGHT = RESULTS / f"{PREFIX}_preflight.json"
OUT_RAW = RESULTS / f"{PREFIX}_raw_counts.csv"
OUT_DAILY = RESULTS / f"{PREFIX}_probe_predictions.csv"
OUT_SUMMARY = RESULTS / f"{PREFIX}_probe_summary.csv"
OUT_WINNER = RESULTS / f"{PREFIX}_winner.json"
OUT_MANIFEST = RESULTS / f"{PREFIX}_manifest.json"

MAX_CIRCUITS_PER_JOB = 100
ENDPOINTS_PER_JOB = 25
EXPECTED_N_BATCHES = 4

BACKEND_NAME = "emerald"
EXPECTED_CANDIDATE = "H6_W06_r2_local_04_XZinj_plus_YX45"

DEFAULT_SHOTS = 512
DEFAULT_TIMEOUT_SECONDS = 4 * 60 * 60
POLL_SECONDS = 5


def import_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required script: {path}\n"
            "Place this script beside 12_06a2_candidate1_emerald_qpu_probe.py."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a2 = import_from_path("qrc_12_06a2", A2_PATH)
base = a2.base


def utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def safe(x: Any):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [safe(v) for v in x]
    if hasattr(x, "model_dump"):
        try:
            return safe(x.model_dump(mode="json"))
        except Exception:
            try:
                return safe(x.model_dump())
            except Exception:
                pass
    if hasattr(x, "__dict__"):
        try:
            return {
                str(k): safe(v)
                for k, v in vars(x).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass
    return str(x)


def write_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe(payload), f, indent=2, ensure_ascii=False)


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def current_calibration_id(client):
    return a2.current_default_calibration_id(client)


def extract_server_failure(job):
    """Retrieve IQM server errors/messages/timeline from the underlying CircuitJob."""
    iqm_job = getattr(job, "_iqm_job", None)
    if iqm_job is None:
        return {
            "errors": None,
            "messages": None,
            "timeline": None,
        }

    try:
        iqm_job.update()
    except Exception:
        pass

    data = getattr(iqm_job, "data", None)
    return {
        "errors": safe(getattr(data, "errors", None)),
        "messages": safe(getattr(data, "messages", None)),
        "timeline": safe(getattr(data, "timeline", None)),
    }


def build_batches(circuits, meta, resources_df, endpoints):
    """
    Keep every endpoint complete inside one IQM job:
        25 endpoints x 2 layouts x 2 settings = 100 circuits.
    """
    endpoints = list(map(int, endpoints))

    if len(endpoints) != 100:
        raise RuntimeError(f"Expected 100 probe endpoints, found {len(endpoints)}.")

    batches = []
    plan_rows = []

    for batch_index in range(EXPECTED_N_BATCHES):
        start = batch_index * ENDPOINTS_PER_JOB
        stop = start + ENDPOINTS_PER_JOB
        batch_eps = set(endpoints[start:stop])

        indices = [
            i for i, m in enumerate(meta)
            if int(m["endpoint"]) in batch_eps
        ]

        if len(indices) != MAX_CIRCUITS_PER_JOB:
            raise RuntimeError(
                f"Batch {batch_index+1}: expected {MAX_CIRCUITS_PER_JOB} circuits, "
                f"found {len(indices)}."
            )

        batch_circuits = [circuits[i] for i in indices]
        batch_meta = [meta[i] for i in indices]
        batch_resources = resources_df.iloc[indices].reset_index(drop=True)

        batch = {
            "batch_id": batch_index + 1,
            "endpoint_start_index": start,
            "endpoint_stop_index_exclusive": stop,
            "endpoints": endpoints[start:stop],
            "circuits": batch_circuits,
            "meta": batch_meta,
            "resources": batch_resources,
        }
        batches.append(batch)

        plan_rows.append({
            "batch_id": batch_index + 1,
            "n_endpoints": len(batch["endpoints"]),
            "first_endpoint": min(batch["endpoints"]),
            "last_endpoint": max(batch["endpoints"]),
            "n_circuits": len(batch_circuits),
        })

    plan = pd.DataFrame(plan_rows)

    if int(plan["n_circuits"].sum()) != 400:
        raise RuntimeError("Batched plan does not contain exactly 400 circuits.")

    return batches, plan


def create_batch_db_row(
    batch_group_uuid,
    batch,
    candidate,
    ref,
    layouts,
    expected_cal,
    shots,
):
    resources = batch["resources"]

    j_json = {
        f"J{i}{j}": float(v)
        for (i, j), v in candidate["J"].items()
    }

    hardware = {
        "selection_method": "qrc_aware_two_layout_training_probe",
        "batch_group_uuid": batch_group_uuid,
        "batch_id": int(batch["batch_id"]),
        "n_batches": EXPECTED_N_BATCHES,
        "layouts": layouts,
        "calibration_set_id": expected_cal,
        "selection_metric": "RMS QPU-vs-ideal prediction distortion",
        "true_targets_used": False,
        "layout_order": "cyclic_interleaved",
        "iqm_max_circuits_per_job": MAX_CIRCUITS_PER_JOB,
    }

    manifest = {
        "stage": "12.6A.2b",
        "candidate_id": EXPECTED_CANDIDATE,
        "backend": BACKEND_NAME,
        "batch_group_uuid": batch_group_uuid,
        "batch_id": int(batch["batch_id"]),
        "n_batches": EXPECTED_N_BATCHES,
        "expected_calibration_set_id": expected_cal,
        "probe_scope": "2022-2024_training_only",
        "batch_endpoints": list(map(int, batch["endpoints"])),
        "shots_per_setting": int(shots),
        "n_circuits": int(len(batch["circuits"])),
        "optimization_level": a2.OPT_LEVEL,
        "seed_transpiler": a2.SEED_TRANSPILE,
        "routing_method": "none",
        "initial_layout": "fixed_per_finalist",
        "true_targets_used_for_ranking": False,
        "full_2025": False,
        "test_2026_used": False,
    }

    return qpu_db.create_run(
        script_name=Path(__file__).name,
        run_status="PREFLIGHT_COMPLETE",
        forecast_dataset_name="property_damage_next_day_v1",
        feature_set_name="F4",
        evaluation_split="training_probe_2022_2024",
        candidate_key=EXPECTED_CANDIDATE,
        selected_by_rules="R1",
        protocol="RWP",
        topology="H6",
        window_size=int(candidate["window"]),
        trotter_r=int(candidate["r"]),
        readout_name=str(candidate["readout"]),
        primitive_name="IQMBackend.run",
        measurement_method="grouped_Pauli_counts",
        alpha=float(candidate["alpha"]),
        dt=float(candidate["dt"]),
        hx=float(candidate["hx"]),
        hy=float(candidate["hy"]),
        j_json=j_json,
        manifest_json=manifest,
        ridge_lambda=float(ref["selected_lambda"]),
        training_cv_rmse=float(ref["cv_rmse"]),
        backend_name=BACKEND_NAME,
        physical_layout_json=layouts,
        hardware_selection_json=hardware,
        shots_per_setting=int(shots),
        # Historical schema name; here it stores TRAINING probe endpoints in this batch.
        n_validation_endpoints=int(len(batch["endpoints"])),
        n_circuits=int(len(batch["circuits"])),
        full_2025=False,
        pilot_selection=f"qrc_aware_training_probe_batch_{batch['batch_id']}_of_4",
        optimization_level=int(a2.OPT_LEVEL),
        seed_transpiler=int(a2.SEED_TRANSPILE),
        median_cz_per_feature=float(
            resources.groupby(["endpoint", "layout_id"])["n_cz"].sum().median()
        ),
        median_max_setting_depth=float(
            resources.groupby(["endpoint", "layout_id"])["depth"].max().median()
        ),
        notes=(
            f"IQM Candidate-1 training-only layout probe batch "
            f"{batch['batch_id']}/4. 25 endpoints, 100 circuits. "
            "True claim targets not used for layout ranking."
        ),
    )


def wait_one_job(job, client, expected_cal, run_uuid, timeout_seconds):
    start = time.monotonic()
    last_status = None

    while True:
        status = job.status()

        if status != last_status:
            print(f"    status: {status}")
            last_status = status
            qpu_db.update_run(
                run_uuid,
                job_status=str(status),
            )

        if status in {
            JobStatus.DONE,
            JobStatus.ERROR,
            JobStatus.CANCELLED,
        }:
            break

        current = current_calibration_id(client)
        if current != expected_cal:
            try:
                job.cancel()
            except Exception:
                pass

            qpu_db.update_run(
                run_uuid,
                run_status="ABORTED_CALIBRATION_CHANGED",
                completed_at_utc=utc_now_naive(),
                job_status=str(job.status()),
                error_message=(
                    f"Calibration changed during batch: "
                    f"expected={expected_cal}, current={current}"
                ),
            )

            raise RuntimeError(
                "Emerald calibration changed during batched probe. "
                "Stop and rerun 12.6A.1."
            )

        if time.monotonic() - start > timeout_seconds:
            try:
                job.cancel()
            except Exception:
                pass

            qpu_db.update_run(
                run_uuid,
                run_status="FAILED_TIMEOUT",
                completed_at_utc=utc_now_naive(),
                job_status=str(job.status()),
                error_message=f"Timeout after {timeout_seconds}s",
            )

            raise TimeoutError(
                f"Batch timed out after {timeout_seconds}s."
            )

        time.sleep(POLL_SECONDS)

    if status != JobStatus.DONE:
        detail = extract_server_failure(job)

        qpu_db.update_run(
            run_uuid,
            run_status="FAILED_QPU_JOB",
            completed_at_utc=utc_now_naive(),
            job_status=str(status),
            error_message=json.dumps(detail, ensure_ascii=False),
            job_metrics_json=detail,
        )

        raise RuntimeError(
            f"IQM batch ended with {status}.\n"
            f"Server detail:\n{json.dumps(detail, indent=2, ensure_ascii=False)}"
        )

    return job.result()


def parse_batch_result(result, batch_meta, expected_cal):
    rows = []
    returned_ids = []

    for i, m in enumerate(batch_meta):
        counts = result.get_counts(i)
        cdict = {str(k): int(v) for k, v in counts.items()}

        cal_id = None
        try:
            cal_id = getattr(
                result.results[i],
                "calibration_set_id",
                None,
            )
        except Exception:
            pass

        if cal_id is not None:
            returned_ids.append(str(cal_id))

        rows.append({
            **m,
            "calibration_set_id_returned": (
                str(cal_id) if cal_id is not None else None
            ),
            "shots_returned": int(sum(cdict.values())),
            "counts_json": json.dumps(cdict, sort_keys=True),
        })

    unique_ids = sorted(set(returned_ids))

    if unique_ids != [expected_cal]:
        raise RuntimeError(
            f"Returned calibration mismatch. "
            f"Expected [{expected_cal}], got {unique_ids}."
        )

    return pd.DataFrame(rows), unique_ids


def aggregate_probe(raw_df, layouts, endpoints, ref):
    by_layout = {
        a2.layout_key(layout): {
            int(ep): {}
            for ep in endpoints
        }
        for layout in layouts
    }

    for _, row in raw_df.iterrows():
        layout = json.loads(str(row["layout"]))
        key = a2.layout_key(layout)
        counts = json.loads(str(row["counts_json"]))

        by_layout[key][int(row["endpoint"])][str(row["setting"])] = {
            str(k): int(v)
            for k, v in counts.items()
        }

    X_ideal = np.asarray(ref["X_probe_ideal"], dtype=float)
    pred_ideal = np.asarray(ref["pred_probe_ideal"], dtype=float)

    daily_rows = []
    summary_rows = []

    for layout_i, layout in enumerate(layouts, start=1):
        key = a2.layout_key(layout)
        X_qpu = []

        for ep in endpoints:
            settings = by_layout[key][int(ep)]

            if set(settings) != {"XXXXYX", "ZZZZZZ"}:
                raise RuntimeError(
                    f"Incomplete counts for layout={layout}, endpoint={ep}: "
                    f"{list(settings)}"
                )

            X_qpu.append(
                base.candidate1_features(settings)
            )

        X_qpu = np.asarray(X_qpu, dtype=float)

        pred_qpu = base.common.predict_scaled_ridge(
            ref["model"],
            ref["scaler"],
            ref["keep"],
            X_qpu,
        )

        d_pred = pred_qpu - pred_ideal
        d_feat = X_qpu - X_ideal

        for j, ep in enumerate(endpoints):
            daily_rows.append({
                "layout_id": f"L{layout_i}",
                "layout": json.dumps(layout),
                "endpoint": int(ep),
                "pred_qpu": float(pred_qpu[j]),
                "pred_ideal": float(pred_ideal[j]),
                "prediction_distortion": float(d_pred[j]),
                "abs_prediction_distortion": float(abs(d_pred[j])),
                "feature_rmse_vs_ideal": float(
                    np.sqrt(np.mean(d_feat[j] ** 2))
                ),
            })

        summary_rows.append({
            "layout_id": f"L{layout_i}",
            "layout": json.dumps(layout),
            "qpu_pred_distortion_rmse": rmse(
                pred_qpu,
                pred_ideal,
            ),
            "qpu_pred_distortion_mae": mae(
                pred_qpu,
                pred_ideal,
            ),
            "qpu_feature_rmse_vs_ideal": rmse(
                X_qpu.ravel(),
                X_ideal.ravel(),
            ),
            "prediction_correlation_qpu_vs_ideal": corr(
                pred_qpu,
                pred_ideal,
            ),
            "n_training_probe_endpoints": int(len(endpoints)),
            "targets_used_for_layout_ranking": False,
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "qpu_pred_distortion_rmse",
            "qpu_feature_rmse_vs_ideal",
        ],
        ascending=[True, True],
    ).reset_index(drop=True)

    summary["qpu_probe_rank"] = np.arange(
        1,
        len(summary) + 1,
    )

    return pd.DataFrame(daily_rows), summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the four real Emerald jobs.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--overwrite-local",
        action="store_true",
    )
    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError("--shots must be positive.")

    print("=" * 136)
    print("WEEK 12.6A.2b — CANDIDATE #1 BATCHED REAL-EMERALD QPU PROBE")
    print("=" * 136)
    print("IQM limit: max 100 circuits/job.")
    print("Plan: 4 jobs x 100 circuits = same original 400-circuit probe.")
    print("True claim targets NOT used. 2025 not evaluated. 2026 untouched.")
    print()

    (
        a1_manifest,
        a1_identity,
        finalist_df,
        endpoints,
        expected_cal,
        layouts,
    ) = a2.load_a1_state()

    candidate, source_row, ref = a2.rebuild_reference(
        endpoints
    )

    client = get_client(
        quantum_computer=BACKEND_NAME
    )

    current_cal = current_calibration_id(client)

    print("CALIBRATION GUARD")
    print("-" * 136)
    print(f"12.6A.1: {expected_cal}")
    print(f"Current:  {current_cal}")
    print(
        f"Match:    "
        f"{'PASS' if current_cal == expected_cal else 'FAIL'}"
    )

    if current_cal != expected_cal:
        print("ZERO QPU JOBS SUBMITTED.")
        print("Rerun 12.6A.1 because Emerald calibration changed.")
        return

    backend = a2.create_pinned_backend(
        expected_cal
    )

    print()
    print("COMPILE SAME INTERLEAVED 400-CIRCUIT PROBE")
    print("-" * 136)

    circuits, meta, resources_df, position_audit = (
        a2.compile_interleaved_probe(
            backend=backend,
            candidate=candidate,
            ref=ref,
            endpoints=endpoints,
            layouts=layouts,
        )
    )

    if len(circuits) != 400:
        raise RuntimeError(
            f"Expected 400 compiled circuits, got {len(circuits)}."
        )

    batches, plan = build_batches(
        circuits,
        meta,
        resources_df,
        endpoints,
    )

    plan.to_csv(
        OUT_BATCH_PLAN,
        index=False,
    )

    print(plan.to_string(index=False))

    for b in batches:
        request = backend.create_run_request(
            b["circuits"],
            shots=int(args.shots),
        )
        request_cal = str(
            getattr(
                request,
                "calibration_set_id",
                expected_cal,
            )
        )

        if request_cal != expected_cal:
            raise RuntimeError(
                f"Batch {b['batch_id']} run-request calibration mismatch: "
                f"{request_cal}"
            )

    preflight = {
        "candidate": EXPECTED_CANDIDATE,
        "backend": BACKEND_NAME,
        "expected_calibration_set_id": expected_cal,
        "current_calibration_set_id": current_cal,
        "n_total_endpoints": len(endpoints),
        "n_total_circuits": len(circuits),
        "n_jobs": len(batches),
        "max_circuits_per_job": MAX_CIRCUITS_PER_JOB,
        "endpoints_per_job": ENDPOINTS_PER_JOB,
        "shots_per_setting": int(args.shots),
        "optimization_level": int(a2.OPT_LEVEL),
        "seed_transpiler": int(a2.SEED_TRANSPILE),
        "routing_method": "none",
        "max_swap_count": int(resources_df["n_swap"].max()),
        "batch_plan": plan.to_dict(orient="records"),
        "true_targets_used": False,
        "full_2025": False,
        "test_2026_used": False,
    }
    write_json(OUT_PREFLIGHT, preflight)

    if not args.submit:
        print()
        print("=" * 136)
        print("12.6A.2b DRY-RUN COMPLETE")
        print("=" * 136)
        print("Calibration guard: PASS")
        print("4 run requests: PASS")
        print("Each IQM job has exactly 100 circuits.")
        print("QPU jobs submitted: 0")
        print()
        print(
            f"Run real probe with:\n"
            f"  python {Path(__file__).name} --submit"
        )
        return

    batch_group_uuid = str(uuid.uuid4())
    all_raw = []
    run_records = []

    print()
    print("=" * 136)
    print("REAL EMERALD EXECUTION")
    print("=" * 136)
    print(f"Batch group UUID: {batch_group_uuid}")

    for batch in batches:
        batch_id = int(batch["batch_id"])

        print()
        print("-" * 136)
        print(
            f"BATCH {batch_id}/{EXPECTED_N_BATCHES}: "
            f"{len(batch['endpoints'])} endpoints, "
            f"{len(batch['circuits'])} circuits"
        )
        print("-" * 136)

        # Hard calibration guard before every actual job.
        current_cal = current_calibration_id(client)
        if current_cal != expected_cal:
            raise RuntimeError(
                f"Calibration changed before batch {batch_id}: "
                f"expected={expected_cal}, current={current_cal}. "
                "Stop; do not freeze a winner."
            )

        run_uuid = create_batch_db_row(
            batch_group_uuid=batch_group_uuid,
            batch=batch,
            candidate=candidate,
            ref=ref,
            layouts=layouts,
            expected_cal=expected_cal,
            shots=args.shots,
        )

        qpu_db.save_dataframe_artifact(
            run_uuid,
            f"{PREFIX}_batch_{batch_id}_resources.csv",
            batch["resources"],
        )

        submitted_at = utc_now_naive()

        try:
            job = backend.run(
                batch["circuits"],
                shots=int(args.shots),
            )

            job_id = str(job.job_id())

            qpu_db.update_run(
                run_uuid,
                run_status="SUBMITTED",
                submitted_at_utc=submitted_at,
                job_id=job_id,
                job_status=str(job.status()),
            )

            print(f"  run_uuid: {run_uuid}")
            print(f"  IQM job : {job_id}")

            result = wait_one_job(
                job=job,
                client=client,
                expected_cal=expected_cal,
                run_uuid=run_uuid,
                timeout_seconds=int(args.timeout_seconds),
            )

            batch_raw, returned_ids = parse_batch_result(
                result=result,
                batch_meta=batch["meta"],
                expected_cal=expected_cal,
            )

            batch_raw["batch_id"] = batch_id
            batch_raw["run_uuid"] = run_uuid
            batch_raw["job_id"] = job_id

            batch_file = (
                RESULTS
                / f"{PREFIX}_batch_{batch_id}_raw_counts.csv"
            )
            batch_raw.to_csv(
                batch_file,
                index=False,
            )

            qpu_db.save_dataframe_artifact(
                run_uuid,
                batch_file.name,
                batch_raw,
            )

            qpu_db.update_run(
                run_uuid,
                run_status="COMPLETED_BATCH",
                completed_at_utc=utc_now_naive(),
                job_status=str(job.status()),
                job_metrics_json={
                    "batch_group_uuid": batch_group_uuid,
                    "batch_id": batch_id,
                    "returned_calibration_set_ids": returned_ids,
                },
                notes=(
                    f"Completed IQM Candidate-1 probe batch {batch_id}/4. "
                    "Awaiting all batches before layout winner is frozen."
                ),
            )

            all_raw.append(batch_raw)

            run_records.append({
                "batch_id": batch_id,
                "run_uuid": run_uuid,
                "job_id": job_id,
                "returned_calibration_set_ids": returned_ids,
            })

            print("  batch result: PASS")

        except Exception as exc:
            try:
                qpu_db.update_run(
                    run_uuid,
                    run_status="FAILED",
                    completed_at_utc=utc_now_naive(),
                    error_message=str(exc),
                )
            except Exception:
                pass
            raise

    raw_df = pd.concat(
        all_raw,
        ignore_index=True,
    )

    if len(raw_df) != 400:
        raise RuntimeError(
            f"Expected 400 returned circuit rows, got {len(raw_df)}."
        )

    raw_df.to_csv(
        OUT_RAW,
        index=False,
    )

    daily_df, summary_df = aggregate_probe(
        raw_df=raw_df,
        layouts=layouts,
        endpoints=endpoints,
        ref=ref,
    )

    daily_df.to_csv(
        OUT_DAILY,
        index=False,
    )
    summary_df.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    winner = summary_df.iloc[0].to_dict()
    winner_layout = json.loads(
        str(winner["layout"])
    )

    winner_payload = {
        "candidate_id": EXPECTED_CANDIDATE,
        "backend": BACKEND_NAME,
        "calibration_set_id": expected_cal,
        "batch_group_uuid": batch_group_uuid,
        "batch_runs": run_records,
        "winner_layout_id": winner["layout_id"],
        "winner_layout": winner_layout,
        "selection_metric": (
            "RMS QPU-vs-ideal prediction distortion on "
            "100 deterministic 2022-2024 training endpoints"
        ),
        "qpu_pred_distortion_rmse": float(
            winner["qpu_pred_distortion_rmse"]
        ),
        "qpu_feature_rmse_vs_ideal": float(
            winner["qpu_feature_rmse_vs_ideal"]
        ),
        "prediction_correlation_qpu_vs_ideal": float(
            winner["prediction_correlation_qpu_vs_ideal"]
        ),
        "targets_used_for_layout_ranking": False,
        "full_2025_opened": False,
        "test_2026_used": False,
        "all_layout_scores": summary_df.to_dict(
            orient="records"
        ),
    }

    write_json(
        OUT_WINNER,
        winner_payload,
    )

    manifest = {
        "stage": "12.6A.2b",
        "status": "COMPLETED",
        "batch_group_uuid": batch_group_uuid,
        "expected_calibration_set_id": expected_cal,
        "n_jobs": EXPECTED_N_BATCHES,
        "n_total_circuits": 400,
        "shots_per_setting": int(args.shots),
        "batch_runs": run_records,
        "winner": winner_payload,
    }
    write_json(
        OUT_MANIFEST,
        manifest,
    )

    # Attach final aggregate evidence to every actual QPU run in the group.
    for rec in run_records:
        run_uuid = rec["run_uuid"]

        qpu_db.save_dataframe_artifact(
            run_uuid,
            OUT_SUMMARY.name,
            summary_df,
        )
        qpu_db.save_json_artifact(
            run_uuid,
            OUT_WINNER.name,
            winner_payload,
        )
        qpu_db.save_json_artifact(
            run_uuid,
            OUT_MANIFEST.name,
            manifest,
        )

        qpu_db.update_run(
            run_uuid,
            run_status="COMPLETED",
            physical_layout_json=winner_layout,
            hardware_selection_json={
                "batch_group_uuid": batch_group_uuid,
                "qpu_winner_layout": winner_layout,
                "qpu_winner_layout_id": winner["layout_id"],
                "qpu_probe_scores": summary_df.to_dict(
                    orient="records"
                ),
                "calibration_set_id": expected_cal,
            },
            qpu_feature_rmse=float(
                winner["qpu_feature_rmse_vs_ideal"]
            ),
            prediction_correlation_qpu_vs_ideal=float(
                winner[
                    "prediction_correlation_qpu_vs_ideal"
                ]
            ),
            notes=(
                "COMPLETED batched IQM Candidate-1 QRC-aware training-only "
                "layout probe. Winner frozen only after all four <=100-circuit "
                "jobs completed under the same calibration set."
            ),
        )

    print()
    print("=" * 136)
    print("CANDIDATE #1 BATCHED REAL-QPU LAYOUT PROBE COMPLETE")
    print("=" * 136)
    print(summary_df.to_string(index=False))
    print()
    print(f"WINNER: {winner['layout_id']} {winner_layout}")
    print(
        "E_delta_y_QPU = "
        f"{float(winner['qpu_pred_distortion_rmse']):.6f}"
    )
    print(
        "Feature RMSE    = "
        f"{float(winner['qpu_feature_rmse_vs_ideal']):.6f}"
    )
    print(
        "Correlation     = "
        f"{float(winner['prediction_correlation_qpu_vs_ideal']):.6f}"
    )
    print()
    print("STOP HERE. Do not run full 2025 yet.")


if __name__ == "__main__":
    main()
