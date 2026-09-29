#!/usr/bin/env python
from __future__ import annotations

"""
WEEK 12.6A.2 — IQM CANDIDATE #1 REAL-EMERALD TWO-LAYOUT QPU PROBE
=================================================================

Continuation of 12.6A.1.

Consumes:
    results/12_06a1_c1_qpu_finalists.csv

Selection metric:
    E_delta_y_QPU(L)
      = sqrt(mean_i[(yhat_QPU(i,L) - yhat_ideal(i))^2])

True claim targets are NOT used to rank layouts.

Calibration guard:
- Requires current Emerald calibration_set_id == 12.6A.1 calibration_set_id.
- Backend is then created PINNED to that exact calibration set.
- While queued/running, current default calibration is polled; if it changes,
  cancellation is attempted and no winner is frozen.
- Returned per-circuit calibration_set_id must equal the expected ID.

Compilation:
- fixed physical layouts from 12.6A.1
- routing_method="none"
- zero SWAP required
- optimization_level=1
- seed_transpiler=42

Database:
- Actual QPU execution is logged to qrc_qpu_run.
- One row represents the single interleaved IQM job.
- Per-layout/per-endpoint results are stored as DB CSV/JSON artifacts.

Default = DRY RUN. Use --submit for the real QPU probe.
No full-2025 run is launched here. 2026 remains untouched.
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.providers import JobStatus

from iqm.qiskit_iqm.iqm_provider import IQMProvider

from iqm_account import get_authentication, get_client
import db_objects as qpu_db


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

A1_SCRIPT = HERE / "12_06a1_candidate1_iqm_fake_layout_screen.py"

A1_FINALISTS = RESULTS / "12_06a1_c1_qpu_finalists.csv"
A1_MANIFEST = RESULTS / "12_06a1_c1_manifest.json"
A1_IDENTITY = RESULTS / "12_06a1_c1_candidate_identity.json"
A1_ENDPOINTS = RESULTS / "12_06a1_c1_training_probe_endpoints.csv"
A1_QUALITY = RESULTS / "12_06a1_c1_live_quality.csv"
A1_DURATIONS = RESULTS / "12_06a1_c1_live_durations.csv"

PREFIX = "12_06a2_c1"

OUT_PREFLIGHT = RESULTS / f"{PREFIX}_preflight.json"
OUT_RESOURCES = RESULTS / f"{PREFIX}_probe_resources.csv"
OUT_CAL_GUARD = RESULTS / f"{PREFIX}_calibration_guard.json"
OUT_RAW = RESULTS / f"{PREFIX}_raw_counts.csv"
OUT_DAILY = RESULTS / f"{PREFIX}_probe_predictions.csv"
OUT_SUMMARY = RESULTS / f"{PREFIX}_probe_summary.csv"
OUT_WINNER = RESULTS / f"{PREFIX}_winner.json"
OUT_JOB = RESULTS / f"{PREFIX}_job.json"

BACKEND_NAME = "emerald"
EXPECTED_CANDIDATE = "H6_W06_r2_local_04_XZinj_plus_YX45"

DEFAULT_PROBE_INPUTS = 100
DEFAULT_PROBE_SHOTS = 512
DEFAULT_FINALISTS = 2

OPT_LEVEL = 1
SEED_TRANSPILE = 42
CALIBRATION_POLL_SECONDS = 5
DEFAULT_TIMEOUT_SECONDS = 4 * 60 * 60

TERMINAL = {
    JobStatus.DONE,
    JobStatus.ERROR,
    JobStatus.CANCELLED,
}


def import_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}.\n"
            "Place 12.6A.2 beside the 12.6A.1 Candidate-1 script."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = import_from_path("qrc_12_06a1_c1", A1_SCRIPT)


def utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def jsonable(x: Any):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if hasattr(x, "to_dict"):
        try:
            return jsonable(x.to_dict())
        except Exception:
            pass
    if hasattr(x, "model_dump"):
        try:
            return jsonable(x.model_dump(mode="json"))
        except Exception:
            try:
                return jsonable(x.model_dump())
            except Exception:
                pass
    if hasattr(x, "__dict__"):
        try:
            return {
                str(k): jsonable(v)
                for k, v in vars(x).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass
    return str(x)


def write_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2)


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


def layout_key(layout):
    return json.dumps([str(x) for x in layout], separators=(",", ":"))


def health_snapshot(client):
    try:
        return jsonable(client.get_health())
    except Exception as exc:
        return {"unavailable": str(exc)}


def current_default_calibration_id(client):
    dqa = client.get_dynamic_quantum_architecture()
    cal = getattr(dqa, "calibration_set_id", None)
    if cal is None:
        plain = jsonable(dqa)
        if isinstance(plain, dict):
            cal = plain.get("calibration_set_id")
    if cal is None:
        raise RuntimeError(
            "Could not retrieve current Emerald default calibration_set_id."
        )
    return str(cal)


def load_a1_state():
    required = [
        A1_FINALISTS,
        A1_MANIFEST,
        A1_IDENTITY,
        A1_ENDPOINTS,
        A1_QUALITY,
        A1_DURATIONS,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "12.6A.1 has not completed or its outputs are missing:\n  "
            + "\n  ".join(missing)
            + "\n\nFinish 12.6A.1 before any QPU probe."
        )

    with open(A1_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(A1_IDENTITY, "r", encoding="utf-8") as f:
        identity = json.load(f)

    finalists = pd.read_csv(A1_FINALISTS)
    ep_df = pd.read_csv(A1_ENDPOINTS)

    if str(manifest.get("candidate_id")) != EXPECTED_CANDIDATE:
        raise RuntimeError(
            "12.6A.1 manifest candidate mismatch: "
            f"{manifest.get('candidate_id')}"
        )

    if str(identity.get("candidate_id")) != EXPECTED_CANDIDATE:
        raise RuntimeError(
            "12.6A.1 identity candidate mismatch: "
            f"{identity.get('candidate_id')}"
        )

    if int(manifest.get("real_qpu_jobs_submitted", -1)) != 0:
        raise RuntimeError(
            "Unexpected: 12.6A.1 claims a real QPU job was already submitted."
        )

    if len(finalists) != DEFAULT_FINALISTS:
        raise RuntimeError(
            f"Expected exactly {DEFAULT_FINALISTS} IQMFake finalists, "
            f"found {len(finalists)}."
        )

    if finalists["layout"].nunique() != DEFAULT_FINALISTS:
        raise RuntimeError("12.6A.1 finalists are not two distinct layouts.")

    if "qpu_probe_status" in finalists.columns:
        bad = finalists[
            ~finalists["qpu_probe_status"].astype(str).eq("PENDING_NEXT_STEP")
        ]
        if len(bad):
            raise RuntimeError(
                "Unexpected qpu_probe_status in finalists:\n"
                + bad.to_string(index=False)
            )

    endpoints = pd.to_numeric(
        ep_df["endpoint"], errors="raise"
    ).astype(int).to_numpy()

    if len(endpoints) != DEFAULT_PROBE_INPUTS:
        raise RuntimeError(
            f"Expected {DEFAULT_PROBE_INPUTS} reserved endpoints, got {len(endpoints)}."
        )

    expected_cal = str(manifest.get("calibration_set_id", "")).strip()
    if not expected_cal:
        raise RuntimeError(
            "12.6A.1 manifest does not contain calibration_set_id."
        )

    layouts = [
        json.loads(str(x))
        for x in finalists["layout"].tolist()
    ]

    if any(len(l) != 6 for l in layouts):
        raise RuntimeError("A finalist layout does not contain six physical qubits.")

    return manifest, identity, finalists, endpoints, expected_cal, layouts


def rebuild_reference(expected_endpoints):
    candidate, source_row, identity = base.load_frozen_candidate1()

    ref = base.build_training_reference(
        candidate,
        qpu_probe_inputs=len(expected_endpoints),
        sim_inputs=min(20, len(expected_endpoints)),
    )

    got = np.asarray(ref["probe_endpoints"], dtype=int)
    expected = np.asarray(expected_endpoints, dtype=int)

    if not np.array_equal(got, expected):
        raise RuntimeError(
            "Training probe endpoints reconstructed differently from 12.6A.1."
        )

    return candidate, source_row, ref


def create_pinned_backend(expected_calibration_id):
    token, server_url, default_qc = get_authentication()

    provider = IQMProvider(
        server_url,
        quantum_computer=BACKEND_NAME,
        token=token,
    )

    return provider.get_backend(
        calibration_set_id=UUID(expected_calibration_id),
        use_metrics=True,
    )


def compare_saved_snapshot_to_current(client, layouts):
    quality = client.get_quality_metric_set()
    calibration = client.get_calibration_set()

    q_now = pd.DataFrame(
        base.e2.parse_quality(BACKEND_NAME, quality)
    )
    d_now = pd.DataFrame(
        base.e2.parse_durations(BACKEND_NAME, calibration)
    )

    q_old = pd.read_csv(A1_QUALITY)

    used_qubits = sorted(
        {str(q) for layout in layouts for q in layout},
        key=base.qb_num,
    )

    used_pairs = set()
    for layout in layouts:
        mapping = {i: str(layout[i]) for i in range(6)}
        for a, b in base.H6_EDGES:
            used_pairs.add(
                base.pair_text(
                    base.normalize_pair(mapping[a], mapping[b])
                )
            )

    rows = []

    for metric in [
        "t1",
        "t2_echo",
        "prx_fidelity",
        "readout_error_0_to_1",
        "readout_error_1_to_0",
    ]:
        for q in used_qubits:
            old = q_old[
                q_old["metric_type"].astype(str).eq(metric)
                & q_old["locus"].astype(str).eq(q)
            ]
            new = q_now[
                q_now["metric_type"].astype(str).eq(metric)
                & q_now["locus"].astype(str).eq(q)
            ]

            old_vals = pd.to_numeric(old.get("value"), errors="coerce").dropna()
            new_vals = pd.to_numeric(new.get("value"), errors="coerce").dropna()

            old_v = float(old_vals.iloc[-1]) if len(old_vals) else np.nan
            new_v = float(new_vals.iloc[-1]) if len(new_vals) else np.nan

            rows.append({
                "kind": "quality",
                "metric": metric,
                "locus": q,
                "old": old_v,
                "new": new_v,
                "abs_change": (
                    abs(new_v - old_v)
                    if np.isfinite(old_v) and np.isfinite(new_v)
                    else np.nan
                ),
            })

    for pair in sorted(used_pairs):
        old = q_old[
            q_old["metric_type"].astype(str).eq("cz_fidelity")
            & q_old["locus"].astype(str).eq(pair)
        ]
        new = q_now[
            q_now["metric_type"].astype(str).eq("cz_fidelity")
            & q_now["locus"].astype(str).eq(pair)
        ]

        old_vals = pd.to_numeric(old.get("value"), errors="coerce").dropna()
        new_vals = pd.to_numeric(new.get("value"), errors="coerce").dropna()

        old_v = float(old_vals.iloc[-1]) if len(old_vals) else np.nan
        new_v = float(new_vals.iloc[-1]) if len(new_vals) else np.nan

        rows.append({
            "kind": "quality",
            "metric": "cz_fidelity",
            "locus": pair,
            "old": old_v,
            "new": new_v,
            "abs_change": (
                abs(new_v - old_v)
                if np.isfinite(old_v) and np.isfinite(new_v)
                else np.nan
            ),
        })

    drift = pd.DataFrame(rows)

    return {
        "used_qubits": used_qubits,
        "used_cz_pairs": sorted(used_pairs),
        "quality_drift_rows": drift.to_dict(orient="records"),
        "max_abs_quality_change": (
            float(
                pd.to_numeric(
                    drift["abs_change"],
                    errors="coerce",
                ).max()
            )
            if len(drift)
            else None
        ),
        "current_quality_rows": int(len(q_now)),
        "current_duration_rows": int(len(d_now)),
    }


def cyclic_layout_order(layouts, endpoint_index):
    n = len(layouts)
    shift = int(endpoint_index) % n
    order = list(layouts[shift:]) + list(layouts[:shift])

    block = int(endpoint_index) // n
    if block % 2 == 1:
        order = list(reversed(order))

    return order


def strict_compile(logical, backend, layout):
    target = backend.get_real_target()
    layout_idx = base.physical_layout_indices(backend, layout)

    tqc = transpile(
        logical,
        target=target,
        initial_layout=layout_idx,
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
    )

    swaps = int(tqc.count_ops().get("swap", 0))
    if swaps != 0:
        raise RuntimeError(
            f"SWAP={swaps} for fixed finalist layout {layout}."
        )

    return tqc


def compile_interleaved_probe(
    backend,
    candidate,
    ref,
    endpoints,
    layouts,
):
    settings = base.readout_settings(candidate["readout"])

    circuits = []
    meta = []
    resources = []

    layout_ids = {
        layout_key(layout): f"L{i+1}"
        for i, layout in enumerate(layouts)
    }

    for endpoint_index, endpoint in enumerate(endpoints):
        order = cyclic_layout_order(layouts, endpoint_index)

        for position, layout in enumerate(order, start=1):
            lid = layout_ids[layout_key(layout)]

            for setting in settings:
                logical = base.build_measurement_circuit(
                    candidate,
                    ref["angles"],
                    int(endpoint),
                    setting,
                )

                tqc = strict_compile(
                    logical,
                    backend,
                    layout,
                )

                item = {
                    "endpoint": int(endpoint),
                    "layout_id": lid,
                    "layout": json.dumps(layout),
                    "layout_position_within_endpoint": int(position),
                    "setting": str(setting[0]),
                }

                tqc.metadata = dict(item)

                circuits.append(tqc)
                meta.append(item)

                ops = {
                    str(k): int(v)
                    for k, v in tqc.count_ops().items()
                }

                try:
                    duration_us = (
                        float(
                            tqc.estimate_duration(
                                backend.target,
                                unit="s",
                            )
                        )
                        * 1e6
                    )
                except Exception:
                    duration_us = np.nan

                resources.append({
                    **item,
                    "depth": int(tqc.depth()),
                    "size": int(tqc.size()),
                    "n_cz": int(ops.get("cz", 0)),
                    "n_reset": int(ops.get("reset", 0)),
                    "n_swap": int(ops.get("swap", 0)),
                    "duration_us": duration_us,
                    "operations_json": json.dumps(ops, sort_keys=True),
                })

    resources_df = pd.DataFrame(resources)

    expected_n = (
        len(endpoints)
        * len(layouts)
        * len(settings)
    )

    if len(circuits) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} probe circuits, got {len(circuits)}."
        )

    if int(resources_df["n_swap"].max()) != 0:
        raise RuntimeError("At least one probe circuit contains a SWAP.")

    pos = (
        resources_df[
            ["endpoint", "layout_id", "layout_position_within_endpoint"]
        ]
        .drop_duplicates()
        .groupby(
            ["layout_id", "layout_position_within_endpoint"]
        )
        .size()
        .unstack(fill_value=0)
    )

    if pos.to_numpy().max() - pos.to_numpy().min() > 1:
        raise RuntimeError(
            "Interleaving position counts are unexpectedly unbalanced."
        )

    return circuits, meta, resources_df, pos


def parse_probe_result(
    result,
    meta,
    layouts,
    endpoints,
    ref,
):
    by_layout = {
        layout_key(layout): {
            int(ep): {}
            for ep in endpoints
        }
        for layout in layouts
    }

    raw_rows = []
    returned_cal_ids = []

    for i, m in enumerate(meta):
        counts = result.get_counts(i)

        cdict = {
            str(k): int(v)
            for k, v in counts.items()
        }

        key = layout_key(
            json.loads(str(m["layout"]))
        )

        by_layout[key][int(m["endpoint"])][str(m["setting"])] = cdict

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
            returned_cal_ids.append(str(cal_id))

        raw_rows.append({
            **m,
            "calibration_set_id_returned": (
                str(cal_id)
                if cal_id is not None
                else None
            ),
            "counts_json": json.dumps(cdict, sort_keys=True),
            "shots_returned": int(sum(cdict.values())),
        })

    X_ideal = np.asarray(ref["X_probe_ideal"], dtype=float)
    pred_ideal = np.asarray(ref["pred_probe_ideal"], dtype=float)

    daily_rows = []
    summary_rows = []

    for layout_i, layout in enumerate(layouts, start=1):
        key = layout_key(layout)
        X_qpu = []

        for ep in endpoints:
            setting_counts = by_layout[key][int(ep)]

            if set(setting_counts) != {"XXXXYX", "ZZZZZZ"}:
                raise RuntimeError(
                    f"Missing settings for layout={layout}, endpoint={ep}: "
                    f"{list(setting_counts)}"
                )

            X_qpu.append(
                base.candidate1_features(setting_counts)
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
                **{
                    f"qpu_{name}": float(X_qpu[j, k])
                    for k, name in enumerate(ref["features"])
                },
                **{
                    f"ideal_{name}": float(X_ideal[j, k])
                    for k, name in enumerate(ref["features"])
                },
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

    summary["qpu_probe_rank"] = np.arange(1, len(summary) + 1)

    raw = pd.DataFrame(raw_rows)
    daily = pd.DataFrame(daily_rows)

    return raw, daily, summary, sorted(set(returned_cal_ids))


def create_qpu_run_record(
    candidate,
    ref,
    layouts,
    expected_cal,
    resources_df,
    shots,
):
    hardware = {
        "selection_method": "qrc_aware_two_layout_training_probe",
        "layouts": layouts,
        "calibration_set_id": expected_cal,
        "selection_metric": "RMS QPU-vs-ideal prediction distortion",
        "true_targets_used": False,
        "layout_order": "cyclic_interleaved",
    }

    manifest = {
        "stage": "12.6A.2",
        "candidate_id": EXPECTED_CANDIDATE,
        "backend": BACKEND_NAME,
        "expected_calibration_set_id": expected_cal,
        "probe_scope": "2022-2024_training_only",
        "probe_endpoints": int(len(ref["probe_endpoints"])),
        "layouts": layouts,
        "shots_per_setting": int(shots),
        "n_circuits": int(len(resources_df)),
        "optimization_level": OPT_LEVEL,
        "seed_transpiler": SEED_TRANSPILE,
        "routing_method": "none",
        "initial_layout": "fixed_per_finalist",
        "true_targets_used_for_ranking": False,
        "full_2025": False,
        "test_2026_used": False,
    }

    j_json = {
        f"J{i}{j}": float(v)
        for (i, j), v in candidate["J"].items()
    }

    run_uuid = qpu_db.create_run(
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
        # Historical schema field reused for training-probe endpoint count.
        n_validation_endpoints=int(len(ref["probe_endpoints"])),
        n_circuits=int(len(resources_df)),
        full_2025=False,
        pilot_selection="qrc_aware_training_probe_2layouts",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
        median_cz_per_feature=float(
            resources_df.groupby(
                ["endpoint", "layout_id"]
            )["n_cz"].sum().median()
        ),
        median_max_setting_depth=float(
            resources_df.groupby(
                ["endpoint", "layout_id"]
            )["depth"].max().median()
        ),
        notes=(
            "IQM Candidate-1 training-only two-layout QPU probe. "
            "n_validation_endpoints stores 100 training-probe endpoints. "
            "No 2025 targets and no true claim targets are used for layout ranking."
        ),
    )

    return run_uuid, manifest, hardware


def guarded_wait_for_result(
    job,
    guard_client,
    expected_calibration_id,
    timeout_seconds,
    run_uuid,
):
    start = time.monotonic()
    last_status = None

    while True:
        status = job.status()

        if status != last_status:
            print(f"  IQM job status: {status}")
            last_status = status
            qpu_db.update_run(
                run_uuid,
                job_status=str(status),
            )

        if status in TERMINAL:
            break

        current_cal = current_default_calibration_id(
            guard_client
        )

        if current_cal != expected_calibration_id:
            print()
            print("CALIBRATION CHANGED WHILE QPU PROBE WAS PENDING/RUNNING.")
            print(f"  expected: {expected_calibration_id}")
            print(f"  current:  {current_cal}")
            print("Attempting to cancel the IQM job...")

            cancelled = False
            try:
                cancelled = bool(job.cancel())
            except Exception:
                pass

            qpu_db.update_run(
                run_uuid,
                run_status="ABORTED_CALIBRATION_CHANGED",
                completed_at_utc=utc_now_naive(),
                job_status=str(job.status()),
                error_message=(
                    "Emerald default calibration_set_id changed during "
                    f"probe: expected={expected_calibration_id}, current={current_cal}; "
                    f"cancel_attempt={cancelled}"
                ),
            )

            raise RuntimeError(
                "Emerald calibration changed during the QPU probe. "
                "Winner NOT frozen. Rerun 12.6A.1."
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
                error_message=(
                    f"Timed out after {timeout_seconds} seconds."
                ),
            )
            raise TimeoutError(
                f"IQM QPU probe timed out after {timeout_seconds} seconds."
            )

        time.sleep(CALIBRATION_POLL_SECONDS)

    if status != JobStatus.DONE:
        msg = f"IQM job ended with non-DONE status: {status}"
        qpu_db.update_run(
            run_uuid,
            run_status="FAILED_QPU_JOB",
            completed_at_utc=utc_now_naive(),
            job_status=str(status),
            error_message=msg,
        )
        raise RuntimeError(msg)

    return job.result()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the real Emerald two-layout training-only QPU probe.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_PROBE_SHOTS,
        help=(
            f"Shots per grouped setting. Default {DEFAULT_PROBE_SHOTS}, "
            "matching the IBM method."
        ),
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
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")

    local_outputs = [
        OUT_PREFLIGHT,
        OUT_RESOURCES,
        OUT_CAL_GUARD,
        OUT_RAW,
        OUT_DAILY,
        OUT_SUMMARY,
        OUT_WINNER,
        OUT_JOB,
    ]

    if args.overwrite_local:
        for p in local_outputs:
            if p.exists():
                p.unlink()

    print("=" * 136)
    print("WEEK 12.6A.2 — IQM CANDIDATE #1 REAL-EMERALD TWO-LAYOUT QPU PROBE")
    print("=" * 136)
    print("Selection metric = QPU prediction minus IDEAL prediction.")
    print("True targets are NOT used.")
    print(
        f"Fixed-layout compilation: optimization_level={OPT_LEVEL}, "
        f"routing_method='none', seed={SEED_TRANSPILE}."
    )
    print("2022-2024 training-only. No full 2025 here. 2026 untouched.")
    print()

    (
        a1_manifest,
        a1_identity,
        finalist_df,
        endpoints,
        expected_cal,
        layouts,
    ) = load_a1_state()

    candidate, source_row, ref = rebuild_reference(
        endpoints
    )

    print("12.6A.1 FINALISTS")
    print("-" * 136)
    display_cols = [
        c for c in [
            "fake_sim_rank",
            "layout",
            "sim_pred_distortion_rmse",
            "sim_feature_rmse",
            "actual_workload_proxy_nll",
            "min_cz_fidelity",
            "min_t2_echo_us",
            "memory_min_t2_echo_us",
        ]
        if c in finalist_df.columns
    ]
    print(finalist_df[display_cols].to_string(index=False))
    print()
    print(f"Expected calibration_set_id: {expected_cal}")
    print(f"Training probe endpoints:     {len(endpoints)}")
    print(f"Shots/setting:                {args.shots}")

    guard_client = get_client(
        quantum_computer=BACKEND_NAME
    )

    health_before = health_snapshot(
        guard_client
    )

    current_cal = current_default_calibration_id(
        guard_client
    )

    drift_audit = compare_saved_snapshot_to_current(
        guard_client,
        layouts,
    )

    guard_payload = {
        "backend": BACKEND_NAME,
        "expected_calibration_set_id_from_12_06a1": expected_cal,
        "current_default_calibration_set_id": current_cal,
        "calibration_id_match": current_cal == expected_cal,
        "health_before": health_before,
        "finalist_metric_drift_audit": drift_audit,
        "qpu_submitted": False,
    }

    write_json(
        OUT_CAL_GUARD,
        guard_payload,
    )

    print()
    print("CALIBRATION GUARD")
    print("-" * 136)
    print(f"12.6A.1 calibration: {expected_cal}")
    print(f"Current default:      {current_cal}")
    print(
        f"ID match:             "
        f"{'PASS' if current_cal == expected_cal else 'FAIL'}"
    )
    print(
        "Max finalist raw-quality change: "
        f"{drift_audit['max_abs_quality_change']}"
    )

    if current_cal != expected_cal:
        print()
        print("ZERO QPU JOBS SUBMITTED.")
        print(
            "Emerald calibration changed since 12.6A.1. "
            "Rerun 12.6A.1 for the new calibration."
        )
        return

    backend = create_pinned_backend(
        expected_cal
    )

    print()
    print("STRICT INTERLEAVED PROBE COMPILE")
    print("-" * 136)

    circuits, meta, resources_df, position_audit = (
        compile_interleaved_probe(
            backend=backend,
            candidate=candidate,
            ref=ref,
            endpoints=endpoints,
            layouts=layouts,
        )
    )

    resources_df.to_csv(
        OUT_RESOURCES,
        index=False,
    )

    print(f"Circuits: {len(circuits)}")
    print(
        f"Expected = {len(endpoints)} endpoints "
        f"x {len(layouts)} layouts x 2 settings = "
        f"{len(endpoints) * len(layouts) * 2}"
    )
    print(f"Maximum SWAP count: {int(resources_df['n_swap'].max())}")
    print()
    print("Layout-position balance:")
    print(position_audit.to_string())
    print()

    resource_summary = (
        resources_df
        .groupby("layout_id", as_index=False)
        .agg(
            n_circuits=("endpoint", "size"),
            median_depth=("depth", "median"),
            max_depth=("depth", "max"),
            median_cz=("n_cz", "median"),
            median_duration_us=("duration_us", "median"),
            max_duration_us=("duration_us", "max"),
        )
    )

    print(resource_summary.to_string(index=False))

    run_request = backend.create_run_request(
        circuits,
        shots=int(args.shots),
    )

    request_cal = str(
        getattr(
            run_request,
            "calibration_set_id",
            expected_cal,
        )
    )

    if request_cal != expected_cal:
        raise RuntimeError(
            "Pinned IQM run request calibration mismatch.\n"
            f"Expected {expected_cal}; request contains {request_cal}."
        )

    preflight = {
        "candidate": EXPECTED_CANDIDATE,
        "backend": BACKEND_NAME,
        "expected_calibration_set_id": expected_cal,
        "current_default_calibration_set_id": current_cal,
        "run_request_calibration_set_id": request_cal,
        "calibration_guard_pass": True,
        "layouts": layouts,
        "n_training_probe_endpoints": int(len(endpoints)),
        "shots_per_setting": int(args.shots),
        "n_circuits": int(len(circuits)),
        "optimization_level": OPT_LEVEL,
        "seed_transpiler": SEED_TRANSPILE,
        "routing_method": "none",
        "max_swap_count": int(resources_df["n_swap"].max()),
        "layout_position_balance": position_audit.to_dict(),
        "resource_summary": resource_summary.to_dict(
            orient="records"
        ),
        "true_targets_used_for_layout_ranking": False,
        "full_2025": False,
        "test_2026_used": False,
    }

    write_json(
        OUT_PREFLIGHT,
        preflight,
    )

    if not args.submit:
        print()
        print("=" * 136)
        print("12.6A.2 DRY-RUN PREFLIGHT COMPLETE")
        print("=" * 136)
        print("Calibration guard: PASS")
        print("Pinned run-request calibration: PASS")
        print("Strict zero-SWAP compile: PASS")
        print("DB row: NOT CREATED")
        print("QPU jobs submitted: 0")
        print()
        print(
            "To execute the real training-only QPU probe, run:\n"
            f"  python {Path(__file__).name} --submit"
        )
        return

    current_cal_2 = current_default_calibration_id(
        guard_client
    )

    if current_cal_2 != expected_cal:
        print()
        print("CALIBRATION CHANGED AFTER COMPILE.")
        print("ZERO QPU JOBS SUBMITTED.")
        print(f"Expected: {expected_cal}")
        print(f"Current:  {current_cal_2}")
        return

    run_uuid = None
    job = None

    try:
        run_uuid, db_manifest, db_hardware = (
            create_qpu_run_record(
                candidate=candidate,
                ref=ref,
                layouts=layouts,
                expected_cal=expected_cal,
                resources_df=resources_df,
                shots=args.shots,
            )
        )

        qpu_db.save_dataframe_artifact(
            run_uuid,
            OUT_RESOURCES.name,
            resources_df,
        )
        qpu_db.save_json_artifact(
            run_uuid,
            OUT_PREFLIGHT.name,
            preflight,
        )
        qpu_db.save_json_artifact(
            run_uuid,
            OUT_CAL_GUARD.name,
            guard_payload,
        )

        print()
        print("DATABASE")
        print("-" * 136)
        print(f"qrc_qpu_run run_uuid: {run_uuid}")

        print()
        print("=" * 136)
        print("SUBMITTING REAL EMERALD QPU PROBE")
        print("=" * 136)
        print(
            f"{len(circuits)} circuits | "
            f"{args.shots} shots/setting | "
            f"calibration pinned to {expected_cal}"
        )

        submitted_at = utc_now_naive()

        job = backend.run(
            circuits,
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

        print(f"IQM job ID: {job_id}")

        guard_payload["qpu_submitted"] = True
        guard_payload["job_id"] = job_id
        guard_payload["submitted_at_utc"] = submitted_at
        write_json(
            OUT_CAL_GUARD,
            guard_payload,
        )

        result = guarded_wait_for_result(
            job=job,
            guard_client=guard_client,
            expected_calibration_id=expected_cal,
            timeout_seconds=int(args.timeout_seconds),
            run_uuid=run_uuid,
        )

        raw_df, daily_df, summary_df, returned_cal_ids = (
            parse_probe_result(
                result=result,
                meta=meta,
                layouts=layouts,
                endpoints=endpoints,
                ref=ref,
            )
        )

        raw_df.to_csv(
            OUT_RAW,
            index=False,
        )
        daily_df.to_csv(
            OUT_DAILY,
            index=False,
        )
        summary_df.to_csv(
            OUT_SUMMARY,
            index=False,
        )

        print()
        print("RETURNED IQM CALIBRATION IDs")
        print("-" * 136)
        print(returned_cal_ids)

        if returned_cal_ids != [expected_cal]:
            qpu_db.save_dataframe_artifact(
                run_uuid,
                OUT_RAW.name,
                raw_df,
            )
            qpu_db.update_run(
                run_uuid,
                run_status="INVALID_CALIBRATION_MISMATCH",
                completed_at_utc=utc_now_naive(),
                job_status=str(job.status()),
                error_message=(
                    "Returned calibration_set_id mismatch. "
                    f"expected={expected_cal}; returned={returned_cal_ids}"
                ),
            )

            raise RuntimeError(
                "QPU result calibration mismatch. Winner NOT frozen."
            )

        winner = summary_df.iloc[0].to_dict()
        winner_layout = json.loads(
            str(winner["layout"])
        )

        winner_payload = {
            "candidate_id": EXPECTED_CANDIDATE,
            "backend": BACKEND_NAME,
            "calibration_set_id": expected_cal,
            "run_uuid": run_uuid,
            "job_id": str(job.job_id()),
            "winner_layout_id": winner["layout_id"],
            "winner_layout": winner_layout,
            "selection_metric": (
                "RMS QPU-vs-ideal prediction distortion "
                "on 100 deterministic 2022-2024 training endpoints"
            ),
            "qpu_pred_distortion_rmse": winner[
                "qpu_pred_distortion_rmse"
            ],
            "qpu_feature_rmse_vs_ideal": winner[
                "qpu_feature_rmse_vs_ideal"
            ],
            "prediction_correlation_qpu_vs_ideal": winner[
                "prediction_correlation_qpu_vs_ideal"
            ],
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

        try:
            result_dict = result.to_dict()
        except Exception:
            result_dict = {}

        job_payload = {
            "run_uuid": run_uuid,
            "job_id": str(job.job_id()),
            "job_status": str(job.status()),
            "backend": BACKEND_NAME,
            "expected_calibration_set_id": expected_cal,
            "returned_calibration_set_ids": returned_cal_ids,
            "timeline": (
                result_dict.get("_metadata", {}).get("timeline")
                if isinstance(result_dict, dict)
                else None
            ),
            "result_metadata": (
                result_dict.get("_metadata")
                if isinstance(result_dict, dict)
                else None
            ),
            "health_after": health_snapshot(
                guard_client
            ),
        }

        write_json(
            OUT_JOB,
            job_payload,
        )

        qpu_db.save_dataframe_artifact(
            run_uuid,
            OUT_RAW.name,
            raw_df,
        )
        qpu_db.save_dataframe_artifact(
            run_uuid,
            OUT_DAILY.name,
            daily_df,
        )
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
            OUT_JOB.name,
            job_payload,
        )

        db_hardware_final = {
            **db_hardware,
            "qpu_winner_layout": winner_layout,
            "qpu_winner_layout_id": winner["layout_id"],
            "qpu_probe_scores": summary_df.to_dict(
                orient="records"
            ),
            "returned_calibration_set_ids": returned_cal_ids,
        }

        qpu_db.update_run(
            run_uuid,
            run_status="COMPLETED",
            completed_at_utc=utc_now_naive(),
            job_status=str(job.status()),
            physical_layout_json=winner_layout,
            hardware_selection_json=db_hardware_final,
            qpu_feature_rmse=float(
                winner["qpu_feature_rmse_vs_ideal"]
            ),
            prediction_correlation_qpu_vs_ideal=float(
                winner[
                    "prediction_correlation_qpu_vs_ideal"
                ]
            ),
            job_metrics_json=job_payload,
            notes=(
                "COMPLETED IQM Candidate-1 QRC-aware training-only "
                "two-layout probe. Winner selected by QPU-vs-ideal prediction "
                "distortion without true claim targets. Returned calibration "
                "ID matched the pinned 12.6A.1 calibration. Full 2025 not run."
            ),
        )

        print()
        print("=" * 136)
        print("CANDIDATE #1 REAL-QPU LAYOUT PROBE COMPLETE")
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
        print(f"qrc_qpu_run run_uuid = {run_uuid}")
        print(f"Winner saved: {OUT_WINNER}")
        print()
        print(
            "STOP HERE. Do NOT run full 2025 yet. "
            "Send the console output and summary table first."
        )

    except Exception as exc:
        if run_uuid is not None:
            try:
                qpu_db.update_run(
                    run_uuid,
                    run_status="FAILED",
                    completed_at_utc=utc_now_naive(),
                    job_status=(
                        str(job.status())
                        if job is not None
                        else None
                    ),
                    error_message=str(exc),
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
