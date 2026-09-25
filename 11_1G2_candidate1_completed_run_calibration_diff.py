#!/usr/bin/env python3
"""
11.1G.2 — Candidate #1 completed-run calibration reconstruction and diff (FIXED)

Purpose
-------
Read completed Candidate #1 QPU runs from the project table `qrc_qpu_run`,
resolve the IBM Kingston calibration snapshot that was active when each job
actually started RUNNING, and compare the IBM-reported calibration properties
between successive completed runs.

This is a READ-ONLY analysis:
  * reads qrc_qpu_run through db_config.engine
  * reads historical IBM backend.properties(datetime=...)
  * DOES NOT submit a QPU job
  * DOES NOT load/use the frozen 2026 forecasting test set

Candidate #1 is identified by:
    candidate_key = 'RWP_H3_R1R3'

Only actual completed IBM QPU executions are included by default:
    run_status = 'COMPLETED'
    backend_name = 'ibm_kingston'
    job_id IS NOT NULL
    ibm_job_running_at_utc IS NOT NULL
    qpu_rmse IS NOT NULL

Important scientific interpretation
-----------------------------------
The script reconstructs the provider-reported calibration snapshot active at
hardware execution time. It can therefore distinguish:

  1) same calibration snapshot, different run/layout/primitive
  2) different calibration snapshot, same physical layout
  3) different calibration snapshot, different physical layout

The calibration diff contains IBM-reported qubit/gate benchmark properties
(T1, T2, readout error, gate error, gate length, property timestamps, etc.).
It does NOT expose every internal pulse/control action IBM may have performed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text

from db_config import engine
from ibm_account import get_service


# ============================================================================
# Frozen project defaults
# ============================================================================
BACKEND_NAME = "ibm_kingston"
CANDIDATE_KEY = "RWP_H3_R1R3"
LOCAL_TZ_NAME = "Europe/Bratislava"
TABLE_NAME = "qrc_qpu_run"

# Candidate-1 non-zero logical ZZ couplings: J01, J04, J12, J23, J45.
C1_LOGICAL_EDGES = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5)]

OUTDIR = Path("results")
PREFIX = "11_1G2_candidate1_completed_run_calibration_diff_fixed"


# ============================================================================
# Helpers
# ============================================================================
def parse_layout(value: Any) -> Optional[List[int]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    s = str(value).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [int(x) for x in obj]
    except Exception:
        pass
    nums = re.findall(r"-?\d+", s)
    return [int(x) for x in nums] if nums else None


def ensure_utc(value: Any) -> datetime:
    """Interpret DB timestamps as UTC when timezone-naive."""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        value = pd.to_datetime(value).to_pydatetime()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def iso_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def numeric_or_nan(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def changed_numeric(a: float, b: float, rtol: float = 1e-12, atol: float = 1e-15) -> bool:
    if np.isnan(a) and np.isnan(b):
        return False
    if np.isnan(a) != np.isnan(b):
        return True
    return not bool(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def rel_change_pct(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b) or a == 0:
        return float("nan")
    return 100.0 * (b - a) / abs(a)


def normalize_pair(qubits: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(q) for q in qubits))


def safe_token(value: Any, max_len: int = 60) -> str:
    s = str(value)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s[:max_len] or "x"


def classify_run(script_name: str) -> str:
    s = (script_name or "").lower()
    if "qiskit_layout" in s:
        return "sampler_qiskit_layout"
    if "estimator" in s:
        return "estimator"
    if "m3" in s:
        return "sampler_m3_ablation"
    if "best_candidate_direct_qpu" in s or "11_1a" in s:
        return "sampler_rule5"
    return "other_completed_c1"


# ============================================================================
# DB loading
# ============================================================================
def load_completed_candidate1_runs(table_name: str, backend_name: str, candidate_key: str) -> pd.DataFrame:
    # Table name is a trusted CLI/default identifier, not user data. Keep it simple
    # but reject unexpected characters before interpolation.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(f"Unsafe table name: {table_name!r}")

    sql = text(
        f"""
        SELECT
            id,
            run_uuid,
            script_name,
            run_status,
            candidate_key,
            selected_by_rules,
            protocol,
            topology,
            window_size,
            trotter_r,
            readout_name,
            backend_name,
            physical_layout_json,
            hardware_selection_json,
            max_2q_error_percent,
            max_readout_error_percent,
            max_1q_error_percent,
            min_t1_us,
            min_t2_us,
            compiled_duration_us,
            r_t2,
            shots_per_setting,
            n_validation_endpoints,
            n_circuits,
            job_id,
            job_status,
            qpu_charge_time_seconds,
            qpu_circuits_execution_time_seconds,
            ibm_job_created_at_utc,
            ibm_job_running_at_utc,
            ibm_job_finished_at_utc,
            qpu_rmse,
            qpu_mae,
            qpu_bias,
            qpu_feature_mae,
            qpu_feature_rmse,
            z3_cross_setting_mae,
            prediction_correlation_qpu_vs_ideal,
            notes
        FROM {table_name}
        WHERE candidate_key = :candidate_key
          AND run_status = 'COMPLETED'
          AND backend_name = :backend_name
          AND job_id IS NOT NULL
          AND ibm_job_running_at_utc IS NOT NULL
          AND qpu_rmse IS NOT NULL
        ORDER BY ibm_job_running_at_utc ASC, id ASC
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"candidate_key": candidate_key, "backend_name": backend_name})

    if df.empty:
        raise RuntimeError(
            f"No completed QPU runs found in {table_name} for candidate_key={candidate_key!r}, backend={backend_name!r}."
        )

    df["layout"] = df["physical_layout_json"].apply(parse_layout)
    df["run_kind"] = df["script_name"].fillna("").map(classify_run)
    df["run_start_utc"] = df["ibm_job_running_at_utc"].map(ensure_utc)
    return df


# ============================================================================
# Historical IBM calibration retrieval
# ============================================================================
@dataclass
class CalibrationSnapshot:
    query_time_utc: datetime
    actual_last_update_utc: datetime
    properties: Any


def fetch_snapshot_for_run(backend: Any, run_start_utc: datetime) -> CalibrationSnapshot:
    """Resolve the IBM-reported properties snapshot active at QPU RUNNING time.

    FIXED methodology:
      * normalize the DB RUNNING timestamp to UTC first;
      * query IBM with that exact aware UTC datetime (no +1 s offset);
      * normalize IBM's returned last_update_date to UTC before comparisons;
      * reject any impossible snapshot whose last_update is later than the run.

    This matches the direct diagnostic:
        backend.properties(datetime=ibm_job_running_at_utc)
    """
    query_time = ensure_utc(run_start_utc)
    props = backend.properties(datetime=query_time)
    if props is None:
        raise RuntimeError(
            f"No historical backend properties returned for exact run time {query_time.isoformat()}"
        )

    actual_raw = props.last_update_date
    actual = ensure_utc(actual_raw)

    # Historical properties should be the snapshot active at or before RUNNING.
    # Allow a tiny tolerance only for serialization/rounding edge cases.
    if actual > query_time + timedelta(seconds=1):
        raise RuntimeError(
            "IBM returned a calibration last_update_date later than the QPU RUNNING time: "
            f"run={query_time.isoformat()}, calibration={actual.isoformat()}, raw={actual_raw!r}"
        )

    return CalibrationSnapshot(
        query_time_utc=query_time,
        actual_last_update_utc=actual,
        properties=props,
    )


# ============================================================================
# Raw property extraction
# ============================================================================
def extract_qubit_long(props: Any, snapshot_label: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for q, nduvs in enumerate(props.qubits):
        for item in nduvs:
            rows.append(
                {
                    "snapshot": snapshot_label,
                    "qubit": int(q),
                    "property": str(getattr(item, "name", "")),
                    "value": numeric_or_nan(getattr(item, "value", None)),
                    "unit": getattr(item, "unit", None),
                    "property_date": iso_or_none(getattr(item, "date", None)),
                }
            )
    return pd.DataFrame(rows)


def extract_gate_long(props: Any, snapshot_label: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for gate in props.gates:
        gate_name = str(getattr(gate, "gate", ""))
        qubits = tuple(int(q) for q in getattr(gate, "qubits", []))
        qubits_text = ",".join(str(q) for q in qubits)
        undirected_text = ",".join(str(q) for q in normalize_pair(qubits))

        for item in getattr(gate, "parameters", []):
            rows.append(
                {
                    "snapshot": snapshot_label,
                    "gate": gate_name,
                    "qubits": qubits_text,
                    "undirected_qubits": undirected_text,
                    "property": str(getattr(item, "name", "")),
                    "value": numeric_or_nan(getattr(item, "value", None)),
                    "unit": getattr(item, "unit", None),
                    "property_date": iso_or_none(getattr(item, "date", None)),
                }
            )
    return pd.DataFrame(rows)


def diff_long(a: pd.DataFrame, b: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    a2 = a.drop(columns=["snapshot"]).rename(
        columns={"value": "value_A", "unit": "unit_A", "property_date": "property_date_A"}
    )
    b2 = b.drop(columns=["snapshot"]).rename(
        columns={"value": "value_B", "unit": "unit_B", "property_date": "property_date_B"}
    )
    out = a2.merge(b2, on=list(key_cols), how="outer", indicator=True)

    statuses: List[str] = []
    value_flags: List[bool] = []
    ts_flags: List[bool] = []
    abs_changes: List[float] = []
    rel_changes: List[float] = []

    for _, row in out.iterrows():
        if row["_merge"] != "both":
            status = "removed_in_B" if row["_merge"] == "left_only" else "added_in_B"
            statuses.append(status)
            value_flags.append(True)
            ts_flags.append(True)
            abs_changes.append(float("nan"))
            rel_changes.append(float("nan"))
            continue

        va = numeric_or_nan(row.get("value_A"))
        vb = numeric_or_nan(row.get("value_B"))
        vc = changed_numeric(va, vb)
        tc = str(row.get("property_date_A")) != str(row.get("property_date_B"))
        uc = str(row.get("unit_A")) != str(row.get("unit_B"))

        if vc and tc:
            status = "value_and_timestamp_changed"
        elif vc:
            status = "value_changed_only"
        elif tc:
            status = "timestamp_changed_only"
        elif uc:
            status = "unit_changed_only"
        else:
            status = "unchanged"

        statuses.append(status)
        value_flags.append(vc)
        ts_flags.append(tc)
        abs_changes.append(vb - va if not (np.isnan(va) or np.isnan(vb)) else float("nan"))
        rel_changes.append(rel_change_pct(va, vb))

    out = out.rename(columns={"_merge": "presence"})
    out["status"] = statuses
    out["value_changed"] = value_flags
    out["timestamp_changed"] = ts_flags
    out["abs_change"] = abs_changes
    out["relative_change_percent"] = rel_changes
    return out


# ============================================================================
# Layout-specific extraction
# ============================================================================
def layout_role_map(layout: Sequence[int]) -> Dict[int, str]:
    return {int(p): f"q{i}" for i, p in enumerate(layout)}


def candidate_physical_edges(layout: Sequence[int]) -> Dict[Tuple[int, int], str]:
    mapping: Dict[Tuple[int, int], str] = {}
    for a, b in C1_LOGICAL_EDGES:
        pa, pb = int(layout[a]), int(layout[b])
        mapping[tuple(sorted((pa, pb)))] = f"q{a}-q{b}"
    return mapping


def filter_layout_qubits(diff_q: pd.DataFrame, layout: Optional[Sequence[int]], label: str) -> pd.DataFrame:
    if not layout or len(layout) < 6:
        return pd.DataFrame()
    role_map = layout_role_map(layout)
    out = diff_q[diff_q["qubit"].isin(layout)].copy()
    out.insert(0, "layout_label", label)
    out.insert(1, "logical_role", out["qubit"].map(role_map))
    return out.sort_values(["logical_role", "property"], kind="stable")


def filter_layout_cz(diff_g: pd.DataFrame, layout: Optional[Sequence[int]], label: str) -> pd.DataFrame:
    if not layout or len(layout) < 6:
        return pd.DataFrame()
    edge_map = candidate_physical_edges(layout)
    wanted = {f"{a},{b}" for a, b in edge_map}
    out = diff_g[(diff_g["gate"].str.lower() == "cz") & (diff_g["undirected_qubits"].isin(wanted))].copy()
    out.insert(0, "layout_label", label)

    def logical_edge(row: pd.Series) -> str:
        pair = tuple(int(x) for x in str(row["undirected_qubits"]).split(","))
        return edge_map.get(pair, "")

    out.insert(1, "logical_edge", out.apply(logical_edge, axis=1))
    return out.sort_values(["logical_edge", "qubits", "property"], kind="stable")


def summarize_changed(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {"rows": 0, "changed_any": 0, "value_changed": 0, "timestamp_changed": 0}
    return {
        "rows": int(len(df)),
        "changed_any": int((df["status"] != "unchanged").sum()),
        "value_changed": int(df["value_changed"].sum()),
        "timestamp_changed": int(df["timestamp_changed"].sum()),
    }


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct and compare IBM calibrations for completed Candidate #1 runs from qrc_qpu_run."
    )
    parser.add_argument("--table", default=TABLE_NAME)
    parser.add_argument("--backend", default=BACKEND_NAME)
    parser.add_argument("--candidate", default=CANDIDATE_KEY)
    parser.add_argument("--timezone", default=LOCAL_TZ_NAME)
    parser.add_argument("--outdir", default=str(OUTDIR))
    args = parser.parse_args()

    tz = ZoneInfo(args.timezone)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 128)
    print("WEEK 11.1G.2 — CANDIDATE #1 COMPLETED-RUN CALIBRATION RECONSTRUCTION + DIFF (FIXED)")
    print("=" * 128)
    print("READ-ONLY: DB query + IBM historical calibration lookup. No QPU submission.")
    print("2026 forecasting/test data are not loaded.")
    print(f"DB table:   {args.table}")
    print(f"Candidate:  {args.candidate}")
    print(f"Backend:    {args.backend}")

    runs = load_completed_candidate1_runs(args.table, args.backend, args.candidate)
    print(f"Completed actual QPU runs found: {len(runs)}")

    service = get_service()
    backend = service.backend(args.backend, use_fractional_gates=False)

    # Cache historical snapshots by exact last_update timestamp so multiple runs
    # under the same calibration do not repeat extraction work.
    snapshot_cache: Dict[str, CalibrationSnapshot] = {}
    qubit_cache: Dict[str, pd.DataFrame] = {}
    gate_cache: Dict[str, pd.DataFrame] = {}

    resolved_rows: List[Dict[str, Any]] = []

    for _, row in runs.iterrows():
        run_start_utc = ensure_utc(row["run_start_utc"])
        snap = fetch_snapshot_for_run(backend, run_start_utc)
        snap_key = snap.actual_last_update_utc.isoformat()

        if snap_key not in snapshot_cache:
            snapshot_cache[snap_key] = snap
            qubit_cache[snap_key] = extract_qubit_long(snap.properties, snap_key)
            gate_cache[snap_key] = extract_gate_long(snap.properties, snap_key)

        layout = row["layout"]
        resolved_rows.append(
            {
                "id": int(row["id"]),
                "run_uuid": row["run_uuid"],
                "script_name": row["script_name"],
                "run_kind": row["run_kind"],
                "job_id": row["job_id"],
                "job_status": row["job_status"],
                "ibm_job_running_at_utc": run_start_utc.isoformat(),
                "ibm_job_running_local": run_start_utc.astimezone(tz).isoformat(),
                "calibration_query_utc": snap.query_time_utc.isoformat(),
                "calibration_resolution_method": "backend.properties(datetime=exact_ibm_job_running_at_utc)",
                "calibration_last_update_utc": snap.actual_last_update_utc.isoformat(),
                "calibration_last_update_local": snap.actual_last_update_utc.astimezone(tz).isoformat(),
                "seconds_from_calibration_to_run": (run_start_utc - snap.actual_last_update_utc).total_seconds(),
                "physical_layout_json": json.dumps(layout) if layout else None,
                "qpu_rmse": row["qpu_rmse"],
                "qpu_mae": row["qpu_mae"],
                "qpu_bias": row["qpu_bias"],
                "qpu_feature_rmse": row["qpu_feature_rmse"],
                "prediction_correlation_qpu_vs_ideal": row["prediction_correlation_qpu_vs_ideal"],
                "max_2q_error_percent_recorded": row["max_2q_error_percent"],
                "max_readout_error_percent_recorded": row["max_readout_error_percent"],
                "max_1q_error_percent_recorded": row["max_1q_error_percent"],
                "min_t1_us_recorded": row["min_t1_us"],
                "min_t2_us_recorded": row["min_t2_us"],
                "shots_per_setting": row["shots_per_setting"],
                "qpu_charge_time_seconds": row["qpu_charge_time_seconds"],
            }
        )

    resolved = pd.DataFrame(resolved_rows).sort_values(["ibm_job_running_at_utc", "id"], kind="stable").reset_index(drop=True)
    resolved.insert(0, "completed_run_index", np.arange(1, len(resolved) + 1))

    # Mark whether the calibration changed from the immediately previous completed run.
    resolved["same_calibration_as_previous"] = False
    resolved["same_layout_as_previous"] = False
    for i in range(1, len(resolved)):
        resolved.loc[i, "same_calibration_as_previous"] = (
            resolved.loc[i, "calibration_last_update_utc"] == resolved.loc[i - 1, "calibration_last_update_utc"]
        )
        resolved.loc[i, "same_layout_as_previous"] = (
            resolved.loc[i, "physical_layout_json"] == resolved.loc[i - 1, "physical_layout_json"]
        )

    print("\nCompleted Candidate #1 runs + active calibration")
    print("-" * 128)
    display_cols = [
        "completed_run_index", "id", "run_kind", "ibm_job_running_local",
        "calibration_last_update_local", "physical_layout_json", "qpu_rmse",
        "qpu_feature_rmse", "prediction_correlation_qpu_vs_ideal"
    ]
    with pd.option_context("display.max_columns", 30, "display.width", 220):
        print(resolved[display_cols].to_string(index=False))

    # Group runs by actual calibration snapshot.
    groups = (
        resolved.groupby(["calibration_last_update_utc", "calibration_last_update_local"], dropna=False)
        .agg(
            n_runs=("id", "count"),
            run_ids=("id", lambda s: ",".join(str(int(x)) for x in s)),
            run_kinds=("run_kind", lambda s: ";".join(str(x) for x in s)),
            rmse_min=("qpu_rmse", "min"),
            rmse_max=("qpu_rmse", "max"),
            rmse_mean=("qpu_rmse", "mean"),
        )
        .reset_index()
    )

    # Consecutive-run calibration diffs.
    pair_rows: List[Dict[str, Any]] = []
    for i in range(1, len(resolved)):
        a = resolved.iloc[i - 1]
        b = resolved.iloc[i]

        key_a = a["calibration_last_update_utc"]
        key_b = b["calibration_last_update_utc"]
        qa = qubit_cache[key_a]
        qb = qubit_cache[key_b]
        ga = gate_cache[key_a]
        gb = gate_cache[key_b]

        qdiff = diff_long(qa, qb, ["qubit", "property"])
        gdiff = diff_long(ga, gb, ["gate", "qubits", "undirected_qubits", "property"])

        layout_a = parse_layout(a["physical_layout_json"])
        layout_b = parse_layout(b["physical_layout_json"])

        a_q = filter_layout_qubits(qdiff, layout_a, "previous_run_layout")
        a_g = filter_layout_cz(gdiff, layout_a, "previous_run_layout")
        b_q = filter_layout_qubits(qdiff, layout_b, "current_run_layout")
        b_g = filter_layout_cz(gdiff, layout_b, "current_run_layout")

        pair_no = i
        pair_tag = (
            f"pair_{pair_no:02d}_run{int(a['id'])}_to_run{int(b['id'])}_"
            f"{safe_token(a['run_kind'])}_to_{safe_token(b['run_kind'])}"
        )

        # Save detailed diffs even when snapshots are identical; identical snapshot
        # pairs become explicit evidence that calibration did not change.
        qdiff.to_csv(outdir / f"{PREFIX}_{pair_tag}_all_qubits.csv", index=False)
        gdiff.to_csv(outdir / f"{PREFIX}_{pair_tag}_all_gates.csv", index=False)
        a_q.to_csv(outdir / f"{PREFIX}_{pair_tag}_previous_layout_qubits.csv", index=False)
        a_g.to_csv(outdir / f"{PREFIX}_{pair_tag}_previous_layout_cz.csv", index=False)
        b_q.to_csv(outdir / f"{PREFIX}_{pair_tag}_current_layout_qubits.csv", index=False)
        b_g.to_csv(outdir / f"{PREFIX}_{pair_tag}_current_layout_cz.csv", index=False)

        qs = summarize_changed(qdiff)
        gs = summarize_changed(gdiff)
        aqs = summarize_changed(a_q)
        ags = summarize_changed(a_g)
        bqs = summarize_changed(b_q)
        bgs = summarize_changed(b_g)

        pair_rows.append(
            {
                "pair_index": pair_no,
                "run_A_id": int(a["id"]),
                "run_B_id": int(b["id"]),
                "run_A_kind": a["run_kind"],
                "run_B_kind": b["run_kind"],
                "run_A_start_local": a["ibm_job_running_local"],
                "run_B_start_local": b["ibm_job_running_local"],
                "snapshot_A_local": a["calibration_last_update_local"],
                "snapshot_B_local": b["calibration_last_update_local"],
                "same_calibration_snapshot": bool(key_a == key_b),
                "layout_A": a["physical_layout_json"],
                "layout_B": b["physical_layout_json"],
                "same_physical_layout": bool(a["physical_layout_json"] == b["physical_layout_json"]),
                "qpu_rmse_A": a["qpu_rmse"],
                "qpu_rmse_B": b["qpu_rmse"],
                "qpu_rmse_change_B_minus_A": float(b["qpu_rmse"] - a["qpu_rmse"]),
                "qpu_feature_rmse_A": a["qpu_feature_rmse"],
                "qpu_feature_rmse_B": b["qpu_feature_rmse"],
                "prediction_corr_A": a["prediction_correlation_qpu_vs_ideal"],
                "prediction_corr_B": b["prediction_correlation_qpu_vs_ideal"],
                "global_qubit_rows_changed": qs["changed_any"],
                "global_gate_rows_changed": gs["changed_any"],
                "previous_layout_qubit_rows_changed": aqs["changed_any"],
                "previous_layout_cz_rows_changed": ags["changed_any"],
                "current_layout_qubit_rows_changed": bqs["changed_any"],
                "current_layout_cz_rows_changed": bgs["changed_any"],
            }
        )

    pair_summary = pd.DataFrame(pair_rows)

    # Save master outputs.
    runs_path = outdir / f"{PREFIX}_completed_runs_with_calibration.csv"
    groups_path = outdir / f"{PREFIX}_calibration_groups.csv"
    pairs_path = outdir / f"{PREFIX}_consecutive_run_diff_summary.csv"
    json_path = outdir / f"{PREFIX}_summary.json"

    resolved.to_csv(runs_path, index=False)
    groups.to_csv(groups_path, index=False)
    pair_summary.to_csv(pairs_path, index=False)

    summary = {
        "table": args.table,
        "backend": args.backend,
        "candidate_key": args.candidate,
        "timezone": args.timezone,
        "n_completed_qpu_runs": int(len(resolved)),
        "n_unique_calibration_snapshots": int(resolved["calibration_last_update_utc"].nunique()),
        "calibration_resolution_method": "backend.properties(datetime=exact_ibm_job_running_at_utc), with UTC normalization",
        "completed_run_ids": [int(x) for x in resolved["id"].tolist()],
        "unique_calibration_snapshots_local": sorted(resolved["calibration_last_update_local"].unique().tolist()),
        "same_snapshot_run_groups": groups.to_dict(orient="records"),
        "interpretation_note": (
            "Historical backend.properties snapshots are resolved using the exact IBM QPU RUNNING timestamp "
            "normalized to UTC. IBM-reported calibration/benchmark data do not expose every internal "
            "pulse/control change. Runs sharing the exact same normalized last_update_date are treated as "
            "the same reported calibration snapshot."
        ),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nCalibration groups")
    print("-" * 128)
    with pd.option_context("display.max_columns", 20, "display.width", 220):
        print(groups.to_string(index=False))

    if not pair_summary.empty:
        print("\nConsecutive completed-run calibration diff summary")
        print("-" * 128)
        cols = [
            "pair_index", "run_A_id", "run_B_id", "run_A_kind", "run_B_kind",
            "same_calibration_snapshot", "same_physical_layout",
            "qpu_rmse_A", "qpu_rmse_B", "qpu_rmse_change_B_minus_A",
            "previous_layout_qubit_rows_changed", "previous_layout_cz_rows_changed",
            "current_layout_qubit_rows_changed", "current_layout_cz_rows_changed",
        ]
        with pd.option_context("display.max_columns", 30, "display.width", 240):
            print(pair_summary[cols].to_string(index=False))

    print("\nSaved master outputs")
    print("-" * 128)
    print(f"  {runs_path}")
    print(f"  {groups_path}")
    print(f"  {pairs_path}")
    print(f"  {json_path}")
    print(f"  plus detailed per-pair qubit/CZ CSV files with prefix {PREFIX}_pair_*")
    print("\nNo QPU job was submitted.")


if __name__ == "__main__":
    main()
