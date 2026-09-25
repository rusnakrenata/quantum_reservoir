#!/usr/bin/env python
from __future__ import annotations

"""
WEEK 11.1H.1 — CANDIDATE #1 QRC-AWARE PHYSICAL-LAYOUT SELECTION (V3 FAST, BENCHMARK-BLIND)
=================================================================

Goal
----
Select the physical 6-qubit layout for frozen Candidate #1 using a hierarchy
that is more task-specific than either the custom Rule-5 lexicographic selector
or Qiskit's generic default placement.

The experiment is intentionally split into four stages:

  A. LOCAL / ZERO-QPU screening
     * Fresh ibm_kingston calibration.
     * Enumerate native H3 embeddings through the existing Week-11 selector.
     * Require strict routing_method='none' / zero-SWAP compatibility.
     * Evaluate the ACTUAL Candidate-1 measurement workload on each layout.
     * Use a physically motivated calibration-likelihood proxy only as a cheap
       pre-filter. No validation targets are used.

  B. LOCAL calibration-based noisy simulation
     * Keep a configurable simulation pool (default 12 layouts), selected ONLY
       by the current-calibration / actual-workload local proxy.
     * NO Qiskit-default or Rule-5 benchmark layout is forced into the pool.
       Prior 2025 hardware outcomes are not consulted.
     * Simulate 20 deterministic 2022-2024 training-only endpoints using an
       AerSimulator built from the fresh IBM backend calibration.
     * Primary local metric:

           E_delta_y_sim(L)
             = RMS_i [ yhat_noisy_sim(i,L) - yhat_ideal(i) ]

       The frozen training-only Ridge readout is used. True claims y_i are NOT
       used in layout ranking.

  C. REAL-QPU layout probe
     * Two finalists total by default, selected ONLY as the two best layouts under the
       training-only calibration-noise simulation metric.
     * NO Qiskit-default or Rule-5 benchmark is forced into the finalists.
     * 100 deterministic 2022-2024 training-only endpoints.
     * 2 grouped measurement settings / endpoint.
     * 512 shots/setting by default.
     * Layout order is cyclically interleaved across endpoints to reduce drift
       bias.
     * Primary final-selection metric:

           E_delta_y_QPU(L)
             = RMS_i [ yhat_QPU(i,L) - yhat_ideal(i) ]

       The layout with the smallest value wins. Again, true target y_i is not
       used for layout selection.

  D. IMMEDIATE full Candidate-1 2025 validation
     * The exact winning physical layout is frozen.
     * NO fresh Rule-5 reselection.
     * NO new Qiskit layout selection.
     * Immediately call the authoritative 11_1A engine for all 365 validation
       endpoints at 2048 shots/setting (default), using that exact layout.
     * 2026 remains frozen / never loaded by this wrapper.

Important scientific rule
-------------------------
The selector is BLIND to all prior 2025 QPU benchmark outcomes. Neither the
previous Qiskit-layout result nor the previous Rule-5 result may enter the
simulation pool, finalist set, ranking metric, or winner decision. The final
QPU probe is the authority; the noisy simulator is only a pre-filter.

The 2025 split is opened only AFTER the winning physical layout has been frozen.
The untouched 2026 split remains the final unbiased test set.

Required project files
----------------------
This script must live beside:
    11_1A_best_candidate_direct_qpu.py
and the files imported by that authoritative script.

Default behavior
----------------
Without --submit, the script performs LOCAL screening + noisy simulation only.
It does NOT submit a QPU job.

With --submit, it submits the finalist QPU probe (two layouts by default). Unless --probe-only is
specified, it immediately submits the full 365-day Candidate-1 run on the
winner after the probe finishes.
"""

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2

from ibm_account import get_service


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

BASE_FILE = HERE / "11_1A_best_candidate_direct_qpu.py"

PREFIX = "11_1H1_candidate1_qrc_aware_layout_selection"
FULL_ALIAS_PREFIX = f"{PREFIX}_full"
BASE_OUTPUT_PREFIX = "11_1A_best_candidate_qpu"

DEFAULT_BACKEND = "ibm_kingston"
LOCAL_TZ = ZoneInfo("Europe/Bratislava")
CALIBRATION_POLL_SECONDS = 5
DEFAULT_PROBE_INPUTS = 100
DEFAULT_PROBE_SHOTS = 512
DEFAULT_FULL_SHOTS = 2048
DEFAULT_SIM_SHOTS = 64
DEFAULT_SIM_POOL = 12
DEFAULT_SIM_INPUTS = 20
DEFAULT_N_FINALISTS = 2
DEFAULT_SELECTOR_SHORTLIST = 5000
DEFAULT_AER_BATCH = 25

OPT_LEVEL = 1
SEED_TRANSPILE = 42
SEED_SIMULATOR = 424242

EXPECTED_FEATURES = [
    "X0", "X1", "X2",
    "Z0", "Z1", "Z2", "Z3",
    "YX45",
]
SETTING_X = "XXXZYX"
SETTING_Z = "ZZZZZZ"

# Historical observed QPU charge for the full 365-day Candidate-1 Sampler run.
# Used only as a safety/planning heuristic, never as an IBM billing formula.
HISTORICAL_FULL_C1_QPU_CHARGE_S = 414.0
HISTORICAL_FULL_C1_N_CIRCUITS = 730
HISTORICAL_FULL_C1_SHOTS = 2048

KNOWN_BASE_SUFFIXES = [
    "_preflight.json",
    "_resources.csv",
    "_selected_endpoints.csv",
    "_raw_counts.csv",
    "_features_predictions.csv",
    "_summary.json",
    "_job.json",
]


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Put this script in the same project directory as "
            "11_1A_best_candidate_direct_qpu.py."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = load_module(BASE_FILE, "qrc_11_1h1_base_candidate1")
selector = base.selector


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def json_safe(x):
    if hasattr(base, "json_safe"):
        try:
            return base.json_safe(x)
        except Exception:
            pass
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def write_json(path: Path, payload):
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def rmse(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def mae(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(x - y)))


def corr(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def parse_layout(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(v) for v in json.loads(str(x))]


def layout_key(layout):
    return tuple(parse_layout(layout))


def layout_json(layout):
    return json.dumps([int(v) for v in parse_layout(layout)])


def bool_series(s: pd.Series):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def ensure_utc(dt):
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = pd.Timestamp(dt).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def calibration_record(props):
    """Return one canonical calibration timestamp in UTC plus a human local view.

    IBM may return ``last_update_date`` already timezone-aware (for example
    ``+02:00`` during Slovak summer time).  We NEVER subtract two hours by
    hand.  ``astimezone(UTC)`` gives the canonical comparison timestamp and
    ``astimezone(Europe/Bratislava)`` gives the human-readable local timestamp.
    """
    if props is None:
        return None
    raw = getattr(props, "last_update_date", None)
    dt_utc = ensure_utc(raw)
    dt_local = None if dt_utc is None else dt_utc.astimezone(LOCAL_TZ)
    return {
        "last_update_date_raw": None if raw is None else str(raw),
        "last_update_date_utc": None if dt_utc is None else dt_utc.isoformat(),
        "last_update_date_local": None if dt_local is None else dt_local.isoformat(),
        "timezone_note": (
            "UTC is used for programmatic comparison; Europe/Bratislava is only "
            "the human-readable display. During CEST, local time is UTC+02:00."
        ),
    }


def calibration_at(backend, running_utc):
    if running_utc is None:
        return None
    dt = ensure_utc(running_utc)
    try:
        props = backend.properties(datetime=dt)
    except Exception as exc:
        return {
            "query_utc": dt.isoformat(),
            "error": str(exc),
        }
    out = calibration_record(props) or {}
    out["query_utc"] = dt.isoformat()
    return out


def current_calibration(backend):
    try:
        # refresh() updates the client-side backend description before reading
        # the currently reported calibration snapshot.
        try:
            backend.refresh()
        except Exception:
            pass
        return calibration_record(backend.properties())
    except Exception as exc:
        return {"error": str(exc)}


def calibration_utc(record):
    if not isinstance(record, dict):
        return None
    return record.get("last_update_date_utc")


def job_status_name(job):
    try:
        status = job.status()
    except Exception:
        return "UNKNOWN"
    name = getattr(status, "name", None)
    if name:
        return str(name).upper()
    text = str(status).upper()
    # Handles representations such as JobStatus.QUEUED.
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def cancel_job_best_effort(job):
    try:
        job.cancel()
        return True, None
    except Exception as exc:
        return False, str(exc)


def guarded_job_result(job, backend, expected_calibration_utc, label, poll_seconds=CALIBRATION_POLL_SECONDS):
    """Wait for a Runtime job while protecting against queue-time recalibration.

    While the job is not yet terminal we repeatedly read the CURRENT reported
    backend calibration.  If it changes before/at RUNNING, we attempt to cancel
    and abort rather than knowingly execute a layout selected under an obsolete
    calibration.  After completion the caller still reconstructs the exact
    RUNNING-time calibration with backend.properties(datetime=running_utc).
    """
    terminal = {"DONE", "CANCELLED", "CANCELED", "ERROR", "FAILED"}
    last_status = None
    while True:
        status = job_status_name(job)
        if status != last_status:
            print(f"[{label}] job status: {status}")
            last_status = status

        if status in terminal:
            break

        now_cal = current_calibration(backend)
        now_utc = calibration_utc(now_cal)
        if expected_calibration_utc and now_utc and now_utc != expected_calibration_utc:
            print()
            print(f"[{label}] CALIBRATION CHANGED BEFORE/AT QPU EXECUTION")
            print(f"  expected UTC: {expected_calibration_utc}")
            print(f"  current  UTC: {now_utc}")
            print(f"  current local: {now_cal.get('last_update_date_local')}")
            cancelled, cancel_error = cancel_job_best_effort(job)
            print(f"  cancellation attempted: {cancelled}")
            if cancel_error:
                print(f"  cancellation error: {cancel_error}")
            raise RuntimeError(
                f"{label}: IBM calibration changed after layout selection. "
                "The job was cancelled if still cancellable; no full validation "
                "will be launched from this stale selection."
            )

        time.sleep(float(poll_seconds))

    # job.result() also raises appropriately for cancelled/failed jobs.
    return job.result()


def get_service_usage(service):
    if hasattr(base, "get_service_usage"):
        try:
            return base.get_service_usage(service)
        except Exception:
            pass
    for name in ("usage", "usage_info"):
        fn = getattr(service, name, None)
        if callable(fn):
            try:
                return json_safe(fn())
            except Exception:
                pass
    return None


def remaining_usage_seconds(usage):
    if not isinstance(usage, dict):
        return None
    for key in (
        "usage_remaining_seconds",
        "remaining_seconds",
        "usage_remaining",
    ):
        if key in usage:
            try:
                return float(usage[key])
            except Exception:
                pass
    return None


# =============================================================================
# CANDIDATE / FROZEN READOUT / TRAINING-ONLY PROBE INPUTS
# =============================================================================


def load_candidate_and_frozen():
    if not hasattr(base, "load_candidate"):
        raise AttributeError("11_1A base has no load_candidate().")
    if not hasattr(base, "build_frozen_readout"):
        raise AttributeError("11_1A base has no build_frozen_readout().")

    candidate, manifest_meta = base.load_candidate()
    frozen = base.build_frozen_readout(candidate, manifest_meta)

    features = list(getattr(base, "FEATURES", []))
    if features != EXPECTED_FEATURES:
        raise RuntimeError(
            "Candidate-1 feature order changed.\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Found:    {features}"
        )

    if int(candidate["window"]) != 4:
        raise RuntimeError(f"Expected Candidate #1 W=4, got {candidate['window']}.")
    if int(candidate["r"]) != 3:
        raise RuntimeError(f"Expected Candidate #1 r=3, got {candidate['r']}.")
    if str(candidate["topology"]) != "H3":
        raise RuntimeError(f"Expected Candidate #1 H3, got {candidate['topology']}.")

    required = [
        "endpoints", "X", "train_mask", "angles",
        "model", "scaler", "keep",
    ]
    missing = [k for k in required if k not in frozen]
    if missing:
        raise KeyError(f"Frozen readout is missing required keys: {missing}")

    return candidate, manifest_meta, frozen


def choose_training_probe_endpoints(frozen, n_inputs):
    endpoints = np.asarray(frozen["endpoints"], dtype=int)
    train_mask = np.asarray(frozen["train_mask"], dtype=bool)
    if len(endpoints) != len(train_mask):
        raise RuntimeError("endpoints/train_mask length mismatch.")

    train_endpoints = endpoints[train_mask]
    if len(train_endpoints) == 0:
        raise RuntimeError("No training endpoints available.")

    n = int(max(1, min(int(n_inputs), len(train_endpoints))))
    offsets = np.unique(np.linspace(0, len(train_endpoints) - 1, n, dtype=int))
    chosen = train_endpoints[offsets]

    if len(chosen) != n:
        raise RuntimeError(
            f"Expected {n} deterministic training probe endpoints, got {len(chosen)}."
        )

    endpoint_to_xrow = {int(e): i for i, e in enumerate(endpoints)}
    rows = np.asarray([endpoint_to_xrow[int(e)] for e in chosen], dtype=int)
    X_ideal = np.asarray(frozen["X"], dtype=float)[rows]

    # IMPORTANT: no y / target values are used here.
    pred_ideal = base.common.predict_scaled_ridge(
        frozen["model"],
        frozen["scaler"],
        frozen["keep"],
        X_ideal,
    )

    return chosen, X_ideal, np.asarray(pred_ideal, dtype=float)


# =============================================================================
# COUNT -> FEATURE PARSING FOR CANDIDATE #1
# =============================================================================


def bit_from_qiskit_string(bitstring, c_index):
    s = str(bitstring).replace(" ", "")
    return int(s[-1 - int(c_index)])


def expectation_from_counts(counts, classical_bits):
    total = float(sum(float(v) for v in counts.values()))
    if total <= 0:
        raise RuntimeError("Empty counts.")
    acc = 0.0
    for bitstring, n in counts.items():
        eig = 1.0
        for c in classical_bits:
            eig *= -1.0 if bit_from_qiskit_string(bitstring, c) else 1.0
        acc += float(n) * eig
    return float(acc / total)


def get_sampler_pub_counts(pub_result):
    data = pub_result.data
    reg = getattr(data, "m", None)
    if reg is None:
        raise RuntimeError("Sampler result does not contain classical register 'm'.")
    return reg.get_counts()


def features_from_setting_counts(per_endpoint):
    """
    per_endpoint[endpoint][setting_label] -> counts
    Returns dict endpoint -> 8-feature vector in EXPECTED_FEATURES order.
    """
    out = {}
    for endpoint, setting_map in per_endpoint.items():
        if SETTING_X not in setting_map or SETTING_Z not in setting_map:
            raise RuntimeError(
                f"Endpoint {endpoint}: missing {SETTING_X} or {SETTING_Z}."
            )
        cx = setting_map[SETTING_X]
        cz = setting_map[SETTING_Z]
        vals = np.asarray(
            [
                expectation_from_counts(cx, [0]),
                expectation_from_counts(cx, [1]),
                expectation_from_counts(cx, [2]),
                expectation_from_counts(cz, [0]),
                expectation_from_counts(cz, [1]),
                expectation_from_counts(cz, [2]),
                expectation_from_counts(cz, [3]),
                expectation_from_counts(cx, [4, 5]),
            ],
            dtype=float,
        )
        out[int(endpoint)] = vals
    return out


def features_to_predictions(frozen, endpoints, features_by_endpoint):
    X = np.vstack([features_by_endpoint[int(e)] for e in endpoints]).astype(float)
    pred = base.common.predict_scaled_ridge(
        frozen["model"],
        frozen["scaler"],
        frozen["keep"],
        X,
    )
    return X, np.asarray(pred, dtype=float)


# =============================================================================
# QISKIT DEFAULT LAYOUT BENCHMARK
# =============================================================================


def extract_initial_layout(transpiled, logical, n_logical=6):
    tl = getattr(transpiled, "layout", None)
    if tl is None:
        raise RuntimeError("Qiskit transpiler returned no layout information.")

    fn = getattr(tl, "initial_index_layout", None)
    if callable(fn):
        try:
            vals = fn(filter_ancillas=True)
        except TypeError:
            vals = fn()
        vals = [int(v) for v in vals]
        if len(vals) >= n_logical:
            return vals[:n_logical]

    init = getattr(tl, "initial_layout", None)
    if init is None:
        raise RuntimeError("Cannot recover Qiskit initial layout.")

    out = []
    for q in logical.qubits[:n_logical]:
        out.append(int(init[q]))
    return out


def qiskit_default_layout(backend, candidate, frozen, endpoint, settings):
    logical = base.build_measurement_circuit(
        candidate,
        frozen["angles"],
        int(endpoint),
        settings[0],
    )
    tqc = transpile(
        logical,
        backend=backend,
        optimization_level=int(getattr(base, "OPT_LEVEL", OPT_LEVEL)),
        seed_transpiler=int(getattr(base, "SEED_TRANSPILE", SEED_TRANSPILE)),
        scheduling_method="alap",
    )
    backend.check_faulty(tqc)
    layout = extract_initial_layout(tqc, logical, 6)

    # Require the selected layout itself to be a strict native embedding.
    audit = transpile(
        logical,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        optimization_level=int(getattr(base, "OPT_LEVEL", OPT_LEVEL)),
        seed_transpiler=int(getattr(base, "SEED_TRANSPILE", SEED_TRANSPILE)),
        scheduling_method="alap",
    )
    backend.check_faulty(audit)
    if int(audit.count_ops().get("swap", 0)) != 0:
        raise RuntimeError("Qiskit default layout failed zero-SWAP audit.")
    return layout


# =============================================================================
# ACTUAL-WORKLOAD CALIBRATION PROXY
# =============================================================================


def target_instruction_error(backend, opname, qargs):
    try:
        ip = backend.target[str(opname)][tuple(int(q) for q in qargs)]
    except Exception:
        return np.nan
    if ip is None:
        return np.nan
    return safe_float(getattr(ip, "error", np.nan))


def physical_qargs(circuit, inst):
    return [int(circuit.find_bit(q).index) for q in inst.qubits]


def calibration_nll_for_circuit(circuit, backend):
    """
    Cheap physical proxy:
        sum_g -log(1-e_g) + sum_measured -log(1-e_readout)

    It is NOT the final selection metric. It is only a no-QPU pre-filter before
    circuit-level noisy simulation. No arbitrary scalar weights are used.
    """
    props = backend.properties()
    gate_nll = 0.0
    readout_nll = 0.0
    missing_gate_errors = 0
    measured_physical = []

    for inst in circuit.data:
        name = str(inst.operation.name)
        qargs = physical_qargs(circuit, inst)

        if name == "measure":
            if len(qargs) == 1:
                measured_physical.append(int(qargs[0]))
            continue
        if name in {"barrier", "delay"}:
            continue

        e = target_instruction_error(backend, name, qargs)
        if np.isfinite(e):
            e = min(max(float(e), 0.0), 1.0 - 1e-15)
            gate_nll += -math.log1p(-e)
        else:
            missing_gate_errors += 1

    # Each measured physical qubit contributes one readout event per circuit.
    for q in measured_physical:
        try:
            e = float(props.readout_error(int(q)))
        except Exception:
            continue
        if np.isfinite(e):
            e = min(max(e, 0.0), 1.0 - 1e-15)
            readout_nll += -math.log1p(-e)

    duration_us = float(circuit.estimate_duration(backend.target, unit="s")) * 1e6
    return {
        "gate_nll": gate_nll,
        "readout_nll": readout_nll,
        "total_nll": gate_nll + readout_nll,
        "proxy_success_probability": math.exp(-(gate_nll + readout_nll)),
        "missing_gate_errors": int(missing_gate_errors),
        "duration_us": duration_us,
        "depth": int(circuit.depth()),
        "n_cz": int(circuit.count_ops().get("cz", 0)),
        "n_reset": int(circuit.count_ops().get("reset", 0)),
    }


def strict_compile_measurement_circuit(backend, logical, layout):
    isa = transpile(
        logical,
        backend=backend,
        initial_layout=parse_layout(layout),
        routing_method="none",
        optimization_level=int(getattr(base, "OPT_LEVEL", OPT_LEVEL)),
        seed_transpiler=int(getattr(base, "SEED_TRANSPILE", SEED_TRANSPILE)),
        scheduling_method="alap",
    )
    backend.check_faulty(isa)
    if int(isa.count_ops().get("swap", 0)) != 0:
        raise RuntimeError(f"SWAP found for layout {layout}.")
    return isa


def actual_workload_proxy_table(
    backend,
    candidate,
    frozen,
    layouts_df,
    representative_endpoint,
    settings,
):
    logical_by_setting = {
        s[0]: base.build_measurement_circuit(
            candidate,
            frozen["angles"],
            int(representative_endpoint),
            s,
        )
        for s in settings
    }

    rows = []
    total = len(layouts_df)
    for k, (_, row) in enumerate(layouts_df.iterrows(), start=1):
        layout = parse_layout(row["layout"])
        rec = {
            "layout": layout_json(layout),
            "proxy_compile_pass": True,
            "proxy_compile_error": "",
        }
        try:
            nll = 0.0
            prob = 1.0
            duration = 0.0
            n_cz = 0
            n_reset = 0
            missing = 0
            max_depth = 0
            for label, logical in logical_by_setting.items():
                isa = strict_compile_measurement_circuit(backend, logical, layout)
                m = calibration_nll_for_circuit(isa, backend)
                rec[f"{label}_nll"] = m["total_nll"]
                rec[f"{label}_duration_us"] = m["duration_us"]
                rec[f"{label}_depth"] = m["depth"]
                nll += m["total_nll"]
                prob *= m["proxy_success_probability"]
                duration += m["duration_us"]
                n_cz += m["n_cz"]
                n_reset += m["n_reset"]
                missing += m["missing_gate_errors"]
                max_depth = max(max_depth, m["depth"])
            rec.update(
                {
                    "actual_workload_proxy_nll": float(nll),
                    "actual_workload_proxy_success_probability": float(prob),
                    "actual_workload_duration_us": float(duration),
                    "actual_workload_max_setting_depth": int(max_depth),
                    "actual_workload_n_cz": int(n_cz),
                    "actual_workload_n_reset": int(n_reset),
                    "actual_workload_missing_gate_errors": int(missing),
                }
            )
        except Exception as exc:
            rec["proxy_compile_pass"] = False
            rec["proxy_compile_error"] = str(exc)
            rec["actual_workload_proxy_nll"] = np.inf
            rec["actual_workload_proxy_success_probability"] = 0.0
        rows.append(rec)

        if k == 1 or k % 50 == 0 or k == total:
            print(f"  workload proxy: {k}/{total} layouts")

    proxy = pd.DataFrame(rows)
    merged = layouts_df.copy()
    merged["_layout_key"] = merged["layout"].apply(lambda x: layout_json(parse_layout(x)))
    proxy["_layout_key"] = proxy["layout"]
    out = merged.merge(proxy.drop(columns=["layout"]), on="_layout_key", how="left")
    out["layout"] = out["_layout_key"]
    return out.drop(columns=["_layout_key"])


def top_unique_proxy_layouts(primary_df, n_total):
    """
    Keep the best unique layouts according to the local actual-workload proxy.

    Strict no-leakage rule: no externally known benchmark layout is forced in.
    The returned pool depends only on the current calibration / compiled workload
    quantities already present in ``primary_df``.
    """
    chosen = []
    seen = set()

    for _, row in primary_df.iterrows():
        key = layout_key(row["layout"])
        if key in seen:
            continue
        chosen.append(("proxy", list(key)))
        seen.add(key)
        if len(chosen) >= int(n_total):
            break

    return chosen


# =============================================================================
# CALIBRATION-BASED AER NOISY SIMULATION
# =============================================================================


def build_aer_from_backend(backend):
    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise RuntimeError(
            "qiskit-aer is required for the local calibration-based noisy "
            "simulation stage. Install/activate the same environment used in "
            "Weeks 9-11."
        ) from exc

    try:
        sim = AerSimulator.from_backend(backend, method="matrix_product_state")
    except TypeError:
        sim = AerSimulator.from_backend(backend)
        try:
            sim.set_options(method="matrix_product_state")
        except Exception:
            pass

    # FAST local-screening configuration.  Aer defaults to only one experiment
    # (circuit) at a time.  Candidate #1 gives us many independent small active
    # circuits, so experiment-level parallelism is substantially more useful
    # than running them serially.  enable_truncation removes inactive physical
    # wires from the simulation state.  For small active systems and tens of
    # shots, constructing probabilities once is usually preferable to copying
    # and directly measuring the MPS for every shot.
    try:
        sim.set_options(
            max_parallel_threads=0,
            max_parallel_experiments=0,
            enable_truncation=True,
            mps_omp_threads=1,
            mps_sample_measure_algorithm="mps_probabilities",
        )
    except Exception as exc:
        print(f"WARNING: could not enable all Aer fast options: {exc}")
    return sim


def run_aer_layout(
    simulator,
    backend,
    candidate,
    frozen,
    endpoints,
    settings,
    layout,
    shots,
    batch_size,
):
    circuits = []
    meta = []
    for endpoint in endpoints:
        for setting in settings:
            logical = base.build_measurement_circuit(
                candidate,
                frozen["angles"],
                int(endpoint),
                setting,
            )
            isa = strict_compile_measurement_circuit(backend, logical, layout)
            circuits.append(isa)
            meta.append((int(endpoint), str(setting[0])))

    counts_by_endpoint = {int(e): {} for e in endpoints}

    for start in range(0, len(circuits), int(batch_size)):
        stop = min(start + int(batch_size), len(circuits))
        batch = circuits[start:stop]
        result = simulator.run(
            batch,
            shots=int(shots),
            seed_simulator=int(SEED_SIMULATOR + start),
        ).result()
        for j in range(len(batch)):
            counts = result.get_counts(j)
            endpoint, setting = meta[start + j]
            counts_by_endpoint[endpoint][setting] = counts

    return counts_by_endpoint


def noisy_simulation_stage(
    backend,
    candidate,
    frozen,
    endpoints,
    X_ideal,
    pred_ideal,
    settings,
    pool,
    shots,
    batch_size,
):
    simulator = build_aer_from_backend(backend)
    summaries = []
    daily_rows = []

    for rank, (source, layout) in enumerate(pool, start=1):
        print()
        print(
            f"[Aer] layout {rank}/{len(pool)} source={source} layout={layout} "
            f"shots={shots}"
        )
        counts = run_aer_layout(
            simulator=simulator,
            backend=backend,
            candidate=candidate,
            frozen=frozen,
            endpoints=endpoints,
            settings=settings,
            layout=layout,
            shots=shots,
            batch_size=batch_size,
        )
        feat = features_from_setting_counts(counts)
        X_sim, pred_sim = features_to_predictions(frozen, endpoints, feat)

        delta_pred = pred_sim - pred_ideal
        delta_feat = X_sim - X_ideal

        summaries.append(
            {
                "layout": layout_json(layout),
                "pool_source": source,
                "sim_shots_per_setting": int(shots),
                "sim_pred_distortion_rmse_claims": float(np.sqrt(np.mean(delta_pred ** 2))),
                "sim_pred_distortion_mae_claims": float(np.mean(np.abs(delta_pred))),
                "sim_pred_distortion_bias_claims": float(np.mean(delta_pred)),
                "sim_feature_rmse": float(np.sqrt(np.mean(delta_feat ** 2))),
                "sim_feature_mae": float(np.mean(np.abs(delta_feat))),
                "sim_pred_corr_vs_ideal": corr(pred_sim, pred_ideal),
            }
        )

        for i, endpoint in enumerate(endpoints):
            row = {
                "layout": layout_json(layout),
                "pool_source": source,
                "endpoint": int(endpoint),
                "pred_ideal": float(pred_ideal[i]),
                "pred_noisy_sim": float(pred_sim[i]),
                "delta_pred": float(delta_pred[i]),
            }
            for j, f in enumerate(EXPECTED_FEATURES):
                row[f"{f}_ideal"] = float(X_ideal[i, j])
                row[f"{f}_sim"] = float(X_sim[i, j])
                row[f"{f}_delta"] = float(X_sim[i, j] - X_ideal[i, j])
            daily_rows.append(row)

    summary_df = pd.DataFrame(summaries).sort_values(
        "sim_pred_distortion_rmse_claims"
    ).reset_index(drop=True)
    summary_df["sim_rank"] = np.arange(len(summary_df)) + 1
    daily_df = pd.DataFrame(daily_rows)
    return summary_df, daily_df


# =============================================================================
# FINALIST LOGIC
# =============================================================================


def choose_finalists(sim_summary, n_finalists=4):
    """Choose finalists strictly by the training-only noisy-simulation ranking."""
    n_finalists = int(n_finalists)
    if n_finalists < 2:
        raise ValueError("Need at least 2 finalists.")

    finalists = []
    seen = set()
    for raw in sim_summary.sort_values(
        "sim_pred_distortion_rmse_claims", ascending=True
    )["layout"].tolist():
        layout = parse_layout(raw)
        key = layout_key(layout)
        if key in seen:
            continue
        finalists.append(list(key))
        seen.add(key)
        if len(finalists) == n_finalists:
            break

    if len(finalists) < n_finalists:
        raise RuntimeError(
            f"Only {len(finalists)} unique simulated layouts available for "
            f"{n_finalists} finalists."
        )
    return finalists


# =============================================================================
# QPU PROBE
# =============================================================================


def cyclic_layout_order(layouts, endpoint_index):
    n = len(layouts)
    shift = int(endpoint_index) % n
    order = list(layouts[shift:]) + list(layouts[:shift])
    # Reverse every complete Latin block to reduce any monotonic queue/drift
    # interaction while preserving balanced position counts.
    block = (int(endpoint_index) // n)
    if block % 2 == 1:
        order = list(reversed(order))
    return order


def compile_interleaved_probe(
    backend,
    candidate,
    frozen,
    endpoints,
    settings,
    finalists,
):
    circuits = []
    meta = []
    resources = []

    layout_ids = {layout_key(l): f"L{i+1}" for i, l in enumerate(finalists)}

    for i, endpoint in enumerate(endpoints):
        order = cyclic_layout_order(finalists, i)
        for layout_position, layout in enumerate(order, start=1):
            lid = layout_ids[layout_key(layout)]
            for setting in settings:
                logical = base.build_measurement_circuit(
                    candidate,
                    frozen["angles"],
                    int(endpoint),
                    setting,
                )
                isa = strict_compile_measurement_circuit(
                    backend,
                    logical,
                    layout,
                )
                circuits.append(isa)
                meta.append(
                    {
                        "endpoint": int(endpoint),
                        "layout_id": lid,
                        "layout": layout_json(layout),
                        "layout_position_within_endpoint": int(layout_position),
                        "setting": str(setting[0]),
                    }
                )
                resources.append(
                    {
                        **meta[-1],
                        "depth": int(isa.depth()),
                        "size": int(isa.size()),
                        "n_cz": int(isa.count_ops().get("cz", 0)),
                        "n_reset": int(isa.count_ops().get("reset", 0)),
                        "n_swap": int(isa.count_ops().get("swap", 0)),
                        "duration_us": float(
                            isa.estimate_duration(backend.target, unit="s")
                        ) * 1e6,
                    }
                )

    return circuits, meta, pd.DataFrame(resources)


def qpu_probe_metric_table(
    result,
    meta,
    finalists,
    endpoints,
    frozen,
    X_ideal,
    pred_ideal,
):
    by_layout = {layout_key(l): {int(e): {} for e in endpoints} for l in finalists}

    raw_rows = []
    for i, m in enumerate(meta):
        counts = get_sampler_pub_counts(result[i])
        key = layout_key(m["layout"])
        by_layout[key][int(m["endpoint"])][str(m["setting"])] = counts
        raw_rows.append(
            {
                **m,
                "counts_json": json.dumps({str(k): int(v) for k, v in counts.items()}),
            }
        )

    summaries = []
    daily_rows = []

    for layout in finalists:
        key = layout_key(layout)
        feat = features_from_setting_counts(by_layout[key])
        X_qpu, pred_qpu = features_to_predictions(frozen, endpoints, feat)
        d_pred = pred_qpu - pred_ideal
        d_feat = X_qpu - X_ideal

        summaries.append(
            {
                "layout": layout_json(layout),
                "qpu_pred_distortion_rmse_claims": float(np.sqrt(np.mean(d_pred ** 2))),
                "qpu_pred_distortion_mae_claims": float(np.mean(np.abs(d_pred))),
                "qpu_pred_distortion_bias_claims": float(np.mean(d_pred)),
                "qpu_feature_rmse": float(np.sqrt(np.mean(d_feat ** 2))),
                "qpu_feature_mae": float(np.mean(np.abs(d_feat))),
                "qpu_pred_corr_vs_ideal": corr(pred_qpu, pred_ideal),
            }
        )

        for i, endpoint in enumerate(endpoints):
            row = {
                "layout": layout_json(layout),
                "endpoint": int(endpoint),
                "pred_ideal": float(pred_ideal[i]),
                "pred_qpu": float(pred_qpu[i]),
                "delta_pred": float(d_pred[i]),
            }
            for j, f in enumerate(EXPECTED_FEATURES):
                row[f"{f}_ideal"] = float(X_ideal[i, j])
                row[f"{f}_qpu"] = float(X_qpu[i, j])
                row[f"{f}_delta"] = float(d_feat[i, j])
            daily_rows.append(row)

    summary_df = pd.DataFrame(summaries).sort_values(
        "qpu_pred_distortion_rmse_claims"
    ).reset_index(drop=True)
    summary_df["qpu_probe_rank"] = np.arange(len(summary_df)) + 1
    return summary_df, pd.DataFrame(daily_rows), pd.DataFrame(raw_rows)


# =============================================================================
# WINNER DESCRIPTION FOR AUTHORITATIVE 11_1A FULL RUN
# =============================================================================


def describe_winning_layout(backend, fresh, candidate, layout, probe_scores=None):
    layout = parse_layout(layout)
    row = {
        "backend": str(getattr(backend, "name", "ibm_kingston")),
        "layout": layout_json(layout),
        "physical_C_t": int(layout[0]),
        "physical_D": int(layout[1]),
        "physical_P_t": int(layout[2]),
        "physical_H": int(layout[3]),
        "physical_M1": int(layout[4]),
        "physical_M2": int(layout[5]),
        "selection_method": "qrc_aware_training_only_qpu_probe",
        "custom_rule5_selection_used": False,
        "strict_compile_pass": True,
        "compiled_n_swap": 0,
        "compile_error": "",
        "calibration_pareto": None,
        "overall_rank": None,
    }

    # Mapping/calibration metrics from existing selector helper if available.
    try:
        mm = selector.mapping_metrics(
            backend,
            str(candidate["topology"]),
            layout,
            fresh.get("qubits"),
        )
        if mm:
            row.update(dict(mm))
    except Exception:
        pass

    try:
        angles, _ = selector.first_input_angles()
        core = selector.build_core(candidate, angles)
        compiled = transpile(
            core,
            backend=backend,
            initial_layout=layout,
            routing_method="none",
            optimization_level=int(getattr(base, "OPT_LEVEL", OPT_LEVEL)),
            seed_transpiler=int(getattr(base, "SEED_TRANSPILE", SEED_TRANSPILE)),
            scheduling_method="alap",
        )
        backend.check_faulty(compiled)
        cm = selector.compiled_metrics(compiled, backend)
        if cm:
            row.update(dict(cm))
    except Exception as exc:
        row["winner_description_compile_warning"] = str(exc)

    # Fill percent aliases expected by the base DB writer if only fractional
    # fields exist.
    for frac, pct in [
        ("compiled_2q_error_max", "compiled_2q_error_max_percent"),
        ("compiled_1q_error_max", "compiled_1q_error_max_percent"),
        ("max_readout_error", "max_readout_error_percent"),
    ]:
        if pct not in row and frac in row and np.isfinite(safe_float(row[frac])):
            row[pct] = 100.0 * float(row[frac])

    if probe_scores:
        row.update({f"qrc_probe_{k}": v for k, v in probe_scores.items()})

    return row


def fixed_selector_factory(chosen_row, fresh):
    def _fixed_selector(
        service,
        candidate,
        backend_names,
        shortlist=120,
        write_prefix=None,
        verbose=True,
        **kwargs,
    ):
        if verbose:
            print("=" * 126)
            print("QRC-AWARE FROZEN PHYSICAL LAYOUT")
            print("=" * 126)
            print("NO Rule-5 reselection. NO Qiskit reselection.")
            print(f"Frozen winner: {chosen_row['layout']}")
            print(
                "Selection method: 100-point 2022-2024 training-only QPU "
                "hardware-distortion probe"
            )
        return {
            "selected": pd.Series(chosen_row),
            "compiled": pd.DataFrame([chosen_row]),
            "qubits": fresh.get("qubits"),
            "selection_method": "qrc_aware_training_only_qpu_probe",
        }
    return _fixed_selector


# =============================================================================
# BASE OUTPUT PRESERVATION / ALIASING
# =============================================================================


def base_output_path(suffix):
    return RESULTS / f"{BASE_OUTPUT_PREFIX}{suffix}"


def full_alias_path(suffix):
    return RESULTS / f"{FULL_ALIAS_PREFIX}{suffix}"


def copy_base_outputs_to_alias():
    for suffix in KNOWN_BASE_SUFFIXES:
        src = base_output_path(suffix)
        dst = full_alias_path(suffix)
        if src.exists():
            shutil.copy2(src, dst)


def run_full_candidate1_immediately(
    backend_name,
    full_shots,
    force_over_budget,
    chosen_row,
    fresh,
    probe_inputs,
    guard_backend=None,
    expected_calibration_utc=None,
):
    original_selector = base.selector.fresh_hardware_reselection
    original_create_run = base.qpu_db.create_run
    original_base_sampler = getattr(base, "SamplerV2", None)
    original_argv = list(sys.argv)
    original_file = base.__file__

    # Make DB metadata explicit.
    def create_run_qrc_aware(**kwargs):
        manifest = kwargs.get("manifest_json")
        if isinstance(manifest, dict):
            manifest = dict(manifest)
            manifest["physical_layout_policy"] = (
                "qrc_aware_training_only_qpu_probe_once_then_frozen"
            )
            manifest["qrc_aware_layout"] = parse_layout(chosen_row["layout"])
            manifest["qrc_aware_probe_inputs"] = int(probe_inputs)
            kwargs["manifest_json"] = manifest
        kwargs["notes"] = (
            "Week 11.1H.1 full Candidate-1 validation. Physical layout was "
            "selected immediately beforehand by a training-only QRC-aware "
            "hardware probe minimizing RMS QPU-vs-ideal prediction distortion. "
            "The exact winning layout is frozen; no Rule-5/Qiskit reselection. "
            "2026 is not loaded."
        )
        return original_create_run(**kwargs)

    # Guard the authoritative full-run Sampler job against recalibration while
    # it waits in IBM's queue.  We return a transparent job proxy whose result()
    # method performs the same calibration polling used for the layout probe.
    if original_base_sampler is not None and guard_backend is not None and expected_calibration_utc:
        class _GuardedJobProxy:
            def __init__(self, wrapped):
                self._wrapped = wrapped
            def __getattr__(self, name):
                return getattr(self._wrapped, name)
            def result(self, *args, **kwargs):
                # The base script normally calls result() with no special args.
                # If args/kwargs are supplied, preserve correctness by first
                # applying the guard and then returning the completed result.
                return guarded_job_result(
                    self._wrapped, guard_backend, expected_calibration_utc,
                    label="full C1",
                )

        class _GuardedSamplerV2:
            def __init__(self, *args, **kwargs):
                self._inner = original_base_sampler(*args, **kwargs)
            def __getattr__(self, name):
                return getattr(self._inner, name)
            def run(self, *args, **kwargs):
                return _GuardedJobProxy(self._inner.run(*args, **kwargs))

        base.SamplerV2 = _GuardedSamplerV2

    base.selector.fresh_hardware_reselection = fixed_selector_factory(chosen_row, fresh)
    base.qpu_db.create_run = create_run_qrc_aware
    base.__file__ = str(Path(__file__).resolve())

    with tempfile.TemporaryDirectory(prefix="qrc_11_1h1_full_") as td:
        td = Path(td)
        backups = {}
        for suffix in KNOWN_BASE_SUFFIXES:
            src = base_output_path(suffix)
            if src.exists():
                bak = td / src.name
                shutil.copy2(src, bak)
                backups[suffix] = bak

        try:
            argv = [
                str(Path(__file__).name),
                "--full",
                "--submit",
                "--backend",
                str(backend_name),
                "--shots",
                str(int(full_shots)),
            ]
            if force_over_budget:
                argv.append("--force-over-budget")
            sys.argv = argv

            print()
            print("=" * 140)
            print("IMMEDIATE FULL 365-DAY CANDIDATE #1 RUN ON QRC-AWARE WINNER")
            print("=" * 140)
            print(f"Frozen physical layout: {chosen_row['layout']}")
            print(f"Shots/setting: {full_shots}")
            print()

            base.main()
            copy_base_outputs_to_alias()

        finally:
            # Restore old authoritative 11_1A result files.
            for suffix in KNOWN_BASE_SUFFIXES:
                src = base_output_path(suffix)
                bak = backups.get(suffix)
                if bak is not None and bak.exists():
                    shutil.copy2(bak, src)
                elif src.exists():
                    try:
                        src.unlink()
                    except Exception:
                        pass

            sys.argv = original_argv
            base.__file__ = original_file
            base.selector.fresh_hardware_reselection = original_selector
            base.qpu_db.create_run = original_create_run
            if original_base_sampler is not None:
                base.SamplerV2 = original_base_sampler


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--probe-inputs", type=int, default=DEFAULT_PROBE_INPUTS)
    parser.add_argument("--probe-shots", type=int, default=DEFAULT_PROBE_SHOTS)
    parser.add_argument("--full-shots", type=int, default=DEFAULT_FULL_SHOTS)
    parser.add_argument("--sim-shots", type=int, default=DEFAULT_SIM_SHOTS)
    parser.add_argument("--sim-pool", type=int, default=DEFAULT_SIM_POOL)
    parser.add_argument(
        "--sim-inputs",
        type=int,
        default=DEFAULT_SIM_INPUTS,
        help=(
            "Training-only endpoints used by the LOCAL calibration-noise simulation. "
            "The real QPU probe still uses --probe-inputs (default 100)."
        ),
    )
    parser.add_argument("--finalists", type=int, default=DEFAULT_N_FINALISTS)
    parser.add_argument(
        "--selector-shortlist",
        type=int,
        default=DEFAULT_SELECTOR_SHORTLIST,
        help=(
            "Large value requests broad/all native H3 compilation from the "
            "existing selector. This is local and uses no QPU time."
        ),
    )
    parser.add_argument("--aer-batch-size", type=int, default=DEFAULT_AER_BATCH)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the real-QPU finalist probe (two layouts by default).",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Submit/score the QPU probe but do not launch the full 2025 run.",
    )
    parser.add_argument(
        "--force-over-budget",
        action="store_true",
        help=(
            "Forward the same intentional quota override used in the Week-11 "
            "scripts. Without it, the wrapper blocks when reported remaining "
            "usage is below the rough probe+full planning estimate."
        ),
    )
    args = parser.parse_args()

    if args.probe_inputs <= 0:
        raise ValueError("--probe-inputs must be positive.")
    if args.sim_inputs <= 0 or args.sim_inputs > args.probe_inputs:
        raise ValueError("--sim-inputs must be in [1, --probe-inputs].")
    if args.probe_shots <= 0 or args.full_shots <= 0 or args.sim_shots <= 0:
        raise ValueError("Shot counts must be positive.")
    if args.sim_pool < args.finalists:
        raise ValueError("--sim-pool must be >= --finalists.")
    if args.finalists < 1:
        raise ValueError("--finalists must be at least 1.")

    print("=" * 140)
    print("WEEK 11.1H.1 — CANDIDATE #1 QRC-AWARE PHYSICAL-LAYOUT SELECTION")
    print("=" * 140)
    print("STRICT BENCHMARK-BLIND SELECTION MODE.")
    print("Selection metric uses QPU/noisy-sim prediction minus IDEAL prediction.")
    print("True claim targets and prior 2025 QPU benchmark outcomes are not used.")
    print("No Qiskit-default or Rule-5 layout is forced into the pool/finalists.")
    print("Probe data are deterministic 2022-2024 training-only endpoints.")
    print("2026 remains FROZEN / UNUSED.")
    print()

    candidate, manifest_meta, frozen = load_candidate_and_frozen()
    settings = base.readout_settings()
    labels = [str(s[0]) for s in settings]
    if SETTING_X not in labels or SETTING_Z not in labels:
        raise RuntimeError(
            f"Unexpected Candidate-1 measurement settings: {labels}; expected "
            f"{SETTING_X} and {SETTING_Z}."
        )

    probe_endpoints, X_ideal, pred_ideal = choose_training_probe_endpoints(
        frozen,
        args.probe_inputs,
    )

    # The local noisy simulator is only a pre-filter.  Use an evenly spaced
    # deterministic subset of the 100 training-only QPU-probe endpoints so the
    # fresh-calibration phase finishes quickly.  The REAL QPU selector still
    # uses all --probe-inputs endpoints.
    sim_pick = np.unique(
        np.rint(np.linspace(0, len(probe_endpoints) - 1, int(args.sim_inputs))).astype(int)
    )
    sim_endpoints = np.asarray(probe_endpoints)[sim_pick]
    X_ideal_sim = np.asarray(X_ideal)[sim_pick]
    pred_ideal_sim = np.asarray(pred_ideal)[sim_pick]

    print("Frozen Candidate #1:")
    print(f"  topology={candidate['topology']} W={candidate['window']} r={candidate['r']}")
    print(f"  alpha={candidate['alpha']} dt={candidate['dt']}")
    print(f"  readout={getattr(base, 'EXPECTED_READOUT', 'XZinj_dropX3_plus_YX45')}")
    print(f"  QPU probe endpoints={len(probe_endpoints)} training-only")
    print(f"  LOCAL simulation endpoints={len(sim_endpoints)} training-only")
    print(f"  simulation shots/setting={args.sim_shots}")
    print(f"  QPU probe shots/setting={args.probe_shots}")
    print(f"  full validation shots/setting={args.full_shots}")
    print()

    pd.DataFrame({"endpoint": probe_endpoints}).to_csv(
        RESULTS / f"{PREFIX}_training_probe_endpoints.csv",
        index=False,
    )

    service = get_service()
    backend = service.backend(args.backend, use_fractional_gates=False)
    try:
        backend.refresh()
    except Exception:
        pass

    if hasattr(selector, "ALPHA"):
        selector.ALPHA = float(candidate["alpha"])
    if hasattr(selector, "DT"):
        selector.DT = float(candidate["dt"])

    calibration_before_local = current_calibration(backend)
    print("Calibration before local screening:")
    print(json.dumps(json_safe(calibration_before_local), indent=2))
    print()

    # ---------------------------------------------------------------------
    # A. Broad native/strict enumeration via existing selector
    # ---------------------------------------------------------------------
    print("=" * 140)
    print("STAGE A — LOCAL NATIVE H3 ENUMERATION + STRICT ZERO-SWAP SCREEN")
    print("=" * 140)
    fresh = selector.fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=[args.backend],
        shortlist=int(args.selector_shortlist),
        write_prefix=f"{PREFIX}_{args.backend}",
        verbose=True,
    )
    if "compiled" not in fresh:
        raise KeyError("Existing selector returned no fresh['compiled'] table.")

    compiled = fresh["compiled"].copy()
    if "strict_compile_pass" not in compiled.columns:
        raise KeyError("fresh['compiled'] lacks strict_compile_pass.")
    strict = compiled[bool_series(compiled["strict_compile_pass"])].copy()
    if len(strict) == 0:
        raise RuntimeError("No strict zero-SWAP H3 layouts survived.")

    print(f"Strict zero-SWAP layouts available: {len(strict)}")

    # IMPORTANT: fresh["selected"] is intentionally ignored.  We use the
    # broad strict-compile table only; the legacy Rule-5 winner does not enter
    # this selector.  Likewise, no Qiskit-default layout is computed here.
    representative_endpoint = int(probe_endpoints[len(probe_endpoints) // 2])
    print("Benchmark-blind mode: Rule-5/Qiskit historical layouts are NOT used.")
    print()

    # Speed-critical broad prefilter: fresh_hardware_reselection has already
    # compiled every strict layout.  Do NOT immediately recompile all of them a
    # second time.  When its calibration Pareto flag is available, restrict the
    # expensive actual-workload proxy to the non-dominated calibration front.
    # This is benchmark-blind and uses no historical 2025 outcomes.
    proxy_source = strict
    if "calibration_pareto" in strict.columns:
        pareto_mask = bool_series(strict["calibration_pareto"])
        pareto = strict[pareto_mask].copy()
        if len(pareto) >= int(args.sim_pool):
            proxy_source = pareto
            print(
                f"Fast mode: actual-workload proxy only on fresh calibration "
                f"Pareto layouts: {len(proxy_source)}/{len(strict)}"
            )
        else:
            print(
                "Fast mode: calibration Pareto front too small for requested "
                "simulation pool; using all strict layouts."
            )
    else:
        print("Fast mode: no calibration_pareto column; using all strict layouts.")

    proxy_table = actual_workload_proxy_table(
        backend=backend,
        candidate=candidate,
        frozen=frozen,
        layouts_df=proxy_source,
        representative_endpoint=representative_endpoint,
        settings=settings,
    )
    proxy_table = proxy_table[bool_series(proxy_table["proxy_compile_pass"])].copy()
    proxy_table = proxy_table.sort_values(
        "actual_workload_proxy_nll",
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)
    proxy_table["proxy_rank"] = np.arange(len(proxy_table)) + 1
    proxy_table.to_csv(
        RESULTS / f"{PREFIX}_all_strict_workload_proxy.csv",
        index=False,
    )

    sim_pool = top_unique_proxy_layouts(
        proxy_table,
        n_total=int(args.sim_pool),
    )

    sim_pool_df = pd.DataFrame(
        [
            {
                "simulation_pool_rank": i + 1,
                "pool_source": source,
                "layout": layout_json(layout),
            }
            for i, (source, layout) in enumerate(sim_pool)
        ]
    )
    sim_pool_df.to_csv(
        RESULTS / f"{PREFIX}_simulation_pool.csv",
        index=False,
    )

    print()
    print(f"Calibration-noise simulation pool: {len(sim_pool)} layouts")
    print(sim_pool_df.to_string(index=False))

    # ---------------------------------------------------------------------
    # B. Aer calibration-based noisy simulation
    # ---------------------------------------------------------------------
    print()
    print("=" * 140)
    print("STAGE B — LOCAL CALIBRATION-BASED NOISY QRC SIMULATION")
    print("=" * 140)
    sim_summary, sim_daily = noisy_simulation_stage(
        backend=backend,
        candidate=candidate,
        frozen=frozen,
        endpoints=sim_endpoints,
        X_ideal=X_ideal_sim,
        pred_ideal=pred_ideal_sim,
        settings=settings,
        pool=sim_pool,
        shots=int(args.sim_shots),
        batch_size=int(args.aer_batch_size),
    )
    sim_summary.to_csv(
        RESULTS / f"{PREFIX}_noisy_simulation_summary.csv",
        index=False,
    )
    sim_daily.to_csv(
        RESULTS / f"{PREFIX}_noisy_simulation_daily.csv",
        index=False,
    )

    print()
    print("NOISY-SIMULATION RANKING")
    print(
        sim_summary[
            [
                "sim_rank",
                "layout",
                "pool_source",
                "sim_pred_distortion_rmse_claims",
                "sim_feature_rmse",
                "sim_pred_corr_vs_ideal",
            ]
        ].to_string(index=False)
    )

    # Refresh immediately before the real-QPU stage, but do NOT recompute or
    # inject any Qiskit/Rule-5 benchmark layout.  The finalists were determined
    # only by the benchmark-blind training-only simulation stage above.
    try:
        backend.refresh()
    except Exception:
        pass
    calibration_before_probe = current_calibration(backend)

    # If a new calibration became active while the local selection was
    # running, do not silently submit a QPU probe from a stale shortlist.
    # In fast mode the correct response is simply to rerun; cached project data
    # remain untouched and no QPU time has been consumed.
    cal_local = (calibration_before_local or {}).get("last_update_date_utc") if isinstance(calibration_before_local, dict) else None
    cal_probe = (calibration_before_probe or {}).get("last_update_date_utc") if isinstance(calibration_before_probe, dict) else None
    calibration_changed_during_local = bool(cal_local and cal_probe and cal_local != cal_probe)
    if calibration_changed_during_local:
        print()
        print("WARNING: IBM calibration changed during local layout selection:")
        print(f"  local-start calibration: {cal_local}")
        print(f"  pre-probe calibration:   {cal_probe}")
        if args.submit:
            raise RuntimeError(
                "Calibration changed during local screening. No QPU job was submitted. "
                "Rerun the FAST script so the finalists are selected from the "
                "current calibration snapshot."
            )

    finalists = choose_finalists(
        sim_summary,
        n_finalists=int(args.finalists),
    )

    finalist_rows = []
    for i, layout in enumerate(finalists, start=1):
        match = sim_summary[
            sim_summary["layout"].apply(layout_key) == layout_key(layout)
        ]
        finalist_rows.append(
            {
                "finalist_id": f"L{i}",
                "layout": layout_json(layout),
                "selection_basis": "training_only_noisy_simulation",
                "sim_pred_distortion_rmse_claims": (
                    float(match.iloc[0]["sim_pred_distortion_rmse_claims"])
                    if len(match)
                    else np.nan
                ),
                "sim_rank": int(match.iloc[0]["sim_rank"]) if len(match) else np.nan,
            }
        )
    finalists_df = pd.DataFrame(finalist_rows)
    finalists_df.to_csv(
        RESULTS / f"{PREFIX}_qpu_finalists.csv",
        index=False,
    )

    print()
    print("QPU FINALISTS")
    print(finalists_df.to_string(index=False))
    print()
    print("Calibration immediately before QPU probe:")
    print(json.dumps(json_safe(calibration_before_probe), indent=2))

    local_summary = {
        "candidate": "RWP_H3_R1R3",
        "backend": args.backend,
        "selection_data_scope": "2022-2024 training-only probe endpoints",
        "targets_used_for_layout_ranking": False,
        "probe_inputs": int(len(probe_endpoints)),
        "local_sim_inputs": int(len(sim_endpoints)),
        "sim_shots_per_setting": int(args.sim_shots),
        "calibration_changed_during_local": bool(calibration_changed_during_local),
        "sim_pool_size": int(len(sim_pool)),
        "qpu_finalists": int(len(finalists)),
        "prior_2025_hardware_results_used_for_selection": False,
        "qiskit_layout_forced_into_pool_or_finalists": False,
        "rule5_layout_forced_into_pool_or_finalists": False,
        "finalist_selection_basis": "training_only_calibration_noise_simulation",
        "calibration_before_local": calibration_before_local,
        "calibration_before_probe": calibration_before_probe,
    }
    write_json(RESULTS / f"{PREFIX}_local_summary.json", local_summary)

    if not args.submit:
        print()
        print("=" * 140)
        print("LOCAL STAGES COMPLETE — NO QPU JOB SUBMITTED")
        print("=" * 140)
        print("Review the simulation/finalist CSVs. To execute the QPU probe and then")
        print("immediately run full Candidate #1 on the winner:")
        cmd = (
            f"python {Path(__file__).name} --submit --backend {args.backend} "
            f"--probe-inputs {args.probe_inputs} --probe-shots {args.probe_shots} "
            f"--full-shots {args.full_shots}"
        )
        print(cmd)
        print("2026 remains FROZEN / UNUSED.")
        return

    # ---------------------------------------------------------------------
    # Budget / quota planning guard
    # ---------------------------------------------------------------------
    probe_resources_preview_circuits = (
        len(probe_endpoints) * len(finalists) * len(settings)
    )
    probe_estimate = HISTORICAL_FULL_C1_QPU_CHARGE_S * (
        (probe_resources_preview_circuits * int(args.probe_shots))
        / (HISTORICAL_FULL_C1_N_CIRCUITS * HISTORICAL_FULL_C1_SHOTS)
    )
    total_estimate = probe_estimate + (
        0.0 if args.probe_only else HISTORICAL_FULL_C1_QPU_CHARGE_S
    )

    usage_before = get_service_usage(service)
    remaining = remaining_usage_seconds(usage_before)
    print()
    print("QPU PLANNING HEURISTIC")
    print(f"  probe circuits:                 {probe_resources_preview_circuits}")
    print(f"  rough probe QPU charge:         {probe_estimate:.1f} s")
    if not args.probe_only:
        print(f"  historical full C1 charge:      {HISTORICAL_FULL_C1_QPU_CHARGE_S:.1f} s")
    print(f"  rough probe+full requirement:   {total_estimate:.1f} s")
    if remaining is not None:
        print(f"  service reported remaining:     {remaining:.1f} s")

    if (
        remaining is not None
        and total_estimate > remaining
        and not args.force_over_budget
    ):
        raise RuntimeError(
            "Submission blocked before consuming QPU time: rough probe+full "
            f"estimate {total_estimate:.1f}s exceeds reported remaining "
            f"{remaining:.1f}s. If you intentionally accept that quota risk, "
            "rerun with --force-over-budget."
        )

    # ---------------------------------------------------------------------
    # C. Real-QPU interleaved layout probe
    # ---------------------------------------------------------------------
    print()
    print("=" * 140)
    print(f"STAGE C — REAL-QPU INTERLEAVED {len(finalists)}-LAYOUT TRAINING-ONLY PROBE")
    print("=" * 140)

    circuits, probe_meta, probe_resources = compile_interleaved_probe(
        backend=backend,
        candidate=candidate,
        frozen=frozen,
        endpoints=probe_endpoints,
        settings=settings,
        finalists=finalists,
    )
    probe_resources.to_csv(
        RESULTS / f"{PREFIX}_qpu_probe_resources.csv",
        index=False,
    )
    pd.DataFrame(probe_meta).to_csv(
        RESULTS / f"{PREFIX}_qpu_probe_circuit_order.csv",
        index=False,
    )

    if len(circuits) != probe_resources_preview_circuits:
        raise RuntimeError(
            f"Expected {probe_resources_preview_circuits} probe circuits, got {len(circuits)}."
        )

    print(f"Submitting {len(circuits)} circuits at {args.probe_shots} shots/setting...")
    sampler = SamplerV2(mode=backend)
    job = sampler.run(circuits, shots=int(args.probe_shots))
    print(f"Probe job ID: {job.job_id()}")
    expected_probe_calibration_utc = calibration_utc(calibration_before_probe)
    result = guarded_job_result(
        job=job,
        backend=backend,
        expected_calibration_utc=expected_probe_calibration_utc,
        label="QPU probe",
    )
    print(f"Probe final status: {job.status()}")

    metrics = None
    try:
        metrics = json_safe(job.metrics())
    except Exception:
        pass

    timing = None
    if hasattr(base, "extract_ibm_qpu_timing"):
        try:
            timing = base.extract_ibm_qpu_timing(job=job, metrics=metrics)
        except Exception:
            timing = None

    probe_running_utc = None
    if isinstance(timing, dict):
        probe_running_utc = timing.get("ibm_job_running_at_utc")
    if probe_running_utc is None and isinstance(metrics, dict):
        try:
            probe_running_utc = metrics["timestamps"]["running"]
        except Exception:
            pass

    calibration_probe_running = calibration_at(backend, probe_running_utc)

    qpu_summary, qpu_daily, qpu_raw = qpu_probe_metric_table(
        result=result,
        meta=probe_meta,
        finalists=finalists,
        endpoints=probe_endpoints,
        frozen=frozen,
        X_ideal=X_ideal,
        pred_ideal=pred_ideal,
    )
    qpu_summary.to_csv(
        RESULTS / f"{PREFIX}_qpu_probe_summary.csv",
        index=False,
    )
    qpu_daily.to_csv(
        RESULTS / f"{PREFIX}_qpu_probe_daily.csv",
        index=False,
    )
    qpu_raw.to_csv(
        RESULTS / f"{PREFIX}_qpu_probe_raw_counts.csv",
        index=False,
    )

    print()
    print("QPU LAYOUT-PROBE RANKING")
    print(
        qpu_summary[
            [
                "qpu_probe_rank",
                "layout",
                "qpu_pred_distortion_rmse_claims",
                "qpu_feature_rmse",
                "qpu_pred_corr_vs_ideal",
            ]
        ].to_string(index=False)
    )

    winner_layout = parse_layout(qpu_summary.iloc[0]["layout"])
    winner_score = float(qpu_summary.iloc[0]["qpu_pred_distortion_rmse_claims"])

    print()
    print("QRC-AWARE PHYSICAL WINNER")
    print(f"  layout={winner_layout}")
    print(f"  E_delta_y_QPU={winner_score:.6f} claims")
    print("  selection used no true claim targets")

    probe_job_payload = {
        "job_id": str(job.job_id()),
        "status": str(job.status()),
        "metrics": metrics,
        "timing": timing,
        "calibration_before_probe": calibration_before_probe,
        "calibration_at_probe_running": calibration_probe_running,
        "winner_layout": winner_layout,
        "winner_qpu_pred_distortion_rmse_claims": winner_score,
        "probe_inputs": int(len(probe_endpoints)),
        "probe_shots_per_setting": int(args.probe_shots),
        "n_probe_circuits": int(len(circuits)),
        "targets_used_for_layout_ranking": False,
    }
    write_json(RESULTS / f"{PREFIX}_qpu_probe_job.json", probe_job_payload)

    if args.probe_only:
        print()
        print("Probe-only mode requested. Full 2025 Candidate #1 run was NOT submitted.")
        return

    # ---------------------------------------------------------------------
    # HARD CALIBRATION GUARD BEFORE FULL 2025 SUBMISSION
    # ---------------------------------------------------------------------
    # 1) Exact historical snapshot at the probe's actual QPU RUNNING time must
    #    match the snapshot used to select/submit the probe.
    expected_probe_cal = calibration_utc(calibration_before_probe)
    actual_probe_cal = calibration_utc(calibration_probe_running)
    if expected_probe_cal and actual_probe_cal and expected_probe_cal != actual_probe_cal:
        raise RuntimeError(
            "Probe actually ran under a different IBM calibration snapshot than "
            "the one used for local layout selection. Full 2025 run NOT submitted. "
            f"Expected={expected_probe_cal}, actual probe-running={actual_probe_cal}."
        )

    # 2) Immediately before the expensive full run, the CURRENT reported
    #    calibration must still equal the probe-running calibration.
    calibration_before_full = current_calibration(backend)
    current_full_cal = calibration_utc(calibration_before_full)
    reference_cal = actual_probe_cal or expected_probe_cal
    print()
    print("Calibration guard immediately before full 2025 submission:")
    print(json.dumps(json_safe(calibration_before_full), indent=2))
    if reference_cal and current_full_cal and reference_cal != current_full_cal:
        raise RuntimeError(
            "IBM calibration changed after the QPU layout probe. Full 2025 run "
            "NOT submitted. Rerun layout selection/probe for the new snapshot. "
            f"Probe calibration={reference_cal}, current={current_full_cal}."
        )

    # ---------------------------------------------------------------------
    # D. Immediately freeze exact winner and launch full 2025 Candidate #1.
    # ---------------------------------------------------------------------
    winner_probe_scores = qpu_summary.iloc[0].to_dict()
    chosen_row = describe_winning_layout(
        backend=backend,
        fresh=fresh,
        candidate=candidate,
        layout=winner_layout,
        probe_scores=winner_probe_scores,
    )

    winner_record = {
        "layout": winner_layout,
        "qpu_probe_score_claims": winner_score,
        "hardware_selection": chosen_row,
        "probe_job_id": str(job.job_id()),
        "calibration_at_probe_running": calibration_probe_running,
    }
    write_json(RESULTS / f"{PREFIX}_winner.json", winner_record)

    # No intentional delay here: immediately enter authoritative full runner.
    run_full_candidate1_immediately(
        backend_name=args.backend,
        full_shots=int(args.full_shots),
        force_over_budget=bool(args.force_over_budget),
        chosen_row=chosen_row,
        fresh=fresh,
        probe_inputs=int(len(probe_endpoints)),
        guard_backend=backend,
        expected_calibration_utc=reference_cal,
    )

    # ---------------------------------------------------------------------
    # Calibration reconstruction for final run + combined summary.
    # ---------------------------------------------------------------------
    full_summary_path = full_alias_path("_summary.json")
    full_job_path = full_alias_path("_job.json")
    full_summary = None
    full_job = None
    if full_summary_path.exists():
        full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
    if full_job_path.exists():
        full_job = json.loads(full_job_path.read_text(encoding="utf-8"))

    full_running_utc = None
    for payload in (full_summary, full_job):
        if not isinstance(payload, dict):
            continue
        for key in (
            "ibm_job_running_at_utc",
            "qpu_running_at_utc",
        ):
            if payload.get(key):
                full_running_utc = payload[key]
                break
        if full_running_utc is not None:
            break
        try:
            full_running_utc = payload["qpu_timing"]["ibm_job_running_at_utc"]
        except Exception:
            pass
        if full_running_utc is not None:
            break

    calibration_full_running = calibration_at(backend, full_running_utc)

    probe_cal = None
    full_cal = None
    if isinstance(calibration_probe_running, dict):
        probe_cal = calibration_probe_running.get("last_update_date_utc")
    if isinstance(calibration_full_running, dict):
        full_cal = calibration_full_running.get("last_update_date_utc")

    combined = {
        "candidate": "RWP_H3_R1R3",
        "backend": args.backend,
        "winner_layout": winner_layout,
        "qpu_probe_score_claims": winner_score,
        "probe_job_id": str(job.job_id()),
        "probe_calibration": calibration_probe_running,
        "calibration_immediately_before_full_submission": calibration_before_full,
        "full_calibration": calibration_full_running,
        "probe_to_full_same_reported_calibration_snapshot": (
            bool(probe_cal == full_cal) if probe_cal and full_cal else None
        ),
        "full_summary_alias": str(full_summary_path),
        "full_job_alias": str(full_job_path),
        "full_qpu_rmse": (
            safe_float(full_summary.get("qpu_rmse"))
            if isinstance(full_summary, dict)
            else np.nan
        ),
        "full_qpu_feature_rmse": (
            safe_float(full_summary.get("qpu_feature_rmse"))
            if isinstance(full_summary, dict)
            else np.nan
        ),
        "full_prediction_correlation_qpu_vs_ideal": (
            safe_float(full_summary.get("prediction_correlation_qpu_vs_ideal"))
            if isinstance(full_summary, dict)
            else np.nan
        ),
        "selection_targets_used": False,
        "probe_data_scope": "2022-2024 training-only endpoints",
        "validation_scope": "full 2025 only after physical layout frozen",
        "test_scope": "2026 untouched",
    }
    write_json(RESULTS / f"{PREFIX}_combined_summary.json", combined)

    print()
    print("=" * 140)
    print("11.1H.1 COMPLETE")
    print("=" * 140)
    print(f"Winner layout: {winner_layout}")
    print(f"Probe hardware distortion: {winner_score:.6f} claims RMS")
    if isinstance(full_summary, dict) and "qpu_rmse" in full_summary:
        print(f"Full 2025 QPU RMSE: {safe_float(full_summary['qpu_rmse']):.6f}")
    print(
        "Probe -> full same reported calibration snapshot: "
        f"{combined['probe_to_full_same_reported_calibration_snapshot']}"
    )
    print("2026 remains FROZEN / UNUSED.")
    print()
    print("Key outputs:")
    for name in [
        f"{PREFIX}_all_strict_workload_proxy.csv",
        f"{PREFIX}_simulation_pool.csv",
        f"{PREFIX}_noisy_simulation_summary.csv",
        f"{PREFIX}_qpu_finalists.csv",
        f"{PREFIX}_qpu_probe_summary.csv",
        f"{PREFIX}_winner.json",
        f"{PREFIX}_combined_summary.json",
    ]:
        print(f"  results/{name}")
    print(f"  results/{FULL_ALIAS_PREFIX}_*")


if __name__ == "__main__":
    main()
