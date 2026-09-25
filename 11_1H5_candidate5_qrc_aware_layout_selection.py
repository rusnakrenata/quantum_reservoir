#!/usr/bin/env python
from __future__ import annotations

"""
WEEK 11.1H.5 — CANDIDATE #5 QRC-AWARE PHYSICAL-LAYOUT SELECTION
===============================================================

Candidate #5 keeps the frozen scientific model from Week 11.1E:
    original model      : CONT_H2_R2
    hardware surrogate  : RWP64
    topology            : H2
    r                   : 2
    readout             : XZinj = X0..X3 + Z0..Z3
    frozen Ridge lambda : 10
    full validation     : 2025 only
    2026                : NEVER LOADED

The full-CONT 2022-2024 scaler/Ridge is never retrained.

Selection pipeline
------------------
A. CURRENT-CALIBRATION NATIVE H2 SCREEN
   1) enumerate every native six-qubit H2 embedding;
   2) build a calibration Pareto front WITHOUT a weighted score:
         minimize max required CZ error,
         minimize max injection readout error,
         minimize max relevant 1Q error,
         maximize min T1,
         maximize min T2;
   3) T1/T2 are Pareto descriptors/diagnostics, NOT mandatory hard cutoffs.

B. ACTUAL RWP64 WORKLOAD PROXY
   On the Pareto layouts, compile the REAL Candidate-5 RWP64 XXXX and ZZZZ
   circuits for a deterministic 2022-2024 training endpoint and evaluate

       ProxyNLL(L)
         = sum_g -log(1-e_g)
           + sum_measured -log(1-e_readout).

   Repeated gates appear repeatedly, so the real 64-step workload exposure is
   represented. Keep the best 12 unique layouts by default.

C. LOCAL CALIBRATION-NOISE AER
   Simulate the 12 layouts on 20 deterministic 2022-2024 training endpoints,
   64 shots/setting by default. Rank by target-free prediction distortion

       E_delta_y_sim(L)
         = RMS[ yhat_noisy_sim(L) - yhat_ideal_RWP64 ].

   True claim targets are NOT used.

D. REAL-QPU TRAINING-ONLY LAYOUT PROBE
   Send the best 2 simulated layouts to ibm_kingston using 100 deterministic
   2022-2024 training endpoints and 256 shots/setting by default.
   Rank by

       E_delta_y_QPU(L)
         = RMS[ yhat_QPU(L) - yhat_ideal_RWP64 ].

   Again, true claim targets are NOT used.

E. IMMEDIATE FULL 2025
   Freeze the QPU-probe winner and call the authoritative
   11_1E7_candidate5_RWP64_full_365_qpu.py engine at 1024 shots/setting.
   Its original memory-aware selector is bypassed. The winning layout is the
   ONLY physical layout used in the full run and written to the database.

Important
---------
- No historical 2025 QPU result is used to select the layout.
- No historical Qiskit layout is forced into the pool.
- No old memory-aware winner is forced into the pool.
- t/T1 and t/T2 are saved as diagnostics, not used as hard rejection rules.
- 2026 remains frozen.
"""

import argparse
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2

from ibm_account import get_service


# =============================================================================
# Paths / constants
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

BASE_FILE = HERE / "11_1E7_candidate5_RWP64_full_365_qpu.py"

PREFIX = "11_1H5_candidate5_qrc_aware_layout_selection"
FULL_ALIAS_PREFIX = f"{PREFIX}_full"

DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_SIM_POOL = 12
DEFAULT_SIM_INPUTS = 20
DEFAULT_SIM_SHOTS = 64
DEFAULT_FINALISTS = 2
DEFAULT_PROBE_INPUTS = 100
DEFAULT_PROBE_SHOTS = 256
DEFAULT_FULL_SHOTS = 1024
DEFAULT_AER_BATCH = 20

OPT_LEVEL = 1
SEED_TRANSPILE = 42
SEED_SIMULATOR = 424242
LOCAL_TZ = ZoneInfo("Europe/Bratislava")
CALIBRATION_POLL_SECONDS = 5

EXPECTED_FEATURES = [
    "X0", "X1", "X2", "X3",
    "Z0", "Z1", "Z2", "Z3",
]
SETTING_X = "XXXX"
SETTING_Z = "ZZZZ"

# Historical observed full Candidate-5 QPU charge; planning only.
HISTORICAL_FULL_C5_QPU_CHARGE_S = 374.0
HISTORICAL_FULL_C5_N_CIRCUITS = 730
HISTORICAL_FULL_C5_SHOTS = 1024

# Authoritative E7 files that must be preserved and copied to an H5 alias.
BASE_OUTPUT_PATH_ATTRS = [
    "OUT_ALL_EMBEDDINGS",
    "OUT_TEST_LAYOUTS",
    "OUT_TEST_COMPILED",
    "OUT_TEST_QUBITS",
    "OUT_TEST_SUMMARY",
    "OUT_RESOURCES",
    "OUT_ENDPOINTS",
    "OUT_PREFLIGHT",
    "OUT_JOB",
    "OUT_RAW_COUNTS",
    "OUT_FEATURES",
    "OUT_SUMMARY",
]


# =============================================================================
# Module loading
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Put this script beside 11_1E7_candidate5_RWP64_full_365_qpu.py."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = load_module(BASE_FILE, "qrc_11_1h5_base_candidate5")
common = base.common
layoutmod = base.layoutmod
refmod = base.refmod


# =============================================================================
# Generic helpers
# =============================================================================

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def json_safe(x):
    try:
        return base.db.json_safe(x)
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
    path.write_text(
        json.dumps(json_safe(payload), indent=2),
        encoding="utf-8",
    )


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


def layout_key(layout):
    if isinstance(layout, str):
        layout = json.loads(layout)
    return tuple(int(v) for v in layout)


def layout_json(layout):
    return json.dumps([int(v) for v in layout_key(layout)])


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
            "UTC is used for programmatic comparison; Europe/Bratislava is "
            "human-readable only."
        ),
    }


def current_calibration(backend):
    try:
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


def calibration_at(backend, running_utc):
    if running_utc is None:
        return None
    dt = ensure_utc(running_utc)
    try:
        props = backend.properties(datetime=dt)
        out = calibration_record(props)
        out["query_utc"] = dt.isoformat()
        return out
    except Exception as exc:
        return {"query_utc": dt.isoformat(), "error": str(exc)}


def job_status_name(job):
    try:
        status = job.status()
    except Exception:
        return "UNKNOWN"
    name = getattr(status, "name", None)
    if name:
        return str(name).upper()
    text = str(status).upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def guarded_job_result(job, backend, expected_calibration_utc, label):
    terminal = {"DONE", "CANCELLED", "CANCELED", "ERROR", "FAILED"}
    last_status = None

    while True:
        status = job_status_name(job)
        if status != last_status:
            print(f"[{label}] job status: {status}")
            last_status = status

        if status in terminal:
            break

        now = current_calibration(backend)
        now_utc = calibration_utc(now)
        if (
            expected_calibration_utc
            and now_utc
            and now_utc != expected_calibration_utc
        ):
            try:
                job.cancel()
            except Exception:
                pass
            raise RuntimeError(
                f"{label}: IBM reported calibration changed after layout "
                "selection. Job cancelled if still cancellable."
            )

        time.sleep(CALIBRATION_POLL_SECONDS)

    return job.result()


def get_service_usage(service):
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
# Frozen Candidate #5 / ideal training-only RWP64 reference
# =============================================================================

def load_candidate_and_frozen():
    package = base.load_json(base.FROZEN_PACKAGE)
    candidate, manifest_meta = base.candmod.load_candidate()

    if candidate["candidate_id"] != base.EXPECTED_CANDIDATE:
        raise RuntimeError("Unexpected Candidate #5 key.")
    if candidate["topology"] != base.EXPECTED_TOPOLOGY:
        raise RuntimeError("Unexpected Candidate #5 topology.")
    if int(candidate["r"]) != int(base.EXPECTED_R):
        raise RuntimeError("Unexpected Candidate #5 r.")
    if int(package["operational_K"]) != int(base.EXPECTED_K):
        raise RuntimeError("Unexpected Candidate #5 RWP K.")
    if package["readout"] != base.EXPECTED_READOUT:
        raise RuntimeError("Unexpected Candidate #5 readout.")

    frozen = base.build_frozen_reference(candidate, package)

    # build_full_cont_bank returns the transition bank used by replay_feature.
    # This lets us obtain IDEAL RWP64 features on 2022-2024 training endpoints
    # without retraining the frozen full-CONT Ridge.
    _, A_list = refmod.build_full_cont_bank(
        candidate,
        frozen["angles"],
    )
    frozen["A_list"] = A_list

    return candidate, manifest_meta, package, frozen


def choose_training_endpoints(frozen, n_inputs):
    start = int(base.EXPECTED_K - 1)
    stop = int(common.N_TRAIN - 1)
    if stop < start:
        raise RuntimeError("Training range is shorter than RWP64.")

    all_train = np.arange(start, stop + 1, dtype=int)
    n = int(max(1, min(int(n_inputs), len(all_train))))
    pick = np.unique(np.linspace(0, len(all_train) - 1, n, dtype=int))
    endpoints = all_train[pick]

    X_ideal = np.vstack(
        [
            refmod.replay_feature(
                frozen["A_list"],
                endpoint=int(e),
                K=int(base.EXPECTED_K),
            )
            for e in endpoints
        ]
    ).astype(float)

    pred_ideal = common.predict_scaled_ridge(
        frozen["model"],
        frozen["scaler"],
        frozen["keep"],
        X_ideal,
    )

    return (
        np.asarray(endpoints, dtype=int),
        X_ideal,
        np.asarray(pred_ideal, dtype=float),
    )


# =============================================================================
# Counts -> Candidate #5 XZinj feature vectors
# =============================================================================

def get_pub_counts(pub_result):
    return base.get_pub_counts(pub_result)


def expectation_from_counts(counts, classical_bits):
    return base.expectation_from_counts(counts, classical_bits)


def features_from_setting_counts(per_endpoint):
    out = {}
    for endpoint, setting_map in per_endpoint.items():
        if SETTING_X not in setting_map or SETTING_Z not in setting_map:
            raise RuntimeError(
                f"Endpoint {endpoint}: missing XXXX or ZZZZ."
            )

        cx = setting_map[SETTING_X]
        cz = setting_map[SETTING_Z]

        out[int(endpoint)] = np.asarray(
            [
                expectation_from_counts(cx, [0]),
                expectation_from_counts(cx, [1]),
                expectation_from_counts(cx, [2]),
                expectation_from_counts(cx, [3]),
                expectation_from_counts(cz, [0]),
                expectation_from_counts(cz, [1]),
                expectation_from_counts(cz, [2]),
                expectation_from_counts(cz, [3]),
            ],
            dtype=float,
        )
    return out


def features_to_predictions(frozen, endpoints, features_by_endpoint):
    X = np.vstack(
        [features_by_endpoint[int(e)] for e in endpoints]
    ).astype(float)

    pred = common.predict_scaled_ridge(
        frozen["model"],
        frozen["scaler"],
        frozen["keep"],
        X,
    )
    return X, np.asarray(pred, dtype=float)


# =============================================================================
# Static native-H2 calibration Pareto
# =============================================================================

def target_instruction_error(backend, opname, qargs):
    try:
        ip = backend.target[str(opname)][tuple(int(q) for q in qargs)]
    except Exception:
        return np.nan
    if ip is None:
        return np.nan
    return safe_float(getattr(ip, "error", np.nan))


def edge_cz_error(backend, a, b):
    vals = []
    for qargs in ((int(a), int(b)), (int(b), int(a))):
        e = target_instruction_error(backend, "cz", qargs)
        if np.isfinite(e):
            vals.append(float(e))
    return min(vals) if vals else np.nan


def qubit_1q_error(backend, q):
    vals = []
    for op in ("sx", "x"):
        e = target_instruction_error(backend, op, (int(q),))
        if np.isfinite(e):
            vals.append(float(e))
    return max(vals) if vals else np.nan


def static_layout_row(backend, layout, logical_edges):
    layout = [int(x) for x in layout]
    props = backend.properties()

    cz_errors = []
    for i, j in logical_edges:
        e = edge_cz_error(
            backend,
            layout[int(i)],
            layout[int(j)],
        )
        if np.isfinite(e):
            cz_errors.append(e)

    readout = []
    # XZinj measures injection q0..q3 only.
    for logical_q in range(4):
        try:
            readout.append(
                float(props.readout_error(layout[logical_q]))
            )
        except Exception:
            pass

    oneq = [
        qubit_1q_error(backend, q)
        for q in layout
    ]
    oneq = [x for x in oneq if np.isfinite(x)]

    t1 = []
    t2 = []
    for q in layout:
        try:
            t1.append(float(props.t1(q)) * 1e6)
        except Exception:
            pass
        try:
            t2.append(float(props.t2(q)) * 1e6)
        except Exception:
            pass

    return {
        "layout": layout_json(layout),
        "max_required_cz_error": max(cz_errors) if cz_errors else np.nan,
        "max_required_cz_error_percent": (
            100.0 * max(cz_errors) if cz_errors else np.nan
        ),
        "max_injection_readout_error": max(readout) if readout else np.nan,
        "max_injection_readout_error_percent": (
            100.0 * max(readout) if readout else np.nan
        ),
        "max_1q_error": max(oneq) if oneq else np.nan,
        "max_1q_error_percent": 100.0 * max(oneq) if oneq else np.nan,
        "min_t1_us": min(t1) if t1 else np.nan,
        "min_t2_us": min(t2) if t2 else np.nan,
        "memory_q4_physical": int(layout[4]),
        "memory_q5_physical": int(layout[5]),
    }


def pareto_mask(df):
    """
    Non-dominated current-calibration front.
    No weighted score.

    Minimize:
      max_required_cz_error
      max_injection_readout_error
      max_1q_error

    Maximize:
      min_t1_us
      min_t2_us
    """
    cols = [
        "max_required_cz_error",
        "max_injection_readout_error",
        "max_1q_error",
        "min_t1_us",
        "min_t2_us",
    ]

    X = df[cols].to_numpy(dtype=float)

    # Convert maximization objectives to minimization.
    X[:, 3] *= -1.0
    X[:, 4] *= -1.0

    finite = np.all(np.isfinite(X), axis=1)
    keep = np.zeros(len(df), dtype=bool)

    finite_idx = np.where(finite)[0]
    for ii, i in enumerate(finite_idx):
        xi = X[i]
        dominated = False
        for j in finite_idx:
            if i == j:
                continue
            xj = X[j]
            if np.all(xj <= xi) and np.any(xj < xi):
                dominated = True
                break
        keep[i] = not dominated

    return keep


def enumerate_native_h2_pareto(backend):
    (
        adjacency,
        eligible,
        qubit_cal,
    ) = layoutmod.current_physical_graph(backend)

    logical_edges = [
        (int(a), int(b))
        for a, b in common.TOPOLOGY_EDGES[base.EXPECTED_TOPOLOGY]
    ]

    layouts = layoutmod.enumerate_native_embeddings(
        logical_edges,
        adjacency,
        eligible,
    )

    rows = [
        static_layout_row(
            backend,
            layout,
            logical_edges,
        )
        for layout in layouts
    ]

    table = pd.DataFrame(rows)
    table["calibration_pareto"] = pareto_mask(table)
    table["native_rank_id"] = np.arange(len(table)) + 1

    return table, qubit_cal


# =============================================================================
# ACTUAL RWP64 workload proxy
# =============================================================================

def physical_qargs(circuit, inst):
    return [
        int(circuit.find_bit(q).index)
        for q in inst.qubits
    ]


def calibration_nll_for_circuit(circuit, backend):
    props = backend.properties()

    gate_nll = 0.0
    readout_nll = 0.0
    missing = 0
    measured = []

    for inst in circuit.data:
        name = str(inst.operation.name)
        qargs = physical_qargs(circuit, inst)

        if name == "measure":
            if len(qargs) == 1:
                measured.append(int(qargs[0]))
            continue

        if name in {"delay", "barrier"}:
            continue

        e = target_instruction_error(
            backend,
            name,
            qargs,
        )

        if np.isfinite(e):
            e = min(max(float(e), 0.0), 1.0 - 1e-15)
            gate_nll += -math.log1p(-e)
        else:
            missing += 1

    for q in measured:
        try:
            e = float(props.readout_error(q))
        except Exception:
            continue

        if np.isfinite(e):
            e = min(max(e, 0.0), 1.0 - 1e-15)
            readout_nll += -math.log1p(-e)

    ops = circuit.count_ops()
    return {
        "gate_nll": float(gate_nll),
        "readout_nll": float(readout_nll),
        "total_nll": float(gate_nll + readout_nll),
        "proxy_success_probability": float(
            math.exp(-(gate_nll + readout_nll))
        ),
        "missing_gate_errors": int(missing),
        "duration_us": float(
            circuit.estimate_duration(
                backend.target,
                unit="s",
            )
        ) * 1e6,
        "depth": int(circuit.depth()),
        "n_cz": int(ops.get("cz", 0)),
        "n_reset": int(ops.get("reset", 0)),
        "n_swap": int(ops.get("swap", 0)),
    }


def strict_compile(backend, logical, layout):
    isa = transpile(
        logical,
        backend=backend,
        initial_layout=[int(x) for x in layout_key(layout)],
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
        scheduling_method="alap",
    )
    backend.check_faulty(isa)

    if int(isa.count_ops().get("swap", 0)) != 0:
        raise RuntimeError(
            f"SWAP detected for layout {layout}."
        )

    return isa


def actual_rwp64_proxy_table(
    backend,
    candidate,
    frozen,
    layouts_df,
    representative_endpoint,
    settings,
):
    logical_by_setting = {
        str(s[0]): layoutmod.build_rwp64_circuit(
            candidate,
            frozen["angles"],
            int(representative_endpoint),
            s,
        )
        for s in settings
    }

    rows = []
    total = len(layouts_df)

    for k, (_, src) in enumerate(layouts_df.iterrows(), start=1):
        layout = list(layout_key(src["layout"]))

        rec = {
            "layout": layout_json(layout),
            "proxy_compile_pass": True,
            "proxy_compile_error": "",
        }

        try:
            total_nll = 0.0
            total_duration = 0.0
            total_cz = 0
            total_reset = 0
            missing = 0
            max_depth = 0
            max_setting_duration = 0.0

            for label, logical in logical_by_setting.items():
                isa = strict_compile(
                    backend,
                    logical,
                    layout,
                )
                m = calibration_nll_for_circuit(
                    isa,
                    backend,
                )

                rec[f"{label}_nll"] = m["total_nll"]
                rec[f"{label}_duration_us"] = m["duration_us"]
                rec[f"{label}_depth"] = m["depth"]

                total_nll += m["total_nll"]
                total_duration += m["duration_us"]
                total_cz += m["n_cz"]
                total_reset += m["n_reset"]
                missing += m["missing_gate_errors"]
                max_depth = max(max_depth, m["depth"])
                max_setting_duration = max(
                    max_setting_duration,
                    m["duration_us"],
                )

            # Memory coherence diagnostics only.
            props = backend.properties()
            p4, p5 = int(layout[4]), int(layout[5])

            def t1_us(q):
                try:
                    return float(props.t1(q)) * 1e6
                except Exception:
                    return np.nan

            def t2_us(q):
                try:
                    return float(props.t2(q)) * 1e6
                except Exception:
                    return np.nan

            mem_t1 = [t1_us(p4), t1_us(p5)]
            mem_t2 = [t2_us(p4), t2_us(p5)]

            ratios_t1 = [
                max_setting_duration / v
                if np.isfinite(v) and v > 0
                else np.nan
                for v in mem_t1
            ]
            ratios_t2 = [
                max_setting_duration / v
                if np.isfinite(v) and v > 0
                else np.nan
                for v in mem_t2
            ]

            rec.update(
                {
                    "actual_rwp64_proxy_nll": float(total_nll),
                    "actual_rwp64_proxy_success_probability": float(
                        math.exp(-total_nll)
                    ),
                    "actual_rwp64_duration_us": float(total_duration),
                    "actual_rwp64_max_setting_duration_us": float(
                        max_setting_duration
                    ),
                    "actual_rwp64_max_setting_depth": int(max_depth),
                    "actual_rwp64_n_cz": int(total_cz),
                    "actual_rwp64_n_reset": int(total_reset),
                    "actual_rwp64_missing_gate_errors": int(missing),
                    "memory_q4_t_over_t1": float(ratios_t1[0]),
                    "memory_q4_t_over_t2": float(ratios_t2[0]),
                    "memory_q5_t_over_t1": float(ratios_t1[1]),
                    "memory_q5_t_over_t2": float(ratios_t2[1]),
                    "memory_worst_t_over_t1": float(
                        np.nanmax(ratios_t1)
                    ),
                    "memory_worst_t_over_t2": float(
                        np.nanmax(ratios_t2)
                    ),
                }
            )

        except Exception as exc:
            rec["proxy_compile_pass"] = False
            rec["proxy_compile_error"] = str(exc)
            rec["actual_rwp64_proxy_nll"] = np.inf

        rows.append(rec)

        if k == 1 or k % 25 == 0 or k == total:
            print(
                f"  actual RWP64 workload proxy: {k}/{total} layouts"
            )

    proxy = pd.DataFrame(rows)

    left = layouts_df.copy()
    left["_key"] = left["layout"].apply(
        lambda x: layout_json(layout_key(x))
    )
    proxy["_key"] = proxy["layout"]

    out = left.merge(
        proxy.drop(columns=["layout"]),
        on="_key",
        how="left",
    )

    out["layout"] = out["_key"]
    return out.drop(columns=["_key"])


def top_unique_proxy_layouts(proxy_table, n_total):
    chosen = []
    seen = set()

    for _, row in proxy_table.sort_values(
        "actual_rwp64_proxy_nll",
        ascending=True,
    ).iterrows():
        key = layout_key(row["layout"])
        if key in seen:
            continue
        chosen.append(("proxy", list(key)))
        seen.add(key)

        if len(chosen) >= int(n_total):
            break

    if len(chosen) < int(n_total):
        raise RuntimeError(
            f"Only {len(chosen)} unique RWP64 proxy layouts available; "
            f"requested {n_total}."
        )

    return chosen


# =============================================================================
# Aer calibration-noise stage
# =============================================================================

def build_aer_from_backend(backend):
    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise RuntimeError(
            "qiskit-aer is required for Candidate #5 local simulation."
        ) from exc

    try:
        sim = AerSimulator.from_backend(
            backend,
            method="matrix_product_state",
        )
    except TypeError:
        sim = AerSimulator.from_backend(backend)
        try:
            sim.set_options(
                method="matrix_product_state"
            )
        except Exception:
            pass

    try:
        sim.set_options(
            max_parallel_threads=0,
            max_parallel_experiments=0,
            enable_truncation=True,
            mps_omp_threads=1,
            mps_sample_measure_algorithm="mps_probabilities",
        )
    except Exception as exc:
        print(
            f"WARNING: could not enable all Aer fast options: {exc}"
        )

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
            logical = layoutmod.build_rwp64_circuit(
                candidate,
                frozen["angles"],
                int(endpoint),
                setting,
            )
            isa = strict_compile(
                backend,
                logical,
                layout,
            )
            circuits.append(isa)
            meta.append(
                (int(endpoint), str(setting[0]))
            )

    counts_by_endpoint = {
        int(e): {} for e in endpoints
    }

    for start in range(
        0,
        len(circuits),
        int(batch_size),
    ):
        stop = min(
            start + int(batch_size),
            len(circuits),
        )

        result = simulator.run(
            circuits[start:stop],
            shots=int(shots),
            seed_simulator=int(
                SEED_SIMULATOR + start
            ),
        ).result()

        for j in range(stop - start):
            endpoint, setting = meta[start + j]
            counts_by_endpoint[endpoint][setting] = (
                result.get_counts(j)
            )

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

    for rank, (source, layout) in enumerate(
        pool,
        start=1,
    ):
        print()
        print(
            f"[Aer] layout {rank}/{len(pool)} "
            f"source={source} layout={layout} shots={shots}"
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

        feat = features_from_setting_counts(
            counts
        )
        X_sim, pred_sim = features_to_predictions(
            frozen,
            endpoints,
            feat,
        )

        d_pred = pred_sim - pred_ideal
        d_feat = X_sim - X_ideal

        summaries.append(
            {
                "layout": layout_json(layout),
                "pool_source": source,
                "sim_shots_per_setting": int(shots),
                "sim_pred_distortion_rmse_claims": rmse(
                    pred_sim,
                    pred_ideal,
                ),
                "sim_pred_distortion_mae_claims": mae(
                    pred_sim,
                    pred_ideal,
                ),
                "sim_pred_distortion_bias_claims": float(
                    np.mean(d_pred)
                ),
                "sim_feature_rmse": float(
                    np.sqrt(np.mean(d_feat ** 2))
                ),
                "sim_feature_mae": float(
                    np.mean(np.abs(d_feat))
                ),
                "sim_pred_corr_vs_ideal": corr(
                    pred_sim,
                    pred_ideal,
                ),
            }
        )

        for i, endpoint in enumerate(endpoints):
            row = {
                "layout": layout_json(layout),
                "endpoint": int(endpoint),
                "pred_ideal_RWP64": float(
                    pred_ideal[i]
                ),
                "pred_noisy_sim": float(
                    pred_sim[i]
                ),
                "delta_pred": float(
                    d_pred[i]
                ),
            }

            for j, f in enumerate(
                EXPECTED_FEATURES
            ):
                row[f"{f}_ideal_RWP64"] = float(
                    X_ideal[i, j]
                )
                row[f"{f}_sim"] = float(
                    X_sim[i, j]
                )
                row[f"{f}_delta"] = float(
                    d_feat[i, j]
                )

            daily_rows.append(row)

    summary_df = pd.DataFrame(
        summaries
    ).sort_values(
        "sim_pred_distortion_rmse_claims"
    ).reset_index(drop=True)

    summary_df["sim_rank"] = (
        np.arange(len(summary_df)) + 1
    )

    return summary_df, pd.DataFrame(daily_rows)


def choose_finalists(sim_summary, n_finalists):
    finalists = []
    seen = set()

    for raw in sim_summary.sort_values(
        "sim_pred_distortion_rmse_claims"
    )["layout"].tolist():
        key = layout_key(raw)
        if key in seen:
            continue

        finalists.append(list(key))
        seen.add(key)

        if len(finalists) >= int(n_finalists):
            break

    if len(finalists) < int(n_finalists):
        raise RuntimeError(
            f"Only {len(finalists)} unique finalists available."
        )

    return finalists


# =============================================================================
# Real-QPU training-only probe
# =============================================================================

def cyclic_layout_order(layouts, endpoint_index):
    n = len(layouts)
    shift = int(endpoint_index) % n
    order = list(layouts[shift:]) + list(layouts[:shift])

    block = int(endpoint_index) // n
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

    layout_ids = {
        layout_key(l): f"L{i+1}"
        for i, l in enumerate(finalists)
    }

    for i, endpoint in enumerate(endpoints):
        order = cyclic_layout_order(
            finalists,
            i,
        )

        for position, layout in enumerate(
            order,
            start=1,
        ):
            lid = layout_ids[layout_key(layout)]

            for setting in settings:
                logical = layoutmod.build_rwp64_circuit(
                    candidate,
                    frozen["angles"],
                    int(endpoint),
                    setting,
                )

                isa = strict_compile(
                    backend,
                    logical,
                    layout,
                )

                rec = {
                    "endpoint": int(endpoint),
                    "layout_id": lid,
                    "layout": layout_json(layout),
                    "layout_position_within_endpoint": int(
                        position
                    ),
                    "setting": str(setting[0]),
                }

                circuits.append(isa)
                meta.append(rec)

                resources.append(
                    {
                        **rec,
                        "depth": int(isa.depth()),
                        "size": int(isa.size()),
                        "n_cz": int(
                            isa.count_ops().get(
                                "cz",
                                0,
                            )
                        ),
                        "n_reset": int(
                            isa.count_ops().get(
                                "reset",
                                0,
                            )
                        ),
                        "n_swap": int(
                            isa.count_ops().get(
                                "swap",
                                0,
                            )
                        ),
                        "duration_us": float(
                            isa.estimate_duration(
                                backend.target,
                                unit="s",
                            )
                        )
                        * 1e6,
                    }
                )

    return (
        circuits,
        meta,
        pd.DataFrame(resources),
    )


def qpu_probe_metric_table(
    result,
    meta,
    finalists,
    endpoints,
    frozen,
    X_ideal,
    pred_ideal,
):
    by_layout = {
        layout_key(l): {
            int(e): {} for e in endpoints
        }
        for l in finalists
    }

    raw_rows = []

    for i, m in enumerate(meta):
        counts = get_pub_counts(result[i])
        key = layout_key(m["layout"])

        by_layout[key][
            int(m["endpoint"])
        ][
            str(m["setting"])
        ] = counts

        raw_rows.append(
            {
                **m,
                "counts_json": json.dumps(
                    {
                        str(k): int(v)
                        for k, v in counts.items()
                    },
                    sort_keys=True,
                ),
            }
        )

    summaries = []
    daily_rows = []

    for layout in finalists:
        key = layout_key(layout)

        feat = features_from_setting_counts(
            by_layout[key]
        )

        X_qpu, pred_qpu = features_to_predictions(
            frozen,
            endpoints,
            feat,
        )

        d_pred = pred_qpu - pred_ideal
        d_feat = X_qpu - X_ideal

        summaries.append(
            {
                "layout": layout_json(layout),
                "qpu_pred_distortion_rmse_claims": rmse(
                    pred_qpu,
                    pred_ideal,
                ),
                "qpu_pred_distortion_mae_claims": mae(
                    pred_qpu,
                    pred_ideal,
                ),
                "qpu_pred_distortion_bias_claims": float(
                    np.mean(d_pred)
                ),
                "qpu_feature_rmse": float(
                    np.sqrt(np.mean(d_feat ** 2))
                ),
                "qpu_feature_mae": float(
                    np.mean(np.abs(d_feat))
                ),
                "qpu_pred_corr_vs_ideal": corr(
                    pred_qpu,
                    pred_ideal,
                ),
            }
        )

        for i, endpoint in enumerate(endpoints):
            row = {
                "layout": layout_json(layout),
                "endpoint": int(endpoint),
                "pred_ideal_RWP64": float(
                    pred_ideal[i]
                ),
                "pred_qpu": float(
                    pred_qpu[i]
                ),
                "delta_pred": float(
                    d_pred[i]
                ),
            }

            for j, f in enumerate(
                EXPECTED_FEATURES
            ):
                row[f"{f}_ideal_RWP64"] = float(
                    X_ideal[i, j]
                )
                row[f"{f}_qpu"] = float(
                    X_qpu[i, j]
                )
                row[f"{f}_delta"] = float(
                    d_feat[i, j]
                )

            daily_rows.append(row)

    summary_df = pd.DataFrame(
        summaries
    ).sort_values(
        "qpu_pred_distortion_rmse_claims"
    ).reset_index(drop=True)

    summary_df["qpu_probe_rank"] = (
        np.arange(len(summary_df)) + 1
    )

    return (
        summary_df,
        pd.DataFrame(daily_rows),
        pd.DataFrame(raw_rows),
    )


# =============================================================================
# Build a complete fixed-layout payload expected by authoritative E7
# =============================================================================

def fixed_layout_payload(
    backend,
    candidate,
    frozen,
    layout,
    selection_method,
    probe_scores=None,
):
    (
        adjacency,
        eligible,
        qubit_cal,
    ) = layoutmod.current_physical_graph(
        backend
    )

    logical_edges = [
        (int(a), int(b))
        for a, b in common.TOPOLOGY_EDGES[
            base.EXPECTED_TOPOLOGY
        ]
    ]

    # Static embedding row, but no hard memory filtering.
    try:
        previous_preflight = base.load_json(
            base.PREVIOUS_PREFLIGHT
        )
        t_est = float(
            previous_preflight[
                "actual_compiled_sample"
            ][
                "median_max_setting_duration_us"
            ]
        )
    except Exception:
        t_est = 232.904

    emb = dict(
        layoutmod.embedding_metrics(
            backend,
            list(layout_key(layout)),
            logical_edges,
            qubit_cal,
            t_est,
        )
    )
    emb["test_layout_id"] = "QRC_WINNER"
    emb["layout_json"] = layout_json(layout)
    emb["selection_method"] = str(
        selection_method
    )

    selected_df = pd.DataFrame([emb])

    # Training-only representative actual RWP64 circuits.
    settings = layoutmod.measurement_settings()
    test_endpoints = np.asarray(
        [
            int(base.EXPECTED_K - 1),
            int(common.N_TRAIN // 2),
            int(common.N_TRAIN - 1),
        ],
        dtype=int,
    )

    compiled_rows = []

    for endpoint in test_endpoints:
        for setting in settings:
            logical = layoutmod.build_rwp64_circuit(
                candidate,
                frozen["angles"],
                int(endpoint),
                setting,
            )

            isa = strict_compile(
                backend,
                logical,
                layout,
            )

            compiled_rows.append(
                layoutmod.compiled_resource_row(
                    isa,
                    backend,
                    "QRC_WINNER",
                    list(layout_key(layout)),
                    int(endpoint),
                    str(setting[0]),
                )
            )

    compiled_df = pd.DataFrame(
        compiled_rows
    )

    (
        qubit_df,
        summary_df,
    ) = layoutmod.build_actual_layout_summaries(
        selected_df,
        compiled_df,
        qubit_cal,
    )

    chosen = summary_df.iloc[0].to_dict()
    chosen["selection_method"] = str(
        selection_method
    )

    if probe_scores:
        for k, v in probe_scores.items():
            chosen[
                f"qrc_probe_{k}"
            ] = v

    chosen_qubits = qubit_df[
        qubit_df["test_layout_id"].astype(str)
        == "QRC_WINNER"
    ].copy()

    return {
        "native_embeddings_found": None,
        "estimated_memory_feasible_embeddings": None,
        "tested_layouts": selected_df,
        "compiled_test_settings": compiled_df,
        "qubit_table": qubit_df,
        "layout_summary": summary_df,
        "chosen_summary": chosen,
        "chosen_layout": [
            int(x) for x in layout_key(layout)
        ],
        "chosen_qubits": chosen_qubits,
    }


# =============================================================================
# Immediate full E7 run on frozen winner
# =============================================================================

def alias_name(path: Path):
    return (
        f"{FULL_ALIAS_PREFIX}__{Path(path).name}"
    )


def run_full_candidate5(
    backend_name,
    full_shots,
    force_over_budget,
    backend,
    expected_calibration_utc,
    candidate,
    frozen,
    winner_layout,
    winner_probe_scores,
):
    payload = fixed_layout_payload(
        backend=backend,
        candidate=candidate,
        frozen=frozen,
        layout=winner_layout,
        selection_method=(
            "qrc_aware_training_only_qpu_probe"
        ),
        probe_scores=winner_probe_scores,
    )

    original_selector = (
        base.choose_fresh_memory_aware_layout
    )
    original_create_run = base.db.create_run
    original_sampler = base.SamplerV2
    original_file = base.__file__
    original_argv = list(sys.argv)

    def fixed_selector(
        backend,
        candidate,
        t_setting_est_us,
        n_layouts=3,
    ):
        print()
        print("=" * 126)
        print(
            "QRC-AWARE FROZEN CANDIDATE #5 PHYSICAL LAYOUT"
        )
        print("=" * 126)
        print(
            "NO memory-aware reselection. NO Qiskit reselection."
        )
        print(
            f"Frozen winner: {list(layout_key(winner_layout))}"
        )
        print(
            "Selection method: 100-point 2022-2024 training-only "
            "QPU hardware-distortion probe"
        )
        print(
            "T1/T2 are diagnostics only."
        )
        print()
        return payload

    def create_run_qrc_aware(**kwargs):
        manifest = kwargs.get(
            "manifest_json"
        )
        if isinstance(manifest, dict):
            manifest = dict(manifest)
            manifest[
                "physical_layout_policy"
            ] = (
                "qrc_aware_training_only_qpu_probe_once_then_frozen"
            )
            manifest[
                "qrc_aware_layout"
            ] = list(
                layout_key(winner_layout)
            )
            manifest[
                "qrc_aware_probe_inputs"
            ] = int(
                DEFAULT_PROBE_INPUTS
            )
            kwargs[
                "manifest_json"
            ] = manifest

        kwargs["notes"] = (
            "Week 11.1H.5 full Candidate-5 RWP64 validation. "
            "Physical layout was selected immediately beforehand using a "
            "training-only QRC-aware QPU probe minimizing RMS QPU-vs-ideal-"
            "RWP64 prediction distortion. The exact winner is frozen; no "
            "memory-aware or Qiskit reselection occurs in the full run. "
            "T1/T2 are recorded as diagnostics only. The original full-CONT "
            "2022-2024 scaler/Ridge remains frozen. 2026 is not loaded."
        )

        return original_create_run(
            **kwargs
        )

    class _GuardedJobProxy:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(
                self._wrapped,
                name,
            )

        def result(self, *args, **kwargs):
            return guarded_job_result(
                self._wrapped,
                backend,
                expected_calibration_utc,
                label="full C5",
            )

    class _GuardedSamplerV2:
        def __init__(self, *args, **kwargs):
            self._inner = original_sampler(
                *args,
                **kwargs,
            )

        def __getattr__(self, name):
            return getattr(
                self._inner,
                name,
            )

        def run(self, *args, **kwargs):
            return _GuardedJobProxy(
                self._inner.run(
                    *args,
                    **kwargs,
                )
            )

    base.choose_fresh_memory_aware_layout = (
        fixed_selector
    )
    base.db.create_run = (
        create_run_qrc_aware
    )
    base.SamplerV2 = (
        _GuardedSamplerV2
    )
    base.__file__ = str(
        Path(__file__).resolve()
    )

    # Preserve historical E7 artifacts.
    known_paths = []
    for attr in BASE_OUTPUT_PATH_ATTRS:
        p = getattr(base, attr, None)
        if p is not None:
            known_paths.append(Path(p))

    with tempfile.TemporaryDirectory(
        prefix="qrc_11_1h5_full_"
    ) as td:
        td = Path(td)
        backups = {}

        for p in known_paths:
            if p.exists():
                bak = td / p.name
                shutil.copy2(p, bak)
                backups[str(p)] = bak

        try:
            argv = [
                str(Path(__file__).name),
                "--backend",
                str(backend_name),
                "--shots",
                str(int(full_shots)),
                "--submit",
            ]

            if force_over_budget:
                argv.append(
                    "--force-over-budget"
                )

            sys.argv = argv

            print()
            print("=" * 140)
            print(
                "IMMEDIATE FULL 365-DAY CANDIDATE #5 RUN "
                "ON QRC-AWARE WINNER"
            )
            print("=" * 140)
            print(
                f"Frozen physical layout: "
                f"{list(layout_key(winner_layout))}"
            )
            print(
                f"Shots/setting: {full_shots}"
            )
            print()

            base.main()

            # Copy the newly generated authoritative outputs to H5 aliases.
            for p in known_paths:
                if p.exists():
                    shutil.copy2(
                        p,
                        RESULTS / alias_name(p),
                    )

        finally:
            # Restore the pre-existing historical E7 files.
            for p in known_paths:
                key = str(p)
                if key in backups:
                    shutil.copy2(
                        backups[key],
                        p,
                    )
                elif p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass

            base.choose_fresh_memory_aware_layout = (
                original_selector
            )
            base.db.create_run = (
                original_create_run
            )
            base.SamplerV2 = (
                original_sampler
            )
            base.__file__ = (
                original_file
            )
            sys.argv = (
                original_argv
            )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
    )
    parser.add_argument(
        "--sim-pool",
        type=int,
        default=DEFAULT_SIM_POOL,
    )
    parser.add_argument(
        "--sim-inputs",
        type=int,
        default=DEFAULT_SIM_INPUTS,
    )
    parser.add_argument(
        "--sim-shots",
        type=int,
        default=DEFAULT_SIM_SHOTS,
    )
    parser.add_argument(
        "--finalists",
        type=int,
        default=DEFAULT_FINALISTS,
    )
    parser.add_argument(
        "--probe-inputs",
        type=int,
        default=DEFAULT_PROBE_INPUTS,
    )
    parser.add_argument(
        "--probe-shots",
        type=int,
        default=DEFAULT_PROBE_SHOTS,
    )
    parser.add_argument(
        "--full-shots",
        type=int,
        default=DEFAULT_FULL_SHOTS,
    )
    parser.add_argument(
        "--aer-batch-size",
        type=int,
        default=DEFAULT_AER_BATCH,
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Submit the real-QPU two-layout training-only probe."
        ),
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help=(
            "Run the QPU probe but do not launch the full 2025 winner run."
        ),
    )
    parser.add_argument(
        "--force-over-budget",
        action="store_true",
    )

    args = parser.parse_args()

    if args.sim_pool < args.finalists:
        raise ValueError(
            "--sim-pool must be >= --finalists."
        )
    if args.finalists != 2:
        print(
            "NOTE: the frozen Week-11 protocol normally uses exactly "
            "2 QPU finalists."
        )
    if args.sim_inputs <= 0:
        raise ValueError(
            "--sim-inputs must be positive."
        )
    if args.probe_inputs <= 0:
        raise ValueError(
            "--probe-inputs must be positive."
        )
    if (
        args.sim_shots <= 0
        or args.probe_shots <= 0
        or args.full_shots <= 0
    ):
        raise ValueError(
            "Shot counts must be positive."
        )

    print("=" * 140)
    print(
        "WEEK 11.1H.5 — CANDIDATE #5 QRC-AWARE "
        "PHYSICAL-LAYOUT SELECTION"
    )
    print("=" * 140)
    print(
        "STRICT BENCHMARK-BLIND SELECTION."
    )
    print(
        "True claim targets and prior 2025 QPU outcomes are NOT "
        "used for physical-layout selection."
    )
    print(
        "Candidate #5 is evaluated as RWP64 using the frozen "
        "full-CONT scaler/Ridge."
    )
    print(
        "T1/T2 ratios are diagnostics, not hard rejection rules."
    )
    print(
        "2026 remains FROZEN / UNUSED."
    )
    print()

    (
        candidate,
        manifest_meta,
        package,
        frozen,
    ) = load_candidate_and_frozen()

    settings = layoutmod.measurement_settings()
    labels = [
        str(s[0]) for s in settings
    ]

    if (
        SETTING_X not in labels
        or SETTING_Z not in labels
    ):
        raise RuntimeError(
            f"Unexpected Candidate-5 settings: {labels}"
        )

    probe_endpoints, X_ideal, pred_ideal = (
        choose_training_endpoints(
            frozen,
            args.probe_inputs,
        )
    )

    sim_pick = np.unique(
        np.rint(
            np.linspace(
                0,
                len(probe_endpoints) - 1,
                int(args.sim_inputs),
            )
        ).astype(int)
    )

    sim_endpoints = probe_endpoints[
        sim_pick
    ]
    X_ideal_sim = X_ideal[
        sim_pick
    ]
    pred_ideal_sim = pred_ideal[
        sim_pick
    ]

    print("Frozen Candidate #5:")
    print(
        f"  original model={base.EXPECTED_CANDIDATE}"
    )
    print(
        f"  hardware surrogate=RWP{base.EXPECTED_K}"
    )
    print(
        f"  topology={candidate['topology']} r={candidate['r']}"
    )
    print(
        f"  alpha={candidate['alpha']} dt={candidate['dt']}"
    )
    print(
        f"  readout={base.EXPECTED_READOUT}"
    )
    print(
        f"  frozen Ridge lambda={frozen['ridge_lambda']}"
    )
    print(
        f"  QPU probe endpoints={len(probe_endpoints)} training-only"
    )
    print(
        f"  local Aer endpoints={len(sim_endpoints)} training-only"
    )
    print(
        f"  simulation shots/setting={args.sim_shots}"
    )
    print(
        f"  probe shots/setting={args.probe_shots}"
    )
    print(
        f"  full shots/setting={args.full_shots}"
    )
    print()

    pd.DataFrame(
        {"endpoint": probe_endpoints}
    ).to_csv(
        RESULTS
        / f"{PREFIX}_training_probe_endpoints.csv",
        index=False,
    )

    service = get_service()
    backend = service.backend(
        args.backend,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    cal_start = current_calibration(
        backend
    )

    print(
        "Calibration before local screening:"
    )
    print(
        json.dumps(
            json_safe(cal_start),
            indent=2,
        )
    )

    # ------------------------------------------------------------------
    # A. Native H2 + Pareto
    # ------------------------------------------------------------------
    print()
    print("=" * 140)
    print(
        "STAGE A — NATIVE H2 ENUMERATION + "
        "CURRENT-CALIBRATION PARETO"
    )
    print("=" * 140)

    native, qubit_cal = (
        enumerate_native_h2_pareto(
            backend
        )
    )

    native.to_csv(
        RESULTS
        / f"{PREFIX}_all_native_h2_layouts.csv",
        index=False,
    )

    pareto = native[
        bool_series(
            native["calibration_pareto"]
        )
    ].copy()

    print(
        f"Native H2 embeddings: {len(native)}"
    )
    print(
        f"Calibration Pareto:   {len(pareto)}"
    )

    if len(pareto) < int(
        args.sim_pool
    ):
        raise RuntimeError(
            "Pareto front is smaller than the requested simulation pool."
        )

    # ------------------------------------------------------------------
    # B0. Actual RWP64 workload proxy
    # ------------------------------------------------------------------
    print()
    print("=" * 140)
    print(
        "STAGE A.2 — ACTUAL RWP64 WORKLOAD PROXY "
        "ON PARETO LAYOUTS"
    )
    print("=" * 140)

    representative_endpoint = int(
        probe_endpoints[
            len(probe_endpoints) // 2
        ]
    )

    proxy = actual_rwp64_proxy_table(
        backend=backend,
        candidate=candidate,
        frozen=frozen,
        layouts_df=pareto,
        representative_endpoint=(
            representative_endpoint
        ),
        settings=settings,
    )

    proxy = proxy[
        bool_series(
            proxy["proxy_compile_pass"]
        )
    ].copy()

    proxy = proxy.sort_values(
        "actual_rwp64_proxy_nll",
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)

    proxy["proxy_rank"] = (
        np.arange(len(proxy)) + 1
    )

    proxy.to_csv(
        RESULTS
        / f"{PREFIX}_pareto_actual_RWP64_proxy.csv",
        index=False,
    )

    sim_pool = top_unique_proxy_layouts(
        proxy,
        args.sim_pool,
    )

    pool_df = pd.DataFrame(
        [
            {
                "simulation_pool_rank": i + 1,
                "pool_source": source,
                "layout": layout_json(layout),
            }
            for i, (
                source,
                layout,
            ) in enumerate(sim_pool)
        ]
    )

    pool_df.to_csv(
        RESULTS
        / f"{PREFIX}_simulation_pool.csv",
        index=False,
    )

    print()
    print(
        f"Calibration-noise simulation pool: {len(sim_pool)} layouts"
    )
    print(
        pool_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # B. Aer
    # ------------------------------------------------------------------
    print()
    print("=" * 140)
    print(
        "STAGE B — LOCAL CALIBRATION-BASED "
        "NOISY RWP64 SIMULATION"
    )
    print("=" * 140)

    sim_summary, sim_daily = (
        noisy_simulation_stage(
            backend=backend,
            candidate=candidate,
            frozen=frozen,
            endpoints=sim_endpoints,
            X_ideal=X_ideal_sim,
            pred_ideal=pred_ideal_sim,
            settings=settings,
            pool=sim_pool,
            shots=args.sim_shots,
            batch_size=args.aer_batch_size,
        )
    )

    sim_summary.to_csv(
        RESULTS
        / f"{PREFIX}_noisy_simulation_summary.csv",
        index=False,
    )

    sim_daily.to_csv(
        RESULTS
        / f"{PREFIX}_noisy_simulation_daily.csv",
        index=False,
    )

    print()
    print(
        "NOISY-SIMULATION RANKING"
    )
    print(
        sim_summary[
            [
                "sim_rank",
                "layout",
                "sim_pred_distortion_rmse_claims",
                "sim_feature_rmse",
                "sim_pred_corr_vs_ideal",
            ]
        ].to_string(index=False)
    )

    finalists = choose_finalists(
        sim_summary,
        args.finalists,
    )

    finalist_rows = []

    for i, layout in enumerate(
        finalists,
        start=1,
    ):
        m = sim_summary[
            sim_summary[
                "layout"
            ].apply(layout_key)
            == layout_key(layout)
        ]

        finalist_rows.append(
            {
                "finalist_id": f"L{i}",
                "layout": layout_json(layout),
                "selection_basis": (
                    "training_only_noisy_RWP64_simulation"
                ),
                "sim_pred_distortion_rmse_claims": float(
                    m.iloc[0][
                        "sim_pred_distortion_rmse_claims"
                    ]
                ),
                "sim_rank": int(
                    m.iloc[0]["sim_rank"]
                ),
            }
        )

    finalists_df = pd.DataFrame(
        finalist_rows
    )

    finalists_df.to_csv(
        RESULTS
        / f"{PREFIX}_qpu_finalists.csv",
        index=False,
    )

    print()
    print("QPU FINALISTS")
    print(
        finalists_df.to_string(
            index=False
        )
    )

    try:
        backend.refresh()
    except Exception:
        pass

    cal_preprobe = current_calibration(
        backend
    )

    if (
        calibration_utc(cal_start)
        and calibration_utc(cal_preprobe)
        and calibration_utc(cal_start)
        != calibration_utc(cal_preprobe)
    ):
        raise RuntimeError(
            "Calibration changed during local Candidate-5 screening. "
            "Rerun before submitting a QPU probe."
        )

    local_summary = {
        "candidate": "CONT_H2_R2_RWP64",
        "selection_scope": (
            "2022-2024 training-only"
        ),
        "true_targets_used": False,
        "historical_2025_qpu_used": False,
        "qiskit_layout_forced": False,
        "old_memory_aware_layout_forced": False,
        "native_embeddings": int(
            len(native)
        ),
        "pareto_layouts": int(
            len(pareto)
        ),
        "simulation_pool": int(
            len(sim_pool)
        ),
        "finalists": int(
            len(finalists)
        ),
        "calibration_start": cal_start,
        "calibration_preprobe": cal_preprobe,
    }

    write_json(
        RESULTS
        / f"{PREFIX}_local_summary.json",
        local_summary,
    )

    if not args.submit:
        print()
        print("=" * 140)
        print(
            "LOCAL C5 STAGES COMPLETE — NO QPU JOB SUBMITTED"
        )
        print("=" * 140)
        print(
            "To run the two-layout QPU probe and then the immediate "
            "full 2025 winner run:"
        )
        print(
            f"python {Path(__file__).name} --submit "
            f"--backend {args.backend} "
            f"--probe-inputs {args.probe_inputs} "
            f"--probe-shots {args.probe_shots} "
            f"--full-shots {args.full_shots}"
        )
        print(
            "2026 remains FROZEN / UNUSED."
        )
        return

    # ------------------------------------------------------------------
    # QPU budget estimate
    # ------------------------------------------------------------------
    probe_circuits = (
        len(probe_endpoints)
        * len(finalists)
        * len(settings)
    )

    probe_estimate = (
        HISTORICAL_FULL_C5_QPU_CHARGE_S
        * (
            probe_circuits
            * int(args.probe_shots)
        )
        / (
            HISTORICAL_FULL_C5_N_CIRCUITS
            * HISTORICAL_FULL_C5_SHOTS
        )
    )

    total_estimate = (
        probe_estimate
        + (
            0.0
            if args.probe_only
            else HISTORICAL_FULL_C5_QPU_CHARGE_S
        )
    )

    usage = get_service_usage(
        service
    )
    remaining = remaining_usage_seconds(
        usage
    )

    print()
    print(
        "QPU PLANNING HEURISTIC"
    )
    print(
        f"  probe circuits:             {probe_circuits}"
    )
    print(
        f"  rough probe QPU charge:     {probe_estimate:.1f} s"
    )

    if not args.probe_only:
        print(
            f"  historical full C5 charge:  "
            f"{HISTORICAL_FULL_C5_QPU_CHARGE_S:.1f} s"
        )

    print(
        f"  rough probe+full total:     {total_estimate:.1f} s"
    )

    if remaining is not None:
        print(
            f"  service reported remaining: {remaining:.1f} s"
        )

    if (
        remaining is not None
        and total_estimate > remaining
        and not args.force_over_budget
    ):
        raise RuntimeError(
            "Submission blocked: rough Candidate-5 probe+full estimate "
            f"{total_estimate:.1f}s exceeds reported remaining "
            f"{remaining:.1f}s. Use --force-over-budget only if "
            "additional usage/cost is intentionally accepted."
        )

    # ------------------------------------------------------------------
    # C. Real QPU probe
    # ------------------------------------------------------------------
    print()
    print("=" * 140)
    print(
        "STAGE C — REAL-QPU INTERLEAVED 2-LAYOUT "
        "TRAINING-ONLY RWP64 PROBE"
    )
    print("=" * 140)

    (
        circuits,
        probe_meta,
        probe_resources,
    ) = compile_interleaved_probe(
        backend=backend,
        candidate=candidate,
        frozen=frozen,
        endpoints=probe_endpoints,
        settings=settings,
        finalists=finalists,
    )

    probe_resources.to_csv(
        RESULTS
        / f"{PREFIX}_qpu_probe_resources.csv",
        index=False,
    )

    pd.DataFrame(
        probe_meta
    ).to_csv(
        RESULTS
        / f"{PREFIX}_qpu_probe_circuit_order.csv",
        index=False,
    )

    print(
        f"Submitting {len(circuits)} circuits at "
        f"{args.probe_shots} shots/setting..."
    )

    sampler = SamplerV2(
        mode=backend
    )

    job = sampler.run(
        circuits,
        shots=int(args.probe_shots),
    )

    print(
        f"Probe job ID: {job.job_id()}"
    )

    result = guarded_job_result(
        job,
        backend,
        calibration_utc(
            cal_preprobe
        ),
        label="C5 QPU probe",
    )

    metrics = None
    try:
        metrics = json_safe(
            job.metrics()
        )
    except Exception:
        pass

    timing = None
    try:
        timing = base.extract_ibm_qpu_timing(
            job,
            metrics,
        )
    except Exception:
        timing = None

    running_utc = None
    if isinstance(timing, dict):
        running_utc = timing.get(
            "ibm_job_running_at_utc"
        )

    cal_probe_running = calibration_at(
        backend,
        running_utc,
    )

    (
        probe_summary,
        probe_daily,
        probe_raw,
    ) = qpu_probe_metric_table(
        result=result,
        meta=probe_meta,
        finalists=finalists,
        endpoints=probe_endpoints,
        frozen=frozen,
        X_ideal=X_ideal,
        pred_ideal=pred_ideal,
    )

    probe_summary.to_csv(
        RESULTS
        / f"{PREFIX}_qpu_probe_summary.csv",
        index=False,
    )
    probe_daily.to_csv(
        RESULTS
        / f"{PREFIX}_qpu_probe_daily.csv",
        index=False,
    )
    probe_raw.to_csv(
        RESULTS
        / f"{PREFIX}_qpu_probe_raw_counts.csv",
        index=False,
    )

    print()
    print(
        "QPU LAYOUT-PROBE RANKING"
    )
    print(
        probe_summary[
            [
                "qpu_probe_rank",
                "layout",
                "qpu_pred_distortion_rmse_claims",
                "qpu_feature_rmse",
                "qpu_pred_corr_vs_ideal",
            ]
        ].to_string(index=False)
    )

    winner = probe_summary.iloc[0]
    winner_layout = list(
        layout_key(
            winner["layout"]
        )
    )

    winner_scores = {
        k: json_safe(v)
        for k, v in winner.to_dict().items()
        if k != "layout"
    }

    print()
    print(
        "QRC-AWARE CANDIDATE #5 PHYSICAL WINNER"
    )
    print(
        f"  layout={winner_layout}"
    )
    print(
        f"  E_delta_y_QPU="
        f"{float(winner['qpu_pred_distortion_rmse_claims']):.6f} claims"
    )
    print(
        "  selection used no true claim targets"
    )

    winner_payload = {
        "layout": winner_layout,
        "qpu_probe": winner_scores,
        "probe_job_id": job.job_id(),
        "probe_job_metrics": metrics,
        "probe_timing": timing,
        "calibration_before_local": cal_start,
        "calibration_before_probe": cal_preprobe,
        "calibration_at_probe_running": cal_probe_running,
        "2026_loaded": False,
    }

    write_json(
        RESULTS
        / f"{PREFIX}_winner.json",
        winner_payload,
    )

    if args.probe_only:
        print()
        print(
            "PROBE-ONLY MODE COMPLETE — no full 2025 run submitted."
        )
        return

    # ------------------------------------------------------------------
    # D. Immediate full 2025
    # ------------------------------------------------------------------
    try:
        backend.refresh()
    except Exception:
        pass

    cal_prefull = current_calibration(
        backend
    )

    probe_running_utc = calibration_utc(
        cal_probe_running
    )
    prefull_utc = calibration_utc(
        cal_prefull
    )

    if (
        probe_running_utc
        and prefull_utc
        and probe_running_utc
        != prefull_utc
    ):
        raise RuntimeError(
            "IBM reported calibration changed between Candidate-5 "
            "probe execution and full-run submission. Full 2025 is blocked."
        )

    print()
    print(
        "Calibration guard before full 2025:"
    )
    print(
        json.dumps(
            json_safe(cal_prefull),
            indent=2,
        )
    )

    run_full_candidate5(
        backend_name=args.backend,
        full_shots=args.full_shots,
        force_over_budget=(
            args.force_over_budget
        ),
        backend=backend,
        expected_calibration_utc=(
            prefull_utc
        ),
        candidate=candidate,
        frozen=frozen,
        winner_layout=winner_layout,
        winner_probe_scores=(
            winner_scores
        ),
    )

    combined = {
        "candidate": "CONT_H2_R2_RWP64",
        "winner_layout": winner_layout,
        "probe_distortion_rmse_claims": float(
            winner[
                "qpu_pred_distortion_rmse_claims"
            ]
        ),
        "probe_feature_rmse": float(
            winner[
                "qpu_feature_rmse"
            ]
        ),
        "probe_pred_corr_vs_ideal": float(
            winner[
                "qpu_pred_corr_vs_ideal"
            ]
        ),
        "same_reported_calibration_probe_to_full_submission": (
            bool(
                probe_running_utc
                and prefull_utc
                and probe_running_utc
                == prefull_utc
            )
        ),
        "2026_loaded": False,
    }

    write_json(
        RESULTS
        / f"{PREFIX}_combined_summary.json",
        combined,
    )

    print()
    print("=" * 140)
    print(
        "11.1H.5 COMPLETE"
    )
    print("=" * 140)
    print(
        f"Winner layout: {winner_layout}"
    )
    print(
        f"Probe hardware distortion: "
        f"{combined['probe_distortion_rmse_claims']:.6f} claims RMS"
    )
    print(
        "2026 remains FROZEN / UNUSED."
    )


if __name__ == "__main__":
    main()
