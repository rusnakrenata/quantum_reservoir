#!/usr/bin/env python
from __future__ import annotations

"""
WEEK 12.6A.1 — IQM CANDIDATE #1 PRE-QPU QRC-AWARE LAYOUT SCREEN
================================================================

Frozen Candidate #1
-------------------
    candidate_id = H6_W06_r2_local_04_XZinj_plus_YX45
    topology     = H6
    protocol     = RWP
    W            = 6
    r            = 2
    readout      = XZinj_plus_YX45

Purpose
-------
Reproduce the LOCAL part of the final IBM task-aware layout-selection pipeline
on IQM Emerald, BEFORE consuming any real-QPU time:

    fresh Emerald profiler
      -> native H6 embeddings
      -> corrected Rule-5 calibration Pareto
      -> actual Candidate-1 workload proxy
      -> custom IQMFakeBackend noise model from CURRENT calibration
      -> noisy simulation on deterministic 2022-2024 training-only endpoints
      -> retain TWO physical layouts for the later real-QPU training-only probe

The next script, not this one, will interleave those two layouts on real Emerald
using 100 deterministic training-only endpoints and select the final physical
layout by

    RMS_i [ yhat_QPU(i,L) - yhat_ideal(i) ].

Scientific protections
----------------------
- Candidate identity is frozen and hard-audited.
- Ridge is fit only on 2022-2024 training data.
- No true y targets are used for physical-layout ranking.
- 2025 validation metrics are NOT computed in this script.
- 2026 is NEVER loaded by this script.
- No real-QPU job is submitted.
- No weighted hardware score is used.
- Rule-5 layouts are diagnostic; they are NOT forced into the fake-simulation
  pool or into the later QPU finalists.
- IQMFakeBackend error parameters are NOT set naively to 1-fidelity.
  For PRX/CZ we numerically solve for the depolarizing parameter p such that

      average_fidelity(
          thermal_relaxation(T1,T2,duration)
          compose depolarizing(p)
      ) ~= reported benchmarking fidelity.

  This matches IQMFakeBackend's documented noise-model semantics.

Expected local files
--------------------
12_03d_iqm_rwp_common.py
12_05e2_iqm_custom_fake_feasibility_fixed.py
iqm_account.py

results/12_05j_iqm_final_five_candidates.csv
results/12_04b_iqm_rule1_rule4_candidates.csv

Outputs
-------
results/12_06a1_c1_candidate_identity.json
results/12_06a1_c1_training_probe_endpoints.csv
results/12_06a1_c1_live_quality.csv
results/12_06a1_c1_live_durations.csv
results/12_06a1_c1_all_h6_embeddings.csv
results/12_06a1_c1_calibration_pareto.csv
results/12_06a1_c1_rule5_top3_diagnostic.csv
results/12_06a1_c1_workload_proxy.csv
results/12_06a1_c1_fake_profile_audit.csv
results/12_06a1_c1_fake_profile_manifest.json
results/12_06a1_c1_fake_sim_per_endpoint.csv
results/12_06a1_c1_fake_sim_summary.csv
results/12_06a1_c1_qpu_finalists.csv
results/12_06a1_c1_manifest.json

Default simulation settings mirror the IBM task-aware pre-screen:
    20 deterministic training-only endpoints
    64 shots/setting
    12 layouts entering noisy simulation
    2 layouts retained for the later QPU probe
"""

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import average_gate_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error

from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import (
    IQMErrorProfile,
    IQMFakeBackend,
)

from iqm_account import get_client, get_iqm_backend


# =============================================================================
# PATHS / FROZEN CONSTANTS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_PATH = HERE / "12_03d_iqm_rwp_common.py"
E2_PATH = HERE / "12_05e_iqm_custom_fake_feasibility.py"

FINAL5 = RESULTS / "12_05j_iqm_final_five_candidates.csv"
H56_SOURCE = RESULTS / "12_04b_iqm_rule1_rule4_candidates.csv"

PREFIX = "12_06a1_c1"

OUT_ID = RESULTS / f"{PREFIX}_candidate_identity.json"
OUT_PROBES = RESULTS / f"{PREFIX}_training_probe_endpoints.csv"
OUT_Q = RESULTS / f"{PREFIX}_live_quality.csv"
OUT_D = RESULTS / f"{PREFIX}_live_durations.csv"
OUT_ALL = RESULTS / f"{PREFIX}_all_h6_embeddings.csv"
OUT_PARETO = RESULTS / f"{PREFIX}_calibration_pareto.csv"
OUT_R5 = RESULTS / f"{PREFIX}_rule5_top3_diagnostic.csv"
OUT_PROXY = RESULTS / f"{PREFIX}_workload_proxy.csv"
OUT_PROFILE = RESULTS / f"{PREFIX}_fake_profile_audit.csv"
OUT_PROFILE_JSON = RESULTS / f"{PREFIX}_fake_profile_manifest.json"
OUT_SIM_EP = RESULTS / f"{PREFIX}_fake_sim_per_endpoint.csv"
OUT_SIM_SUM = RESULTS / f"{PREFIX}_fake_sim_summary.csv"
OUT_FINALISTS = RESULTS / f"{PREFIX}_qpu_finalists.csv"
OUT_MANIFEST = RESULTS / f"{PREFIX}_manifest.json"

BACKEND_NAME = "emerald"

EXPECTED_CANDIDATE_ID = "H6_W06_r2_local_04_XZinj_plus_YX45"
EXPECTED_TOPOLOGY = "H6"
EXPECTED_WINDOW = 6
EXPECTED_R = 2
EXPECTED_READOUT = "XZinj_plus_YX45"
EXPECTED_SEARCH_ID = "local_04"
EXPECTED_CV = 3.330960

OPT_LEVEL = 1
SEED_TRANSPILE = 42
SEED_SIMULATOR = 42

DEFAULT_SIM_POOL = 12
DEFAULT_SIM_INPUTS = 20
DEFAULT_SIM_SHOTS = 64
DEFAULT_QPU_FINALISTS = 2
DEFAULT_QPU_PROBE_INPUTS = 100

MEMORY_LOGICAL = (4, 5)

# Correct H6 logical graph frozen in Week 12.
H6_EDGES = (
    (0, 1),
    (0, 4),
    (1, 5),
    (2, 3),
    (2, 5),
    (3, 4),
    (4, 5),
)


# =============================================================================
# MODULE IMPORT
# =============================================================================

def import_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required project script: {path}\n"
            "Put this 12.6A.1 script in the project root beside the Week-12 scripts."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


common = import_from_path("12_03d_iqm_rwp_common", COMMON_PATH)
e2 = import_from_path("qrc_12_05e2", E2_PATH)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def jsonable(x: Any):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if hasattr(x, "model_dump"):
        try:
            return jsonable(x.model_dump(mode="json"))
        except Exception:
            try:
                return jsonable(x.model_dump())
            except Exception:
                pass
    if hasattr(x, "dict"):
        try:
            return jsonable(x.dict())
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


def finite(x, default=np.nan):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def normalize_qb(q):
    q = str(q).strip()
    if not q.startswith("QB"):
        raise ValueError(f"Unexpected IQM qubit label {q!r}")
    return q


def qb_num(q):
    return int(normalize_qb(q)[2:])


def normalize_pair(a, b):
    a = normalize_qb(a)
    b = normalize_qb(b)
    return tuple(sorted((a, b), key=qb_num))


def pair_text(pair):
    return "__".join(normalize_pair(*pair))


def parse_locus_pair(value):
    s = str(value).strip()
    if "__" not in s:
        return None
    a, b = s.split("__", 1)
    if not (a.startswith("QB") and b.startswith("QB")):
        return None
    return normalize_pair(a, b)


def deterministic_even_subset(values, n):
    values = np.asarray(values, dtype=int)
    if len(values) == 0:
        raise RuntimeError("Cannot sample an empty endpoint array.")
    n = int(max(1, min(int(n), len(values))))
    idx = np.unique(np.linspace(0, len(values) - 1, n, dtype=int))
    out = values[idx]
    if len(out) != n:
        raise RuntimeError(f"Expected {n} endpoints, got {len(out)}.")
    return out


# =============================================================================
# CANDIDATE #1 FREEZE + IDEAL TRAINING REFERENCE
# =============================================================================

def load_frozen_candidate1():
    if not FINAL5.exists():
        raise FileNotFoundError(FINAL5)
    if not H56_SOURCE.exists():
        raise FileNotFoundError(H56_SOURCE)

    final5 = pd.read_csv(FINAL5)
    role1 = final5[pd.to_numeric(final5["role_id"], errors="coerce").eq(1)]

    if len(role1) != 1:
        raise RuntimeError(f"Expected exactly one IQM role-1 row, found {len(role1)}.")

    frozen = role1.iloc[0]
    cid = str(frozen["candidate_id"])

    if cid != EXPECTED_CANDIDATE_ID:
        raise RuntimeError(
            "Candidate #1 identity changed.\n"
            f"Expected: {EXPECTED_CANDIDATE_ID}\n"
            f"Found:    {cid}"
        )

    source = pd.read_csv(H56_SOURCE)
    g = source[source["candidate_uid"].astype(str).eq(cid)]

    if len(g) != 1:
        raise RuntimeError(
            f"Expected one authoritative source row for {cid}, found {len(g)}."
        )

    row = g.iloc[0]
    candidate = common.candidate_from_row(row)
    candidate["window"] = int(row["window"])
    candidate["readout"] = str(row["readout"])
    candidate["trotter_search_id"] = str(row["trotter_search_id"])

    checks = {
        "topology": candidate["topology"] == EXPECTED_TOPOLOGY,
        "window": candidate["window"] == EXPECTED_WINDOW,
        "r": candidate["r"] == EXPECTED_R,
        "readout": candidate["readout"] == EXPECTED_READOUT,
        "search_id": candidate["trotter_search_id"] == EXPECTED_SEARCH_ID,
    }

    if not all(checks.values()):
        raise RuntimeError(f"Frozen Candidate #1 identity audit failed: {checks}")

    payload = {
        "candidate_id": cid,
        "topology": candidate["topology"],
        "protocol": "RWP",
        "window": candidate["window"],
        "r": candidate["r"],
        "readout": candidate["readout"],
        "trotter_search_id": candidate["trotter_search_id"],
        "alpha": candidate["alpha"],
        "dt": candidate["dt"],
        "hx": candidate["hx"],
        "hy": candidate["hy"],
        "J": {f"{i}{j}": v for (i, j), v in candidate["J"].items()},
        "frozen_cv_from_final5": finite(frozen["cv_rmse"]),
        "identity_checks": checks,
    }

    return candidate, row, payload


def build_training_reference(candidate, qpu_probe_inputs, sim_inputs):
    # Existing shared Week-12 loader constructs the 2022-2025 dataframe,
    # but this script uses ONLY endpoints < N_TRAIN for fitting/ranking.
    # No 2025 metric is calculated.
    work_tv, cols, y_all = common.load_train_validation()

    angles = common.make_angles(
        work_tv,
        cols,
        float(candidate["alpha"]),
    )

    channels = common.build_channels(candidate, angles)
    endpoints, master = common.rwp_master_feature_bank_from_channels(
        channels,
        int(candidate["window"]),
    )

    features = list(common.READOUT_FEATURES[candidate["readout"]])
    X = master[features].to_numpy(dtype=float)
    y_ep = np.asarray(y_all, dtype=float)[endpoints]

    train_mask = endpoints < int(common.N_TRAIN)
    train_endpoints = np.asarray(endpoints[train_mask], dtype=int)
    X_train = X[train_mask]
    y_train = y_ep[train_mask]

    selected_lambda, folds, cv_summary = common.select_lambda_training_only(
        X_train,
        y_train,
        n_splits=5,
    )

    model, scaler, keep = common.fit_scaled_ridge(
        X_train,
        y_train,
        selected_lambda,
    )

    cv = float(cv_summary.iloc[0]["cv_rmse_mean"])
    cv_sd = float(cv_summary.iloc[0]["cv_rmse_std"])

    if not math.isclose(cv, EXPECTED_CV, rel_tol=0.0, abs_tol=5e-5):
        raise RuntimeError(
            "Candidate #1 ideal-CV reproduction failed.\n"
            f"Expected ~{EXPECTED_CV:.6f}; recomputed {cv:.9f}."
        )

    probe_endpoints = deterministic_even_subset(
        train_endpoints,
        qpu_probe_inputs,
    )
    sim_endpoints = deterministic_even_subset(
        probe_endpoints,
        sim_inputs,
    )

    endpoint_to_row = {
        int(ep): i
        for i, ep in enumerate(endpoints)
    }

    probe_rows = np.asarray(
        [endpoint_to_row[int(ep)] for ep in probe_endpoints],
        dtype=int,
    )
    sim_rows = np.asarray(
        [endpoint_to_row[int(ep)] for ep in sim_endpoints],
        dtype=int,
    )

    X_probe_ideal = X[probe_rows]
    X_sim_ideal = X[sim_rows]

    pred_probe_ideal = common.predict_scaled_ridge(
        model, scaler, keep, X_probe_ideal
    )
    pred_sim_ideal = common.predict_scaled_ridge(
        model, scaler, keep, X_sim_ideal
    )

    pd.DataFrame({
        "endpoint": probe_endpoints,
        "used_in_fake_sim": np.isin(probe_endpoints, sim_endpoints),
    }).to_csv(OUT_PROBES, index=False)

    return {
        "work_tv": work_tv,
        "cols": cols,
        "angles": angles,
        "endpoints": endpoints,
        "features": features,
        "model": model,
        "scaler": scaler,
        "keep": keep,
        "selected_lambda": float(selected_lambda),
        "cv_rmse": cv,
        "cv_rmse_std": cv_sd,
        "probe_endpoints": probe_endpoints,
        "sim_endpoints": sim_endpoints,
        "X_probe_ideal": X_probe_ideal,
        "X_sim_ideal": X_sim_ideal,
        "pred_probe_ideal": np.asarray(pred_probe_ideal, dtype=float),
        "pred_sim_ideal": np.asarray(pred_sim_ideal, dtype=float),
        "endpoint_to_row": endpoint_to_row,
        "master": master,
    }


# =============================================================================
# LIVE IQM PROFILER
# =============================================================================

def extract_calibrated_cz_graph(dqa_plain):
    cz = (
        dqa_plain
        .get("gates", {})
        .get("cz", {})
    )
    implementations = cz.get("implementations", {}) or {}
    default_impl = cz.get("default_implementation")
    overrides = cz.get("override_default_implementation", {}) or {}

    all_pairs = set()
    impl_pairs = defaultdict(set)

    for impl_name, info in implementations.items():
        for locus in info.get("loci", []) or []:
            if len(locus) != 2:
                continue
            a, b = locus
            if (
                isinstance(a, str)
                and isinstance(b, str)
                and a.startswith("QB")
                and b.startswith("QB")
            ):
                p = normalize_pair(a, b)
                all_pairs.add(p)
                impl_pairs[str(impl_name)].add(p)

    active = {p: default_impl for p in all_pairs}

    for key, impl_name in overrides.items():
        if isinstance(key, str):
            # Handles "QB1,QB2" and serialized tuple-like keys.
            parts = re.findall(r"QB\d+", key)
        elif isinstance(key, (list, tuple)):
            parts = list(key)
        else:
            parts = []

        if len(parts) == 2:
            active[normalize_pair(parts[0], parts[1])] = str(impl_name)

    for p in all_pairs:
        impl = active.get(p)
        if impl is None or p not in impl_pairs.get(str(impl), set()):
            candidates = [
                name for name, pairs in impl_pairs.items()
                if p in pairs
            ]
            if len(candidates) == 1:
                active[p] = candidates[0]

    adjacency = defaultdict(set)
    for a, b in all_pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)

    return dict(adjacency), all_pairs, active


def logical_adjacency(edges):
    out = {q: set() for q in range(6)}
    for a, b in edges:
        out[a].add(b)
        out[b].add(a)
    return out


def enumerate_native_embeddings(physical_adjacency, logical_edges):
    logical_adj = logical_adjacency(logical_edges)

    physical_nodes = sorted(
        physical_adjacency,
        key=qb_num,
    )
    logical_order = sorted(
        range(6),
        key=lambda q: (-len(logical_adj[q]), q),
    )

    candidates = {
        q: [
            p for p in physical_nodes
            if len(physical_adjacency[p]) >= len(logical_adj[q])
        ]
        for q in range(6)
    }

    mappings = []
    assigned = {}
    used = set()

    def backtrack(depth):
        if depth == len(logical_order):
            mappings.append({q: assigned[q] for q in range(6)})
            return

        q = logical_order[depth]

        for p in candidates[q]:
            if p in used:
                continue

            ok = True
            for qn in logical_adj[q]:
                if qn not in assigned:
                    continue
                pn = assigned[qn]
                if pn not in physical_adjacency.get(p, set()):
                    ok = False
                    break

            if not ok:
                continue

            assigned[q] = p
            used.add(p)
            backtrack(depth + 1)
            used.remove(p)
            del assigned[q]

    backtrack(0)
    return mappings


def select_metric(df, metric_type, locus, preferred_impl=None):
    g = df[
        df["metric_type"].astype(str).eq(str(metric_type))
        & df["locus"].astype(str).eq(str(locus))
    ].copy()

    if len(g) == 0:
        return np.nan, None, "MISSING"

    if preferred_impl is not None and "implementation" in g.columns:
        exact = g[g["implementation"].astype(str).eq(str(preferred_impl))]
        if len(exact) >= 1:
            r = exact.iloc[-1]
            return finite(r["value"]), str(r["implementation"]), "IMPLEMENTATION_MATCH"

    if len(g) == 1:
        r = g.iloc[0]
        return finite(r["value"]), str(r.get("implementation", "")), "UNIQUE"

    # Do not silently average multiple implementation-specific observations.
    return np.nan, None, "AMBIGUOUS"


def select_duration(df, gate_type, locus, preferred_impl=None):
    g = df[
        df["gate_type"].astype(str).eq(str(gate_type))
        & df["locus"].astype(str).eq(str(locus))
    ].copy()

    if len(g) == 0:
        return np.nan, None, "MISSING"

    if preferred_impl is not None:
        exact = g[g["implementation"].astype(str).eq(str(preferred_impl))]
        if len(exact) >= 1:
            r = exact.iloc[-1]
            return finite(r["duration_ns"]), str(r["implementation"]), "IMPLEMENTATION_MATCH"

    if len(g) == 1:
        r = g.iloc[0]
        return finite(r["duration_ns"]), str(r["implementation"]), "UNIQUE"

    return np.nan, None, "AMBIGUOUS"


def fresh_profiler():
    client = get_client(quantum_computer=BACKEND_NAME)

    sqa = client.get_static_quantum_architecture()
    dqa = client.get_dynamic_quantum_architecture()
    calibration = client.get_calibration_set()
    quality = client.get_quality_metric_set()

    backend = get_iqm_backend(
        quantum_computer=BACKEND_NAME,
        use_metrics=True,
    )

    sqa_plain = e2.jsonable(sqa)
    dqa_plain = e2.jsonable(dqa)

    qrows = e2.parse_quality(BACKEND_NAME, quality)
    drows = e2.parse_durations(BACKEND_NAME, calibration)

    qdf = pd.DataFrame(qrows)
    ddf = pd.DataFrame(drows)

    qdf.to_csv(OUT_Q, index=False)
    ddf.to_csv(OUT_D, index=False)

    adjacency, active_pairs, active_cz_impl = extract_calibrated_cz_graph(
        dqa_plain
    )

    if len(active_pairs) == 0:
        raise RuntimeError("Fresh Emerald profiler found no calibrated CZ graph.")

    gates = dqa_plain.get("gates", {})
    measure_default = (
        gates.get("measure", {}).get("default_implementation")
    )
    prx_default = (
        gates.get("prx", {}).get("default_implementation")
    )

    calibration_id = dqa_plain.get("calibration_set_id")
    if calibration_id is None:
        calibration_id = sqa_plain.get("calibration_set_id")

    return {
        "client": client,
        "backend": backend,
        "sqa": sqa,
        "sqa_plain": sqa_plain,
        "dqa_plain": dqa_plain,
        "qdf": qdf,
        "ddf": ddf,
        "adjacency": adjacency,
        "active_pairs": active_pairs,
        "active_cz_impl": active_cz_impl,
        "measure_default": measure_default,
        "prx_default": prx_default,
        "calibration_set_id": str(calibration_id),
    }


# =============================================================================
# CURRENT H6 CALIBRATION METRICS / PARETO / RULE 5
# =============================================================================

def physical_metrics_for_mapping(mapping, prof):
    qdf = prof["qdf"]
    active_impl = prof["active_cz_impl"]

    qubits = [mapping[q] for q in range(6)]
    memory = [mapping[q] for q in MEMORY_LOGICAL]
    edges = [
        normalize_pair(mapping[a], mapping[b])
        for a, b in H6_EDGES
    ]

    t1 = []
    t2 = []
    mem_t2 = []
    e01 = []
    e10 = []
    rofid = []
    czfid = []

    for q in qubits:
        v, _, _ = select_metric(qdf, "t1", q)
        t1.append(v)

        v, _, _ = select_metric(qdf, "t2_echo", q)
        t2.append(v)

        v, _, _ = select_metric(
            qdf, "readout_error_0_to_1", q, prof["measure_default"]
        )
        e01.append(v)

        v, _, _ = select_metric(
            qdf, "readout_error_1_to_0", q, prof["measure_default"]
        )
        e10.append(v)

        v, _, _ = select_metric(
            qdf, "readout_fidelity", q, prof["measure_default"]
        )
        rofid.append(v)

    for q in memory:
        v, _, _ = select_metric(qdf, "t2_echo", q)
        mem_t2.append(v)

    for p in edges:
        impl = active_impl.get(p)
        v, _, _ = select_metric(
            qdf, "cz_fidelity", pair_text(p), impl
        )
        czfid.append(v)

    vals = t1 + t2 + mem_t2 + e01 + e10 + czfid
    complete = all(np.isfinite(v) for v in vals)

    max_asym = np.nan
    if all(np.isfinite(v) for v in e01 + e10):
        max_asym = float(max(max(e01), max(e10)))

    return {
        "complete_metrics": bool(complete),
        "min_t1_us": float(np.min(t1) * 1e6) if all(np.isfinite(t1)) else np.nan,
        "min_t2_echo_us": float(np.min(t2) * 1e6) if all(np.isfinite(t2)) else np.nan,
        "memory_min_t2_echo_us": (
            float(np.min(mem_t2) * 1e6)
            if all(np.isfinite(mem_t2))
            else np.nan
        ),
        "max_asym_readout_error": max_asym,
        "min_readout_fidelity": (
            float(np.min(rofid))
            if all(np.isfinite(rofid))
            else np.nan
        ),
        "min_cz_fidelity": (
            float(np.min(czfid))
            if all(np.isfinite(czfid))
            else np.nan
        ),
    }


def pareto_core(df):
    work = df[df["complete_metrics"].astype(bool)].copy()
    if len(work) == 0:
        return work

    cols = [
        "min_t1_us",
        "min_t2_echo_us",
        "max_asym_readout_error",
        "min_cz_fidelity",
    ]
    X = work[cols].to_numpy(dtype=float)
    keep = np.ones(len(work), dtype=bool)

    for i in range(len(work)):
        for j in range(len(work)):
            if i == j:
                continue

            # j dominates i:
            # T1/T2/CZ higher or equal, readout error lower or equal.
            no_worse = (
                X[j, 0] >= X[i, 0]
                and X[j, 1] >= X[i, 1]
                and X[j, 2] <= X[i, 2]
                and X[j, 3] >= X[i, 3]
            )
            strict = (
                X[j, 0] > X[i, 0]
                or X[j, 1] > X[i, 1]
                or X[j, 2] < X[i, 2]
                or X[j, 3] > X[i, 3]
            )

            if no_worse and strict:
                keep[i] = False
                break

    return work.iloc[np.flatnonzero(keep)].copy()


def profile_h6_embeddings(prof):
    mappings = enumerate_native_embeddings(
        prof["adjacency"],
        H6_EDGES,
    )

    rows = []

    for eid, mapping in enumerate(mappings, start=1):
        metrics = physical_metrics_for_mapping(mapping, prof)
        rows.append({
            "embedding_id": eid,
            "layout": json.dumps([mapping[q] for q in range(6)]),
            "q0_C": mapping[0],
            "q1_D": mapping[1],
            "q2_P": mapping[2],
            "q3_H": mapping[3],
            "q4_M1": mapping[4],
            "q5_M2": mapping[5],
            **metrics,
        })

    all_df = pd.DataFrame(rows)
    all_df.to_csv(OUT_ALL, index=False)

    pareto = pareto_core(all_df)
    pareto = pareto.sort_values(
        ["min_cz_fidelity", "min_t2_echo_us", "memory_min_t2_echo_us"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    pareto.to_csv(OUT_PARETO, index=False)

    if len(pareto) < 3:
        raise RuntimeError(
            f"Fresh H6 Pareto set unexpectedly contains only {len(pareto)} layouts."
        )

    # IBM-mirrored Rule-5 hierarchy retained as diagnostic/reference only.
    r5 = pareto.sort_values(
        ["min_cz_fidelity", "min_t2_echo_us", "memory_min_t2_echo_us"],
        ascending=[False, False, False],
    ).head(10)

    r5 = r5.sort_values(
        ["min_t2_echo_us", "memory_min_t2_echo_us", "min_cz_fidelity"],
        ascending=[False, False, False],
    ).head(5)

    r5 = r5.sort_values(
        ["memory_min_t2_echo_us", "min_t2_echo_us", "min_cz_fidelity"],
        ascending=[False, False, False],
    ).head(3).copy()

    r5.insert(0, "rule5_rank", np.arange(1, len(r5) + 1))
    r5.to_csv(OUT_R5, index=False)

    return all_df, pareto, r5


# =============================================================================
# ACTUAL CANDIDATE-1 CIRCUITS
# =============================================================================

def readout_settings(readout):
    if readout != "XZinj_plus_YX45":
        raise RuntimeError(
            f"This Candidate-1 script expects {EXPECTED_READOUT}; got {readout}."
        )

    return [
        (
            "XXXXYX",
            {0: "X", 1: "X", 2: "X", 3: "X", 4: "Y", 5: "X"},
            list(range(6)),
        ),
        (
            "ZZZZZZ",
            {q: "Z" for q in range(6)},
            list(range(6)),
        ),
    ]


def append_rwp_step(qc, candidate, angle_row, first_step):
    if not first_step:
        for q in range(4):
            qc.reset(q)

    for q in range(4):
        qc.ry(float(angle_row[q]), q)

    r = int(candidate["r"])
    dt = float(candidate["dt"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])
    J = candidate["J"]

    for _ in range(r):
        for i, j in common.TOPOLOGY_EDGES[candidate["topology"]]:
            qc.rzz(
                2.0 * float(J[(i, j)]) * dt / r,
                int(i),
                int(j),
            )

        theta_x = 2.0 * hx * dt / r
        for q in range(6):
            qc.rx(theta_x, q)

        if not np.isclose(hy, 0.0):
            theta_y = 2.0 * hy * dt / r
            qc.ry(theta_y, 4)
            qc.ry(theta_y, 5)


def build_measurement_circuit(candidate, angles, endpoint, setting):
    label, basis, measured = setting

    qreg = QuantumRegister(6, "q")
    creg = ClassicalRegister(len(measured), "m")
    qc = QuantumCircuit(
        qreg,
        creg,
        name=f"C1_{endpoint}_{label}",
    )

    W = int(candidate["window"])
    start = int(endpoint) - W + 1

    if start < 0:
        raise ValueError(f"Endpoint {endpoint} lacks W={W} history.")

    for k, idx in enumerate(range(start, int(endpoint) + 1)):
        append_rwp_step(
            qc,
            candidate,
            angles[idx],
            first_step=(k == 0),
        )

    for q, axis in basis.items():
        if axis == "X":
            qc.h(q)
        elif axis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif axis != "Z":
            raise ValueError(axis)

    for c, q in enumerate(measured):
        qc.measure(q, c)

    return qc


def mapping_from_row(row):
    return [
        str(row[f"q{i}_{label}"])
        for i, label in enumerate(["C", "D", "P", "H", "M1", "M2"])
    ]


def physical_layout_indices(backend, layout):
    return [
        int(backend.qubit_name_to_index(normalize_qb(q)))
        for q in layout
    ]


def compiled_op_qubits(circuit, inst):
    return [
        int(circuit.find_bit(q).index)
        for q in inst.qubits
    ]


def qname_from_compiled_index(backend, idx):
    try:
        return str(backend.index_to_qubit_name(int(idx)))
    except Exception:
        return f"QB{int(idx) + 1}"


def current_gate_fidelity(opname, physical_names, prof):
    qdf = prof["qdf"]

    if opname == "cz" and len(physical_names) == 2:
        p = normalize_pair(*physical_names)
        impl = prof["active_cz_impl"].get(p)
        v, _, _ = select_metric(
            qdf, "cz_fidelity", pair_text(p), impl
        )
        return v

    # PRX-native physical one-qubit rotations may be represented in Qiskit
    # as r / rx / ry after translation. Virtual z phases are excluded.
    if opname in {"r", "rx", "ry", "x", "y", "sx"} and len(physical_names) == 1:
        q = physical_names[0]
        v, _, _ = select_metric(
            qdf, "prx_fidelity", q, prof["prx_default"]
        )
        return v

    if opname == "measure" and len(physical_names) == 1:
        q = physical_names[0]
        v, _, _ = select_metric(
            qdf, "readout_fidelity", q, prof["measure_default"]
        )
        return v

    return np.nan


def calibration_nll(circuit, backend, prof):
    nll = 0.0
    missing = []
    n_cz = 0
    n_reset = 0

    for item in circuit.data:
        op = item.operation.name
        qidx = compiled_op_qubits(circuit, item)
        qnames = [qname_from_compiled_index(backend, i) for i in qidx]

        if op == "reset":
            n_reset += 1
            continue

        if op == "cz":
            n_cz += 1

        # Virtual/structural operations do not contribute to this cheap proxy.
        if op in {"rz", "delay", "barrier"}:
            continue

        if op not in {"cz", "r", "rx", "ry", "x", "y", "sx", "measure"}:
            continue

        f = current_gate_fidelity(op, qnames, prof)

        if not np.isfinite(f) or f <= 0.0 or f > 1.0:
            missing.append({
                "operation": op,
                "qubits": qnames,
            })
            continue

        nll += -math.log(max(float(f), 1e-15))

    return {
        "proxy_nll": float(nll),
        "proxy_success_probability": float(math.exp(-nll)),
        "missing_calibration_items": len(missing),
        "missing_json": json.dumps(missing),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "n_cz": int(n_cz),
        "n_reset": int(n_reset),
        "n_swap": int(circuit.count_ops().get("swap", 0)),
        "operations_json": json.dumps(
            {str(k): int(v) for k, v in circuit.count_ops().items()},
            sort_keys=True,
        ),
    }


def transpile_fixed(logical, backend, real_target, layout):
    idx = physical_layout_indices(backend, layout)

    tqc = transpile(
        logical,
        target=real_target,
        initial_layout=idx,
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
    )

    if int(tqc.count_ops().get("swap", 0)) != 0:
        raise RuntimeError(f"Unexpected SWAP for layout {layout}")

    return tqc


def workload_proxy(candidate, ref, pareto, prof):
    backend = prof["backend"]
    real_target = backend.get_real_target()

    rep_endpoint = int(
        ref["probe_endpoints"][len(ref["probe_endpoints"]) // 2]
    )
    settings = readout_settings(candidate["readout"])

    logical = {
        label: build_measurement_circuit(
            candidate,
            ref["angles"],
            rep_endpoint,
            setting,
        )
        for setting in settings
        for label in [setting[0]]
    }

    rows = []

    print()
    print("ACTUAL-WORKLOAD PROXY ON FRESH H6 CALIBRATION PARETO")
    print("-" * 120)

    for k, (_, row) in enumerate(pareto.iterrows(), start=1):
        layout = mapping_from_row(row)

        rec = {
            "embedding_id": int(row["embedding_id"]),
            "layout": json.dumps(layout),
            "proxy_compile_pass": True,
            "representative_training_endpoint": rep_endpoint,
            "calibration_pareto": True,
            "rule5_top3": False,
        }

        nll = 0.0
        max_depth = 0
        total_cz = 0
        total_reset = 0
        missing = 0
        errors = []

        try:
            for label, qc in logical.items():
                tqc = transpile_fixed(
                    qc, backend, real_target, layout
                )
                m = calibration_nll(tqc, backend, prof)

                rec[f"{label}_depth"] = m["depth"]
                rec[f"{label}_cz"] = m["n_cz"]
                rec[f"{label}_reset"] = m["n_reset"]
                rec[f"{label}_nll"] = m["proxy_nll"]

                nll += m["proxy_nll"]
                max_depth = max(max_depth, m["depth"])
                total_cz += m["n_cz"]
                total_reset += m["n_reset"]
                missing += m["missing_calibration_items"]

        except Exception as exc:
            rec["proxy_compile_pass"] = False
            errors.append(str(exc))

        rec.update({
            "actual_workload_proxy_nll": (
                float(nll)
                if rec["proxy_compile_pass"]
                else np.nan
            ),
            "actual_workload_proxy_success_probability": (
                float(math.exp(-nll))
                if rec["proxy_compile_pass"]
                else np.nan
            ),
            "actual_workload_max_setting_depth": max_depth,
            "actual_workload_total_cz_two_settings": total_cz,
            "actual_workload_total_reset_two_settings": total_reset,
            "missing_calibration_items": missing,
            "proxy_error": " | ".join(errors),
            "min_t1_us": row["min_t1_us"],
            "min_t2_echo_us": row["min_t2_echo_us"],
            "memory_min_t2_echo_us": row["memory_min_t2_echo_us"],
            "max_asym_readout_error": row["max_asym_readout_error"],
            "min_cz_fidelity": row["min_cz_fidelity"],
        })
        rows.append(rec)

        if k == 1 or k % 10 == 0 or k == len(pareto):
            print(f"  {k:3d}/{len(pareto)} layouts compiled")

    out = pd.DataFrame(rows)

    rule5_layouts = set()
    if OUT_R5.exists():
        r5 = pd.read_csv(OUT_R5)
        rule5_layouts = set(r5["layout"].astype(str))

    out["rule5_top3"] = out["layout"].astype(str).isin(rule5_layouts)

    out = out.sort_values(
        [
            "proxy_compile_pass",
            "missing_calibration_items",
            "actual_workload_proxy_nll",
            "actual_workload_max_setting_depth",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    out["proxy_rank"] = np.arange(1, len(out) + 1)
    out.to_csv(OUT_PROXY, index=False)

    return out


# =============================================================================
# IQM ERROR PROFILE: MATCH IQM'S OWN THERMAL+DEPOL SEMANTICS
# =============================================================================

def combined_gate_fidelity_1q(t1_ns, t2_ns, duration_ns, p):
    thermal = thermal_relaxation_error(
        float(t1_ns), float(t2_ns), float(duration_ns)
    )
    dep = depolarizing_error(float(p), 1)
    return float(
        average_gate_fidelity(
            thermal.compose(dep).to_quantumchannel()
        )
    )


def combined_gate_fidelity_2q(t1a, t2a, t1b, t2b, duration_ns, p):
    ta = thermal_relaxation_error(
        float(t1a), float(t2a), float(duration_ns)
    )
    tb = thermal_relaxation_error(
        float(t1b), float(t2b), float(duration_ns)
    )
    thermal = ta.tensor(tb)
    dep = depolarizing_error(float(p), 2)
    return float(
        average_gate_fidelity(
            thermal.compose(dep).to_quantumchannel()
        )
    )


def solve_p_for_fidelity(target, fidelity_func):
    target = float(target)
    f0 = float(fidelity_func(0.0))

    # If thermal relaxation alone is already worse than the reported benchmark
    # fidelity, no non-negative depolarizing parameter can repair it.
    # Use p=0 and record this explicitly.
    if f0 <= target:
        return 0.0, f0, "THERMAL_ALREADY_AT_OR_BELOW_TARGET"

    lo = 0.0
    hi = 1.0
    f_hi = float(fidelity_func(hi))

    if f_hi > target:
        # Should be rare. Keep the largest conservative valid p in [0,1].
        return hi, f_hi, "TARGET_BELOW_P1_RANGE"

    for _ in range(70):
        mid = 0.5 * (lo + hi)
        fm = float(fidelity_func(mid))

        # Fidelity decreases as p increases.
        if fm > target:
            lo = mid
        else:
            hi = mid

    p = 0.5 * (lo + hi)
    achieved = float(fidelity_func(p))
    return p, achieved, "SOLVED"


def build_iqm_fake_profile(prof):
    qdf = prof["qdf"]
    ddf = prof["ddf"]
    sqa = prof["sqa"]
    sqa_plain = prof["sqa_plain"]

    qubits = [str(q) for q in sqa_plain.get("qubits", [])]
    if not qubits:
        qubits = [str(q) for q in getattr(sqa, "qubits", [])]

    static_pairs = set()
    for locus in sqa_plain.get("connectivity", []):
        if len(locus) == 2:
            static_pairs.add(normalize_pair(locus[0], locus[1]))

    if not static_pairs:
        for locus in getattr(sqa, "connectivity", []):
            if len(locus) == 2:
                static_pairs.add(normalize_pair(locus[0], locus[1]))

    if not qubits or not static_pairs:
        raise RuntimeError(
            "Could not recover full Emerald static qubits/connectivity."
        )

    # Current raw per-qubit properties.
    t1s = {}
    t2s = {}
    prx_fid = {}
    e01 = {}
    e10 = {}
    clamp_rows = []

    for q in qubits:
        t1_s, _, _ = select_metric(qdf, "t1", q)
        t2_s, _, _ = select_metric(qdf, "t2_echo", q)

        if not np.isfinite(t1_s) or not np.isfinite(t2_s):
            raise RuntimeError(f"Missing T1/T2_echo for {q}.")

        t1_ns = float(t1_s) * 1e9
        t2_raw_ns = float(t2_s) * 1e9

        # Qiskit thermal_relaxation_error requires T2 <= 2*T1.
        t2_limit = 2.0 * t1_ns * (1.0 - 1e-12)
        t2_eff_ns = min(t2_raw_ns, t2_limit)

        t1s[q] = t1_ns
        t2s[q] = t2_eff_ns

        clamp_rows.append({
            "kind": "T2_CLAMP",
            "locus": q,
            "raw_t2_ns": t2_raw_ns,
            "effective_t2_ns": t2_eff_ns,
            "t1_ns": t1_ns,
            "clamped": bool(t2_eff_ns < t2_raw_ns),
        })

        f, _, status = select_metric(
            qdf, "prx_fidelity", q, prof["prx_default"]
        )
        if not np.isfinite(f):
            raise RuntimeError(
                f"Missing/ambiguous PRX fidelity for {q} ({status})."
            )
        prx_fid[q] = float(f)

        a, _, s1 = select_metric(
            qdf, "readout_error_0_to_1", q, prof["measure_default"]
        )
        b, _, s2 = select_metric(
            qdf, "readout_error_1_to_0", q, prof["measure_default"]
        )

        if not np.isfinite(a) or not np.isfinite(b):
            raise RuntimeError(
                f"Missing readout asymmetry for {q}: {s1}/{s2}"
            )

        e01[q] = float(a)
        e10[q] = float(b)

    # PRX duration: use implementation-matched live observations, but the
    # IQMErrorProfile schema stores one duration per gate type.
    prx_durations = []
    for q in qubits:
        d, _, status = select_duration(
            ddf, "prx", q, prof["prx_default"]
        )
        if not np.isfinite(d):
            raise RuntimeError(
                f"Missing/ambiguous PRX duration for {q} ({status})."
            )
        prx_durations.append(float(d))

    prx_duration_ns = float(np.median(prx_durations))

    # Calibrated CZ fidelities and implementation-matched durations.
    calibrated_cz_fid = {}
    calibrated_cz_dur = {}

    for p in prof["active_pairs"]:
        impl = prof["active_cz_impl"].get(p)

        f, _, fs = select_metric(
            qdf, "cz_fidelity", pair_text(p), impl
        )
        d, _, ds = select_duration(
            ddf, "cz", pair_text(p), impl
        )

        if np.isfinite(f):
            calibrated_cz_fid[p] = float(f)

        if np.isfinite(d):
            calibrated_cz_dur[p] = float(d)

    if not calibrated_cz_fid:
        raise RuntimeError("No current calibrated CZ fidelity values.")

    if not calibrated_cz_dur:
        raise RuntimeError("No current calibrated CZ duration values.")

    cz_duration_ns = float(np.median(list(calibrated_cz_dur.values())))
    worst_calibrated_cz_fidelity = float(
        min(calibrated_cz_fid.values())
    )

    # IQMFakeBackend validation requires the error profile to cover the full
    # StaticQuantumArchitecture connectivity. Current Resonance may expose
    # static edges without a current CZ calibration. Those edges are assigned
    # the WORST current calibrated fidelity only to complete the fake profile.
    # They are marked as imputed and are never eligible for Candidate-1 native
    # layouts, because layout enumeration used the CURRENT calibrated CZ graph.
    target_cz_fid = {}
    imputed_pairs = []

    for p in static_pairs:
        if p in calibrated_cz_fid:
            target_cz_fid[p] = calibrated_cz_fid[p]
        else:
            target_cz_fid[p] = worst_calibrated_cz_fidelity
            imputed_pairs.append(p)

    audit_rows = list(clamp_rows)

    # Solve PRX depolarizing parameters.
    prx_p = {}

    for q in qubits:
        target = prx_fid[q]

        def f1(p):
            return combined_gate_fidelity_1q(
                t1s[q],
                t2s[q],
                prx_duration_ns,
                p,
            )

        p, achieved, status = solve_p_for_fidelity(target, f1)
        prx_p[q] = float(p)

        audit_rows.append({
            "kind": "PRX",
            "locus": q,
            "target_fidelity": target,
            "thermal_only_fidelity": f1(0.0),
            "depolarizing_parameter": p,
            "achieved_fidelity": achieved,
            "abs_fidelity_residual": abs(achieved - target),
            "duration_ns": prx_duration_ns,
            "imputed": False,
            "solve_status": status,
        })

    # Solve CZ depolarizing parameters.
    cz_p = {}

    for p_locus in sorted(static_pairs, key=lambda x: (qb_num(x[0]), qb_num(x[1]))):
        qa, qb = p_locus
        target = target_cz_fid[p_locus]

        def f2(p):
            return combined_gate_fidelity_2q(
                t1s[qa],
                t2s[qa],
                t1s[qb],
                t2s[qb],
                cz_duration_ns,
                p,
            )

        pdep, achieved, status = solve_p_for_fidelity(target, f2)
        cz_p[p_locus] = float(pdep)

        audit_rows.append({
            "kind": "CZ",
            "locus": pair_text(p_locus),
            "target_fidelity": target,
            "thermal_only_fidelity": f2(0.0),
            "depolarizing_parameter": pdep,
            "achieved_fidelity": achieved,
            "abs_fidelity_residual": abs(achieved - target),
            "duration_ns": cz_duration_ns,
            "imputed": bool(p_locus in imputed_pairs),
            "solve_status": status,
        })

    readout = {
        q: {"0": e01[q], "1": e10[q]}
        for q in qubits
    }

    profile = IQMErrorProfile(
        t1s=t1s,
        t2s=t2s,
        single_qubit_gate_depolarizing_error_parameters={
            "prx": prx_p,
        },
        two_qubit_gate_depolarizing_error_parameters={
            "cz": cz_p,
        },
        single_qubit_gate_durations={
            "prx": prx_duration_ns,
        },
        two_qubit_gate_durations={
            "cz": cz_duration_ns,
        },
        readout_errors=readout,
        name=f"{BACKEND_NAME}_live_{prof['calibration_set_id']}",
    )

    fake = IQMFakeBackend(
        architecture=sqa,
        error_profile=profile,
        name=f"{BACKEND_NAME}_qrc_live_fake",
    )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_PROFILE, index=False)

    finite_resid = pd.to_numeric(
        audit.get("abs_fidelity_residual"),
        errors="coerce",
    ).dropna()

    manifest = {
        "backend": BACKEND_NAME,
        "calibration_set_id": prof["calibration_set_id"],
        "n_qubits": len(qubits),
        "static_connectivity_edges": len(static_pairs),
        "current_calibrated_cz_edges": len(calibrated_cz_fid),
        "profile_imputed_static_cz_edges": [
            pair_text(p) for p in imputed_pairs
        ],
        "imputation_policy": (
            "Worst current calibrated CZ fidelity is used only to satisfy "
            "IQMFakeBackend full-static-architecture validation. Candidate-1 "
            "native layout enumeration never uses uncalibrated edges."
        ),
        "prx_duration_ns_median": prx_duration_ns,
        "cz_duration_ns_median": cz_duration_ns,
        "t2_clamped_count": int(
            audit.loc[
                audit["kind"].eq("T2_CLAMP"),
                "clamped",
            ].fillna(False).astype(bool).sum()
        ),
        "max_fidelity_residual": (
            float(finite_resid.max())
            if len(finite_resid)
            else None
        ),
        "naive_one_minus_fidelity_used": False,
        "profile_semantics": (
            "depolarizing parameter numerically matched after thermal "
            "relaxation, consistent with IQMFakeBackend composition"
        ),
    }
    write_json(OUT_PROFILE_JSON, manifest)

    return fake, manifest


# =============================================================================
# COUNT -> FROZEN CANDIDATE-1 FEATURES
# =============================================================================

def clean_bits(key):
    return str(key).replace(" ", "")


def single_expectation(counts, cbit):
    total = float(sum(float(v) for v in counts.values()))
    if total <= 0:
        raise RuntimeError("Empty counts.")

    acc = 0.0

    for key, n in counts.items():
        bits = clean_bits(key)
        bit = int(bits[-1 - int(cbit)])
        acc += (1.0 if bit == 0 else -1.0) * float(n)

    return float(acc / total)


def parity_expectation(counts, cbits):
    total = float(sum(float(v) for v in counts.values()))
    if total <= 0:
        raise RuntimeError("Empty counts.")

    acc = 0.0

    for key, n in counts.items():
        bits = clean_bits(key)
        parity = 0

        for c in cbits:
            parity ^= int(bits[-1 - int(c)])

        acc += (1.0 if parity == 0 else -1.0) * float(n)

    return float(acc / total)


def candidate1_features(setting_counts):
    if "XXXXYX" not in setting_counts or "ZZZZZZ" not in setting_counts:
        raise RuntimeError(
            f"Expected XXXXYX/ZZZZZZ counts; got {list(setting_counts)}"
        )

    cx = setting_counts["XXXXYX"]
    cz = setting_counts["ZZZZZZ"]

    # Exact frozen order for XZinj_plus_YX45:
    # X0 X1 X2 X3 Z0 Z1 Z2 Z3 YX45
    return np.asarray([
        single_expectation(cx, 0),
        single_expectation(cx, 1),
        single_expectation(cx, 2),
        single_expectation(cx, 3),
        single_expectation(cz, 0),
        single_expectation(cz, 1),
        single_expectation(cz, 2),
        single_expectation(cz, 3),
        parity_expectation(cx, [4, 5]),
    ], dtype=float)


# =============================================================================
# IQMFAKE NOISY LAYOUT SCREEN
# =============================================================================


# =============================================================================
# REDUCED 6-QUBIT IQMFAKE SIMULATION
# =============================================================================

def _profile_cz_parameter(profile, qa, qb):
    """Return the frozen IQMErrorProfile CZ depolarizing parameter for a pair."""
    table = profile.two_qubit_gate_depolarizing_error_parameters["cz"]

    if (qa, qb) in table:
        return float(table[(qa, qb)])
    if (qb, qa) in table:
        return float(table[(qb, qa)])

    # Error-profile validation treats connectivity as unordered, but keep this
    # explicit so a missing pair can never be silently ignored.
    raise KeyError(
        f"IQMErrorProfile has no CZ parameter for physical pair ({qa}, {qb})."
    )


def build_reduced_layout_noise_model(fake_backend, layout):
    """
    Build a SIX-QUBIT Aer NoiseModel for one real Emerald physical layout.

    The numerical parameters are copied exactly from the already-built
    calibration-derived IQMErrorProfile:
      * physical T1/T2_echo
      * PRX depolarizing parameter
      * CZ depolarizing parameter
      * PRX/CZ duration
      * asymmetric readout errors

    This is mathematically the same thermal-relaxation + depolarizing
    composition used by IQMFakeBackend._create_noise_model(), but indices are
    remapped from the full 54-qubit Emerald backend to local qubits 0..5.

    This prevents Aer from allocating a 54-qubit noisy state for a circuit that
    physically acts on only six qubits.
    """
    profile = fake_backend.error_profile

    prx_duration = float(
        profile.single_qubit_gate_durations["prx"]
    )
    cz_duration = float(
        profile.two_qubit_gate_durations["cz"]
    )

    # IQMFakeBackend maps IQM PRX -> Qiskit's r gate and CZ -> cz.
    noise = NoiseModel(
        basis_gates=["r", "cz"]
    )

    # 1Q + readout errors.
    for local_q, physical_q in enumerate(layout):
        physical_q = str(physical_q)

        thermal = thermal_relaxation_error(
            float(profile.t1s[physical_q]),
            float(profile.t2s[physical_q]),
            prx_duration,
        )
        dep = depolarizing_error(
            float(
                profile
                .single_qubit_gate_depolarizing_error_parameters[
                    "prx"
                ][physical_q]
            ),
            1,
        )

        # Exact IQMFakeBackend composition order.
        noise.add_quantum_error(
            thermal.compose(dep),
            "r",
            [int(local_q)],
        )

        ro = profile.readout_errors[physical_q]
        probs = [
            [1.0 - float(ro["0"]), float(ro["0"])],
            [float(ro["1"]), 1.0 - float(ro["1"])],
        ]
        noise.add_readout_error(
            ReadoutError(probs),
            [int(local_q)],
        )

    # 2Q errors only on the actual H6 logical edges used by Candidate 1.
    for logical_a, logical_b in H6_EDGES:
        qa = str(layout[int(logical_a)])
        qb = str(layout[int(logical_b)])

        pdep = _profile_cz_parameter(
            profile,
            qa,
            qb,
        )

        # IQMFakeBackend adds both physical locus orders.
        for la, lb, pa, pb in (
            (logical_a, logical_b, qa, qb),
            (logical_b, logical_a, qb, qa),
        ):
            ta = thermal_relaxation_error(
                float(profile.t1s[pa]),
                float(profile.t2s[pa]),
                cz_duration,
            )
            tb = thermal_relaxation_error(
                float(profile.t1s[pb]),
                float(profile.t2s[pb]),
                cz_duration,
            )

            thermal_2q = ta.tensor(tb)
            dep_2q = depolarizing_error(
                pdep,
                2,
            )

            noise.add_quantum_error(
                thermal_2q.compose(dep_2q),
                "cz",
                [int(la), int(lb)],
            )

    return noise


def compress_physical_circuit_to_six_qubits(
    compiled,
    backend,
    layout,
):
    """
    Remove the 48 idle Emerald qubits from a strictly fixed-layout circuit.

    The input circuit was already compiled against backend.get_real_target()
    using the real physical layout and routing_method='none'. Therefore all
    non-structural quantum operations must act only on those six physical
    qubits.

    The returned circuit has exactly six qubits in logical/layout order:
        local 0 -> layout[0]
        ...
        local 5 -> layout[5]

    Classical bit indices are preserved, so the existing feature extractor is
    unchanged.
    """
    physical_indices = physical_layout_indices(
        backend,
        layout,
    )
    phys_to_local = {
        int(p): int(i)
        for i, p in enumerate(physical_indices)
    }

    reduced = QuantumCircuit(
        6,
        int(compiled.num_clbits),
        name=f"{compiled.name}_active6",
    )

    reduced.metadata = dict(
        compiled.metadata or {}
    )

    for item in compiled.data:
        opname = str(item.operation.name)

        qidx = [
            int(compiled.find_bit(q).index)
            for q in item.qubits
        ]
        cidx = [
            int(compiled.find_bit(c).index)
            for c in item.clbits
        ]

        # Barriers are semantically irrelevant for Aer and may span idle
        # physical qubits after transpilation.
        if opname == "barrier":
            continue

        # Ignore delay instructions on idle qubits. Candidate circuits are not
        # scheduled here, so this is normally absent.
        if qidx and any(
            q not in phys_to_local
            for q in qidx
        ):
            if opname == "delay":
                continue

            raise RuntimeError(
                "Strict fixed-layout circuit contains a non-structural "
                "operation outside the six selected physical qubits: "
                f"op={opname}, physical_indices={qidx}, layout={layout}"
            )

        local_qargs = [
            reduced.qubits[
                phys_to_local[q]
            ]
            for q in qidx
        ]
        local_cargs = [
            reduced.clbits[c]
            for c in cidx
        ]

        reduced.append(
            item.operation.copy(),
            local_qargs,
            local_cargs,
        )

    if reduced.num_qubits != 6:
        raise RuntimeError(
            f"Reduced circuit has {reduced.num_qubits} qubits, expected 6."
        )

    return reduced


def _run_aer_in_small_batches(
    simulator,
    circuits,
    shots,
    batch_size=8,
):
    """
    Run small circuit batches to avoid Windows/Aer memory spikes.

    Returns counts in the same order as input circuits.
    """
    counts = []

    for start in range(
        0,
        len(circuits),
        int(batch_size),
    ):
        stop = min(
            start + int(batch_size),
            len(circuits),
        )
        batch = circuits[start:stop]

        job = simulator.run(
            batch,
            shots=int(shots),
            seed_simulator=SEED_SIMULATOR + int(start),
        )
        result = job.result()

        for i in range(len(batch)):
            counts.append(
                result.get_counts(i)
            )

    if len(counts) != len(circuits):
        raise RuntimeError(
            f"Aer returned {len(counts)} count sets for "
            f"{len(circuits)} circuits."
        )

    return counts


def fake_screen(
    candidate,
    ref,
    prof,
    fake_backend,
    proxy,
    sim_pool_size,
    sim_shots,
    finalists_n,
):
    eligible = proxy[
        proxy["proxy_compile_pass"].astype(bool)
        & pd.to_numeric(
            proxy["missing_calibration_items"],
            errors="coerce",
        ).eq(0)
    ].copy()

    if len(eligible) < finalists_n:
        raise RuntimeError(
            f"Only {len(eligible)} workload-valid layouts; "
            f"need at least {finalists_n}."
        )

    pool = eligible.head(
        min(int(sim_pool_size), len(eligible))
    ).copy()

    backend = prof["backend"]
    real_target = backend.get_real_target()
    settings = readout_settings(
        candidate["readout"]
    )
    sim_endpoints = np.asarray(
        ref["sim_endpoints"],
        dtype=int,
    )

    ideal_by_endpoint = {
        int(ep): {
            "X": np.asarray(
                ref["X_sim_ideal"][i],
                dtype=float,
            ),
            "pred": float(
                ref["pred_sim_ideal"][i]
            ),
        }
        for i, ep in enumerate(sim_endpoints)
    }

    per_endpoint_rows = []
    summary_rows = []

    print()
    print("IQMFAKEBACKEND NOISY SCREEN — REDUCED ACTIVE-6 MODE")
    print("-" * 120)
    print(
        f"Layouts={len(pool)}, training endpoints={len(sim_endpoints)}, "
        f"settings/endpoint={len(settings)}, shots/setting={sim_shots}"
    )
    print(
        "Real Emerald circuits are first compiled on the 54-qubit target, "
        "then compressed to their six active physical qubits for Aer."
    )

    for rank, (_, prow) in enumerate(
        pool.iterrows(),
        start=1,
    ):
        layout = json.loads(
            str(prow["layout"])
        )
        layout_idx = physical_layout_indices(
            backend,
            layout,
        )

        logical_circuits = []
        metadata = []

        for endpoint in sim_endpoints:
            for setting in settings:
                logical_circuits.append(
                    build_measurement_circuit(
                        candidate,
                        ref["angles"],
                        int(endpoint),
                        setting,
                    )
                )
                metadata.append({
                    "endpoint": int(endpoint),
                    "setting": setting[0],
                })

        # Real Emerald compilation remains authoritative for native
        # decomposition/resource structure.
        compiled_full = transpile(
            logical_circuits,
            target=real_target,
            initial_layout=layout_idx,
            routing_method="none",
            optimization_level=OPT_LEVEL,
            seed_transpiler=SEED_TRANSPILE,
        )

        if not isinstance(
            compiled_full,
            list,
        ):
            compiled_full = [
                compiled_full
            ]

        if len(compiled_full) != len(
            logical_circuits
        ):
            raise RuntimeError(
                "Batch transpilation changed circuit count."
            )

        reduced = []

        for c in compiled_full:
            if int(
                c.count_ops().get(
                    "swap",
                    0,
                )
            ) != 0:
                raise RuntimeError(
                    f"SWAP appeared during fake screen for {layout}."
                )

            reduced.append(
                compress_physical_circuit_to_six_qubits(
                    c,
                    backend,
                    layout,
                )
            )

        # Layout-specific six-qubit noise model with EXACT parameters from
        # the current calibration-derived IQMErrorProfile.
        local_noise = build_reduced_layout_noise_model(
            fake_backend,
            layout,
        )

        simulator = AerSimulator(
            noise_model=local_noise,
            method="density_matrix",
        )

        # Small batches prevent abrupt Windows/Aer process termination.
        count_sets = _run_aer_in_small_batches(
            simulator,
            reduced,
            shots=int(sim_shots),
            batch_size=8,
        )

        counts_by_endpoint = defaultdict(dict)

        for meta, counts in zip(
            metadata,
            count_sets,
        ):
            counts_by_endpoint[
                meta["endpoint"]
            ][
                meta["setting"]
            ] = counts

        X_noisy = []
        X_ideal = []
        pred_noisy = []
        pred_ideal = []

        for endpoint in sim_endpoints:
            feat = candidate1_features(
                counts_by_endpoint[
                    int(endpoint)
                ]
            )

            ideal = ideal_by_endpoint[
                int(endpoint)
            ]["X"]
            p_ideal = ideal_by_endpoint[
                int(endpoint)
            ]["pred"]

            p_noisy = float(
                common.predict_scaled_ridge(
                    ref["model"],
                    ref["scaler"],
                    ref["keep"],
                    feat.reshape(
                        1,
                        -1,
                    ),
                )[0]
            )

            X_noisy.append(feat)
            X_ideal.append(ideal)
            pred_noisy.append(p_noisy)
            pred_ideal.append(p_ideal)

            per_endpoint_rows.append({
                "layout_proxy_rank": int(
                    prow["proxy_rank"]
                ),
                "layout": json.dumps(
                    layout
                ),
                "endpoint": int(endpoint),
                "pred_noisy": p_noisy,
                "pred_ideal": p_ideal,
                "prediction_error_noisy_minus_ideal": (
                    p_noisy
                    - p_ideal
                ),
                "feature_rmse": rmse(
                    feat,
                    ideal,
                ),
                **{
                    f"noisy_{name}": float(v)
                    for name, v
                    in zip(
                        ref["features"],
                        feat,
                    )
                },
                **{
                    f"ideal_{name}": float(v)
                    for name, v
                    in zip(
                        ref["features"],
                        ideal,
                    )
                },
            })

        X_noisy = np.asarray(
            X_noisy,
            dtype=float,
        )
        X_ideal = np.asarray(
            X_ideal,
            dtype=float,
        )
        pred_noisy = np.asarray(
            pred_noisy,
            dtype=float,
        )
        pred_ideal = np.asarray(
            pred_ideal,
            dtype=float,
        )

        summary_rows.append({
            "layout_proxy_rank": int(
                prow["proxy_rank"]
            ),
            "layout": json.dumps(
                layout
            ),
            "sim_pred_distortion_rmse": rmse(
                pred_noisy,
                pred_ideal,
            ),
            "sim_feature_rmse": rmse(
                X_noisy.ravel(),
                X_ideal.ravel(),
            ),
            "sim_prediction_corr_vs_ideal": corr(
                pred_noisy,
                pred_ideal,
            ),
            "actual_workload_proxy_nll": float(
                prow[
                    "actual_workload_proxy_nll"
                ]
            ),
            "actual_workload_max_setting_depth": int(
                prow[
                    "actual_workload_max_setting_depth"
                ]
            ),
            "actual_workload_total_cz_two_settings": int(
                prow[
                    "actual_workload_total_cz_two_settings"
                ]
            ),
            "min_t1_us": float(
                prow["min_t1_us"]
            ),
            "min_t2_echo_us": float(
                prow["min_t2_echo_us"]
            ),
            "memory_min_t2_echo_us": float(
                prow[
                    "memory_min_t2_echo_us"
                ]
            ),
            "max_asym_readout_error": float(
                prow[
                    "max_asym_readout_error"
                ]
            ),
            "min_cz_fidelity": float(
                prow[
                    "min_cz_fidelity"
                ]
            ),
            "rule5_top3": bool(
                prow["rule5_top3"]
            ),
            "n_sim_endpoints": int(
                len(sim_endpoints)
            ),
            "shots_per_setting": int(
                sim_shots
            ),
            "aer_active_qubits": 6,
            "aer_batch_size": 8,
        })

        # Incremental checkpoint: even an external interruption after this
        # point preserves completed layouts for diagnosis.
        pd.DataFrame(
            per_endpoint_rows
        ).to_csv(
            OUT_SIM_EP,
            index=False,
        )
        pd.DataFrame(
            summary_rows
        ).to_csv(
            OUT_SIM_SUM,
            index=False,
        )

        print(
            f"  {rank:2d}/{len(pool)} "
            f"layout={layout} "
            f"E_delta_y_sim="
            f"{summary_rows[-1]['sim_pred_distortion_rmse']:.6f}"
        )

    ep_df = pd.DataFrame(
        per_endpoint_rows
    )

    sim_df = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "sim_pred_distortion_rmse",
            "sim_feature_rmse",
            "actual_workload_proxy_nll",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    sim_df["fake_sim_rank"] = np.arange(
        1,
        len(sim_df) + 1,
    )

    ep_df.to_csv(
        OUT_SIM_EP,
        index=False,
    )
    sim_df.to_csv(
        OUT_SIM_SUM,
        index=False,
    )

    finalists = sim_df.head(
        int(finalists_n)
    ).copy()

    finalists[
        "qpu_probe_status"
    ] = "PENDING_NEXT_STEP"

    finalists[
        "qpu_probe_inputs_planned"
    ] = DEFAULT_QPU_PROBE_INPUTS

    finalists[
        "qpu_probe_ranking_metric"
    ] = (
        "RMS QPU-vs-ideal prediction distortion; "
        "true claims not used"
    )

    finalists.to_csv(
        OUT_FINALISTS,
        index=False,
    )

    return sim_df, finalists


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sim-pool",
        type=int,
        default=DEFAULT_SIM_POOL,
        help="Number of workload-proxy layouts entering IQMFake noisy screening.",
    )
    parser.add_argument(
        "--sim-inputs",
        type=int,
        default=DEFAULT_SIM_INPUTS,
        help="Deterministic 2022-2024 training endpoints used in IQMFake screening.",
    )
    parser.add_argument(
        "--sim-shots",
        type=int,
        default=DEFAULT_SIM_SHOTS,
        help="IQMFake shots per measurement setting.",
    )
    parser.add_argument(
        "--qpu-finalists",
        type=int,
        default=DEFAULT_QPU_FINALISTS,
        help="Number of layouts retained for the NEXT real-QPU probe.",
    )
    parser.add_argument(
        "--qpu-probe-inputs",
        type=int,
        default=DEFAULT_QPU_PROBE_INPUTS,
        help="Training-only endpoints reserved for the NEXT real-QPU probe.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    outputs = [
        OUT_ID,
        OUT_PROBES,
        OUT_Q,
        OUT_D,
        OUT_ALL,
        OUT_PARETO,
        OUT_R5,
        OUT_PROXY,
        OUT_PROFILE,
        OUT_PROFILE_JSON,
        OUT_SIM_EP,
        OUT_SIM_SUM,
        OUT_FINALISTS,
        OUT_MANIFEST,
    ]

    if args.overwrite:
        for p in outputs:
            if p.exists():
                p.unlink()

    if args.sim_pool < args.qpu_finalists:
        raise ValueError("--sim-pool must be >= --qpu-finalists.")
    if args.sim_inputs <= 0:
        raise ValueError("--sim-inputs must be positive.")
    if args.sim_shots <= 0:
        raise ValueError("--sim-shots must be positive.")
    if args.qpu_probe_inputs <= 0:
        raise ValueError("--qpu-probe-inputs must be positive.")

    print("=" * 132)
    print("WEEK 12.6A.1 — IQM CANDIDATE #1 PRE-QPU QRC-AWARE LAYOUT SCREEN")
    print("=" * 132)
    print("Backend: Emerald")
    print("NO REAL-QPU JOBS.")
    print("Physical-layout ranking is benchmark-blind to true claim targets.")
    print("2025 validation metrics are not computed here. 2026 is untouched.")
    print()

    # -------------------------------------------------------------------------
    # A. Candidate identity and ideal training reference
    # -------------------------------------------------------------------------
    candidate, source_row, identity = load_frozen_candidate1()

    print("FROZEN CANDIDATE #1")
    print("-" * 132)
    print(
        f"{EXPECTED_CANDIDATE_ID} | "
        f"H6 | W={candidate['window']} | r={candidate['r']} | "
        f"{candidate['readout']}"
    )
    print(
        f"alpha={candidate['alpha']:.6f} "
        f"dt={candidate['dt']:.6f} "
        f"hx={candidate['hx']:+.9f} "
        f"hy={candidate['hy']:+.9f}"
    )

    ref = build_training_reference(
        candidate,
        qpu_probe_inputs=args.qpu_probe_inputs,
        sim_inputs=args.sim_inputs,
    )

    identity.update({
        "selected_lambda_recomputed_training_only": ref["selected_lambda"],
        "cv_rmse_recomputed_training_only": ref["cv_rmse"],
        "cv_rmse_std_recomputed_training_only": ref["cv_rmse_std"],
        "n_qpu_probe_endpoints_reserved": len(ref["probe_endpoints"]),
        "n_fake_sim_endpoints": len(ref["sim_endpoints"]),
        "validation_2025_metric_computed": False,
        "test_2026_loaded": False,
    })
    write_json(OUT_ID, identity)

    print(
        f"Training-only CV reproduced: {ref['cv_rmse']:.6f} "
        f"(SD={ref['cv_rmse_std']:.6f}, lambda={ref['selected_lambda']})"
    )
    print(
        f"Reserved training-only probe endpoints: {len(ref['probe_endpoints'])}; "
        f"IQMFake subset: {len(ref['sim_endpoints'])}"
    )

    # -------------------------------------------------------------------------
    # B. Fresh live profiler
    # -------------------------------------------------------------------------
    print()
    print("FRESH EMERALD PROFILER")
    print("-" * 132)

    prof = fresh_profiler()

    print(f"Calibration set: {prof['calibration_set_id']}")
    print(
        f"Current calibrated CZ edges: {len(prof['active_pairs'])}"
    )
    print(
        f"Parsed live quality rows: {len(prof['qdf'])}; "
        f"duration rows: {len(prof['ddf'])}"
    )

    # -------------------------------------------------------------------------
    # C. Native H6 embeddings / Rule-5 current calibration screen
    # -------------------------------------------------------------------------
    all_emb, pareto, rule5 = profile_h6_embeddings(prof)

    print()
    print("FRESH H6 EMBEDDING SCREEN")
    print("-" * 132)
    print(f"Native H6 embeddings: {len(all_emb)}")
    print(f"Complete-metric embeddings: {int(all_emb['complete_metrics'].sum())}")
    print(f"Corrected four-objective Pareto: {len(pareto)}")
    print("Rule-5 top3 retained as DIAGNOSTIC only:")
    print(
        rule5[
            [
                "rule5_rank",
                "layout",
                "min_cz_fidelity",
                "min_t2_echo_us",
                "memory_min_t2_echo_us",
                "max_asym_readout_error",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # D. Actual Candidate-1 workload proxy
    # -------------------------------------------------------------------------
    proxy = workload_proxy(
        candidate,
        ref,
        pareto,
        prof,
    )

    valid_proxy = proxy[
        proxy["proxy_compile_pass"].astype(bool)
        & pd.to_numeric(
            proxy["missing_calibration_items"],
            errors="coerce",
        ).eq(0)
    ]

    if len(valid_proxy) < args.qpu_finalists:
        raise RuntimeError(
            f"Only {len(valid_proxy)} valid actual-workload layouts."
        )

    print()
    print("BEST ACTUAL-WORKLOAD PROXY LAYOUTS")
    print("-" * 132)
    print(
        valid_proxy[
            [
                "proxy_rank",
                "layout",
                "actual_workload_proxy_nll",
                "actual_workload_max_setting_depth",
                "actual_workload_total_cz_two_settings",
                "min_cz_fidelity",
                "min_t2_echo_us",
                "memory_min_t2_echo_us",
                "max_asym_readout_error",
                "rule5_top3",
            ]
        ].head(args.sim_pool).to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # E. Fresh IQMFakeBackend profile
    # -------------------------------------------------------------------------
    print()
    print("BUILD CURRENT IQMFAKEBACKEND PROFILE")
    print("-" * 132)

    fake_backend, fake_manifest = build_iqm_fake_profile(prof)

    print(
        f"Profile built: {fake_manifest['n_qubits']} qubits, "
        f"{fake_manifest['static_connectivity_edges']} static edges."
    )
    print(
        f"Current calibrated CZ edges: "
        f"{fake_manifest['current_calibrated_cz_edges']}"
    )
    print(
        f"Static edges imputed only for fake-profile completeness: "
        f"{len(fake_manifest['profile_imputed_static_cz_edges'])}"
    )
    print(
        f"T2 clamps required by T2<=2T1: "
        f"{fake_manifest['t2_clamped_count']}"
    )
    print(
        f"Max PRX/CZ fidelity reconstruction residual: "
        f"{fake_manifest['max_fidelity_residual']}"
    )

    # -------------------------------------------------------------------------
    # F. IQMFake noisy layout screen
    # -------------------------------------------------------------------------
    sim_summary, finalists = fake_screen(
        candidate=candidate,
        ref=ref,
        prof=prof,
        fake_backend=fake_backend,
        proxy=proxy,
        sim_pool_size=args.sim_pool,
        sim_shots=args.sim_shots,
        finalists_n=args.qpu_finalists,
    )

    print()
    print("=" * 132)
    print("CANDIDATE #1 — TWO LAYOUTS RETAINED FOR NEXT REAL-QPU TRAINING PROBE")
    print("=" * 132)
    print(
        finalists[
            [
                "fake_sim_rank",
                "layout",
                "sim_pred_distortion_rmse",
                "sim_feature_rmse",
                "sim_prediction_corr_vs_ideal",
                "actual_workload_proxy_nll",
                "min_cz_fidelity",
                "min_t2_echo_us",
                "memory_min_t2_echo_us",
                "max_asym_readout_error",
                "rule5_top3",
            ]
        ].to_string(index=False)
    )

    manifest = {
        "stage": "12.6A.1",
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "backend": BACKEND_NAME,
        "calibration_set_id": prof["calibration_set_id"],
        "selection_chain": [
            "fresh_profiler",
            "native_H6_embeddings",
            "four_objective_calibration_Pareto",
            "actual_candidate_workload_proxy",
            "IQMFakeBackend_current_noise_screen",
            "retain_two_for_next_real_QPU_training_probe",
        ],
        "rule5_top3_forced_into_simulation_pool": False,
        "qiskit_default_layout_forced_into_pool": False,
        "true_targets_used_for_layout_ranking": False,
        "training_scope": "2022-2024 only",
        "qpu_probe_endpoints_reserved": int(len(ref["probe_endpoints"])),
        "fake_sim_endpoints": int(len(ref["sim_endpoints"])),
        "fake_sim_shots_per_setting": int(args.sim_shots),
        "fake_sim_pool_size": int(min(args.sim_pool, len(valid_proxy))),
        "qpu_finalists": int(len(finalists)),
        "real_qpu_jobs_submitted": 0,
        "validation_2025_metric_computed": False,
        "test_2026_loaded": False,
        "iqm_fake_profile": fake_manifest,
        "next_step": (
            "12.6A.2: interleaved real-Emerald QPU probe of exactly these "
            "two layouts on the reserved 100 deterministic 2022-2024 "
            "training-only endpoints; rank by RMS QPU-vs-ideal prediction "
            "distortion; then freeze one layout before full 2025."
        ),
    }
    write_json(OUT_MANIFEST, manifest)

    print()
    print("=" * 132)
    print("12.6A.1 COMPLETE — LOCAL PRE-QPU SCREEN ONLY")
    print("=" * 132)
    print("NO REAL-QPU JOB WAS SUBMITTED.")
    print("2025 was not used for layout ranking; no 2025 metric was computed.")
    print("2026 remains untouched.")
    print()
    print("Key outputs:")
    for path in [
        OUT_ID,
        OUT_PROBES,
        OUT_ALL,
        OUT_PARETO,
        OUT_R5,
        OUT_PROXY,
        OUT_PROFILE,
        OUT_PROFILE_JSON,
        OUT_SIM_SUM,
        OUT_FINALISTS,
        OUT_MANIFEST,
    ]:
        print(f"  {path}")
    print()
    print(
        "STOP HERE. Send the console output and qpu_finalists table "
        "before running any real Emerald QPU job."
    )


if __name__ == "__main__":
    main()
