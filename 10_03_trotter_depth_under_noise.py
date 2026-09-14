#!/usr/bin/env python
"""
10.03 — Trotter depth under realistic Kingston noise
====================================================

Scientific question
-------------------
Does realistic hardware noise make a shallower Trotter depth preferable?

We test:
    r in {1, 2, 3}

while freezing:
    topology      = H3
    protocol      = RWP
    window        = 4
    alpha         = 0.25
    dt            = 1.6
    hx            = 0.777492389747...
    hy            = -0.293484785758...
    J couplings   = frozen Week-9/10 reference values
    readout       = XZinj_dropX3_plus_YX45
    physical map  = Week-10.1 Kingston Rule-5 rank-1 layout

IMPORTANT:
Changing r changes the reservoir features. Therefore, for EACH r we:
    1. recompute ideal 2022-2025 QRC features,
    2. re-select Ridge lambda using TRAINING-ONLY CV,
    3. fit the Ridge/scaler on ideal 2022-2024 training features,
    4. freeze that readout for all noisy variants of that r.

We DO NOT re-optimize:
    J, hx, hy, alpha, W, topology, protocol, readout, physical embedding.

Primary regimes per r
---------------------
A_IDEAL_DIRECT
    Logical ideal reference, exact expectation values.

N4_ALL_GATE_DIRECT
    Backend-calibrated 1Q + 2Q gate noise, no thermal relaxation,
    no finite-shot sampling, no readout error.

N5_FULL_QUANTUM_DIRECT
    Backend-calibrated gate noise + T1/T2 thermal relaxation,
    no finite-shot sampling, no readout error.

S3_FULL_NOISE_SAMPLED
    Backend-calibrated gate noise + T1/T2 + readout assignment error
    + finite-shot sampling. Default: 1024 shots/setting, 3 repetitions.

A/B audit
---------
For every r, the transpiled ideal physical circuit is also evaluated with
Aer EstimatorV2. It must reproduce A to numerical precision:
    feature RMSE(A, B) < 1e-6
otherwise the script aborts.

Selection rule
--------------
Primary practical criterion = S3 full-noise sampled RMSE.

To preserve the project's no-arbitrary-weighted-score principle, the script
also reports a standard one-SE resource rule:

    1. find the r with the lowest mean S3 RMSE;
    2. threshold = best_mean + best_standard_error;
    3. choose the SHALLOWEST r whose mean S3 RMSE is within that threshold.

This is reported as a candidate recommendation, not as an automatic scientific
conclusion. The full RMSE/resource table must still be inspected.

Calibration semantics
---------------------
The physical layout is frozen from Week 10.1, but the Kingston calibration is
refreshed ONCE at the beginning of this run. The same backend snapshot/noise
models are then used for r=1,2,3, ensuring a fair within-run comparison.

Aer-memory safeguard
--------------------
Simulation circuits are intentionally NOT scheduled with ALAP before Aer.
Scheduling a circuit against the 156-qubit Kingston target can insert delays on
otherwise idle physical qubits, making Aer treat the whole backend register as
active and causing an impossible 156-qubit density-matrix allocation.
Resource durations are obtained with circuit.estimate_duration(backend.target)
instead, so scheduling is unnecessary for the metrics used in this step.

NO QPU JOBS ARE SUBMITTED.
NO ERROR MITIGATION.
2026 remains untouched.

Required files
--------------
ibm_account.py
09_04_rwp_common.py

results/09_04c_rwp_all_trials.csv
results/10_01_kingston_fresh_top3_embeddings.csv
plus files already required by 09_04_rwp_common.py.

Outputs
-------
results/10_03_reference.json
results/10_03_trotter_ideal_training_summary.csv
results/10_03_trotter_regime_runs.csv
results/10_03_trotter_regime_summary.csv
results/10_03_trotter_feature_metrics.csv
results/10_03_trotter_predictions.csv
results/10_03_trotter_resource_metrics.csv
results/10_03_trotter_selection.json

Full run
--------
python 10_03_trotter_depth_under_noise.py --shots 1024 --sampling-reps 3

Quick plumbing check
--------------------
python 10_03_trotter_depth_under_noise.py --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

from ibm_account import get_service


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
TRIAL_FILE = RESULTS / "09_04c_rwp_all_trials.csv"
TOP3_FILE = RESULTS / "10_01_kingston_fresh_top3_embeddings.csv"

BACKEND_NAME = "ibm_kingston"

TOPOLOGY = "H3"
WINDOW = 4
SEARCH_ID = "alpha_0.25"
READOUT = "XZinj_dropX3_plus_YX45"

R_VALUES = [1, 2, 3]

FEATURES = [
    "X0", "X1", "X2",
    "Z0", "Z1", "Z2", "Z3",
    "YX45",
]

SETTINGS = {
    "XXXZYX": {
        "basis": {0: "X", 1: "X", 2: "X", 3: "Z", 4: "Y", 5: "X"},
    },
    "ZZZZZZ": {
        "basis": {q: "Z" for q in range(6)},
    },
}

OPT_LEVEL = 1
SEED_TRANSPILE = 42
DEFAULT_SHOTS = 1024
DEFAULT_SAMPLING_REPS = 3
DEFAULT_BATCH = 80
DEFAULT_SIM_SEED = 20260912


# =============================================================================
# IMPORT PROJECT COMMON CODE
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load {path}")

    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_1003_common")


# =============================================================================
# BASIC HELPERS
# =============================================================================

def rmse(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p)))


def bias(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(p - y))


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def json_safe(x):
    if x is None:
        return None

    if isinstance(x, (str, int, float, bool)):
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    if isinstance(x, np.generic):
        return json_safe(x.item())

    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]

    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            pass

    return str(x)


def parse_layout(value):
    if isinstance(value, str):
        return [int(x) for x in json.loads(value)]
    return [int(x) for x in value]


def clone_candidate(candidate):
    try:
        return common._clone_candidate(candidate)
    except Exception:
        out = dict(candidate)
        out["J"] = dict(candidate["J"])
        return out


# =============================================================================
# RECONSTRUCT FROZEN REFERENCE + PHYSICAL LAYOUT
# =============================================================================

def reconstruct_reference_candidate():
    if not TRIAL_FILE.exists():
        raise FileNotFoundError(TRIAL_FILE)

    trials = pd.read_csv(TRIAL_FILE)

    mask = (
        (trials["topology"].astype(str) == TOPOLOGY)
        & (pd.to_numeric(trials["window"], errors="coerce") == WINDOW)
        & (trials["trial_search_id"].astype(str) == SEARCH_ID)
        & (trials["readout"].astype(str) == READOUT)
    )

    rows = trials.loc[mask].copy()

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one frozen Stage-C reference row; found {len(rows)}."
        )

    source_row = rows.iloc[0]

    baselines = common.load_topology_baselines([TOPOLOGY])
    base = common._clone_candidate(baselines[TOPOLOGY])
    base["candidate_id"] = f"{TOPOLOGY}_W{WINDOW:02d}_Aparent"
    base["window"] = WINDOW

    regenerated = {
        str(c["search_id"]): c
        for c in common.stage_c_candidates(base)
    }

    if SEARCH_ID not in regenerated:
        raise RuntimeError(
            f"Could not regenerate Stage-C search_id={SEARCH_ID}."
        )

    candidate = regenerated[SEARCH_ID]

    checks = {
        "alpha": np.isclose(candidate["alpha"], source_row["alpha"]),
        "dt": np.isclose(candidate["dt"], source_row["dt"]),
        "hx": np.isclose(candidate["hx"], source_row["hx"]),
        "hy": np.isclose(candidate["hy"], source_row["hy"]),
    }

    if not all(checks.values()):
        raise RuntimeError(
            "Frozen reference reconstruction audit failed: "
            + ", ".join(k for k, v in checks.items() if not v)
        )

    # r is intentionally varied in 10.3, but source r must still be r=3.
    if int(candidate["r"]) != 3:
        raise RuntimeError(
            f"Expected frozen source candidate r=3, got r={candidate['r']}."
        )

    return candidate, source_row


def load_rank1_layout():
    if not TOP3_FILE.exists():
        raise FileNotFoundError(TOP3_FILE)

    df = pd.read_csv(TOP3_FILE)

    g = df[
        pd.to_numeric(
            df["rule5_rank_current"],
            errors="coerce",
        ) == 1
    ].copy()

    if len(g) != 1:
        raise RuntimeError(
            f"Expected one Week-10.1 Rule-5 rank-1 row; found {len(g)}."
        )

    row = g.iloc[0]
    return parse_layout(row["layout"]), row


# =============================================================================
# IDEAL R-SPECIFIC FEATURES + TRAINING-ONLY RIDGE
# =============================================================================

def ideal_reference_for_r(candidate):
    """
    Recompute ideal reservoir features for THIS r and fit THIS r's own
    training-only Ridge readout.
    """
    work_tv, cols, y_all = common.load_train_validation()

    angles = common.make_angles(
        work_tv,
        cols,
        float(candidate["alpha"]),
    )

    channels = common.build_channels(candidate, angles)

    endpoints, master = common.rwp_master_feature_bank_from_channels(
        channels,
        WINDOW,
    )

    X = master[FEATURES].to_numpy(dtype=float)
    y_endpoint = y_all[endpoints]

    train_mask = endpoints < common.N_TRAIN
    val_mask = endpoints >= common.N_TRAIN

    X_train = X[train_mask]
    y_train = y_endpoint[train_mask]

    X_val = X[val_mask]
    y_val = y_endpoint[val_mask]
    val_endpoints = endpoints[val_mask]

    if len(y_val) != common.N_VAL:
        raise RuntimeError(
            f"Expected {common.N_VAL} validation endpoints; got {len(y_val)}."
        )

    selected_lambda, folds, summary = common.select_lambda_training_only(
        X_train,
        y_train,
        n_splits=5,
    )

    model, scaler, keep = common.fit_scaled_ridge(
        X_train,
        y_train,
        selected_lambda,
    )

    pred_val = common.predict_scaled_ridge(
        model,
        scaler,
        keep,
        X_val,
    )

    return {
        "angles": angles,
        "val_endpoints": val_endpoints,
        "X_train": X_train,
        "y_train": y_train,
        "X_val_ideal": X_val,
        "y_val": y_val,
        "pred_val_ideal": pred_val,
        "model": model,
        "scaler": scaler,
        "keep": keep,
        "selected_lambda": float(selected_lambda),
        "cv_rmse": float(summary.iloc[0]["cv_rmse_mean"]),
        "cv_sd": float(summary.iloc[0]["cv_rmse_std"]),
        "ideal_rmse": rmse(y_val, pred_val),
        "ideal_mae": mae(y_val, pred_val),
        "ideal_bias": bias(y_val, pred_val),
        "train_target_sd": float(np.std(y_train, ddof=0)),
    }


# =============================================================================
# PARAMETERIZED RWP CIRCUITS
# =============================================================================

def append_rwp_step_parameterized(qc, candidate, angle_params, first_step):
    if not first_step:
        for q in range(4):
            qc.reset(q)

    for q in range(4):
        qc.ry(angle_params[q], q)

    r = int(candidate["r"])
    dt = float(candidate["dt"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])
    J = candidate["J"]

    for _ in range(r):
        for i, j in common.TOPOLOGY_EDGES[TOPOLOGY]:
            qc.rzz(
                2.0 * float(J[(i, j)]) * dt / r,
                i,
                j,
            )

        theta_x = 2.0 * hx * dt / r
        for q in range(6):
            qc.rx(theta_x, q)

        if not np.isclose(hy, 0.0):
            theta_y = 2.0 * hy * dt / r
            qc.ry(theta_y, 4)
            qc.ry(theta_y, 5)


def build_parameterized_core_circuit(candidate):
    qreg = QuantumRegister(6, "q")
    qc = QuantumCircuit(qreg, name=f"H3_W4_r{int(candidate['r'])}_core")
    pars = ParameterVector("theta", WINDOW * 4)

    for step in range(WINDOW):
        step_params = [
            pars[step * 4 + q]
            for q in range(4)
        ]

        append_rwp_step_parameterized(
            qc,
            candidate,
            step_params,
            first_step=(step == 0),
        )

    return qc, pars


def build_parameterized_measurement_circuit(candidate, setting_label):
    qreg = QuantumRegister(6, "q")
    creg = ClassicalRegister(6, "m")

    qc = QuantumCircuit(
        qreg,
        creg,
        name=f"H3_W4_r{int(candidate['r'])}_{setting_label}",
    )

    pars = ParameterVector("theta", WINDOW * 4)

    for step in range(WINDOW):
        step_params = [
            pars[step * 4 + q]
            for q in range(4)
        ]

        append_rwp_step_parameterized(
            qc,
            candidate,
            step_params,
            first_step=(step == 0),
        )

    for q, axis in SETTINGS[setting_label]["basis"].items():
        if axis == "X":
            qc.h(q)
        elif axis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif axis == "Z":
            pass
        else:
            raise ValueError(axis)

    for q in range(6):
        qc.measure(q, q)

    return qc, pars


def bind_endpoint_circuit(
    transpiled_param_circuit,
    params,
    angles,
    endpoint,
):
    start = int(endpoint) - WINDOW + 1

    if start < 0:
        raise ValueError("Insufficient rewind history.")

    values = {}

    for step, idx in enumerate(range(start, int(endpoint) + 1)):
        for q in range(4):
            values[params[step * 4 + q]] = float(angles[idx, q])

    return transpiled_param_circuit.assign_parameters(
        values,
        inplace=False,
    )


# =============================================================================
# OBSERVABLES
# =============================================================================

def logical_sparse_observable(feature):
    if feature == "X0":
        return SparsePauliOp.from_sparse_list(
            [("X", [0], 1.0)], num_qubits=6
        )
    if feature == "X1":
        return SparsePauliOp.from_sparse_list(
            [("X", [1], 1.0)], num_qubits=6
        )
    if feature == "X2":
        return SparsePauliOp.from_sparse_list(
            [("X", [2], 1.0)], num_qubits=6
        )
    if feature == "Z0":
        return SparsePauliOp.from_sparse_list(
            [("Z", [0], 1.0)], num_qubits=6
        )
    if feature == "Z1":
        return SparsePauliOp.from_sparse_list(
            [("Z", [1], 1.0)], num_qubits=6
        )
    if feature == "Z2":
        return SparsePauliOp.from_sparse_list(
            [("Z", [2], 1.0)], num_qubits=6
        )
    if feature == "Z3":
        return SparsePauliOp.from_sparse_list(
            [("Z", [3], 1.0)], num_qubits=6
        )
    if feature == "YX45":
        return SparsePauliOp.from_sparse_list(
            [("YX", [4, 5], 1.0)], num_qubits=6
        )
    raise KeyError(feature)


# =============================================================================
# COUNTS -> FEATURES
# =============================================================================

def clean_bits(key):
    return str(key).replace(" ", "")


def single_exp_from_counts(counts, cbit):
    total = float(sum(counts.values()))
    acc = 0.0

    for key, n in counts.items():
        bits = clean_bits(key)
        bit = int(bits[-1 - int(cbit)])
        acc += (1.0 if bit == 0 else -1.0) * float(n)

    return acc / total


def parity_exp_from_counts(counts, cbits):
    total = float(sum(counts.values()))
    acc = 0.0

    for key, n in counts.items():
        bits = clean_bits(key)
        parity = 0

        for cbit in cbits:
            parity ^= int(bits[-1 - int(cbit)])

        acc += (1.0 if parity == 0 else -1.0) * float(n)

    return acc / total


def combine_feature_pair(counts_a, counts_z):
    return np.array(
        [
            single_exp_from_counts(counts_a, 0),
            single_exp_from_counts(counts_a, 1),
            single_exp_from_counts(counts_a, 2),
            single_exp_from_counts(counts_z, 0),
            single_exp_from_counts(counts_z, 1),
            single_exp_from_counts(counts_z, 2),
            single_exp_from_counts(counts_z, 3),
            parity_exp_from_counts(counts_a, [4, 5]),
        ],
        dtype=float,
    )


# =============================================================================
# BACKEND / RESOURCE HELPERS
# =============================================================================

def qubit_properties_single(backend, q):
    """
    Robustly retrieve BackendV2 qubit T1/T2 properties.
    Values returned by Qiskit are expected in seconds.
    """
    try:
        prop = backend.qubit_properties(int(q))
        if isinstance(prop, (list, tuple)):
            prop = prop[0]
        return prop
    except Exception:
        pass

    try:
        return backend.target.qubit_properties[int(q)]
    except Exception:
        return None


def min_layout_t1_t2_us(backend, layout):
    t1_us = []
    t2_us = []

    for q in layout:
        prop = qubit_properties_single(backend, q)

        if prop is None:
            continue

        t1 = getattr(prop, "t1", None)
        t2 = getattr(prop, "t2", None)

        if t1 is not None and np.isfinite(float(t1)):
            t1_us.append(float(t1) * 1e6)

        if t2 is not None and np.isfinite(float(t2)):
            t2_us.append(float(t2) * 1e6)

    return (
        float(np.min(t1_us)) if t1_us else np.nan,
        float(np.min(t2_us)) if t2_us else np.nan,
    )


def circuit_duration_us(circuit, backend):
    try:
        return float(
            circuit.estimate_duration(
                backend.target,
                unit="s",
            )
        ) * 1e6
    except Exception:
        return np.nan


def prepare_physical_circuits(
    backend,
    candidate,
    layout,
):
    logical_core, core_params = build_parameterized_core_circuit(
        candidate
    )

    transpiled_core = transpile(
        logical_core,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
    )

    backend.check_faulty(transpiled_core)

    if int(transpiled_core.count_ops().get("swap", 0)) != 0:
        raise RuntimeError(
            f"r={candidate['r']} introduced SWAPs in the frozen layout."
        )

    mapped_observables = [
        logical_sparse_observable(feature).apply_layout(
            transpiled_core.layout
        )
        for feature in FEATURES
    ]

    # IMPORTANT: keep the circuit unscheduled for Aer simulation.
    # Scheduling with ALAP on a 156-qubit backend can insert delay instructions
    # on otherwise idle physical qubits and defeat Aer qubit truncation.
    # The circuit may still have backend-width registers, but only the mapped
    # six-qubit region is operationally active.

    transpiled_meas = {}
    meas_params = {}

    for setting_label in SETTINGS:
        logical, pars = build_parameterized_measurement_circuit(
            candidate,
            setting_label,
        )

        tqc = transpile(
            logical,
            backend=backend,
            initial_layout=layout,
            routing_method="none",
            optimization_level=OPT_LEVEL,
            seed_transpiler=SEED_TRANSPILE,
            scheduling_method="alap",
        )

        backend.check_faulty(tqc)

        if int(tqc.count_ops().get("swap", 0)) != 0:
            raise RuntimeError(
                f"r={candidate['r']} introduced SWAPs in {setting_label}."
            )

        transpiled_meas[setting_label] = tqc
        meas_params[setting_label] = pars

    min_t1_us, min_t2_us = min_layout_t1_t2_us(
        backend,
        layout,
    )

    setting_rows = []

    for label, circ in transpiled_meas.items():
        ops = {
            str(k): int(v)
            for k, v in circ.count_ops().items()
        }

        duration_us = circuit_duration_us(
            circ,
            backend,
        )

        setting_rows.append({
            "r": int(candidate["r"]),
            "setting": label,
            "depth": int(circ.depth()),
            "size": int(circ.size()),
            "n_cz": int(ops.get("cz", 0)),
            "n_swap": int(ops.get("swap", 0)),
            "n_reset": int(ops.get("reset", 0)),
            "n_measure": int(ops.get("measure", 0)),
            "duration_us": duration_us,
        })

    setting_df = pd.DataFrame(setting_rows)

    core_ops = {
        str(k): int(v)
        for k, v in transpiled_core.count_ops().items()
    }

    core_duration = circuit_duration_us(
        transpiled_core,
        backend,
    )

    max_setting_duration = float(
        setting_df["duration_us"].max()
    )

    sum_setting_duration = float(
        setting_df["duration_us"].sum()
    )

    # Coherence exposure should use the longest single circuit execution,
    # not the sum of two separately executed measurement settings.
    R_T1 = (
        max_setting_duration / min_t1_us
        if np.isfinite(max_setting_duration)
        and np.isfinite(min_t1_us)
        and min_t1_us > 0
        else np.nan
    )

    R_T2 = (
        max_setting_duration / min_t2_us
        if np.isfinite(max_setting_duration)
        and np.isfinite(min_t2_us)
        and min_t2_us > 0
        else np.nan
    )

    summary = {
        "r": int(candidate["r"]),
        "layout": json.dumps(layout),
        "core_depth": int(transpiled_core.depth()),
        "core_n_cz": int(core_ops.get("cz", 0)),
        "core_n_swap": int(core_ops.get("swap", 0)),
        "core_duration_us": core_duration,
        "n_settings": int(len(setting_df)),
        "max_setting_depth": int(setting_df["depth"].max()),
        "sum_setting_depth": int(setting_df["depth"].sum()),
        "feature_vector_n_cz": int(setting_df["n_cz"].sum()),
        "feature_vector_n_swap": int(setting_df["n_swap"].sum()),
        "max_setting_duration_us": max_setting_duration,
        "feature_vector_duration_sum_us": sum_setting_duration,
        "min_layout_t1_us": min_t1_us,
        "min_layout_t2_us": min_t2_us,
        "max_setting_time_over_min_t1": R_T1,
        "max_setting_time_over_min_t2": R_T2,
    }

    return {
        "core": transpiled_core,
        "core_params": core_params,
        "mapped_observables": mapped_observables,
        "meas": transpiled_meas,
        "meas_params": meas_params,
        "setting_resources": setting_df,
        "resource_summary": summary,
    }


# =============================================================================
# DIRECT EXPECTATION SIMULATION
# =============================================================================

def run_direct_regime(
    noise_model,
    physical,
    ideal,
    batch_size,
):
    if noise_model is None:
        backend_options = {
            "method": "density_matrix",
            "enable_truncation": True,
        }
    else:
        backend_options = {
            "method": "density_matrix",
            "noise_model": noise_model,
            "enable_truncation": True,
        }

    estimator = AerEstimatorV2(
        options={
            "default_precision": 0.0,
            "backend_options": backend_options,
        }
    )

    bound_cores = [
        bind_endpoint_circuit(
            physical["core"],
            physical["core_params"],
            ideal["angles"],
            int(endpoint),
        )
        for endpoint in ideal["val_endpoints"]
    ]

    X = np.empty(
        (len(bound_cores), len(FEATURES)),
        dtype=float,
    )

    for start in range(0, len(bound_cores), int(batch_size)):
        stop = min(
            start + int(batch_size),
            len(bound_cores),
        )

        pubs = [
            (
                bound_cores[i],
                physical["mapped_observables"],
            )
            for i in range(start, stop)
        ]

        result = estimator.run(
            pubs,
            precision=0.0,
        ).result()

        for local_i, pub_result in enumerate(result):
            evs = np.asarray(
                pub_result.data.evs,
                dtype=float,
            ).reshape(-1)

            if evs.size != len(FEATURES):
                raise RuntimeError(
                    f"Unexpected estimator EV shape {evs.shape}."
                )

            X[start + local_i, :] = evs

    return X


# =============================================================================
# FINITE-SHOT S3 SIMULATION
# =============================================================================

def build_bound_measured_circuits(
    physical,
    ideal,
):
    circuits = []
    metadata = []

    for endpoint in ideal["val_endpoints"]:
        for setting_label in SETTINGS:
            bound = bind_endpoint_circuit(
                physical["meas"][setting_label],
                physical["meas_params"][setting_label],
                ideal["angles"],
                int(endpoint),
            )

            circuits.append(bound)
            metadata.append(
                (int(endpoint), setting_label)
            )

    return circuits, metadata


def run_sampled_regime(
    noise_model,
    physical,
    ideal,
    shots,
    batch_size,
    seed,
):
    simulator = AerSimulator(
        noise_model=noise_model,
        enable_truncation=True,
    )

    circuits, metadata = build_bound_measured_circuits(
        physical,
        ideal,
    )

    counts_by_endpoint = {
        int(e): {}
        for e in ideal["val_endpoints"]
    }

    for start in range(0, len(circuits), int(batch_size)):
        stop = min(
            start + int(batch_size),
            len(circuits),
        )

        sub = circuits[start:stop]
        submeta = metadata[start:stop]

        result = simulator.run(
            sub,
            shots=int(shots),
            seed_simulator=int(seed + start),
        ).result()

        for local_i, (endpoint, setting_label) in enumerate(submeta):
            counts_by_endpoint[endpoint][setting_label] = (
                result.get_counts(local_i)
            )

    X = []

    for endpoint in ideal["val_endpoints"]:
        pair = counts_by_endpoint[int(endpoint)]

        if set(pair) != set(SETTINGS):
            raise RuntimeError(
                f"Missing measurement setting for endpoint={endpoint}."
            )

        X.append(
            combine_feature_pair(
                pair["XXXZYX"],
                pair["ZZZZZZ"],
            )
        )

    return np.asarray(X, dtype=float)


# =============================================================================
# METRICS
# =============================================================================

def evaluate_feature_bank(
    r,
    regime,
    X,
    ideal,
    replicate,
    sampled,
    shots,
):
    pred = common.predict_scaled_ridge(
        ideal["model"],
        ideal["scaler"],
        ideal["keep"],
        X,
    )

    X0 = ideal["X_val_ideal"]
    p0 = ideal["pred_val_ideal"]
    y = ideal["y_val"]

    diff = X - X0

    row = {
        "r": int(r),
        "regime": regime,
        "replicate": int(replicate),
        "sampled": bool(sampled),
        "shots": int(shots) if shots is not None else np.nan,
        "selected_lambda": float(ideal["selected_lambda"]),
        "training_cv_rmse": float(ideal["cv_rmse"]),
        "training_cv_sd": float(ideal["cv_sd"]),
        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "bias": bias(y, pred),
        "nrmse_train_sd": rmse(y, pred) / ideal["train_target_sd"],
        "delta_rmse_vs_same_r_ideal": (
            rmse(y, pred) - ideal["ideal_rmse"]
        ),
        "prediction_rmse_vs_same_r_ideal": rmse(p0, pred),
        "prediction_mae_vs_same_r_ideal": mae(p0, pred),
        "prediction_corr_vs_same_r_ideal": safe_corr(p0, pred),
        "feature_rmse_vs_same_r_ideal": float(
            np.sqrt(np.mean(diff ** 2))
        ),
        "feature_mae_vs_same_r_ideal": float(
            np.mean(np.abs(diff))
        ),
        "max_abs_feature_bias": float(
            np.max(np.abs(np.mean(diff, axis=0)))
        ),
    }

    feature_rows = []

    for j, feature in enumerate(FEATURES):
        d = diff[:, j]

        feature_rows.append({
            "r": int(r),
            "regime": regime,
            "replicate": int(replicate),
            "feature": feature,
            "feature_mean_ideal": float(np.mean(X0[:, j])),
            "feature_mean_regime": float(np.mean(X[:, j])),
            "feature_bias": float(np.mean(d)),
            "feature_rmse": float(np.sqrt(np.mean(d ** 2))),
            "feature_mae": float(np.mean(np.abs(d))),
            "feature_corr_vs_same_r_ideal": safe_corr(
                X0[:, j],
                X[:, j],
            ),
        })

    pred_df = pd.DataFrame({
        "r": int(r),
        "regime": regime,
        "replicate": int(replicate),
        "endpoint": ideal["val_endpoints"],
        "y_true": y,
        "prediction": pred,
        "prediction_same_r_ideal": p0,
        "prediction_error": pred - y,
        "prediction_shift_vs_same_r_ideal": pred - p0,
    })

    return row, feature_rows, pred_df


def summarize_runs(run_rows):
    df = pd.DataFrame(run_rows)

    rows = []

    for (r, regime), g in df.groupby(
        ["r", "regime"],
        sort=False,
    ):
        rows.append({
            "r": int(r),
            "regime": regime,
            "n_runs": int(len(g)),
            "rmse_mean": float(g["rmse"].mean()),
            "rmse_sd": (
                float(g["rmse"].std(ddof=1))
                if len(g) > 1
                else 0.0
            ),
            "rmse_se": (
                float(g["rmse"].std(ddof=1) / math.sqrt(len(g)))
                if len(g) > 1
                else 0.0
            ),
            "mae_mean": float(g["mae"].mean()),
            "mae_sd": (
                float(g["mae"].std(ddof=1))
                if len(g) > 1
                else 0.0
            ),
            "bias_mean": float(g["bias"].mean()),
            "bias_sd": (
                float(g["bias"].std(ddof=1))
                if len(g) > 1
                else 0.0
            ),
            "delta_rmse_vs_same_r_ideal_mean": float(
                g["delta_rmse_vs_same_r_ideal"].mean()
            ),
            "prediction_corr_vs_same_r_ideal_mean": float(
                g["prediction_corr_vs_same_r_ideal"].mean()
            ),
            "feature_rmse_vs_same_r_ideal_mean": float(
                g["feature_rmse_vs_same_r_ideal"].mean()
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["regime", "r"])
        .reset_index(drop=True)
    )


def one_se_resource_selection(summary):
    """
    Standard one-SE rule applied ONLY to S3 mean RMSE.
    """
    s3 = summary[
        summary["regime"] == "S3_FULL_NOISE_SAMPLED"
    ].copy()

    if len(s3) != len(R_VALUES):
        raise RuntimeError(
            "S3 summary is incomplete; cannot apply one-SE rule."
        )

    best_idx = s3["rmse_mean"].idxmin()
    best = s3.loc[best_idx]

    threshold = float(
        best["rmse_mean"] + best["rmse_se"]
    )

    eligible = s3[
        s3["rmse_mean"] <= threshold
    ].copy()

    selected_r = int(eligible["r"].min())

    return {
        "criterion": (
            "Shallowest r whose mean S3 RMSE is <= "
            "best mean S3 RMSE + one standard error of the best."
        ),
        "best_mean_rmse_r": int(best["r"]),
        "best_mean_rmse": float(best["rmse_mean"]),
        "best_rmse_sd": float(best["rmse_sd"]),
        "best_rmse_se": float(best["rmse_se"]),
        "one_se_threshold": threshold,
        "eligible_r": [
            int(x)
            for x in sorted(eligible["r"].tolist())
        ],
        "selected_shallowest_r": selected_r,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
    )

    parser.add_argument(
        "--sampling-reps",
        type=int,
        default=DEFAULT_SAMPLING_REPS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
    )

    parser.add_argument(
        "--sim-seed",
        type=int,
        default=DEFAULT_SIM_SEED,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use 30 validation endpoints, 256 shots and 1 S3 replicate."
        ),
    )

    args = parser.parse_args()

    shots = 256 if args.quick else int(args.shots)
    sampling_reps = 1 if args.quick else int(args.sampling_reps)

    print("=" * 120)
    print("10.03 — TROTTER DEPTH UNDER REALISTIC KINGSTON NOISE")
    print("=" * 120)
    print("NO QPU JOBS ARE SUBMITTED.")
    print("NO ERROR MITIGATION.")
    print("2026 remains frozen.")
    print("Aer simulation circuits: UNSCHEDULED (prevents 156-qubit density-matrix blow-up).")
    print()

    source_candidate, source_row = reconstruct_reference_candidate()
    layout, layout_row = load_rank1_layout()

    print("FROZEN DESIGN")
    print(
        f"  H3 / RWP W={WINDOW} / alpha={source_candidate['alpha']} / "
        f"dt={source_candidate['dt']} / {READOUT}"
    )
    print(
        f"  hx={source_candidate['hx']:.12f}, "
        f"hy={source_candidate['hy']:.12f}"
    )
    print(f"  J={source_candidate['J']}")
    print(f"  Trotter r tested: {R_VALUES}")
    print(f"  Fixed Kingston physical layout: {layout}")
    print()

    # ------------------------------------------------------------------
    # Refresh backend ONCE. All r values use the same current snapshot.
    # ------------------------------------------------------------------
    service = get_service()

    backend = service.backend(
        BACKEND_NAME,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    try:
        properties = backend.properties(refresh=True)
    except TypeError:
        properties = backend.properties()

    calibration_time = getattr(
        properties,
        "last_update_date",
        None,
    )

    status = backend.status()

    # Noise models are built ONCE from this snapshot.
    model_N4_all_gate = NoiseModel.from_backend(
        backend,
        gate_error=True,
        thermal_relaxation=False,
        readout_error=False,
    )

    model_N5_full_quantum = NoiseModel.from_backend(
        backend,
        gate_error=True,
        thermal_relaxation=True,
        readout_error=False,
    )

    model_S3_full = NoiseModel.from_backend(
        backend,
        gate_error=True,
        thermal_relaxation=True,
        readout_error=True,
    )

    reference = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": BACKEND_NAME,
        "backend_properties_last_update": json_safe(calibration_time),
        "backend_pending_jobs": int(status.pending_jobs),
        "fixed_layout_from_week10_01": layout,
        "r_values": R_VALUES,
        "shots": shots,
        "sampling_reps": sampling_reps,
        "frozen_candidate": {
            "topology": TOPOLOGY,
            "protocol": "RWP",
            "window": WINDOW,
            "alpha": float(source_candidate["alpha"]),
            "dt": float(source_candidate["dt"]),
            "hx": float(source_candidate["hx"]),
            "hy": float(source_candidate["hy"]),
            "J": {
                f"J{i}{j}": float(v)
                for (i, j), v in source_candidate["J"].items()
            },
            "readout": READOUT,
        },
        "week10_01_layout_metrics_at_selection": {
            "compiled_2q_error_max_percent": float(
                layout_row["compiled_2q_error_max_percent"]
            ),
            "compiled_1q_error_max_percent": float(
                layout_row["compiled_1q_error_max_percent"]
            ),
            "max_readout_error_percent": float(
                layout_row["max_readout_error_percent"]
            ),
            "min_t1_us": float(layout_row["min_t1_us"]),
            "min_t2_us": float(layout_row["min_t2_us"]),
        },
        "notes": [
            (
                "Physical layout is frozen from Week 10.1; no embedding "
                "re-selection occurs in 10.3."
            ),
            (
                "Kingston calibration is refreshed once at the start; the "
                "same snapshot/noise models are used for all r values."
            ),
            (
                "Each r gets its own ideal training-only Ridge selection "
                "because r changes the reservoir feature representation."
            ),
            "No QPU jobs. No error mitigation. 2026 untouched.",
        ],
    }

    (RESULTS / "10_03_reference.json").write_text(
        json.dumps(
            json_safe(reference),
            indent=2,
        ),
        encoding="utf-8",
    )

    run_rows = []
    feature_rows = []
    prediction_parts = []
    ideal_training_rows = []
    resource_rows = []

    for r in R_VALUES:
        print()
        print("-" * 120)
        print(f"TROTTER r={r}")
        print("-" * 120)

        candidate = clone_candidate(source_candidate)
        candidate["r"] = int(r)

        # --------------------------------------------------------------
        # 1) Recompute THIS r's ideal reservoir + training-only Ridge
        # --------------------------------------------------------------
        ideal = ideal_reference_for_r(candidate)

        if args.quick:
            idx = np.linspace(
                0,
                len(ideal["val_endpoints"]) - 1,
                30,
                dtype=int,
            )

            ideal = dict(ideal)
            ideal["val_endpoints"] = ideal["val_endpoints"][idx]
            ideal["X_val_ideal"] = ideal["X_val_ideal"][idx]
            ideal["y_val"] = ideal["y_val"][idx]
            ideal["pred_val_ideal"] = ideal["pred_val_ideal"][idx]

        ideal_training_rows.append({
            "r": int(r),
            "selected_lambda": float(ideal["selected_lambda"]),
            "training_cv_rmse": float(ideal["cv_rmse"]),
            "training_cv_sd": float(ideal["cv_sd"]),
            "validation_ideal_rmse": float(ideal["ideal_rmse"]),
            "validation_ideal_mae": float(ideal["ideal_mae"]),
            "validation_ideal_bias": float(ideal["ideal_bias"]),
            "train_target_sd": float(ideal["train_target_sd"]),
        })

        print(
            f"IDEAL: lambda={ideal['selected_lambda']}, "
            f"CV={ideal['cv_rmse']:.6f}±{ideal['cv_sd']:.6f}, "
            f"2025 RMSE={ideal['ideal_rmse']:.6f}"
        )

        # --------------------------------------------------------------
        # 2) Compile physical circuits and resources
        # --------------------------------------------------------------
        physical = prepare_physical_circuits(
            backend,
            candidate,
            layout,
        )

        resource = physical["resource_summary"]
        resource_rows.append(resource)

        active_core_qubits = sorted({
            int(q)
            for obs in physical["mapped_observables"]
            for _, qargs, _ in obs.to_sparse_list()
            for q in qargs
        })

        print(
            f"SIMULATION SUPPORT: mapped observable qubits={active_core_qubits}"
        )

        print(
            f"RESOURCES: CZ/feature={resource['feature_vector_n_cz']}, "
            f"depth={resource['max_setting_depth']}, "
            f"max-setting-time={resource['max_setting_duration_us']:.3f} us, "
            f"R_T2={resource['max_setting_time_over_min_t2']:.4f}"
        )

        # --------------------------------------------------------------
        # 3) A/B transpilation equivalence audit
        # --------------------------------------------------------------
        print("A/B AUDIT: transpiled ideal direct")

        X_B = run_direct_regime(
            noise_model=None,
            physical=physical,
            ideal=ideal,
            batch_size=args.batch_size,
        )

        ab_feature_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        X_B
                        - ideal["X_val_ideal"]
                    ) ** 2
                )
            )
        )

        print(
            f"  feature RMSE(A,B)={ab_feature_rmse:.3e}"
        )

        if ab_feature_rmse > 1e-6:
            raise RuntimeError(
                f"A/B equivalence audit FAILED for r={r}: "
                f"feature RMSE={ab_feature_rmse:.3e}. "
                "Do not interpret noisy results."
            )

        resource_rows[-1][
            "A_B_feature_rmse_audit"
        ] = ab_feature_rmse

        # --------------------------------------------------------------
        # 4) A logical ideal
        # --------------------------------------------------------------
        A_row, A_feat, A_pred = evaluate_feature_bank(
            r=r,
            regime="A_IDEAL_DIRECT",
            X=ideal["X_val_ideal"],
            ideal=ideal,
            replicate=0,
            sampled=False,
            shots=None,
        )

        run_rows.append(A_row)
        feature_rows.extend(A_feat)
        prediction_parts.append(A_pred)

        # --------------------------------------------------------------
        # 5) N4 direct = all gate noise, no thermal
        # --------------------------------------------------------------
        print("N4_ALL_GATE_DIRECT")
        t0 = time.time()

        X_N4 = run_direct_regime(
            noise_model=model_N4_all_gate,
            physical=physical,
            ideal=ideal,
            batch_size=args.batch_size,
        )

        N4_row, N4_feat, N4_pred = evaluate_feature_bank(
            r=r,
            regime="N4_ALL_GATE_DIRECT",
            X=X_N4,
            ideal=ideal,
            replicate=0,
            sampled=False,
            shots=None,
        )

        run_rows.append(N4_row)
        feature_rows.extend(N4_feat)
        prediction_parts.append(N4_pred)

        print(
            f"  RMSE={N4_row['rmse']:.6f}, "
            f"Δ={N4_row['delta_rmse_vs_same_r_ideal']:+.6f}, "
            f"time={(time.time()-t0)/60:.2f} min"
        )

        # --------------------------------------------------------------
        # 6) N5 direct = gate + thermal
        # --------------------------------------------------------------
        print("N5_FULL_QUANTUM_DIRECT")
        t0 = time.time()

        X_N5 = run_direct_regime(
            noise_model=model_N5_full_quantum,
            physical=physical,
            ideal=ideal,
            batch_size=args.batch_size,
        )

        N5_row, N5_feat, N5_pred = evaluate_feature_bank(
            r=r,
            regime="N5_FULL_QUANTUM_DIRECT",
            X=X_N5,
            ideal=ideal,
            replicate=0,
            sampled=False,
            shots=None,
        )

        run_rows.append(N5_row)
        feature_rows.extend(N5_feat)
        prediction_parts.append(N5_pred)

        print(
            f"  RMSE={N5_row['rmse']:.6f}, "
            f"Δ={N5_row['delta_rmse_vs_same_r_ideal']:+.6f}, "
            f"time={(time.time()-t0)/60:.2f} min"
        )

        # --------------------------------------------------------------
        # 7) S3 = full quantum + readout + finite shots
        # --------------------------------------------------------------
        print(
            f"S3_FULL_NOISE_SAMPLED: "
            f"{sampling_reps} replicate(s), {shots} shots/setting"
        )

        for rep in range(sampling_reps):
            # Same replicate seed is reused across r values -> common random
            # numbers, reducing purely arbitrary seed differences between r.
            rep_seed = int(
                args.sim_seed + 1000000 * rep
            )

            t0 = time.time()

            X_S3 = run_sampled_regime(
                noise_model=model_S3_full,
                physical=physical,
                ideal=ideal,
                shots=shots,
                batch_size=args.batch_size,
                seed=rep_seed,
            )

            S3_row, S3_feat, S3_pred = evaluate_feature_bank(
                r=r,
                regime="S3_FULL_NOISE_SAMPLED",
                X=X_S3,
                ideal=ideal,
                replicate=rep + 1,
                sampled=True,
                shots=shots,
            )

            run_rows.append(S3_row)
            feature_rows.extend(S3_feat)
            prediction_parts.append(S3_pred)

            print(
                f"  rep {rep+1}/{sampling_reps}: "
                f"RMSE={S3_row['rmse']:.6f}, "
                f"Δ={S3_row['delta_rmse_vs_same_r_ideal']:+.6f}, "
                f"time={(time.time()-t0)/60:.2f} min"
            )

    # =============================================================================
    # SUMMARIZE / SELECT
    # =============================================================================

    runs = pd.DataFrame(run_rows)
    summary = summarize_runs(run_rows)
    ideal_training = pd.DataFrame(ideal_training_rows)
    resources = pd.DataFrame(resource_rows)
    features = pd.DataFrame(feature_rows)
    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    selection = one_se_resource_selection(
        summary
    )

    # Add resource context for selected/best r.
    resource_by_r = {
        int(row["r"]): row
        for row in resources.to_dict(orient="records")
    }

    selection["best_mean_resource"] = json_safe(
        resource_by_r[
            int(selection["best_mean_rmse_r"])
        ]
    )

    selection["selected_shallowest_resource"] = json_safe(
        resource_by_r[
            int(selection["selected_shallowest_r"])
        ]
    )

    selection["created_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    selection["warning"] = (
        "One-SE result is a resource-efficiency candidate only. "
        "Inspect paired S3 replicate performance and physical cost before "
        "making the final scientific choice."
    )

    # =============================================================================
    # SAVE
    # =============================================================================

    ideal_training.to_csv(
        RESULTS / "10_03_trotter_ideal_training_summary.csv",
        index=False,
    )

    runs.to_csv(
        RESULTS / "10_03_trotter_regime_runs.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS / "10_03_trotter_regime_summary.csv",
        index=False,
    )

    features.to_csv(
        RESULTS / "10_03_trotter_feature_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        RESULTS / "10_03_trotter_predictions.csv",
        index=False,
    )

    resources.to_csv(
        RESULTS / "10_03_trotter_resource_metrics.csv",
        index=False,
    )

    (RESULTS / "10_03_trotter_selection.json").write_text(
        json.dumps(
            json_safe(selection),
            indent=2,
        ),
        encoding="utf-8",
    )

    # =============================================================================
    # CONSOLE SUMMARY
    # =============================================================================

    print()
    print("=" * 120)
    print("10.03 FINAL SUMMARY")
    print("=" * 120)

    print()
    print("IDEAL TRAINING / VALIDATION")
    print(
        ideal_training.to_string(
            index=False
        )
    )

    print()
    print("PERFORMANCE")
    display_cols = [
        "r",
        "regime",
        "n_runs",
        "rmse_mean",
        "rmse_sd",
        "rmse_se",
        "mae_mean",
        "bias_mean",
        "delta_rmse_vs_same_r_ideal_mean",
        "prediction_corr_vs_same_r_ideal_mean",
    ]

    print(
        summary[display_cols]
        .sort_values(
            ["regime", "r"]
        )
        .to_string(index=False)
    )

    print()
    print("RESOURCES")
    resource_cols = [
        "r",
        "feature_vector_n_cz",
        "max_setting_depth",
        "max_setting_duration_us",
        "feature_vector_duration_sum_us",
        "min_layout_t1_us",
        "min_layout_t2_us",
        "max_setting_time_over_min_t1",
        "max_setting_time_over_min_t2",
        "A_B_feature_rmse_audit",
    ]

    print(
        resources[resource_cols]
        .sort_values("r")
        .to_string(index=False)
    )

    print()
    print("ONE-SE RESOURCE CANDIDATE")
    print(
        f"  best S3 mean RMSE: r={selection['best_mean_rmse_r']} "
        f"({selection['best_mean_rmse']:.6f})"
    )
    print(
        f"  one-SE threshold: {selection['one_se_threshold']:.6f}"
    )
    print(
        f"  eligible r: {selection['eligible_r']}"
    )
    print(
        f"  shallowest eligible r: "
        f"{selection['selected_shallowest_r']}"
    )

    print()
    print("Saved:")
    print("  results/10_03_reference.json")
    print("  results/10_03_trotter_ideal_training_summary.csv")
    print("  results/10_03_trotter_regime_runs.csv")
    print("  results/10_03_trotter_regime_summary.csv")
    print("  results/10_03_trotter_feature_metrics.csv")
    print("  results/10_03_trotter_predictions.csv")
    print("  results/10_03_trotter_resource_metrics.csv")
    print("  results/10_03_trotter_selection.json")

    print()
    print("10.03 COMPLETE.")
    print(
        "Stop here. Inspect the r=1/2/3 performance-resource trade-off "
        "before proceeding to Week 10.4 shot sensitivity."
    )


if __name__ == "__main__":
    main()
