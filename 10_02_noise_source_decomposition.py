#!/usr/bin/env python
"""
10.02 — Kingston noise-source decomposition for the frozen Week-10 reference
============================================================================

Goal
----
Explain the Week-10.1 observation that:
    * quantum noise alone (direct expectations) reduced validation RMSE, while
    * full noisy finite-shot measurement returned RMSE close to the ideal value.

This script keeps the logical QRC fixed and uses ONLY the Week-10.1 Rule-5
rank-1 physical embedding.  It does NOT re-open topology, protocol, window,
alpha, readout, couplings, Ridge, or physical-layout selection.

Frozen logical candidate
------------------------
H3 / RWP W=4 / alpha=0.25 / r=3 / XZinj_dropX3_plus_YX45

Physical layout
---------------
Loaded from:
    results/10_01_kingston_fresh_top3_embeddings.csv
where rule5_rank_current == 1.

Noise regimes
-------------
DIRECT expectation-value regimes (no finite-shot sampling, no readout error):
    A_IDEAL_DIRECT
    N1_1Q_GATE_DIRECT
    N2_2Q_GATE_DIRECT
    N3_THERMAL_DIRECT
    N4_ALL_GATE_DIRECT
    N5_FULL_QUANTUM_DIRECT

FINITE-SHOT regimes (default 1024 shots):
    S0_SAMPLING_ONLY
    S1_READOUT_ONLY
    S2_FULL_QUANTUM_SAMPLED_NO_READOUT
    S3_FULL_NOISE_SAMPLED

Interpretation
--------------
N1/N2/N4 use backend-calibrated depolarizing gate-error channels only.
N3 uses backend T1/T2 + calibrated instruction durations only.
N5 = gate error + thermal relaxation, but no readout.
S1 isolates measurement assignment error + shot noise.
S2 = quantum noise + shot noise, readout OFF.
S3 = quantum noise + shot noise + readout error.

No mitigation is used.

Important:
NoiseModel.from_backend() is calibration-derived and approximate.  For
BackendV2, Aer uses backend.target gate errors, gate durations, qubit T1/T2,
and readout properties.

2026 remains untouched.
NO QPU jobs are submitted.

Required files
--------------
ibm_account.py
09_04_rwp_common.py

results/09_04c_rwp_all_trials.csv
results/10_01_kingston_fresh_top3_embeddings.csv
plus files already required by 09_04_rwp_common.py.

Outputs
-------
results/10_02_reference.json
results/10_02_noise_model_audit.json
results/10_02_noise_regime_runs.csv
results/10_02_noise_regime_summary.csv
results/10_02_noise_regime_feature_metrics.csv
results/10_02_noise_regime_predictions.csv

Run
---
python 10_02_noise_source_decomposition.py --shots 1024 --sampling-reps 3

Quick plumbing check
--------------------
python 10_02_noise_source_decomposition.py --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
from qiskit_aer.noise.device.models import basic_device_gate_errors
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


common = load_module(COMMON_FILE, "qrc_1002_common")


# =============================================================================
# GENERIC HELPERS
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


# =============================================================================
# RECONSTRUCT FROZEN CANDIDATE + LOAD WEEK-10.1 RANK-1 LAYOUT
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
            f"Expected exactly one frozen Stage-C row; found {len(rows)}."
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
        "r": int(candidate["r"]) == int(source_row["r"]),
        "hx": np.isclose(candidate["hx"], source_row["hx"]),
        "hy": np.isclose(candidate["hy"], source_row["hy"]),
    }

    if not all(checks.values()):
        raise RuntimeError(
            "Frozen candidate reconstruction audit failed: "
            + ", ".join(k for k, v in checks.items() if not v)
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
# LOGICAL IDEAL REFERENCE AND FROZEN RIDGE
# =============================================================================

def ideal_reference(candidate):
    work_tv, cols, y_all = common.load_train_validation()

    angles = common.make_angles(
        work_tv,
        cols,
        float(candidate["alpha"]),
    )

    A_list = common.build_channels(candidate, angles)

    endpoints, master = common.rwp_master_feature_bank_from_channels(
        A_list,
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
    qc = QuantumCircuit(qreg, name="H3_W4_core")
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
        name=f"H3_W4_{setting_label}",
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
# COUNT -> EXPECTATION FEATURES
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
# NOISE MODELS
# =============================================================================

def build_arity_gate_noise_model(backend, arity):
    """
    Exact Aer backend-calibrated depolarizing gate-error construction, filtered
    by gate arity.

    This uses the same helper NoiseModel.from_backend() uses internally for
    BackendV2 targets, but keeps only 1Q or 2Q gate-error channels.
    Thermal relaxation is OFF here.
    """
    model = NoiseModel(
        basis_gates=list(backend.operation_names)
    )

    errors = basic_device_gate_errors(
        properties=None,
        gate_error=True,
        thermal_relaxation=False,
        target=backend.target,
    )

    added = 0

    for name, qubits, error in errors:
        if len(qubits) == int(arity):
            model.add_quantum_error(
                error,
                name,
                tuple(qubits),
            )
            added += 1

    return model, added


def build_noise_models(backend):
    one_q, n_1q = build_arity_gate_noise_model(
        backend, 1
    )

    two_q, n_2q = build_arity_gate_noise_model(
        backend, 2
    )

    thermal = NoiseModel.from_backend(
        backend,
        gate_error=False,
        thermal_relaxation=True,
        readout_error=False,
    )

    all_gate = NoiseModel.from_backend(
        backend,
        gate_error=True,
        thermal_relaxation=False,
        readout_error=False,
    )

    full_quantum = NoiseModel.from_backend(
        backend,
        gate_error=True,
        thermal_relaxation=True,
        readout_error=False,
    )

    readout_only = NoiseModel.from_backend(
        backend,
        gate_error=False,
        thermal_relaxation=False,
        readout_error=True,
    )

    full = NoiseModel.from_backend(
        backend,
        gate_error=True,
        thermal_relaxation=True,
        readout_error=True,
    )

    audit = {
        "1q_gate_only": {
            "custom_errors_added": int(n_1q),
            "noise_instructions": list(one_q.noise_instructions),
            "noise_qubits_count": len(one_q.noise_qubits),
        },
        "2q_gate_only": {
            "custom_errors_added": int(n_2q),
            "noise_instructions": list(two_q.noise_instructions),
            "noise_qubits_count": len(two_q.noise_qubits),
        },
        "thermal_only": {
            "noise_instructions": list(thermal.noise_instructions),
            "noise_qubits_count": len(thermal.noise_qubits),
        },
        "all_gate_only": {
            "noise_instructions": list(all_gate.noise_instructions),
            "noise_qubits_count": len(all_gate.noise_qubits),
        },
        "full_quantum_no_readout": {
            "noise_instructions": list(full_quantum.noise_instructions),
            "noise_qubits_count": len(full_quantum.noise_qubits),
        },
        "readout_only": {
            "noise_instructions": list(readout_only.noise_instructions),
            "noise_qubits_count": len(readout_only.noise_qubits),
        },
        "full": {
            "noise_instructions": list(full.noise_instructions),
            "noise_qubits_count": len(full.noise_qubits),
        },
    }

    return {
        "1q": one_q,
        "2q": two_q,
        "thermal": thermal,
        "all_gate": all_gate,
        "full_quantum": full_quantum,
        "readout_only": readout_only,
        "full": full,
    }, audit


# =============================================================================
# TRANSPILATION
# =============================================================================

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
            "Frozen Week-10.1 rank-1 layout is no longer zero-SWAP."
        )

    mapped_observables = [
        logical_sparse_observable(feature).apply_layout(
            transpiled_core.layout
        )
        for feature in FEATURES
    ]

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
        )

        backend.check_faulty(tqc)

        if int(tqc.count_ops().get("swap", 0)) != 0:
            raise RuntimeError(
                f"Frozen rank-1 layout introduced SWAPs for {setting_label}."
            )

        transpiled_meas[setting_label] = tqc
        meas_params[setting_label] = pars

    resource = {
        "core_depth": int(transpiled_core.depth()),
        "core_n_cz": int(
            transpiled_core.count_ops().get("cz", 0)
        ),
        "measurement_depth_max": int(
            max(c.depth() for c in transpiled_meas.values())
        ),
        "feature_vector_n_cz": int(
            sum(
                int(c.count_ops().get("cz", 0))
                for c in transpiled_meas.values()
            )
        ),
        "feature_vector_n_swap": int(
            sum(
                int(c.count_ops().get("swap", 0))
                for c in transpiled_meas.values()
            )
        ),
    }

    return {
        "core": transpiled_core,
        "core_params": core_params,
        "mapped_observables": mapped_observables,
        "meas": transpiled_meas,
        "meas_params": meas_params,
        "resource": resource,
    }


# =============================================================================
# DIRECT EXPECTATION REGIMES
# =============================================================================

def run_direct_regime(
    name,
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
                    f"{name}: unexpected estimator EV shape."
                )

            X[start + local_i, :] = evs

    return X


# =============================================================================
# SAMPLED REGIMES
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
    if noise_model is None:
        simulator = AerSimulator(
            enable_truncation=True,
        )
    else:
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
    regime,
    X,
    ideal,
    replicate,
    sampled,
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

    run_row = {
        "regime": regime,
        "replicate": replicate,
        "sampled": bool(sampled),
        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "bias": bias(y, pred),
        "delta_rmse_vs_ideal": rmse(y, pred) - ideal["ideal_rmse"],
        "prediction_rmse_vs_ideal": rmse(p0, pred),
        "prediction_mae_vs_ideal": mae(p0, pred),
        "prediction_corr_vs_ideal": safe_corr(p0, pred),
        "feature_rmse_vs_ideal": float(
            np.sqrt(np.mean(diff ** 2))
        ),
        "feature_mae_vs_ideal": float(
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
            "regime": regime,
            "replicate": replicate,
            "feature": feature,
            "feature_mean_ideal": float(
                np.mean(X0[:, j])
            ),
            "feature_mean_regime": float(
                np.mean(X[:, j])
            ),
            "feature_bias": float(
                np.mean(d)
            ),
            "feature_rmse": float(
                np.sqrt(np.mean(d ** 2))
            ),
            "feature_mae": float(
                np.mean(np.abs(d))
            ),
            "feature_corr_vs_ideal": safe_corr(
                X0[:, j],
                X[:, j],
            ),
        })

    pred_df = pd.DataFrame({
        "regime": regime,
        "replicate": replicate,
        "endpoint": ideal["val_endpoints"],
        "y_true": y,
        "prediction": pred,
        "prediction_ideal": p0,
    })

    return run_row, feature_rows, pred_df


def summarize_runs(runs):
    df = pd.DataFrame(runs)

    rows = []

    for regime, g in df.groupby("regime", sort=False):
        rows.append({
            "regime": regime,
            "n_runs": int(len(g)),
            "rmse_mean": float(g["rmse"].mean()),
            "rmse_sd": float(
                g["rmse"].std(ddof=1)
            ) if len(g) > 1 else 0.0,
            "mae_mean": float(g["mae"].mean()),
            "mae_sd": float(
                g["mae"].std(ddof=1)
            ) if len(g) > 1 else 0.0,
            "bias_mean": float(g["bias"].mean()),
            "bias_sd": float(
                g["bias"].std(ddof=1)
            ) if len(g) > 1 else 0.0,
            "delta_rmse_vs_ideal_mean": float(
                g["delta_rmse_vs_ideal"].mean()
            ),
            "prediction_corr_vs_ideal_mean": float(
                g["prediction_corr_vs_ideal"].mean()
            ),
            "feature_rmse_vs_ideal_mean": float(
                g["feature_rmse_vs_ideal"].mean()
            ),
        })

    out = pd.DataFrame(rows)

    order = [
        "A_IDEAL_DIRECT",
        "N1_1Q_GATE_DIRECT",
        "N2_2Q_GATE_DIRECT",
        "N3_THERMAL_DIRECT",
        "N4_ALL_GATE_DIRECT",
        "N5_FULL_QUANTUM_DIRECT",
        "S0_SAMPLING_ONLY",
        "S1_READOUT_ONLY",
        "S2_FULL_QUANTUM_SAMPLED_NO_READOUT",
        "S3_FULL_NOISE_SAMPLED",
    ]

    out["_order"] = out["regime"].map(
        {name: i for i, name in enumerate(order)}
    )

    return (
        out.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


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
            "Use 30 validation endpoints, 256 shots and 1 sampling replicate."
        ),
    )

    args = parser.parse_args()

    shots = 256 if args.quick else int(args.shots)
    sampling_reps = 1 if args.quick else int(args.sampling_reps)

    print("=" * 118)
    print("10.02 — KINGSTON NOISE-SOURCE DECOMPOSITION")
    print("=" * 118)
    print("NO QPU JOBS ARE SUBMITTED.")
    print("NO ERROR MITIGATION.")
    print("2026 remains frozen.")
    print()

    candidate, stage_row = reconstruct_reference_candidate()
    layout, layout_row = load_rank1_layout()

    print("FROZEN CANDIDATE")
    print(
        f"  H3 / RWP W={WINDOW} / alpha={candidate['alpha']} / "
        f"r={candidate['r']} / {READOUT}"
    )
    print(f"  Kingston layout from 10.01 rank 1: {layout}")
    print()

    ideal = ideal_reference(candidate)

    # Optional quick subset.
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

    # Backend refresh, but NO physical re-selection.
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

    physical = prepare_physical_circuits(
        backend,
        candidate,
        layout,
    )

    models, noise_audit = build_noise_models(
        backend
    )

    reference = {
        "backend": BACKEND_NAME,
        "backend_calibration_time": json_safe(calibration_time),
        "layout": layout,
        "candidate": {
            "topology": TOPOLOGY,
            "protocol": "RWP",
            "window": WINDOW,
            "alpha": float(candidate["alpha"]),
            "r": int(candidate["r"]),
            "dt": float(candidate["dt"]),
            "hx": float(candidate["hx"]),
            "hy": float(candidate["hy"]),
            "readout": READOUT,
            "J": {
                f"J{i}{j}": float(v)
                for (i, j), v in candidate["J"].items()
            },
        },
        "shots": shots,
        "sampling_reps": sampling_reps,
        "resource": physical["resource"],
        "week10_01_rank1_metrics": {
            "compiled_2q_error_max_percent": float(
                layout_row["compiled_2q_error_max_percent"]
            ),
            "max_readout_error_percent": float(
                layout_row["max_readout_error_percent"]
            ),
            "compiled_1q_error_max_percent": float(
                layout_row["compiled_1q_error_max_percent"]
            ),
            "min_t1_us": float(layout_row["min_t1_us"]),
            "min_t2_us": float(layout_row["min_t2_us"]),
        },
    }

    # Save audit early.
    (RESULTS / "10_02_reference.json").write_text(
        json.dumps(
            json_safe(reference),
            indent=2,
        ),
        encoding="utf-8",
    )

    (RESULTS / "10_02_noise_model_audit.json").write_text(
        json.dumps(
            json_safe({
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "backend_calibration_time": calibration_time,
                "noise_models": noise_audit,
                "notes": [
                    (
                        "1Q-only and 2Q-only models use the same "
                        "basic_device_gate_errors() construction used by "
                        "NoiseModel.from_backend(), filtered by gate arity."
                    ),
                    (
                        "Thermal-only uses backend T1/T2 and calibrated "
                        "instruction durations."
                    ),
                    (
                        "No mitigation is applied."
                    ),
                ],
            }),
            indent=2,
        ),
        encoding="utf-8",
    )

    run_rows = []
    feature_rows = []
    prediction_parts = []

    # ------------------------------------------------------------------
    # A: original logical exact reference
    # ------------------------------------------------------------------
    A_row, A_features, A_pred = evaluate_feature_bank(
        "A_IDEAL_DIRECT",
        ideal["X_val_ideal"],
        ideal,
        replicate=0,
        sampled=False,
    )

    run_rows.append(A_row)
    feature_rows.extend(A_features)
    prediction_parts.append(A_pred)

    # ------------------------------------------------------------------
    # Direct deterministic noise regimes
    # ------------------------------------------------------------------
    direct_regimes = [
        ("N1_1Q_GATE_DIRECT", models["1q"]),
        ("N2_2Q_GATE_DIRECT", models["2q"]),
        ("N3_THERMAL_DIRECT", models["thermal"]),
        ("N4_ALL_GATE_DIRECT", models["all_gate"]),
        ("N5_FULL_QUANTUM_DIRECT", models["full_quantum"]),
    ]

    for regime, model in direct_regimes:
        print(f"DIRECT: {regime}")
        t0 = time.time()

        X = run_direct_regime(
            regime,
            model,
            physical,
            ideal,
            args.batch_size,
        )

        row, feats, preds = evaluate_feature_bank(
            regime,
            X,
            ideal,
            replicate=0,
            sampled=False,
        )

        run_rows.append(row)
        feature_rows.extend(feats)
        prediction_parts.append(preds)

        print(
            f"  RMSE={row['rmse']:.6f}, "
            f"Δ={row['delta_rmse_vs_ideal']:+.6f}, "
            f"time={(time.time()-t0)/60:.2f} min"
        )

    # ------------------------------------------------------------------
    # Sampled stochastic regimes
    # ------------------------------------------------------------------
    sampled_regimes = [
        ("S0_SAMPLING_ONLY", None),
        ("S1_READOUT_ONLY", models["readout_only"]),
        (
            "S2_FULL_QUANTUM_SAMPLED_NO_READOUT",
            models["full_quantum"],
        ),
        ("S3_FULL_NOISE_SAMPLED", models["full"]),
    ]

    for rep in range(sampling_reps):
        print()
        print(
            f"SAMPLED REPLICATE {rep + 1}/{sampling_reps}, shots={shots}"
        )

        rep_seed = int(
            args.sim_seed + 1000000 * rep
        )

        for regime, model in sampled_regimes:
            t0 = time.time()

            # Common random-number seed within each replicate.
            X = run_sampled_regime(
                model,
                physical,
                ideal,
                shots,
                args.batch_size,
                rep_seed,
            )

            row, feats, preds = evaluate_feature_bank(
                regime,
                X,
                ideal,
                replicate=rep + 1,
                sampled=True,
            )

            run_rows.append(row)
            feature_rows.extend(feats)
            prediction_parts.append(preds)

            print(
                f"  {regime:<38} "
                f"RMSE={row['rmse']:.6f}, "
                f"Δ={row['delta_rmse_vs_ideal']:+.6f}, "
                f"time={(time.time()-t0)/60:.2f} min"
            )

    runs = pd.DataFrame(run_rows)
    summary = summarize_runs(run_rows)
    features = pd.DataFrame(feature_rows)
    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    runs.to_csv(
        RESULTS / "10_02_noise_regime_runs.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS / "10_02_noise_regime_summary.csv",
        index=False,
    )

    features.to_csv(
        RESULTS / "10_02_noise_regime_feature_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        RESULTS / "10_02_noise_regime_predictions.csv",
        index=False,
    )

    print()
    print("=" * 118)
    print("10.02 SUMMARY")
    print("=" * 118)

    print(
        summary[
            [
                "regime",
                "n_runs",
                "rmse_mean",
                "rmse_sd",
                "mae_mean",
                "bias_mean",
                "delta_rmse_vs_ideal_mean",
                "prediction_corr_vs_ideal_mean",
                "feature_rmse_vs_ideal_mean",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print("  results/10_02_reference.json")
    print("  results/10_02_noise_model_audit.json")
    print("  results/10_02_noise_regime_runs.csv")
    print("  results/10_02_noise_regime_summary.csv")
    print("  results/10_02_noise_regime_feature_metrics.csv")
    print("  results/10_02_noise_regime_predictions.csv")
    print()
    print("10.02 COMPLETE.")
    print(
        "Stop here. Do not start the Trotter-depth experiment until these "
        "noise-source results are interpreted."
    )


if __name__ == "__main__":
    main()
