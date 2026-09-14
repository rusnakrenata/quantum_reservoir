#!/usr/bin/env python
"""
10.01 — Reference candidate + fresh Kingston embedding refresh + Aer comparison
==============================================================================

Scientific purpose
------------------
Week 10 does NOT reopen the Week-9 topology/protocol/readout search.

We freeze ONE reference reservoir:
    H3 / RWP W=4 / alpha=0.25 / XZinj_dropX3_plus_YX45

Then, because IBM calibrations are time-dependent, we do four things:

1. Re-evaluate the OLD Kingston H3 embedding from Week 9.05C using CURRENT
   calibration data.  This quantifies calibration drift without silently
   changing the physical qubits.

2. Re-scan CURRENT Kingston hardware and find the fresh Rule-5 top 3 native
   H3 embeddings.  We record whether the old embedding is still among them.

3. Test the union
       {fresh top 3} U {old Week-9 embedding}
   so the old region is still evaluated even if calibration drift pushes it
   out of the top 3.

4. For every tested embedding, evaluate the same frozen QRC on the 2025
   validation period using FIVE deliberately separated references:
       A. exact logical ideal expectation values from the density matrix,
       B. transpiled physical circuit, direct expectation values, NO noise,
       C. transpiled physical circuit, finite-shot sampling, NO noise,
       D. transpiled physical circuit, direct expectation values with
          Kingston quantum noise (gate/decoherence noise; no readout),
       E. transpiled physical circuit, finite-shot sampling with the full
          Kingston model (quantum noise + readout error).

   NO error mitigation is applied in any mode.

Observable-mapping correction:
B and D now use Qiskit Aer EstimatorV2 directly.  Logical observables are
mapped with the documented
    logical_observable.apply_layout(transpiled_circuit.layout)
workflow and are passed to Estimator as full mapped observables.  There is no
manual post-transpile qubit indexing and no save_expectation_value snapshot.
The script aborts unless B reproduces A to numerical precision.  Finite-shot
modes use common simulator seeds across embeddings.

The classical scaler and Ridge readout are fitted ONCE on ideal 2022-2024
features and then frozen.  They are never refitted to noisy features.
Readout-error mitigation, ZNE, PEC, postselection, and other mitigation
methods are intentionally OFF in Week 10.1.
2026 remains untouched.

NO QPU JOBS ARE SUBMITTED.

Important Stage-C serialization warning
---------------------------------------
The H1/H2/H3 topology-specific J columns in 09_04c_rwp_all_trials.csv are not
safe reconstruction sources.  The exact candidate is reconstructed from the
clean 09_03c H3 baseline + deterministic Stage-C search_id.

Required project files
----------------------
ibm_account.py
09_04_rwp_common.py                  (revised post-QPU version)
11_0A_live_embedding_reselection.py

results/09_03c_forecast_best_135.csv
results/09_04c_rwp_all_trials.csv
results/09_05c_rule5_fresh_embedding_metrics.csv
results/03_01_preprocessed_samples.csv
plus files already required by 09_04_rwp_common.py.

Outputs
-------
results/10_01_reference_candidate.json
results/10_01_kingston_refresh_backend_status.csv
results/10_01_kingston_fresh_top3_embeddings.csv
results/10_01_old_embedding_drift.csv
results/10_01_embedding_comparison.csv
results/10_01_embedding_feature_metrics.csv
results/10_01_embedding_predictions.csv
results/10_01_embedding_resource_metrics.csv
results/10_01_simulation_reference_decomposition.csv
results/10_01_summary.json

Typical full run
----------------
python 10_01_reference_embedding_refresh_and_simulation.py --shots 1024

Optional quick plumbing check
-----------------------------
python 10_01_reference_embedding_refresh_and_simulation.py --quick
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

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli, SparsePauliOp

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
SELECTOR_FILE = HERE / "11_0A_live_embedding_reselection.py"

TRIAL_FILE = RESULTS / "09_04c_rwp_all_trials.csv"
RULE5_FILE = RESULTS / "09_05c_rule5_fresh_embedding_metrics.csv"

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
        "measured": [0, 1, 2, 3, 4, 5],
    },
    "ZZZZZZ": {
        "basis": {q: "Z" for q in range(6)},
        "measured": [0, 1, 2, 3, 4, 5],
    },
}

# Direct-expectation observables. For a two-qubit Pauli on q4,q5, the label
# "XY" corresponds to Y on the first listed qubit (q4) and X on the second
# listed qubit (q5), because Qiskit Pauli labels are ordered high-to-low.
DIRECT_OBSERVABLES = {
    "X0": ("X", [0]),
    "X1": ("X", [1]),
    "X2": ("X", [2]),
    "Z0": ("Z", [0]),
    "Z1": ("Z", [1]),
    "Z2": ("Z", [2]),
    "Z3": ("Z", [3]),
    "YX45": ("XY", [4, 5]),
}

OPT_LEVEL = 1
SEED_TRANSPILE = 42
DEFAULT_SHOTS = 1024
DEFAULT_SHORTLIST = 120
DEFAULT_BATCH = 80
DEFAULT_SIM_SEED = 20260911

H3_EDGES = [(2, 3), (1, 2), (0, 1), (0, 4), (4, 5)]

OLD_METRIC_COLUMNS = {
    "compiled_2q_error_max_percent": "max_2q_error_percent",
    "compiled_1q_error_max_percent": "max_1q_error_percent",
    "max_readout_error_percent": "max_readout_error_percent",
    "min_t1_us": "min_t1_us",
    "min_t2_us": "min_t2_us",
    "compiled_duration_us_core": "core_duration_us",
}


# =============================================================================
# IMPORT PROJECT MODULES
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


common = load_module(COMMON_FILE, "qrc_1001_common")
selector = load_module(SELECTOR_FILE, "qrc_1001_selector")


# =============================================================================
# BASIC METRICS
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


def layout_key(layout):
    return tuple(int(q) for q in layout)


def parse_layout(value):
    if isinstance(value, str):
        return [int(x) for x in json.loads(value)]
    return [int(x) for x in value]


# =============================================================================
# EXACT REFERENCE CANDIDATE
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
            f"Expected one Stage-C reference row; found {len(rows)}."
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
            "Reference reconstruction audit failed: "
            + ", ".join(k for k, v in checks.items() if not v)
        )

    if READOUT not in common.READOUT_FEATURES:
        raise RuntimeError(
            f"{COMMON_FILE} does not contain the revised readout family "
            f"{READOUT!r}. Use the post-QPU revised 09_04_rwp_common.py."
        )

    if list(common.READOUT_FEATURES[READOUT]) != FEATURES:
        raise RuntimeError(
            "Reference feature ordering differs from the expected "
            "post-QPU DROP-X3+YX45 family."
        )

    frozen = {
        "topology": candidate["topology"],
        "protocol": "RWP",
        "window": WINDOW,
        "search_id": SEARCH_ID,
        "alpha": float(candidate["alpha"]),
        "dt": float(candidate["dt"]),
        "r": int(candidate["r"]),
        "hx": float(candidate["hx"]),
        "hy": float(candidate["hy"]),
        "J": {
            f"J{i}{j}": float(v)
            for (i, j), v in candidate["J"].items()
        },
        "readout": READOUT,
        "features": FEATURES,
        "measurement_settings": list(SETTINGS),
        "selected_lambda_week9": float(source_row["selected_lambda"]),
        "week9_cv_rmse": float(source_row["cv_rmse"]),
        "week9_cv_sd": float(source_row["cv_rmse_std"]),
        "week9_2025_rmse": float(source_row["validation_rmse"]),
        "week9_2025_mae": float(source_row["validation_mae"]),
        "week9_2025_bias": float(source_row["validation_bias"]),
        "week9_shotSD_1024": float(
            source_row["shot_noise_forecast_sd_proxy_1024"]
        ),
        "week9_Rmax_1024": float(
            source_row["max_shot_to_train_std_ratio_1024"]
        ),
        "week9_Smax": float(
            source_row["max_abs_raw_prediction_sensitivity"]
        ),
    }

    return candidate, source_row, frozen


# =============================================================================
# OLD WEEK-9 EMBEDDING
# =============================================================================

def load_old_embedding():
    if not RULE5_FILE.exists():
        raise FileNotFoundError(RULE5_FILE)

    df = pd.read_csv(RULE5_FILE)

    g = df[
        (df["topology"].astype(str) == TOPOLOGY)
        & (df["backend"].astype(str) == BACKEND_NAME)
    ].copy()

    if len(g) != 1:
        raise RuntimeError(
            f"Expected one old H3/Kingston Rule-5 row; found {len(g)}."
        )

    row = g.iloc[0]
    layout = parse_layout(row["layout"])

    metrics = {
        new: float(row[old])
        for old, new in OLD_METRIC_COLUMNS.items()
    }

    try:
        file_mtime = datetime.fromtimestamp(
            RULE5_FILE.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
    except Exception:
        file_mtime = None

    return layout, metrics, file_mtime


# =============================================================================
# FRESH EMBEDDING SEARCH
# =============================================================================

def strict_rule5_sort(df):
    """
    Exact Rule-5 hierarchy used in the article for a fixed backend:
       2Q max error ↓
       readout max error ↓
       1Q max error ↓
       min T2 ↑
       min T1 ↑
       compiled duration ↓

    No weighted score.
    """
    cols = [
        "compiled_2q_error_max",
        "max_readout_error",
        "compiled_1q_error_max",
        "min_t2_us",
        "min_t1_us",
        "compiled_duration_us",
    ]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing Rule-5 columns: {missing}")

    return df.sort_values(
        cols,
        ascending=[True, True, True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def refresh_and_rank(candidate, service, shortlist):
    # Make selector's first-input compilation consistent with THIS reference.
    selector.ALPHA = float(candidate["alpha"])
    selector.DT = float(candidate["dt"])

    fresh = selector.fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=[BACKEND_NAME],
        shortlist=int(shortlist),
        write_prefix="10_01_H3_kingston_refresh",
        verbose=True,
    )

    compiled = fresh["compiled"].copy()
    good = compiled[
        compiled["strict_compile_pass"].astype(bool)
    ].copy()

    if len(good) < 3:
        raise RuntimeError(
            f"Only {len(good)} strict zero-SWAP Kingston embeddings survived."
        )

    ranked = strict_rule5_sort(good)
    ranked["rule5_rank_current"] = np.arange(len(ranked)) + 1

    return fresh, ranked


def current_row_for_old_embedding(
    old_layout,
    candidate,
    fresh,
    ranked,
    service,
):
    old_json = json.dumps(old_layout)

    match = ranked[
        ranked["layout"].apply(lambda x: layout_key(parse_layout(x)))
        == layout_key(old_layout)
    ].copy()

    if len(match):
        return match.iloc[0].to_dict(), True

    # The old region can be valid but absent from the calibration shortlist.
    # Evaluate it explicitly on the same current backend.
    backend = service.backend(
        BACKEND_NAME,
        use_fractional_gates=False,
    )
    try:
        backend.refresh()
    except Exception:
        pass

    qubits = fresh["qubits"]
    metrics = selector.mapping_metrics(
        backend,
        TOPOLOGY,
        old_layout,
        qubits,
    )

    if metrics is None:
        return {
            "layout": old_json,
            "strict_compile_pass": False,
            "compile_error": (
                "Old layout no longer has complete native calibrated H3 edges."
            ),
        }, False

    angles, _ = selector.first_input_angles()
    core = selector.build_core(candidate, angles)

    try:
        compiled = transpile(
            core,
            backend=backend,
            initial_layout=old_layout,
            routing_method="none",
            optimization_level=OPT_LEVEL,
            seed_transpiler=SEED_TRANSPILE,
            scheduling_method="alap",
        )

        backend.check_faulty(compiled)
        cm = selector.compiled_metrics(compiled, backend)
        strict_pass = int(cm.get("compiled_n_swap", -1)) == 0
        compile_error = ""

    except Exception as exc:
        cm = {}
        strict_pass = False
        compile_error = str(exc)

    status = backend.status()

    row = {
        "backend": BACKEND_NAME,
        "pending_jobs": int(status.pending_jobs),
        "layout": old_json,
        "physical_C_t": old_layout[0],
        "physical_D": old_layout[1],
        "physical_P_t": old_layout[2],
        "physical_H": old_layout[3],
        "physical_M1": old_layout[4],
        "physical_M2": old_layout[5],
        **metrics,
        **cm,
        "strict_compile_pass": strict_pass,
        "compile_error": compile_error,
    }

    return row, False


def enrich_rank_with_old(ranked, old_row):
    if not bool(old_row.get("strict_compile_pass", False)):
        return ranked.copy(), np.nan

    work = ranked.copy()

    if not np.any(
        work["layout"].apply(lambda x: layout_key(parse_layout(x)))
        == layout_key(parse_layout(old_row["layout"]))
    ):
        work = pd.concat(
            [work, pd.DataFrame([old_row])],
            ignore_index=True,
        )

    work = strict_rule5_sort(work)
    work["rule5_rank_current"] = np.arange(len(work)) + 1

    old_mask = work["layout"].apply(
        lambda x: layout_key(parse_layout(x))
        == layout_key(parse_layout(old_row["layout"]))
    )

    old_rank = int(work.loc[old_mask, "rule5_rank_current"].iloc[0])
    return work, old_rank


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
    """
    Parameterized RWP core WITHOUT basis rotation and WITHOUT measurement.

    This is used for:
      B) transpiled ideal direct expectation values
      D) transpiled quantum-noisy direct expectation values

    Because there is no measurement instruction, backend readout error cannot
    enter B or D.
    """
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
    if setting_label not in SETTINGS:
        raise KeyError(setting_label)

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



def logical_sparse_observable(feature):
    """
    Build the requested observable on the ORIGINAL six logical qubits.

    Qiskit SparsePauliOp.from_sparse_list uses:
        (local_pauli_string, logical_qubit_indices, coefficient)

    Therefore ("YX", [4, 5], 1.0) means Y on logical q4 and X on logical q5.
    """
    if feature == "X0":
        return SparsePauliOp.from_sparse_list([("X", [0], 1.0)], num_qubits=6)
    if feature == "X1":
        return SparsePauliOp.from_sparse_list([("X", [1], 1.0)], num_qubits=6)
    if feature == "X2":
        return SparsePauliOp.from_sparse_list([("X", [2], 1.0)], num_qubits=6)
    if feature == "Z0":
        return SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=6)
    if feature == "Z1":
        return SparsePauliOp.from_sparse_list([("Z", [1], 1.0)], num_qubits=6)
    if feature == "Z2":
        return SparsePauliOp.from_sparse_list([("Z", [2], 1.0)], num_qubits=6)
    if feature == "Z3":
        return SparsePauliOp.from_sparse_list([("Z", [3], 1.0)], num_qubits=6)
    if feature == "YX45":
        return SparsePauliOp.from_sparse_list([("YX", [4, 5], 1.0)], num_qubits=6)
    raise KeyError(feature)




# =============================================================================
# COUNT -> FEATURES
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
            single_exp_from_counts(counts_a, 0),  # X0
            single_exp_from_counts(counts_a, 1),  # X1
            single_exp_from_counts(counts_a, 2),  # X2
            single_exp_from_counts(counts_z, 0),  # Z0
            single_exp_from_counts(counts_z, 1),  # Z1
            single_exp_from_counts(counts_z, 2),  # Z2
            single_exp_from_counts(counts_z, 3),  # Z3
            parity_exp_from_counts(counts_a, [4, 5]),  # Y4 X5
        ],
        dtype=float,
    )


# =============================================================================
# IDEAL FEATURE BANK + FROZEN RIDGE
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

    if len(y_val) != common.N_VAL:
        raise RuntimeError(
            f"Expected {common.N_VAL} validation endpoints; got {len(y_val)}."
        )

    # Re-select lambda strictly inside training as an audit.
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

    y_train_sd = float(np.std(y_train, ddof=0))

    metrics = {
        "selected_lambda_recomputed": float(selected_lambda),
        "cv_rmse_recomputed": float(summary.iloc[0]["cv_rmse_mean"]),
        "cv_sd_recomputed": float(summary.iloc[0]["cv_rmse_std"]),
        "ideal_2025_rmse": rmse(y_val, pred_val),
        "ideal_2025_mae": mae(y_val, pred_val),
        "ideal_2025_bias": bias(y_val, pred_val),
        "ideal_2025_nrmse_train_sd": rmse(y_val, pred_val) / y_train_sd,
        "train_target_sd": y_train_sd,
    }

    return {
        "work_tv": work_tv,
        "cols": cols,
        "y_all": y_all,
        "angles": angles,
        "endpoints": endpoints,
        "val_endpoints": val_endpoints,
        "X_train": X_train,
        "y_train": y_train,
        "X_val_ideal": X_val,
        "y_val": y_val,
        "pred_val_ideal": pred_val,
        "model": model,
        "scaler": scaler,
        "keep": keep,
        "metrics": metrics,
    }


# =============================================================================
# SIMULATION
# =============================================================================

def resource_metrics_for_settings(transpiled_by_setting, backend):
    rows = []

    for label, circuit in transpiled_by_setting.items():
        ops = {
            str(k): int(v)
            for k, v in circuit.count_ops().items()
        }

        try:
            duration_us = (
                float(
                    circuit.estimate_duration(
                        backend.target,
                        unit="s",
                    )
                )
                * 1e6
            )
        except Exception:
            duration_us = np.nan

        rows.append({
            "setting": label,
            "depth": int(circuit.depth()),
            "size": int(circuit.size()),
            "n_cz": int(ops.get("cz", 0)),
            "n_swap": int(ops.get("swap", 0)),
            "n_reset": int(ops.get("reset", 0)),
            "n_measure": int(ops.get("measure", 0)),
            "duration_us": duration_us,
            "operations_json": json.dumps(ops, sort_keys=True),
        })

    df = pd.DataFrame(rows)

    summary = {
        "n_settings": int(len(df)),
        "feature_vector_n_cz": int(df["n_cz"].sum()),
        "feature_vector_n_swap": int(df["n_swap"].sum()),
        "max_setting_depth": int(df["depth"].max()),
        "sum_setting_depth": int(df["depth"].sum()),
        "feature_vector_duration_us": float(df["duration_us"].sum()),
        "max_setting_duration_us": float(df["duration_us"].max()),
        "feature_vector_reset_count": int(df["n_reset"].sum()),
    }

    return df, summary


def simulate_embedding(
    backend,
    candidate,
    layout,
    angles,
    val_endpoints,
    shots,
    batch_size,
    seed,
):
    """
    Evaluate one physical embedding in four embedding-dependent modes:

      B. transpiled ideal DIRECT expectation values
      C. transpiled ideal SAMPLED finite-shot measurement
      D. transpiled quantum-noisy DIRECT expectation values
      E. transpiled full-noisy SAMPLED measurement

    A. exact logical ideal comes from ideal_reference() and is
       embedding-independent.

    Implementation note
    -------------------
    B and D use Aer EstimatorV2 directly.

    The observable workflow is exactly:

        logical SparsePauliOp
             -> apply_layout(transpiled_core.layout)
             -> Aer EstimatorV2

    Therefore there is NO manual interpretation of physical-qubit numbers as
    circuit-wire indices.

    D uses the backend-derived quantum noise model.  Estimator evaluates
    expectation values without measurement instructions, so backend readout
    assignment error does not act in D.

    E uses measured circuits with AerSimulator.from_backend(), so it contains:
        quantum noise + readout error + finite-shot sampling.

    No mitigation is applied anywhere.
    """

    # --------------------------------------------------------------
    # 1) Transpile measured circuits for C and E
    # --------------------------------------------------------------
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
                f"Layout {layout} unexpectedly introduced SWAPs "
                f"in measurement setting {setting_label}."
            )

        transpiled_meas[setting_label] = tqc
        meas_params[setting_label] = pars

    # --------------------------------------------------------------
    # 2) Transpile unmeasured core for B and D
    # --------------------------------------------------------------
    logical_core, core_params = build_parameterized_core_circuit(candidate)

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
            f"Layout {layout} unexpectedly introduced SWAPs in core circuit."
        )

    # Authoritative post-transpile observables.
    mapped_observables = [
        logical_sparse_observable(feature).apply_layout(
            transpiled_core.layout
        )
        for feature in FEATURES
    ]

    observable_audit = {}
    for feature, obs in zip(FEATURES, mapped_observables):
        observable_audit[feature] = {
            "num_qubits": int(obs.num_qubits),
            "sparse_terms": [
                [
                    str(pauli),
                    [int(q) for q in qargs],
                    [float(np.real(coeff)), float(np.imag(coeff))],
                ]
                for pauli, qargs, coeff in obs.to_sparse_list()
            ],
        }

    # --------------------------------------------------------------
    # 3) Physical resource metrics
    # --------------------------------------------------------------
    setting_resources, resource_summary = resource_metrics_for_settings(
        transpiled_meas,
        backend,
    )

    core_ops = {
        str(k): int(v)
        for k, v in transpiled_core.count_ops().items()
    }

    try:
        core_duration_us = (
            float(
                transpiled_core.estimate_duration(
                    backend.target,
                    unit="s",
                )
            )
            * 1e6
        )
    except Exception:
        core_duration_us = np.nan

    resource_summary.update({
        "core_n_cz": int(core_ops.get("cz", 0)),
        "core_n_swap": int(core_ops.get("swap", 0)),
        "core_depth": int(transpiled_core.depth()),
        "core_duration_us": core_duration_us,
        "requested_physical_layout": json.dumps(layout),
        "mapped_observables_json": json.dumps(
            observable_audit,
            sort_keys=True,
        ),
    })

    # --------------------------------------------------------------
    # 4) Direct expectation values: B and D via Aer EstimatorV2
    # --------------------------------------------------------------
    # Exact direct Estimator. default_precision=0 means no artificial
    # estimator sampling noise.
    estimator_B = AerEstimatorV2(
        options={
            "default_precision": 0.0,
            "backend_options": {
                "method": "density_matrix",
                "enable_truncation": True,
            },
        }
    )

    # Backend-derived noise model. It may contain readout-error entries, but
    # Estimator circuits have no measurements, so readout assignment errors
    # cannot act in D.
    full_noise_model = NoiseModel.from_backend(backend)

    estimator_D = AerEstimatorV2(
        options={
            "default_precision": 0.0,
            "backend_options": {
                "method": "density_matrix",
                "noise_model": full_noise_model,
                "enable_truncation": True,
            },
        }
    )

    def run_estimator_direct(estimator):
        X = np.empty(
            (len(val_endpoints), len(FEATURES)),
            dtype=float,
        )

        # Bind endpoint circuits first. Using bound circuits keeps the PUB
        # structure simple and makes debugging much easier.
        bound_cores = []
        for endpoint in val_endpoints:
            bound_cores.append(
                bind_endpoint_circuit(
                    transpiled_core,
                    core_params,
                    angles,
                    int(endpoint),
                )
            )

        for start in range(0, len(bound_cores), int(batch_size)):
            stop = min(start + int(batch_size), len(bound_cores))

            pubs = [
                (
                    bound_cores[i],
                    mapped_observables,
                )
                for i in range(start, stop)
            ]

            result = estimator.run(
                pubs,
                precision=0.0,
            ).result()

            if len(result) != len(pubs):
                raise RuntimeError(
                    "Estimator result length mismatch: "
                    f"{len(result)} results for {len(pubs)} PUBs."
                )

            for local_i, pub_result in enumerate(result):
                evs = np.asarray(
                    pub_result.data.evs,
                    dtype=float,
                ).reshape(-1)

                if evs.size != len(FEATURES):
                    raise RuntimeError(
                        "Estimator returned unexpected expectation-value "
                        f"shape {np.asarray(pub_result.data.evs).shape}; "
                        f"expected {len(FEATURES)} observables."
                    )

                X[start + local_i, :] = evs

        return X

    X_B_direct_ideal = run_estimator_direct(estimator_B)
    X_D_direct_noisy = run_estimator_direct(estimator_D)

    # --------------------------------------------------------------
    # 5) Finite-shot sampled measurements: C and E
    # --------------------------------------------------------------
    sim_sampled_ideal = AerSimulator(
        enable_truncation=True,
    )

    sim_sampled_full = AerSimulator.from_backend(backend)
    sim_sampled_full.set_options(enable_truncation=True)

    sampled_circuits = []
    sampled_metadata = []

    for endpoint in val_endpoints:
        for setting_label in SETTINGS:
            bound = bind_endpoint_circuit(
                transpiled_meas[setting_label],
                meas_params[setting_label],
                angles,
                int(endpoint),
            )

            sampled_circuits.append(bound)
            sampled_metadata.append(
                (int(endpoint), setting_label)
            )

    def run_sampled(simulator, mode_seed):
        counts_by_endpoint = {
            int(e): {}
            for e in val_endpoints
        }

        for start in range(
            0,
            len(sampled_circuits),
            int(batch_size),
        ):
            stop = min(
                start + int(batch_size),
                len(sampled_circuits),
            )

            sub = sampled_circuits[start:stop]
            submeta = sampled_metadata[start:stop]

            result = simulator.run(
                sub,
                shots=int(shots),
                seed_simulator=int(mode_seed + start),
            ).result()

            for local_i, (endpoint, setting_label) in enumerate(submeta):
                counts_by_endpoint[endpoint][setting_label] = (
                    result.get_counts(local_i)
                )

        X = []

        for endpoint in val_endpoints:
            pair = counts_by_endpoint[int(endpoint)]

            if set(pair) != set(SETTINGS):
                raise RuntimeError(
                    f"Missing setting counts for endpoint {endpoint}."
                )

            X.append(
                combine_feature_pair(
                    pair["XXXZYX"],
                    pair["ZZZZZZ"],
                )
            )

        return np.asarray(X, dtype=float)

    X_C_sampled_ideal = run_sampled(
        sim_sampled_ideal,
        seed,
    )

    X_E_sampled_full = run_sampled(
        sim_sampled_full,
        seed + 100000,
    )

    return {
        "X_B_direct_ideal": X_B_direct_ideal,
        "X_C_sampled_ideal": X_C_sampled_ideal,
        "X_D_direct_noisy": X_D_direct_noisy,
        "X_E_sampled_full": X_E_sampled_full,
        "setting_resources": setting_resources,
        "resource_summary": resource_summary,
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_feature_bank(
    label,
    X_sim,
    ideal,
    layout,
    embedding_rank_current,
    shots,
    quantum_noise=False,
    readout_noise=False,
    finite_sampling=False,
):
    X_ideal = ideal["X_val_ideal"]
    y_val = ideal["y_val"]

    pred = common.predict_scaled_ridge(
        ideal["model"],
        ideal["scaler"],
        ideal["keep"],
        X_sim,
    )

    pred_ideal = ideal["pred_val_ideal"]

    residual_feature = X_sim - X_ideal

    metrics = {
        "mode": label,
        "layout": json.dumps(layout),
        "rule5_rank_current": embedding_rank_current,
        "shots": (int(shots) if shots is not None else np.nan),
        "quantum_noise": bool(quantum_noise),
        "readout_noise": bool(readout_noise),
        "finite_sampling": bool(finite_sampling),
        "error_mitigation": False,
        "rmse": rmse(y_val, pred),
        "mae": mae(y_val, pred),
        "bias": bias(y_val, pred),
        "nrmse_train_sd": (
            rmse(y_val, pred)
            / ideal["metrics"]["train_target_sd"]
        ),
        "delta_rmse_vs_exact_ideal": (
            rmse(y_val, pred)
            - ideal["metrics"]["ideal_2025_rmse"]
        ),
        "prediction_rmse_vs_exact_ideal": rmse(pred_ideal, pred),
        "prediction_mae_vs_exact_ideal": mae(pred_ideal, pred),
        "prediction_corr_vs_exact_ideal": safe_corr(pred_ideal, pred),
        "feature_mae_vs_exact_ideal": float(
            np.mean(np.abs(residual_feature))
        ),
        "feature_rmse_vs_exact_ideal": float(
            np.sqrt(np.mean(residual_feature ** 2))
        ),
        "max_abs_feature_bias": float(
            np.max(
                np.abs(
                    np.mean(residual_feature, axis=0)
                )
            )
        ),
    }

    prediction_df = pd.DataFrame({
        "endpoint": ideal["val_endpoints"],
        "y_true": y_val,
        "prediction": pred,
        "prediction_exact_ideal": pred_ideal,
        "prediction_error": pred - y_val,
        "prediction_shift_vs_exact_ideal": pred - pred_ideal,
    })

    feature_rows = []

    for j, feat in enumerate(FEATURES):
        diff = residual_feature[:, j]

        feature_rows.append({
            "mode": label,
            "layout": json.dumps(layout),
            "rule5_rank_current": embedding_rank_current,
            "feature": feat,
            "feature_mean_exact": float(np.mean(X_ideal[:, j])),
            "feature_mean_sim": float(np.mean(X_sim[:, j])),
            "feature_bias": float(np.mean(diff)),
            "feature_mae": float(np.mean(np.abs(diff))),
            "feature_rmse": float(np.sqrt(np.mean(diff ** 2))),
            "feature_corr": safe_corr(X_ideal[:, j], X_sim[:, j]),
        })

    return metrics, prediction_df, pd.DataFrame(feature_rows)


# =============================================================================
# CALIBRATION DRIFT
# =============================================================================

def normalized_current_metrics(row):
    return {
        "max_2q_error_percent": float(row["compiled_2q_error_max_percent"]),
        "max_1q_error_percent": float(row["compiled_1q_error_max_percent"]),
        "max_readout_error_percent": float(row["max_readout_error_percent"]),
        "min_t1_us": float(row["min_t1_us"]),
        "min_t2_us": float(row["min_t2_us"]),
        "core_duration_us": float(row["compiled_duration_us"]),
    }


def drift_table(old_metrics, current_metrics):
    rows = []

    for metric in old_metrics:
        old = float(old_metrics[metric])
        new = float(current_metrics[metric])
        delta = new - old
        pct = (
            100.0 * delta / old
            if np.isfinite(old) and old != 0.0
            else np.nan
        )

        rows.append({
            "metric": metric,
            "week9_value": old,
            "current_value": new,
            "absolute_change": delta,
            "relative_change_percent": pct,
        })

    return pd.DataFrame(rows)


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
        "--shortlist",
        type=int,
        default=DEFAULT_SHORTLIST,
        help=(
            "Number of calibration-hierarchy embeddings added to the Pareto "
            "front before strict physical compilation."
        ),
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
        help="Plumbing check: 30 validation endpoints and 256 shots.",
    )

    args = parser.parse_args()

    shots = 256 if args.quick else int(args.shots)

    print("=" * 122)
    print("10.01 — REFERENCE QRC + FRESH KINGSTON EMBEDDINGS + IDEAL/NOISY AER")
    print("=" * 122)
    print("NO QPU JOBS ARE SUBMITTED.")
    print("2026 remains frozen and is not loaded.")
    print(f"Embedding helper: {SELECTOR_FILE.name}")
    print("Direct B/D engine: Aer EstimatorV2 + SparsePauliOp.apply_layout()")
    print()

    # ------------------------------------------------------------------
    # A. Freeze logical candidate and ideal readout
    # ------------------------------------------------------------------
    candidate, stage_row, frozen = reconstruct_reference_candidate()

    print("FROZEN REFERENCE")
    print(
        f"  H3 / RWP_W4 / alpha={candidate['alpha']} / "
        f"r={candidate['r']} / {READOUT}"
    )
    print(f"  hx={candidate['hx']:.12f}, hy={candidate['hy']:.12f}, dt={candidate['dt']}")
    print(f"  J={candidate['J']}")
    print(
        f"  Week-9 CV={float(stage_row['cv_rmse']):.6f}, "
        f"2025 RMSE={float(stage_row['validation_rmse']):.6f}"
    )
    print()

    ideal = ideal_reference(candidate)

    print("IDEAL RECOMPUTATION AUDIT")
    for k, v in ideal["metrics"].items():
        print(f"  {k:<30} {v:.9f}" if isinstance(v, float) else f"  {k:<30} {v}")

    if not np.isclose(
        ideal["metrics"]["selected_lambda_recomputed"],
        float(stage_row["selected_lambda"]),
    ):
        raise RuntimeError(
            "Recomputed Ridge lambda differs from Stage-C reference."
        )

    # Loose numerical audit: independent recomputation should reproduce Week 9.
    if abs(
        ideal["metrics"]["ideal_2025_rmse"]
        - float(stage_row["validation_rmse"])
    ) > 1e-6:
        raise RuntimeError(
            "Ideal 2025 RMSE does not reproduce the Week-9 reference."
        )

    # ------------------------------------------------------------------
    # B. Old embedding + fresh search
    # ------------------------------------------------------------------
    old_layout, old_metrics, old_file_mtime = load_old_embedding()

    service = get_service()

    fresh, ranked = refresh_and_rank(
        candidate,
        service,
        args.shortlist,
    )

    # Get a backend object after the refresh for simulation.
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

    old_row, old_was_in_compiled_shortlist = current_row_for_old_embedding(
        old_layout,
        candidate,
        fresh,
        ranked,
        service,
    )

    ranked_with_old, old_current_rank = enrich_rank_with_old(
        ranked,
        old_row,
    )

    top3 = ranked_with_old.head(3).copy()

    top3_layouts = [
        parse_layout(x)
        for x in top3["layout"]
    ]

    old_in_top3 = layout_key(old_layout) in {
        layout_key(x)
        for x in top3_layouts
    }

    old_still_valid = bool(
        old_row.get("strict_compile_pass", False)
    )

    if old_still_valid:
        current_old_metrics = normalized_current_metrics(
            pd.Series(old_row)
        )
        drift = drift_table(
            old_metrics,
            current_old_metrics,
        )
    else:
        drift = pd.DataFrame([{
            "metric": "old_embedding_validity",
            "week9_value": 1.0,
            "current_value": 0.0,
            "absolute_change": -1.0,
            "relative_change_percent": -100.0,
        }])

    print()
    print("OLD WEEK-9 KINGSTON H3 EMBEDDING")
    print(f"  layout: {old_layout}")
    print(f"  still native / strict zero-SWAP: {old_still_valid}")
    print(f"  rank in current compiled Rule-5 set: {old_current_rank}")
    print(f"  among current top 3: {old_in_top3}")
    print()

    print("CURRENT KINGSTON RULE-5 TOP 3")
    show_cols = [
        "rule5_rank_current",
        "layout",
        "compiled_2q_error_max_percent",
        "max_readout_error_percent",
        "compiled_1q_error_max_percent",
        "min_t1_us",
        "min_t2_us",
        "compiled_duration_us",
        "compiled_n_swap",
    ]

    print(top3[show_cols].to_string(index=False))
    print()

    if old_still_valid:
        print("SAME OLD EMBEDDING: WEEK-9 -> CURRENT CALIBRATION")
        print(drift.to_string(index=False))
        print()

    # ------------------------------------------------------------------
    # C. Embeddings to simulate = current top3 U old
    # ------------------------------------------------------------------
    tested = []

    for _, row in top3.iterrows():
        tested.append({
            "source": "fresh_top3",
            "rank": int(row["rule5_rank_current"]),
            "layout": parse_layout(row["layout"]),
            "rule5_row": row.to_dict(),
        })

    if old_still_valid and layout_key(old_layout) not in {
        layout_key(x["layout"])
        for x in tested
    }:
        old_ranked_row = ranked_with_old[
            ranked_with_old["layout"].apply(
                lambda x: layout_key(parse_layout(x))
                == layout_key(old_layout)
            )
        ].iloc[0]

        tested.append({
            "source": "old_week9_extra",
            "rank": int(old_ranked_row["rule5_rank_current"]),
            "layout": old_layout,
            "rule5_row": old_ranked_row.to_dict(),
        })

    # Validation subset only for --quick.
    val_endpoints = ideal["val_endpoints"]

    if args.quick:
        idx = np.linspace(
            0,
            len(val_endpoints) - 1,
            30,
            dtype=int,
        )
        val_endpoints_run = val_endpoints[idx]

        # Subset ideal arrays in the exact same order.
        ideal_run = dict(ideal)
        ideal_run["val_endpoints"] = val_endpoints_run
        ideal_run["X_val_ideal"] = ideal["X_val_ideal"][idx]
        ideal_run["y_val"] = ideal["y_val"][idx]
        ideal_run["pred_val_ideal"] = ideal["pred_val_ideal"][idx]
    else:
        ideal_run = ideal
        val_endpoints_run = val_endpoints

    # ------------------------------------------------------------------
    # D. Simulate top3 + old if needed
    # ------------------------------------------------------------------
    comparison_rows = []
    feature_parts = []
    prediction_parts = []
    resource_parts = []

    print()
    print(
        f"SIMULATING {len(tested)} UNIQUE EMBEDDING(S), "
        f"{len(val_endpoints_run)} VALIDATION ENDPOINTS, {shots} SHOTS"
    )
    print("  Five references:")
    print("    A exact logical ideal direct expectation (embedding-independent)")
    print("    B transpiled ideal direct expectation")
    print("    C transpiled ideal finite-shot sampling")
    print("    D transpiled quantum-noisy direct expectation (NO readout)")
    print("    E transpiled full-noisy finite-shot measurement")
    print("  Error mitigation: OFF")
    print()

    for pos, item in enumerate(tested, start=1):
        layout = item["layout"]
        rank = item["rank"]

        print(
            f"[{pos:02d}/{len(tested):02d}] "
            f"rank={rank}, source={item['source']}, layout={layout}"
        )

        t0 = time.time()

        sim = simulate_embedding(
            backend=backend,
            candidate=candidate,
            layout=layout,
            angles=ideal["angles"],
            val_endpoints=val_endpoints_run,
            shots=shots,
            batch_size=args.batch_size,
            # Common random numbers across embeddings: the same finite-shot
            # simulator seeds are used so embedding differences are not dominated
            # by arbitrary seed changes.
            seed=args.sim_seed,
        )

        resource = sim["resource_summary"]

        # B must reproduce A to numerical precision.  If it does not, direct
        # expectation extraction is still wrong and the experiment must stop.
        B_feature_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        sim["X_B_direct_ideal"]
                        - ideal_run["X_val_ideal"]
                    ) ** 2
                )
            )
        )

        if B_feature_rmse > 1e-6:
            raise RuntimeError(
                "A/B equivalence audit FAILED even with the EstimatorV2 path for layout "
                f"{layout}: feature RMSE={B_feature_rmse:.3e}. "
                "Stop: logical/transpiled circuit equivalence must be debugged."
            )

        resource = sim["resource_summary"]

        resource_row = {
            "layout": json.dumps(layout),
            "rule5_rank_current": rank,
            "source": item["source"],
            "A_B_feature_rmse_audit": B_feature_rmse,
            **resource,
        }
        resource_parts.append(resource_row)

        simulation_modes = [
            (
                "B_transpiled_ideal_direct",
                sim["X_B_direct_ideal"],
                None,
                False,
                False,
                False,
            ),
            (
                "C_transpiled_ideal_sampled",
                sim["X_C_sampled_ideal"],
                shots,
                False,
                False,
                True,
            ),
            (
                "D_transpiled_quantum_noise_direct",
                sim["X_D_direct_noisy"],
                None,
                True,
                False,
                False,
            ),
            (
                "E_transpiled_full_noise_sampled",
                sim["X_E_sampled_full"],
                shots,
                True,
                True,
                True,
            ),
        ]

        for (
            mode_label,
            X_sim,
            mode_shots,
            quantum_noise,
            readout_noise,
            finite_sampling,
        ) in simulation_modes:
            m, pred_df, feat_df = evaluate_feature_bank(
                mode_label,
                X_sim,
                ideal_run,
                layout,
                rank,
                mode_shots,
                quantum_noise=quantum_noise,
                readout_noise=readout_noise,
                finite_sampling=finite_sampling,
            )

            rule5_row = item["rule5_row"]

            m.update({
                "source": item["source"],
                "max_2q_error_percent": float(
                    rule5_row["compiled_2q_error_max_percent"]
                ),
                "max_1q_error_percent": float(
                    rule5_row["compiled_1q_error_max_percent"]
                ),
                "max_readout_error_percent": float(
                    rule5_row["max_readout_error_percent"]
                ),
                "min_t1_us": float(rule5_row["min_t1_us"]),
                "min_t2_us": float(rule5_row["min_t2_us"]),
                **resource,
            })

            comparison_rows.append(m)

            pred_df.insert(0, "mode", mode_label)
            pred_df.insert(0, "rule5_rank_current", rank)
            pred_df.insert(0, "layout", json.dumps(layout))
            prediction_parts.append(pred_df)

            feat_df.insert(0, "source", item["source"])
            feature_parts.append(feat_df)

        elapsed = time.time() - t0

        print(
            f"    finished in {elapsed / 60.0:.2f} min | "
            f"CZ/feature={resource['feature_vector_n_cz']}, "
            f"depth={resource['max_setting_depth']}, "
            f"Tfeature={resource['feature_vector_duration_us']:.3f} us"
        )

    comparison = pd.DataFrame(comparison_rows)
    features = pd.concat(feature_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    resources = pd.DataFrame(resource_parts)

    # Add exact ideal row once; embedding-independent.
    exact_row = {
        "mode": "A_exact_logical_ideal_direct",
        "quantum_noise": False,
        "readout_noise": False,
        "finite_sampling": False,
        "error_mitigation": False,
        "layout": "embedding-independent",
        "rule5_rank_current": np.nan,
        "shots": np.nan,
        "rmse": rmse(
            ideal_run["y_val"],
            ideal_run["pred_val_ideal"],
        ),
        "mae": mae(
            ideal_run["y_val"],
            ideal_run["pred_val_ideal"],
        ),
        "bias": bias(
            ideal_run["y_val"],
            ideal_run["pred_val_ideal"],
        ),
        "nrmse_train_sd": (
            rmse(
                ideal_run["y_val"],
                ideal_run["pred_val_ideal"],
            )
            / ideal["metrics"]["train_target_sd"]
        ),
        "delta_rmse_vs_exact_ideal": 0.0,
        "prediction_rmse_vs_exact_ideal": 0.0,
        "prediction_mae_vs_exact_ideal": 0.0,
        "prediction_corr_vs_exact_ideal": 1.0,
        "feature_mae_vs_exact_ideal": 0.0,
        "feature_rmse_vs_exact_ideal": 0.0,
        "max_abs_feature_bias": 0.0,
        "source": "logical_reference",
    }

    comparison = pd.concat(
        [pd.DataFrame([exact_row]), comparison],
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Explicit A-E decomposition for each physical embedding
    # --------------------------------------------------------------
    decomposition_rows = []

    A_rmse = float(exact_row["rmse"])

    for item in tested:
        layout_json = json.dumps(item["layout"])
        rank = item["rank"]

        g = comparison[
            comparison["layout"].astype(str) == layout_json
        ].copy()

        def mode_rmse(name):
            r = g[g["mode"] == name]
            if len(r) != 1:
                raise RuntimeError(
                    f"Expected one {name} row for layout {layout_json}; "
                    f"found {len(r)}."
                )
            return float(r.iloc[0]["rmse"])

        B_rmse = mode_rmse("B_transpiled_ideal_direct")
        C_rmse = mode_rmse("C_transpiled_ideal_sampled")
        D_rmse = mode_rmse("D_transpiled_quantum_noise_direct")
        E_rmse = mode_rmse("E_transpiled_full_noise_sampled")

        decomposition_rows.append({
            "layout": layout_json,
            "rule5_rank_current": rank,
            "A_exact_logical_ideal_rmse": A_rmse,
            "B_transpiled_ideal_direct_rmse": B_rmse,
            "C_transpiled_ideal_sampled_rmse": C_rmse,
            "D_quantum_noise_direct_rmse": D_rmse,
            "E_full_noise_sampled_rmse": E_rmse,
            "B_minus_A_compilation_effect": B_rmse - A_rmse,
            "C_minus_B_sampling_effect": C_rmse - B_rmse,
            "D_minus_B_quantum_noise_effect": D_rmse - B_rmse,
            "E_minus_D_sampling_plus_readout_effect": E_rmse - D_rmse,
            "E_minus_A_total_end_to_end_effect": E_rmse - A_rmse,
            "readout_mitigation_used": False,
        })

    decomposition = pd.DataFrame(decomposition_rows)

    # ------------------------------------------------------------------
    # E. Save
    # ------------------------------------------------------------------
    status_df = fresh["backend_status"].copy()

    top3_out = top3.copy()
    top3_out["old_week9_layout"] = top3_out["layout"].apply(
        lambda x: layout_key(parse_layout(x)) == layout_key(old_layout)
    )

    frozen.update({
        "backend": BACKEND_NAME,
        "old_week9_layout": old_layout,
        "old_week9_embedding_file_mtime_utc": old_file_mtime,
        "current_backend_properties_last_update": json_safe(
            calibration_time
        ),
        "old_embedding_still_valid": old_still_valid,
        "old_embedding_current_rule5_rank": json_safe(old_current_rank),
        "old_embedding_in_current_top3": old_in_top3,
        "current_top3_layouts": top3_layouts,
        "simulation_shots": shots,
        "simulation_validation_endpoint_count": len(val_endpoints_run),
        "quick_mode": bool(args.quick),
    })

    (RESULTS / "10_01_reference_candidate.json").write_text(
        json.dumps(json_safe(frozen), indent=2),
        encoding="utf-8",
    )

    status_df.to_csv(
        RESULTS / "10_01_kingston_refresh_backend_status.csv",
        index=False,
    )

    top3_out.to_csv(
        RESULTS / "10_01_kingston_fresh_top3_embeddings.csv",
        index=False,
    )

    drift.to_csv(
        RESULTS / "10_01_old_embedding_drift.csv",
        index=False,
    )

    comparison.to_csv(
        RESULTS / "10_01_embedding_comparison.csv",
        index=False,
    )

    features.to_csv(
        RESULTS / "10_01_embedding_feature_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        RESULTS / "10_01_embedding_predictions.csv",
        index=False,
    )

    resources.to_csv(
        RESULTS / "10_01_embedding_resource_metrics.csv",
        index=False,
    )

    decomposition.to_csv(
        RESULTS / "10_01_simulation_reference_decomposition.csv",
        index=False,
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": BACKEND_NAME,
        "current_properties_last_update": json_safe(calibration_time),
        "old_week9_result_file_mtime_utc": old_file_mtime,
        "old_layout": old_layout,
        "old_embedding_still_valid": old_still_valid,
        "old_current_rule5_rank": json_safe(old_current_rank),
        "old_in_current_top3": old_in_top3,
        "fresh_top3": top3_out.to_dict(orient="records"),
        "ideal_metrics": ideal["metrics"],
        "simulation_comparison": comparison.to_dict(orient="records"),
        "resource_metrics": resources.to_dict(orient="records"),
        "A_to_E_decomposition": decomposition.to_dict(orient="records"),
        "notes": [
            "No QPU job submitted.",
            "2026 test set untouched.",
            (
                "A is the original exact logical density-matrix expectation "
                "reference."
            ),
            (
                "B and D are direct expectation values with shots=None. "
                "D contains backend-derived quantum gate/decoherence noise but "
                "no readout error because there is no measurement instruction."
            ),
            (
                "C and E reconstruct Pauli expectations from finite-shot counts. "
                "E additionally includes backend-derived readout error."
            ),
            (
                "No readout-error mitigation, ZNE, PEC, postselection, or other "
                "error mitigation is applied in Week 10.1."
            ),
            (
                "The backend-derived Aer model is an approximate calibration-"
                "based noise model, not a perfect reproduction of live QPU "
                "behavior."
            ),
            (
                "Explicit separation of readout-only, 1Q-only, 2Q-only, and "
                "idle/decoherence-only regimes is intentionally reserved for "
                "Week 10.2."
            ),
        ],
    }

    (RESULTS / "10_01_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # F. Console summary
    # ------------------------------------------------------------------
    print()
    print("=" * 122)
    print("10.01 FINAL SUMMARY")
    print("=" * 122)
    print(
        f"Old layout {old_layout}: valid={old_still_valid}, "
        f"current rank={old_current_rank}, top3={old_in_top3}"
    )
    print()

    print("FRESH TOP 3")
    print(top3_out[show_cols + ["old_week9_layout"]].to_string(index=False))
    print()

    print("SIMULATION / FORECAST METRICS")
    display_cols = [
        "mode",
        "rule5_rank_current",
        "layout",
        "rmse",
        "mae",
        "bias",
        "nrmse_train_sd",
        "delta_rmse_vs_exact_ideal",
        "feature_rmse_vs_exact_ideal",
        "prediction_corr_vs_exact_ideal",
        "max_2q_error_percent",
        "max_readout_error_percent",
        "min_t2_us",
    ]

    existing = [
        c for c in display_cols
        if c in comparison.columns
    ]

    print(
        comparison[existing]
        .sort_values(
            ["mode", "rule5_rank_current"],
            na_position="first",
        )
        .to_string(index=False)
    )

    print()
    print("A-E RMSE DECOMPOSITION")
    print(decomposition.to_string(index=False))

    print()
    print("Saved:")
    for name in [
        "10_01_reference_candidate.json",
        "10_01_kingston_refresh_backend_status.csv",
        "10_01_kingston_fresh_top3_embeddings.csv",
        "10_01_old_embedding_drift.csv",
        "10_01_embedding_comparison.csv",
        "10_01_embedding_feature_metrics.csv",
        "10_01_embedding_predictions.csv",
        "10_01_embedding_resource_metrics.csv",
        "10_01_simulation_reference_decomposition.csv",
        "10_01_summary.json",
    ]:
        print(f"  results/{name}")

    print()
    print("10.01 COMPLETE.")
    print(
        "Stop here and inspect these results before defining the "
        "Week-10.2 noise regimes."
    )


if __name__ == "__main__":
    main()
