
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler

from ibm_account import get_service


# =============================================================================
# WEEK 11.0D — FULL 2025 VALIDATION: REAL-QPU RWP_W1 + XZ_INJECTION
#
# THIS SCRIPT PRESERVES THE PROJECT'S CHRONOLOGICAL PROTOCOL:
#
#   2022-2024  = TRAIN / MODEL ESTABLISHMENT
#   2025       = VALIDATION / EVALUATION
#   2026       = FROZEN / UNTOUCHED
#
# Nothing is fitted, tuned, standardized, or recalibrated using 2025.
# Nothing from 2026 is loaded into this experiment.
#
# Classical readout establishment (2022-2024 ONLY):
#   - Ridge lambda selected by chronological CV inside 2022-2024
#   - feature scaler mean/std fitted on 2022-2024
#   - final Ridge weights fitted on 2022-2024
#   - Ridge intercept fitted on 2022-2024
#
# These are then FROZEN and applied unchanged to:
#   - ideal 2025 W1 features
#   - real-QPU 2025 W1 features
#
# Quantum protocol:
#   - keep current H4 quantum dynamics fixed from the 9.3c winner
#   - W=1 => each day is one independent reservoir execution
#   - all 6 qubits start in |0>
#   - inject q0..q3 once
#   - evolve one H4 reservoir step
#   - measure ONLY q0..q3
#   - two grouped settings: XXXX and ZZZZ
#   - extracted features:
#       X0, X1, X2, X3, Z0, Z1, Z2, Z3
#
# Hardware workload:
#   365 validation days × 2 settings = 730 Sampler circuits
#
# Default = DRY RUN.
#
# Dry run:
#   python 11_0D_RWP_W1_full_2025_validation.py
#
# Real QPU:
#   python 11_0D_RWP_W1_full_2025_validation.py --submit
# =============================================================================


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

SELECTOR_SCRIPT = HERE / "11_0A_live_embedding_reselection.py"
QRC_SCRIPT = HERE / "07_02b_memory_pair_isolation.py"

FAIR_FILE = RESULTS / "09_03c_forecast_best_135.csv"

# Previous real hardware runs are used only for rough planning of QPU charge.
PRIOR_SMOKE_PREFLIGHT = RESULTS / "11_0B_preflight.json"
PRIOR_SMOKE_USAGE = RESULTS / "11_0B_usage.json"
PRIOR_W7_SUMMARY = RESULTS / "11_0C_summary.json"

ALPHA = 0.75
DT = 1.6
WINDOW = 1

N_TRAIN = 1095
N_VAL = 365

DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SHOTS = 256
DEFAULT_SHORTLIST = 80

OPTIMIZATION_LEVEL = 1
SEED_TRANSPILER = 42

VAR_TOL = 1e-12

LAMBDA_GRID = np.array(
    [
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        300.0,
    ],
    dtype=float,
)

TOPOLOGY_EDGES = {
    "H0": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    "H1": [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)],
    "H2": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)],
    "H3": [(2, 3), (1, 2), (0, 1), (0, 4), (4, 5)],
    "H4": [(1, 2), (0, 1), (0, 4), (3, 4), (4, 5)],
}

EDGE_TO_COLUMN = {
    "H0": {
        (0, 1): "J01",
        (1, 2): "J12",
        (2, 3): "J23",
        (3, 4): "J34",
        (4, 5): "J45",
    },
    "H1": {
        (0, 1): "J01",
        (1, 2): "J12",
        (2, 3): "J23",
        (3, 4): "J34",
        (3, 5): "J35",
    },
    "H2": {
        (0, 1): "J01",
        (1, 2): "J12",
        (2, 3): "J23",
        (2, 4): "J24",
        (4, 5): "J45",
    },
    "H3": {
        (2, 3): "J23",
        (1, 2): "J12",
        (0, 1): "J01",
        (0, 4): "J04",
        (4, 5): "J45",
    },
    "H4": {
        (1, 2): "J12",
        (0, 1): "J01",
        (0, 4): "J04",
        (3, 4): "J34",
        (4, 5): "J45",
    },
}

XZ_COLUMNS = [
    "X0",
    "X1",
    "X2",
    "X3",
    "Z0",
    "Z1",
    "Z2",
    "Z3",
]


# =============================================================================
# IMPORT EXISTING PROJECT MODULES
# =============================================================================

def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put this script in the project root."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    spec.loader.exec_module(module)
    return module


selector = import_module_from_path(
    "selector_11_0A",
    SELECTOR_SCRIPT,
)

qrc = import_module_from_path(
    "qrc_week7",
    QRC_SCRIPT,
)

qrc.ALPHA = ALPHA


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def json_safe(obj):
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            json_safe(v)
            for v in obj
        ]

    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass

    return str(obj)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


def bias(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.mean(
            y_pred - y_true
        )
    )


def safe_service_usage(service):
    try:
        return service.usage()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }


# =============================================================================
# LOAD CURRENT H4 QUANTUM DYNAMICS
# =============================================================================

def load_h4_winner_dynamics():
    if not FAIR_FILE.exists():
        raise FileNotFoundError(FAIR_FILE)

    df = pd.read_csv(FAIR_FILE)

    # Absolute forecasting winner from fair 9.3c comparison.
    row = (
        df.sort_values(
            ["cv_rmse", "validation_rmse"]
        )
        .iloc[0]
    )

    topology = str(row["topology"])

    if topology != "H4":
        raise RuntimeError(
            f"Current absolute 9.3c winner is {topology}, not H4. "
            f"Review before using this W1 transfer experiment."
        )

    J = {
        edge: float(row[column])
        for edge, column
        in EDGE_TO_COLUMN[topology].items()
    }

    return {
        "source_candidate_id":
            str(row["seed_candidate_id"]),
        "source_config_id":
            str(row["config_id"]),
        "candidate_id":
            str(row["seed_candidate_id"])
            +
            "__RWP_W1_XZinj",
        "topology":
            topology,
        "readout":
            "XZ_injection",
        "r":
            int(row["trotter_r"]),
        "hx":
            float(row["hx"]),
        "hy":
            float(row["hy"]),
        "J":
            J,
        "source_cv_rmse_XYZ_CONT":
            float(row["cv_rmse"]),
        "source_validation_rmse_XYZ_CONT":
            float(row["validation_rmse"]),
    }


# =============================================================================
# LOAD ONLY TRAIN + VALIDATION
# 2026 IS NOT USED BY THIS SCRIPT
# =============================================================================

def load_train_validation_only():
    work, train, val, cols = qrc.load_data()

    if len(train) != N_TRAIN:
        raise RuntimeError(
            f"Expected {N_TRAIN} training rows; got {len(train)}."
        )

    if len(val) != N_VAL:
        raise RuntimeError(
            f"Expected {N_VAL} validation rows; got {len(val)}."
        )

    # Deliberately discard any later rows present in `work`.
    # The experiment is constructed strictly from train + validation.
    work_tv = pd.concat(
        [train, val],
        axis=0,
    ).reset_index(drop=True)

    if len(work_tv) != N_TRAIN + N_VAL:
        raise RuntimeError(
            "Unexpected train+validation alignment."
        )

    angles = np.asarray(
        qrc.make_input_angles(
            work_tv,
            cols,
        ),
        dtype=float,
    )

    y = work_tv[
        cols["target"]
    ].to_numpy(dtype=float)

    train_idx = np.arange(
        0,
        N_TRAIN,
        dtype=int,
    )

    val_idx = np.arange(
        N_TRAIN,
        N_TRAIN + N_VAL,
        dtype=int,
    )

    return (
        work_tv,
        cols,
        angles,
        y,
        train_idx,
        val_idx,
    )


# =============================================================================
# IDEAL W1 QUANTUM FEATURES
# =============================================================================

I64 = qrc.I64

INJ_OPS = getattr(
    qrc,
    "INJECTION_SINGLE_OPS",
    getattr(
        qrc,
        "INJECTION_OPS",
        None,
    ),
)

if INJ_OPS is None:
    raise AttributeError(
        "Could not resolve injection-qubit Pauli operators."
    )

ZZ_OPS = {}

for topology, edges in TOPOLOGY_EDGES.items():
    for edge in edges:
        if edge not in ZZ_OPS:
            ZZ_OPS[edge] = qrc.pauli_product_full(
                qrc.Z2,
                edge[0],
                qrc.Z2,
                edge[1],
            )


def pauli_exp(operator, theta):
    return (
        np.cos(theta) * I64
        -
        1j * np.sin(theta) * operator
    )


def build_trotter_unitary(candidate):
    topology = candidate["topology"]
    r = int(candidate["r"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])
    J = candidate["J"]

    U = I64.copy()

    for _ in range(r):
        # ZZ interactions
        for edge in TOPOLOGY_EDGES[topology]:
            theta = (
                float(J[edge])
                *
                DT
                /
                r
            )

            U = (
                pauli_exp(
                    ZZ_OPS[edge],
                    theta,
                )
                @
                U
            )

        # X field on all six qubits
        theta_x = (
            hx
            *
            DT
            /
            r
        )

        for q in range(qrc.N_QUBITS):
            U = (
                pauli_exp(
                    qrc.FULL_SINGLE_OPS[f"X{q}"],
                    theta_x,
                )
                @
                U
            )

        # memory-only Y field
        if not np.isclose(hy, 0.0):
            theta_y = (
                hy
                *
                DT
                /
                r
            )

            for q in [4, 5]:
                U = (
                    pauli_exp(
                        qrc.FULL_SINGLE_OPS[f"Y{q}"],
                        theta_y,
                    )
                    @
                    U
                )

    return U


def expectation(rho, op):
    return float(
        np.real_if_close(
            np.trace(
                rho @ op
            ),
            tol=1000,
        ).real
    )


def build_ideal_w1_feature_bank(
    candidate,
    angles,
):
    U = build_trotter_unitary(
        candidate
    )

    A_list, _ = qrc.build_input_channels(
        U,
        angles,
    )

    rho_m_zero = qrc.memory_zero_density()

    rows = []

    for A in A_list:
        rho_i, _ = qrc.final_reduced_states(
            A,
            rho_m_zero,
        )

        row = []

        for q in range(4):
            row.append(
                expectation(
                    rho_i,
                    INJ_OPS[f"X{q}"],
                )
            )

        for q in range(4):
            row.append(
                expectation(
                    rho_i,
                    INJ_OPS[f"Z{q}"],
                )
            )

        rows.append(row)

    return np.asarray(
        rows,
        dtype=float,
    )


# =============================================================================
# RIDGE READOUT
# EVERYTHING HERE IS ESTABLISHED FROM 2022-2024 ONLY
# =============================================================================

def fit_scaled_ridge(
    X,
    y,
    ridge_alpha,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    std = np.std(
        X,
        axis=0,
        ddof=0,
    )

    keep = std > VAR_TOL

    if not np.any(keep):
        raise RuntimeError(
            "No nonconstant W1 XZ-injection features."
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X[:, keep]
    )

    model = Ridge(
        alpha=float(ridge_alpha),
        fit_intercept=True,
    )

    model.fit(
        X_scaled,
        y,
    )

    return (
        model,
        scaler,
        keep,
    )


def predict_scaled_ridge(
    model,
    scaler,
    keep,
    X,
):
    X = np.asarray(
        X,
        dtype=float,
    )

    X_scaled = scaler.transform(
        X[:, keep]
    )

    return model.predict(
        X_scaled
    )


def select_lambda_training_only(
    X_train,
    y_train,
):
    splitter = TimeSeriesSplit(
        n_splits=5
    )

    rows = []

    for ridge_alpha in LAMBDA_GRID:
        fold_scores = []

        for fold, (
            fit_idx,
            cv_idx,
        ) in enumerate(
            splitter.split(X_train)
        ):
            (
                model,
                scaler,
                keep,
            ) = fit_scaled_ridge(
                X_train[fit_idx],
                y_train[fit_idx],
                ridge_alpha,
            )

            pred = predict_scaled_ridge(
                model,
                scaler,
                keep,
                X_train[cv_idx],
            )

            score = rmse(
                y_train[cv_idx],
                pred,
            )

            fold_scores.append(score)

            rows.append({
                "lambda":
                    float(ridge_alpha),
                "fold":
                    fold + 1,
                "rmse":
                    score,
            })

    fold_df = pd.DataFrame(rows)

    summary = (
        fold_df.groupby(
            "lambda",
            as_index=False,
        )
        .agg(
            cv_rmse_mean=(
                "rmse",
                "mean",
            ),
            cv_rmse_std=(
                "rmse",
                "std",
            ),
        )
        .sort_values(
            ["cv_rmse_mean", "lambda"]
        )
        .reset_index(drop=True)
    )

    selected_lambda = float(
        summary.iloc[0][
            "lambda"
        ]
    )

    return (
        selected_lambda,
        fold_df,
        summary,
    )


def export_frozen_readout(
    selected_lambda,
    model,
    scaler,
    keep,
):
    active_features = [
        feature
        for feature, active
        in zip(
            XZ_COLUMNS,
            keep,
        )
        if active
    ]

    rows = []

    coef_idx = 0

    for feature, active in zip(
        XZ_COLUMNS,
        keep,
    ):
        if active:
            mu = float(
                scaler.mean_[coef_idx]
            )

            sigma = float(
                scaler.scale_[coef_idx]
            )

            weight_scaled = float(
                model.coef_[coef_idx]
            )

            rows.append({
                "feature":
                    feature,
                "active":
                    True,
                "train_mean":
                    mu,
                "train_std":
                    sigma,
                "ridge_weight_on_standardized_feature":
                    weight_scaled,
            })

            coef_idx += 1

        else:
            rows.append({
                "feature":
                    feature,
                "active":
                    False,
                "train_mean":
                    np.nan,
                "train_std":
                    np.nan,
                "ridge_weight_on_standardized_feature":
                    0.0,
            })

    weights_df = pd.DataFrame(rows)

    weights_df.to_csv(
        RESULTS
        /
        "11_0D_frozen_readout_weights.csv",
        index=False,
    )

    freeze_record = {
        "established_from":
            "2022-2024 training only",
        "validation_period":
            "2025",
        "2026_status":
            "frozen / unused",
        "selected_lambda":
            float(selected_lambda),
        "ridge_intercept":
            float(model.intercept_),
        "active_features":
            active_features,
        "feature_order":
            XZ_COLUMNS,
    }

    with open(
        RESULTS
        /
        "11_0D_frozen_readout.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            freeze_record,
            fp,
            indent=2,
        )

    return (
        weights_df,
        freeze_record,
    )


# =============================================================================
# HARDWARE W1 CIRCUITS
# =============================================================================

def append_one_h4_step(
    qc,
    candidate,
    angle_row,
):
    # W1 starts from fresh |000000>; no reset is needed inside the circuit.
    for q in range(4):
        qc.ry(
            float(angle_row[q]),
            q,
        )

    topology = candidate["topology"]
    r = int(candidate["r"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])
    J = candidate["J"]

    for _ in range(r):
        for i, j in TOPOLOGY_EDGES[topology]:
            qc.rzz(
                2.0
                *
                float(J[(i, j)])
                *
                DT
                /
                r,
                i,
                j,
            )

        theta_x = (
            2.0
            *
            hx
            *
            DT
            /
            r
        )

        for q in range(6):
            qc.rx(
                theta_x,
                q,
            )

        if not np.isclose(hy, 0.0):
            theta_y = (
                2.0
                *
                hy
                *
                DT
                /
                r
            )

            qc.ry(
                theta_y,
                4,
            )

            qc.ry(
                theta_y,
                5,
            )


def build_w1_measurement_circuit(
    candidate,
    angle_row,
    basis,
    endpoint,
):
    qreg = QuantumRegister(
        6,
        "q",
    )

    creg = ClassicalRegister(
        4,
        "inj",
    )

    qc = QuantumCircuit(
        qreg,
        creg,
        name=f"W1_t{endpoint}_{basis}",
    )

    append_one_h4_step(
        qc,
        candidate,
        angle_row,
    )

    if basis == "X":
        for q in range(4):
            qc.h(q)

    elif basis == "Z":
        pass

    else:
        raise ValueError(
            basis
        )

    # Measure injection qubits only.
    for q in range(4):
        qc.measure(
            q,
            q,
        )

    return qc


# =============================================================================
# COUNTS -> XZ INJECTION FEATURES
# =============================================================================

def bit_for_injection_qubit(
    bitstring,
    q,
):
    clean = bitstring.replace(
        " ",
        "",
    )

    # Qiskit prints four-bit classical register as c3 c2 c1 c0.
    return int(
        clean[-(q + 1)]
    )


def expectation_from_counts(
    counts,
    q,
):
    total = int(
        sum(
            counts.values()
        )
    )

    if total <= 0:
        raise RuntimeError(
            "Empty counts."
        )

    acc = 0.0

    for bitstring, count in counts.items():
        bit = bit_for_injection_qubit(
            bitstring,
            q,
        )

        eig = (
            1.0
            if bit == 0
            else -1.0
        )

        acc += (
            eig
            *
            int(count)
        )

    return acc / total


def xz_features_from_counts(
    counts_x,
    counts_z,
):
    values = []

    for q in range(4):
        values.append(
            expectation_from_counts(
                counts_x,
                q,
            )
        )

    for q in range(4):
        values.append(
            expectation_from_counts(
                counts_z,
                q,
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


# =============================================================================
# FRESH PHYSICAL-QUBIT SELECTION
# =============================================================================

def injection_readout_metrics(
    qubit_df,
    backend_name,
    layout,
):
    qdf = (
        qubit_df[
            qubit_df["backend"]
            ==
            backend_name
        ]
        .set_index("qubit")
    )

    values = np.array(
        [
            float(
                qdf.loc[
                    int(layout[q]),
                    "readout_error",
                ]
            )
            for q in range(4)
        ],
        dtype=float,
    )

    return (
        float(values.max()),
        float(values.mean()),
    )


def choose_fresh_layout_for_xz_injection(
    fresh,
    backend_name,
):
    good = fresh["compiled"].copy()

    good = good[
        (
            good["backend"]
            ==
            backend_name
        )
        &
        (
            good["strict_compile_pass"]
            ==
            True
        )
    ].copy()

    if len(good) == 0:
        raise RuntimeError(
            f"No fresh strict zero-SWAP layouts on {backend_name}."
        )

    max_ro = []
    mean_ro = []

    for _, row in good.iterrows():
        layout = json.loads(
            row["layout"]
        )

        mx, av = injection_readout_metrics(
            fresh["qubits"],
            backend_name,
            layout,
        )

        max_ro.append(mx)
        mean_ro.append(av)

    good[
        "injection_readout_error_max"
    ] = max_ro

    good[
        "injection_readout_error_mean"
    ] = mean_ro

    good[
        "injection_readout_error_max_percent"
    ] = (
        100.0
        *
        good[
            "injection_readout_error_max"
        ]
    )

    # Hardware hierarchy only; no queue-based scientific score.
    good = good.sort_values(
        [
            "compiled_2q_error_max",
            "compiled_2q_error_mean",
            "injection_readout_error_max",
            "injection_readout_error_mean",
            "compiled_1q_error_max",
            "min_t2_us",
            "min_t1_us",
            "compiled_duration_us",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    good[
        "W1_hardware_rank"
    ] = np.arange(
        len(good)
    ) + 1

    return (
        good.iloc[0],
        good,
    )


# =============================================================================
# RESOURCE AUDIT AND ROUGH CHARGE PLANNING
# =============================================================================

def circuit_resource_row(
    circuit,
    backend,
    metadata,
):
    ops = {
        str(k): int(v)
        for k, v
        in circuit.count_ops().items()
    }

    duration_s = float(
        circuit.estimate_duration(
            backend.target,
            unit="s",
        )
    )

    return {
        **metadata,
        "depth":
            int(circuit.depth()),
        "size":
            int(circuit.size()),
        "n_cz":
            int(
                ops.get(
                    "cz",
                    0,
                )
            ),
        "n_swap":
            int(
                ops.get(
                    "swap",
                    0,
                )
            ),
        "n_measure":
            int(
                ops.get(
                    "measure",
                    0,
                )
            ),
        "duration_us":
            duration_s
            *
            1e6,
        "operations":
            json.dumps(
                ops,
                sort_keys=True,
            ),
    }


def smoke_charge_ratio():
    if (
        not PRIOR_SMOKE_PREFLIGHT.exists()
        or
        not PRIOR_SMOKE_USAGE.exists()
    ):
        return np.nan

    try:
        pre = json.loads(
            PRIOR_SMOKE_PREFLIGHT.read_text(
                encoding="utf-8"
            )
        )

        use = json.loads(
            PRIOR_SMOKE_USAGE.read_text(
                encoding="utf-8"
            )
        )

        scheduled = float(
            pre[
                "estimated_scheduled_circuit_time_all_shots_s"
            ]
        )

        charged = float(
            use[
                "job_usage_seconds"
            ]
        )

        if scheduled > 0 and charged > 0:
            return charged / scheduled

    except Exception:
        pass

    return np.nan


def estimate_from_actual_w7(
    n_w1_circuits,
    w1_cz_per_circuit,
    shots,
):
    if not PRIOR_W7_SUMMARY.exists():
        return np.nan

    try:
        summary = json.loads(
            PRIOR_W7_SUMMARY.read_text(
                encoding="utf-8"
            )
        )

        w7_charge = float(
            summary[
                "job_usage_seconds"
            ]
        )

        # Actual 11.0C pilot:
        # 16 days × 2 settings × 210 CZ × 256 shots.
        reference_cz_shots = (
            16
            *
            2
            *
            210
            *
            256
        )

        this_cz_shots = (
            int(n_w1_circuits)
            *
            int(w1_cz_per_circuit)
            *
            int(shots)
        )

        return (
            w7_charge
            *
            this_cz_shots
            /
            reference_cz_shots
        )

    except Exception:
        return np.nan


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Submit all 365 validation days to the real QPU. "
            "Default is dry-run."
        ),
    )

    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=[
            "ibm_fez",
            "ibm_kingston",
            "ibm_marrakesh",
        ],
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
    )

    parser.add_argument(
        "--shortlist",
        type=int,
        default=DEFAULT_SHORTLIST,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Override conservative allocation guard."
        ),
    )

    args = parser.parse_args()

    if args.shots < 1:
        raise ValueError(
            "--shots must be >= 1"
        )

    candidate = load_h4_winner_dynamics()

    (
        work_tv,
        cols,
        angles,
        y,
        train_idx,
        val_idx,
    ) = load_train_validation_only()

    print("=" * 120)
    print("WEEK 11.0D — FULL 2025 VALIDATION: REAL-QPU RWP_W1 + XZ_INJECTION")
    print("=" * 120)
    print()
    print("Chronological protocol:")
    print("  2022-2024 = establish EVERYTHING")
    print("  2025      = validation only")
    print("  2026      = frozen / not used")
    print()
    print(
        f"Transferred quantum dynamics: "
        f"{candidate['source_candidate_id']}"
    )
    print(
        f"H4, r={candidate['r']}, "
        f"hx={candidate['hx']:+.6f}, "
        f"hy={candidate['hy']:+.6f}"
    )
    print(
        "New deployment protocol: RWP_W1 + XZ_injection."
    )

    # =========================================================================
    # IDEAL W1 FEATURES
    # =========================================================================

    X_ideal = build_ideal_w1_feature_bank(
        candidate,
        angles,
    )

    X_train = X_ideal[train_idx]
    y_train = y[train_idx]

    X_val = X_ideal[val_idx]
    y_val = y[val_idx]

    # =========================================================================
    # ESTABLISH READOUT USING 2022-2024 ONLY
    # =========================================================================

    (
        selected_lambda,
        cv_folds,
        cv_summary,
    ) = select_lambda_training_only(
        X_train,
        y_train,
    )

    cv_folds.to_csv(
        RESULTS
        /
        "11_0D_ridge_cv_folds.csv",
        index=False,
    )

    cv_summary.to_csv(
        RESULTS
        /
        "11_0D_ridge_cv_summary.csv",
        index=False,
    )

    (
        ridge,
        scaler,
        keep,
    ) = fit_scaled_ridge(
        X_train,
        y_train,
        selected_lambda,
    )

    (
        frozen_weights,
        frozen_record,
    ) = export_frozen_readout(
        selected_lambda,
        ridge,
        scaler,
        keep,
    )

    pred_val_ideal = predict_scaled_ridge(
        ridge,
        scaler,
        keep,
        X_val,
    )

    ideal_val_rmse = rmse(
        y_val,
        pred_val_ideal,
    )

    ideal_val_mae = mae(
        y_val,
        pred_val_ideal,
    )

    ideal_val_bias = bias(
        y_val,
        pred_val_ideal,
    )

    print()
    print("=" * 120)
    print("FROZEN 2022-2024 READOUT")
    print("=" * 120)
    print(
        f"Selected lambda: {selected_lambda}"
    )
    print(
        f"CV RMSE: "
        f"{float(cv_summary.iloc[0]['cv_rmse_mean']):.6f}"
    )
    print(
        f"Frozen Ridge intercept: "
        f"{float(ridge.intercept_):+.6f}"
    )
    print()
    print(
        frozen_weights.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("IDEAL 2025 VALIDATION REFERENCE")
    print("=" * 120)
    print(
        f"2025 ideal W1 RMSE: "
        f"{ideal_val_rmse:.6f}"
    )
    print(
        f"2025 ideal W1 MAE:  "
        f"{ideal_val_mae:.6f}"
    )
    print(
        f"2025 ideal W1 bias: "
        f"{ideal_val_bias:+.6f}"
    )
    print(
        "No 2025 samples were used to fit lambda/scaler/weights/intercept."
    )

    # =========================================================================
    # FRESH HARDWARE SCAN
    # =========================================================================

    service = get_service()

    print()
    print("=" * 120)
    print(
        f"FRESH {args.backend} HARDWARE SCAN"
    )
    print("=" * 120)

    fresh = selector.fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=[
            args.backend
        ],
        shortlist=args.shortlist,
        write_prefix="11_0D_hardware",
        verbose=True,
    )

    (
        selected_layout_row,
        hardware_ranking,
    ) = choose_fresh_layout_for_xz_injection(
        fresh,
        args.backend,
    )

    hardware_ranking.to_csv(
        RESULTS
        /
        "11_0D_hardware_ranking_XZinj.csv",
        index=False,
    )

    layout = json.loads(
        selected_layout_row[
            "layout"
        ]
    )

    print()
    print("Fresh physical choice:")
    print(
        f"  layout={layout}"
    )
    print(
        f"  max CZ error="
        f"{selected_layout_row['compiled_2q_error_max_percent']:.4f}%"
    )
    print(
        f"  max injection readout error="
        f"{selected_layout_row['injection_readout_error_max_percent']:.4f}%"
    )
    print(
        f"  min T2="
        f"{selected_layout_row['min_t2_us']:.2f} us"
    )

    backend = service.backend(
        args.backend,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    properties = backend.properties(
        refresh=True
    )

    if properties is None:
        raise RuntimeError(
            f"{args.backend}: current properties unavailable."
        )

    # =========================================================================
    # BUILD ALL 365 × 2 VALIDATION CIRCUITS
    # =========================================================================

    logical = []
    metadata = []

    for endpoint in val_idx:
        for basis in [
            "X",
            "Z",
        ]:
            qc = build_w1_measurement_circuit(
                candidate,
                angles[int(endpoint)],
                basis,
                int(endpoint),
            )

            logical.append(qc)

            metadata.append({
                "endpoint_global":
                    int(endpoint),
                "validation_offset":
                    int(
                        endpoint
                        -
                        N_TRAIN
                    ),
                "basis":
                    basis,
            })

    print()
    print(
        f"Logical circuits: {len(logical)} "
        f"({N_VAL} validation days × 2 settings)"
    )

    # Transpile all at once so every circuit uses the same fresh fixed layout.
    isa = transpile(
        logical,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
        scheduling_method="alap",
    )

    if not isinstance(
        isa,
        list,
    ):
        isa = [isa]

    audit_rows = []

    for meta, circuit in zip(
        metadata,
        isa,
    ):
        backend.check_faulty(
            circuit
        )

        row = circuit_resource_row(
            circuit,
            backend,
            meta,
        )

        if row["n_swap"] != 0:
            raise RuntimeError(
                f"SWAP detected at validation offset "
                f"{meta['validation_offset']} "
                f"basis={meta['basis']}; refusing to continue."
            )

        audit_rows.append(row)

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df[
        "scheduled_time_all_shots_s"
    ] = (
        audit_df[
            "duration_us"
        ]
        *
        1e-6
        *
        args.shots
    )

    audit_df.to_csv(
        RESULTS
        /
        "11_0D_transpilation.csv",
        index=False,
    )

    scheduled_sum_s = float(
        audit_df[
            "scheduled_time_all_shots_s"
        ].sum()
    )

    cz_per_circuit = int(
        audit_df[
            "n_cz"
        ].median()
    )

    old_ratio = smoke_charge_ratio()

    rough_old_ratio_charge = (
        scheduled_sum_s
        *
        old_ratio
        if np.isfinite(old_ratio)
        else np.nan
    )

    rough_w7_scaled_charge = (
        estimate_from_actual_w7(
            n_w1_circuits=len(isa),
            w1_cz_per_circuit=cz_per_circuit,
            shots=args.shots,
        )
    )

    usage_before = safe_service_usage(
        service
    )

    remaining = np.nan

    if isinstance(
        usage_before,
        dict,
    ):
        try:
            remaining = float(
                usage_before[
                    "usage_remaining_seconds"
                ]
            )
        except Exception:
            pass

    print()
    print("=" * 120)
    print("FULL 2025 W1 QPU PREFLIGHT")
    print("=" * 120)
    print(
        f"Validation endpoints: {N_VAL}"
    )
    print(
        f"Circuits:             {len(isa)}"
    )
    print(
        f"Shots/setting:        {args.shots}"
    )
    print(
        f"Median CZ/circuit:    "
        f"{cz_per_circuit}"
    )
    print(
        f"Median depth:         "
        f"{float(audit_df['depth'].median()):.1f}"
    )
    print(
        f"Median duration:      "
        f"{float(audit_df['duration_us'].median()):.3f} us"
    )
    print(
        f"Scheduled-duration × shots sum: "
        f"{scheduled_sum_s:.6f} s"
    )

    if np.isfinite(
        rough_old_ratio_charge
    ):
        print(
            f"Old 11.0B-ratio rough charge estimate: "
            f"{rough_old_ratio_charge:.1f} s"
        )

    if np.isfinite(
        rough_w7_scaled_charge
    ):
        print(
            f"Scaling from ACTUAL 11.0C W7 pilot charge: "
            f"{rough_w7_scaled_charge:.1f} s"
        )

    print(
        "Both charge estimates are planning heuristics only."
    )

    print()
    print("Service usage before:")
    print(
        json.dumps(
            json_safe(
                usage_before
            ),
            indent=2,
        )
    )

    preflight = {
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "mode":
            (
                "SUBMIT"
                if args.submit
                else "DRY_RUN"
            ),
        "protocol":
            "RWP_W1",
        "readout":
            "XZ_injection",
        "train_period":
            "2022-2024",
        "validation_period":
            "2025",
        "test_2026":
            "FROZEN / UNUSED",
        "n_train":
            N_TRAIN,
        "n_validation":
            N_VAL,
        "selected_lambda_from_training_only":
            selected_lambda,
        "frozen_ridge_intercept":
            float(
                ridge.intercept_
            ),
        "ideal_2025_validation_rmse":
            ideal_val_rmse,
        "ideal_2025_validation_mae":
            ideal_val_mae,
        "ideal_2025_validation_bias":
            ideal_val_bias,
        "backend":
            args.backend,
        "layout":
            layout,
        "shots":
            args.shots,
        "n_circuits":
            len(isa),
        "median_cz_per_circuit":
            cz_per_circuit,
        "scheduled_duration_all_shots_s":
            scheduled_sum_s,
        "rough_old_ratio_charge_s":
            rough_old_ratio_charge,
        "rough_scaled_from_actual_W7_charge_s":
            rough_w7_scaled_charge,
        "service_usage_before":
            json_safe(
                usage_before
            ),
        "candidate":
            {
                **{
                    k: v
                    for k, v
                    in candidate.items()
                    if k != "J"
                },
                "J":
                    {
                        f"{i}{j}":
                            value
                        for (i, j), value
                        in candidate["J"].items()
                    },
            },
    }

    with open(
        RESULTS
        /
        "11_0D_preflight.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(
                preflight
            ),
            fp,
            indent=2,
        )

    if not args.submit:
        print()
        print(
            "DRY RUN COMPLETE — NO QPU JOB SUBMITTED."
        )
        print(
            "If acceptable, submit with:"
        )
        print(
            f"python {Path(__file__).name} "
            f"--submit "
            f"--backend {args.backend} "
            f"--shots {args.shots}"
        )
        return

    # =========================================================================
    # CONSERVATIVE BUDGET GUARD
    # =========================================================================

    usable_estimates = [
        value
        for value in [
            rough_old_ratio_charge,
            rough_w7_scaled_charge,
        ]
        if np.isfinite(value)
    ]

    guard_estimate = (
        max(usable_estimates)
        if usable_estimates
        else np.nan
    )

    if (
        np.isfinite(remaining)
        and
        np.isfinite(guard_estimate)
        and
        guard_estimate
        >
        0.65
        *
        remaining
        and
        not args.force
    ):
        raise RuntimeError(
            f"Conservative estimated charge {guard_estimate:.1f}s "
            f"exceeds 65% of remaining allocation {remaining:.1f}s. "
            f"Review first or explicitly add --force."
        )

    # =========================================================================
    # SUBMIT ONE SAMPLER JOB
    # =========================================================================

    sampler = Sampler(
        mode=backend
    )

    print()
    print(
        "Submitting all 365 W1 validation days "
        "as ONE SamplerV2 job..."
    )

    job = sampler.run(
        isa,
        shots=int(
            args.shots
        ),
    )

    print(
        f"Job ID: {job.job_id()}"
    )

    result = job.result()

    print(
        f"Final status: {job.status()}"
    )

    # =========================================================================
    # RAW COUNTS
    # =========================================================================

    endpoint_counts = {
        int(endpoint): {}
        for endpoint in val_idx
    }

    counts_rows = []

    for meta, pub_result in zip(
        metadata,
        result,
    ):
        counts = (
            pub_result
            .data
            .inj
            .get_counts()
        )

        endpoint = int(
            meta[
                "endpoint_global"
            ]
        )

        basis = str(
            meta[
                "basis"
            ]
        )

        endpoint_counts[
            endpoint
        ][
            basis
        ] = counts

        for bitstring, count in counts.items():
            counts_rows.append({
                "endpoint_global":
                    endpoint,
                "validation_offset":
                    int(
                        meta[
                            "validation_offset"
                        ]
                    ),
                "basis":
                    basis,
                "bitstring":
                    bitstring,
                "count":
                    int(count),
            })

    pd.DataFrame(
        counts_rows
    ).to_csv(
        RESULTS
        /
        "11_0D_counts.csv",
        index=False,
    )

    # =========================================================================
    # QPU FEATURES
    # =========================================================================

    X_qpu_val = np.vstack(
        [
            xz_features_from_counts(
                endpoint_counts[
                    int(endpoint)
                ]["X"],
                endpoint_counts[
                    int(endpoint)
                ]["Z"],
            )
            for endpoint in val_idx
        ]
    )

    # IMPORTANT:
    # same frozen 2022-2024 scaler + Ridge weights are used here.
    pred_qpu_val = predict_scaled_ridge(
        ridge,
        scaler,
        keep,
        X_qpu_val,
    )

    qpu_val_rmse = rmse(
        y_val,
        pred_qpu_val,
    )

    qpu_val_mae = mae(
        y_val,
        pred_qpu_val,
    )

    qpu_val_bias = bias(
        y_val,
        pred_qpu_val,
    )

    feature_mae = mae(
        X_val.ravel(),
        X_qpu_val.ravel(),
    )

    feature_rmse = rmse(
        X_val.ravel(),
        X_qpu_val.ravel(),
    )

    # =========================================================================
    # SAVE PER-DAY PREDICTIONS
    # =========================================================================

    prediction_rows = []

    for local_i, endpoint in enumerate(
        val_idx
    ):
        row = {
            "validation_offset":
                int(local_i),
            "endpoint_global":
                int(endpoint),
            "target":
                float(
                    y_val[
                        local_i
                    ]
                ),
            "pred_ideal_W1":
                float(
                    pred_val_ideal[
                        local_i
                    ]
                ),
            "pred_QPU_W1":
                float(
                    pred_qpu_val[
                        local_i
                    ]
                ),
            "abs_error_ideal":
                float(
                    abs(
                        pred_val_ideal[
                            local_i
                        ]
                        -
                        y_val[
                            local_i
                        ]
                    )
                ),
            "abs_error_QPU":
                float(
                    abs(
                        pred_qpu_val[
                            local_i
                        ]
                        -
                        y_val[
                            local_i
                        ]
                    )
                ),
            "feature_MAE_QPU_vs_ideal":
                mae(
                    X_val[
                        local_i
                    ],
                    X_qpu_val[
                        local_i
                    ],
                ),
        }

        # Preserve date if available without assuming a column name.
        for date_col in [
            "target_date",
            "date",
            "Date",
        ]:
            if date_col in work_tv.columns:
                row[
                    "target_date"
                ] = str(
                    pd.to_datetime(
                        work_tv.iloc[
                            int(endpoint)
                        ][
                            date_col
                        ]
                    ).date()
                )
                break

        prediction_rows.append(row)

    predictions = pd.DataFrame(
        prediction_rows
    )

    predictions.to_csv(
        RESULTS
        /
        "11_0D_predictions.csv",
        index=False,
    )

    # =========================================================================
    # SAVE FEATURE COMPARISON
    # =========================================================================

    feature_rows = []

    for local_i, endpoint in enumerate(
        val_idx
    ):
        for feature_idx, feature in enumerate(
            XZ_COLUMNS
        ):
            ideal_value = float(
                X_val[
                    local_i,
                    feature_idx,
                ]
            )

            hardware_value = float(
                X_qpu_val[
                    local_i,
                    feature_idx,
                ]
            )

            feature_rows.append({
                "validation_offset":
                    int(local_i),
                "feature":
                    feature,
                "ideal":
                    ideal_value,
                "hardware":
                    hardware_value,
                "error_hardware_minus_ideal":
                    hardware_value
                    -
                    ideal_value,
                "absolute_error":
                    abs(
                        hardware_value
                        -
                        ideal_value
                    ),
            })

    pd.DataFrame(
        feature_rows
    ).to_csv(
        RESULTS
        /
        "11_0D_features.csv",
        index=False,
    )

    # =========================================================================
    # ACTUAL QPU USAGE
    # =========================================================================

    try:
        job_usage = float(
            job.usage(
                partial=False
            )
        )
    except Exception as exc:
        job_usage = np.nan
        usage_error = str(exc)
    else:
        usage_error = ""

    try:
        metrics = job.metrics()
    except Exception as exc:
        metrics = {
            "available":
                False,
            "error":
                str(exc),
        }

    usage_after = safe_service_usage(
        service
    )

    usage_record = {
        "job_id":
            job.job_id(),
        "job_usage_seconds":
            job_usage,
        "job_usage_error":
            usage_error,
        "job_metrics":
            json_safe(metrics),
        "service_usage_before":
            json_safe(
                usage_before
            ),
        "service_usage_after":
            json_safe(
                usage_after
            ),
    }

    with open(
        RESULTS
        /
        "11_0D_usage.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            usage_record,
            fp,
            indent=2,
        )

    summary = {
        **preflight,
        "job_id":
            job.job_id(),
        "job_usage_seconds":
            job_usage,
        "service_usage_after":
            json_safe(
                usage_after
            ),
        "QPU_2025_validation_RMSE":
            qpu_val_rmse,
        "QPU_2025_validation_MAE":
            qpu_val_mae,
        "QPU_2025_validation_bias":
            qpu_val_bias,
        "QPU_feature_MAE_vs_ideal":
            feature_mae,
        "QPU_feature_RMSE_vs_ideal":
            feature_rmse,
    }

    with open(
        RESULTS
        /
        "11_0D_summary.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(summary),
            fp,
            indent=2,
        )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print()
    print("=" * 120)
    print("FULL 2025 REAL-QPU W1 VALIDATION RESULTS")
    print("=" * 120)
    print(
        "Readout establishment: 2022-2024 ONLY"
    )
    print(
        "2026: FROZEN / UNUSED"
    )
    print()
    print(
        f"Ideal 2025 W1 RMSE:     "
        f"{ideal_val_rmse:.6f}"
    )
    print(
        f"REAL-QPU 2025 W1 RMSE:  "
        f"{qpu_val_rmse:.6f}"
    )
    print(
        f"REAL-QPU 2025 W1 MAE:   "
        f"{qpu_val_mae:.6f}"
    )
    print(
        f"REAL-QPU 2025 W1 bias:  "
        f"{qpu_val_bias:+.6f}"
    )
    print(
        f"QPU feature MAE:         "
        f"{feature_mae:.6f}"
    )
    print(
        f"QPU feature RMSE:        "
        f"{feature_rmse:.6f}"
    )
    print(
        f"Actual QPU charge:       "
        f"{job_usage} s"
    )

    print()
    print(
        predictions.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    for name in [
        "11_0D_ridge_cv_folds.csv",
        "11_0D_ridge_cv_summary.csv",
        "11_0D_frozen_readout_weights.csv",
        "11_0D_frozen_readout.json",
        "11_0D_hardware_ranking_XZinj.csv",
        "11_0D_transpilation.csv",
        "11_0D_preflight.json",
        "11_0D_counts.csv",
        "11_0D_features.csv",
        "11_0D_predictions.csv",
        "11_0D_usage.json",
        "11_0D_summary.json",
    ]:
        print(
            f"  results/{name}"
        )


if __name__ == "__main__":
    main()
