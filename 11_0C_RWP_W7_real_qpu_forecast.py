
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

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)
from qiskit_ibm_runtime import SamplerV2 as Sampler

from ibm_account import get_service


# =============================================================================
# WEEK 11.0C — PRELIMINARY REAL-QPU RWP_W7 FORECAST
#
# Protocol
# --------
# For every forecast endpoint t:
#
#       start all 6 qubits from |0>
#              |
#       replay inputs t-6,...,t  (7 consecutive days)
#              |
#       q0..q3 reset + reinjected between replay steps
#       q4,q5 are NEVER reset inside the 7-day replay
#              |
#       after day 7 measure ONLY injection qubits q0..q3
#              |
#       X setting -> <X0>,...,<X3>
#       Z setting -> <Z0>,...,<Z3>
#
# Then the next forecast endpoint starts again from |000000>.
#
# This is RWP_W7: a 7-day rewind/replay protocol.
#
# IMPORTANT
# ---------
# Measurement occurs only at the END of each independent 7-day replay.
# Therefore the final measurement cannot disturb a future reservoir state,
# because that trajectory ends immediately after measurement.
#
# We deliberately do NOT use the old XYZ_all Ridge readout unchanged.
# Changing both:
#     CONT -> RWP_W7
#     XYZ_all -> XZ_injection
# changes the feature map.
#
# Therefore the script:
#   1. keeps the current best H4 quantum dynamics fixed,
#   2. generates IDEAL RWP_W7 XZ_injection training features,
#   3. reselects Ridge lambda by chronological training-only CV,
#   4. trains the classical readout on ideal training features,
#   5. evaluates the full ideal 2025 RWP_W7 validation RMSE,
#   6. runs a small consecutive validation subset on a real IBM QPU,
#   7. reports QPU RMSE/MAE/bias and feature degradation.
#
# Hardware
# --------
# Every invocation performs a FRESH hardware scan on the requested backend.
# No historical physical-qubit layout is used.
#
# Default = DRY RUN.
#
# Real execution:
#
#   python 11_0C_RWP_W7_real_qpu_forecast.py --submit
#
# Default preliminary workload:
#   backend   : ibm_fez
#   eval days : 16
#   shots     : 256 per setting
#   settings  : 2 per endpoint (X and Z)
# =============================================================================


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

SELECTOR_SCRIPT = (
    HERE
    /
    "11_0A_live_embedding_reselection.py"
)

QRC_SCRIPT = (
    HERE
    /
    "07_02b_memory_pair_isolation.py"
)

FAIR_FILE = (
    RESULTS
    /
    "09_03c_forecast_best_135.csv"
)

PRIOR_PREFLIGHT = (
    RESULTS
    /
    "11_0B_preflight.json"
)

PRIOR_USAGE = (
    RESULTS
    /
    "11_0B_usage.json"
)

PREFIX = (
    RESULTS
    /
    "11_0C"
)

ALPHA = 0.75
DT = 1.6

WINDOW = 7

DEFAULT_BACKEND = "ibm_fez"
DEFAULT_EVAL_DAYS = 16
DEFAULT_SHOTS = 256
DEFAULT_SHORTLIST = 80

OPTIMIZATION_LEVEL = 1
SEED_TRANSPILER = 42

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

VAR_TOL = 1e-12

TOPOLOGY_EDGES = {
    "H0": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ],
    "H1": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (3, 5),
    ],
    "H2": [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4),
        (4, 5),
    ],
    "H3": [
        (2, 3),
        (1, 2),
        (0, 1),
        (0, 4),
        (4, 5),
    ],
    "H4": [
        (1, 2),
        (0, 1),
        (0, 4),
        (3, 4),
        (4, 5),
    ],
}

EDGE_TO_COLUMN = {
    "H0": {
        (0, 1):
            "J01",
        (1, 2):
            "J12",
        (2, 3):
            "J23",
        (3, 4):
            "J34",
        (4, 5):
            "J45",
    },
    "H1": {
        (0, 1):
            "J01",
        (1, 2):
            "J12",
        (2, 3):
            "J23",
        (3, 4):
            "J34",
        (3, 5):
            "J35",
    },
    "H2": {
        (0, 1):
            "J01",
        (1, 2):
            "J12",
        (2, 3):
            "J23",
        (2, 4):
            "J24",
        (4, 5):
            "J45",
    },
    "H3": {
        (2, 3):
            "J23",
        (1, 2):
            "J12",
        (0, 1):
            "J01",
        (0, 4):
            "J04",
        (4, 5):
            "J45",
    },
    "H4": {
        (1, 2):
            "J12",
        (0, 1):
            "J01",
        (0, 4):
            "J04",
        (3, 4):
            "J34",
        (4, 5):
            "J45",
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
# IMPORT PROJECT MODULES
# =============================================================================

def import_module_from_path(
    name,
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. "
            f"Place this script in the project directory."
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    if spec.loader is None:
        raise RuntimeError(
            f"Could not import {path}"
        )

    spec.loader.exec_module(
        module
    )

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

def json_safe(
    obj,
):
    if obj is None:
        return None

    if isinstance(
        obj,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return obj

    if isinstance(
        obj,
        np.generic,
    ):
        return obj.item()

    if isinstance(
        obj,
        dict,
    ):
        return {
            str(k):
                json_safe(v)
            for k, v
            in obj.items()
        }

    if isinstance(
        obj,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(v)
            for v in obj
        ]

    if hasattr(
        obj,
        "isoformat",
    ):
        try:
            return obj.isoformat()
        except Exception:
            pass

    return str(
        obj
    )


def metric_rmse(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    y_true
                    -
                    y_pred
                )
                **
                2
            )
        )
    )


def metric_mae(
    y_true,
    y_pred,
):
    return float(
        np.mean(
            np.abs(
                np.asarray(
                    y_true,
                    dtype=float,
                )
                -
                np.asarray(
                    y_pred,
                    dtype=float,
                )
            )
        )
    )


def metric_bias(
    y_true,
    y_pred,
):
    return float(
        np.mean(
            np.asarray(
                y_pred,
                dtype=float,
            )
            -
            np.asarray(
                y_true,
                dtype=float,
            )
        )
    )


def safe_service_usage(
    service,
):
    try:
        return service.usage()

    except Exception as exc:
        return {
            "available":
                False,
            "error":
                str(
                    exc
                ),
        }


# =============================================================================
# KEEP THE CURRENT H4 QUANTUM DYNAMICS FIXED
# =============================================================================

def load_h4_winner_dynamics():
    if not FAIR_FILE.exists():
        raise FileNotFoundError(
            FAIR_FILE
        )

    df = pd.read_csv(
        FAIR_FILE
    )

    # Absolute forecast winner from the fair 9.3c experiment.
    row = (
        df.sort_values(
            [
                "cv_rmse",
                "validation_rmse",
            ]
        )
        .iloc[
            0
        ]
    )

    topology = str(
        row[
            "topology"
        ]
    )

    if topology != "H4":
        raise RuntimeError(
            f"The current absolute 9.3c winner is {topology}, not H4. "
            f"Review before running this script."
        )

    J = {
        edge:
            float(
                row[
                    column
                ]
            )
        for edge, column
        in EDGE_TO_COLUMN[
            topology
        ].items()
    }

    return {
        "source_candidate_id":
            str(
                row[
                    "seed_candidate_id"
                ]
            ),
        "source_config_id":
            str(
                row[
                    "config_id"
                ]
            ),
        "topology":
            topology,
        "r":
            int(
                row[
                    "trotter_r"
                ]
            ),
        "hx":
            float(
                row[
                    "hx"
                ]
            ),
        "hy":
            float(
                row[
                    "hy"
                ]
            ),
        "J":
            J,
        "source_cv_rmse_XYZ_CONT":
            float(
                row[
                    "cv_rmse"
                ]
            ),
        "source_validation_rmse_XYZ_CONT":
            float(
                row[
                    "validation_rmse"
                ]
            ),
        # New protocol/readout label for this experiment.
        "candidate_id":
            (
                str(
                    row[
                        "seed_candidate_id"
                    ]
                )
                +
                "__RWP_W7_XZinj"
            ),
        "readout":
            "XZ_injection",
    }


# =============================================================================
# DATA
# =============================================================================

def load_project_data():
    work, train, val, cols = (
        qrc.load_data()
    )

    if (
        len(
            train
        )
        !=
        1095
        or
        len(
            val
        )
        !=
        365
    ):
        raise RuntimeError(
            f"Expected train/val = 1095/365, "
            f"got {len(train)}/{len(val)}"
        )

    work_eval = pd.concat(
        [
            train,
            val,
        ],
        axis=0,
    ).reset_index(
        drop=True
    )

    angles = np.asarray(
        qrc.make_input_angles(
            work_eval,
            cols,
        ),
        dtype=float,
    )

    y = work_eval[
        cols[
            "target"
        ]
    ].to_numpy(
        dtype=float
    )

    return (
        work_eval,
        train,
        val,
        cols,
        angles,
        y,
    )


# =============================================================================
# IDEAL RWP_W7 QUANTUM CHANNEL
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
        "Could not find injection-qubit Pauli operators "
        "in 07_02b_memory_pair_isolation.py."
    )

ZZ_OPS = {}

for topology, edges in TOPOLOGY_EDGES.items():
    for edge in edges:
        if edge not in ZZ_OPS:
            ZZ_OPS[
                edge
            ] = qrc.pauli_product_full(
                qrc.Z2,
                edge[
                    0
                ],
                qrc.Z2,
                edge[
                    1
                ],
            )


def pauli_exp(
    operator,
    theta,
):
    return (
        np.cos(
            theta
        )
        *
        I64
        -
        1j
        *
        np.sin(
            theta
        )
        *
        operator
    )


def build_trotter_unitary(
    candidate,
):
    topology = candidate[
        "topology"
    ]

    r = int(
        candidate[
            "r"
        ]
    )

    hx = float(
        candidate[
            "hx"
        ]
    )

    hy = float(
        candidate[
            "hy"
        ]
    )

    J = candidate[
        "J"
    ]

    U = I64.copy()

    for _ in range(
        r
    ):
        # ZZ layer.
        for edge in TOPOLOGY_EDGES[
            topology
        ]:
            theta = (
                float(
                    J[
                        edge
                    ]
                )
                *
                DT
                /
                r
            )

            U = (
                pauli_exp(
                    ZZ_OPS[
                        edge
                    ],
                    theta,
                )
                @
                U
            )

        # X layer.
        theta_x = (
            hx
            *
            DT
            /
            r
        )

        for q in range(
            qrc.N_QUBITS
        ):
            U = (
                pauli_exp(
                    qrc.FULL_SINGLE_OPS[
                        f"X{q}"
                    ],
                    theta_x,
                )
                @
                U
            )

        # memory-only Y layer.
        if not np.isclose(
            hy,
            0.0,
        ):
            theta_y = (
                hy
                *
                DT
                /
                r
            )

            for q in [
                4,
                5,
            ]:
                U = (
                    pauli_exp(
                        qrc.FULL_SINGLE_OPS[
                            f"Y{q}"
                        ],
                        theta_y,
                    )
                    @
                    U
                )

    return U


def expectation(
    rho,
    op,
):
    return float(
        np.real_if_close(
            np.trace(
                rho
                @
                op
            ),
            tol=1000,
        ).real
    )


def ideal_rwp_w7_feature(
    A_list,
    endpoint,
):
    start = (
        int(
            endpoint
        )
        -
        WINDOW
        +
        1
    )

    if start < 0:
        raise ValueError(
            "RWP_W7 endpoint does not have seven historical rows."
        )

    rho_m = qrc.memory_zero_density()
    rho_i = None

    for idx in range(
        start,
        int(
            endpoint
        )
        +
        1,
    ):
        rho_i, rho_m = (
            qrc.final_reduced_states(
                A_list[
                    idx
                ],
                rho_m,
            )
        )

    row = {}

    for q in range(
        4
    ):
        row[
            f"X{q}"
        ] = expectation(
            rho_i,
            INJ_OPS[
                f"X{q}"
            ],
        )

        row[
            f"Z{q}"
        ] = expectation(
            rho_i,
            INJ_OPS[
                f"Z{q}"
            ],
        )

    return np.array(
        [
            row[
                c
            ]
            for c in XZ_COLUMNS
        ],
        dtype=float,
    )


def build_ideal_feature_bank(
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

    endpoints = np.arange(
        WINDOW
        -
        1,
        len(
            angles
        ),
        dtype=int,
    )

    X = np.vstack(
        [
            ideal_rwp_w7_feature(
                A_list,
                endpoint=int(
                    endpoint
                ),
            )
            for endpoint in endpoints
        ]
    )

    return (
        endpoints,
        X,
    )


# =============================================================================
# RIDGE: TRAINING-ONLY CHRONOLOGICAL CV
# =============================================================================

def fit_scaled_ridge(
    X,
    y,
    ridge_alpha,
):
    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    std = np.std(
        X,
        axis=0,
        ddof=0,
    )

    keep = (
        std
        >
        VAR_TOL
    )

    if not np.any(
        keep
    ):
        raise RuntimeError(
            "All RWP_W7 XZ-injection features are constant."
        )

    scaler = StandardScaler()

    Xz = scaler.fit_transform(
        X[
            :,
            keep
        ]
    )

    model = Ridge(
        alpha=float(
            ridge_alpha
        ),
        fit_intercept=True,
    )

    model.fit(
        Xz,
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

    return model.predict(
        scaler.transform(
            X[
                :,
                keep
            ]
        )
    )


def select_lambda_cv(
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
            train_idx,
            valid_idx,
        ) in enumerate(
            splitter.split(
                X_train
            )
        ):
            (
                model,
                scaler,
                keep,
            ) = fit_scaled_ridge(
                X_train[
                    train_idx
                ],
                y_train[
                    train_idx
                ],
                ridge_alpha,
            )

            pred = predict_scaled_ridge(
                model,
                scaler,
                keep,
                X_train[
                    valid_idx
                ],
            )

            score = metric_rmse(
                y_train[
                    valid_idx
                ],
                pred,
            )

            fold_scores.append(
                score
            )

            rows.append({
                "lambda":
                    ridge_alpha,
                "fold":
                    fold
                    +
                    1,
                "rmse":
                    score,
            })

    cv_df = pd.DataFrame(
        rows
    )

    summary = (
        cv_df.groupby(
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
            [
                "cv_rmse_mean",
                "lambda",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    best_lambda = float(
        summary.iloc[
            0
        ][
            "lambda"
        ]
    )

    return (
        best_lambda,
        cv_df,
        summary,
    )


# =============================================================================
# HARDWARE RWP_W7 CIRCUITS
# =============================================================================

def append_qrc_step(
    qc,
    candidate,
    angles_row,
    first_step,
):
    # Reset only injection qubits between rewind steps.
    # Memory q4,q5 continues through the whole 7-day replay.
    if not first_step:
        for q in range(
            4
        ):
            qc.reset(
                q
            )

    for q in range(
        4
    ):
        qc.ry(
            float(
                angles_row[
                    q
                ]
            ),
            q,
        )

    topology = candidate[
        "topology"
    ]

    r = int(
        candidate[
            "r"
        ]
    )

    hx = float(
        candidate[
            "hx"
        ]
    )

    hy = float(
        candidate[
            "hy"
        ]
    )

    J = candidate[
        "J"
    ]

    for _ in range(
        r
    ):
        for i, j in TOPOLOGY_EDGES[
            topology
        ]:
            qc.rzz(
                2.0
                *
                float(
                    J[
                        (
                            i,
                            j,
                        )
                    ]
                )
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

        for q in range(
            6
        ):
            qc.rx(
                theta_x,
                q,
            )

        if not np.isclose(
            hy,
            0.0,
        ):
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


def build_rwp_w7_measurement_circuit(
    candidate,
    all_angles,
    endpoint,
    basis,
):
    qreg = QuantumRegister(
        6,
        "q",
    )

    # ONLY four injection-qubit classical outcomes.
    creg = ClassicalRegister(
        4,
        "inj",
    )

    qc = QuantumCircuit(
        qreg,
        creg,
        name=(
            f"RWP7_t{endpoint}_{basis}"
        ),
    )

    start = (
        int(
            endpoint
        )
        -
        WINDOW
        +
        1
    )

    if start < 0:
        raise ValueError(
            "Insufficient 7-day context."
        )

    for local_step, idx in enumerate(
        range(
            start,
            int(
                endpoint
            )
            +
            1,
        )
    ):
        append_qrc_step(
            qc,
            candidate,
            all_angles[
                idx
            ],
            first_step=(
                local_step
                ==
                0
            ),
        )

    # Final measurement only.
    if basis == "X":
        for q in range(
            4
        ):
            qc.h(
                q
            )

    elif basis == "Z":
        pass

    else:
        raise ValueError(
            basis
        )

    # q4,q5 are deliberately NOT measured.
    for q in range(
        4
    ):
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

    # Four-bit classical string = c3 c2 c1 c0.
    return int(
        clean[
            -(
                int(
                    q
                )
                +
                1
            )
        ]
    )


def one_qubit_expectation(
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
            int(
                count
            )
        )

    return (
        acc
        /
        total
    )


def xz_from_counts(
    counts_x,
    counts_z,
):
    values = {}

    for q in range(
        4
    ):
        values[
            f"X{q}"
        ] = one_qubit_expectation(
            counts_x,
            q,
        )

        values[
            f"Z{q}"
        ] = one_qubit_expectation(
            counts_z,
            q,
        )

    return np.array(
        [
            values[
                c
            ]
            for c in XZ_COLUMNS
        ],
        dtype=float,
    )


# =============================================================================
# FRESH HARDWARE SELECTION
# =============================================================================

def injection_readout_error(
    qubit_df,
    backend_name,
    layout,
):
    qdf = qubit_df[
        qubit_df[
            "backend"
        ]
        ==
        backend_name
    ].set_index(
        "qubit"
    )

    values = np.array(
        [
            float(
                qdf.loc[
                    int(
                        layout[
                            q
                        ]
                    ),
                    "readout_error",
                ]
            )
            for q in range(
                4
            )
        ],
        dtype=float,
    )

    return (
        float(
            np.max(
                values
            )
        ),
        float(
            np.mean(
                values
            )
        ),
    )


def choose_fresh_layout_for_injection_measurement(
    fresh,
    backend_name,
):
    good = fresh[
        "compiled"
    ].copy()

    good = good[
        (
            good[
                "backend"
            ]
            ==
            backend_name
        )
        &
        (
            good[
                "strict_compile_pass"
            ]
            ==
            True
        )
    ].copy()

    if len(
        good
    ) == 0:
        raise RuntimeError(
            f"No fresh zero-SWAP layouts on {backend_name}."
        )

    inj_max = []
    inj_mean = []

    for _, row in good.iterrows():
        layout = json.loads(
            row[
                "layout"
            ]
        )

        max_ro, mean_ro = (
            injection_readout_error(
                fresh[
                    "qubits"
                ],
                backend_name,
                layout,
            )
        )

        inj_max.append(
            max_ro
        )

        inj_mean.append(
            mean_ro
        )

    good[
        "injection_readout_error_max"
    ] = inj_max

    good[
        "injection_readout_error_mean"
    ] = inj_mean

    good[
        "injection_readout_error_max_percent"
    ] = (
        100.0
        *
        good[
            "injection_readout_error_max"
        ]
    )

    # Scientific hardware hierarchy:
    # no queue term, because backend already fixed.
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
    ).reset_index(
        drop=True
    )

    good[
        "RWP7_hardware_rank"
    ] = np.arange(
        len(
            good
        )
    ) + 1

    return (
        good.iloc[
            0
        ],
        good,
    )


# =============================================================================
# RESOURCE / USAGE PLANNING
# =============================================================================

def resource_row(
    circuit,
    backend,
    metadata,
):
    counts = {
        str(k):
            int(
                v
            )
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
            int(
                circuit.depth()
            ),
        "size":
            int(
                circuit.size()
            ),
        "n_cz":
            int(
                counts.get(
                    "cz",
                    0,
                )
            ),
        "n_swap":
            int(
                counts.get(
                    "swap",
                    0,
                )
            ),
        "n_reset":
            int(
                counts.get(
                    "reset",
                    0,
                )
            ),
        "n_measure":
            int(
                counts.get(
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
                counts,
                sort_keys=True,
            ),
    }


def prior_empirical_charge_ratio():
    if (
        not PRIOR_PREFLIGHT.exists()
        or
        not PRIOR_USAGE.exists()
    ):
        return np.nan

    try:
        pre = json.loads(
            PRIOR_PREFLIGHT.read_text(
                encoding="utf-8"
            )
        )

        use = json.loads(
            PRIOR_USAGE.read_text(
                encoding="utf-8"
            )
        )

        scheduled = float(
            pre[
                "estimated_scheduled_circuit_time_all_shots_s"
            ]
        )

        charge = float(
            use[
                "job_usage_seconds"
            ]
        )

        if (
            scheduled
            >
            0
            and
            charge
            >
            0
        ):
            return (
                charge
                /
                scheduled
            )

    except Exception:
        pass

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
            "Actually submit the RWP_W7 validation subset. "
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
        "--eval-days",
        type=int,
        default=DEFAULT_EVAL_DAYS,
    )

    parser.add_argument(
        "--val-offset",
        type=int,
        default=0,
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
            "Override budget guard if the rough empirical extrapolation "
            "looks large."
        ),
    )

    args = parser.parse_args()

    if args.eval_days < 2:
        raise ValueError(
            "Use at least two validation days."
        )

    if args.shots < 1:
        raise ValueError(
            "shots must be >= 1."
        )

    # =========================================================================
    # FIX H4 QUANTUM DYNAMICS
    # =========================================================================

    candidate = load_h4_winner_dynamics()

    (
        work_eval,
        train,
        val,
        cols,
        angles,
        y_all,
    ) = load_project_data()

    n_train = len(
        train
    )

    n_val = len(
        val
    )

    if (
        args.val_offset
        <
        0
        or
        args.val_offset
        +
        args.eval_days
        >
        n_val
    ):
        raise ValueError(
            "Validation subset outside available 2025 data."
        )

    print("=" * 120)
    print("WEEK 11.0C — REAL-QPU RWP_W7 FORECAST")
    print("=" * 120)
    print(
        "Protocol: 7-day rewind/replay; memory q4,q5 carried inside each "
        "7-day replay; final measurement only."
    )
    print(
        "Readout: injection qubits q0..q3 only, X and Z."
    )
    print()
    print(
        f"Source H4 dynamics: {candidate['source_candidate_id']}"
    )
    print(
        f"r={candidate['r']}, "
        f"hx={candidate['hx']:+.6f}, "
        f"hy={candidate['hy']:+.6f}"
    )
    print(
        f"Source CONT/XYZ CV RMSE: "
        f"{candidate['source_cv_rmse_XYZ_CONT']:.6f}"
    )
    print(
        "Ridge will now be RETRAINED for RWP_W7 + XZ_injection."
    )

    # =========================================================================
    # IDEAL RWP_W7 FEATURE BANK
    # =========================================================================

    endpoints, X_bank = (
        build_ideal_feature_bank(
            candidate,
            angles,
        )
    )

    y_bank = y_all[
        endpoints
    ]

    # Training endpoints: t <= last training row.
    train_mask = (
        endpoints
        <
        n_train
    )

    val_mask = (
        endpoints
        >=
        n_train
    )

    X_train = X_bank[
        train_mask
    ]

    y_train = y_bank[
        train_mask
    ]

    X_val = X_bank[
        val_mask
    ]

    y_val = y_bank[
        val_mask
    ]

    val_endpoints = endpoints[
        val_mask
    ]

    if len(
        y_val
    ) != 365:
        raise RuntimeError(
            f"Expected 365 RWP_W7 validation rows; got {len(y_val)}."
        )

    # =========================================================================
    # TRAINING-ONLY RIDGE CALIBRATION
    # =========================================================================

    (
        selected_lambda,
        cv_folds,
        cv_summary,
    ) = select_lambda_cv(
        X_train,
        y_train,
    )

    cv_folds.to_csv(
        RESULTS
        /
        "11_0C_ridge_cv_folds.csv",
        index=False,
    )

    cv_summary.to_csv(
        RESULTS
        /
        "11_0C_ridge_cv_summary.csv",
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

    pred_val_ideal = (
        predict_scaled_ridge(
            ridge,
            scaler,
            keep,
            X_val,
        )
    )

    ideal_val_rmse = metric_rmse(
        y_val,
        pred_val_ideal,
    )

    ideal_val_mae = metric_mae(
        y_val,
        pred_val_ideal,
    )

    print()
    print("=" * 120)
    print("IDEAL RWP_W7 + XZ_INJECTION REFERENCE")
    print("=" * 120)
    print(
        f"Training endpoints: {len(y_train)}"
    )
    print(
        f"Selected lambda:    {selected_lambda}"
    )
    print(
        f"CV RMSE:            "
        f"{float(cv_summary.iloc[0]['cv_rmse_mean']):.6f}"
    )
    print(
        f"2025 validation RMSE: {ideal_val_rmse:.6f}"
    )
    print(
        f"2025 validation MAE:  {ideal_val_mae:.6f}"
    )

    # =========================================================================
    # SMALL CONSECUTIVE VALIDATION SUBSET
    # =========================================================================

    subset_positions = np.arange(
        args.val_offset,
        args.val_offset
        +
        args.eval_days,
        dtype=int,
    )

    subset_endpoints = val_endpoints[
        subset_positions
    ]

    X_subset_ideal = X_val[
        subset_positions
    ]

    y_subset = y_val[
        subset_positions
    ]

    pred_subset_ideal = pred_val_ideal[
        subset_positions
    ]

    ideal_subset_rmse = metric_rmse(
        y_subset,
        pred_subset_ideal,
    )

    print()
    print(
        f"Pre-QPU ideal subset: offset={args.val_offset}, "
        f"N={args.eval_days}, "
        f"RMSE={ideal_subset_rmse:.6f}"
    )

    # =========================================================================
    # FRESH HARDWARE SCAN — ONE BACKEND
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
        write_prefix="11_0C_hardware",
        verbose=True,
    )

    (
        selected_layout_row,
        hardware_ranking,
    ) = (
        choose_fresh_layout_for_injection_measurement(
            fresh,
            args.backend,
        )
    )

    hardware_ranking.to_csv(
        RESULTS
        /
        "11_0C_hardware_ranking_XZinj.csv",
        index=False,
    )

    layout = json.loads(
        selected_layout_row[
            "layout"
        ]
    )

    print()
    print(
        "Fresh RWP_W7/XZ-injection physical choice:"
    )
    print(
        f"layout={layout}"
    )
    print(
        f"max CZ err="
        f"{selected_layout_row['compiled_2q_error_max_percent']:.4f}%"
    )
    print(
        f"max injection readout err="
        f"{selected_layout_row['injection_readout_error_max_percent']:.4f}%"
    )
    print(
        f"min T2="
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

    if (
        "reset"
        not in
        backend.target.operation_names
    ):
        raise RuntimeError(
            f"{args.backend} does not advertise reset."
        )

    # =========================================================================
    # BUILD TWO CIRCUITS PER ENDPOINT: X AND Z
    # =========================================================================

    logical_circuits = []
    metadata = []

    for endpoint in subset_endpoints:
        for basis in [
            "X",
            "Z",
        ]:
            qc = build_rwp_w7_measurement_circuit(
                candidate,
                angles,
                int(
                    endpoint
                ),
                basis,
            )

            logical_circuits.append(
                qc
            )

            metadata.append({
                "endpoint_global":
                    int(
                        endpoint
                    ),
                "validation_offset":
                    int(
                        endpoint
                        -
                        n_train
                    ),
                "basis":
                    basis,
            })

    isa = transpile(
        logical_circuits,
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
        isa = [
            isa
        ]

    audit_rows = []

    for meta, circuit in zip(
        metadata,
        isa,
    ):
        backend.check_faulty(
            circuit
        )

        row = resource_row(
            circuit,
            backend,
            meta,
        )

        if row[
            "n_swap"
        ] != 0:
            raise RuntimeError(
                f"SWAP detected at endpoint "
                f"{meta['endpoint_global']} basis={meta['basis']}; "
                f"refusing to submit."
            )

        audit_rows.append(
            row
        )

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
        "11_0C_transpilation.csv",
        index=False,
    )

    scheduled_sum_s = float(
        audit_df[
            "scheduled_time_all_shots_s"
        ].sum()
    )

    empirical_ratio = (
        prior_empirical_charge_ratio()
    )

    rough_charge = (
        scheduled_sum_s
        *
        empirical_ratio
        if np.isfinite(
            empirical_ratio
        )
        else np.nan
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
    print("QPU PREFLIGHT")
    print("=" * 120)
    print(
        f"Endpoints:          {args.eval_days}"
    )
    print(
        f"Circuits:           {len(isa)} "
        f"({args.eval_days} × 2 settings)"
    )
    print(
        f"Shots/setting:      {args.shots}"
    )
    print(
        f"Median CZ/circuit:  "
        f"{int(audit_df['n_cz'].median())}"
    )
    print(
        f"Median reset/circuit: "
        f"{int(audit_df['n_reset'].median())}"
    )
    print(
        f"Median depth:       "
        f"{float(audit_df['depth'].median()):.1f}"
    )
    print(
        f"Median duration:    "
        f"{float(audit_df['duration_us'].median()):.3f} us"
    )
    print(
        f"Scheduled-duration × shots sum: "
        f"{scheduled_sum_s:.6f} s"
    )

    if np.isfinite(
        empirical_ratio
    ):
        print(
            f"Previous smoke-test empirical charge ratio: "
            f"{empirical_ratio:.1f}×"
        )
        print(
            f"VERY ROUGH charge extrapolation: "
            f"{rough_charge:.1f} s"
        )

    print(
        "The extrapolation is only a planning heuristic, "
        "not an IBM billing formula."
    )

    print()
    print(
        "Service usage before:"
    )
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
                else
                "DRY_RUN"
            ),
        "protocol":
            "RWP_W7",
        "window":
            WINDOW,
        "readout":
            "XZ_injection",
        "measured_qubits":
            [
                0,
                1,
                2,
                3,
            ],
        "candidate":
            {
                **{
                    k:
                        v
                    for k, v
                    in candidate.items()
                    if k != "J"
                },
                "J":
                    {
                        f"{i}{j}":
                            value
                        for (
                            i,
                            j,
                        ), value
                        in candidate[
                            "J"
                        ].items()
                    },
            },
        "selected_lambda":
            selected_lambda,
        "ideal_full_2025_RWP7_RMSE":
            ideal_val_rmse,
        "ideal_full_2025_RWP7_MAE":
            ideal_val_mae,
        "ideal_subset_RMSE":
            ideal_subset_rmse,
        "backend":
            args.backend,
        "layout":
            layout,
        "eval_days":
            args.eval_days,
        "val_offset":
            args.val_offset,
        "shots":
            args.shots,
        "n_circuits":
            len(
                isa
            ),
        "scheduled_duration_all_shots_s":
            scheduled_sum_s,
        "prior_empirical_charge_ratio":
            empirical_ratio,
        "rough_estimated_charge_s":
            rough_charge,
        "service_usage_before":
            json_safe(
                usage_before
            ),
    }

    with open(
        RESULTS
        /
        "11_0C_preflight.json",
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
            "If this workload is acceptable, run:"
        )
        print(
            f"python {Path(__file__).name} "
            f"--submit "
            f"--backend {args.backend} "
            f"--eval-days {args.eval_days} "
            f"--val-offset {args.val_offset} "
            f"--shots {args.shots}"
        )
        return

    # Conservative budget guard.
    if (
        np.isfinite(
            remaining
        )
        and
        np.isfinite(
            rough_charge
        )
        and
        rough_charge
        >
        0.35
        *
        remaining
        and
        not args.force
    ):
        raise RuntimeError(
            f"Rough extrapolated charge {rough_charge:.1f}s exceeds "
            f"35% of remaining allocation {remaining:.1f}s. "
            f"Reduce --eval-days/--shots or explicitly add --force."
        )

    # =========================================================================
    # REAL QPU SUBMISSION
    # =========================================================================

    sampler = Sampler(
        mode=backend
    )

    print()
    print(
        "Submitting one SamplerV2 job for the preliminary RWP_W7 forecast..."
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
        int(
            endpoint
        ):
            {}
        for endpoint in subset_endpoints
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
                    meta[
                        "validation_offset"
                    ],
                "basis":
                    basis,
                "bitstring":
                    bitstring,
                "count":
                    int(
                        count
                    ),
            })

    pd.DataFrame(
        counts_rows
    ).to_csv(
        RESULTS
        /
        "11_0C_counts.csv",
        index=False,
    )

    # =========================================================================
    # HARDWARE FEATURES + FORECAST
    # =========================================================================

    X_qpu = np.vstack(
        [
            xz_from_counts(
                endpoint_counts[
                    int(
                        endpoint
                    )
                ][
                    "X"
                ],
                endpoint_counts[
                    int(
                        endpoint
                    )
                ][
                    "Z"
                ],
            )
            for endpoint in subset_endpoints
        ]
    )

    pred_qpu = predict_scaled_ridge(
        ridge,
        scaler,
        keep,
        X_qpu,
    )

    qpu_rmse = metric_rmse(
        y_subset,
        pred_qpu,
    )

    qpu_mae = metric_mae(
        y_subset,
        pred_qpu,
    )

    qpu_bias = metric_bias(
        y_subset,
        pred_qpu,
    )

    feature_mae = metric_mae(
        X_subset_ideal.ravel(),
        X_qpu.ravel(),
    )

    feature_rmse = metric_rmse(
        X_subset_ideal.ravel(),
        X_qpu.ravel(),
    )

    # =========================================================================
    # OUTPUT TABLES
    # =========================================================================

    prediction_rows = []

    for local_i, endpoint in enumerate(
        subset_endpoints
    ):
        row = {
            "validation_offset":
                int(
                    endpoint
                    -
                    n_train
                ),
            "endpoint_global":
                int(
                    endpoint
                ),
            "target":
                float(
                    y_subset[
                        local_i
                    ]
                ),
            "pred_ideal_RWP_W7":
                float(
                    pred_subset_ideal[
                        local_i
                    ]
                ),
            "pred_QPU_RWP_W7":
                float(
                    pred_qpu[
                        local_i
                    ]
                ),
            "abs_error_ideal":
                float(
                    abs(
                        pred_subset_ideal[
                            local_i
                        ]
                        -
                        y_subset[
                            local_i
                        ]
                    )
                ),
            "abs_error_QPU":
                float(
                    abs(
                        pred_qpu[
                            local_i
                        ]
                        -
                        y_subset[
                            local_i
                        ]
                    )
                ),
            "feature_MAE_QPU_vs_ideal":
                metric_mae(
                    X_subset_ideal[
                        local_i
                    ],
                    X_qpu[
                        local_i
                    ],
                ),
        }

        if (
            "target_date"
            in
            work_eval.columns
        ):
            row[
                "target_date"
            ] = str(
                pd.to_datetime(
                    work_eval.iloc[
                        int(
                            endpoint
                        )
                    ][
                        "target_date"
                    ]
                ).date()
            )

        prediction_rows.append(
            row
        )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    predictions_df.to_csv(
        RESULTS
        /
        "11_0C_predictions.csv",
        index=False,
    )

    feature_rows = []

    for local_i, endpoint in enumerate(
        subset_endpoints
    ):
        for j, feature in enumerate(
            XZ_COLUMNS
        ):
            feature_rows.append({
                "validation_offset":
                    int(
                        endpoint
                        -
                        n_train
                    ),
                "feature":
                    feature,
                "ideal":
                    float(
                        X_subset_ideal[
                            local_i,
                            j,
                        ]
                    ),
                "hardware":
                    float(
                        X_qpu[
                            local_i,
                            j,
                        ]
                    ),
                "error_hardware_minus_ideal":
                    float(
                        X_qpu[
                            local_i,
                            j,
                        ]
                        -
                        X_subset_ideal[
                            local_i,
                            j,
                        ]
                    ),
                "absolute_error":
                    float(
                        abs(
                            X_qpu[
                                local_i,
                                j,
                            ]
                            -
                            X_subset_ideal[
                                local_i,
                                j,
                            ]
                        )
                    ),
            })

    pd.DataFrame(
        feature_rows
    ).to_csv(
        RESULTS
        /
        "11_0C_features.csv",
        index=False,
    )

    # =========================================================================
    # ACTUAL IBM USAGE
    # =========================================================================

    try:
        job_usage = float(
            job.usage(
                partial=False
            )
        )
    except Exception as exc:
        job_usage = np.nan
        job_usage_error = str(
            exc
        )
    else:
        job_usage_error = ""

    try:
        job_metrics = job.metrics()
    except Exception as exc:
        job_metrics = {
            "available":
                False,
            "error":
                str(
                    exc
                ),
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
            job_usage_error,
        "job_metrics":
            json_safe(
                job_metrics
            ),
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
        "11_0C_usage.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(
                usage_record
            ),
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
        "QPU_subset_RMSE":
            qpu_rmse,
        "QPU_subset_MAE":
            qpu_mae,
        "QPU_subset_bias":
            qpu_bias,
        "QPU_feature_MAE_vs_ideal":
            feature_mae,
        "QPU_feature_RMSE_vs_ideal":
            feature_rmse,
    }

    with open(
        RESULTS
        /
        "11_0C_summary.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(
                summary
            ),
            fp,
            indent=2,
        )

    # =========================================================================
    # REPORT
    # =========================================================================

    print()
    print("=" * 120)
    print("PRELIMINARY REAL-QPU RWP_W7 FORECAST RESULTS")
    print("=" * 120)
    print(
        f"Full ideal 2025 RWP_W7 RMSE: "
        f"{ideal_val_rmse:.6f}"
    )
    print(
        f"Subset ideal RWP_W7 RMSE:    "
        f"{ideal_subset_rmse:.6f}"
    )
    print(
        f"Subset REAL-QPU RMSE:         "
        f"{qpu_rmse:.6f}"
    )
    print(
        f"Subset REAL-QPU MAE:          "
        f"{qpu_mae:.6f}"
    )
    print(
        f"Subset REAL-QPU bias:         "
        f"{qpu_bias:+.6f}"
    )
    print(
        f"Hardware feature MAE:         "
        f"{feature_mae:.6f}"
    )
    print(
        f"Hardware feature RMSE:        "
        f"{feature_rmse:.6f}"
    )
    print(
        f"Actual QPU charge:            "
        f"{job_usage} s"
    )

    print()
    print(
        predictions_df.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    for name in [
        "11_0C_ridge_cv_folds.csv",
        "11_0C_ridge_cv_summary.csv",
        "11_0C_hardware_ranking_XZinj.csv",
        "11_0C_transpilation.csv",
        "11_0C_preflight.json",
        "11_0C_counts.csv",
        "11_0C_features.csv",
        "11_0C_predictions.csv",
        "11_0C_usage.json",
        "11_0C_summary.json",
    ]:
        print(
            f"  results/{name}"
        )


if __name__ == "__main__":
    main()
