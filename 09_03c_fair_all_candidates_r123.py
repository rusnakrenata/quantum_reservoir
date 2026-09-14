
from __future__ import annotations

import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit, transpile

from ibm_account import get_service


# =============================================================================
# WEEK 9 — FAIR TROTTER-DEPTH RE-OPTIMIZATION
#
# PURPOSE
# -------
# Re-optimize ALL historical H0-H4 candidate branches under the SAME search
# budget for r = 1, 2, 3.
#
# This replaces the earlier idea of testing only the final 10 retained rows.
#
# Candidate universe
# ------------------
# Historical h_y branches:
#
#   H0:  0.0, -0.4
#   H1:  0.0, -0.6
#   H2:  0.0, +0.3, +0.4, +0.6
#   H3: -0.2,  0.0, +0.3
#   H4: -0.6,  0.0, +0.4, +0.6
#
# For EVERY dynamics seed we retain all three readouts:
#
#   XZ_injection
#   XZinj_plus_YX45
#   XYZ_all
#
# Total historical branches = 15 dynamics seeds * 3 readouts = 45.
#
# FAIRNESS RULE
# -------------
# r=1, r=2 and r=3 receive the SAME:
#
#   * h_x search grid
#   * h_y search rule
#   * global J scaling grid
#   * local independent h_x / h_y / J refinement budget
#   * Ridge lambda grid
#   * chronological training-only CV
#   * exact-vs-Trotter fidelity calculation
#   * memory benchmark
#   * IBM physical-region compilation
#
# Therefore r=2 is RE-OPTIMIZED too.  We do not compare a tuned r=1/r=3
# against an untuned frozen r=2, because that would give unequal search budgets.
#
# Search design
# -------------
# Stage A:
#   h_x in {0.25, 0.50, 0.75}
#
#   Y-OFF:
#       h_y = 0 exactly
#
#   Y-ON, starting from historical h_y0:
#       h_y in {0.5*h_y0, 1.0*h_y0, 1.5*h_y0, -1.0*h_y0}
#
#   global J scale in {0.5, 1.0, 1.5}
#
# Stage B:
#   local independent perturbations around the UNION of:
#       * best chronological-CV parent for each of the 3 readouts
#       * best exact-vs-Trotter-fidelity parent
#
#   Every local child varies:
#       h_x
#       h_y  (unless Y-OFF)
#       every topology-specific J independently
#
# No weighted score is used.
#
# Full expensive diagnostics
# --------------------------
# For every historical branch and r, the forecast-best configuration receives:
#
#   * selected Ridge lambda
#   * 5-fold training-only chronological CV RMSE
#   * 2025 validation RMSE and bias
#   * exact-vs-Trotter process fidelity / infidelity
#   * five-seed intrinsic MC / washout
#   * all 3 top physical embeddings
#   * IBM backends: Fez, Kingston, Marrakesh
#   * optimization levels 0,1,2,3 (0/1 primary)
#   * CZ, SWAP, 1Q/2Q counts, total/1Q/2Q depth
#   * ALAP scheduled duration
#   * duration / min(T1), duration / min(T2)
#   * grouped readout timing at 1024 shots/setting
#
# Fidelity-best rows and the full CV-vs-fidelity Pareto front are also saved,
# but the expensive MC/hardware pass is performed on forecast-best rows.
#
# 2026 test data are never used.
# =============================================================================


# =============================================================================
# PATHS / DEPENDENCIES
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

BASE = HERE / "07_02b_memory_pair_isolation.py"
J_WINNER_FILE = RESULTS / "08_03b_J_search_topology_winners.csv"
EMBEDDING_FILE = RESULTS / "08_03a_final_top3_embeddings.csv"

for path in [BASE, J_WINNER_FILE, EMBEDDING_FILE]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required dependency not found: {path}"
        )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


qrc = load_module(BASE, "qrc_09_03c_all_candidates")


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

ALPHA = 0.75
DT = 1.6
R_VALUES = [1, 2, 3]

HX_REFERENCE = 0.5
HX_GRID = [0.25, 0.50, 0.75]
J_SCALE_GRID = [0.50, 1.00, 1.50]

HY_FACTORS = [0.50, 1.00, 1.50, -1.00]

HX_BOUNDS = (0.10, 1.00)
HY_BOUNDS = (-1.00, 1.00)
J_BOUNDS = (-2.00, 2.00)

LOCAL_CHILDREN_PER_PARENT = 4
HX_LOCAL_SIGMA = 0.12
HY_LOCAL_REL_SIGMA = 0.20
HY_LOCAL_MIN_SIGMA = 0.08
J_LOCAL_REL_SIGMA = 0.20
J_LOCAL_MIN_SIGMA = 0.10

RIDGE_GRID = np.array(
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0],
    dtype=float,
)
N_CV_SPLITS = 5
VAR_TOL = 1e-12

PROBE_SEEDS = [79001, 42, 101, 505, 707]
K_MAX = 20
MIN_BURNIN = 100
POST_WASHOUT_MARGIN = 20
MIN_REMAINING_TRAIN = 100
MC_RIDGE_ALPHA = 1e-6
WASHOUT_EPS = 0.01

BACKEND_NAMES = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh",
]
OPT_LEVELS = [0, 1, 2, 3]
PRIMARY_LEVELS = [0, 1]
SEED_TRANSPILER = 42
SCHEDULING_METHOD = "alap"
SCHEDULING_LABEL = "ALAP"
SHOTS_PER_SETTING = 1024

# Resume completed search blocks after interruption.
RESUME_SEARCH = True
RESUME_MEMORY = True
RESUME_HARDWARE = True

qrc.ALPHA = ALPHA


# =============================================================================
# HISTORICAL 45-CANDIDATE UNIVERSE
# =============================================================================

HY_SEEDS = {
    "H0": [0.0, -0.4],
    "H1": [0.0, -0.6],
    "H2": [0.0, +0.3, +0.4, +0.6],
    "H3": [-0.2, 0.0, +0.3],
    "H4": [-0.6, 0.0, +0.4, +0.6],
}

READOUTS = [
    "XZ_injection",
    "XZinj_plus_YX45",
    "XYZ_all",
]

READOUT_FEATURES = {
    "XZ_injection": [
        "X0", "X1", "X2", "X3",
        "Z0", "Z1", "Z2", "Z3",
    ],
    "XZinj_plus_YX45": [
        "X0", "X1", "X2", "X3",
        "Z0", "Z1", "Z2", "Z3",
        "YX_45",
    ],
    "XYZ_all": [
        "X0", "X1", "X2", "X3", "X4", "X5",
        "Y0", "Y1", "Y2", "Y3", "Y4", "Y5",
        "Z0", "Z1", "Z2", "Z3", "Z4", "Z5",
    ],
}

READOUT_SETTINGS = {
    "XZ_injection": [
        "XXXXZZ",
        "ZZZZZZ",
    ],
    "XZinj_plus_YX45": [
        "XXXXYX",
        "ZZZZZZ",
    ],
    "XYZ_all": [
        "XXXXXX",
        "YYYYYY",
        "ZZZZZZ",
    ],
}


def value_slug(x: float) -> str:
    if np.isclose(x, 0.0):
        return "0"
    sign = "p" if x > 0 else "m"
    s = f"{abs(float(x)):.3f}".rstrip("0").rstrip(".")
    return sign + s.replace(".", "p")


candidate_rows = []

for topology, hy_values in HY_SEEDS.items():
    for seed_hy in hy_values:
        y_state = "OFF" if np.isclose(seed_hy, 0.0) else "ON"
        dynamics_seed_id = f"{topology}_hy{value_slug(seed_hy)}"

        for readout in READOUTS:
            candidate_rows.append({
                "seed_candidate_id":
                    f"{dynamics_seed_id}__{readout}",
                "dynamics_seed_id":
                    dynamics_seed_id,
                "topology":
                    topology,
                "seed_hy":
                    float(seed_hy),
                "y_state":
                    y_state,
                "readout":
                    readout,
            })

candidate_universe = pd.DataFrame(candidate_rows)

if len(candidate_universe) != 45:
    raise RuntimeError(
        f"Expected 45 historical branches, got {len(candidate_universe)}"
    )

candidate_universe.to_csv(
    RESULTS / "09_03c_candidate_universe_45.csv",
    index=False,
)


# =============================================================================
# TOPOLOGIES
# =============================================================================

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


# =============================================================================
# DATA
# =============================================================================

work, train, val, cols = qrc.load_data()

N_TRAIN = len(train)
N_VAL = len(val)

if (N_TRAIN, N_VAL) != (1095, 365):
    raise RuntimeError(
        f"Expected train/val=1095/365, got {N_TRAIN}/{N_VAL}"
    )

work_eval = pd.concat(
    [train, val],
    axis=0,
).reset_index(drop=True)

N_TOTAL = len(work_eval)

if N_TOTAL != 1460:
    raise RuntimeError(
        f"Expected 1460 train+val rows, got {N_TOTAL}"
    )

REAL_ANGLES = np.asarray(
    qrc.make_input_angles(
        work_eval,
        cols,
    ),
    dtype=float,
)

Y_TARGET = work_eval[
    cols["target"]
].to_numpy(dtype=float)

FIRST_INPUT_ANGLES = REAL_ANGLES[0].copy()


# =============================================================================
# FROZEN WEEK-8 J
# =============================================================================

j_df = pd.read_csv(J_WINNER_FILE)

REFERENCE_J = {}

for topology in TOPOLOGY_EDGES:
    matches = j_df[
        j_df["topology"] == topology
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{topology}: expected exactly one Week-8 J winner, "
            f"got {len(matches)}"
        )

    row = matches.iloc[0]

    REFERENCE_J[topology] = {}

    for edge, column in EDGE_TO_COLUMN[topology].items():
        REFERENCE_J[topology][edge] = float(row[column])


# =============================================================================
# PROJECT-CONSISTENT OPERATORS
# =============================================================================

I64 = qrc.I64

INJ_OPS = getattr(
    qrc,
    "INJECTION_SINGLE_OPS",
    getattr(qrc, "INJECTION_OPS", None),
)

MEM_OPS = getattr(
    qrc,
    "MEMORY_SINGLE_OPS",
    getattr(qrc, "MEMORY_OPS", None),
)

if INJ_OPS is None or MEM_OPS is None:
    raise AttributeError(
        "Could not resolve injection/memory Pauli operator dictionaries "
        "from 07_02b_memory_pair_isolation.py"
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

# q4 is local memory qubit 0, q5 is local memory qubit 1.
YX45_MEMORY_OP = np.kron(
    qrc.Y2,
    qrc.X2,
)


# =============================================================================
# UNITARIES
# =============================================================================

def pauli_exp(operator, theta: float):
    # exp(-i theta P), P^2=I
    return (
        np.cos(theta) * I64
        - 1j * np.sin(theta) * operator
    )


def build_hamiltonian(
    topology: str,
    hx: float,
    hy: float,
    J: dict,
):
    H = np.zeros_like(I64)

    for edge in TOPOLOGY_EDGES[topology]:
        H += float(J[edge]) * ZZ_OPS[edge]

    for q in range(qrc.N_QUBITS):
        H += float(hx) * qrc.FULL_SINGLE_OPS[f"X{q}"]

    if not np.isclose(hy, 0.0):
        H += float(hy) * (
            qrc.FULL_SINGLE_OPS["Y4"]
            + qrc.FULL_SINGLE_OPS["Y5"]
        )

    return H


def exact_unitary(H):
    eigvals, eigvecs = np.linalg.eigh(H)

    phases = np.exp(
        -1j * eigvals * DT
    )

    return (
        eigvecs * phases[np.newaxis, :]
    ) @ eigvecs.conj().T


def trotter_unitary(
    topology: str,
    hx: float,
    hy: float,
    J: dict,
    r: int,
):
    U = I64.copy()

    for _ in range(int(r)):
        # 1) ZZ layer
        for edge in TOPOLOGY_EDGES[topology]:
            theta = float(J[edge]) * DT / r
            U = pauli_exp(
                ZZ_OPS[edge],
                theta,
            ) @ U

        # 2) X layer
        theta_x = float(hx) * DT / r

        for q in range(qrc.N_QUBITS):
            U = pauli_exp(
                qrc.FULL_SINGLE_OPS[f"X{q}"],
                theta_x,
            ) @ U

        # 3) memory-only Y layer
        if not np.isclose(hy, 0.0):
            theta_y = float(hy) * DT / r

            for q in [4, 5]:
                U = pauli_exp(
                    qrc.FULL_SINGLE_OPS[f"Y{q}"],
                    theta_y,
                ) @ U

    return U


def unitary_accuracy(U_exact, U_trotter):
    d = U_exact.shape[0]

    overlap = np.trace(
        U_exact.conj().T @ U_trotter
    )

    fidelity = float(
        np.clip(
            (abs(overlap) / d) ** 2,
            0.0,
            1.0,
        )
    )

    phase = float(np.angle(overlap))

    aligned = (
        np.exp(-1j * phase)
        * U_trotter
    )

    fro_error = float(
        np.linalg.norm(
            U_exact - aligned,
            ord="fro",
        )
        / math.sqrt(d)
    )

    return {
        "process_fidelity":
            fidelity,
        "process_infidelity":
            1.0 - fidelity,
        "phase_aligned_normalized_fro_error":
            fro_error,
        "global_phase_alignment_rad":
            phase,
    }


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def expectation(rho, op):
    return float(
        np.real_if_close(
            np.trace(rho @ op),
            tol=1000,
        ).real
    )


def feature_bank_from_A(A_list):
    rows = []
    rho_m = qrc.memory_zero_density()

    for A in A_list:
        rho_i, rho_m_out = qrc.final_reduced_states(
            A,
            rho_m,
        )

        row = {}

        for q in range(4):
            for p in ["X", "Y", "Z"]:
                name = f"{p}{q}"
                row[name] = expectation(
                    rho_i,
                    INJ_OPS[name],
                )

        for q in [4, 5]:
            for p in ["X", "Y", "Z"]:
                name = f"{p}{q}"
                row[name] = expectation(
                    rho_m_out,
                    MEM_OPS[name],
                )

        row["YX_45"] = expectation(
            rho_m_out,
            YX45_MEMORY_OP,
        )

        rows.append(row)
        rho_m = rho_m_out

    return pd.DataFrame(rows)


def simulate_all_readouts(U, angles):
    A_list, S_list = qrc.build_input_channels(
        U,
        angles,
    )

    bank = feature_bank_from_A(A_list)

    return bank, A_list, S_list


# =============================================================================
# RIDGE / FORECAST
# =============================================================================

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


def fit_transform_fold(Xtr, Xva):
    Xtr = np.asarray(Xtr, dtype=float)
    Xva = np.asarray(Xva, dtype=float)

    std = np.std(
        Xtr,
        axis=0,
        ddof=0,
    )

    keep = std > VAR_TOL

    if not np.any(keep):
        return None, None, keep

    scaler = StandardScaler()
    Xtr_z = scaler.fit_transform(
        Xtr[:, keep]
    )
    Xva_z = scaler.transform(
        Xva[:, keep]
    )

    return Xtr_z, Xva_z, keep


def forecast_cv_for_lambda(
    X_train,
    y_train,
    ridge_alpha,
):
    splitter = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )

    values = []

    for tr, va in splitter.split(X_train):
        Xtr_z, Xva_z, keep = fit_transform_fold(
            X_train[tr],
            X_train[va],
        )

        if not np.any(keep):
            pred = np.full(
                len(va),
                float(np.mean(y_train[tr])),
            )

        else:
            model = Ridge(
                alpha=float(ridge_alpha),
                fit_intercept=True,
            )

            model.fit(
                Xtr_z,
                y_train[tr],
            )

            pred = model.predict(
                Xva_z
            )

        values.append(
            rmse(
                y_train[va],
                pred,
            )
        )

    return np.asarray(values, dtype=float)


def select_ridge_and_validate(
    X,
):
    X = np.asarray(X, dtype=float)

    X_train = X[:N_TRAIN]
    X_val = X[N_TRAIN:]

    y_train = Y_TARGET[:N_TRAIN]
    y_val = Y_TARGET[N_TRAIN:]

    grid_rows = []

    for ridge_alpha in RIDGE_GRID:
        folds = forecast_cv_for_lambda(
            X_train,
            y_train,
            ridge_alpha,
        )

        grid_rows.append({
            "ridge_alpha":
                float(ridge_alpha),
            "cv_rmse_mean":
                float(np.mean(folds)),
            "cv_rmse_std":
                float(np.std(folds, ddof=1)),
        })

    grid = pd.DataFrame(grid_rows).sort_values(
        ["cv_rmse_mean", "ridge_alpha"],
        ascending=[True, True],
    ).reset_index(drop=True)

    best = grid.iloc[0]
    selected_lambda = float(
        best["ridge_alpha"]
    )

    Xtr_z, Xval_z, keep = fit_transform_fold(
        X_train,
        X_val,
    )

    if not np.any(keep):
        pred = np.full(
            N_VAL,
            float(np.mean(y_train)),
        )
        n_features = 0
        feature_rank = 0

    else:
        model = Ridge(
            alpha=selected_lambda,
            fit_intercept=True,
        )

        model.fit(
            Xtr_z,
            y_train,
        )

        pred = model.predict(
            Xval_z
        )

        n_features = int(np.sum(keep))
        feature_rank = int(
            np.linalg.matrix_rank(
                Xtr_z
            )
        )

    return {
        "selected_lambda":
            selected_lambda,
        "cv_rmse":
            float(best["cv_rmse_mean"]),
        "cv_rmse_std":
            float(best["cv_rmse_std"]),
        "validation_rmse":
            rmse(y_val, pred),
        "validation_bias":
            float(np.mean(pred - y_val)),
        "n_features":
            n_features,
        "feature_rank":
            feature_rank,
        "ridge_grid":
            grid,
    }


# =============================================================================
# SEARCH HELPERS
# =============================================================================

def clipped(value, bounds):
    return float(
        np.clip(
            float(value),
            bounds[0],
            bounds[1],
        )
    )


def scale_J(J0, scale):
    return {
        edge:
            clipped(
                float(value) * float(scale),
                J_BOUNDS,
            )
        for edge, value in J0.items()
    }


def J_columns(topology, J):
    out = {
        "J01": np.nan,
        "J12": np.nan,
        "J23": np.nan,
        "J24": np.nan,
        "J34": np.nan,
        "J35": np.nan,
        "J04": np.nan,
        "J45": np.nan,
    }

    for edge, column in EDGE_TO_COLUMN[topology].items():
        out[column] = float(J[edge])

    return out


def J_from_row(row):
    topology = str(row["topology"])

    return {
        edge:
            float(row[column])
        for edge, column
        in EDGE_TO_COLUMN[topology].items()
    }


def config_key(topology, hx, hy, J):
    values = [
        topology,
        round(float(hx), 10),
        round(float(hy), 10),
    ]

    for edge in TOPOLOGY_EDGES[topology]:
        values.append(
            round(
                float(J[edge]),
                10,
            )
        )

    return tuple(values)


# =============================================================================
# SEARCH CHECKPOINTS
# =============================================================================

SEARCH_CHECKPOINT = (
    RESULTS / "09_03c_search_all_checkpoint.csv"
)
RIDGE_CHECKPOINT = (
    RESULTS / "09_03c_ridge_grid_checkpoint.csv"
)
COMPLETED_BLOCKS_FILE = (
    RESULTS / "09_03c_completed_search_blocks.csv"
)

if (
    RESUME_SEARCH
    and SEARCH_CHECKPOINT.exists()
):
    search_rows_global = pd.read_csv(
        SEARCH_CHECKPOINT
    ).to_dict("records")
else:
    search_rows_global = []

if (
    RESUME_SEARCH
    and RIDGE_CHECKPOINT.exists()
):
    ridge_rows_global = pd.read_csv(
        RIDGE_CHECKPOINT
    ).to_dict("records")
else:
    ridge_rows_global = []

if (
    RESUME_SEARCH
    and COMPLETED_BLOCKS_FILE.exists()
):
    completed_blocks_df = pd.read_csv(
        COMPLETED_BLOCKS_FILE
    )
    completed_blocks = set(
        zip(
            completed_blocks_df[
                "dynamics_seed_id"
            ].astype(str),
            completed_blocks_df[
                "trotter_r"
            ].astype(int),
        )
    )
else:
    completed_blocks = set()


def save_search_checkpoints():
    pd.DataFrame(
        search_rows_global
    ).to_csv(
        SEARCH_CHECKPOINT,
        index=False,
    )

    pd.DataFrame(
        ridge_rows_global
    ).to_csv(
        RIDGE_CHECKPOINT,
        index=False,
    )

    pd.DataFrame(
        sorted(completed_blocks),
        columns=[
            "dynamics_seed_id",
            "trotter_r",
        ],
    ).to_csv(
        COMPLETED_BLOCKS_FILE,
        index=False,
    )


# =============================================================================
# DYNAMICS EVALUATOR
#
# One quantum trajectory is reused for ALL 3 readouts.
# =============================================================================

def evaluate_dynamics(
    *,
    dynamics_seed_id,
    topology,
    seed_hy,
    y_state,
    r,
    stage,
    hx,
    hy,
    J,
    config_serial,
    parent_ids=None,
):
    H = build_hamiltonian(
        topology,
        hx,
        hy,
        J,
    )

    U_exact = exact_unitary(H)

    U_trotter = trotter_unitary(
        topology,
        hx,
        hy,
        J,
        r,
    )

    accuracy = unitary_accuracy(
        U_exact,
        U_trotter,
    )

    bank, _, _ = simulate_all_readouts(
        U_trotter,
        REAL_ANGLES,
    )

    config_id = (
        f"{dynamics_seed_id}_r{r}_"
        f"{stage}_{config_serial:04d}"
    )

    rows = []
    ridge_rows = []

    for readout in READOUTS:
        X = bank[
            READOUT_FEATURES[readout]
        ].to_numpy(dtype=float)

        forecast = select_ridge_and_validate(X)

        seed_candidate_id = (
            f"{dynamics_seed_id}__{readout}"
        )

        row = {
            "config_id":
                config_id,
            "seed_candidate_id":
                seed_candidate_id,
            "dynamics_seed_id":
                dynamics_seed_id,
            "topology":
                topology,
            "seed_hy":
                float(seed_hy),
            "y_state":
                y_state,
            "readout":
                readout,
            "trotter_r":
                int(r),
            "stage":
                stage,
            "parent_ids":
                (
                    ""
                    if parent_ids is None
                    else "|".join(
                        sorted(set(parent_ids))
                    )
                ),
            "hx":
                float(hx),
            "hy":
                float(hy),
            **J_columns(
                topology,
                J,
            ),
            "selected_lambda":
                forecast["selected_lambda"],
            "cv_rmse":
                forecast["cv_rmse"],
            "cv_rmse_std":
                forecast["cv_rmse_std"],
            "validation_rmse":
                forecast["validation_rmse"],
            "validation_bias":
                forecast["validation_bias"],
            "n_features":
                forecast["n_features"],
            "feature_rank":
                forecast["feature_rank"],
            **accuracy,
        }

        rows.append(row)

        for _, rr in forecast[
            "ridge_grid"
        ].iterrows():
            ridge_rows.append({
                "config_id":
                    config_id,
                "seed_candidate_id":
                    seed_candidate_id,
                "dynamics_seed_id":
                    dynamics_seed_id,
                "trotter_r":
                    int(r),
                "readout":
                    readout,
                "ridge_alpha":
                    float(rr["ridge_alpha"]),
                "cv_rmse_mean":
                    float(rr["cv_rmse_mean"]),
                "cv_rmse_std":
                    float(rr["cv_rmse_std"]),
            })

    return rows, ridge_rows


# =============================================================================
# FAIR SEARCH FOR r=1,2,3
# =============================================================================

print("=" * 132)
print("WEEK 9 — FAIR r=1 / r=2 / r=3 RE-OPTIMIZATION")
print("=" * 132)
print(f"Historical branches: {len(candidate_universe)}")
print(f"Dynamics seeds:      {candidate_universe['dynamics_seed_id'].nunique()}")
print(f"Readouts:            {READOUTS}")
print(f"r values:            {R_VALUES}")
print("r=2 is re-optimized with the same budget as r=1 and r=3.")
print("2026 test remains untouched.")
print()

global_start = time.perf_counter()

dynamics_seed_table = (
    candidate_universe[
        [
            "dynamics_seed_id",
            "topology",
            "seed_hy",
            "y_state",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "topology",
            "seed_hy",
        ]
    )
    .reset_index(drop=True)
)

for _, seed_row in dynamics_seed_table.iterrows():
    dynamics_seed_id = str(
        seed_row["dynamics_seed_id"]
    )
    topology = str(
        seed_row["topology"]
    )
    seed_hy = float(
        seed_row["seed_hy"]
    )
    y_state = str(
        seed_row["y_state"]
    )

    J0 = REFERENCE_J[topology]

    for r in R_VALUES:
        block_key = (
            dynamics_seed_id,
            int(r),
        )

        if block_key in completed_blocks:
            print(
                f"[resume] {dynamics_seed_id} r={r} already complete"
            )
            continue

        block_start = time.perf_counter()

        print()
        print("-" * 132)
        print(
            f"{dynamics_seed_id} | {topology} | "
            f"seed_hy={seed_hy:+.3f} | {y_state} | r={r}"
        )
        print("-" * 132)

        block_rows = []
        block_ridge_rows = []
        cache = {}
        config_lookup = {}
        serial = [0]

        def evaluate_unique(
            stage,
            hx,
            hy,
            J,
            parent_ids=None,
        ):
            key = config_key(
                topology,
                hx,
                hy,
                J,
            )

            if key in cache:
                return cache[key]

            serial[0] += 1

            rows, ridge_rows = evaluate_dynamics(
                dynamics_seed_id=dynamics_seed_id,
                topology=topology,
                seed_hy=seed_hy,
                y_state=y_state,
                r=r,
                stage=stage,
                hx=hx,
                hy=hy,
                J=J,
                config_serial=serial[0],
                parent_ids=parent_ids,
            )

            cache[key] = rows
            config_lookup[
                rows[0]["config_id"]
            ] = rows[0]

            block_rows.extend(rows)
            block_ridge_rows.extend(
                ridge_rows
            )

            return rows

        # ---------------------------------------------------------------------
        # Baseline
        # ---------------------------------------------------------------------

        baseline_rows = evaluate_unique(
            stage="baseline",
            hx=HX_REFERENCE,
            hy=seed_hy,
            J=J0,
        )

        print(
            "baseline:"
        )

        for row in baseline_rows:
            print(
                f"  {row['readout']:20s} "
                f"CV={row['cv_rmse']:.6f} "
                f"Val={row['validation_rmse']:.6f} "
                f"Fproc={row['process_fidelity']:.6f}"
            )

        # ---------------------------------------------------------------------
        # Stage A — coarse
        # ---------------------------------------------------------------------

        if y_state == "OFF":
            hy_values = [0.0]
        else:
            hy_values = []

            for factor in HY_FACTORS:
                value = clipped(
                    seed_hy * factor,
                    HY_BOUNDS,
                )

                if abs(value) < 0.05:
                    value = (
                        0.05
                        if seed_hy > 0
                        else -0.05
                    )

                if not any(
                    np.isclose(
                        value,
                        x,
                    )
                    for x in hy_values
                ):
                    hy_values.append(
                        value
                    )

        for hx in HX_GRID:
            for hy in hy_values:
                for scale in J_SCALE_GRID:
                    J = scale_J(
                        J0,
                        scale,
                    )

                    evaluate_unique(
                        stage="coarse",
                        hx=hx,
                        hy=hy,
                        J=J,
                    )

        current = pd.DataFrame(
            block_rows
        )

        # ---------------------------------------------------------------------
        # Parent union:
        #   best CV for EACH readout + best fidelity
        # ---------------------------------------------------------------------

        parent_config_ids = set()

        for readout in READOUTS:
            g = current[
                current["readout"] == readout
            ]

            best = (
                g
                .sort_values(
                    [
                        "cv_rmse",
                        "process_infidelity",
                    ],
                    ascending=[
                        True,
                        True,
                    ],
                )
                .iloc[0]
            )

            parent_config_ids.add(
                str(best["config_id"])
            )

        # Fidelity is dynamics-only, so one row per config is enough.
        unique_dynamics = (
            current
            .sort_values(
                [
                    "process_infidelity",
                    "cv_rmse",
                ]
            )
            .drop_duplicates(
                subset=["config_id"]
            )
        )

        best_fidelity = unique_dynamics.iloc[0]

        parent_config_ids.add(
            str(
                best_fidelity[
                    "config_id"
                ]
            )
        )

        # ---------------------------------------------------------------------
        # Stage B — local independent refinement
        # ---------------------------------------------------------------------

        for parent_index, parent_config_id in enumerate(
            sorted(parent_config_ids)
        ):
            parent = current[
                current["config_id"]
                ==
                parent_config_id
            ].iloc[0]

            parent_hx = float(
                parent["hx"]
            )
            parent_hy = float(
                parent["hy"]
            )
            parent_J = J_from_row(
                parent
            )

            rng_seed = (
                910000
                +
                sum(
                    ord(c)
                    for c in dynamics_seed_id
                )
                +
                1000 * int(r)
                +
                100 * parent_index
            )

            rng = np.random.default_rng(
                rng_seed
            )

            for _ in range(
                LOCAL_CHILDREN_PER_PARENT
            ):
                child_hx = clipped(
                    rng.normal(
                        parent_hx,
                        HX_LOCAL_SIGMA,
                    ),
                    HX_BOUNDS,
                )

                if y_state == "OFF":
                    child_hy = 0.0

                else:
                    hy_sigma = max(
                        HY_LOCAL_MIN_SIGMA,
                        HY_LOCAL_REL_SIGMA
                        *
                        abs(parent_hy),
                    )

                    child_hy = clipped(
                        rng.normal(
                            parent_hy,
                            hy_sigma,
                        ),
                        HY_BOUNDS,
                    )

                    if abs(child_hy) < 0.05:
                        child_hy = (
                            0.05
                            if parent_hy >= 0
                            else -0.05
                        )

                child_J = {}

                for edge, value in parent_J.items():
                    sigma = max(
                        J_LOCAL_MIN_SIGMA,
                        J_LOCAL_REL_SIGMA
                        *
                        max(
                            abs(value),
                            0.5,
                        ),
                    )

                    child_J[edge] = clipped(
                        rng.normal(
                            value,
                            sigma,
                        ),
                        J_BOUNDS,
                    )

                evaluate_unique(
                    stage="local",
                    hx=child_hx,
                    hy=child_hy,
                    J=child_J,
                    parent_ids=[
                        parent_config_id
                    ],
                )

        # ---------------------------------------------------------------------
        # Block summary
        # ---------------------------------------------------------------------

        block_df = pd.DataFrame(
            block_rows
        )

        for readout in READOUTS:
            g = block_df[
                block_df[
                    "readout"
                ]
                ==
                readout
            ]

            best_cv = (
                g
                .sort_values(
                    [
                        "cv_rmse",
                        "process_infidelity",
                    ]
                )
                .iloc[0]
            )

            best_fid = (
                g
                .sort_values(
                    [
                        "process_infidelity",
                        "cv_rmse",
                    ]
                )
                .iloc[0]
            )

            print(
                f"{readout:20s} | "
                f"best CV={best_cv['cv_rmse']:.6f} "
                f"(hx={best_cv['hx']:.3f}, "
                f"hy={best_cv['hy']:+.3f}) | "
                f"best Fproc={best_fid['process_fidelity']:.6f}"
            )

        search_rows_global.extend(
            block_rows
        )

        ridge_rows_global.extend(
            block_ridge_rows
        )

        completed_blocks.add(
            block_key
        )

        save_search_checkpoints()

        print(
            f"block runtime: "
            f"{time.perf_counter() - block_start:.1f}s"
        )


search_df = pd.DataFrame(
    search_rows_global
)

ridge_df = pd.DataFrame(
    ridge_rows_global
)

# De-duplicate on reruns.
search_df = (
    search_df
    .drop_duplicates(
        subset=[
            "config_id",
            "readout",
        ],
        keep="last",
    )
    .reset_index(drop=True)
)

ridge_df = (
    ridge_df
    .drop_duplicates(
        subset=[
            "config_id",
            "readout",
            "ridge_alpha",
        ],
        keep="last",
    )
    .reset_index(drop=True)
)

search_df.to_csv(
    RESULTS / "09_03c_search_all.csv",
    index=False,
)

ridge_df.to_csv(
    RESULTS / "09_03c_ridge_grid_all.csv",
    index=False,
)


# =============================================================================
# PARETO: CV RMSE vs PROCESS INFIDELITY
# =============================================================================

def pareto_mask(
    frame,
    minimize_columns,
    maximize_columns=None,
):
    if maximize_columns is None:
        maximize_columns = []

    cols = []

    for c in minimize_columns:
        cols.append(
            pd.to_numeric(
                frame[c],
                errors="coerce",
            ).to_numpy(dtype=float)
        )

    for c in maximize_columns:
        cols.append(
            -pd.to_numeric(
                frame[c],
                errors="coerce",
            ).to_numpy(dtype=float)
        )

    M = np.column_stack(cols)

    finite = np.all(
        np.isfinite(M),
        axis=1,
    )

    keep = np.zeros(
        len(frame),
        dtype=bool,
    )

    idx = np.where(finite)[0]
    V = M[finite]

    for local_i, global_i in enumerate(idx):
        v = V[local_i]

        no_worse = np.all(
            V <= v,
            axis=1,
        )

        strictly_better = np.any(
            V < v,
            axis=1,
        )

        dominates = (
            no_worse
            &
            strictly_better
        )

        dominates[local_i] = False

        if not np.any(dominates):
            keep[global_i] = True

    return keep


pareto_parts = []

for (
    seed_candidate_id,
    r,
), group in search_df.groupby(
    [
        "seed_candidate_id",
        "trotter_r",
    ],
    sort=True,
):
    g = group.copy().reset_index(
        drop=True
    )

    g[
        "cv_fidelity_pareto"
    ] = pareto_mask(
        g,
        minimize_columns=[
            "cv_rmse",
            "process_infidelity",
        ],
    )

    pareto_parts.append(g)

search_pareto_all = pd.concat(
    pareto_parts,
    ignore_index=True,
)

search_pareto = search_pareto_all[
    search_pareto_all[
        "cv_fidelity_pareto"
    ]
].copy()

search_pareto_all.to_csv(
    RESULTS
    /
    "09_03c_search_pareto_all.csv",
    index=False,
)

search_pareto.to_csv(
    RESULTS
    /
    "09_03c_search_pareto_candidates.csv",
    index=False,
)


# =============================================================================
# FINALISTS
# =============================================================================

forecast_best_rows = []
fidelity_best_rows = []

for (
    seed_candidate_id,
    r,
), group in search_df.groupby(
    [
        "seed_candidate_id",
        "trotter_r",
    ],
    sort=True,
):
    best_cv = (
        group
        .sort_values(
            [
                "cv_rmse",
                "process_infidelity",
            ]
        )
        .iloc[0]
        .copy()
    )

    best_cv[
        "finalist_role"
    ] = "forecast_best"

    forecast_best_rows.append(
        best_cv
    )

    best_fid = (
        group
        .sort_values(
            [
                "process_infidelity",
                "cv_rmse",
            ]
        )
        .iloc[0]
        .copy()
    )

    best_fid[
        "finalist_role"
    ] = "fidelity_best"

    fidelity_best_rows.append(
        best_fid
    )


forecast_best_df = pd.DataFrame(
    forecast_best_rows
).reset_index(drop=True)

fidelity_best_df = pd.DataFrame(
    fidelity_best_rows
).reset_index(drop=True)

forecast_best_df[
    "model_id"
] = [
    f"{row.seed_candidate_id}_r{int(row.trotter_r)}_forecast"
    for _, row
    in forecast_best_df.iterrows()
]

fidelity_best_df[
    "model_id"
] = [
    f"{row.seed_candidate_id}_r{int(row.trotter_r)}_fidelity"
    for _, row
    in fidelity_best_df.iterrows()
]

forecast_best_df.to_csv(
    RESULTS
    /
    "09_03c_forecast_best_135.csv",
    index=False,
)

fidelity_best_df.to_csv(
    RESULTS
    /
    "09_03c_fidelity_best_135.csv",
    index=False,
)


# =============================================================================
# BASELINE AUDIT:
# original Week-8 J, hx=0.5, historical h_y at each r.
# =============================================================================

baseline_audit = search_df[
    search_df["stage"] == "baseline"
].copy()

baseline_audit.to_csv(
    RESULTS
    /
    "09_03c_baseline_all_r.csv",
    index=False,
)


# =============================================================================
# MEMORY / WASHOUT
# Only forecast-best models receive the expensive five-seed MC pass.
# If several readouts share the same config_id, memory is computed once.
# =============================================================================

def ket_density(v):
    v = np.asarray(v, dtype=complex)
    v = v / np.linalg.norm(v)

    return np.outer(
        v,
        v.conj(),
    )


e00 = np.array(
    [1, 0, 0, 0],
    dtype=complex,
)

e11 = np.array(
    [0, 0, 0, 1],
    dtype=complex,
)

epp = (
    np.array(
        [1, 1, 1, 1],
        dtype=complex,
    )
    /
    2.0
)

ebell = (
    np.array(
        [1, 0, 0, 1],
        dtype=complex,
    )
    /
    np.sqrt(2.0)
)

INITIAL_STATES = {
    "00":
        ket_density(e00),
    "11":
        ket_density(e11),
    "++":
        ket_density(epp),
    "BellPhi+":
        ket_density(ebell),
    "I/4":
        np.eye(
            4,
            dtype=complex,
        )
        /
        4.0,
}


def trace_distance(rho, sigma):
    delta = np.asarray(
        rho - sigma,
        dtype=complex,
    )

    delta = (
        0.5
        *
        (
            delta
            +
            delta.conj().T
        )
    )

    vals = np.linalg.eigvalsh(
        delta
    )

    return (
        0.5
        *
        float(
            np.sum(np.abs(vals))
        )
    )


def max_pairwise_trace_distance(
    states,
):
    return max(
        trace_distance(
            states[a],
            states[b],
        )
        for a, b
        in combinations(
            states.keys(),
            2,
        )
    )


def stable_threshold_crossing(
    values,
    threshold,
):
    arr = np.asarray(
        values,
        dtype=float,
    )

    suffix_max = np.maximum.accumulate(
        arr[::-1]
    )[::-1]

    idx = np.flatnonzero(
        suffix_max <= threshold
    )

    return (
        None
        if len(idx) == 0
        else int(idx[0])
    )


def build_probe(seed):
    rng = np.random.default_rng(
        int(seed)
    )

    probe = np.empty_like(
        REAL_ANGLES
    )

    for j in range(
        REAL_ANGLES.shape[1]
    ):
        probe[:, j] = REAL_ANGLES[
            rng.permutation(N_TOTAL),
            j,
        ]

    return probe


def washout_from_S(S_list):
    states = {
        name:
            rho.copy()
        for name, rho
        in INITIAL_STATES.items()
    }

    D_curve = []

    for S in S_list:
        states = {
            name:
                qrc.apply_memory_channel(
                    S,
                    rho,
                )
            for name, rho
            in states.items()
        }

        D_curve.append(
            max_pairwise_trace_distance(
                states
            )
        )

    Tw = stable_threshold_crossing(
        D_curve,
        WASHOUT_EPS,
    )

    return {
        "Tw_0.01":
            (
                np.nan
                if Tw is None
                else int(Tw)
            ),
        "D_final":
            float(D_curve[-1]),
        "washout_valid":
            Tw is not None,
    }


def standardized_ridge_predict(
    Xtr,
    ytr,
    Xva,
    alpha,
):
    Xtr = np.asarray(
        Xtr,
        dtype=float,
    )

    Xva = np.asarray(
        Xva,
        dtype=float,
    )

    ytr = np.asarray(
        ytr,
        dtype=float,
    )

    mu = np.mean(
        Xtr,
        axis=0,
    )

    sd = np.std(
        Xtr,
        axis=0,
        ddof=0,
    )

    keep = sd > VAR_TOL

    if not np.any(keep):
        return np.full(
            len(Xva),
            float(np.mean(ytr)),
        )

    Xtr_z = (
        Xtr[:, keep]
        -
        mu[keep]
    ) / sd[keep]

    Xva_z = (
        Xva[:, keep]
        -
        mu[keep]
    ) / sd[keep]

    model = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
    )

    model.fit(
        Xtr_z,
        ytr,
    )

    return model.predict(
        Xva_z
    )


def safe_corr2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        np.std(a) <= VAR_TOL
        or
        np.std(b) <= VAR_TOL
    ):
        return 0.0

    r = np.corrcoef(a, b)[0, 1]

    if not np.isfinite(r):
        return 0.0

    return float(r * r)


def evaluate_memory_capacity(
    X_xyz,
    probe,
    burnin,
):
    null_floor = (
        1.0
        /
        (
            N_VAL
            -
            1.0
        )
    )

    rows = []

    for channel in range(
        probe.shape[1]
    ):
        for delay in range(
            1,
            K_MAX + 1,
        ):
            start = max(
                int(burnin),
                int(delay),
            )

            tr = np.arange(
                start,
                N_TRAIN,
            )

            va = np.arange(
                N_TRAIN,
                N_TRAIN + N_VAL,
            )

            ytr = probe[
                tr - delay,
                channel,
            ]

            yva = probe[
                va - delay,
                channel,
            ]

            pred = standardized_ridge_predict(
                X_xyz[tr],
                ytr,
                X_xyz[va],
                MC_RIDGE_ALPHA,
            )

            raw = safe_corr2(
                yva,
                pred,
            )

            corrected = max(
                raw - null_floor,
                0.0,
            )

            rows.append({
                "channel":
                    channel,
                "delay":
                    delay,
                "MC_raw":
                    raw,
                "MC_corrected":
                    corrected,
            })

    detail = pd.DataFrame(rows)

    total_corrected = float(
        detail[
            "MC_corrected"
        ].sum()
    )

    def delay_mean(k):
        return float(
            detail.loc[
                detail["delay"] == k,
                "MC_raw",
            ].mean()
        )

    return {
        "MC_total_corrected":
            total_corrected,
        "MC_per_channel_corrected":
            total_corrected / 4.0,
        "MC_delay1_mean":
            delay_mean(1),
        "MC_delay2_mean":
            delay_mean(2),
        "MC_delay5_mean":
            delay_mean(5),
        "MC_delay10_mean":
            delay_mean(10),
        "MC_delay20_mean":
            delay_mean(20),
        "detail":
            detail,
    }


MEMORY_BY_SEED_FILE = (
    RESULTS
    /
    "09_03c_memory_by_seed.csv"
)

MEMORY_BY_DELAY_FILE = (
    RESULTS
    /
    "09_03c_memory_by_delay.csv"
)

if (
    RESUME_MEMORY
    and MEMORY_BY_SEED_FILE.exists()
):
    memory_seed_rows = pd.read_csv(
        MEMORY_BY_SEED_FILE
    ).to_dict("records")
else:
    memory_seed_rows = []

completed_memory = {
    (
        str(row["config_id"]),
        int(row["probe_seed"]),
    )
    for row in memory_seed_rows
}

memory_delay_parts = []

if (
    RESUME_MEMORY
    and MEMORY_BY_DELAY_FILE.exists()
):
    existing_delay = pd.read_csv(
        MEMORY_BY_DELAY_FILE
    )

    if len(existing_delay):
        memory_delay_parts.append(
            existing_delay
        )


forecast_unique_dynamics = (
    forecast_best_df
    .sort_values(
        [
            "config_id",
            "readout",
        ]
    )
    .drop_duplicates(
        subset=[
            "config_id",
        ]
    )
    .reset_index(drop=True)
)

print()
print("=" * 132)
print("FIVE-SEED MC / WASHOUT ON UNIQUE FORECAST-BEST DYNAMICS")
print("=" * 132)
print(
    f"Unique forecast-best dynamics: "
    f"{len(forecast_unique_dynamics)}"
)

for _, row in forecast_unique_dynamics.iterrows():
    config_id = str(
        row["config_id"]
    )

    topology = str(
        row["topology"]
    )

    r = int(
        row["trotter_r"]
    )

    hx = float(
        row["hx"]
    )

    hy = float(
        row["hy"]
    )

    J = J_from_row(row)

    U = trotter_unitary(
        topology,
        hx,
        hy,
        J,
        r,
    )

    for probe_seed in PROBE_SEEDS:
        mem_key = (
            config_id,
            int(probe_seed),
        )

        if mem_key in completed_memory:
            continue

        probe = build_probe(
            probe_seed
        )

        bank, A_list, S_list = simulate_all_readouts(
            U,
            probe,
        )

        wash = washout_from_S(
            S_list
        )

        Tw = wash["Tw_0.01"]

        if np.isfinite(Tw):
            postwash_burnin = max(
                MIN_BURNIN,
                int(Tw)
                +
                POST_WASHOUT_MARGIN,
            )

            max_allowed = (
                N_TRAIN
                -
                MIN_REMAINING_TRAIN
            )

            mc_valid = (
                postwash_burnin
                <=
                max_allowed
            )
        else:
            postwash_burnin = np.nan
            mc_valid = False

        if mc_valid:
            X_xyz = bank[
                READOUT_FEATURES[
                    "XYZ_all"
                ]
            ].to_numpy(dtype=float)

            mc = evaluate_memory_capacity(
                X_xyz,
                probe,
                int(postwash_burnin),
            )
        else:
            mc = {
                "MC_total_corrected":
                    np.nan,
                "MC_per_channel_corrected":
                    np.nan,
                "MC_delay1_mean":
                    np.nan,
                "MC_delay2_mean":
                    np.nan,
                "MC_delay5_mean":
                    np.nan,
                "MC_delay10_mean":
                    np.nan,
                "MC_delay20_mean":
                    np.nan,
                "detail":
                    pd.DataFrame(),
            }

        memory_seed_rows.append({
            "config_id":
                config_id,
            "dynamics_seed_id":
                row["dynamics_seed_id"],
            "topology":
                topology,
            "trotter_r":
                r,
            "probe_seed":
                int(probe_seed),
            "Tw_0.01":
                Tw,
            "D_final":
                wash["D_final"],
            "washout_valid":
                wash["washout_valid"],
            "postwash_burnin":
                postwash_burnin,
            "postwash_MC_valid":
                mc_valid,
            "MC_per_channel_corrected":
                mc[
                    "MC_per_channel_corrected"
                ],
            "MC_total_corrected":
                mc[
                    "MC_total_corrected"
                ],
            "MC_delay1_mean":
                mc[
                    "MC_delay1_mean"
                ],
            "MC_delay2_mean":
                mc[
                    "MC_delay2_mean"
                ],
            "MC_delay5_mean":
                mc[
                    "MC_delay5_mean"
                ],
            "MC_delay10_mean":
                mc[
                    "MC_delay10_mean"
                ],
            "MC_delay20_mean":
                mc[
                    "MC_delay20_mean"
                ],
        })

        if mc_valid:
            d = mc["detail"].copy()
            d["config_id"] = config_id
            d["probe_seed"] = int(
                probe_seed
            )
            d["trotter_r"] = r
            d["topology"] = topology

            memory_delay_parts.append(d)

        completed_memory.add(
            mem_key
        )

        pd.DataFrame(
            memory_seed_rows
        ).to_csv(
            MEMORY_BY_SEED_FILE,
            index=False,
        )

        if memory_delay_parts:
            pd.concat(
                memory_delay_parts,
                ignore_index=True,
            ).drop_duplicates(
                subset=[
                    "config_id",
                    "probe_seed",
                    "channel",
                    "delay",
                ],
                keep="last",
            ).to_csv(
                MEMORY_BY_DELAY_FILE,
                index=False,
            )

        print(
            f"{config_id} seed={probe_seed}: "
            f"MC/ch="
            f"{mc['MC_per_channel_corrected']:.6f} "
            f"Tw={Tw}"
        )


memory_seed_df = pd.DataFrame(
    memory_seed_rows
).drop_duplicates(
    subset=[
        "config_id",
        "probe_seed",
    ],
    keep="last",
)

memory_seed_df.to_csv(
    MEMORY_BY_SEED_FILE,
    index=False,
)

memory_summary = (
    memory_seed_df
    .groupby(
        [
            "config_id",
            "dynamics_seed_id",
            "topology",
            "trotter_r",
        ],
        as_index=False,
    )
    .agg(
        **{
            "n_probe_seeds": (
                "probe_seed",
                "count",
            ),
            "n_valid_washout": (
                "washout_valid",
                "sum",
            ),
            "n_valid_MC": (
                "postwash_MC_valid",
                "sum",
            ),
            "MC_ch_mean": (
                "MC_per_channel_corrected",
                "mean",
            ),
            "MC_ch_std": (
                "MC_per_channel_corrected",
                "std",
            ),
            "MC_total_mean": (
                "MC_total_corrected",
                "mean",
            ),
            "Tw_0.01_mean": (
                "Tw_0.01",
                "mean",
            ),
            "Tw_0.01_std": (
                "Tw_0.01",
                "std",
            ),
            "D_final_mean": (
                "D_final",
                "mean",
            ),
            "MC_delay1_mean": (
                "MC_delay1_mean",
                "mean",
            ),
            "MC_delay2_mean": (
                "MC_delay2_mean",
                "mean",
            ),
            "MC_delay5_mean": (
                "MC_delay5_mean",
                "mean",
            ),
            "MC_delay10_mean": (
                "MC_delay10_mean",
                "mean",
            ),
            "MC_delay20_mean": (
                "MC_delay20_mean",
                "mean",
            ),
        }
    )
)

memory_summary.to_csv(
    RESULTS
    /
    "09_03c_memory_summary.csv",
    index=False,
)


# =============================================================================
# PHYSICAL EMBEDDINGS
# =============================================================================

embeddings = pd.read_csv(
    EMBEDDING_FILE
)

required_embedding_columns = [
    "backend",
    "candidate",
    "final_rank",
    "physical_C_t",
    "physical_D",
    "physical_P_t",
    "physical_H",
    "physical_M1",
    "physical_M2",
]

missing = [
    c
    for c in required_embedding_columns
    if c not in embeddings.columns
]

if missing:
    raise KeyError(
        f"Embedding file missing columns: {missing}"
    )

embeddings = embeddings[
    embeddings[
        "candidate"
    ].isin(
        list(TOPOLOGY_EDGES)
    )
].copy()

embeddings[
    "layout"
] = embeddings.apply(
    lambda row: [
        int(row["physical_C_t"]),
        int(row["physical_D"]),
        int(row["physical_P_t"]),
        int(row["physical_H"]),
        int(row["physical_M1"]),
        int(row["physical_M2"]),
    ],
    axis=1,
)


# =============================================================================
# IBM CURRENT CALIBRATION SNAPSHOT
# =============================================================================

service = get_service()

backend_map = {
    name:
        service.backend(
            name,
            use_fractional_gates=False,
        )
    for name in BACKEND_NAMES
}


def target_instruction_error(
    target,
    gate_name,
    qargs,
):
    try:
        props = target[
            gate_name
        ][tuple(qargs)]

        if props is None:
            return np.nan

        value = getattr(
            props,
            "error",
            None,
        )

        return (
            np.nan
            if value is None
            else float(value)
        )

    except Exception:
        return np.nan


calibration_rows = []

for backend_name, backend in backend_map.items():
    properties = backend.properties(
        refresh=True
    )

    if properties is None:
        raise RuntimeError(
            f"{backend_name}: backend properties unavailable"
        )

    for _, emb in embeddings[
        embeddings[
            "backend"
        ]
        ==
        backend_name
    ].iterrows():
        topology = str(
            emb["candidate"]
        )

        layout = list(
            emb["layout"]
        )

        t1 = np.array(
            [
                float(properties.t1(q))
                for q in layout
            ],
            dtype=float,
        )

        t2 = np.array(
            [
                float(properties.t2(q))
                for q in layout
            ],
            dtype=float,
        )

        ro = np.array(
            [
                float(
                    properties.readout_error(q)
                )
                for q in layout
            ],
            dtype=float,
        )

        cz_errors = []

        for li, lj in TOPOLOGY_EDGES[
            topology
        ]:
            pi = layout[li]
            pj = layout[lj]

            err = target_instruction_error(
                backend.target,
                "cz",
                (pi, pj),
            )

            if not np.isfinite(err):
                err = target_instruction_error(
                    backend.target,
                    "cz",
                    (pj, pi),
                )

            if np.isfinite(err):
                cz_errors.append(err)

        calibration_rows.append({
            "backend":
                backend_name,
            "topology":
                topology,
            "embedding_rank":
                int(
                    emb["final_rank"]
                ),
            "layout":
                json.dumps(layout),
            "min_t1_us":
                float(np.min(t1) * 1e6),
            "mean_t1_us":
                float(np.mean(t1) * 1e6),
            "min_t2_us":
                float(np.min(t2) * 1e6),
            "mean_t2_us":
                float(np.mean(t2) * 1e6),
            "max_readout_error_percent":
                float(np.max(ro) * 100.0),
            "mean_readout_error_percent":
                float(np.mean(ro) * 100.0),
            "max_cz_error_percent":
                (
                    float(
                        np.max(cz_errors)
                        *
                        100.0
                    )
                    if cz_errors
                    else np.nan
                ),
            "mean_cz_error_percent":
                (
                    float(
                        np.mean(cz_errors)
                        *
                        100.0
                    )
                    if cz_errors
                    else np.nan
                ),
        })

calibration_df = pd.DataFrame(
    calibration_rows
)

calibration_df.to_csv(
    RESULTS
    /
    "09_03c_calibration_snapshot.csv",
    index=False,
)


# =============================================================================
# QISKIT CIRCUITS
# =============================================================================

def build_core_circuit(
    topology,
    hx,
    hy,
    J,
    r,
):
    qc = QuantumCircuit(
        6
    )

    # Fresh F4 injection for the first chronological row.
    for q in range(4):
        qc.ry(
            float(
                FIRST_INPUT_ANGLES[q]
            ),
            q,
        )

    for _ in range(int(r)):
        for i, j in TOPOLOGY_EDGES[
            topology
        ]:
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
            float(hx)
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
                float(hy)
                *
                DT
                /
                r
            )

            qc.ry(theta_y, 4)
            qc.ry(theta_y, 5)

    return qc


def build_measurement_circuit(
    core,
    setting,
):
    qc = QuantumCircuit(
        6,
        6,
    )

    qc.compose(
        core,
        qubits=range(6),
        inplace=True,
    )

    for q, basis in enumerate(setting):
        if basis == "X":
            qc.h(q)
        elif basis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif basis == "Z":
            pass
        else:
            raise ValueError(
                f"Unsupported basis: {basis}"
            )

    qc.measure(
        range(6),
        range(6),
    )

    return qc


def count_resources(circuit):
    counts = {
        str(k):
            int(v)
        for k, v
        in circuit.count_ops().items()
    }

    excluded = {
        "measure",
        "reset",
        "delay",
        "barrier",
    }

    n1 = 0
    n2 = 0

    for inst in circuit.data:
        nq = len(inst.qubits)
        name = inst.operation.name

        if (
            nq == 1
            and
            name not in excluded
        ):
            n1 += 1

        elif nq == 2:
            n2 += 1

    try:
        d1 = int(
            circuit.depth(
                filter_function=lambda inst:
                    (
                        len(inst.qubits) == 1
                        and
                        inst.operation.name
                        not in excluded
                    )
            )
        )

        d2 = int(
            circuit.depth(
                filter_function=lambda inst:
                    len(inst.qubits) == 2
            )
        )

    except TypeError:
        d1 = int(
            circuit.depth(
                filter_function=lambda inst:
                    (
                        len(inst[1]) == 1
                        and
                        inst[0].name
                        not in excluded
                    )
            )
        )

        d2 = int(
            circuit.depth(
                filter_function=lambda inst:
                    len(inst[1]) == 2
            )
        )

    return {
        "n_qubits":
            6,
        "depth":
            int(circuit.depth()),
        "depth_1q":
            d1,
        "depth_2q":
            d2,
        "size":
            int(circuit.size()),
        "n_1q":
            int(n1),
        "n_2q":
            int(n2),
        "n_cz":
            int(
                counts.get("cz", 0)
            ),
        "n_swap":
            int(
                counts.get("swap", 0)
            ),
        "n_measure":
            int(
                counts.get("measure", 0)
            ),
        "n_delay":
            int(
                counts.get("delay", 0)
            ),
        "operations":
            json.dumps(
                counts,
                sort_keys=True,
            ),
    }


def duration_seconds(
    circuit,
    backend,
):
    return float(
        circuit.estimate_duration(
            backend.target,
            unit="s",
        )
    )


# =============================================================================
# HARDWARE RESUME
# =============================================================================

HARDWARE_FILE = (
    RESULTS
    /
    "09_03c_hardware_all_levels.csv"
)

MEASUREMENT_FILE = (
    RESULTS
    /
    "09_03c_measurement_all_levels.csv"
)

if (
    RESUME_HARDWARE
    and HARDWARE_FILE.exists()
):
    hardware_rows = pd.read_csv(
        HARDWARE_FILE
    ).to_dict("records")
else:
    hardware_rows = []

if (
    RESUME_HARDWARE
    and MEASUREMENT_FILE.exists()
):
    measurement_rows = pd.read_csv(
        MEASUREMENT_FILE
    ).to_dict("records")
else:
    measurement_rows = []

completed_core = {
    (
        str(row["model_id"]),
        str(row["backend"]),
        int(row["embedding_rank"]),
        int(row["optimization_level"]),
    )
    for row in hardware_rows
}

completed_measurement = {
    (
        str(row["model_id"]),
        str(row["backend"]),
        int(row["embedding_rank"]),
        int(row["optimization_level"]),
        str(row["measurement_setting"]),
    )
    for row in measurement_rows
}


# =============================================================================
# COMPILE ALL 135 FORECAST-BEST BRANCH/R MODELS
# =============================================================================

print()
print("=" * 132)
print("IBM PHYSICAL COMPILATION OF ALL 135 FORECAST-BEST BRANCH/R MODELS")
print("=" * 132)

for _, model in forecast_best_df.iterrows():
    model_id = str(
        model["model_id"]
    )

    topology = str(
        model["topology"]
    )

    readout = str(
        model["readout"]
    )

    r = int(
        model["trotter_r"]
    )

    hx = float(
        model["hx"]
    )

    hy = float(
        model["hy"]
    )

    J = J_from_row(model)

    core = build_core_circuit(
        topology,
        hx,
        hy,
        J,
        r,
    )

    settings = READOUT_SETTINGS[
        readout
    ]

    for _, emb in embeddings[
        embeddings["candidate"]
        ==
        topology
    ].iterrows():
        backend_name = str(
            emb["backend"]
        )

        if backend_name not in backend_map:
            continue

        backend = backend_map[
            backend_name
        ]

        embedding_rank = int(
            emb["final_rank"]
        )

        layout = list(
            emb["layout"]
        )

        cal = calibration_df[
            (
                calibration_df["backend"]
                ==
                backend_name
            )
            &
            (
                calibration_df["topology"]
                ==
                topology
            )
            &
            (
                calibration_df[
                    "embedding_rank"
                ]
                ==
                embedding_rank
            )
        ]

        if len(cal) != 1:
            raise RuntimeError(
                "Calibration row not uniquely matched for "
                f"{backend_name}/{topology}/rank{embedding_rank}"
            )

        cal = cal.iloc[0]

        min_t1_us = float(
            cal["min_t1_us"]
        )

        min_t2_us = float(
            cal["min_t2_us"]
        )

        for level in OPT_LEVELS:
            core_key = (
                model_id,
                backend_name,
                embedding_rank,
                int(level),
            )

            transpile_kwargs = {
                "backend":
                    backend,
                "initial_layout":
                    layout,
                "routing_method":
                    "none",
                "optimization_level":
                    level,
                "seed_transpiler":
                    SEED_TRANSPILER,
                "scheduling_method":
                    SCHEDULING_METHOD,
            }

            if core_key not in completed_core:
                compiled_core = transpile(
                    core,
                    **transpile_kwargs,
                )

                resources = count_resources(
                    compiled_core
                )

                duration_s = duration_seconds(
                    compiled_core,
                    backend,
                )

                duration_us = (
                    duration_s * 1e6
                )

                if resources["n_swap"] != 0:
                    raise RuntimeError(
                        f"{model_id}/{backend_name}/"
                        f"rank{embedding_rank}/L{level}: "
                        f"unexpected SWAP={resources['n_swap']}"
                    )

                hardware_rows.append({
                    "model_id":
                        model_id,
                    "seed_candidate_id":
                        model["seed_candidate_id"],
                    "dynamics_seed_id":
                        model["dynamics_seed_id"],
                    "config_id":
                        model["config_id"],
                    "topology":
                        topology,
                    "seed_hy":
                        float(model["seed_hy"]),
                    "y_state":
                        model["y_state"],
                    "readout":
                        readout,
                    "trotter_r":
                        r,
                    "hx":
                        hx,
                    "hy":
                        hy,
                    **J_columns(
                        topology,
                        J,
                    ),
                    "selected_lambda":
                        float(
                            model[
                                "selected_lambda"
                            ]
                        ),
                    "cv_rmse":
                        float(
                            model["cv_rmse"]
                        ),
                    "validation_rmse":
                        float(
                            model[
                                "validation_rmse"
                            ]
                        ),
                    "process_fidelity":
                        float(
                            model[
                                "process_fidelity"
                            ]
                        ),
                    "process_infidelity":
                        float(
                            model[
                                "process_infidelity"
                            ]
                        ),
                    "backend":
                        backend_name,
                    "embedding_rank":
                        embedding_rank,
                    "layout":
                        json.dumps(layout),
                    "optimization_level":
                        int(level),
                    "is_primary_level":
                        level in PRIMARY_LEVELS,
                    "scheduling":
                        SCHEDULING_LABEL,
                    "duration_us":
                        duration_us,
                    "R_T1":
                        duration_us
                        /
                        min_t1_us,
                    "R_T2":
                        duration_us
                        /
                        min_t2_us,
                    "min_t1_us":
                        min_t1_us,
                    "min_t2_us":
                        min_t2_us,
                    "max_readout_error_percent":
                        float(
                            cal[
                                "max_readout_error_percent"
                            ]
                        ),
                    "max_cz_error_percent":
                        float(
                            cal[
                                "max_cz_error_percent"
                            ]
                        ),
                    **resources,
                })

                completed_core.add(
                    core_key
                )

                pd.DataFrame(
                    hardware_rows
                ).to_csv(
                    HARDWARE_FILE,
                    index=False,
                )

            for setting in settings:
                meas_key = (
                    model_id,
                    backend_name,
                    embedding_rank,
                    int(level),
                    setting,
                )

                if meas_key in completed_measurement:
                    continue

                measurement_circuit = (
                    build_measurement_circuit(
                        core,
                        setting,
                    )
                )

                compiled_measurement = transpile(
                    measurement_circuit,
                    **transpile_kwargs,
                )

                mresources = count_resources(
                    compiled_measurement
                )

                mduration_s = duration_seconds(
                    compiled_measurement,
                    backend,
                )

                mduration_us = (
                    mduration_s
                    *
                    1e6
                )

                measurement_rows.append({
                    "model_id":
                        model_id,
                    "seed_candidate_id":
                        model[
                            "seed_candidate_id"
                        ],
                    "config_id":
                        model["config_id"],
                    "trotter_r":
                        r,
                    "backend":
                        backend_name,
                    "embedding_rank":
                        embedding_rank,
                    "optimization_level":
                        int(level),
                    "is_primary_level":
                        level in PRIMARY_LEVELS,
                    "readout":
                        readout,
                    "measurement_setting":
                        setting,
                    "duration_per_shot_us":
                        mduration_us,
                    "duration_1024_shots_s":
                        mduration_s
                        *
                        SHOTS_PER_SETTING,
                    "R_T1":
                        mduration_us
                        /
                        min_t1_us,
                    "R_T2":
                        mduration_us
                        /
                        min_t2_us,
                    **mresources,
                })

                completed_measurement.add(
                    meas_key
                )

                pd.DataFrame(
                    measurement_rows
                ).to_csv(
                    MEASUREMENT_FILE,
                    index=False,
                )


hardware_df = pd.DataFrame(
    hardware_rows
).drop_duplicates(
    subset=[
        "model_id",
        "backend",
        "embedding_rank",
        "optimization_level",
    ],
    keep="last",
)

measurement_df = pd.DataFrame(
    measurement_rows
).drop_duplicates(
    subset=[
        "model_id",
        "backend",
        "embedding_rank",
        "optimization_level",
        "measurement_setting",
    ],
    keep="last",
)

hardware_df.to_csv(
    HARDWARE_FILE,
    index=False,
)

measurement_df.to_csv(
    MEASUREMENT_FILE,
    index=False,
)


# =============================================================================
# FEATURE-VECTOR MEASUREMENT COST
# =============================================================================

feature_vector_df = (
    measurement_df
    .groupby(
        [
            "model_id",
            "seed_candidate_id",
            "config_id",
            "trotter_r",
            "backend",
            "embedding_rank",
            "optimization_level",
            "is_primary_level",
            "readout",
        ],
        as_index=False,
    )
    .agg(
        n_settings=(
            "measurement_setting",
            "nunique",
        ),
        longest_setting_duration_us=(
            "duration_per_shot_us",
            "max",
        ),
        total_feature_vector_duration_us=(
            "duration_per_shot_us",
            "sum",
        ),
        estimated_1024shot_circuit_time_s=(
            "duration_1024_shots_s",
            "sum",
        ),
        total_cz_per_feature_vector=(
            "n_cz",
            "sum",
        ),
        max_swap_per_setting=(
            "n_swap",
            "max",
        ),
        max_measurement_R_T2=(
            "R_T2",
            "max",
        ),
    )
)

feature_vector_df.to_csv(
    RESULTS
    /
    "09_03c_feature_vector_timing_all.csv",
    index=False,
)


# =============================================================================
# PRIMARY HARDWARE SUMMARIES
# =============================================================================

hardware_primary = hardware_df[
    hardware_df[
        "optimization_level"
    ].isin(PRIMARY_LEVELS)
].copy()

feature_primary = feature_vector_df[
    feature_vector_df[
        "optimization_level"
    ].isin(PRIMARY_LEVELS)
].copy()


hardware_summary = (
    hardware_primary
    .groupby(
        [
            "model_id",
            "seed_candidate_id",
            "config_id",
            "trotter_r",
            "backend",
            "optimization_level",
        ],
        as_index=False,
    )
    .agg(
        n_embeddings=(
            "embedding_rank",
            "count",
        ),
        n_qubits=(
            "n_qubits",
            "first",
        ),
        cz_median=(
            "n_cz",
            "median",
        ),
        swap_median=(
            "n_swap",
            "median",
        ),
        oneq_median=(
            "n_1q",
            "median",
        ),
        twoq_median=(
            "n_2q",
            "median",
        ),
        depth_median=(
            "depth",
            "median",
        ),
        depth_1q_median=(
            "depth_1q",
            "median",
        ),
        depth_2q_median=(
            "depth_2q",
            "median",
        ),
        duration_us_median=(
            "duration_us",
            "median",
        ),
        R_T1_median=(
            "R_T1",
            "median",
        ),
        R_T2_median=(
            "R_T2",
            "median",
        ),
        R_T2_max=(
            "R_T2",
            "max",
        ),
        max_cz_error_percent=(
            "max_cz_error_percent",
            "max",
        ),
        max_readout_error_percent=(
            "max_readout_error_percent",
            "max",
        ),
        cv_rmse=(
            "cv_rmse",
            "first",
        ),
        validation_rmse=(
            "validation_rmse",
            "first",
        ),
        process_fidelity=(
            "process_fidelity",
            "first",
        ),
        process_infidelity=(
            "process_infidelity",
            "first",
        ),
    )
)

hardware_summary.to_csv(
    RESULTS
    /
    "09_03c_hardware_summary_primary.csv",
    index=False,
)


feature_summary = (
    feature_primary
    .groupby(
        [
            "model_id",
            "seed_candidate_id",
            "config_id",
            "trotter_r",
            "backend",
            "optimization_level",
            "readout",
        ],
        as_index=False,
    )
    .agg(
        n_embeddings=(
            "embedding_rank",
            "count",
        ),
        n_settings=(
            "n_settings",
            "first",
        ),
        longest_setting_duration_us_median=(
            "longest_setting_duration_us",
            "median",
        ),
        total_feature_vector_duration_us_median=(
            "total_feature_vector_duration_us",
            "median",
        ),
        estimated_1024shot_circuit_time_s_median=(
            "estimated_1024shot_circuit_time_s",
            "median",
        ),
        total_cz_per_feature_vector_median=(
            "total_cz_per_feature_vector",
            "median",
        ),
        max_measurement_R_T2_median=(
            "max_measurement_R_T2",
            "median",
        ),
    )
)

feature_summary.to_csv(
    RESULTS
    /
    "09_03c_feature_vector_summary_primary.csv",
    index=False,
)


# =============================================================================
# FINAL DECISION TABLE
# =============================================================================

decision = forecast_best_df.merge(
    memory_summary[
        [
            "config_id",
            "MC_ch_mean",
            "MC_ch_std",
            "MC_total_mean",
            "Tw_0.01_mean",
            "Tw_0.01_std",
            "D_final_mean",
            "MC_delay1_mean",
            "MC_delay2_mean",
            "MC_delay5_mean",
            "MC_delay10_mean",
            "MC_delay20_mean",
        ]
    ],
    on="config_id",
    how="left",
)

decision = decision.merge(
    hardware_summary,
    on=[
        "model_id",
        "seed_candidate_id",
        "config_id",
        "trotter_r",
    ],
    how="left",
    suffixes=(
        "",
        "_hardware",
    ),
)

decision = decision.merge(
    feature_summary,
    on=[
        "model_id",
        "seed_candidate_id",
        "config_id",
        "trotter_r",
        "backend",
        "optimization_level",
    ],
    how="left",
    suffixes=(
        "",
        "_measurement",
    ),
)

decision.to_csv(
    RESULTS
    /
    "09_03c_decision_table.csv",
    index=False,
)


# =============================================================================
# r COMPARISON SUMMARY
#
# For each original historical branch:
# compare its independently optimized r=1,2,3 forecast-best rows.
# No weighted score.
# =============================================================================

comparison_rows = []

for (
    seed_candidate_id,
    backend,
    level,
), group in decision.groupby(
    [
        "seed_candidate_id",
        "backend",
        "optimization_level",
    ],
    sort=True,
):
    for _, row in group.iterrows():
        comparison_rows.append({
            "seed_candidate_id":
                seed_candidate_id,
            "topology":
                row["topology"],
            "seed_hy":
                row["seed_hy"],
            "y_state":
                row["y_state"],
            "readout":
                row["readout"],
            "backend":
                backend,
            "optimization_level":
                level,
            "trotter_r":
                int(row["trotter_r"]),
            "hx":
                row["hx"],
            "hy":
                row["hy"],
            "cv_rmse":
                row["cv_rmse"],
            "validation_rmse":
                row["validation_rmse"],
            "process_fidelity":
                row["process_fidelity"],
            "process_infidelity":
                row["process_infidelity"],
            "MC_ch_mean":
                row["MC_ch_mean"],
            "Tw_0.01_mean":
                row["Tw_0.01_mean"],
            "cz_median":
                row["cz_median"],
            "swap_median":
                row["swap_median"],
            "depth_median":
                row["depth_median"],
            "depth_2q_median":
                row["depth_2q_median"],
            "duration_us_median":
                row["duration_us_median"],
            "R_T2_median":
                row["R_T2_median"],
            "n_settings":
                row["n_settings"],
            "estimated_1024shot_circuit_time_s_median":
                row[
                    "estimated_1024shot_circuit_time_s_median"
                ],
        })

r_comparison = pd.DataFrame(
    comparison_rows
)

r_comparison.to_csv(
    RESULTS
    /
    "09_03c_r1_r2_r3_comparison.csv",
    index=False,
)


# =============================================================================
# NON-WEIGHTED PARETO ACROSS r=1,2,3 FOR EACH HISTORICAL BRANCH
# =============================================================================

pareto_parts = []

for (
    seed_candidate_id,
    backend,
    level,
), group in r_comparison.groupby(
    [
        "seed_candidate_id",
        "backend",
        "optimization_level",
    ],
    sort=True,
):
    g = group.copy().reset_index(
        drop=True
    )

    g[
        "forecast_hardware_pareto"
    ] = pareto_mask(
        g,
        minimize_columns=[
            "cv_rmse",
            "process_infidelity",
            "duration_us_median",
            "cz_median",
            "R_T2_median",
        ],
    )

    finite_memory = (
        np.isfinite(
            pd.to_numeric(
                g["MC_ch_mean"],
                errors="coerce",
            )
        )
        &
        np.isfinite(
            pd.to_numeric(
                g["Tw_0.01_mean"],
                errors="coerce",
            )
        )
    )

    g[
        "memory_hardware_pareto"
    ] = False

    if finite_memory.any():
        sub = g[
            finite_memory
        ].copy()

        mask = pareto_mask(
            sub,
            minimize_columns=[
                "Tw_0.01_mean",
                "duration_us_median",
                "cz_median",
                "R_T2_median",
            ],
            maximize_columns=[
                "MC_ch_mean",
            ],
        )

        sub_idx = sub.index.to_numpy()

        g.loc[
            sub_idx[mask],
            "memory_hardware_pareto",
        ] = True

    pareto_parts.append(g)


r_pareto_all = pd.concat(
    pareto_parts,
    ignore_index=True,
)

r_pareto_all.to_csv(
    RESULTS
    /
    "09_03c_r_pareto_all.csv",
    index=False,
)

r_pareto_all[
    r_pareto_all[
        "forecast_hardware_pareto"
    ]
].to_csv(
    RESULTS
    /
    "09_03c_r_forecast_hardware_pareto.csv",
    index=False,
)

r_pareto_all[
    r_pareto_all[
        "memory_hardware_pareto"
    ]
].to_csv(
    RESULTS
    /
    "09_03c_r_memory_hardware_pareto.csv",
    index=False,
)


# =============================================================================
# MANIFEST
# =============================================================================

manifest = {
    "step":
        "Week 9 fair Trotter-depth re-optimization",
    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
    "historical_branch_count":
        int(
            len(candidate_universe)
        ),
    "dynamics_seed_count":
        int(
            candidate_universe[
                "dynamics_seed_id"
            ].nunique()
        ),
    "r_values":
        R_VALUES,
    "fairness_rule":
        (
            "r=1, r=2 and r=3 receive the same h_x/h_y/J search budget, "
            "Ridge grid, CV procedure, fidelity calculation, memory benchmark "
            "and hardware/resource profiling."
        ),
    "candidate_universe":
        {
            "H0":
                HY_SEEDS["H0"],
            "H1":
                HY_SEEDS["H1"],
            "H2":
                HY_SEEDS["H2"],
            "H3":
                HY_SEEDS["H3"],
            "H4":
                HY_SEEDS["H4"],
            "readouts":
                READOUTS,
        },
    "search":
        {
            "hx_grid":
                HX_GRID,
            "J_scale_grid":
                J_SCALE_GRID,
            "YON_hy_rule":
                "{0.5*h0, h0, 1.5*h0, -h0}",
            "YOFF_hy":
                0.0,
            "local_children_per_parent":
                LOCAL_CHILDREN_PER_PARENT,
            "local_parent_union":
                (
                    "best CV parent for each readout plus best "
                    "process-fidelity parent"
                ),
            "hx_bounds":
                HX_BOUNDS,
            "hy_bounds":
                HY_BOUNDS,
            "J_bounds":
                J_BOUNDS,
            "weighted_score":
                False,
        },
    "forecast":
        {
            "ridge_grid":
                [
                    float(x)
                    for x in RIDGE_GRID
                ],
            "cv_folds":
                N_CV_SPLITS,
            "validation":
                "2025 diagnostic only",
            "test_2026_used":
                False,
        },
    "memory":
        {
            "probe_seeds":
                PROBE_SEEDS,
            "observable":
                "XYZ_all",
            "K_max":
                K_MAX,
            "postwash_burnin":
                "max(100, Tw(0.01)+20)",
            "null_floor":
                "1/(365-1)",
        },
    "hardware":
        {
            "backends":
                BACKEND_NAMES,
            "embeddings_per_topology":
                3,
            "optimization_levels":
                OPT_LEVELS,
            "primary_levels":
                PRIMARY_LEVELS,
            "routing_method":
                "none",
            "scheduling":
                SCHEDULING_LABEL,
            "shots_per_setting":
                SHOTS_PER_SETTING,
            "R_T1":
                "scheduled duration / min T1 of embedding",
            "R_T2":
                "scheduled duration / min T2 of embedding",
            "coherence_ratio_warning":
                "R_T1 and R_T2 are exposure ratios, not error probabilities.",
        },
    "expensive_full_diagnostics":
        (
            "forecast-best row for every historical branch and r; "
            "fidelity-best and full CV/fidelity Pareto saved separately"
        ),
}

with open(
    RESULTS / "09_03c_manifest.json",
    "w",
    encoding="utf-8",
) as fp:
    json.dump(
        manifest,
        fp,
        indent=2,
    )


# =============================================================================
# COMPACT CONSOLE SUMMARY
# =============================================================================

print()
print("=" * 132)
print("SEARCH COMPLETE")
print("=" * 132)
print(
    f"search rows: "
    f"{len(search_df)}"
)
print(
    f"forecast-best rows: "
    f"{len(forecast_best_df)} "
    f"(expected 45*3 = 135)"
)
print(
    f"fidelity-best rows: "
    f"{len(fidelity_best_df)}"
)
print()

show = (
    forecast_best_df[
        [
            "seed_candidate_id",
            "trotter_r",
            "hx",
            "hy",
            "selected_lambda",
            "cv_rmse",
            "validation_rmse",
            "process_fidelity",
        ]
    ]
    .sort_values(
        [
            "seed_candidate_id",
            "trotter_r",
        ]
    )
)

print(
    show.to_string(
        index=False
    )
)

print()
print("Key files to send for interpretation:")
print(
    "  results/09_03c_candidate_universe_45.csv"
)
print(
    "  results/09_03c_forecast_best_135.csv"
)
print(
    "  results/09_03c_fidelity_best_135.csv"
)
print(
    "  results/09_03c_memory_summary.csv"
)
print(
    "  results/09_03c_hardware_summary_primary.csv"
)
print(
    "  results/09_03c_feature_vector_summary_primary.csv"
)
print(
    "  results/09_03c_r1_r2_r3_comparison.csv"
)
print(
    "  results/09_03c_r_pareto_all.csv"
)
print(
    "  results/09_03c_decision_table.csv"
)
print(
    "  results/09_03c_manifest.json"
)

print()
print(
    f"Total runtime: "
    f"{time.perf_counter() - global_start:.1f}s"
)
print(
    "Do not freeze r yet. Send the CSVs first; "
    "the decision should be made from the joint "
    "forecast/fidelity/memory/hardware evidence."
)
