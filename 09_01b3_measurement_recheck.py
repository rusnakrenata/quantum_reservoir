from pathlib import Path
import json
import math
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Week 9 — Step 9.1B.3
# Compact measurement/readout recheck after adding the memory-only Y field
#
# Readout families:
#   1. XZ_injection
#   2. XZ_injection + YX45
#   3. XYZ_all
#
# The reservoir Hamiltonian, topology, J values, alpha, hx, dt, r, and CONT
# propagation remain frozen. Only the readout layer is compared here.
#
# No randomized MC probes are required in this step because forecasting uses the
# real chronological insurance sequence.
# =============================================================================

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DATA_FILE = RESULTS / "03_01_preprocessed_samples.csv"
J_WINNER_FILE = RESULTS / "08_03b_J_search_topology_winners.csv"
TUNING_FORECAST_FILE = RESULTS / "09_01b2_hy_forecast_summary.csv"

for path in [DATA_FILE, J_WINNER_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# -----------------------------------------------------------------------------
# Frozen QRC parameters
# -----------------------------------------------------------------------------

N_QUBITS = 6
N_INPUT = 4
D_INPUT = 16
D_MEMORY = 4
D_GLOBAL = 64

ALPHA = 0.75
HX = 0.5
DT = 1.6
TROTTER_R = 2

FORECAST_RIDGE_FIXED = 0.01

RIDGE_GRID = np.array(
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0],
    dtype=float,
)

N_CV_SPLITS = 5
VAR_TOL = 1e-12

# -----------------------------------------------------------------------------
# Compact set retained from Step 9.1B.2
# -----------------------------------------------------------------------------

CANDIDATE_HY = {
    "H0": [-0.4, 0.0],
    "H1": [-0.6, 0.0],
    "H2": [0.3, 0.4, 0.6],
    "H3": [-0.2, 0.0, 0.3],
    "H4": [-0.6, 0.4, 0.6],
}

ACTIVE_TOPOLOGIES = list(CANDIDATE_HY.keys())

# -----------------------------------------------------------------------------
# Measurement/readout families
#
# Cost metadata refers to ideal grouped direct measurement plans.
#
# XZ_injection:
#   XXXXZZ
#   ZZZZZZ
#
# XZ_injection + YX45:
#   XXXXYX
#   ZZZZZZ
#
# XYZ_all:
#   XXXXXX
#   YYYYYY
#   ZZZZZZ
#
# For Y measurement, Sdg + H are counted as two basis-change gates.
# -----------------------------------------------------------------------------

READOUT_METADATA = {
    "XZ_injection": {
        "n_features": 8,
        "n_settings": 2,
        "basis_change_gates": 4,
        "ancilla": 0,
        "settings": ["XXXXZZ", "ZZZZZZ"],
    },
    "XZinj_plus_YX45": {
        "n_features": 9,
        "n_settings": 2,
        "basis_change_gates": 7,
        "ancilla": 0,
        "settings": ["XXXXYX", "ZZZZZZ"],
    },
    "XYZ_all": {
        "n_features": 18,
        "n_settings": 3,
        "basis_change_gates": 18,
        "ancilla": 0,
        "settings": ["XXXXXX", "YYYYYY", "ZZZZZZ"],
    },
}

READOUTS = list(READOUT_METADATA.keys())

# =============================================================================
# Input encoding
# =============================================================================

def claim_mapping(z):
    return float(np.clip(float(z) / 3.0, -1.0, 1.0))


def policy_mapping(z):
    z = float(z)
    return z / (3.0 + abs(z))


def weekday_mapping(d_sin, d_cos):
    return math.atan2(float(d_sin), float(d_cos)) % (2.0 * math.pi)


def holiday_mapping(h):
    h = int(h)
    if h not in (0, 1):
        raise ValueError(f"Holiday must be 0/1, got {h}")
    return math.pi * h


def encode_f4_row(row):
    return np.array(
        [
            ALPHA * claim_mapping(row["C_t_z"]),
            weekday_mapping(row["D_sin"], row["D_cos"]),
            ALPHA * policy_mapping(row["P_t_z"]),
            holiday_mapping(row["is_public_holiday_t_plus_1"]),
        ],
        dtype=float,
    )

# =============================================================================
# Data
# =============================================================================

df = pd.read_csv(DATA_FILE)

required_columns = [
    "input_date",
    "target_date",
    "split",
    "C_t_z",
    "D_sin",
    "D_cos",
    "P_t_z",
    "is_public_holiday_t_plus_1",
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise KeyError(
        f"Missing preprocessing columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

df["input_date"] = pd.to_datetime(df["input_date"])
df["target_date"] = pd.to_datetime(df["target_date"])
df = df.sort_values("target_date").reset_index(drop=True)


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_target_column(frame):
    normalized = {normalize_name(c): c for c in frame.columns}

    candidates = [
        "target",
        "y",
        "C_t_plus_1",
        "claims_t_plus_1",
        "claim_count_t_plus_1",
        "property_damage_claim_count_t_plus_1",
        "target_claim_count",
        "target_property_damage",
        "target_property_damage_claim_count",
    ]

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            column = normalized[key]
            if pd.api.types.is_numeric_dtype(frame[column]):
                return column

    target_like = [
        c
        for c in frame.columns
        if "target" in normalize_name(c)
        and "date" not in normalize_name(c)
        and pd.api.types.is_numeric_dtype(frame[c])
    ]

    if len(target_like) == 1:
        return target_like[0]

    raise KeyError(
        "Could not determine the next-day property-damage target column.\n"
        f"Available columns: {list(frame.columns)}"
    )


TARGET_COLUMN = find_target_column(df)

split_norm = df["split"].astype(str).str.strip().str.lower()
train_mask_all = split_norm.eq("train")
validation_mask_all = split_norm.isin(["validation", "val"])

work_df = (
    df[train_mask_all | validation_mask_all]
    .copy()
    .reset_index(drop=True)
)

work_split = work_df["split"].astype(str).str.strip().str.lower()
train_mask = work_split.eq("train").to_numpy()
validation_mask = work_split.isin(["validation", "val"]).to_numpy()

N_TRAIN = int(train_mask.sum())
N_VALIDATION = int(validation_mask.sum())
N_TOTAL = len(work_df)

if (N_TRAIN, N_VALIDATION, N_TOTAL) != (1095, 365, 1460):
    raise RuntimeError(
        "Expected train/validation/total = 1095/365/1460; got "
        f"{N_TRAIN}/{N_VALIDATION}/{N_TOTAL}"
    )

if not np.all(train_mask[:N_TRAIN]) or not np.all(validation_mask[N_TRAIN:]):
    raise RuntimeError("Train/validation chronology is not the expected contiguous split.")

Y_TARGET = work_df[TARGET_COLUMN].to_numpy(dtype=float)
REAL_ANGLES = np.vstack([encode_f4_row(row) for _, row in work_df.iterrows()])

# =============================================================================
# Topologies and Week-8 J winners
# =============================================================================

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

j_df = pd.read_csv(J_WINNER_FILE)
COUPLINGS = {}

for topology in ACTIVE_TOPOLOGIES:
    matches = j_df[j_df["topology"] == topology]

    if len(matches) != 1:
        raise RuntimeError(
            f"{topology}: expected exactly one Week-8 J winner; found {len(matches)}"
        )

    winner = matches.iloc[0]

    if not np.isclose(float(winner["alpha"]), ALPHA):
        raise RuntimeError(f"{topology}: alpha mismatch")
    if not np.isclose(float(winner["hx"]), HX):
        raise RuntimeError(f"{topology}: hx mismatch")
    if not np.isclose(float(winner["dt"]), DT):
        raise RuntimeError(f"{topology}: dt mismatch")
    if int(winner["Trotter_r"]) != TROTTER_R:
        raise RuntimeError(f"{topology}: Trotter-r mismatch")

    COUPLINGS[topology] = {}

    for edge, column in EDGE_TO_COLUMN[topology].items():
        value = winner[column]

        if pd.isna(value):
            raise RuntimeError(f"{topology}: missing {column}")

        COUPLINGS[topology][edge] = float(value)

# =============================================================================
# Operators and Trotter unitary
# =============================================================================

I2 = np.eye(2, dtype=complex)

X2 = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=complex,
)

Y2 = np.array(
    [
        [0, -1j],
        [1j, 0],
    ],
    dtype=complex,
)

Z2 = np.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=complex,
)

I64 = np.eye(D_GLOBAL, dtype=complex)


def operator_on_qubit(op, q):
    result = np.array([[1.0 + 0j]], dtype=complex)

    for logical_q in reversed(range(N_QUBITS)):
        result = np.kron(
            result,
            op if logical_q == q else I2,
        )

    return result


X_OPS = [operator_on_qubit(X2, q) for q in range(N_QUBITS)]
Y_OPS = [operator_on_qubit(Y2, q) for q in range(N_QUBITS)]
Z_OPS = [operator_on_qubit(Z2, q) for q in range(N_QUBITS)]

ZZ_OPS = {}

for topology in ACTIVE_TOPOLOGIES:
    for edge in TOPOLOGY_EDGES[topology]:
        canonical = tuple(sorted(edge))

        if canonical not in ZZ_OPS:
            ZZ_OPS[canonical] = (
                Z_OPS[canonical[0]]
                @
                Z_OPS[canonical[1]]
            )

# Joint memory observable:
#
#     <Y4 X5> = Tr[rho (Y4 tensor X5)]
#
# Y4 and X5 act on different qubits and therefore commute.
YX45_OP = Y_OPS[4] @ X_OPS[5]


def pauli_exponential(op, angle):
    return (
        math.cos(angle) * I64
        -
        1j * math.sin(angle) * op
    )


def build_unitary(topology, hy):
    U = np.eye(D_GLOBAL, dtype=complex)

    for _ in range(TROTTER_R):

        # ZZ sector
        for edge in TOPOLOGY_EDGES[topology]:
            Jij = COUPLINGS[topology][edge]

            a_zz = (
                Jij
                * DT
                / TROTTER_R
            )

            U = (
                pauli_exponential(
                    ZZ_OPS[tuple(sorted(edge))],
                    a_zz,
                )
                @
                U
            )

        # X field on all 6 qubits
        a_x = (
            HX
            * DT
            / TROTTER_R
        )

        for q in range(N_QUBITS):
            U = (
                pauli_exponential(
                    X_OPS[q],
                    a_x,
                )
                @
                U
            )

        # Memory-only Y field on q4,q5
        if abs(hy) > 0.0:
            a_y = (
                hy
                * DT
                / TROTTER_R
            )

            for q in (4, 5):
                U = (
                    pauli_exponential(
                        Y_OPS[q],
                        a_y,
                    )
                    @
                    U
                )

    unitary_error = float(
        np.linalg.norm(
            U.conj().T @ U - I64,
            ord="fro",
        )
    )

    return U, unitary_error

# =============================================================================
# Input states and CONT propagation
# =============================================================================

def ry_zero_state(theta):
    return np.array(
        [
            math.cos(theta / 2.0),
            math.sin(theta / 2.0),
        ],
        dtype=complex,
    )


def input_state_vector(angles):
    q0 = ry_zero_state(angles[0])
    q1 = ry_zero_state(angles[1])
    q2 = ry_zero_state(angles[2])
    q3 = ry_zero_state(angles[3])

    # q3 q2 q1 q0
    return np.kron(
        np.kron(
            np.kron(q3, q2),
            q1,
        ),
        q0,
    )


REAL_PHI = [
    input_state_vector(angles)
    for angles in REAL_ANGLES
]

I4 = np.eye(D_MEMORY, dtype=complex)

KET_00 = np.array(
    [1, 0, 0, 0],
    dtype=complex,
)

RHO_00 = np.outer(
    KET_00,
    KET_00.conj(),
)


def isometry_for_input(U, phi):
    embedding = np.kron(
        I4,
        np.asarray(
            phi,
            dtype=complex,
        ).reshape(-1, 1),
    )

    return U @ embedding


def partial_trace_input(rho_global):
    reshaped = rho_global.reshape(
        D_MEMORY,
        D_INPUT,
        D_MEMORY,
        D_INPUT,
    )

    rho_memory = np.einsum(
        "aibi->ab",
        reshaped,
    )

    rho_memory = 0.5 * (
        rho_memory
        +
        rho_memory.conj().T
    )

    rho_memory /= np.trace(
        rho_memory
    )

    return rho_memory


BASIS_INDICES = np.arange(
    D_GLOBAL
)


def xyz_expectations(rho_global):
    diagonal = np.real(
        np.diag(
            rho_global
        )
    )

    x_values = np.zeros(
        N_QUBITS,
        dtype=float,
    )

    y_values = np.zeros(
        N_QUBITS,
        dtype=float,
    )

    z_values = np.zeros(
        N_QUBITS,
        dtype=float,
    )

    for q in range(N_QUBITS):
        mask = (
            1
            <<
            q
        )

        flipped = (
            BASIS_INDICES
            ^
            mask
        )

        offdiag = (
            rho_global[
                BASIS_INDICES,
                flipped,
            ]
        )

        x_values[q] = float(
            np.real(
                np.sum(
                    offdiag
                )
            )
        )

        bits = (
            BASIS_INDICES
            >>
            q
        ) & 1

        y_coeff = np.where(
            bits == 0,
            1j,
            -1j,
        )

        y_values[q] = float(
            np.real(
                np.sum(
                    offdiag
                    *
                    y_coeff
                )
            )
        )

        z_sign = (
            1.0
            -
            2.0
            *
            bits
        )

        z_values[q] = float(
            np.sum(
                diagonal
                *
                z_sign
            )
        )

    return (
        x_values,
        y_values,
        z_values,
    )


def expectation(
    rho,
    operator,
):
    return float(
        np.real(
            np.trace(
                rho
                @
                operator
            )
        )
    )


def simulate_cont_readouts(
    U,
    phi_sequence,
):

    n = len(
        phi_sequence
    )

    XZ_injection = np.zeros(
        (
            n,
            8,
        ),
        dtype=float,
    )

    XZinj_plus_YX45 = np.zeros(
        (
            n,
            9,
        ),
        dtype=float,
    )

    XYZ_all = np.zeros(
        (
            n,
            18,
        ),
        dtype=float,
    )

    rho_memory = RHO_00.copy()

    max_trace_error = 0.0
    max_hermiticity_error = 0.0
    min_memory_eigenvalue = float("inf")

    yx45_abs_max = 0.0
    yx45_std_values = np.zeros(
        n,
        dtype=float,
    )

    for t, phi in enumerate(
        phi_sequence
    ):

        V = isometry_for_input(
            U,
            phi,
        )

        rho_global = (
            V
            @
            rho_memory
            @
            V.conj().T
        )

        (
            x_values,
            y_values,
            z_values,
        ) = xyz_expectations(
            rho_global
        )

        yx45 = expectation(
            rho_global,
            YX45_OP,
        )

        yx45_std_values[t] = yx45
        yx45_abs_max = max(
            yx45_abs_max,
            abs(
                yx45
            ),
        )

        xzinj = np.concatenate(
            [
                x_values[:4],
                z_values[:4],
            ]
        )

        xyz = np.concatenate(
            [
                x_values,
                y_values,
                z_values,
            ]
        )

        XZ_injection[t] = xzinj

        XZinj_plus_YX45[t] = np.concatenate(
            [
                xzinj,
                [
                    yx45
                ],
            ]
        )

        XYZ_all[t] = xyz

        rho_memory = partial_trace_input(
            rho_global
        )

        max_trace_error = max(
            max_trace_error,
            float(
                abs(
                    np.trace(
                        rho_memory
                    )
                    -
                    1.0
                )
            ),
        )

        max_hermiticity_error = max(
            max_hermiticity_error,
            float(
                np.linalg.norm(
                    rho_memory
                    -
                    rho_memory.conj().T,
                    ord="fro",
                )
            ),
        )

        min_memory_eigenvalue = min(
            min_memory_eigenvalue,
            float(
                np.min(
                    np.linalg.eigvalsh(
                        rho_memory
                    )
                )
            ),
        )

    features = {
        "XZ_injection":
            XZ_injection,

        "XZinj_plus_YX45":
            XZinj_plus_YX45,

        "XYZ_all":
            XYZ_all,
    }

    diagnostics = {
        "max_trace_error":
            max_trace_error,

        "max_hermiticity_error":
            max_hermiticity_error,

        "min_memory_eigenvalue":
            min_memory_eigenvalue,

        "final_memory_purity":
            float(
                np.real(
                    np.trace(
                        rho_memory
                        @
                        rho_memory
                    )
                )
            ),

        "YX45_train_std":
            float(
                np.std(
                    yx45_std_values[
                        :N_TRAIN
                    ],
                    ddof=0,
                )
            ),

        "YX45_all_std":
            float(
                np.std(
                    yx45_std_values,
                    ddof=0,
                )
            ),

        "YX45_max_abs":
            yx45_abs_max,
    }

    return (
        features,
        diagnostics,
    )

# =============================================================================
# Ridge forecasting
# =============================================================================

def rmse(
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
                ** 2
            )
        )
    )


def fit_feature_transform(
    X_train,
):
    X_train = np.asarray(
        X_train,
        dtype=float,
    )

    std = np.std(
        X_train,
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
        return (
            None,
            None,
            None,
        )

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train[
                :,
                keep
            ]
        )
    )

    return (
        keep,
        scaler,
        X_train_scaled,
    )


def chronological_cv_for_lambda(
    X_train,
    y_train,
    ridge_alpha,
):
    splitter = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )

    fold_rmse = []

    for (
        tr,
        va,
    ) in splitter.split(
        X_train
    ):

        (
            keep,
            scaler,
            Xtr_z,
        ) = fit_feature_transform(
            X_train[
                tr
            ]
        )

        if keep is None:
            pred = np.full(
                len(
                    va
                ),
                float(
                    np.mean(
                        y_train[
                            tr
                        ]
                    )
                ),
            )

        else:
            Xva_z = scaler.transform(
                X_train[
                    va
                ][
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
                Xtr_z,
                y_train[
                    tr
                ],
            )

            pred = model.predict(
                Xva_z
            )

        fold_rmse.append(
            rmse(
                y_train[
                    va
                ],
                pred,
            )
        )

    return {
        "cv_rmse_mean":
            float(
                np.mean(
                    fold_rmse
                )
            ),

        "cv_rmse_std":
            float(
                np.std(
                    fold_rmse,
                    ddof=1,
                )
            ),

        "fold_rmse":
            [
                float(
                    value
                )
                for value in fold_rmse
            ],
    }


def final_fit_validation(
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

    X_train = X[
        :N_TRAIN
    ]

    y_train = y[
        :N_TRAIN
    ]

    X_val = X[
        N_TRAIN:
    ]

    y_val = y[
        N_TRAIN:
    ]

    (
        keep,
        scaler,
        Xtr_z,
    ) = fit_feature_transform(
        X_train
    )

    if keep is None:
        pred = np.full(
            len(
                X_val
            ),
            float(
                np.mean(
                    y_train
                )
            ),
        )

        n_retained = 0
        rank = 0

    else:
        Xval_z = scaler.transform(
            X_val[
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
            Xtr_z,
            y_train,
        )

        pred = model.predict(
            Xval_z
        )

        n_retained = int(
            keep.sum()
        )

        rank = int(
            np.linalg.matrix_rank(
                Xtr_z
            )
        )

    return {
        "validation_rmse":
            rmse(
                y_val,
                pred,
            ),

        "validation_bias":
            float(
                np.mean(
                    pred
                    -
                    y_val
                )
            ),

        "n_features_retained":
            n_retained,

        "feature_rank":
            rank,
    }


def evaluate_readout(
    X,
    y,
):
    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    X_train = X[
        :N_TRAIN
    ]

    y_train = y[
        :N_TRAIN
    ]

    # -------------------------------------------------------------------------
    # Fixed lambda = 0.01 audit, matching the Y-field tuning stage.
    # -------------------------------------------------------------------------

    fixed_cv = chronological_cv_for_lambda(
        X_train,
        y_train,
        FORECAST_RIDGE_FIXED,
    )

    fixed_val = final_fit_validation(
        X,
        y,
        FORECAST_RIDGE_FIXED,
    )

    # -------------------------------------------------------------------------
    # Readout-specific training-only lambda selection.
    # -------------------------------------------------------------------------

    grid_rows = []

    for ridge_alpha in RIDGE_GRID:
        cv = chronological_cv_for_lambda(
            X_train,
            y_train,
            ridge_alpha,
        )

        grid_rows.append(
            {
                "ridge_alpha":
                    float(
                        ridge_alpha
                    ),

                "cv_rmse_mean":
                    cv[
                        "cv_rmse_mean"
                    ],

                "cv_rmse_std":
                    cv[
                        "cv_rmse_std"
                    ],

                "fold_rmse":
                    cv[
                        "fold_rmse"
                    ],
            }
        )

    grid_df = pd.DataFrame(
        grid_rows
    )

    # Deterministic tie-break:
    # lower CV RMSE first, then smaller lambda.
    grid_df = grid_df.sort_values(
        [
            "cv_rmse_mean",
            "ridge_alpha",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    best = grid_df.iloc[0]

    selected_alpha = float(
        best[
            "ridge_alpha"
        ]
    )

    selected_val = final_fit_validation(
        X,
        y,
        selected_alpha,
    )

    return {
        "fixed_lambda":
            FORECAST_RIDGE_FIXED,

        "fixed_cv_rmse_mean":
            fixed_cv[
                "cv_rmse_mean"
            ],

        "fixed_cv_rmse_std":
            fixed_cv[
                "cv_rmse_std"
            ],

        "fixed_validation_rmse":
            fixed_val[
                "validation_rmse"
            ],

        "fixed_validation_bias":
            fixed_val[
                "validation_bias"
            ],

        "selected_lambda":
            selected_alpha,

        "selected_cv_rmse_mean":
            float(
                best[
                    "cv_rmse_mean"
                ]
            ),

        "selected_cv_rmse_std":
            float(
                best[
                    "cv_rmse_std"
                ]
            ),

        "selected_fold_rmse":
            best[
                "fold_rmse"
            ],

        "selected_validation_rmse":
            selected_val[
                "validation_rmse"
            ],

        "selected_validation_bias":
            selected_val[
                "validation_bias"
            ],

        "n_features_retained":
            selected_val[
                "n_features_retained"
            ],

        "feature_rank":
            selected_val[
                "feature_rank"
            ],

        "grid":
            grid_df,
    }

# =============================================================================
# Run experiment
# =============================================================================

print("=" * 118)
print("WEEK 9 — STEP 9.1B.3")
print("COMPACT MEASUREMENT / READOUT RECHECK AFTER MEMORY-ONLY Y FIELD")
print("=" * 118)

print(f"Target column: {TARGET_COLUMN}")
print(f"Train rows: {N_TRAIN}")
print(f"Validation rows: {N_VALIDATION}")
print("Test rows used: 0")
print(f"Ridge grid: {RIDGE_GRID.tolist()}")

print("\nCandidate Hamiltonians:")
for topology in ACTIVE_TOPOLOGIES:
    print(
        f"  {topology}: "
        f"h_y={CANDIDATE_HY[topology]}"
    )

print("\nReadouts:")
for family in READOUTS:
    metadata = READOUT_METADATA[
        family
    ]

    print(
        f"  {family:22s} "
        f"d={metadata['n_features']:2d} "
        f"settings={metadata['n_settings']} "
        f"basis_gates={metadata['basis_change_gates']} "
        f"ancilla={metadata['ancilla']} "
        f"{metadata['settings']}"
    )

result_rows = []
grid_rows = []
diagnostic_rows = []

experiment_start = time.time()

for topology in ACTIVE_TOPOLOGIES:

    print(
        "\n"
        +
        "-" * 118
    )

    print(
        topology
    )

    print(
        "-" * 118
    )

    for hy in CANDIDATE_HY[
        topology
    ]:

        start = time.time()

        U, unitary_error = build_unitary(
            topology,
            hy,
        )

        (
            feature_map,
            diagnostics,
        ) = simulate_cont_readouts(
            U,
            REAL_PHI,
        )

        diagnostic_rows.append(
            {
                "topology":
                    topology,

                "hy":
                    hy,

                "unitary_error_fro":
                    unitary_error,

                **diagnostics,
            }
        )

        family_results = {}

        for family in READOUTS:

            result = evaluate_readout(
                feature_map[
                    family
                ],
                Y_TARGET,
            )

            family_results[
                family
            ] = result

            metadata = READOUT_METADATA[
                family
            ]

            result_rows.append(
                {
                    "topology":
                        topology,

                    "hy":
                        hy,

                    "readout":
                        family,

                    "n_features":
                        metadata[
                            "n_features"
                        ],

                    "n_settings":
                        metadata[
                            "n_settings"
                        ],

                    "basis_change_gates":
                        metadata[
                            "basis_change_gates"
                        ],

                    "ancilla":
                        metadata[
                            "ancilla"
                        ],

                    "settings":
                        json.dumps(
                            metadata[
                                "settings"
                            ]
                        ),

                    "fixed_lambda":
                        result[
                            "fixed_lambda"
                        ],

                    "fixed_cv_rmse":
                        result[
                            "fixed_cv_rmse_mean"
                        ],

                    "fixed_cv_std":
                        result[
                            "fixed_cv_rmse_std"
                        ],

                    "fixed_validation_rmse":
                        result[
                            "fixed_validation_rmse"
                        ],

                    "fixed_validation_bias":
                        result[
                            "fixed_validation_bias"
                        ],

                    "selected_lambda":
                        result[
                            "selected_lambda"
                        ],

                    "cv_rmse":
                        result[
                            "selected_cv_rmse_mean"
                        ],

                    "cv_rmse_std":
                        result[
                            "selected_cv_rmse_std"
                        ],

                    "validation_rmse":
                        result[
                            "selected_validation_rmse"
                        ],

                    "validation_bias":
                        result[
                            "selected_validation_bias"
                        ],

                    "n_features_retained":
                        result[
                            "n_features_retained"
                        ],

                    "feature_rank":
                        result[
                            "feature_rank"
                        ],

                    "fold_rmse":
                        json.dumps(
                            result[
                                "selected_fold_rmse"
                            ]
                        ),

                    "unitary_error_fro":
                        unitary_error,

                    "YX45_train_std":
                        diagnostics[
                            "YX45_train_std"
                        ],
                }
            )

            grid_df = result[
                "grid"
            ].copy()

            grid_df[
                "topology"
            ] = topology

            grid_df[
                "hy"
            ] = hy

            grid_df[
                "readout"
            ] = family

            grid_df[
                "fold_rmse"
            ] = grid_df[
                "fold_rmse"
            ].apply(
                json.dumps
            )

            grid_rows.append(
                grid_df
            )

        xz = family_results[
            "XZ_injection"
        ]

        print(
            f"h_y={hy:+.2f} | "
            f"XZ={family_results['XZ_injection']['selected_cv_rmse_mean']:.6f} "
            f"(lam={family_results['XZ_injection']['selected_lambda']:g}) | "
            f"XZ+YX45={family_results['XZinj_plus_YX45']['selected_cv_rmse_mean']:.6f} "
            f"(lam={family_results['XZinj_plus_YX45']['selected_lambda']:g}) | "
            f"XYZ={family_results['XYZ_all']['selected_cv_rmse_mean']:.6f} "
            f"(lam={family_results['XYZ_all']['selected_lambda']:g}) | "
            f"YX45 std={diagnostics['YX45_train_std']:.6f} | "
            f"time={time.time() - start:.1f}s"
        )

# =============================================================================
# Result tables
# =============================================================================

results_df = pd.DataFrame(
    result_rows
)

grid_df = pd.concat(
    grid_rows,
    ignore_index=True,
)

diagnostics_df = pd.DataFrame(
    diagnostic_rows
)

# -----------------------------------------------------------------------------
# Readout deltas against XZ_injection at the SAME topology and h_y
# -----------------------------------------------------------------------------

xz_reference = (
    results_df[
        results_df[
            "readout"
        ] == "XZ_injection"
    ][
        [
            "topology",
            "hy",
            "cv_rmse",
            "validation_rmse",
            "fixed_cv_rmse",
            "fixed_validation_rmse",
        ]
    ]
    .rename(
        columns={
            "cv_rmse":
                "XZ_reference_cv_rmse",

            "validation_rmse":
                "XZ_reference_validation_rmse",

            "fixed_cv_rmse":
                "XZ_reference_fixed_cv_rmse",

            "fixed_validation_rmse":
                "XZ_reference_fixed_validation_rmse",
        }
    )
)

comparison_df = results_df.merge(
    xz_reference,
    on=[
        "topology",
        "hy",
    ],
    how="left",
)

comparison_df[
    "delta_cv_vs_XZ"
] = (
    comparison_df[
        "cv_rmse"
    ]
    -
    comparison_df[
        "XZ_reference_cv_rmse"
    ]
)

comparison_df[
    "delta_validation_vs_XZ"
] = (
    comparison_df[
        "validation_rmse"
    ]
    -
    comparison_df[
        "XZ_reference_validation_rmse"
    ]
)

comparison_df[
    "delta_fixed_cv_vs_XZ"
] = (
    comparison_df[
        "fixed_cv_rmse"
    ]
    -
    comparison_df[
        "XZ_reference_fixed_cv_rmse"
    ]
)

comparison_df[
    "delta_fixed_validation_vs_XZ"
] = (
    comparison_df[
        "fixed_validation_rmse"
    ]
    -
    comparison_df[
        "XZ_reference_fixed_validation_rmse"
    ]
)

# -----------------------------------------------------------------------------
# Best family by CV for every Hamiltonian candidate
# -----------------------------------------------------------------------------

best_family_rows = []

for (
    topology,
    hy,
), group in comparison_df.groupby(
    [
        "topology",
        "hy",
    ],
    sort=True,
):

    ranked = group.sort_values(
        [
            "cv_rmse",
            "n_settings",
            "basis_change_gates",
            "n_features",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    best = ranked.iloc[0]

    best_family_rows.append(
        {
            "topology":
                topology,

            "hy":
                hy,

            "best_readout_by_cv":
                best[
                    "readout"
                ],

            "best_cv_rmse":
                best[
                    "cv_rmse"
                ],

            "best_validation_rmse":
                best[
                    "validation_rmse"
                ],

            "best_lambda":
                best[
                    "selected_lambda"
                ],

            "n_features":
                best[
                    "n_features"
                ],

            "n_settings":
                best[
                    "n_settings"
                ],

            "basis_change_gates":
                best[
                    "basis_change_gates"
                ],
        }
    )

best_family_df = pd.DataFrame(
    best_family_rows
)

# -----------------------------------------------------------------------------
# Pareto analysis across prediction/resource axes
#
# Lower is better:
#   CV RMSE
#   number of settings
#   basis-change gates
#   feature count
#
# Per topology/h_y so the Hamiltonian is held fixed.
# -----------------------------------------------------------------------------

def pareto_mask(
    frame,
    columns,
):
    values = frame[
        columns
    ].to_numpy(
        dtype=float
    )

    keep = np.ones(
        len(
            frame
        ),
        dtype=bool,
    )

    for i in range(
        len(
            frame
        )
    ):
        candidate = values[
            i
        ]

        no_worse = np.all(
            values
            <=
            candidate,
            axis=1,
        )

        strictly_better = np.any(
            values
            <
            candidate,
            axis=1,
        )

        dominates = (
            no_worse
            &
            strictly_better
        )

        dominates[
            i
        ] = False

        if np.any(
            dominates
        ):
            keep[
                i
            ] = False

    return keep


pareto_rows = []

for (
    topology,
    hy,
), group in comparison_df.groupby(
    [
        "topology",
        "hy",
    ],
    sort=True,
):

    group = (
        group
        .copy()
        .reset_index(
            drop=True
        )
    )

    group[
        "is_pareto_readout"
    ] = pareto_mask(
        group,
        [
            "cv_rmse",
            "n_settings",
            "basis_change_gates",
            "n_features",
        ],
    )

    pareto_rows.append(
        group
    )

pareto_df = pd.concat(
    pareto_rows,
    ignore_index=True,
)

pareto_candidates_df = pareto_df[
    pareto_df[
        "is_pareto_readout"
    ]
].copy()

# =============================================================================
# Reproduction audit against Step 9.1B.2 fixed-lambda XZ/XYZ results
# =============================================================================

audit_rows = []

if TUNING_FORECAST_FILE.exists():

    previous = pd.read_csv(
        TUNING_FORECAST_FILE
    )

    previous_required = {
        "topology",
        "hy",
        "readout",
        "cv_rmse_mean",
        "validation_rmse",
    }

    if previous_required.issubset(
        previous.columns
    ):

        readout_name_map = {
            "XZ_injection":
                "XZ_injection",

            "XYZ_all":
                "XYZ_all",
        }

        for _, row in results_df.iterrows():

            if row[
                "readout"
            ] not in readout_name_map:
                continue

            matches = previous[
                (
                    previous[
                        "topology"
                    ] == row[
                        "topology"
                    ]
                )
                &
                (
                    np.isclose(
                        previous[
                            "hy"
                        ].astype(
                            float
                        ),
                        float(
                            row[
                                "hy"
                            ]
                        ),
                    )
                )
                &
                (
                    previous[
                        "readout"
                    ] == readout_name_map[
                        row[
                            "readout"
                        ]
                    ]
                )
            ]

            if len(
                matches
            ) != 1:
                continue

            old = matches.iloc[
                0
            ]

            audit_rows.append(
                {
                    "topology":
                        row[
                            "topology"
                        ],

                    "hy":
                        row[
                            "hy"
                        ],

                    "readout":
                        row[
                            "readout"
                        ],

                    "previous_fixed_cv":
                        float(
                            old[
                                "cv_rmse_mean"
                            ]
                        ),

                    "current_fixed_cv":
                        row[
                            "fixed_cv_rmse"
                        ],

                    "delta_cv":
                        (
                            row[
                                "fixed_cv_rmse"
                            ]
                            -
                            float(
                                old[
                                    "cv_rmse_mean"
                                ]
                            )
                        ),

                    "previous_fixed_validation":
                        float(
                            old[
                                "validation_rmse"
                            ]
                        ),

                    "current_fixed_validation":
                        row[
                            "fixed_validation_rmse"
                        ],

                    "delta_validation":
                        (
                            row[
                                "fixed_validation_rmse"
                            ]
                            -
                            float(
                                old[
                                    "validation_rmse"
                                ]
                            )
                        ),
                }
            )

audit_df = pd.DataFrame(
    audit_rows
)

# =============================================================================
# Save
# =============================================================================

results_df.to_csv(
    RESULTS
    / "09_01b3_measurement_recheck_all.csv",
    index=False,
)

grid_df.to_csv(
    RESULTS
    / "09_01b3_measurement_recheck_ridge_grid.csv",
    index=False,
)

comparison_df.to_csv(
    RESULTS
    / "09_01b3_measurement_recheck_vs_XZ.csv",
    index=False,
)

best_family_df.to_csv(
    RESULTS
    / "09_01b3_best_readout_by_candidate.csv",
    index=False,
)

pareto_df.to_csv(
    RESULTS
    / "09_01b3_readout_pareto_all.csv",
    index=False,
)

pareto_candidates_df.to_csv(
    RESULTS
    / "09_01b3_readout_pareto_candidates.csv",
    index=False,
)

diagnostics_df.to_csv(
    RESULTS
    / "09_01b3_diagnostics.csv",
    index=False,
)

if len(
    audit_df
) > 0:
    audit_df.to_csv(
        RESULTS
        / "09_01b3_fixed_lambda_reproduction_audit.csv",
        index=False,
    )

# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------

manifest = {
    "step":
        "Week 9 Step 9.1B.3",

    "experiment":
        "compact readout recheck after memory-only Y-field tuning",

    "active_topologies":
        ACTIVE_TOPOLOGIES,

    "candidate_hy":
        CANDIDATE_HY,

    "readouts":
        READOUT_METADATA,

    "ridge_grid":
        [
            float(
                x
            )
            for x in RIDGE_GRID
        ],

    "fixed_lambda_audit":
        FORECAST_RIDGE_FIXED,

    "train_rows":
        N_TRAIN,

    "validation_rows":
        N_VALIDATION,

    "test_rows_used":
        0,

    "alpha":
        ALPHA,

    "hx":
        HX,

    "dt":
        DT,

    "Trotter_r":
        TROTTER_R,

    "memory_initialization":
        "|00><00|",

    "protocol":
        "CONT",

    "YX45_definition":
        "Tr[rho (Y4 tensor X5)]",

    "J_source":
        str(
            J_WINNER_FILE
        ),

    "data_source":
        str(
            DATA_FILE
        ),

    "weighted_score":
        False,
}

with open(
    RESULTS
    / "09_01b3_manifest.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        manifest,
        file,
        indent=2,
    )

# =============================================================================
# Console summaries
# =============================================================================

print(
    "\n"
    +
    "=" * 118
)

print(
    "MEASUREMENT RECHECK — READOUT COMPARISON"
)

print(
    "=" * 118
)

display_columns = [
    "topology",
    "hy",
    "readout",
    "n_features",
    "n_settings",
    "basis_change_gates",
    "selected_lambda",
    "cv_rmse",
    "validation_rmse",
    "delta_cv_vs_XZ",
    "delta_validation_vs_XZ",
]

print(
    comparison_df[
        display_columns
    ]
    .sort_values(
        [
            "topology",
            "hy",
            "cv_rmse",
        ]
    )
    .to_string(
        index=False
    )
)

print(
    "\n"
    +
    "=" * 118
)

print(
    "BEST READOUT BY TRAINING-ONLY CV FOR EACH HAMILTONIAN CANDIDATE"
)

print(
    "=" * 118
)

print(
    best_family_df
    .sort_values(
        [
            "topology",
            "hy",
        ]
    )
    .to_string(
        index=False
    )
)

print(
    "\n"
    +
    "=" * 118
)

print(
    "PARETO READOUT CANDIDATES — CV / SETTINGS / BASIS GATES / FEATURE COUNT"
)

print(
    "=" * 118
)

pareto_display_columns = [
    "topology",
    "hy",
    "readout",
    "n_features",
    "n_settings",
    "basis_change_gates",
    "selected_lambda",
    "cv_rmse",
    "validation_rmse",
]

print(
    pareto_candidates_df[
        pareto_display_columns
    ]
    .sort_values(
        [
            "topology",
            "hy",
            "cv_rmse",
        ]
    )
    .to_string(
        index=False
    )
)

if len(
    audit_df
) > 0:

    print(
        "\n"
        +
        "=" * 118
    )

    print(
        "FIXED-LAMBDA 0.01 REPRODUCTION AUDIT AGAINST STEP 9.1B.2"
    )

    print(
        "=" * 118
    )

    print(
        audit_df.to_string(
            index=False
        )
    )

print(
    "\n"
    +
    "=" * 118
)

print(
    "YX45 SIGNAL DIAGNOSTIC"
)

print(
    "=" * 118
)

print(
    diagnostics_df[
        [
            "topology",
            "hy",
            "YX45_train_std",
            "YX45_all_std",
            "YX45_max_abs",
            "unitary_error_fro",
        ]
    ]
    .sort_values(
        [
            "topology",
            "hy",
        ]
    )
    .to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

for filename in [
    "09_01b3_measurement_recheck_all.csv",
    "09_01b3_measurement_recheck_ridge_grid.csv",
    "09_01b3_measurement_recheck_vs_XZ.csv",
    "09_01b3_best_readout_by_candidate.csv",
    "09_01b3_readout_pareto_all.csv",
    "09_01b3_readout_pareto_candidates.csv",
    "09_01b3_diagnostics.csv",
    "09_01b3_fixed_lambda_reproduction_audit.csv",
    "09_01b3_manifest.json",
]:

    path = RESULTS / filename

    if path.exists():
        print(
            f"  {path}"
        )

print(
    f"\nTotal runtime: "
    f"{time.time() - experiment_start:.1f}s"
)

print(
    "Step 9.1B.3 complete."
)
