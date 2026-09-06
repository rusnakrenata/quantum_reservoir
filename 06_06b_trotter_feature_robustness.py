from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATA_FILE = (
    RESULTS_DIR
    / "03_01_preprocessed_samples.csv"
)

COUPLING_FILE = (
    RESULTS_DIR
    / "05_03a_reservoir_couplings.csv"
)

# Existing 6.6A F4 results.
# Used only as a reproducibility audit if the file exists.
REFERENCE_66A_FILE = (
    RESULTS_DIR
    / "06_06_trotter_validation.csv"
)

CV_OUTPUT = (
    RESULTS_DIR
    / "06_06b_trotter_feature_cv.csv"
)

FOLD_OUTPUT = (
    RESULTS_DIR
    / "06_06b_trotter_feature_folds.csv"
)

RESULT_OUTPUT = (
    RESULTS_DIR
    / "06_06b_trotter_feature_validation.csv"
)

ERROR_OUTPUT = (
    RESULTS_DIR
    / "06_06b_trotter_errors.csv"
)

GEOMETRY_OUTPUT = (
    RESULTS_DIR
    / "06_06b_trotter_feature_geometry.csv"
)

PURITY_OUTPUT = (
    RESULTS_DIR
    / "06_06b_trotter_feature_purity.csv"
)

RESOURCE_OUTPUT = (
    RESULTS_DIR
    / "06_06b_resource_summary.csv"
)

PARETO_OUTPUT = (
    RESULTS_DIR
    / "06_06b_pareto_front.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "06_06b_summary.txt"
)


# ============================================================
# Frozen architecture / dynamics
# ============================================================

N_INPUT = 4
N_MEMORY = 2
N_QUBITS = 6

D_INPUT = 2 ** N_INPUT
D_MEMORY = 2 ** N_MEMORY
D_TOTAL = 2 ** N_QUBITS

# Selected small-but-nonzero continuous input gain
ALPHA = 0.005

J_MULT = 1.0

H_X = 0.5

DELTA_T = 0.8

CLAIM_CLIP_Z = 3.0


# ============================================================
# Feature sets
#
# Fixed six-qubit topology:
#
# q0 = claims
# q1 = weekday
# q2 = policy
# q3 = holiday
# q4,q5 = memory
#
# F2 = claims + weekday
# F3 = claims + weekday + policy
# F4 = claims + weekday + policy + holiday
# ============================================================

FEATURE_SETS = [
    "F2",
    "F3",
    "F4",
]


# ============================================================
# Trotter depths
# ============================================================

TROTTER_DEPTHS = [
    1,
    2,
    4,
    8,
    16,
]


# ============================================================
# Temporal protocols
# ============================================================

WINDOWS = [
    1,
    2,
    5,
    7,
    14,
    21,
    28,
]

MAX_WINDOW = max(
    WINDOWS
)


# ============================================================
# Ridge grid
# ============================================================

RIDGE_LAMBDAS = [
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
    100000.0,
]

N_CV_SPLITS = 5


# ============================================================
# QRC readout features
# ============================================================

QRC_FEATURE_COLUMNS = [
    "X0", "Z0",
    "X1", "Z1",
    "X2", "Z2",
    "X3", "Z3",
    "X4", "Z4",
    "X5", "Z5",
]


TARGET_CANDIDATES = [
    "target_property_damage_claim_count",
    "property_damage_claim_count_t_plus_1",
    "property_damage_claim_count_target",
    "target_claim_count",
    "C_t_plus_1",
]


# ============================================================
# Logging
# ============================================================

output_lines = []


def log(text=""):

    text = str(text)

    print(text)

    output_lines.append(
        text
    )


# ============================================================
# Matrices
# ============================================================

I2 = np.eye(
    2,
    dtype=complex,
)

X = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=complex,
)

Z = np.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=complex,
)


# ============================================================
# Resolve target
# ============================================================

def resolve_target_column(df):

    for candidate in TARGET_CANDIDATES:

        if candidate in df.columns:

            return candidate

    raise ValueError(
        "Could not resolve target column."
    )


# ============================================================
# Policy softsign selected in 6.4B
# ============================================================

def softsign3(z):

    z = float(z)

    return (
        z
        /
        (
            3.0
            +
            abs(z)
        )
    )


# ============================================================
# Ry state
# ============================================================

def ry_state(theta):

    return np.array(
        [
            np.cos(
                theta / 2.0
            ),
            np.sin(
                theta / 2.0
            ),
        ],
        dtype=complex,
    )


# ============================================================
# Input encoding
# ============================================================

def encode_input_state(
    row,
    feature_set,
):

    # --------------------------------------------------------
    # q0: Claims
    #
    # Present in F2/F3/F4.
    # --------------------------------------------------------

    claim_z = float(
        row[
            "C_t_z"
        ]
    )

    claim_encoded = (
        np.clip(
            claim_z,
            -CLAIM_CLIP_Z,
            CLAIM_CLIP_Z,
        )
        /
        CLAIM_CLIP_Z
    )

    theta_claim = (
        ALPHA
        *
        claim_encoded
    )

    # --------------------------------------------------------
    # q1: Day of week
    #
    # Present in F2/F3/F4.
    # Keep natural cyclic angle.
    # --------------------------------------------------------

    theta_day = np.mod(
        np.arctan2(
            float(
                row[
                    "D_sin"
                ]
            ),
            float(
                row[
                    "D_cos"
                ]
            ),
        ),
        2.0
        *
        np.pi,
    )

    # --------------------------------------------------------
    # q2: Policy exposure
    #
    # F2: OFF
    # F3: ON
    # F4: ON
    # --------------------------------------------------------

    if feature_set in {
        "F3",
        "F4",
    }:

        policy_z = float(
            row[
                "P_t_z"
            ]
        )

        theta_policy = (
            ALPHA
            *
            softsign3(
                policy_z
            )
        )

    else:

        theta_policy = 0.0

    # --------------------------------------------------------
    # q3: Holiday
    #
    # F2: OFF
    # F3: OFF
    # F4: ON
    # --------------------------------------------------------

    if feature_set == "F4":

        theta_holiday = (
            np.pi
            *
            float(
                row[
                    "is_public_holiday_t_plus_1"
                ]
            )
        )

    else:

        theta_holiday = 0.0

    # --------------------------------------------------------
    # Prepare four injection qubits
    # --------------------------------------------------------

    q0 = ry_state(
        theta_claim
    )

    q1 = ry_state(
        theta_day
    )

    q2 = ry_state(
        theta_policy
    )

    q3 = ry_state(
        theta_holiday
    )

    # Qiskit little-endian convention:
    #
    # q3 kron q2 kron q1 kron q0
    #
    # followed later by memory subsystem.

    return np.kron(
        q3,
        np.kron(
            q2,
            np.kron(
                q1,
                q0,
            ),
        ),
    )


# ============================================================
# Initial memory state |00><00|
# ============================================================

def initial_memory():

    rho = np.zeros(
        (
            D_MEMORY,
            D_MEMORY,
        ),
        dtype=complex,
    )

    rho[
        0,
        0,
    ] = 1.0

    return rho


# ============================================================
# Couplings
# ============================================================

def load_couplings():

    coupling_df = pd.read_csv(
        COUPLING_FILE
    )

    couplings = {}

    for _, row in (
        coupling_df.iterrows()
    ):

        edge = (
            int(
                row[
                    "q_i"
                ]
            ),
            int(
                row[
                    "q_j"
                ]
            ),
        )

        couplings[
            edge
        ] = (
            J_MULT
            *
            float(
                row[
                    "J_ij"
                ]
            )
        )

    return couplings


# ============================================================
# Single-qubit Pauli operator
# ============================================================

def single_qubit_operator(
    target_q,
    pauli,
):

    matrices = []

    for q in reversed(
        range(
            N_QUBITS
        )
    ):

        if q == target_q:

            matrices.append(
                pauli
            )

        else:

            matrices.append(
                I2
            )

    result = (
        matrices[0]
    )

    for matrix in (
        matrices[
            1:
        ]
    ):

        result = np.kron(
            result,
            matrix,
        )

    return result


# ============================================================
# ZZ operator
# ============================================================

def zz_operator(
    q_i,
    q_j,
):

    matrices = []

    for q in reversed(
        range(
            N_QUBITS
        )
    ):

        if q in {
            q_i,
            q_j,
        }:

            matrices.append(
                Z
            )

        else:

            matrices.append(
                I2
            )

    result = (
        matrices[0]
    )

    for matrix in (
        matrices[
            1:
        ]
    ):

        result = np.kron(
            result,
            matrix,
        )

    return result


# ============================================================
# Hamiltonian
#
# H = sum J_ij Z_i Z_j + hx sum X_i
# ============================================================

def build_hamiltonian(
    couplings,
):

    H = np.zeros(
        (
            D_TOTAL,
            D_TOTAL,
        ),
        dtype=complex,
    )

    # ZZ interactions
    for (
        q_i,
        q_j,
    ), J in (
        couplings.items()
    ):

        H += (
            J
            *
            zz_operator(
                q_i,
                q_j,
            )
        )

    # X field
    for q in range(
        N_QUBITS
    ):

        H += (
            H_X
            *
            single_qubit_operator(
                q,
                X,
            )
        )

    return H


# ============================================================
# Exact unitary
# ============================================================

def build_exact_unitary(
    couplings,
):

    H = (
        build_hamiltonian(
            couplings
        )
    )

    return expm(
        -1j
        *
        H
        *
        DELTA_T
    )


# ============================================================
# First-order Trotter unitary
# ============================================================

def build_trotter_unitary(
    couplings,
    r,
):

    qc = QuantumCircuit(
        N_QUBITS
    )

    dt_step = (
        DELTA_T
        /
        r
    )

    for _ in range(
        r
    ):

        # ----------------------------------------------------
        # ZZ layer
        # ----------------------------------------------------

        for (
            q_i,
            q_j,
        ), J in (
            couplings.items()
        ):

            qc.rzz(
                2.0
                *
                J
                *
                dt_step,
                q_i,
                q_j,
            )

        # ----------------------------------------------------
        # X layer
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            qc.rx(
                2.0
                *
                H_X
                *
                dt_step,
                q,
            )

    return np.asarray(
        Operator(
            qc
        ).data,
        dtype=complex,
    )


# ============================================================
# Observables
# ============================================================

def build_observables():

    operators = []

    for q in range(
        N_QUBITS
    ):

        operators.append(
            single_qubit_operator(
                q,
                X,
            )
        )

        operators.append(
            single_qubit_operator(
                q,
                Z,
            )
        )

    return np.stack(
        operators,
        axis=0,
    )


# ============================================================
# Trotter approximation metrics
# ============================================================

def unitary_error_metrics(
    U_exact,
    U_test,
):

    d = (
        U_exact.shape[0]
    )

    overlap = np.trace(
        U_exact.conj().T
        @ U_test
    )

    # Global-phase-insensitive
    # average gate fidelity.

    average_gate_fidelity = float(
        (
            abs(
                overlap
            ) ** 2
            +
            d
        )
        /
        (
            d
            *
            (
                d + 1
            )
        )
    )

    # Align global phase before
    # Frobenius comparison.

    phase = np.angle(
        overlap
    )

    U_aligned = (
        np.exp(
            -1j
            *
            phase
        )
        *
        U_test
    )

    normalized_frobenius_error = float(
        np.linalg.norm(
            U_exact
            -
            U_aligned,
            ord="fro",
        )
        /
        np.linalg.norm(
            U_exact,
            ord="fro",
        )
    )

    return {
        "normalized_frobenius_error":
            normalized_frobenius_error,

        "average_gate_fidelity":
            average_gate_fidelity,
    }


# ============================================================
# Partial trace over input subsystem
# ============================================================

def trace_out_input(
    rho_global,
):

    reshaped = (
        rho_global.reshape(
            D_MEMORY,
            D_INPUT,
            D_MEMORY,
            D_INPUT,
        )
    )

    return np.einsum(
        "aibi->ab",
        reshaped,
    )


# ============================================================
# Direct density-matrix step
# ============================================================

def direct_step(
    rho_memory,
    phi,
    U,
    observables,
):

    rho_input = np.outer(
        phi,
        phi.conj(),
    )

    rho_pre = np.kron(
        rho_memory,
        rho_input,
    )

    rho_global = (
        U
        @ rho_pre
        @ U.conj().T
    )

    rho_memory_new = (
        trace_out_input(
            rho_global
        )
    )

    features = np.asarray(
        [
            np.sum(
                rho_global
                *
                operator.T
            )

            for operator
            in observables
        ],
        dtype=complex,
    )

    return (
        rho_memory_new,
        features,
    )


# ============================================================
# Kraus / superoperator
# ============================================================

def build_kraus(
    phi,
    U4,
):

    return np.einsum(
        "abcd,d->bac",
        U4,
        phi,
    )


def kraus_to_superoperator(
    kraus,
):

    S4 = np.einsum(
        "aij,alk->iljk",
        kraus,
        kraus.conj(),
        optimize=True,
    )

    return S4.reshape(
        D_MEMORY ** 2,
        D_MEMORY ** 2,
    )


# ============================================================
# Observable transformation
# ============================================================

def build_transformed_observables(
    U,
    observables,
):

    transformed = []

    for operator in observables:

        B = (
            U.conj().T
            @ operator
            @ U
        )

        transformed.append(
            B.reshape(
                D_MEMORY,
                D_INPUT,
                D_MEMORY,
                D_INPUT,
            )
        )

    return np.stack(
        transformed,
        axis=0,
    )


def build_feature_map(
    phi,
    transformed,
):

    A = np.einsum(
        "j,onjmi,i->onm",
        phi.conj(),
        transformed,
        phi,
        optimize=True,
    )

    return (
        np.transpose(
            A,
            (
                0,
                2,
                1,
            ),
        )
        .reshape(
            len(
                QRC_FEATURE_COLUMNS
            ),
            D_MEMORY ** 2,
        )
    )


# ============================================================
# Generate CONT + all RWP windows
# ============================================================

def generate_protocols(
    df,
    target_column,
    input_states,
    feature_set,
    evolution_name,
    U,
    observables,
):

    U4 = U.reshape(
        D_MEMORY,
        D_INPUT,
        D_MEMORY,
        D_INPUT,
    )

    transformed = (
        build_transformed_observables(
            U,
            observables,
        )
    )

    rho0 = (
        initial_memory()
    )

    vec0 = (
        rho0.reshape(
            -1
        )
    )

    # --------------------------------------------------------
    # age_vectors[k]
    #
    # memory after k chronological inputs
    # starting from reset state
    # --------------------------------------------------------

    age_vectors = np.zeros(
        (
            MAX_WINDOW + 1,
            D_MEMORY ** 2,
        ),
        dtype=complex,
    )

    age_vectors[0] = (
        vec0
    )

    cont_vec = (
        vec0.copy()
    )

    rows = {
        "CONT": [],
    }

    purities = {
        "CONT": [],
    }

    for W in WINDOWS:

        rows[
            f"RWP_W{W}"
        ] = []

        purities[
            f"RWP_W{W}"
        ] = []

    audit_feature_errors = []
    audit_memory_errors = []

    # ========================================================
    # Chronological pass
    # ========================================================

    for t, phi in enumerate(
        input_states
    ):

        kraus = (
            build_kraus(
                phi,
                U4,
            )
        )

        S = (
            kraus_to_superoperator(
                kraus
            )
        )

        L = (
            build_feature_map(
                phi,
                transformed,
            )
        )

        current = (
            df.iloc[t]
        )

        # ----------------------------------------------------
        # CONT
        # ----------------------------------------------------

        cont_features = (
            L
            @ cont_vec
        )

        cont_new = (
            S
            @ cont_vec
        )

        cont_rho = (
            cont_new.reshape(
                D_MEMORY,
                D_MEMORY,
            )
        )

        cont_purity = float(
            np.real(
                np.trace(
                    cont_rho
                    @ cont_rho
                )
            )
        )

        row = {
            "feature_set": (
                feature_set
            ),

            "evolution": (
                evolution_name
            ),

            "protocol": "CONT",

            "window_W": np.nan,

            "input_date": (
                current[
                    "input_date"
                ]
            ),

            "target_date": (
                current[
                    "target_date"
                ]
            ),

            "split": (
                current[
                    "split"
                ]
            ),

            "target": float(
                current[
                    target_column
                ]
            ),

            "memory_purity": (
                cont_purity
            ),
        }

        for (
            name,
            value,
        ) in zip(
            QRC_FEATURE_COLUMNS,
            cont_features,
        ):

            row[
                name
            ] = float(
                np.real(
                    value
                )
            )

        rows[
            "CONT"
        ].append(
            row
        )

        purities[
            "CONT"
        ].append(
            cont_purity
        )

        # ----------------------------------------------------
        # RWP
        # ----------------------------------------------------

        valid_windows = [
            W
            for W in WINDOWS
            if t >= W - 1
        ]

        if valid_windows:

            indices = np.asarray(
                [
                    W - 1
                    for W
                    in valid_windows
                ],
                dtype=int,
            )

            previous_memories = (
                age_vectors[
                    indices
                ]
            )

            feature_batch = (
                previous_memories
                @ L.T
            )

        old_age_vectors = (
            age_vectors.copy()
        )

        propagated = (
            S
            @ old_age_vectors[
                :MAX_WINDOW
            ].T
        ).T

        age_vectors.fill(
            0.0
        )

        age_vectors[0] = (
            vec0
        )

        age_vectors[
            1:
            MAX_WINDOW + 1
        ] = propagated

        # ----------------------------------------------------
        # Store each valid temporal window
        # ----------------------------------------------------

        for position, W in enumerate(
            valid_windows
        ):

            feature_vec = (
                feature_batch[
                    position
                ]
            )

            memory_vec = (
                age_vectors[
                    W
                ]
            )

            rho_memory = (
                memory_vec.reshape(
                    D_MEMORY,
                    D_MEMORY,
                )
            )

            purity = float(
                np.real(
                    np.trace(
                        rho_memory
                        @ rho_memory
                    )
                )
            )

            start_idx = (
                t
                -
                W
                +
                1
            )

            row = {
                "feature_set": (
                    feature_set
                ),

                "evolution": (
                    evolution_name
                ),

                "protocol": (
                    f"RWP_W{W}"
                ),

                "window_W": W,

                "window_start_input_date": (
                    df.iloc[
                        start_idx
                    ][
                        "input_date"
                    ]
                ),

                "input_date": (
                    current[
                        "input_date"
                    ]
                ),

                "target_date": (
                    current[
                        "target_date"
                    ]
                ),

                "split": (
                    current[
                        "split"
                    ]
                ),

                "target": float(
                    current[
                        target_column
                    ]
                ),

                "memory_purity": (
                    purity
                ),
            }

            for (
                name,
                value,
            ) in zip(
                QRC_FEATURE_COLUMNS,
                feature_vec,
            ):

                row[
                    name
                ] = float(
                    np.real(
                        value
                    )
                )

            rows[
                f"RWP_W{W}"
            ].append(
                row
            )

            purities[
                f"RWP_W{W}"
            ].append(
                purity
            )

        # ----------------------------------------------------
        # Direct-vs-superoperator correctness audit
        #
        # Use W=7 state at two dates.
        # ----------------------------------------------------

        if (
            t in {
                500,
                1000,
            }
            and
            t >= 6
        ):

            previous_vec = (
                old_age_vectors[
                    6
                ]
            )

            previous_rho = (
                previous_vec.reshape(
                    D_MEMORY,
                    D_MEMORY,
                )
            )

            (
                direct_memory,
                direct_features,
            ) = direct_step(
                rho_memory=(
                    previous_rho
                ),
                phi=phi,
                U=U,
                observables=(
                    observables
                ),
            )

            fast_features = (
                L
                @ previous_vec
            )

            fast_memory = (
                (
                    S
                    @ previous_vec
                )
                .reshape(
                    D_MEMORY,
                    D_MEMORY,
                )
            )

            audit_feature_errors.append(
                float(
                    np.max(
                        np.abs(
                            fast_features
                            -
                            direct_features
                        )
                    )
                )
            )

            audit_memory_errors.append(
                float(
                    np.max(
                        np.abs(
                            fast_memory
                            -
                            direct_memory
                        )
                    )
                )
            )

        # Continue CONT memory
        cont_vec = (
            cont_new
        )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    tables = {
        protocol:
            pd.DataFrame(
                protocol_rows
            )

        for (
            protocol,
            protocol_rows,
        ) in rows.items()
    }

    # --------------------------------------------------------
    # Purity summary
    # --------------------------------------------------------

    purity_rows = []

    for (
        protocol,
        values,
    ) in purities.items():

        values = np.asarray(
            values,
            dtype=float,
        )

        purity_rows.append(
            {
                "feature_set": (
                    feature_set
                ),

                "evolution": (
                    evolution_name
                ),

                "protocol": (
                    protocol
                ),

                "min_purity": float(
                    np.min(
                        values
                    )
                ),

                "mean_purity": float(
                    np.mean(
                        values
                    )
                ),

                "max_purity": float(
                    np.max(
                        values
                    )
                ),

                "final_purity": float(
                    values[-1]
                ),
            }
        )

    return (
        tables,
        pd.DataFrame(
            purity_rows
        ),
        max(
            audit_feature_errors
        ),
        max(
            audit_memory_errors
        ),
    )


# ============================================================
# Ridge model
# ============================================================

def make_model(
    ridge_lambda,
):

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "ridge",
                Ridge(
                    alpha=(
                        ridge_lambda
                    ),
                    fit_intercept=True,
                ),
            ),
        ]
    )


# ============================================================
# Chronological Ridge CV
#
# Also returns every fold result so 6.6B can compare
# depths at fold level.
# ============================================================

def ridge_cv(
    X,
    y,
    configuration,
):

    splitter = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )

    summary_rows = []
    fold_rows = []

    for ridge_lambda in (
        RIDGE_LAMBDAS
    ):

        fold_rmse = []

        for fold_number, (
            train_idx,
            val_idx,
        ) in enumerate(
            splitter.split(
                X
            ),
            start=1,
        ):

            model = (
                make_model(
                    ridge_lambda
                )
            )

            model.fit(
                X[
                    train_idx
                ],
                y[
                    train_idx
                ],
            )

            prediction = (
                model.predict(
                    X[
                        val_idx
                    ]
                )
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y[
                            val_idx
                        ],
                        prediction,
                    )
                )
            )

            fold_rmse.append(
                rmse
            )

            fold_rows.append(
                {
                    "configuration": (
                        configuration
                    ),

                    "lambda": (
                        ridge_lambda
                    ),

                    "fold": (
                        fold_number
                    ),

                    "train_fold_rows": (
                        len(
                            train_idx
                        )
                    ),

                    "validation_fold_rows": (
                        len(
                            val_idx
                        )
                    ),

                    "fold_rmse": (
                        rmse
                    ),
                }
            )

        summary_rows.append(
            {
                "configuration": (
                    configuration
                ),

                "lambda": (
                    ridge_lambda
                ),

                "cv_rmse_mean": float(
                    np.mean(
                        fold_rmse
                    )
                ),

                "cv_rmse_std": float(
                    np.std(
                        fold_rmse,
                        ddof=1,
                    )
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    best = (
        summary_df
        .sort_values(
            [
                "cv_rmse_mean",
                "lambda",
            ]
        )
        .iloc[0]
    )

    best_lambda = float(
        best[
            "lambda"
        ]
    )

    best_folds = (
        fold_df[
            np.isclose(
                fold_df[
                    "lambda"
                ],
                best_lambda,
            )
        ]
        .copy()
    )

    return (
        best_lambda,

        float(
            best[
                "cv_rmse_mean"
            ]
        ),

        float(
            best[
                "cv_rmse_std"
            ]
        ),

        summary_df,

        best_folds,
    )


# ============================================================
# Feature geometry
# ============================================================

def feature_geometry(
    X_train,
):

    scaler = (
        StandardScaler()
    )

    Xs = (
        scaler.fit_transform(
            X_train
        )
    )

    raw_std = np.std(
        X_train,
        axis=0,
        ddof=0,
    )

    min_raw_std = float(
        np.min(
            raw_std
        )
    )

    max_raw_std = float(
        np.max(
            raw_std
        )
    )

    near_zero_std_count = int(
        np.sum(
            raw_std
            <
            1e-10
        )
    )

    singular = np.linalg.svd(
        Xs,
        compute_uv=False,
    )

    rank = int(
        np.linalg.matrix_rank(
            Xs
        )
    )

    sigma_max = float(
        singular[0]
    )

    sigma_min = float(
        singular[-1]
    )

    if (
        rank
        <
        Xs.shape[1]
        or
        sigma_min <= 0
    ):

        kappa = np.inf

    else:

        kappa = float(
            sigma_max
            /
            sigma_min
        )

    C = (
        Xs.T
        @ Xs
    ) / len(
        Xs
    )

    rho_C = float(
        np.max(
            np.linalg.eigvalsh(
                C
            )
        )
    )

    return {
        "rank": (
            rank
        ),

        "sigma_max": (
            sigma_max
        ),

        "sigma_min": (
            sigma_min
        ),

        "kappa_X": (
            kappa
        ),

        "rho_C": (
            rho_C
        ),

        "min_raw_feature_std": (
            min_raw_std
        ),

        "max_raw_feature_std": (
            max_raw_std
        ),

        "near_zero_std_count":
            near_zero_std_count,
    }


# ============================================================
# Validation diagnostics
# ============================================================

def validation_metrics(
    X_train,
    y_train,
    X_val,
    y_val,
    best_lambda,
):

    model = (
        make_model(
            best_lambda
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    prediction = (
        model.predict(
            X_val
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_val,
                prediction,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_val,
            prediction,
        )
    )

    bias = float(
        np.mean(
            prediction
            -
            y_val
        )
    )

    train_std = float(
        np.std(
            y_train,
            ddof=0,
        )
    )

    nrmse = (
        rmse
        /
        train_std
    )

    return {
        "RMSE": (
            rmse
        ),

        "MAE": (
            mae
        ),

        "Bias": (
            bias
        ),

        "NRMSE": (
            nrmse
        ),
    }


# ============================================================
# Resource cost
# ============================================================

def resource_values(
    evolution,
    r,
):

    if evolution == "EXACT":

        return {
            "logical_RZZ": np.nan,
            "logical_RX": np.nan,
            "approx_CX": np.nan,
        }

    logical_rzz = (
        5
        *
        int(
            r
        )
    )

    logical_rx = (
        6
        *
        int(
            r
        )
    )

    # Standard logical decomposition:
    #
    # RZZ(theta)
    # ~ CX - RZ(theta) - CX
    #
    # before backend-specific transpilation.

    approx_cx = (
        2
        *
        logical_rzz
    )

    return {
        "logical_RZZ": (
            logical_rzz
        ),

        "logical_RX": (
            logical_rx
        ),

        "approx_CX": (
            approx_cx
        ),
    }


# ============================================================
# F4 reproduction audit against 6.6A
# ============================================================

def audit_against_66a(
    results,
):

    if not (
        REFERENCE_66A_FILE.exists()
    ):

        log()

        log(
            "6.6A reference file "
            "not found; skipping "
            "F4 reproduction audit."
        )

        return

    old = pd.read_csv(
        REFERENCE_66A_FILE
    )

    new = (
        results[
            results[
                "feature_set"
            ]
            == "F4"
        ]
        .copy()
    )

    comparison = old.merge(
        new,
        on=[
            "evolution",
            "protocol",
        ],
        suffixes=(
            "_old",
            "_new",
        ),
    )

    if len(
        comparison
    ) == 0:

        log()

        log(
            "WARNING: could not align "
            "6.6A and 6.6B F4 results."
        )

        return

    cv_error = float(
        np.max(
            np.abs(
                comparison[
                    "cv_rmse_mean_old"
                ]
                -
                comparison[
                    "cv_rmse_mean_new"
                ]
            )
        )
    )

    validation_error = float(
        np.max(
            np.abs(
                comparison[
                    "RMSE_old"
                ]
                -
                comparison[
                    "RMSE_new"
                ]
            )
        )
    )

    log()
    log(
        "6.6A F4 reproduction audit:"
    )

    log(
        "  max CV RMSE error  = "
        f"{cv_error:.3e}"
    )

    log(
        "  max Val RMSE error = "
        f"{validation_error:.3e}"
    )

    if (
        cv_error > 1e-9
        or
        validation_error > 1e-9
    ):

        raise RuntimeError(
            "6.6A F4 reproduction "
            "audit FAILED."
        )


# ============================================================
# Pareto frontier
#
# A configuration is dominated if another finite-depth
# configuration has:
#
#   <= CV RMSE
#   <= CX cost
#
# and is strictly better in at least one.
#
# EXACT is excluded because it is not a hardware circuit
# candidate.
# ============================================================

def calculate_pareto_front(
    resource_df,
):

    pareto_rows = []

    for feature_set in (
        FEATURE_SETS
    ):

        subset = (
            resource_df[
                (
                    resource_df[
                        "feature_set"
                    ]
                    == feature_set
                )
                &
                (
                    resource_df[
                        "evolution"
                    ]
                    != "EXACT"
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        is_pareto = []

        for i, row in (
            subset.iterrows()
        ):

            dominated = False

            for j, other in (
                subset.iterrows()
            ):

                if i == j:

                    continue

                no_worse_rmse = (
                    other[
                        "cv_rmse_mean"
                    ]
                    <=
                    row[
                        "cv_rmse_mean"
                    ]
                )

                no_worse_cost = (
                    other[
                        "approx_CX"
                    ]
                    <=
                    row[
                        "approx_CX"
                    ]
                )

                strictly_better = (
                    (
                        other[
                            "cv_rmse_mean"
                        ]
                        <
                        row[
                            "cv_rmse_mean"
                        ]
                    )
                    or
                    (
                        other[
                            "approx_CX"
                        ]
                        <
                        row[
                            "approx_CX"
                        ]
                    )
                )

                if (
                    no_worse_rmse
                    and
                    no_worse_cost
                    and
                    strictly_better
                ):

                    dominated = True
                    break

            is_pareto.append(
                not dominated
            )

        subset[
            "is_pareto"
        ] = is_pareto

        pareto_rows.append(
            subset
        )

    return pd.concat(
        pareto_rows,
        ignore_index=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 6 - STEP 6.6B")
    log("TROTTER-DEPTH ROBUSTNESS ACROSS F2 / F3 / F4")
    log("TASK PERFORMANCE vs PHYSICS ACCURACY vs RESOURCE COST")
    log("=" * 80)
    log()

    log(
        "Frozen dynamics:"
    )

    log(
        f"  alpha = {ALPHA}"
    )

    log(
        f"  J multiplier = {J_MULT}"
    )

    log(
        f"  hx = {H_X}"
    )

    log(
        f"  dt = {DELTA_T}"
    )

    log(
        "  memory qubits = 2"
    )

    log(
        "  six-qubit topology retained "
        "for F2/F3/F4"
    )

    log()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_FILE
    )

    df[
        "input_date"
    ] = pd.to_datetime(
        df[
            "input_date"
        ]
    )

    df[
        "target_date"
    ] = pd.to_datetime(
        df[
            "target_date"
        ]
    )

    df = (
        df
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        df
    ) != 1692:

        raise RuntimeError(
            "Expected exactly "
            "1692 aligned samples."
        )

    target_column = (
        resolve_target_column(
            df
        )
    )

    couplings = (
        load_couplings()
    )

    observables = (
        build_observables()
    )

    # ========================================================
    # Encode F2/F3/F4 once
    # ========================================================

    encoded_inputs = {}

    for feature_set in (
        FEATURE_SETS
    ):

        encoded_inputs[
            feature_set
        ] = [
            encode_input_state(
                row=row,
                feature_set=(
                    feature_set
                ),
            )

            for _, row
            in df.iterrows()
        ]

    # ========================================================
    # Exact Hamiltonian evolution
    # ========================================================

    U_exact = (
        build_exact_unitary(
            couplings
        )
    )

    exact_unitarity_error = float(
        np.linalg.norm(
            U_exact.conj().T
            @ U_exact
            -
            np.eye(
                D_TOTAL,
                dtype=complex,
            ),
            ord="fro",
        )
    )

    log(
        "Exact-unitary "
        "unitarity error = "
        f"{exact_unitarity_error:.3e}"
    )

    # ========================================================
    # Construct all evolution settings
    # ========================================================

    evolution_settings = []

    error_rows = []

    for r in (
        TROTTER_DEPTHS
    ):

        U_r = (
            build_trotter_unitary(
                couplings=(
                    couplings
                ),
                r=r,
            )
        )

        metrics = (
            unitary_error_metrics(
                U_exact=(
                    U_exact
                ),
                U_test=(
                    U_r
                ),
            )
        )

        resources = (
            resource_values(
                evolution=(
                    f"r{r}"
                ),
                r=r,
            )
        )

        error_rows.append(
            {
                "evolution": (
                    f"r{r}"
                ),

                "r": (
                    r
                ),

                **metrics,

                **resources,
            }
        )

        evolution_settings.append(
            (
                f"r{r}",
                r,
                U_r,
            )
        )

    # Exact physics reference
    error_rows.append(
        {
            "evolution": "EXACT",

            "r": np.nan,

            "normalized_frobenius_error":
                0.0,

            "average_gate_fidelity":
                1.0,

            "logical_RZZ": np.nan,

            "logical_RX": np.nan,

            "approx_CX": np.nan,
        }
    )

    evolution_settings.append(
        (
            "EXACT",
            None,
            U_exact,
        )
    )

    error_df = pd.DataFrame(
        error_rows
    )

    error_df.to_csv(
        ERROR_OUTPUT,
        index=False,
    )

    log()
    log(
        "Trotter approximation / "
        "logical cost:"
    )

    for _, row in (
        error_df.iterrows()
    ):

        if (
            row[
                "evolution"
            ]
            == "EXACT"
        ):

            log(
                "  EXACT "
                "FroErr=0 "
                "Favg=1"
            )

        else:

            log(
                f"  "
                f"{row['evolution']:<4s} "
                f"FroErr="
                f"{row['normalized_frobenius_error']:.6e} "
                f"Favg="
                f"{row['average_gate_fidelity']:.10f} "
                f"RZZ="
                f"{int(row['logical_RZZ']):3d} "
                f"~CX="
                f"{int(row['approx_CX']):3d}"
            )

    # ========================================================
    # Storage
    # ========================================================

    all_cv = []
    all_best_folds = []
    all_results = []
    all_geometry = []
    all_purity = []

    # ========================================================
    # Feature-set loop
    # ========================================================

    for feature_set in (
        FEATURE_SETS
    ):

        log()
        log("#" * 80)

        log(
            f"FEATURE SET = "
            f"{feature_set}"
        )

        log("#" * 80)

        # ====================================================
        # Evolution loop
        # ====================================================

        for (
            evolution_name,
            r,
            U,
        ) in (
            evolution_settings
        ):

            log()
            log("=" * 80)

            log(
                f"{feature_set} | "
                f"EVOLUTION = "
                f"{evolution_name}"
            )

            log("=" * 80)

            unitarity_error = float(
                np.linalg.norm(
                    U.conj().T
                    @ U
                    -
                    np.eye(
                        D_TOTAL,
                        dtype=complex,
                    ),
                    ord="fro",
                )
            )

            log(
                "Unitarity error = "
                f"{unitarity_error:.3e}"
            )

            # ------------------------------------------------
            # Generate quantum features
            # ------------------------------------------------

            (
                tables,
                purity_df,
                feature_error,
                memory_error,
            ) = generate_protocols(
                df=df,

                target_column=(
                    target_column
                ),

                input_states=(
                    encoded_inputs[
                        feature_set
                    ]
                ),

                feature_set=(
                    feature_set
                ),

                evolution_name=(
                    evolution_name
                ),

                U=U,

                observables=(
                    observables
                ),
            )

            log(
                "Direct audit feature error = "
                f"{feature_error:.3e}"
            )

            log(
                "Direct audit memory error  = "
                f"{memory_error:.3e}"
            )

            if (
                feature_error > 1e-10
                or
                memory_error > 1e-10
            ):

                raise RuntimeError(
                    f"{feature_set} / "
                    f"{evolution_name}: "
                    "superoperator audit FAILED."
                )

            purity_df[
                "r"
            ] = (
                np.nan
                if r is None
                else r
            )

            all_purity.append(
                purity_df
            )

            # ================================================
            # Temporal protocol loop
            # ================================================

            for (
                protocol,
                table,
            ) in tables.items():

                table = (
                    table
                    .sort_values(
                        "target_date"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                split = (
                    table[
                        "split"
                    ]
                    .astype(str)
                    .str.lower()
                )

                train_mask = (
                    split
                    == "train"
                )

                val_mask = (
                    split.isin(
                        [
                            "validation",
                            "val",
                        ]
                    )
                )

                test_mask = (
                    split
                    == "test"
                )

                n_train = int(
                    train_mask.sum()
                )

                n_val = int(
                    val_mask.sum()
                )

                n_test = int(
                    test_mask.sum()
                )

                # --------------------------------------------
                # Expected sample counts
                # --------------------------------------------

                if protocol == "CONT":

                    expected_train = 1095

                else:

                    W = int(
                        protocol.split(
                            "W"
                        )[-1]
                    )

                    expected_train = (
                        1095
                        -
                        (
                            W - 1
                        )
                    )

                if (
                    n_train
                    !=
                    expected_train
                ):

                    raise RuntimeError(
                        f"{feature_set} "
                        f"{evolution_name} "
                        f"{protocol}: "
                        "wrong training count."
                    )

                if (
                    n_val != 365
                    or
                    n_test != 232
                ):

                    raise RuntimeError(
                        f"{feature_set} "
                        f"{evolution_name} "
                        f"{protocol}: "
                        "wrong val/test count."
                    )

                # --------------------------------------------
                # Data arrays
                # --------------------------------------------

                X_all = (
                    table[
                        QRC_FEATURE_COLUMNS
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                y_all = (
                    table[
                        "target"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                X_train = (
                    X_all[
                        train_mask
                    ]
                )

                y_train = (
                    y_all[
                        train_mask
                    ]
                )

                X_val = (
                    X_all[
                        val_mask
                    ]
                )

                y_val = (
                    y_all[
                        val_mask
                    ]
                )

                configuration = (
                    f"{feature_set}_"
                    f"{evolution_name}_"
                    f"{protocol}"
                )

                # --------------------------------------------
                # Geometry
                # --------------------------------------------

                geometry = (
                    feature_geometry(
                        X_train
                    )
                )

                all_geometry.append(
                    {
                        "feature_set": (
                            feature_set
                        ),

                        "evolution": (
                            evolution_name
                        ),

                        "r": (
                            np.nan
                            if r is None
                            else r
                        ),

                        "protocol": (
                            protocol
                        ),

                        "configuration": (
                            configuration
                        ),

                        "train_rows": (
                            n_train
                        ),

                        **geometry,
                    }
                )

                # --------------------------------------------
                # Training-only chronological CV
                # --------------------------------------------

                (
                    best_lambda,
                    best_cv_mean,
                    best_cv_std,
                    cv_df,
                    best_fold_df,
                ) = ridge_cv(
                    X=X_train,
                    y=y_train,
                    configuration=(
                        configuration
                    ),
                )

                cv_df[
                    "feature_set"
                ] = feature_set

                cv_df[
                    "evolution"
                ] = evolution_name

                cv_df[
                    "r"
                ] = (
                    np.nan
                    if r is None
                    else r
                )

                cv_df[
                    "protocol"
                ] = protocol

                all_cv.append(
                    cv_df
                )

                # Best-lambda fold-level data
                best_fold_df[
                    "feature_set"
                ] = feature_set

                best_fold_df[
                    "evolution"
                ] = evolution_name

                best_fold_df[
                    "r"
                ] = (
                    np.nan
                    if r is None
                    else r
                )

                best_fold_df[
                    "protocol"
                ] = protocol

                best_fold_df[
                    "selected_lambda"
                ] = best_lambda

                all_best_folds.append(
                    best_fold_df
                )

                # --------------------------------------------
                # Validation diagnostic
                # --------------------------------------------

                metrics = (
                    validation_metrics(
                        X_train=(
                            X_train
                        ),

                        y_train=(
                            y_train
                        ),

                        X_val=(
                            X_val
                        ),

                        y_val=(
                            y_val
                        ),

                        best_lambda=(
                            best_lambda
                        ),
                    )
                )

                resources = (
                    resource_values(
                        evolution=(
                            evolution_name
                        ),
                        r=r,
                    )
                )

                all_results.append(
                    {
                        "feature_set": (
                            feature_set
                        ),

                        "evolution": (
                            evolution_name
                        ),

                        "r": (
                            np.nan
                            if r is None
                            else r
                        ),

                        "protocol": (
                            protocol
                        ),

                        "configuration": (
                            configuration
                        ),

                        "alpha": (
                            ALPHA
                        ),

                        "j_mult": (
                            J_MULT
                        ),

                        "hx": (
                            H_X
                        ),

                        "dt": (
                            DELTA_T
                        ),

                        "train_rows": (
                            n_train
                        ),

                        "validation_rows": (
                            n_val
                        ),

                        "test_rows": (
                            n_test
                        ),

                        "best_lambda": (
                            best_lambda
                        ),

                        "cv_rmse_mean": (
                            best_cv_mean
                        ),

                        "cv_rmse_std": (
                            best_cv_std
                        ),

                        **resources,

                        **metrics,
                    }
                )

                log(
                    f"{configuration:<26s} "
                    f"CV="
                    f"{best_cv_mean:.6f} "
                    f"(+/- "
                    f"{best_cv_std:.6f}) "
                    f"lambda="
                    f"{best_lambda:<8g} "
                    f"rank="
                    f"{geometry['rank']:2d}/12 "
                    f"Val="
                    f"{metrics['RMSE']:.6f}"
                )

    # ========================================================
    # Combine / save raw results
    # ========================================================

    cv_all = pd.concat(
        all_cv,
        ignore_index=True,
    )

    folds_all = pd.concat(
        all_best_folds,
        ignore_index=True,
    )

    results = pd.DataFrame(
        all_results
    )

    geometry_df = pd.DataFrame(
        all_geometry
    )

    purity_all = pd.concat(
        all_purity,
        ignore_index=True,
    )

    cv_all.to_csv(
        CV_OUTPUT,
        index=False,
    )

    folds_all.to_csv(
        FOLD_OUTPUT,
        index=False,
    )

    results.to_csv(
        RESULT_OUTPUT,
        index=False,
    )

    geometry_df.to_csv(
        GEOMETRY_OUTPUT,
        index=False,
    )

    purity_all.to_csv(
        PURITY_OUTPUT,
        index=False,
    )

    # ========================================================
    # Reproduce previous 6.6A F4 result
    # ========================================================

    audit_against_66a(
        results
    )

    # ========================================================
    # Best temporal protocol within each
    # feature-set x evolution pair
    # ========================================================

    resource_rows = []

    log()
    log("=" * 80)

    log(
        "BEST TEMPORAL PROTOCOL "
        "WITHIN EACH FEATURE SET x DEPTH"
    )

    log("=" * 80)

    for feature_set in (
        FEATURE_SETS
    ):

        log()
        log(
            f"--- {feature_set} ---"
        )

        for (
            evolution_name,
            r,
            _,
        ) in evolution_settings:

            subset = (
                results[
                    (
                        results[
                            "feature_set"
                        ]
                        == feature_set
                    )
                    &
                    (
                        results[
                            "evolution"
                        ]
                        == evolution_name
                    )
                ]
                .sort_values(
                    "cv_rmse_mean"
                )
                .reset_index(
                    drop=True
                )
            )

            best = (
                subset.iloc[0]
            )

            resource_rows.append(
                best.to_dict()
            )

            if (
                evolution_name
                == "EXACT"
            ):

                cost_text = (
                    "physics reference"
                )

            else:

                cost_text = (
                    f"~CX="
                    f"{int(best['approx_CX'])}"
                )

            log(
                f"{evolution_name:<6s}: "
                f"{best['protocol']:<8s} "
                f"CV="
                f"{best['cv_rmse_mean']:.6f} "
                f"(+/- "
                f"{best['cv_rmse_std']:.6f}) "
                f"Val="
                f"{best['RMSE']:.6f} "
                f"{cost_text}"
            )

    resource_df = pd.DataFrame(
        resource_rows
    )

    resource_df.to_csv(
        RESOURCE_OUTPUT,
        index=False,
    )

    # ========================================================
    # Pareto analysis
    # ========================================================

    pareto_df = (
        calculate_pareto_front(
            resource_df
        )
    )

    pareto_df.to_csv(
        PARETO_OUTPUT,
        index=False,
    )

    log()
    log("=" * 80)
    log(
        "RESOURCE / TASK PARETO FRONT"
    )
    log("=" * 80)

    for feature_set in (
        FEATURE_SETS
    ):

        log()
        log(
            f"--- {feature_set} ---"
        )

        subset = (
            pareto_df[
                (
                    pareto_df[
                        "feature_set"
                    ]
                    == feature_set
                )
                &
                (
                    pareto_df[
                        "is_pareto"
                    ]
                )
            ]
            .sort_values(
                "approx_CX"
            )
        )

        for _, row in (
            subset.iterrows()
        ):

            log(
                f"{row['evolution']:<5s} "
                f"{row['protocol']:<8s} "
                f"CV="
                f"{row['cv_rmse_mean']:.6f} "
                f"~CX="
                f"{int(row['approx_CX'])}"
            )

    # ========================================================
    # Overall best per feature set
    # ========================================================

    log()
    log("=" * 80)

    log(
        "PRIMARY RESULT BY FEATURE SET"
    )

    log(
        "PRIMARY RANKING = TRAINING-ONLY CV"
    )

    log("=" * 80)

    for feature_set in (
        FEATURE_SETS
    ):

        subset = (
            results[
                results[
                    "feature_set"
                ]
                == feature_set
            ]
            .sort_values(
                "cv_rmse_mean"
            )
            .reset_index(
                drop=True
            )
        )

        best = (
            subset.iloc[0]
        )

        log()

        log(
            f"{feature_set}:"
        )

        log(
            "  Best configuration = "
            f"{best['configuration']}"
        )

        log(
            "  CV RMSE = "
            f"{best['cv_rmse_mean']:.6f} "
            f"(+/- "
            f"{best['cv_rmse_std']:.6f})"
        )

        log(
            "  Validation RMSE = "
            f"{best['RMSE']:.6f}"
        )

        if (
            best[
                "evolution"
            ]
            != "EXACT"
        ):

            log(
                "  Approx. CX = "
                f"{int(best['approx_CX'])}"
            )

        else:

            log(
                "  Exact evolution = "
                "physics reference only"
            )

    # ========================================================
    # r=2 robustness summary
    # ========================================================

    log()
    log("=" * 80)
    log(
        "R=2 ROBUSTNESS CHECK"
    )
    log("=" * 80)

    for feature_set in (
        FEATURE_SETS
    ):

        finite = (
            resource_df[
                (
                    resource_df[
                        "feature_set"
                    ]
                    == feature_set
                )
                &
                (
                    resource_df[
                        "evolution"
                    ]
                    != "EXACT"
                )
            ]
            .sort_values(
                "cv_rmse_mean"
            )
            .reset_index(
                drop=True
            )
        )

        best_finite = (
            finite.iloc[0]
        )

        r2_row = (
            finite[
                finite[
                    "evolution"
                ]
                == "r2"
            ]
            .iloc[0]
        )

        delta = (
            r2_row[
                "cv_rmse_mean"
            ]
            -
            best_finite[
                "cv_rmse_mean"
            ]
        )

        log()
        log(
            f"{feature_set}:"
        )

        log(
            "  Best finite depth = "
            f"{best_finite['evolution']} "
            f"({best_finite['protocol']})"
        )

        log(
            "  Best finite CV = "
            f"{best_finite['cv_rmse_mean']:.6f}"
        )

        log(
            "  r2 best protocol = "
            f"{r2_row['protocol']}"
        )

        log(
            "  r2 CV = "
            f"{r2_row['cv_rmse_mean']:.6f}"
        )

        log(
            "  r2 - best delta = "
            f"{delta:+.6f}"
        )

    # ========================================================
    # Overall full ranking
    # ========================================================

    ranking = (
        results
        .sort_values(
            "cv_rmse_mean"
        )
        .reset_index(
            drop=True
        )
    )

    log()
    log("=" * 80)

    log(
        "TOP 30 OVERALL CONFIGURATIONS"
    )

    log("=" * 80)
    log()

    for i, row in (
        ranking.head(
            30
        ).iterrows()
    ):

        log(
            f"{i + 1:2d}. "
            f"{row['configuration']:<27s} "
            f"CV="
            f"{row['cv_rmse_mean']:.6f} "
            f"(+/- "
            f"{row['cv_rmse_std']:.6f}) "
            f"Val="
            f"{row['RMSE']:.6f}"
        )

    # ========================================================
    # Final notes
    # ========================================================

    log()
    log("=" * 80)
    log(
        "INTERPRETATION RULES FOR 6.6B"
    )
    log("=" * 80)

    log()

    log(
        "1. F4 was optimized previously; "
        "F2/F3 are robustness checks."
    )

    log(
        "2. The six-qubit topology is "
        "identical for F2/F3/F4."
    )

    log(
        "3. Missing variables are switched "
        "off by zero input rotations."
    )

    log(
        "4. Exact evolution is only a "
        "physics reference."
    )

    log(
        "5. Primary model selection remains "
        "training-only chronological CV."
    )

    log(
        "6. Validation 2025 remains "
        "diagnostic."
    )

    log(
        "7. Test 2026 remains untouched."
    )

    log(
        "8. Prefer shallow depth when a "
        "deeper configuration is dominated "
        "in both RMSE and gate cost."
    )

    log(
        "9. Fold-level results are saved "
        "for direct robustness analysis."
    )

    log(
        "10. The ideal small alpha may later "
        "need reconsideration under hardware "
        "shot/device noise."
    )

    # ========================================================
    # Write summary
    # ========================================================

    SUMMARY_OUTPUT.write_text(
        "\n".join(
            output_lines
        ),
        encoding="utf-8",
    )

    print()

    print(
        f"Saved Trotter errors to: "
        f"{ERROR_OUTPUT}"
    )

    print(
        f"Saved CV results to: "
        f"{CV_OUTPUT}"
    )

    print(
        f"Saved fold-level CV to: "
        f"{FOLD_OUTPUT}"
    )

    print(
        f"Saved validation results to: "
        f"{RESULT_OUTPUT}"
    )

    print(
        f"Saved geometry to: "
        f"{GEOMETRY_OUTPUT}"
    )

    print(
        f"Saved purity to: "
        f"{PURITY_OUTPUT}"
    )

    print(
        f"Saved resource summary to: "
        f"{RESOURCE_OUTPUT}"
    )

    print(
        f"Saved Pareto front to: "
        f"{PARETO_OUTPUT}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":

    main()