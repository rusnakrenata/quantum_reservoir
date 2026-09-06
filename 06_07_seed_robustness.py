from pathlib import Path

import numpy as np
import pandas as pd

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

REFERENCE_COUPLING_FILE = (
    RESULTS_DIR
    / "05_03a_reservoir_couplings.csv"
)

PER_SEED_OUTPUT = (
    RESULTS_DIR
    / "06_07_seed_results.csv"
)

CV_OUTPUT = (
    RESULTS_DIR
    / "06_07_seed_cv.csv"
)

AGGREGATE_OUTPUT = (
    RESULTS_DIR
    / "06_07_seed_aggregate.csv"
)

PAIR_OUTPUT = (
    RESULTS_DIR
    / "06_07_r1_r2_paired.csv"
)

COUPLING_OUTPUT = (
    RESULTS_DIR
    / "06_07_seed_couplings.csv"
)

GEOMETRY_OUTPUT = (
    RESULTS_DIR
    / "06_07_seed_geometry.csv"
)

PURITY_OUTPUT = (
    RESULTS_DIR
    / "06_07_seed_purity.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "06_07_summary.txt"
)


# ============================================================
# Frozen final ideal-QRC architecture
# ============================================================

N_INPUT = 4
N_MEMORY = 2
N_QUBITS = 6

D_INPUT = 2 ** N_INPUT
D_MEMORY = 2 ** N_MEMORY
D_TOTAL = 2 ** N_QUBITS

ALPHA = 0.005

H_X = 0.5

DELTA_T = 0.8

CLAIM_CLIP_Z = 3.0


# ============================================================
# Seeds
# ============================================================

SEEDS = [
    42,
    101,
    202,
    505,
    707,
]


# ============================================================
# Frozen chain topology
# ============================================================

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]

J_LOW = -0.7
J_HIGH = +0.7


# ============================================================
# Trotter candidates
# ============================================================

TROTTER_DEPTHS = [
    1,
    2,
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
# QRC readout
# ============================================================

FEATURE_COLUMNS = [
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
# Target
# ============================================================

def resolve_target_column(df):

    for candidate in TARGET_CANDIDATES:

        if candidate in df.columns:

            return candidate

    raise ValueError(
        "Could not resolve target column."
    )


# ============================================================
# Softsign
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
# Ry
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
# Frozen F4 encoding
# ============================================================

def encode_input_state(row):

    # Claims
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

    # Weekday
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

    # Policy
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

    # Holiday
    theta_holiday = (
        np.pi
        *
        float(
            row[
                "is_public_holiday_t_plus_1"
            ]
        )
    )

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
# Initial memory
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
# Seeded reservoir couplings
#
# IMPORTANT:
# np.random.default_rng(42).uniform(-0.7, 0.7, 5)
# reproduces the original Week-5 couplings.
# ============================================================

def generate_couplings(seed):

    rng = np.random.default_rng(
        seed
    )

    values = rng.uniform(
        J_LOW,
        J_HIGH,
        size=len(
            EDGES
        ),
    )

    return {
        edge: float(value)

        for edge, value
        in zip(
            EDGES,
            values,
        )
    }


# ============================================================
# Audit seed 42
# ============================================================

def audit_seed_42(
    couplings,
):

    if not (
        REFERENCE_COUPLING_FILE.exists()
    ):

        log(
            "Seed-42 reference file "
            "not found; audit skipped."
        )

        return

    reference = pd.read_csv(
        REFERENCE_COUPLING_FILE
    )

    errors = []

    for _, row in (
        reference.iterrows()
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

        expected = float(
            row[
                "J_ij"
            ]
        )

        generated = (
            couplings[
                edge
            ]
        )

        errors.append(
            abs(
                generated
                -
                expected
            )
        )

    max_error = float(
        np.max(
            errors
        )
    )

    log(
        "Seed-42 coupling "
        "reproduction error = "
        f"{max_error:.3e}"
    )

    if max_error > 1e-10:

        raise RuntimeError(
            "Seed-42 coupling "
            "audit FAILED."
        )


# ============================================================
# Single-qubit operator
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

        matrices.append(
            pauli
            if q == target_q
            else I2
        )

    result = matrices[0]

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
# Reservoir unitary
# ============================================================

def build_unitary(
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

    for _ in range(r):

        # ZZ layer
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

        # X layer
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
# Partial trace
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
# Direct step for audit
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
# Superoperator
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
# Feature map
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
                FEATURE_COLUMNS
            ),
            D_MEMORY ** 2,
        )
    )


# ============================================================
# Generate all temporal protocols
# ============================================================

def generate_protocols(
    df,
    target_column,
    input_states,
    seed,
    r,
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

    rho0 = initial_memory()

    vec0 = (
        rho0.reshape(
            -1
        )
    )

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
    # Chronological loop
    # ========================================================

    for t, phi in enumerate(
        input_states
    ):

        kraus = build_kraus(
            phi,
            U4,
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
            "seed": seed,
            "r": r,
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
            FEATURE_COLUMNS,
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
        ] = (
            propagated
        )

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

            row = {
                "seed": seed,
                "r": r,
                "protocol": (
                    f"RWP_W{W}"
                ),
                "window_W": W,
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
                FEATURE_COLUMNS,
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
        # Direct correctness audits
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
                previous_rho,
                phi,
                U,
                observables,
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

        cont_vec = (
            cont_new
        )

    tables = {
        protocol: pd.DataFrame(
            values
        )
        for protocol, values
        in rows.items()
    }

    purity_rows = []

    for protocol, values in (
        purities.items()
    ):

        values = np.asarray(
            values,
            dtype=float,
        )

        purity_rows.append(
            {
                "seed": seed,
                "r": r,
                "protocol": protocol,
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
# Ridge
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
                    alpha=ridge_lambda,
                    fit_intercept=True,
                ),
            ),
        ]
    )


def ridge_cv(
    X,
    y,
    configuration,
):

    splitter = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )

    rows = []

    for ridge_lambda in (
        RIDGE_LAMBDAS
    ):

        fold_rmse = []

        for train_idx, val_idx in (
            splitter.split(
                X
            )
        ):

            model = make_model(
                ridge_lambda
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

        rows.append(
            {
                "configuration":
                    configuration,

                "lambda":
                    ridge_lambda,

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

    result = pd.DataFrame(
        rows
    )

    best = (
        result
        .sort_values(
            [
                "cv_rmse_mean",
                "lambda",
            ]
        )
        .iloc[0]
    )

    return (
        float(
            best[
                "lambda"
            ]
        ),
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
        result,
    )


# ============================================================
# Geometry
# ============================================================

def feature_geometry(
    X_train,
):

    raw_std = np.std(
        X_train,
        axis=0,
        ddof=0,
    )

    Xs = (
        StandardScaler()
        .fit_transform(
            X_train
        )
    )

    rank = int(
        np.linalg.matrix_rank(
            Xs
        )
    )

    singular = np.linalg.svd(
        Xs,
        compute_uv=False,
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

    return {
        "rank": rank,

        "min_raw_std": float(
            np.min(
                raw_std
            )
        ),

        "max_raw_std": float(
            np.max(
                raw_std
            )
        ),

        "near_zero_std_count": int(
            np.sum(
                raw_std
                <
                1e-10
            )
        ),

        "kappa_X": (
            kappa
        ),
    }


# ============================================================
# Validation
# ============================================================

def validation_metrics(
    X_train,
    y_train,
    X_val,
    y_val,
    best_lambda,
):

    model = make_model(
        best_lambda
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

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "NRMSE": (
            rmse
            /
            train_std
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 6 - STEP 6.7")
    log("SEED ROBUSTNESS + FINAL IDEAL-QRC FREEZE")
    log("=" * 80)

    log()

    log(
        "Frozen:"
    )

    log(
        f"  F4"
    )

    log(
        f"  alpha = {ALPHA}"
    )

    log(
        f"  hx = {H_X}"
    )

    log(
        f"  dt = {DELTA_T}"
    )

    log(
        "  n_memory = 2"
    )

    log(
        "  softsign policy"
    )

    log(
        f"  r = {TROTTER_DEPTHS}"
    )

    log(
        f"  seeds = {SEEDS}"
    )

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

    if len(df) != 1692:

        raise RuntimeError(
            "Expected 1692 samples."
        )

    target_column = (
        resolve_target_column(
            df
        )
    )

    input_states = [
        encode_input_state(
            row
        )

        for _, row
        in df.iterrows()
    ]

    observables = (
        build_observables()
    )

    all_results = []
    all_cv = []
    all_geometry = []
    all_purity = []
    coupling_rows = []

    # ========================================================
    # Seed loop
    # ========================================================

    for seed in SEEDS:

        log()
        log("#" * 80)

        log(
            f"RESERVOIR SEED = {seed}"
        )

        log("#" * 80)

        couplings = (
            generate_couplings(
                seed
            )
        )

        if seed == 42:

            audit_seed_42(
                couplings
            )

        # Save couplings
        for edge, J in (
            couplings.items()
        ):

            coupling_rows.append(
                {
                    "seed": seed,
                    "q_i": edge[0],
                    "q_j": edge[1],
                    "J_ij": J,
                }
            )

        log(
            "Couplings:"
        )

        for edge, J in (
            couplings.items()
        ):

            log(
                f"  J{edge} = "
                f"{J:+.6f}"
            )

        # ====================================================
        # r loop
        # ====================================================

        for r in TROTTER_DEPTHS:

            log()
            log("=" * 80)

            log(
                f"SEED={seed} | r={r}"
            )

            log("=" * 80)

            U = build_unitary(
                couplings,
                r,
            )

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
                    input_states
                ),
                seed=seed,
                r=r,
                U=U,
                observables=(
                    observables
                ),
            )

            log(
                "Direct feature audit = "
                f"{feature_error:.3e}"
            )

            log(
                "Direct memory audit  = "
                f"{memory_error:.3e}"
            )

            if (
                feature_error > 1e-10
                or
                memory_error > 1e-10
            ):

                raise RuntimeError(
                    "Correctness audit FAILED."
                )

            all_purity.append(
                purity_df
            )

            # ================================================
            # Protocol loop
            # ================================================

            for protocol, table in (
                tables.items()
            ):

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
                    or
                    n_val != 365
                    or
                    n_test != 232
                ):

                    raise RuntimeError(
                        "Split-count audit FAILED."
                    )

                X_all = (
                    table[
                        FEATURE_COLUMNS
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
                    f"s{seed}_"
                    f"r{r}_"
                    f"{protocol}"
                )

                geometry = (
                    feature_geometry(
                        X_train
                    )
                )

                all_geometry.append(
                    {
                        "seed": seed,
                        "r": r,
                        "protocol": protocol,
                        "configuration":
                            configuration,
                        **geometry,
                    }
                )

                (
                    best_lambda,
                    cv_mean,
                    cv_std,
                    cv_df,
                ) = ridge_cv(
                    X_train,
                    y_train,
                    configuration,
                )

                cv_df[
                    "seed"
                ] = seed

                cv_df[
                    "r"
                ] = r

                cv_df[
                    "protocol"
                ] = protocol

                all_cv.append(
                    cv_df
                )

                metrics = (
                    validation_metrics(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        best_lambda,
                    )
                )

                all_results.append(
                    {
                        "seed": seed,
                        "r": r,
                        "protocol": protocol,
                        "configuration":
                            configuration,

                        "train_rows": n_train,
                        "validation_rows": n_val,
                        "test_rows": n_test,

                        "best_lambda":
                            best_lambda,

                        "cv_rmse_mean":
                            cv_mean,

                        "cv_rmse_std":
                            cv_std,

                        **metrics,
                    }
                )

                log(
                    f"{configuration:<22s} "
                    f"CV="
                    f"{cv_mean:.6f} "
                    f"(+/- "
                    f"{cv_std:.6f}) "
                    f"lambda="
                    f"{best_lambda:<8g} "
                    f"rank="
                    f"{geometry['rank']:2d}/12 "
                    f"Val="
                    f"{metrics['RMSE']:.6f}"
                )

    # ========================================================
    # Save raw outputs
    # ========================================================

    results = pd.DataFrame(
        all_results
    )

    cv_all = pd.concat(
        all_cv,
        ignore_index=True,
    )

    geometry_df = pd.DataFrame(
        all_geometry
    )

    purity_df = pd.concat(
        all_purity,
        ignore_index=True,
    )

    coupling_df = pd.DataFrame(
        coupling_rows
    )

    results.to_csv(
        PER_SEED_OUTPUT,
        index=False,
    )

    cv_all.to_csv(
        CV_OUTPUT,
        index=False,
    )

    geometry_df.to_csv(
        GEOMETRY_OUTPUT,
        index=False,
    )

    purity_df.to_csv(
        PURITY_OUTPUT,
        index=False,
    )

    coupling_df.to_csv(
        COUPLING_OUTPUT,
        index=False,
    )

    # ========================================================
    # Aggregate across seeds
    # ========================================================

    aggregate = (
        results
        .groupby(
            [
                "r",
                "protocol",
            ],
            as_index=False,
        )
        .agg(
            seed_count=(
                "seed",
                "count",
            ),

            cv_rmse_seed_mean=(
                "cv_rmse_mean",
                "mean",
            ),

            cv_rmse_seed_std=(
                "cv_rmse_mean",
                "std",
            ),

            cv_rmse_seed_min=(
                "cv_rmse_mean",
                "min",
            ),

            cv_rmse_seed_max=(
                "cv_rmse_mean",
                "max",
            ),

            validation_rmse_mean=(
                "RMSE",
                "mean",
            ),

            validation_rmse_std=(
                "RMSE",
                "std",
            ),

            validation_bias_mean=(
                "Bias",
                "mean",
            ),
        )
        .sort_values(
            "cv_rmse_seed_mean"
        )
        .reset_index(
            drop=True
        )
    )

    aggregate.to_csv(
        AGGREGATE_OUTPUT,
        index=False,
    )

    # ========================================================
    # Paired r=1 vs r=2 comparison
    # ========================================================

    pivot = (
        results
        .pivot_table(
            index=[
                "seed",
                "protocol",
            ],
            columns="r",
            values="cv_rmse_mean",
        )
        .reset_index()
    )

    pivot.columns.name = None

    pivot = pivot.rename(
        columns={
            1: "cv_r1",
            2: "cv_r2",
        }
    )

    pivot[
        "delta_r2_minus_r1"
    ] = (
        pivot[
            "cv_r2"
        ]
        -
        pivot[
            "cv_r1"
        ]
    )

    pivot[
        "r2_better"
    ] = (
        pivot[
            "delta_r2_minus_r1"
        ]
        <
        0
    )

    pivot.to_csv(
        PAIR_OUTPUT,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    log()
    log("=" * 80)

    log(
        "AGGREGATE ACROSS RESERVOIR SEEDS"
    )

    log(
        "PRIMARY = MEAN TRAINING-ONLY CV "
        "ACROSS SEEDS"
    )

    log("=" * 80)

    log()

    for i, row in (
        aggregate.iterrows()
    ):

        log(
            f"{i + 1:2d}. "
            f"r={int(row['r'])} "
            f"{row['protocol']:<8s} "
            f"CV="
            f"{row['cv_rmse_seed_mean']:.6f} "
            f"(seed std="
            f"{row['cv_rmse_seed_std']:.6f}) "
            f"Val="
            f"{row['validation_rmse_mean']:.6f} "
            f"(seed std="
            f"{row['validation_rmse_std']:.6f})"
        )

    # ========================================================
    # Best per seed
    # ========================================================

    log()
    log("=" * 80)
    log(
        "BEST CONFIGURATION WITHIN EACH SEED"
    )
    log("=" * 80)

    for seed in SEEDS:

        subset = (
            results[
                results[
                    "seed"
                ]
                == seed
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
            f"seed {seed}: "
            f"r={int(best['r'])} "
            f"{best['protocol']} "
            f"CV="
            f"{best['cv_rmse_mean']:.6f} "
            f"Val="
            f"{best['RMSE']:.6f}"
        )

    # ========================================================
    # W1 paired depth comparison
    # ========================================================

    log()
    log("=" * 80)
    log(
        "PAIRED r=1 vs r=2 FOR W=1"
    )
    log("=" * 80)

    w1 = (
        pivot[
            pivot[
                "protocol"
            ]
            == "RWP_W1"
        ]
        .copy()
    )

    for _, row in (
        w1.iterrows()
    ):

        log(
            f"seed {int(row['seed'])}: "
            f"r1="
            f"{row['cv_r1']:.6f} "
            f"r2="
            f"{row['cv_r2']:.6f} "
            f"delta="
            f"{row['delta_r2_minus_r1']:+.6f}"
        )

    mean_delta = float(
        w1[
            "delta_r2_minus_r1"
        ].mean()
    )

    std_delta = float(
        w1[
            "delta_r2_minus_r1"
        ].std(
            ddof=1
        )
    )

    r2_wins = int(
        w1[
            "r2_better"
        ].sum()
    )

    log()

    log(
        "W1 paired summary:"
    )

    log(
        "  mean(r2-r1) = "
        f"{mean_delta:+.6f}"
    )

    log(
        "  std(r2-r1)  = "
        f"{std_delta:.6f}"
    )

    log(
        "  r2 wins = "
        f"{r2_wins}/{len(w1)} seeds"
    )

    # ========================================================
    # Final current winner
    # ========================================================

    best = (
        aggregate.iloc[0]
    )

    log()
    log("=" * 80)
    log(
        "FINAL WEEK-6 IDEAL-QRC CANDIDATE"
    )
    log("=" * 80)

    log()

    log(
        f"r = "
        f"{int(best['r'])}"
    )

    log(
        f"protocol = "
        f"{best['protocol']}"
    )

    log(
        "mean seed CV RMSE = "
        f"{best['cv_rmse_seed_mean']:.6f}"
    )

    log(
        "seed-to-seed std = "
        f"{best['cv_rmse_seed_std']:.6f}"
    )

    log()

    log(
        "Frozen architecture entering "
        "Week 7:"
    )

    log(
        "  Feature set: F4"
    )

    log(
        "  Policy mapping: softsign"
    )

    log(
        "  Memory qubits: 2"
    )

    log(
        f"  alpha: {ALPHA}"
    )

    log(
        f"  hx: {H_X}"
    )

    log(
        f"  dt: {DELTA_T}"
    )

    log(
        "  Candidate Trotter depths: "
        "r={1,2}"
    )

    log(
        "  Primary ideal depth/protocol "
        "will be frozen after interpreting "
        "this seed-robustness output."
    )

    log(
        "  2025 validation diagnostic only."
    )

    log(
        "  2026 test remains untouched."
    )

    SUMMARY_OUTPUT.write_text(
        "\n".join(
            output_lines
        ),
        encoding="utf-8",
    )

    print()

    print(
        f"Saved per-seed results to: "
        f"{PER_SEED_OUTPUT}"
    )

    print(
        f"Saved CV grid to: "
        f"{CV_OUTPUT}"
    )

    print(
        f"Saved aggregate results to: "
        f"{AGGREGATE_OUTPUT}"
    )

    print(
        f"Saved paired r1/r2 results to: "
        f"{PAIR_OUTPUT}"
    )

    print(
        f"Saved couplings to: "
        f"{COUPLING_OUTPUT}"
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
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":

    main()