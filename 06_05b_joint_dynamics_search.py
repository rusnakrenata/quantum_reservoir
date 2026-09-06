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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = (
    RESULTS_DIR
    / "03_01_preprocessed_samples.csv"
)

COUPLING_FILE = (
    RESULTS_DIR
    / "05_03a_reservoir_couplings.csv"
)

SOFTSIGN_REFERENCE_FILE = (
    RESULTS_DIR
    / "06_04b_features_softsign3.csv"
)

CV_OUTPUT = (
    RESULTS_DIR
    / "06_05b_joint_dynamics_cv.csv"
)

RESULT_OUTPUT = (
    RESULTS_DIR
    / "06_05b_joint_dynamics_validation.csv"
)

GEOMETRY_OUTPUT = (
    RESULTS_DIR
    / "06_05b_joint_dynamics_geometry.csv"
)

PURITY_OUTPUT = (
    RESULTS_DIR
    / "06_05b_joint_dynamics_purity.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "06_05b_summary.txt"
)


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

MAX_WINDOW = max(WINDOWS)


# ============================================================
# Frozen QRC architecture
# ============================================================

N_INPUT = 4
N_MEMORY = 2
N_QUBITS = 6

D_INPUT = 2 ** N_INPUT
D_MEMORY = 2 ** N_MEMORY
D_TOTAL = 2 ** N_QUBITS

H_X = 0.5
DELTA_T = 0.8
TROTTER_STEPS = 2

CLAIM_CLIP_Z = 3.0


# ============================================================
# Joint search
# ============================================================

ALPHAS = [
    0.0,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.50,
    0.75,
]

J_MULTIPLIERS = [
    1.0,
    1.5,
    2.0,
]


def make_config_name(
    alpha,
    j_mult,
):

    alpha_text = str(
        alpha
    ).replace(
        ".",
        "p",
    )

    j_text = str(
        j_mult
    ).replace(
        ".",
        "p",
    )

    return (
        f"a{alpha_text}_"
        f"J{j_text}"
    )


DYNAMICS_CONFIGS = []

for alpha in ALPHAS:

    for j_mult in J_MULTIPLIERS:

        DYNAMICS_CONFIGS.append(
            {
                "name": (
                    make_config_name(
                        alpha,
                        j_mult,
                    )
                ),
                "alpha": alpha,
                "j_mult": j_mult,
            }
        )


# Add original Week-5/6.4B reference
DYNAMICS_CONFIGS.append(
    {
        "name": "baseline_a1_J1",
        "alpha": 1.0,
        "j_mult": 1.0,
    }
)


# ============================================================
# Ridge grid frozen from 6.1
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
# Observables
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
# Pauli matrices
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
# Target column
# ============================================================

def resolve_target_column(df):

    for candidate in TARGET_CANDIDATES:

        if candidate in df.columns:

            return candidate

    raise ValueError(
        "Could not resolve target column."
    )


# ============================================================
# Selected softsign policy encoding
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
# F4 encoding
# ============================================================

def encode_input_state(
    row,
    alpha,
):

    # --------------------------------------------------------
    # Claim
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
        alpha
        *
        claim_encoded
    )

    # --------------------------------------------------------
    # Day of week
    #
    # Keep its natural cyclic angle unchanged.
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
    # Policy
    # --------------------------------------------------------

    policy_z = float(
        row[
            "P_t_z"
        ]
    )

    theta_policy = (
        alpha
        *
        softsign3(
            policy_z
        )
    )

    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

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

    # Qiskit order:
    #
    # q3 kron q2 kron q1 kron q0

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
# Couplings
# ============================================================

def load_base_couplings():

    df = pd.read_csv(
        COUPLING_FILE
    )

    couplings = {}

    for _, row in (
        df.iterrows()
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
        ] = float(
            row[
                "J_ij"
            ]
        )

    return couplings


# ============================================================
# Reservoir unitary
# ============================================================

def build_unitary(
    base_couplings,
    j_mult,
):

    qc = QuantumCircuit(
        N_QUBITS
    )

    dt_step = (
        DELTA_T
        /
        TROTTER_STEPS
    )

    for _ in range(
        TROTTER_STEPS
    ):

        # ----------------------------------------------------
        # ZZ interactions
        # ----------------------------------------------------

        for (
            q_i,
            q_j,
        ), J_base in (
            base_couplings.items()
        ):

            J_used = (
                j_mult
                *
                J_base
            )

            qc.rzz(
                2.0
                *
                J_used
                *
                dt_step,
                q_i,
                q_j,
            )

        # ----------------------------------------------------
        # Frozen transverse field
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

    operator = (
        matrices[0]
    )

    for matrix in (
        matrices[
            1:
        ]
    ):

        operator = np.kron(
            operator,
            matrix,
        )

    return operator


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
# Direct calculation
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
# Superoperator machinery
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
# Generate CONT + all W
# ============================================================

def generate_protocols(
    df,
    target_column,
    input_states,
    dynamics_name,
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
    # Daily chronological loop
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
            "dynamics": dynamics_name,
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
        # All RWP windows
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
            1:MAX_WINDOW + 1
        ] = (
            propagated
        )

        # ----------------------------------------------------
        # Store all valid W
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
                - W
                + 1
            )

            row = {
                "dynamics": (
                    dynamics_name
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
        # Fast/direct audit using W=7 state
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

        # ----------------------------------------------------
        # Continue CONT
        # ----------------------------------------------------

        cont_vec = (
            cont_new
        )

    tables = {
        protocol: pd.DataFrame(
            protocol_rows
        )
        for (
            protocol,
            protocol_rows
        ) in rows.items()
    }

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
                "dynamics": (
                    dynamics_name
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
# Ridge readout
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

        for (
            train_idx,
            val_idx,
        ) in splitter.split(
            X
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
# Feature geometry
# ============================================================

def feature_geometry(
    X_train,
):

    Xs = (
        StandardScaler()
        .fit_transform(
            X_train
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
        "rank": rank,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "kappa_X": kappa,
        "rho_C": rho_C,
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

    prediction = model.predict(
        X_val
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
# Baseline reference audit
# ============================================================

def reference_audit(
    baseline_tables,
):

    if not (
        SOFTSIGN_REFERENCE_FILE.exists()
    ):

        return {}

    reference = pd.read_csv(
        SOFTSIGN_REFERENCE_FILE
    )

    reference[
        "target_date"
    ] = pd.to_datetime(
        reference[
            "target_date"
        ]
    )

    errors = {}

    for protocol in [
        "CONT",
        "RWP_W7",
    ]:

        old = (
            reference[
                reference[
                    "protocol"
                ]
                == protocol
            ]
            .copy()
        )

        new = (
            baseline_tables[
                protocol
            ]
            .copy()
        )

        old[
            "target_date"
        ] = pd.to_datetime(
            old[
                "target_date"
            ]
        )

        new[
            "target_date"
        ] = pd.to_datetime(
            new[
                "target_date"
            ]
        )

        merged = old.merge(
            new,
            on="target_date",
            suffixes=(
                "_old",
                "_new",
            ),
        )

        differences = []

        for feature in (
            FEATURE_COLUMNS
        ):

            differences.append(
                np.max(
                    np.abs(
                        merged[
                            f"{feature}_old"
                        ]
                        -
                        merged[
                            f"{feature}_new"
                        ]
                    )
                )
            )

        errors[
            protocol
        ] = float(
            np.max(
                differences
            )
        )

    return errors


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 6 - STEP 6.5B")
    log("JOINT ALPHA x COUPLING SEARCH")
    log("F4 + SOFTSIGN + M=2 + ALL WINDOWS")
    log("=" * 80)
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

    if len(df) != 1692:

        raise RuntimeError(
            "Expected 1692 samples."
        )

    target_column = (
        resolve_target_column(
            df
        )
    )

    base_couplings = (
        load_base_couplings()
    )

    observables = (
        build_observables()
    )

    # --------------------------------------------------------
    # Cache inputs for each alpha
    # --------------------------------------------------------

    unique_alphas = sorted(
        {
            config[
                "alpha"
            ]
            for config
            in DYNAMICS_CONFIGS
        }
    )

    encoded_cache = {}

    for alpha in (
        unique_alphas
    ):

        encoded_cache[
            alpha
        ] = [
            encode_input_state(
                row=row,
                alpha=alpha,
            )
            for _, row
            in df.iterrows()
        ]

    all_cv = []
    all_results = []
    all_geometry = []
    all_purity = []

    # ========================================================
    # Dynamics loop
    # ========================================================

    for config in (
        DYNAMICS_CONFIGS
    ):

        name = (
            config[
                "name"
            ]
        )

        alpha = float(
            config[
                "alpha"
            ]
        )

        j_mult = float(
            config[
                "j_mult"
            ]
        )

        log()
        log("=" * 80)

        log(
            f"DYNAMICS = "
            f"{name}"
        )

        log("=" * 80)

        log(
            f"alpha  = "
            f"{alpha}"
        )

        log(
            f"J mult = "
            f"{j_mult}"
        )

        log(
            f"hx     = "
            f"{H_X}"
        )

        log(
            f"dt     = "
            f"{DELTA_T}"
        )

        # ----------------------------------------------------
        # Gate-angle diagnostics
        # ----------------------------------------------------

        used_J = np.asarray(
            [
                j_mult
                *
                J
                for J
                in base_couplings.values()
            ],
            dtype=float,
        )

        rzz_angles = (
            2.0
            *
            used_J
            *
            (
                DELTA_T
                /
                TROTTER_STEPS
            )
        )

        rx_angle = (
            2.0
            *
            H_X
            *
            (
                DELTA_T
                /
                TROTTER_STEPS
            )
        )

        log(
            "RZZ per-step range = "
            f"["
            f"{np.min(rzz_angles):+.6f}, "
            f"{np.max(rzz_angles):+.6f}"
            f"]"
        )

        log(
            f"RX per-step angle   = "
            f"{rx_angle:+.6f}"
        )

        # ----------------------------------------------------
        # Reservoir unitary
        # ----------------------------------------------------

        U = build_unitary(
            base_couplings=(
                base_couplings
            ),
            j_mult=(
                j_mult
            ),
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
            f"Unitarity error = "
            f"{unitarity_error:.3e}"
        )

        # ----------------------------------------------------
        # Generate features
        # ----------------------------------------------------

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
                encoded_cache[
                    alpha
                ]
            ),
            dynamics_name=(
                name
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
                f"{name}: "
                "superoperator audit FAILED."
            )

        # ----------------------------------------------------
        # Baseline reference
        # ----------------------------------------------------

        if (
            alpha == 1.0
            and
            j_mult == 1.0
        ):

            errors = (
                reference_audit(
                    tables
                )
            )

            if errors:

                log()

                log(
                    "6.4B baseline audit:"
                )

                for (
                    protocol,
                    error,
                ) in errors.items():

                    log(
                        f"  "
                        f"{protocol:<7s} "
                        f"max error = "
                        f"{error:.3e}"
                    )

                    if error > 1e-10:

                        raise RuntimeError(
                            "Baseline reference "
                            "audit FAILED."
                        )

        # ----------------------------------------------------
        # Purity
        # ----------------------------------------------------

        purity_df[
            "alpha"
        ] = alpha

        purity_df[
            "j_mult"
        ] = j_mult

        all_purity.append(
            purity_df
        )

        # ====================================================
        # Evaluate all temporal protocols
        # ====================================================

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

            if protocol == "CONT":

                expected_train = (
                    1095
                )

            else:

                W = int(
                    protocol.split(
                        "W"
                    )[-1]
                )

                expected_train = (
                    1095
                    -
                    (W - 1)
                )

            if (
                n_train
                !=
                expected_train
            ):

                raise RuntimeError(
                    f"{name}, "
                    f"{protocol}: "
                    "wrong train count."
                )

            if (
                n_val != 365
                or
                n_test != 232
            ):

                raise RuntimeError(
                    f"{name}, "
                    f"{protocol}: "
                    "wrong val/test count."
                )

            Xqrc = (
                table[
                    FEATURE_COLUMNS
                ]
                .to_numpy(
                    dtype=float
                )
            )

            y = (
                table[
                    "target"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            X_train = (
                Xqrc[
                    train_mask
                ]
            )

            y_train = (
                y[
                    train_mask
                ]
            )

            X_val = (
                Xqrc[
                    val_mask
                ]
            )

            y_val = (
                y[
                    val_mask
                ]
            )

            configuration = (
                f"{name}_"
                f"{protocol}"
            )

            # ------------------------------------------------
            # Geometry
            # ------------------------------------------------

            geometry = (
                feature_geometry(
                    X_train
                )
            )

            all_geometry.append(
                {
                    "dynamics": (
                        name
                    ),
                    "alpha": (
                        alpha
                    ),
                    "j_mult": (
                        j_mult
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

            # ------------------------------------------------
            # Training-only CV
            # ------------------------------------------------

            (
                best_lambda,
                best_cv_mean,
                best_cv_std,
                cv_df,
            ) = ridge_cv(
                X=X_train,
                y=y_train,
                configuration=(
                    configuration
                ),
            )

            cv_df[
                "dynamics"
            ] = name

            cv_df[
                "alpha"
            ] = alpha

            cv_df[
                "j_mult"
            ] = j_mult

            cv_df[
                "protocol"
            ] = protocol

            all_cv.append(
                cv_df
            )

            # ------------------------------------------------
            # 2025 validation diagnostic
            # ------------------------------------------------

            metrics = (
                validation_metrics(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    best_lambda=(
                        best_lambda
                    ),
                )
            )

            all_results.append(
                {
                    "dynamics": (
                        name
                    ),
                    "alpha": (
                        alpha
                    ),
                    "j_mult": (
                        j_mult
                    ),
                    "hx": (
                        H_X
                    ),
                    "dt": (
                        DELTA_T
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
                    **metrics,
                }
            )

            log(
                f"{configuration:<25s} "
                f"CV="
                f"{best_cv_mean:.6f} "
                f"(+/- "
                f"{best_cv_std:.6f}) "
                f"lambda="
                f"{best_lambda:<7g} "
                f"rank="
                f"{geometry['rank']:2d}/12 "
                f"Val="
                f"{metrics['RMSE']:.6f}"
            )

    # ========================================================
    # Save
    # ========================================================

    cv_all = pd.concat(
        all_cv,
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
    # Best temporal protocol for every alpha/J pair
    # ========================================================

    log()
    log("=" * 80)

    log(
        "BEST TEMPORAL PROTOCOL "
        "FOR EACH ALPHA x J SETTING"
    )

    log("=" * 80)

    pair_best_rows = []

    for config in (
        DYNAMICS_CONFIGS
    ):

        name = (
            config[
                "name"
            ]
        )

        subset = (
            results[
                results[
                    "dynamics"
                ]
                == name
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

        pair_best_rows.append(
            best
        )

        log()

        log(
            f"{name:<18s} "
            f"{best['protocol']:<8s} "
            f"CV="
            f"{best['cv_rmse_mean']:.6f} "
            f"(+/- "
            f"{best['cv_rmse_std']:.6f}) "
            f"Val="
            f"{best['RMSE']:.6f}"
        )

    # ========================================================
    # Overall ranking
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
        "OVERALL JOINT-DYNAMICS x "
        "TEMPORAL RANKING"
    )

    log(
        "PRIMARY RANKING = "
        "TRAINING-ONLY CV"
    )

    log("=" * 80)
    log()

    for i, row in (
        ranking.iterrows()
    ):

        log(
            f"{i + 1:2d}. "
            f"{row['configuration']:<25s} "
            f"CV="
            f"{row['cv_rmse_mean']:.6f} "
            f"(+/- "
            f"{row['cv_rmse_std']:.6f}) "
            f"Val="
            f"{row['RMSE']:.6f}"
        )

    best = (
        ranking.iloc[0]
    )

    log()

    log(
        "Best 6.5B configuration:"
    )

    log(
        f"  "
        f"{best['configuration']}"
    )

    log(
        f"  alpha = "
        f"{best['alpha']}"
    )

    log(
        f"  J multiplier = "
        f"{best['j_mult']}"
    )

    log(
        f"  protocol = "
        f"{best['protocol']}"
    )

    log(
        f"  CV RMSE = "
        f"{best['cv_rmse_mean']:.6f}"
    )

    log(
        f"  Validation RMSE = "
        f"{best['RMSE']:.6f}"
    )

    log()

    log(
        "IMPORTANT:"
    )

    log(
        "  alpha and J were "
        "jointly varied."
    )

    log(
        "  hx=0.5 and dt=0.8 "
        "remain frozen."
    )

    log(
        "  Trotter depth remains r=2."
    )

    log(
        "  F4 + softsign + M=2 "
        "remain frozen."
    )

    log(
        "  All temporal protocols "
        "were tested."
    )

    log(
        "  Primary selection uses "
        "training-only CV."
    )

    log(
        "  2025 validation is "
        "diagnostic."
    )

    log(
        "  2026 test remains "
        "untouched."
    )

    SUMMARY_OUTPUT.write_text(
        "\n".join(
            output_lines
        ),
        encoding="utf-8",
    )

    print()

    print(
        f"Saved CV results to: "
        f"{CV_OUTPUT}"
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
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":

    main()