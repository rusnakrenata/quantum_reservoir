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

WEEK5_COUPLING_FILE = (
    RESULTS_DIR
    / "05_03a_reservoir_couplings.csv"
)

WEEK5_CONT_FILE = (
    RESULTS_DIR
    / "05_03a_qrc_features.csv"
)

WEEK5_RWP7_FILE = (
    RESULTS_DIR
    / "05_03b_rwp_w7_qrc_features.csv"
)

CV_OUTPUT = (
    RESULTS_DIR
    / "06_03_memory_cv_results.csv"
)

RESULT_OUTPUT = (
    RESULTS_DIR
    / "06_03_memory_validation_results.csv"
)

GEOMETRY_OUTPUT = (
    RESULTS_DIR
    / "06_03_memory_geometry.csv"
)

PURITY_OUTPUT = (
    RESULTS_DIR
    / "06_03_memory_purity.csv"
)

COUPLING_OUTPUT = (
    RESULTS_DIR
    / "06_03_memory_couplings.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "06_03_summary.txt"
)


# ============================================================
# Search space
# ============================================================

MEMORY_QUBITS = [
    1,
    2,
    3,
    4,
]

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
# Frozen input architecture
# ============================================================

N_INPUT = 4
D_INPUT = 2 ** N_INPUT

ALPHA = 1.0

H_X = 0.5

DELTA_T = 0.8

TROTTER_STEPS = 2

J_SCALE = 0.7

RESERVOIR_SEED = 42


# ============================================================
# Ridge grid frozen from Step 6.1
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
# Ry input encoding
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


def encode_input_state(row):

    claim = (
        np.clip(
            float(
                row["C_t_z"]
            ),
            -3.0,
            3.0,
        )
        / 3.0
    )

    theta_claim = (
        ALPHA
        * claim
    )

    theta_day = np.mod(
        np.arctan2(
            float(
                row["D_sin"]
            ),
            float(
                row["D_cos"]
            ),
        ),
        2.0 * np.pi,
    )

    policy = (
        np.clip(
            float(
                row["P_t_z"]
            ),
            -3.0,
            3.0,
        )
        / 3.0
    )

    theta_policy = (
        ALPHA
        * policy
    )

    theta_holiday = (
        np.pi
        * float(
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
# Nested couplings
# ============================================================

def build_nested_couplings():

    # Maximum system:
    #
    # 4 input + 4 memory = 8 qubits
    #
    # chain contains 7 edges.

    max_total_qubits = (
        N_INPUT
        + max(
            MEMORY_QUBITS
        )
    )

    max_edges = (
        max_total_qubits
        - 1
    )

    rng = np.random.default_rng(
        RESERVOIR_SEED
    )

    generated = rng.uniform(
        -J_SCALE,
        +J_SCALE,
        size=max_edges,
    )

    # --------------------------------------------------------
    # Reuse Week-5 couplings exactly
    # --------------------------------------------------------

    old = pd.read_csv(
        WEEK5_COUPLING_FILE
    )

    old_dict = {}

    for _, row in old.iterrows():

        edge = (
            int(
                row["q_i"]
            ),
            int(
                row["q_j"]
            ),
        )

        old_dict[
            edge
        ] = float(
            row["J_ij"]
        )

    all_couplings = {}

    for i in range(
        max_edges
    ):

        edge = (
            i,
            i + 1,
        )

        if edge in old_dict:

            value = (
                old_dict[
                    edge
                ]
            )

        else:

            value = float(
                generated[i]
            )

        all_couplings[
            edge
        ] = value

    return all_couplings


# ============================================================
# Reservoir unitary
# ============================================================

def build_unitary(
    n_memory,
    all_couplings,
):

    n_qubits = (
        N_INPUT
        + n_memory
    )

    qc = QuantumCircuit(
        n_qubits
    )

    dt = (
        DELTA_T
        / TROTTER_STEPS
    )

    for _ in range(
        TROTTER_STEPS
    ):

        for i in range(
            n_qubits - 1
        ):

            edge = (
                i,
                i + 1,
            )

            J = (
                all_couplings[
                    edge
                ]
            )

            qc.rzz(
                2.0
                * J
                * dt,
                i,
                i + 1,
            )

        for q in range(
            n_qubits
        ):

            qc.rx(
                2.0
                * H_X
                * dt,
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
    n_qubits,
    target_q,
    pauli,
):

    matrices = []

    for q in reversed(
        range(
            n_qubits
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

    O = matrices[0]

    for matrix in matrices[1:]:

        O = np.kron(
            O,
            matrix,
        )

    return O


def build_observables(
    n_qubits,
):

    names = []

    operators = []

    for q in range(
        n_qubits
    ):

        names.append(
            f"X{q}"
        )

        operators.append(
            single_qubit_operator(
                n_qubits,
                q,
                X,
            )
        )

        names.append(
            f"Z{q}"
        )

        operators.append(
            single_qubit_operator(
                n_qubits,
                q,
                Z,
            )
        )

    return (
        names,
        np.stack(
            operators,
            axis=0,
        ),
    )


# ============================================================
# Initial memory
# ============================================================

def initial_memory(
    d_memory,
):

    rho = np.zeros(
        (
            d_memory,
            d_memory,
        ),
        dtype=complex,
    )

    rho[0, 0] = 1.0

    return rho


# ============================================================
# Daily Kraus operators
# ============================================================

def build_kraus(
    phi,
    U4,
):

    # U4:
    #
    # [m_out, i_out, m_in, i_in]
    #
    # K_a[m_out,m_in]
    # =
    # sum_i U[m_out,a,m_in,i] phi[i]

    return np.einsum(
        "abcd,d->bac",
        U4,
        phi,
    )


# ============================================================
# Kraus -> row-major superoperator
# ============================================================

def kraus_to_superoperator(
    kraus,
):

    # S[(i,l),(j,k)]
    #
    # =
    # sum_a
    # K[a,i,j]
    # K*[a,l,k]

    d_memory = (
        kraus.shape[1]
    )

    S4 = np.einsum(
        "aij,alk->iljk",
        kraus,
        kraus.conj(),
        optimize=True,
    )

    return S4.reshape(
        d_memory ** 2,
        d_memory ** 2,
    )


# ============================================================
# Effective observable map L_t
# ============================================================

def build_transformed_observables(
    U,
    observables,
    d_memory,
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
                d_memory,
                D_INPUT,
                d_memory,
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

    # A_o =
    #
    # <phi| U^dagger O U |phi>

    A = np.einsum(
        "j,onjmi,i->onm",
        phi.conj(),
        transformed,
        phi,
        optimize=True,
    )

    # Tr(rho A)
    #
    # row-major vectorization

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
            A.shape[0],
            -1,
        )
    )


# ============================================================
# Partial trace for direct audits
# ============================================================

def trace_out_input(
    rho_global,
    d_memory,
):

    reshaped = (
        rho_global.reshape(
            d_memory,
            D_INPUT,
            d_memory,
            D_INPUT,
        )
    )

    return np.einsum(
        "aibi->ab",
        reshaped,
    )


# ============================================================
# Direct one-step audit
# ============================================================

def direct_step(
    rho_memory,
    phi,
    U,
    observables,
    d_memory,
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
            rho_global,
            d_memory,
        )
    )

    features = np.array(
        [
            np.sum(
                rho_global
                * operator.T
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
# Generate CONT + ALL W in a single chronological pass
# ============================================================

def generate_features_for_memory_size(
    df,
    target_column,
    input_states,
    n_memory,
    U,
    feature_names,
    observables,
):

    d_memory = (
        2 ** n_memory
    )

    d_memory_sq = (
        d_memory ** 2
    )

    # --------------------------------------------------------
    # Reshape U once
    # --------------------------------------------------------

    U4 = U.reshape(
        d_memory,
        D_INPUT,
        d_memory,
        D_INPUT,
    )

    transformed = (
        build_transformed_observables(
            U=U,
            observables=observables,
            d_memory=d_memory,
        )
    )

    rho0 = initial_memory(
        d_memory
    )

    vec0 = rho0.reshape(
        -1
    )

    # --------------------------------------------------------
    # age_vectors[a]
    #
    # state after last a inputs,
    # ending at previous day.
    #
    # We need ages 0 ... 28.
    # --------------------------------------------------------

    age_vectors = np.zeros(
        (
            MAX_WINDOW + 1,
            d_memory_sq,
        ),
        dtype=complex,
    )

    age_vectors[0] = (
        vec0
    )

    cont_vec = (
        vec0.copy()
    )

    protocol_rows = {
        "CONT": [],
    }

    purity_values = {
        "CONT": [],
    }

    for W in WINDOWS:

        protocol_rows[
            f"RWP_W{W}"
        ] = []

        purity_values[
            f"RWP_W{W}"
        ] = []

    audit_feature_errors = []

    audit_memory_errors = []

    audit_days = {
        100,
        500,
    }

    # ========================================================
    # Chronological daily loop
    # ========================================================

    for t, phi in enumerate(
        input_states
    ):

        # ----------------------------------------------------
        # Construct today's exact channel
        # ----------------------------------------------------

        kraus = build_kraus(
            phi=phi,
            U4=U4,
        )

        S = kraus_to_superoperator(
            kraus
        )

        L = build_feature_map(
            phi=phi,
            transformed=transformed,
        )

        # ----------------------------------------------------
        # CONT features:
        #
        # memory before today's input is cont_vec.
        # ----------------------------------------------------

        cont_features = (
            L
            @ cont_vec
        )

        cont_new = (
            S
            @ cont_vec
        )

        cont_rho_new = (
            cont_new.reshape(
                d_memory,
                d_memory,
            )
        )

        cont_purity = float(
            np.real(
                np.trace(
                    cont_rho_new
                    @ cont_rho_new
                )
            )
        )

        current = (
            df.iloc[t]
        )

        cont_row = {
            "memory_qubits": (
                n_memory
            ),

            "protocol": (
                "CONT"
            ),

            "window_W": (
                np.nan
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
                cont_purity
            ),
        }

        for (
            feature_name,
            value,
        ) in zip(
            feature_names,
            cont_features,
        ):

            cont_row[
                feature_name
            ] = float(
                np.real(
                    value
                )
            )

        protocol_rows[
            "CONT"
        ].append(
            cont_row
        )

        purity_values[
            "CONT"
        ].append(
            cont_purity
        )

        # ----------------------------------------------------
        # ALL rewinding windows
        #
        # Before today's channel:
        #
        # age_vectors[W-1]
        #
        # contains exactly the previous
        # W-1 inputs.
        # ----------------------------------------------------

        valid_windows = [
            W
            for W in WINDOWS
            if t >= W - 1
        ]

        if valid_windows:

            age_indices = np.array(
                [
                    W - 1
                    for W
                    in valid_windows
                ],
                dtype=int,
            )

            selected_memory = (
                age_vectors[
                    age_indices
                ]
            )

            # One matrix operation generates
            # features for all valid W.
            feature_batch = (
                selected_memory
                @ L.T
            )

        # ----------------------------------------------------
        # Update ALL ages simultaneously
        #
        # old age 0 -> new age 1
        # old age 1 -> new age 2
        # ...
        # old age 27 -> new age 28
        # ----------------------------------------------------

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
        ] = propagated

        # ----------------------------------------------------
        # Save each W
        # ----------------------------------------------------

        for row_number, W in enumerate(
            valid_windows
        ):

            feature_vec = (
                feature_batch[
                    row_number
                ]
            )

            final_memory_vec = (
                age_vectors[
                    W
                ]
            )

            final_memory = (
                final_memory_vec.reshape(
                    d_memory,
                    d_memory,
                )
            )

            purity = float(
                np.real(
                    np.trace(
                        final_memory
                        @ final_memory
                    )
                )
            )

            start_idx = (
                t
                - W
                + 1
            )

            row = {
                "memory_qubits": (
                    n_memory
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
                feature_name,
                value,
            ) in zip(
                feature_names,
                feature_vec,
            ):

                row[
                    feature_name
                ] = float(
                    np.real(
                        value
                    )
                )

            protocol_rows[
                f"RWP_W{W}"
            ].append(
                row
            )

            purity_values[
                f"RWP_W{W}"
            ].append(
                purity
            )

        # ----------------------------------------------------
        # Exact direct audit on nontrivial W=7 memory
        # ----------------------------------------------------

        if (
            t in audit_days
            and
            t >= 6
        ):

            old_memory_vec = (
                old_age_vectors[
                    6
                ]
            )

            old_memory = (
                old_memory_vec.reshape(
                    d_memory,
                    d_memory,
                )
            )

            (
                direct_memory,
                direct_features,
            ) = direct_step(
                rho_memory=(
                    old_memory
                ),
                phi=phi,
                U=U,
                observables=observables,
                d_memory=d_memory,
            )

            fast_features = (
                L
                @ old_memory_vec
            )

            fast_memory = (
                (
                    S
                    @ old_memory_vec
                )
                .reshape(
                    d_memory,
                    d_memory,
                )
            )

            feature_error = float(
                np.max(
                    np.abs(
                        fast_features
                        -
                        direct_features
                    )
                )
            )

            memory_error = float(
                np.max(
                    np.abs(
                        fast_memory
                        -
                        direct_memory
                    )
                )
            )

            audit_feature_errors.append(
                feature_error
            )

            audit_memory_errors.append(
                memory_error
            )

        # ----------------------------------------------------
        # Advance continuous memory
        # ----------------------------------------------------

        cont_vec = (
            cont_new
        )

        if (
            (t + 1) % 250 == 0
            or
            t == len(df) - 1
        ):

            log(
                f"  M={n_memory}: "
                f"processed "
                f"{t + 1:4d}/"
                f"{len(df)} days..."
            )

    # --------------------------------------------------------
    # Convert to DataFrames
    # --------------------------------------------------------

    tables = {
        protocol: pd.DataFrame(
            rows
        )
        for (
            protocol,
            rows
        ) in protocol_rows.items()
    }

    purity_summary = []

    for (
        protocol,
        values,
    ) in purity_values.items():

        values = np.asarray(
            values,
            dtype=float,
        )

        purity_summary.append(
            {
                "memory_qubits": (
                    n_memory
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

    max_feature_error = (
        max(
            audit_feature_errors
        )
        if audit_feature_errors
        else 0.0
    )

    max_memory_error = (
        max(
            audit_memory_errors
        )
        if audit_memory_errors
        else 0.0
    )

    return (
        tables,
        pd.DataFrame(
            purity_summary
        ),
        max_feature_error,
        max_memory_error,
    )


# ============================================================
# Ridge
# ============================================================

def make_model(
    alpha,
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
                    alpha=alpha,
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

    for alpha in RIDGE_LAMBDAS:

        rmses = []

        for (
            train_idx,
            val_idx,
        ) in splitter.split(
            X
        ):

            model = make_model(
                alpha
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

            rmses.append(
                rmse
            )

        rows.append(
            {
                "configuration": (
                    configuration
                ),

                "lambda": (
                    alpha
                ),

                "cv_rmse_mean": float(
                    np.mean(
                        rmses
                    )
                ),

                "cv_rmse_std": float(
                    np.std(
                        rmses,
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

    Xs = StandardScaler().fit_transform(
        X_train
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
        rank < Xs.shape[1]
        or
        sigma_min <= 0
    ):

        kappa = np.inf

    else:

        kappa = float(
            sigma_max
            / sigma_min
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
        "n_features": (
            Xs.shape[1]
        ),
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

    target_std = float(
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
            target_std
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 6 - STEP 6.3")
    log("MEMORY-QUBIT SEARCH")
    log("M={1,2,3,4} x CONT + ALL WINDOWS")
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

    target_column = (
        resolve_target_column(
            df
        )
    )

    if len(df) != 1692:

        raise RuntimeError(
            "Expected 1692 rows."
        )

    input_states = [
        encode_input_state(
            row
        )
        for _, row
        in df.iterrows()
    ]

    # --------------------------------------------------------
    # Nested fixed couplings
    # --------------------------------------------------------

    all_couplings = (
        build_nested_couplings()
    )

    coupling_rows = []

    log(
        "Nested fixed couplings:"
    )

    for (
        edge,
        value,
    ) in all_couplings.items():

        log(
            f"  J{edge} = "
            f"{value:+.12f}"
        )

        coupling_rows.append(
            {
                "q_i": edge[0],
                "q_j": edge[1],
                "J_ij": value,
            }
        )

    pd.DataFrame(
        coupling_rows
    ).to_csv(
        COUPLING_OUTPUT,
        index=False,
    )

    all_cv = []

    all_results = []

    all_geometry = []

    all_purity = []

    # ========================================================
    # MEMORY SIZE LOOP
    # ========================================================

    for n_memory in MEMORY_QUBITS:

        n_qubits = (
            N_INPUT
            + n_memory
        )

        log()
        log("=" * 80)
        log(
            f"MEMORY QUBITS = "
            f"{n_memory}"
        )
        log("=" * 80)

        log(
            f"Total qubits = "
            f"{n_qubits}"
        )

        # ----------------------------------------------------
        # Reservoir
        # ----------------------------------------------------

        U = build_unitary(
            n_memory=(
                n_memory
            ),
            all_couplings=(
                all_couplings
            ),
        )

        unitarity_error = float(
            np.linalg.norm(
                U.conj().T
                @ U
                -
                np.eye(
                    U.shape[0],
                    dtype=complex,
                ),
                ord="fro",
            )
        )

        log(
            f"Unitarity error = "
            f"{unitarity_error:.3e}"
        )

        (
            feature_names,
            observables,
        ) = build_observables(
            n_qubits
        )

        log(
            f"QRC features = "
            f"{len(feature_names)}"
        )

        # ----------------------------------------------------
        # Generate CONT + ALL W
        # ----------------------------------------------------

        (
            tables,
            purity_df,
            audit_feature_error,
            audit_memory_error,
        ) = generate_features_for_memory_size(
            df=df,
            target_column=target_column,
            input_states=input_states,
            n_memory=n_memory,
            U=U,
            feature_names=feature_names,
            observables=observables,
        )

        log(
            f"Direct audit feature error = "
            f"{audit_feature_error:.3e}"
        )

        log(
            f"Direct audit memory error  = "
            f"{audit_memory_error:.3e}"
        )

        if (
            audit_feature_error > 1e-10
            or
            audit_memory_error > 1e-10
        ):

            raise RuntimeError(
                f"M={n_memory}: "
                "fast-channel audit FAILED."
            )

        # ----------------------------------------------------
        # Save generated features for this M
        # ----------------------------------------------------

        combined = pd.concat(
            tables.values(),
            ignore_index=True,
        )

        memory_feature_file = (
            RESULTS_DIR
            / (
                f"06_03_features_"
                f"M{n_memory}.csv"
            )
        )

        combined.to_csv(
            memory_feature_file,
            index=False,
        )

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

            if n_train != expected_train:

                raise RuntimeError(
                    f"M={n_memory}, "
                    f"{protocol}: "
                    f"wrong training count."
                )

            if (
                n_val != 365
                or
                n_test != 232
            ):

                raise RuntimeError(
                    f"M={n_memory}, "
                    f"{protocol}: "
                    "validation/test count error."
                )

            Xqrc = (
                table[
                    feature_names
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
                f"M{n_memory}_"
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
                    "memory_qubits": (
                        n_memory
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
            # CV
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
                "memory_qubits"
            ] = n_memory

            cv_df[
                "protocol"
            ] = protocol

            all_cv.append(
                cv_df
            )

            # ------------------------------------------------
            # Validation diagnostic
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
                    "memory_qubits": (
                        n_memory
                    ),

                    "total_qubits": (
                        n_qubits
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

                    "n_features": (
                        len(
                            feature_names
                        )
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
                f"{configuration:<14s} "
                f"N={n_train:4d} "
                f"CV="
                f"{best_cv_mean:.6f} "
                f"(+/- "
                f"{best_cv_std:.6f}) "
                f"lambda="
                f"{best_lambda:<7g} "
                f"rank="
                f"{geometry['rank']:2d}/"
                f"{geometry['n_features']:2d} "
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
    # Best configuration for each memory size
    # ========================================================

    log()
    log("=" * 80)
    log(
        "BEST TEMPORAL PROTOCOL "
        "WITHIN EACH MEMORY SIZE"
    )
    log("=" * 80)

    for n_memory in MEMORY_QUBITS:

        subset = (
            results[
                results[
                    "memory_qubits"
                ]
                == n_memory
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
            f"M={n_memory}: "
            f"{best['protocol']} "
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
        "OVERALL MEMORY x TEMPORAL RANKING"
    )
    log(
        "PRIMARY RANKING = TRAINING-ONLY CV"
    )
    log("=" * 80)
    log()

    for i, row in (
        ranking.iterrows()
    ):

        log(
            f"{i + 1:2d}. "
            f"{row['configuration']:<14s} "
            f"Q={int(row['total_qubits'])} "
            f"F={int(row['n_features']):2d} "
            f"N={int(row['train_rows']):4d} "
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
        "Best overall configuration "
        "by training CV:"
    )

    log(
        f"  {best['configuration']}"
    )

    log()

    log(
        "IMPORTANT:"
    )

    log(
        "  All windows were tested "
        "for every memory size."
    )

    log(
        "  Each W uses its maximum "
        "legitimate training data."
    )

    log(
        "  All configurations retain "
        "365 validation days."
    )

    log(
        "  2025 validation is "
        "diagnostic only."
    )

    log(
        "  2026 test set remains "
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
        f"Saved couplings to: "
        f"{COUPLING_OUTPUT}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":

    main()