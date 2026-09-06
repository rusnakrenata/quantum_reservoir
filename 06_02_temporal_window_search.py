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

CONT_FILE = (
    RESULTS_DIR
    / "05_03a_qrc_features.csv"
)

RWP7_REFERENCE_FILE = (
    RESULTS_DIR
    / "05_03b_rwp_w7_qrc_features.csv"
)

COUPLING_FILE = (
    RESULTS_DIR
    / "05_03a_reservoir_couplings.csv"
)

FEATURE_OUTPUT = (
    RESULTS_DIR
    / "06_02_all_rwp_features.csv"
)

CV_OUTPUT = (
    RESULTS_DIR
    / "06_02_temporal_cv_results.csv"
)

VALIDATION_OUTPUT = (
    RESULTS_DIR
    / "06_02_temporal_validation_results.csv"
)

GEOMETRY_OUTPUT = (
    RESULTS_DIR
    / "06_02_temporal_geometry.csv"
)

PURITY_OUTPUT = (
    RESULTS_DIR
    / "06_02_temporal_purity.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "06_02_summary.txt"
)


# ============================================================
# Temporal search
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


# ============================================================
# Frozen Week-5 QRC architecture
# ============================================================

N_INPUT = 4
N_MEMORY = 2
N_QUBITS = (
    N_INPUT
    + N_MEMORY
)

D_INPUT = (
    2 ** N_INPUT
)

D_MEMORY = (
    2 ** N_MEMORY
)

D_TOTAL = (
    2 ** N_QUBITS
)

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]

ALPHA = 1.0

H_X = 0.5

DELTA_T = 0.8

TROTTER_STEPS = 2


# ============================================================
# Ridge grid frozen after Step 6.1
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
# QRC observables
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
# Encode one F4 input
# ============================================================

def encode_input_state(row):

    # --------------------------------------------------------
    # Claim count
    # --------------------------------------------------------

    claim_scaled = (
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
        * claim_scaled
    )

    # --------------------------------------------------------
    # Weekday
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Policy count
    # --------------------------------------------------------

    policy_scaled = (
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
        * policy_scaled
    )

    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    theta_holiday = (
        np.pi
        * float(
            row[
                "is_public_holiday_t_plus_1"
            ]
        )
    )

    # --------------------------------------------------------
    # Single-qubit states
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

    # --------------------------------------------------------
    # Numerical/Qiskit ordering:
    #
    # q3 kron q2 kron q1 kron q0
    # --------------------------------------------------------

    phi = np.kron(
        q3,
        np.kron(
            q2,
            np.kron(
                q1,
                q0,
            ),
        ),
    )

    return phi


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

    rho[0, 0] = 1.0

    return rho


# ============================================================
# Load frozen Week-5 couplings
# ============================================================

def load_couplings():

    if not COUPLING_FILE.exists():

        raise FileNotFoundError(
            f"Missing coupling file: "
            f"{COUPLING_FILE}"
        )

    coupling_df = pd.read_csv(
        COUPLING_FILE
    )

    couplings = {}

    for _, row in (
        coupling_df.iterrows()
    ):

        edge = (
            int(
                row["q_i"]
            ),
            int(
                row["q_j"]
            ),
        )

        couplings[
            edge
        ] = float(
            row["J_ij"]
        )

    if (
        set(
            couplings.keys()
        )
        !=
        set(
            EDGES
        )
    ):

        raise RuntimeError(
            "Coupling topology mismatch."
        )

    return couplings


# ============================================================
# Build frozen reservoir unitary
# ============================================================

def build_reservoir_unitary(
    couplings,
):

    qc = QuantumCircuit(
        N_QUBITS
    )

    dt = (
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
        ), J_ij in (
            couplings.items()
        ):

            qc.rzz(
                2.0
                * J_ij
                * dt,
                q_i,
                q_j,
            )

        # ----------------------------------------------------
        # X field
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            qc.rx(
                2.0
                * H_X
                * dt,
                q,
            )

    U = np.asarray(
        Operator(
            qc
        ).data,
        dtype=complex,
    )

    return U


# ============================================================
# Observable construction
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
        matrices[1:]
    ):

        result = np.kron(
            result,
            matrix,
        )

    return result


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
# Partial trace over input register
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

    rho_memory = np.einsum(
        "aibi->ab",
        reshaped,
    )

    return rho_memory


# ============================================================
# Direct full-density propagation
#
# Used for correctness audits only
# ============================================================

def direct_step(
    rho_memory,
    phi,
    U,
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

    return (
        rho_global,
        rho_memory_new,
    )


# ============================================================
# Expectation value
# ============================================================

def expectation(
    rho,
    operator,
):

    return np.sum(
        rho
        * operator.T
    )


# ============================================================
# Exact memory-channel superoperator S_t
# ============================================================

def build_memory_superoperator(
    phi,
    U4,
):

    # --------------------------------------------------------
    # U4 indexing:
    #
    # U4[m_out, i_out, m_in, i_in]
    #
    # Kraus:
    #
    # K_a[m_out,m_in]
    #
    # =
    #
    # sum_i
    # U[m_out,a,m_in,i] phi[i]
    # --------------------------------------------------------

    kraus = np.einsum(
        "abcd,d->bac",
        U4,
        phi,
    )

    # --------------------------------------------------------
    # Row-major vectorization:
    #
    # vec_r(K rho K^dagger)
    #
    # =
    #
    # (K kron K*) vec_r(rho)
    # --------------------------------------------------------

    S = np.zeros(
        (
            D_MEMORY ** 2,
            D_MEMORY ** 2,
        ),
        dtype=complex,
    )

    for K in kraus:

        S += np.kron(
            K,
            K.conj(),
        )

    return S


# ============================================================
# Exact final-feature map L_t
# ============================================================

def build_feature_map(
    phi,
    transformed_observables,
):

    # --------------------------------------------------------
    # transformed_observables:
    #
    # B_o = U^dagger O U
    #
    # shape:
    #
    # [observable,
    #  memory_out,
    #  input_out,
    #  memory_in,
    #  input_in]
    #
    # Effective memory observable:
    #
    # A_o =
    # <phi| B_o |phi>_input
    # --------------------------------------------------------

    A = np.einsum(
        "j,onjmi,i->onm",
        phi.conj(),
        transformed_observables,
        phi,
    )

    # --------------------------------------------------------
    # Tr(rho A)
    #
    # =
    #
    # vec_r(A^T)^T vec_r(rho)
    # --------------------------------------------------------

    L = (
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

    return L


# ============================================================
# Precompute S_t and L_t for every day
# ============================================================

def precompute_daily_maps(
    input_states,
    U,
    observables,
):

    n = len(
        input_states
    )

    S_all = np.zeros(
        (
            n,
            D_MEMORY ** 2,
            D_MEMORY ** 2,
        ),
        dtype=complex,
    )

    L_all = np.zeros(
        (
            n,
            len(
                FEATURE_COLUMNS
            ),
            D_MEMORY ** 2,
        ),
        dtype=complex,
    )

    # --------------------------------------------------------
    # Reshape U:
    #
    # [memory_out,
    #  input_out,
    #  memory_in,
    #  input_in]
    # --------------------------------------------------------

    U4 = U.reshape(
        D_MEMORY,
        D_INPUT,
        D_MEMORY,
        D_INPUT,
    )

    # --------------------------------------------------------
    # U^dagger O U is independent
    # of daily input, so compute once.
    # --------------------------------------------------------

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

    transformed = np.stack(
        transformed,
        axis=0,
    )

    # --------------------------------------------------------
    # Daily maps
    # --------------------------------------------------------

    for t, phi in enumerate(
        input_states
    ):

        S_all[t] = (
            build_memory_superoperator(
                phi=phi,
                U4=U4,
            )
        )

        L_all[t] = (
            build_feature_map(
                phi=phi,
                transformed_observables=(
                    transformed
                ),
            )
        )

        if (
            (t + 1) % 250 == 0
            or
            t == n - 1
        ):

            log(
                f"Precomputed maps "
                f"{t + 1:4d}/{n}..."
            )

    return (
        S_all,
        L_all,
    )


# ============================================================
# Generate rewinding features for one W
# ============================================================

def generate_window_features(
    df,
    target_column,
    W,
    S_all,
    L_all,
):

    rho0 = initial_memory()

    memory_initial_vec = (
        rho0.reshape(
            -1
        )
    )

    rows = []

    purities = []

    # --------------------------------------------------------
    # First complete window ends at W-1
    # --------------------------------------------------------

    for end_idx in range(
        W - 1,
        len(df),
    ):

        start_idx = (
            end_idx
            - W
            + 1
        )

        # ----------------------------------------------------
        # RESET memory at beginning of every window
        # ----------------------------------------------------

        memory_vec = (
            memory_initial_vec.copy()
        )

        # ----------------------------------------------------
        # Replay first W-1 inputs
        # ----------------------------------------------------

        for t in range(
            start_idx,
            end_idx,
        ):

            memory_vec = (
                S_all[t]
                @ memory_vec
            )

        # ----------------------------------------------------
        # Final input:
        #
        # obtain features from L_t
        # ----------------------------------------------------

        feature_vec = (
            L_all[
                end_idx
            ]
            @ memory_vec
        )

        # ----------------------------------------------------
        # Also propagate final input into memory
        # for purity diagnostic
        # ----------------------------------------------------

        final_memory_vec = (
            S_all[
                end_idx
            ]
            @ memory_vec
        )

        final_memory = (
            final_memory_vec.reshape(
                D_MEMORY,
                D_MEMORY,
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

        purities.append(
            purity
        )

        current = (
            df.iloc[
                end_idx
            ]
        )

        row = {
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
            FEATURE_COLUMNS,
            feature_vec,
        ):

            row[
                feature_name
            ] = float(
                np.real(
                    value
                )
            )

        rows.append(
            row
        )

    feature_df = pd.DataFrame(
        rows
    )

    return (
        feature_df,
        np.asarray(
            purities,
            dtype=float,
        ),
    )


# ============================================================
# Direct W-window calculation
#
# For validating superoperator implementation
# ============================================================

def direct_window(
    end_idx,
    W,
    input_states,
    U,
    observables,
):

    start_idx = (
        end_idx
        - W
        + 1
    )

    rho_memory = (
        initial_memory()
    )

    rho_global = None

    for t in range(
        start_idx,
        end_idx + 1,
    ):

        (
            rho_global,
            rho_memory,
        ) = direct_step(
            rho_memory=(
                rho_memory
            ),
            phi=input_states[t],
            U=U,
        )

    features = np.asarray(
        [
            expectation(
                rho_global,
                operator,
            )
            for operator
            in observables
        ],
        dtype=complex,
    )

    return (
        features,
        rho_memory,
    )


# ============================================================
# Fast-vs-direct audits
# ============================================================

def audit_fast_method(
    all_window_tables,
    input_states,
    U,
    observables,
):

    checks = [
        (1, 27),
        (2, 100),
        (5, 500),
        (7, 100),
        (7, 500),
        (14, 500),
        (21, 800),
        (28, 500),
        (28, 1080),
    ]

    max_feature_error = 0.0
    max_purity_error = 0.0

    for W, end_idx in checks:

        if end_idx >= len(
            input_states
        ):
            continue

        (
            direct_features,
            direct_memory,
        ) = direct_window(
            end_idx=end_idx,
            W=W,
            input_states=input_states,
            U=U,
            observables=observables,
        )

        table = (
            all_window_tables[
                W
            ]
        )

        row_idx = (
            end_idx
            - (W - 1)
        )

        fast_features = (
            table.iloc[
                row_idx
            ][
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )

        feature_error = float(
            np.max(
                np.abs(
                    np.real(
                        direct_features
                    )
                    -
                    fast_features
                )
            )
        )

        fast_purity = float(
            table.iloc[
                row_idx
            ][
                "memory_purity"
            ]
        )

        direct_purity = float(
            np.real(
                np.trace(
                    direct_memory
                    @ direct_memory
                )
            )
        )

        purity_error = abs(
            fast_purity
            -
            direct_purity
        )

        max_feature_error = max(
            max_feature_error,
            feature_error,
        )

        max_purity_error = max(
            max_purity_error,
            purity_error,
        )

        target_date = (
            table.iloc[
                row_idx
            ][
                "target_date"
            ]
        )

        log(
            f"Audit W={W:2d}, "
            f"target="
            f"{pd.to_datetime(target_date).date()}: "
            f"feature error="
            f"{feature_error:.3e}, "
            f"purity error="
            f"{purity_error:.3e}"
        )

    return (
        max_feature_error,
        max_purity_error,
    )


# ============================================================
# Ridge model
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


# ============================================================
# Training-only chronological CV
# ============================================================

def ridge_cv(
    X,
    y,
    protocol,
):

    splitter = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )

    rows = []

    for alpha in RIDGE_LAMBDAS:

        fold_rmse = []

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

            fold_rmse.append(
                rmse
            )

        rows.append(
            {
                "protocol": (
                    protocol
                ),

                "lambda": alpha,

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

                "fold_1_rmse": (
                    fold_rmse[0]
                ),

                "fold_2_rmse": (
                    fold_rmse[1]
                ),

                "fold_3_rmse": (
                    fold_rmse[2]
                ),

                "fold_4_rmse": (
                    fold_rmse[3]
                ),

                "fold_5_rmse": (
                    fold_rmse[4]
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

    scaler = StandardScaler()

    X_std = scaler.fit_transform(
        X_train
    )

    singular_values = np.linalg.svd(
        X_std,
        compute_uv=False,
    )

    rank = int(
        np.linalg.matrix_rank(
            X_std
        )
    )

    sigma_max = float(
        singular_values[0]
    )

    sigma_min = float(
        singular_values[-1]
    )

    if (
        rank
        <
        X_std.shape[1]
        or
        sigma_min <= 0
    ):

        kappa_X = np.inf

    else:

        kappa_X = float(
            sigma_max
            /
            sigma_min
        )

    C = (
        X_std.T
        @ X_std
    ) / len(
        X_std
    )

    eigenvalues = np.linalg.eigvalsh(
        C
    )

    rho_C = float(
        np.max(
            eigenvalues
        )
    )

    return {
        "rank": rank,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "kappa_X": kappa_X,
        "rho_C": rho_C,
    }


# ============================================================
# Validation metrics
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

    nrmse = float(
        rmse
        /
        target_std
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "NRMSE": nrmse,
    }


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 6 - STEP 6.2")
    log("TEMPORAL MEMORY SEARCH")
    log("CONT vs RWP W={1,2,5,7,14,21,28}")
    log("=" * 80)
    log()

    # --------------------------------------------------------
    # Load original insurance dataset
    # --------------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Missing: "
            f"{DATA_FILE}"
        )

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
            "Expected exactly "
            "1692 samples."
        )

    log(
        f"Loaded samples       = "
        f"{len(df)}"
    )

    log(
        f"Target column        = "
        f"{target_column}"
    )

    log()

    # --------------------------------------------------------
    # Fixed Week-5 reservoir
    # --------------------------------------------------------

    couplings = (
        load_couplings()
    )

    U = (
        build_reservoir_unitary(
            couplings
        )
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
        f"Unitarity error      = "
        f"{unitarity_error:.3e}"
    )

    # --------------------------------------------------------
    # Encode all daily inputs once
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Precompute exact reduced maps
    # --------------------------------------------------------

    log()

    log(
        "Precomputing exact daily "
        "superoperators..."
    )

    (
        S_all,
        L_all,
    ) = precompute_daily_maps(
        input_states=input_states,
        U=U,
        observables=observables,
    )

    # --------------------------------------------------------
    # Generate all RWP feature tables
    # --------------------------------------------------------

    log()

    log(
        "Generating rewinding "
        "feature tables..."
    )

    all_window_tables = {}

    combined_tables = []

    purity_rows = []

    for W in WINDOWS:

        (
            feature_df,
            purities,
        ) = generate_window_features(
            df=df,
            target_column=target_column,
            W=W,
            S_all=S_all,
            L_all=L_all,
        )

        all_window_tables[
            W
        ] = feature_df

        combined_tables.append(
            feature_df
        )

        split_counts = (
            feature_df[
                "split"
            ]
            .astype(str)
            .str.lower()
            .value_counts()
            .to_dict()
        )

        purity_rows.append(
            {
                "protocol": (
                    f"RWP_W{W}"
                ),

                "W": W,

                "min_purity": float(
                    np.min(
                        purities
                    )
                ),

                "mean_purity": float(
                    np.mean(
                        purities
                    )
                ),

                "max_purity": float(
                    np.max(
                        purities
                    )
                ),
            }
        )

        log(
            f"  W={W:2d}: "
            f"rows={len(feature_df)}, "
            f"split={split_counts}, "
            f"mean purity="
            f"{np.mean(purities):.6f}"
        )

    # --------------------------------------------------------
    # Save generated RWP features
    # --------------------------------------------------------

    all_rwp = pd.concat(
        combined_tables,
        ignore_index=True,
    )

    all_rwp.to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Superoperator correctness audits
    # --------------------------------------------------------

    log()

    log(
        "Superoperator vs direct "
        "density-matrix audits:"
    )

    (
        max_feature_error,
        max_purity_error,
    ) = audit_fast_method(
        all_window_tables=(
            all_window_tables
        ),
        input_states=(
            input_states
        ),
        U=U,
        observables=(
            observables
        ),
    )

    log()

    log(
        f"Maximum feature error = "
        f"{max_feature_error:.3e}"
    )

    log(
        f"Maximum purity error  = "
        f"{max_purity_error:.3e}"
    )

    # Hard correctness criterion
    if (
        max_feature_error
        >
        1e-10
        or
        max_purity_error
        >
        1e-10
    ):

        raise RuntimeError(
            "Superoperator correctness "
            "audit FAILED."
        )

    # --------------------------------------------------------
    # Cross-check against validated Week-5 W=7
    # --------------------------------------------------------

    if (
        RWP7_REFERENCE_FILE.exists()
    ):

        old_w7 = pd.read_csv(
            RWP7_REFERENCE_FILE
        )

        old_w7[
            "target_date"
        ] = pd.to_datetime(
            old_w7[
                "target_date"
            ]
        )

        new_w7 = (
            all_window_tables[
                7
            ]
            .copy()
        )

        new_w7[
            "target_date"
        ] = pd.to_datetime(
            new_w7[
                "target_date"
            ]
        )

        merged = (
            old_w7.merge(
                new_w7,
                on="target_date",
                suffixes=(
                    "_old",
                    "_new",
                ),
            )
        )

        differences = []

        for feature in FEATURE_COLUMNS:

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

        reference_error = float(
            np.max(
                differences
            )
        )

        log()

        log(
            "RWP W=7 vs validated "
            "Week-5 implementation:"
        )

        log(
            f"  maximum feature error = "
            f"{reference_error:.3e}"
        )

        if reference_error > 1e-10:

            raise RuntimeError(
                "W=7 reference "
                "comparison FAILED."
            )

    # ========================================================
    # Load CONT
    # ========================================================

    cont = pd.read_csv(
        CONT_FILE
    )

    cont[
        "target_date"
    ] = pd.to_datetime(
        cont[
            "target_date"
        ]
    )

    cont = (
        cont
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    cont_target_column = (
        resolve_target_column(
            cont
        )
    )

    # ========================================================
    # Build protocol datasets
    #
    # IMPORTANT:
    #
    # Each W uses ALL legitimately available
    # samples for that W.
    #
    # No common W=28 date restriction.
    # ========================================================

    protocol_tables = {}

    # --------------------------------------------------------
    # CONT
    # --------------------------------------------------------

    cont_table = pd.DataFrame(
        {
            "target_date": (
                cont[
                    "target_date"
                ]
            ),

            "split": (
                cont[
                    "split"
                ]
            ),

            "target": (
                cont[
                    cont_target_column
                ]
                .astype(float)
            ),
        }
    )

    for feature in FEATURE_COLUMNS:

        cont_table[
            feature
        ] = (
            cont[
                feature
            ]
            .astype(float)
        )

    protocol_tables[
        "CONT"
    ] = cont_table

    # --------------------------------------------------------
    # RWP windows
    # --------------------------------------------------------

    for W in WINDOWS:

        protocol_tables[
            f"RWP_W{W}"
        ] = (
            all_window_tables[
                W
            ]
            .copy()
        )

    # ========================================================
    # Evaluate each protocol independently
    # ========================================================

    cv_tables = []

    result_rows = []

    geometry_rows = []

    log()

    log("=" * 80)

    log(
        "TEMPORAL SEARCH"
    )

    log(
        "Each W uses maximum legitimate "
        "training history"
    )

    log("=" * 80)

    for (
        protocol,
        table,
    ) in protocol_tables.items():

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

        # ----------------------------------------------------
        # Validation and test should always remain complete.
        # ----------------------------------------------------

        if n_val != 365:

            raise RuntimeError(
                f"{protocol}: expected "
                f"365 validation rows, "
                f"found {n_val}."
            )

        if n_test != 232:

            raise RuntimeError(
                f"{protocol}: expected "
                f"232 test rows, "
                f"found {n_test}."
            )

        # ----------------------------------------------------
        # Expected training size
        # ----------------------------------------------------

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
                f"{protocol}: expected "
                f"{expected_train} "
                f"training rows, "
                f"found {n_train}."
            )

        # ----------------------------------------------------
        # Matrices
        # ----------------------------------------------------

        X = (
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
            X[
                train_mask
            ]
        )

        y_train = (
            y[
                train_mask
            ]
        )

        X_val = (
            X[
                val_mask
            ]
        )

        y_val = (
            y[
                val_mask
            ]
        )

        # ----------------------------------------------------
        # Feature geometry
        # ----------------------------------------------------

        geometry = (
            feature_geometry(
                X_train
            )
        )

        geometry_rows.append(
            {
                "protocol": (
                    protocol
                ),

                "train_rows": (
                    n_train
                ),

                **geometry,
            }
        )

        # ----------------------------------------------------
        # Training-only Ridge CV
        # ----------------------------------------------------

        (
            best_lambda,
            best_cv_mean,
            best_cv_std,
            cv_table,
        ) = ridge_cv(
            X=X_train,
            y=y_train,
            protocol=protocol,
        )

        cv_tables.append(
            cv_table
        )

        # ----------------------------------------------------
        # 2025 validation diagnostic
        # ----------------------------------------------------

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

        result_rows.append(
            {
                "protocol": (
                    protocol
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

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        log()

        log(
            f"{protocol}:"
        )

        log(
            f"  Train rows     = "
            f"{n_train}"
        )

        log(
            f"  Validation     = "
            f"{n_val}"
        )

        log(
            f"  Test           = "
            f"{n_test} "
            f"(NOT evaluated)"
        )

        log(
            f"  lambda*        = "
            f"{best_lambda:g}"
        )

        log(
            f"  CV RMSE        = "
            f"{best_cv_mean:.6f} "
            f"(+/- "
            f"{best_cv_std:.6f})"
        )

        log(
            f"  rank           = "
            f"{geometry['rank']}"
        )

        log(
            f"  kappa(X)       = "
            f"{geometry['kappa_X']:.6e}"
        )

        log(
            f"  rho(C)         = "
            f"{geometry['rho_C']:.6f}"
        )

        log(
            f"  Val RMSE       = "
            f"{metrics['RMSE']:.6f}"
        )

        log(
            f"  Val Bias       = "
            f"{metrics['Bias']:+.6f}"
        )

    # ========================================================
    # Save results
    # ========================================================

    cv_all = pd.concat(
        cv_tables,
        ignore_index=True,
    )

    results = pd.DataFrame(
        result_rows
    )

    geometry_df = pd.DataFrame(
        geometry_rows
    )

    purity_df = pd.DataFrame(
        purity_rows
    )

    cv_all.to_csv(
        CV_OUTPUT,
        index=False,
    )

    results.to_csv(
        VALIDATION_OUTPUT,
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

    # ========================================================
    # Primary ranking:
    #
    # TRAINING-ONLY chronological CV
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
        "TEMPORAL PROTOCOL RANKING"
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
            f"{row['protocol']:<8s} "
            f"Ntrain="
            f"{int(row['train_rows']):4d} "
            f"CV="
            f"{row['cv_rmse_mean']:.6f} "
            f"(+/- "
            f"{row['cv_rmse_std']:.6f}) "
            f"lambda*="
            f"{row['best_lambda']:<8g} "
            f"Val="
            f"{row['RMSE']:.6f}"
        )

    best_protocol = str(
        ranking.iloc[
            0
        ][
            "protocol"
        ]
    )

    log()

    log(
        "Best temporal protocol "
        "by training CV:"
    )

    log(
        f"  {best_protocol}"
    )

    log()

    log(
        "IMPORTANT:"
    )

    log(
        "  Each W used all legitimate "
        "training samples available "
        "for that W."
    )

    log(
        "  Validation contains all "
        "365 days for every protocol."
    )

    log(
        "  Historical inputs before "
        "the validation boundary are "
        "allowed."
    )

    log(
        "  2025 validation is "
        "diagnostic only."
    )

    log(
        "  2026 test set remains "
        "untouched."
    )

    # ========================================================
    # Save summary
    # ========================================================

    log()

    log(
        "Saved all RWP features to:"
    )

    log(
        f"  {FEATURE_OUTPUT}"
    )

    log(
        "Saved CV results to:"
    )

    log(
        f"  {CV_OUTPUT}"
    )

    log(
        "Saved temporal results to:"
    )

    log(
        f"  {VALIDATION_OUTPUT}"
    )

    log(
        "Saved feature geometry to:"
    )

    log(
        f"  {GEOMETRY_OUTPUT}"
    )

    log(
        "Saved purity diagnostics to:"
    )

    log(
        f"  {PURITY_OUTPUT}"
    )

    SUMMARY_OUTPUT.write_text(
        "\n".join(
            output_lines
        ),
        encoding="utf-8",
    )

    print()

    print(
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":

    main()