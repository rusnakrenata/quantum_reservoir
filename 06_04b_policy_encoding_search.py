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
    / "06_04b_policy_encoding_cv.csv"
)

RESULT_OUTPUT = (
    RESULTS_DIR
    / "06_04b_policy_encoding_validation.csv"
)

GEOMETRY_OUTPUT = (
    RESULTS_DIR
    / "06_04b_policy_encoding_geometry.csv"
)

PURITY_OUTPUT = (
    RESULTS_DIR
    / "06_04b_policy_encoding_purity.csv"
)

ENCODING_DIAGNOSTICS_OUTPUT = (
    RESULTS_DIR
    / "06_04b_policy_encoding_diagnostics.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "06_04b_summary.txt"
)


# ============================================================
# Policy encoding strategies
# ============================================================

POLICY_ENCODINGS = [
    "clip3",
    "tanh3",
    "arctan",
    "softsign3",
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
# Frozen QRC architecture
# ============================================================

N_INPUT = 4
N_MEMORY = 2
N_QUBITS = 6

D_INPUT = 2 ** N_INPUT
D_MEMORY = 2 ** N_MEMORY
D_TOTAL = 2 ** N_QUBITS

ALPHA = 1.0
CLAIM_CLIP_Z = 3.0

H_X = 0.5
DELTA_T = 0.8
TROTTER_STEPS = 2

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]


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
# Observable features
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
# Policy encoding
# ============================================================

def policy_map(
    z,
    method,
):

    z = float(z)

    if method == "clip3":

        # Existing Week-5 encoding.
        return float(
            np.clip(
                z / 3.0,
                -1.0,
                1.0,
            )
        )

    if method == "tanh3":

        # derivative at z=0 = 1/3
        return float(
            np.tanh(
                z / 3.0
            )
        )

    if method == "arctan":

        # g(z) =
        #
        # (2/pi) atan(pi z / 6)
        #
        # g'(0) = 1/3
        return float(
            (
                2.0
                / np.pi
            )
            * np.arctan(
                np.pi
                * z
                / 6.0
            )
        )

    if method == "softsign3":

        # g(z) =
        #
        # z / (3 + |z|)
        #
        # derivative at zero = 1/3
        return float(
            z
            / (
                3.0
                +
                abs(z)
            )
        )

    raise ValueError(
        f"Unknown policy encoding: "
        f"{method}"
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
    policy_encoding,
):

    # --------------------------------------------------------
    # Claims:
    #
    # keep existing encoding frozen.
    # --------------------------------------------------------

    claim_z = float(
        row["C_t_z"]
    )

    claim_encoded = (
        np.clip(
            claim_z,
            -CLAIM_CLIP_Z,
            CLAIM_CLIP_Z,
        )
        / CLAIM_CLIP_Z
    )

    theta_claim = (
        ALPHA
        * claim_encoded
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
        2.0
        * np.pi,
    )

    # --------------------------------------------------------
    # Policy:
    #
    # this is the ONLY encoding changed.
    # --------------------------------------------------------

    policy_z = float(
        row["P_t_z"]
    )

    policy_encoded = (
        policy_map(
            policy_z,
            policy_encoding,
        )
    )

    theta_policy = (
        ALPHA
        * policy_encoded
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
    # Prepare single-qubit states
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

    # Qiskit ordering:
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
# Frozen couplings
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
# Reservoir unitary
# ============================================================

def build_unitary(
    couplings,
):

    qc = QuantumCircuit(
        N_QUBITS
    )

    dt = (
        DELTA_T
        / TROTTER_STEPS
    )

    for _ in range(
        TROTTER_STEPS
    ):

        for (
            q_i,
            q_j,
        ), J in (
            couplings.items()
        ):

            qc.rzz(
                2.0
                * J
                * dt,
                q_i,
                q_j,
            )

        for q in range(
            N_QUBITS
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
# Direct calculation helpers
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
                * O.T
            )
            for O in observables
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

    for O in observables:

        B = (
            U.conj().T
            @ O
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
# Generate CONT + all RWP windows
# ============================================================

def generate_protocols(
    df,
    target_column,
    input_states,
    policy_encoding,
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
            U=U,
            observables=observables,
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

    purity_values = {
        "CONT": [],
    }

    for W in WINDOWS:

        rows[
            f"RWP_W{W}"
        ] = []

        purity_values[
            f"RWP_W{W}"
        ] = []

    audit_feature_errors = []
    audit_memory_errors = []

    # ========================================================
    # Chronological daily loop
    # ========================================================

    for t, phi in enumerate(
        input_states
    ):

        kraus = (
            build_kraus(
                phi=phi,
                U4=U4,
            )
        )

        S = (
            kraus_to_superoperator(
                kraus
            )
        )

        L = (
            build_feature_map(
                phi=phi,
                transformed=transformed,
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

        rho_cont = (
            cont_new.reshape(
                D_MEMORY,
                D_MEMORY,
            )
        )

        cont_purity = float(
            np.real(
                np.trace(
                    rho_cont
                    @ rho_cont
                )
            )
        )

        row = {
            "policy_encoding": (
                policy_encoding
            ),

            "protocol": (
                "CONT"
            ),

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

        purity_values[
            "CONT"
        ].append(
            cont_purity
        )

        # ----------------------------------------------------
        # RWP features
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
                    for W in valid_windows
                ],
                dtype=int,
            )

            previous_memory = (
                age_vectors[
                    indices
                ]
            )

            feature_batch = (
                previous_memory
                @ L.T
            )

        # Save previous ages before updating.
        old_age_vectors = (
            age_vectors.copy()
        )

        # Propagate all histories one day.
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
                "policy_encoding": (
                    policy_encoding
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

            purity_values[
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
                rho_memory=(
                    previous_rho
                ),
                phi=phi,
                U=U,
                observables=observables,
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

    # ========================================================
    # Convert tables
    # ========================================================

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
    ) in purity_values.items():

        values = np.asarray(
            values,
            dtype=float,
        )

        purity_rows.append(
            {
                "policy_encoding": (
                    policy_encoding
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
                "configuration": (
                    configuration
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
# Policy transform diagnostics
# ============================================================

def encoding_diagnostics(
    df,
):

    split = (
        df[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    groups = {
        "train": (
            split == "train"
        ).to_numpy(),

        "validation": (
            split.isin(
                [
                    "validation",
                    "val",
                ]
            )
        ).to_numpy(),

        "test": (
            split == "test"
        ).to_numpy(),
    }

    policy_z = (
        df[
            "P_t_z"
        ]
        .to_numpy(
            dtype=float
        )
    )

    rows = []

    for method in (
        POLICY_ENCODINGS
    ):

        encoded = np.asarray(
            [
                policy_map(
                    z,
                    method,
                )
                for z in policy_z
            ],
            dtype=float,
        )

        for (
            group_name,
            mask,
        ) in groups.items():

            z_current = (
                policy_z[
                    mask
                ]
            )

            encoded_current = (
                encoded[
                    mask
                ]
            )

            exact_upper = int(
                np.sum(
                    np.isclose(
                        encoded_current,
                        1.0,
                        atol=1e-14,
                    )
                )
            )

            exact_lower = int(
                np.sum(
                    np.isclose(
                        encoded_current,
                        -1.0,
                        atol=1e-14,
                    )
                )
            )

            unique_rounded = int(
                len(
                    np.unique(
                        np.round(
                            encoded_current,
                            decimals=12,
                        )
                    )
                )
            )

            rows.append(
                {
                    "policy_encoding": (
                        method
                    ),

                    "group": (
                        group_name
                    ),

                    "n": int(
                        len(
                            z_current
                        )
                    ),

                    "raw_z_min": float(
                        np.min(
                            z_current
                        )
                    ),

                    "raw_z_max": float(
                        np.max(
                            z_current
                        )
                    ),

                    "encoded_min": float(
                        np.min(
                            encoded_current
                        )
                    ),

                    "encoded_max": float(
                        np.max(
                            encoded_current
                        )
                    ),

                    "encoded_std": float(
                        np.std(
                            encoded_current,
                            ddof=0,
                        )
                    ),

                    "exact_plus_one": (
                        exact_upper
                    ),

                    "exact_minus_one": (
                        exact_lower
                    ),

                    "unique_encoded_12dp": (
                        unique_rounded
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Baseline Week-5 reference audit
# ============================================================

def reference_audit(
    generated_tables,
):

    references = {
        "CONT": (
            WEEK5_CONT_FILE
        ),

        "RWP_W7": (
            WEEK5_RWP7_FILE
        ),
    }

    errors = {}

    for (
        protocol,
        file_path,
    ) in references.items():

        if not file_path.exists():

            continue

        old = pd.read_csv(
            file_path
        )

        new = (
            generated_tables[
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

        feature_errors = []

        for feature in (
            FEATURE_COLUMNS
        ):

            feature_errors.append(
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
                feature_errors
            )
        )

    return errors


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 6 - STEP 6.4B")
    log("POLICY ENCODING SEARCH")
    log("F4 + M=2 + ALL TEMPORAL WINDOWS")
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
            "Expected 1692 samples."
        )

    # ========================================================
    # Encoding diagnostics
    # ========================================================

    diagnostics = (
        encoding_diagnostics(
            df
        )
    )

    diagnostics.to_csv(
        ENCODING_DIAGNOSTICS_OUTPUT,
        index=False,
    )

    log(
        "Policy encoding diagnostics:"
    )

    for _, row in (
        diagnostics.iterrows()
    ):

        log(
            f"  "
            f"{row['policy_encoding']:<10s} "
            f"{row['group']:<10s} "
            f"N={int(row['n']):4d} "
            f"raw=["
            f"{row['raw_z_min']:.3f},"
            f"{row['raw_z_max']:.3f}] "
            f"encoded=["
            f"{row['encoded_min']:.4f},"
            f"{row['encoded_max']:.4f}] "
            f"std="
            f"{row['encoded_std']:.4f} "
            f"+1="
            f"{int(row['exact_plus_one']):4d} "
            f"unique="
            f"{int(row['unique_encoded_12dp']):4d}"
        )

    # ========================================================
    # Frozen reservoir
    # ========================================================

    couplings = (
        load_couplings()
    )

    U = (
        build_unitary(
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

    log()

    log(
        f"Unitarity error = "
        f"{unitarity_error:.3e}"
    )

    observables = (
        build_observables()
    )

    all_cv = []
    all_results = []
    all_geometry = []
    all_purity = []

    # ========================================================
    # Encoding loop
    # ========================================================

    for policy_encoding in (
        POLICY_ENCODINGS
    ):

        log()
        log("=" * 80)

        log(
            f"POLICY ENCODING = "
            f"{policy_encoding}"
        )

        log("=" * 80)

        input_states = [
            encode_input_state(
                row=row,
                policy_encoding=(
                    policy_encoding
                ),
            )
            for _, row
            in df.iterrows()
        ]

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
            policy_encoding=(
                policy_encoding
            ),
            U=U,
            observables=(
                observables
            ),
        )

        log(
            f"Direct audit feature error = "
            f"{feature_error:.3e}"
        )

        log(
            f"Direct audit memory error  = "
            f"{memory_error:.3e}"
        )

        if (
            feature_error > 1e-10
            or
            memory_error > 1e-10
        ):

            raise RuntimeError(
                f"{policy_encoding}: "
                "superoperator audit FAILED."
            )

        # ----------------------------------------------------
        # Current clip3 must reproduce Week 5
        # ----------------------------------------------------

        if policy_encoding == "clip3":

            errors = (
                reference_audit(
                    tables
                )
            )

            log()

            log(
                "Week-5 baseline "
                "reference audit:"
            )

            for (
                protocol,
                error,
            ) in (
                errors.items()
            ):

                log(
                    f"  {protocol:<7s} "
                    f"max feature error = "
                    f"{error:.3e}"
                )

                if error > 1e-10:

                    raise RuntimeError(
                        "Week-5 reference "
                        "audit FAILED."
                    )

        # ----------------------------------------------------
        # Save generated features
        # ----------------------------------------------------

        combined = pd.concat(
            tables.values(),
            ignore_index=True,
        )

        combined.to_csv(
            RESULTS_DIR
            /
            (
                "06_04b_features_"
                f"{policy_encoding}.csv"
            ),
            index=False,
        )

        all_purity.append(
            purity_df
        )

        # ====================================================
        # Every temporal protocol
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
                    "Unexpected training "
                    "sample count."
                )

            if (
                n_val != 365
                or
                n_test != 232
            ):

                raise RuntimeError(
                    "Unexpected validation/"
                    "test count."
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
                f"{policy_encoding}_"
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
                    "policy_encoding": (
                        policy_encoding
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
                "policy_encoding"
            ] = policy_encoding

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
                    "policy_encoding": (
                        policy_encoding
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
                f"{configuration:<20s} "
                f"N={n_train:4d} "
                f"CV="
                f"{best_cv_mean:.6f} "
                f"(+/- "
                f"{best_cv_std:.6f}) "
                f"lambda="
                f"{best_lambda:<7g} "
                f"rank="
                f"{geometry['rank']:2d}/12 "
                f"Val="
                f"{metrics['RMSE']:.6f} "
                f"Bias="
                f"{metrics['Bias']:+.6f}"
            )

    # ========================================================
    # Save all results
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
    # Best window for each encoding
    # ========================================================

    log()
    log("=" * 80)

    log(
        "BEST TEMPORAL PROTOCOL "
        "WITHIN EACH POLICY ENCODING"
    )

    log("=" * 80)

    for policy_encoding in (
        POLICY_ENCODINGS
    ):

        subset = (
            results[
                results[
                    "policy_encoding"
                ]
                == policy_encoding
            ]
            .sort_values(
                "cv_rmse_mean"
            )
            .reset_index(
                drop=True
            )
        )

        best = subset.iloc[0]

        log()

        log(
            f"{policy_encoding:<10s}: "
            f"{best['protocol']} "
            f"CV="
            f"{best['cv_rmse_mean']:.6f} "
            f"(+/- "
            f"{best['cv_rmse_std']:.6f}) "
            f"Val="
            f"{best['RMSE']:.6f}"
        )

    # ========================================================
    # Overall ranking by TRAINING CV
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
        "OVERALL POLICY-ENCODING x "
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
            f"{row['configuration']:<20s} "
            f"N="
            f"{int(row['train_rows']):4d} "
            f"CV="
            f"{row['cv_rmse_mean']:.6f} "
            f"(+/- "
            f"{row['cv_rmse_std']:.6f}) "
            f"Val="
            f"{row['RMSE']:.6f} "
            f"Bias="
            f"{row['Bias']:+.6f}"
        )

    best = ranking.iloc[0]

    log()

    log(
        "Best configuration "
        "by training CV:"
    )

    log(
        f"  "
        f"{best['configuration']}"
    )

    log()

    log(
        "IMPORTANT:"
    )

    log(
        "  Only policy encoding "
        "was changed."
    )

    log(
        "  F4, M=2, reservoir "
        "topology and dynamics "
        "were frozen."
    )

    log(
        "  All temporal windows "
        "were tested."
    )

    log(
        "  Soft encodings preserve "
        "ordering beyond z=3 "
        "instead of collapsing "
        "all values to +1."
    )

    log(
        "  Primary ranking remains "
        "training-only CV."
    )

    log(
        "  2025 validation is an "
        "out-of-distribution "
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
        f"Saved encoding diagnostics to: "
        f"{ENCODING_DIAGNOSTICS_OUTPUT}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":

    main()