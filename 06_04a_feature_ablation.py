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
    / "06_04a_feature_ablation_cv.csv"
)

RESULT_OUTPUT = (
    RESULTS_DIR
    / "06_04a_feature_ablation_validation.csv"
)

GEOMETRY_OUTPUT = (
    RESULTS_DIR
    / "06_04a_feature_ablation_geometry.csv"
)

PURITY_OUTPUT = (
    RESULTS_DIR
    / "06_04a_feature_ablation_purity.csv"
)

CLIPPING_OUTPUT = (
    RESULTS_DIR
    / "06_04a_clipping_diagnostics.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "06_04a_summary.txt"
)


# ============================================================
# Search space
# ============================================================

FEATURE_SETS = [
    "F2",
    "F3",
    "F4",
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
# Frozen architecture
# ============================================================

N_INPUT = 4
N_MEMORY = 2
N_QUBITS = 6

D_INPUT = 2 ** N_INPUT
D_MEMORY = 2 ** N_MEMORY
D_TOTAL = 2 ** N_QUBITS

ALPHA = 1.0

H_X = 0.5

DELTA_T = 0.8

TROTTER_STEPS = 2

CLIP_Z = 3.0

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]


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
# F2 / F3 / F4 encoding
# ============================================================

def encode_input_state(
    row,
    feature_set,
):

    # --------------------------------------------------------
    # Claims
    # --------------------------------------------------------

    claim_z = float(
        row["C_t_z"]
    )

    claim_scaled = (
        np.clip(
            claim_z,
            -CLIP_Z,
            CLIP_Z,
        )
        / CLIP_Z
    )

    theta_claim = (
        ALPHA
        * claim_scaled
    )

    # --------------------------------------------------------
    # Day of week
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
    # Policy
    # --------------------------------------------------------

    if feature_set in (
        "F3",
        "F4",
    ):

        policy_z = float(
            row["P_t_z"]
        )

        policy_scaled = (
            np.clip(
                policy_z,
                -CLIP_Z,
                CLIP_Z,
            )
            / CLIP_Z
        )

        theta_policy = (
            ALPHA
            * policy_scaled
        )

    else:

        # F2:
        # no policy information injected
        theta_policy = 0.0

    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    if feature_set == "F4":

        theta_holiday = (
            np.pi
            * float(
                row[
                    "is_public_holiday_t_plus_1"
                ]
            )
        )

    else:

        # F2 / F3:
        # no holiday information injected
        theta_holiday = 0.0

    # --------------------------------------------------------
    # States
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
    # Qiskit ordering:
    #
    # q3 kron q2 kron q1 kron q0
    # --------------------------------------------------------

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

    rho[0, 0] = 1.0

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
            "Week-5 coupling "
            "topology mismatch."
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

    operator = matrices[0]

    for matrix in (
        matrices[1:]
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
#
# correctness audits only
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
# Kraus operators
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


# ============================================================
# Kraus -> superoperator
# ============================================================

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
# Effective observable maps
# ============================================================

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
# Generate CONT and every W
# ============================================================

def generate_protocols(
    df,
    target_column,
    input_states,
    feature_set,
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

    # --------------------------------------------------------
    # age_vectors[k]
    #
    # memory state after exactly k most
    # recent inputs, starting from |00>.
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
    # Daily loop
    # ========================================================

    for t, phi in enumerate(
        input_states
    ):

        kraus = build_kraus(
            phi=phi,
            U4=U4,
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

        cont_row = {
            "feature_set": (
                feature_set
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
            name,
            value,
        ) in zip(
            FEATURE_COLUMNS,
            cont_features,
        ):

            cont_row[
                name
            ] = float(
                np.real(
                    value
                )
            )

        rows[
            "CONT"
        ].append(
            cont_row
        )

        purities[
            "CONT"
        ].append(
            cont_purity
        )

        # ----------------------------------------------------
        # Available rewinding windows
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

            previous_memories = (
                age_vectors[
                    indices
                ]
            )

            feature_batch = (
                previous_memories
                @ L.T
            )

        # ----------------------------------------------------
        # Preserve previous ages for audit
        # ----------------------------------------------------

        old_age_vectors = (
            age_vectors.copy()
        )

        # ----------------------------------------------------
        # Propagate every age through today's channel
        # ----------------------------------------------------

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
        # Store each W
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
                "feature_set": (
                    feature_set
                ),

                "protocol": (
                    f"RWP_W{W}"
                ),

                "window_W": (
                    W
                ),

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
        # Direct audit:
        #
        # test actual W=7 memory at two dates
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

        # ----------------------------------------------------
        # Advance CONT
        # ----------------------------------------------------

        cont_vec = (
            cont_new
        )

    # ========================================================
    # Tables
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

                "lambda": (
                    alpha
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
            rmse
            /
            train_std
        ),
    }


# ============================================================
# Clipping diagnostics
# ============================================================

def clipping_diagnostics(
    df,
):

    rows = []

    split_series = (
        df[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    groups = {
        "all": np.ones(
            len(df),
            dtype=bool,
        ),

        "train": (
            split_series
            == "train"
        ).to_numpy(),

        "validation": (
            split_series.isin(
                [
                    "validation",
                    "val",
                ]
            )
        ).to_numpy(),

        "test": (
            split_series
            == "test"
        ).to_numpy(),
    }

    for variable in [
        "C_t_z",
        "P_t_z",
    ]:

        values = (
            df[
                variable
            ]
            .to_numpy(
                dtype=float
            )
        )

        for (
            group_name,
            mask,
        ) in groups.items():

            current = (
                values[
                    mask
                ]
            )

            clipped = (
                np.abs(
                    current
                )
                > CLIP_Z
            )

            rows.append(
                {
                    "variable": (
                        variable
                    ),

                    "group": (
                        group_name
                    ),

                    "n": int(
                        len(
                            current
                        )
                    ),

                    "n_clipped": int(
                        np.sum(
                            clipped
                        )
                    ),

                    "pct_clipped": float(
                        100.0
                        * np.mean(
                            clipped
                        )
                    ),

                    "min_z": float(
                        np.min(
                            current
                        )
                    ),

                    "max_z": float(
                        np.max(
                            current
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Reference comparison for F4
# ============================================================

def compare_reference(
    generated_tables,
):

    if (
        not WEEK5_CONT_FILE.exists()
        or
        not WEEK5_RWP7_FILE.exists()
    ):

        return None

    errors = {}

    references = {
        "CONT": (
            WEEK5_CONT_FILE
        ),

        "RWP_W7": (
            WEEK5_RWP7_FILE
        ),
    }

    for (
        protocol,
        file_path,
    ) in references.items():

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
    log("WEEK 6 - STEP 6.4A")
    log("FEATURE-INFORMATION ABLATION")
    log("FIXED SIX-QUBIT TOPOLOGY")
    log("F2 vs F3 vs F4 x ALL TEMPORAL PROTOCOLS")
    log("=" * 80)
    log()

    # --------------------------------------------------------
    # Load data
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
    # Clipping diagnostic
    # ========================================================

    clipping_df = (
        clipping_diagnostics(
            df
        )
    )

    clipping_df.to_csv(
        CLIPPING_OUTPUT,
        index=False,
    )

    log(
        "Current +/-3 z clipping:"
    )

    for _, row in (
        clipping_df.iterrows()
    ):

        log(
            f"  {row['variable']:<6s} "
            f"{row['group']:<10s} "
            f"N={int(row['n']):4d} "
            f"clipped="
            f"{int(row['n_clipped']):4d} "
            f"("
            f"{row['pct_clipped']:.2f}%) "
            f"range=["
            f"{row['min_z']:.3f}, "
            f"{row['max_z']:.3f}]"
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
    # F2 / F3 / F4 loop
    # ========================================================

    for feature_set in (
        FEATURE_SETS
    ):

        log()

        log("=" * 80)

        log(
            f"FEATURE SET = "
            f"{feature_set}"
        )

        log("=" * 80)

        # ----------------------------------------------------
        # Encode all days
        # ----------------------------------------------------

        input_states = [
            encode_input_state(
                row=row,
                feature_set=(
                    feature_set
                ),
            )
            for _, row
            in df.iterrows()
        ]

        # ----------------------------------------------------
        # Generate all temporal protocols
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
                input_states
            ),
            feature_set=(
                feature_set
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
                f"{feature_set}: "
                "superoperator audit FAILED."
            )

        # ----------------------------------------------------
        # F4 must reproduce Week 5
        # ----------------------------------------------------

        if feature_set == "F4":

            reference_errors = (
                compare_reference(
                    tables
                )
            )

            if (
                reference_errors
                is not None
            ):

                log()

                log(
                    "F4 reference audits:"
                )

                for (
                    protocol,
                    error,
                ) in (
                    reference_errors.items()
                ):

                    log(
                        f"  {protocol:<7s} "
                        f"max feature error = "
                        f"{error:.3e}"
                    )

                    if error > 1e-10:

                        raise RuntimeError(
                            "F4 Week-5 "
                            "reference audit FAILED."
                        )

        # ----------------------------------------------------
        # Save features
        # ----------------------------------------------------

        combined = pd.concat(
            tables.values(),
            ignore_index=True,
        )

        feature_file = (
            RESULTS_DIR
            /
            (
                f"06_04a_features_"
                f"{feature_set}.csv"
            )
        )

        combined.to_csv(
            feature_file,
            index=False,
        )

        all_purity.append(
            purity_df
        )

        # ====================================================
        # Evaluate every temporal protocol
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
                split == "train"
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
                split == "test"
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
                    (W - 1)
                )

            if (
                n_train
                !=
                expected_train
            ):

                raise RuntimeError(
                    f"{feature_set} "
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
                f"{feature_set}_"
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
                    "feature_set": (
                        feature_set
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
            # Ridge CV
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
                "feature_set"
            ] = feature_set

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
                    "feature_set": (
                        feature_set
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
                f"{configuration:<14s} "
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
    # Best temporal protocol per feature set
    # ========================================================

    log()

    log("=" * 80)

    log(
        "BEST TEMPORAL PROTOCOL "
        "WITHIN EACH FEATURE SET"
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
            f"{feature_set}: "
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
        "OVERALL F2/F3/F4 x "
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
            f"{row['configuration']:<14s} "
            f"N="
            f"{int(row['train_rows']):4d} "
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
        "Best configuration "
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
        "  Six-qubit topology was "
        "kept fixed for F2/F3/F4."
    )

    log(
        "  F2/F3 inactive inputs were "
        "prepared as |0> via Ry(0)."
    )

    log(
        "  Therefore this is an "
        "INFORMATION ABLATION, not "
        "yet a hardware-resource comparison."
    )

    log(
        "  All temporal windows were "
        "tested."
    )

    log(
        "  2025 validation is "
        "diagnostic only."
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
        f"Saved clipping diagnostics to: "
        f"{CLIPPING_OUTPUT}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":

    main()