from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit

from sqlalchemy import text

from db_config import engine


# =============================================================================
# WEEK 4 - STEP 4.2b
# ESN SEED ROBUSTNESS
# =============================================================================
#
# Research question
# -----------------
#
# The best Step 4.2 ESN obtained:
#
#     validation RMSE = 4.856483
#
# versus:
#
#     Ridge F4 RMSE = 4.874844
#
# i.e. only approximately 0.377% improvement.
#
# Because an ESN reservoir is randomly generated, one random seed may be
# unusually good or unusually bad.
#
# Therefore this experiment asks:
#
#     Is the selected ESN architecture robust across independent
#     random reservoir realizations?
#
#
# IMPORTANT:
#
# We DO NOT:
#
#     choose the best seed.
#
# Instead we report:
#
#     mean RMSE
#     std RMSE
#     median RMSE
#     min / max
#     number of seeds beating Ridge F4
#
#
# Architecture is FIXED.
#
# Only:
#
#     W_r
#     W_in
#
# change with reservoir seed.
#
# Ridge alpha is allowed to change because it is selected by
# TRAINING-ONLY chronological cross-validation.
#
# Validation is NEVER used to select Ridge alpha.
#
# Test data are NEVER loaded.
#
# =============================================================================


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# INPUT FILES
# =============================================================================

TEMPORAL_FILE = (
    RESULTS_DIR
    / "04_01_windows_F4_W14.npz"
)

PREPROCESSED_FILE = (
    RESULTS_DIR
    / "03_01_preprocessed_samples.csv"
)

RIDGE_METRICS_FILE = (
    RESULTS_DIR
    / "03_03_ridge_validation_metrics.csv"
)

NAIVE_METRICS_FILE = (
    RESULTS_DIR
    / "03_02_naive_baseline_metrics.csv"
)

ESN_MAIN_RESULTS_FILE = (
    RESULTS_DIR
    / "04_02_esn_validation_metrics.csv"
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

SEED_RESULTS_FILE = (
    RESULTS_DIR
    / "04_02b_esn_seed_robustness.csv"
)

SEED_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "04_02b_esn_seed_predictions.csv"
)

SEED_SUMMARY_FILE = (
    RESULTS_DIR
    / "04_02b_esn_seed_summary.csv"
)


# =============================================================================
# FIXED SELECTED ARCHITECTURE
# =============================================================================
#
# We will verify these values against the best result saved by Step 4.2.
# =============================================================================

FEATURE_SET = "F4"
WINDOW_LENGTH = 14

RESERVOIR_SIZE = 100

CONNECTIVITY = 0.10

SPECTRAL_RADIUS = 0.90

INPUT_SCALING = 0.25

LEAK_RATE = 1.00


# =============================================================================
# PREDEFINED RESERVOIR SEEDS
# =============================================================================
#
# These seeds are fixed BEFORE seeing their validation results.
#
# Seed 42 is retained because it was the original Step 4.2 reservoir.
# The remaining nine provide independent random reservoir realizations.
#
# We must NOT choose the best-performing seed afterward.
# =============================================================================

RESERVOIR_SEEDS = [
    42,
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
]


# =============================================================================
# RIDGE REGULARIZATION
# =============================================================================

RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    30.0,
    100.0,
]


# =============================================================================
# TRAINING-ONLY CHRONOLOGICAL CV
# =============================================================================

N_CV_SPLITS = 5


# =============================================================================
# HELPER
# =============================================================================

def section(title):

    print()

    print(
        "=" * 110
    )

    print(
        title
    )

    print(
        "=" * 110
    )


# =============================================================================
# TARGET COLUMN
# =============================================================================

def resolve_target_column(df):

    candidates = [
        "target_property_damage_claim_count",
        "property_damage_claim_count_t_plus_1",
        "property_damage_claim_count_target",
        "target_claim_count",
        "C_t_plus_1",
    ]

    for column in candidates:

        if column in df.columns:

            return column


    raise ValueError(
        "Could not identify target column."
    )


# =============================================================================
# LOAD REFERENCE INFORMATION
# =============================================================================

def load_reference_context():

    required_files = [
        TEMPORAL_FILE,
        PREPROCESSED_FILE,
        RIDGE_METRICS_FILE,
        NAIVE_METRICS_FILE,
        ESN_MAIN_RESULTS_FILE,
    ]


    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                file_path
            )


    # -------------------------------------------------------------------------
    # CANONICAL PREPROCESSED DATA
    # -------------------------------------------------------------------------

    df = pd.read_csv(
        PREPROCESSED_FILE,
        parse_dates=[
            "input_date",
            "target_date",
        ],
    )


    target_column = (
        resolve_target_column(
            df
        )
    )


    # -------------------------------------------------------------------------
    # TRAINING TARGET STANDARD DEVIATION
    # -------------------------------------------------------------------------

    y_train_reference = (
        df.loc[
            df["split"] == "train",
            target_column,
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    train_target_std = float(
        np.std(
            y_train_reference,
            ddof=0,
        )
    )


    # -------------------------------------------------------------------------
    # VALIDATION DATES
    # -------------------------------------------------------------------------

    validation_dates = (
        df.loc[
            df["split"] == "validation",
            "target_date",
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .to_numpy(
            dtype="U10"
        )
    )


    # -------------------------------------------------------------------------
    # RIDGE F4 REFERENCE
    # -------------------------------------------------------------------------

    ridge_df = pd.read_csv(
        RIDGE_METRICS_FILE
    )


    ridge_f4 = ridge_df[
        ridge_df[
            "feature_set"
        ]
        == "F4"
    ]


    if len(
        ridge_f4
    ) != 1:

        raise RuntimeError(
            "Could not uniquely identify "
            "Ridge F4."
        )


    ridge_f4_rmse = float(
        ridge_f4.iloc[0][
            "validation_rmse"
        ]
    )


    # -------------------------------------------------------------------------
    # SEASONAL NAIVE REFERENCE
    # -------------------------------------------------------------------------

    naive_df = pd.read_csv(
        NAIVE_METRICS_FILE
    )


    seasonal_row = naive_df[
        naive_df[
            "model"
        ]
        .astype(str)
        .str.contains(
            "seasonal",
            case=False,
            na=False,
        )
    ]


    if len(
        seasonal_row
    ) != 1:

        raise RuntimeError(
            "Could not uniquely identify "
            "seasonal-naive model."
        )


    seasonal_rmse = float(
        seasonal_row.iloc[0][
            "rmse"
        ]
    )


    # -------------------------------------------------------------------------
    # VERIFY SELECTED ESN ARCHITECTURE
    # -------------------------------------------------------------------------

    esn_df = pd.read_csv(
        ESN_MAIN_RESULTS_FILE
    )


    best_esn = (
        esn_df
        .sort_values(
            "validation_rmse"
        )
        .iloc[0]
    )


    expected = {
        "feature_set":
            FEATURE_SET,

        "window_length":
            WINDOW_LENGTH,

        "reservoir_size":
            RESERVOIR_SIZE,

        "requested_connectivity":
            CONNECTIVITY,

        "spectral_radius":
            SPECTRAL_RADIUS,

        "input_scaling":
            INPUT_SCALING,

        "leak_rate":
            LEAK_RATE,
    }


    if (
        str(
            best_esn[
                "feature_set"
            ]
        )
        != FEATURE_SET
    ):

        raise RuntimeError(
            "Saved best ESN feature set "
            "does not match expected F4."
        )


    if int(
        best_esn[
            "window_length"
        ]
    ) != WINDOW_LENGTH:

        raise RuntimeError(
            "Saved best ESN window "
            "does not match W=14."
        )


    if int(
        best_esn[
            "reservoir_size"
        ]
    ) != RESERVOIR_SIZE:

        raise RuntimeError(
            "Saved reservoir size "
            "does not match."
        )


    if not np.isclose(
        float(
            best_esn[
                "requested_connectivity"
            ]
        ),
        CONNECTIVITY,
    ):

        raise RuntimeError(
            "Saved connectivity "
            "does not match p=0.10."
        )


    if not np.isclose(
        float(
            best_esn[
                "spectral_radius"
            ]
        ),
        SPECTRAL_RADIUS,
    ):

        raise RuntimeError(
            "Saved spectral radius "
            "does not match rho=0.9."
        )


    if not np.isclose(
        float(
            best_esn[
                "input_scaling"
            ]
        ),
        INPUT_SCALING,
    ):

        raise RuntimeError(
            "Saved input scaling "
            "does not match s=0.25."
        )


    if not np.isclose(
        float(
            best_esn[
                "leak_rate"
            ]
        ),
        LEAK_RATE,
    ):

        raise RuntimeError(
            "Saved leak rate "
            "does not match alpha=1."
        )


    original_seed42_rmse = float(
        best_esn[
            "validation_rmse"
        ]
    )


    return (
        train_target_std,
        validation_dates,
        ridge_f4_rmse,
        seasonal_rmse,
        original_seed42_rmse,
    )


# =============================================================================
# RANDOM RESERVOIR
# =============================================================================

def generate_reservoir(
    reservoir_size,
    input_dimension,
    connectivity,
    spectral_radius,
    input_scaling,
    seed,
):
    """
    Generate one complete ESN reservoir for a specific random seed.

    ------------------------------------------------------------
    Recurrent matrix
    ------------------------------------------------------------

    1. Generate random weights.
    2. Generate sparse connectivity mask.
    3. Calculate spectral radius.
    4. Rescale to requested rho.

    ------------------------------------------------------------
    Input matrix
    ------------------------------------------------------------

    Generate an independent random input projection using the same
    reservoir seed but a deterministic offset.
    """


    # -------------------------------------------------------------------------
    # RECURRENT MATRIX
    # -------------------------------------------------------------------------

    rng_recurrent = (
        np.random.default_rng(
            seed
        )
    )


    W0 = (
        rng_recurrent.uniform(
            low=-1.0,
            high=1.0,
            size=(
                reservoir_size,
                reservoir_size,
            ),
        )
    )


    mask = (
        rng_recurrent.random(
            (
                reservoir_size,
                reservoir_size,
            )
        )
        < connectivity
    )


    W0 *= mask


    actual_connectivity = float(
        np.count_nonzero(
            W0
        )
        / W0.size
    )


    eigenvalues = (
        np.linalg.eigvals(
            W0
        )
    )


    original_rho = float(
        np.max(
            np.abs(
                eigenvalues
            )
        )
    )


    if original_rho <= 0:

        raise RuntimeError(
            f"Seed {seed}: "
            "zero spectral radius."
        )


    W_reservoir = (
        spectral_radius
        / original_rho
        * W0
    )


    # -------------------------------------------------------------------------
    # VERIFY ACTUAL SPECTRAL RADIUS
    # -------------------------------------------------------------------------

    final_eigenvalues = (
        np.linalg.eigvals(
            W_reservoir
        )
    )


    actual_rho = float(
        np.max(
            np.abs(
                final_eigenvalues
            )
        )
    )


    # -------------------------------------------------------------------------
    # SINGULAR VALUE
    # -------------------------------------------------------------------------

    singular_values = np.linalg.svd(
        W_reservoir,
        compute_uv=False,
    )


    sigma_max = float(
        singular_values[0]
    )


    sigma_rho_ratio = float(
        sigma_max
        / actual_rho
    )


    # -------------------------------------------------------------------------
    # INPUT MATRIX
    # -------------------------------------------------------------------------

    rng_input = (
        np.random.default_rng(
            seed
            + 10000
            + input_dimension
        )
    )


    W_input = (
        rng_input.uniform(
            low=-1.0,
            high=1.0,
            size=(
                reservoir_size,
                input_dimension,
            ),
        )
    )


    # Same dimensionality normalization as Step 4.2.
    W_input /= np.sqrt(
        input_dimension
    )


    W_input *= (
        input_scaling
    )


    return (
        W_reservoir,
        W_input,
        actual_connectivity,
        actual_rho,
        sigma_max,
        sigma_rho_ratio,
    )


# =============================================================================
# ESN FORWARD PASS
# =============================================================================

def compute_final_states(
    X,
    W_reservoir,
    W_input,
    leak_rate,
    calculate_diagnostics=False,
):
    """
    Process temporal windows.

    For every sample:

        r_0 = 0

    Candidate:

        r_tilde =
            tanh(
                W_in u_t
                +
                W_r r_(t-1)
            )

    Leaky update:

        r_t =
            (1-alpha) r_(t-1)
            +
            alpha r_tilde

    We keep only:

        r_W
    """


    n_samples = (
        X.shape[0]
    )


    reservoir_size = (
        W_reservoir.shape[0]
    )


    states = np.zeros(
        (
            n_samples,
            reservoir_size,
        ),
        dtype=np.float64,
    )


    saturation_count = 0

    candidate_count = 0

    abs_candidate_sum = 0.0


    for time_index in range(
        X.shape[1]
    ):


        u_t = (
            X[
                :,
                time_index,
                :
            ]
        )


        input_drive = (
            u_t
            @ W_input.T
        )


        recurrent_drive = (
            states
            @ W_reservoir.T
        )


        candidate = np.tanh(
            input_drive
            +
            recurrent_drive
        )


        if calculate_diagnostics:

            absolute_candidate = (
                np.abs(
                    candidate
                )
            )


            saturation_count += int(
                np.count_nonzero(
                    absolute_candidate
                    >= 0.95
                )
            )


            candidate_count += int(
                candidate.size
            )


            abs_candidate_sum += float(
                np.sum(
                    absolute_candidate
                )
            )


        states = (
            (1.0 - leak_rate)
            * states
            +
            leak_rate
            * candidate
        )


    diagnostics = None


    if calculate_diagnostics:

        diagnostics = {

            "saturation_fraction":
                (
                    saturation_count
                    / candidate_count
                ),

            "mean_abs_candidate_state":
                (
                    abs_candidate_sum
                    / candidate_count
                ),

            "mean_abs_final_state":
                float(
                    np.mean(
                        np.abs(
                            states
                        )
                    )
                ),
        }


    return (
        states,
        diagnostics,
    )


# =============================================================================
# RIDGE ALPHA SELECTION
# =============================================================================

def choose_ridge_alpha(
    R_train,
    y_train,
):
    """
    Ridge alpha is selected independently for each random reservoir.

    Selection uses ONLY the 2022-2024 training period.

    Five chronological folds are used.
    """


    splitter = (
        TimeSeriesSplit(
            n_splits=
                N_CV_SPLITS
        )
    )


    results = []


    for ridge_alpha in (
        RIDGE_ALPHAS
    ):


        fold_rmses = []


        for (
            fit_indices,
            cv_indices,
        ) in splitter.split(
            R_train
        ):


            model = Ridge(
                alpha=
                    ridge_alpha,

                fit_intercept=
                    True,
            )


            model.fit(
                R_train[
                    fit_indices
                ],

                y_train[
                    fit_indices
                ],
            )


            prediction = (
                model.predict(
                    R_train[
                        cv_indices
                    ]
                )
            )


            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_train[
                            cv_indices
                        ],
                        prediction,
                    )
                )
            )


            fold_rmses.append(
                rmse
            )


        results.append({

            "ridge_alpha":
                float(
                    ridge_alpha
                ),

            "mean_cv_rmse":
                float(
                    np.mean(
                        fold_rmses
                    )
                ),

            "std_cv_rmse":
                float(
                    np.std(
                        fold_rmses,
                        ddof=0,
                    )
                ),
        })


    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "mean_cv_rmse",
                "std_cv_rmse",
                "ridge_alpha",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    best = (
        results_df.iloc[0]
    )


    return (
        float(
            best[
                "ridge_alpha"
            ]
        ),

        float(
            best[
                "mean_cv_rmse"
            ]
        ),

        float(
            best[
                "std_cv_rmse"
            ]
        ),
    )


# =============================================================================
# DATABASE TABLE
# =============================================================================

def recreate_database_table():

    with engine.begin() as connection:


        connection.execute(
            text("""
                DROP TABLE IF EXISTS
                    qrc_esn_seed_robustness
            """)
        )


        connection.execute(
            text("""
                CREATE TABLE
                    qrc_esn_seed_robustness
                (
                    esn_seed_result_id
                        BIGINT AUTO_INCREMENT
                        PRIMARY KEY,

                    feature_set_code
                        VARCHAR(10) NOT NULL,

                    window_length
                        INT NOT NULL,

                    reservoir_seed
                        INT NOT NULL,

                    reservoir_size
                        INT NOT NULL,

                    requested_connectivity
                        DOUBLE NOT NULL,

                    actual_connectivity
                        DOUBLE NOT NULL,

                    spectral_radius
                        DOUBLE NOT NULL,

                    max_singular_value
                        DOUBLE NOT NULL,

                    sigma_rho_ratio
                        DOUBLE NOT NULL,

                    input_scaling
                        DOUBLE NOT NULL,

                    leak_rate
                        DOUBLE NOT NULL,

                    ridge_alpha
                        DOUBLE NOT NULL,

                    training_cv_rmse
                        DOUBLE NOT NULL,

                    training_cv_std
                        DOUBLE NOT NULL,

                    candidate_saturation_fraction
                        DOUBLE NOT NULL,

                    validation_mae
                        DOUBLE NOT NULL,

                    validation_rmse
                        DOUBLE NOT NULL,

                    validation_nrmse
                        DOUBLE NOT NULL,

                    validation_bias
                        DOUBLE NOT NULL,

                    improvement_vs_ridge_f4_pct
                        DOUBLE NOT NULL,

                    improvement_vs_seasonal_pct
                        DOUBLE NOT NULL,

                    beats_ridge_f4
                        BOOLEAN NOT NULL,

                    model_artifact_path
                        VARCHAR(500) NOT NULL,

                    created_at
                        TIMESTAMP NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE KEY uq_esn_seed
                    (
                        feature_set_code,
                        window_length,
                        reservoir_seed
                    )
                )
            """)
        )


# =============================================================================
# INSERT DATABASE RESULT
# =============================================================================

def insert_database_result(
    row
):

    sql = text("""
        INSERT INTO
            qrc_esn_seed_robustness
        (
            feature_set_code,
            window_length,

            reservoir_seed,
            reservoir_size,

            requested_connectivity,
            actual_connectivity,

            spectral_radius,
            max_singular_value,
            sigma_rho_ratio,

            input_scaling,
            leak_rate,

            ridge_alpha,

            training_cv_rmse,
            training_cv_std,

            candidate_saturation_fraction,

            validation_mae,
            validation_rmse,
            validation_nrmse,
            validation_bias,

            improvement_vs_ridge_f4_pct,
            improvement_vs_seasonal_pct,

            beats_ridge_f4,

            model_artifact_path
        )
        VALUES
        (
            :feature_set_code,
            :window_length,

            :reservoir_seed,
            :reservoir_size,

            :requested_connectivity,
            :actual_connectivity,

            :spectral_radius,
            :max_singular_value,
            :sigma_rho_ratio,

            :input_scaling,
            :leak_rate,

            :ridge_alpha,

            :training_cv_rmse,
            :training_cv_std,

            :candidate_saturation_fraction,

            :validation_mae,
            :validation_rmse,
            :validation_nrmse,
            :validation_bias,

            :improvement_vs_ridge_f4_pct,
            :improvement_vs_seasonal_pct,

            :beats_ridge_f4,

            :model_artifact_path
        )
    """)


    with engine.begin() as connection:

        connection.execute(
            sql,
            row,
        )


# =============================================================================
# START
# =============================================================================

section(
    "WEEK 4 - STEP 4.2b: "
    "ESN SEED ROBUSTNESS"
)


# =============================================================================
# REFERENCES
# =============================================================================

(
    TRAIN_TARGET_STD,
    VALIDATION_DATES,
    RIDGE_F4_RMSE,
    SEASONAL_RMSE,
    ORIGINAL_SEED42_RMSE,
) = load_reference_context()


section(
    "FIXED ARCHITECTURE"
)


print(
    f"Feature set:          "
    f"{FEATURE_SET}"
)

print(
    f"Window length:        "
    f"{WINDOW_LENGTH}"
)

print(
    f"Reservoir size:       "
    f"{RESERVOIR_SIZE}"
)

print(
    f"Connectivity p:       "
    f"{CONNECTIVITY}"
)

print(
    f"Spectral radius rho:  "
    f"{SPECTRAL_RADIUS}"
)

print(
    f"Input scaling s:      "
    f"{INPUT_SCALING}"
)

print(
    f"Leak rate alpha:      "
    f"{LEAK_RATE}"
)

print()
print(
    f"Reservoir seeds:      "
    f"{RESERVOIR_SEEDS}"
)

print()
print(
    f"Ridge F4 RMSE:        "
    f"{RIDGE_F4_RMSE:.6f}"
)

print(
    f"Original seed-42 ESN: "
    f"{ORIGINAL_SEED42_RMSE:.6f}"
)


# =============================================================================
# LOAD TEMPORAL DATA
# =============================================================================

section(
    "TEMPORAL DATA"
)


with np.load(
    TEMPORAL_FILE,
    allow_pickle=False,
) as data:


    X_train = np.asarray(
        data[
            "X_train"
        ],
        dtype=np.float64,
    )


    y_train = np.asarray(
        data[
            "y_train"
        ],
        dtype=np.float64,
    )


    X_val = np.asarray(
        data[
            "X_val"
        ],
        dtype=np.float64,
    )


    y_val = np.asarray(
        data[
            "y_val"
        ],
        dtype=np.float64,
    )


print(
    f"X_train: "
    f"{X_train.shape}"
)

print(
    f"y_train: "
    f"{y_train.shape}"
)

print(
    f"X_val:   "
    f"{X_val.shape}"
)

print(
    f"y_val:   "
    f"{y_val.shape}"
)


input_dimension = int(
    X_train.shape[2]
)


if len(
    VALIDATION_DATES
) != len(
    y_val
):

    raise RuntimeError(
        "Validation date count mismatch."
    )


# =============================================================================
# DATABASE
# =============================================================================

section(
    "DATABASE INITIALIZATION"
)


recreate_database_table()


print(
    "Database table ready:"
)

print(
    "  qrc_esn_seed_robustness"
)


# =============================================================================
# SEED EXPERIMENT
# =============================================================================

seed_rows = []

prediction_rows = []


for seed in (
    RESERVOIR_SEEDS
):


    section(
        f"RESERVOIR SEED {seed}"
    )


    # -------------------------------------------------------------------------
    # GENERATE RESERVOIR
    # -------------------------------------------------------------------------

    (
        W_reservoir,
        W_input,
        actual_connectivity,
        actual_rho,
        sigma_max,
        sigma_rho_ratio,
    ) = generate_reservoir(

        reservoir_size=
            RESERVOIR_SIZE,

        input_dimension=
            input_dimension,

        connectivity=
            CONNECTIVITY,

        spectral_radius=
            SPECTRAL_RADIUS,

        input_scaling=
            INPUT_SCALING,

        seed=
            seed,
    )


    print(
        f"Actual connectivity: "
        f"{actual_connectivity:.6f}"
    )

    print(
        f"Actual rho:          "
        f"{actual_rho:.6f}"
    )

    print(
        f"Sigma max:           "
        f"{sigma_max:.6f}"
    )

    print(
        f"Sigma/rho:           "
        f"{sigma_rho_ratio:.6f}"
    )


    # -------------------------------------------------------------------------
    # TRAIN RESERVOIR FEATURES
    # -------------------------------------------------------------------------

    (
        R_train,
        diagnostics,
    ) = compute_final_states(

        X=
            X_train,

        W_reservoir=
            W_reservoir,

        W_input=
            W_input,

        leak_rate=
            LEAK_RATE,

        calculate_diagnostics=
            True,
    )


    # -------------------------------------------------------------------------
    # SELECT RIDGE ALPHA USING TRAINING ONLY
    # -------------------------------------------------------------------------

    (
        best_ridge_alpha,
        cv_rmse,
        cv_std,
    ) = choose_ridge_alpha(

        R_train=
            R_train,

        y_train=
            y_train,
    )


    print(
        f"Selected Ridge alpha: "
        f"{best_ridge_alpha}"
    )

    print(
        f"Training CV RMSE:      "
        f"{cv_rmse:.6f}"
    )

    print(
        f"Training CV std:       "
        f"{cv_std:.6f}"
    )


    # -------------------------------------------------------------------------
    # VALIDATION RESERVOIR FEATURES
    # -------------------------------------------------------------------------

    (
        R_val,
        _,
    ) = compute_final_states(

        X=
            X_val,

        W_reservoir=
            W_reservoir,

        W_input=
            W_input,

        leak_rate=
            LEAK_RATE,

        calculate_diagnostics=
            False,
    )


    # -------------------------------------------------------------------------
    # FIT FINAL READOUT ON ALL TRAINING DATA
    # -------------------------------------------------------------------------

    readout = Ridge(

        alpha=
            best_ridge_alpha,

        fit_intercept=
            True,
    )


    readout.fit(
        R_train,
        y_train,
    )


    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    prediction = (
        readout.predict(
            R_val
        )
    )


    validation_mae = float(
        mean_absolute_error(
            y_val,
            prediction,
        )
    )


    validation_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_val,
                prediction,
            )
        )
    )


    validation_nrmse = float(
        validation_rmse
        / TRAIN_TARGET_STD
    )


    validation_bias = float(
        np.mean(
            prediction
            - y_val
        )
    )


    improvement_vs_ridge = float(
        100.0
        * (
            RIDGE_F4_RMSE
            - validation_rmse
        )
        / RIDGE_F4_RMSE
    )


    improvement_vs_seasonal = float(
        100.0
        * (
            SEASONAL_RMSE
            - validation_rmse
        )
        / SEASONAL_RMSE
    )


    beats_ridge = bool(
        validation_rmse
        < RIDGE_F4_RMSE
    )


    # -------------------------------------------------------------------------
    # SAVE THIS EXACT RESERVOIR
    # -------------------------------------------------------------------------

    model_artifact = (
        RESULTS_DIR
        / (
            "04_02b_esn_seed_"
            f"{seed}.npz"
        )
    )


    np.savez_compressed(

        model_artifact,

        W_reservoir=
            W_reservoir,

        W_input=
            W_input,

        readout_coef=
            np.asarray(
                readout.coef_,
                dtype=np.float64,
            ),

        readout_intercept=
            np.asarray(
                [
                    readout.intercept_
                ],
                dtype=np.float64,
            ),

        reservoir_seed=
            np.asarray(
                [seed],
                dtype=np.int64,
            ),

        ridge_alpha=
            np.asarray(
                [
                    best_ridge_alpha
                ],
                dtype=np.float64,
            ),
    )


    # -------------------------------------------------------------------------
    # RESULT
    # -------------------------------------------------------------------------

    row = {

        "feature_set":
            FEATURE_SET,

        "window_length":
            WINDOW_LENGTH,

        "reservoir_seed":
            seed,

        "reservoir_size":
            RESERVOIR_SIZE,

        "requested_connectivity":
            CONNECTIVITY,

        "actual_connectivity":
            actual_connectivity,

        "spectral_radius":
            actual_rho,

        "max_singular_value":
            sigma_max,

        "sigma_rho_ratio":
            sigma_rho_ratio,

        "input_scaling":
            INPUT_SCALING,

        "leak_rate":
            LEAK_RATE,

        "ridge_alpha":
            best_ridge_alpha,

        "training_cv_rmse":
            cv_rmse,

        "training_cv_std":
            cv_std,

        "candidate_saturation_fraction":
            diagnostics[
                "saturation_fraction"
            ],

        "validation_mae":
            validation_mae,

        "validation_rmse":
            validation_rmse,

        "validation_nrmse":
            validation_nrmse,

        "validation_bias":
            validation_bias,

        "improvement_vs_ridge_f4_pct":
            improvement_vs_ridge,

        "improvement_vs_seasonal_pct":
            improvement_vs_seasonal,

        "beats_ridge_f4":
            beats_ridge,

        "model_artifact_path":
            model_artifact.as_posix(),
    }


    seed_rows.append(
        row
    )


    # -------------------------------------------------------------------------
    # DATABASE
    # -------------------------------------------------------------------------

    db_row = {

        "feature_set_code":
            FEATURE_SET,

        "window_length":
            WINDOW_LENGTH,

        "reservoir_seed":
            seed,

        "reservoir_size":
            RESERVOIR_SIZE,

        "requested_connectivity":
            CONNECTIVITY,

        "actual_connectivity":
            actual_connectivity,

        "spectral_radius":
            actual_rho,

        "max_singular_value":
            sigma_max,

        "sigma_rho_ratio":
            sigma_rho_ratio,

        "input_scaling":
            INPUT_SCALING,

        "leak_rate":
            LEAK_RATE,

        "ridge_alpha":
            best_ridge_alpha,

        "training_cv_rmse":
            cv_rmse,

        "training_cv_std":
            cv_std,

        "candidate_saturation_fraction":
            diagnostics[
                "saturation_fraction"
            ],

        "validation_mae":
            validation_mae,

        "validation_rmse":
            validation_rmse,

        "validation_nrmse":
            validation_nrmse,

        "validation_bias":
            validation_bias,

        "improvement_vs_ridge_f4_pct":
            improvement_vs_ridge,

        "improvement_vs_seasonal_pct":
            improvement_vs_seasonal,

        "beats_ridge_f4":
            beats_ridge,

        "model_artifact_path":
            model_artifact.as_posix(),
    }


    insert_database_result(
        db_row
    )


    # -------------------------------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------------------------------

    for (
        date,
        actual,
        predicted,
    ) in zip(
        VALIDATION_DATES,
        y_val,
        prediction,
    ):


        prediction_rows.append({

            "reservoir_seed":
                seed,

            "target_date":
                str(
                    date
                ),

            "actual":
                float(
                    actual
                ),

            "prediction":
                float(
                    predicted
                ),
        })


    # -------------------------------------------------------------------------
    # CONSOLE
    # -------------------------------------------------------------------------

    print()

    print(
        f"Validation MAE:      "
        f"{validation_mae:.6f}"
    )

    print(
        f"Validation RMSE:     "
        f"{validation_rmse:.6f}"
    )

    print(
        f"Validation NRMSE:    "
        f"{validation_nrmse:.6f}"
    )

    print(
        f"Validation bias:     "
        f"{validation_bias:.6f}"
    )

    print(
        f"vs Ridge F4:         "
        f"{improvement_vs_ridge:.3f}%"
    )

    print(
        f"vs seasonal naive:   "
        f"{improvement_vs_seasonal:.3f}%"
    )

    print(
        f"Beats Ridge F4:      "
        f"{beats_ridge}"
    )


# =============================================================================
# SAVE SEED RESULTS
# =============================================================================

seed_df = (
    pd.DataFrame(
        seed_rows
    )
)


prediction_df = (
    pd.DataFrame(
        prediction_rows
    )
)


seed_df.to_csv(
    SEED_RESULTS_FILE,
    index=False,
)


prediction_df.to_csv(
    SEED_PREDICTIONS_FILE,
    index=False,
)


# =============================================================================
# ROBUSTNESS SUMMARY
# =============================================================================

section(
    "ESN SEED ROBUSTNESS SUMMARY"
)


rmse_values = (
    seed_df[
        "validation_rmse"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


mae_values = (
    seed_df[
        "validation_mae"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


mean_rmse = float(
    np.mean(
        rmse_values
    )
)


std_rmse = float(
    np.std(
        rmse_values,
        ddof=1,
    )
)


median_rmse = float(
    np.median(
        rmse_values
    )
)


min_rmse = float(
    np.min(
        rmse_values
    )
)


max_rmse = float(
    np.max(
        rmse_values
    )
)


mean_mae = float(
    np.mean(
        mae_values
    )
)


n_beating_ridge = int(
    np.sum(
        rmse_values
        < RIDGE_F4_RMSE
    )
)


fraction_beating_ridge = float(
    n_beating_ridge
    / len(
        rmse_values
    )
)


mean_improvement_vs_ridge = float(
    100.0
    * (
        RIDGE_F4_RMSE
        - mean_rmse
    )
    / RIDGE_F4_RMSE
)


# =============================================================================
# APPROXIMATE 95% CI FOR MEAN RMSE
#
# With 10 seeds and unknown population variance we use a t critical value
# for df = 9:
#
#     t_(0.975, 9) ~= 2.262
#
# =============================================================================

T_CRITICAL_95_DF9 = 2.262


standard_error_rmse = float(
    std_rmse
    / np.sqrt(
        len(
            rmse_values
        )
    )
)


ci95_low = float(
    mean_rmse
    - T_CRITICAL_95_DF9
    * standard_error_rmse
)


ci95_high = float(
    mean_rmse
    + T_CRITICAL_95_DF9
    * standard_error_rmse
)


summary_df = pd.DataFrame(
    [
        {
            "feature_set":
                FEATURE_SET,

            "window_length":
                WINDOW_LENGTH,

            "n_seeds":
                len(
                    RESERVOIR_SEEDS
                ),

            "mean_validation_rmse":
                mean_rmse,

            "std_validation_rmse":
                std_rmse,

            "median_validation_rmse":
                median_rmse,

            "min_validation_rmse":
                min_rmse,

            "max_validation_rmse":
                max_rmse,

            "mean_validation_mae":
                mean_mae,

            "ridge_f4_rmse":
                RIDGE_F4_RMSE,

            "mean_improvement_vs_ridge_f4_pct":
                mean_improvement_vs_ridge,

            "n_seeds_beating_ridge_f4":
                n_beating_ridge,

            "fraction_seeds_beating_ridge_f4":
                fraction_beating_ridge,

            "mean_rmse_ci95_low":
                ci95_low,

            "mean_rmse_ci95_high":
                ci95_high,
        }
    ]
)


summary_df.to_csv(
    SEED_SUMMARY_FILE,
    index=False,
)


# =============================================================================
# PRINT INDIVIDUAL RESULTS
# =============================================================================

section(
    "INDIVIDUAL SEED RESULTS"
)


print(
    seed_df[
        [
            "reservoir_seed",
            "actual_connectivity",
            "max_singular_value",
            "sigma_rho_ratio",
            "ridge_alpha",
            "training_cv_rmse",
            "validation_mae",
            "validation_rmse",
            "validation_nrmse",
            "improvement_vs_ridge_f4_pct",
            "beats_ridge_f4",
        ]
    ]
    .sort_values(
        "reservoir_seed"
    )
    .to_string(
        index=False
    )
)


# =============================================================================
# PRINT SUMMARY
# =============================================================================

section(
    "FINAL ROBUSTNESS RESULT"
)


print(
    f"Number of seeds:              "
    f"{len(RESERVOIR_SEEDS)}"
)

print()

print(
    f"Mean validation RMSE:         "
    f"{mean_rmse:.6f}"
)

print(
    f"Std validation RMSE:          "
    f"{std_rmse:.6f}"
)

print(
    f"Median validation RMSE:       "
    f"{median_rmse:.6f}"
)

print(
    f"Minimum validation RMSE:      "
    f"{min_rmse:.6f}"
)

print(
    f"Maximum validation RMSE:      "
    f"{max_rmse:.6f}"
)

print()

print(
    f"Approx. 95% CI mean RMSE:     "
    f"[{ci95_low:.6f}, "
    f"{ci95_high:.6f}]"
)

print()

print(
    f"Ridge F4 validation RMSE:     "
    f"{RIDGE_F4_RMSE:.6f}"
)

print(
    f"Original seed-42 ESN RMSE:    "
    f"{ORIGINAL_SEED42_RMSE:.6f}"
)

print()

print(
    f"Mean improvement vs Ridge F4: "
    f"{mean_improvement_vs_ridge:.3f}%"
)

print()

print(
    f"Seeds beating Ridge F4:       "
    f"{n_beating_ridge}"
    f"/"
    f"{len(RESERVOIR_SEEDS)}"
)

print(
    f"Fraction beating Ridge F4:    "
    f"{100.0 * fraction_beating_ridge:.1f}%"
)


# =============================================================================
# INTERPRETATION FLAGS
# =============================================================================

section(
    "ROBUSTNESS INTERPRETATION"
)


if mean_rmse < RIDGE_F4_RMSE:

    print(
        "Mean ESN RMSE is LOWER than Ridge F4."
    )

else:

    print(
        "Mean ESN RMSE is HIGHER than Ridge F4."
    )


if (
    ci95_high
    < RIDGE_F4_RMSE
):

    print(
        "The complete approximate 95% CI of "
        "the mean lies below Ridge F4."
    )

    print(
        "This would provide stronger evidence "
        "of a robust ESN advantage."
    )


elif (
    ci95_low
    > RIDGE_F4_RMSE
):

    print(
        "The complete approximate 95% CI of "
        "the mean lies above Ridge F4."
    )

    print(
        "The selected ESN architecture appears "
        "systematically worse than Ridge F4."
    )


else:

    print(
        "The approximate 95% CI overlaps the "
        "Ridge F4 RMSE."
    )

    print(
        "Therefore the current evidence does NOT "
        "support a clear ESN-vs-Ridge superiority claim."
    )


# =============================================================================
# DATABASE AUDIT
# =============================================================================

section(
    "DATABASE AUDIT"
)


with engine.connect() as connection:


    db_df = pd.read_sql(
        text("""
            SELECT
                reservoir_seed,
                actual_connectivity,
                spectral_radius,
                max_singular_value,
                sigma_rho_ratio,
                ridge_alpha,
                training_cv_rmse,
                validation_rmse,
                improvement_vs_ridge_f4_pct,
                beats_ridge_f4
            FROM
                qrc_esn_seed_robustness
            ORDER BY
                reservoir_seed
        """),
        connection,
    )


print(
    db_df.to_string(
        index=False
    )
)


print()

print(
    f"Expected database rows: "
    f"{len(RESERVOIR_SEEDS)}"
)

print(
    f"Stored database rows:   "
    f"{len(db_df)}"
)


if len(
    db_df
) != len(
    RESERVOIR_SEEDS
):

    raise RuntimeError(
        "Database seed-count audit failed."
    )


# =============================================================================
# FINAL STATUS
# =============================================================================

section(
    "STEP 4.2b RESULT"
)


print(
    "ESN seed-robustness experiment completed."
)

print()

print(
    "Methodological checks:"
)

print(
    "  PASS - ESN architecture fixed before seed experiment"
)

print(
    "  PASS - 10 predefined random reservoir seeds used"
)

print(
    "  PASS - seeds were not selected by validation performance"
)

print(
    "  PASS - spectral radius fixed at rho=0.9"
)

print(
    "  PASS - connectivity fixed at p=0.10"
)

print(
    "  PASS - input scaling fixed at s=0.25"
)

print(
    "  PASS - leak rate fixed at alpha=1.0"
)

print(
    "  PASS - reservoir size fixed at 100"
)

print(
    "  PASS - Ridge alpha selected separately for each seed"
)

print(
    "  PASS - Ridge alpha selection used training-only chronological CV"
)

print(
    "  PASS - validation used only for seed robustness measurement"
)

print(
    "  PASS - no best seed selected"
)

print(
    "  PASS - mean/std/median/range reported"
)

print(
    "  PASS - test arrays were NOT loaded"
)

print(
    "  PASS - results stored in MariaDB"
)

print()

print(
    "Saved:"
)

print(
    f"  {SEED_RESULTS_FILE}"
)

print(
    f"  {SEED_PREDICTIONS_FILE}"
)

print(
    f"  {SEED_SUMMARY_FILE}"
)

print()

print(
    "Database table:"
)

print(
    "  qrc_esn_seed_robustness"
)

print()

print(
    "Test set remains untouched."
)

print()

print(
    "Week 4 Step 4.2b finished."
)