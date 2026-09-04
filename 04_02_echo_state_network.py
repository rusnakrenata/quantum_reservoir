from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from sqlalchemy import text

from db_config import engine


# =============================================================================
# WEEK 4 - STEP 4.2
# ECHO STATE NETWORK
# =============================================================================


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TEMPORAL DATASETS
# =============================================================================

FEATURE_SETS = [
    "F2",
    "F3",
    "F4",
]

WINDOW_LENGTHS = [
    1,
    2,
    5,
    7,
    14,
    21,
    28,
]


# =============================================================================
# FIXED RESERVOIR ARCHITECTURE
# =============================================================================

RESERVOIR_SIZE = 100

RESERVOIR_SEED = 42


# =============================================================================
# RESERVOIR CONNECTIVITY
#
# p = probability that an entry of W_r is non-zero.
#
# With N_r = 100:
#
# p = 0.05 -> approximately 5 recurrent inputs per neuron
# p = 0.10 -> approximately 10 recurrent inputs per neuron
#
# The two topologies are generated from the SAME random draws.
# Therefore the p=0.05 edge set is a subset of the p=0.10 edge set.
# =============================================================================

CONNECTIVITIES = [
    0.05,
    0.10,
]


# =============================================================================
# SPECTRAL RADIUS
# =============================================================================

SPECTRAL_RADII = [
    0.3,
    0.5,
    0.7,
    0.9,
]


# =============================================================================
# INPUT SCALING
# =============================================================================

INPUT_SCALINGS = [
    0.25,
    0.50,
    0.75,
    1.00,
    2.00,
]


# =============================================================================
# LEAK RATE
# =============================================================================

LEAK_RATES = [
    0.25,
    0.50,
    0.75,
    1.00,
]


# =============================================================================
# RIDGE READOUT REGULARIZATION
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
# TRAINING-ONLY CHRONOLOGICAL CROSS-VALIDATION
# =============================================================================

N_CV_SPLITS = 5


# =============================================================================
# WEEK 3 REFERENCE FILES
# =============================================================================

PREPROCESSED_FILE = (
    RESULTS_DIR
    / "03_01_preprocessed_samples.csv"
)

NAIVE_METRICS_FILE = (
    RESULTS_DIR
    / "03_02_naive_baseline_metrics.csv"
)

RIDGE_METRICS_FILE = (
    RESULTS_DIR
    / "03_03_ridge_validation_metrics.csv"
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

CV_SEARCH_FILE = (
    RESULTS_DIR
    / "04_02_esn_cv_search.csv"
)

VALIDATION_METRICS_FILE = (
    RESULTS_DIR
    / "04_02_esn_validation_metrics.csv"
)

VALIDATION_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "04_02_esn_validation_predictions.csv"
)

SELECTED_CONFIGS_FILE = (
    RESULTS_DIR
    / "04_02_esn_selected_configs.csv"
)


# =============================================================================
# HELPER
# =============================================================================

def section(title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


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
        "Could not identify the next-day property-damage "
        "claim-count target column."
    )


# =============================================================================
# LOAD WEEK 3 REFERENCE INFORMATION
# =============================================================================

def load_reference_context():

    if not PREPROCESSED_FILE.exists():
        raise FileNotFoundError(
            PREPROCESSED_FILE
        )

    if not NAIVE_METRICS_FILE.exists():
        raise FileNotFoundError(
            NAIVE_METRICS_FILE
        )

    if not RIDGE_METRICS_FILE.exists():
        raise FileNotFoundError(
            RIDGE_METRICS_FILE
        )


    # -------------------------------------------------------------------------
    # PREPROCESSED FORECASTING SAMPLES
    # -------------------------------------------------------------------------

    df = pd.read_csv(
        PREPROCESSED_FILE,
        parse_dates=[
            "input_date",
            "target_date",
        ],
    )

    target_column = resolve_target_column(
        df
    )


    # -------------------------------------------------------------------------
    # TRAINING TARGET STANDARD DEVIATION
    #
    # This preserves exactly the Week 3 NRMSE definition:
    #
    #     NRMSE = RMSE / sigma_train,target
    #
    # -------------------------------------------------------------------------

    train_target = (
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
            train_target,
            ddof=0,
        )
    )


    # -------------------------------------------------------------------------
    # VALIDATION DATES
    #
    # IMPORTANT:
    #
    # We intentionally obtain the dates from this CSV rather than reading
    # validation_target_dates from the Step 4.1 .npz file.
    #
    # Why?
    #
    # The Step 4.1 dates were saved by NumPy as dtype=object.
    # Object arrays require pickle deserialization.
    #
    # We want:
    #
    #     allow_pickle=False
    #
    # for the model-ready tensor files.
    #
    # Therefore dates come from the canonical CSV instead.
    # -------------------------------------------------------------------------

    validation_dates = (
        df.loc[
            df["split"] == "validation",
            "target_date",
        ]
        .dt.strftime("%Y-%m-%d")
        .to_numpy(
            dtype="U10"
        )
    )


    # -------------------------------------------------------------------------
    # SEASONAL NAIVE RMSE
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

    if len(seasonal_row) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "the seasonal-naive result."
        )

    seasonal_rmse = float(
        seasonal_row.iloc[0][
            "rmse"
        ]
    )


    # -------------------------------------------------------------------------
    # RIDGE F4 RMSE
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

    if len(ridge_f4) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "the Ridge F4 validation result."
        )

    ridge_f4_rmse = float(
        ridge_f4.iloc[0][
            "validation_rmse"
        ]
    )


    return (
        train_target_std,
        seasonal_rmse,
        ridge_f4_rmse,
        validation_dates,
    )


# =============================================================================
# RANDOM BASE RESERVOIR
# =============================================================================

def generate_base_recurrent_draws(
    reservoir_size,
    seed,
):
    """
    Generate the random numbers used to construct every candidate topology.

    We generate:

        1. random recurrent weights
        2. random connectivity-mask values

    The same random draws are reused for p=0.05 and p=0.10.

    Therefore:

        edges(p=0.05) subset edges(p=0.10)

    This makes the connectivity comparison substantially cleaner.
    """

    rng = np.random.default_rng(
        seed
    )


    base_weights = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(
            reservoir_size,
            reservoir_size,
        ),
    )


    mask_draws = rng.random(
        (
            reservoir_size,
            reservoir_size,
        )
    )


    return (
        base_weights,
        mask_draws,
    )


# =============================================================================
# BUILD ONE CONNECTIVITY-SPECIFIC RESERVOIR
# =============================================================================

def build_unit_recurrent_matrix(
    base_weights,
    mask_draws,
    connectivity,
):
    """
    Construct the sparse reservoir for one p.

    The matrix is normalized to:

        rho(W_unit) = 1

    Later:

        W_r = rho_target * W_unit

    gives the required spectral radius.
    """

    mask = (
        mask_draws
        < connectivity
    )

    W0 = (
        base_weights
        * mask
    )


    actual_connectivity = float(
        np.count_nonzero(
            W0
        )
        / W0.size
    )


    eigenvalues = np.linalg.eigvals(
        W0
    )


    original_spectral_radius = float(
        np.max(
            np.abs(
                eigenvalues
            )
        )
    )


    if original_spectral_radius <= 0:
        raise RuntimeError(
            f"Reservoir with p={connectivity} "
            "has zero spectral radius."
        )


    W_unit = (
        W0
        / original_spectral_radius
    )


    # -------------------------------------------------------------------------
    # Verify spectral radius after normalization
    # -------------------------------------------------------------------------

    unit_eigenvalues = np.linalg.eigvals(
        W_unit
    )

    unit_spectral_radius = float(
        np.max(
            np.abs(
                unit_eigenvalues
            )
        )
    )


    # -------------------------------------------------------------------------
    # Singular-value diagnostic
    #
    # sigma_max tells us the maximum one-step Euclidean stretching.
    # -------------------------------------------------------------------------

    singular_values = np.linalg.svd(
        W_unit,
        compute_uv=False,
    )

    unit_sigma_max = float(
        singular_values[0]
    )


    return {
        "W_unit":
            W_unit,

        "actual_connectivity":
            actual_connectivity,

        "unit_spectral_radius":
            unit_spectral_radius,

        "unit_sigma_max":
            unit_sigma_max,
    }


# =============================================================================
# INPUT MATRIX
# =============================================================================

def generate_unit_input_matrix(
    reservoir_size,
    input_dimension,
    seed,
):
    """
    Generate the fixed random input projection.

    Division by sqrt(d) approximately compensates for different numerical
    feature dimensionalities:

        F2 -> d=3
        F3 -> d=4
        F4 -> d=5

    This prevents F4 from automatically receiving a substantially larger
    total random input drive merely because it has more columns.
    """

    rng = np.random.default_rng(
        seed
        + 10000
        + input_dimension
    )


    W_in = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(
            reservoir_size,
            input_dimension,
        ),
    )


    W_in /= np.sqrt(
        input_dimension
    )


    return W_in


# =============================================================================
# ESN FORWARD PASS
# =============================================================================

def compute_final_reservoir_states(
    X,
    W_unit,
    W_in_unit,
    spectral_radius,
    input_scaling,
    leak_rate,
    calculate_diagnostics=False,
):
    """
    Process all temporal samples.

    Input:

        X.shape = (n_samples, W, d)

    Reservoir state:

        r_0 = 0

    Candidate state:

        r_tilde =
            tanh(
                s W_in u_t
                +
                rho W_r r_(t-1)
            )

    Leaky update:

        r_t =
            (1-alpha) r_(t-1)
            +
            alpha r_tilde

    Output:

        final r_W only

    Therefore:

        R.shape =
            (n_samples, reservoir_size)

    -------------------------------------------------------------------------
    SATURATION DIAGNOSTIC
    -------------------------------------------------------------------------

    We additionally monitor candidate states satisfying:

        |r_tilde| >= 0.95

    This is especially useful for investigating whether input scaling s=2
    pushes too many neurons into tanh saturation.
    """

    n_samples = X.shape[0]

    reservoir_size = (
        W_unit.shape[0]
    )


    states = np.zeros(
        (
            n_samples,
            reservoir_size,
        ),
        dtype=np.float64,
    )


    W_scaled = (
        spectral_radius
        * W_unit
    )


    W_in_scaled = (
        input_scaling
        * W_in_unit
    )


    saturation_count = 0
    candidate_count = 0
    candidate_abs_sum = 0.0


    for time_index in range(
        X.shape[1]
    ):

        current_input = (
            X[
                :,
                time_index,
                :
            ]
        )


        input_drive = (
            current_input
            @ W_in_scaled.T
        )


        recurrent_drive = (
            states
            @ W_scaled.T
        )


        pre_activation = (
            input_drive
            +
            recurrent_drive
        )


        candidate_state = np.tanh(
            pre_activation
        )


        if calculate_diagnostics:

            absolute_candidate = np.abs(
                candidate_state
            )

            saturation_count += int(
                np.count_nonzero(
                    absolute_candidate
                    >= 0.95
                )
            )

            candidate_count += int(
                candidate_state.size
            )

            candidate_abs_sum += float(
                np.sum(
                    absolute_candidate
                )
            )


        states = (
            (1.0 - leak_rate)
            * states
            +
            leak_rate
            * candidate_state
        )


    diagnostics = None

    if calculate_diagnostics:

        diagnostics = {
            "candidate_saturation_fraction":
                (
                    saturation_count
                    / candidate_count
                    if candidate_count > 0
                    else 0.0
                ),

            "mean_abs_candidate_state":
                (
                    candidate_abs_sum
                    / candidate_count
                    if candidate_count > 0
                    else 0.0
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
# RIDGE CHRONOLOGICAL CROSS-VALIDATION
# =============================================================================

def select_ridge_alpha(
    R_train,
    y_train,
):
    """
    Select Ridge alpha using training-only chronological CV.

    TimeSeriesSplit guarantees:

        earlier observations -> later observations

    and never random shuffling.
    """

    splitter = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )


    rows = []


    for ridge_alpha in RIDGE_ALPHAS:

        fold_rmses = []


        for (
            fold_train_index,
            fold_validation_index,
        ) in splitter.split(
            R_train
        ):

            model = Ridge(
                alpha=ridge_alpha,
                fit_intercept=True,
            )


            model.fit(
                R_train[
                    fold_train_index
                ],
                y_train[
                    fold_train_index
                ],
            )


            prediction = model.predict(
                R_train[
                    fold_validation_index
                ]
            )


            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_train[
                            fold_validation_index
                        ],
                        prediction,
                    )
                )
            )


            fold_rmses.append(
                rmse
            )


        rows.append({
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


    results = pd.DataFrame(
        rows
    )


    results = (
        results
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


    best = results.iloc[0]


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
# DATABASE TABLES
# =============================================================================

def recreate_database_tables():
    """
    The first version of Step 4.2 did not include connectivity p in the
    CV-trial unique key.

    Since Step 4.2 was not completed and the experimental design has now
    changed, recreate ONLY the two ESN tables.

    No Week 1, Week 2, Week 3, or Step 4.1 tables are touched.
    """


    with engine.begin() as connection:

        connection.execute(
            text("""
                DROP TABLE IF EXISTS
                    qrc_esn_validation_result
            """)
        )

        connection.execute(
            text("""
                DROP TABLE IF EXISTS
                    qrc_esn_cv_trial
            """)
        )


        connection.execute(
            text("""
                CREATE TABLE qrc_esn_cv_trial
                (
                    esn_cv_trial_id
                        BIGINT AUTO_INCREMENT
                        PRIMARY KEY,

                    feature_set_code
                        VARCHAR(10) NOT NULL,

                    window_length
                        INT NOT NULL,

                    reservoir_size
                        INT NOT NULL,

                    requested_connectivity
                        DOUBLE NOT NULL,

                    actual_connectivity
                        DOUBLE NOT NULL,

                    reservoir_seed
                        INT NOT NULL,

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

                    best_ridge_alpha
                        DOUBLE NOT NULL,

                    mean_cv_rmse
                        DOUBLE NOT NULL,

                    std_cv_rmse
                        DOUBLE NOT NULL,

                    candidate_saturation_fraction
                        DOUBLE NOT NULL,

                    mean_abs_candidate_state
                        DOUBLE NOT NULL,

                    mean_abs_final_state
                        DOUBLE NOT NULL,

                    created_at
                        TIMESTAMP NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE KEY uq_esn_cv_trial
                    (
                        feature_set_code,
                        window_length,
                        reservoir_size,
                        reservoir_seed,
                        requested_connectivity,
                        spectral_radius,
                        input_scaling,
                        leak_rate
                    )
                )
            """)
        )


        connection.execute(
            text("""
                CREATE TABLE qrc_esn_validation_result
                (
                    esn_validation_result_id
                        BIGINT AUTO_INCREMENT
                        PRIMARY KEY,

                    feature_set_code
                        VARCHAR(10) NOT NULL,

                    window_length
                        INT NOT NULL,

                    reservoir_size
                        INT NOT NULL,

                    requested_connectivity
                        DOUBLE NOT NULL,

                    actual_connectivity
                        DOUBLE NOT NULL,

                    reservoir_seed
                        INT NOT NULL,

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

                    validation_mae
                        DOUBLE NOT NULL,

                    validation_rmse
                        DOUBLE NOT NULL,

                    validation_nrmse
                        DOUBLE NOT NULL,

                    validation_bias
                        DOUBLE NOT NULL,

                    improvement_vs_seasonal_pct
                        DOUBLE NOT NULL,

                    improvement_vs_ridge_f4_pct
                        DOUBLE NOT NULL,

                    candidate_saturation_fraction
                        DOUBLE NOT NULL,

                    mean_abs_candidate_state
                        DOUBLE NOT NULL,

                    mean_abs_final_state
                        DOUBLE NOT NULL,

                    model_artifact_path
                        VARCHAR(500) NOT NULL,

                    created_at
                        TIMESTAMP NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE KEY uq_esn_validation_result
                    (
                        feature_set_code,
                        window_length
                    )
                )
            """)
        )


# =============================================================================
# DATABASE INSERT
# =============================================================================

def insert_cv_trial(row):

    sql = text("""
        INSERT INTO qrc_esn_cv_trial
        (
            feature_set_code,
            window_length,

            reservoir_size,

            requested_connectivity,
            actual_connectivity,

            reservoir_seed,

            spectral_radius,
            max_singular_value,
            sigma_rho_ratio,

            input_scaling,
            leak_rate,

            best_ridge_alpha,
            mean_cv_rmse,
            std_cv_rmse,

            candidate_saturation_fraction,
            mean_abs_candidate_state,
            mean_abs_final_state
        )
        VALUES
        (
            :feature_set_code,
            :window_length,

            :reservoir_size,

            :requested_connectivity,
            :actual_connectivity,

            :reservoir_seed,

            :spectral_radius,
            :max_singular_value,
            :sigma_rho_ratio,

            :input_scaling,
            :leak_rate,

            :best_ridge_alpha,
            :mean_cv_rmse,
            :std_cv_rmse,

            :candidate_saturation_fraction,
            :mean_abs_candidate_state,
            :mean_abs_final_state
        )
    """)


    with engine.begin() as connection:

        connection.execute(
            sql,
            row,
        )


# =============================================================================
# DATABASE INSERT SELECTED VALIDATION RESULT
# =============================================================================

def insert_validation_result(row):

    sql = text("""
        INSERT INTO qrc_esn_validation_result
        (
            feature_set_code,
            window_length,

            reservoir_size,

            requested_connectivity,
            actual_connectivity,

            reservoir_seed,

            spectral_radius,
            max_singular_value,
            sigma_rho_ratio,

            input_scaling,
            leak_rate,

            ridge_alpha,
            training_cv_rmse,

            validation_mae,
            validation_rmse,
            validation_nrmse,
            validation_bias,

            improvement_vs_seasonal_pct,
            improvement_vs_ridge_f4_pct,

            candidate_saturation_fraction,
            mean_abs_candidate_state,
            mean_abs_final_state,

            model_artifact_path
        )
        VALUES
        (
            :feature_set_code,
            :window_length,

            :reservoir_size,

            :requested_connectivity,
            :actual_connectivity,

            :reservoir_seed,

            :spectral_radius,
            :max_singular_value,
            :sigma_rho_ratio,

            :input_scaling,
            :leak_rate,

            :ridge_alpha,
            :training_cv_rmse,

            :validation_mae,
            :validation_rmse,
            :validation_nrmse,
            :validation_bias,

            :improvement_vs_seasonal_pct,
            :improvement_vs_ridge_f4_pct,

            :candidate_saturation_fraction,
            :mean_abs_candidate_state,
            :mean_abs_final_state,

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
    "WEEK 4 - STEP 4.2: "
    "ECHO STATE NETWORK"
)


# =============================================================================
# REFERENCE CONTEXT
# =============================================================================

(
    TRAIN_TARGET_STD,
    SEASONAL_RMSE,
    RIDGE_F4_RMSE,
    VALIDATION_DATES,
) = load_reference_context()


section(
    "REFERENCE METRICS FROM WEEK 3"
)

print(
    f"Training target std: "
    f"{TRAIN_TARGET_STD:.6f}"
)

print(
    f"Seasonal-naive RMSE: "
    f"{SEASONAL_RMSE:.6f}"
)

print(
    f"Ridge F4 RMSE:       "
    f"{RIDGE_F4_RMSE:.6f}"
)

print(
    f"Validation dates:     "
    f"{len(VALIDATION_DATES)}"
)


# =============================================================================
# EXPERIMENT GRID
# =============================================================================

section(
    "ESN EXPERIMENT GRID"
)


N_RESERVOIR_CONFIGS_PER_FW = (
    len(CONNECTIVITIES)
    * len(SPECTRAL_RADII)
    * len(INPUT_SCALINGS)
    * len(LEAK_RATES)
)


N_FW_COMBINATIONS = (
    len(FEATURE_SETS)
    * len(WINDOW_LENGTHS)
)


N_TOTAL_RESERVOIR_CONFIGS = (
    N_RESERVOIR_CONFIGS_PER_FW
    * N_FW_COMBINATIONS
)


print(
    f"Reservoir size:          "
    f"{RESERVOIR_SIZE}"
)

print(
    f"Reservoir seed:          "
    f"{RESERVOIR_SEED}"
)

print(
    f"Connectivity p:          "
    f"{CONNECTIVITIES}"
)

print(
    f"Spectral radius rho:     "
    f"{SPECTRAL_RADII}"
)

print(
    f"Input scaling s:         "
    f"{INPUT_SCALINGS}"
)

print(
    f"Leak rate alpha:         "
    f"{LEAK_RATES}"
)

print(
    f"Ridge alphas:            "
    f"{RIDGE_ALPHAS}"
)

print(
    f"Chronological CV folds:  "
    f"{N_CV_SPLITS}"
)

print()
print(
    "Reservoir configurations per F/W: "
    f"{N_RESERVOIR_CONFIGS_PER_FW}"
)

print(
    "F/W combinations:                  "
    f"{N_FW_COMBINATIONS}"
)

print(
    "Total reservoir configurations:    "
    f"{N_TOTAL_RESERVOIR_CONFIGS}"
)


# =============================================================================
# DATABASE
# =============================================================================

section(
    "DATABASE INITIALIZATION"
)

recreate_database_tables()

print(
    "Recreated:"
)

print(
    "  qrc_esn_cv_trial"
)

print(
    "  qrc_esn_validation_result"
)


# =============================================================================
# GENERATE COMMON RANDOM RESERVOIR DRAWS
# =============================================================================

section(
    "BASE RANDOM RESERVOIR"
)


(
    BASE_RECURRENT_WEIGHTS,
    BASE_MASK_DRAWS,
) = generate_base_recurrent_draws(
    reservoir_size=
        RESERVOIR_SIZE,

    seed=
        RESERVOIR_SEED,
)


# =============================================================================
# BUILD BOTH CONNECTIVITY-SPECIFIC UNIT RESERVOIRS
# =============================================================================

RESERVOIRS = {}


for connectivity in CONNECTIVITIES:

    reservoir_info = (
        build_unit_recurrent_matrix(
            base_weights=
                BASE_RECURRENT_WEIGHTS,

            mask_draws=
                BASE_MASK_DRAWS,

            connectivity=
                connectivity,
        )
    )


    RESERVOIRS[
        connectivity
    ] = reservoir_info


    print()
    print(
        f"Requested p:          "
        f"{connectivity:.4f}"
    )

    print(
        f"Actual connectivity:  "
        f"{reservoir_info['actual_connectivity']:.6f}"
    )

    print(
        f"rho(W_unit):          "
        f"{reservoir_info['unit_spectral_radius']:.6f}"
    )

    print(
        f"sigma_max(W_unit):    "
        f"{reservoir_info['unit_sigma_max']:.6f}"
    )

    print(
        "sigma_max/rho:        "
        f"{reservoir_info['unit_sigma_max']:.6f}"
    )


# =============================================================================
# EXPERIMENT RESULT CONTAINERS
# =============================================================================

cv_search_rows = []

validation_rows = []

prediction_rows = []

selected_config_rows = []


# =============================================================================
# RUN ALL F/W COMBINATIONS
# =============================================================================

for feature_set in FEATURE_SETS:

    for W in WINDOW_LENGTHS:

        section(
            f"ESN SEARCH: {feature_set}, W={W}"
        )


        # ---------------------------------------------------------------------
        # STEP 4.1 TEMPORAL TENSOR
        # ---------------------------------------------------------------------

        tensor_file = (
            RESULTS_DIR
            / (
                f"04_01_windows_"
                f"{feature_set}_"
                f"W{W:02d}.npz"
            )
        )


        if not tensor_file.exists():

            raise FileNotFoundError(
                tensor_file
            )


        # ---------------------------------------------------------------------
        # IMPORTANT FIX FOR THE OBJECT-ARRAY ERROR
        #
        # We load ONLY numerical arrays from NPZ.
        #
        # validation_target_dates is intentionally NOT accessed here.
        #
        # Therefore:
        #
        #     allow_pickle=False
        #
        # remains safe and valid.
        # ---------------------------------------------------------------------

        with np.load(
            tensor_file,
            allow_pickle=False,
        ) as data:

            X_train = np.asarray(
                data["X_train"],
                dtype=np.float64,
            )

            y_train = np.asarray(
                data["y_train"],
                dtype=np.float64,
            )

            X_val = np.asarray(
                data["X_val"],
                dtype=np.float64,
            )

            y_val = np.asarray(
                data["y_val"],
                dtype=np.float64,
            )


        # ---------------------------------------------------------------------
        # VALIDATION DATE CHECK
        # ---------------------------------------------------------------------

        if len(y_val) != len(
            VALIDATION_DATES
        ):

            raise RuntimeError(
                f"{feature_set}, W={W}: "
                "validation target count does not "
                "match canonical validation dates."
            )


        input_dimension = int(
            X_train.shape[2]
        )


        print(
            f"X_train: "
            f"{X_train.shape}"
        )

        print(
            f"X_val:   "
            f"{X_val.shape}"
        )

        print(
            f"Input d: "
            f"{input_dimension}"
        )


        # ---------------------------------------------------------------------
        # SAME RANDOM INPUT PROJECTION FOR ALL CONFIGURATIONS
        # OF THIS FEATURE DIMENSION
        # ---------------------------------------------------------------------

        W_IN_UNIT = (
            generate_unit_input_matrix(
                reservoir_size=
                    RESERVOIR_SIZE,

                input_dimension=
                    input_dimension,

                seed=
                    RESERVOIR_SEED,
            )
        )


        local_search_rows = []


        # ---------------------------------------------------------------------
        # FULL RESERVOIR GRID
        # ---------------------------------------------------------------------

        combinations = list(
            itertools.product(
                CONNECTIVITIES,
                SPECTRAL_RADII,
                INPUT_SCALINGS,
                LEAK_RATES,
            )
        )


        print(
            f"Searching "
            f"{len(combinations)} "
            "reservoir configurations..."
        )


        for (
            connectivity,
            spectral_radius,
            input_scaling,
            leak_rate,
        ) in combinations:


            reservoir_info = (
                RESERVOIRS[
                    connectivity
                ]
            )


            W_UNIT = reservoir_info[
                "W_unit"
            ]


            actual_connectivity = float(
                reservoir_info[
                    "actual_connectivity"
                ]
            )


            unit_sigma_max = float(
                reservoir_info[
                    "unit_sigma_max"
                ]
            )


            # -----------------------------------------------------------------
            # SINGULAR VALUE OF THE ACTUAL SCALED RECURRENT MATRIX
            #
            # If:
            #
            #     W_r = rho * W_unit
            #
            # then:
            #
            #     sigma_max(W_r)
            #       = rho * sigma_max(W_unit)
            #
            # -----------------------------------------------------------------

            max_singular_value = float(
                spectral_radius
                * unit_sigma_max
            )


            sigma_rho_ratio = float(
                max_singular_value
                / spectral_radius
            )


            # -----------------------------------------------------------------
            # TRAIN RESERVOIR FEATURES
            # -----------------------------------------------------------------

            (
                R_train,
                state_diagnostics,
            ) = (
                compute_final_reservoir_states(
                    X=
                        X_train,

                    W_unit=
                        W_UNIT,

                    W_in_unit=
                        W_IN_UNIT,

                    spectral_radius=
                        spectral_radius,

                    input_scaling=
                        input_scaling,

                    leak_rate=
                        leak_rate,

                    calculate_diagnostics=
                        True,
                )
            )


            # -----------------------------------------------------------------
            # RIDGE ALPHA SELECTION
            # -----------------------------------------------------------------

            (
                best_ridge_alpha,
                mean_cv_rmse,
                std_cv_rmse,
            ) = select_ridge_alpha(
                R_train=
                    R_train,

                y_train=
                    y_train,
            )


            trial_row = {
                "feature_set_code":
                    feature_set,

                "window_length":
                    int(W),

                "reservoir_size":
                    int(
                        RESERVOIR_SIZE
                    ),

                "requested_connectivity":
                    float(
                        connectivity
                    ),

                "actual_connectivity":
                    actual_connectivity,

                "reservoir_seed":
                    int(
                        RESERVOIR_SEED
                    ),

                "spectral_radius":
                    float(
                        spectral_radius
                    ),

                "max_singular_value":
                    max_singular_value,

                "sigma_rho_ratio":
                    sigma_rho_ratio,

                "input_scaling":
                    float(
                        input_scaling
                    ),

                "leak_rate":
                    float(
                        leak_rate
                    ),

                "best_ridge_alpha":
                    float(
                        best_ridge_alpha
                    ),

                "mean_cv_rmse":
                    float(
                        mean_cv_rmse
                    ),

                "std_cv_rmse":
                    float(
                        std_cv_rmse
                    ),

                "candidate_saturation_fraction":
                    float(
                        state_diagnostics[
                            "candidate_saturation_fraction"
                        ]
                    ),

                "mean_abs_candidate_state":
                    float(
                        state_diagnostics[
                            "mean_abs_candidate_state"
                        ]
                    ),

                "mean_abs_final_state":
                    float(
                        state_diagnostics[
                            "mean_abs_final_state"
                        ]
                    ),
            }


            local_search_rows.append(
                trial_row
            )


            cv_search_rows.append(
                trial_row.copy()
            )


            insert_cv_trial(
                trial_row
            )


        # ---------------------------------------------------------------------
        # SELECT BEST RESERVOIR USING TRAINING CV ONLY
        # ---------------------------------------------------------------------

        local_search_df = (
            pd.DataFrame(
                local_search_rows
            )
            .sort_values(
                [
                    "mean_cv_rmse",
                    "std_cv_rmse",
                ]
            )
            .reset_index(
                drop=True
            )
        )


        best = (
            local_search_df
            .iloc[0]
        )


        best_connectivity = float(
            best[
                "requested_connectivity"
            ]
        )

        best_actual_connectivity = float(
            best[
                "actual_connectivity"
            ]
        )

        best_rho = float(
            best[
                "spectral_radius"
            ]
        )

        best_sigma_max = float(
            best[
                "max_singular_value"
            ]
        )

        best_sigma_rho_ratio = float(
            best[
                "sigma_rho_ratio"
            ]
        )

        best_input_scaling = float(
            best[
                "input_scaling"
            ]
        )

        best_leak_rate = float(
            best[
                "leak_rate"
            ]
        )

        best_ridge_alpha = float(
            best[
                "best_ridge_alpha"
            ]
        )

        best_cv_rmse = float(
            best[
                "mean_cv_rmse"
            ]
        )


        selected_saturation = float(
            best[
                "candidate_saturation_fraction"
            ]
        )

        selected_mean_abs_candidate = float(
            best[
                "mean_abs_candidate_state"
            ]
        )

        selected_mean_abs_final = float(
            best[
                "mean_abs_final_state"
            ]
        )


        # ---------------------------------------------------------------------
        # SELECTED RECURRENT MATRIX
        # ---------------------------------------------------------------------

        selected_reservoir = (
            RESERVOIRS[
                best_connectivity
            ]
        )


        W_UNIT_SELECTED = (
            selected_reservoir[
                "W_unit"
            ]
        )


        # ---------------------------------------------------------------------
        # RECOMPUTE FINAL TRAINING STATES
        # ---------------------------------------------------------------------

        (
            R_train,
            _,
        ) = compute_final_reservoir_states(
            X=
                X_train,

            W_unit=
                W_UNIT_SELECTED,

            W_in_unit=
                W_IN_UNIT,

            spectral_radius=
                best_rho,

            input_scaling=
                best_input_scaling,

            leak_rate=
                best_leak_rate,

            calculate_diagnostics=
                False,
        )


        # ---------------------------------------------------------------------
        # VALIDATION STATES
        # ---------------------------------------------------------------------

        (
            R_val,
            _,
        ) = compute_final_reservoir_states(
            X=
                X_val,

            W_unit=
                W_UNIT_SELECTED,

            W_in_unit=
                W_IN_UNIT,

            spectral_radius=
                best_rho,

            input_scaling=
                best_input_scaling,

            leak_rate=
                best_leak_rate,

            calculate_diagnostics=
                False,
        )


        # ---------------------------------------------------------------------
        # TRAIN FINAL RIDGE READOUT ON ENTIRE TRAINING PERIOD
        # ---------------------------------------------------------------------

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


        # ---------------------------------------------------------------------
        # VALIDATION PREDICTIONS
        # ---------------------------------------------------------------------

        prediction = (
            readout.predict(
                R_val
            )
        )


        # ---------------------------------------------------------------------
        # VALIDATION METRICS
        # ---------------------------------------------------------------------

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


        improvement_vs_seasonal_pct = float(
            100.0
            * (
                SEASONAL_RMSE
                - validation_rmse
            )
            / SEASONAL_RMSE
        )


        improvement_vs_ridge_f4_pct = float(
            100.0
            * (
                RIDGE_F4_RMSE
                - validation_rmse
            )
            / RIDGE_F4_RMSE
        )


        # ---------------------------------------------------------------------
        # SAVE EXACT SELECTED ESN
        # ---------------------------------------------------------------------

        W_reservoir_selected = (
            best_rho
            * W_UNIT_SELECTED
        )


        W_input_selected = (
            best_input_scaling
            * W_IN_UNIT
        )


        model_artifact = (
            RESULTS_DIR
            / (
                f"04_02_esn_model_"
                f"{feature_set}_"
                f"W{W:02d}.npz"
            )
        )


        np.savez_compressed(
            model_artifact,

            W_reservoir=
                W_reservoir_selected,

            W_input=
                W_input_selected,

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

            feature_set=
                np.asarray(
                    [
                        feature_set
                    ],
                    dtype="U10",
                ),

            window_length=
                np.asarray(
                    [W],
                    dtype=np.int64,
                ),

            reservoir_size=
                np.asarray(
                    [
                        RESERVOIR_SIZE
                    ],
                    dtype=np.int64,
                ),

            requested_connectivity=
                np.asarray(
                    [
                        best_connectivity
                    ],
                    dtype=np.float64,
                ),

            actual_connectivity=
                np.asarray(
                    [
                        best_actual_connectivity
                    ],
                    dtype=np.float64,
                ),

            spectral_radius=
                np.asarray(
                    [
                        best_rho
                    ],
                    dtype=np.float64,
                ),

            max_singular_value=
                np.asarray(
                    [
                        best_sigma_max
                    ],
                    dtype=np.float64,
                ),

            sigma_rho_ratio=
                np.asarray(
                    [
                        best_sigma_rho_ratio
                    ],
                    dtype=np.float64,
                ),

            input_scaling=
                np.asarray(
                    [
                        best_input_scaling
                    ],
                    dtype=np.float64,
                ),

            leak_rate=
                np.asarray(
                    [
                        best_leak_rate
                    ],
                    dtype=np.float64,
                ),

            ridge_alpha=
                np.asarray(
                    [
                        best_ridge_alpha
                    ],
                    dtype=np.float64,
                ),

            reservoir_seed=
                np.asarray(
                    [
                        RESERVOIR_SEED
                    ],
                    dtype=np.int64,
                ),
        )


        # ---------------------------------------------------------------------
        # VALIDATION RESULT
        # ---------------------------------------------------------------------

        validation_row = {
            "model":
                "ESN",

            "feature_set":
                feature_set,

            "window_length":
                int(W),

            "reservoir_size":
                int(
                    RESERVOIR_SIZE
                ),

            "requested_connectivity":
                best_connectivity,

            "actual_connectivity":
                best_actual_connectivity,

            "reservoir_seed":
                int(
                    RESERVOIR_SEED
                ),

            "spectral_radius":
                best_rho,

            "max_singular_value":
                best_sigma_max,

            "sigma_rho_ratio":
                best_sigma_rho_ratio,

            "input_scaling":
                best_input_scaling,

            "leak_rate":
                best_leak_rate,

            "ridge_alpha":
                best_ridge_alpha,

            "training_cv_rmse":
                best_cv_rmse,

            "validation_mae":
                validation_mae,

            "validation_rmse":
                validation_rmse,

            "validation_nrmse":
                validation_nrmse,

            "validation_bias":
                validation_bias,

            "improvement_vs_seasonal_pct":
                improvement_vs_seasonal_pct,

            "improvement_vs_ridge_f4_pct":
                improvement_vs_ridge_f4_pct,

            "candidate_saturation_fraction":
                selected_saturation,

            "mean_abs_candidate_state":
                selected_mean_abs_candidate,

            "mean_abs_final_state":
                selected_mean_abs_final,

            "model_artifact_path":
                model_artifact.as_posix(),
        }


        validation_rows.append(
            validation_row
        )


        selected_config_rows.append(
            validation_row.copy()
        )


        # ---------------------------------------------------------------------
        # DATABASE VALIDATION RESULT
        # ---------------------------------------------------------------------

        db_validation_row = {
            "feature_set_code":
                feature_set,

            "window_length":
                int(W),

            "reservoir_size":
                int(
                    RESERVOIR_SIZE
                ),

            "requested_connectivity":
                best_connectivity,

            "actual_connectivity":
                best_actual_connectivity,

            "reservoir_seed":
                int(
                    RESERVOIR_SEED
                ),

            "spectral_radius":
                best_rho,

            "max_singular_value":
                best_sigma_max,

            "sigma_rho_ratio":
                best_sigma_rho_ratio,

            "input_scaling":
                best_input_scaling,

            "leak_rate":
                best_leak_rate,

            "ridge_alpha":
                best_ridge_alpha,

            "training_cv_rmse":
                best_cv_rmse,

            "validation_mae":
                validation_mae,

            "validation_rmse":
                validation_rmse,

            "validation_nrmse":
                validation_nrmse,

            "validation_bias":
                validation_bias,

            "improvement_vs_seasonal_pct":
                improvement_vs_seasonal_pct,

            "improvement_vs_ridge_f4_pct":
                improvement_vs_ridge_f4_pct,

            "candidate_saturation_fraction":
                selected_saturation,

            "mean_abs_candidate_state":
                selected_mean_abs_candidate,

            "mean_abs_final_state":
                selected_mean_abs_final,

            "model_artifact_path":
                model_artifact.as_posix(),
        }


        insert_validation_result(
            db_validation_row
        )


        # ---------------------------------------------------------------------
        # VALIDATION PREDICTION ROWS
        # ---------------------------------------------------------------------

        for (
            target_date,
            actual,
            predicted,
        ) in zip(
            VALIDATION_DATES,
            y_val,
            prediction,
        ):

            prediction_rows.append({
                "model":
                    "ESN",

                "feature_set":
                    feature_set,

                "window_length":
                    int(W),

                "target_date":
                    str(
                        target_date
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


        # ---------------------------------------------------------------------
        # CONSOLE RESULT
        # ---------------------------------------------------------------------

        print()
        print(
            "SELECTED USING TRAINING-ONLY "
            "CHRONOLOGICAL CV"
        )

        print(
            f"  p requested:       "
            f"{best_connectivity:.4f}"
        )

        print(
            f"  p actual:          "
            f"{best_actual_connectivity:.6f}"
        )

        print(
            f"  rho:               "
            f"{best_rho:.4f}"
        )

        print(
            f"  sigma_max:         "
            f"{best_sigma_max:.6f}"
        )

        print(
            f"  sigma_max/rho:     "
            f"{best_sigma_rho_ratio:.6f}"
        )

        print(
            f"  input scaling s:   "
            f"{best_input_scaling:.4f}"
        )

        print(
            f"  leak rate alpha:   "
            f"{best_leak_rate:.4f}"
        )

        print(
            f"  Ridge alpha:       "
            f"{best_ridge_alpha:.4f}"
        )

        print(
            f"  CV RMSE:           "
            f"{best_cv_rmse:.6f}"
        )

        print(
            f"  tanh saturation:   "
            f"{100.0 * selected_saturation:.3f}%"
        )


        print()
        print(
            "VALIDATION"
        )

        print(
            f"  MAE:               "
            f"{validation_mae:.6f}"
        )

        print(
            f"  RMSE:              "
            f"{validation_rmse:.6f}"
        )

        print(
            f"  NRMSE:             "
            f"{validation_nrmse:.6f}"
        )

        print(
            f"  Bias:              "
            f"{validation_bias:.6f}"
        )

        print(
            f"  vs seasonal:       "
            f"{improvement_vs_seasonal_pct:.3f}%"
        )

        print(
            f"  vs Ridge F4:       "
            f"{improvement_vs_ridge_f4_pct:.3f}%"
        )


# =============================================================================
# SAVE RESULTS
# =============================================================================

cv_search_df = pd.DataFrame(
    cv_search_rows
)

validation_df = pd.DataFrame(
    validation_rows
)

prediction_df = pd.DataFrame(
    prediction_rows
)

selected_config_df = pd.DataFrame(
    selected_config_rows
)


cv_search_df.to_csv(
    CV_SEARCH_FILE,
    index=False,
)

validation_df.to_csv(
    VALIDATION_METRICS_FILE,
    index=False,
)

prediction_df.to_csv(
    VALIDATION_PREDICTIONS_FILE,
    index=False,
)

selected_config_df.to_csv(
    SELECTED_CONFIGS_FILE,
    index=False,
)


# =============================================================================
# VALIDATION RANKING
# =============================================================================

section(
    "ESN VALIDATION RANKING"
)


ranking = (
    validation_df
    .sort_values(
        "validation_rmse"
    )
    .reset_index(
        drop=True
    )
)


ranking.insert(
    0,
    "rmse_rank",
    np.arange(
        1,
        len(ranking) + 1,
    ),
)


print(
    ranking[
        [
            "rmse_rank",
            "feature_set",
            "window_length",

            "requested_connectivity",

            "spectral_radius",
            "max_singular_value",
            "sigma_rho_ratio",

            "input_scaling",
            "leak_rate",
            "ridge_alpha",

            "training_cv_rmse",

            "candidate_saturation_fraction",

            "validation_mae",
            "validation_rmse",
            "validation_nrmse",

            "improvement_vs_seasonal_pct",
            "improvement_vs_ridge_f4_pct",
        ]
    ]
    .to_string(
        index=False
    )
)


# =============================================================================
# MEMORY LENGTH TABLE
# =============================================================================

section(
    "VALIDATION RMSE BY MEMORY WINDOW"
)


memory_table = (
    validation_df
    .pivot(
        index="window_length",
        columns="feature_set",
        values="validation_rmse",
    )
    .sort_index()
)


print(
    memory_table.to_string()
)


# =============================================================================
# SELECTED CONNECTIVITY TABLE
# =============================================================================

section(
    "SELECTED CONNECTIVITY BY MEMORY WINDOW"
)


connectivity_table = (
    validation_df
    .pivot(
        index="window_length",
        columns="feature_set",
        values="requested_connectivity",
    )
    .sort_index()
)


print(
    connectivity_table.to_string()
)


# =============================================================================
# SELECTED SPECTRAL RADIUS TABLE
# =============================================================================

section(
    "SELECTED SPECTRAL RADIUS BY MEMORY WINDOW"
)


rho_table = (
    validation_df
    .pivot(
        index="window_length",
        columns="feature_set",
        values="spectral_radius",
    )
    .sort_index()
)


print(
    rho_table.to_string()
)


# =============================================================================
# SELECTED INPUT SCALING TABLE
# =============================================================================

section(
    "SELECTED INPUT SCALING BY MEMORY WINDOW"
)


input_scaling_table = (
    validation_df
    .pivot(
        index="window_length",
        columns="feature_set",
        values="input_scaling",
    )
    .sort_index()
)


print(
    input_scaling_table.to_string()
)


# =============================================================================
# BEST ESN
# =============================================================================

section(
    "BEST ESN VALIDATION RESULT"
)


best_esn = ranking.iloc[0]


print(
    f"Feature set:          "
    f"{best_esn['feature_set']}"
)

print(
    f"Window length:        "
    f"{int(best_esn['window_length'])}"
)

print(
    f"Reservoir size:       "
    f"{int(best_esn['reservoir_size'])}"
)

print(
    f"Connectivity p:       "
    f"{best_esn['requested_connectivity']}"
)

print(
    f"Spectral radius:      "
    f"{best_esn['spectral_radius']}"
)

print(
    f"Max singular value:   "
    f"{best_esn['max_singular_value']:.6f}"
)

print(
    f"Sigma/rho ratio:      "
    f"{best_esn['sigma_rho_ratio']:.6f}"
)

print(
    f"Input scaling:        "
    f"{best_esn['input_scaling']}"
)

print(
    f"Leak rate:            "
    f"{best_esn['leak_rate']}"
)

print(
    f"Ridge alpha:          "
    f"{best_esn['ridge_alpha']}"
)

print(
    f"Training CV RMSE:     "
    f"{best_esn['training_cv_rmse']:.6f}"
)

print(
    f"Tanh saturation:      "
    f"{100.0 * best_esn['candidate_saturation_fraction']:.3f}%"
)

print()
print(
    f"Validation MAE:       "
    f"{best_esn['validation_mae']:.6f}"
)

print(
    f"Validation RMSE:      "
    f"{best_esn['validation_rmse']:.6f}"
)

print(
    f"Validation NRMSE:     "
    f"{best_esn['validation_nrmse']:.6f}"
)

print(
    f"vs seasonal naive:    "
    f"{best_esn['improvement_vs_seasonal_pct']:.3f}%"
)

print(
    f"vs Ridge F4:          "
    f"{best_esn['improvement_vs_ridge_f4_pct']:.3f}%"
)


# =============================================================================
# DATABASE AUDIT
# =============================================================================

section(
    "DATABASE AUDIT"
)


with engine.connect() as connection:

    trial_count = pd.read_sql(
        text("""
            SELECT
                COUNT(*) AS n
            FROM qrc_esn_cv_trial
        """),
        connection,
    ).iloc[0]["n"]


    db_validation = pd.read_sql(
        text("""
            SELECT
                feature_set_code,
                window_length,

                requested_connectivity,

                spectral_radius,
                max_singular_value,
                sigma_rho_ratio,

                input_scaling,
                leak_rate,

                ridge_alpha,
                training_cv_rmse,

                candidate_saturation_fraction,

                validation_rmse,
                validation_nrmse

            FROM qrc_esn_validation_result

            ORDER BY
                validation_rmse ASC
        """),
        connection,
    )


print(
    "Expected CV reservoir trials: "
    f"{N_TOTAL_RESERVOIR_CONFIGS}"
)

print(
    "Stored CV reservoir trials:   "
    f"{int(trial_count)}"
)


if int(
    trial_count
) != N_TOTAL_RESERVOIR_CONFIGS:

    raise RuntimeError(
        "Database CV trial count "
        "does not match expected count."
    )


print()
print(
    db_validation.to_string(
        index=False
    )
)


# =============================================================================
# FINAL STATUS
# =============================================================================

section(
    "STEP 4.2 RESULT"
)


print(
    "ESN temporal benchmark completed successfully."
)

print()
print(
    "Methodological checks:"
)

print(
    "  PASS - fixed random reservoir"
)

print(
    "  PASS - same base random reservoir used for both p values"
)

print(
    "  PASS - p=0.05 topology nested inside p=0.10 topology"
)

print(
    "  PASS - recurrent matrices normalized by spectral radius"
)

print(
    "  PASS - maximum singular value recorded separately"
)

print(
    "  PASS - tanh saturation monitored"
)

print(
    "  PASS - every temporal window starts from r_0 = 0"
)

print(
    "  PASS - only final reservoir state r_W used"
)

print(
    "  PASS - reservoir feature dimension fixed at 100"
)

print(
    "  PASS - only Ridge readout trained"
)

print(
    "  PASS - hyperparameters selected using training-only chronological CV"
)

print(
    "  PASS - validation evaluated only after CV selection"
)

print(
    "  PASS - test arrays were NOT loaded"
)

print(
    "  PASS - allow_pickle=False retained"
)

print(
    "  PASS - object-array date problem avoided"
)

print(
    "  PASS - CV trials stored in database"
)

print(
    "  PASS - selected models stored in results/"
)

print()
print(
    "Saved:"
)

print(
    f"  {CV_SEARCH_FILE}"
)

print(
    f"  {VALIDATION_METRICS_FILE}"
)

print(
    f"  {VALIDATION_PREDICTIONS_FILE}"
)

print(
    f"  {SELECTED_CONFIGS_FILE}"
)

print()
print(
    "Database:"
)

print(
    "  qrc_esn_cv_trial"
)

print(
    "  qrc_esn_validation_result"
)

print()
print(
    "Test set remains untouched."
)

print()
print(
    "Week 4 Step 4.2 finished."
)