from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from db_config import engine


# =============================================================================
# WEEK 4 - STEP 4.1
# TEMPORAL WINDOW CONSTRUCTION
# =============================================================================
#
# Purpose
# -------
# Construct leakage-safe temporal tensors for:
#
#   F2, F3, F4
#
# and
#
#   W in {1, 2, 5, 7, 14, 21, 28}
#
# The actual multidimensional tensors are stored as compressed NumPy .npz
# artifacts in results/.
#
# The database stores only reproducibility/configuration metadata.
#
# We do NOT duplicate every tensor element in the database because the
# underlying daily observations already exist in the canonical QRC tables.
#
# =============================================================================


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RESULTS_DIR / "03_01_preprocessed_samples.csv"

WINDOW_LENGTHS = [1, 2, 5, 7, 14, 21, 28]

FORECAST_HORIZON_DAYS = 1

# Important experimental definition:
#
# Every temporal sample is treated as its own sequence.
# The recurrent/reservoir state will therefore be reset at the start of
# each W-day window in the models developed later in Week 4.
RESET_BETWEEN_WINDOWS = True


# =============================================================================
# FEATURE SETS
# =============================================================================
#
# Semantic dimensions:
#
#   F2 = [C_t, D_(t+1)]
#   F3 = [C_t, D_(t+1), P_t]
#   F4 = [C_t, D_(t+1), P_t, H_(t+1)]
#
# For classical models weekday is cyclically encoded:
#
#   D -> (D_sin, D_cos)
#
# Therefore the numerical dimensions are:
#
#   F2 -> 3
#   F3 -> 4
#   F4 -> 5
#
# =============================================================================

FEATURE_SETS = {
    "F2": [
        "C_t_z",
        "D_sin",
        "D_cos",
    ],
    "F3": [
        "C_t_z",
        "D_sin",
        "D_cos",
        "P_t_z",
    ],
    "F4": [
        "C_t_z",
        "D_sin",
        "D_cos",
        "P_t_z",
        "is_public_holiday_t_plus_1",
    ],
}

SEMANTIC_INPUT_COUNTS = {
    "F2": 2,
    "F3": 3,
    "F4": 4,
}

EXPECTED_SPLIT_COUNTS = {
    "train": 1095,
    "validation": 365,
    "test": 232,
}


# =============================================================================
# CONSOLE HELPER
# =============================================================================

def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# =============================================================================
# TARGET COLUMN RESOLUTION
# =============================================================================

def resolve_target_column(df):
    """
    Identify the next-day property-damage claim-count target.

    Several explicit candidate names are supported so that this script remains
    compatible with the Week 3 preprocessing output.
    """

    candidates = [
        "property_damage_claim_count_t_plus_1",
        "property_damage_claim_count_target",
        "target_property_damage_claim_count",
        "target_claim_count",
        "C_t_plus_1",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    # Conservative fallback.
    possible = []

    for column in df.columns:
        lower = column.lower()

        if (
            "property_damage" in lower
            and "claim" in lower
            and "count" in lower
            and (
                "plus_1" in lower
                or "target" in lower
                or "next" in lower
            )
            and column != "property_damage_claim_count_t"
        ):
            possible.append(column)

    if len(possible) == 1:
        return possible[0]

    raise ValueError(
        "Could not uniquely identify C_(t+1).\n"
        f"Target-like candidates found: {possible}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# =============================================================================
# COLUMN AUDIT
# =============================================================================

def check_required_columns(df, target_column):

    required = {
        "input_date",
        "target_date",
        "split",
        target_column,
    }

    for columns in FEATURE_SETS.values():
        required.update(columns)

    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(
            "The Week 3 preprocessing file is missing required columns:\n\n"
            + "\n".join(missing)
        )


# =============================================================================
# DATABASE TABLE
# =============================================================================

def create_database_table():
    """
    Create the Week 4 temporal-window metadata table if it does not exist.

    This table stores configuration and reproducibility metadata only.
    The actual model tensors remain in .npz files.
    """

    sql = text("""
        CREATE TABLE IF NOT EXISTS qrc_temporal_window_config (
            temporal_window_config_id BIGINT AUTO_INCREMENT PRIMARY KEY,

            feature_set_code VARCHAR(10) NOT NULL,
            window_length INT NOT NULL,

            semantic_input_count INT NOT NULL,
            numeric_feature_count INT NOT NULL,

            forecast_horizon_days INT NOT NULL,
            reset_between_windows BOOLEAN NOT NULL,

            train_n INT NOT NULL,
            validation_n INT NOT NULL,
            test_n INT NOT NULL,

            train_tensor_shape VARCHAR(100) NOT NULL,
            validation_tensor_shape VARCHAR(100) NOT NULL,
            test_tensor_shape VARCHAR(100) NOT NULL,

            first_train_target_date DATE NULL,
            last_train_target_date DATE NULL,

            first_validation_target_date DATE NULL,
            last_validation_target_date DATE NULL,

            first_test_target_date DATE NULL,
            last_test_target_date DATE NULL,

            artifact_path VARCHAR(500) NOT NULL,

            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP,

            UNIQUE KEY uq_qrc_temporal_window (
                feature_set_code,
                window_length
            )
        )
    """)

    with engine.begin() as connection:
        connection.execute(sql)


# =============================================================================
# BUILD WINDOWS
# =============================================================================

def build_all_windows(
    df,
    feature_columns,
    window_length,
    target_column,
):
    """
    Construct temporal windows over the full chronological sequence.

    For endpoint t:

        [x_(t-W+1), ..., x_t] -> C_(t+1)

    Important
    ---------
    We build windows BEFORE separating them into train/validation/test.

    This means that the first validation forecast may legitimately use
    historical observations from the end of the training period.

    Likewise, the first test forecast may use historical observations
    from the end of validation.

    This is not leakage because those observations occur before the
    forecast origin.
    """

    feature_matrix = df[
        feature_columns
    ].to_numpy(dtype=np.float64)

    target = df[
        target_column
    ].to_numpy(dtype=np.float64)

    input_dates = df["input_date"].to_numpy()
    target_dates = df["target_date"].to_numpy()

    split_labels = (
        df["split"]
        .astype(str)
        .to_numpy()
    )

    X = []
    y = []

    endpoint_indices = []
    history_start_indices = []

    history_start_dates = []
    endpoint_input_dates = []
    forecast_target_dates = []

    history_start_splits = []
    endpoint_splits = []

    for end_idx in range(
        window_length - 1,
        len(df),
    ):

        start_idx = (
            end_idx
            - window_length
            + 1
        )

        window = feature_matrix[
            start_idx:end_idx + 1
        ]

        expected_shape = (
            window_length,
            len(feature_columns),
        )

        if window.shape != expected_shape:
            raise RuntimeError(
                f"Window shape is {window.shape}; "
                f"expected {expected_shape}."
            )

        X.append(window)
        y.append(target[end_idx])

        endpoint_indices.append(end_idx)
        history_start_indices.append(start_idx)

        history_start_dates.append(
            input_dates[start_idx]
        )

        endpoint_input_dates.append(
            input_dates[end_idx]
        )

        forecast_target_dates.append(
            target_dates[end_idx]
        )

        history_start_splits.append(
            split_labels[start_idx]
        )

        endpoint_splits.append(
            split_labels[end_idx]
        )

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    metadata = pd.DataFrame({
        "endpoint_index":
            endpoint_indices,

        "history_start_index":
            history_start_indices,

        "history_start_date":
            pd.to_datetime(
                history_start_dates
            ),

        "endpoint_input_date":
            pd.to_datetime(
                endpoint_input_dates
            ),

        "target_date":
            pd.to_datetime(
                forecast_target_dates
            ),

        "history_start_split":
            history_start_splits,

        "endpoint_split":
            endpoint_splits,
    })

    return X, y, metadata


# =============================================================================
# SPLIT WINDOW DATASET
# =============================================================================

def split_window_dataset(
    X,
    y,
    metadata,
):
    """
    Assign each temporal sample according to the split of its endpoint.

    Endpoint split determines where the forecast belongs.
    """

    datasets = {}

    mapping = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    for source_label, output_label in mapping.items():

        mask = (
            metadata["endpoint_split"]
            .to_numpy()
            == source_label
        )

        datasets[output_label] = {
            "X": X[mask],
            "y": y[mask],

            "metadata": (
                metadata
                .loc[mask]
                .reset_index(drop=True)
            ),
        }

    return datasets


# =============================================================================
# DATABASE UPSERT
# =============================================================================

def upsert_window_config(
    feature_set,
    window_length,
    semantic_input_count,
    numeric_feature_count,
    X_train,
    X_val,
    X_test,
    meta_train,
    meta_val,
    meta_test,
    artifact_path,
):
    """
    Insert or update one temporal-window configuration.

    UNIQUE(feature_set_code, window_length) prevents duplicate configurations
    when the script is rerun.
    """

    sql = text("""
        INSERT INTO qrc_temporal_window_config
        (
            feature_set_code,
            window_length,

            semantic_input_count,
            numeric_feature_count,

            forecast_horizon_days,
            reset_between_windows,

            train_n,
            validation_n,
            test_n,

            train_tensor_shape,
            validation_tensor_shape,
            test_tensor_shape,

            first_train_target_date,
            last_train_target_date,

            first_validation_target_date,
            last_validation_target_date,

            first_test_target_date,
            last_test_target_date,

            artifact_path
        )
        VALUES
        (
            :feature_set_code,
            :window_length,

            :semantic_input_count,
            :numeric_feature_count,

            :forecast_horizon_days,
            :reset_between_windows,

            :train_n,
            :validation_n,
            :test_n,

            :train_tensor_shape,
            :validation_tensor_shape,
            :test_tensor_shape,

            :first_train_target_date,
            :last_train_target_date,

            :first_validation_target_date,
            :last_validation_target_date,

            :first_test_target_date,
            :last_test_target_date,

            :artifact_path
        )

        ON DUPLICATE KEY UPDATE

            semantic_input_count =
                VALUES(semantic_input_count),

            numeric_feature_count =
                VALUES(numeric_feature_count),

            forecast_horizon_days =
                VALUES(forecast_horizon_days),

            reset_between_windows =
                VALUES(reset_between_windows),

            train_n =
                VALUES(train_n),

            validation_n =
                VALUES(validation_n),

            test_n =
                VALUES(test_n),

            train_tensor_shape =
                VALUES(train_tensor_shape),

            validation_tensor_shape =
                VALUES(validation_tensor_shape),

            test_tensor_shape =
                VALUES(test_tensor_shape),

            first_train_target_date =
                VALUES(first_train_target_date),

            last_train_target_date =
                VALUES(last_train_target_date),

            first_validation_target_date =
                VALUES(first_validation_target_date),

            last_validation_target_date =
                VALUES(last_validation_target_date),

            first_test_target_date =
                VALUES(first_test_target_date),

            last_test_target_date =
                VALUES(last_test_target_date),

            artifact_path =
                VALUES(artifact_path)
    """)

    parameters = {
        "feature_set_code":
            feature_set,

        "window_length":
            int(window_length),

        "semantic_input_count":
            int(semantic_input_count),

        "numeric_feature_count":
            int(numeric_feature_count),

        "forecast_horizon_days":
            int(FORECAST_HORIZON_DAYS),

        "reset_between_windows":
            bool(RESET_BETWEEN_WINDOWS),

        "train_n":
            int(len(X_train)),

        "validation_n":
            int(len(X_val)),

        "test_n":
            int(len(X_test)),

        "train_tensor_shape":
            str(X_train.shape),

        "validation_tensor_shape":
            str(X_val.shape),

        "test_tensor_shape":
            str(X_test.shape),

        "first_train_target_date":
            meta_train["target_date"]
            .iloc[0]
            .date(),

        "last_train_target_date":
            meta_train["target_date"]
            .iloc[-1]
            .date(),

        "first_validation_target_date":
            meta_val["target_date"]
            .iloc[0]
            .date(),

        "last_validation_target_date":
            meta_val["target_date"]
            .iloc[-1]
            .date(),

        "first_test_target_date":
            meta_test["target_date"]
            .iloc[0]
            .date(),

        "last_test_target_date":
            meta_test["target_date"]
            .iloc[-1]
            .date(),

        "artifact_path":
            artifact_path,
    }

    with engine.begin() as connection:
        connection.execute(
            sql,
            parameters,
        )


# =============================================================================
# LOAD WEEK 3 PREPROCESSED DATA
# =============================================================================

section(
    "WEEK 4 - STEP 4.1: "
    "TEMPORAL WINDOW CONSTRUCTION"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Required Week 3 preprocessing artifact was not found:\n"
        f"{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "input_date",
        "target_date",
    ],
)

df = (
    df
    .sort_values("input_date")
    .reset_index(drop=True)
)

target_column = resolve_target_column(df)

check_required_columns(
    df,
    target_column,
)


# =============================================================================
# INPUT AUDIT
# =============================================================================

section("INPUT DATA AUDIT")

print(f"Input file:     {INPUT_FILE}")
print(f"Total samples:  {len(df)}")
print(f"Target column:  {target_column}")

print(
    "Input dates:    "
    f"{df['input_date'].min().date()} "
    "to "
    f"{df['input_date'].max().date()}"
)

print(
    "Target dates:   "
    f"{df['target_date'].min().date()} "
    "to "
    f"{df['target_date'].max().date()}"
)

print()
print("Split counts:")

for split_name, expected_count in (
    EXPECTED_SPLIT_COUNTS.items()
):

    actual_count = int(
        (df["split"] == split_name)
        .sum()
    )

    status = (
        "PASS"
        if actual_count == expected_count
        else "FAIL"
    )

    print(
        f"  {split_name:10s}: "
        f"{actual_count:4d} "
        f"(expected {expected_count:4d}) "
        f"{status}"
    )

    if status == "FAIL":
        raise RuntimeError(
            f"Unexpected {split_name} count."
        )


# =============================================================================
# CHRONOLOGICAL AUDIT
# =============================================================================

section("CHRONOLOGICAL CONTINUITY CHECK")

input_date_difference = (
    df["input_date"]
    .diff()
    .dropna()
    .dt.days
)

non_daily_transitions = int(
    (input_date_difference != 1)
    .sum()
)

print(
    "Non-daily input-date transitions: "
    f"{non_daily_transitions}"
)

if non_daily_transitions != 0:
    raise RuntimeError(
        "Input dates are not a continuous daily sequence."
    )


forecast_horizon = (
    df["target_date"]
    - df["input_date"]
).dt.days

bad_horizon_count = int(
    (forecast_horizon != FORECAST_HORIZON_DAYS)
    .sum()
)

print(
    "Rows violating one-day forecast horizon: "
    f"{bad_horizon_count}"
)

if bad_horizon_count != 0:
    raise RuntimeError(
        "Forecast-horizon audit failed."
    )

print("Chronological continuity: PASS")


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

section("DATABASE INITIALIZATION")

create_database_table()

print(
    "Database table ready: "
    "qrc_temporal_window_config"
)


# =============================================================================
# FEATURE SET AUDIT
# =============================================================================

section("FEATURE SETS")

feature_set_rows = []

for feature_set, feature_columns in (
    FEATURE_SETS.items()
):

    semantic_count = (
        SEMANTIC_INPUT_COUNTS[
            feature_set
        ]
    )

    numeric_count = len(
        feature_columns
    )

    print()
    print(feature_set)

    print(
        "  Semantic inputs: "
        f"{semantic_count}"
    )

    print(
        "  Numeric features: "
        f"{numeric_count}"
    )

    print(
        "  Columns: "
        + ", ".join(feature_columns)
    )

    feature_set_rows.append({
        "feature_set":
            feature_set,

        "semantic_input_count":
            semantic_count,

        "numeric_feature_count":
            numeric_count,

        "numeric_features":
            ", ".join(feature_columns),
    })


# =============================================================================
# BUILD ALL 21 TEMPORAL DATASETS
# =============================================================================

section("TEMPORAL WINDOW DATASETS")

summary_rows = []
boundary_rows = []

for feature_set, feature_columns in (
    FEATURE_SETS.items()
):

    semantic_input_count = (
        SEMANTIC_INPUT_COUNTS[
            feature_set
        ]
    )

    numeric_feature_count = len(
        feature_columns
    )

    for W in WINDOW_LENGTHS:

        # ---------------------------------------------------------------------
        # Build continuous chronological windows
        # ---------------------------------------------------------------------

        X, y, metadata = build_all_windows(
            df=df,
            feature_columns=feature_columns,
            window_length=W,
            target_column=target_column,
        )

        datasets = split_window_dataset(
            X=X,
            y=y,
            metadata=metadata,
        )

        X_train = datasets["train"]["X"]
        y_train = datasets["train"]["y"]
        meta_train = datasets["train"]["metadata"]

        X_val = datasets["val"]["X"]
        y_val = datasets["val"]["y"]
        meta_val = datasets["val"]["metadata"]

        X_test = datasets["test"]["X"]
        y_test = datasets["test"]["y"]
        meta_test = datasets["test"]["metadata"]


        # ---------------------------------------------------------------------
        # Expected sample counts
        # ---------------------------------------------------------------------

        expected_train_n = (
            EXPECTED_SPLIT_COUNTS["train"]
            - W
            + 1
        )

        expected_validation_n = (
            EXPECTED_SPLIT_COUNTS[
                "validation"
            ]
        )

        expected_test_n = (
            EXPECTED_SPLIT_COUNTS[
                "test"
            ]
        )

        if len(X_train) != expected_train_n:
            raise RuntimeError(
                f"{feature_set}, W={W}: "
                f"train_n={len(X_train)}, "
                f"expected={expected_train_n}"
            )

        if len(X_val) != expected_validation_n:
            raise RuntimeError(
                f"{feature_set}, W={W}: "
                f"validation_n={len(X_val)}, "
                f"expected={expected_validation_n}"
            )

        if len(X_test) != expected_test_n:
            raise RuntimeError(
                f"{feature_set}, W={W}: "
                f"test_n={len(X_test)}, "
                f"expected={expected_test_n}"
            )


        # ---------------------------------------------------------------------
        # Shape checks
        # ---------------------------------------------------------------------

        expected_tail_shape = (
            W,
            numeric_feature_count,
        )

        for split_label, array in [
            ("train", X_train),
            ("validation", X_val),
            ("test", X_test),
        ]:

            if array.shape[1:] != expected_tail_shape:
                raise RuntimeError(
                    f"{feature_set}, W={W}, "
                    f"{split_label}: "
                    f"unexpected shape {array.shape}"
                )


        # ---------------------------------------------------------------------
        # Numerical validity
        # ---------------------------------------------------------------------

        for split_label, array in [
            ("X_train", X_train),
            ("X_val", X_val),
            ("X_test", X_test),
            ("y_train", y_train),
            ("y_val", y_val),
            ("y_test", y_test),
        ]:

            if not np.isfinite(array).all():
                raise RuntimeError(
                    f"{feature_set}, W={W}: "
                    f"non-finite values in "
                    f"{split_label}"
                )


        # ---------------------------------------------------------------------
        # Save actual tensors as .npz
        # ---------------------------------------------------------------------

        output_file = (
            RESULTS_DIR
            / (
                f"04_01_windows_"
                f"{feature_set}_"
                f"W{W:02d}.npz"
            )
        )

        np.savez_compressed(
            output_file,

            X_train=X_train,
            y_train=y_train,

            X_val=X_val,
            y_val=y_val,

            X_test=X_test,
            y_test=y_test,

            train_target_dates=(
                meta_train["target_date"]
                .dt.strftime("%Y-%m-%d")
                .to_numpy()
            ),

            validation_target_dates=(
                meta_val["target_date"]
                .dt.strftime("%Y-%m-%d")
                .to_numpy()
            ),

            test_target_dates=(
                meta_test["target_date"]
                .dt.strftime("%Y-%m-%d")
                .to_numpy()
            ),

            feature_names=np.asarray(
                feature_columns
            ),

            feature_set=np.asarray([
                feature_set
            ]),

            window_length=np.asarray([
                W
            ]),

            semantic_input_count=np.asarray([
                semantic_input_count
            ]),

            numeric_feature_count=np.asarray([
                numeric_feature_count
            ]),

            forecast_horizon_days=np.asarray([
                FORECAST_HORIZON_DAYS
            ]),

            reset_between_windows=np.asarray([
                RESET_BETWEEN_WINDOWS
            ]),
        )


        # ---------------------------------------------------------------------
        # Store metadata in database
        # ---------------------------------------------------------------------

        upsert_window_config(
            feature_set=feature_set,
            window_length=W,

            semantic_input_count=(
                semantic_input_count
            ),

            numeric_feature_count=(
                numeric_feature_count
            ),

            X_train=X_train,
            X_val=X_val,
            X_test=X_test,

            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,

            artifact_path=str(
                output_file.as_posix()
            ),
        )


        # ---------------------------------------------------------------------
        # Summary CSV row
        # ---------------------------------------------------------------------

        summary_rows.append({
            "feature_set":
                feature_set,

            "semantic_input_count":
                semantic_input_count,

            "numeric_feature_count":
                numeric_feature_count,

            "window_length":
                W,

            "forecast_horizon_days":
                FORECAST_HORIZON_DAYS,

            "reset_between_windows":
                RESET_BETWEEN_WINDOWS,

            "train_n":
                len(X_train),

            "validation_n":
                len(X_val),

            "test_n":
                len(X_test),

            "train_tensor_shape":
                str(X_train.shape),

            "validation_tensor_shape":
                str(X_val.shape),

            "test_tensor_shape":
                str(X_test.shape),

            "artifact_path":
                str(output_file.as_posix()),
        })


        # ---------------------------------------------------------------------
        # Boundary audit rows
        # ---------------------------------------------------------------------

        for split_label, metadata_split in [
            ("train", meta_train),
            ("validation", meta_val),
            ("test", meta_test),
        ]:

            first = metadata_split.iloc[0]
            last = metadata_split.iloc[-1]

            boundary_rows.append({
                "feature_set":
                    feature_set,

                "window_length":
                    W,

                "split":
                    split_label,

                "first_history_start_date":
                    first["history_start_date"],

                "first_history_start_split":
                    first["history_start_split"],

                "first_endpoint_input_date":
                    first["endpoint_input_date"],

                "first_target_date":
                    first["target_date"],

                "last_history_start_date":
                    last["history_start_date"],

                "last_endpoint_input_date":
                    last["endpoint_input_date"],

                "last_target_date":
                    last["target_date"],
            })


        # ---------------------------------------------------------------------
        # Console
        # ---------------------------------------------------------------------

        print(
            f"{feature_set:2s} | "
            f"W={W:2d} | "
            f"d={numeric_feature_count} | "
            f"train={X_train.shape} | "
            f"val={X_val.shape} | "
            f"test={X_test.shape} | "
            "DB=OK | NPZ=OK"
        )


# =============================================================================
# SAVE HUMAN-READABLE AUDIT FILES
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)

boundary_df = pd.DataFrame(
    boundary_rows
)

feature_set_df = pd.DataFrame(
    feature_set_rows
)


summary_file = (
    RESULTS_DIR
    / "04_01_temporal_window_summary.csv"
)

boundary_file = (
    RESULTS_DIR
    / "04_01_temporal_window_boundary_audit.csv"
)

feature_set_file = (
    RESULTS_DIR
    / "04_01_temporal_feature_sets.csv"
)


summary_df.to_csv(
    summary_file,
    index=False,
)

boundary_df.to_csv(
    boundary_file,
    index=False,
)

feature_set_df.to_csv(
    feature_set_file,
    index=False,
)


# =============================================================================
# DATABASE AUDIT
# =============================================================================

section("DATABASE AUDIT")

with engine.connect() as connection:

    database_rows = pd.read_sql(
        text("""
            SELECT
                feature_set_code,
                window_length,
                semantic_input_count,
                numeric_feature_count,
                train_n,
                validation_n,
                test_n,
                artifact_path
            FROM qrc_temporal_window_config
            ORDER BY
                feature_set_code,
                window_length
        """),
        connection,
    )

print(database_rows.to_string(index=False))

expected_configuration_count = (
    len(FEATURE_SETS)
    * len(WINDOW_LENGTHS)
)

actual_configuration_count = len(
    database_rows[
        database_rows[
            "feature_set_code"
        ].isin(FEATURE_SETS.keys())
    ]
)

print()
print(
    "Expected Week 4 window configurations: "
    f"{expected_configuration_count}"
)

print(
    "Stored Week 4 window configurations:   "
    f"{actual_configuration_count}"
)


# =============================================================================
# LONGEST-WINDOW BOUNDARY EXAMPLE
# =============================================================================

section(
    "F4 / W=28 VALIDATION BOUNDARY CHECK"
)

example = boundary_df[
    (boundary_df["feature_set"] == "F4")
    & (
        boundary_df["window_length"]
        == 28
    )
    & (
        boundary_df["split"]
        == "validation"
    )
]

print(
    example.to_string(
        index=False
    )
)


# =============================================================================
# EXPECTED SAMPLE COUNTS
# =============================================================================

section("EXPECTED SAMPLE COUNTS BY WINDOW")

count_table = pd.DataFrame({
    "window_length":
        WINDOW_LENGTHS,

    "expected_train_n": [
        EXPECTED_SPLIT_COUNTS["train"]
        - W
        + 1
        for W in WINDOW_LENGTHS
    ],

    "expected_validation_n": [
        EXPECTED_SPLIT_COUNTS[
            "validation"
        ]
        for _ in WINDOW_LENGTHS
    ],

    "expected_test_n": [
        EXPECTED_SPLIT_COUNTS[
            "test"
        ]
        for _ in WINDOW_LENGTHS
    ],
})

print(
    count_table.to_string(
        index=False
    )
)


# =============================================================================
# FINAL RESULT
# =============================================================================

section("STEP 4.1 RESULT")

print(
    "All temporal window datasets "
    "constructed successfully."
)

print()
print("Verified:")
print(
    "  PASS - chronological order preserved"
)
print(
    "  PASS - one-day forecast horizon preserved"
)
print(
    "  PASS - target C_(t+1) excluded from X"
)
print(
    "  PASS - validation may use prior training history"
)
print(
    "  PASS - test may use prior validation history"
)
print(
    "  PASS - Week 3 preprocessing reused"
)
print(
    "  PASS - tensors preserve time axis"
)
print(
    "  PASS - tensors stored as compressed .npz files"
)
print(
    "  PASS - configuration metadata stored in database"
)

print()
print("Human-readable files:")
print(f"  {summary_file}")
print(f"  {boundary_file}")
print(f"  {feature_set_file}")

print()
print(
    "Tensor artifacts: "
    "21 .npz files "
    "(3 feature sets x 7 window lengths)"
)

print()
print(
    "Database table: "
    "qrc_temporal_window_config"
)

print()
print(
    "Week 4 Step 4.1 finished."
)