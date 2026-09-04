from pathlib import Path

import numpy as np
import pandas as pd

from db_config import engine


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Load forecasting samples
# ---------------------------------------------------------------------

query = """
SELECT
    input_date,
    target_date,
    property_damage_claim_count_t,
    active_policy_count_t,
    day_of_week_t_plus_1,
    is_public_holiday_t_plus_1,
    target_property_damage_claim_count,
    split
FROM qrc_forecast_sample
ORDER BY input_date
"""

df = pd.read_sql(query, engine)

df["input_date"] = pd.to_datetime(df["input_date"])
df["target_date"] = pd.to_datetime(df["target_date"])


numeric_columns = [
    "property_damage_claim_count_t",
    "active_policy_count_t",
    "day_of_week_t_plus_1",
    "is_public_holiday_t_plus_1",
    "target_property_damage_claim_count",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


# ---------------------------------------------------------------------
# Verify split sizes
# ---------------------------------------------------------------------

expected_counts = {
    "train": 1095,
    "validation": 365,
    "test": 232,
}

observed_counts = (
    df["split"]
    .value_counts()
    .to_dict()
)

print("=" * 78)
print("WEEK 3 - STEP 1: PREPROCESSING")
print("=" * 78)

print("\nSplit counts:")

for split_name in [
    "train",
    "validation",
    "test",
]:
    observed = observed_counts.get(
        split_name,
        0,
    )

    expected = expected_counts[
        split_name
    ]

    status = (
        "PASS"
        if observed == expected
        else "FAIL"
    )

    print(
        f"{split_name:10s}: "
        f"{observed:4d} "
        f"(expected {expected}) "
        f"{status}"
    )


# ---------------------------------------------------------------------
# Training set only
# ---------------------------------------------------------------------

train = df[
    df["split"] == "train"
].copy()


# ---------------------------------------------------------------------
# Calculate preprocessing parameters from TRAIN ONLY
# ---------------------------------------------------------------------
#
# We use population standard deviation (ddof=0), which is also
# the convention used by sklearn StandardScaler.
# ---------------------------------------------------------------------

continuous_variables = {
    "C_t": "property_damage_claim_count_t",
    "P_t": "active_policy_count_t",
}

parameter_rows = []


for feature_name, column in continuous_variables.items():

    mean_train = train[
        column
    ].mean()

    std_train = train[
        column
    ].std(ddof=0)

    parameter_rows.append(
        {
            "feature": feature_name,
            "column": column,
            "train_mean": mean_train,
            "train_std": std_train,
        }
    )


parameters = pd.DataFrame(
    parameter_rows
)


print("\n" + "=" * 78)
print("TRAINING PREPROCESSING PARAMETERS")
print("=" * 78)

print(
    parameters.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Extract parameters
# ---------------------------------------------------------------------

C_mean = parameters.loc[
    parameters["feature"] == "C_t",
    "train_mean",
].iloc[0]

C_std = parameters.loc[
    parameters["feature"] == "C_t",
    "train_std",
].iloc[0]


P_mean = parameters.loc[
    parameters["feature"] == "P_t",
    "train_mean",
].iloc[0]

P_std = parameters.loc[
    parameters["feature"] == "P_t",
    "train_std",
].iloc[0]


# ---------------------------------------------------------------------
# Standardize using TRAIN parameters
# ---------------------------------------------------------------------

df["C_t_z"] = (
    df["property_damage_claim_count_t"]
    - C_mean
) / C_std


df["P_t_z"] = (
    df["active_policy_count_t"]
    - P_mean
) / P_std


# ---------------------------------------------------------------------
# Cyclic weekday encoding
# ---------------------------------------------------------------------

df["weekday_angle"] = (
    2.0
    * np.pi
    * (
        df["day_of_week_t_plus_1"]
        - 1
    )
    / 7.0
)


df["D_sin"] = np.sin(
    df["weekday_angle"]
)

df["D_cos"] = np.cos(
    df["weekday_angle"]
)


# ---------------------------------------------------------------------
# Verify unit-circle geometry
# ---------------------------------------------------------------------

df["weekday_radius"] = np.sqrt(
    df["D_sin"] ** 2
    +
    df["D_cos"] ** 2
)


max_radius_error = np.max(
    np.abs(
        df["weekday_radius"]
        - 1.0
    )
)


# ---------------------------------------------------------------------
# Feature-set definitions
# ---------------------------------------------------------------------

feature_sets = {
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


feature_set_rows = []

for feature_set_name, columns in feature_sets.items():

    feature_set_rows.append(
        {
            "feature_set": feature_set_name,
            "semantic_dimension": {
                "F2": 2,
                "F3": 3,
                "F4": 4,
            }[feature_set_name],

            "classical_columns": ", ".join(
                columns
            ),
        }
    )


feature_set_df = pd.DataFrame(
    feature_set_rows
)


# ---------------------------------------------------------------------
# Check standardized TRAIN data
# ---------------------------------------------------------------------

train_processed = df[
    df["split"] == "train"
]


checks = pd.DataFrame(
    [
        {
            "variable": "C_t_z",
            "train_mean_after_scaling":
                train_processed[
                    "C_t_z"
                ].mean(),

            "train_std_after_scaling":
                train_processed[
                    "C_t_z"
                ].std(ddof=0),
        },

        {
            "variable": "P_t_z",
            "train_mean_after_scaling":
                train_processed[
                    "P_t_z"
                ].mean(),

            "train_std_after_scaling":
                train_processed[
                    "P_t_z"
                ].std(ddof=0),
        },
    ]
)


print("\n" + "=" * 78)
print("STANDARDIZATION CHECK")
print("=" * 78)

print(
    checks.to_string(
        index=False
    )
)


print("\n" + "=" * 78)
print("CYCLIC WEEKDAY ENCODING")
print("=" * 78)


weekday_examples = (
    df[
        [
            "day_of_week_t_plus_1",
            "D_sin",
            "D_cos",
            "weekday_radius",
        ]
    ]
    .drop_duplicates(
        subset=[
            "day_of_week_t_plus_1"
        ]
    )
    .sort_values(
        "day_of_week_t_plus_1"
    )
)


print(
    weekday_examples.round(6)
    .to_string(index=False)
)


print(
    "\nMaximum deviation from "
    f"unit-circle radius = "
    f"{max_radius_error:.12e}"
)


# ---------------------------------------------------------------------
# Print split behaviour after scaling
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("TRANSFORMED VARIABLES BY SPLIT")
print("=" * 78)


split_stats = (
    df.groupby("split")
    .agg(
        C_t_z_mean=("C_t_z", "mean"),
        C_t_z_std=("C_t_z", lambda x: x.std(ddof=0)),
        P_t_z_mean=("P_t_z", "mean"),
        P_t_z_std=("P_t_z", lambda x: x.std(ddof=0)),
    )
    .reset_index()
)


print(
    split_stats.round(6)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Missing-value check
# ---------------------------------------------------------------------

processed_columns = [
    "C_t_z",
    "P_t_z",
    "D_sin",
    "D_cos",
    "is_public_holiday_t_plus_1",
    "target_property_damage_claim_count",
]


missing_count = int(
    df[processed_columns]
    .isna()
    .sum()
    .sum()
)


print("\n" + "=" * 78)
print("PREPROCESSING CHECKS")
print("=" * 78)

print(
    f"Missing processed values: "
    f"{missing_count}"
)

print(
    "Target standardized: NO"
)

print(
    "Scaler fitted on training only: YES"
)


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

parameters.to_csv(
    RESULTS_DIR /
    "03_01_preprocessing_parameters.csv",
    index=False,
)


feature_set_df.to_csv(
    RESULTS_DIR /
    "03_01_feature_sets.csv",
    index=False,
)


df.to_csv(
    RESULTS_DIR /
    "03_01_preprocessed_samples.csv",
    index=False,
)


checks.to_csv(
    RESULTS_DIR /
    "03_01_standardization_checks.csv",
    index=False,
)


split_stats.to_csv(
    RESULTS_DIR /
    "03_01_transformed_split_statistics.csv",
    index=False,
)


print("\nSaved:")
print(
    " results/"
    "03_01_preprocessing_parameters.csv"
)
print(
    " results/"
    "03_01_feature_sets.csv"
)
print(
    " results/"
    "03_01_preprocessed_samples.csv"
)
print(
    " results/"
    "03_01_standardization_checks.csv"
)
print(
    " results/"
    "03_01_transformed_split_statistics.csv"
)

print(
    "\nStep 1 preprocessing finished."
)