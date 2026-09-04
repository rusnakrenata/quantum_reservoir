from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from db_config import engine


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Load data
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


# ---------------------------------------------------------------------
# Training set only
# ---------------------------------------------------------------------

train = df[df["split"] == "train"].copy()

required_columns = [
    "property_damage_claim_count_t",
    "active_policy_count_t",
    "day_of_week_t_plus_1",
    "is_public_holiday_t_plus_1",
    "target_property_damage_claim_count",
]

for col in required_columns:
    train[col] = pd.to_numeric(
        train[col],
        errors="coerce",
    )

train = train.dropna(
    subset=required_columns
).copy()

train["year"] = train["target_date"].dt.year


print("=" * 78)
print("WEEK 2 - STEP 11: REGIME / STRUCTURAL-STABILITY SENSITIVITY")
print("=" * 78)

print(f"\nTraining samples: {len(train)}")

print(
    f"Training period: "
    f"{train['target_date'].min().date()} "
    f"to {train['target_date'].max().date()}"
)

print("\nSamples by year:")
print(
    train.groupby("year")
    .size()
    .to_string()
)


# ---------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------

feature_map = {
    "C_t": "property_damage_claim_count_t",
    "D_t+1": "day_of_week_t_plus_1",
    "P_t": "active_policy_count_t",
    "H_t+1": "is_public_holiday_t_plus_1",
}

discrete_map = {
    "C_t": True,
    "D_t+1": True,
    "P_t": False,
    "H_t+1": True,
}

target_col = "target_property_damage_claim_count"


# ---------------------------------------------------------------------
# Helper: mutual information
# ---------------------------------------------------------------------

def calculate_mi(x, y, discrete):

    x = np.asarray(x).reshape(-1, 1)

    mi = mutual_info_regression(
        x,
        y,
        discrete_features=[discrete],
        random_state=RANDOM_STATE,
    )

    return float(mi[0])


# ---------------------------------------------------------------------
# Helper: analyse one regime
# ---------------------------------------------------------------------

def analyse_regime(data, regime_name):

    rows = []

    y = data[target_col].to_numpy(dtype=float)

    for feature_name, column in feature_map.items():

        x = data[column].to_numpy(dtype=float)

        pearson = pd.Series(x).corr(
            pd.Series(y),
            method="pearson",
        )

        spearman = pd.Series(x).corr(
            pd.Series(y),
            method="spearman",
        )

        mi = calculate_mi(
            x=x,
            y=y,
            discrete=discrete_map[feature_name],
        )

        rows.append(
            {
                "regime": regime_name,
                "n": len(data),
                "feature": feature_name,
                "pearson": pearson,
                "spearman": spearman,
                "mutual_information": mi,
            }
        )

    return rows


# ---------------------------------------------------------------------
# Analyse full training set
# ---------------------------------------------------------------------

all_results = []

all_results.extend(
    analyse_regime(
        train,
        "FULL_TRAIN",
    )
)


# ---------------------------------------------------------------------
# Analyse each year separately
# ---------------------------------------------------------------------

for year in sorted(
    train["year"].unique()
):

    subset = train[
        train["year"] == year
    ].copy()

    all_results.extend(
        analyse_regime(
            subset,
            str(year),
        )
    )


# ---------------------------------------------------------------------
# Early versus late training regime
# ---------------------------------------------------------------------
#
# Split chronologically near the middle of training.
# ---------------------------------------------------------------------

midpoint = len(train) // 2

early = train.iloc[
    :midpoint
].copy()

late = train.iloc[
    midpoint:
].copy()


all_results.extend(
    analyse_regime(
        early,
        "EARLY_HALF",
    )
)

all_results.extend(
    analyse_regime(
        late,
        "LATE_HALF",
    )
)


results = pd.DataFrame(
    all_results
)


# ---------------------------------------------------------------------
# Print correlations
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("PEARSON CORRELATION BY REGIME")
print("=" * 78)

pearson_table = results.pivot(
    index="feature",
    columns="regime",
    values="pearson",
)

print(
    pearson_table.round(4).to_string()
)


print("\n" + "=" * 78)
print("SPEARMAN CORRELATION BY REGIME")
print("=" * 78)

spearman_table = results.pivot(
    index="feature",
    columns="regime",
    values="spearman",
)

print(
    spearman_table.round(4).to_string()
)


# ---------------------------------------------------------------------
# Print mutual information
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("MUTUAL INFORMATION BY REGIME")
print("=" * 78)

mi_table = results.pivot(
    index="feature",
    columns="regime",
    values="mutual_information",
)

print(
    mi_table.round(6).to_string()
)


# ---------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------
#
# We use year-to-year Pearson and Spearman variation as a simple
# descriptive stability diagnostic.
#
# For MI we calculate its range across years.
# ---------------------------------------------------------------------

year_results = results[
    results["regime"].isin(
        [str(y) for y in train["year"].unique()]
    )
].copy()


stability_rows = []


for feature in feature_map.keys():

    subset = year_results[
        year_results["feature"] == feature
    ]

    pearson_values = subset[
        "pearson"
    ].to_numpy()

    spearman_values = subset[
        "spearman"
    ].to_numpy()

    mi_values = subset[
        "mutual_information"
    ].to_numpy()

    stability_rows.append(
        {
            "feature": feature,

            "pearson_min": np.nanmin(
                pearson_values
            ),

            "pearson_max": np.nanmax(
                pearson_values
            ),

            "pearson_range": (
                np.nanmax(pearson_values)
                - np.nanmin(pearson_values)
            ),

            "spearman_min": np.nanmin(
                spearman_values
            ),

            "spearman_max": np.nanmax(
                spearman_values
            ),

            "spearman_range": (
                np.nanmax(spearman_values)
                - np.nanmin(spearman_values)
            ),

            "mi_min": np.nanmin(
                mi_values
            ),

            "mi_max": np.nanmax(
                mi_values
            ),

            "mi_range": (
                np.nanmax(mi_values)
                - np.nanmin(mi_values)
            ),
        }
    )


stability = pd.DataFrame(
    stability_rows
)


print("\n" + "=" * 78)
print("YEAR-TO-YEAR STABILITY SUMMARY")
print("=" * 78)

print(
    stability.round(6).to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Sign consistency
# ---------------------------------------------------------------------

sign_rows = []

for feature in feature_map.keys():

    subset = year_results[
        year_results["feature"] == feature
    ]

    pearson_values = subset[
        "pearson"
    ].dropna().to_numpy()

    positive = np.sum(
        pearson_values > 0
    )

    negative = np.sum(
        pearson_values < 0
    )

    if positive == len(
        pearson_values
    ):
        consistency = "POSITIVE_ALL"

    elif negative == len(
        pearson_values
    ):
        consistency = "NEGATIVE_ALL"

    else:
        consistency = "SIGN_CHANGE"

    sign_rows.append(
        {
            "feature": feature,
            "positive_years": positive,
            "negative_years": negative,
            "pearson_sign_consistency": consistency,
        }
    )


sign_df = pd.DataFrame(
    sign_rows
)


print("\n" + "=" * 78)
print("PEARSON SIGN CONSISTENCY")
print("=" * 78)

print(
    sign_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Target and exposure descriptive statistics by year
# ---------------------------------------------------------------------

yearly_descriptive = (
    train.groupby("year")
    .agg(
        n_days=(
            "target_date",
            "size",
        ),

        mean_claims=(
            target_col,
            "mean",
        ),

        std_claims=(
            target_col,
            "std",
        ),

        mean_active_policies=(
            "active_policy_count_t",
            "mean",
        ),

        std_active_policies=(
            "active_policy_count_t",
            "std",
        ),
    )
    .reset_index()
)


print("\n" + "=" * 78)
print("YEARLY TARGET / EXPOSURE DESCRIPTIVE STATISTICS")
print("=" * 78)

print(
    yearly_descriptive.round(4).to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

results.to_csv(
    RESULTS_DIR /
    "02_11_regime_feature_relevance.csv",
    index=False,
)

stability.to_csv(
    RESULTS_DIR /
    "02_11_regime_stability_summary.csv",
    index=False,
)

sign_df.to_csv(
    RESULTS_DIR /
    "02_11_regime_sign_consistency.csv",
    index=False,
)

yearly_descriptive.to_csv(
    RESULTS_DIR /
    "02_11_yearly_descriptive.csv",
    index=False,
)


print("\nSaved:")
print(
    " results/"
    "02_11_regime_feature_relevance.csv"
)
print(
    " results/"
    "02_11_regime_stability_summary.csv"
)
print(
    " results/"
    "02_11_regime_sign_consistency.csv"
)
print(
    " results/"
    "02_11_yearly_descriptive.csv"
)

print(
    "\nStep 11 analysis finished."
)