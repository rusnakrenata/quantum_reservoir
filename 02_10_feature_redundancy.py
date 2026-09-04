from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from db_config import engine


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


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

feature_columns = [
    "property_damage_claim_count_t",
    "active_policy_count_t",
    "day_of_week_t_plus_1",
    "is_public_holiday_t_plus_1",
]

for col in feature_columns:
    train[col] = pd.to_numeric(
        train[col],
        errors="coerce",
    )

train = train.dropna(
    subset=feature_columns
).copy()


print("=" * 78)
print("WEEK 2 - STEP 10: FEATURE REDUNDANCY")
print("=" * 78)

print(f"\nTraining samples: {len(train)}")

print(
    f"Training period: "
    f"{train['target_date'].min().date()} "
    f"to {train['target_date'].max().date()}"
)


# ---------------------------------------------------------------------
# Friendly feature names
# ---------------------------------------------------------------------

rename_map = {
    "property_damage_claim_count_t": "C_t",
    "active_policy_count_t": "P_t",
    "day_of_week_t_plus_1": "D_t+1",
    "is_public_holiday_t_plus_1": "H_t+1",
}

feature_df = train[
    feature_columns
].rename(
    columns=rename_map
)


# ---------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------

pearson_corr = feature_df.corr(
    method="pearson"
)

print("\n" + "=" * 78)
print("PEARSON FEATURE-FEATURE CORRELATION")
print("=" * 78)

print(
    pearson_corr.round(6).to_string()
)


pearson_corr.to_csv(
    RESULTS_DIR /
    "02_10_feature_pearson_correlation.csv"
)


# ---------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------

spearman_corr = feature_df.corr(
    method="spearman"
)

print("\n" + "=" * 78)
print("SPEARMAN FEATURE-FEATURE CORRELATION")
print("=" * 78)

print(
    spearman_corr.round(6).to_string()
)


spearman_corr.to_csv(
    RESULTS_DIR /
    "02_10_feature_spearman_correlation.csv"
)


# ---------------------------------------------------------------------
# Maximum absolute pairwise correlation
# ---------------------------------------------------------------------

features = list(feature_df.columns)

pairwise_rows = []

for i in range(len(features)):
    for j in range(i + 1, len(features)):

        f1 = features[i]
        f2 = features[j]

        pairwise_rows.append(
            {
                "feature_1": f1,
                "feature_2": f2,
                "pearson": pearson_corr.loc[f1, f2],
                "abs_pearson": abs(
                    pearson_corr.loc[f1, f2]
                ),
                "spearman": spearman_corr.loc[f1, f2],
                "abs_spearman": abs(
                    spearman_corr.loc[f1, f2]
                ),
            }
        )


pairwise = pd.DataFrame(pairwise_rows)

pairwise = pairwise.sort_values(
    "abs_spearman",
    ascending=False,
).reset_index(drop=True)


print("\n" + "=" * 78)
print("PAIRWISE REDUNDANCY RANKING")
print("=" * 78)

print(
    pairwise.to_string(
        index=False
    )
)


pairwise.to_csv(
    RESULTS_DIR /
    "02_10_pairwise_redundancy.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Day-of-week / holiday contingency table
# ---------------------------------------------------------------------
#
# D_t+1 and H_t+1 are categorical variables.
# Their raw Pearson correlation is not the most meaningful measure.
# We therefore inspect their contingency table separately.
# ---------------------------------------------------------------------

contingency = pd.crosstab(
    train["day_of_week_t_plus_1"],
    train["is_public_holiday_t_plus_1"],
)

print("\n" + "=" * 78)
print("DAY-OF-WEEK / HOLIDAY CONTINGENCY TABLE")
print("=" * 78)

print(contingency.to_string())


contingency.to_csv(
    RESULTS_DIR /
    "02_10_dow_holiday_contingency.csv"
)


# ---------------------------------------------------------------------
# Cramer's V for D_t+1 versus H_t+1
# ---------------------------------------------------------------------

def cramers_v(table):

    observed = table.to_numpy(dtype=float)

    n = observed.sum()

    row_totals = observed.sum(
        axis=1,
        keepdims=True,
    )

    col_totals = observed.sum(
        axis=0,
        keepdims=True,
    )

    expected = (
        row_totals @ col_totals
    ) / n

    valid = expected > 0

    chi2 = np.sum(
        (
            observed[valid]
            - expected[valid]
        ) ** 2
        / expected[valid]
    )

    r, k = observed.shape

    denominator = n * min(
        r - 1,
        k - 1,
    )

    if denominator <= 0:
        return np.nan

    return np.sqrt(
        chi2 / denominator
    )


cramers_v_dow_holiday = cramers_v(
    contingency
)


print(
    "\nCramer's V "
    f"(D_t+1 vs H_t+1) = "
    f"{cramers_v_dow_holiday:.6f}"
)


# ---------------------------------------------------------------------
# Proper design matrix for VIF
# ---------------------------------------------------------------------
#
# Important:
# D_t+1 is categorical.
#
# We therefore must NOT treat values 0,1,...,6 as a continuous
# variable for VIF.
#
# Instead we one-hot encode weekday and drop one reference category.
# ---------------------------------------------------------------------

weekday_dummies = pd.get_dummies(
    train[
        "day_of_week_t_plus_1"
    ].astype(int).astype(str),
    prefix="D",
    drop_first=True,
    dtype=float,
)


# Standardize C_t and P_t.
#
# Standardization is not mathematically required for VIF,
# but makes the design matrix easier to inspect.
# ---------------------------------------------------------------------

C = train[
    "property_damage_claim_count_t"
].astype(float)

P = train[
    "active_policy_count_t"
].astype(float)


C_scaled = (
    C - C.mean()
) / C.std()

P_scaled = (
    P - P.mean()
) / P.std()


X_vif = pd.DataFrame(
    {
        "C_t": C_scaled,
        "P_t": P_scaled,
        "H_t+1": train[
            "is_public_holiday_t_plus_1"
        ].astype(float),
    },
    index=train.index,
)


X_vif = pd.concat(
    [
        X_vif,
        weekday_dummies.set_axis(
            train.index
        ),
    ],
    axis=1,
)


# Add intercept.
X_vif = sm.add_constant(
    X_vif,
    has_constant="add",
)


# ---------------------------------------------------------------------
# Calculate VIF
# ---------------------------------------------------------------------

vif_rows = []

for i, column in enumerate(
    X_vif.columns
):

    vif_value = variance_inflation_factor(
        X_vif.values,
        i,
    )

    vif_rows.append(
        {
            "variable": column,
            "vif": vif_value,
        }
    )


vif_df = pd.DataFrame(
    vif_rows
)

# We do not interpret the intercept VIF.
vif_predictors = vif_df[
    vif_df["variable"] != "const"
].copy()


# ---------------------------------------------------------------------
# VIF classification
# ---------------------------------------------------------------------

def classify_vif(vif):

    if vif < 5:
        return "LOW"

    if vif < 10:
        return "MODERATE"

    return "HIGH"


vif_predictors["redundancy"] = (
    vif_predictors["vif"]
    .apply(classify_vif)
)


vif_predictors = (
    vif_predictors
    .sort_values(
        "vif",
        ascending=False,
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 78)
print("VARIANCE INFLATION FACTORS")
print("=" * 78)

print(
    vif_predictors.to_string(
        index=False
    )
)


vif_predictors.to_csv(
    RESULTS_DIR /
    "02_10_vif.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Aggregate weekday dummy VIF
# ---------------------------------------------------------------------

weekday_vifs = vif_predictors[
    vif_predictors[
        "variable"
    ].str.startswith("D_")
]


if len(weekday_vifs) > 0:

    max_weekday_vif = (
        weekday_vifs["vif"].max()
    )

else:

    max_weekday_vif = np.nan


summary = pd.DataFrame(
    {
        "metric": [
            "max_abs_pairwise_pearson",
            "max_abs_pairwise_spearman",
            "cramers_v_D_vs_H",
            "vif_C_t",
            "vif_P_t",
            "vif_H_t+1",
            "max_weekday_dummy_vif",
        ],
        "value": [
            pairwise[
                "abs_pearson"
            ].max(),
            pairwise[
                "abs_spearman"
            ].max(),
            cramers_v_dow_holiday,
            vif_predictors.loc[
                vif_predictors[
                    "variable"
                ] == "C_t",
                "vif",
            ].iloc[0],
            vif_predictors.loc[
                vif_predictors[
                    "variable"
                ] == "P_t",
                "vif",
            ].iloc[0],
            vif_predictors.loc[
                vif_predictors[
                    "variable"
                ] == "H_t+1",
                "vif",
            ].iloc[0],
            max_weekday_vif,
        ],
    }
)


print("\n" + "=" * 78)
print("REDUNDANCY SUMMARY")
print("=" * 78)

print(
    summary.to_string(
        index=False
    )
)


summary.to_csv(
    RESULTS_DIR /
    "02_10_redundancy_summary.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(6, 5)
)

matrix = spearman_corr.to_numpy()

image = ax.imshow(
    matrix,
    vmin=-1,
    vmax=1,
)

ax.set_xticks(
    np.arange(
        len(features)
    )
)

ax.set_yticks(
    np.arange(
        len(features)
    )
)

ax.set_xticklabels(
    features
)

ax.set_yticklabels(
    features
)


for i in range(
    len(features)
):

    for j in range(
        len(features)
    ):

        ax.text(
            j,
            i,
            f"{matrix[i, j]:.2f}",
            ha="center",
            va="center",
        )


ax.set_title(
    "Training-set feature redundancy\n"
    "Spearman correlation"
)

fig.colorbar(
    image,
    ax=ax,
    label="Spearman correlation",
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "02_10_feature_redundancy.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------

print("\nSaved:")
print(
    " results/"
    "02_10_feature_pearson_correlation.csv"
)
print(
    " results/"
    "02_10_feature_spearman_correlation.csv"
)
print(
    " results/"
    "02_10_pairwise_redundancy.csv"
)
print(
    " results/"
    "02_10_dow_holiday_contingency.csv"
)
print(
    " results/"
    "02_10_vif.csv"
)
print(
    " results/"
    "02_10_redundancy_summary.csv"
)
print(
    " results/"
    "02_10_feature_redundancy.png"
)

print(
    "\nStep 10 analysis finished."
)