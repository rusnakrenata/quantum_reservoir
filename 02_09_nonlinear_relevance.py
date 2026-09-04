from pathlib import Path

import matplotlib.pyplot as plt
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
N_PERMUTATIONS = 500


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
# Use TRAINING SET ONLY for feature relevance decisions
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
    train[col] = pd.to_numeric(train[col], errors="coerce")

train = train.dropna(subset=required_columns).copy()


print("=" * 78)
print("WEEK 2 - STEP 9: NONLINEAR FEATURE RELEVANCE")
print("=" * 78)

print(f"\nTraining samples: {len(train)}")

print(
    f"Training period: "
    f"{train['target_date'].min().date()} "
    f"to {train['target_date'].max().date()}"
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

# sklearn needs to know which variables are discrete.
#
# C_t  : treated as continuous/count-valued here
# D_t+1: discrete
# P_t  : continuous
# H_t+1: discrete

discrete_map = {
    "C_t": True,
    "D_t+1": True,
    "P_t": False,
    "H_t+1": True,
}

target_col = "target_property_damage_claim_count"

y = train[target_col].to_numpy(dtype=float)


# ---------------------------------------------------------------------
# Mutual-information helper
# ---------------------------------------------------------------------

def calculate_mi(x, y, discrete, random_state):
    x = np.asarray(x).reshape(-1, 1)

    mi = mutual_info_regression(
        x,
        y,
        discrete_features=[discrete],
        random_state=random_state,
    )

    return float(mi[0])


# ---------------------------------------------------------------------
# Observed relevance
# ---------------------------------------------------------------------

results = []

rng = np.random.default_rng(RANDOM_STATE)


for feature_name, column in feature_map.items():

    x = train[column].to_numpy(dtype=float)

    # -------------------------------------------------------------
    # Pearson and Spearman
    # -------------------------------------------------------------

    pearson = pd.Series(x).corr(
        pd.Series(y),
        method="pearson",
    )

    spearman = pd.Series(x).corr(
        pd.Series(y),
        method="spearman",
    )

    # -------------------------------------------------------------
    # Observed mutual information
    # -------------------------------------------------------------

    observed_mi = calculate_mi(
        x=x,
        y=y,
        discrete=discrete_map[feature_name],
        random_state=RANDOM_STATE,
    )

    # -------------------------------------------------------------
    # Permutation baseline
    # -------------------------------------------------------------

    permutation_mi = []

    for i in range(N_PERMUTATIONS):

        y_perm = rng.permutation(y)

        mi_perm = calculate_mi(
            x=x,
            y=y_perm,
            discrete=discrete_map[feature_name],
            random_state=RANDOM_STATE + i + 1,
        )

        permutation_mi.append(mi_perm)

    permutation_mi = np.asarray(permutation_mi)

    perm_mean = permutation_mi.mean()
    perm_std = permutation_mi.std()

    perm_95 = np.quantile(
        permutation_mi,
        0.95,
    )

    # Empirical one-sided permutation p-value
    permutation_p = (
        np.sum(permutation_mi >= observed_mi) + 1
    ) / (
        N_PERMUTATIONS + 1
    )

    mi_above_baseline = observed_mi - perm_mean

    results.append(
        {
            "feature": feature_name,
            "column": column,
            "pearson": pearson,
            "spearman": spearman,
            "mutual_information": observed_mi,
            "permutation_mean": perm_mean,
            "permutation_std": perm_std,
            "permutation_95pct": perm_95,
            "mi_above_baseline": mi_above_baseline,
            "permutation_pvalue": permutation_p,
        }
    )


# ---------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mutual_information",
    ascending=False,
).reset_index(drop=True)


print("\n" + "=" * 78)
print("NONLINEAR RELEVANCE RESULTS")
print("=" * 78)

print(
    results_df[
        [
            "feature",
            "pearson",
            "spearman",
            "mutual_information",
            "permutation_mean",
            "permutation_95pct",
            "mi_above_baseline",
            "permutation_pvalue",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------------------
# Simple relevance classification
# ---------------------------------------------------------------------

results_df["above_95pct_permutation"] = (
    results_df["mutual_information"]
    >
    results_df["permutation_95pct"]
)


print("\n" + "=" * 78)
print("PERMUTATION TEST")
print("=" * 78)

for _, row in results_df.iterrows():

    status = (
        "YES"
        if row["above_95pct_permutation"]
        else "NO"
    )

    print(
        f"{row['feature']:6s}: "
        f"MI = {row['mutual_information']:.6f}, "
        f"95% random threshold = "
        f"{row['permutation_95pct']:.6f}, "
        f"above threshold = {status}, "
        f"p = {row['permutation_pvalue']:.6f}"
    )


# ---------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------

results_df.to_csv(
    RESULTS_DIR / "02_09_nonlinear_relevance.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Plot: observed MI versus random 95% threshold
# ---------------------------------------------------------------------

plot_df = results_df.sort_values(
    "mutual_information",
    ascending=True,
)

y_pos = np.arange(len(plot_df))


plt.figure(figsize=(8, 5))

plt.barh(
    y_pos,
    plot_df["mutual_information"],
)

plt.scatter(
    plot_df["permutation_95pct"],
    y_pos,
    marker="x",
    s=70,
    label="95% permutation threshold",
)

plt.yticks(
    y_pos,
    plot_df["feature"],
)

plt.xlabel("Mutual information")
plt.ylabel("Feature")

plt.title(
    "Nonlinear relevance to next-day claim count\n"
    "(training set only)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "02_09_mutual_information.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------
# Plot: linear correlation versus mutual information
# ---------------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    np.abs(results_df["pearson"]),
    results_df["mutual_information"],
    s=80,
)

for _, row in results_df.iterrows():

    plt.annotate(
        row["feature"],
        (
            abs(row["pearson"]),
            row["mutual_information"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.xlabel("|Pearson correlation|")
plt.ylabel("Mutual information")

plt.title(
    "Linear versus nonlinear feature relevance"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "02_09_correlation_vs_mi.png",
    dpi=200,
)

plt.close()


print("\nSaved:")
print(" results/02_09_nonlinear_relevance.csv")
print(" results/02_09_mutual_information.png")
print(" results/02_09_correlation_vs_mi.png")

print("\nStep 9 analysis finished.")