from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from db_config import engine


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RATE_SCALE = 100_000


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

df["active_policy_count_t"] = pd.to_numeric(
    df["active_policy_count_t"], errors="coerce"
)

df["target_property_damage_claim_count"] = pd.to_numeric(
    df["target_property_damage_claim_count"], errors="coerce"
)

df["is_public_holiday_t_plus_1"] = (
    df["is_public_holiday_t_plus_1"]
    .astype(int)
)

df = df.dropna(
    subset=[
        "target_date",
        "active_policy_count_t",
        "target_property_damage_claim_count",
        "day_of_week_t_plus_1",
        "is_public_holiday_t_plus_1",
    ]
).copy()

df = df[df["active_policy_count_t"] > 0].copy()

df = df.sort_values("target_date").reset_index(drop=True)


# ---------------------------------------------------------------------
# Basic information
# ---------------------------------------------------------------------

print("=" * 75)
print("WEEK 2 - STEP 8: EXPOSURE ANALYSIS")
print("=" * 75)

print(f"\nNumber of samples: {len(df)}")
print(
    f"Date range: {df['target_date'].min().date()} "
    f"to {df['target_date'].max().date()}"
)

print("\nSplit counts:")
print(df["split"].value_counts().sort_index())


# ---------------------------------------------------------------------
# Daily claim rate per 100,000 active policies
# ---------------------------------------------------------------------

df["claim_rate_per_100k"] = (
    df["target_property_damage_claim_count"]
    / df["active_policy_count_t"]
    * RATE_SCALE
)

print("\n" + "=" * 75)
print("ACTIVE POLICY COUNT")
print("=" * 75)

print(df["active_policy_count_t"].describe())

print("\n" + "=" * 75)
print("DAILY CLAIM RATE PER 100,000 ACTIVE POLICIES")
print("=" * 75)

print(df["claim_rate_per_100k"].describe())


# ---------------------------------------------------------------------
# Yearly exposure-adjusted rates
# ---------------------------------------------------------------------

df["year"] = df["target_date"].dt.year

yearly = (
    df.groupby("year")
    .agg(
        n_days=("target_date", "size"),
        total_claims=("target_property_damage_claim_count", "sum"),
        total_policy_days=("active_policy_count_t", "sum"),
        mean_active_policies=("active_policy_count_t", "mean"),
    )
    .reset_index()
)

yearly["claims_per_100k_policy_days"] = (
    yearly["total_claims"]
    / yearly["total_policy_days"]
    * RATE_SCALE
)

print("\n" + "=" * 75)
print("YEARLY EXPOSURE-ADJUSTED CLAIM RATES")
print("=" * 75)

print(
    yearly[
        [
            "year",
            "n_days",
            "total_claims",
            "mean_active_policies",
            "claims_per_100k_policy_days",
        ]
    ].to_string(index=False)
)

yearly.to_csv(
    RESULTS_DIR / "02_08_yearly_exposure_rates.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Raw correlation: P_t versus C_(t+1)
# ---------------------------------------------------------------------

pearson = df[
    [
        "active_policy_count_t",
        "target_property_damage_claim_count",
    ]
].corr(method="pearson").iloc[0, 1]

spearman = df[
    [
        "active_policy_count_t",
        "target_property_damage_claim_count",
    ]
].corr(method="spearman").iloc[0, 1]

print("\n" + "=" * 75)
print("RAW ASSOCIATION")
print("=" * 75)

print(f"Pearson r   = {pearson:.6f}")
print(f"Spearman rho = {spearman:.6f}")


# ---------------------------------------------------------------------
# Prepare regression variables
# ---------------------------------------------------------------------

# Linear time trend.
df["time_index"] = np.arange(len(df), dtype=float)

# Scale time to approximately [0, 1].
if len(df) > 1:
    df["time_scaled"] = (
        df["time_index"] / df["time_index"].max()
    )
else:
    df["time_scaled"] = 0.0


# Standardize exposure so regression coefficients are numerically easier
# to interpret and compare.
p_mean = df["active_policy_count_t"].mean()
p_std = df["active_policy_count_t"].std()

df["P_scaled"] = (
    df["active_policy_count_t"] - p_mean
) / p_std


# Day-of-week dummy variables.
weekday = pd.get_dummies(
    df["day_of_week_t_plus_1"].astype(str),
    prefix="dow",
    drop_first=True,
    dtype=float,
)

holiday = df[
    ["is_public_holiday_t_plus_1"]
].astype(float)

P = df[["P_scaled"]].astype(float)
time = df[["time_scaled"]].astype(float)

y = df["target_property_damage_claim_count"].astype(float)


# ---------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------

def fit_model(name, X):
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X).fit()

    return {
        "model": name,
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "n_parameters": len(model.params),
        "fitted_model": model,
    }


# ---------------------------------------------------------------------
# Regression models
# ---------------------------------------------------------------------

calendar = pd.concat(
    [
        weekday,
        holiday,
    ],
    axis=1,
)

calendar_P = pd.concat(
    [
        weekday,
        holiday,
        P,
    ],
    axis=1,
)

calendar_time = pd.concat(
    [
        weekday,
        holiday,
        time,
    ],
    axis=1,
)

calendar_time_P = pd.concat(
    [
        weekday,
        holiday,
        time,
        P,
    ],
    axis=1,
)


models = []

models.append(
    fit_model(
        "Exposure only",
        P,
    )
)

models.append(
    fit_model(
        "Calendar",
        calendar,
    )
)

models.append(
    fit_model(
        "Calendar + P_t",
        calendar_P,
    )
)

models.append(
    fit_model(
        "Calendar + time",
        calendar_time,
    )
)

models.append(
    fit_model(
        "Calendar + time + P_t",
        calendar_time_P,
    )
)


# ---------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------

comparison = pd.DataFrame(
    [
        {
            key: value
            for key, value in model.items()
            if key != "fitted_model"
        }
        for model in models
    ]
)

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(comparison.to_string(index=False))

comparison.to_csv(
    RESULTS_DIR / "02_08_exposure_model_comparison.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Incremental explanatory value of P_t
# ---------------------------------------------------------------------

results_by_name = {
    item["model"]: item
    for item in models
}

delta_calendar = (
    results_by_name["Calendar + P_t"]["r_squared"]
    - results_by_name["Calendar"]["r_squared"]
)

delta_calendar_time = (
    results_by_name["Calendar + time + P_t"]["r_squared"]
    - results_by_name["Calendar + time"]["r_squared"]
)

print("\n" + "=" * 75)
print("INCREMENTAL VALUE OF P_t")
print("=" * 75)

print(
    "Delta R^2: Calendar -> Calendar + P_t"
    f" = {delta_calendar:.6f}"
)

print(
    "Delta R^2: Calendar + time -> Calendar + time + P_t"
    f" = {delta_calendar_time:.6f}"
)


# ---------------------------------------------------------------------
# Residual association after calendar + time removal
# ---------------------------------------------------------------------

base_model = results_by_name[
    "Calendar + time"
]["fitted_model"]

claim_residuals = base_model.resid


# Remove calendar + time structure from P_t as well.
X_for_P = sm.add_constant(
    calendar_time,
    has_constant="add",
)

P_model = sm.OLS(
    df["P_scaled"].astype(float),
    X_for_P,
).fit()

P_residuals = P_model.resid


residual_pearson = pd.Series(
    claim_residuals
).corr(
    pd.Series(P_residuals),
    method="pearson",
)

residual_spearman = pd.Series(
    claim_residuals
).corr(
    pd.Series(P_residuals),
    method="spearman",
)


print("\n" + "=" * 75)
print("PARTIAL / RESIDUAL ASSOCIATION")
print("=" * 75)

print(
    "After removing weekday + holiday + linear time:"
)

print(
    f"Residual Pearson r   = {residual_pearson:.6f}"
)

print(
    f"Residual Spearman rho = {residual_spearman:.6f}"
)


# ---------------------------------------------------------------------
# Exposure coefficient in the full model
# ---------------------------------------------------------------------

full_model = results_by_name[
    "Calendar + time + P_t"
]["fitted_model"]

print("\n" + "=" * 75)
print("P_t COEFFICIENT IN FULL MODEL")
print("=" * 75)

if "P_scaled" in full_model.params.index:
    print(
        f"Coefficient = "
        f"{full_model.params['P_scaled']:.6f}"
    )

    print(
        f"p-value     = "
        f"{full_model.pvalues['P_scaled']:.6g}"
    )


# ---------------------------------------------------------------------
# Save numerical summary
# ---------------------------------------------------------------------

summary = pd.DataFrame(
    {
        "metric": [
            "pearson_P_vs_target",
            "spearman_P_vs_target",
            "delta_r2_calendar_plus_P",
            "delta_r2_calendar_time_plus_P",
            "residual_pearson",
            "residual_spearman",
            "P_coefficient_full_model",
            "P_pvalue_full_model",
        ],
        "value": [
            pearson,
            spearman,
            delta_calendar,
            delta_calendar_time,
            residual_pearson,
            residual_spearman,
            full_model.params.get("P_scaled", np.nan),
            full_model.pvalues.get("P_scaled", np.nan),
        ],
    }
)

summary.to_csv(
    RESULTS_DIR / "02_08_exposure_summary.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Plot 1: exposure versus target
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["active_policy_count_t"],
    df["target_property_damage_claim_count"],
    alpha=0.35,
)

plt.xlabel("Active policy count P_t")
plt.ylabel("Next-day property-damage claims C_(t+1)")
plt.title("Exposure versus next-day claim count")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "02_08_exposure_vs_claims.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------
# Plot 2: exposure over time
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df["target_date"],
    df["active_policy_count_t"],
)

plt.xlabel("Date")
plt.ylabel("Active policy count P_t")
plt.title("Active policy exposure over time")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "02_08_exposure_over_time.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------------------
# Plot 3: residual exposure relationship
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    P_residuals,
    claim_residuals,
    alpha=0.35,
)

plt.xlabel(
    "Residual P_t after calendar + time removal"
)

plt.ylabel(
    "Residual C_(t+1) after calendar + time removal"
)

plt.title(
    "Exposure information beyond calendar and time"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "02_08_exposure_residuals.png",
    dpi=200,
)

plt.close()


print("\nSaved:")
print(" results/02_08_yearly_exposure_rates.csv")
print(" results/02_08_exposure_model_comparison.csv")
print(" results/02_08_exposure_summary.csv")
print(" results/02_08_exposure_vs_claims.png")
print(" results/02_08_exposure_over_time.png")
print(" results/02_08_exposure_residuals.png")

print("\nStep 8 analysis finished.")