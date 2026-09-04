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
# Load forecasting data
# ---------------------------------------------------------------------

query = """
SELECT
    input_date,
    target_date,
    property_damage_claim_count_t,
    target_property_damage_claim_count,
    split
FROM qrc_forecast_sample
ORDER BY input_date
"""

df = pd.read_sql(query, engine)

df["input_date"] = pd.to_datetime(df["input_date"])
df["target_date"] = pd.to_datetime(df["target_date"])

df["property_damage_claim_count_t"] = pd.to_numeric(
    df["property_damage_claim_count_t"],
    errors="coerce",
)

df["target_property_damage_claim_count"] = pd.to_numeric(
    df["target_property_damage_claim_count"],
    errors="coerce",
)


# ---------------------------------------------------------------------
# Train / validation
# ---------------------------------------------------------------------

train = df[
    df["split"] == "train"
].copy()

validation = df[
    df["split"] == "validation"
].copy()


print("=" * 78)
print("WEEK 3 - STEP 2: NAIVE BASELINES")
print("=" * 78)

print(f"\nTraining samples:   {len(train)}")
print(f"Validation samples: {len(validation)}")


# ---------------------------------------------------------------------
# Training-only target statistics
# ---------------------------------------------------------------------

train_mean = train[
    "target_property_damage_claim_count"
].mean()

train_std = train[
    "target_property_damage_claim_count"
].std(ddof=0)


print("\n" + "=" * 78)
print("TRAINING TARGET PARAMETERS")
print("=" * 78)

print(
    f"Training mean: {train_mean:.6f}"
)

print(
    f"Training std:  {train_std:.6f}"
)


# ---------------------------------------------------------------------
# Actual validation target
# ---------------------------------------------------------------------

y_true = validation[
    "target_property_damage_claim_count"
].to_numpy(dtype=float)


# ---------------------------------------------------------------------
# Baseline 1: training mean
# ---------------------------------------------------------------------

y_pred_mean = np.full(
    len(validation),
    train_mean,
    dtype=float,
)


# ---------------------------------------------------------------------
# Baseline 2: persistence
# ---------------------------------------------------------------------

y_pred_persistence = validation[
    "property_damage_claim_count_t"
].to_numpy(dtype=float)


# ---------------------------------------------------------------------
# Baseline 3: seasonal naive
#
# Forecast:
#
# C_(t+1) = C_(t-6)
#
# For each current input date t, we therefore retrieve the observed
# claim count at date t-6.
# ---------------------------------------------------------------------

claim_by_date = (
    df.set_index("input_date")[
        "property_damage_claim_count_t"
    ]
    .to_dict()
)


seasonal_predictions = []

missing_seasonal = 0


for input_date in validation["input_date"]:

    reference_date = (
        input_date
        - pd.Timedelta(days=6)
    )

    prediction = claim_by_date.get(
        reference_date,
        np.nan,
    )

    if pd.isna(prediction):
        missing_seasonal += 1

    seasonal_predictions.append(
        prediction
    )


y_pred_seasonal = np.asarray(
    seasonal_predictions,
    dtype=float,
)


# ---------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------

def calculate_metrics(
    model_name,
    y_true,
    y_pred,
    train_std,
):

    valid = (
        np.isfinite(y_true)
        &
        np.isfinite(y_pred)
    )

    yt = y_true[valid]
    yp = y_pred[valid]

    errors = yt - yp

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    nrmse = (
        rmse / train_std
    )

    bias = np.mean(
        yp - yt
    )

    return {
        "model": model_name,
        "n": len(yt),
        "mae": mae,
        "rmse": rmse,
        "nrmse_train_std": nrmse,
        "bias_pred_minus_actual": bias,
    }


# ---------------------------------------------------------------------
# Evaluate baselines
# ---------------------------------------------------------------------

results = []

results.append(
    calculate_metrics(
        "training_mean",
        y_true,
        y_pred_mean,
        train_std,
    )
)

results.append(
    calculate_metrics(
        "persistence",
        y_true,
        y_pred_persistence,
        train_std,
    )
)

results.append(
    calculate_metrics(
        "seasonal_naive_7d",
        y_true,
        y_pred_seasonal,
        train_std,
    )
)


results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        "rmse"
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("VALIDATION BASELINE RESULTS")
print("=" * 78)

print(
    results_df.round(6)
    .to_string(index=False)
)


print("\n" + "=" * 78)
print("SEASONAL-NAIVE CHECK")
print("=" * 78)

print(
    f"Missing seasonal predictions: "
    f"{missing_seasonal}"
)


# ---------------------------------------------------------------------
# Example predictions
# ---------------------------------------------------------------------

example_df = pd.DataFrame(
    {
        "target_date":
            validation[
                "target_date"
            ].head(10),

        "actual":
            y_true[:10],

        "mean_prediction":
            y_pred_mean[:10],

        "persistence_prediction":
            y_pred_persistence[:10],

        "seasonal_naive_prediction":
            y_pred_seasonal[:10],
    }
)


print("\n" + "=" * 78)
print("EXAMPLE VALIDATION FORECASTS")
print("=" * 78)

print(
    example_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Save prediction-level results
# ---------------------------------------------------------------------

prediction_df = validation[
    [
        "input_date",
        "target_date",
        "target_property_damage_claim_count",
    ]
].copy()

prediction_df[
    "prediction_training_mean"
] = y_pred_mean

prediction_df[
    "prediction_persistence"
] = y_pred_persistence

prediction_df[
    "prediction_seasonal_naive_7d"
] = y_pred_seasonal


results_df.to_csv(
    RESULTS_DIR /
    "03_02_naive_baseline_metrics.csv",
    index=False,
)


prediction_df.to_csv(
    RESULTS_DIR /
    "03_02_naive_baseline_predictions.csv",
    index=False,
)


print("\nSaved:")
print(
    " results/"
    "03_02_naive_baseline_metrics.csv"
)
print(
    " results/"
    "03_02_naive_baseline_predictions.csv"
)

print(
    "\nStep 2 naive-baseline analysis finished."
)