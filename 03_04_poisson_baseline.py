from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

INPUT_FILE = RESULTS_DIR / "03_01_preprocessed_samples.csv"

SEASONAL_NAIVE_RMSE = 6.405691
RIDGE_F4_RMSE = 4.874844


# ---------------------------------------------------------------------
# Load preprocessed samples
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["target_date"] = pd.to_datetime(
    df["target_date"]
)

train = df[
    df["split"] == "train"
].copy()

validation = df[
    df["split"] == "validation"
].copy()


print("=" * 78)
print("WEEK 3 - STEP 4: POISSON REGRESSION")
print("=" * 78)

print(f"\nTraining samples:   {len(train)}")
print(f"Validation samples: {len(validation)}")


# ---------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------

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


TARGET = "target_property_damage_claim_count"


# ---------------------------------------------------------------------
# Training target standard deviation
# ---------------------------------------------------------------------

train_target_std = train[
    TARGET
].std(ddof=0)


# ---------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true,
    y_pred,
):

    errors = (
        y_true
        - y_pred
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    nrmse = (
        rmse
        / train_target_std
    )

    bias = np.mean(
        y_pred
        - y_true
    )

    return (
        mae,
        rmse,
        nrmse,
        bias,
    )


# ---------------------------------------------------------------------
# Poisson deviance
# ---------------------------------------------------------------------

def poisson_deviance(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    # Numerical protection
    y_pred = np.clip(
        y_pred,
        1e-12,
        None,
    )

    positive = (
        y_true > 0
    )

    terms = np.zeros_like(
        y_true,
        dtype=float,
    )

    terms[positive] = (
        y_true[positive]
        * np.log(
            y_true[positive]
            / y_pred[positive]
        )
        - (
            y_true[positive]
            - y_pred[positive]
        )
    )

    terms[~positive] = (
        y_pred[~positive]
    )

    return (
        2.0
        * np.sum(terms)
    )


# ---------------------------------------------------------------------
# Fit each feature set
# ---------------------------------------------------------------------

result_rows = []
coefficient_rows = []
prediction_rows = []


for feature_set_name, feature_columns in FEATURE_SETS.items():

    print("\n" + "=" * 78)
    print(
        f"FEATURE SET {feature_set_name}"
    )
    print("=" * 78)

    print(
        "Features: "
        + ", ".join(feature_columns)
    )


    # -------------------------------------------------------------
    # Design matrices
    # -------------------------------------------------------------

    X_train = train[
        feature_columns
    ].astype(float)

    X_val = validation[
        feature_columns
    ].astype(float)

    # Add intercept
    X_train = sm.add_constant(
        X_train,
        has_constant="add",
    )

    X_val = sm.add_constant(
        X_val,
        has_constant="add",
    )


    y_train = train[
        TARGET
    ].to_numpy(dtype=float)

    y_val = validation[
        TARGET
    ].to_numpy(dtype=float)


    # -------------------------------------------------------------
    # Poisson GLM
    # -------------------------------------------------------------

    model = sm.GLM(
        y_train,
        X_train,
        family=sm.families.Poisson(),
    )

    fitted_model = model.fit()


    # -------------------------------------------------------------
    # Validation predictions
    # -------------------------------------------------------------

    y_pred = fitted_model.predict(
        X_val
    )


    (
        mae,
        rmse,
        nrmse,
        bias,
    ) = calculate_metrics(
        y_val,
        y_pred,
    )


    deviance = poisson_deviance(
        y_val,
        y_pred,
    )

    mean_deviance = (
        deviance
        / len(y_val)
    )


    improvement_vs_seasonal = (
        100.0
        * (
            SEASONAL_NAIVE_RMSE
            - rmse
        )
        / SEASONAL_NAIVE_RMSE
    )


    improvement_vs_ridge_f4 = (
        100.0
        * (
            RIDGE_F4_RMSE
            - rmse
        )
        / RIDGE_F4_RMSE
    )


    result_rows.append(
        {
            "model":
                "Poisson",

            "feature_set":
                feature_set_name,

            "validation_mae":
                mae,

            "validation_rmse":
                rmse,

            "validation_nrmse":
                nrmse,

            "validation_bias":
                bias,

            "poisson_deviance":
                deviance,

            "mean_poisson_deviance":
                mean_deviance,

            "improvement_vs_seasonal_pct":
                improvement_vs_seasonal,

            "improvement_vs_ridge_F4_pct":
                improvement_vs_ridge_f4,
        }
    )


    # -------------------------------------------------------------
    # Coefficients
    # -------------------------------------------------------------

    for variable in fitted_model.params.index:

        beta = fitted_model.params[
            variable
        ]

        coefficient_rows.append(
            {
                "feature_set":
                    feature_set_name,

                "variable":
                    variable,

                "coefficient_beta":
                    beta,

                "exp_beta":
                    np.exp(beta),
            }
        )


    # -------------------------------------------------------------
    # Store predictions
    # -------------------------------------------------------------

    for date, actual, pred in zip(
        validation["target_date"],
        y_val,
        y_pred,
    ):

        prediction_rows.append(
            {
                "target_date":
                    date,

                "feature_set":
                    feature_set_name,

                "actual":
                    actual,

                "prediction":
                    pred,
            }
        )


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

results_df = pd.DataFrame(
    result_rows
)

results_df = (
    results_df
    .sort_values(
        "validation_rmse"
    )
    .reset_index(drop=True)
)


coefficients_df = pd.DataFrame(
    coefficient_rows
)

predictions_df = pd.DataFrame(
    prediction_rows
)


# ---------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("POISSON VALIDATION RESULTS")
print("=" * 78)

print(
    results_df.round(6)
    .to_string(index=False)
)


print("\n" + "=" * 78)
print("POISSON COEFFICIENTS")
print("=" * 78)

print(
    coefficients_df.round(6)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Prediction range check
# ---------------------------------------------------------------------

prediction_summary = (
    predictions_df
    .groupby("feature_set")
    .agg(
        prediction_min=(
            "prediction",
            "min",
        ),

        prediction_mean=(
            "prediction",
            "mean",
        ),

        prediction_max=(
            "prediction",
            "max",
        ),
    )
    .reset_index()
)


print("\n" + "=" * 78)
print("PREDICTION RANGE")
print("=" * 78)

print(
    prediction_summary.round(6)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

results_df.to_csv(
    RESULTS_DIR /
    "03_04_poisson_validation_metrics.csv",
    index=False,
)


coefficients_df.to_csv(
    RESULTS_DIR /
    "03_04_poisson_coefficients.csv",
    index=False,
)


predictions_df.to_csv(
    RESULTS_DIR /
    "03_04_poisson_validation_predictions.csv",
    index=False,
)


print("\nSaved:")
print(
    " results/"
    "03_04_poisson_validation_metrics.csv"
)

print(
    " results/"
    "03_04_poisson_coefficients.csv"
)

print(
    " results/"
    "03_04_poisson_validation_predictions.csv"
)


print(
    "\nStep 4 Poisson analysis finished."
)