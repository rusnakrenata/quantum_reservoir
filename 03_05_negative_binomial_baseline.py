from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import NegativeBinomial


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

INPUT_FILE = RESULTS_DIR / "03_01_preprocessed_samples.csv"

SEASONAL_NAIVE_RMSE = 6.405691
RIDGE_F4_RMSE = 4.874844
POISSON_F4_RMSE = 4.879339


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
print("WEEK 3 - STEP 5: NEGATIVE BINOMIAL REGRESSION")
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

    error = (
        y_true
        - y_pred
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
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
# Fit feature sets
# ---------------------------------------------------------------------

result_rows = []
coefficient_rows = []
prediction_rows = []


for feature_set_name, feature_columns in FEATURE_SETS.items():

    print("\n" + "=" * 78)
    print(f"FEATURE SET {feature_set_name}")
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
    # Negative Binomial NB2
    #
    # Variance:
    #
    # Var(Y | X) = mu + alpha * mu^2
    #
    # alpha is estimated from the training data.
    # -------------------------------------------------------------

    model = NegativeBinomial(
        y_train,
        X_train,
        loglike_method="nb2",
    )


    fitted_model = model.fit(
        method="bfgs",
        maxiter=1000,
        disp=False,
    )


    # -------------------------------------------------------------
    # Validation prediction
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


    # -------------------------------------------------------------
    # Extract estimated alpha
    # -------------------------------------------------------------

    if "alpha" in fitted_model.params.index:

        alpha = float(
            fitted_model.params[
                "alpha"
            ]
        )

    else:

        alpha = np.nan


    # -------------------------------------------------------------
    # Implied validation variance
    # -------------------------------------------------------------

    if np.isfinite(alpha):

        predicted_variance = (
            y_pred
            +
            alpha
            * y_pred ** 2
        )

        mean_predicted_variance = (
            np.mean(
                predicted_variance
            )
        )

    else:

        mean_predicted_variance = np.nan


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


    improvement_vs_poisson_f4 = (
        100.0
        * (
            POISSON_F4_RMSE
            - rmse
        )
        / POISSON_F4_RMSE
    )


    result_rows.append(
        {
            "model":
                "Negative Binomial",

            "feature_set":
                feature_set_name,

            "alpha":
                alpha,

            "validation_mae":
                mae,

            "validation_rmse":
                rmse,

            "validation_nrmse":
                nrmse,

            "validation_bias":
                bias,

            "mean_predicted_variance":
                mean_predicted_variance,

            "log_likelihood_train":
                fitted_model.llf,

            "aic_train":
                fitted_model.aic,

            "bic_train":
                fitted_model.bic,

            "improvement_vs_seasonal_pct":
                improvement_vs_seasonal,

            "improvement_vs_ridge_F4_pct":
                improvement_vs_ridge_f4,

            "improvement_vs_poisson_F4_pct":
                improvement_vs_poisson_f4,
        }
    )


    # -------------------------------------------------------------
    # Coefficients
    # -------------------------------------------------------------

    for variable in fitted_model.params.index:

        parameter = fitted_model.params[
            variable
        ]

        if variable == "alpha":

            exp_beta = np.nan

        else:

            exp_beta = np.exp(
                parameter
            )


        coefficient_rows.append(
            {
                "feature_set":
                    feature_set_name,

                "variable":
                    variable,

                "parameter":
                    parameter,

                "exp_beta":
                    exp_beta,
            }
        )


    # -------------------------------------------------------------
    # Predictions
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
# Results tables
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
print("NEGATIVE BINOMIAL VALIDATION RESULTS")
print("=" * 78)

print(
    results_df.round(6)
    .to_string(index=False)
)


print("\n" + "=" * 78)
print("NEGATIVE BINOMIAL PARAMETERS")
print("=" * 78)

print(
    coefficients_df.round(6)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Dispersion summary
# ---------------------------------------------------------------------

dispersion_df = results_df[
    [
        "feature_set",
        "alpha",
        "mean_predicted_variance",
    ]
].copy()


print("\n" + "=" * 78)
print("DISPERSION ESTIMATES")
print("=" * 78)

print(
    dispersion_df.round(6)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Prediction range
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
    "03_05_negative_binomial_validation_metrics.csv",
    index=False,
)


coefficients_df.to_csv(
    RESULTS_DIR /
    "03_05_negative_binomial_parameters.csv",
    index=False,
)


predictions_df.to_csv(
    RESULTS_DIR /
    "03_05_negative_binomial_validation_predictions.csv",
    index=False,
)


print("\nSaved:")
print(
    " results/"
    "03_05_negative_binomial_validation_metrics.csv"
)

print(
    " results/"
    "03_05_negative_binomial_parameters.csv"
)

print(
    " results/"
    "03_05_negative_binomial_validation_predictions.csv"
)


print(
    "\nStep 5 Negative Binomial analysis finished."
)