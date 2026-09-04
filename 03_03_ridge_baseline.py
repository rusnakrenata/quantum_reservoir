from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

INPUT_FILE = RESULTS_DIR / "03_01_preprocessed_samples.csv"

SEASONAL_NAIVE_RMSE = 6.405691

ALPHAS = [
    1.0,
    2.0,
    3.0,
    5.0,
    6.0,
    6.25,
    6.5,
    6.6,
    6.75,
    7.0,
    7.5,
    8.0,
    9.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    50.0,
    75.0,
    100.0,
]

N_SPLITS = 5


# ---------------------------------------------------------------------
# Load Step 1 preprocessed data
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["input_date"] = pd.to_datetime(df["input_date"])
df["target_date"] = pd.to_datetime(df["target_date"])


train = df[df["split"] == "train"].copy()
validation = df[df["split"] == "validation"].copy()


print("=" * 78)
print("WEEK 3 - STEP 3: RIDGE REGRESSION")
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
# Metric helper
# ---------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, train_target_std):

    error = y_true - y_pred

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
        y_pred - y_true
    )

    return mae, rmse, nrmse, bias


# ---------------------------------------------------------------------
# Training target standard deviation
# ---------------------------------------------------------------------

train_target_std = train[
    TARGET
].std(ddof=0)


# ---------------------------------------------------------------------
# Time-series cross-validation
# ---------------------------------------------------------------------

cv_rows = []
validation_rows = []
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

    X_train = train[
        feature_columns
    ].to_numpy(dtype=float)

    y_train = train[
        TARGET
    ].to_numpy(dtype=float)

    X_val = validation[
        feature_columns
    ].to_numpy(dtype=float)

    y_val = validation[
        TARGET
    ].to_numpy(dtype=float)


    # -------------------------------------------------------------
    # Chronological CV
    # -------------------------------------------------------------

    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    alpha_scores = []


    for alpha in ALPHAS:

        fold_rmses = []


        for fold_number, (
            train_idx,
            cv_idx,
        ) in enumerate(
            tscv.split(X_train),
            start=1,
        ):

            X_fold_train = X_train[
                train_idx
            ]

            y_fold_train = y_train[
                train_idx
            ]

            X_fold_cv = X_train[
                cv_idx
            ]

            y_fold_cv = y_train[
                cv_idx
            ]


            model = Ridge(
                alpha=alpha,
                fit_intercept=True,
            )

            model.fit(
                X_fold_train,
                y_fold_train,
            )


            prediction = model.predict(
                X_fold_cv
            )


            rmse = np.sqrt(
                np.mean(
                    (
                        y_fold_cv
                        - prediction
                    ) ** 2
                )
            )


            fold_rmses.append(
                rmse
            )


            cv_rows.append(
                {
                    "feature_set":
                        feature_set_name,

                    "alpha":
                        alpha,

                    "fold":
                        fold_number,

                    "fold_rmse":
                        rmse,
                }
            )


        mean_cv_rmse = np.mean(
            fold_rmses
        )

        std_cv_rmse = np.std(
            fold_rmses
        )


        alpha_scores.append(
            {
                "alpha":
                    alpha,

                "mean_cv_rmse":
                    mean_cv_rmse,

                "std_cv_rmse":
                    std_cv_rmse,
            }
        )


    alpha_scores_df = pd.DataFrame(
        alpha_scores
    )


    best_row = (
        alpha_scores_df
        .sort_values(
            "mean_cv_rmse"
        )
        .iloc[0]
    )


    best_alpha = float(
        best_row["alpha"]
    )

    best_cv_rmse = float(
        best_row["mean_cv_rmse"]
    )


    print(
        f"\nBest alpha: "
        f"{best_alpha}"
    )

    print(
        f"Mean CV RMSE: "
        f"{best_cv_rmse:.6f}"
    )


    # -------------------------------------------------------------
    # Refit using ALL training data
    # -------------------------------------------------------------

    final_model = Ridge(
        alpha=best_alpha,
        fit_intercept=True,
    )


    final_model.fit(
        X_train,
        y_train,
    )


    # -------------------------------------------------------------
    # Validation prediction
    # -------------------------------------------------------------

    y_pred = final_model.predict(
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
        train_target_std,
    )


    improvement_vs_seasonal = (
        100.0
        * (
            SEASONAL_NAIVE_RMSE
            - rmse
        )
        / SEASONAL_NAIVE_RMSE
    )


    validation_rows.append(
        {
            "model":
                "Ridge",

            "feature_set":
                feature_set_name,

            "best_alpha":
                best_alpha,

            "cv_rmse":
                best_cv_rmse,

            "validation_mae":
                mae,

            "validation_rmse":
                rmse,

            "validation_nrmse":
                nrmse,

            "validation_bias":
                bias,

            "improvement_vs_seasonal_pct":
                improvement_vs_seasonal,
        }
    )


    # -------------------------------------------------------------
    # Coefficients
    # -------------------------------------------------------------

    coefficient_rows.append(
        {
            "feature_set":
                feature_set_name,

            "variable":
                "intercept",

            "coefficient":
                final_model.intercept_,
        }
    )


    for variable, coefficient in zip(
        feature_columns,
        final_model.coef_,
    ):

        coefficient_rows.append(
            {
                "feature_set":
                    feature_set_name,

                "variable":
                    variable,

                "coefficient":
                    coefficient,
            }
        )


    # -------------------------------------------------------------
    # Prediction rows
    # -------------------------------------------------------------

    for target_date, actual, prediction in zip(
        validation["target_date"],
        y_val,
        y_pred,
    ):

        prediction_rows.append(
            {
                "target_date":
                    target_date,

                "feature_set":
                    feature_set_name,

                "actual":
                    actual,

                "prediction":
                    prediction,
            }
        )


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------

cv_df = pd.DataFrame(
    cv_rows
)

results_df = pd.DataFrame(
    validation_rows
)

coefficients_df = pd.DataFrame(
    coefficient_rows
)

predictions_df = pd.DataFrame(
    prediction_rows
)


results_df = (
    results_df
    .sort_values(
        "validation_rmse"
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("RIDGE VALIDATION RESULTS")
print("=" * 78)

print(
    results_df.round(6)
    .to_string(index=False)
)


print("\n" + "=" * 78)
print("RIDGE COEFFICIENTS")
print("=" * 78)

print(
    coefficients_df.round(6)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

cv_df.to_csv(
    RESULTS_DIR /
    "03_03_ridge_cv_results.csv",
    index=False,
)


results_df.to_csv(
    RESULTS_DIR /
    "03_03_ridge_validation_metrics.csv",
    index=False,
)


coefficients_df.to_csv(
    RESULTS_DIR /
    "03_03_ridge_coefficients.csv",
    index=False,
)


predictions_df.to_csv(
    RESULTS_DIR /
    "03_03_ridge_validation_predictions.csv",
    index=False,
)


print("\nSaved:")
print(
    " results/"
    "03_03_ridge_cv_results.csv"
)

print(
    " results/"
    "03_03_ridge_validation_metrics.csv"
)

print(
    " results/"
    "03_03_ridge_coefficients.csv"
)

print(
    " results/"
    "03_03_ridge_validation_predictions.csv"
)


print(
    "\nStep 3 Ridge analysis finished."
)