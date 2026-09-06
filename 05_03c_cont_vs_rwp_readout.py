from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONT_FILE = (
    RESULTS_DIR
    / "05_03a_qrc_features.csv"
)

RWP_FILE = (
    RESULTS_DIR
    / "05_03b_rwp_w7_qrc_features.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "05_03c_cont_vs_rwp_summary.txt"
)

COMPARISON_FILE = (
    RESULTS_DIR
    / "05_03c_cont_vs_rwp_validation_results.csv"
)

CV_FILE = (
    RESULTS_DIR
    / "05_03c_ridge_cv_results.csv"
)

PREDICTION_FILE = (
    RESULTS_DIR
    / "05_03c_validation_predictions.csv"
)


# ============================================================
# Frozen target / QRC features
# ============================================================

TARGET = "target_property_damage_claim_count"

FEATURE_COLUMNS = [
    "X0", "Z0",
    "X1", "Z1",
    "X2", "Z2",
    "X3", "Z3",
    "X4", "Z4",
    "X5", "Z5",
]


# ============================================================
# Ridge configuration
# ============================================================

RIDGE_LAMBDAS = [
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
]

N_CV_SPLITS = 5


# ============================================================
# Logging
# ============================================================

output_lines = []


def log(text=""):
    text = str(text)
    print(text)
    output_lines.append(text)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    train_target_std,
):
    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    bias = float(
        np.mean(
            y_pred - y_true
        )
    )

    nrmse = float(
        rmse / train_target_std
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "NRMSE": nrmse,
    }


# ============================================================
# Build Ridge pipeline
# ============================================================

def make_model(alpha):
    """
    Scaling is fitted ONLY from the current
    training portion.

    Ridge then receives standardized QRC features.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha,
                    fit_intercept=True,
                ),
            ),
        ]
    )


# ============================================================
# Chronological CV for readout lambda
# ============================================================

def tune_ridge_lambda(
    X_train,
    y_train,
    protocol_name,
):

    tscv = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )

    rows = []

    for alpha in RIDGE_LAMBDAS:

        fold_rmse = []

        for fold_id, (
            train_idx,
            val_idx,
        ) in enumerate(
            tscv.split(X_train),
            start=1,
        ):

            X_fold_train = (
                X_train[
                    train_idx
                ]
            )

            y_fold_train = (
                y_train[
                    train_idx
                ]
            )

            X_fold_val = (
                X_train[
                    val_idx
                ]
            )

            y_fold_val = (
                y_train[
                    val_idx
                ]
            )

            model = make_model(
                alpha
            )

            model.fit(
                X_fold_train,
                y_fold_train,
            )

            pred = model.predict(
                X_fold_val
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_fold_val,
                        pred,
                    )
                )
            )

            fold_rmse.append(
                rmse
            )

        mean_rmse = float(
            np.mean(
                fold_rmse
            )
        )

        std_rmse = float(
            np.std(
                fold_rmse,
                ddof=1,
            )
        )

        rows.append(
            {
                "protocol": protocol_name,
                "lambda": alpha,
                "cv_rmse_mean": mean_rmse,
                "cv_rmse_std": std_rmse,
                "fold_rmse": ";".join(
                    f"{x:.8f}"
                    for x in fold_rmse
                ),
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    best_row = (
        result_df
        .sort_values(
            [
                "cv_rmse_mean",
                "lambda",
            ]
        )
        .iloc[0]
    )

    best_lambda = float(
        best_row["lambda"]
    )

    return (
        best_lambda,
        result_df,
    )


# ============================================================
# Fit one protocol
# ============================================================

def evaluate_protocol(
    protocol_name,
    X_train,
    y_train,
    X_val,
    y_val,
    train_target_std,
):

    (
        best_lambda,
        cv_df,
    ) = tune_ridge_lambda(
        X_train=X_train,
        y_train=y_train,
        protocol_name=protocol_name,
    )

    model = make_model(
        best_lambda
    )

    model.fit(
        X_train,
        y_train,
    )

    val_pred = model.predict(
        X_val
    )

    metrics = calculate_metrics(
        y_true=y_val,
        y_pred=val_pred,
        train_target_std=train_target_std,
    )

    result = {
        "protocol": protocol_name,
        "best_lambda": best_lambda,
        **metrics,
    }

    return (
        result,
        cv_df,
        val_pred,
    )


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 5 - STEP 5.3C")
    log("FIRST IDEAL-QRC FORECASTING COMPARISON")
    log("CONTINUOUS vs REWINDING W=7")
    log("=" * 80)
    log()

    # --------------------------------------------------------
    # Load QRC features
    # --------------------------------------------------------

    if not CONT_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {CONT_FILE}"
        )

    if not RWP_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {RWP_FILE}"
        )

    cont = pd.read_csv(
        CONT_FILE
    )

    rwp = pd.read_csv(
        RWP_FILE
    )

    # --------------------------------------------------------
    # Parse dates
    # --------------------------------------------------------

    for df in [
        cont,
        rwp,
    ]:
        df["target_date"] = (
            pd.to_datetime(
                df["target_date"]
            )
        )

        df["input_date"] = (
            pd.to_datetime(
                df["input_date"]
            )
        )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required = (
        [
            "target_date",
            "input_date",
            "split",
            TARGET,
        ]
        + FEATURE_COLUMNS
    )

    for name, df in [
        ("CONT", cont),
        ("RWP7", rwp),
    ]:

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} missing columns:\n"
                + "\n".join(
                    missing
                )
            )

    # --------------------------------------------------------
    # Align CONT to exactly the RWP target dates
    # --------------------------------------------------------

    rwp_dates = set(
        rwp["target_date"]
    )

    cont_aligned = (
        cont[
            cont["target_date"]
            .isin(rwp_dates)
        ]
        .copy()
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    rwp_aligned = (
        rwp
        .copy()
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Alignment audits
    # --------------------------------------------------------

    if len(cont_aligned) != len(rwp_aligned):
        raise RuntimeError(
            "CONT / RWP row-count mismatch "
            "after alignment."
        )

    if not (
        cont_aligned["target_date"]
        .equals(
            rwp_aligned[
                "target_date"
            ]
        )
    ):
        raise RuntimeError(
            "Target dates do not align."
        )

    if not (
        cont_aligned["split"]
        .astype(str)
        .to_numpy()
        ==
        rwp_aligned["split"]
        .astype(str)
        .to_numpy()
    ).all():
        raise RuntimeError(
            "Split assignments differ."
        )

    target_difference = np.max(
        np.abs(
            cont_aligned[
                TARGET
            ].to_numpy(
                dtype=float
            )
            -
            rwp_aligned[
                TARGET
            ].to_numpy(
                dtype=float
            )
        )
    )

    if target_difference > 1e-12:
        raise RuntimeError(
            "Target values differ "
            "between CONT and RWP."
        )

    log(
        f"Aligned samples       = "
        f"{len(rwp_aligned)}"
    )

    log(
        f"First target date     = "
        f"{rwp_aligned['target_date'].min().date()}"
    )

    log(
        f"Last target date      = "
        f"{rwp_aligned['target_date'].max().date()}"
    )

    log(
        f"Maximum target diff   = "
        f"{target_difference:.3e}"
    )

    log()

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    split_normalized = (
        rwp_aligned[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    train_mask = (
        split_normalized
        == "train"
    )

    val_mask = (
        split_normalized
        .isin(
            [
                "validation",
                "val",
            ]
        )
    )

    test_mask = (
        split_normalized
        == "test"
    )

    log(
        f"Train rows            = "
        f"{train_mask.sum()}"
    )

    log(
        f"Validation rows       = "
        f"{val_mask.sum()}"
    )

    log(
        f"Test rows              = "
        f"{test_mask.sum()} "
        f"(NOT evaluated)"
    )

    log()

    if (
        train_mask.sum()
        != 1089
    ):
        raise RuntimeError(
            "Expected 1089 aligned "
            "training rows."
        )

    if (
        val_mask.sum()
        != 365
    ):
        raise RuntimeError(
            "Expected 365 validation rows."
        )

    if (
        test_mask.sum()
        != 232
    ):
        raise RuntimeError(
            "Expected 232 test rows."
        )

    # --------------------------------------------------------
    # Common targets
    # --------------------------------------------------------

    y = (
        rwp_aligned[
            TARGET
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_train = y[
        train_mask
    ]

    y_val = y[
        val_mask
    ]

    train_target_mean = float(
        np.mean(
            y_train
        )
    )

    train_target_std = float(
        np.std(
            y_train,
            ddof=0,
        )
    )

    log(
        f"Aligned train target mean = "
        f"{train_target_mean:.6f}"
    )

    log(
        f"Aligned train target std  = "
        f"{train_target_std:.6f}"
    )

    log()

    # --------------------------------------------------------
    # Protocol-specific feature matrices
    # --------------------------------------------------------

    X_cont = (
        cont_aligned[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    X_rwp = (
        rwp_aligned[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Train / validation only
    # --------------------------------------------------------

    X_cont_train = (
        X_cont[
            train_mask
        ]
    )

    X_cont_val = (
        X_cont[
            val_mask
        ]
    )

    X_rwp_train = (
        X_rwp[
            train_mask
        ]
    )

    X_rwp_val = (
        X_rwp[
            val_mask
        ]
    )

    # --------------------------------------------------------
    # Evaluate CONT
    # --------------------------------------------------------

    (
        cont_result,
        cont_cv,
        cont_val_pred,
    ) = evaluate_protocol(
        protocol_name="CONT",
        X_train=X_cont_train,
        y_train=y_train,
        X_val=X_cont_val,
        y_val=y_val,
        train_target_std=train_target_std,
    )

    # --------------------------------------------------------
    # Evaluate RWP(7)
    # --------------------------------------------------------

    (
        rwp_result,
        rwp_cv,
        rwp_val_pred,
    ) = evaluate_protocol(
        protocol_name="RWP_W7",
        X_train=X_rwp_train,
        y_train=y_train,
        X_val=X_rwp_val,
        y_val=y_val,
        train_target_std=train_target_std,
    )

    # --------------------------------------------------------
    # Save CV results
    # --------------------------------------------------------

    cv_all = pd.concat(
        [
            cont_cv,
            rwp_cv,
        ],
        ignore_index=True,
    )

    cv_all.to_csv(
        CV_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final validation comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            cont_result,
            rwp_result,
        ]
    )

    results_df.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "target_date": (
                rwp_aligned.loc[
                    val_mask,
                    "target_date",
                ]
                .reset_index(
                    drop=True
                )
            ),

            "y_true": y_val,

            "CONT_prediction": (
                cont_val_pred
            ),

            "RWP_W7_prediction": (
                rwp_val_pred
            ),
        }
    )

    prediction_df.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Report CV
    # --------------------------------------------------------

    log("=" * 80)
    log("RIDGE READOUT - TRAINING-ONLY CHRONOLOGICAL CV")
    log("=" * 80)
    log()

    for protocol in [
        "CONT",
        "RWP_W7",
    ]:

        protocol_cv = (
            cv_all[
                cv_all["protocol"]
                == protocol
            ]
            .sort_values(
                "lambda"
            )
        )

        log(
            f"{protocol}:"
        )

        for _, row in protocol_cv.iterrows():

            log(
                f"  lambda="
                f"{row['lambda']:<10g} "
                f"CV RMSE="
                f"{row['cv_rmse_mean']:.6f} "
                f"(+/- "
                f"{row['cv_rmse_std']:.6f})"
            )

        log()

    # --------------------------------------------------------
    # Report validation results
    # --------------------------------------------------------

    log("=" * 80)
    log("2025 VALIDATION RESULTS")
    log("=" * 80)
    log()

    log(
        "Protocol    lambda        "
        "RMSE       MAE        "
        "NRMSE      Bias"
    )

    for result in [
        cont_result,
        rwp_result,
    ]:

        log(
            f"{result['protocol']:<11s} "
            f"{result['best_lambda']:<13g} "
            f"{result['RMSE']:<10.6f} "
            f"{result['MAE']:<10.6f} "
            f"{result['NRMSE']:<10.6f} "
            f"{result['Bias']:+.6f}"
        )

    # --------------------------------------------------------
    # Direct difference
    # --------------------------------------------------------

    delta_rmse = (
        rwp_result["RMSE"]
        -
        cont_result["RMSE"]
    )

    relative_delta = (
        100.0
        * delta_rmse
        / cont_result["RMSE"]
    )

    log()

    log(
        "RWP(7) - CONT:"
    )

    log(
        f"  Delta RMSE          = "
        f"{delta_rmse:+.6f}"
    )

    log(
        f"  Relative Delta RMSE = "
        f"{relative_delta:+.3f}%"
    )

    if delta_rmse < 0:

        log(
            "  Interpretation      = "
            "RWP(7) is better on validation."
        )

    elif delta_rmse > 0:

        log(
            "  Interpretation      = "
            "CONT is better on validation."
        )

    else:

        log(
            "  Interpretation      = "
            "equal validation RMSE."
        )

    log()

    log(
        "IMPORTANT: 2026 test set was "
        "not evaluated."
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    SUMMARY_FILE.write_text(
        "\n".join(
            output_lines
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved CV results to: "
        f"{CV_FILE}"
    )

    print(
        f"Saved validation comparison to: "
        f"{COMPARISON_FILE}"
    )

    print(
        f"Saved validation predictions to: "
        f"{PREDICTION_FILE}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()