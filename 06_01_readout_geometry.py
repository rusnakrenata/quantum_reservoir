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

CV_OUTPUT = (
    RESULTS_DIR
    / "06_01_ridge_cv_results.csv"
)

GEOMETRY_OUTPUT = (
    RESULTS_DIR
    / "06_01_feature_geometry.csv"
)

SINGULAR_OUTPUT = (
    RESULTS_DIR
    / "06_01_singular_values.csv"
)

EIGEN_OUTPUT = (
    RESULTS_DIR
    / "06_01_correlation_eigenvalues.csv"
)

VALIDATION_OUTPUT = (
    RESULTS_DIR
    / "06_01_validation_results.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "06_01_summary.txt"
)


# ============================================================
# Frozen QRC features
# ============================================================

FEATURE_COLUMNS = [
    "X0", "Z0",
    "X1", "Z1",
    "X2", "Z2",
    "X3", "Z3",
    "X4", "Z4",
    "X5", "Z5",
]


# ============================================================
# Target candidates
# ============================================================

TARGET_CANDIDATES = [
    "target_property_damage_claim_count",
    "property_damage_claim_count_t_plus_1",
    "property_damage_claim_count_target",
    "target_claim_count",
    "C_t_plus_1",
]


# ============================================================
# Expanded Ridge grid
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
    150.0,
    200.0,
    250.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
    100000.0,
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
# Utilities
# ============================================================

def resolve_target_column(df):

    for candidate in TARGET_CANDIDATES:

        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not resolve target column."
    )


def make_model(alpha):

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
        rmse
        / train_target_std
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "NRMSE": nrmse,
    }


# ============================================================
# Chronological Ridge CV
# ============================================================

def ridge_cv(
    X_train,
    y_train,
    protocol,
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

            X_fold_val = (
                X_train[
                    val_idx
                ]
            )

            y_fold_train = (
                y_train[
                    train_idx
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

            prediction = (
                model.predict(
                    X_fold_val
                )
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_fold_val,
                        prediction,
                    )
                )
            )

            fold_rmse.append(
                rmse
            )

        rows.append(
            {
                "protocol": protocol,
                "lambda": alpha,

                "cv_rmse_mean": float(
                    np.mean(
                        fold_rmse
                    )
                ),

                "cv_rmse_std": float(
                    np.std(
                        fold_rmse,
                        ddof=1,
                    )
                ),

                "fold_1_rmse": fold_rmse[0],
                "fold_2_rmse": fold_rmse[1],
                "fold_3_rmse": fold_rmse[2],
                "fold_4_rmse": fold_rmse[3],
                "fold_5_rmse": fold_rmse[4],
            }
        )

    cv_df = pd.DataFrame(
        rows
    )

    best_row = (
        cv_df
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
        cv_df,
    )


# ============================================================
# Feature geometry
# ============================================================

def analyze_feature_geometry(
    X_train,
    protocol,
):

    # --------------------------------------------------------
    # IMPORTANT:
    # Geometry is calculated on standardized TRAINING
    # features only.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_std = scaler.fit_transform(
        X_train
    )

    n_samples = X_std.shape[0]
    n_features = X_std.shape[1]

    # --------------------------------------------------------
    # Singular-value decomposition
    #
    # X = U Sigma V^T
    # --------------------------------------------------------

    singular_values = np.linalg.svd(
        X_std,
        compute_uv=False,
        full_matrices=False,
    )

    # NumPy returns descending order.
    sigma_max = float(
        singular_values[0]
    )

    sigma_min = float(
        singular_values[-1]
    )

    # --------------------------------------------------------
    # Numerical rank
    # --------------------------------------------------------

    rank = int(
        np.linalg.matrix_rank(
            X_std
        )
    )

    # --------------------------------------------------------
    # Condition number kappa(X)
    # --------------------------------------------------------

    if (
        rank < n_features
        or
        sigma_min <= 0
    ):

        kappa_X = np.inf

    else:

        kappa_X = float(
            sigma_max
            / sigma_min
        )

    # --------------------------------------------------------
    # Standardized feature correlation matrix
    #
    # C = (1/N) X^T X
    #
    # Because X_std has zero mean and population std=1,
    # diagonal entries should be approximately 1.
    # --------------------------------------------------------

    C = (
        X_std.T
        @ X_std
    ) / n_samples

    # Symmetric matrix -> eigvalsh
    eigenvalues = np.linalg.eigvalsh(
        C
    )

    # Sort descending
    eigenvalues = np.sort(
        eigenvalues
    )[::-1]

    lambda_max = float(
        eigenvalues[0]
    )

    lambda_min = float(
        eigenvalues[-1]
    )

    spectral_radius_C = (
        lambda_max
    )

    # --------------------------------------------------------
    # Verify:
    #
    # lambda_i(C)
    # =
    # sigma_i(X)^2 / N
    # --------------------------------------------------------

    eigenvalues_from_svd = (
        singular_values ** 2
        / n_samples
    )

    spectrum_identity_error = float(
        np.max(
            np.abs(
                eigenvalues
                -
                eigenvalues_from_svd
            )
        )
    )

    # --------------------------------------------------------
    # Condition number of C
    #
    # kappa(C) = kappa(X)^2
    # --------------------------------------------------------

    if (
        rank < n_features
        or
        lambda_min <= 0
    ):

        kappa_C = np.inf

    else:

        kappa_C = float(
            lambda_max
            / lambda_min
        )

    # --------------------------------------------------------
    # Diagonal check:
    #
    # standardized C should have diag ~1
    # --------------------------------------------------------

    diagonal_error = float(
        np.max(
            np.abs(
                np.diag(C)
                - 1.0
            )
        )
    )

    # --------------------------------------------------------
    # Save singular-value rows
    # --------------------------------------------------------

    singular_rows = []

    for i, sigma in enumerate(
        singular_values,
        start=1,
    ):

        singular_rows.append(
            {
                "protocol": protocol,
                "index": i,
                "singular_value": float(
                    sigma
                ),
                "sigma_squared": float(
                    sigma ** 2
                ),
                "sigma_squared_over_N": float(
                    sigma ** 2
                    / n_samples
                ),
            }
        )

    # --------------------------------------------------------
    # Save covariance/correlation eigenvalues
    # --------------------------------------------------------

    eigen_rows = []

    for i, eigenvalue in enumerate(
        eigenvalues,
        start=1,
    ):

        eigen_rows.append(
            {
                "protocol": protocol,
                "index": i,
                "eigenvalue_C": float(
                    eigenvalue
                ),
            }
        )

    geometry = {
        "protocol": protocol,

        "n_samples": n_samples,
        "n_features": n_features,

        "rank": rank,

        "sigma_max": sigma_max,
        "sigma_min": sigma_min,

        "kappa_X": kappa_X,

        "lambda_max_C": lambda_max,
        "lambda_min_C": lambda_min,

        "rho_C": spectral_radius_C,

        "kappa_C": kappa_C,

        "diag_C_max_error": (
            diagonal_error
        ),

        "svd_eigen_identity_error": (
            spectrum_identity_error
        ),
    }

    return (
        geometry,
        pd.DataFrame(
            singular_rows
        ),
        pd.DataFrame(
            eigen_rows
        ),
    )


# ============================================================
# Ridge-conditioned geometry
# ============================================================

def ridge_condition_number(
    sigma_max,
    sigma_min,
    ridge_lambda,
):

    numerator = (
        sigma_max ** 2
        +
        ridge_lambda
    )

    denominator = (
        sigma_min ** 2
        +
        ridge_lambda
    )

    return float(
        numerator
        / denominator
    )


# ============================================================
# Final validation fit
# ============================================================

def fit_and_validate(
    X_train,
    y_train,
    X_val,
    y_val,
    best_lambda,
    train_target_std,
):

    model = make_model(
        best_lambda
    )

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_val
    )

    metrics = calculate_metrics(
        y_true=y_val,
        y_pred=prediction,
        train_target_std=train_target_std,
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 6 - STEP 6.1")
    log("RIDGE READOUT CALIBRATION + QRC FEATURE GEOMETRY")
    log("=" * 80)
    log()

    # --------------------------------------------------------
    # Load previous Week 5 features
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

    target_cont = (
        resolve_target_column(
            cont
        )
    )

    target_rwp = (
        resolve_target_column(
            rwp
        )
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    cont["target_date"] = (
        pd.to_datetime(
            cont["target_date"]
        )
    )

    rwp["target_date"] = (
        pd.to_datetime(
            rwp["target_date"]
        )
    )

    # --------------------------------------------------------
    # Align to RWP W=7 dates
    # --------------------------------------------------------

    common_dates = set(
        rwp["target_date"]
    )

    cont = (
        cont[
            cont["target_date"]
            .isin(
                common_dates
            )
        ]
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    rwp = (
        rwp
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

    if len(cont) != len(rwp):
        raise RuntimeError(
            "CONT / RWP aligned "
            "row counts differ."
        )

    if not (
        cont["target_date"]
        .equals(
            rwp["target_date"]
        )
    ):
        raise RuntimeError(
            "Target dates differ."
        )

    if not (
        cont["split"]
        .astype(str)
        .to_numpy()
        ==
        rwp["split"]
        .astype(str)
        .to_numpy()
    ).all():

        raise RuntimeError(
            "Split assignments differ."
        )

    y_cont = (
        cont[
            target_cont
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_rwp = (
        rwp[
            target_rwp
        ]
        .to_numpy(
            dtype=float
        )
    )

    max_target_difference = float(
        np.max(
            np.abs(
                y_cont
                -
                y_rwp
            )
        )
    )

    if max_target_difference > 1e-12:

        raise RuntimeError(
            "Target mismatch."
        )

    # --------------------------------------------------------
    # Split masks
    # --------------------------------------------------------

    split = (
        rwp["split"]
        .astype(str)
        .str.lower()
    )

    train_mask = (
        split
        == "train"
    )

    val_mask = (
        split
        .isin(
            [
                "validation",
                "val",
            ]
        )
    )

    test_mask = (
        split
        == "test"
    )

    log(
        f"Aligned rows        = "
        f"{len(rwp)}"
    )

    log(
        f"Train rows          = "
        f"{train_mask.sum()}"
    )

    log(
        f"Validation rows     = "
        f"{val_mask.sum()}"
    )

    log(
        f"Test rows           = "
        f"{test_mask.sum()} "
        f"(NOT evaluated)"
    )

    log(
        f"Maximum target diff = "
        f"{max_target_difference:.3e}"
    )

    log()

    # --------------------------------------------------------
    # Common target vectors
    # --------------------------------------------------------

    y = y_rwp

    y_train = y[
        train_mask
    ]

    y_val = y[
        val_mask
    ]

    train_target_std = float(
        np.std(
            y_train,
            ddof=0,
        )
    )

    # --------------------------------------------------------
    # Protocol matrices
    # --------------------------------------------------------

    protocol_data = {
        "CONT": cont,
        "RWP_W7": rwp,
    }

    all_cv_results = []
    all_geometry = []
    all_singular = []
    all_eigenvalues = []
    validation_results = []

    # ========================================================
    # Analyze each protocol
    # ========================================================

    for protocol, df in (
        protocol_data.items()
    ):

        log("=" * 80)
        log(protocol)
        log("=" * 80)

        X = (
            df[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )

        X_train = X[
            train_mask
        ]

        X_val = X[
            val_mask
        ]

        # ----------------------------------------------------
        # Feature geometry
        # ----------------------------------------------------

        (
            geometry,
            singular_df,
            eigen_df,
        ) = analyze_feature_geometry(
            X_train=X_train,
            protocol=protocol,
        )

        # ----------------------------------------------------
        # Expanded Ridge CV
        # ----------------------------------------------------

        (
            best_lambda,
            cv_df,
        ) = ridge_cv(
            X_train=X_train,
            y_train=y_train,
            protocol=protocol,
        )

        # ----------------------------------------------------
        # Ridge-conditioned normal matrix
        # ----------------------------------------------------

        kappa_ridge = (
            ridge_condition_number(
                sigma_max=geometry[
                    "sigma_max"
                ],
                sigma_min=geometry[
                    "sigma_min"
                ],
                ridge_lambda=best_lambda,
            )
        )

        geometry[
            "best_lambda"
        ] = best_lambda

        geometry[
            "kappa_XTX_plus_lambdaI"
        ] = kappa_ridge

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        metrics = (
            fit_and_validate(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                best_lambda=best_lambda,
                train_target_std=(
                    train_target_std
                ),
            )
        )

        validation_results.append(
            {
                "protocol": protocol,
                "best_lambda": (
                    best_lambda
                ),
                **metrics,
            }
        )

        # ----------------------------------------------------
        # Store tables
        # ----------------------------------------------------

        all_cv_results.append(
            cv_df
        )

        all_geometry.append(
            geometry
        )

        all_singular.append(
            singular_df
        )

        all_eigenvalues.append(
            eigen_df
        )

        # ----------------------------------------------------
        # Print geometry
        # ----------------------------------------------------

        log(
            f"Training matrix shape   = "
            f"{X_train.shape}"
        )

        log(
            f"Numerical rank          = "
            f"{geometry['rank']}"
        )

        log(
            f"sigma_max               = "
            f"{geometry['sigma_max']:.12f}"
        )

        log(
            f"sigma_min               = "
            f"{geometry['sigma_min']:.12e}"
        )

        log(
            f"kappa(X)                = "
            f"{geometry['kappa_X']:.6e}"
        )

        log()

        log(
            f"lambda_max(C)           = "
            f"{geometry['lambda_max_C']:.12f}"
        )

        log(
            f"lambda_min(C)           = "
            f"{geometry['lambda_min_C']:.12e}"
        )

        log(
            f"rho(C)                  = "
            f"{geometry['rho_C']:.12f}"
        )

        log(
            f"kappa(C)                = "
            f"{geometry['kappa_C']:.6e}"
        )

        log()

        log(
            "Identity check:"
        )

        log(
            "  max |lambda_i(C) "
            "- sigma_i^2/N| "
            f"= "
            f"{geometry['svd_eigen_identity_error']:.3e}"
        )

        log(
            "  max |diag(C)-1| "
            f"= "
            f"{geometry['diag_C_max_error']:.3e}"
        )

        log()

        # ----------------------------------------------------
        # Singular values
        # ----------------------------------------------------

        log(
            "Singular values:"
        )

        for _, row in (
            singular_df.iterrows()
        ):

            log(
                f"  sigma_{int(row['index']):02d} "
                f"= "
                f"{row['singular_value']:.12f}"
            )

        log()

        # ----------------------------------------------------
        # Eigenvalues
        # ----------------------------------------------------

        log(
            "Eigenvalues of C:"
        )

        for _, row in (
            eigen_df.iterrows()
        ):

            log(
                f"  lambda_{int(row['index']):02d} "
                f"= "
                f"{row['eigenvalue_C']:.12f}"
            )

        log()

        # ----------------------------------------------------
        # CV grid
        # ----------------------------------------------------

        log(
            "Expanded Ridge CV:"
        )

        for _, row in (
            cv_df
            .sort_values(
                "lambda"
            )
            .iterrows()
        ):

            log(
                f"  lambda="
                f"{row['lambda']:<10g} "
                f"CV RMSE="
                f"{row['cv_rmse_mean']:.6f} "
                f"(+/- "
                f"{row['cv_rmse_std']:.6f})"
            )

        log()

        log(
            f"Selected lambda*        = "
            f"{best_lambda:g}"
        )

        log(
            "Condition number of "
            "X^T X + lambda* I = "
            f"{kappa_ridge:.6e}"
        )

        log()

        log(
            "2025 validation:"
        )

        log(
            f"  RMSE  = "
            f"{metrics['RMSE']:.6f}"
        )

        log(
            f"  MAE   = "
            f"{metrics['MAE']:.6f}"
        )

        log(
            f"  NRMSE = "
            f"{metrics['NRMSE']:.6f}"
        )

        log(
            f"  Bias  = "
            f"{metrics['Bias']:+.6f}"
        )

        log()

    # ========================================================
    # Save outputs
    # ========================================================

    cv_all = pd.concat(
        all_cv_results,
        ignore_index=True,
    )

    geometry_df = pd.DataFrame(
        all_geometry
    )

    singular_all = pd.concat(
        all_singular,
        ignore_index=True,
    )

    eigen_all = pd.concat(
        all_eigenvalues,
        ignore_index=True,
    )

    validation_df = pd.DataFrame(
        validation_results
    )

    cv_all.to_csv(
        CV_OUTPUT,
        index=False,
    )

    geometry_df.to_csv(
        GEOMETRY_OUTPUT,
        index=False,
    )

    singular_all.to_csv(
        SINGULAR_OUTPUT,
        index=False,
    )

    eigen_all.to_csv(
        EIGEN_OUTPUT,
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )

    # ========================================================
    # Final comparison
    # ========================================================

    log("=" * 80)
    log("STEP 6.1 FINAL SUMMARY")
    log("=" * 80)
    log()

    for _, row in (
        validation_df.iterrows()
    ):

        protocol = row[
            "protocol"
        ]

        geometry_row = (
            geometry_df[
                geometry_df[
                    "protocol"
                ]
                == protocol
            ]
            .iloc[0]
        )

        log(
            f"{protocol}:"
        )

        log(
            f"  rank        = "
            f"{int(geometry_row['rank'])}"
        )

        log(
            f"  kappa(X)    = "
            f"{geometry_row['kappa_X']:.6e}"
        )

        log(
            f"  rho(C)      = "
            f"{geometry_row['rho_C']:.6f}"
        )

        log(
            f"  lambda*     = "
            f"{row['best_lambda']:g}"
        )

        log(
            f"  Ridge kappa = "
            f"{geometry_row['kappa_XTX_plus_lambdaI']:.6e}"
        )

        log(
            f"  Val RMSE    = "
            f"{row['RMSE']:.6f}"
        )

        log()

    log(
        "IMPORTANT: 2026 test set "
        "remains untouched."
    )

    log()

    log(
        f"Saved CV results to:"
    )
    log(
        f"  {CV_OUTPUT}"
    )

    log(
        f"Saved geometry results to:"
    )
    log(
        f"  {GEOMETRY_OUTPUT}"
    )

    log(
        f"Saved singular values to:"
    )
    log(
        f"  {SINGULAR_OUTPUT}"
    )

    log(
        f"Saved correlation eigenvalues to:"
    )
    log(
        f"  {EIGEN_OUTPUT}"
    )

    log(
        f"Saved validation results to:"
    )
    log(
        f"  {VALIDATION_OUTPUT}"
    )

    SUMMARY_OUTPUT.write_text(
        "\n".join(
            output_lines
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()