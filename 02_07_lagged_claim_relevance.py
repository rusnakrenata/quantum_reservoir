"""
02_07_lagged_claim_relevance.py
-------------------------------

Week 2 - Step 7

Lagged predictive relevance of historical property-damage
claim counts.

Forecasting target
------------------
    y_t = C_{t+1}

Lag convention
--------------
    L_k = C_{(t+1)-k}

Therefore:

    lag 1 = C_t
    lag 2 = C_{t-1}
    lag 7 = claim count exactly seven days before target day
    lag 14 = fourteen days before target day
    etc.

Main research question
----------------------
Do older claim counts contain predictive information beyond:

    - target-day weekday
    - target-day holiday
    - current exposure
    - current claim count C_t

This is important for choosing the temporal memory/window
required by the classical and quantum reservoirs.

For every lag 1 ... MAX_LAG the script calculates:

1. Raw Pearson correlation:
       Corr(L_k, C_{t+1})

2. Spearman correlation.

3. Incremental R^2 after calendar + holiday + exposure:
       B
       versus
       B + L_k

4. Incremental R^2 after the CURRENT claim is already known:
       B + L_1
       versus
       B + L_1 + L_k

5. The same conditional comparison after adding
   a linear time trend.

Outputs
-------
results/lagged_claim_relevance.csv
results/lagged_claim_key_lags.csv
results/lagged_claim_correlation.png
results/lagged_claim_incremental_r2.png
results/lagged_claim_incremental_r2_after_current.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import text

from db_config import engine


# ============================================================
# Configuration
# ============================================================

FORECAST_DATASET_NAME = "property_damage_next_day_v1"

TARGET_COLUMN = "target_property_damage_claim_count"

WEEKDAY_COLUMN = "day_of_week_t_plus_1"

HOLIDAY_COLUMN = "is_public_holiday_t_plus_1"

CURRENT_CLAIM_CANDIDATES = [
    "property_damage_claim_count_t",
    "current_property_damage_claim_count",
    "current_claim_count",
]

EXPOSURE_CANDIDATES = [
    "active_policy_count_t",
    "current_active_policy_count",
    "active_policy_count",
]

MAX_LAG = 28

IMPORTANT_LAGS = [
    1,
    2,
    3,
    7,
    14,
    21,
    28,
]

OUTPUT_DIR = Path("results")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SQL
# ============================================================

QUERY = text("""
SELECT
    fs.*

FROM qrc_forecast_sample fs

JOIN qrc_forecast_dataset fd
    ON fd.id = fs.dataset_id

WHERE fd.dataset_name = :dataset_name

ORDER BY fs.target_date
""")


# ============================================================
# Resolve column helper
# ============================================================

def resolve_column(
    df,
    candidates,
    description,
):

    for candidate in candidates:

        if candidate in df.columns:

            print(
                f"[analysis] {description}: "
                f"{candidate}"
            )

            return candidate

    raise RuntimeError(
        f"Could not find {description}.\n"
        f"Tried:\n"
        f"{candidates}\n\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )


# ============================================================
# Load training data
# ============================================================

def load_training_data():

    with engine.connect() as connection:

        df = pd.read_sql(
            QUERY,
            connection,
            params={
                "dataset_name": FORECAST_DATASET_NAME
            },
        )

    if df.empty:

        raise RuntimeError(
            f"No rows found for "
            f"'{FORECAST_DATASET_NAME}'."
        )

    df["target_date"] = pd.to_datetime(
        df["target_date"]
    )

    train = (
        df[
            df["split"] == "train"
        ]
        .copy()
        .sort_values(
            "target_date"
        )
        .reset_index(
            drop=True
        )
    )

    if train.empty:

        raise RuntimeError(
            "Training split is empty."
        )

    current_claim_column = (
        resolve_column(
            train,
            CURRENT_CLAIM_CANDIDATES,
            "current claim column",
        )
    )

    exposure_column = (
        resolve_column(
            train,
            EXPOSURE_CANDIDATES,
            "exposure column",
        )
    )

    # --------------------------------------------------------
    # Type conversion
    # --------------------------------------------------------

    train[TARGET_COLUMN] = (
        pd.to_numeric(
            train[TARGET_COLUMN],
            errors="coerce",
        )
    )

    train[current_claim_column] = (
        pd.to_numeric(
            train[current_claim_column],
            errors="coerce",
        )
    )

    train[exposure_column] = (
        pd.to_numeric(
            train[exposure_column],
            errors="coerce",
        )
    )

    train[WEEKDAY_COLUMN] = (
        train[WEEKDAY_COLUMN]
        .astype(int)
    )

    train[HOLIDAY_COLUMN] = (
        train[HOLIDAY_COLUMN]
        .astype(bool)
    )

    # --------------------------------------------------------
    # Make sure target dates are consecutive.
    #
    # Shifting rows is valid only if one row corresponds
    # to one consecutive calendar day.
    # --------------------------------------------------------

    date_differences = (
        train[
            "target_date"
        ]
        .diff()
        .dropna()
    )

    bad_gaps = (
        date_differences
        != pd.Timedelta(
            days=1
        )
    )

    if bad_gaps.any():

        problematic = (
            train.loc[
                bad_gaps[
                    bad_gaps
                ].index,
                "target_date",
            ]
        )

        raise RuntimeError(
            "Target dates are not consecutive. "
            "Lag construction by row shifting "
            "would therefore be invalid.\n"
            f"Problem dates:\n{problematic}"
        )

    return (
        train,
        current_claim_column,
        exposure_column,
    )


# ============================================================
# Dataset checks
# ============================================================

def print_dataset_info(
    df,
    current_claim_column,
    exposure_column,
):

    print()
    print("=" * 80)
    print("TRAINING DATA")
    print("=" * 80)

    print(
        f"N: "
        f"{len(df)}"
    )

    print(
        f"Date range: "
        f"{df['target_date'].min().date()} "
        f"-> "
        f"{df['target_date'].max().date()}"
    )

    print(
        f"Target column: "
        f"{TARGET_COLUMN}"
    )

    print(
        f"Current claim column: "
        f"{current_claim_column}"
    )

    print(
        f"Exposure column: "
        f"{exposure_column}"
    )

    print("=" * 80)


# ============================================================
# Construct lagged claim variables
# ============================================================

def construct_claim_lags(
    df,
    current_claim_column,
):
    """
    Target row corresponds to C_{t+1}.

    current_claim_column contains C_t.

    Therefore:

        lag 1 = C_t

    For lag k:

        lag k =
            current_claim_column.shift(k - 1)

    Example:

        lag 7 =
            C_t shifted by 6 rows
          = C_{t-6}
          = C_{(t+1)-7}
    """

    result = df.copy()

    for lag in range(
        1,
        MAX_LAG + 1,
    ):

        result[
            f"claim_lag_{lag}"
        ] = (
            result[
                current_claim_column
            ]
            .shift(
                lag - 1
            )
        )

    return result


# ============================================================
# Manual Pearson correlation
# ============================================================

def calculate_manual_pearson(
    x,
    y,
):
    """
    Pearson correlation:

                 sum((x-xbar)(y-ybar))
    r = ------------------------------------------
        sqrt(sum((x-xbar)^2) sum((y-ybar)^2))
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    x_mean = np.mean(
        x
    )

    y_mean = np.mean(
        y
    )

    x_centered = (
        x
        -
        x_mean
    )

    y_centered = (
        y
        -
        y_mean
    )

    numerator = np.sum(
        x_centered
        *
        y_centered
    )

    sum_squared_x = np.sum(
        x_centered ** 2
    )

    sum_squared_y = np.sum(
        y_centered ** 2
    )

    denominator = np.sqrt(
        sum_squared_x
        *
        sum_squared_y
    )

    if denominator == 0:

        r = np.nan

    else:

        r = (
            numerator
            /
            denominator
        )

    return {
        "n": len(x),

        "mean_lag": (
            x_mean
        ),

        "mean_target": (
            y_mean
        ),

        "numerator": (
            numerator
        ),

        "sum_squared_lag": (
            sum_squared_x
        ),

        "sum_squared_target": (
            sum_squared_y
        ),

        "denominator": (
            denominator
        ),

        "pearson_r": (
            r
        ),
    }


# ============================================================
# Spearman correlation
# ============================================================

def calculate_spearman(
    x,
    y,
):

    try:

        from scipy.stats import (
            spearmanr,
        )

        result = spearmanr(
            x,
            y,
        )

        return {
            "spearman_r": (
                result.statistic
            ),

            "spearman_p_value": (
                result.pvalue
            ),
        }

    except ImportError:

        return {
            "spearman_r": np.nan,
            "spearman_p_value": np.nan,
        }


# ============================================================
# OLS helper
# ============================================================

def fit_ols(
    X,
    y,
):
    """
    Ordinary least squares.

    beta minimizes:

        SSE =
            sum(
                y_i - yhat_i
            )^2

    R^2:

            SSE
        1 - ----
            SST
    """

    beta, _, _, _ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    predictions = (
        X
        @
        beta
    )

    residuals = (
        y
        -
        predictions
    )

    sse = np.sum(
        residuals ** 2
    )

    y_mean = np.mean(
        y
    )

    sst = np.sum(
        (
            y
            -
            y_mean
        ) ** 2
    )

    r_squared = (
        1
        -
        sse / sst
    )

    n = len(
        y
    )

    p = (
        X.shape[1]
        -
        1
    )

    adjusted_r_squared = (
        1
        -
        (
            1 - r_squared
        )
        *
        (
            n - 1
        )
        /
        (
            n - p - 1
        )
    )

    return {
        "beta": beta,

        "predictions": (
            predictions
        ),

        "residuals": (
            residuals
        ),

        "sse": (
            sse
        ),

        "sst": (
            sst
        ),

        "r_squared": (
            r_squared
        ),

        "adjusted_r_squared": (
            adjusted_r_squared
        ),
    }


# ============================================================
# Build baseline design matrix
# ============================================================

def build_baseline_matrix(
    df,
    exposure_column,
    include_time=False,
):
    """
    Baseline predictors:

        weekday
        holiday
        exposure

    Optional robustness baseline:

        weekday
        holiday
        exposure
        linear time trend
    """

    n = len(
        df
    )

    # --------------------------------------------------------
    # Intercept
    # --------------------------------------------------------

    intercept = np.ones(
        (
            n,
            1,
        )
    )

    # --------------------------------------------------------
    # Weekday dummy variables
    # --------------------------------------------------------

    weekday_dummies = pd.get_dummies(
        df[WEEKDAY_COLUMN],
        prefix="weekday",
        drop_first=True,
        dtype=float,
    )

    weekday_matrix = (
        weekday_dummies
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    holiday = (
        df[HOLIDAY_COLUMN]
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )

    # --------------------------------------------------------
    # Standardized exposure
    #
    # Since this is training-only analysis, scaling is
    # estimated from the current training subset.
    # --------------------------------------------------------

    exposure = (
        df[exposure_column]
        .astype(float)
        .to_numpy()
    )

    exposure_mean = np.mean(
        exposure
    )

    exposure_std = np.std(
        exposure,
        ddof=1,
    )

    exposure_z = (
        (
            exposure
            -
            exposure_mean
        )
        /
        exposure_std
    ).reshape(-1, 1)

    X = np.column_stack(
        [
            intercept,
            weekday_matrix,
            holiday,
            exposure_z,
        ]
    )

    # --------------------------------------------------------
    # Optional linear time trend
    # --------------------------------------------------------

    if include_time:

        time = (
            df[
                "target_date"
            ]
            -
            df[
                "target_date"
            ].min()
        ).dt.days.to_numpy(
            dtype=float
        )

        time_z = (
            time
            -
            np.mean(
                time
            )
        ) / np.std(
            time,
            ddof=1,
        )

        X = np.column_stack(
            [
                X,
                time_z.reshape(
                    -1,
                    1,
                ),
            ]
        )

    return X


# ============================================================
# Analyze one lag
# ============================================================

def analyze_lag(
    df,
    lag,
    exposure_column,
):
    """
    Analyze one historical claim lag.

    Comparisons
    -----------

    A:
        baseline

    B:
        baseline + lag_k

    C:
        baseline + lag_1

    D:
        baseline + lag_1 + lag_k

    For lag > 1:

        Delta R^2 after current =
            R^2(D) - R^2(C)

    This is the main memory-relevance quantity.
    """

    lag_column = (
        f"claim_lag_{lag}"
    )

    # --------------------------------------------------------
    # For fair comparison, use only rows for which
    # both lag 1 and lag k exist.
    # --------------------------------------------------------

    required_columns = [
    "target_date",
    TARGET_COLUMN,
    "claim_lag_1",
    lag_column,
    exposure_column,
    WEEKDAY_COLUMN,
    HOLIDAY_COLUMN,
]

# Remove duplicate column names while preserving order.
# This is necessary for lag == 1 because:
#
#     lag_column == "claim_lag_1"
#
# and otherwise claim_lag_1 would appear twice.
    required_columns = list(
        dict.fromkeys(
            required_columns
        )
    )

    subset = (
        df[
            required_columns
        ]
        .dropna()
        .copy()
        .reset_index(
            drop=True
        )
    )

    y = (
        subset[
            TARGET_COLUMN
        ]
        .astype(float)
        .to_numpy()
    )

    lag_values = (
        subset[
            lag_column
        ]
        .astype(float)
        .to_numpy()
    )

    lag1_values = (
        subset[
            "claim_lag_1"
        ]
        .astype(float)
        .to_numpy()
    )

    # ========================================================
    # Raw correlations
    # ========================================================

    pearson = (
        calculate_manual_pearson(
            lag_values,
            y,
        )
    )

    spearman = (
        calculate_spearman(
            lag_values,
            y,
        )
    )

    # ========================================================
    # Baseline:
    #
    # weekday + holiday + exposure
    # ========================================================

    X_baseline = (
        build_baseline_matrix(
            subset,
            exposure_column,
            include_time=False,
        )
    )

    model_baseline = fit_ols(
        X_baseline,
        y,
    )

    # ========================================================
    # Baseline + lag k
    # ========================================================

    X_baseline_lag = np.column_stack(
        [
            X_baseline,
            lag_values,
        ]
    )

    model_baseline_lag = fit_ols(
        X_baseline_lag,
        y,
    )

    delta_r2_lag_vs_baseline = (
        model_baseline_lag[
            "r_squared"
        ]
        -
        model_baseline[
            "r_squared"
        ]
    )

    # ========================================================
    # Baseline + current claim (lag 1)
    # ========================================================

    X_baseline_current = (
        np.column_stack(
            [
                X_baseline,
                lag1_values,
            ]
        )
    )

    model_baseline_current = fit_ols(
        X_baseline_current,
        y,
    )

    delta_r2_current_vs_baseline = (
        model_baseline_current[
            "r_squared"
        ]
        -
        model_baseline[
            "r_squared"
        ]
    )

    # ========================================================
    # Baseline + current + lag k
    # ========================================================

    if lag == 1:

        model_baseline_current_lag = (
            model_baseline_current
        )

        delta_r2_after_current = (
            np.nan
        )

        delta_adjusted_r2_after_current = (
            np.nan
        )

    else:

        X_baseline_current_lag = (
            np.column_stack(
                [
                    X_baseline,
                    lag1_values,
                    lag_values,
                ]
            )
        )

        model_baseline_current_lag = (
            fit_ols(
                X_baseline_current_lag,
                y,
            )
        )

        delta_r2_after_current = (
            model_baseline_current_lag[
                "r_squared"
            ]
            -
            model_baseline_current[
                "r_squared"
            ]
        )

        delta_adjusted_r2_after_current = (
            model_baseline_current_lag[
                "adjusted_r_squared"
            ]
            -
            model_baseline_current[
                "adjusted_r_squared"
            ]
        )

    # ========================================================
    # Robustness:
    # baseline + linear time
    # ========================================================

    X_baseline_time = (
        build_baseline_matrix(
            subset,
            exposure_column,
            include_time=True,
        )
    )

    X_time_current = (
        np.column_stack(
            [
                X_baseline_time,
                lag1_values,
            ]
        )
    )

    model_time_current = (
        fit_ols(
            X_time_current,
            y,
        )
    )

    if lag == 1:

        delta_r2_after_current_time = (
            np.nan
        )

    else:

        X_time_current_lag = (
            np.column_stack(
                [
                    X_baseline_time,
                    lag1_values,
                    lag_values,
                ]
            )
        )

        model_time_current_lag = (
            fit_ols(
                X_time_current_lag,
                y,
            )
        )

        delta_r2_after_current_time = (
            model_time_current_lag[
                "r_squared"
            ]
            -
            model_time_current[
                "r_squared"
            ]
        )

    # ========================================================
    # Output
    # ========================================================

    result = {
        "lag": lag,

        "n_pairs": (
            len(subset)
        ),

        "pearson_r": (
            pearson[
                "pearson_r"
            ]
        ),

        "spearman_r": (
            spearman[
                "spearman_r"
            ]
        ),

        "spearman_p_value": (
            spearman[
                "spearman_p_value"
            ]
        ),

        "baseline_r_squared": (
            model_baseline[
                "r_squared"
            ]
        ),

        "baseline_plus_lag_r_squared": (
            model_baseline_lag[
                "r_squared"
            ]
        ),

        "delta_r2_lag_vs_baseline": (
            delta_r2_lag_vs_baseline
        ),

        "baseline_plus_current_r_squared": (
            model_baseline_current[
                "r_squared"
            ]
        ),

        "delta_r2_current_vs_baseline": (
            delta_r2_current_vs_baseline
        ),

        "baseline_plus_current_plus_lag_r_squared": (
            model_baseline_current_lag[
                "r_squared"
            ]
        ),

        "delta_r2_lag_after_current": (
            delta_r2_after_current
        ),

        "delta_adjusted_r2_lag_after_current": (
            delta_adjusted_r2_after_current
        ),

        "delta_r2_lag_after_current_and_time": (
            delta_r2_after_current_time
        ),

        # Manual Pearson components
        "pearson_numerator": (
            pearson[
                "numerator"
            ]
        ),

        "pearson_denominator": (
            pearson[
                "denominator"
            ]
        ),
    }

    return result


# ============================================================
# Analyze all lags
# ============================================================

def analyze_all_lags(
    df,
    exposure_column,
):

    rows = []

    for lag in range(
        1,
        MAX_LAG + 1,
    ):

        print(
            f"[analysis] Analyzing lag "
            f"{lag:2d} / {MAX_LAG}"
        )

        result = analyze_lag(
            df,
            lag,
            exposure_column,
        )

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Print results
# ============================================================

def print_results(
    results,
):

    print()
    print("=" * 105)
    print("LAGGED CLAIM RELEVANCE")
    print("=" * 105)

    compact_columns = [
        "lag",
        "n_pairs",
        "pearson_r",
        "spearman_r",
        "delta_r2_lag_vs_baseline",
        "delta_r2_current_vs_baseline",
        "delta_r2_lag_after_current",
        "delta_r2_lag_after_current_and_time",
    ]

    print(
        results[
            compact_columns
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    print("=" * 105)


# ============================================================
# Print important lags
# ============================================================

def print_important_lags(
    results,
):

    important = (
        results[
            results[
                "lag"
            ].isin(
                IMPORTANT_LAGS
            )
        ]
        .copy()
    )

    print()
    print("=" * 110)
    print("IMPORTANT TEMPORAL LAGS")
    print("=" * 110)

    columns = [
        "lag",
        "n_pairs",
        "pearson_r",
        "spearman_r",
        "delta_r2_lag_vs_baseline",
        "delta_r2_lag_after_current",
        "delta_r2_lag_after_current_and_time",
    ]

    print(
        important[
            columns
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    print("=" * 110)

    return important


# ============================================================
# Print strongest conditional lags
# ============================================================

def print_strongest_lags(
    results,
):

    valid = (
        results[
            results[
                "lag"
            ] > 1
        ]
        .copy()
    )

    strongest = (
        valid
        .sort_values(
            "delta_r2_lag_after_current",
            ascending=False,
        )
        .head(10)
    )

    print()
    print("=" * 100)
    print(
        "TOP 10 LAGS AFTER CURRENT CLAIM "
        "+ CALENDAR + HOLIDAY + EXPOSURE"
    )
    print("=" * 100)

    print(
        strongest[
            [
                "lag",
                "pearson_r",
                "delta_r2_lag_after_current",
                "delta_r2_lag_after_current_and_time",
            ]
        ]
        .round(6)
        .to_string(
            index=False
        )
    )

    print("=" * 100)


# ============================================================
# Save
# ============================================================

def save_results(
    results,
    important,
):

    results_file = (
        OUTPUT_DIR
        /
        "lagged_claim_relevance.csv"
    )

    important_file = (
        OUTPUT_DIR
        /
        "lagged_claim_key_lags.csv"
    )

    results.to_csv(
        results_file,
        index=False,
    )

    important.to_csv(
        important_file,
        index=False,
    )

    print()
    print("[analysis] Saved:")

    print(
        f"           {results_file}"
    )

    print(
        f"           {important_file}"
    )


# ============================================================
# Plot raw correlation
# ============================================================

def plot_correlations(
    results,
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.stem(
        results[
            "lag"
        ],
        results[
            "pearson_r"
        ],
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "Lagged Claim Count vs Next-Day Target "
        "— Raw Pearson Correlation"
    )

    ax.set_xlabel(
        "Lag k (days before target)"
    )

    ax.set_ylabel(
        "Pearson correlation"
    )

    ax.set_xticks(
        range(
            1,
            MAX_LAG + 1,
        )
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        /
        "lagged_claim_correlation.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"[analysis] Plot saved: "
        f"{output_file}"
    )

    plt.show()


# ============================================================
# Plot incremental R^2 vs baseline
# ============================================================

def plot_incremental_r2(
    results,
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        results[
            "lag"
        ],
        results[
            "delta_r2_lag_vs_baseline"
        ],
    )

    ax.set_title(
        "Incremental R² of Each Claim Lag "
        "After Calendar + Holiday + Exposure"
    )

    ax.set_xlabel(
        "Lag k"
    )

    ax.set_ylabel(
        "Delta R²"
    )

    ax.set_xticks(
        range(
            1,
            MAX_LAG + 1,
        )
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        /
        "lagged_claim_incremental_r2.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"[analysis] Plot saved: "
        f"{output_file}"
    )

    plt.show()


# ============================================================
# Plot additional value beyond current claim
# ============================================================

def plot_incremental_after_current(
    results,
):

    plot_data = (
        results[
            results[
                "lag"
            ] > 1
        ]
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        plot_data[
            "lag"
        ],
        plot_data[
            "delta_r2_lag_after_current"
        ],
    )

    ax.set_title(
        "Additional Predictive Value of Older Claim Lags "
        "After Current Claim Is Already Known"
    )

    ax.set_xlabel(
        "Lag k"
    )

    ax.set_ylabel(
        "Additional Delta R²"
    )

    ax.set_xticks(
        range(
            2,
            MAX_LAG + 1,
        )
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        /
        "lagged_claim_incremental_r2_after_current.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"[analysis] Plot saved: "
        f"{output_file}"
    )

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "[analysis] Week 2 / Step 7"
    )

    print(
        "[analysis] Lagged predictive relevance"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        train,
        current_claim_column,
        exposure_column,
    ) = load_training_data()

    print_dataset_info(
        train,
        current_claim_column,
        exposure_column,
    )

    # --------------------------------------------------------
    # Construct lags
    # --------------------------------------------------------

    train = construct_claim_lags(
        train,
        current_claim_column,
    )

    # --------------------------------------------------------
    # Analyze lags
    # --------------------------------------------------------

    results = analyze_all_lags(
        train,
        exposure_column,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        results
    )

    important = print_important_lags(
        results
    )

    print_strongest_lags(
        results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        important,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_correlations(
        results
    )

    plot_incremental_r2(
        results
    )

    plot_incremental_after_current(
        results
    )

    print()
    print(
        "[analysis] Week 2 / Step 7 completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()