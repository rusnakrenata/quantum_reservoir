"""
02_06_exposure_analysis.py
--------------------------

Week 2 - Step 6

Exposure analysis for property-damage claim forecasting.

Main question
-------------
Does active policy exposure P_t provide useful information
about next-day claim count C_{t+1}?

We investigate:

1. Descriptive exposure statistics.
2. Yearly exposure and target statistics.
3. Claim rate per 100,000 active policies.
4. Pearson correlation between exposure and target.
5. Spearman rank correlation.
6. Exposure-only OLS regression.
7. Incremental value of exposure after:
       weekday + holiday
8. Incremental value of exposure after:
       weekday + holiday + linear time trend
9. Residual correlation after calendar + time effects
   have been removed.
10. Visualizations.

Important
---------
P_t is the active-policy count known at time t.

The forecasting target is:

    C_{t+1}

so the analysis respects the forecasting information set.

Outputs
-------
results/exposure_yearly_statistics.csv
results/exposure_correlation_statistics.csv
results/exposure_model_comparison.csv
results/exposure_training_data.csv
results/exposure_vs_target_scatter.png
results/exposure_time_series.png
results/exposure_claim_rate_by_year.png
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

# Expected name in qrc_forecast_sample.
#
# A small fallback mechanism below is included in case the
# canonical column was named slightly differently.
EXPOSURE_CANDIDATES = [
    "active_policy_count_t",
    "current_active_policy_count",
    "active_policy_count",
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
# Resolve exposure column
# ============================================================

def resolve_exposure_column(df):
    """
    Find the canonical exposure column in the forecasting table.
    """

    for candidate in EXPOSURE_CANDIDATES:

        if candidate in df.columns:

            print(
                f"[analysis] Exposure column: "
                f"{candidate}"
            )

            return candidate

    raise RuntimeError(
        "Could not find active-policy exposure column.\n"
        f"Tried: {EXPOSURE_CANDIDATES}\n"
        f"Available columns:\n{list(df.columns)}"
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
            f"No data found for "
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
        .sort_values("target_date")
        .reset_index(drop=True)
    )

    if train.empty:

        raise RuntimeError(
            "Training split is empty."
        )

    exposure_column = (
        resolve_exposure_column(
            train
        )
    )

    # --------------------------------------------------------
    # Convert variables to appropriate types
    # --------------------------------------------------------

    train[TARGET_COLUMN] = pd.to_numeric(
        train[TARGET_COLUMN],
        errors="coerce",
    )

    train[exposure_column] = pd.to_numeric(
        train[exposure_column],
        errors="coerce",
    )

    train[WEEKDAY_COLUMN] = (
        train[WEEKDAY_COLUMN]
        .astype(int)
    )

    train[HOLIDAY_COLUMN] = (
        train[HOLIDAY_COLUMN]
        .astype(bool)
    )

    return train, exposure_column


# ============================================================
# Structural checks
# ============================================================

def check_exposure_data(
    df,
    exposure_column,
):

    print()
    print("=" * 80)
    print("EXPOSURE DATA CHECK")
    print("=" * 80)

    print(
        f"Training observations: "
        f"{len(df)}"
    )

    print(
        f"Date range: "
        f"{df['target_date'].min().date()} "
        f"-> "
        f"{df['target_date'].max().date()}"
    )

    missing_exposure = (
        df[exposure_column]
        .isna()
        .sum()
    )

    missing_target = (
        df[TARGET_COLUMN]
        .isna()
        .sum()
    )

    nonpositive_exposure = (
        df[exposure_column]
        <= 0
    ).sum()

    print(
        f"Missing exposure:       "
        f"{missing_exposure}"
    )

    print(
        f"Missing target:         "
        f"{missing_target}"
    )

    print(
        f"Non-positive exposure:  "
        f"{nonpositive_exposure}"
    )

    if missing_exposure > 0:

        raise ValueError(
            "Exposure contains missing values."
        )

    if missing_target > 0:

        raise ValueError(
            "Target contains missing values."
        )

    if nonpositive_exposure > 0:

        raise ValueError(
            "Exposure contains zero or "
            "negative values."
        )

    print()
    print(
        f"Exposure minimum: "
        f"{df[exposure_column].min():,.2f}"
    )

    print(
        f"Exposure maximum: "
        f"{df[exposure_column].max():,.2f}"
    )

    print(
        f"Exposure mean:    "
        f"{df[exposure_column].mean():,.2f}"
    )

    print(
        f"Exposure std:     "
        f"{df[exposure_column].std():,.2f}"
    )

    print("=" * 80)


# ============================================================
# Claim rate per exposure
# ============================================================

def add_claim_rate(
    df,
    exposure_column,
):
    """
    Calculate target claims per 100,000 active policies.

    rate =
        C_{t+1} / P_t * 100000

    This is useful primarily as a relative exposure-normalized
    quantity.
    """

    result = df.copy()

    result[
        "claim_rate_per_100k_policies"
    ] = (
        result[TARGET_COLUMN]
        /
        result[exposure_column]
        *
        100000.0
    )

    return result


# ============================================================
# Yearly statistics
# ============================================================

def calculate_yearly_statistics(
    df,
    exposure_column,
):

    working = df.copy()

    working["year"] = (
        working["target_date"]
        .dt.year
    )

    yearly = (
        working
        .groupby(
            "year"
        )
        .agg(
            count=(
                TARGET_COLUMN,
                "count",
            ),

            mean_claim_count=(
                TARGET_COLUMN,
                "mean",
            ),

            variance_claim_count=(
                TARGET_COLUMN,
                "var",
            ),

            mean_active_policy_count=(
                exposure_column,
                "mean",
            ),

            min_active_policy_count=(
                exposure_column,
                "min",
            ),

            max_active_policy_count=(
                exposure_column,
                "max",
            ),

            mean_claim_rate_per_100k=(
                "claim_rate_per_100k_policies",
                "mean",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Raw target dispersion within each year
    # --------------------------------------------------------

    yearly[
        "claim_dispersion_ratio"
    ] = (
        yearly[
            "variance_claim_count"
        ]
        /
        yearly[
            "mean_claim_count"
        ]
    )

    # --------------------------------------------------------
    # Percentage change from previous year
    # --------------------------------------------------------

    yearly[
        "policy_count_change_pct"
    ] = (
        yearly[
            "mean_active_policy_count"
        ]
        .pct_change()
        *
        100
    )

    yearly[
        "claim_count_change_pct"
    ] = (
        yearly[
            "mean_claim_count"
        ]
        .pct_change()
        *
        100
    )

    yearly[
        "claim_rate_change_pct"
    ] = (
        yearly[
            "mean_claim_rate_per_100k"
        ]
        .pct_change()
        *
        100
    )

    return yearly


# ============================================================
# Print yearly statistics
# ============================================================

def print_yearly_statistics(
    yearly,
):

    print()
    print("=" * 120)
    print("YEARLY EXPOSURE AND CLAIM STATISTICS")
    print("=" * 120)

    print(
        yearly
        .round(4)
        .to_string(
            index=False
        )
    )

    print("=" * 120)


# ============================================================
# Manual Pearson correlation
# ============================================================

def calculate_manual_pearson(
    x,
    y,
):
    """
    Calculate Pearson correlation manually.

    r =
        sum((x-xbar)(y-ybar))
        /
        sqrt(
            sum((x-xbar)^2)
            sum((y-ybar)^2)
        )
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

        correlation = np.nan

    else:

        correlation = (
            numerator
            /
            denominator
        )

    return {
        "n": len(x),

        "mean_x": x_mean,

        "mean_y": y_mean,

        "numerator": numerator,

        "sum_squared_x": (
            sum_squared_x
        ),

        "sum_squared_y": (
            sum_squared_y
        ),

        "denominator": denominator,

        "pearson_r": correlation,
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

        print(
            "[analysis] scipy not installed; "
            "Spearman test skipped."
        )

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
    Ordinary least squares using np.linalg.lstsq.

    Returns:
        beta
        predictions
        residuals
        SSE
        SST
        R^2
        adjusted R^2
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
        "predictions": predictions,
        "residuals": residuals,

        "sse": sse,
        "sst": sst,

        "r_squared": r_squared,

        "adjusted_r_squared": (
            adjusted_r_squared
        ),
    }


# ============================================================
# Standardize exposure
# ============================================================

def standardize_exposure(
    df,
    exposure_column,
):
    """
    Standardization performed using TRAINING data.

    z_P =
        (P_t - mean(P_train))
        /
        std(P_train)

    Because Step 6 uses training only, there is no leakage.
    """

    mean_exposure = (
        df[exposure_column]
        .mean()
    )

    std_exposure = (
        df[exposure_column]
        .std(
            ddof=1
        )
    )

    z = (
        df[exposure_column]
        -
        mean_exposure
    ) / std_exposure

    return (
        z.to_numpy(
            dtype=float
        ),
        mean_exposure,
        std_exposure,
    )


# ============================================================
# Calendar matrix
# ============================================================

def build_calendar_matrix(
    df,
):
    """
    Create:

        intercept
        Tuesday-Sunday weekday dummies
        holiday indicator

    Monday is the reference weekday.
    """

    weekday_dummies = pd.get_dummies(
        df[WEEKDAY_COLUMN],
        prefix="weekday",
        drop_first=True,
        dtype=float,
    )

    intercept = np.ones(
        (
            len(df),
            1,
        )
    )

    holiday = (
        df[HOLIDAY_COLUMN]
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )

    calendar_matrix = np.column_stack(
        [
            intercept,

            weekday_dummies.to_numpy(
                dtype=float
            ),

            holiday,
        ]
    )

    return calendar_matrix


# ============================================================
# Model comparisons
# ============================================================

def calculate_exposure_models(
    df,
    exposure_column,
):

    y = (
        df[TARGET_COLUMN]
        .astype(float)
        .to_numpy()
    )

    n = len(
        df
    )

    intercept = np.ones(
        (
            n,
            1,
        )
    )

    # --------------------------------------------------------
    # Standardized exposure
    # --------------------------------------------------------

    (
        exposure_z,
        exposure_mean,
        exposure_std,
    ) = standardize_exposure(
        df,
        exposure_column,
    )

    exposure_z_column = (
        exposure_z
        .reshape(-1, 1)
    )

    # ========================================================
    # MODEL A
    #
    # Exposure only
    # ========================================================

    X_exposure_only = np.column_stack(
        [
            intercept,
            exposure_z_column,
        ]
    )

    model_exposure_only = fit_ols(
        X_exposure_only,
        y,
    )

    # ========================================================
    # MODEL B
    #
    # Calendar:
    # weekday + holiday
    # ========================================================

    X_calendar = (
        build_calendar_matrix(
            df
        )
    )

    model_calendar = fit_ols(
        X_calendar,
        y,
    )

    # ========================================================
    # MODEL C
    #
    # Calendar + exposure
    # ========================================================

    X_calendar_exposure = np.column_stack(
        [
            X_calendar,
            exposure_z_column,
        ]
    )

    model_calendar_exposure = fit_ols(
        X_calendar_exposure,
        y,
    )

    delta_r2_calendar = (
        model_calendar_exposure[
            "r_squared"
        ]
        -
        model_calendar[
            "r_squared"
        ]
    )

    # ========================================================
    # Linear time trend
    #
    # scaled approximately to [-1,1]
    # ========================================================

    time_index = np.arange(
        n,
        dtype=float,
    )

    time_centered = (
        time_index
        -
        np.mean(
            time_index
        )
    )

    time_scaled = (
        time_centered
        /
        np.std(
            time_index,
            ddof=1,
        )
    )

    time_column = (
        time_scaled
        .reshape(-1, 1)
    )

    # ========================================================
    # MODEL D
    #
    # Calendar + time
    # ========================================================

    X_calendar_time = np.column_stack(
        [
            X_calendar,
            time_column,
        ]
    )

    model_calendar_time = fit_ols(
        X_calendar_time,
        y,
    )

    # ========================================================
    # MODEL E
    #
    # Calendar + time + exposure
    # ========================================================

    X_calendar_time_exposure = (
        np.column_stack(
            [
                X_calendar,
                time_column,
                exposure_z_column,
            ]
        )
    )

    model_calendar_time_exposure = (
        fit_ols(
            X_calendar_time_exposure,
            y,
        )
    )

    delta_r2_after_time = (
        model_calendar_time_exposure[
            "r_squared"
        ]
        -
        model_calendar_time[
            "r_squared"
        ]
    )

    # --------------------------------------------------------
    # Exposure coefficient:
    #
    # Because exposure is standardized, this coefficient
    # means change in predicted claims for +1 SD exposure.
    # --------------------------------------------------------

    exposure_only_beta = (
        model_exposure_only[
            "beta"
        ][-1]
    )

    calendar_exposure_beta = (
        model_calendar_exposure[
            "beta"
        ][-1]
    )

    calendar_time_exposure_beta = (
        model_calendar_time_exposure[
            "beta"
        ][-1]
    )

    models = pd.DataFrame(
        [
            {
                "model": "exposure_only",

                "sse": (
                    model_exposure_only[
                        "sse"
                    ]
                ),

                "r_squared": (
                    model_exposure_only[
                        "r_squared"
                    ]
                ),

                "adjusted_r_squared": (
                    model_exposure_only[
                        "adjusted_r_squared"
                    ]
                ),

                "exposure_beta_per_1sd": (
                    exposure_only_beta
                ),
            },

            {
                "model": "calendar_only",

                "sse": (
                    model_calendar[
                        "sse"
                    ]
                ),

                "r_squared": (
                    model_calendar[
                        "r_squared"
                    ]
                ),

                "adjusted_r_squared": (
                    model_calendar[
                        "adjusted_r_squared"
                    ]
                ),

                "exposure_beta_per_1sd": (
                    np.nan
                ),
            },

            {
                "model": "calendar_plus_exposure",

                "sse": (
                    model_calendar_exposure[
                        "sse"
                    ]
                ),

                "r_squared": (
                    model_calendar_exposure[
                        "r_squared"
                    ]
                ),

                "adjusted_r_squared": (
                    model_calendar_exposure[
                        "adjusted_r_squared"
                    ]
                ),

                "exposure_beta_per_1sd": (
                    calendar_exposure_beta
                ),
            },

            {
                "model": "calendar_plus_time",

                "sse": (
                    model_calendar_time[
                        "sse"
                    ]
                ),

                "r_squared": (
                    model_calendar_time[
                        "r_squared"
                    ]
                ),

                "adjusted_r_squared": (
                    model_calendar_time[
                        "adjusted_r_squared"
                    ]
                ),

                "exposure_beta_per_1sd": (
                    np.nan
                ),
            },

            {
                "model": (
                    "calendar_plus_time_plus_exposure"
                ),

                "sse": (
                    model_calendar_time_exposure[
                        "sse"
                    ]
                ),

                "r_squared": (
                    model_calendar_time_exposure[
                        "r_squared"
                    ]
                ),

                "adjusted_r_squared": (
                    model_calendar_time_exposure[
                        "adjusted_r_squared"
                    ]
                ),

                "exposure_beta_per_1sd": (
                    calendar_time_exposure_beta
                ),
            },
        ]
    )

    additional = {
        "exposure_mean": (
            exposure_mean
        ),

        "exposure_std": (
            exposure_std
        ),

        "delta_r2_after_calendar": (
            delta_r2_calendar
        ),

        "delta_r2_after_calendar_and_time": (
            delta_r2_after_time
        ),
    }

    return (
        models,
        additional,
        X_calendar_time,
        model_calendar_time,
        exposure_z,
    )


# ============================================================
# Residual exposure relationship
# ============================================================

def calculate_residual_correlation(
    df,
    exposure_z,
    X_calendar_time,
    calendar_time_model,
):
    """
    Remove weekday + holiday + linear time trend from BOTH:

        target
        exposure

    Then correlate the residuals.

    This asks:

        after calendar and simple trend are removed,
        do unexpectedly high-exposure days correspond to
        unexpectedly high claim-count days?
    """

    # --------------------------------------------------------
    # Target residuals
    # --------------------------------------------------------

    target_residuals = (
        calendar_time_model[
            "residuals"
        ]
    )

    # --------------------------------------------------------
    # Regress exposure itself on calendar + time
    # --------------------------------------------------------

    exposure_model = fit_ols(
        X_calendar_time,
        exposure_z,
    )

    exposure_residuals = (
        exposure_model[
            "residuals"
        ]
    )

    residual_pearson = (
        calculate_manual_pearson(
            exposure_residuals,
            target_residuals,
        )
    )

    return (
        residual_pearson,
        exposure_residuals,
        target_residuals,
    )


# ============================================================
# Print correlations
# ============================================================

def print_correlation_results(
    pearson,
    spearman,
    residual_pearson,
):

    print()
    print("=" * 90)
    print("EXPOSURE-TARGET CORRELATION")
    print("=" * 90)

    print()
    print("RAW PEARSON CALCULATION")
    print()

    print(
        f"N:                     "
        f"{pearson['n']}"
    )

    print(
        f"Mean exposure:         "
        f"{pearson['mean_x']:.6f}"
    )

    print(
        f"Mean target:           "
        f"{pearson['mean_y']:.6f}"
    )

    print(
        f"Numerator:             "
        f"{pearson['numerator']:.6f}"
    )

    print(
        f"Sum squared exposure:  "
        f"{pearson['sum_squared_x']:.6f}"
    )

    print(
        f"Sum squared target:    "
        f"{pearson['sum_squared_y']:.6f}"
    )

    print(
        f"Denominator:           "
        f"{pearson['denominator']:.6f}"
    )

    print(
        f"Pearson r:             "
        f"{pearson['pearson_r']:.6f}"
    )

    print()
    print("SPEARMAN")

    print(
        f"Spearman rho:          "
        f"{spearman['spearman_r']:.6f}"
    )

    print(
        f"Spearman p-value:      "
        f"{spearman['spearman_p_value']:.12g}"
    )

    print()
    print(
        "PEARSON AFTER REMOVING "
        "WEEKDAY + HOLIDAY + LINEAR TIME"
    )

    print(
        f"Residual Pearson r:    "
        f"{residual_pearson['pearson_r']:.6f}"
    )

    print("=" * 90)


# ============================================================
# Print model results
# ============================================================

def print_model_results(
    models,
    additional,
):

    print()
    print("=" * 105)
    print("INCREMENTAL VALUE OF ACTIVE-POLICY EXPOSURE")
    print("=" * 105)

    print(
        models
        .round(6)
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Training exposure mean: "
        f"{additional['exposure_mean']:,.4f}"
    )

    print(
        f"Training exposure std:  "
        f"{additional['exposure_std']:,.4f}"
    )

    print()

    print(
        "Delta R^2 from adding exposure "
        "after weekday + holiday:"
    )

    print(
        f"    "
        f"{additional['delta_r2_after_calendar']:.6f}"
    )

    print()

    print(
        "Delta R^2 from adding exposure "
        "after weekday + holiday + time trend:"
    )

    print(
        f"    "
        f"{additional['delta_r2_after_calendar_and_time']:.6f}"
    )

    print()
    print(
        "The exposure beta is expressed per "
        "+1 training standard deviation of exposure."
    )

    print("=" * 105)


# ============================================================
# Save outputs
# ============================================================

def save_results(
    df,
    yearly,
    pearson,
    spearman,
    residual_pearson,
    models,
    additional,
):

    # --------------------------------------------------------
    # Processed training data
    # --------------------------------------------------------

    training_file = (
        OUTPUT_DIR
        /
        "exposure_training_data.csv"
    )

    df.to_csv(
        training_file,
        index=False,
    )

    # --------------------------------------------------------
    # Yearly statistics
    # --------------------------------------------------------

    yearly_file = (
        OUTPUT_DIR
        /
        "exposure_yearly_statistics.csv"
    )

    yearly.to_csv(
        yearly_file,
        index=False,
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlation_file = (
        OUTPUT_DIR
        /
        "exposure_correlation_statistics.csv"
    )

    correlation_output = {
        "raw_pearson_r": (
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

        "residual_pearson_r_after_calendar_time": (
            residual_pearson[
                "pearson_r"
            ]
        ),
    }

    pd.DataFrame(
        [correlation_output]
    ).to_csv(
        correlation_file,
        index=False,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    model_file = (
        OUTPUT_DIR
        /
        "exposure_model_comparison.csv"
    )

    model_output = (
        models.copy()
    )

    model_output[
        "delta_r2_after_calendar"
    ] = (
        additional[
            "delta_r2_after_calendar"
        ]
    )

    model_output[
        "delta_r2_after_calendar_and_time"
    ] = (
        additional[
            "delta_r2_after_calendar_and_time"
        ]
    )

    model_output.to_csv(
        model_file,
        index=False,
    )

    print()
    print("[analysis] Saved:")

    print(
        f"           {training_file}"
    )

    print(
        f"           {yearly_file}"
    )

    print(
        f"           {correlation_file}"
    )

    print(
        f"           {model_file}"
    )


# ============================================================
# Plot exposure vs target
# ============================================================

def plot_exposure_vs_target(
    df,
    exposure_column,
):

    x = (
        df[exposure_column]
        .astype(float)
        .to_numpy()
    )

    y = (
        df[TARGET_COLUMN]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Simple linear fit for visualization only
    # --------------------------------------------------------

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    x_line = np.linspace(
        x.min(),
        x.max(),
        200,
    )

    y_line = (
        intercept
        +
        slope
        *
        x_line
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        x,
        y,
        alpha=0.35,
    )

    ax.plot(
        x_line,
        y_line,
    )

    ax.set_title(
        "Active Policy Exposure vs "
        "Next-Day Property-Damage Claims"
    )

    ax.set_xlabel(
        "Active policy count P(t)"
    )

    ax.set_ylabel(
        "Property-damage claims C(t+1)"
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        /
        "exposure_vs_target_scatter.png"
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
# Time-series plot
# ============================================================

def plot_exposure_time_series(
    df,
    exposure_column,
):

    working = df.copy()

    working[
        "claims_30d_mean"
    ] = (
        working[
            TARGET_COLUMN
        ]
        .rolling(
            window=30,
            min_periods=30,
        )
        .mean()
    )

    working[
        "exposure_30d_mean"
    ] = (
        working[
            exposure_column
        ]
        .rolling(
            window=30,
            min_periods=30,
        )
        .mean()
    )

    fig, ax1 = plt.subplots(
        figsize=(13, 6)
    )

    ax1.plot(
        working[
            "target_date"
        ],
        working[
            "claims_30d_mean"
        ],
        label="30-day mean claims",
    )

    ax1.set_xlabel(
        "Date"
    )

    ax1.set_ylabel(
        "30-day mean claims"
    )

    ax2 = ax1.twinx()

    ax2.plot(
        working[
            "target_date"
        ],
        working[
            "exposure_30d_mean"
        ],
        alpha=0.7,
        label="30-day mean active policies",
    )

    ax2.set_ylabel(
        "30-day mean active policy count"
    )

    ax1.set_title(
        "Claim Trend and Active-Policy Exposure "
        "— Training Period"
    )

    ax1.grid(
        True,
        alpha=0.2,
    )

    lines1, labels1 = (
        ax1.get_legend_handles_labels()
    )

    lines2, labels2 = (
        ax2.get_legend_handles_labels()
    )

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        /
        "exposure_time_series.png"
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
# Yearly normalized claim-rate plot
# ============================================================

def plot_yearly_claim_rate(
    yearly,
):

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.bar(
        yearly[
            "year"
        ].astype(str),
        yearly[
            "mean_claim_rate_per_100k"
        ],
    )

    ax.set_title(
        "Mean Property-Damage Claim Rate "
        "per 100,000 Active Policies"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Claims per 100,000 active policies"
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
        "exposure_claim_rate_by_year.png"
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
        "[analysis] Week 2 / Step 6"
    )

    print(
        "[analysis] Active-policy exposure analysis"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        train,
        exposure_column,
    ) = load_training_data()

    # --------------------------------------------------------
    # Data checks
    # --------------------------------------------------------

    check_exposure_data(
        train,
        exposure_column,
    )

    # --------------------------------------------------------
    # Exposure-normalized claim rate
    # --------------------------------------------------------

    train = add_claim_rate(
        train,
        exposure_column,
    )

    # --------------------------------------------------------
    # Yearly statistics
    # --------------------------------------------------------

    yearly = (
        calculate_yearly_statistics(
            train,
            exposure_column,
        )
    )

    print_yearly_statistics(
        yearly
    )

    # --------------------------------------------------------
    # Raw correlation
    # --------------------------------------------------------

    pearson = (
        calculate_manual_pearson(
            train[
                exposure_column
            ],
            train[
                TARGET_COLUMN
            ],
        )
    )

    spearman = calculate_spearman(
        train[
            exposure_column
        ],
        train[
            TARGET_COLUMN
        ],
    )

    # --------------------------------------------------------
    # Regression models
    # --------------------------------------------------------

    (
        models,
        additional,
        X_calendar_time,
        calendar_time_model,
        exposure_z,
    ) = calculate_exposure_models(
        train,
        exposure_column,
    )

    # --------------------------------------------------------
    # Residual relationship after controlling calendar + time
    # --------------------------------------------------------

    (
        residual_pearson,
        exposure_residuals,
        target_residuals,
    ) = calculate_residual_correlation(
        train,
        exposure_z,
        X_calendar_time,
        calendar_time_model,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_correlation_results(
        pearson,
        spearman,
        residual_pearson,
    )

    print_model_results(
        models,
        additional,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        df=train,
        yearly=yearly,
        pearson=pearson,
        spearman=spearman,
        residual_pearson=residual_pearson,
        models=models,
        additional=additional,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_exposure_vs_target(
        train,
        exposure_column,
    )

    plot_exposure_time_series(
        train,
        exposure_column,
    )

    plot_yearly_claim_rate(
        yearly
    )

    print()
    print(
        "[analysis] Week 2 / Step 6 completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()