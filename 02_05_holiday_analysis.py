"""
02_05_holiday_analysis.py
-------------------------

Public-holiday effect analysis for the property-damage
claim forecasting target.

Main question
-------------
Does target-day public-holiday status provide useful information
beyond target-day weekday?

This script:

1. Loads TRAINING forecasting samples.
2. Compares holiday vs non-holiday claim distributions.
3. Calculates:
       - count
       - mean
       - median
       - standard deviation
       - variance
       - dispersion ratio
       - mean difference
       - holiday reduction percentage
       - Cohen's d
4. Compares holidays and non-holidays within each weekday.
5. Fits two simple OLS models:
       Model 1: weekday only
       Model 2: weekday + holiday
6. Calculates:
       - SSE
       - SST
       - R^2
       - adjusted R^2
       - delta R^2 from adding holiday
7. Performs a STRATIFIED permutation test:
       holiday labels are shuffled only within the same weekday.
8. Saves numerical outputs to results/.
9. Produces holiday comparison plots.

Null hypothesis for stratified permutation test
-----------------------------------------------
H0:
    Conditional on weekday, public-holiday status provides
    no additional information about property-damage claim count.

Formally:

    C independent of H given D

Alternative:

    Holiday status adds information beyond weekday.

Outputs
-------
results/holiday_claim_statistics.csv
results/holiday_effect_summary.csv
results/holiday_by_weekday_statistics.csv
results/holiday_incremental_model_comparison.csv
results/holiday_stratified_permutation_test.csv
results/holiday_mean_claim_count.png
results/holiday_by_weekday_mean_claim_count.png
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

OUTPUT_DIR = Path("results")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WEEKDAY_NAMES = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday",
}

RANDOM_SEED = 42

N_PERMUTATIONS = 10000


# ============================================================
# SQL query
# ============================================================

QUERY = text("""
SELECT
    fs.target_date,
    fs.day_of_week_t_plus_1,
    fs.is_public_holiday_t_plus_1,
    fs.target_property_damage_claim_count,
    fs.split

FROM qrc_forecast_sample fs

JOIN qrc_forecast_dataset fd
    ON fd.id = fs.dataset_id

WHERE fd.dataset_name = :dataset_name

ORDER BY fs.target_date
""")


# ============================================================
# Load training data
# ============================================================

def load_training_data():
    """
    Load forecasting samples and retain only the training split.
    """

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
        .sort_values("target_date")
        .reset_index(drop=True)
    )

    if train.empty:
        raise RuntimeError(
            "Training split is empty."
        )

    train[HOLIDAY_COLUMN] = (
        train[HOLIDAY_COLUMN]
        .astype(bool)
    )

    train[WEEKDAY_COLUMN] = (
        train[WEEKDAY_COLUMN]
        .astype(int)
    )

    train["weekday_name"] = (
        train[WEEKDAY_COLUMN]
        .map(WEEKDAY_NAMES)
    )

    return train


# ============================================================
# Dataset info
# ============================================================

def print_dataset_info(df):

    print()
    print("=" * 75)
    print("TRAINING DATA")
    print("=" * 75)

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
        f"Public-holiday days: "
        f"{df[HOLIDAY_COLUMN].sum()}"
    )

    print(
        f"Non-holiday days: "
        f"{(~df[HOLIDAY_COLUMN]).sum()}"
    )

    print("=" * 75)


# ============================================================
# Holiday descriptive statistics
# ============================================================

def calculate_holiday_statistics(df):
    """
    Calculate descriptive statistics for holiday and
    non-holiday target days.
    """

    temp = df.copy()

    temp["holiday_group"] = np.where(
        temp[HOLIDAY_COLUMN],
        "holiday",
        "nonholiday",
    )

    stats = (
        temp
        .groupby(
            "holiday_group"
        )[TARGET_COLUMN]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            variance="var",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    stats[
        "dispersion_ratio"
    ] = (
        stats["variance"]
        /
        stats["mean"]
    )

    return stats


# ============================================================
# Holiday effect summary
# ============================================================

def calculate_holiday_effect(df):
    """
    Calculate overall holiday vs non-holiday effect.
    """

    nonholiday = (
        df.loc[
            ~df[HOLIDAY_COLUMN],
            TARGET_COLUMN,
        ]
        .astype(float)
        .to_numpy()
    )

    holiday = (
        df.loc[
            df[HOLIDAY_COLUMN],
            TARGET_COLUMN,
        ]
        .astype(float)
        .to_numpy()
    )

    n_nonholiday = len(
        nonholiday
    )

    n_holiday = len(
        holiday
    )

    mean_nonholiday = np.mean(
        nonholiday
    )

    mean_holiday = np.mean(
        holiday
    )

    variance_nonholiday = np.var(
        nonholiday,
        ddof=1,
    )

    variance_holiday = np.var(
        holiday,
        ddof=1,
    )

    mean_difference = (
        mean_nonholiday
        -
        mean_holiday
    )

    holiday_reduction_pct = (
        mean_difference
        /
        mean_nonholiday
        *
        100
    )

    # --------------------------------------------------------
    # Pooled standard deviation
    # --------------------------------------------------------

    pooled_variance = (
        (
            (n_nonholiday - 1)
            *
            variance_nonholiday
        )
        +
        (
            (n_holiday - 1)
            *
            variance_holiday
        )
    ) / (
        n_nonholiday
        +
        n_holiday
        -
        2
    )

    pooled_std = np.sqrt(
        pooled_variance
    )

    cohens_d = (
        mean_difference
        /
        pooled_std
        if pooled_std > 0
        else np.nan
    )

    result = {
        "n_nonholiday": n_nonholiday,
        "n_holiday": n_holiday,

        "mean_nonholiday": (
            mean_nonholiday
        ),

        "mean_holiday": (
            mean_holiday
        ),

        "mean_difference": (
            mean_difference
        ),

        "holiday_reduction_pct": (
            holiday_reduction_pct
        ),

        "cohens_d": (
            cohens_d
        ),
    }

    return result


# ============================================================
# Print holiday effect
# ============================================================

def print_holiday_effect(
    stats,
    effect,
):

    print()
    print("=" * 85)
    print("HOLIDAY VS NON-HOLIDAY")
    print("=" * 85)

    print(
        stats
        .round(3)
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Non-holiday mean:      "
        f"{effect['mean_nonholiday']:.4f}"
    )

    print(
        f"Holiday mean:          "
        f"{effect['mean_holiday']:.4f}"
    )

    print(
        f"Difference:            "
        f"{effect['mean_difference']:.4f}"
    )

    print(
        f"Holiday reduction:     "
        f"{effect['holiday_reduction_pct']:.2f}%"
    )

    print(
        f"Cohen's d:             "
        f"{effect['cohens_d']:.4f}"
    )

    print("=" * 85)


# ============================================================
# Holiday effect within weekday
# ============================================================

def calculate_holiday_by_weekday(df):
    """
    Compare holiday and non-holiday observations
    within the same weekday category.

    This helps distinguish the holiday effect from
    the weekday effect.
    """

    rows = []

    for weekday in range(
        1,
        8,
    ):

        weekday_df = df[
            df[WEEKDAY_COLUMN]
            == weekday
        ]

        nonholiday = (
            weekday_df.loc[
                ~weekday_df[HOLIDAY_COLUMN],
                TARGET_COLUMN,
            ]
            .astype(float)
        )

        holiday = (
            weekday_df.loc[
                weekday_df[HOLIDAY_COLUMN],
                TARGET_COLUMN,
            ]
            .astype(float)
        )

        rows.append(
            {
                "day_of_week": weekday,

                "weekday_name": (
                    WEEKDAY_NAMES[
                        weekday
                    ]
                ),

                "nonholiday_count": (
                    len(nonholiday)
                ),

                "holiday_count": (
                    len(holiday)
                ),

                "nonholiday_mean": (
                    nonholiday.mean()
                    if len(nonholiday) > 0
                    else np.nan
                ),

                "holiday_mean": (
                    holiday.mean()
                    if len(holiday) > 0
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "mean_difference"
    ] = (
        result["nonholiday_mean"]
        -
        result["holiday_mean"]
    )

    result[
        "holiday_reduction_pct"
    ] = (
        result["mean_difference"]
        /
        result["nonholiday_mean"]
        *
        100
    )

    return result


# ============================================================
# Print holiday-by-weekday results
# ============================================================

def print_holiday_by_weekday(
    result,
):

    print()
    print("=" * 100)
    print(
        "HOLIDAY EFFECT WITHIN EACH WEEKDAY"
    )
    print("=" * 100)

    print(
        result
        .round(3)
        .to_string(
            index=False
        )
    )

    print("=" * 100)


# ============================================================
# OLS helper
# ============================================================

def fit_ols(
    X,
    y,
):
    """
    Fit ordinary least squares.

    beta is estimated numerically using np.linalg.lstsq.

    Returns:
        coefficients
        predictions
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
        X @ beta
    )

    residuals = (
        y
        -
        predictions
    )

    sse = np.sum(
        residuals ** 2
    )

    mean_y = np.mean(
        y
    )

    sst = np.sum(
        (
            y
            -
            mean_y
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

    # Number of predictors excluding intercept
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
        "sse": sse,
        "sst": sst,
        "r_squared": r_squared,
        "adjusted_r_squared": (
            adjusted_r_squared
        ),
    }


# ============================================================
# Compare weekday-only vs weekday+holiday
# ============================================================

def compare_incremental_holiday_value(df):
    """
    Model 1:
        target ~ weekday

    Model 2:
        target ~ weekday + holiday

    Weekday is one-hot encoded.

    The change in R^2 measures the additional
    in-sample target variance explained by holiday
    once weekday is already included.
    """

    y = (
        df[TARGET_COLUMN]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # One-hot encode weekday
    #
    # One weekday category is dropped as a reference category.
    # --------------------------------------------------------

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

    weekday_matrix = (
        weekday_dummies
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Model 1:
    # weekday only
    # --------------------------------------------------------

    X_weekday = np.column_stack(
        [
            intercept,
            weekday_matrix,
        ]
    )

    model_weekday = fit_ols(
        X_weekday,
        y,
    )

    # --------------------------------------------------------
    # Model 2:
    # weekday + holiday
    # --------------------------------------------------------

    holiday = (
        df[HOLIDAY_COLUMN]
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )

    X_weekday_holiday = np.column_stack(
        [
            intercept,
            weekday_matrix,
            holiday,
        ]
    )

    model_weekday_holiday = fit_ols(
        X_weekday_holiday,
        y,
    )

    delta_r_squared = (
        model_weekday_holiday[
            "r_squared"
        ]
        -
        model_weekday[
            "r_squared"
        ]
    )

    delta_adjusted_r_squared = (
        model_weekday_holiday[
            "adjusted_r_squared"
        ]
        -
        model_weekday[
            "adjusted_r_squared"
        ]
    )

    result = pd.DataFrame(
        [
            {
                "model": "weekday_only",

                "sse": (
                    model_weekday[
                        "sse"
                    ]
                ),

                "r_squared": (
                    model_weekday[
                        "r_squared"
                    ]
                ),

                "adjusted_r_squared": (
                    model_weekday[
                        "adjusted_r_squared"
                    ]
                ),
            },

            {
                "model": (
                    "weekday_plus_holiday"
                ),

                "sse": (
                    model_weekday_holiday[
                        "sse"
                    ]
                ),

                "r_squared": (
                    model_weekday_holiday[
                        "r_squared"
                    ]
                ),

                "adjusted_r_squared": (
                    model_weekday_holiday[
                        "adjusted_r_squared"
                    ]
                ),
            },
        ]
    )

    return (
        result,
        delta_r_squared,
        delta_adjusted_r_squared,
    )


# ============================================================
# Stratified permutation test
# ============================================================

def stratified_holiday_permutation_test(
    df,
    n_permutations=N_PERMUTATIONS,
    random_seed=RANDOM_SEED,
):
    """
    Stratified permutation test for the incremental value
    of public-holiday status after controlling for weekday.

    Null hypothesis
    ---------------
    H0:
        Conditional on weekday, holiday status is unrelated
        to property-damage claim count.

        C independent of H given D

    Alternative hypothesis
    ----------------------
    H1:
        Holiday status provides additional information
        about claim count after weekday is already known.

    Permutation strategy
    --------------------
    Holiday labels are shuffled ONLY WITHIN the same weekday.

    Thus:

        Monday holiday labels
            -> shuffled among Mondays only

        Tuesday holiday labels
            -> shuffled among Tuesdays only

        ...

        Sunday holiday labels
            -> shuffled among Sundays only

    This preserves:
        - weekday counts
        - number of holidays within each weekday
        - the claim-count series itself

    while breaking the association between holiday status
    and claim count within weekday.

    Test statistic
    --------------
    Delta R^2:

        R^2(weekday + holiday)
        -
        R^2(weekday)

    If the observed Delta R^2 is much larger than the
    permuted values, the holiday feature contributes
    information beyond weekday.
    """

    rng = np.random.default_rng(
        random_seed
    )

    working = df.copy()

    y = (
        working[TARGET_COLUMN]
        .astype(float)
        .to_numpy()
    )

    # ========================================================
    # Weekday-only design matrix
    # ========================================================

    weekday_dummies = pd.get_dummies(
        working[WEEKDAY_COLUMN],
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

    intercept = np.ones(
        (
            len(working),
            1,
        )
    )

    X_weekday = np.column_stack(
        [
            intercept,
            weekday_matrix,
        ]
    )

    # --------------------------------------------------------
    # Fit weekday-only model once
    # --------------------------------------------------------

    weekday_model = fit_ols(
        X_weekday,
        y,
    )

    r2_weekday = (
        weekday_model[
            "r_squared"
        ]
    )

    # ========================================================
    # Observed weekday + holiday model
    # ========================================================

    observed_holiday = (
        working[HOLIDAY_COLUMN]
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )

    X_observed = np.column_stack(
        [
            intercept,
            weekday_matrix,
            observed_holiday,
        ]
    )

    observed_model = fit_ols(
        X_observed,
        y,
    )

    observed_delta_r2 = (
        observed_model[
            "r_squared"
        ]
        -
        r2_weekday
    )

    # ========================================================
    # Prepare original labels
    # ========================================================

    original_holiday = (
        working[HOLIDAY_COLUMN]
        .astype(int)
        .to_numpy()
    )

    weekdays = (
        working[WEEKDAY_COLUMN]
        .astype(int)
        .to_numpy()
    )

    permutation_delta_r2 = np.empty(
        n_permutations
    )

    # ========================================================
    # Permutation loop
    # ========================================================

    for permutation_index in range(
        n_permutations
    ):

        shuffled_holiday = (
            original_holiday.copy()
        )

        # ----------------------------------------------------
        # Shuffle holiday labels independently
        # inside every weekday.
        # ----------------------------------------------------

        for weekday in range(
            1,
            8,
        ):

            indices = np.where(
                weekdays == weekday
            )[0]

            shuffled_holiday[
                indices
            ] = rng.permutation(
                shuffled_holiday[
                    indices
                ]
            )

        shuffled_holiday_column = (
            shuffled_holiday
            .astype(float)
            .reshape(-1, 1)
        )

        # ----------------------------------------------------
        # Weekday + permuted holiday model
        # ----------------------------------------------------

        X_permuted = np.column_stack(
            [
                intercept,
                weekday_matrix,
                shuffled_holiday_column,
            ]
        )

        permuted_model = fit_ols(
            X_permuted,
            y,
        )

        permutation_delta_r2[
            permutation_index
        ] = (
            permuted_model[
                "r_squared"
            ]
            -
            r2_weekday
        )

    # ========================================================
    # Permutation p-value
    # ========================================================

    exceedances = np.sum(
        permutation_delta_r2
        >=
        observed_delta_r2
    )

    p_value = (
        exceedances
        +
        1
    ) / (
        n_permutations
        +
        1
    )

    # ========================================================
    # Reference distribution statistics
    # ========================================================

    permutation_mean = np.mean(
        permutation_delta_r2
    )

    permutation_std = np.std(
        permutation_delta_r2,
        ddof=1,
    )

    percentile_95 = np.quantile(
        permutation_delta_r2,
        0.95,
    )

    percentile_99 = np.quantile(
        permutation_delta_r2,
        0.99,
    )

    result = {
        "observed_delta_r_squared": (
            observed_delta_r2
        ),

        "permutation_mean_delta_r_squared": (
            permutation_mean
        ),

        "permutation_std_delta_r_squared": (
            permutation_std
        ),

        "permutation_95_percentile": (
            percentile_95
        ),

        "permutation_99_percentile": (
            percentile_99
        ),

        "number_exceeding_observed": (
            int(exceedances)
        ),

        "n_permutations": (
            n_permutations
        ),

        "permutation_p_value": (
            p_value
        ),
    }

    return result


# ============================================================
# Print incremental model comparison
# ============================================================

def print_model_comparison(
    result,
    delta_r_squared,
    delta_adjusted_r_squared,
):

    print()
    print("=" * 90)
    print(
        "INCREMENTAL VALUE OF HOLIDAY AFTER WEEKDAY"
    )
    print("=" * 90)

    print(
        result
        .round(6)
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Delta R^2 from adding holiday: "
        f"{delta_r_squared:.6f}"
    )

    print(
        f"Delta adjusted R^2:             "
        f"{delta_adjusted_r_squared:.6f}"
    )

    print("=" * 90)


# ============================================================
# Print stratified permutation results
# ============================================================

def print_stratified_permutation_test(
    permutation,
):

    print()
    print("=" * 90)
    print(
        "STRATIFIED HOLIDAY PERMUTATION TEST"
    )
    print("=" * 90)

    print(
        "H0: holiday status adds no information "
        "after weekday is known."
    )

    print(
        "Holiday labels are shuffled within "
        "weekday only."
    )

    print()

    print(
        f"Observed Delta R^2:           "
        f"{permutation['observed_delta_r_squared']:.6f}"
    )

    print(
        f"Mean permuted Delta R^2:      "
        f"{permutation['permutation_mean_delta_r_squared']:.6f}"
    )

    print(
        f"Std permuted Delta R^2:       "
        f"{permutation['permutation_std_delta_r_squared']:.6f}"
    )

    print(
        f"95% permutation percentile:   "
        f"{permutation['permutation_95_percentile']:.6f}"
    )

    print(
        f"99% permutation percentile:   "
        f"{permutation['permutation_99_percentile']:.6f}"
    )

    print(
        f"Permutations >= observed:     "
        f"{permutation['number_exceeding_observed']}"
    )

    print(
        f"Number of permutations:       "
        f"{permutation['n_permutations']}"
    )

    print(
        f"Permutation p-value:          "
        f"{permutation['permutation_p_value']:.8f}"
    )

    print("=" * 90)


# ============================================================
# Save results
# ============================================================

def save_results(
    stats,
    effect,
    by_weekday,
    model_comparison,
    delta_r_squared,
    delta_adjusted_r_squared,
    permutation,
):

    # --------------------------------------------------------
    # Holiday descriptive statistics
    # --------------------------------------------------------

    stats_file = (
        OUTPUT_DIR
        /
        "holiday_claim_statistics.csv"
    )

    stats.to_csv(
        stats_file,
        index=False,
    )

    # --------------------------------------------------------
    # Overall holiday effect
    # --------------------------------------------------------

    effect_file = (
        OUTPUT_DIR
        /
        "holiday_effect_summary.csv"
    )

    pd.DataFrame(
        [effect]
    ).to_csv(
        effect_file,
        index=False,
    )

    # --------------------------------------------------------
    # Holiday by weekday
    # --------------------------------------------------------

    weekday_file = (
        OUTPUT_DIR
        /
        "holiday_by_weekday_statistics.csv"
    )

    by_weekday.to_csv(
        weekday_file,
        index=False,
    )

    # --------------------------------------------------------
    # Incremental model comparison
    # --------------------------------------------------------

    model_file = (
        OUTPUT_DIR
        /
        "holiday_incremental_model_comparison.csv"
    )

    model_output = (
        model_comparison.copy()
    )

    model_output[
        "delta_r_squared"
    ] = (
        delta_r_squared
    )

    model_output[
        "delta_adjusted_r_squared"
    ] = (
        delta_adjusted_r_squared
    )

    model_output.to_csv(
        model_file,
        index=False,
    )

    # --------------------------------------------------------
    # Stratified permutation test
    # --------------------------------------------------------

    permutation_file = (
        OUTPUT_DIR
        /
        "holiday_stratified_permutation_test.csv"
    )

    pd.DataFrame(
        [permutation]
    ).to_csv(
        permutation_file,
        index=False,
    )

    print()
    print("[analysis] Saved:")

    print(
        f"           {stats_file}"
    )

    print(
        f"           {effect_file}"
    )

    print(
        f"           {weekday_file}"
    )

    print(
        f"           {model_file}"
    )

    print(
        f"           {permutation_file}"
    )


# ============================================================
# Plot holiday means
# ============================================================

def plot_holiday_means(
    stats,
):

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.bar(
        stats[
            "holiday_group"
        ],
        stats[
            "mean"
        ],
    )

    ax.set_title(
        "Mean Property-Damage Claim Count "
        "by Public-Holiday Status"
    )

    ax.set_xlabel(
        "Target-day holiday status"
    )

    ax.set_ylabel(
        "Mean claims per day"
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
        "holiday_mean_claim_count.png"
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
# Plot holiday effect within weekday
# ============================================================

def plot_holiday_by_weekday(
    by_weekday,
):

    x = np.arange(
        len(by_weekday)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        x - width / 2,
        by_weekday[
            "nonholiday_mean"
        ],
        width,
        label="Non-holiday",
    )

    ax.bar(
        x + width / 2,
        by_weekday[
            "holiday_mean"
        ],
        width,
        label="Public holiday",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        by_weekday[
            "weekday_name"
        ],
        rotation=30,
    )

    ax.set_title(
        "Property-Damage Claims: "
        "Holiday vs Non-Holiday Within Weekday"
    )

    ax.set_xlabel(
        "Target-day weekday"
    )

    ax.set_ylabel(
        "Mean claims per day"
    )

    ax.legend()

    ax.grid(
        True,
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        /
        "holiday_by_weekday_mean_claim_count.png"
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
        "[analysis] Week 2 / Step 5"
    )

    print(
        "[analysis] Public-holiday effect"
    )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    train = load_training_data()

    print_dataset_info(
        train
    )

    # --------------------------------------------------------
    # Descriptive holiday statistics
    # --------------------------------------------------------

    stats = calculate_holiday_statistics(
        train
    )

    effect = calculate_holiday_effect(
        train
    )

    print_holiday_effect(
        stats,
        effect,
    )

    # --------------------------------------------------------
    # Holiday effect within weekday
    # --------------------------------------------------------

    by_weekday = (
        calculate_holiday_by_weekday(
            train
        )
    )

    print_holiday_by_weekday(
        by_weekday
    )

    # --------------------------------------------------------
    # Incremental explanatory value
    # --------------------------------------------------------

    (
        model_comparison,
        delta_r_squared,
        delta_adjusted_r_squared,
    ) = compare_incremental_holiday_value(
        train
    )

    print_model_comparison(
        model_comparison,
        delta_r_squared,
        delta_adjusted_r_squared,
    )

    # --------------------------------------------------------
    # Corrected stratified permutation test
    # --------------------------------------------------------

    permutation = (
        stratified_holiday_permutation_test(
            train,
            n_permutations=N_PERMUTATIONS,
            random_seed=RANDOM_SEED,
        )
    )

    print_stratified_permutation_test(
        permutation
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        stats=stats,
        effect=effect,
        by_weekday=by_weekday,
        model_comparison=model_comparison,
        delta_r_squared=delta_r_squared,
        delta_adjusted_r_squared=(
            delta_adjusted_r_squared
        ),
        permutation=permutation,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_holiday_means(
        stats
    )

    plot_holiday_by_weekday(
        by_weekday
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()