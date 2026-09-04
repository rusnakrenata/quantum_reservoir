"""
02_04_weekday_analysis.py
-------------------------

Analysis of the relationship between target-day weekday and the
next-day property-damage claim count.

Target
------
C_{t+1} =
    target_property_damage_claim_count

Calendar feature
----------------
D_{t+1} =
    day_of_week_t_plus_1

The weekday belongs to the TARGET DAY, which is exactly what we want
for one-day-ahead forecasting.

The weekday is known in advance, so using D_{t+1} does not constitute
data leakage.

This script:

1. Loads only TRAINING samples.
2. Calculates claim-count statistics for Monday ... Sunday.
3. Calculates dispersion ratio variance / mean for each weekday.
4. Compares weekdays with weekends.
5. Calculates the percentage weekend reduction.
6. Calculates Cohen's d for weekday/weekend difference.
7. Calculates one-way ANOVA components manually:
       SS_between
       SS_within
       MS_between
       MS_within
       F statistic
       eta-squared
8. Optionally calculates the ANOVA p-value using scipy.
9. Calculates a Kruskal-Wallis test.
10. Repeats weekday descriptive statistics after removing public holidays.
11. Saves numerical results to CSV.
12. Creates:
       - mean claim count by weekday
       - boxplot of claim distribution by weekday

Outputs
-------
results/weekday_claim_statistics.csv
results/weekday_claim_statistics_nonholiday.csv
results/weekend_vs_weekday_statistics.csv
results/weekday_anova_statistics.csv
results/weekday_mean_claim_count.png
results/weekday_claim_boxplot.png
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


# ISO weekday convention:
#
# 1 = Monday
# 2 = Tuesday
# ...
# 7 = Sunday

WEEKDAY_NAMES = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday",
}


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
    Load forecasting samples and retain only training data.
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
            f"No data found for forecast dataset "
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

    # --------------------------------------------------------
    # Validate weekday values
    # --------------------------------------------------------

    valid_weekdays = set(
        range(1, 8)
    )

    actual_weekdays = set(
        train[WEEKDAY_COLUMN]
        .dropna()
        .astype(int)
        .unique()
    )

    if not actual_weekdays.issubset(
        valid_weekdays
    ):

        raise ValueError(
            f"Unexpected weekday values: "
            f"{sorted(actual_weekdays)}"
        )

    # --------------------------------------------------------
    # Add human-readable weekday name
    # --------------------------------------------------------

    train["weekday_name"] = (
        train[WEEKDAY_COLUMN]
        .astype(int)
        .map(WEEKDAY_NAMES)
    )

    # --------------------------------------------------------
    # Define weekend
    #
    # Saturday = 6
    # Sunday   = 7
    # --------------------------------------------------------

    train["is_weekend"] = (
        train[WEEKDAY_COLUMN]
        .astype(int)
        .isin([6, 7])
    )

    return train


# ============================================================
# Dataset information
# ============================================================

def print_dataset_info(df):

    print()
    print("=" * 75)
    print("TRAINING DATA")
    print("=" * 75)

    print(
        f"Forecast dataset: "
        f"{FORECAST_DATASET_NAME}"
    )

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

    print(
        f"Overall target mean: "
        f"{df[TARGET_COLUMN].mean():.4f}"
    )

    print("=" * 75)


# ============================================================
# Weekday descriptive statistics
# ============================================================

def calculate_weekday_statistics(df):
    """
    Calculate descriptive statistics separately for
    Monday through Sunday.
    """

    weekday_stats = (
        df
        .groupby(
            [
                WEEKDAY_COLUMN,
                "weekday_name",
            ],
            observed=True,
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

    # --------------------------------------------------------
    # Poisson-style dispersion ratio
    #
    # D = variance / mean
    #
    # D = 1   -> equidispersion
    # D > 1   -> overdispersion
    # D < 1   -> underdispersion
    # --------------------------------------------------------

    weekday_stats[
        "dispersion_ratio"
    ] = (
        weekday_stats["variance"]
        /
        weekday_stats["mean"]
    )

    weekday_stats = (
        weekday_stats
        .sort_values(
            WEEKDAY_COLUMN
        )
        .reset_index(drop=True)
    )

    return weekday_stats


# ============================================================
# Print weekday statistics
# ============================================================

def print_weekday_statistics(
    weekday_stats,
    title,
):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    display = weekday_stats.copy()

    numeric_columns = [
        "mean",
        "median",
        "std",
        "variance",
        "dispersion_ratio",
    ]

    display[numeric_columns] = (
        display[numeric_columns]
        .round(3)
    )

    print(
        display.to_string(
            index=False
        )
    )

    print("=" * 100)


# ============================================================
# Weekday versus weekend comparison
# ============================================================

def calculate_weekend_comparison(df):
    """
    Compare Monday-Friday with Saturday-Sunday.
    """

    weekday_values = (
        df.loc[
            ~df["is_weekend"],
            TARGET_COLUMN,
        ]
        .astype(float)
        .to_numpy()
    )

    weekend_values = (
        df.loc[
            df["is_weekend"],
            TARGET_COLUMN,
        ]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    n_weekday = len(
        weekday_values
    )

    n_weekend = len(
        weekend_values
    )

    # --------------------------------------------------------
    # Means
    # --------------------------------------------------------

    mean_weekday = np.mean(
        weekday_values
    )

    mean_weekend = np.mean(
        weekend_values
    )

    # --------------------------------------------------------
    # Variances
    # --------------------------------------------------------

    variance_weekday = np.var(
        weekday_values,
        ddof=1,
    )

    variance_weekend = np.var(
        weekend_values,
        ddof=1,
    )

    # --------------------------------------------------------
    # Mean difference
    # --------------------------------------------------------

    mean_difference = (
        mean_weekday
        -
        mean_weekend
    )

    # --------------------------------------------------------
    # Relative weekend reduction
    #
    # Example:
    #
    # weekday mean = 12
    # weekend mean = 7
    #
    # reduction =
    #     (12 - 7) / 12 = 41.7 %
    # --------------------------------------------------------

    weekend_reduction_pct = (
        mean_difference
        /
        mean_weekday
        *
        100
    )

    # --------------------------------------------------------
    # Pooled standard deviation
    # --------------------------------------------------------

    pooled_variance = (
        (
            (n_weekday - 1)
            *
            variance_weekday
        )
        +
        (
            (n_weekend - 1)
            *
            variance_weekend
        )
    ) / (
        n_weekday
        +
        n_weekend
        -
        2
    )

    pooled_std = np.sqrt(
        pooled_variance
    )

    # --------------------------------------------------------
    # Cohen's d
    #
    # Standardized difference in means
    # --------------------------------------------------------

    if pooled_std == 0:

        cohens_d = np.nan

    else:

        cohens_d = (
            mean_weekday
            -
            mean_weekend
        ) / pooled_std

    comparison = pd.DataFrame(
        [
            {
                "group": "weekday",
                "count": n_weekday,
                "mean": mean_weekday,
                "variance": variance_weekday,
            },
            {
                "group": "weekend",
                "count": n_weekend,
                "mean": mean_weekend,
                "variance": variance_weekend,
            },
        ]
    )

    summary = {
        "mean_weekday": mean_weekday,
        "mean_weekend": mean_weekend,
        "mean_difference": mean_difference,
        "weekend_reduction_pct": (
            weekend_reduction_pct
        ),
        "cohens_d": cohens_d,
    }

    return comparison, summary


# ============================================================
# Print weekday/weekend comparison
# ============================================================

def print_weekend_comparison(
    comparison,
    summary,
):

    print()
    print("=" * 75)
    print("WEEKDAY VS WEEKEND")
    print("=" * 75)

    print(
        comparison
        .round(3)
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Weekday mean:           "
        f"{summary['mean_weekday']:.4f}"
    )

    print(
        f"Weekend mean:           "
        f"{summary['mean_weekend']:.4f}"
    )

    print(
        f"Difference:             "
        f"{summary['mean_difference']:.4f}"
    )

    print(
        f"Weekend reduction:      "
        f"{summary['weekend_reduction_pct']:.2f}%"
    )

    print(
        f"Cohen's d:              "
        f"{summary['cohens_d']:.4f}"
    )

    print("=" * 75)


# ============================================================
# Manual one-way ANOVA calculation
# ============================================================

def calculate_manual_anova(df):
    """
    Calculate one-way ANOVA manually.

    We ask whether the seven weekday groups have the same mean.

    H0:
        mu_Mon = mu_Tue = ... = mu_Sun

    H1:
        at least one weekday mean differs.

    The total variation is decomposed into:

        SS_total =
            SS_between + SS_within

    where:

        SS_between =
            sum_d n_d (mean_d - grand_mean)^2

    and:

        SS_within =
            sum_d sum_i
            (y_di - mean_d)^2
    """

    y = (
        df[TARGET_COLUMN]
        .astype(float)
        .to_numpy()
    )

    grand_mean = np.mean(
        y
    )

    n_total = len(
        y
    )

    groups = []

    for weekday in range(
        1,
        8,
    ):

        group_values = (
            df.loc[
                df[WEEKDAY_COLUMN]
                == weekday,
                TARGET_COLUMN,
            ]
            .astype(float)
            .to_numpy()
        )

        groups.append(
            group_values
        )

    k_groups = len(
        groups
    )

    # --------------------------------------------------------
    # Between-group sum of squares
    #
    # SS_between =
    #     sum n_d (mean_d - grand_mean)^2
    # --------------------------------------------------------

    ss_between = 0.0

    for group in groups:

        group_mean = np.mean(
            group
        )

        ss_between += (
            len(group)
            *
            (
                group_mean
                -
                grand_mean
            ) ** 2
        )

    # --------------------------------------------------------
    # Within-group sum of squares
    #
    # SS_within =
    #     sum sum(y_i - group_mean)^2
    # --------------------------------------------------------

    ss_within = 0.0

    for group in groups:

        group_mean = np.mean(
            group
        )

        ss_within += np.sum(
            (
                group
                -
                group_mean
            ) ** 2
        )

    # --------------------------------------------------------
    # Total sum of squares
    # --------------------------------------------------------

    ss_total = np.sum(
        (
            y
            -
            grand_mean
        ) ** 2
    )

    # --------------------------------------------------------
    # Degrees of freedom
    # --------------------------------------------------------

    df_between = (
        k_groups
        -
        1
    )

    df_within = (
        n_total
        -
        k_groups
    )

    # --------------------------------------------------------
    # Mean squares
    # --------------------------------------------------------

    ms_between = (
        ss_between
        /
        df_between
    )

    ms_within = (
        ss_within
        /
        df_within
    )

    # --------------------------------------------------------
    # F statistic
    # --------------------------------------------------------

    f_statistic = (
        ms_between
        /
        ms_within
    )

    # --------------------------------------------------------
    # Eta squared
    #
    # eta^2 =
    #     SS_between / SS_total
    #
    # Interpretation:
    #
    # fraction of total claim-count variation
    # associated with weekday grouping.
    # --------------------------------------------------------

    eta_squared = (
        ss_between
        /
        ss_total
    )

    # --------------------------------------------------------
    # Numerical identity check
    #
    # SS_total should equal:
    #
    # SS_between + SS_within
    # --------------------------------------------------------

    decomposition_error = (
        ss_total
        -
        (
            ss_between
            +
            ss_within
        )
    )

    result = {
        "n_total": n_total,
        "number_of_groups": k_groups,
        "grand_mean": grand_mean,

        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,

        "df_between": df_between,
        "df_within": df_within,

        "ms_between": ms_between,
        "ms_within": ms_within,

        "f_statistic": f_statistic,
        "eta_squared": eta_squared,

        "decomposition_error": (
            decomposition_error
        ),
    }

    return result


# ============================================================
# Optional significance tests
# ============================================================

def calculate_significance_tests(df):
    """
    Calculate ANOVA p-value and Kruskal-Wallis test.

    scipy is used only for the probability distributions.
    The ANOVA sums of squares and F statistic are calculated
    manually above.
    """

    try:

        from scipy.stats import (
            f,
            kruskal,
        )

    except ImportError:

        print()
        print(
            "[analysis] scipy is not installed."
        )

        print(
            "[analysis] p-values and "
            "Kruskal-Wallis test skipped."
        )

        return {
            "anova_p_value": np.nan,
            "kruskal_h": np.nan,
            "kruskal_p_value": np.nan,
        }

    # --------------------------------------------------------
    # Manual ANOVA F statistic
    # --------------------------------------------------------

    anova = calculate_manual_anova(
        df
    )

    f_statistic = (
        anova["f_statistic"]
    )

    df_between = (
        anova["df_between"]
    )

    df_within = (
        anova["df_within"]
    )

    # --------------------------------------------------------
    # p-value from F distribution
    # --------------------------------------------------------

    anova_p_value = f.sf(
        f_statistic,
        df_between,
        df_within,
    )

    # --------------------------------------------------------
    # Kruskal-Wallis
    #
    # Non-parametric test that does not require
    # normal distributions within each weekday.
    # --------------------------------------------------------

    groups = []

    for weekday in range(
        1,
        8,
    ):

        group_values = (
            df.loc[
                df[WEEKDAY_COLUMN]
                == weekday,
                TARGET_COLUMN,
            ]
            .astype(float)
            .to_numpy()
        )

        groups.append(
            group_values
        )

    kruskal_result = kruskal(
        *groups
    )

    return {
        "anova_p_value": (
            anova_p_value
        ),

        "kruskal_h": (
            kruskal_result.statistic
        ),

        "kruskal_p_value": (
            kruskal_result.pvalue
        ),
    }


# ============================================================
# Print ANOVA calculations
# ============================================================

def print_anova_results(
    anova,
    significance,
):

    print()
    print("=" * 90)
    print("ONE-WAY WEEKDAY ANOVA")
    print("=" * 90)

    print(
        f"N:                         "
        f"{anova['n_total']}"
    )

    print(
        f"Number of weekday groups:  "
        f"{anova['number_of_groups']}"
    )

    print(
        f"Grand mean:                "
        f"{anova['grand_mean']:.6f}"
    )

    print()

    print(
        "SS_between:"
    )

    print(
        "  sum n_d (mean_d - grand_mean)^2"
    )

    print(
        f"  = {anova['ss_between']:.6f}"
    )

    print()

    print(
        "SS_within:"
    )

    print(
        "  sum sum(y_di - mean_d)^2"
    )

    print(
        f"  = {anova['ss_within']:.6f}"
    )

    print()

    print(
        f"SS_total:                  "
        f"{anova['ss_total']:.6f}"
    )

    print(
        f"SS_between + SS_within:    "
        f"{anova['ss_between'] + anova['ss_within']:.6f}"
    )

    print(
        f"Decomposition error:       "
        f"{anova['decomposition_error']:.12f}"
    )

    print()

    print(
        f"df_between:                "
        f"{anova['df_between']}"
    )

    print(
        f"df_within:                 "
        f"{anova['df_within']}"
    )

    print()

    print(
        f"MS_between:                "
        f"{anova['ms_between']:.6f}"
    )

    print(
        f"MS_within:                 "
        f"{anova['ms_within']:.6f}"
    )

    print()

    print(
        f"F statistic:               "
        f"{anova['f_statistic']:.6f}"
    )

    print(
        f"ANOVA p-value:             "
        f"{significance['anova_p_value']:.12g}"
    )

    print()

    print(
        f"Eta squared:               "
        f"{anova['eta_squared']:.6f}"
    )

    print()

    print(
        "Eta squared represents the fraction "
        "of total target variance associated "
        "with weekday group differences."
    )

    print()

    print(
        f"Kruskal-Wallis H:          "
        f"{significance['kruskal_h']:.6f}"
    )

    print(
        f"Kruskal-Wallis p-value:    "
        f"{significance['kruskal_p_value']:.12g}"
    )

    print("=" * 90)


# ============================================================
# Save numerical results
# ============================================================

def save_results(
    weekday_stats,
    nonholiday_stats,
    weekend_comparison,
    weekend_summary,
    anova,
    significance,
):

    # --------------------------------------------------------
    # Weekday statistics
    # --------------------------------------------------------

    weekday_file = (
        OUTPUT_DIR
        /
        "weekday_claim_statistics.csv"
    )

    weekday_stats.to_csv(
        weekday_file,
        index=False,
    )

    # --------------------------------------------------------
    # Non-holiday weekday statistics
    # --------------------------------------------------------

    nonholiday_file = (
        OUTPUT_DIR
        /
        "weekday_claim_statistics_nonholiday.csv"
    )

    nonholiday_stats.to_csv(
        nonholiday_file,
        index=False,
    )

    # --------------------------------------------------------
    # Weekend comparison
    # --------------------------------------------------------

    weekend_file = (
        OUTPUT_DIR
        /
        "weekend_vs_weekday_statistics.csv"
    )

    weekend_output = (
        weekend_comparison.copy()
    )

    for key, value in (
        weekend_summary.items()
    ):

        weekend_output[key] = (
            value
        )

    weekend_output.to_csv(
        weekend_file,
        index=False,
    )

    # --------------------------------------------------------
    # ANOVA
    # --------------------------------------------------------

    anova_file = (
        OUTPUT_DIR
        /
        "weekday_anova_statistics.csv"
    )

    combined_anova = {
        **anova,
        **significance,
    }

    pd.DataFrame(
        [combined_anova]
    ).to_csv(
        anova_file,
        index=False,
    )

    print()
    print("[analysis] Saved:")

    print(
        f"           {weekday_file}"
    )

    print(
        f"           {nonholiday_file}"
    )

    print(
        f"           {weekend_file}"
    )

    print(
        f"           {anova_file}"
    )


# ============================================================
# Plot mean claim count by weekday
# ============================================================

def plot_weekday_means(
    weekday_stats,
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        weekday_stats[
            "weekday_name"
        ],
        weekday_stats[
            "mean"
        ],
    )

    ax.set_title(
        "Mean Property-Damage Claim Count by Weekday "
        "— Training Period"
    )

    ax.set_xlabel(
        "Target-day weekday"
    )

    ax.set_ylabel(
        "Mean claims per day"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.2,
    )

    plt.xticks(
        rotation=30,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        /
        "weekday_mean_claim_count.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(
        f"[analysis] Plot saved: "
        f"{output_file}"
    )

    plt.show()


# ============================================================
# Boxplot
# ============================================================

def plot_weekday_boxplot(
    df,
):

    weekday_data = []

    weekday_labels = []

    for weekday in range(
        1,
        8,
    ):

        values = (
            df.loc[
                df[WEEKDAY_COLUMN]
                == weekday,
                TARGET_COLUMN,
            ]
            .astype(float)
            .to_numpy()
        )

        weekday_data.append(
            values
        )

        weekday_labels.append(
            WEEKDAY_NAMES[
                weekday
            ]
        )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.boxplot(
        weekday_data,
        tick_labels=weekday_labels,
        showfliers=True,
    )

    ax.set_title(
        "Distribution of Property-Damage Claims "
        "by Weekday — Training Period"
    )

    ax.set_xlabel(
        "Target-day weekday"
    )

    ax.set_ylabel(
        "Claims per day"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.2,
    )

    plt.xticks(
        rotation=30,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        /
        "weekday_claim_boxplot.png"
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
        "[analysis] Week 2 / Step 4"
    )

    print(
        "[analysis] Weekday structure analysis"
    )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    train = load_training_data()

    print_dataset_info(
        train
    )

    # --------------------------------------------------------
    # Weekday statistics - all days
    # --------------------------------------------------------

    weekday_stats = (
        calculate_weekday_statistics(
            train
        )
    )

    print_weekday_statistics(
        weekday_stats,
        title=(
            "PROPERTY-DAMAGE CLAIMS BY "
            "TARGET-DAY WEEKDAY"
        ),
    )

    # --------------------------------------------------------
    # Repeat after removing public holidays
    #
    # This helps determine whether the weekday pattern is
    # present independently of holiday effects.
    # --------------------------------------------------------

    nonholiday = (
        train[
            ~train[HOLIDAY_COLUMN]
            .astype(bool)
        ]
        .copy()
    )

    nonholiday_stats = (
        calculate_weekday_statistics(
            nonholiday
        )
    )

    print_weekday_statistics(
        nonholiday_stats,
        title=(
            "PROPERTY-DAMAGE CLAIMS BY WEEKDAY "
            "— PUBLIC HOLIDAYS REMOVED"
        ),
    )

    # --------------------------------------------------------
    # Weekday vs weekend
    # --------------------------------------------------------

    (
        weekend_comparison,
        weekend_summary,
    ) = calculate_weekend_comparison(
        train
    )

    print_weekend_comparison(
        weekend_comparison,
        weekend_summary,
    )

    # --------------------------------------------------------
    # Manual ANOVA
    # --------------------------------------------------------

    anova = calculate_manual_anova(
        train
    )

    significance = (
        calculate_significance_tests(
            train
        )
    )

    print_anova_results(
        anova,
        significance,
    )

    # --------------------------------------------------------
    # Save numerical outputs
    # --------------------------------------------------------

    save_results(
        weekday_stats=weekday_stats,
        nonholiday_stats=nonholiday_stats,
        weekend_comparison=weekend_comparison,
        weekend_summary=weekend_summary,
        anova=anova,
        significance=significance,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_weekday_means(
        weekday_stats
    )

    plot_weekday_boxplot(
        train
    )

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()