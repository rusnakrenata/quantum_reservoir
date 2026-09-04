"""
02_03_autocorrelation.py
------------------------

Autocorrelation analysis of the property-damage claim count.

The script:

1. Loads only the TRAINING target from the database.
2. Calculates lagged autocorrelation manually for lags 1...30.
3. Explicitly calculates:
       - aligned vectors X and Y
       - their means
       - deviations from the means
       - covariance numerator
       - squared-deviation sums
       - denominator
       - final Pearson autocorrelation
4. Verifies the result using pandas.corr().
5. Prints detailed calculations for important lags:
       1, 7, 14, 21, 28
6. Saves all autocorrelation values to CSV.
7. Creates an autocorrelation plot.

Important
---------
This is the autocorrelation of the RAW training target.

No detrending, exposure normalization, or differencing is performed yet.

Therefore the calculated autocorrelation may contain contributions from:
    - true temporal memory
    - weekly seasonality
    - long-term trend

These effects will be separated in later analysis.
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

MAX_LAG = 30

IMPORTANT_LAGS = [
    1,
    7,
    14,
    21,
    28,
]

OUTPUT_DIR = Path("results/week2")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SQL query
# ============================================================

QUERY = text("""
SELECT
    fs.target_date,
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
    Load the forecasting samples and retain only the
    training-period target.
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
            f"No rows found for dataset "
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

    return train


# ============================================================
# Manual autocorrelation calculation
# ============================================================

def calculate_autocorrelation_for_lag(
    values,
    lag,
):
    """
    Calculate Pearson autocorrelation manually.

    For lag k:

        X = [C_{k+1}, ..., C_T]
        Y = [C_1, ..., C_{T-k}]

    Then:

             sum((X-Xbar)(Y-Ybar))
        r = ------------------------
             sqrt(sum((X-Xbar)^2)
                  sum((Y-Ybar)^2))

    Parameters
    ----------
    values : numpy.ndarray
        Time-series values in chronological order.

    lag : int
        Lag k.

    Returns
    -------
    dict
        All intermediate and final calculation values.
    """

    if lag <= 0:
        raise ValueError(
            "Lag must be positive."
        )

    if lag >= len(values):
        raise ValueError(
            "Lag must be smaller than series length."
        )

    # --------------------------------------------------------
    # Construct aligned vectors
    # --------------------------------------------------------
    #
    # Example lag = 1:
    #
    # X = C_2, C_3, ..., C_T
    # Y = C_1, C_2, ..., C_{T-1}
    #
    # Example lag = 7:
    #
    # X = C_8, C_9, ..., C_T
    # Y = C_1, C_2, ..., C_{T-7}
    #

    x = values[lag:]
    y = values[:-lag]

    n_pairs = len(x)

    # --------------------------------------------------------
    # Means
    # --------------------------------------------------------

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # --------------------------------------------------------
    # Deviations from means
    # --------------------------------------------------------

    deviation_x = x - mean_x
    deviation_y = y - mean_y

    # --------------------------------------------------------
    # Numerator
    #
    # sum[(x_i - xbar)(y_i - ybar)]
    # --------------------------------------------------------

    numerator_terms = (
        deviation_x
        *
        deviation_y
    )

    numerator = np.sum(
        numerator_terms
    )

    # --------------------------------------------------------
    # Squared deviations
    # --------------------------------------------------------

    squared_deviation_x = (
        deviation_x ** 2
    )

    squared_deviation_y = (
        deviation_y ** 2
    )

    sum_squared_x = np.sum(
        squared_deviation_x
    )

    sum_squared_y = np.sum(
        squared_deviation_y
    )

    # --------------------------------------------------------
    # Denominator
    #
    # sqrt(
    #     sum((x-xbar)^2)
    #     *
    #     sum((y-ybar)^2)
    # )
    # --------------------------------------------------------

    denominator = np.sqrt(
        sum_squared_x
        *
        sum_squared_y
    )

    # --------------------------------------------------------
    # Autocorrelation
    # --------------------------------------------------------

    if denominator == 0:

        autocorrelation = np.nan

    else:

        autocorrelation = (
            numerator
            /
            denominator
        )

    # --------------------------------------------------------
    # Verification using pandas Pearson correlation
    # --------------------------------------------------------

    pandas_correlation = pd.Series(x).corr(
        pd.Series(y)
    )

    difference = (
        autocorrelation
        -
        pandas_correlation
    )

    return {
        "lag": lag,

        "n_pairs": n_pairs,

        "mean_current": mean_x,
        "mean_lagged": mean_y,

        "numerator": numerator,

        "sum_squared_current": sum_squared_x,
        "sum_squared_lagged": sum_squared_y,

        "denominator": denominator,

        "autocorrelation": autocorrelation,

        "pandas_correlation": pandas_correlation,

        "verification_difference": difference,
    }


# ============================================================
# Calculate all lags
# ============================================================

def calculate_all_autocorrelations(
    df,
    max_lag,
):
    """
    Calculate autocorrelation for lags 1...max_lag.
    """

    values = (
        df[TARGET_COLUMN]
        .astype(float)
        .to_numpy()
    )

    results = []

    for lag in range(
        1,
        max_lag + 1,
    ):

        result = (
            calculate_autocorrelation_for_lag(
                values=values,
                lag=lag,
            )
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    return results_df


# ============================================================
# Print dataset information
# ============================================================

def print_dataset_info(df):

    print()
    print("=" * 75)
    print("TRAINING DATA")
    print("=" * 75)

    print(
        f"Dataset: "
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
        f"Mean target: "
        f"{df[TARGET_COLUMN].mean():.4f}"
    )

    print("=" * 75)


# ============================================================
# Print all autocorrelations
# ============================================================

def print_autocorrelation_table(
    results_df,
):
    """
    Print compact ACF table for lags 1...30.
    """

    print()
    print("=" * 75)
    print("AUTOCORRELATION BY LAG")
    print("=" * 75)

    compact = results_df[
        [
            "lag",
            "n_pairs",
            "autocorrelation",
        ]
    ].copy()

    print(
        compact
        .round(4)
        .to_string(
            index=False
        )
    )

    print("=" * 75)


# ============================================================
# Print detailed calculations for important lags
# ============================================================

def print_detailed_calculations(
    results_df,
):
    """
    Print the exact components of the Pearson calculation
    for lags 1, 7, 14, 21, and 28.
    """

    print()
    print("=" * 90)
    print("DETAILED AUTOCORRELATION CALCULATIONS")
    print("=" * 90)

    for lag in IMPORTANT_LAGS:

        row = (
            results_df[
                results_df["lag"] == lag
            ]
            .iloc[0]
        )

        print()
        print("-" * 90)

        print(
            f"LAG k = {lag}"
        )

        print("-" * 90)

        print(
            f"Number of aligned pairs:"
            f"       {int(row['n_pairs'])}"
        )

        print(
            f"Mean current values X:"
            f"          {row['mean_current']:.6f}"
        )

        print(
            f"Mean lagged values Y:"
            f"           {row['mean_lagged']:.6f}"
        )

        print()

        print(
            "Numerator:"
        )

        print(
            "  sum[(X-Xbar)(Y-Ybar)]"
        )

        print(
            f"  = {row['numerator']:.6f}"
        )

        print()

        print(
            "Current squared deviations:"
        )

        print(
            "  sum[(X-Xbar)^2]"
        )

        print(
            f"  = "
            f"{row['sum_squared_current']:.6f}"
        )

        print()

        print(
            "Lagged squared deviations:"
        )

        print(
            "  sum[(Y-Ybar)^2]"
        )

        print(
            f"  = "
            f"{row['sum_squared_lagged']:.6f}"
        )

        print()

        print(
            "Denominator:"
        )

        print(
            "  sqrt("
            "sum[(X-Xbar)^2] "
            "* "
            "sum[(Y-Ybar)^2]"
            ")"
        )

        print(
            f"  = {row['denominator']:.6f}"
        )

        print()

        print(
            "Final autocorrelation:"
        )

        print(
            "  rho(k) = numerator / denominator"
        )

        print(
            f"  rho({lag}) = "
            f"{row['autocorrelation']:.6f}"
        )

        print()

        print(
            "Pandas verification:"
        )

        print(
            f"  corr(X,Y) = "
            f"{row['pandas_correlation']:.6f}"
        )

        print(
            f"  difference = "
            f"{row['verification_difference']:.12f}"
        )

    print()
    print("=" * 90)


# ============================================================
# Print important lags summary
# ============================================================

def print_important_lags(
    results_df,
):

    important = (
        results_df[
            results_df["lag"].isin(
                IMPORTANT_LAGS
            )
        ]
        [
            [
                "lag",
                "autocorrelation",
            ]
        ]
    )

    print()
    print("=" * 60)
    print("IMPORTANT TEMPORAL LAGS")
    print("=" * 60)

    print(
        important
        .round(4)
        .to_string(
            index=False
        )
    )

    print("=" * 60)


# ============================================================
# Find strongest lags
# ============================================================

def print_strongest_lags(
    results_df,
    number=10,
):
    """
    Rank lags by absolute autocorrelation.
    """

    ranked = results_df.copy()

    ranked[
        "absolute_autocorrelation"
    ] = (
        ranked["autocorrelation"]
        .abs()
    )

    ranked = (
        ranked
        .sort_values(
            "absolute_autocorrelation",
            ascending=False,
        )
        .head(number)
    )

    print()
    print("=" * 60)
    print(
        f"TOP {number} LAGS BY "
        f"ABSOLUTE AUTOCORRELATION"
    )
    print("=" * 60)

    print(
        ranked[
            [
                "lag",
                "autocorrelation",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print("=" * 60)


# ============================================================
# Save results
# ============================================================

def save_results(
    results_df,
):

    output_file = (
        OUTPUT_DIR
        /
        "target_training_autocorrelation.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "[analysis] Autocorrelation "
        "calculations saved to:"
    )

    print(
        f"           {output_file}"
    )


# ============================================================
# Plot autocorrelation
# ============================================================

def plot_autocorrelation(
    results_df,
):
    """
    Plot autocorrelation against lag.
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.stem(
        results_df["lag"],
        results_df["autocorrelation"],
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "Autocorrelation of Property-Damage Claim Count "
        "— Training Period"
    )

    ax.set_xlabel(
        "Lag (days)"
    )

    ax.set_ylabel(
        "Autocorrelation"
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
        "target_training_autocorrelation.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(
        "[analysis] Autocorrelation "
        "plot saved to:"
    )

    print(
        f"           {output_file}"
    )

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "[analysis] Week 2 / Step 3"
    )

    print(
        "[analysis] Autocorrelation analysis"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train = load_training_data()

    print_dataset_info(
        train
    )

    # --------------------------------------------------------
    # Calculate autocorrelations
    # --------------------------------------------------------

    results = (
        calculate_all_autocorrelations(
            df=train,
            max_lag=MAX_LAG,
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_autocorrelation_table(
        results
    )

    print_important_lags(
        results
    )

    print_strongest_lags(
        results
    )

    # --------------------------------------------------------
    # Explicit mathematical calculation
    # --------------------------------------------------------

    print_detailed_calculations(
        results
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        results
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_autocorrelation(
        results
    )



# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()