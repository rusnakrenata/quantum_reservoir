"""
02_02_target_visualization.py
-----------------------------


Visual inspection of the property-damage claim time series.
For model-development EDA, only the TRAINING target is plotted.

The script:
    1. Loads the training forecasting samples from the database
    2. Calculates 7-day and 30-day rolling means
    3. Prints summary statistics of the rolling means
    4. Prints example calculated rolling-mean values
    5. Saves the rolling-mean values to CSV
    6. Plots:
        - daily property-damage claim count
        - 7-day rolling mean
        - 30-day rolling mean

Output:
    results/week2/target_training_timeseries.png
    results/week2/target_training_with_rolling_means.csv
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import text

from db_config import engine


# ============================================================
# Configuration
# ============================================================

FORECAST_DATASET_NAME = "property_damage_next_day_v1"

OUTPUT_DIR = Path("results")

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
    Load forecasting samples from the database
    and keep only the training split.
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
            f"No rows found for forecasting dataset "
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
# Calculate rolling means
# ============================================================

def calculate_rolling_means(df):
    """
    Calculate 7-day and 30-day rolling means.

    For a rolling window of size w:

        rolling_mean_t =
            (C_t + C_{t-1} + ... + C_{t-w+1}) / w

    Therefore:

        7-day rolling mean
            = current day + previous 6 days

        30-day rolling mean
            = current day + previous 29 days

    The first 6 rows of the 7-day rolling mean are NaN,
    because a complete 7-day window does not yet exist.

    The first 29 rows of the 30-day rolling mean are NaN.
    """

    target = (
        "target_property_damage_claim_count"
    )

    # --------------------------------------------------------
    # 7-day rolling mean
    # --------------------------------------------------------

    df["rolling_mean_7d"] = (
        df[target]
        .rolling(
            window=7,
            min_periods=7,
        )
        .mean()
    )

    # --------------------------------------------------------
    # 30-day rolling mean
    # --------------------------------------------------------

    df["rolling_mean_30d"] = (
        df[target]
        .rolling(
            window=30,
            min_periods=30,
        )
        .mean()
    )

    return df


# ============================================================
# Print rolling-mean summary
# ============================================================

def print_rolling_summary(df):
    """
    Print descriptive statistics for the calculated
    7-day and 30-day rolling means.
    """

    print()
    print("=" * 70)
    print("ROLLING-MEAN SUMMARY")
    print("=" * 70)

    print()
    print("7-day rolling mean:")
    print("-" * 70)

    print(
        df["rolling_mean_7d"]
        .describe()
        .round(3)
    )

    print()

    print("30-day rolling mean:")
    print("-" * 70)

    print(
        df["rolling_mean_30d"]
        .describe()
        .round(3)
    )

    print("=" * 70)


# ============================================================
# Print example rolling calculations
# ============================================================

def print_rolling_examples(df):
    """
    Print example rows where both rolling means are available.
    """

    print()
    print("=" * 90)
    print("EXAMPLE ROLLING-MEAN VALUES")
    print("=" * 90)

    columns = [
        "target_date",
        "target_property_damage_claim_count",
        "rolling_mean_7d",
        "rolling_mean_30d",
    ]

    valid = df.dropna(
        subset=[
            "rolling_mean_7d",
            "rolling_mean_30d",
        ]
    )

    print(
        valid[columns]
        .head(10)
        .round(
            {
                "rolling_mean_7d": 3,
                "rolling_mean_30d": 3,
            }
        )
        .to_string(
            index=False
        )
    )

    print("=" * 90)


# ============================================================
# Save rolling data
# ============================================================

def save_rolling_data(df):
    """
    Save the daily target together with the calculated
    rolling means to a CSV file.
    """

    output_file = (
        OUTPUT_DIR
        /
        "target_training_with_rolling_means.csv"
    )

    columns = [
        "target_date",
        "target_property_damage_claim_count",
        "rolling_mean_7d",
        "rolling_mean_30d",
    ]

    df[columns].to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        f"[analysis] Rolling-mean data saved to:"
    )

    print(
        f"           {output_file}"
    )


# ============================================================
# Plot target and rolling means
# ============================================================

def plot_target(df):
    """
    Plot:
        - raw daily claim count
        - 7-day rolling mean
        - 30-day rolling mean
    """

    target = (
        "target_property_damage_claim_count"
    )

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    # --------------------------------------------------------
    # Raw daily observations
    # --------------------------------------------------------

    ax.plot(
        df["target_date"],
        df[target],
        label="Daily claim count",
        alpha=0.35,
        linewidth=0.9,
    )

    # --------------------------------------------------------
    # 7-day rolling mean
    # --------------------------------------------------------

    ax.plot(
        df["target_date"],
        df["rolling_mean_7d"],
        label="7-day rolling mean",
        linewidth=1.5,
    )

    # --------------------------------------------------------
    # 30-day rolling mean
    # --------------------------------------------------------

    ax.plot(
        df["target_date"],
        df["rolling_mean_30d"],
        label="30-day rolling mean",
        linewidth=2.0,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_title(
        "Property-Damage Claim Count — Training Period"
    )

    ax.set_xlabel(
        "Target date"
    )

    ax.set_ylabel(
        "Claims per day"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.2,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save plot
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        /
        "target_training_timeseries.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(
        f"[analysis] Plot saved to:"
    )

    print(
        f"           {output_file}"
    )

    plt.show()


# ============================================================
# Print basic dataset information
# ============================================================

def print_dataset_info(df):
    """
    Print basic information about the loaded training data.
    """

    print()
    print("=" * 70)
    print("TRAINING DATA")
    print("=" * 70)

    print(
        f"Forecast dataset: "
        f"{FORECAST_DATASET_NAME}"
    )

    print(
        f"Training observations: "
        f"{len(df)}"
    )

    print(
        f"Training date range: "
        f"{df['target_date'].min().date()} "
        f"-> "
        f"{df['target_date'].max().date()}"
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "[analysis] Week 2 / Step 2"
    )

    print(
        "[analysis] Target visualization "
        "and rolling-mean calculation"
    )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    train = load_training_data()

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print_dataset_info(
        train
    )

    # --------------------------------------------------------
    # Calculate rolling means
    # --------------------------------------------------------

    train = calculate_rolling_means(
        train
    )

    # --------------------------------------------------------
    # Print numerical results
    # --------------------------------------------------------

    print_rolling_summary(
        train
    )

    print_rolling_examples(
        train
    )

    # --------------------------------------------------------
    # Save calculated values
    # --------------------------------------------------------

    save_rolling_data(
        train
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_target(
        train
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()