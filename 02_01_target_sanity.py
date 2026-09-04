"""
02_01_target_sanity.py
----------------------

Structural validation and descriptive statistics for the
property-damage claim forecasting target.

The script reads the already prepared forecasting samples
from the database.

Important:
    - structural checks use the complete forecasting dataset;
    - detailed descriptive statistics used for model development
      focus primarily on the TRAINING set;
    - no scaling or normalization is performed here.
"""

import pandas as pd

from sqlalchemy import text

from db_config import engine


# ============================================================
# Configuration
# ============================================================

FORECAST_DATASET_NAME = "property_damage_next_day_v1"


# ============================================================
# Load forecasting samples
# ============================================================

QUERY = text("""
SELECT
    fs.id,
    fs.input_date,
    fs.target_date,
    fs.property_damage_claim_count_t,
    fs.active_policy_count_t,
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


def load_data():
    with engine.connect() as connection:
        df = pd.read_sql(
            QUERY,
            connection,
            params={
                "dataset_name": FORECAST_DATASET_NAME
            },
        )

    df["input_date"] = pd.to_datetime(
        df["input_date"]
    )

    df["target_date"] = pd.to_datetime(
        df["target_date"]
    )

    return df


# ============================================================
# Structural checks
# ============================================================

def structural_checks(df):

    print()
    print("=" * 70)
    print("STRUCTURAL CHECKS")
    print("=" * 70)

    print(f"Total samples: {len(df)}")

    print(
        f"Target date range: "
        f"{df['target_date'].min().date()} "
        f"-> "
        f"{df['target_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Duplicate target dates
    # --------------------------------------------------------

    duplicate_count = (
        df["target_date"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate target dates: "
        f"{duplicate_count}"
    )

    # --------------------------------------------------------
    # Missing target values
    # --------------------------------------------------------

    missing_target = (
        df["target_property_damage_claim_count"]
        .isna()
        .sum()
    )

    print(
        f"Missing target values: "
        f"{missing_target}"
    )

    # --------------------------------------------------------
    # Negative target values
    # --------------------------------------------------------

    negative_target = (
        df["target_property_damage_claim_count"]
        < 0
    ).sum()

    print(
        f"Negative target values: "
        f"{negative_target}"
    )

    # --------------------------------------------------------
    # Verify input_date -> target_date = exactly one day
    # --------------------------------------------------------

    day_difference = (
        df["target_date"]
        -
        df["input_date"]
    ).dt.days

    invalid_horizon = (
        day_difference != 1
    ).sum()

    print(
        f"Invalid one-day horizons: "
        f"{invalid_horizon}"
    )

    # --------------------------------------------------------
    # Verify consecutive TARGET dates
    # --------------------------------------------------------

    target_gaps = (
        df["target_date"]
        .diff()
        .dropna()
        .dt.days
    )

    gap_count = (
        target_gaps != 1
    ).sum()

    print(
        f"Gaps in target-date sequence: "
        f"{gap_count}"
    )

    # --------------------------------------------------------
    # Split counts
    # --------------------------------------------------------

    print()
    print("Split counts:")
    print(
        df["split"]
        .value_counts()
        .sort_index()
    )

    expected = {
        "train": 1095,
        "validation": 365,
        "test": 232,
    }

    actual = (
        df["split"]
        .value_counts()
        .to_dict()
    )

    print()

    if actual == expected:
        print("Split sizes: OK")
    else:
        print("WARNING: unexpected split sizes")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")

    print("=" * 70)


# ============================================================
# Descriptive statistics
# ============================================================

def descriptive_statistics(df):

    train = df[
        df["split"] == "train"
    ].copy()

    target = (
        train[
            "target_property_damage_claim_count"
        ]
        .astype(float)
    )

    print()
    print("=" * 70)
    print("TRAINING TARGET STATISTICS")
    print("=" * 70)

    print(
        f"N:                 "
        f"{len(target)}"
    )

    print(
        f"Mean:              "
        f"{target.mean():.4f}"
    )

    print(
        f"Median:            "
        f"{target.median():.4f}"
    )

    print(
        f"Std deviation:     "
        f"{target.std(ddof=1):.4f}"
    )

    print(
        f"Variance:          "
        f"{target.var(ddof=1):.4f}"
    )

    print(
        f"Minimum:           "
        f"{target.min():.0f}"
    )

    print(
        f"Maximum:           "
        f"{target.max():.0f}"
    )

    # --------------------------------------------------------
    # Zero frequency
    # --------------------------------------------------------

    zero_count = (
        target == 0
    ).sum()

    zero_fraction = (
        zero_count / len(target)
    )

    print()
    print(
        f"Zero-count days:   "
        f"{zero_count}"
    )

    print(
        f"Zero fraction:     "
        f"{100 * zero_fraction:.4f}%"
    )

    # --------------------------------------------------------
    # Dispersion
    # --------------------------------------------------------

    mean = target.mean()
    variance = target.var(ddof=1)

    dispersion = (
        variance / mean
        if mean != 0
        else float("nan")
    )

    print()
    print(
        f"Dispersion ratio "
        f"(variance / mean): "
        f"{dispersion:.4f}"
    )

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

    quantiles = target.quantile(
        [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )

    print()
    print("Quantiles:")
    print(quantiles)

    print("=" * 70)


# ============================================================
# Yearly statistics
# ============================================================

def yearly_statistics(df):

    train = df[
        df["split"] == "train"
    ].copy()

    train["year"] = (
        train["target_date"]
        .dt.year
    )

    target_column = (
        "target_property_damage_claim_count"
    )

    yearly = (
        train
        .groupby("year")[target_column]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            variance="var",
            minimum="min",
            maximum="max",
        )
    )
    yearly["dispersion_ratio"] = (
        yearly["variance"] / yearly["mean"]
    )   

    print()
    print("=" * 70)
    print("TRAINING TARGET BY YEAR")
    print("=" * 70)

    print(
        yearly.round(3)
    )

    print("=" * 70)

    return yearly


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"[analysis] Loading forecast dataset: "
        f"{FORECAST_DATASET_NAME}"
    )

    df = load_data()

    if df.empty:
        raise RuntimeError(
            "No forecasting samples were found. "
            "Check FORECAST_DATASET_NAME."
        )

    structural_checks(df)
    descriptive_statistics(df)
    yearly_statistics(df)


if __name__ == "__main__":
    main()