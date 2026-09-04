from pathlib import Path

import numpy as np
import pandas as pd

from db_config import engine


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Load canonical forecasting sample
# ---------------------------------------------------------------------

query = """
SELECT
    input_date,
    target_date,
    property_damage_claim_count_t,
    active_policy_count_t,
    day_of_week_t_plus_1,
    is_public_holiday_t_plus_1,
    target_property_damage_claim_count,
    split
FROM qrc_forecast_sample
ORDER BY input_date
"""

df = pd.read_sql(query, engine)

df["input_date"] = pd.to_datetime(df["input_date"])
df["target_date"] = pd.to_datetime(df["target_date"])

numeric_columns = [
    "property_damage_claim_count_t",
    "active_policy_count_t",
    "day_of_week_t_plus_1",
    "is_public_holiday_t_plus_1",
    "target_property_damage_claim_count",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

df = (
    df
    .sort_values("input_date")
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

audit_rows = []


def add_check(check, passed, details):

    audit_rows.append(
        {
            "check": check,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
            "details": details,
        }
    )


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

print("=" * 78)
print("WEEK 2 - STEP 12: LEAKAGE AND FEATURE-AVAILABILITY AUDIT")
print("=" * 78)

print(f"\nTotal samples: {len(df)}")

print(
    f"Overall input-date range: "
    f"{df['input_date'].min().date()} "
    f"to {df['input_date'].max().date()}"
)

print(
    f"Overall target-date range: "
    f"{df['target_date'].min().date()} "
    f"to {df['target_date'].max().date()}"
)


# =====================================================================
# CHECK 1
# Exact one-day horizon
# =====================================================================

expected_target_date = (
    df["input_date"]
    + pd.Timedelta(days=1)
)

horizon_ok = (
    df["target_date"]
    == expected_target_date
)

n_horizon_fail = int(
    (~horizon_ok).sum()
)

add_check(
    "One-day forecast horizon",
    n_horizon_fail == 0,
    (
        f"{n_horizon_fail} rows where "
        "target_date != input_date + 1 day"
    ),
)


# =====================================================================
# CHECK 2
# Duplicate input dates
# =====================================================================

n_duplicate_input = int(
    df["input_date"]
    .duplicated()
    .sum()
)

add_check(
    "Unique input_date",
    n_duplicate_input == 0,
    f"{n_duplicate_input} duplicated input dates",
)


# =====================================================================
# CHECK 3
# Duplicate target dates
# =====================================================================

n_duplicate_target = int(
    df["target_date"]
    .duplicated()
    .sum()
)

add_check(
    "Unique target_date",
    n_duplicate_target == 0,
    f"{n_duplicate_target} duplicated target dates",
)


# =====================================================================
# CHECK 4
# Missing required values
# =====================================================================

required_columns = [
    "input_date",
    "target_date",
    "property_damage_claim_count_t",
    "active_policy_count_t",
    "day_of_week_t_plus_1",
    "is_public_holiday_t_plus_1",
    "target_property_damage_claim_count",
    "split",
]

missing_counts = (
    df[required_columns]
    .isna()
    .sum()
)

n_missing = int(
    missing_counts.sum()
)

add_check(
    "No missing required values",
    n_missing == 0,
    f"{n_missing} missing values in required fields",
)


# =====================================================================
# CHECK 5
# Daily continuity
# =====================================================================

input_diff = (
    df["input_date"]
    .diff()
    .dropna()
)

gap_mask = (
    input_diff
    != pd.Timedelta(days=1)
)

n_gaps = int(
    gap_mask.sum()
)

add_check(
    "Daily sample continuity",
    n_gaps == 0,
    f"{n_gaps} non-daily transitions",
)


# =====================================================================
# CHECK 6
# C_t -> C_(t+1) temporal consistency
# =====================================================================
#
# Current row:
#
# input_date = t
# target      = C_(t+1)
#
# Next row:
#
# input_date = t+1
# C_t column  = C_(t+1)
#
# Therefore, for consecutive rows:
#
# target[i] == C_t[i+1]
# ---------------------------------------------------------------------

next_input_date = (
    df["input_date"]
    .shift(-1)
)

next_C_t = (
    df[
        "property_damage_claim_count_t"
    ]
    .shift(-1)
)

consecutive = (
    next_input_date
    == (
        df["input_date"]
        + pd.Timedelta(days=1)
    )
)

claim_alignment_ok = (
    df[
        "target_property_damage_claim_count"
    ]
    == next_C_t
)

claim_alignment_tested = (
    consecutive
    & next_C_t.notna()
)

claim_alignment_fail = (
    claim_alignment_tested
    & (~claim_alignment_ok)
)

n_claim_alignment_tested = int(
    claim_alignment_tested.sum()
)

n_claim_alignment_fail = int(
    claim_alignment_fail.sum()
)

add_check(
    "C_t / C_(t+1) temporal alignment",
    n_claim_alignment_fail == 0,
    (
        f"{n_claim_alignment_fail} mismatches "
        f"among {n_claim_alignment_tested} "
        "consecutive row pairs"
    ),
)


# =====================================================================
# CHECK 7
# Day-of-week alignment
# =====================================================================
#
# Assumption used in our dataset:
#
# Monday = 1
# Tuesday = 2
# ...
# Sunday = 7
#
# pandas:
# Monday = 0 ... Sunday = 6
# ---------------------------------------------------------------------

expected_dow = (
    df["target_date"]
    .dt.dayofweek
    + 1
)

dow_ok = (
    df["day_of_week_t_plus_1"]
    == expected_dow
)

n_dow_fail = int(
    (~dow_ok).sum()
)

add_check(
    "D_t+1 matches target_date",
    n_dow_fail == 0,
    (
        f"{n_dow_fail} weekday mismatches "
        "assuming Monday=1,...,Sunday=7"
    ),
)


# =====================================================================
# CHECK 8
# Holiday variable must be binary
# =====================================================================

holiday_values = sorted(
    df[
        "is_public_holiday_t_plus_1"
    ]
    .dropna()
    .unique()
    .tolist()
)

holiday_binary = set(
    holiday_values
).issubset(
    {0, 1}
)

add_check(
    "H_t+1 is binary",
    holiday_binary,
    f"Observed values: {holiday_values}",
)


# =====================================================================
# CHECK 9
# Claim counts non-negative
# =====================================================================

negative_Ct = int(
    (
        df[
            "property_damage_claim_count_t"
        ] < 0
    ).sum()
)

negative_target = int(
    (
        df[
            "target_property_damage_claim_count"
        ] < 0
    ).sum()
)

add_check(
    "Claim counts non-negative",
    (
        negative_Ct == 0
        and negative_target == 0
    ),
    (
        f"C_t negatives={negative_Ct}, "
        f"target negatives={negative_target}"
    ),
)


# =====================================================================
# CHECK 10
# Exposure positive
# =====================================================================

invalid_exposure = int(
    (
        df[
            "active_policy_count_t"
        ] <= 0
    ).sum()
)

add_check(
    "P_t positive",
    invalid_exposure == 0,
    (
        f"{invalid_exposure} rows with "
        "active_policy_count_t <= 0"
    ),
)


# =====================================================================
# Split audit
# =====================================================================

print("\n" + "=" * 78)
print("CHRONOLOGICAL SPLIT AUDIT")
print("=" * 78)

split_summary = (
    df.groupby("split")
    .agg(
        n=("target_date", "size"),
        input_start=("input_date", "min"),
        input_end=("input_date", "max"),
        target_start=("target_date", "min"),
        target_end=("target_date", "max"),
    )
    .reset_index()
)

print(
    split_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Expected split names
# ---------------------------------------------------------------------

required_splits = {
    "train",
    "validation",
    "test",
}

observed_splits = set(
    df["split"]
    .dropna()
    .unique()
)

split_names_ok = (
    observed_splits
    == required_splits
)

add_check(
    "Expected split labels",
    split_names_ok,
    f"Observed: {sorted(observed_splits)}",
)


# ---------------------------------------------------------------------
# Extract split blocks
# ---------------------------------------------------------------------

train = df[
    df["split"] == "train"
]

validation = df[
    df["split"] == "validation"
]

test = df[
    df["split"] == "test"
]


if (
    len(train) > 0
    and len(validation) > 0
    and len(test) > 0
):

    train_val_ok = (
        train["target_date"].max()
        <
        validation["target_date"].min()
    )

    val_test_ok = (
        validation["target_date"].max()
        <
        test["target_date"].min()
    )

    chronology_ok = (
        train_val_ok
        and val_test_ok
    )

else:

    chronology_ok = False


add_check(
    "Chronological split ordering",
    chronology_ok,
    (
        "Requires "
        "max(train) < min(validation) "
        "<= max(validation) < min(test)"
    ),
)


# =====================================================================
# CHECK 11
# A date must belong to only one split
# =====================================================================

target_split_counts = (
    df.groupby("target_date")[
        "split"
    ]
    .nunique()
)

overlap_target_dates = int(
    (
        target_split_counts > 1
    ).sum()
)

add_check(
    "No target-date split overlap",
    overlap_target_dates == 0,
    (
        f"{overlap_target_dates} target dates "
        "assigned to multiple splits"
    ),
)


# =====================================================================
# CHECK 12
# Split changes must be chronological
# =====================================================================

split_order = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

split_numeric = (
    df["split"]
    .map(split_order)
)

backward_split_transitions = int(
    (
        split_numeric.diff()
        .dropna()
        < 0
    ).sum()
)

add_check(
    "No backward split transitions",
    backward_split_transitions == 0,
    (
        f"{backward_split_transitions} "
        "backward split transitions"
    ),
)


# =====================================================================
# Print audit table
# =====================================================================

audit = pd.DataFrame(
    audit_rows
)


print("\n" + "=" * 78)
print("AUTOMATED LEAKAGE AUDIT")
print("=" * 78)

print(
    audit.to_string(
        index=False
    )
)


# =====================================================================
# Feature availability table
# =====================================================================
#
# This is a conceptual availability audit.
#
# "known_at_forecast_time" describes whether the variable is allowed
# when forecasting C_(t+1) at the end of day t.
# ---------------------------------------------------------------------

availability = pd.DataFrame(
    [
        {
            "feature": "C_t",
            "source_time": "t",
            "known_at_forecast_time": True,
            "reason": (
                "Current-day observed claim count; "
                "forecast is issued after day t is finalized"
            ),
        },
        {
            "feature": "D_t+1",
            "source_time": "t+1 calendar",
            "known_at_forecast_time": True,
            "reason": (
                "Target-day weekday is deterministic "
                "and known in advance"
            ),
        },
        {
            "feature": "P_t",
            "source_time": "t",
            "known_at_forecast_time": True,
            "reason": (
                "Active-policy exposure measured "
                "at forecast origin t"
            ),
        },
        {
            "feature": "H_t+1",
            "source_time": "t+1 calendar",
            "known_at_forecast_time": True,
            "reason": (
                "Public-holiday calendar is known "
                "before the target day"
            ),
        },
        {
            "feature": "C_t+1",
            "source_time": "t+1",
            "known_at_forecast_time": False,
            "reason": (
                "This is the prediction target and "
                "must never enter the feature matrix"
            ),
        },
    ]
)


print("\n" + "=" * 78)
print("FEATURE AVAILABILITY AT FORECAST ORIGIN t")
print("=" * 78)

print(
    availability.to_string(
        index=False
    )
)


# =====================================================================
# Overall result
# =====================================================================

n_fail = int(
    (audit["status"] == "FAIL")
    .sum()
)

print("\n" + "=" * 78)
print("OVERALL LEAKAGE-AUDIT RESULT")
print("=" * 78)

if n_fail == 0:

    print(
        "PASS: all automated temporal and "
        "split checks passed."
    )

else:

    print(
        f"FAIL: {n_fail} automated checks failed."
    )


# =====================================================================
# Save
# =====================================================================

audit.to_csv(
    RESULTS_DIR /
    "02_12_leakage_audit.csv",
    index=False,
)

split_summary.to_csv(
    RESULTS_DIR /
    "02_12_split_audit.csv",
    index=False,
)

availability.to_csv(
    RESULTS_DIR /
    "02_12_feature_availability.csv",
    index=False,
)


# Save problematic rows if they exist.
problems = df[
    (~horizon_ok)
    | claim_alignment_fail
    | (~dow_ok)
].copy()

problems.to_csv(
    RESULTS_DIR /
    "02_12_temporal_alignment_problems.csv",
    index=False,
)


print("\nSaved:")
print(
    " results/02_12_leakage_audit.csv"
)
print(
    " results/02_12_split_audit.csv"
)
print(
    " results/02_12_feature_availability.csv"
)
print(
    " results/02_12_temporal_alignment_problems.csv"
)

print(
    "\nStep 12 leakage audit finished."
)