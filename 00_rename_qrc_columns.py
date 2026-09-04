"""
00_rename_qrc_columns.py
------------------------

Create a canonical English-column version of qrc_daily_dataset.csv.

The original CSV is never modified. Only column names are changed; all row
values and row ordering are preserved.

Default usage
-------------
python 00_rename_qrc_columns.py --input qrc_daily_dataset.csv

This creates:
    qrc_daily_dataset_en.csv

Optional explicit output:
    python 00_rename_qrc_columns.py \
        --input qrc_daily_dataset.csv \
        --output qrc_daily_dataset_en.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMN_MAPPING = {
    "date": "date",
    "day_of_week": "day_of_week",
    "day_of_week_name": "day_of_week_name",
    "is_weekend": "is_weekend",
    "is_public_holiday": "is_public_holiday",
    "is_working_day": "is_working_day",

    "claim_count_total": "total_claim_count",
    "claim_count_maj": "property_damage_claim_count",
    "claim_count_nem": "injury_claim_count",
    "claim_count_usly": "loss_of_profit_claim_count",

    "policy_live_total": "active_policy_count",

    "policy_age0": "policy_age_0y_count",
    "policy_age1to3": "policy_age_1_3y_count",
    "policy_age4plus": "policy_age_4plus_y_count",

    "share_age0": "policy_age_0y_share",
    "share_age1to3": "policy_age_1_3y_share",
    "share_age4plus": "policy_age_4plus_y_share",

    "legal_entity_count": "legal_entity_policy_count",
    "share_legal_entity": "legal_entity_policy_share",

    "vehicle_avg_age_years": "average_vehicle_age_years",
}


def default_output_path(input_path: Path) -> Path:
    """Return qrc_daily_dataset_en.csv next to the input file."""
    return input_path.with_name(f"{input_path.stem}_en{input_path.suffix}")


def validate_input_columns(df: pd.DataFrame) -> None:
    """Verify that the source CSV has exactly the expected project columns."""
    expected = set(COLUMN_MAPPING.keys())
    actual = set(df.columns)

    missing = expected - actual
    unexpected = actual - expected

    if missing:
        raise ValueError(
            "Input CSV is missing expected columns:\n"
            f"    {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            "Input CSV contains unexpected columns. "
            "Review the source before renaming:\n"
            f"    {sorted(unexpected)}"
        )


def validate_output(df_original: pd.DataFrame, df_renamed: pd.DataFrame) -> None:
    """Ensure renaming did not change the data shape or row count."""
    if len(df_original) != len(df_renamed):
        raise RuntimeError("Row count changed during column renaming.")

    if df_original.shape[1] != df_renamed.shape[1]:
        raise RuntimeError("Column count changed during column renaming.")

    expected_columns = list(COLUMN_MAPPING.values())

    if list(df_renamed.columns) != expected_columns:
        raise RuntimeError(
            "Renamed columns are not in the expected canonical order."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename QRC insurance CSV columns to canonical English names."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original qrc_daily_dataset.csv",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path. Default: <input_stem>_en.csv "
            "in the same directory."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    if input_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {input_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(input_path)
    )

    if output_path == input_path:
        raise ValueError(
            "Refusing to overwrite the original source CSV. "
            "Choose a different output path."
        )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --overwrite if you intentionally want to replace it."
        )

    print(f"[rename] Reading: {input_path}")

    df = pd.read_csv(input_path)

    validate_input_columns(df)

    renamed = df.rename(columns=COLUMN_MAPPING)

    validate_output(df, renamed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    renamed.to_csv(
        output_path,
        index=False,
    )

    print(f"[rename] Rows: {len(renamed)}")
    print(f"[rename] Columns: {len(renamed.columns)}")
    print(f"[rename] Written: {output_path}")

    print("\n[rename] Column mapping:")
    for old, new in COLUMN_MAPPING.items():
        if old == new:
            print(f"  {old}")
        else:
            print(f"  {old}  ->  {new}")


if __name__ == "__main__":
    main()
