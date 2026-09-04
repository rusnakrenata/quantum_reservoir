
"""
01_prepare_qrc_dataset.py
-------------------------

Create and populate the initial database tables for the QRC insurance
forecasting project using the canonical English-column CSV.

Expected input
--------------
qrc_daily_dataset_en.csv

Primary analysis window
-----------------------
2022-01-01 ... 2026-08-20

One-day-ahead target
--------------------
property_damage_claim_count(t+1)

Nested semantic feature sets
----------------------------
F2:
    property_damage_claim_count(t)
    day_of_week(t+1)

F3:
    property_damage_claim_count(t)
    day_of_week(t+1)
    active_policy_count(t)

F4:
    property_damage_claim_count(t)
    day_of_week(t+1)
    active_policy_count(t)
    is_public_holiday(t+1)

Chronological split by TARGET DATE
----------------------------------
train:
    2022-01-02 ... 2024-12-31     1095 samples

validation:
    2025-01-01 ... 2025-12-31      365 samples

test:
    2026-01-01 ... 2026-08-20      232 samples

Database design
---------------
qrc_dataset_source
    Source/provenance metadata.

qrc_insurance_daily
    Canonical English daily source observations. All 2069 rows are preserved.

qrc_forecast_dataset
    Defines the one-day-ahead forecasting problem and chronological split.

qrc_feature_set
    Stores F2, F3, and F4 separately from the samples.

qrc_forecast_sample
    Stores each aligned (t, t+1) sample exactly once, with all four candidate
    semantic feature values. A future model/run selects one qrc_feature_set.

Important
---------
Do NOT store standardized values or quantum angles in these source/sample
tables. Scalers and encodings are fitted/configured per experiment and belong
in later experiment/run configuration tables.

Usage
-----
python 01_prepare_qrc_dataset.py --csv qrc_daily_dataset_en.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    delete,
    func,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

from db_config import engine


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_START_DATE = date(2022, 1, 1)
DEFAULT_CUTOFF_DATE = date(2026, 8, 20)

TRAIN_END = date(2024, 12, 31)
VALIDATION_END = date(2025, 12, 31)

FORECAST_HORIZON_DAYS = 1

FORECAST_DATASET_NAME = "property_damage_next_day_v1"
TARGET_COLUMN = "property_damage_claim_count"

FEATURE_SETS = {
    "F2": [
        "property_damage_claim_count_t",
        "day_of_week_t_plus_1",
    ],
    "F3": [
        "property_damage_claim_count_t",
        "day_of_week_t_plus_1",
        "active_policy_count_t",
    ],
    "F4": [
        "property_damage_claim_count_t",
        "day_of_week_t_plus_1",
        "active_policy_count_t",
        "is_public_holiday_t_plus_1",
    ],
}


# =============================================================================
# SQLAlchemy base
# =============================================================================

class Base(DeclarativeBase):
    pass


# =============================================================================
# Table 1: source dataset metadata
# =============================================================================

class DatasetSource(Base):
    __tablename__ = "qrc_dataset_source"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    dataset_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    source_file: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    source_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )

    source_start_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    source_end_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    source_row_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    claim_observation_cutoff: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    daily_rows = relationship(
        "InsuranceDaily",
        back_populates="source",
        cascade="all, delete-orphan",
    )

    forecast_datasets = relationship(
        "ForecastDataset",
        back_populates="source",
        cascade="all, delete-orphan",
    )


# =============================================================================
# Table 2: canonical English daily data
# =============================================================================

class InsuranceDaily(Base):
    __tablename__ = "qrc_insurance_daily"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    source_id: Mapped[int] = mapped_column(
        ForeignKey(
            "qrc_dataset_source.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    day_of_week: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    day_of_week_name: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    is_weekend: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    is_public_holiday: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    is_working_day: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    total_claim_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    property_damage_claim_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    injury_claim_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    loss_of_profit_claim_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    active_policy_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    policy_age_0y_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    policy_age_1_3y_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    policy_age_4plus_y_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    policy_age_0y_share: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    policy_age_1_3y_share: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    policy_age_4plus_y_share: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    legal_entity_policy_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    legal_entity_policy_share: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    average_vehicle_age_years: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    source = relationship(
        "DatasetSource",
        back_populates="daily_rows",
    )

    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "date",
            name="uq_qrc_daily_source_date",
        ),
        Index(
            "ix_qrc_daily_date",
            "date",
        ),
    )


# =============================================================================
# Table 3: forecasting dataset definition
# =============================================================================

class ForecastDataset(Base):
    __tablename__ = "qrc_forecast_dataset"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    source_id: Mapped[int] = mapped_column(
        ForeignKey(
            "qrc_dataset_source.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    dataset_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        unique=True,
    )

    start_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    cutoff_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    forecast_horizon_days: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    target_column: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    train_end_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    validation_end_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    source = relationship(
        "DatasetSource",
        back_populates="forecast_datasets",
    )

    samples = relationship(
        "ForecastSample",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )

    feature_sets = relationship(
        "FeatureSet",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


# =============================================================================
# Table 4: feature-set definitions
# =============================================================================

class FeatureSet(Base):
    __tablename__ = "qrc_feature_set"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey(
            "qrc_forecast_dataset.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    feature_set_name: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    semantic_dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    features_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    dataset = relationship(
        "ForecastDataset",
        back_populates="feature_sets",
    )

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "feature_set_name",
            name="uq_qrc_feature_set_dataset_name",
        ),
    )


# =============================================================================
# Table 5: aligned one-day forecasting samples
# =============================================================================

class ForecastSample(Base):
    __tablename__ = "qrc_forecast_sample"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey(
            "qrc_forecast_dataset.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    input_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    target_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    # Candidate semantic features.
    # Feature-set membership is defined in qrc_feature_set.
    property_damage_claim_count_t: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    active_policy_count_t: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    day_of_week_t_plus_1: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    is_public_holiday_t_plus_1: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    # One-day-ahead target.
    target_property_damage_claim_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # train / validation / test
    split: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    dataset = relationship(
        "ForecastDataset",
        back_populates="samples",
    )

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "target_date",
            name="uq_qrc_forecast_dataset_target_date",
        ),
        Index(
            "ix_qrc_forecast_target_date",
            "target_date",
        ),
        Index(
            "ix_qrc_forecast_split",
            "split",
        ),
    )


# =============================================================================
# Helpers
# =============================================================================

REQUIRED_COLUMNS = {
    "date",
    "day_of_week",
    "day_of_week_name",
    "is_weekend",
    "is_public_holiday",
    "is_working_day",
    "total_claim_count",
    "property_damage_claim_count",
    "injury_claim_count",
    "loss_of_profit_claim_count",
    "active_policy_count",
    "policy_age_0y_count",
    "policy_age_1_3y_count",
    "policy_age_4plus_y_count",
    "policy_age_0y_share",
    "policy_age_1_3y_share",
    "policy_age_4plus_y_share",
    "legal_entity_policy_count",
    "legal_entity_policy_share",
    "average_vehicle_age_years",
}


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 for exact source-file provenance."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for chunk in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def python_value(value):
    """Convert pandas/numpy scalar values to ordinary Python values."""
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date()

    if isinstance(value, np.generic):
        return value.item()

    return value


def validate_source_dataframe(df: pd.DataFrame) -> None:
    """Validate canonical English CSV structure and daily continuity."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    unexpected = set(df.columns) - REQUIRED_COLUMNS

    if missing:
        raise ValueError(
            "CSV is missing required canonical English columns:\n"
            f"    {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            "CSV contains unexpected columns:\n"
            f"    {sorted(unexpected)}"
        )

    if df["date"].isna().any():
        raise ValueError("CSV contains invalid or missing dates.")

    if df["date"].duplicated().any():
        duplicate_dates = (
            df.loc[df["date"].duplicated(), "date"]
            .dt.date
            .tolist()
        )

        raise ValueError(
            f"Duplicate dates found: {duplicate_dates}"
        )

    df_sorted = df.sort_values("date")

    differences = (
        df_sorted["date"]
        .diff()
        .dropna()
        .dt.days
    )

    if not (differences == 1).all():
        raise ValueError(
            "The source dataset is not a contiguous daily time series."
        )

    if not df["day_of_week"].between(1, 7).all():
        raise ValueError("day_of_week must use ISO values 1..7.")

    # Verify the property residual identity from the source methodology.
    residual = (
        df["total_claim_count"]
        - df["injury_claim_count"]
        - df["loss_of_profit_claim_count"]
    )

    if not (residual == df["property_damage_claim_count"]).all():
        mismatch_count = int(
            (residual != df["property_damage_claim_count"]).sum()
        )

        raise ValueError(
            "property_damage_claim_count does not match "
            "total - injury - loss_of_profit for "
            f"{mismatch_count} row(s)."
        )


def split_from_target_date(target_date: date) -> str:
    """Assign train/validation/test by target date."""
    if target_date <= TRAIN_END:
        return "train"

    if target_date <= VALIDATION_END:
        return "validation"

    return "test"


def feature_set_description(name: str) -> str:
    """Human-readable explanation for each nested semantic feature set."""
    descriptions = {
        "F2": (
            "Minimal temporal/calendar feature set: current property-damage "
            "claim count plus target-day weekday."
        ),
        "F3": (
            "F2 plus current active-policy exposure."
        ),
        "F4": (
            "F3 plus target-day public-holiday indicator."
        ),
    }

    return descriptions[name]


# =============================================================================
# Create SQL tables
# =============================================================================

def create_tables() -> None:
    Base.metadata.create_all(engine)
    print("[database] Tables created / verified.")


# =============================================================================
# Source metadata
# =============================================================================

def get_or_create_source(
    session: Session,
    csv_path: Path,
    df: pd.DataFrame,
    cutoff_date: date,
) -> DatasetSource:
    file_hash = calculate_sha256(csv_path)

    existing = session.execute(
        select(DatasetSource).where(
            DatasetSource.source_sha256 == file_hash
        )
    ).scalar_one_or_none()

    if existing is not None:
        print(
            f"[database] Source already registered: id={existing.id}"
        )
        return existing

    source = DatasetSource(
        dataset_name="qrc_daily_dataset_en",
        source_file=str(csv_path.resolve()),
        source_sha256=file_hash,
        source_start_date=df["date"].min().date(),
        source_end_date=df["date"].max().date(),
        source_row_count=len(df),
        claim_observation_cutoff=cutoff_date,
        notes=(
            "Canonical English-column version of the daily PZP passenger-car "
            "dataset. Column names were translated from the original source; "
            "values are unchanged. Rows after the modelling cutoff are "
            "preserved for provenance but excluded from forecasting samples."
        ),
    )

    session.add(source)
    session.flush()

    print(
        f"[database] Registered source dataset: id={source.id}"
    )

    return source


# =============================================================================
# Load canonical daily rows
# =============================================================================

def load_daily_rows(
    session: Session,
    source: DatasetSource,
    df: pd.DataFrame,
) -> None:
    existing_count = session.scalar(
        select(func.count())
        .select_from(InsuranceDaily)
        .where(
            InsuranceDaily.source_id == source.id
        )
    )

    if existing_count and existing_count > 0:
        print(
            "[database] Canonical daily data already loaded "
            f"for source {source.id}: {existing_count} rows."
        )
        return

    mappings = []

    for _, row in df.iterrows():
        mappings.append(
            {
                "source_id": source.id,
                "date": row["date"].date(),
                "day_of_week": int(row["day_of_week"]),
                "day_of_week_name": str(row["day_of_week_name"]),
                "is_weekend": bool(row["is_weekend"]),
                "is_public_holiday": bool(row["is_public_holiday"]),
                "is_working_day": bool(row["is_working_day"]),

                "total_claim_count": int(row["total_claim_count"]),
                "property_damage_claim_count": int(
                    row["property_damage_claim_count"]
                ),
                "injury_claim_count": int(row["injury_claim_count"]),
                "loss_of_profit_claim_count": int(
                    row["loss_of_profit_claim_count"]
                ),

                "active_policy_count": python_value(
                    row["active_policy_count"]
                ),

                "policy_age_0y_count": python_value(
                    row["policy_age_0y_count"]
                ),
                "policy_age_1_3y_count": python_value(
                    row["policy_age_1_3y_count"]
                ),
                "policy_age_4plus_y_count": python_value(
                    row["policy_age_4plus_y_count"]
                ),

                "policy_age_0y_share": python_value(
                    row["policy_age_0y_share"]
                ),
                "policy_age_1_3y_share": python_value(
                    row["policy_age_1_3y_share"]
                ),
                "policy_age_4plus_y_share": python_value(
                    row["policy_age_4plus_y_share"]
                ),

                "legal_entity_policy_count": python_value(
                    row["legal_entity_policy_count"]
                ),
                "legal_entity_policy_share": python_value(
                    row["legal_entity_policy_share"]
                ),

                "average_vehicle_age_years": python_value(
                    row["average_vehicle_age_years"]
                ),
            }
        )

    session.bulk_insert_mappings(
        InsuranceDaily,
        mappings,
    )

    print(
        f"[database] Loaded {len(mappings)} canonical daily rows."
    )


# =============================================================================
# Forecast dataset definition
# =============================================================================

def get_or_create_forecast_dataset(
    session: Session,
    source: DatasetSource,
    start_date: date,
    cutoff_date: date,
) -> ForecastDataset:
    existing = session.execute(
        select(ForecastDataset).where(
            ForecastDataset.dataset_name
            == FORECAST_DATASET_NAME
        )
    ).scalar_one_or_none()

    if existing is not None:
        # Keep metadata synchronized with the current script configuration.
        existing.source_id = source.id
        existing.start_date = start_date
        existing.cutoff_date = cutoff_date
        existing.forecast_horizon_days = FORECAST_HORIZON_DAYS
        existing.target_column = TARGET_COLUMN
        existing.train_end_date = TRAIN_END
        existing.validation_end_date = VALIDATION_END
        existing.notes = (
            "Primary one-day-ahead property-damage forecasting dataset. "
            "Samples are stored once; F2/F3/F4 are defined separately."
        )

        session.flush()

        print(
            "[database] Forecast dataset already exists; "
            f"metadata updated: id={existing.id}"
        )

        return existing

    dataset = ForecastDataset(
        source_id=source.id,
        dataset_name=FORECAST_DATASET_NAME,
        start_date=start_date,
        cutoff_date=cutoff_date,
        forecast_horizon_days=FORECAST_HORIZON_DAYS,
        target_column=TARGET_COLUMN,
        train_end_date=TRAIN_END,
        validation_end_date=VALIDATION_END,
        notes=(
            "Primary one-day-ahead property-damage forecasting dataset. "
            "Samples are stored once; F2/F3/F4 are defined separately."
        ),
    )

    session.add(dataset)
    session.flush()

    print(
        f"[database] Created forecast dataset: id={dataset.id}"
    )

    return dataset


# =============================================================================
# Feature-set definitions
# =============================================================================

def upsert_feature_sets(
    session: Session,
    dataset: ForecastDataset,
) -> None:
    for name, features in FEATURE_SETS.items():
        existing = session.execute(
            select(FeatureSet).where(
                FeatureSet.dataset_id == dataset.id,
                FeatureSet.feature_set_name == name,
            )
        ).scalar_one_or_none()

        features_json = json.dumps(
            features,
            ensure_ascii=False,
        )

        if existing is None:
            row = FeatureSet(
                dataset_id=dataset.id,
                feature_set_name=name,
                semantic_dimension=len(features),
                features_json=features_json,
                description=feature_set_description(name),
            )

            session.add(row)

        else:
            existing.semantic_dimension = len(features)
            existing.features_json = features_json
            existing.description = feature_set_description(name)

    session.flush()

    print(
        "[database] Feature sets F2, F3, and F4 created / updated."
    )


# =============================================================================
# Generate supervised one-day samples
# =============================================================================

def rebuild_forecast_samples(
    session: Session,
    dataset: ForecastDataset,
    df: pd.DataFrame,
    start_date: date,
    cutoff_date: date,
) -> None:
    working = df[
        (df["date"].dt.date >= start_date)
        &
        (df["date"].dt.date <= cutoff_date)
    ].copy()

    working = (
        working
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(working) < 2:
        raise ValueError(
            "Not enough rows to build one-day-ahead samples."
        )

    # Rebuild deterministically so rerunning the script is safe.
    session.execute(
        delete(ForecastSample).where(
            ForecastSample.dataset_id == dataset.id
        )
    )

    session.flush()

    samples = []

    for i in range(len(working) - 1):
        row_t = working.iloc[i]
        row_t1 = working.iloc[i + 1]

        input_date = row_t["date"].date()
        target_date = row_t1["date"].date()

        if (target_date - input_date).days != 1:
            raise ValueError(
                "Non-consecutive forecasting pair: "
                f"{input_date} -> {target_date}"
            )

        active_policy_count = python_value(
            row_t["active_policy_count"]
        )

        if active_policy_count is None:
            raise ValueError(
                "active_policy_count missing on "
                f"{input_date}"
            )

        samples.append(
            {
                "dataset_id": dataset.id,

                "input_date": input_date,
                "target_date": target_date,

                "property_damage_claim_count_t": int(
                    row_t["property_damage_claim_count"]
                ),

                "active_policy_count_t": int(
                    active_policy_count
                ),

                # Target-day calendar values are known in advance.
                "day_of_week_t_plus_1": int(
                    row_t1["day_of_week"]
                ),

                "is_public_holiday_t_plus_1": bool(
                    row_t1["is_public_holiday"]
                ),

                "target_property_damage_claim_count": int(
                    row_t1["property_damage_claim_count"]
                ),

                "split": split_from_target_date(
                    target_date
                ),
            }
        )

    session.bulk_insert_mappings(
        ForecastSample,
        samples,
    )

    print(
        f"[database] Generated {len(samples)} forecasting samples."
    )


# =============================================================================
# Verification
# =============================================================================

def verify_forecast_dataset(
    session: Session,
    dataset: ForecastDataset,
) -> None:
    rows = session.execute(
        select(
            ForecastSample.split,
            func.count(ForecastSample.id),
            func.min(ForecastSample.target_date),
            func.max(ForecastSample.target_date),
        )
        .where(
            ForecastSample.dataset_id == dataset.id
        )
        .group_by(
            ForecastSample.split
        )
        .order_by(
            ForecastSample.split
        )
    ).all()

    print()
    print("=" * 78)
    print("FORECAST DATASET VERIFICATION")
    print("=" * 78)

    total = 0

    for split, count, min_date, max_date in rows:
        total += count

        print(
            f"{split:<12}"
            f"{count:>6} samples   "
            f"{min_date} -> {max_date}"
        )

    print("-" * 78)
    print(
        f"{'TOTAL':<12}"
        f"{total:>6} samples"
    )
    print("=" * 78)

    expected = {
        "train": 1095,
        "validation": 365,
        "test": 232,
    }

    actual = {
        split: count
        for split, count, _, _ in rows
    }

    if actual != expected:
        raise RuntimeError(
            "Unexpected split sizes.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    feature_rows = session.execute(
        select(
            FeatureSet.feature_set_name,
            FeatureSet.semantic_dimension,
            FeatureSet.features_json,
        )
        .where(
            FeatureSet.dataset_id == dataset.id
        )
        .order_by(
            FeatureSet.semantic_dimension
        )
    ).all()

    print()
    print("FEATURE SETS")
    print("-" * 78)

    for name, dimension, features_json in feature_rows:
        print(
            f"{name}: d={dimension}  "
            f"{features_json}"
        )

    print()
    print("[database] Dataset verification successful.")
    print()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create and populate the initial QRC insurance "
            "forecasting database tables."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to canonical qrc_daily_dataset_en.csv",
    )

    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE.isoformat(),
        help=(
            "First daily observation used for forecasting "
            f"(default: {DEFAULT_START_DATE})"
        ),
    )

    parser.add_argument(
        "--cutoff-date",
        default=DEFAULT_CUTOFF_DATE.isoformat(),
        help=(
            "Last observed date used for forecasting "
            f"(default: {DEFAULT_CUTOFF_DATE})"
        ),
    )

    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    start_date = date.fromisoformat(
        args.start_date
    )

    cutoff_date = date.fromisoformat(
        args.cutoff_date
    )

    if start_date >= cutoff_date:
        raise ValueError(
            "start-date must be earlier than cutoff-date."
        )

    print(f"[dataset] Reading: {csv_path}")

    df = pd.read_csv(
        csv_path,
        parse_dates=["date"],
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    validate_source_dataframe(df)

    if start_date < df["date"].min().date():
        raise ValueError(
            "Requested start date precedes source data."
        )

    if cutoff_date > df["date"].max().date():
        raise ValueError(
            "Requested cutoff date exceeds source data."
        )

    print(
        f"[dataset] Source rows: {len(df)}"
    )

    print(
        "[dataset] Source range: "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    print(
        "[dataset] Forecast range: "
        f"{start_date} -> {cutoff_date}"
    )

    create_tables()

    with Session(engine) as session:
        try:
            source = get_or_create_source(
                session=session,
                csv_path=csv_path,
                df=df,
                cutoff_date=cutoff_date,
            )

            load_daily_rows(
                session=session,
                source=source,
                df=df,
            )

            forecast_dataset = get_or_create_forecast_dataset(
                session=session,
                source=source,
                start_date=start_date,
                cutoff_date=cutoff_date,
            )

            upsert_feature_sets(
                session=session,
                dataset=forecast_dataset,
            )

            rebuild_forecast_samples(
                session=session,
                dataset=forecast_dataset,
                df=df,
                start_date=start_date,
                cutoff_date=cutoff_date,
            )

            verify_forecast_dataset(
                session=session,
                dataset=forecast_dataset,
            )

            session.commit()

            print(
                "[database] Transaction committed successfully."
            )

        except Exception:
            session.rollback()

            print(
                "[database] Transaction rolled back."
            )

            raise


if __name__ == "__main__":
    main()