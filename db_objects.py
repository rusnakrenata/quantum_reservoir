from __future__ import annotations

"""
All SQLAlchemy database objects for the QRC project.

Database layout
---------------
Dataset / forecasting objects:
    qrc_dataset_source
    qrc_insurance_daily
    qrc_forecast_dataset
    qrc_feature_set
    qrc_forecast_sample

Direct-QPU experiment objects:
    qrc_qpu_run
    qrc_qpu_run_csv_row
    qrc_qpu_run_json_artifact

This module contains ORM objects and database persistence helpers only.
Connection credentials / engine / Session live exclusively in db_config.py.
"""

import json
import math
import uuid
from datetime import date, datetime, timezone
from typing import Any

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
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

from db_config import engine


# =============================================================================
# Single declarative base for the whole QRC database
# =============================================================================

class Base(DeclarativeBase):
    pass


# =============================================================================
# Dataset / forecasting tables
# =============================================================================

class DatasetSource(Base):
    __tablename__ = "qrc_dataset_source"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_name: Mapped[str] = mapped_column(String(150), nullable=False)
    source_file: Mapped[str] = mapped_column(String(500), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    source_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    source_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    source_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    claim_observation_cutoff: Mapped[date] = mapped_column(Date, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
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


class InsuranceDaily(Base):
    __tablename__ = "qrc_insurance_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("qrc_dataset_source.id", ondelete="CASCADE"),
        nullable=False,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week_name: Mapped[str] = mapped_column(String(20), nullable=False)
    is_weekend: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_public_holiday: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_working_day: Mapped[bool] = mapped_column(Boolean, nullable=False)
    total_claim_count: Mapped[int] = mapped_column(Integer, nullable=False)
    property_damage_claim_count: Mapped[int] = mapped_column(Integer, nullable=False)
    injury_claim_count: Mapped[int] = mapped_column(Integer, nullable=False)
    loss_of_profit_claim_count: Mapped[int] = mapped_column(Integer, nullable=False)
    active_policy_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    policy_age_0y_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    policy_age_1_3y_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    policy_age_4plus_y_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    policy_age_0y_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    policy_age_1_3y_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    policy_age_4plus_y_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    legal_entity_policy_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    legal_entity_policy_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_vehicle_age_years: Mapped[float | None] = mapped_column(Float, nullable=True)

    source = relationship("DatasetSource", back_populates="daily_rows")

    __table_args__ = (
        UniqueConstraint("source_id", "date", name="uq_qrc_daily_source_date"),
        Index("ix_qrc_daily_date", "date"),
    )


class ForecastDataset(Base):
    __tablename__ = "qrc_forecast_dataset"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("qrc_dataset_source.id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_name: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    cutoff_date: Mapped[date] = mapped_column(Date, nullable=False)
    forecast_horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    target_column: Mapped[str] = mapped_column(String(100), nullable=False)
    train_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    validation_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    source = relationship("DatasetSource", back_populates="forecast_datasets")
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


class FeatureSet(Base):
    __tablename__ = "qrc_feature_set"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("qrc_forecast_dataset.id", ondelete="CASCADE"),
        nullable=False,
    )
    feature_set_name: Mapped[str] = mapped_column(String(30), nullable=False)
    semantic_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    features_json: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    dataset = relationship("ForecastDataset", back_populates="feature_sets")

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "feature_set_name",
            name="uq_qrc_feature_set_dataset_name",
        ),
    )


class ForecastSample(Base):
    __tablename__ = "qrc_forecast_sample"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("qrc_forecast_dataset.id", ondelete="CASCADE"),
        nullable=False,
    )
    input_date: Mapped[date] = mapped_column(Date, nullable=False)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    property_damage_claim_count_t: Mapped[int] = mapped_column(Integer, nullable=False)
    active_policy_count_t: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week_t_plus_1: Mapped[int] = mapped_column(Integer, nullable=False)
    is_public_holiday_t_plus_1: Mapped[bool] = mapped_column(Boolean, nullable=False)
    target_property_damage_claim_count: Mapped[int] = mapped_column(Integer, nullable=False)
    split: Mapped[str] = mapped_column(String(20), nullable=False)

    dataset = relationship("ForecastDataset", back_populates="samples")

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "target_date",
            name="uq_qrc_forecast_dataset_target_date",
        ),
        Index("ix_qrc_forecast_target_date", "target_date"),
        Index("ix_qrc_forecast_split", "split"),
    )


# =============================================================================
# Direct-QPU experiment tables
# =============================================================================

class QpuRun(Base):
    __tablename__ = "qrc_qpu_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_uuid: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    script_name: Mapped[str] = mapped_column(String(255), nullable=False)
    run_status: Mapped[str] = mapped_column(String(40), nullable=False, default="CREATED")

    started_at_utc: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    submitted_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    forecast_dataset_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    feature_set_name: Mapped[str | None] = mapped_column(String(30), nullable=True)
    evaluation_split: Mapped[str | None] = mapped_column(String(30), nullable=True)
    candidate_key: Mapped[str] = mapped_column(String(120), nullable=False)
    selected_by_rules: Mapped[str | None] = mapped_column(String(100), nullable=True)
    protocol: Mapped[str | None] = mapped_column(String(30), nullable=True)
    topology: Mapped[str | None] = mapped_column(String(30), nullable=True)
    window_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trotter_r: Mapped[int | None] = mapped_column(Integer, nullable=True)
    readout_name: Mapped[str | None] = mapped_column(String(120), nullable=True)

    alpha: Mapped[float | None] = mapped_column(Float, nullable=True)
    dt: Mapped[float | None] = mapped_column(Float, nullable=True)
    hx: Mapped[float | None] = mapped_column(Float, nullable=True)
    hy: Mapped[float | None] = mapped_column(Float, nullable=True)
    j_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    manifest_json: Mapped[str | None] = mapped_column(LONGTEXT, nullable=True)

    ridge_lambda: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_cv_rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    ideal_validation_rmse_full_2025: Mapped[float | None] = mapped_column(Float, nullable=True)
    ideal_validation_mae_full_2025: Mapped[float | None] = mapped_column(Float, nullable=True)
    ideal_validation_bias_full_2025: Mapped[float | None] = mapped_column(Float, nullable=True)

    backend_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    physical_layout_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    hardware_selection_json: Mapped[str | None] = mapped_column(LONGTEXT, nullable=True)

    max_2q_error_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_readout_error_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_1q_error_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_t1_us: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_t2_us: Mapped[float | None] = mapped_column(Float, nullable=True)
    compiled_duration_us: Mapped[float | None] = mapped_column(Float, nullable=True)
    r_t2: Mapped[float | None] = mapped_column(Float, nullable=True)

    shots_per_setting: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_validation_endpoints: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_circuits: Mapped[int | None] = mapped_column(Integer, nullable=True)
    full_2025: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    pilot_selection: Mapped[str | None] = mapped_column(String(120), nullable=True)
    optimization_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    seed_transpiler: Mapped[int | None] = mapped_column(Integer, nullable=True)

    median_cz_per_feature: Mapped[float | None] = mapped_column(Float, nullable=True)
    median_max_setting_depth: Mapped[float | None] = mapped_column(Float, nullable=True)
    median_feature_duration_us: Mapped[float | None] = mapped_column(Float, nullable=True)
    median_max_setting_duration_us: Mapped[float | None] = mapped_column(Float, nullable=True)
    scheduled_duration_times_shots_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    empirical_usage_estimate_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    job_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    job_status: Mapped[str | None] = mapped_column(String(80), nullable=True)

    # Backward-compatible usage field used by earlier Week-11 scripts.
    job_usage_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # IBM-reported physical execution timing.
    #
    # qpu_charge_time_seconds:
    #     Capacity/usage time charged by IBM; the time the QPU resource is
    #     locked for this workload.
    #
    # qpu_circuits_execution_time_seconds:
    #     IBM `circuits_execution_time_ns` converted to seconds.  This is the
    #     actual circuit-execution time reported for the QPU.
    #
    # qpu_running_wall_seconds:
    #     Difference between IBM job-metric `running` and `finished`
    #     timestamps.  This is useful for distinguishing QPU execution from
    #     the user's total wall-clock wait.
    qpu_charge_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    qpu_circuits_execution_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    qpu_running_wall_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    ibm_job_created_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ibm_job_running_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ibm_job_finished_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    job_metrics_json: Mapped[str | None] = mapped_column(LONGTEXT, nullable=True)
    service_usage_before_json: Mapped[str | None] = mapped_column(LONGTEXT, nullable=True)
    service_usage_after_json: Mapped[str | None] = mapped_column(LONGTEXT, nullable=True)

    qpu_rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    qpu_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    qpu_bias: Mapped[float | None] = mapped_column(Float, nullable=True)
    ideal_same_subset_rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    ideal_same_subset_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    ideal_same_subset_bias: Mapped[float | None] = mapped_column(Float, nullable=True)
    qpu_feature_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    qpu_feature_rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    z3_cross_setting_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_correlation_qpu_vs_ideal: Mapped[float | None] = mapped_column(Float, nullable=True)

    error_message: Mapped[str | None] = mapped_column(LONGTEXT, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_qrc_qpu_run_candidate_started", "candidate_key", "started_at_utc"),
        Index("ix_qrc_qpu_run_backend_started", "backend_name", "started_at_utc"),
    )


class QpuRunCsvRow(Base):
    __tablename__ = "qrc_qpu_run_csv_row"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_uuid: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    artifact_name: Mapped[str] = mapped_column(String(180), nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    endpoint: Mapped[int | None] = mapped_column(Integer, nullable=True)
    setting: Mapped[str | None] = mapped_column(String(80), nullable=True)
    row_json: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "run_uuid",
            "artifact_name",
            "row_index",
            name="uq_qrc_qpu_csv_run_artifact_row",
        ),
        Index("ix_qrc_qpu_csv_run_artifact", "run_uuid", "artifact_name"),
    )


class QpuRunJsonArtifact(Base):
    __tablename__ = "qrc_qpu_run_json_artifact"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_uuid: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    artifact_name: Mapped[str] = mapped_column(String(180), nullable=False)
    payload_json: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "run_uuid",
            "artifact_name",
            name="uq_qrc_qpu_json_run_artifact",
        ),
    )


# =============================================================================
# General DB helpers
# =============================================================================

def _ensure_qpu_timing_columns() -> None:
    """
    Add Week-11 timing columns to an already-existing qrc_qpu_run table.

    SQLAlchemy create_all() creates missing tables but does not ALTER an
    existing table.  This small additive migration keeps old databases usable.
    """
    inspector = inspect(engine)

    if "qrc_qpu_run" not in inspector.get_table_names():
        return

    existing = {
        col["name"]
        for col in inspector.get_columns("qrc_qpu_run")
    }

    additions = {
        "qpu_charge_time_seconds": "DOUBLE NULL",
        "qpu_circuits_execution_time_seconds": "DOUBLE NULL",
        "qpu_running_wall_seconds": "DOUBLE NULL",
        "ibm_job_created_at_utc": "DATETIME NULL",
        "ibm_job_running_at_utc": "DATETIME NULL",
        "ibm_job_finished_at_utc": "DATETIME NULL",
    }

    missing = [
        (name, sql_type)
        for name, sql_type in additions.items()
        if name not in existing
    ]

    if not missing:
        return

    with engine.begin() as connection:
        for name, sql_type in missing:
            # Column names/types are fixed constants above, not user input.
            connection.execute(
                text(
                    f"ALTER TABLE qrc_qpu_run "
                    f"ADD COLUMN {name} {sql_type}"
                )
            )

    print(
        "[database] Added QPU timing column(s): "
        + ", ".join(name for name, _ in missing)
    )


def create_all_tables() -> None:
    Base.metadata.create_all(engine)
    _ensure_qpu_timing_columns()


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if value is pd.NA:
        return None

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def dumps_json(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
    )


def normalize_db_value(key: str, value: Any) -> Any:
    if value is None:
        return None

    if key.endswith("_json"):
        if isinstance(value, str):
            return value
        return dumps_json(value)

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None

    return value


# =============================================================================
# Direct-QPU persistence helpers
# =============================================================================

def create_run(**fields) -> str:
    create_all_tables()

    run_uuid = str(uuid.uuid4())

    allowed = {
        c.name
        for c in QpuRun.__table__.columns
        if c.name not in {"id", "run_uuid", "created_at", "updated_at"}
    }

    clean = {
        k: normalize_db_value(k, v)
        for k, v in fields.items()
        if k in allowed
    }
    clean.setdefault("started_at_utc", utc_now_naive())

    with Session(engine) as session:
        session.add(QpuRun(run_uuid=run_uuid, **clean))
        session.commit()

    return run_uuid


def update_run(run_uuid: str, **fields) -> None:
    allowed = {
        c.name
        for c in QpuRun.__table__.columns
        if c.name not in {"id", "run_uuid", "created_at", "updated_at"}
    }

    with Session(engine) as session:
        row = session.execute(
            select(QpuRun).where(QpuRun.run_uuid == run_uuid)
        ).scalar_one()

        for key, value in fields.items():
            if key in allowed:
                setattr(row, key, normalize_db_value(key, value))

        session.commit()


def save_dataframe_artifact(
    run_uuid: str,
    artifact_name: str,
    dataframe: pd.DataFrame,
) -> None:
    df = dataframe.reset_index(drop=True)

    with Session(engine) as session:
        session.execute(
            delete(QpuRunCsvRow).where(
                QpuRunCsvRow.run_uuid == run_uuid,
                QpuRunCsvRow.artifact_name == artifact_name,
            )
        )

        mappings = []
        for row_index, record in enumerate(df.to_dict(orient="records")):
            safe = json_safe(record)

            endpoint = safe.get("endpoint")
            if endpoint is not None:
                endpoint = int(endpoint)

            setting = safe.get("setting")
            if setting is not None:
                setting = str(setting)

            mappings.append(
                {
                    "run_uuid": run_uuid,
                    "artifact_name": artifact_name,
                    "row_index": int(row_index),
                    "endpoint": endpoint,
                    "setting": setting,
                    "row_json": dumps_json(safe),
                }
            )

        if mappings:
            session.bulk_insert_mappings(QpuRunCsvRow, mappings)

        session.commit()


def save_json_artifact(
    run_uuid: str,
    artifact_name: str,
    payload: Any,
) -> None:
    payload_json = dumps_json(payload)

    with Session(engine) as session:
        existing = session.execute(
            select(QpuRunJsonArtifact).where(
                QpuRunJsonArtifact.run_uuid == run_uuid,
                QpuRunJsonArtifact.artifact_name == artifact_name,
            )
        ).scalar_one_or_none()

        if existing is None:
            session.add(
                QpuRunJsonArtifact(
                    run_uuid=run_uuid,
                    artifact_name=artifact_name,
                    payload_json=payload_json,
                )
            )
        else:
            existing.payload_json = payload_json

        session.commit()


def fetch_run(run_uuid: str) -> dict[str, Any]:
    with Session(engine) as session:
        row = session.execute(
            select(QpuRun).where(QpuRun.run_uuid == run_uuid)
        ).scalar_one()

        return {
            c.name: getattr(row, c.name)
            for c in QpuRun.__table__.columns
        }
