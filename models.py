"""
SQLAlchemy ORM models for storing QRC experiment results.

All tables are prefixed with `qrc_` to avoid collisions with other
projects sharing the same database.

Schema overview
---------------
qrc_runs        -- one row per script execution (timestamp, note, git hash)
qrc_datasets    -- one row per time series (MG-17, NARMA-10, Lorenz, ...)
qrc_models      -- one row per model type (naive, linear, ESN, LSTM, QRC, ...)
qrc_experiments -- one (run, dataset, model, horizon, hyperparameters) combination
qrc_results     -- metrics for a single split of one experiment
"""

from sqlalchemy import (
    Column, Integer, String, Float, Text,
    DateTime, ForeignKey, JSON,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class QRCRun(Base):
    """
    One script execution.  Created at the start of 05_baselines.py (or any
    future experiment script) so every result can be traced back to when and
    why it was produced.

    git_hash: short SHA of HEAD at run time (optional, populated automatically
              if git is available).
    note    : free-text label, e.g. 'initial baselines' or 'after ESN fix'.
    """
    __tablename__ = "qrc_runs"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    git_hash   = Column(String(16))
    note       = Column(Text)

    experiments = relationship("QRCExperiment", back_populates="run")

    def __repr__(self):
        return f"<QRCRun id={self.id} started_at={self.started_at} note={self.note!r}>"


class QRCDataset(Base):
    """Time series dataset used in experiments."""
    __tablename__ = "qrc_datasets"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(100), unique=True, nullable=False)
    # 'dde' | 'ode' | 'discrete_map' | 'synthetic'
    series_type = Column(String(50))
    # e.g. {"tau": 17, "T": 10000, "beta": 0.2}
    parameters  = Column(JSON)
    n_train     = Column(Integer)
    n_val       = Column(Integer)
    n_test      = Column(Integer)
    description = Column(Text)
    created_at  = Column(DateTime, server_default=func.now())

    experiments = relationship("QRCExperiment", back_populates="dataset")

    def __repr__(self):
        return f"<QRCDataset id={self.id} name={self.name!r}>"


class QRCModel(Base):
    """Model / algorithm type."""
    __tablename__ = "qrc_models"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(100), unique=True, nullable=False)
    # 'trivial' | 'classical_rc' | 'rnn' | 'quantum'
    category    = Column(String(50))
    description = Column(Text)

    experiments = relationship("QRCExperiment", back_populates="model")

    def __repr__(self):
        return f"<QRCModel id={self.id} name={self.name!r}>"


class QRCExperiment(Base):
    """
    One experimental unit: (run, dataset, model, horizon, hyperparameters).
    Multiple results (train / val / test splits) belong to one experiment.
    """
    __tablename__ = "qrc_experiments"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_id          = Column(Integer, ForeignKey("qrc_runs.id"), nullable=False)
    dataset_id      = Column(Integer, ForeignKey("qrc_datasets.id"), nullable=False)
    model_id        = Column(Integer, ForeignKey("qrc_models.id"),   nullable=False)
    horizon         = Column(Integer, default=1)
    # e.g. {"n_reservoir": 100, "spectral_radius": 0.9, "seed": 42}
    hyperparameters = Column(JSON)
    notes           = Column(Text)
    created_at      = Column(DateTime, server_default=func.now())

    run     = relationship("QRCRun",     back_populates="experiments")
    dataset = relationship("QRCDataset", back_populates="experiments")
    model   = relationship("QRCModel",   back_populates="experiments")
    results = relationship("QRCResult",  back_populates="experiment")

    def __repr__(self):
        return (f"<QRCExperiment id={self.id} run_id={self.run_id} "
                f"dataset_id={self.dataset_id} model_id={self.model_id} "
                f"horizon={self.horizon}>")


class QRCResult(Base):
    """
    Metrics for one split (train / val / test) of one experiment.
    valid_time is only populated for Lorenz autonomous-prediction runs.
    """
    __tablename__ = "qrc_results"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id     = Column(Integer, ForeignKey("qrc_experiments.id"), nullable=False)
    # 'train' | 'val' | 'test'
    split             = Column(String(10), nullable=False)
    nrmse             = Column(Float)
    rmse              = Column(Float)
    mae               = Column(Float)
    # Lyapunov times — only meaningful for autonomous Lorenz prediction
    valid_time        = Column(Float)
    training_time_sec = Column(Float)
    created_at        = Column(DateTime, server_default=func.now())

    experiment = relationship("QRCExperiment", back_populates="results")

    def __repr__(self):
        return (f"<QRCResult id={self.id} "
                f"experiment_id={self.experiment_id} "
                f"split={self.split!r} nrmse={self.nrmse:.4f}>")
