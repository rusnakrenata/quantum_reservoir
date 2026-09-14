from __future__ import annotations

"""
Backfill the already-computed Week 11.1D.1 washout audit into the database.

The original washout calculation finished successfully and wrote its CSV/JSON
artifacts.  Only the database INSERT failed because evaluation_split exceeded
VARCHAR(30).

This script DOES NOT recompute the washout and DOES NOT contact IBM.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd

import db_objects as db


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")

AUDIT_SCRIPT = HERE / "11_1D_candidate4_cont_context_audit.py"

TIMEWISE = (
    RESULTS
    / "11_1D1_candidate4_washout_timewise.csv"
)

THRESHOLDS = (
    RESULTS
    / "11_1D1_candidate4_washout_thresholds.csv"
)

SUMMARY = (
    RESULTS
    / "11_1D1_candidate4_washout_summary.json"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def main():
    print(
        "=" * 110
    )
    print(
        "BACKFILL WEEK 11.1D.1 WASHOUT AUDIT"
    )
    print(
        "=" * 110
    )

    for path in (
        TIMEWISE,
        THRESHOLDS,
        SUMMARY,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {path}"
            )

    audit = load_module(
        AUDIT_SCRIPT,
        "qrc_11_1d_backfill_audit",
    )

    candidate, manifest_meta = (
        audit.load_candidate()
    )

    timewise_df = pd.read_csv(
        TIMEWISE
    )

    thresholds_df = pd.read_csv(
        THRESHOLDS
    )

    summary = json.loads(
        SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    primary_tw = summary.get(
        "primary_trace_washout_steps"
    )

    db.create_all_tables()

    run_uuid = db.create_run(
        script_name=(
            "11_1D1_candidate4_washout_trace_distance.py"
        ),
        run_status=(
            "WASHOUT_AUDIT_COMPLETE"
        ),
        started_at_utc=(
            db.utc_now_naive()
        ),
        completed_at_utc=(
            db.utc_now_naive()
        ),
        forecast_dataset_name=(
            "property_damage_next_day_v1"
        ),
        feature_set_name="F4",
        # Must stay <= VARCHAR(30).
        evaluation_split=(
            "trainval_2022_2025"
        ),
        candidate_key=(
            "CONT_H0_R4"
        ),
        selected_by_rules=(
            "Rule4"
        ),
        protocol="CONT",
        topology=(
            candidate[
                "topology"
            ]
        ),
        window_size=(
            None
            if primary_tw is None
            else int(
                primary_tw
            )
        ),
        trotter_r=int(
            candidate[
                "r"
            ]
        ),
        readout_name=(
            audit.EXPECTED_READOUT
        ),
        alpha=float(
            candidate[
                "alpha"
            ]
        ),
        dt=float(
            candidate[
                "dt"
            ]
        ),
        hx=float(
            candidate[
                "hx"
            ]
        ),
        hy=float(
            candidate[
                "hy"
            ]
        ),
        j_json=(
            summary.get(
                "J"
            )
        ),
        manifest_json=(
            manifest_meta
        ),
        shots_per_setting=(
            audit.SHOTS_PER_SETTING
        ),
        notes=(
            "Backfilled candidate #4 global trace-distance "
            "washout audit. Original numerical calculation "
            "completed successfully; only the original DB "
            "INSERT failed because evaluation_split exceeded "
            "VARCHAR(30). No QPU job submitted."
        ),
    )

    db.save_dataframe_artifact(
        run_uuid,
        TIMEWISE.name,
        timewise_df,
    )

    db.save_dataframe_artifact(
        run_uuid,
        THRESHOLDS.name,
        thresholds_df,
    )

    db.save_json_artifact(
        run_uuid,
        SUMMARY.name,
        summary,
    )

    print(
        f"DB backfill complete: {run_uuid}"
    )
    print(
        "No washout recomputation and no IBM job were performed."
    )


if __name__ == "__main__":
    main()
