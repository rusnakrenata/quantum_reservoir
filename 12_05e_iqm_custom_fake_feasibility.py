"""
Week 12.5E2 — Corrected live IQM calibration -> custom IQMFakeBackend feasibility audit.

Fixes 12.5E:
- Quality metrics are parsed from IQM observation records via `dut_field`,
  exactly like the successful Week-12.2 region-search parser.
- Handles empty categories without DataFrame-column crashes.
- Parses:
    T1
    T2_echo
    PRX RB fidelity
    CZ IRB fidelity
    readout fidelity
    readout error 0->1
    readout error 1->0
    PRX duration
    CZ duration
    measurement duration

NO QPU jobs.
NO simulator execution.
NO shots.
NO QRC.
NO forecasting data.
2026 untouched.

Outputs
-------
results/12_05e2_iqm_duration_observations.csv
results/12_05e2_iqm_quality_observations.csv
results/12_05e2_iqm_custom_fake_feasibility.csv
results/12_05e2_iqm_custom_fake_feasibility.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from iqm.iqm_client import IQMClient

from iqm_account import get_authentication


RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

OUT_DUR = RESULTS / "12_05e2_iqm_duration_observations.csv"
OUT_QM = RESULTS / "12_05e2_iqm_quality_observations.csv"
OUT_SUM = RESULTS / "12_05e2_iqm_custom_fake_feasibility.csv"
OUT_JSON = RESULTS / "12_05e2_iqm_custom_fake_feasibility.json"

ALIASES = ("emerald", "garnet")


# -----------------------------------------------------------------------------
# Exact IQM quality-metric patterns observed in Week 12.1/12.2.
# -----------------------------------------------------------------------------

RE_T1 = re.compile(
    r"^characterization\.model\.(QB\d+)\.t1_time$"
)

RE_T2_ECHO = re.compile(
    r"^characterization\.model\.(QB\d+)\.t2_echo_time$"
)

RE_PRX_FIDELITY = re.compile(
    r"^metrics\.rb\.prx\.([^.]+)\.(QB\d+)\.fidelity(?::.*)?$"
)

RE_CZ_FIDELITY = re.compile(
    r"^metrics\.irb\.cz\.([^.]+)\.(QB\d+)__(QB\d+)\.fidelity(?::.*)?$"
)

RE_RO_FIDELITY = re.compile(
    r"^metrics\.ssro\.measure\.([^.]+)\.(QB\d+)\.fidelity$"
)

RE_RO_ERR_01 = re.compile(
    r"^metrics\.ssro\.measure\.([^.]+)\.(QB\d+)\.error_0_to_1$"
)

RE_RO_ERR_10 = re.compile(
    r"^metrics\.ssro\.measure\.([^.]+)\.(QB\d+)\.error_1_to_0$"
)


# -----------------------------------------------------------------------------
# Calibration-duration patterns.
# -----------------------------------------------------------------------------

RE_PRX_DURATION = re.compile(
    r"^gates\.prx\.([^.]+)\.(QB\d+)\.duration$"
)

RE_CZ_DURATION = re.compile(
    r"^gates\.cz\.([^.]+)\."
    r"((?:QB\d+|COMPR\d+)__(?:QB\d+|COMPR\d+))\.duration$"
)

RE_MEASURE_DURATION = re.compile(
    r"^gates\.measure\.([^.]+)\.(QB\d+)\.duration$"
)


def jsonable(x: Any):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]

    if hasattr(x, "model_dump"):
        try:
            return jsonable(x.model_dump())
        except Exception:
            pass

    if hasattr(x, "dict"):
        try:
            return jsonable(x.dict())
        except Exception:
            pass

    if hasattr(x, "__dict__"):
        try:
            return jsonable(vars(x))
        except Exception:
            pass

    return str(x)


def finite_number(x):
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def collect_observations(obj):
    """
    Recursively find IQM observation records:
      {
        "dut_field": "...",
        "value": ...,
        "unit": ...,
        "invalid": ...
      }
    """
    rows = []

    if isinstance(obj, dict):
        if "dut_field" in obj and "value" in obj:
            rows.append(obj)

        for value in obj.values():
            rows.extend(
                collect_observations(value)
            )

    elif isinstance(obj, list):
        for value in obj:
            rows.extend(
                collect_observations(value)
            )

    return rows


def normalize_pair(a, b):
    return tuple(sorted((str(a), str(b))))


def parse_quality(alias, quality):
    observations = collect_observations(
        jsonable(quality)
    )

    rows = []

    for obs in observations:
        if bool(obs.get("invalid", False)):
            continue

        field = str(
            obs.get("dut_field", "")
        )

        value = finite_number(
            obs.get("value")
        )

        if value is None:
            continue

        metric_type = None
        locus = None
        implementation = None

        m = RE_T1.match(field)
        if m:
            metric_type = "t1"
            locus = m.group(1)

        if metric_type is None:
            m = RE_T2_ECHO.match(field)
            if m:
                metric_type = "t2_echo"
                locus = m.group(1)

        if metric_type is None:
            m = RE_PRX_FIDELITY.match(field)
            if m:
                metric_type = "prx_fidelity"
                implementation, locus = m.groups()

        if metric_type is None:
            m = RE_CZ_FIDELITY.match(field)
            if m:
                implementation, qa, qb = m.groups()
                metric_type = "cz_fidelity"
                locus = "__".join(
                    normalize_pair(qa, qb)
                )

        if metric_type is None:
            m = RE_RO_FIDELITY.match(field)
            if m:
                implementation, locus = m.groups()
                metric_type = "readout_fidelity"

        if metric_type is None:
            m = RE_RO_ERR_01.match(field)
            if m:
                implementation, locus = m.groups()
                metric_type = "readout_error_0_to_1"

        if metric_type is None:
            m = RE_RO_ERR_10.match(field)
            if m:
                implementation, locus = m.groups()
                metric_type = "readout_error_1_to_0"

        if metric_type is None:
            continue

        rows.append({
            "backend": alias,
            "metric_type": metric_type,
            "implementation": implementation,
            "locus": locus,
            "dut_field": field,
            "value": value,
            "unit": obs.get("unit"),
            "uncertainty": obs.get("uncertainty"),
            "created_timestamp": obs.get("created_timestamp"),
            "modified_timestamp": obs.get("modified_timestamp"),
        })

    return rows


def parse_durations(alias, calibration):
    observations = collect_observations(
        jsonable(calibration)
    )

    rows = []

    for obs in observations:
        if bool(obs.get("invalid", False)):
            continue

        field = str(
            obs.get("dut_field", "")
        )

        value = finite_number(
            obs.get("value")
        )

        if value is None:
            continue

        gate_type = None
        implementation = None
        locus = None

        m = RE_PRX_DURATION.match(field)
        if m:
            implementation, locus = m.groups()
            gate_type = "prx"

        if gate_type is None:
            m = RE_CZ_DURATION.match(field)
            if m:
                implementation, locus = m.groups()
                gate_type = "cz"

        if gate_type is None:
            m = RE_MEASURE_DURATION.match(field)
            if m:
                implementation, locus = m.groups()
                gate_type = "measure"

        if gate_type is None:
            continue

        rows.append({
            "backend": alias,
            "gate_type": gate_type,
            "implementation": implementation,
            "locus": locus,
            "dut_field": field,
            "duration_s": value,
            "duration_ns": value * 1e9,
            "unit": obs.get("unit"),
            "uncertainty": obs.get("uncertainty"),
            "created_timestamp": obs.get("created_timestamp"),
            "modified_timestamp": obs.get("modified_timestamp"),
        })

    return rows


def safe_subset(df, backend, category_col, category):
    if df.empty:
        return pd.DataFrame()

    required = {
        "backend",
        category_col,
    }

    if not required.issubset(
        set(df.columns)
    ):
        return pd.DataFrame()

    return df[
        df["backend"].eq(backend)
        & df[category_col].eq(category)
    ].copy()


def numeric_summary(values):
    arr = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    if arr.size == 0:
        return {
            "n_numeric": 0,
            "min": None,
            "median": None,
            "max": None,
        }

    return {
        "n_numeric": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def build_summary(duration_df, quality_df):
    rows = []

    for alias in ALIASES:
        # Durations
        for gate_type in (
            "prx",
            "cz",
            "measure",
        ):
            sub = safe_subset(
                duration_df,
                alias,
                "gate_type",
                gate_type,
            )

            stats = numeric_summary(
                sub["duration_ns"]
                if "duration_ns" in sub.columns
                else pd.Series(dtype=float)
            )

            rows.append({
                "backend": alias,
                "category": f"duration_{gate_type}",
                "n_rows": int(len(sub)),
                "n_unique_loci": (
                    int(sub["locus"].nunique())
                    if "locus" in sub.columns
                    else 0
                ),
                "n_unique_implementations": (
                    int(sub["implementation"].nunique())
                    if "implementation" in sub.columns
                    else 0
                ),
                "n_numeric": stats["n_numeric"],
                "min": stats["min"],
                "median": stats["median"],
                "max": stats["max"],
                "units_for_summary": "ns",
            })

        # Quality metrics
        for metric_type in (
            "t1",
            "t2_echo",
            "prx_fidelity",
            "cz_fidelity",
            "readout_fidelity",
            "readout_error_0_to_1",
            "readout_error_1_to_0",
        ):
            sub = safe_subset(
                quality_df,
                alias,
                "metric_type",
                metric_type,
            )

            stats = numeric_summary(
                sub["value"]
                if "value" in sub.columns
                else pd.Series(dtype=float)
            )

            rows.append({
                "backend": alias,
                "category": metric_type,
                "n_rows": int(len(sub)),
                "n_unique_loci": (
                    int(sub["locus"].nunique())
                    if "locus" in sub.columns
                    else 0
                ),
                "n_unique_implementations": (
                    int(sub["implementation"].nunique())
                    if "implementation" in sub.columns
                    else 0
                ),
                "n_numeric": stats["n_numeric"],
                "min": stats["min"],
                "median": stats["median"],
                "max": stats["max"],
                "units_for_summary": (
                    "s"
                    if metric_type in (
                        "t1",
                        "t2_echo",
                    )
                    else "dimensionless"
                ),
            })

    return pd.DataFrame(rows)


def category_present(summary, backend, category):
    sub = summary[
        summary["backend"].eq(backend)
        & summary["category"].eq(category)
    ]

    if len(sub) != 1:
        return False

    return int(
        sub.iloc[0]["n_rows"]
    ) > 0


def extract_set_id(obj):
    plain = jsonable(obj)

    if not isinstance(plain, dict):
        return None

    for key in (
        "observation_set_id",
        "quality_metric_set_id",
        "calibration_set_id",
        "id",
        "uuid",
    ):
        if key in plain:
            return str(plain[key])

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    outputs = (
        OUT_DUR,
        OUT_QM,
        OUT_SUM,
        OUT_JSON,
    )

    if args.overwrite:
        for path in outputs:
            if path.exists():
                path.unlink()

    token, server_url, _ = (
        get_authentication()
    )

    print("=" * 128)
    print(
        "WEEK 12.5E2 — CORRECTED LIVE IQM CALIBRATION -> CUSTOM IQMFAKEBACKEND FEASIBILITY"
    )
    print("=" * 128)
    print(
        "No QPU jobs. No simulator execution. No shots. "
        "No QRC/data. 2026 untouched."
    )

    duration_rows = []
    quality_rows = []

    ids = {}

    for alias in ALIASES:
        print()
        print(
            f"[{alias}] loading architecture, calibration and quality metrics..."
        )

        client = IQMClient(
            server_url,
            quantum_computer=alias,
            token=token,
        )

        # Fetch to confirm exact current architecture is available.
        _ = client.get_static_quantum_architecture()

        calibration = (
            client.get_calibration_set()
        )

        quality = (
            client.get_quality_metric_set()
        )

        drows = parse_durations(
            alias,
            calibration,
        )

        qrows = parse_quality(
            alias,
            quality,
        )

        duration_rows.extend(
            drows
        )

        quality_rows.extend(
            qrows
        )

        ids[alias] = {
            "calibration_set_id":
                extract_set_id(calibration),
            "quality_metric_set_id":
                extract_set_id(quality),
        }

        print(
            f"  parsed durations:      {len(drows)}"
        )
        print(
            f"  parsed quality metrics:{len(qrows)}"
        )

        if qrows:
            counts = (
                pd.DataFrame(qrows)[
                    "metric_type"
                ]
                .value_counts()
                .to_dict()
            )

            print(
                f"  quality breakdown: {counts}"
            )

    duration_df = pd.DataFrame(
        duration_rows,
        columns=[
            "backend",
            "gate_type",
            "implementation",
            "locus",
            "dut_field",
            "duration_s",
            "duration_ns",
            "unit",
            "uncertainty",
            "created_timestamp",
            "modified_timestamp",
        ],
    )

    quality_df = pd.DataFrame(
        quality_rows,
        columns=[
            "backend",
            "metric_type",
            "implementation",
            "locus",
            "dut_field",
            "value",
            "unit",
            "uncertainty",
            "created_timestamp",
            "modified_timestamp",
        ],
    )

    duration_df.to_csv(
        OUT_DUR,
        index=False,
    )

    quality_df.to_csv(
        OUT_QM,
        index=False,
    )

    summary = build_summary(
        duration_df,
        quality_df,
    )

    summary.to_csv(
        OUT_SUM,
        index=False,
    )

    print()
    print("=" * 128)
    print(
        "CALIBRATION / QUALITY COVERAGE SUMMARY"
    )
    print("=" * 128)

    print(
        summary.to_string(
            index=False
        )
    )

    required = (
        "duration_prx",
        "duration_cz",
        "t1",
        "t2_echo",
        "prx_fidelity",
        "cz_fidelity",
        "readout_error_0_to_1",
        "readout_error_1_to_0",
    )

    assessment_backends = {}

    print()
    print("=" * 128)
    print(
        "CUSTOM IQMFAKEBACKEND FEASIBILITY"
    )
    print("=" * 128)

    for alias in ALIASES:
        present = {
            category:
                category_present(
                    summary,
                    alias,
                    category,
                )
            for category in required
        }

        all_present = all(
            present.values()
        )

        assessment_backends[
            alias
        ] = {
            **ids[alias],
            "required_categories_present":
                present,
            "all_required_categories_present":
                bool(all_present),
            "status":
                (
                    "PASS_PROFILE_CONSTRUCTION_NEXT"
                    if all_present
                    else "INCOMPLETE_REVIEW_REQUIRED"
                ),
        }

        print(
            f"{alias}: "
            f"{assessment_backends[alias]['status']}"
        )

        for category in required:
            print(
                f"  {category:24s}: "
                f"{'YES' if present[category] else 'NO'}"
            )

    assessment = {
        "stage":
            "12.5E2",
        "qpu_jobs":
            0,
        "simulator_jobs":
            0,
        "shots":
            0,
        "qrc_loaded":
            False,
        "forecast_data_loaded":
            False,
        "test_2026_used":
            False,
        "facade_result":
            (
                "Packaged IQMFacadeBackend profiles were architecture-incompatible "
                "with the current Resonance mock architectures."
            ),
        "custom_strategy":
            (
                "Use the exact current StaticQuantumArchitecture with a "
                "calibration-derived IQMErrorProfile."
            ),
        "backends":
            assessment_backends,
        "important_next_step":
            (
                "If PASS, derive IQMErrorProfile depolarizing parameters consistently "
                "with IQM semantics so that thermal relaxation + depolarizing error "
                "reproduce the observed RB/IRB gate fidelity. Then validate the custom "
                "IQMFakeBackend on tiny circuits before the 40-candidate QRC sweep."
            ),
    }

    with open(
        OUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            assessment,
            f,
            indent=2,
        )

    print()
    print("Saved:")
    for path in outputs:
        print(
            f"  {path}"
        )

    print()
    print(
        "STOP HERE. Send this output before constructing IQMErrorProfile "
        "or running noisy simulation."
    )


if __name__ == "__main__":
    main()
