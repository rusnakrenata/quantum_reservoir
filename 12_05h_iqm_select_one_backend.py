"""
Week 12.5H — Select ONE IQM backend using the IBM-mirrored Rule-5 hierarchy.

Inputs:
    results/12_05g1_iqm_backend_comparison.csv
    results/12_05g1_iqm_backend_topology_comparison.csv

Selection:
    No weighted score.
    1) require compile_pass_rate=1, zero_swap_rate=1, all 7 topologies clean
    2) maximize worst_retained_min_cz_fidelity
    3) maximize median_retained_min_cz_fidelity
    4) maximize worst_retained_min_prx_fidelity
    5) minimize worst_retained_max_asym_ro_error
    6) maximize worst_retained_min_t1_us
    7) maximize worst_retained_min_t2_echo_us

The first differing criterion decides the backend.

Raw IQM properties only.
No p_depol.
No synthetic noise model.
No QPU jobs.
No simulator execution.
No forecasting evaluation.
2026 untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


RESULTS = Path("results")

IN_BACKEND = RESULTS / "12_05g1_iqm_backend_comparison.csv"
IN_TOPOLOGY = RESULTS / "12_05g1_iqm_backend_topology_comparison.csv"

OUT_CSV = RESULTS / "12_05h_iqm_backend_selection.csv"
OUT_JSON = RESULTS / "12_05h_iqm_backend_selection.json"

EXPECTED_BACKENDS = {"emerald", "garnet"}

HIERARCHY = [
    ("worst_retained_min_cz_fidelity", "max"),
    ("median_retained_min_cz_fidelity", "max"),
    ("worst_retained_min_prx_fidelity", "max"),
    ("worst_retained_max_asym_ro_error", "min"),
    ("worst_retained_min_t1_us", "max"),
    ("worst_retained_min_t2_echo_us", "max"),
]


def finite(x):
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def boolish(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def require_inputs():
    missing = [
        str(p)
        for p in (IN_BACKEND, IN_TOPOLOGY)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n  "
            + "\n  ".join(missing)
        )


def audit_inputs(df):
    observed = set(df["backend"].astype(str))
    if observed != EXPECTED_BACKENDS:
        raise RuntimeError(
            f"Expected {sorted(EXPECTED_BACKENDS)}, found {sorted(observed)}"
        )

    required = {
        "backend",
        "unique_candidates",
        "topologies",
        "compile_pass_rate",
        "zero_swap_rate",
        "all_7_topologies_compile_cleanly",
        *[name for name, _ in HIERARCHY],
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"Backend comparison missing columns: {missing}"
        )

    for _, row in df.iterrows():
        backend = str(row["backend"])

        if int(row["unique_candidates"]) != 47:
            raise RuntimeError(
                f"{backend}: expected 47 unique candidates, "
                f"found {row['unique_candidates']}"
            )

        if int(row["topologies"]) != 7:
            raise RuntimeError(
                f"{backend}: expected 7 topologies, "
                f"found {row['topologies']}"
            )

        if not math.isclose(float(row["compile_pass_rate"]), 1.0, abs_tol=1e-12):
            raise RuntimeError(f"{backend}: compile_pass_rate != 1")

        if not math.isclose(float(row["zero_swap_rate"]), 1.0, abs_tol=1e-12):
            raise RuntimeError(f"{backend}: zero_swap_rate != 1")

        if not boolish(row["all_7_topologies_compile_cleanly"]):
            raise RuntimeError(
                f"{backend}: all_7_topologies_compile_cleanly is false"
            )


def select_backend(df):
    work = df.copy().reset_index(drop=True)
    surviving = list(work.index)
    trace = []

    for rank, (metric, direction) in enumerate(HIERARCHY, start=1):
        values = {
            int(i): finite(work.loc[i, metric])
            for i in surviving
        }

        if any(v is None for v in values.values()):
            raise RuntimeError(
                f"Non-finite value in hierarchy metric {metric}"
            )

        best = (
            max(values.values())
            if direction == "max"
            else min(values.values())
        )

        kept = [
            i
            for i, value in values.items()
            if math.isclose(
                float(value),
                float(best),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]

        trace.append({
            "hierarchy_rank": rank,
            "metric": metric,
            "direction": direction,
            "values": {
                str(work.loc[i, "backend"]): float(values[i])
                for i in surviving
            },
            "best_value": float(best),
            "survivors_after_metric": [
                str(work.loc[i, "backend"])
                for i in kept
            ],
        })

        surviving = kept

        if len(surviving) == 1:
            return (
                str(work.loc[surviving[0], "backend"]),
                metric,
                trace,
            )

    if len(surviving) != 1:
        raise RuntimeError(
            "Backend selection tied after all declared Rule-5 criteria."
        )

    return (
        str(work.loc[surviving[0], "backend"]),
        HIERARCHY[-1][0],
        trace,
    )


def main():
    require_inputs()

    backend_df = pd.read_csv(IN_BACKEND)
    topology_df = pd.read_csv(IN_TOPOLOGY)  # loaded to lock the handoff inputs

    audit_inputs(backend_df)

    print("=" * 128)
    print("WEEK 12.5H — SELECT ONE IQM BACKEND | IBM-MIRRORED RULE-5 HIERARCHY")
    print("=" * 128)
    print("No weighted score. Raw IQM properties only.")
    print("No simulator/QPU jobs. No forecasting evaluation. 2026 untouched.")
    print()

    selected, decisive_metric, trace = select_backend(backend_df)

    show_cols = [
        "backend",
        "worst_retained_min_cz_fidelity",
        "median_retained_min_cz_fidelity",
        "worst_retained_min_prx_fidelity",
        "worst_retained_max_asym_ro_error",
        "worst_retained_min_t1_us",
        "worst_retained_min_t2_echo_us",
        "median_raw_prx_duration_ns",
        "median_raw_cz_duration_ns",
        "compile_pass_rate",
        "zero_swap_rate",
    ]

    print("Backend comparison:")
    print(
        backend_df[
            [c for c in show_cols if c in backend_df.columns]
        ].to_string(index=False)
    )

    print()
    print("Hierarchy trace:")
    for step in trace:
        print(
            f"  {step['hierarchy_rank']}. "
            f"{step['metric']} ({step['direction']}) -> "
            f"{step['survivors_after_metric']}"
        )

    print()
    print("=" * 128)
    print(f"SELECTED IQM BACKEND: {selected.upper()}")
    print(f"DECISIVE METRIC:      {decisive_metric}")
    print("=" * 128)

    out = backend_df.copy()
    out["selected_backend"] = (
        out["backend"].astype(str).eq(selected)
    )
    out["decisive_metric"] = decisive_metric
    out["selection_method"] = (
        "IBM-mirrored lexicographic Rule-5 hierarchy; no weighted score"
    )
    out.to_csv(OUT_CSV, index=False)

    payload = {
        "stage": "12.5H",
        "selected_backend": selected,
        "decisive_metric": decisive_metric,
        "weighted_score": False,
        "raw_iqm_properties_only": True,
        "candidate_universe": {
            "unique_candidates": 47,
            "rule_assignments": 56,
            "topologies": 7,
            "protocols": ["CONT", "RWP"],
            "rule_coverage":
                "R1-R4 complete for all 14 topology/protocol cells",
        },
        "compile_gate": {
            "compile_pass_rate_required": 1.0,
            "zero_swap_rate_required": 1.0,
            "all_7_topologies_compile_cleanly_required": True,
        },
        "hierarchy": [
            {"metric": metric, "direction": direction}
            for metric, direction in HIERARCHY
        ],
        "trace": trace,
        "backend_rows": backend_df.to_dict(orient="records"),
        "qpu_jobs": 0,
        "simulator_jobs": 0,
        "forecasting_evaluation": False,
        "test_2026_used": False,
        "next_step": (
            "Select the final five IQM QPU candidates only within the "
            f"frozen backend {selected}, preserving the 47-candidate / "
            "56-rule-assignment mapping."
        ),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print()
    print("Saved:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")
    print()
    print(
        "STOP HERE. Send this output before selecting the final five IQM QPU candidates."
    )


if __name__ == "__main__":
    main()
