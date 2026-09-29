"""
Week 12.5F — Full H0-H6 IQM Rule-5 refresh using RAW IQM properties only.

Scientific purpose
------------------
Re-run the physical-layout selection for every logical topology H0-H6 on both
Emerald and Garnet using the SAME Rule-5 scientific hierarchy used for IBM,
while attaching the newly recovered native IQM PRX/CZ/measurement durations.

NO synthetic noise parameters are derived.
NO depolarizing probabilities are calculated.
NO weighted hardware score is used.

Raw IQM properties used
-----------------------
Selection properties:
    T1
    T2_echo
    CZ fidelity
    asymmetric readout errors P(1|0), P(0|1)

Diagnostic/commentary properties only:
    PRX fidelity
    PRX duration
    CZ duration
    measurement duration
    mean CZ fidelity
    mean T1 / T2_echo

IBM-mirrored Rule-5 hierarchy
-----------------------------
Core Pareto objectives:
    min T1                  -> maximize
    min T2_echo             -> maximize
    max asymmetric RO error -> minimize
    min CZ fidelity         -> maximize

Then hard hierarchy:
    Pareto
        -> 10 highest worst-edge CZ fidelity
        -> 5 highest global minimum T2_echo
        -> 3 highest memory minimum T2_echo

The final three physical layouts are RETAINED.
No weighted aggregation is introduced.

Inputs
------
results/12_02_iqm_all_native_embeddings.csv
results/12_05e2_iqm_duration_observations.csv
results/12_05e2_iqm_quality_observations.csv

Outputs
-------
results/12_05f_iqm_rule5_all_embeddings_refreshed.csv
results/12_05f_iqm_rule5_pareto.csv
results/12_05f_iqm_rule5_hierarchy_trace.csv
results/12_05f_iqm_rule5_selected_top3.csv
results/12_05f_iqm_rule5_raw_duration_summary.csv
results/12_05f_iqm_rule5_manifest.json

Important
---------
- No QPU jobs.
- No simulator execution.
- No shots.
- No forecasting data.
- 2026 untouched.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS / CONFIG
# =============================================================================

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

IN_EMBED = RESULTS / "12_02_iqm_all_native_embeddings.csv"
IN_DUR = RESULTS / "12_05e2_iqm_duration_observations.csv"
IN_QM = RESULTS / "12_05e2_iqm_quality_observations.csv"

OUT_ALL = RESULTS / "12_05f_iqm_rule5_all_embeddings_refreshed.csv"
OUT_PARETO = RESULTS / "12_05f_iqm_rule5_pareto.csv"
OUT_TRACE = RESULTS / "12_05f_iqm_rule5_hierarchy_trace.csv"
OUT_TOP3 = RESULTS / "12_05f_iqm_rule5_selected_top3.csv"
OUT_DURATION = RESULTS / "12_05f_iqm_rule5_raw_duration_summary.csv"
OUT_MANIFEST = RESULTS / "12_05f_iqm_rule5_manifest.json"

BACKENDS = ("emerald", "garnet")
TOPOLOGIES = tuple(f"H{i}" for i in range(7))

MEMORY_LOGICAL = (4, 5)

TOPOLOGY_EDGES = {
    "H0": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
    "H1": ((0, 1), (1, 2), (2, 3), (3, 4), (3, 5)),
    "H2": ((0, 1), (1, 2), (2, 3), (2, 4), (4, 5)),
    "H3": ((0, 1), (0, 4), (1, 2), (2, 3), (4, 5)),
    "H4": ((0, 1), (0, 4), (1, 2), (3, 4), (4, 5)),
    "H5": ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (4, 5)),
    "H6": ((0, 1), (0, 4), (1, 5), (2, 3), (2, 5), (3, 4), (4, 5)),
}

EXPECTED_EDGE_COUNTS = {
    "H0": 5,
    "H1": 5,
    "H2": 5,
    "H3": 5,
    "H4": 5,
    "H5": 7,
    "H6": 7,
}


# =============================================================================
# SMALL HELPERS
# =============================================================================

def finite(x):
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def normalize_qb(q):
    q = str(q).strip()
    if not q.startswith("QB"):
        raise ValueError(f"Unexpected physical-qubit label: {q!r}")
    return q


def qb_sort_key(q):
    q = normalize_qb(q)
    try:
        return int(q[2:])
    except Exception:
        return q


def normalize_pair(a, b):
    a = normalize_qb(a)
    b = normalize_qb(b)
    return tuple(sorted((a, b), key=qb_sort_key))


def pair_key(a, b):
    x, y = normalize_pair(a, b)
    return f"{x}__{y}"


def parse_pair_locus(value):
    value = str(value).strip()

    if "__" not in value:
        return None

    a, b = value.split("__", 1)

    if not (
        a.startswith("QB")
        and b.startswith("QB")
    ):
        return None

    return normalize_pair(a, b)


def parse_layout(row):
    """
    Prefer explicit q0..q5 columns. Fall back to layout text if needed.
    """
    qcols = [
        "q0_C",
        "q1_D",
        "q2_P",
        "q3_H",
        "q4_M1",
        "q5_M2",
    ]

    if all(c in row.index for c in qcols):
        vals = [
            str(row[c]).strip()
            for c in qcols
        ]

        if all(v.startswith("QB") for v in vals):
            return vals

    raw = str(row.get("layout", "")).strip()

    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        vals = [
            x.strip().strip("'\"")
            for x in inner.split(",")
            if x.strip()
        ]
        if len(vals) == 6:
            return vals

    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, (list, tuple)) and len(obj) == 6:
            return [str(x) for x in obj]
    except Exception:
        pass

    raise ValueError(
        f"Cannot parse six-qubit layout from row: {raw!r}"
    )


def safe_min(values):
    vals = [
        float(v)
        for v in values
        if finite(v) is not None
    ]
    return min(vals) if vals else None


def safe_max(values):
    vals = [
        float(v)
        for v in values
        if finite(v) is not None
    ]
    return max(vals) if vals else None


def safe_mean(values):
    vals = [
        float(v)
        for v in values
        if finite(v) is not None
    ]
    return float(np.mean(vals)) if vals else None


def safe_median(values):
    vals = [
        float(v)
        for v in values
        if finite(v) is not None
    ]
    return float(np.median(vals)) if vals else None


def nan_if_none(x):
    return np.nan if x is None else x


# =============================================================================
# LOAD NATIVE IQM TABLES
# =============================================================================

def require_inputs():
    missing = [
        str(p)
        for p in (
            IN_EMBED,
            IN_DUR,
            IN_QM,
        )
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n  "
            + "\n  ".join(missing)
        )


def load_inputs():
    require_inputs()

    emb = pd.read_csv(IN_EMBED)
    dur = pd.read_csv(IN_DUR)
    qm = pd.read_csv(IN_QM)

    required_emb = {
        "backend",
        "topology",
        "embedding_id",
        "layout",
    }

    if not required_emb.issubset(emb.columns):
        raise RuntimeError(
            f"{IN_EMBED} missing columns: "
            f"{sorted(required_emb - set(emb.columns))}"
        )

    required_dur = {
        "backend",
        "gate_type",
        "implementation",
        "locus",
        "duration_ns",
    }

    if not required_dur.issubset(dur.columns):
        raise RuntimeError(
            f"{IN_DUR} missing columns: "
            f"{sorted(required_dur - set(dur.columns))}"
        )

    required_qm = {
        "backend",
        "metric_type",
        "implementation",
        "locus",
        "value",
    }

    if not required_qm.issubset(qm.columns):
        raise RuntimeError(
            f"{IN_QM} missing columns: "
            f"{sorted(required_qm - set(qm.columns))}"
        )

    return emb, dur, qm


# =============================================================================
# RAW IQM PROPERTY LOOKUPS
# =============================================================================

def build_quality_lookups(qm):
    """
    No model conversion occurs here.
    Every lookup stores the RAW IQM metric value.
    """
    result = {}

    for backend in BACKENDS:
        b = qm[
            qm["backend"].astype(str).eq(backend)
        ].copy()

        def single_qubit_map(metric_type):
            sub = b[
                b["metric_type"].astype(str).eq(metric_type)
            ]

            out = {}

            for _, row in sub.iterrows():
                locus = str(row["locus"])
                value = finite(row["value"])

                if (
                    locus.startswith("QB")
                    and value is not None
                ):
                    out[locus] = value

            return out

        t1 = single_qubit_map("t1")
        t2 = single_qubit_map("t2_echo")
        prx_fid = single_qubit_map("prx_fidelity")
        ro_fid = single_qubit_map("readout_fidelity")
        ro01 = single_qubit_map("readout_error_0_to_1")
        ro10 = single_qubit_map("readout_error_1_to_0")

        cz_fid = {}

        sub = b[
            b["metric_type"].astype(str).eq("cz_fidelity")
        ]

        for _, row in sub.iterrows():
            pair = parse_pair_locus(
                row["locus"]
            )
            value = finite(row["value"])

            if pair is not None and value is not None:
                cz_fid[pair] = {
                    "value": value,
                    "implementation":
                        str(row["implementation"]),
                }

        result[backend] = {
            "t1": t1,
            "t2_echo": t2,
            "prx_fidelity": prx_fid,
            "readout_fidelity": ro_fid,
            "readout_error_0_to_1": ro01,
            "readout_error_1_to_0": ro10,
            "cz_fidelity": cz_fid,
        }

    return result


def build_duration_lookups(dur):
    result = {}

    for backend in BACKENDS:
        b = dur[
            dur["backend"].astype(str).eq(backend)
        ].copy()

        prx = {}
        measure = {}
        cz = {}

        for _, row in b.iterrows():
            gate = str(row["gate_type"])
            locus = str(row["locus"])
            duration_ns = finite(
                row["duration_ns"]
            )
            impl = str(
                row["implementation"]
            )

            if duration_ns is None:
                continue

            if gate == "prx":
                prx[locus] = {
                    "duration_ns":
                        duration_ns,
                    "implementation":
                        impl,
                }

            elif gate == "measure":
                # There can be multiple measurement implementations.
                # Preserve them all rather than silently choosing one.
                measure.setdefault(
                    locus,
                    [],
                ).append({
                    "duration_ns":
                        duration_ns,
                    "implementation":
                        impl,
                })

            elif gate == "cz":
                pair = parse_pair_locus(
                    locus
                )

                if pair is None:
                    continue

                cz.setdefault(
                    pair,
                    [],
                ).append({
                    "duration_ns":
                        duration_ns,
                    "implementation":
                        impl,
                })

        result[backend] = {
            "prx": prx,
            "measure": measure,
            "cz": cz,
        }

    return result


def resolve_cz_duration(
    duration_lookup,
    pair,
    active_impl,
):
    """
    Match the duration to the SAME IQM implementation named by the CZ
    fidelity metric whenever possible.

    If there is exactly one raw duration for the pair, use it.
    If multiple exist and active_impl is known, use only the matching one.
    Otherwise return missing rather than inventing/averaging.
    """
    rows = duration_lookup[
        "cz"
    ].get(pair, [])

    if not rows:
        return None, None, "MISSING"

    if active_impl:
        exact = [
            r
            for r in rows
            if str(
                r["implementation"]
            ) == str(active_impl)
        ]

        if len(exact) == 1:
            r = exact[0]
            return (
                float(r["duration_ns"]),
                str(r["implementation"]),
                "IMPLEMENTATION_MATCH",
            )

    if len(rows) == 1:
        r = rows[0]
        return (
            float(r["duration_ns"]),
            str(r["implementation"]),
            "UNIQUE_PAIR_DURATION",
        )

    return None, None, "AMBIGUOUS_MULTI_IMPLEMENTATION"


def resolve_measure_duration(
    duration_lookup,
    qb,
):
    rows = duration_lookup[
        "measure"
    ].get(qb, [])

    if not rows:
        return None, None, "MISSING"

    # Do not invent which readout implementation is active.
    # If all exposed implementations have the same duration, the raw value
    # is unambiguous and may be reported.
    values = sorted(
        {
            float(r["duration_ns"])
            for r in rows
        }
    )

    if len(values) == 1:
        return (
            values[0],
            ";".join(
                sorted(
                    {
                        str(r["implementation"])
                        for r in rows
                    }
                )
            ),
            "SAME_DURATION_ACROSS_IMPLEMENTATIONS",
        )

    return None, None, "AMBIGUOUS_MULTI_IMPLEMENTATION"


# =============================================================================
# FRESH RAW-METRIC SCORING
# =============================================================================

def score_embedding(
    backend,
    topology,
    layout,
    qlook,
    dlook,
):
    logical_edges = TOPOLOGY_EDGES[
        topology
    ]

    t1_vals = []
    t2_vals = []
    prx_fid_vals = []
    ro_fid_vals = []
    ro01_vals = []
    ro10_vals = []
    prx_durations = []
    measure_durations = []

    missing = {
        "t1": 0,
        "t2_echo": 0,
        "prx_fidelity": 0,
        "readout_fidelity": 0,
        "readout_error_0_to_1": 0,
        "readout_error_1_to_0": 0,
        "prx_duration": 0,
        "measure_duration": 0,
        "cz_fidelity": 0,
        "cz_duration": 0,
    }

    measure_duration_status = []

    for qb in layout:
        qb = normalize_qb(qb)

        def add_single(metric_name, target):
            value = qlook[
                metric_name
            ].get(qb)

            if value is None:
                missing[
                    metric_name
                ] += 1
            else:
                target.append(
                    float(value)
                )

        add_single("t1", t1_vals)
        add_single("t2_echo", t2_vals)
        add_single(
            "prx_fidelity",
            prx_fid_vals,
        )
        add_single(
            "readout_fidelity",
            ro_fid_vals,
        )
        add_single(
            "readout_error_0_to_1",
            ro01_vals,
        )
        add_single(
            "readout_error_1_to_0",
            ro10_vals,
        )

        prx_row = dlook[
            "prx"
        ].get(qb)

        if prx_row is None:
            missing[
                "prx_duration"
            ] += 1
        else:
            prx_durations.append(
                float(
                    prx_row[
                        "duration_ns"
                    ]
                )
            )

        mdur, _, mstatus = (
            resolve_measure_duration(
                dlook,
                qb,
            )
        )

        measure_duration_status.append(
            f"{qb}:{mstatus}"
        )

        if mdur is None:
            missing[
                "measure_duration"
            ] += 1
        else:
            measure_durations.append(
                float(mdur)
            )

    # Memory qubits q4, q5.
    memory_qbs = [
        layout[q]
        for q in MEMORY_LOGICAL
    ]

    memory_t1 = [
        qlook["t1"].get(qb)
        for qb in memory_qbs
    ]

    memory_t2 = [
        qlook[
            "t2_echo"
        ].get(qb)
        for qb in memory_qbs
    ]

    cz_fids = []
    cz_durations = []
    cz_impls = []
    cz_duration_status = []

    required_pairs = []

    for i, j in logical_edges:
        a = layout[i]
        b = layout[j]
        pair = normalize_pair(a, b)
        required_pairs.append(
            pair_key(a, b)
        )

        fid_info = qlook[
            "cz_fidelity"
        ].get(pair)

        if fid_info is None:
            missing[
                "cz_fidelity"
            ] += 1
            active_impl = None
        else:
            cz_fids.append(
                float(
                    fid_info[
                        "value"
                    ]
                )
            )
            active_impl = str(
                fid_info[
                    "implementation"
                ]
            )
            cz_impls.append(
                f"{pair_key(a,b)}:{active_impl}"
            )

        duration_ns, dur_impl, status = (
            resolve_cz_duration(
                dlook,
                pair,
                active_impl,
            )
        )

        cz_duration_status.append(
            f"{pair_key(a,b)}:{status}"
        )

        if duration_ns is None:
            missing[
                "cz_duration"
            ] += 1
        else:
            cz_durations.append(
                float(duration_ns)
            )

    # Raw asymmetric readout error:
    # worst physical measurement direction over all selected qubits.
    all_asym_ro = (
        list(ro01_vals)
        + list(ro10_vals)
    )

    complete_rule5 = (
        missing["t1"] == 0
        and missing["t2_echo"] == 0
        and missing[
            "readout_error_0_to_1"
        ] == 0
        and missing[
            "readout_error_1_to_0"
        ] == 0
        and missing[
            "cz_fidelity"
        ] == 0
    )

    complete_raw_duration = (
        missing[
            "prx_duration"
        ] == 0
        and missing[
            "cz_duration"
        ] == 0
    )

    return {
        # IBM-equivalent Rule-5 core raw properties.
        "rule5_complete":
            bool(complete_rule5),
        "min_t1_us":
            nan_if_none(
                safe_min(t1_vals)
                * 1e6
                if safe_min(t1_vals)
                is not None
                else None
            ),
        "mean_t1_us":
            nan_if_none(
                safe_mean(t1_vals)
                * 1e6
                if safe_mean(t1_vals)
                is not None
                else None
            ),
        "min_t2_echo_us":
            nan_if_none(
                safe_min(t2_vals)
                * 1e6
                if safe_min(t2_vals)
                is not None
                else None
            ),
        "mean_t2_echo_us":
            nan_if_none(
                safe_mean(t2_vals)
                * 1e6
                if safe_mean(t2_vals)
                is not None
                else None
            ),
        "memory_min_t1_us":
            nan_if_none(
                safe_min(memory_t1)
                * 1e6
                if safe_min(memory_t1)
                is not None
                else None
            ),
        "memory_min_t2_echo_us":
            nan_if_none(
                safe_min(memory_t2)
                * 1e6
                if safe_min(memory_t2)
                is not None
                else None
            ),
        "min_cz_fidelity":
            nan_if_none(
                safe_min(cz_fids)
            ),
        "mean_cz_fidelity":
            nan_if_none(
                safe_mean(cz_fids)
            ),
        "min_readout_fidelity":
            nan_if_none(
                safe_min(ro_fid_vals)
            ),
        "max_readout_error_0_to_1":
            nan_if_none(
                safe_max(ro01_vals)
            ),
        "max_readout_error_1_to_0":
            nan_if_none(
                safe_max(ro10_vals)
            ),
        "max_asymmetric_readout_error":
            nan_if_none(
                safe_max(all_asym_ro)
            ),

        # Native PRX quality: diagnostic/commentary only.
        "min_prx_fidelity":
            nan_if_none(
                safe_min(prx_fid_vals)
            ),
        "mean_prx_fidelity":
            nan_if_none(
                safe_mean(prx_fid_vals)
            ),

        # Raw native durations only. No noise conversion.
        "raw_duration_complete_prx_cz":
            bool(
                complete_raw_duration
            ),
        "prx_duration_min_ns":
            nan_if_none(
                safe_min(prx_durations)
            ),
        "prx_duration_median_ns":
            nan_if_none(
                safe_median(
                    prx_durations
                )
            ),
        "prx_duration_max_ns":
            nan_if_none(
                safe_max(prx_durations)
            ),
        "cz_duration_min_ns":
            nan_if_none(
                safe_min(cz_durations)
            ),
        "cz_duration_median_ns":
            nan_if_none(
                safe_median(
                    cz_durations
                )
            ),
        "cz_duration_max_ns":
            nan_if_none(
                safe_max(cz_durations)
            ),
        "measure_duration_min_ns":
            nan_if_none(
                safe_min(
                    measure_durations
                )
            ),
        "measure_duration_median_ns":
            nan_if_none(
                safe_median(
                    measure_durations
                )
            ),
        "measure_duration_max_ns":
            nan_if_none(
                safe_max(
                    measure_durations
                )
            ),

        # Transparent raw metadata strings.
        "required_physical_edges":
            ";".join(
                required_pairs
            ),
        "cz_implementation_by_edge":
            ";".join(
                cz_impls
            ),
        "cz_duration_resolution":
            ";".join(
                cz_duration_status
            ),
        "measure_duration_resolution":
            ";".join(
                measure_duration_status
            ),

        # Missing counts kept explicit.
        **{
            f"missing_{k}_count":
                int(v)
            for k, v
            in missing.items()
        },
    }


# =============================================================================
# IBM-MIRRORED PARETO + HIERARCHY
# =============================================================================

PARETO_DIRECTIONS = {
    "min_t1_us": "max",
    "min_t2_echo_us": "max",
    "max_asymmetric_readout_error": "min",
    "min_cz_fidelity": "max",
}


def dominates(a, b):
    better_or_equal = True
    strictly_better = False

    for field, direction in (
        PARETO_DIRECTIONS.items()
    ):
        av = finite(a[field])
        bv = finite(b[field])

        if av is None or bv is None:
            return False

        if direction == "max":
            if av < bv:
                better_or_equal = False
                break
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                better_or_equal = False
                break
            if av < bv:
                strictly_better = True

    return (
        better_or_equal
        and strictly_better
    )


def pareto_mask(group):
    records = group.to_dict(
        orient="records"
    )

    keep = []

    for i, row in enumerate(records):
        dominated = False

        for j, other in enumerate(records):
            if i == j:
                continue

            if dominates(
                other,
                row,
            ):
                dominated = True
                break

        keep.append(
            not dominated
        )

    return pd.Series(
        keep,
        index=group.index,
        dtype=bool,
    )


def hierarchy_for_group(group):
    """
    Exact hard hierarchy:
      Pareto
      -> top 10 min CZ fidelity
      -> top 5 min T2_echo
      -> top 3 memory min T2_echo

    No candidate excluded at an earlier stage can re-enter later.
    """
    p = group[
        group["pareto"]
    ].copy()

    stage10 = (
        p.sort_values(
            by=[
                "min_cz_fidelity",
                "mean_cz_fidelity",
                "min_t2_echo_us",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            kind="mergesort",
        )
        .head(10)
        .copy()
    )

    stage5 = (
        stage10.sort_values(
            by=[
                "min_t2_echo_us",
                "memory_min_t2_echo_us",
                "min_cz_fidelity",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            kind="mergesort",
        )
        .head(5)
        .copy()
    )

    stage3 = (
        stage5.sort_values(
            by=[
                "memory_min_t2_echo_us",
                "min_t2_echo_us",
                "min_cz_fidelity",
                "min_t1_us",
                "max_asymmetric_readout_error",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
            ],
            kind="mergesort",
        )
        .head(3)
        .copy()
    )

    return p, stage10, stage5, stage3


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    outputs = (
        OUT_ALL,
        OUT_PARETO,
        OUT_TRACE,
        OUT_TOP3,
        OUT_DURATION,
        OUT_MANIFEST,
    )

    if args.overwrite:
        for p in outputs:
            if p.exists():
                p.unlink()

    emb, dur, qm = load_inputs()

    qlookup = build_quality_lookups(
        qm
    )

    dlookup = build_duration_lookups(
        dur
    )

    print("=" * 132)
    print(
        "WEEK 12.5F — FULL H0-H6 IQM RULE-5 REFRESH | RAW IQM PROPERTIES ONLY"
    )
    print("=" * 132)
    print(
        "No p_depol. No synthetic noise parameter. No weighted score."
    )
    print(
        "No QPU jobs. No simulator execution. No shots. "
        "No forecasting data. 2026 untouched."
    )
    print()
    print(
        "Rule-5 hierarchy:"
    )
    print(
        "  Pareto[min T1↑, min T2echo↑, max asymmetric RO error↓, min CZ fidelity↑]"
    )
    print(
        "  -> 10 highest worst-edge CZ fidelity"
    )
    print(
        "  -> 5 highest global minimum T2echo"
    )
    print(
        "  -> 3 highest memory minimum T2echo"
    )

    refreshed_rows = []

    for _, source in emb.iterrows():
        backend = str(
            source["backend"]
        )
        topology = str(
            source["topology"]
        )

        if (
            backend not in BACKENDS
            or topology not in TOPOLOGIES
        ):
            continue

        layout = parse_layout(
            source
        )

        if len(layout) != 6:
            raise RuntimeError(
                f"Non-six-qubit layout for "
                f"{backend}/{topology}: {layout}"
            )

        scores = score_embedding(
            backend=backend,
            topology=topology,
            layout=layout,
            qlook=qlookup[backend],
            dlook=dlookup[backend],
        )

        row = {
            "backend":
                backend,
            "topology":
                topology,
            "edge_count":
                EXPECTED_EDGE_COUNTS[
                    topology
                ],
            "embedding_id":
                int(
                    source[
                        "embedding_id"
                    ]
                ),
            "layout":
                "["
                + ",".join(layout)
                + "]",
            "q0_C":
                layout[0],
            "q1_D":
                layout[1],
            "q2_P":
                layout[2],
            "q3_H":
                layout[3],
            "q4_M1":
                layout[4],
            "q5_M2":
                layout[5],
            **scores,
            "pareto":
                False,
        }

        refreshed_rows.append(
            row
        )

    refreshed = pd.DataFrame(
        refreshed_rows
    )

    # Rule-5 validity uses only required raw Rule-5 properties.
    valid = refreshed[
        refreshed[
            "rule5_complete"
        ].astype(bool)
    ].copy()

    # Fresh Pareto per backend/topology using IBM's four core objectives.
    pareto_flags = pd.Series(
        False,
        index=valid.index,
        dtype=bool,
    )

    for (
        backend,
        topology,
    ), group in valid.groupby(
        [
            "backend",
            "topology",
        ],
        sort=False,
    ):
        mask = pareto_mask(
            group
        )

        pareto_flags.loc[
            mask.index
        ] = mask

    valid.loc[
        :,
        "pareto",
    ] = pareto_flags

    refreshed.loc[
        valid.index,
        "pareto",
    ] = valid[
        "pareto"
    ]

    refreshed.to_csv(
        OUT_ALL,
        index=False,
    )

    pareto_df = valid[
        valid[
            "pareto"
        ].astype(bool)
    ].copy()

    pareto_df.to_csv(
        OUT_PARETO,
        index=False,
    )

    trace_rows = []
    top3_rows = []

    print()
    print("=" * 132)
    print(
        "IBM-MIRRORED RULE-5 HIERARCHY"
    )
    print("=" * 132)

    for backend in BACKENDS:
        print()
        print(
            f"BACKEND: {backend.upper()}"
        )
        print(
            "-" * 132
        )

        for topology in TOPOLOGIES:
            group = valid[
                valid[
                    "backend"
                ].eq(backend)
                & valid[
                    "topology"
                ].eq(topology)
            ].copy()

            if group.empty:
                print(
                    f"{topology}: NO VALID EMBEDDINGS"
                )
                continue

            (
                p,
                s10,
                s5,
                s3,
            ) = hierarchy_for_group(
                group
            )

            stage_sets = {
                "pareto":
                    set(
                        p[
                            "embedding_id"
                        ].astype(int)
                    ),
                "top10_cz":
                    set(
                        s10[
                            "embedding_id"
                        ].astype(int)
                    ),
                "top5_t2":
                    set(
                        s5[
                            "embedding_id"
                        ].astype(int)
                    ),
                "top3_memory_t2":
                    set(
                        s3[
                            "embedding_id"
                        ].astype(int)
                    ),
            }

            for _, row in group.iterrows():
                emb_id = int(
                    row[
                        "embedding_id"
                    ]
                )

                trace_rows.append({
                    "backend":
                        backend,
                    "topology":
                        topology,
                    "embedding_id":
                        emb_id,
                    "layout":
                        row["layout"],
                    "in_pareto":
                        emb_id
                        in stage_sets[
                            "pareto"
                        ],
                    "in_top10_cz":
                        emb_id
                        in stage_sets[
                            "top10_cz"
                        ],
                    "in_top5_t2":
                        emb_id
                        in stage_sets[
                            "top5_t2"
                        ],
                    "in_top3_memory_t2":
                        emb_id
                        in stage_sets[
                            "top3_memory_t2"
                        ],
                    "min_t1_us":
                        row[
                            "min_t1_us"
                        ],
                    "min_t2_echo_us":
                        row[
                            "min_t2_echo_us"
                        ],
                    "memory_min_t2_echo_us":
                        row[
                            "memory_min_t2_echo_us"
                        ],
                    "min_cz_fidelity":
                        row[
                            "min_cz_fidelity"
                        ],
                    "mean_cz_fidelity":
                        row[
                            "mean_cz_fidelity"
                        ],
                    "max_asymmetric_readout_error":
                        row[
                            "max_asymmetric_readout_error"
                        ],
                    "min_prx_fidelity":
                        row[
                            "min_prx_fidelity"
                        ],
                    "prx_duration_min_ns":
                        row[
                            "prx_duration_min_ns"
                        ],
                    "prx_duration_max_ns":
                        row[
                            "prx_duration_max_ns"
                        ],
                    "cz_duration_min_ns":
                        row[
                            "cz_duration_min_ns"
                        ],
                    "cz_duration_max_ns":
                        row[
                            "cz_duration_max_ns"
                        ],
                    "raw_duration_complete_prx_cz":
                        row[
                            "raw_duration_complete_prx_cz"
                        ],
                })

            s3 = s3.reset_index(
                drop=True
            )

            for rank, row in s3.iterrows():
                out = row.to_dict()
                out[
                    "rule5_rank_within_top3"
                ] = int(
                    rank + 1
                )
                top3_rows.append(
                    out
                )

            print(
                f"{topology}: "
                f"valid={len(group)} | "
                f"Pareto={len(p)} | "
                f"CZ10={len(s10)} | "
                f"T2-5={len(s5)} | "
                f"TOP3={len(s3)}"
            )

            for rank, row in s3.iterrows():
                print(
                    "  "
                    f"#{rank+1} "
                    f"{row['layout']} | "
                    f"CZmin={row['min_cz_fidelity']:.6f} | "
                    f"T2min={row['min_t2_echo_us']:.3f} us | "
                    f"T2mem={row['memory_min_t2_echo_us']:.3f} us | "
                    f"ROmax={100.0*row['max_asymmetric_readout_error']:.3f}% | "
                    f"PRX={row['prx_duration_min_ns']:.1f}"
                    f"-{row['prx_duration_max_ns']:.1f} ns | "
                    f"CZdur={row['cz_duration_min_ns']:.1f}"
                    f"-{row['cz_duration_max_ns']:.1f} ns"
                )

    trace = pd.DataFrame(
        trace_rows
    )

    top3 = pd.DataFrame(
        top3_rows
    )

    trace.to_csv(
        OUT_TRACE,
        index=False,
    )

    top3.to_csv(
        OUT_TOP3,
        index=False,
    )

    # Compact duration-only table for the retained layouts.
    duration_columns = [
        "backend",
        "topology",
        "rule5_rank_within_top3",
        "layout",
        "min_prx_fidelity",
        "prx_duration_min_ns",
        "prx_duration_median_ns",
        "prx_duration_max_ns",
        "cz_duration_min_ns",
        "cz_duration_median_ns",
        "cz_duration_max_ns",
        "measure_duration_min_ns",
        "measure_duration_median_ns",
        "measure_duration_max_ns",
        "cz_implementation_by_edge",
        "cz_duration_resolution",
        "raw_duration_complete_prx_cz",
    ]

    duration_summary = top3[
        [
            c
            for c in duration_columns
            if c in top3.columns
        ]
    ].copy()

    duration_summary.to_csv(
        OUT_DURATION,
        index=False,
    )

    # Audits.
    expected_groups = (
        len(BACKENDS)
        * len(TOPOLOGIES)
    )

    observed_top3_groups = (
        top3[
            [
                "backend",
                "topology",
            ]
        ]
        .drop_duplicates()
        .shape[0]
        if not top3.empty
        else 0
    )

    duration_complete_top3 = (
        int(
            top3[
                "raw_duration_complete_prx_cz"
            ]
            .astype(bool)
            .sum()
        )
        if not top3.empty
        else 0
    )

    manifest = {
        "stage":
            "12.5F",
        "rule5_definition":
            {
                "pareto_core": {
                    "min_t1_us":
                        "maximize",
                    "min_t2_echo_us":
                        "maximize",
                    "max_asymmetric_readout_error":
                        "minimize",
                    "min_cz_fidelity":
                        "maximize",
                },
                "hierarchy": [
                    "Pareto",
                    "top 10 highest min_cz_fidelity",
                    "top 5 highest min_t2_echo_us",
                    "top 3 highest memory_min_t2_echo_us",
                ],
                "weighted_score":
                    False,
                "top3_retained":
                    True,
            },
        "raw_iqm_only":
            True,
        "synthetic_noise_parameters":
            False,
        "p_depol_calculated":
            False,
        "properties_used_for_selection": [
            "T1",
            "T2_echo",
            "CZ fidelity",
            "asymmetric readout errors",
        ],
        "properties_recorded_for_commentary": [
            "PRX fidelity",
            "PRX duration",
            "CZ duration",
            "measurement duration",
            "mean CZ fidelity",
        ],
        "topologies":
            list(TOPOLOGIES),
        "backends":
            list(BACKENDS),
        "expected_backend_topology_groups":
            expected_groups,
        "observed_selected_top3_groups":
            int(
                observed_top3_groups
            ),
        "selected_layout_rows":
            int(
                len(top3)
            ),
        "selected_layouts_with_complete_prx_cz_duration":
            duration_complete_top3,
        "qpu_jobs":
            0,
        "simulator_jobs":
            0,
        "shots":
            0,
        "forecast_data_loaded":
            False,
        "test_2026_used":
            False,
        "next_step": (
            "Interpret the H0-H6 Rule-5 refresh. "
            "Do not derive p_depol. The next hardware/noise step must "
            "continue using raw IQM properties and explicitly documented "
            "native simulator semantics only."
        ),
    }

    with open(
        OUT_MANIFEST,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 132)
    print(
        "12.5F SUMMARY"
    )
    print("=" * 132)
    print(
        f"Backend/topology groups expected: {expected_groups}"
    )
    print(
        f"Backend/topology groups with retained top3: {observed_top3_groups}"
    )
    print(
        f"Selected physical-layout rows: {len(top3)}"
    )
    print(
        "Selected layouts with complete raw PRX+CZ durations: "
        f"{duration_complete_top3}/{len(top3)}"
    )
    print()
    print(
        "No p_depol or synthetic noise parameter was calculated."
    )
    print(
        "No QPU/simulator jobs were submitted. 2026 untouched."
    )
    print()
    print("Saved:")
    for path in outputs:
        print(
            f"  {path}"
        )
    print()
    print(
        "STOP HERE. Send this output before the next simulation/hardware step."
    )


if __name__ == "__main__":
    main()
