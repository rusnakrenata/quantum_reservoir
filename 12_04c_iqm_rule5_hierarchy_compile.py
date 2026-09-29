"""
Week 12.4C — IQM corrected Rule-5 hierarchy + explicit circuit-duration audit.

This script corrects two issues discovered after Week 12.4B:

1) Rule-5 physical selection must mirror the IBM hierarchy exactly:
       Pareto
         -> 10 best worst-edge CZ
         -> 5 best global minimum T2_echo
         -> 3 best memory minimum T2_echo

   No weighted score is used and no single layout is forced at this stage.

2) T_circuit was NaN in 12.4B because Qiskit's estimate_duration() requires
   instruction-duration metadata in the Target.  Instead of silently swallowing
   the exception, this script explicitly audits duration coverage in both:
       backend.get_real_target()   (the target used for strict compilation)
       backend.target              (diagnostic only; may contain fictional edges)

   Circuit duration is reported only if the SAME real target used for compilation
   contains enough duration data.  Otherwise the CSV records:
       duration_status = "UNAVAILABLE"
       duration_error  = exact Qiskit exception text

Scientific constraints
----------------------
- Uses the seven R1/R2/R3/R4 logical candidates already frozen in 12.4B.
- Refresh 12.1 discovery + 12.2A region search before running this stage.
- Emerald and Garnet only.
- Actual RWP measurement circuits are strictly compiled.
- backend.get_real_target()
- routing_method="none"
- zero SWAP required.
- No QPU jobs.
- No shots executed.
- 2025 targets are not used for selection.
- 2026 is not loaded.

Run:
    python 12_04c_iqm_rule5_hierarchy_compile.py --overwrite
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from qiskit import transpile

from iqm.qiskit_iqm.iqm_provider import IQMProvider
from iqm_account import get_authentication


# =============================================================================
# PATHS / SETTINGS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

BASE_FILE = HERE / "12_04b_iqm_multi_rule_physical_compile_closure.py"
COMMON_FILE = HERE / "12_03d_iqm_rwp_common.py"

CANDIDATE_FILE = RESULTS / "12_04b_iqm_rule1_rule4_candidates.csv"
DISCOVERY_JSON = RESULTS / "12_01_iqm_discovery.json"
PARETO_CSV = RESULTS / "12_02_iqm_pareto_embeddings.csv"

OUT_TRACE = RESULTS / "12_04c_iqm_rule5_hierarchy_trace.csv"
OUT_SELECTED = RESULTS / "12_04c_iqm_rule5_selected_top3.csv"
OUT_DURATION_AUDIT = RESULTS / "12_04c_iqm_duration_target_audit.json"
OUT_DETAIL = RESULTS / "12_04c_iqm_compile_detail.csv"
OUT_ENDPOINT = RESULTS / "12_04c_iqm_compile_per_endpoint.csv"
OUT_SUMMARY = RESULTS / "12_04c_iqm_compile_summary.csv"
OUT_MANIFEST = RESULTS / "12_04c_iqm_manifest.json"

BACKENDS = ("emerald", "garnet")
TOPOLOGIES = ("H5", "H6")

SEED_TRANSPILE = 79001
OPTIMIZATION_LEVEL = 1
DEFAULT_SAMPLES = 5

CZ_TOP_N = 10
GLOBAL_T2_TOP_N = 5
MEMORY_T2_TOP_N = 3


# =============================================================================
# IMPORT EXISTING FROZEN HELPERS
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    spec.loader.exec_module(mod)
    return mod


base = load_module(
    BASE_FILE,
    "qrc_1204c_base",
)

common = load_module(
    COMMON_FILE,
    "qrc_1204c_common",
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def finite_float(value, default=np.nan):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)

    return x if math.isfinite(x) else float(default)


def normalize_edge(a: Any, b: Any):
    return tuple(sorted((str(a), str(b)), key=str))


def operation_counts(qc):
    return dict(
        sorted(
            Counter(
                instruction.operation.name
                for instruction in qc.data
            ).items()
        )
    )


def bool_from_csv(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "1", "true", "yes", "y"
    }


# =============================================================================
# EXACT IBM-STYLE RULE-5 HIERARCHY
# =============================================================================

def hierarchical_rule5_selection(pareto: pd.DataFrame):
    """
    Apply independently per (backend, topology):

        Pareto
          -> 10 highest minimum CZ fidelity
          -> 5 highest global minimum T2_echo
          -> 3 highest memory minimum T2_echo

    Equivalent CZ language:
        highest min_cz_fidelity == lowest worst-edge CZ error.

    Tie-breaks never replace the primary hierarchy; they only make ordering
    deterministic inside equal values.
    """
    traces = []
    selected = []

    for backend in BACKENDS:
        for H in TOPOLOGIES:
            g = pareto[
                (pareto["backend"].astype(str) == backend)
                & (pareto["topology"].astype(str) == H)
            ].copy()

            if len(g) == 0:
                raise RuntimeError(
                    f"No Pareto layouts found for {backend}/{H}."
                )

            # The input file is already Pareto-only, but preserve explicit stage.
            g["rule5_stage_pareto"] = True

            # -------------------------------------------------------------
            # Stage 1: top 10 by best worst-edge CZ.
            # -------------------------------------------------------------
            stage1 = (
                g.sort_values(
                    [
                        "min_cz_fidelity",
                        "min_t2_echo_us",
                        "memory_min_t2_echo_us",
                        "min_readout_fidelity",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                        False,
                    ],
                )
                .head(CZ_TOP_N)
                .copy()
            )

            stage1["rule5_stage_cz_top10"] = True

            # -------------------------------------------------------------
            # Stage 2: from those 10, top 5 global T2_echo.
            # -------------------------------------------------------------
            stage2 = (
                stage1.sort_values(
                    [
                        "min_t2_echo_us",
                        "memory_min_t2_echo_us",
                        "min_cz_fidelity",
                        "min_readout_fidelity",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                        False,
                    ],
                )
                .head(GLOBAL_T2_TOP_N)
                .copy()
            )

            stage2["rule5_stage_global_t2_top5"] = True

            # -------------------------------------------------------------
            # Stage 3: from those 5, top 3 memory T2_echo.
            # -------------------------------------------------------------
            stage3 = (
                stage2.sort_values(
                    [
                        "memory_min_t2_echo_us",
                        "min_t2_echo_us",
                        "min_cz_fidelity",
                        "min_readout_fidelity",
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                        False,
                    ],
                )
                .head(MEMORY_T2_TOP_N)
                .copy()
            )

            stage3["rule5_stage_memory_t2_top3"] = True
            stage3["rule5_rank_within_top3"] = np.arange(
                1,
                len(stage3) + 1,
            )

            # Build a full trace with stage membership flags.
            trace = g.copy()

            for col in [
                "rule5_stage_cz_top10",
                "rule5_stage_global_t2_top5",
                "rule5_stage_memory_t2_top3",
            ]:
                trace[col] = False

            key_col = "embedding_id"

            stage1_ids = set(stage1[key_col].astype(str))
            stage2_ids = set(stage2[key_col].astype(str))
            stage3_ids = set(stage3[key_col].astype(str))

            trace.loc[
                trace[key_col].astype(str).isin(stage1_ids),
                "rule5_stage_cz_top10",
            ] = True

            trace.loc[
                trace[key_col].astype(str).isin(stage2_ids),
                "rule5_stage_global_t2_top5",
            ] = True

            trace.loc[
                trace[key_col].astype(str).isin(stage3_ids),
                "rule5_stage_memory_t2_top3",
            ] = True

            traces.append(trace)
            selected.append(stage3)

            print()
            print(
                f"{backend.upper()} {H}: "
                f"Pareto={len(g)} -> CZ={len(stage1)} "
                f"-> global T2echo={len(stage2)} "
                f"-> memory T2echo={len(stage3)}"
            )

            print(
                stage3[
                    [
                        "embedding_id",
                        "layout",
                        "min_cz_fidelity",
                        "min_t2_echo_us",
                        "memory_min_t2_echo_us",
                        "min_readout_fidelity",
                        "rule5_rank_within_top3",
                    ]
                ].to_string(index=False)
            )

    trace_df = pd.concat(
        traces,
        ignore_index=True,
    )

    selected_df = pd.concat(
        selected,
        ignore_index=True,
    )

    return trace_df, selected_df


# =============================================================================
# TARGET-DURATION AUDIT
# =============================================================================

def target_duration_coverage(target):
    """
    Inspect InstructionProperties.duration entries in a Qiskit Target.

    Returns only factual target metadata; it does not estimate missing values.
    """
    total_loci = 0
    loci_with_properties = 0
    loci_with_duration = 0

    by_operation = {}

    operation_names = sorted(
        str(name)
        for name in target.operation_names
    )

    for name in operation_names:
        op_total = 0
        op_props = 0
        op_duration = 0
        examples_missing = []

        try:
            qarg_map = target[name]
        except Exception as exc:
            by_operation[name] = {
                "target_access_error": (
                    f"{type(exc).__name__}: {exc}"
                )
            }
            continue

        try:
            items = list(qarg_map.items())
        except Exception:
            items = []

        for qargs, props in items:
            total_loci += 1
            op_total += 1

            if props is not None:
                loci_with_properties += 1
                op_props += 1

                duration = getattr(
                    props,
                    "duration",
                    None,
                )

                if (
                    duration is not None
                    and math.isfinite(float(duration))
                ):
                    loci_with_duration += 1
                    op_duration += 1

                elif len(examples_missing) < 5:
                    examples_missing.append(
                        str(qargs)
                    )

            elif len(examples_missing) < 5:
                examples_missing.append(
                    str(qargs)
                )

        by_operation[name] = {
            "loci_total": op_total,
            "loci_with_instruction_properties": op_props,
            "loci_with_duration": op_duration,
            "missing_duration_examples": examples_missing,
        }

    return {
        "operation_names": operation_names,
        "loci_total": total_loci,
        "loci_with_instruction_properties": loci_with_properties,
        "loci_with_duration": loci_with_duration,
        "duration_coverage_fraction": (
            loci_with_duration / total_loci
            if total_loci
            else 0.0
        ),
        "by_operation": by_operation,
    }


def estimate_duration_explicit(circuit, target):
    """
    Do NOT swallow the Qiskit exception.

    Returns:
        duration_us
        duration_status
        duration_error
    """
    try:
        seconds = circuit.estimate_duration(
            target,
            unit="s",
        )

        seconds = float(seconds)

        if not math.isfinite(seconds):
            return (
                np.nan,
                "UNAVAILABLE",
                "estimate_duration returned a non-finite value",
            )

        return (
            seconds * 1e6,
            "AVAILABLE",
            "",
        )

    except Exception as exc:
        return (
            np.nan,
            "UNAVAILABLE",
            f"{type(exc).__name__}: {exc}",
        )


# =============================================================================
# STRICT ACTUAL-CIRCUIT COMPILATION
# =============================================================================

def compile_one(
    logical,
    backend,
    real_target,
    initial_layout,
    expected_edges,
):
    tqc = transpile(
        logical,
        target=real_target,
        initial_layout=initial_layout,
        routing_method="none",
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
    )

    counts = operation_counts(tqc)

    loci, mapping_mode = (
        base.strict_helper
        .physical_two_qubit_edges_from_transpiled(
            tqc,
            backend,
            initial_layout,
        )
    )

    all_two_q_edges = {
        item["pair"]
        for item in loci
    }

    cz_edges = {
        item["pair"]
        for item in loci
        if item["operation"] == "cz"
    }

    unexpected_two_q = (
        all_two_q_edges
        - expected_edges
    )

    unexpected_cz = (
        cz_edges
        - expected_edges
    )

    swap_count = int(
        counts.get("swap", 0)
    )

    routing_perm, routing_identity = (
        base.strict_helper
        .routing_permutation_info(
            tqc
        )
    )

    allowed = set(
        str(name)
        for name in real_target.operation_names
    ) | {"barrier"}

    non_native_ops = sorted(
        set(counts)
        - allowed
    )

    strict_pass = (
        swap_count == 0
        and routing_identity is True
        and not unexpected_two_q
        and not unexpected_cz
        and not non_native_ops
    )

    (
        duration_us,
        duration_status,
        duration_error,
    ) = estimate_duration_explicit(
        tqc,
        real_target,
    )

    return {
        "strict_pass": bool(strict_pass),
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cz_count": int(
            counts.get("cz", 0)
        ),
        "swap_count": swap_count,
        "reset_count": int(
            counts.get("reset", 0)
        ),
        "measure_count": int(
            counts.get("measure", 0)
        ),
        "duration_us": duration_us,
        "duration_status": duration_status,
        "duration_error": duration_error,
        "operations_json": json.dumps(
            counts,
            sort_keys=True,
        ),
        "mapping_mode": mapping_mode,
        "routing_permutation_json": json.dumps(
            routing_perm
        ),
        "routing_identity": routing_identity,
        "unexpected_two_q_edges_json": json.dumps(
            sorted(
                [list(x) for x in unexpected_two_q]
            )
        ),
        "unexpected_cz_edges_json": json.dumps(
            sorted(
                [list(x) for x in unexpected_cz]
            )
        ),
        "non_native_ops_json": json.dumps(
            non_native_ops
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
    )

    parser.add_argument(
        "--backend",
        choices=["all", *BACKENDS],
        default="all",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError(
            "--samples must be >= 1."
        )

    required = [
        BASE_FILE,
        COMMON_FILE,
        CANDIDATE_FILE,
        DISCOVERY_JSON,
        PARETO_CSV,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    outputs = [
        OUT_TRACE,
        OUT_SELECTED,
        OUT_DURATION_AUDIT,
        OUT_DETAIL,
        OUT_ENDPOINT,
        OUT_SUMMARY,
        OUT_MANIFEST,
    ]

    if args.overwrite:
        for path in outputs:
            if path.exists():
                path.unlink()

    candidates = pd.read_csv(
        CANDIDATE_FILE
    )

    pareto = pd.read_csv(
        PARETO_CSV
    )

    with open(
        DISCOVERY_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        discovery = json.load(f)

    # -------------------------------------------------------------------------
    # Correct Rule-5 hierarchy.
    # -------------------------------------------------------------------------
    print("=" * 126)
    print(
        "WEEK 12.4C — CORRECTED IQM RULE-5 HIERARCHY + DURATION AUDIT"
    )
    print("=" * 126)

    trace_df, selected_layouts = (
        hierarchical_rule5_selection(
            pareto
        )
    )

    trace_df.to_csv(
        OUT_TRACE,
        index=False,
    )

    selected_layouts.to_csv(
        OUT_SELECTED,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Provider / data initialization.
    # -------------------------------------------------------------------------
    token, server_url, _ = (
        get_authentication()
    )

    backend_names = (
        list(BACKENDS)
        if args.backend == "all"
        else [args.backend]
    )

    # 2022-2025 input feature construction only. No 2026 path is called.
    work_tv, cols, _ = (
        common.load_train_validation()
    )

    angle_cache = {}
    detail_rows = []
    duration_audit = {}

    # -------------------------------------------------------------------------
    # Backend loop.
    # -------------------------------------------------------------------------
    for backend_name in backend_names:
        system = discovery[
            "systems"
        ][backend_name]

        calibration_set_id = (
            system[
                "summary"
            ][
                "calibration_set_id"
            ]
        )

        provider = IQMProvider(
            server_url,
            quantum_computer=backend_name,
            token=token,
        )

        backend = provider.get_backend(
            calibration_set_id=UUID(
                calibration_set_id
            ),
            use_metrics=False,
        )

        real_target = (
            backend.get_real_target()
        )

        # Explicitly audit duration metadata.
        duration_audit[
            backend_name
        ] = {
            "calibration_set_id":
                calibration_set_id,

            "real_target":
                target_duration_coverage(
                    real_target
                ),

            "backend_target_diagnostic_only":
                target_duration_coverage(
                    backend.target
                ),
        }

        print()
        print("=" * 126)
        print(
            f"BACKEND {backend_name.upper()} | "
            f"calibration_set_id={calibration_set_id}"
        )
        print("=" * 126)

        ra = duration_audit[
            backend_name
        ][
            "real_target"
        ]

        ba = duration_audit[
            backend_name
        ][
            "backend_target_diagnostic_only"
        ]

        print(
            "REAL target duration coverage: "
            f"{ra['loci_with_duration']}/{ra['loci_total']} "
            f"({100.0 * ra['duration_coverage_fraction']:.2f}%)"
        )

        print(
            "backend.target duration coverage (diagnostic only): "
            f"{ba['loci_with_duration']}/{ba['loci_total']} "
            f"({100.0 * ba['duration_coverage_fraction']:.2f}%)"
        )

        physical_to_index = (
            base.strict_helper
            .qiskit_physical_index_map(
                backend
            )
        )

        # -------------------------------------------------------------
        # Logical candidate loop.
        # -------------------------------------------------------------
        for cand_i, row in candidates.iterrows():
            candidate = (
                base.candidate_from_trial_row(
                    row
                )
            )

            H = str(
                candidate["topology"]
            )

            uid = str(
                row["candidate_uid"]
            )

            layouts = selected_layouts[
                (selected_layouts["backend"].astype(str) == backend_name)
                & (selected_layouts["topology"].astype(str) == H)
            ].copy()

            if len(layouts) != MEMORY_T2_TOP_N:
                raise RuntimeError(
                    f"Expected {MEMORY_T2_TOP_N} final Rule-5 layouts "
                    f"for {backend_name}/{H}, got {len(layouts)}."
                )

            alpha = float(
                candidate["alpha"]
            )

            if alpha not in angle_cache:
                angle_cache[
                    alpha
                ] = (
                    common.make_angles(
                        work_tv,
                        cols,
                        alpha,
                    )
                )

            angles = (
                angle_cache[
                    alpha
                ]
            )

            valid_endpoints = np.arange(
                common.N_TRAIN,
                common.N_TRAIN
                + common.N_VAL,
            )

            sample_idx = np.linspace(
                0,
                len(valid_endpoints) - 1,
                min(
                    int(args.samples),
                    len(valid_endpoints),
                ),
                dtype=int,
            )

            sampled_endpoints = (
                valid_endpoints[
                    sample_idx
                ]
            )

            print(
                f"[{cand_i+1:02d}/{len(candidates)}] "
                f"{uid} | roles={row['selected_rules']} | "
                f"Rule-5 layouts={len(layouts)}"
            )

            # ---------------------------------------------------------
            # Final top-3 physical layouts.
            # ---------------------------------------------------------
            for _, layout_row in layouts.iterrows():
                physical_names = [
                    layout_row["q0_C"],
                    layout_row["q1_D"],
                    layout_row["q2_P"],
                    layout_row["q3_H"],
                    layout_row["q4_M1"],
                    layout_row["q5_M2"],
                ]

                missing = [
                    qb
                    for qb in physical_names
                    if qb not in physical_to_index
                ]

                if missing:
                    raise RuntimeError(
                        f"{backend_name}/{H}: "
                        f"physical names absent from backend: {missing}"
                    )

                initial_layout = [
                    physical_to_index[qb]
                    for qb in physical_names
                ]

                expected_edges = {
                    normalize_edge(
                        physical_names[int(a)],
                        physical_names[int(b)],
                    )
                    for a, b in common.TOPOLOGY_EDGES[H]
                }

                for endpoint in sampled_endpoints:
                    for setting in base.readout_settings(
                        candidate["readout"]
                    ):
                        logical = (
                            base.build_measurement_circuit(
                                candidate,
                                angles,
                                int(endpoint),
                                setting,
                            )
                        )

                        result = compile_one(
                            logical,
                            backend,
                            real_target,
                            initial_layout,
                            expected_edges,
                        )

                        detail_rows.append(
                            {
                                "candidate_uid":
                                    uid,

                                "selected_rules":
                                    str(row["selected_rules"]),

                                "topology":
                                    H,

                                "window":
                                    int(candidate["window"]),

                                "r":
                                    int(candidate["r"]),

                                "trotter_search_id":
                                    str(candidate["trotter_search_id"]),

                                "readout":
                                    str(candidate["readout"]),

                                "cv_rmse":
                                    float(row["cv_rmse"]),

                                "cv_rmse_std":
                                    float(row["cv_rmse_std"]),

                                "backend":
                                    backend_name,

                                "calibration_set_id":
                                    calibration_set_id,

                                "embedding_id":
                                    str(layout_row["embedding_id"]),

                                "rule5_rank_within_top3":
                                    int(
                                        layout_row[
                                            "rule5_rank_within_top3"
                                        ]
                                    ),

                                "layout":
                                    str(layout_row["layout"]),

                                "q0_C":
                                    layout_row["q0_C"],

                                "q1_D":
                                    layout_row["q1_D"],

                                "q2_P":
                                    layout_row["q2_P"],

                                "q3_H":
                                    layout_row["q3_H"],

                                "q4_M1":
                                    layout_row["q4_M1"],

                                "q5_M2":
                                    layout_row["q5_M2"],

                                "min_cz_fidelity":
                                    finite_float(
                                        layout_row[
                                            "min_cz_fidelity"
                                        ]
                                    ),

                                "min_t2_echo_us":
                                    finite_float(
                                        layout_row[
                                            "min_t2_echo_us"
                                        ]
                                    ),

                                "memory_min_t2_echo_us":
                                    finite_float(
                                        layout_row[
                                            "memory_min_t2_echo_us"
                                        ]
                                    ),

                                "min_t1_us":
                                    finite_float(
                                        layout_row[
                                            "min_t1_us"
                                        ]
                                    ),

                                "memory_min_t1_us":
                                    finite_float(
                                        layout_row[
                                            "memory_min_t1_us"
                                        ]
                                    ),

                                "min_readout_fidelity":
                                    finite_float(
                                        layout_row[
                                            "min_readout_fidelity"
                                        ]
                                    ),

                                "endpoint":
                                    int(endpoint),

                                "setting":
                                    str(setting[0]),

                                "2025_used_for_selection":
                                    False,

                                "qpu_job_submitted":
                                    False,

                                **result,
                            }
                        )

    # -------------------------------------------------------------------------
    # Save duration audit before any aggregate logic.
    # -------------------------------------------------------------------------
    with open(
        OUT_DURATION_AUDIT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            duration_audit,
            f,
            indent=2,
        )

    detail = pd.DataFrame(
        detail_rows
    )

    detail.to_csv(
        OUT_DETAIL,
        index=False,
    )

    if len(detail) == 0:
        raise RuntimeError(
            "No compile-detail rows produced."
        )

    # Strict compile is non-negotiable.
    strict_failures = detail[
        ~detail[
            "strict_pass"
        ].astype(bool)
    ]

    if len(strict_failures):
        raise RuntimeError(
            f"Strict Rule-5 compilation failed for "
            f"{len(strict_failures)} setting-level circuits."
        )

    # -------------------------------------------------------------------------
    # Per endpoint aggregation.
    # -------------------------------------------------------------------------
    endpoint_group = [
        "candidate_uid",
        "selected_rules",
        "topology",
        "window",
        "r",
        "trotter_search_id",
        "readout",
        "cv_rmse",
        "cv_rmse_std",
        "backend",
        "calibration_set_id",
        "embedding_id",
        "rule5_rank_within_top3",
        "layout",
        "q0_C",
        "q1_D",
        "q2_P",
        "q3_H",
        "q4_M1",
        "q5_M2",
        "min_cz_fidelity",
        "min_t2_echo_us",
        "memory_min_t2_echo_us",
        "min_t1_us",
        "memory_min_t1_us",
        "min_readout_fidelity",
        "endpoint",
    ]

    endpoint_rows = []

    for key, g in detail.groupby(
        endpoint_group,
        dropna=False,
        sort=False,
    ):
        meta = dict(
            zip(
                endpoint_group,
                key
                if isinstance(key, tuple)
                else (key,),
            )
        )

        duration_values = pd.to_numeric(
            g["duration_us"],
            errors="coerce",
        )

        all_duration_available = bool(
            (
                g["duration_status"].astype(str)
                == "AVAILABLE"
            ).all()
        )

        duration_errors = sorted(
            set(
                err
                for err in g[
                    "duration_error"
                ].astype(str)
                if err and err != "nan"
            )
        )

        max_setting_duration = (
            float(
                duration_values.max()
            )
            if (
                all_duration_available
                and duration_values.notna().all()
            )
            else np.nan
        )

        feature_duration = (
            float(
                duration_values.sum()
            )
            if (
                all_duration_available
                and duration_values.notna().all()
            )
            else np.nan
        )

        endpoint_rows.append(
            {
                **meta,

                "n_settings":
                    int(
                        g["setting"].nunique()
                    ),

                "NCZ_feature_vector":
                    int(
                        pd.to_numeric(
                            g["cz_count"],
                            errors="raise",
                        ).sum()
                    ),

                "NSWAP_feature_vector":
                    int(
                        pd.to_numeric(
                            g["swap_count"],
                            errors="raise",
                        ).sum()
                    ),

                "reset_feature_vector":
                    int(
                        pd.to_numeric(
                            g["reset_count"],
                            errors="raise",
                        ).sum()
                    ),

                "depth_max_setting":
                    int(
                        pd.to_numeric(
                            g["depth"],
                            errors="raise",
                        ).max()
                    ),

                "depth_sum_settings":
                    int(
                        pd.to_numeric(
                            g["depth"],
                            errors="raise",
                        ).sum()
                    ),

                "duration_available_all_settings":
                    all_duration_available,

                "duration_unavailable_reason":
                    " | ".join(
                        duration_errors
                    ),

                "Tcircuit_max_setting_us":
                    max_setting_duration,

                "Tcircuit_feature_vector_us":
                    feature_duration,
            }
        )

    endpoint_df = pd.DataFrame(
        endpoint_rows
    )

    endpoint_df.to_csv(
        OUT_ENDPOINT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Candidate × backend × physical layout summary.
    # -------------------------------------------------------------------------
    summary_group = [
        c
        for c in endpoint_group
        if c != "endpoint"
    ]

    summary_rows = []

    for key, g in endpoint_df.groupby(
        summary_group,
        dropna=False,
        sort=False,
    ):
        meta = dict(
            zip(
                summary_group,
                key
                if isinstance(key, tuple)
                else (key,),
            )
        )

        duration_ok = bool(
            g[
                "duration_available_all_settings"
            ].astype(bool).all()
        )

        reasons = sorted(
            set(
                str(x)
                for x in g[
                    "duration_unavailable_reason"
                ]
                if str(x).strip()
                and str(x) != "nan"
            )
        )

        summary_rows.append(
            {
                **meta,

                "sampled_endpoints":
                    int(
                        g["endpoint"].nunique()
                    ),

                "strict_pass":
                    True,

                "n_settings":
                    float(
                        g["n_settings"].median()
                    ),

                "NCZ_median":
                    float(
                        g["NCZ_feature_vector"].median()
                    ),

                "NSWAP_median":
                    float(
                        g["NSWAP_feature_vector"].median()
                    ),

                "reset_median":
                    float(
                        g["reset_feature_vector"].median()
                    ),

                "depth_max_setting_median":
                    float(
                        g["depth_max_setting"].median()
                    ),

                "depth_sum_settings_median":
                    float(
                        g["depth_sum_settings"].median()
                    ),

                "duration_available":
                    duration_ok,

                "duration_unavailable_reason":
                    " | ".join(reasons),

                "Tcircuit_max_setting_us_median":
                    (
                        float(
                            g[
                                "Tcircuit_max_setting_us"
                            ].median()
                        )
                        if duration_ok
                        else np.nan
                    ),

                "Tcircuit_feature_vector_us_median":
                    (
                        float(
                            g[
                                "Tcircuit_feature_vector_us"
                            ].median()
                        )
                        if duration_ok
                        else np.nan
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    manifest = {
        "stage": "12.4C",
        "rule5_hierarchy": [
            "Pareto",
            f"top {CZ_TOP_N} highest min_cz_fidelity",
            f"top {GLOBAL_T2_TOP_N} highest min_t2_echo_us",
            f"top {MEMORY_T2_TOP_N} highest memory_min_t2_echo_us",
        ],
        "n_logical_candidates":
            int(len(candidates)),
        "n_final_layouts_per_backend_topology":
            MEMORY_T2_TOP_N,
        "backends":
            backend_names,
        "samples_per_candidate_layout":
            int(args.samples),
        "optimization_level":
            OPTIMIZATION_LEVEL,
        "routing_method":
            "none",
        "compile_target":
            "backend.get_real_target()",
        "duration_policy": (
            "Report T_circuit only when estimate_duration succeeds on the "
            "same real target used for strict compilation; never impute."
        ),
        "qpu_jobs":
            False,
        "shots":
            0,
        "uses_2025_target_for_selection":
            False,
        "loads_2026":
            False,
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

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------
    print()
    print("=" * 126)
    print(
        "12.4C COMPLETE — CORRECTED RULE-5 TOP-3"
    )
    print("=" * 126)

    print(
        f"Logical candidates:          {len(candidates)}"
    )
    print(
        f"Final physical layouts:      {len(selected_layouts)} "
        "(3 per backend/topology)"
    )
    print(
        f"Compile summary rows:        {len(summary)}"
    )
    print(
        f"Strict failures:             0"
    )

    print()
    print(
        "DURATION AVAILABILITY"
    )
    print("-" * 126)

    for backend_name in backend_names:
        ra = duration_audit[
            backend_name
        ][
            "real_target"
        ]

        print(
            f"{backend_name:8s}: "
            f"real-target loci with duration "
            f"{ra['loci_with_duration']}/{ra['loci_total']} "
            f"({100.0 * ra['duration_coverage_fraction']:.2f}%)"
        )

    print()
    print(
        summary[
            [
                "candidate_uid",
                "backend",
                "rule5_rank_within_top3",
                "layout",
                "min_cz_fidelity",
                "min_t2_echo_us",
                "memory_min_t2_echo_us",
                "NCZ_median",
                "NSWAP_median",
                "depth_max_setting_median",
                "duration_available",
                "Tcircuit_max_setting_us_median",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    for path in outputs:
        print(f"  {path}")

    print()
    print("No QPU jobs submitted.")
    print("No shots executed.")
    print("2026 untouched.")
    print(
        "STOP HERE. Interpret corrected Rule-5 layouts and duration audit "
        "before noise/QPU execution."
    )


if __name__ == "__main__":
    main()
