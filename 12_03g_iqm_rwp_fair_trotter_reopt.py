"""
Week 12.3G — IQM H5/H6 fair local Trotter-depth re-optimization.

Direct analogue of IBM RWP 09.04C2.

Scientific rule
---------------
For every topology-window branch (H5/H6 × W=1..28):

1. Take ONE shared parent:
       the Week-12.3F TRAINING-CV winner for that topology/window.
2. Use that exact same parent Hamiltonian as the starting point for
       r in {1, 2, 3}.
3. Give every r the identical local search budget implemented in
       common.trotter_local_candidates():
           - parent_same_H
           - 2 global-dynamics variations
           - 2 dt variations
           - 8 fixed-seed local perturbations
       = 13 dynamics trials per r.
4. For every dynamics trial, evaluate all five readout families and
   reselect Ridge lambda using 2022-2024 chronological CV only.
5. Save:
       - CV-best candidate per topology/window/r
       - hardware-aware representative per topology/window/r
       - final CV-best candidate per topology/window across r
       - final hardware-aware representative per topology/window across r

Important
---------
- r=1, r=2 and r=3 receive the SAME local search budget.
- We do NOT compare tuned r against untuned r.
- 2025 is diagnostic only.
- 2026 remains untouched.
- No QPU jobs are submitted.
- No weighted score is used.

Expected full-run counts
------------------------
56 topology-window branches
3 r values per branch
13 dynamics trials per r
5 readouts per dynamics trial

Total rows:
    56 * 3 * 13 * 5 = 10,920
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from importlib import import_module

import pandas as pd


common = import_module("12_03d_iqm_rwp_common")


# =============================================================================
# FROZEN C2 SETTINGS — MATCH IBM 09.04C2
# =============================================================================

R_VALUES = [1, 2, 3]
LOCAL_SEED = 42
N_RANDOM_LOCAL = 8

EXPECTED_DYNAMICS_PER_R = 13
EXPECTED_READOUTS = len(common.READOUTS)

BEST_CLASSICAL_CV = 3.392277


# =============================================================================
# CANONICAL J-COLUMN COMPATIBILITY
# =============================================================================
#
# The original 12.3D CSV writer used literal Trotter-edge orientation for
# several H5/H6 J columns (e.g. J30 instead of canonical J03).
#
# 12.3F_FIXED already writes canonical J columns, but this stage remains
# backward-compatible and also forces all NEW 12.3G output to use canonical
# column names.


def _read_coupling_from_row(
    row_dict: dict,
    topology: str,
    edge: tuple[int, int],
) -> float:

    canonical_col = common.EDGE_TO_COLUMN[topology][edge]
    direct_col = f"J{edge[0]}{edge[1]}"
    reverse_col = f"J{edge[1]}{edge[0]}"

    candidates = []

    for col in [
        canonical_col,
        direct_col,
        reverse_col,
    ]:
        if col not in candidates:
            candidates.append(col)

    for col in candidates:
        if (
            col in row_dict
            and not pd.isna(row_dict[col])
        ):
            return float(row_dict[col])

    raise KeyError(
        f"Missing coupling for topology={topology}, "
        f"edge={edge}. Tried columns {candidates}."
    )


def canonical_candidate_to_flat(
    candidate: dict,
) -> dict:
    """
    Serialize NEW 12.3G rows using the canonical J-column names from
    common.EDGE_TO_COLUMN, independent of the directed ordering used by the
    first-order Trotter product.
    """

    topology = str(candidate["topology"])

    row = {
        "topology":
            topology,

        "alpha":
            float(candidate["alpha"]),

        "dt":
            float(candidate["dt"]),

        "r":
            int(candidate["r"]),

        "hx":
            float(candidate["hx"]),

        "hy":
            float(candidate["hy"]),
    }

    for edge, value in candidate["J"].items():

        col = (
            common
            .EDGE_TO_COLUMN[topology][edge]
        )

        row[col] = float(value)

    for key in [
        "source_config_id",
        "source_candidate_id",
        "candidate_id",
        "search_id",
        "parent_id",
    ]:
        if key in candidate:
            row[key] = candidate[key]

    return row


# evaluate_candidate_window() resolves this symbol at runtime.
common.candidate_to_flat = (
    canonical_candidate_to_flat
)


# =============================================================================
# PARENT RECONSTRUCTION
# =============================================================================


def row_to_parent(
    row: pd.Series,
) -> dict:
    """
    One shared 12.3F TRAINING-CV parent per topology/window.

    The same parent is passed to r=1,2,3 before the equal-budget local
    Trotter search.
    """

    d = row.to_dict()
    topology = str(d["topology"])
    W = int(d["window"])

    J = {
        edge:
            _read_coupling_from_row(
                d,
                topology,
                edge,
            )

        for edge
        in common.TOPOLOGY_EDGES[topology]
    }

    parent = {
        "topology":
            topology,

        "alpha":
            float(d["alpha"]),

        "dt":
            float(d["dt"]),

        "r":
            int(
                d["r"]
                if "r" in d
                else d["trotter_r"]
            ),

        "hx":
            float(d["hx"]),

        "hy":
            float(d["hy"]),

        "J":
            J,

        "candidate_id":
            (
                f"{topology}_W{W:02d}"
                "_12p3F_CV_parent"
            ),

        "window":
            W,
    }

    return parent


# =============================================================================
# SELECTION HELPERS
# =============================================================================


def select_cv_best(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Training-CV primary selection.
    Hardware/robustness metrics only break ties.
    2025 is never used.
    """

    return (
        df.sort_values(
            [
                "cv_rmse",
                "shot_noise_forecast_sd_proxy_1024",
                "max_abs_raw_prediction_sensitivity",
                "n_settings",
                "feature_vector_cz",
            ]
        )
        .iloc[0]
        .copy()
    )


def compact_row(
    row: pd.Series,
) -> dict:
    """Small JSON-safe summary for the manifest."""

    return {
        "topology":
            str(row["topology"]),

        "window":
            int(row["window"]),

        "r":
            int(row["r"]),

        "search_id":
            str(row["trotter_search_id"]),

        "readout":
            str(row["readout"]),

        "cv_rmse":
            float(row["cv_rmse"]),

        "cv_rmse_std":
            float(row["cv_rmse_std"]),

        "validation_rmse_diagnostic":
            float(row["validation_rmse"]),

        "shot_noise_forecast_sd_proxy_1024":
            float(
                row[
                    "shot_noise_forecast_sd_proxy_1024"
                ]
            ),

        "max_shot_to_train_std_ratio_1024":
            float(
                row[
                    "max_shot_to_train_std_ratio_1024"
                ]
            ),

        "feature_vector_cz":
            int(row["feature_vector_cz"]),

        "n_settings":
            int(row["n_settings"]),
    }


# =============================================================================
# MAIN
# =============================================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete existing 12.3G outputs "
            "before the run."
        ),
    )

    parser.add_argument(
        "--max-branches",
        type=int,
        default=None,
        help=(
            "Optional smoke-test branch limit. "
            "Omit for the full 56-branch run."
        ),
    )

    args = parser.parse_args()

    src = (
        common.RESULTS
        / "12_03f_rwp_best_cv_per_topology_window.csv"
    )

    out_trials = (
        common.RESULTS
        / "12_03g_rwp_trotter_all_trials.csv"
    )

    out_cv_wr = (
        common.RESULTS
        / "12_03g_rwp_best_cv_per_topology_window_r.csv"
    )

    out_hw_wr = (
        common.RESULTS
        / "12_03g_rwp_hardware_aware_per_topology_window_r.csv"
    )

    out_cv_w = (
        common.RESULTS
        / "12_03g_rwp_best_cv_per_topology_window.csv"
    )

    out_hw_w = (
        common.RESULTS
        / "12_03g_rwp_hardware_aware_per_topology_window.csv"
    )

    out_manifest = (
        common.RESULTS
        / "12_03g_rwp_trotter_manifest.json"
    )

    outputs = [
        out_trials,
        out_cv_wr,
        out_hw_wr,
        out_cv_w,
        out_hw_w,
        out_manifest,
    ]

    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing. "
            "Run Week 12.3F first."
        )

    if args.overwrite:

        for path in outputs:

            if path.exists():
                path.unlink()

    parent_df = (
        pd.read_csv(src)
        .sort_values(
            [
                "topology",
                "window",
            ]
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Full-run structural audit.
    # -------------------------------------------------------------------------

    expected_topologies = set(
        common.DEFAULT_TOPOLOGIES
    )

    observed_topologies = set(
        parent_df["topology"]
        .astype(str)
    )

    if (
        observed_topologies
        != expected_topologies
    ):
        raise RuntimeError(
            "Unexpected topology set in 12.3F CV parents: "
            f"observed={sorted(observed_topologies)}, "
            f"expected={sorted(expected_topologies)}."
        )

    expected_branches = (
        len(common.DEFAULT_TOPOLOGIES)
        *
        len(common.DEFAULT_WINDOWS)
    )

    if len(parent_df) != expected_branches:
        raise RuntimeError(
            "Expected one 12.3F CV parent for every "
            f"H×W branch ({expected_branches}); "
            f"got {len(parent_df)}."
        )

    duplicate_count = int(
        parent_df.duplicated(
            [
                "topology",
                "window",
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "12.3F CV parent file contains duplicate "
            "topology-window rows."
        )

    if args.max_branches is not None:

        if args.max_branches < 1:
            raise ValueError(
                "--max-branches must be >= 1."
            )

        parent_df = (
            parent_df
            .head(args.max_branches)
            .copy()
        )

    work_tv, cols, y_all = (
        common.load_train_validation()
    )

    # -------------------------------------------------------------------------
    # Run header.
    # -------------------------------------------------------------------------

    print("=" * 128)
    print(
        "WEEK 12.3G — IQM H5/H6 FAIR LOCAL "
        "TROTTER-r RE-OPTIMIZATION"
    )
    print("=" * 128)

    print(
        "Direct analogue of IBM RWP 09.04C2."
    )

    print(
        "Shared parent per H×W = "
        "12.3F TRAINING-CV winner."
    )

    print(
        f"r values: {R_VALUES}"
    )

    print(
        "Equal local budget per r: "
        "parent_same_H + 2 global dynamics + "
        "2 dt + 8 fixed-seed local perturbations."
    )

    print(
        f"Random local children per r: "
        f"{N_RANDOM_LOCAL}; seed={LOCAL_SEED}"
    )

    print(
        f"Readouts per dynamics trial: "
        f"{EXPECTED_READOUTS}"
    )

    print(
        "Selection: 2022-2024 chronological "
        "CV only."
    )

    print(
        "2025 = diagnostic only; "
        "2026 = untouched / not loaded."
    )

    print(
        "No QPU jobs."
    )

    print()

    print(
        f"H×W branches to process: "
        f"{len(parent_df)}"
    )

    expected_rows_this_run = (
        len(parent_df)
        *
        len(R_VALUES)
        *
        EXPECTED_DYNAMICS_PER_R
        *
        EXPECTED_READOUTS
    )

    print(
        f"Expected all-trial rows: "
        f"{expected_rows_this_run}"
    )

    print()

    # -------------------------------------------------------------------------
    # Global accumulators for final summaries.
    # -------------------------------------------------------------------------

    final_cv_rows = []
    final_hw_rows = []

    # -------------------------------------------------------------------------
    # H×W loop.
    # -------------------------------------------------------------------------

    for branch_i, parent_row in (
        parent_df.iterrows()
    ):

        topology = str(
            parent_row["topology"]
        )

        W = int(
            parent_row["window"]
        )

        parent = row_to_parent(
            parent_row
        )

        print("-" * 128)

        print(
            f"[{branch_i + 1}/{len(parent_df)}] "
            f"{topology} W={W:02d} | "
            f"12.3F parent="
            f"{parent_row.get('trial_search_id', '')} / "
            f"{parent_row['readout']} | "
            f"CV={float(parent_row['cv_rmse']):.6f} | "
            f"r_parent={parent['r']}"
        )

        window_all_rows = []
        window_cv_r_rows = []
        window_hw_r_rows = []

        # ---------------------------------------------------------------------
        # Fair r loop.
        # ---------------------------------------------------------------------

        for r in R_VALUES:

            candidates = (
                common.trotter_local_candidates(
                    parent=parent,
                    r=int(r),
                    seed=LOCAL_SEED,
                    n_random=N_RANDOM_LOCAL,
                )
            )

            if (
                len(candidates)
                != EXPECTED_DYNAMICS_PER_R
            ):
                raise RuntimeError(
                    f"{topology} W={W} r={r}: "
                    "unexpected local-search size "
                    f"{len(candidates)}; expected "
                    f"{EXPECTED_DYNAMICS_PER_R}."
                )

            r_rows = []

            for trial_i, candidate in enumerate(
                candidates
            ):

                readout_df, _ = (
                    common.evaluate_candidate_window(
                        candidate=candidate,
                        window=W,
                        work_tv=work_tv,
                        cols=cols,
                        y_all=y_all,
                    )
                )

                readout_df[
                    "window"
                ] = int(W)

                readout_df[
                    "trotter_r_tested"
                ] = int(r)

                readout_df[
                    "trotter_search_id"
                ] = str(
                    candidate["search_id"]
                )

                readout_df[
                    "trotter_trial_index"
                ] = int(trial_i)

                readout_df[
                    "shared_parent_r"
                ] = int(
                    parent["r"]
                )

                readout_df[
                    "shared_parent_cv_rmse"
                ] = float(
                    parent_row["cv_rmse"]
                )

                readout_df[
                    "shared_parent_readout"
                ] = str(
                    parent_row["readout"]
                )

                readout_df[
                    "2025_used_for_selection"
                ] = False

                rows = readout_df.to_dict(
                    "records"
                )

                r_rows.extend(rows)
                window_all_rows.extend(rows)

            r_df = pd.DataFrame(
                r_rows
            )

            expected_r_rows = (
                EXPECTED_DYNAMICS_PER_R
                *
                EXPECTED_READOUTS
            )

            if len(r_df) != expected_r_rows:
                raise RuntimeError(
                    f"{topology} W={W} r={r}: "
                    f"got {len(r_df)} rows; "
                    f"expected {expected_r_rows}."
                )

            # -------------------------------------------------------------
            # Best training-CV row for THIS r.
            # -------------------------------------------------------------

            cv_best_r = select_cv_best(
                r_df
            )

            cv_best_r[
                "selection_rule"
            ] = (
                "training CV primary within fixed r; "
                "robustness/resources tie-break; "
                "2025 diagnostic only"
            )

            cv_best_r[
                "2025_used_for_selection"
            ] = False

            # -------------------------------------------------------------
            # Hardware-aware representative for THIS r.
            # -------------------------------------------------------------

            hw_best_r, hw_mode_r = (
                common
                .select_hardware_aware_row(
                    r_df
                )
            )

            hw_best_r = (
                hw_best_r.copy()
            )

            hw_best_r[
                "selection_rule"
            ] = hw_mode_r

            hw_best_r[
                "2025_used_for_selection"
            ] = False

            window_cv_r_rows.append(
                cv_best_r.to_dict()
            )

            window_hw_r_rows.append(
                hw_best_r.to_dict()
            )

            print(
                f"  r={r}: "
                f"CV-best="
                f"{cv_best_r['trotter_search_id']} / "
                f"{cv_best_r['readout']} "
                f"CV={cv_best_r['cv_rmse']:.6f} "
                f"Val={cv_best_r['validation_rmse']:.6f} "
                f"shotSD="
                f"{cv_best_r['shot_noise_forecast_sd_proxy_1024']:.3f} "
                f"maxR="
                f"{cv_best_r['max_shot_to_train_std_ratio_1024']:.2f} "
                f"|| HW="
                f"{hw_best_r['trotter_search_id']} / "
                f"{hw_best_r['readout']} "
                f"CV={hw_best_r['cv_rmse']:.6f} "
                f"maxR="
                f"{hw_best_r['max_shot_to_train_std_ratio_1024']:.2f}"
            )

        # ---------------------------------------------------------------------
        # Save this branch's complete trial rows and per-r winners.
        # ---------------------------------------------------------------------

        common.write_csv_incremental(
            out_trials,
            window_all_rows,
        )

        common.write_csv_incremental(
            out_cv_wr,
            window_cv_r_rows,
        )

        common.write_csv_incremental(
            out_hw_wr,
            window_hw_r_rows,
        )

        window_df = pd.DataFrame(
            window_all_rows
        )

        # ---------------------------------------------------------------------
        # FINAL training-CV winner across r for this H×W.
        # ---------------------------------------------------------------------

        final_cv = select_cv_best(
            window_df
        )

        final_cv[
            "selection_rule"
        ] = (
            "training CV primary across equally reoptimized "
            "r={1,2,3}; robustness/resources tie-break; "
            "2025 diagnostic only"
        )

        final_cv[
            "2025_used_for_selection"
        ] = False

        # ---------------------------------------------------------------------
        # FINAL hardware-aware representative across ALL equal-budget r trials.
        # ---------------------------------------------------------------------

        final_hw, final_hw_mode = (
            common
            .select_hardware_aware_row(
                window_df
            )
        )

        final_hw = final_hw.copy()

        final_hw[
            "selection_rule"
        ] = (
            "equal-budget r={1,2,3}; "
            + final_hw_mode
        )

        final_hw[
            "2025_used_for_selection"
        ] = False

        final_cv_rows.append(
            final_cv.to_dict()
        )

        final_hw_rows.append(
            final_hw.to_dict()
        )

        common.write_csv_incremental(
            out_cv_w,
            [
                final_cv.to_dict()
            ],
        )

        common.write_csv_incremental(
            out_hw_w,
            [
                final_hw.to_dict()
            ],
        )

        print(
            f"  => {topology} W={W:02d} FINAL CV: "
            f"r={int(final_cv['r'])}, "
            f"{final_cv['trotter_search_id']} / "
            f"{final_cv['readout']}, "
            f"CV={final_cv['cv_rmse']:.6f}, "
            f"SD={final_cv['cv_rmse_std']:.6f}"
        )

        print(
            f"  => {topology} W={W:02d} FINAL HW: "
            f"r={int(final_hw['r'])}, "
            f"{final_hw['trotter_search_id']} / "
            f"{final_hw['readout']}, "
            f"CV={final_hw['cv_rmse']:.6f}, "
            f"maxR="
            f"{final_hw['max_shot_to_train_std_ratio_1024']:.2f}"
        )

    # =========================================================================
    # FINAL GLOBAL SUMMARY
    # =========================================================================

    cv_final_df = pd.DataFrame(
        final_cv_rows
    )

    hw_final_df = pd.DataFrame(
        final_hw_rows
    )

    if len(cv_final_df) == 0:
        raise RuntimeError(
            "No final CV rows produced."
        )

    if len(hw_final_df) == 0:
        raise RuntimeError(
            "No final hardware-aware rows produced."
        )

    global_cv = select_cv_best(
        cv_final_df
    )

    global_hw, global_hw_mode = (
        common.select_hardware_aware_row(
            hw_final_df
        )
    )

    global_hw = global_hw.copy()

    cv_r_counts = {
        str(k):
            int(v)

        for k, v
        in Counter(
            cv_final_df["r"]
            .astype(int)
        ).items()
    }

    hw_r_counts = {
        str(k):
            int(v)

        for k, v
        in Counter(
            hw_final_df["r"]
            .astype(int)
        ).items()
    }

    cv_topology_counts = {
        str(k):
            int(v)

        for k, v
        in Counter(
            cv_final_df["topology"]
            .astype(str)
        ).items()
    }

    hw_topology_counts = {
        str(k):
            int(v)

        for k, v
        in Counter(
            hw_final_df["topology"]
            .astype(str)
        ).items()
    }

    n_below_classical_cv = int(
        (
            cv_final_df["cv_rmse"]
            <
            BEST_CLASSICAL_CV
        ).sum()
    )

    n_hw_below_classical_cv = int(
        (
            hw_final_df["cv_rmse"]
            <
            BEST_CLASSICAL_CV
        ).sum()
    )

    # Re-open written all-trial result only for an exact row-count audit.
    written_trials = pd.read_csv(
        out_trials
    )

    written_cv_wr = pd.read_csv(
        out_cv_wr
    )

    written_hw_wr = pd.read_csv(
        out_hw_wr
    )

    written_cv_w = pd.read_csv(
        out_cv_w
    )

    written_hw_w = pd.read_csv(
        out_hw_w
    )

    expected_wr_rows = (
        len(parent_df)
        *
        len(R_VALUES)
    )

    if len(written_trials) != expected_rows_this_run:
        raise RuntimeError(
            "All-trial row-count audit failed: "
            f"got {len(written_trials)}, "
            f"expected {expected_rows_this_run}."
        )

    if len(written_cv_wr) != expected_wr_rows:
        raise RuntimeError(
            "CV window-r row-count audit failed: "
            f"got {len(written_cv_wr)}, "
            f"expected {expected_wr_rows}."
        )

    if len(written_hw_wr) != expected_wr_rows:
        raise RuntimeError(
            "HW window-r row-count audit failed: "
            f"got {len(written_hw_wr)}, "
            f"expected {expected_wr_rows}."
        )

    if len(written_cv_w) != len(parent_df):
        raise RuntimeError(
            "Final CV window row-count audit failed."
        )

    if len(written_hw_w) != len(parent_df):
        raise RuntimeError(
            "Final HW window row-count audit failed."
        )

    manifest = {
        "step":
            "Week_12.3G_IQM_RWP_fair_Trotter_reoptimization",

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "purpose":
            (
                "Fair local Trotter r=1,2,3 "
                "reoptimization for IQM H5/H6 RWP."
            ),

        "source_step_f":
            str(src),

        "shared_parent_policy":
            (
                "one 12.3F training-CV winner per "
                "topology/window; same parent used "
                "for all r"
            ),

        "r_values":
            R_VALUES,

        "seed":
            LOCAL_SEED,

        "n_random_local_per_r":
            N_RANDOM_LOCAL,

        "dynamics_trials_per_r":
            EXPECTED_DYNAMICS_PER_R,

        "readouts":
            list(common.READOUTS),

        "selection_period":
            "2022-2024",

        "validation_2025_role":
            "diagnostic_only",

        "test_2026_used":
            False,

        "qpu_jobs":
            0,

        "weighted_score_used":
            False,

        "best_classical_cv_reference":
            BEST_CLASSICAL_CV,

        "counts": {
            "topology_window_branches":
                int(len(parent_df)),

            "all_trial_rows":
                int(len(written_trials)),

            "best_cv_topology_window_r_rows":
                int(len(written_cv_wr)),

            "hardware_aware_topology_window_r_rows":
                int(len(written_hw_wr)),

            "best_cv_topology_window_rows":
                int(len(written_cv_w)),

            "hardware_aware_topology_window_rows":
                int(len(written_hw_w)),
        },

        "r_counts_among_final_window_cv_winners":
            cv_r_counts,

        "r_counts_among_final_window_hw_representatives":
            hw_r_counts,

        "topology_counts_among_final_window_cv_winners":
            cv_topology_counts,

        "topology_counts_among_final_window_hw_representatives":
            hw_topology_counts,

        "final_cv_rows_below_classical_reference":
            n_below_classical_cv,

        "final_hw_rows_below_classical_reference":
            n_hw_below_classical_cv,

        "global_cv_winner":
            compact_row(
                global_cv
            ),

        "global_hardware_aware_representative": {
            **compact_row(
                global_hw
            ),
            "selection_rule":
                global_hw_mode,
        },
    }

    with open(
        out_manifest,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 128)
    print(
        "WEEK 12.3G — FAIR TROTTER-r "
        "RE-OPTIMIZATION COMPLETE"
    )
    print("=" * 128)

    print(
        f"Total trial rows: "
        f"{len(written_trials)}"
    )

    print(
        "r counts among final H×W CV winners: "
        f"{cv_r_counts}"
    )

    print(
        "r counts among final H×W HW reps:    "
        f"{hw_r_counts}"
    )

    print(
        "Final CV H×W rows below classical "
        f"{BEST_CLASSICAL_CV:.6f}: "
        f"{n_below_classical_cv}"
    )

    print(
        "Final HW H×W rows below classical "
        f"{BEST_CLASSICAL_CV:.6f}: "
        f"{n_hw_below_classical_cv}"
    )

    print()

    print(
        "GLOBAL 12.3G CV WINNER:"
    )

    print(
        f"  {global_cv['topology']} "
        f"W={int(global_cv['window'])}, "
        f"r={int(global_cv['r'])}, "
        f"{global_cv['trotter_search_id']} / "
        f"{global_cv['readout']}"
    )

    print(
        f"  CV={global_cv['cv_rmse']:.6f}, "
        f"CV_SD={global_cv['cv_rmse_std']:.6f}, "
        f"2025={global_cv['validation_rmse']:.6f} "
        "diagnostic-only"
    )

    print(
        f"  shotSD="
        f"{global_cv['shot_noise_forecast_sd_proxy_1024']:.3f}, "
        f"maxR="
        f"{global_cv['max_shot_to_train_std_ratio_1024']:.2f}, "
        f"CZ={int(global_cv['feature_vector_cz'])}"
    )

    print()

    print(
        "GLOBAL 12.3G HARDWARE-AWARE REPRESENTATIVE:"
    )

    print(
        f"  {global_hw['topology']} "
        f"W={int(global_hw['window'])}, "
        f"r={int(global_hw['r'])}, "
        f"{global_hw['trotter_search_id']} / "
        f"{global_hw['readout']}"
    )

    print(
        f"  CV={global_hw['cv_rmse']:.6f}, "
        f"2025={global_hw['validation_rmse']:.6f} "
        "diagnostic-only"
    )

    print(
        f"  shotSD="
        f"{global_hw['shot_noise_forecast_sd_proxy_1024']:.3f}, "
        f"maxR="
        f"{global_hw['max_shot_to_train_std_ratio_1024']:.2f}, "
        f"CZ={int(global_hw['feature_vector_cz'])}"
    )

    print()

    print("Saved:")
    print(f"  {out_trials}")
    print(f"  {out_cv_wr}")
    print(f"  {out_hw_wr}")
    print(f"  {out_cv_w}")
    print(f"  {out_hw_w}")
    print(f"  {out_manifest}")

    print()

    print(
        "No QPU jobs submitted."
    )

    print(
        "2026 untouched."
    )

    print(
        "STOP HERE. Interpret 12.3G before "
        "moving to the next IQM hardware/noise stage."
    )


if __name__ == "__main__":
    main()
