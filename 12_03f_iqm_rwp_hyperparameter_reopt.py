"""
Week 12.3F — IQM H5/H6 RWP hyperparameter re-optimization.

Direct one-to-one adaptation of IBM `09_04c_rwp_hyperparameter_reopt.py`.

Scientific role
---------------
For every IQM-native topology/window branch H×W produced in Week 12.3D:

1. take the 12.3D hardware-aware representative as the parent;
2. apply exactly the deterministic Stage-C dynamics search already implemented
   in `12_03d_iqm_rwp_common.stage_c_candidates`;
3. for every dynamics trial, re-evaluate all five readout families;
4. reselect Ridge lambda using 2022-2024 chronological CV only;
5. save both:
       - the pure training-CV winner,
       - the hardware-aware winner.

Frozen in this stage
--------------------
- RWP window W is fixed within each branch.
- Trotter r is fixed to the branch parent's value.
- Trotter r will be re-optimized fairly only in the later IQM analogue of
  IBM 09.04C2.
- 2025 is diagnostic only.
- 2026 remains untouched.
- No QPU jobs are submitted.
- No weighted score is used.

Search dimensions
-----------------
The shared Stage-C helper searches:
- alpha,
- dt,
- hx,
- hy,
- global J scale,
- individual-edge J perturbations,
- combined local parents.

Because H5 and H6 both have seven logical edges, both receive the same
per-edge search budget.
"""

from __future__ import annotations

import argparse
from importlib import import_module

import pandas as pd


common = import_module("12_03d_iqm_rwp_common")


# =============================================================================
# 12.3D CSV J-COLUMN COMPATIBILITY FIX
# =============================================================================
#
# Week 12.3D serialized J columns using the literal Trotter edge orientation:
#   H5 examples: J30, J21, J32
#   H6 examples: J52, J40
#
# while common.EDGE_TO_COLUMN uses canonical names:
#   H5: J03, J12, J23
#   H6: J25, J04
#
# The coupling VALUES and dynamics were correct. Only CSV column naming differed.
# 12.3E did not reconstruct J, so it was unaffected.
#
# This 12.3F script is deliberately backward-compatible with the existing 12.3D
# CSVs and also forces all NEW 12.3F output to use canonical J-column names.


def _read_coupling_from_row(row_dict: dict, topology: str, edge: tuple[int, int]) -> float:
    canonical_col = common.EDGE_TO_COLUMN[topology][edge]
    direct_col = f"J{edge[0]}{edge[1]}"
    reverse_col = f"J{edge[1]}{edge[0]}"

    candidates = []
    for col in [canonical_col, direct_col, reverse_col]:
        if col not in candidates:
            candidates.append(col)

    for col in candidates:
        if col in row_dict and not pd.isna(row_dict[col]):
            return float(row_dict[col])

    raise KeyError(
        f"Missing coupling for topology={topology}, edge={edge}. "
        f"Tried columns {candidates}."
    )


def canonical_candidate_to_flat(candidate: dict) -> dict:
    """Serialize all NEW 12.3F candidates with canonical J-column names."""
    topology = str(candidate["topology"])

    row = {
        "topology": topology,
        "alpha": float(candidate["alpha"]),
        "dt": float(candidate["dt"]),
        "r": int(candidate["r"]),
        "hx": float(candidate["hx"]),
        "hy": float(candidate["hy"]),
    }

    for edge, value in candidate["J"].items():
        col = common.EDGE_TO_COLUMN[topology][edge]
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


# evaluate_candidate_window() resolves candidate_to_flat from the common module
# at runtime, so this makes 12.3F output canonical without changing any dynamics.
common.candidate_to_flat = canonical_candidate_to_flat


def row_to_parent(row: pd.Series) -> dict:
    """
    Convert one Week-12.3D H×W hardware-aware representative into the
    dynamics parent consumed by common.stage_c_candidates().

    Accept both the legacy 12.3D orientation-dependent J columns and the
    canonical column names.
    """
    row_dict = row.to_dict()
    topology = str(row_dict["topology"])

    J = {
        edge: _read_coupling_from_row(
            row_dict,
            topology,
            edge,
        )
        for edge in common.TOPOLOGY_EDGES[topology]
    }

    c = {
        "topology": topology,
        "alpha": float(row_dict.get("alpha", common.DEFAULT_ALPHA)),
        "dt": float(row_dict.get("dt", common.DEFAULT_DT)),
        "r": int(
            row_dict["r"]
            if "r" in row_dict
            else row_dict["trotter_r"]
        ),
        "hx": float(row_dict["hx"]),
        "hy": float(row_dict["hy"]),
        "J": J,
        "candidate_id": (
            f"{topology}_W{int(row_dict['window']):02d}_Dparent"
        ),
        "window": int(row_dict["window"]),
    }

    return c


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing 12.3F outputs before running.",
    )

    parser.add_argument(
        "--max-branches",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. "
            "For the full scientific run, omit this argument."
        ),
    )

    args = parser.parse_args()

    # IMPORTANT:
    # Direct IBM 09.04C logic uses the hardware-aware representatives from
    # Stage A / our 12.3D transfer matrix as Stage-C parents.
    #
    # 12.3E is a Pareto analysis layer only; it does not prune windows and is
    # not used to replace the per-H×W Stage-C parents.
    src = (
        common.RESULTS
        / "12_03d_rwp_hardware_aware_per_topology_window.csv"
    )

    out_trials = (
        common.RESULTS
        / "12_03f_rwp_all_trials.csv"
    )

    out_cv = (
        common.RESULTS
        / "12_03f_rwp_best_cv_per_topology_window.csv"
    )

    out_hw = (
        common.RESULTS
        / "12_03f_rwp_hardware_aware_per_topology_window.csv"
    )

    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing. Run Week 12.3D first."
        )

    if args.overwrite:
        for p in [
            out_trials,
            out_cv,
            out_hw,
        ]:
            if p.exists():
                p.unlink()

    base_df = (
        pd.read_csv(src)
        .sort_values(
            [
                "topology",
                "window",
            ]
        )
        .reset_index(drop=True)
    )

    # Scientific full run must contain all 56 H×W branches.
    expected_topologies = set(common.DEFAULT_TOPOLOGIES)
    observed_topologies = set(
        base_df["topology"].astype(str)
    )

    if observed_topologies != expected_topologies:
        raise RuntimeError(
            "Unexpected topology set in 12.3D parents: "
            f"observed={sorted(observed_topologies)}, "
            f"expected={sorted(expected_topologies)}."
        )

    full_branch_count = (
        len(common.DEFAULT_TOPOLOGIES)
        * len(common.DEFAULT_WINDOWS)
    )

    if len(base_df) != full_branch_count:
        raise RuntimeError(
            "Expected one hardware-aware parent for every H×W branch: "
            f"{full_branch_count}; got {len(base_df)}."
        )

    if args.max_branches is not None:
        if args.max_branches < 1:
            raise ValueError("--max-branches must be >= 1.")

        base_df = (
            base_df
            .head(args.max_branches)
            .copy()
        )

    work_tv, cols, y_all = (
        common.load_train_validation()
    )

    print("=" * 120)
    print(
        "WEEK 12.3F — IQM H5/H6 RWP HYPERPARAMETER "
        "RE-OPTIMIZATION, HARDWARE-AWARE REVISION"
    )
    print("=" * 120)

    print(
        "Direct one-to-one adaptation of IBM 09.04C."
    )
    print(
        "Every H×W branch receives the same deterministic "
        "Stage-C search recipe."
    )
    print(
        "Search: alpha, dt, hx, hy, global J, per-edge J, "
        "combined local parents."
    )
    print(
        "Every dynamics trial rechecks all five readout families "
        "and Ridge lambda on 2022-2024 only."
    )
    print(
        "Trotter r is held fixed here and will be reoptimized "
        "fairly in the later 09.04C2 analogue."
    )
    print(
        "2025 = diagnostic only; 2026 = untouched; QPU jobs = 0."
    )
    print()

    for branch_i, row in base_df.iterrows():
        topology = str(row["topology"])
        W = int(row["window"])

        parent = row_to_parent(row)

        candidates = (
            common.stage_c_candidates(
                parent
            )
        )

        print("-" * 120)

        print(
            f"[{branch_i + 1}/{len(base_df)}] "
            f"{topology} W={W:02d}: "
            f"{len(candidates)} dynamics trials × "
            f"{len(common.READOUTS)} readouts"
        )

        print(
            "Parent: "
            f"r={parent['r']}, "
            f"alpha={parent['alpha']:.6f}, "
            f"dt={parent['dt']:.6f}, "
            f"hx={parent['hx']:+.6f}, "
            f"hy={parent['hy']:+.6f}, "
            f"12.3D readout={row['readout']}, "
            f"CV={float(row['cv_rmse']):.6f}"
        )

        branch_rows = []

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

            readout_df["window"] = W

            readout_df[
                "trial_search_id"
            ] = candidate["search_id"]

            readout_df[
                "trial_index"
            ] = int(trial_i)

            readout_df[
                "2025_used_for_selection"
            ] = False

            branch_rows.extend(
                readout_df.to_dict(
                    "records"
                )
            )

        # Save the complete search surface for this H×W branch.
        common.write_csv_incremental(
            out_trials,
            branch_rows,
        )

        trial_df = pd.DataFrame(
            branch_rows
        )

        # -------------------------------------------------------------
        # Pure training-CV winner.
        # -------------------------------------------------------------
        cv_best = (
            trial_df
            .sort_values(
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

        cv_best["selection_rule"] = (
            "training CV primary; robustness/resources tie-break; "
            "2025 diagnostic only"
        )

        cv_best[
            "2025_used_for_selection"
        ] = False

        # -------------------------------------------------------------
        # Hardware-aware winner.
        # -------------------------------------------------------------
        hw_best, mode = (
            common.select_hardware_aware_row(
                trial_df
            )
        )

        hw_best = hw_best.copy()

        hw_best["selection_rule"] = mode

        hw_best[
            "2025_used_for_selection"
        ] = False

        common.write_csv_incremental(
            out_cv,
            [
                cv_best.to_dict()
            ],
        )

        common.write_csv_incremental(
            out_hw,
            [
                hw_best.to_dict()
            ],
        )

        print(
            f"CV-best: "
            f"{cv_best['trial_search_id']} / "
            f"{cv_best['readout']} "
            f"CV={cv_best['cv_rmse']:.6f} "
            f"Val={cv_best['validation_rmse']:.6f} "
            f"shotSD="
            f"{cv_best['shot_noise_forecast_sd_proxy_1024']:.3f} "
            f"maxR="
            f"{cv_best['max_shot_to_train_std_ratio_1024']:.2f}"
        )

        print(
            f"HW-aware: "
            f"{hw_best['trial_search_id']} / "
            f"{hw_best['readout']} "
            f"CV={hw_best['cv_rmse']:.6f} "
            f"Val={hw_best['validation_rmse']:.6f} "
            f"shotSD="
            f"{hw_best['shot_noise_forecast_sd_proxy_1024']:.3f} "
            f"maxR="
            f"{hw_best['max_shot_to_train_std_ratio_1024']:.2f}"
        )

    print()
    print("=" * 120)
    print("12.3F COMPLETE")
    print("=" * 120)

    print(
        f"Processed H×W branches: {len(base_df)}"
    )

    print(
        f"Saved all trials:       {out_trials}"
    )

    print(
        f"Saved CV winners:       {out_cv}"
    )

    print(
        f"Saved HW-aware winners: {out_hw}"
    )

    print()
    print(
        "No QPU jobs submitted."
    )

    print(
        "2026 untouched."
    )

    print(
        "STOP HERE after the run and interpret 12.3F before "
        "the fair Trotter-r reoptimization stage."
    )


if __name__ == "__main__":
    main()
