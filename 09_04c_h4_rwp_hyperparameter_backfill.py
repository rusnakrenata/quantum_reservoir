"""
IBM H4 GAP REPAIR — Step C
H4-only RWP hyperparameter re-optimization, matching original Week 9.04C.

PURPOSE
-------
Backfill the missing H4 branch of the systematic IBM RWP search without
rerunning H0-H3.

METHOD
------
For every H4 replay window W=1..28:
- start from the Step-A hardware-aware transferred parent,
- generate the exact deterministic Stage-C candidate family from
  `09_04_rwp_common.py`,
- re-evaluate all five readout families,
- reselect Ridge lambda on 2022-2024 only,
- keep both:
    (a) training-CV winner,
    (b) hardware-aware representative.

Trotter r stays frozen in this stage and is treated later in the separate
09.04C2-style fair Trotter reoptimization if Step C justifies continuation.

IMPORTANT
---------
- 2025 is diagnostic only.
- 2026 is never loaded.
- No weighted score.
- No H0-H3 rerun.
- This script mirrors the original 09_04c_rwp_hyperparameter_reopt.py logic.

INPUT
-----
results/09_04a_h4_backfill_rwp_hardware_aware_per_window.csv

DEPENDENCIES
------------
Keep beside this script:
    09_04_rwp_common.py
    07_02b_memory_pair_isolation.py

OUTPUTS
-------
results/09_04c_h4_backfill_rwp_all_trials.csv
results/09_04c_h4_backfill_rwp_best_cv_per_window.csv
results/09_04c_h4_backfill_rwp_hardware_aware_per_window.csv
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

COMMON_PATH = HERE / "09_04_rwp_common.py"


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Keep the original IBM 09_04_rwp_common.py "
            f"beside this script."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    spec.loader.exec_module(module)
    return module


common = load_module(
    "ibm_09_04_rwp_common_for_h4_backfill_c",
    COMMON_PATH,
)


def row_to_parent(row: pd.Series) -> dict:
    c = common.candidate_from_row(row)
    c["candidate_id"] = f"H4_W{int(row['window']):02d}_Aparent_backfill"
    c["window"] = int(row["window"])
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-branches", type=int, default=None)
    args = parser.parse_args()

    src = RESULTS / "09_04a_h4_backfill_rwp_hardware_aware_per_window.csv"

    out_trials = RESULTS / "09_04c_h4_backfill_rwp_all_trials.csv"
    out_cv = RESULTS / "09_04c_h4_backfill_rwp_best_cv_per_window.csv"
    out_hw = RESULTS / "09_04c_h4_backfill_rwp_hardware_aware_per_window.csv"

    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing. Run 09_04a_h4_rwp_transfer_backfill.py first."
        )

    if args.overwrite:
        for p in [out_trials, out_cv, out_hw]:
            if p.exists():
                p.unlink()
    else:
        existing = [p for p in [out_trials, out_cv, out_hw] if p.exists()]
        if existing:
            raise FileExistsError(
                "Backfill Step-C output already exists. "
                "Use --overwrite only for an intentional full rerun:\n  "
                + "\n  ".join(str(p) for p in existing)
            )

    base_df = (
        pd.read_csv(src)
        .sort_values(["topology", "window"])
        .reset_index(drop=True)
    )

    if set(base_df["topology"].astype(str).unique()) != {"H4"}:
        raise RuntimeError(
            "Expected the Step-A hardware-aware input to contain only H4."
        )

    if args.max_branches is not None:
        base_df = base_df.head(args.max_branches).copy()

    work_tv, cols, y_all = common.load_train_validation()

    print("=" * 124)
    print("IBM H4 GAP REPAIR — STEP C")
    print("H4-ONLY RWP HYPERPARAMETER RE-OPTIMIZATION")
    print("=" * 124)
    print("Exact original 09.04C deterministic search recipe.")
    print("Every H4×W branch receives the same search budget.")
    print("Search dimensions: alpha, dt, hx, hy, global J, per-edge J, combined local parents.")
    print("Every dynamics trial rechecks all five readouts and Ridge lambda on 2022-2024 only.")
    print("Trotter r is frozen here; fair r reoptimization belongs to 09.04C2.")
    print("2025 = diagnostic only; 2026 = untouched / not loaded.")
    print()
    print(f"H4×W branches to process: {len(base_df)}")
    print(f"Readouts per dynamics trial: {len(common.READOUTS)}")
    print()

    for branch_i, row in base_df.iterrows():
        topology = str(row["topology"])
        W = int(row["window"])
        parent = row_to_parent(row)
        candidates = common.stage_c_candidates(parent)

        print("-" * 124)
        print(
            f"[{branch_i + 1}/{len(base_df)}] "
            f"{topology} W={W:02d}: "
            f"{len(candidates)} dynamics trials × "
            f"{len(common.READOUTS)} readouts"
        )
        print(
            f"Parent: readout={row['readout']}, "
            f"CV={row['cv_rmse']:.6f}, "
            f"r={parent['r']}, "
            f"alpha={parent['alpha']}, "
            f"dt={parent['dt']}, "
            f"hx={parent['hx']:+.6f}, "
            f"hy={parent['hy']:+.6f}"
        )

        branch_rows = []

        for trial_i, candidate in enumerate(candidates):
            readout_df, _ = common.evaluate_candidate_window(
                candidate=candidate,
                window=W,
                work_tv=work_tv,
                cols=cols,
                y_all=y_all,
            )

            readout_df["window"] = W
            readout_df["trial_search_id"] = candidate["search_id"]
            readout_df["trial_index"] = trial_i
            readout_df["backfill_step"] = "H4_RWP_09.04C"
            readout_df["2025_used_for_selection"] = False
            readout_df["2026_used"] = False

            branch_rows.extend(
                readout_df.to_dict("records")
            )

        common.write_csv_incremental(
            out_trials,
            branch_rows,
        )

        trial_df = pd.DataFrame(branch_rows)

        cv_best = (
            trial_df.sort_values(
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

        hw_best, mode = common.select_hardware_aware_row(
            trial_df
        )
        hw_best = hw_best.copy()
        hw_best["selection_rule"] = mode
        hw_best["2025_used_for_selection"] = False
        hw_best["2026_used"] = False

        common.write_csv_incremental(
            out_cv,
            [cv_best.to_dict()],
        )
        common.write_csv_incremental(
            out_hw,
            [hw_best.to_dict()],
        )

        print(
            f"CV-best: "
            f"{cv_best['trial_search_id']} / "
            f"{cv_best['readout']} | "
            f"CV={cv_best['cv_rmse']:.6f} | "
            f"Val={cv_best['validation_rmse']:.6f} | "
            f"shotSD={cv_best['shot_noise_forecast_sd_proxy_1024']:.3f} | "
            f"maxR={cv_best['max_shot_to_train_std_ratio_1024']:.2f}"
        )
        print(
            f"HW-aware: "
            f"{hw_best['trial_search_id']} / "
            f"{hw_best['readout']} | "
            f"CV={hw_best['cv_rmse']:.6f} | "
            f"Val={hw_best['validation_rmse']:.6f} | "
            f"shotSD={hw_best['shot_noise_forecast_sd_proxy_1024']:.3f} | "
            f"maxR={hw_best['max_shot_to_train_std_ratio_1024']:.2f}"
        )

    print()
    print("=" * 124)
    print("IBM H4 GAP REPAIR — STEP C COMPLETE")
    print("=" * 124)
    print(f"Saved all trials:       {out_trials}")
    print(f"Saved CV winners:       {out_cv}")
    print(f"Saved HW-aware winners: {out_hw}")
    print()
    print(
        "STOP HERE. Send the full console output and all three Step-C CSV files. "
        "Do not run the fair Trotter backfill until these results are interpreted."
    )


if __name__ == "__main__":
    main()
