from __future__ import annotations

import argparse
from importlib import import_module

import pandas as pd

common = import_module("09_04_rwp_common")


def row_to_parent(row: pd.Series) -> dict:
    c = common.candidate_from_row(row)
    c["candidate_id"] = f"{row['topology']}_W{int(row['window']):02d}_Aparent"
    c["window"] = int(row["window"])
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-branches", type=int, default=None)
    args = parser.parse_args()

    src = common.RESULTS / "09_04a_rwp_hardware_aware_per_topology_window.csv"
    out_trials = common.RESULTS / "09_04c_rwp_all_trials.csv"
    out_cv = common.RESULTS / "09_04c_rwp_best_cv_per_topology_window.csv"
    out_hw = common.RESULTS / "09_04c_rwp_hardware_aware_per_topology_window.csv"

    if not src.exists():
        raise FileNotFoundError(f"{src} missing. Run 09_04a first.")

    if args.overwrite:
        for p in [out_trials, out_cv, out_hw]:
            if p.exists():
                p.unlink()

    base_df = pd.read_csv(src).sort_values(["topology", "window"]).reset_index(drop=True)

    if args.max_branches is not None:
        base_df = base_df.head(args.max_branches).copy()

    work_tv, cols, y_all = common.load_train_validation()

    print("=" * 120)
    print("WEEK 9.04C — RWP HYPERPARAMETER RE-OPTIMIZATION, HARDWARE-AWARE REVISION")
    print("=" * 120)
    print("Every H×W branch receives the same deterministic search recipe.")
    print("Search: alpha, dt, hx, hy, global J, per-edge J, combined local parents.")
    print("Every dynamics trial rechecks all five readout families and lambda on 2022-2024 only.")
    print("Trotter r is held fixed here and reoptimized fairly in 9.04C2.")
    print()

    for branch_i, row in base_df.iterrows():
        topology = str(row["topology"])
        W = int(row["window"])
        parent = row_to_parent(row)
        candidates = common.stage_c_candidates(parent)

        print("-" * 120)
        print(f"[{branch_i+1}/{len(base_df)}] {topology} W={W:02d}: {len(candidates)} dynamics trials × {len(common.READOUTS)} readouts")

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
            readout_df["2025_used_for_selection"] = False
            branch_rows.extend(readout_df.to_dict("records"))

        common.write_csv_incremental(out_trials, branch_rows)
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
        cv_best["selection_rule"] = "training CV primary; robustness/resources tie-break; 2025 diagnostic only"

        hw_best, mode = common.select_hardware_aware_row(trial_df)
        hw_best = hw_best.copy()
        hw_best["selection_rule"] = mode
        hw_best["2025_used_for_selection"] = False

        common.write_csv_incremental(out_cv, [cv_best.to_dict()])
        common.write_csv_incremental(out_hw, [hw_best.to_dict()])

        print(
            f"CV-best: {cv_best['trial_search_id']} / {cv_best['readout']} "
            f"CV={cv_best['cv_rmse']:.6f} Val={cv_best['validation_rmse']:.6f} "
            f"shotSD={cv_best['shot_noise_forecast_sd_proxy_1024']:.3f} "
            f"maxR={cv_best['max_shot_to_train_std_ratio_1024']:.2f}"
        )
        print(
            f"HW-aware: {hw_best['trial_search_id']} / {hw_best['readout']} "
            f"CV={hw_best['cv_rmse']:.6f} Val={hw_best['validation_rmse']:.6f} "
            f"shotSD={hw_best['shot_noise_forecast_sd_proxy_1024']:.3f} "
            f"maxR={hw_best['max_shot_to_train_std_ratio_1024']:.2f}"
        )

    print()
    print("=" * 120)
    print("9.04C COMPLETE")
    print("=" * 120)
    print(f"Saved all trials: {out_trials}")
    print(f"Saved CV winners: {out_cv}")
    print(f"Saved HW-aware winners: {out_hw}")


if __name__ == "__main__":
    main()
