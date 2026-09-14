from __future__ import annotations

import argparse
from importlib import import_module

import pandas as pd

common = import_module("09_04_rwp_common")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-values", default="1,2,3,4")
    parser.add_argument("--local-budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-branches", type=int, default=None, help="Debug only; default runs all 112 H×W parents.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow fewer than the full H0-H3 × W1-28 parent grid.")
    args = parser.parse_args()

    r_values = [int(x) for x in args.r_values.split(",") if x.strip()]

    src = common.RESULTS / "09_04c_rwp_hardware_aware_per_topology_window.csv"
    out_all = common.RESULTS / "09_04c2_rwp_trotter_all_trials.csv"
    out_per_r_cv = common.RESULTS / "09_04c2_rwp_best_cv_per_r.csv"
    out_per_r_hw = common.RESULTS / "09_04c2_rwp_hardware_aware_per_r.csv"
    out_final_cv = common.RESULTS / "09_04c2_rwp_final_best_cv_per_topology_window.csv"
    out_final_hw = common.RESULTS / "09_04c2_rwp_final_hardware_aware_per_topology_window.csv"

    if not src.exists():
        raise FileNotFoundError(f"{src} missing. Run 09_04c first.")

    if args.overwrite:
        for p in [out_all, out_per_r_cv, out_per_r_hw, out_final_cv, out_final_hw]:
            if p.exists():
                p.unlink()

    parents = pd.read_csv(src).sort_values(["topology", "window"]).reset_index(drop=True)

    # Full-grid audit: by default C2 MUST use exactly one Stage-C hardware-aware
    # parent for every topology H0-H3 and every rewind window W=1..28.
    expected_topologies = ["H0", "H1", "H2", "H3"]
    expected_windows = list(range(1, 29))
    expected_pairs = {(h, w) for h in expected_topologies for w in expected_windows}
    actual_pairs = {(str(r.topology), int(r.window)) for _, r in parents.iterrows()}

    duplicate_count = int(parents.duplicated(["topology", "window"]).sum())
    missing_pairs = sorted(expected_pairs - actual_pairs)
    extra_pairs = sorted(actual_pairs - expected_pairs)

    if not args.allow_partial and args.max_branches is None:
        if len(parents) != 112 or duplicate_count or missing_pairs or extra_pairs:
            raise RuntimeError(
                "Stage-C2 parent audit failed. Expected exactly one hardware-aware "
                "Stage-C parent for each H0-H3 × W1-28 pair (112 total).\n"
                f"rows={len(parents)}, duplicates={duplicate_count}, "
                f"missing={missing_pairs[:10]}, extra={extra_pairs[:10]}"
            )

    if args.max_branches is not None:
        parents = parents.head(args.max_branches).copy()

    work_tv, cols, y_all = common.load_train_validation()

    print("=" * 120)
    print("WEEK 9.04C2 — FAIR RWP TROTTER-DEPTH RE-OPTIMIZATION")
    print("=" * 120)
    print(f"Stage-C parent rows: {len(parents)}")
    print("Parents by topology:")
    print(parents.groupby("topology")["window"].nunique().to_string())
    print(f"r values: {r_values}")
    print("Every r gets the same local dynamics-search budget and all five readout families.")
    print("Both pure-CV and hardware-aware representatives are saved. 2025 is never used for selection.")
    print(f"Per H×W parent: {len(r_values)} r values × (5 deterministic + {args.local_budget} random dynamics trials) × 5 readouts")
    print(f"Expected readout evaluations for full run: {len(parents) * len(r_values) * (5 + args.local_budget) * 5}")
    print()

    # Safe resume: a branch is considered complete only after a final HW-aware row
    # has been written. --overwrite clears all previous C2 outputs.
    completed = set()
    if out_final_hw.exists() and not args.overwrite:
        done_df = pd.read_csv(out_final_hw)
        if {"topology", "window"}.issubset(done_df.columns):
            completed = {(str(r.topology), int(r.window)) for _, r in done_df.iterrows()}
            if completed:
                print(f"Resume mode: {len(completed)} completed H×W branches will be skipped.")
                print()

    for branch_i, row in parents.iterrows():
        topology = str(row["topology"])
        W = int(row["window"])

        if (topology, W) in completed:
            print(f"[{branch_i+1}/{len(parents)}] {topology} W={W:02d}: already complete, skipping")
            continue

        parent = common.candidate_from_row(row)
        parent["candidate_id"] = f"{topology}_W{W:02d}_C_HW_parent"
        parent["window"] = W

        branch_all = []
        per_r_cv_rows = []
        per_r_hw_rows = []

        print("-" * 120)
        print(f"[{branch_i+1}/{len(parents)}] {topology} W={W:02d}")

        for r in r_values:
            dynamics_trials = common.trotter_local_candidates(
                parent=parent,
                r=r,
                seed=args.seed,
                n_random=args.local_budget,
            )

            r_rows = []

            for trial in dynamics_trials:
                readout_df, _ = common.evaluate_candidate_window(
                    candidate=trial,
                    window=W,
                    work_tv=work_tv,
                    cols=cols,
                    y_all=y_all,
                )

                readout_df["window"] = W
                readout_df["trotter_test_r"] = int(r)
                readout_df["trial_search_id"] = trial["search_id"]
                readout_df["2025_used_for_selection"] = False

                rows = readout_df.to_dict("records")
                r_rows.extend(rows)
                branch_all.extend(rows)

            r_df = pd.DataFrame(r_rows)

            cv_best = (
                r_df.sort_values(
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
            cv_best["selection_rule"] = "training CV primary within equal-budget r search"

            hw_best, mode = common.select_hardware_aware_row(r_df)
            hw_best = hw_best.copy()
            hw_best["selection_rule"] = mode
            hw_best["2025_used_for_selection"] = False

            per_r_cv_rows.append(cv_best.to_dict())
            per_r_hw_rows.append(hw_best.to_dict())

            print(
                f"  r={r}: CV-best={cv_best['readout']} CV={cv_best['cv_rmse']:.6f} "
                f"shotSD={cv_best['shot_noise_forecast_sd_proxy_1024']:.3f} || "
                f"HW-aware={hw_best['readout']} CV={hw_best['cv_rmse']:.6f} "
                f"shotSD={hw_best['shot_noise_forecast_sd_proxy_1024']:.3f} "
                f"maxR={hw_best['max_shot_to_train_std_ratio_1024']:.2f}"
            )

        per_r_cv_df = pd.DataFrame(per_r_cv_rows)
        per_r_hw_df = pd.DataFrame(per_r_hw_rows)

        final_cv = (
            per_r_cv_df.sort_values(
                [
                    "cv_rmse",
                    "shot_noise_forecast_sd_proxy_1024",
                    "feature_vector_cz",
                ]
            )
            .iloc[0]
            .copy()
        )
        final_cv["selection_rule"] = "best training-CV r after equal-budget reoptimization"

        final_hw, mode = common.select_hardware_aware_row(per_r_hw_df)
        final_hw = final_hw.copy()
        final_hw["selection_rule"] = mode
        final_hw["2025_used_for_selection"] = False

        common.write_csv_incremental(out_all, branch_all)
        common.write_csv_incremental(out_per_r_cv, per_r_cv_rows)
        common.write_csv_incremental(out_per_r_hw, per_r_hw_rows)
        common.write_csv_incremental(out_final_cv, [final_cv.to_dict()])
        common.write_csv_incremental(out_final_hw, [final_hw.to_dict()])

        print(
            f"  FINAL HW-aware: r={int(final_hw['r'])}, {final_hw['readout']}, "
            f"CV={final_hw['cv_rmse']:.6f}, Val={final_hw['validation_rmse']:.6f}, "
            f"shotSD={final_hw['shot_noise_forecast_sd_proxy_1024']:.3f}, "
            f"maxR={final_hw['max_shot_to_train_std_ratio_1024']:.2f}"
        )

    print()
    print("=" * 120)
    print("9.04C2 COMPLETE")
    print("=" * 120)
    print(f"Saved: {out_all}")
    print(f"Saved: {out_per_r_cv}")
    print(f"Saved: {out_per_r_hw}")
    print(f"Saved: {out_final_cv}")
    print(f"Saved: {out_final_hw}")


if __name__ == "__main__":
    main()
