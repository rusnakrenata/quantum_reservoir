from __future__ import annotations

import argparse
from importlib import import_module

import pandas as pd

common = import_module("09_04_rwp_common")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="1-28")
    parser.add_argument("--include-h4", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    windows = common.parse_windows(args.windows)
    topologies = common.parse_topologies(args.include_h4)

    out_all = common.RESULTS / "09_04a_rwp_all_readouts.csv"
    out_cv = common.RESULTS / "09_04a_rwp_best_cv_per_topology_window.csv"
    out_hw = common.RESULTS / "09_04a_rwp_hardware_aware_per_topology_window.csv"

    if args.overwrite:
        for p in [out_all, out_cv, out_hw]:
            if p.exists():
                p.unlink()

    work_tv, cols, y_all = common.load_train_validation()
    baselines = common.load_topology_baselines(topologies)

    all_rows = []
    cv_rows = []
    hw_rows = []

    print("=" * 120)
    print("WEEK 9.04A — RWP TRANSFER MATRIX, REVISED AFTER REAL-QPU READOUT DIAGNOSTICS")
    print("=" * 120)
    print(f"Topologies: {topologies}")
    print(f"Windows:    {windows[0]} ... {windows[-1]} ({len(windows)} values)")
    print(f"H×W branches: {len(topologies) * len(windows)}")
    print(f"Readouts: {common.READOUTS}")
    print(f"Finite-shot conditioning proxy: {common.SHOTS_PROXY} shots")
    print("2022-2024 = training/CV/conditioning; 2025 = diagnostic validation; 2026 = frozen.")
    print()

    for topology in topologies:
        parent = baselines[topology]

        print("-" * 120)
        print(
            f"{topology}: source={parent['source_candidate_id']}, "
            f"r={parent['r']}, alpha={parent['alpha']}, dt={parent['dt']}, "
            f"hx={parent['hx']:+.6f}, hy={parent['hy']:+.6f}"
        )

        angles = common.make_angles(work_tv, cols, parent["alpha"])
        A_list = common.build_channels(parent, angles)

        for W in windows:
            endpoints, master = common.rwp_master_feature_bank_from_channels(
                A_list, W
            )

            branch_rows = []

            for readout in common.READOUTS:
                metrics = common.evaluate_master_bank(
                    endpoints=endpoints,
                    master_df=master,
                    y_all=y_all,
                    window=W,
                    readout=readout,
                )

                row = {
                    **common.candidate_to_flat(parent),
                    **{
                        k: v
                        for k, v in metrics.items()
                        if k not in {"folds", "cv_summary"}
                    },
                    **common.resource_metrics(parent, W, readout),
                }
                row["candidate_id"] = f"{topology}_W{W:02d}_{readout}"
                row["2025_used_for_selection"] = False

                branch_rows.append(row)
                all_rows.append(row)

            current = pd.DataFrame(branch_rows)

            cv_best = (
                current.sort_values(
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
                "training CV primary; robustness/resources only tie-break; 2025 diagnostic only"
            )
            cv_rows.append(cv_best)

            hw_best, mode = common.select_hardware_aware_row(current)
            hw_best = hw_best.copy()
            hw_best["selection_rule"] = mode
            hw_best["2025_used_for_selection"] = False
            hw_rows.append(hw_best)

            print(
                f"W={W:02d} | CV-best={cv_best['readout']:<28s} "
                f"CV={cv_best['cv_rmse']:.6f} "
                f"shotSD={cv_best['shot_noise_forecast_sd_proxy_1024']:.3f} "
                f"maxR={cv_best['max_shot_to_train_std_ratio_1024']:.2f} || "
                f"HW-aware={hw_best['readout']:<28s} "
                f"CV={hw_best['cv_rmse']:.6f} "
                f"shotSD={hw_best['shot_noise_forecast_sd_proxy_1024']:.3f} "
                f"maxR={hw_best['max_shot_to_train_std_ratio_1024']:.2f}"
            )

    all_df = pd.DataFrame(all_rows)
    cv_df = pd.DataFrame(cv_rows)
    hw_df = pd.DataFrame(hw_rows)

    all_df.to_csv(out_all, index=False)
    cv_df.to_csv(out_cv, index=False)
    hw_df.to_csv(out_hw, index=False)

    print()
    print("=" * 120)
    print("9.04A COMPLETE")
    print("=" * 120)
    print(f"All readout rows: {len(all_df)}")
    print(f"CV winners:       {len(cv_df)}")
    print(f"HW-aware reps:    {len(hw_df)}")
    print(f"Saved: {out_all}")
    print(f"Saved: {out_cv}")
    print(f"Saved: {out_hw}")


if __name__ == "__main__":
    main()
