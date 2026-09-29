"""
IBM H4 GAP REPAIR — Step A
H4-only RWP transfer matrix, matching Week 9.04A exactly.

PURPOSE
-------
The original systematic RWP matrix was frozen to H0-H3, while H4 remained
auxiliary. This script backfills ONLY H4 so that we can later test whether H4
would have altered the Rule-1–Rule-4 candidate pool.

IMPORTANT
---------
- Reuses the original IBM `09_04_rwp_common.py`.
- Does NOT rerun H0-H3.
- Does NOT use IBM QPU calibration/noise.
- 2022-2024 = training/CV/conditioning.
- 2025 = diagnostic validation only.
- 2026 is never loaded.
- Dynamics are transferred from the best H4 CONT candidate in
  `results/09_03c_forecast_best_135.csv`, exactly as 09.04A did for H0-H3.
- All five 09.04A readout families are evaluated.
- W = 1..28 by default.
- No H4 dynamics reoptimization is performed in this step.

DEPENDENCIES
------------
Keep beside this script:
    09_04_rwp_common.py
    07_02b_memory_pair_isolation.py

Required result:
    results/09_03c_forecast_best_135.csv

OUTPUTS
-------
results/09_04a_h4_backfill_rwp_all_readouts.csv
results/09_04a_h4_backfill_rwp_best_cv_per_window.csv
results/09_04a_h4_backfill_rwp_hardware_aware_per_window.csv
results/09_04a_h4_backfill_manifest.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

COMMON_PATH = HERE / "09_04_rwp_common.py"


def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Keep the original IBM 09_04_rwp_common.py "
            f"beside this script."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    spec.loader.exec_module(module)
    return module


common = import_module_from_path(
    "ibm_09_04_rwp_common_for_h4_backfill",
    COMMON_PATH,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="1-28")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    windows = common.parse_windows(args.windows)

    out_all = RESULTS / "09_04a_h4_backfill_rwp_all_readouts.csv"
    out_cv = RESULTS / "09_04a_h4_backfill_rwp_best_cv_per_window.csv"
    out_hw = RESULTS / "09_04a_h4_backfill_rwp_hardware_aware_per_window.csv"
    out_manifest = RESULTS / "09_04a_h4_backfill_manifest.json"

    if args.overwrite:
        for p in [out_all, out_cv, out_hw, out_manifest]:
            if p.exists():
                p.unlink()

    for p in [out_all, out_cv, out_hw]:
        if p.exists() and not args.overwrite:
            raise FileExistsError(
                f"{p} already exists. Use --overwrite only if you intentionally "
                f"want to rerun Step A from scratch."
            )

    work_tv, cols, y_all = common.load_train_validation()

    # IMPORTANT: only H4 is backfilled.
    baselines = common.load_topology_baselines(["H4"])
    parent = baselines["H4"]

    print("=" * 124)
    print("IBM H4 GAP REPAIR — STEP A")
    print("H4-ONLY RWP TRANSFER MATRIX — DIRECT 09.04A BACKFILL")
    print("=" * 124)
    print("Purpose: fill the historical H4 RWP gap without rerunning H0-H3.")
    print(f"Windows: {windows[0]} ... {windows[-1]} ({len(windows)} values)")
    print(f"Readouts: {common.READOUTS}")
    print(f"Finite-shot conditioning proxy: {common.SHOTS_PROXY} shots")
    print("2022-2024 = training/CV/conditioning")
    print("2025      = diagnostic only")
    print("2026      = untouched / not loaded")
    print()
    print("Frozen H4 CONT parent transferred into RWP:")
    print(
        f"  source_candidate_id = {parent['source_candidate_id']}\n"
        f"  source_config_id    = {parent['source_config_id']}\n"
        f"  source_readout      = {parent['source_readout']}\n"
        f"  source_CV           = {parent['source_cv_rmse']:.6f}\n"
        f"  source_2025_RMSE    = {parent['source_validation_rmse']:.6f} "
        f"(diagnostic only)\n"
        f"  r                   = {parent['r']}\n"
        f"  alpha               = {parent['alpha']}\n"
        f"  dt                  = {parent['dt']}\n"
        f"  hx                  = {parent['hx']:+.12f}\n"
        f"  hy                  = {parent['hy']:+.12f}"
    )
    print()

    angles = common.make_angles(
        work_tv,
        cols,
        parent["alpha"],
    )
    A_list = common.build_channels(
        parent,
        angles,
    )

    all_rows = []
    cv_rows = []
    hw_rows = []

    for W in windows:
        endpoints, master = common.rwp_master_feature_bank_from_channels(
            A_list,
            W,
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
                **common.resource_metrics(
                    parent,
                    W,
                    readout,
                ),
            }

            row["candidate_id"] = f"H4_W{W:02d}_{readout}_backfill"
            row["backfill_step"] = "H4_RWP_09.04A"
            row["historical_scope_note"] = (
                "Retrospective H4 completeness repair before frozen 2026 test"
            )
            row["2025_used_for_selection"] = False
            row["2026_used"] = False

            branch_rows.append(row)
            all_rows.append(row)

        current = pd.DataFrame(branch_rows)

        # EXACT same CV-first selection hierarchy as IBM 09.04A.
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
            "training CV primary; robustness/resources only tie-break; "
            "2025 diagnostic only"
        )
        cv_rows.append(cv_best)

        # EXACT same hardware-oriented proxy selector as IBM 09.04A.
        hw_best, mode = common.select_hardware_aware_row(current)
        hw_best = hw_best.copy()
        hw_best["selection_rule"] = mode
        hw_best["2025_used_for_selection"] = False
        hw_best["2026_used"] = False
        hw_rows.append(hw_best)

        print(
            f"W={W:02d} | "
            f"CV-best={cv_best['readout']:<28s} "
            f"CV={cv_best['cv_rmse']:.6f} "
            f"Val={cv_best['validation_rmse']:.6f} "
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

    best_cv = (
        cv_df.sort_values(
            [
                "cv_rmse",
                "shot_noise_forecast_sd_proxy_1024",
                "max_abs_raw_prediction_sensitivity",
                "n_settings",
                "feature_vector_cz",
            ]
        )
        .iloc[0]
        .to_dict()
    )

    best_hw = (
        hw_df.sort_values(
            [
                "cv_rmse",
                "shot_noise_forecast_sd_proxy_1024",
                "max_abs_raw_prediction_sensitivity",
                "n_settings",
                "feature_vector_cz",
            ]
        )
        .iloc[0]
        .to_dict()
    )

    manifest = {
        "step": "IBM_H4_gap_repair_A",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Backfill H4-only RWP transfer matrix omitted from the original "
            "systematic H0-H3 09.04A experiment."
        ),
        "methodology_source": "09_04a_rwp_transfer_matrix.py",
        "shared_logic_source": "09_04_rwp_common.py",
        "topologies": ["H4"],
        "windows": windows,
        "readouts": list(common.READOUTS),
        "shots_proxy": int(common.SHOTS_PROXY),
        "selection": {
            "training_period": "2022-2024",
            "cv": "5-fold chronological TimeSeriesSplit",
            "validation_2025": "diagnostic only",
            "test_2026_used": False,
            "weighted_score": False,
        },
        "h4_cont_parent": {
            "source_config_id": parent["source_config_id"],
            "source_candidate_id": parent["source_candidate_id"],
            "source_readout": parent["source_readout"],
            "source_lambda": parent["source_lambda"],
            "source_cv_rmse": parent["source_cv_rmse"],
            "source_validation_rmse": parent["source_validation_rmse"],
            "alpha": parent["alpha"],
            "dt": parent["dt"],
            "r": parent["r"],
            "hx": parent["hx"],
            "hy": parent["hy"],
            "J": {
                f"J{edge[0]}{edge[1]}": value
                for edge, value in parent["J"].items()
            },
        },
        "counts": {
            "windows": len(windows),
            "readouts_per_window": len(common.READOUTS),
            "all_rows": len(all_df),
            "cv_winner_rows": len(cv_df),
            "hardware_aware_rows": len(hw_df),
        },
        "best_transfer_cv_row": {
            k: (
                None
                if pd.isna(v)
                else (
                    bool(v) if isinstance(v, (bool,))
                    else int(v) if isinstance(v, (int,))
                    else float(v) if isinstance(v, (float,))
                    else str(v)
                )
            )
            for k, v in best_cv.items()
            if k in {
                "window",
                "readout",
                "selected_lambda",
                "cv_rmse",
                "cv_rmse_std",
                "validation_rmse",
                "shot_noise_forecast_sd_proxy_1024",
                "max_shot_to_train_std_ratio_1024",
                "feature_vector_cz",
                "n_settings",
                "n_features",
            }
        },
        "best_hardware_aware_transfer_row": {
            k: (
                None
                if pd.isna(v)
                else (
                    bool(v) if isinstance(v, (bool,))
                    else int(v) if isinstance(v, (int,))
                    else float(v) if isinstance(v, (float,))
                    else str(v)
                )
            )
            for k, v in best_hw.items()
            if k in {
                "window",
                "readout",
                "selected_lambda",
                "cv_rmse",
                "cv_rmse_std",
                "validation_rmse",
                "shot_noise_forecast_sd_proxy_1024",
                "max_shot_to_train_std_ratio_1024",
                "feature_vector_cz",
                "n_settings",
                "n_features",
                "selection_rule",
            }
        },
    }

    with out_manifest.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print()
    print("=" * 124)
    print("IBM H4 GAP REPAIR — STEP A COMPLETE")
    print("=" * 124)
    print(f"All readout rows: {len(all_df)}")
    print(f"CV winners:       {len(cv_df)}")
    print(f"HW-aware reps:    {len(hw_df)}")
    print()
    print(
        "Best transferred H4 RWP branch by training CV:\n"
        f"  W={int(best_cv['window'])}, "
        f"readout={best_cv['readout']}, "
        f"CV={best_cv['cv_rmse']:.6f}, "
        f"CV_SD={best_cv['cv_rmse_std']:.6f}, "
        f"2025={best_cv['validation_rmse']:.6f} diagnostic-only"
    )
    print()
    print("Saved:")
    print(f"  {out_all}")
    print(f"  {out_cv}")
    print(f"  {out_hw}")
    print(f"  {out_manifest}")
    print()
    print(
        "STOP HERE. Send the console output and the four files above. "
        "Do not run the Pareto/reoptimization stages until Step A is interpreted."
    )


if __name__ == "__main__":
    main()
