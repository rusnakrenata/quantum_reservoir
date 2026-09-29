"""
IBM H4 GAP REPAIR — Step C2
H4-only fair local Trotter re-optimization, matching the original 09.04C2 logic.

PURPOSE
-------
Complete the missing H4 RWP branch by fairly re-optimizing Trotter depth
r in {1, 2, 3} AFTER the H4-specific 09.04C hyperparameter search.

FAIRNESS RULE
-------------
For each replay window W:
1. Take ONE shared Stage-C parent: the H4 Step-C training-CV winner for that W.
2. For EACH r in {1,2,3}, give exactly the same local search budget using
   common.trotter_local_candidates():
      - same parent Hamiltonian parameters,
      - ±10% global dynamics variations,
      - ±10% dt variations,
      - 8 fixed-seed local perturbations.
3. Every dynamics trial rechecks all five readout families and Ridge lambda.
4. Selection uses 2022-2024 chronological CV only.
5. 2025 is diagnostic only.
6. 2026 is never loaded.

This avoids an unfair "change r only" comparison.

INPUT
-----
results/09_04c_h4_backfill_rwp_best_cv_per_window.csv

DEPENDENCIES
------------
Keep beside this script:
    09_04_rwp_common.py
    07_02b_memory_pair_isolation.py

OUTPUTS
-------
results/09_04c2_h4_backfill_all_trials.csv
results/09_04c2_h4_backfill_best_cv_per_window_r.csv
results/09_04c2_h4_backfill_hardware_aware_per_window_r.csv
results/09_04c2_h4_backfill_best_cv_per_window.csv
results/09_04c2_h4_backfill_hardware_aware_per_window.csv
results/09_04c2_h4_backfill_manifest.json
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

R_VALUES = (1, 2, 3)
SEED = 42
N_RANDOM = 8


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Keep the original IBM 09_04_rwp_common.py "
            "beside this script."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    spec.loader.exec_module(module)
    return module


common = load_module(
    "ibm_09_04_rwp_common_for_h4_backfill_c2",
    COMMON_PATH,
)


def row_to_shared_parent(row: pd.Series) -> dict:
    """Convert one Step-C CV-winning row to the shared C2 parent for one W."""
    c = common.candidate_from_row(row)

    W = int(row["window"])
    stage_c_tag = str(
        row.get(
            "trial_search_id",
            row.get("search_id", "stage_c_winner"),
        )
    )

    c["candidate_id"] = f"H4_W{W:02d}_Cparent_{stage_c_tag}"
    c["window"] = W

    return c


def cv_winner(df: pd.DataFrame) -> pd.Series:
    """Original CV-primary hierarchy; 2025 is never a selector."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    src = RESULTS / "09_04c_h4_backfill_rwp_best_cv_per_window.csv"

    out_trials = RESULTS / "09_04c2_h4_backfill_all_trials.csv"
    out_cv_wr = RESULTS / "09_04c2_h4_backfill_best_cv_per_window_r.csv"
    out_hw_wr = RESULTS / "09_04c2_h4_backfill_hardware_aware_per_window_r.csv"
    out_cv_w = RESULTS / "09_04c2_h4_backfill_best_cv_per_window.csv"
    out_hw_w = RESULTS / "09_04c2_h4_backfill_hardware_aware_per_window.csv"
    out_manifest = RESULTS / "09_04c2_h4_backfill_manifest.json"

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
            f"{src} missing. Run 09_04c_h4_rwp_hyperparameter_backfill.py first."
        )

    if args.overwrite:
        for p in outputs:
            if p.exists():
                p.unlink()
    else:
        existing = [p for p in outputs if p.exists()]
        if existing:
            raise FileExistsError(
                "C2 backfill output already exists. Use --overwrite only for "
                "an intentional full rerun:\n  "
                + "\n  ".join(str(p) for p in existing)
            )

    base_df = (
        pd.read_csv(src)
        .sort_values("window")
        .reset_index(drop=True)
    )

    if set(base_df["topology"].astype(str).unique()) != {"H4"}:
        raise RuntimeError("Expected H4-only Step-C winners.")

    if args.max_windows is not None:
        base_df = base_df.head(args.max_windows).copy()

    work_tv, cols, y_all = common.load_train_validation()

    all_trial_rows = []
    cv_wr_rows = []
    hw_wr_rows = []
    cv_w_rows = []
    hw_w_rows = []

    print("=" * 128)
    print("IBM H4 GAP REPAIR — STEP C2")
    print("H4-ONLY FAIR LOCAL TROTTER RE-OPTIMIZATION")
    print("=" * 128)
    print("Shared parent per W = Step-C training-CV winner.")
    print(f"r values: {list(R_VALUES)}")
    print(
        "Equal local budget per r: parent_same_H + 2 global dynamics + "
        "2 dt + 8 fixed-seed local perturbations."
    )
    print(f"Random local children per r: {N_RANDOM}; seed={SEED}")
    print(f"Readouts per dynamics trial: {len(common.READOUTS)}")
    print("Selection: 2022-2024 chronological CV only.")
    print("2025 = diagnostic only; 2026 = untouched / not loaded.")
    print()
    print(f"H4 replay windows to process: {len(base_df)}")
    print()

    for wi, row in base_df.iterrows():
        W = int(row["window"])
        parent = row_to_shared_parent(row)

        print("-" * 128)
        print(
            f"[{wi + 1}/{len(base_df)}] H4 W={W:02d} | "
            f"Step-C parent={row.get('trial_search_id', row.get('search_id', ''))} / "
            f"{row['readout']} | CV={row['cv_rmse']:.6f} | "
            f"r_parent={int(row['r'])}"
        )

        window_rows = []

        for r in R_VALUES:
            candidates = common.trotter_local_candidates(
                parent=parent,
                r=r,
                seed=SEED,
                n_random=N_RANDOM,
            )

            r_rows = []

            for trial_i, candidate in enumerate(candidates):
                readout_df, _ = common.evaluate_candidate_window(
                    candidate=candidate,
                    window=W,
                    work_tv=work_tv,
                    cols=cols,
                    y_all=y_all,
                )

                readout_df["window"] = W
                readout_df["c2_r"] = int(r)
                readout_df["c2_search_id"] = candidate["search_id"]
                readout_df["c2_trial_index"] = trial_i
                readout_df["stage_c_parent_search_id"] = str(
                    row.get(
                        "trial_search_id",
                        row.get("search_id", ""),
                    )
                )
                readout_df["stage_c_parent_readout"] = str(row["readout"])
                readout_df["stage_c_parent_cv_rmse"] = float(row["cv_rmse"])
                readout_df["backfill_step"] = "H4_RWP_09.04C2"
                readout_df["2025_used_for_selection"] = False
                readout_df["2026_used"] = False

                records = readout_df.to_dict("records")
                r_rows.extend(records)
                window_rows.extend(records)
                all_trial_rows.extend(records)

            r_df = pd.DataFrame(r_rows)

            cv_best_r = cv_winner(r_df)
            cv_best_r["selection_rule"] = (
                "within-W,r training CV primary; robustness/resources tie-break; "
                "2025 diagnostic only"
            )
            cv_wr_rows.append(cv_best_r.to_dict())

            hw_best_r, hw_mode_r = common.select_hardware_aware_row(r_df)
            hw_best_r = hw_best_r.copy()
            hw_best_r["selection_rule"] = hw_mode_r
            hw_best_r["2025_used_for_selection"] = False
            hw_best_r["2026_used"] = False
            hw_wr_rows.append(hw_best_r.to_dict())

            print(
                f"  r={r}: "
                f"CV-best={cv_best_r['c2_search_id']} / {cv_best_r['readout']} "
                f"CV={cv_best_r['cv_rmse']:.6f} "
                f"Val={cv_best_r['validation_rmse']:.6f} "
                f"shotSD={cv_best_r['shot_noise_forecast_sd_proxy_1024']:.3f} "
                f"maxR={cv_best_r['max_shot_to_train_std_ratio_1024']:.2f} || "
                f"HW={hw_best_r['c2_search_id']} / {hw_best_r['readout']} "
                f"CV={hw_best_r['cv_rmse']:.6f} "
                f"maxR={hw_best_r['max_shot_to_train_std_ratio_1024']:.2f}"
            )

        window_df = pd.DataFrame(window_rows)

        cv_best_w = cv_winner(window_df)
        cv_best_w["selection_rule"] = (
            "across r: training CV primary; robustness/resources tie-break; "
            "2025 diagnostic only"
        )
        cv_w_rows.append(cv_best_w.to_dict())

        hw_best_w, hw_mode_w = common.select_hardware_aware_row(window_df)
        hw_best_w = hw_best_w.copy()
        hw_best_w["selection_rule"] = hw_mode_w
        hw_best_w["2025_used_for_selection"] = False
        hw_best_w["2026_used"] = False
        hw_w_rows.append(hw_best_w.to_dict())

        print(
            f"  => W={W:02d} FINAL CV: r={int(cv_best_w['r'])}, "
            f"{cv_best_w['c2_search_id']} / {cv_best_w['readout']}, "
            f"CV={cv_best_w['cv_rmse']:.6f}, "
            f"SD={cv_best_w['cv_rmse_std']:.6f}"
        )
        print(
            f"  => W={W:02d} FINAL HW: r={int(hw_best_w['r'])}, "
            f"{hw_best_w['c2_search_id']} / {hw_best_w['readout']}, "
            f"CV={hw_best_w['cv_rmse']:.6f}, "
            f"maxR={hw_best_w['max_shot_to_train_std_ratio_1024']:.2f}"
        )

    all_df = pd.DataFrame(all_trial_rows)
    cv_wr_df = pd.DataFrame(cv_wr_rows)
    hw_wr_df = pd.DataFrame(hw_wr_rows)
    cv_w_df = pd.DataFrame(cv_w_rows)
    hw_w_df = pd.DataFrame(hw_w_rows)

    all_df.to_csv(out_trials, index=False)
    cv_wr_df.to_csv(out_cv_wr, index=False)
    hw_wr_df.to_csv(out_hw_wr, index=False)
    cv_w_df.to_csv(out_cv_w, index=False)
    hw_w_df.to_csv(out_hw_w, index=False)

    global_cv = cv_winner(cv_w_df)
    global_hw, global_hw_mode = common.select_hardware_aware_row(hw_w_df)
    global_hw = global_hw.copy()

    r_counts_cv = {
        str(int(k)): int(v)
        for k, v in cv_w_df["r"].value_counts().sort_index().items()
    }
    r_counts_hw = {
        str(int(k)): int(v)
        for k, v in hw_w_df["r"].value_counts().sort_index().items()
    }

    manifest = {
        "step": "IBM_H4_gap_repair_09.04C2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Fair local Trotter r=1,2,3 reoptimization for the missing H4 RWP branch."
        ),
        "source_step_c": str(src),
        "shared_parent_policy": (
            "one Step-C training-CV winner per window; same parent used for all r"
        ),
        "r_values": list(R_VALUES),
        "seed": SEED,
        "n_random_local_per_r": N_RANDOM,
        "readouts": list(common.READOUTS),
        "selection_period": "2022-2024",
        "validation_2025_role": "diagnostic_only",
        "test_2026_used": False,
        "counts": {
            "windows": int(len(base_df)),
            "all_trial_rows": int(len(all_df)),
            "best_cv_window_r_rows": int(len(cv_wr_df)),
            "hardware_aware_window_r_rows": int(len(hw_wr_df)),
            "best_cv_window_rows": int(len(cv_w_df)),
            "hardware_aware_window_rows": int(len(hw_w_df)),
        },
        "r_counts_among_window_cv_winners": r_counts_cv,
        "r_counts_among_window_hw_representatives": r_counts_hw,
        "global_cv_winner": {
            "window": int(global_cv["window"]),
            "r": int(global_cv["r"]),
            "search_id": str(global_cv["c2_search_id"]),
            "readout": str(global_cv["readout"]),
            "cv_rmse": float(global_cv["cv_rmse"]),
            "cv_rmse_std": float(global_cv["cv_rmse_std"]),
            "validation_rmse_diagnostic": float(global_cv["validation_rmse"]),
            "shot_noise_forecast_sd_proxy_1024": float(
                global_cv["shot_noise_forecast_sd_proxy_1024"]
            ),
            "max_shot_to_train_std_ratio_1024": float(
                global_cv["max_shot_to_train_std_ratio_1024"]
            ),
        },
        "global_hardware_aware_representative": {
            "window": int(global_hw["window"]),
            "r": int(global_hw["r"]),
            "search_id": str(global_hw["c2_search_id"]),
            "readout": str(global_hw["readout"]),
            "cv_rmse": float(global_hw["cv_rmse"]),
            "validation_rmse_diagnostic": float(global_hw["validation_rmse"]),
            "shot_noise_forecast_sd_proxy_1024": float(
                global_hw["shot_noise_forecast_sd_proxy_1024"]
            ),
            "max_shot_to_train_std_ratio_1024": float(
                global_hw["max_shot_to_train_std_ratio_1024"]
            ),
            "selection_rule": str(global_hw_mode),
        },
    }

    with out_manifest.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print()
    print("=" * 128)
    print("IBM H4 GAP REPAIR — STEP C2 COMPLETE")
    print("=" * 128)
    print(f"Total trial rows: {len(all_df)}")
    print(f"r counts among final per-window CV winners: {r_counts_cv}")
    print(f"r counts among final per-window HW reps:    {r_counts_hw}")
    print()
    print(
        "GLOBAL H4 C2 CV WINNER:\n"
        f"  W={int(global_cv['window'])}, r={int(global_cv['r'])}, "
        f"{global_cv['c2_search_id']} / {global_cv['readout']}\n"
        f"  CV={global_cv['cv_rmse']:.6f}, "
        f"CV_SD={global_cv['cv_rmse_std']:.6f}, "
        f"2025={global_cv['validation_rmse']:.6f} diagnostic-only\n"
        f"  shotSD={global_cv['shot_noise_forecast_sd_proxy_1024']:.3f}, "
        f"maxR={global_cv['max_shot_to_train_std_ratio_1024']:.2f}"
    )
    print()
    print(
        "GLOBAL H4 C2 HARDWARE-AWARE REPRESENTATIVE:\n"
        f"  W={int(global_hw['window'])}, r={int(global_hw['r'])}, "
        f"{global_hw['c2_search_id']} / {global_hw['readout']}\n"
        f"  CV={global_hw['cv_rmse']:.6f}, "
        f"2025={global_hw['validation_rmse']:.6f} diagnostic-only\n"
        f"  shotSD={global_hw['shot_noise_forecast_sd_proxy_1024']:.3f}, "
        f"maxR={global_hw['max_shot_to_train_std_ratio_1024']:.2f}"
    )
    print()
    print("Saved:")
    for p in outputs:
        print(f"  {p}")
    print()
    print(
        "STOP HERE. Send the full console output plus the five CSV files and "
        "manifest. Do not start 09.05 H4 gap-fill until C2 is interpreted."
    )


if __name__ == "__main__":
    main()
