"""
IBM H4 — Week 9.05B
H4 RWP intrinsic-memory evaluation on the predeclared memory-window grid.

PURPOSE
-------
Extend the original 09_05b_rwp_memory_gapfill.py logic to H4 after the
09.04C2 fair Trotter search.

IMPORTANT DIFFERENCE FROM THE OLD H0-H3 RUN
-------------------------------------------
We do NOT collapse H4 to r=3 before the Rule-2 memory test.

For every predeclared memory window
    W in {1, 2, 5, 7, 14, 21, 28}
and for every
    r in {1, 2, 3},
we take the UNION of:
    - the 09.04C2 CV winner for that (W, r),
    - the 09.04C2 hardware-aware winner for that (W, r).

If both roles point to the same dynamics, they are evaluated only once.

Thus Rule 2 can still reveal a memory-favorable r that is not the
forecasting-optimal r=3.

MEMORY METHOD
-------------
Exactly reuse the original 09.05B implementation:
- probe seeds [79001, 42, 101, 505, 707],
- independently permuted real input-angle channels,
- RWP restarts every endpoint,
- XYZ_all memory readout,
- delays 1..20,
- standardized Ridge(alpha=1e-6),
- corrected memory capacity with null floor 1/(N_val-1).

SELECTION
---------
Rule-2 winner = maximum mean corrected MC/ch across the retained H4 finalists.
Forecast CV is used only as a tie-break after memory.

2025 is diagnostic only.
2026 is untouched / never loaded.

INPUTS
------
results/09_04c2_h4_backfill_best_cv_per_window_r.csv
results/09_04c2_h4_backfill_hardware_aware_per_window_r.csv

DEPENDENCIES
------------
Keep beside this script:
    09_04_rwp_common.py
    09_05b_rwp_memory_gapfill.py
    07_02b_memory_pair_isolation.py

OUTPUTS
-------
results/09_05b_h4_rwp_memory_finalist_pool.csv
results/09_05b_h4_rwp_memory_by_seed.csv
results/09_05b_h4_rwp_memory_summary.csv
results/09_05b_h4_rwp_memory_rule2_winner.csv
results/09_05b_h4_rwp_memory_manifest.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
ORIGINAL_MEMORY_FILE = HERE / "09_05b_rwp_memory_gapfill.py"

CV_FILE = RESULTS / "09_04c2_h4_backfill_best_cv_per_window_r.csv"
HW_FILE = RESULTS / "09_04c2_h4_backfill_hardware_aware_per_window_r.csv"

POOL_FILE = RESULTS / "09_05b_h4_rwp_memory_finalist_pool.csv"
BY_SEED_FILE = RESULTS / "09_05b_h4_rwp_memory_by_seed.csv"
SUMMARY_FILE = RESULTS / "09_05b_h4_rwp_memory_summary.csv"
WINNER_FILE = RESULTS / "09_05b_h4_rwp_memory_rule2_winner.csv"
MANIFEST_FILE = RESULTS / "09_05b_h4_rwp_memory_manifest.json"

PROBE_SEEDS = [79001, 42, 101, 505, 707]
MEMORY_WINDOWS = [1, 2, 5, 7, 14, 21, 28]
R_VALUES = [1, 2, 3]


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Keep the original project script beside this file."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_0905b_h4_common")
original = load_module(ORIGINAL_MEMORY_FILE, "qrc_0905b_original")


def dynamics_signature(row: pd.Series) -> tuple:
    """Numerical H4 dynamics identity; readout is intentionally excluded."""
    edges = common.TOPOLOGY_EDGES["H4"]
    return (
        int(row["window"]),
        int(row["r"]),
        round(float(row["alpha"]), 12),
        round(float(row["dt"]), 12),
        round(float(row["hx"]), 12),
        round(float(row["hy"]), 12),
        tuple(
            round(float(row[f"J{i}{j}"]), 12)
            for i, j in edges
        ),
    )


def build_finalist_pool(cv_df: pd.DataFrame, hw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain CV + HW finalists for every required (W,r), then deduplicate only
    identical dynamics.  This deliberately preserves r=1,2,3.
    """
    required = {
        (W, r)
        for W in MEMORY_WINDOWS
        for r in R_VALUES
    }

    for name, df in [("CV", cv_df), ("HW", hw_df)]:
        if set(df["topology"].astype(str).unique()) != {"H4"}:
            raise RuntimeError(f"{name} input must contain H4 only.")

        observed = {
            (int(W), int(r))
            for W, r in (
                df[df["window"].isin(MEMORY_WINDOWS)][["window", "r"]]
                .drop_duplicates()
                .itertuples(index=False)
            )
        }

        missing = sorted(required - observed)
        if missing:
            raise RuntimeError(
                f"{name} input is missing required H4 (W,r) groups: "
                + ", ".join(f"W{W}/r{r}" for W, r in missing)
            )

    selected = []

    cv_use = cv_df[
        cv_df["window"].isin(MEMORY_WINDOWS)
        & cv_df["r"].isin(R_VALUES)
    ].copy()

    hw_use = hw_df[
        hw_df["window"].isin(MEMORY_WINDOWS)
        & hw_df["r"].isin(R_VALUES)
    ].copy()

    for _, row in cv_use.iterrows():
        x = row.copy()
        x["pool_role"] = "CV_WINNER"
        selected.append(x)

    for _, row in hw_use.iterrows():
        x = row.copy()
        x["pool_role"] = "HW_AWARE_WINNER"
        selected.append(x)

    pool_raw = pd.DataFrame(selected)

    # Deduplicate identical dynamics.  Prefer the CV row as the representative
    # when the same dynamics appears in both roles, matching the spirit of the
    # original 09.05B union.
    grouped = []
    pool_raw["_dyn_sig"] = pool_raw.apply(dynamics_signature, axis=1)

    for _, g in pool_raw.groupby("_dyn_sig", sort=False):
        roles = sorted(set(g["pool_role"].astype(str)))

        cv_rows = g[g["pool_role"] == "CV_WINNER"]
        if len(cv_rows):
            rep = cv_rows.sort_values("cv_rmse").iloc[0].copy()
        else:
            rep = g.sort_values("cv_rmse").iloc[0].copy()

        rep["pool_role"] = "+".join(roles)
        rep["source_cv_rmse_for_tiebreak"] = float(
            g["cv_rmse"].min()
        )
        grouped.append(rep)

    pool = pd.DataFrame(grouped).drop(columns=["_dyn_sig"], errors="ignore")
    pool = pool.sort_values(
        ["window", "r", "source_cv_rmse_for_tiebreak"]
    ).reset_index(drop=True)

    return pool


def row_to_candidate(row: pd.Series) -> dict:
    """
    H4-only C2 CSVs have a stable H4 coupling schema, so candidate_from_row()
    safely reconstructs the exact numerical dynamics.
    """
    candidate = common.candidate_from_row(row)

    if candidate["topology"] != "H4":
        raise RuntimeError("Expected H4 candidate.")

    candidate["window"] = int(row["window"])
    candidate["candidate_id"] = (
        f"H4_W{int(row['window']):02d}_r{int(row['r'])}_"
        f"{str(row['c2_search_id'])}"
    )

    # Numerical audit.
    checks = {
        "alpha": float(row["alpha"]),
        "dt": float(row["dt"]),
        "r": int(row["r"]),
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
    }

    for key, observed in checks.items():
        expected = candidate[key]
        if key == "r":
            ok = int(expected) == int(observed)
        else:
            ok = np.isclose(
                float(expected), float(observed),
                rtol=1e-10, atol=1e-12
            )

        if not ok:
            raise RuntimeError(
                f"Candidate reconstruction audit failed: "
                f"W={int(row['window'])}, r={int(row['r'])}, "
                f"{key}: row={observed}, candidate={expected}"
            )

    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--probe-seeds",
        default=",".join(str(x) for x in PROBE_SEEDS),
    )
    args = parser.parse_args()

    seeds = [
        int(x.strip())
        for x in args.probe_seeds.split(",")
        if x.strip()
    ]

    for p in [CV_FILE, HW_FILE]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run 09_04c2_h4_rwp_trotter_backfill.py first."
            )

    cv_df = pd.read_csv(CV_FILE)
    hw_df = pd.read_csv(HW_FILE)
    pool = build_finalist_pool(cv_df, hw_df)

    if args.overwrite:
        for p in [
            POOL_FILE,
            BY_SEED_FILE,
            SUMMARY_FILE,
            WINNER_FILE,
            MANIFEST_FILE,
        ]:
            if p.exists():
                p.unlink()

    pool.to_csv(POOL_FILE, index=False)

    work_tv, cols, _ = common.load_train_validation()

    if BY_SEED_FILE.exists():
        rows = pd.read_csv(BY_SEED_FILE).to_dict("records")
    else:
        rows = []

    completed = {
        (
            int(r["window"]),
            int(r["r"]),
            str(r["c2_search_id"]),
            int(r["probe_seed"]),
            str(r["pool_role"]),
        )
        for r in rows
    }

    angle_cache = {}

    print("=" * 128)
    print("WEEK 9.05B — H4 RWP MEMORY COMPLETION")
    print("=" * 128)
    print(f"Memory windows: {MEMORY_WINDOWS}")
    print(f"Trotter values retained for Rule 2: {R_VALUES}")
    print(f"Required (W,r) groups: {len(MEMORY_WINDOWS) * len(R_VALUES)}")
    print(f"Unique retained dynamics after CV/HW union: {len(pool)}")
    print(f"Probe seeds: {seeds}")
    print("Memory readout: XYZ_all")
    print("Delays: 1..20")
    print("RWP restarts every endpoint; no CONT washout.")
    print(
        "IMPORTANT: r=1,2,3 are all retained here. "
        "Rule 2 is not restricted to the forecasting-optimal r=3."
    )
    print("Selection uses memory first; forecasting CV only as a tie-break.")
    print("2025 = diagnostic only; 2026 = untouched / not loaded.")
    print()

    for i, row in pool.iterrows():
        candidate = row_to_candidate(row)

        W = int(row["window"])
        r = int(row["r"])
        search_id = str(row["c2_search_id"])
        alpha = float(candidate["alpha"])
        role = str(row["pool_role"])

        if alpha not in angle_cache:
            angle_cache[alpha] = common.make_angles(
                work_tv, cols, alpha
            )

        real_angles = angle_cache[alpha]

        for seed in seeds:
            key = (
                W,
                r,
                search_id,
                int(seed),
                role,
            )
            if key in completed:
                continue

            probe = original.build_probe(
                real_angles,
                seed,
            )

            A_list = common.build_channels(
                candidate,
                probe,
            )

            endpoints, master = (
                common.rwp_master_feature_bank_from_channels(
                    A_list,
                    W,
                )
            )

            X = common.select_features(
                master,
                "XYZ_all",
            )

            mc = original.evaluate_rwp_mc(
                endpoints,
                X,
                probe,
            )

            rows.append(
                {
                    "topology": "H4",
                    "window": W,
                    "r": r,
                    "c2_search_id": search_id,
                    "pool_role": role,
                    "probe_seed": int(seed),
                    "alpha": candidate["alpha"],
                    "dt": candidate["dt"],
                    "hx": candidate["hx"],
                    "hy": candidate["hy"],
                    "source_readout": str(row["readout"]),
                    "source_cv_rmse": float(row["cv_rmse"]),
                    "source_cv_rmse_for_tiebreak": float(
                        row["source_cv_rmse_for_tiebreak"]
                    ),
                    "source_validation_rmse_diagnostic": float(
                        row["validation_rmse"]
                    ),
                    **common.candidate_to_flat(candidate),
                    **mc,
                    "2025_used_for_selection": False,
                    "2026_used": False,
                }
            )

            completed.add(key)

            pd.DataFrame(rows).to_csv(
                BY_SEED_FILE,
                index=False,
            )

        print(
            f"[{i+1:02d}/{len(pool)}] "
            f"H4 W={W:02d} r={r} {search_id} "
            f"role={role}"
        )

    by_seed = pd.DataFrame(rows)

    group_cols = [
        "topology",
        "window",
        "r",
        "c2_search_id",
        "pool_role",
        "alpha",
        "dt",
        "hx",
        "hy",
        "source_readout",
        "source_cv_rmse",
        "source_cv_rmse_for_tiebreak",
        "source_validation_rmse_diagnostic",
    ]

    summary = (
        by_seed.groupby(
            group_cols,
            dropna=False,
        )
        .agg(
            n_probe_seeds=("probe_seed", "nunique"),
            MC_ch_mean=("MC_ch_corrected", "mean"),
            MC_ch_std=("MC_ch_corrected", "std"),
            MC_total_mean=("MC_total_corrected", "mean"),
            MC_delay1_mean=("MC_delay1_mean", "mean"),
            MC_delay2_mean=("MC_delay2_mean", "mean"),
            MC_delay5_mean=("MC_delay5_mean", "mean"),
            MC_delay10_mean=("MC_delay10_mean", "mean"),
            MC_delay20_mean=("MC_delay20_mean", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    winner = (
        summary.sort_values(
            [
                "MC_ch_mean",
                "source_cv_rmse_for_tiebreak",
            ],
            ascending=[False, True],
        )
        .iloc[[0]]
        .copy()
    )

    winner.to_csv(
        WINNER_FILE,
        index=False,
    )

    w = winner.iloc[0]

    manifest = {
        "step": "09.05B_H4_RWP_memory",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "topology": "H4",
        "memory_windows": MEMORY_WINDOWS,
        "r_values_tested": R_VALUES,
        "probe_seeds": seeds,
        "memory_readout": "XYZ_all",
        "delays": [1, 20],
        "pool_policy": (
            "union of CV and hardware-aware C2 finalists for every "
            "(memory_window, r), with identical dynamics deduplicated"
        ),
        "selection_rule": (
            "maximum mean corrected MC/ch; source training CV tie-break"
        ),
        "selection_period_for_forecast_tiebreak": "2022-2024",
        "validation_2025_role": "diagnostic_only",
        "test_2026_used": False,
        "counts": {
            "required_window_r_groups": int(
                len(MEMORY_WINDOWS) * len(R_VALUES)
            ),
            "unique_dynamics": int(len(pool)),
            "probe_rows": int(len(by_seed)),
            "summary_rows": int(len(summary)),
        },
        "rule2_winner": {
            "window": int(w["window"]),
            "r": int(w["r"]),
            "c2_search_id": str(w["c2_search_id"]),
            "pool_role": str(w["pool_role"]),
            "MC_ch_mean": float(w["MC_ch_mean"]),
            "MC_ch_std": float(w["MC_ch_std"]),
            "MC_total_mean": float(w["MC_total_mean"]),
            "source_cv_rmse_for_tiebreak": float(
                w["source_cv_rmse_for_tiebreak"]
            ),
            "source_validation_rmse_diagnostic": float(
                w["source_validation_rmse_diagnostic"]
            ),
        },
    }

    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 128)
    print("09.05B H4 COMPLETE")
    print("=" * 128)
    print("RULE-2 H4 RWP WINNER:")
    print(
        f"  W={int(w['window'])}, r={int(w['r'])}, "
        f"{w['c2_search_id']} | "
        f"MC/ch={w['MC_ch_mean']:.6f} ± {w['MC_ch_std']:.6f}"
    )
    print(
        f"  source CV={w['source_cv_rmse_for_tiebreak']:.6f}, "
        f"2025={w['source_validation_rmse_diagnostic']:.6f} "
        f"diagnostic-only"
    )
    print()
    print("Saved:")
    print(f"  {POOL_FILE}")
    print(f"  {BY_SEED_FILE}")
    print(f"  {SUMMARY_FILE}")
    print(f"  {WINNER_FILE}")
    print(f"  {MANIFEST_FILE}")
    print()
    print(
        "STOP HERE. Send the console output and the five files above. "
        "Do not run 09.05C until the H4 Rule-2 result is interpreted."
    )


if __name__ == "__main__":
    main()
