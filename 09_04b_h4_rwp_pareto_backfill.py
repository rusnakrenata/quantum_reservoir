from __future__ import annotations

import importlib.util
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
COMMON_PATH = HERE / "09_04_rwp_common.py"

def load_module(name, path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Keep 09_04_rwp_common.py beside this script."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

common = load_module("h4_backfill_common_b", COMMON_PATH)

def main():
    src_all = RESULTS / "09_04a_h4_backfill_rwp_all_readouts.csv"
    src_cv = RESULTS / "09_04a_h4_backfill_rwp_best_cv_per_window.csv"
    src_hw = RESULTS / "09_04a_h4_backfill_rwp_hardware_aware_per_window.csv"

    for p in (src_all, src_cv, src_hw):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run 09_04a_h4_rwp_transfer_backfill.py first."
            )

    df = pd.read_csv(src_all)
    cv_best = pd.read_csv(src_cv)
    hw_best = pd.read_csv(src_hw)

    if set(df["topology"].astype(str).unique()) != {"H4"}:
        raise RuntimeError("Expected H4-only Step-A input.")

    marked_parts = []
    pareto_parts = []

    for (topology, W), group in df.groupby(["topology", "window"], sort=True):
        g = group.copy().reset_index(drop=True)
        g["pareto_training_hardware"] = common.pareto_mask(
            g,
            minimize=[
                "cv_rmse",
                "shot_noise_forecast_sd_proxy_1024",
                "max_abs_raw_prediction_sensitivity",
                "max_shot_to_train_std_ratio_1024",
                "feature_vector_cz",
                "n_settings",
            ],
        )
        marked_parts.append(g)
        pareto_parts.append(g[g["pareto_training_hardware"]].copy())

    marked = pd.concat(marked_parts, ignore_index=True)
    pareto = pd.concat(pareto_parts, ignore_index=True)

    out_all = RESULTS / "09_04b_h4_backfill_rwp_readout_pareto_all.csv"
    out_par = RESULTS / "09_04b_h4_backfill_rwp_readout_pareto_candidates.csv"
    out_hw = RESULTS / "09_04b_h4_backfill_best_hardware_aware.csv"
    out_summary = RESULTS / "09_04b_h4_backfill_summary.csv"

    marked.to_csv(out_all, index=False)
    pareto.to_csv(out_par, index=False)

    best_hw = (
        hw_best.sort_values(
            ["cv_rmse", "shot_noise_forecast_sd_proxy_1024", "feature_vector_cz"]
        )
        .iloc[[0]]
        .copy()
    )
    best_hw.to_csv(out_hw, index=False)

    rows = []
    for W in sorted(marked["window"].unique()):
        gp = pareto[pareto["window"] == W]
        c = cv_best[cv_best["window"] == W].sort_values(
            ["cv_rmse", "shot_noise_forecast_sd_proxy_1024", "feature_vector_cz"]
        ).iloc[0]
        h = hw_best[hw_best["window"] == W].sort_values(
            ["cv_rmse", "shot_noise_forecast_sd_proxy_1024", "feature_vector_cz"]
        ).iloc[0]
        rows.append({
            "topology": "H4",
            "window": int(W),
            "n_pareto_readouts": int(len(gp)),
            "pareto_readouts": ";".join(sorted(gp["readout"].astype(str))),
            "cv_best_readout": c["readout"],
            "cv_best_cv_rmse": c["cv_rmse"],
            "cv_best_cv_sd": c["cv_rmse_std"],
            "cv_best_validation_rmse_diagnostic": c["validation_rmse"],
            "cv_best_shot_sd_1024": c["shot_noise_forecast_sd_proxy_1024"],
            "cv_best_max_shot_to_train_std_1024": c["max_shot_to_train_std_ratio_1024"],
            "hw_readout": h["readout"],
            "hw_cv_rmse": h["cv_rmse"],
            "hw_shot_sd_1024": h["shot_noise_forecast_sd_proxy_1024"],
            "hw_max_shot_to_train_std_1024": h["max_shot_to_train_std_ratio_1024"],
            "2025_used_for_selection": False,
            "2026_used": False,
        })
    pd.DataFrame(rows).to_csv(out_summary, index=False)

    global_cv = cv_best.sort_values(
        ["cv_rmse", "shot_noise_forecast_sd_proxy_1024", "feature_vector_cz"]
    ).iloc[0]
    global_hw = best_hw.iloc[0]

    print("=" * 120)
    print("IBM H4 GAP REPAIR — STEP B")
    print("H4-ONLY TRAINING/HARDWARE READOUT PARETO ANALYSIS")
    print("=" * 120)
    print("No weighted score. 2025 diagnostic only. 2026 untouched.")
    print("This step does NOT discard replay windows.")
    print()
    print(f"Input rows:    {len(marked)}")
    print(f"H4×W branches: {marked['window'].nunique()}")
    print(f"Pareto rows:   {len(pareto)}")
    print()
    print("CV-winner readout counts:")
    print(cv_best["readout"].value_counts().to_string())
    print()
    print("Hardware-aware readout counts:")
    print(hw_best["readout"].value_counts().to_string())
    print()
    print("Pareto-front readout counts:")
    print(pareto["readout"].value_counts().to_string())
    print()
    print(
        "Best H4 transfer by CV: "
        f"W={int(global_cv['window'])}, {global_cv['readout']}, "
        f"CV={global_cv['cv_rmse']:.6f}, SD={global_cv['cv_rmse_std']:.6f}, "
        f"2025={global_cv['validation_rmse']:.6f} diagnostic-only, "
        f"shotSD={global_cv['shot_noise_forecast_sd_proxy_1024']:.3f}, "
        f"maxR={global_cv['max_shot_to_train_std_ratio_1024']:.2f}"
    )
    print(
        "Best H4 hardware-aware transfer: "
        f"W={int(global_hw['window'])}, {global_hw['readout']}, "
        f"CV={global_hw['cv_rmse']:.6f}, "
        f"shotSD={global_hw['shot_noise_forecast_sd_proxy_1024']:.3f}, "
        f"maxR={global_hw['max_shot_to_train_std_ratio_1024']:.2f}"
    )
    print()
    print("Saved:")
    print(f"  {out_all}")
    print(f"  {out_par}")
    print(f"  {out_hw}")
    print(f"  {out_summary}")
    print()
    print("STOP HERE and send the console output plus all four files.")

if __name__ == "__main__":
    main()
