from __future__ import annotations

from importlib import import_module

import pandas as pd

common = import_module("09_04_rwp_common")


def main():
    src = common.RESULTS / "09_04a_rwp_all_readouts.csv"

    if not src.exists():
        raise FileNotFoundError(f"{src} missing. Run 09_04a first.")

    df = pd.read_csv(src)

    marked_parts = []
    pareto_parts = []

    # Pareto is evaluated *within each topology-window branch* across readouts.
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

    out_all = common.RESULTS / "09_04b_rwp_readout_pareto_all.csv"
    out_par = common.RESULTS / "09_04b_rwp_readout_pareto_candidates.csv"

    marked.to_csv(out_all, index=False)
    pareto.to_csv(out_par, index=False)

    # Descriptive summaries only.  Stage C still processes EVERY H×W branch.
    cv_best = pd.read_csv(
        common.RESULTS / "09_04a_rwp_best_cv_per_topology_window.csv"
    )
    hw_best = pd.read_csv(
        common.RESULTS / "09_04a_rwp_hardware_aware_per_topology_window.csv"
    )

    by_topology = []
    for topology, group in hw_best.groupby("topology", sort=True):
        row = (
            group.sort_values(
                [
                    "cv_rmse",
                    "shot_noise_forecast_sd_proxy_1024",
                    "feature_vector_cz",
                ]
            )
            .iloc[0]
            .copy()
        )
        by_topology.append(row)

    pd.DataFrame(by_topology).to_csv(
        common.RESULTS / "09_04b_best_hardware_aware_by_topology.csv",
        index=False,
    )

    print("=" * 120)
    print("WEEK 9.04B — TRAINING-ONLY READOUT/HARDWARE PARETO ANALYSIS")
    print("=" * 120)
    print("No weighted score. 2025 validation is not used for selection.")
    print()
    print(f"Input readout rows: {len(marked)}")
    print(f"Pareto rows:        {len(pareto)}")
    print(f"H×W branches:       {marked.groupby(['topology','window']).ngroups}")
    print()
    print("Readout counts among CV winners:")
    print(cv_best["readout"].value_counts().to_string())
    print()
    print("Readout counts among hardware-aware representatives:")
    print(hw_best["readout"].value_counts().to_string())
    print()
    print("IMPORTANT: Stage C re-optimizes all H×W branches; this stage does not discard windows.")
    print(f"Saved: {out_all}")
    print(f"Saved: {out_par}")


if __name__ == "__main__":
    main()
