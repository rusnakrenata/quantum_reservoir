"""
Week 12.3E — IQM H5/H6 RWP Pareto / hardware-aware shortlist.

Direct one-to-one adaptation of IBM `09_04b_rwp_pareto_shortlist.py`.

Input:
    results/12_03d_rwp_all_readouts.csv
    results/12_03d_rwp_best_cv_per_topology_window.csv
    results/12_03d_rwp_hardware_aware_per_topology_window.csv

For each topology-window branch, evaluate the five readout families on the same
training-only accuracy / finite-shot robustness / logical-resource Pareto rule
used in the IBM Week-9.04B pipeline.

IMPORTANT:
- No weighted score.
- Pareto is evaluated within each H×W branch across readouts.
- 2025 validation is diagnostic only and is not used for selection.
- No windows are discarded here.
- No dynamics are re-optimized here.
- 12.3F will perform the direct analogue of IBM 09.04C.
- No QPU jobs are submitted.
- 2026 remains untouched.
"""

from __future__ import annotations

from importlib import import_module

import pandas as pd


common = import_module("12_03d_iqm_rwp_common")


def main():
    src = common.RESULTS / "12_03d_rwp_all_readouts.csv"

    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing. Run Week 12.3D first."
        )

    df = pd.read_csv(src)

    marked_parts = []
    pareto_parts = []

    # Direct IBM 09.04B rule:
    # Pareto is evaluated *within each topology-window branch* across readouts.
    #
    # All objectives are minimized:
    #   1. chronological training CV RMSE,
    #   2. finite-shot forecast-noise proxy,
    #   3. maximum absolute raw prediction sensitivity,
    #   4. worst shot-noise / training-STD resolvability ratio,
    #   5. logical feature-vector CZ-equivalent cost,
    #   6. number of measurement settings.
    #
    # 2025 validation metrics remain in the rows only as diagnostics.
    for (topology, W), group in df.groupby(
        ["topology", "window"],
        sort=True,
    ):
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
        pareto_parts.append(
            g[g["pareto_training_hardware"]].copy()
        )

    if not marked_parts:
        raise RuntimeError(
            "12.3D input contains no topology-window branches."
        )

    marked = pd.concat(
        marked_parts,
        ignore_index=True,
    )

    pareto = pd.concat(
        pareto_parts,
        ignore_index=True,
    )

    out_all = (
        common.RESULTS
        / "12_03e_rwp_readout_pareto_all.csv"
    )

    out_par = (
        common.RESULTS
        / "12_03e_rwp_readout_pareto_candidates.csv"
    )

    marked.to_csv(
        out_all,
        index=False,
    )

    pareto.to_csv(
        out_par,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Descriptive summaries only.
    #
    # Exactly as in IBM 09.04B, this does NOT prune windows before the next
    # hyperparameter-reoptimization stage. 12.3F will still process every H×W
    # branch.
    # -------------------------------------------------------------------------

    cv_path = (
        common.RESULTS
        / "12_03d_rwp_best_cv_per_topology_window.csv"
    )

    hw_path = (
        common.RESULTS
        / "12_03d_rwp_hardware_aware_per_topology_window.csv"
    )

    if not cv_path.exists():
        raise FileNotFoundError(
            f"{cv_path} missing. Run Week 12.3D first."
        )

    if not hw_path.exists():
        raise FileNotFoundError(
            f"{hw_path} missing. Run Week 12.3D first."
        )

    cv_best = pd.read_csv(cv_path)
    hw_best = pd.read_csv(hw_path)

    by_topology = []

    for topology, group in hw_best.groupby(
        "topology",
        sort=True,
    ):
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

    out_best_hw = (
        common.RESULTS
        / "12_03e_best_hardware_aware_by_topology.csv"
    )

    pd.DataFrame(
        by_topology
    ).to_csv(
        out_best_hw,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print("=" * 120)
    print(
        "WEEK 12.3E — IQM H5/H6 "
        "TRAINING-ONLY READOUT/HARDWARE PARETO ANALYSIS"
    )
    print("=" * 120)

    print(
        "Direct one-to-one adaptation of IBM 09.04B."
    )
    print(
        "No weighted score. "
        "2025 validation is not used for selection. "
        "2026 is untouched."
    )
    print()

    print(
        f"Input readout rows: {len(marked)}"
    )
    print(
        f"Pareto rows:        {len(pareto)}"
    )
    print(
        "H×W branches:       "
        f"{marked.groupby(['topology', 'window']).ngroups}"
    )
    print()

    print(
        "Readout counts among CV winners:"
    )
    print(
        cv_best["readout"]
        .value_counts()
        .to_string()
    )
    print()

    print(
        "Readout counts among hardware-aware representatives:"
    )
    print(
        hw_best["readout"]
        .value_counts()
        .to_string()
    )
    print()

    print(
        "Best descriptive hardware-aware representative "
        "per IQM-native topology:"
    )

    best_hw_df = pd.DataFrame(by_topology)

    show_cols = [
        "topology",
        "window",
        "readout",
        "cv_rmse",
        "validation_rmse",
        "shot_noise_forecast_sd_proxy_1024",
        "max_shot_to_train_std_ratio_1024",
        "feature_vector_cz",
        "n_settings",
    ]

    existing_show_cols = [
        c
        for c in show_cols
        if c in best_hw_df.columns
    ]

    if len(best_hw_df) > 0:
        print(
            best_hw_df[
                existing_show_cols
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "IMPORTANT: 12.3F re-optimizes ALL H×W branches; "
        "12.3E does not discard windows."
    )
    print(
        "No QPU jobs submitted."
    )
    print(
        "2026 untouched."
    )
    print()

    print(
        f"Saved: {out_all}"
    )
    print(
        f"Saved: {out_par}"
    )
    print(
        f"Saved: {out_best_hw}"
    )


if __name__ == "__main__":
    main()
