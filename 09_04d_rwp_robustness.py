
from __future__ import annotations

import argparse
from importlib import import_module

import numpy as np
import pandas as pd

common = import_module("09_04_rwp_common")


def perturbed_candidates(parent: dict, seed: int, n: int):
    rng = np.random.default_rng(seed)
    out = []

    # Include exact finalist.
    base = {
        **parent,
        "J": dict(parent["J"]),
        "robustness_id": "exact_finalist",
    }
    out.append(base)

    for k in range(n):
        c = {
            **parent,
            "J": dict(parent["J"]),
        }

        c["alpha"] = float(
            np.clip(
                c["alpha"]
                *
                rng.normal(1.0, 0.04),
                0.10,
                1.25,
            )
        )

        c["dt"] = float(
            c["dt"]
            *
            np.clip(
                rng.normal(1.0, 0.04),
                0.90,
                1.10,
            )
        )

        c["hx"] = float(
            c["hx"]
            *
            np.clip(
                rng.normal(1.0, 0.05),
                0.88,
                1.12,
            )
        )

        if not np.isclose(c["hy"], 0.0):
            c["hy"] = float(
                c["hy"]
                *
                np.clip(
                    rng.normal(1.0, 0.06),
                    0.85,
                    1.15,
                )
            )

        c["J"] = {
            edge: float(value)
            *
            float(
                np.clip(
                    rng.normal(1.0, 0.06),
                    0.85,
                    1.15,
                )
            )
            for edge, value in c["J"].items()
        }

        c["robustness_id"] = f"local_seed_{k:02d}"
        out.append(c)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perturbations",
        type=int,
        default=5,
        help="Local parameter-robustness perturbations per H×W finalist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    src = common.RESULTS / "09_04c2_rwp_final_hardware_aware_per_topology_window.csv"
    out_all = common.RESULTS / "09_04d_rwp_robustness_all.csv"
    out_summary = common.RESULTS / "09_04d_rwp_robustness_summary.csv"
    out_final = common.RESULTS / "09_04d_rwp_final_table.csv"

    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing. Run 09_04c2 first."
        )

    if args.overwrite:
        for p in [out_all, out_summary, out_final]:
            if p.exists():
                p.unlink()

    finalists = pd.read_csv(src).sort_values(
        ["topology", "window"]
    ).reset_index(drop=True)

    if args.max_branches is not None:
        finalists = finalists.head(
            args.max_branches
        ).copy()

    work_tv, cols, y_all = common.load_train_validation()

    robustness_rows = []

    print("=" * 120)
    print("WEEK 9.04D — RWP FINALIST ROBUSTNESS")
    print("=" * 120)
    print(
        "Finalists are tested under small local parameter perturbations."
    )
    print(
        "This is a stability diagnostic, NOT another winner-selection score."
    )
    print(
        "2022-2024 CV remains primary; 2025 remains diagnostic; 2026 frozen."
    )
    print()

    for branch_i, row in finalists.iterrows():
        topology = str(row["topology"])
        W = int(row["window"])
        readout = str(row["readout"])

        parent = common.candidate_from_row(row)
        parent["window"] = W
        parent["candidate_id"] = (
            f"{topology}_W{W:02d}_finalist"
        )

        variants = perturbed_candidates(
            parent=parent,
            seed=args.seed + 1000 * branch_i,
            n=args.perturbations,
        )

        local_rows = []

        for variant in variants:
            readout_df, _ = common.evaluate_candidate_window(
                candidate=variant,
                window=W,
                work_tv=work_tv,
                cols=cols,
                y_all=y_all,
                readouts=[readout],
            )

            rr = readout_df.iloc[0].to_dict()
            rr["window"] = W
            rr["robustness_id"] = variant["robustness_id"]

            local_rows.append(rr)
            robustness_rows.append(rr)

        local = pd.DataFrame(local_rows)

        print(
            f"{topology} W={W:02d} r={int(row['r'])} "
            f"{readout:<22s}  "
            f"CV exact={float(local.iloc[0]['cv_rmse']):.6f}  "
            f"local mean±sd="
            f"{local['cv_rmse'].mean():.6f}±{local['cv_rmse'].std(ddof=1):.6f}"
        )

    all_df = pd.DataFrame(robustness_rows)

    all_df.to_csv(
        out_all,
        index=False,
    )

    summary = (
        all_df.groupby(
            ["topology", "window", "r", "readout"],
            as_index=False,
        )
        .agg(
            cv_rmse_mean=("cv_rmse", "mean"),
            cv_rmse_std=("cv_rmse", "std"),
            cv_rmse_max=("cv_rmse", "max"),
            validation_rmse_mean=("validation_rmse", "mean"),
            validation_rmse_std=("validation_rmse", "std"),
            shot_noise_forecast_sd_proxy_1024_mean=("shot_noise_forecast_sd_proxy_1024", "mean"),
            shot_noise_forecast_sd_proxy_1024_std=("shot_noise_forecast_sd_proxy_1024", "std"),
            max_abs_raw_prediction_sensitivity_mean=("max_abs_raw_prediction_sensitivity", "mean"),
            max_shot_to_train_std_ratio_1024_mean=("max_shot_to_train_std_ratio_1024", "mean"),
            feature_vector_cz=("feature_vector_cz", "first"),
            n_settings=("n_settings", "first"),
            n_features=("n_features", "first"),
        )
        .sort_values(
            ["topology", "window"]
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        out_summary,
        index=False,
    )

    # Merge robustness summary onto the exact final candidates.
    final = finalists.merge(
        summary,
        on=["topology", "window", "r", "readout"],
        how="left",
        suffixes=("", "_robust"),
    )

    final.to_csv(
        out_final,
        index=False,
    )

    print()
    print("=" * 120)
    print("9.04D COMPLETE")
    print("=" * 120)
    print(f"Saved: {out_all}")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_final}")


if __name__ == "__main__":
    main()
