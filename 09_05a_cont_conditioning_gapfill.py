
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
CONT_FILE = RESULTS / "09_03c_forecast_best_135.csv"
CHECKPOINT = RESULTS / "09_05a_cont_conditioning_all.csv"

ALPHA = 0.75
DT = 1.6


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_0905_common")


def candidate_from_cont_row(row):
    topology = str(row["topology"])
    J = {
        edge: float(row[col])
        for edge, col in common.EDGE_TO_COLUMN[topology].items()
    }
    return {
        "topology": topology,
        "alpha": ALPHA,
        "dt": DT,
        "r": int(row["trotter_r"]),
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
        "J": J,
    }


def cont_master_feature_bank(A_list):
    rows = []
    rho_m = common.qrc.memory_zero_density()

    for A in A_list:
        rho_i, rho_m = common.qrc.final_reduced_states(A, rho_m)
        rows.append(common.reduced_feature_row(rho_i, rho_m))

    return pd.DataFrame(rows)


def evaluate_cont(master, y_all, readout):
    X = common.select_features(master, readout)
    Xtr = X[: common.N_TRAIN]
    Xva = X[common.N_TRAIN :]
    ytr = y_all[: common.N_TRAIN]
    yva = y_all[common.N_TRAIN :]

    lam, folds, cv_summary = common.select_lambda_training_only(Xtr, ytr)
    model, scaler, keep = common.fit_scaled_ridge(Xtr, ytr, lam)
    pred = common.predict_scaled_ridge(model, scaler, keep, Xva)

    cond = common.readout_conditioning_metrics(
        feature_names=common.READOUT_FEATURES[readout],
        X_train=Xtr,
        model=model,
        scaler=scaler,
        keep=keep,
        shots_proxy=common.SHOTS_PROXY,
    )

    return {
        "readout": readout,
        "measurement_plan": common.READOUT_MEASUREMENT_PLAN[readout],
        "n_features": len(common.READOUT_FEATURES[readout]),
        "n_settings": common.READOUT_SETTINGS[readout],
        "selected_lambda": float(lam),
        "cv_rmse": float(cv_summary.iloc[0]["cv_rmse_mean"]),
        "cv_rmse_std": float(cv_summary.iloc[0]["cv_rmse_std"]),
        "validation_rmse": common.rmse(yva, pred),
        "validation_mae": common.mae(yva, pred),
        "validation_bias": common.bias(yva, pred),
        "active_feature_count": int(np.sum(keep)),
        **cond,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not CONT_FILE.exists():
        raise FileNotFoundError(CONT_FILE)

    source = pd.read_csv(CONT_FILE)

    # 84 unique retained dynamics; readouts will be rechecked symmetrically.
    dynamics = (
        source.sort_values(["topology", "config_id"])
        .drop_duplicates("config_id")
        .reset_index(drop=True)
    )

    work_tv, cols, y_all = common.load_train_validation()
    angles = common.make_angles(work_tv, cols, ALPHA)

    if args.overwrite and CHECKPOINT.exists():
        CHECKPOINT.unlink()

    if CHECKPOINT.exists():
        rows = pd.read_csv(CHECKPOINT).to_dict("records")
    else:
        rows = []

    completed = {
        (str(r["config_id"]), str(r["readout"]))
        for r in rows
    }

    print("=" * 120)
    print("09.05A — CONT REVISED READOUT CONDITIONING GAP-FILL")
    print("=" * 120)
    print(f"Unique CONT dynamics: {len(dynamics)}")
    print(f"Readouts per dynamics: {len(common.READOUTS)}")
    print(f"Expected evaluations: {len(dynamics) * len(common.READOUTS)}")
    print("2022-2024 selects lambda/readout conditioning; 2025 diagnostic only; 2026 unused.")
    print()

    for idx, row in dynamics.iterrows():
        config_id = str(row["config_id"])

        pending = [
            ro for ro in common.READOUTS
            if (config_id, ro) not in completed
        ]
        if not pending:
            continue

        candidate = candidate_from_cont_row(row)
        A_list = common.build_channels(candidate, angles)
        master = cont_master_feature_bank(A_list)

        for readout in pending:
            result = evaluate_cont(master, y_all, readout)
            resource = common.resource_metrics(candidate, 1, readout)

            out = {
                "protocol": "CONT",
                "config_id": config_id,
                "source_seed_candidate_id": str(row["seed_candidate_id"]),
                "topology": candidate["topology"],
                "alpha": ALPHA,
                "dt": DT,
                "r": candidate["r"],
                "hx": candidate["hx"],
                "hy": candidate["hy"],
                **common.candidate_to_flat(candidate),
                **result,
                # These are terminal-state logical cost proxies only.
                # CONT repeated-measurement deployment semantics are more subtle.
                "terminal_measurement_core_cz_proxy": resource["core_cz_per_step"],
                "terminal_feature_vector_cz_proxy": resource["feature_vector_cz"],
                "2025_used_for_selection": False,
            }
            rows.append(out)
            completed.add((config_id, readout))

        pd.DataFrame(rows).to_csv(CHECKPOINT, index=False)

        print(
            f"[{idx+1:02d}/{len(dynamics)}] {candidate['topology']} "
            f"{config_id}: evaluated {len(pending)} readouts"
        )

    all_df = pd.DataFrame(rows)

    hw_rows = []
    cv_rows = []

    for H, group in all_df.groupby("topology"):
        cv_rows.append(group.sort_values("cv_rmse").iloc[0].to_dict())

        # common selector expects feature_vector_cz.
        g = group.copy()
        g["feature_vector_cz"] = g["terminal_feature_vector_cz_proxy"]
        hw, mode = common.select_hardware_aware_row(g)
        d = hw.to_dict()
        d["hardware_aware_selection_mode"] = mode
        hw_rows.append(d)

    pd.DataFrame(cv_rows).to_csv(
        RESULTS / "09_05a_cont_revised_readout_cv_winners.csv",
        index=False,
    )
    pd.DataFrame(hw_rows).to_csv(
        RESULTS / "09_05a_cont_revised_hardware_aware_winners.csv",
        index=False,
    )

    print()
    print("COMPLETE")
    print(f"Saved: {CHECKPOINT}")
    print("Saved: results/09_05a_cont_revised_readout_cv_winners.csv")
    print("Saved: results/09_05a_cont_revised_hardware_aware_winners.csv")


if __name__ == "__main__":
    main()
