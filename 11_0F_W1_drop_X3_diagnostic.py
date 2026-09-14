
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# WEEK 11.0F — POST-HOC DIAGNOSTIC: REMOVE X3 FROM THE W1 READOUT
#
# NO QPU CALLS ARE MADE.
#
# Reuses already-collected 11.0E 1024-shot RAW and M3 features and compares:
#
#   FULL_XZ8 = [X0,X1,X2,X3,Z0,Z1,Z2,Z3]
#   DROP_X3  = [X0,X1,X2,   Z0,Z1,Z2,Z3]
#
# For each readout:
#   - lambda selected on 2022-2024 ONLY
#   - scaler fit on 2022-2024 ONLY
#   - Ridge weights/intercept fit on 2022-2024 ONLY
#   - 2025 used only for this diagnostic evaluation
#   - 2026 remains frozen / unused
#
# SCIENTIFIC CAVEAT:
# DROP_X3 was motivated after inspecting 2025 QPU behavior, so its 2025
# result is exploratory/post-hoc, not an unbiased final validation estimate.
# =============================================================================


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

BASE_SCRIPT = HERE / "11_0D_RWP_W1_full_2025_validation.py"
FEATURE_FILE = RESULTS / "11_0E_features_raw_vs_M3.csv"

N_VAL = 365
SHOTS = 1024

FEATURE_SETS = {
    "FULL_XZ8": [
        "X0", "X1", "X2", "X3",
        "Z0", "Z1", "Z2", "Z3",
    ],
    "DROP_X3": [
        "X0", "X1", "X2",
        "Z0", "Z1", "Z2", "Z3",
    ],
}


def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put this script beside the validated 11.0D script."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    spec.loader.exec_module(module)
    return module


base = import_module_from_path(
    "week11_0D_base_for_11_0F",
    BASE_SCRIPT,
)


def load_qpu_feature_matrix():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"{FEATURE_FILE} not found.\n"
            "Run 11_0E_RWP_W1_1024_M3.py first."
        )

    df = pd.read_csv(FEATURE_FILE)

    required = {
        "validation_offset",
        "endpoint_global",
        "feature",
        "ideal",
        "raw",
        "M3",
    }

    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"{FEATURE_FILE} is missing columns: {sorted(missing)}"
        )

    def pivot(value_col):
        p = (
            df.pivot_table(
                index=["validation_offset", "endpoint_global"],
                columns="feature",
                values=value_col,
                aggfunc="first",
            )
            .sort_index()
        )
        return p[FEATURE_SETS["FULL_XZ8"]]

    ideal = pivot("ideal")
    raw = pivot("raw")
    m3 = pivot("M3")

    if not (
        ideal.index.equals(raw.index)
        and ideal.index.equals(m3.index)
    ):
        raise RuntimeError("Ideal/raw/M3 indexes do not align.")

    if len(ideal) != N_VAL:
        raise RuntimeError(
            f"Expected {N_VAL} validation rows, got {len(ideal)}."
        )

    return ideal, raw, m3


def feature_indexes(all_names, selected_names):
    lookup = {name: i for i, name in enumerate(all_names)}
    return np.array(
        [lookup[name] for name in selected_names],
        dtype=int,
    )


def prediction_metrics(y_true, pred):
    return {
        "rmse": base.rmse(y_true, pred),
        "mae": base.mae(y_true, pred),
        "bias": base.bias(y_true, pred),
    }


def training_readout_sensitivity(
    feature_names,
    X_train,
    model,
    scaler,
    keep,
    shots,
):
    rows = []

    active_idx = np.flatnonzero(keep)
    coef_by_original = np.zeros(len(feature_names), dtype=float)
    coef_by_original[active_idx] = np.asarray(model.coef_, dtype=float)

    mean_by_original = np.full(len(feature_names), np.nan, dtype=float)
    scale_by_original = np.full(len(feature_names), np.nan, dtype=float)
    mean_by_original[active_idx] = scaler.mean_
    scale_by_original[active_idx] = scaler.scale_

    for j, feature in enumerate(feature_names):
        train_mean_raw = float(np.mean(X_train[:, j]))
        train_std_raw = float(np.std(X_train[:, j], ddof=0))
        active = bool(keep[j])

        if active:
            ridge_weight = float(coef_by_original[j])
            scale = float(scale_by_original[j])
            raw_sensitivity = ridge_weight / scale

            pauli_mean = float(np.clip(train_mean_raw, -1.0, 1.0))
            shot_sd = float(
                np.sqrt(max(0.0, 1.0 - pauli_mean ** 2) / shots)
            )
            forecast_shot_sd_proxy = abs(raw_sensitivity) * shot_sd
        else:
            ridge_weight = 0.0
            raw_sensitivity = 0.0
            shot_sd = np.nan
            forecast_shot_sd_proxy = 0.0

        rows.append({
            "feature": feature,
            "active": active,
            "train_mean_raw": train_mean_raw,
            "train_std_raw": train_std_raw,
            "ridge_weight_standardized": ridge_weight,
            "raw_prediction_sensitivity_dy_dx": raw_sensitivity,
            f"shot_sd_proxy_{shots}": shot_sd,
            f"forecast_shot_sd_proxy_{shots}": forecast_shot_sd_proxy,
        })

    out = pd.DataFrame(rows)

    total_proxy = float(
        np.sqrt(
            np.sum(
                out[f"forecast_shot_sd_proxy_{shots}"].to_numpy(dtype=float) ** 2
            )
        )
    )

    return out, total_proxy


def main():
    print("=" * 120)
    print("WEEK 11.0F — POST-HOC W1 DIAGNOSTIC: REMOVE X3")
    print("=" * 120)
    print()
    print("NO QPU JOBS WILL BE SUBMITTED.")
    print()
    print("Chronology:")
    print("  2022-2024 = lambda/scaler/Ridge weights/intercept")
    print("  2025      = diagnostic evaluation")
    print("  2026      = frozen / unused")
    print()
    print(
        "IMPORTANT: DROP_X3 was motivated after seeing 2025 hardware data, "
        "so its 2025 result is exploratory/post-hoc."
    )

    candidate = base.load_h4_winner_dynamics()

    (
        work_tv,
        cols,
        angles,
        y,
        train_idx,
        val_idx,
    ) = base.load_train_validation_only()

    X_ideal_all = base.build_ideal_w1_feature_bank(
        candidate,
        angles,
    )

    all_feature_names = list(base.XZ_COLUMNS)

    if all_feature_names != FEATURE_SETS["FULL_XZ8"]:
        raise RuntimeError(
            "Unexpected feature order from 11.0D base script:\n"
            f"{all_feature_names}"
        )

    ideal_11e, raw_11e, m3_11e = load_qpu_feature_matrix()

    X_val_rebuilt = X_ideal_all[val_idx]

    max_ideal_diff = float(
        np.max(
            np.abs(
                ideal_11e.to_numpy(dtype=float) - X_val_rebuilt
            )
        )
    )

    print()
    print("11.0E alignment audit:")
    print(
        f"  max |ideal_from_11E - rebuilt_ideal| = {max_ideal_diff:.3e}"
    )

    if max_ideal_diff > 1e-9:
        raise RuntimeError(
            "Ideal feature alignment failed. Refusing to mix incompatible runs."
        )

    y_train = y[train_idx]
    y_val = y[val_idx]

    summary_rows = []
    weight_rows = []
    contribution_rows = []
    prediction_rows = []

    for readout_name, selected_features in FEATURE_SETS.items():
        idx = feature_indexes(
            all_feature_names,
            selected_features,
        )

        X_train = X_ideal_all[np.ix_(train_idx, idx)]
        X_val_ideal = X_ideal_all[np.ix_(val_idx, idx)]

        X_val_raw = raw_11e[selected_features].to_numpy(dtype=float)
        X_val_m3 = m3_11e[selected_features].to_numpy(dtype=float)

        (
            selected_lambda,
            cv_folds,
            cv_summary,
        ) = base.select_lambda_training_only(
            X_train,
            y_train,
        )

        ridge, scaler, keep = base.fit_scaled_ridge(
            X_train,
            y_train,
            selected_lambda,
        )

        pred_ideal = base.predict_scaled_ridge(
            ridge,
            scaler,
            keep,
            X_val_ideal,
        )

        pred_raw = base.predict_scaled_ridge(
            ridge,
            scaler,
            keep,
            X_val_raw,
        )

        pred_m3 = base.predict_scaled_ridge(
            ridge,
            scaler,
            keep,
            X_val_m3,
        )

        ideal_metrics = prediction_metrics(y_val, pred_ideal)
        raw_metrics = prediction_metrics(y_val, pred_raw)
        m3_metrics = prediction_metrics(y_val, pred_m3)

        sensitivity_df, shot_proxy_total = training_readout_sensitivity(
            feature_names=selected_features,
            X_train=X_train,
            model=ridge,
            scaler=scaler,
            keep=keep,
            shots=SHOTS,
        )

        sensitivity_df.insert(
            0,
            "readout",
            readout_name,
        )

        weight_rows.extend(
            sensitivity_df.to_dict("records")
        )

        max_sensitivity = float(
            sensitivity_df[
                "raw_prediction_sensitivity_dy_dx"
            ].abs().max()
        )

        worst_feature = str(
            sensitivity_df.loc[
                sensitivity_df[
                    "raw_prediction_sensitivity_dy_dx"
                ].abs().idxmax(),
                "feature",
            ]
        )

        sensitivity_map = dict(
            zip(
                sensitivity_df["feature"],
                sensitivity_df[
                    "raw_prediction_sensitivity_dy_dx"
                ],
            )
        )

        for source_name, X_hw in [
            ("RAW", X_val_raw),
            ("M3", X_val_m3),
        ]:
            for j, feature in enumerate(selected_features):
                mean_feature_error = float(
                    np.mean(
                        X_hw[:, j] - X_val_ideal[:, j]
                    )
                )

                sensitivity = float(
                    sensitivity_map[feature]
                )

                contribution_rows.append({
                    "readout": readout_name,
                    "source": source_name,
                    "feature": feature,
                    "mean_feature_error": mean_feature_error,
                    "raw_prediction_sensitivity_dy_dx": sensitivity,
                    "mean_prediction_shift_claims":
                        sensitivity * mean_feature_error,
                })

        summary_rows.append({
            "readout": readout_name,
            "n_features": len(selected_features),
            "features": ",".join(selected_features),
            "selected_lambda_training_only":
                float(selected_lambda),
            "training_cv_rmse":
                float(cv_summary.iloc[0]["cv_rmse_mean"]),
            "training_cv_rmse_std":
                float(cv_summary.iloc[0]["cv_rmse_std"]),
            "ideal_2025_rmse": ideal_metrics["rmse"],
            "ideal_2025_mae": ideal_metrics["mae"],
            "ideal_2025_bias": ideal_metrics["bias"],
            "raw_1024_2025_rmse": raw_metrics["rmse"],
            "raw_1024_2025_mae": raw_metrics["mae"],
            "raw_1024_2025_bias": raw_metrics["bias"],
            "M3_1024_2025_rmse": m3_metrics["rmse"],
            "M3_1024_2025_mae": m3_metrics["mae"],
            "M3_1024_2025_bias": m3_metrics["bias"],
            "max_abs_raw_prediction_sensitivity":
                max_sensitivity,
            "worst_sensitivity_feature":
                worst_feature,
            "approx_total_forecast_shot_sd_proxy_1024":
                shot_proxy_total,
            "scientific_status":
                (
                    "pre-existing baseline"
                    if readout_name == "FULL_XZ8"
                    else
                    "post-hoc diagnostic motivated by 2025 hardware behavior"
                ),
        })

        for local_i, endpoint in enumerate(val_idx):
            prediction_rows.append({
                "readout": readout_name,
                "validation_offset": int(local_i),
                "endpoint_global": int(endpoint),
                "target": float(y_val[local_i]),
                "pred_ideal": float(pred_ideal[local_i]),
                "pred_RAW_1024": float(pred_raw[local_i]),
                "pred_M3_1024": float(pred_m3[local_i]),
            })

    summary = pd.DataFrame(summary_rows)
    weights = pd.DataFrame(weight_rows)
    contributions = pd.DataFrame(contribution_rows)
    predictions = pd.DataFrame(prediction_rows)

    summary_path = RESULTS / "11_0F_drop_X3_comparison.csv"
    weights_path = RESULTS / "11_0F_readout_sensitivity.csv"
    contributions_path = RESULTS / "11_0F_mean_error_contributions.csv"
    predictions_path = RESULTS / "11_0F_predictions.csv"

    summary.to_csv(summary_path, index=False)
    weights.to_csv(weights_path, index=False)
    contributions.to_csv(contributions_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    manifest = {
        "purpose":
            "Post-hoc mechanism diagnostic: remove X3 from H4 RWP_W1 XZ readout.",
        "qpu_calls": 0,
        "training_period": "2022-2024 only",
        "diagnostic_period": "2025",
        "test_period": "2026 frozen / unused",
        "shots_of_reused_qpu_data": SHOTS,
        "feature_sets": FEATURE_SETS,
        "important_caveat":
            (
                "DROP_X3 was motivated after observing 2025 hardware behavior. "
                "Its 2025 result is exploratory and not an unbiased final "
                "validation estimate."
            ),
        "alignment_max_abs_difference": max_ideal_diff,
        "outputs": [
            str(summary_path),
            str(weights_path),
            str(contributions_path),
            str(predictions_path),
        ],
    }

    with open(
        RESULTS / "11_0F_manifest.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            manifest,
            fp,
            indent=2,
        )

    print()
    print("=" * 120)
    print("11.0F RESULTS")
    print("=" * 120)
    print()
    print(
        summary[
            [
                "readout",
                "n_features",
                "selected_lambda_training_only",
                "training_cv_rmse",
                "ideal_2025_rmse",
                "raw_1024_2025_rmse",
                "raw_1024_2025_bias",
                "M3_1024_2025_rmse",
                "M3_1024_2025_bias",
                "max_abs_raw_prediction_sensitivity",
                "worst_sensitivity_feature",
                "approx_total_forecast_shot_sd_proxy_1024",
            ]
        ].to_string(index=False)
    )

    print()
    print("Mean prediction-shift contributions from hardware feature bias:")
    print(
        contributions.sort_values(
            "mean_prediction_shift_claims",
            key=lambda s: s.abs(),
            ascending=False,
        ).to_string(index=False)
    )

    print()
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {weights_path}")
    print(f"  {contributions_path}")
    print(f"  {predictions_path}")
    print("  results/11_0F_manifest.json")

    print()
    print("IMPORTANT:")
    print(
        "If DROP_X3 dramatically improves RAW/M3 RMSE, that confirms the "
        "mechanism. But because X3 was removed after inspecting 2025 hardware "
        "behavior, this remains a post-hoc diagnostic. Final feature-selection "
        "rules should be frozen from training-only robustness criteria before "
        "the still-frozen 2026 test."
    )


if __name__ == "__main__":
    main()
