"""
Week 7 - Step 7.4B.2
Forecasting sensitivity to CONT washout.

Question
--------
After discarding the first T_wash training samples, does forecasting become
insensitive to the arbitrary initial two-qubit memory state?

Modes
-----
0) ideal_unmeasured
1) direct_local_Y4_X5_selective
2) ancilla_joint_YX_nonselective
3) ancilla_joint_YX_selective

Washout candidates
------------------
T_wash in {0, 5, 10, 25, 50, 100}

Important
---------
The quantum reservoir is NOT reset at T_wash.
It evolves chronologically from t=0 in every case.
Washout means only that the first T_wash training feature/target pairs are
excluded from the classical Ridge readout training and chronological CV.

This test uses exact expectation-value features. It isolates initialization
and measurement-back-action effects; finite-shot noise is NOT added here.

For selective modes, common random numbers are used across the five different
initial memory states within each Monte-Carlo replicate. Therefore prediction
differences across initial states mainly reflect initialization dependence,
rather than unrelated random-number streams.

Dependency
----------
Keep these files beside this script:
    07_02b_memory_pair_isolation.py
    07_04b1_cont_washout_diagnostic.py
"""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

HERE = Path(__file__).resolve().parent
WASHOUT_SCRIPT = HERE / "07_04b1_cont_washout_diagnostic.py"

if not WASHOUT_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing {WASHOUT_SCRIPT.name}. Keep it beside this script."
    )

spec = importlib.util.spec_from_file_location("washout74b1", WASHOUT_SCRIPT)
w = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(w)

qrc = w.qrc
RESULTS = Path("results")

ALPHA = 0.75
SEED = 42
WASHOUTS = [0, 5, 10, 25, 50, 100]
N_MC = 20
MC_SEED_BASE = 74200
RIDGE_ALPHA = 100.0

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]

FAMILY = "XZinj_plus_YX45"
FEATURES = qrc.FAMILIES[FAMILY]


def fixed_alpha_cv(X, y, ridge_alpha):
    splitter = TimeSeriesSplit(n_splits=qrc.N_CV_SPLITS)
    scores = []
    for tr_idx, va_idx in splitter.split(X):
        pred = qrc.fit_predict_ridge(
            X[tr_idx], y[tr_idx], X[va_idx], ridge_alpha
        )
        scores.append(qrc.rmse(y[va_idx], pred))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def pairwise_prediction_rmse(predictions_by_init):
    names = list(predictions_by_init)
    vals = []
    for a, b in combinations(names, 2):
        pa = np.asarray(predictions_by_init[a], dtype=float)
        pb = np.asarray(predictions_by_init[b], dtype=float)
        vals.append(float(np.sqrt(np.mean((pa - pb) ** 2))))
    return float(max(vals)) if vals else 0.0


def deterministic_feature_trajectories(A_list, n_steps, mode):
    rho = {name: state.copy() for name, state in w.INITIAL_STATES.items()}
    features = {
        name: np.zeros((n_steps, len(FEATURES)), dtype=float)
        for name in rho
    }

    for t in range(n_steps):
        next_states = {}
        for name, rho_before in rho.items():
            rho_pre_measure, x = w.evolve_one_input(A_list[t], rho_before)
            features[name][t] = x

            if mode == "ideal_unmeasured":
                rho_post = rho_pre_measure
            elif mode == "ancilla_joint_YX_nonselective":
                rho_post = w.joint_nonselective_channel(rho_pre_measure)
            else:
                raise ValueError(mode)

            next_states[name] = rho_post
        rho = next_states

    return features


def selective_feature_trajectories(A_list, n_steps, projectors, rng):
    rho = {name: state.copy() for name, state in w.INITIAL_STATES.items()}
    features = {
        name: np.zeros((n_steps, len(FEATURES)), dtype=float)
        for name in rho
    }

    for t in range(n_steps):
        u = float(rng.random())
        next_states = {}

        for name, rho_before in rho.items():
            rho_pre_measure, x = w.evolve_one_input(A_list[t], rho_before)
            features[name][t] = x

            _, _, rho_post = w.sample_projective_outcome(
                rho_pre_measure, projectors, u
            )
            next_states[name] = rho_post

        rho = next_states

    return features


def evaluate_bundle(
    mode,
    features_by_init,
    y_train_full,
    y_val,
    n_train,
    n_val,
    mc_rep=None,
):
    rows = []
    summary_rows = []

    for T in WASHOUTS:
        y_train = y_train_full[T:]
        cv_by_init = {}
        val_by_init = {}
        predictions = {}

        for init_name, Xall in features_by_init.items():
            Xtr = Xall[T:n_train]
            Xva = Xall[n_train:n_train + n_val]

            cv_mean, cv_std = fixed_alpha_cv(Xtr, y_train, RIDGE_ALPHA)
            pred = qrc.fit_predict_ridge(
                Xtr, y_train, Xva, RIDGE_ALPHA
            )
            val_rmse = qrc.rmse(y_val, pred)

            cv_by_init[init_name] = cv_mean
            val_by_init[init_name] = val_rmse
            predictions[init_name] = pred

            rows.append({
                "mode": mode,
                "mc_rep": np.nan if mc_rep is None else mc_rep,
                "washout": T,
                "initial_state": init_name,
                "ridge_alpha": RIDGE_ALPHA,
                "cv_rmse": cv_mean,
                "cv_rmse_fold_std": cv_std,
                "validation_rmse": val_rmse,
            })

        cv_vals = np.asarray(list(cv_by_init.values()), dtype=float)
        val_vals = np.asarray(list(val_by_init.values()), dtype=float)

        summary_rows.append({
            "mode": mode,
            "mc_rep": np.nan if mc_rep is None else mc_rep,
            "washout": T,
            "mean_cv_rmse_across_initializations": float(np.mean(cv_vals)),
            "std_cv_rmse_across_initializations": float(np.std(cv_vals, ddof=1)),
            "range_cv_rmse_across_initializations": float(np.ptp(cv_vals)),
            "mean_validation_rmse_across_initializations": float(np.mean(val_vals)),
            "std_validation_rmse_across_initializations": float(np.std(val_vals, ddof=1)),
            "range_validation_rmse_across_initializations": float(np.ptp(val_vals)),
            "max_pairwise_validation_prediction_rmse": (
                pairwise_prediction_rmse(predictions)
            ),
        })

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 124)
    print("WEEK 7 - STEP 7.4B.2")
    print("FORECASTING SENSITIVITY TO CONT WASHOUT")
    print("=" * 124)
    print()
    print("Frozen:")
    print(f"  F4 | alpha={ALPHA} | seed={SEED}")
    print(f"  feature family={FAMILY}")
    print(f"  Ridge alpha fixed at {RIDGE_ALPHA}")
    print(f"  washout candidates={WASHOUTS}")
    print(f"  selective forecasting MC trajectories={N_MC}")
    print("  finite-shot noise=OFF")
    print()
    print("Washout removes early samples from Ridge training/CV only;")
    print("the quantum CONT trajectory itself still starts at t=0.")
    print()

    work, train, val, cols = qrc.load_data()
    n_train = len(train)
    n_val = len(val)
    n_steps = n_train + n_val

    y_train_full = train[cols["target"]].to_numpy(dtype=float)
    y_val = val[cols["target"]].to_numpy(dtype=float)

    angles_all = qrc.make_input_angles(work, cols)
    U, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, _ = qrc.build_input_channels(U, angles_all)

    print(f"QRC unitarity error = {unitary_err:.3e}")
    print(f"Train={n_train}, validation={n_val}, total CONT steps={n_steps}")
    print()

    detail_parts = []
    case_summary_parts = []

    for mode in [
        "ideal_unmeasured",
        "ancilla_joint_YX_nonselective",
    ]:
        print(f"Building deterministic features: {mode} ...")
        feats = deterministic_feature_trajectories(A_list, n_steps, mode)
        detail, case = evaluate_bundle(
            mode, feats, y_train_full, y_val, n_train, n_val, mc_rep=None
        )
        detail_parts.append(detail)
        case_summary_parts.append(case)

    print()
    print("Running selective direct-local Y4/X5 forecasting trajectories ...")
    for rep in range(N_MC):
        rng = np.random.default_rng(MC_SEED_BASE + rep * 1009)
        feats = selective_feature_trajectories(
            A_list, n_steps, w.LOCAL_PROJECTORS, rng
        )
        detail, case = evaluate_bundle(
            "direct_local_Y4_X5_selective",
            feats,
            y_train_full,
            y_val,
            n_train,
            n_val,
            mc_rep=rep,
        )
        detail_parts.append(detail)
        case_summary_parts.append(case)

    print("Running selective ancilla joint-YX forecasting trajectories ...")
    for rep in range(N_MC):
        rng = np.random.default_rng(MC_SEED_BASE + rep * 1009)
        feats = selective_feature_trajectories(
            A_list, n_steps, w.JOINT_PROJECTORS, rng
        )
        detail, case = evaluate_bundle(
            "ancilla_joint_YX_selective",
            feats,
            y_train_full,
            y_val,
            n_train,
            n_val,
            mc_rep=rep,
        )
        detail_parts.append(detail)
        case_summary_parts.append(case)

    detail_df = pd.concat(detail_parts, ignore_index=True)
    case_df = pd.concat(case_summary_parts, ignore_index=True)

    detail_df.to_csv(
        RESULTS / "07_04b2_washout_forecast_by_initialization.csv",
        index=False,
    )
    case_df.to_csv(
        RESULTS / "07_04b2_washout_forecast_case_summary.csv",
        index=False,
    )

    global_rows = []

    for mode in [
        "ideal_unmeasured",
        "ancilla_joint_YX_nonselective",
        "direct_local_Y4_X5_selective",
        "ancilla_joint_YX_selective",
    ]:
        for T in WASHOUTS:
            sub = case_df[
                (case_df["mode"] == mode)
                & (case_df["washout"] == T)
            ]

            global_rows.append({
                "mode": mode,
                "washout": T,
                "n_cases": len(sub),
                "mean_cv_rmse": float(
                    sub["mean_cv_rmse_across_initializations"].mean()
                ),
                "mean_validation_rmse": float(
                    sub["mean_validation_rmse_across_initializations"].mean()
                ),
                "median_init_range_validation_rmse": float(
                    sub["range_validation_rmse_across_initializations"].median()
                ),
                "p90_init_range_validation_rmse": float(
                    np.quantile(
                        sub["range_validation_rmse_across_initializations"], 0.90
                    )
                ),
                "median_max_pairwise_prediction_rmse": float(
                    sub["max_pairwise_validation_prediction_rmse"].median()
                ),
                "p90_max_pairwise_prediction_rmse": float(
                    np.quantile(
                        sub["max_pairwise_validation_prediction_rmse"], 0.90
                    )
                ),
            })

    global_df = pd.DataFrame(global_rows)

    global_df["delta_mean_cv_vs_T0"] = np.nan
    global_df["delta_mean_validation_vs_T0"] = np.nan

    for mode in global_df["mode"].unique():
        mask = global_df["mode"] == mode
        base = global_df[mask & (global_df["washout"] == 0)].iloc[0]

        global_df.loc[mask, "delta_mean_cv_vs_T0"] = (
            global_df.loc[mask, "mean_cv_rmse"] - base["mean_cv_rmse"]
        )
        global_df.loc[mask, "delta_mean_validation_vs_T0"] = (
            global_df.loc[mask, "mean_validation_rmse"]
            - base["mean_validation_rmse"]
        )

    global_df.to_csv(
        RESULTS / "07_04b2_washout_forecast_global_summary.csv",
        index=False,
    )

    for mode in global_df["mode"].unique():
        print()
        print("-" * 124)
        print(f"MODE = {mode}")
        print("-" * 124)

        cols_print = [
            "washout",
            "mean_cv_rmse",
            "mean_validation_rmse",
            "delta_mean_cv_vs_T0",
            "median_init_range_validation_rmse",
            "p90_init_range_validation_rmse",
            "median_max_pairwise_prediction_rmse",
            "p90_max_pairwise_prediction_rmse",
        ]

        print(
            global_df[global_df["mode"] == mode][cols_print]
            .to_string(index=False)
        )

    with open(
        RESULTS / "07_04b2_washout_forecast_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("WEEK 7 STEP 7.4B.2 - WASHOUT FORECASTING TEST\n")
        f.write("=" * 88 + "\n\n")
        f.write(
            f"alpha={ALPHA}, seed={SEED}, family={FAMILY}, "
            f"ridge_alpha={RIDGE_ALPHA}\n"
        )
        f.write(f"washouts={WASHOUTS}, selective MC={N_MC}\n\n")
        f.write(global_df.to_string(index=False))
        f.write("\n")

    print()
    print("=" * 124)
    print("Saved:")
    print("  results/07_04b2_washout_forecast_by_initialization.csv")
    print("  results/07_04b2_washout_forecast_case_summary.csv")
    print("  results/07_04b2_washout_forecast_global_summary.csv")
    print("  results/07_04b2_washout_forecast_summary.txt")
    print()
    print("Do NOT freeze T_wash automatically.")
    print("Interpret prediction accuracy AND initialization sensitivity together.")


if __name__ == "__main__":
    main()
