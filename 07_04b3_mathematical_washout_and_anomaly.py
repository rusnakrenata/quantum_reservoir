"""
Week 7 - Step 7.4B.3
Mathematical washout + empirical comparison + T_wash=100 anomaly audit.

Uses the notation from Part I, Unit 3:
    D_t <= kappa^t D_0
    tau_mem = -1 / ln(kappa)
    T_w >= ln(epsilon) / ln(kappa)
        = tau_mem * ln(1/epsilon)

For an absolute trace-distance threshold epsilon_abs when D_0 != 1:
    T_w >= ln(epsilon_abs / D_0) / ln(kappa)

Scope
-----
A single deterministic contraction factor kappa is meaningful for:
    0) ideal/unmeasured CONT
    2) ancilla joint YX NON-SELECTIVE CONT

It is NOT assigned to the selective conditioned trajectories because those
updates are stochastic and outcome-conditioned. Their washout remains the
Monte-Carlo result from Step 7.4B.1.

The script also diagnoses the suspicious Step-7.4B.2 result in which the
non-selective case showed larger initialization-sensitive predictions at
T_wash=100 even though the state trajectories had converged earlier.

Dependencies
------------
Keep beside this script:
    07_04b1_cont_washout_diagnostic.py
    07_04b2_cont_washout_forecasting.py
"""

from __future__ import annotations

import importlib.util
import inspect
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")

B1 = HERE / "07_04b1_cont_washout_diagnostic.py"
B2 = HERE / "07_04b2_cont_washout_forecasting.py"

for p in [B1, B2]:
    if not p.exists():
        raise FileNotFoundError(f"Missing {p.name}. Keep it beside this script.")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


w = load_module(B1, "washout74b1")
f = load_module(B2, "washout74b2")
qrc = w.qrc

ALPHA = 0.75
SEED = 42
EPSILONS = [0.1, 0.05, 0.01, 0.001]
FIT_MAX_T = 100
NUMERICAL_FLOOR = 1e-12

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]


def fit_effective_kappa(df):
    d = df[["t", "max_pairwise_trace_distance"]].copy()
    D0 = float(d.iloc[0]["max_pairwise_trace_distance"])

    d = d[
        (d["t"] <= FIT_MAX_T)
        & (d["max_pairwise_trace_distance"] > NUMERICAL_FLOOR)
    ].copy()

    t = d["t"].to_numpy(dtype=float)
    D = d["max_pairwise_trace_distance"].to_numpy(dtype=float)

    mask = t > 0
    t = t[mask]
    y = np.log(D[mask] / D0)

    if len(t) < 2:
        return {
            "D0": D0,
            "kappa_eff": np.nan,
            "tau_mem": np.nan,
            "r2_log_fit": np.nan,
        }

    slope = float(np.dot(t, y) / np.dot(t, t))
    kappa = float(np.exp(slope))

    yhat = slope * t
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    tau = np.inf if not (0 < kappa < 1) else float(-1.0 / np.log(kappa))

    return {
        "D0": D0,
        "kappa_eff": kappa,
        "tau_mem": tau,
        "r2_log_fit": r2,
    }


def empirical_washout(df, epsilon):
    return w.stable_threshold_crossing(
        df["max_pairwise_trace_distance"],
        epsilon,
    )


def predicted_washout_absolute(D0, kappa, epsilon_abs):
    if not (0 < kappa < 1):
        return None
    if epsilon_abs >= D0:
        return 0
    val = np.log(epsilon_abs / D0) / np.log(kappa)
    return int(np.ceil(val))


def stepwise_kappa_stats(df):
    D = df["max_pairwise_trace_distance"].to_numpy(dtype=float)
    ratios = []

    for t in range(1, len(D)):
        if D[t - 1] > NUMERICAL_FLOOR and D[t] > NUMERICAL_FLOOR:
            ratios.append(D[t] / D[t - 1])

    ratios = np.asarray(ratios, dtype=float)

    if len(ratios) == 0:
        return {
            "median_kappa_t": np.nan,
            "p90_kappa_t": np.nan,
            "max_kappa_t": np.nan,
            "fraction_kappa_t_lt_1": np.nan,
        }

    return {
        "median_kappa_t": float(np.median(ratios)),
        "p90_kappa_t": float(np.quantile(ratios, 0.90)),
        "max_kappa_t": float(np.max(ratios)),
        "fraction_kappa_t_lt_1": float(np.mean(ratios < 1.0)),
    }


def max_cross_init_feature_diff(features_by_init, start, stop):
    names = list(features_by_init)
    max_diff = 0.0
    for a, b in combinations(names, 2):
        Xa = np.asarray(features_by_init[a][start:stop], dtype=float)
        Xb = np.asarray(features_by_init[b][start:stop], dtype=float)
        max_diff = max(max_diff, float(np.max(np.abs(Xa - Xb))))
    return max_diff


def min_column_std(features_by_init, start, stop):
    rows = []
    for name, X in features_by_init.items():
        stds = np.std(np.asarray(X[start:stop], dtype=float), axis=0, ddof=0)
        rows.append({
            "initial_state": name,
            "min_feature_std": float(np.min(stds)),
            "max_feature_std": float(np.max(stds)),
        })
    return pd.DataFrame(rows)


def validation_predictions(features_by_init, y_train, y_val, n_train, n_val, T, rounding=None):
    predictions = {}
    val_rmse = {}

    for name, Xall in features_by_init.items():
        X = np.asarray(Xall, dtype=float).copy()

        if rounding is not None:
            X = np.round(X, rounding)

        Xtr = X[T:n_train]
        Xva = X[n_train:n_train + n_val]
        ytr = y_train[T:]

        pred = qrc.fit_predict_ridge(
            Xtr,
            ytr,
            Xva,
            f.RIDGE_ALPHA,
        )

        predictions[name] = pred
        val_rmse[name] = qrc.rmse(y_val, pred)

    max_pair_pred_rmse = f.pairwise_prediction_rmse(predictions)
    val_range = float(np.ptp(np.asarray(list(val_rmse.values()), dtype=float)))

    return max_pair_pred_rmse, val_range, val_rmse


def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 118)
    print("WEEK 7 - STEP 7.4B.3")
    print("MATHEMATICAL WASHOUT + EMPIRICAL COMPARISON + T_WASH=100 ANOMALY AUDIT")
    print("=" * 118)
    print()
    print("Foundations notation:")
    print("  D_t <= kappa^t D_0")
    print("  tau_mem = -1 / ln(kappa)")
    print("  T_w >= ln(epsilon) / ln(kappa)")
    print()
    print("For our absolute D threshold, the script uses")
    print("  T_w >= ln(epsilon / D_0) / ln(kappa)")
    print()
    print("Selective conditioned measurements are NOT assigned one kappa;")
    print("their washout remains the Monte-Carlo result from Step 7.4B.1.")
    print()

    work, train, val, cols = qrc.load_data()
    n_train = len(train)
    n_val = len(val)
    n_steps = n_train + n_val

    y_train = train[cols["target"]].to_numpy(dtype=float)
    y_val = val[cols["target"]].to_numpy(dtype=float)

    angles_all = qrc.make_input_angles(work, cols)
    U, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, _ = qrc.build_input_channels(U, angles_all)

    print(f"QRC unitarity error = {unitary_err:.3e}")
    print()

    ideal_df = w.run_ideal_unmeasured(A_list, n_steps)
    ns_df = w.run_joint_nonselective(A_list, n_steps)

    math_rows = []

    for mode, df in [
        ("ideal_unmeasured", ideal_df),
        ("ancilla_joint_YX_nonselective", ns_df),
    ]:
        fit = fit_effective_kappa(df)
        step = stepwise_kappa_stats(df)

        print("-" * 118)
        print(f"MODE = {mode}")
        print("-" * 118)
        print(f"D_0       = {fit['D0']:.6f}")
        print(f"kappa_eff = {fit['kappa_eff']:.9f}")
        print(f"tau_mem   = {fit['tau_mem']:.6f}")
        print(f"log-fit R2= {fit['r2_log_fit']:.6f}")
        print(
            f"stepwise kappa_t: median={step['median_kappa_t']:.6f}, "
            f"p90={step['p90_kappa_t']:.6f}, max={step['max_kappa_t']:.6f}, "
            f"fraction<1={step['fraction_kappa_t_lt_1']:.3f}"
        )
        print()
        print("epsilon | predicted T_w | empirical T_w")
        print("-" * 44)

        for eps in EPSILONS:
            pred = predicted_washout_absolute(
                fit["D0"],
                fit["kappa_eff"],
                eps,
            )
            emp = empirical_washout(df, eps)

            print(f"{eps:7g} | {str(pred):13s} | {str(emp):13s}")

            math_rows.append({
                "mode": mode,
                "D0": fit["D0"],
                "kappa_eff": fit["kappa_eff"],
                "tau_mem": fit["tau_mem"],
                "r2_log_fit": fit["r2_log_fit"],
                "median_kappa_t": step["median_kappa_t"],
                "p90_kappa_t": step["p90_kappa_t"],
                "max_kappa_t": step["max_kappa_t"],
                "fraction_kappa_t_lt_1": step["fraction_kappa_t_lt_1"],
                "epsilon": eps,
                "predicted_Tw": np.nan if pred is None else pred,
                "empirical_Tw": np.nan if emp is None else emp,
            })

        print()

    math_df = pd.DataFrame(math_rows)
    math_df.to_csv(
        RESULTS / "07_04b3_mathematical_vs_empirical_washout.csv",
        index=False,
    )

    print("=" * 118)
    print("T_WASH=100 ANOMALY AUDIT: ANCILLA JOINT YX NON-SELECTIVE")
    print("=" * 118)

    features_ns = f.deterministic_feature_trajectories(
        A_list,
        n_steps,
        "ancilla_joint_YX_nonselective",
    )

    checkpoints = [0, 5, 10, 25, 50, 54, 55, 100, n_train - 1, n_train, n_steps - 1]
    checkpoint_rows = []

    names = list(features_ns)

    for t in checkpoints:
        vals = np.stack([features_ns[name][t] for name in names], axis=0)
        max_abs = float(np.max(np.ptp(vals, axis=0)))
        checkpoint_rows.append({
            "t": t,
            "max_abs_feature_difference_across_initial_states": max_abs,
        })

    checkpoint_df = pd.DataFrame(checkpoint_rows)
    print()
    print("Feature differences at important time points:")
    print(checkpoint_df.to_string(index=False))

    raw_train_diff_T100 = max_cross_init_feature_diff(
        features_ns, 100, n_train
    )
    raw_val_diff = max_cross_init_feature_diff(
        features_ns, n_train, n_train + n_val
    )

    print()
    print(f"Max raw feature difference, train t>=100 = {raw_train_diff_T100:.6e}")
    print(f"Max raw feature difference, validation   = {raw_val_diff:.6e}")

    std_df = min_column_std(features_ns, 100, n_train)
    print()
    print("Feature standard deviations after T=100:")
    print(std_df.to_string(index=False))

    pred_rmse_raw, val_range_raw, _ = validation_predictions(
        features_ns, y_train, y_val, n_train, n_val, T=100, rounding=None
    )

    pred_rmse_round12, val_range_round12, _ = validation_predictions(
        features_ns, y_train, y_val, n_train, n_val, T=100, rounding=12
    )
    pred_rmse_round10, val_range_round10, _ = validation_predictions(
        features_ns, y_train, y_val, n_train, n_val, T=100, rounding=10
    )

    anomaly_df = pd.DataFrame([
        {
            "case": "raw",
            "max_pairwise_validation_prediction_rmse": pred_rmse_raw,
            "validation_rmse_range": val_range_raw,
        },
        {
            "case": "features_rounded_12_decimals",
            "max_pairwise_validation_prediction_rmse": pred_rmse_round12,
            "validation_rmse_range": val_range_round12,
        },
        {
            "case": "features_rounded_10_decimals",
            "max_pairwise_validation_prediction_rmse": pred_rmse_round10,
            "validation_rmse_range": val_range_round10,
        },
    ])

    print()
    print("Prediction anomaly reproduction:")
    print(anomaly_df.to_string(index=False))

    try:
        ridge_source = inspect.getsource(qrc.fit_predict_ridge)
    except Exception as exc:
        ridge_source = f"Could not inspect fit_predict_ridge: {exc}"

    with open(
        RESULTS / "07_04b3_fit_predict_ridge_source.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(ridge_source)

    checkpoint_df.to_csv(
        RESULTS / "07_04b3_nonselective_feature_convergence_checkpoints.csv",
        index=False,
    )
    std_df.to_csv(
        RESULTS / "07_04b3_nonselective_feature_std_T100.csv",
        index=False,
    )
    anomaly_df.to_csv(
        RESULTS / "07_04b3_T100_anomaly_rounding_audit.csv",
        index=False,
    )

    print()
    print("Saved:")
    print("  results/07_04b3_mathematical_vs_empirical_washout.csv")
    print("  results/07_04b3_nonselective_feature_convergence_checkpoints.csv")
    print("  results/07_04b3_nonselective_feature_std_T100.csv")
    print("  results/07_04b3_T100_anomaly_rounding_audit.csv")
    print("  results/07_04b3_fit_predict_ridge_source.txt")
    print()
    print("Interpret kappa/tau_mem/T_w first, then resolve the T=100 anomaly.")


if __name__ == "__main__":
    main()
