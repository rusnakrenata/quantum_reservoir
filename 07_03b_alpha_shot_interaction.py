"""
Week 7 - Step 7.3B
Input-strength (alpha) x finite-shot interaction study for the QRC.

Purpose
-------
Step 7.3 showed an unexpectedly persistent finite-shot degradation even at
8192 shots. This diagnostic tests whether the frozen ideal-simulator input
strength alpha=0.005 produces reservoir features whose useful temporal
variation is small relative to finite-shot estimator noise.

We vary only the continuous-input scale alpha while keeping the remaining QRC
architecture fixed:
    F4, 4 injection + 2 memory qubits, softsign policy,
    hx=0.5, dt=0.8, r=2,
    reservoir seeds=[42,101,202,505,707],
    CONT and RWP W in {1,2,5,7,14,21,28}.

Important encoding detail
-------------------------
alpha scales only the continuous C_t and P_t injection angles. The weekday
angle and holiday angle remain unchanged exactly as in the frozen F4 encoder.

For each alpha / seed / protocol / observable family we compute:
1) exact-feature chronological CV performance;
2) feature temporal variation (signal scale);
3) expected finite-shot standard deviation;
4) signal-to-shot-noise ratio (SSNR);
5) finite-shot CV degradation at representative shot budgets.

For a time-varying Pauli expectation m_t=<P>_t:

    signal_std = Std_t(m_t)

    shot_variance_t = (1 - m_t^2) / N

A single family-level shot-noise scale is defined using the RMS over time:

    shot_sigma_rms = sqrt( mean_t[(1 - m_t^2)/N] )

and the feature-level signal-to-shot-noise ratio is

    SSNR = signal_std / shot_sigma_rms.

Interpretation:
    SSNR < 1  -> shot fluctuations are larger than temporal feature variation
    SSNR ~ 1  -> comparable scales
    SSNR > 1  -> temporal feature variation dominates shot fluctuations

This is still an estimator-level ideal-projective-shot study. It does NOT yet
simulate basis-change gates, ancillas, readout error, gate noise, covariance
from commuting-group measurement, or measurement back-action.

Dependency
----------
Keep 07_02b_memory_pair_isolation.py beside this file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "07_02b_memory_pair_isolation.py"
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing {BASE_SCRIPT.name}. Keep it beside this Step-7.3B script."
    )

spec = importlib.util.spec_from_file_location("qrc_step_72b", BASE_SCRIPT)
qrc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qrc)

RESULTS = Path("results")

# Requested alpha landscape: includes the original alpha=0.005 plus much
# stronger injection values 0.5 and 1.0 to expose possible saturation/overdrive.
ALPHAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]

# Representative shot budgets. Step 7.3 already studied the denser shot grid.
SHOT_BUDGETS = [128, 512, 2048, 8192]
N_SHOT_REPEATS = 20
SHOT_SEED_BASE = 73111

# Keep the focused candidate landscape from Step 7.2B.
FAMILIES = {
    "XZ_injection": qrc.FAMILIES["XZ_injection"],
    "XZ_all": qrc.FAMILIES["XZ_all"],
    "XZinj_plus_YX45": qrc.FAMILIES["XZinj_plus_YX45"],
    "XZinj_plus_XZ45": qrc.FAMILIES["XZinj_plus_XZ45"],
    "XZinj_plus_YX45_XZ45": qrc.FAMILIES["XZinj_plus_YX45_XZ45"],
}

PROTOCOL_SPECS = [("CONT", None)] + [
    (f"RWP_W{W}", W) for W in qrc.WINDOWS
]


def shot_sample_expectations(X_exact: np.ndarray, shots: int, rng) -> np.ndarray:
    X = np.clip(np.asarray(X_exact, dtype=float), -1.0, 1.0)
    p_plus = np.clip((1.0 + X) / 2.0, 0.0, 1.0)
    n_plus = rng.binomial(int(shots), p_plus)
    return 2.0 * n_plus / float(shots) - 1.0


def fixed_alpha_cv(X: np.ndarray, y: np.ndarray, ridge_alpha: float):
    splitter = TimeSeriesSplit(n_splits=qrc.N_CV_SPLITS)
    scores = []
    for tr_idx, va_idx in splitter.split(X):
        pred = qrc.fit_predict_ridge(
            X[tr_idx], y[tr_idx], X[va_idx], ridge_alpha
        )
        scores.append(qrc.rmse(y[va_idx], pred))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def protocol_banks(
    A_list, S_list, cont_bank_all,
    n_train, n_val, y_train_full, y_val,
    protocol, W,
):
    if protocol == "CONT":
        bank_train = cont_bank_all.iloc[:n_train].reset_index(drop=True)
        bank_val = cont_bank_all.iloc[n_train:n_train+n_val].reset_index(drop=True)
        y_train = y_train_full
        w_numeric = np.nan
    else:
        train_end = np.arange(W - 1, n_train, dtype=int)
        val_end = np.arange(n_train, n_train + n_val, dtype=int)
        y_train = y_train_full[W - 1:]
        bank_train, _, _ = qrc.build_window_bank(A_list, S_list, train_end, W)
        bank_val, _, _ = qrc.build_window_bank(A_list, S_list, val_end, W)
        w_numeric = W

    assert len(bank_train) == len(y_train)
    assert len(bank_val) == len(y_val)
    return bank_train, bank_val, y_train, w_numeric


def feature_signal_stats(X: np.ndarray, shots: int | None = None):
    """Return per-feature temporal signal and, if requested, shot-noise scale."""
    X = np.clip(np.asarray(X, dtype=float), -1.0, 1.0)
    signal_std = np.std(X, axis=0, ddof=1)
    mean_abs = np.mean(np.abs(X), axis=0)
    mean_value = np.mean(X, axis=0)

    out = {
        "signal_std": signal_std,
        "mean_abs_expectation": mean_abs,
        "mean_expectation": mean_value,
    }

    if shots is not None:
        # RMS of the point-wise estimator standard deviations:
        # sqrt(mean_t[(1-m_t^2)/N]).
        shot_sigma_rms = np.sqrt(
            np.mean(np.maximum(0.0, 1.0 - X**2), axis=0) / float(shots)
        )
        ssnr = np.divide(
            signal_std,
            shot_sigma_rms,
            out=np.full_like(signal_std, np.inf),
            where=shot_sigma_rms > 0,
        )
        out["shot_sigma_rms"] = shot_sigma_rms
        out["ssnr"] = ssnr

    return out


def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 124)
    print("WEEK 7 - STEP 7.3B")
    print("INPUT-STRENGTH alpha x FINITE-SHOT INTERACTION STUDY")
    print("=" * 124)
    print()
    print("Frozen except alpha:")
    print("  F4 | 4 injection + 2 memory | softsign policy")
    print(f"  alpha grid={ALPHAS}")
    print(f"  hx={qrc.HX} | dt={qrc.DT} | r={qrc.TROTTER_R}")
    print(f"  reservoir seeds={qrc.SEEDS}")
    print(f"  protocols=CONT + RWP {qrc.WINDOWS}")
    print(f"  shot budgets={SHOT_BUDGETS}")
    print(f"  shot Monte-Carlo repeats={N_SHOT_REPEATS}")
    print("  2025 diagnostic only; 2026 untouched")
    print()
    print("Signal-to-shot-noise diagnostic:")
    print("  signal_std = Std_t(<P>_t)")
    print("  shot_sigma_rms = sqrt(mean_t[(1-<P>_t^2)/N])")
    print("  SSNR = signal_std / shot_sigma_rms")
    print("  SSNR<1: shot noise larger than temporal feature variation")
    print()

    work, train, val, cols = qrc.load_data()
    n_train = len(train)
    n_val = len(val)
    y_train_full = train[cols["target"]].to_numpy(dtype=float)
    y_val = val[cols["target"]].to_numpy(dtype=float)

    exact_rows = []
    feature_rows = []
    noisy_rows = []

    for alpha in ALPHAS:
        print("#" * 124)
        print(f"ENCODING ALPHA = {alpha}")
        print("#" * 124)

        # qrc.make_input_angles reads qrc.ALPHA dynamically from its module.
        qrc.ALPHA = float(alpha)
        angles_all = qrc.make_input_angles(work, cols)

        # Helpful direct encoding-amplitude diagnostics for continuous channels.
        c_amp = np.std(angles_all[:n_train, 0], ddof=1)
        p_amp = np.std(angles_all[:n_train, 2], ddof=1)
        print(
            f"Train angle std: q0(C)={c_amp:.6f}, q2(P)={p_amp:.6f} "
            f"(weekday/holiday not alpha-scaled)"
        )

        for seed in qrc.SEEDS:
            print(f"  seed={seed}: building channels ...")
            U, unitary_err = qrc.build_trotter_unitary(seed)
            if unitary_err > 1e-10:
                raise RuntimeError(f"Unexpected unitarity error {unitary_err}")

            A_list, S_list = qrc.build_input_channels(U, angles_all)
            cont_bank_all, _, _ = qrc.build_cont_bank(A_list)

            for protocol, W in PROTOCOL_SPECS:
                bank_train, bank_val, y_train, w_numeric = protocol_banks(
                    A_list, S_list, cont_bank_all,
                    n_train, n_val, y_train_full, y_val,
                    protocol, W,
                )

                for family_name, feature_names in FAMILIES.items():
                    X_train_exact = bank_train[feature_names].to_numpy(dtype=float)
                    X_val_exact = bank_val[feature_names].to_numpy(dtype=float)

                    # For each alpha we allow the exact model to select its Ridge
                    # regularization, because changing alpha changes feature geometry.
                    best_exact, _ = qrc.chronological_cv(X_train_exact, y_train)
                    ridge_alpha = float(best_exact["ridge_alpha"])
                    exact_cv = float(best_exact["cv_rmse_mean"])
                    pred_exact = qrc.fit_predict_ridge(
                        X_train_exact, y_train, X_val_exact, ridge_alpha
                    )
                    exact_val = qrc.rmse(y_val, pred_exact)

                    exact_rows.append({
                        "encoding_alpha": alpha,
                        "seed": seed,
                        "protocol": protocol,
                        "W": w_numeric,
                        "family": family_name,
                        "n_features": len(feature_names),
                        "ridge_alpha_exact": ridge_alpha,
                        "exact_cv_rmse": exact_cv,
                        "exact_validation_rmse": exact_val,
                        "q0_C_angle_std_train": c_amp,
                        "q2_P_angle_std_train": p_amp,
                    })

                    # Feature-level signal statistics use TRAIN only.
                    base_stats = feature_signal_stats(X_train_exact)
                    for j, feat in enumerate(feature_names):
                        for shots in SHOT_BUDGETS:
                            sstats = feature_signal_stats(X_train_exact[:, [j]], shots)
                            feature_rows.append({
                                "encoding_alpha": alpha,
                                "seed": seed,
                                "protocol": protocol,
                                "W": w_numeric,
                                "family": family_name,
                                "feature": feat,
                                "shots": shots,
                                "signal_std": float(base_stats["signal_std"][j]),
                                "mean_expectation": float(base_stats["mean_expectation"][j]),
                                "mean_abs_expectation": float(base_stats["mean_abs_expectation"][j]),
                                "shot_sigma_rms": float(sstats["shot_sigma_rms"][0]),
                                "ssnr": float(sstats["ssnr"][0]),
                            })

                    # Freeze the exact-alpha-specific Ridge optimum while adding
                    # finite-shot noise, isolating measurement degradation.
                    for shots in SHOT_BUDGETS:
                        for rep in range(N_SHOT_REPEATS):
                            name_code = sum((i + 1) * ord(c) for i, c in enumerate(family_name))
                            protocol_code = sum((i + 1) * ord(c) for i, c in enumerate(protocol))
                            alpha_code = int(round(alpha * 1_000_000))
                            rng_seed = (
                                SHOT_SEED_BASE
                                + seed * 1_000_003
                                + shots * 101
                                + rep * 10_007
                                + name_code * 17
                                + protocol_code * 19
                                + alpha_code * 23
                            ) % (2**63 - 1)
                            rng = np.random.default_rng(rng_seed)

                            Xtr_noisy = shot_sample_expectations(
                                X_train_exact, shots, rng
                            )
                            Xva_noisy = shot_sample_expectations(
                                X_val_exact, shots, rng
                            )

                            noisy_cv, noisy_cv_fold_std = fixed_alpha_cv(
                                Xtr_noisy, y_train, ridge_alpha
                            )
                            pred_noisy = qrc.fit_predict_ridge(
                                Xtr_noisy, y_train, Xva_noisy, ridge_alpha
                            )
                            noisy_val = qrc.rmse(y_val, pred_noisy)

                            noisy_rows.append({
                                "encoding_alpha": alpha,
                                "seed": seed,
                                "protocol": protocol,
                                "W": w_numeric,
                                "family": family_name,
                                "n_features": len(feature_names),
                                "shots": shots,
                                "repeat": rep,
                                "ridge_alpha_exact": ridge_alpha,
                                "exact_cv_rmse": exact_cv,
                                "noisy_cv_rmse": noisy_cv,
                                "noisy_cv_fold_std": noisy_cv_fold_std,
                                "delta_cv_rmse": noisy_cv - exact_cv,
                                "relative_cv_degradation_pct":
                                    100.0 * (noisy_cv - exact_cv) / exact_cv,
                                "exact_validation_rmse": exact_val,
                                "noisy_validation_rmse": noisy_val,
                                "delta_validation_rmse": noisy_val - exact_val,
                            })

    exact_df = pd.DataFrame(exact_rows)
    feat_df = pd.DataFrame(feature_rows)
    noisy_df = pd.DataFrame(noisy_rows)

    exact_df.to_csv(RESULTS / "07_03b_alpha_exact_reference.csv", index=False)
    feat_df.to_csv(RESULTS / "07_03b_alpha_feature_ssnr_raw.csv", index=False)
    noisy_df.to_csv(RESULTS / "07_03b_alpha_shot_noise_raw.csv", index=False)

    # ------------------------------------------------------------------
    # Aggregate exact performance by alpha and family.
    # ------------------------------------------------------------------
    exact_global = (
        exact_df.groupby(["encoding_alpha", "family"], as_index=False)
        .agg(
            mean_exact_cv_rmse=("exact_cv_rmse", "mean"),
            std_exact_cv_rmse=("exact_cv_rmse", "std"),
            mean_exact_validation_rmse=("exact_validation_rmse", "mean"),
        )
        .sort_values(["encoding_alpha", "mean_exact_cv_rmse"])
    )
    exact_global.to_csv(
        RESULTS / "07_03b_alpha_exact_global_summary.csv", index=False
    )

    # Feature SSNR aggregation. Median is useful because observables can have
    # very different raw amplitudes and a few near-deterministic features can
    # create very large SSNR values.
    ssnr_global = (
        feat_df.groupby(["encoding_alpha", "family", "shots"], as_index=False)
        .agg(
            median_feature_signal_std=("signal_std", "median"),
            mean_feature_signal_std=("signal_std", "mean"),
            median_shot_sigma_rms=("shot_sigma_rms", "median"),
            mean_shot_sigma_rms=("shot_sigma_rms", "mean"),
            median_feature_ssnr=("ssnr", "median"),
            mean_feature_ssnr=("ssnr", "mean"),
            fraction_features_ssnr_gt_1=("ssnr", lambda s: float(np.mean(np.asarray(s) > 1.0))),
            fraction_features_ssnr_gt_2=("ssnr", lambda s: float(np.mean(np.asarray(s) > 2.0))),
        )
        .sort_values(["shots", "family", "encoding_alpha"])
    )
    ssnr_global.to_csv(
        RESULTS / "07_03b_alpha_feature_ssnr_summary.csv", index=False
    )

    # First average MC repeats within each physical case, then aggregate across
    # the 5 seeds x 8 protocols. This prevents repeats from acting as extra
    # pseudo-independent reservoir cases.
    case_summary = (
        noisy_df.groupby(
            ["encoding_alpha", "seed", "protocol", "W", "family", "shots"],
            dropna=False,
            as_index=False,
        )
        .agg(
            exact_cv_rmse=("exact_cv_rmse", "first"),
            mean_noisy_cv_rmse=("noisy_cv_rmse", "mean"),
            std_mc_noisy_cv_rmse=("noisy_cv_rmse", "std"),
            mean_delta_cv_rmse=("delta_cv_rmse", "mean"),
            mean_relative_cv_degradation_pct=("relative_cv_degradation_pct", "mean"),
            exact_validation_rmse=("exact_validation_rmse", "first"),
            mean_noisy_validation_rmse=("noisy_validation_rmse", "mean"),
        )
    )
    case_summary.to_csv(
        RESULTS / "07_03b_alpha_shot_noise_case_summary.csv", index=False
    )

    shot_global = (
        case_summary.groupby(["encoding_alpha", "family", "shots"], as_index=False)
        .agg(
            mean_exact_cv_rmse=("exact_cv_rmse", "mean"),
            mean_noisy_cv_rmse=("mean_noisy_cv_rmse", "mean"),
            mean_delta_cv_rmse=("mean_delta_cv_rmse", "mean"),
            mean_relative_cv_degradation_pct=("mean_relative_cv_degradation_pct", "mean"),
            mean_mc_std_cv_rmse=("std_mc_noisy_cv_rmse", "mean"),
        )
        .sort_values(["shots", "family", "encoding_alpha"])
    )
    shot_global.to_csv(
        RESULTS / "07_03b_alpha_shot_noise_global_summary.csv", index=False
    )

    # Compact landscape specifically for the strongest ideal baseline family.
    xzinj = shot_global[shot_global["family"] == "XZ_injection"].copy()
    xzinj_pivot = xzinj.pivot(
        index="encoding_alpha",
        columns="shots",
        values="mean_relative_cv_degradation_pct",
    )
    xzinj_pivot.to_csv(
        RESULTS / "07_03b_XZinj_degradation_alpha_by_shots.csv"
    )

    ssnr_xzinj = ssnr_global[ssnr_global["family"] == "XZ_injection"].copy()
    ssnr_pivot = ssnr_xzinj.pivot(
        index="encoding_alpha",
        columns="shots",
        values="median_feature_ssnr",
    )
    ssnr_pivot.to_csv(
        RESULTS / "07_03b_XZinj_median_ssnr_alpha_by_shots.csv"
    )

    print()
    print("=" * 124)
    print("EXACT CV LANDSCAPE BY alpha")
    print("Averaged over 5 seeds x 8 temporal protocols")
    print("=" * 124)
    print(exact_global.to_string(index=False))

    print()
    print("=" * 124)
    print("XZ_injection: MEAN RELATIVE CV DEGRADATION (%)")
    print("Rows=alpha, columns=shots")
    print("=" * 124)
    print(xzinj_pivot.to_string())

    print()
    print("=" * 124)
    print("XZ_injection: MEDIAN FEATURE SIGNAL-TO-SHOT-NOISE RATIO")
    print("SSNR = Std_t(<P>) / sqrt(mean_t[(1-<P>^2)/N])")
    print("Rows=alpha, columns=shots")
    print("=" * 124)
    print(ssnr_pivot.to_string())

    print()
    print("=" * 124)
    print("GLOBAL alpha x SHOT SUMMARY")
    print("=" * 124)
    print(shot_global.to_string(index=False))

    summary_path = RESULTS / "07_03b_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("WEEK 7 - STEP 7.3B\n")
        f.write("INPUT-STRENGTH alpha x FINITE-SHOT INTERACTION STUDY\n\n")
        f.write("ALPHAS = " + repr(ALPHAS) + "\n")
        f.write("SHOT_BUDGETS = " + repr(SHOT_BUDGETS) + "\n")
        f.write(f"N_SHOT_REPEATS = {N_SHOT_REPEATS}\n\n")
        f.write("EXACT CV LANDSCAPE BY alpha\n")
        f.write(exact_global.to_string(index=False))
        f.write("\n\nXZ_injection RELATIVE CV DEGRADATION (%)\n")
        f.write(xzinj_pivot.to_string())
        f.write("\n\nXZ_injection MEDIAN FEATURE SSNR\n")
        f.write(ssnr_pivot.to_string())
        f.write("\n\nGLOBAL alpha x SHOT SUMMARY\n")
        f.write(shot_global.to_string(index=False))
        f.write("\n")

    print()
    print("Saved:")
    for name in [
        "07_03b_alpha_exact_reference.csv",
        "07_03b_alpha_feature_ssnr_raw.csv",
        "07_03b_alpha_shot_noise_raw.csv",
        "07_03b_alpha_exact_global_summary.csv",
        "07_03b_alpha_feature_ssnr_summary.csv",
        "07_03b_alpha_shot_noise_case_summary.csv",
        "07_03b_alpha_shot_noise_global_summary.csv",
        "07_03b_XZinj_degradation_alpha_by_shots.csv",
        "07_03b_XZinj_median_ssnr_alpha_by_shots.csv",
        "07_03b_summary.txt",
    ]:
        print(f"  results/{name}")

    print()
    print("Step 7.3B computation finished.")
    print("Interpret exact-performance vs SSNR vs finite-shot degradation before moving on.")


if __name__ == "__main__":
    main()
