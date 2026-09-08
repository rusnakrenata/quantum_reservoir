"""
Week 7 - Step 7.3
Finite-shot estimator study for the frozen ideal QRC.

Purpose
-------
Steps 7.1-7.2B mapped which exact ideal observables contain forecasting
information. Step 7.3 asks how robust those candidate readouts are when an
expectation value <P> is estimated from a finite number of projective shots.

This is intentionally NOT yet a circuit-specific ancilla study and NOT a
hardware-noise study. Each Pauli expectation is treated as an estimator of a
+/-1 observable:

    Pr(+1) = (1 + <P>)/2
    Pr(-1) = (1 - <P>)/2

and the finite-shot estimate is

    <P>_N = (n_+ - n_-)/N = 2*n_+/N - 1.

Scientific rule
---------------
Keep the reservoir, temporal protocols, feature families, and data split
frozen. Do NOT select one final measurement family here.

The Ridge alpha chosen from the exact-feature chronological CV is also frozen
within each seed/protocol/family. This isolates measurement-shot degradation
rather than conflating it with a second hyperparameter search under every
Monte-Carlo shot draw.

Primary evidence: 2022-2024 chronological CV.
2025: diagnostic only.
2026: untouched.

Dependency
----------
Keep 07_02b_memory_pair_isolation.py in the same directory. This script reuses
its already-audited frozen QRC dynamics and observable extraction functions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "07_02b_memory_pair_isolation.py"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing {BASE_SCRIPT.name}. Keep it beside this Step-7.3 script."
    )

spec = importlib.util.spec_from_file_location("qrc_step_72b", BASE_SCRIPT)
qrc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qrc)

RESULTS = Path("results")

# Finite-shot budgets. Exact expectation values are stored separately as the
# noiseless reference, so 'infinite shots' is not placed in this array.
SHOT_BUDGETS = [128, 256, 512, 1024, 2048, 4096, 8192]

# Independent Monte-Carlo realizations of measurement statistics.
N_SHOT_REPEATS = 20
SHOT_SEED_BASE = 73001

# Carry forward the focused candidate landscape. This is NOT a final choice.
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
    """Independent +/-1 finite-shot estimator for every expectation value."""
    X = np.clip(np.asarray(X_exact, dtype=float), -1.0, 1.0)
    p_plus = np.clip((1.0 + X) / 2.0, 0.0, 1.0)
    n_plus = rng.binomial(int(shots), p_plus)
    return 2.0 * n_plus / float(shots) - 1.0


def fixed_alpha_cv(X: np.ndarray, y: np.ndarray, alpha: float):
    splitter = TimeSeriesSplit(n_splits=qrc.N_CV_SPLITS)
    scores = []
    for tr_idx, va_idx in splitter.split(X):
        pred = qrc.fit_predict_ridge(
            X[tr_idx], y[tr_idx], X[va_idx], alpha
        )
        scores.append(qrc.rmse(y[va_idx], pred))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def protocol_banks(A_list, S_list, cont_bank_all, n_train, n_val, y_train_full, y_val, protocol, W):
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


def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 124)
    print("WEEK 7 - STEP 7.3")
    print("FINITE-SHOT OBSERVABLE ROBUSTNESS STUDY")
    print("=" * 124)
    print()
    print("Frozen QRC:")
    print("  F4 | 4 injection + 2 memory | softsign policy")
    print(f"  alpha={qrc.ALPHA} | hx={qrc.HX} | dt={qrc.DT} | r={qrc.TROTTER_R}")
    print(f"  reservoir seeds={qrc.SEEDS}")
    print(f"  protocols=CONT + RWP {qrc.WINDOWS}")
    print(f"  shot budgets={SHOT_BUDGETS}")
    print(f"  shot Monte-Carlo repeats={N_SHOT_REPEATS}")
    print("  2025 diagnostic only; 2026 untouched")
    print()
    print("Finite-shot model:")
    print("  P(+1)=(1+<P>)/2; <P>_N=2*n_plus/N-1")
    print("  Each observable estimator is sampled independently in this step.")
    print("  Circuit grouping/covariance and ancilla implementation are deferred.")
    print("  Ridge lambda is frozen to the exact-feature CV optimum per case.")
    print()

    # Sanity check of the Bernoulli estimator.
    rng_check = np.random.default_rng(123)
    e0 = 0.35
    ncheck = 250_000
    draws = 2.0 * rng_check.binomial(1, (1.0 + e0) / 2.0, size=ncheck) - 1.0
    empirical = float(draws.mean())
    if abs(empirical - e0) > 0.01:
        raise RuntimeError("Finite-shot Bernoulli estimator self-check failed.")
    print(f"Shot-estimator self-check: PASS (target={e0:.3f}, empirical={empirical:.3f})")
    print()

    work, train, val, cols = qrc.load_data()
    n_train = len(train)
    n_val = len(val)
    angles_all = qrc.make_input_angles(work, cols)
    y_train_full = train[cols["target"]].to_numpy(dtype=float)
    y_val = val[cols["target"]].to_numpy(dtype=float)

    exact_rows = []
    raw_rows = []
    audit_rows = []

    for seed in qrc.SEEDS:
        print("#" * 124)
        print(f"RESERVOIR SEED = {seed}")
        print("#" * 124)

        U, unitary_err = qrc.build_trotter_unitary(seed)
        print(f"Unitarity error = {unitary_err:.3e}")
        print("Precomputing input-conditioned memory channels ...")
        A_list, S_list = qrc.build_input_channels(U, angles_all)
        cont_bank_all, _, _ = qrc.build_cont_bank(A_list)

        for protocol, W in PROTOCOL_SPECS:
            bank_train, bank_val, y_train, w_numeric = protocol_banks(
                A_list, S_list, cont_bank_all,
                n_train, n_val, y_train_full, y_val,
                protocol, W,
            )

            print()
            print(f"{protocol}: train={len(y_train)}, val={len(y_val)}")

            for family_name, feature_names in FAMILIES.items():
                X_train_exact = bank_train[feature_names].to_numpy(dtype=float)
                X_val_exact = bank_val[feature_names].to_numpy(dtype=float)

                # Exact-feature baseline and exact-feature optimal Ridge alpha.
                best_exact, _ = qrc.chronological_cv(X_train_exact, y_train)
                alpha_exact = float(best_exact["ridge_alpha"])
                exact_cv = float(best_exact["cv_rmse_mean"])
                pred_exact = qrc.fit_predict_ridge(
                    X_train_exact, y_train, X_val_exact, alpha_exact
                )
                exact_val = qrc.rmse(y_val, pred_exact)

                exact_rows.append({
                    "seed": seed,
                    "protocol": protocol,
                    "W": w_numeric,
                    "family": family_name,
                    "n_features": len(feature_names),
                    "ridge_alpha_exact": alpha_exact,
                    "exact_cv_rmse": exact_cv,
                    "exact_validation_rmse": exact_val,
                })

                # Audit unchanged XZ_all against Week 6.
                if family_name == "XZ_all":
                    ref = qrc.WEEK6_XZ_REF[seed][protocol]
                    dcv = exact_cv - ref["cv"]
                    dval = exact_val - ref["val"]
                    audit_rows.append({
                        "seed": seed,
                        "protocol": protocol,
                        "W": w_numeric,
                        "delta_cv": dcv,
                        "delta_val": dval,
                        "pass": abs(dcv) <= qrc.AUDIT_TOL and abs(dval) <= qrc.AUDIT_TOL,
                    })

                for shots in SHOT_BUDGETS:
                    for rep in range(N_SHOT_REPEATS):
                        # Deterministic but distinct random stream for every case.
                        name_code = sum((i + 1) * ord(c) for i, c in enumerate(family_name))
                        protocol_code = sum((i + 1) * ord(c) for i, c in enumerate(protocol))
                        rng_seed = (
                            SHOT_SEED_BASE
                            + seed * 1_000_003
                            + shots * 101
                            + rep * 10_007
                            + name_code * 17
                            + protocol_code * 19
                        ) % (2**63 - 1)
                        rng = np.random.default_rng(rng_seed)

                        X_train_noisy = shot_sample_expectations(
                            X_train_exact, shots, rng
                        )
                        X_val_noisy = shot_sample_expectations(
                            X_val_exact, shots, rng
                        )

                        noisy_cv, noisy_fold_std = fixed_alpha_cv(
                            X_train_noisy, y_train, alpha_exact
                        )
                        pred_val = qrc.fit_predict_ridge(
                            X_train_noisy, y_train,
                            X_val_noisy, alpha_exact
                        )
                        noisy_val = qrc.rmse(y_val, pred_val)

                        raw_rows.append({
                            "seed": seed,
                            "protocol": protocol,
                            "W": w_numeric,
                            "family": family_name,
                            "n_features": len(feature_names),
                            "shots": shots,
                            "shot_repeat": rep,
                            "ridge_alpha_frozen": alpha_exact,
                            "exact_cv_rmse": exact_cv,
                            "noisy_cv_rmse": noisy_cv,
                            "delta_cv_rmse": noisy_cv - exact_cv,
                            "relative_cv_degradation_pct": 100.0 * (noisy_cv - exact_cv) / exact_cv,
                            "noisy_cv_fold_std": noisy_fold_std,
                            "exact_validation_rmse": exact_val,
                            "noisy_validation_rmse": noisy_val,
                            "delta_validation_rmse": noisy_val - exact_val,
                            "noisy_validation_mae": float(mean_absolute_error(y_val, pred_val)),
                            "noisy_validation_bias": float(np.mean(pred_val - y_val)),
                        })

                print(
                    f"  {family_name:26s} exact CV={exact_cv:.6f} "
                    f"lambda={alpha_exact:g}"
                )

    exact_df = pd.DataFrame(exact_rows)
    raw_df = pd.DataFrame(raw_rows)
    audit_df = pd.DataFrame(audit_rows)

    exact_df.to_csv(RESULTS / "07_03_exact_reference.csv", index=False)
    raw_df.to_csv(RESULTS / "07_03_shot_noise_raw.csv", index=False)
    audit_df.to_csv(RESULTS / "07_03_week6_reproduction_audit.csv", index=False)

    print()
    print("=" * 124)
    print("WEEK-6 REPRODUCTION AUDIT")
    print("=" * 124)
    print(audit_df.to_string(index=False))
    print()
    if not audit_df["pass"].all():
        print("STOP: exact XZ_all reference did not reproduce Week 6.")
        return

    # Monte-Carlo mean within each seed/protocol/family/shot budget.
    case = (
        raw_df.groupby(["seed", "protocol", "W", "family", "n_features", "shots"], dropna=False, as_index=False)
        .agg(
            exact_cv_rmse=("exact_cv_rmse", "first"),
            mean_noisy_cv_rmse=("noisy_cv_rmse", "mean"),
            std_noisy_cv_rmse=("noisy_cv_rmse", "std"),
            mean_delta_cv_rmse=("delta_cv_rmse", "mean"),
            std_delta_cv_rmse=("delta_cv_rmse", "std"),
            mean_relative_cv_degradation_pct=("relative_cv_degradation_pct", "mean"),
            exact_validation_rmse=("exact_validation_rmse", "first"),
            mean_noisy_validation_rmse=("noisy_validation_rmse", "mean"),
            std_noisy_validation_rmse=("noisy_validation_rmse", "std"),
            mean_delta_validation_rmse=("delta_validation_rmse", "mean"),
        )
    )
    case.to_csv(RESULTS / "07_03_shot_noise_case_summary.csv", index=False)

    # Global robustness across 5 seeds x 8 protocols for each family/shot budget.
    global_summary = (
        case.groupby(["family", "n_features", "shots"], as_index=False)
        .agg(
            mean_exact_cv_rmse=("exact_cv_rmse", "mean"),
            mean_noisy_cv_rmse=("mean_noisy_cv_rmse", "mean"),
            mean_delta_cv_rmse=("mean_delta_cv_rmse", "mean"),
            median_delta_cv_rmse=("mean_delta_cv_rmse", "median"),
            max_mean_delta_cv_rmse=("mean_delta_cv_rmse", "max"),
            mean_relative_cv_degradation_pct=("mean_relative_cv_degradation_pct", "mean"),
            mean_mc_std_cv_rmse=("std_noisy_cv_rmse", "mean"),
            cases_with_mean_degradation_le_001=("mean_delta_cv_rmse", lambda s: int((s <= 0.01).sum())),
            cases_with_mean_degradation_le_1pct=("mean_relative_cv_degradation_pct", lambda s: int((s <= 1.0).sum())),
            mean_exact_validation_rmse=("exact_validation_rmse", "mean"),
            mean_noisy_validation_rmse=("mean_noisy_validation_rmse", "mean"),
            mean_delta_validation_rmse=("mean_delta_validation_rmse", "mean"),
        )
        .sort_values(["shots", "mean_noisy_cv_rmse"])
        .reset_index(drop=True)
    )
    global_summary.to_csv(RESULTS / "07_03_shot_noise_global_summary.csv", index=False)

    # Protocol-sensitive landscape averaged over reservoir seeds and MC repeats.
    protocol_summary = (
        case.groupby(["protocol", "family", "shots"], as_index=False)
        .agg(
            mean_exact_cv_rmse=("exact_cv_rmse", "mean"),
            mean_noisy_cv_rmse=("mean_noisy_cv_rmse", "mean"),
            mean_delta_cv_rmse=("mean_delta_cv_rmse", "mean"),
            mean_relative_cv_degradation_pct=("mean_relative_cv_degradation_pct", "mean"),
        )
    )
    protocol_summary.to_csv(RESULTS / "07_03_shot_noise_protocol_summary.csv", index=False)

    # Minimum tested shot budget at which each seed/protocol/family case is on
    # average within 1% of its exact CV RMSE.
    thresholds = []
    key_cols = ["seed", "protocol", "W", "family", "n_features"]
    for key, g in case.groupby(key_cols, dropna=False):
        g = g.sort_values("shots")
        ok = g[g["mean_relative_cv_degradation_pct"] <= 1.0]
        min_shots = int(ok.iloc[0]["shots"]) if len(ok) else np.nan
        thresholds.append({
            **dict(zip(key_cols, key if isinstance(key, tuple) else (key,))),
            "min_tested_shots_within_1pct_exact_cv": min_shots,
        })
    threshold_df = pd.DataFrame(thresholds)
    threshold_df.to_csv(RESULTS / "07_03_shot_thresholds.csv", index=False)

    threshold_family = (
        threshold_df.groupby(["family", "n_features"], as_index=False)
        .agg(
            cases_reaching_1pct=("min_tested_shots_within_1pct_exact_cv", lambda s: int(s.notna().sum())),
            median_min_shots_within_1pct=("min_tested_shots_within_1pct_exact_cv", "median"),
            max_min_shots_within_1pct=("min_tested_shots_within_1pct_exact_cv", "max"),
        )
    )
    threshold_family.to_csv(RESULTS / "07_03_shot_thresholds_by_family.csv", index=False)

    print("=" * 124)
    print("GLOBAL FINITE-SHOT ROBUSTNESS")
    print("Averaged over 5 reservoir seeds x 8 temporal protocols; exact Ridge lambda frozen.")
    print("Positive delta = shot noise worsens CV RMSE.")
    print("=" * 124)
    print(global_summary[[
        "shots", "family", "n_features",
        "mean_exact_cv_rmse", "mean_noisy_cv_rmse",
        "mean_delta_cv_rmse", "mean_relative_cv_degradation_pct",
        "mean_mc_std_cv_rmse", "cases_with_mean_degradation_le_1pct",
    ]].to_string(index=False))
    print()

    print("=" * 124)
    print("SHOT BUDGET NEEDED TO BE WITHIN 1% OF EXACT CV")
    print("40 cases per family = 5 seeds x 8 protocols.")
    print("=" * 124)
    print(threshold_family.to_string(index=False))
    print()

    # Compact pivot of average CV degradation by family and shot budget.
    pivot = global_summary.pivot(
        index="family", columns="shots", values="mean_relative_cv_degradation_pct"
    ).reindex(index=list(FAMILIES.keys()), columns=SHOT_BUDGETS)
    pivot.to_csv(RESULTS / "07_03_relative_degradation_pct_pivot.csv")

    print("=" * 124)
    print("MEAN RELATIVE CV DEGRADATION (%)")
    print("=" * 124)
    print(pivot.to_string())
    print()

    summary_path = RESULTS / "07_03_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("WEEK 7 - STEP 7.3\n")
        f.write("Finite-shot observable robustness study\n\n")
        f.write("Estimator model: independent +/-1 Pauli sampling per observable.\n")
        f.write(f"Shot budgets: {SHOT_BUDGETS}\n")
        f.write(f"Monte-Carlo repeats: {N_SHOT_REPEATS}\n")
        f.write("Ridge alpha frozen to exact-feature CV optimum per seed/protocol/family.\n")
        f.write("This is not yet ancilla/circuit-specific or hardware-noise simulation.\n\n")
        f.write("Global summary:\n")
        f.write(global_summary.to_string(index=False))
        f.write("\n\n1% shot thresholds by family:\n")
        f.write(threshold_family.to_string(index=False))
        f.write("\n\nMean relative degradation pivot:\n")
        f.write(pivot.to_string())
        f.write("\n")

    print("Saved:")
    for p in [
        RESULTS / "07_03_exact_reference.csv",
        RESULTS / "07_03_shot_noise_raw.csv",
        RESULTS / "07_03_shot_noise_case_summary.csv",
        RESULTS / "07_03_shot_noise_global_summary.csv",
        RESULTS / "07_03_shot_noise_protocol_summary.csv",
        RESULTS / "07_03_shot_thresholds.csv",
        RESULTS / "07_03_shot_thresholds_by_family.csv",
        RESULTS / "07_03_relative_degradation_pct_pivot.csv",
        RESULTS / "07_03_week6_reproduction_audit.csv",
        RESULTS / "07_03_summary.txt",
    ]:
        print(f"  {p}")

    print()
    print("Step 7.3 computation finished.")
    print("Interpret shot robustness across families/protocols before ancilla implementation.")
    print("Do NOT freeze one observable setup in this step.")


if __name__ == "__main__":
    main()
