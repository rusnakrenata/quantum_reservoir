"""
Week 7 - Step 7.4C.4
Real chronological task-conditioned accessible memory.

Question
--------
On the REAL chronological insurance input sequence, do the current QRC
features x(t) contain information about earlier inputs u_j(t-k) beyond what
can already be inferred from the current input u(t)?

This is NOT the same as intrinsic memory capacity on randomized inputs.
It is a task-conditioned memory diagnostic for the actual forecasting data.

Definitions
-----------
For input channel j and delay k:

    MC_current(j,k)
        = Corr^2[u_j(t-k), uhat_current(j,k,t)]

    MC_QRC(j,k)
        = Corr^2[u_j(t-k), uhat_QRC(j,k,t)]

    MC_current+QRC(j,k)
        = Corr^2[u_j(t-k), uhat_current+QRC(j,k,t)]

The most useful incremental quantity is

    Delta_MC_add(j,k)
        = MC_current+QRC(j,k) - MC_current(j,k)

because it asks whether QRC features add delayed-history information beyond
the current encoded input itself.

Reservoir / feature families
----------------------------
Ideal/unmeasured CONT only; no YX45 measurement back-action.

Compare:
    1) XZ_injection
    2) XZ_injection + YX45 expectation

The YX45 term in family (2) is only an exact expectation feature here.

Chronology
----------
Original chronological encoded F4 inputs are used.
Train = original 2022-2024 portion
Validation = original 2025 portion

Delays k=1..28 are tested, matching the temporal windows used elsewhere in the
project. Selected delays printed: 1,2,5,7,14,21,28.

Readout
-------
For every channel/delay/model:
- train-only feature standardization
- near-constant columns removed
- Ridge alpha chosen by 5-fold TimeSeriesSplit on training data
- final delayed-input reconstruction evaluated on untouched 2025 validation

Dependency
----------
Keep beside this script:
    07_04c2_XZinj_memory_null_corrected.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
C2 = HERE / "07_04c2_XZinj_memory_null_corrected.py"

if not C2.exists():
    raise FileNotFoundError(
        f"Missing {C2.name}. Keep it beside this script."
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


c2 = load_module(C2, "memory74c2")
qrc = c2.qrc

ALPHA = 0.75
SEED = 42

BURN_IN = 100
K_MAX = 28
SELECTED_DELAYS = [1, 2, 5, 7, 14, 21, 28]

RIDGE_GRID = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
N_CV_SPLITS = 5
VAR_TOL = 1e-12

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]


# =============================================================================
# Statistics / stable Ridge
# =============================================================================

def corr2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2:
        return np.nan

    if np.std(a) <= VAR_TOL or np.std(b) <= VAR_TOL:
        return 0.0

    r = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(r):
        return 0.0

    return float(r * r)


def fit_scaler(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    keep = sd > VAR_TOL
    return mu, sd, keep


def transform_with_scaler(X, mu, sd, keep):
    X = np.asarray(X, dtype=float)

    if not np.any(keep):
        return np.zeros((len(X), 0), dtype=float)

    return (X[:, keep] - mu[keep]) / sd[keep]


def predict_mean(ytr, n):
    return np.full(n, float(np.mean(ytr)))


def choose_ridge_alpha_timeseries(X, y):
    """
    Chronological 5-fold CV using validation Corr^2 as the score.
    The alpha with largest mean CV Corr^2 is selected.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    splitter = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    rows = []

    for alpha in RIDGE_GRID:
        scores = []

        for tr_idx, va_idx in splitter.split(X):
            Xtr = X[tr_idx]
            Xva = X[va_idx]
            ytr = y[tr_idx]
            yva = y[va_idx]

            mu, sd, keep = fit_scaler(Xtr)

            if not np.any(keep):
                pred = predict_mean(ytr, len(yva))
            else:
                Xtr_z = transform_with_scaler(Xtr, mu, sd, keep)
                Xva_z = transform_with_scaler(Xva, mu, sd, keep)

                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(Xtr_z, ytr)
                pred = model.predict(Xva_z)

            scores.append(corr2(yva, pred))

        rows.append({
            "ridge_alpha": alpha,
            "cv_MC_mean": float(np.mean(scores)),
            "cv_MC_std": float(np.std(scores, ddof=1)),
        })

    cv_df = pd.DataFrame(rows).sort_values(
        ["cv_MC_mean", "ridge_alpha"],
        ascending=[False, True],
    )

    best_alpha = float(cv_df.iloc[0]["ridge_alpha"])
    return best_alpha, cv_df


def train_and_validate(Xtr, ytr, Xva, yva):
    best_alpha, cv_df = choose_ridge_alpha_timeseries(Xtr, ytr)

    mu, sd, keep = fit_scaler(Xtr)

    if not np.any(keep):
        pred = predict_mean(ytr, len(yva))
        n_features_kept = 0
    else:
        Xtr_z = transform_with_scaler(Xtr, mu, sd, keep)
        Xva_z = transform_with_scaler(Xva, mu, sd, keep)

        model = Ridge(alpha=best_alpha, fit_intercept=True)
        model.fit(Xtr_z, ytr)
        pred = model.predict(Xva_z)
        n_features_kept = int(np.sum(keep))

    return {
        "MC_validation": corr2(yva, pred),
        "ridge_alpha": best_alpha,
        "n_features_kept": n_features_kept,
        "cv_MC_mean": float(cv_df.iloc[0]["cv_MC_mean"]),
    }


# =============================================================================
# Chronological memory calculation
# =============================================================================

def evaluate_feature_family(
    family_name,
    X_qrc,
    U,
    n_train,
    n_val,
):
    """
    For each (channel j, delay k), compare three readouts:

      current       : U(t)
      QRC           : X_qrc(t)
      current+QRC   : [U(t), X_qrc(t)]
    """
    rows = []

    for j in range(U.shape[1]):
        print(f"  {family_name}: channel {j} ...")

        for k in range(1, K_MAX + 1):
            train_start = max(BURN_IN, k)

            train_t = np.arange(train_start, n_train, dtype=int)
            val_t = np.arange(n_train, n_train + n_val, dtype=int)

            ytr = U[train_t - k, j]
            yva = U[val_t - k, j]

            matrices = {
                "current_input": U,
                "QRC": X_qrc,
                "current_plus_QRC": np.hstack([U, X_qrc]),
            }

            result_by_model = {}

            for model_name, Xall in matrices.items():
                res = train_and_validate(
                    Xall[train_t],
                    ytr,
                    Xall[val_t],
                    yva,
                )
                result_by_model[model_name] = res

            mc_current = result_by_model["current_input"]["MC_validation"]
            mc_qrc = result_by_model["QRC"]["MC_validation"]
            mc_combined = result_by_model["current_plus_QRC"]["MC_validation"]

            rows.append({
                "feature_family": family_name,
                "channel": j,
                "delay": k,

                "MC_current": mc_current,
                "MC_QRC": mc_qrc,
                "MC_current_plus_QRC": mc_combined,

                "Delta_MC_QRC_vs_current": mc_qrc - mc_current,
                "Delta_MC_add": mc_combined - mc_current,

                "alpha_current": result_by_model["current_input"]["ridge_alpha"],
                "alpha_QRC": result_by_model["QRC"]["ridge_alpha"],
                "alpha_current_plus_QRC": result_by_model["current_plus_QRC"]["ridge_alpha"],

                "cv_MC_current": result_by_model["current_input"]["cv_MC_mean"],
                "cv_MC_QRC": result_by_model["QRC"]["cv_MC_mean"],
                "cv_MC_current_plus_QRC": result_by_model["current_plus_QRC"]["cv_MC_mean"],

                "n_features_current": result_by_model["current_input"]["n_features_kept"],
                "n_features_QRC": result_by_model["QRC"]["n_features_kept"],
                "n_features_current_plus_QRC": result_by_model["current_plus_QRC"]["n_features_kept"],
            })

    return pd.DataFrame(rows)


def summarize(df):
    rows = []

    for family in df["feature_family"].unique():
        sf = df[df["feature_family"] == family]

        for j in sorted(sf["channel"].unique()):
            s = sf[sf["channel"] == j].sort_values("delay")

            positive_add = s[s["Delta_MC_add"] > 0]

            rows.append({
                "feature_family": family,
                "channel": j,
                "sum_MC_current": float(s["MC_current"].sum()),
                "sum_MC_QRC": float(s["MC_QRC"].sum()),
                "sum_MC_current_plus_QRC": float(s["MC_current_plus_QRC"].sum()),
                "sum_Delta_MC_QRC_vs_current": float(
                    s["Delta_MC_QRC_vs_current"].sum()
                ),
                "sum_Delta_MC_add": float(s["Delta_MC_add"].sum()),
                "n_delays_QRC_better_than_current": int(
                    np.sum(s["Delta_MC_QRC_vs_current"] > 0)
                ),
                "n_delays_combined_better_than_current": int(
                    np.sum(s["Delta_MC_add"] > 0)
                ),
                "max_positive_Delta_MC_add": float(
                    max(s["Delta_MC_add"].max(), 0.0)
                ),
                "delay_of_max_Delta_MC_add": int(
                    s.loc[s["Delta_MC_add"].idxmax(), "delay"]
                ),
            })

        # Aggregate across all 4 channels and 28 delays.
        rows.append({
            "feature_family": family,
            "channel": "ALL",
            "sum_MC_current": float(sf["MC_current"].sum()),
            "sum_MC_QRC": float(sf["MC_QRC"].sum()),
            "sum_MC_current_plus_QRC": float(sf["MC_current_plus_QRC"].sum()),
            "sum_Delta_MC_QRC_vs_current": float(
                sf["Delta_MC_QRC_vs_current"].sum()
            ),
            "sum_Delta_MC_add": float(sf["Delta_MC_add"].sum()),
            "n_delays_QRC_better_than_current": int(
                np.sum(sf["Delta_MC_QRC_vs_current"] > 0)
            ),
            "n_delays_combined_better_than_current": int(
                np.sum(sf["Delta_MC_add"] > 0)
            ),
            "max_positive_Delta_MC_add": float(
                max(sf["Delta_MC_add"].max(), 0.0)
            ),
            "delay_of_max_Delta_MC_add": np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 128)
    print("WEEK 7 - STEP 7.4C.4")
    print("REAL CHRONOLOGICAL TASK-CONDITIONED ACCESSIBLE MEMORY")
    print("=" * 128)
    print()
    print("Input:")
    print("  ORIGINAL chronological F4 injection-angle sequence")
    print("  no shuffling")
    print()
    print("Reservoir:")
    print("  ideal/unmeasured CONT")
    print("  fixed initial memory=I/4")
    print("  NO YX45 measurement back-action")
    print()
    print("Feature families:")
    print("  XZ_injection")
    print("  XZ_injection + YX45 expectation")
    print()
    print("For each input channel j and delay k:")
    print("  current baseline    : u(t) -> u_j(t-k)")
    print("  QRC                 : x(t) -> u_j(t-k)")
    print("  current + QRC       : [u(t),x(t)] -> u_j(t-k)")
    print()
    print("Key incremental quantity:")
    print("  Delta_MC_add = MC_current+QRC - MC_current")
    print()
    print(f"Delays k=1..{K_MAX}")
    print(f"Common burn-in={BURN_IN}")
    print(f"Ridge grid={RIDGE_GRID}; 5-fold chronological CV")
    print()

    work, train, val, cols = qrc.load_data()
    n_train = len(train)
    n_val = len(val)
    n_steps = n_train + n_val

    Uchron = np.asarray(
        qrc.make_input_angles(work, cols),
        dtype=float,
    )[:n_steps]

    print(f"Train={n_train}, validation={n_val}, total={n_steps}")
    print(f"Chronological injection channels={Uchron.shape[1]}")

    # Basic autocorrelation audit: the whole reason the current-input baseline
    # is necessary in this task-conditioned test.
    ac_rows = []
    for j in range(Uchron.shape[1]):
        for k in range(1, K_MAX + 1):
            r = np.corrcoef(Uchron[k:, j], Uchron[:-k, j])[0, 1]
            if not np.isfinite(r):
                r = 0.0
            ac_rows.append({
                "channel": j,
                "delay": k,
                "autocorr": float(r),
                "autocorr2": float(r * r),
            })

    ac_df = pd.DataFrame(ac_rows)
    ac_df.to_csv(
        RESULTS / "07_04c4_chronological_input_autocorrelation.csv",
        index=False,
    )

    print(
        "Maximum |input autocorrelation| over k=1..28 = "
        f"{ac_df['autocorr'].abs().max():.6f}"
    )

    Ures, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, _ = qrc.build_input_channels(Ures, Uchron)

    print(f"QRC unitarity error = {unitary_err:.3e}")
    print()

    X9 = c2.evolve_ideal_features(A_list)

    feature_sets = {
        "XZ_injection": X9[:, :8],
        "XZ_injection_plus_YX45": X9[:, :9],
    }

    parts = []

    for family, X in feature_sets.items():
        print(f"Evaluating chronological memory: {family}")
        parts.append(
            evaluate_feature_family(
                family,
                X,
                Uchron,
                n_train,
                n_val,
            )
        )

    detail = pd.concat(parts, ignore_index=True)
    summary = summarize(detail)

    detail.to_csv(
        RESULTS / "07_04c4_chronological_memory_by_delay.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS / "07_04c4_chronological_memory_summary.csv",
        index=False,
    )

    print()
    print("-" * 128)
    print("SUMMARY")
    print("-" * 128)
    print(summary.to_string(index=False))

    print()
    print("-" * 128)
    print("SELECTED DELAYS")
    print("-" * 128)

    selected = detail[
        detail["delay"].isin(SELECTED_DELAYS)
    ][
        [
            "feature_family",
            "channel",
            "delay",
            "MC_current",
            "MC_QRC",
            "MC_current_plus_QRC",
            "Delta_MC_QRC_vs_current",
            "Delta_MC_add",
        ]
    ]

    print(selected.to_string(index=False))

    print()
    print("-" * 128)
    print("TOP POSITIVE INCREMENTAL QRC GAINS")
    print("-" * 128)

    top = detail.sort_values(
        "Delta_MC_add",
        ascending=False,
    ).head(20)

    print(
        top[
            [
                "feature_family",
                "channel",
                "delay",
                "MC_current",
                "MC_QRC",
                "MC_current_plus_QRC",
                "Delta_MC_add",
            ]
        ].to_string(index=False)
    )

    with open(
        RESULTS / "07_04c4_chronological_memory_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "WEEK 7 STEP 7.4C.4 - REAL CHRONOLOGICAL TASK-CONDITIONED MEMORY\n"
        )
        fp.write("=" * 100 + "\n\n")
        fp.write(
            "Delta_MC_add = MC_current+QRC - MC_current\n"
        )
        fp.write(
            "Positive Delta_MC_add means QRC features add delayed-history "
            "information beyond current inputs.\n\n"
        )
        fp.write(summary.to_string(index=False))
        fp.write("\n")

    print()
    print("=" * 128)
    print("Saved:")
    print("  results/07_04c4_chronological_input_autocorrelation.csv")
    print("  results/07_04c4_chronological_memory_by_delay.csv")
    print("  results/07_04c4_chronological_memory_summary.csv")
    print("  results/07_04c4_chronological_memory_summary.txt")
    print()
    print("Interpretation rule:")
    print("  do not call high MC 'reservoir memory' if current input alone already")
    print("  reconstructs the same delay.")
    print("  Focus on Delta_MC_add > 0.")


if __name__ == "__main__":
    main()
