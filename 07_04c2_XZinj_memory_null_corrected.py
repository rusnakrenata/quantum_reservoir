"""
Week 7 - Step 7.4C.2
Intrinsic memory in XZ_injection without YX45 memory measurement/back-action.

Purpose
-------
Determine whether the reservoir has genuinely accessible delayed-input memory
when we remove the Y4 X5 memory observable/measurement and use only the
injection-qubit feature family:

    XZ_injection =
    [<X0>,<X1>,<X2>,<X3>,<Z0>,<Z1>,<Z2>,<Z3>]

For comparison, the same ideal/unmeasured trajectory is also evaluated with

    XZ_injection + <Y4 X5>.

No YX45 projective/ancilla measurement back-action is applied in either case.

Memory definition
-----------------
For input channel j and delay k:

    MC_{j,k} = Corr^2[u_j(t-k), uhat_{j,k}(t)]

A separate Ridge readout is trained for each (j,k).

Null correction
---------------
With finite validation size N, Corr^2 is positive even for unrelated variables.
Therefore, for every fitted delayed readout we estimate a permutation null by
randomly permuting the validation target and recomputing Corr^2.

Reported:
    raw MC_{j,k}
    null mean
    null 95th percentile
    bias-corrected MC = max(raw MC - null mean, 0)
    significant = raw MC > null 95th percentile

Probe
-----
The exact F4 injection-angle values are independently permuted in time per
channel. This preserves their empirical marginal distributions while
destroying chronological autocorrelation.

Dependency
----------
Keep beside this script:
    07_04b1_cont_washout_diagnostic.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
B1 = HERE / "07_04b1_cont_washout_diagnostic.py"

if not B1.exists():
    raise FileNotFoundError(f"Missing {B1.name}. Keep it beside this script.")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


w = load_module(B1, "washout74b1")
qrc = w.qrc

ALPHA = 0.75
SEED = 42

BURN_IN = 100
K_MAX = 100

PROBE_SEED = 74001
NULL_SEED = 74002
N_NULL = 200

RIDGE_ALPHA = 1e-6
VAR_TOL = 1e-12

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]


# =============================================================================
# Helpers
# =============================================================================

def build_distribution_matched_probe(angles, rng):
    angles = np.asarray(angles, dtype=float)
    probe = np.empty_like(angles)
    for j in range(angles.shape[1]):
        probe[:, j] = angles[rng.permutation(len(angles)), j]
    return probe


def safe_corr2(a, b):
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


def fit_predict_memory_readout(Xtr, ytr, Xva):
    Xtr = np.asarray(Xtr, dtype=float)
    Xva = np.asarray(Xva, dtype=float)
    ytr = np.asarray(ytr, dtype=float)

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=0)

    keep = sd > VAR_TOL

    if not np.any(keep):
        return np.full(len(Xva), float(np.mean(ytr)))

    Xtr_z = (Xtr[:, keep] - mu[keep]) / sd[keep]
    Xva_z = (Xva[:, keep] - mu[keep]) / sd[keep]

    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(Xtr_z, ytr)

    return model.predict(Xva_z)


def evolve_ideal_features(A_list):
    """
    Ideal/unmeasured CONT from fixed memory initialization I/4.

    The inherited evolve_one_input returns the focused 9-feature vector:
      first 8 = XZ_injection
      last 1  = <Y4 X5>
    """
    rho = w.INITIAL_STATES["I/4"].copy()
    rows = []

    for A in A_list:
        rho, x = w.evolve_one_input(A, rho)
        rows.append(np.asarray(x, dtype=float))

    return np.vstack(rows)


def permutation_null(y_true, pred, rng, n_null):
    vals = np.zeros(n_null, dtype=float)

    for r in range(n_null):
        vals[r] = safe_corr2(
            rng.permutation(y_true),
            pred,
        )

    return (
        float(np.mean(vals)),
        float(np.quantile(vals, 0.95)),
    )


def memory_curve(X, probe, n_train, n_val, family_name, null_rng):
    rows = []

    for j in range(probe.shape[1]):
        for k in range(1, K_MAX + 1):

            train_start = max(BURN_IN, k)

            train_t = np.arange(train_start, n_train, dtype=int)
            val_t = np.arange(n_train, n_train + n_val, dtype=int)

            Xtr = X[train_t]
            ytr = probe[train_t - k, j]

            Xva = X[val_t]
            yva = probe[val_t - k, j]

            pred = fit_predict_memory_readout(Xtr, ytr, Xva)

            raw_mc = safe_corr2(yva, pred)
            null_mean, null_p95 = permutation_null(
                yva,
                pred,
                null_rng,
                N_NULL,
            )

            corrected = max(raw_mc - null_mean, 0.0)
            significant = int(raw_mc > null_p95)

            rows.append({
                "feature_family": family_name,
                "channel": j,
                "delay": k,
                "MC_raw": raw_mc,
                "MC_null_mean": null_mean,
                "MC_null_p95": null_p95,
                "MC_corrected": corrected,
                "significant_95": significant,
            })

    return pd.DataFrame(rows)


def summarize(curve):
    rows = []

    for family in curve["feature_family"].unique():
        sf = curve[curve["feature_family"] == family]

        for j in sorted(sf["channel"].unique()):
            s = sf[sf["channel"] == j]

            sig_delays = s[s["significant_95"] == 1]["delay"].to_numpy(dtype=int)

            rows.append({
                "feature_family": family,
                "channel": j,
                "raw_MC_sum": float(s["MC_raw"].sum()),
                "null_mean_sum": float(s["MC_null_mean"].sum()),
                "corrected_MC_sum": float(s["MC_corrected"].sum()),
                "n_significant_delays_95": int(s["significant_95"].sum()),
                "max_significant_delay_95": (
                    int(np.max(sig_delays)) if len(sig_delays) else 0
                ),
                "max_raw_MC": float(s["MC_raw"].max()),
                "max_corrected_MC": float(s["MC_corrected"].max()),
            })

        rows.append({
            "feature_family": family,
            "channel": "ALL",
            "raw_MC_sum": float(sf["MC_raw"].sum()),
            "null_mean_sum": float(sf["MC_null_mean"].sum()),
            "corrected_MC_sum": float(sf["MC_corrected"].sum()),
            "n_significant_delays_95": int(sf["significant_95"].sum()),
            "max_significant_delay_95": np.nan,
            "max_raw_MC": float(sf["MC_raw"].max()),
            "max_corrected_MC": float(sf["MC_corrected"].max()),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 124)
    print("WEEK 7 - STEP 7.4C.2")
    print("MEMORY CAPACITY OF XZ_INJECTION WITHOUT YX45 MEMORY MEASUREMENT")
    print("=" * 124)
    print()
    print("Reservoir dynamics:")
    print("  ideal/unmeasured CONT only")
    print("  fixed initial memory = I/4")
    print("  NO Y4X5 measurement back-action")
    print()
    print("Feature families:")
    print("  1) XZ_injection          (8 features)")
    print("  2) XZ_injection + YX45   (9 features, expectation only)")
    print()
    print("Memory:")
    print("  MC_{j,k}=Corr^2[u_j(t-k), uhat_{j,k}(t)]")
    print(f"  burn-in={BURN_IN}, delays=1..{K_MAX}")
    print()
    print("Null test:")
    print(f"  {N_NULL} validation-target permutations per (channel,delay)")
    print("  corrected MC = max(raw MC - null mean, 0)")
    print("  significant if raw MC > null 95th percentile")
    print()

    work, train, val, cols = qrc.load_data()
    n_train = len(train)
    n_val = len(val)
    n_steps = n_train + n_val

    angles = np.asarray(
        qrc.make_input_angles(work, cols),
        dtype=float,
    )[:n_steps]

    probe_rng = np.random.default_rng(PROBE_SEED)
    probe = build_distribution_matched_probe(angles, probe_rng)

    marginal_error = float(
        np.max(
            np.abs(
                np.sort(probe, axis=0)
                - np.sort(angles, axis=0)
            )
        )
    )

    max_ac = 0.0
    for j in range(probe.shape[1]):
        for lag in range(1, 21):
            r = np.corrcoef(probe[lag:, j], probe[:-lag, j])[0, 1]
            if np.isfinite(r):
                max_ac = max(max_ac, abs(float(r)))

    print(f"Marginal preservation max error = {marginal_error:.3e}")
    print(f"Randomized probe max |autocorr|, lags 1..20 = {max_ac:.6f}")

    U, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, _ = qrc.build_input_channels(U, probe)

    print(f"QRC unitarity error = {unitary_err:.3e}")
    print()

    X9 = evolve_ideal_features(A_list)

    if X9.shape[1] < 9:
        raise RuntimeError(
            f"Expected at least 9 focused features, got {X9.shape[1]}."
        )

    feature_sets = {
        "XZ_injection": X9[:, :8],
        "XZ_injection_plus_YX45": X9[:, :9],
    }

    curve_parts = []

    for idx, (family, X) in enumerate(feature_sets.items()):
        print(f"Calculating memory curve: {family} ...")
        null_rng = np.random.default_rng(NULL_SEED + idx * 100000)

        curve_parts.append(
            memory_curve(
                X,
                probe,
                n_train,
                n_val,
                family,
                null_rng,
            )
        )

    curve_df = pd.concat(curve_parts, ignore_index=True)
    summary_df = summarize(curve_df)

    curve_df.to_csv(
        RESULTS / "07_04c2_XZinj_memory_null_corrected_by_delay.csv",
        index=False,
    )
    summary_df.to_csv(
        RESULTS / "07_04c2_XZinj_memory_null_corrected_summary.csv",
        index=False,
    )

    print()
    print("-" * 124)
    print("SUMMARY")
    print("-" * 124)
    print(summary_df.to_string(index=False))

    print()
    print("-" * 124)
    print("TOP SIGNIFICANT DELAYS")
    print("-" * 124)

    sig = curve_df[curve_df["significant_95"] == 1].copy()

    if len(sig) == 0:
        print("No delay/channel exceeds the 95% permutation-null threshold.")
    else:
        sig = sig.sort_values(
            ["feature_family", "MC_corrected"],
            ascending=[True, False],
        )
        print(
            sig[
                [
                    "feature_family",
                    "channel",
                    "delay",
                    "MC_raw",
                    "MC_null_mean",
                    "MC_null_p95",
                    "MC_corrected",
                ]
            ]
            .groupby("feature_family", group_keys=False)
            .head(15)
            .to_string(index=False)
        )

    with open(
        RESULTS / "07_04c2_XZinj_memory_null_corrected_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "WEEK 7 STEP 7.4C.2 - XZ_INJECTION MEMORY CAPACITY\n"
        )
        fp.write("=" * 90 + "\n\n")
        fp.write(
            "Ideal/unmeasured CONT; no YX45 measurement back-action.\n"
        )
        fp.write(
            "Feature families: XZ_injection vs XZ_injection+YX45.\n"
        )
        fp.write(
            f"Burn-in={BURN_IN}, K_MAX={K_MAX}, N_NULL={N_NULL}.\n\n"
        )
        fp.write(summary_df.to_string(index=False))
        fp.write("\n")

    print()
    print("=" * 124)
    print("Saved:")
    print("  results/07_04c2_XZinj_memory_null_corrected_by_delay.csv")
    print("  results/07_04c2_XZinj_memory_null_corrected_summary.csv")
    print("  results/07_04c2_XZinj_memory_null_corrected_summary.txt")
    print()
    print("Interpretation rule:")
    print("  raw MC alone is not sufficient.")
    print("  Look for corrected MC and delays exceeding the 95% null threshold.")


if __name__ == "__main__":
    main()
