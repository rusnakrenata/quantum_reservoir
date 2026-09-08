"""
Week 7 - Step 7.4C.3
Multiple-comparison-corrected significance test for intrinsic memory.

Purpose
-------
Re-test the ideal/unmeasured CONT memory result for:

    1) XZ_injection
    2) XZ_injection + YX45 expectation

without any YX45 measurement back-action.

For each input channel j and delay k:

    MC_{j,k} = Corr^2[u_j(t-k), uhat_{j,k}(t)]

A permutation p-value is calculated for the observed MC_{j,k}.
Benjamini-Hochberg false-discovery-rate (FDR) correction is then applied:

    A) over all 4 x 100 = 400 tests per feature family
    B) separately over the predeclared short-delay window
       4 x 20 = 80 tests per feature family

This addresses the false-positive problem created by testing many delays.

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

# =============================================================================
# Frozen settings
# =============================================================================

ALPHA = 0.75
SEED = 42

BURN_IN = 100
K_MAX = 100
SHORT_K_MAX = 20

PROBE_SEED = 74001
PERM_SEED = 74003

# 10,000 gives p-value resolution 1/10001 ≈ 1e-4, sufficient for the
# earliest BH thresholds among 400 tests.
N_PERM = 10_000
PERM_BATCH = 500

FDR_ALPHA = 0.05

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]


# =============================================================================
# Statistics
# =============================================================================

def bh_fdr(p_values):
    """
    Benjamini-Hochberg adjusted p-values (q-values).

    Returns q-values in original order.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    q_ranked = ranked * n / np.arange(1, n + 1, dtype=float)

    # Enforce monotonicity from largest rank backward.
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    return q


def permutation_pvalue_corr2(y_true, pred, rng, n_perm=N_PERM):
    """
    Permutation test for Corr^2.

    Under permutation, the validation target is shuffled relative to the
    fixed prediction vector. Correlation can be computed efficiently via
    centered dot products.

    p = (1 + #null >= observed) / (1 + N_perm)
    """
    y = np.asarray(y_true, dtype=float)
    pvec = np.asarray(pred, dtype=float)

    observed = c2.safe_corr2(y, pvec)

    yc = y - np.mean(y)
    pc = pvec - np.mean(pvec)

    denom = float(np.linalg.norm(yc) * np.linalg.norm(pc))
    if denom <= c2.VAR_TOL:
        return observed, 1.0, 0.0, 0.0

    exceed = 0
    null_sum = 0.0
    null_max = 0.0
    remaining = n_perm

    while remaining > 0:
        b = min(PERM_BATCH, remaining)

        # Build a batch of independently permuted centered targets.
        Yperm = np.empty((b, len(yc)), dtype=float)
        for r in range(b):
            Yperm[r] = yc[rng.permutation(len(yc))]

        corr = (Yperm @ pc) / denom
        null_mc = corr * corr

        exceed += int(np.sum(null_mc >= observed))
        null_sum += float(np.sum(null_mc))
        null_max = max(null_max, float(np.max(null_mc)))

        remaining -= b

    p_value = (1.0 + exceed) / (1.0 + n_perm)
    null_mean = null_sum / n_perm

    return observed, p_value, null_mean, null_max


# =============================================================================
# Reconstruct delayed-memory predictions
# =============================================================================

def calculate_tests(X, probe, n_train, n_val, family_name, rng):
    rows = []

    for j in range(probe.shape[1]):
        print(f"  {family_name}: channel {j} ...")

        for k in range(1, K_MAX + 1):
            train_start = max(BURN_IN, k)

            train_t = np.arange(train_start, n_train, dtype=int)
            val_t = np.arange(n_train, n_train + n_val, dtype=int)

            Xtr = X[train_t]
            ytr = probe[train_t - k, j]

            Xva = X[val_t]
            yva = probe[val_t - k, j]

            pred = c2.fit_predict_memory_readout(
                Xtr,
                ytr,
                Xva,
            )

            raw_mc, p_perm, null_mean, null_max = (
                permutation_pvalue_corr2(
                    yva,
                    pred,
                    rng,
                )
            )

            rows.append({
                "feature_family": family_name,
                "channel": j,
                "delay": k,
                "MC_raw": raw_mc,
                "MC_null_mean": null_mean,
                "MC_corrected": max(raw_mc - null_mean, 0.0),
                "p_perm": p_perm,
                "null_max_seen": null_max,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 126)
    print("WEEK 7 - STEP 7.4C.3")
    print("MULTIPLE-COMPARISON-CORRECTED MEMORY SIGNIFICANCE TEST")
    print("=" * 126)
    print()
    print("Reservoir:")
    print("  ideal/unmeasured CONT")
    print("  fixed initial memory = I/4")
    print("  NO YX45 measurement back-action")
    print()
    print("Feature families:")
    print("  XZ_injection")
    print("  XZ_injection + YX45 expectation")
    print()
    print(f"Delays: 1..{K_MAX}")
    print(f"Short-delay window: 1..{SHORT_K_MAX}")
    print(f"Permutation repetitions per test: {N_PERM}")
    print(f"FDR alpha: {FDR_ALPHA}")
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
    probe = c2.build_distribution_matched_probe(
        angles,
        probe_rng,
    )

    U, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, _ = qrc.build_input_channels(U, probe)

    print(f"QRC unitarity error = {unitary_err:.3e}")
    print()

    X9 = c2.evolve_ideal_features(A_list)

    feature_sets = {
        "XZ_injection": X9[:, :8],
        "XZ_injection_plus_YX45": X9[:, :9],
    }

    parts = []

    for idx, (family, X) in enumerate(feature_sets.items()):
        print(f"Calculating permutation tests: {family}")
        rng = np.random.default_rng(PERM_SEED + idx * 1_000_000)

        df = calculate_tests(
            X,
            probe,
            n_train,
            n_val,
            family,
            rng,
        )
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)

    # ---------------------------------------------------------------------
    # BH FDR over all 400 tests within each feature family
    # ---------------------------------------------------------------------
    out["q_FDR_full"] = np.nan
    out["significant_FDR_full"] = 0

    # Short-window q-values are defined only for k <= SHORT_K_MAX.
    out["q_FDR_short"] = np.nan
    out["significant_FDR_short"] = 0

    for family in out["feature_family"].unique():
        mask = out["feature_family"] == family

        q_full = bh_fdr(
            out.loc[mask, "p_perm"].to_numpy(dtype=float)
        )
        out.loc[mask, "q_FDR_full"] = q_full
        out.loc[mask, "significant_FDR_full"] = (
            q_full <= FDR_ALPHA
        ).astype(int)

        short_mask = mask & (out["delay"] <= SHORT_K_MAX)
        q_short = bh_fdr(
            out.loc[short_mask, "p_perm"].to_numpy(dtype=float)
        )
        out.loc[short_mask, "q_FDR_short"] = q_short
        out.loc[short_mask, "significant_FDR_short"] = (
            q_short <= FDR_ALPHA
        ).astype(int)

    out.to_csv(
        RESULTS / "07_04c3_memory_FDR_by_delay.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    rows = []

    for family in out["feature_family"].unique():
        sf = out[out["feature_family"] == family]

        full_sig = sf[sf["significant_FDR_full"] == 1]
        short = sf[sf["delay"] <= SHORT_K_MAX]
        short_sig = short[short["significant_FDR_short"] == 1]

        rows.append({
            "feature_family": family,
            "n_tests_full": len(sf),
            "n_uncorrected_p_lt_0.05_full": int(
                np.sum(sf["p_perm"] < 0.05)
            ),
            "n_FDR_significant_full": len(full_sig),
            "max_FDR_significant_delay_full": (
                int(full_sig["delay"].max())
                if len(full_sig) else 0
            ),
            "sum_corrected_MC_full": float(
                sf["MC_corrected"].sum()
            ),
            "n_tests_short": len(short),
            "n_uncorrected_p_lt_0.05_short": int(
                np.sum(short["p_perm"] < 0.05)
            ),
            "n_FDR_significant_short": len(short_sig),
            "max_FDR_significant_delay_short": (
                int(short_sig["delay"].max())
                if len(short_sig) else 0
            ),
            "sum_corrected_MC_short": float(
                short["MC_corrected"].sum()
            ),
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        RESULTS / "07_04c3_memory_FDR_summary.csv",
        index=False,
    )

    print()
    print("-" * 126)
    print("SUMMARY")
    print("-" * 126)
    print(summary.to_string(index=False))

    print()
    print("-" * 126)
    print("FDR-SIGNIFICANT FULL-RANGE RESULTS")
    print("-" * 126)

    full_sig = out[out["significant_FDR_full"] == 1].sort_values(
        ["feature_family", "p_perm", "delay"]
    )

    if len(full_sig) == 0:
        print("No channel-delay pair survives full-range FDR correction.")
    else:
        print(
            full_sig[
                [
                    "feature_family",
                    "channel",
                    "delay",
                    "MC_raw",
                    "MC_null_mean",
                    "MC_corrected",
                    "p_perm",
                    "q_FDR_full",
                ]
            ].to_string(index=False)
        )

    print()
    print("-" * 126)
    print("FDR-SIGNIFICANT SHORT-DELAY RESULTS (k <= 20)")
    print("-" * 126)

    short_sig = out[
        (out["delay"] <= SHORT_K_MAX)
        & (out["significant_FDR_short"] == 1)
    ].sort_values(
        ["feature_family", "p_perm", "delay"]
    )

    if len(short_sig) == 0:
        print("No short-delay channel pair survives FDR correction.")
    else:
        print(
            short_sig[
                [
                    "feature_family",
                    "channel",
                    "delay",
                    "MC_raw",
                    "MC_null_mean",
                    "MC_corrected",
                    "p_perm",
                    "q_FDR_short",
                ]
            ].to_string(index=False)
        )

    with open(
        RESULTS / "07_04c3_memory_FDR_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "WEEK 7 STEP 7.4C.3 - MULTIPLE-COMPARISON MEMORY TEST\n"
        )
        fp.write("=" * 96 + "\n\n")
        fp.write(
            f"N_PERM={N_PERM}, FDR alpha={FDR_ALPHA}, "
            f"short K<= {SHORT_K_MAX}\n\n"
        )
        fp.write(summary.to_string(index=False))
        fp.write("\n\nFULL-RANGE FDR SIGNIFICANT:\n")
        fp.write(full_sig.to_string(index=False))
        fp.write("\n\nSHORT-RANGE FDR SIGNIFICANT:\n")
        fp.write(short_sig.to_string(index=False))
        fp.write("\n")

    print()
    print("=" * 126)
    print("Saved:")
    print("  results/07_04c3_memory_FDR_by_delay.csv")
    print("  results/07_04c3_memory_FDR_summary.csv")
    print("  results/07_04c3_memory_FDR_summary.txt")
    print()
    print("Decision rule:")
    print("  interpret reservoir memory only if delays survive FDR correction")
    print("  and form a physically coherent short-delay pattern.")


if __name__ == "__main__":
    main()
