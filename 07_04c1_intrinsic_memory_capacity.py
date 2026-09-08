"""
Week 7 - Step 7.4C.1
Intrinsic measurement-conditioned memory capacity.

Purpose
-------
Measure how much delayed input information is ACCESSIBLE through the measured
QRC feature vector x(t), under the four CONT measurement models.

Foundations notation
--------------------
For input channel j and delay k:

    MC_{j,k} = Corr^2[ u_j(t-k), uhat_{j,k}(t) ]

where uhat_{j,k}(t) is produced by a separate linear/ridge readout from x(t).

For each channel:

    MC_j = sum_{k=1}^{K} MC_{j,k}

and an aggregate score is also reported:

    MC_total = sum_j MC_j

Important experimental design
-----------------------------
The original insurance inputs are temporally structured (weekday periodicity,
claims persistence, holidays, etc.). Directly using them for memory capacity
could confuse INPUT AUTOCORRELATION with RESERVOIR MEMORY.

Therefore this script:
1) builds the exact F4 injection angles used by the QRC;
2) independently permutes each injection-angle channel in time;
3) preserves each channel's empirical marginal distribution and scale;
4) destroys its chronological autocorrelation;
5) uses the resulting distribution-matched randomized sequence as the memory
   probe.

This makes delayed reconstruction a much cleaner test of intrinsic reservoir
memory while keeping the probe amplitudes realistic for our architecture.

Modes
-----
0) ideal_unmeasured
1) direct_local_Y4_X5_selective
2) ancilla_joint_YX_nonselective
3) ancilla_joint_YX_selective

Measurement dynamics
--------------------
Features x(t) are exact expectation-value features immediately BEFORE the
measurement back-action at time t. Finite-shot noise is OFF in this step so
that memory dynamics and measurement back-action are isolated.

Selective modes use Monte-Carlo trajectories.

Common benchmark choices
------------------------
- fixed initial memory state: I/4
- common burn-in: 100 timesteps
- maximum delay: K=100
- selective trajectories: 20
- stable ridge readout with train-only feature scaling and near-constant
  feature removal

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
    raise FileNotFoundError(
        f"Missing {B1.name}. Keep it beside this script."
    )


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

N_MC = 20
MC_SEED_BASE = 75400
PROBE_SEED = 74001

RIDGE_ALPHA = 1e-6
VAR_TOL = 1e-12

HORIZON_THRESHOLDS = [0.5, 0.1]
PRINT_DELAYS = [1, 2, 5, 10, 20, 50, 100]

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]


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

    sa = float(np.std(a))
    sb = float(np.std(b))

    if sa <= VAR_TOL or sb <= VAR_TOL:
        return 0.0

    r = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(r):
        return 0.0

    return r * r


def autocorrelation_audit(probe, max_lag=20):
    rows = []

    for j in range(probe.shape[1]):
        x = probe[:, j]

        for lag in range(1, max_lag + 1):
            raw = np.corrcoef(x[lag:], x[:-lag])[0, 1]
            if not np.isfinite(raw):
                raw = 0.0

            rows.append({
                "channel": j,
                "lag": lag,
                "corr": float(raw),
                "abs_corr": float(abs(raw)),
            })

    return pd.DataFrame(rows)


def evolve_probe_features(A_list, mode, rng=None):
    rho = w.INITIAL_STATES["I/4"].copy()
    n_steps = len(A_list)

    feature_rows = []

    for t in range(n_steps):
        rho_pre_measure, x = w.evolve_one_input(A_list[t], rho)
        feature_rows.append(np.asarray(x, dtype=float))

        if mode == "ideal_unmeasured":
            rho = rho_pre_measure

        elif mode == "ancilla_joint_YX_nonselective":
            rho = w.joint_nonselective_channel(rho_pre_measure)

        elif mode == "direct_local_Y4_X5_selective":
            if rng is None:
                raise ValueError("Selective mode requires rng.")
            u = float(rng.random())
            _, _, rho = w.sample_projective_outcome(
                rho_pre_measure,
                w.LOCAL_PROJECTORS,
                u,
            )

        elif mode == "ancilla_joint_YX_selective":
            if rng is None:
                raise ValueError("Selective mode requires rng.")
            u = float(rng.random())
            _, _, rho = w.sample_projective_outcome(
                rho_pre_measure,
                w.JOINT_PROJECTORS,
                u,
            )

        else:
            raise ValueError(mode)

    return np.vstack(feature_rows)


def fit_predict_memory_readout(Xtr, ytr, Xva):
    Xtr = np.asarray(Xtr, dtype=float)
    Xva = np.asarray(Xva, dtype=float)
    ytr = np.asarray(ytr, dtype=float)

    mu = np.mean(Xtr, axis=0)
    sd = np.std(Xtr, axis=0, ddof=0)

    keep = sd > VAR_TOL

    if not np.any(keep):
        return np.full(len(Xva), float(np.mean(ytr)))

    Xtr_z = (Xtr[:, keep] - mu[keep]) / sd[keep]
    Xva_z = (Xva[:, keep] - mu[keep]) / sd[keep]

    model = Ridge(
        alpha=RIDGE_ALPHA,
        fit_intercept=True,
    )
    model.fit(Xtr_z, ytr)

    return model.predict(Xva_z)


def calculate_memory_curve(
    X,
    probe,
    n_train,
    n_val,
    mode,
    mc_rep,
):
    rows = []

    for j in range(probe.shape[1]):
        for k in range(1, K_MAX + 1):

            train_start = max(BURN_IN, k)

            train_t = np.arange(
                train_start,
                n_train,
                dtype=int,
            )
            val_t = np.arange(
                n_train,
                n_train + n_val,
                dtype=int,
            )

            Xtr = X[train_t]
            ytr = probe[train_t - k, j]

            Xva = X[val_t]
            yva = probe[val_t - k, j]

            pred = fit_predict_memory_readout(
                Xtr,
                ytr,
                Xva,
            )

            mc = safe_corr2(yva, pred)

            rows.append({
                "mode": mode,
                "mc_rep": mc_rep,
                "channel": j,
                "delay": k,
                "MC": mc,
                "n_train_readout": len(train_t),
                "n_validation": len(val_t),
            })

    return pd.DataFrame(rows)


def summarize_one_curve(curve_df):
    rows = []

    mode = curve_df["mode"].iloc[0]
    rep = curve_df["mc_rep"].iloc[0]

    total = 0.0

    for j in sorted(curve_df["channel"].unique()):
        sub = curve_df[curve_df["channel"] == j].sort_values("delay")
        mc_j = float(sub["MC"].sum())
        total += mc_j

        row = {
            "mode": mode,
            "mc_rep": rep,
            "channel": j,
            "MC_sum": mc_j,
        }

        for thr in HORIZON_THRESHOLDS:
            good = sub[sub["MC"] >= thr]["delay"].to_numpy(dtype=int)
            row[f"horizon_MC_ge_{thr}"] = (
                int(np.max(good)) if len(good) else 0
            )

        rows.append(row)

    agg = {
        "mode": mode,
        "mc_rep": rep,
        "channel": "ALL",
        "MC_sum": float(total),
    }

    for thr in HORIZON_THRESHOLDS:
        agg[f"horizon_MC_ge_{thr}"] = np.nan

    rows.append(agg)

    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 126)
    print("WEEK 7 - STEP 7.4C.1")
    print("INTRINSIC MEASUREMENT-CONDITIONED MEMORY CAPACITY")
    print("=" * 126)
    print()
    print("Definition:")
    print("  MC_{j,k} = Corr^2[ u_j(t-k), uhat_{j,k}(t) ]")
    print("  MC_j     = sum_k MC_{j,k}")
    print()
    print("Design:")
    print("  distribution-matched randomized F4 injection-angle probe")
    print(f"  alpha={ALPHA} | reservoir seed={SEED}")
    print(f"  fixed initial memory=I/4 | common burn-in={BURN_IN}")
    print(f"  delays k=1..{K_MAX}")
    print(f"  selective MC trajectories={N_MC}")
    print(f"  Ridge alpha={RIDGE_ALPHA} | variance tolerance={VAR_TOL}")
    print("  finite-shot noise=OFF")
    print()

    work, train, val, cols = qrc.load_data()

    n_train = len(train)
    n_val = len(val)
    n_steps = n_train + n_val

    angles_all = np.asarray(
        qrc.make_input_angles(work, cols),
        dtype=float,
    )[:n_steps]

    n_channels = angles_all.shape[1]

    print(f"Train={n_train}, validation={n_val}, total={n_steps}")
    print(f"Number of actual injection-angle channels={n_channels}")
    print()

    probe_rng = np.random.default_rng(PROBE_SEED)
    probe = build_distribution_matched_probe(
        angles_all,
        probe_rng,
    )

    marginal_error = float(
        np.max(
            np.abs(
                np.sort(probe, axis=0)
                - np.sort(angles_all, axis=0)
            )
        )
    )

    ac_df = autocorrelation_audit(
        probe,
        max_lag=20,
    )

    print(f"Marginal-distribution preservation max error = {marginal_error:.3e}")
    print(
        "Randomized-probe max |autocorrelation| over lags 1..20 = "
        f"{ac_df['abs_corr'].max():.6f}"
    )
    print()

    ac_df.to_csv(
        RESULTS / "07_04c1_probe_autocorrelation_audit.csv",
        index=False,
    )

    pd.DataFrame(
        probe,
        columns=[
            f"injection_angle_{j}"
            for j in range(n_channels)
        ],
    ).to_csv(
        RESULTS / "07_04c1_randomized_probe_angles.csv",
        index=False,
    )

    U, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, _ = qrc.build_input_channels(
        U,
        probe,
    )

    print(f"QRC unitarity error = {unitary_err:.3e}")
    print()

    modes = [
        "ideal_unmeasured",
        "ancilla_joint_YX_nonselective",
        "direct_local_Y4_X5_selective",
        "ancilla_joint_YX_selective",
    ]

    curve_parts = []
    replicate_summary_parts = []

    for mode in modes[:2]:
        print(f"Running deterministic memory capacity: {mode} ...")

        X = evolve_probe_features(
            A_list,
            mode,
        )

        curve = calculate_memory_curve(
            X,
            probe,
            n_train,
            n_val,
            mode,
            mc_rep=-1,
        )

        curve_parts.append(curve)
        replicate_summary_parts.append(
            summarize_one_curve(curve)
        )

    for mode in modes[2:]:
        print(f"Running selective memory capacity: {mode} ...")

        for rep in range(N_MC):
            rng = np.random.default_rng(
                MC_SEED_BASE + rep * 1009
            )

            X = evolve_probe_features(
                A_list,
                mode,
                rng=rng,
            )

            curve = calculate_memory_curve(
                X,
                probe,
                n_train,
                n_val,
                mode,
                mc_rep=rep,
            )

            curve_parts.append(curve)
            replicate_summary_parts.append(
                summarize_one_curve(curve)
            )

    curve_df = pd.concat(
        curve_parts,
        ignore_index=True,
    )
    rep_summary_df = pd.concat(
        replicate_summary_parts,
        ignore_index=True,
    )

    curve_df.to_csv(
        RESULTS / "07_04c1_memory_capacity_by_delay_raw.csv",
        index=False,
    )
    rep_summary_df.to_csv(
        RESULTS / "07_04c1_memory_capacity_replicate_summary.csv",
        index=False,
    )

    summary_rows = []

    for mode in modes:
        sub_mode = curve_df[curve_df["mode"] == mode]

        for j in sorted(sub_mode["channel"].unique()):
            sub_ch = sub_mode[sub_mode["channel"] == j]

            for k in range(1, K_MAX + 1):
                vals = sub_ch[
                    sub_ch["delay"] == k
                ]["MC"].to_numpy(dtype=float)

                summary_rows.append({
                    "mode": mode,
                    "channel": j,
                    "delay": k,
                    "MC_mean": float(np.mean(vals)),
                    "MC_median": float(np.median(vals)),
                    "MC_p10": float(np.quantile(vals, 0.10)),
                    "MC_p90": float(np.quantile(vals, 0.90)),
                    "n_replicates": len(vals),
                })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        RESULTS / "07_04c1_memory_capacity_by_delay_summary.csv",
        index=False,
    )

    global_rows = []

    for mode in modes:
        sub = rep_summary_df[
            rep_summary_df["mode"] == mode
        ]

        for channel in list(range(n_channels)) + ["ALL"]:
            s = sub[sub["channel"] == channel]

            vals = s["MC_sum"].to_numpy(dtype=float)

            row = {
                "mode": mode,
                "channel": channel,
                "MC_sum_mean": float(np.mean(vals)),
                "MC_sum_median": float(np.median(vals)),
                "MC_sum_p10": float(np.quantile(vals, 0.10)),
                "MC_sum_p90": float(np.quantile(vals, 0.90)),
                "n_replicates": len(vals),
            }

            if channel != "ALL":
                for thr in HORIZON_THRESHOLDS:
                    h = s[
                        f"horizon_MC_ge_{thr}"
                    ].to_numpy(dtype=float)

                    row[
                        f"horizon_MC_ge_{thr}_median"
                    ] = float(np.median(h))
                    row[
                        f"horizon_MC_ge_{thr}_p90"
                    ] = float(np.quantile(h, 0.90))

            global_rows.append(row)

    global_df = pd.DataFrame(global_rows)

    global_df.to_csv(
        RESULTS / "07_04c1_memory_capacity_global_summary.csv",
        index=False,
    )

    for mode in modes:
        print()
        print("-" * 126)
        print(f"MODE = {mode}")
        print("-" * 126)

        g = global_df[
            global_df["mode"] == mode
        ]

        print("Integrated memory capacity:")
        print(
            g[
                [
                    "channel",
                    "MC_sum_median",
                    "MC_sum_p10",
                    "MC_sum_p90",
                ]
            ].to_string(index=False)
        )

        print()
        print("Selected delays: median MC_{j,k}")

        selected = summary_df[
            (summary_df["mode"] == mode)
            & (summary_df["delay"].isin(PRINT_DELAYS))
        ][
            [
                "channel",
                "delay",
                "MC_median",
                "MC_p10",
                "MC_p90",
            ]
        ]

        print(selected.to_string(index=False))

    with open(
        RESULTS / "07_04c1_memory_capacity_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "WEEK 7 STEP 7.4C.1 - INTRINSIC MEMORY CAPACITY\n"
        )
        fp.write("=" * 96 + "\n\n")
        fp.write(
            "MC_{j,k}=Corr^2[u_j(t-k), uhat_{j,k}(t)]\n"
        )
        fp.write(
            "Probe: independently permuted actual F4 injection angles\n"
        )
        fp.write(
            f"Burn-in={BURN_IN}, K_MAX={K_MAX}, "
            f"selective MC={N_MC}, Ridge alpha={RIDGE_ALPHA}\n\n"
        )
        fp.write(global_df.to_string(index=False))
        fp.write("\n")

    print()
    print("=" * 126)
    print("Saved:")
    print("  results/07_04c1_probe_autocorrelation_audit.csv")
    print("  results/07_04c1_randomized_probe_angles.csv")
    print("  results/07_04c1_memory_capacity_by_delay_raw.csv")
    print("  results/07_04c1_memory_capacity_by_delay_summary.csv")
    print("  results/07_04c1_memory_capacity_replicate_summary.csv")
    print("  results/07_04c1_memory_capacity_global_summary.csv")
    print("  results/07_04c1_memory_capacity_summary.txt")
    print()
    print("Interpret MC together with washout:")
    print("  useful reservoir = forget arbitrary initialization")
    print("                     while retaining reconstructable delayed inputs.")


if __name__ == "__main__":
    main()
