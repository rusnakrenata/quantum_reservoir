"""
Week 7 - Step 7.4B.4
Monte-Carlo contraction/washout analysis for the TWO missing selective modes:

1) direct_local_Y4_X5_selective
2) ancilla_joint_YX_selective

Notation follows Part I, Unit 3:

    D_t <= kappa^t D_0
    tau_mem = -1 / ln(kappa)
    T_w >= ln(epsilon / D_0) / ln(kappa)

For selective measurement there is no single deterministic channel kappa.
Therefore, for every Monte-Carlo trajectory r we calculate:

    kappa_t^(r) = D_t^(r) / D_(t-1)^(r)

and fit an empirical trajectory-level contraction

    ln(D_t^(r) / D_0^(r)) ~= t ln(kappa_eff^(r))

before D_t reaches the numerical floor.

We then compare:
    predicted T_w from kappa_eff^(r)
versus
    empirical stable threshold crossing of D_t^(r).

IMPORTANT:
- common random numbers are used across the five initial states within
  each Monte-Carlo replicate, exactly as in 07_04b1;
- direct-local selective measurement may collapse D_t exactly to zero,
  so an exponential kappa model can be only an approximation;
- the empirical washout remains the primary result for selective modes.

Dependency:
    07_04b1_cont_washout_diagnostic.py
"""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

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
N_MC = 100
MC_SEED_BASE = 74200
EPSILONS = [0.1, 0.05, 0.01, 0.001]

NUMERICAL_FLOOR = 1e-12
MIN_POSITIVE_POINTS_FOR_FIT = 2

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]


# =============================================================================
# Distance helpers
# =============================================================================

def trace_distance(rho, sigma):
    diff = rho - sigma
    vals = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(vals)))


def max_pairwise_trace_distance(states):
    names = list(states.keys())
    vals = []

    for a, b in combinations(names, 2):
        vals.append(trace_distance(states[a], states[b]))

    return float(max(vals)) if vals else 0.0


def stable_threshold_crossing(values, epsilon):
    """
    First t such that all values from t onward are <= epsilon.
    """
    arr = np.asarray(values, dtype=float)

    for t in range(len(arr)):
        if np.all(arr[t:] <= epsilon):
            return int(t)

    return None


# =============================================================================
# One selective Monte-Carlo trajectory
# =============================================================================

def run_selective_trace_trajectory(A_list, n_steps, projectors, rng):
    """
    Propagate all five initial states under:
      - same chronological inputs,
      - same uniform random number u_t at every time step,
      - state-dependent measurement probabilities.

    Returns D_t = maximum pairwise trace distance across the five
    initial-state trajectories.
    """
    rho = {
        name: state.copy()
        for name, state in w.INITIAL_STATES.items()
    }

    D = np.zeros(n_steps, dtype=float)

    for t in range(n_steps):
        u = float(rng.random())
        next_states = {}

        for name, rho_before in rho.items():
            rho_pre_measure, _ = w.evolve_one_input(
                A_list[t],
                rho_before,
            )

            _, _, rho_post = w.sample_projective_outcome(
                rho_pre_measure,
                projectors,
                u,
            )

            next_states[name] = rho_post

        rho = next_states
        D[t] = max_pairwise_trace_distance(rho)

    return D


# =============================================================================
# kappa analysis for one trajectory
# =============================================================================

def fit_trajectory_kappa(D):
    """
    Fit:
        ln(D_t / D_0) ~= t ln(kappa_eff)
    through the origin, using only positive D_t values above numerical floor.

    Here D[0] is the first post-measurement distance reported by this script.
    """
    D = np.asarray(D, dtype=float)
    D0 = float(D[0])

    if D0 <= NUMERICAL_FLOOR:
        return {
            "D0": D0,
            "kappa_eff": 0.0,
            "tau_mem": 0.0,
            "r2_log_fit": np.nan,
            "n_fit_points": 0,
        }

    t_all = np.arange(len(D), dtype=float)

    mask = (
        (t_all > 0)
        & (D > NUMERICAL_FLOOR)
        & np.isfinite(D)
    )

    t = t_all[mask]
    y = np.log(D[mask] / D0)

    if len(t) < MIN_POSITIVE_POINTS_FOR_FIT:
        return {
            "D0": D0,
            "kappa_eff": np.nan,
            "tau_mem": np.nan,
            "r2_log_fit": np.nan,
            "n_fit_points": int(len(t)),
        }

    slope = float(np.dot(t, y) / np.dot(t, t))
    kappa = float(np.exp(slope))

    yhat = slope * t
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    if 0 < kappa < 1:
        tau = float(-1.0 / np.log(kappa))
    elif kappa == 0:
        tau = 0.0
    else:
        tau = np.inf

    return {
        "D0": D0,
        "kappa_eff": kappa,
        "tau_mem": tau,
        "r2_log_fit": r2,
        "n_fit_points": int(len(t)),
    }


def stepwise_kappa_summary(D):
    """
    kappa_t = D_t / D_(t-1)

    Ratios are included while D_(t-1) is above numerical floor.
    If D_t reaches zero, kappa_t = 0 for that collapse step.
    Later 0/0 steps are excluded.
    """
    D = np.asarray(D, dtype=float)

    kappas = []

    for t in range(1, len(D)):
        prev = D[t - 1]
        curr = D[t]

        if prev <= NUMERICAL_FLOOR:
            continue

        if curr <= NUMERICAL_FLOOR:
            kappas.append(0.0)
        else:
            kappas.append(float(curr / prev))

    arr = np.asarray(kappas, dtype=float)

    if len(arr) == 0:
        return {
            "median_kappa_t": np.nan,
            "p90_kappa_t": np.nan,
            "max_kappa_t": np.nan,
            "fraction_kappa_t_lt_1": np.nan,
            "fraction_kappa_t_gt_1": np.nan,
            "n_stepwise_kappa": 0,
        }

    return {
        "median_kappa_t": float(np.median(arr)),
        "p90_kappa_t": float(np.quantile(arr, 0.90)),
        "max_kappa_t": float(np.max(arr)),
        "fraction_kappa_t_lt_1": float(np.mean(arr < 1.0)),
        "fraction_kappa_t_gt_1": float(np.mean(arr > 1.0)),
        "n_stepwise_kappa": int(len(arr)),
    }


def predicted_washout(D0, kappa, epsilon):
    """
    Solve:
        kappa^T D0 <= epsilon

    giving:
        T >= ln(epsilon / D0) / ln(kappa)
    """
    if D0 <= epsilon:
        return 0

    if not np.isfinite(kappa):
        return None

    if kappa == 0:
        return 1

    if not (0 < kappa < 1):
        return None

    T = np.log(epsilon / D0) / np.log(kappa)
    return int(np.ceil(T))


# =============================================================================
# Aggregation
# =============================================================================

def q(values, p):
    arr = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)],
        dtype=float,
    )
    if len(arr) == 0:
        return np.nan
    return float(np.quantile(arr, p))


def med(values):
    return q(values, 0.50)


def mean(values):
    arr = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)],
        dtype=float,
    )
    if len(arr) == 0:
        return np.nan
    return float(np.mean(arr))


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 122)
    print("WEEK 7 - STEP 7.4B.4")
    print("MONTE-CARLO WASHOUT / kappa ANALYSIS FOR THE TWO SELECTIVE MEASUREMENT MODES")
    print("=" * 122)
    print()
    print("Frozen:")
    print(f"  F4 | alpha={ALPHA} | seed={SEED}")
    print(f"  Monte-Carlo trajectories={N_MC}")
    print(f"  epsilons={EPSILONS}")
    print()
    print("Unit-3 notation:")
    print("  D_t <= kappa^t D_0")
    print("  tau_mem = -1 / ln(kappa)")
    print("  T_w >= ln(epsilon / D_0) / ln(kappa)")
    print()
    print("For selective measurement, kappa is empirical and trajectory-specific.")
    print("Empirical stable threshold crossing remains the primary washout result.")
    print()

    work, train, val, cols = qrc.load_data()
    n_steps = len(train) + len(val)

    angles_all = qrc.make_input_angles(work, cols)
    U, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, _ = qrc.build_input_channels(U, angles_all)

    print(f"QRC unitarity error = {unitary_err:.3e}")
    print(f"Chronological CONT steps = {n_steps}")
    print()

    modes = [
        (
            "direct_local_Y4_X5_selective",
            w.LOCAL_PROJECTORS,
        ),
        (
            "ancilla_joint_YX_selective",
            w.JOINT_PROJECTORS,
        ),
    ]

    trajectory_rows = []
    threshold_rows = []
    timewise_rows = []

    for mode, projectors in modes:
        print("=" * 122)
        print(f"RUNNING MODE = {mode}")
        print("=" * 122)

        all_D = []

        for rep in range(N_MC):
            rng = np.random.default_rng(
                MC_SEED_BASE + rep * 1009
            )

            D = run_selective_trace_trajectory(
                A_list,
                n_steps,
                projectors,
                rng,
            )
            all_D.append(D)

            fit = fit_trajectory_kappa(D)
            step = stepwise_kappa_summary(D)

            trajectory_rows.append({
                "mode": mode,
                "mc_rep": rep,
                "D0": fit["D0"],
                "kappa_eff": fit["kappa_eff"],
                "tau_mem": fit["tau_mem"],
                "r2_log_fit": fit["r2_log_fit"],
                "n_fit_points": fit["n_fit_points"],
                **step,
            })

            for eps in EPSILONS:
                pred = predicted_washout(
                    fit["D0"],
                    fit["kappa_eff"],
                    eps,
                )
                emp = stable_threshold_crossing(D, eps)

                threshold_rows.append({
                    "mode": mode,
                    "mc_rep": rep,
                    "epsilon": eps,
                    "predicted_Tw": (
                        np.nan if pred is None else pred
                    ),
                    "empirical_Tw": (
                        np.nan if emp is None else emp
                    ),
                    "empirical_achieved": int(emp is not None),
                })

        Dmat = np.stack(all_D, axis=0)

        for t in range(n_steps):
            vals = Dmat[:, t]
            timewise_rows.append({
                "mode": mode,
                "t": t,
                "median_D": float(np.median(vals)),
                "p90_D": float(np.quantile(vals, 0.90)),
                "mean_D": float(np.mean(vals)),
            })

    traj_df = pd.DataFrame(trajectory_rows)
    thr_df = pd.DataFrame(threshold_rows)
    time_df = pd.DataFrame(timewise_rows)

    traj_df.to_csv(
        RESULTS / "07_04b4_selective_mc_kappa_trajectories.csv",
        index=False,
    )
    thr_df.to_csv(
        RESULTS / "07_04b4_selective_mc_predicted_vs_empirical_Tw.csv",
        index=False,
    )
    time_df.to_csv(
        RESULTS / "07_04b4_selective_mc_timewise_D.csv",
        index=False,
    )

    # ---------------------------------------------------------------------
    # Compact summaries
    # ---------------------------------------------------------------------
    summary_rows = []

    for mode in traj_df["mode"].unique():
        sub = traj_df[traj_df["mode"] == mode]

        print()
        print("-" * 122)
        print(f"MODE = {mode}")
        print("-" * 122)
        print(
            f"kappa_eff: median={med(sub['kappa_eff']):.9f}, "
            f"p90={q(sub['kappa_eff'], 0.90):.9f}"
        )
        print(
            f"tau_mem:   median={med(sub['tau_mem']):.3f}, "
            f"p90={q(sub['tau_mem'], 0.90):.3f}"
        )
        print(
            f"log-fit R2: median={med(sub['r2_log_fit']):.6f}, "
            f"p10={q(sub['r2_log_fit'], 0.10):.6f}"
        )
        print(
            f"stepwise kappa_t median across trajectories="
            f"{med(sub['median_kappa_t']):.6f}"
        )
        print(
            f"fraction kappa_t<1: median="
            f"{med(sub['fraction_kappa_t_lt_1']):.3f}"
        )
        print(
            f"fraction kappa_t>1: median="
            f"{med(sub['fraction_kappa_t_gt_1']):.3f}"
        )
        print()
        print(
            "epsilon | achieved | predicted Tw med/p90 | "
            "empirical Tw med/p90"
        )
        print("-" * 78)

        for eps in EPSILONS:
            s = thr_df[
                (thr_df["mode"] == mode)
                & (thr_df["epsilon"] == eps)
            ]

            achieved = float(s["empirical_achieved"].mean())

            pred_vals = s["predicted_Tw"].to_numpy(dtype=float)
            emp_vals = s["empirical_Tw"].to_numpy(dtype=float)

            pred_med = med(pred_vals)
            pred_p90 = q(pred_vals, 0.90)
            emp_med = med(emp_vals)
            emp_p90 = q(emp_vals, 0.90)

            print(
                f"{eps:7g} | {achieved:8.3f} | "
                f"{pred_med:8.1f}/{pred_p90:8.1f} | "
                f"{emp_med:8.1f}/{emp_p90:8.1f}"
            )

            summary_rows.append({
                "mode": mode,
                "epsilon": eps,
                "empirical_achieved_fraction": achieved,
                "median_kappa_eff": med(sub["kappa_eff"]),
                "p90_kappa_eff": q(sub["kappa_eff"], 0.90),
                "median_tau_mem": med(sub["tau_mem"]),
                "p90_tau_mem": q(sub["tau_mem"], 0.90),
                "median_log_fit_R2": med(sub["r2_log_fit"]),
                "median_fraction_kappa_t_lt_1": med(
                    sub["fraction_kappa_t_lt_1"]
                ),
                "median_fraction_kappa_t_gt_1": med(
                    sub["fraction_kappa_t_gt_1"]
                ),
                "predicted_Tw_median": pred_med,
                "predicted_Tw_p90": pred_p90,
                "empirical_Tw_median": emp_med,
                "empirical_Tw_p90": emp_p90,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        RESULTS / "07_04b4_selective_mc_summary.csv",
        index=False,
    )

    with open(
        RESULTS / "07_04b4_selective_mc_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "WEEK 7 STEP 7.4B.4 - SELECTIVE MONTE-CARLO WASHOUT\n"
        )
        fp.write("=" * 90 + "\n\n")
        fp.write(summary_df.to_string(index=False))
        fp.write("\n")

    print()
    print("=" * 122)
    print("Saved:")
    print("  results/07_04b4_selective_mc_kappa_trajectories.csv")
    print("  results/07_04b4_selective_mc_predicted_vs_empirical_Tw.csv")
    print("  results/07_04b4_selective_mc_timewise_D.csv")
    print("  results/07_04b4_selective_mc_summary.csv")
    print("  results/07_04b4_selective_mc_summary.txt")
    print()
    print("Do NOT treat selective kappa_eff as a deterministic channel constant.")
    print("Compare it with the empirical Monte-Carlo washout times.")


if __name__ == "__main__":
    main()
