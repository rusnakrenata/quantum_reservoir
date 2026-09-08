"""
Week 7 - Step 7.4B.1 (REVISED)
CONT washout / initialization-forgetting diagnostic under three measurement models.

Purpose
-------
Compare initialization forgetting under:

0) IDEAL / UNMEASURED CONT reference

1) DIRECT LOCAL measurement of memory qubits:
       q4 measured in Y
       q5 measured in X
   A single trajectory resolves one of four outcomes:
       (++), (+-), (-+), (--)
   and carries the corresponding conditional post-measurement state.
   This is a SELECTIVE local measurement trajectory.

2) ANCILLA JOINT YX measurement, NON-SELECTIVE:
       P = Y4 tensor X5
       rho -> Pi+ rho Pi+ + Pi- rho Pi-
   The ancilla outcome is averaged/ignored when propagating the ensemble state.

3) ANCILLA JOINT YX measurement, SELECTIVE:
       sample + or - from p± = Tr(Pi± rho)
       carry only the corresponding conditional state
           rho± = Pi± rho Pi± / p±
   This is simulated with Monte-Carlo trajectories.

Important
---------
Direct local Y/X measurement and joint YX measurement have the SAME
YX expectation statistics after parity grouping, but DIFFERENT back-action:
the local measurement resolves four one-dimensional product outcomes,
while the joint ancilla measurement resolves only two parity subspaces.

Washout question
----------------
How quickly do trajectories starting from very different initial memory
states cease to depend on that initialization?

Initial states:
    |00><00|
    |11><11|
    |++><++|
    Bell Phi+
    maximally mixed I/4

For deterministic modes, the script reports the earliest t after which
the maximum pairwise distance stays permanently below epsilon.

For selective stochastic modes, all initial states in one MC replicate use
the SAME uniform random number at each time step ("common random numbers").
This coupling is intentional: it suppresses irrelevant Monte-Carlo noise and
asks whether two systems driven by the same input sequence and same underlying
measurement randomness forget their different initial conditions.

For each selective mode, the script reports:
    - median and 90th-percentile distance curves over MC trajectories
    - distribution of stable washout times
    - fraction of MC trajectories that achieve each epsilon threshold

No washout length is frozen automatically.

Dependency
----------
Keep 07_02b_memory_pair_isolation.py beside this script.
"""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Frozen configuration
# =============================================================================

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "07_02b_memory_pair_isolation.py"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing {BASE_SCRIPT.name}. Keep it beside this washout script."
    )

spec = importlib.util.spec_from_file_location("qrc_step_72b", BASE_SCRIPT)
qrc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qrc)

RESULTS = Path("results")

ALPHA = 0.75
SEED = 42
N_MC = 100
MC_SEED_BASE = 74100

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]

FAMILY = "XZinj_plus_YX45"
FEATURES = qrc.FAMILIES[FAMILY]

THRESHOLDS = [0.1, 0.05, 0.01, 0.001]


# =============================================================================
# Operators on the two memory qubits q4,q5
# =============================================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
I4 = np.eye(4, dtype=complex)

P_YX = np.kron(Y, X)

# Joint-parity projectors (rank 2)
PI_JOINT_PLUS = 0.5 * (I4 + P_YX)
PI_JOINT_MINUS = 0.5 * (I4 - P_YX)

# Local single-qubit projectors
PI_Y_PLUS = 0.5 * (I2 + Y)
PI_Y_MINUS = 0.5 * (I2 - Y)
PI_X_PLUS = 0.5 * (I2 + X)
PI_X_MINUS = 0.5 * (I2 - X)

# Direct local measurement resolves FOUR outcomes.
LOCAL_PROJECTORS = {
    "Y+_X+": np.kron(PI_Y_PLUS, PI_X_PLUS),
    "Y+_X-": np.kron(PI_Y_PLUS, PI_X_MINUS),
    "Y-_X+": np.kron(PI_Y_MINUS, PI_X_PLUS),
    "Y-_X-": np.kron(PI_Y_MINUS, PI_X_MINUS),
}

# Joint ancilla measurement resolves TWO outcomes.
JOINT_PROJECTORS = {
    "+": PI_JOINT_PLUS,
    "-": PI_JOINT_MINUS,
}


# =============================================================================
# Initial memory states
# =============================================================================

ket0 = np.array([1.0, 0.0], dtype=complex)
ket1 = np.array([0.0, 1.0], dtype=complex)
ketp = (ket0 + ket1) / np.sqrt(2.0)

ket00 = np.kron(ket0, ket0)
ket11 = np.kron(ket1, ket1)
ketpp = np.kron(ketp, ketp)
ket_phi = (ket00 + ket11) / np.sqrt(2.0)


def pure_density(ket):
    return np.outer(ket, ket.conj())


INITIAL_STATES = {
    "00": pure_density(ket00),
    "11": pure_density(ket11),
    "++": pure_density(ketpp),
    "BellPhi+": pure_density(ket_phi),
    "I/4": I4 / 4.0,
}


# =============================================================================
# General helpers
# =============================================================================

def normalize_density(rho):
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) < 1e-15:
        raise RuntimeError("Encountered zero-trace state.")
    return rho / tr


def trace_distance(rho, sigma):
    delta = 0.5 * ((rho - sigma) + (rho - sigma).conj().T)
    eigvals = np.linalg.eigvalsh(delta)
    return float(0.5 * np.sum(np.abs(eigvals)))


def max_pairwise_trace_distance(states_by_name):
    names = list(states_by_name)
    vals = [
        trace_distance(states_by_name[a], states_by_name[b])
        for a, b in combinations(names, 2)
    ]
    return float(max(vals)) if vals else 0.0


def max_pairwise_feature_distance(features_by_name):
    names = list(features_by_name)
    vals = []
    for a, b in combinations(names, 2):
        xa = np.asarray(features_by_name[a], dtype=float)
        xb = np.asarray(features_by_name[b], dtype=float)
        vals.append(float(np.linalg.norm(xa - xb)))
    return float(max(vals)) if vals else 0.0


def stable_threshold_crossing(values, threshold):
    """
    Earliest t such that every later value remains <= threshold.
    """
    arr = np.asarray(values, dtype=float)
    suffix_max = np.maximum.accumulate(arr[::-1])[::-1]
    idx = np.flatnonzero(suffix_max <= threshold)
    return None if len(idx) == 0 else int(idx[0])


def projective_probabilities(rho, projectors):
    probs = np.asarray(
        [float(np.real(np.trace(M @ rho))) for M in projectors.values()],
        dtype=float,
    )
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs


def sample_projective_outcome(rho, projectors, u):
    """
    Select one projective outcome using supplied common uniform u in [0,1).

    Returns
    -------
    label, probability, normalized conditional state
    """
    labels = list(projectors.keys())
    probs = projective_probabilities(rho, projectors)
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, u, side="right"))
    idx = min(idx, len(labels) - 1)

    label = labels[idx]
    p = probs[idx]
    M = projectors[label]

    rho_u = M @ rho @ M
    if p <= 1e-15:
        raise RuntimeError("Sampled a numerically zero-probability outcome.")
    rho_post = normalize_density(rho_u / p)

    return label, p, rho_post


def joint_nonselective_channel(rho):
    out = (
        PI_JOINT_PLUS @ rho @ PI_JOINT_PLUS
        + PI_JOINT_MINUS @ rho @ PI_JOINT_MINUS
    )
    return normalize_density(out)


def local_nonselective_channel(rho):
    """
    Included only as an algebra/reference helper.
    This is NOT the direct-local trajectory propagated in the main comparison.
    """
    out = np.zeros_like(rho)
    for M in LOCAL_PROJECTORS.values():
        out += M @ rho @ M
    return normalize_density(out)


# =============================================================================
# Operator sanity checks
# =============================================================================

def operator_audits():
    errs = {}

    errs["joint_projector_completeness"] = float(
        np.max(np.abs(PI_JOINT_PLUS + PI_JOINT_MINUS - I4))
    )
    errs["joint_projector_orthogonality"] = float(
        np.max(np.abs(PI_JOINT_PLUS @ PI_JOINT_MINUS))
    )

    local_sum = sum(LOCAL_PROJECTORS.values())
    errs["local_projector_completeness"] = float(
        np.max(np.abs(local_sum - I4))
    )

    local_orth = 0.0
    vals = list(LOCAL_PROJECTORS.values())
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            local_orth = max(
                local_orth,
                float(np.max(np.abs(vals[i] @ vals[j]))),
            )
    errs["local_projector_orthogonality"] = local_orth

    # Important identity: grouping local outcomes by parity reproduces
    # the JOINT projectors as probabilities, despite different post states.
    local_plus = (
        LOCAL_PROJECTORS["Y+_X+"]
        + LOCAL_PROJECTORS["Y-_X-"]
    )
    local_minus = (
        LOCAL_PROJECTORS["Y+_X-"]
        + LOCAL_PROJECTORS["Y-_X+"]
    )

    errs["local_parity_plus_vs_joint_plus"] = float(
        np.max(np.abs(local_plus - PI_JOINT_PLUS))
    )
    errs["local_parity_minus_vs_joint_minus"] = float(
        np.max(np.abs(local_minus - PI_JOINT_MINUS))
    )

    return errs


# =============================================================================
# Readout features before measurement back-action
# =============================================================================

def evolve_one_input(A, rho_before):
    rho_i, rho_m_out = qrc.final_reduced_states(A, rho_before)
    feat = qrc.extract_feature_row(
        A,
        rho_before,
        rho_i,
        rho_m_out,
    )
    x = [feat[f] for f in FEATURES]
    return rho_m_out, x


# =============================================================================
# Deterministic modes
# =============================================================================

def run_ideal_unmeasured(A_list, n_steps):
    rho = {name: state.copy() for name, state in INITIAL_STATES.items()}
    rows = []

    for t in range(n_steps):
        next_states = {}
        features = {}

        for name, rho_before in rho.items():
            rho_pre_measure, x = evolve_one_input(A_list[t], rho_before)
            features[name] = x
            next_states[name] = rho_pre_measure

        rows.append({
            "t": t,
            "max_pairwise_trace_distance": max_pairwise_trace_distance(next_states),
            "max_pairwise_feature_distance": max_pairwise_feature_distance(features),
        })
        rho = next_states

    return pd.DataFrame(rows)


def run_joint_nonselective(A_list, n_steps):
    rho = {name: state.copy() for name, state in INITIAL_STATES.items()}
    rows = []

    for t in range(n_steps):
        next_states = {}
        features = {}

        for name, rho_before in rho.items():
            rho_pre_measure, x = evolve_one_input(A_list[t], rho_before)
            features[name] = x
            next_states[name] = joint_nonselective_channel(rho_pre_measure)

        rows.append({
            "t": t,
            "max_pairwise_trace_distance": max_pairwise_trace_distance(next_states),
            "max_pairwise_feature_distance": max_pairwise_feature_distance(features),
        })
        rho = next_states

    return pd.DataFrame(rows)


# =============================================================================
# Selective stochastic modes
# =============================================================================

def run_selective_coupled_trajectory(A_list, n_steps, projectors, rng):
    """
    Run all five initial states under the same input sequence and the same
    underlying uniform random number at each t.

    Each state still uses ITS OWN outcome probabilities.
    """
    rho = {name: state.copy() for name, state in INITIAL_STATES.items()}
    rows = []

    for t in range(n_steps):
        # One common random number for this time step.
        u = float(rng.random())

        next_states = {}
        features = {}
        outcomes = {}

        for name, rho_before in rho.items():
            rho_pre_measure, x = evolve_one_input(A_list[t], rho_before)
            features[name] = x

            label, p, rho_post = sample_projective_outcome(
                rho_pre_measure,
                projectors,
                u,
            )
            outcomes[name] = label
            next_states[name] = rho_post

        # How often did different initial-state trajectories choose different
        # projective outcomes at this time step?
        unique_outcomes = len(set(outcomes.values()))

        rows.append({
            "t": t,
            "max_pairwise_trace_distance": max_pairwise_trace_distance(next_states),
            "max_pairwise_feature_distance": max_pairwise_feature_distance(features),
            "number_of_distinct_outcomes_across_initial_states": unique_outcomes,
            "all_initial_states_same_outcome": int(unique_outcomes == 1),
        })

        rho = next_states

    return pd.DataFrame(rows)


def run_selective_mc(A_list, n_steps, mode_name, projectors):
    raw_parts = []
    washout_rows = []

    for rep in range(N_MC):
        rng = np.random.default_rng(MC_SEED_BASE + 1009 * rep)
        df = run_selective_coupled_trajectory(
            A_list,
            n_steps,
            projectors,
            rng,
        )
        df["mode"] = mode_name
        df["mc_rep"] = rep
        raw_parts.append(df)

        for eps in THRESHOLDS:
            t_trace = stable_threshold_crossing(
                df["max_pairwise_trace_distance"],
                eps,
            )
            t_feat = stable_threshold_crossing(
                df["max_pairwise_feature_distance"],
                eps,
            )
            washout_rows.append({
                "mode": mode_name,
                "mc_rep": rep,
                "epsilon": eps,
                "trace_distance_washout_t": np.nan if t_trace is None else t_trace,
                "feature_distance_washout_t": np.nan if t_feat is None else t_feat,
            })

    raw = pd.concat(raw_parts, ignore_index=True)
    washout = pd.DataFrame(washout_rows)

    time_summary = (
        raw.groupby("t", as_index=False)
        .agg(
            trace_distance_median=("max_pairwise_trace_distance", "median"),
            trace_distance_p90=("max_pairwise_trace_distance", lambda s: np.quantile(s, 0.90)),
            feature_distance_median=("max_pairwise_feature_distance", "median"),
            feature_distance_p90=("max_pairwise_feature_distance", lambda s: np.quantile(s, 0.90)),
            same_outcome_fraction=("all_initial_states_same_outcome", "mean"),
        )
    )

    threshold_summary_rows = []
    for eps in THRESHOLDS:
        sub = washout[washout["epsilon"] == eps]

        trace_valid = sub["trace_distance_washout_t"].dropna()
        feat_valid = sub["feature_distance_washout_t"].dropna()

        threshold_summary_rows.append({
            "mode": mode_name,
            "epsilon": eps,
            "trace_achieved_fraction": len(trace_valid) / N_MC,
            "trace_washout_median_t": (
                np.nan if len(trace_valid) == 0 else float(trace_valid.median())
            ),
            "trace_washout_p90_t": (
                np.nan if len(trace_valid) == 0 else float(np.quantile(trace_valid, 0.90))
            ),
            "feature_achieved_fraction": len(feat_valid) / N_MC,
            "feature_washout_median_t": (
                np.nan if len(feat_valid) == 0 else float(feat_valid.median())
            ),
            "feature_washout_p90_t": (
                np.nan if len(feat_valid) == 0 else float(np.quantile(feat_valid, 0.90))
            ),
        })

    threshold_summary = pd.DataFrame(threshold_summary_rows)

    return raw, washout, time_summary, threshold_summary


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 118)
    print("WEEK 7 - STEP 7.4B.1 (REVISED)")
    print("CONT WASHOUT UNDER DIRECT-LOCAL AND ANCILLA JOINT MEASUREMENT MODELS")
    print("=" * 118)
    print()
    print("Frozen:")
    print(f"  F4 | alpha={ALPHA} | seed={SEED}")
    print(f"  feature family={FAMILY}")
    print("  protocol=CONT")
    print(f"  selective MC trajectories={N_MC}")
    print()
    print("Modes:")
    print("  0) ideal_unmeasured")
    print("  1) direct_local_Y4_X5_selective")
    print("  2) ancilla_joint_YX_nonselective")
    print("  3) ancilla_joint_YX_selective")
    print()
    print("Direct local mode:")
    print("  measure Y4 and X5 separately -> four possible outcomes -> carry that conditional state")
    print("Ancilla joint selective mode:")
    print("  measure only joint parity +/- -> two possible outcomes -> carry that conditional parity state")
    print()

    audits = operator_audits()
    print("Operator audits:")
    for k, v in audits.items():
        print(f"  {k:42s} = {v:.3e}")
    print()

    work, train, val, cols = qrc.load_data()
    n_steps = len(train) + len(val)

    angles_all = qrc.make_input_angles(work, cols)
    U, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, S_list = qrc.build_input_channels(U, angles_all)

    print(f"QRC unitarity error = {unitary_err:.3e}")
    print(f"Chronological steps = {n_steps}")
    print()

    # -------------------------------------------------------------------------
    # Deterministic reference and non-selective ancilla
    # -------------------------------------------------------------------------
    print("Running ideal/unmeasured CONT ...")
    ideal = run_ideal_unmeasured(A_list, n_steps)
    ideal["mode"] = "ideal_unmeasured"

    print("Running non-selective ancilla joint-YX CONT ...")
    joint_ns = run_joint_nonselective(A_list, n_steps)
    joint_ns["mode"] = "ancilla_joint_YX_nonselective"

    deterministic = pd.concat([ideal, joint_ns], ignore_index=True)
    deterministic.to_csv(
        RESULTS / "07_04b1_washout_deterministic_timewise.csv",
        index=False,
    )

    deterministic_summary_rows = []
    for mode_name, df in [
        ("ideal_unmeasured", ideal),
        ("ancilla_joint_YX_nonselective", joint_ns),
    ]:
        print()
        print("-" * 118)
        print(f"DETERMINISTIC MODE = {mode_name}")
        print("-" * 118)
        print(
            f"initial trace distance = "
            f"{df.iloc[0]['max_pairwise_trace_distance']:.6f}"
        )
        print(
            f"final trace distance   = "
            f"{df.iloc[-1]['max_pairwise_trace_distance']:.6e}"
        )
        print(
            f"initial feature dist   = "
            f"{df.iloc[0]['max_pairwise_feature_distance']:.6f}"
        )
        print(
            f"final feature dist     = "
            f"{df.iloc[-1]['max_pairwise_feature_distance']:.6e}"
        )
        print()

        for eps in THRESHOLDS:
            t_trace = stable_threshold_crossing(
                df["max_pairwise_trace_distance"],
                eps,
            )
            t_feat = stable_threshold_crossing(
                df["max_pairwise_feature_distance"],
                eps,
            )

            print(
                f"epsilon={eps:<6g} | "
                f"trace washout={t_trace} | "
                f"feature washout={t_feat}"
            )

            deterministic_summary_rows.append({
                "mode": mode_name,
                "epsilon": eps,
                "trace_distance_washout_t": np.nan if t_trace is None else t_trace,
                "feature_distance_washout_t": np.nan if t_feat is None else t_feat,
            })

    deterministic_summary = pd.DataFrame(deterministic_summary_rows)
    deterministic_summary.to_csv(
        RESULTS / "07_04b1_washout_deterministic_thresholds.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Selective direct-local measurement
    # -------------------------------------------------------------------------
    print()
    print("=" * 118)
    print("Running SELECTIVE direct local Y4/X5 trajectories ...")
    print("=" * 118)

    local_raw, local_washout, local_time, local_threshold = run_selective_mc(
        A_list,
        n_steps,
        "direct_local_Y4_X5_selective",
        LOCAL_PROJECTORS,
    )

    # -------------------------------------------------------------------------
    # Selective joint ancilla measurement
    # -------------------------------------------------------------------------
    print()
    print("=" * 118)
    print("Running SELECTIVE ancilla joint YX trajectories ...")
    print("=" * 118)

    joint_raw, joint_washout, joint_time, joint_threshold = run_selective_mc(
        A_list,
        n_steps,
        "ancilla_joint_YX_selective",
        JOINT_PROJECTORS,
    )

    selective_raw = pd.concat([local_raw, joint_raw], ignore_index=True)
    selective_raw.to_csv(
        RESULTS / "07_04b1_washout_selective_mc_raw.csv",
        index=False,
    )

    selective_time = pd.concat([local_time.assign(
        mode="direct_local_Y4_X5_selective"
    ), joint_time.assign(
        mode="ancilla_joint_YX_selective"
    )], ignore_index=True)

    selective_time.to_csv(
        RESULTS / "07_04b1_washout_selective_timewise_summary.csv",
        index=False,
    )

    selective_threshold = pd.concat(
        [local_threshold, joint_threshold],
        ignore_index=True,
    )
    selective_threshold.to_csv(
        RESULTS / "07_04b1_washout_selective_threshold_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Print selective summary
    # -------------------------------------------------------------------------
    for mode_name, threshold_df, time_df in [
        (
            "direct_local_Y4_X5_selective",
            local_threshold,
            local_time,
        ),
        (
            "ancilla_joint_YX_selective",
            joint_threshold,
            joint_time,
        ),
    ]:
        print()
        print("-" * 118)
        print(f"SELECTIVE MODE = {mode_name}")
        print("-" * 118)
        print(
            f"final median trace distance  = "
            f"{time_df.iloc[-1]['trace_distance_median']:.6e}"
        )
        print(
            f"final p90 trace distance     = "
            f"{time_df.iloc[-1]['trace_distance_p90']:.6e}"
        )
        print(
            f"final median feature dist    = "
            f"{time_df.iloc[-1]['feature_distance_median']:.6e}"
        )
        print(
            f"final p90 feature dist       = "
            f"{time_df.iloc[-1]['feature_distance_p90']:.6e}"
        )
        print(
            f"final same-outcome fraction  = "
            f"{time_df.iloc[-1]['same_outcome_fraction']:.3f}"
        )
        print()
        print(threshold_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    # State convergence
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        ideal["t"],
        ideal["max_pairwise_trace_distance"],
        label="ideal/unmeasured",
    )
    ax.plot(
        joint_ns["t"],
        joint_ns["max_pairwise_trace_distance"],
        label="ancilla joint YX non-selective",
    )
    ax.plot(
        local_time["t"],
        local_time["trace_distance_median"],
        label="direct local Y/X selective — median",
    )
    ax.plot(
        joint_time["t"],
        joint_time["trace_distance_median"],
        label="ancilla joint YX selective — median",
    )
    ax.set_xlabel("CONT time step t")
    ax.set_ylabel("Maximum pairwise trace distance")
    ax.set_title("CONT initialization forgetting under measurement models")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        RESULTS / "07_04b1_washout_trace_distance_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    # Feature convergence
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        ideal["t"],
        ideal["max_pairwise_feature_distance"],
        label="ideal/unmeasured",
    )
    ax.plot(
        joint_ns["t"],
        joint_ns["max_pairwise_feature_distance"],
        label="ancilla joint YX non-selective",
    )
    ax.plot(
        local_time["t"],
        local_time["feature_distance_median"],
        label="direct local Y/X selective — median",
    )
    ax.plot(
        joint_time["t"],
        joint_time["feature_distance_median"],
        label="ancilla joint YX selective — median",
    )
    ax.set_xlabel("CONT time step t")
    ax.set_ylabel("Maximum pairwise feature-vector distance")
    ax.set_title("Observable initialization dependence under measurement models")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        RESULTS / "07_04b1_washout_feature_distance_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    # Selective trajectory p90 state-distance plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        local_time["t"],
        local_time["trace_distance_p90"],
        label="direct local Y/X selective — p90",
    )
    ax.plot(
        joint_time["t"],
        joint_time["trace_distance_p90"],
        label="ancilla joint YX selective — p90",
    )
    ax.set_xlabel("CONT time step t")
    ax.set_ylabel("90th percentile max pairwise trace distance")
    ax.set_title("Selective-measurement washout robustness across trajectories")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        RESULTS / "07_04b1_washout_selective_p90_trace.png",
        dpi=180,
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Text summary
    # -------------------------------------------------------------------------
    with open(
        RESULTS / "07_04b1_washout_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("WEEK 7 STEP 7.4B.1 REVISED - CONT WASHOUT\n")
        f.write("=" * 88 + "\n\n")
        f.write(f"alpha={ALPHA}, seed={SEED}, MC={N_MC}\n")
        f.write(f"feature family={FAMILY}\n\n")
        f.write("Modes:\n")
        f.write("0 ideal/unmeasured\n")
        f.write("1 direct local Y4/X5 selective\n")
        f.write("2 ancilla joint YX non-selective\n")
        f.write("3 ancilla joint YX selective\n\n")
        f.write("Operator audits:\n")
        for k, v in audits.items():
            f.write(f"{k}: {v:.6e}\n")
        f.write("\nDeterministic thresholds:\n")
        f.write(deterministic_summary.to_string(index=False))
        f.write("\n\nSelective thresholds:\n")
        f.write(selective_threshold.to_string(index=False))
        f.write("\n")

    print()
    print("=" * 118)
    print("Saved:")
    print("  results/07_04b1_washout_deterministic_timewise.csv")
    print("  results/07_04b1_washout_deterministic_thresholds.csv")
    print("  results/07_04b1_washout_selective_mc_raw.csv")
    print("  results/07_04b1_washout_selective_timewise_summary.csv")
    print("  results/07_04b1_washout_selective_threshold_summary.csv")
    print("  results/07_04b1_washout_trace_distance_comparison.png")
    print("  results/07_04b1_washout_feature_distance_comparison.png")
    print("  results/07_04b1_washout_selective_p90_trace.png")
    print("  results/07_04b1_washout_summary.txt")
    print()
    print("Do NOT freeze a washout length until all three measurement models are interpreted.")


if __name__ == "__main__":
    main()
