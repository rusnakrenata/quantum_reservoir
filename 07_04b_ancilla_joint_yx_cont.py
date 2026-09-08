"""
Week 7 - Step 7.4B
Ancilla-assisted JOINT YX_45 parity measurement for CONT.

Purpose
-------
Implement the genuine two-outcome projective measurement of

    P = Y_4 \otimes X_5

on the two memory qubits using an ancilla.

This is different from measuring Y_4 and X_5 separately and multiplying
their outcomes. The expectation statistics are the same, but the
post-measurement state is different.

Frozen diagnostic point
-----------------------
F4
alpha = 0.75
hx = 0.5
dt = 0.8
r = 2
reservoir seed = 42
CONT protocol
shots = 1024
MC repeats = 20

The script verifies:
1) ancilla probabilities satisfy
       p(0) = (1 + <YX>)/2
       p(1) = (1 - <YX>)/2
2) ancilla conditional memory states equal
       Pi_+ rho Pi_+ / p_+
       Pi_- rho Pi_- / p_-
   with Pi_± = (I ± YX)/2
3) the non-selective memory channel is
       rho -> Pi_+ rho Pi_+ + Pi_- rho Pi_-
            = (rho + P rho P)/2
4) trace-distance disturbance at every CONT step
5) cumulative forecasting impact when the disturbed memory is carried
   into the next time step
6) 1024-shot ancilla estimation error for <YX_45>

Dependency
----------
Keep 07_02b_memory_pair_isolation.py beside this script.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


# =============================================================================
# Load frozen QRC implementation
# =============================================================================

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "07_02b_memory_pair_isolation.py"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing {BASE_SCRIPT.name}. Keep it beside this Step-7.4B script."
    )

spec = importlib.util.spec_from_file_location("qrc_step_72b", BASE_SCRIPT)
qrc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qrc)

RESULTS = Path("results")

ALPHA = 0.75
SEED = 42
SHOTS = 1024
N_MC = 20
SHOT_SEED_BASE = 74020

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]


# =============================================================================
# Pauli operators and spectral projectors on the 2-qubit memory subsystem
# =============================================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)

I4 = np.eye(4, dtype=complex)
P_YX = np.kron(Y, X)

PI_PLUS = 0.5 * (I4 + P_YX)
PI_MINUS = 0.5 * (I4 - P_YX)

P0_A = np.array([[1, 0], [0, 0]], dtype=complex)
P1_A = np.array([[0, 0], [0, 1]], dtype=complex)
RHO_A0 = P0_A.copy()


# =============================================================================
# Explicit ancilla circuit
#
# |0>_a -- H --●-- H -- measure Z
#              |
#              P = Y_4 X_5
#
# Controlled-P = |0><0|⊗I + |1><1|⊗P
# =============================================================================

CONTROLLED_P = np.kron(P0_A, I4) + np.kron(P1_A, P_YX)
H_A = np.kron(H, I4)

# Right-most operation acts first.
U_ANCILLA = H_A @ CONTROLLED_P @ H_A


def ancilla_joint_yx_instrument(rho_m):
    """
    Run the explicit 1-ancilla + 2-memory-qubit ideal circuit.

    Returns
    -------
    p_plus, p_minus:
        ancilla Z probabilities (0 -> + parity, 1 -> - parity)
    rho_plus, rho_minus:
        normalized conditional memory states
    rho_nonselective:
        post-measurement memory state when the ancilla outcome is ignored
    """
    rho_joint_in = np.kron(RHO_A0, rho_m)
    rho_joint_out = U_ANCILLA @ rho_joint_in @ U_ANCILLA.conj().T

    # In ancilla-first tensor order, each diagonal 4x4 block is the
    # unnormalized conditional memory state.
    block_plus = rho_joint_out[0:4, 0:4]
    block_minus = rho_joint_out[4:8, 4:8]

    p_plus = float(np.real(np.trace(block_plus)))
    p_minus = float(np.real(np.trace(block_minus)))

    rho_plus = block_plus / p_plus if p_plus > 1e-15 else np.zeros((4, 4), complex)
    rho_minus = block_minus / p_minus if p_minus > 1e-15 else np.zeros((4, 4), complex)

    rho_nonselective = block_plus + block_minus
    rho_nonselective = 0.5 * (
        rho_nonselective + rho_nonselective.conj().T
    )
    rho_nonselective /= np.trace(rho_nonselective)

    return p_plus, p_minus, rho_plus, rho_minus, rho_nonselective


# =============================================================================
# Diagnostics
# =============================================================================

def trace_distance(rho, sigma):
    """
    D(rho,sigma) = 1/2 ||rho-sigma||_1.
    For Hermitian delta, trace norm = sum absolute eigenvalues.
    """
    delta = 0.5 * ((rho - sigma) + (rho - sigma).conj().T)
    eigvals = np.linalg.eigvalsh(delta)
    return float(0.5 * np.sum(np.abs(eigvals)))


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def yx_expectation(rho):
    return float(np.real(np.trace(rho @ P_YX)))


def spectral_instrument(rho):
    """
    Reference joint projective instrument from Pi_±.
    """
    plus_u = PI_PLUS @ rho @ PI_PLUS
    minus_u = PI_MINUS @ rho @ PI_MINUS

    p_plus = float(np.real(np.trace(plus_u)))
    p_minus = float(np.real(np.trace(minus_u)))

    plus_n = plus_u / p_plus if p_plus > 1e-15 else np.zeros((4, 4), complex)
    minus_n = minus_u / p_minus if p_minus > 1e-15 else np.zeros((4, 4), complex)

    nonselective = plus_u + minus_u
    nonselective = 0.5 * (nonselective + nonselective.conj().T)
    nonselective /= np.trace(nonselective)

    return p_plus, p_minus, plus_n, minus_n, nonselective


def dephasing_formula(rho):
    return 0.5 * (rho + P_YX @ rho @ P_YX)


# =============================================================================
# Ridge helpers
# =============================================================================

def fixed_alpha_cv(X, y, ridge_alpha):
    splitter = TimeSeriesSplit(n_splits=qrc.N_CV_SPLITS)
    scores = []

    for tr_idx, va_idx in splitter.split(X):
        pred = qrc.fit_predict_ridge(
            X[tr_idx], y[tr_idx], X[va_idx], ridge_alpha
        )
        scores.append(qrc.rmse(y[va_idx], pred))

    return float(np.mean(scores)), float(np.std(scores, ddof=1))


# =============================================================================
# CONT trajectory
# =============================================================================

FAMILY = "XZinj_plus_YX45"
FEATURES = qrc.FAMILIES[FAMILY]


def build_cont_trajectories(A_list, n_train, n_val):
    """
    Build two CONT trajectories:

    baseline:
        memory is carried with NO measurement back-action

    ancilla:
        after each time-step readout, a genuine joint YX projective
        measurement is applied and the non-selective post-measurement
        memory state is carried to the next time step.

    Readout features are always taken immediately BEFORE the measurement
    back-action at that time step.
    """
    total = n_train + n_val

    rho_base = qrc.memory_zero_density()
    rho_anc = qrc.memory_zero_density()

    X_base = []
    X_anc_exact = []
    ancilla_rows = []

    for t in range(total):
        A = A_list[t]

        # ---------------------------------------------------------------------
        # Baseline CONT
        # ---------------------------------------------------------------------
        rho_i_base, rho_m_base_out = qrc.final_reduced_states(A, rho_base)
        feat_base = qrc.extract_feature_row(
            A, rho_base, rho_i_base, rho_m_base_out
        )
        X_base.append([feat_base[f] for f in FEATURES])
        rho_base = rho_m_base_out

        # ---------------------------------------------------------------------
        # CONT with ancilla YX joint measurement
        # ---------------------------------------------------------------------
        rho_i_anc, rho_m_pre = qrc.final_reduced_states(A, rho_anc)
        feat_anc = qrc.extract_feature_row(
            A, rho_anc, rho_i_anc, rho_m_pre
        )
        X_anc_exact.append([feat_anc[f] for f in FEATURES])

        ideal_yx = yx_expectation(rho_m_pre)

        p_plus, p_minus, rho_plus, rho_minus, rho_post = (
            ancilla_joint_yx_instrument(rho_m_pre)
        )

        # Spectral-projector reference
        sp, sm, sp_state, sm_state, sp_nonselective = spectral_instrument(rho_m_pre)

        # Algebra audits
        prob_formula_err = max(
            abs(p_plus - (1.0 + ideal_yx) / 2.0),
            abs(p_minus - (1.0 - ideal_yx) / 2.0),
        )
        expectation_err = abs((p_plus - p_minus) - ideal_yx)
        projector_prob_err = max(abs(p_plus - sp), abs(p_minus - sm))
        nonselective_projector_err = float(
            np.max(np.abs(rho_post - sp_nonselective))
        )
        dephasing_err = float(
            np.max(np.abs(rho_post - dephasing_formula(rho_m_pre)))
        )

        plus_state_err = 0.0
        minus_state_err = 0.0
        if p_plus > 1e-12:
            plus_state_err = float(np.max(np.abs(rho_plus - sp_state)))
        if p_minus > 1e-12:
            minus_state_err = float(np.max(np.abs(rho_minus - sm_state)))

        disturbance = trace_distance(rho_m_pre, rho_post)

        ancilla_rows.append({
            "t": t,
            "ideal_yx": ideal_yx,
            "p_plus": p_plus,
            "p_minus": p_minus,
            "p_plus_minus_p_minus": p_plus - p_minus,
            "probability_formula_error": prob_formula_err,
            "expectation_error": expectation_err,
            "projector_probability_error": projector_prob_err,
            "plus_conditional_state_error": plus_state_err,
            "minus_conditional_state_error": minus_state_err,
            "nonselective_projector_error": nonselective_projector_err,
            "dephasing_formula_error": dephasing_err,
            "trace_distance_pre_vs_post": disturbance,
            "purity_pre": purity(rho_m_pre),
            "purity_post_nonselective": purity(rho_post),
            "purity_change": purity(rho_post) - purity(rho_m_pre),
        })

        # Carry the disturbed memory into the next CONT time step.
        rho_anc = rho_post

    return (
        np.asarray(X_base, dtype=float),
        np.asarray(X_anc_exact, dtype=float),
        pd.DataFrame(ancilla_rows),
    )


# =============================================================================
# Finite-shot ancilla readout
# =============================================================================

def shot_noisy_yx_feature(X_exact, ancilla_df, rng):
    """
    Replace only the YX_45 column by the finite-shot ancilla estimator.

    For each time step:
      n_plus ~ Binomial(N, p_plus)
      <YX>_hat = (n_plus - n_minus)/N = 2*n_plus/N - 1

    The carried memory trajectory is the non-selective projective trajectory
    built above; this function isolates finite-shot readout uncertainty from
    the already modeled measurement back-action.
    """
    X = X_exact.copy()
    yx_col = FEATURES.index("YX_45")

    n_plus = rng.binomial(
        SHOTS,
        ancilla_df["p_plus"].to_numpy(dtype=float),
    )
    X[:, yx_col] = 2.0 * n_plus / SHOTS - 1.0

    return X


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 124)
    print("WEEK 7 - STEP 7.4B")
    print("ANCILLA-ASSISTED JOINT YX_45 PARITY MEASUREMENT FOR CONT")
    print("=" * 124)
    print()
    print("Frozen diagnostic point:")
    print("  F4 | 4 injection + 2 memory")
    print(f"  alpha={ALPHA} | hx={qrc.HX} | dt={qrc.DT} | r={qrc.TROTTER_R}")
    print(f"  reservoir seed={SEED}")
    print(f"  protocol=CONT")
    print(f"  shots={SHOTS}")
    print(f"  shot Monte-Carlo repeats={N_MC}")
    print()
    print("Joint observable:")
    print("  P = Y_4 tensor X_5")
    print("  Pi_+ = (I + P)/2")
    print("  Pi_- = (I - P)/2")
    print()
    print("Ancilla circuit:")
    print("  |0> -- H -- controlled-(Y4 X5) -- H -- measure Z")
    print("  ancilla 0 -> +1 parity")
    print("  ancilla 1 -> -1 parity")
    print()

    # -------------------------------------------------------------------------
    # Basic operator audits
    # -------------------------------------------------------------------------
    projector_idempotence = max(
        np.max(np.abs(PI_PLUS @ PI_PLUS - PI_PLUS)),
        np.max(np.abs(PI_MINUS @ PI_MINUS - PI_MINUS)),
    )
    projector_orthogonality = np.max(np.abs(PI_PLUS @ PI_MINUS))
    projector_completeness = np.max(np.abs(PI_PLUS + PI_MINUS - I4))
    p_square_error = np.max(np.abs(P_YX @ P_YX - I4))
    ancilla_unitarity = np.max(
        np.abs(U_ANCILLA.conj().T @ U_ANCILLA - np.eye(8))
    )

    print("Operator/circuit audits:")
    print(f"  ||P^2-I||_max                    = {p_square_error:.3e}")
    print(f"  projector idempotence error      = {projector_idempotence:.3e}")
    print(f"  projector orthogonality error    = {projector_orthogonality:.3e}")
    print(f"  projector completeness error     = {projector_completeness:.3e}")
    print(f"  ancilla-circuit unitarity error  = {ancilla_unitarity:.3e}")
    print()

    work, train, val, cols = qrc.load_data()
    n_train = len(train)
    n_val = len(val)

    y_train = train[cols["target"]].to_numpy(dtype=float)
    y_val = val[cols["target"]].to_numpy(dtype=float)

    angles_all = qrc.make_input_angles(work, cols)
    U, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, S_list = qrc.build_input_channels(U, angles_all)

    print(f"QRC unitarity error = {unitary_err:.3e}")
    print("Building baseline and ancilla-disturbed CONT trajectories ...")

    X_base_all, X_anc_all, ancilla_df = build_cont_trajectories(
        A_list, n_train, n_val
    )

    Xtr_base = X_base_all[:n_train]
    Xva_base = X_base_all[n_train:n_train + n_val]

    Xtr_anc = X_anc_all[:n_train]
    Xva_anc = X_anc_all[n_train:n_train + n_val]

    # -------------------------------------------------------------------------
    # Verify the actual ancilla circuit = projector measurement
    # -------------------------------------------------------------------------
    max_prob_err = float(ancilla_df["probability_formula_error"].max())
    max_expectation_err = float(ancilla_df["expectation_error"].max())
    max_projector_prob_err = float(ancilla_df["projector_probability_error"].max())
    max_plus_state_err = float(ancilla_df["plus_conditional_state_error"].max())
    max_minus_state_err = float(ancilla_df["minus_conditional_state_error"].max())
    max_nonselective_err = float(
        ancilla_df["nonselective_projector_error"].max()
    )
    max_dephasing_err = float(ancilla_df["dephasing_formula_error"].max())

    print()
    print("Ancilla measurement algebra:")
    print(f"  max p_± formula error                = {max_prob_err:.3e}")
    print(f"  max (p_+ - p_-) vs <YX> error        = {max_expectation_err:.3e}")
    print(f"  max ancilla vs projector prob error  = {max_projector_prob_err:.3e}")
    print(f"  max + conditional-state error        = {max_plus_state_err:.3e}")
    print(f"  max - conditional-state error        = {max_minus_state_err:.3e}")
    print(f"  max nonselective projector error     = {max_nonselective_err:.3e}")
    print(f"  max rho' vs (rho+P rho P)/2 error    = {max_dephasing_err:.3e}")

    # -------------------------------------------------------------------------
    # State disturbance
    # -------------------------------------------------------------------------
    disturbance_summary = pd.DataFrame([{
        "mean_trace_distance": float(
            ancilla_df["trace_distance_pre_vs_post"].mean()
        ),
        "median_trace_distance": float(
            ancilla_df["trace_distance_pre_vs_post"].median()
        ),
        "max_trace_distance": float(
            ancilla_df["trace_distance_pre_vs_post"].max()
        ),
        "mean_purity_pre": float(ancilla_df["purity_pre"].mean()),
        "mean_purity_post_nonselective": float(
            ancilla_df["purity_post_nonselective"].mean()
        ),
        "mean_purity_change": float(ancilla_df["purity_change"].mean()),
    }])

    print()
    print("Memory-state disturbance from projective joint YX measurement:")
    print(disturbance_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # Forecasting: unmeasured CONT vs CONT carrying disturbed memory
    # -------------------------------------------------------------------------
    best_base, _ = qrc.chronological_cv(Xtr_base, y_train)
    alpha_base = float(best_base["ridge_alpha"])
    base_cv = float(best_base["cv_rmse_mean"])
    base_pred = qrc.fit_predict_ridge(
        Xtr_base, y_train, Xva_base, alpha_base
    )
    base_val = qrc.rmse(y_val, base_pred)

    best_anc, _ = qrc.chronological_cv(Xtr_anc, y_train)
    alpha_anc = float(best_anc["ridge_alpha"])
    anc_cv = float(best_anc["cv_rmse_mean"])
    anc_pred = qrc.fit_predict_ridge(
        Xtr_anc, y_train, Xva_anc, alpha_anc
    )
    anc_val = qrc.rmse(y_val, anc_pred)

    # Also isolate how much the carried memory trajectory changes the features.
    feature_trajectory_rmse = float(
        np.sqrt(np.mean((X_anc_all - X_base_all) ** 2))
    )

    forecast_compare = pd.DataFrame([
        {
            "case": "CONT_unmeasured_reference",
            "ridge_alpha": alpha_base,
            "cv_rmse": base_cv,
            "validation_rmse": base_val,
            "delta_cv_vs_unmeasured": 0.0,
            "relative_cv_change_pct": 0.0,
        },
        {
            "case": "CONT_joint_YX_projective_backaction",
            "ridge_alpha": alpha_anc,
            "cv_rmse": anc_cv,
            "validation_rmse": anc_val,
            "delta_cv_vs_unmeasured": anc_cv - base_cv,
            "relative_cv_change_pct": 100.0 * (anc_cv - base_cv) / base_cv,
        },
    ])

    print()
    print("CONT forecasting impact of carrying the disturbed memory:")
    print(forecast_compare.to_string(index=False))
    print()
    print(
        f"RMS change in complete reservoir feature trajectory = "
        f"{feature_trajectory_rmse:.6e}"
    )

    # -------------------------------------------------------------------------
    # Finite-shot ancilla estimation, on top of projective back-action
    # -------------------------------------------------------------------------
    noisy_cv = []
    noisy_val = []
    yx_feature_rmse = []

    yx_col = FEATURES.index("YX_45")

    for rep in range(N_MC):
        rng = np.random.default_rng(SHOT_SEED_BASE + rep * 1009)
        X_shot_all = shot_noisy_yx_feature(X_anc_all, ancilla_df, rng)

        Xtr_shot = X_shot_all[:n_train]
        Xva_shot = X_shot_all[n_train:n_train + n_val]

        cv, _ = fixed_alpha_cv(Xtr_shot, y_train, alpha_anc)
        pred = qrc.fit_predict_ridge(
            Xtr_shot, y_train, Xva_shot, alpha_anc
        )
        val_rmse = qrc.rmse(y_val, pred)

        noisy_cv.append(cv)
        noisy_val.append(val_rmse)
        yx_feature_rmse.append(
            float(np.sqrt(np.mean(
                (X_shot_all[:, yx_col] - X_anc_all[:, yx_col]) ** 2
            )))
        )

    shot_summary = pd.DataFrame([{
        "shots": SHOTS,
        "mc_repeats": N_MC,
        "exact_backaction_cv_rmse": anc_cv,
        "shot_cv_rmse_mean": float(np.mean(noisy_cv)),
        "shot_cv_rmse_std": float(np.std(noisy_cv, ddof=1)),
        "relative_shot_degradation_pct": float(
            100.0 * (np.mean(noisy_cv) - anc_cv) / anc_cv
        ),
        "exact_backaction_validation_rmse": anc_val,
        "shot_validation_rmse_mean": float(np.mean(noisy_val)),
        "shot_validation_rmse_std": float(np.std(noisy_val, ddof=1)),
        "yx_estimation_rmse_mean": float(np.mean(yx_feature_rmse)),
    }])

    print()
    print("1024-shot ancilla readout on top of projective back-action:")
    print(shot_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    ancilla_df.to_csv(
        RESULTS / "07_04b_ancilla_joint_yx_timewise.csv",
        index=False,
    )
    disturbance_summary.to_csv(
        RESULTS / "07_04b_ancilla_disturbance_summary.csv",
        index=False,
    )
    forecast_compare.to_csv(
        RESULTS / "07_04b_cont_forecast_backaction.csv",
        index=False,
    )
    shot_summary.to_csv(
        RESULTS / "07_04b_ancilla_shot_summary.csv",
        index=False,
    )

    with open(
        RESULTS / "07_04b_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("WEEK 7 STEP 7.4B - ANCILLA JOINT YX_45 MEASUREMENT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"alpha={ALPHA}, seed={SEED}, shots={SHOTS}, protocol=CONT\n\n")
        f.write("Ancilla measurement algebra:\n")
        f.write(f"max p formula error = {max_prob_err:.6e}\n")
        f.write(f"max expectation error = {max_expectation_err:.6e}\n")
        f.write(f"max projector probability error = {max_projector_prob_err:.6e}\n")
        f.write(f"max conditional + state error = {max_plus_state_err:.6e}\n")
        f.write(f"max conditional - state error = {max_minus_state_err:.6e}\n")
        f.write(f"max nonselective error = {max_nonselective_err:.6e}\n")
        f.write(f"max dephasing-formula error = {max_dephasing_err:.6e}\n\n")
        f.write("Disturbance:\n")
        f.write(disturbance_summary.to_string(index=False))
        f.write("\n\nForecast comparison:\n")
        f.write(forecast_compare.to_string(index=False))
        f.write("\n\nShot summary:\n")
        f.write(shot_summary.to_string(index=False))
        f.write(
            f"\n\nfeature trajectory RMSE vs unmeasured CONT = "
            f"{feature_trajectory_rmse:.6e}\n"
        )

    print()
    print("Saved:")
    print("  results/07_04b_ancilla_joint_yx_timewise.csv")
    print("  results/07_04b_ancilla_disturbance_summary.csv")
    print("  results/07_04b_cont_forecast_backaction.csv")
    print("  results/07_04b_ancilla_shot_summary.csv")
    print("  results/07_04b_summary.txt")
    print()
    print("Step 7.4B finished.")
    print("Interpret ancilla correctness + CONT back-action before moving on.")


if __name__ == "__main__":
    main()
