"""
Week 7 - Step 7.4A
Direct basis-rotation + parity measurement verification.

Purpose
-------
Verify the actual projective-measurement algebra used later on hardware.

Frozen verification point:
    F4
    alpha = 0.75
    hx = 0.5
    dt = 0.8
    r = 2
    reservoir seed = 42
    shots = 1024 PER MEASUREMENT SETTING
    CONT + RWP W in {1,2,5,7,14,21,28}

This step explicitly:
1) constructs the final six-qubit density matrix;
2) applies the basis rotations corresponding to X/Y/Z measurement;
3) extracts computational-basis probabilities;
4) converts measured bitstrings to Pauli eigenvalues +/-1;
5) forms single-qubit and two-qubit parity estimators;
6) checks that infinite-shot probabilities reproduce the exact Pauli
   expectations from Step 7.2B;
7) samples finite shots from grouped measurement settings and checks the
   forecasting effect.

No ancilla, gate noise, readout noise, or hardware transpilation yet.

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


# =============================================================================
# Load the frozen QRC implementation
# =============================================================================

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "07_02b_memory_pair_isolation.py"
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing {BASE_SCRIPT.name}. Keep it beside this Step-7.4A script."
    )

spec = importlib.util.spec_from_file_location("qrc_step_72b", BASE_SCRIPT)
qrc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qrc)

RESULTS = Path("results")

ALPHA = 0.75
SEED = 42
SHOTS = 1024
N_SHOT_REPEATS = 20
SHOT_SEED_BASE = 74001

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]

PROTOCOL_SPECS = [("CONT", None)] + [
    (f"RWP_W{W}", W) for W in qrc.WINDOWS
]

FAMILIES = {
    "XZ_injection": qrc.FAMILIES["XZ_injection"],
    "XZ_all": qrc.FAMILIES["XZ_all"],
    "XZinj_plus_YX45": qrc.FAMILIES["XZinj_plus_YX45"],
    "XZinj_plus_XZ45": qrc.FAMILIES["XZinj_plus_XZ45"],
    "XZinj_plus_YX45_XZ45": qrc.FAMILIES["XZinj_plus_YX45_XZ45"],
}


# =============================================================================
# Measurement basis rotations
# =============================================================================

I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)

# Circuit order for Y measurement is Sdg then H, therefore total matrix H @ Sdg.
BASIS_ROTATION = {
    "Z": I2,
    "X": H,
    "Y": H @ SDG,
}


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def rotation_for_setting(setting):
    return kron_all([BASIS_ROTATION[b] for b in setting])


# Each family is estimated from exactly two commuting measurement settings.
# q0 is the most-significant computational-basis bit in the QRC tensor order.
MEASUREMENT_PLANS = {
    "XZ_injection": [
        ("A", ("X", "X", "X", "X", "Z", "Z")),
        ("B", ("Z", "Z", "Z", "Z", "Z", "Z")),
    ],
    "XZ_all": [
        ("A", ("X", "X", "X", "X", "X", "X")),
        ("B", ("Z", "Z", "Z", "Z", "Z", "Z")),
    ],
    "XZinj_plus_YX45": [
        ("A", ("X", "X", "X", "X", "Y", "X")),
        ("B", ("Z", "Z", "Z", "Z", "Z", "Z")),
    ],
    "XZinj_plus_XZ45": [
        ("A", ("X", "X", "X", "X", "Z", "Z")),
        ("B", ("Z", "Z", "Z", "Z", "X", "Z")),
    ],
    "XZinj_plus_YX45_XZ45": [
        ("A", ("X", "X", "X", "X", "Y", "X")),
        ("B", ("Z", "Z", "Z", "Z", "X", "Z")),
    ],
}


def basis_gate_count(setting):
    # X -> H = 1 gate; Y -> Sdg + H = 2 gates; Z -> 0.
    return sum(1 if b == "X" else 2 if b == "Y" else 0 for b in setting)


# =============================================================================
# Bitstring/parity algebra
# =============================================================================

N_STATES = 2 ** qrc.N_QUBITS

# bits[index, q] is the computational-basis bit for global qubit q.
BITS = np.zeros((N_STATES, qrc.N_QUBITS), dtype=int)
for idx in range(N_STATES):
    for q in range(qrc.N_QUBITS):
        shift = qrc.N_QUBITS - 1 - q
        BITS[idx, q] = (idx >> shift) & 1


def eigenvalue_vector(qubits):
    """Return +/-1 parity eigenvalue for each computational basis outcome."""
    qlist = tuple(qubits)
    parity = np.sum(BITS[:, qlist], axis=1) % 2
    return 1.0 - 2.0 * parity


def feature_measurement_map(family):
    """
    Map each feature to one family measurement setting and parity qubits.
    """
    fmap = {}
    if family == "XZ_all":
        for q in range(6):
            fmap[f"X{q}"] = ("A", (q,))
            fmap[f"Z{q}"] = ("B", (q,))
        return fmap

    # All remaining families contain X/Z on injection q0..q3.
    for q in range(4):
        fmap[f"X{q}"] = ("A", (q,))
        fmap[f"Z{q}"] = ("B", (q,))

    if "YX_45" in FAMILIES[family]:
        fmap["YX_45"] = ("A", (4, 5))
    if "XZ_45" in FAMILIES[family]:
        fmap["XZ_45"] = ("B", (4, 5))
    return fmap


EIGENVECTORS = {}
for family in FAMILIES:
    EIGENVECTORS[family] = {
        feat: eigenvalue_vector(qubits)
        for feat, (_, qubits) in feature_measurement_map(family).items()
    }


# =============================================================================
# Final six-qubit state and measurement probabilities
# =============================================================================

def full_density_matrix(A, rho_before):
    """
    rho_out[a,m,b,p] = sum_nk A[a,m,n] rho_before[n,k] A*[b,p,k].
    Reshape (injection,memory) indices into the 64x64 full density matrix.
    """
    rho4 = np.einsum(
        "amn,nk,bpk->ambp",
        A, rho_before, A.conj(),
        optimize=True,
    )
    rho = rho4.reshape(N_STATES, N_STATES)
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho


# Cache the unique basis rotations.
UNIQUE_SETTINGS = {}
for plan in MEASUREMENT_PLANS.values():
    for _, setting in plan:
        UNIQUE_SETTINGS[setting] = rotation_for_setting(setting)


def probabilities_for_setting(rho, setting):
    U = UNIQUE_SETTINGS[setting]
    rho_rot = U @ rho @ U.conj().T
    probs = np.real(np.diag(rho_rot))
    probs = np.clip(probs, 0.0, None)
    probs /= np.sum(probs)
    return probs


def family_probabilities(rho, family):
    return {
        label: probabilities_for_setting(rho, setting)
        for label, setting in MEASUREMENT_PLANS[family]
    }


def expectations_from_probabilities(prob_by_label, family):
    fmap = feature_measurement_map(family)
    row = {}
    for feat in FAMILIES[family]:
        label, _ = fmap[feat]
        row[feat] = float(prob_by_label[label] @ EIGENVECTORS[family][feat])
    return row


def expectations_from_counts(count_by_label, family):
    fmap = feature_measurement_map(family)
    row = {}
    for feat in FAMILIES[family]:
        label, _ = fmap[feat]
        counts = count_by_label[label]
        row[feat] = float(counts @ EIGENVECTORS[family][feat] / np.sum(counts))
    return row


# =============================================================================
# Protocol-state construction
# =============================================================================

def build_cont_prob_bank(A_list, family):
    rows = []
    rho_m = qrc.memory_zero_density()

    for A in A_list:
        rho_before = rho_m
        rho_i, rho_m_out = qrc.final_reduced_states(A, rho_before)
        exact_row = qrc.extract_feature_row(A, rho_before, rho_i, rho_m_out)

        rho = full_density_matrix(A, rho_before)
        probs = family_probabilities(rho, family)
        meas_row = expectations_from_probabilities(probs, family)

        rows.append({
            "exact": {f: exact_row[f] for f in FAMILIES[family]},
            "prob_expectation": meas_row,
            "probs": probs,
        })
        rho_m = rho_m_out

    return rows


def build_window_prob_bank(A_list, S_list, end_indices, W, family):
    rows = []
    rho0 = qrc.memory_zero_density()

    for j in end_indices:
        start = j - W + 1
        rho_m = rho0.copy()
        for k in range(start, j):
            rho_m = qrc.apply_memory_channel(S_list[k], rho_m)

        rho_before = rho_m
        rho_i, rho_m_out = qrc.final_reduced_states(A_list[j], rho_before)
        exact_row = qrc.extract_feature_row(
            A_list[j], rho_before, rho_i, rho_m_out
        )

        rho = full_density_matrix(A_list[j], rho_before)
        probs = family_probabilities(rho, family)
        meas_row = expectations_from_probabilities(probs, family)

        rows.append({
            "exact": {f: exact_row[f] for f in FAMILIES[family]},
            "prob_expectation": meas_row,
            "probs": probs,
        })

    return rows


def rows_to_exact_matrix(rows, family):
    feats = FAMILIES[family]
    return np.asarray([[r["exact"][f] for f in feats] for r in rows], dtype=float)


def rows_to_prob_matrix(rows, family):
    feats = FAMILIES[family]
    return np.asarray(
        [[r["prob_expectation"][f] for f in feats] for r in rows], dtype=float
    )


def sample_rows(rows, family, shots, rng):
    feats = FAMILIES[family]
    X = np.zeros((len(rows), len(feats)), dtype=float)

    for i, r in enumerate(rows):
        count_by_label = {
            label: rng.multinomial(shots, probs)
            for label, probs in r["probs"].items()
        }
        vals = expectations_from_counts(count_by_label, family)
        X[i] = [vals[f] for f in feats]

    return X


# =============================================================================
# Ridge helper with frozen exact lambda
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
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 124)
    print("WEEK 7 - STEP 7.4A")
    print("DIRECT BASIS-ROTATION + PARITY MEASUREMENT VERIFICATION")
    print("=" * 124)
    print()
    print("Frozen verification point:")
    print("  F4 | 4 injection + 2 memory")
    print(f"  alpha={ALPHA} | hx={qrc.HX} | dt={qrc.DT} | r={qrc.TROTTER_R}")
    print(f"  reservoir seed={SEED}")
    print(f"  shots={SHOTS} PER MEASUREMENT SETTING")
    print(f"  Monte-Carlo repeats={N_SHOT_REPEATS}")
    print(f"  protocols=CONT + RWP {qrc.WINDOWS}")
    print()
    print("Basis rules:")
    print("  Z: measure directly")
    print("  X: H -> measure")
    print("  Y: Sdg -> H -> measure")
    print("  Pair observables: multiply the two +/-1 outcomes shot by shot")
    print()

    print("Measurement plans:")
    cost_rows = []
    for family, plan in MEASUREMENT_PLANS.items():
        print(f"  {family}:")
        total_basis_gates = 0
        for label, setting in plan:
            ng = basis_gate_count(setting)
            total_basis_gates += ng
            print(f"    setting {label}: {''.join(setting)} | basis-change gates={ng}")
        print(
            f"    -> 2 settings, {2*SHOTS} total circuit shots/state, "
            f"{total_basis_gates} basis-change gates across the two settings"
        )
        cost_rows.append({
            "family": family,
            "measurement_settings": 2,
            "shots_per_setting": SHOTS,
            "total_circuit_shots_per_state": 2 * SHOTS,
            "basis_change_gates_across_settings": total_basis_gates,
        })
    print()

    work, train, val, cols = qrc.load_data()
    n_train = len(train)
    n_val = len(val)
    y_train_full = train[cols["target"]].to_numpy(dtype=float)
    y_val = val[cols["target"]].to_numpy(dtype=float)

    angles_all = qrc.make_input_angles(work, cols)
    U, unitary_err = qrc.build_trotter_unitary(SEED)
    print(f"Unitarity error = {unitary_err:.3e}")
    A_list, S_list = qrc.build_input_channels(U, angles_all)

    audit_rows = []
    forecast_rows = []

    for protocol, W in PROTOCOL_SPECS:
        print()
        print("-" * 124)
        print(f"PROTOCOL = {protocol}")
        print("-" * 124)

        if protocol == "CONT":
            y_train = y_train_full
            train_selector = slice(0, n_train)
            val_selector = slice(n_train, n_train + n_val)
        else:
            y_train = y_train_full[W - 1:]
            train_end = np.arange(W - 1, n_train, dtype=int)
            val_end = np.arange(n_train, n_train + n_val, dtype=int)

        for family in FAMILIES:
            if protocol == "CONT":
                all_rows = build_cont_prob_bank(A_list, family)
                train_rows = all_rows[train_selector]
                val_rows = all_rows[val_selector]
            else:
                train_rows = build_window_prob_bank(
                    A_list, S_list, train_end, W, family
                )
                val_rows = build_window_prob_bank(
                    A_list, S_list, val_end, W, family
                )

            Xtr_exact = rows_to_exact_matrix(train_rows, family)
            Xva_exact = rows_to_exact_matrix(val_rows, family)
            Xtr_prob = rows_to_prob_matrix(train_rows, family)
            Xva_prob = rows_to_prob_matrix(val_rows, family)

            # Infinite-shot measurement-circuit algebra audit.
            max_err = float(max(
                np.max(np.abs(Xtr_exact - Xtr_prob)),
                np.max(np.abs(Xva_exact - Xva_prob)),
            ))

            if max_err > 1e-10:
                raise RuntimeError(
                    f"Measurement algebra audit FAILED: {protocol} {family} "
                    f"max error={max_err:.3e}"
                )

            audit_rows.append({
                "protocol": protocol,
                "W": np.nan if W is None else W,
                "family": family,
                "max_abs_exact_vs_rotated_probability": max_err,
                "pass": True,
            })

            # Exact reference readout.
            best_exact, _ = qrc.chronological_cv(Xtr_exact, y_train)
            ridge_alpha = float(best_exact["ridge_alpha"])
            exact_cv = float(best_exact["cv_rmse_mean"])
            exact_pred = qrc.fit_predict_ridge(
                Xtr_exact, y_train, Xva_exact, ridge_alpha
            )
            exact_val = qrc.rmse(y_val, exact_pred)

            noisy_cvs = []
            noisy_vals = []
            feature_rmses = []

            for rep in range(N_SHOT_REPEATS):
                family_code = sum((i + 1) * ord(c) for i, c in enumerate(family))
                protocol_code = sum((i + 1) * ord(c) for i, c in enumerate(protocol))
                rng_seed = (
                    SHOT_SEED_BASE
                    + rep * 10007
                    + family_code * 17
                    + protocol_code * 19
                ) % (2**63 - 1)
                rng = np.random.default_rng(rng_seed)

                Xtr_shot = sample_rows(train_rows, family, SHOTS, rng)
                Xva_shot = sample_rows(val_rows, family, SHOTS, rng)

                noisy_cv, _ = fixed_alpha_cv(Xtr_shot, y_train, ridge_alpha)
                noisy_pred = qrc.fit_predict_ridge(
                    Xtr_shot, y_train, Xva_shot, ridge_alpha
                )
                noisy_val = qrc.rmse(y_val, noisy_pred)

                feature_rmse = float(np.sqrt(np.mean(
                    np.concatenate([
                        (Xtr_shot - Xtr_exact).ravel() ** 2,
                        (Xva_shot - Xva_exact).ravel() ** 2,
                    ])
                )))

                noisy_cvs.append(noisy_cv)
                noisy_vals.append(noisy_val)
                feature_rmses.append(feature_rmse)

            row = {
                "protocol": protocol,
                "W": np.nan if W is None else W,
                "family": family,
                "n_features": len(FAMILIES[family]),
                "measurement_settings": 2,
                "shots_per_setting": SHOTS,
                "total_circuit_shots_per_state": 2 * SHOTS,
                "ridge_alpha_exact": ridge_alpha,
                "exact_cv_rmse": exact_cv,
                "grouped_shot_cv_rmse_mean": float(np.mean(noisy_cvs)),
                "grouped_shot_cv_rmse_std": float(np.std(noisy_cvs, ddof=1)),
                "delta_cv_rmse": float(np.mean(noisy_cvs) - exact_cv),
                "relative_cv_degradation_pct": float(
                    100.0 * (np.mean(noisy_cvs) - exact_cv) / exact_cv
                ),
                "exact_validation_rmse": exact_val,
                "grouped_shot_validation_rmse_mean": float(np.mean(noisy_vals)),
                "grouped_shot_validation_rmse_std": float(np.std(noisy_vals, ddof=1)),
                "feature_estimation_rmse_mean": float(np.mean(feature_rmses)),
                "algebra_max_abs_error": max_err,
            }
            forecast_rows.append(row)

            print(
                f"{family:28s} | algebra err={max_err:.2e} | "
                f"exact CV={exact_cv:.6f} | "
                f"1024-shot grouped CV={row['grouped_shot_cv_rmse_mean']:.6f} "
                f"(delta={row['relative_cv_degradation_pct']:+.3f}%)"
            )

    audit_df = pd.DataFrame(audit_rows)
    forecast_df = pd.DataFrame(forecast_rows)
    cost_df = pd.DataFrame(cost_rows)

    audit_df.to_csv(
        RESULTS / "07_04a_measurement_algebra_audit.csv", index=False
    )
    forecast_df.to_csv(
        RESULTS / "07_04a_grouped_shot_forecast_results.csv", index=False
    )
    cost_df.to_csv(
        RESULTS / "07_04a_measurement_setting_costs.csv", index=False
    )

    summary = (
        forecast_df.groupby("family", as_index=False)
        .agg(
            mean_exact_cv_rmse=("exact_cv_rmse", "mean"),
            mean_grouped_shot_cv_rmse=("grouped_shot_cv_rmse_mean", "mean"),
            mean_relative_cv_degradation_pct=(
                "relative_cv_degradation_pct", "mean"
            ),
            mean_feature_estimation_rmse=("feature_estimation_rmse_mean", "mean"),
            max_algebra_error=("algebra_max_abs_error", "max"),
        )
        .merge(cost_df, on="family", how="left")
        .sort_values("mean_grouped_shot_cv_rmse")
    )

    summary.to_csv(
        RESULTS / "07_04a_measurement_summary.csv", index=False
    )

    print()
    print("=" * 124)
    print("MEASUREMENT-CIRCUIT SUMMARY")
    print("=" * 124)
    print(summary.to_string(index=False))
    print()
    print("Saved:")
    print("  results/07_04a_measurement_algebra_audit.csv")
    print("  results/07_04a_grouped_shot_forecast_results.csv")
    print("  results/07_04a_measurement_setting_costs.csv")
    print("  results/07_04a_measurement_summary.csv")
    print()
    print("Step 7.4A finished.")
    print("Interpret the circuit-level measurement audit before adding ancilla or hardware noise.")


if __name__ == "__main__":
    main()
