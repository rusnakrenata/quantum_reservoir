#!/usr/bin/env python
from __future__ import annotations

"""
WEEK 11.1E.9E — CANDIDATE #3 MEASURED-RESET ERROR INJECTION
================================================================

NO QPU job is submitted.
2026 is never loaded.

Question
--------
Can the measured mid-circuit reset imperfection on the current Candidate #3
layout explain a substantial fraction of the observed real-QPU collapse?

Frozen candidate:
    RWP_H2_R3
    H2, W=2, r=2
    alpha=0.75
    XZinj_dropX3
    frozen training-only Ridge readout

Measured reset layout:
    q0 -> P105
    q1 -> P117
    q2 -> P125
    q3 -> P124

Models
------
1. IDEAL:
   perfect reset between W=2 steps.

2. REPLACEMENT:
   after the first reservoir step, discard the injection subsystem exactly
   as an ideal reset would, but reprepare each injection qubit as

       rho_q = (1-p_q)|0><0| + p_q|1><1|.

   This models residual excited-state preparation after reset.

3. AMPLITUDE_DAMPING:
   apply an independent finite amplitude-damping channel to each injection
   qubit before the second input, calibrated so that

       |1><1| -> residual excited population p_q.

   This retains a small amount of the pre-reset state/correlation and is a
   complementary first-order model.

Scenarios use the measured same-job readout-corrected reset residuals from
11_1E9D, including q2-only isolation.

The script first proves that its custom perfect-reset simulation reproduces
the frozen ideal feature bank before interpreting any imperfect-reset result.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DIRECT_SCRIPT_NAMES = [
    "11_1C_third_candidate_direct_qpu.py",
    "11_1c_third_candidate_direct_qpu.py",
]

FEATURES = ["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3"]

PHYSICAL_BY_LOGICAL = {
    0: 105,
    1: 117,
    2: 125,
    3: 124,
}

# Fallbacks from the measured 11.1E.9D output.
FALLBACK_SINGLE_RESET1 = np.array(
    [0.012216, 0.008391, 0.040482, 0.008852],
    dtype=float,
)
FALLBACK_PAR_RESET1 = np.array(
    [0.007230, 0.014561, 0.039969, 0.015736],
    dtype=float,
)
FALLBACK_PAR_RESETPLUS = np.array(
    [0.004737, 0.006910, 0.016398, 0.008606],
    dtype=float,
)


def find_direct_script():
    for name in DIRECT_SCRIPT_NAMES:
        p = HERE / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find 11_1C_third_candidate_direct_qpu.py"
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


direct = load_module(
    find_direct_script(),
    "qrc_candidate3_reset_injection",
)


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def bias(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(pred - y))


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def kron_all(items):
    out = np.array([[1.0 + 0.0j]])
    for item in items:
        out = np.kron(out, item)
    return out


def ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array(
        [[c, -s], [s, c]],
        dtype=complex,
    )


I2 = np.eye(2, dtype=complex)
P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)


def injection_pure_density(angle_row):
    # Use the project's own injection-vector implementation so ordering is
    # exactly the same as the frozen ideal QRC.
    phi = direct.common.qrc.injection_vector(
        np.asarray(angle_row, dtype=float)
    )
    return np.outer(phi, phi.conj())


def injection_rotation_full(angle_row):
    local = [ry(float(angle_row[q])) for q in range(4)]
    return kron_all(local + [I2, I2])


def partial_traces_4plus2(rho6):
    """
    Ordering follows the project's U reshape:
        injection dimension 16 first,
        memory dimension 4 second.
    """
    rho4 = np.asarray(rho6, dtype=complex).reshape(16, 4, 16, 4)

    rho_i = np.einsum(
        "ambm->ab",
        rho4,
        optimize=True,
    )
    rho_m = np.einsum(
        "aman->mn",
        rho4,
        optimize=True,
    )

    rho_i = 0.5 * (rho_i + rho_i.conj().T)
    rho_m = 0.5 * (rho_m + rho_m.conj().T)

    # Numerical trace hygiene.
    rho_i /= np.trace(rho_i)
    rho_m /= np.trace(rho_m)

    return rho_i, rho_m


def evolve(U, rho_i, rho_m):
    rho_pre = np.kron(rho_i, rho_m)
    rho = U @ rho_pre @ U.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho


def feature_row_from_rho_i(rho_i):
    row = {}
    for feature in FEATURES:
        row[feature] = float(
            np.real_if_close(
                np.trace(
                    rho_i @ direct.common.INJ_OPS[feature]
                ),
                tol=1000,
            ).real
        )
    return row


def replacement_reset_then_inject(rho_after_step1, second_angles, p):
    """
    Perfectly discard old injection state/correlations, preserve the memory
    reduced state, then reprepare imperfect reset states and apply the
    second-step RY input encoding.
    """
    _, rho_m = partial_traces_4plus2(rho_after_step1)

    reset_local = [
        (1.0 - float(p[q])) * P0
        + float(p[q]) * P1
        for q in range(4)
    ]
    rho_i_reset = kron_all(reset_local)

    Uinj = kron_all([ry(float(a)) for a in second_angles])
    rho_i_encoded = Uinj @ rho_i_reset @ Uinj.conj().T

    return rho_i_encoded, rho_m


def embedded_single_operator(local_op, q):
    ops = [I2 for _ in range(6)]
    ops[int(q)] = np.asarray(local_op, dtype=complex)
    return kron_all(ops)


def amplitude_damping_on_qubit(rho, q, p_residual):
    """
    Finite amplitude damping with gamma=1-p_residual.

    For an initial |1>, the output excited population is p_residual.
    """
    p = float(np.clip(p_residual, 0.0, 1.0))
    gamma = 1.0 - p

    E0 = np.array(
        [
            [1.0, 0.0],
            [0.0, np.sqrt(max(0.0, 1.0 - gamma))],
        ],
        dtype=complex,
    )
    E1 = np.array(
        [
            [0.0, np.sqrt(max(0.0, gamma))],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    K0 = embedded_single_operator(E0, q)
    K1 = embedded_single_operator(E1, q)

    out = (
        K0 @ rho @ K0.conj().T
        + K1 @ rho @ K1.conj().T
    )
    out = 0.5 * (out + out.conj().T)
    out /= np.trace(out)
    return out


def amplitude_damping_reset_then_inject(
    rho_after_step1,
    second_angles,
    p,
):
    rho = np.asarray(rho_after_step1, dtype=complex).copy()

    for q in range(4):
        rho = amplitude_damping_on_qubit(
            rho,
            q,
            float(p[q]),
        )

    Uinj = injection_rotation_full(second_angles)
    rho = Uinj @ rho @ Uinj.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)

    return rho


def perfect_reset_then_inject(rho_after_step1, second_angles):
    _, rho_m = partial_traces_4plus2(rho_after_step1)
    rho_i = injection_pure_density(second_angles)
    return rho_i, rho_m


def load_measured_reset_probabilities():
    """
    Prefer saved 11.1E.9D CSVs. Fall back to the values from the console
    output if files are not present.
    """
    p_single = FALLBACK_SINGLE_RESET1.copy()
    p_par1 = FALLBACK_PAR_RESET1.copy()
    p_parplus = FALLBACK_PAR_RESETPLUS.copy()

    single_file = (
        RESULTS /
        "11_1E9D_candidate3_reset_single_diagnostics.csv"
    )
    parallel_file = (
        RESULTS /
        "11_1E9D_candidate3_reset_parallel_diagnostics.csv"
    )

    if single_file.exists():
        df = pd.read_csv(single_file)
        vals = []
        for q in range(4):
            pphys = PHYSICAL_BY_LOGICAL[q]
            sub = df[
                (df["physical_qubit"].astype(int) == pphys)
                & (df["test"].astype(str) == "RESET1")
            ]
            if len(sub) != 1:
                raise RuntimeError(
                    f"Expected one RESET1 row for P{pphys}, found {len(sub)}"
                )
            vals.append(
                float(
                    sub.iloc[0][
                        "corrected_P1_after_reset_clipped"
                    ]
                )
            )
        p_single = np.asarray(vals, dtype=float)

    if parallel_file.exists():
        df = pd.read_csv(parallel_file)

        def get_parallel(kind):
            vals = []
            for q in range(4):
                pphys = PHYSICAL_BY_LOGICAL[q]
                sub = df[
                    (df["physical_qubit"].astype(int) == pphys)
                    & (df["parallel_test"].astype(str) == kind)
                ]
                if len(sub) != 1:
                    raise RuntimeError(
                        f"Expected one {kind} row for P{pphys}, "
                        f"found {len(sub)}"
                    )
                vals.append(
                    float(
                        sub.iloc[0][
                            "corrected_marginal_P1_clipped"
                        ]
                    )
                )
            return np.asarray(vals, dtype=float)

        p_par1 = get_parallel("PAR_RESET1")
        p_parplus = get_parallel("PAR_RESETPLUS")

    return {
        "single_RESET1": p_single,
        "parallel_RESET1": p_par1,
        "parallel_RESETPLUS": p_parplus,
    }


def simulate_window2_features(
    U,
    angles,
    endpoints,
    mode,
    p=None,
):
    rows = []

    rho_m0 = direct.common.qrc.memory_zero_density()

    for endpoint in endpoints:
        endpoint = int(endpoint)
        first_idx = endpoint - 1
        second_idx = endpoint

        # First RWP input is always ideal/fresh.
        rho_i1 = injection_pure_density(angles[first_idx])
        rho1 = evolve(U, rho_i1, rho_m0)

        # Mid-window reset and second input.
        if mode == "ideal":
            rho_i2, rho_m1 = perfect_reset_then_inject(
                rho1,
                angles[second_idx],
            )
            rho2 = evolve(U, rho_i2, rho_m1)

        elif mode == "replacement":
            rho_i2, rho_m1 = replacement_reset_then_inject(
                rho1,
                angles[second_idx],
                p,
            )
            rho2 = evolve(U, rho_i2, rho_m1)

        elif mode == "amplitude_damping":
            rho_pre2 = amplitude_damping_reset_then_inject(
                rho1,
                angles[second_idx],
                p,
            )
            rho2 = U @ rho_pre2 @ U.conj().T
            rho2 = 0.5 * (rho2 + rho2.conj().T)
            rho2 /= np.trace(rho2)

        else:
            raise ValueError(mode)

        rho_i_out, _ = partial_traces_4plus2(rho2)
        rows.append(feature_row_from_rho_i(rho_i_out))

    return pd.DataFrame(rows)[FEATURES].to_numpy(dtype=float)


def load_latest_qpu_reference(frozen):
    """
    Optional comparison with the latest direct no-M3 Candidate #3 output.
    The simulation itself does not depend on this file.
    """
    path = (
        RESULTS /
        "11_1C_third_candidate_qpu_features_predictions.csv"
    )

    if not path.exists():
        return None

    df = pd.read_csv(path)

    required = (
        ["endpoint", "target", "pred_ideal", "pred_qpu"]
        + [f"{f}_ideal" for f in FEATURES]
        + [f"{f}_qpu" for f in FEATURES]
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(
            "[QPU comparison] latest file exists but lacks columns: "
            + ", ".join(missing)
        )
        return None

    return df


def feature_diagnostics(name, X, X_ideal, X_qpu=None):
    rows = []

    for j, f in enumerate(FEATURES):
        delta = X[:, j] - X_ideal[:, j]
        sigma = float(np.std(X_ideal[:, j], ddof=0))

        row = {
            "scenario": name,
            "feature": f,
            "bias_vs_ideal": float(np.mean(delta)),
            "rmse_vs_ideal": float(np.sqrt(np.mean(delta ** 2))),
            "corr_vs_ideal": corr(X[:, j], X_ideal[:, j]),
            "ideal_std": sigma,
            "noise_to_signal": (
                np.nan
                if sigma <= 0
                else float(np.sqrt(np.mean(delta ** 2)) / sigma)
            ),
        }

        if X_qpu is not None:
            qd = X_qpu[:, j] - X_ideal[:, j]
            q_bias = float(np.mean(qd))
            q_rmse = float(np.sqrt(np.mean(qd ** 2)))

            row["qpu_bias_vs_ideal"] = q_bias
            row["qpu_rmse_vs_ideal"] = q_rmse
            row["signed_bias_fraction_of_qpu"] = (
                np.nan if abs(q_bias) < 1e-15
                else float(row["bias_vs_ideal"] / q_bias)
            )
            row["rmse_fraction_of_qpu"] = (
                np.nan if q_rmse < 1e-15
                else float(row["rmse_vs_ideal"] / q_rmse)
            )

        rows.append(row)

    return rows


def main():
    print("=" * 124)
    print("WEEK 11.1E.9E — CANDIDATE #3 MEASURED-RESET ERROR INJECTION")
    print("=" * 124)
    print("NO QPU JOB WILL BE SUBMITTED.")
    print("2026 remains FROZEN / NOT LOADED.")
    print()

    candidate, manifest_meta = direct.load_candidate()
    frozen = direct.build_frozen_readout(
        candidate,
        manifest_meta,
    )

    if int(candidate["window"]) != 2:
        raise RuntimeError("This diagnostic is written for Candidate #3 W=2.")

    U = direct.common.build_trotter_unitary(candidate)

    reset_p = load_measured_reset_probabilities()

    print("Frozen Candidate #3:")
    print(f"  topology={candidate['topology']}")
    print(f"  W={candidate['window']}")
    print(f"  r={candidate['r']}")
    print(f"  ideal 2025 RMSE={frozen['ideal_rmse']:.6f}")
    print()

    print("Measured corrected reset residuals:")
    for name, p in reset_p.items():
        print(
            f"  {name:20s}: "
            + ", ".join(
                f"q{q}/P{PHYSICAL_BY_LOGICAL[q]}={100*p[q]:.3f}%"
                for q in range(4)
            )
        )
    print()

    endpoints = np.asarray(frozen["val_endpoints"], dtype=int)
    X_frozen = np.asarray(frozen["X_val"], dtype=float)
    y = np.asarray(frozen["y_val"], dtype=float)
    pred_frozen = np.asarray(frozen["pred_val"], dtype=float)

    # ------------------------------------------------------------------
    # Independent ideal audit of the custom full-density implementation.
    # ------------------------------------------------------------------
    print("Running custom perfect-reset audit...")
    X_ideal_custom = simulate_window2_features(
        U=U,
        angles=frozen["angles"],
        endpoints=endpoints,
        mode="ideal",
    )

    ideal_feature_rmse = rmse(
        X_ideal_custom.reshape(-1),
        X_frozen.reshape(-1),
    )
    ideal_feature_max = float(
        np.max(np.abs(X_ideal_custom - X_frozen))
    )

    pred_ideal_custom = direct.common.predict_scaled_ridge(
        frozen["model"],
        frozen["scaler"],
        frozen["keep"],
        X_ideal_custom,
    )

    pred_ideal_audit = rmse(
        pred_ideal_custom,
        pred_frozen,
    )

    print(f"  feature RMSE vs frozen ideal: {ideal_feature_rmse:.12e}")
    print(f"  max |feature difference|:    {ideal_feature_max:.12e}")
    print(f"  prediction RMSE vs frozen:   {pred_ideal_audit:.12e}")

    if (
        ideal_feature_rmse > 1e-10
        or ideal_feature_max > 1e-9
        or pred_ideal_audit > 1e-8
    ):
        raise RuntimeError(
            "Custom ideal simulation does not reproduce the frozen "
            "Candidate #3 feature bank. Stop before interpreting reset noise."
        )

    print("  IDEAL AUDIT: PASS")
    print()

    # ------------------------------------------------------------------
    # Define measured-reset scenarios.
    # ------------------------------------------------------------------
    p_q2_only = np.zeros(4, dtype=float)
    p_q2_only[2] = reset_p["parallel_RESET1"][2]

    scenarios = [
        {
            "name": "replacement_parallel_RESETPLUS",
            "mode": "replacement",
            "p": reset_p["parallel_RESETPLUS"],
        },
        {
            "name": "replacement_parallel_RESET1",
            "mode": "replacement",
            "p": reset_p["parallel_RESET1"],
        },
        {
            "name": "replacement_q2_only",
            "mode": "replacement",
            "p": p_q2_only,
        },
        {
            "name": "ampdamp_single_RESET1",
            "mode": "amplitude_damping",
            "p": reset_p["single_RESET1"],
        },
        {
            "name": "ampdamp_parallel_RESET1",
            "mode": "amplitude_damping",
            "p": reset_p["parallel_RESET1"],
        },
        {
            "name": "ampdamp_q2_only",
            "mode": "amplitude_damping",
            "p": p_q2_only,
        },
    ]

    qpu_df = load_latest_qpu_reference(frozen)

    X_qpu = None
    qpu_pred = None
    qpu_endpoints = None

    if qpu_df is not None:
        qpu_endpoints = qpu_df["endpoint"].to_numpy(dtype=int)

        if not np.array_equal(qpu_endpoints, endpoints):
            raise RuntimeError(
                "Latest QPU feature file does not contain the full aligned "
                "365 Candidate #3 validation endpoints."
            )

        X_qpu = qpu_df[
            [f"{f}_qpu" for f in FEATURES]
        ].to_numpy(dtype=float)
        qpu_pred = qpu_df["pred_qpu"].to_numpy(dtype=float)

        print(
            "[QPU comparison] loaded latest direct Candidate #3 result:"
        )
        print(f"  QPU RMSE={rmse(y, qpu_pred):.6f}")
        print(
            f"  QPU feature RMSE={rmse(X_qpu.reshape(-1), X_frozen.reshape(-1)):.6f}"
        )
        print()

    summary_rows = []
    feature_rows = []
    prediction_rows = []

    # Ideal row.
    summary_rows.append({
        "scenario": "ideal",
        "reset_model": "perfect",
        "forecast_rmse": rmse(y, pred_frozen),
        "forecast_mae": mae(y, pred_frozen),
        "forecast_bias": bias(y, pred_frozen),
        "pred_corr_vs_ideal": 1.0,
        "pred_rmse_vs_ideal": 0.0,
        "feature_rmse_vs_ideal": 0.0,
    })

    for i, endpoint in enumerate(endpoints):
        prediction_rows.append({
            "scenario": "ideal",
            "endpoint": int(endpoint),
            "target": float(y[i]),
            "prediction": float(pred_frozen[i]),
            "pred_minus_ideal": 0.0,
        })

    # Measured reset scenarios.
    for spec in scenarios:
        name = spec["name"]
        print(f"Simulating {name}...")

        X = simulate_window2_features(
            U=U,
            angles=frozen["angles"],
            endpoints=endpoints,
            mode=spec["mode"],
            p=spec["p"],
        )

        pred = direct.common.predict_scaled_ridge(
            frozen["model"],
            frozen["scaler"],
            frozen["keep"],
            X,
        )

        f_rmse = rmse(
            X.reshape(-1),
            X_frozen.reshape(-1),
        )

        row = {
            "scenario": name,
            "reset_model": spec["mode"],
            "p_q0": float(spec["p"][0]),
            "p_q1": float(spec["p"][1]),
            "p_q2": float(spec["p"][2]),
            "p_q3": float(spec["p"][3]),
            "forecast_rmse": rmse(y, pred),
            "forecast_mae": mae(y, pred),
            "forecast_bias": bias(y, pred),
            "pred_corr_vs_ideal": corr(pred, pred_frozen),
            "pred_rmse_vs_ideal": rmse(pred, pred_frozen),
            "feature_rmse_vs_ideal": f_rmse,
        }

        if qpu_pred is not None:
            qpu_delta_rmse = rmse(qpu_pred, pred_frozen)
            row["latest_qpu_rmse"] = rmse(y, qpu_pred)
            row["latest_qpu_pred_rmse_vs_ideal"] = qpu_delta_rmse
            row["pred_distortion_fraction_of_qpu"] = (
                np.nan
                if qpu_delta_rmse < 1e-15
                else row["pred_rmse_vs_ideal"] / qpu_delta_rmse
            )

        summary_rows.append(row)

        feature_rows.extend(
            feature_diagnostics(
                name,
                X,
                X_frozen,
                X_qpu=X_qpu,
            )
        )

        for i, endpoint in enumerate(endpoints):
            pr = {
                "scenario": name,
                "endpoint": int(endpoint),
                "target": float(y[i]),
                "prediction": float(pred[i]),
                "pred_ideal": float(pred_frozen[i]),
                "pred_minus_ideal": float(
                    pred[i] - pred_frozen[i]
                ),
            }

            for j, f in enumerate(FEATURES):
                pr[f"{f}_scenario"] = float(X[i, j])
                pr[f"{f}_ideal"] = float(X_frozen[i, j])
                pr[f"{f}_delta"] = float(
                    X[i, j] - X_frozen[i, j]
                )

            prediction_rows.append(pr)

    summary_df = pd.DataFrame(summary_rows)
    feature_df = pd.DataFrame(feature_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    print()
    print("=" * 124)
    print("MEASURED-RESET SIMULATION SUMMARY")
    print("=" * 124)

    cols = [
        "scenario",
        "forecast_rmse",
        "forecast_bias",
        "pred_rmse_vs_ideal",
        "pred_corr_vs_ideal",
        "feature_rmse_vs_ideal",
    ]

    if "pred_distortion_fraction_of_qpu" in summary_df.columns:
        cols.append("pred_distortion_fraction_of_qpu")

    print(
        summary_df[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 124)
    print("PRIMARY Z2 COMPARISON")
    print("=" * 124)

    z2 = feature_df[
        feature_df["feature"] == "Z2"
    ].copy()

    z2_cols = [
        "scenario",
        "bias_vs_ideal",
        "rmse_vs_ideal",
        "corr_vs_ideal",
        "noise_to_signal",
    ]

    if X_qpu is not None:
        z2_cols += [
            "qpu_bias_vs_ideal",
            "qpu_rmse_vs_ideal",
            "signed_bias_fraction_of_qpu",
            "rmse_fraction_of_qpu",
        ]

    print(
        z2[z2_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ------------------------------------------------------------------
    # Small q2-only sensitivity sweep.
    # ------------------------------------------------------------------
    print()
    print("=" * 124)
    print("Q2/P125 RESET SENSITIVITY SWEEP — REPLACEMENT MODEL")
    print("=" * 124)

    sweep_rows = []

    for p2 in [0.0, 0.01, 0.02, 0.039969, 0.06, 0.10, 0.20]:
        p = np.zeros(4, dtype=float)
        p[2] = p2

        X = simulate_window2_features(
            U=U,
            angles=frozen["angles"],
            endpoints=endpoints,
            mode="replacement",
            p=p,
        )

        pred = direct.common.predict_scaled_ridge(
            frozen["model"],
            frozen["scaler"],
            frozen["keep"],
            X,
        )

        z2_delta = X[:, 5] - X_frozen[:, 5]

        sweep_rows.append({
            "p2": p2,
            "Z2_bias": float(np.mean(z2_delta)),
            "Z2_rmse": float(
                np.sqrt(np.mean(z2_delta ** 2))
            ),
            "forecast_rmse": rmse(y, pred),
            "pred_rmse_vs_ideal": rmse(
                pred,
                pred_frozen,
            ),
        })

    sweep_df = pd.DataFrame(sweep_rows)

    print(
        sweep_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    summary_file = (
        RESULTS /
        "11_1E9E_candidate3_reset_injection_summary.csv"
    )
    feature_file = (
        RESULTS /
        "11_1E9E_candidate3_reset_injection_features.csv"
    )
    pred_file = (
        RESULTS /
        "11_1E9E_candidate3_reset_injection_predictions.csv"
    )
    sweep_file = (
        RESULTS /
        "11_1E9E_candidate3_q2_reset_sweep.csv"
    )
    json_file = (
        RESULTS /
        "11_1E9E_candidate3_reset_injection_summary.json"
    )

    summary_df.to_csv(summary_file, index=False)
    feature_df.to_csv(feature_file, index=False)
    predictions_df.to_csv(pred_file, index=False)
    sweep_df.to_csv(sweep_file, index=False)

    payload = {
        "candidate": direct.CANDIDATE_KEY,
        "ideal_audit": {
            "feature_rmse": ideal_feature_rmse,
            "feature_max_abs": ideal_feature_max,
            "prediction_rmse": pred_ideal_audit,
        },
        "measured_reset_probabilities": {
            k: [float(x) for x in v]
            for k, v in reset_p.items()
        },
        "summary": summary_df.to_dict(orient="records"),
        "q2_sweep": sweep_df.to_dict(orient="records"),
        "latest_qpu_loaded": bool(X_qpu is not None),
        "2026_loaded": False,
    }

    json_file.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(f"  {summary_file}")
    print(f"  {feature_file}")
    print(f"  {pred_file}")
    print(f"  {sweep_file}")
    print(f"  {json_file}")
    print()
    print("2026 remains FROZEN / UNUSED.")


if __name__ == "__main__":
    main()
