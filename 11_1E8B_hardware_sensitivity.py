"""
WEEK 11.1E.8B — FEATURE-LEVEL HARDWARE SENSITIVITY
==================================================

Purpose
-------
Quantify how strongly each frozen QRC observable is affected by real-QPU error,
relative to its useful ideal variation, and how strongly the frozen Ridge
readout amplifies that error.

This is a POST-HARDWARE DIAGNOSTIC ONLY.

It:
  * DOES NOT submit a QPU job.
  * DOES NOT load 2026.
  * DOES NOT fit mitigation/calibration.
  * DOES NOT change any frozen model.
  * Uses the corrected 11.1E.8 feature diagnostics.

Input
-----
results/11_1E8_corrected_feature_diagnostics.csv

Expected columns
----------------
candidate
feature
effective_raw_readout_beta
feature_bias_qpu_minus_ideal
feature_rmse
feature_corr_qpu_vs_ideal
ideal_feature_std
qpu_feature_std
qpu_to_ideal_std_ratio

Definitions
-----------
Let

    delta_j(t) = x_QPU,j(t) - x_ideal,j(t)

and let beta_j be the effective frozen Ridge coefficient in raw feature
coordinates.

1) Hardware feature RMSE

    E_j = RMSE(delta_j)

This measures total QPU-vs-ideal distortion of feature j.

2) Hardware feature bias

    Bx_j = mean(delta_j)

This is the systematic offset of the feature.

3) Stochastic / non-bias component

Using population moments,

    E_j^2 = Bx_j^2 + S_j^2

so

    S_j = sqrt(max(E_j^2 - Bx_j^2, 0))

This is the RMS feature distortion that remains after removing the constant
offset.

4) Noise-to-signal ratio

    N_j = E_j / sigma_ideal,j

where sigma_ideal,j is the standard deviation of the useful ideal feature over
the 2025 validation endpoints.

Interpretation:
    N_j < 1 : total hardware error is smaller than the natural ideal variation.
    N_j = 1 : hardware error is as large as the useful ideal variation.
    N_j > 1 : hardware error exceeds the feature's natural ideal variation.

No arbitrary threshold is used for ranking; the actual continuous N_j is saved.

5) Normalized hardware sensitivity

    H_j = |beta_j| * N_j
        = |beta_j| * E_j / sigma_ideal,j

This combines:
    a) how noisy the feature is relative to its useful signal, and
    b) how strongly the frozen readout depends on that feature.

Because the quantum observables are dimensionless expectation values, beta_j
is expressed in target units (claims), so H_j is also reported in claims-like
readout units. H_j is a comparative vulnerability index; it is NOT itself the
actual forecast error.

6) Direct RMS prediction perturbation

    P_j = |beta_j| * E_j

This IS the direct RMS size, in claims, of feature j's contribution to the
QPU-vs-ideal prediction perturbation if considered individually.

7) Systematic prediction shift from feature j

    M_j = beta_j * Bx_j

This keeps the sign and tells us whether the feature pushes predictions upward
or downward on average.

8) Random prediction perturbation from feature j

    R_j = |beta_j| * S_j

This measures the non-constant component of the feature's forecast distortion.

Important:
----------
The individual P_j and R_j values do NOT add linearly to total prediction RMSE,
because errors from different features can cancel or correlate.

Outputs
-------
results/11_1E8B_hardware_sensitivity.csv
results/11_1E8B_candidate_summary.csv
results/11_1E8B_top_features.csv
results/11_1E8B_report.txt
results/11_1E8B_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

RESULTS = Path("results")
INPUT = RESULTS / "11_1E8_corrected_feature_diagnostics.csv"

OUT_DETAIL = RESULTS / "11_1E8B_hardware_sensitivity.csv"
OUT_CANDIDATE = RESULTS / "11_1E8B_candidate_summary.csv"
OUT_TOP = RESULTS / "11_1E8B_top_features.csv"
OUT_REPORT = RESULTS / "11_1E8B_report.txt"
OUT_JSON = RESULTS / "11_1E8B_summary.json"

TOP_K = 5

REQUIRED = [
    "candidate",
    "feature",
    "effective_raw_readout_beta",
    "feature_bias_qpu_minus_ideal",
    "feature_rmse",
    "feature_corr_qpu_vs_ideal",
    "ideal_feature_std",
    "qpu_feature_std",
    "qpu_to_ideal_std_ratio",
]


# =============================================================================
# LOAD + SAFETY CHECKS
# =============================================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Required corrected diagnostic file not found:\n  {INPUT}\n"
        "Run corrected Week 11.1E.8 first."
    )

df = pd.read_csv(INPUT)

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise RuntimeError(
        "Input file is missing required columns:\n  "
        + "\n  ".join(missing)
    )

if df.empty:
    raise RuntimeError("Input feature diagnostics file is empty.")

for col in REQUIRED[2:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[REQUIRED[2:]].isna().any().any():
    bad = df[df[REQUIRED[2:]].isna().any(axis=1)]
    raise RuntimeError(
        "Non-numeric or missing diagnostic values detected:\n"
        + bad.to_string(index=False)
    )

if (df["ideal_feature_std"] <= 0).any():
    bad = df[df["ideal_feature_std"] <= 0][
        ["candidate", "feature", "ideal_feature_std"]
    ]
    raise RuntimeError(
        "Cannot normalize features with non-positive ideal std:\n"
        + bad.to_string(index=False)
    )


# =============================================================================
# CORE QUANTITIES
# =============================================================================

beta = df["effective_raw_readout_beta"].to_numpy(float)
bias_x = df["feature_bias_qpu_minus_ideal"].to_numpy(float)
rmse_x = df["feature_rmse"].to_numpy(float)
sigma_ideal = df["ideal_feature_std"].to_numpy(float)

# Population non-bias RMS component:
#
#     RMSE^2 = bias^2 + variance(error)
#
# Numerical clipping protects against tiny negative round-off.
stochastic_feature_rms = np.sqrt(
    np.maximum(rmse_x**2 - bias_x**2, 0.0)
)

noise_to_signal = rmse_x / sigma_ideal

# The requested normalized hardware sensitivity.
hardware_sensitivity = np.abs(beta) * noise_to_signal

# Direct readout effects.
prediction_perturbation_rms = np.abs(beta) * rmse_x
systematic_prediction_shift = beta * bias_x
random_prediction_perturbation_rms = (
    np.abs(beta) * stochastic_feature_rms
)

# Useful extra diagnostics.
bias_fraction_of_feature_mse = np.divide(
    bias_x**2,
    rmse_x**2,
    out=np.zeros_like(bias_x),
    where=rmse_x > 0,
)

stochastic_fraction_of_feature_mse = 1.0 - bias_fraction_of_feature_mse

# How much ideal target variation this feature can create through the readout.
# This is not a quality metric; it is the natural prediction-scale signal of
# this feature.
ideal_prediction_signal_std = np.abs(beta) * sigma_ideal

# Check:
# prediction_perturbation_rms / ideal_prediction_signal_std
# must equal noise_to_signal wherever beta != 0.
ratio_check = np.divide(
    prediction_perturbation_rms,
    ideal_prediction_signal_std,
    out=np.full_like(prediction_perturbation_rms, np.nan),
    where=ideal_prediction_signal_std > 0,
)


# =============================================================================
# BUILD DETAIL TABLE
# =============================================================================

out = df.copy()

out["feature_stochastic_rms"] = stochastic_feature_rms
out["noise_to_signal_ratio"] = noise_to_signal
out["hardware_sensitivity"] = hardware_sensitivity

out["ideal_prediction_signal_std_claims"] = ideal_prediction_signal_std
out["prediction_perturbation_rms_claims"] = prediction_perturbation_rms
out["systematic_prediction_shift_claims"] = systematic_prediction_shift
out["random_prediction_perturbation_rms_claims"] = (
    random_prediction_perturbation_rms
)

out["bias_fraction_of_feature_mse"] = bias_fraction_of_feature_mse
out["stochastic_fraction_of_feature_mse"] = (
    stochastic_fraction_of_feature_mse
)
out["noise_to_signal_check"] = ratio_check

# Transparent ranks. No weighted aggregate score.
out["rank_hardware_sensitivity"] = (
    out.groupby("candidate")["hardware_sensitivity"]
    .rank(method="first", ascending=False)
    .astype(int)
)

out["rank_direct_prediction_perturbation"] = (
    out.groupby("candidate")["prediction_perturbation_rms_claims"]
    .rank(method="first", ascending=False)
    .astype(int)
)

out["rank_abs_systematic_shift"] = (
    out.assign(
        _abs_shift=np.abs(out["systematic_prediction_shift_claims"])
    )
    .groupby("candidate")["_abs_shift"]
    .rank(method="first", ascending=False)
    .astype(int)
)

out["rank_random_prediction_perturbation"] = (
    out.groupby("candidate")["random_prediction_perturbation_rms_claims"]
    .rank(method="first", ascending=False)
    .astype(int)
)

out = out.sort_values(
    ["candidate", "rank_hardware_sensitivity", "feature"]
).reset_index(drop=True)

out.to_csv(OUT_DETAIL, index=False)


# =============================================================================
# CANDIDATE SUMMARY
# =============================================================================

candidate_rows = []

for candidate, g in out.groupby("candidate", sort=True):
    # Identify the single most vulnerable features under distinct diagnostics.
    top_H = g.loc[g["hardware_sensitivity"].idxmax()]
    top_P = g.loc[g["prediction_perturbation_rms_claims"].idxmax()]
    top_M = g.loc[
        np.abs(g["systematic_prediction_shift_claims"]).idxmax()
    ]
    top_R = g.loc[
        g["random_prediction_perturbation_rms_claims"].idxmax()
    ]

    candidate_rows.append({
        "candidate": candidate,
        "n_features": len(g),

        "median_noise_to_signal_ratio": float(
            g["noise_to_signal_ratio"].median()
        ),
        "max_noise_to_signal_ratio": float(
            g["noise_to_signal_ratio"].max()
        ),

        "median_hardware_sensitivity": float(
            g["hardware_sensitivity"].median()
        ),
        "max_hardware_sensitivity": float(
            g["hardware_sensitivity"].max()
        ),

        "top_hardware_sensitivity_feature": top_H["feature"],
        "top_hardware_sensitivity": float(
            top_H["hardware_sensitivity"]
        ),
        "top_hardware_sensitivity_noise_to_signal": float(
            top_H["noise_to_signal_ratio"]
        ),
        "top_hardware_sensitivity_corr": float(
            top_H["feature_corr_qpu_vs_ideal"]
        ),

        "top_direct_prediction_perturbation_feature": top_P["feature"],
        "top_direct_prediction_perturbation_rms_claims": float(
            top_P["prediction_perturbation_rms_claims"]
        ),

        "top_systematic_shift_feature": top_M["feature"],
        "top_systematic_shift_claims": float(
            top_M["systematic_prediction_shift_claims"]
        ),

        "top_random_perturbation_feature": top_R["feature"],
        "top_random_perturbation_rms_claims": float(
            top_R["random_prediction_perturbation_rms_claims"]
        ),

        "mean_feature_corr_qpu_vs_ideal": float(
            g["feature_corr_qpu_vs_ideal"].mean()
        ),
        "min_feature_corr_qpu_vs_ideal": float(
            g["feature_corr_qpu_vs_ideal"].min()
        ),
    })

candidate_summary = pd.DataFrame(candidate_rows)
candidate_summary.to_csv(OUT_CANDIDATE, index=False)


# =============================================================================
# TOP FEATURES TABLE
# =============================================================================

top_rows = []

for candidate, g in out.groupby("candidate", sort=True):
    top = g.nsmallest(TOP_K, "rank_hardware_sensitivity")

    for _, r in top.iterrows():
        top_rows.append({
            "candidate": candidate,
            "rank": int(r["rank_hardware_sensitivity"]),
            "feature": r["feature"],
            "beta": r["effective_raw_readout_beta"],
            "ideal_std": r["ideal_feature_std"],
            "feature_bias": r["feature_bias_qpu_minus_ideal"],
            "feature_rmse": r["feature_rmse"],
            "stochastic_feature_rms": r["feature_stochastic_rms"],
            "noise_to_signal_ratio": r["noise_to_signal_ratio"],
            "hardware_sensitivity": r["hardware_sensitivity"],
            "prediction_perturbation_rms_claims": (
                r["prediction_perturbation_rms_claims"]
            ),
            "systematic_prediction_shift_claims": (
                r["systematic_prediction_shift_claims"]
            ),
            "random_prediction_perturbation_rms_claims": (
                r["random_prediction_perturbation_rms_claims"]
            ),
            "feature_corr_qpu_vs_ideal": (
                r["feature_corr_qpu_vs_ideal"]
            ),
            "qpu_to_ideal_std_ratio": (
                r["qpu_to_ideal_std_ratio"]
            ),
        })

top_df = pd.DataFrame(top_rows)
top_df.to_csv(OUT_TOP, index=False)


# =============================================================================
# CONSOLE
# =============================================================================

print("=" * 132)
print("WEEK 11.1E.8B — FEATURE-LEVEL HARDWARE SENSITIVITY")
print("=" * 132)
print("2025 diagnostics only | no QPU job | no mitigation fitting | 2026 NOT LOADED")
print()

for candidate, g in out.groupby("candidate", sort=True):
    print("-" * 132)
    print(candidate)
    print()

    cols = [
        "rank_hardware_sensitivity",
        "feature",
        "effective_raw_readout_beta",
        "ideal_feature_std",
        "feature_bias_qpu_minus_ideal",
        "feature_rmse",
        "feature_stochastic_rms",
        "noise_to_signal_ratio",
        "hardware_sensitivity",
        "prediction_perturbation_rms_claims",
        "systematic_prediction_shift_claims",
        "random_prediction_perturbation_rms_claims",
        "feature_corr_qpu_vs_ideal",
    ]

    printable = g[cols].copy()

    rename = {
        "rank_hardware_sensitivity": "rank",
        "effective_raw_readout_beta": "beta",
        "ideal_feature_std": "sigma_ideal",
        "feature_bias_qpu_minus_ideal": "bias_x",
        "feature_rmse": "rmse_x",
        "feature_stochastic_rms": "random_x",
        "noise_to_signal_ratio": "noise/signal",
        "hardware_sensitivity": "H",
        "prediction_perturbation_rms_claims": "pred_rms",
        "systematic_prediction_shift_claims": "pred_bias",
        "random_prediction_perturbation_rms_claims": "pred_random",
        "feature_corr_qpu_vs_ideal": "corr",
    }
    printable = printable.rename(columns=rename)

    print(
        printable.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    print()

print("=" * 132)
print("CANDIDATE SUMMARY")
print("=" * 132)
print(
    candidate_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# REPORT
# =============================================================================

lines = []

lines.append("=" * 110)
lines.append("WEEK 11.1E.8B — HARDWARE SENSITIVITY REPORT")
lines.append("=" * 110)
lines.append("")
lines.append("No QPU job. No mitigation fitting. 2026 not loaded.")
lines.append("")
lines.append("DEFINITIONS")
lines.append("-" * 110)
lines.append(
    "delta_j = x_QPU,j - x_ideal,j"
)
lines.append(
    "feature RMSE E_j = RMSE(delta_j)"
)
lines.append(
    "feature bias Bx_j = mean(delta_j)"
)
lines.append(
    "non-bias feature RMS S_j = sqrt(E_j^2 - Bx_j^2)"
)
lines.append(
    "noise-to-signal N_j = E_j / sigma_ideal,j"
)
lines.append(
    "normalized hardware sensitivity H_j = |beta_j| * N_j"
)
lines.append(
    "direct RMS prediction perturbation P_j = |beta_j| * E_j"
)
lines.append(
    "systematic prediction shift M_j = beta_j * Bx_j"
)
lines.append(
    "random prediction perturbation R_j = |beta_j| * S_j"
)
lines.append("")
lines.append(
    "H_j is a comparative vulnerability measure. P_j, M_j, and R_j are "
    "the more direct quantities in claims."
)
lines.append("")
lines.append("TOP VULNERABILITIES")
lines.append("-" * 110)

for candidate, g in out.groupby("candidate", sort=True):
    lines.append(candidate)

    top = g.nsmallest(TOP_K, "rank_hardware_sensitivity")

    for _, r in top.iterrows():
        lines.append(
            f"  #{int(r['rank_hardware_sensitivity'])} "
            f"{r['feature']}: "
            f"H={r['hardware_sensitivity']:.6f}, "
            f"noise/signal={r['noise_to_signal_ratio']:.6f}, "
            f"pred_RMS={r['prediction_perturbation_rms_claims']:.6f}, "
            f"pred_bias={r['systematic_prediction_shift_claims']:+.6f}, "
            f"pred_random={r['random_prediction_perturbation_rms_claims']:.6f}, "
            f"corr={r['feature_corr_qpu_vs_ideal']:.6f}"
        )

    lines.append("")

lines.append("INTERPRETATION")
lines.append("-" * 110)
lines.append(
    "A feature is especially dangerous when all three conditions occur:"
)
lines.append(
    "  1) hardware error is large relative to the ideal feature variation,"
)
lines.append(
    "  2) |beta| is large, so the frozen readout amplifies that error,"
)
lines.append(
    "  3) the error has a large non-bias component or poor ideal/QPU "
    "correlation, so a constant offset cannot repair it."
)
lines.append("")
lines.append(
    "Do not add individual feature RMS perturbations to estimate total forecast "
    "RMSE; feature errors can be correlated and can cancel."
)

OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# JSON
# =============================================================================

payload = {
    "step": "11.1E.8B",
    "scope": "feature-level real-QPU hardware sensitivity",
    "input": str(INPUT),
    "2026_loaded": False,
    "qpu_job_submitted": False,
    "mitigation_fitted": False,
    "formulae": {
        "feature_error": "delta_j = x_QPU,j - x_ideal,j",
        "feature_rmse": "E_j = RMSE(delta_j)",
        "feature_bias": "Bx_j = mean(delta_j)",
        "feature_stochastic_rms": "S_j = sqrt(E_j^2 - Bx_j^2)",
        "noise_to_signal": "N_j = E_j / sigma_ideal,j",
        "hardware_sensitivity": "H_j = |beta_j| * N_j",
        "prediction_perturbation_rms": "P_j = |beta_j| * E_j",
        "systematic_prediction_shift": "M_j = beta_j * Bx_j",
        "random_prediction_perturbation": "R_j = |beta_j| * S_j",
    },
    "candidate_summary": candidate_summary.to_dict(orient="records"),
    "top_features": top_df.to_dict(orient="records"),
}

OUT_JSON.write_text(
    json.dumps(payload, indent=2, default=str),
    encoding="utf-8"
)

print()
print("Saved:")
for p in [
    OUT_DETAIL,
    OUT_CANDIDATE,
    OUT_TOP,
    OUT_REPORT,
    OUT_JSON,
]:
    print(f"  {p}")

print()
print("2026 remains FROZEN / UNUSED.")
