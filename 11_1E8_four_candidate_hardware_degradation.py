"""
WEEK 11.1E.8 — CORRECTED FOUR-CANDIDATE POST-HARDWARE DIAGNOSIS
================================================================

Purpose
-------
Diagnose WHY the already-measured 2025 SamplerV2 QPU results differ from the
matching ideal reservoir results.

This script:
  * DOES NOT submit any QPU job.
  * DOES NOT load 2026 data.
  * DOES NOT fit any mitigation/calibration to the 2025 targets.
  * DOES NOT use Estimator results.
  * DOES NOT auto-pick arbitrary QPU files.
  * Uses Candidate #5 RWP64 ideal features/predictions, NOT full-CONT ideal.

Main diagnostic identity
------------------------
For a frozen linear Ridge readout,

    prediction_QPU - prediction_ideal
        = sum_j beta_j * (feature_QPU_j - feature_ideal_j)

where beta_j are the effective coefficients of the already-frozen readout in
raw feature coordinates.

The script reconstructs those effective coefficients ONLY from the saved ideal
features and saved ideal predictions. This is not refitting to y/targets.
It then verifies that the reconstructed readout reproduces BOTH the saved ideal
and QPU predictions to numerical precision.

If that audit fails, the script stops rather than reporting misleading feature
contributions.

Outputs
-------
results/11_1E8_corrected_candidate_summary.csv
results/11_1E8_corrected_feature_diagnostics.csv
results/11_1E8_corrected_feature_contributions.csv
results/11_1E8_corrected_daily_prediction_shift.csv
results/11_1E8_corrected_report.txt
results/11_1E8_corrected_summary.json
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

OUT = RESULTS / "11_1E8_corrected"

EXPECTED_N = 365

# IMPORTANT:
# Candidate #1 previously got mixed up with an Estimator result.
# We therefore resolve it ONLY inside the 11_1A family and explicitly reject
# every path containing "estimator".
#
# Candidate #2/#3/#5 are the exact files already produced in the Week-11 runs.
CANDIDATES = [
    {
        "label": "Candidate_1",
        "description": "final Candidate #1 — SamplerV2 only",
        "exact_path": None,
        "glob_patterns": [
            "11_1A_*qpu*features*predictions*.csv",
            "11_1A_*features*predictions*.csv",
        ],
        "reject_filename_tokens": ["estimator"],
        "ideal_mode": "standard",
    },
    {
        "label": "Candidate_2",
        "description": "final Candidate #2 — SamplerV2",
        "exact_path": RESULTS / "11_1B_second_candidate_qpu_features_predictions.csv",
        "glob_patterns": [],
        "reject_filename_tokens": ["estimator"],
        "ideal_mode": "standard",
    },
    {
        "label": "Candidate_3",
        "description": "final Candidate #3 — SamplerV2",
        "exact_path": RESULTS / "11_1C_third_candidate_qpu_features_predictions.csv",
        "glob_patterns": [],
        "reject_filename_tokens": ["estimator"],
        "ideal_mode": "standard",
    },
    {
        "label": "Candidate_5",
        "description": "Candidate #5 — CONT_H2_R2 executed as RWP64",
        "exact_path": RESULTS / "11_1E7_candidate5_full365_qpu_features_predictions.csv",
        "glob_patterns": [],
        "reject_filename_tokens": ["estimator"],
        "ideal_mode": "rwp64",
    },
]

# Numerical audit tolerance. We expect the saved predictions to come from the
# same frozen linear readout, so reconstruction should be extremely close.
READOUT_RECON_RMSE_TOL = 1e-7


# =============================================================================
# HELPERS
# =============================================================================

def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    aa = a[m]
    bb = b[m]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def bias(y, pred):
    return float(np.mean(np.asarray(pred, float) - np.asarray(y, float)))


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def first_existing(columns, names):
    cols_norm = {norm(c): c for c in columns}
    for name in names:
        k = norm(name)
        if k in cols_norm:
            return cols_norm[k]
    return None


def resolve_candidate_file(spec):
    reject = [t.lower() for t in spec["reject_filename_tokens"]]

    if spec["exact_path"] is not None:
        p = Path(spec["exact_path"])
        if not p.exists():
            raise FileNotFoundError(
                f"{spec['label']}: expected file not found:\n  {p}"
            )
        low = p.name.lower()
        if any(t in low for t in reject):
            raise RuntimeError(
                f"{spec['label']}: rejected non-Sampler file:\n  {p}"
            )
        return p

    matches = []
    for pat in spec["glob_patterns"]:
        matches.extend(RESULTS.glob(pat))

    # deduplicate + reject Estimator etc.
    unique = {}
    for p in matches:
        if not p.is_file():
            continue
        low = p.name.lower()
        if any(t in low for t in reject):
            continue
        unique[str(p.resolve())] = p

    matches = sorted(unique.values())

    if len(matches) == 0:
        raise FileNotFoundError(
            f"{spec['label']}: no Sampler result file found.\n"
            "Searched:\n  "
            + "\n  ".join(str(RESULTS / p) for p in spec["glob_patterns"])
            + "\nEstimator files are deliberately rejected."
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"{spec['label']}: more than one non-Estimator 11_1A result matches.\n"
            "I will NOT guess which one is Candidate #1.\n"
            "Matching files:\n  "
            + "\n  ".join(str(p) for p in matches)
            + "\nSet exact_path for Candidate_1 in CANDIDATES."
        )

    return matches[0]


def detect_target(df):
    col = first_existing(
        df.columns,
        ["target", "y_true", "actual", "claims", "claim_count"],
    )
    if col is None:
        raise RuntimeError(
            "Could not find target column. Columns:\n"
            + ", ".join(map(str, df.columns))
        )
    return col


def detect_qpu_prediction(df):
    col = first_existing(
        df.columns,
        [
            "pred_qpu",
            "qpu_pred",
            "prediction_qpu",
            "y_pred_qpu",
            "pred_QPU",
        ],
    )
    if col is not None:
        return col

    cands = [
        c for c in df.columns
        if "qpu" in norm(c)
        and ("pred" in norm(c) or "prediction" in norm(c))
        and "feature" not in norm(c)
    ]
    if len(cands) != 1:
        raise RuntimeError(
            "Could not uniquely identify QPU prediction column. Candidates: "
            + repr(cands)
        )
    return cands[0]


def detect_ideal_prediction(df, mode):
    """
    Standard candidates:
        use pred_ideal / protocol-matching ideal prediction.

    Candidate #5:
        MUST use RWP64 ideal prediction.
        full-CONT ideal is explicitly forbidden.
    """
    if mode == "standard":
        col = first_existing(
            df.columns,
            [
                "pred_ideal",
                "ideal_pred",
                "prediction_ideal",
                "y_pred_ideal",
                "pred_ideal_W4",
                "pred_ideal_W2",
            ],
        )
        if col is not None:
            return col

        cands = [
            c for c in df.columns
            if "ideal" in norm(c)
            and ("pred" in norm(c) or "prediction" in norm(c))
            and "full_cont" not in norm(c)
            and "rwp64" not in norm(c)
        ]
        if len(cands) != 1:
            raise RuntimeError(
                "Could not uniquely identify matching ideal prediction. "
                f"Candidates={cands}"
            )
        return cands[0]

    if mode == "rwp64":
        cands = [
            c for c in df.columns
            if "rwp64" in norm(c)
            and "ideal" in norm(c)
            and ("pred" in norm(c) or "prediction" in norm(c))
        ]

        # Accept common naming where RWP64 + pred appears but ideal is implicit.
        if not cands:
            cands = [
                c for c in df.columns
                if "rwp64" in norm(c)
                and ("pred" in norm(c) or "prediction" in norm(c))
                and "qpu" not in norm(c)
            ]

        if len(cands) != 1:
            full_cont = [
                c for c in df.columns
                if "full_cont" in norm(c)
                and ("pred" in norm(c) or "prediction" in norm(c))
            ]
            raise RuntimeError(
                "Candidate #5 must be compared to the matching RWP64 ideal "
                "prediction, not full-CONT.\n"
                f"RWP64 prediction candidates: {cands}\n"
                f"Full-CONT columns found (forbidden comparator): {full_cont}\n"
                "All columns:\n  " + "\n  ".join(map(str, df.columns))
            )

        if "full_cont" in norm(cands[0]):
            raise RuntimeError("Internal safety check: full-CONT comparator selected.")

        return cands[0]

    raise ValueError(mode)


def detect_date_column(df):
    return first_existing(
        df.columns,
        ["target_date", "date", "validation_date", "ds"],
    )


# =============================================================================
# FEATURE MATCHING
# =============================================================================

def feature_token(col):
    """
    Extract a stable observable token from a column.

    Handles typical saved names such as:
      X0_ideal
      X0_qpu
      X0_RWP64_ideal
      ideal_X0
      qpu_X0
      YX45_ideal
    """
    s = norm(col).upper()

    # two-body labels first
    m = re.search(r"(XX|XY|XZ|YX|YY|YZ|ZX|ZY|ZZ)(\d)(\d)", s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    # single-qubit Pauli labels
    m = re.search(r"(X|Y|Z)(\d+)", s)
    if m:
        return f"{m.group(1)}{m.group(2)}"

    return None


def find_feature_pairs(df, ideal_mode):
    """
    Returns list of:
        (feature, ideal_column, qpu_column)

    Candidate #5 prefers/requires RWP64 ideal features whenever multiple ideal
    variants exist.
    """
    cols = list(df.columns)

    # QPU observable columns only.
    #
    # IMPORTANT:
    # Saved Week-11 result files can also contain derived diagnostic columns such
    # as:
    #     X0_qpu_minus_RWP64
    #     Z1_qpu_minus_ideal
    #
    # Those are NOT reservoir features and must never be paired as raw QPU
    # observables. Prefer the canonical raw schema "<feature>_qpu" (or
    # "qpu_<feature>") and reject any difference/error/diagnostic column.
    diagnostic_tokens = (
        "minus", "diff", "difference", "delta", "error", "residual",
        "abs_error", "sq_error", "mae", "rmse",
    )

    qpu_cols = []
    for c in cols:
        n = norm(c)

        if "pred" in n or "prediction" in n:
            continue

        if any(tok in n for tok in diagnostic_tokens):
            continue

        if "qpu" not in n:
            continue

        # Strong preference for a raw-QPU naming pattern.
        is_raw_qpu = (
            n.endswith("_qpu")
            or n.startswith("qpu_")
            or "_qpu_" not in n
        )
        if not is_raw_qpu:
            continue

        tok = feature_token(c)
        if tok is not None:
            qpu_cols.append((tok, c))

    # If multiple columns for the same observable survive, prefer the exact
    # canonical "<feature>_qpu" form.
    grouped_qpu = {}
    for tok, c in qpu_cols:
        grouped_qpu.setdefault(tok, []).append(c)

    qpu_cols = []
    for tok, candidates in grouped_qpu.items():
        exact = [
            c for c in candidates
            if norm(c) == f"{tok.lower()}_qpu"
        ]
        if len(exact) == 1:
            qpu_cols.append((tok, exact[0]))
        elif len(candidates) == 1:
            qpu_cols.append((tok, candidates[0]))
        else:
            raise RuntimeError(
                f"Feature {tok}: multiple raw QPU feature columns remain "
                f"after diagnostic-column filtering: {candidates}"
            )

    qpu_by_feature = {}
    for tok, c in qpu_cols:
        qpu_by_feature.setdefault(tok, []).append(c)

    pairs = []

    for tok, qcols in sorted(qpu_by_feature.items()):
        if len(qcols) != 1:
            raise RuntimeError(
                f"Feature {tok}: multiple QPU columns found: {qcols}"
            )
        qcol = qcols[0]

        ideal_candidates = []
        for c in cols:
            n = norm(c)

            if "pred" in n or "prediction" in n:
                continue
            if "qpu" in n:
                continue
            if any(t in n for t in diagnostic_tokens):
                continue
            if feature_token(c) != tok:
                continue

            if ideal_mode == "rwp64":
                # Candidate #5 may save the matching ideal RWP64 feature as
                # either X0_RWP64_ideal or simply X0_RWP64.
                if "rwp64" in n or "ideal" in n:
                    ideal_candidates.append(c)
            else:
                if "ideal" in n:
                    ideal_candidates.append(c)

        if ideal_mode == "rwp64":
            # First preference: explicitly RWP64-tagged feature.
            rwp64 = [c for c in ideal_candidates if "rwp64" in norm(c)]
            if rwp64:
                ideal_candidates = rwp64
            else:
                # Fallback to a plain ideal feature only if no full-CONT-tagged
                # alternative is being selected.
                ideal_candidates = [
                    c for c in ideal_candidates
                    if "full_cont" not in norm(c)
                ]

        else:
            # Avoid accidentally pairing a protocol-specific foreign comparator.
            ideal_candidates = [
                c for c in ideal_candidates
                if "full_cont" not in norm(c)
                and "rwp64" not in norm(c)
            ] or ideal_candidates

        if len(ideal_candidates) != 1:
            raise RuntimeError(
                f"Feature {tok}: could not uniquely pair QPU and matching ideal.\n"
                f"  QPU={qcol}\n"
                f"  ideal candidates={ideal_candidates}"
            )

        pairs.append((tok, ideal_candidates[0], qcol))

    if not pairs:
        raise RuntimeError(
            "No QPU/ideal observable feature pairs found.\nColumns:\n  "
            + "\n  ".join(map(str, cols))
        )

    return pairs


# =============================================================================
# FROZEN READOUT RECONSTRUCTION
# =============================================================================

def reconstruct_effective_linear_readout(X_ideal, pred_ideal):
    """
    Recover the effective affine mapping in RAW feature coordinates:

        pred = intercept + X @ beta

    This uses only SAVED IDEAL FEATURES + SAVED IDEAL PREDICTIONS.
    Targets y are not used.

    Since the frozen Ridge readout is linear after standardization, an exactly
    equivalent affine map exists in raw feature coordinates.
    """
    X_ideal = np.asarray(X_ideal, float)
    pred_ideal = np.asarray(pred_ideal, float)

    A = np.column_stack([np.ones(len(X_ideal)), X_ideal])
    coef, *_ = np.linalg.lstsq(A, pred_ideal, rcond=None)

    intercept = float(coef[0])
    beta = np.asarray(coef[1:], float)

    pred_reconstructed = intercept + X_ideal @ beta
    recon_rmse = rmse(pred_ideal, pred_reconstructed)

    return intercept, beta, pred_reconstructed, recon_rmse


# =============================================================================
# LOAD + VALIDATE
# =============================================================================

summary_rows = []
feature_rows = []
contrib_rows = []
daily_rows = []
json_candidates = []

print("=" * 124)
print("WEEK 11.1E.8 — CORRECTED FOUR-CANDIDATE 2025 POST-HARDWARE DIAGNOSIS")
print("=" * 124)
print("SamplerV2 results ONLY | no QPU job | no mitigation fitting | 2026 NOT LOADED")
print()

for spec in CANDIDATES:
    path = resolve_candidate_file(spec)

    if "estimator" in path.name.lower():
        raise RuntimeError(
            f"SAFETY FAILURE: Estimator result selected: {path}"
        )

    df = pd.read_csv(path)

    if len(df) != EXPECTED_N:
        raise RuntimeError(
            f"{spec['label']}: expected {EXPECTED_N} rows, found {len(df)} "
            f"in {path}"
        )

    target_col = detect_target(df)
    qpu_pred_col = detect_qpu_prediction(df)
    ideal_pred_col = detect_ideal_prediction(df, spec["ideal_mode"])
    date_col = detect_date_column(df)

    if spec["ideal_mode"] == "rwp64" and "full_cont" in norm(ideal_pred_col):
        raise RuntimeError(
            "Candidate #5 safety failure: full-CONT ideal prediction selected."
        )

    # 2025-only safety check if date is stored.
    dates = None
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        valid_dates = dates.dropna()
        if len(valid_dates):
            years = sorted(valid_dates.dt.year.unique().tolist())
            if years != [2025]:
                raise RuntimeError(
                    f"{spec['label']}: file contains years {years}; "
                    "this step must be 2025 only."
                )

    y = numeric(df, target_col)
    pred_qpu = numeric(df, qpu_pred_col)
    pred_ideal = numeric(df, ideal_pred_col)

    core_mask = np.isfinite(y) & np.isfinite(pred_qpu) & np.isfinite(pred_ideal)
    if core_mask.sum() != EXPECTED_N:
        raise RuntimeError(
            f"{spec['label']}: core columns contain missing/non-numeric values."
        )

    pairs = find_feature_pairs(df, spec["ideal_mode"])

    feature_names = [p[0] for p in pairs]
    ideal_cols = [p[1] for p in pairs]
    qpu_cols = [p[2] for p in pairs]

    X_ideal = np.column_stack([numeric(df, c) for c in ideal_cols])
    X_qpu = np.column_stack([numeric(df, c) for c in qpu_cols])

    if not np.all(np.isfinite(X_ideal)) or not np.all(np.isfinite(X_qpu)):
        raise RuntimeError(
            f"{spec['label']}: feature columns contain NaN/inf."
        )

    # -------------------------------------------------------------------------
    # Frozen readout reconstruction audit
    # -------------------------------------------------------------------------
    intercept, beta, pred_ideal_recon, ideal_recon_rmse = (
        reconstruct_effective_linear_readout(X_ideal, pred_ideal)
    )

    pred_qpu_recon = intercept + X_qpu @ beta
    qpu_recon_rmse = rmse(pred_qpu, pred_qpu_recon)

    if ideal_recon_rmse > READOUT_RECON_RMSE_TOL:
        raise RuntimeError(
            f"{spec['label']}: could not reconstruct frozen ideal readout from "
            f"saved matching features. RMSE={ideal_recon_rmse:.3e} > "
            f"{READOUT_RECON_RMSE_TOL:.1e}.\n"
            "This usually means the wrong ideal feature family was paired."
        )

    if qpu_recon_rmse > READOUT_RECON_RMSE_TOL:
        raise RuntimeError(
            f"{spec['label']}: reconstructed frozen readout does not reproduce "
            f"saved QPU predictions. RMSE={qpu_recon_rmse:.3e} > "
            f"{READOUT_RECON_RMSE_TOL:.1e}.\n"
            "Do NOT interpret feature contributions until the feature family "
            "or comparator is corrected."
        )

    feature_error = X_qpu - X_ideal

    # Exact prediction-shift decomposition.
    contributions = feature_error * beta.reshape(1, -1)
    decomposed_shift = contributions.sum(axis=1)
    saved_shift = pred_qpu - pred_ideal

    shift_decomp_rmse = rmse(saved_shift, decomposed_shift)

    if shift_decomp_rmse > READOUT_RECON_RMSE_TOL:
        raise RuntimeError(
            f"{spec['label']}: prediction shift decomposition audit failed: "
            f"{shift_decomp_rmse:.3e}"
        )

    # -------------------------------------------------------------------------
    # Candidate summary
    # -------------------------------------------------------------------------
    ideal_rmse = rmse(y, pred_ideal)
    qpu_rmse = rmse(y, pred_qpu)

    srow = {
        "candidate": spec["label"],
        "description": spec["description"],
        "source_file": str(path),
        "target_column": target_col,
        "ideal_prediction_column": ideal_pred_col,
        "qpu_prediction_column": qpu_pred_col,
        "n": len(df),
        "n_features": len(feature_names),
        "ideal_rmse": ideal_rmse,
        "qpu_rmse": qpu_rmse,
        "delta_rmse_qpu_minus_ideal": qpu_rmse - ideal_rmse,
        "ideal_mae": mae(y, pred_ideal),
        "qpu_mae": mae(y, pred_qpu),
        "ideal_bias_vs_target": bias(y, pred_ideal),
        "qpu_bias_vs_target": bias(y, pred_qpu),
        "prediction_shift_mean_qpu_minus_ideal": float(np.mean(saved_shift)),
        "prediction_shift_mae": float(np.mean(np.abs(saved_shift))),
        "prediction_shift_rmse": float(np.sqrt(np.mean(saved_shift ** 2))),
        "prediction_shift_std": float(np.std(saved_shift, ddof=1)),
        "prediction_corr_qpu_vs_ideal": corr(pred_qpu, pred_ideal),
        "global_feature_mae_qpu_vs_ideal": float(np.mean(np.abs(feature_error))),
        "global_feature_rmse_qpu_vs_ideal": float(
            np.sqrt(np.mean(feature_error ** 2))
        ),
        "readout_reconstruction_rmse_ideal": ideal_recon_rmse,
        "readout_reconstruction_rmse_qpu": qpu_recon_rmse,
        "prediction_shift_decomposition_rmse": shift_decomp_rmse,
    }
    summary_rows.append(srow)

    # -------------------------------------------------------------------------
    # Per-feature distortion + contribution to prediction shift
    # -------------------------------------------------------------------------
    for j, feat in enumerate(feature_names):
        d = feature_error[:, j]
        c = contributions[:, j]
        ideal_j = X_ideal[:, j]
        qpu_j = X_qpu[:, j]

        feature_rows.append({
            "candidate": spec["label"],
            "feature": feat,
            "ideal_column": ideal_cols[j],
            "qpu_column": qpu_cols[j],
            "effective_raw_readout_beta": float(beta[j]),
            "ideal_feature_mean": float(np.mean(ideal_j)),
            "qpu_feature_mean": float(np.mean(qpu_j)),
            "feature_bias_qpu_minus_ideal": float(np.mean(d)),
            "feature_mae": float(np.mean(np.abs(d))),
            "feature_rmse": float(np.sqrt(np.mean(d ** 2))),
            "feature_corr_qpu_vs_ideal": corr(qpu_j, ideal_j),
            "ideal_feature_std": float(np.std(ideal_j, ddof=1)),
            "qpu_feature_std": float(np.std(qpu_j, ddof=1)),
            "qpu_to_ideal_std_ratio": (
                float(np.std(qpu_j, ddof=1) / np.std(ideal_j, ddof=1))
                if np.std(ideal_j, ddof=1) > 0 else np.nan
            ),
        })

        contrib_rows.append({
            "candidate": spec["label"],
            "feature": feat,
            "effective_raw_readout_beta": float(beta[j]),
            "mean_prediction_shift_contribution": float(np.mean(c)),
            "mean_abs_prediction_shift_contribution": float(np.mean(np.abs(c))),
            "rms_prediction_shift_contribution": float(
                np.sqrt(np.mean(c ** 2))
            ),
            "contribution_corr_with_total_prediction_shift": corr(c, saved_shift),
            "fraction_of_mean_shift": (
                float(np.mean(c) / np.mean(saved_shift))
                if abs(np.mean(saved_shift)) > 1e-12 else np.nan
            ),
        })

    # -------------------------------------------------------------------------
    # Per-day saved/decomposed shifts
    # -------------------------------------------------------------------------
    for i in range(len(df)):
        row = {
            "candidate": spec["label"],
            "row": i,
            "date": (
                str(dates.iloc[i].date())
                if dates is not None and pd.notna(dates.iloc[i])
                else ""
            ),
            "target": float(y[i]),
            "pred_ideal": float(pred_ideal[i]),
            "pred_qpu": float(pred_qpu[i]),
            "prediction_shift_qpu_minus_ideal": float(saved_shift[i]),
            "prediction_shift_from_feature_decomposition": float(
                decomposed_shift[i]
            ),
        }
        daily_rows.append(row)

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------
    print("-" * 124)
    print(f"{spec['label']}: {spec['description']}")
    print(f"  file:               {path}")
    print(f"  comparator:         {ideal_pred_col}")
    print(f"  QPU prediction:     {qpu_pred_col}")
    print(f"  feature pairs:      {len(feature_names)} -> {feature_names}")
    print(f"  ideal RMSE:         {ideal_rmse:.6f}")
    print(f"  QPU RMSE:           {qpu_rmse:.6f}")
    print(f"  delta RMSE:         {qpu_rmse - ideal_rmse:+.6f}")
    print(f"  ideal bias:         {bias(y, pred_ideal):+.6f}")
    print(f"  QPU bias:           {bias(y, pred_qpu):+.6f}")
    print(f"  QPU-ideal shift:    {np.mean(saved_shift):+.6f} mean")
    print(f"  prediction corr:    {corr(pred_qpu, pred_ideal):.6f}")
    print(
        f"  feature MAE/RMSE:   "
        f"{np.mean(np.abs(feature_error)):.6f} / "
        f"{np.sqrt(np.mean(feature_error ** 2)):.6f}"
    )
    print(
        f"  readout audit:      ideal={ideal_recon_rmse:.3e}, "
        f"QPU={qpu_recon_rmse:.3e}, shift={shift_decomp_rmse:.3e}"
    )

    # Show largest mean shift contributors now; this is the central diagnostic.
    order = np.argsort(-np.abs(np.mean(contributions, axis=0)))
    print("  largest mean prediction-shift contributions:")
    for j in order[: min(5, len(order))]:
        print(
            f"    {feature_names[j]:>6}: "
            f"feature bias={np.mean(feature_error[:, j]):+.6f}, "
            f"beta={beta[j]:+.6f}, "
            f"mean contribution={np.mean(contributions[:, j]):+.6f}"
        )

    json_candidates.append({
        "candidate": spec["label"],
        "source_file": str(path),
        "ideal_prediction_column": ideal_pred_col,
        "qpu_prediction_column": qpu_pred_col,
        "feature_names": feature_names,
        "effective_intercept": intercept,
        "effective_beta": {
            feature_names[j]: float(beta[j])
            for j in range(len(feature_names))
        },
        "summary": srow,
    })


# =============================================================================
# SAVE
# =============================================================================

summary_df = pd.DataFrame(summary_rows)
feature_df = pd.DataFrame(feature_rows)
contrib_df = pd.DataFrame(contrib_rows)
daily_df = pd.DataFrame(daily_rows)

summary_df.to_csv(f"{OUT}_candidate_summary.csv", index=False)
feature_df.to_csv(f"{OUT}_feature_diagnostics.csv", index=False)
contrib_df.to_csv(f"{OUT}_feature_contributions.csv", index=False)
daily_df.to_csv(f"{OUT}_daily_prediction_shift.csv", index=False)

# Rank feature contributions inside each candidate for convenience.
ranked = contrib_df.copy()
ranked["abs_mean_contribution"] = np.abs(
    ranked["mean_prediction_shift_contribution"]
)
ranked["rank_by_abs_mean_shift"] = (
    ranked.groupby("candidate")["abs_mean_contribution"]
    .rank(method="first", ascending=False)
)
ranked = ranked.sort_values(
    ["candidate", "rank_by_abs_mean_shift"]
)
ranked.to_csv(f"{OUT}_feature_contributions_ranked.csv", index=False)

# =============================================================================
# REPORT
# =============================================================================

lines = []
lines.append("=" * 100)
lines.append("WEEK 11.1E.8 — CORRECTED POST-HARDWARE DIAGNOSIS")
lines.append("=" * 100)
lines.append("")
lines.append("SamplerV2 only. No Estimator. No QPU submission. No 2026.")
lines.append(
    "Candidate #5 is explicitly compared to its matching RWP64 ideal "
    "reference, not full-CONT."
)
lines.append("")
lines.append("CANDIDATE-LEVEL RESULTS")
lines.append("-" * 100)

for _, r in summary_df.iterrows():
    lines.append(
        f"{r['candidate']}: "
        f"ideal RMSE={r['ideal_rmse']:.6f}, "
        f"QPU RMSE={r['qpu_rmse']:.6f}, "
        f"delta={r['delta_rmse_qpu_minus_ideal']:+.6f}, "
        f"mean QPU-ideal prediction shift="
        f"{r['prediction_shift_mean_qpu_minus_ideal']:+.6f}, "
        f"corr={r['prediction_corr_qpu_vs_ideal']:.6f}"
    )

lines.append("")
lines.append("TOP FEATURE CAUSES OF MEAN PREDICTION SHIFT")
lines.append("-" * 100)

for candidate, g in ranked.groupby("candidate"):
    lines.append(candidate)
    for _, r in g.head(5).iterrows():
        lines.append(
            f"  {r['feature']}: beta={r['effective_raw_readout_beta']:+.6f}, "
            f"mean contribution="
            f"{r['mean_prediction_shift_contribution']:+.6f}"
        )

lines.append("")
lines.append("INTERPRETATION RULE FOR THE NEXT STEP")
lines.append("-" * 100)
lines.append(
    "1. Large stable feature bias + preserved feature correlation -> "
    "systematic hardware/readout distortion is plausible."
)
lines.append(
    "2. Near-zero feature bias but large feature RMSE -> random/shot-like "
    "distortion dominates."
)
lines.append(
    "3. Low QPU-vs-ideal feature correlation -> information loss, not just "
    "a simple offset."
)
lines.append(
    "4. Small feature error multiplied by very large |beta| -> the frozen "
    "classical readout is amplifying hardware noise."
)
lines.append(
    "We decide on a mitigation experiment only AFTER interpreting these four "
    "candidate feature-level diagnostics."
)

report = "\n".join(lines)
Path(f"{OUT}_report.txt").write_text(report, encoding="utf-8")

payload = {
    "step": "11.1E.8 corrected",
    "scope": "2025 SamplerV2 post-hardware diagnosis",
    "2026_loaded": False,
    "mitigation_fitted": False,
    "estimator_allowed": False,
    "candidate5_ideal_comparator": "RWP64 only",
    "candidates": json_candidates,
}
Path(f"{OUT}_summary.json").write_text(
    json.dumps(payload, indent=2, default=str),
    encoding="utf-8",
)


# =============================================================================
# FINAL CONSOLE TABLE
# =============================================================================

print()
print("=" * 124)
print("CORRECTED CANDIDATE SUMMARY")
print("=" * 124)

print(
    summary_df[
        [
            "candidate",
            "ideal_rmse",
            "qpu_rmse",
            "delta_rmse_qpu_minus_ideal",
            "ideal_bias_vs_target",
            "qpu_bias_vs_target",
            "prediction_shift_mean_qpu_minus_ideal",
            "prediction_corr_qpu_vs_ideal",
            "global_feature_rmse_qpu_vs_ideal",
        ]
    ].to_string(index=False)
)

print()
print("Saved:")
for suffix in [
    "_candidate_summary.csv",
    "_feature_diagnostics.csv",
    "_feature_contributions.csv",
    "_feature_contributions_ranked.csv",
    "_daily_prediction_shift.csv",
    "_report.txt",
    "_summary.json",
]:
    print(f"  {OUT}{suffix}")

print()
print("2026 remains FROZEN / UNUSED.")
