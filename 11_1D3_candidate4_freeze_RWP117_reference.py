from __future__ import annotations

"""
WEEK 11.1D.3 — CANDIDATE #4 FREEZE THE WASHOUT-DERIVED RWP REFERENCE
=====================================================================

GOAL
----
Assemble everything already established for Candidate #4 and create the exact
IDEAL RWP(K) reference that the future QPU run must reproduce.

Original selected model:
    candidate = CONT_H0_R4
    protocol  = full CONT
    topology  = H0
    r         = 4
    alpha     = 0.75
    readout   = XZinj_plus_YX45

Already-computed evidence is EXTRACTED, not re-selected:
    1. Week-10 / CONT frozen readout:
         lambda
         training-only CV RMSE
         ideal full-CONT 2025 metrics

    2. Global mathematical washout:
         trace-distance T_w(0.01)
         readout-feature T_w(0.01)

    3. Local 365-endpoint washout validation:
         all-endpoint trace criterion
         all-endpoint feature criterion

    4. Week-10 one-step resource point, if available.

Operational RWP length:
    K_RWP = max(
        global trace T_w(0.01),
        global readout-feature T_w(0.01)
    )

and that K MUST also pass the local all-365-endpoint criteria.

For the current Candidate #4 results this should freeze:
    K_RWP = 117

IMPORTANT MODEL SEMANTICS
-------------------------
We DO NOT train a new RWP Ridge model.

The selected model remains:
    full-CONT reservoir + full-CONT 2022-2024 trained Ridge readout.

We reconstruct that already-frozen readout with the already-selected lambda,
then apply the SAME scaler/Ridge to three conceptual references:

    A) FULL CONT IDEAL
       The original selected model.

    B) FULL RESTART / RSP IDEAL
       Replay from the beginning of chronology for an endpoint.
       In ideal, unmeasured dynamics this is mathematically the same state as
       A, because it applies exactly the same channel composition from the same
       initial memory state.  Therefore RSP ideal == full CONT ideal.

    C) WASHOUT-DERIVED RWP IDEAL
       Reset memory to |00>, replay exactly K_RWP recent chronological inputs,
       measure only at the final endpoint.

The future real-QPU comparison must therefore be:

    full CONT ideal
          |
          | protocol approximation
          v
    RWP(K=117) ideal
          |
          | hardware-transfer error
          v
    RWP(K=117) QPU

This script creates the exact B/C references and freezes a machine-readable
package for the later QPU script.

NO IBM CONNECTION.
NO QPU JOB.
2026 IS NEVER LOADED.

Expected prior artifacts in results/
------------------------------------
10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv
10_05_direct_runs.csv                                  (optional resources)
11_1D_candidate4_CONT_context_audit.json
11_1D_candidate4_CONT_context_audit.csv                (optional comparison)
11_1D1_candidate4_washout_thresholds.csv
11_1D1_candidate4_washout_summary.json
11_1D2_candidate4_local_washout_summary.csv
11_1D2_candidate4_local_washout_summary.json           (optional)

Required project modules
------------------------
11_1D_candidate4_cont_context_audit.py
09_04_rwp_common.py        (loaded by the audit module)
db_objects.py
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import db_objects as db


# =============================================================================
# Paths / constants
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

AUDIT_SCRIPT = HERE / "11_1D_candidate4_cont_context_audit.py"

CONTEXT_JSON = RESULTS / "11_1D_candidate4_CONT_context_audit.json"
CONTEXT_CSV = RESULTS / "11_1D_candidate4_CONT_context_audit.csv"

GLOBAL_THRESHOLDS = RESULTS / "11_1D1_candidate4_washout_thresholds.csv"
GLOBAL_WASHOUT_JSON = RESULTS / "11_1D1_candidate4_washout_summary.json"

LOCAL_WASHOUT_CSV = RESULTS / "11_1D2_candidate4_local_washout_summary.csv"
LOCAL_WASHOUT_JSON = RESULTS / "11_1D2_candidate4_local_washout_summary.json"

DIRECT_RUNS = RESULTS / "10_05_direct_runs.csv"

CANDIDATE_KEY = "CONT_H0_R4"
PRIMARY_EPSILON = 0.01

OUT_PREDICTIONS = RESULTS / "11_1D3_candidate4_RWP117_predictions.csv"
OUT_FEATURES = RESULTS / "11_1D3_candidate4_RWP117_feature_comparison.csv"
OUT_FEATURE_SUMMARY = RESULTS / "11_1D3_candidate4_RWP117_feature_summary.csv"
OUT_PACKAGE = RESULTS / "11_1D3_candidate4_RWP117_frozen_package.json"


# =============================================================================
# Generic helpers
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required project module not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import module from {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required prior artifact not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required prior artifact not found: {path}"
        )

    return pd.read_csv(
        path
    )


def rmse(y_true, y_pred):
    a = np.asarray(
        y_true,
        dtype=float,
    )

    b = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (a - b) ** 2
            )
        )
    )


def mae(y_true, y_pred):
    a = np.asarray(
        y_true,
        dtype=float,
    )

    b = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(
                a - b
            )
        )
    )


def bias(y_true, y_pred):
    a = np.asarray(
        y_true,
        dtype=float,
    )

    b = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.mean(
            b - a
        )
    )


def safe_corr(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if (
        np.std(a) == 0
        or np.std(b) == 0
    ):
        return None

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def numeric_close(
    actual,
    expected,
    atol=1e-6,
    label="value",
):
    if not np.isclose(
        float(actual),
        float(expected),
        rtol=0.0,
        atol=float(atol),
    ):
        raise RuntimeError(
            f"Audit failed for {label}: "
            f"reconstructed={actual}, "
            f"stored={expected}, "
            f"atol={atol}"
        )


# =============================================================================
# Load exact Candidate #4 implementation
# =============================================================================

audit = load_module(
    AUDIT_SCRIPT,
    "qrc_11_1d3_candidate4_audit",
)

common = audit.common

FEATURES = list(
    audit.FEATURES
)


# =============================================================================
# Extract the already-computed washout decision
# =============================================================================

def extract_frozen_washout():
    thresholds = load_csv(
        GLOBAL_THRESHOLDS
    )

    local = load_csv(
        LOCAL_WASHOUT_CSV
    )

    context = load_json(
        CONTEXT_JSON
    )

    global_summary = load_json(
        GLOBAL_WASHOUT_JSON
    )

    # ------------------------------------------------------------------
    # Global epsilon = 0.01.
    # ------------------------------------------------------------------
    eps_rows = thresholds[
        np.isclose(
            thresholds["epsilon"].astype(float),
            PRIMARY_EPSILON,
        )
    ]

    if len(eps_rows) != 1:
        raise RuntimeError(
            "Expected exactly one global washout row "
            f"for epsilon={PRIMARY_EPSILON}."
        )

    eps_row = eps_rows.iloc[0]

    trace_tw = int(
        eps_row[
            "trace_distance_stable_washout_steps"
        ]
    )

    feature_tw = int(
        eps_row[
            "feature_distance_stable_washout_steps"
        ]
    )

    K = int(
        max(
            trace_tw,
            feature_tw,
        )
    )

    # ------------------------------------------------------------------
    # This exact K must pass local all-365-endpoint validation.
    # ------------------------------------------------------------------
    local_rows = local[
        local[
            "context_steps"
        ].astype(int)
        == K
    ]

    if len(local_rows) != 1:
        raise RuntimeError(
            f"Local washout audit does not contain K={K}. "
            "Do not silently choose another K."
        )

    local_row = local_rows.iloc[0]

    if not bool(
        local_row[
            "all_trace_le_0.01"
        ]
    ):
        raise RuntimeError(
            f"K={K} does not pass all-endpoint "
            "trace-distance <= 0.01."
        )

    if not bool(
        local_row[
            "all_feature_le_0.01"
        ]
    ):
        raise RuntimeError(
            f"K={K} does not pass all-endpoint "
            "feature-distance <= 0.01."
        )

    extracted = {
        "epsilon": (
            PRIMARY_EPSILON
        ),
        "global_trace_washout_steps": (
            trace_tw
        ),
        "global_feature_washout_steps": (
            feature_tw
        ),
        "operational_K": (
            K
        ),
        "local_trace_median_at_K": float(
            local_row[
                "trace_median"
            ]
        ),
        "local_trace_p95_at_K": float(
            local_row[
                "trace_p95"
            ]
        ),
        "local_trace_max_at_K": float(
            local_row[
                "trace_max"
            ]
        ),
        "local_feature_median_at_K": float(
            local_row[
                "feature_median"
            ]
        ),
        "local_feature_p95_at_K": float(
            local_row[
                "feature_p95"
            ]
        ),
        "local_feature_max_at_K": float(
            local_row[
                "feature_max"
            ]
        ),
        "local_all_trace_pass": bool(
            local_row[
                "all_trace_le_0.01"
            ]
        ),
        "local_all_feature_pass": bool(
            local_row[
                "all_feature_le_0.01"
            ]
        ),
        "global_contraction_fit": (
            global_summary.get(
                "contraction_fit"
            )
        ),
        "stored_full_cont": {
            "ridge_lambda": float(
                context[
                    "selected_lambda"
                ]
            ),
            "training_cv_rmse": float(
                context[
                    "training_cv_rmse"
                ]
            ),
            "rmse": float(
                context[
                    "full_cont_ideal_rmse"
                ]
            ),
            "mae": float(
                context[
                    "full_cont_ideal_mae"
                ]
            ),
            "bias": float(
                context[
                    "full_cont_ideal_bias"
                ]
            ),
        },
    }

    return extracted


# =============================================================================
# Extract Week-10 one-step resource point
# =============================================================================

def extract_week10_resource_point():
    if not DIRECT_RUNS.exists():
        return None

    df = pd.read_csv(
        DIRECT_RUNS
    )

    sub = df[
        (
            df[
                "candidate_key"
            ].astype(str)
            == CANDIDATE_KEY
        )
        &
        (
            df[
                "r"
            ].astype(int)
            == int(
                audit.OPERATING_R
            )
        )
    ].copy()

    if len(sub) != 1:
        return None

    row = sub.iloc[0]

    return {
        "source": str(
            DIRECT_RUNS
        ),
        "r": int(
            row["r"]
        ),
        "feature_vector_n_cz_one_step": int(
            row[
                "feature_vector_n_cz"
            ]
        ),
        "max_setting_depth_one_step": float(
            row[
                "max_setting_depth"
            ]
        ),
        "max_setting_duration_us_one_step": float(
            row[
                "max_setting_duration_us"
            ]
        ),
        "n_settings": int(
            row[
                "n_settings"
            ]
        ),
        "week10_ideal_validation_rmse": float(
            row[
                "ideal_validation_rmse"
            ]
        ),
        "week10_sampled_validation_rmse": float(
            row[
                "rmse"
            ]
        ),
        "week10_fresh_layout": (
            None
            if pd.isna(
                row[
                    "fresh_layout"
                ]
            )
            else str(
                row[
                    "fresh_layout"
                ]
            )
        ),
    }


# =============================================================================
# Reconstruct ONLY what is necessary to obtain the frozen Ridge object
# and exact 2025 full-CONT + RWP(K) predictions.
# =============================================================================

def build_reference_package(
    candidate,
    extracted,
):
    (
        work_tv,
        cols,
        y_all,
    ) = common.load_train_validation()

    # 2026 is not loaded by common.load_train_validation().
    angles = common.make_angles(
        work_tv,
        cols,
        float(
            candidate[
                "alpha"
            ]
        ),
    )

    # Reconstruct the already-defined full CONT trajectory.
    # This does NOT select a new candidate or new hyperparameters.
    (
        full_bank,
        A_list,
    ) = audit.build_full_cont_bank(
        candidate,
        angles,
    )

    endpoints = np.arange(
        len(
            full_bank
        ),
        dtype=int,
    )

    y_endpoint = y_all[
        endpoints
    ]

    train_mask = (
        endpoints
        < common.N_TRAIN
    )

    val_mask = (
        endpoints
        >= common.N_TRAIN
    )

    X_train = full_bank[
        train_mask
    ]

    y_train = y_endpoint[
        train_mask
    ]

    X_val_full = full_bank[
        val_mask
    ]

    y_val = y_endpoint[
        val_mask
    ]

    val_endpoints = endpoints[
        val_mask
    ]

    if len(
        val_endpoints
    ) != common.N_VAL:
        raise RuntimeError(
            f"Expected {common.N_VAL} 2025 endpoints, "
            f"found {len(val_endpoints)}."
        )

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Do NOT re-run hyperparameter selection.
    # Extract the already-selected full-CONT lambda.
    # ------------------------------------------------------------------
    selected_lambda = float(
        extracted[
            "stored_full_cont"
        ][
            "ridge_lambda"
        ]
    )

    (
        model,
        scaler,
        keep,
    ) = common.fit_scaled_ridge(
        X_train,
        y_train,
        selected_lambda,
    )

    pred_full = common.predict_scaled_ridge(
        model,
        scaler,
        keep,
        X_val_full,
    )

    full_metrics = {
        "rmse": rmse(
            y_val,
            pred_full,
        ),
        "mae": mae(
            y_val,
            pred_full,
        ),
        "bias": bias(
            y_val,
            pred_full,
        ),
    }

    stored = extracted[
        "stored_full_cont"
    ]

    numeric_close(
        full_metrics["rmse"],
        stored["rmse"],
        atol=2e-6,
        label="full-CONT 2025 RMSE",
    )

    numeric_close(
        full_metrics["mae"],
        stored["mae"],
        atol=2e-6,
        label="full-CONT 2025 MAE",
    )

    numeric_close(
        full_metrics["bias"],
        stored["bias"],
        atol=2e-6,
        label="full-CONT 2025 bias",
    )

    # ------------------------------------------------------------------
    # Exact ideal washout-derived RWP reference.
    # ------------------------------------------------------------------
    K = int(
        extracted[
            "operational_K"
        ]
    )

    X_val_rwp = np.vstack(
        [
            audit.replay_feature(
                A_list,
                endpoint=int(
                    endpoint
                ),
                context_steps=K,
            )
            for endpoint
            in val_endpoints
        ]
    )

    pred_rwp = common.predict_scaled_ridge(
        model,
        scaler,
        keep,
        X_val_rwp,
    )

    rwp_metrics = {
        "rmse": rmse(
            y_val,
            pred_rwp,
        ),
        "mae": mae(
            y_val,
            pred_rwp,
        ),
        "bias": bias(
            y_val,
            pred_rwp,
        ),
    }

    # ------------------------------------------------------------------
    # Protocol approximation:
    # same Ridge, same target dates, only CONT vs RWP state construction.
    # ------------------------------------------------------------------
    feature_delta = (
        X_val_rwp
        - X_val_full
    )

    prediction_delta = (
        pred_rwp
        - pred_full
    )

    protocol_metrics = {
        "delta_target_rmse_rwp_minus_cont": (
            rwp_metrics[
                "rmse"
            ]
            - full_metrics[
                "rmse"
            ]
        ),
        "delta_target_mae_rwp_minus_cont": (
            rwp_metrics[
                "mae"
            ]
            - full_metrics[
                "mae"
            ]
        ),
        "delta_target_bias_rwp_minus_cont": (
            rwp_metrics[
                "bias"
            ]
            - full_metrics[
                "bias"
            ]
        ),
        "prediction_rmse_rwp_vs_cont": rmse(
            pred_full,
            pred_rwp,
        ),
        "prediction_mae_rwp_vs_cont": mae(
            pred_full,
            pred_rwp,
        ),
        "prediction_max_abs_rwp_vs_cont": float(
            np.max(
                np.abs(
                    prediction_delta
                )
            )
        ),
        "prediction_corr_rwp_vs_cont": (
            safe_corr(
                pred_full,
                pred_rwp,
            )
        ),
        "feature_rmse_rwp_vs_cont": float(
            np.sqrt(
                np.mean(
                    feature_delta ** 2
                )
            )
        ),
        "feature_mae_rwp_vs_cont": float(
            np.mean(
                np.abs(
                    feature_delta
                )
            )
        ),
        "feature_max_abs_rwp_vs_cont": float(
            np.max(
                np.abs(
                    feature_delta
                )
            )
        ),
    }

    # ------------------------------------------------------------------
    # Daily prediction artifact.
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame(
        {
            "endpoint": (
                val_endpoints.astype(
                    int
                )
            ),
            "validation_row": np.arange(
                len(
                    val_endpoints
                ),
                dtype=int,
            ),
            "target": y_val,
            "pred_full_CONT_ideal": (
                pred_full
            ),
            f"pred_RWP_K{K}_ideal": (
                pred_rwp
            ),
            "pred_RWP_minus_CONT": (
                prediction_delta
            ),
            "sq_error_full_CONT": (
                (
                    y_val
                    - pred_full
                )
                ** 2
            ),
            f"sq_error_RWP_K{K}": (
                (
                    y_val
                    - pred_rwp
                )
                ** 2
            ),
            "abs_error_full_CONT": (
                np.abs(
                    y_val
                    - pred_full
                )
            ),
            f"abs_error_RWP_K{K}": (
                np.abs(
                    y_val
                    - pred_rwp
                )
            ),
        }
    )

    # ------------------------------------------------------------------
    # Per-endpoint feature artifact.
    # ------------------------------------------------------------------
    feature_rows = []

    for i, endpoint in enumerate(
        val_endpoints
    ):
        row = {
            "endpoint": int(
                endpoint
            ),
            "validation_row": int(
                i
            ),
        }

        for j, name in enumerate(
            FEATURES
        ):
            full_value = float(
                X_val_full[
                    i,
                    j,
                ]
            )

            rwp_value = float(
                X_val_rwp[
                    i,
                    j,
                ]
            )

            row[
                f"{name}_CONT"
            ] = full_value

            row[
                f"{name}_RWP_K{K}"
            ] = rwp_value

            row[
                f"{name}_delta"
            ] = (
                rwp_value
                - full_value
            )

        feature_rows.append(
            row
        )

    feature_df = pd.DataFrame(
        feature_rows
    )

    # ------------------------------------------------------------------
    # Per-feature summary.
    # ------------------------------------------------------------------
    feature_summary_rows = []

    for j, name in enumerate(
        FEATURES
    ):
        delta = feature_delta[
            :,
            j,
        ]

        feature_summary_rows.append(
            {
                "feature": name,
                "bias_RWP_minus_CONT": float(
                    np.mean(
                        delta
                    )
                ),
                "mae_RWP_vs_CONT": float(
                    np.mean(
                        np.abs(
                            delta
                        )
                    )
                ),
                "rmse_RWP_vs_CONT": float(
                    np.sqrt(
                        np.mean(
                            delta ** 2
                        )
                    )
                ),
                "max_abs_RWP_vs_CONT": float(
                    np.max(
                        np.abs(
                            delta
                        )
                    )
                ),
                "corr_RWP_vs_CONT": (
                    safe_corr(
                        X_val_full[
                            :,
                            j,
                        ],
                        X_val_rwp[
                            :,
                            j,
                        ],
                    )
                ),
            }
        )

    feature_summary_df = pd.DataFrame(
        feature_summary_rows
    )

    # ------------------------------------------------------------------
    # RSP ideal reference.
    #
    # Full restarting from chronology start applies exactly the same
    # composition Phi_t o ... o Phi_0 to the same rho_0. Therefore its ideal
    # endpoint state equals full CONT. We record this as an identity, rather
    # than spending time replaying the entire chronology 365 times.
    # ------------------------------------------------------------------
    rsp_identity = {
        "definition": (
            "For each endpoint, reset to the same initial memory state "
            "used at the beginning of the full CONT chronology and replay "
            "all inputs from chronology start through that endpoint."
        ),
        "ideal_equivalence": (
            "RSP ideal equals full CONT ideal by identical channel composition."
        ),
        "rmse": (
            full_metrics[
                "rmse"
            ]
        ),
        "mae": (
            full_metrics[
                "mae"
            ]
        ),
        "bias": (
            full_metrics[
                "bias"
            ]
        ),
    }

    return {
        "model": model,
        "scaler": scaler,
        "keep": keep,
        "full_metrics": (
            full_metrics
        ),
        "rwp_metrics": (
            rwp_metrics
        ),
        "protocol_metrics": (
            protocol_metrics
        ),
        "rsp_identity": (
            rsp_identity
        ),
        "predictions": (
            pred_df
        ),
        "features": (
            feature_df
        ),
        "feature_summary": (
            feature_summary_df
        ),
        "n_validation_endpoints": int(
            len(
                val_endpoints
            )
        ),
    }


# =============================================================================
# Resource projections extracted from the already-computed Week-10 one-step
# operating point. These are explicitly PROJECTIONS, not long-circuit compile
# results.
# =============================================================================

def build_resource_projection(
    one_step,
    K,
    n_endpoints,
    shots,
):
    if one_step is None:
        return {
            "available": False,
            "reason": (
                "Week-10 one-step direct-run row "
                "was not found."
            ),
        }

    n_settings = int(
        one_step[
            "n_settings"
        ]
    )

    one_step_cz = int(
        one_step[
            "feature_vector_n_cz_one_step"
        ]
    )

    max_setting_duration_one_step = float(
        one_step[
            "max_setting_duration_us_one_step"
        ]
    )

    # RWP: K input steps per endpoint.
    rwp_cz_per_feature = (
        one_step_cz
        * int(
            K
        )
    )

    rwp_max_setting_duration_us_linear = (
        max_setting_duration_one_step
        * int(
            K
        )
    )

    # Approximate all-settings duration as n_settings * max-setting duration.
    # We label this clearly as a projection only.
    rwp_projected_all_settings_duration_us_per_endpoint = (
        n_settings
        * rwp_max_setting_duration_us_linear
    )

    rwp_projected_scheduled_times_shots_s = (
        rwp_projected_all_settings_duration_us_per_endpoint
        * int(
            n_endpoints
        )
        * int(
            shots
        )
        / 1e6
    )

    # Full restart / RSP for 2025:
    # endpoint chronology lengths are N_TRAIN+1 ... N_TRAIN+N_VAL.
    first_steps = (
        int(
            common.N_TRAIN
        )
        + 1
    )

    last_steps = (
        int(
            common.N_TRAIN
        )
        + int(
            n_endpoints
        )
    )

    mean_rsp_steps = (
        0.5
        * (
            first_steps
            + last_steps
        )
    )

    rsp_mean_cz_per_feature = (
        one_step_cz
        * mean_rsp_steps
    )

    rsp_mean_max_setting_duration_us_linear = (
        max_setting_duration_one_step
        * mean_rsp_steps
    )

    rsp_projected_all_settings_duration_us_per_endpoint = (
        n_settings
        * rsp_mean_max_setting_duration_us_linear
    )

    rsp_projected_scheduled_times_shots_s = (
        rsp_projected_all_settings_duration_us_per_endpoint
        * int(
            n_endpoints
        )
        * int(
            shots
        )
        / 1e6
    )

    return {
        "available": True,
        "IMPORTANT": (
            "All K-scaled depth/duration/CZ values below are linear projections "
            "from the already-compiled Week-10 ONE-STEP operating point. "
            "They are NOT substitutes for compiling the real K=117 circuit."
        ),
        "source_one_step": (
            one_step
        ),
        "shots_per_setting": int(
            shots
        ),
        "n_settings": (
            n_settings
        ),
        "n_validation_endpoints": int(
            n_endpoints
        ),
        "rwp": {
            "K": int(
                K
            ),
            "projected_cz_per_feature_vector": int(
                rwp_cz_per_feature
            ),
            "projected_max_setting_duration_us": float(
                rwp_max_setting_duration_us_linear
            ),
            "projected_all_settings_duration_us_per_endpoint": float(
                rwp_projected_all_settings_duration_us_per_endpoint
            ),
            "projected_scheduled_duration_times_shots_s_full_2025": float(
                rwp_projected_scheduled_times_shots_s
            ),
        },
        "rsp_full_restart": {
            "first_2025_endpoint_replay_steps": int(
                first_steps
            ),
            "last_2025_endpoint_replay_steps": int(
                last_steps
            ),
            "mean_replay_steps": float(
                mean_rsp_steps
            ),
            "RSP_to_RWP_step_ratio": float(
                mean_rsp_steps
                / float(
                    K
                )
            ),
            "projected_mean_cz_per_feature_vector": float(
                rsp_mean_cz_per_feature
            ),
            "projected_mean_max_setting_duration_us": float(
                rsp_mean_max_setting_duration_us_linear
            ),
            "projected_scheduled_duration_times_shots_s_full_2025": float(
                rsp_projected_scheduled_times_shots_s
            ),
        },
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print(
        "=" * 126
    )
    print(
        "WEEK 11.1D.3 — FREEZE CANDIDATE #4 WASHOUT-DERIVED RWP REFERENCE"
    )
    print(
        "=" * 126
    )
    print(
        "NO IBM CONNECTION."
    )
    print(
        "NO QPU JOB WILL BE SUBMITTED."
    )
    print(
        "2026 = FROZEN / NOT LOADED"
    )
    print()

    # ------------------------------------------------------------------
    # Candidate definition from the final enriched manifest.
    # ------------------------------------------------------------------
    candidate, manifest_meta = (
        audit.load_candidate()
    )

    print(
        "Original selected candidate:"
    )
    print(
        f"  candidate={CANDIDATE_KEY}"
    )
    print(
        "  original protocol=full CONT"
    )
    print(
        f"  topology={candidate['topology']}"
    )
    print(
        f"  r={candidate['r']}"
    )
    print(
        f"  alpha={candidate['alpha']}"
    )
    print(
        f"  dt={candidate['dt']}"
    )
    print(
        f"  hx={candidate['hx']:+.12f}"
    )
    print(
        f"  hy={candidate['hy']:+.12f}"
    )
    print(
        f"  readout={audit.EXPECTED_READOUT}"
    )
    print()

    # ------------------------------------------------------------------
    # Extract previously computed evidence.
    # ------------------------------------------------------------------
    extracted = (
        extract_frozen_washout()
    )

    K = int(
        extracted[
            "operational_K"
        ]
    )

    print(
        "Extracted washout evidence:"
    )
    print(
        f"  global trace Tw(0.01)="
        f"{extracted['global_trace_washout_steps']}"
    )
    print(
        f"  global feature Tw(0.01)="
        f"{extracted['global_feature_washout_steps']}"
    )
    print(
        f"  operational K=max(...)={K}"
    )
    print(
        f"  local all-365 trace pass="
        f"{extracted['local_all_trace_pass']}"
    )
    print(
        f"  local trace max at K={K}: "
        f"{extracted['local_trace_max_at_K']:.6f}"
    )
    print(
        f"  local all-365 feature pass="
        f"{extracted['local_all_feature_pass']}"
    )
    print(
        f"  local feature max at K={K}: "
        f"{extracted['local_feature_max_at_K']:.6f}"
    )
    print()

    if K != 117:
        raise RuntimeError(
            "Current evidence does not freeze K=117. "
            f"Computed K={K}; stop and review."
        )

    print(
        "Extracted frozen full-CONT readout:"
    )
    print(
        f"  lambda="
        f"{extracted['stored_full_cont']['ridge_lambda']}"
    )
    print(
        f"  training-only CV RMSE="
        f"{extracted['stored_full_cont']['training_cv_rmse']:.6f}"
    )
    print(
        f"  stored ideal full-CONT 2025 RMSE="
        f"{extracted['stored_full_cont']['rmse']:.6f}"
    )
    print()

    # ------------------------------------------------------------------
    # Reconstruct exact feature trajectories only because we need the actual
    # frozen scaler/Ridge object and the exact K=117 predictions.
    # There is no hyperparameter re-selection here.
    # ------------------------------------------------------------------
    print(
        "Reconstructing the already-frozen full-CONT Ridge "
        "and exact ideal RWP(K=117) 2025 reference..."
    )

    reference = (
        build_reference_package(
            candidate,
            extracted,
        )
    )

    full_metrics = (
        reference[
            "full_metrics"
        ]
    )

    rwp_metrics = (
        reference[
            "rwp_metrics"
        ]
    )

    protocol = (
        reference[
            "protocol_metrics"
        ]
    )

    print()
    print(
        "REFERENCE DECOMPOSITION"
    )
    print(
        "-" * 126
    )
    print(
        f"Full CONT ideal 2025 RMSE:       "
        f"{full_metrics['rmse']:.6f}"
    )
    print(
        f"RSP ideal 2025 RMSE:             "
        f"{reference['rsp_identity']['rmse']:.6f} "
        f"(identical to full CONT by construction)"
    )
    print(
        f"RWP K={K} ideal 2025 RMSE:        "
        f"{rwp_metrics['rmse']:.6f}"
    )
    print(
        f"RWP K={K} ideal 2025 MAE:         "
        f"{rwp_metrics['mae']:.6f}"
    )
    print(
        f"RWP K={K} ideal 2025 bias:        "
        f"{rwp_metrics['bias']:+.6f}"
    )
    print()
    print(
        "Protocol approximation only "
        "(RWP ideal versus full CONT ideal):"
    )
    print(
        f"  target-RMSE delta:              "
        f"{protocol['delta_target_rmse_rwp_minus_cont']:+.6f}"
    )
    print(
        f"  prediction RMSE:                "
        f"{protocol['prediction_rmse_rwp_vs_cont']:.6f}"
    )
    print(
        f"  prediction MAE:                 "
        f"{protocol['prediction_mae_rwp_vs_cont']:.6f}"
    )
    print(
        f"  prediction correlation:         "
        f"{protocol['prediction_corr_rwp_vs_cont']:.9f}"
    )
    print(
        f"  feature RMSE:                   "
        f"{protocol['feature_rmse_rwp_vs_cont']:.6f}"
    )
    print(
        f"  feature MAE:                    "
        f"{protocol['feature_mae_rwp_vs_cont']:.6f}"
    )

    # ------------------------------------------------------------------
    # Extract resource point rather than pretending we already compiled
    # the actual 117-step circuits.
    # ------------------------------------------------------------------
    one_step = (
        extract_week10_resource_point()
    )

    resource_projection = (
        build_resource_projection(
            one_step=one_step,
            K=K,
            n_endpoints=(
                reference[
                    "n_validation_endpoints"
                ]
            ),
            shots=int(
                audit.SHOTS_PER_SETTING
            ),
        )
    )

    print()
    print(
        "RESOURCE PROJECTION"
    )
    print(
        "-" * 126
    )

    if resource_projection[
        "available"
    ]:
        rwp_res = (
            resource_projection[
                "rwp"
            ]
        )

        rsp_res = (
            resource_projection[
                "rsp_full_restart"
            ]
        )

        print(
            f"Week-10 one-step CZ/feature:      "
            f"{one_step['feature_vector_n_cz_one_step']}"
        )
        print(
            f"Projected RWP K={K} CZ/feature:    "
            f"{rwp_res['projected_cz_per_feature_vector']}"
        )
        print(
            f"Projected RWP max-setting duration:"
            f" {rwp_res['projected_max_setting_duration_us']:.3f} us"
        )
        print(
            f"Projected RWP scheduled*shots, "
            f"365 days: "
            f"{rwp_res['projected_scheduled_duration_times_shots_s_full_2025']:.3f} s"
        )
        print(
            f"RSP mean replay length:           "
            f"{rsp_res['mean_replay_steps']:.1f}"
        )
        print(
            f"RSP/RWP replay-step ratio:        "
            f"{rsp_res['RSP_to_RWP_step_ratio']:.3f}x"
        )
        print(
            f"Projected RSP mean CZ/feature:    "
            f"{rsp_res['projected_mean_cz_per_feature_vector']:.0f}"
        )
        print()
        print(
            "IMPORTANT: these are linear projections from the "
            "already-compiled ONE-STEP Week-10 point."
        )
        print(
            "The next hardware-preflight step must compile the "
            "actual 117-step circuit on a FRESH layout before any submission."
        )
    else:
        print(
            "Week-10 resource row unavailable; "
            "reference freeze continues without resource projection."
        )

    # ------------------------------------------------------------------
    # Save exact prediction / feature artifacts.
    # ------------------------------------------------------------------
    reference[
        "predictions"
    ].to_csv(
        OUT_PREDICTIONS,
        index=False,
    )

    reference[
        "features"
    ].to_csv(
        OUT_FEATURES,
        index=False,
    )

    reference[
        "feature_summary"
    ].to_csv(
        OUT_FEATURE_SUMMARY,
        index=False,
    )

    package = {
        "candidate_key": (
            CANDIDATE_KEY
        ),
        "original_model_protocol": (
            "CONT"
        ),
        "hardware_surrogate_protocol": (
            "washout_derived_RWP"
        ),
        "operational_K": (
            K
        ),
        "epsilon_primary": (
            PRIMARY_EPSILON
        ),
        "topology": (
            candidate[
                "topology"
            ]
        ),
        "r": int(
            candidate[
                "r"
            ]
        ),
        "alpha": float(
            candidate[
                "alpha"
            ]
        ),
        "dt": float(
            candidate[
                "dt"
            ]
        ),
        "hx": float(
            candidate[
                "hx"
            ]
        ),
        "hy": float(
            candidate[
                "hy"
            ]
        ),
        "J": {
            f"J{i}{j}": float(v)
            for (i, j), v
            in candidate[
                "J"
            ].items()
        },
        "readout": (
            audit.EXPECTED_READOUT
        ),
        "features": (
            FEATURES
        ),
        "shots_per_setting_planned": int(
            audit.SHOTS_PER_SETTING
        ),
        "n_validation_endpoints": int(
            reference[
                "n_validation_endpoints"
            ]
        ),
        "planned_measurement_settings": 2,
        "planned_qpu_circuits": int(
            2
            * reference[
                "n_validation_endpoints"
            ]
        ),
        "frozen_ridge": {
            "lambda": float(
                extracted[
                    "stored_full_cont"
                ][
                    "ridge_lambda"
                ]
            ),
            "training_cv_rmse_from_existing_audit": float(
                extracted[
                    "stored_full_cont"
                ][
                    "training_cv_rmse"
                ]
            ),
            "trained_on": (
                "full-CONT 2022-2024 feature trajectory"
            ),
            "retrained_for_RWP": False,
        },
        "washout_evidence": (
            extracted
        ),
        "full_CONT_ideal_2025": (
            full_metrics
        ),
        "RSP_ideal_2025": (
            reference[
                "rsp_identity"
            ]
        ),
        f"RWP_K{K}_ideal_2025": (
            rwp_metrics
        ),
        "protocol_approximation_RWP_vs_CONT": (
            protocol
        ),
        "resource_projection": (
            resource_projection
        ),
        "artifacts": {
            "predictions_csv": str(
                OUT_PREDICTIONS
            ),
            "feature_comparison_csv": str(
                OUT_FEATURES
            ),
            "feature_summary_csv": str(
                OUT_FEATURE_SUMMARY
            ),
            "package_json": str(
                OUT_PACKAGE
            ),
        },
        "next_required_step_before_QPU": (
            "Fresh backend/layout reselection followed by compilation of "
            "the ACTUAL K=117 RWP circuits. Measure true compiled CZ, depth, "
            "scheduled duration, swaps and duration/T1/T2. Do not infer "
            "hardware feasibility only from the linear resource projection."
        ),
        "2026_loaded": False,
    }

    OUT_PACKAGE.write_text(
        json.dumps(
            db.json_safe(
                package
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Database persistence.
    # ------------------------------------------------------------------
    db.create_all_tables()

    run_uuid = db.create_run(
        script_name=(
            Path(
                __file__
            ).name
        ),
        run_status=(
            "RWP_REFERENCE_FROZEN"
        ),
        started_at_utc=(
            db.utc_now_naive()
        ),
        completed_at_utc=(
            db.utc_now_naive()
        ),
        forecast_dataset_name=(
            "property_damage_next_day_v1"
        ),
        feature_set_name="F4",
        evaluation_split=(
            "validation_2025"
        ),
        candidate_key=(
            "CONT_H0_R4_RWP117_REF"
        ),
        selected_by_rules=(
            "Rule4"
        ),
        protocol=(
            "RWP_from_CONT"
        ),
        topology=(
            candidate[
                "topology"
            ]
        ),
        window_size=(
            K
        ),
        trotter_r=int(
            candidate[
                "r"
            ]
        ),
        readout_name=(
            audit.EXPECTED_READOUT
        ),
        primitive_name=(
            "ideal_simulation"
        ),
        measurement_method=(
            "endpoint_only_RWP_reference"
        ),
        alpha=float(
            candidate[
                "alpha"
            ]
        ),
        dt=float(
            candidate[
                "dt"
            ]
        ),
        hx=float(
            candidate[
                "hx"
            ]
        ),
        hy=float(
            candidate[
                "hy"
            ]
        ),
        j_json=(
            package[
                "J"
            ]
        ),
        manifest_json=(
            manifest_meta
        ),
        ridge_lambda=float(
            extracted[
                "stored_full_cont"
            ][
                "ridge_lambda"
            ]
        ),
        training_cv_rmse=float(
            extracted[
                "stored_full_cont"
            ][
                "training_cv_rmse"
            ]
        ),
        ideal_validation_rmse_full_2025=float(
            rwp_metrics[
                "rmse"
            ]
        ),
        ideal_validation_mae_full_2025=float(
            rwp_metrics[
                "mae"
            ]
        ),
        ideal_validation_bias_full_2025=float(
            rwp_metrics[
                "bias"
            ]
        ),
        shots_per_setting=int(
            audit.SHOTS_PER_SETTING
        ),
        n_validation_endpoints=int(
            reference[
                "n_validation_endpoints"
            ]
        ),
        n_circuits=int(
            2
            * reference[
                "n_validation_endpoints"
            ]
        ),
        full_2025=True,
        pilot_selection=(
            "all_2025_validation"
        ),
        notes=(
            "Candidate #4 frozen washout-derived RWP reference. "
            "Original model is full CONT; same full-CONT 2022-2024 "
            "feature scaler/Ridge is reused without RWP retraining. "
            "K=117 is extracted from prior global/local washout audits. "
            "RSP ideal equals full CONT ideal by identical channel "
            "composition. No IBM connection and no QPU job."
        ),
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_PREDICTIONS.name,
        reference[
            "predictions"
        ],
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_FEATURES.name,
        reference[
            "features"
        ],
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_FEATURE_SUMMARY.name,
        reference[
            "feature_summary"
        ],
    )

    db.save_json_artifact(
        run_uuid,
        OUT_PACKAGE.name,
        package,
    )

    print()
    print(
        f"[database] Frozen RWP-reference run UUID: "
        f"{run_uuid}"
    )

    print()
    print(
        "Saved:"
    )
    print(
        f"  {OUT_PREDICTIONS}"
    )
    print(
        f"  {OUT_FEATURES}"
    )
    print(
        f"  {OUT_FEATURE_SUMMARY}"
    )
    print(
        f"  {OUT_PACKAGE}"
    )

    print()
    print(
        "STOP HERE."
    )
    print(
        "Send me the REFERENCE DECOMPOSITION and RESOURCE PROJECTION."
    )
    print(
        "Only after that should we build/compile the actual K=117 "
        "hardware circuits."
    )


if __name__ == "__main__":
    main()
