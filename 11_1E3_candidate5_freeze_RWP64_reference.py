from __future__ import annotations

"""
WEEK 11.1E.3 — CANDIDATE #5 FREEZE WASHOUT-DERIVED RWP(64) REFERENCE
====================================================================

Purpose
-------
Freeze Candidate #5 exactly as we did Candidate #4:

    full CONT ideal
        -> ideal RWP(K=64)
        -> future QPU RWP(K=64), only if hardware preflight is viable.

This script:
    - extracts the already-selected Candidate #5 from Week-10 artifacts,
    - extracts the already-selected Ridge lambda / CV / ideal validation metrics,
    - verifies global + local washout evidence,
    - freezes K_RWP = max(global trace Tw, global feature Tw) = 64,
    - reconstructs the SAME full-CONT training trajectory,
    - fits the SAME frozen Ridge readout using the already-selected lambda,
    - computes exact ideal RWP64 predictions for all 365 validation endpoints,
    - keeps ideal RSP == full CONT by construction,
    - decomposes protocol-approximation error,
    - saves daily predictions, feature comparisons, JSON, and DB records.

IMPORTANT
---------
We DO NOT train a new RWP-specific model.
The selected model remains:
    full CONT reservoir + full-CONT 2022-2024 trained scaler/Ridge.

RWP64 is only a hardware-realizable state-reconstruction surrogate.

NO IBM CONNECTION.
NO QPU JOB.
2026 IS NEVER LOADED.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import db_objects as db


# =============================================================================
# Paths / frozen identity
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

GLOBAL_SCRIPT = HERE / "11_1E1_candidate5_washout_trace_distance.py"

MANIFEST_FILE = (
    RESULTS
    / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"
)

DIRECT_FILE = (
    RESULTS
    / "10_05_direct_runs.csv"
)

GLOBAL_THRESHOLDS = (
    RESULTS
    / "11_1E1_candidate5_washout_thresholds.csv"
)

GLOBAL_SUMMARY = (
    RESULTS
    / "11_1E1_candidate5_washout_summary.json"
)

LOCAL_SUMMARY_CSV = (
    RESULTS
    / "11_1E2_candidate5_local_washout_summary.csv"
)

LOCAL_SUMMARY_JSON = (
    RESULTS
    / "11_1E2_candidate5_local_washout_summary.json"
)

OUT_PREDICTIONS = (
    RESULTS
    / "11_1E3_candidate5_RWP64_predictions.csv"
)

OUT_FEATURES = (
    RESULTS
    / "11_1E3_candidate5_RWP64_feature_comparison.csv"
)

OUT_FEATURE_SUMMARY = (
    RESULTS
    / "11_1E3_candidate5_RWP64_feature_summary.csv"
)

OUT_PACKAGE = (
    RESULTS
    / "11_1E3_candidate5_RWP64_frozen_package.json"
)

EXPECTED_CANDIDATE = "CONT_H2_R2"
EXPECTED_PROTOCOL = "CONT"
EXPECTED_TOPOLOGY = "H2"
EXPECTED_R = 2
EXPECTED_READOUT = "XZinj"
EXPECTED_K = 64
PRIMARY_EPSILON = 0.01
PLANNED_SHOTS = 1024


# =============================================================================
# Module loading
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required project file not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


prev = load_module(
    GLOBAL_SCRIPT,
    "qrc_11_1e3_prev",
)

common = prev.common
qrc = prev.qrc
FEATURES = list(prev.FEATURES)
INITIAL_STATES = prev.INITIAL_STATES


# =============================================================================
# Helpers
# =============================================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
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
    atol=2e-6,
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


def normalize_density(rho):
    rho = np.asarray(
        rho,
        dtype=complex,
    )

    rho = 0.5 * (
        rho
        + rho.conj().T
    )

    tr = np.trace(
        rho
    )

    if abs(tr) < 1e-15:
        raise RuntimeError(
            "Zero-trace memory state."
        )

    return rho / tr


def get_first_present(row, names, required=False):
    for name in names:
        if name in row.index:
            value = row[name]

            if pd.notna(value):
                return value

    if required:
        raise KeyError(
            f"None of required columns present/non-null: {names}"
        )

    return None


# =============================================================================
# Extract frozen Candidate #5 and its already-selected Ridge metadata
# =============================================================================

def resolve_direct_row(candidate_key):
    """
    Resolve Candidate #5 by its EXACT frozen Week-10 candidate key.

    Important:
    ----------
    In 10_05_direct_runs.csv, keys such as CONT_H2_R2 / CONT_H2_R3 /
    CONT_H2_R4 denote different retained candidate branches.  Their column
    `r` can still be the same Trotter order, so protocol+topology+r+readout is
    NOT a unique identifier.

    Therefore Candidate #5 must be resolved strictly as CONT_H2_R2.
    No fallback matching is allowed.
    """
    if not DIRECT_FILE.exists():
        raise FileNotFoundError(
            f"Missing Week-10 direct-run artifact: {DIRECT_FILE}"
        )

    df = pd.read_csv(
        DIRECT_FILE
    )

    if "candidate_key" not in df.columns:
        raise RuntimeError(
            "10_05_direct_runs.csv has no candidate_key column. "
            "Candidate #5 cannot be resolved safely without the exact key."
        )

    exact_key = df[
        df[
            "candidate_key"
        ].astype(str)
        .str.strip()
        == str(
            candidate_key
        ).strip()
    ].copy()

    if len(exact_key) == 0:
        raise RuntimeError(
            f"Exact Candidate #5 row '{candidate_key}' was not found in "
            "10_05_direct_runs.csv. No fallback matching will be used."
        )

    # IMPORTANT:
    # The same candidate_key is repeated across the Week-10 Trotter sweep.
    # Candidate #5 is the OPERATING r=2 row of CONT_H2_R2.
    #
    # Composite frozen identity:
    #   candidate_key = CONT_H2_R2
    #   r             = 2
    #   protocol      = CONT
    #   topology      = H2
    #
    # This uniquely identifies the actual retained Candidate #5 row.
    if "r" not in exact_key.columns:
        raise RuntimeError(
            "10_05_direct_runs.csv has no 'r' column, so the frozen "
            "Candidate #5 operating point cannot be resolved safely."
        )

    exact = exact_key[
        pd.to_numeric(
            exact_key[
                "r"
            ],
            errors="coerce",
        )
        == EXPECTED_R
    ].copy()

    if len(exact) == 0:
        raise RuntimeError(
            f"Found candidate_key='{candidate_key}' but no row with "
            f"operating r={EXPECTED_R}."
        )

    if len(exact) > 1:
        cols = [
            c
            for c in [
                "candidate_key",
                "protocol",
                "topology",
                "r",
                "readout",
                "test_readout",
                "ridge_lambda",
                "training_cv_rmse",
                "ideal_validation_rmse",
                "rmse",
            ]
            if c in exact.columns
        ]

        print(
            "Duplicate Candidate #5 rows after exact key + exact r filtering:"
        )
        print(
            exact[
                cols
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"Found {len(exact)} rows with candidate_key='{candidate_key}' "
            f"and r={EXPECTED_R}. Expected exactly one."
        )

    row = exact.iloc[0]

    # Hard audit of the branch identity.
    if "protocol" in row.index:
        if str(
            row[
                "protocol"
            ]
        ).strip().upper() != EXPECTED_PROTOCOL:
            raise RuntimeError(
                "Exact Candidate #5 row has unexpected protocol: "
                f"{row['protocol']}"
            )

    if "topology" in row.index:
        if str(
            row[
                "topology"
            ]
        ).strip() != EXPECTED_TOPOLOGY:
            raise RuntimeError(
                "Exact Candidate #5 row has unexpected topology: "
                f"{row['topology']}"
            )

    return row


def extract_frozen_readout_metadata(
    candidate,
):
    """
    Extract the quantities that are actually stored in Week-10.

    Important:
    ----------
    10_05_direct_runs.csv does NOT store the selected Ridge lambda for this
    candidate.  Therefore we must not invent or guess it.

    Instead we extract the historical training-only CV RMSE and ideal 2025
    validation metrics, then deterministically RECOMPUTE the selected lambda
    from the original 2022-2024 full-CONT training features using the same
    common.select_lambda_training_only() routine.  The recomputed CV and ideal
    validation RMSE are then audited against the stored Week-10 values.

    This is a reconstruction audit, not a new model-selection experiment:
    - same candidate dynamics,
    - same readout,
    - same 2022-2024 training set,
    - same lambda grid/CV routine,
    - 2025 used only to verify the already-reported ideal validation result.
    """
    row = resolve_direct_row(
        candidate[
            "candidate_id"
        ]
    )

    training_cv_rmse = get_first_present(
        row,
        [
            "training_cv_rmse",
            "cv_rmse",
            "mean_cv_rmse",
            "cv_mean_rmse",
        ],
        required=True,
    )

    ideal_rmse = get_first_present(
        row,
        [
            "ideal_validation_rmse",
            "ideal_rmse",
            "validation_ideal_rmse",
        ],
        required=True,
    )

    ideal_mae = get_first_present(
        row,
        [
            "ideal_validation_mae",
            "ideal_mae",
            "validation_ideal_mae",
        ],
        required=False,
    )

    ideal_bias = get_first_present(
        row,
        [
            "ideal_validation_bias",
            "ideal_bias",
            "validation_ideal_bias",
        ],
        required=False,
    )

    return {
        "ridge_lambda": None,
        "lambda_source": (
            "recomputed from frozen 2022-2024 full-CONT training features "
            "with common.select_lambda_training_only(); Week-10 direct-run "
            "table does not store lambda"
        ),
        "training_cv_rmse": float(
            training_cv_rmse
        ),
        "stored_ideal_validation_rmse": float(
            ideal_rmse
        ),
        "stored_ideal_validation_mae": (
            None
            if ideal_mae is None
            else float(
                ideal_mae
            )
        ),
        "stored_ideal_validation_bias": (
            None
            if ideal_bias is None
            else float(
                ideal_bias
            )
        ),
        "source_row": (
            row.to_dict()
        ),
    }


# =============================================================================
# Freeze K from prior global/local evidence
# =============================================================================

def extract_frozen_washout():
    thresholds = load_csv(
        GLOBAL_THRESHOLDS
    )

    local = load_csv(
        LOCAL_SUMMARY_CSV
    )

    global_summary = load_json(
        GLOBAL_SUMMARY
    )

    local_summary_json = load_json(
        LOCAL_SUMMARY_JSON
    )

    eps_rows = thresholds[
        np.isclose(
            thresholds[
                "epsilon"
            ].astype(float),
            PRIMARY_EPSILON,
        )
    ]

    if len(eps_rows) != 1:
        raise RuntimeError(
            "Expected exactly one epsilon=0.01 global row."
        )

    eps_row = eps_rows.iloc[
        0
    ]

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

    K = max(
        trace_tw,
        feature_tw,
    )

    if K != EXPECTED_K:
        raise RuntimeError(
            f"Expected K={EXPECTED_K}, got K={K}."
        )

    local_row = local[
        local[
            "context_steps"
        ].astype(int)
        == K
    ]

    if len(local_row) != 1:
        raise RuntimeError(
            f"Local audit does not contain exact K={K}."
        )

    local_row = local_row.iloc[
        0
    ]

    if not bool(
        local_row[
            "all_trace_le_0.01"
        ]
    ):
        raise RuntimeError(
            f"K={K} fails local all-endpoint trace criterion."
        )

    if not bool(
        local_row[
            "all_feature_le_0.01"
        ]
    ):
        raise RuntimeError(
            f"K={K} fails local all-endpoint feature criterion."
        )

    return {
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
        "smallest_tested_local_both": (
            local_summary_json.get(
                "smallest_tested_all_both_le_0p01"
            )
        ),
    }


# =============================================================================
# Full-CONT trajectory and exact RWP replay
# =============================================================================

def feature_vector(
    rho_i,
    rho_m,
):
    row = common.reduced_feature_row(
        rho_i,
        rho_m,
    )

    return np.asarray(
        [
            float(
                row[
                    name
                ]
            )
            for name
            in FEATURES
        ],
        dtype=float,
    )


def build_full_cont_bank(
    candidate,
    angles,
):
    A_list = common.build_channels(
        candidate,
        angles,
    )

    rho_m = INITIAL_STATES[
        "00"
    ].copy()

    rows = []

    for A_t in A_list:
        rho_i, rho_m_next = (
            qrc.final_reduced_states(
                A_t,
                rho_m,
            )
        )

        rho_m = normalize_density(
            rho_m_next
        )

        rows.append(
            feature_vector(
                rho_i,
                rho_m,
            )
        )

    return (
        np.vstack(
            rows
        ),
        A_list,
    )


def replay_feature(
    A_list,
    endpoint,
    K,
):
    start = (
        int(
            endpoint
        )
        - int(
            K
        )
        + 1
    )

    if start < 0:
        raise ValueError(
            f"endpoint={endpoint}, K={K} => start={start}"
        )

    rho_m = INITIAL_STATES[
        "00"
    ].copy()

    rho_i = None

    for idx in range(
        start,
        int(
            endpoint
        )
        + 1,
    ):
        rho_i, rho_m_next = (
            qrc.final_reduced_states(
                A_list[
                    idx
                ],
                rho_m,
            )
        )

        rho_m = normalize_density(
            rho_m_next
        )

    return feature_vector(
        rho_i,
        rho_m,
    )


# =============================================================================
# Main reference reconstruction
# =============================================================================

def build_reference(
    candidate,
    frozen_readout,
    washout,
):
    (
        work_tv,
        cols,
        y_all,
    ) = common.load_train_validation()

    angles = common.make_angles(
        work_tv,
        cols,
        float(
            candidate[
                "alpha"
            ]
        ),
    )

    (
        full_bank,
        A_list,
    ) = build_full_cont_bank(
        candidate,
        angles,
    )

    endpoints = np.arange(
        len(
            full_bank
        ),
        dtype=int,
    )

    if len(
        endpoints
    ) != (
        common.N_TRAIN
        + common.N_VAL
    ):
        raise RuntimeError(
            "Unexpected train+validation chronology length."
        )

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

    y_train = np.asarray(
        y_all[
            train_mask
        ],
        dtype=float,
    )

    X_val_full = full_bank[
        val_mask
    ]

    y_val = np.asarray(
        y_all[
            val_mask
        ],
        dtype=float,
    )

    val_endpoints = endpoints[
        val_mask
    ]

    # ------------------------------------------------------------------
    # Reconstruct the historically selected Ridge lambda from the ORIGINAL
    # full-CONT 2022-2024 training features.
    #
    # The Week-10 direct-run table stores the resulting CV RMSE but not the
    # lambda itself.  We therefore rerun the exact training-only selector and
    # audit its CV result against the stored Week-10 value.
    # ------------------------------------------------------------------
    (
        selected_lambda,
        cv_folds,
        cv_summary,
    ) = common.select_lambda_training_only(
        X_train,
        y_train,
        n_splits=5,
    )

    selected_lambda = float(
        selected_lambda
    )

    recomputed_cv_rmse = float(
        cv_summary.iloc[
            0
        ][
            "cv_rmse_mean"
        ]
    )

    expected_cv_rmse = float(
        frozen_readout[
            "training_cv_rmse"
        ]
    )

    if abs(
        recomputed_cv_rmse
        - expected_cv_rmse
    ) > 1e-5:
        raise RuntimeError(
            "Candidate #5 CV reconstruction audit failed: "
            f"recomputed={recomputed_cv_rmse:.9f}, "
            f"stored Week-10={expected_cv_rmse:.9f}. "
            "Do not continue with an inconsistent Ridge model."
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

    # Validate against stored Week-10 ideal result where available.
    if (
        frozen_readout[
            "stored_ideal_validation_rmse"
        ]
        is not None
    ):
        numeric_close(
            full_metrics[
                "rmse"
            ],
            frozen_readout[
                "stored_ideal_validation_rmse"
            ],
            atol=3e-6,
            label=(
                "Candidate #5 full-CONT ideal validation RMSE"
            ),
        )

    if (
        frozen_readout[
            "stored_ideal_validation_mae"
        ]
        is not None
    ):
        numeric_close(
            full_metrics[
                "mae"
            ],
            frozen_readout[
                "stored_ideal_validation_mae"
            ],
            atol=3e-6,
            label=(
                "Candidate #5 full-CONT ideal validation MAE"
            ),
        )

    if (
        frozen_readout[
            "stored_ideal_validation_bias"
        ]
        is not None
    ):
        numeric_close(
            full_metrics[
                "bias"
            ],
            frozen_readout[
                "stored_ideal_validation_bias"
            ],
            atol=3e-6,
            label=(
                "Candidate #5 full-CONT ideal validation bias"
            ),
        )

    K = int(
        washout[
            "operational_K"
        ]
    )

    X_val_rwp = np.vstack(
        [
            replay_feature(
                A_list,
                endpoint=int(
                    endpoint
                ),
                K=K,
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

    pred_delta = (
        pred_rwp
        - pred_full
    )

    feature_delta = (
        X_val_rwp
        - X_val_full
    )

    protocol = {
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
                    pred_delta
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
            "target": (
                y_val
            ),
            "pred_full_CONT_ideal": (
                pred_full
            ),
            f"pred_RWP_K{K}_ideal": (
                pred_rwp
            ),
            "pred_RWP_minus_CONT": (
                pred_delta
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
        }
    )

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
                "feature": (
                    name
                ),
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

    rsp = {
        "definition": (
            "Reset to the same chronology-start memory state and replay "
            "all inputs from chronology start through each endpoint."
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
        "selected_lambda": (
            selected_lambda
        ),
        "cv_rmse": (
            recomputed_cv_rmse
        ),
        "cv_folds": (
            cv_folds
        ),
        "full_metrics": (
            full_metrics
        ),
        "rwp_metrics": (
            rwp_metrics
        ),
        "protocol": (
            protocol
        ),
        "rsp": (
            rsp
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
        "n_validation_endpoints": (
            len(
                val_endpoints
            )
        ),
    }


# =============================================================================
# Optional Week-10 one-step resource extraction
# =============================================================================

def extract_one_step_resources(
    candidate_key,
):
    if not DIRECT_FILE.exists():
        return None

    row = resolve_direct_row(
        candidate_key
    )

    def maybe(names):
        return get_first_present(
            row,
            names,
            required=False,
        )

    values = {
        "feature_vector_n_cz_one_step": maybe(
            [
                "feature_vector_n_cz",
                "n_cz_per_feature",
            ]
        ),
        "max_setting_depth_one_step": maybe(
            [
                "max_setting_depth",
            ]
        ),
        "max_setting_duration_us_one_step": maybe(
            [
                "max_setting_duration_us",
            ]
        ),
        "n_settings": maybe(
            [
                "n_settings",
            ]
        ),
        "week10_sampled_rmse": maybe(
            [
                "rmse",
            ]
        ),
        "week10_sampled_rmse_sd": maybe(
            [
                "rmse_sd",
                "rmse_std",
            ]
        ),
    }

    if values[
        "feature_vector_n_cz_one_step"
    ] is None:
        return None

    return {
        k: (
            None
            if v is None
            else (
                int(v)
                if k in {
                    "feature_vector_n_cz_one_step",
                    "n_settings",
                }
                else float(v)
            )
        )
        for k, v
        in values.items()
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print(
        "=" * 126
    )
    print(
        "WEEK 11.1E.3 — FREEZE CANDIDATE #5 WASHOUT-DERIVED RWP(64) REFERENCE"
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

    candidate, manifest_meta = (
        prev.load_candidate()
    )

    if candidate[
        "candidate_id"
    ] != EXPECTED_CANDIDATE:
        raise RuntimeError(
            f"Expected candidate {EXPECTED_CANDIDATE}, "
            f"got {candidate['candidate_id']}"
        )

    washout = extract_frozen_washout()

    frozen_readout = (
        extract_frozen_readout_metadata(
            candidate
        )
    )

    print(
        "Frozen Candidate #5:"
    )
    print(
        f"  candidate={candidate['candidate_id']}"
    )
    print(
        f"  protocol={candidate['protocol']}"
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
        f"  readout={EXPECTED_READOUT}"
    )
    print()

    print(
        "Frozen washout evidence:"
    )
    print(
        f"  global trace Tw(0.01)="
        f"{washout['global_trace_washout_steps']}"
    )
    print(
        f"  global feature Tw(0.01)="
        f"{washout['global_feature_washout_steps']}"
    )
    print(
        f"  operational K="
        f"{washout['operational_K']}"
    )
    print(
        f"  local trace max at K=64="
        f"{washout['local_trace_max_at_K']:.6f}"
    )
    print(
        f"  local feature max at K=64="
        f"{washout['local_feature_max_at_K']:.6f}"
    )
    print(
        f"  smallest tested local K satisfying both="
        f"{washout['smallest_tested_local_both']}"
    )
    print()

    print(
        "Exact Week-10 Candidate #5 row resolved:"
    )
    print(
        f"  candidate_key={candidate['candidate_id']}"
    )
    print(
        f"  operating r={EXPECTED_R}"
    )
    print(
        "  resolution=EXACT candidate_key + operating r match"
    )
    print()

    print(
        "Frozen full-CONT readout metadata:"
    )
    print(
        "  lambda=NOT STORED in 10_05_direct_runs.csv "
        "(will be reconstructed from 2022-2024 training only)"
    )
    print(
        f"  stored training-only CV RMSE="
        f"{frozen_readout['training_cv_rmse']}"
    )
    print(
        f"  stored full-CONT ideal 2025 RMSE="
        f"{frozen_readout['stored_ideal_validation_rmse']}"
    )
    print()

    print(
        "Reconstructing frozen full-CONT trajectory/readout "
        "and exact ideal RWP64 reference..."
    )

    reference = build_reference(
        candidate,
        frozen_readout,
        washout,
    )

    full = reference[
        "full_metrics"
    ]

    rwp = reference[
        "rwp_metrics"
    ]

    protocol = reference[
        "protocol"
    ]

    print()
    print(
        "RIDGE RECONSTRUCTION AUDIT"
    )
    print(
        "-" * 126
    )
    print(
        f"Recomputed selected lambda:       "
        f"{reference['selected_lambda']}"
    )
    print(
        f"Recomputed training CV RMSE:      "
        f"{reference['cv_rmse']:.6f}"
    )
    print(
        f"Stored Week-10 training CV RMSE:  "
        f"{frozen_readout['training_cv_rmse']:.6f}"
    )
    print(
        "CV audit: PASS"
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
        f"{full['rmse']:.6f}"
    )
    print(
        f"Full CONT ideal 2025 MAE:        "
        f"{full['mae']:.6f}"
    )
    print(
        f"Full CONT ideal 2025 bias:       "
        f"{full['bias']:+.6f}"
    )
    print(
        f"RSP ideal 2025 RMSE:             "
        f"{reference['rsp']['rmse']:.6f} "
        f"(identical to full CONT by construction)"
    )
    print(
        f"RWP K=64 ideal 2025 RMSE:        "
        f"{rwp['rmse']:.6f}"
    )
    print(
        f"RWP K=64 ideal 2025 MAE:         "
        f"{rwp['mae']:.6f}"
    )
    print(
        f"RWP K=64 ideal 2025 bias:        "
        f"{rwp['bias']:+.6f}"
    )
    print()

    print(
        "Protocol approximation only "
        "(RWP64 ideal versus full CONT ideal):"
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
    # One-step resource projection only; actual K=64 compile is next step.
    # ------------------------------------------------------------------
    one_step = extract_one_step_resources(
        candidate[
            "candidate_id"
        ]
    )

    resource_projection = None

    if (
        one_step is not None
        and one_step[
            "feature_vector_n_cz_one_step"
        ] is not None
    ):
        resource_projection = {
            "source": (
                "10_05_direct_runs.csv one-step operating point"
            ),
            "IMPORTANT": (
                "Linear projection only. The next step must compile "
                "the actual K=64 circuit on a fresh layout."
            ),
            "one_step": (
                one_step
            ),
            "RWP64_projected_cz_per_feature_vector": int(
                one_step[
                    "feature_vector_n_cz_one_step"
                ]
                * EXPECTED_K
            ),
        }

        if (
            one_step[
                "max_setting_duration_us_one_step"
            ]
            is not None
        ):
            resource_projection[
                "RWP64_projected_max_setting_duration_us"
            ] = float(
                one_step[
                    "max_setting_duration_us_one_step"
                ]
                * EXPECTED_K
            )

        print()
        print(
            "RESOURCE PROJECTION"
        )
        print(
            "-" * 126
        )
        print(
            f"Week-10 one-step CZ/feature:      "
            f"{one_step['feature_vector_n_cz_one_step']}"
        )
        print(
            f"Projected RWP K=64 CZ/feature:    "
            f"{resource_projection['RWP64_projected_cz_per_feature_vector']}"
        )

        if (
            "RWP64_projected_max_setting_duration_us"
            in resource_projection
        ):
            print(
                f"Projected RWP max-setting duration:"
                f" "
                f"{resource_projection['RWP64_projected_max_setting_duration_us']:.3f} us"
            )

        print(
            "IMPORTANT: projection only; actual K=64 "
            "long-circuit compilation comes next."
        )

    # ------------------------------------------------------------------
    # Save exact artifacts.
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
            candidate[
                "candidate_id"
            ]
        ),
        "original_model_protocol": (
            "CONT"
        ),
        "hardware_surrogate_protocol": (
            "washout_derived_RWP"
        ),
        "operational_K": (
            EXPECTED_K
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
            for (
                i,
                j,
            ), v
            in candidate[
                "J"
            ].items()
        },
        "readout": (
            EXPECTED_READOUT
        ),
        "features": (
            FEATURES
        ),
        "planned_shots_per_setting": (
            PLANNED_SHOTS
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
                reference[
                    "selected_lambda"
                ]
            ),
            "lambda_source": (
                frozen_readout[
                    "lambda_source"
                ]
            ),
            "training_cv_rmse": float(
                reference[
                    "cv_rmse"
                ]
            ),
            "trained_on": (
                "full-CONT 2022-2024 feature trajectory"
            ),
            "retrained_for_RWP": False,
        },
        "washout_evidence": (
            washout
        ),
        "full_CONT_ideal_2025": (
            full
        ),
        "RSP_ideal_2025": (
            reference[
                "rsp"
            ]
        ),
        "RWP_K64_ideal_2025": (
            rwp
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
            "Fresh H2 hardware reselection and compilation of the "
            "ACTUAL K=64 RWP circuits. Measure real CZ count, depth, "
            "scheduled duration, SWAPs, duration/T1 and duration/T2."
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
    # Database
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
            "CONT_H2_R2_RWP64_REF"
        ),
        selected_by_rules=(
            str(
                manifest_meta.get(
                    "selected_by_rules",
                    "Rule2",
                )
            )
        ),
        protocol=(
            "RWP_from_CONT"
        ),
        topology=(
            EXPECTED_TOPOLOGY
        ),
        window_size=(
            EXPECTED_K
        ),
        trotter_r=(
            EXPECTED_R
        ),
        readout_name=(
            EXPECTED_READOUT
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
            reference[
                "selected_lambda"
            ]
        ),
        training_cv_rmse=float(
            reference[
                "cv_rmse"
            ]
        ),
        ideal_validation_rmse_full_2025=float(
            rwp[
                "rmse"
            ]
        ),
        ideal_validation_mae_full_2025=float(
            rwp[
                "mae"
            ]
        ),
        ideal_validation_bias_full_2025=float(
            rwp[
                "bias"
            ]
        ),
        shots_per_setting=(
            PLANNED_SHOTS
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
            "Candidate #5 frozen washout-derived RWP64 reference. "
            "Original model remains full CONT. The same 2022-2024 "
            "full-CONT scaler/Ridge is reused without RWP retraining. "
            "K=64 is max(global trace Tw=64, global feature Tw=54) "
            "and passes local all-365 trace+feature criteria. "
            "No IBM connection and no QPU job."
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
        f"[database] Candidate #5 frozen RWP-reference run UUID: "
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
        "Only after that do we compile the actual K=64 hardware circuits."
    )


if __name__ == "__main__":
    main()
