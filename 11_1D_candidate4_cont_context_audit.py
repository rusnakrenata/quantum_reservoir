from __future__ import annotations

"""
WEEK 11.1D — CANDIDATE #4 CONTEXT AUDIT BEFORE REAL-QPU SUBMISSION

Candidate #4:
    CONT_H0_R4
    protocol = CONT
    topology = H0
    Week-10 hardware operating point: r=4
    alpha = 0.75
    readout = XZinj_plus_YX45
    shots/setting = 512

WHY THIS AUDIT IS REQUIRED
--------------------------
Candidates #1-#3 were RWP reservoirs.  Their state can be reconstructed from a
short finite replay window by definition.

Candidate #4 is CONT.  In the ideal CONT model the memory state is propagated
chronologically through the whole series.  On a real QPU we cannot copy/save
the unknown memory state at every 2025 endpoint and then measure it without
disturbing subsequent memory.

Therefore the physically executable hardware protocol must use an independent
finite-context replay for each 2025 endpoint:

    |00> memory
      -> replay K chronological inputs without measurement
      -> measure only at the FINAL endpoint

This script does NOT submit any QPU job.  It determines how large K must be
before finite-context replay faithfully approximates the frozen full-CONT
candidate.

It stores its audit results in both:
    results/*.csv / *.json
and
    qrc_qpu_run + qrc_qpu_run_csv_row + qrc_qpu_run_json_artifact

2026 is never loaded.
"""

import argparse
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import db_objects as db


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
MANIFEST_FILE = RESULTS / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"
DIRECT_FILE = RESULTS / "10_05_direct_runs.csv"

CANDIDATE_KEY = "CONT_H0_R4"
EXPECTED_TOPOLOGY = "H0"
EXPECTED_READOUT = "XZinj_plus_YX45"
EXPECTED_ALPHA = 0.75

# Rule-4 dynamics were originally selected at r=1, but Week-10 hardware testing
# found the best finite-shot operating point for this SAME dynamics at r=4.
OPERATING_R = 4
SHOTS_PER_SETTING = 512

# Default convergence grid.  These values are deliberately reported, not
# automatically combined into an arbitrary weighted score.
DEFAULT_CONTEXT_GRID = [
    1, 2, 4, 8, 12, 16, 24, 32, 48, 64,
    80, 96, 128, 160, 192, 256, 320, 365,
]

FEATURES = [
    "X0", "X1", "X2", "X3",
    "Z0", "Z1", "Z2", "Z3",
    "YX45",
]


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required project module not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module(
    COMMON_FILE,
    "qrc_11_1d_common",
)


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, float) and (
        math.isnan(value) or math.isinf(value)
    ):
        return None

    return value


def rmse(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (a - b) ** 2
            )
        )
    )


def mae(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)

    return float(
        np.mean(
            np.abs(a - b)
        )
    )


def bias(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)

    return float(
        np.mean(
            b - a
        )
    )


def load_candidate():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing final Week-10 manifest: {MANIFEST_FILE}"
        )

    df = pd.read_csv(MANIFEST_FILE)

    sub = df[
        df["candidate_key"].astype(str)
        == CANDIDATE_KEY
    ].copy()

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected exactly one {CANDIDATE_KEY} row; "
            f"found {len(sub)}."
        )

    row = sub.iloc[0]

    if str(row["protocol"]).upper() != "CONT":
        raise RuntimeError("Candidate #4 must be CONT.")

    if str(row["topology"]) != EXPECTED_TOPOLOGY:
        raise RuntimeError("Unexpected topology in manifest.")

    if str(row["test_readout"]) != EXPECTED_READOUT:
        raise RuntimeError("Unexpected readout in manifest.")

    if not np.isclose(
        float(row["alpha"]),
        EXPECTED_ALPHA,
    ):
        raise RuntimeError("Unexpected alpha in manifest.")

    raw_j = json.loads(
        str(row["J_json"])
    )

    J = {}

    for i, j in common.TOPOLOGY_EDGES[
        EXPECTED_TOPOLOGY
    ]:
        key = f"J{i}{j}"
        rev = f"J{j}{i}"

        if key in raw_j:
            value = raw_j[key]
        elif rev in raw_j:
            value = raw_j[rev]
        else:
            raise KeyError(
                f"Missing J for edge {(i, j)}."
            )

        J[(i, j)] = float(value)

    candidate = {
        "candidate_id": CANDIDATE_KEY,
        "protocol": "CONT",
        "topology": EXPECTED_TOPOLOGY,
        "r": OPERATING_R,
        "alpha": float(row["alpha"]),
        "dt": float(row["dt"]),
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
        "J": J,
    }

    return candidate, row.to_dict()


def feature_vector_from_reduced(
    rho_i,
    rho_m,
):
    full_row = common.reduced_feature_row(
        rho_i,
        rho_m,
    )

    return np.asarray(
        [
            float(full_row[f])
            for f in FEATURES
        ],
        dtype=float,
    )


def build_full_cont_bank(
    candidate,
    angles,
):
    """
    Exact Week-9/10 CONT recurrence.

    memory starts |00><00| once and is then propagated through all chronology.
    No measurement is fed back into the reservoir.
    """
    A_list = common.build_channels(
        candidate,
        angles,
    )

    qrc = common.qrc
    rho_m = qrc.memory_zero_density()

    rows = []

    for A_t in A_list:
        rho_i, rho_m = qrc.final_reduced_states(
            A_t,
            rho_m,
        )

        rows.append(
            feature_vector_from_reduced(
                rho_i,
                rho_m,
            )
        )

    return (
        np.asarray(rows, dtype=float),
        A_list,
    )


def replay_feature(
    A_list,
    endpoint,
    context_steps,
):
    """
    Independent K-step CONT replay ending at one endpoint.

    memory is reset to |00> at the start of every replay.
    There is no measurement until the final endpoint.
    """
    endpoint = int(endpoint)
    K = int(context_steps)

    start = endpoint - K + 1

    if start < 0:
        raise ValueError(
            f"Endpoint {endpoint} has only {endpoint + 1} "
            f"available steps, cannot replay K={K}."
        )

    qrc = common.qrc
    rho_m = qrc.memory_zero_density()

    rho_i = None

    for idx in range(
        start,
        endpoint + 1,
    ):
        rho_i, rho_m = qrc.final_reduced_states(
            A_list[idx],
            rho_m,
        )

    return feature_vector_from_reduced(
        rho_i,
        rho_m,
    )


def audit_week10_operating_point(
    selected_lambda,
    cv_rmse,
    ideal_rmse,
):
    if not DIRECT_FILE.exists():
        print(
            f"[warning] {DIRECT_FILE} not found; "
            "Week-10 operating-point audit skipped."
        )
        return None

    df = pd.read_csv(DIRECT_FILE)

    sub = df[
        (df["candidate_key"].astype(str) == CANDIDATE_KEY)
        &
        (df["r"].astype(int) == OPERATING_R)
    ].copy()

    if len(sub) != 1:
        print(
            "[warning] Could not uniquely resolve the Week-10 "
            "direct operating-point row; audit skipped."
        )
        return None

    row = sub.iloc[0]

    expected_lambda = float(
        row["selected_lambda"]
    )
    expected_cv = float(
        row["training_cv_rmse"]
    )
    expected_ideal = float(
        row["ideal_validation_rmse"]
    )

    if not np.isclose(
        selected_lambda,
        expected_lambda,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Week-10 lambda audit failed: "
            f"{selected_lambda} vs {expected_lambda}"
        )

    if abs(cv_rmse - expected_cv) > 1e-5:
        raise RuntimeError(
            "Week-10 CV audit failed: "
            f"{cv_rmse:.9f} vs {expected_cv:.9f}"
        )

    if abs(ideal_rmse - expected_ideal) > 1e-5:
        raise RuntimeError(
            "Week-10 ideal validation audit failed: "
            f"{ideal_rmse:.9f} vs {expected_ideal:.9f}"
        )

    return {
        "selected_lambda": expected_lambda,
        "training_cv_rmse": expected_cv,
        "ideal_validation_rmse": expected_ideal,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context-grid",
        nargs="+",
        type=int,
        default=DEFAULT_CONTEXT_GRID,
        help=(
            "Finite replay lengths K to test. "
            "No QPU job is submitted."
        ),
    )

    args = parser.parse_args()

    context_grid = sorted(
        {
            int(k)
            for k in args.context_grid
            if int(k) >= 1
        }
    )

    print("=" * 126)
    print("WEEK 11.1D — CANDIDATE #4 CONT FINITE-CONTEXT AUDIT")
    print("=" * 126)
    print("NO QPU JOB WILL BE SUBMITTED.")
    print("2026 = FROZEN / NOT LOADED")
    print()

    candidate, manifest_meta = (
        load_candidate()
    )

    print("Candidate #4:")
    print(f"  candidate={CANDIDATE_KEY}")
    print(f"  protocol=CONT")
    print(f"  topology={candidate['topology']}")
    print(f"  operating r={candidate['r']}")
    print(f"  alpha={candidate['alpha']}")
    print(f"  dt={candidate['dt']}")
    print(f"  hx={candidate['hx']:+.12f}")
    print(f"  hy={candidate['hy']:+.12f}")
    print(f"  readout={EXPECTED_READOUT}")
    print(f"  QPU shots/setting={SHOTS_PER_SETTING}")
    print(
        "  J="
        + json.dumps(
            {
                f"J{i}{j}": v
                for (i, j), v
                in candidate["J"].items()
            },
            sort_keys=True,
        )
    )
    print()

    # ------------------------------------------------------------------
    # Load ONLY train + validation.
    # ------------------------------------------------------------------
    work_tv, cols, y_all = (
        common.load_train_validation()
    )

    angles = common.make_angles(
        work_tv,
        cols,
        float(candidate["alpha"]),
    )

    full_bank, A_list = (
        build_full_cont_bank(
            candidate,
            angles,
        )
    )

    endpoints = np.arange(
        len(full_bank),
        dtype=int,
    )

    y_endpoint = y_all[endpoints]

    train_mask = (
        endpoints < common.N_TRAIN
    )
    val_mask = (
        endpoints >= common.N_TRAIN
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

    if len(val_endpoints) != common.N_VAL:
        raise RuntimeError(
            f"Expected {common.N_VAL} validation endpoints; "
            f"got {len(val_endpoints)}."
        )

    # ------------------------------------------------------------------
    # Freeze the full-CONT Ridge readout from 2022-2024 only.
    # ------------------------------------------------------------------
    selected_lambda, _, cv_summary = (
        common.select_lambda_training_only(
            X_train,
            y_train,
            n_splits=5,
        )
    )

    cv_rmse = float(
        cv_summary.iloc[0][
            "cv_rmse_mean"
        ]
    )

    model, scaler, keep = (
        common.fit_scaled_ridge(
            X_train,
            y_train,
            selected_lambda,
        )
    )

    pred_full = (
        common.predict_scaled_ridge(
            model,
            scaler,
            keep,
            X_val_full,
        )
    )

    full_rmse = rmse(
        y_val,
        pred_full,
    )
    full_mae = mae(
        y_val,
        pred_full,
    )
    full_bias = bias(
        y_val,
        pred_full,
    )

    audit = audit_week10_operating_point(
        selected_lambda=float(
            selected_lambda
        ),
        cv_rmse=cv_rmse,
        ideal_rmse=full_rmse,
    )

    print("Frozen full-CONT readout:")
    print(
        f"  lambda={float(selected_lambda)}"
    )
    print(
        f"  training-only CV RMSE="
        f"{cv_rmse:.6f}"
    )
    print(
        f"  ideal full-CONT 2025 RMSE="
        f"{full_rmse:.6f}"
    )
    print(
        f"  ideal full-CONT 2025 MAE ="
        f"{full_mae:.6f}"
    )
    print(
        f"  ideal full-CONT 2025 bias="
        f"{full_bias:+.6f}"
    )
    print()

    # ------------------------------------------------------------------
    # Context sweep.
    # ------------------------------------------------------------------
    rows = []

    print(
        f"{'K':>6} "
        f"{'Replay RMSE':>14} "
        f"{'Delta target':>14} "
        f"{'Pred RMSE vs full':>18} "
        f"{'Feature RMSE':>14} "
        f"{'Pred corr':>11}"
    )
    print("-" * 86)

    for K in context_grid:
        # Every validation endpoint has at least 1095 previous training steps,
        # so all practical K values are available.
        if K > int(
            np.min(val_endpoints)
        ) + 1:
            continue

        X_replay = np.vstack(
            [
                replay_feature(
                    A_list,
                    endpoint=e,
                    context_steps=K,
                )
                for e in val_endpoints
            ]
        )

        pred_replay = (
            common.predict_scaled_ridge(
                model,
                scaler,
                keep,
                X_replay,
            )
        )

        replay_rmse = rmse(
            y_val,
            pred_replay,
        )

        replay_mae = mae(
            y_val,
            pred_replay,
        )

        replay_bias = bias(
            y_val,
            pred_replay,
        )

        feature_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        X_replay
                        - X_val_full
                    )
                    ** 2
                )
            )
        )

        feature_mae = float(
            np.mean(
                np.abs(
                    X_replay
                    - X_val_full
                )
            )
        )

        pred_rmse_vs_full = rmse(
            pred_full,
            pred_replay,
        )

        pred_mae_vs_full = mae(
            pred_full,
            pred_replay,
        )

        pred_corr = float(
            np.corrcoef(
                pred_full,
                pred_replay,
            )[0, 1]
        )

        row = {
            "candidate_key": CANDIDATE_KEY,
            "protocol": "CONT",
            "topology": EXPECTED_TOPOLOGY,
            "operating_r": OPERATING_R,
            "readout": EXPECTED_READOUT,
            "alpha": float(
                candidate["alpha"]
            ),
            "context_steps": int(K),
            "full_cont_rmse": full_rmse,
            "replay_rmse": replay_rmse,
            "delta_replay_rmse_vs_full_cont": (
                replay_rmse
                - full_rmse
            ),
            "replay_mae": replay_mae,
            "replay_bias": replay_bias,
            "prediction_rmse_replay_vs_full": (
                pred_rmse_vs_full
            ),
            "prediction_mae_replay_vs_full": (
                pred_mae_vs_full
            ),
            "feature_rmse_replay_vs_full": (
                feature_rmse
            ),
            "feature_mae_replay_vs_full": (
                feature_mae
            ),
            "prediction_corr_replay_vs_full": (
                pred_corr
            ),
        }

        rows.append(row)

        print(
            f"{K:6d} "
            f"{replay_rmse:14.6f} "
            f"{replay_rmse-full_rmse:+14.6f} "
            f"{pred_rmse_vs_full:18.6f} "
            f"{feature_rmse:14.6f} "
            f"{pred_corr:11.6f}"
        )

    audit_df = pd.DataFrame(
        rows
    )

    csv_path = (
        RESULTS
        / "11_1D_candidate4_CONT_context_audit.csv"
    )

    audit_df.to_csv(
        csv_path,
        index=False,
    )

    summary = {
        "candidate_key": CANDIDATE_KEY,
        "protocol": "CONT",
        "topology": EXPECTED_TOPOLOGY,
        "operating_r": OPERATING_R,
        "shots_per_setting": (
            SHOTS_PER_SETTING
        ),
        "readout": EXPECTED_READOUT,
        "alpha": float(
            candidate["alpha"]
        ),
        "dt": float(
            candidate["dt"]
        ),
        "hx": float(
            candidate["hx"]
        ),
        "hy": float(
            candidate["hy"]
        ),
        "J": {
            f"J{i}{j}": float(v)
            for (i, j), v
            in candidate["J"].items()
        },
        "selected_lambda": float(
            selected_lambda
        ),
        "training_cv_rmse": (
            cv_rmse
        ),
        "full_cont_ideal_rmse": (
            full_rmse
        ),
        "full_cont_ideal_mae": (
            full_mae
        ),
        "full_cont_ideal_bias": (
            full_bias
        ),
        "week10_operating_point_audit": (
            audit
        ),
        "context_grid": context_grid,
        "2026_loaded": False,
    }

    json_path = (
        RESULTS
        / "11_1D_candidate4_CONT_context_audit.json"
    )

    json_path.write_text(
        json.dumps(
            json_safe(summary),
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Persist this audit in the same DB infrastructure.
    # ------------------------------------------------------------------
    db.create_all_tables()

    run_uuid = db.create_run(
        script_name=Path(__file__).name,
        run_status="CONTEXT_AUDIT_COMPLETE",
        started_at_utc=db.utc_now_naive(),
        completed_at_utc=db.utc_now_naive(),
        forecast_dataset_name=(
            "property_damage_next_day_v1"
        ),
        feature_set_name="F4",
        evaluation_split="validation_2025",
        candidate_key=CANDIDATE_KEY,
        selected_by_rules="Rule4",
        protocol="CONT",
        topology=EXPECTED_TOPOLOGY,
        window_size=None,
        trotter_r=OPERATING_R,
        readout_name=EXPECTED_READOUT,
        alpha=float(
            candidate["alpha"]
        ),
        dt=float(
            candidate["dt"]
        ),
        hx=float(
            candidate["hx"]
        ),
        hy=float(
            candidate["hy"]
        ),
        j_json=summary["J"],
        manifest_json=manifest_meta,
        ridge_lambda=float(
            selected_lambda
        ),
        training_cv_rmse=(
            cv_rmse
        ),
        ideal_validation_rmse_full_2025=(
            full_rmse
        ),
        ideal_validation_mae_full_2025=(
            full_mae
        ),
        ideal_validation_bias_full_2025=(
            full_bias
        ),
        backend_name=None,
        shots_per_setting=(
            SHOTS_PER_SETTING
        ),
        n_validation_endpoints=(
            int(len(val_endpoints))
        ),
        full_2025=True,
        notes=(
            "Candidate #4 CONT finite-context convergence audit. "
            "No IBM QPU job submitted. Context K must be selected "
            "before a physical-QPU replay experiment."
        ),
    )

    db.save_dataframe_artifact(
        run_uuid,
        csv_path.name,
        audit_df,
    )

    db.save_json_artifact(
        run_uuid,
        json_path.name,
        summary,
    )

    print()
    print(
        f"[database] Context-audit run UUID: "
        f"{run_uuid}"
    )
    print()
    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print()
    print(
        "STOP HERE. Do not submit candidate #4 to the QPU yet. "
        "Send me the context-sweep table so we can choose K "
        "from the actual convergence behavior rather than arbitrarily."
    )


if __name__ == "__main__":
    main()
