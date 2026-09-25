from __future__ import annotations

"""
WEEK 11.1E.2 — CANDIDATE #5 LOCAL-WINDOW WASHOUT
==================================================

Purpose
-------
Validate the Candidate #5 global washout result locally at EVERY 2025
validation endpoint.

Global Candidate #5 result from 11.1E.1:
    trace-distance T_w(0.01)   = 64 steps
    readout-feature T_w(0.01) = 54 steps

The global result proves fading memory from the beginning of the chronological
stream, but it does not by itself prove that a restart window of length K is
sufficient for EVERY 2025 endpoint under the nonstationary input sequence.

For each validation endpoint e and each tested K:
    1. start K steps before e,
    2. initialize the two memory qubits in each of five different states,
    3. feed the SAME K chronological F4 inputs,
    4. apply NO intermediate measurement,
    5. compare the final memory states and frozen readout features.

We compute:

    D_trace(e,K) =
        max_{a,b} 1/2 || rho_e^(a)(K) - rho_e^(b)(K) ||_1

and the maximum Euclidean distance in the frozen Candidate #5 readout space:

    [X0,X1,X2,X3,Z0,Z1,Z2,Z3]

The operational RWP length is NOT frozen by this script alone.
We will freeze K only after inspecting:
    - global trace washout,
    - global readout-feature washout,
    - local all-365 endpoint trace criterion,
    - local all-365 endpoint feature criterion.

NO IBM CONNECTION.
NO QPU JOB.
2026 IS NEVER LOADED.
"""

import importlib.util
import json
from itertools import combinations
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

GLOBAL_SCRIPT = HERE / "11_1E1_candidate5_washout_trace_distance.py"
GLOBAL_SUMMARY = RESULTS / "11_1E1_candidate5_washout_summary.json"

OUT_ENDPOINTS = RESULTS / "11_1E2_candidate5_local_washout_endpoints.csv"
OUT_SUMMARY_CSV = RESULTS / "11_1E2_candidate5_local_washout_summary.csv"
OUT_SUMMARY_JSON = RESULTS / "11_1E2_candidate5_local_washout_summary.json"

PRIMARY_EPSILON = 0.01
SECONDARY_THRESHOLDS = [0.02, 0.01, 0.005]

# Use the grid proposed by the global Candidate #5 washout result.
DEFAULT_K_GRID = [40, 48, 56, 60, 63, 64, 65, 68, 72, 80, 96]


# =============================================================================
# Import exact Candidate #5 dynamics from the previous step
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
    "qrc_11_1e2_prev",
)

common = prev.common
qrc = prev.qrc

INITIAL_STATES = prev.INITIAL_STATES
FEATURES = list(prev.FEATURES)


# =============================================================================
# Helpers
# =============================================================================

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

    if abs(
        tr
    ) < 1e-15:
        raise RuntimeError(
            "Zero-trace memory state."
        )

    return rho / tr


def trace_distance(
    rho,
    sigma,
):
    delta = np.asarray(
        rho - sigma,
        dtype=complex,
    )

    delta = 0.5 * (
        delta
        + delta.conj().T
    )

    eigvals = np.linalg.eigvalsh(
        delta
    )

    return float(
        0.5
        * np.sum(
            np.abs(
                eigvals
            )
        )
    )


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


def final_endpoint_distances(
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
            f"endpoint={endpoint}, K={K} has start={start}"
        )

    states = {
        name: rho.copy()
        for name, rho
        in INITIAL_STATES.items()
    }

    final_features = None

    for idx in range(
        start,
        int(
            endpoint
        )
        + 1,
    ):
        A_t = A_list[
            idx
        ]

        next_states = {}
        endpoint_features = {}

        for name, rho_before in states.items():
            rho_i, rho_m = (
                qrc.final_reduced_states(
                    A_t,
                    rho_before,
                )
            )

            rho_m = normalize_density(
                rho_m
            )

            next_states[
                name
            ] = rho_m

            endpoint_features[
                name
            ] = feature_vector(
                rho_i,
                rho_m,
            )

        states = next_states
        final_features = endpoint_features

    trace_values = [
        trace_distance(
            states[
                a
            ],
            states[
                b
            ],
        )
        for a, b
        in combinations(
            states.keys(),
            2,
        )
    ]

    feature_values = [
        float(
            np.linalg.norm(
                final_features[
                    a
                ]
                - final_features[
                    b
                ]
            )
        )
        for a, b
        in combinations(
            final_features.keys(),
            2,
        )
    ]

    return {
        "trace_distance": float(
            max(
                trace_values
            )
        ),
        "feature_distance": float(
            max(
                feature_values
            )
        ),
    }


def boolean_all(series):
    return bool(
        np.all(
            np.asarray(
                series,
                dtype=bool,
            )
        )
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print(
        "=" * 126
    )
    print(
        "WEEK 11.1E.2 — CANDIDATE #5 LOCAL-WINDOW WASHOUT"
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

    global_summary = load_json(
        GLOBAL_SUMMARY
    )

    candidate, manifest_meta = (
        prev.load_candidate()
    )

    global_trace = int(
        global_summary[
            "global_trace_washout_steps"
        ]
    )

    global_feature = int(
        global_summary[
            "global_feature_washout_steps"
        ]
    )

    proposed_center = int(
        global_summary[
            "proposed_local_audit_center_K"
        ]
    )

    K_GRID = list(
        global_summary.get(
            "suggested_local_K_grid"
        )
        or DEFAULT_K_GRID
    )

    K_GRID = sorted(
        {
            int(k)
            for k
            in K_GRID
        }
    )

    print(
        "Candidate:"
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
        f"  readout={prev.EXPECTED_READOUT}"
    )
    print()

    print(
        "Global washout evidence:"
    )
    print(
        f"  trace Tw(0.01)={global_trace}"
    )
    print(
        f"  feature Tw(0.01)={global_feature}"
    )
    print(
        f"  proposed center K={proposed_center}"
    )
    print(
        f"  tested K={K_GRID}"
    )
    print()

    # ------------------------------------------------------------------
    # Train + validation only. 2026 is never loaded.
    # ------------------------------------------------------------------
    (
        work_tv,
        cols,
        _,
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

    A_list = common.build_channels(
        candidate,
        angles,
    )

    val_endpoints = np.arange(
        common.N_TRAIN,
        common.N_TRAIN
        + common.N_VAL,
        dtype=int,
    )

    if len(
        val_endpoints
    ) != 365:
        raise RuntimeError(
            f"Expected 365 validation endpoints, got {len(val_endpoints)}."
        )

    endpoint_rows = []

    for K in K_GRID:
        print(
            f"Evaluating K={K} ..."
        )

        for validation_row, endpoint in enumerate(
            val_endpoints
        ):
            dist = final_endpoint_distances(
                A_list,
                int(
                    endpoint
                ),
                int(
                    K
                ),
            )

            row = {
                "context_steps": int(
                    K
                ),
                "validation_row": int(
                    validation_row
                ),
                "endpoint": int(
                    endpoint
                ),
                "trace_distance": float(
                    dist[
                        "trace_distance"
                    ]
                ),
                "feature_distance": float(
                    dist[
                        "feature_distance"
                    ]
                ),
            }

            for eps in SECONDARY_THRESHOLDS:
                suffix = str(
                    eps
                ).replace(
                    ".",
                    "p",
                )

                row[
                    f"trace_le_{suffix}"
                ] = bool(
                    dist[
                        "trace_distance"
                    ]
                    <= eps
                )

                row[
                    f"feature_le_{suffix}"
                ] = bool(
                    dist[
                        "feature_distance"
                    ]
                    <= eps
                )

            endpoint_rows.append(
                row
            )

    endpoints_df = pd.DataFrame(
        endpoint_rows
    )

    endpoints_df.to_csv(
        OUT_ENDPOINTS,
        index=False,
    )

    summary_rows = []

    for K in K_GRID:
        sub = endpoints_df[
            endpoints_df[
                "context_steps"
            ]
            == int(
                K
            )
        ]

        row = {
            "context_steps": int(
                K
            ),
            "trace_median": float(
                sub[
                    "trace_distance"
                ].median()
            ),
            "trace_p95": float(
                sub[
                    "trace_distance"
                ].quantile(
                    0.95
                )
            ),
            "trace_max": float(
                sub[
                    "trace_distance"
                ].max()
            ),
            "feature_median": float(
                sub[
                    "feature_distance"
                ].median()
            ),
            "feature_p95": float(
                sub[
                    "feature_distance"
                ].quantile(
                    0.95
                )
            ),
            "feature_max": float(
                sub[
                    "feature_distance"
                ].max()
            ),
        }

        for eps in SECONDARY_THRESHOLDS:
            suffix = str(
                eps
            ).replace(
                ".",
                "p",
            )

            trace_pass = (
                sub[
                    "trace_distance"
                ]
                <= eps
            )

            feature_pass = (
                sub[
                    "feature_distance"
                ]
                <= eps
            )

            row[
                f"frac_trace_le_{eps}"
            ] = float(
                np.mean(
                    trace_pass
                )
            )

            row[
                f"frac_feature_le_{eps}"
            ] = float(
                np.mean(
                    feature_pass
                )
            )

            row[
                f"all_trace_le_{eps}"
            ] = bool(
                np.all(
                    trace_pass
                )
            )

            row[
                f"all_feature_le_{eps}"
            ] = bool(
                np.all(
                    feature_pass
                )
            )

        summary_rows.append(
            row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUT_SUMMARY_CSV,
        index=False,
    )

    print()
    print(
        "LOCAL-WINDOW SUMMARY"
    )
    print(
        "-" * 126
    )

    show_cols = [
        "context_steps",
        "trace_median",
        "trace_p95",
        "trace_max",
        "feature_median",
        "feature_p95",
        "feature_max",
        "all_trace_le_0.01",
        "all_feature_le_0.01",
    ]

    print(
        summary_df[
            show_cols
        ].to_string(
            index=False
        )
    )

    trace_pass_rows = summary_df[
        summary_df[
            "all_trace_le_0.01"
        ]
        == True
    ]

    feature_pass_rows = summary_df[
        summary_df[
            "all_feature_le_0.01"
        ]
        == True
    ]

    both_pass_rows = summary_df[
        (
            summary_df[
                "all_trace_le_0.01"
            ]
            == True
        )
        &
        (
            summary_df[
                "all_feature_le_0.01"
            ]
            == True
        )
    ]

    smallest_trace = (
        None
        if len(
            trace_pass_rows
        ) == 0
        else int(
            trace_pass_rows[
                "context_steps"
            ].min()
        )
    )

    smallest_feature = (
        None
        if len(
            feature_pass_rows
        ) == 0
        else int(
            feature_pass_rows[
                "context_steps"
            ].min()
        )
    )

    smallest_both = (
        None
        if len(
            both_pass_rows
        ) == 0
        else int(
            both_pass_rows[
                "context_steps"
            ].min()
        )
    )

    print()
    print(
        "Smallest tested K with all 365 endpoints "
        f"trace-distance <= 0.01: {smallest_trace}"
    )
    print(
        "Smallest tested K with all 365 endpoints "
        f"feature-distance <= 0.01: {smallest_feature}"
    )
    print(
        "Smallest tested K satisfying BOTH: "
        f"{smallest_both}"
    )

    # ------------------------------------------------------------------
    # Do not silently freeze K here.
    # ------------------------------------------------------------------
    proposed_operational_K = max(
        global_trace,
        global_feature,
    )

    exact_global_K_row = summary_df[
        summary_df[
            "context_steps"
        ]
        == proposed_operational_K
    ]

    exact_global_K_local_pass = None

    if len(
        exact_global_K_row
    ) == 1:
        exact_global_K_local_pass = bool(
            exact_global_K_row.iloc[
                0
            ][
                "all_trace_le_0.01"
            ]
            and exact_global_K_row.iloc[
                0
            ][
                "all_feature_le_0.01"
            ]
        )

    print()
    print(
        "GLOBAL/LOCAL CONSISTENCY CHECK"
    )
    print(
        "-" * 126
    )
    print(
        f"Global candidate K=max(trace,feature)="
        f"{proposed_operational_K}"
    )
    print(
        f"Does that exact K pass both local all-endpoint "
        f"criteria? {exact_global_K_local_pass}"
    )
    print()

    summary_json = {
        "candidate_key": (
            candidate[
                "candidate_id"
            ]
        ),
        "protocol": (
            candidate[
                "protocol"
            ]
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
        "readout": (
            prev.EXPECTED_READOUT
        ),
        "features": (
            FEATURES
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
        "global_trace_washout_steps_epsilon_0p01": (
            global_trace
        ),
        "global_feature_washout_steps_epsilon_0p01": (
            global_feature
        ),
        "global_candidate_K": (
            proposed_operational_K
        ),
        "tested_K": (
            K_GRID
        ),
        "smallest_tested_all_trace_le_0p01": (
            smallest_trace
        ),
        "smallest_tested_all_feature_le_0p01": (
            smallest_feature
        ),
        "smallest_tested_all_both_le_0p01": (
            smallest_both
        ),
        "global_candidate_K_passes_local_both": (
            exact_global_K_local_pass
        ),
        "operational_K_frozen": False,
        "measurement_backaction_included": False,
        "n_validation_endpoints": int(
            len(
                val_endpoints
            )
        ),
        "2026_loaded": False,
    }

    OUT_SUMMARY_JSON.write_text(
        json.dumps(
            db.json_safe(
                summary_json
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
            "LOCAL_WASHOUT_AUDIT_COMPLETE"
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
            candidate[
                "candidate_id"
            ]
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
            candidate[
                "protocol"
            ]
        ),
        topology=(
            candidate[
                "topology"
            ]
        ),
        window_size=(
            proposed_operational_K
        ),
        trotter_r=int(
            candidate[
                "r"
            ]
        ),
        readout_name=(
            prev.EXPECTED_READOUT
        ),
        primitive_name=(
            "ideal_simulation"
        ),
        measurement_method=(
            "local_365_endpoint_washout"
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
            summary_json[
                "J"
            ]
        ),
        manifest_json=(
            manifest_meta
        ),
        shots_per_setting=(
            prev.PLANNED_SHOTS
        ),
        n_validation_endpoints=(
            365
        ),
        full_2025=True,
        notes=(
            "Candidate #5 local 365-endpoint washout audit. "
            "Five initial memory states restarted K steps before each "
            "validation endpoint. No intermediate measurement. "
            "Operational K is intentionally not frozen until results "
            "are inspected."
        ),
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_ENDPOINTS.name,
        endpoints_df,
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_SUMMARY_CSV.name,
        summary_df,
    )

    db.save_json_artifact(
        run_uuid,
        OUT_SUMMARY_JSON.name,
        summary_json,
    )

    print()
    print(
        f"[database] Candidate #5 local-washout run UUID: "
        f"{run_uuid}"
    )

    print()
    print(
        "Saved:"
    )
    print(
        f"  {OUT_ENDPOINTS}"
    )
    print(
        f"  {OUT_SUMMARY_CSV}"
    )
    print(
        f"  {OUT_SUMMARY_JSON}"
    )
    print()

    print(
        "STOP HERE."
    )
    print(
        "Send me the LOCAL-WINDOW SUMMARY and GLOBAL/LOCAL "
        "CONSISTENCY CHECK."
    )
    print(
        "Only then will we freeze Candidate #5 K_RWP and build "
        "the exact ideal RWP reference."
    )


if __name__ == "__main__":
    main()
