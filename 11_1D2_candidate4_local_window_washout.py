from __future__ import annotations

"""
WEEK 11.1D.2 — CANDIDATE #4 LOCAL-WINDOW WASHOUT VALIDATION
============================================================

WHY THIS STEP
-------------
The previous 11.1D.1 script measured forgetting from the start of the complete
2022-2025 chronology.  That is useful, but RWP restarts at MANY different
2025 endpoints.

For a publication-quality RWP definition, we also need to verify that the
candidate forgets its initialization when restarted K steps before EACH
validation endpoint.

For every 2025 endpoint e and every tested context length K:

    1. start five very different memory states at e-K+1,
    2. feed the SAME final K chronological inputs,
    3. perform NO intermediate measurement,
    4. calculate the endpoint pairwise trace distance

       D_e(K) = max_{A,B} 1/2 || rho_A(e;K) - rho_B(e;K) ||_1.

The same readout feature vectors are also compared.

This script tests the critical region around the global washout result:
    K = 96, 104, 112, 113, 117, 120, 128, 144, 160

The operational RWP length is NOT selected by an arbitrary weighted score.
We will choose the smallest K satisfying the desired physical threshold
across the required validation endpoints, then cross-check it against the
previous task-convergence audit.

NO IBM QPU JOB IS SUBMITTED.
2026 IS NEVER LOADED.
"""

import argparse
import importlib.util
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import db_objects as db


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

AUDIT_SCRIPT = HERE / "11_1D_candidate4_cont_context_audit.py"

CANDIDATE_KEY = "CONT_H0_R4"

DEFAULT_K = [
    96,
    104,
    112,
    113,
    117,
    120,
    128,
    144,
    160,
]

EPSILONS = [
    0.02,
    0.01,
    0.005,
]


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


audit = load_module(
    AUDIT_SCRIPT,
    "qrc_11_1d_local_audit",
)

common = audit.common
qrc = common.qrc

FEATURES = list(
    audit.FEATURES
)


# =============================================================================
# Initial memory states
# =============================================================================

ket0 = np.array(
    [1.0, 0.0],
    dtype=complex,
)

ket1 = np.array(
    [0.0, 1.0],
    dtype=complex,
)

ketp = (
    ket0 + ket1
) / np.sqrt(2.0)

ket00 = np.kron(
    ket0,
    ket0,
)

ket11 = np.kron(
    ket1,
    ket1,
)

ketpp = np.kron(
    ketp,
    ketp,
)

ket_phi = (
    ket00 + ket11
) / np.sqrt(2.0)


def pure_density(ket):
    return np.outer(
        ket,
        ket.conj(),
    )


INITIAL_STATES = {
    "00": pure_density(
        ket00
    ),
    "11": pure_density(
        ket11
    ),
    "++": pure_density(
        ketpp
    ),
    "BellPhi+": pure_density(
        ket_phi
    ),
    "I/4": (
        np.eye(
            4,
            dtype=complex,
        )
        / 4.0
    ),
}


# =============================================================================
# Helpers
# =============================================================================

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


def trace_distance(rho, sigma):
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


def max_pairwise_trace_distance(states):
    return float(
        max(
            trace_distance(
                states[a],
                states[b],
            )
            for a, b
            in combinations(
                states.keys(),
                2,
            )
        )
    )


def feature_vector(rho_i, rho_m):
    row = common.reduced_feature_row(
        rho_i,
        rho_m,
    )

    return np.asarray(
        [
            float(
                row[name]
            )
            for name
            in FEATURES
        ],
        dtype=float,
    )


def max_pairwise_feature_distance(features):
    return float(
        max(
            np.linalg.norm(
                features[a]
                - features[b]
            )
            for a, b
            in combinations(
                features.keys(),
                2,
            )
        )
    )


def evaluate_endpoint(
    A_list,
    endpoint,
    K,
):
    start = int(
        endpoint
        - K
        + 1
    )

    if start < 0:
        raise ValueError(
            f"Endpoint {endpoint} cannot support K={K}."
        )

    states = {
        name: rho.copy()
        for name, rho
        in INITIAL_STATES.items()
    }

    final_features = {}

    for idx in range(
        start,
        int(endpoint) + 1,
    ):
        next_states = {}
        step_features = {}

        for name, rho_before in states.items():
            rho_i, rho_m = (
                qrc.final_reduced_states(
                    A_list[idx],
                    rho_before,
                )
            )

            rho_m = (
                normalize_density(
                    rho_m
                )
            )

            next_states[
                name
            ] = rho_m

            step_features[
                name
            ] = feature_vector(
                rho_i,
                rho_m,
            )

        states = next_states
        final_features = step_features

    return {
        "trace_distance": (
            max_pairwise_trace_distance(
                states
            )
        ),
        "feature_distance": (
            max_pairwise_feature_distance(
                final_features
            )
        ),
    }


def summarize_group(values):
    arr = np.asarray(
        values,
        dtype=float,
    )

    return {
        "median": float(
            np.median(
                arr
            )
        ),
        "p95": float(
            np.quantile(
                arr,
                0.95,
            )
        ),
        "max": float(
            np.max(
                arr
            )
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=DEFAULT_K,
        help=(
            "Replay lengths to test. Default focuses "
            "on the region around the global Tw."
        ),
    )

    args = parser.parse_args()

    K_values = sorted(
        {
            int(k)
            for k in args.k
            if int(k) >= 1
        }
    )

    print(
        "=" * 126
    )
    print(
        "WEEK 11.1D.2 — CANDIDATE #4 LOCAL-WINDOW WASHOUT"
    )
    print(
        "=" * 126
    )
    print(
        "NO QPU JOB WILL BE SUBMITTED."
    )
    print(
        "2026 = FROZEN / NOT LOADED"
    )
    print()

    candidate, manifest_meta = (
        audit.load_candidate()
    )

    print(
        f"candidate={CANDIDATE_KEY}"
    )
    print(
        f"topology={candidate['topology']}"
    )
    print(
        f"r={candidate['r']}"
    )
    print(
        f"alpha={candidate['alpha']}"
    )
    print(
        f"tested K={K_values}"
    )
    print()

    # --------------------------------------------------------------
    # Train + validation only.
    # --------------------------------------------------------------
    work_tv, cols, _ = (
        common.load_train_validation()
    )

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

    validation_endpoints = np.arange(
        common.N_TRAIN,
        common.N_TRAIN
        + common.N_VAL,
        dtype=int,
    )

    rows = []

    for K in K_values:
        print(
            f"Evaluating K={K} ..."
        )

        for endpoint in validation_endpoints:
            result = evaluate_endpoint(
                A_list,
                int(endpoint),
                int(K),
            )

            rows.append(
                {
                    "candidate_key": (
                        CANDIDATE_KEY
                    ),
                    "endpoint": int(
                        endpoint
                    ),
                    "context_steps": int(
                        K
                    ),
                    "trace_distance": float(
                        result[
                            "trace_distance"
                        ]
                    ),
                    "feature_distance": float(
                        result[
                            "feature_distance"
                        ]
                    ),
                }
            )

    endpoint_df = pd.DataFrame(
        rows
    )

    endpoint_path = (
        RESULTS
        / "11_1D2_candidate4_local_washout_endpoints.csv"
    )

    endpoint_df.to_csv(
        endpoint_path,
        index=False,
    )

    summary_rows = []

    for K in K_values:
        sub = endpoint_df[
            endpoint_df[
                "context_steps"
            ]
            == K
        ]

        td = summarize_group(
            sub[
                "trace_distance"
            ]
        )

        fd = summarize_group(
            sub[
                "feature_distance"
            ]
        )

        row = {
            "context_steps": int(
                K
            ),
            "trace_median": td[
                "median"
            ],
            "trace_p95": td[
                "p95"
            ],
            "trace_max": td[
                "max"
            ],
            "feature_median": fd[
                "median"
            ],
            "feature_p95": fd[
                "p95"
            ],
            "feature_max": fd[
                "max"
            ],
        }

        for epsilon in EPSILONS:
            row[
                f"fraction_trace_le_{epsilon:g}"
            ] = float(
                np.mean(
                    sub[
                        "trace_distance"
                    ]
                    <= epsilon
                )
            )

            row[
                f"all_trace_le_{epsilon:g}"
            ] = bool(
                np.all(
                    sub[
                        "trace_distance"
                    ]
                    <= epsilon
                )
            )

            row[
                f"fraction_feature_le_{epsilon:g}"
            ] = float(
                np.mean(
                    sub[
                        "feature_distance"
                    ]
                    <= epsilon
                )
            )

            row[
                f"all_feature_le_{epsilon:g}"
            ] = bool(
                np.all(
                    sub[
                        "feature_distance"
                    ]
                    <= epsilon
                )
            )

        summary_rows.append(
            row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        RESULTS
        / "11_1D2_candidate4_local_washout_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    print()
    print(
        "LOCAL-WINDOW SUMMARY"
    )
    print(
        "-" * 126
    )

    print(
        summary_df[
            [
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
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Candidate operational choices, reported transparently.
    # --------------------------------------------------------------
    trace_candidates = summary_df[
        summary_df[
            "all_trace_le_0.01"
        ]
    ]

    feature_candidates = summary_df[
        summary_df[
            "all_feature_le_0.01"
        ]
    ]

    both_candidates = summary_df[
        summary_df[
            "all_trace_le_0.01"
        ]
        &
        summary_df[
            "all_feature_le_0.01"
        ]
    ]

    K_trace = (
        None
        if len(
            trace_candidates
        ) == 0
        else int(
            trace_candidates.iloc[0][
                "context_steps"
            ]
        )
    )

    K_feature = (
        None
        if len(
            feature_candidates
        ) == 0
        else int(
            feature_candidates.iloc[0][
                "context_steps"
            ]
        )
    )

    K_both = (
        None
        if len(
            both_candidates
        ) == 0
        else int(
            both_candidates.iloc[0][
                "context_steps"
            ]
        )
    )

    print()
    print(
        f"Smallest tested K with all 365 endpoints "
        f"trace-distance <= 0.01: {K_trace}"
    )
    print(
        f"Smallest tested K with all 365 endpoints "
        f"feature-distance <= 0.01: {K_feature}"
    )
    print(
        f"Smallest tested K satisfying BOTH: {K_both}"
    )

    summary_payload = {
        "candidate_key": (
            CANDIDATE_KEY
        ),
        "purpose": (
            "local-window washout validation "
            "for RWP operational replay length"
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
        "tested_context_steps": (
            K_values
        ),
        "epsilon_primary": (
            0.01
        ),
        "smallest_tested_K_all_trace_le_0_01": (
            K_trace
        ),
        "smallest_tested_K_all_feature_le_0_01": (
            K_feature
        ),
        "smallest_tested_K_both_le_0_01": (
            K_both
        ),
        "summary_rows": (
            summary_df.to_dict(
                orient="records"
            )
        ),
        "n_validation_endpoints": int(
            len(
                validation_endpoints
            )
        ),
        "initial_states": list(
            INITIAL_STATES.keys()
        ),
        "measurement_backaction_included": False,
        "2026_loaded": False,
    }

    json_path = (
        RESULTS
        / "11_1D2_candidate4_local_washout_summary.json"
    )

    json_path.write_text(
        json.dumps(
            db.json_safe(
                summary_payload
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Database persistence.
    #
    # evaluation_split deliberately stays <= VARCHAR(30).
    # --------------------------------------------------------------
    db.create_all_tables()

    run_uuid = db.create_run(
        script_name=(
            Path(
                __file__
            ).name
        ),
        run_status=(
            "LOCAL_WASHOUT_COMPLETE"
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
            CANDIDATE_KEY
        ),
        selected_by_rules=(
            "Rule4"
        ),
        protocol="CONT",
        topology=(
            candidate[
                "topology"
            ]
        ),
        window_size=(
            K_both
        ),
        trotter_r=int(
            candidate[
                "r"
            ]
        ),
        readout_name=(
            audit.EXPECTED_READOUT
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
            summary_payload[
                "J"
            ]
        ),
        manifest_json=(
            manifest_meta
        ),
        shots_per_setting=(
            audit.SHOTS_PER_SETTING
        ),
        n_validation_endpoints=int(
            len(
                validation_endpoints
            )
        ),
        full_2025=True,
        notes=(
            "Candidate #4 local-window trace-distance "
            "washout validation across all 365 validation "
            "endpoints. Five initial states, same K-step "
            "input window, no intermediate measurement. "
            "No QPU job submitted."
        ),
    )

    db.save_dataframe_artifact(
        run_uuid,
        endpoint_path.name,
        endpoint_df,
    )

    db.save_dataframe_artifact(
        run_uuid,
        summary_csv.name,
        summary_df,
    )

    db.save_json_artifact(
        run_uuid,
        json_path.name,
        summary_payload,
    )

    print()
    print(
        f"[database] Local-washout run UUID: "
        f"{run_uuid}"
    )
    print()
    print(
        "STOP HERE. Send me this summary. "
        "We will freeze the RWP replay length only after "
        "checking the all-endpoint trace and feature criteria."
    )


if __name__ == "__main__":
    main()
