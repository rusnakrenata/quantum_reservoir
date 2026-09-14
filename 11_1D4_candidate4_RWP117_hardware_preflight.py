from __future__ import annotations

"""
WEEK 11.1D.4 — CANDIDATE #4 RWP(117) REAL-HARDWARE PREFLIGHT
================================================================

PURPOSE
-------
Compile the ACTUAL washout-derived RWP(K=117) circuits against a freshly
reselected IBM hardware layout and determine whether the protocol is physically
reasonable BEFORE spending any QPU time.

This script DOES:
    - load the already-frozen Candidate #4 RWP package,
    - verify K_RWP = 117 and all frozen candidate parameters,
    - refresh current IBM calibration/connectivity,
    - reselect a fresh native H0 six-qubit embedding,
    - build the REAL 117-step RWP circuits,
    - compile representative 2025 endpoints (or all 365 with --all-compile),
    - measure actual compiled CZ counts, depth, reset count and scheduled duration,
    - calculate duration/min(T1) and duration/min(T2),
    - project the full 365-day, two-setting hardware workload,
    - save CSV/JSON + database records.

This script DOES NOT:
    - submit Sampler/Estimator jobs,
    - re-train or re-select the Ridge readout,
    - change the washout length,
    - load 2026.

FROZEN MODEL
------------
Original model:
    CONT_H0_R4
    full CONT training/readout

Hardware surrogate:
    washout-derived RWP
    K = 117
    endpoint-only measurement

Readout:
    XZinj_plus_YX45

Features:
    X0, X1, X2, X3,
    Z0, Z1, Z2, Z3,
    Y4 X5

Grouped measurement settings:
    A = XXXXYX
    B = ZZZZZZ

No QPU job is submitted by this script.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)

from ibm_account import get_service
import db_objects as db


# =============================================================================
# Paths / frozen constants
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

AUDIT_SCRIPT = HERE / "11_1D_candidate4_cont_context_audit.py"
SELECTOR_FILE = HERE / "11_0A_live_embedding_reselection.py"

FROZEN_PACKAGE = RESULTS / "11_1D3_candidate4_RWP117_frozen_package.json"

CANDIDATE_KEY = "CONT_H0_R4"
PREFLIGHT_KEY = "CONT_H0_R4_RWP117_PREFLIGHT"

EXPECTED_TOPOLOGY = "H0"
EXPECTED_R = 4
EXPECTED_ALPHA = 0.75
EXPECTED_READOUT = "XZinj_plus_YX45"
EXPECTED_K = 117

DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_SHOTS = 512

OPT_LEVEL = 1
SEED_TRANSPILE = 42
SHORTLIST = 120

OUT_RESOURCES = RESULTS / "11_1D4_candidate4_RWP117_preflight_resources.csv"
OUT_ENDPOINT_RESOURCES = RESULTS / "11_1D4_candidate4_RWP117_preflight_endpoint_resources.csv"
OUT_SUMMARY = RESULTS / "11_1D4_candidate4_RWP117_preflight_summary.json"


# =============================================================================
# Generic helpers
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

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
    "qrc_11_1d4_audit",
)

common = audit.common

selector = load_module(
    SELECTOR_FILE,
    "qrc_11_1d4_selector",
)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen artifact not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def safe_float(value):
    try:
        value = float(value)
    except Exception:
        return None

    if not np.isfinite(value):
        return None

    return value


def parse_layout(value):
    if isinstance(value, list):
        return [
            int(x)
            for x in value
        ]

    if isinstance(value, tuple):
        return [
            int(x)
            for x in value
        ]

    if isinstance(value, str):
        return [
            int(x)
            for x in json.loads(
                value
            )
        ]

    raise TypeError(
        f"Unsupported layout representation: {type(value)}"
    )


def get_service_usage(service):
    try:
        return db.json_safe(
            service.usage()
        )
    except Exception as exc:
        return {
            "error": str(exc)
        }


# =============================================================================
# Frozen package audit
# =============================================================================

def audit_frozen_package(package, candidate):
    errors = []

    if package.get(
        "candidate_key"
    ) != CANDIDATE_KEY:
        errors.append(
            "candidate_key mismatch"
        )

    if package.get(
        "hardware_surrogate_protocol"
    ) != "washout_derived_RWP":
        errors.append(
            "hardware surrogate protocol mismatch"
        )

    if int(
        package.get(
            "operational_K",
            -1,
        )
    ) != EXPECTED_K:
        errors.append(
            "operational_K mismatch"
        )

    if package.get(
        "topology"
    ) != EXPECTED_TOPOLOGY:
        errors.append(
            "topology mismatch"
        )

    if int(
        package.get(
            "r",
            -1,
        )
    ) != EXPECTED_R:
        errors.append(
            "r mismatch"
        )

    if not np.isclose(
        float(
            package.get(
                "alpha",
                np.nan,
            )
        ),
        EXPECTED_ALPHA,
    ):
        errors.append(
            "alpha mismatch"
        )

    if package.get(
        "readout"
    ) != EXPECTED_READOUT:
        errors.append(
            "readout mismatch"
        )

    if candidate[
        "topology"
    ] != EXPECTED_TOPOLOGY:
        errors.append(
            "candidate topology mismatch"
        )

    if int(
        candidate[
            "r"
        ]
    ) != EXPECTED_R:
        errors.append(
            "candidate r mismatch"
        )

    if not np.isclose(
        float(
            candidate[
                "alpha"
            ]
        ),
        EXPECTED_ALPHA,
    ):
        errors.append(
            "candidate alpha mismatch"
        )

    if errors:
        raise RuntimeError(
            "Frozen-package audit failed: "
            + "; ".join(
                errors
            )
        )


# =============================================================================
# Actual RWP(K=117) circuit
# =============================================================================

def append_rwp_step(
    qc,
    candidate,
    angle_row,
    first_step,
):
    """
    One physical RWP input step.

    q0..q3 = injection qubits.
    q4,q5  = memory qubits.

    Between chronological inputs:
        q0..q3 are reset/reprepared,
        q4,q5 retain the reservoir memory.

    No endpoint readout occurs until all K=117 inputs have been replayed.
    """
    if not first_step:
        for q in range(4):
            qc.reset(
                q
            )

    for q in range(4):
        qc.ry(
            float(
                angle_row[q]
            ),
            q,
        )

    r = int(
        candidate[
            "r"
        ]
    )

    dt = float(
        candidate[
            "dt"
        ]
    )

    hx = float(
        candidate[
            "hx"
        ]
    )

    hy = float(
        candidate[
            "hy"
        ]
    )

    J = candidate[
        "J"
    ]

    for _ in range(r):
        # ZZ interactions.
        for i, j in common.TOPOLOGY_EDGES[
            candidate[
                "topology"
            ]
        ]:
            qc.rzz(
                2.0
                * float(
                    J[
                        (i, j)
                    ]
                )
                * dt
                / r,
                i,
                j,
            )

        # X field on all six qubits.
        theta_x = (
            2.0
            * hx
            * dt
            / r
        )

        for q in range(6):
            qc.rx(
                theta_x,
                q,
            )

        # Memory-only Y field.
        if not np.isclose(
            hy,
            0.0,
        ):
            theta_y = (
                2.0
                * hy
                * dt
                / r
            )

            qc.ry(
                theta_y,
                4,
            )
            qc.ry(
                theta_y,
                5,
            )


def readout_settings():
    """
    Exact grouped measurement settings for XZinj_plus_YX45.

    Setting A = XXXXYX
        X0, X1, X2, X3
        Y4 X5

    Setting B = ZZZZZZ
        Z0, Z1, Z2, Z3

    q4/q5 are also physically measured in setting B only to keep a uniform
    six-bit register; they are not part of the frozen Ridge feature vector
    in that setting.
    """
    return [
        (
            "XXXXYX",
            {
                0: "X",
                1: "X",
                2: "X",
                3: "X",
                4: "Y",
                5: "X",
            },
            [
                0,
                1,
                2,
                3,
                4,
                5,
            ],
        ),
        (
            "ZZZZZZ",
            {
                0: "Z",
                1: "Z",
                2: "Z",
                3: "Z",
                4: "Z",
                5: "Z",
            },
            [
                0,
                1,
                2,
                3,
                4,
                5,
            ],
        ),
    ]


def build_measurement_circuit(
    candidate,
    all_angles,
    endpoint,
    setting,
):
    label, basis, measured = setting

    qreg = QuantumRegister(
        6,
        "q",
    )

    creg = ClassicalRegister(
        len(
            measured
        ),
        "m",
    )

    qc = QuantumCircuit(
        qreg,
        creg,
        name=(
            f"H0_RWP117_e{int(endpoint)}_{label}"
        ),
    )

    start = (
        int(
            endpoint
        )
        - EXPECTED_K
        + 1
    )

    if start < 0:
        raise ValueError(
            f"Endpoint {endpoint} does not have "
            f"enough K={EXPECTED_K} history."
        )

    for k, idx in enumerate(
        range(
            start,
            int(
                endpoint
            )
            + 1,
        )
    ):
        append_rwp_step(
            qc,
            candidate,
            all_angles[
                idx
            ],
            first_step=(
                k == 0
            ),
        )

    # Endpoint-only basis rotation.
    for q, axis in basis.items():
        if axis == "X":
            qc.h(
                q
            )
        elif axis == "Y":
            qc.sdg(
                q
            )
            qc.h(
                q
            )
        elif axis == "Z":
            pass
        else:
            raise ValueError(
                axis
            )

    for c, q in enumerate(
        measured
    ):
        qc.measure(
            q,
            c,
        )

    return qc


# =============================================================================
# Compilation / resource extraction
# =============================================================================

def compile_one(
    logical,
    backend,
    layout,
):
    compiled = transpile(
        logical,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
        scheduling_method="alap",
    )

    backend.check_faulty(
        compiled
    )

    return compiled


def circuit_resource_row(
    compiled,
    backend,
    endpoint,
    setting,
    min_t1_us,
    min_t2_us,
):
    ops = {
        str(k): int(v)
        for k, v
        in compiled.count_ops().items()
    }

    duration_s = float(
        compiled.estimate_duration(
            backend.target,
            unit="s",
        )
    )

    duration_us = (
        duration_s
        * 1e6
    )

    r_t1 = (
        None
        if not min_t1_us
        else duration_us
        / float(
            min_t1_us
        )
    )

    r_t2 = (
        None
        if not min_t2_us
        else duration_us
        / float(
            min_t2_us
        )
    )

    return {
        "endpoint": int(
            endpoint
        ),
        "setting": str(
            setting
        ),
        "depth": int(
            compiled.depth()
        ),
        "size": int(
            compiled.size()
        ),
        "n_cz": int(
            ops.get(
                "cz",
                0,
            )
        ),
        "n_swap": int(
            ops.get(
                "swap",
                0,
            )
        ),
        "n_reset": int(
            ops.get(
                "reset",
                0,
            )
        ),
        "n_measure": int(
            ops.get(
                "measure",
                0,
            )
        ),
        "duration_us": float(
            duration_us
        ),
        "duration_over_min_t1": (
            None
            if r_t1 is None
            else float(
                r_t1
            )
        ),
        "duration_over_min_t2": (
            None
            if r_t2 is None
            else float(
                r_t2
            )
        ),
        "operations": json.dumps(
            ops,
            sort_keys=True,
        ),
    }


def endpoint_summary(
    resources_df,
):
    grouped = (
        resources_df
        .groupby(
            "endpoint",
            as_index=False,
        )
        .agg(
            cz_feature_vector=(
                "n_cz",
                "sum",
            ),
            swap_feature_vector=(
                "n_swap",
                "sum",
            ),
            reset_feature_vector=(
                "n_reset",
                "sum",
            ),
            max_setting_depth=(
                "depth",
                "max",
            ),
            feature_duration_us=(
                "duration_us",
                "sum",
            ),
            max_setting_duration_us=(
                "duration_us",
                "max",
            ),
            max_duration_over_min_t1=(
                "duration_over_min_t1",
                "max",
            ),
            max_duration_over_min_t2=(
                "duration_over_min_t2",
                "max",
            ),
        )
    )

    return grouped


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compile actual Candidate #4 RWP(117) "
            "circuits without QPU submission."
        )
    )

    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
    )

    parser.add_argument(
        "--all-compile",
        action="store_true",
        help=(
            "Compile all 365 endpoints x 2 settings. "
            "Default compiles representative first/middle/last "
            "validation endpoints because all have the same "
            "RWP length and gate skeleton."
        ),
    )

    args = parser.parse_args()

    print(
        "=" * 126
    )
    print(
        "WEEK 11.1D.4 — CANDIDATE #4 RWP(117) HARDWARE PREFLIGHT"
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

    package = load_json(
        FROZEN_PACKAGE
    )

    candidate, manifest_meta = (
        audit.load_candidate()
    )

    audit_frozen_package(
        package,
        candidate,
    )

    print(
        "Frozen reference:"
    )
    print(
        f"  original model={CANDIDATE_KEY} / full CONT"
    )
    print(
        f"  hardware surrogate=RWP"
    )
    print(
        f"  K={EXPECTED_K}"
    )
    print(
        f"  topology={EXPECTED_TOPOLOGY}"
    )
    print(
        f"  r={EXPECTED_R}"
    )
    print(
        f"  readout={EXPECTED_READOUT}"
    )
    print(
        f"  ideal RWP117 2025 RMSE="
        f"{package['RWP_K117_ideal_2025']['rmse']:.6f}"
    )
    print(
        f"  full CONT ideal 2025 RMSE="
        f"{package['full_CONT_ideal_2025']['rmse']:.6f}"
    )
    print()

    # ------------------------------------------------------------------
    # Load ONLY train + validation angles.
    # ------------------------------------------------------------------
    work_tv, cols, _ = (
        common.load_train_validation()
    )

    all_angles = common.make_angles(
        work_tv,
        cols,
        float(
            candidate[
                "alpha"
            ]
        ),
    )

    val_endpoints = np.arange(
        common.N_TRAIN,
        common.N_TRAIN
        + common.N_VAL,
        dtype=int,
    )

    if args.all_compile:
        compile_endpoints = (
            val_endpoints.copy()
        )
    else:
        compile_endpoints = np.unique(
            np.asarray(
                [
                    val_endpoints[0],
                    val_endpoints[
                        len(
                            val_endpoints
                        )
                        // 2
                    ],
                    val_endpoints[-1],
                ],
                dtype=int,
            )
        )

    print(
        "Compilation mode:"
    )
    print(
        f"  representative endpoints="
        f"{compile_endpoints.tolist()}"
        if not args.all_compile
        else
        f"  ALL {len(compile_endpoints)} validation endpoints"
    )
    print()

    # ------------------------------------------------------------------
    # Fresh IBM calibration + native H0 layout selection.
    # ------------------------------------------------------------------
    service = get_service()

    usage_before = (
        get_service_usage(
            service
        )
    )

    print(
        "=" * 126
    )
    print(
        f"FRESH {args.backend} H0 HARDWARE RESELECTION"
    )
    print(
        "=" * 126
    )

    selection = (
        selector.fresh_hardware_reselection(
            service,
            candidate,
            backend_names=[
                args.backend
            ],
            shortlist=SHORTLIST,
            write_prefix=(
                f"11_1D4_{CANDIDATE_KEY}_{args.backend}"
            ),
            verbose=True,
        )
    )

    selected = (
        selection[
            "selected"
        ]
    )

    layout = parse_layout(
        selected[
            "layout"
        ]
    )

    min_t1_us = safe_float(
        selected.get(
            "min_t1_us"
        )
    )

    min_t2_us = safe_float(
        selected.get(
            "min_t2_us"
        )
    )

    print()
    print(
        "Fresh physical choice:"
    )
    print(
        f"  backend={args.backend}"
    )
    print(
        f"  layout={layout}"
    )
    print(
        f"  max CZ error="
        f"{float(selected['compiled_2q_error_max_percent']):.6f}%"
    )
    print(
        f"  max readout error="
        f"{float(selected['max_readout_error_percent']):.6f}%"
    )
    print(
        f"  max 1Q error="
        f"{float(selected['compiled_1q_error_max_percent']):.6f}%"
    )
    print(
        f"  min T1="
        f"{min_t1_us:.3f} us"
    )
    print(
        f"  min T2="
        f"{min_t2_us:.3f} us"
    )
    print()

    # Refresh backend object after selection.
    backend, _ = (
        selector.refresh_backend_fully(
            service,
            args.backend,
        )
    )

    # ------------------------------------------------------------------
    # Compile ACTUAL K=117 circuits.
    # ------------------------------------------------------------------
    settings = (
        readout_settings()
    )

    resource_rows = []

    print(
        "=" * 126
    )
    print(
        "COMPILING ACTUAL RWP(117) CIRCUITS"
    )
    print(
        "=" * 126
    )

    for endpoint in compile_endpoints:
        for setting in settings:
            label = (
                setting[
                    0
                ]
            )

            print(
                f"  compiling endpoint={int(endpoint)}, "
                f"setting={label} ..."
            )

            logical = (
                build_measurement_circuit(
                    candidate,
                    all_angles,
                    int(
                        endpoint
                    ),
                    setting,
                )
            )

            compiled = (
                compile_one(
                    logical,
                    backend,
                    layout,
                )
            )

            row = circuit_resource_row(
                compiled,
                backend,
                int(
                    endpoint
                ),
                label,
                min_t1_us,
                min_t2_us,
            )

            resource_rows.append(
                row
            )

    resources_df = pd.DataFrame(
        resource_rows
    )

    endpoint_df = endpoint_summary(
        resources_df
    )

    resources_df.to_csv(
        OUT_RESOURCES,
        index=False,
    )

    endpoint_df.to_csv(
        OUT_ENDPOINT_RESOURCES,
        index=False,
    )

    # ------------------------------------------------------------------
    # Actual compiled sample summary.
    # ------------------------------------------------------------------
    median_cz = float(
        endpoint_df[
            "cz_feature_vector"
        ].median()
    )

    median_depth = float(
        endpoint_df[
            "max_setting_depth"
        ].median()
    )

    median_feature_duration_us = float(
        endpoint_df[
            "feature_duration_us"
        ].median()
    )

    median_max_setting_duration_us = float(
        endpoint_df[
            "max_setting_duration_us"
        ].median()
    )

    median_resets = float(
        endpoint_df[
            "reset_feature_vector"
        ].median()
    )

    median_r_t1 = float(
        endpoint_df[
            "max_duration_over_min_t1"
        ].median()
    )

    median_r_t2 = float(
        endpoint_df[
            "max_duration_over_min_t2"
        ].median()
    )

    max_r_t1 = float(
        endpoint_df[
            "max_duration_over_min_t1"
        ].max()
    )

    max_r_t2 = float(
        endpoint_df[
            "max_duration_over_min_t2"
        ].max()
    )

    max_swaps = int(
        endpoint_df[
            "swap_feature_vector"
        ].max()
    )

    if max_swaps != 0:
        raise RuntimeError(
            "Actual RWP117 compilation introduced SWAP gates. "
            "Do not proceed to QPU."
        )

    # Project full 365 x 2-setting workload from ACTUAL long-circuit sample.
    projected_scheduled_times_shots_s = (
        median_feature_duration_us
        * common.N_VAL
        * int(
            args.shots
        )
        / 1e6
    )

    print()
    print(
        "=" * 126
    )
    print(
        "ACTUAL LONG-CIRCUIT PREFLIGHT"
    )
    print(
        "=" * 126
    )
    print(
        f"Compiled validation endpoints:     "
        f"{len(compile_endpoints)}"
    )
    print(
        f"Planned full-2025 endpoints:       "
        f"{common.N_VAL}"
    )
    print(
        f"Planned QPU circuits:              "
        f"{common.N_VAL * len(settings)}"
    )
    print(
        f"Shots/setting:                     "
        f"{args.shots}"
    )
    print(
        f"Median CZ/feature vector:          "
        f"{median_cz:.0f}"
    )
    print(
        f"Median reset/feature vector:       "
        f"{median_resets:.0f}"
    )
    print(
        f"Median max setting depth:          "
        f"{median_depth:.0f}"
    )
    print(
        f"Median feature duration:           "
        f"{median_feature_duration_us:.3f} us"
    )
    print(
        f"Median max-setting duration:       "
        f"{median_max_setting_duration_us:.3f} us"
    )
    print(
        f"Median max duration/min T1:        "
        f"{median_r_t1:.3f}"
    )
    print(
        f"Median max duration/min T2:        "
        f"{median_r_t2:.3f}"
    )
    print(
        f"Worst sampled duration/min T1:     "
        f"{max_r_t1:.3f}"
    )
    print(
        f"Worst sampled duration/min T2:     "
        f"{max_r_t2:.3f}"
    )
    print(
        f"SWAPs in sampled feature vectors:  "
        f"{max_swaps}"
    )
    print(
        f"Projected scheduled-duration*shots "
        f"for all 365 days: "
        f"{projected_scheduled_times_shots_s:.3f} s"
    )

    print()
    print(
        "PHYSICAL INTERPRETATION"
    )
    print(
        "-" * 126
    )

    if median_r_t2 < 0.25:
        print(
            "  Memory-circuit duration is comfortably below min T2."
        )
    elif median_r_t2 < 1.0:
        print(
            "  Memory-circuit duration is a substantial fraction of min T2."
        )
    else:
        print(
            "  WARNING: the RWP(117) circuit lasts longer than min T2."
        )
        print(
            "  The memory qubits must remain coherent across the entire "
            "117-step replay, so decoherence is expected to be a dominant "
            "physical limitation even though the ideal RWP approximation "
            "is excellent."
        )

    if median_r_t1 >= 1.0:
        print(
            "  WARNING: the circuit also lasts longer than min T1."
        )

    # ------------------------------------------------------------------
    # Summary / database.
    # ------------------------------------------------------------------
    summary = {
        "candidate_key": (
            CANDIDATE_KEY
        ),
        "preflight_key": (
            PREFLIGHT_KEY
        ),
        "original_model_protocol": (
            "CONT"
        ),
        "hardware_surrogate_protocol": (
            "washout_derived_RWP"
        ),
        "K": (
            EXPECTED_K
        ),
        "backend": (
            args.backend
        ),
        "layout": (
            layout
        ),
        "shots_per_setting": int(
            args.shots
        ),
        "readout": (
            EXPECTED_READOUT
        ),
        "measurement_settings": [
            x[0]
            for x
            in settings
        ],
        "compiled_endpoint_mode": (
            "all_365"
            if args.all_compile
            else "representative_first_middle_last"
        ),
        "compiled_endpoints": [
            int(x)
            for x in compile_endpoints
        ],
        "planned_validation_endpoints": int(
            common.N_VAL
        ),
        "planned_qpu_circuits": int(
            common.N_VAL
            * len(
                settings
            )
        ),
        "frozen_reference": {
            "full_CONT_ideal_2025": package[
                "full_CONT_ideal_2025"
            ],
            "RWP_K117_ideal_2025": package[
                "RWP_K117_ideal_2025"
            ],
            "protocol_approximation": package[
                "protocol_approximation_RWP_vs_CONT"
            ],
        },
        "fresh_hardware_selection": (
            db.json_safe(
                selected
            )
        ),
        "actual_compiled_sample": {
            "median_cz_per_feature_vector": (
                median_cz
            ),
            "median_reset_per_feature_vector": (
                median_resets
            ),
            "median_max_setting_depth": (
                median_depth
            ),
            "median_feature_duration_us": (
                median_feature_duration_us
            ),
            "median_max_setting_duration_us": (
                median_max_setting_duration_us
            ),
            "median_max_duration_over_min_t1": (
                median_r_t1
            ),
            "median_max_duration_over_min_t2": (
                median_r_t2
            ),
            "worst_sampled_duration_over_min_t1": (
                max_r_t1
            ),
            "worst_sampled_duration_over_min_t2": (
                max_r_t2
            ),
            "max_swap_feature_vector": (
                max_swaps
            ),
        },
        "full_2025_projection_from_actual_long_circuits": {
            "endpoints": int(
                common.N_VAL
            ),
            "circuits": int(
                common.N_VAL
                * len(
                    settings
                )
            ),
            "shots_per_setting": int(
                args.shots
            ),
            "scheduled_duration_times_shots_s": float(
                projected_scheduled_times_shots_s
            ),
        },
        "service_usage_before": (
            usage_before
        ),
        "qpu_submitted": False,
        "2026_loaded": False,
    }

    OUT_SUMMARY.write_text(
        json.dumps(
            db.json_safe(
                summary
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    db.create_all_tables()

    run_uuid = db.create_run(
        script_name=(
            Path(
                __file__
            ).name
        ),
        run_status=(
            "HARDWARE_PREFLIGHT_COMPLETE"
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
            PREFLIGHT_KEY
        ),
        selected_by_rules=(
            "Rule4"
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
            "SamplerV2_planned"
        ),
        measurement_method=(
            "grouped_endpoint_measurement"
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
        j_json={
            f"J{i}{j}": float(v)
            for (i, j), v
            in candidate[
                "J"
            ].items()
        },
        manifest_json=(
            manifest_meta
        ),
        ridge_lambda=float(
            package[
                "frozen_ridge"
            ][
                "lambda"
            ]
        ),
        training_cv_rmse=float(
            package[
                "frozen_ridge"
            ][
                "training_cv_rmse_from_existing_audit"
            ]
        ),
        ideal_validation_rmse_full_2025=float(
            package[
                "RWP_K117_ideal_2025"
            ][
                "rmse"
            ]
        ),
        ideal_validation_mae_full_2025=float(
            package[
                "RWP_K117_ideal_2025"
            ][
                "mae"
            ]
        ),
        ideal_validation_bias_full_2025=float(
            package[
                "RWP_K117_ideal_2025"
            ][
                "bias"
            ]
        ),
        backend_name=(
            args.backend
        ),
        physical_layout_json=(
            layout
        ),
        hardware_selection_json=(
            db.json_safe(
                selected
            )
        ),
        max_2q_error_percent=safe_float(
            selected.get(
                "compiled_2q_error_max_percent"
            )
        ),
        max_readout_error_percent=safe_float(
            selected.get(
                "max_readout_error_percent"
            )
        ),
        max_1q_error_percent=safe_float(
            selected.get(
                "compiled_1q_error_max_percent"
            )
        ),
        min_t1_us=(
            min_t1_us
        ),
        min_t2_us=(
            min_t2_us
        ),
        compiled_duration_us=(
            median_max_setting_duration_us
        ),
        r_t2=(
            median_r_t2
        ),
        shots_per_setting=int(
            args.shots
        ),
        n_validation_endpoints=int(
            common.N_VAL
        ),
        n_circuits=int(
            common.N_VAL
            * len(
                settings
            )
        ),
        full_2025=True,
        pilot_selection=(
            "preflight_only_no_submission"
        ),
        optimization_level=(
            OPT_LEVEL
        ),
        seed_transpiler=(
            SEED_TRANSPILE
        ),
        median_cz_per_feature=(
            median_cz
        ),
        median_max_setting_depth=(
            median_depth
        ),
        median_feature_duration_us=(
            median_feature_duration_us
        ),
        median_max_setting_duration_us=(
            median_max_setting_duration_us
        ),
        scheduled_duration_times_shots_s=float(
            projected_scheduled_times_shots_s
        ),
        service_usage_before_json=(
            usage_before
        ),
        notes=(
            "Candidate #4 actual K=117 washout-derived RWP "
            "hardware compilation preflight. Fresh H0 layout. "
            "No QPU submission. Long-circuit resources are "
            "measured from compiled representative endpoints "
            "unless --all-compile is used. 2026 not loaded."
        ),
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_RESOURCES.name,
        resources_df,
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_ENDPOINT_RESOURCES.name,
        endpoint_df,
    )

    db.save_json_artifact(
        run_uuid,
        OUT_SUMMARY.name,
        summary,
    )

    print()
    print(
        f"[database] Hardware-preflight run UUID: "
        f"{run_uuid}"
    )

    print()
    print(
        "Saved:"
    )
    print(
        f"  {OUT_RESOURCES}"
    )
    print(
        f"  {OUT_ENDPOINT_RESOURCES}"
    )
    print(
        f"  {OUT_SUMMARY}"
    )

    print()
    print(
        "STOP HERE. DO NOT SUBMIT CANDIDATE #4 YET."
    )
    print(
        "Send me the ACTUAL LONG-CIRCUIT PREFLIGHT section. "
        "We will decide from the real compiled duration/T1/T2 "
        "whether a full, pilot-only, or simulation-only hardware "
        "test is scientifically meaningful."
    )


if __name__ == "__main__":
    main()
