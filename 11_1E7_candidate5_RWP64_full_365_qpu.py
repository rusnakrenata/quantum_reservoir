from __future__ import annotations

"""
WEEK 11.1E.7 — CANDIDATE #5 FULL 365-DAY REAL-QPU RWP64 RUN
================================================================

This is the FULL 2025 validation run for Candidate #5.
There is NO pilot branch in this script.

Frozen scientific model
-----------------------
Candidate key      : CONT_H2_R2
Operating Trotter r: 2
Original protocol  : full CONT
Hardware surrogate : washout-derived RWP
K_RWP              : 64
Readout             : XZinj
Shots/setting       : 1024 by default
Validation endpoints: ALL 365 days of 2025
QPU circuits        : 365 x 2 = 730
2026                : FROZEN / NEVER LOADED

Reference chain
---------------
    full CONT ideal
        ->
    ideal RWP64
        ->
    REAL-QPU RWP64

The SAME full-CONT 2022-2024 Ridge readout is used throughout.
No RWP-specific readout is trained.

Fresh layout policy immediately before submission
-------------------------------------------------
This script repeats the memory-aware H2 layout search using CURRENT calibration:

1. enumerate all native H2 embeddings;
2. rank without a weighted score:
       memory feasibility first,
       larger memory coherence margin,
       lower worst required CZ error,
       lower worst injection readout error;
3. choose at least 3 distinct candidate layouts, preferring distinct
   physical memory pairs;
4. compile REAL K=64 circuits for first/middle/last 2025 endpoints for
   XXXX and ZZZZ on each layout;
5. require both persistent memory qubits q4,q5 to satisfy, using ACTUAL
   compiled max-setting duration,

       duration/T1 < 1
       duration/T2 < 1

6. choose the best ACTUALLY feasible layout lexicographically;
7. compile all 730 circuits on that layout;
8. only then submit one SamplerV2 job if --submit is supplied.

Safety / cost
-------------
The script prints IBM usage, scheduled-duration*shots, and an empirical
planning estimate.  If the service reports remaining usage and the empirical
estimate exceeds it, submission is blocked unless --force-over-budget is
explicitly supplied.

That flag should only be used when the account/plan is intentionally allowed
to consume more QPU time / incur charges.

No QPU job is submitted without --submit.
"""

import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2

from ibm_account import get_service
import db_objects as db


# =============================================================================
# Paths / frozen constants
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

CANDIDATE_SCRIPT = HERE / "11_1E1_candidate5_washout_trace_distance.py"
REFERENCE_SCRIPT = HERE / "11_1E3_candidate5_freeze_RWP64_reference.py"
LAYOUT_SCRIPT = HERE / "11_1E6_candidate5_memory_aware_layout_reselection.py"

FROZEN_PACKAGE = RESULTS / "11_1E3_candidate5_RWP64_frozen_package.json"
PREVIOUS_PREFLIGHT = RESULTS / "11_1E4_candidate5_RWP64_preflight_summary.json"

EXPECTED_CANDIDATE = "CONT_H2_R2"
EXPECTED_TOPOLOGY = "H2"
EXPECTED_R = 2
EXPECTED_K = 64
EXPECTED_READOUT = "XZinj"
EXPECTED_ALPHA = 0.75

DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_SHOTS = 1024
N_LAYOUTS_TO_TEST = 3

OPT_LEVEL = 1
SEED_TRANSPILE = 42

# Existing Week-11 empirical planning heuristic used in prior direct-QPU code.
# This is NOT an IBM billing formula.
EMPIRICAL_USAGE_SCALE = 52.0 / 0.701174

OUT_ALL_EMBEDDINGS = RESULTS / "11_1E7_candidate5_fresh_all_embeddings.csv"
OUT_TEST_LAYOUTS = RESULTS / "11_1E7_candidate5_fresh_test_layouts.csv"
OUT_TEST_COMPILED = RESULTS / "11_1E7_candidate5_fresh_test_compiled_settings.csv"
OUT_TEST_QUBITS = RESULTS / "11_1E7_candidate5_fresh_test_qubit_coherence.csv"
OUT_TEST_SUMMARY = RESULTS / "11_1E7_candidate5_fresh_test_layout_summary.csv"

OUT_RESOURCES = RESULTS / "11_1E7_candidate5_full365_qpu_resources.csv"
OUT_ENDPOINTS = RESULTS / "11_1E7_candidate5_full365_endpoints.csv"
OUT_PREFLIGHT = RESULTS / "11_1E7_candidate5_full365_qpu_preflight.json"
OUT_JOB = RESULTS / "11_1E7_candidate5_full365_qpu_job.json"
OUT_RAW_COUNTS = RESULTS / "11_1E7_candidate5_full365_qpu_raw_counts.csv"
OUT_FEATURES = RESULTS / "11_1E7_candidate5_full365_qpu_features_predictions.csv"
OUT_SUMMARY = RESULTS / "11_1E7_candidate5_full365_qpu_summary.json"

FEATURES = [
    "X0", "X1", "X2", "X3",
    "Z0", "Z1", "Z2", "Z3",
]

CURRENT_RUN_UUID = None


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


candmod = load_module(
    CANDIDATE_SCRIPT,
    "qrc_11_1e7_candidate",
)

refmod = load_module(
    REFERENCE_SCRIPT,
    "qrc_11_1e7_reference",
)

layoutmod = load_module(
    LAYOUT_SCRIPT,
    "qrc_11_1e7_layout",
)

common = candmod.common


# =============================================================================
# Generic helpers
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


def utc_now_naive():
    return datetime.now(
        timezone.utc
    ).replace(
        tzinfo=None
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


def get_service_usage(service):
    for attr in [
        "usage",
        "usage_info",
    ]:
        fn = getattr(
            service,
            attr,
            None,
        )

        if callable(
            fn
        ):
            try:
                out = fn()

                if hasattr(
                    out,
                    "to_dict",
                ):
                    out = out.to_dict()

                return db.json_safe(
                    out
                )
            except Exception:
                pass

    return None


def extract_remaining_usage_seconds(usage):
    if not isinstance(
        usage,
        dict,
    ):
        return None

    for key in [
        "usage_remaining_seconds",
        "remaining_seconds",
        "usage_remaining",
    ]:
        if key in usage:
            try:
                return float(
                    usage[
                        key
                    ]
                )
            except Exception:
                pass

    return None


def _parse_ibm_utc_timestamp(value):
    if value is None:
        return None

    try:
        ts = pd.to_datetime(
            value,
            utc=True,
        )

        return (
            ts.to_pydatetime()
            .replace(
                tzinfo=None
            )
        )
    except Exception:
        return None


def get_job_usage_seconds(job):
    fn = getattr(
        job,
        "usage",
        None,
    )

    if callable(
        fn
    ):
        try:
            value = fn()

            if value is None:
                return None

            if isinstance(
                value,
                dict,
            ):
                for key in [
                    "quantum_seconds",
                    "usage_seconds",
                    "seconds",
                ]:
                    if key in value:
                        return float(
                            value[
                                key
                            ]
                        )

                return None

            return float(
                value
            )
        except Exception:
            pass

    return None


def extract_ibm_qpu_timing(
    job,
    metrics,
):
    metrics = metrics or {}

    usage = metrics.get(
        "usage"
    ) or {}

    timestamps = metrics.get(
        "timestamps"
    ) or {}

    qpu_charge = usage.get(
        "qpu_charge_time_seconds"
    )

    if qpu_charge is None:
        qpu_charge = usage.get(
            "quantum_seconds"
        )

    if qpu_charge is None:
        qpu_charge = usage.get(
            "seconds"
        )

    if qpu_charge is None:
        qpu_charge = get_job_usage_seconds(
            job
        )

    if qpu_charge is not None:
        try:
            qpu_charge = float(
                qpu_charge
            )
        except Exception:
            qpu_charge = None

    execution_ns = metrics.get(
        "circuits_execution_time_ns"
    )

    if execution_ns is None:
        execution_ns = usage.get(
            "circuits_execution_time_ns"
        )

    qpu_circuit_seconds = None

    if execution_ns is not None:
        try:
            qpu_circuit_seconds = float(
                execution_ns
            ) / 1e9
        except Exception:
            qpu_circuit_seconds = None

    created_at = _parse_ibm_utc_timestamp(
        timestamps.get(
            "created"
        )
    )

    running_at = _parse_ibm_utc_timestamp(
        timestamps.get(
            "running"
        )
    )

    finished_at = _parse_ibm_utc_timestamp(
        timestamps.get(
            "finished"
        )
    )

    running_wall_seconds = None

    if (
        running_at is not None
        and finished_at is not None
    ):
        running_wall_seconds = float(
            (
                finished_at
                - running_at
            ).total_seconds()
        )

    return {
        "qpu_charge_time_seconds": (
            qpu_charge
        ),
        "qpu_circuits_execution_time_seconds": (
            qpu_circuit_seconds
        ),
        "qpu_running_wall_seconds": (
            running_wall_seconds
        ),
        "ibm_job_created_at_utc": (
            created_at
        ),
        "ibm_job_running_at_utc": (
            running_at
        ),
        "ibm_job_finished_at_utc": (
            finished_at
        ),
    }


# =============================================================================
# Frozen model/reference reconstruction
# =============================================================================

def build_frozen_reference(
    candidate,
    package,
):
    """
    Reconstruct the SAME full-CONT training readout and the exact ideal RWP64
    reference. No hyperparameter selection is performed here.

    Lambda comes from the already-frozen 11.1E.3 package.
    """
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
    ) = refmod.build_full_cont_bank(
        candidate,
        angles,
    )

    endpoints = np.arange(
        len(
            full_bank
        ),
        dtype=int,
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

    X_val_cont = full_bank[
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

    if len(
        val_endpoints
    ) != 365:
        raise RuntimeError(
            f"Expected all 365 2025 endpoints, got {len(val_endpoints)}."
        )

    ridge_lambda = float(
        package[
            "frozen_ridge"
        ][
            "lambda"
        ]
    )

    (
        model,
        scaler,
        keep,
    ) = common.fit_scaled_ridge(
        X_train,
        y_train,
        ridge_lambda,
    )

    pred_cont = common.predict_scaled_ridge(
        model,
        scaler,
        keep,
        X_val_cont,
    )

    cont_metrics = {
        "rmse": rmse(
            y_val,
            pred_cont,
        ),
        "mae": mae(
            y_val,
            pred_cont,
        ),
        "bias": bias(
            y_val,
            pred_cont,
        ),
    }

    stored_cont = package[
        "full_CONT_ideal_2025"
    ]

    if abs(
        cont_metrics[
            "rmse"
        ]
        - float(
            stored_cont[
                "rmse"
            ]
        )
    ) > 3e-6:
        raise RuntimeError(
            "Full-CONT reference reconstruction does not match frozen package."
        )

    X_val_rwp = np.vstack(
        [
            refmod.replay_feature(
                A_list,
                endpoint=int(
                    endpoint
                ),
                K=EXPECTED_K,
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

    stored_rwp = package[
        "RWP_K64_ideal_2025"
    ]

    if abs(
        rwp_metrics[
            "rmse"
        ]
        - float(
            stored_rwp[
                "rmse"
            ]
        )
    ) > 3e-6:
        raise RuntimeError(
            "Ideal RWP64 reference reconstruction does not match frozen package."
        )

    return {
        "angles": (
            angles
        ),
        "val_endpoints": (
            val_endpoints
        ),
        "y_val": (
            y_val
        ),
        "X_val_cont": (
            X_val_cont
        ),
        "X_val_rwp": (
            X_val_rwp
        ),
        "pred_cont": (
            pred_cont
        ),
        "pred_rwp": (
            pred_rwp
        ),
        "model": (
            model
        ),
        "scaler": (
            scaler
        ),
        "keep": (
            keep
        ),
        "ridge_lambda": (
            ridge_lambda
        ),
        "cv_rmse": float(
            package[
                "frozen_ridge"
            ][
                "training_cv_rmse"
            ]
        ),
        "cont_metrics": (
            cont_metrics
        ),
        "rwp_metrics": (
            rwp_metrics
        ),
    }


# =============================================================================
# Sampler decoding
# =============================================================================

def bit_from_qiskit_string(
    bitstring,
    c_index,
):
    s = str(
        bitstring
    ).replace(
        " ",
        "",
    )

    return int(
        s[
            -1
            - int(
                c_index
            )
        ]
    )


def expectation_from_counts(
    counts,
    classical_bits,
):
    total = float(
        sum(
            counts.values()
        )
    )

    if total <= 0:
        raise RuntimeError(
            "Empty counts."
        )

    acc = 0.0

    for bitstring, n in counts.items():
        eig = 1.0

        for c in classical_bits:
            eig *= (
                -1.0
                if bit_from_qiskit_string(
                    bitstring,
                    c,
                )
                else 1.0
            )

        acc += (
            float(
                n
            )
            * eig
        )

    return (
        acc
        / total
    )


def get_pub_counts(
    pub_result,
):
    data = pub_result.data

    reg = getattr(
        data,
        "m",
        None,
    )

    if reg is None:
        raise RuntimeError(
            "Sampler result does not contain classical register 'm'."
        )

    return reg.get_counts()


# =============================================================================
# Current memory-aware layout selection
# =============================================================================

def choose_fresh_memory_aware_layout(
    backend,
    candidate,
    t_setting_est_us,
    n_layouts=3,
):
    (
        adjacency,
        eligible,
        qubit_cal,
    ) = layoutmod.current_physical_graph(
        backend
    )

    logical_edges = [
        (
            int(
                a
            ),
            int(
                b
            ),
        )
        for a, b
        in common.TOPOLOGY_EDGES[
            EXPECTED_TOPOLOGY
        ]
    ]

    layouts = layoutmod.enumerate_native_embeddings(
        logical_edges,
        adjacency,
        eligible,
    )

    embedding_rows = [
        layoutmod.embedding_metrics(
            backend,
            layout,
            logical_edges,
            qubit_cal,
            t_setting_est_us,
        )
        for layout
        in layouts
    ]

    metrics_df = pd.DataFrame(
        embedding_rows
    )

    ranked_df, selected_df = (
        layoutmod.select_distinct_layouts(
            metrics_df,
            max(
                3,
                int(
                    n_layouts
                ),
            ),
        )
    )

    ranked_df.to_csv(
        OUT_ALL_EMBEDDINGS,
        index=False,
    )

    selected_df.to_csv(
        OUT_TEST_LAYOUTS,
        index=False,
    )

    # Compile actual K64 representative circuits for each selected layout.
    test_endpoints = np.asarray(
        [
            int(
                common.N_TRAIN
            ),
            int(
                common.N_TRAIN
                + common.N_VAL
                // 2
            ),
            int(
                common.N_TRAIN
                + common.N_VAL
                - 1
            ),
        ],
        dtype=int,
    )

    settings = layoutmod.measurement_settings()

    compiled_rows = []

    for _, selected in selected_df.iterrows():
        layout_id = str(
            selected[
                "test_layout_id"
            ]
        )

        layout = json.loads(
            selected[
                "layout_json"
            ]
        )

        print()
        print(
            f"{layout_id}: candidate layout={layout}"
        )

        for endpoint in test_endpoints:
            for setting in settings:
                setting_name = setting[
                    0
                ]

                logical = layoutmod.build_rwp64_circuit(
                    candidate,
                    FROZEN_REFERENCE[
                        "angles"
                    ],
                    int(
                        endpoint
                    ),
                    setting,
                )

                compiled = layoutmod.compile_circuit(
                    logical,
                    backend,
                    layout,
                )

                compiled_rows.append(
                    layoutmod.compiled_resource_row(
                        compiled,
                        backend,
                        layout_id,
                        layout,
                        int(
                            endpoint
                        ),
                        setting_name,
                    )
                )

    compiled_df = pd.DataFrame(
        compiled_rows
    )

    compiled_df.to_csv(
        OUT_TEST_COMPILED,
        index=False,
    )

    (
        qubit_df,
        layout_summary_df,
    ) = layoutmod.build_actual_layout_summaries(
        selected_df,
        compiled_df,
        qubit_cal,
    )

    qubit_df.to_csv(
        OUT_TEST_QUBITS,
        index=False,
    )

    layout_summary_df.to_csv(
        OUT_TEST_SUMMARY,
        index=False,
    )

    actual_feasible = layout_summary_df[
        layout_summary_df[
            "actual_memory_pass"
        ]
        == True
    ].copy()

    if len(
        actual_feasible
    ) == 0:
        raise RuntimeError(
            "Fresh memory-aware reselection found no tested layout that "
            "passes actual K=64 memory T1/T2 < 1. QPU submission aborted."
        )

    actual_feasible = actual_feasible.sort_values(
        by=[
            "actual_min_memory_coherence_margin",
            "worst_required_cz_error",
            "worst_injection_readout_error",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        na_position="last",
    )

    chosen = actual_feasible.iloc[
        0
    ].to_dict()

    chosen_layout = [
        int(
            x
        )
        for x in json.loads(
            chosen[
                "layout_json"
            ]
        )
    ]

    chosen_qubits = qubit_df[
        qubit_df[
            "test_layout_id"
        ]
        == chosen[
            "test_layout_id"
        ]
    ].copy()

    return {
        "native_embeddings_found": int(
            len(
                layouts
            )
        ),
        "estimated_memory_feasible_embeddings": int(
            metrics_df[
                "memory_pass_est"
            ].sum()
        ),
        "tested_layouts": selected_df,
        "compiled_test_settings": compiled_df,
        "qubit_table": qubit_df,
        "layout_summary": layout_summary_df,
        "chosen_summary": chosen,
        "chosen_layout": chosen_layout,
        "chosen_qubits": chosen_qubits,
    }


# =============================================================================
# Full compile resource helpers
# =============================================================================

def resource_row(
    circuit,
    backend,
    endpoint,
    setting,
):
    ops = {
        str(
            key
        ): int(
            value
        )
        for key, value
        in circuit.count_ops().items()
    }

    duration_s = float(
        circuit.estimate_duration(
            backend.target,
            unit="s",
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
            circuit.depth()
        ),
        "size": int(
            circuit.size()
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
            duration_s
            * 1e6
        ),
        "operations": json.dumps(
            ops,
            sort_keys=True,
        ),
    }


# Global is assigned inside main before fresh layout selection.
FROZEN_REFERENCE = None


# =============================================================================
# Main
# =============================================================================

def main():
    global CURRENT_RUN_UUID
    global FROZEN_REFERENCE

    parser = argparse.ArgumentParser(
        description=(
            "Candidate #5 FULL 365-day RWP64 real-QPU validation."
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
        "--submit",
        action="store_true",
        help=(
            "Actually submit the single 730-circuit SamplerV2 job. "
            "Without this flag the script performs the complete fresh "
            "layout selection and full 730-circuit compile only."
        ),
    )

    parser.add_argument(
        "--force-over-budget",
        action="store_true",
        help=(
            "Permit submission even when the empirical planning estimate "
            "exceeds service-reported remaining usage. Use only when "
            "additional usage/cost is intentional."
        ),
    )

    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError(
            "--shots must be positive."
        )

    print(
        "=" * 126
    )
    print(
        "WEEK 11.1E.7 — CANDIDATE #5 FULL 365-DAY REAL-QPU RWP64"
    )
    print(
        "=" * 126
    )
    print(
        "2022-2024 = frozen full-CONT readout"
    )
    print(
        "2025      = ALL 365 validation endpoints"
    )
    print(
        "2026      = FROZEN / NOT LOADED"
    )
    print()

    package = load_json(
        FROZEN_PACKAGE
    )

    previous_preflight = load_json(
        PREVIOUS_PREFLIGHT
    )

    candidate, manifest_meta = (
        candmod.load_candidate()
    )

    # Hard identity audit.
    if candidate[
        "candidate_id"
    ] != EXPECTED_CANDIDATE:
        raise RuntimeError(
            "Unexpected Candidate #5 key."
        )

    if candidate[
        "topology"
    ] != EXPECTED_TOPOLOGY:
        raise RuntimeError(
            "Unexpected Candidate #5 topology."
        )

    if int(
        candidate[
            "r"
        ]
    ) != EXPECTED_R:
        raise RuntimeError(
            "Unexpected Candidate #5 operating r."
        )

    if not np.isclose(
        float(
            candidate[
                "alpha"
            ]
        ),
        EXPECTED_ALPHA,
    ):
        raise RuntimeError(
            "Unexpected Candidate #5 alpha."
        )

    if int(
        package[
            "operational_K"
        ]
    ) != EXPECTED_K:
        raise RuntimeError(
            "Unexpected frozen RWP K."
        )

    if package[
        "readout"
    ] != EXPECTED_READOUT:
        raise RuntimeError(
            "Unexpected frozen readout."
        )

    FROZEN_REFERENCE = build_frozen_reference(
        candidate,
        package,
    )

    print(
        "Frozen Candidate #5:"
    )
    print(
        f"  candidate={EXPECTED_CANDIDATE}"
    )
    print(
        f"  topology={EXPECTED_TOPOLOGY}"
    )
    print(
        f"  operating r={EXPECTED_R}"
    )
    print(
        f"  K_RWP={EXPECTED_K}"
    )
    print(
        f"  readout={EXPECTED_READOUT}"
    )
    print(
        f"  shots/setting={args.shots}"
    )
    print(
        f"  lambda={FROZEN_REFERENCE['ridge_lambda']}"
    )
    print(
        f"  training CV RMSE={FROZEN_REFERENCE['cv_rmse']:.6f}"
    )
    print(
        f"  full CONT ideal 2025 RMSE="
        f"{FROZEN_REFERENCE['cont_metrics']['rmse']:.6f}"
    )
    print(
        f"  ideal RWP64 2025 RMSE="
        f"{FROZEN_REFERENCE['rwp_metrics']['rmse']:.6f}"
    )
    print()

    # Database record is created before any hardware submission.
    db.create_all_tables()

    run_uuid = db.create_run(
        script_name=(
            Path(
                __file__
            ).name
        ),
        run_status=(
            "PREPARING"
        ),
        started_at_utc=(
            utc_now_naive()
        ),
        forecast_dataset_name=(
            "property_damage_next_day_v1"
        ),
        feature_set_name=(
            "F4"
        ),
        evaluation_split=(
            "validation_2025"
        ),
        candidate_key=(
            "CONT_H2_R2_RWP64_FULL365"
        ),
        selected_by_rules=(
            "Rule2"
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
            "SamplerV2"
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
            f"J{i}{j}": float(
                value
            )
            for (
                i,
                j,
            ), value
            in candidate[
                "J"
            ].items()
        },
        manifest_json=(
            manifest_meta
        ),
        ridge_lambda=float(
            FROZEN_REFERENCE[
                "ridge_lambda"
            ]
        ),
        training_cv_rmse=float(
            FROZEN_REFERENCE[
                "cv_rmse"
            ]
        ),
        ideal_validation_rmse_full_2025=float(
            FROZEN_REFERENCE[
                "rwp_metrics"
            ][
                "rmse"
            ]
        ),
        ideal_validation_mae_full_2025=float(
            FROZEN_REFERENCE[
                "rwp_metrics"
            ][
                "mae"
            ]
        ),
        ideal_validation_bias_full_2025=float(
            FROZEN_REFERENCE[
                "rwp_metrics"
            ][
                "bias"
            ]
        ),
        backend_name=(
            args.backend
        ),
        shots_per_setting=int(
            args.shots
        ),
        n_validation_endpoints=(
            365
        ),
        n_circuits=(
            730
        ),
        full_2025=True,
        pilot_selection=(
            "all_2025_validation"
        ),
        optimization_level=(
            OPT_LEVEL
        ),
        seed_transpiler=(
            SEED_TRANSPILE
        ),
        notes=(
            "Candidate #5 full 365-day QPU RWP64 validation. Fresh "
            "memory-aware H2 reselection immediately before compile/run. "
            "Same frozen full-CONT 2022-2024 scaler/Ridge; ideal RWP64 is "
            "the immediate hardware-transfer reference. 2026 not loaded."
        ),
    )

    CURRENT_RUN_UUID = run_uuid

    print(
        f"[database] QPU run UUID: {run_uuid}"
    )
    print()

    service = get_service()

    backend = service.backend(
        args.backend
    )

    # ------------------------------------------------------------------
    # Fresh memory-aware reselection from current calibration.
    # ------------------------------------------------------------------
    print(
        "=" * 126
    )
    print(
        f"FRESH {args.backend} MEMORY-AWARE H2 RESELECTION"
    )
    print(
        "=" * 126
    )

    t_setting_est_us = float(
        previous_preflight[
            "actual_compiled_sample"
        ][
            "median_max_setting_duration_us"
        ]
    )

    fresh = choose_fresh_memory_aware_layout(
        backend=backend,
        candidate=candidate,
        t_setting_est_us=(
            t_setting_est_us
        ),
        n_layouts=(
            N_LAYOUTS_TO_TEST
        ),
    )

    chosen = fresh[
        "chosen_summary"
    ]

    layout = fresh[
        "chosen_layout"
    ]

    print()
    print(
        "Fresh memory-aware physical choice:"
    )
    print(
        f"  layout={layout}"
    )
    print(
        f"  test layout id={chosen['test_layout_id']}"
    )
    print(
        f"  memory q4 -> P{chosen['memory_q4_physical']}"
    )
    print(
        f"  memory q5 -> P{chosen['memory_q5_physical']}"
    )
    print(
        f"  actual representative t_setting="
        f"{chosen['actual_median_max_setting_duration_us']:.3f} us"
    )
    print(
        f"  worst memory t/T1="
        f"{chosen['actual_worst_memory_duration_over_t1']:.6f}"
    )
    print(
        f"  worst memory t/T2="
        f"{chosen['actual_worst_memory_duration_over_t2']:.6f}"
    )
    print(
        f"  minimum memory coherence margin="
        f"{chosen['actual_min_memory_coherence_margin']:.6f}"
    )
    print()

    # Store fresh selection artifacts immediately.
    db.save_dataframe_artifact(
        run_uuid,
        OUT_ALL_EMBEDDINGS.name,
        pd.read_csv(
            OUT_ALL_EMBEDDINGS
        ),
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_TEST_LAYOUTS.name,
        fresh[
            "tested_layouts"
        ],
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_TEST_COMPILED.name,
        fresh[
            "compiled_test_settings"
        ],
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_TEST_QUBITS.name,
        fresh[
            "qubit_table"
        ],
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_TEST_SUMMARY.name,
        fresh[
            "layout_summary"
        ],
    )

    # ------------------------------------------------------------------
    # Compile ALL 365 x 2 = 730 circuits on the chosen layout.
    # ------------------------------------------------------------------
    settings = layoutmod.measurement_settings()

    circuits = []
    circuit_meta = []
    resource_rows = []

    print(
        "=" * 126
    )
    print(
        "COMPILING FULL 365-DAY RWP64 WORKLOAD"
    )
    print(
        "=" * 126
    )

    for idx, endpoint in enumerate(
        FROZEN_REFERENCE[
            "val_endpoints"
        ]
    ):
        endpoint = int(
            endpoint
        )

        if (
            idx % 25 == 0
            or idx
            == len(
                FROZEN_REFERENCE[
                    "val_endpoints"
                ]
            )
            - 1
        ):
            print(
                f"  compiling validation endpoint "
                f"{idx+1}/365 (global endpoint={endpoint})"
            )

        for setting in settings:
            setting_name = setting[
                0
            ]

            logical = layoutmod.build_rwp64_circuit(
                candidate,
                FROZEN_REFERENCE[
                    "angles"
                ],
                endpoint,
                setting,
            )

            isa = transpile(
                logical,
                backend=backend,
                initial_layout=layout,
                routing_method="none",
                optimization_level=(
                    OPT_LEVEL
                ),
                seed_transpiler=(
                    SEED_TRANSPILE
                ),
                scheduling_method="alap",
            )

            backend.check_faulty(
                isa
            )

            rr = resource_row(
                isa,
                backend,
                endpoint,
                setting_name,
            )

            if rr[
                "n_swap"
            ] != 0:
                raise RuntimeError(
                    f"SWAP detected at endpoint={endpoint}, "
                    f"setting={setting_name}. Submission aborted."
                )

            circuits.append(
                isa
            )

            circuit_meta.append(
                {
                    "endpoint": (
                        endpoint
                    ),
                    "setting": (
                        setting_name
                    ),
                }
            )

            resource_rows.append(
                rr
            )

    if len(
        circuits
    ) != 730:
        raise RuntimeError(
            f"Expected 730 circuits, compiled {len(circuits)}."
        )

    resources = pd.DataFrame(
        resource_rows
    )

    resources.to_csv(
        OUT_RESOURCES,
        index=False,
    )

    endpoint_resources = (
        resources.groupby(
            "endpoint",
            as_index=False,
        )
        .agg(
            n_settings=(
                "setting",
                "nunique",
            ),
            cz_feature_vector=(
                "n_cz",
                "sum",
            ),
            swaps_feature_vector=(
                "n_swap",
                "sum",
            ),
            resets_feature_vector=(
                "n_reset",
                "sum",
            ),
            max_setting_depth=(
                "depth",
                "max",
            ),
            duration_feature_vector_us=(
                "duration_us",
                "sum",
            ),
            max_setting_duration_us=(
                "duration_us",
                "max",
            ),
        )
    )

    selected_endpoints_df = pd.DataFrame(
        {
            "endpoint": FROZEN_REFERENCE[
                "val_endpoints"
            ].astype(
                int
            )
        }
    )

    selected_endpoints_df.to_csv(
        OUT_ENDPOINTS,
        index=False,
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_RESOURCES.name,
        resources,
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_ENDPOINTS.name,
        selected_endpoints_df,
    )

    scheduled_shot_seconds = float(
        np.sum(
            resources[
                "duration_us"
            ].to_numpy(
                dtype=float
            )
            * 1e-6
            * int(
                args.shots
            )
        )
    )

    empirical_usage_estimate = (
        scheduled_shot_seconds
        * EMPIRICAL_USAGE_SCALE
    )

    usage_before = get_service_usage(
        service
    )

    remaining = extract_remaining_usage_seconds(
        usage_before
    )

    print()
    print(
        "=" * 126
    )
    print(
        "FULL 365-DAY QPU PREFLIGHT"
    )
    print(
        "=" * 126
    )
    print(
        f"Backend:                          {args.backend}"
    )
    print(
        f"Fresh physical layout:            {layout}"
    )
    print(
        f"Validation endpoints:             365"
    )
    print(
        f"Circuits:                         730"
    )
    print(
        f"Settings/endpoint:                2"
    )
    print(
        f"Shots/setting:                    {args.shots}"
    )
    print(
        f"Median CZ/feature vector:         "
        f"{endpoint_resources['cz_feature_vector'].median():.0f}"
    )
    print(
        f"Median resets/feature vector:     "
        f"{endpoint_resources['resets_feature_vector'].median():.0f}"
    )
    print(
        f"Median max setting depth:         "
        f"{endpoint_resources['max_setting_depth'].median():.0f}"
    )
    print(
        f"Median feature duration:          "
        f"{endpoint_resources['duration_feature_vector_us'].median():.3f} us"
    )
    print(
        f"Median max-setting duration:      "
        f"{endpoint_resources['max_setting_duration_us'].median():.3f} us"
    )
    print(
        f"Scheduled-duration*shots:         "
        f"{scheduled_shot_seconds:.3f} s"
    )
    print(
        f"Empirical usage planning estimate:"
        f" {empirical_usage_estimate:.1f} s "
        f"(heuristic only)"
    )

    if remaining is not None:
        print(
            f"Service-reported remaining usage: "
            f"{remaining:.1f} s"
        )

    if usage_before is not None:
        print()
        print(
            "Service usage before:"
        )
        print(
            json.dumps(
                db.json_safe(
                    usage_before
                ),
                indent=2,
            )
        )

    preflight = {
        "candidate_key": (
            EXPECTED_CANDIDATE
        ),
        "run_candidate_key": (
            "CONT_H2_R2_RWP64_FULL365"
        ),
        "backend": (
            args.backend
        ),
        "layout": (
            layout
        ),
        "memory_q4_physical": (
            chosen[
                "memory_q4_physical"
            ]
        ),
        "memory_q5_physical": (
            chosen[
                "memory_q5_physical"
            ]
        ),
        "actual_memory_pass": bool(
            chosen[
                "actual_memory_pass"
            ]
        ),
        "actual_worst_memory_duration_over_t1": float(
            chosen[
                "actual_worst_memory_duration_over_t1"
            ]
        ),
        "actual_worst_memory_duration_over_t2": float(
            chosen[
                "actual_worst_memory_duration_over_t2"
            ]
        ),
        "actual_min_memory_coherence_margin": float(
            chosen[
                "actual_min_memory_coherence_margin"
            ]
        ),
        "shots_per_setting": int(
            args.shots
        ),
        "n_endpoints": 365,
        "n_circuits": 730,
        "full_2025": True,
        "ridge_lambda": (
            FROZEN_REFERENCE[
                "ridge_lambda"
            ]
        ),
        "training_cv_rmse": (
            FROZEN_REFERENCE[
                "cv_rmse"
            ]
        ),
        "full_CONT_ideal_2025": (
            FROZEN_REFERENCE[
                "cont_metrics"
            ]
        ),
        "ideal_RWP64_2025": (
            FROZEN_REFERENCE[
                "rwp_metrics"
            ]
        ),
        "scheduled_duration_times_shots_s": (
            scheduled_shot_seconds
        ),
        "empirical_usage_estimate_s": (
            empirical_usage_estimate
        ),
        "service_usage_before": (
            usage_before
        ),
        "fresh_layout_test_summary": (
            db.json_safe(
                fresh[
                    "layout_summary"
                ].to_dict(
                    orient="records"
                )
            )
        ),
    }

    OUT_PREFLIGHT.write_text(
        json.dumps(
            db.json_safe(
                preflight
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    db.save_json_artifact(
        run_uuid,
        OUT_PREFLIGHT.name,
        preflight,
    )

    # Chosen physical region summary for DB.
    chosen_qubits = fresh[
        "chosen_qubits"
    ]

    min_t1 = float(
        chosen_qubits[
            "t1_us"
        ].min()
    )

    min_t2 = float(
        chosen_qubits[
            "t2_us"
        ].min()
    )

    db.update_run(
        run_uuid,
        run_status=(
            "PREFLIGHT_COMPLETE"
        ),
        physical_layout_json=(
            layout
        ),
        hardware_selection_json=(
            db.json_safe(
                chosen
            )
        ),
        max_2q_error_percent=(
            None
            if chosen.get(
                "worst_required_cz_error"
            ) is None
            else float(
                chosen[
                    "worst_required_cz_error"
                ]
            )
            * 100.0
        ),
        max_readout_error_percent=(
            None
            if chosen.get(
                "worst_injection_readout_error"
            ) is None
            else float(
                chosen[
                    "worst_injection_readout_error"
                ]
            )
            * 100.0
        ),
        min_t1_us=(
            min_t1
        ),
        min_t2_us=(
            min_t2
        ),
        compiled_duration_us=float(
            endpoint_resources[
                "max_setting_duration_us"
            ].median()
        ),
        r_t2=float(
            chosen[
                "actual_worst_memory_duration_over_t2"
            ]
        ),
        n_circuits=(
            730
        ),
        median_cz_per_feature=float(
            endpoint_resources[
                "cz_feature_vector"
            ].median()
        ),
        median_max_setting_depth=float(
            endpoint_resources[
                "max_setting_depth"
            ].median()
        ),
        median_feature_duration_us=float(
            endpoint_resources[
                "duration_feature_vector_us"
            ].median()
        ),
        median_max_setting_duration_us=float(
            endpoint_resources[
                "max_setting_duration_us"
            ].median()
        ),
        scheduled_duration_times_shots_s=(
            scheduled_shot_seconds
        ),
        empirical_usage_estimate_s=(
            empirical_usage_estimate
        ),
        service_usage_before_json=(
            usage_before
        ),
    )

    if not args.submit:
        db.update_run(
            run_uuid,
            run_status=(
                "DRY_RUN_COMPLETE"
            ),
            completed_at_utc=(
                utc_now_naive()
            ),
        )

        print()
        print(
            "FULL 365-DAY DRY RUN COMPLETE — NO QPU JOB SUBMITTED."
        )
        print()
        print(
            "To submit the full 365-day run using a NEW fresh reselection:"
        )
        print(
            f"  python {Path(__file__).name} "
            f"--backend {args.backend} "
            f"--shots {args.shots} "
            f"--submit"
        )
        print()
        print(
            "If the budget guard blocks submission and extra usage/cost is "
            "intentionally allowed, add --force-over-budget."
        )
        return

    # ------------------------------------------------------------------
    # Budget guard.
    # ------------------------------------------------------------------
    if (
        remaining is not None
        and empirical_usage_estimate
        > remaining
        and not args.force_over_budget
    ):
        raise RuntimeError(
            "Submission blocked by usage guard: empirical planning estimate "
            f"{empirical_usage_estimate:.1f}s exceeds service-reported "
            f"remaining usage {remaining:.1f}s. If additional usage/cost is "
            "intentional, re-run with --force-over-budget."
        )

    # ------------------------------------------------------------------
    # Submit ONE SamplerV2 job containing all 730 circuits.
    # ------------------------------------------------------------------
    print()
    print(
        "=" * 126
    )
    print(
        "SUBMITTING FULL 365-DAY SamplerV2 JOB"
    )
    print(
        "=" * 126
    )
    print(
        f"Submitting 730 ISA circuits at "
        f"{args.shots} shots/setting..."
    )

    db.update_run(
        run_uuid,
        run_status=(
            "SUBMITTING"
        ),
        submitted_at_utc=(
            utc_now_naive()
        ),
    )

    sampler = SamplerV2(
        mode=backend
    )

    job = sampler.run(
        circuits,
        shots=int(
            args.shots
        ),
    )

    job_id = job.job_id()

    print(
        f"Job ID: {job_id}"
    )

    db.update_run(
        run_uuid,
        run_status=(
            "SUBMITTED"
        ),
        job_id=str(
            job_id
        ),
    )

    # Save job id immediately before waiting.
    early_job_payload = {
        "run_uuid": (
            run_uuid
        ),
        "job_id": (
            job_id
        ),
        "backend": (
            args.backend
        ),
        "layout": (
            layout
        ),
        "shots": int(
            args.shots
        ),
        "n_circuits": 730,
        "status_at_save": str(
            job.status()
        ),
    }

    OUT_JOB.write_text(
        json.dumps(
            db.json_safe(
                early_job_payload
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    db.save_json_artifact(
        run_uuid,
        OUT_JOB.name,
        early_job_payload,
    )

    result = job.result()

    final_job_status = str(
        job.status()
    )

    print(
        f"Final status: {final_job_status}"
    )

    metrics = None

    try:
        metrics = db.json_safe(
            job.metrics()
        )
    except Exception as exc:
        print(
            f"[timing] WARNING: job.metrics() unavailable: {exc}"
        )

    qpu_timing = extract_ibm_qpu_timing(
        job,
        metrics,
    )

    print()
    print(
        "IBM QPU TIMING"
    )
    print(
        "-" * 126
    )
    print(
        f"QPU charge / locked time:       "
        f"{qpu_timing['qpu_charge_time_seconds']}"
    )
    print(
        f"Actual circuit execution on QPU:"
        f" {qpu_timing['qpu_circuits_execution_time_seconds']}"
    )
    print(
        f"IBM RUNNING -> FINISHED elapsed:"
        f" {qpu_timing['qpu_running_wall_seconds']}"
    )

    job_payload = {
        **early_job_payload,
        "final_status": (
            final_job_status
        ),
        "qpu_charge_time_seconds": (
            qpu_timing[
                "qpu_charge_time_seconds"
            ]
        ),
        "qpu_circuits_execution_time_seconds": (
            qpu_timing[
                "qpu_circuits_execution_time_seconds"
            ]
        ),
        "qpu_running_wall_seconds": (
            qpu_timing[
                "qpu_running_wall_seconds"
            ]
        ),
        "ibm_job_created_at_utc": (
            qpu_timing[
                "ibm_job_created_at_utc"
            ]
        ),
        "ibm_job_running_at_utc": (
            qpu_timing[
                "ibm_job_running_at_utc"
            ]
        ),
        "ibm_job_finished_at_utc": (
            qpu_timing[
                "ibm_job_finished_at_utc"
            ]
        ),
        "metrics": (
            metrics
        ),
    }

    OUT_JOB.write_text(
        json.dumps(
            db.json_safe(
                job_payload
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # Update same DB artifact payload by inserting under a second explicit name,
    # avoiding duplicate-unique-key semantics in save_json_artifact.
    db.save_json_artifact(
        run_uuid,
        "11_1E7_candidate5_full365_qpu_job_final.json",
        job_payload,
    )

    db.update_run(
        run_uuid,
        run_status=(
            "QPU_COMPLETE_PARSING"
        ),
        job_status=(
            final_job_status
        ),
        job_usage_seconds=(
            qpu_timing[
                "qpu_charge_time_seconds"
            ]
        ),
        qpu_charge_time_seconds=(
            qpu_timing[
                "qpu_charge_time_seconds"
            ]
        ),
        qpu_circuits_execution_time_seconds=(
            qpu_timing[
                "qpu_circuits_execution_time_seconds"
            ]
        ),
        qpu_running_wall_seconds=(
            qpu_timing[
                "qpu_running_wall_seconds"
            ]
        ),
        ibm_job_created_at_utc=(
            qpu_timing[
                "ibm_job_created_at_utc"
            ]
        ),
        ibm_job_running_at_utc=(
            qpu_timing[
                "ibm_job_running_at_utc"
            ]
        ),
        ibm_job_finished_at_utc=(
            qpu_timing[
                "ibm_job_finished_at_utc"
            ]
        ),
        job_metrics_json=(
            metrics
        ),
    )

    # ------------------------------------------------------------------
    # Parse raw QPU counts into XZinj feature vectors.
    # ------------------------------------------------------------------
    if len(
        result
    ) != len(
        circuits
    ):
        raise RuntimeError(
            f"Sampler returned {len(result)} pubs for "
            f"{len(circuits)} circuits."
        )

    by_endpoint = {
        int(
            endpoint
        ): {}
        for endpoint
        in FROZEN_REFERENCE[
            "val_endpoints"
        ]
    }

    raw_rows = []

    for pub_result, meta in zip(
        result,
        circuit_meta,
    ):
        counts = get_pub_counts(
            pub_result
        )

        endpoint = int(
            meta[
                "endpoint"
            ]
        )

        setting = str(
            meta[
                "setting"
            ]
        )

        raw_rows.append(
            {
                "endpoint": (
                    endpoint
                ),
                "setting": (
                    setting
                ),
                "counts_json": json.dumps(
                    counts,
                    sort_keys=True,
                ),
            }
        )

        if setting == "XXXX":
            by_endpoint[
                endpoint
            ][
                "X0"
            ] = expectation_from_counts(
                counts,
                [
                    0
                ],
            )

            by_endpoint[
                endpoint
            ][
                "X1"
            ] = expectation_from_counts(
                counts,
                [
                    1
                ],
            )

            by_endpoint[
                endpoint
            ][
                "X2"
            ] = expectation_from_counts(
                counts,
                [
                    2
                ],
            )

            by_endpoint[
                endpoint
            ][
                "X3"
            ] = expectation_from_counts(
                counts,
                [
                    3
                ],
            )

        elif setting == "ZZZZ":
            by_endpoint[
                endpoint
            ][
                "Z0"
            ] = expectation_from_counts(
                counts,
                [
                    0
                ],
            )

            by_endpoint[
                endpoint
            ][
                "Z1"
            ] = expectation_from_counts(
                counts,
                [
                    1
                ],
            )

            by_endpoint[
                endpoint
            ][
                "Z2"
            ] = expectation_from_counts(
                counts,
                [
                    2
                ],
            )

            by_endpoint[
                endpoint
            ][
                "Z3"
            ] = expectation_from_counts(
                counts,
                [
                    3
                ],
            )

        else:
            raise RuntimeError(
                f"Unexpected measurement setting: {setting}"
            )

    raw_counts_df = pd.DataFrame(
        raw_rows
    )

    raw_counts_df.to_csv(
        OUT_RAW_COUNTS,
        index=False,
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_RAW_COUNTS.name,
        raw_counts_df,
    )

    # ------------------------------------------------------------------
    # QPU predictions vs ideal RWP64 and full CONT.
    # ------------------------------------------------------------------
    feature_rows = []
    X_qpu = []

    for i, endpoint in enumerate(
        FROZEN_REFERENCE[
            "val_endpoints"
        ]
    ):
        endpoint = int(
            endpoint
        )

        vals = by_endpoint[
            endpoint
        ]

        missing = [
            feature
            for feature
            in FEATURES
            if feature
            not in vals
        ]

        if missing:
            raise RuntimeError(
                f"Endpoint {endpoint}: missing QPU features {missing}."
            )

        qpu_vec = np.asarray(
            [
                float(
                    vals[
                        feature
                    ]
                )
                for feature
                in FEATURES
            ],
            dtype=float,
        )

        ideal_rwp_vec = (
            FROZEN_REFERENCE[
                "X_val_rwp"
            ][
                i
            ]
        )

        ideal_cont_vec = (
            FROZEN_REFERENCE[
                "X_val_cont"
            ][
                i
            ]
        )

        X_qpu.append(
            qpu_vec
        )

        row = {
            "validation_row": int(
                i
            ),
            "endpoint": (
                endpoint
            ),
            "target": float(
                FROZEN_REFERENCE[
                    "y_val"
                ][
                    i
                ]
            ),
            "pred_full_CONT_ideal": float(
                FROZEN_REFERENCE[
                    "pred_cont"
                ][
                    i
                ]
            ),
            "pred_RWP64_ideal": float(
                FROZEN_REFERENCE[
                    "pred_rwp"
                ][
                    i
                ]
            ),
        }

        for j, feature in enumerate(
            FEATURES
        ):
            row[
                f"{feature}_CONT_ideal"
            ] = float(
                ideal_cont_vec[
                    j
                ]
            )

            row[
                f"{feature}_RWP64_ideal"
            ] = float(
                ideal_rwp_vec[
                    j
                ]
            )

            row[
                f"{feature}_qpu"
            ] = float(
                qpu_vec[
                    j
                ]
            )

            row[
                f"{feature}_qpu_minus_RWP64"
            ] = float(
                qpu_vec[
                    j
                ]
                - ideal_rwp_vec[
                    j
                ]
            )

        feature_rows.append(
            row
        )

    X_qpu = np.asarray(
        X_qpu,
        dtype=float,
    )

    pred_qpu = common.predict_scaled_ridge(
        FROZEN_REFERENCE[
            "model"
        ],
        FROZEN_REFERENCE[
            "scaler"
        ],
        FROZEN_REFERENCE[
            "keep"
        ],
        X_qpu,
    )

    features_df = pd.DataFrame(
        feature_rows
    )

    features_df[
        "pred_qpu"
    ] = pred_qpu

    targets = np.asarray(
        FROZEN_REFERENCE[
            "y_val"
        ],
        dtype=float,
    )

    pred_rwp = np.asarray(
        FROZEN_REFERENCE[
            "pred_rwp"
        ],
        dtype=float,
    )

    pred_cont = np.asarray(
        FROZEN_REFERENCE[
            "pred_cont"
        ],
        dtype=float,
    )

    features_df[
        "sq_error_qpu"
    ] = (
        targets
        - pred_qpu
    ) ** 2

    features_df[
        "sq_error_RWP64_ideal"
    ] = (
        targets
        - pred_rwp
    ) ** 2

    features_df[
        "sq_error_full_CONT_ideal"
    ] = (
        targets
        - pred_cont
    ) ** 2

    features_df.to_csv(
        OUT_FEATURES,
        index=False,
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_FEATURES.name,
        features_df,
    )

    feature_diff = (
        X_qpu
        - FROZEN_REFERENCE[
            "X_val_rwp"
        ]
    )

    service_usage_after = get_service_usage(
        service
    )

    summary = {
        "run_uuid": (
            run_uuid
        ),
        "candidate_key": (
            EXPECTED_CANDIDATE
        ),
        "protocol": (
            "RWP_from_CONT"
        ),
        "K_RWP": (
            EXPECTED_K
        ),
        "backend": (
            args.backend
        ),
        "layout": (
            layout
        ),
        "memory_q4_physical": (
            chosen[
                "memory_q4_physical"
            ]
        ),
        "memory_q5_physical": (
            chosen[
                "memory_q5_physical"
            ]
        ),
        "memory_worst_t_over_t1": (
            chosen[
                "actual_worst_memory_duration_over_t1"
            ]
        ),
        "memory_worst_t_over_t2": (
            chosen[
                "actual_worst_memory_duration_over_t2"
            ]
        ),
        "memory_min_coherence_margin": (
            chosen[
                "actual_min_memory_coherence_margin"
            ]
        ),
        "job_id": (
            job_id
        ),
        "shots_per_setting": int(
            args.shots
        ),
        "n_validation_endpoints": 365,
        "n_circuits": 730,
        "qpu_rmse": rmse(
            targets,
            pred_qpu,
        ),
        "qpu_mae": mae(
            targets,
            pred_qpu,
        ),
        "qpu_bias": bias(
            targets,
            pred_qpu,
        ),
        "ideal_RWP64_rmse": rmse(
            targets,
            pred_rwp,
        ),
        "ideal_RWP64_mae": mae(
            targets,
            pred_rwp,
        ),
        "ideal_RWP64_bias": bias(
            targets,
            pred_rwp,
        ),
        "full_CONT_ideal_rmse": rmse(
            targets,
            pred_cont,
        ),
        "full_CONT_ideal_mae": mae(
            targets,
            pred_cont,
        ),
        "full_CONT_ideal_bias": bias(
            targets,
            pred_cont,
        ),
        "qpu_minus_ideal_RWP64_rmse_delta": (
            rmse(
                targets,
                pred_qpu,
            )
            - rmse(
                targets,
                pred_rwp,
            )
        ),
        "qpu_feature_mae_vs_RWP64": float(
            np.mean(
                np.abs(
                    feature_diff
                )
            )
        ),
        "qpu_feature_rmse_vs_RWP64": float(
            np.sqrt(
                np.mean(
                    feature_diff ** 2
                )
            )
        ),
        "prediction_correlation_qpu_vs_RWP64": float(
            np.corrcoef(
                pred_qpu,
                pred_rwp,
            )[
                0,
                1
            ]
        ),
        "prediction_correlation_qpu_vs_CONT": float(
            np.corrcoef(
                pred_qpu,
                pred_cont,
            )[
                0,
                1
            ]
        ),
        "scheduled_duration_times_shots_s": (
            scheduled_shot_seconds
        ),
        "empirical_usage_estimate_s": (
            empirical_usage_estimate
        ),
        "qpu_charge_time_seconds": (
            qpu_timing[
                "qpu_charge_time_seconds"
            ]
        ),
        "qpu_circuits_execution_time_seconds": (
            qpu_timing[
                "qpu_circuits_execution_time_seconds"
            ]
        ),
        "qpu_running_wall_seconds": (
            qpu_timing[
                "qpu_running_wall_seconds"
            ]
        ),
        "ibm_job_created_at_utc": (
            qpu_timing[
                "ibm_job_created_at_utc"
            ]
        ),
        "ibm_job_running_at_utc": (
            qpu_timing[
                "ibm_job_running_at_utc"
            ]
        ),
        "ibm_job_finished_at_utc": (
            qpu_timing[
                "ibm_job_finished_at_utc"
            ]
        ),
        "job_metrics": (
            metrics
        ),
        "service_usage_before": (
            usage_before
        ),
        "service_usage_after": (
            service_usage_after
        ),
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

    db.save_json_artifact(
        run_uuid,
        OUT_SUMMARY.name,
        summary,
    )

    db.update_run(
        run_uuid,
        run_status=(
            "COMPLETED"
        ),
        completed_at_utc=(
            utc_now_naive()
        ),
        job_status=(
            final_job_status
        ),
        job_usage_seconds=(
            qpu_timing[
                "qpu_charge_time_seconds"
            ]
        ),
        qpu_charge_time_seconds=(
            qpu_timing[
                "qpu_charge_time_seconds"
            ]
        ),
        qpu_circuits_execution_time_seconds=(
            qpu_timing[
                "qpu_circuits_execution_time_seconds"
            ]
        ),
        qpu_running_wall_seconds=(
            qpu_timing[
                "qpu_running_wall_seconds"
            ]
        ),
        ibm_job_created_at_utc=(
            qpu_timing[
                "ibm_job_created_at_utc"
            ]
        ),
        ibm_job_running_at_utc=(
            qpu_timing[
                "ibm_job_running_at_utc"
            ]
        ),
        ibm_job_finished_at_utc=(
            qpu_timing[
                "ibm_job_finished_at_utc"
            ]
        ),
        qpu_rmse=float(
            summary[
                "qpu_rmse"
            ]
        ),
        qpu_mae=float(
            summary[
                "qpu_mae"
            ]
        ),
        qpu_bias=float(
            summary[
                "qpu_bias"
            ]
        ),
        ideal_same_subset_rmse=float(
            summary[
                "ideal_RWP64_rmse"
            ]
        ),
        ideal_same_subset_mae=float(
            summary[
                "ideal_RWP64_mae"
            ]
        ),
        ideal_same_subset_bias=float(
            summary[
                "ideal_RWP64_bias"
            ]
        ),
        qpu_feature_mae=float(
            summary[
                "qpu_feature_mae_vs_RWP64"
            ]
        ),
        qpu_feature_rmse=float(
            summary[
                "qpu_feature_rmse_vs_RWP64"
            ]
        ),
        prediction_correlation_qpu_vs_ideal=float(
            summary[
                "prediction_correlation_qpu_vs_RWP64"
            ]
        ),
        service_usage_after_json=(
            service_usage_after
        ),
    )

    print()
    print(
        "=" * 126
    )
    print(
        "REAL-QPU CANDIDATE #5 FULL 365 RESULTS"
    )
    print(
        "=" * 126
    )
    print(
        f"QPU RMSE:                      "
        f"{summary['qpu_rmse']:.6f}"
    )
    print(
        f"QPU MAE:                       "
        f"{summary['qpu_mae']:.6f}"
    )
    print(
        f"QPU bias:                      "
        f"{summary['qpu_bias']:+.6f}"
    )
    print(
        f"Ideal RWP64 RMSE:              "
        f"{summary['ideal_RWP64_rmse']:.6f}"
    )
    print(
        f"Full CONT ideal RMSE:          "
        f"{summary['full_CONT_ideal_rmse']:.6f}"
    )
    print(
        f"QPU - ideal RWP64 RMSE delta:  "
        f"{summary['qpu_minus_ideal_RWP64_rmse_delta']:+.6f}"
    )
    print(
        f"QPU feature MAE vs RWP64:      "
        f"{summary['qpu_feature_mae_vs_RWP64']:.6f}"
    )
    print(
        f"QPU feature RMSE vs RWP64:     "
        f"{summary['qpu_feature_rmse_vs_RWP64']:.6f}"
    )
    print(
        f"QPU/RWP64 prediction corr:     "
        f"{summary['prediction_correlation_qpu_vs_RWP64']:.6f}"
    )

    if summary[
        "qpu_charge_time_seconds"
    ] is not None:
        print(
            f"QPU charge / locked time:      "
            f"{summary['qpu_charge_time_seconds']:.6f} s"
        )

    if summary[
        "qpu_circuits_execution_time_seconds"
    ] is not None:
        print(
            f"Actual circuit execution QPU:  "
            f"{summary['qpu_circuits_execution_time_seconds']:.6f} s"
        )

    print()
    print(
        "2026 remains FROZEN / UNUSED."
    )

    print()
    print(
        "Saved:"
    )

    for p in [
        OUT_ALL_EMBEDDINGS,
        OUT_TEST_LAYOUTS,
        OUT_TEST_COMPILED,
        OUT_TEST_QUBITS,
        OUT_TEST_SUMMARY,
        OUT_RESOURCES,
        OUT_ENDPOINTS,
        OUT_PREFLIGHT,
        OUT_JOB,
        OUT_RAW_COUNTS,
        OUT_FEATURES,
        OUT_SUMMARY,
    ]:
        print(
            f"  {p}"
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        if CURRENT_RUN_UUID is not None:
            try:
                db.update_run(
                    CURRENT_RUN_UUID,
                    run_status=(
                        "FAILED"
                    ),
                    completed_at_utc=(
                        utc_now_naive()
                    ),
                    error_message=str(
                        exc
                    ),
                )

                print(
                    f"[database] Run {CURRENT_RUN_UUID} marked FAILED."
                )
            except Exception as db_exc:
                print(
                    "[database] WARNING: could not mark failed run: "
                    f"{db_exc}"
                )

        raise
