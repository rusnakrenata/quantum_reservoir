from __future__ import annotations

"""
WEEK 11.1B-EST
RWP_H3_R4 on real IBM hardware using EstimatorV2 expectation-value output.

IMPORTANT
---------
This is NOT an exact/no-shot expectation value calculation.

On a real QPU, IBM EstimatorV2 still estimates expectation values statistically.
The difference from the original SamplerV2 run is the interface:

    SamplerV2:
        raw sampled bitstrings/counts -> we reconstruct <O>

    EstimatorV2:
        circuit + observable -> IBM returns estimated <O> directly
        together with a standard error.

For a truly exact expectation value, use an ideal/noisy simulator.

Frozen candidate
----------------
    RWP_H3_R4
    H3, W=4, r=3
    alpha=0.25
    readout=XZinj_dropX3

Observables
-----------
    X0, X1, X2, Z0, Z1, Z2, Z3, Y4X5

Scientific protocol
-------------------
    2022-2024 : establish/freeze Ridge readout only
    2025      : QPU validation / hardware-transfer comparison
    2026      : FROZEN / NEVER LOADED

Default behavior is DRY RUN.
Use --submit only after reviewing the preflight.

The default Estimator target precision is chosen as:

    1/sqrt(2048) ~= 0.022097

to be roughly comparable to the statistical scale of the original
2048-shot Sampler run.  This does NOT mean Estimator uses exactly 2048 shots.
"""

import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2

from ibm_account import get_service
import db_objects as qpu_db


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
SELECTOR_FILE = HERE / "11_0A_fresh_full_hardware_reselection.py"
MANIFEST_FILE = RESULTS / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"

CANDIDATE_KEY = "RWP_H3_R4"
EXPECTED_TOPOLOGY = "H3"
EXPECTED_WINDOW = 2
EXPECTED_R = 3
EXPECTED_READOUT = "XZinj_dropX3"
EXPECTED_ALPHA = 0.25

DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_EQUIVALENT_SHOTS = 2048
DEFAULT_PRECISION = 1.0 / math.sqrt(DEFAULT_EQUIVALENT_SHOTS)
DEFAULT_PILOT_SIZE = 64

OPT_LEVEL = 1
SEED_TRANSPILE = 42
SHORTLIST = 120

FEATURES = ['X0', 'X1', 'X2', 'Z0', 'Z1', 'Z2', 'Z3']

CURRENT_RUN_UUID = None


# =============================================================================
# General helpers
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(mod)

    return mod


common = load_module(
    COMMON_FILE,
    "qrc_111a2_common",
)

selector = load_module(
    SELECTOR_FILE,
    "qrc_111a2_selector",
)


def json_safe(x):
    if isinstance(x, dict):
        return {
            str(k): json_safe(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [
            json_safe(v)
            for v in x
        ]

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, pd.Timestamp):
        return str(x)

    if isinstance(x, datetime):
        return x.isoformat()

    if isinstance(x, float) and (
        math.isnan(x) or math.isinf(x)
    ):
        return None

    return x


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


def utc_now_naive():
    return (
        datetime.now(timezone.utc)
        .replace(tzinfo=None)
    )


def get_service_usage(service):
    """
    Best-effort IBM usage lookup.
    Failure must not invalidate the scientific run.
    """
    for attr in [
        "usage",
        "usage_info",
    ]:
        fn = getattr(
            service,
            attr,
            None,
        )

        if callable(fn):
            try:
                out = fn()

                if hasattr(
                    out,
                    "to_dict",
                ):
                    out = out.to_dict()

                return json_safe(out)

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
            .replace(tzinfo=None)
        )

    except Exception:
        return None


def get_job_usage_seconds(job):
    fn = getattr(
        job,
        "usage",
        None,
    )

    if not callable(fn):
        return None

    try:
        value = fn()

        if value is None:
            return None

        if isinstance(
            value,
            dict,
        ):
            for key in [
                "qpu_charge_time_seconds",
                "quantum_seconds",
                "qpu_seconds",
                "usage_seconds",
                "seconds",
            ]:
                if value.get(key) is not None:
                    return float(
                        value[key]
                    )

            return None

        return float(value)

    except Exception:
        return None


def extract_ibm_qpu_timing(
    job,
    metrics,
):
    metrics = metrics or {}

    usage = (
        metrics.get("usage")
        or {}
    )

    timestamps = (
        metrics.get("timestamps")
        or {}
    )

    qpu_charge = None

    for key in [
        "qpu_charge_time_seconds",
        "quantum_seconds",
        "qpu_seconds",
        "usage_seconds",
        "seconds",
    ]:
        if usage.get(key) is not None:
            try:
                qpu_charge = float(
                    usage[key]
                )
                break
            except Exception:
                pass

    if qpu_charge is None:
        qpu_charge = (
            get_job_usage_seconds(
                job
            )
        )

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
            qpu_circuit_seconds = (
                float(execution_ns)
                / 1e9
            )
        except Exception:
            pass

    created_at = (
        _parse_ibm_utc_timestamp(
            timestamps.get(
                "created"
            )
        )
    )

    running_at = (
        _parse_ibm_utc_timestamp(
            timestamps.get(
                "running"
            )
        )
    )

    finished_at = (
        _parse_ibm_utc_timestamp(
            timestamps.get(
                "finished"
            )
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
        "qpu_charge_time_seconds":
            qpu_charge,

        "qpu_circuits_execution_time_seconds":
            qpu_circuit_seconds,

        "qpu_running_wall_seconds":
            running_wall_seconds,

        "ibm_job_created_at_utc":
            created_at,

        "ibm_job_running_at_utc":
            running_at,

        "ibm_job_finished_at_utc":
            finished_at,
    }


# =============================================================================
# Candidate / frozen classical readout
# =============================================================================

def load_candidate():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"{MANIFEST_FILE} not found. "
            "Use the final enriched Week-10 manifest."
        )

    df = pd.read_csv(
        MANIFEST_FILE
    )

    sub = df[
        df[
            "candidate_key"
        ].astype(str)
        == CANDIDATE_KEY
    ].copy()

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected exactly one row "
            f"for {CANDIDATE_KEY}, "
            f"found {len(sub)}."
        )

    row = sub.iloc[0]

    if (
        str(
            row["protocol"]
        ).upper()
        != "RWP"
    ):
        raise RuntimeError(
            "Frozen winner is expected "
            "to be RWP."
        )

    if (
        str(
            row["topology"]
        )
        != EXPECTED_TOPOLOGY
    ):
        raise RuntimeError(
            "Unexpected topology "
            "in enriched manifest."
        )

    if (
        int(
            row["window"]
        )
        != EXPECTED_WINDOW
    ):
        raise RuntimeError(
            "Unexpected W "
            "in enriched manifest."
        )

    if (
        int(
            row["original_r"]
        )
        != EXPECTED_R
    ):
        raise RuntimeError(
            "Unexpected r "
            "in enriched manifest."
        )

    if (
        str(
            row["test_readout"]
        )
        != EXPECTED_READOUT
    ):
        raise RuntimeError(
            "Unexpected readout "
            "in enriched manifest."
        )

    if not np.isclose(
        float(
            row["alpha"]
        ),
        EXPECTED_ALPHA,
    ):
        raise RuntimeError(
            "Unexpected alpha "
            "in enriched manifest."
        )

    raw_j = json.loads(
        str(
            row["J_json"]
        )
    )

    J = {}

    for edge in common.TOPOLOGY_EDGES[
        EXPECTED_TOPOLOGY
    ]:
        i, j = edge

        key = f"J{i}{j}"
        rev = f"J{j}{i}"

        if key in raw_j:
            val = raw_j[key]

        elif rev in raw_j:
            val = raw_j[rev]

        else:
            raise KeyError(
                f"Missing coupling "
                f"for edge {edge} "
                "in J_json."
            )

        J[edge] = float(val)

    candidate = {
        "candidate_id":
            CANDIDATE_KEY,

        "topology":
            EXPECTED_TOPOLOGY,

        "window":
            EXPECTED_WINDOW,

        "r":
            EXPECTED_R,

        "alpha":
            float(
                row["alpha"]
            ),

        "dt":
            float(
                row["dt"]
            ),

        "hx":
            float(
                row["hx"]
            ),

        "hy":
            float(
                row["hy"]
            ),

        "J":
            J,
    }

    return (
        candidate,
        row.to_dict(),
    )


def build_frozen_readout(
    candidate,
    meta,
):
    """
    Rebuild the exact ideal 2022-2025 RWP feature bank.

    Ridge selection/fitting is done ONLY on 2022-2024.
    2025 is validation only.
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
            candidate["alpha"]
        ),
    )

    A_list = common.build_channels(
        candidate,
        angles,
    )

    (
        endpoints,
        master,
    ) = common.rwp_master_feature_bank_from_channels(
        A_list,
        EXPECTED_WINDOW,
    )

    X = common.select_features(
        master,
        EXPECTED_READOUT,
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

    X_train = X[
        train_mask
    ]

    y_train = y_endpoint[
        train_mask
    ]

    X_val = X[
        val_mask
    ]

    y_val = y_endpoint[
        val_mask
    ]

    val_endpoints = endpoints[
        val_mask
    ]

    (
        selected_lambda,
        _,
        cv_summary,
    ) = common.select_lambda_training_only(
        X_train,
        y_train,
        n_splits=5,
    )

    expected_lambda = float(
        meta[
            "source_selected_lambda"
        ]
    )

    expected_cv = float(
        meta[
            "source_readout_cv_rmse"
        ]
    )

    if not np.isclose(
        selected_lambda,
        expected_lambda,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Lambda audit failed: "
            f"recomputed="
            f"{selected_lambda}, "
            f"manifest="
            f"{expected_lambda}."
        )

    cv_rmse = float(
        cv_summary.iloc[0][
            "cv_rmse_mean"
        ]
    )

    if abs(
        cv_rmse
        - expected_cv
    ) > 1e-5:
        raise RuntimeError(
            "CV audit failed: "
            f"recomputed="
            f"{cv_rmse:.9f}, "
            f"manifest="
            f"{expected_cv:.9f}."
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

    pred_val = (
        common.predict_scaled_ridge(
            model,
            scaler,
            keep,
            X_val,
        )
    )

    ideal_rmse = rmse(
        y_val,
        pred_val,
    )

    ideal_mae = mae(
        y_val,
        pred_val,
    )

    ideal_bias = bias(
        y_val,
        pred_val,
    )

    expected_val = float(
        meta[
            "source_validation_rmse"
        ]
    )

    if abs(
        ideal_rmse
        - expected_val
    ) > 1e-5:
        raise RuntimeError(
            "Ideal validation audit failed: "
            f"recomputed="
            f"{ideal_rmse:.9f}, "
            f"manifest="
            f"{expected_val:.9f}."
        )

    return {
        "work_tv":
            work_tv,

        "cols":
            cols,

        "y_all":
            y_all,

        "angles":
            angles,

        "endpoints":
            endpoints,

        "master":
            master,

        "X":
            X,

        "y_endpoint":
            y_endpoint,

        "train_mask":
            train_mask,

        "val_mask":
            val_mask,

        "val_endpoints":
            val_endpoints,

        "X_val":
            X_val,

        "y_val":
            y_val,

        "pred_val":
            pred_val,

        "model":
            model,

        "scaler":
            scaler,

        "keep":
            keep,

        "lambda":
            selected_lambda,

        "cv_rmse":
            cv_rmse,

        "ideal_rmse":
            ideal_rmse,

        "ideal_mae":
            ideal_mae,

        "ideal_bias":
            ideal_bias,
    }


# =============================================================================
# Quantum circuit
# =============================================================================

def append_rwp_step(
    qc,
    candidate,
    angle_row,
    first_step,
):
    """
    One RWP input step.

    q0..q3 are reset between replay inputs.
    q4,q5 retain memory inside the W=4 window.
    """
    if not first_step:
        for q in range(4):
            qc.reset(q)

    for q in range(4):
        qc.ry(
            float(
                angle_row[q]
            ),
            q,
        )

    r = int(
        candidate["r"]
    )

    dt = float(
        candidate["dt"]
    )

    hx = float(
        candidate["hx"]
    )

    hy = float(
        candidate["hy"]
    )

    J = candidate[
        "J"
    ]

    for _ in range(r):
        for (
            i,
            j,
        ) in common.TOPOLOGY_EDGES[
            candidate["topology"]
        ]:
            qc.rzz(
                2.0
                * float(
                    J[(i, j)]
                )
                * dt
                / r,
                i,
                j,
            )

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


def build_state_circuit(
    candidate,
    all_angles,
    endpoint,
):
    """
    Build the W=4 state-preparation circuit only.

    No measurement basis rotations and no classical measurements are added.
    EstimatorV2 receives the observables separately.
    """
    qc = QuantumCircuit(
        6,
        name=(
            f"{candidate['topology']}_W{EXPECTED_WINDOW}_estimator_"
            f"e{int(endpoint)}"
        ),
    )

    start = (
        int(endpoint)
        - EXPECTED_WINDOW
        + 1
    )

    if start < 0:
        raise ValueError(
            "Endpoint does not have "
            f"enough W={EXPECTED_WINDOW} history."
        )

    for (
        k,
        idx,
    ) in enumerate(
        range(
            start,
            int(endpoint) + 1,
        )
    ):
        append_rwp_step(
            qc,
            candidate,
            all_angles[idx],
            first_step=(
                k == 0
            ),
        )

    return qc


def logical_observables():
    """
    Return logical observables in exactly the feature order expected by the
    frozen Ridge readout.
    """
    return [
        SparsePauliOp.from_sparse_list(
            [("X", [0], 1.0)],
            num_qubits=6,
        ),
        SparsePauliOp.from_sparse_list(
            [("X", [1], 1.0)],
            num_qubits=6,
        ),
        SparsePauliOp.from_sparse_list(
            [("X", [2], 1.0)],
            num_qubits=6,
        ),
        SparsePauliOp.from_sparse_list(
            [("Z", [0], 1.0)],
            num_qubits=6,
        ),
        SparsePauliOp.from_sparse_list(
            [("Z", [1], 1.0)],
            num_qubits=6,
        ),
        SparsePauliOp.from_sparse_list(
            [("Z", [2], 1.0)],
            num_qubits=6,
        ),
        SparsePauliOp.from_sparse_list(
            [("Z", [3], 1.0)],
            num_qubits=6,
        ),
    ]


def resource_row(
    circuit,
    backend,
    endpoint,
):
    """
    Resource count for one transpiled state-preparation circuit.

    Estimator may internally create multiple measurement configurations for the
    observables, so this row intentionally describes the ISA state-preparation
    circuit itself, not an inferred total measurement overhead.
    """
    ops = {
        str(k):
            int(v)
        for (
            k,
            v,
        ) in circuit.count_ops().items()
    }

    duration_s = float(
        circuit.estimate_duration(
            backend.target,
            unit="s",
        )
    )

    return {
        "endpoint":
            int(endpoint),

        "depth":
            int(
                circuit.depth()
            ),

        "size":
            int(
                circuit.size()
            ),

        "n_cz":
            int(
                ops.get(
                    "cz",
                    0,
                )
            ),

        "n_swap":
            int(
                ops.get(
                    "swap",
                    0,
                )
            ),

        "n_reset":
            int(
                ops.get(
                    "reset",
                    0,
                )
            ),

        "duration_us":
            duration_s
            * 1e6,

        "operations":
            json.dumps(
                ops,
                sort_keys=True,
            ),
    }


def choose_validation_endpoints(
    val_endpoints,
    full,
    pilot_size,
):
    val_endpoints = np.asarray(
        val_endpoints,
        dtype=int,
    )

    if full:
        return (
            val_endpoints.copy()
        )

    n = max(
        1,
        min(
            int(
                pilot_size
            ),
            len(
                val_endpoints
            ),
        ),
    )

    offsets = np.unique(
        np.linspace(
            0,
            len(
                val_endpoints
            ) - 1,
            n,
            dtype=int,
        )
    )

    return val_endpoints[
        offsets
    ]


# =============================================================================
# Main
# =============================================================================

def main():
    global CURRENT_RUN_UUID

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=(
            "IBM backend for the "
            "EstimatorV2 hardware run."
        ),
    )

    parser.add_argument(
        "--precision",
        type=float,
        default=DEFAULT_PRECISION,
        help=(
            "Estimator target precision. "
            "Default is 1/sqrt(2048) "
            f"= {DEFAULT_PRECISION:.9f}."
        ),
    )

    parser.add_argument(
        "--pilot-size",
        type=int,
        default=DEFAULT_PILOT_SIZE,
        help=(
            "Number of evenly spread "
            "2025 validation endpoints "
            "for a pilot."
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Use all 365 validation "
            "endpoints."
        ),
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Actually submit the "
            "EstimatorV2 job. "
            "Default is dry-run."
        ),
    )

    args = parser.parse_args()

    if args.precision <= 0:
        raise ValueError(
            "--precision must be > 0."
        )

    print("=" * 126)
    print(
        "WEEK 11.1B-EST — CANDIDATE #1 "
        "REAL-QPU ESTIMATOR EXPECTATION VALUES"
    )
    print("=" * 126)

    print(
        "2022-2024 = "
        "establish/freeze readout"
    )

    print(
        "2025      = "
        "validation / hardware-transfer test"
    )

    print(
        "2026      = "
        "FROZEN / NOT LOADED"
    )

    print()

    print(
        "IMPORTANT: EstimatorV2 on a real QPU "
        "returns statistical expectation-value estimates, "
        "not mathematically exact values."
    )

    print()

    candidate, manifest_meta = (
        load_candidate()
    )

    print(
        "Frozen candidate:"
    )

    print(
        f"  candidate="
        f"{CANDIDATE_KEY}"
    )

    print(
        f"  topology="
        f"{candidate['topology']}"
    )

    print(
        f"  W="
        f"{candidate['window']}"
    )

    print(
        f"  r="
        f"{candidate['r']}"
    )

    print(
        f"  alpha="
        f"{candidate['alpha']}"
    )

    print(
        f"  dt="
        f"{candidate['dt']}"
    )

    print(
        f"  hx="
        f"{candidate['hx']:+.12f}"
    )

    print(
        f"  hy="
        f"{candidate['hy']:+.12f}"
    )

    print(
        f"  readout="
        f"{EXPECTED_READOUT}"
    )

    print(
        f"  primitive="
        "EstimatorV2"
    )

    print(
        f"  target precision="
        f"{args.precision:.9f}"
    )

    print(
        "  approximate reference scale="
        f"1/sqrt({DEFAULT_EQUIVALENT_SHOTS})"
    )

    print(
        "  J="
        + json.dumps(
            {
                f"J{i}{j}":
                    v
                for (
                    i,
                    j,
                ), v
                in candidate[
                    "J"
                ].items()
            },
            sort_keys=True,
        )
    )

    print()

    frozen = (
        build_frozen_readout(
            candidate,
            manifest_meta,
        )
    )

    print(
        "Training-only readout audit:"
    )

    print(
        f"  lambda="
        f"{frozen['lambda']}"
    )

    print(
        f"  CV RMSE="
        f"{frozen['cv_rmse']:.6f}"
    )

    print()

    print(
        "Ideal 2025 reference:"
    )

    print(
        f"  RMSE="
        f"{frozen['ideal_rmse']:.6f}"
    )

    print(
        f"  MAE ="
        f"{frozen['ideal_mae']:.6f}"
    )

    print(
        f"  bias="
        f"{frozen['ideal_bias']:+.6f}"
    )

    print()

    selected_endpoints = (
        choose_validation_endpoints(
            frozen[
                "val_endpoints"
            ],
            full=args.full,
            pilot_size=args.pilot_size,
        )
    )

    mode_name = (
        "FULL 2025"
        if args.full
        else (
            f"PILOT "
            f"{len(selected_endpoints)}"
        )
    )

    print(
        f"QPU evaluation mode: "
        f"{mode_name}"
    )

    print(
        "Validation endpoints selected: "
        f"{len(selected_endpoints)} "
        "of "
        f"{len(frozen['val_endpoints'])}"
    )

    print()

    # ---------------------------------------------------------------------
    # Create DB run BEFORE using hardware.
    # ---------------------------------------------------------------------
    qpu_db.create_all_tables()

    run_uuid = qpu_db.create_run(
        script_name=Path(
            __file__
        ).name,

        run_status=
            "PREPARING",

        started_at_utc=
            utc_now_naive(),

        forecast_dataset_name=
            "property_damage_next_day_v1",

        feature_set_name=
            "F4",

        evaluation_split=
            "validation_2025",

        candidate_key=
            CANDIDATE_KEY,

        selected_by_rules=
            str(
                manifest_meta.get(
                    "selected_by_rules",
                    "Rule4",
                )
            ),

        protocol=
            "RWP",

        topology=
            EXPECTED_TOPOLOGY,

        window_size=
            EXPECTED_WINDOW,

        trotter_r=
            EXPECTED_R,

        readout_name=
            EXPECTED_READOUT,

        primitive_name=
            "EstimatorV2",

        measurement_method=
            "direct_expectation_value_estimation",

        estimator_precision=
            float(
                args.precision
            ),

        resilience_level=
            0,

        alpha=
            float(
                candidate[
                    "alpha"
                ]
            ),

        dt=
            float(
                candidate[
                    "dt"
                ]
            ),

        hx=
            float(
                candidate[
                    "hx"
                ]
            ),

        hy=
            float(
                candidate[
                    "hy"
                ]
            ),

        j_json={
            f"J{i}{j}":
                float(v)
            for (
                i,
                j,
            ), v
            in candidate[
                "J"
            ].items()
        },

        manifest_json=
            manifest_meta,

        ridge_lambda=
            float(
                frozen[
                    "lambda"
                ]
            ),

        training_cv_rmse=
            float(
                frozen[
                    "cv_rmse"
                ]
            ),

        ideal_validation_rmse_full_2025=
            float(
                frozen[
                    "ideal_rmse"
                ]
            ),

        ideal_validation_mae_full_2025=
            float(
                frozen[
                    "ideal_mae"
                ]
            ),

        ideal_validation_bias_full_2025=
            float(
                frozen[
                    "ideal_bias"
                ]
            ),

        backend_name=
            args.backend,

        shots_per_setting=
            None,

        n_validation_endpoints=
            int(
                len(
                    selected_endpoints
                )
            ),

        full_2025=
            bool(
                args.full
            ),

        pilot_selection=(
            "all_2025_validation"
            if args.full
            else (
                "deterministic_evenly_"
                "spaced_2025_endpoints"
            )
        ),

        optimization_level=
            OPT_LEVEL,

        seed_transpiler=
            SEED_TRANSPILE,

        notes=(
            "Candidate #1 repeated with "
            "IBM EstimatorV2 expectation-value "
            "output. Resilience level 0; "
            "2026 not loaded."
        ),
    )

    CURRENT_RUN_UUID = (
        run_uuid
    )

    print(
        f"[database] "
        f"QPU run UUID: "
        f"{run_uuid}"
    )

    print()

    service = (
        get_service()
    )

    print("=" * 126)
    print(
        f"FRESH {args.backend} "
        "H3 HARDWARE RESELECTION"
    )
    print("=" * 126)

    fresh = (
        selector.fresh_hardware_reselection(
            service=service,
            candidate=candidate,
            backend_names=[
                args.backend
            ],
            shortlist=
                SHORTLIST,
            write_prefix=(
                f"11_1B_EST_"
                f"{CANDIDATE_KEY}_"
                f"{args.backend}"
            ),
            verbose=True,
        )
    )

    chosen = fresh[
        "selected"
    ]

    layout = json.loads(
        chosen[
            "layout"
        ]
    )

    print()
    print(
        "Fresh physical choice:"
    )

    print(
        f"  backend="
        f"{args.backend}"
    )

    print(
        f"  layout="
        f"{layout}"
    )

    print(
        "  max CZ error="
        f"{float(chosen.get('compiled_2q_error_max_percent', np.nan)):.6f}%"
    )

    print(
        "  max readout error="
        f"{float(chosen.get('max_readout_error_percent', np.nan)):.6f}%"
    )

    print(
        "  max 1Q error="
        f"{float(chosen.get('compiled_1q_error_max_percent', np.nan)):.6f}%"
    )

    print(
        f"  min T1="
        f"{float(chosen.get('min_t1_us', np.nan)):.3f} us"
    )

    print(
        f"  min T2="
        f"{float(chosen.get('min_t2_us', np.nan)):.3f} us"
    )

    print()

    backend = service.backend(
        args.backend,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    logical_obs = (
        logical_observables()
    )

    pubs = []
    resource_rows = []
    compiled_circuits = []

    for endpoint in selected_endpoints:
        logical = (
            build_state_circuit(
                candidate,
                frozen[
                    "angles"
                ],
                int(
                    endpoint
                ),
            )
        )

        isa = transpile(
            logical,
            backend=backend,
            initial_layout=
                layout,
            routing_method=
                "none",
            optimization_level=
                OPT_LEVEL,
            seed_transpiler=
                SEED_TRANSPILE,
            scheduling_method=
                "alap",
        )

        backend.check_faulty(
            isa
        )

        rr = resource_row(
            isa,
            backend,
            endpoint=int(
                endpoint
            ),
        )

        if rr[
            "n_swap"
        ] != 0:
            raise RuntimeError(
                "Strict native compile failed: "
                "SWAP detected at "
                f"endpoint={endpoint}."
            )

        isa_observables = [
            obs.apply_layout(
                isa.layout
            )
            for obs
            in logical_obs
        ]

        pubs.append(
            (
                isa,
                isa_observables,
            )
        )

        compiled_circuits.append(
            isa
        )

        resource_rows.append(
            rr
        )

    resources = pd.DataFrame(
        resource_rows
    )

    resources_path = (
        RESULTS
        / "11_1B_estimator_qpu_resources.csv"
    )

    resources.to_csv(
        resources_path,
        index=False,
    )

    qpu_db.save_dataframe_artifact(
        run_uuid,
        resources_path.name,
        resources,
    )

    usage_before = (
        get_service_usage(
            service
        )
    )

    print("=" * 126)
    print(
        "ESTIMATOR QPU PREFLIGHT"
    )
    print("=" * 126)

    print(
        f"Backend:                    "
        f"{args.backend}"
    )

    print(
        f"Physical layout:            "
        f"{layout}"
    )

    print(
        f"Validation endpoints:       "
        f"{len(selected_endpoints)}"
    )

    print(
        f"Estimator PUBs:             "
        f"{len(pubs)}"
    )

    print(
        f"Observables/PUB:            "
        f"{len(FEATURES)}"
    )

    print(
        f"Target precision:           "
        f"{args.precision:.9f}"
    )

    print(
        "Resilience level:          0 "
        "(raw / no built-in mitigation)"
    )

    print(
        "Median state-prep CZ:      "
        f"{resources['n_cz'].median():.0f}"
    )

    print(
        "Median state-prep depth:   "
        f"{resources['depth'].median():.0f}"
    )

    print(
        "Median state-prep duration:"
        f" {resources['duration_us'].median():.3f} us"
    )

    print()

    print(
        "NOTE: Estimator internally handles "
        "measurement configurations. "
        "The resource values above describe "
        "the transpiled state-preparation circuit, "
        "not total hidden measurement overhead."
    )

    if usage_before is not None:
        print()
        print(
            "Service usage before:"
        )

        print(
            json.dumps(
                usage_before,
                indent=2,
            )
        )

    preflight = {
        "candidate_key":
            CANDIDATE_KEY,

        "primitive":
            "EstimatorV2",

        "measurement_method":
            "direct_expectation_value_estimation",

        "backend":
            args.backend,

        "layout":
            layout,

        "window":
            EXPECTED_WINDOW,

        "r":
            EXPECTED_R,

        "precision":
            float(
                args.precision
            ),

        "resilience_level":
            0,

        "full_2025":
            bool(
                args.full
            ),

        "n_endpoints":
            int(
                len(
                    selected_endpoints
                )
            ),

        "n_pubs":
            int(
                len(
                    pubs
                )
            ),

        "n_observables_per_pub":
            int(
                len(
                    FEATURES
                )
            ),

        "ideal_validation_rmse_full_2025":
            float(
                frozen[
                    "ideal_rmse"
                ]
            ),

        "training_cv_rmse":
            float(
                frozen[
                    "cv_rmse"
                ]
            ),

        "ridge_lambda":
            float(
                frozen[
                    "lambda"
                ]
            ),

        "hardware_selection":
            json_safe(
                chosen.to_dict()
                if hasattr(
                    chosen,
                    "to_dict",
                )
                else chosen
            ),

        "service_usage_before":
            usage_before,
    }

    preflight_path = (
        RESULTS
        / "11_1B_estimator_qpu_preflight.json"
    )

    preflight_path.write_text(
        json.dumps(
            json_safe(
                preflight
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    qpu_db.save_json_artifact(
        run_uuid,
        preflight_path.name,
        preflight,
    )

    selected_df = pd.DataFrame(
        {
            "endpoint":
                selected_endpoints
        }
    )

    selected_path = (
        RESULTS
        / "11_1B_estimator_qpu_selected_endpoints.csv"
    )

    selected_df.to_csv(
        selected_path,
        index=False,
    )

    qpu_db.save_dataframe_artifact(
        run_uuid,
        selected_path.name,
        selected_df,
    )

    chosen_dict = json_safe(
        chosen.to_dict()
        if hasattr(
            chosen,
            "to_dict",
        )
        else chosen
    )

    qpu_db.update_run(
        run_uuid,
        run_status=
            "PREFLIGHT_COMPLETE",

        physical_layout_json=
            layout,

        hardware_selection_json=
            chosen_dict,

        max_2q_error_percent=
            float(
                chosen.get(
                    "compiled_2q_error_max_percent",
                    np.nan,
                )
            ),

        max_readout_error_percent=
            float(
                chosen.get(
                    "max_readout_error_percent",
                    np.nan,
                )
            ),

        max_1q_error_percent=
            float(
                chosen.get(
                    "compiled_1q_error_max_percent",
                    np.nan,
                )
            ),

        min_t1_us=
            float(
                chosen.get(
                    "min_t1_us",
                    np.nan,
                )
            ),

        min_t2_us=
            float(
                chosen.get(
                    "min_t2_us",
                    np.nan,
                )
            ),

        compiled_duration_us=
            float(
                chosen.get(
                    "compiled_duration_us",
                    np.nan,
                )
            ),

        r_t2=
            float(
                chosen.get(
                    "R_T2",
                    np.nan,
                )
            ),

        n_circuits=
            int(
                len(
                    pubs
                )
            ),

        median_cz_per_feature=
            float(
                resources[
                    "n_cz"
                ].median()
            ),

        median_max_setting_depth=
            float(
                resources[
                    "depth"
                ].median()
            ),

        median_feature_duration_us=
            float(
                resources[
                    "duration_us"
                ].median()
            ),

        service_usage_before_json=
            usage_before,
    )

    if not args.submit:
        qpu_db.update_run(
            run_uuid,
            run_status=
                "DRY_RUN_COMPLETE",
            completed_at_utc=
                utc_now_naive(),
        )

        print()
        print(
            "DRY RUN COMPLETE — "
            "NO QPU JOB SUBMITTED."
        )

        print()

        if args.full:
            print(
                "To submit FULL 2025:"
            )

            print(
                f"python "
                f"{Path(__file__).name} "
                f"--full --submit "
                f"--backend {args.backend} "
                f"--precision "
                f"{args.precision}"
            )

        else:
            print(
                "To submit this pilot:"
            )

            print(
                f"python "
                f"{Path(__file__).name} "
                f"--submit "
                f"--backend {args.backend} "
                f"--precision "
                f"{args.precision} "
                f"--pilot-size "
                f"{len(selected_endpoints)}"
            )

        return

    print()
    print("=" * 126)
    print(
        "SUBMITTING ONE EstimatorV2 JOB "
        "IN JOB MODE"
    )
    print("=" * 126)

    print(
        f"Submitting "
        f"{len(pubs)} PUBs, "
        f"{len(FEATURES)} observables/PUB, "
        f"target precision "
        f"{args.precision:.9f}..."
    )

    qpu_db.update_run(
        run_uuid,
        run_status=
            "SUBMITTING",
        submitted_at_utc=
            utc_now_naive(),
    )

    estimator = EstimatorV2(
        mode=backend
    )

    # Make this as close as possible to the original raw-Sampler hardware test:
    # no built-in resilience/mitigation.
    estimator.options.resilience_level = 0
    estimator.options.dynamical_decoupling.enable = False
    estimator.options.twirling.enable_gates = False
    estimator.options.twirling.enable_measure = False

    job = estimator.run(
        pubs,
        precision=float(
            args.precision
        ),
    )

    job_id = (
        job.job_id()
    )

    print(
        f"Job ID: "
        f"{job_id}"
    )

    qpu_db.update_run(
        run_uuid,
        run_status=
            "SUBMITTED",
        job_id=
            str(
                job_id
            ),
    )

    result = (
        job.result()
    )

    final_job_status = str(
        job.status()
    )

    print(
        f"Final status: "
        f"{final_job_status}"
    )

    metrics = None

    try:
        metrics = json_safe(
            job.metrics()
        )
    except Exception as exc:
        print(
            "[timing] WARNING: "
            f"job.metrics() unavailable: "
            f"{exc}"
        )

    qpu_timing = (
        extract_ibm_qpu_timing(
            job=job,
            metrics=metrics,
        )
    )

    print()
    print(
        "IBM QPU TIMING"
    )
    print(
        "-" * 126
    )

    if (
        qpu_timing[
            "qpu_charge_time_seconds"
        ]
        is not None
    ):
        print(
            "QPU charge / locked time:       "
            f"{qpu_timing['qpu_charge_time_seconds']:.6f} s"
        )

    else:
        print(
            "QPU charge / locked time:       "
            "unavailable"
        )

    if (
        qpu_timing[
            "qpu_circuits_execution_time_seconds"
        ]
        is not None
    ):
        print(
            "Actual circuit execution on QPU:"
            f" {qpu_timing['qpu_circuits_execution_time_seconds']:.6f} s"
        )

    else:
        print(
            "Actual circuit execution on QPU: "
            "unavailable"
        )

    if (
        qpu_timing[
            "qpu_running_wall_seconds"
        ]
        is not None
    ):
        print(
            "IBM RUNNING -> FINISHED elapsed:"
            f" {qpu_timing['qpu_running_wall_seconds']:.6f} s"
        )

    else:
        print(
            "IBM RUNNING -> FINISHED elapsed: "
            "unavailable"
        )

    print()

    job_payload = {
        "job_id":
            job_id,

        "backend":
            args.backend,

        "layout":
            layout,

        "primitive":
            "EstimatorV2",

        "precision":
            float(
                args.precision
            ),

        "resilience_level":
            0,

        "n_pubs":
            int(
                len(
                    pubs
                )
            ),

        "n_observables_per_pub":
            int(
                len(
                    FEATURES
                )
            ),

        "status":
            final_job_status,

        "qpu_charge_time_seconds":
            qpu_timing[
                "qpu_charge_time_seconds"
            ],

        "qpu_circuits_execution_time_seconds":
            qpu_timing[
                "qpu_circuits_execution_time_seconds"
            ],

        "qpu_running_wall_seconds":
            qpu_timing[
                "qpu_running_wall_seconds"
            ],

        "ibm_job_created_at_utc":
            qpu_timing[
                "ibm_job_created_at_utc"
            ],

        "ibm_job_running_at_utc":
            qpu_timing[
                "ibm_job_running_at_utc"
            ],

        "ibm_job_finished_at_utc":
            qpu_timing[
                "ibm_job_finished_at_utc"
            ],

        "metrics":
            metrics,
    }

    job_path = (
        RESULTS
        / "11_1B_estimator_qpu_job.json"
    )

    job_path.write_text(
        json.dumps(
            json_safe(
                job_payload
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    qpu_db.save_json_artifact(
        run_uuid,
        job_path.name,
        job_payload,
    )

    # ---------------------------------------------------------------------
    # Parse expectation values + standard errors.
    # ---------------------------------------------------------------------
    if len(result) != len(
        selected_endpoints
    ):
        raise RuntimeError(
            "Estimator returned "
            f"{len(result)} PUB results "
            "for "
            f"{len(selected_endpoints)} "
            "endpoints."
        )

    expectation_rows = []
    X_qpu = []
    X_std = []

    for (
        endpoint,
        pub_result,
    ) in zip(
        selected_endpoints,
        result,
    ):
        evs = np.asarray(
            pub_result.data.evs,
            dtype=float,
        ).reshape(-1)

        stds = np.asarray(
            pub_result.data.stds,
            dtype=float,
        ).reshape(-1)

        if len(
            evs
        ) != len(
            FEATURES
        ):
            raise RuntimeError(
                f"Endpoint {endpoint}: "
                f"expected {len(FEATURES)} "
                "expectation values, "
                f"got {len(evs)}."
            )

        if len(
            stds
        ) != len(
            FEATURES
        ):
            raise RuntimeError(
                f"Endpoint {endpoint}: "
                f"expected {len(FEATURES)} "
                "standard errors, "
                f"got {len(stds)}."
            )

        X_qpu.append(
            evs
        )

        X_std.append(
            stds
        )

        row = {
            "endpoint":
                int(
                    endpoint
                )
        }

        for (
            j,
            feature,
        ) in enumerate(
            FEATURES
        ):
            row[
                f"{feature}_ev"
            ] = float(
                evs[j]
            )

            row[
                f"{feature}_std"
            ] = float(
                stds[j]
            )

        expectation_rows.append(
            row
        )

    X_qpu = np.asarray(
        X_qpu,
        dtype=float,
    )

    X_std = np.asarray(
        X_std,
        dtype=float,
    )

    expectation_df = pd.DataFrame(
        expectation_rows
    )

    expectation_path = (
        RESULTS
        / "11_1B_estimator_qpu_expectation_values.csv"
    )

    expectation_df.to_csv(
        expectation_path,
        index=False,
    )

    qpu_db.save_dataframe_artifact(
        run_uuid,
        expectation_path.name,
        expectation_df,
    )

    # ---------------------------------------------------------------------
    # Align with ideal features + frozen Ridge.
    # ---------------------------------------------------------------------
    endpoint_to_master_row = {
        int(
            endpoint
        ):
            idx
        for (
            idx,
            endpoint,
        ) in enumerate(
            frozen[
                "endpoints"
            ]
        )
    }

    val_endpoint_to_pos = {
        int(
            endpoint
        ):
            idx
        for (
            idx,
            endpoint,
        ) in enumerate(
            frozen[
                "val_endpoints"
            ]
        )
    }

    X_ideal = []
    targets = []
    ideal_preds = []

    feature_rows = []

    for (
        row_idx,
        endpoint,
    ) in enumerate(
        selected_endpoints
    ):
        endpoint = int(
            endpoint
        )

        master_idx = (
            endpoint_to_master_row[
                endpoint
            ]
        )

        ideal_vec = (
            frozen[
                "X"
            ][
                master_idx
            ]
        )

        val_pos = (
            val_endpoint_to_pos[
                endpoint
            ]
        )

        target = float(
            frozen[
                "y_val"
            ][
                val_pos
            ]
        )

        ideal_pred = float(
            frozen[
                "pred_val"
            ][
                val_pos
            ]
        )

        X_ideal.append(
            ideal_vec
        )

        targets.append(
            target
        )

        ideal_preds.append(
            ideal_pred
        )

        row = {
            "endpoint":
                endpoint,

            "target":
                target,

            "pred_ideal":
                ideal_pred,
        }

        for (
            j,
            feature,
        ) in enumerate(
            FEATURES
        ):
            row[
                f"{feature}_ideal"
            ] = float(
                ideal_vec[j]
            )

            row[
                f"{feature}_qpu"
            ] = float(
                X_qpu[
                    row_idx,
                    j,
                ]
            )

            row[
                f"{feature}_std"
            ] = float(
                X_std[
                    row_idx,
                    j,
                ]
            )

            row[
                f"{feature}_error"
            ] = float(
                X_qpu[
                    row_idx,
                    j,
                ]
                - ideal_vec[j]
            )

        feature_rows.append(
            row
        )

    X_ideal = np.asarray(
        X_ideal,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    ideal_preds = np.asarray(
        ideal_preds,
        dtype=float,
    )

    pred_qpu = (
        common.predict_scaled_ridge(
            frozen[
                "model"
            ],
            frozen[
                "scaler"
            ],
            frozen[
                "keep"
            ],
            X_qpu,
        )
    )

    features_df = pd.DataFrame(
        feature_rows
    )

    features_df[
        "pred_qpu"
    ] = pred_qpu

    features_df[
        "abs_error_ideal"
    ] = np.abs(
        ideal_preds
        - targets
    )

    features_df[
        "abs_error_qpu"
    ] = np.abs(
        pred_qpu
        - targets
    )

    features_path = (
        RESULTS
        / "11_1B_estimator_qpu_features_predictions.csv"
    )

    features_df.to_csv(
        features_path,
        index=False,
    )

    qpu_db.save_dataframe_artifact(
        run_uuid,
        features_path.name,
        features_df,
    )

    feature_diff = (
        X_qpu
        - X_ideal
    )

    qpu_feature_mae = float(
        np.mean(
            np.abs(
                feature_diff
            )
        )
    )

    qpu_feature_rmse = float(
        np.sqrt(
            np.mean(
                feature_diff ** 2
            )
        )
    )

    pred_corr = float(
        np.corrcoef(
            pred_qpu,
            ideal_preds,
        )[0, 1]
    )

    estimator_mean_std = float(
        np.mean(
            X_std
        )
    )

    estimator_max_std = float(
        np.max(
            X_std
        )
    )

    service_usage_after = (
        get_service_usage(
            service
        )
    )

    summary = {
        "run_uuid":
            run_uuid,

        "candidate_key":
            CANDIDATE_KEY,

        "primitive":
            "EstimatorV2",

        "measurement_method":
            "direct_expectation_value_estimation",

        "precision":
            float(
                args.precision
            ),

        "resilience_level":
            0,

        "backend":
            args.backend,

        "layout":
            layout,

        "n_validation_endpoints":
            int(
                len(
                    selected_endpoints
                )
            ),

        "full_2025":
            bool(
                args.full
            ),

        "qpu_rmse":
            rmse(
                targets,
                pred_qpu,
            ),

        "qpu_mae":
            mae(
                targets,
                pred_qpu,
            ),

        "qpu_bias":
            bias(
                targets,
                pred_qpu,
            ),

        "ideal_same_subset_rmse":
            rmse(
                targets,
                ideal_preds,
            ),

        "ideal_same_subset_mae":
            mae(
                targets,
                ideal_preds,
            ),

        "ideal_same_subset_bias":
            bias(
                targets,
                ideal_preds,
            ),

        "qpu_feature_mae":
            qpu_feature_mae,

        "qpu_feature_rmse":
            qpu_feature_rmse,

        "prediction_correlation_qpu_vs_ideal":
            pred_corr,

        "estimator_mean_standard_error":
            estimator_mean_std,

        "estimator_max_standard_error":
            estimator_max_std,

        "qpu_charge_time_seconds":
            qpu_timing[
                "qpu_charge_time_seconds"
            ],

        "qpu_circuits_execution_time_seconds":
            qpu_timing[
                "qpu_circuits_execution_time_seconds"
            ],

        "qpu_running_wall_seconds":
            qpu_timing[
                "qpu_running_wall_seconds"
            ],

        "ibm_job_created_at_utc":
            qpu_timing[
                "ibm_job_created_at_utc"
            ],

        "ibm_job_running_at_utc":
            qpu_timing[
                "ibm_job_running_at_utc"
            ],

        "ibm_job_finished_at_utc":
            qpu_timing[
                "ibm_job_finished_at_utc"
            ],

        "job_id":
            job_id,

        "job_status":
            final_job_status,

        "job_metrics":
            metrics,

        "service_usage_before":
            usage_before,

        "service_usage_after":
            service_usage_after,
    }

    summary_path = (
        RESULTS
        / "11_1B_estimator_qpu_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            json_safe(
                summary
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    qpu_db.save_json_artifact(
        run_uuid,
        summary_path.name,
        summary,
    )

    qpu_db.update_run(
        run_uuid,

        run_status=
            "COMPLETED",

        completed_at_utc=
            utc_now_naive(),

        job_status=
            final_job_status,

        job_id=
            str(
                job_id
            ),

        qpu_charge_time_seconds=
            qpu_timing[
                "qpu_charge_time_seconds"
            ],

        qpu_circuits_execution_time_seconds=
            qpu_timing[
                "qpu_circuits_execution_time_seconds"
            ],

        qpu_running_wall_seconds=
            qpu_timing[
                "qpu_running_wall_seconds"
            ],

        ibm_job_created_at_utc=
            qpu_timing[
                "ibm_job_created_at_utc"
            ],

        ibm_job_running_at_utc=
            qpu_timing[
                "ibm_job_running_at_utc"
            ],

        ibm_job_finished_at_utc=
            qpu_timing[
                "ibm_job_finished_at_utc"
            ],

        job_metrics_json=
            metrics,

        qpu_rmse=
            float(
                summary[
                    "qpu_rmse"
                ]
            ),

        qpu_mae=
            float(
                summary[
                    "qpu_mae"
                ]
            ),

        qpu_bias=
            float(
                summary[
                    "qpu_bias"
                ]
            ),

        ideal_same_subset_rmse=
            float(
                summary[
                    "ideal_same_subset_rmse"
                ]
            ),

        ideal_same_subset_mae=
            float(
                summary[
                    "ideal_same_subset_mae"
                ]
            ),

        ideal_same_subset_bias=
            float(
                summary[
                    "ideal_same_subset_bias"
                ]
            ),

        qpu_feature_mae=
            float(
                summary[
                    "qpu_feature_mae"
                ]
            ),

        qpu_feature_rmse=
            float(
                summary[
                    "qpu_feature_rmse"
                ]
            ),

        prediction_correlation_qpu_vs_ideal=
            float(
                summary[
                    "prediction_correlation_qpu_vs_ideal"
                ]
            ),

        estimator_mean_standard_error=
            estimator_mean_std,

        estimator_max_standard_error=
            estimator_max_std,

        service_usage_after_json=
            service_usage_after,
    )

    print()
    print("=" * 126)
    print(
        "REAL-QPU ESTIMATOR RESULTS"
    )
    print("=" * 126)

    print(
        f"Endpoints tested:               "
        f"{len(selected_endpoints)}"
    )

    print(
        f"Target precision:               "
        f"{args.precision:.9f}"
    )

    print(
        f"Estimator QPU RMSE:             "
        f"{summary['qpu_rmse']:.6f}"
    )

    print(
        f"Estimator QPU MAE:              "
        f"{summary['qpu_mae']:.6f}"
    )

    print(
        f"Estimator QPU bias:             "
        f"{summary['qpu_bias']:+.6f}"
    )

    print(
        f"Ideal same-subset RMSE:         "
        f"{summary['ideal_same_subset_rmse']:.6f}"
    )

    print(
        f"QPU feature MAE:                "
        f"{summary['qpu_feature_mae']:.6f}"
    )

    print(
        f"QPU feature RMSE:               "
        f"{summary['qpu_feature_rmse']:.6f}"
    )

    print(
        f"QPU-vs-ideal pred correlation:  "
        f"{summary['prediction_correlation_qpu_vs_ideal']:.6f}"
    )

    print(
        f"Mean Estimator std error:       "
        f"{summary['estimator_mean_standard_error']:.6f}"
    )

    print(
        f"Max Estimator std error:        "
        f"{summary['estimator_max_standard_error']:.6f}"
    )

    if (
        summary[
            "qpu_charge_time_seconds"
        ]
        is not None
    ):
        print(
            f"QPU charge / locked time:       "
            f"{summary['qpu_charge_time_seconds']:.6f} s"
        )

    if (
        summary[
            "qpu_circuits_execution_time_seconds"
        ]
        is not None
    ):
        print(
            f"Actual circuit execution QPU:   "
            f"{summary['qpu_circuits_execution_time_seconds']:.6f} s"
        )

    if (
        summary[
            "qpu_running_wall_seconds"
        ]
        is not None
    ):
        print(
            f"IBM running elapsed time:       "
            f"{summary['qpu_running_wall_seconds']:.6f} s"
        )

    print()
    print(
        "2026 remains FROZEN / UNUSED."
    )

    print()
    print(
        f"[database] Run "
        f"{run_uuid} "
        "persisted successfully."
    )

    print()
    print(
        "Saved:"
    )

    for path in [
        preflight_path,
        resources_path,
        selected_path,
        expectation_path,
        features_path,
        job_path,
        summary_path,
    ]:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        if (
            CURRENT_RUN_UUID
            is not None
        ):
            try:
                qpu_db.update_run(
                    CURRENT_RUN_UUID,
                    run_status=
                        "FAILED",
                    completed_at_utc=
                        utc_now_naive(),
                    error_message=
                        str(exc),
                )

                print(
                    "[database] Run "
                    f"{CURRENT_RUN_UUID} "
                    "marked FAILED."
                )

            except Exception as db_exc:
                print(
                    "[database] WARNING: "
                    "could not mark "
                    "failed run: "
                    f"{db_exc}"
                )

        raise
