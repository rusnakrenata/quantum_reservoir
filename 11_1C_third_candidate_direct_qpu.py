from __future__ import annotations

"""
WEEK 11.1C
Direct-QPU test for the third Week-10 QPU candidate with database persistence.

Candidate frozen from:
    results/10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv

Expected candidate:
    RWP_H2_R3
    H2, W=2, r=2
    alpha=0.75
    readout=XZinj_dropX3
    shots/setting=1024

Scientific protocol:
    2022-2024 : establish/freeze Ridge readout only
    2025      : QPU validation / hardware-transfer diagnostics
    2026      : FROZEN / NEVER LOADED

Default behaviour is a DRY RUN.
Use --submit to send the deterministic 2025 pilot subset.
Use --full only after the pilot has been interpreted.

The script refreshes the requested IBM backend and reselects the physical
H3 embedding immediately before compilation/submission.

No readout-error mitigation is applied in this first raw-hardware transfer test.
"""

import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import SamplerV2

from ibm_account import get_service
import db_objects as qpu_db


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
SELECTOR_FILE = HERE / "11_0A_live_embedding_reselection.py"
MANIFEST_FILE = RESULTS / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"

CANDIDATE_KEY = "RWP_H2_R3"
EXPECTED_TOPOLOGY = "H2"
EXPECTED_WINDOW = 2
EXPECTED_R = 2
EXPECTED_READOUT = "XZinj_dropX3"
EXPECTED_ALPHA = 0.75
DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_SHOTS = 1024
DEFAULT_PILOT_SIZE = 64

OPT_LEVEL = 1
SEED_TRANSPILE = 42
SHORTLIST = 120

# Empirical planning heuristic from the earlier 11.0D run:
# actual QPU usage 52 s for scheduled-duration*shots sum 0.701174 s.
# It is only a rough planning conversion, not an IBM billing formula.
EMPIRICAL_USAGE_SCALE = 52.0 / 0.701174

FEATURES = ["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3"]

CURRENT_RUN_UUID = None


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_111a_common")
selector = load_module(SELECTOR_FILE, "qrc_111a_selector")


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def rmse(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(a - b)))


def bias(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.mean(b - a))


def get_service_usage(service):
    """
    Best-effort only. IBM account/service APIs have changed over time.
    Failure to retrieve usage must never block the scientific run.
    """
    for attr in ["usage", "usage_info"]:
        fn = getattr(service, attr, None)
        if callable(fn):
            try:
                out = fn()
                if hasattr(out, "to_dict"):
                    out = out.to_dict()
                return json_safe(out)
            except Exception:
                pass
    return None



def utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_job_usage_seconds(job):
    """
    Best-effort finalized QPU usage for this IBM job.
    Returns None when the installed Runtime version does not expose it.
    """
    fn = getattr(job, "usage", None)
    if callable(fn):
        try:
            value = fn()
            if value is None:
                return None
            if isinstance(value, dict):
                for key in [
                    "quantum_seconds",
                    "usage_seconds",
                    "seconds",
                ]:
                    if key in value:
                        return float(value[key])
                return None
            return float(value)
        except Exception:
            pass
    return None



def _parse_ibm_utc_timestamp(value):
    """Convert IBM ISO timestamp to naive UTC datetime for MariaDB."""
    if value is None:
        return None

    try:
        ts = pd.to_datetime(value, utc=True)
        return ts.to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None


def extract_ibm_qpu_timing(job, metrics):
    """
    Extract three distinct timing concepts from finalized IBM job metrics.

    Returns
    -------
    dict with:
        qpu_charge_time_seconds
            IBM capacity/usage charge time (QPU locked for workload).

        qpu_circuits_execution_time_seconds
            IBM circuits_execution_time_ns converted to seconds; this is the
            actual circuit execution time reported for the QPU.

        qpu_running_wall_seconds
            IBM `running` -> `finished` lifecycle elapsed time.

        ibm_job_*_at_utc
            IBM lifecycle timestamps.

    The current IBM Compute Service metrics schema uses
    usage.qpu_charge_time_seconds and root-level circuits_execution_time_ns.
    Older Runtime versions may expose usage.quantum_seconds, so that is kept
    as a compatibility fallback.
    """
    metrics = metrics or {}
    usage = metrics.get("usage") or {}
    timestamps = metrics.get("timestamps") or {}

    qpu_charge = usage.get("qpu_charge_time_seconds")
    if qpu_charge is None:
        qpu_charge = usage.get("quantum_seconds")
    if qpu_charge is None:
        qpu_charge = usage.get("seconds")
    if qpu_charge is None:
        qpu_charge = get_job_usage_seconds(job)

    if qpu_charge is not None:
        try:
            qpu_charge = float(qpu_charge)
        except Exception:
            qpu_charge = None

    execution_ns = metrics.get("circuits_execution_time_ns")
    if execution_ns is None:
        # Compatibility with older metrics shapes.
        execution_ns = usage.get("circuits_execution_time_ns")

    qpu_circuit_seconds = None
    if execution_ns is not None:
        try:
            qpu_circuit_seconds = float(execution_ns) / 1e9
        except Exception:
            qpu_circuit_seconds = None

    created_at = _parse_ibm_utc_timestamp(timestamps.get("created"))
    running_at = _parse_ibm_utc_timestamp(timestamps.get("running"))
    finished_at = _parse_ibm_utc_timestamp(timestamps.get("finished"))

    running_wall_seconds = None
    if running_at is not None and finished_at is not None:
        running_wall_seconds = float(
            (finished_at - running_at).total_seconds()
        )

    return {
        "qpu_charge_time_seconds": qpu_charge,
        "qpu_circuits_execution_time_seconds": qpu_circuit_seconds,
        "qpu_running_wall_seconds": running_wall_seconds,
        "ibm_job_created_at_utc": created_at,
        "ibm_job_running_at_utc": running_at,
        "ibm_job_finished_at_utc": finished_at,
    }


def load_candidate():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"{MANIFEST_FILE} not found. Use the final enriched Week-10 manifest."
        )

    df = pd.read_csv(MANIFEST_FILE)
    sub = df[df["candidate_key"].astype(str) == CANDIDATE_KEY].copy()

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected exactly one row for {CANDIDATE_KEY}, found {len(sub)}."
        )

    row = sub.iloc[0]

    if str(row["protocol"]).upper() != "RWP":
        raise RuntimeError("Frozen winner is expected to be RWP.")
    if str(row["topology"]) != EXPECTED_TOPOLOGY:
        raise RuntimeError("Unexpected topology in enriched manifest.")
    if int(row["window"]) != EXPECTED_WINDOW:
        raise RuntimeError("Unexpected W in enriched manifest.")
    if int(row["original_r"]) != EXPECTED_R:
        raise RuntimeError("Unexpected r in enriched manifest.")
    if str(row["test_readout"]) != EXPECTED_READOUT:
        raise RuntimeError("Unexpected readout in enriched manifest.")
    if not np.isclose(float(row["alpha"]), EXPECTED_ALPHA):
        raise RuntimeError("Unexpected alpha in enriched manifest.")

    raw_j = json.loads(str(row["J_json"]))
    J = {}
    for edge in common.TOPOLOGY_EDGES[EXPECTED_TOPOLOGY]:
        i, j = edge
        key = f"J{i}{j}"
        rev = f"J{j}{i}"
        if key in raw_j:
            val = raw_j[key]
        elif rev in raw_j:
            val = raw_j[rev]
        else:
            raise KeyError(f"Missing coupling for edge {edge} in J_json.")
        J[edge] = float(val)

    candidate = {
        "candidate_id": CANDIDATE_KEY,
        "topology": EXPECTED_TOPOLOGY,
        "window": EXPECTED_WINDOW,
        "r": EXPECTED_R,
        "alpha": float(row["alpha"]),
        "dt": float(row["dt"]),
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
        "J": J,
    }

    meta = row.to_dict()
    return candidate, meta


def build_frozen_readout(candidate, meta):
    """
    Reconstruct the exact ideal RWP feature bank for 2022-2025,
    fit Ridge ONLY on 2022-2024, and audit the Week-10 winner.
    """
    work_tv, cols, y_all = common.load_train_validation()
    angles = common.make_angles(work_tv, cols, float(candidate["alpha"]))

    A_list = common.build_channels(candidate, angles)
    endpoints, master = common.rwp_master_feature_bank_from_channels(
        A_list,
        EXPECTED_WINDOW,
    )

    X = common.select_features(master, EXPECTED_READOUT)
    y_endpoint = y_all[endpoints]

    train_mask = endpoints < common.N_TRAIN
    val_mask = endpoints >= common.N_TRAIN

    X_train = X[train_mask]
    y_train = y_endpoint[train_mask]
    X_val = X[val_mask]
    y_val = y_endpoint[val_mask]
    val_endpoints = endpoints[val_mask]

    selected_lambda, _, cv_summary = common.select_lambda_training_only(
        X_train,
        y_train,
        n_splits=5,
    )

    expected_lambda = float(meta["source_selected_lambda"])
    expected_cv = float(meta["source_readout_cv_rmse"])

    if not np.isclose(selected_lambda, expected_lambda, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Lambda audit failed: recomputed={selected_lambda}, "
            f"manifest={expected_lambda}."
        )

    cv_rmse = float(cv_summary.iloc[0]["cv_rmse_mean"])
    if abs(cv_rmse - expected_cv) > 1e-5:
        raise RuntimeError(
            f"CV audit failed: recomputed={cv_rmse:.9f}, "
            f"manifest={expected_cv:.9f}."
        )

    model, scaler, keep = common.fit_scaled_ridge(
        X_train,
        y_train,
        selected_lambda,
    )

    pred_val = common.predict_scaled_ridge(
        model,
        scaler,
        keep,
        X_val,
    )

    ideal_rmse = rmse(y_val, pred_val)
    ideal_mae = mae(y_val, pred_val)
    ideal_bias = bias(y_val, pred_val)

    expected_val = float(meta["source_validation_rmse"])
    if abs(ideal_rmse - expected_val) > 1e-5:
        raise RuntimeError(
            f"Ideal validation audit failed: recomputed={ideal_rmse:.9f}, "
            f"manifest={expected_val:.9f}."
        )

    return {
        "work_tv": work_tv,
        "cols": cols,
        "y_all": y_all,
        "angles": angles,
        "endpoints": endpoints,
        "master": master,
        "X": X,
        "y_endpoint": y_endpoint,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "val_endpoints": val_endpoints,
        "X_val": X_val,
        "y_val": y_val,
        "pred_val": pred_val,
        "model": model,
        "scaler": scaler,
        "keep": keep,
        "lambda": selected_lambda,
        "cv_rmse": cv_rmse,
        "ideal_rmse": ideal_rmse,
        "ideal_mae": ideal_mae,
        "ideal_bias": ideal_bias,
    }


def append_rwp_step(qc, candidate, angle_row, first_step):
    """
    One RWP input step.
    q0..q3 are reset between inputs; q4,q5 retain memory inside the W=2 window.
    """
    if not first_step:
        for q in range(4):
            qc.reset(q)

    for q in range(4):
        qc.ry(float(angle_row[q]), q)

    r = int(candidate["r"])
    dt = float(candidate["dt"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])
    J = candidate["J"]

    for _ in range(r):
        for i, j in common.TOPOLOGY_EDGES[candidate["topology"]]:
            qc.rzz(
                2.0 * float(J[(i, j)]) * dt / r,
                i,
                j,
            )

        theta_x = 2.0 * hx * dt / r
        for q in range(6):
            qc.rx(theta_x, q)

        if not np.isclose(hy, 0.0):
            theta_y = 2.0 * hy * dt / r
            qc.ry(theta_y, 4)
            qc.ry(theta_y, 5)


def readout_settings():
    """
    Exact grouped measurements for XZinj_dropX3.

    Setting A: XXXZ on injection qubits q0..q3
        -> X0, X1, X2
        -> Z3 is retained only as a cross-setting consistency diagnostic

    Setting B: ZZZZ on injection qubits q0..q3
        -> authoritative Z0, Z1, Z2, Z3

    Memory qubits q4,q5 are not measured for this readout.
    """
    return [
        (
            "XXXZ",
            {0: "X", 1: "X", 2: "X", 3: "Z"},
            [0, 1, 2, 3],
        ),
        (
            "ZZZZ",
            {0: "Z", 1: "Z", 2: "Z", 3: "Z"},
            [0, 1, 2, 3],
        ),
    ]


def build_measurement_circuit(candidate, all_angles, endpoint, setting):
    label, basis, measured = setting

    qreg = QuantumRegister(6, "q")
    creg = ClassicalRegister(len(measured), "m")
    qc = QuantumCircuit(
        qreg,
        creg,
        name=f"{candidate['topology']}_W{EXPECTED_WINDOW}_e{int(endpoint)}_{label}",
    )

    start = int(endpoint) - EXPECTED_WINDOW + 1
    if start < 0:
        raise ValueError("Endpoint does not have enough W=2 history.")

    for k, idx in enumerate(range(start, int(endpoint) + 1)):
        append_rwp_step(
            qc,
            candidate,
            all_angles[idx],
            first_step=(k == 0),
        )

    for q, axis in basis.items():
        if axis == "X":
            qc.h(q)
        elif axis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif axis == "Z":
            pass
        else:
            raise ValueError(axis)

    for c, q in enumerate(measured):
        qc.measure(q, c)

    return qc


def resource_row(circuit, backend, endpoint, setting):
    ops = {str(k): int(v) for k, v in circuit.count_ops().items()}
    duration_s = float(
        circuit.estimate_duration(
            backend.target,
            unit="s",
        )
    )
    return {
        "endpoint": int(endpoint),
        "setting": str(setting),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "n_cz": int(ops.get("cz", 0)),
        "n_swap": int(ops.get("swap", 0)),
        "n_reset": int(ops.get("reset", 0)),
        "n_measure": int(ops.get("measure", 0)),
        "duration_us": duration_s * 1e6,
        "operations": json.dumps(ops, sort_keys=True),
    }


def bit_from_qiskit_string(bitstring, c_index):
    """
    Qiskit count strings display the highest classical bit on the left.
    Classical bit c0 is therefore the rightmost character.
    """
    s = str(bitstring).replace(" ", "")
    return int(s[-1 - int(c_index)])


def expectation_from_counts(counts, classical_bits):
    total = float(sum(counts.values()))
    if total <= 0:
        raise RuntimeError("Empty counts.")
    acc = 0.0
    for bitstring, n in counts.items():
        eig = 1.0
        for c in classical_bits:
            eig *= -1.0 if bit_from_qiskit_string(bitstring, c) else 1.0
        acc += float(n) * eig
    return acc / total


def get_pub_counts(pub_result):
    data = pub_result.data
    reg = getattr(data, "m", None)
    if reg is None:
        raise RuntimeError(
            "Sampler result does not contain classical register 'm'."
        )
    return reg.get_counts()


def choose_validation_endpoints(val_endpoints, full, pilot_size):
    val_endpoints = np.asarray(val_endpoints, dtype=int)

    if full:
        return val_endpoints.copy()

    n = max(1, min(int(pilot_size), len(val_endpoints)))

    # Deterministic calendar-spread sample. No target values are used.
    offsets = np.unique(
        np.linspace(
            0,
            len(val_endpoints) - 1,
            n,
            dtype=int,
        )
    )
    return val_endpoints[offsets]


def main():
    global CURRENT_RUN_UUID

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help="Backend for the Week-11.1 direct-QPU transfer test.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=DEFAULT_PILOT_SIZE,
        help="Number of evenly spaced 2025 validation endpoints in the pilot.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use all 365 validation endpoints. Keep 2026 frozen.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit one SamplerV2 job. Default is dry-run.",
    )
    parser.add_argument(
        "--force-over-budget",
        action="store_true",
        help="Allow submission even when the empirical usage estimate exceeds reported remaining usage.",
    )
    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError("--shots must be positive.")

    print("=" * 126)
    print("WEEK 11.1C — THIRD CANDIDATE DIRECT-QPU TEST")
    print("=" * 126)
    print("2022-2024 = establish/freeze readout")
    print("2025      = validation / hardware-transfer test")
    print("2026      = FROZEN / NOT LOADED")
    print()

    candidate, manifest_meta = load_candidate()

    print("Frozen Week-10 winner:")
    print(f"  candidate={CANDIDATE_KEY}")
    print(f"  topology={candidate['topology']}")
    print(f"  W={candidate['window']}")
    print(f"  r={candidate['r']}")
    print(f"  alpha={candidate['alpha']}")
    print(f"  dt={candidate['dt']}")
    print(f"  hx={candidate['hx']:+.12f}")
    print(f"  hy={candidate['hy']:+.12f}")
    print(f"  readout={EXPECTED_READOUT}")
    print(f"  shots/setting={args.shots}")
    print("  Week-10 sampled reference RMSE=4.997545 ± 0.061397")
    print("  Week-10 resource point: 80 CZ/feature, depth≈142, duration≈7.704 us")
    print(
        "  J="
        + json.dumps(
            {f"J{i}{j}": v for (i, j), v in candidate["J"].items()},
            sort_keys=True,
        )
    )
    print()

    frozen = build_frozen_readout(candidate, manifest_meta)

    print("Training-only readout audit:")
    print(f"  lambda={frozen['lambda']}")
    print(f"  CV RMSE={frozen['cv_rmse']:.6f}")
    print()
    print("Ideal 2025 reference:")
    print(f"  RMSE={frozen['ideal_rmse']:.6f}")
    print(f"  MAE ={frozen['ideal_mae']:.6f}")
    print(f"  bias={frozen['ideal_bias']:+.6f}")
    print()

    selected_endpoints = choose_validation_endpoints(
        frozen["val_endpoints"],
        full=args.full,
        pilot_size=args.pilot_size,
    )

    mode_name = "FULL 2025" if args.full else f"PILOT {len(selected_endpoints)}"
    print(f"QPU evaluation mode: {mode_name}")
    print(
        f"Validation endpoints selected: {len(selected_endpoints)} "
        f"of {len(frozen['val_endpoints'])}"
    )
    print(
        "Selection is deterministic and evenly spread through 2025; "
        "targets are not used to choose pilot dates."
    )
    print()

    # -----------------------------------------------------------------------
    # Database run record
    # -----------------------------------------------------------------------
    # We create/commit the DB record BEFORE any QPU submission.  If the
    # database is unavailable, the script stops here rather than consuming
    # hardware time without recording the experiment.
    qpu_db.create_all_tables()

    run_uuid = qpu_db.create_run(
        script_name=Path(__file__).name,
        run_status="PREPARING",
        started_at_utc=utc_now_naive(),
        forecast_dataset_name="property_damage_next_day_v1",
        feature_set_name="F4",
        evaluation_split="validation_2025",
        candidate_key=CANDIDATE_KEY,
        selected_by_rules=str(manifest_meta.get("selected_by_rules", "")),
        protocol=str(manifest_meta.get("protocol", "RWP")),
        topology=str(candidate["topology"]),
        window_size=int(candidate["window"]),
        trotter_r=int(candidate["r"]),
        readout_name=EXPECTED_READOUT,
        alpha=float(candidate["alpha"]),
        dt=float(candidate["dt"]),
        hx=float(candidate["hx"]),
        hy=float(candidate["hy"]),
        j_json={
            f"J{i}{j}": float(v)
            for (i, j), v in candidate["J"].items()
        },
        manifest_json=manifest_meta,
        ridge_lambda=float(frozen["lambda"]),
        training_cv_rmse=float(frozen["cv_rmse"]),
        ideal_validation_rmse_full_2025=float(frozen["ideal_rmse"]),
        ideal_validation_mae_full_2025=float(frozen["ideal_mae"]),
        ideal_validation_bias_full_2025=float(frozen["ideal_bias"]),
        backend_name=args.backend,
        shots_per_setting=int(args.shots),
        n_validation_endpoints=int(len(selected_endpoints)),
        full_2025=bool(args.full),
        pilot_selection=(
            "all_2025_validation"
            if args.full
            else "deterministic_evenly_spaced_2025_endpoints"
        ),
        optimization_level=int(OPT_LEVEL),
        seed_transpiler=int(SEED_TRANSPILE),
        notes=(
            "Week 11.1C direct-QPU transfer. "
            "2022-2024 establishes/fits the readout; 2025 is validation; "
            "2026 is not loaded."
        ),
    )
    CURRENT_RUN_UUID = run_uuid

    print(f"[database] QPU run UUID: {run_uuid}")
    print()

    service = get_service()

    # Fresh H2 reselection on the chosen backend immediately before compile/run.
    print("=" * 126)
    print(f"FRESH {args.backend} H2 HARDWARE RESELECTION")
    print("=" * 126)

    fresh = selector.fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=[args.backend],
        shortlist=SHORTLIST,
        write_prefix=f"11_1C_{CANDIDATE_KEY}_{args.backend}",
        verbose=True,
    )

    chosen = fresh["selected"]
    layout = json.loads(chosen["layout"])

    print()
    print("Fresh physical choice:")
    print(f"  backend={args.backend}")
    print(f"  layout={layout}")
    print(
        f"  max CZ error={float(chosen.get('compiled_2q_error_max_percent', np.nan)):.6f}%"
    )
    print(
        f"  max readout error={float(chosen.get('max_readout_error_percent', np.nan)):.6f}%"
    )
    print(
        f"  max 1Q error={float(chosen.get('compiled_1q_error_max_percent', np.nan)):.6f}%"
    )
    print(f"  min T1={float(chosen.get('min_t1_us', np.nan)):.3f} us")
    print(f"  min T2={float(chosen.get('min_t2_us', np.nan)):.3f} us")
    print()

    backend = service.backend(
        args.backend,
        use_fractional_gates=False,
    )
    try:
        backend.refresh()
    except Exception:
        pass

    circuits = []
    circuit_meta = []
    resource_rows = []

    settings = readout_settings()

    for endpoint in selected_endpoints:
        for setting in settings:
            logical = build_measurement_circuit(
                candidate,
                frozen["angles"],
                int(endpoint),
                setting,
            )

            isa = transpile(
                logical,
                backend=backend,
                initial_layout=layout,
                routing_method="none",
                optimization_level=OPT_LEVEL,
                seed_transpiler=SEED_TRANSPILE,
                scheduling_method="alap",
            )

            backend.check_faulty(isa)

            rr = resource_row(
                isa,
                backend,
                endpoint=int(endpoint),
                setting=setting[0],
            )

            if rr["n_swap"] != 0:
                raise RuntimeError(
                    f"Strict native compile failed: SWAP detected at "
                    f"endpoint={endpoint}, setting={setting[0]}."
                )

            circuits.append(isa)
            circuit_meta.append(
                {
                    "endpoint": int(endpoint),
                    "setting": setting[0],
                }
            )
            resource_rows.append(rr)

    resources = pd.DataFrame(resource_rows)
    resources.to_csv(
        RESULTS / "11_1C_third_candidate_qpu_resources.csv",
        index=False,
    )
    qpu_db.save_dataframe_artifact(
        run_uuid,
        "11_1C_third_candidate_qpu_resources.csv",
        resources,
    )

    feature_resources = (
        resources.groupby("endpoint", as_index=False)
        .agg(
            n_settings=("setting", "nunique"),
            cz_feature_vector=("n_cz", "sum"),
            swaps_feature_vector=("n_swap", "sum"),
            max_setting_depth=("depth", "max"),
            duration_feature_vector_us=("duration_us", "sum"),
            max_setting_duration_us=("duration_us", "max"),
            resets_feature_vector=("n_reset", "sum"),
        )
    )

    scheduled_shot_seconds = float(
        np.sum(
            resources["duration_us"].to_numpy(dtype=float)
            * 1e-6
            * int(args.shots)
        )
    )
    estimated_usage_seconds = scheduled_shot_seconds * EMPIRICAL_USAGE_SCALE

    usage_before = get_service_usage(service)

    print("=" * 126)
    print("QPU PREFLIGHT")
    print("=" * 126)
    print(f"Backend:                    {args.backend}")
    print(f"Physical layout:            {layout}")
    print(f"Validation endpoints:       {len(selected_endpoints)}")
    print(f"Circuits:                   {len(circuits)}")
    print(f"Settings/endpoint:          2")
    print(f"Shots/setting:              {args.shots}")
    print(
        f"Median CZ/feature vector:   "
        f"{feature_resources['cz_feature_vector'].median():.0f}"
    )
    print(
        f"Median max setting depth:   "
        f"{feature_resources['max_setting_depth'].median():.0f}"
    )
    print(
        f"Median feature duration:    "
        f"{feature_resources['duration_feature_vector_us'].median():.3f} us"
    )
    print(
        f"Median max-setting duration:"
        f" {feature_resources['max_setting_duration_us'].median():.3f} us"
    )
    print(
        f"Scheduled-duration*shots:   {scheduled_shot_seconds:.6f} s"
    )
    print(
        f"Empirical usage estimate:   {estimated_usage_seconds:.1f} s "
        "(planning heuristic only)"
    )

    if usage_before is not None:
        print()
        print("Service usage before:")
        print(json.dumps(usage_before, indent=2))

    preflight = {
        "candidate_key": CANDIDATE_KEY,
        "backend": args.backend,
        "layout": layout,
        "window": EXPECTED_WINDOW,
        "r": EXPECTED_R,
        "shots": int(args.shots),
        "full_2025": bool(args.full),
        "n_endpoints": int(len(selected_endpoints)),
        "n_circuits": int(len(circuits)),
        "ideal_validation_rmse_full_2025": frozen["ideal_rmse"],
        "training_cv_rmse": frozen["cv_rmse"],
        "ridge_lambda": frozen["lambda"],
        "scheduled_duration_times_shots_s": scheduled_shot_seconds,
        "empirical_usage_estimate_s": estimated_usage_seconds,
        "hardware_selection": json_safe(chosen.to_dict() if hasattr(chosen, "to_dict") else chosen),
        "service_usage_before": usage_before,
    }

    (
        RESULTS / "11_1C_third_candidate_qpu_preflight.json"
    ).write_text(
        json.dumps(json_safe(preflight), indent=2),
        encoding="utf-8",
    )

    selected_endpoints_df = pd.DataFrame(
        {"endpoint": selected_endpoints}
    )
    selected_endpoints_df.to_csv(
        RESULTS / "11_1C_third_candidate_qpu_selected_endpoints.csv",
        index=False,
    )

    chosen_dict = json_safe(
        chosen.to_dict() if hasattr(chosen, "to_dict") else chosen
    )

    qpu_db.save_json_artifact(
        run_uuid,
        "11_1C_third_candidate_qpu_preflight.json",
        preflight,
    )
    qpu_db.save_dataframe_artifact(
        run_uuid,
        "11_1C_third_candidate_qpu_selected_endpoints.csv",
        selected_endpoints_df,
    )

    qpu_db.update_run(
        run_uuid,
        run_status="PREFLIGHT_COMPLETE",
        physical_layout_json=layout,
        hardware_selection_json=chosen_dict,
        max_2q_error_percent=float(
            chosen.get("compiled_2q_error_max_percent", np.nan)
        ),
        max_readout_error_percent=float(
            chosen.get("max_readout_error_percent", np.nan)
        ),
        max_1q_error_percent=float(
            chosen.get("compiled_1q_error_max_percent", np.nan)
        ),
        min_t1_us=float(chosen.get("min_t1_us", np.nan)),
        min_t2_us=float(chosen.get("min_t2_us", np.nan)),
        compiled_duration_us=float(
            chosen.get("compiled_duration_us", np.nan)
        ),
        r_t2=float(chosen.get("R_T2", np.nan)),
        n_circuits=int(len(circuits)),
        median_cz_per_feature=float(
            feature_resources["cz_feature_vector"].median()
        ),
        median_max_setting_depth=float(
            feature_resources["max_setting_depth"].median()
        ),
        median_feature_duration_us=float(
            feature_resources["duration_feature_vector_us"].median()
        ),
        median_max_setting_duration_us=float(
            feature_resources["max_setting_duration_us"].median()
        ),
        scheduled_duration_times_shots_s=float(scheduled_shot_seconds),
        empirical_usage_estimate_s=float(estimated_usage_seconds),
        service_usage_before_json=usage_before,
    )

    if not args.submit:
        qpu_db.update_run(
            run_uuid,
            run_status="DRY_RUN_COMPLETE",
            completed_at_utc=utc_now_naive(),
        )
        print()
        print("DRY RUN COMPLETE — NO QPU JOB SUBMITTED.")
        print()
        if args.full:
            print(
                "To submit the FULL 2025 validation after reviewing this preflight:"
            )
            print(
                f"python {Path(__file__).name} --full --submit "
                f"--backend {args.backend} --shots {args.shots}"
            )
        else:
            print("To submit this pilot after reviewing the preflight:")
            print(
                f"python {Path(__file__).name} --submit "
                f"--backend {args.backend} --shots {args.shots} "
                f"--pilot-size {len(selected_endpoints)}"
            )
        return

    # Optional budget guard when the service exposes remaining seconds.
    remaining = None
    if isinstance(usage_before, dict):
        for key in [
            "usage_remaining_seconds",
            "remaining_seconds",
            "usage_remaining",
        ]:
            if key in usage_before:
                try:
                    remaining = float(usage_before[key])
                    break
                except Exception:
                    pass

    if (
        remaining is not None
        and estimated_usage_seconds > remaining
        and not args.force_over_budget
    ):
        raise RuntimeError(
            f"Submission blocked by safety guard: empirical usage estimate "
            f"{estimated_usage_seconds:.1f}s exceeds reported remaining "
            f"usage {remaining:.1f}s. Re-run only if appropriate with "
            f"--force-over-budget."
        )

    print()
    print("=" * 126)
    print("SUBMITTING ONE SamplerV2 JOB IN JOB MODE")
    print("=" * 126)
    print(
        f"Submitting {len(circuits)} ISA circuits at "
        f"{args.shots} shots/setting..."
    )

    qpu_db.update_run(
        run_uuid,
        run_status="SUBMITTING",
        submitted_at_utc=utc_now_naive(),
    )

    sampler = SamplerV2(mode=backend)
    job = sampler.run(
        circuits,
        shots=int(args.shots),
    )

    job_id = job.job_id()
    print(f"Job ID: {job_id}")

    qpu_db.update_run(
        run_uuid,
        run_status="SUBMITTED",
        job_id=str(job_id),
    )

    result = job.result()
    final_job_status = str(job.status())
    print(f"Final status: {final_job_status}")

    metrics = None
    try:
        metrics = json_safe(job.metrics())
    except Exception as exc:
        print(f"[timing] WARNING: job.metrics() unavailable: {exc}")

    qpu_timing = extract_ibm_qpu_timing(
        job=job,
        metrics=metrics,
    )

    # Keep the old field as an alias for backwards compatibility.
    job_usage_seconds = qpu_timing["qpu_charge_time_seconds"]

    print()
    print("IBM QPU TIMING")
    print("-" * 126)

    if qpu_timing["qpu_charge_time_seconds"] is not None:
        print(
            "QPU charge / locked time:       "
            f"{qpu_timing['qpu_charge_time_seconds']:.6f} s"
        )
    else:
        print("QPU charge / locked time:       unavailable")

    if qpu_timing["qpu_circuits_execution_time_seconds"] is not None:
        print(
            "Actual circuit execution on QPU:"
            f" {qpu_timing['qpu_circuits_execution_time_seconds']:.6f} s"
        )
    else:
        print("Actual circuit execution on QPU: unavailable")

    if qpu_timing["qpu_running_wall_seconds"] is not None:
        print(
            "IBM RUNNING -> FINISHED elapsed:"
            f" {qpu_timing['qpu_running_wall_seconds']:.6f} s"
        )
    else:
        print("IBM RUNNING -> FINISHED elapsed: unavailable")

    if qpu_timing["ibm_job_running_at_utc"] is not None:
        print(
            "IBM job started running (UTC):  "
            f"{qpu_timing['ibm_job_running_at_utc'].isoformat()}Z"
        )

    if qpu_timing["ibm_job_finished_at_utc"] is not None:
        print(
            "IBM job finished (UTC):         "
            f"{qpu_timing['ibm_job_finished_at_utc'].isoformat()}Z"
        )

    print()

    job_payload = {
        "job_id": job_id,
        "backend": args.backend,
        "layout": layout,
        "shots": int(args.shots),
        "n_circuits": int(len(circuits)),
        "status": final_job_status,
        "job_usage_seconds": job_usage_seconds,
        "qpu_charge_time_seconds": qpu_timing["qpu_charge_time_seconds"],
        "qpu_circuits_execution_time_seconds": (
            qpu_timing["qpu_circuits_execution_time_seconds"]
        ),
        "qpu_running_wall_seconds": qpu_timing["qpu_running_wall_seconds"],
        "ibm_job_created_at_utc": qpu_timing["ibm_job_created_at_utc"],
        "ibm_job_running_at_utc": qpu_timing["ibm_job_running_at_utc"],
        "ibm_job_finished_at_utc": qpu_timing["ibm_job_finished_at_utc"],
        "metrics": metrics,
    }

    (
        RESULTS / "11_1C_third_candidate_qpu_job.json"
    ).write_text(
        json.dumps(
            json_safe(job_payload),
            indent=2,
        ),
        encoding="utf-8",
    )

    qpu_db.save_json_artifact(
        run_uuid,
        "11_1C_third_candidate_qpu_job.json",
        job_payload,
    )
    qpu_db.update_run(
        run_uuid,
        run_status="QPU_COMPLETE_PARSING",
        job_status=final_job_status,
        job_usage_seconds=job_usage_seconds,
        qpu_charge_time_seconds=qpu_timing["qpu_charge_time_seconds"],
        qpu_circuits_execution_time_seconds=(
            qpu_timing["qpu_circuits_execution_time_seconds"]
        ),
        qpu_running_wall_seconds=qpu_timing["qpu_running_wall_seconds"],
        ibm_job_created_at_utc=qpu_timing["ibm_job_created_at_utc"],
        ibm_job_running_at_utc=qpu_timing["ibm_job_running_at_utc"],
        ibm_job_finished_at_utc=qpu_timing["ibm_job_finished_at_utc"],
        job_metrics_json=metrics,
    )

    # Parse the two grouped measurement settings for every endpoint.
    by_endpoint = {
        int(e): {}
        for e in selected_endpoints
    }

    raw_rows = []

    if len(result) != len(circuits):
        raise RuntimeError(
            f"Sampler returned {len(result)} pubs for {len(circuits)} circuits."
        )

    for pub_result, meta in zip(result, circuit_meta):
        counts = get_pub_counts(pub_result)
        endpoint = int(meta["endpoint"])
        setting = str(meta["setting"])

        raw_rows.append(
            {
                "endpoint": endpoint,
                "setting": setting,
                "counts_json": json.dumps(counts, sort_keys=True),
            }
        )

        if setting == "XXXZ":
            by_endpoint[endpoint]["X0"] = expectation_from_counts(counts, [0])
            by_endpoint[endpoint]["X1"] = expectation_from_counts(counts, [1])
            by_endpoint[endpoint]["X2"] = expectation_from_counts(counts, [2])
            by_endpoint[endpoint]["Z3_from_X_setting"] = expectation_from_counts(
                counts, [3]
            )

        elif setting == "ZZZZ":
            by_endpoint[endpoint]["Z0"] = expectation_from_counts(counts, [0])
            by_endpoint[endpoint]["Z1"] = expectation_from_counts(counts, [1])
            by_endpoint[endpoint]["Z2"] = expectation_from_counts(counts, [2])
            by_endpoint[endpoint]["Z3"] = expectation_from_counts(counts, [3])

        else:
            raise RuntimeError(f"Unexpected measurement setting {setting}.")

    raw_counts_df = pd.DataFrame(raw_rows)
    raw_counts_df.to_csv(
        RESULTS / "11_1C_third_candidate_qpu_raw_counts.csv",
        index=False,
    )
    qpu_db.save_dataframe_artifact(
        run_uuid,
        "11_1C_third_candidate_qpu_raw_counts.csv",
        raw_counts_df,
    )

    # Align ideal rows by global endpoint.
    endpoint_to_master_row = {
        int(endpoint): idx
        for idx, endpoint in enumerate(frozen["endpoints"])
    }
    val_endpoint_to_pos = {
        int(endpoint): idx
        for idx, endpoint in enumerate(frozen["val_endpoints"])
    }

    feature_rows = []
    X_qpu = []
    X_ideal = []
    targets = []
    ideal_preds = []

    for endpoint in selected_endpoints:
        endpoint = int(endpoint)
        vals = by_endpoint[endpoint]

        missing = [f for f in FEATURES if f not in vals]
        if missing:
            raise RuntimeError(
                f"Endpoint {endpoint}: missing QPU features {missing}."
            )

        qpu_vec = np.array(
            [float(vals[f]) for f in FEATURES],
            dtype=float,
        )

        master_idx = endpoint_to_master_row[endpoint]
        ideal_vec = frozen["X"][master_idx]

        val_pos = val_endpoint_to_pos[endpoint]
        target = float(frozen["y_val"][val_pos])
        ideal_pred = float(frozen["pred_val"][val_pos])

        X_qpu.append(qpu_vec)
        X_ideal.append(ideal_vec)
        targets.append(target)
        ideal_preds.append(ideal_pred)

        row = {
            "endpoint": endpoint,
            "target": target,
            "pred_ideal": ideal_pred,
            "Z3_from_X_setting": float(vals["Z3_from_X_setting"]),
            "Z3_from_Z_setting": float(vals["Z3"]),
            "Z3_setting_difference": float(
                vals["Z3_from_X_setting"] - vals["Z3"]
            ),
        }

        for j, feature in enumerate(FEATURES):
            row[f"{feature}_ideal"] = float(ideal_vec[j])
            row[f"{feature}_qpu"] = float(qpu_vec[j])
            row[f"{feature}_error"] = float(qpu_vec[j] - ideal_vec[j])

        feature_rows.append(row)

    X_qpu = np.asarray(X_qpu, dtype=float)
    X_ideal = np.asarray(X_ideal, dtype=float)
    targets = np.asarray(targets, dtype=float)
    ideal_preds = np.asarray(ideal_preds, dtype=float)

    pred_qpu = common.predict_scaled_ridge(
        frozen["model"],
        frozen["scaler"],
        frozen["keep"],
        X_qpu,
    )

    features_df = pd.DataFrame(feature_rows)
    features_df["pred_qpu"] = pred_qpu
    features_df["abs_error_ideal"] = np.abs(
        targets - ideal_preds
    )
    features_df["abs_error_qpu"] = np.abs(
        targets - pred_qpu
    )

    features_df.to_csv(
        RESULTS / "11_1C_third_candidate_qpu_features_predictions.csv",
        index=False,
    )
    qpu_db.save_dataframe_artifact(
        run_uuid,
        "11_1C_third_candidate_qpu_features_predictions.csv",
        features_df,
    )

    feature_diff = X_qpu - X_ideal

    summary = {
        "run_uuid": run_uuid,
        "candidate_key": CANDIDATE_KEY,
        "backend": args.backend,
        "layout": layout,
        "job_id": job_id,
        "shots_per_setting": int(args.shots),
        "n_validation_endpoints": int(len(selected_endpoints)),
        "full_2025": bool(args.full),
        "pilot_selection": (
            "all_2025_validation"
            if args.full
            else "deterministic_evenly_spaced_2025_endpoints"
        ),
        "qpu_rmse": rmse(targets, pred_qpu),
        "qpu_mae": mae(targets, pred_qpu),
        "qpu_bias": bias(targets, pred_qpu),
        "ideal_same_subset_rmse": rmse(targets, ideal_preds),
        "ideal_same_subset_mae": mae(targets, ideal_preds),
        "ideal_same_subset_bias": bias(targets, ideal_preds),
        "qpu_feature_mae": float(np.mean(np.abs(feature_diff))),
        "qpu_feature_rmse": float(np.sqrt(np.mean(feature_diff ** 2))),
        "z3_cross_setting_mae": float(
            np.mean(np.abs(features_df["Z3_setting_difference"]))
        ),
        "prediction_correlation_qpu_vs_ideal": float(
            np.corrcoef(pred_qpu, ideal_preds)[0, 1]
        ),
        "job_usage_seconds": job_usage_seconds,
        "qpu_charge_time_seconds": qpu_timing["qpu_charge_time_seconds"],
        "qpu_circuits_execution_time_seconds": (
            qpu_timing["qpu_circuits_execution_time_seconds"]
        ),
        "qpu_running_wall_seconds": qpu_timing["qpu_running_wall_seconds"],
        "ibm_job_created_at_utc": qpu_timing["ibm_job_created_at_utc"],
        "ibm_job_running_at_utc": qpu_timing["ibm_job_running_at_utc"],
        "ibm_job_finished_at_utc": qpu_timing["ibm_job_finished_at_utc"],
        "job_metrics": metrics,
        "service_usage_before": usage_before,
        "service_usage_after": get_service_usage(service),
    }

    (
        RESULTS / "11_1C_third_candidate_qpu_summary.json"
    ).write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )

    qpu_db.save_json_artifact(
        run_uuid,
        "11_1C_third_candidate_qpu_summary.json",
        summary,
    )

    qpu_db.update_run(
        run_uuid,
        run_status="COMPLETED",
        completed_at_utc=utc_now_naive(),
        job_status=final_job_status,
        job_usage_seconds=job_usage_seconds,
        qpu_charge_time_seconds=qpu_timing["qpu_charge_time_seconds"],
        qpu_circuits_execution_time_seconds=(
            qpu_timing["qpu_circuits_execution_time_seconds"]
        ),
        qpu_running_wall_seconds=qpu_timing["qpu_running_wall_seconds"],
        ibm_job_created_at_utc=qpu_timing["ibm_job_created_at_utc"],
        ibm_job_running_at_utc=qpu_timing["ibm_job_running_at_utc"],
        ibm_job_finished_at_utc=qpu_timing["ibm_job_finished_at_utc"],
        qpu_rmse=float(summary["qpu_rmse"]),
        qpu_mae=float(summary["qpu_mae"]),
        qpu_bias=float(summary["qpu_bias"]),
        ideal_same_subset_rmse=float(summary["ideal_same_subset_rmse"]),
        ideal_same_subset_mae=float(summary["ideal_same_subset_mae"]),
        ideal_same_subset_bias=float(summary["ideal_same_subset_bias"]),
        qpu_feature_mae=float(summary["qpu_feature_mae"]),
        qpu_feature_rmse=float(summary["qpu_feature_rmse"]),
        z3_cross_setting_mae=float(summary["z3_cross_setting_mae"]),
        prediction_correlation_qpu_vs_ideal=float(
            summary["prediction_correlation_qpu_vs_ideal"]
        ),
        service_usage_after_json=summary["service_usage_after"],
    )

    print(f"[database] Run {run_uuid} persisted successfully.")
    print()
    print("=" * 126)
    print("REAL-QPU THIRD-CANDIDATE RESULTS")
    print("=" * 126)
    print(f"Endpoints tested:               {len(selected_endpoints)}")
    print(f"Shots/setting:                 {args.shots}")
    print(f"QPU RMSE:                      {summary['qpu_rmse']:.6f}")
    print(f"QPU MAE:                       {summary['qpu_mae']:.6f}")
    print(f"QPU bias:                      {summary['qpu_bias']:+.6f}")
    print(f"Ideal same-subset RMSE:        {summary['ideal_same_subset_rmse']:.6f}")
    print(f"QPU feature MAE:               {summary['qpu_feature_mae']:.6f}")
    print(f"QPU feature RMSE:              {summary['qpu_feature_rmse']:.6f}")
    print(
        f"QPU-vs-ideal pred correlation: "
        f"{summary['prediction_correlation_qpu_vs_ideal']:.6f}"
    )
    print(
        f"Z3 cross-setting MAE:          "
        f"{summary['z3_cross_setting_mae']:.6f}"
    )

    if summary["qpu_charge_time_seconds"] is not None:
        print(
            f"QPU charge / locked time:      "
            f"{summary['qpu_charge_time_seconds']:.6f} s"
        )

    if summary["qpu_circuits_execution_time_seconds"] is not None:
        print(
            f"Actual circuit execution QPU:  "
            f"{summary['qpu_circuits_execution_time_seconds']:.6f} s"
        )

    if summary["qpu_running_wall_seconds"] is not None:
        print(
            f"IBM running elapsed time:      "
            f"{summary['qpu_running_wall_seconds']:.6f} s"
        )

    print()
    print("2026 remains FROZEN / UNUSED.")
    print()
    print("Saved:")
    print("  results/11_1C_third_candidate_qpu_preflight.json")
    print("  results/11_1C_third_candidate_qpu_resources.csv")
    print("  results/11_1C_third_candidate_qpu_selected_endpoints.csv")
    print("  results/11_1C_third_candidate_qpu_raw_counts.csv")
    print("  results/11_1C_third_candidate_qpu_features_predictions.csv")
    print("  results/11_1C_third_candidate_qpu_summary.json")
    print("  results/11_1C_third_candidate_qpu_job.json")


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        if CURRENT_RUN_UUID is not None:
            try:
                qpu_db.update_run(
                    CURRENT_RUN_UUID,
                    run_status="FAILED",
                    completed_at_utc=utc_now_naive(),
                    error_message=str(exc),
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
