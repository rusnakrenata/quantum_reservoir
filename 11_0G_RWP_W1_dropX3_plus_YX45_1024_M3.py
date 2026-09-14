
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler

from ibm_account import get_service

try:
    import mthree
except ImportError as exc:
    raise ImportError(
        'Install M3 first with: pip install "mthree>=3.0"'
    ) from exc


# =============================================================================
# WEEK 11.0G — FINAL QPU TEST
#
# H4 + RWP_W1 + 1024 shots + grouped memory YX45 observable
#
# Feature vector:
#
#   [X0, X1, X2, Z0, Z1, Z2, Z3, YX45]
#
# IMPORTANT:
#   X3 IS NOT USED.
#
# Two grouped measurement settings:
#
#   A = XXXZYX
#       -> X0, X1, X2, Y4*X5
#       -> q3 is measured in Z, never X
#
#   B = ZZZZZZ
#       -> Z0, Z1, Z2, Z3
#
# Both settings measure all six qubits so the same M3 calibration/mapping
# can be reused.
#
# Chronology:
#   2022-2024 = establish lambda/scaler/Ridge weights/intercept
#   2025      = validation only
#   2026      = frozen / unused
#
# RAW and M3-mitigated results are produced from the SAME QPU counts.
#
# Dry run:
#   python 11_0G_RWP_W1_dropX3_plus_YX45_1024_M3.py
#
# Submit:
#   python 11_0G_RWP_W1_dropX3_plus_YX45_1024_M3.py --submit
#
# =============================================================================


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

BASE_SCRIPT = HERE / "11_0D_RWP_W1_full_2025_validation.py"

N_TRAIN = 1095
N_VAL = 365

DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SHOTS = 1024
DEFAULT_SHORTLIST = 80

OPTIMIZATION_LEVEL = 1
SEED_TRANSPILER = 42

FEATURES = [
    "X0", "X1", "X2",
    "Z0", "Z1", "Z2", "Z3",
    "YX45",
]


# =============================================================================
# IMPORT VALIDATED 11.0D BASE
# =============================================================================

def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put this script beside 11.0D."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    spec.loader.exec_module(module)
    return module


base = import_module_from_path(
    "week11_0D_base_for_11_0G",
    BASE_SCRIPT,
)

qrc = base.qrc


# =============================================================================
# IDEAL YX45 FEATURE
# =============================================================================

MEM_OPS = getattr(
    qrc,
    "MEMORY_SINGLE_OPS",
    getattr(qrc, "MEMORY_OPS", None),
)

if MEM_OPS is None:
    raise AttributeError(
        "Could not resolve memory-qubit Pauli operators from Week-7 QRC module."
    )

YX45_MEM = MEM_OPS["Y4"] @ MEM_OPS["X5"]


def expectation(rho, op):
    return float(
        np.real_if_close(
            np.trace(rho @ op),
            tol=1000,
        ).real
    )


def build_ideal_w1_dropx3_yx45(candidate, angles):
    U = base.build_trotter_unitary(candidate)

    A_list, _ = qrc.build_input_channels(
        U,
        angles,
    )

    rho_m_zero = qrc.memory_zero_density()

    rows = []

    for A in A_list:
        rho_i, rho_m = qrc.final_reduced_states(
            A,
            rho_m_zero,
        )

        row = [
            expectation(rho_i, base.INJ_OPS["X0"]),
            expectation(rho_i, base.INJ_OPS["X1"]),
            expectation(rho_i, base.INJ_OPS["X2"]),
            expectation(rho_i, base.INJ_OPS["Z0"]),
            expectation(rho_i, base.INJ_OPS["Z1"]),
            expectation(rho_i, base.INJ_OPS["Z2"]),
            expectation(rho_i, base.INJ_OPS["Z3"]),
            expectation(rho_m, YX45_MEM),
        ]

        rows.append(row)

    return np.asarray(rows, dtype=float)


# =============================================================================
# GROUPED CIRCUITS
# =============================================================================

def build_w1_grouped_circuit(
    candidate,
    angle_row,
    setting,
    endpoint,
):
    qreg = QuantumRegister(6, "q")
    creg = ClassicalRegister(6, "m")

    qc = QuantumCircuit(
        qreg,
        creg,
        name=f"W1_t{endpoint}_{setting}",
    )

    base.append_one_h4_step(
        qc,
        candidate,
        angle_row,
    )

    if setting == "XXXZYX":
        # X0, X1, X2
        for q in [0, 1, 2]:
            qc.h(q)

        # q3 stays Z.

        # Y4 measurement: S^\dagger then H.
        qc.sdg(4)
        qc.h(4)

        # X5 measurement.
        qc.h(5)

    elif setting == "ZZZZZZ":
        pass

    else:
        raise ValueError(setting)

    # Measure all six so both settings share the same M3 measurement map.
    for q in range(6):
        qc.measure(q, q)

    return qc


# =============================================================================
# COUNTS / QUASI DISTRIBUTIONS -> OBSERVABLES
# =============================================================================

def clean_bitstring(key, n_bits=6):
    if isinstance(key, (int, np.integer)):
        return format(int(key), f"0{n_bits}b")

    s = str(key).replace(" ", "")

    if s.startswith("0x"):
        return format(int(s, 16), f"0{n_bits}b")

    if s.startswith("0b"):
        s = s[2:]

    return s.zfill(n_bits)


def expectation_from_distribution(dist, q, n_bits=6):
    total = 0.0
    acc = 0.0

    for key, weight in dist.items():
        bits = clean_bitstring(key, n_bits=n_bits)
        bit = int(bits[-(q + 1)])

        eig = 1.0 if bit == 0 else -1.0

        w = float(weight)
        total += w
        acc += eig * w

    if np.isclose(total, 0.0):
        raise RuntimeError("Near-zero distribution normalization.")

    return acc / total


def parity_y4x5_from_distribution(dist, n_bits=6):
    total = 0.0
    acc = 0.0

    for key, weight in dist.items():
        bits = clean_bitstring(key, n_bits=n_bits)

        b4 = int(bits[-5])
        b5 = int(bits[-6])

        eig = 1.0 if ((b4 + b5) % 2 == 0) else -1.0

        w = float(weight)
        total += w
        acc += eig * w

    if np.isclose(total, 0.0):
        raise RuntimeError("Near-zero distribution normalization.")

    return acc / total


def grouped_features(dist_grouped, dist_z):
    return np.asarray(
        [
            expectation_from_distribution(dist_grouped, 0),
            expectation_from_distribution(dist_grouped, 1),
            expectation_from_distribution(dist_grouped, 2),
            expectation_from_distribution(dist_z, 0),
            expectation_from_distribution(dist_z, 1),
            expectation_from_distribution(dist_z, 2),
            expectation_from_distribution(dist_z, 3),
            parity_y4x5_from_distribution(dist_grouped),
        ],
        dtype=float,
    )


# =============================================================================
# HARDWARE LAYOUT — ALL SIX QUBITS MATTER FOR READOUT NOW
# =============================================================================

def choose_layout_all6_readout(fresh, backend_name):
    good = fresh["compiled"].copy()

    good = good[
        (good["backend"] == backend_name)
        &
        (good["strict_compile_pass"] == True)
    ].copy()

    if len(good) == 0:
        raise RuntimeError(
            f"No fresh strict zero-SWAP layout on {backend_name}."
        )

    # Explicit hierarchy; no weighted score.
    good = good.sort_values(
        [
            "compiled_2q_error_max",
            "compiled_2q_error_mean",
            "max_readout_error",
            "mean_readout_error",
            "compiled_1q_error_max",
            "min_t2_us",
            "min_t1_us",
            "compiled_duration_us",
        ],
        ascending=[
            True, True,
            True, True,
            True,
            False, False,
            True,
        ],
    ).reset_index(drop=True)

    good["final_QPU_rank"] = np.arange(len(good)) + 1

    return good.iloc[0], good


# =============================================================================
# UTILITIES
# =============================================================================

def safe_service_usage(service):
    try:
        return service.usage()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }


def json_safe(obj):
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass

    return str(obj)


def metrics(y_true, pred):
    return {
        "rmse": base.rmse(y_true, pred),
        "mae": base.mae(y_true, pred),
        "bias": base.bias(y_true, pred),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submit",
        action="store_true",
    )

    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=[
            "ibm_fez",
            "ibm_kingston",
            "ibm_marrakesh",
        ],
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
    )

    parser.add_argument(
        "--shortlist",
        type=int,
        default=DEFAULT_SHORTLIST,
    )

    args = parser.parse_args()

    # =========================================================================
    # FROZEN LOGICAL MODEL / TRAINING-ONLY READOUT
    # =========================================================================

    candidate = base.load_h4_winner_dynamics()

    (
        work_tv,
        cols,
        angles,
        y,
        train_idx,
        val_idx,
    ) = base.load_train_validation_only()

    X_ideal = build_ideal_w1_dropx3_yx45(
        candidate,
        angles,
    )

    X_train = X_ideal[train_idx]
    y_train = y[train_idx]

    X_val = X_ideal[val_idx]
    y_val = y[val_idx]

    (
        selected_lambda,
        cv_folds,
        cv_summary,
    ) = base.select_lambda_training_only(
        X_train,
        y_train,
    )

    ridge, scaler, keep = base.fit_scaled_ridge(
        X_train,
        y_train,
        selected_lambda,
    )

    pred_ideal = base.predict_scaled_ridge(
        ridge,
        scaler,
        keep,
        X_val,
    )

    ideal_metrics = metrics(
        y_val,
        pred_ideal,
    )

    print("=" * 120)
    print("WEEK 11.0G — FINAL QPU TEST: DROP X3 + ADD MEMORY YX45")
    print("=" * 120)
    print()
    print("Feature vector:")
    print("  [X0, X1, X2, Z0, Z1, Z2, Z3, YX45]")
    print()
    print("Grouped measurement settings:")
    print("  A = XXXZYX")
    print("  B = ZZZZZZ")
    print()
    print("X3 is NOT measured as an X observable.")
    print()
    print("Chronology:")
    print("  2022-2024 = lambda/scaler/Ridge weights/intercept")
    print("  2025      = validation only")
    print("  2026      = frozen / unused")
    print()
    print(
        f"H4, r={candidate['r']}, "
        f"hx={candidate['hx']:+.6f}, "
        f"hy={candidate['hy']:+.6f}"
    )
    print(f"Selected lambda: {selected_lambda}")
    print(
        f"Training-CV RMSE: "
        f"{float(cv_summary.iloc[0]['cv_rmse_mean']):.6f}"
    )
    print(
        f"Ideal 2025 RMSE: {ideal_metrics['rmse']:.6f}"
    )
    print(
        f"Ideal 2025 MAE:  {ideal_metrics['mae']:.6f}"
    )
    print(
        f"Ideal 2025 bias: {ideal_metrics['bias']:+.6f}"
    )
    print(f"Shots/setting:    {args.shots}")

    # =========================================================================
    # FRESH HARDWARE SCAN
    # =========================================================================

    service = get_service()

    print()
    print("=" * 120)
    print(f"FRESH {args.backend} HARDWARE SCAN")
    print("=" * 120)

    fresh = base.selector.fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=[args.backend],
        shortlist=args.shortlist,
        write_prefix="11_0G_hardware",
        verbose=True,
    )

    selected_layout_row, ranking = choose_layout_all6_readout(
        fresh,
        args.backend,
    )

    ranking.to_csv(
        RESULTS / "11_0G_hardware_ranking_all6.csv",
        index=False,
    )

    layout = json.loads(
        selected_layout_row["layout"]
    )

    print()
    print("Fresh physical choice:")
    print(f"  layout={layout}")
    print(
        f"  max CZ error="
        f"{selected_layout_row['compiled_2q_error_max_percent']:.4f}%"
    )
    print(
        f"  max all-6 readout error="
        f"{selected_layout_row['max_readout_error_percent']:.4f}%"
    )
    print(
        f"  mean all-6 readout error="
        f"{100.0 * selected_layout_row['mean_readout_error']:.4f}%"
    )
    print(
        f"  min T2="
        f"{selected_layout_row['min_t2_us']:.2f} us"
    )

    backend = service.backend(
        args.backend,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    # =========================================================================
    # BUILD 365 × 2 CIRCUITS
    # =========================================================================

    logical = []
    metadata = []

    for endpoint in val_idx:
        for setting in [
            "XXXZYX",
            "ZZZZZZ",
        ]:
            qc = build_w1_grouped_circuit(
                candidate,
                angles[int(endpoint)],
                setting,
                int(endpoint),
            )

            logical.append(qc)

            metadata.append({
                "endpoint_global":
                    int(endpoint),
                "validation_offset":
                    int(endpoint - N_TRAIN),
                "setting":
                    setting,
            })

    isa = transpile(
        logical,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
        scheduling_method="alap",
    )

    if not isinstance(isa, list):
        isa = [isa]

    audit_rows = []

    for meta, circuit in zip(metadata, isa):
        backend.check_faulty(circuit)

        row = base.circuit_resource_row(
            circuit,
            backend,
            meta,
        )

        if row["n_swap"] != 0:
            raise RuntimeError(
                "SWAP detected; refusing final controlled QPU test."
            )

        audit_rows.append(row)

    audit_df = pd.DataFrame(audit_rows)

    audit_df[
        "scheduled_time_all_shots_s"
    ] = (
        audit_df["duration_us"]
        *
        1e-6
        *
        int(args.shots)
    )

    audit_df.to_csv(
        RESULTS / "11_0G_transpilation.csv",
        index=False,
    )

    # =========================================================================
    # M3 MAPPING — BOTH SETTINGS MEASURE ALL SIX
    # =========================================================================

    mappings = [
        mthree.utils.final_measurement_mapping(circuit)
        for circuit in isa
    ]

    first_mapping = mappings[0]

    if not all(
        mapping == first_mapping
        for mapping in mappings
    ):
        raise RuntimeError(
            "M3 measurement mapping differs across circuits."
        )

    qubit_mapping = first_mapping

    print()
    print("M3 final measurement mapping:")
    print(qubit_mapping)

    # =========================================================================
    # PREFLIGHT
    # =========================================================================

    usage_before = safe_service_usage(
        service
    )

    scheduled_sum = float(
        audit_df[
            "scheduled_time_all_shots_s"
        ].sum()
    )

    print()
    print("=" * 120)
    print("FINAL QPU PREFLIGHT")
    print("=" * 120)
    print(f"Validation endpoints: {N_VAL}")
    print(f"Circuits:             {len(isa)}")
    print(f"Shots/setting:        {args.shots}")
    print(
        f"Median CZ/circuit:    "
        f"{int(audit_df['n_cz'].median())}"
    )
    print(
        f"Median depth:         "
        f"{float(audit_df['depth'].median()):.1f}"
    )
    print(
        f"Median duration:      "
        f"{float(audit_df['duration_us'].median()):.3f} us"
    )
    print(
        f"Scheduled-duration × shots sum: "
        f"{scheduled_sum:.6f} s"
    )
    print()
    print("Service usage before:")
    print(
        json.dumps(
            json_safe(usage_before),
            indent=2,
        )
    )

    preflight = {
        "timestamp_utc":
            datetime.now(timezone.utc).isoformat(),
        "mode":
            "SUBMIT" if args.submit else "DRY_RUN",
        "backend":
            args.backend,
        "shots":
            int(args.shots),
        "candidate":
            candidate,
        "features":
            FEATURES,
        "settings":
            ["XXXZYX", "ZZZZZZ"],
        "selected_lambda_training_only":
            float(selected_lambda),
        "training_cv_rmse":
            float(cv_summary.iloc[0]["cv_rmse_mean"]),
        "ideal_2025_metrics":
            ideal_metrics,
        "layout":
            layout,
        "m3_mapping":
            qubit_mapping,
        "scheduled_duration_all_shots_s":
            scheduled_sum,
        "service_usage_before":
            usage_before,
        "2026":
            "FROZEN / UNUSED",
    }

    with open(
        RESULTS / "11_0G_preflight.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(preflight),
            fp,
            indent=2,
        )

    if not args.submit:
        print()
        print("DRY RUN COMPLETE — NO QPU JOB SUBMITTED.")
        print()
        print("If acceptable, submit with:")
        print(
            f"python {Path(__file__).name} "
            f"--submit --backend {args.backend} "
            f"--shots {args.shots}"
        )
        return

    # =========================================================================
    # SUBMIT ONCE
    # =========================================================================

    sampler = Sampler(
        mode=backend
    )

    print()
    print("=" * 120)
    print("SUBMISSION")
    print("=" * 120)

    print(
        f"Submitting {len(isa)} SamplerV2 circuits "
        f"at {args.shots} shots/setting..."
    )

    job = sampler.run(
        isa,
        shots=int(args.shots),
    )

    print(
        f"Forecast job ID: {job.job_id()}"
    )

    # Calibrate M3 for all six measured physical qubits.
    print()
    print(
        "Learning M3 readout calibrations for all six measured qubits..."
    )

    mit = mthree.M3Mitigation(
        backend
    )

    mit.cals_from_system(
        qubit_mapping,
        rep_delay=None,
    )

    print("M3 calibration complete.")

    result = job.result()

    print(
        f"Forecast job final status: {job.status()}"
    )

    # =========================================================================
    # RAW + M3 DISTRIBUTIONS
    # =========================================================================

    endpoint_raw = {
        int(endpoint): {}
        for endpoint in val_idx
    }

    endpoint_m3 = {
        int(endpoint): {}
        for endpoint in val_idx
    }

    counts_rows = []
    quasi_rows = []

    for meta, pub_result in zip(
        metadata,
        result,
    ):
        counts = (
            pub_result
            .data
            .m
            .get_counts()
        )

        quasis = mit.apply_correction(
            counts,
            qubit_mapping,
        )

        endpoint = int(
            meta["endpoint_global"]
        )

        setting = str(
            meta["setting"]
        )

        endpoint_raw[
            endpoint
        ][
            setting
        ] = counts

        endpoint_m3[
            endpoint
        ][
            setting
        ] = quasis

        for bitstring, count in counts.items():
            counts_rows.append({
                "endpoint_global":
                    endpoint,
                "validation_offset":
                    int(meta["validation_offset"]),
                "setting":
                    setting,
                "bitstring":
                    bitstring,
                "count":
                    int(count),
            })

        for bitstring, quasi in quasis.items():
            quasi_rows.append({
                "endpoint_global":
                    endpoint,
                "validation_offset":
                    int(meta["validation_offset"]),
                "setting":
                    setting,
                "bitstring":
                    str(bitstring),
                "quasi_probability":
                    float(quasi),
            })

    pd.DataFrame(
        counts_rows
    ).to_csv(
        RESULTS / "11_0G_counts_raw.csv",
        index=False,
    )

    pd.DataFrame(
        quasi_rows
    ).to_csv(
        RESULTS / "11_0G_quasiprobabilities_M3.csv",
        index=False,
    )

    # =========================================================================
    # FEATURES
    # =========================================================================

    X_raw = np.vstack(
        [
            grouped_features(
                endpoint_raw[int(endpoint)]["XXXZYX"],
                endpoint_raw[int(endpoint)]["ZZZZZZ"],
            )
            for endpoint in val_idx
        ]
    )

    X_m3 = np.vstack(
        [
            grouped_features(
                endpoint_m3[int(endpoint)]["XXXZYX"],
                endpoint_m3[int(endpoint)]["ZZZZZZ"],
            )
            for endpoint in val_idx
        ]
    )

    pred_raw = base.predict_scaled_ridge(
        ridge,
        scaler,
        keep,
        X_raw,
    )

    pred_m3 = base.predict_scaled_ridge(
        ridge,
        scaler,
        keep,
        X_m3,
    )

    raw_metrics = metrics(
        y_val,
        pred_raw,
    )

    m3_metrics = metrics(
        y_val,
        pred_m3,
    )

    # =========================================================================
    # FEATURE AUDIT
    # =========================================================================

    feature_rows = []

    for j, feature in enumerate(FEATURES):
        ideal = X_val[:, j]
        raw = X_raw[:, j]
        m3v = X_m3[:, j]

        feature_rows.append({
            "feature":
                feature,
            "ideal_mean":
                float(np.mean(ideal)),
            "raw_mean":
                float(np.mean(raw)),
            "M3_mean":
                float(np.mean(m3v)),
            "raw_mean_error":
                float(np.mean(raw - ideal)),
            "M3_mean_error":
                float(np.mean(m3v - ideal)),
            "raw_MAE":
                base.mae(ideal, raw),
            "M3_MAE":
                base.mae(ideal, m3v),
            "raw_RMSE":
                base.rmse(ideal, raw),
            "M3_RMSE":
                base.rmse(ideal, m3v),
        })

    feature_summary = pd.DataFrame(
        feature_rows
    )

    feature_summary.to_csv(
        RESULTS / "11_0G_feature_summary.csv",
        index=False,
    )

    # =========================================================================
    # PREDICTIONS
    # =========================================================================

    prediction_rows = []

    for local_i, endpoint in enumerate(val_idx):
        prediction_rows.append({
            "validation_offset":
                int(local_i),
            "endpoint_global":
                int(endpoint),
            "target":
                float(y_val[local_i]),
            "pred_ideal":
                float(pred_ideal[local_i]),
            "pred_RAW":
                float(pred_raw[local_i]),
            "pred_M3":
                float(pred_m3[local_i]),
            "abs_error_ideal":
                float(abs(pred_ideal[local_i] - y_val[local_i])),
            "abs_error_RAW":
                float(abs(pred_raw[local_i] - y_val[local_i])),
            "abs_error_M3":
                float(abs(pred_m3[local_i] - y_val[local_i])),
        })

    pd.DataFrame(
        prediction_rows
    ).to_csv(
        RESULTS / "11_0G_predictions.csv",
        index=False,
    )

    # =========================================================================
    # USAGE
    # =========================================================================

    try:
        job_usage = float(
            job.usage(partial=False)
        )
    except Exception:
        job_usage = np.nan

    usage_after = safe_service_usage(
        service
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    summary = {
        **preflight,
        "forecast_job_id":
            job.job_id(),
        "forecast_job_usage_seconds":
            job_usage,
        "service_usage_after":
            usage_after,
        "raw_2025_metrics":
            raw_metrics,
        "M3_2025_metrics":
            m3_metrics,
        "raw_feature_MAE":
            base.mae(
                X_val.ravel(),
                X_raw.ravel(),
            ),
        "raw_feature_RMSE":
            base.rmse(
                X_val.ravel(),
                X_raw.ravel(),
            ),
        "M3_feature_MAE":
            base.mae(
                X_val.ravel(),
                X_m3.ravel(),
            ),
        "M3_feature_RMSE":
            base.rmse(
                X_val.ravel(),
                X_m3.ravel(),
            ),
    }

    with open(
        RESULTS / "11_0G_summary.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(summary),
            fp,
            indent=2,
        )

    print()
    print("=" * 120)
    print("11.0G — FINAL QPU RESULTS")
    print("=" * 120)
    print()
    print(
        f"Ideal RMSE: {ideal_metrics['rmse']:.6f}"
    )
    print()
    print(
        f"RAW RMSE:   {raw_metrics['rmse']:.6f}"
    )
    print(
        f"RAW MAE:    {raw_metrics['mae']:.6f}"
    )
    print(
        f"RAW bias:   {raw_metrics['bias']:+.6f}"
    )
    print()
    print(
        f"M3 RMSE:    {m3_metrics['rmse']:.6f}"
    )
    print(
        f"M3 MAE:     {m3_metrics['mae']:.6f}"
    )
    print(
        f"M3 bias:    {m3_metrics['bias']:+.6f}"
    )
    print()
    print(
        f"RAW feature MAE/RMSE: "
        f"{summary['raw_feature_MAE']:.6f} / "
        f"{summary['raw_feature_RMSE']:.6f}"
    )
    print(
        f"M3 feature MAE/RMSE:  "
        f"{summary['M3_feature_MAE']:.6f} / "
        f"{summary['M3_feature_RMSE']:.6f}"
    )
    print()
    print(
        f"Forecast job reported QPU charge: "
        f"{job_usage} s"
    )
    print()
    print("Per-feature audit:")
    print(
        feature_summary.to_string(
            index=False
        )
    )
    print()
    print("Saved:")
    for name in [
        "11_0G_hardware_ranking_all6.csv",
        "11_0G_transpilation.csv",
        "11_0G_preflight.json",
        "11_0G_counts_raw.csv",
        "11_0G_quasiprobabilities_M3.csv",
        "11_0G_feature_summary.csv",
        "11_0G_predictions.csv",
        "11_0G_summary.json",
    ]:
        print(
            f"  results/{name}"
        )

    print()
    print(
        "Scientific caveat: this final hardware test is exploratory because "
        "DROP_X3 was motivated after inspecting 2025 QPU behavior. "
        "The still-frozen 2026 test remains untouched."
    )


if __name__ == "__main__":
    main()
