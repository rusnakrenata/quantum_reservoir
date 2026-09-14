
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler

from ibm_account import get_service

try:
    import mthree
except ImportError as exc:
    raise ImportError(
        "This experiment uses IBM's documented M3 readout-mitigation workflow "
        "for SamplerV2. Install it first with:\n\n"
        "    pip install \"mthree>=3.0\"\n"
    ) from exc


# =============================================================================
# WEEK 11.0E — RWP_W1, 1024 SHOTS + M3 READOUT MITIGATION
#
# Controlled follow-up to 11.0D.
#
# IMPORTANT:
#   2022-2024 = establish lambda, scaler, Ridge weights, intercept
#   2025      = validation ONLY
#   2026      = frozen / unused
#
# The same 1024-shot Sampler counts are evaluated twice:
#
#   1) RAW
#      counts -> observables -> frozen Ridge -> RMSE_2025
#
#   2) M3-MITIGATED
#      counts -> M3 quasi-probabilities -> observables
#             -> SAME frozen Ridge -> RMSE_2025
#
# Therefore there is NO second 730-circuit QPU forecast job for mitigation.
# M3 requires an additional small readout-calibration workload, but the
# forecast circuits themselves are executed only once.
#
# Default:
#   1024 shots / setting
#
# Dry run:
#   python 11_0E_RWP_W1_1024_M3.py
#
# Submit:
#   python 11_0E_RWP_W1_1024_M3.py --submit
#
# Optional:
#   python 11_0E_RWP_W1_1024_M3.py --submit --shots 2048
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

PRIOR_W1_USAGE = RESULTS / "11_0D_usage.json"
PRIOR_W1_SUMMARY = RESULTS / "11_0D_summary.json"


# =============================================================================
# IMPORT THE ALREADY-VALIDATED 11.0D IMPLEMENTATION
# =============================================================================

def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put this script beside the validated 11.0D script."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    spec.loader.exec_module(module)
    return module


base = import_module_from_path(
    "week11_0D_base",
    BASE_SCRIPT,
)


# =============================================================================
# HELPERS
# =============================================================================

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


def safe_service_usage(service):
    try:
        return service.usage()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }


def parse_distribution_key(key, n_bits=4):
    """
    M3 quasi distributions normally use bitstring keys. Handle integer keys too.
    """
    if isinstance(key, (int, np.integer)):
        return format(int(key), f"0{n_bits}b")

    text = str(key).replace(" ", "")

    if text.startswith("0x"):
        return format(int(text, 16), f"0{n_bits}b")

    if text.startswith("0b"):
        text = text[2:]

    return text.zfill(n_bits)


def expectation_from_distribution(distribution, q, n_bits=4):
    """
    Expectation from either probabilities or quasi-probabilities.

    M3 quasi-probabilities may be negative. That is expected and must NOT
    be clipped, otherwise the mitigation would be biased.
    """
    total = 0.0
    acc = 0.0

    for key, weight in distribution.items():
        bitstring = parse_distribution_key(
            key,
            n_bits=n_bits,
        )

        bit = int(
            bitstring[-(q + 1)]
        )

        eig = (
            1.0
            if bit == 0
            else -1.0
        )

        w = float(weight)

        total += w
        acc += eig * w

    if np.isclose(total, 0.0):
        raise RuntimeError(
            "Mitigated quasi-distribution has near-zero normalization."
        )

    return acc / total


def xz_features_from_distributions(dist_x, dist_z):
    vals = []

    for q in range(4):
        vals.append(
            expectation_from_distribution(
                dist_x,
                q,
            )
        )

    for q in range(4):
        vals.append(
            expectation_from_distribution(
                dist_z,
                q,
            )
        )

    return np.asarray(
        vals,
        dtype=float,
    )


def previous_w1_charge():
    if not PRIOR_W1_USAGE.exists():
        return np.nan

    try:
        data = json.loads(
            PRIOR_W1_USAGE.read_text(
                encoding="utf-8"
            )
        )

        return float(
            data["job_usage_seconds"]
        )

    except Exception:
        return np.nan


def service_remaining_seconds(usage):
    if not isinstance(usage, dict):
        return np.nan

    try:
        return float(
            usage["usage_remaining_seconds"]
        )
    except Exception:
        return np.nan


def feature_summary(
    X_ideal,
    X_raw,
    X_m3,
):
    rows = []

    for j, feature in enumerate(
        base.XZ_COLUMNS
    ):
        ideal = X_ideal[:, j]
        raw = X_raw[:, j]
        m3 = X_m3[:, j]

        rows.append({
            "feature": feature,
            "ideal_mean": float(
                np.mean(ideal)
            ),
            "raw_mean": float(
                np.mean(raw)
            ),
            "m3_mean": float(
                np.mean(m3)
            ),
            "raw_mean_error": float(
                np.mean(raw - ideal)
            ),
            "m3_mean_error": float(
                np.mean(m3 - ideal)
            ),
            "raw_MAE": base.mae(
                ideal,
                raw,
            ),
            "m3_MAE": base.mae(
                ideal,
                m3,
            ),
            "raw_RMSE": base.rmse(
                ideal,
                raw,
            ),
            "m3_RMSE": base.rmse(
                ideal,
                m3,
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the 365-day validation workload.",
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

    parser.add_argument(
        "--force",
        action="store_true",
        help="Override conservative allocation guard.",
    )

    args = parser.parse_args()

    if args.shots < 1:
        raise ValueError(
            "--shots must be >= 1"
        )

    # =========================================================================
    # SAME LOGICAL MODEL / SAME 2022-2024 READOUT ESTABLISHMENT AS 11.0D
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

    X_ideal = base.build_ideal_w1_feature_bank(
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

    (
        ridge,
        scaler,
        keep,
    ) = base.fit_scaled_ridge(
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

    ideal_rmse = base.rmse(
        y_val,
        pred_ideal,
    )

    ideal_mae = base.mae(
        y_val,
        pred_ideal,
    )

    ideal_bias = base.bias(
        y_val,
        pred_ideal,
    )

    print("=" * 120)
    print("WEEK 11.0E — RWP_W1: 1024 SHOTS + M3 READOUT MITIGATION")
    print("=" * 120)
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
    print(
        f"Selected lambda: {selected_lambda}"
    )
    print(
        f"Training-CV RMSE: "
        f"{float(cv_summary.iloc[0]['cv_rmse_mean']):.6f}"
    )
    print(
        f"Ideal 2025 RMSE: {ideal_rmse:.6f}"
    )
    print(
        f"Shots/setting for this run: {args.shots}"
    )

    # =========================================================================
    # FRESH HARDWARE SCAN
    # =========================================================================

    service = get_service()

    print()
    print("=" * 120)
    print(
        f"FRESH {args.backend} HARDWARE SCAN"
    )
    print("=" * 120)

    fresh = base.selector.fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=[
            args.backend
        ],
        shortlist=args.shortlist,
        write_prefix="11_0E_hardware",
        verbose=True,
    )

    (
        selected_layout_row,
        hardware_ranking,
    ) = base.choose_fresh_layout_for_xz_injection(
        fresh,
        args.backend,
    )

    hardware_ranking.to_csv(
        RESULTS
        /
        "11_0E_hardware_ranking_XZinj.csv",
        index=False,
    )

    layout = json.loads(
        selected_layout_row[
            "layout"
        ]
    )

    print()
    print("Fresh physical choice:")
    print(
        f"  layout={layout}"
    )
    print(
        f"  max CZ error="
        f"{selected_layout_row['compiled_2q_error_max_percent']:.4f}%"
    )
    print(
        f"  max injection readout error="
        f"{selected_layout_row['injection_readout_error_max_percent']:.4f}%"
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
    # BUILD THE SAME 365 × 2 W1 CIRCUITS
    # =========================================================================

    logical = []
    metadata = []

    for endpoint in val_idx:
        for basis in [
            "X",
            "Z",
        ]:
            qc = base.build_w1_measurement_circuit(
                candidate,
                angles[int(endpoint)],
                basis,
                int(endpoint),
            )

            logical.append(qc)

            metadata.append({
                "endpoint_global":
                    int(endpoint),
                "validation_offset":
                    int(
                        endpoint
                        -
                        N_TRAIN
                    ),
                "basis":
                    basis,
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

    if not isinstance(
        isa,
        list,
    ):
        isa = [isa]

    audit_rows = []

    for meta, circuit in zip(
        metadata,
        isa,
    ):
        backend.check_faulty(
            circuit
        )

        row = base.circuit_resource_row(
            circuit,
            backend,
            meta,
        )

        if row["n_swap"] != 0:
            raise RuntimeError(
                "SWAP detected; refusing the controlled comparison."
            )

        audit_rows.append(row)

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df[
        "scheduled_time_all_shots_s"
    ] = (
        audit_df["duration_us"]
        *
        1e-6
        *
        args.shots
    )

    audit_df.to_csv(
        RESULTS
        /
        "11_0E_transpilation.csv",
        index=False,
    )

    # =========================================================================
    # M3 MAPPING AUDIT
    # =========================================================================

    mappings = [
        mthree.utils.final_measurement_mapping(
            circuit
        )
        for circuit in isa
    ]

    first_mapping = mappings[0]

    if not all(
        mapping == first_mapping
        for mapping in mappings
    ):
        raise RuntimeError(
            "Final measurement mapping is not identical across all circuits. "
            "This script deliberately requires one fixed physical mapping so "
            "one M3 calibration can be reused for all 730 circuits."
        )

    qubit_mapping = first_mapping

    print()
    print("M3 final measurement mapping:")
    print(
        qubit_mapping
    )

    # =========================================================================
    # COST PREFLIGHT
    # =========================================================================

    usage_before = safe_service_usage(
        service
    )

    remaining = service_remaining_seconds(
        usage_before
    )

    previous_charge = previous_w1_charge()

    rough_linear_charge = np.nan

    if (
        np.isfinite(previous_charge)
        and
        previous_charge > 0
    ):
        rough_linear_charge = (
            previous_charge
            *
            args.shots
            /
            256.0
        )

    scheduled_sum = float(
        audit_df[
            "scheduled_time_all_shots_s"
        ].sum()
    )

    print()
    print("=" * 120)
    print("QPU PREFLIGHT")
    print("=" * 120)
    print(
        f"Validation endpoints:     {N_VAL}"
    )
    print(
        f"Forecast circuits:        {len(isa)}"
    )
    print(
        f"Shots/setting:            {args.shots}"
    )
    print(
        f"Median CZ/circuit:        "
        f"{int(audit_df['n_cz'].median())}"
    )
    print(
        f"Median depth:             "
        f"{float(audit_df['depth'].median()):.1f}"
    )
    print(
        f"Scheduled duration×shots: "
        f"{scheduled_sum:.6f} s"
    )

    if np.isfinite(
        rough_linear_charge
    ):
        print(
            f"Very rough linear estimate from the "
            f"previous 256-shot W1 charge: "
            f"{rough_linear_charge:.1f} s"
        )

    print(
        "M3 adds a separate readout-calibration workload; "
        "its actual charge is not assumed in the estimate."
    )

    print()
    print("Service usage before:")
    print(
        json.dumps(
            json_safe(
                usage_before
            ),
            indent=2,
        )
    )

    preflight = {
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "mode":
            (
                "SUBMIT"
                if args.submit
                else "DRY_RUN"
            ),
        "protocol":
            "RWP_W1",
        "readout":
            "XZ_injection",
        "mitigation":
            "M3 readout-error mitigation",
        "shots":
            int(args.shots),
        "n_validation":
            N_VAL,
        "n_forecast_circuits":
            len(isa),
        "backend":
            args.backend,
        "layout":
            layout,
        "m3_qubit_mapping":
            json_safe(
                qubit_mapping
            ),
        "selected_lambda_training_only":
            float(selected_lambda),
        "ideal_2025_rmse":
            ideal_rmse,
        "scheduled_duration_all_shots_s":
            scheduled_sum,
        "previous_256shot_W1_charge_s":
            previous_charge,
        "rough_linear_forecast_charge_s":
            rough_linear_charge,
        "service_usage_before":
            json_safe(
                usage_before
            ),
        "2026":
            "FROZEN / UNUSED",
    }

    with open(
        RESULTS
        /
        "11_0E_preflight.json",
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
        print(
            "DRY RUN COMPLETE — NO QPU JOB OR M3 CALIBRATION SUBMITTED."
        )
        print()
        print(
            "Submit with:"
        )
        print(
            f"python {Path(__file__).name} "
            f"--submit "
            f"--backend {args.backend} "
            f"--shots {args.shots}"
        )
        return

    # Conservative guard based only on the known 256-shot W1 workload.
    # M3 calibration is additional, so leave headroom.
    if (
        np.isfinite(remaining)
        and
        np.isfinite(rough_linear_charge)
        and
        rough_linear_charge
        >
        0.70
        *
        remaining
        and
        not args.force
    ):
        raise RuntimeError(
            f"Rough forecast-job estimate {rough_linear_charge:.1f}s "
            f"already exceeds 70% of remaining allocation {remaining:.1f}s, "
            f"before M3 calibration. Review first or explicitly use --force."
        )

    # =========================================================================
    # SUBMIT FORECAST JOB, THEN CALIBRATE M3 CLOSE IN TIME
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
        shots=int(
            args.shots
        ),
    )

    print(
        f"Forecast job ID: {job.job_id()}"
    )

    print()
    print(
        "Submitting/learning M3 readout calibrations "
        "for the fixed measured physical qubits..."
    )

    mit = mthree.M3Mitigation(
        backend
    )

    # IBM's documented Sampler + M3 workflow.
    mit.cals_from_system(
        qubit_mapping,
        rep_delay=None,
    )

    usage_after_m3_calibration = safe_service_usage(
        service
    )

    print(
        "M3 calibration complete."
    )
    print(
        "Service usage after M3 calibration:"
    )
    print(
        json.dumps(
            json_safe(
                usage_after_m3_calibration
            ),
            indent=2,
        )
    )

    result = job.result()

    print(
        f"Forecast job final status: {job.status()}"
    )

    # =========================================================================
    # RAW COUNTS + M3 QUASI-DISTRIBUTIONS
    # =========================================================================

    raw_endpoint = {
        int(endpoint): {}
        for endpoint in val_idx
    }

    m3_endpoint = {
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
            .inj
            .get_counts()
        )

        quasis = mit.apply_correction(
            counts,
            qubit_mapping,
        )

        endpoint = int(
            meta[
                "endpoint_global"
            ]
        )

        basis = str(
            meta[
                "basis"
            ]
        )

        raw_endpoint[
            endpoint
        ][
            basis
        ] = counts

        m3_endpoint[
            endpoint
        ][
            basis
        ] = quasis

        for bitstring, count in counts.items():
            counts_rows.append({
                "endpoint_global":
                    endpoint,
                "validation_offset":
                    int(
                        meta[
                            "validation_offset"
                        ]
                    ),
                "basis":
                    basis,
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
                    int(
                        meta[
                            "validation_offset"
                        ]
                    ),
                "basis":
                    basis,
                "bitstring":
                    str(bitstring),
                "quasi_probability":
                    float(quasi),
            })

    pd.DataFrame(
        counts_rows
    ).to_csv(
        RESULTS
        /
        "11_0E_counts_raw.csv",
        index=False,
    )

    pd.DataFrame(
        quasi_rows
    ).to_csv(
        RESULTS
        /
        "11_0E_quasiprobabilities_M3.csv",
        index=False,
    )

    # =========================================================================
    # RAW AND MITIGATED FEATURES
    # =========================================================================

    X_raw = np.vstack(
        [
            base.xz_features_from_counts(
                raw_endpoint[
                    int(endpoint)
                ]["X"],
                raw_endpoint[
                    int(endpoint)
                ]["Z"],
            )
            for endpoint in val_idx
        ]
    )

    X_m3 = np.vstack(
        [
            xz_features_from_distributions(
                m3_endpoint[
                    int(endpoint)
                ]["X"],
                m3_endpoint[
                    int(endpoint)
                ]["Z"],
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

    # =========================================================================
    # METRICS
    # =========================================================================

    raw_rmse = base.rmse(
        y_val,
        pred_raw,
    )

    raw_mae = base.mae(
        y_val,
        pred_raw,
    )

    raw_bias = base.bias(
        y_val,
        pred_raw,
    )

    m3_rmse = base.rmse(
        y_val,
        pred_m3,
    )

    m3_mae = base.mae(
        y_val,
        pred_m3,
    )

    m3_bias = base.bias(
        y_val,
        pred_m3,
    )

    raw_feature_mae = base.mae(
        X_val.ravel(),
        X_raw.ravel(),
    )

    raw_feature_rmse = base.rmse(
        X_val.ravel(),
        X_raw.ravel(),
    )

    m3_feature_mae = base.mae(
        X_val.ravel(),
        X_m3.ravel(),
    )

    m3_feature_rmse = base.rmse(
        X_val.ravel(),
        X_m3.ravel(),
    )

    per_feature = feature_summary(
        X_ideal=X_val,
        X_raw=X_raw,
        X_m3=X_m3,
    )

    per_feature.to_csv(
        RESULTS
        /
        "11_0E_feature_summary_raw_vs_M3.csv",
        index=False,
    )

    # =========================================================================
    # SAVE PER-DAY FEATURE VECTORS
    # =========================================================================

    feature_rows = []

    for local_i, endpoint in enumerate(
        val_idx
    ):
        for j, feature in enumerate(
            base.XZ_COLUMNS
        ):
            feature_rows.append({
                "validation_offset":
                    int(local_i),
                "endpoint_global":
                    int(endpoint),
                "feature":
                    feature,
                "ideal":
                    float(
                        X_val[
                            local_i,
                            j,
                        ]
                    ),
                "raw":
                    float(
                        X_raw[
                            local_i,
                            j,
                        ]
                    ),
                "M3":
                    float(
                        X_m3[
                            local_i,
                            j,
                        ]
                    ),
                "raw_error":
                    float(
                        X_raw[
                            local_i,
                            j,
                        ]
                        -
                        X_val[
                            local_i,
                            j,
                        ]
                    ),
                "M3_error":
                    float(
                        X_m3[
                            local_i,
                            j,
                        ]
                        -
                        X_val[
                            local_i,
                            j,
                        ]
                    ),
            })

    pd.DataFrame(
        feature_rows
    ).to_csv(
        RESULTS
        /
        "11_0E_features_raw_vs_M3.csv",
        index=False,
    )

    # =========================================================================
    # SAVE PER-DAY PREDICTIONS
    # =========================================================================

    prediction_rows = []

    for local_i, endpoint in enumerate(
        val_idx
    ):
        row = {
            "validation_offset":
                int(local_i),
            "endpoint_global":
                int(endpoint),
            "target":
                float(
                    y_val[
                        local_i
                    ]
                ),
            "pred_ideal":
                float(
                    pred_ideal[
                        local_i
                    ]
                ),
            "pred_raw":
                float(
                    pred_raw[
                        local_i
                    ]
                ),
            "pred_M3":
                float(
                    pred_m3[
                        local_i
                    ]
                ),
            "abs_error_ideal":
                float(
                    abs(
                        pred_ideal[
                            local_i
                        ]
                        -
                        y_val[
                            local_i
                        ]
                    )
                ),
            "abs_error_raw":
                float(
                    abs(
                        pred_raw[
                            local_i
                        ]
                        -
                        y_val[
                            local_i
                        ]
                    )
                ),
            "abs_error_M3":
                float(
                    abs(
                        pred_m3[
                            local_i
                        ]
                        -
                        y_val[
                            local_i
                        ]
                    )
                ),
        }

        for date_col in [
            "target_date",
            "date",
            "Date",
        ]:
            if date_col in work_tv.columns:
                row[
                    "target_date"
                ] = str(
                    pd.to_datetime(
                        work_tv.iloc[
                            int(endpoint)
                        ][
                            date_col
                        ]
                    ).date()
                )
                break

        prediction_rows.append(
            row
        )

    predictions = pd.DataFrame(
        prediction_rows
    )

    predictions.to_csv(
        RESULTS
        /
        "11_0E_predictions_raw_vs_M3.csv",
        index=False,
    )

    # =========================================================================
    # QPU USAGE
    # =========================================================================

    try:
        job_usage = float(
            job.usage(
                partial=False
            )
        )
    except Exception as exc:
        job_usage = np.nan
        job_usage_error = str(exc)
    else:
        job_usage_error = ""

    try:
        metrics = job.metrics()
    except Exception as exc:
        metrics = {
            "available": False,
            "error": str(exc),
        }

    usage_after_all = safe_service_usage(
        service
    )

    # The account-level delta includes whatever the platform charged for
    # both the M3 calibration work and the forecast job during this script.
    account_delta = np.nan

    before_consumed = np.nan
    after_consumed = np.nan

    try:
        before_consumed = float(
            usage_before[
                "usage_consumed_seconds"
            ]
        )

        after_consumed = float(
            usage_after_all[
                "usage_consumed_seconds"
            ]
        )

        account_delta = (
            after_consumed
            -
            before_consumed
        )
    except Exception:
        pass

    usage_record = {
        "forecast_job_id":
            job.job_id(),
        "forecast_job_usage_seconds":
            job_usage,
        "forecast_job_usage_error":
            job_usage_error,
        "forecast_job_metrics":
            json_safe(metrics),
        "service_usage_before":
            json_safe(
                usage_before
            ),
        "service_usage_after_m3_calibration":
            json_safe(
                usage_after_m3_calibration
            ),
        "service_usage_after_all":
            json_safe(
                usage_after_all
            ),
        "account_consumed_delta_during_script_s":
            account_delta,
    }

    with open(
        RESULTS
        /
        "11_0E_usage.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(
                usage_record
            ),
            fp,
            indent=2,
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    x3 = per_feature[
        per_feature[
            "feature"
        ]
        ==
        "X3"
    ].iloc[0]

    summary = {
        **preflight,
        "forecast_job_id":
            job.job_id(),
        "forecast_job_usage_seconds":
            job_usage,
        "account_consumed_delta_during_script_s":
            account_delta,
        "ideal_2025_RMSE":
            ideal_rmse,
        "ideal_2025_MAE":
            ideal_mae,
        "ideal_2025_bias":
            ideal_bias,
        "raw_2025_RMSE":
            raw_rmse,
        "raw_2025_MAE":
            raw_mae,
        "raw_2025_bias":
            raw_bias,
        "M3_2025_RMSE":
            m3_rmse,
        "M3_2025_MAE":
            m3_mae,
        "M3_2025_bias":
            m3_bias,
        "raw_feature_MAE":
            raw_feature_mae,
        "raw_feature_RMSE":
            raw_feature_rmse,
        "M3_feature_MAE":
            m3_feature_mae,
        "M3_feature_RMSE":
            m3_feature_rmse,
        "X3_ideal_mean":
            float(
                x3[
                    "ideal_mean"
                ]
            ),
        "X3_raw_mean":
            float(
                x3[
                    "raw_mean"
                ]
            ),
        "X3_M3_mean":
            float(
                x3[
                    "m3_mean"
                ]
            ),
        "X3_raw_mean_error":
            float(
                x3[
                    "raw_mean_error"
                ]
            ),
        "X3_M3_mean_error":
            float(
                x3[
                    "m3_mean_error"
                ]
            ),
        "service_usage_after_all":
            json_safe(
                usage_after_all
            ),
    }

    with open(
        RESULTS
        /
        "11_0E_summary.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(
                summary
            ),
            fp,
            indent=2,
        )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print()
    print("=" * 120)
    print("11.0E — RAW VS M3 RESULTS")
    print("=" * 120)
    print(
        f"Ideal 2025 RMSE:         {ideal_rmse:.6f}"
    )
    print()
    print(
        f"RAW {args.shots}-shot RMSE:       {raw_rmse:.6f}"
    )
    print(
        f"RAW {args.shots}-shot MAE:        {raw_mae:.6f}"
    )
    print(
        f"RAW {args.shots}-shot bias:       {raw_bias:+.6f}"
    )
    print(
        f"RAW feature MAE/RMSE:    "
        f"{raw_feature_mae:.6f} / {raw_feature_rmse:.6f}"
    )
    print()
    print(
        f"M3 {args.shots}-shot RMSE:        {m3_rmse:.6f}"
    )
    print(
        f"M3 {args.shots}-shot MAE:         {m3_mae:.6f}"
    )
    print(
        f"M3 {args.shots}-shot bias:        {m3_bias:+.6f}"
    )
    print(
        f"M3 feature MAE/RMSE:     "
        f"{m3_feature_mae:.6f} / {m3_feature_rmse:.6f}"
    )
    print()
    print("X3 systematic shift:")
    print(
        f"  ideal mean = {float(x3['ideal_mean']):+.6f}"
    )
    print(
        f"  raw mean   = {float(x3['raw_mean']):+.6f}, "
        f"error={float(x3['raw_mean_error']):+.6f}"
    )
    print(
        f"  M3 mean    = {float(x3['m3_mean']):+.6f}, "
        f"error={float(x3['m3_mean_error']):+.6f}"
    )
    print()
    print(
        f"Forecast job reported QPU charge: {job_usage} s"
    )

    if np.isfinite(
        account_delta
    ):
        print(
            f"Account consumed-seconds delta during script "
            f"(forecast + M3 calibration if reflected): "
            f"{account_delta:.1f} s"
        )

    print()
    print("Per-feature raw vs M3:")
    print(
        per_feature.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    for name in [
        "11_0E_hardware_ranking_XZinj.csv",
        "11_0E_transpilation.csv",
        "11_0E_preflight.json",
        "11_0E_counts_raw.csv",
        "11_0E_quasiprobabilities_M3.csv",
        "11_0E_features_raw_vs_M3.csv",
        "11_0E_feature_summary_raw_vs_M3.csv",
        "11_0E_predictions_raw_vs_M3.csv",
        "11_0E_usage.json",
        "11_0E_summary.json",
    ]:
        print(
            f"  results/{name}"
        )


if __name__ == "__main__":
    main()
