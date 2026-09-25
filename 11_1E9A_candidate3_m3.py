from __future__ import annotations

"""
WEEK 11.1E.9A
Candidate #3 controlled M3 measurement-mitigation experiment.

SCIENTIFIC QUESTION
-------------------
For the already-selected Candidate #3:

    RWP H2
    W = 2
    r = 2
    alpha = 0.75
    readout = XZinj_dropX3
    shots/setting = 1024
    frozen Ridge lambda = 10
    ideal 2025 RMSE ~= 4.922051

does M3 readout-error mitigation reduce the real-QPU feature distortion,
especially Z2/Z0, and improve the frozen 2025 forecast?

IMPORTANT DESIGN
----------------
* 2022-2024 establishes/fits the frozen Ridge readout.
* 2025 is the hardware validation set.
* 2026 is NEVER loaded.
* The quantum candidate itself is NOT changed.
* The same real-QPU workload counts are used twice:
      1) RAW features/predictions
      2) M3-corrected features/predictions
  Therefore RAW vs M3 is a paired comparison from the SAME workload shots.
* M3 adds only its calibration job(s).
* Fresh H2 hardware reselection is still performed immediately before execution.

DEPENDENCIES
------------
This script deliberately reuses the already-audited Candidate #3 implementation:

    11_1C_third_candidate_direct_qpu.py

It also needs:

    pip install "mthree>=3.0"

Default behavior is a FULL-2025 DRY RUN.
Use --submit only after inspecting preflight.

EXAMPLE
-------
Dry run:
    python 11_1E9A_candidate3_m3.py

Submit full 2025:
    python 11_1E9A_candidate3_m3.py --submit

Optional:
    python 11_1E9A_candidate3_m3.py --submit --backend ibm_kingston \
        --shots 1024 --m3-cal-shots 10000
"""

import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mthree
except ImportError as exc:
    raise ImportError(
        "M3 is not installed.\n"
        "Install it with:\n\n"
        "    pip install \"mthree>=3.0\"\n"
    ) from exc

from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2

from ibm_account import get_service


# =============================================================================
# PATHS / FROZEN SETTINGS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

BASE_SCRIPT = HERE / "11_1C_third_candidate_direct_qpu.py"

PREFIX = "11_1E9A_candidate3_m3"

DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_SHOTS = 1024

# Explicit, reproducible M3 calibration size.
# M3's documented hardware default is up to 10,000 shots; we state it rather
# than silently inheriting a package default.
DEFAULT_M3_CAL_SHOTS = 10_000
M3_CAL_METHOD = "balanced"

OPT_LEVEL = 1
SEED_TRANSPILE = 42
SHORTLIST = 120

EXPECTED_N_2025 = 365

# Candidate #3 feature order, exactly as in the original Week-11.1C run.
FEATURES = ["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3"]

# The exact grouped settings are imported from the original Candidate #3 script:
#   XXXZ -> X0 X1 X2 (+ Z3 cross-setting diagnostic)
#   ZZZZ -> Z0 Z1 Z2 Z3


# =============================================================================
# LOAD THE EXISTING, AUDITED CANDIDATE-3 IMPLEMENTATION
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required base Candidate #3 script not found:\n  {path}"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = load_module(BASE_SCRIPT, "week111c_candidate3_base")


# =============================================================================
# SMALL HELPERS
# =============================================================================

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


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pd.Timestamp):
        return str(x)
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def expectation_from_distribution(dist, classical_bits):
    """
    Expectation from either raw counts OR an M3 quasi-probability distribution.

    For raw counts the values are non-negative shot counts.
    For M3, values can be negative quasi-probabilities but still sum to ~1.

    Qiskit displays c0 as the right-most bit, matching the original Candidate #3
    parser.
    """
    total = float(sum(float(v) for v in dist.values()))
    if abs(total) < 1e-15:
        raise RuntimeError("Distribution has zero total weight.")

    acc = 0.0
    for bitstring, weight in dist.items():
        s = str(bitstring).replace(" ", "")
        eig = 1.0
        for c in classical_bits:
            bit = int(s[-1 - int(c)])
            eig *= -1.0 if bit else 1.0
        acc += float(weight) * eig

    return float(acc / total)


def normalize_mapping(mapping):
    """
    Convert M3 final_measurement_mapping output to a stable Python list.

    In the standard case it is already a list where element c is the physical
    qubit measured into classical bit c.
    """
    if isinstance(mapping, np.ndarray):
        mapping = mapping.tolist()

    if isinstance(mapping, tuple):
        mapping = list(mapping)

    if isinstance(mapping, list):
        return [int(x) for x in mapping]

    # Compatibility if a dict-like mapping is returned.
    if isinstance(mapping, dict):
        # If keys look like classical-bit indices, sort by key.
        try:
            return [int(mapping[k]) for k in sorted(mapping, key=int)]
        except Exception:
            return [int(v) for _, v in sorted(mapping.items(), key=lambda kv: str(kv[0]))]

    raise TypeError(
        f"Unsupported final_measurement_mapping type: {type(mapping)} -> {mapping!r}"
    )


def extract_job_record(job):
    out = {
        "job_id": None,
        "status": None,
        "metrics": None,
        "qpu_charge_time_seconds": None,
        "qpu_circuits_execution_time_seconds": None,
        "qpu_running_wall_seconds": None,
    }

    try:
        out["job_id"] = str(job.job_id())
    except Exception:
        pass

    try:
        out["status"] = str(job.status())
    except Exception:
        pass

    metrics = None
    try:
        metrics = base.json_safe(job.metrics())
    except Exception:
        pass

    out["metrics"] = metrics

    try:
        timing = base.extract_ibm_qpu_timing(job, metrics)
        out.update(timing)
    except Exception:
        pass

    return json_safe(out)


def reconstruct_effective_raw_beta(X_ideal, pred_ideal):
    """
    Reconstruct the exact effective affine readout in raw feature coordinates:

        pred = intercept + X @ beta

    using ONLY saved/recomputed ideal features and ideal predictions.
    Targets are not used.
    """
    X = np.asarray(X_ideal, dtype=float)
    p = np.asarray(pred_ideal, dtype=float)

    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, p, rcond=None)

    intercept = float(coef[0])
    beta = np.asarray(coef[1:], dtype=float)

    recon = intercept + X @ beta
    audit = rmse(p, recon)

    if audit > 1e-8:
        raise RuntimeError(
            f"Frozen readout reconstruction audit failed: RMSE={audit:.3e}"
        )

    return intercept, beta, audit


# =============================================================================
# M3 FEATURE PARSING
# =============================================================================

def extract_features_from_distributions(per_endpoint):
    """
    Convert endpoint -> setting -> distribution into the seven Candidate #3
    features plus Z3 cross-setting diagnostic.
    """
    by_endpoint = {}

    for endpoint, setting_map in per_endpoint.items():
        vals = {}

        if "XXXZ" not in setting_map or "ZZZZ" not in setting_map:
            raise RuntimeError(
                f"Endpoint {endpoint}: missing XXXZ or ZZZZ setting."
            )

        dist_x = setting_map["XXXZ"]
        dist_z = setting_map["ZZZZ"]

        vals["X0"] = expectation_from_distribution(dist_x, [0])
        vals["X1"] = expectation_from_distribution(dist_x, [1])
        vals["X2"] = expectation_from_distribution(dist_x, [2])
        vals["Z3_from_X_setting"] = expectation_from_distribution(dist_x, [3])

        vals["Z0"] = expectation_from_distribution(dist_z, [0])
        vals["Z1"] = expectation_from_distribution(dist_z, [1])
        vals["Z2"] = expectation_from_distribution(dist_z, [2])
        vals["Z3"] = expectation_from_distribution(dist_z, [3])

        by_endpoint[int(endpoint)] = vals

    return by_endpoint


# =============================================================================
# FEATURE DIAGNOSTICS
# =============================================================================

def make_feature_diagnostics(
    X_ideal,
    X_raw,
    X_m3,
    beta_raw,
):
    rows = []

    for j, feature in enumerate(FEATURES):
        ideal = X_ideal[:, j]
        raw = X_raw[:, j]
        m3 = X_m3[:, j]

        sigma_ideal = float(np.std(ideal, ddof=1))
        if sigma_ideal <= 0:
            raise RuntimeError(f"{feature}: ideal feature std is zero.")

        for method, qpu in [("RAW", raw), ("M3", m3)]:
            delta = qpu - ideal
            feature_bias = float(np.mean(delta))
            feature_rmse = float(np.sqrt(np.mean(delta ** 2)))
            stochastic_rms = float(
                np.sqrt(max(feature_rmse ** 2 - feature_bias ** 2, 0.0))
            )

            noise_to_signal = feature_rmse / sigma_ideal

            rows.append({
                "feature": feature,
                "method": method,
                "beta": float(beta_raw[j]),
                "ideal_std": sigma_ideal,
                "feature_bias": feature_bias,
                "feature_mae": float(np.mean(np.abs(delta))),
                "feature_rmse": feature_rmse,
                "feature_stochastic_rms": stochastic_rms,
                "feature_corr_qpu_vs_ideal": corr(qpu, ideal),
                "qpu_std": float(np.std(qpu, ddof=1)),
                "qpu_to_ideal_std_ratio": float(
                    np.std(qpu, ddof=1) / sigma_ideal
                ),
                "noise_to_signal_ratio": noise_to_signal,
                "direct_prediction_perturbation_rms_claims": (
                    abs(float(beta_raw[j])) * feature_rmse
                ),
                "systematic_prediction_shift_claims": (
                    float(beta_raw[j]) * feature_bias
                ),
                "random_prediction_perturbation_rms_claims": (
                    abs(float(beta_raw[j])) * stochastic_rms
                ),
            })

    diag = pd.DataFrame(rows)

    raw = diag[diag["method"] == "RAW"].set_index("feature")
    m3 = diag[diag["method"] == "M3"].set_index("feature")

    improvement_rows = []
    for feature in FEATURES:
        r = raw.loc[feature]
        m = m3.loc[feature]

        improvement_rows.append({
            "feature": feature,

            "raw_feature_bias": r["feature_bias"],
            "m3_feature_bias": m["feature_bias"],
            "abs_bias_reduction": (
                abs(r["feature_bias"]) - abs(m["feature_bias"])
            ),

            "raw_feature_rmse": r["feature_rmse"],
            "m3_feature_rmse": m["feature_rmse"],
            "feature_rmse_reduction": (
                r["feature_rmse"] - m["feature_rmse"]
            ),
            "feature_rmse_reduction_pct": (
                100.0 * (r["feature_rmse"] - m["feature_rmse"])
                / r["feature_rmse"]
                if r["feature_rmse"] != 0 else np.nan
            ),

            "raw_corr": r["feature_corr_qpu_vs_ideal"],
            "m3_corr": m["feature_corr_qpu_vs_ideal"],
            "corr_change": (
                m["feature_corr_qpu_vs_ideal"]
                - r["feature_corr_qpu_vs_ideal"]
            ),

            "raw_noise_to_signal": r["noise_to_signal_ratio"],
            "m3_noise_to_signal": m["noise_to_signal_ratio"],
            "noise_to_signal_reduction": (
                r["noise_to_signal_ratio"]
                - m["noise_to_signal_ratio"]
            ),

            "raw_pred_perturbation_rms_claims": (
                r["direct_prediction_perturbation_rms_claims"]
            ),
            "m3_pred_perturbation_rms_claims": (
                m["direct_prediction_perturbation_rms_claims"]
            ),
            "pred_perturbation_rms_reduction_claims": (
                r["direct_prediction_perturbation_rms_claims"]
                - m["direct_prediction_perturbation_rms_claims"]
            ),

            "raw_systematic_shift_claims": (
                r["systematic_prediction_shift_claims"]
            ),
            "m3_systematic_shift_claims": (
                m["systematic_prediction_shift_claims"]
            ),
            "abs_systematic_shift_reduction_claims": (
                abs(r["systematic_prediction_shift_claims"])
                - abs(m["systematic_prediction_shift_claims"])
            ),

            "raw_random_perturbation_rms_claims": (
                r["random_prediction_perturbation_rms_claims"]
            ),
            "m3_random_perturbation_rms_claims": (
                m["random_prediction_perturbation_rms_claims"]
            ),
        })

    return diag, pd.DataFrame(improvement_rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help="IBM backend; default matches Candidate #3 Week-11 workflow.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help="Workload shots per measurement setting.",
    )
    parser.add_argument(
        "--m3-cal-shots",
        type=int,
        default=DEFAULT_M3_CAL_SHOTS,
        help="Shots per M3 calibration circuit.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit M3 calibration + ONE full-2025 SamplerV2 workload job.",
    )

    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError("--shots must be > 0.")
    if args.m3_cal_shots <= 0:
        raise ValueError("--m3-cal-shots must be > 0.")

    print("=" * 132)
    print("WEEK 11.1E.9A — CANDIDATE #3 M3 MEASUREMENT-MITIGATION TEST")
    print("=" * 132)
    print("2022-2024 = frozen training/readout")
    print("2025      = FULL 365-day RAW vs M3 paired hardware validation")
    print("2026      = FROZEN / NOT LOADED")
    print()

    # -------------------------------------------------------------------------
    # Freeze/audit exact Candidate #3 using original script functions
    # -------------------------------------------------------------------------
    candidate, manifest_meta = base.load_candidate()
    frozen = base.build_frozen_readout(candidate, manifest_meta)

    selected_endpoints = np.asarray(
        frozen["val_endpoints"],
        dtype=int,
    )

    if len(selected_endpoints) != EXPECTED_N_2025:
        raise RuntimeError(
            f"Expected 365 validation endpoints, found {len(selected_endpoints)}."
        )

    print("Frozen Candidate #3:")
    print(f"  candidate={base.CANDIDATE_KEY}")
    print(f"  topology={candidate['topology']}")
    print(f"  W={candidate['window']}")
    print(f"  r={candidate['r']}")
    print(f"  alpha={candidate['alpha']}")
    print(f"  dt={candidate['dt']}")
    print(f"  hx={candidate['hx']:+.12f}")
    print(f"  hy={candidate['hy']:+.12f}")
    print(f"  readout={base.EXPECTED_READOUT}")
    print(f"  workload shots/setting={args.shots}")
    print(f"  M3 calibration shots={args.m3_cal_shots}")
    print(f"  lambda={frozen['lambda']}")
    print(f"  training CV RMSE={frozen['cv_rmse']:.6f}")
    print(f"  ideal 2025 RMSE={frozen['ideal_rmse']:.6f}")
    print(
        "  J="
        + json.dumps(
            {
                f"J{i}{j}": float(v)
                for (i, j), v in candidate["J"].items()
            },
            sort_keys=True,
        )
    )
    print()

    # -------------------------------------------------------------------------
    # IBM service + fresh H2 layout reselection
    # -------------------------------------------------------------------------
    service = get_service()

    print("=" * 132)
    print(f"FRESH {args.backend} H2 HARDWARE RESELECTION")
    print("=" * 132)

    fresh = base.selector.fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=[args.backend],
        shortlist=SHORTLIST,
        write_prefix=f"{PREFIX}_{args.backend}",
        verbose=True,
    )

    chosen = fresh["selected"]
    layout = json.loads(chosen["layout"])

    backend = service.backend(
        args.backend,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    print()
    print("Fresh physical choice:")
    print(f"  backend={args.backend}")
    print(f"  layout={layout}")
    print(
        f"  max CZ error="
        f"{float(chosen.get('compiled_2q_error_max_percent', np.nan)):.6f}%"
    )
    print(
        f"  max readout error="
        f"{float(chosen.get('max_readout_error_percent', np.nan)):.6f}%"
    )
    print(
        f"  min T1={float(chosen.get('min_t1_us', np.nan)):.3f} us"
    )
    print(
        f"  min T2={float(chosen.get('min_t2_us', np.nan)):.3f} us"
    )
    print()

    # -------------------------------------------------------------------------
    # Compile exact same 365-day Candidate #3 workload
    # -------------------------------------------------------------------------
    settings = base.readout_settings()

    circuits = []
    circuit_meta = []
    resource_rows = []

    print("=" * 132)
    print("COMPILING FULL 365-DAY CANDIDATE #3 WORKLOAD")
    print("=" * 132)

    for k, endpoint in enumerate(selected_endpoints, 1):
        if k == 1 or k % 25 == 0 or k == len(selected_endpoints):
            print(
                f"  compiling validation endpoint {k}/{len(selected_endpoints)} "
                f"(global endpoint={int(endpoint)})"
            )

        for setting in settings:
            logical = base.build_measurement_circuit(
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

            rr = base.resource_row(
                isa,
                backend,
                endpoint=int(endpoint),
                setting=setting[0],
            )

            if rr["n_swap"] != 0:
                raise RuntimeError(
                    f"SWAP detected at endpoint={endpoint}, "
                    f"setting={setting[0]}."
                )

            circuits.append(isa)
            circuit_meta.append({
                "endpoint": int(endpoint),
                "setting": str(setting[0]),
            })
            resource_rows.append(rr)

    resources = pd.DataFrame(resource_rows)

    # -------------------------------------------------------------------------
    # Exact M3 final measurement mapping
    # -------------------------------------------------------------------------
    mappings = [
        normalize_mapping(
            mthree.utils.final_measurement_mapping(circuit)
        )
        for circuit in circuits
    ]

    unique_mappings = sorted(
        {tuple(m) for m in mappings}
    )

    if len(unique_mappings) != 1:
        raise RuntimeError(
            "All Candidate #3 circuits should measure the same four physical "
            "qubits in the same classical-bit order, but multiple final M3 "
            f"mappings were found:\n{unique_mappings}"
        )

    m3_mapping = list(unique_mappings[0])

    if len(m3_mapping) != 4:
        raise RuntimeError(
            f"Expected four measured qubits for XZinj_dropX3; "
            f"M3 mapping={m3_mapping}"
        )

    print()
    print("M3 final measurement mapping:")
    for classical_bit, physical_qubit in enumerate(m3_mapping):
        print(
            f"  classical bit c{classical_bit} <- physical qubit P{physical_qubit}"
        )

    # -------------------------------------------------------------------------
    # Resource preflight
    # -------------------------------------------------------------------------
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

    usage_before = base.get_service_usage(service)

    print()
    print("=" * 132)
    print("FULL-2025 RAW + M3 PREFLIGHT")
    print("=" * 132)
    print(f"Backend:                          {args.backend}")
    print(f"Fresh layout:                     {layout}")
    print(f"M3 measured physical qubits:      {m3_mapping}")
    print(f"Validation endpoints:             {len(selected_endpoints)}")
    print(f"Workload circuits:                {len(circuits)}")
    print(f"Settings/endpoint:                2")
    print(f"Workload shots/setting:           {args.shots}")
    print(f"M3 calibration method:            {M3_CAL_METHOD}")
    print(f"M3 calibration shots/circuit:     {args.m3_cal_shots}")
    print(
        f"Median CZ/feature vector:         "
        f"{feature_resources['cz_feature_vector'].median():.0f}"
    )
    print(
        f"Median max setting depth:         "
        f"{feature_resources['max_setting_depth'].median():.0f}"
    )
    print(
        f"Median feature duration:          "
        f"{feature_resources['duration_feature_vector_us'].median():.3f} us"
    )
    print(
        f"Scheduled workload duration*shots:"
        f" {scheduled_shot_seconds:.6f} s"
    )

    if usage_before is not None:
        print()
        print("Service usage before:")
        print(json.dumps(json_safe(usage_before), indent=2))

    # Save dry-run preflight.
    resources.to_csv(
        RESULTS / f"{PREFIX}_resources.csv",
        index=False,
    )

    preflight = {
        "step": "11.1E.9A",
        "candidate": base.CANDIDATE_KEY,
        "protocol": "RWP",
        "topology": candidate["topology"],
        "window": int(candidate["window"]),
        "r": int(candidate["r"]),
        "readout": base.EXPECTED_READOUT,
        "ridge_lambda": float(frozen["lambda"]),
        "training_cv_rmse": float(frozen["cv_rmse"]),
        "ideal_2025_rmse": float(frozen["ideal_rmse"]),
        "backend": args.backend,
        "layout": layout,
        "m3_mapping": m3_mapping,
        "n_endpoints": len(selected_endpoints),
        "n_workload_circuits": len(circuits),
        "workload_shots_per_setting": int(args.shots),
        "m3_cal_method": M3_CAL_METHOD,
        "m3_cal_shots": int(args.m3_cal_shots),
        "scheduled_workload_duration_times_shots_s": scheduled_shot_seconds,
        "service_usage_before": usage_before,
        "hardware_selection": json_safe(
            chosen.to_dict()
            if hasattr(chosen, "to_dict")
            else chosen
        ),
        "created_utc": utc_now_iso(),
    }

    (
        RESULTS / f"{PREFIX}_preflight.json"
    ).write_text(
        json.dumps(json_safe(preflight), indent=2),
        encoding="utf-8",
    )

    if not args.submit:
        print()
        print("DRY RUN COMPLETE — NO QPU JOB SUBMITTED.")
        print()
        print("To run the controlled M3 experiment:")
        print(
            f"  python {Path(__file__).name} --submit "
            f"--backend {args.backend} --shots {args.shots} "
            f"--m3-cal-shots {args.m3_cal_shots}"
        )
        print()
        print("2026 remains FROZEN / UNUSED.")
        return

    # =========================================================================
    # M3 CALIBRATION
    # =========================================================================

    print()
    print("=" * 132)
    print("M3 READOUT CALIBRATION")
    print("=" * 132)
    print(
        "Calibrating only the four physical qubits that generate c0..c3 "
        "for Candidate #3."
    )

    m3_cals_path = RESULTS / f"{PREFIX}_m3_calibrations.json"

    mit = mthree.M3Mitigation(backend)

    calibration_jobs = mit.cals_from_system(
        qubits=m3_mapping,
        shots=int(args.m3_cal_shots),
        method=M3_CAL_METHOD,
        rep_delay=None,
        cals_file=str(m3_cals_path),
        async_cal=False,
    )

    calibration_job_records = [
        extract_job_record(job)
        for job in (calibration_jobs or [])
    ]

    readout_fidelities = [
        float(x)
        for x in mit.readout_fidelity(m3_mapping)
    ]

    print("M3 calibration complete.")
    print(f"  mapping={m3_mapping}")
    print(
        "  readout fidelities="
        + json.dumps(
            {
                f"P{q}": f
                for q, f in zip(m3_mapping, readout_fidelities)
            },
            indent=2,
        )
    )

    (
        RESULTS / f"{PREFIX}_m3_calibration_jobs.json"
    ).write_text(
        json.dumps(
            json_safe({
                "mapping": m3_mapping,
                "cal_method": getattr(mit, "cal_method", None),
                "cal_timestamp": getattr(mit, "cal_timestamp", None),
                "cal_shots": getattr(mit, "cal_shots", None),
                "readout_fidelities": {
                    f"P{q}": f
                    for q, f in zip(m3_mapping, readout_fidelities)
                },
                "jobs": calibration_job_records,
            }),
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # ONE WORKLOAD JOB
    # =========================================================================

    print()
    print("=" * 132)
    print("SUBMITTING FULL 365-DAY CANDIDATE #3 SamplerV2 JOB")
    print("=" * 132)
    print(
        f"Submitting {len(circuits)} ISA circuits at "
        f"{args.shots} shots/setting..."
    )
    print(
        "The SAME returned counts will be scored RAW and then M3-corrected."
    )

    sampler = SamplerV2(mode=backend)

    job = sampler.run(
        circuits,
        shots=int(args.shots),
    )

    job_id = job.job_id()
    print(f"Job ID: {job_id}")

    result = job.result()
    final_status = str(job.status())
    print(f"Final status: {final_status}")

    main_job_record = extract_job_record(job)

    if len(result) != len(circuits):
        raise RuntimeError(
            f"Sampler returned {len(result)} pubs for "
            f"{len(circuits)} circuits."
        )

    # =========================================================================
    # RAW COUNTS + M3 QUASI DISTRIBUTIONS
    # =========================================================================

    raw_by_endpoint = {
        int(e): {}
        for e in selected_endpoints
    }
    m3_by_endpoint = {
        int(e): {}
        for e in selected_endpoints
    }

    raw_rows = []
    m3_rows = []

    print()
    print("=" * 132)
    print("APPLYING M3 TO THE SAME WORKLOAD COUNTS")
    print("=" * 132)

    for pub_result, meta, mapping in zip(result, circuit_meta, mappings):
        counts = base.get_pub_counts(pub_result)

        endpoint = int(meta["endpoint"])
        setting = str(meta["setting"])

        raw_by_endpoint[endpoint][setting] = counts

        quasi = mit.apply_correction(
            counts,
            mapping,
            method="auto",
        )

        quasi_dict = {
            str(k): float(v)
            for k, v in quasi.items()
        }

        m3_by_endpoint[endpoint][setting] = quasi_dict

        raw_rows.append({
            "endpoint": endpoint,
            "setting": setting,
            "counts_json": json.dumps(
                {str(k): int(v) for k, v in counts.items()},
                sort_keys=True,
            ),
        })

        m3_rows.append({
            "endpoint": endpoint,
            "setting": setting,
            "m3_mapping_json": json.dumps(mapping),
            "quasi_json": json.dumps(
                quasi_dict,
                sort_keys=True,
            ),
            "quasi_sum": float(sum(quasi_dict.values())),
            "quasi_min": float(min(quasi_dict.values())),
            "quasi_max": float(max(quasi_dict.values())),
        })

    raw_counts_df = pd.DataFrame(raw_rows)
    m3_quasi_df = pd.DataFrame(m3_rows)

    raw_counts_df.to_csv(
        RESULTS / f"{PREFIX}_raw_counts.csv",
        index=False,
    )
    m3_quasi_df.to_csv(
        RESULTS / f"{PREFIX}_m3_quasi_distributions.csv",
        index=False,
    )

    # =========================================================================
    # FEATURES
    # =========================================================================

    raw_features_by_endpoint = extract_features_from_distributions(
        raw_by_endpoint
    )
    m3_features_by_endpoint = extract_features_from_distributions(
        m3_by_endpoint
    )

    endpoint_to_master_row = {
        int(endpoint): idx
        for idx, endpoint in enumerate(frozen["endpoints"])
    }

    val_endpoint_to_pos = {
        int(endpoint): idx
        for idx, endpoint in enumerate(frozen["val_endpoints"])
    }

    X_ideal = []
    X_raw = []
    X_m3 = []
    targets = []
    pred_ideal = []

    daily_rows = []

    for endpoint in selected_endpoints:
        endpoint = int(endpoint)

        raw_vals = raw_features_by_endpoint[endpoint]
        m3_vals = m3_features_by_endpoint[endpoint]

        raw_vec = np.asarray(
            [raw_vals[f] for f in FEATURES],
            dtype=float,
        )
        m3_vec = np.asarray(
            [m3_vals[f] for f in FEATURES],
            dtype=float,
        )

        ideal_vec = np.asarray(
            frozen["X"][endpoint_to_master_row[endpoint]],
            dtype=float,
        )

        val_pos = val_endpoint_to_pos[endpoint]
        target = float(frozen["y_val"][val_pos])
        pideal = float(frozen["pred_val"][val_pos])

        X_ideal.append(ideal_vec)
        X_raw.append(raw_vec)
        X_m3.append(m3_vec)
        targets.append(target)
        pred_ideal.append(pideal)

        row = {
            "endpoint": endpoint,
            "target": target,
            "pred_ideal": pideal,

            "Z3_from_X_setting_raw": float(
                raw_vals["Z3_from_X_setting"]
            ),
            "Z3_from_Z_setting_raw": float(
                raw_vals["Z3"]
            ),
            "Z3_cross_setting_diff_raw": float(
                raw_vals["Z3_from_X_setting"] - raw_vals["Z3"]
            ),

            "Z3_from_X_setting_m3": float(
                m3_vals["Z3_from_X_setting"]
            ),
            "Z3_from_Z_setting_m3": float(
                m3_vals["Z3"]
            ),
            "Z3_cross_setting_diff_m3": float(
                m3_vals["Z3_from_X_setting"] - m3_vals["Z3"]
            ),
        }

        for j, feature in enumerate(FEATURES):
            row[f"{feature}_ideal"] = float(ideal_vec[j])
            row[f"{feature}_raw"] = float(raw_vec[j])
            row[f"{feature}_m3"] = float(m3_vec[j])
            row[f"{feature}_raw_minus_ideal"] = float(
                raw_vec[j] - ideal_vec[j]
            )
            row[f"{feature}_m3_minus_ideal"] = float(
                m3_vec[j] - ideal_vec[j]
            )

        daily_rows.append(row)

    X_ideal = np.asarray(X_ideal, dtype=float)
    X_raw = np.asarray(X_raw, dtype=float)
    X_m3 = np.asarray(X_m3, dtype=float)
    targets = np.asarray(targets, dtype=float)
    pred_ideal = np.asarray(pred_ideal, dtype=float)

    pred_raw = base.common.predict_scaled_ridge(
        frozen["model"],
        frozen["scaler"],
        frozen["keep"],
        X_raw,
    )

    pred_m3 = base.common.predict_scaled_ridge(
        frozen["model"],
        frozen["scaler"],
        frozen["keep"],
        X_m3,
    )

    daily_df = pd.DataFrame(daily_rows)
    daily_df["pred_raw"] = pred_raw
    daily_df["pred_m3"] = pred_m3

    daily_df["abs_error_ideal"] = np.abs(
        targets - pred_ideal
    )
    daily_df["abs_error_raw"] = np.abs(
        targets - pred_raw
    )
    daily_df["abs_error_m3"] = np.abs(
        targets - pred_m3
    )

    daily_df.to_csv(
        RESULTS / f"{PREFIX}_features_predictions.csv",
        index=False,
    )

    # =========================================================================
    # FEATURE-LEVEL RAW VS M3 DIAGNOSIS
    # =========================================================================

    intercept_raw, beta_raw, beta_audit = reconstruct_effective_raw_beta(
        X_ideal,
        pred_ideal,
    )

    feature_diag, feature_improvement = make_feature_diagnostics(
        X_ideal=X_ideal,
        X_raw=X_raw,
        X_m3=X_m3,
        beta_raw=beta_raw,
    )

    feature_diag.to_csv(
        RESULTS / f"{PREFIX}_feature_diagnostics.csv",
        index=False,
    )

    feature_improvement.to_csv(
        RESULTS / f"{PREFIX}_feature_improvement.csv",
        index=False,
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    raw_diff = X_raw - X_ideal
    m3_diff = X_m3 - X_ideal

    summary = {
        "step": "11.1E.9A",
        "candidate": base.CANDIDATE_KEY,
        "backend": args.backend,
        "layout": layout,
        "m3_mapping": m3_mapping,
        "workload_job_id": str(job_id),
        "workload_job_status": final_status,

        "shots_per_setting": int(args.shots),
        "m3_cal_shots": int(args.m3_cal_shots),
        "m3_cal_method": M3_CAL_METHOD,
        "m3_cal_timestamp": getattr(mit, "cal_timestamp", None),
        "m3_readout_fidelities": {
            f"P{q}": f
            for q, f in zip(m3_mapping, readout_fidelities)
        },

        "n_validation_endpoints": int(len(selected_endpoints)),

        "ideal_rmse": rmse(targets, pred_ideal),
        "raw_rmse": rmse(targets, pred_raw),
        "m3_rmse": rmse(targets, pred_m3),

        "m3_minus_raw_rmse": (
            rmse(targets, pred_m3) - rmse(targets, pred_raw)
        ),
        "m3_rmse_improvement": (
            rmse(targets, pred_raw) - rmse(targets, pred_m3)
        ),

        "ideal_mae": mae(targets, pred_ideal),
        "raw_mae": mae(targets, pred_raw),
        "m3_mae": mae(targets, pred_m3),

        "ideal_bias": bias(targets, pred_ideal),
        "raw_bias": bias(targets, pred_raw),
        "m3_bias": bias(targets, pred_m3),

        "raw_prediction_corr_vs_ideal": corr(
            pred_raw, pred_ideal
        ),
        "m3_prediction_corr_vs_ideal": corr(
            pred_m3, pred_ideal
        ),

        "raw_feature_mae": float(
            np.mean(np.abs(raw_diff))
        ),
        "m3_feature_mae": float(
            np.mean(np.abs(m3_diff))
        ),

        "raw_feature_rmse": float(
            np.sqrt(np.mean(raw_diff ** 2))
        ),
        "m3_feature_rmse": float(
            np.sqrt(np.mean(m3_diff ** 2))
        ),

        "raw_z3_cross_setting_mae": float(
            np.mean(
                np.abs(daily_df["Z3_cross_setting_diff_raw"])
            )
        ),
        "m3_z3_cross_setting_mae": float(
            np.mean(
                np.abs(daily_df["Z3_cross_setting_diff_m3"])
            )
        ),

        "effective_raw_readout_intercept": intercept_raw,
        "effective_raw_readout_beta": {
            f: float(b)
            for f, b in zip(FEATURES, beta_raw)
        },
        "readout_reconstruction_audit_rmse": beta_audit,

        "calibration_jobs": calibration_job_records,
        "workload_job": main_job_record,
        "service_usage_before": usage_before,
        "service_usage_after": base.get_service_usage(service),
        "completed_utc": utc_now_iso(),
        "2026_loaded": False,
    }

    (
        RESULTS / f"{PREFIX}_summary.json"
    ).write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )

    # =========================================================================
    # CONSOLE RESULTS
    # =========================================================================

    print()
    print("=" * 132)
    print("CANDIDATE #3 — RAW VS M3 RESULTS")
    print("=" * 132)
    print(f"Ideal 2025 RMSE:                 {summary['ideal_rmse']:.6f}")
    print(f"RAW QPU RMSE:                    {summary['raw_rmse']:.6f}")
    print(f"M3 QPU RMSE:                     {summary['m3_rmse']:.6f}")
    print(
        f"M3 RMSE improvement vs RAW:      "
        f"{summary['m3_rmse_improvement']:+.6f}"
    )
    print()
    print(f"RAW QPU MAE:                     {summary['raw_mae']:.6f}")
    print(f"M3 QPU MAE:                      {summary['m3_mae']:.6f}")
    print()
    print(f"RAW QPU bias:                    {summary['raw_bias']:+.6f}")
    print(f"M3 QPU bias:                     {summary['m3_bias']:+.6f}")
    print()
    print(
        f"RAW/ideal prediction corr:       "
        f"{summary['raw_prediction_corr_vs_ideal']:.6f}"
    )
    print(
        f"M3/ideal prediction corr:        "
        f"{summary['m3_prediction_corr_vs_ideal']:.6f}"
    )
    print()
    print(f"RAW feature RMSE:                {summary['raw_feature_rmse']:.6f}")
    print(f"M3 feature RMSE:                 {summary['m3_feature_rmse']:.6f}")
    print()

    print("Feature-level RAW -> M3 changes:")
    print("-" * 132)

    display_cols = [
        "feature",
        "raw_feature_bias",
        "m3_feature_bias",
        "abs_bias_reduction",
        "raw_feature_rmse",
        "m3_feature_rmse",
        "feature_rmse_reduction_pct",
        "raw_corr",
        "m3_corr",
        "raw_noise_to_signal",
        "m3_noise_to_signal",
        "raw_pred_perturbation_rms_claims",
        "m3_pred_perturbation_rms_claims",
    ]

    print(
        feature_improvement[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    z2 = feature_improvement[
        feature_improvement["feature"] == "Z2"
    ].iloc[0]

    print("PRIMARY Z2 TEST:")
    print(
        f"  |bias|:                 "
        f"{abs(z2['raw_feature_bias']):.6f} -> "
        f"{abs(z2['m3_feature_bias']):.6f}"
    )
    print(
        f"  feature RMSE:            "
        f"{z2['raw_feature_rmse']:.6f} -> "
        f"{z2['m3_feature_rmse']:.6f}"
    )
    print(
        f"  QPU/ideal correlation:   "
        f"{z2['raw_corr']:.6f} -> "
        f"{z2['m3_corr']:.6f}"
    )
    print(
        f"  noise/signal:            "
        f"{z2['raw_noise_to_signal']:.6f} -> "
        f"{z2['m3_noise_to_signal']:.6f}"
    )
    print(
        f"  pred perturbation RMS:   "
        f"{z2['raw_pred_perturbation_rms_claims']:.6f} -> "
        f"{z2['m3_pred_perturbation_rms_claims']:.6f} claims"
    )

    print()
    print("Saved:")
    for name in [
        f"{PREFIX}_preflight.json",
        f"{PREFIX}_resources.csv",
        f"{PREFIX}_m3_calibrations.json",
        f"{PREFIX}_m3_calibration_jobs.json",
        f"{PREFIX}_raw_counts.csv",
        f"{PREFIX}_m3_quasi_distributions.csv",
        f"{PREFIX}_features_predictions.csv",
        f"{PREFIX}_feature_diagnostics.csv",
        f"{PREFIX}_feature_improvement.csv",
        f"{PREFIX}_summary.json",
    ]:
        print(f"  results/{name}")

    print()
    print("2026 remains FROZEN / UNUSED.")


if __name__ == "__main__":
    main()
