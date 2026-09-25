from __future__ import annotations

"""
WEEK 11.1E.10A
Candidate #1 — basic M3 measurement-mitigation completion test.

This is the same scientific design as 11_1E9A_candidate3_m3.py:
  * frozen logical candidate/readout;
  * 2022-2024 only for the frozen Ridge readout;
  * full 365-day 2025 hardware validation;
  * 2026 is never loaded;
  * ordinary RWP reset semantics (NO reset-aware selection, NO reset_2);
  * fresh Rule-5 Kingston H3 physical reselection immediately before execution;
  * ONE SamplerV2 workload job;
  * the SAME returned counts are scored RAW and M3-corrected;
  * M3 corrects final measurement assignment only.

Frozen Candidate #1:
  H3 / RWP W=4 / r=3 / alpha=0.25
  readout = XZinj_dropX3_plus_YX45
  features = [X0,X1,X2,Z0,Z1,Z2,Z3,YX45]

The exact logical candidate and ideal readout are reconstructed from the already
validated Week-10 reference implementation:
    10_01_reference_embedding_refresh_and_simulation.py

Default is a DRY RUN.  No QPU job is submitted unless --submit is supplied.

Examples
--------
Dry run:
    python 11_1E10A_candidate1_m3.py

Submit:
    python 11_1E10A_candidate1_m3.py --submit --backend ibm_kingston \
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
        'M3 is not installed. Install it with:\n\n'
        '    pip install "mthree>=3.0"\n'
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

BASE_SCRIPT = HERE / "10_01_reference_embedding_refresh_and_simulation.py"
PREFIX = "11_1E10A_candidate1_m3"

DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_SHOTS = 1024
DEFAULT_M3_CAL_SHOTS = 10_000
M3_CAL_METHOD = "balanced"

OPT_LEVEL = 1
SEED_TRANSPILE = 42
SHORTLIST = 120
EXPECTED_N_2025 = 365

FEATURES = [
    "X0", "X1", "X2",
    "Z0", "Z1", "Z2", "Z3",
    "YX45",
]

# Exact Candidate-1 settings from Week 10:
#   XXXZYX: X0,X1,X2; Z3; Y4,X5 -> YX45 parity
#   ZZZZZZ: authoritative Z0,Z1,Z2,Z3
SETTING_X = "XXXZYX"
SETTING_Z = "ZZZZZZ"


# =============================================================================
# LOAD AUTHORITATIVE WEEK-10 CANDIDATE-1 IMPLEMENTATION
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required Week-10 Candidate #1 source not found:\n  {path}"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = load_module(BASE_SCRIPT, "week10_candidate1_base")


# =============================================================================
# METRICS / SERIALIZATION
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


def get_pub_counts(pub_result):
    data = pub_result.data
    reg = getattr(data, "m", None)
    if reg is None:
        raise RuntimeError("Sampler result does not contain classical register 'm'.")
    return reg.get_counts()


def expectation_from_distribution(dist, classical_bits):
    """Expectation from raw counts or an M3 quasi-probability distribution."""
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
    if isinstance(mapping, np.ndarray):
        mapping = mapping.tolist()
    if isinstance(mapping, tuple):
        mapping = list(mapping)
    if isinstance(mapping, list):
        return [int(x) for x in mapping]
    if isinstance(mapping, dict):
        try:
            return [int(mapping[k]) for k in sorted(mapping, key=int)]
        except Exception:
            return [int(v) for _, v in sorted(mapping.items(), key=lambda kv: str(kv[0]))]
    raise TypeError(
        f"Unsupported final_measurement_mapping type: {type(mapping)} -> {mapping!r}"
    )


def get_service_usage(service):
    """Best-effort IBM usage snapshot; absence never blocks the experiment."""
    for name in ("usage", "usage_info"):
        fn = getattr(service, name, None)
        if callable(fn):
            try:
                return json_safe(fn())
            except Exception:
                pass
    return None


def extract_job_record(job):
    out = {"job_id": None, "status": None, "metrics": None}
    try:
        out["job_id"] = str(job.job_id())
    except Exception:
        pass
    try:
        out["status"] = str(job.status())
    except Exception:
        pass
    try:
        out["metrics"] = json_safe(job.metrics())
    except Exception:
        pass
    return json_safe(out)


def reconstruct_effective_raw_beta(X_ideal, pred_ideal):
    """Reconstruct pred = intercept + X @ beta without using targets."""
    X = np.asarray(X_ideal, dtype=float)
    p = np.asarray(pred_ideal, dtype=float)
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, p, rcond=None)
    intercept = float(coef[0])
    beta = np.asarray(coef[1:], dtype=float)
    audit = rmse(p, intercept + X @ beta)
    if audit > 1e-8:
        raise RuntimeError(
            f"Frozen readout reconstruction audit failed: RMSE={audit:.3e}"
        )
    return intercept, beta, audit


# =============================================================================
# FEATURE PARSING
# =============================================================================

def extract_features_from_distributions(per_endpoint):
    by_endpoint = {}

    for endpoint, setting_map in per_endpoint.items():
        if SETTING_X not in setting_map or SETTING_Z not in setting_map:
            raise RuntimeError(
                f"Endpoint {endpoint}: missing {SETTING_X} or {SETTING_Z}."
            )

        dist_x = setting_map[SETTING_X]
        dist_z = setting_map[SETTING_Z]

        vals = {
            "X0": expectation_from_distribution(dist_x, [0]),
            "X1": expectation_from_distribution(dist_x, [1]),
            "X2": expectation_from_distribution(dist_x, [2]),
            "Z3_from_X_setting": expectation_from_distribution(dist_x, [3]),
            "Z0": expectation_from_distribution(dist_z, [0]),
            "Z1": expectation_from_distribution(dist_z, [1]),
            "Z2": expectation_from_distribution(dist_z, [2]),
            "Z3": expectation_from_distribution(dist_z, [3]),
            "YX45": expectation_from_distribution(dist_x, [4, 5]),
        }
        by_endpoint[int(endpoint)] = vals

    return by_endpoint


# =============================================================================
# FEATURE DIAGNOSTICS
# =============================================================================

def make_feature_diagnostics(X_ideal, X_raw, X_m3, beta_raw):
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
            f_bias = float(np.mean(delta))
            f_rmse = float(np.sqrt(np.mean(delta ** 2)))
            stochastic_rms = float(
                np.sqrt(max(f_rmse ** 2 - f_bias ** 2, 0.0))
            )

            rows.append({
                "feature": feature,
                "method": method,
                "beta": float(beta_raw[j]),
                "ideal_std": sigma_ideal,
                "feature_bias": f_bias,
                "feature_mae": float(np.mean(np.abs(delta))),
                "feature_rmse": f_rmse,
                "feature_stochastic_rms": stochastic_rms,
                "feature_corr_qpu_vs_ideal": corr(qpu, ideal),
                "qpu_std": float(np.std(qpu, ddof=1)),
                "qpu_to_ideal_std_ratio": float(np.std(qpu, ddof=1) / sigma_ideal),
                "noise_to_signal_ratio": f_rmse / sigma_ideal,
                "direct_prediction_perturbation_rms_claims": abs(float(beta_raw[j])) * f_rmse,
                "systematic_prediction_shift_claims": float(beta_raw[j]) * f_bias,
                "random_prediction_perturbation_rms_claims": abs(float(beta_raw[j])) * stochastic_rms,
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
            "abs_bias_reduction": abs(r["feature_bias"]) - abs(m["feature_bias"]),
            "raw_feature_rmse": r["feature_rmse"],
            "m3_feature_rmse": m["feature_rmse"],
            "feature_rmse_reduction": r["feature_rmse"] - m["feature_rmse"],
            "feature_rmse_reduction_pct": (
                100.0 * (r["feature_rmse"] - m["feature_rmse"]) / r["feature_rmse"]
                if r["feature_rmse"] != 0 else np.nan
            ),
            "raw_corr": r["feature_corr_qpu_vs_ideal"],
            "m3_corr": m["feature_corr_qpu_vs_ideal"],
            "corr_change": m["feature_corr_qpu_vs_ideal"] - r["feature_corr_qpu_vs_ideal"],
            "raw_noise_to_signal": r["noise_to_signal_ratio"],
            "m3_noise_to_signal": m["noise_to_signal_ratio"],
            "noise_to_signal_reduction": r["noise_to_signal_ratio"] - m["noise_to_signal_ratio"],
            "raw_pred_perturbation_rms_claims": r["direct_prediction_perturbation_rms_claims"],
            "m3_pred_perturbation_rms_claims": m["direct_prediction_perturbation_rms_claims"],
            "pred_perturbation_rms_reduction_claims": (
                r["direct_prediction_perturbation_rms_claims"]
                - m["direct_prediction_perturbation_rms_claims"]
            ),
            "raw_systematic_shift_claims": r["systematic_prediction_shift_claims"],
            "m3_systematic_shift_claims": m["systematic_prediction_shift_claims"],
            "abs_systematic_shift_reduction_claims": (
                abs(r["systematic_prediction_shift_claims"])
                - abs(m["systematic_prediction_shift_claims"])
            ),
            "raw_random_perturbation_rms_claims": r["random_prediction_perturbation_rms_claims"],
            "m3_random_perturbation_rms_claims": m["random_prediction_perturbation_rms_claims"],
        })

    return diag, pd.DataFrame(improvement_rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--m3-cal-shots", type=int, default=DEFAULT_M3_CAL_SHOTS)
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
    print("WEEK 11.1E.10A — CANDIDATE #1 BASIC M3 COMPLETION TEST")
    print("=" * 132)
    print("2022-2024 = frozen training/readout")
    print("2025      = FULL 365-day RAW vs M3 paired hardware validation")
    print("2026      = FROZEN / NOT LOADED")
    print("ordinary RWP reset + ordinary Rule-5 layout selection; no reset-aware changes")
    print()

    # -------------------------------------------------------------------------
    # Exact logical Candidate #1 and frozen ideal Ridge
    # -------------------------------------------------------------------------
    candidate, source_row, frozen_meta = base.reconstruct_reference_candidate()
    ideal = base.ideal_reference(candidate)

    if list(base.FEATURES) != FEATURES:
        raise RuntimeError(
            f"Week-10 feature order changed. Expected {FEATURES}, got {list(base.FEATURES)}"
        )

    selected_endpoints = np.asarray(ideal["val_endpoints"], dtype=int)
    if len(selected_endpoints) != EXPECTED_N_2025:
        raise RuntimeError(
            f"Expected 365 validation endpoints, found {len(selected_endpoints)}."
        )

    print("Frozen Candidate #1:")
    print(f"  topology={candidate['topology']}")
    print(f"  W={candidate['window']}")
    print(f"  r={candidate['r']}")
    print(f"  alpha={candidate['alpha']}")
    print(f"  dt={candidate['dt']}")
    print(f"  hx={candidate['hx']:+.12f}")
    print(f"  hy={candidate['hy']:+.12f}")
    print(f"  readout={base.READOUT}")
    print(f"  workload shots/setting={args.shots}")
    print(f"  M3 calibration shots={args.m3_cal_shots}")
    print(f"  lambda={ideal['metrics']['selected_lambda_recomputed']}")
    print(f"  training CV RMSE={ideal['metrics']['cv_rmse_recomputed']:.6f}")
    print(f"  ideal 2025 RMSE={ideal['metrics']['ideal_2025_rmse']:.6f}")
    print(
        "  J="
        + json.dumps(
            {f"J{i}{j}": float(v) for (i, j), v in candidate["J"].items()},
            sort_keys=True,
        )
    )
    print()

    # -------------------------------------------------------------------------
    # Fresh ordinary Rule-5 H3 reselection
    # -------------------------------------------------------------------------
    service = get_service()
    backend = service.backend(args.backend, use_fractional_gates=False)
    try:
        backend.refresh()
    except Exception:
        pass

    # The imported Week-10 helper uses this global backend name.
    base.BACKEND_NAME = str(args.backend)
    base.selector.ALPHA = float(candidate["alpha"])
    base.selector.DT = float(candidate["dt"])

    print("=" * 132)
    print(f"FRESH {args.backend} H3 RULE-5 HARDWARE RESELECTION")
    print("=" * 132)

    fresh = base.selector.fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=[args.backend],
        shortlist=SHORTLIST,
        write_prefix=f"{PREFIX}_{args.backend}",
        verbose=True,
    )

    compiled = fresh["compiled"].copy()
    good = compiled[compiled["strict_compile_pass"].astype(bool)].copy()
    if len(good) == 0:
        raise RuntimeError("No strict zero-SWAP H3 embeddings survived fresh reselection.")

    ranked = base.strict_rule5_sort(good)
    ranked["rule5_rank_current"] = np.arange(len(ranked)) + 1
    chosen = ranked.iloc[0]
    layout = base.parse_layout(chosen["layout"])

    print()
    print("Fresh physical choice:")
    print(f"  backend={args.backend}")
    print(f"  layout={layout}")
    print(f"  max CZ error={float(chosen.get('compiled_2q_error_max_percent', np.nan)):.6f}%")
    print(f"  max readout error={float(chosen.get('max_readout_error_percent', np.nan)):.6f}%")
    print(f"  max 1Q error={float(chosen.get('compiled_1q_error_max_percent', np.nan)):.6f}%")
    print(f"  min T1={float(chosen.get('min_t1_us', np.nan)):.3f} us")
    print(f"  min T2={float(chosen.get('min_t2_us', np.nan)):.3f} us")
    print()

    # -------------------------------------------------------------------------
    # Transpile the two exact parameterized Candidate-1 measurement settings
    # once, then bind all 365 endpoint histories.
    # -------------------------------------------------------------------------
    transpiled = {}
    params_by_setting = {}
    setting_resources = []

    print("=" * 132)
    print("COMPILING CANDIDATE #1 PARAMETERIZED MEASUREMENT SETTINGS")
    print("=" * 132)

    for setting_label in (SETTING_X, SETTING_Z):
        logical, pars = base.build_parameterized_measurement_circuit(
            candidate, setting_label
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

        ops = {str(k): int(v) for k, v in isa.count_ops().items()}
        if int(ops.get("swap", 0)) != 0:
            raise RuntimeError(f"SWAP detected in setting {setting_label}.")

        try:
            duration_us = float(isa.estimate_duration(backend.target, unit="s")) * 1e6
        except Exception:
            duration_us = np.nan

        transpiled[setting_label] = isa
        params_by_setting[setting_label] = pars
        setting_resources.append({
            "setting": setting_label,
            "depth": int(isa.depth()),
            "size": int(isa.size()),
            "n_cz": int(ops.get("cz", 0)),
            "n_swap": int(ops.get("swap", 0)),
            "n_reset": int(ops.get("reset", 0)),
            "n_measure": int(ops.get("measure", 0)),
            "duration_us": duration_us,
            "operations_json": json.dumps(ops, sort_keys=True),
        })

    setting_resources_df = pd.DataFrame(setting_resources)

    circuits = []
    circuit_meta = []

    print()
    print("Binding full 365-day Candidate #1 workload...")
    for k, endpoint in enumerate(selected_endpoints, 1):
        if k == 1 or k % 25 == 0 or k == len(selected_endpoints):
            print(
                f"  binding validation endpoint {k}/{len(selected_endpoints)} "
                f"(global endpoint={int(endpoint)})"
            )

        for setting_label in (SETTING_X, SETTING_Z):
            bound = base.bind_endpoint_circuit(
                transpiled[setting_label],
                params_by_setting[setting_label],
                ideal["angles"],
                int(endpoint),
            )
            circuits.append(bound)
            circuit_meta.append({
                "endpoint": int(endpoint),
                "setting": setting_label,
            })

    # -------------------------------------------------------------------------
    # Exact M3 final measurement mapping. Candidate #1 measures all 6 qubits
    # because YX45 needs memory q4/q5 in the first setting.
    # -------------------------------------------------------------------------
    mappings = [
        normalize_mapping(mthree.utils.final_measurement_mapping(circuit))
        for circuit in circuits
    ]
    unique_mappings = sorted({tuple(m) for m in mappings})
    if len(unique_mappings) != 1:
        raise RuntimeError(
            "Candidate #1 circuits should share one final measurement mapping, "
            f"but found: {unique_mappings}"
        )

    m3_mapping = list(unique_mappings[0])
    if len(m3_mapping) != 6:
        raise RuntimeError(
            "Candidate #1 measures six qubits for XZinj_dropX3_plus_YX45; "
            f"M3 mapping={m3_mapping}"
        )

    print()
    print("M3 final measurement mapping:")
    for c, q in enumerate(m3_mapping):
        print(f"  classical bit c{c} <- physical qubit P{q}")

    # -------------------------------------------------------------------------
    # Preflight
    # -------------------------------------------------------------------------
    feature_cz = int(setting_resources_df["n_cz"].sum())
    feature_duration_us = float(setting_resources_df["duration_us"].sum())
    max_depth = int(setting_resources_df["depth"].max())
    max_setting_duration_us = float(setting_resources_df["duration_us"].max())
    feature_resets = int(setting_resources_df["n_reset"].sum())

    scheduled_shot_seconds = float(
        feature_duration_us * 1e-6 * len(selected_endpoints) * int(args.shots)
    )
    usage_before = get_service_usage(service)

    print()
    print("=" * 132)
    print("FULL-2025 CANDIDATE #1 RAW + M3 PREFLIGHT")
    print("=" * 132)
    print(f"Backend:                          {args.backend}")
    print(f"Fresh layout:                     {layout}")
    print(f"M3 measured physical qubits:      {m3_mapping}")
    print(f"Validation endpoints:             {len(selected_endpoints)}")
    print(f"Workload circuits:                {len(circuits)}")
    print("Settings/endpoint:                2")
    print(f"Workload shots/setting:           {args.shots}")
    print(f"M3 calibration method:            {M3_CAL_METHOD}")
    print(f"M3 calibration shots/circuit:     {args.m3_cal_shots}")
    print(f"CZ/feature vector:                {feature_cz}")
    print(f"Max setting depth:                {max_depth}")
    print(f"Generic resets/feature vector:    {feature_resets}")
    print(f"Feature duration:                 {feature_duration_us:.3f} us")
    print(f"Max-setting duration:             {max_setting_duration_us:.3f} us")
    print(f"Scheduled workload duration*shots:{scheduled_shot_seconds:.6f} s")

    if usage_before is not None:
        print()
        print("Service usage before:")
        print(json.dumps(json_safe(usage_before), indent=2))

    setting_resources_df.to_csv(
        RESULTS / f"{PREFIX}_setting_resources.csv", index=False
    )

    preflight = {
        "step": "11.1E.10A",
        "candidate": "Candidate_1_H3_RWP_W4_R1_R3",
        "protocol": "RWP",
        "topology": candidate["topology"],
        "window": int(candidate["window"]),
        "r": int(candidate["r"]),
        "alpha": float(candidate["alpha"]),
        "readout": base.READOUT,
        "features": FEATURES,
        "ridge_lambda": float(ideal["metrics"]["selected_lambda_recomputed"]),
        "training_cv_rmse": float(ideal["metrics"]["cv_rmse_recomputed"]),
        "ideal_2025_rmse": float(ideal["metrics"]["ideal_2025_rmse"]),
        "backend": args.backend,
        "layout": layout,
        "m3_mapping": m3_mapping,
        "n_endpoints": int(len(selected_endpoints)),
        "n_workload_circuits": int(len(circuits)),
        "workload_shots_per_setting": int(args.shots),
        "m3_cal_method": M3_CAL_METHOD,
        "m3_cal_shots": int(args.m3_cal_shots),
        "feature_vector_n_cz": feature_cz,
        "max_setting_depth": max_depth,
        "feature_vector_duration_us": feature_duration_us,
        "max_setting_duration_us": max_setting_duration_us,
        "feature_vector_reset_count": feature_resets,
        "scheduled_workload_duration_times_shots_s": scheduled_shot_seconds,
        "service_usage_before": usage_before,
        "hardware_selection": json_safe(chosen.to_dict()),
        "created_utc": utc_now_iso(),
    }
    (RESULTS / f"{PREFIX}_preflight.json").write_text(
        json.dumps(json_safe(preflight), indent=2), encoding="utf-8"
    )

    if not args.submit:
        print()
        print("DRY RUN COMPLETE — NO QPU JOB SUBMITTED.")
        print()
        print("To submit Candidate #1 after reviewing this preflight:")
        print(
            f"  python {Path(__file__).name} --submit --backend {args.backend} "
            f"--shots {args.shots} --m3-cal-shots {args.m3_cal_shots}"
        )
        print()
        print("2026 remains FROZEN / UNUSED.")
        return

    # =========================================================================
    # M3 CALIBRATION
    # =========================================================================
    print()
    print("=" * 132)
    print("M3 READOUT CALIBRATION — CANDIDATE #1")
    print("=" * 132)
    print(
        "Calibrating the six physical qubits participating in the final "
        "Candidate #1 measurement."
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
        extract_job_record(job) for job in (calibration_jobs or [])
    ]
    readout_fidelities = [float(x) for x in mit.readout_fidelity(m3_mapping)]

    print("M3 calibration complete.")
    print(f"  mapping={m3_mapping}")
    print(
        "  readout fidelities="
        + json.dumps(
            {f"P{q}": f for q, f in zip(m3_mapping, readout_fidelities)},
            indent=2,
        )
    )

    (RESULTS / f"{PREFIX}_m3_calibration_jobs.json").write_text(
        json.dumps(
            json_safe({
                "mapping": m3_mapping,
                "cal_method": getattr(mit, "cal_method", None),
                "cal_timestamp": getattr(mit, "cal_timestamp", None),
                "cal_shots": getattr(mit, "cal_shots", None),
                "readout_fidelities": {
                    f"P{q}": f for q, f in zip(m3_mapping, readout_fidelities)
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
    print("SUBMITTING FULL 365-DAY CANDIDATE #1 SamplerV2 JOB")
    print("=" * 132)
    print(f"Submitting {len(circuits)} ISA circuits at {args.shots} shots/setting...")
    print("The SAME returned counts will be scored RAW and then M3-corrected.")

    sampler = SamplerV2(mode=backend)
    job = sampler.run(circuits, shots=int(args.shots))
    job_id = job.job_id()
    print(f"Job ID: {job_id}")
    result = job.result()
    final_status = str(job.status())
    print(f"Final status: {final_status}")

    main_job_record = extract_job_record(job)
    if len(result) != len(circuits):
        raise RuntimeError(
            f"Sampler returned {len(result)} pubs for {len(circuits)} circuits."
        )

    # =========================================================================
    # SAME COUNTS -> RAW AND M3
    # =========================================================================
    raw_by_endpoint = {int(e): {} for e in selected_endpoints}
    m3_by_endpoint = {int(e): {} for e in selected_endpoints}
    raw_rows = []
    m3_rows = []

    print()
    print("=" * 132)
    print("APPLYING M3 TO THE SAME CANDIDATE #1 WORKLOAD COUNTS")
    print("=" * 132)

    for pub_result, meta, mapping in zip(result, circuit_meta, mappings):
        counts = get_pub_counts(pub_result)
        endpoint = int(meta["endpoint"])
        setting = str(meta["setting"])

        raw_by_endpoint[endpoint][setting] = counts
        quasi = mit.apply_correction(counts, mapping, method="auto")
        quasi_dict = {str(k): float(v) for k, v in quasi.items()}
        m3_by_endpoint[endpoint][setting] = quasi_dict

        raw_rows.append({
            "endpoint": endpoint,
            "setting": setting,
            "counts_json": json.dumps(
                {str(k): int(v) for k, v in counts.items()}, sort_keys=True
            ),
        })
        m3_rows.append({
            "endpoint": endpoint,
            "setting": setting,
            "m3_mapping_json": json.dumps(mapping),
            "quasi_json": json.dumps(quasi_dict, sort_keys=True),
            "quasi_sum": float(sum(quasi_dict.values())),
            "quasi_min": float(min(quasi_dict.values())),
            "quasi_max": float(max(quasi_dict.values())),
        })

    pd.DataFrame(raw_rows).to_csv(
        RESULTS / f"{PREFIX}_raw_counts.csv", index=False
    )
    pd.DataFrame(m3_rows).to_csv(
        RESULTS / f"{PREFIX}_m3_quasi_distributions.csv", index=False
    )

    # =========================================================================
    # FEATURES + FROZEN RIDGE
    # =========================================================================
    raw_features_by_endpoint = extract_features_from_distributions(raw_by_endpoint)
    m3_features_by_endpoint = extract_features_from_distributions(m3_by_endpoint)

    val_endpoint_to_pos = {
        int(endpoint): idx for idx, endpoint in enumerate(ideal["val_endpoints"])
    }

    X_ideal = []
    X_raw = []
    X_m3 = []
    targets = []
    pred_ideal = []
    daily_rows = []

    for endpoint in selected_endpoints:
        endpoint = int(endpoint)
        pos = val_endpoint_to_pos[endpoint]
        raw_vals = raw_features_by_endpoint[endpoint]
        m3_vals = m3_features_by_endpoint[endpoint]

        raw_vec = np.asarray([raw_vals[f] for f in FEATURES], dtype=float)
        m3_vec = np.asarray([m3_vals[f] for f in FEATURES], dtype=float)
        ideal_vec = np.asarray(ideal["X_val_ideal"][pos], dtype=float)
        target = float(ideal["y_val"][pos])
        pideal = float(ideal["pred_val_ideal"][pos])

        X_ideal.append(ideal_vec)
        X_raw.append(raw_vec)
        X_m3.append(m3_vec)
        targets.append(target)
        pred_ideal.append(pideal)

        row = {
            "endpoint": endpoint,
            "target": target,
            "pred_ideal": pideal,
            "Z3_from_X_setting_raw": float(raw_vals["Z3_from_X_setting"]),
            "Z3_from_Z_setting_raw": float(raw_vals["Z3"]),
            "Z3_cross_setting_diff_raw": float(
                raw_vals["Z3_from_X_setting"] - raw_vals["Z3"]
            ),
            "Z3_from_X_setting_m3": float(m3_vals["Z3_from_X_setting"]),
            "Z3_from_Z_setting_m3": float(m3_vals["Z3"]),
            "Z3_cross_setting_diff_m3": float(
                m3_vals["Z3_from_X_setting"] - m3_vals["Z3"]
            ),
        }
        for j, feature in enumerate(FEATURES):
            row[f"{feature}_ideal"] = float(ideal_vec[j])
            row[f"{feature}_raw"] = float(raw_vec[j])
            row[f"{feature}_m3"] = float(m3_vec[j])
            row[f"{feature}_raw_minus_ideal"] = float(raw_vec[j] - ideal_vec[j])
            row[f"{feature}_m3_minus_ideal"] = float(m3_vec[j] - ideal_vec[j])
        daily_rows.append(row)

    X_ideal = np.asarray(X_ideal, dtype=float)
    X_raw = np.asarray(X_raw, dtype=float)
    X_m3 = np.asarray(X_m3, dtype=float)
    targets = np.asarray(targets, dtype=float)
    pred_ideal = np.asarray(pred_ideal, dtype=float)

    pred_raw = base.common.predict_scaled_ridge(
        ideal["model"], ideal["scaler"], ideal["keep"], X_raw
    )
    pred_m3 = base.common.predict_scaled_ridge(
        ideal["model"], ideal["scaler"], ideal["keep"], X_m3
    )

    daily_df = pd.DataFrame(daily_rows)
    daily_df["pred_raw"] = pred_raw
    daily_df["pred_m3"] = pred_m3
    daily_df["abs_error_ideal"] = np.abs(targets - pred_ideal)
    daily_df["abs_error_raw"] = np.abs(targets - pred_raw)
    daily_df["abs_error_m3"] = np.abs(targets - pred_m3)
    daily_df.to_csv(
        RESULTS / f"{PREFIX}_features_predictions.csv", index=False
    )

    # =========================================================================
    # FEATURE DIAGNOSTICS
    # =========================================================================
    _, beta_raw, beta_audit = reconstruct_effective_raw_beta(X_ideal, pred_ideal)
    feature_diag, feature_improvement = make_feature_diagnostics(
        X_ideal, X_raw, X_m3, beta_raw
    )
    feature_diag.to_csv(
        RESULTS / f"{PREFIX}_feature_diagnostics.csv", index=False
    )
    feature_improvement.to_csv(
        RESULTS / f"{PREFIX}_feature_improvement.csv", index=False
    )

    raw_diff = X_raw - X_ideal
    m3_diff = X_m3 - X_ideal

    summary = {
        "step": "11.1E.10A",
        "candidate": "Candidate_1_H3_RWP_W4_R1_R3",
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
            f"P{q}": f for q, f in zip(m3_mapping, readout_fidelities)
        },
        "n_validation_endpoints": int(len(selected_endpoints)),
        "ideal_rmse": rmse(targets, pred_ideal),
        "raw_rmse": rmse(targets, pred_raw),
        "m3_rmse": rmse(targets, pred_m3),
        "m3_minus_raw_rmse": rmse(targets, pred_m3) - rmse(targets, pred_raw),
        "m3_rmse_improvement": rmse(targets, pred_raw) - rmse(targets, pred_m3),
        "ideal_mae": mae(targets, pred_ideal),
        "raw_mae": mae(targets, pred_raw),
        "m3_mae": mae(targets, pred_m3),
        "ideal_bias": bias(targets, pred_ideal),
        "raw_bias": bias(targets, pred_raw),
        "m3_bias": bias(targets, pred_m3),
        "raw_prediction_corr_vs_ideal": corr(pred_raw, pred_ideal),
        "m3_prediction_corr_vs_ideal": corr(pred_m3, pred_ideal),
        "raw_feature_mae": float(np.mean(np.abs(raw_diff))),
        "m3_feature_mae": float(np.mean(np.abs(m3_diff))),
        "raw_feature_rmse": float(np.sqrt(np.mean(raw_diff ** 2))),
        "m3_feature_rmse": float(np.sqrt(np.mean(m3_diff ** 2))),
        "z3_cross_setting_mae_raw": float(
            np.mean(np.abs(daily_df["Z3_cross_setting_diff_raw"]))
        ),
        "z3_cross_setting_mae_m3": float(
            np.mean(np.abs(daily_df["Z3_cross_setting_diff_m3"]))
        ),
        "raw_beta_reconstruction_audit_rmse": float(beta_audit),
        "feature_vector_n_cz": feature_cz,
        "max_setting_depth": max_depth,
        "feature_vector_duration_us": feature_duration_us,
        "service_usage_before": usage_before,
        "service_usage_after": get_service_usage(service),
        "main_job": main_job_record,
        "created_utc": utc_now_iso(),
    }

    (RESULTS / f"{PREFIX}_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    (RESULTS / f"{PREFIX}_job.json").write_text(
        json.dumps(json_safe(main_job_record), indent=2), encoding="utf-8"
    )

    print()
    print("=" * 132)
    print("CANDIDATE #1 — RAW VS M3 RESULTS")
    print("=" * 132)
    print(f"Ideal 2025 RMSE:                 {summary['ideal_rmse']:.6f}")
    print(f"RAW QPU RMSE:                    {summary['raw_rmse']:.6f}")
    print(f"M3 QPU RMSE:                     {summary['m3_rmse']:.6f}")
    print(f"M3 RMSE improvement vs RAW:      {summary['m3_rmse_improvement']:+.6f}")
    print()
    print(f"RAW QPU MAE:                     {summary['raw_mae']:.6f}")
    print(f"M3 QPU MAE:                      {summary['m3_mae']:.6f}")
    print()
    print(f"RAW QPU bias:                    {summary['raw_bias']:+.6f}")
    print(f"M3 QPU bias:                     {summary['m3_bias']:+.6f}")
    print()
    print(f"RAW/ideal prediction corr:       {summary['raw_prediction_corr_vs_ideal']:.6f}")
    print(f"M3/ideal prediction corr:        {summary['m3_prediction_corr_vs_ideal']:.6f}")
    print()
    print(f"RAW feature RMSE:                {summary['raw_feature_rmse']:.6f}")
    print(f"M3 feature RMSE:                 {summary['m3_feature_rmse']:.6f}")
    print()
    print("Feature-level RAW -> M3 changes:")
    cols = [
        "feature",
        "raw_feature_bias", "m3_feature_bias", "abs_bias_reduction",
        "raw_feature_rmse", "m3_feature_rmse", "feature_rmse_reduction_pct",
        "raw_corr", "m3_corr",
        "raw_noise_to_signal", "m3_noise_to_signal",
        "raw_pred_perturbation_rms_claims",
        "m3_pred_perturbation_rms_claims",
    ]
    print(feature_improvement[cols].to_string(index=False))
    print()
    print("2026 remains FROZEN / UNUSED.")
    print()
    print("Saved:")
    for suffix in [
        "preflight.json",
        "setting_resources.csv",
        "m3_calibrations.json",
        "m3_calibration_jobs.json",
        "raw_counts.csv",
        "m3_quasi_distributions.csv",
        "features_predictions.csv",
        "feature_diagnostics.csv",
        "feature_improvement.csv",
        "summary.json",
        "job.json",
    ]:
        print(f"  results/{PREFIX}_{suffix}")


if __name__ == "__main__":
    main()
