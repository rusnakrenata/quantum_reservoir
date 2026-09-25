from __future__ import annotations

"""
WEEK 11.1E.10B
Candidate #1 — IBM EstimatorV2 with built-in Runtime mitigation.

Scientific purpose
------------------
This is a clean alternative to the Candidate-1 Sampler+M3 experiment.
There is NO M3 and no quasi-probability reconstruction.

Frozen Candidate #1:
  H3 / RWP W=4 / r=3 / alpha=0.25
  readout = XZinj_dropX3_plus_YX45
  features = [X0,X1,X2,Z0,Z1,Z2,Z3,YX45]

Protocol:
  * 2022-2024: frozen training/readout only
  * 2025: full 365-day QPU validation
  * 2026: never loaded
  * ordinary Rule-5 fresh Kingston H3 layout selection
  * ordinary frozen Candidate-1 RWP circuit; no reset-aware changes
  * IBM Runtime EstimatorV2 returns the eight expectation values directly
  * default resilience_level=1 (IBM TREX measurement mitigation)
  * optional resilience_level=2 adds the Runtime's stronger mitigation stack

The exact logical candidate and ideal readout are reconstructed from:
    10_01_reference_embedding_refresh_and_simulation.py

Default is DRY RUN.  No QPU job is submitted without --submit.

Examples
--------
Dry run (recommended first):
    python 11_1E10B_candidate1_estimator.py

Submit with TREX / resilience level 1:
    python 11_1E10B_candidate1_estimator.py --submit --resilience-level 1 --shots 1024

Optional later test with resilience level 2:
    python 11_1E10B_candidate1_estimator.py --submit --resilience-level 2 --shots 1024
"""

import argparse
import importlib.util
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit_ibm_runtime import EstimatorV2
from qiskit_ibm_runtime.options import EstimatorOptions

from ibm_account import get_service


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

BASE_SCRIPT = HERE / "10_01_reference_embedding_refresh_and_simulation.py"
PREFIX = "11_1E10B_candidate1_estimator"

DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_SHOTS = 1024
DEFAULT_RESILIENCE_LEVEL = 1

OPT_LEVEL = 1
SEED_TRANSPILE = 42
SHORTLIST = 120
EXPECTED_N_2025 = 365

FEATURES = [
    "X0", "X1", "X2",
    "Z0", "Z1", "Z2", "Z3",
    "YX45",
]


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
# HELPERS
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
    """Recursively convert runtime/Qiskit objects to JSON-safe values.

    Qiskit Runtime option dataclasses contain ``UnsetType`` sentinel values.
    They are configuration placeholders, not scientific data, so serialize
    them as ``None`` rather than letting json.dumps fail.
    """
    if x is None or isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, np.generic):
        return json_safe(x.item())
    if isinstance(x, pd.Timestamp):
        return str(x)
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)

    # qiskit-ibm-runtime options use an UnsetType sentinel for values that
    # were intentionally left at Runtime defaults.  The exact import path
    # is version-dependent, so detect it by class name instead of importing
    # an internal Runtime symbol.
    if x.__class__.__name__ == "UnsetType":
        return None

    # Enums and enum-like configuration values.
    value = getattr(x, "value", None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if value is not None:
            return json_safe(value)

    # Last-resort audit-safe representation for future Runtime option types.
    try:
        json.dumps(x)
        return x
    except TypeError:
        return repr(x)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_service_usage(service):
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


def make_feature_diagnostics(X_ideal, X_qpu, stds, beta_raw):
    rows = []
    for j, feature in enumerate(FEATURES):
        ideal = X_ideal[:, j]
        qpu = X_qpu[:, j]
        delta = qpu - ideal
        sigma_ideal = float(np.std(ideal, ddof=1))
        f_bias = float(np.mean(delta))
        f_rmse = float(np.sqrt(np.mean(delta ** 2)))
        stochastic_rms = float(np.sqrt(max(f_rmse ** 2 - f_bias ** 2, 0.0)))
        est_std = np.asarray(stds[:, j], dtype=float)
        rows.append({
            "feature": feature,
            "beta": float(beta_raw[j]),
            "ideal_std": sigma_ideal,
            "feature_bias": f_bias,
            "feature_mae": float(np.mean(np.abs(delta))),
            "feature_rmse": f_rmse,
            "feature_stochastic_rms": stochastic_rms,
            "feature_corr_qpu_vs_ideal": corr(qpu, ideal),
            "qpu_std": float(np.std(qpu, ddof=1)),
            "noise_to_signal_ratio": f_rmse / sigma_ideal if sigma_ideal > 0 else np.nan,
            "direct_prediction_perturbation_rms_claims": abs(float(beta_raw[j])) * f_rmse,
            "systematic_prediction_shift_claims": float(beta_raw[j]) * f_bias,
            "mean_estimator_std": float(np.nanmean(est_std)),
            "max_estimator_std": float(np.nanmax(est_std)),
        })
    return pd.DataFrame(rows)


def options_to_dict(options):
    try:
        return json_safe(asdict(options))
    except Exception:
        try:
            return json_safe(dict(options))
        except Exception:
            return {"repr": repr(options)}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument(
        "--resilience-level",
        type=int,
        choices=[0, 1, 2],
        default=DEFAULT_RESILIENCE_LEVEL,
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit one full-2025 EstimatorV2 job.",
    )
    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError("--shots must be > 0.")

    print("=" * 132)
    print("WEEK 11.1E.10B — CANDIDATE #1 ESTIMATORV2 BUILT-IN MITIGATION TEST")
    print("=" * 132)
    print("2022-2024 = frozen training/readout")
    print("2025      = FULL 365-day EstimatorV2 hardware validation")
    print("2026      = FROZEN / NOT LOADED")
    print("NO M3; IBM Runtime built-in Estimator mitigation only")
    print()

    # -------------------------------------------------------------------------
    # Frozen Candidate #1 and ideal reference
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
    print(f"  features={FEATURES}")
    print(f"  Estimator default_shots={args.shots}")
    print(f"  resilience_level={args.resilience_level}")
    if args.resilience_level == 0:
        print("  mitigation=OFF")
    elif args.resilience_level == 1:
        print("  mitigation=IBM built-in measurement mitigation (TREX)")
    else:
        print("  mitigation=IBM resilience level 2 (TREX + ZNE + gate twirling per Runtime policy)")
    print(f"  lambda={ideal['metrics']['selected_lambda_recomputed']}")
    print(f"  training CV RMSE={ideal['metrics']['cv_rmse_recomputed']:.6f}")
    print(f"  ideal 2025 RMSE={ideal['metrics']['ideal_2025_rmse']:.6f}")
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
    # Build and transpile the unmeasured Candidate-1 core ONCE.
    # Estimator receives observables directly; no user-built measurement circuits.
    # -------------------------------------------------------------------------
    logical_core, core_params = base.build_parameterized_core_circuit(candidate)

    isa_core = transpile(
        logical_core,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
        scheduling_method="alap",
    )
    backend.check_faulty(isa_core)

    core_ops = {str(k): int(v) for k, v in isa_core.count_ops().items()}
    if int(core_ops.get("swap", 0)) != 0:
        raise RuntimeError("SWAP detected in Candidate #1 Estimator core.")

    mapped_observables = [
        base.logical_sparse_observable(feature).apply_layout(isa_core.layout)
        for feature in FEATURES
    ]

    observable_audit = {}
    for feature, obs in zip(FEATURES, mapped_observables):
        observable_audit[feature] = {
            "num_qubits": int(obs.num_qubits),
            "sparse_terms": [
                [
                    str(pauli),
                    [int(q) for q in qargs],
                    [float(np.real(coeff)), float(np.imag(coeff))],
                ]
                for pauli, qargs, coeff in obs.to_sparse_list()
            ],
        }

    try:
        core_duration_us = float(isa_core.estimate_duration(backend.target, unit="s")) * 1e6
    except Exception:
        core_duration_us = np.nan

    bound_cores = []
    print("Binding full 365-day Candidate #1 Estimator workload...")
    for k, endpoint in enumerate(selected_endpoints, 1):
        if k == 1 or k % 25 == 0 or k == len(selected_endpoints):
            print(
                f"  binding validation endpoint {k}/{len(selected_endpoints)} "
                f"(global endpoint={int(endpoint)})"
            )
        bound_cores.append(
            base.bind_endpoint_circuit(
                isa_core,
                core_params,
                ideal["angles"],
                int(endpoint),
            )
        )

    # -------------------------------------------------------------------------
    # Estimator options. Explicitly set resilience level and shot budget.
    # Level 1 is the recommended first test: TREX only, no M3.
    # -------------------------------------------------------------------------
    options = EstimatorOptions()
    options.default_shots = int(args.shots)
    options.resilience_level = int(args.resilience_level)

    estimator = EstimatorV2(mode=backend, options=options)
    estimator_options = options_to_dict(estimator.options)

    usage_before = get_service_usage(service)

    print()
    print("=" * 132)
    print("FULL-2025 CANDIDATE #1 ESTIMATOR PREFLIGHT")
    print("=" * 132)
    print(f"Backend:                          {args.backend}")
    print(f"Fresh layout:                     {layout}")
    print(f"Validation endpoints / PUBs:      {len(bound_cores)}")
    print(f"Observables per PUB:               {len(FEATURES)}")
    print(f"Estimator default_shots:           {args.shots}")
    print(f"Estimator resilience_level:        {args.resilience_level}")
    print(f"Core CZ count:                     {int(core_ops.get('cz', 0))}")
    print(f"Core depth:                        {int(isa_core.depth())}")
    print(f"Core generic reset count:          {int(core_ops.get('reset', 0))}")
    print(f"Core duration:                     {core_duration_us:.3f} us")
    print(f"Core SWAP count:                   {int(core_ops.get('swap', 0))}")
    print("M3:                                OFF / NOT USED")

    if usage_before is not None:
        print()
        print("Service usage before:")
        print(json.dumps(json_safe(usage_before), indent=2))

    preflight = {
        "step": "11.1E.10B",
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
        "n_endpoints": int(len(selected_endpoints)),
        "n_estimator_pubs": int(len(bound_cores)),
        "n_observables_per_pub": int(len(FEATURES)),
        "default_shots": int(args.shots),
        "resilience_level": int(args.resilience_level),
        "m3_used": False,
        "core_n_cz": int(core_ops.get("cz", 0)),
        "core_n_swap": int(core_ops.get("swap", 0)),
        "core_n_reset": int(core_ops.get("reset", 0)),
        "core_depth": int(isa_core.depth()),
        "core_duration_us": core_duration_us,
        "hardware_selection": json_safe(chosen.to_dict()),
        "mapped_observables": observable_audit,
        "estimator_options": estimator_options,
        "service_usage_before": usage_before,
        "created_utc": utc_now_iso(),
    }
    (RESULTS / f"{PREFIX}_preflight.json").write_text(
        json.dumps(json_safe(preflight), indent=2), encoding="utf-8"
    )

    if not args.submit:
        print()
        print("DRY RUN COMPLETE — NO QPU JOB SUBMITTED.")
        print()
        print("Recommended first hardware test (TREX only):")
        print(
            f"  python {Path(__file__).name} --submit --backend {args.backend} "
            f"--shots {args.shots} --resilience-level 1"
        )
        print()
        print("2026 remains FROZEN / UNUSED.")
        return

    # =========================================================================
    # SUBMIT ONE ESTIMATOR JOB
    # =========================================================================
    print()
    print("=" * 132)
    print("SUBMITTING FULL 365-DAY CANDIDATE #1 EstimatorV2 JOB")
    print("=" * 132)
    print(f"Submitting {len(bound_cores)} PUBs x {len(FEATURES)} observables...")
    print(f"resilience_level={args.resilience_level}, default_shots={args.shots}")
    print("M3 is NOT used.")

    pubs = [(circuit, mapped_observables) for circuit in bound_cores]
    job = estimator.run(pubs)
    job_id = job.job_id()
    print(f"Job ID: {job_id}")
    result = job.result()
    final_status = str(job.status())
    print(f"Final status: {final_status}")

    if len(result) != len(pubs):
        raise RuntimeError(
            f"Estimator returned {len(result)} PUB results for {len(pubs)} PUBs."
        )

    X_qpu = np.empty((len(result), len(FEATURES)), dtype=float)
    X_stds = np.full((len(result), len(FEATURES)), np.nan, dtype=float)
    result_metadata = []

    for i, pub_result in enumerate(result):
        evs = np.asarray(pub_result.data.evs, dtype=float).reshape(-1)
        if evs.size != len(FEATURES):
            raise RuntimeError(
                f"PUB {i}: Estimator returned {evs.size} expectation values; "
                f"expected {len(FEATURES)}."
            )
        X_qpu[i, :] = evs

        stds_obj = getattr(pub_result.data, "stds", None)
        if stds_obj is not None:
            stds = np.asarray(stds_obj, dtype=float).reshape(-1)
            if stds.size == len(FEATURES):
                X_stds[i, :] = stds

        result_metadata.append(json_safe(getattr(pub_result, "metadata", {})))

    # =========================================================================
    # FROZEN RIDGE / METRICS
    # =========================================================================
    X_ideal = np.asarray(ideal["X_val_ideal"], dtype=float)
    targets = np.asarray(ideal["y_val"], dtype=float)
    pred_ideal = np.asarray(ideal["pred_val_ideal"], dtype=float)

    if X_ideal.shape != X_qpu.shape:
        raise RuntimeError(
            f"Feature shape mismatch: ideal={X_ideal.shape}, qpu={X_qpu.shape}."
        )

    pred_qpu = base.common.predict_scaled_ridge(
        ideal["model"], ideal["scaler"], ideal["keep"], X_qpu
    )

    _, beta_raw, beta_audit = reconstruct_effective_raw_beta(X_ideal, pred_ideal)
    feature_diag = make_feature_diagnostics(X_ideal, X_qpu, X_stds, beta_raw)
    feature_diag.to_csv(
        RESULTS / f"{PREFIX}_feature_diagnostics.csv", index=False
    )

    rows = []
    for i, endpoint in enumerate(selected_endpoints):
        row = {
            "endpoint": int(endpoint),
            "target": float(targets[i]),
            "pred_ideal": float(pred_ideal[i]),
            "pred_estimator": float(pred_qpu[i]),
            "abs_error_ideal": float(abs(targets[i] - pred_ideal[i])),
            "abs_error_estimator": float(abs(targets[i] - pred_qpu[i])),
        }
        for j, feature in enumerate(FEATURES):
            row[f"{feature}_ideal"] = float(X_ideal[i, j])
            row[f"{feature}_estimator"] = float(X_qpu[i, j])
            row[f"{feature}_estimator_minus_ideal"] = float(X_qpu[i, j] - X_ideal[i, j])
            row[f"{feature}_estimator_std"] = float(X_stds[i, j]) if np.isfinite(X_stds[i, j]) else np.nan
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        RESULTS / f"{PREFIX}_features_predictions.csv", index=False
    )

    qpu_diff = X_qpu - X_ideal
    main_job_record = extract_job_record(job)

    summary = {
        "step": "11.1E.10B",
        "candidate": "Candidate_1_H3_RWP_W4_R1_R3",
        "backend": args.backend,
        "layout": layout,
        "workload_job_id": str(job_id),
        "workload_job_status": final_status,
        "primitive": "EstimatorV2",
        "default_shots": int(args.shots),
        "resilience_level": int(args.resilience_level),
        "m3_used": False,
        "n_validation_endpoints": int(len(selected_endpoints)),
        "ideal_rmse": rmse(targets, pred_ideal),
        "estimator_rmse": rmse(targets, pred_qpu),
        "ideal_mae": mae(targets, pred_ideal),
        "estimator_mae": mae(targets, pred_qpu),
        "ideal_bias": bias(targets, pred_ideal),
        "estimator_bias": bias(targets, pred_qpu),
        "prediction_corr_vs_ideal": corr(pred_qpu, pred_ideal),
        "feature_mae": float(np.mean(np.abs(qpu_diff))),
        "feature_rmse": float(np.sqrt(np.mean(qpu_diff ** 2))),
        "mean_estimator_std": float(np.nanmean(X_stds)) if np.isfinite(X_stds).any() else None,
        "max_estimator_std": float(np.nanmax(X_stds)) if np.isfinite(X_stds).any() else None,
        "raw_beta_reconstruction_audit_rmse": float(beta_audit),
        "core_n_cz": int(core_ops.get("cz", 0)),
        "core_n_swap": int(core_ops.get("swap", 0)),
        "core_n_reset": int(core_ops.get("reset", 0)),
        "core_depth": int(isa_core.depth()),
        "core_duration_us": core_duration_us,
        "estimator_options": estimator_options,
        "result_metadata": result_metadata,
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
    print("CANDIDATE #1 — ESTIMATORV2 RESULTS")
    print("=" * 132)
    print(f"Ideal 2025 RMSE:                 {summary['ideal_rmse']:.6f}")
    print(f"Estimator QPU RMSE:              {summary['estimator_rmse']:.6f}")
    print(f"Estimator QPU MAE:               {summary['estimator_mae']:.6f}")
    print(f"Estimator QPU bias:              {summary['estimator_bias']:+.6f}")
    print(f"Estimator/ideal pred corr:       {summary['prediction_corr_vs_ideal']:.6f}")
    print(f"Estimator feature RMSE:          {summary['feature_rmse']:.6f}")
    if summary["mean_estimator_std"] is not None:
        print(f"Mean Estimator reported std:     {summary['mean_estimator_std']:.6f}")
        print(f"Max Estimator reported std:      {summary['max_estimator_std']:.6f}")
    print()
    print("Feature diagnostics:")
    cols = [
        "feature",
        "feature_bias",
        "feature_rmse",
        "feature_corr_qpu_vs_ideal",
        "noise_to_signal_ratio",
        "direct_prediction_perturbation_rms_claims",
        "mean_estimator_std",
        "max_estimator_std",
    ]
    print(feature_diag[cols].to_string(index=False))
    print()
    print("2026 remains FROZEN / UNUSED.")
    print()
    print("Saved:")
    for suffix in [
        "preflight.json",
        "features_predictions.csv",
        "feature_diagnostics.csv",
        "summary.json",
        "job.json",
    ]:
        print(f"  results/{PREFIX}_{suffix}")


if __name__ == "__main__":
    main()
