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
* Within the prequalified Rule-5 layout pool, a small real-QPU reset probe
  measures simultaneous injection-qubit reset quality.
* Final layout selection is hierarchical, not a weighted score:
      existing Rule-5 prequalification
      -> minimum worst injection-qubit reset residual
      -> original selector rank as tie-break.
* The final workload uses IBM hardware-specific reset_2 when available and
  explicitly requested (default).

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
Stage 1 — reset-quality probe only:
    python 11_1E9A_candidate3_m3.py --probe-reset

Stage 2 — inspect full reset-aware dry-run:
    python 11_1E9A_candidate3_m3.py

Stage 3 — submit full 2025 after inspection:
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

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.transpiler.passes import (
    ConvertToMidCircuitResetAndMeasure,
)

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

# Reset-aware selection:
# probe only a small number of already-good hardware layouts selected by the
# existing Rule-5 logic, then use measured reset quality as the deciding
# criterion inside that prequalified set.
DEFAULT_RESET_LAYOUT_CANDIDATES = 8
DEFAULT_RESET_PROBE_SHOTS = 2048
DEFAULT_RESET_MODE = "reset_2"

RESET_PROBE_LAYOUTS_FILE = RESULTS / f"{PREFIX}_reset_probe_layouts.csv"
RESET_PROBE_QUBITS_FILE = RESULTS / f"{PREFIX}_reset_probe_qubits.csv"
RESET_PROBE_META_FILE = RESULTS / f"{PREFIX}_reset_probe_meta.json"
RESET_PROBE_JOB_FILE = RESULTS / f"{PREFIX}_reset_probe_job.json"

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
# RESET_2 + RESET-AWARE HARDWARE SELECTION HELPERS
# =============================================================================

def parse_layout_value(value):
    """Parse a layout stored as list/tuple/JSON/Python-list string."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [int(x) for x in value]

    if isinstance(value, str):
        s = value.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [int(x) for x in obj]
        except Exception:
            pass

        # Safe fallback for strings such as "[105, 117, ...]".
        import ast
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return [int(x) for x in obj]

    raise TypeError(f"Cannot parse layout from {value!r}")


def backend_calibration_signature(backend):
    """
    Return a stable calibration timestamp/signature when available.
    The reset probe is accepted for the main workload only if the backend
    calibration snapshot still matches.
    """
    try:
        props = backend.properties(refresh=True)
        return str(getattr(props, "last_update_date", None))
    except Exception:
        return None


def _collect_layout_rows(obj, rows, source="fresh"):
    """
    Recursively collect objects carrying a `layout` field from the selector
    return value. This deliberately does not assume one exact selector schema.
    """
    if isinstance(obj, pd.DataFrame):
        if "layout" in obj.columns:
            for i, row in obj.iterrows():
                rec = row.to_dict()
                rec["_source"] = source
                rec["_source_order"] = int(len(rows))
                rows.append(rec)
        return

    if isinstance(obj, pd.Series):
        if "layout" in obj.index:
            rec = obj.to_dict()
            rec["_source"] = source
            rec["_source_order"] = int(len(rows))
            rows.append(rec)
        return

    if isinstance(obj, dict):
        if "layout" in obj:
            rec = dict(obj)
            rec["_source"] = source
            rec["_source_order"] = int(len(rows))
            rows.append(rec)
        for key, value in obj.items():
            _collect_layout_rows(
                value,
                rows,
                source=f"{source}.{key}",
            )
        return

    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _collect_layout_rows(
                value,
                rows,
                source=f"{source}[{i}]",
            )


def candidate_layout_pool_from_selector(
    fresh,
    selector_prefix,
    top_k,
):
    """
    Build the reset-probe pool from the EXACT current Rule-5 ranking.

    IMPORTANT:
    Do not recursively scrape layouts from the selector return object or from
    CSV artifacts.  `fresh["compiled"]` is the authoritative current
    circuit-aware table.  We:
        1. retain strict zero-SWAP compilations,
        2. apply the exact Rule-5 hierarchy already used in Week 10,
        3. verify rank 1 equals fresh["selected"],
        4. take only the top-k rows for reset probing.

    Exact Rule-5 hierarchy:
        compiled max 2Q error  ↓
        max readout error      ↓
        compiled max 1Q error  ↓
        min T2                 ↑
        min T1                 ↑
        compiled duration      ↓

    No weighted score is introduced.
    """
    if "compiled" not in fresh:
        raise RuntimeError(
            "Fresh selector output does not contain authoritative "
            "`compiled` dataframe."
        )

    compiled = fresh["compiled"].copy()

    if "strict_compile_pass" not in compiled.columns:
        raise RuntimeError(
            "Fresh compiled dataframe has no strict_compile_pass column."
        )

    good = compiled[
        compiled["strict_compile_pass"].astype(bool)
    ].copy()

    if good.empty:
        raise RuntimeError(
            "No strict zero-SWAP embeddings survived fresh compilation."
        )

    rule5_cols = [
        "compiled_2q_error_max",
        "max_readout_error",
        "compiled_1q_error_max",
        "min_t2_us",
        "min_t1_us",
        "compiled_duration_us",
    ]

    missing = [
        c for c in rule5_cols
        if c not in good.columns
    ]
    if missing:
        raise RuntimeError(
            "Cannot reconstruct exact Rule-5 ranking. Missing columns: "
            + ", ".join(missing)
        )

    ranked = good.sort_values(
        rule5_cols,
        ascending=[True, True, True, False, False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    ranked["selector_rank"] = np.arange(1, len(ranked) + 1)

    # ------------------------------------------------------------------
    # Hard consistency audit:
    # our reconstructed rank 1 MUST be the selector's selected layout.
    # ------------------------------------------------------------------
    if "selected" not in fresh:
        raise RuntimeError(
            "Fresh selector output does not contain `selected` row."
        )

    selected_layout = parse_layout_value(
        fresh["selected"]["layout"]
    )
    reconstructed_rank1 = parse_layout_value(
        ranked.iloc[0]["layout"]
    )

    if tuple(selected_layout) != tuple(reconstructed_rank1):
        raise RuntimeError(
            "Rule-5 ranking reconstruction mismatch. "
            f"fresh['selected']={selected_layout}, "
            f"reconstructed rank1={reconstructed_rank1}. "
            "Abort rather than probing the wrong physical regions."
        )

    # Add percent columns if the selector table does not already contain
    # them, purely for readable console output.
    if (
        "compiled_2q_error_max_percent" not in ranked.columns
        and "compiled_2q_error_max" in ranked.columns
    ):
        ranked["compiled_2q_error_max_percent"] = (
            100.0 * pd.to_numeric(
                ranked["compiled_2q_error_max"],
                errors="coerce",
            )
        )

    if (
        "compiled_1q_error_max_percent" not in ranked.columns
        and "compiled_1q_error_max" in ranked.columns
    ):
        ranked["compiled_1q_error_max_percent"] = (
            100.0 * pd.to_numeric(
                ranked["compiled_1q_error_max"],
                errors="coerce",
            )
        )

    if (
        "max_readout_error_percent" not in ranked.columns
        and "max_readout_error" in ranked.columns
    ):
        ranked["max_readout_error_percent"] = (
            100.0 * pd.to_numeric(
                ranked["max_readout_error"],
                errors="coerce",
            )
        )

    pool = ranked.head(int(top_k)).copy()

    # Normalize layout storage only after all ranking/audits are complete.
    pool["layout"] = pool["layout"].map(
        lambda x: json.dumps(parse_layout_value(x))
    )

    return pool


def reset2_supported_on_layout(backend, layout):
    """
    Candidate #3 resets logical q0..q3 only.
    Require reset_2 on all four mapped physical injection qubits.
    """
    if "reset_2" not in set(backend.target.operation_names):
        return False

    for p in layout[:4]:
        try:
            if not backend.target.instruction_supported(
                "reset_2",
                (int(p),),
            ):
                return False
        except Exception:
            return False

    return True


def convert_mid_circuit_resets(
    circuit,
    backend,
    reset_mode,
):
    """
    Convert generic non-terminal reset -> hardware-specific reset_2 after
    routing. Terminal measurement remains ordinary `measure`.

    IBM Runtime 0.49 introduced MidCircuitReset and
    ConvertToMidCircuitResetAndMeasure.  We use:
        mcm_name="measure"  -> do not alter terminal measurement
        mcr_name="reset_2"  -> replace only non-terminal resets
    """
    if reset_mode == "reset":
        return circuit

    if reset_mode != "reset_2":
        raise ValueError(f"Unknown reset mode: {reset_mode}")

    if "reset_2" not in set(backend.target.operation_names):
        raise RuntimeError(
            f"{backend.name} does not currently expose reset_2 in its target. "
            "Re-run with --reset-mode reset only if you intentionally want "
            "the legacy generic reset implementation."
        )

    pm = PassManager([
        ConvertToMidCircuitResetAndMeasure(
            target=backend.target,
            mcm_name="measure",
            mcr_name="reset_2",
        )
    ])

    out = pm.run(circuit)
    return out


def compile_candidate3_circuit(
    logical,
    backend,
    layout,
    reset_mode,
    expected_mid_resets=4,
):
    """
    Strict no-routing Candidate #3 compilation with post-routing reset_2
    conversion, followed by ALAP scheduling.
    """
    isa = transpile(
        logical,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
    )

    isa = convert_mid_circuit_resets(
        isa,
        backend,
        reset_mode,
    )

    # Re-run only target validation/scheduling on the already-physical circuit.
    isa = transpile(
        isa,
        backend=backend,
        routing_method="none",
        optimization_level=0,
        seed_transpiler=SEED_TRANSPILE,
        scheduling_method="alap",
    )

    backend.check_faulty(isa)

    ops = {
        str(k): int(v)
        for k, v in isa.count_ops().items()
    }

    if int(ops.get("swap", 0)) != 0:
        raise RuntimeError("Strict native compile failed: SWAP detected.")

    if reset_mode == "reset_2":
        n_generic = int(ops.get("reset", 0))
        n_reset2 = int(ops.get("reset_2", 0))

        if n_generic != 0:
            raise RuntimeError(
                f"reset_2 conversion incomplete: {n_generic} generic "
                "reset instructions remain."
            )

        if n_reset2 != int(expected_mid_resets):
            raise RuntimeError(
                f"Expected {expected_mid_resets} reset_2 instructions, "
                f"found {n_reset2}."
            )

    return isa


def bit_from_qiskit_string(bitstring, c_index):
    s = str(bitstring).replace(" ", "")
    return int(s[-1 - int(c_index)])


def p_one_from_counts(counts, c_index):
    total = float(sum(counts.values()))
    if total <= 0:
        raise RuntimeError("Empty counts in reset probe.")

    ones = 0.0
    for bitstring, n in counts.items():
        if bit_from_qiskit_string(bitstring, c_index) == 1:
            ones += float(n)

    return ones / total


def corrected_reset_residual(p_obs, e0, e1):
    """
    Same-job readout correction:
        p_obs = e0 + (1-e0-e1) p_reset
    """
    denom = 1.0 - float(e0) - float(e1)

    if abs(denom) < 1e-12:
        return np.nan

    return float(
        np.clip(
            (float(p_obs) - float(e0)) / denom,
            0.0,
            1.0,
        )
    )


def build_single_readout_control(prep_one):
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "m")
    qc = QuantumCircuit(q, c)

    if prep_one:
        qc.x(0)

    qc.measure(0, 0)
    return qc


def build_parallel_reset_probe():
    """
    Mimic Candidate #3's simultaneous q0..q3 reset:
        |1111> -> reset each injection qubit -> terminal measure.
    """
    q = QuantumRegister(4, "q")
    c = ClassicalRegister(4, "m")
    qc = QuantumCircuit(q, c)

    for i in range(4):
        qc.x(i)

    for i in range(4):
        qc.reset(i)

    for i in range(4):
        qc.measure(i, i)

    return qc


def compile_reset_probe_circuit(
    logical,
    backend,
    initial_layout,
    reset_mode,
    expected_mid_resets,
):
    isa = transpile(
        logical,
        backend=backend,
        initial_layout=initial_layout,
        routing_method="none",
        optimization_level=0,
        seed_transpiler=SEED_TRANSPILE,
    )

    if expected_mid_resets:
        isa = convert_mid_circuit_resets(
            isa,
            backend,
            reset_mode,
        )

    isa = transpile(
        isa,
        backend=backend,
        routing_method="none",
        optimization_level=0,
        seed_transpiler=SEED_TRANSPILE,
        scheduling_method="alap",
    )

    backend.check_faulty(isa)

    ops = {
        str(k): int(v)
        for k, v in isa.count_ops().items()
    }

    if int(ops.get("swap", 0)) != 0:
        raise RuntimeError("SWAP detected in reset probe.")

    if reset_mode == "reset_2" and expected_mid_resets:
        if int(ops.get("reset", 0)) != 0:
            raise RuntimeError(
                "Generic reset remained in reset_2 probe circuit."
            )
        if int(ops.get("reset_2", 0)) != int(expected_mid_resets):
            raise RuntimeError(
                "Unexpected reset_2 count in reset probe: "
                f"{ops.get('reset_2', 0)}"
            )

    return isa


def run_reset_probe(
    backend,
    candidate_pool,
    shots,
    reset_mode,
):
    """
    Measure reset quality for the injection qubits of the prequalified layouts.

    Cost-efficient paired design:
      * RO0 and RO1 once per unique physical injection qubit.
      * one simultaneous |1111> -> reset(q0..q3) circuit per candidate layout.

    The selection metric is the maximum same-job readout-corrected residual
    excitation among the four injection qubits.
    """
    if reset_mode == "reset_2":
        candidate_pool = candidate_pool[
            candidate_pool["reset2_supported"]
        ].copy()

    if candidate_pool.empty:
        raise RuntimeError(
            "No prequalified layout supports the requested reset mode."
        )

    layouts = [
        parse_layout_value(x)
        for x in candidate_pool["layout"]
    ]

    unique_injection_qubits = sorted({
        int(p)
        for layout in layouts
        for p in layout[:4]
    })

    circuits = []
    meta = []

    # Same-job readout controls.
    for p in unique_injection_qubits:
        for prep_one in [False, True]:
            logical = build_single_readout_control(prep_one)
            isa = compile_reset_probe_circuit(
                logical,
                backend,
                initial_layout=[p],
                reset_mode=reset_mode,
                expected_mid_resets=0,
            )
            circuits.append(isa)
            meta.append({
                "kind": "RO1" if prep_one else "RO0",
                "physical_qubit": int(p),
            })

    # One simultaneous reset probe per candidate layout.
    for _, row in candidate_pool.iterrows():
        layout = parse_layout_value(row["layout"])
        logical = build_parallel_reset_probe()

        isa = compile_reset_probe_circuit(
            logical,
            backend,
            initial_layout=layout[:4],
            reset_mode=reset_mode,
            expected_mid_resets=4,
        )

        circuits.append(isa)
        meta.append({
            "kind": "PAR_RESET1",
            "layout": json.dumps(layout),
            "selector_rank": int(row["selector_rank"]),
        })

    print()
    print("=" * 132)
    print("RESET-AWARE LAYOUT PROBE")
    print("=" * 132)
    print(f"Candidate layouts probed:          {len(candidate_pool)}")
    print(f"Unique injection qubits:           {len(unique_injection_qubits)}")
    print(f"Probe circuits:                    {len(circuits)}")
    print(f"Probe shots/circuit:               {shots}")
    print(f"Reset implementation:              {reset_mode}")
    print()

    sampler = SamplerV2(mode=backend)
    job = sampler.run(
        circuits,
        shots=int(shots),
    )

    print(f"Reset probe job ID: {job.job_id()}")
    result = job.result()
    print(f"Reset probe status: {job.status()}")

    if len(result) != len(circuits):
        raise RuntimeError(
            f"Reset probe returned {len(result)} results for "
            f"{len(circuits)} circuits."
        )

    controls = {}
    parallel_counts = {}

    for pub, m in zip(result, meta):
        counts = base.get_pub_counts(pub)

        if m["kind"] in {"RO0", "RO1"}:
            p = int(m["physical_qubit"])
            controls[(p, m["kind"])] = p_one_from_counts(
                counts,
                0,
            )
        else:
            parallel_counts[
                (m["layout"], int(m["selector_rank"]))
            ] = counts

    qubit_rows = []
    layout_rows = []

    for _, row in candidate_pool.iterrows():
        layout = parse_layout_value(row["layout"])
        key = (json.dumps(layout), int(row["selector_rank"]))
        counts = parallel_counts[key]

        corrected = []

        for c_index, p in enumerate(layout[:4]):
            p = int(p)
            e0 = float(controls[(p, "RO0")])
            e1 = float(1.0 - controls[(p, "RO1")])
            p_obs = p_one_from_counts(
                counts,
                c_index,
            )
            p_corr = corrected_reset_residual(
                p_obs,
                e0,
                e1,
            )
            corrected.append(p_corr)

            qubit_rows.append({
                "selector_rank": int(row["selector_rank"]),
                "layout": json.dumps(layout),
                "logical_injection_qubit": int(c_index),
                "physical_qubit": p,
                "readout_e0_P1_given_0": e0,
                "readout_e1_P0_given_1": e1,
                "observed_P1_after_parallel_reset": p_obs,
                "corrected_reset_residual": p_corr,
                "reset_mode": reset_mode,
                "shots": int(shots),
            })

        out = row.to_dict()
        out["layout"] = json.dumps(layout)
        out["reset_mode"] = reset_mode
        out["reset_probe_shots"] = int(shots)
        out["max_reset_error"] = float(np.nanmax(corrected))
        out["mean_reset_error"] = float(np.nanmean(corrected))

        for q, val in enumerate(corrected):
            out[f"q{q}_reset_error"] = float(val)

        layout_rows.append(out)

    qubit_df = pd.DataFrame(qubit_rows)
    layout_df = pd.DataFrame(layout_rows)

    # Transparent hierarchical selection:
    #   1) candidates are already prequalified by the existing Rule-5 selector;
    #   2) minimize the WORST injection-qubit reset error;
    #   3) if equal, preserve original selector preference.
    layout_df = layout_df.sort_values(
        ["max_reset_error", "selector_rank"],
        kind="stable",
    ).reset_index(drop=True)

    return {
        "job": job,
        "qubits": qubit_df,
        "layouts": layout_df,
    }


def save_reset_probe(
    probe,
    backend,
    backend_name,
    reset_mode,
    shots,
):
    probe["qubits"].to_csv(
        RESET_PROBE_QUBITS_FILE,
        index=False,
    )
    probe["layouts"].to_csv(
        RESET_PROBE_LAYOUTS_FILE,
        index=False,
    )

    meta = {
        "backend": backend_name,
        "backend_last_update_date": backend_calibration_signature(backend),
        "reset_mode": reset_mode,
        "shots": int(shots),
        "created_utc": utc_now_iso(),
        "job": extract_job_record(probe["job"]),
    }

    RESET_PROBE_META_FILE.write_text(
        json.dumps(json_safe(meta), indent=2),
        encoding="utf-8",
    )

    RESET_PROBE_JOB_FILE.write_text(
        json.dumps(
            json_safe(extract_job_record(probe["job"])),
            indent=2,
        ),
        encoding="utf-8",
    )

    return meta


def load_valid_reset_probe(
    backend,
    backend_name,
    reset_mode,
):
    if not RESET_PROBE_LAYOUTS_FILE.exists():
        return None, "reset probe layout file is missing"

    if not RESET_PROBE_META_FILE.exists():
        return None, "reset probe metadata file is missing"

    meta = json.loads(
        RESET_PROBE_META_FILE.read_text(encoding="utf-8")
    )

    if str(meta.get("backend")) != str(backend_name):
        return None, "reset probe was produced on a different backend"

    if str(meta.get("reset_mode")) != str(reset_mode):
        return None, "reset probe used a different reset implementation"

    current_sig = backend_calibration_signature(backend)
    probe_sig = meta.get("backend_last_update_date")

    if (
        current_sig is not None
        and probe_sig is not None
        and str(current_sig) != str(probe_sig)
    ):
        return None, (
            "backend calibration changed since the reset probe "
            f"({probe_sig} -> {current_sig})"
        )

    df = pd.read_csv(RESET_PROBE_LAYOUTS_FILE)
    if df.empty:
        return None, "reset probe contains no layout rows"

    return df, None


def choose_reset_aware_layout(
    candidate_pool,
    reset_probe_df,
):
    """
    Keep only layouts that are still in the current fresh selector pool, then
    choose minimum worst reset error; original selector rank is the tie-break.
    """
    current = {
        tuple(parse_layout_value(x))
        for x in candidate_pool["layout"]
    }

    probe = reset_probe_df.copy()
    probe["_layout_tuple"] = probe["layout"].map(
        lambda x: tuple(parse_layout_value(x))
    )
    probe = probe[
        probe["_layout_tuple"].isin(current)
    ].copy()

    if probe.empty:
        raise RuntimeError(
            "Reset probe layouts no longer overlap the current fresh "
            "hardware-selector candidate pool. Re-run --probe-reset."
        )

    probe = probe.sort_values(
        ["max_reset_error", "selector_rank"],
        kind="stable",
    ).reset_index(drop=True)

    best = probe.iloc[0].copy()
    return best, probe


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

    parser.add_argument(
        "--probe-reset",
        action="store_true",
        help=(
            "Submit ONLY the reset-quality probe for the top fresh H2 "
            "layouts, save the reset-aware ranking, then stop. "
            "No M3 or 365-day workload is submitted."
        ),
    )
    parser.add_argument(
        "--reset-probe-shots",
        type=int,
        default=DEFAULT_RESET_PROBE_SHOTS,
        help="Shots per circuit in the reset-quality layout probe.",
    )
    parser.add_argument(
        "--reset-layout-candidates",
        type=int,
        default=DEFAULT_RESET_LAYOUT_CANDIDATES,
        help=(
            "Number of already-prequalified Rule-5 layouts whose "
            "injection-qubit reset quality is measured."
        ),
    )
    parser.add_argument(
        "--reset-mode",
        choices=["reset_2", "reset"],
        default=DEFAULT_RESET_MODE,
        help=(
            "Mid-circuit reset implementation. Default reset_2 uses IBM's "
            "hardware-specific MidCircuitReset path."
        ),
    )

    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError("--shots must be > 0.")
    if args.m3_cal_shots <= 0:
        raise ValueError("--m3-cal-shots must be > 0.")
    if args.reset_probe_shots <= 0:
        raise ValueError("--reset-probe-shots must be > 0.")
    if args.reset_layout_candidates <= 0:
        raise ValueError("--reset-layout-candidates must be > 0.")
    if args.probe_reset and args.submit:
        raise ValueError(
            "Use --probe-reset first, inspect the reset-aware layout, then "
            "run the normal dry-run / --submit workflow separately."
        )

    print("=" * 132)
    print("WEEK 11.1E.9A — CANDIDATE #3 M3 + RESET_2 RESET-AWARE TEST (v2)")
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

    backend = service.backend(
        args.backend,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    selector_prefix = f"{PREFIX}_{args.backend}"

    candidate_pool = candidate_layout_pool_from_selector(
        fresh=fresh,
        selector_prefix=selector_prefix,
        top_k=int(args.reset_layout_candidates),
    )

    candidate_pool["reset2_supported"] = candidate_pool["layout"].map(
        lambda x: reset2_supported_on_layout(
            backend,
            parse_layout_value(x),
        )
    )

    print()
    print(
        "Rule-5 rank-1 consistency audit: PASS — "
        f"{parse_layout_value(candidate_pool.iloc[0]['layout'])} "
        "matches fresh['selected']."
    )

    print()
    print("Reset-aware prequalified layout pool:")
    show_cols = [
        c for c in [
            "selector_rank",
            "layout",
            "compiled_2q_error_max_percent",
            "max_readout_error_percent",
            "min_t2_us",
            "reset2_supported",
        ]
        if c in candidate_pool.columns
    ]
    print(candidate_pool[show_cols].to_string(index=False))

    if args.reset_mode == "reset_2":
        n_reset2 = int(candidate_pool["reset2_supported"].sum())
        print()
        print(
            f"Layouts supporting reset_2 on all four injection qubits: "
            f"{n_reset2}/{len(candidate_pool)}"
        )
        if n_reset2 == 0:
            raise RuntimeError(
                f"{args.backend} currently exposes no probed H2 layout with "
                "reset_2 support on all four Candidate #3 injection qubits."
            )

    # ---------------------------------------------------------------------
    # Optional reset-probe stage: this is intentionally a separate QPU job.
    # It measures reset quality first, saves the ranking, and STOPS so the
    # selected layout can be inspected before spending the full workload.
    # ---------------------------------------------------------------------
    if args.probe_reset:
        probe_pool = candidate_pool.copy()

        if args.reset_mode == "reset_2":
            probe_pool = probe_pool[
                probe_pool["reset2_supported"]
            ].copy()

        probe = run_reset_probe(
            backend=backend,
            candidate_pool=probe_pool,
            shots=int(args.reset_probe_shots),
            reset_mode=args.reset_mode,
        )

        probe_meta = save_reset_probe(
            probe=probe,
            backend=backend,
            backend_name=args.backend,
            reset_mode=args.reset_mode,
            shots=int(args.reset_probe_shots),
        )

        print()
        print("=" * 132)
        print("RESET-AWARE H2 LAYOUT RANKING")
        print("=" * 132)

        ranking = probe["layouts"].copy()
        print_cols = [
            c for c in [
                "selector_rank",
                "layout",
                "max_reset_error",
                "mean_reset_error",
                "q0_reset_error",
                "q1_reset_error",
                "q2_reset_error",
                "q3_reset_error",
                "compiled_2q_error_max_percent",
                "max_readout_error_percent",
                "min_t2_us",
            ]
            if c in ranking.columns
        ]

        print(
            ranking[print_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        best = ranking.iloc[0]
        print()
        print("RESET-AWARE SELECTED LAYOUT:")
        print(f"  layout={best['layout']}")
        print(
            f"  worst injection reset error="
            f"{float(best['max_reset_error']):.6f}"
        )
        print(
            f"  original selector rank="
            f"{int(best['selector_rank'])}"
        )
        print()
        print("Probe saved. No M3 calibration or full workload was submitted.")
        print(
            "Now rerun this script WITHOUT --probe-reset to inspect the "
            "full Candidate #3 + reset_2 + M3 preflight."
        )
        print("2026 remains FROZEN / UNUSED.")
        return

    # ---------------------------------------------------------------------
    # Normal dry-run / workload submission requires a reset probe from the
    # SAME backend calibration snapshot.
    # ---------------------------------------------------------------------
    reset_probe_df, reset_probe_error = load_valid_reset_probe(
        backend=backend,
        backend_name=args.backend,
        reset_mode=args.reset_mode,
    )

    if reset_probe_df is None:
        print()
        print("=" * 132)
        print("RESET PROBE REQUIRED BEFORE FINAL LAYOUT SELECTION")
        print("=" * 132)
        print(f"Reason: {reset_probe_error}")
        print()
        print("Run:")
        print(
            f"  python {Path(__file__).name} --probe-reset "
            f"--backend {args.backend} "
            f"--reset-mode {args.reset_mode} "
            f"--reset-probe-shots {args.reset_probe_shots} "
            f"--reset-layout-candidates {args.reset_layout_candidates}"
        )
        print()
        print(
            "This submits only the small reset-quality probe and then stops."
        )
        print("2026 remains FROZEN / UNUSED.")
        return

    chosen, reset_ranking = choose_reset_aware_layout(
        candidate_pool=candidate_pool,
        reset_probe_df=reset_probe_df,
    )

    layout = parse_layout_value(chosen["layout"])

    print()
    print("Fresh + reset-aware physical choice:")
    print(f"  backend={args.backend}")
    print(f"  layout={layout}")
    print(f"  reset implementation={args.reset_mode}")
    print(
        f"  worst measured injection reset error="
        f"{float(chosen['max_reset_error']):.6f}"
    )
    print(
        f"  mean measured injection reset error="
        f"{float(chosen['mean_reset_error']):.6f}"
    )
    print(
        f"  original Rule-5 selector rank="
        f"{int(chosen['selector_rank'])}"
    )

    if "compiled_2q_error_max_percent" in chosen:
        print(
            f"  max CZ error="
            f"{float(chosen['compiled_2q_error_max_percent']):.6f}%"
        )
    if "max_readout_error_percent" in chosen:
        print(
            f"  max readout error="
            f"{float(chosen['max_readout_error_percent']):.6f}%"
        )
    if "min_t1_us" in chosen:
        print(
            f"  min T1={float(chosen['min_t1_us']):.3f} us"
        )
    if "min_t2_us" in chosen:
        print(
            f"  min T2={float(chosen['min_t2_us']):.3f} us"
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

            isa = compile_candidate3_circuit(
                logical=logical,
                backend=backend,
                layout=layout,
                reset_mode=args.reset_mode,
                expected_mid_resets=4,
            )

            rr = base.resource_row(
                isa,
                backend,
                endpoint=int(endpoint),
                setting=setting[0],
            )

            ops_now = {
                str(k): int(v)
                for k, v in isa.count_ops().items()
            }
            rr["n_reset_2"] = int(ops_now.get("reset_2", 0))
            rr["n_reset_generic"] = int(ops_now.get("reset", 0))
            rr["reset_mode"] = args.reset_mode

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

    expected_m3_mapping = [int(x) for x in layout[:4]]
    if m3_mapping != expected_m3_mapping:
        raise RuntimeError(
            "Final measurement mapping changed unexpectedly after reset_2 "
            f"conversion/scheduling. Expected {expected_m3_mapping}, "
            f"got {m3_mapping}."
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
            reset2_feature_vector=("n_reset_2", "sum"),
            generic_reset_feature_vector=("n_reset_generic", "sum"),
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
        f"Median reset_2/feature vector:    "
        f"{feature_resources['reset2_feature_vector'].median():.0f}"
    )
    print(
        f"Median generic reset/feature:     "
        f"{feature_resources['generic_reset_feature_vector'].median():.0f}"
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
        "reset_mode": args.reset_mode,
        "reset_aware_selection": {
            "selector_rank": int(chosen["selector_rank"]),
            "max_reset_error": float(chosen["max_reset_error"]),
            "mean_reset_error": float(chosen["mean_reset_error"]),
            "q0_reset_error": float(chosen["q0_reset_error"]),
            "q1_reset_error": float(chosen["q1_reset_error"]),
            "q2_reset_error": float(chosen["q2_reset_error"]),
            "q3_reset_error": float(chosen["q3_reset_error"]),
            "probe_file": str(RESET_PROBE_LAYOUTS_FILE),
        },
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
            f"--m3-cal-shots {args.m3_cal_shots} "
            f"--reset-mode {args.reset_mode}"
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
        "reset_mode": args.reset_mode,
        "reset_aware_selection": {
            "selector_rank": int(chosen["selector_rank"]),
            "max_reset_error": float(chosen["max_reset_error"]),
            "mean_reset_error": float(chosen["mean_reset_error"]),
            "q0_reset_error": float(chosen["q0_reset_error"]),
            "q1_reset_error": float(chosen["q1_reset_error"]),
            "q2_reset_error": float(chosen["q2_reset_error"]),
            "q3_reset_error": float(chosen["q3_reset_error"]),
        },
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
