#!/usr/bin/env python
"""
10.06B — H4 COMPLETE WEEK-10 NOISY BACKFILL (v2)
================================================

Purpose
-------
Close the IBM Week-10 H4 completeness gap in ONE script after the successful
10.06A handoff audit.

This script intentionally does NOT rerun the frozen H0-H3 Week-10 experiment.
It runs only the missing H4 candidate-wide hardware-noise work, then (when the
historical Week-10 files are present) merges H4 with the frozen H0-H3 results
and recomputes the expanded Pareto frontier.

Scientific scope
----------------
The historical Week-10.1--10.4 reference-candidate experiments established the
candidate-wide protocol.  They are therefore not repeated as a new H4
hyperparameter search.  Instead this script performs all missing H4 Week-10
work needed for a scientifically symmetric candidate-wide backfill:

  B0. Audit/freeze the successful 10.06A artifacts.
  B1. Reconstruct all 8 H4 Rule-1--Rule-4 candidates and freeze their
      2022-2024 ideal Ridge readouts independently at r={2,3,4}.
  B2. Strictly transpile the H4 cores onto the frozen shared Kingston layout,
      compact the exact transpiled six-qubit workload, and perform an A/B
      noiseless feature-equivalence audit before any noisy result is accepted.
  B3. N5: full backend-calibrated quantum noise, direct expectation values,
      readout assignment error OFF, for all 8 candidates × r={2,3,4}.
  B4. S2: finite-shot estimates of the SAME N5 noisy states, readout assignment
      error OFF, at the historical candidate-wide operating cells

          (r, shots) in {
              (2,1024),
              (3,512),
              (3,1024),
              (3,2048),
              (4,512)
          }

      with 3 independent simulator repetitions.  The r=3 shot budgets are
      nested prefixes of the same 2048-shot realization within each replicate.
  B5. H4-only direct/sampled summaries and Pareto frontier.
  B6. If the frozen historical Week-10.5 H0-H3 result files are present,
      merge (do not rerun) them with H4 and recompute the expanded Pareto
      frontier.
  B7. Write a reproducibility manifest and a LaTeX-ready result snippet.

Critical experimental rules
----------------------------
* 2022-2024: model/readout fitting only.
* 2025: diagnostic hardware-transfer validation.
* 2026: NEVER LOADED.
* No weighted score.
* No QPU jobs.
* No readout-error mitigation, ZNE, PEC, or postselection.
* H0-H3 are never simulated again here.
* The 10.06A shared H4 layout is frozen and reused.
* CONT memory starts in |00><00| and is propagated chronologically.
* Injection qubits are reset/re-encoded while q4,q5 carry the memory state.
* S2 samples the N5 noisy density matrix directly, so N5→S2 differs only by
  finite-shot estimation, exactly matching the Week-10 candidate-wide
  definition (readout assignment error remains OFF).

Why compact six-qubit Aer is used
---------------------------------
The circuit is first transpiled against the REAL IBM backend with the frozen
physical layout and routing_method="none".  The exact native operations on the
six selected physical qubits are then relabelled to compact indices 0..5.
The current backend-derived quantum errors for those physical qubits/gates are
remapped to the compact indices.  This preserves the actual compiled gate
sequence while making density-matrix propagation practical.

For CONT, this compact representation is essential: after each chronological
input, the reduced q4,q5 density matrix is carried to the next input.  This
avoids the incorrect I/4 initialization used in an earlier historical draft;
the memory state is initialized to |00><00|.

Required project files beside this script
-----------------------------------------
09_04_rwp_common.py
09_05c_rwp_resource_and_hardware_gapfill.py
ibm_account.py

Required results
----------------
results/09_05d_h4_week10_additions.csv

results/10_06a_h4_current_kingston_layout.csv
results/10_06a_h4_ideal_depth_audit.csv
results/10_06a_h4_resource_compile_detail.csv
results/10_06a_h4_resource_summary.csv
results/10_06a_h4_manifest.json

Optional historical Week-10 files for the final H0-H4 merge
-----------------------------------------------------------
results/10_05_direct_runs.csv
results/10_05_sampled_summary.csv
results/10_05_pareto_front.csv
results/10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv

Outputs
-------
results/10_06b_h4_ab_audit.csv
results/10_06b_h4_direct_runs.csv
results/10_06b_h4_n5_features_predictions.csv
results/10_06b_h4_sampled_runs.csv
results/10_06b_h4_sampled_summary.csv
results/10_06b_h4_candidate_summary.csv
results/10_06b_h4_pareto_front.csv
results/10_06b_h4_expanded_sampled_summary.csv          (if historical 10.05 exists)
results/10_06b_h4_expanded_pareto_front.csv             (if historical 10.05 exists)
results/10_06b_h4_manifest.json
results/10_06b_h4_week10_latex_snippet.tex

Typical full run
----------------
python 10_06b_h4_week10_all_noisy_backfill.py

Resume after interruption
-------------------------
Simply run the same command again.  Completed candidate/r cells are skipped.

Intentional fresh rerun
-----------------------
python 10_06b_h4_week10_all_noisy_backfill.py --overwrite
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, partial_trace

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
RESOURCE_HELPER_FILE = HERE / "09_05c_rwp_resource_and_hardware_gapfill.py"

H4_CANDIDATES_FILE = RESULTS / "09_05d_h4_week10_additions.csv"

AUDIT_LAYOUT_FILE = RESULTS / "10_06a_h4_current_kingston_layout.csv"
AUDIT_IDEAL_FILE = RESULTS / "10_06a_h4_ideal_depth_audit.csv"
AUDIT_COMPILE_DETAIL_FILE = RESULTS / "10_06a_h4_resource_compile_detail.csv"
AUDIT_RESOURCE_FILE = RESULTS / "10_06a_h4_resource_summary.csv"
AUDIT_MANIFEST_FILE = RESULTS / "10_06a_h4_manifest.json"

HIST_DIRECT_FILE = RESULTS / "10_05_direct_runs.csv"
HIST_SAMPLED_SUMMARY_FILE = RESULTS / "10_05_sampled_summary.csv"
HIST_PARETO_FILE = RESULTS / "10_05_pareto_front.csv"
HIST_MANIFEST_FILE = RESULTS / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"

OUT_AB = RESULTS / "10_06b_h4_ab_audit.csv"
OUT_DIRECT = RESULTS / "10_06b_h4_direct_runs.csv"
OUT_N5_FEATURES = RESULTS / "10_06b_h4_n5_features_predictions.csv"
OUT_SAMPLED_RUNS = RESULTS / "10_06b_h4_sampled_runs.csv"
OUT_SAMPLED_SUMMARY = RESULTS / "10_06b_h4_sampled_summary.csv"
OUT_CANDIDATE_SUMMARY = RESULTS / "10_06b_h4_candidate_summary.csv"
OUT_H4_PARETO = RESULTS / "10_06b_h4_pareto_front.csv"
OUT_EXPANDED_SUMMARY = RESULTS / "10_06b_h4_expanded_sampled_summary.csv"
OUT_EXPANDED_PARETO = RESULTS / "10_06b_h4_expanded_pareto_front.csv"
OUT_MANIFEST = RESULTS / "10_06b_h4_manifest.json"
OUT_LATEX = RESULTS / "10_06b_h4_week10_latex_snippet.tex"

BACKEND_NAME = "ibm_kingston"
TOPOLOGY = "H4"
R_VALUES = [2, 3, 4]

OPERATING_CELLS = [
    (2, 1024),
    (3, 512),
    (3, 1024),
    (3, 2048),
    (4, 512),
]
N_REPLICATES = 3

OPT_LEVEL = 1
SEED_TRANSPILE = 42
BASE_SAMPLE_SEED = 20260926

AB_ENDPOINT_OFFSETS = [0, 182, 364]
AB_TOL = 1e-9
IDEAL_AUDIT_TOL = 2e-5

# Batch size is only used for independent RWP density-matrix jobs.
DEFAULT_RWP_BATCH = 24

# Remove scheduling delays from the compact workload.  Gate/decoherence
# effects already come from the backend-derived per-operation noise model.
DROP_COMPACT_OPS = {"barrier", "delay"}

OUTPUTS_TO_OVERWRITE = [
    OUT_AB,
    OUT_DIRECT,
    OUT_N5_FEATURES,
    OUT_SAMPLED_RUNS,
    OUT_SAMPLED_SUMMARY,
    OUT_CANDIDATE_SUMMARY,
    OUT_H4_PARETO,
    OUT_EXPANDED_SUMMARY,
    OUT_EXPANDED_PARETO,
    OUT_MANIFEST,
    OUT_LATEX,
]


# =============================================================================
# IMPORT LOCAL PROJECT MODULES
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_1006b_common")
resource_helper = load_module(
    RESOURCE_HELPER_FILE,
    "qrc_1006b_resource_helper",
)


# =============================================================================
# GENERIC UTILITIES
# =============================================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def bias(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(pred - y))


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def json_safe(x):
    if x is None:
        return None
    if isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, np.generic):
        return json_safe(x.item())
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            pass
    return str(x)


def parse_json_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    return [int(v) for v in json.loads(str(value))]


def append_csv(path: Path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def pareto_mask(df: pd.DataFrame, minimize):
    vals = df[minimize].to_numpy(dtype=float)
    n = len(df)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        vi = vals[i]
        for j in range(n):
            if i == j:
                continue
            vj = vals[j]
            if np.all(vj <= vi) and np.any(vj < vi):
                keep[i] = False
                break
    return keep


def pick_existing_column(df: pd.DataFrame, names):
    for name in names:
        if name in df.columns:
            return name
    return None


# =============================================================================
# 10.06A AUDIT LOAD / FREEZE
# =============================================================================

def load_and_validate_audit():
    required = [
        H4_CANDIDATES_FILE,
        AUDIT_LAYOUT_FILE,
        AUDIT_IDEAL_FILE,
        AUDIT_COMPILE_DETAIL_FILE,
        AUDIT_RESOURCE_FILE,
        AUDIT_MANIFEST_FILE,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required H4 backfill inputs:\n  "
            + "\n  ".join(missing)
        )

    candidates = pd.read_csv(H4_CANDIDATES_FILE)
    ideal = pd.read_csv(AUDIT_IDEAL_FILE)
    resources = pd.read_csv(AUDIT_RESOURCE_FILE)
    layout_df = pd.read_csv(AUDIT_LAYOUT_FILE)
    manifest = json.loads(AUDIT_MANIFEST_FILE.read_text(encoding="utf-8"))

    if len(candidates) != 8:
        raise RuntimeError(
            f"Expected exactly 8 H4 Rule-1--Rule-4 candidates; found {len(candidates)}."
        )
    if set(candidates["topology"].astype(str)) != {"H4"}:
        raise RuntimeError("09_05d input must contain H4 only.")

    if manifest.get("candidate_count") != 8:
        raise RuntimeError("10.06A manifest candidate_count is not 8.")
    if manifest.get("topology") != "H4":
        raise RuntimeError("10.06A manifest topology is not H4.")
    if manifest.get("backend") != BACKEND_NAME:
        raise RuntimeError("10.06A backend differs from ibm_kingston.")
    if manifest.get("test_2026_used") is not False:
        raise RuntimeError("10.06A manifest indicates 2026 was used.")
    if manifest.get("all_zero_swap") is not True:
        raise RuntimeError("10.06A strict zero-SWAP audit did not pass.")
    if manifest.get("noisy_simulation_run") is not False:
        raise RuntimeError("10.06A unexpectedly says noisy simulation already ran.")

    if len(ideal) != 24:
        raise RuntimeError(
            f"Expected 24 ideal audit rows (8×3); found {len(ideal)}."
        )
    if set(pd.to_numeric(ideal["evaluated_r"]).astype(int)) != {2, 3, 4}:
        raise RuntimeError("10.06A ideal audit does not contain exactly r={2,3,4}.")
    if int(ideal["is_original_r"].astype(bool).sum()) != 8:
        raise RuntimeError("10.06A original-r audit does not contain 8 PASS rows.")

    if len(resources) != 24:
        raise RuntimeError(
            f"Expected 24 resource summary rows (8×3); found {len(resources)}."
        )
    if not np.allclose(
        pd.to_numeric(resources["NSWAP_median"], errors="coerce"),
        0.0,
        equal_nan=False,
    ):
        raise RuntimeError("10.06A resource summary contains non-zero SWAPs.")

    if len(layout_df) != 1:
        raise RuntimeError("Expected exactly one frozen H4 layout row.")

    layout = parse_json_list(layout_df.iloc[0]["layout"])
    manifest_layout = [int(x) for x in manifest["shared_layout"]]
    if layout != manifest_layout:
        raise RuntimeError(
            f"Layout mismatch: CSV={layout}, manifest={manifest_layout}"
        )

    if layout != [117, 105, 104, 124, 125, 126]:
        print(
            "WARNING: frozen H4 layout differs from the previously audited "
            "[117,105,104,124,125,126]. The script will use the files as "
            "authoritative, but inspect provenance before publication."
        )

    return candidates, ideal, resources, layout_df.iloc[0], manifest, layout


# =============================================================================
# EXACT H4 CANDIDATE RECONSTRUCTION
# =============================================================================

def parse_j(row: pd.Series):
    raw = json.loads(str(row["J_json"]))
    J = {}

    for edge in common.TOPOLOGY_EDGES["H4"]:
        i, j = edge
        key = f"J{i}{j}"
        rev = f"J{j}{i}"

        if key in raw and raw[key] is not None:
            val = raw[key]
        elif rev in raw and raw[rev] is not None:
            val = raw[rev]
        else:
            raise KeyError(
                f"{row['candidate_key']}: missing H4 coupling {key}/{rev}"
            )

        J[edge] = float(val)

    return J


def candidate_from_manifest(row: pd.Series, r: int):
    protocol = str(row["protocol"]).upper()

    return {
        "candidate_id": str(row["candidate_key"]),
        "topology": "H4",
        "protocol": protocol,
        "window": (
            1
            if protocol == "CONT"
            else int(float(row["window"]))
        ),
        "r": int(r),
        "alpha": float(row["alpha"]),
        "dt": float(row["dt"]),
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
        "J": parse_j(row),
    }


# =============================================================================
# IDEAL FEATURE BANK + FROZEN READOUT
# =============================================================================

def build_cont_ideal(candidate, angles):
    """
    Exact ideal CONT chronology.

    q4,q5 memory starts at |00><00| and is carried through the complete
    2022-2025 chronology.  The pre-step memory density matrices are retained
    for the transpiled A/B audit.
    """
    A_list = common.build_channels(candidate, angles)

    rho_m = common.qrc.memory_zero_density()
    rows = []
    pre_memory = []

    for A in A_list:
        pre_memory.append(np.asarray(rho_m, dtype=complex).copy())

        rho_i, rho_m_out = common.qrc.final_reduced_states(
            A,
            rho_m,
        )
        rows.append(
            common.reduced_feature_row(
                rho_i,
                rho_m_out,
            )
        )
        rho_m = rho_m_out

    endpoints = np.arange(len(A_list), dtype=int)

    return (
        endpoints,
        pd.DataFrame(rows),
        pre_memory,
    )


def build_ideal_reference(
    candidate,
    readout,
    audit_row,
):
    work_tv, cols, y_all = common.load_train_validation()
    angles = common.make_angles(
        work_tv,
        cols,
        float(candidate["alpha"]),
    )

    if candidate["protocol"] == "CONT":
        endpoints, master, pre_memory = build_cont_ideal(
            candidate,
            angles,
        )
    else:
        A_list = common.build_channels(candidate, angles)
        endpoints, master = common.rwp_master_feature_bank_from_channels(
            A_list,
            int(candidate["window"]),
        )
        pre_memory = None

    features = list(common.READOUT_FEATURES[readout])
    X = master[features].to_numpy(dtype=float)
    y_ep = y_all[endpoints]

    train_mask = endpoints < common.N_TRAIN
    val_mask = endpoints >= common.N_TRAIN

    X_train = X[train_mask]
    y_train = y_ep[train_mask]
    X_val = X[val_mask]
    y_val = y_ep[val_mask]
    val_endpoints = endpoints[val_mask]

    if len(y_val) != common.N_VAL:
        raise RuntimeError(
            f"{candidate['candidate_id']} r={candidate['r']}: "
            f"expected {common.N_VAL} validation rows, got {len(y_val)}"
        )

    lam = float(audit_row["selected_lambda"])

    model, scaler, keep = common.fit_scaled_ridge(
        X_train,
        y_train,
        lam,
    )
    pred_ideal = common.predict_scaled_ridge(
        model,
        scaler,
        keep,
        X_val,
    )

    # Strong reproduction guard.
    got_rmse = rmse(y_val, pred_ideal)
    expected_rmse = float(audit_row["ideal_2025_rmse"])

    if abs(got_rmse - expected_rmse) > IDEAL_AUDIT_TOL:
        raise RuntimeError(
            f"{candidate['candidate_id']} r={candidate['r']}: "
            f"ideal 2025 reproduction failed. "
            f"recomputed={got_rmse:.9f}, "
            f"10.06A={expected_rmse:.9f}"
        )

    return {
        "work_tv": work_tv,
        "cols": cols,
        "y_all": y_all,
        "angles": angles,
        "endpoints": endpoints,
        "master": master,
        "features": features,
        "X": X,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "val_endpoints": val_endpoints,
        "pred_ideal": pred_ideal,
        "model": model,
        "scaler": scaler,
        "keep": keep,
        "lambda": lam,
        "audit_cv_rmse": float(audit_row["cv_rmse"]),
        "audit_cv_rmse_std": float(audit_row["cv_rmse_std"]),
        "ideal_rmse": got_rmse,
        "ideal_mae": mae(y_val, pred_ideal),
        "ideal_bias": bias(y_val, pred_ideal),
        "pre_memory": pre_memory,
    }


# =============================================================================
# PARAMETERIZED LOGICAL CORE
# =============================================================================

def append_dynamics_parameterized(
    qc,
    candidate,
    angle_params,
    reset_injection,
):
    if reset_injection:
        for q in range(4):
            qc.reset(q)

    for q in range(4):
        qc.ry(angle_params[q], q)

    r = int(candidate["r"])
    dt = float(candidate["dt"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])

    for _ in range(r):
        for i, j in common.TOPOLOGY_EDGES["H4"]:
            qc.rzz(
                2.0
                * float(candidate["J"][(i, j)])
                * dt
                / r,
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


def build_parameterized_core(candidate):
    protocol = candidate["protocol"]

    qreg = QuantumRegister(6, "q")
    qc = QuantumCircuit(
        qreg,
        name=(
            f"{candidate['candidate_id']}_"
            f"{protocol}_r{candidate['r']}_core"
        ),
    )

    if protocol == "RWP":
        W = int(candidate["window"])
        pars = ParameterVector("theta", 4 * W)

        for step in range(W):
            step_params = [
                pars[4 * step + q]
                for q in range(4)
            ]
            append_dynamics_parameterized(
                qc,
                candidate,
                step_params,
                reset_injection=(step > 0),
            )

    elif protocol == "CONT":
        pars = ParameterVector("theta", 4)

        # CONT carries q4,q5 only.  q0-q3 are reset/re-encoded at each input.
        append_dynamics_parameterized(
            qc,
            candidate,
            [pars[q] for q in range(4)],
            reset_injection=True,
        )

    else:
        raise ValueError(protocol)

    return qc, list(pars)


# =============================================================================
# PHYSICAL TRANSPILE -> COMPACT SIX-QUBIT WORKLOAD
# =============================================================================

def compact_transpiled_circuit(
    transpiled_circuit,
    physical_layout,
):
    """
    Relabel the exact active physical wires back to compact indices 0..5.

    physical_layout[logical_q] = physical_q.
    """
    phys_to_compact = {
        int(physical_q): int(logical_q)
        for logical_q, physical_q
        in enumerate(physical_layout)
    }

    compact = QuantumCircuit(6)
    compact.global_phase = transpiled_circuit.global_phase

    used_physical = set()

    for ci in transpiled_circuit.data:
        op = ci.operation

        if op.name in DROP_COMPACT_OPS:
            continue

        if ci.clbits:
            raise RuntimeError(
                "Core circuit unexpectedly contains classical bits."
            )

        phys_qargs = [
            int(transpiled_circuit.find_bit(q).index)
            for q in ci.qubits
        ]

        if any(q not in phys_to_compact for q in phys_qargs):
            raise RuntimeError(
                "Transpiled core contains an operation outside the frozen "
                f"H4 layout. op={op.name}, qargs={phys_qargs}, "
                f"layout={physical_layout}"
            )

        compact_qargs = [
            compact.qubits[phys_to_compact[q]]
            for q in phys_qargs
        ]

        used_physical.update(phys_qargs)

        # append preserves the exact transpiled native operation/parameters.
        compact.append(
            op.copy() if hasattr(op, "copy") else op,
            compact_qargs,
            [],
        )

    if not used_physical.issubset(set(physical_layout)):
        raise RuntimeError("Compact mapping audit failed.")

    return compact


def compile_compact_core(
    candidate,
    backend,
    physical_layout,
):
    logical, original_params = build_parameterized_core(candidate)

    physical = transpile(
        logical,
        backend=backend,
        initial_layout=physical_layout,
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
    )

    backend.check_faulty(physical)

    if int(physical.count_ops().get("swap", 0)) != 0:
        raise RuntimeError(
            f"{candidate['candidate_id']} r={candidate['r']}: "
            "SWAP detected in strict physical core."
        )

    compact = compact_transpiled_circuit(
        physical,
        physical_layout,
    )

    compact_params_by_name = {
        p.name: p
        for p in compact.parameters
    }

    param_names = [p.name for p in original_params]

    missing = [
        name
        for name in param_names
        if name not in compact_params_by_name
    ]

    # Transpilation may mathematically remove a parameter only if its gate
    # vanished.  That must not happen silently for this workload.
    if missing:
        raise RuntimeError(
            f"{candidate['candidate_id']} r={candidate['r']}: "
            f"transpilation removed expected parameters: {missing}"
        )

    return {
        "physical": physical,
        "compact": compact,
        "param_names": param_names,
        "param_objects": compact_params_by_name,
    }


def bind_compact_core(
    compiled,
    parameter_values,
):
    if len(parameter_values) != len(compiled["param_names"]):
        raise ValueError(
            f"Parameter length mismatch: got {len(parameter_values)}, "
            f"expected {len(compiled['param_names'])}"
        )

    mapping = {
        compiled["param_objects"][name]: float(value)
        for name, value
        in zip(
            compiled["param_names"],
            parameter_values,
        )
    }

    return compiled["compact"].assign_parameters(
        mapping,
        inplace=False,
    )


def parameter_values_for_endpoint(
    candidate,
    angles,
    endpoint,
):
    if candidate["protocol"] == "CONT":
        return [
            float(angles[int(endpoint), q])
            for q in range(4)
        ]

    W = int(candidate["window"])
    start = int(endpoint) - W + 1

    if start < 0:
        raise ValueError("RWP endpoint has insufficient history.")

    vals = []
    for idx in range(start, int(endpoint) + 1):
        for q in range(4):
            vals.append(float(angles[idx, q]))

    return vals


# =============================================================================
# BACKEND NOISE MODEL -> COMPACT SIX-QUBIT NOISE MODEL
# =============================================================================

def build_compact_quantum_noise_model(
    backend,
    physical_layout,
):
    """
    Current backend-derived full QUANTUM noise with readout assignment OFF.

    The physical local quantum errors are remapped from the selected Kingston
    qubit IDs to compact q0..q5 indices.  Readout errors are intentionally
    omitted.
    """
    full = NoiseModel.from_backend(
        backend,
        gate_error=True,
        readout_error=False,
        thermal_relaxation=True,
    )

    phys_to_compact = {
        int(p): int(i)
        for i, p in enumerate(physical_layout)
    }

    compact = NoiseModel(
        basis_gates=list(full.basis_gates)
    )

    mapped_local = 0
    mapped_default = 0

    # Qiskit Aer 0.17.x internal representation.  Using the QuantumError
    # objects directly is more reliable than serializing Kraus circuits.
    default_errors = getattr(
        full,
        "_default_quantum_errors",
        {},
    )

    for instruction, error in default_errors.items():
        compact.add_all_qubit_quantum_error(
            error,
            instruction,
        )
        mapped_default += 1

    local_errors = getattr(
        full,
        "_local_quantum_errors",
        {},
    )

    for instruction, by_qargs in local_errors.items():
        if not isinstance(by_qargs, dict):
            continue

        for qargs, error in by_qargs.items():
            qargs_tuple = tuple(int(q) for q in qargs)

            if all(q in phys_to_compact for q in qargs_tuple):
                mapped = [
                    phys_to_compact[q]
                    for q in qargs_tuple
                ]
                compact.add_quantum_error(
                    error,
                    instruction,
                    mapped,
                )
                mapped_local += 1

    # Defensive fallback for a future Aer internal representation.
    if mapped_local == 0 and mapped_default == 0:
        noise_dict = full.to_dict(serializable=False)
        new_errors = []

        for err in noise_dict.get("errors", []):
            if str(err.get("type", "")).lower() == "roerror":
                continue

            e = copy.deepcopy(err)
            gate_qubits = e.get("gate_qubits")

            if gate_qubits is None:
                new_errors.append(e)
                continue

            mapped_sets = []

            for qset in gate_qubits:
                qset = [int(q) for q in qset]
                if all(q in phys_to_compact for q in qset):
                    mapped_sets.append(
                        [
                            phys_to_compact[q]
                            for q in qset
                        ]
                    )

            if mapped_sets:
                e["gate_qubits"] = mapped_sets
                new_errors.append(e)

        if not hasattr(NoiseModel, "from_dict"):
            raise RuntimeError(
                "Could not remap backend noise model with this Aer version."
            )

        compact = NoiseModel.from_dict(
            {
                "errors": new_errors,
            }
        )
        mapped_local = len(new_errors)

    if compact.is_ideal():
        raise RuntimeError(
            "Compact backend noise model is ideal; physical calibration "
            "errors were not mapped successfully."
        )

    info = {
        "physical_layout": list(physical_layout),
        "mapped_local_quantum_errors": int(mapped_local),
        "mapped_default_quantum_errors": int(mapped_default),
        "noise_basis_gates": list(compact.basis_gates),
        "noise_instructions": sorted(
            str(x)
            for x in compact.noise_instructions
        ),
        "noise_qubits": sorted(
            int(x)
            for x in compact.noise_qubits
        ),
        "readout_error_included": False,
    }

    return compact, info


# =============================================================================
# DENSITY-MATRIX EXECUTION
# =============================================================================

def zero_injection_density():
    rho = np.zeros((16, 16), dtype=complex)
    rho[0, 0] = 1.0
    return rho


ZERO_INJECTION_RHO = zero_injection_density()

# The ideal NumPy QRC stores the two-qubit memory subsystem in tensor order
# |q4 q5>, while Qiskit's DensityMatrix convention for qubits [4,5] uses basis
# ordering |q5 q4>.  The states are physically identical but their 4x4 matrix
# representations differ by a two-qubit bit-reversal permutation.
_MEMORY_IDEAL_TO_QISKIT = np.array([0, 2, 1, 3], dtype=int)


def ideal_memory_to_qiskit_order(memory_rho):
    rho = np.asarray(memory_rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 ideal memory density matrix; got {rho.shape}"
        )
    p = _MEMORY_IDEAL_TO_QISKIT
    return rho[np.ix_(p, p)]


def full_density_from_memory(memory_rho):
    # Qiskit little-endian ordering:
    # q0..q3 injection are the low-order subsystem,
    # q4,q5 memory are the high-order subsystem.
    return np.kron(
        np.asarray(memory_rho, dtype=complex),
        ZERO_INJECTION_RHO,
    )


def execute_density(
    simulator,
    bound_core,
    initial_density=None,
):
    qc = QuantumCircuit(6)

    if initial_density is not None:
        qc.set_density_matrix(
            np.asarray(
                initial_density,
                dtype=complex,
            )
        )

    qc.compose(bound_core, inplace=True)
    qc.save_density_matrix(label="rho")

    result = simulator.run(
        qc,
        shots=None,
    ).result()

    return np.asarray(
        result.data(0)["rho"],
        dtype=complex,
    )


def execute_rwp_batch(
    simulator,
    compiled,
    candidate,
    angles,
    endpoints,
    batch_size,
):
    out = []

    for start in range(0, len(endpoints), batch_size):
        stop = min(
            start + batch_size,
            len(endpoints),
        )

        circuits = []

        for endpoint in endpoints[start:stop]:
            values = parameter_values_for_endpoint(
                candidate,
                angles,
                int(endpoint),
            )
            bound = bind_compact_core(
                compiled,
                values,
            )

            qc = QuantumCircuit(6)
            qc.compose(bound, inplace=True)
            qc.save_density_matrix(label="rho")
            circuits.append(qc)

        result = simulator.run(
            circuits,
            shots=None,
        ).result()

        for i in range(len(circuits)):
            out.append(
                np.asarray(
                    result.data(i)["rho"],
                    dtype=complex,
                )
            )

    return out


def execute_cont_chronology(
    simulator,
    compiled,
    candidate,
    angles,
    total_steps,
    collect_from,
):
    """
    Propagate q4,q5 chronologically under the compact noisy channel.

    Memory is initialized as |00><00| and then carried step by step.
    """
    memory_rho = np.zeros((4, 4), dtype=complex)
    memory_rho[0, 0] = 1.0

    collected = []

    for endpoint in range(int(total_steps)):
        values = parameter_values_for_endpoint(
            candidate,
            angles,
            endpoint,
        )
        bound = bind_compact_core(
            compiled,
            values,
        )

        rho6 = execute_density(
            simulator,
            bound,
            initial_density=full_density_from_memory(
                memory_rho
            ),
        )

        memory_rho = np.asarray(
            partial_trace(
                DensityMatrix(rho6),
                [0, 1, 2, 3],
            ).data,
            dtype=complex,
        )

        if endpoint >= int(collect_from):
            collected.append(rho6)

    return collected


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def sparse_feature_observable(feature):
    if feature == "YX45":
        return SparsePauliOp.from_sparse_list(
            [("YX", [4, 5], 1.0)],
            num_qubits=6,
        )

    axis = feature[0]
    q = int(feature[1:])

    if axis not in {"X", "Y", "Z"}:
        raise KeyError(feature)

    return SparsePauliOp.from_sparse_list(
        [(axis, [q], 1.0)],
        num_qubits=6,
    )


_OBSERVABLE_CACHE = {}


def direct_feature_vector(rho6, features):
    dm = DensityMatrix(rho6)
    vals = []

    for feature in features:
        if feature not in _OBSERVABLE_CACHE:
            _OBSERVABLE_CACHE[feature] = sparse_feature_observable(
                feature
            )

        vals.append(
            float(
                np.real(
                    dm.expectation_value(
                        _OBSERVABLE_CACHE[feature]
                    )
                )
            )
        )

    return np.asarray(vals, dtype=float)


def setting_probabilities(rho6, setting):
    """
    Ideal projective measurement probabilities of the already-noisy N5 state.

    Measurement-basis rotations are mathematical readout rotations here, not
    additional noisy reservoir gates.  Therefore S2 differs from N5 only by
    finite-shot estimation, as required by the historical Week-10 definition.
    """
    _, basis, measured = setting

    rot = QuantumCircuit(6)

    for q, axis in basis.items():
        if axis == "X":
            rot.h(q)
        elif axis == "Y":
            rot.sdg(q)
            rot.h(q)
        elif axis == "Z":
            pass
        else:
            raise ValueError(axis)

    dm = DensityMatrix(rho6).evolve(rot)

    probs = np.asarray(
        dm.probabilities(qargs=list(measured)),
        dtype=float,
    )

    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()

    return probs


def sampled_feature_vector_from_outcomes(
    settings,
    outcomes_by_label,
    features,
    shots,
):
    vals = []

    for feature in features:
        if feature == "YX45":
            chosen = None

            for setting in settings:
                label, basis, measured = setting
                if (
                    basis.get(4) == "Y"
                    and basis.get(5) == "X"
                    and 4 in measured
                    and 5 in measured
                ):
                    chosen = setting
                    break

            if chosen is None:
                raise RuntimeError(
                    "No Y4/X5 setting available for YX45."
                )

            label, _, measured = chosen
            samples = outcomes_by_label[label][:shots]

            p4 = list(measured).index(4)
            p5 = list(measured).index(5)

            b4 = (samples >> p4) & 1
            b5 = (samples >> p5) & 1

            vals.append(
                float(
                    np.mean(
                        np.where(
                            (b4 ^ b5) == 0,
                            1.0,
                            -1.0,
                        )
                    )
                )
            )
            continue

        axis = feature[0]
        q = int(feature[1:])

        chosen = None

        for setting in settings:
            label, basis, measured = setting
            if basis.get(q) == axis and q in measured:
                chosen = setting
                break

        if chosen is None:
            raise RuntimeError(
                f"No measurement setting found for feature {feature}."
            )

        label, _, measured = chosen
        samples = outcomes_by_label[label][:shots]

        pos = list(measured).index(q)
        bits = (samples >> pos) & 1

        vals.append(
            float(
                np.mean(
                    np.where(
                        bits == 0,
                        1.0,
                        -1.0,
                    )
                )
            )
        )

    return np.asarray(vals, dtype=float)


def sample_feature_bank_nested(
    noisy_rhos,
    readout,
    features,
    shot_budgets,
    replicate_seed,
):
    """
    Draw one maximum-shot realization per endpoint/setting and reuse prefixes
    for all smaller shot budgets.  This is the same nested-shot principle used
    in the Week-10 shot-sensitivity study.
    """
    settings = list(
        resource_helper.readout_settings(readout)
    )
    max_shots = int(max(shot_budgets))

    banks = {
        int(n): np.empty(
            (len(noisy_rhos), len(features)),
            dtype=float,
        )
        for n in shot_budgets
    }

    rng = np.random.default_rng(
        int(replicate_seed)
    )

    for row_i, rho6 in enumerate(noisy_rhos):
        outcomes_by_label = {}

        for setting in settings:
            label = str(setting[0])
            probs = setting_probabilities(
                rho6,
                setting,
            )

            outcomes_by_label[label] = rng.choice(
                len(probs),
                size=max_shots,
                replace=True,
                p=probs,
            ).astype(np.int64)

        for shots in shot_budgets:
            banks[int(shots)][row_i, :] = (
                sampled_feature_vector_from_outcomes(
                    settings,
                    outcomes_by_label,
                    features,
                    int(shots),
                )
            )

    return banks


# =============================================================================
# A/B NOISELESS TRANSPILE AUDIT
# =============================================================================

def audit_transpiled_ideal(
    candidate,
    frozen,
    compiled,
):
    noiseless = AerSimulator(
        method="density_matrix"
    )

    audit_rows = []
    val_eps = list(
        map(int, frozen["val_endpoints"])
    )

    selected_positions = [
        0,
        len(val_eps) // 2,
        len(val_eps) - 1,
    ]

    endpoint_to_master = {
        int(ep): i
        for i, ep
        in enumerate(frozen["endpoints"])
    }

    for pos in selected_positions:
        endpoint = val_eps[pos]

        values = parameter_values_for_endpoint(
            candidate,
            frozen["angles"],
            endpoint,
        )
        bound = bind_compact_core(
            compiled,
            values,
        )

        if candidate["protocol"] == "CONT":
            # IMPORTANT:
            # frozen["pre_memory"] comes from the ideal NumPy reservoir, whose
            # memory basis is |q4 q5>.  Before injecting that state into a
            # Qiskit DensityMatrix we must convert it to Qiskit's |q5 q4>
            # ordering.  Without this permutation the A/B audit compares the
            # same physical state written in two different tensor conventions.
            pre = ideal_memory_to_qiskit_order(
                frozen["pre_memory"][endpoint]
            )
            rho_b = execute_density(
                noiseless,
                bound,
                initial_density=full_density_from_memory(
                    pre
                ),
            )
        else:
            rho_b = execute_density(
                noiseless,
                bound,
                initial_density=None,
            )

        x_b = direct_feature_vector(
            rho_b,
            frozen["features"],
        )

        master_i = endpoint_to_master[endpoint]
        x_a = frozen["X"][master_i, :]

        ferr = rmse(x_a, x_b)
        maxerr = float(
            np.max(
                np.abs(x_a - x_b)
            )
        )

        audit_rows.append(
            {
                "candidate_key": candidate["candidate_id"],
                "protocol": candidate["protocol"],
                "window": (
                    np.nan
                    if candidate["protocol"] == "CONT"
                    else candidate["window"]
                ),
                "readout": None,
                "r": int(candidate["r"]),
                "endpoint": int(endpoint),
                "feature_rmse_A_vs_B": ferr,
                "max_abs_feature_error_A_vs_B": maxerr,
                "pass": bool(ferr <= AB_TOL),
            }
        )

    if not all(row["pass"] for row in audit_rows):
        worst = max(
            audit_rows,
            key=lambda x: x["feature_rmse_A_vs_B"],
        )
        raise RuntimeError(
            f"{candidate['candidate_id']} r={candidate['r']} "
            f"A/B audit FAILED. Worst feature RMSE="
            f"{worst['feature_rmse_A_vs_B']:.3e}"
        )

    return audit_rows


# =============================================================================
# RESOURCE LOOKUP
# =============================================================================

def resource_row_for(
    resources,
    candidate_key,
    r,
):
    g = resources[
        (resources["candidate_key"].astype(str) == str(candidate_key))
        & (
            pd.to_numeric(
                resources["r"],
                errors="coerce",
            ).astype(int)
            == int(r)
        )
    ]

    if len(g) != 1:
        raise RuntimeError(
            f"Expected one resource row for {candidate_key}, r={r}; "
            f"found {len(g)}"
        )

    row = g.iloc[0]

    return {
        "feature_vector_n_cz": float(row["NCZ_median"]),
        "feature_vector_n_swap": float(row["NSWAP_median"]),
        "max_setting_depth": float(
            row["depth_max_setting_median"]
        ),
        "feature_vector_duration_us": float(
            row["Tcircuit_feature_vector_us_median"]
        ),
        "max_setting_duration_us": float(
            row["Tcircuit_max_setting_us_median"]
        ),
        "n_settings": int(
            round(float(row["n_settings"]))
        ),
        "resets_feature_vector": float(
            row["resets_feature_vector_median"]
        ),
    }


# =============================================================================
# N5 + S2 FOR ONE CANDIDATE/R
# =============================================================================

def evaluate_candidate_r(
    candidate_row,
    audit_row,
    resources,
    backend,
    compact_noise,
    physical_layout,
    rwp_batch,
    candidate_index,
):
    candidate = candidate_from_manifest(
        candidate_row,
        int(audit_row["evaluated_r"]),
    )
    readout = str(candidate_row["test_readout"])

    frozen = build_ideal_reference(
        candidate,
        readout,
        audit_row,
    )

    compiled = compile_compact_core(
        candidate,
        backend,
        physical_layout,
    )

    ab_rows = audit_transpiled_ideal(
        candidate,
        frozen,
        compiled,
    )

    noisy_sim = AerSimulator(
        method="density_matrix",
        noise_model=compact_noise,
    )
    noisy_sim.set_options(
        enable_truncation=False
    )

    t0 = time.time()

    if candidate["protocol"] == "RWP":
        noisy_rhos = execute_rwp_batch(
            noisy_sim,
            compiled,
            candidate,
            frozen["angles"],
            list(map(int, frozen["val_endpoints"])),
            int(rwp_batch),
        )
    else:
        noisy_rhos = execute_cont_chronology(
            noisy_sim,
            compiled,
            candidate,
            frozen["angles"],
            total_steps=(
                common.N_TRAIN
                + common.N_VAL
            ),
            collect_from=common.N_TRAIN,
        )

    elapsed_n5 = time.time() - t0

    if len(noisy_rhos) != common.N_VAL:
        raise RuntimeError(
            f"{candidate['candidate_id']} r={candidate['r']}: "
            f"expected {common.N_VAL} N5 states, got {len(noisy_rhos)}"
        )

    X_n5 = np.vstack(
        [
            direct_feature_vector(
                rho,
                frozen["features"],
            )
            for rho in noisy_rhos
        ]
    )

    pred_n5 = common.predict_scaled_ridge(
        frozen["model"],
        frozen["scaler"],
        frozen["keep"],
        X_n5,
    )

    resources_here = resource_row_for(
        resources,
        candidate["candidate_id"],
        candidate["r"],
    )

    direct_row = {
        "candidate_key": candidate["candidate_id"],
        "selected_by_rules": str(
            candidate_row["selected_by_rules"]
        ),
        "protocol": candidate["protocol"],
        "topology": "H4",
        "window": (
            np.nan
            if candidate["protocol"] == "CONT"
            else int(candidate["window"])
        ),
        "readout": readout,
        "r": int(candidate["r"]),
        "alpha": float(candidate["alpha"]),
        "dt": float(candidate["dt"]),
        "hx": float(candidate["hx"]),
        "hy": float(candidate["hy"]),
        "selected_lambda": float(frozen["lambda"]),
        "training_cv_rmse": float(
            frozen["audit_cv_rmse"]
        ),
        "training_cv_rmse_std": float(
            frozen["audit_cv_rmse_std"]
        ),
        "ideal_2025_rmse": float(
            frozen["ideal_rmse"]
        ),
        "ideal_2025_mae": float(
            frozen["ideal_mae"]
        ),
        "ideal_2025_bias": float(
            frozen["ideal_bias"]
        ),
        "rmse": rmse(
            frozen["y_val"],
            pred_n5,
        ),
        "mae": mae(
            frozen["y_val"],
            pred_n5,
        ),
        "bias": bias(
            frozen["y_val"],
            pred_n5,
        ),
        "delta_rmse_vs_same_r_ideal": (
            rmse(
                frozen["y_val"],
                pred_n5,
            )
            - frozen["ideal_rmse"]
        ),
        "prediction_corr_vs_same_r_ideal": safe_corr(
            frozen["pred_ideal"],
            pred_n5,
        ),
        "prediction_rmse_vs_same_r_ideal": rmse(
            frozen["pred_ideal"],
            pred_n5,
        ),
        "feature_rmse_vs_same_r_ideal": rmse(
            frozen["X_val"],
            X_n5,
        ),
        "feature_mae_vs_same_r_ideal": mae(
            frozen["X_val"],
            X_n5,
        ),
        "n_validation_endpoints": int(
            len(frozen["y_val"])
        ),
        "backend": BACKEND_NAME,
        "layout": json.dumps(
            list(physical_layout)
        ),
        "noise_regime": "N5_FULL_QUANTUM_DIRECT_READOUT_OFF",
        "readout_assignment_error": False,
        "finite_shots": False,
        "runtime_seconds": float(elapsed_n5),
        "2025_used_for_selection": False,
        "2026_used": False,
        **resources_here,
    }

    # Feature/prediction detail.
    detail_rows = []

    for i, endpoint in enumerate(frozen["val_endpoints"]):
        row = {
            "candidate_key": candidate["candidate_id"],
            "r": int(candidate["r"]),
            "endpoint": int(endpoint),
            "target": float(frozen["y_val"][i]),
            "pred_ideal": float(frozen["pred_ideal"][i]),
            "pred_n5": float(pred_n5[i]),
        }

        for j, feature in enumerate(frozen["features"]):
            row[f"{feature}_ideal"] = float(
                frozen["X_val"][i, j]
            )
            row[f"{feature}_n5"] = float(
                X_n5[i, j]
            )
            row[f"{feature}_error"] = float(
                X_n5[i, j]
                - frozen["X_val"][i, j]
            )

        detail_rows.append(row)

    # -----------------------------------------------------------------
    # S2 sampled operating points for this r
    # -----------------------------------------------------------------
    shot_budgets = sorted(
        shots
        for rr, shots in OPERATING_CELLS
        if int(rr) == int(candidate["r"])
    )

    sampled_rows = []

    for rep in range(N_REPLICATES):
        seed = (
            BASE_SAMPLE_SEED
            + int(candidate_index) * 1_000_000
            + int(candidate["r"]) * 10_000
            + int(rep) * 1_000
        )

        t_rep = time.time()

        banks = sample_feature_bank_nested(
            noisy_rhos,
            readout,
            frozen["features"],
            shot_budgets,
            replicate_seed=seed,
        )

        for shots in shot_budgets:
            X_s2 = banks[int(shots)]

            pred_s2 = common.predict_scaled_ridge(
                frozen["model"],
                frozen["scaler"],
                frozen["keep"],
                X_s2,
            )

            sampled_rows.append(
                {
                    "candidate_key": candidate["candidate_id"],
                    "selected_by_rules": str(
                        candidate_row["selected_by_rules"]
                    ),
                    "protocol": candidate["protocol"],
                    "topology": "H4",
                    "window": (
                        np.nan
                        if candidate["protocol"] == "CONT"
                        else int(candidate["window"])
                    ),
                    "readout": readout,
                    "r": int(candidate["r"]),
                    "shots_per_setting": int(shots),
                    "replicate": int(rep + 1),
                    "seed": int(seed),
                    "rmse": rmse(
                        frozen["y_val"],
                        pred_s2,
                    ),
                    "mae": mae(
                        frozen["y_val"],
                        pred_s2,
                    ),
                    "bias": bias(
                        frozen["y_val"],
                        pred_s2,
                    ),
                    "direct_full_quantum_rmse": float(
                        direct_row["rmse"]
                    ),
                    "delta_rmse_vs_full_quantum_direct": (
                        rmse(
                            frozen["y_val"],
                            pred_s2,
                        )
                        - direct_row["rmse"]
                    ),
                    "prediction_corr_vs_full_quantum_direct": safe_corr(
                        pred_n5,
                        pred_s2,
                    ),
                    "prediction_rmse_vs_full_quantum_direct": rmse(
                        pred_n5,
                        pred_s2,
                    ),
                    "feature_rmse_vs_full_quantum_direct": rmse(
                        X_n5,
                        X_s2,
                    ),
                    "feature_mae_vs_full_quantum_direct": mae(
                        X_n5,
                        X_s2,
                    ),
                    "ideal_2025_rmse": float(
                        frozen["ideal_rmse"]
                    ),
                    "feature_vector_n_cz": float(
                        resources_here["feature_vector_n_cz"]
                    ),
                    "max_setting_depth": float(
                        resources_here["max_setting_depth"]
                    ),
                    "feature_vector_duration_us": float(
                        resources_here["feature_vector_duration_us"]
                    ),
                    "max_setting_duration_us": float(
                        resources_here["max_setting_duration_us"]
                    ),
                    "cz_applications_per_feature_vector": (
                        float(
                            resources_here["feature_vector_n_cz"]
                        )
                        * int(shots)
                    ),
                    "noise_regime": "S2_FULL_QUANTUM_SAMPLED_READOUT_OFF",
                    "readout_assignment_error": False,
                    "error_mitigation": False,
                    "2025_used_for_selection": False,
                    "2026_used": False,
                    "replicate_runtime_seconds": (
                        time.time()
                        - t_rep
                    ),
                }
            )

    return (
        ab_rows,
        direct_row,
        detail_rows,
        sampled_rows,
    )


# =============================================================================
# SUMMARY / PARETO
# =============================================================================

def summarize_sampled(sampled_runs: pd.DataFrame):
    group_cols = [
        "candidate_key",
        "selected_by_rules",
        "protocol",
        "topology",
        "window",
        "readout",
        "r",
        "shots_per_setting",
    ]

    summary = (
        sampled_runs.groupby(
            group_cols,
            dropna=False,
        )
        .agg(
            n_runs=("rmse", "count"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            bias_mean=("bias", "mean"),
            direct_full_quantum_rmse=(
                "direct_full_quantum_rmse",
                "first",
            ),
            delta_rmse_vs_full_quantum_direct_mean=(
                "delta_rmse_vs_full_quantum_direct",
                "mean",
            ),
            prediction_corr_vs_full_quantum_direct_mean=(
                "prediction_corr_vs_full_quantum_direct",
                "mean",
            ),
            feature_rmse_vs_full_quantum_direct_mean=(
                "feature_rmse_vs_full_quantum_direct",
                "mean",
            ),
            ideal_2025_rmse=(
                "ideal_2025_rmse",
                "first",
            ),
            feature_vector_n_cz=(
                "feature_vector_n_cz",
                "first",
            ),
            max_setting_depth=(
                "max_setting_depth",
                "first",
            ),
            feature_vector_duration_us=(
                "feature_vector_duration_us",
                "first",
            ),
            max_setting_duration_us=(
                "max_setting_duration_us",
                "first",
            ),
            cz_applications_per_feature_vector=(
                "cz_applications_per_feature_vector",
                "first",
            ),
        )
        .reset_index()
    )

    summary["rmse_sd"] = summary["rmse_sd"].fillna(0.0)
    summary["rmse_se"] = (
        summary["rmse_sd"]
        / np.sqrt(summary["n_runs"])
    )

    return summary


def h4_candidate_summary(
    direct_df,
    sampled_summary,
):
    rows = []

    for candidate_key in sorted(
        sampled_summary["candidate_key"]
        .astype(str)
        .unique()
    ):
        d = direct_df[
            direct_df["candidate_key"].astype(str)
            == candidate_key
        ].copy()

        s = sampled_summary[
            sampled_summary["candidate_key"].astype(str)
            == candidate_key
        ].copy()

        best_d = (
            d.sort_values(
                [
                    "rmse",
                    "feature_vector_n_cz",
                    "r",
                ]
            )
            .iloc[0]
        )

        best_s = (
            s.sort_values(
                [
                    "rmse_mean",
                    "feature_vector_n_cz",
                    "shots_per_setting",
                ]
            )
            .iloc[0]
        )

        rows.append(
            {
                "candidate_key": candidate_key,
                "selected_by_rules": best_s["selected_by_rules"],
                "protocol": best_s["protocol"],
                "window": best_s["window"],
                "readout": best_s["readout"],
                "best_direct_r": int(best_d["r"]),
                "best_direct_ideal_rmse": float(
                    best_d["ideal_2025_rmse"]
                ),
                "best_direct_n5_rmse": float(
                    best_d["rmse"]
                ),
                "best_direct_delta": float(
                    best_d["delta_rmse_vs_same_r_ideal"]
                ),
                "best_sampled_r": int(best_s["r"]),
                "best_sampled_shots": int(
                    best_s["shots_per_setting"]
                ),
                "best_sampled_rmse_mean": float(
                    best_s["rmse_mean"]
                ),
                "best_sampled_rmse_sd": float(
                    best_s["rmse_sd"]
                ),
                "best_sampled_delta_vs_n5": float(
                    best_s[
                        "delta_rmse_vs_full_quantum_direct_mean"
                    ]
                ),
                "feature_vector_n_cz": float(
                    best_s["feature_vector_n_cz"]
                ),
                "max_setting_depth": float(
                    best_s["max_setting_depth"]
                ),
                "max_setting_duration_us": float(
                    best_s["max_setting_duration_us"]
                ),
                "cz_applications_per_feature_vector": float(
                    best_s[
                        "cz_applications_per_feature_vector"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def h4_pareto(sampled_summary):
    df = sampled_summary.copy()
    df["pareto"] = pareto_mask(
        df,
        minimize=[
            "rmse_mean",
            "feature_vector_n_cz",
            "shots_per_setting",
        ],
    )

    return (
        df[df["pareto"]]
        .sort_values(
            [
                "rmse_mean",
                "feature_vector_n_cz",
                "shots_per_setting",
            ]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# MERGE WITH FROZEN H0-H3 WEEK-10 RESULTS
# =============================================================================

def canonicalize_historical_sampled():
    if not HIST_SAMPLED_SUMMARY_FILE.exists():
        return None, (
            "Historical results/10_05_sampled_summary.csv not found; "
            "expanded H0-H4 merge skipped."
        )

    hist = pd.read_csv(
        HIST_SAMPLED_SUMMARY_FILE
    )

    c_candidate = pick_existing_column(
        hist,
        ["candidate_key", "candidate", "candidate_id"],
    )
    c_r = pick_existing_column(
        hist,
        ["r", "trotter_r"],
    )
    c_shots = pick_existing_column(
        hist,
        ["shots_per_setting", "shots"],
    )
    c_rmse = pick_existing_column(
        hist,
        ["rmse_mean", "sampled_rmse_mean", "rmse"],
    )
    c_sd = pick_existing_column(
        hist,
        ["rmse_sd", "sampled_rmse_sd"],
    )
    c_cz = pick_existing_column(
        hist,
        [
            "feature_vector_n_cz",
            "cz_per_feature",
            "feature_vector_cz",
        ],
    )
    c_depth = pick_existing_column(
        hist,
        ["max_setting_depth", "depth"],
    )
    c_dur = pick_existing_column(
        hist,
        [
            "max_setting_duration_us",
            "duration_us",
        ],
    )

    required = {
        "candidate": c_candidate,
        "r": c_r,
        "shots": c_shots,
        "rmse": c_rmse,
    }

    missing = [
        name
        for name, col in required.items()
        if col is None
    ]

    if missing:
        return None, (
            "Historical 10_05_sampled_summary.csv schema could not be "
            f"normalized; missing {missing}. Expanded merge skipped."
        )

    # If resource columns are not in sampled summary, try the historical direct
    # table and merge candidate/r resource metrics.
    if c_cz is None and HIST_DIRECT_FILE.exists():
        direct = pd.read_csv(HIST_DIRECT_FILE)

        dc = pick_existing_column(
            direct,
            ["candidate_key", "candidate", "candidate_id"],
        )
        dr = pick_existing_column(
            direct,
            ["r", "trotter_r"],
        )
        dcz = pick_existing_column(
            direct,
            [
                "feature_vector_n_cz",
                "cz_per_feature",
                "feature_vector_cz",
            ],
        )
        dd = pick_existing_column(
            direct,
            ["max_setting_depth", "depth"],
        )
        du = pick_existing_column(
            direct,
            [
                "max_setting_duration_us",
                "duration_us",
            ],
        )

        if dc and dr and dcz:
            resource_sub = (
                direct[
                    [
                        dc,
                        dr,
                        dcz,
                        *(
                            [dd]
                            if dd
                            else []
                        ),
                        *(
                            [du]
                            if du
                            else []
                        ),
                    ]
                ]
                .drop_duplicates(
                    [dc, dr]
                )
                .copy()
            )

            rename = {
                dc: "_candidate",
                dr: "_r",
                dcz: "_cz",
            }
            if dd:
                rename[dd] = "_depth"
            if du:
                rename[du] = "_dur"

            resource_sub = resource_sub.rename(
                columns=rename
            )

            hist = hist.merge(
                resource_sub,
                left_on=[c_candidate, c_r],
                right_on=["_candidate", "_r"],
                how="left",
            )

            c_cz = "_cz"
            if c_depth is None and "_depth" in hist:
                c_depth = "_depth"
            if c_dur is None and "_dur" in hist:
                c_dur = "_dur"

    if c_cz is None:
        return None, (
            "Historical Week-10 sampled/direct files do not expose "
            "CZ/feature; expanded Pareto merge skipped."
        )

    out = pd.DataFrame(
        {
            "source": "historical_H0_H3",
            "candidate_key": hist[c_candidate].astype(str),
            "topology": (
                hist["topology"].astype(str)
                if "topology" in hist.columns
                else hist[c_candidate]
                .astype(str)
                .str.extract(r"(H[0-3])", expand=False)
            ),
            "protocol": (
                hist["protocol"].astype(str)
                if "protocol" in hist.columns
                else np.nan
            ),
            "window": (
                pd.to_numeric(
                    hist["window"],
                    errors="coerce",
                )
                if "window" in hist.columns
                else (
                    pd.to_numeric(
                        hist["window_size"],
                        errors="coerce",
                    )
                    if "window_size" in hist.columns
                    else np.nan
                )
            ),
            "readout": (
                hist["readout"].astype(str)
                if "readout" in hist.columns
                else (
                    hist["test_readout"].astype(str)
                    if "test_readout" in hist.columns
                    else np.nan
                )
            ),
            "r": pd.to_numeric(
                hist[c_r],
                errors="coerce",
            ).astype(int),
            "shots_per_setting": pd.to_numeric(
                hist[c_shots],
                errors="coerce",
            ).astype(int),
            "rmse_mean": pd.to_numeric(
                hist[c_rmse],
                errors="coerce",
            ),
            "rmse_sd": (
                pd.to_numeric(
                    hist[c_sd],
                    errors="coerce",
                )
                if c_sd
                else np.nan
            ),
            "feature_vector_n_cz": pd.to_numeric(
                hist[c_cz],
                errors="coerce",
            ),
            "max_setting_depth": (
                pd.to_numeric(
                    hist[c_depth],
                    errors="coerce",
                )
                if c_depth
                else np.nan
            ),
            "max_setting_duration_us": (
                pd.to_numeric(
                    hist[c_dur],
                    errors="coerce",
                )
                if c_dur
                else np.nan
            ),
        }
    )

    return out, None


def expanded_merge(h4_summary):
    hist, warning = canonicalize_historical_sampled()

    if hist is None:
        return None, None, warning

    h4 = pd.DataFrame(
        {
            "source": "H4_backfill",
            "candidate_key": h4_summary["candidate_key"].astype(str),
            "topology": "H4",
            "protocol": h4_summary["protocol"].astype(str),
            "window": h4_summary["window"],
            "readout": h4_summary["readout"].astype(str),
            "r": pd.to_numeric(
                h4_summary["r"]
            ).astype(int),
            "shots_per_setting": pd.to_numeric(
                h4_summary["shots_per_setting"]
            ).astype(int),
            "rmse_mean": pd.to_numeric(
                h4_summary["rmse_mean"]
            ),
            "rmse_sd": pd.to_numeric(
                h4_summary["rmse_sd"]
            ),
            "feature_vector_n_cz": pd.to_numeric(
                h4_summary["feature_vector_n_cz"]
            ),
            "max_setting_depth": pd.to_numeric(
                h4_summary["max_setting_depth"]
            ),
            "max_setting_duration_us": pd.to_numeric(
                h4_summary[
                    "max_setting_duration_us"
                ]
            ),
        }
    )

    expanded = pd.concat(
        [hist, h4],
        ignore_index=True,
    )

    expanded["pareto"] = pareto_mask(
        expanded,
        minimize=[
            "rmse_mean",
            "feature_vector_n_cz",
            "shots_per_setting",
        ],
    )

    front = (
        expanded[expanded["pareto"]]
        .sort_values(
            [
                "rmse_mean",
                "feature_vector_n_cz",
                "shots_per_setting",
            ]
        )
        .reset_index(drop=True)
    )

    return expanded, front, None


# =============================================================================
# LATEX SNIPPET
# =============================================================================

def latex_escape(text):
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def make_latex_snippet(
    candidate_summary,
    h4_front,
    expanded_front,
    layout,
):
    best = (
        candidate_summary.sort_values(
            "best_sampled_rmse_mean"
        )
        .iloc[0]
    )

    h4_pareto_count = len(h4_front)

    if expanded_front is not None:
        h4_expanded = expanded_front[
            expanded_front["topology"].astype(str)
            == "H4"
        ]
        expanded_sentence = (
            f"The expanded H0--H4 Pareto frontier contains "
            f"{len(expanded_front)} operating cells, of which "
            f"{len(h4_expanded)} are H4."
        )
    else:
        expanded_sentence = (
            "The historical H0--H3 result files were not available at run "
            "time, so the expanded H0--H4 Pareto merge was not produced."
        )

    lines = [
        r"% Auto-generated by 10_06b_h4_week10_all_noisy_backfill.py",
        r"\subsection{Retrospective H4 Week-10 noisy-simulation backfill}",
        "",
        (
            "To close the topology-completeness gap without rerunning the "
            "frozen H0--H3 experiment, eight H4 Rule-1--Rule-4 reservoirs "
            "were evaluated on the frozen shared Kingston layout "
            f"\\texttt{{{latex_escape(layout)}}}. "
            "All candidate dynamics and readouts were inherited from the "
            "Week-9 repair; 2022--2024 remained the readout-fitting period, "
            "2025 was diagnostic hardware-transfer validation, and 2026 "
            "remained untouched."
        ),
        "",
        (
            "For each candidate, N5 direct full-quantum simulation was "
            "performed at $r\\in\\{2,3,4\\}$ with readout-assignment error "
            "disabled. Finite-shot S2 simulation then used the frozen Week-10 "
            "operating cells "
            "$\\{(2,1024),(3,512),(3,1024),(3,2048),(4,512)\\}$, "
            "with three independent repetitions per cell. "
            "No weighted score was used."
        ),
        "",
        (
            "The strongest sampled H4 operating point was "
            f"\\texttt{{{latex_escape(best['candidate_key'])}}} at "
            f"$r={int(best['best_sampled_r'])}$ and "
            f"$N={int(best['best_sampled_shots'])}$ shots/setting, with "
            f"2025 RMSE ${best['best_sampled_rmse_mean']:.6f}"
            f"\\pm{best['best_sampled_rmse_sd']:.6f}$, "
            f"{best['feature_vector_n_cz']:.0f} CZ/feature, and "
            f"max-setting depth {best['max_setting_depth']:.0f}."
        ),
        "",
        (
            f"The H4-only Pareto frontier contains {h4_pareto_count} "
            "non-dominated operating cells when minimizing validation RMSE, "
            "CZ/feature, and shots/setting. "
            + expanded_sentence
        ),
        "",
        (
            "These retrospective H4 results are used only to repair the "
            "candidate-universe completeness of the historical IBM study. "
            "They do not alter the untouched-2026 evaluation protocol."
        ),
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default=BACKEND_NAME,
    )
    parser.add_argument(
        "--rwp-batch",
        type=int,
        default=DEFAULT_RWP_BATCH,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    if args.backend != BACKEND_NAME:
        raise ValueError(
            "This repair intentionally matches the historical Kingston "
            "Week-10 candidate-wide experiment."
        )

    if args.overwrite:
        for path in OUTPUTS_TO_OVERWRITE:
            if path.exists():
                path.unlink()

    (
        candidates,
        ideal_audit,
        resources,
        layout_row,
        audit_manifest,
        physical_layout,
    ) = load_and_validate_audit()

    print("=" * 148)
    print("WEEK 10.06B — H4 COMPLETE WEEK-10 NOISY BACKFILL")
    print("=" * 148)
    print("H4 only. Frozen H0-H3 results are NOT rerun.")
    print("NO QPU jobs are submitted.")
    print("2022-2024 = readout fitting; 2025 = diagnostic hardware validation.")
    print("2026 = FROZEN / NEVER LOADED.")
    print("No weighted score.")
    print()
    print(f"H4 candidates: {len(candidates)}")
    print(f"Direct N5 cells: {len(candidates) * len(R_VALUES)}")
    print(
        "Sampled S2 summary cells: "
        f"{len(candidates) * len(OPERATING_CELLS)}"
    )
    print(
        "Sampled S2 individual runs: "
        f"{len(candidates) * len(OPERATING_CELLS) * N_REPLICATES}"
    )
    print(f"Frozen backend: {BACKEND_NAME}")
    print(f"Frozen shared H4 layout: {physical_layout}")
    print()

    # -------------------------------------------------------------------------
    # IBM backend / current calibration
    # -------------------------------------------------------------------------
    from ibm_account import get_service

    service = get_service()
    backend = service.backend(
        BACKEND_NAME,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    compact_noise, noise_info = (
        build_compact_quantum_noise_model(
            backend,
            physical_layout,
        )
    )

    print("COMPACT QUANTUM-NOISE MODEL")
    print(
        f"  mapped local errors: "
        f"{noise_info['mapped_local_quantum_errors']}"
    )
    print(
        f"  mapped default errors: "
        f"{noise_info['mapped_default_quantum_errors']}"
    )
    print(
        f"  readout assignment error included: "
        f"{noise_info['readout_error_included']}"
    )
    print(
        f"  compact noise qubits: "
        f"{noise_info['noise_qubits']}"
    )
    print()

    # Resume bookkeeping.
    if OUT_DIRECT.exists():
        done_direct = pd.read_csv(OUT_DIRECT)
    else:
        done_direct = pd.DataFrame()

    if OUT_SAMPLED_RUNS.exists():
        done_sampled = pd.read_csv(
            OUT_SAMPLED_RUNS
        )
    else:
        done_sampled = pd.DataFrame()

    # -------------------------------------------------------------------------
    # 8 candidates × r={2,3,4}
    # -------------------------------------------------------------------------
    sorted_candidates = (
        candidates.sort_values("candidate_key")
        .reset_index(drop=True)
    )

    for candidate_index, candidate_row in sorted_candidates.iterrows():
        candidate_key = str(
            candidate_row["candidate_key"]
        )

        print("=" * 148)
        print(
            f"[{candidate_index + 1}/{len(sorted_candidates)}] "
            f"{candidate_key} | "
            f"{candidate_row['protocol']} | "
            f"readout={candidate_row['test_readout']} | "
            f"rules={candidate_row['selected_by_rules']}"
        )
        print("=" * 148)

        for r in R_VALUES:
            expected_shots = sorted(
                shots
                for rr, shots in OPERATING_CELLS
                if rr == r
            )

            direct_complete = False
            sampled_complete = False

            if len(done_direct):
                direct_complete = bool(
                    np.any(
                        (
                            done_direct["candidate_key"].astype(str)
                            == candidate_key
                        )
                        & (
                            pd.to_numeric(
                                done_direct["r"],
                                errors="coerce",
                            )
                            == r
                        )
                    )
                )

            if len(done_sampled):
                gdone = done_sampled[
                    (
                        done_sampled["candidate_key"].astype(str)
                        == candidate_key
                    )
                    & (
                        pd.to_numeric(
                            done_sampled["r"],
                            errors="coerce",
                        )
                        == r
                    )
                ]

                expected_pairs = {
                    (int(shots), int(rep))
                    for shots in expected_shots
                    for rep in range(1, N_REPLICATES + 1)
                }

                got_pairs = {
                    (
                        int(row["shots_per_setting"]),
                        int(row["replicate"]),
                    )
                    for _, row in gdone.iterrows()
                }

                sampled_complete = (
                    expected_pairs
                    <= got_pairs
                )

            if direct_complete and sampled_complete:
                print(
                    f"  r={r}: already complete -> resume skip"
                )
                continue

            audit_rows = ideal_audit[
                (
                    ideal_audit["candidate_key"].astype(str)
                    == candidate_key
                )
                & (
                    pd.to_numeric(
                        ideal_audit["evaluated_r"],
                        errors="coerce",
                    ).astype(int)
                    == r
                )
            ]

            if len(audit_rows) != 1:
                raise RuntimeError(
                    f"Expected one 10.06A ideal row for "
                    f"{candidate_key}, r={r}; found {len(audit_rows)}"
                )

            audit_row = audit_rows.iloc[0]

            print(
                f"  r={r}: lambda={float(audit_row['selected_lambda']):.6g}, "
                f"CV={float(audit_row['cv_rmse']):.6f}, "
                f"ideal2025={float(audit_row['ideal_2025_rmse']):.6f}"
            )

            (
                ab_rows,
                direct_row,
                detail_rows,
                sampled_rows,
            ) = evaluate_candidate_r(
                candidate_row=candidate_row,
                audit_row=audit_row,
                resources=resources,
                backend=backend,
                compact_noise=compact_noise,
                physical_layout=physical_layout,
                rwp_batch=int(args.rwp_batch),
                candidate_index=int(candidate_index),
            )

            # If resuming a partially completed cell, avoid duplicate direct
            # rows and only add absent sampled replicate/shot combinations.
            if not direct_complete:
                append_csv(
                    OUT_AB,
                    ab_rows,
                )
                append_csv(
                    OUT_DIRECT,
                    [direct_row],
                )
                append_csv(
                    OUT_N5_FEATURES,
                    detail_rows,
                )

                print(
                    f"    N5: RMSE={direct_row['rmse']:.6f} | "
                    f"ideal={direct_row['ideal_2025_rmse']:.6f} | "
                    f"delta={direct_row['delta_rmse_vs_same_r_ideal']:+.6f} | "
                    f"corr={direct_row['prediction_corr_vs_same_r_ideal']:.4f}"
                )
                print(
                    f"    A/B feature audit max RMSE="
                    f"{max(x['feature_rmse_A_vs_B'] for x in ab_rows):.3e}"
                )

            existing_pairs = set()

            if len(done_sampled):
                gd = done_sampled[
                    (
                        done_sampled["candidate_key"].astype(str)
                        == candidate_key
                    )
                    & (
                        pd.to_numeric(
                            done_sampled["r"],
                            errors="coerce",
                        )
                        == r
                    )
                ]

                existing_pairs = {
                    (
                        int(row["shots_per_setting"]),
                        int(row["replicate"]),
                    )
                    for _, row in gd.iterrows()
                }

            new_sampled = [
                row
                for row in sampled_rows
                if (
                    int(row["shots_per_setting"]),
                    int(row["replicate"]),
                )
                not in existing_pairs
            ]

            append_csv(
                OUT_SAMPLED_RUNS,
                new_sampled,
            )

            temp = pd.DataFrame(sampled_rows)

            for shots in expected_shots:
                g = temp[
                    temp["shots_per_setting"]
                    == shots
                ]
                print(
                    f"    S2 r={r}, N={shots}: "
                    f"RMSE={g['rmse'].mean():.6f}"
                    f"±{g['rmse'].std(ddof=1):.6f} | "
                    f"deltaN5={g['delta_rmse_vs_full_quantum_direct'].mean():+.6f}"
                )

            # Refresh resume tables after every completed r cell.
            done_direct = pd.read_csv(OUT_DIRECT)
            done_sampled = pd.read_csv(
                OUT_SAMPLED_RUNS
            )

            print()

    # -------------------------------------------------------------------------
    # Build final H4 summaries from persisted rows.
    # -------------------------------------------------------------------------
    direct_df = pd.read_csv(OUT_DIRECT)
    sampled_runs_df = pd.read_csv(
        OUT_SAMPLED_RUNS
    )
    ab_df = pd.read_csv(OUT_AB)

    expected_direct = 8 * 3
    expected_sample_runs = (
        8
        * len(OPERATING_CELLS)
        * N_REPLICATES
    )

    if len(direct_df) != expected_direct:
        raise RuntimeError(
            f"Expected {expected_direct} direct rows; got {len(direct_df)}"
        )

    if len(sampled_runs_df) != expected_sample_runs:
        raise RuntimeError(
            f"Expected {expected_sample_runs} sampled runs; "
            f"got {len(sampled_runs_df)}"
        )

    if not ab_df["pass"].astype(bool).all():
        raise RuntimeError(
            "At least one persisted A/B audit row failed."
        )

    sampled_summary = summarize_sampled(
        sampled_runs_df
    )
    sampled_summary.to_csv(
        OUT_SAMPLED_SUMMARY,
        index=False,
    )

    candidate_summary = h4_candidate_summary(
        direct_df,
        sampled_summary,
    )
    candidate_summary.to_csv(
        OUT_CANDIDATE_SUMMARY,
        index=False,
    )

    h4_front = h4_pareto(
        sampled_summary
    )
    h4_front.to_csv(
        OUT_H4_PARETO,
        index=False,
    )

    expanded, expanded_front, merge_warning = (
        expanded_merge(
            sampled_summary
        )
    )

    if expanded is not None:
        expanded.to_csv(
            OUT_EXPANDED_SUMMARY,
            index=False,
        )
        expanded_front.to_csv(
            OUT_EXPANDED_PARETO,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 148)
    print("10.06B H4 WEEK-10 BACKFILL — FINAL SUMMARY")
    print("=" * 148)

    print()
    print("H4 BEST DIRECT/SAMPLED RESULT PER CANDIDATE")
    print(
        candidate_summary[
            [
                "candidate_key",
                "selected_by_rules",
                "protocol",
                "window",
                "readout",
                "best_direct_r",
                "best_direct_n5_rmse",
                "best_sampled_r",
                "best_sampled_shots",
                "best_sampled_rmse_mean",
                "best_sampled_rmse_sd",
                "feature_vector_n_cz",
                "max_setting_depth",
            ]
        ].to_string(index=False)
    )

    print()
    print("H4-ONLY PARETO FRONT")
    print(
        h4_front[
            [
                "candidate_key",
                "protocol",
                "window",
                "readout",
                "r",
                "shots_per_setting",
                "rmse_mean",
                "rmse_sd",
                "feature_vector_n_cz",
                "max_setting_depth",
            ]
        ].to_string(index=False)
    )

    if expanded_front is not None:
        print()
        print("EXPANDED FROZEN H0-H3 + NEW H4 PARETO FRONT")
        print(
            expanded_front[
                [
                    "source",
                    "candidate_key",
                    "topology",
                    "r",
                    "shots_per_setting",
                    "rmse_mean",
                    "rmse_sd",
                    "feature_vector_n_cz",
                    "max_setting_depth",
                ]
            ].to_string(index=False)
        )

        h4_on_expanded = expanded_front[
            expanded_front["topology"].astype(str)
            == "H4"
        ]

        print()
        print(
            f"H4 cells on expanded Pareto frontier: "
            f"{len(h4_on_expanded)}"
        )
    else:
        print()
        print("EXPANDED MERGE WARNING:")
        print(f"  {merge_warning}")

    # -------------------------------------------------------------------------
    # Manifest + LaTeX snippet
    # -------------------------------------------------------------------------
    best_h4 = (
        candidate_summary.sort_values(
            "best_sampled_rmse_mean"
        )
        .iloc[0]
    )

    manifest = {
        "step": "10.06B_H4_complete_week10_noisy_backfill",
        "timestamp_utc": utc_now(),
        "backend": BACKEND_NAME,
        "topology": "H4",
        "candidate_count": 8,
        "physical_layout": list(physical_layout),
        "layout_source": str(AUDIT_LAYOUT_FILE),
        "audit_source": str(AUDIT_MANIFEST_FILE),
        "r_values_direct": R_VALUES,
        "operating_cells_sampled": [
            [int(r), int(n)]
            for r, n in OPERATING_CELLS
        ],
        "sample_replicates": N_REPLICATES,
        "direct_rows": int(len(direct_df)),
        "sampled_runs": int(
            len(sampled_runs_df)
        ),
        "sampled_summary_rows": int(
            len(sampled_summary)
        ),
        "h4_pareto_rows": int(
            len(h4_front)
        ),
        "ab_rows": int(len(ab_df)),
        "ab_all_pass": bool(
            ab_df["pass"].astype(bool).all()
        ),
        "ab_max_feature_rmse": float(
            pd.to_numeric(
                ab_df["feature_rmse_A_vs_B"],
                errors="coerce",
            ).max()
        ),
        "compact_noise_model": noise_info,
        "readout_assignment_error": False,
        "error_mitigation": False,
        "selection_period": "2022-2024",
        "validation_2025": "diagnostic hardware-transfer validation",
        "test_2026_used": False,
        "qpu_jobs_submitted": 0,
        "historical_h0_h3_rerun": False,
        "historical_merge_performed": bool(
            expanded_front is not None
        ),
        "expanded_operating_rows": (
            int(len(expanded))
            if expanded is not None
            else None
        ),
        "expanded_pareto_rows": (
            int(len(expanded_front))
            if expanded_front is not None
            else None
        ),
        "expanded_h4_pareto_rows": (
            int(
                np.sum(
                    expanded_front["topology"]
                    .astype(str)
                    == "H4"
                )
            )
            if expanded_front is not None
            else None
        ),
        "best_h4_sampled_candidate": str(
            best_h4["candidate_key"]
        ),
        "best_h4_sampled_r": int(
            best_h4["best_sampled_r"]
        ),
        "best_h4_sampled_shots": int(
            best_h4["best_sampled_shots"]
        ),
        "best_h4_sampled_rmse_mean": float(
            best_h4["best_sampled_rmse_mean"]
        ),
        "best_h4_sampled_rmse_sd": float(
            best_h4["best_sampled_rmse_sd"]
        ),
        "merge_warning": merge_warning,
        "next_step": (
            "Interpret the expanded Week-10 Pareto/handoff. "
            "Only if H4 changes the hardware finalist set should a "
            "targeted Week-11 H4 QPU experiment be prepared."
        ),
    }

    OUT_MANIFEST.write_text(
        json.dumps(
            json_safe(manifest),
            indent=2,
        ),
        encoding="utf-8",
    )

    latex = make_latex_snippet(
        candidate_summary,
        h4_front,
        expanded_front,
        physical_layout,
    )
    OUT_LATEX.write_text(
        latex,
        encoding="utf-8",
    )

    print()
    print("Saved:")
    for path in [
        OUT_AB,
        OUT_DIRECT,
        OUT_N5_FEATURES,
        OUT_SAMPLED_RUNS,
        OUT_SAMPLED_SUMMARY,
        OUT_CANDIDATE_SUMMARY,
        OUT_H4_PARETO,
        *(
            [
                OUT_EXPANDED_SUMMARY,
                OUT_EXPANDED_PARETO,
            ]
            if expanded_front is not None
            else []
        ),
        OUT_MANIFEST,
        OUT_LATEX,
    ]:
        print(f"  {path}")

    print()
    print("NO QPU jobs submitted. 2026 untouched.")
    print(
        "STOP HERE. Send the console output, manifest, candidate summary, "
        "H4 Pareto, and expanded Pareto (if produced). "
        "We will decide whether an H4 Week-11 QPU backfill is justified "
        "only after interpreting these results."
    )


if __name__ == "__main__":
    main()
