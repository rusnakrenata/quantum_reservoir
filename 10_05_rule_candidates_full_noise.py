#!/usr/bin/env python
"""
10.05 — Rule-selected candidate validation under current Kingston noise (FIXED V3 / ENRICHED MANIFEST)
======================================================================

ONE SCRIPT: candidate resolution + audit + noisy testing.\n\nThis FIXED version contains its own exact logical CONT feature-bank\nimplementation and does not require a CONT helper from 09_04_rwp_common.py.

Candidate set
-------------
Reads:
    results/10_05_rule1_to_rule5_candidate_manifest.csv

The manifest contains the 25 UNIQUE reservoir representatives selected by
Rules 1–4. Rule 5 is NOT used as a candidate source.

Physical layouts
----------------
Reads:
    results/10_05_current_kingston_layouts.csv

These are the FRESH/current Kingston topology-specific layouts produced by
10_05A. Historical Rule-5 layouts in the article/manifest are ignored.

Operating points
----------------
    (r, N) = (2,1024), (3,512), (3,1024), (3,2048), (4,512)

where N is shots PER measurement setting.

Noise / measurement modes
-------------------------
For each candidate and unique r={2,3,4}:

N5_FULL_QUANTUM_DIRECT
    current Kingston 1Q + 2Q gate noise + T1/T2 thermal relaxation,
    readout assignment error OFF,
    direct expectation values.

S2_FULL_QUANTUM_SAMPLED
    same current full-quantum noise,
    readout assignment error OFF,
    finite-shot Pauli measurement.

The lower shot levels are nested prefixes where possible.

IMPORTANT: CONT and RWP are simulated differently
--------------------------------------------------
RWP:
    Each endpoint is replayed from a reset memory state exactly as in the
    preceding Week-10 RWP scripts.

CONT:
    Memory is carried through the complete chronology.
    The noisy CONT state is evolved ONCE through the chronological sequence.
    Direct expectations and pre-measurement density matrices are saved at
    validation endpoints. Finite-shot measurements are then made on independent
    copies of those saved states, so measurement does NOT feed back into the
    CONT memory trajectory.

Candidate dynamics: J, hx, hy, alpha, dt
-----------------------------------------
RWP:
    DO NOT read the topology-specific J columns from
    results/09_04c_rwp_all_trials.csv. Those columns are not a safe
    reconstruction source for H1/H2/H3.

    Instead the exact RWP dynamics are reconstructed deterministically from:
        results/09_03c_forecast_best_135.csv
            -> clean topology baseline
        + results/09_04c_rwp_all_trials.csv
            -> topology, W, trial_search_id, scalar audit values
        + common.stage_c_candidates(...)

    This is the same safe reconstruction method already used in Week 10.1.

CONT:
    The manifest row is resolved back to the exact Week-9 CONT source row.
    The preferred source is:
        results/09_05a_cont_conditioning_all.csv
    with fallback to:
        results/09_03c_forecast_best_135.csv
        results/09_03c_r1_r2_r3_comparison.csv
        results/09_05a_cont_revised_readout_cv_winners.csv
        results/09_05a_cont_revised_hardware_aware_winners.csv

    The candidate dictionary is reconstructed from that exact source row via
    common.candidate_from_row(...) when possible. If a derived CONT table lacks
    dynamics columns, the script follows candidate/search identifiers back to
    09_03c_forecast_best_135.csv.

The script writes the resolved alpha/dt/hx/hy/J values to:
    results/10_05_resolved_dynamics.csv

so the parameter provenance is explicit and auditable.

NO QPU JOBS.
NO MITIGATION.
READOUT ASSIGNMENT ERROR OFF.
2026 remains frozen.

Required files
--------------
ibm_account.py
09_04_rwp_common.py

results/10_05_rule1_to_rule5_candidate_manifest.csv
results/10_05_current_kingston_layouts.csv
results/09_03c_forecast_best_135.csv
results/09_04c_rwp_all_trials.csv
results/09_05a_cont_conditioning_all.csv   (preferred for CONT)

Outputs
-------
results/10_05_reference.json
results/10_05_resolved_dynamics.csv
results/10_05_direct_runs.csv
results/10_05_sampled_runs.csv
results/10_05_sampled_summary.csv
results/10_05_resource_grid.csv
results/10_05_pareto_front.csv

Quick validation:
    python 10_05_rule_candidates_full_noise.py --max-candidates 2 --sampling-reps 1

Full run:
    python 10_05_rule_candidates_full_noise.py --sampling-reps 3
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import (
    DensityMatrix,
    Pauli,
    SparsePauliOp,
)
from qiskit.transpiler import CouplingMap

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

from ibm_account import get_service


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"

MANIFEST_FILE = RESULTS / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"
CURRENT_LAYOUTS_FILE = RESULTS / "10_05_current_kingston_layouts.csv"

CONT_FILES = [
    RESULTS / "09_05a_cont_conditioning_all.csv",
    RESULTS / "09_03c_forecast_best_135.csv",
    RESULTS / "09_03c_r1_r2_r3_comparison.csv",
    RESULTS / "09_05a_cont_revised_readout_cv_winners.csv",
    RESULTS / "09_05a_cont_revised_hardware_aware_winners.csv",
]

CONT_BASE_FILE = RESULTS / "09_03c_forecast_best_135.csv"
RWP_TRIAL_FILE = RESULTS / "09_04c_rwp_all_trials.csv"

BACKEND_NAME = "ibm_kingston"

OPERATING_POINTS = [
    (2, 1024),
    (3, 512),
    (3, 1024),
    (3, 2048),
    (4, 512),
]

R_VALUES = [2, 3, 4]

SHOT_LEVELS_BY_R = {
    2: [1024],
    3: [512, 1024, 2048],
    4: [512],
}

DEFAULT_REPS = 3
DEFAULT_BATCH = 64
DEFAULT_SEED = 20260912

OPT_LEVEL = 1
SEED_TRANSPILE = 42

CV_TOL = 8e-7

DIRECT_OUT = RESULTS / "10_05_direct_runs.csv"
SAMPLED_OUT = RESULTS / "10_05_sampled_runs.csv"


# =============================================================================
# IMPORT COMMON
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_1005_common")


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def rmse(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p)))


def bias(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(p - y))


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def json_safe(x):
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    if isinstance(x, np.generic):
        return json_safe(x.item())
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    return str(x)


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def parse_layout(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    if isinstance(value, str):
        try:
            return [int(x) for x in json.loads(value)]
        except Exception:
            return [int(x) for x in ast.literal_eval(value)]
    raise TypeError(type(value))


def clone_candidate(c):
    try:
        return common._clone_candidate(c)
    except Exception:
        z = dict(c)
        z["J"] = dict(c["J"])
        return z


def append_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# =============================================================================
# READOUT DEFINITIONS
# =============================================================================

def features_for_readout(readout):
    if readout not in common.READOUT_FEATURES:
        raise KeyError(
            f"Readout {readout!r} missing from {COMMON_FILE}. "
            "Use the revised post-QPU 09_04_rwp_common.py."
        )
    return list(common.READOUT_FEATURES[readout])


def settings_for_readout(readout):
    if readout == "XZ_injection":
        return ["XXXXZZ", "ZZZZZZ"]
    if readout == "XZinj_dropX3":
        return ["XXXZZZ", "ZZZZZZ"]
    if readout == "XZinj_dropX3_plus_YX45":
        return ["XXXZYX", "ZZZZZZ"]
    if readout == "XZinj_plus_YX45":
        return ["XXXXYX", "ZZZZZZ"]
    if readout == "XYZ_all":
        return ["XXXXXX", "YYYYYY", "ZZZZZZ"]
    raise KeyError(readout)


def feature_setting(readout, feature):
    settings = settings_for_readout(readout)

    if feature == "YX45":
        for s in settings:
            if s[4] == "Y" and s[5] == "X":
                return s
        raise RuntimeError(f"No YX45 measurement setting in {readout}")

    axis = feature[0]
    q = int(feature[1:])

    for s in settings:
        if s[q] == axis:
            return s

    raise RuntimeError(f"No setting for {feature} in {readout}")


def sparse_observable(feature):
    if feature == "YX45":
        return SparsePauliOp.from_sparse_list(
            [("YX", [4, 5], 1.0)],
            num_qubits=6,
        )

    axis = feature[0]
    q = int(feature[1:])

    return SparsePauliOp.from_sparse_list(
        [(axis, [q], 1.0)],
        num_qubits=6,
    )


# =============================================================================
# LOAD MANIFEST / FRESH LAYOUTS
# =============================================================================

def load_manifest():
    """
    Load the ENRICHED 25-candidate manifest.

    The enriched CSV is now the authoritative source for:
        alpha, dt, hx, hy, J, original r,
        test readout, original-r Ridge lambda,
        source CV/validation diagnostics and provenance.

    Week-9 result CSVs are no longer needed to reconstruct J/hx/hy/alpha/dt
    inside this 10.05 tester.
    """
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"{MANIFEST_FILE} missing. Put the enriched 25-candidate CSV "
            "into results/ with this exact filename."
        )

    df = pd.read_csv(MANIFEST_FILE)

    required = {
        "candidate_key",
        "protocol",
        "topology",
        "original_r",
        "test_readout",
        "selected_by_rules",
        "alpha",
        "dt",
        "hx",
        "hy",
        "J_json",
        "source_selected_lambda",
        "source_readout_cv_rmse",
        "source_readout_cv_sd",
        "table_cv_rmse",
        "dynamics_provenance",
        "lambda_provenance",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Enriched candidate manifest is missing columns: "
            f"{sorted(missing)}"
        )

    if len(df) != 25:
        raise RuntimeError(
            f"Expected 25 enriched candidate rows, got {len(df)}"
        )

    # Preserve the historical internal variable name used by the tester.
    df = df.copy()
    df["readout"] = df["test_readout"].astype(str)

    return df

def load_current_layouts():
    if not CURRENT_LAYOUTS_FILE.exists():
        raise FileNotFoundError(
            f"{CURRENT_LAYOUTS_FILE} missing. Run 10_05A first."
        )

    df = pd.read_csv(CURRENT_LAYOUTS_FILE)

    layouts = {}

    for topology in ["H0", "H1", "H2", "H3"]:
        g = df[df["topology"].astype(str) == topology]
        if len(g) != 1:
            raise RuntimeError(
                f"Expected one current Kingston layout for {topology}; got {len(g)}"
            )

        layouts[topology] = parse_layout(g.iloc[0]["layout"])

    return layouts


# =============================================================================
# EXACT CANDIDATE RESOLUTION
# =============================================================================

def _readout_equal(a, b):
    return norm(a) == norm(b)


def find_exact_rwp_row(mrow):
    if not RWP_TRIAL_FILE.exists():
        raise FileNotFoundError(RWP_TRIAL_FILE)

    df = pd.read_csv(RWP_TRIAL_FILE)

    H = str(mrow["topology"])
    W = int(round(float(mrow["window"])))
    readout = str(mrow["readout"])
    target_cv = float(mrow["table_cv_rmse"])
    original_r = int(mrow["original_r"])

    mask = (
        (df["topology"].astype(str) == H)
        & (pd.to_numeric(df["window"], errors="coerce") == W)
        & (df["readout"].astype(str).map(norm) == norm(readout))
        & (
            np.abs(
                pd.to_numeric(df["cv_rmse"], errors="coerce")
                - target_cv
            )
            <= CV_TOL
        )
    )

    if "r" in df.columns:
        mask &= (
            pd.to_numeric(df["r"], errors="coerce")
            == original_r
        )

    g = df.loc[mask].copy()

    if len(g) != 1:
        raise RuntimeError(
            f"RWP resolution failed for {mrow['candidate_key']}: "
            f"expected 1 row, found {len(g)}."
        )

    return g.iloc[0]


def reconstruct_rwp_candidate(mrow, baselines):
    """
    SAFE Stage-C reconstruction.

    IMPORTANT:
    09_04c_rwp_all_trials.csv J columns are not used for H1/H2/H3.
    """
    row = find_exact_rwp_row(mrow)

    H = str(row["topology"])
    W = int(row["window"])
    search_id = str(row["trial_search_id"])

    if H not in baselines:
        raise KeyError(H)

    base = clone_candidate(baselines[H])
    base["candidate_id"] = f"{H}_W{W:02d}_Aparent"
    base["window"] = W

    regenerated = {
        str(c["search_id"]): c
        for c in common.stage_c_candidates(base)
    }

    if search_id not in regenerated:
        raise RuntimeError(
            f"Cannot regenerate {H}, W={W}, search_id={search_id}"
        )

    candidate = regenerated[search_id]

    # Audit safe scalar values from CSV.
    for key in ["alpha", "dt", "hx", "hy"]:
        if key in row.index and pd.notna(row[key]):
            if not np.isclose(
                float(candidate[key]),
                float(row[key]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"RWP scalar audit failed {mrow['candidate_key']} {key}: "
                    f"regen={candidate[key]}, csv={row[key]}"
                )

    if int(candidate["r"]) != int(row["r"]):
        raise RuntimeError(
            f"RWP r audit failed {mrow['candidate_key']}"
        )

    candidate["window"] = W
    candidate["protocol"] = "RWP"

    return candidate, row.to_dict(), "stage_c_regenerated"


def candidate_from_row_best_effort(row):
    # Preferred path: project helper.
    try:
        c = common.candidate_from_row(row)
        if "J" in c and "hx" in c:
            return c
    except Exception:
        pass

    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    c = {
        "topology": str(d["topology"]),
        "alpha": float(d["alpha"]),
        "dt": float(d["dt"]),
        "r": int(float(d["r"])),
        "hx": float(d["hx"]),
        "hy": float(d.get("hy", 0.0)),
    }

    edges = common.TOPOLOGY_EDGES[c["topology"]]
    J = {}

    for i, j in edges:
        candidates = [
            f"J{i}{j}",
            f"J{j}{i}",
            f"J_{i}_{j}",
            f"J_{j}_{i}",
            f"j{i}{j}",
            f"j{j}{i}",
        ]

        found = None

        for col in candidates:
            if col in d and pd.notna(d[col]):
                found = float(d[col])
                break

        if found is None:
            raise RuntimeError(
                f"Cannot extract J({i},{j}) from CONT source row."
            )

        J[(i, j)] = found

    c["J"] = J
    return c


def find_cont_source_row(mrow):
    target_cv = float(mrow["table_cv_rmse"])
    H = str(mrow["topology"])
    readout = str(mrow["readout"])
    original_r = int(mrow["original_r"])

    candidates = []

    for priority, path in enumerate(CONT_FILES):
        if not path.exists():
            continue

        df = pd.read_csv(path)

        # Find likely columns.
        cv_col = None
        for col in df.columns:
            if norm(col) in {
                "cvrmse",
                "cvrmsemean",
                "rmsecv",
                "forecastcvrmse",
            }:
                cv_col = col
                break

        if cv_col is None:
            continue

        mask = (
            np.abs(
                pd.to_numeric(df[cv_col], errors="coerce")
                - target_cv
            )
            <= CV_TOL
        )

        if "topology" in df.columns:
            mask &= df["topology"].astype(str).eq(H)

        if "readout" in df.columns:
            mask &= df["readout"].astype(str).map(norm).eq(norm(readout))

        if "r" in df.columns:
            mask &= pd.to_numeric(df["r"], errors="coerce").eq(original_r)

        g = df.loc[mask]

        for idx, row in g.iterrows():
            candidates.append(
                (
                    priority,
                    path,
                    idx,
                    row,
                )
            )

    if not candidates:
        raise RuntimeError(
            f"CONT source row unresolved for {mrow['candidate_key']}"
        )

    # Prefer the earliest declared source.
    candidates.sort(key=lambda x: x[0])

    # If multiple rows tie in the same source file, stop rather than guess.
    best_priority = candidates[0][0]
    tied = [x for x in candidates if x[0] == best_priority]

    if len(tied) != 1:
        raise RuntimeError(
            f"CONT source row ambiguous for {mrow['candidate_key']}: "
            f"{len(tied)} matches in {tied[0][1].name}"
        )

    return tied[0]


def find_base_row_by_identifier(source_row):
    if not CONT_BASE_FILE.exists():
        raise FileNotFoundError(CONT_BASE_FILE)

    base = pd.read_csv(CONT_BASE_FILE)

    # Try strong identifiers first.
    id_cols = [
        "candidate_id",
        "search_id",
        "dynamics_id",
        "config_id",
    ]

    for col in id_cols:
        if (
            col in source_row.index
            and col in base.columns
            and pd.notna(source_row[col])
        ):
            g = base[
                base[col].astype(str)
                == str(source_row[col])
            ]

            if len(g) == 1:
                return g.iloc[0]

    # If the exact source row already has dynamics, return it.
    needed = {"alpha", "dt", "r", "hx"}
    if needed.issubset(set(source_row.index)):
        return source_row

    return None


def reconstruct_cont_candidate(mrow):
    priority, path, idx, row = find_cont_source_row(mrow)

    base_row = find_base_row_by_identifier(row)

    if base_row is None:
        raise RuntimeError(
            f"CONT dynamics unresolved for {mrow['candidate_key']}. "
            f"Matched {path.name}[{idx}] but could not link to a full dynamics row."
        )

    candidate = candidate_from_row_best_effort(base_row)
    candidate["protocol"] = "CONT"

    return candidate, row.to_dict(), path.name


def resolve_all_candidates(manifest):
    """
    Build all 25 reservoir candidates DIRECTLY from the enriched manifest.

    This removes the previous runtime dependency on the Week-9 source CSVs.
    The enriched manifest already contains the audited exact dynamics.

    J_json uses labels such as J01, J12, J24, J45.  They are mapped onto the
    exact logical edges declared for the candidate topology.
    """
    resolved = {}
    audit_rows = []

    for _, mrow in manifest.iterrows():
        key = str(mrow["candidate_key"])
        protocol = str(mrow["protocol"]).upper()
        topology = str(mrow["topology"])

        raw_J = json.loads(str(mrow["J_json"]))

        J = {}

        for edge in common.TOPOLOGY_EDGES[topology]:
            i, j = edge
            label = f"J{i}{j}"
            reverse = f"J{j}{i}"

            if label in raw_J:
                value = raw_J[label]
            elif reverse in raw_J:
                value = raw_J[reverse]
            else:
                raise RuntimeError(
                    f"{key}: enriched manifest J_json is missing the "
                    f"required topology edge {label}."
                )

            J[edge] = float(value)

        candidate = {
            "topology": topology,
            "alpha": float(mrow["alpha"]),
            "dt": float(mrow["dt"]),
            "r": int(mrow["original_r"]),
            "hx": float(mrow["hx"]),
            "hy": float(mrow["hy"]),
            "J": J,
            "protocol": protocol,
            "candidate_id": key,
        }

        if protocol == "RWP":
            if pd.isna(mrow["window"]):
                raise RuntimeError(
                    f"{key}: RWP candidate has no window in enriched manifest."
                )
            candidate["window"] = int(float(mrow["window"]))

        resolved[key] = {
            "manifest": mrow.to_dict(),
            "candidate": candidate,
            "source": {
                "dynamics_provenance": str(
                    mrow["dynamics_provenance"]
                ),
                "lambda_provenance": str(
                    mrow["lambda_provenance"]
                ),
            },
            "provenance": "ENRICHED_MANIFEST",
        }

        audit_rows.append({
            "candidate_key": key,
            "protocol": protocol,
            "topology": topology,
            "window": (
                np.nan
                if pd.isna(mrow.get("window", np.nan))
                else int(float(mrow["window"]))
            ),
            "test_readout": str(mrow["test_readout"]),
            "original_r": int(mrow["original_r"]),
            "alpha": float(candidate["alpha"]),
            "dt": float(candidate["dt"]),
            "hx": float(candidate["hx"]),
            "hy": float(candidate["hy"]),
            "J": json.dumps(
                {
                    f"J{i}{j}": float(v)
                    for (i, j), v in candidate["J"].items()
                },
                separators=(",", ":"),
            ),
            "source_selected_lambda": float(
                mrow["source_selected_lambda"]
            ),
            "source_readout_cv_rmse": float(
                mrow["source_readout_cv_rmse"]
            ),
            "source_readout_cv_sd": float(
                mrow["source_readout_cv_sd"]
            ),
            "table_cv_rmse": float(
                mrow["table_cv_rmse"]
            ),
            "dynamics_provenance": str(
                mrow["dynamics_provenance"]
            ),
            "lambda_provenance": str(
                mrow["lambda_provenance"]
            ),
        })

    pd.DataFrame(audit_rows).to_csv(
        RESULTS / "10_05_resolved_dynamics.csv",
        index=False,
    )

    return resolved


# =============================================================================
# IDEAL FEATURES / RIDGE
# =============================================================================

def fit_readout_from_bank(
    X,
    endpoints,
    y_all,
):
    y_endpoint = y_all[endpoints]

    train_mask = endpoints < common.N_TRAIN
    val_mask = endpoints >= common.N_TRAIN

    X_train = X[train_mask]
    y_train = y_endpoint[train_mask]

    X_val = X[val_mask]
    y_val = y_endpoint[val_mask]
    val_endpoints = endpoints[val_mask]

    if len(y_val) != common.N_VAL:
        raise RuntimeError(
            f"Expected {common.N_VAL} validation rows, got {len(y_val)}"
        )

    selected_lambda, _, summary = (
        common.select_lambda_training_only(
            X_train,
            y_train,
            n_splits=5,
        )
    )

    model, scaler, keep = (
        common.fit_scaled_ridge(
            X_train,
            y_train,
            selected_lambda,
        )
    )

    pred = common.predict_scaled_ridge(
        model,
        scaler,
        keep,
        X_val,
    )

    return {
        "X_val_ideal": X_val,
        "y_val": y_val,
        "val_endpoints": val_endpoints,
        "model": model,
        "scaler": scaler,
        "keep": keep,
        "selected_lambda": float(selected_lambda),
        "cv_rmse": float(summary.iloc[0]["cv_rmse_mean"]),
        "cv_sd": float(summary.iloc[0]["cv_rmse_std"]),
        "ideal_rmse": rmse(y_val, pred),
        "pred_ideal": pred,
    }


def ideal_rwp_reference(
    candidate,
    readout,
    work_tv,
    cols,
    y_all,
):
    W = int(candidate["window"])
    angles = common.make_angles(
        work_tv,
        cols,
        float(candidate["alpha"]),
    )

    channels = common.build_channels(
        candidate,
        angles,
    )

    endpoints, master = (
        common.rwp_master_feature_bank_from_channels(
            channels,
            W,
        )
    )

    features = features_for_readout(readout)
    X = master[features].to_numpy(dtype=float)

    out = fit_readout_from_bank(
        X,
        endpoints,
        y_all,
    )

    out["angles"] = angles
    out["features"] = features
    return out


def build_ideal_cont_feature_bank(
    candidate,
    angles,
    features,
):
    """
    Exact Week-9 CONT logical feature bank.

    IMPORTANT FIX
    -------------
    The previous 10.05 version rebuilt CONT with a Qiskit density-matrix
    circuit and initialized the memory subsystem as I/4.  The Week-9 CONT
    pipeline did NOT use that initialization.  It used the same reduced-state
    channel recurrence as the original QRC implementation, starting from
    q4,q5 = |00><00|.

    To reproduce the published Week-9 CV values exactly, use the original
    reduced-state recurrence already exposed through 09_04_rwp_common.py:

        A_t = input-dependent memory channel
        rho_M(0) = |00><00|
        (rho_I(t), rho_M(t)) =
            qrc.final_reduced_states(A_t, rho_M(t-1))

    This is the authoritative ideal CONT reconstruction for the CV/Ridge
    audit.  No measurement is fed back into the reservoir.
    """
    A_list = common.build_channels(
        candidate,
        angles,
    )

    # The qrc module imported by 09_04_rwp_common.py is the original Week-7/9
    # QRC implementation used to generate the CONT result tables.
    qrc = common.qrc

    rho_m = qrc.memory_zero_density()
    rows = []

    for A_t in A_list:
        rho_i, rho_m = qrc.final_reduced_states(
            A_t,
            rho_m,
        )

        full_row = common.reduced_feature_row(
            rho_i,
            rho_m,
        )

        rows.append(
            {
                feature: float(full_row[feature])
                for feature in features
            }
        )

    endpoints = np.arange(
        len(rows),
        dtype=int,
    )

    bank = pd.DataFrame(
        rows,
        columns=features,
    )

    return endpoints, bank

def ideal_cont_reference(
    candidate,
    readout,
    work_tv,
    cols,
    y_all,
):
    angles = common.make_angles(
        work_tv,
        cols,
        float(candidate["alpha"]),
    )

    features = features_for_readout(
        readout
    )

    endpoints, bank = build_ideal_cont_feature_bank(
        candidate,
        angles,
        features,
    )

    X = bank[
        features
    ].to_numpy(
        dtype=float
    )

    out = fit_readout_from_bank(
        X,
        endpoints,
        y_all,
    )

    out["angles"] = angles
    out["features"] = features

    return out


# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def append_step(
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
        for i, j in common.TOPOLOGY_EDGES[
            candidate["topology"]
        ]:
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


def build_rwp_core(candidate):
    W = int(candidate["window"])

    q = QuantumRegister(6, "q")
    qc = QuantumCircuit(q)

    pars = ParameterVector(
        "theta",
        4 * W,
    )

    for step in range(W):
        p = [
            pars[4 * step + k]
            for k in range(4)
        ]

        append_step(
            qc,
            candidate,
            p,
            reset_injection=(step > 0),
        )

    return qc, pars


def build_rwp_measured(
    candidate,
    setting,
):
    W = int(candidate["window"])

    q = QuantumRegister(6, "q")
    c = ClassicalRegister(6, "m")
    qc = QuantumCircuit(q, c)

    pars = ParameterVector(
        "theta",
        4 * W,
    )

    for step in range(W):
        p = [
            pars[4 * step + k]
            for k in range(4)
        ]

        append_step(
            qc,
            candidate,
            p,
            reset_injection=(step > 0),
        )

    append_basis_rotation(qc, setting)

    for k in range(6):
        qc.measure(k, k)

    return qc, pars


def build_cont_step(candidate):
    q = QuantumRegister(6, "q")
    qc = QuantumCircuit(q)

    pars = ParameterVector(
        "theta",
        4,
    )

    append_step(
        qc,
        candidate,
        pars,
        reset_injection=True,
    )

    return qc, pars


def append_basis_rotation(qc, setting):
    for q, axis in enumerate(setting):
        if axis == "X":
            qc.h(q)
        elif axis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif axis == "Z":
            pass
        else:
            raise ValueError(axis)


def bind_rwp(
    circuit,
    pars,
    angles,
    endpoint,
    W,
):
    start = int(endpoint) - int(W) + 1

    values = {}

    for step, idx in enumerate(
        range(start, int(endpoint) + 1)
    ):
        for q in range(4):
            values[
                pars[4 * step + q]
            ] = float(
                angles[idx, q]
            )

    return circuit.assign_parameters(
        values,
        inplace=False,
    )


# =============================================================================
# RWP PHYSICAL NOISE
# =============================================================================

def prepare_rwp_physical(
    backend,
    candidate,
    readout,
    layout,
):
    logical_core, core_pars = build_rwp_core(
        candidate
    )

    core = transpile(
        logical_core,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
    )

    backend.check_faulty(core)

    if int(core.count_ops().get("swap", 0)) != 0:
        raise RuntimeError("RWP core introduced SWAPs.")

    mapped_obs = [
        sparse_observable(f).apply_layout(
            core.layout
        )
        for f in features_for_readout(readout)
    ]

    measured = {}
    measured_pars = {}

    for setting in settings_for_readout(readout):
        logical, pars = build_rwp_measured(
            candidate,
            setting,
        )

        qc = transpile(
            logical,
            backend=backend,
            initial_layout=layout,
            routing_method="none",
            optimization_level=OPT_LEVEL,
            seed_transpiler=SEED_TRANSPILE,
        )

        backend.check_faulty(qc)

        if int(qc.count_ops().get("swap", 0)) != 0:
            raise RuntimeError(
                f"RWP setting {setting} introduced SWAPs."
            )

        measured[setting] = qc
        measured_pars[setting] = pars

    rows = []

    for setting, qc in measured.items():
        ops = qc.count_ops()
        try:
            duration_us = (
                float(
                    qc.estimate_duration(
                        backend.target,
                        unit="s",
                    )
                )
                * 1e6
            )
        except Exception:
            duration_us = np.nan

        rows.append({
            "setting": setting,
            "n_cz": int(ops.get("cz", 0)),
            "depth": int(qc.depth()),
            "duration_us": duration_us,
        })

    rdf = pd.DataFrame(rows)

    return {
        "core": core,
        "core_pars": core_pars,
        "mapped_obs": mapped_obs,
        "measured": measured,
        "measured_pars": measured_pars,
        "resource": {
            "feature_vector_n_cz": int(rdf["n_cz"].sum()),
            "max_setting_depth": int(rdf["depth"].max()),
            "max_setting_duration_us": float(
                rdf["duration_us"].max()
            ),
            "n_settings": int(len(rdf)),
        },
    }


def run_rwp_direct(
    noise_model,
    physical,
    ideal,
    candidate,
    batch_size,
):
    estimator = AerEstimatorV2(
        options={
            "default_precision": 0.0,
            "backend_options": {
                "method": "density_matrix",
                "noise_model": noise_model,
                "enable_truncation": True,
            },
        }
    )

    W = int(candidate["window"])

    circuits = [
        bind_rwp(
            physical["core"],
            physical["core_pars"],
            ideal["angles"],
            int(endpoint),
            W,
        )
        for endpoint in ideal["val_endpoints"]
    ]

    X = np.empty(
        (
            len(circuits),
            len(ideal["features"]),
        ),
        dtype=float,
    )

    for start in range(
        0,
        len(circuits),
        int(batch_size),
    ):
        stop = min(
            start + int(batch_size),
            len(circuits),
        )

        pubs = [
            (
                circuits[i],
                physical["mapped_obs"],
            )
            for i in range(start, stop)
        ]

        result = estimator.run(
            pubs,
            precision=0.0,
        ).result()

        for k, pub in enumerate(result):
            X[start + k, :] = np.asarray(
                pub.data.evs,
                dtype=float,
            ).reshape(-1)

    return X


def clean_bits(key):
    return str(key).replace(" ", "")


def single_exp(counts, q):
    total = float(sum(counts.values()))
    s = 0.0

    for key, n in counts.items():
        bits = clean_bits(key)
        bit = int(bits[-1 - int(q)])
        s += (1.0 if bit == 0 else -1.0) * float(n)

    return s / total


def parity_exp(counts, qs):
    total = float(sum(counts.values()))
    s = 0.0

    for key, n in counts.items():
        bits = clean_bits(key)
        p = 0

        for q in qs:
            p ^= int(bits[-1 - int(q)])

        s += (1.0 if p == 0 else -1.0) * float(n)

    return s / total


def feature_from_counts(feature, counts):
    if feature == "YX45":
        return parity_exp(
            counts,
            [4, 5],
        )

    return single_exp(
        counts,
        int(feature[1:]),
    )


def run_rwp_nested_sampled(
    noise_model,
    physical,
    ideal,
    candidate,
    readout,
    shot_levels,
    batch_size,
    seed,
):
    max_shots = int(max(shot_levels))

    sim = AerSimulator(
        noise_model=noise_model,
        enable_truncation=True,
    )

    W = int(candidate["window"])

    circuits = []
    meta = []

    for endpoint in ideal["val_endpoints"]:
        for setting, qc in physical["measured"].items():
            circuits.append(
                bind_rwp(
                    qc,
                    physical["measured_pars"][setting],
                    ideal["angles"],
                    int(endpoint),
                    W,
                )
            )
            meta.append(
                (
                    int(endpoint),
                    setting,
                )
            )

    mem = {
        int(N): {
            int(e): {}
            for e in ideal["val_endpoints"]
        }
        for N in shot_levels
    }

    for start in range(
        0,
        len(circuits),
        int(batch_size),
    ):
        stop = min(
            start + int(batch_size),
            len(circuits),
        )

        result = sim.run(
            circuits[start:stop],
            shots=max_shots,
            memory=True,
            seed_simulator=int(seed + start),
        ).result()

        for local_i, (
            endpoint,
            setting,
        ) in enumerate(
            meta[start:stop]
        ):
            shots_memory = list(
                result.get_memory(local_i)
            )

            for N in shot_levels:
                mem[int(N)][endpoint][setting] = Counter(
                    shots_memory[: int(N)]
                )

    X_by_N = {}

    for N in shot_levels:
        rows = []

        for endpoint in ideal["val_endpoints"]:
            row = []

            for feature in ideal["features"]:
                setting = feature_setting(
                    readout,
                    feature,
                )

                row.append(
                    feature_from_counts(
                        feature,
                        mem[int(N)][
                            int(endpoint)
                        ][setting],
                    )
                )

            rows.append(row)

        X_by_N[int(N)] = np.asarray(
            rows,
            dtype=float,
        )

    return X_by_N


# =============================================================================
# LOCAL SIX-QUBIT KINGSTON NOISE FOR CONT
# =============================================================================

def remap_noise_model_to_layout(
    backend,
    layout,
):
    """
    Build a 6-qubit noise model by taking the current backend noise entries
    acting wholly inside `layout` and remapping physical qubits -> local 0..5.

    This allows a true continuous 6-qubit density-matrix trajectory while
    preserving calibration-derived gate/thermal errors for the selected
    physical region.
    """
    full = NoiseModel.from_backend(
        backend,
        gate_error=True,
        thermal_relaxation=True,
        readout_error=False,
    )

    d = full.to_dict()

    p2l = {
        int(p): int(i)
        for i, p in enumerate(layout)
    }

    new_errors = []

    for err in d.get("errors", []):
        e = copy.deepcopy(err)

        gate_qubits = e.get(
            "gate_qubits",
            [],
        )

        if not gate_qubits:
            continue

        remapped_gate_qubits = []
        keep = True

        for qtuple in gate_qubits:
            qtuple = list(qtuple)

            if not all(
                int(q) in p2l
                for q in qtuple
            ):
                keep = False
                break

            remapped_gate_qubits.append(
                [
                    p2l[int(q)]
                    for q in qtuple
                ]
            )

        if not keep:
            continue

        e["gate_qubits"] = remapped_gate_qubits
        new_errors.append(e)

    local_dict = copy.deepcopy(d)
    local_dict["errors"] = new_errors

    try:
        return NoiseModel.from_dict(
            local_dict
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not remap current Kingston NoiseModel to the selected "
            "six-qubit region. Qiskit Aer NoiseModel.from_dict(...) failed."
        ) from exc


def local_coupling_for_layout(
    backend,
    layout,
):
    p2l = {
        int(p): int(i)
        for i, p in enumerate(layout)
    }

    edges = []

    coupling = backend.coupling_map

    for a, b in coupling:
        if int(a) in p2l and int(b) in p2l:
            edges.append(
                [
                    p2l[int(a)],
                    p2l[int(b)],
                ]
            )

    if not edges:
        raise RuntimeError(
            "No local coupling edges found for current layout."
        )

    return CouplingMap(edges)


def prepare_cont_local_step(
    backend,
    candidate,
    layout,
    local_noise,
):
    logical, pars = build_cont_step(
        candidate
    )

    cmap = local_coupling_for_layout(
        backend,
        layout,
    )

    step = transpile(
        logical,
        basis_gates=local_noise.basis_gates,
        coupling_map=cmap,
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
    )

    if int(step.count_ops().get("swap", 0)) != 0:
        raise RuntimeError(
            "CONT local step introduced SWAPs."
        )

    return step, pars


def initial_cont_density_matrix():
    """
    Week-9-compatible CONT initial state.

    q0..q3: |0000><0000|
    q4,q5:  |00><00|

    The injection qubits are reset and re-encoded at every chronological
    step.  The memory subsystem must start from |00><00|, exactly as in the
    original reduced-state CONT pipeline.

    Previous 10.05 draft used I/4 on q4,q5; that was the source of the small
    original-r CV mismatch (~7.2e-4 for CONT_H0_R1R3).
    """
    rho = np.zeros(
        (64, 64),
        dtype=complex,
    )
    rho[0, 0] = 1.0
    return rho

def run_cont_direct_and_states(
    backend,
    local_noise,
    step,
    pars,
    ideal,
):
    """
    Evolve one noisy CONT trajectory across all train+validation chronology.
    Save direct expectations + full pre-measurement density matrix at 2025
    validation endpoints.
    """
    sim = AerSimulator(
        method="density_matrix",
        noise_model=local_noise,
    )

    master = QuantumCircuit(6)

    master.set_density_matrix(
        initial_cont_density_matrix()
    )

    validation_set = {
        int(e): idx
        for idx, e in enumerate(
            ideal["val_endpoints"]
        )
    }

    save_labels = []

    for t in range(
        len(
            ideal["angles"]
        )
    ):
        values = {
            pars[q]: float(
                ideal["angles"][t, q]
            )
            for q in range(4)
        }

        bound = step.assign_parameters(
            values,
            inplace=False,
        )

        master.compose(
            bound,
            inplace=True,
        )

        if t in validation_set:
            row_index = validation_set[t]

            for feature in ideal["features"]:
                label = (
                    f"ev_{row_index}_{feature}"
                )

                if feature == "YX45":
                    pauli = Pauli("XY")
                    qargs = [4, 5]
                else:
                    pauli = Pauli(
                        feature[0]
                    )
                    qargs = [
                        int(
                            feature[1:]
                        )
                    ]

                master.save_expectation_value(
                    pauli,
                    qargs,
                    label=label,
                )

            rho_label = f"rho_{row_index}"
            master.save_density_matrix(
                list(range(6)),
                label=rho_label,
            )
            save_labels.append(rho_label)

    result = sim.run(
        master,
        shots=None,
    ).result()

    data = result.data(0)

    X = np.empty(
        (
            len(ideal["val_endpoints"]),
            len(ideal["features"]),
        ),
        dtype=float,
    )

    states = []

    for row_index in range(
        len(
            ideal["val_endpoints"]
        )
    ):
        for j, feature in enumerate(
            ideal["features"]
        ):
            X[row_index, j] = float(
                np.real(
                    data[
                        f"ev_{row_index}_{feature}"
                    ]
                )
            )

        states.append(
            DensityMatrix(
                np.asarray(
                    data[
                        f"rho_{row_index}"
                    ],
                    dtype=complex,
                )
            )
        )

    return X, states


def build_state_measurement_circuit(
    rho,
    setting,
):
    q = QuantumRegister(6, "q")
    c = ClassicalRegister(6, "m")
    qc = QuantumCircuit(q, c)

    qc.set_density_matrix(
        np.asarray(
            rho.data,
            dtype=complex,
        )
    )

    append_basis_rotation(
        qc,
        setting,
    )

    for k in range(6):
        qc.measure(k, k)

    return qc


def run_cont_nested_sampled(
    local_noise,
    states,
    ideal,
    readout,
    shot_levels,
    batch_size,
    seed,
):
    """
    Measurements are made on independent copies of the saved noisy CONT state,
    so they do not alter subsequent reservoir memory.
    """
    max_shots = int(
        max(
            shot_levels
        )
    )

    sim = AerSimulator(
        noise_model=local_noise,
    )

    circuits = []
    meta = []

    for idx, rho in enumerate(states):
        for setting in settings_for_readout(
            readout
        ):
            circuits.append(
                build_state_measurement_circuit(
                    rho,
                    setting,
                )
            )
            meta.append(
                (
                    idx,
                    setting,
                )
            )

    mem = {
        int(N): {
            idx: {}
            for idx in range(
                len(states)
            )
        }
        for N in shot_levels
    }

    for start in range(
        0,
        len(circuits),
        int(batch_size),
    ):
        stop = min(
            start + int(batch_size),
            len(circuits),
        )

        result = sim.run(
            circuits[start:stop],
            shots=max_shots,
            memory=True,
            seed_simulator=int(seed + start),
        ).result()

        for local_i, (
            idx,
            setting,
        ) in enumerate(
            meta[start:stop]
        ):
            shot_memory = list(
                result.get_memory(local_i)
            )

            for N in shot_levels:
                mem[int(N)][idx][setting] = Counter(
                    shot_memory[: int(N)]
                )

    X_by_N = {}

    for N in shot_levels:
        rows = []

        for idx in range(
            len(states)
        ):
            row = []

            for feature in ideal["features"]:
                setting = feature_setting(
                    readout,
                    feature,
                )

                row.append(
                    feature_from_counts(
                        feature,
                        mem[int(N)][idx][setting],
                    )
                )

            rows.append(row)

        X_by_N[int(N)] = np.asarray(
            rows,
            dtype=float,
        )

    return X_by_N


# =============================================================================
# PHYSICAL CONT RESOURCE AUDIT
# =============================================================================

def cont_resource_metrics(
    backend,
    candidate,
    readout,
    layout,
):
    rows = []

    for setting in settings_for_readout(
        readout
    ):
        q = QuantumRegister(6, "q")
        c = ClassicalRegister(6, "m")
        qc = QuantumCircuit(q, c)

        pars = ParameterVector(
            "theta",
            4,
        )

        append_step(
            qc,
            candidate,
            pars,
            reset_injection=True,
        )

        append_basis_rotation(
            qc,
            setting,
        )

        for k in range(6):
            qc.measure(k, k)

        tqc = transpile(
            qc,
            backend=backend,
            initial_layout=layout,
            routing_method="none",
            optimization_level=OPT_LEVEL,
            seed_transpiler=SEED_TRANSPILE,
        )

        backend.check_faulty(tqc)

        if int(tqc.count_ops().get("swap", 0)) != 0:
            raise RuntimeError(
                "CONT physical resource circuit introduced SWAPs."
            )

        try:
            duration_us = (
                float(
                    tqc.estimate_duration(
                        backend.target,
                        unit="s",
                    )
                )
                * 1e6
            )
        except Exception:
            duration_us = np.nan

        rows.append({
            "n_cz": int(
                tqc.count_ops().get(
                    "cz",
                    0,
                )
            ),
            "depth": int(
                tqc.depth()
            ),
            "duration_us": duration_us,
        })

    d = pd.DataFrame(rows)

    return {
        "feature_vector_n_cz": int(
            d["n_cz"].sum()
        ),
        "max_setting_depth": int(
            d["depth"].max()
        ),
        "max_setting_duration_us": float(
            d["duration_us"].max()
        ),
        "n_settings": int(
            len(d)
        ),
    }


# =============================================================================
# METRICS
# =============================================================================

def predict_metrics(
    X,
    ideal,
):
    pred = common.predict_scaled_ridge(
        ideal["model"],
        ideal["scaler"],
        ideal["keep"],
        X,
    )

    return {
        "pred": pred,
        "rmse": rmse(
            ideal["y_val"],
            pred,
        ),
        "mae": mae(
            ideal["y_val"],
            pred,
        ),
        "bias": bias(
            ideal["y_val"],
            pred,
        ),
        "prediction_corr_vs_ideal": safe_corr(
            ideal["pred_ideal"],
            pred,
        ),
        "feature_rmse_vs_ideal": float(
            np.sqrt(
                np.mean(
                    (
                        X
                        - ideal[
                            "X_val_ideal"
                        ]
                    )
                    ** 2
                )
            )
        ),
    }


def base_ids(
    mrow,
    r,
):
    return {
        "candidate_key": str(
            mrow["candidate_key"]
        ),
        "selected_by_rules": str(
            mrow[
                "selected_by_rules"
            ]
        ),
        "protocol": str(
            mrow["protocol"]
        ),
        "topology": str(
            mrow["topology"]
        ),
        "window": (
            np.nan
            if pd.isna(
                mrow.get(
                    "window",
                    np.nan,
                )
            )
            else int(
                float(
                    mrow["window"]
                )
            )
        ),
        "readout": str(
            mrow["readout"]
        ),
        "r": int(r),
    }


# =============================================================================
# PARETO
# =============================================================================

def pareto_mask(
    df,
    cols,
):
    x = df[cols].to_numpy(
        dtype=float
    )

    keep = np.ones(
        len(df),
        dtype=bool,
    )

    for i in range(
        len(df)
    ):
        for j in range(
            len(df)
        ):
            if i == j:
                continue

            if (
                np.all(
                    x[j] <= x[i]
                )
                and np.any(
                    x[j] < x[i]
                )
            ):
                keep[i] = False
                break

    return keep


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sampling-reps",
        type=int,
        default=DEFAULT_REPS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
    )

    parser.add_argument(
        "--sim-seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.overwrite:
        for p in [
            DIRECT_OUT,
            SAMPLED_OUT,
            RESULTS
            / "10_05_sampled_summary.csv",
            RESULTS
            / "10_05_resource_grid.csv",
            RESULTS
            / "10_05_pareto_front.csv",
        ]:
            if p.exists():
                p.unlink()

    manifest = load_manifest()

    if args.max_candidates is not None:
        manifest = manifest.head(
            int(
                args.max_candidates
            )
        ).copy()

    layouts = load_current_layouts()

    print("=" * 126)
    print("10.05 — RULE-SELECTED CANDIDATES / CURRENT KINGSTON")
    print("=" * 126)
    print(f"Candidates: {len(manifest)}")
    print(f"Operating points: {OPERATING_POINTS}")
    print(
        "Current physical layouts loaded from "
        "results/10_05_current_kingston_layouts.csv"
    )
    print("Historical Rule-5 layouts: IGNORED")
    print("NO QPU JOBS. 2026 frozen.")
    print()

    print("Loading exact J/hx/hy/alpha/dt dynamics from ENRICHED candidate CSV...")
    resolved = resolve_all_candidates(
        manifest
    )

    print(
        "PASS: enriched dynamics loaded and audit copy written to "
        "results/10_05_resolved_dynamics.csv"
    )
    print()

    # Current Kingston calibration snapshot, one refresh.
    service = get_service()

    backend = service.backend(
        BACKEND_NAME,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    try:
        props = backend.properties(
            refresh=True
        )
    except TypeError:
        props = backend.properties()

    calibration_time = getattr(
        props,
        "last_update_date",
        None,
    )

    physical_noise = NoiseModel.from_backend(
        backend,
        gate_error=True,
        thermal_relaxation=True,
        readout_error=False,
    )

    reference = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "backend": BACKEND_NAME,
        "backend_calibration": json_safe(
            calibration_time
        ),
        "candidate_count": int(
            len(manifest)
        ),
        "candidate_manifest": MANIFEST_FILE.name,
        "operating_points": [
            {
                "r": r,
                "shots_per_setting": N,
            }
            for r, N in OPERATING_POINTS
        ],
        "fresh_layouts": layouts,
        "historical_rule5_layouts_used": False,
        "cont_initial_memory_state": "|00><00| (Week-9-compatible)",
        "noise": {
            "gate_error": True,
            "thermal_relaxation": True,
            "readout_error": False,
        },
        "rwp_parameter_provenance": "10_05 enriched candidate manifest",
        "cont_parameter_provenance": "10_05 enriched candidate manifest",
    }

    (
        RESULTS
        / "10_05_reference.json"
    ).write_text(
        json.dumps(
            json_safe(reference),
            indent=2,
        ),
        encoding="utf-8",
    )

    # Resume keys.
    completed_direct = set()

    if DIRECT_OUT.exists():
        d = pd.read_csv(
            DIRECT_OUT
        )
        completed_direct = {
            (
                str(row["candidate_key"]),
                int(row["r"]),
            )
            for _, row in d.iterrows()
        }

    completed_sampled = set()

    if SAMPLED_OUT.exists():
        s = pd.read_csv(
            SAMPLED_OUT
        )
        completed_sampled = {
            (
                str(row["candidate_key"]),
                int(row["r"]),
                int(
                    row[
                        "shots_per_setting"
                    ]
                ),
                int(
                    row["replicate"]
                ),
            )
            for _, row in s.iterrows()
        }

    work_tv, cols, y_all = (
        common.load_train_validation()
    )

    # Local-noise models cached per topology/layout.
    local_noise_cache = {}

    for ci, mrow in manifest.iterrows():
        key = str(
            mrow["candidate_key"]
        )

        info = resolved[key]
        original = info["candidate"]

        protocol = str(
            mrow["protocol"]
        ).upper()

        topology = str(
            mrow["topology"]
        )

        readout = str(
            mrow["readout"]
        )

        layout = layouts[topology]

        print()
        print("-" * 126)
        print(
            f"[{ci+1}/{len(manifest)}] "
            f"{key} | {protocol} | {topology} | {readout}"
        )
        print(
            f"  alpha={original['alpha']}, dt={original['dt']}, "
            f"hx={original['hx']}, hy={original['hy']}"
        )
        print(
            "  J="
            + json.dumps(
                {
                    f"J{i}{j}": float(v)
                    for (i, j), v
                    in original["J"].items()
                }
            )
        )
        print(
            f"  current Kingston layout={layout}"
        )
        print("-" * 126)

        for r in R_VALUES:
            direct_key = (
                key,
                int(r),
            )

            shot_levels = (
                SHOT_LEVELS_BY_R[
                    int(r)
                ]
            )

            need_direct = (
                direct_key
                not in completed_direct
            )

            needed_sample = [
                (
                    int(N),
                    int(rep),
                )
                for rep in range(
                    1,
                    int(
                        args.sampling_reps
                    )
                    + 1
                )
                for N in shot_levels
                if (
                    key,
                    int(r),
                    int(N),
                    int(rep),
                )
                not in completed_sampled
            ]

            if (
                not need_direct
                and not needed_sample
            ):
                print(
                    f"  r={r}: complete, skip"
                )
                continue

            candidate = clone_candidate(
                original
            )
            candidate["r"] = int(r)

            # Recompute r-specific ideal representation/readout.
            if protocol == "RWP":
                candidate["window"] = int(
                    float(
                        mrow["window"]
                    )
                )

                ideal = ideal_rwp_reference(
                    candidate,
                    readout,
                    work_tv,
                    cols,
                    y_all,
                )

                if int(r) == int(mrow["original_r"]):
                    expected_cv = float(
                        mrow["source_readout_cv_rmse"]
                    )
                    expected_lambda = float(
                        mrow["source_selected_lambda"]
                    )

                    cv_delta = abs(
                        float(ideal["cv_rmse"])
                        - expected_cv
                    )

                    lambda_ok = np.isclose(
                        float(ideal["selected_lambda"]),
                        expected_lambda,
                        rtol=0.0,
                        atol=1e-12,
                    )

                    print(
                        f"       ORIGINAL-r AUDIT: "
                        f"CV recomputed={ideal['cv_rmse']:.6f}, "
                        f"source-readout CV={expected_cv:.6f}, "
                        f"|Δ|={cv_delta:.3g}; "
                        f"lambda recomputed={ideal['selected_lambda']}, "
                        f"source={expected_lambda}"
                    )

                    if cv_delta > 1e-5 or not lambda_ok:
                        raise RuntimeError(
                            f"Original-r enriched-manifest audit failed for "
                            f"{key}: CV recomputed={ideal['cv_rmse']:.9f}, "
                            f"source-readout CV={expected_cv:.9f}, "
                            f"lambda recomputed={ideal['selected_lambda']}, "
                            f"source lambda={expected_lambda}."
                        )

                physical = prepare_rwp_physical(
                    backend,
                    candidate,
                    readout,
                    layout,
                )

                resource = physical[
                    "resource"
                ]

                t0 = time.time()

                X_direct = run_rwp_direct(
                    physical_noise,
                    physical,
                    ideal,
                    candidate,
                    args.batch_size,
                )

                states = None

            else:
                ideal = ideal_cont_reference(
                    candidate,
                    readout,
                    work_tv,
                    cols,
                    y_all,
                )

                if int(r) == int(mrow["original_r"]):
                    expected_cv = float(
                        mrow["source_readout_cv_rmse"]
                    )
                    expected_lambda = float(
                        mrow["source_selected_lambda"]
                    )

                    cv_delta = abs(
                        float(ideal["cv_rmse"])
                        - expected_cv
                    )

                    lambda_ok = np.isclose(
                        float(ideal["selected_lambda"]),
                        expected_lambda,
                        rtol=0.0,
                        atol=1e-12,
                    )

                    print(
                        f"       ORIGINAL-r AUDIT: "
                        f"CV recomputed={ideal['cv_rmse']:.6f}, "
                        f"source-readout CV={expected_cv:.6f}, "
                        f"|Δ|={cv_delta:.3g}; "
                        f"lambda recomputed={ideal['selected_lambda']}, "
                        f"source={expected_lambda}"
                    )

                    if cv_delta > 1e-5 or not lambda_ok:
                        raise RuntimeError(
                            f"Original-r enriched-manifest audit failed for "
                            f"{key}: CV recomputed={ideal['cv_rmse']:.9f}, "
                            f"source-readout CV={expected_cv:.9f}, "
                            f"lambda recomputed={ideal['selected_lambda']}, "
                            f"source lambda={expected_lambda}. "
                            "Do not continue until ideal reconstruction "
                            "matches the enriched candidate manifest."
                        )

                cache_key = (
                    topology,
                    tuple(layout),
                )

                if cache_key not in local_noise_cache:
                    local_noise_cache[
                        cache_key
                    ] = remap_noise_model_to_layout(
                        backend,
                        layout,
                    )

                local_noise = (
                    local_noise_cache[
                        cache_key
                    ]
                )

                step, step_pars = (
                    prepare_cont_local_step(
                        backend,
                        candidate,
                        layout,
                        local_noise,
                    )
                )

                resource = cont_resource_metrics(
                    backend,
                    candidate,
                    readout,
                    layout,
                )

                t0 = time.time()

                X_direct, states = (
                    run_cont_direct_and_states(
                        backend,
                        local_noise,
                        step,
                        step_pars,
                        ideal,
                    )
                )

            direct_metrics = predict_metrics(
                X_direct,
                ideal,
            )

            print(
                f"  r={r}: lambda={ideal['selected_lambda']}, "
                f"CV={ideal['cv_rmse']:.6f}, "
                f"ideal={ideal['ideal_rmse']:.6f}, "
                f"direct-noisy={direct_metrics['rmse']:.6f}, "
                f"CZ/feature={resource['feature_vector_n_cz']}, "
                f"time={(time.time()-t0)/60:.2f} min"
            )

            if need_direct:
                row = {
                    **base_ids(
                        mrow,
                        r,
                    ),
                    "selected_lambda": ideal[
                        "selected_lambda"
                    ],
                    "training_cv_rmse": ideal[
                        "cv_rmse"
                    ],
                    "training_cv_sd": ideal[
                        "cv_sd"
                    ],
                    "ideal_validation_rmse": ideal[
                        "ideal_rmse"
                    ],
                    "rmse": direct_metrics[
                        "rmse"
                    ],
                    "mae": direct_metrics[
                        "mae"
                    ],
                    "bias": direct_metrics[
                        "bias"
                    ],
                    "prediction_corr_vs_ideal": direct_metrics[
                        "prediction_corr_vs_ideal"
                    ],
                    "feature_rmse_vs_ideal": direct_metrics[
                        "feature_rmse_vs_ideal"
                    ],
                    **resource,
                    "fresh_layout": json.dumps(
                        layout
                    ),
                }

                append_csv(
                    DIRECT_OUT,
                    [row],
                )

                completed_direct.add(
                    direct_key
                )

            # Sampling repetitions.
            for rep in range(
                1,
                int(
                    args.sampling_reps
                )
                + 1
            ):
                missing_N = [
                    N
                    for N in shot_levels
                    if (
                        key,
                        int(r),
                        int(N),
                        int(rep),
                    )
                    not in completed_sampled
                ]

                if not missing_N:
                    continue

                seed = (
                    int(args.sim_seed)
                    + 1000000 * (ci + 1)
                    + 10000 * int(r)
                    + 100 * int(rep)
                )

                t1 = time.time()

                if protocol == "RWP":
                    X_by_N = (
                        run_rwp_nested_sampled(
                            physical_noise,
                            physical,
                            ideal,
                            candidate,
                            readout,
                            shot_levels,
                            args.batch_size,
                            seed,
                        )
                    )
                else:
                    X_by_N = (
                        run_cont_nested_sampled(
                            local_noise,
                            states,
                            ideal,
                            readout,
                            shot_levels,
                            args.batch_size,
                            seed,
                        )
                    )

                rows = []

                for N in missing_N:
                    metrics = predict_metrics(
                        X_by_N[int(N)],
                        ideal,
                    )

                    rows.append({
                        **base_ids(
                            mrow,
                            r,
                        ),
                        "shots_per_setting": int(
                            N
                        ),
                        "replicate": int(
                            rep
                        ),
                        "selected_lambda": ideal[
                            "selected_lambda"
                        ],
                        "training_cv_rmse": ideal[
                            "cv_rmse"
                        ],
                        "ideal_validation_rmse": ideal[
                            "ideal_rmse"
                        ],
                        "direct_full_quantum_rmse": direct_metrics[
                            "rmse"
                        ],
                        "rmse": metrics[
                            "rmse"
                        ],
                        "mae": metrics[
                            "mae"
                        ],
                        "bias": metrics[
                            "bias"
                        ],
                        "delta_rmse_vs_direct": (
                            metrics[
                                "rmse"
                            ]
                            - direct_metrics[
                                "rmse"
                            ]
                        ),
                        **resource,
                        "cz_applications_per_feature_vector": int(
                            resource[
                                "feature_vector_n_cz"
                            ]
                            * int(N)
                        ),
                        "fresh_layout": json.dumps(
                            layout
                        ),
                    })

                    print(
                        f"       rep={rep}, N={N}: "
                        f"RMSE={metrics['rmse']:.6f}, "
                        f"Δdirect="
                        f"{metrics['rmse']-direct_metrics['rmse']:+.6f}"
                    )

                append_csv(
                    SAMPLED_OUT,
                    rows,
                )

                for row in rows:
                    completed_sampled.add(
                        (
                            row[
                                "candidate_key"
                            ],
                            int(
                                row["r"]
                            ),
                            int(
                                row[
                                    "shots_per_setting"
                                ]
                            ),
                            int(
                                row[
                                    "replicate"
                                ]
                            ),
                        )
                    )

                print(
                    f"       sampled runtime="
                    f"{(time.time()-t1)/60:.2f} min"
                )

    # =========================================================================
    # FINAL SUMMARIES
    # =========================================================================
    direct = pd.read_csv(
        DIRECT_OUT
    )

    sampled = pd.read_csv(
        SAMPLED_OUT
    )

    rows = []

    for (
        candidate_key,
        r,
        N,
    ), g in sampled.groupby(
        [
            "candidate_key",
            "r",
            "shots_per_setting",
        ],
        sort=True,
    ):
        n = len(g)

        rows.append({
            "candidate_key": candidate_key,
            "selected_by_rules": str(
                g.iloc[0][
                    "selected_by_rules"
                ]
            ),
            "protocol": str(
                g.iloc[0][
                    "protocol"
                ]
            ),
            "topology": str(
                g.iloc[0][
                    "topology"
                ]
            ),
            "window": g.iloc[0][
                "window"
            ],
            "readout": str(
                g.iloc[0][
                    "readout"
                ]
            ),
            "r": int(r),
            "shots_per_setting": int(
                N
            ),
            "n_runs": int(n),
            "rmse_mean": float(
                g["rmse"].mean()
            ),
            "rmse_sd": (
                float(
                    g["rmse"].std(
                        ddof=1
                    )
                )
                if n > 1
                else 0.0
            ),
            "rmse_se": (
                float(
                    g["rmse"].std(
                        ddof=1
                    )
                    / math.sqrt(n)
                )
                if n > 1
                else 0.0
            ),
            "mae_mean": float(
                g["mae"].mean()
            ),
            "bias_mean": float(
                g["bias"].mean()
            ),
            "direct_full_quantum_rmse": float(
                g[
                    "direct_full_quantum_rmse"
                ].iloc[0]
            ),
            "delta_rmse_vs_direct_mean": float(
                g[
                    "delta_rmse_vs_direct"
                ].mean()
            ),
            "feature_vector_n_cz": int(
                g[
                    "feature_vector_n_cz"
                ].iloc[0]
            ),
            "max_setting_depth": int(
                g[
                    "max_setting_depth"
                ].iloc[0]
            ),
            "max_setting_duration_us": float(
                g[
                    "max_setting_duration_us"
                ].iloc[0]
            ),
            "n_settings": int(
                g[
                    "n_settings"
                ].iloc[0]
            ),
            "cz_applications_per_feature_vector": int(
                g[
                    "cz_applications_per_feature_vector"
                ].iloc[0]
            ),
        })

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        RESULTS
        / "10_05_sampled_summary.csv",
        index=False,
    )

    grid = summary.copy()

    grid[
        "pareto_nondominated"
    ] = pareto_mask(
        grid,
        [
            "rmse_mean",
            "feature_vector_n_cz",
            "shots_per_setting",
        ],
    )

    grid.to_csv(
        RESULTS
        / "10_05_resource_grid.csv",
        index=False,
    )

    pareto = (
        grid[
            grid[
                "pareto_nondominated"
            ]
        ]
        .sort_values(
            [
                "rmse_mean",
                "feature_vector_n_cz",
                "shots_per_setting",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pareto.to_csv(
        RESULTS
        / "10_05_pareto_front.csv",
        index=False,
    )

    print()
    print("=" * 126)
    print("10.05 COMPLETE")
    print("=" * 126)
    print(
        f"Direct rows: {len(direct)}"
    )
    print(
        f"Sampled rows: {len(sampled)}"
    )
    print(
        f"Summary cells: {len(summary)}"
    )
    print(
        f"Pareto cells: {len(pareto)}"
    )
    print()
    print(
        summary.sort_values(
            "rmse_mean"
        )
        .head(20)
        [
            [
                "candidate_key",
                "selected_by_rules",
                "protocol",
                "topology",
                "window",
                "readout",
                "r",
                "shots_per_setting",
                "rmse_mean",
                "rmse_sd",
                "feature_vector_n_cz",
            ]
        ]
        .to_string(
            index=False
        )
    )
    print()
    print("Saved:")
    for name in [
        "10_05_reference.json",
        "10_05_resolved_dynamics.csv",
        "10_05_direct_runs.csv",
        "10_05_sampled_runs.csv",
        "10_05_sampled_summary.csv",
        "10_05_resource_grid.csv",
        "10_05_pareto_front.csv",
    ]:
        print(
            f"  results/{name}"
        )


if __name__ == "__main__":
    main()
