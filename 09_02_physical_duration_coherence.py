
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile

from ibm_account import get_service


# =============================================================================
# WEEK 11.0A — LIVE PHYSICAL-QUBIT / EMBEDDING RESELECTION
#
# PURPOSE
# -------
# Before spending Open-plan QPU time, refresh the physical mapping for the
# best logical QRC candidate available so far.
#
# This script:
#   1. selects the best available logical candidate from completed Week-9 files,
#   2. refreshes IBM backend properties,
#   3. rebuilds the CURRENT CZ connectivity graph,
#   4. enumerates every native 6-qubit embedding of the candidate topology,
#   5. rejects faulty/unavailable regions,
#   6. records live T1/T2/readout/CZ calibration metrics,
#   7. Pareto-screens embeddings WITHOUT a weighted score,
#   8. compiles the exact candidate circuit with routing_method="none",
#   9. verifies zero SWAP,
#  10. records circuit-aware 1Q/CZ errors, depth and scheduled duration,
#  11. saves the current top-3 embeddings per backend and one overall choice.
#
# NO REAL QPU JOB IS SUBMITTED.
# This script should not consume the 10-minute QPU allocation.
# =============================================================================


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DATA_FILE = RESULTS / "03_01_preprocessed_samples.csv"
J_WINNER_FILE = RESULTS / "08_03b_J_search_topology_winners.csv"

FAIR_FILE = RESULTS / "09_03c_forecast_best_135.csv"
R_ABLATION_FILE = RESULTS / "09_03_trotter_forecast_accuracy.csv"
FINAL10_FILE = RESULTS / "09_01y_final_10_candidates.csv"

PREFIX = RESULTS / "11_0A"

ALPHA = 0.75
DT = 1.6
REFERENCE_HX = 0.5
REFERENCE_R = 2

BACKENDS = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh",
]

OPTIMIZATION_LEVEL = 1
SEED_TRANSPILER = 42
SHORTLIST_PER_BACKEND = 60
TOP_K_PER_BACKEND = 3


TOPOLOGY_EDGES = {
    "H0": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    "H1": [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)],
    "H2": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)],
    "H3": [(2, 3), (1, 2), (0, 1), (0, 4), (4, 5)],
    "H4": [(1, 2), (0, 1), (0, 4), (3, 4), (4, 5)],
}

EDGE_TO_COLUMN = {
    "H0": {(0, 1): "J01", (1, 2): "J12", (2, 3): "J23",
           (3, 4): "J34", (4, 5): "J45"},
    "H1": {(0, 1): "J01", (1, 2): "J12", (2, 3): "J23",
           (3, 4): "J34", (3, 5): "J35"},
    "H2": {(0, 1): "J01", (1, 2): "J12", (2, 3): "J23",
           (2, 4): "J24", (4, 5): "J45"},
    "H3": {(2, 3): "J23", (1, 2): "J12", (0, 1): "J01",
           (0, 4): "J04", (4, 5): "J45"},
    "H4": {(1, 2): "J12", (0, 1): "J01", (0, 4): "J04",
           (3, 4): "J34", (4, 5): "J45"},
}

READOUT_SETTINGS = {
    "XZ_injection": ["XXXXZZ", "ZZZZZZ"],
    "XZinj_plus_YX45": ["XXXXYX", "ZZZZZZ"],
    "XYZ_all": ["XXXXXX", "YYYYYY", "ZZZZZZ"],
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    return str(obj)


def pareto_mask(frame, minimize_cols, maximize_cols=None):
    if maximize_cols is None:
        maximize_cols = []

    arrays = []
    for c in minimize_cols:
        arrays.append(
            pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        )
    for c in maximize_cols:
        arrays.append(
            -pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        )

    M = np.column_stack(arrays)
    finite = np.all(np.isfinite(M), axis=1)
    keep = np.zeros(len(frame), dtype=bool)

    valid_idx = np.where(finite)[0]
    V = M[finite]

    for local_i, global_i in enumerate(valid_idx):
        v = V[local_i]
        no_worse = np.all(V <= v, axis=1)
        strictly_better = np.any(V < v, axis=1)
        dominates = no_worse & strictly_better
        dominates[local_i] = False

        if not np.any(dominates):
            keep[global_i] = True

    return keep


# =============================================================================
# INPUT ENCODING
# =============================================================================

def claim_mapping(z):
    return float(np.clip(float(z) / 3.0, -1.0, 1.0))


def policy_mapping(z):
    z = float(z)
    return z / (3.0 + abs(z))


def weekday_mapping(d_sin, d_cos):
    return math.atan2(float(d_sin), float(d_cos)) % (2.0 * math.pi)


def holiday_mapping(h):
    h = int(h)
    if h not in (0, 1):
        raise ValueError(f"Holiday must be 0/1, got {h}")
    return math.pi * h


def first_input_angles():
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)
    df["input_date"] = pd.to_datetime(df["input_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.sort_values("target_date").reset_index(drop=True)
    row = df.iloc[0]

    angles = np.array([
        ALPHA * claim_mapping(row["C_t_z"]),
        weekday_mapping(row["D_sin"], row["D_cos"]),
        ALPHA * policy_mapping(row["P_t_z"]),
        holiday_mapping(row["is_public_holiday_t_plus_1"]),
    ], dtype=float)

    audit = {
        "input_date": str(row["input_date"].date()),
        "target_date": str(row["target_date"].date()),
        "theta_C": float(angles[0]),
        "theta_D": float(angles[1]),
        "theta_P": float(angles[2]),
        "theta_H": float(angles[3]),
    }

    return angles, audit


# =============================================================================
# CANDIDATE SELECTION
# =============================================================================

def week8_J(topology):
    if not J_WINNER_FILE.exists():
        raise FileNotFoundError(J_WINNER_FILE)

    jdf = pd.read_csv(J_WINNER_FILE)
    m = jdf[jdf["topology"] == topology]

    if len(m) != 1:
        raise RuntimeError(
            f"{topology}: expected exactly one Week-8 J winner, got {len(m)}"
        )

    row = m.iloc[0]
    return {
        edge: float(row[col])
        for edge, col in EDGE_TO_COLUMN[topology].items()
    }


def J_from_row_or_week8(row, topology):
    J = {}
    for edge, col in EDGE_TO_COLUMN[topology].items():
        if col not in row.index or pd.isna(row[col]):
            return week8_J(topology)
        J[edge] = float(row[col])
    return J


def select_best_candidate():
    # Preferred: completed fair r=1,2,3 re-optimization.
    if FAIR_FILE.exists():
        df = pd.read_csv(FAIR_FILE)

        required = {
            "seed_candidate_id", "topology", "readout",
            "trotter_r", "hx", "hy", "cv_rmse",
        }

        if required.issubset(df.columns) and len(df) > 0:
            row = (
                df.sort_values(
                    ["cv_rmse", "validation_rmse"],
                    ascending=[True, True],
                )
                .iloc[0]
            )

            topology = str(row["topology"])

            return {
                "selection_source": str(FAIR_FILE),
                "selection_complete_135": bool(len(df) >= 135),
                "candidate_id": str(row["seed_candidate_id"]),
                "topology": topology,
                "readout": str(row["readout"]),
                "r": int(row["trotter_r"]),
                "hx": float(row["hx"]),
                "hy": float(row["hy"]),
                "cv_rmse": float(row["cv_rmse"]),
                "validation_rmse": (
                    float(row["validation_rmse"])
                    if "validation_rmse" in row.index
                    and pd.notna(row["validation_rmse"])
                    else np.nan
                ),
                "J": J_from_row_or_week8(row, topology),
            }

    # Fallback: r ablation on final 10.
    if R_ABLATION_FILE.exists():
        df = pd.read_csv(R_ABLATION_FILE)
        required = {
            "candidate_id", "topology", "readout",
            "trotter_r", "hy", "cv_rmse",
        }

        if required.issubset(df.columns) and len(df) > 0:
            row = (
                df.sort_values(
                    ["cv_rmse", "validation_rmse"],
                    ascending=[True, True],
                )
                .iloc[0]
            )
            topology = str(row["topology"])

            return {
                "selection_source": str(R_ABLATION_FILE),
                "selection_complete_135": False,
                "candidate_id": str(row["candidate_id"]),
                "topology": topology,
                "readout": str(row["readout"]),
                "r": int(row["trotter_r"]),
                "hx": REFERENCE_HX,
                "hy": float(row["hy"]),
                "cv_rmse": float(row["cv_rmse"]),
                "validation_rmse": (
                    float(row["validation_rmse"])
                    if "validation_rmse" in row.index
                    and pd.notna(row["validation_rmse"])
                    else np.nan
                ),
                "J": week8_J(topology),
            }

    # Last fallback: best final-10 CV at r=2.
    if FINAL10_FILE.exists():
        df = pd.read_csv(FINAL10_FILE)
        row = df.sort_values(
            ["cv_rmse", "validation_rmse"],
            ascending=[True, True],
        ).iloc[0]

        topology = str(row["topology"])

        return {
            "selection_source": str(FINAL10_FILE),
            "selection_complete_135": False,
            "candidate_id": str(row["candidate_id"]),
            "topology": topology,
            "readout": str(row["readout"]),
            "r": REFERENCE_R,
            "hx": REFERENCE_HX,
            "hy": float(row["hy"]),
            "cv_rmse": float(row["cv_rmse"]),
            "validation_rmse": float(row["validation_rmse"]),
            "J": week8_J(topology),
        }

    raise FileNotFoundError(
        "No candidate-selection source found. Expected one of:\n"
        f"  {FAIR_FILE}\n"
        f"  {R_ABLATION_FILE}\n"
        f"  {FINAL10_FILE}"
    )


# =============================================================================
# CIRCUIT
# =============================================================================

def build_core(candidate, angles):
    qc = QuantumCircuit(6, name="qrc_core")

    for q in range(4):
        qc.ry(float(angles[q]), q)

    topology = candidate["topology"]
    J = candidate["J"]
    r = int(candidate["r"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])

    for _ in range(r):
        for i, j in TOPOLOGY_EDGES[topology]:
            qc.rzz(
                2.0 * float(J[(i, j)]) * DT / r,
                i,
                j,
            )

        theta_x = 2.0 * hx * DT / r
        for q in range(6):
            qc.rx(theta_x, q)

        if not np.isclose(hy, 0.0):
            theta_y = 2.0 * hy * DT / r
            qc.ry(theta_y, 4)
            qc.ry(theta_y, 5)

    return qc


# =============================================================================
# LIVE TARGET / CALIBRATION
# =============================================================================

def target_instruction(target, name, qargs):
    try:
        return target[name][tuple(qargs)]
    except Exception:
        return None


def target_error(target, name, qargs):
    props = target_instruction(target, name, qargs)
    if props is None:
        return np.nan

    value = getattr(props, "error", None)
    return np.nan if value is None else float(value)


def target_duration(target, name, qargs):
    props = target_instruction(target, name, qargs)
    if props is None:
        return np.nan

    value = getattr(props, "duration", None)
    return np.nan if value is None else float(value)


def cz_error_any_direction(target, i, j):
    values = []

    for qargs in [(i, j), (j, i)]:
        v = target_error(target, "cz", qargs)
        if np.isfinite(v):
            values.append(v)

    if not values:
        return np.nan

    return float(min(values))


def get_faulty_qubits(properties):
    try:
        return set(int(x) for x in properties.faulty_qubits())
    except Exception:
        return set()


def build_undirected_cz_graph(backend):
    cmap = backend.target.build_coupling_map(
        two_q_gate="cz",
        filter_idle_qubits=True,
    )

    if cmap is None:
        raise RuntimeError(
            f"{backend.name}: could not build CZ coupling map."
        )

    adjacency = {
        q: set()
        for q in range(backend.num_qubits)
    }

    for i, j in cmap.get_edges():
        i = int(i)
        j = int(j)
        adjacency[i].add(j)
        adjacency[j].add(i)

    return adjacency


# =============================================================================
# EMBEDDING ENUMERATION
# =============================================================================

def logical_adjacency(topology):
    adj = {q: set() for q in range(6)}

    for i, j in TOPOLOGY_EDGES[topology]:
        adj[i].add(j)
        adj[j].add(i)

    return adj


def enumerate_native_embeddings(
    topology,
    physical_adj,
    faulty_qubits,
):
    """
    Backtracking subgraph monomorphism.

    Logical labels are preserved: mapping[q_logical] = q_physical.
    Extra physical edges between mapped qubits are allowed.
    """
    logical_adj = logical_adjacency(topology)

    logical_order = sorted(
        range(6),
        key=lambda q: (
            -len(logical_adj[q]),
            q,
        ),
    )

    physical_nodes = [
        q
        for q, neigh in physical_adj.items()
        if q not in faulty_qubits
        and len(neigh) >= 1
    ]

    degree_candidates = {
        lq: [
            pq
            for pq in physical_nodes
            if len(physical_adj[pq]) >= len(logical_adj[lq])
        ]
        for lq in range(6)
    }

    mappings = []
    assigned = {}
    used = set()

    def rec(pos):
        if pos == len(logical_order):
            mappings.append(
                tuple(
                    int(assigned[q])
                    for q in range(6)
                )
            )
            return

        lq = logical_order[pos]

        # If logical neighbors are already assigned, candidate must be adjacent
        # to all of their physical images.
        assigned_neighbors = [
            n
            for n in logical_adj[lq]
            if n in assigned
        ]

        if assigned_neighbors:
            candidate_set = None

            for n in assigned_neighbors:
                neigh = physical_adj[
                    assigned[n]
                ]

                candidate_set = (
                    set(neigh)
                    if candidate_set is None
                    else candidate_set & set(neigh)
                )

            candidates = sorted(
                candidate_set
                &
                set(degree_candidates[lq])
            )
        else:
            candidates = degree_candidates[lq]

        for pq in candidates:
            if pq in used:
                continue

            # Verify every already assigned required logical edge.
            ok = True

            for n in assigned_neighbors:
                if assigned[n] not in physical_adj[pq]:
                    ok = False
                    break

            if not ok:
                continue

            assigned[lq] = pq
            used.add(pq)

            rec(pos + 1)

            used.remove(pq)
            del assigned[lq]

    rec(0)

    # Remove exact duplicates while preserving order.
    unique = list(dict.fromkeys(mappings))
    return unique


# =============================================================================
# REGION METRICS
# =============================================================================

def safe_qubit_coherence(backend, qubit):
    """Return (t1_us, t2_us, source) without crashing on missing calibration.

    IBM qubit calibration fields are optional.  If a current T1 or T2 value is
    absent, this function returns NaN for that value.  Embeddings containing
    such qubits are kept in the audit CSV but excluded from ranking.
    """
    try:
        qp = backend.qubit_properties(int(qubit))
        if qp is not None:
            t1 = getattr(qp, "t1", None)
            t2 = getattr(qp, "t2", None)

            t1_us = (
                float(t1) * 1e6
                if t1 is not None and np.isfinite(float(t1))
                else np.nan
            )
            t2_us = (
                float(t2) * 1e6
                if t2 is not None and np.isfinite(float(t2))
                else np.nan
            )
            return t1_us, t2_us, "backend.qubit_properties"
    except Exception:
        pass

    try:
        properties = backend.properties()
    except Exception:
        properties = None

    t1_us = np.nan
    t2_us = np.nan

    if properties is not None:
        try:
            value = float(properties.t1(int(qubit)))
            if np.isfinite(value):
                t1_us = value * 1e6
        except Exception:
            pass

        try:
            value = float(properties.t2(int(qubit)))
            if np.isfinite(value):
                t2_us = value * 1e6
        except Exception:
            pass

    return t1_us, t2_us, "backend.properties_fallback"


def safe_readout_error(backend, properties, qubit):
    try:
        value = target_error(
            backend.target,
            "measure",
            (int(qubit),),
        )
        if np.isfinite(value):
            return float(value)
    except Exception:
        pass

    try:
        value = float(properties.readout_error(int(qubit)))
        if np.isfinite(value):
            return value
    except Exception:
        pass

    return np.nan


def region_calibration_metrics(
    backend,
    properties,
    topology,
    layout,
):
    target = backend.target

    t1_values = []
    t2_values = []
    coherence_sources = []

    for q in layout:
        t1_us, t2_us, source = safe_qubit_coherence(backend, q)
        t1_values.append(t1_us)
        t2_values.append(t2_us)
        coherence_sources.append(source)

    t1_us = np.asarray(t1_values, dtype=float)
    t2_us = np.asarray(t2_values, dtype=float)

    if (
        not np.all(np.isfinite(t1_us))
        or not np.all(np.isfinite(t2_us))
    ):
        return {
            "calibration_complete": False,
            "missing_t1_qubits": json.dumps(
                [
                    int(layout[i])
                    for i, value in enumerate(t1_us)
                    if not np.isfinite(value)
                ]
            ),
            "missing_t2_qubits": json.dumps(
                [
                    int(layout[i])
                    for i, value in enumerate(t2_us)
                    if not np.isfinite(value)
                ]
            ),
            "missing_readout_qubits": "[]",
            "missing_cz_edge": "",
            "coherence_sources": json.dumps(coherence_sources),
        }

    readout = np.asarray(
        [
            safe_readout_error(backend, properties, q)
            for q in layout
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(readout)):
        return {
            "calibration_complete": False,
            "missing_t1_qubits": "[]",
            "missing_t2_qubits": "[]",
            "missing_readout_qubits": json.dumps(
                [
                    int(layout[i])
                    for i, value in enumerate(readout)
                    if not np.isfinite(value)
                ]
            ),
            "missing_cz_edge": "",
            "coherence_sources": json.dumps(coherence_sources),
        }

    measure_duration = np.asarray(
        [
            target_duration(target, "measure", (q,))
            for q in layout
        ],
        dtype=float,
    )

    cz_errors = []

    for li, lj in TOPOLOGY_EDGES[topology]:
        pi = layout[li]
        pj = layout[lj]
        err = cz_error_any_direction(target, pi, pj)

        if not np.isfinite(err):
            return {
                "calibration_complete": False,
                "missing_t1_qubits": "[]",
                "missing_t2_qubits": "[]",
                "missing_readout_qubits": "[]",
                "missing_cz_edge": json.dumps([int(pi), int(pj)]),
                "coherence_sources": json.dumps(coherence_sources),
            }

        cz_errors.append(err)

    cz_errors = np.asarray(cz_errors, dtype=float)

    return {
        "calibration_complete": True,
        "missing_t1_qubits": "[]",
        "missing_t2_qubits": "[]",
        "missing_readout_qubits": "[]",
        "missing_cz_edge": "",
        "coherence_sources": json.dumps(coherence_sources),
        "min_t1_us": float(np.min(t1_us)),
        "mean_t1_us": float(np.mean(t1_us)),
        "min_t2_us": float(np.min(t2_us)),
        "mean_t2_us": float(np.mean(t2_us)),
        "max_readout_error": float(np.max(readout)),
        "mean_readout_error": float(np.mean(readout)),
        "max_readout_error_percent": float(np.max(readout) * 100.0),
        "mean_readout_error_percent": float(np.mean(readout) * 100.0),
        "max_cz_error": float(np.max(cz_errors)),
        "mean_cz_error": float(np.mean(cz_errors)),
        "max_cz_error_percent": float(np.max(cz_errors) * 100.0),
        "mean_cz_error_percent": float(np.mean(cz_errors) * 100.0),
        "max_measure_duration_us": (
            float(np.nanmax(measure_duration) * 1e6)
            if np.any(np.isfinite(measure_duration))
            else np.nan
        ),
    }


# =============================================================================
# CIRCUIT-AWARE COMPILED METRICS
# =============================================================================

def compiled_resource_metrics(
    circuit,
    backend,
):
    counts = {
        str(k): int(v)
        for k, v in circuit.count_ops().items()
    }

    oneq_errors = []
    twoq_errors = []

    excluded = {
        "measure",
        "reset",
        "delay",
        "barrier",
    }

    n_1q = 0
    n_2q = 0

    for inst in circuit.data:
        op_name = inst.operation.name
        qargs = tuple(
            circuit.find_bit(q).index
            for q in inst.qubits
        )

        if len(qargs) == 1 and op_name not in excluded:
            n_1q += 1
            err = target_error(
                backend.target,
                op_name,
                qargs,
            )
            if np.isfinite(err):
                oneq_errors.append(err)

        elif len(qargs) == 2:
            n_2q += 1
            err = target_error(
                backend.target,
                op_name,
                qargs,
            )

            if (
                not np.isfinite(err)
                and op_name == "cz"
            ):
                err = target_error(
                    backend.target,
                    op_name,
                    qargs[::-1],
                )

            if np.isfinite(err):
                twoq_errors.append(err)

    try:
        depth_1q = int(
            circuit.depth(
                filter_function=lambda inst:
                    (
                        len(inst.qubits) == 1
                        and inst.operation.name not in excluded
                    )
            )
        )

        depth_2q = int(
            circuit.depth(
                filter_function=lambda inst:
                    len(inst.qubits) == 2
            )
        )
    except TypeError:
        depth_1q = np.nan
        depth_2q = np.nan

    duration_s = float(
        circuit.estimate_duration(
            backend.target,
            unit="s",
        )
    )

    return {
        "compiled_depth": int(circuit.depth()),
        "compiled_depth_1q": depth_1q,
        "compiled_depth_2q": depth_2q,
        "compiled_size": int(circuit.size()),
        "compiled_n_1q": int(n_1q),
        "compiled_n_2q": int(n_2q),
        "compiled_n_cz": int(counts.get("cz", 0)),
        "compiled_n_swap": int(counts.get("swap", 0)),
        "compiled_duration_us": duration_s * 1e6,
        "compiled_1q_error_max": (
            float(np.max(oneq_errors))
            if oneq_errors
            else np.nan
        ),
        "compiled_1q_error_mean_occurrence": (
            float(np.mean(oneq_errors))
            if oneq_errors
            else np.nan
        ),
        "compiled_2q_error_max": (
            float(np.max(twoq_errors))
            if twoq_errors
            else np.nan
        ),
        "compiled_2q_error_mean_occurrence": (
            float(np.mean(twoq_errors))
            if twoq_errors
            else np.nan
        ),
        "compiled_1q_error_max_percent": (
            float(np.max(oneq_errors) * 100.0)
            if oneq_errors
            else np.nan
        ),
        "compiled_2q_error_max_percent": (
            float(np.max(twoq_errors) * 100.0)
            if twoq_errors
            else np.nan
        ),
        "operations": json.dumps(
            counts,
            sort_keys=True,
        ),
    }


# =============================================================================
# RANKING
# =============================================================================

def calibration_hierarchy_sort(df):
    """
    Deterministic non-weighted hierarchy for shortlisting.
    Pareto status is handled first outside this function.
    """
    return df.sort_values(
        [
            "max_cz_error",
            "mean_cz_error",
            "max_readout_error",
            "mean_readout_error",
            "min_t2_us",
            "min_t1_us",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
            False,
        ],
    )


def final_hierarchy_sort(df):
    """
    Circuit-aware non-weighted hierarchy after strict transpilation.
    """
    return df.sort_values(
        [
            "compiled_n_swap",
            "compiled_2q_error_max",
            "compiled_2q_error_mean_occurrence",
            "max_readout_error",
            "compiled_1q_error_max",
            "min_t2_us",
            "min_t1_us",
            "compiled_duration_us",
            "compiled_depth_2q",
            "compiled_depth",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
        ],
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        default="all",
        choices=["all", *BACKENDS],
        help="Refresh all allocated backends or only one.",
    )

    parser.add_argument(
        "--shortlist",
        type=int,
        default=SHORTLIST_PER_BACKEND,
        help=(
            "Maximum calibration-screened embeddings compiled per backend. "
            "All Pareto embeddings are retained even if this number is smaller."
        ),
    )

    args = parser.parse_args()

    candidate = select_best_candidate()
    angles, input_audit = first_input_angles()
    core = build_core(candidate, angles)

    service = get_service()

    backend_names = (
        BACKENDS
        if args.backend == "all"
        else [args.backend]
    )

    candidate_json = {
        **{
            k: v
            for k, v in candidate.items()
            if k != "J"
        },
        "J": {
            f"{i}{j}": float(v)
            for (i, j), v in candidate["J"].items()
        },
        "input": input_audit,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(
        PREFIX.with_name(
            PREFIX.name + "_selected_candidate.json"
        ),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(candidate_json),
            fp,
            indent=2,
        )

    all_rows = []
    compiled_rows = []
    backend_summary_rows = []

    print("=" * 118)
    print("WEEK 11.0A — LIVE EMBEDDING RESELECTION")
    print("=" * 118)
    print(f"Candidate: {candidate['candidate_id']}")
    print(f"Source:    {candidate['selection_source']}")
    print(f"Topology:  {candidate['topology']}")
    print(f"r:         {candidate['r']}")
    print(f"hx / hy:   {candidate['hx']:+.6f} / {candidate['hy']:+.6f}")
    print(f"Readout:   {candidate['readout']}")
    print(f"CV RMSE:   {candidate['cv_rmse']:.6f}")
    print("NO QPU JOB WILL BE SUBMITTED.")
    print()

    for backend_name in backend_names:
        print("-" * 118)
        print(f"Refreshing {backend_name}")
        print("-" * 118)

        backend = service.backend(
            backend_name,
            use_fractional_gates=False,
        )

        status = backend.status()
        properties = backend.properties(refresh=True)

        if properties is None:
            print(f"{backend_name}: properties unavailable — skipped")
            continue

        faulty_qubits = get_faulty_qubits(
            properties
        )

        physical_adj = build_undirected_cz_graph(
            backend
        )

        mappings = enumerate_native_embeddings(
            candidate["topology"],
            physical_adj,
            faulty_qubits,
        )

        print(
            f"Operational={status.operational}, "
            f"pending_jobs={status.pending_jobs}, "
            f"faulty_qubits={sorted(faulty_qubits)}, "
            f"native mappings={len(mappings)}"
        )

        region_rows = []

        for layout in mappings:
            metrics = region_calibration_metrics(
                backend,
                properties,
                candidate["topology"],
                list(layout),
            )

            row = {
                "backend": backend_name,
                "pending_jobs": int(status.pending_jobs),
                "operational": bool(status.operational),
                "topology": candidate["topology"],
                "candidate_id": candidate["candidate_id"],
                "layout": json.dumps(list(layout)),
                "physical_C_t": int(layout[0]),
                "physical_D": int(layout[1]),
                "physical_P_t": int(layout[2]),
                "physical_H": int(layout[3]),
                "physical_M1": int(layout[4]),
                "physical_M2": int(layout[5]),
                **metrics,
            }

            region_rows.append(row)

        if not region_rows:
            print(f"{backend_name}: no valid native embeddings.")
            continue

        region_df = pd.DataFrame(region_rows)

        n_incomplete = int(
            (
                region_df["calibration_complete"]
                != True
            ).sum()
        )

        if n_incomplete:
            print(
                f"Excluded from ranking because live calibration is incomplete: "
                f"{n_incomplete}/{len(region_df)} embeddings"
            )

        complete_region_df = region_df[
            region_df["calibration_complete"]
            == True
        ].copy()

        if len(complete_region_df) == 0:
            print(
                f"{backend_name}: no embeddings with complete "
                f"T1/T2/readout/CZ calibration."
            )
            all_rows.extend(region_df.to_dict("records"))
            continue

        complete_region_df[
            "calibration_pareto"
        ] = pareto_mask(
            complete_region_df,
            minimize_cols=[
                "max_cz_error",
                "mean_cz_error",
                "max_readout_error",
                "mean_readout_error",
            ],
            maximize_cols=[
                "min_t2_us",
                "min_t1_us",
            ],
        )

        # Save complete and incomplete mappings for audit.
        all_rows.extend(
            region_df.to_dict("records")
        )

        # Compile every Pareto embedding plus the first N under the explicit
        # non-weighted calibration hierarchy.
        pareto_df = complete_region_df[
            complete_region_df["calibration_pareto"]
        ].copy()

        hierarchy_df = calibration_hierarchy_sort(
            complete_region_df
        ).head(
            max(
                int(args.shortlist),
                1,
            )
        )

        shortlist_df = (
            pd.concat(
                [
                    pareto_df,
                    hierarchy_df,
                ],
                ignore_index=True,
            )
            .drop_duplicates(
                subset=["layout"]
            )
            .reset_index(drop=True)
        )

        print(
            f"Calibration Pareto={len(pareto_df)}, "
            f"strict-compile shortlist={len(shortlist_df)}"
        )

        for _, region in shortlist_df.iterrows():
            layout = json.loads(
                region["layout"]
            )

            try:
                compiled = transpile(
                    core,
                    backend=backend,
                    initial_layout=layout,
                    routing_method="none",
                    optimization_level=OPTIMIZATION_LEVEL,
                    seed_transpiler=SEED_TRANSPILER,
                    scheduling_method="alap",
                )

                # IBM-specific fault audit.
                backend.check_faulty(
                    compiled
                )

                resources = compiled_resource_metrics(
                    compiled,
                    backend,
                )

                strict_pass = (
                    resources[
                        "compiled_n_swap"
                    ]
                    ==
                    0
                )

                error_text = ""

            except Exception as exc:
                resources = {}
                strict_pass = False
                error_text = str(exc)

            out = region.to_dict()
            out["strict_compile_pass"] = bool(strict_pass)
            out["compile_error"] = error_text
            out.update(resources)

            if (
                strict_pass
                and
                np.isfinite(
                    out.get(
                        "compiled_duration_us",
                        np.nan,
                    )
                )
            ):
                out["R_T1"] = (
                    out["compiled_duration_us"]
                    /
                    out["min_t1_us"]
                )
                out["R_T2"] = (
                    out["compiled_duration_us"]
                    /
                    out["min_t2_us"]
                )
            else:
                out["R_T1"] = np.nan
                out["R_T2"] = np.nan

            compiled_rows.append(out)

        compiled_backend = pd.DataFrame(
            [
                r
                for r in compiled_rows
                if r["backend"] == backend_name
            ]
        )

        good = compiled_backend[
            compiled_backend[
                "strict_compile_pass"
            ]
            ==
            True
        ].copy()

        if len(good) == 0:
            print(
                f"{backend_name}: no strict native compile survivors."
            )
            continue

        good[
            "final_pareto"
        ] = pareto_mask(
            good,
            minimize_cols=[
                "compiled_2q_error_max",
                "compiled_2q_error_mean_occurrence",
                "max_readout_error",
                "compiled_1q_error_max",
                "compiled_duration_us",
                "R_T2",
            ],
            maximize_cols=[
                "min_t2_us",
                "min_t1_us",
            ],
        )

        # Pareto first, then hierarchy within Pareto; if <3 Pareto, fill from
        # remaining strict survivors.
        front = final_hierarchy_sort(
            good[
                good["final_pareto"]
            ].copy()
        )

        rest = final_hierarchy_sort(
            good[
                ~good["final_pareto"]
            ].copy()
        )

        ranked = pd.concat(
            [front, rest],
            ignore_index=True,
        ).drop_duplicates(
            subset=["layout"]
        )

        ranked["backend_rank"] = (
            np.arange(len(ranked)) + 1
        )

        top = ranked.head(
            TOP_K_PER_BACKEND
        )

        for _, row in top.iterrows():
            print(
                f"rank {int(row['backend_rank'])}: "
                f"layout={row['layout']} | "
                f"CZmax={row['compiled_2q_error_max_percent']:.4f}% | "
                f"ROmax={row['max_readout_error_percent']:.4f}% | "
                f"1Qmax={row['compiled_1q_error_max_percent']:.4f}% | "
                f"T2min={row['min_t2_us']:.2f} us | "
                f"duration={row['compiled_duration_us']:.3f} us | "
                f"R_T2={row['R_T2']:.5f}"
            )

        # Store ranks back into compiled_rows.
        rank_map = {
            row["layout"]: int(row["backend_rank"])
            for _, row in ranked.iterrows()
        }

        for rrow in compiled_rows:
            if rrow["backend"] == backend_name:
                rrow["backend_rank"] = rank_map.get(
                    rrow["layout"],
                    np.nan,
                )

        backend_best = ranked.iloc[0]

        backend_summary_rows.append({
            "backend": backend_name,
            "pending_jobs": int(status.pending_jobs),
            "candidate_id": candidate["candidate_id"],
            "topology": candidate["topology"],
            "best_layout": backend_best["layout"],
            "best_backend_rank": 1,
            "compiled_2q_error_max": backend_best[
                "compiled_2q_error_max"
            ],
            "compiled_1q_error_max": backend_best[
                "compiled_1q_error_max"
            ],
            "max_readout_error": backend_best[
                "max_readout_error"
            ],
            "min_t1_us": backend_best["min_t1_us"],
            "min_t2_us": backend_best["min_t2_us"],
            "compiled_duration_us": backend_best[
                "compiled_duration_us"
            ],
            "R_T1": backend_best["R_T1"],
            "R_T2": backend_best["R_T2"],
            "compiled_n_cz": backend_best[
                "compiled_n_cz"
            ],
            "compiled_n_swap": backend_best[
                "compiled_n_swap"
            ],
        })

    all_df = pd.DataFrame(all_rows)
    compiled_df = pd.DataFrame(compiled_rows)
    backend_summary_df = pd.DataFrame(
        backend_summary_rows
    )

    all_df.to_csv(
        PREFIX.with_name(
            PREFIX.name
            +
            "_all_native_embeddings.csv"
        ),
        index=False,
    )

    compiled_df.to_csv(
        PREFIX.with_name(
            PREFIX.name
            +
            "_compiled_shortlist.csv"
        ),
        index=False,
    )

    backend_summary_df.to_csv(
        PREFIX.with_name(
            PREFIX.name
            +
            "_backend_summary.csv"
        ),
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final top-3 per backend.
    # -------------------------------------------------------------------------

    top_rows = []

    for backend_name, g in compiled_df.groupby(
        "backend"
    ):
        good = g[
            g["strict_compile_pass"]
            ==
            True
        ].copy()

        if len(good) == 0:
            continue

        # backend_rank was assigned above.
        good = good.sort_values(
            "backend_rank"
        ).head(
            TOP_K_PER_BACKEND
        )

        top_rows.extend(
            good.to_dict("records")
        )

    top_df = pd.DataFrame(top_rows)

    if len(top_df) == 0:
        raise RuntimeError(
            "No strict native embeddings survived on any backend."
        )

    # -------------------------------------------------------------------------
    # Overall backend/embedding choice.
    #
    # Non-weighted hierarchy:
    # 1) backend pending jobs,
    # 2) worst used 2Q error,
    # 3) worst readout error,
    # 4) worst used 1Q error,
    # 5) min T2 (higher),
    # 6) min T1 (higher),
    # 7) physical duration,
    # 8) backend rank.
    #
    # This is deliberately explicit rather than a synthetic score.
    # -------------------------------------------------------------------------

    overall = top_df.sort_values(
        [
            "pending_jobs",
            "compiled_2q_error_max",
            "max_readout_error",
            "compiled_1q_error_max",
            "min_t2_us",
            "min_t1_us",
            "compiled_duration_us",
            "backend_rank",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    overall[
        "overall_rank"
    ] = np.arange(
        len(overall)
    ) + 1

    overall.to_csv(
        PREFIX.with_name(
            PREFIX.name
            +
            "_top_embeddings.csv"
        ),
        index=False,
    )

    selected = overall.iloc[0].to_dict()

    with open(
        PREFIX.with_name(
            PREFIX.name
            +
            "_selected_embedding.json"
        ),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(selected),
            fp,
            indent=2,
        )

    manifest = {
        "step": "11.0A",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate_json,
        "backends_checked": backend_names,
        "optimization_level": OPTIMIZATION_LEVEL,
        "routing_method": "none",
        "shortlist_per_backend": int(args.shortlist),
        "top_k_per_backend": TOP_K_PER_BACKEND,
        "weighted_score": False,
        "ranking_policy": (
            "Pareto screening followed by explicit hierarchical ordering; "
            "no weighted aggregate score."
        ),
        "qpu_job_submitted": False,
    }

    with open(
        PREFIX.with_name(
            PREFIX.name
            +
            "_manifest.json"
        ),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(manifest),
            fp,
            indent=2,
        )

    print()
    print("=" * 118)
    print("OVERALL CURRENT SELECTION")
    print("=" * 118)
    print(
        overall[
            [
                "overall_rank",
                "backend",
                "backend_rank",
                "layout",
                "pending_jobs",
                "compiled_2q_error_max_percent",
                "max_readout_error_percent",
                "compiled_1q_error_max_percent",
                "min_t1_us",
                "min_t2_us",
                "compiled_duration_us",
                "R_T1",
                "R_T2",
                "compiled_n_cz",
                "compiled_n_swap",
            ]
        ]
        .head(9)
        .to_string(index=False)
    )

    print()
    print("Saved:")
    for name in [
        "11_0A_selected_candidate.json",
        "11_0A_all_native_embeddings.csv",
        "11_0A_compiled_shortlist.csv",
        "11_0A_backend_summary.csv",
        "11_0A_top_embeddings.csv",
        "11_0A_selected_embedding.json",
        "11_0A_manifest.json",
    ]:
        print(f"  results/{name}")

    print()
    print(
        "11.0A COMPLETE. No QPU job was submitted. "
        "Inspect 11_0A_top_embeddings.csv before running 11.0B."
    )


if __name__ == "__main__":
    main()
