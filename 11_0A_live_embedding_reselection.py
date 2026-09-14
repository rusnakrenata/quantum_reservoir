
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
# WEEK 11.0A — FRESH FULL-HARDWARE RESELECTION
#
# IMPORTANT DESIGN
# ----------------
# This script does NOT use yesterday's physical-qubit embeddings.
#
# Every execution:
#   1. refreshes each backend object,
#   2. refreshes current backend properties,
#   3. reconstructs the CURRENT CZ graph from backend.target,
#   4. scans ALL currently available physical qubits,
#   5. enumerates ALL native 6-qubit embeddings of the selected topology,
#   6. removes mappings containing faulty or incompletely calibrated qubits,
#   7. evaluates live T1/T2/readout/CZ calibration,
#   8. compiles current promising mappings with routing_method="none",
#   9. checks zero SWAP and current instruction errors/duration,
#  10. saves fresh top embeddings.
#
# NO QPU JOB IS SUBMITTED.
# =============================================================================


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DATA_FILE = RESULTS / "03_01_preprocessed_samples.csv"
J_WINNER_FILE = RESULTS / "08_03b_J_search_topology_winners.csv"
FAIR_FILE = RESULTS / "09_03c_forecast_best_135.csv"
R_ABLATION_FILE = RESULTS / "09_03_trotter_forecast_accuracy.csv"
FINAL10_FILE = RESULTS / "09_01y_final_10_candidates.csv"

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
DEFAULT_SHORTLIST = 80
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


# =============================================================================
# GENERIC
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


def pareto_mask(frame, minimize, maximize=None):
    if maximize is None:
        maximize = []

    cols = []
    for c in minimize:
        cols.append(
            pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        )
    for c in maximize:
        cols.append(
            -pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        )

    M = np.column_stack(cols)
    finite = np.all(np.isfinite(M), axis=1)
    keep = np.zeros(len(frame), dtype=bool)
    valid_idx = np.where(finite)[0]
    V = M[finite]

    for local_i, global_i in enumerate(valid_idx):
        v = V[local_i]
        no_worse = np.all(V <= v, axis=1)
        strictly_better = np.any(V < v, axis=1)
        dominated = no_worse & strictly_better
        dominated[local_i] = False

        if not np.any(dominated):
            keep[global_i] = True

    return keep


# =============================================================================
# LOGICAL CANDIDATE
# =============================================================================

def week8_J(topology):
    jdf = pd.read_csv(J_WINNER_FILE)
    match = jdf[jdf["topology"] == topology]

    if len(match) != 1:
        raise RuntimeError(
            f"{topology}: expected one Week-8 J winner, got {len(match)}"
        )

    row = match.iloc[0]
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
    if FAIR_FILE.exists():
        df = pd.read_csv(FAIR_FILE)

        if len(df) > 0:
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
                "candidate_id": str(row["seed_candidate_id"]),
                "topology": topology,
                "readout": str(row["readout"]),
                "r": int(row["trotter_r"]),
                "hx": float(row["hx"]),
                "hy": float(row["hy"]),
                "cv_rmse": float(row["cv_rmse"]),
                "validation_rmse": (
                    float(row["validation_rmse"])
                    if pd.notna(row.get("validation_rmse", np.nan))
                    else np.nan
                ),
                "J": J_from_row_or_week8(row, topology),
            }

    if R_ABLATION_FILE.exists():
        df = pd.read_csv(R_ABLATION_FILE)

        if len(df) > 0:
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
                "candidate_id": str(row["candidate_id"]),
                "topology": topology,
                "readout": str(row["readout"]),
                "r": int(row["trotter_r"]),
                "hx": REFERENCE_HX,
                "hy": float(row["hy"]),
                "cv_rmse": float(row["cv_rmse"]),
                "validation_rmse": (
                    float(row["validation_rmse"])
                    if pd.notna(row.get("validation_rmse", np.nan))
                    else np.nan
                ),
                "J": week8_J(topology),
            }

    if FINAL10_FILE.exists():
        df = pd.read_csv(FINAL10_FILE)
        row = df.sort_values(
            ["cv_rmse", "validation_rmse"]
        ).iloc[0]

        topology = str(row["topology"])

        return {
            "selection_source": str(FINAL10_FILE),
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

    raise FileNotFoundError("No candidate result file found.")


# =============================================================================
# FIRST REAL INPUT
# =============================================================================

def first_input_angles():
    df = pd.read_csv(DATA_FILE)
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["input_date"] = pd.to_datetime(df["input_date"])
    row = df.sort_values("target_date").iloc[0]

    claim = float(np.clip(float(row["C_t_z"]) / 3.0, -1.0, 1.0))
    policy_z = float(row["P_t_z"])
    policy = policy_z / (3.0 + abs(policy_z))
    weekday = math.atan2(
        float(row["D_sin"]),
        float(row["D_cos"]),
    ) % (2.0 * math.pi)

    holiday = math.pi * int(
        row["is_public_holiday_t_plus_1"]
    )

    angles = np.array([
        ALPHA * claim,
        weekday,
        ALPHA * policy,
        holiday,
    ], dtype=float)

    audit = {
        "input_date": str(row["input_date"].date()),
        "target_date": str(row["target_date"].date()),
        "angles": angles.tolist(),
    }

    return angles, audit


def build_core(candidate, angles):
    qc = QuantumCircuit(6, name="qrc_core")

    for q in range(4):
        qc.ry(float(angles[q]), q)

    topology = candidate["topology"]
    r = int(candidate["r"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])
    J = candidate["J"]

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
# CURRENT HARDWARE — NO OLD PHYSICAL MAPPINGS
# =============================================================================

def refresh_backend_fully(service, backend_name):
    backend = service.backend(
        backend_name,
        use_fractional_gates=False,
    )

    # Refresh configuration + Target itself.
    try:
        backend.refresh()
    except Exception:
        # Runtime versions differ; properties(refresh=True) below still forces
        # a server-side property refresh.
        pass

    properties = backend.properties(
        refresh=True
    )

    if properties is None:
        raise RuntimeError(
            f"{backend_name}: current backend properties unavailable."
        )

    return backend, properties


def safe_qubit_properties(backend, properties, q):
    """
    Missing T1/T2 is expected to be possible on live calibration data.
    Never call properties.t2(q) without catching exceptions.
    """
    t1 = np.nan
    t2 = np.nan

    # Preferred BackendV2 interface.
    try:
        qp = backend.qubit_properties(int(q))

        if qp is not None:
            raw_t1 = getattr(qp, "t1", None)
            raw_t2 = getattr(qp, "t2", None)

            if raw_t1 is not None and np.isfinite(float(raw_t1)):
                t1 = float(raw_t1) * 1e6

            if raw_t2 is not None and np.isfinite(float(raw_t2)):
                t2 = float(raw_t2) * 1e6
    except Exception:
        pass

    # Independent fallback queries.
    if not np.isfinite(t1):
        try:
            raw = float(properties.t1(int(q)))
            if np.isfinite(raw):
                t1 = raw * 1e6
        except Exception:
            pass

    if not np.isfinite(t2):
        try:
            raw = float(properties.t2(int(q)))
            if np.isfinite(raw):
                t2 = raw * 1e6
        except Exception:
            pass

    # Readout error: prefer Target measurement error.
    readout = np.nan

    try:
        p = backend.target["measure"][(int(q),)]
        if p is not None and p.error is not None:
            readout = float(p.error)
    except Exception:
        pass

    if not np.isfinite(readout):
        try:
            readout = float(
                properties.readout_error(
                    int(q)
                )
            )
        except Exception:
            pass

    return {
        "qubit": int(q),
        "t1_us": t1,
        "t2_us": t2,
        "readout_error": readout,
        "complete": bool(
            np.isfinite(t1)
            and np.isfinite(t2)
            and np.isfinite(readout)
        ),
    }


def current_cz_graph(backend):
    """
    Build the CURRENT CZ graph from freshly refreshed backend.target.
    """
    target = backend.target

    try:
        cmap = target.build_coupling_map(
            two_q_gate="cz",
            filter_idle_qubits=True,
        )
    except TypeError:
        cmap = target.build_coupling_map(
            two_q_gate="cz",
        )

    if cmap is None:
        raise RuntimeError(
            f"{backend.name}: target has no CZ coupling map."
        )

    adjacency = {
        q: set()
        for q in range(
            backend.num_qubits
        )
    }

    edges = []

    for i, j in cmap.get_edges():
        i = int(i)
        j = int(j)
        adjacency[i].add(j)
        adjacency[j].add(i)
        edges.append((i, j))

    return adjacency, edges


def current_faulty_qubits(properties):
    try:
        return set(
            int(q)
            for q in properties.faulty_qubits()
        )
    except Exception:
        return set()


def cz_error(backend, i, j):
    values = []

    for qargs in [
        (int(i), int(j)),
        (int(j), int(i)),
    ]:
        try:
            props = backend.target["cz"][qargs]
            if (
                props is not None
                and props.error is not None
                and np.isfinite(float(props.error))
            ):
                values.append(
                    float(props.error)
                )
        except Exception:
            pass

    return (
        float(min(values))
        if values
        else np.nan
    )


# =============================================================================
# ENUMERATE EVERY NATIVE EMBEDDING
# =============================================================================

def logical_adjacency(topology):
    adj = {
        q: set()
        for q in range(6)
    }

    for i, j in TOPOLOGY_EDGES[topology]:
        adj[i].add(j)
        adj[j].add(i)

    return adj


def enumerate_all_native_embeddings(
    topology,
    physical_adj,
    eligible_qubits,
):
    logical_adj = logical_adjacency(topology)

    logical_order = sorted(
        range(6),
        key=lambda q: (
            -len(logical_adj[q]),
            q,
        ),
    )

    eligible = set(
        int(q)
        for q in eligible_qubits
    )

    degree_candidates = {
        lq: [
            pq
            for pq in eligible
            if len(
                physical_adj.get(
                    pq,
                    set(),
                )
                &
                eligible
            )
            >=
            len(logical_adj[lq])
        ]
        for lq in range(6)
    }

    assigned = {}
    used = set()
    mappings = []

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

        assigned_neighbors = [
            n
            for n in logical_adj[lq]
            if n in assigned
        ]

        if assigned_neighbors:
            candidates = None

            for n in assigned_neighbors:
                neigh = (
                    physical_adj[
                        assigned[n]
                    ]
                    &
                    eligible
                )

                candidates = (
                    set(neigh)
                    if candidates is None
                    else candidates & neigh
                )

            candidates = sorted(
                candidates
                &
                set(
                    degree_candidates[lq]
                )
            )
        else:
            candidates = sorted(
                degree_candidates[lq]
            )

        for pq in candidates:
            if pq in used:
                continue

            if any(
                assigned[n]
                not in
                physical_adj[pq]
                for n in assigned_neighbors
            ):
                continue

            assigned[lq] = pq
            used.add(pq)

            rec(pos + 1)

            used.remove(pq)
            del assigned[lq]

    rec(0)

    return list(
        dict.fromkeys(
            mappings
        )
    )


# =============================================================================
# LIVE METRICS
# =============================================================================

def mapping_metrics(
    backend,
    topology,
    layout,
    qubit_table,
):
    qrows = qubit_table.set_index(
        "qubit"
    )

    t1 = np.array(
        [
            float(
                qrows.loc[
                    q,
                    "t1_us"
                ]
            )
            for q in layout
        ],
        dtype=float,
    )

    t2 = np.array(
        [
            float(
                qrows.loc[
                    q,
                    "t2_us"
                ]
            )
            for q in layout
        ],
        dtype=float,
    )

    ro = np.array(
        [
            float(
                qrows.loc[
                    q,
                    "readout_error"
                ]
            )
            for q in layout
        ],
        dtype=float,
    )

    cz = []

    for li, lj in TOPOLOGY_EDGES[topology]:
        pi = layout[li]
        pj = layout[lj]

        e = cz_error(
            backend,
            pi,
            pj,
        )

        if not np.isfinite(e):
            return None

        cz.append(e)

    cz = np.array(
        cz,
        dtype=float,
    )

    return {
        "min_t1_us": float(np.min(t1)),
        "mean_t1_us": float(np.mean(t1)),
        "min_t2_us": float(np.min(t2)),
        "mean_t2_us": float(np.mean(t2)),
        "max_readout_error": float(np.max(ro)),
        "mean_readout_error": float(np.mean(ro)),
        "max_readout_error_percent": float(np.max(ro) * 100.0),
        "mean_readout_error_percent": float(np.mean(ro) * 100.0),
        "max_cz_error": float(np.max(cz)),
        "mean_cz_error": float(np.mean(cz)),
        "max_cz_error_percent": float(np.max(cz) * 100.0),
        "mean_cz_error_percent": float(np.mean(cz) * 100.0),
    }


def instruction_error(backend, op_name, qargs):
    try:
        props = backend.target[
            op_name
        ][tuple(qargs)]

        if (
            props is not None
            and props.error is not None
            and np.isfinite(float(props.error))
        ):
            return float(props.error)
    except Exception:
        pass

    return np.nan


def compiled_metrics(
    circuit,
    backend,
):
    counts = {
        str(k): int(v)
        for k, v in circuit.count_ops().items()
    }

    oneq_errors = []
    twoq_errors = []
    n1 = 0
    n2 = 0

    excluded = {
        "measure",
        "reset",
        "delay",
        "barrier",
    }

    for inst in circuit.data:
        name = inst.operation.name
        qargs = tuple(
            circuit.find_bit(q).index
            for q in inst.qubits
        )

        if len(qargs) == 1 and name not in excluded:
            n1 += 1
            e = instruction_error(
                backend,
                name,
                qargs,
            )

            if np.isfinite(e):
                oneq_errors.append(e)

        elif len(qargs) == 2:
            n2 += 1
            e = instruction_error(
                backend,
                name,
                qargs,
            )

            if (
                not np.isfinite(e)
                and name == "cz"
            ):
                e = instruction_error(
                    backend,
                    name,
                    qargs[::-1],
                )

            if np.isfinite(e):
                twoq_errors.append(e)

    duration_s = float(
        circuit.estimate_duration(
            backend.target,
            unit="s",
        )
    )

    return {
        "compiled_depth": int(circuit.depth()),
        "compiled_size": int(circuit.size()),
        "compiled_n_1q": n1,
        "compiled_n_2q": n2,
        "compiled_n_cz": int(counts.get("cz", 0)),
        "compiled_n_swap": int(counts.get("swap", 0)),
        "compiled_duration_us": duration_s * 1e6,
        "compiled_1q_error_max": (
            float(np.max(oneq_errors))
            if oneq_errors
            else np.nan
        ),
        "compiled_1q_error_mean": (
            float(np.mean(oneq_errors))
            if oneq_errors
            else np.nan
        ),
        "compiled_2q_error_max": (
            float(np.max(twoq_errors))
            if twoq_errors
            else np.nan
        ),
        "compiled_2q_error_mean": (
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
# FRESH SELECTION FUNCTION — USED BY BOTH 11.0A AND 11.0B
# =============================================================================

def fresh_hardware_reselection(
    service,
    candidate,
    backend_names=None,
    shortlist=DEFAULT_SHORTLIST,
    write_prefix="11_0A",
    verbose=True,
):
    """
    Re-scan ALL current physical qubits and CURRENT CZ connectivity.
    Does not read any historical embedding CSV.
    """
    if backend_names is None:
        backend_names = BACKENDS

    angles, input_audit = first_input_angles()
    core = build_core(
        candidate,
        angles,
    )

    all_qubit_rows = []
    all_mapping_rows = []
    compiled_rows = []
    backend_status_rows = []

    for backend_name in backend_names:
        if verbose:
            print()
            print("-" * 120)
            print(f"FRESH REFRESH: {backend_name}")
            print("-" * 120)

        backend, properties = (
            refresh_backend_fully(
                service,
                backend_name,
            )
        )

        status = backend.status()
        faulty = current_faulty_qubits(
            properties
        )

        physical_adj, cz_edges = (
            current_cz_graph(
                backend
            )
        )

        qubit_rows = []

        for q in range(
            backend.num_qubits
        ):
            info = safe_qubit_properties(
                backend,
                properties,
                q,
            )

            info[
                "backend"
            ] = backend_name

            info[
                "faulty"
            ] = (
                q in faulty
            )

            info[
                "eligible"
            ] = bool(
                info["complete"]
                and
                q not in faulty
            )

            qubit_rows.append(info)

        qubit_df = pd.DataFrame(
            qubit_rows
        )

        all_qubit_rows.extend(
            qubit_rows
        )

        eligible_qubits = (
            qubit_df.loc[
                qubit_df[
                    "eligible"
                ]
                ==
                True,
                "qubit",
            ]
            .astype(int)
            .tolist()
        )

        mappings = enumerate_all_native_embeddings(
            candidate["topology"],
            physical_adj,
            eligible_qubits,
        )

        backend_status_rows.append({
            "backend": backend_name,
            "operational": bool(status.operational),
            "pending_jobs": int(status.pending_jobs),
            "num_qubits": int(backend.num_qubits),
            "current_cz_directed_edge_count": len(cz_edges),
            "faulty_qubit_count": len(faulty),
            "complete_calibration_qubit_count": int(
                qubit_df["complete"].sum()
            ),
            "eligible_qubit_count": len(eligible_qubits),
            "native_embedding_count": len(mappings),
            "properties_last_update": json_safe(
                getattr(
                    properties,
                    "last_update_date",
                    None,
                )
            ),
        })

        if verbose:
            print(
                f"qubits={backend.num_qubits}, "
                f"eligible={len(eligible_qubits)}, "
                f"faulty={sorted(faulty)}, "
                f"current CZ edges={len(cz_edges)}, "
                f"native H{candidate['topology'][-1]} embeddings={len(mappings)}"
            )

        mapping_rows = []

        for layout_tuple in mappings:
            layout = list(
                layout_tuple
            )

            metrics = mapping_metrics(
                backend,
                candidate["topology"],
                layout,
                qubit_df,
            )

            if metrics is None:
                continue

            row = {
                "backend": backend_name,
                "pending_jobs": int(status.pending_jobs),
                "layout": json.dumps(layout),
                "physical_C_t": layout[0],
                "physical_D": layout[1],
                "physical_P_t": layout[2],
                "physical_H": layout[3],
                "physical_M1": layout[4],
                "physical_M2": layout[5],
                **metrics,
            }

            mapping_rows.append(row)

        if not mapping_rows:
            if verbose:
                print("No fully calibrated native embeddings survived.")
            continue

        mapping_df = pd.DataFrame(
            mapping_rows
        )

        mapping_df[
            "calibration_pareto"
        ] = pareto_mask(
            mapping_df,
            minimize=[
                "max_cz_error",
                "mean_cz_error",
                "max_readout_error",
                "mean_readout_error",
            ],
            maximize=[
                "min_t2_us",
                "min_t1_us",
            ],
        )

        all_mapping_rows.extend(
            mapping_df.to_dict("records")
        )

        calibration_hierarchy = mapping_df.sort_values(
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

        pareto_df = mapping_df[
            mapping_df[
                "calibration_pareto"
            ]
            ==
            True
        ]

        shortlist_df = (
            pd.concat(
                [
                    pareto_df,
                    calibration_hierarchy.head(
                        max(
                            int(shortlist),
                            1,
                        )
                    ),
                ],
                ignore_index=True,
            )
            .drop_duplicates(
                subset=[
                    "layout"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        if verbose:
            print(
                f"calibration Pareto={len(pareto_df)}, "
                f"strict-compile shortlist={len(shortlist_df)}"
            )

        backend_compiled = []

        for _, region in shortlist_df.iterrows():
            layout = json.loads(
                region[
                    "layout"
                ]
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

                backend.check_faulty(
                    compiled
                )

                cm = compiled_metrics(
                    compiled,
                    backend,
                )

                strict_pass = (
                    cm[
                        "compiled_n_swap"
                    ]
                    ==
                    0
                )

                compile_error = ""

            except Exception as exc:
                cm = {}
                strict_pass = False
                compile_error = str(exc)

            row = region.to_dict()
            row[
                "strict_compile_pass"
            ] = bool(
                strict_pass
            )
            row[
                "compile_error"
            ] = compile_error
            row.update(cm)

            if (
                strict_pass
                and
                np.isfinite(
                    row.get(
                        "compiled_duration_us",
                        np.nan,
                    )
                )
            ):
                row[
                    "R_T1"
                ] = (
                    row[
                        "compiled_duration_us"
                    ]
                    /
                    row[
                        "min_t1_us"
                    ]
                )

                row[
                    "R_T2"
                ] = (
                    row[
                        "compiled_duration_us"
                    ]
                    /
                    row[
                        "min_t2_us"
                    ]
                )
            else:
                row["R_T1"] = np.nan
                row["R_T2"] = np.nan

            backend_compiled.append(
                row
            )
            compiled_rows.append(
                row
            )

        good = pd.DataFrame(
            backend_compiled
        )

        good = good[
            good[
                "strict_compile_pass"
            ]
            ==
            True
        ].copy()

        if len(good) == 0:
            if verbose:
                print(
                    "No zero-SWAP strict compilation survived."
                )
            continue

        good[
            "final_pareto"
        ] = pareto_mask(
            good,
            minimize=[
                "compiled_2q_error_max",
                "compiled_2q_error_mean",
                "max_readout_error",
                "compiled_1q_error_max",
                "compiled_duration_us",
                "R_T2",
            ],
            maximize=[
                "min_t2_us",
                "min_t1_us",
            ],
        )

        front = good[
            good[
                "final_pareto"
            ]
            ==
            True
        ].copy()

        rest = good[
            good[
                "final_pareto"
            ]
            !=
            True
        ].copy()

        hierarchy_cols = [
            "compiled_2q_error_max",
            "compiled_2q_error_mean",
            "max_readout_error",
            "compiled_1q_error_max",
            "min_t2_us",
            "min_t1_us",
            "compiled_duration_us",
        ]

        hierarchy_ascending = [
            True,
            True,
            True,
            True,
            False,
            False,
            True,
        ]

        front = front.sort_values(
            hierarchy_cols,
            ascending=hierarchy_ascending,
        )

        rest = rest.sort_values(
            hierarchy_cols,
            ascending=hierarchy_ascending,
        )

        ranked = (
            pd.concat(
                [
                    front,
                    rest,
                ],
                ignore_index=True,
            )
            .drop_duplicates(
                subset=[
                    "layout"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        rank_map = {
            row[
                "layout"
            ]:
                i + 1
            for i, (_, row)
            in enumerate(
                ranked.iterrows()
            )
        }

        for row in compiled_rows:
            if (
                row[
                    "backend"
                ]
                ==
                backend_name
            ):
                row[
                    "backend_rank"
                ] = rank_map.get(
                    row[
                        "layout"
                    ],
                    np.nan,
                )

        if verbose:
            print(
                ranked[
                    [
                        "layout",
                        "compiled_2q_error_max_percent",
                        "max_readout_error_percent",
                        "compiled_1q_error_max_percent",
                        "min_t2_us",
                        "compiled_duration_us",
                        "R_T2",
                    ]
                ]
                .head(
                    TOP_K_PER_BACKEND
                )
                .to_string(
                    index=False
                )
            )

    qubits_df = pd.DataFrame(
        all_qubit_rows
    )

    mappings_df = pd.DataFrame(
        all_mapping_rows
    )

    compiled_df = pd.DataFrame(
        compiled_rows
    )

    status_df = pd.DataFrame(
        backend_status_rows
    )

    if len(compiled_df) == 0:
        raise RuntimeError(
            "No backend produced a valid fresh zero-SWAP embedding."
        )

    good = compiled_df[
        compiled_df[
            "strict_compile_pass"
        ]
        ==
        True
    ].copy()

    # Top 3 per backend.
    top_parts = []

    for backend_name, group in good.groupby(
        "backend"
    ):
        top_parts.append(
            group
            .sort_values(
                "backend_rank"
            )
            .head(
                TOP_K_PER_BACKEND
            )
        )

    top_df = pd.concat(
        top_parts,
        ignore_index=True,
    )

    # Overall selection — explicit hierarchy, no weighted score.
    top_df = top_df.sort_values(
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

    top_df[
        "overall_rank"
    ] = np.arange(
        len(top_df)
    ) + 1

    prefix = RESULTS / write_prefix

    qubits_df.to_csv(
        prefix.with_name(
            prefix.name
            +
            "_fresh_qubits.csv"
        ),
        index=False,
    )

    status_df.to_csv(
        prefix.with_name(
            prefix.name
            +
            "_fresh_backend_status.csv"
        ),
        index=False,
    )

    mappings_df.to_csv(
        prefix.with_name(
            prefix.name
            +
            "_all_fresh_native_embeddings.csv"
        ),
        index=False,
    )

    compiled_df.to_csv(
        prefix.with_name(
            prefix.name
            +
            "_fresh_compiled_shortlist.csv"
        ),
        index=False,
    )

    top_df.to_csv(
        prefix.with_name(
            prefix.name
            +
            "_fresh_top_embeddings.csv"
        ),
        index=False,
    )

    return {
        "candidate": candidate,
        "input_audit": input_audit,
        "backend_status": status_df,
        "qubits": qubits_df,
        "mappings": mappings_df,
        "compiled": compiled_df,
        "top": top_df,
        "selected": top_df.iloc[0].to_dict(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        default="all",
        choices=[
            "all",
            *BACKENDS,
        ],
    )

    parser.add_argument(
        "--shortlist",
        type=int,
        default=DEFAULT_SHORTLIST,
    )

    args = parser.parse_args()

    candidate = select_best_candidate()
    service = get_service()

    backend_names = (
        BACKENDS
        if args.backend == "all"
        else [
            args.backend
        ]
    )

    print("=" * 120)
    print("WEEK 11.0A — FRESH FULL-HARDWARE RESELECTION")
    print("=" * 120)
    print(
        "No historical physical-qubit map will be used."
    )
    print(
        "Every backend target, every qubit calibration and current CZ "
        "connectivity are refreshed now."
    )
    print()
    print(
        f"Candidate: {candidate['candidate_id']}"
    )
    print(
        f"Topology:  {candidate['topology']}"
    )
    print(
        f"r:         {candidate['r']}"
    )
    print(
        f"hx/hy:     {candidate['hx']:+.6f} / {candidate['hy']:+.6f}"
    )
    print(
        f"Readout:   {candidate['readout']}"
    )
    print(
        f"CV RMSE:   {candidate['cv_rmse']:.6f}"
    )

    result = fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=backend_names,
        shortlist=args.shortlist,
        write_prefix="11_0A",
        verbose=True,
    )

    selected = result[
        "selected"
    ]

    candidate_out = {
        **{
            k: v
            for k, v in candidate.items()
            if k != "J"
        },
        "J": {
            f"{i}{j}": float(v)
            for (
                i,
                j,
            ), v
            in candidate[
                "J"
            ].items()
        },
        "refreshed_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    with open(
        RESULTS
        /
        "11_0A_selected_candidate.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(
                candidate_out
            ),
            fp,
            indent=2,
        )

    with open(
        RESULTS
        /
        "11_0A_fresh_selected_embedding.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(
                selected
            ),
            fp,
            indent=2,
        )

    print()
    print("=" * 120)
    print("FRESH OVERALL TOP EMBEDDINGS")
    print("=" * 120)

    print(
        result[
            "top"
        ][
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
                "R_T2",
                "compiled_n_cz",
                "compiled_n_swap",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "11.0A COMPLETE — no QPU job submitted."
    )
    print(
        "Key result: results/11_0A_fresh_top_embeddings.csv"
    )


if __name__ == "__main__":
    main()
