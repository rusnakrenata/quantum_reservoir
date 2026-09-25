from __future__ import annotations

"""
WEEK 11.1E.6 — CANDIDATE #5 MEMORY-AWARE LAYOUT RESELECTION
============================================================

Goal
----
Candidate #5 narrowly failed the first memory-specific coherence check on one
fresh Kingston layout:

    K_RWP = 64
    actual max-setting duration ~= 232.904 us

    q4 -> P126: duration/T1 ~= 1.159, duration/T2 ~= 0.791
    q5 -> P127: duration/T1 ~= 0.667, duration/T2 ~= 0.899

That is close enough that ONE layout is not sufficient to declare the
candidate globally hardware-infeasible.

This script therefore performs a new zero-QPU, memory-aware H2 embedding search.

It will:
    1. refresh the current backend calibration,
    2. enumerate native six-qubit H2 embeddings directly from the current
       coupling graph,
    3. evaluate T1/T2 for ALL SIX physical qubits in every embedding,
    4. explicitly test whether BOTH persistent memory qubits q4,q5 satisfy

           T1 > t_setting_est
           T2 > t_setting_est

       where t_setting_est is taken from the actual previous RWP64 preflight,
    5. rank layouts lexicographically, without a weighted score:
           a) estimated memory-coherence feasibility first,
           b) larger minimum memory coherence margin,
           c) lower worst required CZ error,
           d) lower worst readout error on injection qubits,
    6. select AT LEAST 3 genuinely different layouts, preferring different
       physical memory-qubit pairs,
    7. compile the ACTUAL K=64 RWP circuits for first/middle/last 2025
       validation endpoints and both X/Z settings for EACH selected layout,
    8. recompute t_setting/T1 and t_setting/T2 using each layout's ACTUAL
       compiled max-setting duration,
    9. report T1/T2 for every logical/physical qubit in every tested layout,
       with memory q4,q5 highlighted,
   10. recommend:
           - small QPU pilot only if at least one layout passes BOTH memory
             T1 and T2 < 1 after actual compilation,
           - otherwise close Candidate #5 as ideal/simulation-only.

Important
---------
The full-setting coherence criterion is physically REQUIRED only for the two
persistent memory qubits q4,q5. Injection qubits q0..q3 are reset/reprepared
between input steps. Nevertheless, per request, this script reports
t_setting/T1 and t_setting/T2 for ALL SIX qubits for every tested layout.

NO QPU JOB IS SUBMITTED.
2026 IS NEVER LOADED.
"""

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    transpile,
)

from ibm_account import get_service
import db_objects as db


# =============================================================================
# Paths / frozen setup
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

CANDIDATE_SCRIPT = HERE / "11_1E1_candidate5_washout_trace_distance.py"

FROZEN_PACKAGE = (
    RESULTS
    / "11_1E3_candidate5_RWP64_frozen_package.json"
)

PREVIOUS_PREFLIGHT = (
    RESULTS
    / "11_1E4_candidate5_RWP64_preflight_summary.json"
)

EXPECTED_CANDIDATE = "CONT_H2_R2"
EXPECTED_TOPOLOGY = "H2"
EXPECTED_R = 2
EXPECTED_K = 64
EXPECTED_READOUT = "XZinj"

DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_N_LAYOUTS = 3
DEFAULT_SHOTS = 1024

OPT_LEVEL = 1
SEED_TRANSPILE = 42

OUT_ALL_EMBEDDINGS = (
    RESULTS
    / "11_1E6_candidate5_all_memory_aware_embeddings.csv"
)

OUT_SELECTED_LAYOUTS = (
    RESULTS
    / "11_1E6_candidate5_selected_layouts.csv"
)

OUT_COMPILED_SETTINGS = (
    RESULTS
    / "11_1E6_candidate5_compiled_settings.csv"
)

OUT_QUBIT_COHERENCE = (
    RESULTS
    / "11_1E6_candidate5_layout_qubit_coherence.csv"
)

OUT_LAYOUT_SUMMARY = (
    RESULTS
    / "11_1E6_candidate5_layout_summary.csv"
)

OUT_JSON = (
    RESULTS
    / "11_1E6_candidate5_memory_aware_reselection_summary.json"
)


# =============================================================================
# Module loading
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required project file not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


candmod = load_module(
    CANDIDATE_SCRIPT,
    "qrc_11_1e6_candidate",
)

common = candmod.common


# =============================================================================
# Generic helpers
# =============================================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def safe_float(value):
    try:
        value = float(
            value
        )
    except Exception:
        return None

    if not np.isfinite(
        value
    ):
        return None

    return value


def seconds_to_us(value):
    if value is None:
        return None

    value = float(
        value
    )

    if not np.isfinite(
        value
    ):
        return None

    return value * 1e6


# =============================================================================
# Backend calibration extraction
# =============================================================================

def get_qubit_t1_t2_us(
    backend,
    q,
):
    props = getattr(
        backend.target,
        "qubit_properties",
        None,
    )

    if props is None:
        raise RuntimeError(
            "Backend target does not expose qubit properties."
        )

    qp = props[
        int(
            q
        )
    ]

    if qp is None:
        return {
            "t1_us": None,
            "t2_us": None,
        }

    return {
        "t1_us": seconds_to_us(
            getattr(
                qp,
                "t1",
                None,
            )
        ),
        "t2_us": seconds_to_us(
            getattr(
                qp,
                "t2",
                None,
            )
        ),
    }


def instruction_error(
    backend,
    name,
    qubits,
):
    """
    Return current instruction error if available.

    For CZ we try both directions because current backend metadata can be
    directional even though native adjacency is treated undirected for
    embedding.
    """
    target = backend.target

    try:
        inst_map = target[
            name
        ]
    except Exception:
        return None

    keys = [
        tuple(
            int(x)
            for x in qubits
        )
    ]

    if len(
        qubits
    ) == 2:
        keys.append(
            (
                int(
                    qubits[
                        1
                    ]
                ),
                int(
                    qubits[
                        0
                    ]
                ),
            )
        )

    values = []

    for key in keys:
        try:
            props = inst_map[
                key
            ]

            if (
                props is not None
                and getattr(
                    props,
                    "error",
                    None,
                )
                is not None
            ):
                err = float(
                    props.error
                )

                if np.isfinite(
                    err
                ):
                    values.append(
                        err
                    )
        except Exception:
            pass

    if not values:
        return None

    # For an undirected physical edge, use the better available directed
    # implementation only for the calibration descriptor. Actual compilation
    # remains the final authority.
    return float(
        min(
            values
        )
    )


def readout_error(
    backend,
    q,
):
    err = instruction_error(
        backend,
        "measure",
        [
            int(
                q
            )
        ],
    )

    return err


def current_physical_graph(
    backend,
):
    """
    Build current undirected adjacency from BackendV2 coupling map.

    A qubit is eligible when:
      - it participates in the current coupling graph,
      - T1 and T2 are both finite/positive.
    """
    coupling_map = backend.coupling_map

    if coupling_map is None:
        raise RuntimeError(
            "Backend has no coupling map."
        )

    directed = coupling_map.get_edges()

    adjacency = defaultdict(
        set
    )

    for a, b in directed:
        a = int(
            a
        )
        b = int(
            b
        )

        adjacency[
            a
        ].add(
            b
        )
        adjacency[
            b
        ].add(
            a
        )

    eligible = []

    qubit_cal = {}

    for q in range(
        int(
            backend.num_qubits
        )
    ):
        cal = get_qubit_t1_t2_us(
            backend,
            q,
        )

        t1 = cal[
            "t1_us"
        ]

        t2 = cal[
            "t2_us"
        ]

        qubit_cal[
            q
        ] = cal

        if (
            q in adjacency
            and t1 is not None
            and t2 is not None
            and t1 > 0
            and t2 > 0
        ):
            eligible.append(
                q
            )

    return (
        adjacency,
        sorted(
            eligible
        ),
        qubit_cal,
    )


# =============================================================================
# Native logical-to-physical H2 embeddings
# =============================================================================

def enumerate_native_embeddings(
    logical_edges,
    adjacency,
    eligible,
):
    logical_nodes = sorted(
        {
            int(q)
            for edge in logical_edges
            for q in edge
        }
    )

    logical_adj = {
        q: set()
        for q in logical_nodes
    }

    for a, b in logical_edges:
        logical_adj[
            int(
                a
            )
        ].add(
            int(
                b
            )
        )
        logical_adj[
            int(
                b
            )
        ].add(
            int(
                a
            )
        )

    # Dynamic backtracking order:
    # choose the unassigned logical node with
    #   1) most already-assigned neighbors,
    #   2) highest logical degree.
    mappings = []

    eligible_set = set(
        int(q)
        for q in eligible
    )

    def recurse(
        mapping,
        used,
    ):
        if len(
            mapping
        ) == len(
            logical_nodes
        ):
            layout = [
                int(
                    mapping[
                        q
                    ]
                )
                for q in sorted(
                    logical_nodes
                )
            ]

            mappings.append(
                layout
            )
            return

        unassigned = [
            q
            for q in logical_nodes
            if q not in mapping
        ]

        def logical_priority(q):
            assigned_neighbors = sum(
                1
                for n in logical_adj[
                    q
                ]
                if n in mapping
            )

            return (
                assigned_neighbors,
                len(
                    logical_adj[
                        q
                    ]
                ),
            )

        logical_q = max(
            unassigned,
            key=logical_priority,
        )

        assigned_neighbors = [
            n
            for n in logical_adj[
                logical_q
            ]
            if n in mapping
        ]

        if assigned_neighbors:
            candidate_phys = None

            for n in assigned_neighbors:
                neigh = adjacency[
                    mapping[
                        n
                    ]
                ]

                candidate_phys = (
                    set(
                        neigh
                    )
                    if candidate_phys is None
                    else candidate_phys.intersection(
                        neigh
                    )
                )

            candidate_phys = (
                candidate_phys
                .intersection(
                    eligible_set
                )
            )
        else:
            candidate_phys = set(
                eligible_set
            )

        candidate_phys = sorted(
            candidate_phys
            - used
        )

        for physical_q in candidate_phys:
            # Ensure adjacency to every already-mapped logical neighbor.
            valid = True

            for n in assigned_neighbors:
                if physical_q not in adjacency[
                    mapping[
                        n
                    ]
                ]:
                    valid = False
                    break

            if not valid:
                continue

            mapping[
                logical_q
            ] = int(
                physical_q
            )

            used.add(
                int(
                    physical_q
                )
            )

            recurse(
                mapping,
                used,
            )

            used.remove(
                int(
                    physical_q
                )
            )

            del mapping[
                logical_q
            ]

    recurse(
        {},
        set(),
    )

    return mappings


# =============================================================================
# Embedding metrics + memory-aware selection
# =============================================================================

def embedding_metrics(
    backend,
    layout,
    logical_edges,
    qubit_cal,
    t_setting_est_us,
):
    layout = [
        int(
            q
        )
        for q in layout
    ]

    if len(
        layout
    ) != 6:
        raise RuntimeError(
            f"Expected six physical qubits, got {layout}"
        )

    # Logical q4,q5 are persistent memory.
    mem_phys = [
        layout[
            4
        ],
        layout[
            5
        ],
    ]

    mem_coherences = []

    for p in mem_phys:
        mem_coherences.extend(
            [
                qubit_cal[
                    p
                ][
                    "t1_us"
                ],
                qubit_cal[
                    p
                ][
                    "t2_us"
                ],
            ]
        )

    min_mem_coherence_us = float(
        min(
            mem_coherences
        )
    )

    memory_margin_est = (
        min_mem_coherence_us
        / float(
            t_setting_est_us
        )
    )

    memory_pass_est = bool(
        memory_margin_est
        > 1.0
    )

    cz_errors = []

    for logical_a, logical_b in logical_edges:
        p_a = layout[
            int(
                logical_a
            )
        ]

        p_b = layout[
            int(
                logical_b
            )
        ]

        err = instruction_error(
            backend,
            "cz",
            [
                p_a,
                p_b,
            ],
        )

        if err is not None:
            cz_errors.append(
                err
            )

    injection_readout = []

    for logical_q in range(
        4
    ):
        p = layout[
            logical_q
        ]

        err = readout_error(
            backend,
            p,
        )

        if err is not None:
            injection_readout.append(
                err
            )

    worst_cz_error = (
        np.nan
        if not cz_errors
        else float(
            max(
                cz_errors
            )
        )
    )

    worst_injection_ro = (
        np.nan
        if not injection_readout
        else float(
            max(
                injection_readout
            )
        )
    )

    return {
        "layout": (
            layout
        ),
        "layout_json": json.dumps(
            layout
        ),
        "memory_q4_physical": int(
            layout[
                4
            ]
        ),
        "memory_q5_physical": int(
            layout[
                5
            ]
        ),
        "memory_pair": (
            f"{layout[4]}-{layout[5]}"
        ),
        "memory_pair_set": (
            "-".join(
                str(x)
                for x in sorted(
                    [
                        layout[
                            4
                        ],
                        layout[
                            5
                        ],
                    ]
                )
            )
        ),
        "memory_min_coherence_us": (
            min_mem_coherence_us
        ),
        "memory_margin_est": float(
            memory_margin_est
        ),
        "memory_pass_est": (
            memory_pass_est
        ),
        "worst_required_cz_error": (
            worst_cz_error
        ),
        "worst_injection_readout_error": (
            worst_injection_ro
        ),
    }


def select_distinct_layouts(
    metrics_df,
    n_layouts,
):
    """
    Lexicographic, not weighted:
      1. memory_pass_est True first
      2. memory_margin_est descending
      3. worst CZ error ascending
      4. worst injection readout error ascending

    First pass requires distinct memory-pair sets.
    Second pass, only if needed, permits repeated memory pairs but still
    requires distinct full layouts.
    """
    ranked = metrics_df.sort_values(
        by=[
            "memory_pass_est",
            "memory_margin_est",
            "worst_required_cz_error",
            "worst_injection_readout_error",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    selected_indices = []
    used_memory_pairs = set()
    used_layouts = set()

    # First pass: distinct memory pairs.
    for idx, row in ranked.iterrows():
        layout_key = str(
            row[
                "layout_json"
            ]
        )

        memory_pair = str(
            row[
                "memory_pair_set"
            ]
        )

        if layout_key in used_layouts:
            continue

        if memory_pair in used_memory_pairs:
            continue

        selected_indices.append(
            idx
        )

        used_layouts.add(
            layout_key
        )

        used_memory_pairs.add(
            memory_pair
        )

        if len(
            selected_indices
        ) >= int(
            n_layouts
        ):
            break

    # Second pass: relax memory-pair distinctness if necessary.
    if len(
        selected_indices
    ) < int(
        n_layouts
    ):
        for idx, row in ranked.iterrows():
            layout_key = str(
                row[
                    "layout_json"
                ]
            )

            if layout_key in used_layouts:
                continue

            selected_indices.append(
                idx
            )

            used_layouts.add(
                layout_key
            )

            if len(
                selected_indices
            ) >= int(
                n_layouts
            ):
                break

    if len(
        selected_indices
    ) < int(
        n_layouts
    ):
        raise RuntimeError(
            f"Only {len(selected_indices)} distinct layouts available; "
            f"requested {n_layouts}."
        )

    selected = ranked.iloc[
        selected_indices
    ].copy()

    selected.insert(
        0,
        "test_layout_id",
        [
            f"L{i+1}"
            for i in range(
                len(
                    selected
                )
            )
        ],
    )

    return (
        ranked,
        selected,
    )


# =============================================================================
# Actual K=64 circuit construction
# =============================================================================

def append_rwp_step(
    qc,
    candidate,
    angle_row,
    first_step,
):
    if not first_step:
        for q in range(
            4
        ):
            qc.reset(
                q
            )

    for q in range(
        4
    ):
        qc.ry(
            float(
                angle_row[
                    q
                ]
            ),
            q,
        )

    r = int(
        candidate[
            "r"
        ]
    )

    dt = float(
        candidate[
            "dt"
        ]
    )

    hx = float(
        candidate[
            "hx"
        ]
    )

    hy = float(
        candidate[
            "hy"
        ]
    )

    J = candidate[
        "J"
    ]

    for _ in range(
        r
    ):
        for i, j in common.TOPOLOGY_EDGES[
            candidate[
                "topology"
            ]
        ]:
            qc.rzz(
                2.0
                * float(
                    J[
                        (i, j)
                    ]
                )
                * dt
                / r,
                i,
                j,
            )

        theta_x = (
            2.0
            * hx
            * dt
            / r
        )

        for q in range(
            6
        ):
            qc.rx(
                theta_x,
                q,
            )

        if not np.isclose(
            hy,
            0.0,
        ):
            theta_y = (
                2.0
                * hy
                * dt
                / r
            )

            qc.ry(
                theta_y,
                4,
            )

            qc.ry(
                theta_y,
                5,
            )


def measurement_settings():
    return [
        (
            "XXXX",
            {
                0: "X",
                1: "X",
                2: "X",
                3: "X",
            },
        ),
        (
            "ZZZZ",
            {
                0: "Z",
                1: "Z",
                2: "Z",
                3: "Z",
            },
        ),
    ]


def build_rwp64_circuit(
    candidate,
    angles,
    endpoint,
    setting,
):
    label, basis = (
        setting
    )

    qreg = QuantumRegister(
        6,
        "q",
    )

    creg = ClassicalRegister(
        4,
        "m",
    )

    qc = QuantumCircuit(
        qreg,
        creg,
        name=(
            f"H2_RWP64_e{int(endpoint)}_{label}"
        ),
    )

    start = (
        int(
            endpoint
        )
        - EXPECTED_K
        + 1
    )

    if start < 0:
        raise ValueError(
            f"endpoint={endpoint}, K={EXPECTED_K} gives start={start}"
        )

    for k, idx in enumerate(
        range(
            start,
            int(
                endpoint
            )
            + 1,
        )
    ):
        append_rwp_step(
            qc,
            candidate,
            angles[
                idx
            ],
            first_step=(
                k == 0
            ),
        )

    for q, axis in basis.items():
        if axis == "X":
            qc.h(
                q
            )
        elif axis == "Z":
            pass
        else:
            raise ValueError(
                axis
            )

    for c, q in enumerate(
        range(
            4
        )
    ):
        qc.measure(
            q,
            c,
        )

    return qc


def compile_circuit(
    logical,
    backend,
    layout,
):
    compiled = transpile(
        logical,
        backend=backend,
        initial_layout=[
            int(
                q
            )
            for q in layout
        ],
        routing_method="none",
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
        scheduling_method="alap",
    )

    backend.check_faulty(
        compiled
    )

    return compiled


def compiled_resource_row(
    compiled,
    backend,
    layout_id,
    layout,
    endpoint,
    setting,
):
    ops = {
        str(
            key
        ): int(
            value
        )
        for key, value
        in compiled.count_ops().items()
    }

    duration_s = float(
        compiled.estimate_duration(
            backend.target,
            unit="s",
        )
    )

    return {
        "test_layout_id": (
            layout_id
        ),
        "layout_json": json.dumps(
            [
                int(
                    q
                )
                for q in layout
            ]
        ),
        "endpoint": int(
            endpoint
        ),
        "setting": (
            setting
        ),
        "depth": int(
            compiled.depth()
        ),
        "size": int(
            compiled.size()
        ),
        "n_cz": int(
            ops.get(
                "cz",
                0,
            )
        ),
        "n_swap": int(
            ops.get(
                "swap",
                0,
            )
        ),
        "n_reset": int(
            ops.get(
                "reset",
                0,
            )
        ),
        "n_measure": int(
            ops.get(
                "measure",
                0,
            )
        ),
        "duration_us": float(
            duration_s
            * 1e6
        ),
        "operations": json.dumps(
            ops,
            sort_keys=True,
        ),
    }


# =============================================================================
# Actual per-layout/per-qubit coherence tables
# =============================================================================

def build_actual_layout_summaries(
    selected_df,
    compiled_df,
    qubit_cal,
):
    qubit_rows = []
    layout_rows = []

    for _, selected in selected_df.iterrows():
        layout_id = str(
            selected[
                "test_layout_id"
            ]
        )

        layout = json.loads(
            selected[
                "layout_json"
            ]
        )

        sub = compiled_df[
            compiled_df[
                "test_layout_id"
            ]
            == layout_id
        ].copy()

        if len(
            sub
        ) == 0:
            raise RuntimeError(
                f"No compiled rows for {layout_id}"
            )

        endpoint_feature = (
            sub
            .groupby(
                "endpoint",
                as_index=False,
            )
            .agg(
                feature_cz=(
                    "n_cz",
                    "sum",
                ),
                feature_resets=(
                    "n_reset",
                    "sum",
                ),
                feature_swaps=(
                    "n_swap",
                    "sum",
                ),
                max_setting_depth=(
                    "depth",
                    "max",
                ),
                feature_duration_us=(
                    "duration_us",
                    "sum",
                ),
                max_setting_duration_us=(
                    "duration_us",
                    "max",
                ),
            )
        )

        actual_t_setting_us = float(
            endpoint_feature[
                "max_setting_duration_us"
            ].median()
        )

        # Report all six qubits.
        local_qubit_rows = []

        for logical_q, physical_q in enumerate(
            layout
        ):
            role = (
                "injection"
                if logical_q
                <= 3
                else "memory"
            )

            t1_us = float(
                qubit_cal[
                    int(
                        physical_q
                    )
                ][
                    "t1_us"
                ]
            )

            t2_us = float(
                qubit_cal[
                    int(
                        physical_q
                    )
                ][
                    "t2_us"
                ]
            )

            ratio_t1 = (
                actual_t_setting_us
                / t1_us
            )

            ratio_t2 = (
                actual_t_setting_us
                / t2_us
            )

            row = {
                "test_layout_id": (
                    layout_id
                ),
                "layout_json": json.dumps(
                    layout
                ),
                "logical_qubit": int(
                    logical_q
                ),
                "physical_qubit": int(
                    physical_q
                ),
                "role": (
                    role
                ),
                "actual_max_setting_duration_us": (
                    actual_t_setting_us
                ),
                "t1_us": (
                    t1_us
                ),
                "t2_us": (
                    t2_us
                ),
                "duration_over_t1": float(
                    ratio_t1
                ),
                "duration_over_t2": float(
                    ratio_t2
                ),
                "whole_setting_t1_pass": bool(
                    ratio_t1
                    < 1.0
                ),
                "whole_setting_t2_pass": bool(
                    ratio_t2
                    < 1.0
                ),
            }

            qubit_rows.append(
                row
            )

            local_qubit_rows.append(
                row
            )

        local_qubit_df = pd.DataFrame(
            local_qubit_rows
        )

        memory_df = local_qubit_df[
            local_qubit_df[
                "role"
            ]
            == "memory"
        ]

        memory_pass_actual = bool(
            (
                memory_df[
                    "duration_over_t1"
                ]
                < 1.0
            ).all()
            and
            (
                memory_df[
                    "duration_over_t2"
                ]
                < 1.0
            ).all()
        )

        min_memory_margin_actual = float(
            min(
                (
                    memory_df[
                        "t1_us"
                    ]
                    / actual_t_setting_us
                ).min(),
                (
                    memory_df[
                        "t2_us"
                    ]
                    / actual_t_setting_us
                ).min(),
            )
        )

        layout_rows.append(
            {
                "test_layout_id": (
                    layout_id
                ),
                "layout_json": json.dumps(
                    layout
                ),
                "memory_q4_physical": int(
                    layout[
                        4
                    ]
                ),
                "memory_q5_physical": int(
                    layout[
                        5
                    ]
                ),
                "estimated_memory_pass": bool(
                    selected[
                        "memory_pass_est"
                    ]
                ),
                "estimated_memory_margin": float(
                    selected[
                        "memory_margin_est"
                    ]
                ),
                "worst_required_cz_error": safe_float(
                    selected[
                        "worst_required_cz_error"
                    ]
                ),
                "worst_injection_readout_error": safe_float(
                    selected[
                        "worst_injection_readout_error"
                    ]
                ),
                "actual_median_cz_per_feature": float(
                    endpoint_feature[
                        "feature_cz"
                    ].median()
                ),
                "actual_median_resets_per_feature": float(
                    endpoint_feature[
                        "feature_resets"
                    ].median()
                ),
                "actual_max_swaps": int(
                    endpoint_feature[
                        "feature_swaps"
                    ].max()
                ),
                "actual_median_max_setting_depth": float(
                    endpoint_feature[
                        "max_setting_depth"
                    ].median()
                ),
                "actual_median_feature_duration_us": float(
                    endpoint_feature[
                        "feature_duration_us"
                    ].median()
                ),
                "actual_median_max_setting_duration_us": (
                    actual_t_setting_us
                ),
                "actual_memory_pass": (
                    memory_pass_actual
                ),
                "actual_min_memory_coherence_margin": (
                    min_memory_margin_actual
                ),
                "actual_worst_memory_duration_over_t1": float(
                    memory_df[
                        "duration_over_t1"
                    ].max()
                ),
                "actual_worst_memory_duration_over_t2": float(
                    memory_df[
                        "duration_over_t2"
                    ].max()
                ),
            }
        )

    return (
        pd.DataFrame(
            qubit_rows
        ),
        pd.DataFrame(
            layout_rows
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Candidate #5 memory-aware native H2 layout reselection."
        )
    )

    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
    )

    parser.add_argument(
        "--n-layouts",
        type=int,
        default=DEFAULT_N_LAYOUTS,
        help=(
            "Number of genuinely different layouts to compile/test. "
            "Default = 3."
        ),
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=(
            "Stored only for later workload projection; no QPU run occurs."
        ),
    )

    args = parser.parse_args()

    if args.n_layouts < 3:
        raise ValueError(
            "--n-layouts must be at least 3 for this experiment."
        )

    print(
        "=" * 126
    )
    print(
        "WEEK 11.1E.6 — CANDIDATE #5 MEMORY-AWARE LAYOUT RESELECTION"
    )
    print(
        "=" * 126
    )
    print(
        "NO QPU JOB WILL BE SUBMITTED."
    )
    print(
        "2026 = FROZEN / NOT LOADED"
    )
    print()

    package = load_json(
        FROZEN_PACKAGE
    )

    prior = load_json(
        PREVIOUS_PREFLIGHT
    )

    candidate, manifest_meta = (
        candmod.load_candidate()
    )

    if candidate[
        "candidate_id"
    ] != EXPECTED_CANDIDATE:
        raise RuntimeError(
            "Unexpected Candidate #5 identity."
        )

    if int(
        candidate[
            "r"
        ]
    ) != EXPECTED_R:
        raise RuntimeError(
            "Unexpected Candidate #5 operating r."
        )

    if int(
        package[
            "operational_K"
        ]
    ) != EXPECTED_K:
        raise RuntimeError(
            "Unexpected frozen K."
        )

    t_setting_est_us = float(
        prior[
            "actual_compiled_sample"
        ][
            "median_max_setting_duration_us"
        ]
    )

    print(
        "Frozen Candidate #5:"
    )
    print(
        f"  candidate={EXPECTED_CANDIDATE}"
    )
    print(
        f"  topology={EXPECTED_TOPOLOGY}"
    )
    print(
        f"  operating r={EXPECTED_R}"
    )
    print(
        f"  K_RWP={EXPECTED_K}"
    )
    print(
        f"  readout={EXPECTED_READOUT}"
    )
    print(
        f"  previous actual t_setting estimate="
        f"{t_setting_est_us:.3f} us"
    )
    print()

    # ------------------------------------------------------------------
    # Fresh backend snapshot.
    # ------------------------------------------------------------------
    service = get_service()

    backend = service.backend(
        args.backend
    )

    (
        adjacency,
        eligible,
        qubit_cal,
    ) = current_physical_graph(
        backend
    )

    logical_edges = [
        (
            int(
                a
            ),
            int(
                b
            ),
        )
        for a, b
        in common.TOPOLOGY_EDGES[
            EXPECTED_TOPOLOGY
        ]
    ]

    print(
        "Fresh hardware snapshot:"
    )
    print(
        f"  backend={args.backend}"
    )
    print(
        f"  qubits={backend.num_qubits}"
    )
    print(
        f"  eligible qubits={len(eligible)}"
    )
    print(
        f"  logical H2 edges={logical_edges}"
    )
    print()

    # ------------------------------------------------------------------
    # Enumerate all native H2 embeddings.
    # ------------------------------------------------------------------
    layouts = enumerate_native_embeddings(
        logical_edges,
        adjacency,
        eligible,
    )

    print(
        f"Native H2 embeddings found: {len(layouts)}"
    )

    embedding_rows = []

    for layout in layouts:
        embedding_rows.append(
            embedding_metrics(
                backend,
                layout,
                logical_edges,
                qubit_cal,
                t_setting_est_us,
            )
        )

    metrics_df = pd.DataFrame(
        embedding_rows
    )

    ranked_df, selected_df = (
        select_distinct_layouts(
            metrics_df,
            args.n_layouts,
        )
    )

    ranked_df.to_csv(
        OUT_ALL_EMBEDDINGS,
        index=False,
    )

    selected_df.to_csv(
        OUT_SELECTED_LAYOUTS,
        index=False,
    )

    n_est_feasible = int(
        ranked_df[
            "memory_pass_est"
        ].sum()
    )

    print()
    print(
        "MEMORY-AWARE PRESELECTION"
    )
    print(
        "-" * 126
    )
    print(
        f"Embeddings with BOTH memory q4/q5 satisfying "
        f"T1,T2 > {t_setting_est_us:.3f} us: "
        f"{n_est_feasible}/{len(ranked_df)}"
    )
    print()

    print(
        "Layouts selected for ACTUAL K=64 compilation:"
    )

    show_pre = [
        "test_layout_id",
        "layout_json",
        "memory_q4_physical",
        "memory_q5_physical",
        "memory_pass_est",
        "memory_margin_est",
        "worst_required_cz_error",
        "worst_injection_readout_error",
    ]

    print(
        selected_df[
            show_pre
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Prepare train+validation angles only.
    # ------------------------------------------------------------------
    (
        work_tv,
        cols,
        _,
    ) = common.load_train_validation()

    angles = common.make_angles(
        work_tv,
        cols,
        float(
            candidate[
                "alpha"
            ]
        ),
    )

    val_endpoints = np.arange(
        common.N_TRAIN,
        common.N_TRAIN
        + common.N_VAL,
        dtype=int,
    )

    test_endpoints = np.asarray(
        [
            int(
                val_endpoints[
                    0
                ]
            ),
            int(
                val_endpoints[
                    len(
                        val_endpoints
                    )
                    // 2
                ]
            ),
            int(
                val_endpoints[
                    -1
                ]
            ),
        ],
        dtype=int,
    )

    settings = measurement_settings()

    # ------------------------------------------------------------------
    # Compile ACTUAL K=64 circuits for each selected layout.
    # ------------------------------------------------------------------
    compiled_rows = []

    print()
    print(
        "=" * 126
    )
    print(
        "ACTUAL K=64 COMPILATION FOR EACH TEST LAYOUT"
    )
    print(
        "=" * 126
    )

    for _, selected in selected_df.iterrows():
        layout_id = str(
            selected[
                "test_layout_id"
            ]
        )

        layout = json.loads(
            selected[
                "layout_json"
            ]
        )

        print()
        print(
            f"{layout_id}: layout={layout}"
        )

        for endpoint in test_endpoints:
            for setting in settings:
                setting_name = setting[
                    0
                ]

                print(
                    f"  compiling endpoint={int(endpoint)}, "
                    f"setting={setting_name} ..."
                )

                logical = build_rwp64_circuit(
                    candidate,
                    angles,
                    int(
                        endpoint
                    ),
                    setting,
                )

                compiled = compile_circuit(
                    logical,
                    backend,
                    layout,
                )

                compiled_rows.append(
                    compiled_resource_row(
                        compiled,
                        backend,
                        layout_id,
                        layout,
                        int(
                            endpoint
                        ),
                        setting_name,
                    )
                )

    compiled_df = pd.DataFrame(
        compiled_rows
    )

    compiled_df.to_csv(
        OUT_COMPILED_SETTINGS,
        index=False,
    )

    qubit_df, layout_summary_df = (
        build_actual_layout_summaries(
            selected_df,
            compiled_df,
            qubit_cal,
        )
    )

    qubit_df.to_csv(
        OUT_QUBIT_COHERENCE,
        index=False,
    )

    layout_summary_df.to_csv(
        OUT_LAYOUT_SUMMARY,
        index=False,
    )

    # ------------------------------------------------------------------
    # Show all-qubit T1/T2 ratios for each tested layout.
    # ------------------------------------------------------------------
    print()
    print(
        "=" * 126
    )
    print(
        "ALL-QUBIT T1/T2 CHECK USING EACH LAYOUT'S ACTUAL t_setting"
    )
    print(
        "=" * 126
    )

    for layout_id in layout_summary_df[
        "test_layout_id"
    ]:
        summary_row = layout_summary_df[
            layout_summary_df[
                "test_layout_id"
            ]
            == layout_id
        ].iloc[
            0
        ]

        print()
        print(
            f"{layout_id}  layout={summary_row['layout_json']}"
        )
        print(
            f"actual t_setting="
            f"{summary_row['actual_median_max_setting_duration_us']:.3f} us"
        )

        sub = qubit_df[
            qubit_df[
                "test_layout_id"
            ]
            == layout_id
        ]

        show_cols = [
            "logical_qubit",
            "physical_qubit",
            "role",
            "t1_us",
            "t2_us",
            "duration_over_t1",
            "duration_over_t2",
            "whole_setting_t1_pass",
            "whole_setting_t2_pass",
        ]

        print(
            sub[
                show_cols
            ].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # Final per-layout comparison and decision.
    # ------------------------------------------------------------------
    print()
    print(
        "=" * 126
    )
    print(
        "LAYOUT COMPARISON"
    )
    print(
        "=" * 126
    )

    show_layout = [
        "test_layout_id",
        "layout_json",
        "memory_q4_physical",
        "memory_q5_physical",
        "actual_median_cz_per_feature",
        "actual_median_max_setting_depth",
        "actual_median_max_setting_duration_us",
        "actual_worst_memory_duration_over_t1",
        "actual_worst_memory_duration_over_t2",
        "actual_min_memory_coherence_margin",
        "actual_memory_pass",
        "actual_max_swaps",
    ]

    print(
        layout_summary_df[
            show_layout
        ].to_string(
            index=False
        )
    )

    actual_feasible = layout_summary_df[
        layout_summary_df[
            "actual_memory_pass"
        ]
        == True
    ].copy()

    recommendation = None

    if len(
        actual_feasible
    ) > 0:
        actual_feasible = actual_feasible.sort_values(
            by=[
                "actual_min_memory_coherence_margin",
                "worst_required_cz_error",
                "worst_injection_readout_error",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            na_position="last",
        )

        recommendation = (
            actual_feasible.iloc[
                0
            ].to_dict()
        )

        print()
        print(
            "FINAL DECISION"
        )
        print(
            "-" * 126
        )
        print(
            f"{len(actual_feasible)} of {len(layout_summary_df)} "
            f"compiled test layouts PASS the necessary memory "
            f"T1/T2 < 1 condition."
        )
        print(
            f"Best memory-aware layout for a SMALL QPU PILOT: "
            f"{recommendation['test_layout_id']} "
            f"{recommendation['layout_json']}"
        )
        print(
            "Do NOT jump directly to the full 365-day QPU run. "
            "A small RWP64 pilot should be evaluated first."
        )
    else:
        print()
        print(
            "FINAL DECISION"
        )
        print(
            "-" * 126
        )
        print(
            "NONE of the compiled memory-aware layouts passes the "
            "necessary persistent-memory T1/T2 < 1 condition."
        )
        print(
            "Candidate #5 should therefore be closed as a full-QPU "
            "competitive candidate and retained as an ideal/simulation "
            "reference, with at most a tiny coherence-demonstration pilot."
        )

    # ------------------------------------------------------------------
    # JSON + DB
    # ------------------------------------------------------------------
    summary = {
        "candidate_key": (
            EXPECTED_CANDIDATE
        ),
        "operating_r": (
            EXPECTED_R
        ),
        "topology": (
            EXPECTED_TOPOLOGY
        ),
        "K_RWP": (
            EXPECTED_K
        ),
        "backend": (
            args.backend
        ),
        "previous_actual_t_setting_us_used_for_preselection": (
            t_setting_est_us
        ),
        "native_embeddings_found": int(
            len(
                layouts
            )
        ),
        "estimated_memory_feasible_embeddings": (
            n_est_feasible
        ),
        "n_tested_layouts": int(
            len(
                selected_df
            )
        ),
        "tested_layouts": (
            db.json_safe(
                layout_summary_df.to_dict(
                    orient="records"
                )
            )
        ),
        "all_qubit_coherence": (
            db.json_safe(
                qubit_df.to_dict(
                    orient="records"
                )
            )
        ),
        "n_actual_memory_feasible_tested_layouts": int(
            len(
                actual_feasible
            )
        ),
        "recommended_small_pilot_layout": (
            db.json_safe(
                recommendation
            )
            if recommendation is not None
            else None
        ),
        "selection_rule": (
            "Lexicographic/no weighted score: memory feasibility first; "
            "then larger minimum memory coherence margin; then lower worst "
            "required CZ error; then lower worst injection readout error. "
            "Test layouts preferentially use distinct physical memory pairs."
        ),
        "memory_feasibility_rule": (
            "For BOTH logical memory qubits q4,q5, actual compiled "
            "max-setting duration / T1 < 1 and duration / T2 < 1."
        ),
        "qpu_submitted": False,
        "2026_loaded": False,
    }

    OUT_JSON.write_text(
        json.dumps(
            db.json_safe(
                summary
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    db.create_all_tables()

    run_uuid = db.create_run(
        script_name=(
            Path(
                __file__
            ).name
        ),
        run_status=(
            "MEMORY_AWARE_LAYOUT_RESELECTION_COMPLETE"
        ),
        started_at_utc=(
            db.utc_now_naive()
        ),
        completed_at_utc=(
            db.utc_now_naive()
        ),
        forecast_dataset_name=(
            "property_damage_next_day_v1"
        ),
        feature_set_name=(
            "F4"
        ),
        evaluation_split=(
            "validation_2025"
        ),
        candidate_key=(
            "CONT_H2_R2_RWP64_MEMORY_AWARE_LAYOUTS"
        ),
        selected_by_rules=(
            "Rule2"
        ),
        protocol=(
            "RWP_from_CONT"
        ),
        topology=(
            EXPECTED_TOPOLOGY
        ),
        window_size=(
            EXPECTED_K
        ),
        trotter_r=(
            EXPECTED_R
        ),
        readout_name=(
            EXPECTED_READOUT
        ),
        primitive_name=(
            "calibration_compile_only"
        ),
        measurement_method=(
            "memory_aware_layout_reselection"
        ),
        backend_name=(
            args.backend
        ),
        shots_per_setting=int(
            args.shots
        ),
        n_validation_endpoints=(
            365
        ),
        n_circuits=(
            730
        ),
        full_2025=True,
        pilot_selection=(
            "memory_aware_preflight_only"
        ),
        notes=(
            "Candidate #5 memory-aware H2 layout reselection. "
            "Enumerates native embeddings, tests T1/T2 for all six qubits, "
            "requires both persistent memory qubits q4/q5 to satisfy actual "
            "K=64 t_setting/T1<1 and t_setting/T2<1. Compiles first/middle/"
            "last validation endpoints x both X/Z settings for at least "
            "three distinct layouts. No QPU job."
        ),
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_ALL_EMBEDDINGS.name,
        ranked_df,
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_SELECTED_LAYOUTS.name,
        selected_df,
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_COMPILED_SETTINGS.name,
        compiled_df,
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_QUBIT_COHERENCE.name,
        qubit_df,
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_LAYOUT_SUMMARY.name,
        layout_summary_df,
    )

    db.save_json_artifact(
        run_uuid,
        OUT_JSON.name,
        summary,
    )

    print()
    print(
        f"[database] Memory-aware layout run UUID: {run_uuid}"
    )

    print()
    print(
        "Saved:"
    )
    print(
        f"  {OUT_ALL_EMBEDDINGS}"
    )
    print(
        f"  {OUT_SELECTED_LAYOUTS}"
    )
    print(
        f"  {OUT_COMPILED_SETTINGS}"
    )
    print(
        f"  {OUT_QUBIT_COHERENCE}"
    )
    print(
        f"  {OUT_LAYOUT_SUMMARY}"
    )
    print(
        f"  {OUT_JSON}"
    )

    print()
    print(
        "STOP HERE. NO QPU SUBMISSION."
    )
    print(
        "Send me the MEMORY-AWARE PRESELECTION, ALL-QUBIT T1/T2 CHECK, "
        "LAYOUT COMPARISON, and FINAL DECISION sections."
    )


if __name__ == "__main__":
    main()
