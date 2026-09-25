"""
12_02B_iqm_strict_compile_audit_v2.py
----------------------------------
Week 12.2B v2: strict REAL-PHYSICAL-TARGET / NO-ROUTING compile audit for IQM Emerald + Garnet.

This script:
- reads the Week-12.2A Pareto embeddings,
- selects a deterministic NON-WEIGHTED anchor set per backend/topology,
- freezes the same calibration-set IDs used in Week 12.1,
- compiles one generic QRC Trotter slice on every anchor,
- compiles against backend.get_real_target() (IQM physical target without fictional CZ gates),
- uses routing_method="none" so Qiskit MUST fail if routing is required,
- resolves transpiled two-qubit loci back to actual IQM physical qubit names,
- verifies that all CZ gates stay on the expected physical H-edges,
- verifies zero SWAPs and identity routing permutation,
- records depth / native operation counts.

No QPU job is submitted.
No shots are used.
No 2026 test data are touched.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

from qiskit import QuantumCircuit, transpile

from iqm.qiskit_iqm.iqm_provider import IQMProvider

# iqm_account.py remains the single source of authentication.
from iqm_account import get_authentication


# =============================================================================
# Paths / fixed experiment settings
# =============================================================================

RESULTS_DIR = Path("results")

DISCOVERY_JSON = RESULTS_DIR / "12_01_iqm_discovery.json"
REGION_JSON = RESULTS_DIR / "12_02_iqm_region_search.json"
PARETO_CSV = RESULTS_DIR / "12_02_iqm_pareto_embeddings.csv"

OUTPUT_CSV = RESULTS_DIR / "12_02B_iqm_strict_compile_audit_v2.csv"
OUTPUT_JSON = RESULTS_DIR / "12_02B_iqm_strict_compile_audit_v2.json"

BACKENDS = ("emerald", "garnet")

SEED_TRANSPILER = 79001
OPTIMIZATION_LEVEL = 0

# One GENERIC structural Trotter slice only.
#
# These angles are deliberately non-special and non-zero.
# Their purpose is NOT to test reservoir performance; only to ensure that the
# whole logical interaction pattern can be translated natively on the chosen
# physical layout.
BASE_ZZ_ANGLE = 0.371
X_ANGLE = 0.233
Y_MEMORY_ANGLE = 0.197


# =============================================================================
# Pareto anchor selection
# =============================================================================
#
# We deliberately DO NOT define a weighted hardware score.
#
# For each (backend, topology), retain the union of the layouts that separately
# maximize each of the six hardware objectives used in 12.2A.
#
# Hence at most six anchors per backend/topology, often fewer because one layout
# can be best in several objectives.
# =============================================================================

ANCHOR_OBJECTIVES = {
    "best_min_t1":
        "min_t1_us",

    "best_min_t2_echo":
        "min_t2_echo_us",

    "best_memory_min_t1":
        "memory_min_t1_us",

    "best_memory_min_t2_echo":
        "memory_min_t2_echo_us",

    "best_min_readout_fidelity":
        "min_readout_fidelity",

    "best_min_cz_fidelity":
        "min_cz_fidelity",
}


# =============================================================================
# Helpers
# =============================================================================

def normalize_edge(a: Any, b: Any):
    return tuple(sorted((str(a), str(b)), key=str))


def numeric(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x


def read_csv(path: Path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_anchor_rows(pareto_rows):
    grouped = defaultdict(list)

    for row in pareto_rows:
        grouped[
            (row["backend"], row["topology"])
        ].append(row)

    selected = {}

    for key, rows in grouped.items():

        by_layout = {}

        for reason, field in ANCHOR_OBJECTIVES.items():

            valid = [
                row
                for row in rows
                if numeric(row.get(field)) is not None
            ]

            if not valid:
                continue

            best = max(
                valid,
                key=lambda row: numeric(row[field]),
            )

            layout = best["layout"]

            if layout not in by_layout:
                by_layout[layout] = {
                    "row": best,
                    "reasons": [],
                }

            by_layout[layout]["reasons"].append(
                reason
            )

        selected[key] = list(
            by_layout.values()
        )

    return selected


def qiskit_physical_index_map(backend):
    """
    IQMBackend.physical_qubits gives physical component labels in the same order
    as the backend's Qiskit physical indices.
    """
    physical = list(backend.physical_qubits)

    return {
        name: index
        for index, name in enumerate(physical)
    }


def build_probe_circuit(logical_edges):
    """
    Build one generic QRC structural Trotter slice.

    Logical Hamiltonian structure:
        ZZ interactions on H-topology edges
        X field on all six qubits
        Y field on memory q4/q5

    We measure all six qubits only to make the final circuit execution-valid.
    """
    qc = QuantumCircuit(6, 6)

    # ZZ interaction layer.
    #
    # Use slightly different generic angles per edge so no accidental
    # identical-angle simplification can hide an interaction.
    for i, (a, b) in enumerate(logical_edges):
        theta = BASE_ZZ_ANGLE * (1.0 + 0.017 * i)
        qc.rzz(theta, int(a), int(b))

    # Global X field.
    for q in range(6):
        qc.rx(X_ANGLE, q)

    # Memory-only Y field.
    qc.ry(Y_MEMORY_ANGLE, 4)
    qc.ry(Y_MEMORY_ANGLE, 5)

    # Explicit measurements without an automatically inserted barrier.
    for q in range(6):
        qc.measure(q, q)

    return qc


def physical_two_qubit_edges_from_transpiled(
    tqc,
    backend,
    initial_layout,
):
    """
    Resolve every transpiled two-qubit operation to REAL IQM physical names.

    Qiskit/IQM can represent the compiled circuit in either of two ways:

    1. full-width physical circuit:
       output wire index == backend physical index;

    2. compact local circuit:
       output wires remain 0..5 and the requested physical placement is
       carried separately by layout/mapping metadata.

    Because routing_method="none" is used, a compact output wire i maps to
    initial_layout[i].  This avoids the v1 bug where local wires 0..5 were
    incorrectly labelled as QB1..QB6.
    """
    n_backend = len(backend.physical_qubits)
    n_local = len(initial_layout)

    if tqc.num_qubits == n_backend:
        def output_to_physical_index(i):
            return i

        mapping_mode = "full_width"

    elif tqc.num_qubits == n_local:
        def output_to_physical_index(i):
            if i < 0 or i >= n_local:
                raise RuntimeError(
                    f"Compact output index {i} outside local width {n_local}"
                )
            return int(initial_layout[i])

        mapping_mode = "compact_initial_layout"

    else:
        raise RuntimeError(
            "Unexpected transpiled circuit width: "
            f"{tqc.num_qubits}; backend width={n_backend}; "
            f"local width={n_local}"
        )

    loci = []

    for instruction in tqc.data:
        operation = instruction.operation
        qargs = instruction.qubits

        if len(qargs) != 2:
            continue

        out0 = tqc.find_bit(qargs[0]).index
        out1 = tqc.find_bit(qargs[1]).index

        p0 = output_to_physical_index(out0)
        p1 = output_to_physical_index(out1)

        name0 = backend.index_to_qubit_name(p0)
        name1 = backend.index_to_qubit_name(p1)

        loci.append(
            {
                "operation": operation.name,
                "pair": normalize_edge(name0, name1),
                "output_indices": [out0, out1],
                "physical_indices": [p0, p1],
            }
        )

    return loci, mapping_mode


def routing_permutation_info(tqc):
    """
    Return Qiskit's routing permutation and whether it is identity.

    We do NOT reject merely because TranspileLayout.final_layout is populated:
    Qiskit 2.x can use layout metadata for permutations/mapping.  What matters
    here is whether routing changed the qubit ordering.
    """
    layout = getattr(tqc, "layout", None)

    if layout is None:
        perm = list(range(tqc.num_qubits))
        return perm, True

    try:
        perm = list(layout.routing_permutation())
    except Exception:
        # routing_method="none" + zero SWAPs is still strong evidence, but for
        # a strict audit we mark an unreadable routing permutation as unknown.
        return None, None

    identity = perm == list(range(len(perm)))
    return perm, identity


def real_target_cz_edges(backend, real_target):
    """
    Extract the CZ loci encoded in IQM's REAL physical Qiskit target.
    """
    if "cz" not in set(real_target.operation_names):
        return set()

    try:
        qargs_map = real_target["cz"]
    except Exception:
        return set()

    edges = set()

    for qargs in qargs_map.keys():
        if qargs is None or len(qargs) != 2:
            continue

        a = backend.index_to_qubit_name(int(qargs[0]))
        b = backend.index_to_qubit_name(int(qargs[1]))
        edges.add(normalize_edge(a, b))

    return edges


def operation_counts(tqc):
    counts = Counter(
        instruction.operation.name
        for instruction in tqc.data
    )

    return dict(
        sorted(counts.items())
    )


def native_operation_names(target):
    try:
        return sorted(str(name) for name in target.operation_names)
    except Exception:
        return []


def layout_from_row(row):
    return [
        row["q0_C"],
        row["q1_D"],
        row["q2_P"],
        row["q3_H"],
        row["q4_M1"],
        row["q5_M2"],
    ]


# =============================================================================
# Main
# =============================================================================

def main():

    for path in (
        DISCOVERY_JSON,
        REGION_JSON,
        PARETO_CSV,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    with open(
        DISCOVERY_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        discovery = json.load(f)

    with open(
        REGION_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        region = json.load(f)

    pareto_rows = read_csv(
        PARETO_CSV
    )

    anchors = select_anchor_rows(
        pareto_rows
    )

    topologies = {
        name: [
            tuple(edge)
            for edge in info["edges"]
        ]
        for name, info
        in region["topologies"].items()
    }

    # Authentication comes only from iqm_account.py.
    token, server_url, _ = get_authentication()

    print("=" * 100)
    print("WEEK 12.2B v2 — IQM REAL-TARGET STRICT NO-ROUTING COMPILE AUDIT")
    print("=" * 100)

    print()
    print(f"Server:              {server_url}")
    print(f"Seed transpiler:     {SEED_TRANSPILER}")
    print(f"Optimization level:  {OPTIMIZATION_LEVEL}")
    print('Routing method:      "none"')
    print("Compile target:      backend.get_real_target()")
    print("QPU jobs submitted:  NO")
    print("Shots used:          0")

    output_rows = []

    output_json = {
        "settings": {
            "seed_transpiler":
                SEED_TRANSPILER,

            "optimization_level":
                OPTIMIZATION_LEVEL,

            "routing_method":
                "none",

            "probe": {
                "description":
                    "one generic structural QRC Trotter slice",

                "base_zz_angle":
                    BASE_ZZ_ANGLE,

                "x_angle":
                    X_ANGLE,

                "y_memory_angle":
                    Y_MEMORY_ANGLE,
            },

            "anchor_selection":
                ANCHOR_OBJECTIVES,

            "qpu_execution":
                False,
        },

        "backends": {},
    }

    # =========================================================================
    # Backend loop
    # =========================================================================

    for backend_name in BACKENDS:

        print()
        print("=" * 100)
        print(
            f"BACKEND: {backend_name.upper()}"
        )
        print("=" * 100)

        system = discovery[
            "systems"
        ][backend_name]

        calibration_set_id = (
            system[
                "summary"
            ][
                "calibration_set_id"
            ]
        )

        print(
            f"Frozen calibration set: "
            f"{calibration_set_id}"
        )

        # Build an IQM Qiskit backend tied to EXACTLY the same calibration
        # snapshot used for Week 12.1 / 12.2A.
        provider = IQMProvider(
            server_url,
            quantum_computer=backend_name,
            token=token,
        )

        backend = provider.get_backend(
            calibration_set_id=UUID(
                calibration_set_id
            ),
            use_metrics=False,
        )

        physical_to_index = (
            qiskit_physical_index_map(
                backend
            )
        )

        # CRITICAL v2 change:
        # IQMBackend.target can contain fictional CZ gates for Qiskit
        # compatibility.  For a hardware-feasibility audit we must compile
        # against the literal physical target.
        real_target = backend.get_real_target()

        target_ops = (
            native_operation_names(
                real_target
            )
        )

        real_cz_edges = real_target_cz_edges(
            backend,
            real_target,
        )

        # Compare with the calibrated CZ graph frozen in Week 12.1.
        dqa_cz = (
            system
            .get("dynamic_architecture", {})
            .get("gates", {})
            .get("cz", {})
            .get("implementations", {})
        )

        discovery_cz_edges = set()

        for impl_data in dqa_cz.values():
            for locus in impl_data.get("loci", []):
                if (
                    len(locus) == 2
                    and str(locus[0]).startswith("QB")
                    and str(locus[1]).startswith("QB")
                ):
                    discovery_cz_edges.add(
                        normalize_edge(locus[0], locus[1])
                    )

        target_vs_discovery_match = (
            real_cz_edges == discovery_cz_edges
        )

        print(
            f"Physical qubits:        "
            f"{len(physical_to_index)}"
        )

        print(
            f"REAL target operations: "
            f"{target_ops}"
        )

        print(
            f"REAL target CZ edges:   "
            f"{len(real_cz_edges)}"
        )

        print(
            f"12.1 calibrated CZ:     "
            f"{len(discovery_cz_edges)}"
        )

        print(
            f"Target/12.1 CZ match:   "
            f"{target_vs_discovery_match}"
        )

        backend_json = {
            "calibration_set_id":
                calibration_set_id,

            "physical_qubits":
                list(
                    backend.physical_qubits
                ),

            "target_operations":
                target_ops,

            "real_target_cz_edge_count":
                len(real_cz_edges),

            "discovery_calibrated_cz_edge_count":
                len(discovery_cz_edges),

            "target_vs_discovery_cz_match":
                target_vs_discovery_match,

            "topologies": {},
        }

        # =====================================================================
        # Topology loop
        # =====================================================================

        for topology_name in sorted(
            topologies.keys()
        ):

            logical_edges = (
                topologies[
                    topology_name
                ]
            )

            key = (
                backend_name,
                topology_name,
            )

            topology_anchors = (
                anchors.get(
                    key,
                    [],
                )
            )

            print()
            print(
                f"{topology_name}: "
                f"{len(topology_anchors)} anchor layout(s)"
            )

            topology_results = []

            for anchor_id, anchor in enumerate(
                topology_anchors,
                start=1,
            ):

                row = anchor["row"]
                reasons = anchor["reasons"]

                physical_layout_names = (
                    layout_from_row(
                        row
                    )
                )

                missing_physical = [
                    qb
                    for qb
                    in physical_layout_names
                    if qb not in physical_to_index
                ]

                if missing_physical:

                    result = {
                        "backend":
                            backend_name,

                        "topology":
                            topology_name,

                        "anchor_id":
                            anchor_id,

                        "selection_reasons":
                            ";".join(reasons),

                        "layout":
                            row["layout"],

                        "calibration_set_id":
                            calibration_set_id,

                        "compile_success":
                            False,

                        "strict_pass":
                            False,

                        "error_type":
                            "PhysicalQubitMappingError",

                        "error":
                            (
                                "Missing backend physical qubits: "
                                + ", ".join(
                                    missing_physical
                                )
                            ),
                    }

                    output_rows.append(
                        result
                    )

                    topology_results.append(
                        result
                    )

                    print(
                        f"  FAIL anchor {anchor_id}: "
                        f"{row['layout']} "
                        f"(physical mapping error)"
                    )

                    continue

                initial_layout = [
                    physical_to_index[qb]
                    for qb
                    in physical_layout_names
                ]

                expected_edges = {
                    normalize_edge(
                        physical_layout_names[
                            int(a)
                        ],
                        physical_layout_names[
                            int(b)
                        ],
                    )
                    for a, b
                    in logical_edges
                }

                qc = build_probe_circuit(
                    logical_edges
                )

                try:
                    tqc = transpile(
                        qc,
                        target=real_target,
                        initial_layout=
                            initial_layout,

                        routing_method=
                            "none",

                        optimization_level=
                            OPTIMIZATION_LEVEL,

                        seed_transpiler=
                            SEED_TRANSPILER,
                    )

                    counts = (
                        operation_counts(
                            tqc
                        )
                    )

                    (
                        two_q_loci,
                        mapping_mode,
                    ) = physical_two_qubit_edges_from_transpiled(
                        tqc,
                        backend,
                        initial_layout,
                    )

                    transpiled_cz_edges = {
                        item["pair"]
                        for item
                        in two_q_loci
                        if item[
                            "operation"
                        ] == "cz"
                    }

                    all_two_q_edges = {
                        item["pair"]
                        for item
                        in two_q_loci
                    }

                    unexpected_two_q_edges = (
                        all_two_q_edges
                        - expected_edges
                    )

                    unexpected_cz_edges = (
                        transpiled_cz_edges
                        - expected_edges
                    )

                    swap_count = int(
                        counts.get(
                            "swap",
                            0,
                        )
                    )

                    routing_perm, routing_identity = (
                        routing_permutation_info(
                            tqc
                        )
                    )

                    # Kept only as a diagnostic.  It is NOT itself a failure
                    # criterion because IQM/Qiskit can carry mapping/permutation
                    # metadata here even when routing_method="none".
                    final_layout_present = (
                        getattr(tqc, "layout", None) is not None
                        and getattr(tqc.layout, "final_layout", None) is not None
                    )

                    # Check final operation names against backend target.
                    #
                    # Qiskit may retain zero-cost/control-flow metadata in some
                    # circumstances; we tolerate only barrier as an extra.
                    allowed_ops = (
                        set(target_ops)
                        | {"barrier"}
                    )

                    non_native_ops = sorted(
                        set(counts)
                        - allowed_ops
                    )

                    strict_pass = (
                        target_vs_discovery_match
                        and swap_count == 0
                        and routing_identity is True
                        and len(unexpected_two_q_edges) == 0
                        and len(unexpected_cz_edges) == 0
                        and len(non_native_ops) == 0
                    )

                    result = {
                        "backend":
                            backend_name,

                        "topology":
                            topology_name,

                        "anchor_id":
                            anchor_id,

                        "selection_reasons":
                            ";".join(reasons),

                        "layout":
                            row["layout"],

                        "q0_C":
                            row["q0_C"],

                        "q1_D":
                            row["q1_D"],

                        "q2_P":
                            row["q2_P"],

                        "q3_H":
                            row["q3_H"],

                        "q4_M1":
                            row["q4_M1"],

                        "q5_M2":
                            row["q5_M2"],

                        "calibration_set_id":
                            calibration_set_id,

                        "initial_layout_indices":
                            json.dumps(
                                initial_layout
                            ),

                        "expected_physical_edges":
                            json.dumps(
                                sorted(
                                    [
                                        list(e)
                                        for e
                                        in expected_edges
                                    ]
                                )
                            ),

                        "compile_success":
                            True,

                        "strict_pass":
                            strict_pass,

                        "depth":
                            tqc.depth(),

                        "size":
                            tqc.size(),

                        "cz_count":
                            int(
                                counts.get(
                                    "cz",
                                    0,
                                )
                            ),

                        "swap_count":
                            swap_count,

                        "measure_count":
                            int(
                                counts.get(
                                    "measure",
                                    0,
                                )
                            ),

                        "operation_counts":
                            json.dumps(
                                counts,
                                sort_keys=True,
                            ),

                        "mapping_mode":
                            mapping_mode,

                        "transpiled_width":
                            tqc.num_qubits,

                        "routing_permutation":
                            json.dumps(routing_perm),

                        "routing_identity":
                            routing_identity,

                        "target_vs_discovery_cz_match":
                            target_vs_discovery_match,

                        "real_target_cz_edge_count":
                            len(real_cz_edges),

                        "transpiled_cz_edges":
                            json.dumps(
                                sorted(
                                    [
                                        list(e)
                                        for e
                                        in transpiled_cz_edges
                                    ]
                                )
                            ),

                        "unexpected_two_q_edges":
                            json.dumps(
                                sorted(
                                    [
                                        list(e)
                                        for e
                                        in unexpected_two_q_edges
                                    ]
                                )
                            ),

                        "unexpected_cz_edges":
                            json.dumps(
                                sorted(
                                    [
                                        list(e)
                                        for e
                                        in unexpected_cz_edges
                                    ]
                                )
                            ),

                        "non_native_ops":
                            json.dumps(
                                non_native_ops
                            ),

                        "final_layout_present":
                            final_layout_present,

                        "error_type":
                            "",

                        "error":
                            "",
                    }

                    output_rows.append(
                        result
                    )

                    topology_results.append(
                        result
                    )

                    status = (
                        "PASS"
                        if strict_pass
                        else "AUDIT_FAIL"
                    )

                    print(
                        f"  {status} anchor {anchor_id}: "
                        f"{row['layout']} | "
                        f"CZ={result['cz_count']} "
                        f"SWAP={swap_count} "
                        f"depth={result['depth']} | "
                        f"{','.join(reasons)}"
                    )

                    if not strict_pass:

                        if unexpected_two_q_edges:
                            print(
                                "    unexpected 2q edges: "
                                f"{sorted(unexpected_two_q_edges)}"
                            )

                        if non_native_ops:
                            print(
                                "    non-native ops: "
                                f"{non_native_ops}"
                            )

                        if routing_identity is not True:
                            print(
                                f"    routing permutation not identity: "
                                f"{routing_perm}"
                            )

                        if not target_vs_discovery_match:
                            print(
                                "    REAL target CZ graph does not match "
                                "Week-12.1 calibrated CZ graph"
                            )

                        if final_layout_present:
                            print(
                                "    note: final_layout metadata present "
                                "(diagnostic only)"
                            )

                except Exception as exc:

                    result = {
                        "backend":
                            backend_name,

                        "topology":
                            topology_name,

                        "anchor_id":
                            anchor_id,

                        "selection_reasons":
                            ";".join(reasons),

                        "layout":
                            row["layout"],

                        "q0_C":
                            row["q0_C"],

                        "q1_D":
                            row["q1_D"],

                        "q2_P":
                            row["q2_P"],

                        "q3_H":
                            row["q3_H"],

                        "q4_M1":
                            row["q4_M1"],

                        "q5_M2":
                            row["q5_M2"],

                        "calibration_set_id":
                            calibration_set_id,

                        "compile_success":
                            False,

                        "strict_pass":
                            False,

                        "error_type":
                            type(exc).__name__,

                        "error":
                            str(exc),
                    }

                    output_rows.append(
                        result
                    )

                    topology_results.append(
                        result
                    )

                    print(
                        f"  FAIL anchor {anchor_id}: "
                        f"{row['layout']} | "
                        f"{type(exc).__name__}: {exc}"
                    )

            n_total = len(
                topology_results
            )

            n_compile = sum(
                bool(r.get(
                    "compile_success"
                ))
                for r
                in topology_results
            )

            n_pass = sum(
                bool(r.get(
                    "strict_pass"
                ))
                for r
                in topology_results
            )

            print(
                f"  summary: "
                f"{n_pass}/{n_total} strict PASS "
                f"({n_compile}/{n_total} compiled)"
            )

            backend_json[
                "topologies"
            ][topology_name] = {
                "anchors":
                    n_total,

                "compiled":
                    n_compile,

                "strict_pass":
                    n_pass,

                "results":
                    topology_results,
            }

        output_json[
            "backends"
        ][backend_name] = (
            backend_json
        )

    # Token is intentionally never stored or printed.
    del token

    # =========================================================================
    # Save outputs
    # =========================================================================

    csv_fields = [
        "backend",
        "topology",
        "anchor_id",
        "selection_reasons",
        "layout",

        "q0_C",
        "q1_D",
        "q2_P",
        "q3_H",
        "q4_M1",
        "q5_M2",

        "calibration_set_id",
        "initial_layout_indices",

        "compile_success",
        "strict_pass",

        "depth",
        "size",
        "cz_count",
        "swap_count",
        "measure_count",

        "operation_counts",

        "mapping_mode",
        "transpiled_width",
        "routing_permutation",
        "routing_identity",
        "target_vs_discovery_cz_match",
        "real_target_cz_edge_count",

        "expected_physical_edges",
        "transpiled_cz_edges",
        "unexpected_two_q_edges",
        "unexpected_cz_edges",

        "non_native_ops",
        "final_layout_present",

        "error_type",
        "error",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in output_rows:
            writer.writerow(row)

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_json,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================================
    # Final summary
    # =========================================================================

    total = len(output_rows)

    compiled = sum(
        bool(row.get(
            "compile_success"
        ))
        for row
        in output_rows
    )

    passed = sum(
        bool(row.get(
            "strict_pass"
        ))
        for row
        in output_rows
    )

    print()
    print("=" * 100)
    print("12.2B v2 REAL-TARGET STRICT COMPILE AUDIT COMPLETE")
    print("=" * 100)

    print(
        f"Anchor layouts audited:  {total}"
    )

    print(
        f"Compiled successfully:   {compiled}/{total}"
    )

    print(
        f"Strict no-routing PASS:  {passed}/{total}"
    )

    print()
    print(
        f"Saved CSV:  {OUTPUT_CSV}"
    )

    print(
        f"Saved JSON: {OUTPUT_JSON}"
    )

    print()
    print("IMPORTANT:")
    print("No backend.run() call exists in this script.")
    print("No QPU job was submitted and zero shots were used.")
    print("Do not continue to 12.3 until this audit output has been interpreted.")


if __name__ == "__main__":
    main()
