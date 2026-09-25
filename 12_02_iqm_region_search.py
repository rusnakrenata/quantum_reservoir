"""
12_02_iqm_region_search.py
--------------------------
Week 12.2A: strict native six-qubit region search on IQM Emerald + Garnet.

Purpose
-------
For each frozen logical topology H0-H6:
1. build the CURRENT calibrated CZ graph from the Week-12.1 discovery JSON,
2. enumerate every injective logical->physical mapping whose required logical
   edges are native calibrated CZ edges,
3. attach calibration metrics:
      - T1
      - T2 echo
      - standard readout fidelity
      - active CZ implementation fidelity
4. construct a NON-WEIGHTED Pareto frontier per (backend, topology).

No circuit is transpiled.
No circuit is submitted.
No 2026 test data are touched.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULTS_DIR = Path("results")
INPUT_JSON = RESULTS_DIR / "12_01_iqm_discovery.json"
BACKENDS = ("emerald", "garnet")

LOGICAL_LABELS = {
    0: "C",
    1: "D",
    2: "P",
    3: "H",
    4: "M1",
    5: "M2",
}

MEMORY_LOGICAL_QUBITS = (4, 5)

TOPOLOGIES_RAW = {
    "H0": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    "H1": [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)],
    "H2": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)],
    "H3": [(3, 2), (2, 1), (1, 0), (0, 4), (4, 5)],
    "H4": [(2, 1), (1, 0), (0, 4), (3, 4), (4, 5)],
    "H5": [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (3, 0), (4, 5)],
    "H6": [(0, 4), (4, 3), (3, 2), (2, 5), (5, 1), (1, 0), (4, 5)],
}


def normalize_edge(a: Any, b: Any):
    return tuple(sorted((a, b), key=str))


TOPOLOGIES = {
    name: tuple(sorted({normalize_edge(a, b) for a, b in edges}))
    for name, edges in TOPOLOGIES_RAW.items()
}

RE_T1 = re.compile(r"^characterization\.model\.(QB\d+)\.t1_time$")
RE_T2_ECHO = re.compile(r"^characterization\.model\.(QB\d+)\.t2_echo_time$")
RE_READOUT = re.compile(r"^metrics\.ssro\.measure\.([^.]+)\.(QB\d+)\.fidelity$")
RE_CZ_FIDELITY = re.compile(
    r"^metrics\.irb\.cz\.([^.]+)\.(QB\d+)__(QB\d+)\.fidelity(?::.*)?$"
)


def finite_number(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def safe_min(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return min(vals) if vals else None


def mean(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def layout_string(mapping: dict[int, str]) -> str:
    return "[" + ",".join(mapping[q] for q in range(6)) + "]"


def logical_mapping_string(mapping: dict[int, str]) -> str:
    return ", ".join(
        f"q{q}({LOGICAL_LABELS[q]})={mapping[q]}"
        for q in range(6)
    )


def extract_calibrated_cz_graph(system: dict):
    dqa = system["dynamic_architecture"]
    cz = dqa.get("gates", {}).get("cz", {})
    implementations = cz.get("implementations", {})
    default_impl = cz.get("default_implementation")
    overrides_raw = cz.get("override_default_implementation", {})

    all_pairs = set()
    impl_pairs = defaultdict(set)

    for impl_name, impl_data in implementations.items():
        for locus in impl_data.get("loci", []):
            if len(locus) != 2:
                continue
            a, b = locus
            if not (
                isinstance(a, str)
                and isinstance(b, str)
                and a.startswith("QB")
                and b.startswith("QB")
            ):
                continue
            pair = normalize_edge(a, b)
            all_pairs.add(pair)
            impl_pairs[impl_name].add(pair)

    active_impl_for_pair = {pair: default_impl for pair in all_pairs}

    for key, impl_name in overrides_raw.items():
        if isinstance(key, str):
            parts = [p.strip() for p in key.split(",") if p.strip()]
        elif isinstance(key, (list, tuple)):
            parts = list(key)
        else:
            continue

        if len(parts) == 2:
            active_impl_for_pair[normalize_edge(parts[0], parts[1])] = impl_name

    for pair in all_pairs:
        active = active_impl_for_pair.get(pair)
        if active is None or pair not in impl_pairs.get(active, set()):
            candidates = [
                impl_name
                for impl_name, pairs in impl_pairs.items()
                if pair in pairs
            ]
            if len(candidates) == 1:
                active_impl_for_pair[pair] = candidates[0]

    adjacency = defaultdict(set)
    for a, b in all_pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)

    return dict(adjacency), all_pairs, active_impl_for_pair


def extract_metrics(system: dict, active_impl_for_pair: dict):
    qm = system.get("quality_metrics", {})
    observations = qm.get("observations", [])

    t1_us = {}
    t2_echo_us = {}
    readout_by_impl = {}
    cz_by_impl_pair = {}
    parsed_counts = defaultdict(int)

    for obs in observations:
        if obs.get("invalid", False):
            continue

        field = obs.get("dut_field", "")
        value = finite_number(obs.get("value"))
        if value is None:
            continue

        m = RE_T1.match(field)
        if m:
            t1_us[m.group(1)] = value * 1e6
            parsed_counts["t1"] += 1
            continue

        m = RE_T2_ECHO.match(field)
        if m:
            t2_echo_us[m.group(1)] = value * 1e6
            parsed_counts["t2_echo"] += 1
            continue

        m = RE_READOUT.match(field)
        if m:
            impl, qb = m.groups()
            readout_by_impl[(impl, qb)] = value
            parsed_counts["readout_fidelity"] += 1
            continue

        m = RE_CZ_FIDELITY.match(field)
        if m:
            impl, qb_a, qb_b = m.groups()
            pair = normalize_edge(qb_a, qb_b)
            cz_by_impl_pair[(impl, pair)] = value
            parsed_counts["cz_fidelity"] += 1
            continue

    measure_default_impl = (
        system["dynamic_architecture"]
        .get("gates", {})
        .get("measure", {})
        .get("default_implementation")
    )

    qubits = system.get("static_architecture", {}).get("qubits", [])

    readout_fidelity = {}
    for qb in qubits:
        value = readout_by_impl.get((measure_default_impl, qb))
        if value is None:
            candidates = [
                v for (impl, q), v in readout_by_impl.items()
                if q == qb
            ]
            if len(candidates) == 1:
                value = candidates[0]
        readout_fidelity[qb] = value

    cz_fidelity = {}
    for pair, active_impl in active_impl_for_pair.items():
        value = cz_by_impl_pair.get((active_impl, pair))
        if value is None:
            candidates = [
                v for (impl, p), v in cz_by_impl_pair.items()
                if p == pair
            ]
            if len(candidates) == 1:
                value = candidates[0]
        cz_fidelity[pair] = value

    return {
        "t1_us": t1_us,
        "t2_echo_us": t2_echo_us,
        "readout_fidelity": readout_fidelity,
        "cz_fidelity": cz_fidelity,
        "parsed_counts": dict(parsed_counts),
        "measure_default_impl": measure_default_impl,
    }


def logical_adjacency(edges):
    adjacency = {q: set() for q in range(6)}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def enumerate_native_embeddings(physical_adjacency, logical_edges):
    logical_adj = logical_adjacency(logical_edges)

    physical_nodes = sorted(
        physical_adjacency.keys(),
        key=lambda x: int(x.removeprefix("QB")),
    )

    logical_order = sorted(
        range(6),
        key=lambda q: (-len(logical_adj[q]), q),
    )

    candidates = {
        logical_q: [
            p for p in physical_nodes
            if len(physical_adjacency[p]) >= len(logical_adj[logical_q])
        ]
        for logical_q in range(6)
    }

    mappings = []
    assigned = {}
    used_physical = set()

    def backtrack(depth: int):
        if depth == len(logical_order):
            mappings.append({q: assigned[q] for q in range(6)})
            return

        logical_q = logical_order[depth]

        for physical_q in candidates[logical_q]:
            if physical_q in used_physical:
                continue

            valid = True
            for logical_neighbor in logical_adj[logical_q]:
                if logical_neighbor not in assigned:
                    continue

                physical_neighbor = assigned[logical_neighbor]
                if physical_neighbor not in physical_adjacency.get(physical_q, set()):
                    valid = False
                    break

            if not valid:
                continue

            assigned[logical_q] = physical_q
            used_physical.add(physical_q)
            backtrack(depth + 1)
            used_physical.remove(physical_q)
            del assigned[logical_q]

    backtrack(0)
    return mappings


def score_embedding(mapping, logical_edges, metrics):
    t1 = metrics["t1_us"]
    t2 = metrics["t2_echo_us"]
    ro = metrics["readout_fidelity"]
    cz = metrics["cz_fidelity"]

    physical_qubits = [mapping[q] for q in range(6)]
    memory_qubits = [mapping[q] for q in MEMORY_LOGICAL_QUBITS]

    physical_edges = [
        normalize_edge(mapping[a], mapping[b])
        for a, b in logical_edges
    ]

    all_t1 = [t1.get(qb) for qb in physical_qubits]
    all_t2 = [t2.get(qb) for qb in physical_qubits]
    mem_t1 = [t1.get(qb) for qb in memory_qubits]
    mem_t2 = [t2.get(qb) for qb in memory_qubits]
    all_ro = [ro.get(qb) for qb in physical_qubits]
    all_cz = [cz.get(edge) for edge in physical_edges]

    complete = all(
        value is not None
        for value in (all_t1 + all_t2 + all_ro + all_cz)
    )

    min_ro_fid = safe_min(all_ro)
    min_cz_fid = safe_min(all_cz)

    return {
        "complete_metrics": complete,
        "min_t1_us": safe_min(all_t1),
        "mean_t1_us": mean(all_t1),
        "min_t2_echo_us": safe_min(all_t2),
        "mean_t2_echo_us": mean(all_t2),
        "memory_min_t1_us": safe_min(mem_t1),
        "memory_min_t2_echo_us": safe_min(mem_t2),
        "min_readout_fidelity": min_ro_fid,
        "worst_readout_error": None if min_ro_fid is None else 1.0 - min_ro_fid,
        "min_cz_fidelity": min_cz_fid,
        "worst_cz_error": None if min_cz_fid is None else 1.0 - min_cz_fid,
        "missing_t1_count": sum(v is None for v in all_t1),
        "missing_t2_count": sum(v is None for v in all_t2),
        "missing_readout_count": sum(v is None for v in all_ro),
        "missing_cz_count": sum(v is None for v in all_cz),
    }


PARETO_FIELDS = (
    "min_t1_us",
    "min_t2_echo_us",
    "memory_min_t1_us",
    "memory_min_t2_echo_us",
    "min_readout_fidelity",
    "min_cz_fidelity",
)


def dominates(a: dict, b: dict) -> bool:
    av = [a[field] for field in PARETO_FIELDS]
    bv = [b[field] for field in PARETO_FIELDS]

    if any(x is None or y is None for x, y in zip(av, bv)):
        return False

    return (
        all(x >= y for x, y in zip(av, bv))
        and any(x > y for x, y in zip(av, bv))
    )


def pareto_frontier(rows):
    frontier = []

    for row in rows:
        if not row["complete_metrics"]:
            continue

        if any(dominates(other, row) for other in frontier):
            continue

        frontier = [
            other
            for other in frontier
            if not dominates(row, other)
        ]
        frontier.append(row)

    return frontier


CSV_FIELDS = [
    "backend", "topology", "edge_count", "embedding_id",
    "layout", "logical_mapping",
    "q0_C", "q1_D", "q2_P", "q3_H", "q4_M1", "q5_M2",
    "complete_metrics",
    "min_t1_us", "mean_t1_us",
    "min_t2_echo_us", "mean_t2_echo_us",
    "memory_min_t1_us", "memory_min_t2_echo_us",
    "min_readout_fidelity", "worst_readout_error",
    "min_cz_fidelity", "worst_cz_error",
    "missing_t1_count", "missing_t2_count",
    "missing_readout_count", "missing_cz_count",
    "pareto",
]


def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_JSON}. Run 12_01_iqm_discovery.py first."
        )

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        discovery = json.load(f)

    print("=" * 96)
    print("WEEK 12.2A — IQM STRICT NATIVE REGION SEARCH")
    print("=" * 96)

    print("\nLogical F4 mapping")
    print("------------------")
    for q in range(6):
        print(f"q{q} = {LOGICAL_LABELS[q]}")

    print("\nTopologies")
    print("----------")
    for name, edges in TOPOLOGIES.items():
        edge_text = ", ".join(f"{a}{b}" for a, b in edges)
        print(f"{name}: {edge_text}   ({len(edges)} edges)")

    all_rows = []
    all_pareto_rows = []
    summary_rows = []

    json_output = {
        "input_file": str(INPUT_JSON),
        "backends": {},
        "topologies": {
            name: {
                "edges": [list(edge) for edge in edges],
                "edge_count": len(edges),
            }
            for name, edges in TOPOLOGIES.items()
        },
    }

    for backend_name in BACKENDS:
        if backend_name not in discovery["systems"]:
            raise RuntimeError(
                f"Backend '{backend_name}' not present in {INPUT_JSON}"
            )

        system = discovery["systems"][backend_name]

        physical_adj, calibrated_edges, active_impl = extract_calibrated_cz_graph(system)
        metrics = extract_metrics(system, active_impl)

        static_edges = {
            normalize_edge(a, b)
            for a, b in system["static_architecture"].get("connectivity", [])
            if str(a).startswith("QB") and str(b).startswith("QB")
        }

        print("\n" + "=" * 96)
        print(f"BACKEND: {backend_name.upper()}")
        print("=" * 96)
        print(f"Qubits:                    {len(system['static_architecture'].get('qubits', []))}")
        print(f"Static qubit edges:        {len(static_edges)}")
        print(f"Currently calibrated CZ:   {len(calibrated_edges)}")
        print(f"Measure default impl:      {metrics['measure_default_impl']}")
        print(f"Parsed quality metrics:    {metrics['parsed_counts']}")

        backend_json = {
            "static_qubit_edge_count": len(static_edges),
            "calibrated_cz_edge_count": len(calibrated_edges),
            "parsed_metric_counts": metrics["parsed_counts"],
            "measure_default_implementation": metrics["measure_default_impl"],
            "topologies": {},
        }

        for topology_name, logical_edges in TOPOLOGIES.items():
            mappings = enumerate_native_embeddings(physical_adj, logical_edges)

            rows = []

            for embedding_id, mapping in enumerate(mappings, start=1):
                scores = score_embedding(mapping, logical_edges, metrics)

                row = {
                    "backend": backend_name,
                    "topology": topology_name,
                    "edge_count": len(logical_edges),
                    "embedding_id": embedding_id,
                    "layout": layout_string(mapping),
                    "logical_mapping": logical_mapping_string(mapping),
                    "q0_C": mapping[0],
                    "q1_D": mapping[1],
                    "q2_P": mapping[2],
                    "q3_H": mapping[3],
                    "q4_M1": mapping[4],
                    "q5_M2": mapping[5],
                    **scores,
                    "pareto": False,
                }
                rows.append(row)

            frontier = pareto_frontier(rows)
            frontier_keys = {row["layout"] for row in frontier}

            for row in rows:
                row["pareto"] = row["layout"] in frontier_keys

            complete_count = sum(row["complete_metrics"] for row in rows)

            all_rows.extend(rows)
            all_pareto_rows.extend(frontier)

            complete_rows = [row for row in rows if row["complete_metrics"]]

            def best_row(field):
                if not complete_rows:
                    return None
                return max(complete_rows, key=lambda r: r[field])

            best_t2 = best_row("min_t2_echo_us")
            best_mem_t2 = best_row("memory_min_t2_echo_us")
            best_cz = best_row("min_cz_fidelity")
            best_ro = best_row("min_readout_fidelity")

            summary = {
                "backend": backend_name,
                "topology": topology_name,
                "edge_count": len(logical_edges),
                "native_embeddings": len(rows),
                "complete_metric_embeddings": complete_count,
                "pareto_embeddings": len(frontier),
                "best_min_t2_echo_us": None if best_t2 is None else best_t2["min_t2_echo_us"],
                "best_min_t2_layout": None if best_t2 is None else best_t2["layout"],
                "best_memory_min_t2_echo_us": None if best_mem_t2 is None else best_mem_t2["memory_min_t2_echo_us"],
                "best_memory_t2_layout": None if best_mem_t2 is None else best_mem_t2["layout"],
                "best_min_cz_fidelity": None if best_cz is None else best_cz["min_cz_fidelity"],
                "best_cz_layout": None if best_cz is None else best_cz["layout"],
                "best_min_readout_fidelity": None if best_ro is None else best_ro["min_readout_fidelity"],
                "best_readout_layout": None if best_ro is None else best_ro["layout"],
            }

            summary_rows.append(summary)

            backend_json["topologies"][topology_name] = {
                **summary,
                "pareto_layouts": [row["layout"] for row in frontier],
            }

            print(f"\n{topology_name} ({len(logical_edges)} edges)")
            print(f"  native embeddings:          {len(rows)}")
            print(f"  complete metrics:           {complete_count}")
            print(f"  Pareto embeddings:          {len(frontier)}")

            if best_t2 is not None:
                print(
                    f"  best min T2echo:             "
                    f"{best_t2['min_t2_echo_us']:.3f} us  {best_t2['layout']}"
                )
                print(
                    f"  best memory min T2echo:      "
                    f"{best_mem_t2['memory_min_t2_echo_us']:.3f} us  "
                    f"{best_mem_t2['layout']}"
                )
                print(
                    f"  best worst-edge CZ fidelity: "
                    f"{best_cz['min_cz_fidelity']:.6f}  {best_cz['layout']}"
                )
                print(
                    f"  best worst-qubit RO fid.:    "
                    f"{best_ro['min_readout_fidelity']:.6f}  {best_ro['layout']}"
                )

        json_output["backends"][backend_name] = backend_json

    all_csv = RESULTS_DIR / "12_02_iqm_all_native_embeddings.csv"
    write_csv(all_csv, all_rows)

    pareto_csv = RESULTS_DIR / "12_02_iqm_pareto_embeddings.csv"
    for row in all_pareto_rows:
        row["pareto"] = True
    write_csv(pareto_csv, all_pareto_rows)

    summary_csv = RESULTS_DIR / "12_02_iqm_region_summary.csv"
    summary_fields = [
        "backend", "topology", "edge_count",
        "native_embeddings", "complete_metric_embeddings", "pareto_embeddings",
        "best_min_t2_echo_us", "best_min_t2_layout",
        "best_memory_min_t2_echo_us", "best_memory_t2_layout",
        "best_min_cz_fidelity", "best_cz_layout",
        "best_min_readout_fidelity", "best_readout_layout",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    output_json = RESULTS_DIR / "12_02_iqm_region_search.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 96)
    print("12.2A REGION SEARCH COMPLETE")
    print("=" * 96)
    print(f"All embeddings:    {all_csv}")
    print(f"Pareto embeddings: {pareto_csv}")
    print(f"Summary:           {summary_csv}")
    print(f"JSON:              {output_json}")
    print("\nIMPORTANT:")
    print("No transpilation and no QPU execution were performed.")
    print("This step used only the Week-12.1 architecture/calibration snapshot.")
    print("Do not continue to 12.2B strict compile audit until these results are interpreted.")


if __name__ == "__main__":
    main()
