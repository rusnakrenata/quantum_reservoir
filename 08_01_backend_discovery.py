"""
Week 8 - Step 8.1
IBM Quantum backend discovery and topology profiling.

Purpose
-------
1. Discover all real IBM Quantum backends accessible to the account.
2. Record backend status and basic hardware information.
3. Extract the physical coupling graph.
4. Compute graph-level topology metrics.
5. Test whether an actual simple 6-qubit chain exists.
6. Count physical hardware triangles.
7. Test whether the exact logical T2 QRC topology can be embedded
   natively into the IBM coupling graph without routing/SWAPs.
8. Save topology data and visualizations.

Important
---------
This script DOES NOT:
- choose the best physical qubits according to noise,
- inspect T1/T2,
- inspect gate/readout errors,
- optimize logical-to-physical placement,
- transpile the QRC.

Those belong to later Week 8 / Week 9 steps.
"""

from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import math

import pandas as pd

from ibm_account import get_service


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_QRC_QUBITS = 6


# ------------------------------------------------------------
# Frozen logical T2 dual-boundary QRC topology
# ------------------------------------------------------------

T2_LOGICAL_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (2, 4),
    (3, 4),
    (3, 5),
    (4, 5),
]


# ============================================================
# General utilities
# ============================================================

def package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def iso_or_none(value):
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return str(value)


def safe_attr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def normalize_edge(edge):
    return int(edge[0]), int(edge[1])


# ============================================================
# Graph construction
# ============================================================

def make_undirected_edges(directed_edges):
    """
    Convert directed coupling edges into unique
    undirected physical edges.
    """

    edges = set()

    for u, v in directed_edges:

        if u == v:
            continue

        edges.add(
            tuple(
                sorted(
                    (
                        int(u),
                        int(v),
                    )
                )
            )
        )

    return sorted(edges)


def adjacency_list(
    num_qubits,
    undirected_edges,
):
    adjacency = {
        q: set()
        for q in range(num_qubits)
    }

    for u, v in undirected_edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    return adjacency


# ============================================================
# Connected components
# ============================================================

def connected_components(
    num_qubits,
    undirected_edges,
):
    adjacency = adjacency_list(
        num_qubits,
        undirected_edges,
    )

    visited = set()
    components = []

    for start in range(num_qubits):

        if start in visited:
            continue

        stack = [start]
        component = []

        while stack:

            node = stack.pop()

            if node in visited:
                continue

            visited.add(node)
            component.append(node)

            for neighbour in adjacency[node]:

                if neighbour not in visited:
                    stack.append(neighbour)

        components.append(
            sorted(component)
        )

    components.sort(
        key=len,
        reverse=True,
    )

    return components


# ============================================================
# Simple 6-qubit path
# ============================================================

def find_simple_path_of_length_n(
    num_qubits,
    undirected_edges,
    n_vertices=6,
):
    """
    Find one simple hardware path containing n_vertices.

    Example for n_vertices=6:

        p0 - p1 - p2 - p3 - p4 - p5
    """

    adjacency = adjacency_list(
        num_qubits,
        undirected_edges,
    )

    def dfs(current, path):

        if len(path) == n_vertices:
            return path.copy()

        for neighbour in sorted(
            adjacency[current]
        ):

            if neighbour in path:
                continue

            path.append(neighbour)

            result = dfs(
                neighbour,
                path,
            )

            if result is not None:
                return result

            path.pop()

        return None

    for start in range(num_qubits):

        result = dfs(
            start,
            [start],
        )

        if result is not None:
            return result

    return None


def verify_path(
    path,
    undirected_edges,
):
    if path is None:
        return False

    edge_set = set(
        undirected_edges
    )

    for i in range(
        len(path) - 1
    ):

        edge = tuple(
            sorted(
                (
                    path[i],
                    path[i + 1],
                )
            )
        )

        if edge not in edge_set:
            return False

    return True


# ============================================================
# Triangle search
# ============================================================

def find_triangles(
    num_qubits,
    undirected_edges,
):
    """
    Find every physical triangle.

    Triangle (a,b,c) requires physical edges:

        a-b
        a-c
        b-c
    """

    adjacency = adjacency_list(
        num_qubits,
        undirected_edges,
    )

    triangles = set()

    for a in range(num_qubits):

        for b in adjacency[a]:

            if b <= a:
                continue

            common = (
                adjacency[a]
                & adjacency[b]
            )

            for c in common:

                if c <= b:
                    continue

                triangles.add(
                    (
                        a,
                        b,
                        c,
                    )
                )

    return sorted(triangles)


# ============================================================
# Exact native logical topology embedding
# ============================================================

def find_native_embedding(
    num_physical_qubits,
    physical_edges,
    logical_edges,
):
    """
    Find an injective mapping

        logical qubit -> physical qubit

    such that every logical edge corresponds to a direct
    physical hardware edge.

    Extra physical edges are allowed.

    No SWAP or routing is allowed.

    Returns
    -------
    dict or None

    Example:

        {
            0: 23,
            1: 24,
            2: 25,
            ...
        }

    means logical q0 -> physical q23, etc.
    """

    physical_adj = adjacency_list(
        num_physical_qubits,
        physical_edges,
    )

    logical_nodes = sorted(
        {
            node
            for edge in logical_edges
            for node in edge
        }
    )

    logical_adj = {
        node: set()
        for node in logical_nodes
    }

    for u, v in logical_edges:
        logical_adj[u].add(v)
        logical_adj[v].add(u)

    # --------------------------------------------------------
    # Assign the most constrained logical qubits first.
    #
    # T2 degrees:
    #
    # q0 = 1
    # q1 = 2
    # q2 = 3
    # q3 = 3
    # q4 = 3
    # q5 = 2
    #
    # Therefore degree-3 nodes are attempted first.
    # --------------------------------------------------------

    logical_order = sorted(
        logical_nodes,
        key=lambda q: (
            -len(logical_adj[q]),
            q,
        ),
    )

    # Physical candidates must at least have sufficient degree.
    candidates = {}

    for logical_q in logical_nodes:

        required_degree = len(
            logical_adj[logical_q]
        )

        candidates[logical_q] = [
            physical_q
            for physical_q
            in range(num_physical_qubits)
            if len(
                physical_adj[physical_q]
            )
            >= required_degree
        ]

    mapping = {}
    used_physical = set()

    def backtrack(index):

        if index == len(logical_order):
            return mapping.copy()

        logical_q = logical_order[index]

        for physical_q in candidates[
            logical_q
        ]:

            if physical_q in used_physical:
                continue

            valid = True

            # Check every logical neighbour that has
            # already been assigned.
            for logical_neighbour in (
                logical_adj[logical_q]
            ):

                if logical_neighbour not in mapping:
                    continue

                physical_neighbour = mapping[
                    logical_neighbour
                ]

                if (
                    physical_neighbour
                    not in
                    physical_adj[physical_q]
                ):
                    valid = False
                    break

            if not valid:
                continue

            mapping[logical_q] = physical_q
            used_physical.add(physical_q)

            result = backtrack(
                index + 1
            )

            if result is not None:
                return result

            used_physical.remove(
                physical_q
            )

            del mapping[
                logical_q
            ]

        return None

    return backtrack(0)


def verify_native_embedding(
    mapping,
    physical_edges,
    logical_edges,
):
    """
    Independently verify the returned mapping.
    """

    if mapping is None:
        return False

    physical_edge_set = set(
        physical_edges
    )

    for logical_u, logical_v in logical_edges:

        physical_u = mapping[
            logical_u
        ]

        physical_v = mapping[
            logical_v
        ]

        physical_edge = tuple(
            sorted(
                (
                    physical_u,
                    physical_v,
                )
            )
        )

        if physical_edge not in physical_edge_set:
            return False

    return True


# ============================================================
# General graph metrics
# ============================================================

def graph_metrics(
    num_qubits,
    undirected_edges,
):
    adjacency = adjacency_list(
        num_qubits,
        undirected_edges,
    )

    degrees = {
        q: len(neighbours)
        for q, neighbours
        in adjacency.items()
    }

    components = connected_components(
        num_qubits,
        undirected_edges,
    )

    largest_component_size = (
        len(components[0])
        if components
        else 0
    )

    num_edges = len(
        undirected_edges
    )

    if num_qubits > 0:

        average_degree = (
            2.0
            * num_edges
            / num_qubits
        )

        max_degree = (
            max(degrees.values())
            if degrees
            else 0
        )

    else:

        average_degree = math.nan
        max_degree = 0

    return {
        "num_undirected_edges":
            num_edges,

        "average_degree":
            average_degree,

        "max_degree":
            max_degree,

        "num_connected_components":
            len(components),

        "largest_component_size":
            largest_component_size,

        "component_has_at_least_6_qubits":
            (
                largest_component_size
                >= N_QRC_QUBITS
            ),

        "degrees":
            degrees,

        "components":
            components,
    }


# ============================================================
# Instruction extraction
# ============================================================

def extract_instruction_sets(backend):

    single_qubit = set()
    two_qubit = set()
    other = set()

    try:
        instructions = backend.instructions

    except Exception:
        instructions = []

    for instruction, qargs in instructions:

        name = getattr(
            instruction,
            "name",
            instruction.__class__.__name__,
        )

        try:
            nq = len(qargs)
        except Exception:
            nq = None

        if nq == 1:
            single_qubit.add(name)

        elif nq == 2:
            two_qubit.add(name)

        else:
            other.add(name)

    return (
        sorted(single_qubit),
        sorted(two_qubit),
        sorted(other),
    )


# ============================================================
# Processor information
# ============================================================

def processor_information(
    configuration,
):

    processor = safe_attr(
        configuration,
        "processor_type",
        None,
    )

    if processor is None:

        return {
            "processor_family": None,
            "processor_revision": None,
            "processor_segment": None,
            "processor_raw": None,
        }

    if isinstance(
        processor,
        dict,
    ):

        return {
            "processor_family":
                processor.get(
                    "family"
                ),

            "processor_revision":
                processor.get(
                    "revision"
                ),

            "processor_segment":
                processor.get(
                    "segment"
                ),

            "processor_raw":
                processor,
        }

    return {
        "processor_family":
            safe_attr(
                processor,
                "family",
                None,
            ),

        "processor_revision":
            safe_attr(
                processor,
                "revision",
                None,
            ),

        "processor_segment":
            safe_attr(
                processor,
                "segment",
                None,
            ),

        "processor_raw":
            str(processor),
    }


# ============================================================
# Chain visualization
# ============================================================

def plot_example_chain(
    backend_name,
    chain,
    output_path,
):
    if chain is None:
        return False

    try:
        import matplotlib.pyplot as plt

    except ImportError:
        return False

    fig, ax = plt.subplots(
        figsize=(10, 2.8)
    )

    x = list(
        range(len(chain))
    )

    y = [
        0
        for _ in chain
    ]

    for i in range(
        len(chain) - 1
    ):

        ax.plot(
            [x[i], x[i + 1]],
            [0, 0],
            linewidth=2,
        )

    ax.scatter(
        x,
        y,
        s=900,
        zorder=3,
    )

    for i, qubit in enumerate(chain):

        ax.text(
            x[i],
            0,
            str(qubit),
            ha="center",
            va="center",
            fontsize=11,
            zorder=4,
        )

    ax.set_title(
        f"{backend_name}: "
        f"example physical 6-qubit chain"
    )

    ax.set_xlim(
        -0.6,
        len(chain) - 0.4,
    )

    ax.set_ylim(
        -0.6,
        0.6,
    )

    ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return True


# ============================================================
# Fallback topology visualization
# ============================================================

def plot_topology_fallback(
    backend_name,
    num_qubits,
    undirected_edges,
    chain,
    output_path,
):
    """
    Generic graph visualization.

    This shows connectivity correctly but is not intended
    to reproduce the physical chip geometry.
    """

    try:
        import matplotlib.pyplot as plt
        import networkx as nx

    except ImportError:
        return False

    graph = nx.Graph()

    graph.add_nodes_from(
        range(num_qubits)
    )

    graph.add_edges_from(
        undirected_edges
    )

    positions = nx.spring_layout(
        graph,
        seed=42,
        iterations=100,
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        width=0.6,
        alpha=0.5,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=35,
    )

    if chain is not None:

        chain_edges = [
            (
                chain[i],
                chain[i + 1],
            )
            for i in range(
                len(chain) - 1
            )
        ]

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=chain_edges,
            ax=ax,
            width=3,
        )

        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=chain,
            ax=ax,
            node_size=140,
        )

        labels = {
            q: str(q)
            for q in chain
        }

        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            ax=ax,
            font_size=8,
        )

    ax.set_title(
        f"{backend_name}: "
        f"hardware connectivity graph\n"
        f"highlighted nodes = example 6Q chain"
    )

    ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return True


# ============================================================
# Environment snapshot
# ============================================================

snapshot_utc = datetime.now(
    timezone.utc
)

environment = {
    "snapshot_utc":
        snapshot_utc.isoformat(),

    "qiskit_version":
        package_version(
            "qiskit"
        ),

    "qiskit_ibm_runtime_version":
        package_version(
            "qiskit-ibm-runtime"
        ),

    "qrc_required_qubits":
        N_QRC_QUBITS,

    "t2_logical_edges":
        T2_LOGICAL_EDGES,
}


print("=" * 78)
print("WEEK 8 - STEP 8.1")
print("IBM QUANTUM BACKEND DISCOVERY")
print("=" * 78)

print()

print("Environment:")

print(
    f"  Qiskit: "
    f"{environment['qiskit_version']}"
)

print(
    f"  qiskit-ibm-runtime: "
    f"{environment['qiskit_ibm_runtime_version']}"
)

print(
    f"  Snapshot UTC: "
    f"{environment['snapshot_utc']}"
)

print()

print(
    "Logical T2 topology:"
)

print(
    f"  {T2_LOGICAL_EDGES}"
)

print()


# ============================================================
# IBM connection
# ============================================================

print(
    "Connecting to IBM Quantum Compute Service..."
)

service = get_service()

try:
    active_instance = (
        service.active_instance()
    )
except Exception:
    active_instance = None

try:
    channel = service.channel
except Exception:
    channel = None


environment["channel"] = str(
    channel
)

environment[
    "active_instance_at_start"
] = (
    str(active_instance)
    if active_instance is not None
    else None
)

print(
    f"Channel: {channel}"
)

print(
    f"Active instance: "
    f"{active_instance}"
)

print()


# ============================================================
# Discover hardware
# ============================================================

print(
    "Querying accessible real IBM backends..."
)

backends = service.backends(
    simulator=False
)

print(
    f"Accessible real backends found: "
    f"{len(backends)}"
)

if len(backends) == 0:

    raise RuntimeError(
        "No real IBM Quantum backends were returned."
    )

print()


# ============================================================
# Result containers
# ============================================================

summary_rows = []
edge_rows = []
chain_rows = []
triangle_rows = []
embedding_rows = []
topology_records = []


# ============================================================
# Process each backend
# ============================================================

for index, backend in enumerate(
    backends,
    start=1,
):

    name = backend.name

    print("-" * 78)

    print(
        f"[{index}/{len(backends)}] "
        f"Processing {name}"
    )

    configuration = (
        backend.configuration()
    )

    num_qubits = int(
        backend.num_qubits
    )

    backend_version = safe_attr(
        backend,
        "backend_version",
        None,
    )

    online_date = safe_attr(
        backend,
        "online_date",
        None,
    )

    basis_gates = sorted(
        set(
            safe_attr(
                configuration,
                "basis_gates",
                [],
            )
            or []
        )
    )

    supported_features = (
        safe_attr(
            configuration,
            "supported_features",
            [],
        )
        or []
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    try:

        status = backend.status()

        operational = bool(
            status.operational
        )

        pending_jobs = int(
            status.pending_jobs
        )

        status_msg = getattr(
            status,
            "status_msg",
            None,
        )

    except Exception as exc:

        operational = None
        pending_jobs = None

        status_msg = (
            f"STATUS_ERROR: {exc}"
        )

    # --------------------------------------------------------
    # Operations
    # --------------------------------------------------------

    try:

        operation_names = sorted(
            set(
                backend.operation_names
            )
        )

    except Exception:

        operation_names = sorted(
            set(
                getattr(
                    backend.target,
                    "operation_names",
                    [],
                )
            )
        )

    (
        single_qubit_instructions,
        two_qubit_instructions,
        other_instructions,
    ) = extract_instruction_sets(
        backend
    )

    supports_measure = (
        "measure"
        in operation_names
    )

    supports_reset = (
        "reset"
        in operation_names
    )

    supports_delay = (
        "delay"
        in operation_names
    )

    supports_if_else = (
        "if_else"
        in operation_names
    )

    supports_while_loop = (
        "while_loop"
        in operation_names
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = processor_information(
        configuration
    )

    # --------------------------------------------------------
    # Coupling graph
    # --------------------------------------------------------

    try:

        coupling_map = (
            backend.coupling_map
        )

        if coupling_map is not None:

            directed_edges = [
                normalize_edge(edge)
                for edge
                in coupling_map.get_edges()
            ]

        else:

            directed_edges = []

    except Exception:

        try:

            coupling_map = (
                backend.target
                .build_coupling_map()
            )

            directed_edges = [
                normalize_edge(edge)
                for edge
                in coupling_map.get_edges()
            ]

        except Exception:

            directed_edges = []

    directed_edges = sorted(
        set(directed_edges)
    )

    undirected_edges = (
        make_undirected_edges(
            directed_edges
        )
    )

    # --------------------------------------------------------
    # Basic graph metrics
    # --------------------------------------------------------

    metrics = graph_metrics(
        num_qubits,
        undirected_edges,
    )

    # --------------------------------------------------------
    # TEST 1:
    # actual simple 6Q chain
    # --------------------------------------------------------

    six_qubit_chain = (
        find_simple_path_of_length_n(
            num_qubits,
            undirected_edges,
            N_QRC_QUBITS,
        )
    )

    has_simple_6q_chain = (
        six_qubit_chain
        is not None
    )

    six_qubit_chain_verified = (
        verify_path(
            six_qubit_chain,
            undirected_edges,
        )
    )

    # --------------------------------------------------------
    # TEST 2:
    # physical triangles
    # --------------------------------------------------------

    hardware_triangles = (
        find_triangles(
            num_qubits,
            undirected_edges,
        )
    )

    num_hardware_triangles = len(
        hardware_triangles
    )

    hardware_has_triangle = (
        num_hardware_triangles > 0
    )

    # --------------------------------------------------------
    # TEST 3:
    # exact native T2 topology embedding
    # --------------------------------------------------------

    t2_native_mapping = (
        find_native_embedding(
            num_physical_qubits=
                num_qubits,

            physical_edges=
                undirected_edges,

            logical_edges=
                T2_LOGICAL_EDGES,
        )
    )

    t2_native_embedding_exists = (
        t2_native_mapping
        is not None
    )

    t2_native_embedding_verified = (
        verify_native_embedding(
            t2_native_mapping,
            undirected_edges,
            T2_LOGICAL_EDGES,
        )
        if t2_native_mapping
        is not None
        else False
    )

    if t2_native_mapping is not None:

        mapping_string = ";".join(
            (
                f"L{logical_q}"
                f"->P{t2_native_mapping[logical_q]}"
            )
            for logical_q
            in sorted(
                t2_native_mapping
            )
        )

    else:

        mapping_string = None

    # --------------------------------------------------------
    # Timing resolution
    # --------------------------------------------------------

    try:
        dt_seconds = backend.dt
    except Exception:
        dt_seconds = None

    try:
        dtm_seconds = backend.dtm
    except Exception:
        dtm_seconds = None

    dt_ns = (
        dt_seconds * 1e9
        if dt_seconds is not None
        else None
    )

    dtm_ns = (
        dtm_seconds * 1e9
        if dtm_seconds is not None
        else None
    )

    # --------------------------------------------------------
    # Main summary
    # --------------------------------------------------------

    summary_rows.append({

        "snapshot_utc":
            snapshot_utc.isoformat(),

        "backend":
            name,

        "num_qubits":
            num_qubits,

        "operational":
            operational,

        "status_msg":
            status_msg,

        "pending_jobs":
            pending_jobs,

        "backend_version":
            backend_version,

        "online_date":
            iso_or_none(
                online_date
            ),

        "processor_family":
            processor[
                "processor_family"
            ],

        "processor_revision":
            processor[
                "processor_revision"
            ],

        "processor_segment":
            processor[
                "processor_segment"
            ],

        "num_directed_edges":
            len(
                directed_edges
            ),

        "num_undirected_edges":
            metrics[
                "num_undirected_edges"
            ],

        "average_degree":
            metrics[
                "average_degree"
            ],

        "max_degree":
            metrics[
                "max_degree"
            ],

        "largest_component_size":
            metrics[
                "largest_component_size"
            ],

        "component_has_at_least_6_qubits":
            metrics[
                "component_has_at_least_6_qubits"
            ],

        "has_simple_6q_chain":
            has_simple_6q_chain,

        "six_qubit_chain_verified":
            six_qubit_chain_verified,

        "example_6q_chain":
            (
                "-".join(
                    map(
                        str,
                        six_qubit_chain,
                    )
                )
                if six_qubit_chain
                is not None
                else None
            ),

        "num_hardware_triangles":
            num_hardware_triangles,

        "hardware_has_triangle":
            hardware_has_triangle,

        "t2_native_embedding_exists":
            t2_native_embedding_exists,

        "t2_native_embedding_verified":
            t2_native_embedding_verified,

        "example_t2_native_mapping":
            mapping_string,

        "supports_measure":
            supports_measure,

        "supports_reset":
            supports_reset,

        "supports_delay":
            supports_delay,

        "supports_if_else":
            supports_if_else,

        "supports_while_loop":
            supports_while_loop,

        "dt_ns":
            dt_ns,

        "dtm_ns":
            dtm_ns,

        "basis_gates":
            ",".join(
                basis_gates
            ),

        "single_qubit_instructions":
            ",".join(
                single_qubit_instructions
            ),

        "two_qubit_instructions":
            ",".join(
                two_qubit_instructions
            ),

        "supported_features":
            ",".join(
                map(
                    str,
                    supported_features,
                )
            ),
    })

    # --------------------------------------------------------
    # Coupling edges
    # --------------------------------------------------------

    for u, v in directed_edges:

        edge_rows.append({

            "backend":
                name,

            "source":
                u,

            "target":
                v,

            "undirected_pair":
                (
                    f"{min(u, v)}-"
                    f"{max(u, v)}"
                ),
        })

    # --------------------------------------------------------
    # Chain output
    # --------------------------------------------------------

    if six_qubit_chain is not None:

        chain_rows.append({

            "backend":
                name,

            "physical_q0":
                six_qubit_chain[0],

            "physical_q1":
                six_qubit_chain[1],

            "physical_q2":
                six_qubit_chain[2],

            "physical_q3":
                six_qubit_chain[3],

            "physical_q4":
                six_qubit_chain[4],

            "physical_q5":
                six_qubit_chain[5],

            "chain_string":
                "-".join(
                    map(
                        str,
                        six_qubit_chain,
                    )
                ),

            "verified":
                six_qubit_chain_verified,
        })

    # --------------------------------------------------------
    # Triangle output
    # --------------------------------------------------------

    for triangle in hardware_triangles:

        triangle_rows.append({

            "backend":
                name,

            "q0":
                triangle[0],

            "q1":
                triangle[1],

            "q2":
                triangle[2],

            "triangle_string":
                "-".join(
                    map(
                        str,
                        triangle,
                    )
                ),
        })

    # --------------------------------------------------------
    # T2 embedding output
    # --------------------------------------------------------

    embedding_rows.append({

        "backend":
            name,

        "native_embedding_exists":
            t2_native_embedding_exists,

        "native_embedding_verified":
            t2_native_embedding_verified,

        "logical_edges":
            str(
                T2_LOGICAL_EDGES
            ),

        "mapping":
            mapping_string,
    })

    # --------------------------------------------------------
    # Detailed JSON
    # --------------------------------------------------------

    topology_records.append({

        "backend":
            name,

        "num_qubits":
            num_qubits,

        "processor":
            processor,

        "directed_edges":
            directed_edges,

        "undirected_edges":
            undirected_edges,

        "degrees":
            metrics[
                "degrees"
            ],

        "connected_components":
            metrics[
                "components"
            ],

        "six_qubit_chain": {

            "exists":
                has_simple_6q_chain,

            "verified":
                six_qubit_chain_verified,

            "example":
                six_qubit_chain,
        },

        "hardware_triangles": {

            "count":
                num_hardware_triangles,

            "triangles":
                hardware_triangles,
        },

        "t2_native_embedding": {

            "logical_edges":
                T2_LOGICAL_EDGES,

            "exists":
                t2_native_embedding_exists,

            "verified":
                t2_native_embedding_verified,

            "mapping":
                t2_native_mapping,
        },
    })

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        f"  Qubits:                    "
        f"{num_qubits}"
    )

    print(
        f"  Operational:               "
        f"{operational}"
    )

    print(
        f"  Pending jobs:              "
        f"{pending_jobs}"
    )

    print(
        f"  Processor family:          "
        f"{processor['processor_family']}"
    )

    print(
        f"  Undirected edges:          "
        f"{len(undirected_edges)}"
    )

    print(
        f"  Average degree:            "
        f"{metrics['average_degree']:.3f}"
    )

    print(
        f"  Maximum degree:            "
        f"{metrics['max_degree']}"
    )

    print(
        f"  Largest component:         "
        f"{metrics['largest_component_size']}"
    )

    print(
        f"  Simple 6Q chain exists:    "
        f"{has_simple_6q_chain}"
    )

    print(
        f"  6Q chain verified:         "
        f"{six_qubit_chain_verified}"
    )

    print(
        f"  Example physical 6Q chain: "
        f"{six_qubit_chain}"
    )

    print(
        f"  Hardware triangles:        "
        f"{num_hardware_triangles}"
    )

    print(
        f"  Hardware has triangle:     "
        f"{hardware_has_triangle}"
    )

    print(
        f"  Exact native T2 embedding: "
        f"{t2_native_embedding_exists}"
    )

    print(
        f"  T2 embedding verified:     "
        f"{t2_native_embedding_verified}"
    )

    print(
        f"  Example T2 mapping:        "
        f"{mapping_string}"
    )

    print(
        f"  Two-Q instructions:        "
        f"{two_qubit_instructions}"
    )

    print()


# ============================================================
# DataFrames
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

edge_df = pd.DataFrame(
    edge_rows
)

chain_df = pd.DataFrame(
    chain_rows
)

triangle_df = pd.DataFrame(
    triangle_rows
)

embedding_df = pd.DataFrame(
    embedding_rows
)


summary_df = summary_df.sort_values(

    by=[
        "t2_native_embedding_exists",
        "has_simple_6q_chain",
        "operational",
        "pending_jobs",
    ],

    ascending=[
        False,
        False,
        False,
        True,
    ],

    na_position="last",

).reset_index(
    drop=True
)


# ============================================================
# Save result files
# ============================================================

summary_path = (
    RESULTS_DIR
    / "08_01_backend_summary.csv"
)

edges_path = (
    RESULTS_DIR
    / "08_01_coupling_edges.csv"
)

chains_path = (
    RESULTS_DIR
    / "08_01_example_6q_chains.csv"
)

triangles_path = (
    RESULTS_DIR
    / "08_01_hardware_triangles.csv"
)

embedding_path = (
    RESULTS_DIR
    / "08_01_t2_native_embedding.csv"
)

json_path = (
    RESULTS_DIR
    / "08_01_backend_topologies.json"
)

environment_path = (
    RESULTS_DIR
    / "08_01_environment.json"
)


summary_df.to_csv(
    summary_path,
    index=False,
)

edge_df.to_csv(
    edges_path,
    index=False,
)

chain_df.to_csv(
    chains_path,
    index=False,
)

triangle_df.to_csv(
    triangles_path,
    index=False,
)

embedding_df.to_csv(
    embedding_path,
    index=False,
)


with open(
    json_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "snapshot_utc":
                snapshot_utc.isoformat(),

            "qrc_required_qubits":
                N_QRC_QUBITS,

            "t2_logical_edges":
                T2_LOGICAL_EDGES,

            "backends":
                topology_records,
        },

        file,

        indent=2,
        ensure_ascii=False,
        default=str,
    )


try:

    environment[
        "active_instance_at_end"
    ] = str(
        service.active_instance()
    )

except Exception:

    environment[
        "active_instance_at_end"
    ] = None


with open(
    environment_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        environment,
        file,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# Visualizations
# ============================================================

print("-" * 78)
print("Generating hardware topology plots...")


# ------------------------------------------------------------
# Chain plots
# ------------------------------------------------------------

for record in topology_records:

    chain = (
        record[
            "six_qubit_chain"
        ][
            "example"
        ]
    )

    if chain is None:
        continue

    output_path = (
        RESULTS_DIR
        / (
            f"08_01_chain_"
            f"{record['backend']}.png"
        )
    )

    if plot_example_chain(
        record["backend"],
        chain,
        output_path,
    ):

        print(
            f"  Saved chain plot: "
            f"{output_path}"
        )


# ------------------------------------------------------------
# Official Qiskit maps
# ------------------------------------------------------------

print()
print(
    "Attempting Qiskit physical coupling maps..."
)

qiskit_plot_success = {}

try:

    import matplotlib.pyplot as plt
    from qiskit.visualization import plot_gate_map

    for backend in backends:

        try:

            fig = plot_gate_map(
                backend,
                plot_directed=False,
                label_qubits=True,
            )

            output_path = (
                RESULTS_DIR
                / (
                    f"08_01_topology_"
                    f"{backend.name}.png"
                )
            )

            fig.savefig(
                output_path,
                dpi=180,
                bbox_inches="tight",
            )

            plt.close(fig)

            qiskit_plot_success[
                backend.name
            ] = True

            print(
                f"  Saved IBM/Qiskit map: "
                f"{output_path}"
            )

        except Exception as exc:

            qiskit_plot_success[
                backend.name
            ] = False

            print(
                f"  Qiskit map unavailable "
                f"for {backend.name}: {exc}"
            )

except Exception as exc:

    print(
        f"Qiskit plotting unavailable: "
        f"{exc}"
    )

    qiskit_plot_success = {
        backend.name: False
        for backend in backends
    }


# ------------------------------------------------------------
# Fallback maps
# ------------------------------------------------------------

print()
print(
    "Generating fallback connectivity plots "
    "where necessary..."
)

for record in topology_records:

    backend_name = record[
        "backend"
    ]

    if qiskit_plot_success.get(
        backend_name,
        False,
    ):
        continue

    output_path = (
        RESULTS_DIR
        / (
            f"08_01_connectivity_"
            f"{backend_name}.png"
        )
    )

    success = plot_topology_fallback(

        backend_name=
            backend_name,

        num_qubits=
            record["num_qubits"],

        undirected_edges=
            record["undirected_edges"],

        chain=
            record[
                "six_qubit_chain"
            ]["example"],

        output_path=
            output_path,
    )

    if success:

        print(
            f"  Saved fallback graph: "
            f"{output_path}"
        )


# ============================================================
# Final report
# ============================================================

print()

print("=" * 78)
print("BACKEND SUMMARY")
print("=" * 78)


columns_to_print = [

    "backend",

    "num_qubits",

    "operational",

    "pending_jobs",

    "processor_family",

    "num_undirected_edges",

    "average_degree",

    "max_degree",

    "has_simple_6q_chain",

    "example_6q_chain",

    "num_hardware_triangles",

    "hardware_has_triangle",

    "t2_native_embedding_exists",

    "t2_native_embedding_verified",

    "example_t2_native_mapping",

    "two_qubit_instructions",
]


print(
    summary_df[
        columns_to_print
    ].to_string(
        index=False
    )
)


print()

print("=" * 78)
print("FILES SAVED")
print("=" * 78)

print(summary_path)
print(edges_path)
print(chains_path)
print(triangles_path)
print(embedding_path)
print(json_path)
print(environment_path)


print()

print("=" * 78)
print("TOPOLOGY INTERPRETATION")
print("=" * 78)


for _, row in summary_df.iterrows():

    print()

    print(
        f"{row['backend']}:"
    )

    print(
        f"  Simple physical 6Q chain: "
        f"{row['has_simple_6q_chain']}"
    )

    print(
        f"  Hardware triangles: "
        f"{row['num_hardware_triangles']}"
    )

    print(
        f"  Exact native T2 embedding: "
        f"{row['t2_native_embedding_exists']}"
    )

    if row[
        "t2_native_embedding_exists"
    ]:

        print(
            "  RESULT: T2 can be placed "
            "without routing."
        )

    else:

        print(
            "  RESULT: T2 cannot be placed "
            "natively on this coupling graph."
        )

        print(
            "          Routing, SWAPs, or a "
            "hardware-adapted topology will "
            "be required."
        )


print()

print(
    "Step 8.1 topology profiling finished."
)

print(
    "Interpret the topology results before "
    "moving to Step 8.2."
)