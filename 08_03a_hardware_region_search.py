"""
Week 8 - Step 8.3A
Hardware-native six-qubit QRC region search.

Inputs
------
results/08_02a_qubit_calibration.csv
results/08_02b_cz_edge_calibration.csv
results/08_01_coupling_edges.csv   [optional consistency check]

Purpose
-------
For every backend and every hardware-native QRC candidate H0-H4:

1. Enumerate all native six-qubit embeddings.
2. Attach qubit-level calibration:
      - T1
      - T2
      - readout error
3. Attach CZ-edge calibration:
      - CZ error
      - CZ duration
4. Remove pathological/incomplete embeddings.
5. Determine the Pareto frontier.
6. Apply a transparent hierarchical shortlist:

      Pareto frontier
          ->
      10 lowest worst-CZ-error embeddings
          ->
      5 highest global minimum-T2 embeddings
          ->
      3 highest memory minimum-T2 embeddings

7. Rank the final three by:
      - highest minimum T1
      - lowest maximum readout error
      - lowest mean CZ error
8. Save all finalists and one selected representative.
9. Plot the selected representative for every backend/topology.

Important
---------
This script DOES NOT:
- optimize J couplings,
- calculate memory capacity,
- calculate forecasting RMSE,
- transpile the QRC,
- add SWAP routing.

Those are later experiments.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("results")

QUBIT_FILE = (
    RESULTS_DIR
    / "08_02a_qubit_calibration.csv"
)

CZ_FILE = (
    RESULTS_DIR
    / "08_02b_cz_edge_calibration.csv"
)

COUPLING_FILE = (
    RESULTS_DIR
    / "08_01_coupling_edges.csv"
)


# ============================================================
# Logical qubit roles
# ============================================================

LOGICAL_ROLES = {
    0: "C_t",
    1: "D",
    2: "P_t",
    3: "H",
    4: "M1",
    5: "M2",
}

INJECTION_QUBITS = {
    0,
    1,
    2,
    3,
}

MEMORY_QUBITS = {
    4,
    5,
}


# ============================================================
# Hardware-native QRC candidates
# ============================================================

CANDIDATES = {

    # --------------------------------------------------------
    # H0:
    #
    # C - D - P - H - M1 - M2
    # --------------------------------------------------------

    "H0": {

        "description":
            "Original native chain: "
            "C-D-P-H-M1-M2",

        "edges": [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ],
    },

    # --------------------------------------------------------
    # H1:
    #
    # C - D - P - H
    #             / \
    #           M1   M2
    # --------------------------------------------------------

    "H1": {

        "description":
            "Holiday fan-out to both memory qubits",

        "edges": [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
        ],
    },

    # --------------------------------------------------------
    # H2:
    #
    # C - D - P - H
    #         |
    #        M1 - M2
    # --------------------------------------------------------

    "H2": {

        "description":
            "P_t branch into interacting memory pair",

        "edges": [
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (4, 5),
        ],
    },

    # --------------------------------------------------------
    # H3:
    #
    # H - P - D - C - M1 - M2
    #
    # C_t is directly adjacent to memory.
    # --------------------------------------------------------

    "H3": {

        "description":
            "Task-aware chain with C_t "
            "at memory boundary",

        "edges": [
            (3, 2),
            (2, 1),
            (1, 0),
            (0, 4),
            (4, 5),
        ],
    },

    # --------------------------------------------------------
    # H4:
    #
    # P - D - C - M1 - M2
    #             |
    #             H
    #
    # M1 receives directly from:
    #
    # C_t
    # H
    # M2
    # --------------------------------------------------------

    "H4": {

        "description":
            "Memory hub: C_t and holiday "
            "directly feed M1",

        "edges": [
            (2, 1),
            (1, 0),
            (0, 4),
            (3, 4),
            (4, 5),
        ],
    },
}


# ============================================================
# Selection configuration
# ============================================================

N_STAGE_CZ = 10
N_STAGE_T2 = 5
N_STAGE_MEMORY_T2 = 3


# ============================================================
# General helpers
# ============================================================

def canonical_edge(a, b):

    return tuple(
        sorted(
            (
                int(a),
                int(b),
            )
        )
    )


def adjacency_from_edges(
    nodes,
    edges,
):

    adjacency = {
        node: set()
        for node in nodes
    }

    for a, b in edges:

        adjacency[a].add(b)
        adjacency[b].add(a)

    return adjacency


def finite_array(values):

    return np.asarray(
        [
            value
            for value in values
            if np.isfinite(value)
        ],
        dtype=float,
    )


def safe_mean(values):

    values = finite_array(
        values
    )

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def safe_min(values):

    values = finite_array(
        values
    )

    if len(values) == 0:
        return np.nan

    return float(
        np.min(values)
    )


def safe_max(values):

    values = finite_array(
        values
    )

    if len(values) == 0:
        return np.nan

    return float(
        np.max(values)
    )


# ============================================================
# Load inputs
# ============================================================

if not QUBIT_FILE.exists():

    raise FileNotFoundError(
        f"Missing:\n{QUBIT_FILE}"
    )


if not CZ_FILE.exists():

    raise FileNotFoundError(
        f"Missing:\n{CZ_FILE}"
    )


qubit_df = pd.read_csv(
    QUBIT_FILE
)

cz_df = pd.read_csv(
    CZ_FILE
)


# ============================================================
# Required columns
# ============================================================

required_qubit_columns = {

    "backend",
    "physical_qubit",
    "t1_us",
    "t2_us",
    "readout_error_percent",
}


required_cz_columns = {

    "backend",
    "q_low",
    "q_high",
    "cz_error_mean_percent",
    "cz_duration_mean_ns",
}


missing_qubit_columns = (
    required_qubit_columns
    -
    set(
        qubit_df.columns
    )
)


missing_cz_columns = (
    required_cz_columns
    -
    set(
        cz_df.columns
    )
)


if missing_qubit_columns:

    raise RuntimeError(
        "Missing qubit columns: "
        f"{sorted(missing_qubit_columns)}"
    )


if missing_cz_columns:

    raise RuntimeError(
        "Missing CZ columns: "
        f"{sorted(missing_cz_columns)}"
    )


# ============================================================
# Run metadata
# ============================================================

analysis_utc = datetime.now(
    timezone.utc
)


print("=" * 92)
print("WEEK 8 - STEP 8.3A")
print("HARDWARE-NATIVE SIX-QUBIT REGION SEARCH")
print("=" * 92)

print()

print(
    f"Analysis UTC: "
    f"{analysis_utc.isoformat()}"
)

print()


backends = sorted(
    set(
        qubit_df[
            "backend"
        ]
    )
    &
    set(
        cz_df[
            "backend"
        ]
    )
)


print(
    "Backends:"
)

for backend in backends:

    print(
        f"  {backend}"
    )

print()


# ============================================================
# Calibration lookup dictionaries
# ============================================================

def build_node_lookup(
    backend_name,
):

    subset = qubit_df[
        qubit_df[
            "backend"
        ]
        ==
        backend_name
    ]


    lookup = {}


    for _, row in subset.iterrows():

        q = int(
            row[
                "physical_qubit"
            ]
        )


        lookup[q] = {

            "t1_us":
                (
                    float(
                        row[
                            "t1_us"
                        ]
                    )
                    if pd.notna(
                        row[
                            "t1_us"
                        ]
                    )
                    else np.nan
                ),

            "t2_us":
                (
                    float(
                        row[
                            "t2_us"
                        ]
                    )
                    if pd.notna(
                        row[
                            "t2_us"
                        ]
                    )
                    else np.nan
                ),

            "readout_error_percent":
                (
                    float(
                        row[
                            "readout_error_percent"
                        ]
                    )
                    if pd.notna(
                        row[
                            "readout_error_percent"
                        ]
                    )
                    else np.nan
                ),
        }


    return lookup


def build_edge_lookup(
    backend_name,
):

    subset = cz_df[
        cz_df[
            "backend"
        ]
        ==
        backend_name
    ]


    lookup = {}


    for _, row in subset.iterrows():

        edge = canonical_edge(
            row[
                "q_low"
            ],
            row[
                "q_high"
            ],
        )


        lookup[edge] = {

            "cz_error_percent":
                (
                    float(
                        row[
                            "cz_error_mean_percent"
                        ]
                    )
                    if pd.notna(
                        row[
                            "cz_error_mean_percent"
                        ]
                    )
                    else np.nan
                ),

            "cz_duration_ns":
                (
                    float(
                        row[
                            "cz_duration_mean_ns"
                        ]
                    )
                    if pd.notna(
                        row[
                            "cz_duration_mean_ns"
                        ]
                    )
                    else np.nan
                ),
        }


    return lookup


# ============================================================
# Optional 8.1 topology consistency
# ============================================================

def coupling_edges_from_08_01(
    backend_name,
):

    if not COUPLING_FILE.exists():

        return None


    coupling_df = pd.read_csv(
        COUPLING_FILE
    )


    subset = coupling_df[
        coupling_df[
            "backend"
        ]
        ==
        backend_name
    ]


    if len(subset) == 0:

        return None


    edges = set()


    for _, row in subset.iterrows():

        edges.add(
            canonical_edge(
                row[
                    "source"
                ],
                row[
                    "target"
                ],
            )
        )


    return edges


# ============================================================
# Native embedding enumeration
# ============================================================

def find_all_embeddings(
    physical_nodes,
    physical_edges,
    logical_edges,
):
    """
    Find all injective logical -> physical mappings
    such that every required logical edge corresponds
    to a native physical CZ edge.

    Extra physical edges are allowed.
    """

    physical_nodes = sorted(
        physical_nodes
    )

    physical_edge_set = set(
        physical_edges
    )


    physical_adj = adjacency_from_edges(
        physical_nodes,
        physical_edges,
    )


    logical_nodes = sorted(
        {
            q
            for edge in logical_edges
            for q in edge
        }
    )


    logical_adj = adjacency_from_edges(
        logical_nodes,
        logical_edges,
    )


    # --------------------------------------------------------
    # Degree-compatible physical candidates
    # --------------------------------------------------------

    degree_candidates = {}


    for logical_q in logical_nodes:

        required_degree = len(
            logical_adj[
                logical_q
            ]
        )


        degree_candidates[
            logical_q
        ] = [

            physical_q

            for physical_q
            in physical_nodes

            if len(
                physical_adj[
                    physical_q
                ]
            )
            >=
            required_degree
        ]


    mappings = []

    mapping = {}

    used_physical = set()


    def choose_next_logical():

        unmapped = [

            q
            for q in logical_nodes

            if q not in mapping
        ]


        return max(

            unmapped,

            key=lambda q: (

                sum(
                    1
                    for neighbour
                    in logical_adj[q]

                    if neighbour
                    in mapping
                ),

                len(
                    logical_adj[q]
                ),

                -q,
            ),
        )


    def backtrack():

        if len(mapping) == len(
            logical_nodes
        ):

            mappings.append(
                mapping.copy()
            )

            return


        logical_q = (
            choose_next_logical()
        )


        mapped_neighbours = [

            neighbour

            for neighbour
            in logical_adj[
                logical_q
            ]

            if neighbour
            in mapping
        ]


        if mapped_neighbours:

            candidate_sets = [

                physical_adj[
                    mapping[
                        neighbour
                    ]
                ]

                for neighbour
                in mapped_neighbours
            ]


            candidates = set(
                candidate_sets[0]
            )


            for candidate_set in (
                candidate_sets[1:]
            ):

                candidates &= (
                    candidate_set
                )


            candidates &= set(
                degree_candidates[
                    logical_q
                ]
            )


            candidates = sorted(
                candidates
            )


        else:

            candidates = (
                degree_candidates[
                    logical_q
                ]
            )


        for physical_q in candidates:

            if physical_q in used_physical:
                continue


            valid = True


            for logical_neighbour in (
                mapped_neighbours
            ):

                physical_neighbour = (
                    mapping[
                        logical_neighbour
                    ]
                )


                physical_edge = (
                    canonical_edge(
                        physical_q,
                        physical_neighbour,
                    )
                )


                if (
                    physical_edge
                    not in
                    physical_edge_set
                ):

                    valid = False
                    break


            if not valid:
                continue


            mapping[
                logical_q
            ] = physical_q

            used_physical.add(
                physical_q
            )


            backtrack()


            used_physical.remove(
                physical_q
            )

            del mapping[
                logical_q
            ]


    backtrack()

    return mappings


# ============================================================
# Evaluate one physical embedding
# ============================================================

def evaluate_embedding(
    backend_name,
    candidate_name,
    candidate_description,
    logical_edges,
    mapping,
    node_lookup,
    edge_lookup,
):

    # --------------------------------------------------------
    # Node metrics
    # --------------------------------------------------------

    t1_values = []
    t2_values = []

    readout_values = []

    injection_t1 = []
    injection_t2 = []

    memory_t1 = []
    memory_t2 = []

    memory_readout = []

    complete_nodes = True


    for logical_q in range(6):

        physical_q = (
            mapping[
                logical_q
            ]
        )


        props = node_lookup.get(
            physical_q
        )


        if props is None:

            complete_nodes = False

            t1 = np.nan
            t2 = np.nan
            readout = np.nan


        else:

            t1 = props[
                "t1_us"
            ]

            t2 = props[
                "t2_us"
            ]

            readout = props[
                "readout_error_percent"
            ]


        if not (
            np.isfinite(t1)
            and
            np.isfinite(t2)
            and
            np.isfinite(readout)
        ):

            complete_nodes = False


        t1_values.append(
            t1
        )

        t2_values.append(
            t2
        )

        readout_values.append(
            readout
        )


        if logical_q in (
            INJECTION_QUBITS
        ):

            injection_t1.append(
                t1
            )

            injection_t2.append(
                t2
            )


        if logical_q in (
            MEMORY_QUBITS
        ):

            memory_t1.append(
                t1
            )

            memory_t2.append(
                t2
            )

            memory_readout.append(
                readout
            )


    # --------------------------------------------------------
    # Edge metrics
    # --------------------------------------------------------

    cz_errors = []
    cz_durations = []

    input_memory_errors = []
    memory_memory_errors = []

    physical_edge_strings = []

    complete_edges = True


    for logical_u, logical_v in (
        logical_edges
    ):

        physical_u = mapping[
            logical_u
        ]

        physical_v = mapping[
            logical_v
        ]


        physical_edge = canonical_edge(
            physical_u,
            physical_v,
        )


        physical_edge_strings.append(

            f"L{logical_u}-L{logical_v}:"
            f"P{physical_edge[0]}-"
            f"P{physical_edge[1]}"
        )


        props = edge_lookup.get(
            physical_edge
        )


        if props is None:

            complete_edges = False

            error = np.nan
            duration = np.nan


        else:

            error = props[
                "cz_error_percent"
            ]

            duration = props[
                "cz_duration_ns"
            ]


        if not (
            np.isfinite(error)
            and
            np.isfinite(duration)
        ):

            complete_edges = False


        cz_errors.append(
            error
        )

        cz_durations.append(
            duration
        )


        # ----------------------------------------------------
        # Injection-memory boundary
        # ----------------------------------------------------

        is_input_memory = (

            (
                logical_u
                in INJECTION_QUBITS

                and

                logical_v
                in MEMORY_QUBITS
            )

            or

            (
                logical_v
                in INJECTION_QUBITS

                and

                logical_u
                in MEMORY_QUBITS
            )
        )


        if is_input_memory:

            input_memory_errors.append(
                error
            )


        # ----------------------------------------------------
        # Memory-memory edge
        # ----------------------------------------------------

        if (
            logical_u
            in MEMORY_QUBITS

            and

            logical_v
            in MEMORY_QUBITS
        ):

            memory_memory_errors.append(
                error
            )


    # --------------------------------------------------------
    # Hard pathological filters only
    # --------------------------------------------------------

    pathological_cz = any(

        np.isfinite(error)
        and
        error >= 99.0

        for error in cz_errors
    )


    pathological_readout = any(

        np.isfinite(error)
        and
        error >= 50.0

        for error in readout_values
    )


    valid_hardware_embedding = (

        complete_nodes
        and
        complete_edges
        and
        not pathological_cz
        and
        not pathological_readout
    )


    # --------------------------------------------------------
    # Approximate one-use CZ product success
    #
    # Diagnostic only.
    # Not a full noise model.
    # --------------------------------------------------------

    if (
        complete_edges
        and
        not pathological_cz
    ):

        approximate_cz_success = float(
            np.prod(
                [
                    max(
                        0.0,
                        1.0
                        -
                        error
                        / 100.0,
                    )

                    for error
                    in cz_errors
                ]
            )
        )


    else:

        approximate_cz_success = (
            np.nan
        )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "backend":
            backend_name,

        "candidate":
            candidate_name,

        "candidate_description":
            candidate_description,

        # ----------------------------------------------------
        # Physical mapping
        # ----------------------------------------------------

        "physical_C_t":
            mapping[0],

        "physical_D":
            mapping[1],

        "physical_P_t":
            mapping[2],

        "physical_H":
            mapping[3],

        "physical_M1":
            mapping[4],

        "physical_M2":
            mapping[5],

        "mapping_string":
            ";".join(
                [
                    (
                        f"{LOGICAL_ROLES[q]}"
                        f"->P{mapping[q]}"
                    )

                    for q in range(6)
                ]
            ),

        "physical_edges":
            ";".join(
                physical_edge_strings
            ),

        # ----------------------------------------------------
        # Global qubit metrics
        # ----------------------------------------------------

        "min_t1_us":
            safe_min(
                t1_values
            ),

        "mean_t1_us":
            safe_mean(
                t1_values
            ),

        "min_t2_us":
            safe_min(
                t2_values
            ),

        "mean_t2_us":
            safe_mean(
                t2_values
            ),

        # ----------------------------------------------------
        # Injection metrics
        # ----------------------------------------------------

        "injection_min_t1_us":
            safe_min(
                injection_t1
            ),

        "injection_min_t2_us":
            safe_min(
                injection_t2
            ),

        "injection_mean_t2_us":
            safe_mean(
                injection_t2
            ),

        # ----------------------------------------------------
        # Memory metrics
        # ----------------------------------------------------

        "memory_min_t1_us":
            safe_min(
                memory_t1
            ),

        "memory_min_t2_us":
            safe_min(
                memory_t2
            ),

        "memory_mean_t2_us":
            safe_mean(
                memory_t2
            ),

        # ----------------------------------------------------
        # Readout metrics
        # ----------------------------------------------------

        "max_readout_error_percent":
            safe_max(
                readout_values
            ),

        "mean_readout_error_percent":
            safe_mean(
                readout_values
            ),

        "memory_max_readout_error_percent":
            safe_max(
                memory_readout
            ),

        # ----------------------------------------------------
        # CZ metrics
        # ----------------------------------------------------

        "max_cz_error_percent":
            safe_max(
                cz_errors
            ),

        "mean_cz_error_percent":
            safe_mean(
                cz_errors
            ),

        "sum_cz_error_percent":
            (
                float(
                    np.sum(
                        cz_errors
                    )
                )
                if complete_edges
                else np.nan
            ),

        "max_cz_duration_ns":
            safe_max(
                cz_durations
            ),

        "mean_cz_duration_ns":
            safe_mean(
                cz_durations
            ),

        # ----------------------------------------------------
        # Boundary-specific CZ
        # ----------------------------------------------------

        "input_memory_max_cz_error_percent":
            safe_max(
                input_memory_errors
            ),

        "input_memory_mean_cz_error_percent":
            safe_mean(
                input_memory_errors
            ),

        "memory_memory_cz_error_percent":
            (
                memory_memory_errors[0]

                if len(
                    memory_memory_errors
                )
                == 1

                else np.nan
            ),

        "approx_one_use_cz_success":
            approximate_cz_success,

        # ----------------------------------------------------
        # Validity
        # ----------------------------------------------------

        "complete_node_calibration":
            complete_nodes,

        "complete_edge_calibration":
            complete_edges,

        "pathological_cz":
            pathological_cz,

        "pathological_readout":
            pathological_readout,

        "valid_hardware_embedding":
            valid_hardware_embedding,
    }


# ============================================================
# Pareto frontier
# ============================================================

PARETO_OBJECTIVES = {

    "min_t1_us":
        "max",

    "min_t2_us":
        "max",

    "max_readout_error_percent":
        "min",

    "max_cz_error_percent":
        "min",
}


def pareto_mask(
    dataframe,
    objectives,
):
    """
    Return Boolean mask indicating Pareto-optimal rows.

    A row is dominated if another row is:
      - no worse in every objective
      - strictly better in at least one objective
    """

    if len(dataframe) == 0:

        return pd.Series(
            dtype=bool
        )


    columns = list(
        objectives.keys()
    )


    values = dataframe[
        columns
    ].to_numpy(
        dtype=float
    )


    transformed = (
        values.copy()
    )


    # Convert every criterion to "higher is better".
    for col_index, column in enumerate(
        columns
    ):

        if (
            objectives[
                column
            ]
            ==
            "min"
        ):

            transformed[
                :,
                col_index
            ] *= -1.0


    is_pareto = np.ones(
        len(dataframe),
        dtype=bool,
    )


    for i in range(
        len(dataframe)
    ):

        if not np.all(
            np.isfinite(
                transformed[i]
            )
        ):

            is_pareto[i] = False
            continue


        for j in range(
            len(dataframe)
        ):

            if i == j:
                continue


            if not np.all(
                np.isfinite(
                    transformed[j]
                )
            ):

                continue


            j_no_worse = np.all(
                transformed[j]
                >=
                transformed[i]
            )


            j_strictly_better = np.any(
                transformed[j]
                >
                transformed[i]
            )


            if (
                j_no_worse
                and
                j_strictly_better
            ):

                is_pareto[i] = False
                break


    return pd.Series(
        is_pareto,
        index=dataframe.index,
    )


# ============================================================
# Hierarchical selection
# ============================================================

def hierarchical_shortlist(
    pareto_df,
):
    """
    Hierarchical transparent selection.

    Stage 1
    -------
    Keep 10 smallest maximum CZ errors.

    Stage 2
    -------
    From Stage 1, keep 5 largest global minimum T2.

    Stage 3
    -------
    From Stage 2, keep 3 largest memory minimum T2.

    Final ranking of those three
    ---------------------------
    1. highest minimum T1
    2. lowest maximum readout error
    3. lowest mean CZ error
    4. highest memory mean T2
    """

    if len(pareto_df) == 0:

        return (
            pareto_df.copy(),
            pareto_df.copy(),
            pareto_df.copy(),
            pareto_df.copy(),
        )


    # ========================================================
    # Stage 1:
    # lowest worst CZ error
    # ========================================================

    stage_cz = (
        pareto_df
        .sort_values(

            by=[
                "max_cz_error_percent",
                "mean_cz_error_percent",
            ],

            ascending=[
                True,
                True,
            ],
        )
        .head(
            N_STAGE_CZ
        )
        .copy()
    )


    stage_cz[
        "selection_stage"
    ] = (
        "CZ_TOP10"
    )


    # ========================================================
    # Stage 2:
    # highest global minimum T2
    # ========================================================

    stage_t2 = (
        stage_cz
        .sort_values(

            by=[
                "min_t2_us",
                "memory_min_t2_us",
                "max_cz_error_percent",
            ],

            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(
            N_STAGE_T2
        )
        .copy()
    )


    stage_t2[
        "selection_stage"
    ] = (
        "T2_TOP5"
    )


    # ========================================================
    # Stage 3:
    # highest memory minimum T2
    # ========================================================

    stage_memory = (
        stage_t2
        .sort_values(

            by=[
                "memory_min_t2_us",
                "min_t2_us",
                "max_cz_error_percent",
            ],

            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(
            N_STAGE_MEMORY_T2
        )
        .copy()
    )


    stage_memory[
        "selection_stage"
    ] = (
        "MEMORY_T2_TOP3"
    )


    # ========================================================
    # Final ranking of the three survivors
    # ========================================================

    finalists = (
        stage_memory
        .sort_values(

            by=[
                "min_t1_us",
                "max_readout_error_percent",
                "mean_cz_error_percent",
                "memory_mean_t2_us",
            ],

            ascending=[
                False,
                True,
                True,
                False,
            ],
        )
        .copy()
        .reset_index(
            drop=True
        )
    )


    finalists[
        "final_rank"
    ] = (
        np.arange(
            1,
            len(finalists) + 1,
        )
    )


    finalists[
        "selection_stage"
    ] = (
        "FINAL_TOP3"
    )


    return (
        stage_cz,
        stage_t2,
        stage_memory,
        finalists,
    )


# ============================================================
# Plot selected physical mapping
# ============================================================

def plot_selected_mapping(
    row,
    logical_edges,
    edge_lookup,
    output_path,
):
    """
    Plot logical topology with actual physical qubit IDs.

    The layout shows logical connectivity.
    It is not intended to reproduce chip geometry.
    """

    try:

        import matplotlib.pyplot as plt
        import networkx as nx

    except ImportError:

        return False


    graph = nx.Graph()

    graph.add_nodes_from(
        range(6)
    )

    graph.add_edges_from(
        logical_edges
    )


    # --------------------------------------------------------
    # Deterministic layout
    # --------------------------------------------------------

    positions = nx.spring_layout(
        graph,
        seed=42,
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        width=2,
    )


    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=2600,
    )


    mapping = {

        0:
            int(
                row[
                    "physical_C_t"
                ]
            ),

        1:
            int(
                row[
                    "physical_D"
                ]
            ),

        2:
            int(
                row[
                    "physical_P_t"
                ]
            ),

        3:
            int(
                row[
                    "physical_H"
                ]
            ),

        4:
            int(
                row[
                    "physical_M1"
                ]
            ),

        5:
            int(
                row[
                    "physical_M2"
                ]
            ),
    }


    labels = {

        logical_q:
            (
                f"{LOGICAL_ROLES[logical_q]}\n"
                f"P{mapping[logical_q]}"
            )

        for logical_q
        in range(6)
    }


    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        ax=ax,
        font_size=9,
    )


    # --------------------------------------------------------
    # CZ-error edge labels
    # --------------------------------------------------------

    edge_labels = {}


    for logical_u, logical_v in (
        logical_edges
    ):

        physical_edge = (
            canonical_edge(
                mapping[
                    logical_u
                ],
                mapping[
                    logical_v
                ],
            )
        )


        props = edge_lookup.get(
            physical_edge
        )


        if props is not None:

            edge_labels[
                (
                    logical_u,
                    logical_v,
                )
            ] = (
                f"{props['cz_error_percent']:.3f}%"
            )


    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        ax=ax,
        font_size=8,
    )


    ax.set_title(

        f"{row['backend']} - "
        f"{row['candidate']}\n"
        f"hierarchical selected representative | "
        f"max CZ="
        f"{row['max_cz_error_percent']:.3f}% | "
        f"min T2="
        f"{row['min_t2_us']:.1f} us | "
        f"memory min T2="
        f"{row['memory_min_t2_us']:.1f} us"
    )


    ax.axis(
        "off"
    )


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    return True


# ============================================================
# Main experiment
# ============================================================

all_rows = []

summary_rows = []

stage_rows = []

finalist_rows = []

selected_rows = []


for backend_name in backends:

    print("=" * 92)
    print(backend_name)
    print("=" * 92)

    print()


    node_lookup = (
        build_node_lookup(
            backend_name
        )
    )


    edge_lookup = (
        build_edge_lookup(
            backend_name
        )
    )


    physical_nodes = sorted(
        node_lookup.keys()
    )


    physical_edges = sorted(
        edge_lookup.keys()
    )


    print(
        f"Physical qubits: "
        f"{len(physical_nodes)}"
    )

    print(
        f"Calibrated CZ edges: "
        f"{len(physical_edges)}"
    )


    # --------------------------------------------------------
    # Step 8.1 consistency check
    # --------------------------------------------------------

    topology_edges = (
        coupling_edges_from_08_01(
            backend_name
        )
    )


    if topology_edges is not None:

        missing_from_cz = (
            topology_edges
            -
            set(
                physical_edges
            )
        )


        extra_in_cz = (
            set(
                physical_edges
            )
            -
            topology_edges
        )


        print(
            f"8.1 vs 8.2B topology check: "
            f"missing={len(missing_from_cz)}, "
            f"extra={len(extra_in_cz)}"
        )


    print()


    # ========================================================
    # Each logical topology
    # ========================================================

    for candidate_name, candidate in (
        CANDIDATES.items()
    ):

        logical_edges = [

            canonical_edge(
                a,
                b,
            )

            for a, b
            in candidate[
                "edges"
            ]
        ]


        print(
            f"  Searching {candidate_name}: "
            f"{candidate['description']}"
        )


        # ----------------------------------------------------
        # Enumerate embeddings
        # ----------------------------------------------------

        embeddings = (
            find_all_embeddings(

                physical_nodes=
                    physical_nodes,

                physical_edges=
                    physical_edges,

                logical_edges=
                    logical_edges,
            )
        )


        print(
            f"    Native labelled embeddings: "
            f"{len(embeddings)}"
        )


        candidate_rows = []


        for embedding_index, mapping in (
            enumerate(
                embeddings
            )
        ):

            result = (
                evaluate_embedding(

                    backend_name=
                        backend_name,

                    candidate_name=
                        candidate_name,

                    candidate_description=
                        candidate[
                            "description"
                        ],

                    logical_edges=
                        logical_edges,

                    mapping=
                        mapping,

                    node_lookup=
                        node_lookup,

                    edge_lookup=
                        edge_lookup,
                )
            )


            result[
                "embedding_id"
            ] = (

                f"{backend_name}_"
                f"{candidate_name}_"
                f"{embedding_index:05d}"
            )


            candidate_rows.append(
                result
            )


        candidate_df = pd.DataFrame(
            candidate_rows
        )


        if len(
            candidate_df
        ) == 0:

            print(
                "    No native embedding."
            )

            continue


        # ----------------------------------------------------
        # Valid embeddings
        # ----------------------------------------------------

        valid_df = (

            candidate_df[
                candidate_df[
                    "valid_hardware_embedding"
                ]
            ]

            .copy()

            .reset_index(
                drop=True
            )
        )


        print(
            f"    Valid calibrated embeddings: "
            f"{len(valid_df)}"
        )


        if len(
            valid_df
        ) == 0:

            continue


        # ----------------------------------------------------
        # Pareto frontier
        # ----------------------------------------------------

        valid_df[
            "is_pareto"
        ] = pareto_mask(
            valid_df,
            PARETO_OBJECTIVES,
        )


        pareto_df = (

            valid_df[
                valid_df[
                    "is_pareto"
                ]
            ]

            .copy()

            .reset_index(
                drop=True
            )
        )


        print(
            f"    Pareto embeddings: "
            f"{len(pareto_df)}"
        )


        # ----------------------------------------------------
        # Hierarchical shortlist
        # ----------------------------------------------------

        (
            stage_cz,
            stage_t2,
            stage_memory,
            finalists,
        ) = hierarchical_shortlist(
            pareto_df
        )


        print(
            f"    Stage CZ      : "
            f"{len(stage_cz)}"
        )

        print(
            f"    Stage min-T2  : "
            f"{len(stage_t2)}"
        )

        print(
            f"    Stage mem-T2  : "
            f"{len(stage_memory)}"
        )

        print(
            f"    Finalists     : "
            f"{len(finalists)}"
        )


        # ----------------------------------------------------
        # Save selection stages
        # ----------------------------------------------------

        for stage_name, stage_df in [

            (
                "CZ_TOP10",
                stage_cz,
            ),

            (
                "T2_TOP5",
                stage_t2,
            ),

            (
                "MEMORY_T2_TOP3",
                stage_memory,
            ),

        ]:

            for _, row in (
                stage_df.iterrows()
            ):

                record = row.to_dict()

                record[
                    "selection_stage"
                ] = stage_name

                stage_rows.append(
                    record
                )


        # ----------------------------------------------------
        # Save finalists
        # ----------------------------------------------------

        for _, row in (
            finalists.iterrows()
        ):

            finalist_rows.append(
                row.to_dict()
            )


        # ----------------------------------------------------
        # Rank-1 selected representative
        # ----------------------------------------------------

        selected = (
            finalists.iloc[0]
            .copy()
        )


        selected[
            "selected_representative"
        ] = True


        selected_rows.append(
            selected.to_dict()
        )


        # ----------------------------------------------------
        # Add all rows to global table
        # ----------------------------------------------------

        pareto_ids = set(
            pareto_df[
                "embedding_id"
            ]
        )


        top10_ids = set(
            stage_cz[
                "embedding_id"
            ]
        )


        top5_ids = set(
            stage_t2[
                "embedding_id"
            ]
        )


        top3_ids = set(
            stage_memory[
                "embedding_id"
            ]
        )


        finalist_rank_lookup = {

            row[
                "embedding_id"
            ]:
                int(
                    row[
                        "final_rank"
                    ]
                )

            for _, row
            in finalists.iterrows()
        }


        for row in candidate_rows:

            embedding_id = (
                row[
                    "embedding_id"
                ]
            )


            row[
                "is_pareto"
            ] = (
                embedding_id
                in pareto_ids
            )


            row[
                "in_cz_top10"
            ] = (
                embedding_id
                in top10_ids
            )


            row[
                "in_t2_top5"
            ] = (
                embedding_id
                in top5_ids
            )


            row[
                "in_memory_t2_top3"
            ] = (
                embedding_id
                in top3_ids
            )


            row[
                "final_rank"
            ] = (
                finalist_rank_lookup.get(
                    embedding_id,
                    np.nan,
                )
            )


            all_rows.append(
                row
            )


        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary_rows.append({

            "backend":
                backend_name,

            "candidate":
                candidate_name,

            "description":
                candidate[
                    "description"
                ],

            "native_embeddings":
                len(
                    candidate_df
                ),

            "valid_embeddings":
                len(
                    valid_df
                ),

            "pareto_embeddings":
                len(
                    pareto_df
                ),

            "cz_top10_count":
                len(
                    stage_cz
                ),

            "t2_top5_count":
                len(
                    stage_t2
                ),

            "memory_t2_top3_count":
                len(
                    stage_memory
                ),

            "selected_mapping":
                selected[
                    "mapping_string"
                ],

            "selected_min_t1_us":
                selected[
                    "min_t1_us"
                ],

            "selected_min_t2_us":
                selected[
                    "min_t2_us"
                ],

            "selected_memory_min_t2_us":
                selected[
                    "memory_min_t2_us"
                ],

            "selected_max_readout_error_percent":
                selected[
                    "max_readout_error_percent"
                ],

            "selected_max_cz_error_percent":
                selected[
                    "max_cz_error_percent"
                ],

            "selected_mean_cz_error_percent":
                selected[
                    "mean_cz_error_percent"
                ],

            "selected_max_cz_duration_ns":
                selected[
                    "max_cz_duration_ns"
                ],
        })


        print()

        print(
            "    Selected representative:"
        )

        print(
            f"      {selected['mapping_string']}"
        )

        print(
            f"      min T1       = "
            f"{selected['min_t1_us']:.2f} us"
        )

        print(
            f"      min T2       = "
            f"{selected['min_t2_us']:.2f} us"
        )

        print(
            f"      memory minT2 = "
            f"{selected['memory_min_t2_us']:.2f} us"
        )

        print(
            f"      max readout  = "
            f"{selected['max_readout_error_percent']:.3f}%"
        )

        print(
            f"      max CZ       = "
            f"{selected['max_cz_error_percent']:.4f}%"
        )

        print(
            f"      mean CZ      = "
            f"{selected['mean_cz_error_percent']:.4f}%"
        )

        print()


# ============================================================
# DataFrames
# ============================================================

all_df = pd.DataFrame(
    all_rows
)

summary_df = pd.DataFrame(
    summary_rows
)

stages_df = pd.DataFrame(
    stage_rows
)

finalists_df = pd.DataFrame(
    finalist_rows
)

selected_df = pd.DataFrame(
    selected_rows
)


# ============================================================
# Save
# ============================================================

all_path = (
    RESULTS_DIR
    / "08_03a_all_native_embeddings.csv"
)

pareto_path = (
    RESULTS_DIR
    / "08_03a_pareto_embeddings.csv"
)

stage_path = (
    RESULTS_DIR
    / "08_03a_hierarchical_selection_stages.csv"
)

finalist_path = (
    RESULTS_DIR
    / "08_03a_final_top3_embeddings.csv"
)

selected_path = (
    RESULTS_DIR
    / "08_03a_selected_representatives.csv"
)

summary_path = (
    RESULTS_DIR
    / "08_03a_topology_summary.csv"
)

manifest_path = (
    RESULTS_DIR
    / "08_03a_candidate_manifest.json"
)


all_df.to_csv(
    all_path,
    index=False,
)


pareto_df_all = (

    all_df[
        all_df[
            "is_pareto"
        ]
    ]
    .copy()
)


pareto_df_all.to_csv(
    pareto_path,
    index=False,
)


stages_df.to_csv(
    stage_path,
    index=False,
)


finalists_df.to_csv(
    finalist_path,
    index=False,
)


selected_df.to_csv(
    selected_path,
    index=False,
)


summary_df.to_csv(
    summary_path,
    index=False,
)


with open(
    manifest_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(

        {
            "analysis_utc":
                analysis_utc.isoformat(),

            "logical_roles":
                LOGICAL_ROLES,

            "candidates":
                CANDIDATES,

            "pareto_objectives":
                PARETO_OBJECTIVES,

            "hierarchical_selection": {

                "stage_1":
                    (
                        "From Pareto frontier keep "
                        "10 embeddings with lowest "
                        "maximum CZ error."
                    ),

                "stage_2":
                    (
                        "From those 10 keep "
                        "5 embeddings with highest "
                        "global minimum T2."
                    ),

                "stage_3":
                    (
                        "From those 5 keep "
                        "3 embeddings with highest "
                        "memory minimum T2."
                    ),

                "final_ranking":
                    [
                        "highest minimum T1",
                        "lowest maximum readout error",
                        "lowest mean CZ error",
                        "highest memory mean T2",
                    ],
            },

            "notes": [

                (
                    "No weighted global hardware "
                    "score is used."
                ),

                (
                    "CZ >= 99% is treated as "
                    "pathological."
                ),

                (
                    "Readout error >= 50% is treated "
                    "as pathological."
                ),

                (
                    "H0 and H3 share the same "
                    "unlabelled chain topology but "
                    "differ in logical role placement."
                ),

                (
                    "H1 and H4 share the same "
                    "unlabelled branched topology but "
                    "differ in logical role placement."
                ),
            ],
        },

        file,
        indent=2,
        default=str,
    )


# ============================================================
# Plot selected representatives
# ============================================================

print("=" * 92)
print("GENERATING SELECTED REPRESENTATIVE PLOTS")
print("=" * 92)

print()


for _, row in (
    selected_df.iterrows()
):

    backend_name = (
        row[
            "backend"
        ]
    )

    candidate_name = (
        row[
            "candidate"
        ]
    )


    logical_edges = [

        canonical_edge(
            a,
            b,
        )

        for a, b
        in CANDIDATES[
            candidate_name
        ][
            "edges"
        ]
    ]


    edge_lookup = (
        build_edge_lookup(
            backend_name
        )
    )


    output_path = (

        RESULTS_DIR
        /
        (
            "08_03a_"
            f"{backend_name}_"
            f"{candidate_name}_"
            "selected.png"
        )
    )


    success = (
        plot_selected_mapping(

            row=
                row,

            logical_edges=
                logical_edges,

            edge_lookup=
                edge_lookup,

            output_path=
                output_path,
        )
    )


    if success:

        print(
            f"Saved: "
            f"{output_path}"
        )


# ============================================================
# Final output
# ============================================================

print()

print("=" * 92)
print("HIERARCHICAL HARDWARE SELECTION SUMMARY")
print("=" * 92)


summary_columns = [

    "backend",

    "candidate",

    "native_embeddings",

    "valid_embeddings",

    "pareto_embeddings",

    "selected_min_t1_us",

    "selected_min_t2_us",

    "selected_memory_min_t2_us",

    "selected_max_readout_error_percent",

    "selected_max_cz_error_percent",

    "selected_mean_cz_error_percent",
]


print(
    summary_df[
        summary_columns
    ].to_string(
        index=False
    )
)


print()

print("=" * 92)
print("FINAL TOP-3 PHYSICAL EMBEDDINGS")
print("=" * 92)


finalist_columns = [

    "backend",

    "candidate",

    "final_rank",

    "mapping_string",

    "min_t1_us",

    "min_t2_us",

    "memory_min_t2_us",

    "max_readout_error_percent",

    "max_cz_error_percent",

    "mean_cz_error_percent",
]


print(
    finalists_df[
        finalist_columns
    ].to_string(
        index=False
    )
)


print()

print("=" * 92)
print("FILES SAVED")
print("=" * 92)

print(all_path)
print(pareto_path)
print(stage_path)
print(finalist_path)
print(selected_path)
print(summary_path)
print(manifest_path)

print()

print(
    "Step 8.3A hierarchical hardware-region "
    "selection complete."
)

print(
    "Interpret the final top-3 physical embeddings "
    "before MC/RMSE retesting."
)