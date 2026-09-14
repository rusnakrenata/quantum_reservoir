from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile

from ibm_account import get_service


# =============================================================================
# Week 9 — Step 9.1 refresh
#
# Final H0-H4 × {Y OFF, Y ON} candidate compilation.
#
# Purpose:
#   - preserve the frozen Week-8 physical embeddings and J values,
#   - add the selected memory-only Y Hamiltonian where required,
#   - transpile the coherent reservoir core,
#   - transpile every selected grouped-measurement setting,
#   - record resources at optimization levels 0,1,2,3,
#   - keep levels 0 and 1 as the primary hardware references,
#   - use strict routing_method="none" for all native H0-H4 candidates.
#
# T2 is NOT rerun here. Its routed reference remains frozen from the original 9.1
# experiment and is retained only as the ideal/non-native comparison in the paper.
# =============================================================================


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DATA_FILE = RESULTS / "03_01_preprocessed_samples.csv"
EMBEDDING_FILE = RESULTS / "08_03a_final_top3_embeddings.csv"
J_WINNER_FILE = RESULTS / "08_03b_J_search_topology_winners.csv"

for path in [DATA_FILE, EMBEDDING_FILE, J_WINNER_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


# =============================================================================
# Frozen QRC parameters
# =============================================================================

ALPHA = 0.75
HX = 0.5
DT = 1.6
TROTTER_R = 2

OPT_LEVELS = [0, 1, 2, 3]
PRIMARY_OPT_LEVELS = [0, 1]

SEED_TRANSPILER = 42

BACKEND_NAMES = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh",
]


# =============================================================================
# Final symmetric 10-candidate table
#
# MC and washout are five-probe-seed means.
# Forecast/readout/lambda are the recalibrated training-only-CV selections.
# =============================================================================

CANDIDATES = [
    {
        "topology": "H0",
        "y_state": "OFF",
        "hy": 0.0,
        "readout": "XYZ_all",
        "ridge_lambda": 100.0,
        "cv_rmse": 3.611416,
        "validation_rmse": 5.134098,
        "mc_ch": 0.025909,
        "washout_mean": 105.4,
    },
    {
        "topology": "H0",
        "y_state": "ON",
        "hy": -0.4,
        "readout": "XZ_injection",
        "ridge_lambda": 1.0,
        "cv_rmse": 3.622098,
        "validation_rmse": 5.013169,
        "mc_ch": 0.025969,
        "washout_mean": 72.4,
    },

    {
        "topology": "H1",
        "y_state": "OFF",
        "hy": 0.0,
        "readout": "XYZ_all",
        "ridge_lambda": 10.0,
        "cv_rmse": 3.631705,
        "validation_rmse": 5.124035,
        "mc_ch": 0.020668,
        "washout_mean": 220.4,
    },
    {
        "topology": "H1",
        "y_state": "ON",
        "hy": -0.6,
        "readout": "XYZ_all",
        "ridge_lambda": 10.0,
        "cv_rmse": 3.608136,
        "validation_rmse": 5.117389,
        "mc_ch": 0.019488,
        "washout_mean": 222.8,
    },

    {
        "topology": "H2",
        "y_state": "OFF",
        "hy": 0.0,
        "readout": "XZ_injection",
        "ridge_lambda": 1.0,
        "cv_rmse": 3.616374,
        "validation_rmse": 5.384808,
        "mc_ch": 1.145518,
        "washout_mean": 105.4,
    },
    {
        "topology": "H2",
        "y_state": "ON",
        "hy": 0.6,
        "readout": "XZinj_plus_YX45",
        "ridge_lambda": 100.0,
        "cv_rmse": 3.591967,
        "validation_rmse": 5.104119,
        "mc_ch": 1.170055,
        "washout_mean": 36.0,
    },

    {
        "topology": "H3",
        "y_state": "OFF",
        "hy": 0.0,
        "readout": "XZinj_plus_YX45",
        "ridge_lambda": 100.0,
        "cv_rmse": 3.597186,
        "validation_rmse": 5.228065,
        "mc_ch": 1.075082,
        "washout_mean": 97.6,
    },
    {
        "topology": "H3",
        "y_state": "ON",
        "hy": 0.3,
        "readout": "XZinj_plus_YX45",
        "ridge_lambda": 100.0,
        "cv_rmse": 3.596648,
        "validation_rmse": 5.229340,
        "mc_ch": 1.272499,
        "washout_mean": 51.4,
    },

    {
        "topology": "H4",
        "y_state": "OFF",
        "hy": 0.0,
        "readout": "XYZ_all",
        "ridge_lambda": 100.0,
        "cv_rmse": 3.690075,
        "validation_rmse": 4.935582,
        "mc_ch": 0.920563,
        "washout_mean": 139.0,
    },
    {
        "topology": "H4",
        "y_state": "ON",
        "hy": 0.4,
        "readout": "XZinj_plus_YX45",
        "ridge_lambda": 0.1,
        "cv_rmse": 3.617839,
        "validation_rmse": 5.034390,
        "mc_ch": 1.312966,
        "washout_mean": 48.2,
    },
]

candidate_df = pd.DataFrame(CANDIDATES)

candidate_df["candidate_id"] = (
    candidate_df["topology"]
    + "_Y"
    + candidate_df["y_state"]
)

candidate_df.to_csv(
    RESULTS / "09_01y_final_10_candidates.csv",
    index=False,
)


# =============================================================================
# Logical topologies and Week-8 J columns
# =============================================================================

TOPOLOGY_EDGES = {
    "H0": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ],

    "H1": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (3, 5),
    ],

    "H2": [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4),
        (4, 5),
    ],

    "H3": [
        (2, 3),
        (1, 2),
        (0, 1),
        (0, 4),
        (4, 5),
    ],

    "H4": [
        (1, 2),
        (0, 1),
        (0, 4),
        (3, 4),
        (4, 5),
    ],
}


EDGE_TO_COLUMN = {
    "H0": {
        (0, 1): "J01",
        (1, 2): "J12",
        (2, 3): "J23",
        (3, 4): "J34",
        (4, 5): "J45",
    },

    "H1": {
        (0, 1): "J01",
        (1, 2): "J12",
        (2, 3): "J23",
        (3, 4): "J34",
        (3, 5): "J35",
    },

    "H2": {
        (0, 1): "J01",
        (1, 2): "J12",
        (2, 3): "J23",
        (2, 4): "J24",
        (4, 5): "J45",
    },

    "H3": {
        (2, 3): "J23",
        (1, 2): "J12",
        (0, 1): "J01",
        (0, 4): "J04",
        (4, 5): "J45",
    },

    "H4": {
        (1, 2): "J12",
        (0, 1): "J01",
        (0, 4): "J04",
        (3, 4): "J34",
        (4, 5): "J45",
    },
}


# =============================================================================
# Exact grouped readout settings
#
# String order is logical q0,q1,q2,q3,q4,q5.
# =============================================================================

READOUT_SETTINGS = {
    "XZ_injection": [
        "XXXXZZ",
        "ZZZZZZ",
    ],

    "XZinj_plus_YX45": [
        "XXXXYX",
        "ZZZZZZ",
    ],

    "XYZ_all": [
        "XXXXXX",
        "YYYYYY",
        "ZZZZZZ",
    ],
}


# =============================================================================
# Input encoding
# =============================================================================

def claim_mapping(z):
    return float(
        np.clip(
            float(z) / 3.0,
            -1.0,
            1.0,
        )
    )


def policy_mapping(z):
    z = float(z)
    return z / (3.0 + abs(z))


def weekday_mapping(d_sin, d_cos):
    return (
        math.atan2(
            float(d_sin),
            float(d_cos),
        )
        %
        (2.0 * math.pi)
    )


def holiday_mapping(h):
    h = int(h)

    if h not in (0, 1):
        raise ValueError(
            f"Holiday must be 0 or 1; got {h}"
        )

    return math.pi * h


def encode_f4_row(row):
    return np.array(
        [
            ALPHA
            *
            claim_mapping(
                row["C_t_z"]
            ),

            weekday_mapping(
                row["D_sin"],
                row["D_cos"],
            ),

            ALPHA
            *
            policy_mapping(
                row["P_t_z"]
            ),

            holiday_mapping(
                row[
                    "is_public_holiday_t_plus_1"
                ]
            ),
        ],
        dtype=float,
    )


# =============================================================================
# Load first chronological insurance input
# =============================================================================

df = pd.read_csv(
    DATA_FILE
)

df[
    "input_date"
] = pd.to_datetime(
    df[
        "input_date"
    ]
)

df[
    "target_date"
] = pd.to_datetime(
    df[
        "target_date"
    ]
)

df = (
    df
    .sort_values(
        "target_date"
    )
    .reset_index(
        drop=True
    )
)

first_row = df.iloc[0]
INPUT_ANGLES = encode_f4_row(
    first_row
)


first_input_audit = pd.DataFrame(
    [
        {
            "input_date":
                first_row[
                    "input_date"
                ],

            "target_date":
                first_row[
                    "target_date"
                ],

            "split":
                first_row[
                    "split"
                ],

            "C_t_z":
                first_row[
                    "C_t_z"
                ],

            "D_sin":
                first_row[
                    "D_sin"
                ],

            "D_cos":
                first_row[
                    "D_cos"
                ],

            "P_t_z":
                first_row[
                    "P_t_z"
                ],

            "H_t_plus_1":
                first_row[
                    "is_public_holiday_t_plus_1"
                ],

            "theta_C":
                INPUT_ANGLES[0],

            "theta_D":
                INPUT_ANGLES[1],

            "theta_P":
                INPUT_ANGLES[2],

            "theta_H":
                INPUT_ANGLES[3],
        }
    ]
)

first_input_audit.to_csv(
    RESULTS
    / "09_01y_first_input_audit.csv",
    index=False,
)


# =============================================================================
# Load exact Week-8 optimized J values
# =============================================================================

j_df = pd.read_csv(
    J_WINNER_FILE
)

COUPLINGS = {}

for topology in TOPOLOGY_EDGES:

    matches = j_df[
        j_df[
            "topology"
        ] == topology
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{topology}: expected exactly one J winner; "
            f"found {len(matches)}"
        )

    winner = matches.iloc[0]

    if not np.isclose(
        float(
            winner[
                "alpha"
            ]
        ),
        ALPHA,
    ):
        raise RuntimeError(
            f"{topology}: alpha mismatch"
        )

    if not np.isclose(
        float(
            winner[
                "hx"
            ]
        ),
        HX,
    ):
        raise RuntimeError(
            f"{topology}: hx mismatch"
        )

    if not np.isclose(
        float(
            winner[
                "dt"
            ]
        ),
        DT,
    ):
        raise RuntimeError(
            f"{topology}: dt mismatch"
        )

    if int(
        winner[
            "Trotter_r"
        ]
    ) != TROTTER_R:
        raise RuntimeError(
            f"{topology}: Trotter-r mismatch"
        )

    COUPLINGS[
        topology
    ] = {}

    for (
        edge,
        column,
    ) in EDGE_TO_COLUMN[
        topology
    ].items():

        value = winner[
            column
        ]

        if pd.isna(
            value
        ):
            raise RuntimeError(
                f"{topology}: missing {column}"
            )

        COUPLINGS[
            topology
        ][
            edge
        ] = float(
            value
        )


# =============================================================================
# Load retained Week-8 physical embeddings
# =============================================================================

embeddings = pd.read_csv(
    EMBEDDING_FILE
)

required_embedding_columns = [
    "backend",
    "candidate",
    "final_rank",
    "physical_C_t",
    "physical_D",
    "physical_P_t",
    "physical_H",
    "physical_M1",
    "physical_M2",
]

missing_embedding_columns = [
    column
    for column in required_embedding_columns
    if column not in embeddings.columns
]

if missing_embedding_columns:
    raise KeyError(
        "Missing embedding columns:\n"
        f"{missing_embedding_columns}\n\n"
        f"Available:\n{list(embeddings.columns)}"
    )

embeddings = embeddings[
    embeddings[
        "candidate"
    ].isin(
        list(
            TOPOLOGY_EDGES.keys()
        )
    )
].copy()

embeddings[
    "layout"
] = embeddings.apply(
    lambda row: [
        int(
            row[
                "physical_C_t"
            ]
        ),
        int(
            row[
                "physical_D"
            ]
        ),
        int(
            row[
                "physical_P_t"
            ]
        ),
        int(
            row[
                "physical_H"
            ]
        ),
        int(
            row[
                "physical_M1"
            ]
        ),
        int(
            row[
                "physical_M2"
            ]
        ),
    ],
    axis=1,
)


# =============================================================================
# Circuit construction
# =============================================================================

def build_core_circuit(
    topology,
    hy,
):

    qc = QuantumCircuit(
        6,
        name=(
            f"{topology}_hy_{hy:+.3f}"
        ),
    )

    # ---------------------------------------------------------
    # Real first F4 input
    # ---------------------------------------------------------

    for q in range(
        4
    ):
        qc.ry(
            float(
                INPUT_ANGLES[
                    q
                ]
            ),
            q,
        )

    # ---------------------------------------------------------
    # First-order product formula:
    #
    # [ ZZ sector -> X sector -> optional memory-Y sector ]^r
    # ---------------------------------------------------------

    for _ in range(
        TROTTER_R
    ):

        # ZZ sector
        for (
            i,
            j,
        ) in TOPOLOGY_EDGES[
            topology
        ]:

            Jij = COUPLINGS[
                topology
            ][
                (
                    i,
                    j,
                )
            ]

            theta_zz = (
                2.0
                *
                Jij
                *
                DT
                /
                TROTTER_R
            )

            qc.rzz(
                theta_zz,
                i,
                j,
            )

        # X field on all qubits
        theta_x = (
            2.0
            *
            HX
            *
            DT
            /
            TROTTER_R
        )

        for q in range(
            6
        ):
            qc.rx(
                theta_x,
                q,
            )

        # Optional Y field only on memory qubits q4,q5
        if not np.isclose(
            hy,
            0.0,
        ):

            theta_y = (
                2.0
                *
                hy
                *
                DT
                /
                TROTTER_R
            )

            qc.ry(
                theta_y,
                4,
            )

            qc.ry(
                theta_y,
                5,
            )

    return qc


def build_measurement_circuit(
    core,
    setting,
):

    if len(
        setting
    ) != 6:
        raise ValueError(
            f"Expected a six-character setting; got {setting}"
        )

    qc = QuantumCircuit(
        6,
        6,
        name=(
            f"{core.name}_{setting}"
        ),
    )

    qc.compose(
        core,
        qubits=range(
            6
        ),
        inplace=True,
    )

    # Basis rotation followed by computational-basis measurement.
    for q, basis in enumerate(
        setting
    ):

        if basis == "X":
            qc.h(
                q
            )

        elif basis == "Y":
            qc.sdg(
                q
            )
            qc.h(
                q
            )

        elif basis == "Z":
            pass

        else:
            raise ValueError(
                f"Unsupported basis {basis!r} "
                f"in setting {setting}"
            )

    for q in range(
        6
    ):
        qc.measure(
            q,
            q,
        )

    return qc


# =============================================================================
# Resource accounting
# =============================================================================

def count_resources(
    circuit,
):

    operation_counts = {
        str(
            name
        ):
            int(
                count
            )
        for (
            name,
            count,
        ) in circuit.count_ops().items()
    }

    n_1q = 0
    n_2q = 0
    n_measure = 0

    excluded_1q = {
        "measure",
        "reset",
        "delay",
        "barrier",
    }

    for instruction in circuit.data:

        operation = (
            instruction.operation
        )

        name = (
            operation.name
        )

        nq = len(
            instruction.qubits
        )

        if name == "measure":
            n_measure += 1

        elif (
            nq == 1
            and
            name not in excluded_1q
        ):
            n_1q += 1

        elif nq == 2:
            n_2q += 1

    try:
        depth_1q = int(
            circuit.depth(
                filter_function=lambda inst:
                    (
                        len(
                            inst.qubits
                        )
                        == 1
                        and
                        inst.operation.name
                        not in excluded_1q
                    )
            )
        )

        depth_2q = int(
            circuit.depth(
                filter_function=lambda inst:
                    len(
                        inst.qubits
                    )
                    == 2
            )
        )

    except TypeError:
        # Compatibility fallback for an older tuple-style filter API.
        depth_1q = int(
            circuit.depth(
                filter_function=lambda inst:
                    (
                        len(
                            inst[1]
                        )
                        == 1
                        and
                        inst[0].name
                        not in excluded_1q
                    )
            )
        )

        depth_2q = int(
            circuit.depth(
                filter_function=lambda inst:
                    len(
                        inst[1]
                    )
                    == 2
            )
        )

    return {
        "depth":
            int(
                circuit.depth()
            ),

        "depth_1q":
            depth_1q,

        "depth_2q":
            depth_2q,

        "size":
            int(
                circuit.size()
            ),

        "n_1q":
            int(
                n_1q
            ),

        "n_2q":
            int(
                n_2q
            ),

        "n_measure":
            int(
                n_measure
            ),

        "n_cz":
            int(
                operation_counts.get(
                    "cz",
                    0,
                )
            ),

        "n_swap":
            int(
                operation_counts.get(
                    "swap",
                    0,
                )
            ),

        "operations":
            json.dumps(
                operation_counts,
                sort_keys=True,
            ),
    }


# =============================================================================
# Logical baseline
# =============================================================================

logical_rows = []

for candidate in CANDIDATES:

    topology = candidate[
        "topology"
    ]

    y_state = candidate[
        "y_state"
    ]

    hy = candidate[
        "hy"
    ]

    readout = candidate[
        "readout"
    ]

    candidate_id = (
        f"{topology}_Y{y_state}"
    )

    core = build_core_circuit(
        topology,
        hy,
    )

    core_resources = count_resources(
        core
    )

    logical_rows.append(
        {
            "candidate_id":
                candidate_id,

            "topology":
                topology,

            "y_state":
                y_state,

            "hy":
                hy,

            "readout":
                readout,

            "circuit_type":
                "core",

            "measurement_setting":
                None,

            **core_resources,
        }
    )

    for setting in READOUT_SETTINGS[
        readout
    ]:

        measurement_circuit = (
            build_measurement_circuit(
                core,
                setting,
            )
        )

        measurement_resources = (
            count_resources(
                measurement_circuit
            )
        )

        logical_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "topology":
                    topology,

                "y_state":
                    y_state,

                "hy":
                    hy,

                "readout":
                    readout,

                "circuit_type":
                    "measurement",

                "measurement_setting":
                    setting,

                **measurement_resources,
            }
        )


logical_df = pd.DataFrame(
    logical_rows
)

logical_df.to_csv(
    RESULTS
    / "09_01y_logical_resource_baseline.csv",
    index=False,
)


# =============================================================================
# IBM service and backends
#
# Conventional non-fractional target is kept for direct comparability with the
# original Week-9 CZ-based transpilation experiment.
# =============================================================================

service = get_service()

backend_map = {}

for backend_name in BACKEND_NAMES:

    backend_map[
        backend_name
    ] = service.backend(
        backend_name,
        use_fractional_gates=False,
    )


# =============================================================================
# Backend ISA snapshot
# =============================================================================

backend_isa_rows = []

for (
    backend_name,
    backend,
) in backend_map.items():

    target = backend.target

    backend_isa_rows.append(
        {
            "backend":
                backend_name,

            "num_qubits":
                int(
                    backend.num_qubits
                ),

            "target_operations":
                json.dumps(
                    sorted(
                        [
                            str(
                                name
                            )
                            for name
                            in target.operation_names
                        ]
                    )
                ),

            "dt":
                (
                    float(
                        target.dt
                    )
                    if getattr(
                        target,
                        "dt",
                        None,
                    )
                    is not None
                    else np.nan
                ),

            "fractional_gates":
                False,
        }
    )


backend_isa_df = pd.DataFrame(
    backend_isa_rows
)

backend_isa_df.to_csv(
    RESULTS
    / "09_01y_backend_isa.csv",
    index=False,
)


# =============================================================================
# Compilation
# =============================================================================

core_rows = []
measurement_rows = []

print(
    "=" * 120
)

print(
    "WEEK 9 — STEP 9.1 REFRESH"
)

print(
    "FINAL H0-H4 × Y OFF/ON PHYSICAL TRANSPILATION"
)

print(
    "=" * 120
)

print(
    "\nOptimization-level policy:"
)

print(
    "  levels 0 and 1 = PRIMARY"
)

print(
    "  levels 2 and 3 = compiler sensitivity"
)

print(
    "  strict routing_method='none' for every H0-H4 candidate"
)

print(
    "  use_fractional_gates=False"
)

print(
    "  T2 is not rerun; original routed reference remains frozen"
)


for candidate in CANDIDATES:

    topology = candidate[
        "topology"
    ]

    y_state = candidate[
        "y_state"
    ]

    hy = float(
        candidate[
            "hy"
        ]
    )

    readout = candidate[
        "readout"
    ]

    candidate_id = (
        f"{topology}_Y{y_state}"
    )

    core = build_core_circuit(
        topology,
        hy,
    )

    settings = READOUT_SETTINGS[
        readout
    ]

    matching_embeddings = (
        embeddings[
            embeddings[
                "candidate"
            ] == topology
        ]
        .copy()
        .sort_values(
            [
                "backend",
                "final_rank",
            ]
        )
    )

    print(
        "\n"
        +
        "-" * 120
    )

    print(
        f"{candidate_id}: "
        f"h_y={hy:+.3f}, "
        f"readout={readout}, "
        f"settings={settings}"
    )

    print(
        "-" * 120
    )

    for _, embedding in matching_embeddings.iterrows():

        backend_name = str(
            embedding[
                "backend"
            ]
        )

        if backend_name not in backend_map:
            continue

        backend = backend_map[
            backend_name
        ]

        layout = list(
            embedding[
                "layout"
            ]
        )

        final_rank = int(
            embedding[
                "final_rank"
            ]
        )

        for optimization_level in OPT_LEVELS:

            common_compile_kwargs = {
                "backend":
                    backend,

                "initial_layout":
                    layout,

                "routing_method":
                    "none",

                "optimization_level":
                    optimization_level,

                "seed_transpiler":
                    SEED_TRANSPILER,
            }

            # -------------------------------------------------
            # Coherent reservoir core
            # -------------------------------------------------

            try:
                compiled_core = transpile(
                    core,
                    **common_compile_kwargs,
                )

            except Exception as exc:
                raise RuntimeError(
                    "\nStrict no-routing compilation FAILED.\n"
                    f"candidate={candidate_id}\n"
                    f"backend={backend_name}\n"
                    f"rank={final_rank}\n"
                    f"layout={layout}\n"
                    f"optimization_level={optimization_level}\n"
                    f"error={exc}"
                ) from exc

            core_resources = count_resources(
                compiled_core
            )

            core_rows.append(
                {
                    "candidate_id":
                        candidate_id,

                    "topology":
                        topology,

                    "y_state":
                        y_state,

                    "hy":
                        hy,

                    "readout":
                        readout,

                    "backend":
                        backend_name,

                    "embedding_rank":
                        final_rank,

                    "layout":
                        json.dumps(
                            layout
                        ),

                    "optimization_level":
                        optimization_level,

                    "is_primary_level":
                        optimization_level
                        in PRIMARY_OPT_LEVELS,

                    "seed_transpiler":
                        SEED_TRANSPILER,

                    **core_resources,
                }
            )

            if core_resources[
                "n_swap"
            ] != 0:
                raise RuntimeError(
                    f"{candidate_id} {backend_name} rank={final_rank} "
                    f"L{optimization_level}: unexpected SWAP count "
                    f"{core_resources['n_swap']}"
                )

            # -------------------------------------------------
            # Every selected grouped-measurement circuit
            # -------------------------------------------------

            for setting in settings:

                measurement_circuit = (
                    build_measurement_circuit(
                        core,
                        setting,
                    )
                )

                try:
                    compiled_measurement = transpile(
                        measurement_circuit,
                        **common_compile_kwargs,
                    )

                except Exception as exc:
                    raise RuntimeError(
                        "\nStrict no-routing measurement compilation FAILED.\n"
                        f"candidate={candidate_id}\n"
                        f"setting={setting}\n"
                        f"backend={backend_name}\n"
                        f"rank={final_rank}\n"
                        f"layout={layout}\n"
                        f"optimization_level={optimization_level}\n"
                        f"error={exc}"
                    ) from exc

                measurement_resources = count_resources(
                    compiled_measurement
                )

                measurement_rows.append(
                    {
                        "candidate_id":
                            candidate_id,

                        "topology":
                            topology,

                        "y_state":
                            y_state,

                        "hy":
                            hy,

                        "readout":
                            readout,

                        "measurement_setting":
                            setting,

                        "backend":
                            backend_name,

                        "embedding_rank":
                            final_rank,

                        "layout":
                            json.dumps(
                                layout
                            ),

                        "optimization_level":
                            optimization_level,

                        "is_primary_level":
                            optimization_level
                            in PRIMARY_OPT_LEVELS,

                        "seed_transpiler":
                            SEED_TRANSPILER,

                        **measurement_resources,
                    }
                )

                if measurement_resources[
                    "n_swap"
                ] != 0:
                    raise RuntimeError(
                        f"{candidate_id} {setting} "
                        f"{backend_name} rank={final_rank} "
                        f"L{optimization_level}: unexpected SWAP count "
                        f"{measurement_resources['n_swap']}"
                    )

            print(
                f"{backend_name:16s} "
                f"rank={final_rank} "
                f"L{optimization_level} "
                f"core: "
                f"SWAP={core_resources['n_swap']:2d} "
                f"CZ={core_resources['n_cz']:2d} "
                f"1Q={core_resources['n_1q']:3d} "
                f"depth={core_resources['depth']:3d} "
                f"1Qdepth={core_resources['depth_1q']:3d} "
                f"2Qdepth={core_resources['depth_2q']:2d}"
            )


# =============================================================================
# Save raw compilation tables
# =============================================================================

core_df = pd.DataFrame(
    core_rows
)

measurement_df = pd.DataFrame(
    measurement_rows
)

core_df.to_csv(
    RESULTS
    / "09_01y_core_all_levels.csv",
    index=False,
)

measurement_df.to_csv(
    RESULTS
    / "09_01y_measurement_all_levels.csv",
    index=False,
)


# =============================================================================
# Summaries
# =============================================================================

core_summary = (
    core_df
    .groupby(
        [
            "candidate_id",
            "topology",
            "y_state",
            "hy",
            "readout",
            "backend",
            "optimization_level",
        ],
        as_index=False,
    )
    .agg(
        n_embeddings=(
            "embedding_rank",
            "count",
        ),

        swap_min=(
            "n_swap",
            "min",
        ),

        swap_median=(
            "n_swap",
            "median",
        ),

        swap_max=(
            "n_swap",
            "max",
        ),

        cz_min=(
            "n_cz",
            "min",
        ),

        cz_median=(
            "n_cz",
            "median",
        ),

        cz_max=(
            "n_cz",
            "max",
        ),

        oneq_min=(
            "n_1q",
            "min",
        ),

        oneq_median=(
            "n_1q",
            "median",
        ),

        oneq_max=(
            "n_1q",
            "max",
        ),

        depth_min=(
            "depth",
            "min",
        ),

        depth_median=(
            "depth",
            "median",
        ),

        depth_max=(
            "depth",
            "max",
        ),

        depth_1q_min=(
            "depth_1q",
            "min",
        ),

        depth_1q_median=(
            "depth_1q",
            "median",
        ),

        depth_1q_max=(
            "depth_1q",
            "max",
        ),

        depth_2q_min=(
            "depth_2q",
            "min",
        ),

        depth_2q_median=(
            "depth_2q",
            "median",
        ),

        depth_2q_max=(
            "depth_2q",
            "max",
        ),
    )
)

core_summary[
    "is_primary_level"
] = core_summary[
    "optimization_level"
].isin(
    PRIMARY_OPT_LEVELS
)

core_summary.to_csv(
    RESULTS
    / "09_01y_core_summary_all_levels.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# Per-candidate grouped-measurement cost:
#
# Each setting is a separate physical circuit.
#
# max_setting_depth is useful for the longest individual circuit.
# sum_*_per_feature_vector represents one execution of every setting once.
# 9.2 will later combine these with shots and scheduled durations.
# -----------------------------------------------------------------------------

measurement_per_embedding = (
    measurement_df
    .groupby(
        [
            "candidate_id",
            "topology",
            "y_state",
            "hy",
            "readout",
            "backend",
            "embedding_rank",
            "layout",
            "optimization_level",
            "is_primary_level",
        ],
        as_index=False,
    )
    .agg(
        n_settings=(
            "measurement_setting",
            "nunique",
        ),

        max_setting_depth=(
            "depth",
            "max",
        ),

        mean_setting_depth=(
            "depth",
            "mean",
        ),

        max_setting_depth_1q=(
            "depth_1q",
            "max",
        ),

        max_setting_depth_2q=(
            "depth_2q",
            "max",
        ),

        max_setting_cz=(
            "n_cz",
            "max",
        ),

        max_setting_1q=(
            "n_1q",
            "max",
        ),

        sum_cz_per_feature_vector=(
            "n_cz",
            "sum",
        ),

        sum_1q_per_feature_vector=(
            "n_1q",
            "sum",
        ),

        sum_measure_per_feature_vector=(
            "n_measure",
            "sum",
        ),

        any_swap=(
            "n_swap",
            "max",
        ),
    )
)

measurement_per_embedding.to_csv(
    RESULTS
    / "09_01y_measurement_per_embedding.csv",
    index=False,
)


measurement_summary = (
    measurement_per_embedding
    .groupby(
        [
            "candidate_id",
            "topology",
            "y_state",
            "hy",
            "readout",
            "backend",
            "optimization_level",
            "is_primary_level",
        ],
        as_index=False,
    )
    .agg(
        n_embeddings=(
            "embedding_rank",
            "count",
        ),

        n_settings=(
            "n_settings",
            "first",
        ),

        max_setting_depth_min=(
            "max_setting_depth",
            "min",
        ),

        max_setting_depth_median=(
            "max_setting_depth",
            "median",
        ),

        max_setting_depth_max=(
            "max_setting_depth",
            "max",
        ),

        max_setting_cz_median=(
            "max_setting_cz",
            "median",
        ),

        max_setting_1q_median=(
            "max_setting_1q",
            "median",
        ),

        sum_cz_per_feature_vector_median=(
            "sum_cz_per_feature_vector",
            "median",
        ),

        sum_1q_per_feature_vector_median=(
            "sum_1q_per_feature_vector",
            "median",
        ),

        any_swap_max=(
            "any_swap",
            "max",
        ),
    )
)

measurement_summary.to_csv(
    RESULTS
    / "09_01y_measurement_summary_all_levels.csv",
    index=False,
)


# =============================================================================
# Y ON minus Y OFF core-resource comparison
#
# Same topology/backend/rank/optimization level.
# =============================================================================

off = (
    core_df[
        core_df[
            "y_state"
        ] == "OFF"
    ]
    .copy()
)

on = (
    core_df[
        core_df[
            "y_state"
        ] == "ON"
    ]
    .copy()
)

comparison_keys = [
    "topology",
    "backend",
    "embedding_rank",
    "optimization_level",
]

y_compare = off.merge(
    on,
    on=comparison_keys,
    suffixes=(
        "_off",
        "_on",
    ),
)

for metric in [
    "n_swap",
    "n_cz",
    "n_1q",
    "depth",
    "depth_1q",
    "depth_2q",
    "size",
]:

    y_compare[
        f"delta_{metric}"
    ] = (
        y_compare[
            f"{metric}_on"
        ]
        -
        y_compare[
            f"{metric}_off"
        ]
    )

y_compare[
    "hy_on"
] = y_compare[
    "hy_on"
].astype(
    float
)

y_compare.to_csv(
    RESULTS
    / "09_01y_YON_minus_YOFF_core.csv",
    index=False,
)


# =============================================================================
# Primary levels 0/1
# =============================================================================

core_primary = core_df[
    core_df[
        "optimization_level"
    ].isin(
        PRIMARY_OPT_LEVELS
    )
].copy()

measurement_primary = (
    measurement_df[
        measurement_df[
            "optimization_level"
        ].isin(
            PRIMARY_OPT_LEVELS
        )
    ]
    .copy()
)

core_primary.to_csv(
    RESULTS
    / "09_01y_core_primary_levels_0_1.csv",
    index=False,
)

measurement_primary.to_csv(
    RESULTS
    / "09_01y_measurement_primary_levels_0_1.csv",
    index=False,
)


# =============================================================================
# Structural audits
# =============================================================================

audit_rows = []

for candidate_id, group in core_df.groupby(
    "candidate_id"
):

    audit_rows.append(
        {
            "candidate_id":
                candidate_id,

            "n_compilations":
                int(
                    len(
                        group
                    )
                ),

            "all_zero_swap":
                bool(
                    (
                        group[
                            "n_swap"
                        ]
                        ==
                        0
                    ).all()
                ),

            "cz_min":
                int(
                    group[
                        "n_cz"
                    ].min()
                ),

            "cz_max":
                int(
                    group[
                        "n_cz"
                    ].max()
                ),

            "all_20_cz":
                bool(
                    (
                        group[
                            "n_cz"
                        ]
                        ==
                        20
                    ).all()
                ),

            "depth_min":
                int(
                    group[
                        "depth"
                    ].min()
                ),

            "depth_max":
                int(
                    group[
                        "depth"
                    ].max()
                ),

            "oneq_min":
                int(
                    group[
                        "n_1q"
                    ].min()
                ),

            "oneq_max":
                int(
                    group[
                        "n_1q"
                    ].max()
                ),
        }
    )


structural_audit_df = pd.DataFrame(
    audit_rows
)

structural_audit_df.to_csv(
    RESULTS
    / "09_01y_structural_audit.csv",
    index=False,
)


# =============================================================================
# Manifest
# =============================================================================

manifest = {
    "step":
        "Week 9 Step 9.1 refresh after Y-field selection",

    "candidate_count":
        len(
            CANDIDATES
        ),

    "candidate_ids":
        list(
            candidate_df[
                "candidate_id"
            ]
        ),

    "backends":
        BACKEND_NAMES,

    "optimization_levels":
        OPT_LEVELS,

    "primary_optimization_levels":
        PRIMARY_OPT_LEVELS,

    "seed_transpiler":
        SEED_TRANSPILER,

    "routing_method":
        "none",

    "use_fractional_gates":
        False,

    "T2_rerun":
        False,

    "T2_policy":
        "retain original Step 9.1 routed reference only",

    "alpha":
        ALPHA,

    "hx":
        HX,

    "dt":
        DT,

    "Trotter_r":
        TROTTER_R,

    "Hamiltonian":
        "sum Jij ZiZj + hx sum Xi + hy(Y4+Y5)",

    "Trotter_order":
        "ZZ -> X -> memory-Y, repeated r times",

    "theta_X_per_slice":
        2.0
        *
        HX
        *
        DT
        /
        TROTTER_R,

    "theta_Y_formula_per_slice":
        "2*hy*dt/r",

    "readout_settings":
        READOUT_SETTINGS,

    "input_source":
        str(
            DATA_FILE
        ),

    "embedding_source":
        str(
            EMBEDDING_FILE
        ),

    "J_source":
        str(
            J_WINNER_FILE
        ),

    "weighted_score":
        False,
}

with open(
    RESULTS
    / "09_01y_manifest.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        manifest,
        file,
        indent=2,
    )


# =============================================================================
# Console summaries
# =============================================================================

print(
    "\n"
    +
    "=" * 120
)

print(
    "FINAL 10-CANDIDATE TABLE"
)

print(
    "=" * 120
)

print(
    candidate_df[
        [
            "candidate_id",
            "topology",
            "y_state",
            "hy",
            "readout",
            "ridge_lambda",
            "cv_rmse",
            "validation_rmse",
            "mc_ch",
            "washout_mean",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 120
)

print(
    "LOGICAL RESOURCE BASELINE"
)

print(
    "=" * 120
)

print(
    logical_df[
        [
            "candidate_id",
            "circuit_type",
            "measurement_setting",
            "n_1q",
            "n_2q",
            "n_cz",
            "n_swap",
            "depth",
            "depth_1q",
            "depth_2q",
            "operations",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 120
)

print(
    "STRUCTURAL AUDIT — ALL BACKENDS / EMBEDDINGS / LEVELS"
)

print(
    "=" * 120
)

print(
    structural_audit_df.to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 120
)

print(
    "CORE SUMMARY — PRIMARY LEVELS 0 AND 1"
)

print(
    "=" * 120
)

primary_summary = core_summary[
    core_summary[
        "optimization_level"
    ].isin(
        PRIMARY_OPT_LEVELS
    )
]

print(
    primary_summary[
        [
            "candidate_id",
            "backend",
            "optimization_level",
            "n_embeddings",
            "swap_median",
            "cz_median",
            "oneq_median",
            "depth_median",
            "depth_1q_median",
            "depth_2q_median",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 120
)

print(
    "GROUPED-MEASUREMENT SUMMARY — PRIMARY LEVELS 0 AND 1"
)

print(
    "=" * 120
)

measurement_primary_summary = (
    measurement_summary[
        measurement_summary[
            "optimization_level"
        ].isin(
            PRIMARY_OPT_LEVELS
        )
    ]
)

print(
    measurement_primary_summary[
        [
            "candidate_id",
            "readout",
            "backend",
            "optimization_level",
            "n_settings",
            "max_setting_depth_median",
            "max_setting_cz_median",
            "max_setting_1q_median",
            "sum_cz_per_feature_vector_median",
            "sum_1q_per_feature_vector_median",
            "any_swap_max",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 120
)

print(
    "Y ON MINUS Y OFF — CORE RESOURCE DELTAS"
)

print(
    "=" * 120
)

delta_columns = [
    "topology",
    "backend",
    "embedding_rank",
    "optimization_level",
    "hy_on",
    "delta_n_swap",
    "delta_n_cz",
    "delta_n_1q",
    "delta_depth",
    "delta_depth_1q",
    "delta_depth_2q",
]

print(
    y_compare[
        delta_columns
    ]
    .sort_values(
        [
            "topology",
            "backend",
            "embedding_rank",
            "optimization_level",
        ]
    )
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

for filename in [
    "09_01y_final_10_candidates.csv",
    "09_01y_first_input_audit.csv",
    "09_01y_logical_resource_baseline.csv",
    "09_01y_backend_isa.csv",
    "09_01y_core_all_levels.csv",
    "09_01y_core_summary_all_levels.csv",
    "09_01y_core_primary_levels_0_1.csv",
    "09_01y_measurement_all_levels.csv",
    "09_01y_measurement_per_embedding.csv",
    "09_01y_measurement_summary_all_levels.csv",
    "09_01y_measurement_primary_levels_0_1.csv",
    "09_01y_YON_minus_YOFF_core.csv",
    "09_01y_structural_audit.csv",
    "09_01y_manifest.json",
]:
    print(
        f"  results/{filename}"
    )


print(
    "\nStep 9.1 refresh complete."
)
