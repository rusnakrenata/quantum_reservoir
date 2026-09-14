from pathlib import Path
import json
import math
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit, transpile

from ibm_account import get_service


# =============================================================================
# WEEK 9 — STEP 9.3
# TROTTER-DEPTH / HARDWARE-COST ABLATION
#
# Controlled ablation:
#   r in {1,2,3,4}
#
# Frozen for each of the 10 retained H0-H4 × {Y OFF,Y ON} candidates:
#   - topology,
#   - Week-8 topology-specific J values (originally optimized at r=2),
#   - h_y,
#   - alpha,
#   - h_x,
#   - dt,
#   - selected readout family.
#
# Re-selected for every candidate × r:
#   - Ridge lambda via TRAINING-ONLY chronological CV.
#
# Metrics:
#   A. Forecast:
#      - CV RMSE mean/std
#      - validation RMSE/bias
#      - selected lambda
#
#   B. Trotter approximation:
#      - exact-vs-Trotter process fidelity
#      - process infidelity
#      - phase-aligned normalized Frobenius unitary error
#
#   C. Physical resources:
#      - qubits
#      - CZ
#      - SWAP
#      - 1Q gates
#      - total / 1Q / 2Q depth
#      - scheduled physical duration
#      - t/T1_min
#      - t/T2_min
#
#   D. Selected grouped measurement:
#      - number of settings
#      - longest setting duration
#      - total feature-vector circuit duration
#      - estimated circuit-time contribution for 1024 shots/setting
#
# Interpretation warning:
#   Because J/h_y/readout are FROZEN from the r=2 design process, this is a
#   controlled robustness/hardware ablation, NOT a global re-optimization of
#   the complete reservoir for every r.
# =============================================================================


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DATA_FILE = RESULTS / "03_01_preprocessed_samples.csv"
CANDIDATE_FILE = RESULTS / "09_01y_final_10_candidates.csv"
J_WINNER_FILE = RESULTS / "08_03b_J_search_topology_winners.csv"
EMBEDDING_FILE = RESULTS / "08_03a_final_top3_embeddings.csv"

# Freeze the Step-9.2 calibration snapshot if available, so the r comparison
# does not get confounded by a fresh T1/T2/readout/CZ calibration pull.
CALIBRATION_FILE = RESULTS / "09_02_calibration_snapshot.csv"

for path in [
    DATA_FILE,
    CANDIDATE_FILE,
    J_WINNER_FILE,
    EMBEDDING_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# =============================================================================
# Frozen experiment policy
# =============================================================================

N_QUBITS = 6
N_INPUT = 4
D_INPUT = 16
D_MEMORY = 4
D_GLOBAL = 64

ALPHA = 0.75
HX = 0.5
DT = 1.6

R_GRID = [1, 2, 3, 4]
REFERENCE_R = 2

RIDGE_GRID = np.array(
    [
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        300.0,
    ],
    dtype=float,
)

N_CV_SPLITS = 5
VAR_TOL = 1e-12

OPT_LEVELS = [0, 1, 2, 3]
PRIMARY_OPT_LEVELS = [0, 1]

# Step 9.2 established that ASAP and ALAP have identical total durations
# (up to floating-point noise), so Step 9.3 uses one fixed schedule.
SCHEDULING_METHOD = "alap"
SCHEDULING_LABEL = "ALAP"

SEED_TRANSPILER = 42
SHOTS_PER_SETTING = 1024

BACKEND_NAMES = [
    "ibm_fez",
    "ibm_kingston",
    "ibm_marrakesh",
]


# =============================================================================
# Topologies and measurement families
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
    return z / (
        3.0
        +
        abs(z)
    )


def weekday_mapping(
    d_sin,
    d_cos,
):
    return (
        math.atan2(
            float(d_sin),
            float(d_cos),
        )
        %
        (
            2.0
            *
            math.pi
        )
    )


def holiday_mapping(h):
    h = int(h)

    if h not in (
        0,
        1,
    ):
        raise ValueError(
            f"Holiday must be 0/1; got {h}"
        )

    return (
        math.pi
        *
        h
    )


def encode_f4_row(row):
    return np.array(
        [
            ALPHA
            *
            claim_mapping(
                row[
                    "C_t_z"
                ]
            ),

            weekday_mapping(
                row[
                    "D_sin"
                ],
                row[
                    "D_cos"
                ],
            ),

            ALPHA
            *
            policy_mapping(
                row[
                    "P_t_z"
                ]
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
# Data
# =============================================================================

df = pd.read_csv(
    DATA_FILE
)

required_columns = [
    "input_date",
    "target_date",
    "split",
    "C_t_z",
    "D_sin",
    "D_cos",
    "P_t_z",
    "is_public_holiday_t_plus_1",
]

missing = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing:
    raise KeyError(
        "Missing preprocessing columns: "
        f"{missing}\n"
        f"Available columns: {list(df.columns)}"
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


def normalize_name(name):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower(),
    )


def find_target_column(frame):
    normalized = {
        normalize_name(c):
            c
        for c in frame.columns
    }

    candidates = [
        "target",
        "y",
        "C_t_plus_1",
        "claims_t_plus_1",
        "claim_count_t_plus_1",
        "property_damage_claim_count_t_plus_1",
        "target_claim_count",
        "target_property_damage",
        "target_property_damage_claim_count",
    ]

    for candidate in candidates:
        key = normalize_name(
            candidate
        )

        if key in normalized:
            column = normalized[
                key
            ]

            if pd.api.types.is_numeric_dtype(
                frame[
                    column
                ]
            ):
                return column

    target_like = [
        c
        for c in frame.columns
        if (
            "target"
            in
            normalize_name(c)
            and
            "date"
            not in
            normalize_name(c)
            and
            pd.api.types.is_numeric_dtype(
                frame[c]
            )
        )
    ]

    if len(
        target_like
    ) == 1:
        return target_like[
            0
        ]

    raise KeyError(
        "Could not determine target column.\n"
        f"Available columns: {list(frame.columns)}"
    )


TARGET_COLUMN = find_target_column(
    df
)

split_norm = (
    df[
        "split"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)

train_mask_all = (
    split_norm
    ==
    "train"
)

validation_mask_all = (
    split_norm
    .isin(
        [
            "validation",
            "val",
        ]
    )
)

work_df = (
    df[
        train_mask_all
        |
        validation_mask_all
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

work_split = (
    work_df[
        "split"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)

train_mask = (
    work_split
    ==
    "train"
).to_numpy()

validation_mask = (
    work_split
    .isin(
        [
            "validation",
            "val",
        ]
    )
).to_numpy()

N_TRAIN = int(
    train_mask.sum()
)

N_VALIDATION = int(
    validation_mask.sum()
)

N_TOTAL = len(
    work_df
)

if (
    N_TRAIN,
    N_VALIDATION,
    N_TOTAL,
) != (
    1095,
    365,
    1460,
):
    raise RuntimeError(
        "Expected train/validation/total = 1095/365/1460; got "
        f"{N_TRAIN}/{N_VALIDATION}/{N_TOTAL}"
    )

if (
    not
    np.all(
        train_mask[
            :N_TRAIN
        ]
    )
    or
    not
    np.all(
        validation_mask[
            N_TRAIN:
        ]
    )
):
    raise RuntimeError(
        "Train/validation chronology is not contiguous."
    )

Y_TARGET = (
    work_df[
        TARGET_COLUMN
    ]
    .to_numpy(
        dtype=float
    )
)

REAL_ANGLES = np.vstack(
    [
        encode_f4_row(
            row
        )
        for _, row
        in work_df.iterrows()
    ]
)


# =============================================================================
# Final 10 candidates
# =============================================================================

candidates = pd.read_csv(
    CANDIDATE_FILE
)

required_candidate_columns = [
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

missing = [
    c
    for c in required_candidate_columns
    if c not in candidates.columns
]

if missing:
    raise KeyError(
        f"Missing candidate columns: {missing}"
    )

if len(
    candidates
) != 10:
    raise RuntimeError(
        f"Expected 10 retained candidates; got {len(candidates)}"
    )

for readout in candidates[
    "readout"
].unique():

    if readout not in READOUT_SETTINGS:
        raise KeyError(
            f"Unknown readout family: {readout}"
        )


# =============================================================================
# Week-8 J winners
#
# IMPORTANT:
# These J values were selected in the original design at r=2.
# We deliberately KEEP them fixed for r=1,2,3,4.
# =============================================================================

j_df = pd.read_csv(
    J_WINNER_FILE
)

COUPLINGS = {}

j_source_audit_rows = []

for topology in TOPOLOGY_EDGES:

    matches = j_df[
        j_df[
            "topology"
        ] == topology
    ]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"{topology}: expected exactly one J winner; "
            f"found {len(matches)}"
        )

    winner = matches.iloc[
        0
    ]

    source_r = int(
        winner[
            "Trotter_r"
        ]
    )

    if source_r != REFERENCE_R:
        raise RuntimeError(
            f"{topology}: expected J source optimized at r=2; "
            f"found r={source_r}"
        )

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

    j_source_audit_rows.append(
        {
            "topology":
                topology,

            "J_source_Trotter_r":
                source_r,

            "tested_r":
                json.dumps(
                    R_GRID
                ),

            "reoptimized_J_per_r":
                False,
        }
    )


j_source_audit_df = pd.DataFrame(
    j_source_audit_rows
)

j_source_audit_df.to_csv(
    RESULTS
    / "09_03_J_source_audit.csv",
    index=False,
)


# =============================================================================
# Operators
# =============================================================================

I2 = np.eye(
    2,
    dtype=complex,
)

X2 = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=complex,
)

Y2 = np.array(
    [
        [0, -1j],
        [1j, 0],
    ],
    dtype=complex,
)

Z2 = np.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=complex,
)

I64 = np.eye(
    D_GLOBAL,
    dtype=complex,
)


def operator_on_qubit(
    op,
    q,
):
    result = np.array(
        [
            [
                1.0
                +
                0j
            ]
        ],
        dtype=complex,
    )

    for logical_q in reversed(
        range(
            N_QUBITS
        )
    ):
        result = np.kron(
            result,
            (
                op
                if logical_q == q
                else I2
            ),
        )

    return result


X_OPS = [
    operator_on_qubit(
        X2,
        q,
    )
    for q in range(
        N_QUBITS
    )
]

Y_OPS = [
    operator_on_qubit(
        Y2,
        q,
    )
    for q in range(
        N_QUBITS
    )
]

Z_OPS = [
    operator_on_qubit(
        Z2,
        q,
    )
    for q in range(
        N_QUBITS
    )
]


ZZ_OPS = {}

for topology in TOPOLOGY_EDGES:

    for edge in TOPOLOGY_EDGES[
        topology
    ]:

        canonical = tuple(
            sorted(
                edge
            )
        )

        if canonical not in ZZ_OPS:

            ZZ_OPS[
                canonical
            ] = (
                Z_OPS[
                    canonical[
                        0
                    ]
                ]
                @
                Z_OPS[
                    canonical[
                        1
                    ]
                ]
            )


YX45_OP = (
    Y_OPS[
        4
    ]
    @
    X_OPS[
        5
    ]
)


# =============================================================================
# Exact and Trotter unitary
# =============================================================================

def pauli_exponential(
    op,
    angle,
):
    return (
        math.cos(
            angle
        )
        *
        I64
        -
        1j
        *
        math.sin(
            angle
        )
        *
        op
    )


def build_hamiltonian(
    topology,
    hy,
):
    H = np.zeros(
        (
            D_GLOBAL,
            D_GLOBAL,
        ),
        dtype=complex,
    )

    for edge in TOPOLOGY_EDGES[
        topology
    ]:

        Jij = COUPLINGS[
            topology
        ][
            edge
        ]

        H += (
            Jij
            *
            ZZ_OPS[
                tuple(
                    sorted(
                        edge
                    )
                )
            ]
        )

    for q in range(
        N_QUBITS
    ):
        H += (
            HX
            *
            X_OPS[
                q
            ]
        )

    if abs(
        hy
    ) > 0.0:

        H += (
            hy
            *
            (
                Y_OPS[
                    4
                ]
                +
                Y_OPS[
                    5
                ]
            )
        )

    hermiticity_error = float(
        np.linalg.norm(
            H
            -
            H.conj().T,
            ord="fro",
        )
    )

    return (
        H,
        hermiticity_error,
    )


def exact_unitary_from_hamiltonian(
    H,
):
    # H is Hermitian, so eigh is stable and avoids requiring scipy.expm.
    eigenvalues, eigenvectors = np.linalg.eigh(
        H
    )

    phases = np.exp(
        -1j
        *
        eigenvalues
        *
        DT
    )

    return (
        (
            eigenvectors
            *
            phases[
                np.newaxis,
                :
            ]
        )
        @
        eigenvectors.conj().T
    )


def build_trotter_unitary(
    topology,
    hy,
    trotter_r,
):
    U = np.eye(
        D_GLOBAL,
        dtype=complex,
    )

    for _ in range(
        int(
            trotter_r
        )
    ):

        # ZZ sector
        for edge in TOPOLOGY_EDGES[
            topology
        ]:

            Jij = COUPLINGS[
                topology
            ][
                edge
            ]

            angle = (
                Jij
                *
                DT
                /
                trotter_r
            )

            U = (
                pauli_exponential(
                    ZZ_OPS[
                        tuple(
                            sorted(
                                edge
                            )
                        )
                    ],
                    angle,
                )
                @
                U
            )

        # X sector
        angle_x = (
            HX
            *
            DT
            /
            trotter_r
        )

        for q in range(
            N_QUBITS
        ):

            U = (
                pauli_exponential(
                    X_OPS[
                        q
                    ],
                    angle_x,
                )
                @
                U
            )

        # Memory-only Y sector
        if abs(
            hy
        ) > 0.0:

            angle_y = (
                hy
                *
                DT
                /
                trotter_r
            )

            for q in (
                4,
                5,
            ):

                U = (
                    pauli_exponential(
                        Y_OPS[
                            q
                        ],
                        angle_y,
                    )
                    @
                    U
                )

    unitary_error = float(
        np.linalg.norm(
            U.conj().T
            @
            U
            -
            I64,
            ord="fro",
        )
    )

    return (
        U,
        unitary_error,
    )


def unitary_accuracy_metrics(
    U_exact,
    U_trotter,
):
    overlap = np.trace(
        U_exact.conj().T
        @
        U_trotter
    )

    process_fidelity = float(
        (
            abs(
                overlap
            )
            /
            D_GLOBAL
        )
        ** 2
    )

    process_fidelity = min(
        max(
            process_fidelity,
            0.0,
        ),
        1.0,
    )

    phase = float(
        np.angle(
            overlap
        )
    )

    U_aligned = (
        np.exp(
            -1j
            *
            phase
        )
        *
        U_trotter
    )

    fro_error = float(
        np.linalg.norm(
            U_exact
            -
            U_aligned,
            ord="fro",
        )
        /
        math.sqrt(
            D_GLOBAL
        )
    )

    return {
        "process_fidelity":
            process_fidelity,

        "process_infidelity":
            (
                1.0
                -
                process_fidelity
            ),

        "phase_aligned_normalized_fro_error":
            fro_error,

        "global_phase_alignment_rad":
            phase,
    }


# =============================================================================
# CONT propagation
# =============================================================================

def ry_zero_state(theta):
    return np.array(
        [
            math.cos(
                theta
                /
                2.0
            ),
            math.sin(
                theta
                /
                2.0
            ),
        ],
        dtype=complex,
    )


def input_state_vector(
    angles,
):
    q0 = ry_zero_state(
        angles[
            0
        ]
    )

    q1 = ry_zero_state(
        angles[
            1
        ]
    )

    q2 = ry_zero_state(
        angles[
            2
        ]
    )

    q3 = ry_zero_state(
        angles[
            3
        ]
    )

    return np.kron(
        np.kron(
            np.kron(
                q3,
                q2,
            ),
            q1,
        ),
        q0,
    )


REAL_PHI = [
    input_state_vector(
        angles
    )
    for angles in REAL_ANGLES
]

I4 = np.eye(
    D_MEMORY,
    dtype=complex,
)

KET_00 = np.array(
    [
        1,
        0,
        0,
        0,
    ],
    dtype=complex,
)

RHO_00 = np.outer(
    KET_00,
    KET_00.conj(),
)


def isometry_for_input(
    U,
    phi,
):
    embedding = np.kron(
        I4,
        np.asarray(
            phi,
            dtype=complex,
        ).reshape(
            -1,
            1,
        ),
    )

    return (
        U
        @
        embedding
    )


def partial_trace_input(
    rho_global,
):
    reshaped = rho_global.reshape(
        D_MEMORY,
        D_INPUT,
        D_MEMORY,
        D_INPUT,
    )

    rho_memory = np.einsum(
        "aibi->ab",
        reshaped,
    )

    rho_memory = (
        0.5
        *
        (
            rho_memory
            +
            rho_memory.conj().T
        )
    )

    rho_memory /= np.trace(
        rho_memory
    )

    return rho_memory


BASIS_INDICES = np.arange(
    D_GLOBAL
)


def xyz_expectations(
    rho_global,
):
    diagonal = np.real(
        np.diag(
            rho_global
        )
    )

    x_values = np.zeros(
        N_QUBITS,
        dtype=float,
    )

    y_values = np.zeros(
        N_QUBITS,
        dtype=float,
    )

    z_values = np.zeros(
        N_QUBITS,
        dtype=float,
    )

    for q in range(
        N_QUBITS
    ):

        mask = (
            1
            <<
            q
        )

        flipped = (
            BASIS_INDICES
            ^
            mask
        )

        offdiag = rho_global[
            BASIS_INDICES,
            flipped,
        ]

        x_values[
            q
        ] = float(
            np.real(
                np.sum(
                    offdiag
                )
            )
        )

        bits = (
            BASIS_INDICES
            >>
            q
        ) & 1

        y_coeff = np.where(
            bits == 0,
            1j,
            -1j,
        )

        y_values[
            q
        ] = float(
            np.real(
                np.sum(
                    offdiag
                    *
                    y_coeff
                )
            )
        )

        z_sign = (
            1.0
            -
            2.0
            *
            bits
        )

        z_values[
            q
        ] = float(
            np.sum(
                diagonal
                *
                z_sign
            )
        )

    return (
        x_values,
        y_values,
        z_values,
    )


def expectation(
    rho,
    operator,
):
    return float(
        np.real(
            np.trace(
                rho
                @
                operator
            )
        )
    )


def simulate_selected_readout(
    U,
    readout,
):
    n = len(
        REAL_PHI
    )

    if readout == "XZ_injection":
        X = np.zeros(
            (
                n,
                8,
            ),
            dtype=float,
        )

    elif readout == "XZinj_plus_YX45":
        X = np.zeros(
            (
                n,
                9,
            ),
            dtype=float,
        )

    elif readout == "XYZ_all":
        X = np.zeros(
            (
                n,
                18,
            ),
            dtype=float,
        )

    else:
        raise KeyError(
            f"Unsupported readout: {readout}"
        )

    rho_memory = RHO_00.copy()

    max_trace_error = 0.0
    max_hermiticity_error = 0.0
    min_memory_eigenvalue = float(
        "inf"
    )

    for t, phi in enumerate(
        REAL_PHI
    ):

        V = isometry_for_input(
            U,
            phi,
        )

        rho_global = (
            V
            @
            rho_memory
            @
            V.conj().T
        )

        (
            x_values,
            y_values,
            z_values,
        ) = xyz_expectations(
            rho_global
        )

        xzinj = np.concatenate(
            [
                x_values[
                    :4
                ],
                z_values[
                    :4
                ],
            ]
        )

        if readout == "XZ_injection":
            X[
                t
            ] = xzinj

        elif readout == "XZinj_plus_YX45":

            yx45 = expectation(
                rho_global,
                YX45_OP,
            )

            X[
                t
            ] = np.concatenate(
                [
                    xzinj,
                    [
                        yx45
                    ],
                ]
            )

        elif readout == "XYZ_all":

            X[
                t
            ] = np.concatenate(
                [
                    x_values,
                    y_values,
                    z_values,
                ]
            )

        rho_memory = partial_trace_input(
            rho_global
        )

        max_trace_error = max(
            max_trace_error,
            float(
                abs(
                    np.trace(
                        rho_memory
                    )
                    -
                    1.0
                )
            ),
        )

        max_hermiticity_error = max(
            max_hermiticity_error,
            float(
                np.linalg.norm(
                    rho_memory
                    -
                    rho_memory.conj().T,
                    ord="fro",
                )
            ),
        )

        min_memory_eigenvalue = min(
            min_memory_eigenvalue,
            float(
                np.min(
                    np.linalg.eigvalsh(
                        rho_memory
                    )
                )
            ),
        )

    diagnostics = {
        "max_trace_error":
            max_trace_error,

        "max_hermiticity_error":
            max_hermiticity_error,

        "min_memory_eigenvalue":
            min_memory_eigenvalue,

        "final_memory_purity":
            float(
                np.real(
                    np.trace(
                        rho_memory
                        @
                        rho_memory
                    )
                )
            ),
    }

    return (
        X,
        diagnostics,
    )


# =============================================================================
# Ridge forecast
# =============================================================================

def rmse(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    y_true
                    -
                    y_pred
                )
                ** 2
            )
        )
    )


def fit_feature_transform(
    X_train,
):
    X_train = np.asarray(
        X_train,
        dtype=float,
    )

    std = np.std(
        X_train,
        axis=0,
        ddof=0,
    )

    keep = (
        std
        >
        VAR_TOL
    )

    if not np.any(
        keep
    ):
        return (
            None,
            None,
            None,
        )

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train[
                :,
                keep
            ]
        )
    )

    return (
        keep,
        scaler,
        X_train_scaled,
    )


def chronological_cv_for_lambda(
    X_train,
    y_train,
    ridge_alpha,
):
    splitter = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )

    fold_rmse = []

    for (
        tr,
        va,
    ) in splitter.split(
        X_train
    ):

        (
            keep,
            scaler,
            Xtr_z,
        ) = fit_feature_transform(
            X_train[
                tr
            ]
        )

        if keep is None:

            pred = np.full(
                len(
                    va
                ),
                float(
                    np.mean(
                        y_train[
                            tr
                        ]
                    )
                ),
            )

        else:

            Xva_z = scaler.transform(
                X_train[
                    va
                ][
                    :,
                    keep
                ]
            )

            model = Ridge(
                alpha=float(
                    ridge_alpha
                ),
                fit_intercept=True,
            )

            model.fit(
                Xtr_z,
                y_train[
                    tr
                ],
            )

            pred = model.predict(
                Xva_z
            )

        fold_rmse.append(
            rmse(
                y_train[
                    va
                ],
                pred,
            )
        )

    return {
        "cv_rmse_mean":
            float(
                np.mean(
                    fold_rmse
                )
            ),

        "cv_rmse_std":
            float(
                np.std(
                    fold_rmse,
                    ddof=1,
                )
            ),

        "fold_rmse":
            [
                float(
                    value
                )
                for value in fold_rmse
            ],
    }


def final_fit_validation(
    X,
    y,
    ridge_alpha,
):
    X_train = X[
        :N_TRAIN
    ]

    y_train = y[
        :N_TRAIN
    ]

    X_val = X[
        N_TRAIN:
    ]

    y_val = y[
        N_TRAIN:
    ]

    (
        keep,
        scaler,
        Xtr_z,
    ) = fit_feature_transform(
        X_train
    )

    if keep is None:

        pred = np.full(
            len(
                X_val
            ),
            float(
                np.mean(
                    y_train
                )
            ),
        )

        n_retained = 0
        rank = 0

    else:

        Xval_z = scaler.transform(
            X_val[
                :,
                keep
            ]
        )

        model = Ridge(
            alpha=float(
                ridge_alpha
            ),
            fit_intercept=True,
        )

        model.fit(
            Xtr_z,
            y_train,
        )

        pred = model.predict(
            Xval_z
        )

        n_retained = int(
            keep.sum()
        )

        rank = int(
            np.linalg.matrix_rank(
                Xtr_z
            )
        )

    return {
        "validation_rmse":
            rmse(
                y_val,
                pred,
            ),

        "validation_bias":
            float(
                np.mean(
                    pred
                    -
                    y_val
                )
            ),

        "n_features_retained":
            n_retained,

        "feature_rank":
            rank,
    }


def evaluate_readout(
    X,
    y,
):
    X_train = X[
        :N_TRAIN
    ]

    y_train = y[
        :N_TRAIN
    ]

    grid_rows = []

    for ridge_alpha in RIDGE_GRID:

        cv = chronological_cv_for_lambda(
            X_train,
            y_train,
            ridge_alpha,
        )

        grid_rows.append(
            {
                "ridge_alpha":
                    float(
                        ridge_alpha
                    ),

                "cv_rmse_mean":
                    cv[
                        "cv_rmse_mean"
                    ],

                "cv_rmse_std":
                    cv[
                        "cv_rmse_std"
                    ],

                "fold_rmse":
                    cv[
                        "fold_rmse"
                    ],
            }
        )

    grid_df = pd.DataFrame(
        grid_rows
    )

    grid_df = (
        grid_df
        .sort_values(
            [
                "cv_rmse_mean",
                "ridge_alpha",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    best = grid_df.iloc[
        0
    ]

    selected_lambda = float(
        best[
            "ridge_alpha"
        ]
    )

    validation = final_fit_validation(
        X,
        y,
        selected_lambda,
    )

    return {
        "selected_lambda":
            selected_lambda,

        "selected_cv_rmse_mean":
            float(
                best[
                    "cv_rmse_mean"
                ]
            ),

        "selected_cv_rmse_std":
            float(
                best[
                    "cv_rmse_std"
                ]
            ),

        "selected_fold_rmse":
            best[
                "fold_rmse"
            ],

        "selected_validation_rmse":
            validation[
                "validation_rmse"
            ],

        "selected_validation_bias":
            validation[
                "validation_bias"
            ],

        "n_features_retained":
            validation[
                "n_features_retained"
            ],

        "feature_rank":
            validation[
                "feature_rank"
            ],

        "grid":
            grid_df,
    }


# =============================================================================
# Forecast + exact-vs-Trotter experiment
# =============================================================================

forecast_rows = []
ridge_grid_rows = []
diagnostic_rows = []

print(
    "=" * 124
)

print(
    "WEEK 9 — STEP 9.3"
)

print(
    "TROTTER-DEPTH / HARDWARE-COST ABLATION"
)

print(
    "=" * 124
)

print(
    f"r grid: {R_GRID}"
)

print(
    "Frozen: topology, J, h_y, alpha, h_x, dt, readout"
)

print(
    "Re-selected per candidate/r: Ridge lambda using training-only chronological CV"
)

print(
    "J values were originally selected at r=2; they are NOT reoptimized here."
)

forecast_start = time.perf_counter()


for _, candidate in candidates.iterrows():

    candidate_id = str(
        candidate[
            "candidate_id"
        ]
    )

    topology = str(
        candidate[
            "topology"
        ]
    )

    hy = float(
        candidate[
            "hy"
        ]
    )

    readout = str(
        candidate[
            "readout"
        ]
    )

    H, h_error = build_hamiltonian(
        topology,
        hy,
    )

    U_exact = exact_unitary_from_hamiltonian(
        H
    )

    exact_unitarity_error = float(
        np.linalg.norm(
            U_exact.conj().T
            @
            U_exact
            -
            I64,
            ord="fro",
        )
    )

    print(
        "\n"
        +
        "-" * 124
    )

    print(
        f"{candidate_id}: topology={topology}, "
        f"h_y={hy:+.3f}, readout={readout}"
    )

    print(
        "-" * 124
    )

    for trotter_r in R_GRID:

        start = time.perf_counter()

        U_trotter, trotter_unitarity_error = (
            build_trotter_unitary(
                topology,
                hy,
                trotter_r,
            )
        )

        accuracy = unitary_accuracy_metrics(
            U_exact,
            U_trotter,
        )

        X, diagnostics = simulate_selected_readout(
            U_trotter,
            readout,
        )

        forecast = evaluate_readout(
            X,
            Y_TARGET,
        )

        runtime_s = (
            time.perf_counter()
            -
            start
        )

        forecast_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "topology":
                    topology,

                "y_state":
                    str(
                        candidate[
                            "y_state"
                        ]
                    ),

                "hy":
                    hy,

                "readout":
                    readout,

                "trotter_r":
                    trotter_r,

                "n_qubits":
                    N_QUBITS,

                "selected_lambda":
                    forecast[
                        "selected_lambda"
                    ],

                "cv_rmse":
                    forecast[
                        "selected_cv_rmse_mean"
                    ],

                "cv_rmse_std":
                    forecast[
                        "selected_cv_rmse_std"
                    ],

                "validation_rmse":
                    forecast[
                        "selected_validation_rmse"
                    ],

                "validation_bias":
                    forecast[
                        "selected_validation_bias"
                    ],

                "n_features_retained":
                    forecast[
                        "n_features_retained"
                    ],

                "feature_rank":
                    forecast[
                        "feature_rank"
                    ],

                "process_fidelity":
                    accuracy[
                        "process_fidelity"
                    ],

                "process_infidelity":
                    accuracy[
                        "process_infidelity"
                    ],

                "phase_aligned_normalized_fro_error":
                    accuracy[
                        "phase_aligned_normalized_fro_error"
                    ],

                "global_phase_alignment_rad":
                    accuracy[
                        "global_phase_alignment_rad"
                    ],

                "logical_rzz_count":
                    (
                        len(
                            TOPOLOGY_EDGES[
                                topology
                            ]
                        )
                        *
                        trotter_r
                    ),

                "expected_cz_if_2_per_rzz":
                    (
                        2
                        *
                        len(
                            TOPOLOGY_EDGES[
                                topology
                            ]
                        )
                        *
                        trotter_r
                    ),

                "runtime_s":
                    runtime_s,
            }
        )

        for _, grid_row in forecast[
            "grid"
        ].iterrows():

            ridge_grid_rows.append(
                {
                    "candidate_id":
                        candidate_id,

                    "topology":
                        topology,

                    "hy":
                        hy,

                    "readout":
                        readout,

                    "trotter_r":
                        trotter_r,

                    "ridge_alpha":
                        float(
                            grid_row[
                                "ridge_alpha"
                            ]
                        ),

                    "cv_rmse_mean":
                        float(
                            grid_row[
                                "cv_rmse_mean"
                            ]
                        ),

                    "cv_rmse_std":
                        float(
                            grid_row[
                                "cv_rmse_std"
                            ]
                        ),

                    "fold_rmse":
                        json.dumps(
                            grid_row[
                                "fold_rmse"
                            ]
                        ),
                }
            )

        diagnostic_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "topology":
                    topology,

                "hy":
                    hy,

                "readout":
                    readout,

                "trotter_r":
                    trotter_r,

                "hamiltonian_hermiticity_error":
                    h_error,

                "exact_unitarity_error":
                    exact_unitarity_error,

                "trotter_unitarity_error":
                    trotter_unitarity_error,

                **diagnostics,
            }
        )

        print(
            f"r={trotter_r} | "
            f"CV={forecast['selected_cv_rmse_mean']:.6f} "
            f"(lam={forecast['selected_lambda']:g}) | "
            f"Val={forecast['selected_validation_rmse']:.6f} | "
            f"Fproc={accuracy['process_fidelity']:.8f} | "
            f"CZ_expected={10*trotter_r:2d} | "
            f"time={runtime_s:.1f}s"
        )


forecast_df = pd.DataFrame(
    forecast_rows
)

ridge_grid_df = pd.DataFrame(
    ridge_grid_rows
)

diagnostics_df = pd.DataFrame(
    diagnostic_rows
)

forecast_df.to_csv(
    RESULTS
    / "09_03_trotter_forecast_accuracy.csv",
    index=False,
)

ridge_grid_df.to_csv(
    RESULTS
    / "09_03_trotter_ridge_grid.csv",
    index=False,
)

diagnostics_df.to_csv(
    RESULTS
    / "09_03_trotter_diagnostics.csv",
    index=False,
)


# =============================================================================
# r=2 reproduction audit
# =============================================================================

r2 = forecast_df[
    forecast_df[
        "trotter_r"
    ] == REFERENCE_R
].copy()

r2_audit = r2.merge(
    candidates[
        [
            "candidate_id",
            "ridge_lambda",
            "cv_rmse",
            "validation_rmse",
        ]
    ],
    on="candidate_id",
    suffixes=(
        "_current",
        "_reference",
    ),
)

r2_audit[
    "delta_lambda"
] = (
    r2_audit[
        "selected_lambda"
    ]
    -
    r2_audit[
        "ridge_lambda"
    ]
)

r2_audit[
    "delta_cv_rmse"
] = (
    r2_audit[
        "cv_rmse_current"
    ]
    -
    r2_audit[
        "cv_rmse_reference"
    ]
)

r2_audit[
    "delta_validation_rmse"
] = (
    r2_audit[
        "validation_rmse_current"
    ]
    -
    r2_audit[
        "validation_rmse_reference"
    ]
)

r2_audit[
    "cv_reproduced_1e-5"
] = (
    np.abs(
        r2_audit[
            "delta_cv_rmse"
        ]
    )
    <=
    1e-5
)

r2_audit[
    "validation_reproduced_1e-5"
] = (
    np.abs(
        r2_audit[
            "delta_validation_rmse"
        ]
    )
    <=
    1e-5
)

r2_audit.to_csv(
    RESULTS
    / "09_03_r2_reproduction_audit.csv",
    index=False,
)

if not bool(
    r2_audit[
        "cv_reproduced_1e-5"
    ].all()
):
    raise RuntimeError(
        "r=2 forecast CV did not reproduce the frozen candidate table "
        "within 1e-5. Inspect 09_03_r2_reproduction_audit.csv before "
        "interpreting the r ablation."
    )


# =============================================================================
# Load physical embeddings
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

missing = [
    c
    for c in required_embedding_columns
    if c not in embeddings.columns
]

if missing:
    raise KeyError(
        f"Missing embedding columns: {missing}"
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
# IBM backends
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
# Calibration snapshot
#
# Prefer the already-frozen Step-9.2 snapshot.
# =============================================================================

if CALIBRATION_FILE.exists():

    calibration_df = pd.read_csv(
        CALIBRATION_FILE
    )

    CALIBRATION_SOURCE = str(
        CALIBRATION_FILE
    )

else:

    calibration_rows = []

    for backend_name, backend in backend_map.items():

        properties = backend.properties(
            refresh=True
        )

        backend_embeddings = embeddings[
            embeddings[
                "backend"
            ] == backend_name
        ]

        for _, embedding in backend_embeddings.iterrows():

            topology = str(
                embedding[
                    "candidate"
                ]
            )

            layout = list(
                embedding[
                    "layout"
                ]
            )

            t1_values = np.array(
                [
                    float(
                        properties.t1(
                            q
                        )
                    )
                    for q in layout
                ],
                dtype=float,
            )

            t2_values = np.array(
                [
                    float(
                        properties.t2(
                            q
                        )
                    )
                    for q in layout
                ],
                dtype=float,
            )

            ro_values = np.array(
                [
                    float(
                        properties.readout_error(
                            q
                        )
                    )
                    for q in layout
                ],
                dtype=float,
            )

            cz_errors = []

            for (
                logical_i,
                logical_j,
            ) in TOPOLOGY_EDGES[
                topology
            ]:

                physical_i = layout[
                    logical_i
                ]

                physical_j = layout[
                    logical_j
                ]

                error = np.nan

                for qargs in [
                    (
                        physical_i,
                        physical_j,
                    ),
                    (
                        physical_j,
                        physical_i,
                    ),
                ]:

                    try:
                        props = backend.target[
                            "cz"
                        ][
                            qargs
                        ]

                        if props is not None:
                            value = getattr(
                                props,
                                "error",
                                None,
                            )

                            if value is not None:
                                error = float(
                                    value
                                )
                                break

                    except Exception:
                        pass

                if np.isfinite(
                    error
                ):
                    cz_errors.append(
                        error
                    )

            calibration_rows.append(
                {
                    "backend":
                        backend_name,

                    "topology":
                        topology,

                    "embedding_rank":
                        int(
                            embedding[
                                "final_rank"
                            ]
                        ),

                    "layout":
                        json.dumps(
                            layout
                        ),

                    "current_min_t1_us":
                        float(
                            np.min(
                                t1_values
                            )
                            *
                            1e6
                        ),

                    "current_min_t2_us":
                        float(
                            np.min(
                                t2_values
                            )
                            *
                            1e6
                        ),

                    "current_max_readout_error_percent":
                        float(
                            np.max(
                                ro_values
                            )
                            *
                            100.0
                        ),

                    "current_max_cz_error_percent":
                        (
                            float(
                                np.max(
                                    cz_errors
                                )
                                *
                                100.0
                            )
                            if len(
                                cz_errors
                            ) > 0
                            else np.nan
                        ),
                }
            )

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    CALIBRATION_SOURCE = (
        "fresh backend.properties(refresh=True)"
    )

    calibration_df.to_csv(
        RESULTS
        / "09_03_calibration_snapshot_fallback.csv",
        index=False,
    )


required_calibration_columns = [
    "backend",
    "topology",
    "embedding_rank",
    "current_min_t1_us",
    "current_min_t2_us",
    "current_max_readout_error_percent",
    "current_max_cz_error_percent",
]

missing = [
    c
    for c in required_calibration_columns
    if c not in calibration_df.columns
]

if missing:
    raise KeyError(
        f"Calibration snapshot missing columns: {missing}"
    )


# =============================================================================
# Qiskit circuits
# =============================================================================

FIRST_INPUT_ANGLES = REAL_ANGLES[
    0
]


def build_core_circuit(
    topology,
    hy,
    trotter_r,
):
    qc = QuantumCircuit(
        N_QUBITS,
        name=(
            f"{topology}_hy_{hy:+.3f}_r{trotter_r}"
        ),
    )

    for q in range(
        N_INPUT
    ):
        qc.ry(
            float(
                FIRST_INPUT_ANGLES[
                    q
                ]
            ),
            q,
        )

    for _ in range(
        int(
            trotter_r
        )
    ):

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
                trotter_r
            )

            qc.rzz(
                theta_zz,
                i,
                j,
            )

        theta_x = (
            2.0
            *
            HX
            *
            DT
            /
            trotter_r
        )

        for q in range(
            N_QUBITS
        ):

            qc.rx(
                theta_x,
                q,
            )

        if abs(
            hy
        ) > 0.0:

            theta_y = (
                2.0
                *
                hy
                *
                DT
                /
                trotter_r
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
    qc = QuantumCircuit(
        N_QUBITS,
        N_QUBITS,
        name=(
            f"{core.name}_{setting}"
        ),
    )

    qc.compose(
        core,
        qubits=range(
            N_QUBITS
        ),
        inplace=True,
    )

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
                f"Unsupported basis: {basis}"
            )

    for q in range(
        N_QUBITS
    ):
        qc.measure(
            q,
            q,
        )

    return qc


def count_resources(
    circuit,
):
    counts = {
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

    excluded_1q = {
        "measure",
        "reset",
        "delay",
        "barrier",
    }

    n_1q = 0
    n_2q = 0

    for instruction in circuit.data:

        name = instruction.operation.name

        nq = len(
            instruction.qubits
        )

        if (
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
                    ==
                    2
            )
        )

    except TypeError:

        depth_1q = int(
            circuit.depth(
                filter_function=lambda inst:
                    (
                        len(
                            inst[
                                1
                            ]
                        )
                        == 1
                        and
                        inst[
                            0
                        ].name
                        not in excluded_1q
                    )
            )
        )

        depth_2q = int(
            circuit.depth(
                filter_function=lambda inst:
                    len(
                        inst[
                            1
                        ]
                    )
                    ==
                    2
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

        "n_cz":
            int(
                counts.get(
                    "cz",
                    0,
                )
            ),

        "n_swap":
            int(
                counts.get(
                    "swap",
                    0,
                )
            ),

        "n_measure":
            int(
                counts.get(
                    "measure",
                    0,
                )
            ),

        "n_delay":
            int(
                counts.get(
                    "delay",
                    0,
                )
            ),

        "operations":
            json.dumps(
                counts,
                sort_keys=True,
            ),
    }


def duration_seconds(
    circuit,
    backend,
):
    return float(
        circuit.estimate_duration(
            backend.target,
            unit="s",
        )
    )


# =============================================================================
# Physical compilation + scheduled timing
# =============================================================================

hardware_rows = []
measurement_rows = []

print(
    "\n"
    +
    "=" * 124
)

print(
    "PHYSICAL COMPILATION / TIMING"
)

print(
    "=" * 124
)

print(
    f"Schedule={SCHEDULING_LABEL}; "
    f"optimization levels={OPT_LEVELS}; "
    f"primary={PRIMARY_OPT_LEVELS}"
)


for _, candidate in candidates.iterrows():

    candidate_id = str(
        candidate[
            "candidate_id"
        ]
    )

    topology = str(
        candidate[
            "topology"
        ]
    )

    hy = float(
        candidate[
            "hy"
        ]
    )

    readout = str(
        candidate[
            "readout"
        ]
    )

    settings = READOUT_SETTINGS[
        readout
    ]

    candidate_embeddings = (
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

    for trotter_r in R_GRID:

        core = build_core_circuit(
            topology,
            hy,
            trotter_r,
        )

        expected_cz = (
            2
            *
            len(
                TOPOLOGY_EDGES[
                    topology
                ]
            )
            *
            trotter_r
        )

        for _, embedding in candidate_embeddings.iterrows():

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

            embedding_rank = int(
                embedding[
                    "final_rank"
                ]
            )

            calibration_match = calibration_df[
                (
                    calibration_df[
                        "backend"
                    ]
                    ==
                    backend_name
                )
                &
                (
                    calibration_df[
                        "topology"
                    ]
                    ==
                    topology
                )
                &
                (
                    calibration_df[
                        "embedding_rank"
                    ]
                    ==
                    embedding_rank
                )
            ]

            if len(
                calibration_match
            ) != 1:
                raise RuntimeError(
                    "Could not uniquely match calibration for "
                    f"{backend_name}/{topology}/rank{embedding_rank}"
                )

            calibration = calibration_match.iloc[
                0
            ]

            min_t1_us = float(
                calibration[
                    "current_min_t1_us"
                ]
            )

            min_t2_us = float(
                calibration[
                    "current_min_t2_us"
                ]
            )

            for optimization_level in OPT_LEVELS:

                kwargs = {
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

                    "scheduling_method":
                        SCHEDULING_METHOD,
                }

                try:

                    compiled_core = transpile(
                        core,
                        **kwargs,
                    )

                except Exception as exc:

                    raise RuntimeError(
                        "\nStrict native compilation failed:\n"
                        f"candidate={candidate_id}\n"
                        f"r={trotter_r}\n"
                        f"backend={backend_name}\n"
                        f"rank={embedding_rank}\n"
                        f"L={optimization_level}\n"
                        f"layout={layout}\n"
                        f"{exc}"
                    ) from exc

                resources = count_resources(
                    compiled_core
                )

                duration_s = duration_seconds(
                    compiled_core,
                    backend,
                )

                duration_us = (
                    duration_s
                    *
                    1e6
                )

                hardware_rows.append(
                    {
                        "candidate_id":
                            candidate_id,

                        "topology":
                            topology,

                        "y_state":
                            str(
                                candidate[
                                    "y_state"
                                ]
                            ),

                        "hy":
                            hy,

                        "readout":
                            readout,

                        "trotter_r":
                            trotter_r,

                        "n_qubits":
                            N_QUBITS,

                        "backend":
                            backend_name,

                        "embedding_rank":
                            embedding_rank,

                        "layout":
                            json.dumps(
                                layout
                            ),

                        "optimization_level":
                            optimization_level,

                        "is_primary_level":
                            optimization_level
                            in
                            PRIMARY_OPT_LEVELS,

                        "scheduling":
                            SCHEDULING_LABEL,

                        "expected_cz":
                            expected_cz,

                        "cz_matches_expected":
                            (
                                resources[
                                    "n_cz"
                                ]
                                ==
                                expected_cz
                            ),

                        "duration_s":
                            duration_s,

                        "duration_us":
                            duration_us,

                        "min_t1_us":
                            min_t1_us,

                        "min_t2_us":
                            min_t2_us,

                        "R_T1":
                            (
                                duration_us
                                /
                                min_t1_us
                            ),

                        "R_T2":
                            (
                                duration_us
                                /
                                min_t2_us
                            ),

                        "max_cz_error_percent":
                            float(
                                calibration[
                                    "current_max_cz_error_percent"
                                ]
                            ),

                        "max_readout_error_percent":
                            float(
                                calibration[
                                    "current_max_readout_error_percent"
                                ]
                            ),

                        **resources,
                    }
                )

                if resources[
                    "n_swap"
                ] != 0:

                    raise RuntimeError(
                        f"{candidate_id}, r={trotter_r}, "
                        f"{backend_name}, rank={embedding_rank}, "
                        f"L{optimization_level}: unexpected SWAP="
                        f"{resources['n_swap']}"
                    )

                # -------------------------------------------------------------
                # Selected grouped-measurement circuits
                # -------------------------------------------------------------

                for setting in settings:

                    measurement_circuit = (
                        build_measurement_circuit(
                            core,
                            setting,
                        )
                    )

                    compiled_measurement = transpile(
                        measurement_circuit,
                        **kwargs,
                    )

                    measurement_resources = count_resources(
                        compiled_measurement
                    )

                    measurement_duration_s = duration_seconds(
                        compiled_measurement,
                        backend,
                    )

                    measurement_duration_us = (
                        measurement_duration_s
                        *
                        1e6
                    )

                    measurement_rows.append(
                        {
                            "candidate_id":
                                candidate_id,

                            "topology":
                                topology,

                            "y_state":
                                str(
                                    candidate[
                                        "y_state"
                                    ]
                                ),

                            "hy":
                                hy,

                            "readout":
                                readout,

                            "measurement_setting":
                                setting,

                            "trotter_r":
                                trotter_r,

                            "backend":
                                backend_name,

                            "embedding_rank":
                                embedding_rank,

                            "layout":
                                json.dumps(
                                    layout
                                ),

                            "optimization_level":
                                optimization_level,

                            "is_primary_level":
                                optimization_level
                                in
                                PRIMARY_OPT_LEVELS,

                            "scheduling":
                                SCHEDULING_LABEL,

                            "shots_per_setting":
                                SHOTS_PER_SETTING,

                            "duration_per_shot_us":
                                measurement_duration_us,

                            "duration_1024_shots_s":
                                (
                                    measurement_duration_s
                                    *
                                    SHOTS_PER_SETTING
                                ),

                            "R_T1":
                                (
                                    measurement_duration_us
                                    /
                                    min_t1_us
                                ),

                            "R_T2":
                                (
                                    measurement_duration_us
                                    /
                                    min_t2_us
                                ),

                            **measurement_resources,
                        }
                    )


hardware_df = pd.DataFrame(
    hardware_rows
)

measurement_df = pd.DataFrame(
    measurement_rows
)

hardware_df.to_csv(
    RESULTS
    / "09_03_hardware_all_levels.csv",
    index=False,
)

measurement_df.to_csv(
    RESULTS
    / "09_03_measurement_all_levels.csv",
    index=False,
)


# =============================================================================
# Feature-vector timing
# =============================================================================

feature_vector_df = (
    measurement_df
    .groupby(
        [
            "candidate_id",
            "topology",
            "y_state",
            "hy",
            "readout",
            "trotter_r",
            "backend",
            "embedding_rank",
            "layout",
            "optimization_level",
            "is_primary_level",
            "scheduling",
        ],
        as_index=False,
    )
    .agg(
        n_settings=(
            "measurement_setting",
            "nunique",
        ),

        longest_setting_duration_us=(
            "duration_per_shot_us",
            "max",
        ),

        total_feature_vector_duration_us=(
            "duration_per_shot_us",
            "sum",
        ),

        estimated_1024shot_circuit_time_s=(
            "duration_1024_shots_s",
            "sum",
        ),

        max_measurement_R_T1=(
            "R_T1",
            "max",
        ),

        max_measurement_R_T2=(
            "R_T2",
            "max",
        ),

        total_cz_per_feature_vector=(
            "n_cz",
            "sum",
        ),

        max_swap_per_setting=(
            "n_swap",
            "max",
        ),
    )
)

feature_vector_df.to_csv(
    RESULTS
    / "09_03_feature_vector_timing_all.csv",
    index=False,
)


# =============================================================================
# Merge forecasting + physical metrics
# =============================================================================

hardware_with_forecast = hardware_df.merge(
    forecast_df[
        [
            "candidate_id",
            "trotter_r",
            "selected_lambda",
            "cv_rmse",
            "cv_rmse_std",
            "validation_rmse",
            "process_fidelity",
            "process_infidelity",
            "phase_aligned_normalized_fro_error",
        ]
    ],
    on=[
        "candidate_id",
        "trotter_r",
    ],
    how="left",
)

feature_with_forecast = feature_vector_df.merge(
    forecast_df[
        [
            "candidate_id",
            "trotter_r",
            "selected_lambda",
            "cv_rmse",
            "cv_rmse_std",
            "validation_rmse",
            "process_fidelity",
            "process_infidelity",
            "phase_aligned_normalized_fro_error",
        ]
    ],
    on=[
        "candidate_id",
        "trotter_r",
    ],
    how="left",
)

hardware_with_forecast.to_csv(
    RESULTS
    / "09_03_hardware_with_forecast.csv",
    index=False,
)

feature_with_forecast.to_csv(
    RESULTS
    / "09_03_feature_vector_with_forecast.csv",
    index=False,
)


# =============================================================================
# Primary L0/L1 summaries
# =============================================================================

hardware_primary = hardware_with_forecast[
    hardware_with_forecast[
        "optimization_level"
    ].isin(
        PRIMARY_OPT_LEVELS
    )
].copy()

feature_primary = feature_with_forecast[
    feature_with_forecast[
        "optimization_level"
    ].isin(
        PRIMARY_OPT_LEVELS
    )
].copy()


hardware_summary = (
    hardware_primary
    .groupby(
        [
            "candidate_id",
            "trotter_r",
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

        n_qubits=(
            "n_qubits",
            "first",
        ),

        cz_median=(
            "n_cz",
            "median",
        ),

        swap_median=(
            "n_swap",
            "median",
        ),

        oneq_median=(
            "n_1q",
            "median",
        ),

        depth_median=(
            "depth",
            "median",
        ),

        depth_1q_median=(
            "depth_1q",
            "median",
        ),

        depth_2q_median=(
            "depth_2q",
            "median",
        ),

        duration_us_median=(
            "duration_us",
            "median",
        ),

        duration_us_max=(
            "duration_us",
            "max",
        ),

        R_T1_median=(
            "R_T1",
            "median",
        ),

        R_T2_median=(
            "R_T2",
            "median",
        ),

        R_T2_max=(
            "R_T2",
            "max",
        ),

        cv_rmse=(
            "cv_rmse",
            "first",
        ),

        validation_rmse=(
            "validation_rmse",
            "first",
        ),

        selected_lambda=(
            "selected_lambda",
            "first",
        ),

        process_fidelity=(
            "process_fidelity",
            "first",
        ),

        process_infidelity=(
            "process_infidelity",
            "first",
        ),
    )
)

hardware_summary.to_csv(
    RESULTS
    / "09_03_hardware_summary_primary.csv",
    index=False,
)


feature_summary = (
    feature_primary
    .groupby(
        [
            "candidate_id",
            "readout",
            "trotter_r",
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

        n_settings=(
            "n_settings",
            "first",
        ),

        longest_setting_duration_us_median=(
            "longest_setting_duration_us",
            "median",
        ),

        total_feature_vector_duration_us_median=(
            "total_feature_vector_duration_us",
            "median",
        ),

        estimated_1024shot_circuit_time_s_median=(
            "estimated_1024shot_circuit_time_s",
            "median",
        ),

        max_measurement_R_T2_median=(
            "max_measurement_R_T2",
            "median",
        ),

        total_cz_per_feature_vector_median=(
            "total_cz_per_feature_vector",
            "median",
        ),

        cv_rmse=(
            "cv_rmse",
            "first",
        ),

        validation_rmse=(
            "validation_rmse",
            "first",
        ),

        selected_lambda=(
            "selected_lambda",
            "first",
        ),

        process_fidelity=(
            "process_fidelity",
            "first",
        ),
    )
)

feature_summary.to_csv(
    RESULTS
    / "09_03_feature_vector_summary_primary.csv",
    index=False,
)


# =============================================================================
# r-wise deltas relative to r=2
# =============================================================================

reference = forecast_df[
    forecast_df[
        "trotter_r"
    ] == REFERENCE_R
][
    [
        "candidate_id",
        "cv_rmse",
        "validation_rmse",
        "process_fidelity",
    ]
].rename(
    columns={
        "cv_rmse":
            "cv_rmse_r2",

        "validation_rmse":
            "validation_rmse_r2",

        "process_fidelity":
            "process_fidelity_r2",
    }
)

r_delta = forecast_df.merge(
    reference,
    on="candidate_id",
    how="left",
)

r_delta[
    "delta_cv_vs_r2"
] = (
    r_delta[
        "cv_rmse"
    ]
    -
    r_delta[
        "cv_rmse_r2"
    ]
)

r_delta[
    "delta_validation_vs_r2"
] = (
    r_delta[
        "validation_rmse"
    ]
    -
    r_delta[
        "validation_rmse_r2"
    ]
)

r_delta[
    "delta_process_fidelity_vs_r2"
] = (
    r_delta[
        "process_fidelity"
    ]
    -
    r_delta[
        "process_fidelity_r2"
    ]
)

r_delta.to_csv(
    RESULTS
    / "09_03_r_deltas_vs_r2.csv",
    index=False,
)


# =============================================================================
# Pareto trade-off over r
#
# Per candidate/backend/optimization level:
# minimize:
#   - CV RMSE
#   - process infidelity
#   - physical core duration
#   - CZ count
#   - R_T2
#
# No weighted score.
# =============================================================================

def pareto_mask(
    frame,
    columns,
):
    values = frame[
        columns
    ].to_numpy(
        dtype=float
    )

    finite_rows = np.all(
        np.isfinite(
            values
        ),
        axis=1,
    )

    keep = np.zeros(
        len(
            frame
        ),
        dtype=bool,
    )

    valid_indices = np.where(
        finite_rows
    )[0]

    valid_values = values[
        finite_rows
    ]

    for local_i, global_i in enumerate(
        valid_indices
    ):

        v = valid_values[
            local_i
        ]

        no_worse = np.all(
            valid_values
            <=
            v,
            axis=1,
        )

        strictly_better = np.any(
            valid_values
            <
            v,
            axis=1,
        )

        dominates = (
            no_worse
            &
            strictly_better
        )

        dominates[
            local_i
        ] = False

        if not np.any(
            dominates
        ):
            keep[
                global_i
            ] = True

    return keep


# Use embedding-level hardware results, but only optimization levels 0/1.
pareto_groups = []

for (
    candidate_id,
    backend_name,
    embedding_rank,
    optimization_level,
), group in hardware_primary.groupby(
    [
        "candidate_id",
        "backend",
        "embedding_rank",
        "optimization_level",
    ],
    sort=True,
):

    group = (
        group
        .copy()
        .reset_index(
            drop=True
        )
    )

    group[
        "is_r_pareto"
    ] = pareto_mask(
        group,
        [
            "cv_rmse",
            "process_infidelity",
            "duration_us",
            "n_cz",
            "R_T2",
        ],
    )

    pareto_groups.append(
        group
    )


r_pareto_all = pd.concat(
    pareto_groups,
    ignore_index=True,
)

r_pareto_candidates = r_pareto_all[
    r_pareto_all[
        "is_r_pareto"
    ]
].copy()

r_pareto_all.to_csv(
    RESULTS
    / "09_03_r_pareto_all.csv",
    index=False,
)

r_pareto_candidates.to_csv(
    RESULTS
    / "09_03_r_pareto_candidates.csv",
    index=False,
)


# =============================================================================
# Best r by individual metric
# =============================================================================

best_rows = []

for candidate_id, group in forecast_df.groupby(
    "candidate_id",
    sort=True,
):

    for (
        metric_name,
        column,
        minimize,
    ) in [
        (
            "CV_RMSE",
            "cv_rmse",
            True,
        ),
        (
            "validation_RMSE",
            "validation_rmse",
            True,
        ),
        (
            "process_infidelity",
            "process_infidelity",
            True,
        ),
    ]:

        if minimize:
            idx = group[
                column
            ].idxmin()

        else:
            idx = group[
                column
            ].idxmax()

        row = group.loc[
            idx
        ]

        best_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "metric":
                    metric_name,

                "best_r":
                    int(
                        row[
                            "trotter_r"
                        ]
                    ),

                "value":
                    float(
                        row[
                            column
                        ]
                    ),

                "selected_lambda":
                    float(
                        row[
                            "selected_lambda"
                        ]
                    ),
            }
        )


best_forecast_metric_df = pd.DataFrame(
    best_rows
)

best_forecast_metric_df.to_csv(
    RESULTS
    / "09_03_best_r_by_forecast_accuracy_metric.csv",
    index=False,
)


# =============================================================================
# Manifest
# =============================================================================

manifest = {
    "step":
        "Week 9 Step 9.3",

    "experiment":
        "Trotter-depth / hardware-cost ablation",

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "r_grid":
        R_GRID,

    "reference_r":
        REFERENCE_R,

    "candidate_count":
        int(
            len(
                candidates
            )
        ),

    "candidate_source":
        str(
            CANDIDATE_FILE
        ),

    "J_source":
        str(
            J_WINNER_FILE
        ),

    "J_source_original_r":
        REFERENCE_R,

    "J_reoptimized_per_r":
        False,

    "hy_reoptimized_per_r":
        False,

    "readout_reoptimized_per_r":
        False,

    "ridge_lambda_reselected_per_r":
        True,

    "ridge_grid":
        [
            float(
                x
            )
            for x in RIDGE_GRID
        ],

    "alpha":
        ALPHA,

    "hx":
        HX,

    "dt":
        DT,

    "optimization_levels":
        OPT_LEVELS,

    "primary_optimization_levels":
        PRIMARY_OPT_LEVELS,

    "scheduling":
        SCHEDULING_LABEL,

    "shots_per_setting":
        SHOTS_PER_SETTING,

    "calibration_source":
        CALIBRATION_SOURCE,

    "Trotter_accuracy":
        {
            "exact_unitary":
                "exp(-i H dt) from Hermitian eigendecomposition",

            "process_fidelity":
                "|Tr(U_exact^dagger U_r)|^2 / d^2",

            "phase_aligned_fro_error":
                "||U_exact - e^{-i phi} U_r||_F / sqrt(d)",
        },

    "interpretation":
        (
            "Controlled robustness/hardware ablation. "
            "J, h_y and readout are frozen from the retained design, "
            "which was selected around r=2. Therefore results do not "
            "represent a globally reoptimized reservoir at each r."
        ),

    "weighted_score":
        False,
}

with open(
    RESULTS
    / "09_03_manifest.json",
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
    "=" * 124
)

print(
    "r=2 REPRODUCTION AUDIT"
)

print(
    "=" * 124
)

print(
    r2_audit[
        [
            "candidate_id",
            "selected_lambda",
            "ridge_lambda",
            "cv_rmse_current",
            "cv_rmse_reference",
            "delta_cv_rmse",
            "validation_rmse_current",
            "validation_rmse_reference",
            "delta_validation_rmse",
            "cv_reproduced_1e-5",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 124
)

print(
    "FORECAST + TROTTER ACCURACY"
)

print(
    "=" * 124
)

print(
    forecast_df[
        [
            "candidate_id",
            "trotter_r",
            "selected_lambda",
            "cv_rmse",
            "validation_rmse",
            "process_fidelity",
            "process_infidelity",
            "phase_aligned_normalized_fro_error",
            "logical_rzz_count",
            "expected_cz_if_2_per_rzz",
        ]
    ]
    .sort_values(
        [
            "candidate_id",
            "trotter_r",
        ]
    )
    .to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 124
)

print(
    "PHYSICAL CORE SUMMARY — PRIMARY LEVELS 0/1"
)

print(
    "=" * 124
)

print(
    hardware_summary[
        [
            "candidate_id",
            "trotter_r",
            "backend",
            "optimization_level",
            "n_qubits",
            "cz_median",
            "swap_median",
            "oneq_median",
            "depth_median",
            "depth_2q_median",
            "duration_us_median",
            "R_T1_median",
            "R_T2_median",
            "cv_rmse",
            "process_fidelity",
        ]
    ]
    .sort_values(
        [
            "backend",
            "candidate_id",
            "trotter_r",
            "optimization_level",
        ]
    )
    .to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 124
)

print(
    "FEATURE-VECTOR TIMING — PRIMARY LEVELS 0/1"
)

print(
    "=" * 124
)

print(
    feature_summary[
        [
            "candidate_id",
            "readout",
            "trotter_r",
            "backend",
            "optimization_level",
            "n_settings",
            "longest_setting_duration_us_median",
            "total_feature_vector_duration_us_median",
            "estimated_1024shot_circuit_time_s_median",
            "total_cz_per_feature_vector_median",
            "max_measurement_R_T2_median",
            "cv_rmse",
            "process_fidelity",
        ]
    ]
    .sort_values(
        [
            "backend",
            "candidate_id",
            "trotter_r",
            "optimization_level",
        ]
    )
    .to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 124
)

print(
    "BEST r BY FORECAST / TROTTER-ACCURACY METRIC"
)

print(
    "=" * 124
)

print(
    best_forecast_metric_df.to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 124
)

print(
    "r-PARETO CANDIDATES — PRIMARY LEVELS"
)

print(
    "=" * 124
)

print(
    r_pareto_candidates[
        [
            "candidate_id",
            "backend",
            "embedding_rank",
            "optimization_level",
            "trotter_r",
            "cv_rmse",
            "process_infidelity",
            "duration_us",
            "n_cz",
            "R_T2",
        ]
    ]
    .sort_values(
        [
            "backend",
            "candidate_id",
            "embedding_rank",
            "optimization_level",
            "trotter_r",
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
    "09_03_J_source_audit.csv",
    "09_03_trotter_forecast_accuracy.csv",
    "09_03_trotter_ridge_grid.csv",
    "09_03_trotter_diagnostics.csv",
    "09_03_r2_reproduction_audit.csv",
    "09_03_hardware_all_levels.csv",
    "09_03_measurement_all_levels.csv",
    "09_03_feature_vector_timing_all.csv",
    "09_03_hardware_with_forecast.csv",
    "09_03_feature_vector_with_forecast.csv",
    "09_03_hardware_summary_primary.csv",
    "09_03_feature_vector_summary_primary.csv",
    "09_03_r_deltas_vs_r2.csv",
    "09_03_r_pareto_all.csv",
    "09_03_r_pareto_candidates.csv",
    "09_03_best_r_by_forecast_accuracy_metric.csv",
    "09_03_manifest.json",
]:
    print(
        f"  results/{filename}"
    )


print(
    "\nStep 9.3 complete."
)

print(
    f"Forecast/Trotter section runtime: "
    f"{time.perf_counter() - forecast_start:.1f}s "
    "(hardware compilation time included after the forecast loop)."
)
