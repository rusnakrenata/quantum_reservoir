"""
Week 8 - Step 8.3B
Hardware-native topology-specific J optimization.

QUESTION
--------
For each hardware-native logical reservoir H0-H4:

    Which independent edge couplings J_ij maximize intrinsic
    delayed-input memory while preserving acceptable fading memory?

Then:

    Do the strongest memory reservoirs also perform well on
    next-day insurance forecasting?

FROZEN
------
alpha       = 0.75
hx          = 0.5
Trotter r   = 2
dt          = 1.6
seed        = 42

MC readout:
    XYZ_all

Forecast readouts:
    XYZ_all
    XZ_injection

SEARCH
------
Each topology has exactly five logical edges.

Stage A:
    32 global J vectors
    - 28 deterministic Latin-hypercube samples
    - 4 fixed anchor vectors

    J_ij in [-2, +2]

Stage B:
    refine the best valid Stage-A MC candidate
    one edge at a time using

        delta J in {-0.50,-0.25,+0.25,+0.50}

    giving at most 20 additional candidates.

Thus approximately:
    52 candidates / topology
    260 candidates total

Forecasting:
    top 5 valid MC candidates / topology only.

SELECTION
---------
Primary:
    maximize post-washout corrected MC per channel

Constraint:
    finite Tw(0.01) <= 300

Secondary diagnostics:
    MC delays 1,2,5,10,20

No artificial MC/RMSE combined score is used.

DEPENDENCY
----------
Keep beside:
    07_02b_memory_pair_isolation.py

OUTPUTS
-------
results/08_03b_J_search_all.csv
results/08_03b_J_search_ranked.csv
results/08_03b_J_search_topology_winners.csv
results/08_03b_J_search_by_delay.csv
results/08_03b_J_search_forecast_finalists.csv
results/08_03b_J_search_summary.txt
results/08_03b_J_search_manifest.json
results/08_03b_checkpoint.csv
"""

from __future__ import annotations

import importlib.util
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


# =============================================================================
# PATHS / DEPENDENCY
# =============================================================================

HERE = Path(__file__).resolve().parent

RESULTS = Path("results")
RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)

BASE = (
    HERE
    / "07_02b_memory_pair_isolation.py"
)

if not BASE.exists():

    raise FileNotFoundError(
        f"Missing {BASE.name}. "
        f"Keep it beside this script."
    )


def load_module(path, name):

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None

    spec.loader.exec_module(
        module
    )

    return module


qrc = load_module(
    BASE,
    "qrc_week8_J_search",
)


# =============================================================================
# FROZEN GLOBAL QRC SETTINGS
# =============================================================================

SEED = 42

ALPHA = 0.75
HX = 0.5
DT = 1.6
TROTTER_R = 2

PROBE_SEED = 79001
J_SEARCH_SEED = 84201

K_MAX = 20

MIN_BURNIN = 100
POST_WASHOUT_MARGIN = 20
MAX_ACCEPTABLE_WASHOUT = 300

MC_RIDGE_ALPHA = 1e-6
FORECAST_RIDGE_ALPHA = 0.01

CV_FOLDS = 5
VAR_TOL = 1e-12

N_GLOBAL_TOTAL = 32
N_GLOBAL_RANDOM = 28

J_GLOBAL_MIN = -2.0
J_GLOBAL_MAX = +2.0

LOCAL_DELTAS = [
    -0.50,
    -0.25,
    +0.25,
    +0.50,
]

J_LOCAL_MIN = -2.5
J_LOCAL_MAX = +2.5

N_FORECAST_FINALISTS = 5


qrc.ALPHA = ALPHA
qrc.HX = HX
qrc.DT = DT
qrc.TROTTER_R = TROTTER_R
qrc.SEEDS = [SEED]


# =============================================================================
# LOGICAL ROLES
# =============================================================================

LOGICAL_ROLES = {
    0: "C_t",
    1: "D",
    2: "P_t",
    3: "H",
    4: "M1",
    5: "M2",
}


# =============================================================================
# HARDWARE-NATIVE LOGICAL TOPOLOGIES
# =============================================================================

CANDIDATES = {

    # C-D-P-H-M1-M2
    "H0": {
        "description":
            "Original native chain C-D-P-H-M1-M2",

        "edges": [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ],
    },

    # C-D-P-H with H -> M1 and H -> M2
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

    # C-D-P-H with P -> M1 -> M2
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

    # H-P-D-C-M1-M2
    "H3": {
        "description":
            "Task-aware chain with C_t at memory boundary",

        "edges": [
            (3, 2),
            (2, 1),
            (1, 0),
            (0, 4),
            (4, 5),
        ],
    },

    # P-D-C-M1-M2, H -> M1
    "H4": {
        "description":
            "Memory hub: C_t and H directly feed M1",

        "edges": [
            (2, 1),
            (1, 0),
            (0, 4),
            (3, 4),
            (4, 5),
        ],
    },
}


# =============================================================================
# EDGE HELPERS
# =============================================================================

def canonical_edge(a, b):

    return tuple(
        sorted(
            (
                int(a),
                int(b),
            )
        )
    )


for topology in CANDIDATES.values():

    topology["edges"] = [
        canonical_edge(
            a,
            b,
        )
        for a, b
        in topology["edges"]
    ]


ALL_EDGES = sorted(
    {
        edge
        for topology
        in CANDIDATES.values()
        for edge
        in topology["edges"]
    }
)


def edge_name(edge):

    return (
        f"J{edge[0]}{edge[1]}"
    )


# =============================================================================
# PRECOMPUTED ZZ OPERATORS
# =============================================================================

ZZ_OPS = {

    edge:
        qrc.pauli_product_full(
            qrc.Z2,
            edge[0],
            qrc.Z2,
            edge[1],
        )

    for edge in ALL_EDGES
}


# =============================================================================
# OBSERVABLES
# =============================================================================

XYZ_ALL = [

    "X0", "X1", "X2",
    "X3", "X4", "X5",

    "Y0", "Y1", "Y2",
    "Y3", "Y4", "Y5",

    "Z0", "Z1", "Z2",
    "Z3", "Z4", "Z5",
]


XZ_INJECTION = [

    "X0", "X1", "X2", "X3",

    "Z0", "Z1", "Z2", "Z3",
]


FORECAST_FAMILIES = {

    "XYZ_all":
        XYZ_ALL,

    "XZ_injection":
        XZ_INJECTION,
}


# =============================================================================
# INITIAL MEMORY STATES
# =============================================================================

def ket_density(vector):

    vector = np.asarray(
        vector,
        dtype=complex,
    )

    vector = (
        vector
        /
        np.linalg.norm(
            vector
        )
    )

    return np.outer(
        vector,
        vector.conj(),
    )


e00 = np.array(
    [1, 0, 0, 0],
    dtype=complex,
)

e11 = np.array(
    [0, 0, 0, 1],
    dtype=complex,
)

epp = (
    np.array(
        [1, 1, 1, 1],
        dtype=complex,
    )
    / 2.0
)

ebell = (
    np.array(
        [1, 0, 0, 1],
        dtype=complex,
    )
    / np.sqrt(2.0)
)


INITIAL_STATES = {

    "00":
        ket_density(
            e00
        ),

    "11":
        ket_density(
            e11
        ),

    "++":
        ket_density(
            epp
        ),

    "BellPhi+":
        ket_density(
            ebell
        ),

    "I/4":
        np.eye(
            4,
            dtype=complex,
        )
        / 4.0,
}


RESERVOIR_INITIAL_STATE = (
    INITIAL_STATES[
        "00"
    ]
)


# =============================================================================
# BASIC MATH
# =============================================================================

def trace_distance(
    rho,
    sigma,
):

    delta = np.asarray(
        rho - sigma,
        dtype=complex,
    )

    delta = (
        0.5
        *
        (
            delta
            +
            delta.conj().T
        )
    )

    vals = np.linalg.eigvalsh(
        delta
    )

    return (
        0.5
        *
        float(
            np.sum(
                np.abs(
                    vals
                )
            )
        )
    )


def max_pairwise_trace_distance(
    states,
):

    return max(

        trace_distance(
            states[a],
            states[b],
        )

        for a, b
        in combinations(
            states.keys(),
            2,
        )
    )


def stable_threshold_crossing(
    values,
    threshold,
):

    arr = np.asarray(
        values,
        dtype=float,
    )

    suffix_max = (
        np.maximum.accumulate(
            arr[::-1]
        )[::-1]
    )

    idx = np.flatnonzero(
        suffix_max
        <=
        threshold
    )

    if len(idx) == 0:

        return None

    return int(
        idx[0]
    )


def fit_kappa(
    D_curve,
):

    D = np.asarray(
        D_curve,
        dtype=float,
    )


    idx = np.flatnonzero(

        (
            np.arange(
                len(D)
            )
            >=
            5
        )

        &

        (
            D
            >
            1e-10
        )

        &

        np.isfinite(
            D
        )
    )


    if len(idx) < 20:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    t = idx.astype(
        float
    )

    y = np.log(
        D[idx]
    )


    slope, intercept = (
        np.polyfit(
            t,
            y,
            1,
        )
    )


    pred = (
        intercept
        +
        slope * t
    )


    ss_res = float(
        np.sum(
            (
                y - pred
            )
            ** 2
        )
    )


    ss_tot = float(
        np.sum(
            (
                y
                -
                np.mean(y)
            )
            ** 2
        )
    )


    r2 = (

        1.0
        -
        ss_res
        /
        ss_tot

        if ss_tot > 0

        else np.nan
    )


    kappa = float(
        np.exp(
            slope
        )
    )


    tau = (

        float(
            -1.0
            /
            np.log(
                kappa
            )
        )

        if (
            0.0
            <
            kappa
            <
            1.0
        )

        else np.inf
    )


    return (
        kappa,
        tau,
        r2,
    )


def rmse(
    y_true,
    y_pred,
):

    return float(

        np.sqrt(

            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


# =============================================================================
# LATIN-HYPERCUBE GLOBAL J SEARCH
# =============================================================================

def latin_hypercube(
    n_samples,
    n_dimensions,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    result = np.empty(
        (
            n_samples,
            n_dimensions,
        ),
        dtype=float,
    )


    for dimension in range(
        n_dimensions
    ):

        permutation = (
            rng.permutation(
                n_samples
            )
        )

        jitter = (
            rng.random(
                n_samples
            )
        )


        result[
            :,
            dimension
        ] = (
            permutation
            +
            jitter
        ) / n_samples


    return result


def global_J_vectors():
    """
    Same candidate J vectors are used for every topology.

    This guarantees an equal and directly comparable
    global search budget.
    """

    lhs = latin_hypercube(

        n_samples=
            N_GLOBAL_RANDOM,

        n_dimensions=5,

        seed=
            J_SEARCH_SEED,
    )


    lhs = (

        J_GLOBAL_MIN

        +

        lhs
        *
        (
            J_GLOBAL_MAX
            -
            J_GLOBAL_MIN
        )
    )


    anchors = np.array(
        [
            [-1, -1, -1, -1, -1],
            [+1, +1, +1, +1, +1],
            [-0.5, -0.5, -0.5, -0.5, -0.5],
            [+0.5, +0.5, +0.5, +0.5, +0.5],
        ],
        dtype=float,
    )


    vectors = np.vstack(
        [
            anchors,
            lhs,
        ]
    )


    assert (
        len(vectors)
        ==
        N_GLOBAL_TOTAL
    )


    return vectors


GLOBAL_J_VECTORS = (
    global_J_vectors()
)


# =============================================================================
# UNITARY
# =============================================================================

def J_dict_from_vector(
    topology_name,
    vector,
):

    edges = (
        CANDIDATES[
            topology_name
        ][
            "edges"
        ]
    )


    return {

        edge:
            float(
                value
            )

        for edge, value
        in zip(
            edges,
            vector,
        )
    }


def build_unitary(
    topology_name,
    J,
):

    edges = (
        CANDIDATES[
            topology_name
        ][
            "edges"
        ]
    )


    U = (
        qrc.I64.copy()
    )


    for _ in range(
        TROTTER_R
    ):

        # -----------------------------------------------------
        # ZZ part
        # -----------------------------------------------------

        for edge in edges:

            theta = (

                2.0

                *
                J[edge]

                *
                DT

                /
                TROTTER_R
            )


            ZZ = (
                ZZ_OPS[
                    edge
                ]
            )


            gate = (

                np.cos(
                    theta / 2.0
                )
                *
                qrc.I64

                -

                1j
                *
                np.sin(
                    theta / 2.0
                )
                *
                ZZ
            )


            U = (
                gate
                @
                U
            )


        # -----------------------------------------------------
        # Transverse X field
        # -----------------------------------------------------

        theta_x = (

            2.0
            *
            HX
            *
            DT
            /
            TROTTER_R
        )


        for qubit in range(
            qrc.N_QUBITS
        ):

            Xq = (
                qrc.FULL_SINGLE_OPS[
                    f"X{qubit}"
                ]
            )


            gate = (

                np.cos(
                    theta_x / 2.0
                )
                *
                qrc.I64

                -

                1j
                *
                np.sin(
                    theta_x / 2.0
                )
                *
                Xq
            )


            U = (
                gate
                @
                U
            )


    error = np.linalg.norm(

        U.conj().T
        @
        U

        -
        qrc.I64,

        ord="fro",
    )


    return (
        U,
        float(error),
    )


# =============================================================================
# PROBE
# =============================================================================

def build_probe(
    real_angles,
):

    rng = np.random.default_rng(
        PROBE_SEED
    )


    real_angles = np.asarray(
        real_angles,
        dtype=float,
    )


    probe = np.empty_like(
        real_angles
    )


    for channel in range(
        real_angles.shape[1]
    ):

        probe[
            :,
            channel
        ] = (

            real_angles[

                rng.permutation(
                    len(
                        real_angles
                    )
                ),

                channel,
            ]
        )


    return probe


def probe_audit(
    probe,
):

    max_abs = 0.0


    for channel in range(
        probe.shape[1]
    ):

        for lag in range(
            1,
            21,
        ):

            corr = np.corrcoef(

                probe[
                    lag:,
                    channel
                ],

                probe[
                    :-lag,
                    channel
                ],
            )[0, 1]


            if np.isfinite(
                corr
            ):

                max_abs = max(
                    max_abs,
                    abs(
                        float(
                            corr
                        )
                    ),
                )


    return max_abs


# =============================================================================
# FEATURES
# =============================================================================

def add_Y_features(
    row,
    rho_i,
    rho_m,
):

    for qubit in range(
        qrc.N_INJECTION
    ):

        name = (
            f"Y{qubit}"
        )

        row[name] = (
            qrc.expectation(

                rho_i,

                qrc.INJECTION_SINGLE_OPS[
                    name
                ],
            )
        )


    for qubit in (
        qrc.MEMORY_QUBITS
    ):

        name = (
            f"Y{qubit}"
        )

        row[name] = (
            qrc.expectation(

                rho_m,

                qrc.MEMORY_SINGLE_OPS[
                    name
                ],
            )
        )


    return row


def build_feature_bank(
    A_list,
):

    rho = (
        RESERVOIR_INITIAL_STATE.copy()
    )


    rows = []


    for A in A_list:

        rho_before = rho


        rho_i, rho_m = (
            qrc.final_reduced_states(
                A,
                rho_before,
            )
        )


        row = (
            qrc.extract_feature_row(
                A,
                rho_before,
                rho_i,
                rho_m,
            )
        )


        row = add_Y_features(
            row,
            rho_i,
            rho_m,
        )


        rows.append(
            row
        )


        rho = rho_m


    return pd.DataFrame(
        rows
    )


# =============================================================================
# WASHOUT
# =============================================================================

def washout_diagnostic(
    S_list,
):

    states = {

        name:
            rho.copy()

        for name, rho
        in INITIAL_STATES.items()
    }


    curve = []


    for S in S_list:

        states = {

            name:
                qrc.apply_memory_channel(
                    S,
                    rho,
                )

            for name, rho
            in states.items()
        }


        curve.append(

            max_pairwise_trace_distance(
                states
            )
        )


    curve = np.asarray(
        curve,
        dtype=float,
    )


    Tw = (
        stable_threshold_crossing(
            curve,
            0.01,
        )
    )


    kappa, tau, r2 = (
        fit_kappa(
            curve
        )
    )


    return {

        "Tw_0.01":
            (
                np.nan
                if Tw is None
                else Tw
            ),

        "D_final":
            float(
                curve[-1]
            ),

        "kappa_eff":
            kappa,

        "tau_mem":
            tau,

        "kappa_fit_R2":
            r2,
    }


# =============================================================================
# RIDGE
# =============================================================================

def standardized_ridge_predict(
    X_train,
    y_train,
    X_validation,
    alpha,
):

    X_train = np.asarray(
        X_train,
        dtype=float,
    )

    X_validation = np.asarray(
        X_validation,
        dtype=float,
    )

    y_train = np.asarray(
        y_train,
        dtype=float,
    )


    mean = np.mean(
        X_train,
        axis=0,
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

        return np.full(
            len(
                X_validation
            ),
            float(
                np.mean(
                    y_train
                )
            ),
        )


    X_train_z = (

        X_train[
            :,
            keep
        ]

        -
        mean[
            keep
        ]

    ) / std[
        keep
    ]


    X_validation_z = (

        X_validation[
            :,
            keep
        ]

        -
        mean[
            keep
        ]

    ) / std[
        keep
    ]


    model = Ridge(

        alpha=
            alpha,

        fit_intercept=
            True,
    )


    model.fit(
        X_train_z,
        y_train,
    )


    return model.predict(
        X_validation_z
    )


# =============================================================================
# MEMORY CAPACITY
# =============================================================================

def safe_corr2(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )


    if (
        np.std(a)
        <=
        VAR_TOL

        or

        np.std(b)
        <=
        VAR_TOL
    ):

        return 0.0


    corr = np.corrcoef(
        a,
        b,
    )[0, 1]


    if not np.isfinite(
        corr
    ):

        return 0.0


    return float(
        corr * corr
    )


def evaluate_memory(
    feature_bank,
    probe,
    n_train,
    n_val,
    burnin,
):

    X = (
        feature_bank[
            XYZ_ALL
        ]
        .to_numpy(
            dtype=float
        )
    )


    null_floor = (
        1.0
        /
        (
            n_val
            -
            1.0
        )
    )


    rows = []


    for channel in range(
        probe.shape[1]
    ):

        for delay in range(
            1,
            K_MAX + 1,
        ):

            start = max(
                burnin,
                delay,
            )


            train_index = np.arange(
                start,
                n_train,
            )


            validation_index = np.arange(
                n_train,
                n_train + n_val,
            )


            y_train = probe[
                train_index
                -
                delay,

                channel,
            ]


            y_validation = probe[
                validation_index
                -
                delay,

                channel,
            ]


            prediction = (
                standardized_ridge_predict(

                    X[
                        train_index
                    ],

                    y_train,

                    X[
                        validation_index
                    ],

                    MC_RIDGE_ALPHA,
                )
            )


            raw = safe_corr2(
                y_validation,
                prediction,
            )


            corrected = max(
                raw
                -
                null_floor,
                0.0,
            )


            rows.append({

                "channel":
                    channel,

                "delay":
                    delay,

                "MC_raw":
                    raw,

                "MC_corrected":
                    corrected,
            })


    detail = pd.DataFrame(
        rows
    )


    total_raw = float(
        detail[
            "MC_raw"
        ].sum()
    )


    total_corrected = float(
        detail[
            "MC_corrected"
        ].sum()
    )


    n_channels = (
        probe.shape[1]
    )


    def delay_mean(
        delay,
        column="MC_raw",
    ):

        values = detail.loc[
            detail[
                "delay"
            ]
            ==
            delay,
            column,
        ]


        return float(
            values.mean()
        )


    return {

        "MC_total_raw":
            total_raw,

        "MC_total_corrected":
            total_corrected,

        "MC_per_channel_raw":
            total_raw
            /
            n_channels,

        "MC_per_channel_corrected":
            total_corrected
            /
            n_channels,

        "MC_delay1_mean":
            delay_mean(1),

        "MC_delay2_mean":
            delay_mean(2),

        "MC_delay5_mean":
            delay_mean(5),

        "MC_delay10_mean":
            delay_mean(10),

        "MC_delay20_mean":
            delay_mean(20),

        "detail":
            detail,
    }


# =============================================================================
# FORECASTING CV
# =============================================================================

def chronological_cv_rmse(
    X,
    y,
    n_train,
):

    X = np.asarray(
        X,
        dtype=float,
    )[:n_train]


    y = np.asarray(
        y,
        dtype=float,
    )[:n_train]


    splitter = TimeSeriesSplit(
        n_splits=
            CV_FOLDS
    )


    values = []


    for train_idx, validation_idx in (
        splitter.split(
            X
        )
    ):

        prediction = (
            standardized_ridge_predict(

                X[
                    train_idx
                ],

                y[
                    train_idx
                ],

                X[
                    validation_idx
                ],

                FORECAST_RIDGE_ALPHA,
            )
        )


        values.append(

            rmse(
                y[
                    validation_idx
                ],
                prediction,
            )
        )


    values = np.asarray(
        values,
        dtype=float,
    )


    return {

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "std":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),

        "folds":
            values,
    }


# =============================================================================
# DATA COLUMN RESOLUTION
# =============================================================================

def resolve_column(
    dataframe,
    candidates,
    label,
):

    lower_map = {

        str(column).lower():
            column

        for column
        in dataframe.columns
    }


    for candidate in candidates:

        if candidate in (
            dataframe.columns
        ):

            return candidate


        if (
            candidate.lower()
            in lower_map
        ):

            return lower_map[
                candidate.lower()
            ]


    raise KeyError(

        f"Could not find {label}. "
        f"Tried {candidates}."
    )


# =============================================================================
# CREATE ROW J COLUMNS
# =============================================================================

def add_J_columns(
    row,
    J,
):

    for edge in ALL_EDGES:

        row[
            edge_name(
                edge
            )
        ] = (

            float(
                J[
                    edge
                ]
            )

            if edge in J

            else np.nan
        )


    return row


# =============================================================================
# EVALUATE ONE J CANDIDATE
# =============================================================================

def evaluate_J_candidate(
    topology_name,
    vector,
    stage,
    candidate_number,
    probe,
    n_train,
    n_val,
):

    J = (
        J_dict_from_vector(
            topology_name,
            vector,
        )
    )


    U, unitarity_error = (
        build_unitary(
            topology_name,
            J,
        )
    )


    A_probe, S_probe = (
        qrc.build_input_channels(
            U,
            probe,
        )
    )


    wash = (
        washout_diagnostic(
            S_probe
        )
    )


    if np.isfinite(
        wash[
            "Tw_0.01"
        ]
    ):

        burnin = int(

            max(

                MIN_BURNIN,

                int(
                    wash[
                        "Tw_0.01"
                    ]
                )
                +
                POST_WASHOUT_MARGIN,
            )
        )

    else:

        burnin = (
            n_train
            -
            100
        )


    burnin = min(
        burnin,
        n_train - 100,
    )


    feature_bank = (
        build_feature_bank(
            A_probe
        )
    )


    memory = (
        evaluate_memory(

            feature_bank,
            probe,
            n_train,
            n_val,
            burnin,
        )
    )


    washout_valid = int(

        np.isfinite(
            wash[
                "Tw_0.01"
            ]
        )

        and

        wash[
            "Tw_0.01"
        ]
        <=
        MAX_ACCEPTABLE_WASHOUT
    )


    row = {

        "topology":
            topology_name,

        "description":
            CANDIDATES[
                topology_name
            ][
                "description"
            ],

        "stage":
            stage,

        "candidate_number":
            candidate_number,

        "dt":
            DT,

        "hx":
            HX,

        "Trotter_r":
            TROTTER_R,

        "alpha":
            ALPHA,

        "MC_readout":
            "XYZ_all",

        "unitarity_error":
            unitarity_error,

        "MC_burnin":
            burnin,

        "MC_per_channel_raw":
            memory[
                "MC_per_channel_raw"
            ],

        "MC_per_channel_corrected":
            memory[
                "MC_per_channel_corrected"
            ],

        "MC_total_corrected":
            memory[
                "MC_total_corrected"
            ],

        "MC_delay1_mean":
            memory[
                "MC_delay1_mean"
            ],

        "MC_delay2_mean":
            memory[
                "MC_delay2_mean"
            ],

        "MC_delay5_mean":
            memory[
                "MC_delay5_mean"
            ],

        "MC_delay10_mean":
            memory[
                "MC_delay10_mean"
            ],

        "MC_delay20_mean":
            memory[
                "MC_delay20_mean"
            ],

        "Tw_0.01":
            wash[
                "Tw_0.01"
            ],

        "tau_mem":
            wash[
                "tau_mem"
            ],

        "D_final":
            wash[
                "D_final"
            ],

        "kappa_eff":
            wash[
                "kappa_eff"
            ],

        "kappa_fit_R2":
            wash[
                "kappa_fit_R2"
            ],

        "washout_valid":
            washout_valid,

        "J_vector":
            json.dumps(
                [
                    float(x)
                    for x in vector
                ]
            ),
    }


    row = add_J_columns(
        row,
        J,
    )


    detail = (
        memory[
            "detail"
        ]
        .copy()
    )


    detail[
        "topology"
    ] = topology_name

    detail[
        "stage"
    ] = stage

    detail[
        "candidate_number"
    ] = candidate_number


    for edge, value in J.items():

        detail[
            edge_name(
                edge
            )
        ] = value


    return (
        row,
        detail,
    )


# =============================================================================
# LOCAL REFINEMENT
# =============================================================================

def vector_from_row(
    row,
    topology_name,
):

    edges = (
        CANDIDATES[
            topology_name
        ][
            "edges"
        ]
    )


    return np.array(

        [
            float(
                row[
                    edge_name(
                        edge
                    )
                ]
            )

            for edge
            in edges
        ],

        dtype=float,
    )


def build_local_vectors(
    centre,
):

    centre = np.asarray(
        centre,
        dtype=float,
    )


    candidates = []


    for dimension in range(
        len(
            centre
        )
    ):

        for delta in (
            LOCAL_DELTAS
        ):

            vector = (
                centre.copy()
            )


            vector[
                dimension
            ] = np.clip(

                vector[
                    dimension
                ]
                +
                delta,

                J_LOCAL_MIN,
                J_LOCAL_MAX,
            )


            candidates.append(
                vector
            )


    # Deduplicate after clipping.
    unique = []

    seen = set()


    for vector in candidates:

        key = tuple(
            np.round(
                vector,
                10,
            )
        )


        if key in seen:
            continue


        seen.add(
            key
        )

        unique.append(
            vector
        )


    return unique


# =============================================================================
# CHECKPOINT
# =============================================================================

CHECKPOINT_PATH = (
    RESULTS
    /
    "08_03b_checkpoint.csv"
)


def save_checkpoint(
    rows,
):

    pd.DataFrame(
        rows
    ).to_csv(
        CHECKPOINT_PATH,
        index=False,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 140)
    print("WEEK 8 - STEP 8.3B")
    print("H0-H4 INDIVIDUAL-J OPTIMIZATION: MC FIRST, THEN RMSE")
    print("=" * 140)

    print()

    print("Frozen:")
    print(f"  alpha       = {ALPHA}")
    print(f"  hx          = {HX}")
    print(f"  dt          = {DT}")
    print(f"  Trotter r   = {TROTTER_R}")
    print(f"  seed        = {SEED}")
    print()

    print(
        f"Global J interval = "
        f"[{J_GLOBAL_MIN:+.1f}, "
        f"{J_GLOBAL_MAX:+.1f}]"
    )

    print(
        f"Global candidates/topology = "
        f"{N_GLOBAL_TOTAL}"
    )

    print(
        "Local refinement deltas = "
        f"{LOCAL_DELTAS}"
    )

    print(
        f"Forecast finalists/topology = "
        f"{N_FORECAST_FINALISTS}"
    )

    print()

    print(
        "Important: every topology receives "
        "the same global J-search budget."
    )

    print(
        "All five available logical couplings "
        "are independently tunable."
    )

    print()

    # =========================================================================
    # DATA
    # =========================================================================

    work, train, val, cols = (
        qrc.load_data()
    )


    n_train = len(
        train
    )

    n_val = len(
        val
    )

    n_steps = (
        n_train
        +
        n_val
    )


    work_eval = (
        work
        .iloc[
            :n_steps
        ]
        .copy()
    )


    real_angles = np.asarray(

        qrc.make_input_angles(
            work_eval,
            cols,
        ),

        dtype=float,
    )[:n_steps]


    probe = build_probe(
        real_angles
    )


    marginal_error = float(

        np.max(

            np.abs(

                np.sort(
                    probe,
                    axis=0,
                )

                -

                np.sort(
                    real_angles,
                    axis=0,
                )
            )
        )
    )


    max_ac = (
        probe_audit(
            probe
        )
    )


    print(
        f"Probe marginal max error = "
        f"{marginal_error:.3e}"
    )

    print(
        f"Probe max |autocorrelation| "
        f"lags 1..20 = "
        f"{max_ac:.6f}"
    )

    print()


    # =========================================================================
    # TARGETS
    # =========================================================================

    target_col = resolve_column(

        work_eval,

        [
            "target_property_damage_claim_count",
            "C_t_plus_1",
            "target",
        ],

        "next-day claim target",
    )


    current_col = resolve_column(

        work_eval,

        [
            "property_damage_claim_count_t",
            "C_t",
            "current_property_damage_claim_count",
        ],

        "current claim count",
    )


    y_level = (

        work_eval[
            target_col
        ]
        .to_numpy(
            dtype=float
        )
    )


    current_claims = (

        work_eval[
            current_col
        ]
        .to_numpy(
            dtype=float
        )
    )


    y_delta = (
        y_level
        -
        current_claims
    )


    # =========================================================================
    # SEARCH
    # =========================================================================

    all_rows = []

    delay_rows = []


    for topology_index, topology_name in enumerate(
        CANDIDATES,
        start=1,
    ):

        print("=" * 140)

        print(
            f"[{topology_index}/{len(CANDIDATES)}] "
            f"{topology_name}: "
            f"{CANDIDATES[topology_name]['description']}"
        )

        print("=" * 140)

        print()

        print(
            "Logical edge order:"
        )


        for index, edge in enumerate(
            CANDIDATES[
                topology_name
            ][
                "edges"
            ],
            start=1,
        ):

            print(
                f"  J[{index}] = "
                f"{edge_name(edge)} "
                f"({LOGICAL_ROLES[edge[0]]}"
                f"-{LOGICAL_ROLES[edge[1]]})"
            )


        print()

        topology_rows = []


        # =====================================================================
        # STAGE A - GLOBAL SEARCH
        # =====================================================================

        print(
            "STAGE A - GLOBAL J SEARCH"
        )

        print("-" * 80)


        for candidate_index, vector in enumerate(
            GLOBAL_J_VECTORS,
            start=1,
        ):

            print(
                f"  [{candidate_index:02d}/"
                f"{N_GLOBAL_TOTAL}] "
                f"J="
                f"{np.round(vector, 3).tolist()}"
            )


            row, detail = (
                evaluate_J_candidate(

                    topology_name=
                        topology_name,

                    vector=
                        vector,

                    stage=
                        "GLOBAL",

                    candidate_number=
                        candidate_index,

                    probe=
                        probe,

                    n_train=
                        n_train,

                    n_val=
                        n_val,
                )
            )


            all_rows.append(
                row
            )

            topology_rows.append(
                row
            )

            delay_rows.append(
                detail
            )


            print(
                f"      MC/ch="
                f"{row['MC_per_channel_corrected']:.5f} | "
                f"MC total="
                f"{row['MC_total_corrected']:.5f} | "
                f"Tw="
                f"{row['Tw_0.01']} | "
                f"valid="
                f"{row['washout_valid']}"
            )


            save_checkpoint(
                all_rows
            )


        # =====================================================================
        # GLOBAL WINNER
        # =====================================================================

        global_df = pd.DataFrame(
            topology_rows
        )


        valid_global = (
            global_df[
                global_df[
                    "washout_valid"
                ]
                ==
                1
            ]
            .copy()
        )


        if len(
            valid_global
        ):

            global_winner = (

                valid_global
                .sort_values(

                    [
                        "MC_per_channel_corrected",
                        "MC_delay5_mean",
                        "MC_delay10_mean",
                    ],

                    ascending=[
                        False,
                        False,
                        False,
                    ],
                )
                .iloc[0]
            )


        else:

            global_winner = (

                global_df
                .sort_values(
                    "MC_per_channel_corrected",
                    ascending=False,
                )
                .iloc[0]
            )


        centre = vector_from_row(
            global_winner,
            topology_name,
        )


        print()

        print(
            "Best global candidate:"
        )

        print(
            f"  J = "
            f"{np.round(centre, 6).tolist()}"
        )

        print(
            f"  corrected MC/ch = "
            f"{global_winner['MC_per_channel_corrected']:.6f}"
        )

        print(
            f"  Tw(.01) = "
            f"{global_winner['Tw_0.01']}"
        )

        print()


        # =====================================================================
        # STAGE B - LOCAL COORDINATE REFINEMENT
        # =====================================================================

        local_vectors = (
            build_local_vectors(
                centre
            )
        )


        print(
            "STAGE B - LOCAL J REFINEMENT"
        )

        print("-" * 80)


        for local_index, vector in enumerate(
            local_vectors,
            start=1,
        ):

            print(
                f"  [{local_index:02d}/"
                f"{len(local_vectors)}] "
                f"J="
                f"{np.round(vector, 3).tolist()}"
            )


            row, detail = (
                evaluate_J_candidate(

                    topology_name=
                        topology_name,

                    vector=
                        vector,

                    stage=
                        "LOCAL",

                    candidate_number=
                        local_index,

                    probe=
                        probe,

                    n_train=
                        n_train,

                    n_val=
                        n_val,
                )
            )


            all_rows.append(
                row
            )

            topology_rows.append(
                row
            )

            delay_rows.append(
                detail
            )


            print(
                f"      MC/ch="
                f"{row['MC_per_channel_corrected']:.5f} | "
                f"MC total="
                f"{row['MC_total_corrected']:.5f} | "
                f"Tw="
                f"{row['Tw_0.01']} | "
                f"valid="
                f"{row['washout_valid']}"
            )


            save_checkpoint(
                all_rows
            )


        # =====================================================================
        # FINAL TOPOLOGY WINNER
        # =====================================================================

        topology_df = pd.DataFrame(
            topology_rows
        )


        valid_topology = (

            topology_df[
                topology_df[
                    "washout_valid"
                ]
                ==
                1
            ]
            .copy()
        )


        ranking_source = (

            valid_topology

            if len(
                valid_topology
            )

            else topology_df
        )


        winner = (

            ranking_source
            .sort_values(

                [
                    "MC_per_channel_corrected",
                    "MC_delay5_mean",
                    "MC_delay10_mean",
                    "MC_delay20_mean",
                ],

                ascending=[
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )


        print()

        print(
            f"FINAL {topology_name} MEMORY WINNER"
        )

        print("-" * 80)

        print(
            f"  MC/ch corrected = "
            f"{winner['MC_per_channel_corrected']:.6f}"
        )

        print(
            f"  MC total        = "
            f"{winner['MC_total_corrected']:.6f}"
        )

        print(
            f"  delay 1         = "
            f"{winner['MC_delay1_mean']:.6f}"
        )

        print(
            f"  delay 5         = "
            f"{winner['MC_delay5_mean']:.6f}"
        )

        print(
            f"  delay 10        = "
            f"{winner['MC_delay10_mean']:.6f}"
        )

        print(
            f"  delay 20        = "
            f"{winner['MC_delay20_mean']:.6f}"
        )

        print(
            f"  Tw(.01)         = "
            f"{winner['Tw_0.01']}"
        )


        for edge in (
            CANDIDATES[
                topology_name
            ][
                "edges"
            ]
        ):

            print(
                f"  {edge_name(edge)}"
                f" = "
                f"{winner[edge_name(edge)]:+.6f}"
            )


        print()


    # =========================================================================
    # MEMORY RESULTS
    # =========================================================================

    all_df = pd.DataFrame(
        all_rows
    )


    ranked_df = (

        all_df
        .sort_values(

            [
                "washout_valid",
                "MC_per_channel_corrected",
                "MC_delay5_mean",
                "MC_delay10_mean",
            ],

            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    ranked_df[
        "global_rank"
    ] = (

        np.arange(
            len(
                ranked_df
            )
        )
        +
        1
    )


    all_df.to_csv(

        RESULTS
        /
        "08_03b_J_search_all.csv",

        index=False,
    )


    ranked_df.to_csv(

        RESULTS
        /
        "08_03b_J_search_ranked.csv",

        index=False,
    )


    pd.concat(
        delay_rows,
        ignore_index=True,
    ).to_csv(

        RESULTS
        /
        "08_03b_J_search_by_delay.csv",

        index=False,
    )


    # =========================================================================
    # WINNER PER TOPOLOGY
    # =========================================================================

    topology_winners = []


    for topology_name in CANDIDATES:

        subset = all_df[
            all_df[
                "topology"
            ]
            ==
            topology_name
        ]


        valid = subset[
            subset[
                "washout_valid"
            ]
            ==
            1
        ]


        source = (
            valid
            if len(valid)
            else subset
        )


        winner = (

            source
            .sort_values(

                [
                    "MC_per_channel_corrected",
                    "MC_delay5_mean",
                    "MC_delay10_mean",
                ],

                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )


        topology_winners.append(
            winner.to_dict()
        )


    topology_winners_df = (
        pd.DataFrame(
            topology_winners
        )
        .sort_values(
            "MC_per_channel_corrected",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    topology_winners_df[
        "MC_rank"
    ] = (

        np.arange(
            len(
                topology_winners_df
            )
        )
        +
        1
    )


    topology_winners_df.to_csv(

        RESULTS
        /
        "08_03b_J_search_topology_winners.csv",

        index=False,
    )


    # =========================================================================
    # FORECASTING TOP 5 MEMORY CANDIDATES PER TOPOLOGY
    # =========================================================================

    print()

    print("=" * 140)
    print("FORECASTING DIAGNOSTICS FOR TOP-5 MEMORY CANDIDATES PER TOPOLOGY")
    print("=" * 140)

    print()


    forecast_rows = []


    for topology_name in CANDIDATES:

        subset = all_df[
            all_df[
                "topology"
            ]
            ==
            topology_name
        ].copy()


        valid = subset[
            subset[
                "washout_valid"
            ]
            ==
            1
        ]


        source = (

            valid

            if len(valid)

            else subset
        )


        finalists = (

            source
            .sort_values(

                [
                    "MC_per_channel_corrected",
                    "MC_delay5_mean",
                    "MC_delay10_mean",
                ],

                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .head(
                N_FORECAST_FINALISTS
            )
        )


        print(
            f"{topology_name}:"
        )


        for finalist_rank, (_, row) in enumerate(
            finalists.iterrows(),
            start=1,
        ):

            vector = (
                vector_from_row(
                    row,
                    topology_name,
                )
            )


            J = (
                J_dict_from_vector(
                    topology_name,
                    vector,
                )
            )


            U, _ = (
                build_unitary(
                    topology_name,
                    J,
                )
            )


            A_real, _ = (
                qrc.build_input_channels(
                    U,
                    real_angles,
                )
            )


            bank_real = (
                build_feature_bank(
                    A_real
                )
            )


            for (
                family_name,
                feature_names
            ) in (
                FORECAST_FAMILIES.items()
            ):

                X = (

                    bank_real[
                        feature_names
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )


                level_cv = (
                    chronological_cv_rmse(
                        X,
                        y_level,
                        n_train,
                    )
                )


                delta_cv = (
                    chronological_cv_rmse(
                        X,
                        y_delta,
                        n_train,
                    )
                )


                out = (
                    row.to_dict()
                )


                out[
                    "memory_finalist_rank"
                ] = finalist_rank


                out[
                    "forecast_family"
                ] = family_name


                out[
                    "n_forecast_features"
                ] = len(
                    feature_names
                )


                out[
                    "level_CV_RMSE_mean"
                ] = (
                    level_cv[
                        "mean"
                    ]
                )


                out[
                    "level_CV_RMSE_std"
                ] = (
                    level_cv[
                        "std"
                    ]
                )


                out[
                    "delta_CV_RMSE_mean"
                ] = (
                    delta_cv[
                        "mean"
                    ]
                )


                out[
                    "delta_CV_RMSE_std"
                ] = (
                    delta_cv[
                        "std"
                    ]
                )


                for fold, value in enumerate(
                    level_cv[
                        "folds"
                    ],
                    start=1,
                ):

                    out[
                        f"level_fold{fold}_RMSE"
                    ] = value


                for fold, value in enumerate(
                    delta_cv[
                        "folds"
                    ],
                    start=1,
                ):

                    out[
                        f"delta_fold{fold}_RMSE"
                    ] = value


                forecast_rows.append(
                    out
                )


                print(

                    f"  #{finalist_rank} "
                    f"{family_name:12s} | "
                    f"MC/ch="
                    f"{row['MC_per_channel_corrected']:.4f} | "
                    f"level="
                    f"{level_cv['mean']:.4f} | "
                    f"delta="
                    f"{delta_cv['mean']:.4f}"
                )


        print()


    forecast_df = pd.DataFrame(
        forecast_rows
    )


    forecast_df.to_csv(

        RESULTS
        /
        "08_03b_J_search_forecast_finalists.csv",

        index=False,
    )


    # =========================================================================
    # FINAL CONSOLE SUMMARY
    # =========================================================================

    print()

    print("=" * 140)
    print("TOPOLOGY WINNERS BY INTRINSIC MEMORY")
    print("=" * 140)


    show_J_columns = [
        edge_name(
            edge
        )
        for edge
        in ALL_EDGES
    ]


    winner_columns = [

        "MC_rank",
        "topology",
        "stage",

        "MC_per_channel_corrected",
        "MC_total_corrected",

        "MC_delay1_mean",
        "MC_delay2_mean",
        "MC_delay5_mean",
        "MC_delay10_mean",
        "MC_delay20_mean",

        "Tw_0.01",
        "tau_mem",

    ] + show_J_columns


    print(

        topology_winners_df[
            winner_columns
        ]
        .to_string(
            index=False
        )
    )


    print()

    print("=" * 140)
    print("BEST FORECAST RESULT AMONG MEMORY FINALISTS")
    print("=" * 140)


    best_forecast = (

        forecast_df
        .sort_values(

            [
                "level_CV_RMSE_mean",
                "MC_per_channel_corrected",
            ],

            ascending=[
                True,
                False,
            ],
        )
    )


    forecast_show = [

        "topology",
        "memory_finalist_rank",
        "forecast_family",

        "MC_per_channel_corrected",

        "level_CV_RMSE_mean",
        "level_CV_RMSE_std",

        "delta_CV_RMSE_mean",
        "delta_CV_RMSE_std",
    ]


    print(

        best_forecast[
            forecast_show
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


    # =========================================================================
    # MANIFEST
    # =========================================================================

    manifest = {

        "frozen": {

            "seed":
                SEED,

            "alpha":
                ALPHA,

            "hx":
                HX,

            "dt":
                DT,

            "Trotter_r":
                TROTTER_R,
        },

        "search": {

            "global_candidates_per_topology":
                N_GLOBAL_TOTAL,

            "global_random_candidates":
                N_GLOBAL_RANDOM,

            "global_J_interval": [
                J_GLOBAL_MIN,
                J_GLOBAL_MAX,
            ],

            "local_deltas":
                LOCAL_DELTAS,

            "local_J_clip": [
                J_LOCAL_MIN,
                J_LOCAL_MAX,
            ],

            "forecast_finalists_per_topology":
                N_FORECAST_FINALISTS,
        },

        "selection": {

            "primary":
                (
                    "maximize corrected post-washout "
                    "MC per channel"
                ),

            "constraint":
                (
                    "finite Tw(0.01) <= "
                    f"{MAX_ACCEPTABLE_WASHOUT}"
                ),

            "secondary": [
                "MC_delay5_mean",
                "MC_delay10_mean",
                "MC_delay20_mean",
            ],
        },

        "MC_readout":
            "XYZ_all",

        "forecast_readouts":
            list(
                FORECAST_FAMILIES.keys()
            ),

        "topologies":
            CANDIDATES,
    }


    with open(

        RESULTS
        /
        "08_03b_J_search_manifest.json",

        "w",
        encoding="utf-8",

    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            default=str,
        )


    # =========================================================================
    # TEXT SUMMARY
    # =========================================================================

    with open(

        RESULTS
        /
        "08_03b_J_search_summary.txt",

        "w",
        encoding="utf-8",

    ) as file:

        file.write(
            "WEEK 8 STEP 8.3B - H0-H4 J OPTIMIZATION\n"
        )

        file.write(
            "=" * 110
            +
            "\n\n"
        )

        file.write(

            "Frozen: "
            f"alpha={ALPHA}, "
            f"hx={HX}, "
            f"dt={DT}, "
            f"r={TROTTER_R}, "
            f"seed={SEED}\n\n"
        )

        file.write(
            "TOPOLOGY MEMORY WINNERS\n"
        )

        file.write(
            topology_winners_df[
                winner_columns
            ].to_string(
                index=False
            )
        )

        file.write(
            "\n\nBEST FORECAST RESULTS "
            "AMONG MEMORY FINALISTS\n"
        )

        file.write(
            best_forecast[
                forecast_show
            ]
            .head(25)
            .to_string(
                index=False
            )
        )


    print()

    print("=" * 140)
    print("FILES SAVED")
    print("=" * 140)

    print(
        "results/08_03b_J_search_all.csv"
    )

    print(
        "results/08_03b_J_search_ranked.csv"
    )

    print(
        "results/08_03b_J_search_topology_winners.csv"
    )

    print(
        "results/08_03b_J_search_by_delay.csv"
    )

    print(
        "results/08_03b_J_search_forecast_finalists.csv"
    )

    print(
        "results/08_03b_J_search_summary.txt"
    )

    print(
        "results/08_03b_J_search_manifest.json"
    )

    print(
        "results/08_03b_checkpoint.csv"
    )

    print()

    print(
        "Step 8.3B complete."
    )

    print(
        "Interpret H0-H4 MC and RMSE before "
        "extending any J boundary or tuning dt."
    )


if __name__ == "__main__":
    main()