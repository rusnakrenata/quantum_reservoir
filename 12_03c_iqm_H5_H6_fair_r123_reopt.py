"""
Week 12 - Step 12.3C
Fair H5/H6 re-optimization over r, hx, hy, J and Ridge lambda.

PURPOSE
-------
Mirror the original IBM Week-9 fair r=1,2,3 re-optimization for the new
IQM-native logical topologies H5 and H6.

The reduced Y-field seed universe is NOT all nine h_y values from 12.3B.
For each topology we take:
    * h_y = 0 as the mandatory Y-OFF control, plus
    * every h_y retained on the 12.3B Pareto set.

For every retained dynamics seed and every r in {1,2,3}, we give exactly
the same search budget.

FROZEN IN THIS STEP
-------------------
alpha = 0.75
dt    = 1.6

J0 for each topology = its Week-12.3A intrinsic-memory winner.

SEARCH — SAME LOGIC AS ORIGINAL 09_03c
---------------------------------------
Stage A:
    hx in {0.25, 0.50, 0.75}

    Y-OFF:
        hy = 0 exactly

    Y-ON, starting from retained seed hy0:
        hy in {0.5*hy0, 1.0*hy0, 1.5*hy0, -1.0*hy0}

    global J scale in {0.5, 1.0, 1.5}

Stage B:
    local refinement around the UNION of:
        * best chronological-CV parent for each readout
        * best exact-vs-Trotter-fidelity parent

    Four local children per parent.

    hx ~ N(parent_hx, 0.12)
    hy sigma = max(0.08, 0.20*|parent_hy|), unless Y-OFF
    each J sigma = max(0.10, 0.20*max(|J|,0.5))

Bounds:
    hx in [0.10,1.00]
    hy in [-1.00,1.00]
    J  in [-2.00,2.00]

READOUTS
--------
XZ_injection
XZinj_plus_YX45
XYZ_all

RIDGE
-----
lambda in [1e-4,1e-3,1e-2,1e-1,1,10,100,300]

Five-fold chronological CV on 2022-2024 selects candidates and lambda.
2025 is diagnostic only.
2026 is untouched.

MEMORY
------
After the search, every CV-best branch/r configuration receives the same
five-seed intrinsic-memory + washout diagnostic used in Week 9:
    probe seeds [79001,42,101,505,707]
    XYZ_all memory readout
    B=max(100,Tw(0.01)+20)

PROCESS FIDELITY
----------------
For each dynamics configuration:
    F_proc = |Tr(U_exact^\dagger U_trotter)|^2 / d^2

No weighted score is used.

IMPORTANT
---------
This script is still an IDEAL logical-dynamics calibration.
It does not submit IQM QPU jobs, use shots, or use backend noise.
IQM real-circuit feasibility/noise comes after this parameter search.

Standing comparison thresholds printed at the end:
    best QRC CV so far      = 3.408184
    best classical CV       = 3.392277
    best ideal QRC 2025     = 4.755921
    best classical 2025     = 4.869006

DEPENDENCIES
------------
Keep beside this script:
    07_02b_memory_pair_isolation.py
    12_03a_iqm_H5_H6_topology_J_mc_rmse_search.py
    12_03b_iqm_H5_H6_yfield_refresh.py

Required results:
    results/12_03a_J_search_topology_winners.csv
    results/12_03b_yfield_pareto.csv

OUTPUTS
-------
results/12_03c_candidate_universe.csv
results/12_03c_search_all.csv
results/12_03c_ridge_grid.csv
results/12_03c_forecast_best_by_branch_r.csv
results/12_03c_fidelity_best_by_branch_r.csv
results/12_03c_branch_global_best.csv
results/12_03c_memory_by_seed.csv
results/12_03c_memory_summary.csv
results/12_03c_forecast_best_with_memory.csv
results/12_03c_global_winner.csv
results/12_03c_manifest.json

Checkpoint files are also written so an interrupted search can resume.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS / DEPENDENCIES
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

PREV_SCRIPT = HERE / "12_03b_iqm_H5_H6_yfield_refresh.py"
J_WINNER_FILE = RESULTS / "12_03a_J_search_topology_winners.csv"
Y_PARETO_FILE = RESULTS / "12_03b_yfield_pareto.csv"

for path in [
    PREV_SCRIPT,
    J_WINNER_FILE,
    Y_PARETO_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required dependency not found: {path}"
        )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


prev = load_module(
    PREV_SCRIPT,
    "qrc_week12_03c_prev",
)

base = prev.base
qrc = prev.qrc


# =============================================================================
# FAIR-SEARCH CONSTANTS — MATCH ORIGINAL 09_03c LOGIC
# =============================================================================

TOPOLOGIES = ["H5", "H6"]

ALPHA = 0.75
DT = 1.6
R_VALUES = [1, 2, 3]

HX_REFERENCE = 0.5
HX_GRID = [0.25, 0.50, 0.75]
J_SCALE_GRID = [0.50, 1.00, 1.50]

HY_FACTORS = [0.50, 1.00, 1.50, -1.00]

HX_BOUNDS = (0.10, 1.00)
HY_BOUNDS = (-1.00, 1.00)
J_BOUNDS = (-2.00, 2.00)

LOCAL_CHILDREN_PER_PARENT = 4
HX_LOCAL_SIGMA = 0.12
HY_LOCAL_REL_SIGMA = 0.20
HY_LOCAL_MIN_SIGMA = 0.08
J_LOCAL_REL_SIGMA = 0.20
J_LOCAL_MIN_SIGMA = 0.10

READOUTS = [
    "XZ_injection",
    "XZinj_plus_YX45",
    "XYZ_all",
]

READOUT_FEATURES = prev.READOUT_FEATURES
READOUT_SETTINGS = prev.READOUT_SETTINGS

PROBE_SEEDS = list(prev.PROBE_SEEDS)

MIN_BURNIN = prev.MIN_BURNIN
POST_WASHOUT_MARGIN = prev.POST_WASHOUT_MARGIN
MAX_ACCEPTABLE_WASHOUT = prev.MAX_ACCEPTABLE_WASHOUT

# Standing project thresholds.
BEST_QRC_CV = 3.408184
BEST_CLASSICAL_CV = 3.392277
BEST_IDEAL_QRC_2025 = 4.755921
BEST_CLASSICAL_2025 = 4.869006

# Resume.
RESUME_SEARCH = True
RESUME_MEMORY = True

SEARCH_CHECKPOINT = RESULTS / "12_03c_search_checkpoint.csv"
RIDGE_CHECKPOINT = RESULTS / "12_03c_ridge_checkpoint.csv"
COMPLETED_BLOCKS_FILE = RESULTS / "12_03c_completed_blocks.json"

MEMORY_BY_SEED_FILE = RESULTS / "12_03c_memory_by_seed.csv"
MEMORY_SUMMARY_FILE = RESULTS / "12_03c_memory_summary.csv"

qrc.ALPHA = ALPHA


# =============================================================================
# DATA — inherited from 12.3B; 2026 is not loaded
# =============================================================================

REAL_ANGLES = prev.REAL_ANGLES
Y_TARGET = prev.Y_TARGET
N_TRAIN = prev.N_TRAIN
N_VAL = prev.N_VAL
N_TOTAL = prev.N_TOTAL


# =============================================================================
# FROZEN REFERENCE J0 — 12.3A MEMORY WINNER PER TOPOLOGY
# =============================================================================

REFERENCE_J = {
    topology: {
        edge: float(value)
        for edge, value in prev.REFERENCE_J[topology].items()
    }
    for topology in TOPOLOGIES
}


# =============================================================================
# REDUCED Y-SEED UNIVERSE
# =============================================================================

pareto_df = pd.read_csv(Y_PARETO_FILE)

HY_SEEDS = {}

for topology in TOPOLOGIES:
    vals = (
        pareto_df.loc[
            pareto_df["topology"].astype(str) == topology,
            "hy",
        ]
        .astype(float)
        .tolist()
    )

    # Mandatory Y-OFF control.
    vals.append(0.0)

    unique = []

    for x in vals:
        x = float(x)

        if not any(
            np.isclose(x, y)
            for y in unique
        ):
            unique.append(x)

    # Stable deterministic order: Y-OFF first, then increasing |h_y|,
    # with negative before positive at equal magnitude.
    unique = sorted(
        unique,
        key=lambda x: (
            0 if np.isclose(x, 0.0) else 1,
            abs(x),
            x,
        ),
    )

    HY_SEEDS[topology] = unique


def value_slug(x: float) -> str:
    if np.isclose(x, 0.0):
        return "0"

    sign = "p" if x > 0 else "m"
    s = f"{abs(float(x)):.3f}".rstrip("0").rstrip(".")
    return sign + s.replace(".", "p")


candidate_rows = []

for topology, hy_values in HY_SEEDS.items():
    for seed_hy in hy_values:
        y_state = (
            "OFF"
            if np.isclose(seed_hy, 0.0)
            else "ON"
        )

        dynamics_seed_id = (
            f"{topology}_hy{value_slug(seed_hy)}"
        )

        for readout in READOUTS:
            candidate_rows.append({
                "seed_candidate_id":
                    f"{dynamics_seed_id}__{readout}",
                "dynamics_seed_id":
                    dynamics_seed_id,
                "topology":
                    topology,
                "seed_hy":
                    float(seed_hy),
                "y_state":
                    y_state,
                "readout":
                    readout,
            })


candidate_universe = pd.DataFrame(candidate_rows)

candidate_universe.to_csv(
    RESULTS / "12_03c_candidate_universe.csv",
    index=False,
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def clipped(x, bounds):
    return float(
        np.clip(
            float(x),
            bounds[0],
            bounds[1],
        )
    )


def clip_J(J):
    return {
        edge: clipped(value, J_BOUNDS)
        for edge, value in J.items()
    }


def scale_J(J0, scale):
    return {
        edge: clipped(
            float(value) * float(scale),
            J_BOUNDS,
        )
        for edge, value in J0.items()
    }


def J_columns(topology, J):
    out = {}

    for edge in base.CANDIDATES[topology]["edges"]:
        out[
            f"J{edge[0]}{edge[1]}"
        ] = float(J[edge])

    return out


def J_from_row(topology, row):
    J = {}

    for edge in base.CANDIDATES[topology]["edges"]:
        col = f"J{edge[0]}{edge[1]}"
        J[edge] = float(row[col])

    return J


def dynamics_key(
    topology,
    r,
    hx,
    hy,
    J,
):
    return (
        topology,
        int(r),
        round(float(hx), 12),
        round(float(hy), 12),
        tuple(
            (
                edge,
                round(float(J[edge]), 12),
            )
            for edge in base.CANDIDATES[topology]["edges"]
        ),
    )


def deterministic_config_id(
    dynamics_seed_id,
    r,
    stage,
    hx,
    hy,
    J,
):
    payload = {
        "dynamics_seed_id": dynamics_seed_id,
        "r": int(r),
        "stage": str(stage),
        "hx": round(float(hx), 12),
        "hy": round(float(hy), 12),
        "J": {
            f"{edge[0]}{edge[1]}":
                round(float(J[edge]), 12)
            for edge in sorted(J)
        },
    }

    digest = hashlib.sha1(
        json.dumps(
            payload,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"{dynamics_seed_id}"
        f"_r{int(r)}"
        f"_{stage}"
        f"_{digest}"
    )


# =============================================================================
# HAMILTONIAN / UNITARIES
# =============================================================================

def pauli_rotation(
    operator,
    physical_rotation_angle,
):
    theta = float(
        physical_rotation_angle
    )

    return (
        np.cos(theta / 2.0) * qrc.I64
        -
        1j
        * np.sin(theta / 2.0)
        * operator
    )


def build_hamiltonian(
    topology,
    hx,
    hy,
    J,
):
    H = np.zeros(
        (64, 64),
        dtype=complex,
    )

    for edge in base.CANDIDATES[topology]["edges"]:
        H += (
            float(J[edge])
            *
            base.ZZ_OPS[edge]
        )

    for qubit in range(qrc.N_QUBITS):
        H += (
            float(hx)
            *
            qrc.FULL_SINGLE_OPS[
                f"X{qubit}"
            ]
        )

    for qubit in [4, 5]:
        H += (
            float(hy)
            *
            qrc.FULL_SINGLE_OPS[
                f"Y{qubit}"
            ]
        )

    return H


def exact_unitary(H):
    eigvals, eigvecs = np.linalg.eigh(H)

    phases = np.exp(
        -1j
        *
        DT
        *
        eigvals
    )

    return (
        eigvecs
        @
        np.diag(phases)
        @
        eigvecs.conj().T
    )


def trotter_unitary(
    topology,
    hx,
    hy,
    J,
    r,
):
    U = qrc.I64.copy()

    for _ in range(int(r)):

        # ZZ layer
        for edge in base.CANDIDATES[topology]["edges"]:
            theta_zz = (
                2.0
                *
                float(J[edge])
                *
                DT
                /
                int(r)
            )

            U = (
                pauli_rotation(
                    base.ZZ_OPS[edge],
                    theta_zz,
                )
                @
                U
            )

        # X layer
        theta_x = (
            2.0
            *
            float(hx)
            *
            DT
            /
            int(r)
        )

        for qubit in range(qrc.N_QUBITS):
            U = (
                pauli_rotation(
                    qrc.FULL_SINGLE_OPS[
                        f"X{qubit}"
                    ],
                    theta_x,
                )
                @
                U
            )

        # memory-only Y layer
        if not np.isclose(hy, 0.0):
            theta_y = (
                2.0
                *
                float(hy)
                *
                DT
                /
                int(r)
            )

            for qubit in [4, 5]:
                U = (
                    pauli_rotation(
                        qrc.FULL_SINGLE_OPS[
                            f"Y{qubit}"
                        ],
                        theta_y,
                    )
                    @
                    U
                )

    return U


def unitary_accuracy(
    U_exact,
    U_trotter,
):
    d = U_exact.shape[0]

    overlap = np.trace(
        U_exact.conj().T
        @
        U_trotter
    )

    process_fidelity = float(
        (
            abs(overlap) ** 2
        )
        /
        (d ** 2)
    )

    process_fidelity = float(
        np.clip(
            process_fidelity,
            0.0,
            1.0,
        )
    )

    return {
        "process_fidelity":
            process_fidelity,

        "process_infidelity":
            1.0
            -
            process_fidelity,
    }


# =============================================================================
# FORECAST EVALUATION
# =============================================================================

def evaluate_dynamics(
    dynamics_seed_id,
    topology,
    seed_hy,
    y_state,
    r,
    stage,
    hx,
    hy,
    J,
    parent_ids=None,
):
    H = build_hamiltonian(
        topology,
        hx,
        hy,
        J,
    )

    U_exact = exact_unitary(H)

    U_trotter = trotter_unitary(
        topology,
        hx,
        hy,
        J,
        r,
    )

    accuracy = unitary_accuracy(
        U_exact,
        U_trotter,
    )

    A_real, _ = qrc.build_input_channels(
        U_trotter,
        REAL_ANGLES,
    )

    bank = base.build_feature_bank(
        A_real
    )

    config_id = deterministic_config_id(
        dynamics_seed_id,
        r,
        stage,
        hx,
        hy,
        J,
    )

    rows = []
    ridge_rows = []

    for readout in READOUTS:
        X = (
            bank[
                READOUT_FEATURES[
                    readout
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )

        forecast = (
            prev.select_lambda_and_validate(
                X
            )
        )

        seed_candidate_id = (
            f"{dynamics_seed_id}"
            f"__{readout}"
        )

        row = {
            "config_id":
                config_id,

            "seed_candidate_id":
                seed_candidate_id,

            "dynamics_seed_id":
                dynamics_seed_id,

            "topology":
                topology,

            "seed_hy":
                float(seed_hy),

            "y_state":
                y_state,

            "readout":
                readout,

            "trotter_r":
                int(r),

            "stage":
                stage,

            "parent_ids":
                (
                    ""
                    if parent_ids is None
                    else "|".join(
                        sorted(
                            set(
                                parent_ids
                            )
                        )
                    )
                ),

            "hx":
                float(hx),

            "hy":
                float(hy),

            **J_columns(
                topology,
                J,
            ),

            "selected_lambda":
                forecast[
                    "selected_lambda"
                ],

            "cv_rmse":
                forecast[
                    "cv_rmse"
                ],

            "cv_rmse_std":
                forecast[
                    "cv_rmse_std"
                ],

            "validation_rmse":
                forecast[
                    "validation_rmse"
                ],

            "validation_bias":
                forecast[
                    "validation_bias"
                ],

            "n_features":
                len(
                    READOUT_FEATURES[
                        readout
                    ]
                ),

            "n_settings":
                int(
                    READOUT_SETTINGS[
                        readout
                    ]
                ),

            **accuracy,
        }

        rows.append(row)

        grid = (
            forecast[
                "ridge_grid"
            ]
            .copy()
        )

        for _, rr in grid.iterrows():
            ridge_rows.append({
                "config_id":
                    config_id,

                "seed_candidate_id":
                    seed_candidate_id,

                "dynamics_seed_id":
                    dynamics_seed_id,

                "topology":
                    topology,

                "seed_hy":
                    float(seed_hy),

                "y_state":
                    y_state,

                "trotter_r":
                    int(r),

                "stage":
                    stage,

                "readout":
                    readout,

                "ridge_alpha":
                    float(
                        rr[
                            "ridge_alpha"
                        ]
                    ),

                "cv_rmse_mean":
                    float(
                        rr[
                            "cv_rmse_mean"
                        ]
                    ),

                "cv_rmse_std":
                    float(
                        rr[
                            "cv_rmse_std"
                        ]
                    ),
            })

    return rows, ridge_rows


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def load_search_checkpoint():
    if (
        RESUME_SEARCH
        and
        SEARCH_CHECKPOINT.exists()
    ):
        search_rows = (
            pd.read_csv(
                SEARCH_CHECKPOINT
            )
            .to_dict(
                "records"
            )
        )
    else:
        search_rows = []

    if (
        RESUME_SEARCH
        and
        RIDGE_CHECKPOINT.exists()
    ):
        ridge_rows = (
            pd.read_csv(
                RIDGE_CHECKPOINT
            )
            .to_dict(
                "records"
            )
        )
    else:
        ridge_rows = []

    if (
        RESUME_SEARCH
        and
        COMPLETED_BLOCKS_FILE.exists()
    ):
        with open(
            COMPLETED_BLOCKS_FILE,
            "r",
            encoding="utf-8",
        ) as fp:
            completed = set(
                json.load(fp)
            )
    else:
        completed = set()

    return (
        search_rows,
        ridge_rows,
        completed,
    )


def save_search_checkpoint(
    search_rows,
    ridge_rows,
    completed,
):
    pd.DataFrame(
        search_rows
    ).to_csv(
        SEARCH_CHECKPOINT,
        index=False,
    )

    pd.DataFrame(
        ridge_rows
    ).to_csv(
        RIDGE_CHECKPOINT,
        index=False,
    )

    with open(
        COMPLETED_BLOCKS_FILE,
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            sorted(
                completed
            ),
            fp,
            indent=2,
        )


# =============================================================================
# FAIR SEARCH
# =============================================================================

def run_search():
    (
        search_rows_global,
        ridge_rows_global,
        completed_blocks,
    ) = load_search_checkpoint()

    dynamics_seeds = (
        candidate_universe[
            [
                "dynamics_seed_id",
                "topology",
                "seed_hy",
                "y_state",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "topology",
                "seed_hy",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    total_blocks = (
        len(dynamics_seeds)
        *
        len(R_VALUES)
    )

    block_counter = 0

    for _, seed_row in dynamics_seeds.iterrows():
        dynamics_seed_id = str(
            seed_row[
                "dynamics_seed_id"
            ]
        )

        topology = str(
            seed_row[
                "topology"
            ]
        )

        seed_hy = float(
            seed_row[
                "seed_hy"
            ]
        )

        y_state = str(
            seed_row[
                "y_state"
            ]
        )

        J0 = REFERENCE_J[
            topology
        ]

        for r in R_VALUES:
            block_counter += 1

            block_key = (
                f"{dynamics_seed_id}"
                f"__r{r}"
            )

            print("=" * 132)
            print(
                f"[{block_counter}/{total_blocks}] "
                f"{block_key}"
            )
            print("=" * 132)

            if block_key in completed_blocks:
                print(
                    "Already complete in checkpoint — skipping."
                )
                print()
                continue

            block_start = time.perf_counter()

            block_rows = []
            block_ridge_rows = []

            cache = {}

            def evaluate_unique(
                stage,
                hx,
                hy,
                J,
                parent_ids=None,
            ):
                hx = clipped(
                    hx,
                    HX_BOUNDS,
                )

                hy = clipped(
                    hy,
                    HY_BOUNDS,
                )

                if y_state == "OFF":
                    hy = 0.0

                J = clip_J(J)

                key = dynamics_key(
                    topology,
                    r,
                    hx,
                    hy,
                    J,
                )

                if key in cache:
                    return cache[key]

                rows, rr = evaluate_dynamics(
                    dynamics_seed_id=
                        dynamics_seed_id,

                    topology=
                        topology,

                    seed_hy=
                        seed_hy,

                    y_state=
                        y_state,

                    r=
                        r,

                    stage=
                        stage,

                    hx=
                        hx,

                    hy=
                        hy,

                    J=
                        J,

                    parent_ids=
                        parent_ids,
                )

                cache[key] = rows

                block_rows.extend(
                    rows
                )

                block_ridge_rows.extend(
                    rr
                )

                return rows

            # -------------------------------------------------------------
            # Baseline
            # -------------------------------------------------------------

            baseline_rows = evaluate_unique(
                stage="baseline",
                hx=HX_REFERENCE,
                hy=seed_hy,
                J=J0,
            )

            print("baseline:")

            for row in baseline_rows:
                print(
                    f"  {row['readout']:20s} "
                    f"CV={row['cv_rmse']:.6f} "
                    f"Val={row['validation_rmse']:.6f} "
                    f"Fproc={row['process_fidelity']:.6f}"
                )

            # -------------------------------------------------------------
            # Stage A — same coarse budget for every r
            # -------------------------------------------------------------

            if y_state == "OFF":
                hy_values = [0.0]

            else:
                hy_values = []

                for factor in HY_FACTORS:
                    value = clipped(
                        seed_hy
                        *
                        factor,
                        HY_BOUNDS,
                    )

                    if abs(value) < 0.05:
                        value = (
                            0.05
                            if seed_hy > 0
                            else -0.05
                        )

                    if not any(
                        np.isclose(
                            value,
                            x,
                        )
                        for x in hy_values
                    ):
                        hy_values.append(
                            value
                        )

            for hx in HX_GRID:
                for hy in hy_values:
                    for scale in J_SCALE_GRID:
                        J = scale_J(
                            J0,
                            scale,
                        )

                        evaluate_unique(
                            stage="coarse",
                            hx=hx,
                            hy=hy,
                            J=J,
                        )

            current = pd.DataFrame(
                block_rows
            )

            # -------------------------------------------------------------
            # Parent union:
            # best CV per readout + best process fidelity
            # -------------------------------------------------------------

            parent_config_ids = set()

            for readout in READOUTS:
                g = current[
                    current[
                        "readout"
                    ]
                    ==
                    readout
                ]

                best = (
                    g
                    .sort_values(
                        [
                            "cv_rmse",
                            "process_infidelity",
                        ],
                        ascending=[
                            True,
                            True,
                        ],
                    )
                    .iloc[0]
                )

                parent_config_ids.add(
                    str(
                        best[
                            "config_id"
                        ]
                    )
                )

            unique_dynamics = (
                current
                .sort_values(
                    [
                        "process_infidelity",
                        "cv_rmse",
                    ],
                    ascending=[
                        True,
                        True,
                    ],
                )
                .drop_duplicates(
                    subset=[
                        "config_id"
                    ]
                )
            )

            best_fidelity = (
                unique_dynamics.iloc[0]
            )

            parent_config_ids.add(
                str(
                    best_fidelity[
                        "config_id"
                    ]
                )
            )

            # -------------------------------------------------------------
            # Stage B — local refinement
            # -------------------------------------------------------------

            for (
                parent_index,
                parent_config_id,
            ) in enumerate(
                sorted(
                    parent_config_ids
                )
            ):
                parent = (
                    current[
                        current[
                            "config_id"
                        ]
                        ==
                        parent_config_id
                    ]
                    .iloc[0]
                )

                parent_hx = float(
                    parent[
                        "hx"
                    ]
                )

                parent_hy = float(
                    parent[
                        "hy"
                    ]
                )

                parent_J = J_from_row(
                    topology,
                    parent,
                )

                rng_seed = (
                    910000
                    +
                    sum(
                        ord(c)
                        for c
                        in dynamics_seed_id
                    )
                    +
                    1000
                    *
                    int(r)
                    +
                    100
                    *
                    parent_index
                )

                rng = (
                    np.random.default_rng(
                        rng_seed
                    )
                )

                for _ in range(
                    LOCAL_CHILDREN_PER_PARENT
                ):
                    child_hx = clipped(
                        rng.normal(
                            parent_hx,
                            HX_LOCAL_SIGMA,
                        ),
                        HX_BOUNDS,
                    )

                    if y_state == "OFF":
                        child_hy = 0.0

                    else:
                        hy_sigma = max(
                            HY_LOCAL_MIN_SIGMA,
                            HY_LOCAL_REL_SIGMA
                            *
                            abs(
                                parent_hy
                            ),
                        )

                        child_hy = clipped(
                            rng.normal(
                                parent_hy,
                                hy_sigma,
                            ),
                            HY_BOUNDS,
                        )

                        if abs(
                            child_hy
                        ) < 0.05:
                            child_hy = (
                                0.05
                                if parent_hy >= 0
                                else -0.05
                            )

                    child_J = {}

                    for (
                        edge,
                        value,
                    ) in parent_J.items():
                        sigma = max(
                            J_LOCAL_MIN_SIGMA,

                            J_LOCAL_REL_SIGMA
                            *
                            max(
                                abs(
                                    value
                                ),
                                0.5,
                            ),
                        )

                        child_J[
                            edge
                        ] = clipped(
                            rng.normal(
                                value,
                                sigma,
                            ),
                            J_BOUNDS,
                        )

                    evaluate_unique(
                        stage="local",
                        hx=child_hx,
                        hy=child_hy,
                        J=child_J,
                        parent_ids=[
                            parent_config_id
                        ],
                    )

            block_df = pd.DataFrame(
                block_rows
            )

            print()
            print("block best:")

            for readout in READOUTS:
                g = block_df[
                    block_df[
                        "readout"
                    ]
                    ==
                    readout
                ]

                best_cv = (
                    g
                    .sort_values(
                        [
                            "cv_rmse",
                            "process_infidelity",
                        ]
                    )
                    .iloc[0]
                )

                best_fid = (
                    g
                    .sort_values(
                        [
                            "process_infidelity",
                            "cv_rmse",
                        ]
                    )
                    .iloc[0]
                )

                print(
                    f"  {readout:20s} | "
                    f"best CV="
                    f"{best_cv['cv_rmse']:.6f} "
                    f"(hx="
                    f"{best_cv['hx']:.3f}, "
                    f"hy="
                    f"{best_cv['hy']:+.3f}) | "
                    f"best Fproc="
                    f"{best_fid['process_fidelity']:.6f}"
                )

            search_rows_global.extend(
                block_rows
            )

            ridge_rows_global.extend(
                block_ridge_rows
            )

            completed_blocks.add(
                block_key
            )

            save_search_checkpoint(
                search_rows_global,
                ridge_rows_global,
                completed_blocks,
            )

            print(
                f"block runtime: "
                f"{time.perf_counter() - block_start:.1f}s"
            )
            print()

    search_df = pd.DataFrame(
        search_rows_global
    )

    ridge_df = pd.DataFrame(
        ridge_rows_global
    )

    search_df.to_csv(
        RESULTS
        /
        "12_03c_search_all.csv",
        index=False,
    )

    ridge_df.to_csv(
        RESULTS
        /
        "12_03c_ridge_grid.csv",
        index=False,
    )

    return (
        search_df,
        ridge_df,
    )


# =============================================================================
# SEARCH SUMMARIES
# =============================================================================

def summarize_search(
    search_df,
):
    # Best CV row for every historical readout branch and r.
    forecast_best = (
        search_df
        .sort_values(
            [
                "seed_candidate_id",
                "trotter_r",
                "cv_rmse",
                "process_infidelity",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
        )
        .groupby(
            [
                "seed_candidate_id",
                "trotter_r",
            ],
            as_index=False,
            sort=False,
        )
        .first()
    )

    forecast_best.to_csv(
        RESULTS
        /
        "12_03c_forecast_best_by_branch_r.csv",
        index=False,
    )

    fidelity_best = (
        search_df
        .sort_values(
            [
                "seed_candidate_id",
                "trotter_r",
                "process_infidelity",
                "cv_rmse",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
        )
        .groupby(
            [
                "seed_candidate_id",
                "trotter_r",
            ],
            as_index=False,
            sort=False,
        )
        .first()
    )

    fidelity_best.to_csv(
        RESULTS
        /
        "12_03c_fidelity_best_by_branch_r.csv",
        index=False,
    )

    # Best r/config for every readout branch.
    branch_global_best = (
        forecast_best
        .sort_values(
            [
                "seed_candidate_id",
                "cv_rmse",
                "process_infidelity",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .groupby(
            "seed_candidate_id",
            as_index=False,
            sort=False,
        )
        .first()
    )

    branch_global_best.to_csv(
        RESULTS
        /
        "12_03c_branch_global_best.csv",
        index=False,
    )

    # Absolute CV winner.
    global_winner = (
        branch_global_best
        .sort_values(
            [
                "cv_rmse",
                "process_infidelity",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .head(1)
        .copy()
    )

    global_winner.to_csv(
        RESULTS
        /
        "12_03c_global_winner.csv",
        index=False,
    )

    return (
        forecast_best,
        fidelity_best,
        branch_global_best,
        global_winner,
    )


# =============================================================================
# MEMORY ON CV-BEST CONFIGURATIONS
# =============================================================================

def evaluate_memory_config(
    config_row,
):
    topology = str(
        config_row[
            "topology"
        ]
    )

    r = int(
        config_row[
            "trotter_r"
        ]
    )

    hx = float(
        config_row[
            "hx"
        ]
    )

    hy = float(
        config_row[
            "hy"
        ]
    )

    J = J_from_row(
        topology,
        config_row,
    )

    U = trotter_unitary(
        topology,
        hx,
        hy,
        J,
        r,
    )

    seed_rows = []

    for probe_seed in PROBE_SEEDS:
        probe = prev.build_probe(
            REAL_ANGLES,
            int(
                probe_seed
            ),
        )

        A_probe, S_probe = (
            qrc.build_input_channels(
                U,
                probe,
            )
        )

        wash = (
            base.washout_diagnostic(
                S_probe
            )
        )

        Tw = wash[
            "Tw_0.01"
        ]

        if np.isfinite(Tw):
            burnin = int(
                max(
                    MIN_BURNIN,
                    int(Tw)
                    +
                    POST_WASHOUT_MARGIN,
                )
            )

        else:
            burnin = (
                N_TRAIN
                -
                100
            )

        burnin = min(
            burnin,
            N_TRAIN - 100,
        )

        bank = (
            base.build_feature_bank(
                A_probe
            )
        )

        memory = (
            base.evaluate_memory(
                bank,
                probe,
                N_TRAIN,
                N_VAL,
                burnin,
            )
        )

        seed_rows.append({
            "config_id":
                str(
                    config_row[
                        "config_id"
                    ]
                ),

            "topology":
                topology,

            "dynamics_seed_id":
                str(
                    config_row[
                        "dynamics_seed_id"
                    ]
                ),

            "seed_candidate_id":
                str(
                    config_row[
                        "seed_candidate_id"
                    ]
                ),

            "readout":
                str(
                    config_row[
                        "readout"
                    ]
                ),

            "trotter_r":
                r,

            "hx":
                hx,

            "hy":
                hy,

            "probe_seed":
                int(
                    probe_seed
                ),

            "MC_ch":
                memory[
                    "MC_per_channel_corrected"
                ],

            "MC_total":
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
                Tw,

            "washout_valid":
                int(
                    np.isfinite(Tw)
                    and
                    Tw
                    <=
                    MAX_ACCEPTABLE_WASHOUT
                ),

            "postwash_burnin":
                int(
                    burnin
                ),

            "tau_mem":
                wash[
                    "tau_mem"
                ],
        })

    return seed_rows


def run_memory(
    forecast_best,
):
    # Dynamics may be shared by multiple readouts. Evaluate each config once.
    unique_configs = (
        forecast_best
        .sort_values(
            [
                "config_id",
                "cv_rmse",
            ]
        )
        .drop_duplicates(
            subset=[
                "config_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        RESUME_MEMORY
        and
        MEMORY_BY_SEED_FILE.exists()
    ):
        memory_rows = (
            pd.read_csv(
                MEMORY_BY_SEED_FILE
            )
            .to_dict(
                "records"
            )
        )

        completed_ids = set(
            pd.DataFrame(
                memory_rows
            )[
                "config_id"
            ].astype(str)
        )

    else:
        memory_rows = []
        completed_ids = set()

    total = len(
        unique_configs
    )

    for idx, row in unique_configs.iterrows():
        config_id = str(
            row[
                "config_id"
            ]
        )

        if config_id in completed_ids:
            print(
                f"[memory {idx+1}/{total}] "
                f"{config_id}: checkpoint hit"
            )
            continue

        print(
            f"[memory {idx+1}/{total}] "
            f"{config_id}"
        )

        new_rows = evaluate_memory_config(
            row
        )

        memory_rows.extend(
            new_rows
        )

        completed_ids.add(
            config_id
        )

        pd.DataFrame(
            memory_rows
        ).to_csv(
            MEMORY_BY_SEED_FILE,
            index=False,
        )

    memory_df = pd.DataFrame(
        memory_rows
    )

    summary_rows = []

    for config_id, g in memory_df.groupby(
        "config_id"
    ):
        finite_tw = (
            pd.to_numeric(
                g[
                    "Tw_0.01"
                ],
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        summary_rows.append({
            "config_id":
                str(
                    config_id
                ),

            "MC_ch_mean":
                float(
                    g[
                        "MC_ch"
                    ].mean()
                ),

            "MC_ch_std":
                float(
                    g[
                        "MC_ch"
                    ].std(
                        ddof=1
                    )
                ),

            "MC_ch_median":
                float(
                    g[
                        "MC_ch"
                    ].median()
                ),

            "MC_ch_min":
                float(
                    g[
                        "MC_ch"
                    ].min()
                ),

            "MC_ch_max":
                float(
                    g[
                        "MC_ch"
                    ].max()
                ),

            "Tw_0.01_mean":
                (
                    float(
                        np.mean(
                            finite_tw
                        )
                    )
                    if len(
                        finite_tw
                    )
                    else np.nan
                ),

            "Tw_0.01_std":
                (
                    float(
                        np.std(
                            finite_tw,
                            ddof=1,
                        )
                    )
                    if len(
                        finite_tw
                    )
                    > 1
                    else np.nan
                ),

            "valid_washout_seeds":
                int(
                    g[
                        "washout_valid"
                    ].sum()
                ),
        })

    memory_summary = pd.DataFrame(
        summary_rows
    )

    memory_summary.to_csv(
        MEMORY_SUMMARY_FILE,
        index=False,
    )

    return (
        memory_df,
        memory_summary,
    )


# =============================================================================
# FINAL REPORT
# =============================================================================

def print_final_report(
    forecast_best,
    branch_global_best,
    global_winner,
    memory_summary,
):
    with_memory = (
        forecast_best
        .merge(
            memory_summary,
            on="config_id",
            how="left",
        )
    )

    with_memory.to_csv(
        RESULTS
        /
        "12_03c_forecast_best_with_memory.csv",
        index=False,
    )

    print()
    print("=" * 132)
    print("BEST CONFIGURATION FOR EACH H5/H6 READOUT BRANCH")
    print("=" * 132)

    show = (
        branch_global_best[
            [
                "topology",
                "seed_hy",
                "readout",
                "trotter_r",
                "hx",
                "hy",
                "selected_lambda",
                "cv_rmse",
                "cv_rmse_std",
                "validation_rmse",
                "process_fidelity",
            ]
        ]
        .sort_values(
            [
                "cv_rmse",
                "topology",
            ]
        )
    )

    print(
        show.to_string(
            index=False
        )
    )

    print()
    print("=" * 132)
    print("ABSOLUTE WEEK-12.3C CV WINNER")
    print("=" * 132)

    w = global_winner.iloc[0]

    print(
        f"topology       = {w['topology']}"
    )
    print(
        f"seed_hy        = {float(w['seed_hy']):+.6f}"
    )
    print(
        f"readout        = {w['readout']}"
    )
    print(
        f"r              = {int(w['trotter_r'])}"
    )
    print(
        f"hx             = {float(w['hx']):+.12f}"
    )
    print(
        f"hy             = {float(w['hy']):+.12f}"
    )
    print(
        f"lambda         = {float(w['selected_lambda']):g}"
    )
    print(
        f"CV RMSE        = {float(w['cv_rmse']):.6f}"
    )
    print(
        f"CV SD          = {float(w['cv_rmse_std']):.6f}"
    )
    print(
        f"2025 RMSE      = {float(w['validation_rmse']):.6f} "
        f"(diagnostic only)"
    )
    print(
        f"F_process      = {float(w['process_fidelity']):.6f}"
    )

    print()
    print("Standing same-sample comparisons:")

    cv = float(
        w[
            "cv_rmse"
        ]
    )

    val = float(
        w[
            "validation_rmse"
        ]
    )

    qrc_delta = (
        cv
        -
        BEST_QRC_CV
    )

    classical_delta = (
        cv
        -
        BEST_CLASSICAL_CV
    )

    print(
        f"  vs best QRC CV {BEST_QRC_CV:.6f}: "
        f"delta={qrc_delta:+.6f}"
    )

    if cv < BEST_QRC_CV:
        print(
            "  >>> NEW QRC TRAINING-CV BEST <<<"
        )

    print(
        f"  vs best classical CV {BEST_CLASSICAL_CV:.6f}: "
        f"delta={classical_delta:+.6f}"
    )

    if cv < BEST_CLASSICAL_CV:
        print(
            "  >>> BEATS CURRENT CLASSICAL TRAINING-CV BENCHMARK <<<"
        )

    print(
        f"  2025 diagnostic vs best ideal QRC {BEST_IDEAL_QRC_2025:.6f}: "
        f"delta={val - BEST_IDEAL_QRC_2025:+.6f}"
    )

    print(
        f"  2025 diagnostic vs best classical {BEST_CLASSICAL_2025:.6f}: "
        f"delta={val - BEST_CLASSICAL_2025:+.6f}"
    )

    if val < BEST_IDEAL_QRC_2025:
        print(
            "  >>> CV-SELECTED WINNER ALSO SETS A NEW IDEAL-QRC 2025 DIAGNOSTIC BEST <<<"
        )
    elif val < BEST_CLASSICAL_2025:
        print(
            "  >>> CV-SELECTED WINNER IS BELOW THE CURRENT CLASSICAL 2025 BENCHMARK <<<"
        )

    # Distribution of winning r values across branch/readout seeds.
    r_counts = (
        branch_global_best[
            "trotter_r"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("Best-r counts across retained readout branches:")

    for r in R_VALUES:
        print(
            f"  r={r}: "
            f"{int(r_counts.get(r, 0))}"
        )

    print()
    print(
        "Reminder: all model selection above uses only 2022-2024 chronological CV. "
        "2025 is diagnostic; 2026 remains untouched."
    )


# =============================================================================
# MANIFEST
# =============================================================================

def save_manifest(
    search_df,
    branch_global_best,
    global_winner,
):
    manifest = {
        "step":
            "12.3C",

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "purpose":
            (
                "Fair H5/H6 r=1,2,3 re-optimization over "
                "hx,hy,J,Ridge lambda and three readout families."
            ),

        "candidate_universe": {
            "hy_seeds":
                HY_SEEDS,

            "readouts":
                READOUTS,

            "dynamics_seed_count":
                int(
                    candidate_universe[
                        "dynamics_seed_id"
                    ].nunique()
                ),

            "readout_branch_count":
                int(
                    len(
                        candidate_universe
                    )
                ),

            "seed_rule":
                (
                    "Y-OFF h_y=0 retained for each topology plus all "
                    "12.3B Pareto h_y seeds."
                ),
        },

        "frozen": {
            "alpha":
                ALPHA,

            "dt":
                DT,

            "J0_source":
                str(
                    J_WINNER_FILE
                ),

            "J0_meaning":
                (
                    "Week-12.3A intrinsic-memory winner for each topology."
                ),
        },

        "search": {
            "r_values":
                R_VALUES,

            "hx_reference":
                HX_REFERENCE,

            "hx_grid":
                HX_GRID,

            "J_scale_grid":
                J_SCALE_GRID,

            "YON_hy_rule":
                "{0.5*h0, 1.0*h0, 1.5*h0, -1.0*h0}",

            "YOFF_hy":
                0.0,

            "local_children_per_parent":
                LOCAL_CHILDREN_PER_PARENT,

            "local_parent_union":
                (
                    "best CV parent for each readout plus best "
                    "process-fidelity parent"
                ),

            "hx_bounds":
                HX_BOUNDS,

            "hy_bounds":
                HY_BOUNDS,

            "J_bounds":
                J_BOUNDS,

            "weighted_score":
                False,
        },

        "forecast": {
            "ridge_grid":
                [
                    float(x)
                    for x
                    in prev.RIDGE_GRID
                ],

            "cv_folds":
                prev.N_CV_SPLITS,

            "selection":
                "2022-2024 chronological CV only",

            "validation_2025":
                "diagnostic only",

            "test_2026_used":
                False,
        },

        "memory": {
            "probe_seeds":
                PROBE_SEEDS,

            "observable":
                "XYZ_all",

            "burnin":
                "max(100,Tw(0.01)+20)",

            "washout_validity":
                (
                    f"finite Tw(0.01) <= "
                    f"{MAX_ACCEPTABLE_WASHOUT}"
                ),
        },

        "hardware": {
            "IQM_QPU_jobs":
                0,

            "shots":
                0,

            "noise_model":
                False,

            "note":
                (
                    "Ideal logical calibration only; IQM circuit/noise stage follows."
                ),
        },

        "standing_thresholds": {
            "best_QRC_CV":
                BEST_QRC_CV,

            "best_classical_CV":
                BEST_CLASSICAL_CV,

            "best_ideal_QRC_2025":
                BEST_IDEAL_QRC_2025,

            "best_classical_2025":
                BEST_CLASSICAL_2025,
        },

        "counts": {
            "search_rows":
                int(
                    len(
                        search_df
                    )
                ),

            "branch_global_best_rows":
                int(
                    len(
                        branch_global_best
                    )
                ),
        },

        "global_winner":
            (
                global_winner
                .iloc[0]
                .to_dict()
            ),
    }

    with open(
        RESULTS
        /
        "12_03c_manifest.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            manifest,
            fp,
            indent=2,
            default=str,
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 132)
    print("WEEK 12 - STEP 12.3C")
    print("FAIR H5/H6 r / hx / hy / J / lambda RE-OPTIMIZATION")
    print("=" * 132)
    print()

    print("Frozen:")
    print(
        f"  alpha = {ALPHA}"
    )
    print(
        f"  dt    = {DT}"
    )
    print(
        "  J0    = Week-12.3A topology memory winner"
    )
    print()

    print("Reduced h_y seed universe:")
    for topology in TOPOLOGIES:
        print(
            f"  {topology}: "
            f"{HY_SEEDS[topology]}"
        )

    print()
    print(
        f"Dynamics seeds: "
        f"{candidate_universe['dynamics_seed_id'].nunique()}"
    )
    print(
        f"Readout branches: "
        f"{len(candidate_universe)}"
    )
    print(
        f"r values: {R_VALUES}"
    )
    print()

    print(
        "Fairness: every retained dynamics seed receives the same "
        "r/hx/hy/J/local/Ridge search logic."
    )
    print(
        "Selection: 2022-2024 chronological CV only; "
        "2025 diagnostic; 2026 untouched."
    )
    print(
        "No IQM QPU jobs are submitted."
    )
    print()

    search_df, ridge_df = run_search()

    (
        forecast_best,
        fidelity_best,
        branch_global_best,
        global_winner,
    ) = summarize_search(
        search_df
    )

    print()
    print("=" * 132)
    print("MEMORY / WASHOUT ON CV-BEST BRANCH-r CONFIGURATIONS")
    print("=" * 132)

    (
        memory_df,
        memory_summary,
    ) = run_memory(
        forecast_best
    )

    print_final_report(
        forecast_best,
        branch_global_best,
        global_winner,
        memory_summary,
    )

    save_manifest(
        search_df,
        branch_global_best,
        global_winner,
    )

    print()
    print("=" * 132)
    print("FILES SAVED")
    print("=" * 132)

    for name in [
        "12_03c_candidate_universe.csv",
        "12_03c_search_all.csv",
        "12_03c_ridge_grid.csv",
        "12_03c_forecast_best_by_branch_r.csv",
        "12_03c_fidelity_best_by_branch_r.csv",
        "12_03c_branch_global_best.csv",
        "12_03c_memory_by_seed.csv",
        "12_03c_memory_summary.csv",
        "12_03c_forecast_best_with_memory.csv",
        "12_03c_global_winner.csv",
        "12_03c_manifest.json",
    ]:
        print(
            f"results/{name}"
        )

    print()
    print("Step 12.3C complete.")
    print(
        "Interpret the fair H5/H6 calibration before IQM real-circuit/noise selection."
    )


if __name__ == "__main__":
    main()
