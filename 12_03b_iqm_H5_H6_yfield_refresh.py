"""
Week 12 - Step 12.3B
Symmetric memory-only Y-field refresh for IQM-native H5/H6.

PURPOSE
-------
Reproduce the Week-9 Y-field refresh logic for the two new IQM-native
topologies H5 and H6 before the later fair r=1,2,3 re-optimization.

The Week-12.3A J winners are frozen as the reference couplings in this step.
We scan the SAME symmetric h_y grid used in the IBM Week-9 refresh:

    h_y in {-0.6,-0.4,-0.3,-0.2,0,+0.2,+0.3,+0.4,+0.6}

For every topology/h_y:
  1. evaluate intrinsic memory over the SAME five paired randomized probes,
  2. evaluate stable washout T_w(0.01),
  3. evaluate chronological 2022-2024 forecast CV,
  4. reselect Ridge lambda from the frozen grid,
  5. recheck all three Week-9 readout families.

This step does NOT freeze one Y-ON winner yet. It saves the full H5/H6
h_y landscape and a transparent Pareto set. The result will define the
Y-OFF/Y-ON seed universe for the next fair (J,hx,hy,lambda,r) search.

FROZEN
------
alpha       = 0.75
hx          = 0.5
dt          = 1.6
Trotter r   = 2
J           = Week-12.3A topology-specific memory winner
initial M   = |00><00|

HAMILTONIAN
-----------
H = sum_(i,j) J_ij Z_i Z_j
    + h_x sum_i X_i
    + h_y (Y_4 + Y_5)

PROBE SEEDS
-----------
[79001, 42, 101, 505, 707]

MC
--
XYZ_all readout.
B = max(100, T_w(0.01)+20) separately for every topology/h_y/probe seed.
Corrected MC uses the same finite-validation null-floor correction as 12.3A.

FORECAST READOUTS
-----------------
XZ_injection
XZinj_plus_YX45
XYZ_all

Ridge lambda grid:
[1e-4,1e-3,1e-2,1e-1,1,10,100,300]

Selection uses 2022-2024 chronological five-fold CV only.
2025 is diagnostic only.
2026 is never loaded or used for selection.

HARDWARE
--------
No IQM QPU job, shots, backend noise, or hardware calibration is used here.
This is an ideal logical-dynamics calibration step.

DEPENDENCIES
------------
Keep beside this script:
    07_02b_memory_pair_isolation.py
    12_03a_iqm_H5_H6_topology_J_mc_rmse_search.py

Required result:
    results/12_03a_J_search_topology_winners.csv

OUTPUTS
-------
results/12_03b_yfield_memory_by_seed.csv
results/12_03b_yfield_forecast_all.csv
results/12_03b_yfield_ridge_grid.csv
results/12_03b_yfield_summary.csv
results/12_03b_yfield_pareto.csv
results/12_03b_yoff_reproduction_audit.csv
results/12_03b_yfield_manifest.json
"""

from __future__ import annotations

import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit


# =============================================================================
# PATHS / DEPENDENCIES
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

BASE_SCRIPT = HERE / "12_03a_iqm_H5_H6_topology_J_mc_rmse_search.py"
J_WINNER_FILE = RESULTS / "12_03a_J_search_topology_winners.csv"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing {BASE_SCRIPT.name}. Keep the Week-12.3A script beside this script."
    )

if not J_WINNER_FILE.exists():
    raise FileNotFoundError(
        f"Missing {J_WINNER_FILE}. Run Week 12.3A first."
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = load_module(
    BASE_SCRIPT,
    "qrc_week12_03b_yfield_base",
)

qrc = base.qrc


# =============================================================================
# FROZEN SETTINGS
# =============================================================================

TOPOLOGIES = ["H5", "H6"]

ALPHA = 0.75
HX = 0.5
DT = 1.6
TROTTER_R = 2

HY_GRID = [
    -0.6,
    -0.4,
    -0.3,
    -0.2,
     0.0,
    +0.2,
    +0.3,
    +0.4,
    +0.6,
]

PROBE_SEEDS = [
    79001,
    42,
    101,
    505,
    707,
]

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

K_MAX = base.K_MAX
MIN_BURNIN = base.MIN_BURNIN
POST_WASHOUT_MARGIN = base.POST_WASHOUT_MARGIN
MAX_ACCEPTABLE_WASHOUT = base.MAX_ACCEPTABLE_WASHOUT
MC_RIDGE_ALPHA = base.MC_RIDGE_ALPHA

N_TRAIN_EXPECTED = 1095
N_VAL_EXPECTED = 365

qrc.ALPHA = ALPHA
qrc.HX = HX
qrc.DT = DT
qrc.TROTTER_R = TROTTER_R


# =============================================================================
# READOUTS
# =============================================================================

READOUT_FEATURES = {
    "XZ_injection": list(base.XZ_INJECTION),

    "XZinj_plus_YX45":
        list(base.XZ_INJECTION)
        + ["YX_45"],

    "XYZ_all": list(base.XYZ_ALL),
}

READOUT_SETTINGS = {
    "XZ_injection": 2,
    "XZinj_plus_YX45": 2,
    "XYZ_all": 3,
}


# =============================================================================
# LOAD FROZEN H5/H6 J WINNERS
# =============================================================================

j_df = pd.read_csv(J_WINNER_FILE)

REFERENCE_J = {}
REFERENCE_MEMORY = {}

for topology in TOPOLOGIES:
    sub = j_df[
        j_df["topology"].astype(str) == topology
    ].copy()

    if len(sub) != 1:
        raise RuntimeError(
            f"{topology}: expected exactly one row in {J_WINNER_FILE}, got {len(sub)}."
        )

    row = sub.iloc[0]

    J = {}

    for edge in base.CANDIDATES[topology]["edges"]:
        col = f"J{edge[0]}{edge[1]}"

        if col not in row.index or pd.isna(row[col]):
            raise RuntimeError(
                f"{topology}: missing frozen coupling {col} in {J_WINNER_FILE}."
            )

        J[edge] = float(row[col])

    REFERENCE_J[topology] = J

    REFERENCE_MEMORY[topology] = {
        "MC_per_channel_corrected":
            float(row["MC_per_channel_corrected"]),
        "MC_total_corrected":
            float(row["MC_total_corrected"]),
        "Tw_0.01":
            float(row["Tw_0.01"]),
    }


# =============================================================================
# DATA — TRAIN + 2025 VALIDATION ONLY
# =============================================================================

work, train, val, cols = qrc.load_data()

if len(train) != N_TRAIN_EXPECTED or len(val) != N_VAL_EXPECTED:
    raise RuntimeError(
        f"Expected train/validation={N_TRAIN_EXPECTED}/{N_VAL_EXPECTED}; "
        f"got {len(train)}/{len(val)}."
    )

work_tv = pd.concat(
    [train, val],
    axis=0,
).reset_index(drop=True)

N_TRAIN = len(train)
N_VAL = len(val)
N_TOTAL = len(work_tv)

REAL_ANGLES = np.asarray(
    qrc.make_input_angles(
        work_tv,
        cols,
    ),
    dtype=float,
)

Y_TARGET = work_tv[
    cols["target"]
].to_numpy(dtype=float)


# =============================================================================
# UNITARY WITH MEMORY-ONLY Y FIELD
# =============================================================================

def pauli_gate(
    operator,
    physical_rotation_angle,
):
    """
    For Pauli P:
        R_P(theta) = exp(-i theta P / 2)
    """
    theta = float(physical_rotation_angle)

    return (
        np.cos(theta / 2.0) * qrc.I64
        -
        1j
        * np.sin(theta / 2.0)
        * operator
    )


def build_unitary(
    topology: str,
    J: dict,
    hy: float,
):
    """
    First-order product formula:

        [ exp(-i H_ZZ dt/r)
          exp(-i H_X  dt/r)
          exp(-i H_Y  dt/r) ]^r

    with
        theta_ZZ = 2 J_ij dt/r
        theta_X  = 2 hx dt/r
        theta_Y  = 2 hy dt/r
    """

    U = qrc.I64.copy()

    for _ in range(TROTTER_R):

        # ZZ layer
        for edge in base.CANDIDATES[topology]["edges"]:
            theta_zz = (
                2.0
                * float(J[edge])
                * DT
                / TROTTER_R
            )

            U = (
                pauli_gate(
                    base.ZZ_OPS[edge],
                    theta_zz,
                )
                @ U
            )

        # X layer
        theta_x = (
            2.0
            * HX
            * DT
            / TROTTER_R
        )

        for qubit in range(qrc.N_QUBITS):
            U = (
                pauli_gate(
                    qrc.FULL_SINGLE_OPS[
                        f"X{qubit}"
                    ],
                    theta_x,
                )
                @ U
            )

        # Memory-only Y layer
        if not np.isclose(hy, 0.0):
            theta_y = (
                2.0
                * float(hy)
                * DT
                / TROTTER_R
            )

            for qubit in [4, 5]:
                U = (
                    pauli_gate(
                        qrc.FULL_SINGLE_OPS[
                            f"Y{qubit}"
                        ],
                        theta_y,
                    )
                    @ U
                )

    error = float(
        np.linalg.norm(
            U.conj().T @ U - qrc.I64,
            ord="fro",
        )
    )

    return U, error


# =============================================================================
# RANDOMIZED F4 PROBE
# =============================================================================

def build_probe(
    real_angles,
    seed: int,
):
    rng = np.random.default_rng(
        int(seed)
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
        probe[:, channel] = (
            real_angles[
                rng.permutation(
                    len(real_angles)
                ),
                channel,
            ]
        )

    return probe


def probe_audit(
    real_angles,
    probe,
):
    real_angles = np.asarray(
        real_angles,
        dtype=float,
    )

    probe = np.asarray(
        probe,
        dtype=float,
    )

    max_marginal_error = 0.0

    for channel in range(
        probe.shape[1]
    ):
        a = np.sort(
            real_angles[:, channel]
        )

        b = np.sort(
            probe[:, channel]
        )

        max_marginal_error = max(
            max_marginal_error,
            float(
                np.max(
                    np.abs(a - b)
                )
            ),
        )

    max_abs_ac = 0.0

    for channel in range(
        probe.shape[1]
    ):
        for lag in range(1, 21):
            corr = np.corrcoef(
                probe[lag:, channel],
                probe[:-lag, channel],
            )[0, 1]

            if np.isfinite(corr):
                max_abs_ac = max(
                    max_abs_ac,
                    abs(float(corr)),
                )

    return {
        "max_marginal_error":
            float(max_marginal_error),

        "max_abs_autocorrelation_lag1_20":
            float(max_abs_ac),
    }


# =============================================================================
# FORECASTING
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
                (y_true - y_pred) ** 2
            )
        )
    )


def bias(
    y_true,
    y_pred,
):
    return float(
        np.mean(
            np.asarray(
                y_pred,
                dtype=float,
            )
            -
            np.asarray(
                y_true,
                dtype=float,
            )
        )
    )


def select_lambda_and_validate(
    X,
):
    """
    Ridge lambda is selected using ONLY 2022-2024 chronological CV.
    The 2025 validation result is diagnostic only.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    X_train = X[:N_TRAIN]
    X_val = X[N_TRAIN:]

    y_train = Y_TARGET[:N_TRAIN]
    y_val = Y_TARGET[N_TRAIN:]

    splitter = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )

    ridge_rows = []

    for ridge_alpha in RIDGE_GRID:
        fold_scores = []

        for tr_idx, cv_idx in splitter.split(
            X_train
        ):
            pred = (
                base.standardized_ridge_predict(

                    X_train[
                        tr_idx
                    ],

                    y_train[
                        tr_idx
                    ],

                    X_train[
                        cv_idx
                    ],

                    float(
                        ridge_alpha
                    ),
                )
            )

            fold_scores.append(
                rmse(
                    y_train[
                        cv_idx
                    ],
                    pred,
                )
            )

        ridge_rows.append({
            "ridge_alpha":
                float(ridge_alpha),

            "cv_rmse_mean":
                float(
                    np.mean(
                        fold_scores
                    )
                ),

            "cv_rmse_std":
                float(
                    np.std(
                        fold_scores,
                        ddof=1,
                    )
                ),
        })

    ridge_df = (
        pd.DataFrame(
            ridge_rows
        )
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

    best = ridge_df.iloc[0]

    selected_lambda = float(
        best[
            "ridge_alpha"
        ]
    )

    pred_val = (
        base.standardized_ridge_predict(

            X_train,
            y_train,

            X_val,

            selected_lambda,
        )
    )

    return {
        "selected_lambda":
            selected_lambda,

        "cv_rmse":
            float(
                best[
                    "cv_rmse_mean"
                ]
            ),

        "cv_rmse_std":
            float(
                best[
                    "cv_rmse_std"
                ]
            ),

        "validation_rmse":
            rmse(
                y_val,
                pred_val,
            ),

        "validation_bias":
            bias(
                y_val,
                pred_val,
            ),

        "ridge_grid":
            ridge_df,
    }


# =============================================================================
# PARETO
# =============================================================================

def pareto_mask(
    dataframe: pd.DataFrame,
):
    """
    Objectives:
      CV RMSE        minimize
      MC/ch mean     maximize
      Tw mean        minimize
      n settings     minimize

    No weighted score.
    """

    if len(dataframe) == 0:
        return pd.Series(
            dtype=bool
        )

    values = dataframe[
        [
            "best_cv_rmse",
            "MC_ch_mean",
            "Tw_mean",
            "best_n_settings",
        ]
    ].to_numpy(dtype=float)

    # Convert to all-minimize form.
    transformed = values.copy()
    transformed[:, 1] *= -1.0

    keep = np.ones(
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
            keep[i] = False
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

            no_worse = np.all(
                transformed[j]
                <=
                transformed[i]
            )

            strictly_better = np.any(
                transformed[j]
                <
                transformed[i]
            )

            if (
                no_worse
                and
                strictly_better
            ):
                keep[i] = False
                break

    return pd.Series(
        keep,
        index=dataframe.index,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 132)
    print("WEEK 12 - STEP 12.3B")
    print("SYMMETRIC MEMORY-ONLY Y-FIELD REFRESH — H5/H6")
    print("=" * 132)
    print()

    print("Frozen:")
    print(f"  alpha       = {ALPHA}")
    print(f"  hx          = {HX}")
    print(f"  dt          = {DT}")
    print(f"  Trotter r   = {TROTTER_R}")
    print("  J           = Week-12.3A topology memory winner")
    print()

    print(f"h_y grid: {HY_GRID}")
    print(f"Probe seeds: {PROBE_SEEDS}")
    print(
        "MC burn-in: B=max(100,Tw(0.01)+20), "
        "computed separately per topology/h_y/probe seed."
    )
    print(
        "Forecasting: 2022-2024 chronological CV selects lambda; "
        "2025 diagnostic only; 2026 untouched."
    )
    print(
        f"Readouts: {list(READOUT_FEATURES.keys())}"
    )
    print()

    # -------------------------------------------------------------------------
    # Probe audits
    # -------------------------------------------------------------------------

    probe_audit_rows = []

    print("Probe audit:")

    for seed in PROBE_SEEDS:
        probe = build_probe(
            REAL_ANGLES,
            seed,
        )

        audit = probe_audit(
            REAL_ANGLES,
            probe,
        )

        probe_audit_rows.append({
            "probe_seed":
                int(seed),

            **audit,
        })

        print(
            f"  seed={seed}: "
            f"marginal_error="
            f"{audit['max_marginal_error']:.3e}, "
            f"max|AC|_lag1..20="
            f"{audit['max_abs_autocorrelation_lag1_20']:.9f}"
        )

    print()

    # -------------------------------------------------------------------------
    # Full scan
    # -------------------------------------------------------------------------

    memory_rows = []
    forecast_rows = []
    ridge_rows = []
    summary_rows = []
    audit_rows = []

    for topology in TOPOLOGIES:
        print("-" * 132)
        print(
            f"{topology}: "
            f"{base.CANDIDATES[topology]['description']}"
        )
        print("-" * 132)

        J = REFERENCE_J[topology]

        print("Frozen J:")

        for edge in base.CANDIDATES[topology]["edges"]:
            print(
                f"  J{edge[0]}{edge[1]}="
                f"{J[edge]:+.6f}"
            )

        print()

        topology_hy_rows = []

        for hy in HY_GRID:
            t0 = time.perf_counter()

            U, unitarity_error = (
                build_unitary(
                    topology,
                    J,
                    hy,
                )
            )

            # -------------------------------------------------------------
            # Forecasting: real chronology once per h_y
            # -------------------------------------------------------------

            A_real, _ = (
                qrc.build_input_channels(
                    U,
                    REAL_ANGLES,
                )
            )

            bank_real = (
                base.build_feature_bank(
                    A_real
                )
            )

            hy_forecast_rows = []

            for (
                readout,
                feature_names
            ) in READOUT_FEATURES.items():

                missing = [
                    name
                    for name
                    in feature_names
                    if name
                    not in bank_real.columns
                ]

                if missing:
                    raise RuntimeError(
                        f"{topology} h_y={hy:+.2f}: "
                        f"missing readout features {missing}"
                    )

                X = (
                    bank_real[
                        feature_names
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                forecast = (
                    select_lambda_and_validate(
                        X
                    )
                )

                row = {
                    "topology":
                        topology,

                    "hy":
                        float(hy),

                    "y_state":
                        (
                            "OFF"
                            if np.isclose(
                                hy,
                                0.0
                            )
                            else "ON"
                        ),

                    "readout":
                        readout,

                    "n_features":
                        int(
                            len(
                                feature_names
                            )
                        ),

                    "n_settings":
                        int(
                            READOUT_SETTINGS[
                                readout
                            ]
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

                    "unitarity_error":
                        unitarity_error,
                }

                hy_forecast_rows.append(
                    row
                )

                forecast_rows.append(
                    row
                )

                grid = (
                    forecast[
                        "ridge_grid"
                    ]
                    .copy()
                )

                grid[
                    "topology"
                ] = topology

                grid[
                    "hy"
                ] = float(hy)

                grid[
                    "readout"
                ] = readout

                ridge_rows.extend(
                    grid.to_dict(
                        "records"
                    )
                )

            # -------------------------------------------------------------
            # Memory: five paired probe seeds
            # -------------------------------------------------------------

            hy_memory_rows = []

            for seed in PROBE_SEEDS:
                probe = build_probe(
                    REAL_ANGLES,
                    seed,
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

                Tw = (
                    wash[
                        "Tw_0.01"
                    ]
                )

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

                feature_bank = (
                    base.build_feature_bank(
                        A_probe
                    )
                )

                memory = (
                    base.evaluate_memory(
                        feature_bank,
                        probe,
                        N_TRAIN,
                        N_VAL,
                        burnin,
                    )
                )

                washout_valid = int(
                    np.isfinite(Tw)
                    and
                    Tw
                    <=
                    MAX_ACCEPTABLE_WASHOUT
                )

                mrow = {
                    "topology":
                        topology,

                    "hy":
                        float(hy),

                    "y_state":
                        (
                            "OFF"
                            if np.isclose(
                                hy,
                                0.0
                            )
                            else "ON"
                        ),

                    "probe_seed":
                        int(seed),

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
                        washout_valid,

                    "postwash_burnin":
                        int(
                            burnin
                        ),

                    "tau_mem":
                        wash[
                            "tau_mem"
                        ],

                    "kappa_eff":
                        wash[
                            "kappa_eff"
                        ],

                    "unitarity_error":
                        unitarity_error,
                }

                memory_rows.append(
                    mrow
                )

                hy_memory_rows.append(
                    mrow
                )

            # -------------------------------------------------------------
            # Aggregate one topology/h_y
            # -------------------------------------------------------------

            mem_df = pd.DataFrame(
                hy_memory_rows
            )

            fc_df = (
                pd.DataFrame(
                    hy_forecast_rows
                )
                .sort_values(
                    [
                        "cv_rmse",
                        "n_settings",
                        "n_features",
                        "selected_lambda",
                    ],
                    ascending=[
                        True,
                        True,
                        True,
                        True,
                    ],
                )
                .reset_index(
                    drop=True
                )
            )

            best_fc = fc_df.iloc[0]

            finite_tw = (
                mem_df[
                    "Tw_0.01"
                ]
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

            summary = {
                "topology":
                    topology,

                "hy":
                    float(hy),

                "y_state":
                    (
                        "OFF"
                        if np.isclose(
                            hy,
                            0.0
                        )
                        else "ON"
                    ),

                "n_probe_seeds":
                    int(
                        len(
                            mem_df
                        )
                    ),

                "valid_washout_seeds":
                    int(
                        mem_df[
                            "washout_valid"
                        ].sum()
                    ),

                "all_washout_valid":
                    bool(
                        (
                            mem_df[
                                "washout_valid"
                            ]
                            == 1
                        ).all()
                    ),

                "MC_ch_mean":
                    float(
                        mem_df[
                            "MC_ch"
                        ].mean()
                    ),

                "MC_ch_std":
                    float(
                        mem_df[
                            "MC_ch"
                        ].std(
                            ddof=1
                        )
                    ),

                "MC_ch_median":
                    float(
                        mem_df[
                            "MC_ch"
                        ].median()
                    ),

                "MC_ch_min":
                    float(
                        mem_df[
                            "MC_ch"
                        ].min()
                    ),

                "MC_ch_max":
                    float(
                        mem_df[
                            "MC_ch"
                        ].max()
                    ),

                "Tw_mean":
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

                "Tw_std":
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

                "best_readout":
                    str(
                        best_fc[
                            "readout"
                        ]
                    ),

                "best_n_features":
                    int(
                        best_fc[
                            "n_features"
                        ]
                    ),

                "best_n_settings":
                    int(
                        best_fc[
                            "n_settings"
                        ]
                    ),

                "best_lambda":
                    float(
                        best_fc[
                            "selected_lambda"
                        ]
                    ),

                "best_cv_rmse":
                    float(
                        best_fc[
                            "cv_rmse"
                        ]
                    ),

                "best_cv_rmse_std":
                    float(
                        best_fc[
                            "cv_rmse_std"
                        ]
                    ),

                "best_validation_rmse":
                    float(
                        best_fc[
                            "validation_rmse"
                        ]
                    ),

                "best_validation_bias":
                    float(
                        best_fc[
                            "validation_bias"
                        ]
                    ),

                "unitarity_error":
                    unitarity_error,
            }

            summary_rows.append(
                summary
            )

            topology_hy_rows.append(
                summary
            )

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            print(
                f"h_y={hy:+.2f} | "
                f"MC/ch="
                f"{summary['MC_ch_mean']:.6f}"
                f"±"
                f"{summary['MC_ch_std']:.6f} | "
                f"Tw={summary['Tw_mean']:.1f} | "
                f"valid="
                f"{summary['valid_washout_seeds']}/"
                f"{len(PROBE_SEEDS)} | "
                f"{summary['best_readout']} "
                f"CV="
                f"{summary['best_cv_rmse']:.6f} "
                f"(lambda="
                f"{summary['best_lambda']:g}) | "
                f"time={elapsed:.1f}s"
            )

            # -------------------------------------------------------------
            # h_y=0 reproduction audit against 12.3A seed 79001
            # -------------------------------------------------------------

            if np.isclose(
                hy,
                0.0
            ):
                seed0 = mem_df[
                    mem_df[
                        "probe_seed"
                    ]
                    ==
                    79001
                ]

                if len(seed0) != 1:
                    raise RuntimeError(
                        f"{topology}: missing unique seed-79001 Y-OFF memory row."
                    )

                observed = seed0.iloc[0]

                expected = (
                    REFERENCE_MEMORY[
                        topology
                    ]
                )

                audit_rows.append({
                    "topology":
                        topology,

                    "expected_MC_ch":
                        expected[
                            "MC_per_channel_corrected"
                        ],

                    "observed_MC_ch":
                        float(
                            observed[
                                "MC_ch"
                            ]
                        ),

                    "MC_ch_error":
                        float(
                            observed[
                                "MC_ch"
                            ]
                        )
                        -
                        expected[
                            "MC_per_channel_corrected"
                        ],

                    "expected_MC_total":
                        expected[
                            "MC_total_corrected"
                        ],

                    "observed_MC_total":
                        float(
                            observed[
                                "MC_total"
                            ]
                        ),

                    "MC_total_error":
                        float(
                            observed[
                                "MC_total"
                            ]
                        )
                        -
                        expected[
                            "MC_total_corrected"
                        ],

                    "expected_Tw":
                        expected[
                            "Tw_0.01"
                        ],

                    "observed_Tw":
                        float(
                            observed[
                                "Tw_0.01"
                            ]
                        ),

                    "Tw_error":
                        float(
                            observed[
                                "Tw_0.01"
                            ]
                        )
                        -
                        expected[
                            "Tw_0.01"
                        ],
                })

        # -----------------------------------------------------------------
        # Console summary per topology
        # -----------------------------------------------------------------

        topology_df = (
            pd.DataFrame(
                topology_hy_rows
            )
        )

        print()
        print(
            f"{topology} best training-CV h_y/readout:"
        )

        best_forecast = (
            topology_df
            .sort_values(
                [
                    "best_cv_rmse",
                    "best_n_settings",
                    "best_n_features",
                ]
            )
            .iloc[0]
        )

        print(
            f"  h_y={best_forecast['hy']:+.2f} | "
            f"{best_forecast['best_readout']} | "
            f"CV={best_forecast['best_cv_rmse']:.6f} | "
            f"MC/ch={best_forecast['MC_ch_mean']:.6f} | "
            f"Tw={best_forecast['Tw_mean']:.1f}"
        )

        valid_memory = topology_df[
            topology_df[
                "all_washout_valid"
            ].astype(bool)
        ].copy()

        if len(valid_memory):
            best_memory = (
                valid_memory
                .sort_values(
                    [
                        "MC_ch_mean",
                        "best_cv_rmse",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                .iloc[0]
            )

            print(
                f"{topology} best mean-memory h_y:"
            )

            print(
                f"  h_y={best_memory['hy']:+.2f} | "
                f"MC/ch={best_memory['MC_ch_mean']:.6f} | "
                f"Tw={best_memory['Tw_mean']:.1f} | "
                f"best CV={best_memory['best_cv_rmse']:.6f}"
            )

        print()

    # =========================================================================
    # SAVE
    # =========================================================================

    memory_df = pd.DataFrame(
        memory_rows
    )

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    ridge_df = pd.DataFrame(
        ridge_rows
    )

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "topology",
                "hy",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    # Pareto set is formed separately per topology.
    pareto_parts = []

    for topology in TOPOLOGIES:
        sub = summary_df[
            summary_df[
                "topology"
            ]
            ==
            topology
        ].copy()

        # CONT candidates should have stable washout for all five probes.
        eligible = sub[
            sub[
                "all_washout_valid"
            ].astype(bool)
        ].copy()

        if len(eligible) == 0:
            eligible = sub.copy()

        eligible[
            "is_pareto"
        ] = pareto_mask(
            eligible
        )

        pareto_parts.append(
            eligible[
                eligible[
                    "is_pareto"
                ]
            ].copy()
        )

    pareto_df = pd.concat(
        pareto_parts,
        ignore_index=True,
    )

    memory_df.to_csv(
        RESULTS
        /
        "12_03b_yfield_memory_by_seed.csv",
        index=False,
    )

    forecast_df.to_csv(
        RESULTS
        /
        "12_03b_yfield_forecast_all.csv",
        index=False,
    )

    ridge_df.to_csv(
        RESULTS
        /
        "12_03b_yfield_ridge_grid.csv",
        index=False,
    )

    summary_df.to_csv(
        RESULTS
        /
        "12_03b_yfield_summary.csv",
        index=False,
    )

    pareto_df.to_csv(
        RESULTS
        /
        "12_03b_yfield_pareto.csv",
        index=False,
    )

    audit_df.to_csv(
        RESULTS
        /
        "12_03b_yoff_reproduction_audit.csv",
        index=False,
    )

    # =========================================================================
    # REPRODUCTION AUDIT
    # =========================================================================

    print("=" * 132)
    print("12.3A Y-OFF REPRODUCTION AUDIT — PROBE SEED 79001")
    print("=" * 132)

    print(
        audit_df.to_string(
            index=False
        )
    )

    mc_tol = 2e-5
    tw_tol = 0.0

    if len(audit_df) != len(
        TOPOLOGIES
    ):
        raise RuntimeError(
            "Incomplete Y-OFF reproduction audit."
        )

    if (
        audit_df[
            "MC_ch_error"
        ].abs().max()
        >
        mc_tol
    ):
        raise RuntimeError(
            "Y-OFF MC reproduction audit failed. "
            "Do not continue to fair parameter calibration."
        )

    if (
        audit_df[
            "Tw_error"
        ].abs().max()
        >
        tw_tol
    ):
        raise RuntimeError(
            "Y-OFF washout reproduction audit failed. "
            "Do not continue to fair parameter calibration."
        )

    print()
    print("Y-OFF reproduction audit: PASS")
    print()

    # =========================================================================
    # FINAL TABLES
    # =========================================================================

    print("=" * 132)
    print("FULL H5/H6 h_y SUMMARY")
    print("=" * 132)

    show_cols = [
        "topology",
        "hy",
        "MC_ch_mean",
        "MC_ch_std",
        "Tw_mean",
        "valid_washout_seeds",
        "best_readout",
        "best_lambda",
        "best_cv_rmse",
        "best_cv_rmse_std",
        "best_validation_rmse",
    ]

    print(
        summary_df[
            show_cols
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 132)
    print("PARETO h_y CANDIDATES — NO WEIGHTED SCORE")
    print("=" * 132)

    print(
        pareto_df[
            show_cols
        ]
        .sort_values(
            [
                "topology",
                "best_cv_rmse",
            ]
        )
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # MANIFEST
    # =========================================================================

    manifest = {
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "step":
            "12.3B",

        "purpose":
            (
                "Symmetric H5/H6 memory-only Y-field refresh before "
                "fair r=1,2,3 parameter re-optimization."
            ),

        "frozen": {
            "alpha":
                ALPHA,

            "hx":
                HX,

            "dt":
                DT,

            "trotter_r":
                TROTTER_R,

            "J_source":
                str(
                    J_WINNER_FILE
                ),

            "initial_memory_state":
                "|00><00|",
        },

        "hy_grid":
            HY_GRID,

        "probe_seeds":
            PROBE_SEEDS,

        "memory": {
            "readout":
                "XYZ_all",

            "K_max":
                K_MAX,

            "ridge_alpha":
                MC_RIDGE_ALPHA,

            "burnin":
                "max(100,Tw(0.01)+20)",

            "washout_validity":
                f"finite Tw(0.01) <= {MAX_ACCEPTABLE_WASHOUT}",
        },

        "forecast": {
            "readouts":
                list(
                    READOUT_FEATURES.keys()
                ),

            "ridge_grid":
                [
                    float(x)
                    for x
                    in RIDGE_GRID
                ],

            "cv_folds":
                N_CV_SPLITS,

            "selection":
                "2022-2024 chronological CV only",

            "validation_2025":
                "diagnostic only",

            "test_2026_used":
                False,
        },

        "pareto": {
            "weighted_score":
                False,

            "objectives": {
                "best_cv_rmse":
                    "min",

                "MC_ch_mean":
                    "max",

                "Tw_mean":
                    "min",

                "best_n_settings":
                    "min",
            },

            "note":
                (
                    "Pareto output is diagnostic only. No Y-ON seed is "
                    "frozen until these results are interpreted."
                ),
        },

        "hardware": {
            "qpu_jobs":
                0,

            "shots":
                0,

            "noise_model":
                False,

            "backend_calibration_used":
                False,
        },

        "topologies": {
            topology: {
                "description":
                    base.CANDIDATES[
                        topology
                    ][
                        "description"
                    ],

                "edges":
                    [
                        list(edge)
                        for edge
                        in base.CANDIDATES[
                            topology
                        ][
                            "edges"
                        ]
                    ],

                "reference_J": {
                    f"J{edge[0]}{edge[1]}":
                        float(
                            REFERENCE_J[
                                topology
                            ][
                                edge
                            ]
                        )

                    for edge
                    in base.CANDIDATES[
                        topology
                    ][
                        "edges"
                    ]
                },
            }

            for topology
            in TOPOLOGIES
        },
    }

    with open(
        RESULTS
        /
        "12_03b_yfield_manifest.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            manifest,
            fp,
            indent=2,
            default=str,
        )

    print()
    print("=" * 132)
    print("FILES SAVED")
    print("=" * 132)

    for name in [
        "12_03b_yfield_memory_by_seed.csv",
        "12_03b_yfield_forecast_all.csv",
        "12_03b_yfield_ridge_grid.csv",
        "12_03b_yfield_summary.csv",
        "12_03b_yfield_pareto.csv",
        "12_03b_yoff_reproduction_audit.csv",
        "12_03b_yfield_manifest.json",
    ]:
        print(
            f"results/{name}"
        )

    print()
    print("Step 12.3B complete.")
    print(
        "Interpret the H5/H6 h_y landscape and Pareto candidates "
        "before building the fair (J,hx,hy,lambda,r) search."
    )


if __name__ == "__main__":
    main()
