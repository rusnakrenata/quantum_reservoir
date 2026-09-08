"""
FINAL HETEROGENEOUS-J MEMORY OPTIMIZATION
=========================================

Goal
----
Perform one final ideal-simulator refinement of the memory-optimal QRC by
allowing the four important T2 couplings to differ:

    J24, J34, J35, J45

instead of forcing

    J24 = J34 = J35 = J_cross.

This is a MEMORY optimization, not a forecasting optimization.

The search is deliberately two-stage:
    A. broad continuous random search
    B. local refinement around the best candidates

This avoids an unnecessary full grid such as 6^4 * 3 = 3888 dynamics.

Frozen
------
    topology     = T2_dual_boundary
    qubits       = 4 injection + 2 memory
    seed         = 42
    hx           = 0.5
    Trotter r    = 2
    alpha        = 0.75
    observable   = XYZ_all
    backbone     = original seed-42 J01, J12, J23

Evolution times tested
----------------------
    dt in {1.0, 1.2, 1.5}

Heterogeneous couplings
-----------------------
    J24, J34, J35, J45 sampled continuously from approximately
    [-1.6, -0.25] U [0.25, 1.6].

Zero-near couplings are skipped during the optimization because the previous
control experiment already established that disconnected/near-disconnected
memory gives weak accessible MC.

Memory metric
-------------
For four independently randomized F4 input channels:

    MC_{j,k} = Corr^2[u_j(t-k), uhat_{j,k}(t)]

Main ranking value:

    corrected MC per channel
      = (1/4) sum_j sum_k max(MC_{j,k} - null_floor, 0)

The reported four-channel total is

    MC_total_corrected = 4 * MC_per_channel_corrected.

Washout gate
------------
A candidate must forget arbitrary initialization:

    T_w(0.01) finite and <= 300

The optimization therefore does NOT reward non-fading persistence.

Reference to beat
-----------------
Previous homogeneous T2 result:

    dt = 1.5
    J24=J34=J35=-1
    J45=-1
    XYZ_all
    corrected MC/ch ~= 0.826055
    corrected four-channel total ~= 3.30422
    T_w(0.01) ~= 22

Stopping interpretation
-----------------------
After the search:

    gain >= 10% over 0.826055:
        meaningful heterogeneous-J improvement

    gain 3%..10%:
        modest improvement; likely freeze after confirmation

    gain < 3%:
        heterogeneous J adds little; freeze the simpler homogeneous design

Forecast diagnostics
--------------------
Forecast RMSE is NOT used to select J.

For the best finalists only, the script calculates:
    - 2022-2024 5-fold chronological CV RMSE for C_{t+1}
    - 2022-2024 5-fold chronological CV RMSE for Delta C_{t+1}

2025 is not used for forecasting selection.
2026 remains untouched.

Dependency
----------
Keep beside this script:
    07_02b_memory_pair_isolation.py

Outputs
-------
results/07_memory_heterogeneous_J_stageA.csv
results/07_memory_heterogeneous_J_stageB.csv
results/07_memory_heterogeneous_J_all_ranked.csv
results/07_memory_heterogeneous_J_MC_by_delay.csv
results/07_memory_heterogeneous_J_finalists_RMSE.csv
results/07_memory_heterogeneous_J_summary.txt
"""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


# =============================================================================
# DEPENDENCY
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
BASE = HERE / "07_02b_memory_pair_isolation.py"

if not BASE.exists():
    raise FileNotFoundError(
        f"Missing {BASE.name}. Keep it beside this script."
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


qrc = load_module(BASE, "qrc_heterogeneous_J")


# =============================================================================
# FROZEN SETTINGS
# =============================================================================

SEED = 42
ALPHA = 0.75
HX = 0.5
TROTTER_R = 2

DT_GRID = [1.0, 1.2, 1.5]

PROBE_SEED = 79001
SEARCH_SEED = 88017

K_MAX = 20
MIN_BURNIN = 100
POST_WASHOUT_MARGIN = 20
MAX_ACCEPTABLE_WASHOUT = 300

MC_RIDGE_ALPHA = 1e-6
FORECAST_RIDGE_ALPHA = 0.01
CV_FOLDS = 5
VAR_TOL = 1e-12

# Search size:
GLOBAL_RANDOM_PER_DT = 20       # 60 random candidates total
TOP_PARENTS = 6
LOCAL_CHILDREN_PER_PARENT = 5   # 30 local candidates
LOCAL_SIGMA = 0.18

J_ABS_MIN = 0.25
J_ABS_MAX = 1.60

# Previous memory reference.
REFERENCE_MC_PER_CHANNEL = 0.826055

# Number of finalists receiving forecasting CV.
N_FORECAST_FINALISTS = 8

qrc.ALPHA = ALPHA
qrc.HX = HX
qrc.TROTTER_R = TROTTER_R
qrc.SEEDS = [SEED]


# =============================================================================
# T2 TOPOLOGY
# =============================================================================

BACKBONE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
]

EDGE_24 = (2, 4)
EDGE_34 = (3, 4)
EDGE_35 = (3, 5)
EDGE_45 = (4, 5)

TUNED_EDGES = [
    EDGE_24,
    EDGE_34,
    EDGE_35,
    EDGE_45,
]

ALL_EDGES = BACKBONE_EDGES + TUNED_EDGES


def original_backbone_couplings():
    original = qrc.seed_couplings(SEED)
    return {
        edge: float(original[edge])
        for edge in BACKBONE_EDGES
    }


J_BACKBONE = original_backbone_couplings()

ZZ_OPS = {
    edge: qrc.pauli_product_full(
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
    "X0", "X1", "X2", "X3", "X4", "X5",
    "Y0", "Y1", "Y2", "Y3", "Y4", "Y5",
    "Z0", "Z1", "Z2", "Z3", "Z4", "Z5",
]


# =============================================================================
# INITIAL STATES
# =============================================================================

def ket_density(v):
    v = np.asarray(v, dtype=complex)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


e00 = np.array([1, 0, 0, 0], dtype=complex)
e11 = np.array([0, 0, 0, 1], dtype=complex)
epp = np.array([1, 1, 1, 1], dtype=complex) / 2.0
ebell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2.0)

INITIAL_STATES = {
    "00": ket_density(e00),
    "11": ket_density(e11),
    "++": ket_density(epp),
    "BellPhi+": ket_density(ebell),
    "I/4": np.eye(4, dtype=complex) / 4.0,
}

RESERVOIR_INITIAL_STATE = INITIAL_STATES["00"]


# =============================================================================
# LINEAR ALGEBRA / WASHOUT
# =============================================================================

def trace_distance(rho, sigma):
    delta = np.asarray(rho - sigma, dtype=complex)
    delta = 0.5 * (delta + delta.conj().T)
    vals = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(vals)))


def max_pairwise_trace_distance(states):
    return max(
        trace_distance(states[a], states[b])
        for a, b in combinations(states.keys(), 2)
    )


def stable_threshold_crossing(values, threshold):
    arr = np.asarray(values, dtype=float)
    suffix_max = np.maximum.accumulate(arr[::-1])[::-1]
    idx = np.flatnonzero(suffix_max <= threshold)
    return None if len(idx) == 0 else int(idx[0])


def fit_kappa(D_curve):
    D = np.asarray(D_curve, dtype=float)

    idx = np.flatnonzero(
        (np.arange(len(D)) >= 5)
        & (D > 1e-10)
        & np.isfinite(D)
    )

    if len(idx) < 20:
        return np.nan, np.nan, np.nan

    t = idx.astype(float)
    y = np.log(D[idx])

    slope, intercept = np.polyfit(t, y, 1)
    pred = intercept + slope * t

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    kappa = float(np.exp(slope))

    tau = (
        float(-1.0 / np.log(kappa))
        if 0.0 < kappa < 1.0
        else np.inf
    )

    return kappa, tau, r2


# =============================================================================
# UNITARY
# =============================================================================

def build_J(J24, J34, J35, J45):
    J = dict(J_BACKBONE)
    J[EDGE_24] = float(J24)
    J[EDGE_34] = float(J34)
    J[EDGE_35] = float(J35)
    J[EDGE_45] = float(J45)
    return J


def build_unitary(dt, J24, J34, J35, J45):
    J = build_J(J24, J34, J35, J45)

    U = qrc.I64.copy()

    for _ in range(TROTTER_R):

        for edge in ALL_EDGES:
            theta = (
                2.0
                * J[edge]
                * dt
                / TROTTER_R
            )

            ZZ = ZZ_OPS[edge]

            G = (
                np.cos(theta / 2.0) * qrc.I64
                - 1j
                * np.sin(theta / 2.0)
                * ZZ
            )

            U = G @ U

        theta_x = (
            2.0 * HX * dt / TROTTER_R
        )

        for qubit in range(qrc.N_QUBITS):
            Xq = qrc.FULL_SINGLE_OPS[f"X{qubit}"]

            G = (
                np.cos(theta_x / 2.0) * qrc.I64
                - 1j
                * np.sin(theta_x / 2.0)
                * Xq
            )

            U = G @ U

    err = np.linalg.norm(
        U.conj().T @ U - qrc.I64,
        ord="fro",
    )

    return U, J, float(err)


# =============================================================================
# RANDOMIZED F4 PROBE
# =============================================================================

def build_probe(real_angles):
    rng = np.random.default_rng(PROBE_SEED)

    real_angles = np.asarray(real_angles, dtype=float)
    probe = np.empty_like(real_angles)

    for j in range(real_angles.shape[1]):
        probe[:, j] = real_angles[
            rng.permutation(len(real_angles)),
            j,
        ]

    return probe


def probe_audit(probe):
    max_abs = 0.0

    for j in range(probe.shape[1]):
        for lag in range(1, 21):
            r = np.corrcoef(
                probe[lag:, j],
                probe[:-lag, j],
            )[0, 1]

            if np.isfinite(r):
                max_abs = max(
                    max_abs,
                    abs(float(r)),
                )

    return max_abs


# =============================================================================
# FEATURE BANK
# =============================================================================

def add_Y_features(row, rho_i, rho_m):
    for qubit in range(qrc.N_INJECTION):
        name = f"Y{qubit}"
        row[name] = qrc.expectation(
            rho_i,
            qrc.INJECTION_SINGLE_OPS[name],
        )

    for qubit in qrc.MEMORY_QUBITS:
        name = f"Y{qubit}"
        row[name] = qrc.expectation(
            rho_m,
            qrc.MEMORY_SINGLE_OPS[name],
        )

    return row


def build_feature_bank(A_list, initial_state):
    rho = initial_state.copy()
    rows = []

    for A in A_list:
        rho_before = rho

        rho_i, rho_m = qrc.final_reduced_states(
            A,
            rho_before,
        )

        row = qrc.extract_feature_row(
            A,
            rho_before,
            rho_i,
            rho_m,
        )

        row = add_Y_features(
            row,
            rho_i,
            rho_m,
        )

        rows.append(row)
        rho = rho_m

    return pd.DataFrame(rows)


# =============================================================================
# WASHOUT
# =============================================================================

def washout_diagnostic(S_list):
    states = {
        name: rho.copy()
        for name, rho in INITIAL_STATES.items()
    }

    D_curve = []

    for S in S_list:
        states = {
            name: qrc.apply_memory_channel(S, rho)
            for name, rho in states.items()
        }

        D_curve.append(
            max_pairwise_trace_distance(states)
        )

    D_curve = np.asarray(D_curve, dtype=float)

    Tw = stable_threshold_crossing(
        D_curve,
        0.01,
    )

    kappa, tau, r2 = fit_kappa(D_curve)

    return {
        "Tw_0.01": np.nan if Tw is None else Tw,
        "D_final": float(D_curve[-1]),
        "kappa_eff": kappa,
        "tau_mem": tau,
        "kappa_fit_R2": r2,
    }


# =============================================================================
# RIDGE / MC
# =============================================================================

def safe_corr2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        np.std(a) <= VAR_TOL
        or np.std(b) <= VAR_TOL
    ):
        return 0.0

    r = np.corrcoef(a, b)[0, 1]

    if not np.isfinite(r):
        return 0.0

    return float(r * r)


def standardized_ridge_predict(
    Xtr,
    ytr,
    Xva,
    alpha,
):
    Xtr = np.asarray(Xtr, dtype=float)
    Xva = np.asarray(Xva, dtype=float)
    ytr = np.asarray(ytr, dtype=float)

    mu = np.mean(Xtr, axis=0)
    sd = np.std(Xtr, axis=0, ddof=0)

    keep = sd > VAR_TOL

    if not np.any(keep):
        return np.full(
            len(Xva),
            float(np.mean(ytr)),
        )

    Xtr_z = (
        Xtr[:, keep] - mu[keep]
    ) / sd[keep]

    Xva_z = (
        Xva[:, keep] - mu[keep]
    ) / sd[keep]

    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
    )

    model.fit(Xtr_z, ytr)
    return model.predict(Xva_z)


def evaluate_memory(
    feature_bank,
    probe,
    n_train,
    n_val,
    burnin,
):
    X = feature_bank[
        XYZ_ALL
    ].to_numpy(dtype=float)

    null_floor = 1.0 / (n_val - 1.0)

    rows = []

    for channel in range(probe.shape[1]):
        for delay in range(1, K_MAX + 1):

            train_start = max(
                burnin,
                delay,
            )

            tr = np.arange(
                train_start,
                n_train,
            )

            va = np.arange(
                n_train,
                n_train + n_val,
            )

            ytr = probe[
                tr - delay,
                channel,
            ]

            yva = probe[
                va - delay,
                channel,
            ]

            pred = standardized_ridge_predict(
                X[tr],
                ytr,
                X[va],
                MC_RIDGE_ALPHA,
            )

            raw = safe_corr2(yva, pred)
            corrected = max(
                raw - null_floor,
                0.0,
            )

            rows.append({
                "channel": channel,
                "delay": delay,
                "MC_raw": raw,
                "MC_corrected": corrected,
            })

    detail = pd.DataFrame(rows)

    total_raw = float(
        detail["MC_raw"].sum()
    )

    total_corrected = float(
        detail["MC_corrected"].sum()
    )

    nch = probe.shape[1]

    delay_means = (
        detail.groupby(
            "delay",
            as_index=False,
        )
        .agg(
            MC_raw_mean=("MC_raw", "mean"),
            MC_corrected_mean=(
                "MC_corrected",
                "mean",
            ),
        )
    )

    def at_delay(k, column="MC_raw_mean"):
        row = delay_means[
            delay_means["delay"] == k
        ]

        return (
            float(row.iloc[0][column])
            if len(row)
            else np.nan
        )

    return {
        "MC_total_raw": total_raw,
        "MC_total_corrected": total_corrected,
        "MC_per_channel_raw":
            total_raw / nch,
        "MC_per_channel_corrected":
            total_corrected / nch,

        "MC_delay1_mean":
            at_delay(1),
        "MC_delay2_mean":
            at_delay(2),
        "MC_delay5_mean":
            at_delay(5),
        "MC_delay10_mean":
            at_delay(10),
        "MC_delay20_mean":
            at_delay(20),

        "detail": detail,
    }


# =============================================================================
# SEARCH CANDIDATES
# =============================================================================

def sample_signed_J(rng):
    magnitude = rng.uniform(
        J_ABS_MIN,
        J_ABS_MAX,
    )

    sign = rng.choice([-1.0, +1.0])

    return float(sign * magnitude)


def make_global_candidates():
    rng = np.random.default_rng(
        SEARCH_SEED
    )

    rows = []

    # Deterministic anchors: known strong homogeneous sign patterns.
    anchors = [
        (-1.0, -1.0, -1.0, -1.0),
        (-1.0, -1.0, -1.0, +1.0),
        (+1.0, +1.0, +1.0, -1.0),
        (+1.0, +1.0, +1.0, +1.0),

        # Some intentionally heterogeneous anchors.
        (-1.0, -0.5, -1.5, -1.0),
        (-1.5, -1.0, -0.5, -1.0),
        (+1.0, -1.0, +1.0, -1.0),
        (-1.0, +1.0, -1.0, +1.0),
    ]

    for dt in DT_GRID:
        for J24, J34, J35, J45 in anchors:
            rows.append({
                "stage": "A_anchor",
                "parent_id": "",
                "dt": dt,
                "J24": J24,
                "J34": J34,
                "J35": J35,
                "J45": J45,
            })

        for _ in range(
            GLOBAL_RANDOM_PER_DT
        ):
            rows.append({
                "stage": "A_random",
                "parent_id": "",
                "dt": dt,
                "J24": sample_signed_J(rng),
                "J34": sample_signed_J(rng),
                "J35": sample_signed_J(rng),
                "J45": sample_signed_J(rng),
            })

    return pd.DataFrame(rows)


def clip_away_from_zero(value):
    value = float(
        np.clip(
            value,
            -J_ABS_MAX,
            J_ABS_MAX,
        )
    )

    if abs(value) < J_ABS_MIN:
        sign = (
            1.0
            if value >= 0
            else -1.0
        )

        value = sign * J_ABS_MIN

    return value


def make_local_candidates(top_stageA):
    rng = np.random.default_rng(
        SEARCH_SEED + 1
    )

    rows = []

    for parent_rank, (_, parent) in enumerate(
        top_stageA.head(TOP_PARENTS).iterrows(),
        start=1,
    ):
        for child in range(
            LOCAL_CHILDREN_PER_PARENT
        ):
            rows.append({
                "stage": "B_local",
                "parent_id":
                    f"A_rank_{parent_rank}",
                "dt": float(parent["dt"]),
                "J24": clip_away_from_zero(
                    parent["J24"]
                    + rng.normal(
                        0.0,
                        LOCAL_SIGMA,
                    )
                ),
                "J34": clip_away_from_zero(
                    parent["J34"]
                    + rng.normal(
                        0.0,
                        LOCAL_SIGMA,
                    )
                ),
                "J35": clip_away_from_zero(
                    parent["J35"]
                    + rng.normal(
                        0.0,
                        LOCAL_SIGMA,
                    )
                ),
                "J45": clip_away_from_zero(
                    parent["J45"]
                    + rng.normal(
                        0.0,
                        LOCAL_SIGMA,
                    )
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# SINGLE-CANDIDATE MEMORY EVALUATION
# =============================================================================

def evaluate_candidate(
    candidate,
    probe,
    n_train,
    n_val,
):
    dt = float(candidate["dt"])
    J24 = float(candidate["J24"])
    J34 = float(candidate["J34"])
    J35 = float(candidate["J35"])
    J45 = float(candidate["J45"])

    U, J, unitary_err = build_unitary(
        dt,
        J24,
        J34,
        J35,
        J45,
    )

    A_probe, S_probe = (
        qrc.build_input_channels(
            U,
            probe,
        )
    )

    wash = washout_diagnostic(
        S_probe
    )

    if np.isfinite(wash["Tw_0.01"]):
        burnin = int(
            max(
                MIN_BURNIN,
                int(wash["Tw_0.01"])
                + POST_WASHOUT_MARGIN,
            )
        )
    else:
        burnin = n_train - 100

    burnin = min(
        burnin,
        n_train - 100,
    )

    feature_bank = build_feature_bank(
        A_probe,
        RESERVOIR_INITIAL_STATE,
    )

    mc = evaluate_memory(
        feature_bank,
        probe,
        n_train,
        n_val,
        burnin,
    )

    washout_valid = int(
        np.isfinite(
            wash["Tw_0.01"]
        )
        and
        wash["Tw_0.01"]
        <= MAX_ACCEPTABLE_WASHOUT
    )

    row = {
        "stage": candidate["stage"],
        "parent_id": candidate["parent_id"],

        "dt": dt,
        "J24": J24,
        "J34": J34,
        "J35": J35,
        "J45": J45,

        "unitarity_error":
            unitary_err,

        "Tw_0.01":
            wash["Tw_0.01"],
        "D_final":
            wash["D_final"],
        "kappa_eff":
            wash["kappa_eff"],
        "tau_mem":
            wash["tau_mem"],
        "kappa_fit_R2":
            wash["kappa_fit_R2"],
        "washout_valid":
            washout_valid,

        "MC_burnin":
            burnin,

        "MC_total_raw":
            mc["MC_total_raw"],
        "MC_total_corrected":
            mc[
                "MC_total_corrected"
            ],
        "MC_per_channel_raw":
            mc[
                "MC_per_channel_raw"
            ],
        "MC_per_channel_corrected":
            mc[
                "MC_per_channel_corrected"
            ],

        "MC_delay1_mean":
            mc["MC_delay1_mean"],
        "MC_delay2_mean":
            mc["MC_delay2_mean"],
        "MC_delay5_mean":
            mc["MC_delay5_mean"],
        "MC_delay10_mean":
            mc["MC_delay10_mean"],
        "MC_delay20_mean":
            mc["MC_delay20_mean"],
    }

    detail = mc["detail"].copy()

    for key, value in {
        "stage": candidate["stage"],
        "parent_id": candidate["parent_id"],
        "dt": dt,
        "J24": J24,
        "J34": J34,
        "J35": J35,
        "J45": J45,
    }.items():
        detail[key] = value

    return row, detail, U


# =============================================================================
# FORECASTING DIAGNOSTICS
# =============================================================================

def rmse(y_true, y_pred):
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


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
        n_splits=CV_FOLDS
    )

    values = []

    for tr, va in splitter.split(X):
        pred = standardized_ridge_predict(
            X[tr],
            y[tr],
            X[va],
            FORECAST_RIDGE_ALPHA,
        )

        values.append(
            rmse(
                y[va],
                pred,
            )
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "std": float(
            np.std(
                values,
                ddof=1,
            )
        ),
        "folds": values,
    }


def resolve_column(
    df,
    candidates,
    label,
):
    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        if candidate.lower() in lower:
            return lower[
                candidate.lower()
            ]

    raise KeyError(
        f"Could not find {label}. "
        f"Tried {candidates}."
    )


# =============================================================================
# RANKING / INTERPRETATION
# =============================================================================

def rank_candidates(df):
    ranked = df.copy()

    # Valid fading-memory reservoirs first, then maximize corrected MC.
    ranked = ranked.sort_values(
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
    ).reset_index(drop=True)

    ranked["rank"] = (
        np.arange(len(ranked)) + 1
    )

    return ranked


def gain_label(best_mc):
    gain = 100.0 * (
        best_mc
        - REFERENCE_MC_PER_CHANNEL
    ) / REFERENCE_MC_PER_CHANNEL

    if gain >= 10.0:
        label = (
            "MEANINGFUL improvement: heterogeneous J materially "
            "improves accessible memory."
        )
    elif gain >= 3.0:
        label = (
            "MODEST improvement: useful, but the simpler homogeneous "
            "configuration remains competitive."
        )
    else:
        label = (
            "SMALL improvement (<3%): freeze the simpler homogeneous "
            "J design unless another criterion strongly favors heterogeneity."
        )

    return gain, label


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 138)
    print("FINAL HETEROGENEOUS-J MEMORY OPTIMIZATION")
    print("=" * 138)
    print()

    print("Frozen:")
    print("  topology = T2_dual_boundary")
    print(f"  seed = {SEED}")
    print(f"  hx = {HX}")
    print(f"  Trotter r = {TROTTER_R}")
    print(f"  alpha = {ALPHA}")
    print("  observable = XYZ_all")
    print()

    print(f"dt = {DT_GRID}")
    print(
        f"global random candidates = "
        f"{GLOBAL_RANDOM_PER_DT} per dt"
    )
    print(
        f"local refinement = top {TOP_PARENTS} "
        f"x {LOCAL_CHILDREN_PER_PARENT} children"
    )
    print(
        f"J search magnitudes = "
        f"[{J_ABS_MIN}, {J_ABS_MAX}] with both signs"
    )
    print()

    print(
        "Reference to beat:"
    )
    print(
        f"  MC corrected/ch = "
        f"{REFERENCE_MC_PER_CHANNEL:.6f}"
    )
    print(
        f"  four-channel total ~= "
        f"{4*REFERENCE_MC_PER_CHANNEL:.6f}"
    )
    print()

    print(
        "Original seed-42 injection backbone:"
    )

    for edge, value in J_BACKBONE.items():
        print(
            f"  J{edge} = {value:+.6f}"
        )

    print()

    # -------------------------------------------------------------------------
    # DATA / PROBE
    # -------------------------------------------------------------------------
    work, train, val, cols = (
        qrc.load_data()
    )

    n_train = len(train)
    n_val = len(val)
    n_steps = n_train + n_val

    work_eval = (
        work.iloc[:n_steps]
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

    max_ac = probe_audit(
        probe
    )

    print(
        f"Probe marginal max error = "
        f"{marginal_error:.3e}"
    )

    print(
        f"Probe max |autocorrelation| "
        f"lags 1..20 = {max_ac:.6f}"
    )

    print()

    # =========================================================================
    # STAGE A
    # =========================================================================
    stageA_candidates = (
        make_global_candidates()
    )

    print("=" * 138)
    print(
        f"STAGE A — BROAD SEARCH "
        f"({len(stageA_candidates)} candidates)"
    )
    print("=" * 138)

    stageA_rows = []
    all_delay_rows = []

    for i, (_, candidate) in enumerate(
        stageA_candidates.iterrows(),
        start=1,
    ):
        print(
            f"[A {i:3d}/{len(stageA_candidates)}] "
            f"dt={candidate['dt']:.1f} | "
            f"J24={candidate['J24']:+.3f} | "
            f"J34={candidate['J34']:+.3f} | "
            f"J35={candidate['J35']:+.3f} | "
            f"J45={candidate['J45']:+.3f}"
        )

        row, detail, _ = (
            evaluate_candidate(
                candidate,
                probe,
                n_train,
                n_val,
            )
        )

        stageA_rows.append(row)
        all_delay_rows.append(detail)

        print(
            f"    MC/ch="
            f"{row['MC_per_channel_corrected']:.4f} | "
            f"MC_total="
            f"{row['MC_total_corrected']:.4f} | "
            f"Tw(.01)="
            f"{row['Tw_0.01']} | "
            f"valid="
            f"{row['washout_valid']}"
        )

    stageA = pd.DataFrame(
        stageA_rows
    )

    stageA_ranked = (
        rank_candidates(
            stageA
        )
    )

    stageA_ranked.to_csv(
        RESULTS /
        "07_memory_heterogeneous_J_stageA.csv",
        index=False,
    )

    print()
    print("-" * 138)
    print("TOP STAGE-A CANDIDATES")
    print("-" * 138)

    show_cols = [
        "rank",
        "stage",
        "dt",
        "J24",
        "J34",
        "J35",
        "J45",
        "MC_per_channel_corrected",
        "MC_total_corrected",
        "MC_delay1_mean",
        "MC_delay5_mean",
        "MC_delay10_mean",
        "Tw_0.01",
        "tau_mem",
        "washout_valid",
    ]

    print(
        stageA_ranked[
            show_cols
        ].head(15).to_string(
            index=False
        )
    )

    # =========================================================================
    # STAGE B
    # =========================================================================
    validA = stageA_ranked[
        stageA_ranked[
            "washout_valid"
        ] == 1
    ]

    parents = (
        validA
        if len(validA)
        >= TOP_PARENTS
        else stageA_ranked
    )

    stageB_candidates = (
        make_local_candidates(
            parents
        )
    )

    print()
    print("=" * 138)
    print(
        f"STAGE B — LOCAL REFINEMENT "
        f"({len(stageB_candidates)} candidates)"
    )
    print("=" * 138)

    stageB_rows = []

    for i, (_, candidate) in enumerate(
        stageB_candidates.iterrows(),
        start=1,
    ):
        print(
            f"[B {i:3d}/{len(stageB_candidates)}] "
            f"{candidate['parent_id']} | "
            f"dt={candidate['dt']:.1f} | "
            f"J24={candidate['J24']:+.3f} | "
            f"J34={candidate['J34']:+.3f} | "
            f"J35={candidate['J35']:+.3f} | "
            f"J45={candidate['J45']:+.3f}"
        )

        row, detail, _ = (
            evaluate_candidate(
                candidate,
                probe,
                n_train,
                n_val,
            )
        )

        stageB_rows.append(row)
        all_delay_rows.append(detail)

        print(
            f"    MC/ch="
            f"{row['MC_per_channel_corrected']:.4f} | "
            f"MC_total="
            f"{row['MC_total_corrected']:.4f} | "
            f"Tw(.01)="
            f"{row['Tw_0.01']} | "
            f"valid="
            f"{row['washout_valid']}"
        )

    stageB = pd.DataFrame(
        stageB_rows
    )

    stageB_ranked = (
        rank_candidates(
            stageB
        )
    )

    stageB_ranked.to_csv(
        RESULTS /
        "07_memory_heterogeneous_J_stageB.csv",
        index=False,
    )

    # =========================================================================
    # COMBINED RANKING
    # =========================================================================
    all_results = pd.concat(
        [
            stageA,
            stageB,
        ],
        ignore_index=True,
    )

    all_ranked = (
        rank_candidates(
            all_results
        )
    )

    all_ranked.to_csv(
        RESULTS /
        "07_memory_heterogeneous_J_all_ranked.csv",
        index=False,
    )

    pd.concat(
        all_delay_rows,
        ignore_index=True,
    ).to_csv(
        RESULTS /
        "07_memory_heterogeneous_J_MC_by_delay.csv",
        index=False,
    )

    print()
    print("=" * 138)
    print("FINAL TOP 20 BY MEMORY")
    print("=" * 138)

    print(
        all_ranked[
            show_cols
        ].head(20).to_string(
            index=False
        )
    )

    # =========================================================================
    # FORECASTING DIAGNOSTICS FOR FINALISTS ONLY
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
        "current raw claim count",
    )

    y_level = work_eval[
        target_col
    ].to_numpy(
        dtype=float
    )

    current_claims = work_eval[
        current_col
    ].to_numpy(
        dtype=float
    )

    y_delta = (
        y_level
        - current_claims
    )

    valid_ranked = all_ranked[
        all_ranked[
            "washout_valid"
        ] == 1
    ]

    finalists = (
        valid_ranked.head(
            N_FORECAST_FINALISTS
        )
        if len(valid_ranked)
        else all_ranked.head(
            N_FORECAST_FINALISTS
        )
    ).copy()

    finalist_rows = []

    print()
    print("=" * 138)
    print(
        f"FORECASTING DIAGNOSTICS FOR TOP "
        f"{len(finalists)} MEMORY FINALISTS"
    )
    print("=" * 138)

    for i, (_, row) in enumerate(
        finalists.iterrows(),
        start=1,
    ):
        dt = float(row["dt"])
        J24 = float(row["J24"])
        J34 = float(row["J34"])
        J35 = float(row["J35"])
        J45 = float(row["J45"])

        U, _, _ = build_unitary(
            dt,
            J24,
            J34,
            J35,
            J45,
        )

        A_real, _ = (
            qrc.build_input_channels(
                U,
                real_angles,
            )
        )

        bank_real = build_feature_bank(
            A_real,
            RESERVOIR_INITIAL_STATE,
        )

        X_real = bank_real[
            XYZ_ALL
        ].to_numpy(
            dtype=float
        )

        level_cv = chronological_cv_rmse(
            X_real,
            y_level,
            n_train,
        )

        delta_cv = chronological_cv_rmse(
            X_real,
            y_delta,
            n_train,
        )

        out = row.to_dict()

        out[
            "level_CV_RMSE_mean"
        ] = level_cv["mean"]

        out[
            "level_CV_RMSE_std"
        ] = level_cv["std"]

        out[
            "delta_CV_RMSE_mean"
        ] = delta_cv["mean"]

        out[
            "delta_CV_RMSE_std"
        ] = delta_cv["std"]

        finalist_rows.append(out)

        print(
            f"[{i}/{len(finalists)}] "
            f"MC/ch="
            f"{row['MC_per_channel_corrected']:.4f} | "
            f"level RMSE="
            f"{level_cv['mean']:.4f} | "
            f"delta RMSE="
            f"{delta_cv['mean']:.4f}"
        )

    finalists_df = pd.DataFrame(
        finalist_rows
    )

    finalists_df.to_csv(
        RESULTS /
        "07_memory_heterogeneous_J_finalists_RMSE.csv",
        index=False,
    )

    # =========================================================================
    # DECISION
    # =========================================================================
    eligible = all_ranked[
        all_ranked[
            "washout_valid"
        ] == 1
    ]

    best = (
        eligible.iloc[0]
        if len(eligible)
        else all_ranked.iloc[0]
    )

    best_mc = float(
        best[
            "MC_per_channel_corrected"
        ]
    )

    gain_pct, decision = gain_label(
        best_mc
    )

    print()
    print("=" * 138)
    print("FINAL HETEROGENEOUS-J DECISION")
    print("=" * 138)

    print(
        best[
            [
                "stage",
                "dt",
                "J24",
                "J34",
                "J35",
                "J45",
                "MC_per_channel_corrected",
                "MC_total_corrected",
                "MC_delay1_mean",
                "MC_delay2_mean",
                "MC_delay5_mean",
                "MC_delay10_mean",
                "MC_delay20_mean",
                "Tw_0.01",
                "tau_mem",
                "washout_valid",
            ]
        ].to_string()
    )

    print()
    print(
        f"Gain over previous 0.826055/ch = "
        f"{gain_pct:+.2f}%"
    )

    print(decision)

    # =========================================================================
    # TEXT SUMMARY
    # =========================================================================
    with open(
        RESULTS /
        "07_memory_heterogeneous_J_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:

        fp.write(
            "FINAL HETEROGENEOUS-J MEMORY OPTIMIZATION\n"
        )
        fp.write(
            "=" * 110 + "\n\n"
        )

        fp.write(
            "Reference MC/ch = "
            f"{REFERENCE_MC_PER_CHANNEL:.6f}\n"
        )

        fp.write(
            "Reference total corrected MC ~= "
            f"{4*REFERENCE_MC_PER_CHANNEL:.6f}\n\n"
        )

        fp.write(
            "Top 20:\n"
        )

        fp.write(
            all_ranked[
                show_cols
            ].head(20).to_string(
                index=False
            )
        )

        fp.write(
            "\n\nBest configuration:\n"
        )

        fp.write(
            best.to_string()
        )

        fp.write(
            "\n\nGain over reference = "
            f"{gain_pct:+.2f}%\n"
        )

        fp.write(
            "Decision: "
            + decision
            + "\n"
        )

        if len(finalists_df):
            fp.write(
                "\nForecast diagnostics for memory finalists:\n"
            )

            fp.write(
                finalists_df[
                    [
                        "dt",
                        "J24",
                        "J34",
                        "J35",
                        "J45",
                        "MC_per_channel_corrected",
                        "MC_total_corrected",
                        "Tw_0.01",
                        "level_CV_RMSE_mean",
                        "delta_CV_RMSE_mean",
                    ]
                ].to_string(
                    index=False
                )
            )

    print()
    print("Saved:")
    print(
        "  results/07_memory_heterogeneous_J_stageA.csv"
    )
    print(
        "  results/07_memory_heterogeneous_J_stageB.csv"
    )
    print(
        "  results/07_memory_heterogeneous_J_all_ranked.csv"
    )
    print(
        "  results/07_memory_heterogeneous_J_MC_by_delay.csv"
    )
    print(
        "  results/07_memory_heterogeneous_J_finalists_RMSE.csv"
    )
    print(
        "  results/07_memory_heterogeneous_J_summary.txt"
    )


if __name__ == "__main__":
    main()
