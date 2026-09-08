"""
FOCUSED QRC MEMORY + FORECASTING SEARCH
=======================================

NEXT QUESTION
-------------
After the coupling/observable search identified T2_dual_boundary as the best
memory topology, optimize the reservoir evolution time dt while simultaneously
checking whether stronger intrinsic memory helps real forecasting.

For every candidate we report:

    1. intrinsic delayed-input memory capacity (MC)
    2. washout / fading-memory diagnostics
    3. 2022-2024 chronological CV RMSE for C_{t+1}
    4. 2022-2024 chronological CV RMSE for Delta C_{t+1}=C_{t+1}-C_t

IMPORTANT:
    - 2025 is NOT used in this script.
    - 2026 test remains untouched.
    - No artificial combined MC/RMSE score is used.
    - We inspect the Pareto trade-off instead.

Frozen
------
    topology = T2_dual_boundary
    seed = 42
    hx = 0.5
    Trotter r = 2
    alpha = 0.75

Screened
--------
    dt in {0.2,0.4,0.6,0.8,1.0,1.2,1.5}

    (J_cross,J45) in
        (-1,-1), (-1,+1), (+1,-1), (+1,+1)

    observable families:
        XZ_injection
        XYZ_memory
        XYZ_all
        XZinj_plus_memory_pairs

The injection-only backbone J01,J12,J23 remains exactly the original
Week-6 seed-42 backbone.

Ridge
-----
Forecast readout lambda is frozen to 0.01, matching the Week-7 exact-feature
QRC readout setting.  Feature standardization is fitted separately inside each
chronological CV fold.

MC methodology
--------------
A distribution-matched randomized F4 input sequence is used:
    - exact empirical marginal distributions preserved
    - each input channel independently permuted
    - chronological autocorrelation removed

Two MC values are reported:

    MC_burnin100:
        directly comparable with the previous focused search.

    MC_postwashout:
        more rigorous; burn-in=max(100,Tw(0.01)+20), capped so enough
        training samples remain.

The post-washout MC is the main quantity for scientific interpretation.

Dependency
----------
Keep this script beside:
    07_02b_memory_pair_isolation.py

Outputs
-------
results/07_memory_dt_mc_rmse_summary.csv
results/07_memory_dt_mc_by_delay.csv
results/07_memory_dt_ranked_by_MC.csv
results/07_memory_dt_ranked_by_level_RMSE.csv
results/07_memory_dt_ranked_by_delta_RMSE.csv
results/07_memory_dt_pareto.csv
results/07_memory_dt_mc_rmse_summary.txt
"""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


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


qrc = load_module(BASE, "qrc_dt_mc_rmse")


# =============================================================================
# FROZEN / SEARCH SETTINGS
# =============================================================================

SEED = 42

ALPHA = 0.75
HX = 0.5
TROTTER_R = 2

DT_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]

J_PAIRS = [
    (-1.0, -1.0),
    (-1.0, +1.0),
    (+1.0, -1.0),
    (+1.0, +1.0),
]

PROBE_SEED = 79001

K_MAX = 20
COMMON_BURNIN = 100
POST_WASHOUT_MARGIN = 20

MC_RIDGE_ALPHA = 1e-6
FORECAST_RIDGE_ALPHA = 0.01

CV_FOLDS = 5
VAR_TOL = 1e-12

qrc.ALPHA = ALPHA
qrc.HX = HX
qrc.TROTTER_R = TROTTER_R
qrc.SEEDS = [SEED]


# =============================================================================
# T2 DUAL-BOUNDARY TOPOLOGY
# =============================================================================

BACKBONE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
]

CROSS_EDGES = [
    (3, 4),
    (2, 4),
    (3, 5),
]

MEMORY_EDGE = (4, 5)

ALL_EDGES = sorted(
    set(BACKBONE_EDGES + CROSS_EDGES + [MEMORY_EDGE])
)


def original_backbone_couplings():
    J_original = qrc.seed_couplings(SEED)
    return {
        edge: float(J_original[edge])
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


def build_J(J_cross, J45):
    J = dict(J_BACKBONE)

    for edge in CROSS_EDGES:
        J[edge] = float(J_cross)

    J[MEMORY_EDGE] = float(J45)

    return J


def build_unitary(dt, J_cross, J45):
    J = build_J(J_cross, J45)

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
                np.cos(theta_x / 2.0)
                * qrc.I64
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
# OBSERVABLE FAMILIES
# =============================================================================

XZ_INJECTION = [
    "X0", "X1", "X2", "X3",
    "Z0", "Z1", "Z2", "Z3",
]

XYZ_MEMORY = [
    "X4", "X5",
    "Y4", "Y5",
    "Z4", "Z5",
]

XYZ_ALL = [
    "X0", "X1", "X2", "X3", "X4", "X5",
    "Y0", "Y1", "Y2", "Y3", "Y4", "Y5",
    "Z0", "Z1", "Z2", "Z3", "Z4", "Z5",
]

XZINJ_MEMORY_PAIRS = XZ_INJECTION + [
    "ZZ_45",
    "ZX_45",
    "XZ_45",
    "YX_45",
]

OBSERVABLE_FAMILIES = {
    "XZ_injection": XZ_INJECTION,
    "XYZ_memory": XYZ_MEMORY,
    "XYZ_all": XYZ_ALL,
    "XZinj_plus_memory_pairs": XZINJ_MEMORY_PAIRS,
}


# =============================================================================
# MEMORY INITIAL STATES
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

# Memory-oriented reservoir initialization.
RESERVOIR_INITIAL_STATE = INITIAL_STATES["00"]


# =============================================================================
# UTILITY FUNCTIONS
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


def rmse(y_true, y_pred):
    return float(
        np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )


# =============================================================================
# PROBE
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
    max_abs_ac = 0.0

    for j in range(probe.shape[1]):
        for lag in range(1, 21):
            r = np.corrcoef(
                probe[lag:, j],
                probe[:-lag, j],
            )[0, 1]

            if np.isfinite(r):
                max_abs_ac = max(
                    max_abs_ac,
                    abs(float(r)),
                )

    return max_abs_ac


# =============================================================================
# FEATURES
# =============================================================================

def add_Y_features(row, rho_i, rho_m):
    for q in range(qrc.N_INJECTION):
        name = f"Y{q}"
        row[name] = qrc.expectation(
            rho_i,
            qrc.INJECTION_SINGLE_OPS[name],
        )

    for q in qrc.MEMORY_QUBITS:
        name = f"Y{q}"
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

    Tw_01 = stable_threshold_crossing(D_curve, 0.1)
    Tw_005 = stable_threshold_crossing(D_curve, 0.05)
    Tw_001 = stable_threshold_crossing(D_curve, 0.01)

    kappa, tau, r2 = fit_kappa(D_curve)

    return {
        "Tw_0.1": np.nan if Tw_01 is None else Tw_01,
        "Tw_0.05": np.nan if Tw_005 is None else Tw_005,
        "Tw_0.01": np.nan if Tw_001 is None else Tw_001,
        "D_final": float(D_curve[-1]),
        "kappa_eff": kappa,
        "tau_mem": tau,
        "kappa_fit_R2": r2,
    }


# =============================================================================
# RIDGE
# =============================================================================

def standardize_fold(Xtr, Xva):
    Xtr = np.asarray(Xtr, dtype=float)
    Xva = np.asarray(Xva, dtype=float)

    mu = np.mean(Xtr, axis=0)
    sd = np.std(Xtr, axis=0, ddof=0)

    keep = sd > VAR_TOL

    if not np.any(keep):
        return None, None

    return (
        (Xtr[:, keep] - mu[keep]) / sd[keep],
        (Xva[:, keep] - mu[keep]) / sd[keep],
    )


def ridge_predict(
    Xtr,
    ytr,
    Xva,
    ridge_alpha,
):
    pair = standardize_fold(Xtr, Xva)

    if pair[0] is None:
        return np.full(
            len(Xva),
            float(np.mean(ytr)),
        )

    Xtr_z, Xva_z = pair

    model = Ridge(
        alpha=ridge_alpha,
        fit_intercept=True,
    )

    model.fit(
        Xtr_z,
        np.asarray(ytr, dtype=float),
    )

    return model.predict(Xva_z)


# =============================================================================
# MEMORY CAPACITY
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


def evaluate_memory(
    feature_bank,
    probe,
    feature_names,
    n_train,
    n_val,
    burnin,
):
    X = feature_bank[
        feature_names
    ].to_numpy(dtype=float)

    null_floor = 1.0 / (n_val - 1.0)

    rows = []

    for j in range(probe.shape[1]):
        for k in range(1, K_MAX + 1):

            start = max(burnin, k)

            tr = np.arange(start, n_train)
            va = np.arange(
                n_train,
                n_train + n_val,
            )

            ytr = probe[tr - k, j]
            yva = probe[va - k, j]

            pred = ridge_predict(
                X[tr],
                ytr,
                X[va],
                MC_RIDGE_ALPHA,
            )

            mc_raw = safe_corr2(yva, pred)
            mc_corrected = max(
                mc_raw - null_floor,
                0.0,
            )

            rows.append({
                "channel": j,
                "delay": k,
                "MC_raw": mc_raw,
                "MC_corrected": mc_corrected,
            })

    detail = pd.DataFrame(rows)

    total_raw = float(detail["MC_raw"].sum())
    total_corr = float(
        detail["MC_corrected"].sum()
    )

    nch = probe.shape[1]

    return {
        "MC_total_raw": total_raw,
        "MC_total_corrected": total_corr,
        "MC_per_channel_raw": total_raw / nch,
        "MC_per_channel_corrected": total_corr / nch,
        "MC_delay1_mean": float(
            detail.loc[
                detail["delay"] == 1,
                "MC_raw",
            ].mean()
        ),
        "MC_delay5_mean": float(
            detail.loc[
                detail["delay"] == 5,
                "MC_raw",
            ].mean()
        ),
        "MC_delay10_mean": float(
            detail.loc[
                detail["delay"] == 10,
                "MC_raw",
            ].mean()
        ),
        "MC_delay20_mean": float(
            detail.loc[
                detail["delay"] == 20,
                "MC_raw",
            ].mean()
        ),
        "detail": detail,
    }


# =============================================================================
# FORECASTING CV
# =============================================================================

def chronological_cv_rmse(
    X,
    y,
    n_train,
):
    X = np.asarray(X, dtype=float)[:n_train]
    y = np.asarray(y, dtype=float)[:n_train]

    splitter = TimeSeriesSplit(
        n_splits=CV_FOLDS
    )

    fold_rmse = []

    for tr_idx, va_idx in splitter.split(X):
        pred = ridge_predict(
            X[tr_idx],
            y[tr_idx],
            X[va_idx],
            FORECAST_RIDGE_ALPHA,
        )

        fold_rmse.append(
            rmse(y[va_idx], pred)
        )

    fold_rmse = np.asarray(
        fold_rmse,
        dtype=float,
    )

    return {
        "mean": float(
            np.mean(fold_rmse)
        ),
        "std": float(
            np.std(
                fold_rmse,
                ddof=1,
            )
        ),
        "folds": fold_rmse,
    }


# =============================================================================
# COLUMN RESOLUTION
# =============================================================================

def resolve_column(df, candidates, label):
    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for c in candidates:
        if c in df.columns:
            return c

        if c.lower() in lower_map:
            return lower_map[c.lower()]

    raise KeyError(
        f"Could not find {label}. Tried {candidates}. "
        f"Available columns include: {list(df.columns)}"
    )


# =============================================================================
# PARETO
# =============================================================================

def pareto_flags(df, mc_col, rmse_col):
    """
    Maximize MC and minimize RMSE.
    """
    mc = df[mc_col].to_numpy(dtype=float)
    er = df[rmse_col].to_numpy(dtype=float)

    flags = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        dominated = (
            (mc >= mc[i])
            & (er <= er[i])
            & (
                (mc > mc[i])
                | (er < er[i])
            )
        )

        if np.any(dominated):
            flags[i] = False

    return flags


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 136)
    print("QRC dt SEARCH: MEMORY CAPACITY + WASHOUT + LEVEL RMSE + DELTA RMSE")
    print("=" * 136)
    print()

    print("Frozen:")
    print("  topology = T2_dual_boundary")
    print(f"  seed = {SEED}")
    print(f"  hx = {HX}")
    print(f"  Trotter r = {TROTTER_R}")
    print(f"  alpha = {ALPHA}")
    print(f"  forecast Ridge lambda = {FORECAST_RIDGE_ALPHA}")
    print()

    print(f"dt grid = {DT_GRID}")
    print(f"J pairs = {J_PAIRS}")
    print(
        "observable families = "
        f"{list(OBSERVABLE_FAMILIES)}"
    )
    print()
    print("2025 is not used.")
    print("2026 test remains untouched.")
    print()

    print("Original seed-42 injection backbone:")
    for edge, val in J_BACKBONE.items():
        print(f"  J{edge} = {val:+.6f}")
    print()

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------
    work, train, val, cols = qrc.load_data()

    n_train = len(train)
    n_val = len(val)
    n_steps = n_train + n_val

    work_eval = work.iloc[:n_steps].copy()

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

    y_level = (
        work_eval[target_col]
        .to_numpy(dtype=float)
    )

    current_claims = (
        work_eval[current_col]
        .to_numpy(dtype=float)
    )

    y_delta = (
        y_level - current_claims
    )

    real_angles = np.asarray(
        qrc.make_input_angles(
            work_eval,
            cols,
        ),
        dtype=float,
    )

    if len(real_angles) != n_steps:
        real_angles = real_angles[:n_steps]

    probe = build_probe(real_angles)

    marginal_error = float(
        np.max(
            np.abs(
                np.sort(probe, axis=0)
                - np.sort(real_angles, axis=0)
            )
        )
    )

    max_ac = probe_audit(probe)

    print(
        f"Probe marginal max error = "
        f"{marginal_error:.3e}"
    )

    print(
        f"Probe max |autocorrelation| "
        f"lags 1..20 = {max_ac:.6f}"
    )

    print(
        f"Train level target: mean="
        f"{np.mean(y_level[:n_train]):.6f}, "
        f"std={np.std(y_level[:n_train], ddof=0):.6f}"
    )

    print(
        f"Train delta target: mean="
        f"{np.mean(y_delta[:n_train]):.6f}, "
        f"std={np.std(y_delta[:n_train], ddof=0):.6f}"
    )
    print()

    # -------------------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------------------
    summary_rows = []
    delay_rows = []

    n_dynamics = (
        len(DT_GRID)
        * len(J_PAIRS)
    )

    counter = 0

    for dt in DT_GRID:

        for J_cross, J45 in J_PAIRS:

            counter += 1

            print(
                f"[{counter:2d}/{n_dynamics}] "
                f"dt={dt:.2f} | "
                f"Jcross={J_cross:+.1f} | "
                f"J45={J45:+.1f}"
            )

            U, J, unitary_err = (
                build_unitary(
                    dt,
                    J_cross,
                    J45,
                )
            )

            # Intrinsic randomized probe.
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

            bank_probe = (
                build_feature_bank(
                    A_probe,
                    RESERVOIR_INITIAL_STATE,
                )
            )

            # Real chronological task trajectory.
            A_real, _ = (
                qrc.build_input_channels(
                    U,
                    real_angles,
                )
            )

            bank_real = (
                build_feature_bank(
                    A_real,
                    RESERVOIR_INITIAL_STATE,
                )
            )

            if np.isfinite(wash["Tw_0.01"]):
                postwash_burnin = int(
                    max(
                        COMMON_BURNIN,
                        int(wash["Tw_0.01"])
                        + POST_WASHOUT_MARGIN,
                    )
                )
            else:
                postwash_burnin = (
                    n_train - 100
                )

            postwash_burnin = min(
                postwash_burnin,
                n_train - 100,
            )

            for family_name, feature_names in OBSERVABLE_FAMILIES.items():

                # Directly comparable to previous search.
                mc100 = evaluate_memory(
                    bank_probe,
                    probe,
                    feature_names,
                    n_train,
                    n_val,
                    COMMON_BURNIN,
                )

                # Main rigorous memory quantity.
                mcpost = evaluate_memory(
                    bank_probe,
                    probe,
                    feature_names,
                    n_train,
                    n_val,
                    postwash_burnin,
                )

                X_real = (
                    bank_real[
                        feature_names
                    ].to_numpy(dtype=float)
                )

                level_cv = (
                    chronological_cv_rmse(
                        X_real,
                        y_level,
                        n_train,
                    )
                )

                delta_cv = (
                    chronological_cv_rmse(
                        X_real,
                        y_delta,
                        n_train,
                    )
                )

                row = {
                    "topology":
                        "T2_dual_boundary",
                    "dt": dt,
                    "J_cross": J_cross,
                    "J45": J45,
                    "family": family_name,
                    "n_features":
                        len(feature_names),

                    "unitarity_error":
                        unitary_err,

                    "Tw_0.1":
                        wash["Tw_0.1"],
                    "Tw_0.05":
                        wash["Tw_0.05"],
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

                    "MC_burnin100":
                        COMMON_BURNIN,
                    "MC_per_channel_corrected_burnin100":
                        mc100[
                            "MC_per_channel_corrected"
                        ],

                    "MC_postwashout_burnin":
                        postwash_burnin,
                    "MC_per_channel_raw_postwashout":
                        mcpost[
                            "MC_per_channel_raw"
                        ],
                    "MC_per_channel_corrected_postwashout":
                        mcpost[
                            "MC_per_channel_corrected"
                        ],

                    "MC_delay1_mean":
                        mcpost["MC_delay1_mean"],
                    "MC_delay5_mean":
                        mcpost["MC_delay5_mean"],
                    "MC_delay10_mean":
                        mcpost["MC_delay10_mean"],
                    "MC_delay20_mean":
                        mcpost["MC_delay20_mean"],

                    "level_CV_RMSE_mean":
                        level_cv["mean"],
                    "level_CV_RMSE_std":
                        level_cv["std"],

                    "delta_CV_RMSE_mean":
                        delta_cv["mean"],
                    "delta_CV_RMSE_std":
                        delta_cv["std"],
                }

                for i, value in enumerate(
                    level_cv["folds"],
                    start=1,
                ):
                    row[
                        f"level_CV_fold{i}_RMSE"
                    ] = value

                for i, value in enumerate(
                    delta_cv["folds"],
                    start=1,
                ):
                    row[
                        f"delta_CV_fold{i}_RMSE"
                    ] = value

                summary_rows.append(row)

                d = mcpost["detail"].copy()
                d["dt"] = dt
                d["J_cross"] = J_cross
                d["J45"] = J45
                d["family"] = family_name
                d["burnin"] = postwash_burnin
                delay_rows.append(d)

            subset = summary_rows[-len(OBSERVABLE_FAMILIES):]

            best_mc = max(
                subset,
                key=lambda r:
                r[
                    "MC_per_channel_corrected_postwashout"
                ],
            )

            best_level = min(
                subset,
                key=lambda r:
                r["level_CV_RMSE_mean"],
            )

            best_delta = min(
                subset,
                key=lambda r:
                r["delta_CV_RMSE_mean"],
            )

            print(
                f"    Tw(.01)={wash['Tw_0.01']} | "
                f"tau={wash['tau_mem']:.2f} | "
                f"postwash burnin={postwash_burnin}"
            )

            print(
                f"    best MC: "
                f"{best_mc['family']} "
                f"{best_mc['MC_per_channel_corrected_postwashout']:.4f}"
            )

            print(
                f"    best level RMSE: "
                f"{best_level['family']} "
                f"{best_level['level_CV_RMSE_mean']:.4f}"
            )

            print(
                f"    best delta RMSE: "
                f"{best_delta['family']} "
                f"{best_delta['delta_CV_RMSE_mean']:.4f}"
            )

    # =========================================================================
    # RESULTS
    # =========================================================================

    summary = pd.DataFrame(summary_rows)
    by_delay = pd.concat(delay_rows, ignore_index=True)

    summary.to_csv(
        RESULTS /
        "07_memory_dt_mc_rmse_summary.csv",
        index=False,
    )

    by_delay.to_csv(
        RESULTS /
        "07_memory_dt_mc_by_delay.csv",
        index=False,
    )

    by_mc = summary.sort_values(
        [
            "MC_per_channel_corrected_postwashout",
            "level_CV_RMSE_mean",
        ],
        ascending=[False, True],
    ).reset_index(drop=True)

    by_mc["MC_rank"] = (
        np.arange(len(by_mc)) + 1
    )

    by_level = summary.sort_values(
        [
            "level_CV_RMSE_mean",
            "MC_per_channel_corrected_postwashout",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)

    by_level["level_RMSE_rank"] = (
        np.arange(len(by_level)) + 1
    )

    by_delta = summary.sort_values(
        [
            "delta_CV_RMSE_mean",
            "MC_per_channel_corrected_postwashout",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)

    by_delta["delta_RMSE_rank"] = (
        np.arange(len(by_delta)) + 1
    )

    by_mc.to_csv(
        RESULTS /
        "07_memory_dt_ranked_by_MC.csv",
        index=False,
    )

    by_level.to_csv(
        RESULTS /
        "07_memory_dt_ranked_by_level_RMSE.csv",
        index=False,
    )

    by_delta.to_csv(
        RESULTS /
        "07_memory_dt_ranked_by_delta_RMSE.csv",
        index=False,
    )

    # Pareto flags: memory vs each forecasting target.
    pareto = summary.copy()

    pareto["pareto_MC_vs_level"] = (
        pareto_flags(
            pareto,
            "MC_per_channel_corrected_postwashout",
            "level_CV_RMSE_mean",
        )
    ).astype(int)

    pareto["pareto_MC_vs_delta"] = (
        pareto_flags(
            pareto,
            "MC_per_channel_corrected_postwashout",
            "delta_CV_RMSE_mean",
        )
    ).astype(int)

    pareto.to_csv(
        RESULTS /
        "07_memory_dt_pareto.csv",
        index=False,
    )

    # =========================================================================
    # PRINT
    # =========================================================================

    cols_show = [
        "dt",
        "J_cross",
        "J45",
        "family",
        "n_features",
        "MC_per_channel_corrected_postwashout",
        "MC_delay1_mean",
        "MC_delay5_mean",
        "MC_delay10_mean",
        "Tw_0.01",
        "tau_mem",
        "level_CV_RMSE_mean",
        "delta_CV_RMSE_mean",
    ]

    print()
    print("=" * 136)
    print("TOP 15 BY POST-WASHOUT MEMORY CAPACITY")
    print("=" * 136)
    print(
        by_mc[cols_show]
        .head(15)
        .to_string(index=False)
    )

    print()
    print("=" * 136)
    print("TOP 15 BY ORIGINAL CLAIM-LEVEL CV RMSE")
    print("=" * 136)
    print(
        by_level[cols_show]
        .head(15)
        .to_string(index=False)
    )

    print()
    print("=" * 136)
    print("TOP 15 BY DELTA-CLAIM CV RMSE")
    print("=" * 136)
    print(
        by_delta[cols_show]
        .head(15)
        .to_string(index=False)
    )

    pareto_level = pareto[
        pareto["pareto_MC_vs_level"] == 1
    ].sort_values(
        "MC_per_channel_corrected_postwashout",
        ascending=False,
    )

    pareto_delta = pareto[
        pareto["pareto_MC_vs_delta"] == 1
    ].sort_values(
        "MC_per_channel_corrected_postwashout",
        ascending=False,
    )

    print()
    print("=" * 136)
    print("PARETO FRONT: MEMORY vs CLAIM-LEVEL RMSE")
    print("=" * 136)
    print(
        pareto_level[cols_show]
        .to_string(index=False)
    )

    print()
    print("=" * 136)
    print("PARETO FRONT: MEMORY vs DELTA RMSE")
    print("=" * 136)
    print(
        pareto_delta[cols_show]
        .to_string(index=False)
    )

    # Current dt=0.8 winner from previous focused experiment.
    ref = summary[
        np.isclose(summary["dt"], 0.8)
        & np.isclose(summary["J_cross"], -1.0)
        & np.isclose(summary["J45"], -1.0)
        & (summary["family"] == "XYZ_all")
    ]

    print()
    print("=" * 136)
    print("REFERENCE: PREVIOUS FIRST-PASS WINNER")
    print("=" * 136)

    if len(ref):
        print(
            ref[cols_show].to_string(index=False)
        )
    else:
        print("Reference row not found.")

    # -------------------------------------------------------------------------
    # TEXT SUMMARY
    # -------------------------------------------------------------------------
    with open(
        RESULTS /
        "07_memory_dt_mc_rmse_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:

        fp.write(
            "QRC dt SEARCH: MC + WASHOUT + LEVEL RMSE + DELTA RMSE\n"
        )
        fp.write("=" * 110 + "\n\n")

        fp.write(
            "Frozen topology=T2_dual_boundary, seed=42, "
            "hx=0.5, r=2, alpha=0.75\n"
        )
        fp.write(
            f"Forecast Ridge lambda={FORECAST_RIDGE_ALPHA}\n"
        )
        fp.write(
            "2025 not used; 2026 untouched.\n\n"
        )

        fp.write(
            "TOP BY POST-WASHOUT MC:\n"
        )
        fp.write(
            by_mc[cols_show]
            .head(15)
            .to_string(index=False)
        )

        fp.write(
            "\n\nTOP BY LEVEL CV RMSE:\n"
        )
        fp.write(
            by_level[cols_show]
            .head(15)
            .to_string(index=False)
        )

        fp.write(
            "\n\nTOP BY DELTA CV RMSE:\n"
        )
        fp.write(
            by_delta[cols_show]
            .head(15)
            .to_string(index=False)
        )

        fp.write(
            "\n\nPARETO MC vs LEVEL:\n"
        )
        fp.write(
            pareto_level[cols_show]
            .to_string(index=False)
        )

        fp.write(
            "\n\nPARETO MC vs DELTA:\n"
        )
        fp.write(
            pareto_delta[cols_show]
            .to_string(index=False)
        )

    print()
    print("Saved:")
    print("  results/07_memory_dt_mc_rmse_summary.csv")
    print("  results/07_memory_dt_mc_by_delay.csv")
    print("  results/07_memory_dt_ranked_by_MC.csv")
    print("  results/07_memory_dt_ranked_by_level_RMSE.csv")
    print("  results/07_memory_dt_ranked_by_delta_RMSE.csv")
    print("  results/07_memory_dt_pareto.csv")
    print("  results/07_memory_dt_mc_rmse_summary.txt")
    print()
    print(
        "Interpretation rule: do not select a single winner from an "
        "artificial combined score. Inspect MC, washout, level RMSE and "
        "delta RMSE together."
    )


if __name__ == "__main__":
    main()
