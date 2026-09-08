"""
FINAL BOUNDARY-EXTENSION TEST FOR MEMORY-OPTIMAL QRC
====================================================

Purpose
-------
The previous heterogeneous-J optimization found its best configuration at

    dt  = 1.5
    J24 = -0.907276
    J34 = -0.444414
    J35 = -1.600000   <-- exactly at the imposed search boundary
    J45 = -1.194296

with

    corrected MC/ch    = 1.212763
    corrected MC total = 4.851052
    T_w(0.01)          = 44

Because J35 hit the boundary, one final focused extension is scientifically
necessary before freezing the memory-optimal QRC.

This script changes ONLY the unresolved neighborhood:

    dt  in {1.4, 1.5, 1.6}
    J35 in {-1.6,-1.8,-2.0,-2.2,-2.4,-2.6}
    J45 in {-1.0,-1.2,-1.4}

while fixing

    J24 = -0.907276
    J34 = -0.444414

Everything else stays frozen:

    topology  = T2_dual_boundary
    seed      = 42
    hx        = 0.5
    Trotter r = 2
    alpha     = 0.75
    readout   = XYZ_all

Search size
-----------
3 dt values * 6 J35 values * 3 J45 values = 54 candidates.

Selection
---------
Primary:
    maximize post-washout corrected MC/ch

Constraint:
    T_w(0.01) finite and <= 300

Secondary:
    inspect MC at delays 1,2,5,10,20

Forecasting
-----------
The top 5 valid memory candidates receive 2022-2024 chronological 5-fold CV
for:
    C_{t+1}
    Delta C_{t+1}

2025 is not used.
2026 remains untouched.

Final conclusion rule
---------------------
Compare against MC/ch = 1.212763:

    < 3% gain:
        freeze previous simpler winner / plateau reached

    3-10% gain:
        freeze new extended-J winner, modest meaningful gain

    > 10% gain:
        freeze new extended-J winner, strong gain

Regardless of outcome, STOP ideal-memory optimization after this script and
move to hardware/resource profiling.

Dependency
----------
Keep beside:
    07_02b_memory_pair_isolation.py

Outputs
-------
results/07_memory_highJ_boundary_summary.csv
results/07_memory_highJ_boundary_by_delay.csv
results/07_memory_highJ_boundary_ranked.csv
results/07_memory_highJ_boundary_finalists_RMSE.csv
results/07_memory_highJ_boundary_conclusion.txt
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


qrc = load_module(BASE, "qrc_highJ_boundary")


# =============================================================================
# FROZEN SETTINGS
# =============================================================================

SEED = 42
ALPHA = 0.75
HX = 0.5
TROTTER_R = 2

J24_FIXED = -0.907276
J34_FIXED = -0.444414

DT_GRID = [1.4, 1.5, 1.6]
J35_GRID = [-1.6, -1.8, -2.0, -2.2, -2.4, -2.6]
J45_GRID = [-1.0, -1.2, -1.4]

REFERENCE_DT = 1.5
REFERENCE_J24 = -0.907276
REFERENCE_J34 = -0.444414
REFERENCE_J35 = -1.600000
REFERENCE_J45 = -1.194296

REFERENCE_MC_CH = 1.212763
REFERENCE_MC_TOTAL = 4.851052
REFERENCE_TW = 44

PROBE_SEED = 79001

K_MAX = 20
MIN_BURNIN = 100
POST_WASHOUT_MARGIN = 20
MAX_ACCEPTABLE_WASHOUT = 300

MC_RIDGE_ALPHA = 1e-6
FORECAST_RIDGE_ALPHA = 0.01
CV_FOLDS = 5
VAR_TOL = 1e-12
N_RMSE_FINALISTS = 5

qrc.ALPHA = ALPHA
qrc.HX = HX
qrc.TROTTER_R = TROTTER_R
qrc.SEEDS = [SEED]


# =============================================================================
# T2 DUAL-BOUNDARY TOPOLOGY
# =============================================================================

BACKBONE_EDGES = [(0, 1), (1, 2), (2, 3)]

EDGE_24 = (2, 4)
EDGE_34 = (3, 4)
EDGE_35 = (3, 5)
EDGE_45 = (4, 5)

ALL_EDGES = BACKBONE_EDGES + [
    EDGE_24,
    EDGE_34,
    EDGE_35,
    EDGE_45,
]


def original_backbone_couplings():
    original = qrc.seed_couplings(SEED)
    return {
        edge: float(original[edge])
        for edge in BACKBONE_EDGES
    }


J_BACKBONE = original_backbone_couplings()

ZZ_OPS = {
    edge: qrc.pauli_product_full(
        qrc.Z2, edge[0], qrc.Z2, edge[1]
    )
    for edge in ALL_EDGES
}


# =============================================================================
# XYZ ALL OBSERVABLES
# =============================================================================

XYZ_ALL = [
    "X0", "X1", "X2", "X3", "X4", "X5",
    "Y0", "Y1", "Y2", "Y3", "Y4", "Y5",
    "Z0", "Z1", "Z2", "Z3", "Z4", "Z5",
]


# =============================================================================
# INITIAL MEMORY STATES
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
# BASIC MATH
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
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    kappa = float(np.exp(slope))
    tau = (
        float(-1.0 / np.log(kappa))
        if 0.0 < kappa < 1.0
        else np.inf
    )

    return kappa, tau, r2


def rmse(y_true, y_pred):
    return float(
        np.sqrt(mean_squared_error(y_true, y_pred))
    )


# =============================================================================
# UNITARY
# =============================================================================

def build_unitary(dt, J35, J45):
    J = dict(J_BACKBONE)
    J[EDGE_24] = J24_FIXED
    J[EDGE_34] = J34_FIXED
    J[EDGE_35] = float(J35)
    J[EDGE_45] = float(J45)

    U = qrc.I64.copy()

    for _ in range(TROTTER_R):
        for edge in ALL_EDGES:
            theta = 2.0 * J[edge] * dt / TROTTER_R
            ZZ = ZZ_OPS[edge]

            G = (
                np.cos(theta / 2.0) * qrc.I64
                - 1j * np.sin(theta / 2.0) * ZZ
            )
            U = G @ U

        theta_x = 2.0 * HX * dt / TROTTER_R

        for qubit in range(qrc.N_QUBITS):
            Xq = qrc.FULL_SINGLE_OPS[f"X{qubit}"]

            G = (
                np.cos(theta_x / 2.0) * qrc.I64
                - 1j * np.sin(theta_x / 2.0) * Xq
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
                max_abs = max(max_abs, abs(float(r)))

    return max_abs


# =============================================================================
# FEATURES
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


def build_feature_bank(A_list):
    rho = RESERVOIR_INITIAL_STATE.copy()
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

        row = add_Y_features(row, rho_i, rho_m)

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
# RIDGE / MEMORY CAPACITY
# =============================================================================

def safe_corr2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if np.std(a) <= VAR_TOL or np.std(b) <= VAR_TOL:
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

            start = max(burnin, delay)

            tr = np.arange(start, n_train)
            va = np.arange(
                n_train,
                n_train + n_val,
            )

            ytr = probe[tr - delay, channel]
            yva = probe[va - delay, channel]

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

    total_raw = float(detail["MC_raw"].sum())
    total_corrected = float(
        detail["MC_corrected"].sum()
    )

    nch = probe.shape[1]

    def delay_mean(k, col="MC_raw"):
        x = detail.loc[
            detail["delay"] == k,
            col,
        ]
        return float(x.mean())

    return {
        "MC_total_raw": total_raw,
        "MC_total_corrected": total_corrected,
        "MC_per_channel_raw": total_raw / nch,
        "MC_per_channel_corrected":
            total_corrected / nch,

        "MC_delay1_mean": delay_mean(1),
        "MC_delay2_mean": delay_mean(2),
        "MC_delay5_mean": delay_mean(5),
        "MC_delay10_mean": delay_mean(10),
        "MC_delay20_mean": delay_mean(20),

        "detail": detail,
    }


# =============================================================================
# FORECASTING DIAGNOSTICS
# =============================================================================

def chronological_cv_rmse(X, y, n_train):
    X = np.asarray(X, dtype=float)[:n_train]
    y = np.asarray(y, dtype=float)[:n_train]

    splitter = TimeSeriesSplit(
        n_splits=CV_FOLDS
    )

    vals = []

    for tr, va in splitter.split(X):
        pred = standardized_ridge_predict(
            X[tr],
            y[tr],
            X[va],
            FORECAST_RIDGE_ALPHA,
        )

        vals.append(
            rmse(y[va], pred)
        )

    vals = np.asarray(vals, dtype=float)

    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "folds": vals,
    }


def resolve_column(df, candidates, label):
    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for c in candidates:
        if c in df.columns:
            return c

        if c.lower() in lower:
            return lower[c.lower()]

    raise KeyError(
        f"Could not find {label}. Tried {candidates}."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 136)
    print("FINAL HIGH-J BOUNDARY EXTENSION")
    print("=" * 136)
    print()

    print("Fixed from current winner:")
    print(f"  J24 = {J24_FIXED:+.6f}")
    print(f"  J34 = {J34_FIXED:+.6f}")
    print()

    print(f"dt grid  = {DT_GRID}")
    print(f"J35 grid = {J35_GRID}")
    print(f"J45 grid = {J45_GRID}")
    print()

    print("Reference:")
    print(
        f"  MC/ch={REFERENCE_MC_CH:.6f}, "
        f"MC_total={REFERENCE_MC_TOTAL:.6f}, "
        f"Tw(.01)={REFERENCE_TW}"
    )
    print()

    work, train, val, cols = qrc.load_data()

    n_train = len(train)
    n_val = len(val)
    n_steps = n_train + n_val

    work_eval = work.iloc[:n_steps].copy()

    real_angles = np.asarray(
        qrc.make_input_angles(
            work_eval,
            cols,
        ),
        dtype=float,
    )[:n_steps]

    probe = build_probe(real_angles)

    print(
        f"Probe max |autocorrelation| lags 1..20 = "
        f"{probe_audit(probe):.6f}"
    )
    print()

    rows = []
    delay_rows = []

    total = (
        len(DT_GRID)
        * len(J35_GRID)
        * len(J45_GRID)
    )

    counter = 0

    for dt in DT_GRID:
        for J35 in J35_GRID:
            for J45 in J45_GRID:

                counter += 1

                print(
                    f"[{counter:2d}/{total}] "
                    f"dt={dt:.1f} | "
                    f"J35={J35:+.2f} | "
                    f"J45={J45:+.2f}"
                )

                U, J, unitary_err = build_unitary(
                    dt,
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

                if np.isfinite(
                    wash["Tw_0.01"]
                ):
                    burnin = int(
                        max(
                            MIN_BURNIN,
                            int(
                                wash[
                                    "Tw_0.01"
                                ]
                            )
                            + POST_WASHOUT_MARGIN,
                        )
                    )
                else:
                    burnin = n_train - 100

                burnin = min(
                    burnin,
                    n_train - 100,
                )

                bank = build_feature_bank(
                    A_probe
                )

                mc = evaluate_memory(
                    bank,
                    probe,
                    n_train,
                    n_val,
                    burnin,
                )

                valid = int(
                    np.isfinite(
                        wash["Tw_0.01"]
                    )
                    and
                    wash["Tw_0.01"]
                    <= MAX_ACCEPTABLE_WASHOUT
                )

                row = {
                    "dt": dt,
                    "J24": J24_FIXED,
                    "J34": J34_FIXED,
                    "J35": J35,
                    "J45": J45,

                    "unitarity_error":
                        unitary_err,

                    "MC_burnin":
                        burnin,

                    "MC_per_channel_corrected":
                        mc[
                            "MC_per_channel_corrected"
                        ],
                    "MC_total_corrected":
                        mc[
                            "MC_total_corrected"
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

                    "Tw_0.01":
                        wash["Tw_0.01"],
                    "tau_mem":
                        wash["tau_mem"],
                    "D_final":
                        wash["D_final"],
                    "kappa_eff":
                        wash["kappa_eff"],
                    "washout_valid":
                        valid,
                }

                rows.append(row)

                d = mc["detail"].copy()
                d["dt"] = dt
                d["J35"] = J35
                d["J45"] = J45

                delay_rows.append(d)

                gain = 100.0 * (
                    row[
                        "MC_per_channel_corrected"
                    ]
                    - REFERENCE_MC_CH
                ) / REFERENCE_MC_CH

                print(
                    f"    MC/ch="
                    f"{row['MC_per_channel_corrected']:.4f} | "
                    f"MC_total="
                    f"{row['MC_total_corrected']:.4f} | "
                    f"gain={gain:+.2f}% | "
                    f"Tw={row['Tw_0.01']} | "
                    f"valid={valid}"
                )

    summary = pd.DataFrame(rows)

    ranked = summary.sort_values(
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

    summary.to_csv(
        RESULTS /
        "07_memory_highJ_boundary_summary.csv",
        index=False,
    )

    ranked.to_csv(
        RESULTS /
        "07_memory_highJ_boundary_ranked.csv",
        index=False,
    )

    pd.concat(
        delay_rows,
        ignore_index=True,
    ).to_csv(
        RESULTS /
        "07_memory_highJ_boundary_by_delay.csv",
        index=False,
    )

    show_cols = [
        "rank",
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

    print()
    print("=" * 136)
    print("TOP 15 HIGH-J CANDIDATES")
    print("=" * 136)

    print(
        ranked[
            show_cols
        ].head(15).to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Forecast only top valid memory finalists.
    # -------------------------------------------------------------------------
    valid = ranked[
        ranked["washout_valid"] == 1
    ]

    finalists = (
        valid.head(N_RMSE_FINALISTS)
        if len(valid)
        else ranked.head(N_RMSE_FINALISTS)
    ).copy()

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
    ].to_numpy(dtype=float)

    current_claims = work_eval[
        current_col
    ].to_numpy(dtype=float)

    y_delta = y_level - current_claims

    finalist_rows = []

    print()
    print("=" * 136)
    print("FORECASTING DIAGNOSTICS FOR TOP 5 MEMORY FINALISTS")
    print("=" * 136)

    for i, (_, row) in enumerate(
        finalists.iterrows(),
        start=1,
    ):
        U, _, _ = build_unitary(
            float(row["dt"]),
            float(row["J35"]),
            float(row["J45"]),
        )

        A_real, _ = qrc.build_input_channels(
            U,
            real_angles,
        )

        bank_real = build_feature_bank(
            A_real
        )

        X = bank_real[
            XYZ_ALL
        ].to_numpy(dtype=float)

        level_cv = chronological_cv_rmse(
            X,
            y_level,
            n_train,
        )

        delta_cv = chronological_cv_rmse(
            X,
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
        "07_memory_highJ_boundary_finalists_RMSE.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final conclusion.
    # -------------------------------------------------------------------------
    best = (
        valid.iloc[0]
        if len(valid)
        else ranked.iloc[0]
    )

    best_mc = float(
        best[
            "MC_per_channel_corrected"
        ]
    )

    gain_pct = 100.0 * (
        best_mc - REFERENCE_MC_CH
    ) / REFERENCE_MC_CH

    # Boundary warning.
    hit_lower_J35_boundary = np.isclose(
        float(best["J35"]),
        min(J35_GRID),
    )

    if gain_pct < 3.0:
        conclusion = (
            "PLATEAU: higher |J35| gives <3% improvement. "
            "Freeze the previous memory-optimal configuration and stop "
            "ideal-memory optimization."
        )
    elif gain_pct < 10.0:
        conclusion = (
            "MODEST FINAL GAIN: freeze the best high-J configuration. "
            "The extension is useful but the memory search is now complete."
        )
    else:
        conclusion = (
            "STRONG FINAL GAIN: freeze the best high-J configuration. "
            "The memory search is now complete."
        )

    if hit_lower_J35_boundary:
        boundary_note = (
            "The winner lies at the most negative J35 tested. "
            "Nevertheless, by design this is the final ideal-memory search; "
            "do not keep extending J indefinitely. Hardware compilation and "
            "resource cost now become the relevant constraints."
        )
    else:
        boundary_note = (
            "The winner is inside the tested J35 interval, so the high-J "
            "boundary is resolved cleanly."
        )

    print()
    print("=" * 136)
    print("FINAL MEMORY-OPTIMIZATION CONCLUSION")
    print("=" * 136)

    print(
        best[
            [
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
        f"Gain over previous memory winner = "
        f"{gain_pct:+.2f}%"
    )
    print(conclusion)
    print(boundary_note)

    with open(
        RESULTS /
        "07_memory_highJ_boundary_conclusion.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "FINAL HIGH-J MEMORY BOUNDARY TEST\n"
        )
        fp.write("=" * 100 + "\n\n")

        fp.write(
            f"Reference MC/ch = {REFERENCE_MC_CH:.6f}\n"
        )
        fp.write(
            f"Reference MC total = {REFERENCE_MC_TOTAL:.6f}\n"
        )
        fp.write(
            f"Reference Tw(.01) = {REFERENCE_TW}\n\n"
        )

        fp.write("Best high-J result:\n")
        fp.write(best.to_string())
        fp.write("\n\n")

        fp.write(
            f"Gain over reference = {gain_pct:+.2f}%\n"
        )
        fp.write(conclusion + "\n")
        fp.write(boundary_note + "\n")

        if len(finalists_df):
            fp.write(
                "\nTop-memory finalist forecasting diagnostics:\n"
            )
            fp.write(
                finalists_df[
                    [
                        "dt",
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
        "  results/07_memory_highJ_boundary_summary.csv"
    )
    print(
        "  results/07_memory_highJ_boundary_by_delay.csv"
    )
    print(
        "  results/07_memory_highJ_boundary_ranked.csv"
    )
    print(
        "  results/07_memory_highJ_boundary_finalists_RMSE.csv"
    )
    print(
        "  results/07_memory_highJ_boundary_conclusion.txt"
    )


if __name__ == "__main__":
    main()
