"""
FOCUSED MEMORY-OPTIMAL QRC SEARCH
=================================

FIRST-PASS QUESTION
-------------------
Before retuning dt, hx, Trotter depth, or random seeds:

    1. Does the injection-to-memory COUPLING TOPOLOGY matter?
    2. What signed J strength transfers useful information into memory?
    3. Which measured OBSERVABLES expose that memory?

This script therefore freezes:

    reservoir seed = 42
    hx             = 0.5
    dt             = 0.8
    Trotter r      = 2
    alpha          = 0.75

and searches only:

    topology
    J_cross
    J_45
    observable family

The original injection-injection couplings J01, J12, J23 remain exactly those
of the frozen Week-6 seed-42 reservoir.

Memory benchmark
----------------
Use the distribution-matched randomized F4 angle probe:

    - same empirical values / marginals as the real F4 operating data
    - each input channel independently permuted in time
    - temporal autocorrelation removed

For each input channel j and delay k:

    MC_{j,k} = Corr^2[u_j(t-k), uhat_{j,k}(t)]

A separate Ridge readout is trained on the original training interval and
evaluated on the original validation interval.

Important:
    This is a four-channel memory benchmark.
    The main ranking quantity is MC per channel, not the four-channel total.

Washout is also audited from five different initial memory states, but it is
NOT used to discard configurations in this first diagnostic.  We first want
to see the coupling/observable structure clearly.

Dependency
----------
Keep beside this script:
    07_02b_memory_pair_isolation.py

Outputs
-------
results/07_memory_J_observable_search_summary.csv
results/07_memory_J_observable_search_by_delay.csv
results/07_memory_J_observable_search_ranked.csv
results/07_memory_J_observable_search_topology_summary.csv
results/07_memory_J_observable_search_observable_summary.csv
results/07_memory_J_observable_search.txt
"""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


# =============================================================================
# LOAD FROZEN QRC IMPLEMENTATION
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


qrc = load_module(BASE, "qrc_mem_J_search")


# =============================================================================
# FROZEN SETTINGS
# =============================================================================

SEED = 42
ALPHA = 0.75
HX = 0.5
DT = 0.8
TROTTER_R = 2

PROBE_SEED = 79001

K_MAX = 20
COMMON_BURNIN = 100

RIDGE_ALPHA = 1e-6
VAR_TOL = 1e-12

WASHOUT_EPS = 0.01

qrc.ALPHA = ALPHA
qrc.HX = HX
qrc.DT = DT
qrc.TROTTER_R = TROTTER_R
qrc.SEEDS = [SEED]


# =============================================================================
# COUPLING SEARCH
# =============================================================================

# Original injection-only backbone is retained.
BACKBONE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
]

# Different paths by which the 4 injection qubits can drive q4-q5 memory.
TOPOLOGIES = {
    # Original chain:
    # 0-1-2-3-4-5
    "T1_chain": {
        "cross": [(3, 4)],
        "memory": [(4, 5)],
    },

    # More than one route from injection to memory.
    "T2_dual_boundary": {
        "cross": [
            (3, 4),
            (2, 4),
            (3, 5),
        ],
        "memory": [(4, 5)],
    },

    # Broader fan-in to memory.
    "T3_memory_fanin": {
        "cross": [
            (1, 4),
            (2, 4),
            (3, 4),
            (2, 5),
            (3, 5),
        ],
        "memory": [(4, 5)],
    },
}

# Explicit signed couplings.
# Zero is deliberately included as a control.
J_CROSS_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]
J_MEMORY_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]


# =============================================================================
# OBSERVABLE FAMILIES
# =============================================================================

XZ_INJECTION = [
    "X0", "X1", "X2", "X3",
    "Z0", "Z1", "Z2", "Z3",
]

XZ_MEMORY = [
    "X4", "X5",
    "Z4", "Z5",
]

XYZ_MEMORY = [
    "X4", "X5",
    "Y4", "Y5",
    "Z4", "Z5",
]

XZ_ALL = XZ_INJECTION + XZ_MEMORY

XYZ_ALL = [
    "X0", "X1", "X2", "X3", "X4", "X5",
    "Y0", "Y1", "Y2", "Y3", "Y4", "Y5",
    "Z0", "Z1", "Z2", "Z3", "Z4", "Z5",
]

# Explicit memory-pair enrichment.
XZINJ_MEMORY_PAIRS = XZ_INJECTION + [
    "ZZ_45",
    "ZX_45",
    "XZ_45",
    "YX_45",
]

OBSERVABLE_FAMILIES = {
    "XZ_injection": XZ_INJECTION,
    "XZ_memory": XZ_MEMORY,
    "XYZ_memory": XYZ_MEMORY,
    "XZ_all": XZ_ALL,
    "XYZ_all": XYZ_ALL,
    "XZinj_plus_memory_pairs": XZINJ_MEMORY_PAIRS,
}


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

# Use |00> for accessible-memory feature trajectory.
# We already know I/4 is a special task-specific trap for the real F4 path.
MC_INITIAL_STATE = INITIAL_STATES["00"]


# =============================================================================
# LINEAR ALGEBRA
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

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0 else np.nan
    )

    kappa = float(np.exp(slope))

    tau = (
        float(-1.0 / np.log(kappa))
        if 0.0 < kappa < 1.0
        else np.inf
    )

    return kappa, tau, float(r2)


# =============================================================================
# CUSTOM COUPLING / UNITARY
# =============================================================================

def all_search_edges():
    edges = set(BACKBONE_EDGES)

    for meta in TOPOLOGIES.values():
        edges.update(meta["cross"])
        edges.update(meta["memory"])

    return sorted(edges)


ALL_SEARCH_EDGES = all_search_edges()

ZZ_OPS = {
    edge: qrc.pauli_product_full(
        qrc.Z2,
        edge[0],
        qrc.Z2,
        edge[1],
    )
    for edge in ALL_SEARCH_EDGES
}


def original_backbone_couplings():
    J_original = qrc.seed_couplings(SEED)

    return {
        edge: float(J_original[edge])
        for edge in BACKBONE_EDGES
    }


J_BACKBONE = original_backbone_couplings()


def build_J(topology_name, J_cross, J_memory):
    meta = TOPOLOGIES[topology_name]

    J = dict(J_BACKBONE)

    for edge in meta["cross"]:
        J[edge] = float(J_cross)

    for edge in meta["memory"]:
        J[edge] = float(J_memory)

    return J


def build_custom_unitary(
    topology_name,
    J_cross,
    J_memory,
):
    J = build_J(
        topology_name,
        J_cross,
        J_memory,
    )

    edges = list(J.keys())

    U = qrc.I64.copy()

    for _ in range(TROTTER_R):

        for edge in edges:
            theta = (
                2.0
                * J[edge]
                * DT
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
            2.0 * HX * DT / TROTTER_R
        )

        for qubit in range(qrc.N_QUBITS):
            Xq = qrc.FULL_SINGLE_OPS[
                f"X{qubit}"
            ]

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
# RANDOMIZED F4 PROBE
# =============================================================================

def build_probe(real_angles):
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

    for j in range(
        real_angles.shape[1]
    ):
        probe[:, j] = real_angles[
            rng.permutation(
                len(real_angles)
            ),
            j,
        ]

    return probe


def probe_audit(probe):
    max_ac = 0.0

    for j in range(probe.shape[1]):
        for lag in range(1, 21):

            r = np.corrcoef(
                probe[lag:, j],
                probe[:-lag, j],
            )[0, 1]

            if np.isfinite(r):
                max_ac = max(
                    max_ac,
                    abs(float(r)),
                )

    return max_ac


# =============================================================================
# FEATURE BANK
# =============================================================================

def add_Y_features(row, rho_i, rho_m):
    for q in range(qrc.N_INJECTION):
        name = f"Y{q}"
        row[name] = qrc.expectation(
            rho_i,
            qrc.INJECTION_SINGLE_OPS[
                name
            ],
        )

    for q in qrc.MEMORY_QUBITS:
        name = f"Y{q}"
        row[name] = qrc.expectation(
            rho_m,
            qrc.MEMORY_SINGLE_OPS[
                name
            ],
        )

    return row


def build_feature_bank(A_list):
    rho = MC_INITIAL_STATE.copy()
    rows = []

    for A in A_list:
        rho_before = rho

        rho_i, rho_m = (
            qrc.final_reduced_states(
                A,
                rho_before,
            )
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
            name:
            qrc.apply_memory_channel(
                S,
                rho,
            )
            for name, rho in states.items()
        }

        D_curve.append(
            max_pairwise_trace_distance(
                states
            )
        )

    D_curve = np.asarray(
        D_curve,
        dtype=float,
    )

    Tw = stable_threshold_crossing(
        D_curve,
        WASHOUT_EPS,
    )

    kappa, tau, r2 = fit_kappa(
        D_curve
    )

    return {
        "D_final": float(
            D_curve[-1]
        ),
        "Tw_0.01": (
            np.nan
            if Tw is None
            else Tw
        ),
        "kappa_eff": kappa,
        "tau_mem": tau,
        "kappa_fit_R2": r2,
    }


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


def ridge_predict(
    Xtr,
    ytr,
    Xva,
):
    Xtr = np.asarray(
        Xtr,
        dtype=float,
    )

    Xva = np.asarray(
        Xva,
        dtype=float,
    )

    ytr = np.asarray(
        ytr,
        dtype=float,
    )

    mu = np.mean(Xtr, axis=0)
    sd = np.std(
        Xtr,
        axis=0,
        ddof=0,
    )

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
        alpha=RIDGE_ALPHA,
        fit_intercept=True,
    )

    model.fit(
        Xtr_z,
        ytr,
    )

    return model.predict(
        Xva_z
    )


def evaluate_memory(
    feature_bank,
    probe,
    feature_names,
    n_train,
    n_val,
):
    X = feature_bank[
        feature_names
    ].to_numpy(dtype=float)

    # Finite-validation positive r^2 floor.
    null_floor = 1.0 / (
        n_val - 1.0
    )

    rows = []

    for j in range(
        probe.shape[1]
    ):

        for k in range(
            1,
            K_MAX + 1,
        ):

            train_start = max(
                COMMON_BURNIN,
                k,
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
                tr - k,
                j,
            ]

            yva = probe[
                va - k,
                j,
            ]

            pred = ridge_predict(
                X[tr],
                ytr,
                X[va],
            )

            raw = safe_corr2(
                yva,
                pred,
            )

            corrected = max(
                raw - null_floor,
                0.0,
            )

            rows.append({
                "channel": j,
                "delay": k,
                "MC_raw": raw,
                "MC_corrected": corrected,
            })

    df = pd.DataFrame(rows)

    by_channel = (
        df.groupby(
            "channel",
            as_index=False,
        )
        .agg(
            MC_raw_sum=(
                "MC_raw",
                "sum",
            ),
            MC_corrected_sum=(
                "MC_corrected",
                "sum",
            ),
        )
    )

    total_raw = float(
        df["MC_raw"].sum()
    )

    total_corrected = float(
        df[
            "MC_corrected"
        ].sum()
    )

    n_channels = (
        probe.shape[1]
    )

    return {
        "MC_total_raw":
            total_raw,
        "MC_total_corrected":
            total_corrected,
        "MC_per_channel_raw":
            total_raw / n_channels,
        "MC_per_channel_corrected":
            total_corrected / n_channels,
        "MC_delay1_mean":
            float(
                df[df["delay"] == 1][
                    "MC_raw"
                ].mean()
            ),
        "MC_delay5_mean":
            float(
                df[df["delay"] == 5][
                    "MC_raw"
                ].mean()
            ),
        "MC_delay10_mean":
            float(
                df[df["delay"] == 10][
                    "MC_raw"
                ].mean()
            ),
        "MC_delay20_mean":
            float(
                df[df["delay"] == 20][
                    "MC_raw"
                ].mean()
            ),
        "detail": df,
        "by_channel": by_channel,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 132)
    print("FOCUSED MEMORY SEARCH: COUPLING + J STRENGTH + OBSERVABLES")
    print("=" * 132)
    print()

    print("Frozen:")
    print(f"  seed={SEED}")
    print(f"  hx={HX}")
    print(f"  dt={DT}")
    print(f"  Trotter r={TROTTER_R}")
    print(f"  alpha={ALPHA}")
    print()

    print(
        "Backbone J01,J12,J23 "
        "remain the original seed-42 values:"
    )

    for edge, value in J_BACKBONE.items():
        print(
            f"  J{edge} = {value:+.6f}"
        )

    print()

    print(
        f"Explicit J_cross grid = "
        f"{J_CROSS_GRID}"
    )

    print(
        f"Explicit J45 grid     = "
        f"{J_MEMORY_GRID}"
    )

    print()

    print(
        "Observable families = "
        f"{list(OBSERVABLE_FAMILIES)}"
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

    real_angles = np.asarray(
        qrc.make_input_angles(
            work,
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

    # -------------------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------------------
    summary_rows = []
    delay_rows = []

    n_configs = (
        len(TOPOLOGIES)
        * len(J_CROSS_GRID)
        * len(J_MEMORY_GRID)
    )

    counter = 0

    for topology in TOPOLOGIES:

        for J_cross in J_CROSS_GRID:

            for J_memory in J_MEMORY_GRID:

                counter += 1

                print(
                    f"[{counter:3d}/{n_configs}] "
                    f"{topology} | "
                    f"Jcross={J_cross:+.2f} | "
                    f"J45={J_memory:+.2f}"
                )

                U, J, unitary_err = (
                    build_custom_unitary(
                        topology,
                        J_cross,
                        J_memory,
                    )
                )

                A_list, S_list = (
                    qrc.build_input_channels(
                        U,
                        probe,
                    )
                )

                wash = (
                    washout_diagnostic(
                        S_list
                    )
                )

                feature_bank = (
                    build_feature_bank(
                        A_list
                    )
                )

                for (
                    family_name,
                    feature_names,
                ) in OBSERVABLE_FAMILIES.items():

                    mc = evaluate_memory(
                        feature_bank,
                        probe,
                        feature_names,
                        n_train,
                        n_val,
                    )

                    summary_rows.append({
                        "topology":
                            topology,
                        "n_cross_edges":
                            len(
                                TOPOLOGIES[
                                    topology
                                ]["cross"]
                            ),
                        "J_cross":
                            J_cross,
                        "J45":
                            J_memory,
                        "family":
                            family_name,
                        "n_features":
                            len(
                                feature_names
                            ),

                        "unitarity_error":
                            unitary_err,

                        "MC_total_raw":
                            mc[
                                "MC_total_raw"
                            ],
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
                            mc[
                                "MC_delay1_mean"
                            ],
                        "MC_delay5_mean":
                            mc[
                                "MC_delay5_mean"
                            ],
                        "MC_delay10_mean":
                            mc[
                                "MC_delay10_mean"
                            ],
                        "MC_delay20_mean":
                            mc[
                                "MC_delay20_mean"
                            ],

                        "Tw_0.01":
                            wash[
                                "Tw_0.01"
                            ],
                        "D_final":
                            wash[
                                "D_final"
                            ],
                        "kappa_eff":
                            wash[
                                "kappa_eff"
                            ],
                        "tau_mem":
                            wash[
                                "tau_mem"
                            ],
                        "kappa_fit_R2":
                            wash[
                                "kappa_fit_R2"
                            ],
                    })

                    d = mc["detail"].copy()

                    d["topology"] = topology
                    d["J_cross"] = J_cross
                    d["J45"] = J_memory
                    d["family"] = family_name

                    delay_rows.append(d)

                # Compact progress line:
                subset = [
                    r
                    for r in summary_rows
                    if (
                        r["topology"]
                        == topology
                        and
                        r["J_cross"]
                        == J_cross
                        and
                        r["J45"]
                        == J_memory
                    )
                ]

                best_local = max(
                    subset,
                    key=lambda r:
                    r[
                        "MC_per_channel_corrected"
                    ],
                )

                print(
                    f"    best obs="
                    f"{best_local['family']} | "
                    f"MC/ch corr="
                    f"{best_local['MC_per_channel_corrected']:.4f} | "
                    f"Tw(.01)="
                    f"{wash['Tw_0.01']} | "
                    f"tau="
                    f"{wash['tau_mem']:.2f}"
                )

    # -------------------------------------------------------------------------
    # SAVE / RANK
    # -------------------------------------------------------------------------
    summary = pd.DataFrame(
        summary_rows
    )

    by_delay = pd.concat(
        delay_rows,
        ignore_index=True,
    )

    summary.to_csv(
        RESULTS /
        "07_memory_J_observable_search_summary.csv",
        index=False,
    )

    by_delay.to_csv(
        RESULTS /
        "07_memory_J_observable_search_by_delay.csv",
        index=False,
    )

    # Ranking is intentionally based on accessible memory first.
    # Washout is shown alongside rather than used as a hard filter at this stage.
    ranked = summary.sort_values(
        [
            "MC_per_channel_corrected",
            "MC_delay1_mean",
            "n_features",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    ranked["rank"] = (
        np.arange(len(ranked)) + 1
    )

    ranked.to_csv(
        RESULTS /
        "07_memory_J_observable_search_ranked.csv",
        index=False,
    )

    # Best observable for each topology/J combination.
    best_per_dynamics = (
        summary.sort_values(
            "MC_per_channel_corrected",
            ascending=False,
        )
        .groupby(
            [
                "topology",
                "J_cross",
                "J45",
            ],
            as_index=False,
        )
        .first()
        .sort_values(
            "MC_per_channel_corrected",
            ascending=False,
        )
    )

    best_per_dynamics.to_csv(
        RESULTS /
        "07_memory_J_observable_search_topology_summary.csv",
        index=False,
    )

    # Best dynamics for each observable family.
    best_per_observable = (
        summary.sort_values(
            "MC_per_channel_corrected",
            ascending=False,
        )
        .groupby(
            "family",
            as_index=False,
        )
        .first()
        .sort_values(
            "MC_per_channel_corrected",
            ascending=False,
        )
    )

    best_per_observable.to_csv(
        RESULTS /
        "07_memory_J_observable_search_observable_summary.csv",
        index=False,
    )

    print()
    print("=" * 132)
    print("TOP 20 COUPLING + OBSERVABLE COMBINATIONS")
    print("=" * 132)

    print(
        ranked[
            [
                "rank",
                "topology",
                "n_cross_edges",
                "J_cross",
                "J45",
                "family",
                "n_features",
                "MC_per_channel_raw",
                "MC_per_channel_corrected",
                "MC_delay1_mean",
                "MC_delay5_mean",
                "MC_delay10_mean",
                "MC_delay20_mean",
                "Tw_0.01",
                "tau_mem",
            ]
        ].head(20).to_string(index=False)
    )

    print()
    print("=" * 132)
    print("BEST DYNAMICS FOR EACH OBSERVABLE FAMILY")
    print("=" * 132)

    print(
        best_per_observable[
            [
                "family",
                "topology",
                "J_cross",
                "J45",
                "MC_per_channel_raw",
                "MC_per_channel_corrected",
                "MC_delay1_mean",
                "MC_delay5_mean",
                "MC_delay10_mean",
                "MC_delay20_mean",
                "Tw_0.01",
                "tau_mem",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 132)
    print("BEST OBSERVABLE FOR EACH OF THE TOP DYNAMICS")
    print("=" * 132)

    print(
        best_per_dynamics[
            [
                "topology",
                "J_cross",
                "J45",
                "family",
                "MC_per_channel_corrected",
                "Tw_0.01",
                "tau_mem",
            ]
        ].head(20).to_string(index=False)
    )

    winner = ranked.iloc[0]

    print()
    print("=" * 132)
    print("FIRST-PASS WINNER")
    print("=" * 132)

    print(
        winner[
            [
                "topology",
                "J_cross",
                "J45",
                "family",
                "n_features",
                "MC_per_channel_raw",
                "MC_per_channel_corrected",
                "MC_delay1_mean",
                "MC_delay5_mean",
                "MC_delay10_mean",
                "MC_delay20_mean",
                "Tw_0.01",
                "kappa_eff",
                "tau_mem",
            ]
        ].to_string()
    )

    with open(
        RESULTS /
        "07_memory_J_observable_search.txt",
        "w",
        encoding="utf-8",
    ) as fp:

        fp.write(
            "FOCUSED MEMORY SEARCH: COUPLING + J + OBSERVABLES\n"
        )
        fp.write(
            "=" * 100 + "\n\n"
        )

        fp.write(
            "Frozen: seed=42, hx=0.5, dt=0.8, r=2, alpha=0.75\n\n"
        )

        fp.write(
            "TOP 20:\n"
        )

        fp.write(
            ranked[
                [
                    "rank",
                    "topology",
                    "J_cross",
                    "J45",
                    "family",
                    "MC_per_channel_corrected",
                    "Tw_0.01",
                    "tau_mem",
                ]
            ].head(20).to_string(index=False)
        )

        fp.write(
            "\n\nBEST PER OBSERVABLE:\n"
        )

        fp.write(
            best_per_observable.to_string(
                index=False
            )
        )

        fp.write(
            "\n\nFIRST-PASS WINNER:\n"
        )

        fp.write(
            winner.to_string()
        )

        fp.write("\n")

    print()
    print("Saved:")
    print(
        "  results/07_memory_J_observable_search_summary.csv"
    )
    print(
        "  results/07_memory_J_observable_search_by_delay.csv"
    )
    print(
        "  results/07_memory_J_observable_search_ranked.csv"
    )
    print(
        "  results/07_memory_J_observable_search_topology_summary.csv"
    )
    print(
        "  results/07_memory_J_observable_search_observable_summary.csv"
    )
    print(
        "  results/07_memory_J_observable_search.txt"
    )
    print()
    print(
        "Do not retune dt/hx/r yet. "
        "First interpret which coupling pattern and observable family "
        "actually create accessible delayed-input memory."
    )


if __name__ == "__main__":
    main()
