"""
Week 7 - Step 7.1 (revised)
Observable-information study across RWP temporal windows and CONT.

Scientific rule
---------------
Keep the Week-6 ideal reservoir fixed and compare the same observable families under the Week-6 temporal protocols: CONT and RWP(W).

For every prediction and every W:
    - reset the two memory qubits to |00><00| at the START of the window,
    - replay the W chronological F4 inputs,
    - after each input/reservoir step, trace out the four injection qubits,
      keep rho_M, and inject the next fresh F4 state,
    - extract observables only after the FINAL step of the window.

RWP W>1 genuinely exercises finite temporal memory, while CONT carries memory through the full chronology. RWP W=1 remains the no-history reference.

Primary selection:
    5-fold chronological CV on 2022-2024 training data.

Diagnostics only:
    2025 validation.

Untouched:
    2026 test is not simulated, fitted, scored, or used.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Frozen Week-6 configuration
# =============================================================================

RESULTS = Path("results")
DATA_FILE = RESULTS / "03_01_preprocessed_samples.csv"

N_QUBITS = 6
N_INJECTION = 4
N_MEMORY = 2
DIM_I = 2**N_INJECTION   # 16
DIM_M = 2**N_MEMORY      # 4
MEMORY_QUBITS = (4, 5)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

ALPHA = 0.005
HX = 0.5
DT = 0.8
TROTTER_R = 2
SEEDS = [42, 101, 202, 505, 707]

# The temporal-window screen requested for the corrected observable study.
WINDOWS = [1, 2, 5, 7, 14, 21, 28]

RIDGE_GRID = np.array(
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0],
    dtype=float,
)
N_CV_SPLITS = 5

# Frozen Week-6 r=2, X+Z references for every temporal protocol.
# These are used only as a reproduction audit; Week 7 changes observables,
# not the reservoir dynamics or temporal protocol definitions.
WEEK6_XZ_REF = {
    42: {
        "CONT":    {"cv": 3.588632, "val": 5.156031},
        "RWP_W1":  {"cv": 3.577811, "val": 5.128391},
        "RWP_W2":  {"cv": 3.584427, "val": 5.129457},
        "RWP_W5":  {"cv": 3.604740, "val": 5.168780},
        "RWP_W7":  {"cv": 3.617304, "val": 5.159828},
        "RWP_W14": {"cv": 3.622905, "val": 5.169476},
        "RWP_W21": {"cv": 3.638049, "val": 5.155071},
        "RWP_W28": {"cv": 3.696277, "val": 5.156583},
    },
    101: {
        "CONT":    {"cv": 3.629735, "val": 5.081633},
        "RWP_W1":  {"cv": 3.581138, "val": 5.136548},
        "RWP_W2":  {"cv": 3.586706, "val": 5.133072},
        "RWP_W5":  {"cv": 3.656074, "val": 5.172265},
        "RWP_W7":  {"cv": 3.645019, "val": 5.165367},
        "RWP_W14": {"cv": 3.635135, "val": 5.164759},
        "RWP_W21": {"cv": 3.697017, "val": 5.156901},
        "RWP_W28": {"cv": 3.644343, "val": 5.170468},
    },
    202: {
        "CONT":    {"cv": 3.600109, "val": 5.154758},
        "RWP_W1":  {"cv": 3.576847, "val": 5.113934},
        "RWP_W2":  {"cv": 3.593091, "val": 5.113949},
        "RWP_W5":  {"cv": 3.643715, "val": 5.154856},
        "RWP_W7":  {"cv": 3.628273, "val": 5.148461},
        "RWP_W14": {"cv": 3.637760, "val": 5.142233},
        "RWP_W21": {"cv": 3.670090, "val": 5.151656},
        "RWP_W28": {"cv": 3.643596, "val": 5.152490},
    },
    505: {
        "CONT":    {"cv": 3.717280, "val": 5.368222},
        "RWP_W1":  {"cv": 3.583733, "val": 5.137218},
        "RWP_W2":  {"cv": 3.625320, "val": 5.553245},
        "RWP_W5":  {"cv": 3.657781, "val": 5.356982},
        "RWP_W7":  {"cv": 3.647114, "val": 5.360796},
        "RWP_W14": {"cv": 3.674902, "val": 5.345020},
        "RWP_W21": {"cv": 3.638331, "val": 5.362426},
        "RWP_W28": {"cv": 3.661268, "val": 5.380532},
    },
    707: {
        "CONT":    {"cv": 3.681279, "val": 5.158272},
        "RWP_W1":  {"cv": 3.583607, "val": 5.166460},
        "RWP_W2":  {"cv": 3.588012, "val": 5.167525},
        "RWP_W5":  {"cv": 3.622606, "val": 5.180314},
        "RWP_W7":  {"cv": 3.679252, "val": 5.190069},
        "RWP_W14": {"cv": 3.626493, "val": 5.166022},
        "RWP_W21": {"cv": 3.639941, "val": 5.178298},
        "RWP_W28": {"cv": 3.683037, "val": 5.176355},
    },
}
AUDIT_TOL = 2e-3


# =============================================================================
# Pauli operators
# =============================================================================

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
I64 = np.eye(2**N_QUBITS, dtype=complex)


def kron_all(mats):
    return reduce(np.kron, mats)


def pauli_on_full_qubit(pauli: np.ndarray, q: int) -> np.ndarray:
    mats = [I2] * N_QUBITS
    mats[q] = pauli
    return kron_all(mats)


def pauli_product_full(pauli_a, q_a, pauli_b, q_b):
    mats = [I2] * N_QUBITS
    mats[q_a] = pauli_a
    mats[q_b] = pauli_b
    return kron_all(mats)


FULL_SINGLE_OPS = {}
for q in range(N_QUBITS):
    FULL_SINGLE_OPS[f"X{q}"] = pauli_on_full_qubit(X2, q)
    FULL_SINGLE_OPS[f"Y{q}"] = pauli_on_full_qubit(Y2, q)
    FULL_SINGLE_OPS[f"Z{q}"] = pauli_on_full_qubit(Z2, q)

ZZ_OPS = {
    (i, j): pauli_product_full(Z2, i, Z2, j)
    for i, j in EDGES
}


def local_op(pauli: np.ndarray, q_local: int, n_qubits: int) -> np.ndarray:
    mats = [I2] * n_qubits
    mats[q_local] = pauli
    return kron_all(mats)


# Reduced-space operators used after partial tracing.
INJECTION_OPS = {}
for q in range(N_INJECTION):
    INJECTION_OPS[f"X{q}"] = local_op(X2, q, N_INJECTION)
    INJECTION_OPS[f"Y{q}"] = local_op(Y2, q, N_INJECTION)
    INJECTION_OPS[f"Z{q}"] = local_op(Z2, q, N_INJECTION)

MEMORY_OPS = {}
for q_global in MEMORY_QUBITS:
    q_local = q_global - N_INJECTION
    MEMORY_OPS[f"X{q_global}"] = local_op(X2, q_local, N_MEMORY)
    MEMORY_OPS[f"Y{q_global}"] = local_op(Y2, q_local, N_MEMORY)
    MEMORY_OPS[f"Z{q_global}"] = local_op(Z2, q_local, N_MEMORY)


# =============================================================================
# Observable families
# =============================================================================

FAMILIES = {
    "Z_all": [f"Z{q}" for q in range(N_QUBITS)],
    "X_all": [f"X{q}" for q in range(N_QUBITS)],
    "Y_all": [f"Y{q}" for q in range(N_QUBITS)],

    # Week-6 reference family.
    "XZ_all": (
        [f"X{q}" for q in range(N_QUBITS)]
        + [f"Z{q}" for q in range(N_QUBITS)]
    ),

    # Injection versus memory location diagnostics.
    "XZ_injection": (
        [f"X{q}" for q in range(N_INJECTION)]
        + [f"Z{q}" for q in range(N_INJECTION)]
    ),
    "XZ_memory": ["X4", "X5", "Z4", "Z5"],

    # Hardware-motivated enrichment.
    "XZ_plus_Ymemory": (
        [f"X{q}" for q in range(N_QUBITS)]
        + [f"Z{q}" for q in range(N_QUBITS)]
        + ["Y4", "Y5"]
    ),

    # All one-body Pauli coordinates.
    "XYZ_all": (
        [f"X{q}" for q in range(N_QUBITS)]
        + [f"Y{q}" for q in range(N_QUBITS)]
        + [f"Z{q}" for q in range(N_QUBITS)]
    ),
}

ALL_OBSERVABLES = sorted(
    {name for names in FAMILIES.values() for name in names},
    key=lambda x: (x[0], int(x[1:])),
)


# =============================================================================
# Data
# =============================================================================

def find_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "None of the expected columns were found:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\n\nAvailable columns:\n"
        + ", ".join(df.columns)
    )


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Run the frozen Week-3 preprocessing first."
        )

    df = pd.read_csv(DATA_FILE)

    input_date_col = find_column(df, ["input_date", "date_t"])
    target_date_col = find_column(df, ["target_date", "date_t_plus_1"])
    split_col = find_column(df, ["split"])
    target_col = find_column(
        df,
        ["target_property_damage_claim_count", "C_t_plus_1", "target"],
    )
    c_col = find_column(df, ["C_t_z"])
    p_col = find_column(df, ["P_t_z"])
    h_col = find_column(
        df,
        ["is_public_holiday_t_plus_1", "H_t_plus_1", "H_t+1"],
    )

    df[input_date_col] = pd.to_datetime(df[input_date_col])
    df[target_date_col] = pd.to_datetime(df[target_date_col])
    df = df.sort_values(target_date_col).reset_index(drop=True)

    # Deliberately exclude test/2026 from the entire Week-7.1 computation.
    work = df[df[split_col].isin(["train", "validation"])].copy()
    work = work.sort_values(target_date_col).reset_index(drop=True)

    train = work[work[split_col] == "train"].copy().reset_index(drop=True)
    val = work[work[split_col] == "validation"].copy().reset_index(drop=True)

    assert len(train) == 1095, f"Expected 1095 train samples, got {len(train)}"
    assert len(val) == 365, f"Expected 365 validation samples, got {len(val)}"
    assert (train[target_date_col].dt.year <= 2024).all()
    assert (val[target_date_col].dt.year == 2025).all()

    cols = {
        "input_date": input_date_col,
        "target_date": target_date_col,
        "split": split_col,
        "target": target_col,
        "C": c_col,
        "P": p_col,
        "H": h_col,
    }
    return work, train, val, cols


# =============================================================================
# Frozen F4 encoding
# =============================================================================

def softsign3(z):
    z = np.asarray(z, dtype=float)
    return z / (3.0 + np.abs(z))


def make_input_angles(df: pd.DataFrame, cols):
    """
    q0 = C_t
    q1 = weekday of t+1
    q2 = P_t
    q3 = H_{t+1}
    q4,q5 = persistent memory inside each rewound window
    """
    c_z = df[cols["C"]].to_numpy(dtype=float)
    p_z = df[cols["P"]].to_numpy(dtype=float)
    h = df[cols["H"]].to_numpy(dtype=float)

    c_enc = np.clip(c_z, -3.0, 3.0) / 3.0
    p_enc = softsign3(p_z)
    weekday0 = df[cols["target_date"]].dt.dayofweek.to_numpy(dtype=float)

    theta = np.zeros((len(df), N_INJECTION), dtype=float)
    theta[:, 0] = ALPHA * c_enc
    theta[:, 1] = 2.0 * np.pi * weekday0 / 7.0
    theta[:, 2] = ALPHA * p_enc
    theta[:, 3] = np.pi * h
    return theta


def ry0(theta):
    return np.array(
        [np.cos(theta / 2.0), np.sin(theta / 2.0)],
        dtype=complex,
    )


def injection_vector(angles_row: np.ndarray) -> np.ndarray:
    return kron_all([ry0(a) for a in angles_row])


# =============================================================================
# Frozen reservoir
# =============================================================================

def seed_couplings(seed: int):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(-0.7, 0.7, len(EDGES))
    return {edge: float(v) for edge, v in zip(EDGES, vals)}


def build_trotter_unitary(seed: int):
    J = seed_couplings(seed)
    U = I64.copy()

    for _ in range(TROTTER_R):
        for edge in EDGES:
            theta = 2.0 * J[edge] * DT / TROTTER_R
            ZZ = ZZ_OPS[edge]
            G = np.cos(theta / 2.0) * I64 - 1j * np.sin(theta / 2.0) * ZZ
            U = G @ U

        theta_x = 2.0 * HX * DT / TROTTER_R
        for q in range(N_QUBITS):
            Xq = FULL_SINGLE_OPS[f"X{q}"]
            G = np.cos(theta_x / 2.0) * I64 - 1j * np.sin(theta_x / 2.0) * Xq
            U = G @ U

    err = np.linalg.norm(U.conj().T @ U - I64, ord="fro")
    return U, err


# =============================================================================
# Efficient rewound-window memory dynamics
# =============================================================================

def build_input_channels(U: np.ndarray, angles: np.ndarray):
    """
    Build, for each chronological input t, the effective memory channel.

    Let phi_t be the fresh 4-qubit injection state and rho_M the carried
    2-qubit memory state. With U reshaped as

        U[a,m,b,n]

    (a,b injection indices; m,n memory indices), define

        A_t[a,m,n] = sum_b U[a,m,b,n] phi_t[b].

    The memory update is

        rho_M' = sum_a K_a rho_M K_a^dagger,
        K_a[m,n] = A_t[a,m,n].

    We precompute the corresponding 16x16 superoperator S_t so replaying
    W steps is cheap.
    """
    U4 = U.reshape(DIM_I, DIM_M, DIM_I, DIM_M)

    A_list = np.empty((len(angles), DIM_I, DIM_M, DIM_M), dtype=complex)
    S_list = np.empty((len(angles), DIM_M * DIM_M, DIM_M * DIM_M), dtype=complex)

    for t, row in enumerate(angles):
        phi = injection_vector(row)
        A = np.einsum("ambn,b->amn", U4, phi, optimize=True)
        # S[m,p,n,k] maps rho[n,k] -> rho'[m,p]
        S = np.einsum("amn,apk->mpnk", A, A.conj(), optimize=True)

        A_list[t] = A
        S_list[t] = S.reshape(DIM_M * DIM_M, DIM_M * DIM_M)

    return A_list, S_list


def memory_zero_density():
    zero_m = np.zeros(DIM_M, dtype=complex)
    zero_m[0] = 1.0
    return np.outer(zero_m, zero_m.conj())


def apply_memory_channel(S: np.ndarray, rho_m: np.ndarray):
    out = (S @ rho_m.reshape(-1)).reshape(DIM_M, DIM_M)
    # Numerical hygiene only; physics is unchanged.
    out = 0.5 * (out + out.conj().T)
    tr = np.trace(out).real
    if abs(tr - 1.0) > 1e-10:
        out = out / tr
    return out


def final_reduced_states(A: np.ndarray, rho_before: np.ndarray):
    """Return rho_I(out) and rho_M(out) for the final step."""
    rho_i = np.einsum(
        "amn,nk,bmk->ab",
        A,
        rho_before,
        A.conj(),
        optimize=True,
    )
    rho_m = np.einsum(
        "amn,nk,apk->mp",
        A,
        rho_before,
        A.conj(),
        optimize=True,
    )

    rho_i = 0.5 * (rho_i + rho_i.conj().T)
    rho_m = 0.5 * (rho_m + rho_m.conj().T)
    return rho_i, rho_m


def expectation(rho: np.ndarray, op: np.ndarray) -> float:
    val = np.trace(rho @ op)
    return float(np.real_if_close(val, tol=1000).real)


def extract_single_qubit_observables(rho_i, rho_m):
    row = {}
    for name, op in INJECTION_OPS.items():
        row[name] = expectation(rho_i, op)
    for name, op in MEMORY_OPS.items():
        row[name] = expectation(rho_m, op)
    return row


def build_window_bank(A_list, S_list, end_indices, W):
    """
    Rewound protocol for one W.

    For each prediction ending at chronological index j:
        rho_M <- |00><00|
        replay inputs j-W+1,...,j
        after every non-final step keep only rho_M
        after final step measure rho_I(out) and rho_M(out)
    """
    rows = []
    max_trace_err = 0.0
    max_herm_err = 0.0

    rho0 = memory_zero_density()

    for j in end_indices:
        start = j - W + 1
        if start < 0:
            raise ValueError(f"Window W={W} ending at index {j} has no history.")

        rho_m = rho0.copy()

        # Replay all but the final input, carrying only memory.
        for k in range(start, j):
            rho_m = apply_memory_channel(S_list[k], rho_m)

        # Final input: retain both reduced subsystems for measurement.
        rho_i_out, rho_m_out = final_reduced_states(A_list[j], rho_m)

        max_trace_err = max(
            max_trace_err,
            abs(np.trace(rho_i_out).real - 1.0),
            abs(np.trace(rho_m_out).real - 1.0),
        )
        max_herm_err = max(
            max_herm_err,
            float(np.linalg.norm(rho_i_out - rho_i_out.conj().T, ord="fro")),
            float(np.linalg.norm(rho_m_out - rho_m_out.conj().T, ord="fro")),
        )

        rows.append(extract_single_qubit_observables(rho_i_out, rho_m_out))

    bank = pd.DataFrame(rows)
    return bank, max_trace_err, max_herm_err


def build_cont_bank(A_list):
    """
    Continuous-memory protocol (CONT).

    Start once with rho_M=|00><00| at the beginning of the available
    chronology, then carry the memory state through every input without
    resetting between predictions.  The train->2025 boundary is therefore
    continuous as in Week 6.
    """
    rows = []
    max_trace_err = 0.0
    max_herm_err = 0.0
    rho_m = memory_zero_density()

    for A in A_list:
        rho_i_out, rho_m_out = final_reduced_states(A, rho_m)

        max_trace_err = max(
            max_trace_err,
            abs(np.trace(rho_i_out).real - 1.0),
            abs(np.trace(rho_m_out).real - 1.0),
        )
        max_herm_err = max(
            max_herm_err,
            float(np.linalg.norm(rho_i_out - rho_i_out.conj().T, ord="fro")),
            float(np.linalg.norm(rho_m_out - rho_m_out.conj().T, ord="fro")),
        )

        rows.append(extract_single_qubit_observables(rho_i_out, rho_m_out))
        rho_m = rho_m_out

    return pd.DataFrame(rows), max_trace_err, max_herm_err


# =============================================================================
# Ridge readout
# =============================================================================

def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


def fit_predict_ridge(X_train, y_train, X_eval, alpha):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xev = scaler.transform(X_eval)

    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(Xtr, y_train)
    return model.predict(Xev)


def chronological_cv(X, y):
    splitter = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    rows = []

    for ridge_alpha in RIDGE_GRID:
        fold_scores = []

        for tr_idx, va_idx in splitter.split(X):
            pred = fit_predict_ridge(
                X[tr_idx], y[tr_idx], X[va_idx], ridge_alpha
            )
            fold_scores.append(rmse(y[va_idx], pred))

        rows.append(
            {
                "ridge_alpha": float(ridge_alpha),
                "cv_rmse_mean": float(np.mean(fold_scores)),
                "cv_rmse_std": float(np.std(fold_scores, ddof=1)),
                "fold_rmse": fold_scores,
            }
        )

    best = min(rows, key=lambda r: r["cv_rmse_mean"])
    return best, rows


def matrix_rank_scaled(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return int(np.linalg.matrix_rank(Xs))


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 116)
    print("WEEK 7 - STEP 7.1 (REVISED)")
    print("OBSERVABLE-INFORMATION STUDY ACROSS RWP WINDOWS + CONTINUOUS MEMORY")
    print("=" * 116)
    print()
    print("Frozen:")
    print("  F4")
    print("  4 injection + 2 memory qubits")
    print("  softsign policy mapping")
    print(f"  alpha = {ALPHA}")
    print(f"  hx    = {HX}")
    print(f"  dt    = {DT}")
    print(f"  r     = {TROTTER_R}")
    print(f"  RWP windows = {WINDOWS}")
    print("  CONT = memory carried through the full chronological sequence")
    print(f"  seeds = {SEEDS}")
    print("  2025 = diagnostic only")
    print("  2026 = untouched")
    print()

    work, train, val, cols = load_data()
    n_train = len(train)
    n_val = len(val)

    angles_all = make_input_angles(work, cols)
    y_train_full = train[cols["target"]].to_numpy(dtype=float)
    y_val = val[cols["target"]].to_numpy(dtype=float)

    protocol_specs = [("CONT", None)] + [
        (f"RWP_W{W}", W) for W in WINDOWS
    ]

    seed_rows = []
    cv_rows = []
    obs_rows = []
    audit_rows = []
    physics_rows = []

    for seed in SEEDS:
        print("#" * 116)
        print(f"RESERVOIR SEED = {seed}")
        print("#" * 116)

        J = seed_couplings(seed)
        for edge in EDGES:
            print(f"  J{edge} = {J[edge]:+.6f}")

        U, unitary_err = build_trotter_unitary(seed)
        print(f"Unitarity error = {unitary_err:.3e}")
        print("Precomputing input-conditioned memory channels ...")
        A_list, S_list = build_input_channels(U, angles_all)

        # CONT is generated once over the complete train->validation chronology.
        cont_bank_all, cont_trace_err, cont_herm_err = build_cont_bank(A_list)

        for protocol, W in protocol_specs:
            print()
            print("-" * 116)
            print(f"PROTOCOL = {protocol}")
            print("-" * 116)

            if protocol == "CONT":
                bank_train = cont_bank_all.iloc[:n_train].reset_index(drop=True)
                bank_val = cont_bank_all.iloc[n_train:n_train + n_val].reset_index(drop=True)
                y_train = y_train_full
                tr_err = cont_trace_err
                herm_err = cont_herm_err
                w_numeric = np.nan
            else:
                train_end_indices = np.arange(W - 1, n_train, dtype=int)
                val_end_indices = np.arange(n_train, n_train + n_val, dtype=int)

                y_train = y_train_full[W - 1:]
                bank_train, tr_err_train, herm_err_train = build_window_bank(
                    A_list, S_list, train_end_indices, W
                )
                bank_val, tr_err_val, herm_err_val = build_window_bank(
                    A_list, S_list, val_end_indices, W
                )
                tr_err = max(tr_err_train, tr_err_val)
                herm_err = max(herm_err_train, herm_err_val)
                w_numeric = W

            assert len(bank_train) == len(y_train)
            assert len(bank_val) == len(y_val)

            physics_rows.append(
                {
                    "seed": seed,
                    "protocol": protocol,
                    "W": w_numeric,
                    "unitarity_error": unitary_err,
                    "max_trace_error": tr_err,
                    "max_hermiticity_error": herm_err,
                    "n_train": len(y_train),
                    "n_validation": len(y_val),
                }
            )

            print(
                f"Samples: train={len(y_train)}, validation={len(y_val)} | "
                f"trace_err={tr_err:.3e} | herm_err={herm_err:.3e}"
            )

            for col in ALL_OBSERVABLES:
                x = bank_train[col].to_numpy(dtype=float)
                x_std = float(np.std(x, ddof=0))
                x_range = float(np.max(x) - np.min(x))

                if x_std > 1e-14 and np.std(y_train) > 1e-14:
                    corr = float(np.corrcoef(x, y_train)[0, 1])
                else:
                    corr = np.nan

                obs_rows.append(
                    {
                        "seed": seed,
                        "protocol": protocol,
                        "W": w_numeric,
                        "observable": col,
                        "pauli": col[0],
                        "qubit": int(col[1:]),
                        "is_memory": int(col[1:]) in MEMORY_QUBITS,
                        "train_mean": float(np.mean(x)),
                        "train_std": x_std,
                        "train_range": x_range,
                        "pearson_target": corr,
                        "abs_pearson_target": (
                            abs(corr) if np.isfinite(corr) else np.nan
                        ),
                    }
                )

            protocol_family_results = {}

            for family_name, feature_names in FAMILIES.items():
                X_train = bank_train[feature_names].to_numpy(dtype=float)
                X_val = bank_val[feature_names].to_numpy(dtype=float)

                best, all_cv = chronological_cv(X_train, y_train)
                alpha_best = best["ridge_alpha"]
                cv_mean = best["cv_rmse_mean"]
                cv_std = best["cv_rmse_std"]

                pred_val = fit_predict_ridge(
                    X_train, y_train, X_val, alpha_best
                )

                val_rmse = rmse(y_val, pred_val)
                val_mae = float(mean_absolute_error(y_val, pred_val))
                val_bias = float(np.mean(pred_val - y_val))
                rank = matrix_rank_scaled(X_train)

                row = {
                    "seed": seed,
                    "protocol": protocol,
                    "W": w_numeric,
                    "family": family_name,
                    "n_features": len(feature_names),
                    "feature_names": ",".join(feature_names),
                    "ridge_alpha": alpha_best,
                    "cv_rmse": cv_mean,
                    "cv_rmse_fold_std": cv_std,
                    "validation_rmse": val_rmse,
                    "validation_mae": val_mae,
                    "validation_bias": val_bias,
                    "rank": rank,
                    "n_train": len(y_train),
                    "n_validation": len(y_val),
                    "unitarity_error": unitary_err,
                }
                seed_rows.append(row)
                protocol_family_results[family_name] = row

                for r in all_cv:
                    cv_rows.append(
                        {
                            "seed": seed,
                            "protocol": protocol,
                            "W": w_numeric,
                            "family": family_name,
                            "ridge_alpha": r["ridge_alpha"],
                            "cv_rmse_mean": r["cv_rmse_mean"],
                            "cv_rmse_std": r["cv_rmse_std"],
                            **{
                                f"fold_{k+1}_rmse": v
                                for k, v in enumerate(r["fold_rmse"])
                            },
                        }
                    )

                print(
                    f"{family_name:18s} "
                    f"d={len(feature_names):2d} "
                    f"rank={rank:2d}/{len(feature_names):2d} "
                    f"CV={cv_mean:.6f} (+/- {cv_std:.6f}) "
                    f"lambda={alpha_best:<8g} "
                    f"Val={val_rmse:.6f}"
                )

            # Full Week-6 temporal-protocol reproduction audit using XZ_all.
            got = protocol_family_results["XZ_all"]
            ref = WEEK6_XZ_REF[seed][protocol]
            delta_cv = got["cv_rmse"] - ref["cv"]
            delta_val = got["validation_rmse"] - ref["val"]
            passed = (
                abs(delta_cv) <= AUDIT_TOL
                and abs(delta_val) <= AUDIT_TOL
            )

            audit_rows.append(
                {
                    "seed": seed,
                    "protocol": protocol,
                    "W": w_numeric,
                    "week6_ref_cv": ref["cv"],
                    "week7_reproduced_cv": got["cv_rmse"],
                    "delta_cv": delta_cv,
                    "week6_ref_val": ref["val"],
                    "week7_reproduced_val": got["validation_rmse"],
                    "delta_val": delta_val,
                    "pass": passed,
                }
            )

            print(
                f"Week-6 XZ audit: {'PASS' if passed else 'FAIL'} | "
                f"delta CV={delta_cv:+.6f}, delta Val={delta_val:+.6f}"
            )

    seed_df = pd.DataFrame(seed_rows)
    cv_df = pd.DataFrame(cv_rows)
    obs_df = pd.DataFrame(obs_rows)
    audit_df = pd.DataFrame(audit_rows)
    physics_df = pd.DataFrame(physics_rows)

    seed_df.to_csv(RESULTS / "07_01_observable_protocol_seed_results.csv", index=False)
    cv_df.to_csv(RESULTS / "07_01_observable_protocol_cv_grid.csv", index=False)
    obs_df.to_csv(RESULTS / "07_01_single_observable_protocol_stats.csv", index=False)
    audit_df.to_csv(RESULTS / "07_01_week6_temporal_reproduction_audit.csv", index=False)
    physics_df.to_csv(RESULTS / "07_01_protocol_physics_audit.csv", index=False)

    print()
    print("=" * 116)
    print("WEEK-6 TEMPORAL-PROTOCOL REPRODUCTION AUDIT")
    print("=" * 116)
    print(audit_df.to_string(index=False))
    print()

    if not audit_df["pass"].all():
        print("STOP.")
        print("At least one CONT/RWP XZ result does not reproduce Week 6.")
        print("Do not interpret the new observable-family comparisons yet.")
        return

    # Compare every family against XZ within the SAME seed and SAME protocol.
    ref = (
        seed_df[seed_df["family"] == "XZ_all"]
        .set_index(["seed", "protocol"])[["cv_rmse", "validation_rmse"]]
        .rename(
            columns={
                "cv_rmse": "xz_cv_rmse",
                "validation_rmse": "xz_validation_rmse",
            }
        )
    )

    comp = seed_df.join(ref, on=["seed", "protocol"])
    comp["delta_cv_vs_XZ"] = comp["cv_rmse"] - comp["xz_cv_rmse"]
    comp["delta_val_vs_XZ"] = (
        comp["validation_rmse"] - comp["xz_validation_rmse"]
    )

    aggregate = (
        comp.groupby(["protocol", "family"], as_index=False, sort=False)
        .agg(
            W=("W", "first"),
            n_features=("n_features", "first"),
            mean_rank=("rank", "mean"),
            mean_cv_rmse=("cv_rmse", "mean"),
            seed_std_cv_rmse=("cv_rmse", "std"),
            mean_delta_cv_vs_XZ=("delta_cv_vs_XZ", "mean"),
            wins_vs_XZ_cv=("delta_cv_vs_XZ", lambda s: int((s < 0).sum())),
            mean_validation_rmse=("validation_rmse", "mean"),
            seed_std_validation_rmse=("validation_rmse", "std"),
            mean_delta_val_vs_XZ=("delta_val_vs_XZ", "mean"),
            wins_vs_XZ_val=("delta_val_vs_XZ", lambda s: int((s < 0).sum())),
        )
        .reset_index(drop=True)
    )

    protocol_order = ["CONT"] + [f"RWP_W{W}" for W in WINDOWS]
    order_map = {p: i for i, p in enumerate(protocol_order)}
    aggregate["_order"] = aggregate["protocol"].map(order_map)
    aggregate = aggregate.sort_values(["_order", "mean_cv_rmse"]).drop(columns="_order")
    aggregate["rank_within_protocol"] = (
        aggregate.groupby("protocol")["mean_cv_rmse"]
        .rank(method="first")
        .astype(int)
    )

    aggregate.to_csv(
        RESULTS / "07_01_observable_protocol_aggregate.csv",
        index=False,
    )

    cv_matrix = aggregate.pivot(
        index="family", columns="protocol", values="mean_cv_rmse"
    ).reindex(columns=protocol_order)
    cv_matrix.to_csv(RESULTS / "07_01_cv_rmse_family_by_protocol.csv")

    pauli_location = (
        obs_df.groupby(["protocol", "pauli", "is_memory"], as_index=False, sort=False)
        .agg(
            mean_train_std=("train_std", "mean"),
            mean_abs_target_corr=("abs_pearson_target", "mean"),
            max_abs_target_corr=("abs_pearson_target", "max"),
        )
    )
    pauli_location["_order"] = pauli_location["protocol"].map(order_map)
    pauli_location = pauli_location.sort_values(
        ["_order", "pauli", "is_memory"]
    ).drop(columns="_order")
    pauli_location.to_csv(
        RESULTS / "07_01_pauli_location_by_protocol.csv",
        index=False,
    )

    winners = (
        aggregate.sort_values(["protocol", "mean_cv_rmse"])
        .groupby("protocol", as_index=False)
        .first()
    )
    winners["_order"] = winners["protocol"].map(order_map)
    winners = winners.sort_values("_order").drop(columns="_order")
    winners.to_csv(RESULTS / "07_01_winner_by_protocol.csv", index=False)

    print("=" * 116)
    print("MEAN CV RMSE: OBSERVABLE FAMILY x TEMPORAL PROTOCOL")
    print("Primary evidence = 2022-2024 chronological CV; averaged over reservoir seeds")
    print("=" * 116)
    print(cv_matrix.to_string(float_format=lambda x: f"{x:.6f}"))
    print()

    print("=" * 116)
    print("WINNER WITHIN EACH TEMPORAL PROTOCOL")
    print("=" * 116)
    print(
        winners[
            [
                "protocol",
                "family",
                "n_features",
                "mean_rank",
                "mean_cv_rmse",
                "seed_std_cv_rmse",
                "mean_validation_rmse",
            ]
        ].to_string(index=False)
    )
    print()

    summary_path = RESULTS / "07_01_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("WEEK 7 - STEP 7.1 (REVISED)\n")
        f.write("Observable-information study across RWP windows + CONT\n\n")
        f.write(f"RWP windows: {WINDOWS}\n")
        f.write("CONT: continuous memory over full chronology\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("F4 and all reservoir hyperparameters frozen.\n")
        f.write("2025 diagnostic only; 2026 untouched.\n\n")
        f.write("Week-6 temporal-protocol reproduction audit:\n")
        f.write(audit_df.to_string(index=False))
        f.write("\n\nMean CV RMSE matrix:\n")
        f.write(cv_matrix.to_string())
        f.write("\n\nWinner by protocol:\n")
        f.write(winners.to_string(index=False))
        f.write("\n")

    print("Saved:")
    for p in [
        RESULTS / "07_01_observable_protocol_seed_results.csv",
        RESULTS / "07_01_observable_protocol_cv_grid.csv",
        RESULTS / "07_01_single_observable_protocol_stats.csv",
        RESULTS / "07_01_week6_temporal_reproduction_audit.csv",
        RESULTS / "07_01_protocol_physics_audit.csv",
        RESULTS / "07_01_observable_protocol_aggregate.csv",
        RESULTS / "07_01_cv_rmse_family_by_protocol.csv",
        RESULTS / "07_01_pauli_location_by_protocol.csv",
        RESULTS / "07_01_winner_by_protocol.csv",
        RESULTS / "07_01_summary.txt",
    ]:
        print(f"  {p}")

    print()
    print("Corrected Step 7.1 computation finished.")
    print("Interpret observables jointly across CONT and all RWP windows before Step 7.2.")


if __name__ == "__main__":
    main()
