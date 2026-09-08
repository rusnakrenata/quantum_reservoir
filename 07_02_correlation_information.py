"""
Week 7 - Step 7.2
Two-qubit Pauli-correlation information study across RWP windows + CONT.

Purpose
-------
Step 7.1 studied only one-body Pauli observables.  Step 7.2 asks whether
selected two-body Pauli products contain incremental forecasting information.

This is still an IDEAL expectation-value study.  We do NOT yet simulate
ancilla/parity/Hadamard-test circuits, finite shots, back-action, or hardware
noise.  Those measurement implementations belong to later Week-7 substeps.

Scientific rule
---------------
Keep the complete Week-6 ideal reservoir and temporal protocols frozen:
    F4, 4 injection + 2 memory qubits, softsign policy mapping,
    alpha=0.005, hx=0.5, dt=0.8, r=2,
    seeds=[42,101,202,505,707],
    CONT and RWP(W), W in {1,2,5,7,14,21,28}.

Compare multiple observable families; DO NOT freeze one family in this step.
Primary evidence is 2022-2024 chronological CV.  2025 is diagnostic only.
2026 remains untouched.

Correlation convention
----------------------
For an ordered chain edge (i,j) with i<j:
    ZX_ij = < Z_i X_j >
    XZ_ij = < X_i Z_j >
    YX_ij = < Y_i X_j >
    ZZ_ij = < Z_i Z_j >

We test nearest-neighbour chain correlations because they are physically
aligned with the frozen reservoir coupling graph and remain hardware-relevant.
We also keep explicit memory-focused families involving edge (4,5) and the
memory-touching edges (3,4),(4,5).
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
DIM_I = 2**N_INJECTION
DIM_M = 2**N_MEMORY
MEMORY_QUBITS = (4, 5)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
MEMORY_TOUCHING_EDGES = [(3, 4), (4, 5)]
MEMORY_INTERNAL_EDGE = (4, 5)

ALPHA = 0.005
HX = 0.5
DT = 0.8
TROTTER_R = 2
SEEDS = [42, 101, 202, 505, 707]
WINDOWS = [1, 2, 5, 7, 14, 21, 28]

RIDGE_GRID = np.array(
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0],
    dtype=float,
)
N_CV_SPLITS = 5

# Frozen Week-6 r=2 X+Z references, used only to verify that Step 7.2 has
# exactly the same reservoir/temporal semantics as Week 6 and Step 7.1.
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
PAULI = {"X": X2, "Y": Y2, "Z": Z2}
I64 = np.eye(2**N_QUBITS, dtype=complex)


def kron_all(mats):
    return reduce(np.kron, mats)


def local_op(pauli: np.ndarray, q_local: int, n_qubits: int) -> np.ndarray:
    mats = [I2] * n_qubits
    mats[q_local] = pauli
    return kron_all(mats)


def local_pair_op(
    pauli_a: np.ndarray,
    q_a_local: int,
    pauli_b: np.ndarray,
    q_b_local: int,
    n_qubits: int,
) -> np.ndarray:
    mats = [I2] * n_qubits
    mats[q_a_local] = pauli_a
    mats[q_b_local] = pauli_b
    return kron_all(mats)


def pauli_on_full_qubit(pauli: np.ndarray, q: int) -> np.ndarray:
    mats = [I2] * N_QUBITS
    mats[q] = pauli
    return kron_all(mats)


def pauli_product_full(pauli_a, q_a, pauli_b, q_b):
    mats = [I2] * N_QUBITS
    mats[q_a] = pauli_a
    mats[q_b] = pauli_b
    return kron_all(mats)


FULL_SINGLE_OPS = {
    f"{p}{q}": pauli_on_full_qubit(PAULI[p], q)
    for p in ("X", "Y", "Z")
    for q in range(N_QUBITS)
}

ZZ_HAMILTONIAN_OPS = {
    edge: pauli_product_full(Z2, edge[0], Z2, edge[1])
    for edge in EDGES
}

INJECTION_SINGLE_OPS = {
    f"{p}{q}": local_op(PAULI[p], q, N_INJECTION)
    for p in ("X", "Y", "Z")
    for q in range(N_INJECTION)
}

MEMORY_SINGLE_OPS = {
    f"{p}{q_global}": local_op(
        PAULI[p], q_global - N_INJECTION, N_MEMORY
    )
    for p in ("X", "Y", "Z")
    for q_global in MEMORY_QUBITS
}


# =============================================================================
# Correlation observable registry
# =============================================================================

PAIR_TYPES = {
    "ZZ": ("Z", "Z"),
    "ZX": ("Z", "X"),
    "XZ": ("X", "Z"),
    "YX": ("Y", "X"),
}


def pair_name(kind: str, edge) -> str:
    i, j = edge
    return f"{kind}_{i}{j}"


CHAIN_CORR = {
    kind: [pair_name(kind, edge) for edge in EDGES]
    for kind in PAIR_TYPES
}
MEMORY_TOUCHING_CORR = {
    kind: [pair_name(kind, edge) for edge in MEMORY_TOUCHING_EDGES]
    for kind in PAIR_TYPES
}

# Prebuild the reduced operators needed for each pair observable.
PAIR_REGISTRY = {}
for kind, (pa, pb) in PAIR_TYPES.items():
    for i, j in EDGES:
        name = pair_name(kind, (i, j))
        A = PAULI[pa]
        B = PAULI[pb]

        if j < N_INJECTION:
            # Pair is entirely within injection subsystem.
            PAIR_REGISTRY[name] = {
                "location": "injection",
                "edge": (i, j),
                "kind": kind,
                "op_i": local_pair_op(A, i, B, j, N_INJECTION),
                "op_m": None,
            }
        elif i >= N_INJECTION:
            # Pair is entirely within memory subsystem: only edge (4,5).
            PAIR_REGISTRY[name] = {
                "location": "memory",
                "edge": (i, j),
                "kind": kind,
                "op_i": None,
                "op_m": local_pair_op(
                    A, i - N_INJECTION,
                    B, j - N_INJECTION,
                    N_MEMORY,
                ),
            }
        else:
            # Boundary edge (3,4): one operator on injection, one on memory.
            PAIR_REGISTRY[name] = {
                "location": "boundary",
                "edge": (i, j),
                "kind": kind,
                "op_i": local_op(A, i, N_INJECTION),
                "op_m": local_op(B, j - N_INJECTION, N_MEMORY),
            }


# =============================================================================
# Observable families
# =============================================================================

XZ_ALL = (
    [f"X{q}" for q in range(N_QUBITS)]
    + [f"Z{q}" for q in range(N_QUBITS)]
)
XZ_INJECTION = (
    [f"X{q}" for q in range(N_INJECTION)]
    + [f"Z{q}" for q in range(N_INJECTION)]
)

ALL_CHAIN_CORR = (
    CHAIN_CORR["ZZ"]
    + CHAIN_CORR["ZX"]
    + CHAIN_CORR["XZ"]
    + CHAIN_CORR["YX"]
)

# The explicitly discussed hardware-motivated memory-pair readout:
# retain simple X/Z measurements on injection qubits and enrich them with
# correlations on q4-q5 that could later be estimated ancilla-assistively.
MEM45_XZ_YX = ["XZ_45", "YX_45"]

# Slightly broader memory-focused set: include both the injection-memory
# boundary (3,4) and the memory-memory edge (4,5).
MEM_TOUCH_XZ_YX = (
    MEMORY_TOUCHING_CORR["XZ"]
    + MEMORY_TOUCHING_CORR["YX"]
)

FAMILIES = {
    # Step-7.1 references retained for direct incremental comparison.
    "XZ_injection": XZ_INJECTION,
    "XZ_all": XZ_ALL,

    # Add one correlation type at a time.
    "XZall_plus_ZZchain": XZ_ALL + CHAIN_CORR["ZZ"],
    "XZall_plus_ZXchain": XZ_ALL + CHAIN_CORR["ZX"],
    "XZall_plus_XZcorr_chain": XZ_ALL + CHAIN_CORR["XZ"],
    "XZall_plus_YXchain": XZ_ALL + CHAIN_CORR["YX"],

    # Planned mixed families.
    "XZall_plus_ZX_YX": XZ_ALL + CHAIN_CORR["ZX"] + CHAIN_CORR["YX"],
    "XZall_plus_all_corr": XZ_ALL + ALL_CHAIN_CORR,

    # Hardware-motivated memory-focused candidates discussed earlier.
    "XZinj_plus_mem45_XZ_YX": XZ_INJECTION + MEM45_XZ_YX,
    "XZinj_plus_memtouch_XZ_YX": XZ_INJECTION + MEM_TOUCH_XZ_YX,

    # Diagnostic: can two-body information stand on its own?
    "all_chain_corr_only": ALL_CHAIN_CORR,
}

ALL_FEATURES = sorted(
    {f for fs in FAMILIES.values() for f in fs}
)
PAIR_FEATURES = sorted(PAIR_REGISTRY.keys())


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
            ZZ = ZZ_HAMILTONIAN_OPS[edge]
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
# Efficient temporal-memory dynamics
# =============================================================================


def build_input_channels(U: np.ndarray, angles: np.ndarray):
    U4 = U.reshape(DIM_I, DIM_M, DIM_I, DIM_M)

    A_list = np.empty((len(angles), DIM_I, DIM_M, DIM_M), dtype=complex)
    S_list = np.empty(
        (len(angles), DIM_M * DIM_M, DIM_M * DIM_M),
        dtype=complex,
    )

    for t, row in enumerate(angles):
        phi = injection_vector(row)
        A = np.einsum("ambn,b->amn", U4, phi, optimize=True)
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
    out = 0.5 * (out + out.conj().T)
    tr = np.trace(out).real
    if abs(tr - 1.0) > 1e-10:
        out = out / tr
    return out


def final_reduced_states(A: np.ndarray, rho_before: np.ndarray):
    rho_i = np.einsum(
        "amn,nk,bmk->ab",
        A, rho_before, A.conj(),
        optimize=True,
    )
    rho_m = np.einsum(
        "amn,nk,apk->mp",
        A, rho_before, A.conj(),
        optimize=True,
    )
    rho_i = 0.5 * (rho_i + rho_i.conj().T)
    rho_m = 0.5 * (rho_m + rho_m.conj().T)
    return rho_i, rho_m


def expectation(rho: np.ndarray, op: np.ndarray) -> float:
    val = np.trace(rho @ op)
    return float(np.real_if_close(val, tol=1000).real)


def cross_expectation(
    A: np.ndarray,
    rho_before: np.ndarray,
    op_i: np.ndarray,
    op_m: np.ndarray,
) -> float:
    """
    Compute <O_I tensor O_M> on the final 6-qubit state without explicitly
    constructing the 64x64 density matrix.

    rho_out[a,m,b,p] = sum_nk A[a,m,n] rho[n,k] A*[b,p,k]

    Tr(rho_out (O_I tensor O_M)) then gives the contraction below.
    """
    val = np.einsum(
        "amn,nk,bpk,ba,pm->",
        A,
        rho_before,
        A.conj(),
        op_i,
        op_m,
        optimize=True,
    )
    return float(np.real_if_close(val, tol=1000).real)


def extract_feature_row(
    A_final: np.ndarray,
    rho_before: np.ndarray,
    rho_i: np.ndarray,
    rho_m: np.ndarray,
):
    row = {}

    # One-body X/Z references.
    for q in range(N_INJECTION):
        for p in ("X", "Z"):
            name = f"{p}{q}"
            row[name] = expectation(rho_i, INJECTION_SINGLE_OPS[name])

    for q in MEMORY_QUBITS:
        for p in ("X", "Z"):
            name = f"{p}{q}"
            row[name] = expectation(rho_m, MEMORY_SINGLE_OPS[name])

    # Two-body correlations.
    for name, meta in PAIR_REGISTRY.items():
        if meta["location"] == "injection":
            row[name] = expectation(rho_i, meta["op_i"])
        elif meta["location"] == "memory":
            row[name] = expectation(rho_m, meta["op_m"])
        else:
            row[name] = cross_expectation(
                A_final,
                rho_before,
                meta["op_i"],
                meta["op_m"],
            )

    return row


def build_window_bank(A_list, S_list, end_indices, W):
    rows = []
    max_trace_err = 0.0
    max_herm_err = 0.0
    rho0 = memory_zero_density()

    for j in end_indices:
        start = j - W + 1
        if start < 0:
            raise ValueError(f"Window W={W} ending at index {j} has no history.")

        rho_m = rho0.copy()
        for k in range(start, j):
            rho_m = apply_memory_channel(S_list[k], rho_m)

        # rho_m is the state immediately BEFORE the final input.
        rho_before = rho_m
        rho_i_out, rho_m_out = final_reduced_states(A_list[j], rho_before)

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

        rows.append(
            extract_feature_row(
                A_list[j], rho_before, rho_i_out, rho_m_out
            )
        )

    return pd.DataFrame(rows), max_trace_err, max_herm_err


def build_cont_bank(A_list):
    rows = []
    max_trace_err = 0.0
    max_herm_err = 0.0
    rho_m = memory_zero_density()

    for A in A_list:
        rho_before = rho_m
        rho_i_out, rho_m_out = final_reduced_states(A, rho_before)

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

        rows.append(extract_feature_row(A, rho_before, rho_i_out, rho_m_out))
        rho_m = rho_m_out

    return pd.DataFrame(rows), max_trace_err, max_herm_err


# =============================================================================
# Numerical self-check for cross-boundary Pauli expectation
# =============================================================================


def cross_expectation_self_check():
    """Compare tensor contraction with an explicitly built 64x64 state."""
    rng = np.random.default_rng(12345)

    # Random isometry-like A is not required for this algebra check.
    A = rng.normal(size=(DIM_I, DIM_M, DIM_M)) + 1j * rng.normal(
        size=(DIM_I, DIM_M, DIM_M)
    )
    B = rng.normal(size=(DIM_M, DIM_M)) + 1j * rng.normal(
        size=(DIM_M, DIM_M)
    )
    rho = B @ B.conj().T
    rho /= np.trace(rho)

    op_i = local_op(Y2, 3, N_INJECTION)
    op_m = local_op(X2, 0, N_MEMORY)

    fast = cross_expectation(A, rho, op_i, op_m)

    rho4 = np.einsum(
        "amn,nk,bpk->ambp",
        A, rho, A.conj(),
        optimize=True,
    )
    rho_full = rho4.reshape(DIM_I * DIM_M, DIM_I * DIM_M)
    op_full = np.kron(op_i, op_m)
    explicit = float(np.real(np.trace(rho_full @ op_full)))

    err = abs(fast - explicit)
    if err > 1e-9:
        raise RuntimeError(
            f"Cross-correlation contraction self-check failed: error={err:.3e}"
        )
    return err


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

    return min(rows, key=lambda r: r["cv_rmse_mean"]), rows


def matrix_rank_scaled(X):
    Xs = StandardScaler().fit_transform(X)
    return int(np.linalg.matrix_rank(Xs))


# =============================================================================
# Main experiment
# =============================================================================


def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 124)
    print("WEEK 7 - STEP 7.2")
    print("TWO-QUBIT PAULI-CORRELATION INFORMATION STUDY")
    print("=" * 124)
    print()
    print("Frozen reservoir and temporal protocols:")
    print("  F4 | 4 injection + 2 memory | softsign policy")
    print(f"  alpha={ALPHA} | hx={HX} | dt={DT} | r={TROTTER_R}")
    print(f"  seeds={SEEDS}")
    print(f"  RWP windows={WINDOWS} plus CONT")
    print("  2025 diagnostic only; 2026 untouched")
    print()
    print("Correlation convention on ordered edge i<j:")
    print("  ZX_ij=<Z_i X_j>, XZ_ij=<X_i Z_j>, YX_ij=<Y_i X_j>, ZZ_ij=<Z_i Z_j>")
    print("  Exact ideal expectations only; ancilla/shot implementations are NOT simulated yet.")
    print()

    algebra_err = cross_expectation_self_check()
    print(f"Cross-boundary expectation algebra self-check: PASS (error={algebra_err:.3e})")
    print()

    work, train, val, cols = load_data()
    n_train = len(train)
    n_val = len(val)

    angles_all = make_input_angles(work, cols)
    y_train_full = train[cols["target"]].to_numpy(dtype=float)
    y_val = val[cols["target"]].to_numpy(dtype=float)

    protocol_specs = [("CONT", None)] + [(f"RWP_W{W}", W) for W in WINDOWS]

    result_rows = []
    cv_rows = []
    pair_stats_rows = []
    audit_rows = []
    physics_rows = []

    for seed in SEEDS:
        print("#" * 124)
        print(f"RESERVOIR SEED = {seed}")
        print("#" * 124)

        J = seed_couplings(seed)
        for edge in EDGES:
            print(f"  J{edge} = {J[edge]:+.6f}")

        U, unitary_err = build_trotter_unitary(seed)
        print(f"Unitarity error = {unitary_err:.3e}")
        print("Precomputing input-conditioned memory channels ...")
        A_list, S_list = build_input_channels(U, angles_all)

        cont_bank_all, cont_trace_err, cont_herm_err = build_cont_bank(A_list)

        for protocol, W in protocol_specs:
            print()
            print("-" * 124)
            print(f"PROTOCOL = {protocol}")
            print("-" * 124)

            if protocol == "CONT":
                bank_train = cont_bank_all.iloc[:n_train].reset_index(drop=True)
                bank_val = cont_bank_all.iloc[n_train:n_train+n_val].reset_index(drop=True)
                y_train = y_train_full
                tr_err = cont_trace_err
                herm_err = cont_herm_err
                w_numeric = np.nan
            else:
                train_end = np.arange(W - 1, n_train, dtype=int)
                val_end = np.arange(n_train, n_train + n_val, dtype=int)
                y_train = y_train_full[W - 1:]

                bank_train, tr1, he1 = build_window_bank(
                    A_list, S_list, train_end, W
                )
                bank_val, tr2, he2 = build_window_bank(
                    A_list, S_list, val_end, W
                )
                tr_err = max(tr1, tr2)
                herm_err = max(he1, he2)
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

            # Descriptive pair-level information.
            for name in PAIR_FEATURES:
                x = bank_train[name].to_numpy(dtype=float)
                x_std = float(np.std(x, ddof=0))
                if x_std > 1e-14 and np.std(y_train) > 1e-14:
                    corr = float(np.corrcoef(x, y_train)[0, 1])
                else:
                    corr = np.nan

                meta = PAIR_REGISTRY[name]
                pair_stats_rows.append(
                    {
                        "seed": seed,
                        "protocol": protocol,
                        "W": w_numeric,
                        "observable": name,
                        "kind": meta["kind"],
                        "edge": f"{meta['edge'][0]}-{meta['edge'][1]}",
                        "location": meta["location"],
                        "train_mean": float(np.mean(x)),
                        "train_std": x_std,
                        "train_range": float(np.max(x) - np.min(x)),
                        "pearson_target": corr,
                        "abs_pearson_target": abs(corr) if np.isfinite(corr) else np.nan,
                    }
                )

            protocol_results = {}

            for family_name, feature_names in FAMILIES.items():
                X_train = bank_train[feature_names].to_numpy(dtype=float)
                X_val = bank_val[feature_names].to_numpy(dtype=float)

                best, all_cv = chronological_cv(X_train, y_train)
                pred_val = fit_predict_ridge(
                    X_train, y_train, X_val, best["ridge_alpha"]
                )

                row = {
                    "seed": seed,
                    "protocol": protocol,
                    "W": w_numeric,
                    "family": family_name,
                    "n_features": len(feature_names),
                    "feature_names": ",".join(feature_names),
                    "rank": matrix_rank_scaled(X_train),
                    "ridge_alpha": best["ridge_alpha"],
                    "cv_rmse": best["cv_rmse_mean"],
                    "cv_fold_std": best["cv_rmse_std"],
                    "validation_rmse": rmse(y_val, pred_val),
                    "validation_mae": float(mean_absolute_error(y_val, pred_val)),
                    "validation_bias": float(np.mean(pred_val - y_val)),
                }
                result_rows.append(row)
                protocol_results[family_name] = row

                for grid_row in all_cv:
                    cv_rows.append(
                        {
                            "seed": seed,
                            "protocol": protocol,
                            "W": w_numeric,
                            "family": family_name,
                            "ridge_alpha": grid_row["ridge_alpha"],
                            "cv_rmse_mean": grid_row["cv_rmse_mean"],
                            "cv_rmse_std": grid_row["cv_rmse_std"],
                            **{
                                f"fold_{k+1}_rmse": v
                                for k, v in enumerate(grid_row["fold_rmse"])
                            },
                        }
                    )

                print(
                    f"{family_name:30s} d={len(feature_names):2d} "
                    f"rank={row['rank']:2d}/{len(feature_names):2d} "
                    f"CV={row['cv_rmse']:.6f} (+/- {row['cv_fold_std']:.6f}) "
                    f"lambda={row['ridge_alpha']:<8g} "
                    f"Val={row['validation_rmse']:.6f}"
                )

            # Reproduction audit with the unchanged single-body XZ_all family.
            got = protocol_results["XZ_all"]
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

    result_df = pd.DataFrame(result_rows)
    cv_df = pd.DataFrame(cv_rows)
    pair_df = pd.DataFrame(pair_stats_rows)
    audit_df = pd.DataFrame(audit_rows)
    physics_df = pd.DataFrame(physics_rows)

    # Save raw evidence first.
    result_df.to_csv(RESULTS / "07_02_correlation_protocol_seed_results.csv", index=False)
    cv_df.to_csv(RESULTS / "07_02_correlation_protocol_cv_grid.csv", index=False)
    pair_df.to_csv(RESULTS / "07_02_pair_observable_stats.csv", index=False)
    audit_df.to_csv(RESULTS / "07_02_week6_reproduction_audit.csv", index=False)
    physics_df.to_csv(RESULTS / "07_02_protocol_physics_audit.csv", index=False)

    print()
    print("=" * 124)
    print("WEEK-6 REPRODUCTION AUDIT")
    print("=" * 124)
    print(audit_df.to_string(index=False))
    print()

    if not audit_df["pass"].all():
        print("STOP: the unchanged XZ_all family did not reproduce Week 6 everywhere.")
        print("Do not interpret correlation families until the temporal convention is aligned.")
        return

    # -------------------------------------------------------------------------
    # Non-exclusive landscape summaries
    # -------------------------------------------------------------------------
    xz_all_ref = (
        result_df[result_df["family"] == "XZ_all"]
        .set_index(["seed", "protocol"])[["cv_rmse", "validation_rmse"]]
        .rename(columns={
            "cv_rmse": "xzall_cv",
            "validation_rmse": "xzall_val",
        })
    )
    xzinj_ref = (
        result_df[result_df["family"] == "XZ_injection"]
        .set_index(["seed", "protocol"])[["cv_rmse", "validation_rmse"]]
        .rename(columns={
            "cv_rmse": "xzinj_cv",
            "validation_rmse": "xzinj_val",
        })
    )

    comp = result_df.join(xz_all_ref, on=["seed", "protocol"])
    comp = comp.join(xzinj_ref, on=["seed", "protocol"])
    comp["delta_cv_vs_XZall"] = comp["cv_rmse"] - comp["xzall_cv"]
    comp["delta_val_vs_XZall"] = comp["validation_rmse"] - comp["xzall_val"]
    comp["delta_cv_vs_XZinj"] = comp["cv_rmse"] - comp["xzinj_cv"]
    comp["delta_val_vs_XZinj"] = comp["validation_rmse"] - comp["xzinj_val"]

    aggregate = (
        comp.groupby(["protocol", "family"], as_index=False)
        .agg(
            n_features=("n_features", "first"),
            mean_rank=("rank", "mean"),
            mean_cv_rmse=("cv_rmse", "mean"),
            seed_std_cv_rmse=("cv_rmse", "std"),
            mean_delta_cv_vs_XZall=("delta_cv_vs_XZall", "mean"),
            wins_vs_XZall_cv=("delta_cv_vs_XZall", lambda s: int((s < 0).sum())),
            mean_delta_cv_vs_XZinj=("delta_cv_vs_XZinj", "mean"),
            wins_vs_XZinj_cv=("delta_cv_vs_XZinj", lambda s: int((s < 0).sum())),
            mean_validation_rmse=("validation_rmse", "mean"),
            seed_std_validation_rmse=("validation_rmse", "std"),
        )
    )
    aggregate.to_csv(RESULTS / "07_02_correlation_protocol_aggregate.csv", index=False)

    # Family x protocol matrix: useful for seeing whether a correlation family
    # helps only in specific memory regimes rather than globally.
    pivot = aggregate.pivot(
        index="family",
        columns="protocol",
        values="mean_cv_rmse",
    )
    protocol_order = ["CONT"] + [f"RWP_W{W}" for W in WINDOWS]
    pivot = pivot.reindex(columns=protocol_order)
    pivot.to_csv(RESULTS / "07_02_cv_rmse_family_by_protocol.csv")

    # Across all 5 seeds x 8 protocols = 40 cases.  This is deliberately a
    # robustness summary, NOT a winner-selection rule.
    across = (
        comp.groupby("family", as_index=False)
        .agg(
            n_features=("n_features", "first"),
            mean_cv_rmse=("cv_rmse", "mean"),
            std_cv_rmse=("cv_rmse", "std"),
            mean_delta_cv_vs_XZall=("delta_cv_vs_XZall", "mean"),
            median_delta_cv_vs_XZall=("delta_cv_vs_XZall", "median"),
            improves_vs_XZall=("delta_cv_vs_XZall", lambda s: int((s < 0).sum())),
            mean_delta_cv_vs_XZinj=("delta_cv_vs_XZinj", "mean"),
            improves_vs_XZinj=("delta_cv_vs_XZinj", lambda s: int((s < 0).sum())),
            mean_validation_rmse=("validation_rmse", "mean"),
        )
        .sort_values("mean_cv_rmse")
        .reset_index(drop=True)
    )
    across.to_csv(RESULTS / "07_02_family_robustness_across_all_protocols.csv", index=False)

    pair_summary = (
        pair_df.groupby(["kind", "location"], as_index=False)
        .agg(
            mean_train_std=("train_std", "mean"),
            mean_abs_target_corr=("abs_pearson_target", "mean"),
            max_abs_target_corr=("abs_pearson_target", "max"),
        )
        .sort_values(["kind", "location"])
    )
    pair_summary.to_csv(RESULTS / "07_02_pair_type_location_summary.csv", index=False)

    print("=" * 124)
    print("MEAN CV RMSE: CORRELATION FAMILY x TEMPORAL PROTOCOL")
    print("Primary evidence = 2022-2024 chronological CV, averaged over reservoir seeds")
    print("This is an information landscape, not a final setup selection.")
    print("=" * 124)
    print(pivot.to_string())
    print()

    print("=" * 124)
    print("ROBUSTNESS ACROSS ALL 5 SEEDS x 8 TEMPORAL PROTOCOLS = 40 CASES")
    print("Negative delta means lower RMSE than the reference family.")
    print("=" * 124)
    print(across.to_string(index=False))
    print()

    print("=" * 124)
    print("DESCRIPTIVE PAIR-TYPE / LOCATION SUMMARY")
    print("Not used alone for model selection.")
    print("=" * 124)
    print(pair_summary.to_string(index=False))
    print()

    summary_path = RESULTS / "07_02_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("WEEK 7 - STEP 7.2\n")
        f.write("Two-qubit Pauli-correlation information study\n\n")
        f.write("Frozen: F4, alpha=0.005, hx=0.5, dt=0.8, r=2, five seeds\n")
        f.write("Protocols: CONT + RWP W={1,2,5,7,14,21,28}\n")
        f.write("Ideal expectation values only; no ancilla/shots/noise yet.\n\n")
        f.write("Reproduction audit:\n")
        f.write(audit_df.to_string(index=False))
        f.write("\n\nFamily x protocol CV landscape:\n")
        f.write(pivot.to_string())
        f.write("\n\nRobustness across all 40 seed/protocol cases:\n")
        f.write(across.to_string(index=False))
        f.write("\n\nPair-type/location descriptive summary:\n")
        f.write(pair_summary.to_string(index=False))
        f.write("\n")

    print("Saved:")
    for p in [
        RESULTS / "07_02_correlation_protocol_seed_results.csv",
        RESULTS / "07_02_correlation_protocol_cv_grid.csv",
        RESULTS / "07_02_pair_observable_stats.csv",
        RESULTS / "07_02_week6_reproduction_audit.csv",
        RESULTS / "07_02_protocol_physics_audit.csv",
        RESULTS / "07_02_correlation_protocol_aggregate.csv",
        RESULTS / "07_02_cv_rmse_family_by_protocol.csv",
        RESULTS / "07_02_family_robustness_across_all_protocols.csv",
        RESULTS / "07_02_pair_type_location_summary.csv",
        RESULTS / "07_02_summary.txt",
    ]:
        print(f"  {p}")

    print()
    print("Step 7.2 computation finished.")
    print("Interpret correlation information across protocols/seeds before moving on.")
    print("Do NOT freeze one observable setup in this step.")


if __name__ == "__main__":
    main()
