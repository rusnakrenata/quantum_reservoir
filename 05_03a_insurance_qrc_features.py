from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RESULTS_DIR / "03_01_preprocessed_samples.csv"

FEATURE_OUTPUT = RESULTS_DIR / "05_03a_qrc_features.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "05_03a_qrc_features_summary.txt"
COUPLING_OUTPUT = RESULTS_DIR / "05_03a_reservoir_couplings.csv"


# ============================================================
# Frozen logical architecture
# ============================================================

N_INPUT = 4
N_MEMORY = 2
N_QUBITS = N_INPUT + N_MEMORY

Q_CLAIM = 0
Q_WEEKDAY = 1
Q_POLICY = 2
Q_HOLIDAY = 3

Q_MEMORY = [4, 5]

# Chain topology:
#
# q0 -- q1 -- q2 -- q3 -- q4 -- q5
#
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]


# ============================================================
# Reference hyperparameters
#
# IMPORTANT:
# These are NOT tuned values.
# Week 6 will search them systematically.
# ============================================================

ALPHA = 1.0

J_SCALE = 0.7

H_X = 0.5

DELTA_T = 0.8

TROTTER_STEPS = 2

RESERVOIR_SEED = 42


# ============================================================
# Required frozen Week-3 columns
# ============================================================

REQUIRED_COLUMNS = [
    "input_date",
    "target_date",
    "split",

    "C_t_z",
    "D_sin",
    "D_cos",
    "P_t_z",

    "is_public_holiday_t_plus_1",
]


# ============================================================
# Logging
# ============================================================

output_lines = []


def log(text=""):
    text = str(text)
    print(text)
    output_lines.append(text)


# ============================================================
# Single-qubit Ry density matrix
# ============================================================

def ry_density(theta):
    """
    Ry(theta)|0> =
        cos(theta/2)|0>
        +
        sin(theta/2)|1>

    The corresponding pure-state density matrix is

        [[c^2, cs],
         [cs,  s^2]]
    """

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    return np.array(
        [
            [c * c, c * s],
            [c * s, s * s],
        ],
        dtype=complex,
    )


# ============================================================
# Quantum input encoding
# ============================================================

def encode_input_row(row):
    """
    F4 semantic encoding:

        q0 = C_t
        q1 = D_{t+1}
        q2 = P_t
        q3 = H_{t+1}

    Continuous variables:
        z -> clip[-3,3] -> /3 -> Ry(alpha * value)

    Weekday:
        recover angle from existing D_sin, D_cos

    Holiday:
        0 -> Ry(0)
        1 -> Ry(pi)
    """

    # --------------------------------------------------------
    # Claim count
    # --------------------------------------------------------

    c_scaled = (
        np.clip(
            float(row["C_t_z"]),
            -3.0,
            3.0,
        )
        / 3.0
    )

    theta_c = ALPHA * c_scaled

    # --------------------------------------------------------
    # Weekday
    # --------------------------------------------------------

    d_sin = float(row["D_sin"])
    d_cos = float(row["D_cos"])

    theta_d = np.arctan2(
        d_sin,
        d_cos,
    )

    # Put angle into [0, 2pi).
    theta_d = np.mod(
        theta_d,
        2.0 * np.pi,
    )

    # --------------------------------------------------------
    # Policy count
    # --------------------------------------------------------

    p_scaled = (
        np.clip(
            float(row["P_t_z"]),
            -3.0,
            3.0,
        )
        / 3.0
    )

    theta_p = ALPHA * p_scaled

    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    holiday = float(
        row["is_public_holiday_t_plus_1"]
    )

    theta_h = np.pi * holiday

    angles = {
        "theta_C": theta_c,
        "theta_D": theta_d,
        "theta_P": theta_p,
        "theta_H": theta_h,
    }

    # --------------------------------------------------------
    # Qiskit subsystem ordering
    #
    # For q0,q1,q2,q3 the matrix order is
    #
    # q3 kron q2 kron q1 kron q0.
    # --------------------------------------------------------

    rho_q0 = ry_density(theta_c)
    rho_q1 = ry_density(theta_d)
    rho_q2 = ry_density(theta_p)
    rho_q3 = ry_density(theta_h)

    rho_input = np.kron(
        rho_q3,
        np.kron(
            rho_q2,
            np.kron(
                rho_q1,
                rho_q0,
            ),
        ),
    )

    return rho_input, angles


# ============================================================
# Initial memory state |00><00|
# ============================================================

def initial_memory_density():
    rho = np.zeros(
        (2 ** N_MEMORY, 2 ** N_MEMORY),
        dtype=complex,
    )

    rho[0, 0] = 1.0

    return rho


# ============================================================
# Fixed reservoir couplings
# ============================================================

def generate_couplings():
    """
    Fixed random J_ij values.

    They are generated ONCE and then remain fixed
    throughout the entire 1692-sample sequence.
    """

    rng = np.random.default_rng(
        RESERVOIR_SEED
    )

    couplings = {}

    for edge in EDGES:
        couplings[edge] = float(
            rng.uniform(
                -J_SCALE,
                J_SCALE,
            )
        )

    return couplings


# ============================================================
# Build reference Trotter unitary
# ============================================================

def build_trotter_unitary(couplings):
    """
    Hamiltonian:

        H =
          sum_(i,j) J_ij Zi Zj
          +
          h_x sum_i Xi

    First-order circuit order:

        ZZ -> X

    repeated TROTTER_STEPS times.
    """

    qc = QuantumCircuit(
        N_QUBITS
    )

    dt_slice = (
        DELTA_T
        / TROTTER_STEPS
    )

    for _ in range(
        TROTTER_STEPS
    ):

        # ----------------------------------------------------
        # ZZ interactions
        # ----------------------------------------------------

        for (
            q_i,
            q_j,
        ), J_ij in couplings.items():

            qc.rzz(
                2.0
                * J_ij
                * dt_slice,
                q_i,
                q_j,
            )

        # ----------------------------------------------------
        # Global transverse X field
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):
            qc.rx(
                2.0
                * H_X
                * dt_slice,
                q,
            )

    U = np.asarray(
        Operator(qc).data,
        dtype=complex,
    )

    return qc, U


# ============================================================
# Physical / numerical checks
# ============================================================

def density_trace(rho):
    return np.trace(rho)


def density_purity(rho):
    return float(
        np.real(
            np.trace(
                rho @ rho
            )
        )
    )


def hermiticity_error(rho):
    return float(
        np.linalg.norm(
            rho - rho.conj().T,
            ord="fro",
        )
    )


# ============================================================
# Fast partial trace over ALL injection qubits
# ============================================================

def trace_out_input(rho_global):
    """
    Matrix ordering:

        memory kron input

    dimensions:

        d_M = 4
        d_I = 16

    Reshape

        rho[a,i,b,j]

    where

        a,b = memory indices
        i,j = input indices.

    Partial trace over input:

        rho_M[a,b]
        =
        sum_i rho[a,i,b,i]
    """

    d_m = 2 ** N_MEMORY
    d_i = 2 ** N_INPUT

    reshaped = rho_global.reshape(
        d_m,
        d_i,
        d_m,
        d_i,
    )

    rho_memory = np.einsum(
        "aibi->ab",
        reshaped,
    )

    return rho_memory


# ============================================================
# Build Pauli operators for reference observables
# ============================================================

I2 = np.array(
    [
        [1, 0],
        [0, 1],
    ],
    dtype=complex,
)

X = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=complex,
)

Z = np.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=complex,
)


def single_qubit_operator(
    target_q,
    pauli_matrix,
):
    """
    Build full 6-qubit operator.

    Matrix ordering is:

        q5 kron q4 kron ... kron q0.
    """

    matrices = []

    for q in reversed(
        range(N_QUBITS)
    ):
        if q == target_q:
            matrices.append(
                pauli_matrix
            )
        else:
            matrices.append(
                I2
            )

    operator = matrices[0]

    for matrix in matrices[1:]:
        operator = np.kron(
            operator,
            matrix,
        )

    return operator


def build_observables():
    observables = []

    for q in range(
        N_QUBITS
    ):
        observables.append(
            (
                f"X{q}",
                single_qubit_operator(
                    q,
                    X,
                ),
            )
        )

        observables.append(
            (
                f"Z{q}",
                single_qubit_operator(
                    q,
                    Z,
                ),
            )
        )

    return observables


# ============================================================
# Fast expectation calculation
# ============================================================

def expectation_value(
    rho,
    operator,
):
    """
    Tr(rho O)
      =
    sum_ij rho_ij O_ji

    Using elementwise multiplication avoids a full
    matrix-matrix multiplication.
    """

    value = np.sum(
        rho
        * operator.T
    )

    return value


# ============================================================
# Resolve target column
# ============================================================

def resolve_target_column(df):
    candidates = [
        "property_damage_claim_count_t_plus_1",
        "property_damage_claim_count_target",
        "target_property_damage_claim_count",
        "target_claim_count",
        "C_t_plus_1",
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not resolve target column. "
        "Available columns are:\n"
        + "\n".join(df.columns)
    )


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 5 - STEP 5.3A")
    log("REAL INSURANCE IDEAL-QRC FEATURE GENERATION")
    log("=" * 80)
    log()

    # --------------------------------------------------------
    # Load frozen Week-3 preprocessing
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    target_column = (
        resolve_target_column(df)
    )

    # --------------------------------------------------------
    # Dates / chronology
    # --------------------------------------------------------

    df["input_date"] = pd.to_datetime(
        df["input_date"]
    )

    df["target_date"] = pd.to_datetime(
        df["target_date"]
    )

    df = (
        df
        .sort_values("target_date")
        .reset_index(drop=True)
    )

    if not df["target_date"].is_monotonic_increasing:
        raise ValueError(
            "Target dates are not chronological."
        )

    # --------------------------------------------------------
    # Expected sample count
    # --------------------------------------------------------

    if len(df) != 1692:
        raise ValueError(
            f"Expected 1692 samples, found {len(df)}."
        )

    split_counts = (
        df["split"]
        .value_counts()
        .to_dict()
    )

    log(
        f"Loaded samples       = {len(df)}"
    )

    log(
        f"Target column        = {target_column}"
    )

    log(
        f"First target date    = "
        f"{df['target_date'].min().date()}"
    )

    log(
        f"Last target date     = "
        f"{df['target_date'].max().date()}"
    )

    log(
        f"Split counts         = {split_counts}"
    )

    log()

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    log("Reference architecture:")
    log(
        f"  F4 injection qubits = {N_INPUT}"
    )

    log(
        f"  memory qubits       = {N_MEMORY}"
    )

    log(
        f"  total qubits        = {N_QUBITS}"
    )

    log(
        f"  topology            = {EDGES}"
    )

    log(
        f"  alpha               = {ALPHA}"
    )

    log(
        f"  h_x                 = {H_X}"
    )

    log(
        f"  Delta_t             = {DELTA_T}"
    )

    log(
        f"  Trotter steps       = {TROTTER_STEPS}"
    )

    log(
        f"  reservoir seed      = {RESERVOIR_SEED}"
    )

    log(
        "  Y_memory            = OFF"
    )

    log()

    # --------------------------------------------------------
    # Fixed random reservoir
    # --------------------------------------------------------

    couplings = (
        generate_couplings()
    )

    log("Fixed J_ij couplings:")

    coupling_rows = []

    for (
        q_i,
        q_j,
    ), value in couplings.items():

        log(
            f"  J({q_i},{q_j}) = "
            f"{value:+.12f}"
        )

        coupling_rows.append(
            {
                "q_i": q_i,
                "q_j": q_j,
                "J_ij": value,
            }
        )

    pd.DataFrame(
        coupling_rows
    ).to_csv(
        COUPLING_OUTPUT,
        index=False,
    )

    log()

    # --------------------------------------------------------
    # Build fixed reservoir unitary ONCE
    # --------------------------------------------------------

    qc_reservoir, U = (
        build_trotter_unitary(
            couplings
        )
    )

    identity = np.eye(
        2 ** N_QUBITS,
        dtype=complex,
    )

    unitarity_error = float(
        np.linalg.norm(
            U.conj().T @ U
            - identity,
            ord="fro",
        )
    )

    log(
        "Reservoir-unitary check:"
    )

    log(
        f"  ||U^dagger U - I||_F "
        f"= {unitarity_error:.12e}"
    )

    log()

    # --------------------------------------------------------
    # Reference observables
    # --------------------------------------------------------

    observables = (
        build_observables()
    )

    feature_names = [
        name
        for name, _
        in observables
    ]

    log(
        "Reference observables:"
    )

    log(
        "  "
        + ", ".join(
            feature_names
        )
    )

    log()

    # --------------------------------------------------------
    # Initial persistent memory
    # --------------------------------------------------------

    rho_memory = (
        initial_memory_density()
    )

    # --------------------------------------------------------
    # Sequential QRC processing
    # --------------------------------------------------------

    feature_rows = []

    memory_purities = []

    max_global_trace_error = 0.0
    max_memory_trace_error = 0.0
    max_global_hermiticity = 0.0
    max_memory_hermiticity = 0.0

    max_feature_imaginary = 0.0
    max_feature_bound_violation = 0.0

    clipped_claim_count = 0
    clipped_policy_count = 0

    for index, row in df.iterrows():

        # ====================================================
        # Encode current F4 input
        # ====================================================

        rho_input, angles = (
            encode_input_row(row)
        )

        if abs(
            float(row["C_t_z"])
        ) > 3.0:
            clipped_claim_count += 1

        if abs(
            float(row["P_t_z"])
        ) > 3.0:
            clipped_policy_count += 1

        # ====================================================
        # Fresh input + persistent previous memory
        #
        # Qiskit matrix ordering:
        #
        # memory kron input
        # ====================================================

        rho_pre = np.kron(
            rho_memory,
            rho_input,
        )

        # ====================================================
        # Fixed reservoir evolution
        # ====================================================

        rho_global = (
            U
            @ rho_pre
            @ U.conj().T
        )

        # ====================================================
        # Physical diagnostics
        # ====================================================

        global_trace_error = abs(
            np.trace(rho_global)
            - 1.0
        )

        global_herm_error = (
            hermiticity_error(
                rho_global
            )
        )

        max_global_trace_error = max(
            max_global_trace_error,
            float(global_trace_error),
        )

        max_global_hermiticity = max(
            max_global_hermiticity,
            global_herm_error,
        )

        # ====================================================
        # Reservoir features
        # ====================================================

        features = {}

        for (
            name,
            operator,
        ) in observables:

            value = expectation_value(
                rho_global,
                operator,
            )

            real_value = float(
                np.real(value)
            )

            imaginary_value = abs(
                float(
                    np.imag(value)
                )
            )

            bound_violation = max(
                0.0,
                abs(real_value) - 1.0,
            )

            max_feature_imaginary = max(
                max_feature_imaginary,
                imaginary_value,
            )

            max_feature_bound_violation = max(
                max_feature_bound_violation,
                bound_violation,
            )

            features[name] = (
                real_value
            )

        # ====================================================
        # Carry memory to next day
        # ====================================================

        rho_memory = trace_out_input(
            rho_global
        )

        memory_trace_error = abs(
            np.trace(rho_memory)
            - 1.0
        )

        memory_herm_error = (
            hermiticity_error(
                rho_memory
            )
        )

        max_memory_trace_error = max(
            max_memory_trace_error,
            float(memory_trace_error),
        )

        max_memory_hermiticity = max(
            max_memory_hermiticity,
            memory_herm_error,
        )

        memory_purity = (
            density_purity(
                rho_memory
            )
        )

        memory_purities.append(
            memory_purity
        )

        # ====================================================
        # Output row
        # ====================================================

        output_row = {
            "input_date": (
                row["input_date"]
            ),

            "target_date": (
                row["target_date"]
            ),

            "split": (
                row["split"]
            ),

            target_column: (
                row[target_column]
            ),

            "theta_C": (
                angles["theta_C"]
            ),

            "theta_D": (
                angles["theta_D"]
            ),

            "theta_P": (
                angles["theta_P"]
            ),

            "theta_H": (
                angles["theta_H"]
            ),

            "memory_purity": (
                memory_purity
            ),
        }

        output_row.update(
            features
        )

        feature_rows.append(
            output_row
        )

        # Simple progress markers.
        if (
            (index + 1) % 250 == 0
            or
            index == len(df) - 1
        ):
            log(
                f"Processed "
                f"{index + 1:4d}/{len(df)} "
                f"samples..."
            )

    # --------------------------------------------------------
    # Save full feature table
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        feature_rows
    )

    feature_df.to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Audits
    # --------------------------------------------------------

    X_qrc = (
        feature_df[
            feature_names
        ]
        .to_numpy(
            dtype=float
        )
    )

    finite_ok = bool(
        np.isfinite(
            X_qrc
        ).all()
    )

    expected_shape = (
        1692,
        2 * N_QUBITS,
    )

    shape_ok = (
        X_qrc.shape
        == expected_shape
    )

    # --------------------------------------------------------
    # Split feature ranges
    # --------------------------------------------------------

    train_mask = (
        feature_df["split"]
        .astype(str)
        .str.lower()
        .eq("train")
    )

    val_mask = (
        feature_df["split"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "validation",
                "val",
            ]
        )
    )

    test_mask = (
        feature_df["split"]
        .astype(str)
        .str.lower()
        .eq("test")
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    log()
    log("=" * 80)
    log("REAL-DATA QRC FEATURE SUMMARY")
    log("=" * 80)

    log(
        f"Feature matrix shape       = "
        f"{X_qrc.shape}"
    )

    log(
        f"Expected shape             = "
        f"{expected_shape}"
    )

    log(
        f"Shape check                = "
        f"{'PASS' if shape_ok else 'FAIL'}"
    )

    log(
        f"All features finite        = "
        f"{'PASS' if finite_ok else 'FAIL'}"
    )

    log()

    log(
        f"Claim observations clipped = "
        f"{clipped_claim_count}"
    )

    log(
        f"Policy observations clipped= "
        f"{clipped_policy_count}"
    )

    log()

    log(
        f"Maximum global trace error = "
        f"{max_global_trace_error:.12e}"
    )

    log(
        f"Maximum memory trace error = "
        f"{max_memory_trace_error:.12e}"
    )

    log(
        f"Maximum global Hermiticity = "
        f"{max_global_hermiticity:.12e}"
    )

    log(
        f"Maximum memory Hermiticity = "
        f"{max_memory_hermiticity:.12e}"
    )

    log()

    log(
        f"Maximum feature imaginary  = "
        f"{max_feature_imaginary:.12e}"
    )

    log(
        f"Maximum bound violation    = "
        f"{max_feature_bound_violation:.12e}"
    )

    log()

    log(
        f"Initial memory purity      = "
        f"{memory_purities[0]:.12f}"
    )

    log(
        f"Minimum memory purity      = "
        f"{np.min(memory_purities):.12f}"
    )

    log(
        f"Final memory purity        = "
        f"{memory_purities[-1]:.12f}"
    )

    log()

    if train_mask.any():
        log(
            f"Train QRC feature range    = "
            f"[{X_qrc[train_mask].min():+.6f}, "
            f"{X_qrc[train_mask].max():+.6f}]"
        )

    if val_mask.any():
        log(
            f"Validation feature range   = "
            f"[{X_qrc[val_mask].min():+.6f}, "
            f"{X_qrc[val_mask].max():+.6f}]"
        )

    if test_mask.any():
        log(
            f"Test QRC feature range     = "
            f"[{X_qrc[test_mask].min():+.6f}, "
            f"{X_qrc[test_mask].max():+.6f}]"
        )

    log()

    log(
        f"Saved feature table to:"
    )
    log(
        f"  {FEATURE_OUTPUT}"
    )

    log(
        f"Saved couplings to:"
    )
    log(
        f"  {COUPLING_OUTPUT}"
    )

    # --------------------------------------------------------
    # Hard failure if core checks fail
    # --------------------------------------------------------

    if not shape_ok:
        raise RuntimeError(
            "QRC feature matrix has wrong shape."
        )

    if not finite_ok:
        raise RuntimeError(
            "QRC feature matrix contains "
            "NaN or infinite values."
        )

    if (
        max_feature_bound_violation
        > 1e-10
    ):
        raise RuntimeError(
            "Pauli expectation values "
            "violate [-1,+1]."
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    SUMMARY_OUTPUT.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()