from pathlib import Path

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    SparsePauliOp,
    partial_trace,
)


# ============================================================
# Configuration
# ============================================================

N_QUBITS = 3

Q_INPUT = 0
Q_MEMORY = [1, 2]

THETA_SEQUENCE = [
    0.2,
    -0.7,
    1.0,
    0.4,
]

J01 = 0.7
J12 = -0.4

H_X = 0.5

DELTA_T = 0.8


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    RESULTS_DIR
    / "05_02c_observable_correctness.txt"
)


# ============================================================
# Logging
# ============================================================

output_lines = []


def log(text=""):
    text = str(text)

    print(text)

    output_lines.append(text)


# ============================================================
# Pauli matrices
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

Y = np.array(
    [
        [0, -1j],
        [1j, 0],
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

PAULI = {
    "I": I2,
    "X": X,
    "Y": Y,
    "Z": Z,
}


# ============================================================
# Build full Pauli operator
# ============================================================

def build_pauli_operator(assignments):
    """
    assignments example:

        {0: "Z", 1: "X"}

    means

        Z on q0
        X on q1
        I on q2.

    Qiskit basis ordering is:

        |q2 q1 q0>

    Therefore the Kronecker-product order is

        q2 kron q1 kron q0.
    """

    matrices = []

    for q in reversed(
        range(N_QUBITS)
    ):
        label = assignments.get(
            q,
            "I",
        )

        matrices.append(
            PAULI[label]
        )

    full_operator = matrices[0]

    for matrix in matrices[1:]:
        full_operator = np.kron(
            full_operator,
            matrix,
        )

    return full_operator


# ============================================================
# Observable expectation value
# ============================================================

def expectation_value(
    rho,
    assignments,
):
    """
    Calculate <O> in two independent ways:

    1. manually using Tr(rho O)
    2. Qiskit DensityMatrix.expectation_value()

    Return both values and their difference.
    """

    O_matrix = build_pauli_operator(
        assignments
    )

    # --------------------------------------------------------
    # Manual calculation
    # --------------------------------------------------------

    manual = np.trace(
        rho.data
        @ O_matrix
    )

    # --------------------------------------------------------
    # Qiskit calculation
    # --------------------------------------------------------

    qiskit_value = (
        rho.expectation_value(
            Operator(O_matrix)
        )
    )

    difference = abs(
        manual - qiskit_value
    )

    return (
        manual,
        qiskit_value,
        float(difference),
    )


# ============================================================
# Input state
# ============================================================

def prepare_input_density_matrix(theta):

    qc = QuantumCircuit(1)

    qc.ry(
        theta,
        0,
    )

    return DensityMatrix(qc)


# ============================================================
# Initial memory
# ============================================================

def initial_memory_state():

    data = np.zeros(
        (4, 4),
        dtype=complex,
    )

    data[0, 0] = 1.0

    return DensityMatrix(
        data,
        dims=(2, 2),
    )


# ============================================================
# Combine fresh input + memory
# ============================================================

def combine_input_and_memory(
    rho_input,
    rho_memory,
):
    """
    Numerical ordering:

        q2 q1 q0

    Therefore:

        memory kron input
    """

    full_data = np.kron(
        rho_memory.data,
        rho_input.data,
    )

    return DensityMatrix(
        full_data,
        dims=(2, 2, 2),
    )


# ============================================================
# Hamiltonian
# ============================================================

def build_hamiltonian():

    terms = [
        ("ZZ", [0, 1], J01),
        ("ZZ", [1, 2], J12),

        ("X", [0], H_X),
        ("X", [1], H_X),
        ("X", [2], H_X),
    ]

    return SparsePauliOp.from_sparse_list(
        terms,
        num_qubits=N_QUBITS,
    )


# ============================================================
# Exact unitary
# ============================================================

def build_exact_unitary():

    H = build_hamiltonian()

    H_matrix = np.asarray(
        H.to_matrix(),
        dtype=complex,
    )

    return expm(
        -1j
        * H_matrix
        * DELTA_T
    )


# ============================================================
# Evolution
# ============================================================

def evolve(
    rho_pre,
    U,
):

    evolved = (
        U
        @ rho_pre.data
        @ U.conj().T
    )

    return DensityMatrix(
        evolved,
        dims=(2, 2, 2),
    )


# ============================================================
# Observable definitions
# ============================================================

REFERENCE_FEATURES = [
    ("X0", {0: "X"}),
    ("Z0", {0: "Z"}),

    ("X1", {1: "X"}),
    ("Z1", {1: "Z"}),

    ("X2", {2: "X"}),
    ("Z2", {2: "Z"}),
]


DIAGNOSTIC_OBSERVABLES = [
    ("Y1", {1: "Y"}),
    ("Y2", {2: "Y"}),

    ("Z0Z1", {
        0: "Z",
        1: "Z",
    }),

    ("Z1Z2", {
        1: "Z",
        2: "Z",
    }),

    ("Z0X1", {
        0: "Z",
        1: "X",
    }),

    ("Y1X2", {
        1: "Y",
        2: "X",
    }),
]


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 80)
    log("WEEK 5 - STEP 5.2C")
    log("RESERVOIR OBSERVABLE / FEATURE EXTRACTION CORRECTNESS")
    log("=" * 80)
    log()

    log("Reference feature vector:")
    log(
        "  r_t = "
        "[X0, Z0, X1, Z1, X2, Z2]"
    )
    log()

    log("Diagnostic observables:")
    log(
        "  Y1, Y2, "
        "Z0Z1, Z1Z2, "
        "Z0X1, Y1X2"
    )
    log()

    U = build_exact_unitary()

    rho_memory = (
        initial_memory_state()
    )

    feature_rows = []

    max_manual_qiskit_error = 0.0
    max_imaginary_part = 0.0
    max_bound_violation = 0.0

    # ========================================================
    # Temporal processing
    # ========================================================

    for t, theta in enumerate(
        THETA_SEQUENCE,
        start=1,
    ):

        log("-" * 80)
        log(
            f"TIME STEP t={t}, "
            f"theta={theta:+.3f}"
        )
        log("-" * 80)

        # ----------------------------------------------------
        # Fresh input
        # ----------------------------------------------------

        rho_input = (
            prepare_input_density_matrix(
                theta
            )
        )

        # ----------------------------------------------------
        # Combine with persistent memory
        # ----------------------------------------------------

        rho_pre = (
            combine_input_and_memory(
                rho_input,
                rho_memory,
            )
        )

        # ----------------------------------------------------
        # Reservoir evolution
        # ----------------------------------------------------

        rho_global = evolve(
            rho_pre,
            U,
        )

        # ----------------------------------------------------
        # Reference QRC features
        # ----------------------------------------------------

        feature_values = []

        log()
        log("Reference features:")

        for (
            name,
            assignments,
        ) in REFERENCE_FEATURES:

            (
                manual,
                qiskit_value,
                difference,
            ) = expectation_value(
                rho_global,
                assignments,
            )

            real_value = float(
                np.real(manual)
            )

            imag_value = abs(
                float(
                    np.imag(manual)
                )
            )

            bound_violation = max(
                0.0,
                abs(real_value) - 1.0,
            )

            max_manual_qiskit_error = max(
                max_manual_qiskit_error,
                difference,
            )

            max_imaginary_part = max(
                max_imaginary_part,
                imag_value,
            )

            max_bound_violation = max(
                max_bound_violation,
                bound_violation,
            )

            feature_values.append(
                real_value
            )

            log(
                f"  {name:<5s} "
                f"= {real_value:+.12f}    "
                f"manual-vs-Qiskit error "
                f"= {difference:.3e}"
            )

        feature_rows.append(
            feature_values
        )

        # ----------------------------------------------------
        # Diagnostic observables
        # ----------------------------------------------------

        log()
        log("Diagnostic observables:")

        for (
            name,
            assignments,
        ) in DIAGNOSTIC_OBSERVABLES:

            (
                manual,
                qiskit_value,
                difference,
            ) = expectation_value(
                rho_global,
                assignments,
            )

            real_value = float(
                np.real(manual)
            )

            imag_value = abs(
                float(
                    np.imag(manual)
                )
            )

            bound_violation = max(
                0.0,
                abs(real_value) - 1.0,
            )

            max_manual_qiskit_error = max(
                max_manual_qiskit_error,
                difference,
            )

            max_imaginary_part = max(
                max_imaginary_part,
                imag_value,
            )

            max_bound_violation = max(
                max_bound_violation,
                bound_violation,
            )

            log(
                f"  {name:<6s} "
                f"= {real_value:+.12f}    "
                f"manual-vs-Qiskit error "
                f"= {difference:.3e}"
            )

        # ----------------------------------------------------
        # Recurrent update
        # ----------------------------------------------------

        rho_memory = partial_trace(
            rho_global,
            [Q_INPUT],
        )

        log()

    # ========================================================
    # QRC feature matrix
    # ========================================================

    X_qrc = np.asarray(
        feature_rows,
        dtype=float,
    )

    log("=" * 80)
    log("QRC FEATURE MATRIX")
    log("=" * 80)
    log()

    log(
        "Columns:"
    )

    log(
        "  "
        + ", ".join(
            name
            for (
                name,
                _
            ) in REFERENCE_FEATURES
        )
    )

    log()

    log(
        np.array2string(
            X_qrc,
            precision=6,
            suppress_small=True,
        )
    )

    log()

    log(
        f"Feature matrix shape = "
        f"{X_qrc.shape}"
    )

    # ========================================================
    # Final correctness summary
    # ========================================================

    log()
    log("=" * 80)
    log("CORRECTNESS SUMMARY")
    log("=" * 80)

    log(
        f"Maximum manual-vs-Qiskit expectation error "
        f"= {max_manual_qiskit_error:.12e}"
    )

    log(
        f"Maximum imaginary component "
        f"= {max_imaginary_part:.12e}"
    )

    log(
        f"Maximum [-1,+1] bound violation "
        f"= {max_bound_violation:.12e}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()