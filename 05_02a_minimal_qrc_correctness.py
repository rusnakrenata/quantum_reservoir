from pathlib import Path

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    SparsePauliOp,
    partial_trace,
    purity,
    state_fidelity,
)


# ============================================================
# Configuration
# ============================================================

N_QUBITS = 3

Q_INPUT = 0
Q_MEMORY = [1, 2]

THETA_INPUT = 0.6

J01 = 0.7
J12 = -0.4

H_X = 0.5
H_Y = 0.3

DELTA_T = 0.8

TROTTER_STEPS = [1, 2, 4, 8]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "05_02a_minimal_qrc_correctness.txt"


# ============================================================
# Logging
# ============================================================

output_lines = []


def log(text=""):
    text = str(text)
    print(text)
    output_lines.append(text)


# ============================================================
# Utility diagnostics
# ============================================================

def density_matrix_diagnostics(rho, name):
    """
    Check basic physical properties of a density matrix.
    """

    data = np.asarray(rho.data, dtype=complex)

    trace_value = np.trace(data)

    hermiticity_error = np.linalg.norm(
        data - data.conj().T,
        ord="fro",
    )

    # Symmetrize only for numerically robust eigenvalue calculation.
    hermitian_data = (data + data.conj().T) / 2.0

    eigenvalues = np.linalg.eigvalsh(hermitian_data)

    min_eigenvalue = float(np.min(eigenvalues))

    purity_value = float(np.real(purity(rho)))

    log(f"{name}:")
    log(f"  dimension             = {data.shape}")
    log(f"  trace                 = {trace_value}")
    log(f"  Hermiticity error     = {hermiticity_error:.12e}")
    log(f"  minimum eigenvalue    = {min_eigenvalue:.12e}")
    log(f"  purity                = {purity_value:.12f}")
    log()

    return {
        "trace": trace_value,
        "hermiticity_error": hermiticity_error,
        "min_eigenvalue": min_eigenvalue,
        "purity": purity_value,
    }


# ============================================================
# Input-state preparation
# ============================================================

def prepare_initial_state():
    """
    q0 = injection qubit
    q1, q2 = memory qubits

    q0 receives Ry(theta_input).
    q1, q2 remain in |00>.
    """

    qc = QuantumCircuit(N_QUBITS)

    qc.ry(THETA_INPUT, Q_INPUT)

    rho = DensityMatrix(qc)

    return qc, rho


# ============================================================
# Hamiltonian
# ============================================================

def build_hamiltonian(eta_y):
    """
    H =
        J01 Z0 Z1
      + J12 Z1 Z2
      + h_x (X0 + X1 + X2)
      + eta_y h_y (Y1 + Y2)

    SparsePauliOp.from_sparse_list allows us to specify
    the physical qubit indices explicitly.
    """

    terms = [
        ("ZZ", [0, 1], J01),
        ("ZZ", [1, 2], J12),

        ("X", [0], H_X),
        ("X", [1], H_X),
        ("X", [2], H_X),
    ]

    if eta_y == 1:
        terms.extend(
            [
                ("Y", [1], H_Y),
                ("Y", [2], H_Y),
            ]
        )

    H = SparsePauliOp.from_sparse_list(
        terms,
        num_qubits=N_QUBITS,
    )

    return H


# ============================================================
# Exact evolution
# ============================================================

def exact_evolution(rho_pre, H):
    """
    Calculate

        U_exact = exp(-i H Delta_t)

    directly as an 8 x 8 matrix.
    """

    H_matrix = np.asarray(H.to_matrix(), dtype=complex)

    # Hamiltonian must be Hermitian.
    h_hermiticity_error = np.linalg.norm(
        H_matrix - H_matrix.conj().T,
        ord="fro",
    )

    U_exact = expm(
        -1j * H_matrix * DELTA_T
    )

    identity = np.eye(
        2 ** N_QUBITS,
        dtype=complex,
    )

    unitarity_error = np.linalg.norm(
        U_exact.conj().T @ U_exact - identity,
        ord="fro",
    )

    rho_exact_data = (
        U_exact
        @ rho_pre.data
        @ U_exact.conj().T
    )

    rho_exact = DensityMatrix(rho_exact_data)

    return (
        H_matrix,
        U_exact,
        rho_exact,
        h_hermiticity_error,
        unitarity_error,
    )


# ============================================================
# First-order Trotter circuit
# ============================================================

def build_trotter_circuit(r, eta_y):
    """
    First-order Trotterization.

    Circuit order inside each slice:

        ZZ -> X -> optional Y

    Therefore the corresponding matrix product for one
    slice is:

        U_slice = U_Y U_X U_ZZ

    because the rightmost matrix acts first.
    """

    qc = QuantumCircuit(N_QUBITS)

    dt_slice = DELTA_T / r

    for _ in range(r):

        # ----------------------------------------------------
        # ZZ interaction block
        # ----------------------------------------------------

        qc.rzz(
            2.0 * J01 * dt_slice,
            0,
            1,
        )

        qc.rzz(
            2.0 * J12 * dt_slice,
            1,
            2,
        )

        # ----------------------------------------------------
        # X-field block
        # ----------------------------------------------------

        for q in range(N_QUBITS):
            qc.rx(
                2.0 * H_X * dt_slice,
                q,
            )

        # ----------------------------------------------------
        # Optional memory-only Y-field block
        # ----------------------------------------------------

        if eta_y == 1:
            for q in Q_MEMORY:
                qc.ry(
                    2.0 * H_Y * dt_slice,
                    q,
                )

    return qc


# ============================================================
# Trotter evolution
# ============================================================

def trotter_evolution(rho_pre, r, eta_y):
    qc = build_trotter_circuit(
        r=r,
        eta_y=eta_y,
    )

    U_trotter = np.asarray(
        Operator(qc).data,
        dtype=complex,
    )

    rho_trotter_data = (
        U_trotter
        @ rho_pre.data
        @ U_trotter.conj().T
    )

    rho_trotter = DensityMatrix(
        rho_trotter_data
    )

    return qc, U_trotter, rho_trotter


# ============================================================
# Run one Hamiltonian variant
# ============================================================

def run_variant(rho_pre, eta_y):

    log("=" * 72)

    if eta_y == 0:
        log("HAMILTONIAN VARIANT: baseline ZZ + X")
    else:
        log("HAMILTONIAN VARIANT: extended ZZ + X + Y_memory")

    log("=" * 72)
    log()

    H = build_hamiltonian(eta_y)

    log("Hamiltonian:")
    log(H)
    log()

    (
        H_matrix,
        U_exact,
        rho_exact,
        h_error,
        u_error,
    ) = exact_evolution(
        rho_pre=rho_pre,
        H=H,
    )

    log("Hamiltonian / exact-unitary checks:")
    log(f"  ||H - H^dagger||_F       = {h_error:.12e}")
    log(f"  ||U^dagger U - I||_F     = {u_error:.12e}")
    log()

    density_matrix_diagnostics(
        rho_exact,
        "Exact global state after reservoir evolution",
    )

    # --------------------------------------------------------
    # Partial trace over injection qubit q0
    # --------------------------------------------------------

    rho_memory_exact = partial_trace(
        rho_exact,
        [Q_INPUT],
    )

    density_matrix_diagnostics(
        rho_memory_exact,
        "Exact reduced MEMORY state Tr_input(rho)",
    )

    log("Interpretation of memory purity:")
    log("  purity = 1   -> pure memory state")
    log("  purity < 1   -> memory is mixed after tracing out input")
    log()

    # --------------------------------------------------------
    # Trotter convergence
    # --------------------------------------------------------

    log("Trotter convergence:")
    log(
        "r    ||U_exact-U_trot||_F    "
        "global fidelity    memory fidelity"
    )

    previous_operator_error = None

    for r in TROTTER_STEPS:

        (
            qc_trotter,
            U_trotter,
            rho_trotter,
        ) = trotter_evolution(
            rho_pre=rho_pre,
            r=r,
            eta_y=eta_y,
        )

        rho_memory_trotter = partial_trace(
            rho_trotter,
            [Q_INPUT],
        )

        operator_error = np.linalg.norm(
            U_exact - U_trotter,
            ord="fro",
        )

        global_fidelity = float(
            np.real(
                state_fidelity(
                    rho_exact,
                    rho_trotter,
                )
            )
        )

        memory_fidelity = float(
            np.real(
                state_fidelity(
                    rho_memory_exact,
                    rho_memory_trotter,
                )
            )
        )

        log(
            f"{r:<4d} "
            f"{operator_error:<24.12e} "
            f"{global_fidelity:<18.12f} "
            f"{memory_fidelity:.12f}"
        )

        if previous_operator_error is not None:
            if operator_error > previous_operator_error + 1e-10:
                log(
                    "  WARNING: operator error increased "
                    "relative to previous r."
                )

        previous_operator_error = operator_error

    log()


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 72)
    log("WEEK 5 - STEP 5.2A")
    log("MINIMAL IDEAL-QRC CORRECTNESS TEST")
    log("=" * 72)
    log()

    log("Logical architecture:")
    log("  q0 = injection qubit")
    log("  q1 = memory qubit")
    log("  q2 = memory qubit")
    log()

    log("Fixed test parameters:")
    log(f"  theta_input = {THETA_INPUT}")
    log(f"  J01         = {J01}")
    log(f"  J12         = {J12}")
    log(f"  h_x         = {H_X}")
    log(f"  h_y         = {H_Y}")
    log(f"  Delta_t     = {DELTA_T}")
    log()

    # --------------------------------------------------------
    # Prepare input + initial memory
    # --------------------------------------------------------

    prep_circuit, rho_pre = prepare_initial_state()

    log("Input-preparation circuit:")
    log(prep_circuit.draw(output="text"))
    log()

    expected_a0 = np.cos(
        THETA_INPUT / 2.0
    )

    expected_a1 = np.sin(
        THETA_INPUT / 2.0
    )

    log("Expected injection amplitudes:")
    log(
        f"  cos(theta/2) = {expected_a0:.12f}"
    )
    log(
        f"  sin(theta/2) = {expected_a1:.12f}"
    )
    log()

    density_matrix_diagnostics(
        rho_pre,
        "Initial global state",
    )

    # --------------------------------------------------------
    # Verify initial memory is |00><00|
    # --------------------------------------------------------

    rho_memory_initial = partial_trace(
        rho_pre,
        [Q_INPUT],
    )

    density_matrix_diagnostics(
        rho_memory_initial,
        "Initial memory state",
    )

    log("Initial memory density matrix:")
    log(
        np.array2string(
            rho_memory_initial.data,
            precision=6,
            suppress_small=True,
        )
    )
    log()

    # --------------------------------------------------------
    # Baseline and extended Hamiltonians
    # --------------------------------------------------------

    run_variant(
        rho_pre=rho_pre,
        eta_y=0,
    )

    run_variant(
        rho_pre=rho_pre,
        eta_y=1,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_FILE.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print()
    print(f"Saved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()