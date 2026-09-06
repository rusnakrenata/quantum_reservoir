from pathlib import Path

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    DensityMatrix,
    SparsePauliOp,
    partial_trace,
    purity,
)


# ============================================================
# Configuration
# ============================================================

N_QUBITS = 3

Q_INPUT = 0
Q_MEMORY = [1, 2]

# Artificial sequence of four consecutive inputs.
THETA_SEQUENCE = [
    0.2,
    -0.7,
    1.0,
    0.4,
]

# Same baseline Hamiltonian as in 5.2A.
J01 = 0.7
J12 = -0.4

H_X = 0.5

DELTA_T = 0.8


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    RESULTS_DIR
    / "05_02b_temporal_memory_correctness.txt"
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
# Basic utilities
# ============================================================

def frobenius_difference(rho_a, rho_b):
    """
    ||rho_a - rho_b||_F
    """

    return float(
        np.linalg.norm(
            rho_a.data - rho_b.data,
            ord="fro",
        )
    )


def density_diagnostics(rho):
    """
    Return basic density-matrix diagnostics.
    """

    data = np.asarray(
        rho.data,
        dtype=complex,
    )

    trace_value = np.trace(data)

    hermiticity_error = np.linalg.norm(
        data - data.conj().T,
        ord="fro",
    )

    purity_value = float(
        np.real(
            purity(rho)
        )
    )

    eigenvalues = np.linalg.eigvalsh(
        (
            data + data.conj().T
        )
        / 2.0
    )

    minimum_eigenvalue = float(
        np.min(eigenvalues)
    )

    return {
        "trace": trace_value,
        "hermiticity_error": hermiticity_error,
        "purity": purity_value,
        "minimum_eigenvalue": minimum_eigenvalue,
    }


# ============================================================
# Input preparation
# ============================================================

def prepare_input_density_matrix(theta):
    """
    Prepare

        rho_I(theta)
        =
        Ry(theta)|0><0|Ry(theta)^dagger

    on a single qubit.
    """

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
    """
    Initial memory:

        |00><00|

    ordered as memory qubits q2 q1.
    """

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
# Hamiltonian
# ============================================================

def build_hamiltonian():
    """
    Baseline Hamiltonian:

        H =
          J01 Z0 Z1
        + J12 Z1 Z2
        + hx (X0 + X1 + X2)
    """

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

    U = expm(
        -1j
        * H_matrix
        * DELTA_T
    )

    return U


# ============================================================
# Construct full state
# ============================================================

def combine_input_and_memory(
    rho_input,
    rho_memory,
):
    """
    IMPORTANT QISKIT ORDERING

    q0 = input
    q1,q2 = memory

    Statevector/density-matrix basis is ordered:

        |q2 q1 q0>

    Therefore numerical tensor-product ordering is

        rho_memory kron rho_input

    so that input occupies subsystem q0.
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
# Exact reservoir evolution
# ============================================================

def evolve_exact(
    rho_pre,
    U,
):
    evolved_data = (
        U
        @ rho_pre.data
        @ U.conj().T
    )

    return DensityMatrix(
        evolved_data,
        dims=(2, 2, 2),
    )


# ============================================================
# One temporal step
# ============================================================

def temporal_step(
    theta,
    rho_memory_previous,
    U,
):
    """
    One recurrent QRC update:

        fresh input
             +
        previous memory
             |
             v
        joint state
             |
             U
             |
             v
        global reservoir state
             |
          Tr_input
             |
             v
        new memory
    """

    # --------------------------------------------------------
    # 1. Fresh input
    # --------------------------------------------------------

    rho_input = prepare_input_density_matrix(
        theta
    )

    # --------------------------------------------------------
    # 2. Combine fresh input with previous memory
    # --------------------------------------------------------

    rho_pre = combine_input_and_memory(
        rho_input=rho_input,
        rho_memory=rho_memory_previous,
    )

    # --------------------------------------------------------
    # 3. Recover input from joint pre-evolution state
    #
    #    Trace out q1 and q2.
    # --------------------------------------------------------

    recovered_input = partial_trace(
        rho_pre,
        Q_MEMORY,
    )

    input_reset_error = frobenius_difference(
        recovered_input,
        rho_input,
    )

    # --------------------------------------------------------
    # 4. Recover memory from pre-evolution state
    #
    #    Trace out q0.
    # --------------------------------------------------------

    recovered_memory = partial_trace(
        rho_pre,
        [Q_INPUT],
    )

    memory_carry_error = frobenius_difference(
        recovered_memory,
        rho_memory_previous,
    )

    # --------------------------------------------------------
    # 5. Reservoir evolution
    # --------------------------------------------------------

    rho_global = evolve_exact(
        rho_pre=rho_pre,
        U=U,
    )

    # --------------------------------------------------------
    # 6. Retain only memory for next time step
    # --------------------------------------------------------

    rho_memory_new = partial_trace(
        rho_global,
        [Q_INPUT],
    )

    return {
        "rho_input": rho_input,
        "rho_pre": rho_pre,
        "rho_global": rho_global,
        "rho_memory_new": rho_memory_new,
        "input_reset_error": input_reset_error,
        "memory_carry_error": memory_carry_error,
    }


# ============================================================
# Main
# ============================================================

def main():

    log("=" * 78)
    log("WEEK 5 - STEP 5.2B")
    log("TEMPORAL MEMORY PROPAGATION / INPUT RESET CORRECTNESS")
    log("=" * 78)
    log()

    log("Architecture:")
    log("  q0 = fresh injection qubit")
    log("  q1 = persistent memory qubit")
    log("  q2 = persistent memory qubit")
    log()

    log("Artificial input sequence:")
    log(
        "  "
        + ", ".join(
            f"{theta:+.3f}"
            for theta in THETA_SEQUENCE
        )
    )
    log()

    log("Hamiltonian:")
    log("  baseline ZZ + X")
    log(f"  J01     = {J01}")
    log(f"  J12     = {J12}")
    log(f"  h_x     = {H_X}")
    log(f"  Delta_t = {DELTA_T}")
    log()

    # --------------------------------------------------------
    # Fixed reservoir evolution
    # --------------------------------------------------------

    U = build_exact_unitary()

    # --------------------------------------------------------
    # Initial recurrent memory
    # --------------------------------------------------------

    rho_memory = initial_memory_state()

    rho_memory_zero = initial_memory_state()

    log("Initial memory:")
    log(
        np.array2string(
            rho_memory.data,
            precision=6,
            suppress_small=True,
        )
    )
    log()

    log("=" * 78)
    log("TEMPORAL PROPAGATION")
    log("=" * 78)
    log()

    log(
        "t   theta     input-reset-error    "
        "memory-carry-error    memory-purity    "
        "||M_recurrent-M_reset||_F"
    )

    for t, theta in enumerate(
        THETA_SEQUENCE,
        start=1,
    ):

        # ====================================================
        # CORRECT recurrent model
        # ====================================================

        recurrent = temporal_step(
            theta=theta,
            rho_memory_previous=rho_memory,
            U=U,
        )

        rho_memory_new = (
            recurrent["rho_memory_new"]
        )

        diagnostics = density_diagnostics(
            rho_memory_new
        )

        # ====================================================
        # WRONG control:
        # reset memory to |00> every time
        # ====================================================

        reset_control = temporal_step(
            theta=theta,
            rho_memory_previous=rho_memory_zero,
            U=U,
        )

        rho_memory_reset = (
            reset_control["rho_memory_new"]
        )

        recurrent_vs_reset = (
            frobenius_difference(
                rho_memory_new,
                rho_memory_reset,
            )
        )

        log(
            f"{t:<3d} "
            f"{theta:+.3f}     "
            f"{recurrent['input_reset_error']:<20.12e} "
            f"{recurrent['memory_carry_error']:<21.12e} "
            f"{diagnostics['purity']:<16.12f} "
            f"{recurrent_vs_reset:.12e}"
        )

        # ====================================================
        # Important:
        #
        # THIS is the recurrence.
        #
        # Today's output memory becomes tomorrow's input memory.
        # ====================================================

        rho_memory = rho_memory_new

    log()
    log("=" * 78)
    log("FINAL RECURRENT MEMORY")
    log("=" * 78)
    log()

    final_diag = density_diagnostics(
        rho_memory
    )

    log(
        np.array2string(
            rho_memory.data,
            precision=6,
            suppress_small=True,
        )
    )

    log()

    log("Final memory diagnostics:")
    log(
        f"  trace              = "
        f"{final_diag['trace']}"
    )

    log(
        f"  Hermiticity error  = "
        f"{final_diag['hermiticity_error']:.12e}"
    )

    log(
        f"  minimum eigenvalue = "
        f"{final_diag['minimum_eigenvalue']:.12e}"
    )

    log(
        f"  purity             = "
        f"{final_diag['purity']:.12f}"
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