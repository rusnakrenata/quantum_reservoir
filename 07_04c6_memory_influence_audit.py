"""
Week 7 - Step 7.4C.6
Audit: Does the carried memory state actually influence XZ_injection?

This diagnostic separates three possibilities:

A) The incoming memory states differ, but the injection reduced state does not.
   -> the one-step injection output is dynamically insensitive to memory.

B) The injection reduced states differ, but XZ_injection features do not.
   -> memory is present in the injection subsystem but hidden from the chosen
      X/Z observables.

C) XZ_injection features differ.
   -> CONT/W1 equality found earlier would indicate a protocol/code issue.

Two audits are performed:

1. Chronological CONT-vs-W1 audit over the real insurance sequence.
2. Controlled-memory perturbation:
   same current input A_t, several deliberately different incoming memory states.

Dependency:
    07_04c5_CONT_vs_W1_temporal_memory.py
"""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
C5 = HERE / "07_04c5_CONT_vs_W1_temporal_memory.py"

if not C5.exists():
    raise FileNotFoundError(f"Missing {C5.name}. Keep it beside this script.")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


c5 = load_module(C5, "memory74c5")
qrc = c5.qrc

ALPHA = 0.75
SEED = 42
qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]

FEATURE_NAMES = [
    "X0", "X1", "X2", "X3",
    "Z0", "Z1", "Z2", "Z3",
]

SELECTED_STEPS = [0, 1, 5, 10, 50, 100, 500, 1094, 1459]


# =============================================================================
# Linear algebra
# =============================================================================

def trace_distance(rho, sigma):
    """
    D(rho,sigma) = 1/2 ||rho-sigma||_1
    for Hermitian rho-sigma.
    """
    delta = np.asarray(rho - sigma, dtype=complex)
    delta = (delta + delta.conj().T) / 2.0
    eigvals = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def ket_density(v):
    v = np.asarray(v, dtype=complex)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def memory_test_states():
    """
    Two-memory-qubit states.
    """
    d = qrc.DIM_M
    if d != 4:
        raise RuntimeError(
            f"This audit expects two memory qubits (DIM_M=4), got DIM_M={d}."
        )

    e00 = np.array([1, 0, 0, 0], dtype=complex)
    e11 = np.array([0, 0, 0, 1], dtype=complex)
    plusplus = np.array([1, 1, 1, 1], dtype=complex) / 2.0
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2.0)

    return {
        "I/4": np.eye(4, dtype=complex) / 4.0,
        "|00>": ket_density(e00),
        "|11>": ket_density(e11),
        "|++>": ket_density(plusplus),
        "BellPhi+": ket_density(bell),
    }


def one_step(A, rho_mem):
    """
    Return injection output, memory output, XZ_injection feature vector.
    """
    rho_i_out, rho_m_out = qrc.final_reduced_states(A, rho_mem)

    feat = qrc.extract_feature_row(
        A,
        rho_mem,
        rho_i_out,
        rho_m_out,
    )

    x = np.array([feat[n] for n in FEATURE_NAMES], dtype=float)

    return rho_i_out, rho_m_out, x


# =============================================================================
# Audit 1: real chronological CONT vs W1
# =============================================================================

def chronological_audit(A_list):
    reset = np.eye(qrc.DIM_M, dtype=complex) / qrc.DIM_M
    rho_cont = reset.copy()

    rows = []

    for t, A in enumerate(A_list):
        D_mem_pre = trace_distance(rho_cont, reset)

        rho_i_cont, rho_m_cont, x_cont = one_step(A, rho_cont)
        rho_i_w1, rho_m_w1, x_w1 = one_step(A, reset)

        rows.append({
            "t": t,
            "D_memory_pre_CONT_vs_reset": D_mem_pre,
            "D_injection_out_CONT_vs_W1": trace_distance(
                rho_i_cont, rho_i_w1
            ),
            "D_memory_out_CONT_vs_W1": trace_distance(
                rho_m_cont, rho_m_w1
            ),
            "XZ_max_abs_diff": float(np.max(np.abs(x_cont - x_w1))),
            "XZ_rms_diff": float(
                np.sqrt(np.mean((x_cont - x_w1) ** 2))
            ),
        })

        rho_cont = rho_m_cont

    return pd.DataFrame(rows)


# =============================================================================
# Audit 2: deliberately alter memory while holding current input fixed
# =============================================================================

def controlled_memory_audit(A_list):
    states = memory_test_states()
    rows = []

    for t in SELECTED_STEPS:
        if t >= len(A_list):
            continue

        A = A_list[t]
        outputs = {}

        for name, rho_mem in states.items():
            rho_i, rho_m, x = one_step(A, rho_mem)
            outputs[name] = (rho_i, rho_m, x)

        for a, b in combinations(states.keys(), 2):
            rho_i_a, rho_m_a, x_a = outputs[a]
            rho_i_b, rho_m_b, x_b = outputs[b]

            rows.append({
                "t": t,
                "memory_A": a,
                "memory_B": b,
                "D_memory_input": trace_distance(
                    states[a], states[b]
                ),
                "D_injection_output": trace_distance(
                    rho_i_a, rho_i_b
                ),
                "D_memory_output": trace_distance(
                    rho_m_a, rho_m_b
                ),
                "XZ_max_abs_diff": float(
                    np.max(np.abs(x_a - x_b))
                ),
                "XZ_rms_diff": float(
                    np.sqrt(np.mean((x_a - x_b) ** 2))
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 124)
    print("WEEK 7 - STEP 7.4C.6")
    print("AUDIT: DOES THE CARRIED MEMORY INFLUENCE XZ_INJECTION?")
    print("=" * 124)
    print()
    print("Logic:")
    print("  memory input differs + injection state same -> dynamics/readout path is memory-insensitive")
    print("  injection state differs + XZ same         -> XZ observables are memory-blind")
    print("  XZ differs                                -> investigate prior CONT/W1 construction")
    print()

    work, train, val, cols = qrc.load_data()
    n_steps = len(train) + len(val)

    Uchron = np.asarray(
        qrc.make_input_angles(work, cols),
        dtype=float,
    )[:n_steps]

    Ures, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, _ = qrc.build_input_channels(Ures, Uchron)

    print(f"Steps={n_steps}")
    print(f"QRC unitarity error={unitary_err:.3e}")
    print()

    # -------------------------------------------------------------------------
    # 1. Chronological audit
    # -------------------------------------------------------------------------
    print("Running chronological CONT-vs-W1 state audit ...")
    chron = chronological_audit(A_list)

    chron.to_csv(
        RESULTS / "07_04c6_chronological_memory_influence_audit.csv",
        index=False,
    )

    print()
    print("-" * 124)
    print("CHRONOLOGICAL CONT-vs-W1 AUDIT")
    print("-" * 124)

    metrics = [
        "D_memory_pre_CONT_vs_reset",
        "D_injection_out_CONT_vs_W1",
        "D_memory_out_CONT_vs_W1",
        "XZ_max_abs_diff",
        "XZ_rms_diff",
    ]

    summary_rows = []
    for m in metrics:
        summary_rows.append({
            "metric": m,
            "mean": float(chron[m].mean()),
            "median": float(chron[m].median()),
            "max": float(chron[m].max()),
            "p90": float(chron[m].quantile(0.90)),
        })

    chron_summary = pd.DataFrame(summary_rows)
    print(chron_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. Controlled memory perturbation
    # -------------------------------------------------------------------------
    print()
    print("Running controlled-memory perturbation audit ...")
    controlled = controlled_memory_audit(A_list)

    controlled.to_csv(
        RESULTS / "07_04c6_controlled_memory_perturbation.csv",
        index=False,
    )

    print()
    print("-" * 124)
    print("CONTROLLED MEMORY PERTURBATION")
    print("-" * 124)

    by_t = (
        controlled.groupby("t")
        .agg(
            max_D_memory_input=("D_memory_input", "max"),
            max_D_injection_output=("D_injection_output", "max"),
            max_D_memory_output=("D_memory_output", "max"),
            max_XZ_abs_diff=("XZ_max_abs_diff", "max"),
            max_XZ_rms_diff=("XZ_rms_diff", "max"),
        )
        .reset_index()
    )

    print(by_t.to_string(index=False))

    # Overall decision quantities.
    max_pre = float(chron["D_memory_pre_CONT_vs_reset"].max())
    max_inj_chron = float(chron["D_injection_out_CONT_vs_W1"].max())
    max_xz_chron = float(chron["XZ_max_abs_diff"].max())

    max_inj_control = float(controlled["D_injection_output"].max())
    max_mem_control = float(controlled["D_memory_output"].max())
    max_xz_control = float(controlled["XZ_max_abs_diff"].max())

    print()
    print("=" * 124)
    print("DECISION NUMBERS")
    print("=" * 124)
    print(f"Chronological max incoming-memory difference       = {max_pre:.12e}")
    print(f"Chronological max injection-output trace distance  = {max_inj_chron:.12e}")
    print(f"Chronological max XZ feature difference            = {max_xz_chron:.12e}")
    print()
    print(f"Controlled max injection-output trace distance     = {max_inj_control:.12e}")
    print(f"Controlled max memory-output trace distance        = {max_mem_control:.12e}")
    print(f"Controlled max XZ feature difference               = {max_xz_control:.12e}")
    print()

    tol = 1e-10

    if max_inj_control <= tol and max_xz_control <= tol:
        verdict = (
            "The injection output is effectively independent of the incoming "
            "memory state for this implemented one-step map. XZ_injection cannot "
            "carry temporal memory because the memory does not feed back into the "
            "measured injection subsystem."
        )
    elif max_inj_control > tol and max_xz_control <= tol:
        verdict = (
            "The incoming memory DOES influence the injection reduced state, but "
            "the chosen XZ_injection observables do not detect that difference. "
            "This is an observable/readout blindness problem."
        )
    else:
        verdict = (
            "The incoming memory influences XZ_injection. The exact CONT/W1 "
            "equality from Step 7.4C.5 therefore needs a protocol/construction audit."
        )

    print("VERDICT:")
    print(verdict)

    with open(
        RESULTS / "07_04c6_memory_influence_audit_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write("WEEK 7 STEP 7.4C.6 - MEMORY INFLUENCE AUDIT\n")
        fp.write("=" * 90 + "\n\n")
        fp.write(chron_summary.to_string(index=False))
        fp.write("\n\nCONTROLLED BY STEP:\n")
        fp.write(by_t.to_string(index=False))
        fp.write("\n\nDECISION NUMBERS:\n")
        fp.write(
            f"chron max memory pre diff = {max_pre:.12e}\n"
            f"chron max injection out D = {max_inj_chron:.12e}\n"
            f"chron max XZ diff = {max_xz_chron:.12e}\n"
            f"controlled max injection out D = {max_inj_control:.12e}\n"
            f"controlled max memory out D = {max_mem_control:.12e}\n"
            f"controlled max XZ diff = {max_xz_control:.12e}\n\n"
        )
        fp.write("VERDICT:\n" + verdict + "\n")

    print()
    print("Saved:")
    print("  results/07_04c6_chronological_memory_influence_audit.csv")
    print("  results/07_04c6_controlled_memory_perturbation.csv")
    print("  results/07_04c6_memory_influence_audit_summary.txt")


if __name__ == "__main__":
    main()
