#!/usr/bin/env python
from __future__ import annotations

"""
WEEK 11.1E.9D — CANDIDATE #3 MID-CIRCUIT RESET DIAGNOSTIC
================================================================

Purpose
-------
Directly test reset quality on the four injection qubits of the repeatedly
failing Candidate #3 physical layout:

    q0 -> P105
    q1 -> P117
    q2 -> P125
    q3 -> P124

The test is deliberately small and independent of the QRC forecast.

For each physical qubit, the SAME SamplerV2 job contains:
    RO0     : |0> -> measure
    RO1     : |1> -> measure
    RESET0  : |0> -> reset -> measure
    RESET1  : |1> -> reset -> measure
    RESET+  : |+> -> reset -> measure

RO0 and RO1 estimate the final readout assignment errors:
    e0 = P(measure 1 | true 0)
    e1 = P(measure 0 | true 1)

For a reset circuit with observed P(measure 1)=p_obs, the first-order
readout-corrected post-reset excited-state population is

    p_exc = (p_obs - e0) / (1 - e0 - e1)

The script also tests simultaneous reset of all four injection qubits from
|1111> and |++++>, because Candidate #3 resets q0..q3 between its W=2 inputs.

No M3 is applied.
No 2026 data are loaded.
Default is DRY RUN.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import SamplerV2

from ibm_account import get_service


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_SHOTS = 4096

# Exact repeatedly failing Candidate #3 injection-qubit placement.
PHYSICAL_QUBITS = [105, 117, 125, 124]

# For readable output.
LOGICAL_LABELS = {
    105: "q0",
    117: "q1",
    125: "q2",
    124: "q3",
}

OPT_LEVEL = 0
SEED_TRANSPILE = 42


def bit_from_qiskit_string(bitstring, c_index):
    s = str(bitstring).replace(" ", "")
    return int(s[-1 - int(c_index)])


def p_one_from_counts(counts, c_index=0):
    total = float(sum(counts.values()))
    if total <= 0:
        raise RuntimeError("Empty counts.")

    ones = 0.0
    for bitstring, n in counts.items():
        if bit_from_qiskit_string(bitstring, c_index) == 1:
            ones += float(n)

    return ones / total


def any_one_probability(counts, n_bits):
    total = float(sum(counts.values()))
    if total <= 0:
        raise RuntimeError("Empty counts.")

    bad = 0.0
    for bitstring, n in counts.items():
        bits = [
            bit_from_qiskit_string(bitstring, c)
            for c in range(n_bits)
        ]
        if any(bits):
            bad += float(n)

    return bad / total


def get_pub_counts(pub_result):
    reg = getattr(pub_result.data, "m", None)
    if reg is None:
        raise RuntimeError(
            "Sampler result does not contain classical register 'm'."
        )
    return reg.get_counts()


def build_single_qubit_circuit(kind):
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(1, "m")
    qc = QuantumCircuit(q, c, name=kind)

    if kind == "RO0":
        pass

    elif kind == "RO1":
        qc.x(0)

    elif kind == "RESET0":
        qc.reset(0)

    elif kind == "RESET1":
        qc.x(0)
        qc.reset(0)

    elif kind == "RESETPLUS":
        qc.h(0)
        qc.reset(0)

    else:
        raise ValueError(kind)

    qc.measure(0, 0)
    return qc


def build_parallel_reset(kind):
    q = QuantumRegister(4, "q")
    c = ClassicalRegister(4, "m")
    qc = QuantumCircuit(q, c, name=kind)

    if kind == "PAR_RESET1":
        for i in range(4):
            qc.x(i)

    elif kind == "PAR_RESETPLUS":
        for i in range(4):
            qc.h(i)

    else:
        raise ValueError(kind)

    # Same logical pattern as Candidate #3: reset all four injection qubits.
    for i in range(4):
        qc.reset(i)

    for i in range(4):
        qc.measure(i, i)

    return qc


def safe_instruction_property(backend, instruction_name, physical_q):
    try:
        props = backend.target[instruction_name].get((physical_q,))
        if props is None:
            return None
        return {
            "duration_s": (
                None if props.duration is None else float(props.duration)
            ),
            "error": (
                None if props.error is None else float(props.error)
            ),
        }
    except Exception:
        return None


def corrected_excited_population(p_obs, e0, e1):
    """
    Assignment model:
        p_obs = e0*(1-p) + (1-e1)*p
              = e0 + (1-e0-e1)*p
    """
    denom = 1.0 - float(e0) - float(e1)

    if abs(denom) < 1e-12:
        return np.nan, np.nan

    p = (float(p_obs) - float(e0)) / denom
    return float(p), float(np.clip(p, 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the small reset diagnostic. Default is dry-run.",
    )

    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError("--shots must be positive.")

    print("=" * 118)
    print("WEEK 11.1E.9D — CANDIDATE #3 MID-CIRCUIT RESET DIAGNOSTIC")
    print("=" * 118)
    print("2026 = FROZEN / NOT LOADED")
    print("No M3.")
    print()
    print(f"Backend: {args.backend}")
    print(f"Physical injection qubits: {PHYSICAL_QUBITS}")
    print(f"Shots/circuit: {args.shots}")
    print()

    service = get_service()

    backend = service.backend(
        args.backend,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Inspect any reset properties exposed by the backend Target.
    # These are metadata only; the real test is the direct QPU experiment.
    # ------------------------------------------------------------------
    print("Backend Target reset properties:")
    reset_target_rows = []

    for p in PHYSICAL_QUBITS:
        prop = safe_instruction_property(
            backend,
            "reset",
            p,
        )

        row = {
            "physical_qubit": int(p),
            "logical_qubit": LOGICAL_LABELS[p],
            "target_reset_duration_us": (
                None if not prop or prop["duration_s"] is None
                else prop["duration_s"] * 1e6
            ),
            "target_reset_error": (
                None if not prop else prop["error"]
            ),
        }
        reset_target_rows.append(row)

        print(
            f"  {LOGICAL_LABELS[p]} -> P{p}: "
            f"{json.dumps(row, default=str)}"
        )

    # ------------------------------------------------------------------
    # Build all circuits.
    # ------------------------------------------------------------------
    single_kinds = [
        "RO0",
        "RO1",
        "RESET0",
        "RESET1",
        "RESETPLUS",
    ]

    circuits = []
    metadata = []
    resource_rows = []

    for p in PHYSICAL_QUBITS:
        for kind in single_kinds:
            logical = build_single_qubit_circuit(kind)

            isa = transpile(
                logical,
                backend=backend,
                initial_layout=[p],
                routing_method="none",
                optimization_level=OPT_LEVEL,
                seed_transpiler=SEED_TRANSPILE,
                scheduling_method="alap",
            )

            backend.check_faulty(isa)

            ops = {
                str(k): int(v)
                for k, v in isa.count_ops().items()
            }

            duration_s = float(
                isa.estimate_duration(
                    backend.target,
                    unit="s",
                )
            )

            circuits.append(isa)
            metadata.append({
                "mode": "single",
                "physical_qubit": int(p),
                "logical_qubit": LOGICAL_LABELS[p],
                "kind": kind,
            })
            resource_rows.append({
                "mode": "single",
                "physical_qubit": int(p),
                "logical_qubit": LOGICAL_LABELS[p],
                "kind": kind,
                "depth": int(isa.depth()),
                "size": int(isa.size()),
                "duration_us": duration_s * 1e6,
                "n_reset": int(ops.get("reset", 0)),
                "n_measure": int(ops.get("measure", 0)),
                "operations": json.dumps(ops, sort_keys=True),
            })

    # Simultaneous four-qubit reset tests.
    for kind in ["PAR_RESET1", "PAR_RESETPLUS"]:
        logical = build_parallel_reset(kind)

        isa = transpile(
            logical,
            backend=backend,
            initial_layout=PHYSICAL_QUBITS,
            routing_method="none",
            optimization_level=OPT_LEVEL,
            seed_transpiler=SEED_TRANSPILE,
            scheduling_method="alap",
        )

        backend.check_faulty(isa)

        ops = {
            str(k): int(v)
            for k, v in isa.count_ops().items()
        }

        duration_s = float(
            isa.estimate_duration(
                backend.target,
                unit="s",
            )
        )

        circuits.append(isa)
        metadata.append({
            "mode": "parallel",
            "physical_qubits": PHYSICAL_QUBITS,
            "kind": kind,
        })
        resource_rows.append({
            "mode": "parallel",
            "physical_qubit": None,
            "logical_qubit": "q0-q3",
            "kind": kind,
            "depth": int(isa.depth()),
            "size": int(isa.size()),
            "duration_us": duration_s * 1e6,
            "n_reset": int(ops.get("reset", 0)),
            "n_measure": int(ops.get("measure", 0)),
            "operations": json.dumps(ops, sort_keys=True),
        })

    resources = pd.DataFrame(resource_rows)

    print()
    print("=" * 118)
    print("RESET TEST PREFLIGHT")
    print("=" * 118)
    print(f"Circuits:                    {len(circuits)}")
    print(f"Shots/circuit:               {args.shots}")
    print(f"Single-qubit circuits:       {4 * len(single_kinds)}")
    print("Parallel reset circuits:     2")
    print(
        f"Median circuit duration:     "
        f"{resources['duration_us'].median():.3f} us"
    )
    print(
        f"Maximum circuit duration:    "
        f"{resources['duration_us'].max():.3f} us"
    )
    print()

    resources_file = (
        RESULTS /
        "11_1E9D_candidate3_reset_resources.csv"
    )
    resources.to_csv(resources_file, index=False)

    target_file = (
        RESULTS /
        "11_1E9D_candidate3_reset_target_properties.csv"
    )
    pd.DataFrame(reset_target_rows).to_csv(
        target_file,
        index=False,
    )

    if not args.submit:
        print("DRY RUN COMPLETE — NO QPU JOB SUBMITTED.")
        print()
        print("To run the reset diagnostic:")
        print(
            f"python {Path(__file__).name} "
            f"--submit --backend {args.backend} --shots {args.shots}"
        )
        print()
        print("Saved:")
        print(f"  {resources_file}")
        print(f"  {target_file}")
        return

    # ------------------------------------------------------------------
    # Submit one small paired job.
    # ------------------------------------------------------------------
    print("=" * 118)
    print("SUBMITTING RESET DIAGNOSTIC")
    print("=" * 118)

    sampler = SamplerV2(mode=backend)
    job = sampler.run(
        circuits,
        shots=int(args.shots),
    )

    print(f"Job ID: {job.job_id()}")

    result = job.result()
    print(f"Final status: {job.status()}")

    if len(result) != len(circuits):
        raise RuntimeError(
            f"Sampler returned {len(result)} results "
            f"for {len(circuits)} circuits."
        )

    # ------------------------------------------------------------------
    # Parse raw counts.
    # ------------------------------------------------------------------
    raw_rows = []
    single_results = {}
    parallel_results = {}

    for pub, meta in zip(result, metadata):
        counts = get_pub_counts(pub)

        raw_rows.append({
            **meta,
            "counts_json": json.dumps(counts, sort_keys=True),
        })

        if meta["mode"] == "single":
            p = int(meta["physical_qubit"])
            kind = str(meta["kind"])
            single_results[(p, kind)] = {
                "p_one": p_one_from_counts(counts, 0),
                "counts": counts,
            }

        else:
            kind = str(meta["kind"])

            marginals = {
                PHYSICAL_QUBITS[c]: p_one_from_counts(counts, c)
                for c in range(4)
            }

            parallel_results[kind] = {
                "marginal_p_one": marginals,
                "p_any_one": any_one_probability(counts, 4),
                "counts": counts,
            }

    raw_counts_df = pd.DataFrame(raw_rows)

    raw_file = (
        RESULTS /
        "11_1E9D_candidate3_reset_raw_counts.csv"
    )
    raw_counts_df.to_csv(raw_file, index=False)

    # ------------------------------------------------------------------
    # Single-qubit reset analysis with same-job readout correction.
    # ------------------------------------------------------------------
    diagnostics = []

    for p in PHYSICAL_QUBITS:
        e0 = float(single_results[(p, "RO0")]["p_one"])
        e1 = float(
            1.0 - single_results[(p, "RO1")]["p_one"]
        )

        assignment_denom = 1.0 - e0 - e1

        for kind in ["RESET0", "RESET1", "RESETPLUS"]:
            p_obs = float(
                single_results[(p, kind)]["p_one"]
            )

            p_corr, p_clip = corrected_excited_population(
                p_obs,
                e0,
                e1,
            )

            diagnostics.append({
                "logical_qubit": LOGICAL_LABELS[p],
                "physical_qubit": int(p),
                "test": kind,
                "shots": int(args.shots),
                "readout_e0_P1_given_0": e0,
                "readout_e1_P0_given_1": e1,
                "assignment_denominator": assignment_denom,
                "observed_P1_after_reset": p_obs,
                "corrected_P1_after_reset_unclipped": p_corr,
                "corrected_P1_after_reset_clipped": p_clip,
            })

    diag_df = pd.DataFrame(diagnostics)

    # ------------------------------------------------------------------
    # Parallel reset analysis.
    # Marginals use the same single-qubit readout correction parameters.
    # ------------------------------------------------------------------
    parallel_rows = []

    for kind, info in parallel_results.items():
        for p in PHYSICAL_QUBITS:
            e0 = float(single_results[(p, "RO0")]["p_one"])
            e1 = float(
                1.0 - single_results[(p, "RO1")]["p_one"]
            )

            p_obs = float(
                info["marginal_p_one"][p]
            )

            p_corr, p_clip = corrected_excited_population(
                p_obs,
                e0,
                e1,
            )

            parallel_rows.append({
                "parallel_test": kind,
                "logical_qubit": LOGICAL_LABELS[p],
                "physical_qubit": int(p),
                "observed_marginal_P1": p_obs,
                "corrected_marginal_P1_unclipped": p_corr,
                "corrected_marginal_P1_clipped": p_clip,
                "raw_P_any_one_among_four": float(
                    info["p_any_one"]
                ),
            })

    parallel_df = pd.DataFrame(parallel_rows)

    # ------------------------------------------------------------------
    # Output.
    # ------------------------------------------------------------------
    print()
    print("=" * 118)
    print("SINGLE-QUBIT RESET RESULTS")
    print("=" * 118)

    show = diag_df[
        [
            "logical_qubit",
            "physical_qubit",
            "test",
            "readout_e0_P1_given_0",
            "readout_e1_P0_given_1",
            "observed_P1_after_reset",
            "corrected_P1_after_reset_clipped",
        ]
    ].copy()

    print(
        show.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 118)
    print("SIMULTANEOUS FOUR-QUBIT RESET RESULTS")
    print("=" * 118)

    print(
        parallel_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # Worst values make interpretation easier without defining an arbitrary
    # pass/fail threshold.
    reset1 = diag_df[diag_df["test"] == "RESET1"]

    worst_single = reset1.loc[
        reset1["corrected_P1_after_reset_clipped"].idxmax()
    ]

    worst_parallel_idx = (
        parallel_df["corrected_marginal_P1_clipped"].idxmax()
    )
    worst_parallel = parallel_df.loc[worst_parallel_idx]

    print()
    print("=" * 118)
    print("RESET DIAGNOSTIC SUMMARY")
    print("=" * 118)

    print(
        "Worst corrected RESET1 residual excitation: "
        f"{worst_single['logical_qubit']} -> "
        f"P{int(worst_single['physical_qubit'])}: "
        f"{worst_single['corrected_P1_after_reset_clipped']:.6f}"
    )

    print(
        "Worst corrected simultaneous-reset marginal: "
        f"{worst_parallel['logical_qubit']} -> "
        f"P{int(worst_parallel['physical_qubit'])}: "
        f"{worst_parallel['corrected_marginal_P1_clipped']:.6f}"
    )

    # IBM job metrics if available.
    metrics = None
    try:
        metrics = job.metrics()
        print()
        print("Job metrics:")
        print(json.dumps(metrics, indent=2, default=str))
    except Exception as exc:
        print(f"Job metrics unavailable: {exc}")

    diag_file = (
        RESULTS /
        "11_1E9D_candidate3_reset_single_diagnostics.csv"
    )
    parallel_file = (
        RESULTS /
        "11_1E9D_candidate3_reset_parallel_diagnostics.csv"
    )
    summary_file = (
        RESULTS /
        "11_1E9D_candidate3_reset_summary.json"
    )

    diag_df.to_csv(diag_file, index=False)
    parallel_df.to_csv(parallel_file, index=False)

    summary = {
        "backend": args.backend,
        "physical_qubits": PHYSICAL_QUBITS,
        "shots": int(args.shots),
        "job_id": job.job_id(),
        "job_status": str(job.status()),
        "worst_RESET1": {
            "logical_qubit": str(worst_single["logical_qubit"]),
            "physical_qubit": int(worst_single["physical_qubit"]),
            "corrected_P1": float(
                worst_single[
                    "corrected_P1_after_reset_clipped"
                ]
            ),
        },
        "worst_parallel": {
            "test": str(worst_parallel["parallel_test"]),
            "logical_qubit": str(worst_parallel["logical_qubit"]),
            "physical_qubit": int(worst_parallel["physical_qubit"]),
            "corrected_marginal_P1": float(
                worst_parallel[
                    "corrected_marginal_P1_clipped"
                ]
            ),
        },
        "metrics": metrics,
        "2026_loaded": False,
    }

    summary_file.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(f"  {resources_file}")
    print(f"  {target_file}")
    print(f"  {raw_file}")
    print(f"  {diag_file}")
    print(f"  {parallel_file}")
    print(f"  {summary_file}")
    print()
    print("2026 remains FROZEN / UNUSED.")


if __name__ == "__main__":
    main()
