#!/usr/bin/env python
from __future__ import annotations

"""
WEEK 11.1F.6 — CANDIDATE #4 QISKIT-LAYOUT FEASIBILITY PREFLIGHT
===============================================================

Candidate #4 was not rejected as a logical QRC model.  It was withheld from a
full real-QPU transfer because the washout-equivalent hardware representation
is RWP117 and the earlier physical realization was far beyond the persistent
memory coherence budget.

This script asks one narrow question:
    Can Qiskit's default physical placement make the frozen Candidate #4 RWP117
    implementation coherence-feasible NOW?

NO QPU job is submitted.

Protocol
--------
Candidate #4:
- H0 chain
- RWP117 hardware representation of frozen CONT Rule-4 model
- r = 4
- alpha = 0.75
- dt = 1.6
- hx = +0.580092560570867
- hy = -0.694330375116467
- readout = XZinj_plus_YX45
- intended full-run operating point = 512 shots/setting

Qiskit selects a layout ONCE from the actual long RWP117 circuit.
The chosen layout is frozen and strict-compiled with routing_method="none".
Both persistent memory qubits q4/q5 must satisfy:
    t_max_setting/T1 < 1
    t_max_setting/T2 < 1

If that fails, Candidate #4 remains non-feasible for the present hardware
snapshot and no QPU time should be spent on it.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile

from ibm_account import get_service


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

PREFIX = "11_1F6_candidate4_qiskit_layout_preflight"
DEFAULT_BACKEND = "ibm_kingston"
OPT_LEVEL = 1
SEED_TRANSPILE = 42

WINDOW = 117
R = 4
DT = 1.6
HX = +0.580092560570867
HY = -0.694330375116467
J = {
    (0, 1): -1.6178834565099725,
    (1, 2): -0.1998267640484581,
    (2, 3): -1.6305688408636787,
    (3, 4): -0.489627102594124,
    (4, 5): +0.5392332533499583,
}
EDGES = list(J)

SETTINGS = {
    "XXXXYX": {
        0: "X", 1: "X", 2: "X", 3: "X", 4: "Y", 5: "X"
    },
    "ZZZZZZ": {
        0: "Z", 1: "Z", 2: "Z", 3: "Z", 4: "Z", 5: "Z"
    },
}


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


def dummy_angles(step: int, phase: float = 0.0):
    s = float(step)
    return [
        0.29 + 0.08 * math.sin(0.041 * s + phase),
        -0.51 + 0.06 * math.cos(0.057 * s + phase),
        0.69 + 0.07 * math.sin(0.071 * s + 0.4 + phase),
        -0.27 + 0.05 * math.cos(0.033 * s + 0.2 + phase),
    ]


def append_step(qc, angles, first_step):
    if not first_step:
        for q in range(4):
            qc.reset(q)

    for q in range(4):
        qc.ry(float(angles[q]), q)

    for _ in range(R):
        for edge in EDGES:
            i, j = edge
            qc.rzz(2.0 * J[edge] * DT / R, i, j)

        theta_x = 2.0 * HX * DT / R
        for q in range(6):
            qc.rx(theta_x, q)

        theta_y = 2.0 * HY * DT / R
        qc.ry(theta_y, 4)
        qc.ry(theta_y, 5)


def build_probe(setting_label: str, phase: float = 0.0):
    qc = QuantumCircuit(6, 6, name=f"C4_RWP117_{setting_label}")

    for step in range(WINDOW):
        append_step(
            qc,
            dummy_angles(step, phase),
            first_step=(step == 0),
        )

    for q, axis in SETTINGS[setting_label].items():
        if axis == "X":
            qc.h(q)
        elif axis == "Y":
            qc.sdg(q)
            qc.h(q)

    for q in range(6):
        qc.measure(q, q)

    return qc


def extract_initial_layout(compiled, logical, n=6):
    tl = getattr(compiled, "layout", None)
    if tl is None:
        raise RuntimeError("Qiskit returned no TranspileLayout.")

    fn = getattr(tl, "initial_index_layout", None)
    if callable(fn):
        try:
            vals = fn(filter_ancillas=True)
        except TypeError:
            vals = fn()
        vals = [int(v) for v in vals]
        if len(vals) >= n:
            return vals[:n]

    init = getattr(tl, "initial_layout", None)
    if init is None:
        raise RuntimeError("Could not recover Qiskit's initial layout.")

    return [int(init[q]) for q in logical.qubits[:n]]


def calibration(backend, layout):
    props = backend.properties()
    rows = []

    for logical_q, physical_q in enumerate(layout):
        try:
            t1 = float(props.t1(int(physical_q))) * 1e6
        except Exception:
            t1 = np.nan
        try:
            t2 = float(props.t2(int(physical_q))) * 1e6
        except Exception:
            t2 = np.nan
        try:
            ro = float(props.readout_error(int(physical_q)))
        except Exception:
            ro = np.nan

        rows.append({
            "logical_qubit": logical_q,
            "physical_qubit": int(physical_q),
            "T1_us": t1,
            "T2_us": t2,
            "readout_error": ro,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    args = parser.parse_args()

    service = get_service()
    backend = service.backend(
        args.backend,
        use_fractional_gates=False,
    )
    try:
        backend.refresh()
    except Exception:
        pass

    print("=" * 132)
    print("CANDIDATE #4 — QISKIT-LAYOUT RWP117 FEASIBILITY PREFLIGHT")
    print("=" * 132)
    print("NO QPU JOB WILL BE SUBMITTED.")
    print("2026 is not loaded.")
    print()

    logical = build_probe("XXXXYX", phase=0.0)

    chosen_compile = transpile(
        logical,
        backend=backend,
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
        scheduling_method="alap",
    )
    backend.check_faulty(chosen_compile)

    layout = extract_initial_layout(chosen_compile, logical)

    rows = []
    max_duration_us = 0.0

    # Three structurally identical but numerically different long endpoints.
    for phase_id, phase in enumerate([0.0, 0.67, 1.31], start=1):
        for setting in ["XXXXYX", "ZZZZZZ"]:
            qc = build_probe(setting, phase=phase)

            try:
                isa = transpile(
                    qc,
                    backend=backend,
                    initial_layout=layout,
                    routing_method="none",
                    optimization_level=OPT_LEVEL,
                    seed_transpiler=SEED_TRANSPILE,
                    scheduling_method="alap",
                )
            except Exception as exc:
                raise RuntimeError(
                    "Qiskit's Candidate-4 placement cannot be frozen as a "
                    "strict native zero-routing H0 embedding."
                ) from exc

            backend.check_faulty(isa)

            ops = {str(k): int(v) for k, v in isa.count_ops().items()}
            if int(ops.get("swap", 0)) != 0:
                raise RuntimeError("SWAP detected in strict Candidate-4 audit.")

            dur = float(
                isa.estimate_duration(backend.target, unit="s")
            ) * 1e6
            max_duration_us = max(max_duration_us, dur)

            rows.append({
                "phase_id": phase_id,
                "setting": setting,
                "layout": json.dumps(layout),
                "depth": int(isa.depth()),
                "size": int(isa.size()),
                "n_cz": int(ops.get("cz", 0)),
                "n_reset": int(ops.get("reset", 0)),
                "n_measure": int(ops.get("measure", 0)),
                "n_swap": int(ops.get("swap", 0)),
                "duration_us": dur,
                "operations": json.dumps(ops, sort_keys=True),
            })

    resources = pd.DataFrame(rows)
    resources_path = RESULTS / f"{PREFIX}_resources.csv"
    resources.to_csv(resources_path, index=False)

    cal = calibration(backend, layout)
    cal["max_setting_duration_us"] = max_duration_us
    cal["duration_over_T1"] = max_duration_us / cal["T1_us"]
    cal["duration_over_T2"] = max_duration_us / cal["T2_us"]
    cal["coherence_pass"] = (
        (cal["duration_over_T1"] < 1.0)
        & (cal["duration_over_T2"] < 1.0)
    )

    cal_path = RESULTS / f"{PREFIX}_qubits.csv"
    cal.to_csv(cal_path, index=False)

    mem = cal[cal["logical_qubit"].isin([4, 5])].copy()
    memory_pass = bool(mem["coherence_pass"].all())

    feature_cz = int(
        resources.groupby("phase_id")["n_cz"].sum().median()
    )
    feature_resets = int(
        resources.groupby("phase_id")["n_reset"].sum().median()
    )
    feature_duration = float(
        resources.groupby("phase_id")["duration_us"].sum().median()
    )

    summary = {
        "candidate": "Candidate #4 / CONT_H0_R4 represented as RWP117",
        "backend": args.backend,
        "selection_method": "qiskit_transpiler_default_once_then_frozen",
        "layout": layout,
        "physical_C_t": int(layout[0]),
        "physical_D": int(layout[1]),
        "physical_P_t": int(layout[2]),
        "physical_H": int(layout[3]),
        "physical_M1": int(layout[4]),
        "physical_M2": int(layout[5]),
        "window": WINDOW,
        "r": R,
        "readout": "XZinj_plus_YX45",
        "intended_shots_per_setting": 512,
        "strict_zero_swap_audit": True,
        "median_cz_per_feature": feature_cz,
        "median_resets_per_feature": feature_resets,
        "median_feature_duration_us": feature_duration,
        "max_setting_duration_us": max_duration_us,
        "memory_qubit_rows": mem.to_dict(orient="records"),
        "memory_coherence_feasible": memory_pass,
        "decision_rule": (
            "Full Candidate-4 QPU testing is considered only if BOTH q4/q5 "
            "satisfy max-setting-duration/T1<1 and /T2<1."
        ),
    }

    summary_path = RESULTS / f"{PREFIX}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Qiskit-selected layout:        {layout}")
    print(f"median CZ / feature:           {feature_cz}")
    print(f"median resets / feature:       {feature_resets}")
    print(f"median feature duration:       {feature_duration:.3f} us")
    print(f"max setting duration:          {max_duration_us:.3f} us")
    print()

    for _, r in mem.iterrows():
        print(
            f"q{int(r.logical_qubit)} -> P{int(r.physical_qubit)}: "
            f"T1={r.T1_us:.3f} us, T2={r.T2_us:.3f} us, "
            f"t/T1={r.duration_over_T1:.3f}, "
            f"t/T2={r.duration_over_T2:.3f}, "
            f"PASS={bool(r.coherence_pass)}"
        )

    print()
    print(
        "CANDIDATE #4 QISKIT-LAYOUT MEMORY FEASIBILITY: "
        + ("PASS" if memory_pass else "FAIL")
    )

    if memory_pass:
        print(
            "A single controlled full-QPU Candidate-4 ablation is now "
            "scientifically defensible, subject to quota review."
        )
    else:
        print(
            "Do NOT spend QPU time on Candidate #4 under this calibration; "
            "Qiskit placement did not repair the RWP117 coherence problem."
        )

    print()
    print(f"Saved: {summary_path}")
    print(f"       {resources_path}")
    print(f"       {cal_path}")
    print("2026 remains FROZEN / UNUSED.")


if __name__ == "__main__":
    main()
