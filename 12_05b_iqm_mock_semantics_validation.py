"""
Week 12.5B — IQM mock execution semantics validation.

Purpose
-------
Test whether `emerald:mock` and `garnet:mock` behave ideally or exhibit
non-ideal/noisy execution. This is NOT a QRC experiment.

Only mock aliases are allowed to execute. Authentication is sourced from
iqm_account.py. No real-QPU jobs and no forecasting data are used.

Deterministic tests:
  T0_ZERO       |00>                  -> allowed {"00"}
  T1_X1         X(q0)|00>             -> allowed {"01"}
  T2_ONES       X(q0)X(q1)|00>        -> allowed {"11"}
  T3_X_STRESS   X(q0)^21|00>          -> allowed {"01"}
  T4_CZ_STRESS  CZ^31|00>             -> allowed {"00"}
  T5_BELL       Bell state            -> allowed {"00","11"}

An ideal simulator assigns exactly zero probability to forbidden outcomes in
all six tests. Therefore finite-shot sampling alone cannot create a forbidden
outcome.

Outputs
-------
results/12_05b_iqm_mock_execution_detail.csv
results/12_05b_iqm_mock_execution_summary.csv
results/12_05b_iqm_mock_target_errors.csv
results/12_05b_iqm_mock_semantics_assessment.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from iqm.qdmi.qiskit import IQMBackend

from iqm_account import get_authentication


RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

OUT_DETAIL = RESULTS / "12_05b_iqm_mock_execution_detail.csv"
OUT_SUMMARY = RESULTS / "12_05b_iqm_mock_execution_summary.csv"
OUT_TARGET = RESULTS / "12_05b_iqm_mock_target_errors.csv"
OUT_ASSESSMENT = RESULTS / "12_05b_iqm_mock_semantics_assessment.json"

MOCK_ALIASES = ("emerald:mock", "garnet:mock")
DEFAULT_SHOTS = 2048
N_X_STRESS = 21
N_CZ_STRESS = 31


@contextmanager
def temporary_iqm_environment(token: str, server_url: str):
    old_token = os.environ.get("IQM_TOKEN")
    old_server = os.environ.get("IQM_SERVER_URL")
    try:
        os.environ["IQM_TOKEN"] = token
        os.environ["IQM_SERVER_URL"] = server_url.rstrip("/")
        yield
    finally:
        if old_token is None:
            os.environ.pop("IQM_TOKEN", None)
        else:
            os.environ["IQM_TOKEN"] = old_token

        if old_server is None:
            os.environ.pop("IQM_SERVER_URL", None)
        else:
            os.environ["IQM_SERVER_URL"] = old_server


def finite_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def target_error_rows(backend, alias):
    rows = []
    for opname in sorted(str(x) for x in backend.target.operation_names):
        try:
            items = list(backend.target[opname].items())
        except Exception:
            items = []

        for qargs, props in items:
            err = None if props is None else finite_float(
                getattr(props, "error", None)
            )
            dur = None if props is None else finite_float(
                getattr(props, "duration", None)
            )
            rows.append({
                "mock_alias": alias,
                "operation": opname,
                "qargs": json.dumps(list(qargs)) if qargs is not None else "null",
                "error": np.nan if err is None else err,
                "duration_s": np.nan if dur is None else dur,
            })
    return rows


def choose_test_edge(backend):
    if "cz" not in backend.target.operation_names:
        raise RuntimeError("Mock target has no mapped CZ operation.")

    candidates = []
    for qargs, props in backend.target["cz"].items():
        if qargs is None or len(qargs) != 2:
            continue
        err = None if props is None else finite_float(
            getattr(props, "error", None)
        )
        candidates.append((int(qargs[0]), int(qargs[1]), err))

    if not candidates:
        raise RuntimeError("No CZ loci found in mock target.")

    finite = [x for x in candidates if x[2] is not None]
    if finite:
        # Highest native target error gives the most sensitive mock-semantics test.
        return max(finite, key=lambda x: x[2])

    return sorted(candidates)[:1][0]


def target_error_for(backend, opname, qargs):
    try:
        props = backend.target[opname][qargs]
    except Exception:
        return None
    if props is None:
        return None
    return finite_float(getattr(props, "error", None))


def build_tests():
    tests = []

    qc = QuantumCircuit(2, 2)
    qc.measure(0, 0)
    qc.measure(1, 1)
    tests.append(("T0_ZERO", "|00> -> measure", qc, {"00"}, "readout baseline"))

    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    tests.append(("T1_X1", "X(q0)|00> -> |01>", qc, {"01"}, "1Q + readout"))

    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.x(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    tests.append(("T2_ONES", "X(q0)X(q1)|00> -> |11>", qc, {"11"}, "1Q + readout"))

    qc = QuantumCircuit(2, 2)
    for _ in range(N_X_STRESS):
        qc.x(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    tests.append((
        "T3_X_STRESS",
        f"X(q0)^{N_X_STRESS}|00> -> |01>",
        qc,
        {"01"},
        "amplified 1Q + readout",
    ))

    qc = QuantumCircuit(2, 2)
    for _ in range(N_CZ_STRESS):
        qc.cz(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    tests.append((
        "T4_CZ_STRESS",
        f"CZ^{N_CZ_STRESS}|00> -> |00>",
        qc,
        {"00"},
        "amplified CZ + readout",
    ))

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    tests.append((
        "T5_BELL",
        "Bell state -> only 00/11 ideally",
        qc,
        {"00", "11"},
        "1Q + entangling + readout",
    ))

    return tests


def clean_counts(counts):
    return {
        str(k).replace(" ", ""): int(v)
        for k, v in counts.items()
    }


def analyze_counts(counts, allowed):
    total = int(sum(counts.values()))
    forbidden = {
        k: int(v)
        for k, v in counts.items()
        if k not in allowed
    }
    n_forbidden = int(sum(forbidden.values()))
    frac = n_forbidden / total if total else np.nan

    return {
        "shots_returned": total,
        "forbidden_counts_json": json.dumps(forbidden, sort_keys=True),
        "n_forbidden": n_forbidden,
        "forbidden_fraction": frac,
        "nonideal_forbidden_observed": bool(n_forbidden > 0),
    }


def run_mock(alias, server_url, shots):
    if not alias.endswith(":mock"):
        raise RuntimeError(f"SAFETY STOP: refusing non-mock alias {alias!r}")

    backend = IQMBackend(
        base_url=server_url.rstrip("/"),
        qc_alias=alias,
    )

    q0, q1, cz_error = choose_test_edge(backend)
    m0_error = target_error_for(backend, "measure", (q0,))
    m1_error = target_error_for(backend, "measure", (q1,))

    print()
    print("=" * 112)
    print(f"MOCK EXECUTION: {alias}")
    print("=" * 112)
    print(f"Chosen target CZ edge: ({q0}, {q1})")
    print(f"Target CZ error:       {cz_error}")
    print(f"Target measure q{q0}:   {m0_error}")
    print(f"Target measure q{q1}:   {m1_error}")

    rows = []

    for test_id, desc, logical, allowed, focus in build_tests():
        tqc = transpile(
            logical,
            backend=backend,
            initial_layout=[q0, q1],
            optimization_level=0,
            seed_transpiler=79001,
        )

        # Final safety guard immediately before execution.
        if not alias.endswith(":mock"):
            raise RuntimeError("Internal safety guard failed.")

        job = backend.run(tqc, shots=shots)
        result = job.result()
        counts = clean_counts(result.get_counts())
        metrics = analyze_counts(counts, allowed)

        ops = {str(k): int(v) for k, v in tqc.count_ops().items()}

        try:
            job_id = str(job.job_id())
        except Exception:
            job_id = None

        print()
        print(f"{test_id}: {desc}")
        print(f"  allowed={sorted(allowed)}")
        print(f"  depth={tqc.depth()} ops={ops}")
        print(f"  counts={counts}")
        print(
            f"  forbidden={metrics['n_forbidden']}/"
            f"{metrics['shots_returned']} "
            f"({100.0 * metrics['forbidden_fraction']:.4f}%)"
        )

        rows.append({
            "mock_alias": alias,
            "test_id": test_id,
            "description": desc,
            "focus": focus,
            "physical_q0": q0,
            "physical_q1": q1,
            "target_cz_error": cz_error,
            "target_measure_error_q0": m0_error,
            "target_measure_error_q1": m1_error,
            "allowed_outcomes_json": json.dumps(sorted(allowed)),
            "counts_json": json.dumps(counts, sort_keys=True),
            "transpiled_depth": int(tqc.depth()),
            "transpiled_size": int(tqc.size()),
            "transpiled_ops_json": json.dumps(ops, sort_keys=True),
            "job_id": job_id,
            "real_qpu_job": False,
            **metrics,
        })

    return rows, target_error_rows(backend, alias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument(
        "--backend",
        choices=["all", *MOCK_ALIASES],
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.shots < 128:
        raise ValueError("--shots must be >= 128")

    if args.overwrite:
        for path in (OUT_DETAIL, OUT_SUMMARY, OUT_TARGET, OUT_ASSESSMENT):
            if path.exists():
                path.unlink()

    token, server_url, _ = get_authentication()

    aliases = list(MOCK_ALIASES) if args.backend == "all" else [args.backend]

    if not all(a.endswith(":mock") for a in aliases):
        raise RuntimeError("Only :mock aliases are permitted.")

    print("=" * 112)
    print("WEEK 12.5B — IQM MOCK EXECUTION SEMANTICS VALIDATION")
    print("=" * 112)
    print(f"Mock aliases only: {aliases}")
    print(f"Shots/circuit:     {args.shots}")
    print("Real QPU jobs:     NO")
    print("QRC/data loaded:   NO")
    print()
    print("Decision rule:")
    print("  forbidden outcome > 0 -> demonstrably non-ideal mock execution")
    print("  forbidden outcome = 0 -> no non-ideality detected by that test")
    print("  non-ideal execution != proven live-calibration-derived noise")

    detail_rows = []
    target_rows = []

    with temporary_iqm_environment(token, server_url):
        for alias in aliases:
            rows, trows = run_mock(alias, server_url, args.shots)
            detail_rows.extend(rows)
            target_rows.extend(trows)

    detail = pd.DataFrame(detail_rows)
    target = pd.DataFrame(target_rows)

    detail.to_csv(OUT_DETAIL, index=False)
    target.to_csv(OUT_TARGET, index=False)

    summary_rows = []
    assessment = {
        "stage": "12.5B",
        "shots_per_circuit": int(args.shots),
        "real_qpu_jobs": False,
        "forecast_data_loaded": False,
        "backends": {},
        "interpretation": (
            "Forbidden outcomes prove non-ideal mock execution, but do not "
            "alone prove that the mock noise is derived from current live-QPU calibration."
        ),
    }

    for alias, g in detail.groupby("mock_alias", sort=False):
        n_nonideal = int(g["nonideal_forbidden_observed"].astype(bool).sum())
        max_forbidden = float(g["forbidden_fraction"].max())

        by_test = {
            str(r["test_id"]): float(r["forbidden_fraction"])
            for _, r in g.iterrows()
        }

        zero = by_test.get("T0_ZERO", np.nan)
        x1 = by_test.get("T1_X1", np.nan)
        xs = by_test.get("T3_X_STRESS", np.nan)
        czs = by_test.get("T4_CZ_STRESS", np.nan)

        x_increment = (
            xs - x1
            if np.isfinite(xs) and np.isfinite(x1)
            else np.nan
        )
        cz_increment = (
            czs - zero
            if np.isfinite(czs) and np.isfinite(zero)
            else np.nan
        )

        status = (
            "NON_IDEAL_MOCK_EXECUTION_CONFIRMED"
            if n_nonideal > 0
            else "NO_NONIDEALITY_DETECTED_IN_THIS_TEST"
        )

        summary_rows.append({
            "mock_alias": alias,
            "n_tests": int(len(g)),
            "n_tests_with_forbidden_outcomes": n_nonideal,
            "max_forbidden_fraction": max_forbidden,
            "x_stress_minus_x1_forbidden": x_increment,
            "cz_stress_minus_zero_forbidden": cz_increment,
            "execution_semantics_status": status,
            "live_calibration_noise_link": "UNVERIFIED",
        })

        assessment["backends"][alias] = {
            "execution_semantics_status": status,
            "n_tests_with_forbidden_outcomes": n_nonideal,
            "max_forbidden_fraction": max_forbidden,
            "x_stress_minus_x1_forbidden": (
                None if not np.isfinite(x_increment) else float(x_increment)
            ),
            "cz_stress_minus_zero_forbidden": (
                None if not np.isfinite(cz_increment) else float(cz_increment)
            ),
            "live_calibration_noise_link": "UNVERIFIED",
        }

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    with open(OUT_ASSESSMENT, "w", encoding="utf-8") as f:
        json.dump(assessment, f, indent=2)

    print()
    print("=" * 112)
    print("12.5B MOCK-SEMANTICS SUMMARY")
    print("=" * 112)
    print(summary.to_string(index=False))

    print()
    print("Saved:")
    print(f"  {OUT_DETAIL}")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_TARGET}")
    print(f"  {OUT_ASSESSMENT}")
    print()
    print("No real-QPU jobs were submitted.")
    print("STOP HERE. Interpret mock semantics before any QRC noise-stage execution.")


if __name__ == "__main__":
    main()
