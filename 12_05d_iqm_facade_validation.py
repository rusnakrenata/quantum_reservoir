"""
Week 12.5D — IQMFacadeBackend validation.

Purpose
-------
Test IQM's facade backend exactly as documented:

    remote :mock submission/integration validation
                  +
    local Qiskit Aer noisy simulation

The raw random bits returned by the remote mock server are discarded by
IQMFacadeBackend and replaced with the local noisy-simulation result.

This step answers:
1. Does a facade backend instantiate for garnet:mock and emerald:mock?
2. Which predefined facade profile is selected?
3. Does facade execution behave physically, unlike raw :mock execution?
4. Is the selected profile Garnet-specific / generic 54-qubit / other?

IMPORTANT
---------
- ONLY :mock environments are contacted.
- NO real-QPU jobs.
- Tiny diagnostic circuits only.
- NO QRC.
- NO forecasting data.
- 2026 untouched.
- Authentication comes only from iqm_account.py.

Current IQM docs:
- facade_garnet is specific to the Garnet Resonance architecture.
- facade_aphrodite is a representative Crystal-54 facade.
- IQMFacadeBackend(client, name=None) may choose a compatible fake backend
  automatically from the mock server's static architecture.

Outputs
-------
results/12_05d_iqm_facade_probe.csv
results/12_05d_iqm_facade_execution.csv
results/12_05d_iqm_facade_assessment.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qiskit import QuantumCircuit, transpile

from iqm.iqm_client import IQMClient
from iqm.qiskit_iqm.iqm_provider import (
    IQMProvider,
    IQMFacadeBackend,
)

from iqm_account import get_authentication


RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

OUT_PROBE = RESULTS / "12_05d_iqm_facade_probe.csv"
OUT_EXEC = RESULTS / "12_05d_iqm_facade_execution.csv"
OUT_JSON = RESULTS / "12_05d_iqm_facade_assessment.json"

MOCKS = {
    "garnet": "garnet:mock",
    "emerald": "emerald:mock",
}

# Explicit documented names worth probing.
NAMED_FACADES = {
    "garnet": ["facade_garnet", "facade_apollo"],
    "emerald": ["facade_aphrodite"],
}

DEFAULT_SHOTS = 2048


def clean_counts(counts):
    return {
        str(k).replace(" ", ""): int(v)
        for k, v in counts.items()
    }


def build_tests():
    tests = []

    # |0> -> 0
    qc = QuantumCircuit(1)
    qc.measure_all()
    tests.append({
        "test_id": "ZERO",
        "description": "|0> -> 0",
        "circuit": qc,
        "allowed": {"0"},
    })

    # X|0> -> 1
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    tests.append({
        "test_id": "ONE",
        "description": "X|0> -> 1",
        "circuit": qc,
        "allowed": {"1"},
    })

    # Bell -> 00 or 11
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    tests.append({
        "test_id": "BELL",
        "description": "Bell -> {00,11}",
        "circuit": qc,
        "allowed": {"00", "11"},
    })

    return tests


def forbidden_stats(counts, allowed):
    total = sum(counts.values())
    forbidden = {
        k: v
        for k, v in counts.items()
        if k not in allowed
    }
    n_forbidden = sum(forbidden.values())

    return {
        "shots_returned": int(total),
        "n_forbidden": int(n_forbidden),
        "forbidden_fraction":
            float(n_forbidden / total)
            if total
            else None,
        "forbidden_counts_json":
            json.dumps(
                forbidden,
                sort_keys=True,
            ),
    }


def safe_backend_metadata(backend):
    """
    Collect only public/stable-ish information plus a limited attribute
    inventory. We do not depend on private attributes for the experiment.
    """
    data = {
        "backend_class":
            f"{type(backend).__module__}.{type(backend).__name__}",
        "backend_name":
            str(getattr(backend, "name", None)),
        "num_qubits":
            getattr(backend, "num_qubits", None),
        "description":
            str(getattr(backend, "description", None)),
    }

    try:
        data["operation_names"] = sorted(
            str(x)
            for x in backend.target.operation_names
        )
    except Exception:
        data["operation_names"] = None

    # Useful only as diagnostic evidence if publicly exposed.
    for attr in (
        "error_profile",
        "fake_backend",
    ):
        if hasattr(backend, attr):
            try:
                obj = getattr(backend, attr)
                data[attr] = str(obj)
            except Exception:
                data[attr] = "<unreadable>"

    return data


def make_client(server_url, token, mock_alias):
    if not mock_alias.endswith(":mock"):
        raise RuntimeError(
            f"SAFETY STOP: {mock_alias!r} is not a :mock alias."
        )

    return IQMClient(
        server_url,
        quantum_computer=mock_alias,
        token=token,
    )


def make_provider(server_url, token, mock_alias):
    if not mock_alias.endswith(":mock"):
        raise RuntimeError(
            f"SAFETY STOP: {mock_alias!r} is not a :mock alias."
        )

    return IQMProvider(
        server_url,
        quantum_computer=mock_alias,
        token=token,
    )


def probe_backends(server_url, token):
    rows = []
    instances = {}

    for real_alias, mock_alias in MOCKS.items():
        print()
        print("=" * 116)
        print(
            f"MOCK ENVIRONMENT: {mock_alias}"
        )
        print("=" * 116)

        client = make_client(
            server_url,
            token,
            mock_alias,
        )

        # ------------------------------------------------------------------
        # A. Direct automatic IQMFacadeBackend selection.
        # ------------------------------------------------------------------
        key = (
            real_alias,
            "auto",
        )

        try:
            backend = IQMFacadeBackend(
                client,
                name=None,
            )

            meta = safe_backend_metadata(
                backend
            )

            rows.append({
                "real_alias":
                    real_alias,
                "mock_alias":
                    mock_alias,
                "construction":
                    "IQMFacadeBackend(client,name=None)",
                "requested_facade":
                    "AUTO",
                "success":
                    True,
                "backend_name":
                    meta["backend_name"],
                "backend_class":
                    meta["backend_class"],
                "num_qubits":
                    meta["num_qubits"],
                "error":
                    None,
            })

            instances[key] = backend

            print(
                "AUTO facade: YES | "
                f"name={meta['backend_name']} | "
                f"qubits={meta['num_qubits']}"
            )

        except Exception as exc:
            rows.append({
                "real_alias":
                    real_alias,
                "mock_alias":
                    mock_alias,
                "construction":
                    "IQMFacadeBackend(client,name=None)",
                "requested_facade":
                    "AUTO",
                "success":
                    False,
                "backend_name":
                    None,
                "backend_class":
                    None,
                "num_qubits":
                    None,
                "error":
                    repr(exc),
            })

            print(
                f"AUTO facade: NO | {exc!r}"
            )

        # ------------------------------------------------------------------
        # B. Explicit documented provider facade names.
        # ------------------------------------------------------------------
        provider = make_provider(
            server_url,
            token,
            mock_alias,
        )

        for facade_name in NAMED_FACADES[
            real_alias
        ]:
            key = (
                real_alias,
                facade_name,
            )

            try:
                backend = provider.get_backend(
                    name=facade_name
                )

                meta = safe_backend_metadata(
                    backend
                )

                rows.append({
                    "real_alias":
                        real_alias,
                    "mock_alias":
                        mock_alias,
                    "construction":
                        "IQMProvider.get_backend(name=...)",
                    "requested_facade":
                        facade_name,
                    "success":
                        True,
                    "backend_name":
                        meta["backend_name"],
                    "backend_class":
                        meta["backend_class"],
                    "num_qubits":
                        meta["num_qubits"],
                    "error":
                        None,
                })

                instances[key] = backend

                print(
                    f"{facade_name}: YES | "
                    f"name={meta['backend_name']} | "
                    f"qubits={meta['num_qubits']}"
                )

            except Exception as exc:
                rows.append({
                    "real_alias":
                        real_alias,
                    "mock_alias":
                        mock_alias,
                    "construction":
                        "IQMProvider.get_backend(name=...)",
                    "requested_facade":
                        facade_name,
                    "success":
                        False,
                    "backend_name":
                        None,
                    "backend_class":
                        None,
                    "num_qubits":
                        None,
                    "error":
                        repr(exc),
                })

                print(
                    f"{facade_name}: NO | {exc!r}"
                )

    return pd.DataFrame(
        rows
    ), instances


def choose_execution_backend(
    real_alias,
    instances,
):
    """
    Prefer the specific Garnet facade when available.
    For Emerald prefer AUTO, because it lets IQM choose a compatible
    representative based on the mock static architecture.
    """
    if real_alias == "garnet":
        preference = [
            (real_alias, "facade_garnet"),
            (real_alias, "auto"),
            (real_alias, "facade_apollo"),
        ]
    else:
        preference = [
            (real_alias, "auto"),
            (real_alias, "facade_aphrodite"),
        ]

    for key in preference:
        if key in instances:
            return key, instances[key]

    return None, None


def execute_diagnostics(
    real_alias,
    mock_alias,
    backend_key,
    backend,
    shots,
):
    if not mock_alias.endswith(":mock"):
        raise RuntimeError(
            "SAFETY STOP: facade execution must use :mock."
        )

    rows = []

    print()
    print(
        f"DIAGNOSTIC FACADE EXECUTION: "
        f"{real_alias} via {backend_key[1]}"
    )

    for test in build_tests():
        tqc = transpile(
            test["circuit"],
            backend=backend,
            optimization_level=0,
            seed_transpiler=79001,
        )

        # Hard guard immediately before run().
        if not mock_alias.endswith(":mock"):
            raise RuntimeError(
                "Internal safety guard failed."
            )

        job = backend.run(
            tqc,
            shots=shots,
        )

        result = job.result()
        counts = clean_counts(
            result.get_counts()
        )

        stats = forbidden_stats(
            counts,
            test["allowed"],
        )

        print(
            f"  {test['test_id']:5s} "
            f"counts={counts} | "
            f"forbidden={stats['n_forbidden']}/"
            f"{stats['shots_returned']} "
            f"({100.0 * stats['forbidden_fraction']:.3f}%)"
        )

        rows.append({
            "real_alias":
                real_alias,
            "mock_alias":
                mock_alias,
            "facade_selection":
                backend_key[1],
            "backend_name":
                str(
                    getattr(
                        backend,
                        "name",
                        None,
                    )
                ),
            "test_id":
                test["test_id"],
            "description":
                test["description"],
            "allowed_json":
                json.dumps(
                    sorted(
                        test["allowed"]
                    )
                ),
            "counts_json":
                json.dumps(
                    counts,
                    sort_keys=True,
                ),
            "transpiled_depth":
                int(
                    tqc.depth()
                ),
            "transpiled_ops_json":
                json.dumps(
                    {
                        str(k): int(v)
                        for k, v
                        in tqc.count_ops().items()
                    },
                    sort_keys=True,
                ),
            **stats,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.shots < 128:
        raise ValueError(
            "--shots must be >= 128"
        )

    if args.overwrite:
        for p in (
            OUT_PROBE,
            OUT_EXEC,
            OUT_JSON,
        ):
            if p.exists():
                p.unlink()

    token, server_url, _ = (
        get_authentication()
    )

    print("=" * 116)
    print(
        "WEEK 12.5D — IQM FACADE BACKEND VALIDATION"
    )
    print("=" * 116)
    print(
        "Remote environments: :mock ONLY"
    )
    print(
        "Local layer: IQM facade noisy Aer simulation"
    )
    print(
        "Real QPU jobs: NO"
    )
    print(
        "QRC / forecasting data: NOT LOADED"
    )
    print(
        f"Shots/diagnostic circuit: {args.shots}"
    )

    probe, instances = probe_backends(
        server_url,
        token,
    )

    probe.to_csv(
        OUT_PROBE,
        index=False,
    )

    execution_rows = []
    selected = {}

    for real_alias, mock_alias in MOCKS.items():
        key, backend = (
            choose_execution_backend(
                real_alias,
                instances,
            )
        )

        if backend is None:
            print()
            print(
                f"{real_alias}: NO usable facade; "
                "diagnostic execution skipped."
            )
            selected[
                real_alias
            ] = None
            continue

        selected[
            real_alias
        ] = {
            "selection_key":
                key[1],
            "backend_name":
                str(
                    getattr(
                        backend,
                        "name",
                        None,
                    )
                ),
            "backend_class":
                f"{type(backend).__module__}.{type(backend).__name__}",
        }

        execution_rows.extend(
            execute_diagnostics(
                real_alias,
                mock_alias,
                key,
                backend,
                args.shots,
            )
        )

    execution = pd.DataFrame(
        execution_rows
    )

    execution.to_csv(
        OUT_EXEC,
        index=False,
    )

    assessment = {
        "stage":
            "12.5D",
        "real_qpu_jobs":
            0,
        "mock_remote_environments":
            list(
                MOCKS.values()
            ),
        "shots_per_diagnostic":
            int(
                args.shots
            ),
        "qrc_loaded":
            False,
        "forecast_data_loaded":
            False,
        "test_2026_used":
            False,
        "selected_facades":
            selected,
        "methodological_note": (
            "IQMFacadeBackend validates the remote mock submission path and "
            "then replaces the mock server's random result with a local Qiskit "
            "Aer noisy simulation. IQM documents the facade noise model as "
            "broadly representative of the mocked QPU; this step therefore "
            "does not yet establish use of the current live calibration snapshot."
        ),
        "next_decision": (
            "If facade execution is valid, inspect the facade/fake error "
            "profile and decide whether the 40-candidate noisy ladder should "
            "use the representative facade profile or a custom IQMErrorProfile "
            "constructed from the frozen live Emerald/Garnet calibration."
        ),
    }

    with open(
        OUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            assessment,
            f,
            indent=2,
        )

    print()
    print("=" * 116)
    print(
        "12.5D SUMMARY"
    )
    print("=" * 116)

    if execution.empty:
        print(
            "No facade diagnostics executed."
        )
    else:
        show = (
            execution.groupby(
                [
                    "real_alias",
                    "facade_selection",
                    "backend_name",
                ],
                as_index=False,
            )
            .agg(
                n_tests=(
                    "test_id",
                    "count",
                ),
                max_forbidden_fraction=(
                    "forbidden_fraction",
                    "max",
                ),
            )
        )

        print(
            show.to_string(
                index=False
            )
        )

    print()
    print("Saved:")
    print(f"  {OUT_PROBE}")
    print(f"  {OUT_EXEC}")
    print(f"  {OUT_JSON}")

    print()
    print(
        "STOP HERE. Send this output before using a facade/noise profile "
        "for the 40-candidate QRC sweep."
    )


if __name__ == "__main__":
    main()
