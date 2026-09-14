
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)
from qiskit.quantum_info import Statevector, Pauli
from qiskit_ibm_runtime import SamplerV2 as Sampler

from ibm_account import get_service


# =============================================================================
# WEEK 11.0B — FRESH-RESELECT + REAL QPU SMOKE TEST
#
# IMPORTANT
# ---------
# 11.0B does NOT blindly reuse the qubits selected by 11.0A.
#
# EVERY TIME 11.0B IS RUN:
#   1. it imports the same fresh selection logic as 11.0A,
#   2. refreshes all current backend targets and properties again,
#   3. rescans all current physical qubits,
#   4. rebuilds current CZ connectivity,
#   5. enumerates native embeddings again,
#   6. selects a current embedding,
#   7. re-transpiles the measurement circuits,
#   8. checks faults and SWAP=0,
#   9. only then can --submit send ONE SamplerV2 job.
#
# Therefore a later --submit run automatically performs a NEW hardware scan.
#
# Default = DRY RUN.
# Actual QPU execution:
#
#   python 11_0B_fresh_reselect_real_qpu_smoke.py --submit
# =============================================================================


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

SELECTION_SCRIPT = (
    HERE
    /
    "11_0A_live_embedding_reselection.py"
)

PREFIX = RESULTS / "11_0B"

OPTIMIZATION_LEVEL = 1
SEED_TRANSPILER = 42

READOUT_SETTINGS = {
    "XZ_injection": [
        "XXXXZZ",
        "ZZZZZZ",
    ],
    "XZinj_plus_YX45": [
        "XXXXYX",
        "ZZZZZZ",
    ],
    "XYZ_all": [
        "XXXXXX",
        "YYYYYY",
        "ZZZZZZ",
    ],
}


# =============================================================================
# LOAD THE EXACT 11.0A FRESH-SELECTION LOGIC
# =============================================================================

def load_selector():
    if not SELECTION_SCRIPT.exists():
        raise FileNotFoundError(
            f"{SELECTION_SCRIPT} not found. "
            "Place 11.0A and 11.0B in the same project directory."
        )

    spec = importlib.util.spec_from_file_location(
        "fresh_selector_11_0A",
        SELECTION_SCRIPT,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None

    spec.loader.exec_module(
        module
    )

    return module


# =============================================================================
# HELPERS
# =============================================================================

def json_safe(obj):
    if obj is None:
        return None

    if isinstance(
        obj,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return obj

    if isinstance(
        obj,
        np.generic,
    ):
        return obj.item()

    if isinstance(
        obj,
        dict,
    ):
        return {
            str(k):
                json_safe(v)
            for k, v
            in obj.items()
        }

    if isinstance(
        obj,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(v)
            for v in obj
        ]

    if hasattr(
        obj,
        "isoformat",
    ):
        try:
            return obj.isoformat()
        except Exception:
            pass

    return str(
        obj
    )


def safe_service_usage(
    service,
):
    try:
        return service.usage()

    except Exception as exc:
        return {
            "available":
                False,
            "error":
                str(
                    exc
                ),
        }


# =============================================================================
# MEASUREMENT CIRCUITS
# =============================================================================

def build_measurement_circuit(
    core,
    setting,
):
    qreg = QuantumRegister(
        6,
        "q",
    )

    creg = ClassicalRegister(
        6,
        "meas",
    )

    qc = QuantumCircuit(
        qreg,
        creg,
        name=(
            f"qrc_{setting}"
        ),
    )

    qc.compose(
        core,
        qubits=list(
            qreg
        ),
        inplace=True,
    )

    for q, basis in enumerate(
        setting
    ):
        if basis == "X":
            qc.h(
                qreg[
                    q
                ]
            )

        elif basis == "Y":
            qc.sdg(
                qreg[
                    q
                ]
            )

            qc.h(
                qreg[
                    q
                ]
            )

        elif basis == "Z":
            pass

        else:
            raise ValueError(
                f"Unsupported basis {basis}"
            )

    qc.measure(
        qreg,
        creg,
    )

    return qc


# =============================================================================
# IDEAL EXPECTATION VALUES
# =============================================================================

def pauli_label(
    ops,
):
    # Qiskit string order = q5 ... q0.
    chars = [
        "I"
        for _ in range(
            6
        )
    ]

    for q, p in ops.items():
        chars[
            5
            -
            int(
                q
            )
        ] = str(
            p
        )

    return "".join(
        chars
    )


def ideal_expectations(
    core,
    readout,
):
    psi = Statevector.from_instruction(
        core
    )

    out = {}

    if readout in (
        "XZ_injection",
        "XZinj_plus_YX45",
    ):
        for q in range(
            4
        ):
            for p in (
                "X",
                "Z",
            ):
                out[
                    f"{p}{q}"
                ] = float(
                    np.real(
                        psi.expectation_value(
                            Pauli(
                                pauli_label(
                                    {
                                        q:
                                            p
                                    }
                                )
                            )
                        )
                    )
                )

        if readout == "XZinj_plus_YX45":
            out[
                "YX45"
            ] = float(
                np.real(
                    psi.expectation_value(
                        Pauli(
                            pauli_label(
                                {
                                    4:
                                        "Y",
                                    5:
                                        "X",
                                }
                            )
                        )
                    )
                )
            )

    elif readout == "XYZ_all":
        for p in (
            "X",
            "Y",
            "Z",
        ):
            for q in range(
                6
            ):
                out[
                    f"{p}{q}"
                ] = float(
                    np.real(
                        psi.expectation_value(
                            Pauli(
                                pauli_label(
                                    {
                                        q:
                                            p
                                    }
                                )
                            )
                        )
                    )
                )

    else:
        raise KeyError(
            readout
        )

    return out


# =============================================================================
# COUNTS -> EXPECTATIONS
# =============================================================================

def bit_for_qubit(
    bitstring,
    q,
):
    clean = bitstring.replace(
        " ",
        "",
    )

    # c5 ... c0
    return int(
        clean[
            -(
                int(
                    q
                )
                +
                1
            )
        ]
    )


def expectation_from_counts(
    counts,
    qubits,
):
    total = int(
        sum(
            counts.values()
        )
    )

    if total <= 0:
        raise RuntimeError(
            "Empty counts."
        )

    value = 0.0

    for bitstring, count in counts.items():
        parity = sum(
            bit_for_qubit(
                bitstring,
                q,
            )
            for q in qubits
        ) % 2

        eig = (
            1.0
            if parity == 0
            else -1.0
        )

        value += (
            eig
            *
            int(
                count
            )
        )

    return (
        value
        /
        total
    )


def measured_expectations(
    counts_by_setting,
    readout,
):
    out = {}

    if readout in (
        "XZ_injection",
        "XZinj_plus_YX45",
    ):
        x_setting = (
            "XXXXYX"
            if readout
            ==
            "XZinj_plus_YX45"
            else
            "XXXXZZ"
        )

        x_counts = counts_by_setting[
            x_setting
        ]

        z_counts = counts_by_setting[
            "ZZZZZZ"
        ]

        for q in range(
            4
        ):
            out[
                f"X{q}"
            ] = expectation_from_counts(
                x_counts,
                [
                    q
                ],
            )

            out[
                f"Z{q}"
            ] = expectation_from_counts(
                z_counts,
                [
                    q
                ],
            )

        if readout == "XZinj_plus_YX45":
            out[
                "YX45"
            ] = expectation_from_counts(
                x_counts,
                [
                    4,
                    5,
                ],
            )

    elif readout == "XYZ_all":
        by_axis = {
            "X":
                "XXXXXX",
            "Y":
                "YYYYYY",
            "Z":
                "ZZZZZZ",
        }

        for p, setting in by_axis.items():
            counts = counts_by_setting[
                setting
            ]

            for q in range(
                6
            ):
                out[
                    f"{p}{q}"
                ] = expectation_from_counts(
                    counts,
                    [
                        q
                    ],
                )

    return out


# =============================================================================
# RESOURCE AUDIT
# =============================================================================

def resource_summary(
    circuit,
    backend,
):
    counts = {
        str(k):
            int(
                v
            )
        for k, v
        in circuit.count_ops().items()
    }

    duration_s = float(
        circuit.estimate_duration(
            backend.target,
            unit="s",
        )
    )

    return {
        "depth":
            int(
                circuit.depth()
            ),
        "size":
            int(
                circuit.size()
            ),
        "n_cz":
            int(
                counts.get(
                    "cz",
                    0,
                )
            ),
        "n_swap":
            int(
                counts.get(
                    "swap",
                    0,
                )
            ),
        "n_measure":
            int(
                counts.get(
                    "measure",
                    0,
                )
            ),
        "duration_us":
            duration_s
            *
            1e6,
        "operations":
            json.dumps(
                counts,
                sort_keys=True,
            ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Actually submit one QPU job. "
            "Without this flag the full fresh scan is only a dry-run."
        ),
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Shots per grouped measurement setting.",
    )

    parser.add_argument(
        "--overall-rank",
        type=int,
        default=1,
        help=(
            "Choose rank N from the FRESH scan performed in this same run."
        ),
    )

    parser.add_argument(
        "--shortlist",
        type=int,
        default=80,
        help="Current embeddings compiled per backend after calibration screen.",
    )

    args = parser.parse_args()

    if args.shots < 1:
        raise ValueError(
            "shots must be >= 1"
        )

    selector = load_selector()

    candidate = selector.select_best_candidate()

    service = get_service()

    print("=" * 120)
    print("WEEK 11.0B — FRESH HARDWARE RESCAN BEFORE QPU")
    print("=" * 120)
    print(
        "11.0B is NOT reading yesterday's physical-qubit selection."
    )
    print(
        "It is refreshing all three backends and rescanning the full current "
        "CZ graph now."
    )
    print()

    scan = selector.fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=selector.BACKENDS,
        shortlist=args.shortlist,
        write_prefix="11_0B_preflight",
        verbose=True,
    )

    top = scan[
        "top"
    ].copy()

    if (
        args.overall_rank
        not in
        top[
            "overall_rank"
        ].astype(
            int
        ).tolist()
    ):
        raise ValueError(
            f"Requested fresh overall rank {args.overall_rank} does not exist."
        )

    selected = top[
        top[
            "overall_rank"
        ]
        ==
        args.overall_rank
    ].iloc[
        0
    ]

    backend_name = str(
        selected[
            "backend"
        ]
    )

    layout = json.loads(
        selected[
            "layout"
        ]
    )

    # Fetch the backend object AFTER the full scan.
    backend = service.backend(
        backend_name,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    properties = backend.properties(
        refresh=True
    )

    if properties is None:
        raise RuntimeError(
            f"{backend_name}: properties unavailable immediately before execution."
        )

    status = backend.status()

    if not bool(
        status.operational
    ):
        raise RuntimeError(
            f"{backend_name} became non-operational after reselection."
        )

    # Rebuild core with the same candidate and first input.
    angles, input_audit = (
        selector.first_input_angles()
    )

    core = selector.build_core(
        candidate,
        angles,
    )

    settings = READOUT_SETTINGS[
        candidate[
            "readout"
        ]
    ]

    logical_circuits = [
        build_measurement_circuit(
            core,
            setting,
        )
        for setting in settings
    ]

    isa_circuits = transpile(
        logical_circuits,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
        scheduling_method="alap",
    )

    if not isinstance(
        isa_circuits,
        list,
    ):
        isa_circuits = [
            isa_circuits
        ]

    audit_rows = []
    estimated_all_shots_s = 0.0

    for setting, circuit in zip(
        settings,
        isa_circuits,
    ):
        # Current faulty check after the second refresh.
        backend.check_faulty(
            circuit
        )

        r = resource_summary(
            circuit,
            backend,
        )

        if r[
            "n_swap"
        ] != 0:
            raise RuntimeError(
                f"{setting}: SWAP={r['n_swap']}; refusing to submit."
            )

        r[
            "measurement_setting"
        ] = setting

        r[
            "estimated_shot_time_s"
        ] = (
            r[
                "duration_us"
            ]
            *
            1e-6
            *
            args.shots
        )

        estimated_all_shots_s += (
            r[
                "estimated_shot_time_s"
            ]
        )

        audit_rows.append(
            r
        )

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df.to_csv(
        RESULTS
        /
        "11_0B_transpilation.csv",
        index=False,
    )

    ideal = ideal_expectations(
        core,
        candidate[
            "readout"
        ],
    )

    usage_before = safe_service_usage(
        service
    )

    preflight = {
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "mode":
            (
                "SUBMIT"
                if args.submit
                else
                "DRY_RUN"
            ),
        "candidate_id":
            candidate[
                "candidate_id"
            ],
        "topology":
            candidate[
                "topology"
            ],
        "r":
            candidate[
                "r"
            ],
        "hx":
            candidate[
                "hx"
            ],
        "hy":
            candidate[
                "hy"
            ],
        "readout":
            candidate[
                "readout"
            ],
        "cv_rmse":
            candidate[
                "cv_rmse"
            ],
        "fresh_overall_rank":
            int(
                args.overall_rank
            ),
        "backend":
            backend_name,
        "layout":
            layout,
        "pending_jobs":
            int(
                status.pending_jobs
            ),
        "fresh_selected_metrics":
            json_safe(
                selected.to_dict()
            ),
        "input":
            input_audit,
        "shots_per_setting":
            args.shots,
        "settings":
            settings,
        "estimated_scheduled_circuit_time_all_shots_s":
            estimated_all_shots_s,
        "service_usage_before":
            json_safe(
                usage_before
            ),
    }

    with open(
        RESULTS
        /
        "11_0B_preflight.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(
                preflight
            ),
            fp,
            indent=2,
        )

    print()
    print("=" * 120)
    print("FRESH SELECTION FOR THIS RUN")
    print("=" * 120)

    print(
        top[
            [
                "overall_rank",
                "backend",
                "backend_rank",
                "layout",
                "pending_jobs",
                "compiled_2q_error_max_percent",
                "max_readout_error_percent",
                "compiled_1q_error_max_percent",
                "min_t2_us",
                "compiled_duration_us",
                "R_T2",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Selected NOW: backend={backend_name}, layout={layout}"
    )

    print(
        f"Estimated scheduled circuit-time sum × shots "
        f"(not IBM billed usage): {estimated_all_shots_s:.6f} s"
    )

    print()

    print(
        audit_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Service usage before:"
    )

    print(
        json.dumps(
            json_safe(
                usage_before
            ),
            indent=2,
        )
    )

    if not args.submit:
        print()
        print(
            "DRY RUN COMPLETE — NO QPU JOB SUBMITTED."
        )
        print(
            "Running again with --submit will perform ANOTHER complete fresh "
            "hardware scan before submission."
        )
        return

    # =========================================================================
    # REAL QPU JOB
    # =========================================================================

    sampler = Sampler(
        mode=backend
    )

    print()
    print(
        "Submitting one SamplerV2 job in JOB MODE..."
    )

    job = sampler.run(
        isa_circuits,
        shots=int(
            args.shots
        ),
    )

    print(
        f"Job ID: {job.job_id()}"
    )

    result = job.result()

    print(
        f"Final status: {job.status()}"
    )

    counts_by_setting = {}
    count_rows = []

    for setting, pub_result in zip(
        settings,
        result,
    ):
        counts = (
            pub_result
            .data
            .meas
            .get_counts()
        )

        counts_by_setting[
            setting
        ] = counts

        for bitstring, count in counts.items():
            count_rows.append({
                "measurement_setting":
                    setting,
                "bitstring":
                    bitstring,
                "count":
                    int(
                        count
                    ),
            })

    pd.DataFrame(
        count_rows
    ).to_csv(
        RESULTS
        /
        "11_0B_counts.csv",
        index=False,
    )

    hardware = measured_expectations(
        counts_by_setting,
        candidate[
            "readout"
        ],
    )

    obs_rows = []

    for observable in sorted(
        ideal
    ):
        iv = float(
            ideal[
                observable
            ]
        )

        hv = float(
            hardware[
                observable
            ]
        )

        obs_rows.append({
            "observable":
                observable,
            "ideal":
                iv,
            "hardware":
                hv,
            "error_hardware_minus_ideal":
                hv
                -
                iv,
            "absolute_error":
                abs(
                    hv
                    -
                    iv
                ),
        })

    obs_df = pd.DataFrame(
        obs_rows
    )

    obs_df.to_csv(
        RESULTS
        /
        "11_0B_observables.csv",
        index=False,
    )

    try:
        job_usage = float(
            job.usage(
                partial=False
            )
        )
    except Exception as exc:
        job_usage = np.nan
        job_usage_error = str(
            exc
        )
    else:
        job_usage_error = ""

    try:
        metrics = job.metrics()
    except Exception as exc:
        metrics = {
            "available":
                False,
            "error":
                str(
                    exc
                ),
        }

    usage_after = safe_service_usage(
        service
    )

    usage_record = {
        "job_id":
            job.job_id(),
        "backend":
            backend_name,
        "layout":
            layout,
        "fresh_hardware_scan":
            True,
        "job_usage_seconds":
            job_usage,
        "job_usage_error":
            job_usage_error,
        "job_metrics":
            json_safe(
                metrics
            ),
        "service_usage_before":
            json_safe(
                usage_before
            ),
        "service_usage_after":
            json_safe(
                usage_after
            ),
        "estimated_scheduled_circuit_time_all_shots_s":
            estimated_all_shots_s,
    }

    with open(
        RESULTS
        /
        "11_0B_usage.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(
                usage_record
            ),
            fp,
            indent=2,
        )

    summary = {
        **preflight,
        "job_id":
            job.job_id(),
        "job_usage_seconds":
            job_usage,
        "service_usage_after":
            json_safe(
                usage_after
            ),
        "mean_absolute_observable_error":
            float(
                obs_df[
                    "absolute_error"
                ].mean()
            ),
        "max_absolute_observable_error":
            float(
                obs_df[
                    "absolute_error"
                ].max()
            ),
    }

    with open(
        RESULTS
        /
        "11_0B_summary.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_safe(
                summary
            ),
            fp,
            indent=2,
        )

    print()
    print("=" * 120)
    print("REAL QPU OBSERVABLES")
    print("=" * 120)

    print(
        obs_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("ACTUAL USAGE")
    print("=" * 120)

    print(
        f"job.usage(partial=False): {job_usage} s"
    )

    print(
        json.dumps(
            json_safe(
                metrics
            ),
            indent=2,
        )
    )

    print()

    print(
        "Service usage after:"
    )

    print(
        json.dumps(
            json_safe(
                usage_after
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
