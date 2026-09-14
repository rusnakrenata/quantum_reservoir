from __future__ import annotations

"""
WEEK 11.1D.5 — CANDIDATE #4 MEMORY-QUBIT COHERENCE CHECK
=========================================================

This is the final zero-cost hardware-feasibility check for Candidate #4.

It does NOT submit a QPU job.

It:
  1. loads the frozen Candidate #4 RWP(117) hardware preflight,
  2. extracts the selected physical layout,
  3. identifies the two physical memory qubits corresponding to logical q4,q5,
  4. refreshes the current IBM backend calibration,
  5. reads T1/T2 specifically for those memory qubits,
  6. compares the already-compiled RWP(117) max-setting duration against
     each memory qubit's T1/T2,
  7. stores CSV/JSON + database records.

The purpose is to distinguish:
  - conservative region-wide min(T1/T2), from
  - the coherence times of the qubits that actually have to preserve memory
    through all 117 replay steps.

NO QPU JOB.
2026 IS NOT LOADED.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ibm_account import get_service
import db_objects as db


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

PREFLIGHT_JSON = (
    RESULTS
    / "11_1D4_candidate4_RWP117_preflight_summary.json"
)

OUT_CSV = (
    RESULTS
    / "11_1D5_candidate4_memory_qubit_coherence.csv"
)

OUT_JSON = (
    RESULTS
    / "11_1D5_candidate4_memory_qubit_coherence_summary.json"
)

CANDIDATE_KEY = "CONT_H0_R4"
CHECK_KEY = "CONT_H0_R4_RWP117_MEMORY_COHERENCE"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required preflight artifact not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def seconds_to_us(value):
    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value * 1e6


def get_qubit_t1_t2_us(backend, qubit: int):
    """
    Prefer BackendV2 target qubit properties.

    Fallback to backend.properties() if available.
    """
    t1_us = None
    t2_us = None
    source = None

    # --------------------------------------------------------------
    # BackendV2 / Target path.
    # --------------------------------------------------------------
    try:
        props = backend.target.qubit_properties

        if props is not None:
            qp = props[int(qubit)]

            if qp is not None:
                if getattr(qp, "t1", None) is not None:
                    t1_us = seconds_to_us(
                        qp.t1
                    )

                if getattr(qp, "t2", None) is not None:
                    t2_us = seconds_to_us(
                        qp.t2
                    )

                if (
                    t1_us is not None
                    or t2_us is not None
                ):
                    source = (
                        "backend.target.qubit_properties"
                    )
    except Exception:
        pass

    # --------------------------------------------------------------
    # BackendProperties fallback.
    # --------------------------------------------------------------
    if (
        t1_us is None
        or t2_us is None
    ):
        try:
            props = backend.properties()

            if props is not None:
                if t1_us is None:
                    t1_us = float(
                        props.t1(
                            int(qubit)
                        )
                    ) * 1e6

                if t2_us is None:
                    t2_us = float(
                        props.t2(
                            int(qubit)
                        )
                    ) * 1e6

                source = (
                    source
                    or "backend.properties"
                )
        except Exception:
            pass

    if (
        t1_us is None
        and t2_us is None
    ):
        raise RuntimeError(
            f"Could not obtain T1/T2 for physical qubit {qubit}."
        )

    return {
        "physical_qubit": int(
            qubit
        ),
        "t1_us": (
            None
            if t1_us is None
            else float(
                t1_us
            )
        ),
        "t2_us": (
            None
            if t2_us is None
            else float(
                t2_us
            )
        ),
        "source": source,
    }


def classify_ratio(ratio):
    if ratio is None:
        return "unknown"

    ratio = float(
        ratio
    )

    if ratio < 0.25:
        return "comfortable"
    if ratio < 0.50:
        return "moderate"
    if ratio < 1.00:
        return "high"
    return "beyond_one_coherence_time"


def main():
    print(
        "=" * 126
    )
    print(
        "WEEK 11.1D.5 — CANDIDATE #4 MEMORY-QUBIT COHERENCE CHECK"
    )
    print(
        "=" * 126
    )
    print(
        "NO QPU JOB WILL BE SUBMITTED."
    )
    print()

    preflight = load_json(
        PREFLIGHT_JSON
    )

    if preflight.get(
        "candidate_key"
    ) != CANDIDATE_KEY:
        raise RuntimeError(
            "Unexpected candidate in preflight JSON."
        )

    layout = [
        int(x)
        for x
        in preflight[
            "layout"
        ]
    ]

    if len(layout) != 6:
        raise RuntimeError(
            f"Expected 6-qubit layout, got {layout}"
        )

    backend_name = str(
        preflight[
            "backend"
        ]
    )

    # logical q4, q5 are the two persistent memory qubits.
    memory_mapping = {
        "M1_logical_q4": int(
            layout[4]
        ),
        "M2_logical_q5": int(
            layout[5]
        ),
    }

    actual = preflight[
        "actual_compiled_sample"
    ]

    setting_duration_us = float(
        actual[
            "median_max_setting_duration_us"
        ]
    )

    feature_duration_us = float(
        actual[
            "median_feature_duration_us"
        ]
    )

    print(
        "Frozen hardware preflight:"
    )
    print(
        f"  backend={backend_name}"
    )
    print(
        f"  layout={layout}"
    )
    print(
        f"  memory q4 -> P{memory_mapping['M1_logical_q4']}"
    )
    print(
        f"  memory q5 -> P{memory_mapping['M2_logical_q5']}"
    )
    print(
        f"  actual max-setting duration="
        f"{setting_duration_us:.3f} us"
    )
    print(
        f"  actual feature duration="
        f"{feature_duration_us:.3f} us"
    )
    print()

    service = get_service()

    # Fresh backend/calibration snapshot.
    backend = service.backend(
        backend_name
    )

    rows = []

    for logical_name, physical_qubit in memory_mapping.items():
        qp = get_qubit_t1_t2_us(
            backend,
            physical_qubit,
        )

        t1_us = qp[
            "t1_us"
        ]
        t2_us = qp[
            "t2_us"
        ]

        ratio_t1 = (
            None
            if t1_us is None
            else setting_duration_us
            / t1_us
        )

        ratio_t2 = (
            None
            if t2_us is None
            else setting_duration_us
            / t2_us
        )

        # Simple exponential survival factors are only diagnostics,
        # NOT a prediction of full circuit fidelity.
        exp_t1 = (
            None
            if (
                t1_us is None
                or t1_us <= 0
            )
            else float(
                math.exp(
                    -setting_duration_us
                    / t1_us
                )
            )
        )

        exp_t2 = (
            None
            if (
                t2_us is None
                or t2_us <= 0
            )
            else float(
                math.exp(
                    -setting_duration_us
                    / t2_us
                )
            )
        )

        rows.append(
            {
                "logical_memory_qubit": (
                    logical_name
                ),
                "physical_qubit": int(
                    physical_qubit
                ),
                "backend": backend_name,
                "calibration_source": (
                    qp[
                        "source"
                    ]
                ),
                "max_setting_duration_us": (
                    setting_duration_us
                ),
                "feature_duration_us": (
                    feature_duration_us
                ),
                "t1_us": t1_us,
                "t2_us": t2_us,
                "duration_over_t1": (
                    ratio_t1
                ),
                "duration_over_t2": (
                    ratio_t2
                ),
                "t1_classification": (
                    classify_ratio(
                        ratio_t1
                    )
                ),
                "t2_classification": (
                    classify_ratio(
                        ratio_t2
                    )
                ),
                "exp_minus_t_over_t1_diagnostic": (
                    exp_t1
                ),
                "exp_minus_t_over_t2_diagnostic": (
                    exp_t2
                ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUT_CSV,
        index=False,
    )

    print(
        "MEMORY-QUBIT COHERENCE"
    )
    print(
        "-" * 126
    )

    show_cols = [
        "logical_memory_qubit",
        "physical_qubit",
        "t1_us",
        "t2_us",
        "duration_over_t1",
        "duration_over_t2",
        "t1_classification",
        "t2_classification",
    ]

    print(
        df[
            show_cols
        ].to_string(
            index=False
        )
    )

    min_mem_t1 = float(
        df[
            "t1_us"
        ].dropna().min()
    )

    min_mem_t2 = float(
        df[
            "t2_us"
        ].dropna().min()
    )

    worst_mem_r_t1 = float(
        df[
            "duration_over_t1"
        ].dropna().max()
    )

    worst_mem_r_t2 = float(
        df[
            "duration_over_t2"
        ].dropna().max()
    )

    print()
    print(
        "MEMORY-SPECIFIC SUMMARY"
    )
    print(
        "-" * 126
    )
    print(
        f"Minimum memory-qubit T1:      "
        f"{min_mem_t1:.3f} us"
    )
    print(
        f"Minimum memory-qubit T2:      "
        f"{min_mem_t2:.3f} us"
    )
    print(
        f"Worst setting-duration/T1:    "
        f"{worst_mem_r_t1:.3f}"
    )
    print(
        f"Worst setting-duration/T2:    "
        f"{worst_mem_r_t2:.3f}"
    )

    physically_faithful = bool(
        worst_mem_r_t1 < 1.0
        and worst_mem_r_t2 < 1.0
    )

    print()
    print(
        "FINAL HARDWARE-FEASIBILITY FLAG"
    )
    print(
        "-" * 126
    )

    if physically_faithful:
        print(
            "Memory-specific T1/T2 ratios are below 1."
        )
        print(
            "This does NOT prove high fidelity, but the "
            "coherence-time test alone does not rule out "
            "a hardware pilot."
        )
    else:
        print(
            "NOT PHYSICALLY FAITHFUL FOR A FULL COMPETITIVE RUN."
        )
        print(
            "At least one persistent memory qubit must retain "
            "state for longer than its T1 or T2 during a single "
            "117-step measurement-setting circuit."
        )
        print(
            "Candidate #4 should therefore remain an ideal/"
            "simulation reference, with at most a small QPU "
            "demonstration pilot if desired."
        )

    summary = {
        "candidate_key": (
            CANDIDATE_KEY
        ),
        "check_key": (
            CHECK_KEY
        ),
        "backend": (
            backend_name
        ),
        "layout": (
            layout
        ),
        "memory_mapping": (
            memory_mapping
        ),
        "K_RWP": 117,
        "max_setting_duration_us": (
            setting_duration_us
        ),
        "feature_duration_us": (
            feature_duration_us
        ),
        "memory_qubits": (
            db.json_safe(
                rows
            )
        ),
        "minimum_memory_t1_us": (
            min_mem_t1
        ),
        "minimum_memory_t2_us": (
            min_mem_t2
        ),
        "worst_setting_duration_over_memory_t1": (
            worst_mem_r_t1
        ),
        "worst_setting_duration_over_memory_t2": (
            worst_mem_r_t2
        ),
        "coherence_time_test_passes_below_one": (
            physically_faithful
        ),
        "interpretation": (
            "This is a necessary but not sufficient hardware "
            "feasibility test. The exponential factors are "
            "diagnostic only and are not full-circuit fidelity "
            "predictions."
        ),
        "qpu_submitted": False,
        "2026_loaded": False,
    }

    OUT_JSON.write_text(
        json.dumps(
            db.json_safe(
                summary
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Database persistence.
    # --------------------------------------------------------------
    db.create_all_tables()

    run_uuid = db.create_run(
        script_name=(
            Path(
                __file__
            ).name
        ),
        run_status=(
            "MEMORY_COHERENCE_CHECK_COMPLETE"
        ),
        started_at_utc=(
            db.utc_now_naive()
        ),
        completed_at_utc=(
            db.utc_now_naive()
        ),
        forecast_dataset_name=(
            "property_damage_next_day_v1"
        ),
        feature_set_name="F4",
        evaluation_split=(
            "validation_2025"
        ),
        candidate_key=(
            CHECK_KEY
        ),
        selected_by_rules=(
            "Rule4"
        ),
        protocol=(
            "RWP_from_CONT"
        ),
        topology="H0",
        window_size=117,
        trotter_r=4,
        readout_name=(
            "XZinj_plus_YX45"
        ),
        primitive_name=(
            "calibration_only"
        ),
        measurement_method=(
            "memory_qubit_T1_T2_check"
        ),
        backend_name=(
            backend_name
        ),
        physical_layout_json=(
            layout
        ),
        min_t1_us=(
            min_mem_t1
        ),
        min_t2_us=(
            min_mem_t2
        ),
        compiled_duration_us=(
            setting_duration_us
        ),
        r_t2=(
            worst_mem_r_t2
        ),
        qpu_running_wall_seconds=None,
        notes=(
            "Candidate #4 final memory-qubit-specific "
            "coherence feasibility check. Uses fresh current "
            "T1/T2 for logical q4/q5 mapped to the physical "
            "memory qubits from the frozen RWP117 preflight. "
            "No QPU job submitted."
        ),
    )

    db.save_dataframe_artifact(
        run_uuid,
        OUT_CSV.name,
        df,
    )

    db.save_json_artifact(
        run_uuid,
        OUT_JSON.name,
        summary,
    )

    print()
    print(
        f"[database] Coherence-check run UUID: "
        f"{run_uuid}"
    )
    print()
    print(
        "Saved:"
    )
    print(
        f"  {OUT_CSV}"
    )
    print(
        f"  {OUT_JSON}"
    )


if __name__ == "__main__":
    main()
