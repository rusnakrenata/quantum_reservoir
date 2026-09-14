"""
Week 8 - Step 8.2A
IBM Quantum qubit-level calibration profiler.

Collect for every physical qubit:
- T1
- T2
- qubit frequency
- measurement/readout error
- measurement duration
- simple T2/(2*T1) diagnostic
- estimated pure-dephasing time T_phi where meaningful

The script profiles every accessible real IBM backend.

It does NOT yet analyze:
- CZ errors
- CZ durations
- optimal 6-qubit regions
- logical-to-physical mappings

Those follow in later substeps.
"""

from pathlib import Path
from datetime import datetime, timezone
import math
import json

import numpy as np
import pandas as pd

from ibm_account import get_service


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Utility helpers
# ============================================================

def safe_float(value):
    """Convert numerical value safely to float."""

    if value is None:
        return np.nan

    try:
        return float(value)

    except Exception:
        return np.nan


def safe_iso(value):
    if value is None:
        return None

    if hasattr(value, "isoformat"):

        try:
            return value.isoformat()

        except Exception:
            pass

    return str(value)


def get_target_property(
    backend,
    instruction_name,
    qargs,
    field,
):
    """
    Safely retrieve an InstructionProperties field
    from backend.target.

    Example
    -------
    get_target_property(
        backend,
        "measure",
        (0,),
        "error",
    )
    """

    try:

        instruction_map = (
            backend.target[
                instruction_name
            ]
        )

        props = instruction_map[
            tuple(qargs)
        ]

        if props is None:
            return np.nan

        return safe_float(
            getattr(
                props,
                field,
                None,
            )
        )

    except Exception:

        return np.nan


def estimate_tphi(
    t1_seconds,
    t2_seconds,
):
    """
    Estimate pure-dephasing time:

        1/T2 = 1/(2*T1) + 1/T_phi

    Returns NaN when the simple model would require
    a negative or zero pure-dephasing rate.
    """

    if (
        not np.isfinite(t1_seconds)
        or not np.isfinite(t2_seconds)
        or t1_seconds <= 0
        or t2_seconds <= 0
    ):
        return np.nan

    rate = (
        1.0 / t2_seconds
        -
        1.0 / (
            2.0 * t1_seconds
        )
    )

    if rate <= 0:
        return np.nan

    return 1.0 / rate


# ============================================================
# Connect
# ============================================================

snapshot_utc = datetime.now(
    timezone.utc
)

print("=" * 78)
print("WEEK 8 - STEP 8.2A")
print("IBM QUBIT-LEVEL CALIBRATION PROFILER")
print("=" * 78)

print()
print(
    f"Snapshot UTC: "
    f"{snapshot_utc.isoformat()}"
)
print()


service = get_service()


backends = service.backends(
    simulator=False
)


if not backends:

    raise RuntimeError(
        "No real IBM backends accessible."
    )


print(
    f"Backends found: "
    f"{len(backends)}"
)

print()


# ============================================================
# Result containers
# ============================================================

qubit_rows = []
backend_rows = []
metadata = []


# ============================================================
# Process every backend
# ============================================================

for backend_index, backend in enumerate(
    backends,
    start=1,
):

    print("-" * 78)

    print(
        f"[{backend_index}/{len(backends)}] "
        f"{backend.name}"
    )

    # --------------------------------------------------------
    # Refresh the current backend calibration snapshot
    # --------------------------------------------------------

    try:

        backend.refresh()

        print(
            "  Backend target refreshed."
        )

    except Exception as exc:

        print(
            f"  Warning: backend.refresh() "
            f"failed: {exc}"
        )


    # --------------------------------------------------------
    # BackendProperties
    # --------------------------------------------------------

    try:

        properties = backend.properties(
            refresh=True
        )

    except Exception as exc:

        print(
            f"  Warning: backend.properties() "
            f"failed: {exc}"
        )

        properties = None


    calibration_date = None

    if properties is not None:

        calibration_date = getattr(
            properties,
            "last_update_date",
            None,
        )


    calibration_id = getattr(
        backend,
        "calibration_id",
        None,
    )


    print(
        f"  Qubits:            "
        f"{backend.num_qubits}"
    )

    print(
        f"  Calibration ID:    "
        f"{calibration_id}"
    )

    print(
        f"  Calibration date:  "
        f"{safe_iso(calibration_date)}"
    )


    backend_qubit_rows = []


    # ========================================================
    # Every physical qubit
    # ========================================================

    for q in range(
        backend.num_qubits
    ):

        # ----------------------------------------------------
        # T1, T2, frequency
        # ----------------------------------------------------

        try:

            qp = backend.qubit_properties(
                q
            )

        except Exception:

            qp = None


        if qp is not None:

            t1_s = safe_float(
                getattr(
                    qp,
                    "t1",
                    None,
                )
            )

            t2_s = safe_float(
                getattr(
                    qp,
                    "t2",
                    None,
                )
            )

            frequency_hz = safe_float(
                getattr(
                    qp,
                    "frequency",
                    None,
                )
            )

        else:

            t1_s = np.nan
            t2_s = np.nan
            frequency_hz = np.nan


        # ----------------------------------------------------
        # Measurement properties
        # ----------------------------------------------------

        readout_error = (
            get_target_property(
                backend,
                "measure",
                (q,),
                "error",
            )
        )

        measure_duration_s = (
            get_target_property(
                backend,
                "measure",
                (q,),
                "duration",
            )
        )


        # ----------------------------------------------------
        # Derived metrics
        # ----------------------------------------------------

        t1_us = (
            t1_s * 1e6
            if np.isfinite(t1_s)
            else np.nan
        )

        t2_us = (
            t2_s * 1e6
            if np.isfinite(t2_s)
            else np.nan
        )

        frequency_ghz = (
            frequency_hz / 1e9
            if np.isfinite(
                frequency_hz
            )
            else np.nan
        )

        measure_duration_us = (
            measure_duration_s * 1e6
            if np.isfinite(
                measure_duration_s
            )
            else np.nan
        )


        if (
            np.isfinite(t1_s)
            and np.isfinite(t2_s)
            and t1_s > 0
        ):

            t2_over_2t1 = (
                t2_s
                /
                (2.0 * t1_s)
            )

        else:

            t2_over_2t1 = np.nan


        tphi_s = estimate_tphi(
            t1_s,
            t2_s,
        )

        tphi_us = (
            tphi_s * 1e6
            if np.isfinite(tphi_s)
            else np.nan
        )


        row = {

            "snapshot_utc":
                snapshot_utc.isoformat(),

            "backend":
                backend.name,

            "calibration_id":
                calibration_id,

            "calibration_date":
                safe_iso(
                    calibration_date
                ),

            "physical_qubit":
                q,

            "t1_us":
                t1_us,

            "t2_us":
                t2_us,

            "tphi_us":
                tphi_us,

            "t2_over_2t1":
                t2_over_2t1,

            "frequency_ghz":
                frequency_ghz,

            "readout_error":
                readout_error,

            "readout_error_percent":
                (
                    100.0
                    * readout_error
                    if np.isfinite(
                        readout_error
                    )
                    else np.nan
                ),

            "measure_duration_us":
                measure_duration_us,
        }


        qubit_rows.append(
            row
        )

        backend_qubit_rows.append(
            row
        )


    # ========================================================
    # Backend summary
    # ========================================================

    backend_df = pd.DataFrame(
        backend_qubit_rows
    )


    def stats(column):

        values = pd.to_numeric(
            backend_df[column],
            errors="coerce",
        ).dropna()

        if len(values) == 0:

            return {
                "mean": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
                "std": np.nan,
            }

        return {
            "mean": values.mean(),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
            "std": values.std(),
        }


    t1_stats = stats(
        "t1_us"
    )

    t2_stats = stats(
        "t2_us"
    )

    ro_stats = stats(
        "readout_error_percent"
    )

    measurement_stats = stats(
        "measure_duration_us"
    )


    backend_rows.append({

        "snapshot_utc":
            snapshot_utc.isoformat(),

        "backend":
            backend.name,

        "num_qubits":
            backend.num_qubits,

        "calibration_id":
            calibration_id,

        "calibration_date":
            safe_iso(
                calibration_date
            ),

        "t1_mean_us":
            t1_stats["mean"],

        "t1_median_us":
            t1_stats["median"],

        "t1_min_us":
            t1_stats["min"],

        "t1_max_us":
            t1_stats["max"],

        "t2_mean_us":
            t2_stats["mean"],

        "t2_median_us":
            t2_stats["median"],

        "t2_min_us":
            t2_stats["min"],

        "t2_max_us":
            t2_stats["max"],

        "readout_error_mean_percent":
            ro_stats["mean"],

        "readout_error_median_percent":
            ro_stats["median"],

        "readout_error_min_percent":
            ro_stats["min"],

        "readout_error_max_percent":
            ro_stats["max"],

        "measure_duration_mean_us":
            measurement_stats[
                "mean"
            ],

        "measure_duration_median_us":
            measurement_stats[
                "median"
            ],
    })


    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()

    print(
        f"  T1 median:         "
        f"{t1_stats['median']:.3f} us"
    )

    print(
        f"  T1 range:          "
        f"{t1_stats['min']:.3f} - "
        f"{t1_stats['max']:.3f} us"
    )

    print(
        f"  T2 median:         "
        f"{t2_stats['median']:.3f} us"
    )

    print(
        f"  T2 range:          "
        f"{t2_stats['min']:.3f} - "
        f"{t2_stats['max']:.3f} us"
    )

    print(
        f"  Readout median:    "
        f"{ro_stats['median']:.3f} %"
    )

    print(
        f"  Readout range:     "
        f"{ro_stats['min']:.3f} - "
        f"{ro_stats['max']:.3f} %"
    )

    print(
        f"  Measure duration:  "
        f"{measurement_stats['median']:.3f} us "
        f"(median)"
    )

    print()


    # --------------------------------------------------------
    # Print best / worst qubits for intuition
    # --------------------------------------------------------

    valid_t2 = backend_df.dropna(
        subset=[
            "t2_us",
            "readout_error_percent",
        ]
    ).copy()


    if len(valid_t2) > 0:

        best_t2 = valid_t2.nlargest(
            5,
            "t2_us",
        )

        lowest_ro = valid_t2.nsmallest(
            5,
            "readout_error_percent",
        )

        print(
            "  Top 5 qubits by T2:"
        )

        for _, r in best_t2.iterrows():

            print(
                f"    q{int(r['physical_qubit']):3d}: "
                f"T2={r['t2_us']:.3f} us, "
                f"T1={r['t1_us']:.3f} us, "
                f"RO={r['readout_error_percent']:.3f}%"
            )

        print()

        print(
            "  Top 5 qubits by lowest readout error:"
        )

        for _, r in lowest_ro.iterrows():

            print(
                f"    q{int(r['physical_qubit']):3d}: "
                f"RO={r['readout_error_percent']:.3f}%, "
                f"T2={r['t2_us']:.3f} us"
            )

        print()


    metadata.append({

        "backend":
            backend.name,

        "calibration_id":
            calibration_id,

        "calibration_date":
            safe_iso(
                calibration_date
            ),
    })


# ============================================================
# Save
# ============================================================

qubit_df = pd.DataFrame(
    qubit_rows
)

backend_df = pd.DataFrame(
    backend_rows
)


qubit_path = (
    RESULTS_DIR
    / "08_02a_qubit_calibration.csv"
)

summary_path = (
    RESULTS_DIR
    / "08_02a_backend_qubit_summary.csv"
)

metadata_path = (
    RESULTS_DIR
    / "08_02a_calibration_metadata.json"
)


qubit_df.to_csv(
    qubit_path,
    index=False,
)

backend_df.to_csv(
    summary_path,
    index=False,
)


with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "snapshot_utc":
                snapshot_utc.isoformat(),

            "backends":
                metadata,
        },

        file,
        indent=2,
        default=str,
    )


# ============================================================
# Final comparison
# ============================================================

print("=" * 78)
print("BACKEND QUBIT-LEVEL SUMMARY")
print("=" * 78)

columns = [

    "backend",

    "t1_median_us",

    "t2_median_us",

    "readout_error_median_percent",

    "measure_duration_median_us",
]


print(
    backend_df[
        columns
    ].to_string(
        index=False
    )
)


print()

print("=" * 78)
print("FILES SAVED")
print("=" * 78)

print(
    qubit_path
)

print(
    summary_path
)

print(
    metadata_path
)

print()

print(
    "Step 8.2A complete."
)

print(
    "Interpret these qubit-level calibration "
    "results before moving to CZ-edge profiling."
)