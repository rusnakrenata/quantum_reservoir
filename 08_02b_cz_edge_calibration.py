"""
Week 8 - Step 8.2B
IBM Quantum CZ-edge calibration profiler.

Collect for every physical CZ connection:
- CZ error
- CZ duration
- both directed Target entries where available
- undirected aggregate statistics

This step profiles EDGES only.

It does NOT yet:
- score six-qubit chains,
- select physical regions,
- map H0-H4,
- transpile the QRC,
- rerun memory capacity or forecasting.

Those belong to Step 8.3 / Week 9.
"""

from pathlib import Path
from datetime import datetime, timezone
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
# Helpers
# ============================================================

def safe_float(value):

    if value is None:
        return np.nan

    try:
        return float(value)

    except Exception:
        return np.nan


def get_instruction_properties(
    backend,
    instruction_name,
    qargs,
):
    """
    Safely retrieve InstructionProperties from backend.target.

    Returns
    -------
    error : float
        Gate error.

    duration_s : float
        Gate duration in seconds.
    """

    try:

        props = backend.target[
            instruction_name
        ][tuple(qargs)]

        if props is None:
            return np.nan, np.nan

        error = safe_float(
            getattr(
                props,
                "error",
                None,
            )
        )

        duration_s = safe_float(
            getattr(
                props,
                "duration",
                None,
            )
        )

        return error, duration_s

    except Exception:

        return np.nan, np.nan


def make_undirected_pair(q0, q1):

    return tuple(
        sorted(
            (
                int(q0),
                int(q1),
            )
        )
    )


# ============================================================
# Connect
# ============================================================

snapshot_utc = datetime.now(
    timezone.utc
)


print("=" * 78)
print("WEEK 8 - STEP 8.2B")
print("IBM CZ-EDGE CALIBRATION PROFILER")
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
        "No IBM hardware backends accessible."
    )


print(
    f"Backends found: "
    f"{len(backends)}"
)

print()


# ============================================================
# Containers
# ============================================================

directed_rows = []
undirected_rows = []
backend_summary_rows = []
metadata_rows = []


# ============================================================
# Process backends
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
    # Refresh current calibration
    # --------------------------------------------------------

    try:

        backend.refresh()

        print(
            "  Backend target refreshed."
        )

    except Exception as exc:

        print(
            f"  Warning: backend.refresh() failed: "
            f"{exc}"
        )


    # --------------------------------------------------------
    # Calibration timestamp
    # --------------------------------------------------------

    calibration_date = None

    try:

        properties = backend.properties(
            refresh=True
        )

        calibration_date = getattr(
            properties,
            "last_update_date",
            None,
        )

    except Exception as exc:

        print(
            f"  Warning: backend.properties() failed: "
            f"{exc}"
        )


    if hasattr(
        calibration_date,
        "isoformat",
    ):

        calibration_date_string = (
            calibration_date.isoformat()
        )

    else:

        calibration_date_string = (
            str(calibration_date)
            if calibration_date is not None
            else None
        )


    # --------------------------------------------------------
    # Find all CZ Target entries
    # --------------------------------------------------------

    try:

        cz_map = backend.target[
            "cz"
        ]

    except Exception as exc:

        raise RuntimeError(
            f"{backend.name} does not expose "
            f"a CZ Target entry."
        ) from exc


    directed_pairs = sorted(
        [
            tuple(
                map(
                    int,
                    qargs,
                )
            )
            for qargs
            in cz_map.keys()
            if qargs is not None
            and len(qargs) == 2
        ]
    )


    print(
        f"  Directed CZ entries: "
        f"{len(directed_pairs)}"
    )


    # --------------------------------------------------------
    # Raw directed entries
    # --------------------------------------------------------

    backend_directed_rows = []


    for q0, q1 in directed_pairs:

        error, duration_s = (
            get_instruction_properties(
                backend,
                "cz",
                (q0, q1),
            )
        )


        row = {

            "snapshot_utc":
                snapshot_utc.isoformat(),

            "calibration_date":
                calibration_date_string,

            "backend":
                backend.name,

            "q0":
                q0,

            "q1":
                q1,

            "undirected_edge":
                (
                    f"{min(q0, q1)}-"
                    f"{max(q0, q1)}"
                ),

            "cz_error":
                error,

            "cz_error_percent":
                (
                    100.0 * error
                    if np.isfinite(error)
                    else np.nan
                ),

            "cz_duration_s":
                duration_s,

            "cz_duration_ns":
                (
                    duration_s * 1e9
                    if np.isfinite(duration_s)
                    else np.nan
                ),
        }


        directed_rows.append(
            row
        )

        backend_directed_rows.append(
            row
        )


    # --------------------------------------------------------
    # Construct unique undirected physical edges
    # --------------------------------------------------------

    undirected_pairs = sorted(
        {
            make_undirected_pair(
                q0,
                q1,
            )
            for q0, q1
            in directed_pairs
        }
    )


    print(
        f"  Undirected CZ edges: "
        f"{len(undirected_pairs)}"
    )


    backend_undirected_rows = []


    for a, b in undirected_pairs:

        # Direction a -> b
        error_ab, duration_ab_s = (
            get_instruction_properties(
                backend,
                "cz",
                (a, b),
            )
        )

        # Direction b -> a
        error_ba, duration_ba_s = (
            get_instruction_properties(
                backend,
                "cz",
                (b, a),
            )
        )


        errors = np.array(
            [
                error_ab,
                error_ba,
            ],
            dtype=float,
        )

        durations = np.array(
            [
                duration_ab_s,
                duration_ba_s,
            ],
            dtype=float,
        )


        valid_errors = errors[
            np.isfinite(errors)
        ]

        valid_durations = durations[
            np.isfinite(durations)
        ]


        if len(valid_errors) > 0:

            error_mean = float(
                np.mean(
                    valid_errors
                )
            )

            error_min = float(
                np.min(
                    valid_errors
                )
            )

            error_max = float(
                np.max(
                    valid_errors
                )
            )

        else:

            error_mean = np.nan
            error_min = np.nan
            error_max = np.nan


        if len(valid_durations) > 0:

            duration_mean_s = float(
                np.mean(
                    valid_durations
                )
            )

            duration_min_s = float(
                np.min(
                    valid_durations
                )
            )

            duration_max_s = float(
                np.max(
                    valid_durations
                )
            )

        else:

            duration_mean_s = np.nan
            duration_min_s = np.nan
            duration_max_s = np.nan


        # Directional mismatch diagnostic.
        if (
            np.isfinite(error_ab)
            and np.isfinite(error_ba)
        ):

            error_direction_difference = abs(
                error_ab
                -
                error_ba
            )

        else:

            error_direction_difference = np.nan


        if (
            np.isfinite(duration_ab_s)
            and np.isfinite(duration_ba_s)
        ):

            duration_direction_difference_ns = (
                abs(
                    duration_ab_s
                    -
                    duration_ba_s
                )
                * 1e9
            )

        else:

            duration_direction_difference_ns = np.nan


        row = {

            "snapshot_utc":
                snapshot_utc.isoformat(),

            "calibration_date":
                calibration_date_string,

            "backend":
                backend.name,

            "q_low":
                a,

            "q_high":
                b,

            "edge":
                f"{a}-{b}",

            # ------------------------------
            # Raw direction a -> b
            # ------------------------------

            "cz_error_ab":
                error_ab,

            "cz_error_ab_percent":
                (
                    100.0 * error_ab
                    if np.isfinite(error_ab)
                    else np.nan
                ),

            "cz_duration_ab_ns":
                (
                    duration_ab_s * 1e9
                    if np.isfinite(duration_ab_s)
                    else np.nan
                ),

            # ------------------------------
            # Raw direction b -> a
            # ------------------------------

            "cz_error_ba":
                error_ba,

            "cz_error_ba_percent":
                (
                    100.0 * error_ba
                    if np.isfinite(error_ba)
                    else np.nan
                ),

            "cz_duration_ba_ns":
                (
                    duration_ba_s * 1e9
                    if np.isfinite(duration_ba_s)
                    else np.nan
                ),

            # ------------------------------
            # Undirected aggregates
            # ------------------------------

            "cz_error_mean":
                error_mean,

            "cz_error_mean_percent":
                (
                    100.0
                    * error_mean
                    if np.isfinite(
                        error_mean
                    )
                    else np.nan
                ),

            "cz_error_min_percent":
                (
                    100.0
                    * error_min
                    if np.isfinite(
                        error_min
                    )
                    else np.nan
                ),

            "cz_error_max_percent":
                (
                    100.0
                    * error_max
                    if np.isfinite(
                        error_max
                    )
                    else np.nan
                ),

            "cz_duration_mean_ns":
                (
                    duration_mean_s
                    * 1e9
                    if np.isfinite(
                        duration_mean_s
                    )
                    else np.nan
                ),

            "cz_duration_min_ns":
                (
                    duration_min_s
                    * 1e9
                    if np.isfinite(
                        duration_min_s
                    )
                    else np.nan
                ),

            "cz_duration_max_ns":
                (
                    duration_max_s
                    * 1e9
                    if np.isfinite(
                        duration_max_s
                    )
                    else np.nan
                ),

            "cz_error_direction_difference":
                error_direction_difference,

            "cz_duration_direction_difference_ns":
                duration_direction_difference_ns,
        }


        undirected_rows.append(
            row
        )

        backend_undirected_rows.append(
            row
        )


    # ========================================================
    # Backend-level statistics
    # ========================================================

    edge_df = pd.DataFrame(
        backend_undirected_rows
    )


    def stats(column):

        values = pd.to_numeric(
            edge_df[column],
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

            "mean":
                float(
                    values.mean()
                ),

            "median":
                float(
                    values.median()
                ),

            "min":
                float(
                    values.min()
                ),

            "max":
                float(
                    values.max()
                ),

            "std":
                float(
                    values.std()
                ),
        }


    error_stats = stats(
        "cz_error_mean_percent"
    )

    duration_stats = stats(
        "cz_duration_mean_ns"
    )


    backend_summary_rows.append({

        "snapshot_utc":
            snapshot_utc.isoformat(),

        "calibration_date":
            calibration_date_string,

        "backend":
            backend.name,

        "num_directed_cz_entries":
            len(
                directed_pairs
            ),

        "num_undirected_cz_edges":
            len(
                undirected_pairs
            ),

        "cz_error_mean_percent":
            error_stats[
                "mean"
            ],

        "cz_error_median_percent":
            error_stats[
                "median"
            ],

        "cz_error_min_percent":
            error_stats[
                "min"
            ],

        "cz_error_max_percent":
            error_stats[
                "max"
            ],

        "cz_duration_mean_ns":
            duration_stats[
                "mean"
            ],

        "cz_duration_median_ns":
            duration_stats[
                "median"
            ],

        "cz_duration_min_ns":
            duration_stats[
                "min"
            ],

        "cz_duration_max_ns":
            duration_stats[
                "max"
            ],
    })


    # ========================================================
    # Console diagnostics
    # ========================================================

    print()

    print(
        f"  CZ error median:   "
        f"{error_stats['median']:.4f} %"
    )

    print(
        f"  CZ error range:    "
        f"{error_stats['min']:.4f} - "
        f"{error_stats['max']:.4f} %"
    )

    print(
        f"  CZ duration median:"
        f" {duration_stats['median']:.3f} ns"
    )

    print(
        f"  CZ duration range: "
        f"{duration_stats['min']:.3f} - "
        f"{duration_stats['max']:.3f} ns"
    )

    print()


    # --------------------------------------------------------
    # Best 10 CZ edges by error
    # --------------------------------------------------------

    valid_edges = edge_df.dropna(
        subset=[
            "cz_error_mean_percent",
            "cz_duration_mean_ns",
        ]
    ).copy()


    if len(valid_edges) > 0:

        best_error = (
            valid_edges
            .sort_values(
                [
                    "cz_error_mean_percent",
                    "cz_duration_mean_ns",
                ]
            )
            .head(10)
        )


        print(
            "  Top 10 CZ edges by lowest error:"
        )


        for _, row in best_error.iterrows():

            print(
                f"    {row['edge']:>7}: "
                f"error="
                f"{row['cz_error_mean_percent']:.4f}%, "
                f"duration="
                f"{row['cz_duration_mean_ns']:.3f} ns"
            )


        print()


        # ----------------------------------------------------
        # Fastest 10 CZ edges
        # ----------------------------------------------------

        fastest = (
            valid_edges
            .sort_values(
                [
                    "cz_duration_mean_ns",
                    "cz_error_mean_percent",
                ]
            )
            .head(10)
        )


        print(
            "  Top 10 CZ edges by shortest duration:"
        )


        for _, row in fastest.iterrows():

            print(
                f"    {row['edge']:>7}: "
                f"duration="
                f"{row['cz_duration_mean_ns']:.3f} ns, "
                f"error="
                f"{row['cz_error_mean_percent']:.4f}%"
            )


        print()


    metadata_rows.append({

        "backend":
            backend.name,

        "calibration_date":
            calibration_date_string,

        "num_undirected_cz_edges":
            len(
                undirected_pairs
            ),
    })


# ============================================================
# Save
# ============================================================

directed_df = pd.DataFrame(
    directed_rows
)

undirected_df = pd.DataFrame(
    undirected_rows
)

summary_df = pd.DataFrame(
    backend_summary_rows
)


directed_path = (
    RESULTS_DIR
    / "08_02b_cz_directed_calibration.csv"
)

undirected_path = (
    RESULTS_DIR
    / "08_02b_cz_edge_calibration.csv"
)

summary_path = (
    RESULTS_DIR
    / "08_02b_backend_cz_summary.csv"
)

metadata_path = (
    RESULTS_DIR
    / "08_02b_calibration_metadata.json"
)


directed_df.to_csv(
    directed_path,
    index=False,
)

undirected_df.to_csv(
    undirected_path,
    index=False,
)

summary_df.to_csv(
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
                metadata_rows,
        },

        file,
        indent=2,
        default=str,
    )


# ============================================================
# Final backend comparison
# ============================================================

print("=" * 78)
print("BACKEND CZ-EDGE SUMMARY")
print("=" * 78)


columns = [

    "backend",

    "num_undirected_cz_edges",

    "cz_error_median_percent",

    "cz_error_min_percent",

    "cz_error_max_percent",

    "cz_duration_median_ns",
]


print(
    summary_df[
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
    directed_path
)

print(
    undirected_path
)

print(
    summary_path
)

print(
    metadata_path
)

print()

print(
    "Step 8.2B complete."
)

print(
    "Interpret edge-level calibration before "
    "combining nodes and edges into 6Q regions."
)