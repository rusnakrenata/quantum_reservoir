from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from ibm_account import get_service


# =============================================================================
# CONFIG
# =============================================================================

JOB_ID = "dak76ngmhr3c73e9q490"
BACKEND_NAME = "ibm_kingston"

# Candidate #3 fresh layout used for the M3 run:
LAYOUT = [105, 117, 125, 124, 126, 127]

# Candidate #3 H2 couplings actually present in the frozen J:
# J01, J12, J23, J24, J45
LOGICAL_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (2, 4),
    (4, 5),
]

PHYSICAL_EDGES = [
    (LAYOUT[a], LAYOUT[b])
    for a, b in LOGICAL_EDGES
]

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def iso(x):
    if x is None:
        return None
    try:
        return x.isoformat()
    except Exception:
        return str(x)


def qubit_value_and_date(props, q, name):
    """
    BackendProperties.qubit_property(q, name)
    normally returns (value, datetime).
    """
    try:
        out = props.qubit_property(q, name)
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], out[1]
        return out, None
    except Exception:
        return None, None


def gate_value_and_date(props, gate, qubits, name):
    """
    Try forward orientation, then reverse orientation.
    """
    for pair in (list(qubits), list(reversed(qubits))):
        try:
            out = props.gate_property(gate, pair, name)
            if isinstance(out, tuple) and len(out) >= 2:
                return out[0], out[1], pair
            return out, None, pair
        except Exception:
            pass

    return None, None, None


def make_qubit_rows(label, props):
    rows = []

    for logical_q, physical_q in enumerate(LAYOUT):

        t1, t1_date = qubit_value_and_date(props, physical_q, "T1")
        t2, t2_date = qubit_value_and_date(props, physical_q, "T2")
        ro, ro_date = qubit_value_and_date(
            props, physical_q, "readout_error"
        )

        rows.append({
            "snapshot": label,
            "logical_qubit": logical_q,
            "physical_qubit": physical_q,
            "T1_us": None if t1 is None else float(t1) * 1e6,
            "T1_calibration_time": iso(t1_date),
            "T2_us": None if t2 is None else float(t2) * 1e6,
            "T2_calibration_time": iso(t2_date),
            "readout_error_percent":
                None if ro is None else float(ro) * 100.0,
            "readout_calibration_time": iso(ro_date),
        })

    return rows


def make_edge_rows(label, props):
    rows = []

    for logical_edge, physical_edge in zip(
        LOGICAL_EDGES, PHYSICAL_EDGES
    ):

        err, err_date, actual_pair = gate_value_and_date(
            props,
            "cz",
            physical_edge,
            "gate_error",
        )

        length, length_date, _ = gate_value_and_date(
            props,
            "cz",
            physical_edge,
            "gate_length",
        )

        rows.append({
            "snapshot": label,
            "logical_edge": f"{logical_edge[0]}-{logical_edge[1]}",
            "physical_edge":
                f"{physical_edge[0]}-{physical_edge[1]}",
            "property_orientation":
                None if actual_pair is None
                else f"{actual_pair[0]}-{actual_pair[1]}",
            "CZ_error_percent":
                None if err is None else float(err) * 100.0,
            "CZ_error_calibration_time": iso(err_date),
            "CZ_duration_ns":
                None if length is None else float(length) * 1e9,
            "CZ_duration_calibration_time": iso(length_date),
        })

    return rows


# =============================================================================
# MAIN
# =============================================================================

service = get_service()

print("=" * 100)
print("WEEK 11.1E.9B — EXECUTION-TIME CALIBRATION EXTRACTION")
print("=" * 100)

job = service.job(JOB_ID)

print(f"Job ID:   {job.job_id()}")
print(f"Status:   {job.status()}")

# -------------------------------------------------------------------------
# Job timing
# -------------------------------------------------------------------------

metrics = job.metrics()

print("\nJOB METRICS")
print("-" * 100)
print(json.dumps(metrics, indent=2, default=str))

timestamps = metrics.get("timestamps", {})

created = timestamps.get("created")
running = (
    timestamps.get("running")
    or timestamps.get("execution_start")
    or timestamps.get("started")
)
finished = timestamps.get("finished")

print("\nImportant timestamps:")
print(f"  created:  {created}")
print(f"  running:  {running}")
print(f"  finished: {finished}")

# -------------------------------------------------------------------------
# Properties that IBM says were active WHEN THE JOB STARTED RUNNING
# -------------------------------------------------------------------------

execution_props = job.properties(refresh=True)

if execution_props is None:
    raise RuntimeError(
        "IBM did not return execution-time backend properties "
        "for this job."
    )

print("\nExecution-time backend calibration:")
print(f"  backend:          {execution_props.backend_name}")
print(f"  last_update_date: {execution_props.last_update_date}")

# -------------------------------------------------------------------------
# Properties NOW, for comparison
# -------------------------------------------------------------------------

backend = service.backend(BACKEND_NAME)
current_props = backend.properties(refresh=True)

print("\nCurrent backend calibration:")
print(f"  backend:          {current_props.backend_name}")
print(f"  last_update_date: {current_props.last_update_date}")


# =============================================================================
# EXTRACT QUBITS
# =============================================================================

qubit_rows = []
qubit_rows += make_qubit_rows("JOB_EXECUTION", execution_props)
qubit_rows += make_qubit_rows("CURRENT_NOW", current_props)

qubit_df = pd.DataFrame(qubit_rows)

print("\n" + "=" * 100)
print("QUBIT PROPERTIES")
print("=" * 100)

print(
    qubit_df[
        [
            "snapshot",
            "logical_qubit",
            "physical_qubit",
            "T1_us",
            "T2_us",
            "readout_error_percent",
        ]
    ].to_string(index=False)
)


# =============================================================================
# EXTRACT CZ EDGES
# =============================================================================

edge_rows = []
edge_rows += make_edge_rows("JOB_EXECUTION", execution_props)
edge_rows += make_edge_rows("CURRENT_NOW", current_props)

edge_df = pd.DataFrame(edge_rows)

print("\n" + "=" * 100)
print("CZ EDGE PROPERTIES")
print("=" * 100)

print(
    edge_df[
        [
            "snapshot",
            "logical_edge",
            "physical_edge",
            "CZ_error_percent",
            "CZ_duration_ns",
        ]
    ].to_string(index=False)
)


# =============================================================================
# EXECUTION-TIME SUMMARY
# =============================================================================

qe = qubit_df[qubit_df["snapshot"] == "JOB_EXECUTION"]
ee = edge_df[edge_df["snapshot"] == "JOB_EXECUTION"]

print("\n" + "=" * 100)
print("EXECUTION-TIME SUMMARY")
print("=" * 100)

print(f"min T1:              {qe['T1_us'].min():.3f} us")
print(f"min T2:              {qe['T2_us'].min():.3f} us")
print(
    f"max readout error:   "
    f"{qe['readout_error_percent'].max():.6f}%"
)
print(
    f"max required CZ err: "
    f"{ee['CZ_error_percent'].max():.6f}%"
)

print("\nPRE-FLIGHT values from our M3 run were:")
print("  min T1:            165.636 us")
print("  min T2:            173.076 us")
print("  max readout error: 5.004883%")
print("  max CZ error:      0.167720%")

print("\nCompare these directly with EXECUTION-TIME SUMMARY above.")


# =============================================================================
# SAVE
# =============================================================================

qubit_file = RESULTS / "11_1E9B_execution_qubit_properties.csv"
edge_file = RESULTS / "11_1E9B_execution_cz_properties.csv"
json_file = RESULTS / "11_1E9B_job_metrics.json"

qubit_df.to_csv(qubit_file, index=False)
edge_df.to_csv(edge_file, index=False)

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, default=str)

print("\nSaved:")
print(f"  {qubit_file}")
print(f"  {edge_file}")
print(f"  {json_file}")