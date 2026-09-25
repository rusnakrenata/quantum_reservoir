from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# ============================================================
# IMPORTANT:
# All IQM authentication and client creation goes through
# iqm_account.py.  This discovery script contains NO token.
# ============================================================

from iqm_account import (
    DEFAULT_QUANTUM_COMPUTER,
    get_authentication,
    get_client,
)


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Real Resonance systems confirmed as accessible by the official
# IQM client for this account.
#
# Emerald is first because it is our default system in iqm_account.py.
REAL_QUANTUM_COMPUTERS = (
    DEFAULT_QUANTUM_COMPUTER,  # emerald
    "garnet",
    "sirius",
)

# We do not profile mock systems during the hardware profiler.
INCLUDE_MOCKS = False


# ============================================================
# Helpers
# ============================================================

def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def jsonable(obj: Any) -> Any:
    """
    Convert Pydantic/dataclass/IQM objects recursively into something
    that json.dump can serialize.
    """

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {
            str(k): jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]

    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return jsonable(obj.model_dump(mode="json"))
        except Exception:
            return jsonable(obj.model_dump())

    # Older Pydantic / generic object
    if hasattr(obj, "dict"):
        try:
            return jsonable(obj.dict())
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return {
                str(k): jsonable(v)
                for k, v in vars(obj).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass

    return str(obj)


def get_field(obj: Any, *names: str, default=None):
    """
    Try attributes first, then dictionary keys.
    """

    for name in names:

        if hasattr(obj, name):
            value = getattr(obj, name)

            if value is not None:
                return value

        if isinstance(obj, dict) and name in obj:
            value = obj[name]

            if value is not None:
                return value

    return default


def unique_preserve_order(values):
    """
    Remove duplicates while preserving the configured order.
    """

    return list(dict.fromkeys(values))


# ============================================================
# Authentication / environment
# ============================================================

# iqm_account.py is the ONLY source of credentials.
#
# get_authentication() returns:
#     token, server_url, default_quantum_computer
#
# We intentionally discard the token immediately and never save/print it.
_token, SERVER_URL, ACCOUNT_DEFAULT_QC = get_authentication()
del _token

SERVER_URL = SERVER_URL.rstrip("/")

quantum_computer_aliases = unique_preserve_order(
    REAL_QUANTUM_COMPUTERS
)

if INCLUDE_MOCKS:
    quantum_computer_aliases += [
        f"{name}:mock"
        for name in quantum_computer_aliases
        if ":" not in name
    ]


environment = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "iqm_client_version": package_version("iqm-client"),
    "qiskit_version": package_version("qiskit"),
    "server_url": SERVER_URL,
    "account_default_quantum_computer": ACCOUNT_DEFAULT_QC,
    "configured_default_quantum_computer": DEFAULT_QUANTUM_COMPUTER,
    "authentication_source": "iqm_account.py",
    "token_saved_in_results": False,
}


# ============================================================
# Header
# ============================================================

print("=" * 80)
print("WEEK 12.1 — IQM RESONANCE DISCOVERY")
print("=" * 80)

print()
print("Environment")
print("-----------")
print(f"Server:             {SERVER_URL}")
print(f"Python:             {platform.python_version()}")
print(f"iqm-client:         {environment['iqm_client_version']}")
print(f"Qiskit:             {environment['qiskit_version']}")
print(f"Authentication:     iqm_account.py")
print(f"Configured default: {DEFAULT_QUANTUM_COMPUTER}")
print(f"Account default:    {ACCOUNT_DEFAULT_QC}")
print("Token in results:   NO")


# ============================================================
# 1. Systems selected for Week 12 hardware profiling
# ============================================================

print()
print("=" * 80)
print("IQM QUANTUM COMPUTERS SELECTED FOR PROFILING")
print("=" * 80)

for i, alias in enumerate(
    quantum_computer_aliases,
    start=1,
):
    marker = "  [DEFAULT]" if alias == DEFAULT_QUANTUM_COMPUTER else ""

    print(
        f"{i:2d}. {alias}{marker}"
    )

print()
print(
    f"Configured systems: {len(quantum_computer_aliases)}"
)


# ============================================================
# 2. Query each real system through iqm_account.py
# ============================================================

all_results = {
    "environment": environment,
    "configured_quantum_computers": quantum_computer_aliases,
    "systems": {},
}


for alias in quantum_computer_aliases:

    print()
    print("=" * 80)
    print(f"SYSTEM: {alias}")
    print("=" * 80)

    try:
        # ----------------------------------------------------
        # Authentication + connection
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # get_client() is imported from iqm_account.py.
        # No token handling is duplicated here.
        #
        client = get_client(
            quantum_computer=alias,
        )

        # ----------------------------------------------------
        # Health
        # ----------------------------------------------------

        health = client.get_health()

        # ----------------------------------------------------
        # Static architecture
        # ----------------------------------------------------

        sqa = client.get_static_quantum_architecture()

        # ----------------------------------------------------
        # Dynamic architecture = currently calibrated
        # operations / loci
        # ----------------------------------------------------

        dqa = client.get_dynamic_quantum_architecture()

        # ----------------------------------------------------
        # Current quality metrics
        # ----------------------------------------------------
        #
        # Quality metrics may not be exposed uniformly for every
        # Resonance system, so failure here must NOT discard the
        # architecture/health data.
        #
        try:
            quality_metrics = client.get_quality_metric_set()

        except Exception as exc:
            quality_metrics = {
                "unavailable": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        # ----------------------------------------------------
        # Current calibration set
        # ----------------------------------------------------

        try:
            calibration_set = client.get_calibration_set()

        except Exception as exc:
            calibration_set = {
                "unavailable": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        # ----------------------------------------------------
        # General backend info
        # ----------------------------------------------------

        try:
            about = client.get_about()

        except Exception as exc:
            about = {
                "unavailable": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        try:
            feedback_groups = client.get_feedback_groups()

        except Exception as exc:
            feedback_groups = {
                "unavailable": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        # ----------------------------------------------------
        # Convert to JSON-safe structures
        # ----------------------------------------------------

        health_json = jsonable(health)
        sqa_json = jsonable(sqa)
        dqa_json = jsonable(dqa)
        qm_json = jsonable(quality_metrics)
        calibration_json = jsonable(calibration_set)
        about_json = jsonable(about)
        feedback_json = jsonable(feedback_groups)

        # ----------------------------------------------------
        # Extract useful summary
        # ----------------------------------------------------

        qubits = get_field(
            sqa_json,
            "qubits",
            default=[],
        )

        connectivity = get_field(
            sqa_json,
            "connectivity",
            "qubit_connectivity",
            default=[],
        )

        calibration_set_id = get_field(
            dqa_json,
            "calibration_set_id",
            default=None,
        )

        gates = get_field(
            dqa_json,
            "gates",
            default={},
        )

        if isinstance(gates, dict):
            native_gate_names = sorted(
                gates.keys()
            )
        else:
            native_gate_names = []

        # Dynamic architecture can also contain computational
        # resonators on IQM Star-type architectures.
        computational_resonators = get_field(
            dqa_json,
            "computational_resonators",
            default=[],
        )

        # ----------------------------------------------------
        # Console summary
        # ----------------------------------------------------

        print()
        print("Connection")
        print("----------")
        print(
            f"Connected QC name: {client.quantum_computer_name}"
        )

        print()
        print("Health")
        print("------")
        print(
            json.dumps(
                health_json,
                indent=2,
                ensure_ascii=False,
            )
        )

        print()
        print("Architecture")
        print("------------")
        print(
            f"Number of qubits:             {len(qubits)}"
        )
        print(
            f"Connectivity edges:           {len(connectivity)}"
        )
        print(
            f"Computational resonators:     "
            f"{len(computational_resonators)}"
        )
        print(
            f"Calibration set ID:           {calibration_set_id}"
        )

        print()
        print("Native calibrated operations")
        print("----------------------------")

        if native_gate_names:

            for gate in native_gate_names:
                print(
                    f"  - {gate}"
                )

        else:
            print(
                "  <gate dictionary not found in expected field>"
            )

        print()
        print("First connectivity edges")
        print("------------------------")

        for edge in connectivity[:20]:
            print(
                f"  {edge}"
            )

        if len(connectivity) > 20:
            print(
                f"  ... ({len(connectivity) - 20} more)"
            )

        # ----------------------------------------------------
        # Save raw information
        # ----------------------------------------------------

        system_result = {
            "quantum_computer": alias,
            "connected_quantum_computer_name":
                client.quantum_computer_name,
            "health": health_json,
            "about": about_json,
            "static_architecture": sqa_json,
            "dynamic_architecture": dqa_json,
            "calibration_set": calibration_json,
            "quality_metrics": qm_json,
            "feedback_groups": feedback_json,
            "summary": {
                "qubit_count": len(qubits),
                "connectivity_edge_count": len(connectivity),
                "computational_resonator_count":
                    len(computational_resonators),
                "native_gate_names": native_gate_names,
                "calibration_set_id": calibration_set_id,
            },
        }

        all_results["systems"][alias] = (
            system_result
        )

        safe_alias = (
            alias
            .replace(":", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        per_system_path = (
            RESULTS_DIR
            / f"12_01_iqm_{safe_alias}.json"
        )

        with open(
            per_system_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                system_result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            f"Saved: {per_system_path}"
        )

    except Exception as exc:

        print()
        print("ERROR")
        print("-----")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        all_results["systems"][alias] = {
            "quantum_computer": alias,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


# ============================================================
# 3. Consolidated result
# ============================================================

output_path = (
    RESULTS_DIR
    / "12_01_iqm_discovery.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 4. Final summary
# ============================================================

successful = [
    name
    for name, result
    in all_results["systems"].items()
    if "error" not in result
]

failed = [
    name
    for name, result
    in all_results["systems"].items()
    if "error" in result
]


print()
print("=" * 80)
print("12.1 DISCOVERY COMPLETE")
print("=" * 80)

print(
    f"Successful systems: {len(successful)} "
    f"({', '.join(successful) if successful else 'none'})"
)

print(
    f"Failed systems:     {len(failed)} "
    f"({', '.join(failed) if failed else 'none'})"
)

print(
    f"Saved: {output_path}"
)

print()
print("IMPORTANT:")
print(
    "No quantum circuit was submitted by this script."
)
print(
    "Authentication was handled only by iqm_account.py."
)
print(
    "The IQM API token is NOT written to the result files."
)
