"""
Week 12.5A v2 — IQM native Rule-5 / simulator capability probe.

Purpose
-------
Prefer IQM-native hardware information for Rule 5 instead of converting IQM
calibration metrics into an external Aer surrogate unless such a surrogate is
scientifically necessary later.

This script submits NO quantum or simulator jobs. It only:
1. enumerates accessible IQM Resonance quantum-computer aliases,
2. probes Emerald/Garnet and documented mock aliases (emerald:mock,
   garnet:mock) for metadata through iqm-client,
3. optionally probes the native QDMI Qiskit backend if iqm-qdmi[qiskit] is
   installed, without calling backend.run(), sampler(), or estimator(),
4. preserves the 12 Rule-5 physical layouts using IQM-native calibration
   quantities only,
5. conservatively assesses whether the mock backends are explicitly documented
   by returned metadata as calibration/noise-aware.

Important interpretation
------------------------
- "mock exists" != "hardware-calibrated noisy simulator".
- topology/calibration metadata on a mock != proof that execution uses it.
- noisy execution remains UNVERIFIED unless explicit metadata says so.

Inputs
------
results/12_01_iqm_discovery.json
results/12_04c_iqm_rule5_selected_top3.csv

Outputs
-------
results/12_05a_iqm_native_system_inventory.json
results/12_05a_iqm_native_backend_probe.json
results/12_05a_iqm_native_rule5_metrics.csv
results/12_05a_iqm_native_simulator_assessment.json

No QPU jobs. No simulator jobs. No shots. No 2026 data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from iqm.iqm_client import IQMClient
from iqm_account import get_authentication


RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

DISCOVERY_JSON = RESULTS / "12_01_iqm_discovery.json"
RULE5_LAYOUTS = RESULTS / "12_04c_iqm_rule5_selected_top3.csv"

OUT_INVENTORY = RESULTS / "12_05a_iqm_native_system_inventory.json"
OUT_PROBE = RESULTS / "12_05a_iqm_native_backend_probe.json"
OUT_RULE5 = RESULTS / "12_05a_iqm_native_rule5_metrics.csv"
OUT_ASSESSMENT = RESULTS / "12_05a_iqm_native_simulator_assessment.json"

REAL_ALIASES = ("emerald", "garnet")
DOCUMENTED_MOCK_ALIASES = ("emerald:mock", "garnet:mock")

SIMULATOR_WORDS = ("simulator", "simulation", "mock")
NOISE_WORDS = (
    "noise",
    "noisy",
    "error model",
    "error_model",
    "calibration-aware",
    "calibration aware",
    "hardware-aware",
    "hardware aware",
    "realistic noise",
)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return jsonable(obj.model_dump(mode="json"))
        except Exception:
            try:
                return jsonable(obj.model_dump())
            except Exception:
                pass
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


def safe_call(fn):
    try:
        return {
            "available": True,
            "value": jsonable(fn()),
            "error_type": None,
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "value": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def finite_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def flatten_strings(obj: Any, prefix: str = ""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten_strings(v, child))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(flatten_strings(v, f"{prefix}[{i}]"))
    elif isinstance(obj, str):
        out.append((prefix, obj))
    return out


def count_matching_metric_fields(obj: Any):
    counts = {
        "t1": 0,
        "t2": 0,
        "fidelity": 0,
        "readout": 0,
        "cz": 0,
        "prx": 0,
        "duration": 0,
        "error": 0,
    }
    names = []

    def visit(x):
        if isinstance(x, dict):
            if "dut_field" in x:
                names.append(str(x["dut_field"]))
            for k, v in x.items():
                names.append(str(k))
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)

    visit(obj)
    for field in names:
        f = field.lower()
        counts["t1"] += int("t1" in f)
        counts["t2"] += int("t2" in f)
        counts["fidelity"] += int("fidelity" in f)
        counts["readout"] += int(
            "readout" in f or "measure" in f or "ssro" in f
        )
        counts["cz"] += int("cz" in f)
        counts["prx"] += int("prx" in f)
        counts["duration"] += int("duration" in f)
        counts["error"] += int("error" in f)
    return counts


def configured_alias_inventory():
    """
    No direct REST inventory call is used.

    Authentication and quantum-computer resolution are delegated only to the
    official IQM client paths:
      - IQMClient for the real Emerald/Garnet systems;
      - optional QDMI IQMBackend for the documented :mock aliases.

    The real aliases are frozen by the Week-12 experiment design and the mock
    aliases are explicitly documented by IQM's QDMI Qiskit integration.
    """
    aliases_to_probe = list(
        dict.fromkeys([
            *REAL_ALIASES,
            *DOCUMENTED_MOCK_ALIASES,
        ])
    )

    return {
        "source": (
            "experiment-configured real aliases plus documented QDMI mock "
            "aliases; no raw /v1/quantum-computers REST request"
        ),
        "real_aliases": list(REAL_ALIASES),
        "documented_mock_aliases": list(DOCUMENTED_MOCK_ALIASES),
        "aliases_to_probe": aliases_to_probe,
    }


def get_alias(item: Any):
    if not isinstance(item, dict):
        return None
    for key in ("alias", "name", "quantum_computer", "display_name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def probe_iqm_client_alias(alias: str, server_url: str, token: str):
    result = {"alias_requested": alias, "client_created": False}
    try:
        client = IQMClient(
            server_url,
            quantum_computer=alias,
            token=token,
        )
        result["client_created"] = True
        result["resolved_quantum_computer_name"] = str(
            getattr(client, "quantum_computer_name", "")
        )
    except Exception as exc:
        result["client_error_type"] = type(exc).__name__
        result["client_error"] = str(exc)
        return result

    result["health"] = safe_call(client.get_health)
    result["static_architecture"] = safe_call(
        client.get_static_quantum_architecture
    )
    result["dynamic_architecture"] = safe_call(
        client.get_dynamic_quantum_architecture
    )
    result["quality_metric_set"] = safe_call(
        client.get_quality_metric_set
    )
    result["calibration_set"] = safe_call(client.get_calibration_set)
    result["about"] = safe_call(client.get_about)

    combined = {
        k: v.get("value")
        for k, v in result.items()
        if isinstance(v, dict) and "available" in v
    }
    result["native_metric_field_counts"] = count_matching_metric_fields(
        combined
    )
    return result


def target_property_coverage(target):
    out = {
        "operation_names": [],
        "instruction_loci_total": 0,
        "instruction_loci_with_properties": 0,
        "instruction_loci_with_duration": 0,
        "instruction_loci_with_error": 0,
        "by_operation": {},
    }
    try:
        names = sorted(str(x) for x in target.operation_names)
    except Exception:
        names = []
    out["operation_names"] = names

    for name in names:
        op = {
            "loci_total": 0,
            "loci_with_properties": 0,
            "loci_with_duration": 0,
            "loci_with_error": 0,
        }
        try:
            items = list(target[name].items())
        except Exception:
            items = []
        for _, props in items:
            out["instruction_loci_total"] += 1
            op["loci_total"] += 1
            if props is not None:
                out["instruction_loci_with_properties"] += 1
                op["loci_with_properties"] += 1
                if finite_float(getattr(props, "duration", None)) is not None:
                    out["instruction_loci_with_duration"] += 1
                    op["loci_with_duration"] += 1
                if finite_float(getattr(props, "error", None)) is not None:
                    out["instruction_loci_with_error"] += 1
                    op["loci_with_error"] += 1
        out["by_operation"][name] = op
    return out


def probe_qdmi_backend(alias: str, server_url: str, token: str):
    """Metadata only. Never calls run/sampler/estimator."""
    result = {
        "alias_requested": alias,
        "package_available": False,
        "backend_created": False,
        "job_submitted": False,
    }
    try:
        from iqm.qdmi.qiskit import IQMBackend
    except Exception as exc:
        result["import_error_type"] = type(exc).__name__
        result["import_error"] = str(exc)
        return result

    result["package_available"] = True
    old_token = os.environ.get("IQM_TOKEN")
    old_server = os.environ.get("IQM_SERVER_URL")
    try:
        os.environ["IQM_TOKEN"] = token
        os.environ["IQM_SERVER_URL"] = server_url.rstrip("/")
        backend = IQMBackend(
            base_url=server_url.rstrip("/"),
            qc_alias=alias,
        )
        result["backend_created"] = True
        result["backend_name"] = str(getattr(backend, "name", ""))
        result["num_qubits"] = int(getattr(backend, "num_qubits", 0))
        result["backend_class"] = (
            f"{backend.__class__.__module__}.{backend.__class__.__name__}"
        )
        result["target_coverage"] = target_property_coverage(backend.target)
        result["options"] = jsonable(getattr(backend, "options", None))

        attrs = {}
        for name in dir(backend):
            if name.startswith("_"):
                continue
            low = name.lower()
            if any(w in low for w in ("sim", "noise", "calib", "error", "mock")):
                try:
                    value = getattr(backend, name)
                    attrs[name] = (
                        f"<callable {getattr(value, '__name__', name)}>"
                        if callable(value)
                        else jsonable(value)
                    )
                except Exception as exc:
                    attrs[name] = f"<unavailable: {type(exc).__name__}: {exc}>"
        result["relevant_public_attributes"] = attrs
    except Exception as exc:
        result["backend_error_type"] = type(exc).__name__
        result["backend_error"] = str(exc)
    finally:
        if old_token is None:
            os.environ.pop("IQM_TOKEN", None)
        else:
            os.environ["IQM_TOKEN"] = old_token
        if old_server is None:
            os.environ.pop("IQM_SERVER_URL", None)
        else:
            os.environ["IQM_SERVER_URL"] = old_server
    return result


def build_native_rule5_table(selected_layouts: pd.DataFrame):
    wanted = [
        "backend", "topology", "rule5_rank_within_top3", "embedding_id",
        "layout", "q0_C", "q1_D", "q2_P", "q3_H", "q4_M1", "q5_M2",
        "min_t1_us", "mean_t1_us", "min_t2_echo_us", "mean_t2_echo_us",
        "memory_min_t1_us", "memory_min_t2_echo_us",
        "min_readout_fidelity", "worst_readout_error",
        "min_cz_fidelity", "worst_cz_error",
    ]
    existing = [c for c in wanted if c in selected_layouts.columns]
    out = selected_layouts[existing].copy()
    out["rule5_metric_source"] = "IQM native calibration snapshot"
    out["external_noise_model_conversion"] = False
    return out


def explicit_semantic_evidence(inventory_items, client_probe, qdmi_probe):
    corpus = {
        "inventory": inventory_items,
        "client_probe": client_probe,
        "qdmi_probe": qdmi_probe,
    }
    evidence = []
    for path, value in flatten_strings(corpus):
        text = value.lower()
        if (
            any(word in text for word in SIMULATOR_WORDS)
            and any(word in text for word in NOISE_WORDS)
        ):
            evidence.append({"path": path, "text": value})
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (DISCOVERY_JSON, RULE5_LAYOUTS):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    if args.overwrite:
        for path in (OUT_INVENTORY, OUT_PROBE, OUT_RULE5, OUT_ASSESSMENT):
            if path.exists():
                path.unlink()

    token, server_url, _ = get_authentication()
    with open(DISCOVERY_JSON, "r", encoding="utf-8") as f:
        discovery = json.load(f)
    selected_layouts = pd.read_csv(RULE5_LAYOUTS)

    # Preserve the native Rule-5 quantities; no conversion to Aer parameters.
    rule5 = build_native_rule5_table(selected_layouts)
    rule5.to_csv(OUT_RULE5, index=False)

    configured_inventory = configured_alias_inventory()
    aliases_to_probe = list(
        configured_inventory["aliases_to_probe"]
    )

    inventory_output = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "iqm_client_version": package_version("iqm-client"),
            "qiskit_version": package_version("qiskit"),
            "iqm_qdmi_version": package_version("iqm-qdmi"),
            "server_url": server_url.rstrip("/"),
            "authentication_source": "iqm_account.py",
            "token_saved": False,
        },
        **configured_inventory,
    }
    with open(OUT_INVENTORY, "w", encoding="utf-8") as f:
        json.dump(inventory_output, f, indent=2)

    print("=" * 126)
    print("WEEK 12.5A v2 — IQM NATIVE RULE-5 / SIMULATOR CAPABILITY PROBE")
    print("=" * 126)
    print("No QPU jobs. No simulator jobs. No shots. 2026 untouched.")
    print()
    print(f"Configured real aliases: {list(REAL_ALIASES)}")
    print(
        f"Documented mock aliases to probe: "
        f"{list(DOCUMENTED_MOCK_ALIASES)}"
    )

    qdmi_version = package_version("iqm-qdmi")
    print(f"iqm-qdmi installed: {qdmi_version or 'NO'}")

    if not qdmi_version:
        print(
            'NOTE: native :mock probing requires '
            'pip install "iqm-qdmi[qiskit]"'
        )

    client_probes = {}
    qdmi_probes = {}
    for alias in aliases_to_probe:
        print()
        print("-" * 126)
        print(f"PROBE ALIAS: {alias}")

        cp = probe_iqm_client_alias(alias, server_url, token)
        client_probes[alias] = cp
        print(
            "IQMClient metadata access: "
            + ("YES" if cp.get("client_created") else "NO")
        )
        if cp.get("client_created"):
            counts = cp.get("native_metric_field_counts", {})
            print(
                "Native metric field evidence: "
                + ", ".join(f"{k}={v}" for k, v in counts.items())
            )

        qp = probe_qdmi_backend(alias, server_url, token)
        qdmi_probes[alias] = qp
        print(
            "QDMI backend metadata access: "
            + ("YES" if qp.get("backend_created") else "NO")
        )
        if qp.get("backend_created"):
            tc = qp.get("target_coverage", {})
            print(
                "QDMI target: "
                f"{tc.get('instruction_loci_with_duration', 0)}/"
                f"{tc.get('instruction_loci_total', 0)} loci with duration; "
                f"{tc.get('instruction_loci_with_error', 0)}/"
                f"{tc.get('instruction_loci_total', 0)} loci with error"
            )

    probe_output = {
        "aliases_to_probe": aliases_to_probe,
        "client_probes": client_probes,
        "qdmi_probes": qdmi_probes,
        "job_submitted": False,
        "shots": 0,
    }
    with open(OUT_PROBE, "w", encoding="utf-8") as f:
        json.dump(probe_output, f, indent=2)

    assessment = {
        "native_rule5": {
            "status": "READY",
            "method": (
                "Use IQM-native calibration quantities directly: native "
                "embedding/SWAP, CZ fidelity, readout fidelity, T1, T2_echo, "
                "memory T1/T2_echo, compiled CZ/depth."
            ),
            "external_aer_conversion_required_for_rule5": False,
            "selected_layout_count": int(len(rule5)),
        },
        "simulators": {},
        "overall_recommendation": None,
    }

    for real_alias in REAL_ALIASES:
        mock_alias = f"{real_alias}:mock"
        cp = client_probes.get(mock_alias, {})
        qp = qdmi_probes.get(mock_alias, {})
        # No raw REST inventory is used. Explicit simulator/noise semantics
        # must come from the official IQMClient/QDMI metadata probes.
        inv_items = []
        explicit = explicit_semantic_evidence(inv_items, cp, qp)
        backend_exists = bool(
            cp.get("client_created")
            or qp.get("backend_created")
        )

        if not backend_exists:
            status = "NOT_ACCESSIBLE_OR_NOT_INSTALLED"
        elif explicit:
            status = "EXPLICIT_NOISE_SEMANTICS_METADATA_FOUND"
        else:
            status = "MOCK_EXISTS_BUT_NOISY_EXECUTION_UNVERIFIED"

        assessment["simulators"][real_alias] = {
            "mock_alias": mock_alias,
            "backend_exists_or_metadata_accessible": backend_exists,
            "status": status,
            "explicit_noise_semantics_evidence": explicit,
            "client_native_metric_field_counts": cp.get(
                "native_metric_field_counts", {}
            ),
            "qdmi_target_coverage": qp.get("target_coverage", {}),
            "interpretation": (
                "Do not treat the mock as a calibration-derived noisy "
                "simulator unless execution semantics are explicitly "
                "documented or later empirically validated."
            ),
        }

    statuses = [x["status"] for x in assessment["simulators"].values()]
    if any(s == "EXPLICIT_NOISE_SEMANTICS_METADATA_FOUND" for s in statuses):
        assessment["overall_recommendation"] = (
            "Explicit noise-related simulator metadata was found. Inspect it "
            "before any execution; then validate with a tiny controlled test "
            "before using the mock as a noisy-simulator scientific result."
        )
    elif any(s == "MOCK_EXISTS_BUT_NOISY_EXECUTION_UNVERIFIED" for s in statuses):
        assessment["overall_recommendation"] = (
            "Use IQM-native metrics for Rule 5. Treat IQM mock backends as "
            "execution/simulator infrastructure only until their noise "
            "semantics are verified; do not call them calibration-derived "
            "noisy models."
        )
    else:
        assessment["overall_recommendation"] = (
            "Use IQM-native metrics for Rule 5. Native mock/simulator access "
            "was not established in this environment; do not substitute an "
            "unverified external noise model yet."
        )

    with open(OUT_ASSESSMENT, "w", encoding="utf-8") as f:
        json.dump(assessment, f, indent=2)

    print()
    print("=" * 126)
    print("12.5A NATIVE CAPABILITY ASSESSMENT")
    print("=" * 126)
    print(f"Native Rule-5 layouts retained: {len(rule5)}")
    print("Rule 5 external Aer conversion: NO")
    for real_alias in REAL_ALIASES:
        info = assessment["simulators"][real_alias]
        print(f"{info['mock_alias']:<16} {info['status']}")
    print()
    print("Recommendation:")
    print(assessment["overall_recommendation"])
    print()
    print("Saved:")
    print(f"  {OUT_INVENTORY}")
    print(f"  {OUT_PROBE}")
    print(f"  {OUT_RULE5}")
    print(f"  {OUT_ASSESSMENT}")
    print()
    print(
        "STOP HERE. Send the console output before any mock/simulator "
        "execution or QPU submission."
    )


if __name__ == "__main__":
    main()
