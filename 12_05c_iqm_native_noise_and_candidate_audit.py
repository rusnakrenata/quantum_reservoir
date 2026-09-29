"""
Week 12.5C — IQM native noisy-simulator + full candidate-universe audit.

Checks before any noisy QRC sweep:
  1) Which installed IQMFakeBackend matches Emerald/Garnet architecture.
  2) What IQMErrorProfile fields/durations the fake backend exposes.
  3) Whether live IQM calibration/quality payloads expose duration-like fields.
  4) Full logical candidate universe:
         H0-H3 = 25
         H4    = 8
         H5-H6 = 7
         TOTAL = 40

No QPU jobs. No noisy QRC simulation. No shots. 2026 untouched.
Authentication comes only from iqm_account.py.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import pkgutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iqm.iqm_client import IQMClient
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMFakeBackend, IQMErrorProfile
from iqm_account import get_authentication

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

OUT_FAKE_INV = RESULTS / "12_05c_iqm_fake_backend_inventory.csv"
OUT_FAKE_PROFILE = RESULTS / "12_05c_iqm_fake_error_profiles.json"
OUT_COMPAT = RESULTS / "12_05c_iqm_real_fake_compatibility.csv"
OUT_DURATION = RESULTS / "12_05c_iqm_duration_field_audit.csv"
OUT_CAND = RESULTS / "12_05c_iqm_candidate_universe.csv"
OUT_CAND_SUM = RESULTS / "12_05c_iqm_candidate_universe_summary.csv"
OUT_JSON = RESULTS / "12_05c_iqm_native_noise_audit.json"

REAL_ALIASES = ("emerald", "garnet")
EXPECTED_COUNTS = {"H0_H3": 25, "H4": 8, "H5_H6": 7}
EXPECTED_TOTAL = 40


def jsonable(x: Any):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if hasattr(x, "model_dump"):
        try:
            return jsonable(x.model_dump())
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return jsonable(x.dict())
        except Exception:
            pass
    if hasattr(x, "__dict__"):
        try:
            return jsonable(vars(x))
        except Exception:
            pass
    return str(x)


def finite_float(x):
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def numeric_stats(mapping):
    vals = []
    def walk(v):
        if isinstance(v, dict):
            for vv in v.values():
                walk(vv)
        else:
            z = finite_float(v)
            if z is not None:
                vals.append(z)
    walk(mapping)
    if not vals:
        return {"n": 0, "min": None, "median": None, "max": None}
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(len(a)),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "max": float(a.max()),
    }


def get_real_metadata(server_url, token):
    real = {}
    for alias in REAL_ALIASES:
        client = IQMClient(server_url, quantum_computer=alias, token=token)
        real[alias] = {
            "static_architecture": client.get_static_quantum_architecture(),
            "calibration": client.get_calibration_set(),
            "quality": client.get_quality_metric_set(),
        }
    return real


def zero_required_constructor(obj):
    try:
        sig = inspect.signature(obj)
    except Exception:
        return False
    for p in sig.parameters.values():
        if p.name in ("self", "cls"):
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect.Parameter.empty:
            return False
    return True


def discover_fake_backends():
    package = importlib.import_module("iqm.qiskit_iqm.fake_backends")
    modules = [package.__name__]
    if hasattr(package, "__path__"):
        for info in pkgutil.iter_modules(package.__path__):
            if info.name.startswith("fake_"):
                modules.append(f"{package.__name__}.{info.name}")

    seen = set()
    instances = []
    failures = []

    for module_name in sorted(set(modules)):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            failures.append({"module": module_name, "object": None, "error": repr(exc)})
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            try:
                is_fake = issubclass(obj, IQMFakeBackend) and obj is not IQMFakeBackend
            except Exception:
                is_fake = False
            if not is_fake:
                continue

            key = f"{obj.__module__}.{obj.__qualname__}"
            if key in seen:
                continue
            seen.add(key)

            if not zero_required_constructor(obj):
                failures.append({"module": module_name, "object": key, "error": "required constructor arguments"})
                continue

            try:
                instances.append({"class_path": key, "backend": obj()})
            except Exception as exc:
                failures.append({"module": module_name, "object": key, "error": repr(exc)})

    return instances, failures


def summarize_profile(profile: IQMErrorProfile):
    fields = [
        "t1s",
        "t2s",
        "single_qubit_gate_depolarizing_error_parameters",
        "two_qubit_gate_depolarizing_error_parameters",
        "single_qubit_gate_durations",
        "two_qubit_gate_durations",
        "readout_errors",
    ]
    out = {"name": getattr(profile, "name", None)}
    for field in fields:
        out[field] = jsonable(getattr(profile, field, {}))
    return out


def audit_fake_backends(instances, real):
    inv_rows = []
    compat_rows = []
    profiles = {}

    for item in instances:
        backend = item["backend"]
        cls = item["class_path"]
        profile = getattr(backend, "error_profile", None)
        pdata = summarize_profile(profile) if profile is not None else {}
        profiles[cls] = pdata

        t1 = numeric_stats(pdata.get("t1s", {}))
        t2 = numeric_stats(pdata.get("t2s", {}))
        e1 = numeric_stats(pdata.get("single_qubit_gate_depolarizing_error_parameters", {}))
        e2 = numeric_stats(pdata.get("two_qubit_gate_depolarizing_error_parameters", {}))
        ro = numeric_stats(pdata.get("readout_errors", {}))

        physical = list(getattr(backend, "physical_qubits", []))

        inv_rows.append({
            "fake_class": cls,
            "backend_name": str(getattr(backend, "name", None)),
            "n_physical_components": len(physical),
            "error_profile_name": pdata.get("name"),
            "n_t1": t1["n"],
            "n_t2": t2["n"],
            "n_1q_error_parameters": e1["n"],
            "n_2q_error_parameters": e2["n"],
            "n_readout_error_parameters": ro["n"],
            "single_qubit_gate_durations_json": json.dumps(pdata.get("single_qubit_gate_durations", {}), sort_keys=True),
            "two_qubit_gate_durations_json": json.dumps(pdata.get("two_qubit_gate_durations", {}), sort_keys=True),
        })

        for alias, payload in real.items():
            compatible = None
            error = None
            try:
                compatible = bool(backend.validate_compatible_architecture(payload["static_architecture"]))
            except Exception as exc:
                error = repr(exc)
            compat_rows.append({
                "real_alias": alias,
                "fake_class": cls,
                "fake_backend_name": str(getattr(backend, "name", None)),
                "architecture_compatible": compatible,
                "compatibility_error": error,
            })

    return pd.DataFrame(inv_rows), pd.DataFrame(compat_rows), profiles


def collect_duration_fields(obj, source, alias, path=""):
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            kl = str(k).lower()
            pl = p.lower()
            if any(tok in kl for tok in ("duration", "time", "length")) and not isinstance(v, (dict, list, tuple)):
                rows.append({
                    "real_alias": alias,
                    "source": source,
                    "path": p,
                    "value": v,
                    "path_mentions_prx": "prx" in pl,
                    "path_mentions_cz": ".cz" in pl or "cz." in pl,
                    "path_mentions_measure": "measure" in pl,
                })
            rows.extend(collect_duration_fields(v, source, alias, p))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            rows.extend(collect_duration_fields(v, source, alias, f"{path}[{i}]"))
    return rows


def duration_audit(real):
    rows = []
    for alias, payload in real.items():
        for source in ("calibration", "quality"):
            rows.extend(collect_duration_fields(jsonable(payload[source]), source, alias))
    return pd.DataFrame(rows)


def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def id_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def normalize_candidates(df, source_group, source_file, id_names):
    cid = id_col(df, id_names)
    if cid is None:
        raise RuntimeError(f"No candidate ID column in {source_file}; columns={list(df.columns)}")

    w = df.copy()
    w["candidate_id"] = w[cid].astype(str).str.strip()
    w = w[w["candidate_id"].ne("")].drop_duplicates("candidate_id").copy()

    def series(names, default=None):
        for c in names:
            if c in w.columns:
                return w[c]
        return pd.Series([default] * len(w), index=w.index)

    return pd.DataFrame({
        "source_group": source_group,
        "source_file": str(source_file),
        "candidate_id": w["candidate_id"],
        "topology": series(["topology", "H"]),
        "protocol": series(["protocol", "mode"]),
        "selected_rules": series(["selected_rules", "selected_by_rules", "rules"]),
        "window": series(["window", "window_size", "W"], np.nan),
        "r": series(["original_r", "r", "trotter_r"], np.nan),
        "readout": series(["test_readout", "readout", "readout_name"]),
    })


def load_candidate_universe():
    p_h03 = first_existing([RESULTS / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"])
    p_h4 = first_existing([
        RESULTS / "10_06a_h4_ideal_depth_audit.csv",
        RESULTS / "10_06b_h4_candidate_summary.csv",
    ])
    p_h56 = first_existing([RESULTS / "12_04b_iqm_rule1_rule4_candidates.csv"])

    missing = []
    if p_h03 is None:
        missing.append("10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv")
    if p_h4 is None:
        missing.append("10_06a_h4_ideal_depth_audit.csv (or 10_06b_h4_candidate_summary.csv)")
    if p_h56 is None:
        missing.append("12_04b_iqm_rule1_rule4_candidates.csv")
    if missing:
        raise FileNotFoundError("Missing result file(s): " + ", ".join(missing))

    h03 = normalize_candidates(pd.read_csv(p_h03), "H0_H3", p_h03, ["candidate_key", "candidate_uid", "candidate_id"])
    h4 = normalize_candidates(pd.read_csv(p_h4), "H4", p_h4, ["candidate_key", "candidate_uid", "candidate_id"])
    h56 = normalize_candidates(pd.read_csv(p_h56), "H5_H6", p_h56, ["candidate_uid", "candidate_key", "candidate_id"])

    combined = pd.concat([h03, h4, h56], ignore_index=True)
    duplicates = combined[combined["candidate_id"].duplicated(keep=False)].copy()

    summary = []
    for group in ("H0_H3", "H4", "H5_H6"):
        n = int(combined.loc[combined["source_group"].eq(group), "candidate_id"].nunique())
        summary.append({
            "source_group": group,
            "observed_unique_candidates": n,
            "expected_unique_candidates": EXPECTED_COUNTS[group],
            "count_pass": n == EXPECTED_COUNTS[group],
        })

    total = int(combined["candidate_id"].nunique())
    summary.append({
        "source_group": "TOTAL",
        "observed_unique_candidates": total,
        "expected_unique_candidates": EXPECTED_TOTAL,
        "count_pass": total == EXPECTED_TOTAL and duplicates.empty,
    })

    return combined, pd.DataFrame(summary), duplicates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = [OUT_FAKE_INV, OUT_FAKE_PROFILE, OUT_COMPAT, OUT_DURATION, OUT_CAND, OUT_CAND_SUM, OUT_JSON]
    if args.overwrite:
        for p in outputs:
            if p.exists():
                p.unlink()

    token, server_url, _ = get_authentication()

    print("=" * 118)
    print("WEEK 12.5C — IQM NATIVE NOISY-SIMULATOR + FULL H0-H6 CANDIDATE AUDIT")
    print("=" * 118)
    print("No QPU jobs. No noisy QRC simulation. No shots. 2026 untouched.")

    print("\n[1/4] Loading Emerald/Garnet architecture + calibration metadata...")
    real = get_real_metadata(server_url, token)
    for alias in REAL_ALIASES:
        print(f"  {alias}: metadata loaded")

    print("\n[2/4] Discovering installed IQMFakeBackend implementations...")
    instances, failures = discover_fake_backends()
    fake_inv, compat, profiles = audit_fake_backends(instances, real)
    fake_inv.to_csv(OUT_FAKE_INV, index=False)
    compat.to_csv(OUT_COMPAT, index=False)
    with open(OUT_FAKE_PROFILE, "w", encoding="utf-8") as f:
        json.dump({"profiles": profiles, "discovery_failures": failures}, f, indent=2)

    if fake_inv.empty:
        print("  WARNING: no zero-argument IQMFakeBackend subclass instantiated")
    else:
        print(fake_inv[[
            "fake_class", "backend_name", "n_physical_components", "error_profile_name",
            "n_t1", "n_t2", "n_1q_error_parameters", "n_2q_error_parameters",
            "n_readout_error_parameters",
        ]].to_string(index=False))

    print("\nREAL ↔ FAKE ARCHITECTURE COMPATIBILITY:")
    if compat.empty:
        print("  none")
    else:
        print(compat[["real_alias", "fake_backend_name", "architecture_compatible"]].to_string(index=False))

    print("\n[3/4] Auditing live calibration/quality payload for duration-like fields...")
    durations = duration_audit(real)
    durations.to_csv(OUT_DURATION, index=False)
    if durations.empty:
        print("  no duration/time/length-like scalar fields found")
    else:
        for alias in REAL_ALIASES:
            sub = durations[durations["real_alias"].eq(alias)]
            focus = sub[sub[["path_mentions_prx", "path_mentions_cz", "path_mentions_measure"]].any(axis=1)]
            print(f"  {alias}: {len(sub)} total duration/time-like fields; {len(focus)} PRX/CZ/measure-related")
            for _, row in focus.head(12).iterrows():
                print(f"    {row['source']}: {row['path']} = {row['value']}")

    print("\n[4/4] Auditing full frozen logical candidate universe H0-H6...")
    candidates, cand_summary, duplicate_ids = load_candidate_universe()
    candidates.to_csv(OUT_CAND, index=False)
    cand_summary.to_csv(OUT_CAND_SUM, index=False)
    print(cand_summary.to_string(index=False))

    count_pass = bool(cand_summary["count_pass"].all())
    compatible = []
    if not compat.empty:
        for _, row in compat.iterrows():
            if bool(row["architecture_compatible"]) if pd.notna(row["architecture_compatible"]) else False:
                compatible.append({
                    "real_alias": row["real_alias"],
                    "fake_class": row["fake_class"],
                    "fake_backend_name": row["fake_backend_name"],
                })

    audit = {
        "stage": "12.5C",
        "qpu_jobs_submitted": 0,
        "shots": 0,
        "noisy_qrc_simulation_run": False,
        "test_2026_used": False,
        "candidate_expected_counts": EXPECTED_COUNTS,
        "candidate_expected_total": EXPECTED_TOTAL,
        "candidate_count_audit_pass": count_pass,
        "candidate_observed_summary": cand_summary.to_dict(orient="records"),
        "duplicate_candidate_ids": duplicate_ids["candidate_id"].astype(str).tolist() if not duplicate_ids.empty else [],
        "compatible_fake_backends": compatible,
        "n_instantiated_fake_backends": int(len(fake_inv)),
        "n_live_duration_like_fields": int(len(durations)),
        "next_step": "Construct the full H0-H6 IQM noisy-simulation ladder only after interpreting this audit; then select five QPU candidates.",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("\n" + "=" * 118)
    print("12.5C AUDIT COMPLETE")
    print("=" * 118)
    print(f"Candidate universe count audit: {'PASS' if count_pass else 'FAIL'}")
    print(f"Architecture-compatible fake backends found: {len(compatible)}")
    for item in compatible:
        print(f"  {item['real_alias']} <- {item['fake_backend_name']} ({item['fake_class']})")

    print("\nSaved:")
    for p in outputs:
        print(f"  {p}")

    print("\nSTOP HERE. Send this console output before constructing the full 40-candidate noisy QRC sweep.")


if __name__ == "__main__":
    main()
