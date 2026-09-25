#!/usr/bin/env python
from __future__ import annotations

"""
WEEK 11.1F.3 — CANDIDATE #2 QISKIT-SELECTED PHYSICAL LAYOUT
===========================================================

Purpose
-------
Repeat the original Candidate #2 direct-QPU SamplerV2 experiment, but replace
our custom Rule-5 physical-layout selection with Qiskit's own transpiler layout
selection.

Important experimental rule
---------------------------
Qiskit chooses the physical layout ONCE from one representative frozen
Candidate #2 circuit.  That selected six-qubit layout is then frozen and reused
for every 2025 endpoint and both grouped measurement settings.

This avoids allowing 730 circuits to choose 730 different layouts.

Everything else is inherited from the authoritative original script:
    11_1B_second_candidate_direct_qpu.py

Therefore this wrapper preserves:
- Candidate #2 logical dynamics / readout / frozen Ridge
- 2022-2024 training-only readout fit
- 2025 validation only
- 2026 NEVER LOADED
- plain SamplerV2
- no M3 / no Estimator mitigation
- original shot count / CLI / QPU timing / parsing
- database persistence

Database
--------
The Qiskit-selected layout is written to:
- physical_layout_json
- hardware_selection_json

The latter also records:
- selection_method = qiskit_transpiler_default
- Qiskit seed and optimization level
- representative probe information
- current T1/T2/readout metrics
- compiled 1Q/2Q metrics
- zero-SWAP strict recompile audit

No custom Rule-5 ranking is used to CHOOSE the layout in this script.
Rule-5 code is used only for metric helper functions after Qiskit has already
made the selection.

Default is DRY RUN.  Use --submit exactly as with the original 11.1B script.
"""

import importlib.util
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import transpile


HERE = Path(__file__).resolve().parent
BASE_FILE = HERE / "11_1B_second_candidate_direct_qpu.py"
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

NEW_PREFIX = "11_1F3_candidate2_qiskit_layout_qpu"
OLD_PREFIX = "11_1B_second_candidate_qpu"

QISKIT_SELECTION_RECORD = RESULTS / f"{NEW_PREFIX}_layout_selection.json"

# The original script writes these names.  We preserve the original files by
# temporarily backing them up, then copy this run to a new 11_1F3 prefix.
KNOWN_SUFFIXES = [
    "_preflight.json",
    "_resources.csv",
    "_selected_endpoints.csv",
    "_raw_counts.csv",
    "_features_predictions.csv",
    "_summary.json",
    "_job.json",
]


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put this script in the same project directory "
            "as 11_1B_second_candidate_direct_qpu.py."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = load_module(BASE_FILE, "qrc_11_1f3_base_candidate2")

# Keep a reference to the selector module and helper methods BEFORE monkeypatch.
selector = base.selector


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _extract_initial_layout(transpiled, logical, n_logical=6):
    """
    Return [physical(q0),...,physical(q5)] from Qiskit's TranspileLayout.
    """
    tl = getattr(transpiled, "layout", None)
    if tl is None:
        raise RuntimeError("Qiskit transpiler returned no layout information.")

    # Preferred public helper in current Qiskit.
    fn = getattr(tl, "initial_index_layout", None)
    if callable(fn):
        try:
            vals = fn(filter_ancillas=True)
            vals = [int(v) for v in vals]
            if len(vals) >= n_logical:
                return vals[:n_logical]
        except TypeError:
            vals = fn()
            vals = [int(v) for v in vals]
            if len(vals) >= n_logical:
                return vals[:n_logical]
        except Exception:
            pass

    # Fallback through Layout virtual-bit mapping.
    init = getattr(tl, "initial_layout", None)
    if init is None:
        raise RuntimeError("TranspileLayout has no initial_layout.")

    out = []
    for q in logical.qubits[:n_logical]:
        try:
            out.append(int(init[q]))
        except Exception as exc:
            raise RuntimeError(
                "Could not recover Qiskit's physical layout for logical qubits."
            ) from exc
    return out


def _extract_final_layout(transpiled, n_logical=6):
    tl = getattr(transpiled, "layout", None)
    if tl is None:
        return None

    fn = getattr(tl, "final_index_layout", None)
    if callable(fn):
        try:
            vals = fn(filter_ancillas=True)
        except TypeError:
            vals = fn()
        except Exception:
            return None

        try:
            vals = [int(v) for v in vals]
            if len(vals) >= n_logical:
                return vals[:n_logical]
        except Exception:
            return None

    return None


def _instruction_error(target, name, qargs):
    try:
        props = target[name][tuple(int(q) for q in qargs)]
    except Exception:
        return np.nan
    if props is None:
        return np.nan
    return _safe_float(getattr(props, "error", np.nan))


def _compiled_instruction_errors(circuit, backend):
    """
    Compute max/mean calibrated 1Q and 2Q errors over instructions actually
    present in the compiled selector-core circuit.
    """
    one_q = []
    two_q = []

    for inst in circuit.data:
        op = inst.operation
        name = str(op.name)
        if name in {"measure", "reset", "delay", "barrier"}:
            continue

        qargs = [int(circuit.find_bit(q).index) for q in inst.qubits]

        if len(qargs) == 1:
            e = _instruction_error(backend.target, name, qargs)
            if np.isfinite(e):
                one_q.append(float(e))

        elif len(qargs) == 2:
            e = _instruction_error(backend.target, name, qargs)
            if np.isfinite(e):
                two_q.append(float(e))

    return {
        "compiled_1q_error_max": max(one_q) if one_q else np.nan,
        "compiled_1q_error_mean": float(np.mean(one_q)) if one_q else np.nan,
        "compiled_2q_error_max": max(two_q) if two_q else np.nan,
        "compiled_2q_error_mean": float(np.mean(two_q)) if two_q else np.nan,
    }


def _qubit_metrics(backend, layout):
    props = backend.properties()

    t1_us = []
    t2_us = []
    ro = []

    for q in layout:
        q = int(q)

        try:
            t1_us.append(float(props.t1(q)) * 1e6)
        except Exception:
            t1_us.append(np.nan)

        try:
            t2_us.append(float(props.t2(q)) * 1e6)
        except Exception:
            t2_us.append(np.nan)

        try:
            ro.append(float(props.readout_error(q)))
        except Exception:
            ro.append(np.nan)

    def finite(vals):
        return [float(v) for v in vals if np.isfinite(v)]

    ft1 = finite(t1_us)
    ft2 = finite(t2_us)
    fro = finite(ro)

    return {
        "min_t1_us": min(ft1) if ft1 else np.nan,
        "mean_t1_us": float(np.mean(ft1)) if ft1 else np.nan,
        "min_t2_us": min(ft2) if ft2 else np.nan,
        "mean_t2_us": float(np.mean(ft2)) if ft2 else np.nan,
        "max_readout_error": max(fro) if fro else np.nan,
        "mean_readout_error": float(np.mean(fro)) if fro else np.nan,
        "per_qubit_t1_us": {
            f"P{int(q)}": _safe_float(v) for q, v in zip(layout, t1_us)
        },
        "per_qubit_t2_us": {
            f"P{int(q)}": _safe_float(v) for q, v in zip(layout, t2_us)
        },
        "per_qubit_readout_error": {
            f"P{int(q)}": _safe_float(v) for q, v in zip(layout, ro)
        },
    }


def _duration_us(circuit, backend):
    return float(circuit.estimate_duration(backend.target, unit="s")) * 1e6


def qiskit_layout_reselection(
    service,
    candidate,
    backend_names,
    shortlist=120,
    write_prefix=None,
    verbose=True,
    **kwargs,
):
    """
    Drop-in replacement for selector.fresh_hardware_reselection.

    Qiskit chooses the physical placement.  We DO NOT enumerate/rank native
    embeddings with Rule 5.  Once chosen, the layout is frozen and audited.
    """
    if not backend_names:
        raise ValueError("backend_names is empty.")

    backend_name = str(backend_names[0])
    backend = service.backend(
        backend_name,
        use_fractional_gates=False,
    )
    try:
        backend.refresh()
    except Exception:
        pass

    # Representative W=2 Candidate #2 circuit.
    # Non-zero dummy input angles prevent accidental simplification of RY(0).
    W = int(candidate["window"])
    dummy = np.asarray(
        [
            [0.271, -0.613, 0.943, -0.421],
            [0.337, -0.517, 0.821, -0.283],
        ],
        dtype=float,
    )
    if W != len(dummy):
        raise RuntimeError(
            f"This C2 Qiskit-layout experiment expects W=2; got W={W}."
        )

    settings = base.readout_settings()
    if not settings:
        raise RuntimeError("Candidate #2 has no measurement settings.")

    probe_setting = settings[0]
    logical_probe = base.build_measurement_circuit(
        candidate,
        dummy,
        endpoint=W - 1,
        setting=probe_setting,
    )

    print("=" * 126)
    print(f"QISKIT TRANSPILER PHYSICAL-LAYOUT SELECTION — {backend_name}")
    print("=" * 126)
    print("Custom Rule-5 physical selection: OFF")
    print("Qiskit transpiler chooses the layout ONCE.")
    print(f"Probe setting:                 {probe_setting[0]}")
    print(f"optimization_level:            {base.OPT_LEVEL}")
    print(f"seed_transpiler:               {base.SEED_TRANSPILE}")

    # IMPORTANT: no initial_layout, no custom layout_method and no custom
    # routing_method.  This is Qiskit's standard transpiler placement.
    qiskit_probe = transpile(
        logical_probe,
        backend=backend,
        optimization_level=int(base.OPT_LEVEL),
        seed_transpiler=int(base.SEED_TRANSPILE),
        scheduling_method="alap",
    )
    backend.check_faulty(qiskit_probe)

    layout = _extract_initial_layout(
        qiskit_probe,
        logical_probe,
        n_logical=6,
    )
    final_layout = _extract_final_layout(qiskit_probe, n_logical=6)

    probe_ops = {
        str(k): int(v)
        for k, v in qiskit_probe.count_ops().items()
    }

    print(f"Qiskit-selected initial layout: {layout}")
    if final_layout is not None:
        print(f"Qiskit final index layout:      {final_layout}")
    print(f"Qiskit probe depth:             {qiskit_probe.depth()}")
    print(f"Qiskit probe operations:        {json.dumps(probe_ops, sort_keys=True)}")

    # ------------------------------------------------------------------
    # Freeze Qiskit's chosen placement and demand a native zero-routing
    # recompile.  This makes the following 730-circuit run comparable with
    # the original strict-layout experiment.
    # ------------------------------------------------------------------
    strict_probe = transpile(
        logical_probe,
        backend=backend,
        initial_layout=layout,
        routing_method="none",
        optimization_level=int(base.OPT_LEVEL),
        seed_transpiler=int(base.SEED_TRANSPILE),
        scheduling_method="alap",
    )
    backend.check_faulty(strict_probe)

    strict_ops = {
        str(k): int(v)
        for k, v in strict_probe.count_ops().items()
    }
    strict_swap = int(strict_ops.get("swap", 0))

    # Even when SWAP is decomposed into basis gates, routing_method="none"
    # would have failed if the chosen placement required routing.
    if strict_swap != 0:
        raise RuntimeError(
            "Qiskit's selected layout did not pass the strict zero-SWAP audit."
        )

    # ------------------------------------------------------------------
    # For database comparability, evaluate the usual selector-core metrics
    # on Qiskit's ALREADY-CHOSEN physical layout.  These metrics do not choose
    # the layout; they only describe it.
    # ------------------------------------------------------------------
    try:
        first_angles, _ = selector.first_input_angles()
        logical_core = selector.build_core(candidate, first_angles)
        compiled_core = transpile(
            logical_core,
            backend=backend,
            initial_layout=layout,
            routing_method="none",
            optimization_level=int(base.OPT_LEVEL),
            seed_transpiler=int(base.SEED_TRANSPILE),
            scheduling_method="alap",
        )
        backend.check_faulty(compiled_core)
    except Exception:
        # Fallback to the representative measurement circuit if selector
        # helper signatures ever change.
        compiled_core = strict_probe

    core_ops = {
        str(k): int(v)
        for k, v in compiled_core.count_ops().items()
    }

    try:
        cm = selector.compiled_metrics(compiled_core, backend)
        cm = dict(cm)
    except Exception:
        cm = {}

    ierr = _compiled_instruction_errors(compiled_core, backend)
    for key, value in ierr.items():
        cm.setdefault(key, value)

    qmet = _qubit_metrics(backend, layout)

    compiled_duration_us = _safe_float(
        cm.get("compiled_duration_us"),
        _duration_us(compiled_core, backend),
    )

    min_t1 = _safe_float(qmet["min_t1_us"])
    min_t2 = _safe_float(qmet["min_t2_us"])

    r_t1 = (
        compiled_duration_us / min_t1
        if np.isfinite(min_t1) and min_t1 > 0
        else np.nan
    )
    r_t2 = (
        compiled_duration_us / min_t2
        if np.isfinite(min_t2) and min_t2 > 0
        else np.nan
    )

    max_1q = _safe_float(cm.get("compiled_1q_error_max"))
    max_2q = _safe_float(cm.get("compiled_2q_error_max"))
    mean_1q = _safe_float(cm.get("compiled_1q_error_mean"))
    mean_2q = _safe_float(cm.get("compiled_2q_error_mean"))
    max_ro = _safe_float(qmet["max_readout_error"])
    mean_ro = _safe_float(qmet["mean_readout_error"])

    status = backend.status()

    chosen = {
        "backend": backend_name,
        "pending_jobs": int(getattr(status, "pending_jobs", 0)),
        "layout": json.dumps([int(q) for q in layout]),

        # Logical-to-physical roles.
        "physical_C_t": int(layout[0]),
        "physical_D": int(layout[1]),
        "physical_P_t": int(layout[2]),
        "physical_H": int(layout[3]),
        "physical_M1": int(layout[4]),
        "physical_M2": int(layout[5]),

        # Explicit selection provenance.
        "selection_method": "qiskit_transpiler_default",
        "custom_rule5_selection_used": False,
        "qiskit_optimization_level": int(base.OPT_LEVEL),
        "qiskit_seed_transpiler": int(base.SEED_TRANSPILE),
        "qiskit_probe_setting": str(probe_setting[0]),
        "qiskit_probe_initial_layout": [int(q) for q in layout],
        "qiskit_probe_final_index_layout": final_layout,
        "qiskit_probe_depth": int(qiskit_probe.depth()),
        "qiskit_probe_size": int(qiskit_probe.size()),
        "qiskit_probe_operations": probe_ops,
        "qiskit_probe_duration_us": _duration_us(qiskit_probe, backend),

        # Strict frozen-layout audit.
        "strict_compile_pass": True,
        "compiled_n_swap": int(core_ops.get("swap", 0)),
        "compile_error": "",

        # Comparable hardware metrics.
        "compiled_1q_error_max": max_1q,
        "compiled_1q_error_max_percent": 100.0 * max_1q,
        "compiled_1q_error_mean": mean_1q,
        "compiled_2q_error_max": max_2q,
        "compiled_2q_error_max_percent": 100.0 * max_2q,
        "compiled_2q_error_mean": mean_2q,
        "max_cz_error": max_2q,
        "max_cz_error_percent": 100.0 * max_2q,
        "mean_cz_error": mean_2q,
        "mean_cz_error_percent": 100.0 * mean_2q,
        "max_readout_error": max_ro,
        "max_readout_error_percent": 100.0 * max_ro,
        "mean_readout_error": mean_ro,
        "mean_readout_error_percent": 100.0 * mean_ro,
        "min_t1_us": min_t1,
        "mean_t1_us": _safe_float(qmet["mean_t1_us"]),
        "min_t2_us": min_t2,
        "mean_t2_us": _safe_float(qmet["mean_t2_us"]),
        "compiled_duration_us": compiled_duration_us,
        "R_T1": r_t1,
        "R_T2": r_t2,

        "compiled_depth": int(cm.get("compiled_depth", compiled_core.depth())),
        "compiled_size": int(cm.get("compiled_size", compiled_core.size())),
        "compiled_n_cz": int(
            cm.get("compiled_n_cz", core_ops.get("cz", 0))
        ),
        "compiled_n_2q": int(
            cm.get("compiled_n_2q", core_ops.get("cz", 0))
        ),
        "compiled_n_1q": int(
            cm.get(
                "compiled_n_1q",
                sum(
                    int(v)
                    for k, v in core_ops.items()
                    if k not in {
                        "cz", "measure", "reset", "delay", "barrier"
                    }
                ),
            )
        ),
        "operations": json.dumps(core_ops, sort_keys=True),

        # Qubit-resolved diagnostics kept inside hardware_selection_json.
        "per_qubit_t1_us": qmet["per_qubit_t1_us"],
        "per_qubit_t2_us": qmet["per_qubit_t2_us"],
        "per_qubit_readout_error": qmet["per_qubit_readout_error"],

        # Pareto rank is intentionally undefined: Qiskit, not our selector,
        # made the physical placement decision.
        "calibration_pareto": None,
        "overall_rank": None,
    }

    QISKIT_SELECTION_RECORD.write_text(
        json.dumps(base.json_safe(chosen), indent=2),
        encoding="utf-8",
    )

    print()
    print("FROZEN QISKIT PHYSICAL CHOICE")
    print(f"  layout={layout}")
    print(f"  max 2Q error={100.0 * max_2q:.6f}%")
    print(f"  max readout error={100.0 * max_ro:.6f}%")
    print(f"  max 1Q error={100.0 * max_1q:.6f}%")
    print(f"  min T1={min_t1:.3f} us")
    print(f"  min T2={min_t2:.3f} us")
    print(f"  selector-core duration={compiled_duration_us:.3f} us")
    print(f"  R_T2={r_t2:.6f}")
    print("  strict frozen-layout routing audit: PASS")
    print()

    # The base script only needs fresh["selected"], but return a transparent
    # one-row compiled table too.
    return {
        "selected": pd.Series(chosen),
        "compiled": pd.DataFrame([chosen]),
        "qubits": None,
        "selection_method": "qiskit_transpiler_default",
    }


# --------------------------------------------------------------------------
# Patch database run metadata so this ablation is unmistakable in SQL.
# --------------------------------------------------------------------------
_original_create_run = base.qpu_db.create_run


def _create_run_with_qiskit_layout_metadata(**kwargs):
    manifest = kwargs.get("manifest_json")
    if isinstance(manifest, dict):
        manifest = dict(manifest)
        manifest["physical_layout_policy"] = (
            "qiskit_transpiler_default_once_then_frozen"
        )
        manifest["custom_rule5_physical_selection"] = False
        manifest["qiskit_layout_optimization_level"] = int(base.OPT_LEVEL)
        manifest["qiskit_layout_seed_transpiler"] = int(base.SEED_TRANSPILE)
        kwargs["manifest_json"] = manifest

    kwargs["notes"] = (
        "Week 11.1F.3 Candidate #2 Qiskit-layout ablation. "
        "Logical Candidate #2 remains frozen by Rule4, but the physical "
        "layout is selected once by Qiskit's default transpiler placement and "
        "then frozen for all 2025 SamplerV2 circuits. Custom Rule-5 physical "
        "layout ranking is not used for selection. 2026 is not loaded."
    )

    return _original_create_run(**kwargs)


base.qpu_db.create_run = _create_run_with_qiskit_layout_metadata

# Replace ONLY the physical-layout selector.
base.selector.fresh_hardware_reselection = qiskit_layout_reselection

# Make the database script_name and dry-run command show this wrapper, while
# preserving base.HERE and all already-initialized project paths.
base.__file__ = str(Path(__file__).resolve())


def _old_path(suffix):
    return RESULTS / f"{OLD_PREFIX}{suffix}"


def _new_path(suffix):
    return RESULTS / f"{NEW_PREFIX}{suffix}"


def _copy_current_outputs_to_new_prefix(run_uuid):
    """
    Copy the files generated by the authoritative 11_1B engine to a distinct
    11_1F3 prefix, and also register aliases as DB artifacts.
    """
    for suffix in KNOWN_SUFFIXES:
        src = _old_path(suffix)
        dst = _new_path(suffix)

        if not src.exists():
            continue

        shutil.copy2(src, dst)

        # Add the clearly named alias to the same DB run.
        try:
            if dst.suffix.lower() == ".json":
                payload = json.loads(dst.read_text(encoding="utf-8"))
                base.qpu_db.save_json_artifact(
                    run_uuid,
                    dst.name,
                    payload,
                )
            elif dst.suffix.lower() == ".csv":
                df = pd.read_csv(dst)
                base.qpu_db.save_dataframe_artifact(
                    run_uuid,
                    dst.name,
                    df,
                )
        except Exception as exc:
            print(
                f"[database] WARNING: could not save alias artifact "
                f"{dst.name}: {exc}"
            )

    # Also store the dedicated Qiskit layout-selection record.
    if QISKIT_SELECTION_RECORD.exists():
        try:
            payload = json.loads(
                QISKIT_SELECTION_RECORD.read_text(encoding="utf-8")
            )
            base.qpu_db.save_json_artifact(
                run_uuid,
                QISKIT_SELECTION_RECORD.name,
                payload,
            )
        except Exception as exc:
            print(
                "[database] WARNING: could not save Qiskit layout artifact: "
                f"{exc}"
            )


def main():
    # Protect the existing authoritative 11_1B result files.  The imported
    # base engine uses those literal filenames internally.
    with tempfile.TemporaryDirectory(prefix="qrc_11_1f3_") as td:
        td = Path(td)
        backups = {}

        for suffix in KNOWN_SUFFIXES:
            src = _old_path(suffix)
            if src.exists():
                bak = td / src.name
                shutil.copy2(src, bak)
                backups[suffix] = bak

        try:
            base.main()

            run_uuid = base.CURRENT_RUN_UUID
            if run_uuid is not None:
                _copy_current_outputs_to_new_prefix(run_uuid)

        finally:
            # Restore the pre-existing 11_1B files so this ablation does not
            # destroy the original Rule-5 experiment artifacts.
            for suffix in KNOWN_SUFFIXES:
                src = _old_path(suffix)
                bak = backups.get(suffix)

                if bak is not None and bak.exists():
                    shutil.copy2(bak, src)
                elif src.exists():
                    # It did not exist before this run; keep only the 11_1F3
                    # alias, not a misleading new 11_1B file.
                    try:
                        src.unlink()
                    except Exception:
                        pass

    print()
    print("=" * 126)
    print("11.1F.3 CANDIDATE #2 QISKIT-LAYOUT ABLATION COMPLETE")
    print("=" * 126)
    print(f"Qiskit layout record: {QISKIT_SELECTION_RECORD}")
    print(f"Run artifacts copied with prefix: results/{NEW_PREFIX}_*")
    if base.CURRENT_RUN_UUID is not None:
        print(f"Database run UUID: {base.CURRENT_RUN_UUID}")
    print("Original 11_1B result files were preserved.")
    print("2026 remains FROZEN / UNUSED.")


if __name__ == "__main__":
    main()
