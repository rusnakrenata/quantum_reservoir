"""
WEEK 10.06A — H4 WEEK-10 HANDOFF AUDIT
Fresh Kingston layout + ideal r={2,3,4} audit + strict resource compilation.

Purpose
-------
Before spending time on the eight H4 backend-noisy simulations, reproduce the
same candidate-wide Week-10 setup used for H0-H3:

  * 8 unique H4 Rule-1–Rule-4 representatives from Week 9.05D;
  * ONE fresh current Kingston H4 layout shared by every H4 candidate;
  * direct-depth candidate universe r in {2,3,4};
  * ideal training-only Ridge re-selection at each r;
  * 2025 used only as diagnostic validation;
  * strict native zero-SWAP resource compilation;
  * 2026 never loaded;
  * NO QPU jobs and NO noisy simulation in this audit step.

Why this audit is separate
--------------------------
The historical Week-10 candidate-wide study evaluated all retained candidates
at r={2,3,4}.  H4 must enter the same pipeline.  This script first verifies
that every H4 candidate can be reconstructed exactly at its Week-9 selected r,
that the r={2,3,4} ideal branches are numerically well-defined, and that all
measurement circuits compile natively on one shared fresh Kingston H4 region.

Only after this audit passes do we run the expensive full-quantum and
finite-shot H4 noisy simulations.

Inputs
------
results/09_05d_h4_week10_additions.csv

Dependencies beside this script
--------------------------------
09_04_rwp_common.py
09_05c_rwp_resource_and_hardware_gapfill.py
11_0A_live_embedding_reselection.py
ibm_account.py
(and dependencies already required by those project files)

Outputs
-------
results/10_06a_h4_current_kingston_layout.csv
results/10_06a_h4_ideal_depth_audit.csv
results/10_06a_h4_resource_compile_detail.csv
results/10_06a_h4_resource_summary.csv
results/10_06a_h4_manifest.json

Typical run
-----------
python 10_06a_h4_week10_handoff_audit.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
RESOURCE_HELPER_FILE = HERE / "09_05c_rwp_resource_and_hardware_gapfill.py"

SELECTOR_CANDIDATES = [
    HERE / "11_0A_live_embedding_reselection.py",
    HERE / "11_0A_fresh_full_hardware_reselection.py",
]
SELECTOR_FILE = next((p for p in SELECTOR_CANDIDATES if p.exists()), None)

INPUT_FILE = RESULTS / "09_05d_h4_week10_additions.csv"

LAYOUT_FILE = RESULTS / "10_06a_h4_current_kingston_layout.csv"
IDEAL_FILE = RESULTS / "10_06a_h4_ideal_depth_audit.csv"
COMPILE_DETAIL_FILE = RESULTS / "10_06a_h4_resource_compile_detail.csv"
RESOURCE_FILE = RESULTS / "10_06a_h4_resource_summary.csv"
MANIFEST_FILE = RESULTS / "10_06a_h4_manifest.json"

BACKEND_NAME = "ibm_kingston"
R_VALUES = [2, 3, 4]
OPT_LEVEL = 1
SEED_TRANSPILE = 42
SHORTLIST = 120

# Three deterministic validation endpoints are enough for a gate/resource audit.
# Gate counts should not depend on the data values; medians protect against
# angle-dependent transpiler simplifications.
RESOURCE_VALIDATION_OFFSETS = [0, 182, 364]

AUDIT_ATOL = 2e-5


def load_module(path: Path, name: str):
    if path is None or not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


if SELECTOR_FILE is None:
    raise FileNotFoundError(
        "Missing fresh IBM selector. Expected one of: "
        + ", ".join(p.name for p in SELECTOR_CANDIDATES)
    )

common = load_module(COMMON_FILE, "qrc_1006a_common")
resource_helper = load_module(
    RESOURCE_HELPER_FILE,
    "qrc_1006a_resource_helper",
)
selector = load_module(SELECTOR_FILE, "qrc_1006a_selector")


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def bias(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(pred - y))


def parse_j(row: pd.Series) -> dict:
    raw = json.loads(str(row["J_json"]))
    J = {}
    for edge in common.TOPOLOGY_EDGES["H4"]:
        i, j = edge
        key = f"J{i}{j}"
        rev = f"J{j}{i}"
        if key in raw and raw[key] is not None:
            val = raw[key]
        elif rev in raw and raw[rev] is not None:
            val = raw[rev]
        else:
            raise KeyError(
                f"{row['candidate_key']}: missing H4 coupling {key}/{rev}"
            )
        J[edge] = float(val)
    return J


def candidate_from_manifest(row: pd.Series, r_override: int | None = None) -> dict:
    r = int(row["original_r"]) if r_override is None else int(r_override)
    window = (
        1
        if str(row["protocol"]).upper() == "CONT"
        else int(row["window"])
    )
    return {
        "candidate_id": str(row["candidate_key"]),
        "topology": "H4",
        "protocol": str(row["protocol"]).upper(),
        "window": int(window),
        "r": r,
        "alpha": float(row["alpha"]),
        "dt": float(row["dt"]),
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
        "J": parse_j(row),
    }


def build_cont_master(candidate: dict, angles: np.ndarray):
    """
    Exact ideal CONT trajectory: memory is initialized once at |00><00| and
    then carried chronologically through the complete 2022-2025 sequence.
    """
    A_list = common.build_channels(candidate, angles)
    rho_m = common.qrc.memory_zero_density()
    rows = []

    for A in A_list:
        rho_i, rho_m_out = common.qrc.final_reduced_states(A, rho_m)
        rows.append(common.reduced_feature_row(rho_i, rho_m_out))
        rho_m = rho_m_out

    endpoints = np.arange(len(A_list), dtype=int)
    return endpoints, pd.DataFrame(rows)


def ideal_branch(candidate: dict, readout: str):
    work_tv, cols, y_all = common.load_train_validation()
    angles = common.make_angles(work_tv, cols, float(candidate["alpha"]))

    if candidate["protocol"] == "CONT":
        endpoints, master = build_cont_master(candidate, angles)
    elif candidate["protocol"] == "RWP":
        A_list = common.build_channels(candidate, angles)
        endpoints, master = common.rwp_master_feature_bank_from_channels(
            A_list,
            int(candidate["window"]),
        )
    else:
        raise ValueError(candidate["protocol"])

    X = common.select_features(master, readout)
    y_ep = y_all[endpoints]

    train_mask = endpoints < common.N_TRAIN
    val_mask = endpoints >= common.N_TRAIN

    X_train = X[train_mask]
    y_train = y_ep[train_mask]
    X_val = X[val_mask]
    y_val = y_ep[val_mask]

    if len(y_val) != common.N_VAL:
        raise RuntimeError(
            f"{candidate['candidate_id']} r={candidate['r']}: "
            f"expected {common.N_VAL} validation rows; got {len(y_val)}"
        )

    lam, _, cv_summary = common.select_lambda_training_only(
        X_train,
        y_train,
        n_splits=5,
    )

    model, scaler, keep = common.fit_scaled_ridge(
        X_train,
        y_train,
        lam,
    )
    pred = common.predict_scaled_ridge(
        model,
        scaler,
        keep,
        X_val,
    )

    return {
        "selected_lambda": float(lam),
        "cv_rmse": float(cv_summary.iloc[0]["cv_rmse_mean"]),
        "cv_rmse_std": float(cv_summary.iloc[0]["cv_rmse_std"]),
        "ideal_2025_rmse": rmse(y_val, pred),
        "ideal_2025_mae": mae(y_val, pred),
        "ideal_2025_bias": bias(y_val, pred),
        "active_features": int(np.sum(keep)),
    }


def append_dynamics(qc, candidate, angles_row):
    # Injection encoding.
    for q in range(4):
        qc.ry(float(angles_row[q]), q)

    r = int(candidate["r"])
    dt = float(candidate["dt"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])

    for _ in range(r):
        for i, j in common.TOPOLOGY_EDGES["H4"]:
            qc.rzz(
                2.0 * float(candidate["J"][(i, j)]) * dt / r,
                i,
                j,
            )

        theta_x = 2.0 * hx * dt / r
        for q in range(6):
            qc.rx(theta_x, q)

        if not np.isclose(hy, 0.0):
            theta_y = 2.0 * hy * dt / r
            qc.ry(theta_y, 4)
            qc.ry(theta_y, 5)


def build_measurement_circuit(
    candidate: dict,
    all_angles: np.ndarray,
    endpoint: int,
    readout: str,
    setting,
):
    """
    Hardware-resource circuit only.

    RWP:
        replay W inputs, resetting q0-q3 between replayed inputs.

    CONT:
        one physical reservoir update per feature vector.  Persistent memory
        is a protocol/state-management issue, so the per-update resource
        circuit contains one injection encoding + one reservoir evolution.
        This is the same resource unit used for historical CONT comparisons.
    """
    label, basis, measured = setting
    qreg = QuantumRegister(6, "q")
    creg = ClassicalRegister(len(measured), "m")
    qc = QuantumCircuit(
        qreg,
        creg,
        name=f"{candidate['candidate_id']}_r{candidate['r']}_{label}",
    )

    if candidate["protocol"] == "RWP":
        W = int(candidate["window"])
        start = int(endpoint) - W + 1
        if start < 0:
            raise ValueError("RWP endpoint has insufficient history.")

        for k, idx in enumerate(range(start, int(endpoint) + 1)):
            if k > 0:
                for q in range(4):
                    qc.reset(q)
            append_dynamics(qc, candidate, all_angles[idx])

    elif candidate["protocol"] == "CONT":
        # One chronological update.  The memory state is carried by the CONT
        # protocol; the circuit cost reported here is the cost per update.
        append_dynamics(qc, candidate, all_angles[int(endpoint)])

    else:
        raise ValueError(candidate["protocol"])

    for q, axis in basis.items():
        if axis == "X":
            qc.h(q)
        elif axis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif axis == "Z":
            pass
        else:
            raise ValueError(axis)

    for c, q in enumerate(measured):
        qc.measure(q, c)

    return qc


def compile_metrics(circuit, backend):
    ops = {str(k): int(v) for k, v in circuit.count_ops().items()}
    duration_us = float(
        circuit.estimate_duration(backend.target, unit="s")
    ) * 1e6

    n_2q = 0
    for inst, qargs, _ in circuit.data:
        if len(qargs) == 2:
            n_2q += 1

    return {
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "n_cz": int(ops.get("cz", 0)),
        "n_swap": int(ops.get("swap", 0)),
        "n_reset": int(ops.get("reset", 0)),
        "n_measure": int(ops.get("measure", 0)),
        "n_2q": int(n_2q),
        "duration_us": duration_us,
        "operations_json": json.dumps(ops, sort_keys=True),
    }


def choose_shared_layout(service, reference_candidate, shortlist):
    """
    One topology-level fresh Kingston region shared by all H4 candidates.
    """
    # Keep selector globals synchronized where supported.
    if hasattr(selector, "ALPHA"):
        selector.ALPHA = float(reference_candidate["alpha"])
    if hasattr(selector, "DT"):
        selector.DT = float(reference_candidate["dt"])

    fresh = selector.fresh_hardware_reselection(
        service=service,
        candidate=reference_candidate,
        backend_names=[BACKEND_NAME],
        shortlist=int(shortlist),
        write_prefix="10_06a_h4",
        verbose=False,
    )

    chosen = fresh["selected"]
    layout = json.loads(chosen["layout"])

    row = {
        "topology": "H4",
        "backend": BACKEND_NAME,
        "layout": json.dumps(layout),
        "physical_C_t": layout[0],
        "physical_D": layout[1],
        "physical_P_t": layout[2],
        "physical_H": layout[3],
        "physical_M1": layout[4],
        "physical_M2": layout[5],
        "compiled_2q_error_max_percent": chosen.get(
            "compiled_2q_error_max_percent"
        ),
        "compiled_1q_error_max_percent": chosen.get(
            "compiled_1q_error_max_percent"
        ),
        "max_readout_error_percent": chosen.get(
            "max_readout_error_percent"
        ),
        "min_t1_us": chosen.get("min_t1_us"),
        "min_t2_us": chosen.get("min_t2_us"),
        "compiled_duration_us_reference": chosen.get("compiled_duration_us"),
        "pending_jobs": chosen.get("pending_jobs"),
        "selection_reference_candidate": reference_candidate["candidate_id"],
        "shared_across_all_H4_candidates": True,
    }
    return layout, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default=BACKEND_NAME,
        help="Kept for explicit provenance; Week-10 comparison uses Kingston.",
    )
    parser.add_argument(
        "--shortlist",
        type=int,
        default=SHORTLIST,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.backend != BACKEND_NAME:
        raise ValueError(
            "This audit intentionally matches the historical Week-10 "
            "Kingston candidate-wide experiment."
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} missing. Run Week 9.05D first."
        )

    df = pd.read_csv(INPUT_FILE)

    if len(df) != 8:
        raise RuntimeError(
            f"Expected exactly 8 unique H4 Week-10 additions; found {len(df)}."
        )
    if set(df["topology"].astype(str)) != {"H4"}:
        raise RuntimeError("Input must contain only H4 candidates.")
    if set(df["protocol"].astype(str).str.upper()) != {"CONT", "RWP"}:
        raise RuntimeError("Expected both CONT and RWP H4 candidates.")

    print("=" * 140)
    print("WEEK 10.06A — H4 WEEK-10 HANDOFF AUDIT")
    print("=" * 140)
    print("NO QPU JOBS ARE SUBMITTED.")
    print("NO NOISY SIMULATION IS RUN IN THIS AUDIT.")
    print("2026 remains frozen and is not loaded.")
    print(f"H4 candidates: {len(df)}")
    print(f"Depths audited: {R_VALUES}")
    print(f"Backend: {BACKEND_NAME}")
    print()

    # ------------------------------------------------------------------
    # 1. Ideal r={2,3,4} audit
    # ------------------------------------------------------------------
    ideal_rows = []

    for i, row in df.sort_values("candidate_key").iterrows():
        readout = str(row["test_readout"])
        original_r = int(row["original_r"])

        print(
            f"{row['candidate_key']}: {row['protocol']} "
            f"W={row['window']} readout={readout} original_r={original_r}"
        )

        for r in R_VALUES:
            cand = candidate_from_manifest(row, r_override=r)
            m = ideal_branch(cand, readout)

            is_original = int(r) == original_r
            cv_delta = (
                m["cv_rmse"] - float(row["source_readout_cv_rmse"])
                if is_original
                else np.nan
            )
            val_delta = (
                m["ideal_2025_rmse"] - float(row["source_validation_rmse"])
                if is_original
                else np.nan
            )

            if is_original:
                if abs(cv_delta) > AUDIT_ATOL:
                    raise RuntimeError(
                        f"{row['candidate_key']} original-r CV audit failed: "
                        f"recomputed={m['cv_rmse']:.9f}, "
                        f"manifest={float(row['source_readout_cv_rmse']):.9f}"
                    )
                if abs(val_delta) > AUDIT_ATOL:
                    raise RuntimeError(
                        f"{row['candidate_key']} original-r 2025 audit failed: "
                        f"recomputed={m['ideal_2025_rmse']:.9f}, "
                        f"manifest={float(row['source_validation_rmse']):.9f}"
                    )

            ideal_rows.append(
                {
                    "candidate_key": str(row["candidate_key"]),
                    "selected_by_rules": str(row["selected_by_rules"]),
                    "protocol": str(row["protocol"]).upper(),
                    "topology": "H4",
                    "window": (
                        np.nan
                        if str(row["protocol"]).upper() == "CONT"
                        else int(row["window"])
                    ),
                    "test_readout": readout,
                    "original_r": original_r,
                    "evaluated_r": int(r),
                    "is_original_r": bool(is_original),
                    "selected_lambda": m["selected_lambda"],
                    "cv_rmse": m["cv_rmse"],
                    "cv_rmse_std": m["cv_rmse_std"],
                    "ideal_2025_rmse": m["ideal_2025_rmse"],
                    "ideal_2025_mae": m["ideal_2025_mae"],
                    "ideal_2025_bias": m["ideal_2025_bias"],
                    "active_features": m["active_features"],
                    "cv_delta_vs_week9_manifest_at_original_r": cv_delta,
                    "validation_delta_vs_week9_manifest_at_original_r": val_delta,
                    "2025_used_for_selection": False,
                    "2026_used": False,
                }
            )

            print(
                f"  r={r}: lambda={m['selected_lambda']:.6g} | "
                f"CV={m['cv_rmse']:.6f}±{m['cv_rmse_std']:.6f} | "
                f"ideal2025={m['ideal_2025_rmse']:.6f}"
                + ("  [Week-9 audit PASS]" if is_original else "")
            )

    ideal_df = pd.DataFrame(ideal_rows)
    ideal_df.to_csv(IDEAL_FILE, index=False)

    # ------------------------------------------------------------------
    # 2. Fresh topology-level Kingston layout shared by all H4 candidates
    # ------------------------------------------------------------------
    from ibm_account import get_service

    service = get_service()

    # Use the resource-oriented H4 candidate as a neutral shallow reference
    # for the topology-level physical-region scan; the selected region is then
    # shared by all eight H4 candidates.
    ref_row = (
        df[df["candidate_key"].astype(str) == "CONT_H4_R4"]
        .iloc[0]
        if np.any(df["candidate_key"].astype(str) == "CONT_H4_R4")
        else df.sort_values("candidate_key").iloc[0]
    )
    ref_candidate = candidate_from_manifest(
        ref_row,
        r_override=int(ref_row["original_r"]),
    )

    layout, layout_row = choose_shared_layout(service, ref_candidate, args.shortlist)
    pd.DataFrame([layout_row]).to_csv(LAYOUT_FILE, index=False)

    print()
    print("FRESH SHARED H4 KINGSTON LAYOUT")
    print(f"  layout={layout}")
    print(
        f"  reference={layout_row['selection_reference_candidate']} "
        "(layout is shared across all H4 candidates)"
    )
    print()

    backend = service.backend(
        BACKEND_NAME,
        use_fractional_gates=False,
    )
    try:
        backend.refresh()
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3. Strict no-routing compile/resource audit at r={2,3,4}
    # ------------------------------------------------------------------
    work_tv, cols, _ = common.load_train_validation()

    compile_rows = []

    for _, row in df.sort_values("candidate_key").iterrows():
        protocol = str(row["protocol"]).upper()
        readout = str(row["test_readout"])
        alpha = float(row["alpha"])
        angles = common.make_angles(work_tv, cols, alpha)

        endpoints = [
            common.N_TRAIN + x
            for x in RESOURCE_VALIDATION_OFFSETS
        ]

        for r in R_VALUES:
            cand = candidate_from_manifest(row, r_override=r)

            for endpoint in endpoints:
                setting_metrics = []

                for setting in resource_helper.readout_settings(readout):
                    qc = build_measurement_circuit(
                        cand,
                        angles,
                        int(endpoint),
                        readout,
                        setting,
                    )

                    isa = transpile(
                        qc,
                        backend=backend,
                        initial_layout=layout,
                        routing_method="none",
                        optimization_level=OPT_LEVEL,
                        seed_transpiler=SEED_TRANSPILE,
                        scheduling_method="alap",
                    )
                    backend.check_faulty(isa)

                    m = compile_metrics(isa, backend)

                    if int(m["n_swap"]) != 0:
                        raise RuntimeError(
                            f"{row['candidate_key']} r={r}: "
                            "SWAP detected under strict native compilation."
                        )

                    detail = {
                        "candidate_key": str(row["candidate_key"]),
                        "selected_by_rules": str(row["selected_by_rules"]),
                        "protocol": protocol,
                        "topology": "H4",
                        "window": (
                            np.nan
                            if protocol == "CONT"
                            else int(row["window"])
                        ),
                        "readout": readout,
                        "r": int(r),
                        "endpoint": int(endpoint),
                        "setting": str(setting[0]),
                        "backend": BACKEND_NAME,
                        "layout": json.dumps(layout),
                        **m,
                    }
                    compile_rows.append(detail)
                    setting_metrics.append(m)

    detail_df = pd.DataFrame(compile_rows)
    detail_df.to_csv(COMPILE_DETAIL_FILE, index=False)

    # First sum settings to one feature vector per endpoint.
    per_endpoint_rows = []
    group_cols = [
        "candidate_key",
        "selected_by_rules",
        "protocol",
        "topology",
        "window",
        "readout",
        "r",
        "endpoint",
        "backend",
        "layout",
    ]

    for keys, g in detail_df.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, keys))
        per_endpoint_rows.append(
            {
                **meta,
                "n_settings": int(g["setting"].nunique()),
                "N2q_feature_vector": int(g["n_2q"].sum()),
                "NCZ_feature_vector": int(g["n_cz"].sum()),
                "NSWAP_feature_vector": int(g["n_swap"].sum()),
                "depth_max_setting": int(g["depth"].max()),
                "depth_sum_settings": int(g["depth"].sum()),
                "Tcircuit_feature_vector_us": float(g["duration_us"].sum()),
                "Tcircuit_max_setting_us": float(g["duration_us"].max()),
                "resets_feature_vector": int(g["n_reset"].sum()),
            }
        )

    per_endpoint = pd.DataFrame(per_endpoint_rows)

    summary = (
        per_endpoint.groupby(
            [
                "candidate_key",
                "selected_by_rules",
                "protocol",
                "topology",
                "window",
                "readout",
                "r",
                "backend",
                "layout",
            ],
            dropna=False,
        )
        .agg(
            sampled_endpoints=("endpoint", "nunique"),
            n_settings=("n_settings", "median"),
            N2q_median=("N2q_feature_vector", "median"),
            NCZ_median=("NCZ_feature_vector", "median"),
            NSWAP_median=("NSWAP_feature_vector", "median"),
            depth_max_setting_median=("depth_max_setting", "median"),
            depth_sum_settings_median=("depth_sum_settings", "median"),
            Tcircuit_feature_vector_us_median=(
                "Tcircuit_feature_vector_us",
                "median",
            ),
            Tcircuit_max_setting_us_median=(
                "Tcircuit_max_setting_us",
                "median",
            ),
            resets_feature_vector_median=(
                "resets_feature_vector",
                "median",
            ),
        )
        .reset_index()
    )

    summary.to_csv(RESOURCE_FILE, index=False)

    print("=" * 140)
    print("H4 RESOURCE AUDIT SUMMARY")
    print("=" * 140)
    print(
        summary[
            [
                "candidate_key",
                "protocol",
                "window",
                "readout",
                "r",
                "NCZ_median",
                "NSWAP_median",
                "depth_max_setting_median",
                "Tcircuit_feature_vector_us_median",
            ]
        ].to_string(index=False)
    )

    manifest = {
        "step": "10.06A_H4_week10_handoff_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_candidates": str(INPUT_FILE),
        "candidate_count": int(len(df)),
        "topology": "H4",
        "backend": BACKEND_NAME,
        "shared_layout": layout,
        "layout_selection_reference": str(
            layout_row["selection_reference_candidate"]
        ),
        "r_values": R_VALUES,
        "original_r_reproduction_tolerance": AUDIT_ATOL,
        "original_r_rows_audited": int(
            ideal_df["is_original_r"].sum()
        ),
        "ideal_rows": int(len(ideal_df)),
        "compile_summary_rows": int(len(summary)),
        "all_zero_swap": bool(
            (pd.to_numeric(summary["NSWAP_median"]) == 0).all()
        ),
        "selection_period": "2022-2024",
        "validation_2025": "diagnostic only",
        "test_2026_used": False,
        "qpu_jobs_submitted": 0,
        "noisy_simulation_run": False,
        "next_step": (
            "Run H4-only Week-10 full-quantum direct and finite-shot "
            "backend-calibrated simulation using this frozen shared layout "
            "and the historical Week-10 operating cells."
        ),
    }

    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 140)
    print("10.06A COMPLETE")
    print("=" * 140)
    print(f"Ideal original-r audits passed: {manifest['original_r_rows_audited']}/8")
    print(f"All strict compiles zero-SWAP: {manifest['all_zero_swap']}")
    print("No QPU jobs submitted. No noisy simulation run. 2026 untouched.")
    print()
    print("Saved:")
    for p in [
        LAYOUT_FILE,
        IDEAL_FILE,
        COMPILE_DETAIL_FILE,
        RESOURCE_FILE,
        MANIFEST_FILE,
    ]:
        print(f"  {p}")
    print()
    print(
        "STOP HERE. Send the console output and the five files above. "
        "Then run the expensive H4 noisy simulation only after this audit "
        "has been interpreted."
    )


if __name__ == "__main__":
    main()
