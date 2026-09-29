"""
Week 12.4B — IQM multi-rule physical compilation and Rule-5 layout closure.

Direct IQM analogue of the IBM 09.05C physical-resource / hardware-embedding
closure, adapted to the completed H5/H6 RWP pipeline.

Logical roles carried forward per topology
-------------------------------------------
R1  predictive accuracy: minimum 2022-2024 chronological CV RMSE.
R2  intrinsic temporal memory: winner from Week 12.4A.
R3  finite-shot/readout robustness: same 1024-shot hierarchy used in IBM and
    IQM 12.3.
R4  resource efficiency: least logical hardware cost inside the one-standard-
    error predictive set.

Exact duplicate candidates are merged only if the SAME
(topology, W, r, dynamics, readout) serves multiple roles.

Rule 5 physical layer
---------------------
For every retained logical candidate on Emerald and Garnet:
- keep several non-weighted Pareto anchor layouts from 12.2A,
- compile actual RWP measurement circuits against backend.get_real_target(),
- use routing_method='none',
- require zero SWAP and no unexpected physical two-qubit edge,
- record actual depth, CZ count, reset count, duration and calibration metrics,
- keep a physical-layout Pareto set,
- keep up to three transparent hierarchical Rule-5 layouts per
  candidate/backend; never collapse to one weighted hardware score.

No QPU job is submitted. No shots are executed. 2026 is not loaded.

Calibration freshness
---------------------
Rule 5 is calibration-sensitive. Before the final physical shortlist, refresh:

    python 12_01_iqm_discovery.py
    python 12_02_iqm_region_search.py

Then run this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from iqm.qiskit_iqm.iqm_provider import IQMProvider

from iqm_account import get_authentication


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "12_03d_iqm_rwp_common.py"
STRICT_HELPER_FILE = HERE / "12_02B_iqm_strict_compile_audit.py"

TRIAL_FILE = RESULTS / "12_03g_rwp_trotter_all_trials.csv"
MEMORY_WINNER_FILE = RESULTS / "12_04a_iqm_rwp_memory_rule2_winners.csv"
DISCOVERY_JSON = RESULTS / "12_01_iqm_discovery.json"
REGION_JSON = RESULTS / "12_02_iqm_region_search.json"
PARETO_CSV = RESULTS / "12_02_iqm_pareto_embeddings.csv"

OUT_RULES = RESULTS / "12_04b_iqm_rule1_rule4_candidates.csv"
OUT_DETAIL = RESULTS / "12_04b_iqm_compile_detail.csv"
OUT_ENDPOINT = RESULTS / "12_04b_iqm_compile_per_endpoint.csv"
OUT_SUMMARY = RESULTS / "12_04b_iqm_compile_summary.csv"
OUT_RULE5_PARETO = RESULTS / "12_04b_iqm_rule5_layout_pareto.csv"
OUT_RULE5_TOP3 = RESULTS / "12_04b_iqm_rule5_top3_layouts.csv"
OUT_MANIFEST = RESULTS / "12_04b_iqm_compile_manifest.json"

BACKENDS = ("emerald", "garnet")
EXPECTED_TOPOLOGIES = ("H5", "H6")
N_FOLDS = 5
SEED_TRANSPILE = 79001
OPTIMIZATION_LEVEL = 1
DEFAULT_SAMPLED_ENDPOINTS = 5
TOP_K_LAYOUTS = 3
EXPECTED_FULL_TRIAL_ROWS = 56 * 3 * 13 * 5


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_1204b_common")
strict_helper = load_module(STRICT_HELPER_FILE, "qrc_1204b_strict")


def ffloat(v, default=np.nan):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def normalize_edge(a: Any, b: Any):
    return tuple(sorted((str(a), str(b)), key=str))


def operation_counts(qc):
    return dict(sorted(Counter(inst.operation.name for inst in qc.data).items()))


def candidate_from_trial_row(row):
    if isinstance(row, pd.Series):
        row = row.to_dict()
    H = str(row["topology"])
    if H not in EXPECTED_TOPOLOGIES:
        raise ValueError(H)
    J = {}
    for edge, col in common.EDGE_TO_COLUMN[H].items():
        if col not in row or pd.isna(row[col]):
            raise KeyError(f"{H}: missing canonical coupling {col} for {edge}")
        J[edge] = float(row[col])
    r = int(row["r"])
    if "trotter_r_tested" in row and not pd.isna(row["trotter_r_tested"]):
        if int(row["trotter_r_tested"]) != r:
            raise RuntimeError("r serialization audit failed")
    return {
        "topology": H,
        "alpha": float(row["alpha"]),
        "dt": float(row["dt"]),
        "r": r,
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
        "J": J,
        "window": int(row["window"]),
        "readout": str(row["readout"]),
        "trotter_search_id": str(row["trotter_search_id"]),
    }


# -----------------------------------------------------------------------------
# Rules 1-4
# -----------------------------------------------------------------------------

def select_rule1(group):
    return group.sort_values(["cv_rmse", "cv_rmse_std"]).iloc[0].copy()


def select_rule2(H, trials, memory_winners):
    mw = memory_winners[memory_winners["topology"].astype(str) == H]
    if len(mw) != 1:
        raise RuntimeError(f"Expected one Rule-2 memory winner for {H}, got {len(mw)}")
    m = mw.iloc[0]
    W = int(m["window"])
    r = int(m["r"])
    sid = str(m["trotter_search_id"])
    role = str(m.get("pool_role", ""))
    dyn = trials[
        (trials["topology"].astype(str) == H)
        & (trials["window"].astype(int) == W)
        & (trials["r"].astype(int) == r)
        & (trials["trotter_search_id"].astype(str) == sid)
    ].copy()
    if len(dyn) == 0:
        raise RuntimeError(f"Cannot reconnect Rule-2 {H}/W{W}/r{r}/{sid}")
    if "HW_AWARE_WINNER" in role:
        chosen, _ = common.select_hardware_aware_row(dyn)
        chosen = chosen.copy()
    else:
        chosen = dyn.sort_values(["cv_rmse", "cv_rmse_std"]).iloc[0].copy()
    chosen["MC_ch_mean_rule2"] = float(m["MC_ch_mean"])
    chosen["MC_ch_std_rule2"] = float(m["MC_ch_std"])
    return chosen


def select_rule3(group):
    chosen, mode = common.select_hardware_aware_row(group)
    chosen = chosen.copy()
    chosen["rule3_selection_mode"] = mode
    return chosen


def select_rule4(group):
    best = select_rule1(group)
    threshold = float(best["cv_rmse"]) + float(best["cv_rmse_std"]) / math.sqrt(N_FOLDS)
    eligible = group[group["cv_rmse"].astype(float) <= threshold].copy()
    if len(eligible) == 0:
        raise RuntimeError("Rule-4 1SE set is empty")
    chosen = eligible.sort_values(
        [
            "feature_vector_cz",
            "n_settings",
            "logical_reset_count",
            "n_features",
            "cv_rmse",
            "shot_noise_forecast_sd_proxy_1024",
            "max_shot_to_train_std_ratio_1024",
        ]
    ).iloc[0].copy()
    chosen["one_se_threshold"] = threshold
    chosen["one_se_set_size"] = len(eligible)
    return chosen


def build_rule_union(trials, memory_winners):
    rows = []
    for H in EXPECTED_TOPOLOGIES:
        g = trials[trials["topology"].astype(str) == H].copy()
        selections = {
            "R1_accuracy": select_rule1(g),
            "R2_memory": select_rule2(H, trials, memory_winners),
            "R3_readout": select_rule3(g),
            "R4_resources": select_rule4(g),
        }
        for rule, row in selections.items():
            d = row.to_dict()
            d["selected_rule"] = rule
            rows.append(d)

    raw = pd.DataFrame(rows)
    merged = []
    key_cols = ["topology", "window", "r", "trotter_search_id", "readout"]
    for _, g in raw.groupby(key_cols, dropna=False, sort=True):
        rep = g.iloc[0].copy()
        rep["selected_rules"] = "+".join(sorted(set(g["selected_rule"].astype(str))))
        rep["n_roles"] = int(g["selected_rule"].nunique())
        for c in ["MC_ch_mean_rule2", "MC_ch_std_rule2"]:
            if c in g.columns:
                vals = pd.to_numeric(g[c], errors="coerce").dropna()
                if len(vals):
                    rep[c] = float(vals.iloc[0])
        merged.append(rep)

    out = pd.DataFrame(merged).reset_index(drop=True)
    out["candidate_uid"] = [
        f"{r.topology}_W{int(r.window):02d}_r{int(r.r)}_{r.trotter_search_id}_{r.readout}"
        for r in out.itertuples()
    ]
    return out.sort_values(["topology", "cv_rmse", "window", "r"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Actual RWP circuits
# -----------------------------------------------------------------------------

def readout_settings(readout):
    if readout == "XZ_injection":
        return [
            ("XXXX", {0:"X",1:"X",2:"X",3:"X"}, [0,1,2,3]),
            ("ZZZZ", {0:"Z",1:"Z",2:"Z",3:"Z"}, [0,1,2,3]),
        ]
    if readout == "XZinj_dropX3":
        return [
            ("XXXZ", {0:"X",1:"X",2:"X",3:"Z"}, [0,1,2,3]),
            ("ZZZZ", {0:"Z",1:"Z",2:"Z",3:"Z"}, [0,1,2,3]),
        ]
    if readout == "XZinj_dropX3_plus_YX45":
        return [
            ("XXXZYX", {0:"X",1:"X",2:"X",3:"Z",4:"Y",5:"X"}, list(range(6))),
            ("ZZZZZZ", {q:"Z" for q in range(6)}, list(range(6))),
        ]
    if readout == "XZinj_plus_YX45":
        return [
            ("XXXXYX", {0:"X",1:"X",2:"X",3:"X",4:"Y",5:"X"}, list(range(6))),
            ("ZZZZZZ", {q:"Z" for q in range(6)}, list(range(6))),
        ]
    if readout == "XYZ_all":
        return [
            ("XXXXXX", {q:"X" for q in range(6)}, list(range(6))),
            ("YYYYYY", {q:"Y" for q in range(6)}, list(range(6))),
            ("ZZZZZZ", {q:"Z" for q in range(6)}, list(range(6))),
        ]
    raise ValueError(readout)


def append_rwp_step(qc, candidate, angle_row, first_step):
    if not first_step:
        for q in range(4):
            qc.reset(q)
    for q in range(4):
        qc.ry(float(angle_row[q]), q)
    r = int(candidate["r"])
    dt = float(candidate["dt"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])
    J = candidate["J"]
    for _ in range(r):
        for i, j in common.TOPOLOGY_EDGES[candidate["topology"]]:
            qc.rzz(2.0 * float(J[(i, j)]) * dt / r, int(i), int(j))
        theta_x = 2.0 * hx * dt / r
        for q in range(6):
            qc.rx(theta_x, q)
        if not np.isclose(hy, 0.0):
            theta_y = 2.0 * hy * dt / r
            qc.ry(theta_y, 4)
            qc.ry(theta_y, 5)


def build_measurement_circuit(candidate, all_angles, endpoint, setting):
    label, basis, measured = setting
    W = int(candidate["window"])
    qreg = QuantumRegister(6, "q")
    creg = ClassicalRegister(len(measured), "m")
    qc = QuantumCircuit(qreg, creg, name=f"{candidate['topology']}_W{W}_{endpoint}_{label}")
    start = int(endpoint) - W + 1
    if start < 0:
        raise ValueError("Endpoint lacks rewind history")
    for k, idx in enumerate(range(start, int(endpoint) + 1)):
        append_rwp_step(qc, candidate, all_angles[idx], first_step=(k == 0))
    for q, axis in basis.items():
        if axis == "X":
            qc.h(q)
        elif axis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif axis != "Z":
            raise ValueError(axis)
    for c, q in enumerate(measured):
        qc.measure(q, c)
    return qc


# -----------------------------------------------------------------------------
# Strict physical compilation
# -----------------------------------------------------------------------------

def duration_us(circuit, target):
    try:
        sec = float(circuit.estimate_duration(target, unit="s"))
        if math.isfinite(sec):
            return sec * 1e6
    except Exception:
        pass
    return np.nan


def compile_one(logical, backend, real_target, initial_layout, expected_edges):
    tqc = transpile(
        logical,
        target=real_target,
        initial_layout=initial_layout,
        routing_method="none",
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
    )
    counts = operation_counts(tqc)
    loci, mapping_mode = strict_helper.physical_two_qubit_edges_from_transpiled(
        tqc, backend, initial_layout
    )
    all_2q = {x["pair"] for x in loci}
    cz_edges = {x["pair"] for x in loci if x["operation"] == "cz"}
    unexpected_2q = all_2q - expected_edges
    unexpected_cz = cz_edges - expected_edges
    swap_count = int(counts.get("swap", 0))
    perm, identity = strict_helper.routing_permutation_info(tqc)
    allowed = set(str(x) for x in real_target.operation_names) | {"barrier"}
    non_native = sorted(set(counts) - allowed)
    strict_pass = (
        swap_count == 0
        and identity is True
        and not unexpected_2q
        and not unexpected_cz
        and not non_native
    )
    return {
        "strict_pass": bool(strict_pass),
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cz_count": int(counts.get("cz", 0)),
        "swap_count": swap_count,
        "reset_count": int(counts.get("reset", 0)),
        "measure_count": int(counts.get("measure", 0)),
        "duration_us": duration_us(tqc, real_target),
        "operations_json": json.dumps(counts, sort_keys=True),
        "mapping_mode": mapping_mode,
        "routing_permutation_json": json.dumps(perm),
        "routing_identity": identity,
        "unexpected_two_q_edges_json": json.dumps(sorted([list(x) for x in unexpected_2q])),
        "unexpected_cz_edges_json": json.dumps(sorted([list(x) for x in unexpected_cz])),
        "non_native_ops_json": json.dumps(non_native),
    }


# -----------------------------------------------------------------------------
# Rule-5 physical Pareto and top-3 hierarchy
# -----------------------------------------------------------------------------

MIN_OBJECTIVES = ["Tcircuit_max_setting_us_median", "depth_max_setting_median"]
MAX_OBJECTIVES = [
    "memory_min_t1_us",
    "memory_min_t2_echo_us",
    "min_readout_fidelity",
    "min_cz_fidelity",
]


def dominates(a, b):
    comps = []
    for field in MIN_OBJECTIVES:
        av, bv = ffloat(a.get(field)), ffloat(b.get(field))
        if np.isfinite(av) and np.isfinite(bv):
            comps.append((av <= bv, av < bv))
    for field in MAX_OBJECTIVES:
        av, bv = ffloat(a.get(field)), ffloat(b.get(field))
        if np.isfinite(av) and np.isfinite(bv):
            comps.append((av >= bv, av > bv))
    return bool(comps) and all(x[0] for x in comps) and any(x[1] for x in comps)


def physical_pareto(rows):
    frontier = []
    for row in rows:
        if any(dominates(other, row) for other in frontier):
            continue
        frontier = [other for other in frontier if not dominates(row, other)]
        frontier.append(row)
    return frontier


def top_k_layouts(group, k=TOP_K_LAYOUTS):
    g = group[group["strict_pass"].astype(bool)].copy()
    if len(g) == 0:
        return g
    p = pd.DataFrame(physical_pareto(g.to_dict("records")))
    if len(p) == 0:
        p = g
    p = p.sort_values(
        [
            "min_cz_fidelity",
            "min_t2_echo_us",
            "memory_min_t2_echo_us",
            "min_readout_fidelity",
            "Tcircuit_max_setting_us_median",
            "depth_max_setting_median",
        ],
        ascending=[False, False, False, False, True, True],
    ).head(int(k)).copy()
    p["rule5_rank"] = np.arange(1, len(p) + 1)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLED_ENDPOINTS)
    parser.add_argument("--backend", choices=["all", *BACKENDS], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be >= 1")

    required = [
        TRIAL_FILE,
        MEMORY_WINNER_FILE,
        DISCOVERY_JSON,
        REGION_JSON,
        PARETO_CSV,
        COMMON_FILE,
        STRICT_HELPER_FILE,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    outputs = [
        OUT_RULES, OUT_DETAIL, OUT_ENDPOINT, OUT_SUMMARY,
        OUT_RULE5_PARETO, OUT_RULE5_TOP3, OUT_MANIFEST,
    ]
    if args.overwrite:
        for p in outputs:
            if p.exists():
                p.unlink()

    trials = pd.read_csv(TRIAL_FILE)
    if len(trials) != EXPECTED_FULL_TRIAL_ROWS:
        raise RuntimeError(
            f"12.3G row audit failed: got {len(trials)}, expected {EXPECTED_FULL_TRIAL_ROWS}"
        )
    memory_winners = pd.read_csv(MEMORY_WINNER_FILE)
    candidates = build_rule_union(trials, memory_winners)
    candidates.to_csv(OUT_RULES, index=False)

    with open(DISCOVERY_JSON, "r", encoding="utf-8") as f:
        discovery = json.load(f)
    # Presence is audited even though topology edges come from the frozen common module.
    with open(REGION_JSON, "r", encoding="utf-8") as f:
        _region = json.load(f)

    pareto_rows = strict_helper.read_csv(PARETO_CSV)
    anchor_map = strict_helper.select_anchor_rows(pareto_rows)
    token, server_url, _ = get_authentication()
    backend_names = list(BACKENDS) if args.backend == "all" else [args.backend]

    work_tv, cols, _ = common.load_train_validation()  # 2022-2025 only
    angle_cache = {}

    print("=" * 126)
    print("WEEK 12.4B — IQM MULTI-RULE PHYSICAL COMPILATION + RULE-5 LAYOUT CLOSURE")
    print("=" * 126)
    print(f"Unique logical candidates after role deduplication: {len(candidates)}")
    print(f"Backends: {backend_names}")
    print(f"Sampled 2025 input endpoints per candidate/layout: {args.samples}")
    print(f"Optimization level: {OPTIMIZATION_LEVEL}")
    print('Routing method: "none"')
    print("Compile target: backend.get_real_target()")
    print("QPU jobs: NO | shots: 0 | 2026: untouched")
    print()
    print(candidates[[
        "candidate_uid", "selected_rules", "topology", "window", "r",
        "trotter_search_id", "readout", "cv_rmse", "cv_rmse_std",
        "max_shot_to_train_std_ratio_1024", "feature_vector_cz",
    ]].to_string(index=False))

    detail_rows = []
    manifest = {
        "stage": "12.4B",
        "n_unique_logical_candidates": int(len(candidates)),
        "settings": {
            "backends": backend_names,
            "samples": int(args.samples),
            "optimization_level": OPTIMIZATION_LEVEL,
            "seed_transpiler": SEED_TRANSPILE,
            "routing_method": "none",
            "compile_target": "backend.get_real_target()",
            "top_k_layouts": TOP_K_LAYOUTS,
            "qpu_jobs": False,
            "shots": 0,
            "uses_2025_target_for_selection": False,
            "loads_2026": False,
        },
        "calibration_sets": {},
    }

    for backend_name in backend_names:
        system = discovery["systems"][backend_name]
        calibration_set_id = system["summary"]["calibration_set_id"]
        manifest["calibration_sets"][backend_name] = calibration_set_id

        provider = IQMProvider(
            server_url,
            quantum_computer=backend_name,
            token=token,
        )
        backend = provider.get_backend(
            calibration_set_id=UUID(calibration_set_id),
            use_metrics=False,
        )
        real_target = backend.get_real_target()
        phys_to_idx = strict_helper.qiskit_physical_index_map(backend)

        print("\n" + "=" * 126)
        print(f"BACKEND {backend_name.upper()} | calibration_set_id={calibration_set_id}")
        print("=" * 126)

        for i, row in candidates.iterrows():
            cand = candidate_from_trial_row(row)
            H = cand["topology"]
            uid = str(row["candidate_uid"])
            anchors = anchor_map.get((backend_name, H), [])
            if not anchors:
                raise RuntimeError(f"No Pareto anchors for {backend_name}/{H}")

            alpha = float(cand["alpha"])
            if alpha not in angle_cache:
                angle_cache[alpha] = common.make_angles(work_tv, cols, alpha)
            angles = angle_cache[alpha]

            val_endpoints = np.arange(common.N_TRAIN, common.N_TRAIN + common.N_VAL)
            idx = np.linspace(0, len(val_endpoints) - 1, min(args.samples, len(val_endpoints)), dtype=int)
            sampled = val_endpoints[idx]

            print(
                f"[{i+1:02d}/{len(candidates)}] {uid} | roles={row['selected_rules']} | anchors={len(anchors)}"
            )

            for ai, anchor in enumerate(anchors, start=1):
                arow = anchor["row"]
                reasons = anchor["reasons"]
                phys_names = strict_helper.layout_from_row(arow)
                missing = [q for q in phys_names if q not in phys_to_idx]
                if missing:
                    raise RuntimeError(f"{backend_name}/{H}: missing physical names {missing}")
                initial_layout = [phys_to_idx[q] for q in phys_names]
                expected_edges = {
                    normalize_edge(phys_names[int(a)], phys_names[int(b)])
                    for a, b in common.TOPOLOGY_EDGES[H]
                }
                anchor_id = str(arow.get("embedding_id", f"{backend_name}_{H}_A{ai}"))

                for endpoint in sampled:
                    for setting in readout_settings(cand["readout"]):
                        logical = build_measurement_circuit(cand, angles, int(endpoint), setting)
                        meta = {
                            "candidate_uid": uid,
                            "selected_rules": str(row["selected_rules"]),
                            "topology": H,
                            "window": int(cand["window"]),
                            "r": int(cand["r"]),
                            "trotter_search_id": str(cand["trotter_search_id"]),
                            "readout": str(cand["readout"]),
                            "cv_rmse": float(row["cv_rmse"]),
                            "cv_rmse_std": float(row["cv_rmse_std"]),
                            "validation_rmse_diagnostic_only": float(row["validation_rmse"]),
                            "backend": backend_name,
                            "calibration_set_id": calibration_set_id,
                            "anchor_id": anchor_id,
                            "anchor_reasons": ";".join(reasons),
                            "layout": str(arow["layout"]),
                            "q0_C": arow["q0_C"],
                            "q1_D": arow["q1_D"],
                            "q2_P": arow["q2_P"],
                            "q3_H": arow["q3_H"],
                            "q4_M1": arow["q4_M1"],
                            "q5_M2": arow["q5_M2"],
                            "endpoint": int(endpoint),
                            "setting": str(setting[0]),
                            "min_t1_us": ffloat(arow.get("min_t1_us")),
                            "min_t2_echo_us": ffloat(arow.get("min_t2_echo_us")),
                            "memory_min_t1_us": ffloat(arow.get("memory_min_t1_us")),
                            "memory_min_t2_echo_us": ffloat(arow.get("memory_min_t2_echo_us")),
                            "min_readout_fidelity": ffloat(arow.get("min_readout_fidelity")),
                            "min_cz_fidelity": ffloat(arow.get("min_cz_fidelity")),
                            "worst_readout_error": ffloat(arow.get("worst_readout_error")),
                            "worst_cz_error": ffloat(arow.get("worst_cz_error")),
                            "2025_used_for_selection": False,
                            "qpu_job_submitted": False,
                        }
                        try:
                            result = compile_one(
                                logical, backend, real_target, initial_layout, expected_edges
                            )
                            detail_rows.append({**meta, "compile_success": True, **result})
                        except Exception as exc:
                            detail_rows.append({
                                **meta,
                                "compile_success": False,
                                "strict_pass": False,
                                "compile_error": f"{type(exc).__name__}: {exc}",
                                "depth": np.nan,
                                "size": np.nan,
                                "cz_count": np.nan,
                                "swap_count": np.nan,
                                "reset_count": np.nan,
                                "measure_count": np.nan,
                                "duration_us": np.nan,
                            })
                print(
                    f"    anchor {ai}/{len(anchors)} {anchor_id} reasons={';'.join(reasons)} layout={arow['layout']}"
                )

    detail = pd.DataFrame(detail_rows)
    if len(detail) == 0:
        raise RuntimeError("No compile rows produced")
    detail.to_csv(OUT_DETAIL, index=False)

    # Per endpoint: sum separate measurement settings for one feature vector.
    endpoint_group = [
        "candidate_uid", "selected_rules", "topology", "window", "r",
        "trotter_search_id", "readout", "cv_rmse", "cv_rmse_std",
        "validation_rmse_diagnostic_only", "backend", "calibration_set_id",
        "anchor_id", "anchor_reasons", "layout", "q0_C", "q1_D", "q2_P",
        "q3_H", "q4_M1", "q5_M2", "min_t1_us", "min_t2_echo_us",
        "memory_min_t1_us", "memory_min_t2_echo_us", "min_readout_fidelity",
        "min_cz_fidelity", "worst_readout_error", "worst_cz_error", "endpoint",
    ]
    endpoint_rows = []
    for key, g in detail.groupby(endpoint_group, dropna=False, sort=False):
        d = dict(zip(endpoint_group, key))
        durations = pd.to_numeric(g["duration_us"], errors="coerce")
        max_dur = float(durations.max()) if durations.notna().any() else np.nan
        sum_dur = float(durations.sum()) if durations.notna().all() else np.nan
        mt1 = ffloat(d["memory_min_t1_us"])
        mt2 = ffloat(d["memory_min_t2_echo_us"])
        endpoint_rows.append({
            **d,
            "n_settings": int(g["setting"].nunique()),
            "compile_success_all_settings": bool(g["compile_success"].astype(bool).all()),
            "strict_pass_all_settings": bool(g["strict_pass"].fillna(False).astype(bool).all()),
            "NCZ_feature_vector": float(pd.to_numeric(g["cz_count"], errors="coerce").sum()),
            "NSWAP_feature_vector": float(pd.to_numeric(g["swap_count"], errors="coerce").sum()),
            "reset_feature_vector": float(pd.to_numeric(g["reset_count"], errors="coerce").sum()),
            "depth_max_setting": float(pd.to_numeric(g["depth"], errors="coerce").max()),
            "depth_sum_settings": float(pd.to_numeric(g["depth"], errors="coerce").sum()),
            "Tcircuit_max_setting_us": max_dur,
            "Tcircuit_feature_vector_us": sum_dur,
            "memory_duration_over_t1": max_dur / mt1 if np.isfinite(max_dur) and np.isfinite(mt1) and mt1 > 0 else np.nan,
            "memory_duration_over_t2_echo": max_dur / mt2 if np.isfinite(max_dur) and np.isfinite(mt2) and mt2 > 0 else np.nan,
        })
    endpoint_df = pd.DataFrame(endpoint_rows)
    endpoint_df.to_csv(OUT_ENDPOINT, index=False)

    # Layout summary across sampled endpoints.
    summary_group = [c for c in endpoint_group if c != "endpoint"]
    summary_rows = []
    for key, g in endpoint_df.groupby(summary_group, dropna=False, sort=False):
        d = dict(zip(summary_group, key))
        summary_rows.append({
            **d,
            "sampled_endpoints": int(g["endpoint"].nunique()),
            "compile_success": bool(g["compile_success_all_settings"].astype(bool).all()),
            "strict_pass": bool(g["strict_pass_all_settings"].astype(bool).all()),
            "n_settings": float(g["n_settings"].median()),
            "NCZ_median": float(g["NCZ_feature_vector"].median()),
            "NSWAP_median": float(g["NSWAP_feature_vector"].median()),
            "reset_median": float(g["reset_feature_vector"].median()),
            "depth_max_setting_median": float(g["depth_max_setting"].median()),
            "depth_sum_settings_median": float(g["depth_sum_settings"].median()),
            "Tcircuit_max_setting_us_median": float(g["Tcircuit_max_setting_us"].median()) if g["Tcircuit_max_setting_us"].notna().any() else np.nan,
            "Tcircuit_feature_vector_us_median": float(g["Tcircuit_feature_vector_us"].median()) if g["Tcircuit_feature_vector_us"].notna().any() else np.nan,
            "memory_duration_over_t1_max": float(g["memory_duration_over_t1"].max()) if g["memory_duration_over_t1"].notna().any() else np.nan,
            "memory_duration_over_t2_echo_max": float(g["memory_duration_over_t2_echo"].max()) if g["memory_duration_over_t2_echo"].notna().any() else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    pareto_parts = []
    top3_parts = []
    for (uid, backend_name), g in summary.groupby(["candidate_uid", "backend"], sort=True):
        valid = g[g["strict_pass"].astype(bool)].copy()
        if len(valid) == 0:
            continue
        p = pd.DataFrame(physical_pareto(valid.to_dict("records")))
        if len(p):
            p["rule5_pareto"] = True
            pareto_parts.append(p)
        top = top_k_layouts(valid, TOP_K_LAYOUTS)
        if len(top):
            top3_parts.append(top)

    pareto_df = pd.concat(pareto_parts, ignore_index=True) if pareto_parts else pd.DataFrame()
    top3_df = pd.concat(top3_parts, ignore_index=True) if top3_parts else pd.DataFrame()
    pareto_df.to_csv(OUT_RULE5_PARETO, index=False)
    top3_df.to_csv(OUT_RULE5_TOP3, index=False)

    failed = summary[~summary["strict_pass"].astype(bool)]
    manifest["n_compile_detail_rows"] = int(len(detail))
    manifest["n_layout_summaries"] = int(len(summary))
    manifest["n_strict_failed_layout_summaries"] = int(len(failed))
    manifest["n_rule5_pareto_rows"] = int(len(pareto_df))
    manifest["n_rule5_top3_rows"] = int(len(top3_df))
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 126)
    print("WEEK 12.4B — PHYSICAL COMPILE SUMMARY")
    print("=" * 126)
    print(f"Unique logical candidates:   {len(candidates)}")
    print(f"Layout summaries:            {len(summary)}")
    print(f"Strict-pass summaries:       {int(summary['strict_pass'].astype(bool).sum())}")
    print(f"Strict-fail summaries:       {len(failed)}")
    print(f"Rule-5 Pareto rows:          {len(pareto_df)}")
    print(f"Rule-5 top-{TOP_K_LAYOUTS} rows:           {len(top3_df)}")

    if len(top3_df):
        cols_print = [
            "candidate_uid", "selected_rules", "backend", "rule5_rank", "layout",
            "min_t2_echo_us", "memory_min_t2_echo_us", "min_cz_fidelity",
            "min_readout_fidelity", "NCZ_median", "NSWAP_median",
            "depth_max_setting_median", "Tcircuit_max_setting_us_median",
            "memory_duration_over_t2_echo_max",
        ]
        cols_print = [c for c in cols_print if c in top3_df.columns]
        print("\nRULE-5 RETAINED PHYSICAL LAYOUTS")
        print("-" * 126)
        print(top3_df[cols_print].to_string(index=False))

    print("\nSaved:")
    for p in outputs:
        print(f"  {p}")
    print("\nNo QPU jobs submitted. No shots executed. 2026 untouched.")
    print("STOP HERE. Interpret Rule-5 physical compilation before noise/QPU execution.")


if __name__ == "__main__":
    main()
