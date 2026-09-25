#!/usr/bin/env python
from __future__ import annotations

"""
WEEK 11.1F.1 — CANDIDATE #1 LAYOUT-POLICY AUDIT
================================================

Audit CURRENT ibm_kingston calibration without submitting any QPU job.

Compare:
A) selector current choice,
B) strict zero-SWAP -> lexicographic Rule-5,
C) strict zero-SWAP -> calibration-Pareto only -> same lexicographic tie-break.

Also re-evaluate three historically relevant Candidate-1 layouts on the same
current calibration.

NO QPU JOB IS SUBMITTED. 2026 is never loaded.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import transpile

from ibm_account import get_service

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
MANIFEST_FILE = RESULTS / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"
SELECTOR_CANDIDATES = [
    HERE / "11_0A_fresh_full_hardware_reselection.py",
    HERE / "11_0A_live_embedding_reselection.py",
]

CANDIDATE_KEY = "RWP_H3_R1R3"
TOPOLOGY = "H3"
WINDOW = 4
TROTTER_R = 3
BACKEND_NAME = "ibm_kingston"
OPT_LEVEL = 1
SEED_TRANSPILE = 42
DEFAULT_SHORTLIST = 120

HISTORICAL_LAYOUTS = {
    "original_good_4p8976": [73, 79, 93, 94, 74, 75],
    "later_m3_raw_6p3772": [82, 83, 96, 103, 81, 80],
    "recent_sampler_9p3559": [2, 3, 16, 23, 1, 0],
}

RULE5_COLS = [
    "compiled_2q_error_max",
    "max_readout_error",
    "compiled_1q_error_max",
    "min_t2_us",
    "min_t1_us",
    "compiled_duration_us",
]
RULE5_ASC = [True, True, True, False, False, True]

DISPLAY_COLS = [
    "layout",
    "calibration_pareto",
    "strict_compile_pass",
    "compiled_2q_error_max_percent",
    "max_readout_error_percent",
    "compiled_1q_error_max_percent",
    "min_t1_us",
    "min_t2_us",
    "compiled_duration_us",
    "R_T2",
    "compiled_n_swap",
]


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def find_selector_file() -> Path:
    for p in SELECTOR_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find selector. Expected one of:\n  "
        + "\n  ".join(str(p) for p in SELECTOR_CANDIDATES)
    )


common = load_module(COMMON_FILE, "qrc_11f1_common")
SELECTOR_FILE = find_selector_file()
selector = load_module(SELECTOR_FILE, "qrc_11f1_selector")


def parse_layout(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x]
    return [int(v) for v in json.loads(str(x))]


def layout_key(x):
    return tuple(parse_layout(x))


def load_candidate():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(MANIFEST_FILE)

    df = pd.read_csv(MANIFEST_FILE)
    sub = df[df["candidate_key"].astype(str) == CANDIDATE_KEY].copy()
    if len(sub) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest row for {CANDIDATE_KEY}; got {len(sub)}."
        )

    row = sub.iloc[0]
    if str(row["topology"]) != TOPOLOGY:
        raise RuntimeError(f"Expected topology {TOPOLOGY}.")
    if int(row["window"]) != WINDOW:
        raise RuntimeError(f"Expected W={WINDOW}.")
    if int(row["original_r"]) != TROTTER_R:
        raise RuntimeError(f"Expected r={TROTTER_R}.")

    raw_j = json.loads(str(row["J_json"]))
    J = {}
    for edge in common.TOPOLOGY_EDGES[TOPOLOGY]:
        i, j = edge
        direct = f"J{i}{j}"
        reverse = f"J{j}{i}"
        if direct in raw_j:
            v = raw_j[direct]
        elif reverse in raw_j:
            v = raw_j[reverse]
        else:
            raise KeyError(f"Missing coupling for H3 edge {edge}.")
        J[edge] = float(v)

    return {
        "candidate_id": CANDIDATE_KEY,
        "topology": TOPOLOGY,
        "window": WINDOW,
        "r": TROTTER_R,
        "alpha": float(row["alpha"]),
        "dt": float(row["dt"]),
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
        "J": J,
    }


def strict_rule5_sort(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RULE5_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing Rule-5 columns: {missing}")
    return (
        df.sort_values(
            RULE5_COLS,
            ascending=RULE5_ASC,
            na_position="last",
        )
        .reset_index(drop=True)
    )


def add_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "compiled_2q_error_max_percent" not in out.columns and "compiled_2q_error_max" in out.columns:
        out["compiled_2q_error_max_percent"] = 100.0 * pd.to_numeric(out["compiled_2q_error_max"], errors="coerce")
    if "compiled_1q_error_max_percent" not in out.columns and "compiled_1q_error_max" in out.columns:
        out["compiled_1q_error_max_percent"] = 100.0 * pd.to_numeric(out["compiled_1q_error_max"], errors="coerce")
    if "max_readout_error_percent" not in out.columns and "max_readout_error" in out.columns:
        out["max_readout_error_percent"] = 100.0 * pd.to_numeric(out["max_readout_error"], errors="coerce")
    if "R_T2" not in out.columns and {"compiled_duration_us", "min_t2_us"}.issubset(out.columns):
        t = pd.to_numeric(out["compiled_duration_us"], errors="coerce")
        t2 = pd.to_numeric(out["min_t2_us"], errors="coerce")
        out["R_T2"] = t / t2
    return out


def bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def evaluate_explicit_layout(backend, fresh, candidate, layout, label):
    qubits = fresh["qubits"]
    metrics = selector.mapping_metrics(backend, TOPOLOGY, layout, qubits)

    if metrics is None:
        return {
            "historical_label": label,
            "layout": json.dumps(layout),
            "strict_compile_pass": False,
            "compile_error": "Native/calibrated H3 mapping unavailable on current snapshot.",
            "calibration_pareto": np.nan,
        }

    angles, _ = selector.first_input_angles()
    core = selector.build_core(candidate, angles)

    try:
        compiled = transpile(
            core,
            backend=backend,
            initial_layout=layout,
            routing_method="none",
            optimization_level=OPT_LEVEL,
            seed_transpiler=SEED_TRANSPILE,
            scheduling_method="alap",
        )
        backend.check_faulty(compiled)
        cm = selector.compiled_metrics(compiled, backend)
        strict_pass = int(cm.get("compiled_n_swap", -1)) == 0
        err = ""
    except Exception as exc:
        cm = {}
        strict_pass = False
        err = str(exc)

    return {
        "historical_label": label,
        "backend": BACKEND_NAME,
        "layout": json.dumps(layout),
        **metrics,
        **cm,
        "strict_compile_pass": strict_pass,
        "calibration_pareto": np.nan,
        "compile_error": err,
    }


def print_table(title, df, n=10):
    print()
    print("=" * 150)
    print(title)
    print("=" * 150)
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    extra = [c for c in ["rank", "historical_label"] if c in df.columns]
    print(df[extra + cols].head(n).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=BACKEND_NAME)
    parser.add_argument("--shortlist", type=int, default=DEFAULT_SHORTLIST)
    args = parser.parse_args()

    print("=" * 150)
    print("WEEK 11.1F.1 — CANDIDATE #1 CURRENT-LAYOUT POLICY AUDIT")
    print("=" * 150)
    print("NO QPU JOB WILL BE SUBMITTED.")
    print("2026 is not loaded.")
    print(f"Selector file: {SELECTOR_FILE.name}")
    print()

    candidate = load_candidate()

    if hasattr(selector, "ALPHA"):
        selector.ALPHA = float(candidate["alpha"])
    if hasattr(selector, "DT"):
        selector.DT = float(candidate["dt"])

    service = get_service()

    fresh = selector.fresh_hardware_reselection(
        service=service,
        candidate=candidate,
        backend_names=[args.backend],
        shortlist=int(args.shortlist),
        write_prefix="11_1F1_C1_layout_policy_audit",
        verbose=True,
    )

    if "compiled" not in fresh:
        raise KeyError("selector result has no fresh['compiled'] table.")

    compiled = add_percent_columns(fresh["compiled"].copy())
    if "strict_compile_pass" not in compiled.columns:
        raise KeyError("fresh['compiled'] has no strict_compile_pass column.")

    good = compiled[bool_series(compiled["strict_compile_pass"])].copy()
    if len(good) == 0:
        raise RuntimeError("No strict zero-SWAP candidates survived.")

    # Policy A: current all-strict lexicographic ranking.
    strict_ranked = strict_rule5_sort(good)
    strict_ranked["rank"] = np.arange(len(strict_ranked)) + 1

    # Policy B: require calibration Pareto membership first.
    if "calibration_pareto" not in good.columns:
        raise KeyError(
            "fresh['compiled'] has no calibration_pareto column; cannot audit Pareto-first faithfully."
        )
    pareto_good = good[bool_series(good["calibration_pareto"])].copy()
    if len(pareto_good) == 0:
        raise RuntimeError("No strict compiled calibration-Pareto embeddings survived.")
    pareto_ranked = strict_rule5_sort(pareto_good)
    pareto_ranked["rank"] = np.arange(len(pareto_ranked)) + 1

    selected = fresh["selected"]
    selected_dict = selected.to_dict() if hasattr(selected, "to_dict") else dict(selected)
    selected_layout = parse_layout(selected_dict["layout"])

    print("SELECTOR CURRENT CHOICE")
    print(f"  layout={selected_layout}")
    print(f"  calibration_pareto={selected_dict.get('calibration_pareto')}")
    print(f"  max CZ={float(selected_dict.get('compiled_2q_error_max_percent', np.nan)):.6f}%")
    print(f"  max RO={float(selected_dict.get('max_readout_error_percent', np.nan)):.6f}%")
    print(f"  min T2={float(selected_dict.get('min_t2_us', np.nan)):.3f} us")

    print_table("POLICY A — STRICT ZERO-SWAP -> LEXICOGRAPHIC RULE 5", strict_ranked, n=10)
    print_table(
        "POLICY B — STRICT ZERO-SWAP -> CALIBRATION PARETO -> SAME LEXICOGRAPHIC TIE-BREAK",
        pareto_ranked,
        n=10,
    )

    backend = service.backend(args.backend, use_fractional_gates=False)
    try:
        backend.refresh()
    except Exception:
        pass

    historical_rows = []
    current_keys = compiled["layout"].apply(layout_key)
    for label, layout in HISTORICAL_LAYOUTS.items():
        mask = current_keys == tuple(layout)
        if mask.any():
            row = compiled.loc[mask].iloc[0].to_dict()
            row["historical_label"] = label
            historical_rows.append(row)
        else:
            historical_rows.append(
                evaluate_explicit_layout(backend, fresh, candidate, layout, label)
            )

    hist = add_percent_columns(pd.DataFrame(historical_rows))
    rank_map = {layout_key(r["layout"]): int(r["rank"]) for _, r in strict_ranked.iterrows()}
    pareto_rank_map = {layout_key(r["layout"]): int(r["rank"]) for _, r in pareto_ranked.iterrows()}
    hist["strict_lexicographic_rank"] = [rank_map.get(layout_key(v), np.nan) for v in hist["layout"]]
    hist["pareto_lexicographic_rank"] = [pareto_rank_map.get(layout_key(v), np.nan) for v in hist["layout"]]

    print()
    print("=" * 150)
    print("HISTORICAL C1 LAYOUTS — SAME CURRENT CALIBRATION")
    print("=" * 150)
    hist_cols = [
        "historical_label",
        "layout",
        "calibration_pareto",
        "strict_compile_pass",
        "strict_lexicographic_rank",
        "pareto_lexicographic_rank",
        "compiled_2q_error_max_percent",
        "max_readout_error_percent",
        "compiled_1q_error_max_percent",
        "min_t1_us",
        "min_t2_us",
        "compiled_duration_us",
        "R_T2",
        "compiled_n_swap",
    ]
    hist_cols = [c for c in hist_cols if c in hist.columns]
    print(hist[hist_cols].to_string(index=False))

    compiled.to_csv(RESULTS / "11_1F1_C1_current_compiled_pool.csv", index=False)
    strict_ranked.to_csv(RESULTS / "11_1F1_C1_policyA_strict_lexicographic.csv", index=False)
    pareto_ranked.to_csv(RESULTS / "11_1F1_C1_policyB_pareto_lexicographic.csv", index=False)
    hist.to_csv(RESULTS / "11_1F1_C1_historical_layouts_current_metrics.csv", index=False)

    summary = {
        "backend": args.backend,
        "selector_file": SELECTOR_FILE.name,
        "n_compiled_rows": int(len(compiled)),
        "n_strict_zero_swap": int(len(good)),
        "n_strict_and_calibration_pareto": int(len(pareto_good)),
        "selector_selected_layout": selected_layout,
        "policyA_selected_layout": parse_layout(strict_ranked.iloc[0]["layout"]),
        "policyB_selected_layout": parse_layout(pareto_ranked.iloc[0]["layout"]),
    }
    (RESULTS / "11_1F1_C1_layout_policy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 150)
    print("SUMMARY")
    print("=" * 150)
    print(json.dumps(summary, indent=2))
    print()
    print("NO QPU JOB WAS SUBMITTED.")


if __name__ == "__main__":
    main()
