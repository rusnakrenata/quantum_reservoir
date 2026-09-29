"""
WEEK 9.05C — H4 RWP RULE-3 / RULE-4 / RULE-5 CLOSURE

This script closes the remaining H4 RWP selection evidence after 09.04C2
and 09.05B.

WHAT IT DOES
------------
1. Rule 3:
   Consolidates the hardware-robust H4 representative from the full
   09.04C2 candidate table using the SAME non-weighted hierarchy already
   implemented in 09_04_rwp_common.py.

2. Rule 4:
   Applies the SAME one-standard-error resource rule used by the original
   H0-H3 09_05c script:
       threshold = best_CV + best_CV_SD/sqrt(5)
   and, inside the eligible set, minimizes:
       feature_vector_cz
       -> n_settings
       -> logical_reset_count
       -> n_features
       -> cv_rmse
       -> shot-noise forecast SD

   IMPORTANT: the candidate universe contains r=1,2,3.
   We do NOT force r=3.

3. Rule 5 / physical closure:
   For the selected Rule-4 H4 reservoir only, refreshes IBM hardware
   embeddings on Fez/Kingston/Marrakesh and strictly compiles sampled RWP
   endpoints with routing_method="none". No QPU jobs are submitted.

DATA DISCIPLINE
---------------
- Selection uses 2022-2024 chronological CV / training-only robustness.
- 2025 values remain diagnostic only.
- 2026 is never loaded.

INPUT
-----
results/09_04c2_h4_backfill_all_trials.csv

DEPENDENCIES
------------
09_04_rwp_common.py
09_05c_rwp_resource_and_hardware_gapfill.py
11_0A_fresh_full_hardware_reselection.py
ibm_account.py
07_02b_memory_pair_isolation.py

OUTPUTS
-------
results/09_05c_h4_rule3_selected.csv
results/09_05c_h4_rule4_eligible_1se.csv
results/09_05c_h4_rule4_selected_before_compile.csv
results/09_05c_h4_rule4_compile_detail.csv
results/09_05c_h4_rule4_compile_per_endpoint.csv
results/09_05c_h4_rule4_compile_summary.csv
results/09_05c_h4_rule5_fresh_embedding_metrics.csv
results/09_05c_h4_manifest.json
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
from qiskit import transpile

from ibm_account import get_service


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
ORIGINAL_0905C_FILE = HERE / "09_05c_rwp_resource_and_hardware_gapfill.py"
SELECTOR_FILE = HERE / "11_0A_live_embedding_reselection.py"

TRIAL_FILE = RESULTS / "09_04c2_h4_backfill_all_trials.csv"

RULE3_FILE = RESULTS / "09_05c_h4_rule3_selected.csv"
RULE4_ELIGIBLE_FILE = RESULTS / "09_05c_h4_rule4_eligible_1se.csv"
RULE4_SELECTED_FILE = RESULTS / "09_05c_h4_rule4_selected_before_compile.csv"
COMPILE_DETAIL_FILE = RESULTS / "09_05c_h4_rule4_compile_detail.csv"
COMPILE_ENDPOINT_FILE = RESULTS / "09_05c_h4_rule4_compile_per_endpoint.csv"
COMPILE_SUMMARY_FILE = RESULTS / "09_05c_h4_rule4_compile_summary.csv"
RULE5_FILE = RESULTS / "09_05c_h4_rule5_fresh_embedding_metrics.csv"
MANIFEST_FILE = RESULTS / "09_05c_h4_manifest.json"

BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]

N_FOLDS = 5
OPT_LEVEL = 1
SEED_TRANSPILE = 42


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Keep the original project script beside this file."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_0905c_h4_common")
original = load_module(ORIGINAL_0905C_FILE, "qrc_0905c_original")
selector = load_module(SELECTOR_FILE, "qrc_0905c_h4_selector")


def as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def reconstruct_candidate(row: pd.Series) -> dict:
    """
    C2 H4 CSVs use the stable H4 coupling schema, so candidate_from_row()
    safely reconstructs the exact numerical dynamics.
    """
    candidate = common.candidate_from_row(row)

    if candidate["topology"] != "H4":
        raise RuntimeError("Expected H4-only C2 input.")

    candidate["window"] = int(row["window"])
    candidate["candidate_id"] = (
        f"H4_W{int(row['window']):02d}_r{int(row['r'])}_"
        f"{str(row.get('c2_search_id', row.get('search_id', '')))}"
    )

    checks = {
        "alpha": float(row["alpha"]),
        "dt": float(row["dt"]),
        "r": int(row["r"]),
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
    }

    for key, observed in checks.items():
        expected = candidate[key]
        if key == "r":
            ok = int(expected) == int(observed)
        else:
            ok = np.isclose(
                float(expected),
                float(observed),
                rtol=1e-10,
                atol=1e-12,
            )
        if not ok:
            raise RuntimeError(
                f"H4 C2 reconstruction audit failed: "
                f"W={int(row['window'])}, r={int(row['r'])}, "
                f"{key}: row={observed}, candidate={expected}"
            )

    return candidate


def select_rule3(trials: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Exact project hardware-aware hierarchy from 09_04_rwp_common.py.

    If any candidate is fully shot-resolvable at 1024 shots:
        restrict to safe set -> minimize CV -> robustness/resources tie-break.
    Otherwise:
        minimize Rmax -> shotSD -> CV -> sensitivity/resources.
    """
    row, mode = common.select_hardware_aware_row(trials)
    return row.copy(), mode


def select_rule4(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    """
    Exact original 09.05C one-standard-error rule for one topology.
    """
    best = trials.sort_values("cv_rmse").iloc[0]

    threshold = (
        float(best["cv_rmse"])
        + float(best["cv_rmse_std"]) / math.sqrt(N_FOLDS)
    )

    eligible = trials[
        trials["cv_rmse"] <= threshold
    ].copy()

    pick = (
        eligible.sort_values(
            [
                "feature_vector_cz",
                "n_settings",
                "logical_reset_count",
                "n_features",
                "cv_rmse",
                "shot_noise_forecast_sd_proxy_1024",
            ]
        )
        .iloc[0]
        .copy()
    )

    pick["one_se_threshold"] = threshold
    pick["best_cv_reference"] = float(best["cv_rmse"])
    pick["best_cv_sd_reference"] = float(best["cv_rmse_std"])

    return eligible, pick, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default="all",
        choices=["all", *BACKENDS],
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of evenly spaced 2025 endpoints used only for compile/resource profiling.",
    )
    parser.add_argument(
        "--shortlist",
        type=int,
        default=80,
        help="Fresh physical-layout shortlist passed to the existing selector.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Run Rule 3/4 logical selection but skip live IBM refresh/compilation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = [
        RULE3_FILE,
        RULE4_ELIGIBLE_FILE,
        RULE4_SELECTED_FILE,
        COMPILE_DETAIL_FILE,
        COMPILE_ENDPOINT_FILE,
        COMPILE_SUMMARY_FILE,
        RULE5_FILE,
        MANIFEST_FILE,
    ]

    if not TRIAL_FILE.exists():
        raise FileNotFoundError(
            f"{TRIAL_FILE} missing. Run the H4 09.04C2 step first."
        )

    if args.overwrite:
        for p in outputs:
            if p.exists():
                p.unlink()

    trials = pd.read_csv(TRIAL_FILE)

    if set(trials["topology"].astype(str).unique()) != {"H4"}:
        raise RuntimeError("Expected H4-only 09.04C2 trial table.")

    observed_r = sorted(
        pd.to_numeric(trials["r"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    if observed_r != [1, 2, 3]:
        raise RuntimeError(
            f"Expected r=[1,2,3] in H4 C2 universe; found {observed_r}"
        )

    print("=" * 132)
    print("WEEK 9.05C — H4 RULE-3 / RULE-4 / RULE-5 CLOSURE")
    print("=" * 132)
    print(f"Candidate rows: {len(trials)}")
    print(f"Trotter values retained: {observed_r}")
    print("2022-2024 = selection; 2025 = diagnostic/compile sampling only; 2026 = untouched.")
    print()

    # ------------------------------------------------------------------
    # RULE 3 — hardware-robust readout
    # ------------------------------------------------------------------
    rule3, rule3_mode = select_rule3(trials)
    rule3["rule"] = "Rule3"
    rule3["selection_rule"] = rule3_mode
    rule3["2025_used_for_selection"] = False
    rule3["2026_used"] = False

    pd.DataFrame([rule3]).to_csv(
        RULE3_FILE,
        index=False,
    )

    print("RULE 3 — H4 HARDWARE-ROBUST READOUT")
    print(
        f"  W={int(rule3['window'])}, r={int(rule3['r'])}, "
        f"{rule3.get('c2_search_id', rule3.get('search_id', ''))} / "
        f"{rule3['readout']}"
    )
    print(
        f"  CV={float(rule3['cv_rmse']):.6f}, "
        f"CV_SD={float(rule3['cv_rmse_std']):.6f}, "
        f"shotSD={float(rule3['shot_noise_forecast_sd_proxy_1024']):.3f}, "
        f"Rmax={float(rule3['max_shot_to_train_std_ratio_1024']):.3f}"
    )
    print(f"  hierarchy: {rule3_mode}")
    print()

    # ------------------------------------------------------------------
    # RULE 4 — one-standard-error resource selection
    # ------------------------------------------------------------------
    eligible, rule4, threshold = select_rule4(trials)

    eligible = eligible.copy()
    eligible["one_se_threshold"] = threshold
    eligible["within_one_se"] = True
    eligible["2025_used_for_selection"] = False
    eligible["2026_used"] = False

    eligible.to_csv(
        RULE4_ELIGIBLE_FILE,
        index=False,
    )

    rule4["rule"] = "Rule4"
    rule4["selection_rule"] = (
        "1SE set -> feature_vector_cz -> n_settings -> logical_reset_count "
        "-> n_features -> CV -> shotSD"
    )
    rule4["2025_used_for_selection"] = False
    rule4["2026_used"] = False

    pd.DataFrame([rule4]).to_csv(
        RULE4_SELECTED_FILE,
        index=False,
    )

    eligible_r_counts = {
        str(int(k)): int(v)
        for k, v in (
            pd.to_numeric(eligible["r"])
            .astype(int)
            .value_counts()
            .sort_index()
            .items()
        )
    }

    print("RULE 4 — H4 ONE-STANDARD-ERROR RESOURCE REPRESENTATIVE")
    print(
        f"  best CV={float(rule4['best_cv_reference']):.6f}, "
        f"best CV_SD={float(rule4['best_cv_sd_reference']):.6f}"
    )
    print(f"  1SE threshold={threshold:.6f}")
    print(f"  eligible rows={len(eligible)}; r counts={eligible_r_counts}")
    print(
        f"  selected: W={int(rule4['window'])}, r={int(rule4['r'])}, "
        f"{rule4.get('c2_search_id', rule4.get('search_id', ''))} / "
        f"{rule4['readout']}"
    )
    print(
        f"  CV={float(rule4['cv_rmse']):.6f}, "
        f"CZ/feature(logical proxy)={int(rule4['feature_vector_cz'])}, "
        f"settings={int(rule4['n_settings'])}, "
        f"resets={int(rule4['logical_reset_count'])}, "
        f"features={int(rule4['n_features'])}"
    )
    print()

    manifest = {
        "step": "09.05C_H4_RWP_rule3_rule4_rule5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(TRIAL_FILE),
        "candidate_rows": int(len(trials)),
        "r_values_retained": observed_r,
        "selection_period": "2022-2024",
        "validation_2025_role": "diagnostic_and_compile_sampling_only",
        "test_2026_used": False,
        "rule3": {
            "window": int(rule3["window"]),
            "r": int(rule3["r"]),
            "search_id": str(
                rule3.get("c2_search_id", rule3.get("search_id", ""))
            ),
            "readout": str(rule3["readout"]),
            "cv_rmse": float(rule3["cv_rmse"]),
            "cv_rmse_std": float(rule3["cv_rmse_std"]),
            "shotSD_1024": float(
                rule3["shot_noise_forecast_sd_proxy_1024"]
            ),
            "Rmax_1024": float(
                rule3["max_shot_to_train_std_ratio_1024"]
            ),
            "selection_rule": rule3_mode,
        },
        "rule4": {
            "best_cv": float(rule4["best_cv_reference"]),
            "best_cv_sd": float(rule4["best_cv_sd_reference"]),
            "one_se_threshold": float(threshold),
            "eligible_rows": int(len(eligible)),
            "eligible_r_counts": eligible_r_counts,
            "window": int(rule4["window"]),
            "r": int(rule4["r"]),
            "search_id": str(
                rule4.get("c2_search_id", rule4.get("search_id", ""))
            ),
            "readout": str(rule4["readout"]),
            "cv_rmse": float(rule4["cv_rmse"]),
            "feature_vector_cz_logical_proxy": int(
                rule4["feature_vector_cz"]
            ),
            "n_settings": int(rule4["n_settings"]),
            "logical_reset_count": int(rule4["logical_reset_count"]),
            "n_features": int(rule4["n_features"]),
        },
        "physical_compilation_skipped": bool(args.selection_only),
    }

    if args.selection_only:
        with MANIFEST_FILE.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("Selection-only mode: live IBM refresh/compilation skipped.")
        print(f"Saved: {RULE3_FILE}")
        print(f"Saved: {RULE4_ELIGIBLE_FILE}")
        print(f"Saved: {RULE4_SELECTED_FILE}")
        print(f"Saved: {MANIFEST_FILE}")
        return

    # ------------------------------------------------------------------
    # RULE 5 / PHYSICAL RESOURCE CLOSURE FOR RULE-4 REPRESENTATIVE
    # ------------------------------------------------------------------
    candidate = reconstruct_candidate(rule4)
    W = int(rule4["window"])
    readout = str(rule4["readout"])

    # Keep selector's globals aligned with the actual H4 candidate.
    if hasattr(selector, "ALPHA"):
        selector.ALPHA = float(candidate["alpha"])
    if hasattr(selector, "DT"):
        selector.DT = float(candidate["dt"])

    work_tv, cols, _ = common.load_train_validation()
    angles = common.make_angles(
        work_tv,
        cols,
        float(candidate["alpha"]),
    )

    valid_endpoints = np.arange(
        common.N_TRAIN,
        common.N_TRAIN + common.N_VAL,
    )
    sample_idx = np.linspace(
        0,
        len(valid_endpoints) - 1,
        min(int(args.samples), len(valid_endpoints)),
        dtype=int,
    )
    endpoints = valid_endpoints[sample_idx]

    service = get_service()
    backend_names = (
        BACKENDS
        if args.backend == "all"
        else [args.backend]
    )

    detailed = []
    mapping_rows = []

    print("=" * 132)
    print("RULE 5 / PHYSICAL RESOURCE CLOSURE")
    print("=" * 132)
    print("NO QPU JOBS ARE SUBMITTED.")
    print(
        f"Compiling H4 Rule-4 candidate: "
        f"W={W}, r={candidate['r']}, readout={readout}"
    )
    print(
        f"Backends={backend_names}; sampled endpoints={len(endpoints)}; "
        f"optimization_level={OPT_LEVEL}; routing=none"
    )
    print()

    for backend_name in backend_names:
        if hasattr(selector, "ALPHA"):
            selector.ALPHA = float(candidate["alpha"])
        if hasattr(selector, "DT"):
            selector.DT = float(candidate["dt"])

        fresh = selector.fresh_hardware_reselection(
            service=service,
            candidate=candidate,
            backend_names=[backend_name],
            shortlist=int(args.shortlist),
            write_prefix=f"09_05c_h4_{backend_name}",
            verbose=False,
        )

        chosen = fresh["selected"]
        layout = json.loads(chosen["layout"])

        mapping_rows.append(
            {
                "topology": "H4",
                "window": W,
                "r": int(candidate["r"]),
                "readout": readout,
                "backend": backend_name,
                "layout": chosen["layout"],
                "physical_C_t": chosen.get("physical_C_t"),
                "physical_D": chosen.get("physical_D"),
                "physical_P_t": chosen.get("physical_P_t"),
                "physical_H": chosen.get("physical_H"),
                "physical_M1": chosen.get("physical_M1"),
                "physical_M2": chosen.get("physical_M2"),
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
                "compiled_duration_us_core": chosen.get(
                    "compiled_duration_us"
                ),
                "pending_jobs": chosen.get("pending_jobs"),
            }
        )

        backend = service.backend(
            backend_name,
            use_fractional_gates=False,
        )
        try:
            backend.refresh()
        except Exception:
            pass

        for endpoint in endpoints:
            for setting in original.readout_settings(readout):
                logical = original.build_rwp_measurement_circuit(
                    candidate,
                    angles,
                    int(endpoint),
                    W,
                    setting,
                )

                isa = transpile(
                    logical,
                    backend=backend,
                    initial_layout=layout,
                    routing_method="none",
                    optimization_level=OPT_LEVEL,
                    seed_transpiler=SEED_TRANSPILE,
                    scheduling_method="alap",
                )

                backend.check_faulty(isa)

                rr = original.resource_row(
                    isa,
                    backend,
                    {
                        "topology": "H4",
                        "window": W,
                        "readout": readout,
                        "r": int(candidate["r"]),
                        "backend": backend_name,
                        "endpoint": int(endpoint),
                        "setting": setting[0],
                    },
                )

                if int(rr["n_swap"]) != 0:
                    raise RuntimeError(
                        f"H4 {backend_name}: SWAP detected in strict native compile."
                    )

                detailed.append(rr)

        print(
            f"H4 W={W:02d} r={candidate['r']} {readout:28s} "
            f"{backend_name}: layout={layout}"
        )

    detail_df = pd.DataFrame(detailed)
    map_df = pd.DataFrame(mapping_rows)

    detail_df.to_csv(
        COMPILE_DETAIL_FILE,
        index=False,
    )
    map_df.to_csv(
        RULE5_FILE,
        index=False,
    )

    per_endpoint = (
        detail_df.groupby(
            [
                "topology",
                "window",
                "readout",
                "r",
                "backend",
                "endpoint",
            ],
            dropna=False,
        )
        .agg(
            n_settings=("setting", "nunique"),
            N2q_feature_vector=("n_2q_reported", "sum"),
            NCZ_feature_vector=("n_cz", "sum"),
            NSWAP_feature_vector=("n_swap", "sum"),
            depth_max_setting=("depth", "max"),
            depth_sum_settings=("depth", "sum"),
            Tcircuit_feature_vector_us=("duration_us", "sum"),
            Tcircuit_max_setting_us=("duration_us", "max"),
            resets_sum_settings=("n_reset", "sum"),
        )
        .reset_index()
    )

    per_endpoint.to_csv(
        COMPILE_ENDPOINT_FILE,
        index=False,
    )

    summary = (
        per_endpoint.groupby(
            [
                "topology",
                "window",
                "readout",
                "r",
                "backend",
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
                "resets_sum_settings",
                "median",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        COMPILE_SUMMARY_FILE,
        index=False,
    )

    manifest["physical_compilation"] = {
        "backends": backend_names,
        "sampled_endpoints": int(len(endpoints)),
        "optimization_level": OPT_LEVEL,
        "seed_transpiler": SEED_TRANSPILE,
        "routing_method": "none",
        "all_zero_swap": bool(
            (pd.to_numeric(summary["NSWAP_median"]) == 0).all()
        ),
    }

    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("PHYSICAL RESOURCE SUMMARY")
    print(summary.to_string(index=False))
    print()
    print("Saved:")
    print(f"  {RULE3_FILE}")
    print(f"  {RULE4_ELIGIBLE_FILE}")
    print(f"  {RULE4_SELECTED_FILE}")
    print(f"  {COMPILE_DETAIL_FILE}")
    print(f"  {COMPILE_ENDPOINT_FILE}")
    print(f"  {COMPILE_SUMMARY_FILE}")
    print(f"  {RULE5_FILE}")
    print(f"  {MANIFEST_FILE}")
    print()
    print(
        "STOP HERE. Send the full console output and the result files. "
        "Do not start Week-10 H4 noise simulation yet."
    )


if __name__ == "__main__":
    main()
