"""
WEEK 9.05A — H4 CONT REVISED READOUT / CONDITIONING COMPLETION

PURPOSE
-------
Complete the H4 CONT branch with the same post-QPU readout families and
1024-shot conditioning diagnostics used by the final H0-H3 Rule-3 analysis.

This is NOT a new dynamics search.

The numerical H4 CONT dynamics come from the already-completed fair
09.03C r={1,2,3} search.  For every UNIQUE H4 forecast-best dynamics in
results/09_03c_forecast_best_135.csv, this script re-evaluates the full
revised readout family:

    XZ_injection
    XZinj_dropX3
    XZinj_dropX3_plus_YX45
    XZinj_plus_YX45
    XYZ_all

The CONT state is propagated persistently through the complete 2022-2025
train+validation sequence.  There is no RWP restart.

SELECTION DISCIPLINE
--------------------
- 2022-2024 chronological CV selects Ridge lambda and candidates.
- 2025 is diagnostic only.
- 2026 is never loaded.
- r=1,2,3 are all retained.
- No weighted score is used.

Outputs include:
- complete H4 CONT revised-readout table,
- CV winner per r,
- hardware-aware Rule-3 representative per r,
- global H4 CONT Rule-1 representative,
- global H4 CONT Rule-3 representative,
- H4 CONT Rule-4 one-standard-error representative,
- manifest.

RULE 3
------
Uses the project hierarchy from 09_04_rwp_common.py:
if a fully shot-resolvable candidate exists (Rmax <= 1 for every active
feature), restrict to that set, then minimize training CV and use
robustness/resources only as tie-breakers.

RULE 4
------
Within one standard error of the H4 CONT best CV:
    CV <= CV_best + CV_SD_best/sqrt(5)
choose minimum:
    feature_vector_cz
    -> n_settings
    -> logical_reset_count
    -> n_features
    -> CV
    -> shotSD

For CONT resource bookkeeping, one temporal update is one logical feature
evaluation step, so window=1 is used ONLY in resource_metrics(); there are
no CONT injection resets.

INPUT
-----
results/09_03c_forecast_best_135.csv

DEPENDENCIES
------------
09_04_rwp_common.py
07_02b_memory_pair_isolation.py

OUTPUTS
-------
results/09_05a_h4_cont_conditioning_all.csv
results/09_05a_h4_cont_cv_winners_per_r.csv
results/09_05a_h4_cont_hw_winners_per_r.csv
results/09_05a_h4_cont_rule1_selected.csv
results/09_05a_h4_cont_rule3_selected.csv
results/09_05a_h4_cont_rule4_eligible_1se.csv
results/09_05a_h4_cont_rule4_selected.csv
results/09_05a_h4_cont_manifest.json
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


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
SOURCE_FILE = RESULTS / "09_03c_forecast_best_135.csv"

ALL_FILE = RESULTS / "09_05a_h4_cont_conditioning_all.csv"
CV_R_FILE = RESULTS / "09_05a_h4_cont_cv_winners_per_r.csv"
HW_R_FILE = RESULTS / "09_05a_h4_cont_hw_winners_per_r.csv"
RULE1_FILE = RESULTS / "09_05a_h4_cont_rule1_selected.csv"
RULE3_FILE = RESULTS / "09_05a_h4_cont_rule3_selected.csv"
RULE4_ELIGIBLE_FILE = RESULTS / "09_05a_h4_cont_rule4_eligible_1se.csv"
RULE4_FILE = RESULTS / "09_05a_h4_cont_rule4_selected.csv"
MANIFEST_FILE = RESULTS / "09_05a_h4_cont_manifest.json"

R_VALUES = [1, 2, 3]
N_FOLDS = 5


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


common = load_module(COMMON_FILE, "qrc_0905a_h4_common")


def dynamics_signature(row: pd.Series) -> tuple:
    """Exact H4 CONT numerical-dynamics identity."""
    return (
        str(row["config_id"]),
        int(row["trotter_r"]),
        round(float(row["hx"]), 12),
        round(float(row["hy"]), 12),
        tuple(
            round(float(row[col]), 12)
            for col in ["J12", "J01", "J04", "J34", "J45"]
        ),
    )


def build_cont_master(candidate: dict, angles: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    """
    Persistent CONT trajectory using the same reduced-state recursion as
    09.03C, but emitting the full revised feature bank needed by the five
    final readout families.
    """
    A_list = common.build_channels(candidate, angles)

    rows = []
    rho_m = common.qrc.memory_zero_density()
    max_trace_err = 0.0
    max_herm_err = 0.0

    for A in A_list:
        rho_i, rho_m_out = common.qrc.final_reduced_states(A, rho_m)

        max_trace_err = max(
            max_trace_err,
            abs(float(np.trace(rho_i).real) - 1.0),
            abs(float(np.trace(rho_m_out).real) - 1.0),
        )
        max_herm_err = max(
            max_herm_err,
            float(np.linalg.norm(rho_i - rho_i.conj().T, ord="fro")),
            float(np.linalg.norm(rho_m_out - rho_m_out.conj().T, ord="fro")),
        )

        rows.append(
            common.reduced_feature_row(
                rho_i,
                rho_m_out,
            )
        )
        rho_m = rho_m_out

    return pd.DataFrame(rows), max_trace_err, max_herm_err


def select_cv(group: pd.DataFrame) -> pd.Series:
    return (
        group.sort_values(
            [
                "cv_rmse",
                "n_settings",
                "n_features",
                "feature_vector_cz",
                "selected_lambda",
            ]
        )
        .iloc[0]
        .copy()
    )


def select_rule4(all_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    best = all_df.sort_values("cv_rmse").iloc[0]
    threshold = (
        float(best["cv_rmse"])
        + float(best["cv_rmse_std"]) / math.sqrt(N_FOLDS)
    )

    eligible = all_df[
        all_df["cv_rmse"] <= threshold
    ].copy()

    # Exact hierarchy used in the RWP 09.05C resource closure, specialized
    # to the CONT candidate table.  CONT has zero temporal injection resets.
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-dynamics",
        type=int,
        default=None,
        help="Optional plumbing test; omit for the scientific full run.",
    )
    args = parser.parse_args()

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"{SOURCE_FILE} missing. Week 09.03C must be complete."
        )

    output_files = [
        ALL_FILE,
        CV_R_FILE,
        HW_R_FILE,
        RULE1_FILE,
        RULE3_FILE,
        RULE4_ELIGIBLE_FILE,
        RULE4_FILE,
        MANIFEST_FILE,
    ]

    if args.overwrite:
        for p in output_files:
            if p.exists():
                p.unlink()

    source = pd.read_csv(SOURCE_FILE)
    h4 = source[source["topology"].astype(str) == "H4"].copy()

    if len(h4) == 0:
        raise RuntimeError(
            "09_03c_forecast_best_135.csv contains no H4 rows."
        )

    observed_r = sorted(
        pd.to_numeric(h4["trotter_r"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    if observed_r != R_VALUES:
        raise RuntimeError(
            f"Expected H4 r={R_VALUES}; found {observed_r}."
        )

    # 09.03C has one forecast-best row for every historical readout branch/r.
    # Multiple branches can occasionally resolve to the same numerical
    # config_id; evaluate each unique dynamics only once.
    h4["_dyn_sig"] = h4.apply(dynamics_signature, axis=1)
    reps = (
        h4.sort_values(
            ["cv_rmse", "validation_rmse", "seed_candidate_id"]
        )
        .drop_duplicates("_dyn_sig")
        .reset_index(drop=True)
    )

    if args.max_dynamics is not None:
        reps = reps.iloc[: int(args.max_dynamics)].copy()

    work_tv, cols, y_all = common.load_train_validation()
    endpoints = np.arange(len(work_tv), dtype=int)

    if len(endpoints) != common.N_TRAIN + common.N_VAL:
        raise RuntimeError(
            f"Expected {common.N_TRAIN + common.N_VAL} CONT endpoints; "
            f"got {len(endpoints)}."
        )

    print("=" * 132)
    print("WEEK 9.05A — H4 CONT REVISED READOUT / CONDITIONING COMPLETION")
    print("=" * 132)
    print(f"H4 rows in 09.03C forecast-best table: {len(h4)}")
    print(f"Unique H4 CONT forecast-best dynamics to replay: {len(reps)}")
    print(f"Trotter values retained: {R_VALUES}")
    print(f"Revised readouts: {common.READOUTS}")
    print("CONT memory is persistent across all endpoints; no RWP restart.")
    print("1024-shot conditioning proxy; 2022-2024 selects; 2025 diagnostic; 2026 untouched.")
    print()

    rows = []

    for idx, src in reps.iterrows():
        candidate = common.candidate_from_row(src)
        candidate["topology"] = "H4"
        candidate["source_config_id"] = str(src["config_id"])
        candidate["source_candidate_id"] = str(src["seed_candidate_id"])
        candidate["candidate_id"] = str(src["config_id"])

        # 09.03C uses alpha=.75 and dt=1.6. If the clean source table does
        # not serialize them explicitly, candidate_from_row() applies the
        # same project defaults from common.
        angles = common.make_angles(
            work_tv,
            cols,
            float(candidate["alpha"]),
        )

        master, trace_err, herm_err = build_cont_master(
            candidate,
            angles,
        )

        if len(master) != len(endpoints):
            raise RuntimeError(
                f"{candidate['candidate_id']}: CONT bank length mismatch."
            )

        for readout in common.READOUTS:
            metrics = common.evaluate_master_bank(
                endpoints=endpoints,
                master_df=master,
                y_all=y_all,
                window=1,  # only endpoint indexing; temporal protocol is CONT
                readout=readout,
                n_splits=N_FOLDS,
            )

            resource = common.resource_metrics(
                candidate,
                1,  # one persistent CONT update per time step
                readout,
            )
            # Explicitly enforce CONT semantics.
            resource["logical_reset_count"] = 0

            row = {
                **common.candidate_to_flat(candidate),
                "protocol": "CONT",
                "source_09_03c_config_id": str(src["config_id"]),
                "source_09_03c_seed_candidate_id": str(
                    src["seed_candidate_id"]
                ),
                "source_09_03c_original_readout": str(src["readout"]),
                "source_09_03c_original_cv_rmse": float(src["cv_rmse"]),
                "source_09_03c_original_validation_rmse": float(
                    src["validation_rmse"]
                ),
                "source_seed_hy": (
                    float(src["seed_hy"])
                    if "seed_hy" in src.index and pd.notna(src["seed_hy"])
                    else np.nan
                ),
                "source_y_state": (
                    str(src["y_state"])
                    if "y_state" in src.index
                    else ""
                ),
                **{
                    k: v
                    for k, v in metrics.items()
                    if k not in {"folds", "cv_summary"}
                },
                **resource,
                "cont_max_trace_error": float(trace_err),
                "cont_max_hermiticity_error": float(herm_err),
                "2025_used_for_selection": False,
                "2026_used": False,
            }
            rows.append(row)

        print(
            f"[{idx+1:02d}/{len(reps)}] "
            f"{candidate['candidate_id']} | r={candidate['r']} | "
            f"five revised readouts complete"
        )

    all_df = pd.DataFrame(rows)

    if set(pd.to_numeric(all_df["r"]).astype(int).unique()) != set(R_VALUES):
        raise RuntimeError("Not all r=1,2,3 survived the full H4 CONT run.")

    all_df.to_csv(
        ALL_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # Per-r revised-readout representatives
    # ---------------------------------------------------------------
    cv_rows = []
    hw_rows = []

    for r, group in all_df.groupby("r", sort=True):
        cv = select_cv(group)
        cv["selection_scope"] = f"H4_CONT_r{int(r)}"
        cv_rows.append(cv)

        hw, mode = common.select_hardware_aware_row(group)
        hw = hw.copy()
        hw["selection_scope"] = f"H4_CONT_r{int(r)}"
        hw["selection_rule"] = mode
        hw_rows.append(hw)

    cv_r = pd.DataFrame(cv_rows).reset_index(drop=True)
    hw_r = pd.DataFrame(hw_rows).reset_index(drop=True)

    cv_r.to_csv(CV_R_FILE, index=False)
    hw_r.to_csv(HW_R_FILE, index=False)

    # ---------------------------------------------------------------
    # Rule 1 — global H4 CONT forecasting representative
    # ---------------------------------------------------------------
    rule1 = select_cv(all_df)
    rule1["rule"] = "Rule1"
    rule1["selection_rule"] = "minimum 2022-2024 chronological CV"
    pd.DataFrame([rule1]).to_csv(RULE1_FILE, index=False)

    # ---------------------------------------------------------------
    # Rule 3 — global H4 CONT hardware-robust representative
    # ---------------------------------------------------------------
    rule3, rule3_mode = common.select_hardware_aware_row(all_df)
    rule3 = rule3.copy()
    rule3["rule"] = "Rule3"
    rule3["selection_rule"] = rule3_mode
    pd.DataFrame([rule3]).to_csv(RULE3_FILE, index=False)

    # ---------------------------------------------------------------
    # Rule 4 — global H4 CONT one-SE resource representative
    # ---------------------------------------------------------------
    eligible, rule4, threshold = select_rule4(all_df)

    eligible["one_se_threshold"] = threshold
    eligible["within_one_se"] = True
    eligible.to_csv(RULE4_ELIGIBLE_FILE, index=False)

    rule4["rule"] = "Rule4"
    rule4["selection_rule"] = (
        "1SE set -> feature_vector_cz -> n_settings -> "
        "logical_reset_count -> n_features -> CV -> shotSD"
    )
    pd.DataFrame([rule4]).to_csv(RULE4_FILE, index=False)

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

    print()
    print("=" * 132)
    print("H4 CONT REVISED READOUT RESULTS")
    print("=" * 132)

    print("PER-r CV WINNERS:")
    for _, x in cv_r.iterrows():
        print(
            f"  r={int(x['r'])}: {x['readout']} | "
            f"CV={float(x['cv_rmse']):.6f} | "
            f"2025={float(x['validation_rmse']):.6f} diagnostic-only | "
            f"Rmax={float(x['max_shot_to_train_std_ratio_1024']):.3f}"
        )

    print()
    print("PER-r HARDWARE-AWARE WINNERS:")
    for _, x in hw_r.iterrows():
        print(
            f"  r={int(x['r'])}: {x['readout']} | "
            f"CV={float(x['cv_rmse']):.6f} | "
            f"shotSD={float(x['shot_noise_forecast_sd_proxy_1024']):.3f} | "
            f"Rmax={float(x['max_shot_to_train_std_ratio_1024']):.3f}"
        )

    print()
    print("RULE 1 — H4 CONT:")
    print(
        f"  r={int(rule1['r'])}, {rule1['readout']} | "
        f"CV={float(rule1['cv_rmse']):.6f}, "
        f"CV_SD={float(rule1['cv_rmse_std']):.6f}, "
        f"2025={float(rule1['validation_rmse']):.6f} diagnostic-only"
    )

    print()
    print("RULE 3 — H4 CONT:")
    print(
        f"  r={int(rule3['r'])}, {rule3['readout']} | "
        f"CV={float(rule3['cv_rmse']):.6f}, "
        f"shotSD={float(rule3['shot_noise_forecast_sd_proxy_1024']):.3f}, "
        f"Rmax={float(rule3['max_shot_to_train_std_ratio_1024']):.3f}"
    )
    print(f"  hierarchy: {rule3_mode}")

    print()
    print("RULE 4 — H4 CONT:")
    print(
        f"  best CV={float(rule4['best_cv_reference']):.6f}, "
        f"best CV_SD={float(rule4['best_cv_sd_reference']):.6f}, "
        f"1SE threshold={threshold:.6f}"
    )
    print(
        f"  eligible rows={len(eligible)}, r counts={eligible_r_counts}"
    )
    print(
        f"  selected r={int(rule4['r'])}, {rule4['readout']} | "
        f"CV={float(rule4['cv_rmse']):.6f}, "
        f"CZ/feature={int(rule4['feature_vector_cz'])}, "
        f"settings={int(rule4['n_settings'])}, "
        f"features={int(rule4['n_features'])}"
    )

    manifest = {
        "step": "09.05A_H4_CONT_revised_readout_conditioning",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE_FILE),
        "h4_source_rows": int(len(h4)),
        "unique_dynamics_evaluated": int(len(reps)),
        "readouts": list(common.READOUTS),
        "r_values": R_VALUES,
        "shots_proxy": int(common.SHOTS_PROXY),
        "protocol": "CONT",
        "persistent_memory": True,
        "selection_period": "2022-2024",
        "validation_2025_role": "diagnostic_only",
        "test_2026_used": False,
        "rule1": {
            "r": int(rule1["r"]),
            "config_id": str(rule1["source_09_03c_config_id"]),
            "readout": str(rule1["readout"]),
            "cv_rmse": float(rule1["cv_rmse"]),
            "cv_rmse_std": float(rule1["cv_rmse_std"]),
            "validation_rmse_diagnostic": float(
                rule1["validation_rmse"]
            ),
        },
        "rule3": {
            "r": int(rule3["r"]),
            "config_id": str(rule3["source_09_03c_config_id"]),
            "readout": str(rule3["readout"]),
            "cv_rmse": float(rule3["cv_rmse"]),
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
            "r": int(rule4["r"]),
            "config_id": str(rule4["source_09_03c_config_id"]),
            "readout": str(rule4["readout"]),
            "cv_rmse": float(rule4["cv_rmse"]),
            "feature_vector_cz": int(rule4["feature_vector_cz"]),
            "n_settings": int(rule4["n_settings"]),
            "logical_reset_count": int(rule4["logical_reset_count"]),
            "n_features": int(rule4["n_features"]),
        },
    }

    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("Saved:")
    for p in [
        ALL_FILE,
        CV_R_FILE,
        HW_R_FILE,
        RULE1_FILE,
        RULE3_FILE,
        RULE4_ELIGIBLE_FILE,
        RULE4_FILE,
        MANIFEST_FILE,
    ]:
        print(f"  {p}")

    print()
    print(
        "STOP HERE. Send the console output and the result files. "
        "After interpretation, the next step is the final H0-H4 "
        "Rule-1–Rule-4 merge/reselection; do not start Week 10 yet."
    )


if __name__ == "__main__":
    main()
