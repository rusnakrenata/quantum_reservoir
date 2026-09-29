"""
WEEK 9.05D — FINAL H0-H4 RULE-1–RULE-4 MERGE / WEEK-10 HANDOFF

PURPOSE
-------
Close Week 9 by integrating H4 naturally into the already-completed H0-H3
multi-rule candidate universe.

This script DOES NOT rerun H0-H3 dynamics and DOES NOT submit QPU jobs.

The exact historical H0-H3 Week-10 manifest is treated as frozen baseline:
    results/10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv

That baseline must contain exactly the 25 unique H0-H3 Rule-1–Rule-4
reservoirs that were evaluated in Week 10.

H4 rule representatives are assembled from the completed H4 evidence:

CONT
----
Rule 1:
    results/09_05a_h4_cont_rule1_selected.csv
Rule 2:
    reconstructed from results/09_03c_memory_summary.csv
    using the original CONT rule:
        maximize MC/ch subject to Tw(0.01) <= 300
    The selected memory dynamics is then paired with the best revised
    training-CV readout for that exact dynamics from:
        results/09_05a_h4_cont_conditioning_all.csv
Rule 3:
    results/09_05a_h4_cont_rule3_selected.csv
Rule 4:
    results/09_05a_h4_cont_rule4_selected.csv

RWP
---
Rule 1:
    global minimum-CV H4 row from:
        results/09_04c2_h4_backfill_best_cv_per_window.csv
Rule 2:
    dynamics from:
        results/09_05b_h4_rwp_memory_rule2_winner.csv
    but the downstream test readout is XYZ_all, matching the original
    H0-H3 Rule-2 Week-10 convention. The exact XYZ_all forecast row for the
    same H4 dynamics is recovered from:
        results/09_04c2_h4_backfill_all_trials.csv
Rule 3:
    results/09_05c_h4_rule3_selected.csv
Rule 4:
    results/09_05c_h4_rule4_selected_before_compile.csv

IMPORTANT SCIENTIFIC POINT
--------------------------
The original Week-10 experiment tested ALL UNIQUE per-topology/protocol
Rule-1–Rule-4 representatives, not only candidates that globally beat H3.
Therefore H4's unique Rule-1–Rule-4 representatives are the additional
Week-10 candidate set. They are not required to beat H3 before receiving
the same backend-calibrated noisy-simulation treatment.

Candidate identity includes:
    protocol, topology, window, original selected r, test readout,
    alpha, dt, hx, hy, and all H4 couplings.

If multiple rules select the same exact H4 reservoir, it is evaluated once
and selected_by_rules is merged.

DATA DISCIPLINE
---------------
- 2022-2024 selection evidence only.
- 2025 values carried as diagnostic metadata.
- 2026 is never loaded.

OUTPUTS
-------
results/09_05d_h4_rule_rows.csv
results/09_05d_h4_unique_candidates.csv
results/09_05d_h4_week10_additions.csv
results/09_05d_h0_h4_candidate_manifest.csv
results/09_05d_h0_h4_rule_map.csv
results/09_05d_manifest.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"

BASELINE_MANIFEST = (
    RESULTS / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"
)

H4_CONT_ALL = RESULTS / "09_05a_h4_cont_conditioning_all.csv"
H4_CONT_R1 = RESULTS / "09_05a_h4_cont_rule1_selected.csv"
H4_CONT_R3 = RESULTS / "09_05a_h4_cont_rule3_selected.csv"
H4_CONT_R4 = RESULTS / "09_05a_h4_cont_rule4_selected.csv"
CONT_MEMORY = RESULTS / "09_03c_memory_summary.csv"

H4_RWP_C2_ALL = RESULTS / "09_04c2_h4_backfill_all_trials.csv"
H4_RWP_C2_PER_W = RESULTS / "09_04c2_h4_backfill_best_cv_per_window.csv"
H4_RWP_R2 = RESULTS / "09_05b_h4_rwp_memory_rule2_winner.csv"
H4_RWP_R3 = RESULTS / "09_05c_h4_rule3_selected.csv"
H4_RWP_R4 = RESULTS / "09_05c_h4_rule4_selected_before_compile.csv"

H4_RULE_ROWS_FILE = RESULTS / "09_05d_h4_rule_rows.csv"
H4_UNIQUE_FILE = RESULTS / "09_05d_h4_unique_candidates.csv"
H4_WEEK10_FILE = RESULTS / "09_05d_h4_week10_additions.csv"
COMBINED_FILE = RESULTS / "09_05d_h0_h4_candidate_manifest.csv"
RULE_MAP_FILE = RESULTS / "09_05d_h0_h4_rule_map.csv"
MANIFEST_FILE = RESULTS / "09_05d_manifest.json"

EXPECTED_BASELINE_COUNT = 25
EXPECTED_BASELINE_TOPOLOGIES = {"H0", "H1", "H2", "H3"}
EXPECTED_PROTOCOLS = {"CONT", "RWP"}
RULES = ["R1", "R2", "R3", "R4"]


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_0905d_common")
H4_EDGES = list(common.TOPOLOGY_EDGES["H4"])


def require(paths):
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required result files:\n  "
            + "\n  ".join(str(p) for p in missing)
        )


def one_row(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise RuntimeError(f"{path}: expected exactly 1 row, found {len(df)}.")
    return df.iloc[0].copy()


def first_present(row, names, default=np.nan):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def normalized_window(x):
    if x is None:
        return np.nan
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass
    return int(x)


def row_r(row):
    return int(first_present(row, ["r", "trotter_r", "original_r"]))


def row_search_id(row):
    x = first_present(
        row,
        ["c2_search_id", "trial_search_id", "search_id"],
        "",
    )
    return "" if pd.isna(x) else str(x)


def row_source_config(row):
    x = first_present(
        row,
        [
            "source_09_03c_config_id",
            "source_config_id",
            "config_id",
        ],
        "",
    )
    return "" if pd.isna(x) else str(x)


def row_source_candidate(row):
    x = first_present(
        row,
        [
            "source_09_03c_seed_candidate_id",
            "source_candidate_id",
            "seed_candidate_id",
            "candidate_id",
        ],
        "",
    )
    return "" if pd.isna(x) else str(x)


def J_dict_from_row(row):
    out = {}
    for i, j in H4_EDGES:
        key = f"J{i}{j}"
        rev = f"J{j}{i}"
        if key in row.index and pd.notna(row[key]):
            val = row[key]
        elif rev in row.index and pd.notna(row[rev]):
            val = row[rev]
        else:
            raise KeyError(
                f"Missing H4 coupling {key}/{rev} in source row."
            )
        out[key] = float(val)
    return out


def json_J(row):
    return json.dumps(
        J_dict_from_row(row),
        separators=(",", ":"),
        sort_keys=True,
    )


def bool_or_nan(x):
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes"}


def canonical_from_source(
    row: pd.Series,
    *,
    protocol: str,
    rule: str,
    test_readout: str | None = None,
    window=None,
    provenance: str,
    memory_meta: dict | None = None,
) -> dict:
    readout = (
        str(test_readout)
        if test_readout is not None
        else str(first_present(row, ["readout", "source_readout"]))
    )

    r = row_r(row)
    w = normalized_window(
        window
        if window is not None
        else first_present(row, ["window"], np.nan)
    )

    candidate = {
        "candidate_key": "",
        "selected_by_rules": rule,
        "protocol": protocol,
        "topology": "H4",
        "window": w,
        "original_r": r,
        "test_readout": readout,
        "alpha": float(first_present(row, ["alpha"], common.DEFAULT_ALPHA)),
        "dt": float(first_present(row, ["dt"], common.DEFAULT_DT)),
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
        "J_json": json_J(row),
        "selection_source_lambda": float(
            first_present(
                row,
                ["selected_lambda", "source_selected_lambda"],
                np.nan,
            )
        ),
        "source_readout_cv_rmse": float(
            first_present(
                row,
                [
                    "cv_rmse",
                    "source_cv_rmse_for_tiebreak",
                    "source_cv_rmse",
                ],
                np.nan,
            )
        ),
        "source_readout_cv_sd": float(
            first_present(
                row,
                ["cv_rmse_std", "source_cv_rmse_std"],
                np.nan,
            )
        ),
        "source_validation_rmse": float(
            first_present(
                row,
                [
                    "validation_rmse",
                    "source_validation_rmse_diagnostic",
                    "source_validation_rmse",
                ],
                np.nan,
            )
        ),
        "source_validation_mae": float(
            first_present(row, ["validation_mae"], np.nan)
        ),
        "source_validation_bias": float(
            first_present(row, ["validation_bias"], np.nan)
        ),
        "source_config_id": row_source_config(row),
        "source_candidate_id": row_source_candidate(row),
        "trial_search_id": row_search_id(row),
        "dynamics_provenance": provenance,
        "lambda_provenance": provenance,
        "source_cv_equals_test_readout_cv": True,
        "2025_used_for_selection": False,
        "2026_used": False,
        "shot_noise_forecast_sd_proxy_1024": float(
            first_present(
                row,
                ["shot_noise_forecast_sd_proxy_1024"],
                np.nan,
            )
        ),
        "max_shot_to_train_std_ratio_1024": float(
            first_present(
                row,
                ["max_shot_to_train_std_ratio_1024"],
                np.nan,
            )
        ),
        "all_features_shot_resolvable_1024": bool_or_nan(
            first_present(
                row,
                ["all_features_shot_resolvable_1024"],
                np.nan,
            )
        ),
        "logical_feature_vector_cz": float(
            first_present(row, ["feature_vector_cz"], np.nan)
        ),
        "logical_n_settings": float(
            first_present(row, ["n_settings"], np.nan)
        ),
        "logical_n_features": float(
            first_present(row, ["n_features"], np.nan)
        ),
        "logical_reset_count": float(
            first_present(row, ["logical_reset_count"], np.nan)
        ),
    }

    for i, j in H4_EDGES:
        key = f"J{i}{j}"
        candidate[key] = J_dict_from_row(row)[key]

    if memory_meta:
        candidate.update(memory_meta)

    return candidate


def fingerprint(row: pd.Series | dict) -> tuple:
    if isinstance(row, dict):
        row = pd.Series(row)

    def rounded(name):
        return round(float(row[name]), 12)

    w = normalized_window(row.get("window", np.nan))
    w_key = None if pd.isna(w) else int(w)

    return (
        str(row["protocol"]).upper(),
        str(row["topology"]),
        w_key,
        int(row["original_r"]),
        str(row["test_readout"]),
        rounded("alpha"),
        rounded("dt"),
        rounded("hx"),
        rounded("hy"),
        tuple(
            round(float(row[f"J{i}{j}"]), 12)
            for i, j in H4_EDGES
        ),
    )


def parse_rules(value) -> list[str]:
    return sorted(
        set(re.findall(r"R[1-4]", str(value))),
        key=lambda x: int(x[1:]),
    )


def merge_h4_rule_rows(rule_rows: pd.DataFrame) -> pd.DataFrame:
    groups = {}

    for _, row in rule_rows.iterrows():
        fp = fingerprint(row)
        groups.setdefault(fp, []).append(row)

    merged = []

    for fp, rows in groups.items():
        g = pd.DataFrame(rows)
        rep = (
            g.sort_values(
                [
                    "source_readout_cv_rmse",
                    "source_readout_cv_sd",
                ],
                na_position="last",
            )
            .iloc[0]
            .to_dict()
        )

        rules = sorted(
            {
                rr
                for value in g["selected_by_rules"]
                for rr in parse_rules(value)
            },
            key=lambda x: int(x[1:]),
        )

        rep["selected_by_rules"] = "/".join(rules)
        merged.append(rep)

    merged_df = pd.DataFrame(merged)

    # Stable candidate keys.
    used = set()
    keys = []
    for _, row in merged_df.sort_values(
        ["protocol", "selected_by_rules", "source_readout_cv_rmse"]
    ).iterrows():
        base = (
            f"{str(row['protocol']).upper()}_H4_"
            + "_".join(parse_rules(row["selected_by_rules"]))
        )
        key = base
        serial = 2
        while key in used:
            key = f"{base}_{serial}"
            serial += 1
        used.add(key)
        keys.append((row.name, key))

    key_map = dict(keys)
    merged_df["candidate_key"] = [
        key_map[i] for i in merged_df.index
    ]

    return merged_df.sort_values(
        ["protocol", "candidate_key"]
    ).reset_index(drop=True)


def build_baseline_rule_assignments(baseline: pd.DataFrame):
    assignments = {}
    for _, row in baseline.iterrows():
        protocol = str(row["protocol"]).upper()
        topology = str(row["topology"])
        for rule in parse_rules(row["selected_by_rules"]):
            key = (protocol, topology, rule)
            if key in assignments:
                raise RuntimeError(
                    f"Baseline manifest maps {key} to more than one candidate."
                )
            assignments[key] = str(row["candidate_key"])
    return assignments


def build_rule_map(
    baseline: pd.DataFrame,
    h4_rules: pd.DataFrame,
    h4_unique: pd.DataFrame,
):
    assignments = build_baseline_rule_assignments(baseline)

    h4_key_by_fp = {
        fingerprint(row): str(row["candidate_key"])
        for _, row in h4_unique.iterrows()
    }

    for _, row in h4_rules.iterrows():
        key = (
            str(row["protocol"]).upper(),
            "H4",
            str(row["selected_by_rules"]),
        )
        assignments[key] = h4_key_by_fp[fingerprint(row)]

    rows = []
    for protocol in ["CONT", "RWP"]:
        for topology in ["H0", "H1", "H2", "H3", "H4"]:
            out = {
                "protocol": protocol,
                "topology": topology,
            }
            for rule in RULES:
                k = (protocol, topology, rule)
                if k not in assignments:
                    raise RuntimeError(
                        f"Missing final rule assignment for {k}."
                    )
                out[rule] = assignments[k]
            rows.append(out)

    return pd.DataFrame(rows)


def make_combined_manifest(
    baseline: pd.DataFrame,
    h4_unique: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve every historical H0-H3 column and append H4 columns by union.
    No historical values are altered.
    """
    all_cols = list(baseline.columns)
    for c in h4_unique.columns:
        if c not in all_cols:
            all_cols.append(c)

    b = baseline.reindex(columns=all_cols)
    h = h4_unique.reindex(columns=all_cols)

    out = pd.concat([b, h], ignore_index=True)

    if out["candidate_key"].astype(str).duplicated().any():
        dup = out.loc[
            out["candidate_key"].astype(str).duplicated(keep=False),
            "candidate_key",
        ].tolist()
        raise RuntimeError(f"Duplicate candidate keys in combined manifest: {dup}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    require(
        [
            COMMON_FILE,
            BASELINE_MANIFEST,
            H4_CONT_ALL,
            H4_CONT_R1,
            H4_CONT_R3,
            H4_CONT_R4,
            CONT_MEMORY,
            H4_RWP_C2_ALL,
            H4_RWP_C2_PER_W,
            H4_RWP_R2,
            H4_RWP_R3,
            H4_RWP_R4,
        ]
    )

    outputs = [
        H4_RULE_ROWS_FILE,
        H4_UNIQUE_FILE,
        H4_WEEK10_FILE,
        COMBINED_FILE,
        RULE_MAP_FILE,
        MANIFEST_FILE,
    ]

    if args.overwrite:
        for p in outputs:
            if p.exists():
                p.unlink()

    baseline = pd.read_csv(BASELINE_MANIFEST)

    required_base_cols = [
        "candidate_key",
        "selected_by_rules",
        "protocol",
        "topology",
    ]
    missing = [c for c in required_base_cols if c not in baseline.columns]
    if missing:
        raise KeyError(
            f"Baseline manifest missing columns: {missing}"
        )

    if len(baseline) != EXPECTED_BASELINE_COUNT:
        raise RuntimeError(
            f"Expected frozen H0-H3 baseline manifest to contain "
            f"{EXPECTED_BASELINE_COUNT} rows; found {len(baseline)}."
        )

    base_topologies = set(baseline["topology"].astype(str))
    if base_topologies != EXPECTED_BASELINE_TOPOLOGIES:
        raise RuntimeError(
            f"Baseline topology set mismatch: {base_topologies}"
        )

    base_protocols = set(baseline["protocol"].astype(str).str.upper())
    if base_protocols != EXPECTED_PROTOCOLS:
        raise RuntimeError(
            f"Baseline protocol set mismatch: {base_protocols}"
        )

    baseline_assign = build_baseline_rule_assignments(baseline)
    expected_assign = {
        (p, h, r)
        for p in EXPECTED_PROTOCOLS
        for h in EXPECTED_BASELINE_TOPOLOGIES
        for r in RULES
    }
    if set(baseline_assign) != expected_assign:
        missing_assign = sorted(expected_assign - set(baseline_assign))
        extra_assign = sorted(set(baseline_assign) - expected_assign)
        raise RuntimeError(
            "Frozen H0-H3 rule coverage is not exactly 2×4×4.\n"
            f"Missing: {missing_assign}\nExtra: {extra_assign}"
        )

    print("=" * 136)
    print("WEEK 9.05D — FINAL H0-H4 RULE-1–RULE-4 MERGE")
    print("=" * 136)
    print(
        f"Frozen H0-H3 Week-10 baseline: {len(baseline)} unique candidates, "
        f"{len(baseline_assign)} rule assignments."
    )
    print("H0-H3 rows are preserved exactly; no dynamics are rerun.")
    print()

    h4_rule_rows = []

    # ------------------------------------------------------------------
    # H4 CONT RULE 1 / 3 / 4
    # ------------------------------------------------------------------
    cont_r1 = one_row(H4_CONT_R1)
    cont_r3 = one_row(H4_CONT_R3)
    cont_r4 = one_row(H4_CONT_R4)

    h4_rule_rows.append(
        canonical_from_source(
            cont_r1,
            protocol="CONT",
            rule="R1",
            provenance="09_05a_h4_cont_rule1_selected.csv",
        )
    )
    h4_rule_rows.append(
        canonical_from_source(
            cont_r3,
            protocol="CONT",
            rule="R3",
            provenance="09_05a_h4_cont_rule3_selected.csv",
        )
    )
    h4_rule_rows.append(
        canonical_from_source(
            cont_r4,
            protocol="CONT",
            rule="R4",
            provenance="09_05a_h4_cont_rule4_selected.csv",
        )
    )

    # ------------------------------------------------------------------
    # H4 CONT RULE 2 — same original washout-valid memory criterion
    # ------------------------------------------------------------------
    mem = pd.read_csv(CONT_MEMORY)
    mem_h4 = mem[mem["topology"].astype(str) == "H4"].copy()

    if len(mem_h4) == 0:
        raise RuntimeError("No H4 rows found in 09_03c_memory_summary.csv.")

    mem_h4 = mem_h4[
        pd.to_numeric(mem_h4["MC_ch_mean"], errors="coerce").notna()
        & (pd.to_numeric(mem_h4["Tw_0.01_mean"], errors="coerce") <= 300.0)
    ].copy()

    if "n_valid_MC" in mem_h4.columns and "n_probe_seeds" in mem_h4.columns:
        mem_h4 = mem_h4[
            pd.to_numeric(mem_h4["n_valid_MC"], errors="coerce")
            == pd.to_numeric(mem_h4["n_probe_seeds"], errors="coerce")
        ].copy()

    if len(mem_h4) == 0:
        raise RuntimeError(
            "No washout-valid H4 CONT memory candidate survives Tw<=300."
        )

    mem_winner = (
        mem_h4.sort_values(
            ["MC_ch_mean", "Tw_0.01_mean"],
            ascending=[False, True],
        )
        .iloc[0]
        .copy()
    )

    cont_all = pd.read_csv(H4_CONT_ALL)
    same_dyn = cont_all[
        cont_all["source_09_03c_config_id"].astype(str)
        == str(mem_winner["config_id"])
    ].copy()

    if len(same_dyn) == 0:
        raise RuntimeError(
            "Could not map H4 CONT Rule-2 memory dynamics into the "
            "revised 09.05A readout table."
        )

    # Pair memory-optimal dynamics with its strongest revised predictive
    # readout, using training CV only.
    cont_r2 = (
        same_dyn.sort_values(
            [
                "cv_rmse",
                "n_settings",
                "n_features",
                "selected_lambda",
            ]
        )
        .iloc[0]
        .copy()
    )

    cont_r2_memory_meta = {
        "MC_ch_mean": float(mem_winner["MC_ch_mean"]),
        "MC_ch_std": float(
            mem_winner["MC_ch_std"]
            if "MC_ch_std" in mem_winner.index
            else np.nan
        ),
        "MC_total_mean": float(
            mem_winner["MC_total_mean"]
            if "MC_total_mean" in mem_winner.index
            else np.nan
        ),
        "Tw_0.01_mean": float(mem_winner["Tw_0.01_mean"]),
        "Tw_0.01_std": float(
            mem_winner["Tw_0.01_std"]
            if "Tw_0.01_std" in mem_winner.index
            else np.nan
        ),
    }

    h4_rule_rows.append(
        canonical_from_source(
            cont_r2,
            protocol="CONT",
            rule="R2",
            provenance=(
                "09_03c_memory_summary.csv washout-valid memory winner + "
                "09_05a_h4_cont_conditioning_all.csv best revised readout "
                "for identical dynamics"
            ),
            memory_meta=cont_r2_memory_meta,
        )
    )

    # ------------------------------------------------------------------
    # H4 RWP RULE 1 — global C2 training-CV winner
    # ------------------------------------------------------------------
    rwp_per_w = pd.read_csv(H4_RWP_C2_PER_W)
    rwp_r1 = (
        rwp_per_w.sort_values(
            [
                "cv_rmse",
                "cv_rmse_std",
                "shot_noise_forecast_sd_proxy_1024",
            ]
        )
        .iloc[0]
        .copy()
    )

    h4_rule_rows.append(
        canonical_from_source(
            rwp_r1,
            protocol="RWP",
            rule="R1",
            provenance=(
                "09_04c2_h4_backfill_best_cv_per_window.csv "
                "global training-CV winner"
            ),
        )
    )

    # ------------------------------------------------------------------
    # H4 RWP RULE 2 — memory winner dynamics + XYZ_all downstream readout
    # ------------------------------------------------------------------
    rwp_r2_mem = one_row(H4_RWP_R2)

    W2 = int(rwp_r2_mem["window"])
    r2 = int(rwp_r2_mem["r"])
    sid2 = str(rwp_r2_mem["c2_search_id"])

    c2_all = pd.read_csv(H4_RWP_C2_ALL)
    same_r2 = c2_all[
        (pd.to_numeric(c2_all["window"]).astype(int) == W2)
        & (pd.to_numeric(c2_all["r"]).astype(int) == r2)
        & (c2_all["c2_search_id"].astype(str) == sid2)
        & (c2_all["readout"].astype(str) == "XYZ_all")
    ].copy()

    if len(same_r2) != 1:
        raise RuntimeError(
            f"Expected exactly one H4 RWP Rule-2 XYZ_all forecast row for "
            f"W={W2}, r={r2}, search={sid2}; found {len(same_r2)}."
        )

    rwp_r2 = same_r2.iloc[0].copy()

    rwp_r2_memory_meta = {
        "MC_ch_mean": float(rwp_r2_mem["MC_ch_mean"]),
        "MC_ch_std": float(rwp_r2_mem["MC_ch_std"]),
        "MC_total_mean": float(rwp_r2_mem["MC_total_mean"]),
        "Tw_0.01_mean": np.nan,
        "Tw_0.01_std": np.nan,
    }

    h4_rule_rows.append(
        canonical_from_source(
            rwp_r2,
            protocol="RWP",
            rule="R2",
            test_readout="XYZ_all",
            provenance=(
                "09_05b_h4_rwp_memory_rule2_winner.csv dynamics + "
                "matching XYZ_all row from 09_04c2_h4_backfill_all_trials.csv"
            ),
            memory_meta=rwp_r2_memory_meta,
        )
    )

    # ------------------------------------------------------------------
    # H4 RWP RULE 3 / RULE 4
    # ------------------------------------------------------------------
    rwp_r3 = one_row(H4_RWP_R3)
    rwp_r4 = one_row(H4_RWP_R4)

    h4_rule_rows.append(
        canonical_from_source(
            rwp_r3,
            protocol="RWP",
            rule="R3",
            provenance="09_05c_h4_rule3_selected.csv",
        )
    )
    h4_rule_rows.append(
        canonical_from_source(
            rwp_r4,
            protocol="RWP",
            rule="R4",
            provenance="09_05c_h4_rule4_selected_before_compile.csv",
        )
    )

    rule_df = pd.DataFrame(h4_rule_rows)

    if len(rule_df) != 8:
        raise RuntimeError(f"Expected 8 H4 rule rows, found {len(rule_df)}.")

    rule_df.to_csv(H4_RULE_ROWS_FILE, index=False)

    # ------------------------------------------------------------------
    # Deduplicate H4 exact reservoirs; merge rule labels.
    # ------------------------------------------------------------------
    h4_unique = merge_h4_rule_rows(rule_df)

    # Map final candidate key back onto each rule row.
    fp_to_key = {
        fingerprint(row): str(row["candidate_key"])
        for _, row in h4_unique.iterrows()
    }
    rule_df["candidate_key"] = [
        fp_to_key[fingerprint(row)]
        for _, row in rule_df.iterrows()
    ]
    rule_df.to_csv(H4_RULE_ROWS_FILE, index=False)

    h4_unique.to_csv(H4_UNIQUE_FILE, index=False)
    h4_unique.to_csv(H4_WEEK10_FILE, index=False)

    # ------------------------------------------------------------------
    # Final H0-H4 rule map + combined candidate manifest.
    # ------------------------------------------------------------------
    rule_map = build_rule_map(
        baseline,
        rule_df,
        h4_unique,
    )
    rule_map.to_csv(RULE_MAP_FILE, index=False)

    combined = make_combined_manifest(
        baseline,
        h4_unique,
    )
    combined.to_csv(COMBINED_FILE, index=False)

    # Audits.
    if len(combined) != len(baseline) + len(h4_unique):
        raise RuntimeError("Combined candidate count audit failed.")

    if set(combined["topology"].astype(str)) != {
        "H0", "H1", "H2", "H3", "H4"
    }:
        raise RuntimeError("Final topology coverage is not H0-H4.")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("H4 FINAL RULE ASSIGNMENTS")
    print("-" * 136)
    show_cols = [
        "protocol",
        "selected_by_rules",
        "candidate_key",
        "window",
        "original_r",
        "test_readout",
        "source_readout_cv_rmse",
        "source_validation_rmse",
        "MC_ch_mean",
        "Tw_0.01_mean",
        "shot_noise_forecast_sd_proxy_1024",
        "max_shot_to_train_std_ratio_1024",
        "logical_feature_vector_cz",
    ]
    for col in show_cols:
        if col not in rule_df.columns:
            rule_df[col] = np.nan

    print(
        rule_df[show_cols]
        .sort_values(["protocol", "selected_by_rules"])
        .to_string(index=False)
    )

    print()
    print("H4 UNIQUE CANDIDATES TO ADD TO WEEK 10")
    print("-" * 136)
    show_u = [
        "candidate_key",
        "protocol",
        "selected_by_rules",
        "window",
        "original_r",
        "test_readout",
        "source_readout_cv_rmse",
        "source_validation_rmse",
    ]
    print(
        h4_unique[show_u]
        .sort_values(["protocol", "candidate_key"])
        .to_string(index=False)
    )

    print()
    print("=" * 136)
    print("FINAL H0-H4 MERGE SUMMARY")
    print("=" * 136)
    print(f"Frozen historical H0-H3 unique candidates: {len(baseline)}")
    print(f"New unique H4 candidates:                  {len(h4_unique)}")
    print(f"Expanded H0-H4 unique candidate universe:  {len(combined)}")
    print("Final rule assignments:                    40 (= 2 protocols × 5 topologies × 4 rules)")
    print()
    print(
        f"H4 CONT Rule 2: r={int(cont_r2['r'])}, "
        f"{cont_r2['readout']} | "
        f"MC/ch={float(mem_winner['MC_ch_mean']):.6f}, "
        f"Tw={float(mem_winner['Tw_0.01_mean']):.1f}, "
        f"CV={float(cont_r2['cv_rmse']):.6f}"
    )
    print(
        f"H4 RWP Rule 2: W={W2}, r={r2}, XYZ_all | "
        f"MC/ch={float(rwp_r2_mem['MC_ch_mean']):.6f}, "
        f"CV={float(rwp_r2['cv_rmse']):.6f}"
    )
    print()
    print("No Week-10 simulations were run by this script.")
    print("2026 was not loaded.")

    manifest = {
        "step": "09.05D_H0_H4_rule_merge",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_manifest": str(BASELINE_MANIFEST),
        "baseline_h0_h3_unique_candidates": int(len(baseline)),
        "baseline_h0_h3_rule_assignments": int(len(baseline_assign)),
        "h4_rule_assignments": int(len(rule_df)),
        "h4_unique_candidates": int(len(h4_unique)),
        "expanded_h0_h4_unique_candidates": int(len(combined)),
        "final_rule_assignments": 40,
        "h4_candidate_keys": h4_unique["candidate_key"].astype(str).tolist(),
        "week10_action": (
            "Run backend-calibrated noisy simulation ONLY for the new unique "
            "H4 candidates; do not rerun unchanged H0-H3 candidates."
        ),
        "validation_2025_role": "diagnostic_only",
        "test_2026_used": False,
        "weighted_score": False,
    }

    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("Saved:")
    for p in [
        H4_RULE_ROWS_FILE,
        H4_UNIQUE_FILE,
        H4_WEEK10_FILE,
        COMBINED_FILE,
        RULE_MAP_FILE,
        MANIFEST_FILE,
    ]:
        print(f"  {p}")

    print()
    print(
        "STOP HERE. Send the console output plus the six files above. "
        "After interpretation we will know the exact number of H4 candidates "
        "that need Week-10 noisy simulation. Do not run H4 Week 10 yet."
    )


if __name__ == "__main__":
    main()
