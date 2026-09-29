"""
Week 12.5J — FINAL cross-platform candidate freeze.

This is the last candidate-selection script before IQM QPU execution.

It does NOT rerun logical optimization, simulation, compilation, or QPU jobs.
It consumes the already-audited Week-12.5I v4.1 Emerald evidence and freezes
the five IQM experimental roles using the SAME five-role logic used for IBM.

Five roles
----------
C1  Accuracy leader
C2  Same-family resource-efficient alternative
C3  Lower-resource, measurement-robust RWP Rule-3 alternative
C4  Cheap CONT Rule-4 feasibility / replay-preflight candidate
C5  Intrinsic-memory Rule-2 candidate

No weighted score.
No 2026 data.
No simulator jobs.
No QPU jobs.

The script also exports:
  * IBM final five
  * IQM final five
  * combined IBM+IQM final-five table
  * complete IBM Rule-1..Rule-4 candidate universe
  * complete IQM 47-candidate universe
  * combined cross-platform candidate universe
  * selection audit + manifest

Expected candidate universes
----------------------------
IBM:
  H0-H3 frozen Week-10 manifest +
  H4 gap-repair additions.

IQM:
  47 unique H0-H6 CONT/RWP candidates from 12.5G0/12.5I.

2026 remains untouched.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path("results")

# ---------------------------------------------------------------------
# Required IQM evidence
# ---------------------------------------------------------------------
IQM_EVIDENCE = RESULTS / "12_05i_v4_1_iqm_emerald_candidate_evidence.csv"
IQM_SOURCE_AUDIT = RESULTS / "12_05i_v4_1_source_audit.csv"
IQM_METRIC_AUDIT = RESULTS / "12_05i_v4_1_metric_completeness_audit.csv"
IQM_UNIFIED = RESULTS / "12_05g0_unified_rule1_rule4_candidates.csv"
IQM_BACKEND_SELECTION = RESULTS / "12_05h_iqm_backend_selection.json"

# ---------------------------------------------------------------------
# IBM frozen candidate sources
# ---------------------------------------------------------------------
IBM_H03 = RESULTS / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"
IBM_H4 = RESULTS / "09_05d_h4_week10_additions.csv"

# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------
OUT_IQM_FINAL = RESULTS / "12_05j_iqm_final_five_candidates.csv"
OUT_IBM_FINAL = RESULTS / "12_05j_ibm_final_five_candidates.csv"
OUT_FINAL_COMBINED = RESULTS / "12_05j_ibm_iqm_final_five_combined.csv"

OUT_IQM_ALL = RESULTS / "12_05j_iqm_all_47_candidates.csv"
OUT_IBM_ALL = RESULTS / "12_05j_ibm_all_rule_candidates.csv"
OUT_ALL_COMBINED = RESULTS / "12_05j_ibm_iqm_all_candidates_combined.csv"

OUT_AUDIT = RESULTS / "12_05j_final_five_selection_audit.csv"
OUT_MANIFEST = RESULTS / "12_05j_manifest.json"


FROZEN_IQM_BACKEND = "emerald"
EXPECTED_IQM_UNIQUE = 47
EXPECTED_IQM_ASSIGNMENTS = 56

# The five IBM Week-11 scientific roles are frozen.
IBM_FINAL_IDS = {
    1: "RWP_H3_R1R3",
    2: "RWP_H3_R4",
    3: "RWP_H2_R3",
    4: "CONT_H0_R4",
    5: "CONT_H2_R2",
}

ROLE_PURPOSE = {
    1: "forecast_accuracy_leader",
    2: "same_family_resource_efficient_alternative",
    3: "lower_resource_measurement_robust_rwp_alternative",
    4: "cheap_cont_replay_feasibility_preflight",
    5: "intrinsic_memory_first_candidate",
}

# Exact values expected from the deterministic IQM role logic.
EXPECTED_IQM_FINAL_IDS = {
    1: "H6_W06_r2_local_04_XZinj_plus_YX45",
    2: "H6_W02_r3_local_05_XZinj_dropX3",
    3: "RWP_H2_R3",
    4: "CONT_H2_R4",
    5: "CONT_H2_R2",
}

# Readout-setting count used only by the Rule-4 resource hierarchy.
READOUT_SETTINGS = {
    "XZ_injection": 2,
    "XZinj_dropX3": 2,
    "XZinj_plus_YX45": 3,
    "XZinj_dropX3_plus_YX45": 3,
    "XYZ_all": 3,
}


def finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if math.isfinite(v) else np.nan


def safe_int(x):
    v = finite(x)
    return int(v) if np.isfinite(v) else None


def rule_set(value):
    text = str(value).upper()
    return set(re.findall(r"R[1-4]", text))


def has_rule(value, rule):
    return rule in rule_set(value)


def pure_rule(value, rule):
    return rule_set(value) == {rule}


def first_existing(row, names, default=np.nan):
    for c in names:
        if c in row.index:
            v = row[c]
            if pd.notna(v):
                return v
    return default


def require_files():
    required = [
        IQM_EVIDENCE,
        IQM_SOURCE_AUDIT,
        IQM_METRIC_AUDIT,
        IQM_UNIFIED,
        IQM_BACKEND_SELECTION,
        IBM_H03,
        IBM_H4,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required frozen input(s):\n  "
            + "\n  ".join(missing)
        )


def verify_iqm_audits():
    source = pd.read_csv(IQM_SOURCE_AUDIT)
    metric = pd.read_csv(IQM_METRIC_AUDIT)

    if source.empty or not source["pass"].astype(bool).all():
        raise RuntimeError(
            "12.5I source/provenance audit is not fully PASS."
        )

    if metric.empty or not metric["pass"].astype(bool).all():
        raise RuntimeError(
            "12.5I rule-metric audit is not fully PASS."
        )

    if len(metric) != EXPECTED_IQM_ASSIGNMENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_IQM_ASSIGNMENTS} IQM rule assignments, "
            f"found {len(metric)}."
        )

    counts = (
        metric.groupby("rule")["pass"]
        .agg(["count", "sum"])
        .to_dict("index")
    )

    for rule in ["R1", "R2", "R3", "R4"]:
        if counts.get(rule, {}).get("count") != 14:
            raise RuntimeError(
                f"{rule}: expected 14 assignments, found "
                f"{counts.get(rule, {}).get('count')}"
            )
        if counts[rule]["sum"] != 14:
            raise RuntimeError(f"{rule}: not all assignments passed.")

    return len(source), len(metric)


def load_iqm():
    evidence = pd.read_csv(IQM_EVIDENCE)
    unified = pd.read_csv(IQM_UNIFIED)

    if evidence["candidate_id"].nunique() != EXPECTED_IQM_UNIQUE:
        raise RuntimeError(
            f"Expected {EXPECTED_IQM_UNIQUE} IQM candidates, "
            f"found {evidence['candidate_id'].nunique()}."
        )

    if unified["candidate_id"].nunique() != EXPECTED_IQM_UNIQUE:
        raise RuntimeError(
            "Unified IQM manifest no longer contains exactly 47 candidates."
        )

    # Keep every original manifest parameter while using the audited evidence
    # columns as authoritative for rule/hardware metrics.
    overlap = [
        c for c in unified.columns
        if c in evidence.columns and c != "candidate_id"
    ]
    unified_extra = unified.drop(columns=overlap)

    merged = evidence.merge(
        unified_extra,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_IQM_UNIQUE:
        raise RuntimeError("IQM evidence/unified merge changed row count.")

    return merged


def add_settings(df):
    out = df.copy()
    out["n_measurement_settings"] = out["readout"].map(READOUT_SETTINGS)

    missing = out.loc[
        out["n_measurement_settings"].isna(),
        ["candidate_id", "readout"],
    ]
    if not missing.empty:
        raise RuntimeError(
            "Unknown readout-setting count:\n"
            + missing.to_string(index=False)
        )

    return out


def select_iqm_final_five(iqm):
    work = add_settings(iqm)
    audit_rows = []

    # ================================================================
    # C1 — accuracy leader
    # ================================================================
    c1_pool = work[
        work["selected_rules"].apply(lambda x: has_rule(x, "R1"))
    ].copy()

    c1_pool = c1_pool.sort_values(
        ["cv_rmse", "cv_rmse_std", "NCZ_feature_vector"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    c1 = c1_pool.iloc[0]

    audit_rows.append({
        "role_id": 1,
        "purpose": ROLE_PURPOSE[1],
        "selection_rule": (
            "minimum chronological training CV among all R1 representatives"
        ),
        "candidate_id": c1["candidate_id"],
        "pool_size": len(c1_pool),
        "primary_value": c1["cv_rmse"],
    })

    # ================================================================
    # C2 — same-family Rule-4 resource alternative
    # ================================================================
    c2_pool = work[
        work["selected_rules"].apply(lambda x: has_rule(x, "R4"))
        & work["topology"].eq(c1["topology"])
        & work["protocol"].eq(c1["protocol"])
        & ~work["candidate_id"].eq(c1["candidate_id"])
    ].copy()

    if c2_pool.empty:
        raise RuntimeError(
            "No same-family R4 alternative exists for IQM Candidate 1."
        )

    c2_pool = c2_pool.sort_values(
        [
            "NCZ_feature_vector",
            "n_measurement_settings",
            "depth_worst_max_setting",
            "cv_rmse",
        ],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    c2 = c2_pool.iloc[0]

    audit_rows.append({
        "role_id": 2,
        "purpose": ROLE_PURPOSE[2],
        "selection_rule": (
            "same topology/protocol as C1; R4; "
            "CZ -> settings -> depth -> CV"
        ),
        "candidate_id": c2["candidate_id"],
        "pool_size": len(c2_pool),
        "primary_value": c2["NCZ_feature_vector"],
    })

    # ================================================================
    # C3 — dedicated lower-resource / measurement-robust RWP Rule-3
    #
    # This reproduces the IBM Candidate-3 QUESTION, not merely the R3
    # accuracy ranking:
    #   * dedicated R3 representative (not co-selected by R1/R4)
    #   * RWP
    #   * distinct topology from C1
    #   * shot-resolvable at 1024: Rmax <= 1
    #   * strictly cheaper than C1 in CZ and depth
    #   * then resource hierarchy, with robustness/CV tie-breaks
    # ================================================================
    c3_pool = work[
        work["selected_rules"].apply(lambda x: pure_rule(x, "R3"))
        & work["protocol"].eq("RWP")
        & ~work["topology"].eq(c1["topology"])
        & (pd.to_numeric(work["Rmax_1024"], errors="coerce") <= 1.0)
        & (
            pd.to_numeric(work["NCZ_feature_vector"], errors="coerce")
            < float(c1["NCZ_feature_vector"])
        )
        & (
            pd.to_numeric(work["depth_worst_max_setting"], errors="coerce")
            < float(c1["depth_worst_max_setting"])
        )
    ].copy()

    if c3_pool.empty:
        raise RuntimeError(
            "No dedicated lower-resource shot-resolvable RWP R3 "
            "candidate remains for Candidate 3."
        )

    c3_pool = c3_pool.sort_values(
        [
            "NCZ_feature_vector",
            "depth_worst_max_setting",
            "Rmax_1024",
            "shotSD_1024",
            "cv_rmse",
        ],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    )

    c3 = c3_pool.iloc[0]

    audit_rows.append({
        "role_id": 3,
        "purpose": ROLE_PURPOSE[3],
        "selection_rule": (
            "pure R3 + RWP + topology != C1 + Rmax<=1 + "
            "CZ<C1 + depth<C1; then CZ -> depth -> Rmax -> shotSD -> CV"
        ),
        "candidate_id": c3["candidate_id"],
        "pool_size": len(c3_pool),
        "primary_value": c3["NCZ_feature_vector"],
    })

    # ================================================================
    # C4 — cheap CONT Rule-4 feasibility / replay-preflight candidate
    #
    # Reuse the frozen Rule-4 resource hierarchy across dedicated CONT
    # representatives. SWAP is already zero for all Emerald candidates.
    # Therefore:
    #     CZ -> measurement settings -> depth -> CV.
    # ================================================================
    c4_pool = work[
        work["selected_rules"].apply(lambda x: pure_rule(x, "R4"))
        & work["protocol"].eq("CONT")
    ].copy()

    if c4_pool.empty:
        raise RuntimeError("No dedicated CONT R4 candidate for Candidate 4.")

    c4_pool = c4_pool.sort_values(
        [
            "NCZ_feature_vector",
            "n_measurement_settings",
            "depth_worst_max_setting",
            "cv_rmse",
        ],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    c4 = c4_pool.iloc[0]

    audit_rows.append({
        "role_id": 4,
        "purpose": ROLE_PURPOSE[4],
        "selection_rule": (
            "pure CONT R4; zero-SWAP frozen; "
            "CZ -> settings -> depth -> CV"
        ),
        "candidate_id": c4["candidate_id"],
        "pool_size": len(c4_pool),
        "primary_value": c4["NCZ_feature_vector"],
    })

    # ================================================================
    # C5 — intrinsic-memory candidate
    #
    # Same scientific objective as IBM Candidate 5:
    # maximize intrinsic MC/ch.
    # CONT candidates must already satisfy Tw(0.01)<=300; this was
    # audited in 12.5I v4.1.
    # ================================================================
    c5_pool = work[
        work["selected_rules"].apply(lambda x: has_rule(x, "R2"))
    ].copy()

    c5_pool = c5_pool.sort_values(
        [
            "MC_ch",
            "NCZ_feature_vector",
            "depth_worst_max_setting",
        ],
        ascending=[False, True, True],
        kind="mergesort",
    )

    c5 = c5_pool.iloc[0]

    audit_rows.append({
        "role_id": 5,
        "purpose": ROLE_PURPOSE[5],
        "selection_rule": (
            "maximum intrinsic MC/ch among R2 representatives; "
            "CONT washout validity already audited"
        ),
        "candidate_id": c5["candidate_id"],
        "pool_size": len(c5_pool),
        "primary_value": c5["MC_ch"],
    })

    selected = pd.DataFrame([c1, c2, c3, c4, c5]).reset_index(drop=True)
    selected.insert(0, "role_id", [1, 2, 3, 4, 5])
    selected.insert(1, "role_purpose", [ROLE_PURPOSE[i] for i in range(1, 6)])

    observed = {
        int(r["role_id"]): str(r["candidate_id"])
        for _, r in selected.iterrows()
    }

    if observed != EXPECTED_IQM_FINAL_IDS:
        raise RuntimeError(
            "IQM final-five logic produced an unexpected identity.\n"
            f"Expected: {EXPECTED_IQM_FINAL_IDS}\n"
            f"Observed: {observed}"
        )

    return selected, pd.DataFrame(audit_rows)


def ibm_candidate_id_column(df):
    for c in ["candidate_key", "candidate_id", "candidate_uid"]:
        if c in df.columns:
            return c
    raise RuntimeError("IBM candidate source has no candidate identifier column.")


def normalize_ibm_row(row, cid, source):
    protocol = str(first_existing(row, ["protocol"], "")).upper()
    topology = str(first_existing(row, ["topology"], ""))
    window = first_existing(row, ["window", "window_size"], np.nan)
    original_r = first_existing(row, ["original_r", "r", "trotter_r"], np.nan)
    readout = first_existing(
        row,
        ["test_readout", "readout", "selection_source_readout"],
        "",
    )

    return {
        "platform": "IBM",
        "candidate_id": cid,
        "selected_rules": str(
            first_existing(row, ["selected_by_rules", "selected_rules"], "")
        ).replace(";", "+"),
        "topology": topology,
        "protocol": protocol,
        "window_effective": 1 if protocol == "CONT" else finite(window),
        "r": finite(original_r),
        "readout": str(readout),
        "alpha": finite(first_existing(row, ["alpha"])),
        "dt": finite(first_existing(row, ["dt"])),
        "hx": finite(first_existing(row, ["hx"])),
        "hy": finite(first_existing(row, ["hy"])),
        "lambda": finite(first_existing(
            row,
            ["selection_source_lambda", "selected_lambda", "lambda"],
        )),
        "J_json": first_existing(row, ["J_json"], ""),
        "cv_rmse": finite(first_existing(
            row,
            ["source_readout_cv_rmse", "cv_rmse", "table_cv_rmse"],
        )),
        "cv_rmse_std": finite(first_existing(
            row,
            ["source_readout_cv_sd", "cv_rmse_std"],
        )),
        "validation_rmse_2025_diagnostic": finite(first_existing(
            row,
            ["source_validation_rmse", "validation_rmse"],
        )),
        "MC_ch": finite(first_existing(
            row,
            ["MC_ch_mean", "MC_ch", "MC_ch_corrected"],
        )),
        "MC_ch_std": finite(first_existing(
            row,
            ["MC_ch_std"],
        )),
        "washout": finite(first_existing(
            row,
            ["Tw_0.01_mean", "Tw_0.01", "washout"],
        )),
        "Rmax_1024": finite(first_existing(
            row,
            ["max_shot_to_train_std_ratio_1024"],
        )),
        "shotSD_1024": finite(first_existing(
            row,
            ["shot_noise_forecast_sd_proxy_1024"],
        )),
        "Smax": finite(first_existing(
            row,
            ["max_abs_raw_prediction_sensitivity"],
        )),
        "source_file": str(source),
    }


def build_ibm_all():
    h03 = pd.read_csv(IBM_H03)
    h4 = pd.read_csv(IBM_H4)

    cid03 = ibm_candidate_id_column(h03)
    cid4 = ibm_candidate_id_column(h4)

    rows = []

    for _, r in h03.iterrows():
        cid = str(r[cid03])
        rows.append(normalize_ibm_row(r, cid, IBM_H03))

    for _, r in h4.iterrows():
        cid = str(r[cid4])
        rows.append(normalize_ibm_row(r, cid, IBM_H4))

    out = pd.DataFrame(rows)

    # H4 additions are separate candidate keys; deduplication should be exact.
    dupes = out[out["candidate_id"].duplicated(keep=False)]
    if not dupes.empty:
        # Allow exact duplicate rows only; otherwise fail.
        for cid, g in dupes.groupby("candidate_id"):
            test = g.drop(columns=["source_file"]).drop_duplicates()
            if len(test) != 1:
                raise RuntimeError(
                    f"IBM candidate {cid} appears with conflicting definitions."
                )
        out = out.drop_duplicates(
            subset=["candidate_id"],
            keep="first",
        ).copy()

    return out.reset_index(drop=True)


def build_ibm_final(ibm_all):
    rows = []

    # Week-11 hardware/deployment interpretation.
    deployment = {
        1: {
            "hardware_r": 3,
            "hardware_shots_per_setting": 2048,
            "deployment_protocol": "RWP",
            "deployment_window": 4,
            "hardware_status": "executed",
        },
        2: {
            "hardware_r": 3,
            "hardware_shots_per_setting": 2048,
            "deployment_protocol": "RWP",
            "deployment_window": 2,
            "hardware_status": "executed",
        },
        3: {
            "hardware_r": 2,
            "hardware_shots_per_setting": 1024,
            "deployment_protocol": "RWP",
            "deployment_window": 2,
            "hardware_status": "executed",
        },
        4: {
            "hardware_r": 4,
            "hardware_shots_per_setting": 512,
            "deployment_protocol": "CONT_to_replay_preflight",
            "deployment_window": np.nan,
            "hardware_status": "feasibility_preflight_not_full_qpu",
        },
        5: {
            "hardware_r": 2,
            "hardware_shots_per_setting": 1024,
            "deployment_protocol": "RWP",
            "deployment_window": 64,
            "hardware_status": "executed_as_RWP64",
        },
    }

    for role_id, cid in IBM_FINAL_IDS.items():
        g = ibm_all[ibm_all["candidate_id"].eq(cid)]
        if len(g) != 1:
            raise RuntimeError(
                f"IBM final candidate {cid}: expected one row, found {len(g)}"
            )

        d = g.iloc[0].to_dict()
        d["role_id"] = role_id
        d["role_purpose"] = ROLE_PURPOSE[role_id]
        d.update(deployment[role_id])
        rows.append(d)

    out = pd.DataFrame(rows)
    cols = ["role_id", "role_purpose"] + [
        c for c in out.columns if c not in {"role_id", "role_purpose"}
    ]
    return out[cols]


def normalize_iqm_all(iqm):
    out = iqm.copy()
    out.insert(0, "platform", "IQM")

    # Keep all columns in the full IQM export, but add normalized common
    # parameter aliases used by the combined table.
    if "lambda" not in out.columns:
        for c in ["selected_lambda", "selection_source_lambda"]:
            if c in out.columns:
                out["lambda"] = out[c]
                break
        else:
            out["lambda"] = np.nan

    if "J_json" not in out.columns:
        # Keep whichever J representation the unified manifest provides.
        jcols = [c for c in out.columns if c.lower() == "j_json"]
        out["J_json"] = out[jcols[0]] if jcols else ""

    return out


def build_iqm_final(selected_iqm):
    out = selected_iqm.copy()
    out.insert(2, "platform", "IQM")

    # Future hardware deployment fields. They are intentionally not used
    # for candidate selection.
    out["hardware_r"] = out["r"]
    out["hardware_shots_per_setting"] = np.nan
    out["deployment_protocol"] = out["protocol"]
    out["deployment_window"] = out["window_effective"]
    out["hardware_status"] = "selected_for_emerald_qpu_handoff"

    # IBM-role-specific deployment interpretation.
    out.loc[
        out["role_id"].eq(4),
        "hardware_status",
    ] = "selected_for_cont_replay_feasibility_preflight"

    # Candidate 5 is the same H2 memory-first logical branch used by IBM.
    # Its measured washout is 57, so the established RWP64 representation
    # remains the directly comparable hardware realization.
    out.loc[out["role_id"].eq(5), "deployment_protocol"] = "RWP"
    out.loc[out["role_id"].eq(5), "deployment_window"] = 64
    out.loc[
        out["role_id"].eq(5),
        "hardware_status",
    ] = "selected_for_RWP64_emerald_handoff"

    return out


def common_final_columns(df):
    preferred = [
        "platform",
        "role_id",
        "role_purpose",
        "candidate_id",
        "selected_rules",
        "topology",
        "protocol",
        "window_effective",
        "r",
        "readout",
        "alpha",
        "dt",
        "hx",
        "hy",
        "lambda",
        "J_json",
        "cv_rmse",
        "cv_rmse_std",
        "validation_rmse_2025_diagnostic",
        "MC_ch",
        "MC_ch_std",
        "washout",
        "Rmax_1024",
        "shotSD_1024",
        "Smax",
        "NCZ_feature_vector",
        "depth_worst_max_setting",
        "worst_min_cz_fidelity",
        "worst_max_asym_ro_error",
        "worst_min_t1_us",
        "worst_min_t2_echo_us",
        "worst_memory_min_t2_echo_us",
        "hardware_r",
        "hardware_shots_per_setting",
        "deployment_protocol",
        "deployment_window",
        "hardware_status",
        "source_file",
    ]

    out = df.copy()
    for c in preferred:
        if c not in out.columns:
            out[c] = np.nan

    return out[preferred]


def common_all_columns(df):
    preferred = [
        "platform",
        "candidate_id",
        "selected_rules",
        "topology",
        "protocol",
        "window_effective",
        "r",
        "readout",
        "alpha",
        "dt",
        "hx",
        "hy",
        "lambda",
        "J_json",
        "cv_rmse",
        "cv_rmse_std",
        "validation_rmse_2025_diagnostic",
        "MC_ch",
        "MC_ch_std",
        "washout",
        "Rmax_1024",
        "shotSD_1024",
        "Smax",
        "NCZ_feature_vector",
        "depth_worst_max_setting",
        "worst_min_cz_fidelity",
        "worst_max_asym_ro_error",
        "worst_min_t1_us",
        "worst_min_t2_echo_us",
        "worst_memory_min_t2_echo_us",
        "source_file",
    ]

    out = df.copy()
    for c in preferred:
        if c not in out.columns:
            out[c] = np.nan
    return out[preferred]


def print_final(title, df):
    print()
    print("=" * 170)
    print(title)
    print("=" * 170)

    cols = [
        "platform",
        "role_id",
        "candidate_id",
        "topology",
        "protocol",
        "window_effective",
        "r",
        "readout",
        "cv_rmse",
        "MC_ch",
        "Rmax_1024",
        "shotSD_1024",
        "NCZ_feature_vector",
        "depth_worst_max_setting",
        "deployment_protocol",
        "deployment_window",
        "hardware_status",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))


def main():
    require_files()
    n_source_checks, n_metric_checks = verify_iqm_audits()

    with open(IQM_BACKEND_SELECTION, "r", encoding="utf-8") as f:
        b = json.load(f)

    if str(b.get("selected_backend", "")).lower() != FROZEN_IQM_BACKEND:
        raise RuntimeError("IQM backend is no longer frozen to Emerald.")

    print("=" * 170)
    print("WEEK 12.5J — FINAL IBM/IQM FIVE-CANDIDATE FREEZE")
    print("=" * 170)
    print("Same five scientific roles as IBM.")
    print("No weighted score. No simulator/QPU jobs. 2026 untouched.")
    print(
        f"Upstream 12.5I audits: "
        f"{n_source_checks} source checks PASS, "
        f"{n_metric_checks}/{EXPECTED_IQM_ASSIGNMENTS} rule assignments PASS."
    )

    # IQM
    iqm = load_iqm()
    iqm_selected, selection_audit = select_iqm_final_five(iqm)
    iqm_final = build_iqm_final(iqm_selected)
    iqm_all = normalize_iqm_all(iqm)

    # IBM
    ibm_all = build_ibm_all()
    ibm_final = build_ibm_final(ibm_all)

    # Cross-platform normalized exports
    iqm_final_common = common_final_columns(iqm_final)
    ibm_final_common = common_final_columns(ibm_final)
    final_combined = pd.concat(
        [ibm_final_common, iqm_final_common],
        ignore_index=True,
    )

    iqm_all_common = common_all_columns(iqm_all)
    ibm_all_common = common_all_columns(ibm_all)
    all_combined = pd.concat(
        [ibm_all_common, iqm_all_common],
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    iqm_final.to_csv(OUT_IQM_FINAL, index=False)
    ibm_final.to_csv(OUT_IBM_FINAL, index=False)
    final_combined.to_csv(OUT_FINAL_COMBINED, index=False)

    iqm_all.to_csv(OUT_IQM_ALL, index=False)
    ibm_all.to_csv(OUT_IBM_ALL, index=False)
    all_combined.to_csv(OUT_ALL_COMBINED, index=False)

    selection_audit.to_csv(OUT_AUDIT, index=False)

    # -----------------------------------------------------------------
    # Final hard audits
    # -----------------------------------------------------------------
    if len(iqm_final) != 5 or iqm_final["candidate_id"].nunique() != 5:
        raise RuntimeError("IQM final five are not five unique candidates.")

    if len(ibm_final) != 5 or ibm_final["candidate_id"].nunique() != 5:
        raise RuntimeError("IBM final five are not five unique candidates.")

    if len(final_combined) != 10:
        raise RuntimeError("Combined final table must contain exactly 10 rows.")

    if len(iqm_all) != EXPECTED_IQM_UNIQUE:
        raise RuntimeError("IQM all-candidate export must contain 47 rows.")

    observed_iqm = dict(
        zip(
            iqm_final["role_id"].astype(int),
            iqm_final["candidate_id"].astype(str),
        )
    )
    observed_ibm = dict(
        zip(
            ibm_final["role_id"].astype(int),
            ibm_final["candidate_id"].astype(str),
        )
    )

    if observed_iqm != EXPECTED_IQM_FINAL_IDS:
        raise RuntimeError("Final IQM identity audit failed.")

    if observed_ibm != IBM_FINAL_IDS:
        raise RuntimeError("Final IBM identity audit failed.")

    manifest = {
        "stage": "12.5J_final_cross_platform_candidate_freeze",
        "iqm_backend": FROZEN_IQM_BACKEND,
        "selection_roles_identical_to_ibm": True,
        "weighted_score": False,
        "test_2026_used": False,
        "qpu_jobs": 0,
        "simulator_jobs": 0,
        "iqm_source_audit_checks_passed": int(n_source_checks),
        "iqm_rule_assignments_passed": int(n_metric_checks),
        "iqm_final_five": {
            str(k): v for k, v in observed_iqm.items()
        },
        "ibm_final_five": {
            str(k): v for k, v in observed_ibm.items()
        },
        "candidate_universes": {
            "IBM_rows": int(len(ibm_all)),
            "IQM_rows": int(len(iqm_all)),
            "combined_rows": int(len(all_combined)),
        },
        "outputs": {
            "iqm_final_five": str(OUT_IQM_FINAL),
            "ibm_final_five": str(OUT_IBM_FINAL),
            "combined_final_five": str(OUT_FINAL_COMBINED),
            "iqm_all_candidates": str(OUT_IQM_ALL),
            "ibm_all_candidates": str(OUT_IBM_ALL),
            "combined_all_candidates": str(OUT_ALL_COMBINED),
            "selection_audit": str(OUT_AUDIT),
        },
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print_final("FINAL IBM FIVE", ibm_final_common)
    print_final("FINAL IQM FIVE — EMERALD", iqm_final_common)

    print()
    print("=" * 170)
    print("12.5J COMPLETE — FINAL CROSS-PLATFORM CANDIDATES FROZEN")
    print("=" * 170)
    print(f"IBM final five:                {OUT_IBM_FINAL}")
    print(f"IQM final five:                {OUT_IQM_FINAL}")
    print(f"Combined final five:           {OUT_FINAL_COMBINED}")
    print(f"IBM all Rule candidates:       {OUT_IBM_ALL}")
    print(f"IQM all 47 candidates:         {OUT_IQM_ALL}")
    print(f"Combined all candidates:       {OUT_ALL_COMBINED}")
    print(f"Selection audit:               {OUT_AUDIT}")
    print(f"Manifest:                      {OUT_MANIFEST}")
    print()
    print("2026 remains untouched.")
    print("STOP HERE. Review the final-five CSV before any Emerald QPU submission.")


if __name__ == "__main__":
    main()
