"""
Week 12.5I v4.1 — FINAL STRICT IBM-style IQM finalist evidence.

Purpose
-------
Build the final Emerald five-role candidate evidence WITHOUT any heuristic
metric recovery.

Hard rules
----------
1. NO directory-wide CSV scan.
2. NO fuzzy/structural matching.
3. NO "pick the lowest-CV duplicate row".
4. Every metric is loaded from one declared authoritative source or from a
   frozen historical IBM Rule-1..Rule-4 constant table.
5. Exact candidate identity is audited before use.
6. 2026 is never loaded or used.
7. No weighted score.
8. No QPU jobs / no simulator jobs.

Required frozen IQM files
-------------------------
results/12_05g0_unified_rule1_rule4_candidates.csv
results/12_05g1_iqm_candidate_layout_summary.csv
results/12_05h_iqm_backend_selection.json

Required authoritative rule-source files
----------------------------------------
results/09_05a_cont_revised_hardware_aware_winners.csv
results/09_05b_rwp_memory_rule2_winners.csv
results/09_05d_h4_week10_additions.csv
results/12_04b_iqm_rule1_rule4_candidates.csv
results/12_05g0_h5h6_cont_rule_candidates.csv
results/09_04c2_h4_backfill_manifest.json
results/09_04c2_h4_backfill_hardware_aware_per_window.csv

Authoritative mapping
---------------------
H0-H3:
  - frozen IBM Rule tables for R1/R2/R3/R4 identities and metrics;
  - CONT Rule-3 values are cross-checked against
    09_05a_cont_revised_hardware_aware_winners.csv;
  - RWP Rule-2 memory values are cross-checked against
    09_05b_rwp_memory_rule2_winners.csv.

H4:
  - ALL Rule-1..Rule-4 logical metrics from
    09_05d_h4_week10_additions.csv;
  - H4 RWP Rule-3 is additionally cross-checked against
    09_04c2_h4_backfill_manifest.json.

H5/H6 RWP:
  - 12_04b_iqm_rule1_rule4_candidates.csv

H5/H6 CONT:
  - 12_05g0_h5h6_cont_rule_candidates.csv

Emerald hardware:
  - 12_05g1_iqm_candidate_layout_summary.csv

Outputs
-------
results/12_05i_v4_1_iqm_emerald_candidate_evidence.csv
results/12_05i_v4_1_metric_provenance.csv
results/12_05i_v4_1_source_audit.csv
results/12_05i_v4_1_metric_completeness_audit.csv
results/12_05i_v4_1_role1_accuracy.csv
results/12_05i_v4_1_role2_same_family_resource.csv
results/12_05i_v4_1_role3_rwp_measurement_robust.csv
results/12_05i_v4_1_role4_cont_resource.csv
results/12_05i_v4_1_role5_memory.csv
results/12_05i_v4_1_manifest.json
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path("results")

UNIFIED = RESULTS / "12_05g0_unified_rule1_rule4_candidates.csv"
LAYOUT = RESULTS / "12_05g1_iqm_candidate_layout_summary.csv"
BACKEND_SELECTION = RESULTS / "12_05h_iqm_backend_selection.json"

H03_CONT_R3 = RESULTS / "09_05a_cont_revised_hardware_aware_winners.csv"
H03_RWP_R2 = RESULTS / "09_05b_rwp_memory_rule2_winners.csv"
H4_SOURCE = RESULTS / "09_05d_h4_week10_additions.csv"
H56_RWP = RESULTS / "12_04b_iqm_rule1_rule4_candidates.csv"
H56_CONT = RESULTS / "12_05g0_h5h6_cont_rule_candidates.csv"
H4_RWP_R3_MANIFEST = RESULTS / "09_04c2_h4_backfill_manifest.json"
H4_RWP_R3_PER_WINDOW = RESULTS / "09_04c2_h4_backfill_hardware_aware_per_window.csv"

OUT_MASTER = RESULTS / "12_05i_v4_1_iqm_emerald_candidate_evidence.csv"
OUT_PROV = RESULTS / "12_05i_v4_1_metric_provenance.csv"
OUT_SOURCE_AUDIT = RESULTS / "12_05i_v4_1_source_audit.csv"
OUT_METRIC_AUDIT = RESULTS / "12_05i_v4_1_metric_completeness_audit.csv"
OUT_R1 = RESULTS / "12_05i_v4_1_role1_accuracy.csv"
OUT_R2 = RESULTS / "12_05i_v4_1_role2_same_family_resource.csv"
OUT_R3 = RESULTS / "12_05i_v4_1_role3_rwp_measurement_robust.csv"
OUT_R4 = RESULTS / "12_05i_v4_1_role4_cont_resource.csv"
OUT_R5 = RESULTS / "12_05i_v4_1_role5_memory.csv"
OUT_MANIFEST = RESULTS / "12_05i_v4_1_manifest.json"

FROZEN_BACKEND = "emerald"
EXPECTED_UNIQUE = 47
EXPECTED_RULE_ASSIGNMENTS = 56
ATOL = 5e-6


# =============================================================================
# Exact frozen H0-H3 IBM rule table
# =============================================================================
#
# These are the frozen IBM logical-selection values already used in Week 9/10.
# This table is intentionally explicit and candidate-keyed. It is not inferred
# from another row and is never overwritten by a heuristic match.
#
# Only metrics relevant to a candidate's selected role(s) are required.
#
H03 = {
    # ---------------- H0 CONT ----------------
    "CONT_H0_R1R3": {
        "cv_rmse": 3.571471,
        "cv_rmse_std": 0.313613,
        "validation_rmse_2025_diagnostic": 5.270509,
        "Rmax_1024": 0.276526,
        "shotSD_1024": 0.279692,
        "Smax": 6.685337,
    },
    "CONT_H0_R2": {
        "cv_rmse": 3.578682,
        "MC_ch": 0.408347,
        "washout": 274.0,
    },
    "CONT_H0_R4": {
        "cv_rmse": 3.599207,
        "validation_rmse_2025_diagnostic": 5.263762,
    },

    # ---------------- H0 RWP ----------------
    "RWP_H0_R1R3": {
        "cv_rmse": 3.561381,
        "cv_rmse_std": 0.302647,
        "validation_rmse_2025_diagnostic": 5.139513,
        "Rmax_1024": 0.807,
        "shotSD_1024": 1.149,
        "Smax": 26.786,
    },
    "RWP_H0_R2": {
        "cv_rmse": 3.569399,
        "validation_rmse_2025_diagnostic": 5.285140,
        "MC_ch": 0.957831,
        "MC_ch_std": 0.056469,
    },
    "RWP_H0_R4": {
        "cv_rmse": 3.564757,
        "validation_rmse_2025_diagnostic": 5.275925,
    },

    # ---------------- H1 CONT ----------------
    "CONT_H1_R1": {
        "cv_rmse": 3.571156,
        "cv_rmse_std": 0.350473,
        "validation_rmse_2025_diagnostic": 5.252657,
    },
    "CONT_H1_R2R3": {
        "cv_rmse": 3.580085,
        "cv_rmse_std": 0.344096,
        "validation_rmse_2025_diagnostic": 5.167855,
        "MC_ch": 0.952249,
        "washout": 61.0,
        "Rmax_1024": 0.709297,
        "shotSD_1024": 0.744517,
        "Smax": 10.927099,
    },
    "CONT_H1_R4": {
        "cv_rmse": 3.591305,
        "validation_rmse_2025_diagnostic": 5.191004,
    },

    # ---------------- H1 RWP ----------------
    "RWP_H1_R1R3R4": {
        "cv_rmse": 3.575678,
        "cv_rmse_std": 0.326812,
        "validation_rmse_2025_diagnostic": 5.175301,
        "Rmax_1024": 0.683,
        "shotSD_1024": 1.105,
        "Smax": 32.700,
    },
    "RWP_H1_R2": {
        "cv_rmse": 3.578489,
        "validation_rmse_2025_diagnostic": 5.298316,
        "MC_ch": 0.734278,
        "MC_ch_std": 0.213293,
    },

    # ---------------- H2 CONT ----------------
    "CONT_H2_R1": {
        "cv_rmse": 3.562473,
        "cv_rmse_std": 0.286663,
        "validation_rmse_2025_diagnostic": 4.939399,
    },
    "CONT_H2_R2": {
        "cv_rmse": 3.563528,
        "MC_ch": 1.348139,
        "washout": 57.0,
    },
    "CONT_H2_R3": {
        "cv_rmse": 3.583461,
        "cv_rmse_std": 0.338455,
        "validation_rmse_2025_diagnostic": 5.370606,
        "Rmax_1024": 0.584766,
        "shotSD_1024": 0.388879,
        "Smax": 9.623723,
    },
    "CONT_H2_R4": {
        "cv_rmse": 3.576098,
        "validation_rmse_2025_diagnostic": 4.815420,
    },

    # ---------------- H2 RWP ----------------
    "RWP_H2_R1": {
        "cv_rmse": 3.554756,
        "cv_rmse_std": 0.292107,
        "validation_rmse_2025_diagnostic": 4.920776,
    },
    "RWP_H2_R2": {
        "cv_rmse": 3.607956,
        "validation_rmse_2025_diagnostic": 4.957045,
        "MC_ch": 1.320159,
        "MC_ch_std": 0.059916,
    },
    "RWP_H2_R3": {
        "cv_rmse": 3.565480,
        "validation_rmse_2025_diagnostic": 4.922051,
        "Rmax_1024": 0.969,
        "shotSD_1024": 0.825,
        "Smax": 25.588,
    },
    "RWP_H2_R4": {
        "cv_rmse": 3.566113,
        "validation_rmse_2025_diagnostic": 5.144412,
    },

    # ---------------- H3 CONT ----------------
    "CONT_H3_R1R3": {
        "cv_rmse": 3.496393,
        "cv_rmse_std": 0.381962,
        "validation_rmse_2025_diagnostic": 4.965947,
        "Rmax_1024": 0.834277,
        "shotSD_1024": 1.440313,
        "Smax": 29.537631,
    },
    "CONT_H3_R2": {
        "cv_rmse": 3.584000,
        "MC_ch": 1.268392,
        "washout": 61.4,
    },
    "CONT_H3_R4": {
        "cv_rmse": 3.593886,
        "validation_rmse_2025_diagnostic": 5.266323,
    },

    # ---------------- H3 RWP ----------------
    "RWP_H3_R1R3": {
        "cv_rmse": 3.40818409305363,
        "cv_rmse_std": 0.324417549932756,
        "validation_rmse_2025_diagnostic": 4.75592107798071,
        "Rmax_1024": 0.677,
        "shotSD_1024": 1.532,
        "Smax": 37.037,
    },
    "RWP_H3_R2": {
        "cv_rmse": 3.463899,
        "validation_rmse_2025_diagnostic": 4.890797,
        "MC_ch": 0.788484,
        "MC_ch_std": 0.037691,
    },
    "RWP_H3_R4": {
        "cv_rmse": 3.480126,
        "validation_rmse_2025_diagnostic": 4.989195,
    },
}


def finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if math.isfinite(v) else np.nan


def close(a, b, atol=ATOL):
    a = finite(a)
    b = finite(b)
    return np.isfinite(a) and np.isfinite(b) and math.isclose(
        a, b, rel_tol=0.0, abs_tol=atol
    )


def safe_int(x):
    v = finite(x)
    return int(v) if np.isfinite(v) else None


def norm_protocol(x):
    return str(x).strip().upper()


def has_rule(value, rule):
    # Semantic rule field only.
    text = str(value).upper()
    return bool(re.search(rf"(^|[^0-9]){re.escape(rule)}([^0-9]|$)", text))


def require_files():
    required = [
        UNIFIED,
        LAYOUT,
        BACKEND_SELECTION,
        H03_CONT_R3,
        H03_RWP_R2,
        H4_SOURCE,
        H56_RWP,
        H56_CONT,
        H4_RWP_R3_MANIFEST,
        H4_RWP_R3_PER_WINDOW,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required authoritative input(s):\n  "
            + "\n  ".join(missing)
        )


def init_metric_frame(cand, lay):
    rows = []

    for cid, g in lay.groupby("candidate_id", sort=False):
        c = cand[cand["candidate_id"].astype(str).eq(str(cid))]
        if len(c) != 1:
            raise RuntimeError(
                f"{cid}: expected exactly one unified candidate row; found {len(c)}"
            )
        cr = c.iloc[0]

        rows.append({
            "candidate_id": str(cid),
            "selected_rules": str(cr["selected_rules"]),
            "topology": str(cr["topology"]),
            "protocol": norm_protocol(cr["protocol"]),
            "window_effective": 1 if norm_protocol(cr["protocol"]) == "CONT"
                                else safe_int(cr["window"]),
            "r": safe_int(cr["r"]),
            "readout": str(cr["readout"]),

            # logical metrics filled ONLY by explicit sources below
            "cv_rmse": np.nan,
            "cv_rmse_std": np.nan,
            "validation_rmse_2025_diagnostic": np.nan,
            "MC_ch": np.nan,
            "MC_ch_std": np.nan,
            "washout": np.nan,
            "Rmax_1024": np.nan,
            "shotSD_1024": np.nan,
            "Smax": np.nan,

            # Emerald compile / hardware evidence
            "emerald_layouts_retained": int(g["layout"].nunique()),
            "compile_pass_all_3":
                bool(g["compile_all_settings_pass"].astype(bool).all()),
            "zero_swap_all_3":
                bool(g["zero_swap_all_settings"].astype(bool).all()),
            "NCZ_feature_vector":
                float(pd.to_numeric(g["NCZ_feature_vector"], errors="coerce").median()),
            "N2q_feature_vector":
                float(pd.to_numeric(g["N2q_feature_vector"], errors="coerce").median()),
            "depth_median_max_setting":
                float(pd.to_numeric(g["depth_max_setting"], errors="coerce").median()),
            "depth_worst_max_setting":
                float(pd.to_numeric(g["depth_max_setting"], errors="coerce").max()),
            "worst_min_cz_fidelity":
                float(pd.to_numeric(g["min_cz_fidelity"], errors="coerce").min()),
            "worst_max_asym_ro_error":
                float(pd.to_numeric(
                    g["max_asymmetric_readout_error"], errors="coerce"
                ).max()),
            "worst_min_t1_us":
                float(pd.to_numeric(g["min_t1_us"], errors="coerce").min()),
            "worst_min_t2_echo_us":
                float(pd.to_numeric(g["min_t2_echo_us"], errors="coerce").min()),
            "worst_memory_min_t2_echo_us":
                float(pd.to_numeric(
                    g["memory_min_t2_echo_us"], errors="coerce"
                ).min()),
            "worst_min_prx_fidelity":
                float(pd.to_numeric(g["min_prx_fidelity"], errors="coerce").min()),
            "median_prx_duration_ns":
                float(pd.to_numeric(
                    g["prx_duration_median_ns"], errors="coerce"
                ).median()),
            "median_cz_duration_ns":
                float(pd.to_numeric(
                    g["cz_duration_median_ns"], errors="coerce"
                ).median()),
        })

    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_UNIQUE:
        raise RuntimeError(
            f"Expected {EXPECTED_UNIQUE} unique candidate rows, found {len(out)}"
        )
    return out


def put(master, cid, metric, value, source, provenance):
    mask = master["candidate_id"].eq(cid)
    if int(mask.sum()) != 1:
        raise RuntimeError(
            f"Cannot attach {metric}: candidate {cid!r} occurs {int(mask.sum())} times."
        )
    v = finite(value)
    if np.isfinite(v):
        master.loc[mask, metric] = v
        provenance.append({
            "candidate_id": cid,
            "metric": metric,
            "value": v,
            "source": source,
        })


def attach_h03(master, provenance):
    present = set(master.loc[
        master["topology"].isin(["H0", "H1", "H2", "H3"]),
        "candidate_id"
    ])

    expected = set(H03)
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise RuntimeError(
            "H0-H3 frozen candidate identity mismatch.\n"
            f"Missing from unified: {missing}\n"
            f"Unexpected in unified: {extra}"
        )

    for cid, metrics in H03.items():
        for metric, value in metrics.items():
            put(
                master, cid, metric, value,
                "frozen_IBM_H0_H3_rule_table",
                provenance,
            )


def crosscheck_h03_cont_r3(master, source_audit):
    src = pd.read_csv(H03_CONT_R3)
    src = src[src["topology"].isin(["H0", "H1", "H2", "H3"])].copy()

    expected_map = {
        "H0": "CONT_H0_R1R3",
        "H1": "CONT_H1_R2R3",
        "H2": "CONT_H2_R3",
        "H3": "CONT_H3_R1R3",
    }

    for topo, cid in expected_map.items():
        g = src[src["topology"].eq(topo)]
        if len(g) != 1:
            raise RuntimeError(
                f"{H03_CONT_R3}: expected one {topo} row, found {len(g)}"
            )
        r = g.iloc[0]
        m = master.loc[master["candidate_id"].eq(cid)].iloc[0]

        checks = {
            "cv_rmse": r["cv_rmse"],
            "cv_rmse_std": r["cv_rmse_std"],
            "validation_rmse_2025_diagnostic": r["validation_rmse"],
            "Rmax_1024": r["max_shot_to_train_std_ratio_1024"],
            "shotSD_1024": r["shot_noise_forecast_sd_proxy_1024"],
            "Smax": r["max_abs_raw_prediction_sensitivity"],
        }

        for metric, observed in checks.items():
            passed = close(m[metric], observed, atol=2e-4)
            source_audit.append({
                "candidate_id": cid,
                "check": metric,
                "expected_or_frozen": m[metric],
                "source_observed": finite(observed),
                "source": str(H03_CONT_R3),
                "pass": passed,
            })


def crosscheck_h03_rwp_r2(master, source_audit):
    src = pd.read_csv(H03_RWP_R2)

    expected_map = {
        "H0": "RWP_H0_R2",
        "H1": "RWP_H1_R2",
        "H2": "RWP_H2_R2",
        "H3": "RWP_H3_R2",
    }

    for topo, cid in expected_map.items():
        g = src[src["topology"].eq(topo)]
        if len(g) != 1:
            raise RuntimeError(
                f"{H03_RWP_R2}: expected one {topo} row, found {len(g)}"
            )
        r = g.iloc[0]
        m = master.loc[master["candidate_id"].eq(cid)].iloc[0]

        checks = {
            "cv_rmse": r["source_cv_rmse"],
            "validation_rmse_2025_diagnostic": r["source_validation_rmse"],
            "MC_ch": r["MC_ch_mean"],
            "MC_ch_std": r["MC_ch_std"],
        }

        for metric, observed in checks.items():
            passed = close(m[metric], observed, atol=2e-4)
            source_audit.append({
                "candidate_id": cid,
                "check": metric,
                "expected_or_frozen": m[metric],
                "source_observed": finite(observed),
                "source": str(H03_RWP_R2),
                "pass": passed,
            })


def attach_h4(master, provenance, source_audit):
    src = pd.read_csv(H4_SOURCE)

    expected_ids = set(
        master.loc[master["topology"].eq("H4"), "candidate_id"]
    )
    source_ids = set(src["candidate_key"].astype(str))

    if expected_ids != source_ids:
        raise RuntimeError(
            "H4 candidate identity mismatch.\n"
            f"Missing in H4 source: {sorted(expected_ids-source_ids)}\n"
            f"Unexpected in H4 source: {sorted(source_ids-expected_ids)}"
        )

    mapping = {
        "source_readout_cv_rmse": "cv_rmse",
        "source_readout_cv_sd": "cv_rmse_std",
        "source_validation_rmse": "validation_rmse_2025_diagnostic",
        "MC_ch_mean": "MC_ch",
        "MC_ch_std": "MC_ch_std",
        "Tw_0.01_mean": "washout",
        "max_shot_to_train_std_ratio_1024": "Rmax_1024",
        "shot_noise_forecast_sd_proxy_1024": "shotSD_1024",
    }

    for _, r in src.iterrows():
        cid = str(r["candidate_key"])
        m = master.loc[master["candidate_id"].eq(cid)].iloc[0]

        identity_checks = {
            "topology": str(r["topology"]) == str(m["topology"]),
            "protocol": norm_protocol(r["protocol"]) == str(m["protocol"]),
            "r": safe_int(r["original_r"]) == safe_int(m["r"]),
            "readout": str(r["test_readout"]) == str(m["readout"]),
            "window": (
                safe_int(r["window"]) == safe_int(m["window_effective"])
            ),
        }

        for name, passed in identity_checks.items():
            source_audit.append({
                "candidate_id": cid,
                "check": f"identity_{name}",
                "expected_or_frozen": str(m[name if name != "window" else "window_effective"])
                    if name != "r" else safe_int(m["r"]),
                "source_observed": (
                    str(r["topology"]) if name == "topology"
                    else norm_protocol(r["protocol"]) if name == "protocol"
                    else safe_int(r["original_r"]) if name == "r"
                    else str(r["test_readout"]) if name == "readout"
                    else safe_int(r["window"])
                ),
                "source": str(H4_SOURCE),
                "pass": bool(passed),
            })

        for src_col, metric in mapping.items():
            put(master, cid, metric, r[src_col], str(H4_SOURCE), provenance)

    # -----------------------------------------------------------------
    # H4 Rule-3 Smax is NOT stored in 09_05d_h4_week10_additions.csv.
    # Recover it only from the exact authoritative conditioning rows.
    # Keep Smax mandatory for Rule 3; do not weaken the audit.
    # -----------------------------------------------------------------

    # CONT_H4_R3: exact H4 row from the CONT hardware-aware winner table.
    cont_r3 = pd.read_csv(H03_CONT_R3)
    cont_r3 = cont_r3[
        cont_r3["topology"].astype(str).eq("H4")
        & pd.to_numeric(cont_r3["r"], errors="coerce").eq(3)
        & cont_r3["readout"].astype(str).eq("XZinj_dropX3_plus_YX45")
    ].copy()

    if len(cont_r3) != 1:
        raise RuntimeError(
            f"{H03_CONT_R3}: expected exactly one CONT_H4_R3 source row, "
            f"found {len(cont_r3)}"
        )

    cr = cont_r3.iloc[0]
    cid = "CONT_H4_R3"
    m = master.loc[master["candidate_id"].eq(cid)].iloc[0]

    # Cross-check all overlapping Rule-3 metrics before accepting Smax.
    for metric, observed in {
        "cv_rmse": cr["cv_rmse"],
        "cv_rmse_std": cr["cv_rmse_std"],
        "validation_rmse_2025_diagnostic": cr["validation_rmse"],
        "shotSD_1024": cr["shot_noise_forecast_sd_proxy_1024"],
        "Rmax_1024": cr["max_shot_to_train_std_ratio_1024"],
    }.items():
        source_audit.append({
            "candidate_id": cid,
            "check": f"H4_CONT_R3_exact_{metric}",
            "expected_or_frozen": m[metric],
            "source_observed": finite(observed),
            "source": str(H03_CONT_R3),
            "pass": close(m[metric], observed),
        })

    put(
        master,
        cid,
        "Smax",
        cr["max_abs_raw_prediction_sensitivity"],
        str(H03_CONT_R3),
        provenance,
    )

    # RWP_H4_R3: exact W=6, r=3, local_01, readout row from C2.
    rwp_r3 = pd.read_csv(H4_RWP_R3_PER_WINDOW)
    rwp_r3 = rwp_r3[
        rwp_r3["topology"].astype(str).eq("H4")
        & pd.to_numeric(rwp_r3["window"], errors="coerce").eq(6)
        & pd.to_numeric(rwp_r3["r"], errors="coerce").eq(3)
        & rwp_r3["search_id"].astype(str).eq("local_01")
        & rwp_r3["readout"].astype(str).eq("XZinj_dropX3_plus_YX45")
    ].copy()

    if len(rwp_r3) != 1:
        raise RuntimeError(
            f"{H4_RWP_R3_PER_WINDOW}: expected exactly one RWP_H4_R3 "
            f"W=6,r=3,local_01 source row, found {len(rwp_r3)}"
        )

    rr = rwp_r3.iloc[0]
    cid = "RWP_H4_R3"
    m = master.loc[master["candidate_id"].eq(cid)].iloc[0]

    for metric, observed in {
        "cv_rmse": rr["cv_rmse"],
        "cv_rmse_std": rr["cv_rmse_std"],
        "validation_rmse_2025_diagnostic": rr["validation_rmse"],
        "shotSD_1024": rr["shot_noise_forecast_sd_proxy_1024"],
        "Rmax_1024": rr["max_shot_to_train_std_ratio_1024"],
    }.items():
        source_audit.append({
            "candidate_id": cid,
            "check": f"H4_RWP_R3_exact_{metric}",
            "expected_or_frozen": m[metric],
            "source_observed": finite(observed),
            "source": str(H4_RWP_R3_PER_WINDOW),
            "pass": close(m[metric], observed),
        })

    put(
        master,
        cid,
        "Smax",
        rr["max_abs_raw_prediction_sensitivity"],
        str(H4_RWP_R3_PER_WINDOW),
        provenance,
    )

    # Independent exact check of H4 RWP Rule-3 manifest.
    with open(H4_RWP_R3_MANIFEST, "r", encoding="utf-8") as f:
        j = json.load(f)

    hw = j["global_hardware_aware_representative"]
    cid = "RWP_H4_R3"
    m = master.loc[master["candidate_id"].eq(cid)].iloc[0]

    checks = {
        "window_effective": hw["window"],
        "r": hw["r"],
        "cv_rmse": hw["cv_rmse"],
        "validation_rmse_2025_diagnostic": hw["validation_rmse_diagnostic"],
        "shotSD_1024": hw["shot_noise_forecast_sd_proxy_1024"],
        "Rmax_1024": hw["max_shot_to_train_std_ratio_1024"],
    }

    for metric, observed in checks.items():
        expected = m[metric]
        passed = (
            safe_int(expected) == safe_int(observed)
            if metric in {"window_effective", "r"}
            else close(expected, observed)
        )
        source_audit.append({
            "candidate_id": cid,
            "check": f"H4_RWP_R3_manifest_{metric}",
            "expected_or_frozen": expected,
            "source_observed": observed,
            "source": str(H4_RWP_R3_MANIFEST),
            "pass": bool(passed),
        })


def attach_h56_rwp(master, provenance, source_audit):
    src = pd.read_csv(H56_RWP)

    expected_ids = set(
        master.loc[
            master["topology"].isin(["H5", "H6"])
            & master["protocol"].eq("RWP"),
            "candidate_id"
        ]
    )
    source_ids = set(src["candidate_uid"].astype(str))

    if expected_ids != source_ids:
        raise RuntimeError(
            "H5/H6 RWP candidate identity mismatch.\n"
            f"Missing in source: {sorted(expected_ids-source_ids)}\n"
            f"Unexpected in source: {sorted(source_ids-expected_ids)}"
        )

    for _, r in src.iterrows():
        cid = str(r["candidate_uid"])
        m = master.loc[master["candidate_id"].eq(cid)].iloc[0]

        identity = {
            "topology": str(r["topology"]) == str(m["topology"]),
            "protocol": str(m["protocol"]) == "RWP",
            "window": safe_int(r["window"]) == safe_int(m["window_effective"]),
            "r": safe_int(r["r"]) == safe_int(m["r"]),
            "readout": str(r["readout"]) == str(m["readout"]),
        }

        for name, passed in identity.items():
            source_audit.append({
                "candidate_id": cid,
                "check": f"identity_{name}",
                "expected_or_frozen": "",
                "source_observed": "",
                "source": str(H56_RWP),
                "pass": bool(passed),
            })

        vals = {
            "cv_rmse": r["cv_rmse"],
            "cv_rmse_std": r["cv_rmse_std"],
            "validation_rmse_2025_diagnostic": r["validation_rmse"],
            "MC_ch": r["MC_ch_mean_rule2"],
            "MC_ch_std": r["MC_ch_std_rule2"],
            "Rmax_1024": r["max_shot_to_train_std_ratio_1024"],
            "shotSD_1024": r["shot_noise_forecast_sd_proxy_1024"],
            "Smax": r["max_abs_raw_prediction_sensitivity"],
        }

        for metric, value in vals.items():
            put(master, cid, metric, value, str(H56_RWP), provenance)


def attach_h56_cont(master, provenance, source_audit):
    src = pd.read_csv(H56_CONT)

    expected_ids = set(
        master.loc[
            master["topology"].isin(["H5", "H6"])
            & master["protocol"].eq("CONT"),
            "candidate_id"
        ]
    )
    source_ids = set(src["candidate_id"].astype(str))

    if expected_ids != source_ids:
        raise RuntimeError(
            "H5/H6 CONT candidate identity mismatch.\n"
            f"Missing in source: {sorted(expected_ids-source_ids)}\n"
            f"Unexpected in source: {sorted(source_ids-expected_ids)}"
        )

    for _, r in src.iterrows():
        cid = str(r["candidate_id"])
        m = master.loc[master["candidate_id"].eq(cid)].iloc[0]

        identity = {
            "topology": str(r["topology"]) == str(m["topology"]),
            "protocol": norm_protocol(r["protocol"]) == str(m["protocol"]),
            "r": safe_int(r["r"]) == safe_int(m["r"]),
            "readout": str(r["readout"]) == str(m["readout"]),
        }

        for name, passed in identity.items():
            source_audit.append({
                "candidate_id": cid,
                "check": f"identity_{name}",
                "expected_or_frozen": "",
                "source_observed": "",
                "source": str(H56_CONT),
                "pass": bool(passed),
            })

        vals = {
            "cv_rmse": r["cv_rmse"],
            "cv_rmse_std": r["cv_rmse_std"],
            "validation_rmse_2025_diagnostic": r["validation_rmse"],
            "MC_ch": r["MC_ch_mean"],
            "MC_ch_std": r["MC_ch_std"],
            "washout": r["Tw_0.01_mean"],
            "Rmax_1024": r["max_shot_to_train_std_ratio_1024"],
            "shotSD_1024": r["shot_noise_forecast_sd_proxy_1024"],
            "Smax": r["max_abs_raw_prediction_sensitivity"],
        }

        for metric, value in vals.items():
            put(master, cid, metric, value, str(H56_CONT), provenance)


def audit_source_rows(source_audit):
    df = pd.DataFrame(source_audit)
    if df.empty:
        raise RuntimeError("No source audit rows were produced.")
    if not df["pass"].astype(bool).all():
        bad = df[~df["pass"].astype(bool)]
        print("\nSOURCE AUDIT FAILURES")
        print(bad.to_string(index=False))
        raise RuntimeError(
            "12.5I v4.1 source audit FAILED. No shortlist will be produced."
        )
    return df


def audit_rule_assignments(master):
    rows = []

    for _, r in master.iterrows():
        rules = str(r["selected_rules"])
        requirements = []

        if has_rule(rules, "R1"):
            requirements.append(("R1", ["cv_rmse"]))

        if has_rule(rules, "R2"):
            req = ["MC_ch"]
            if r["protocol"] == "CONT":
                req.append("washout")
            requirements.append(("R2", req))

        if has_rule(rules, "R3"):
            requirements.append(
                ("R3", ["cv_rmse", "Rmax_1024", "shotSD_1024", "Smax"])
            )

        if has_rule(rules, "R4"):
            requirements.append(
                ("R4", ["cv_rmse", "NCZ_feature_vector", "depth_worst_max_setting"])
            )

        for rule, cols in requirements:
            missing = [
                c for c in cols
                if not np.isfinite(finite(r[c]))
            ]

            # CONT Rule-2 washout validity is part of the rule.
            washout_valid = True
            if rule == "R2" and r["protocol"] == "CONT":
                washout_valid = (
                    np.isfinite(finite(r["washout"]))
                    and finite(r["washout"]) <= 300.0
                )

            rows.append({
                "candidate_id": r["candidate_id"],
                "topology": r["topology"],
                "protocol": r["protocol"],
                "rule": rule,
                "required_metrics": "+".join(cols),
                "missing_metrics": "+".join(missing),
                "washout_valid_if_required": washout_valid,
                "pass": (len(missing) == 0 and washout_valid),
            })

    audit = pd.DataFrame(rows)

    if len(audit) != EXPECTED_RULE_ASSIGNMENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_RULE_ASSIGNMENTS} rule assignments, "
            f"found {len(audit)}."
        )

    if not audit["pass"].astype(bool).all():
        bad = audit[~audit["pass"].astype(bool)]
        print("\nRULE-METRIC AUDIT FAILURES")
        print(bad.to_string(index=False))
        raise RuntimeError(
            "12.5I v4.1 rule-metric audit FAILED. No shortlist will be produced."
        )

    return audit


def pareto(df, objectives):
    needed = [c for c, _ in objectives]
    work = df.dropna(subset=needed).copy()

    if work.empty:
        return work

    X = work[needed].to_numpy(float)
    keep = np.ones(len(work), dtype=bool)

    for i in range(len(work)):
        for j in range(len(work)):
            if i == j:
                continue

            no_worse = True
            strictly_better = False

            for k, (_, direction) in enumerate(objectives):
                vi = X[i, k]
                vj = X[j, k]

                if direction == "min":
                    if vj > vi:
                        no_worse = False
                        break
                    if vj < vi:
                        strictly_better = True
                else:
                    if vj < vi:
                        no_worse = False
                        break
                    if vj > vi:
                        strictly_better = True

            if no_worse and strictly_better:
                keep[i] = False
                break

    return work.iloc[np.flatnonzero(keep)].copy()


def build_roles(master):
    # Role 1 — IBM Candidate-1 analogue:
    # minimum chronological training CV among R1 representatives.
    role1 = master[
        master["selected_rules"].apply(lambda x: has_rule(x, "R1"))
    ].copy().sort_values(
        ["cv_rmse", "cv_rmse_std", "NCZ_feature_vector"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    leader = role1.iloc[0]

    # Role 2 — IBM Candidate-2 analogue:
    # Rule-4 resource alternative from SAME topology + protocol family
    # as the accuracy leader.
    role2 = master[
        master["selected_rules"].apply(lambda x: has_rule(x, "R4"))
        & master["topology"].eq(leader["topology"])
        & master["protocol"].eq(leader["protocol"])
        & ~master["candidate_id"].eq(leader["candidate_id"])
    ].copy().sort_values(
        ["NCZ_feature_vector", "depth_worst_max_setting", "cv_rmse"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    # Role 3 — IBM Candidate-3 analogue:
    # RWP Rule-3 candidate from a different topology than the leader.
    # Apply the Rule-3 hierarchy first.
    role3_pool = master[
        master["selected_rules"].apply(lambda x: has_rule(x, "R3"))
        & master["protocol"].eq("RWP")
        & ~master["topology"].eq(leader["topology"])
    ].copy()

    safe = role3_pool[role3_pool["Rmax_1024"] <= 1.0].copy()

    if not safe.empty:
        role3_pool = safe
        role3_mode = "Rmax<=1 set -> CV -> shotSD -> Smax -> resources"
    else:
        role3_mode = "no Rmax<=1 -> Rmax -> shotSD -> CV -> Smax -> resources"

    if not safe.empty:
        role3 = role3_pool.sort_values(
            [
                "cv_rmse",
                "shotSD_1024",
                "Smax",
                "NCZ_feature_vector",
                "depth_worst_max_setting",
            ],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        )
    else:
        role3 = role3_pool.sort_values(
            [
                "Rmax_1024",
                "shotSD_1024",
                "cv_rmse",
                "Smax",
                "NCZ_feature_vector",
                "depth_worst_max_setting",
            ],
            ascending=[True, True, True, True, True, True],
            kind="mergesort",
        )

    # Role 4 — IBM Candidate-4 analogue:
    # cheap CONT Rule-4 feasibility baseline.
    # All R4 representatives are already topology/protocol-level one-SE choices.
    # We expose a resource/hardware Pareto frontier, not a weighted score.
    role4_pool = master[
        master["selected_rules"].apply(lambda x: has_rule(x, "R4"))
        & master["protocol"].eq("CONT")
    ].copy()

    role4 = pareto(
        role4_pool,
        [
            ("NCZ_feature_vector", "min"),
            ("depth_worst_max_setting", "min"),
            ("cv_rmse", "min"),
            ("worst_min_cz_fidelity", "max"),
            ("worst_max_asym_ro_error", "min"),
        ],
    ).sort_values(
        ["NCZ_feature_vector", "depth_worst_max_setting", "cv_rmse"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    # Role 5 — IBM Candidate-5 analogue:
    # memory-first Rule-2 candidate.
    # The scientific primary is MC/ch. Hardware/resource quantities are shown,
    # but do not replace the Rule-2 objective.
    role5 = master[
        master["selected_rules"].apply(lambda x: has_rule(x, "R2"))
    ].copy().sort_values(
        [
            "MC_ch",
            "NCZ_feature_vector",
            "depth_worst_max_setting",
            "worst_memory_min_t2_echo_us",
        ],
        ascending=[False, True, True, False],
        kind="mergesort",
    )

    return role1, role2, role3, role4, role5, leader, role3_mode


def print_table(title, df, max_rows=40):
    print()
    print("=" * 170)
    print(title)
    print("=" * 170)

    if df.empty:
        print("<empty>")
        return

    cols = [
        "candidate_id",
        "selected_rules",
        "topology",
        "protocol",
        "window_effective",
        "r",
        "readout",
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
    ]
    print(df[cols].head(max_rows).to_string(index=False))


def main():
    require_files()

    with open(BACKEND_SELECTION, "r", encoding="utf-8") as f:
        backend_info = json.load(f)

    selected_backend = str(
        backend_info.get("selected_backend", "")
    ).strip().lower()

    if selected_backend != FROZEN_BACKEND:
        raise RuntimeError(
            f"Expected frozen backend {FROZEN_BACKEND!r}, "
            f"found {selected_backend!r}."
        )

    cand = pd.read_csv(UNIFIED)
    lay = pd.read_csv(LAYOUT)
    lay = lay[
        lay["backend"].astype(str).str.lower().eq(FROZEN_BACKEND)
    ].copy()

    if cand["candidate_id"].nunique() != EXPECTED_UNIQUE:
        raise RuntimeError(
            f"Expected {EXPECTED_UNIQUE} unique candidates, "
            f"found {cand['candidate_id'].nunique()}."
        )

    if lay["candidate_id"].nunique() != EXPECTED_UNIQUE:
        raise RuntimeError(
            "Emerald layout summary does not contain all 47 candidates."
        )

    if not lay["compile_all_settings_pass"].astype(bool).all():
        raise RuntimeError("At least one Emerald candidate/layout failed compile.")

    if not lay["zero_swap_all_settings"].astype(bool).all():
        raise RuntimeError("At least one Emerald candidate/layout is not zero-SWAP.")

    print("=" * 170)
    print("WEEK 12.5I v4.1 — FINAL STRICT IBM-STYLE IQM FINALIST EVIDENCE")
    print("=" * 170)
    print("Frozen backend: EMERALD")
    print("47 unique candidates / 56 rule assignments")
    print("No directory scan. No fuzzy matching. No weighted score.")
    print("No simulator/QPU jobs. 2026 untouched.")
    print()

    master = init_metric_frame(cand, lay)
    provenance = []
    source_audit = []

    # Explicit source attachment only.
    attach_h03(master, provenance)
    crosscheck_h03_cont_r3(master, source_audit)
    crosscheck_h03_rwp_r2(master, source_audit)
    attach_h4(master, provenance, source_audit)
    attach_h56_rwp(master, provenance, source_audit)
    attach_h56_cont(master, provenance, source_audit)

    source_audit_df = audit_source_rows(source_audit)
    metric_audit = audit_rule_assignments(master)

    master.to_csv(OUT_MASTER, index=False)
    pd.DataFrame(provenance).to_csv(OUT_PROV, index=False)
    source_audit_df.to_csv(OUT_SOURCE_AUDIT, index=False)
    metric_audit.to_csv(OUT_METRIC_AUDIT, index=False)

    print(
        f"Source/provenance audit: PASS — "
        f"{int(source_audit_df['pass'].sum())}/{len(source_audit_df)} checks."
    )

    print()
    print("RULE-METRIC COMPLETENESS AUDIT")
    print("-" * 170)
    summary = (
        metric_audit.groupby("rule", as_index=False)
        .agg(
            assignments=("candidate_id", "count"),
            passed=("pass", "sum"),
        )
    )
    print(summary.to_string(index=False))
    print()
    print(
        f"Rule assignment audit: PASS — "
        f"{int(metric_audit['pass'].sum())}/{len(metric_audit)}."
    )

    (
        role1,
        role2,
        role3,
        role4,
        role5,
        leader,
        role3_mode,
    ) = build_roles(master)

    role1.to_csv(OUT_R1, index=False)
    role2.to_csv(OUT_R2, index=False)
    role3.to_csv(OUT_R3, index=False)
    role4.to_csv(OUT_R4, index=False)
    role5.to_csv(OUT_R5, index=False)

    print_table("ROLE 1 — ACCURACY LEADER (R1)", role1)
    print()
    print(
        "Provisional Role-1 leader from complete R1 evidence: "
        f"{leader['candidate_id']} | CV={leader['cv_rmse']:.6f}"
    )

    print_table(
        "ROLE 2 — SAME-FAMILY RULE-4 RESOURCE ALTERNATIVE",
        role2,
    )

    print()
    print(f"Role-3 hierarchy applied: {role3_mode}")
    print_table(
        "ROLE 3 — RWP RULE-3 MEASUREMENT-ROBUST / LOWER-RESOURCE ALTERNATIVE",
        role3,
    )

    print_table(
        "ROLE 4 — CONT RULE-4 RESOURCE / PERSISTENCE-FEASIBILITY PARETO",
        role4,
    )

    print_table(
        "ROLE 5 — RULE-2 MEMORY-FIRST SHORTLIST",
        role5,
    )

    manifest = {
        "stage": "12.5I_v4_1",
        "backend": FROZEN_BACKEND,
        "unique_candidates": EXPECTED_UNIQUE,
        "rule_assignments": EXPECTED_RULE_ASSIGNMENTS,
        "source_audit_pass": True,
        "metric_audit_pass": True,
        "weighted_score": False,
        "directory_scan_used": False,
        "fuzzy_matching_used": False,
        "final_five_frozen": False,
        "role1_provisional_leader": str(leader["candidate_id"]),
        "role3_selection_mode": role3_mode,
        "authoritative_sources": {
            "H0_H3_frozen_rules": "embedded frozen IBM rule table",
            "H0_H3_CONT_R3_crosscheck": str(H03_CONT_R3),
            "H0_H3_RWP_R2_crosscheck": str(H03_RWP_R2),
            "H4_all_rules": str(H4_SOURCE),
            "H4_RWP_R3_crosscheck": str(H4_RWP_R3_MANIFEST),
            "H4_RWP_R3_Smax_exact_row": str(H4_RWP_R3_PER_WINDOW),
            "H4_CONT_R3_Smax_exact_row": str(H03_CONT_R3),
            "H5_H6_RWP": str(H56_RWP),
            "H5_H6_CONT": str(H56_CONT),
            "Emerald_hardware": str(LAYOUT),
        },
        "qpu_jobs": 0,
        "simulator_jobs": 0,
        "test_2026_used": False,
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 170)
    print("12.5I v4.1 COMPLETE — STRICT EVIDENCE BUILT")
    print("=" * 170)
    print("Saved:")
    for p in [
        OUT_MASTER,
        OUT_PROV,
        OUT_SOURCE_AUDIT,
        OUT_METRIC_AUDIT,
        OUT_R1,
        OUT_R2,
        OUT_R3,
        OUT_R4,
        OUT_R5,
        OUT_MANIFEST,
    ]:
        print(f"  {p}")

    print()
    print(
        "STOP HERE. Send the source/provenance audit, rule-metric audit, "
        "and five role tables. Only then freeze the exact five Emerald QPU finalists."
    )


if __name__ == "__main__":
    main()
