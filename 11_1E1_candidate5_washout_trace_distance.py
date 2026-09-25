from __future__ import annotations

"""
WEEK 11.1E.1 — CANDIDATE #5 MATHEMATICAL WASHOUT AUDIT
=======================================================

This starts Candidate #5 using exactly the same methodology as Candidate #4.

Candidate #5 is the frozen Week-10 CONT/H2/Rule-2 branch:
    protocol = CONT
    topology = H2
    operating r = 2
    readout = XZinj
    planned shots/setting = 1024

The exact hx, hy, alpha, dt and J values are EXTRACTED from the final Week-10
candidate manifest.  They are not re-selected here.

Five initial two-memory-qubit states are driven by the SAME chronological F4
inputs:

    |00><00|
    |11><11|
    |++><++|
    Bell |Phi+><Phi+|
    I/4

For every chronological step:

    D_t = max_{A,B} 1/2 || rho_A(t) - rho_B(t) ||_1

Stable washout at threshold epsilon is:

    T_w(epsilon) =
        first t such that D_s <= epsilon for EVERY s >= t.

The script also computes the maximum pairwise distance in the exact frozen
readout feature space:

    X0,X1,X2,X3,Z0,Z1,Z2,Z3

No intermediate measurement is applied.  This is the original unmeasured CONT
channel; later RWP/RSP will measure only at the endpoint.

NO IBM QPU JOB IS SUBMITTED.
2026 IS NEVER LOADED.
"""

import importlib.util
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import db_objects as db


# =============================================================================
# Paths / frozen identity
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
MANIFEST_FILE = (
    RESULTS
    / "10_05_rule1_to_rule5_candidate_manifest_ENRICHED.csv"
)


CANDIDATE_LABEL = "Candidate #5"
EXPECTED_PROTOCOL = "CONT"
EXPECTED_TOPOLOGY = "H2"
EXPECTED_R = 2
EXPECTED_READOUT = "XZinj"
PLANNED_SHOTS = 1024

FEATURES = [
    "X0", "X1", "X2", "X3",
    "Z0", "Z1", "Z2", "Z3",
]

THRESHOLDS = [
    0.10,
    0.05,
    0.02,
    0.01,
    0.005,
    0.001,
]


# =============================================================================
# Module loading
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required project file not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


common = load_module(
    COMMON_FILE,
    "qrc_11_1e1_common",
)

qrc = common.qrc


# =============================================================================
# Candidate extraction
# =============================================================================

def norm_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def readout_matches(value):
    value = norm_text(
        value
    ).lower().replace(
        "_",
        "",
    )

    return value in {
        "xzinj",
        "xzinjection",
    }


def candidate_rule2_mask(df):
    if "selected_by_rules" not in df.columns:
        return np.ones(
            len(df),
            dtype=bool,
        )

    s = (
        df[
            "selected_by_rules"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    return (
        s.str.contains(
            "r2",
            regex=False,
        )
        |
        s.str.contains(
            "rule2",
            regex=False,
        )
        |
        s.str.contains(
            "rule 2",
            regex=False,
        )
    )


def resolve_candidate_row():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing Week-10 manifest: {MANIFEST_FILE}"
        )

    df = pd.read_csv(
        MANIFEST_FILE
    )

    # First try the expected exact name if present.
    if "candidate_key" in df.columns:
        exact = df[
            df[
                "candidate_key"
            ].astype(str)
            == "CONT_H2_R2"
        ]

        if len(exact) == 1:
            return exact.iloc[0]

    protocol_mask = (
        df[
            "protocol"
        ].astype(str)
        .str.upper()
        == EXPECTED_PROTOCOL
    )

    topology_mask = (
        df[
            "topology"
        ].astype(str)
        == EXPECTED_TOPOLOGY
    )

    readout_mask = (
        df[
            "test_readout"
        ].apply(
            readout_matches
        )
    )

    rule_mask = candidate_rule2_mask(
        df
    )

    r_col = (
        "original_r"
        if "original_r"
        in df.columns
        else "r"
    )

    r_mask = (
        pd.to_numeric(
            df[
                r_col
            ],
            errors="coerce",
        )
        == EXPECTED_R
    )

    sub = df[
        protocol_mask
        & topology_mask
        & readout_mask
        & rule_mask
        & r_mask
    ].copy()

    if len(sub) != 1:
        cols = [
            c
            for c in [
                "candidate_key",
                "protocol",
                "topology",
                r_col,
                "test_readout",
                "selected_by_rules",
                "alpha",
                "dt",
                "hx",
                "hy",
            ]
            if c in df.columns
        ]

        print(
            "Candidate #5 resolution candidates:"
        )
        print(
            df[
                protocol_mask
                & topology_mask
            ][
                cols
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Could not uniquely resolve Candidate #5 "
            "from the final Week-10 manifest."
        )

    return sub.iloc[0]


def load_candidate():
    row = resolve_candidate_row()

    raw_j = json.loads(
        str(
            row[
                "J_json"
            ]
        )
    )

    J = {}

    for i, j in common.TOPOLOGY_EDGES[
        EXPECTED_TOPOLOGY
    ]:
        key = f"J{i}{j}"
        rev = f"J{j}{i}"

        if key in raw_j:
            value = raw_j[
                key
            ]
        elif rev in raw_j:
            value = raw_j[
                rev
            ]
        else:
            raise KeyError(
                f"Missing coupling for edge {(i, j)}"
            )

        J[
            (i, j)
        ] = float(
            value
        )

    candidate_key = norm_text(
        row.get(
            "candidate_key",
            "CONT_H2_R2",
        )
    )

    candidate = {
        "candidate_id": (
            candidate_key
            or "CONT_H2_R2"
        ),
        "protocol": EXPECTED_PROTOCOL,
        "topology": EXPECTED_TOPOLOGY,
        "r": EXPECTED_R,
        "alpha": float(
            row[
                "alpha"
            ]
        ),
        "dt": float(
            row[
                "dt"
            ]
        ),
        "hx": float(
            row[
                "hx"
            ]
        ),
        "hy": float(
            row[
                "hy"
            ]
        ),
        "J": J,
    }

    return candidate, row.to_dict()


# =============================================================================
# Initial memory states
# =============================================================================

ket0 = np.array(
    [1.0, 0.0],
    dtype=complex,
)

ket1 = np.array(
    [0.0, 1.0],
    dtype=complex,
)

ketp = (
    ket0 + ket1
) / np.sqrt(2.0)

ket00 = np.kron(
    ket0,
    ket0,
)

ket11 = np.kron(
    ket1,
    ket1,
)

ketpp = np.kron(
    ketp,
    ketp,
)

ket_phi = (
    ket00 + ket11
) / np.sqrt(2.0)


def pure_density(ket):
    return np.outer(
        ket,
        ket.conj(),
    )


INITIAL_STATES = {
    "00": pure_density(
        ket00
    ),
    "11": pure_density(
        ket11
    ),
    "++": pure_density(
        ketpp
    ),
    "BellPhi+": pure_density(
        ket_phi
    ),
    "I/4": (
        np.eye(
            4,
            dtype=complex,
        )
        / 4.0
    ),
}


# =============================================================================
# Mathematical helpers
# =============================================================================

def normalize_density(rho):
    rho = np.asarray(
        rho,
        dtype=complex,
    )

    rho = 0.5 * (
        rho
        + rho.conj().T
    )

    tr = np.trace(
        rho
    )

    if abs(
        tr
    ) < 1e-15:
        raise RuntimeError(
            "Zero-trace memory state."
        )

    return rho / tr


def trace_distance(
    rho,
    sigma,
):
    delta = np.asarray(
        rho - sigma,
        dtype=complex,
    )

    delta = 0.5 * (
        delta
        + delta.conj().T
    )

    eigvals = np.linalg.eigvalsh(
        delta
    )

    return float(
        0.5
        * np.sum(
            np.abs(
                eigvals
            )
        )
    )


def max_pairwise_trace_distance(
    states,
):
    return float(
        max(
            trace_distance(
                states[
                    a
                ],
                states[
                    b
                ],
            )
            for a, b
            in combinations(
                states.keys(),
                2,
            )
        )
    )


def feature_vector(
    rho_i,
    rho_m,
):
    row = common.reduced_feature_row(
        rho_i,
        rho_m,
    )

    return np.asarray(
        [
            float(
                row[
                    name
                ]
            )
            for name
            in FEATURES
        ],
        dtype=float,
    )


def max_pairwise_feature_distance(
    features,
):
    return float(
        max(
            np.linalg.norm(
                features[
                    a
                ]
                - features[
                    b
                ]
            )
            for a, b
            in combinations(
                features.keys(),
                2,
            )
        )
    )


def stable_threshold_crossing(
    values,
    epsilon,
):
    arr = np.asarray(
        values,
        dtype=float,
    )

    suffix_max = np.maximum.accumulate(
        arr[
            ::-1
        ]
    )[
        ::-1
    ]

    idx = np.flatnonzero(
        suffix_max
        <= float(
            epsilon
        )
    )

    return (
        None
        if len(
            idx
        ) == 0
        else int(
            idx[
                0
            ]
        )
    )


def contraction_fit(
    t,
    d,
):
    t = np.asarray(
        t,
        dtype=float,
    )

    d = np.asarray(
        d,
        dtype=float,
    )

    keep = (
        np.isfinite(
            d
        )
        & (
            d
            > 1e-14
        )
    )

    t = t[
        keep
    ]

    d = d[
        keep
    ]

    if len(
        t
    ) < 3:
        return {
            "slope": None,
            "intercept": None,
            "kappa_eff": None,
            "tau_mem_steps": None,
            "r2_log_fit": None,
        }

    y = np.log(
        d
    )

    slope, intercept = np.polyfit(
        t,
        y,
        deg=1,
    )

    pred = (
        intercept
        + slope
        * t
    )

    ss_res = float(
        np.sum(
            (
                y
                - pred
            )
            ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                y
                - np.mean(
                    y
                )
            )
            ** 2
        )
    )

    r2 = (
        None
        if ss_tot <= 0
        else float(
            1.0
            - ss_res
            / ss_tot
        )
    )

    kappa = float(
        np.exp(
            slope
        )
    )

    tau = (
        None
        if not (
            0.0
            < kappa
            < 1.0
        )
        else float(
            -1.0
            / np.log(
                kappa
            )
        )
    )

    return {
        "slope": float(
            slope
        ),
        "intercept": float(
            intercept
        ),
        "kappa_eff": (
            kappa
        ),
        "tau_mem_steps": (
            tau
        ),
        "r2_log_fit": (
            r2
        ),
    }


def run_global_washout(
    candidate,
    angles,
):
    A_list = common.build_channels(
        candidate,
        angles,
    )

    states = {
        name: rho.copy()
        for name, rho
        in INITIAL_STATES.items()
    }

    rows = []

    for t, A_t in enumerate(
        A_list
    ):
        next_states = {}
        features = {}

        for name, rho_before in states.items():
            rho_i, rho_m = (
                qrc.final_reduced_states(
                    A_t,
                    rho_before,
                )
            )

            rho_m = normalize_density(
                rho_m
            )

            next_states[
                name
            ] = rho_m

            features[
                name
            ] = feature_vector(
                rho_i,
                rho_m,
            )

        rows.append(
            {
                "t": int(
                    t
                ),
                "split": (
                    "train"
                    if t
                    < common.N_TRAIN
                    else "validation"
                ),
                "max_pairwise_trace_distance": (
                    max_pairwise_trace_distance(
                        next_states
                    )
                ),
                "max_pairwise_feature_distance": (
                    max_pairwise_feature_distance(
                        features
                    )
                ),
            }
        )

        states = next_states

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print(
        "=" * 126
    )
    print(
        "WEEK 11.1E.1 — CANDIDATE #5 TRACE-DISTANCE WASHOUT"
    )
    print(
        "=" * 126
    )
    print(
        "NO QPU JOB WILL BE SUBMITTED."
    )
    print(
        "2026 = FROZEN / NOT LOADED"
    )
    print()

    candidate, manifest_meta = (
        load_candidate()
    )

    print(
        "Resolved frozen Candidate #5:"
    )
    print(
        f"  candidate={candidate['candidate_id']}"
    )
    print(
        f"  protocol={candidate['protocol']}"
    )
    print(
        f"  topology={candidate['topology']}"
    )
    print(
        f"  r={candidate['r']}"
    )
    print(
        f"  alpha={candidate['alpha']}"
    )
    print(
        f"  dt={candidate['dt']}"
    )
    print(
        f"  hx={candidate['hx']:+.12f}"
    )
    print(
        f"  hy={candidate['hy']:+.12f}"
    )
    print(
        f"  readout={EXPECTED_READOUT}"
    )
    print(
        "  J="
        + json.dumps(
            {
                f"J{i}{j}": float(v)
                for (
                    i,
                    j,
                ), v
                in candidate[
                    "J"
                ].items()
            },
            sort_keys=True,
        )
    )
    print()

    (
        work_tv,
        cols,
        _,
    ) = common.load_train_validation()

    angles = common.make_angles(
        work_tv,
        cols,
        float(
            candidate[
                "alpha"
            ]
        ),
    )

    timewise = run_global_washout(
        candidate,
        angles,
    )

    timewise_path = (
        RESULTS
        / "11_1E1_candidate5_washout_timewise.csv"
    )

    timewise.to_csv(
        timewise_path,
        index=False,
    )

    rows = []

    for epsilon in THRESHOLDS:
        trace_t = stable_threshold_crossing(
            timewise[
                "max_pairwise_trace_distance"
            ],
            epsilon,
        )

        feature_t = stable_threshold_crossing(
            timewise[
                "max_pairwise_feature_distance"
            ],
            epsilon,
        )

        rows.append(
            {
                "epsilon": float(
                    epsilon
                ),
                "trace_distance_stable_washout_t": (
                    np.nan
                    if trace_t is None
                    else int(
                        trace_t
                    )
                ),
                "trace_distance_stable_washout_steps": (
                    np.nan
                    if trace_t is None
                    else int(
                        trace_t
                        + 1
                    )
                ),
                "feature_distance_stable_washout_t": (
                    np.nan
                    if feature_t is None
                    else int(
                        feature_t
                    )
                ),
                "feature_distance_stable_washout_steps": (
                    np.nan
                    if feature_t is None
                    else int(
                        feature_t
                        + 1
                    )
                ),
                "trace_threshold_achieved": (
                    trace_t
                    is not None
                ),
                "feature_threshold_achieved": (
                    feature_t
                    is not None
                ),
            }
        )

    thresholds_df = pd.DataFrame(
        rows
    )

    threshold_path = (
        RESULTS
        / "11_1E1_candidate5_washout_thresholds.csv"
    )

    thresholds_df.to_csv(
        threshold_path,
        index=False,
    )

    fit = contraction_fit(
        timewise[
            "t"
        ],
        timewise[
            "max_pairwise_trace_distance"
        ],
    )

    print(
        "Trace-distance / feature-space washout:"
    )
    print(
        thresholds_df.to_string(
            index=False
        )
    )
    print()

    print(
        "Exponential contraction diagnostic:"
    )
    print(
        f"  kappa_eff={fit['kappa_eff']}"
    )
    print(
        f"  tau_mem_steps={fit['tau_mem_steps']}"
    )
    print(
        f"  R^2(log fit)={fit['r2_log_fit']}"
    )
    print()

    eps_row = thresholds_df[
        np.isclose(
            thresholds_df[
                "epsilon"
            ],
            0.01,
        )
    ]

    if len(
        eps_row
    ) != 1:
        raise RuntimeError(
            "Could not resolve epsilon=0.01 row."
        )

    eps_row = eps_row.iloc[
        0
    ]

    trace_tw = (
        None
        if pd.isna(
            eps_row[
                "trace_distance_stable_washout_steps"
            ]
        )
        else int(
            eps_row[
                "trace_distance_stable_washout_steps"
            ]
        )
    )

    feature_tw = (
        None
        if pd.isna(
            eps_row[
                "feature_distance_stable_washout_steps"
            ]
        )
        else int(
            eps_row[
                "feature_distance_stable_washout_steps"
            ]
        )
    )

    if (
        trace_tw is not None
        and feature_tw is not None
    ):
        proposed_K = max(
            trace_tw,
            feature_tw,
        )
    else:
        proposed_K = None

    print(
        "PRIMARY EPSILON = 0.01"
    )
    print(
        f"  global trace washout steps="
        f"{trace_tw}"
    )
    print(
        f"  global feature washout steps="
        f"{feature_tw}"
    )
    print(
        f"  proposed local-audit center K="
        f"{proposed_K}"
    )
    print()

    # A centered local grid for the NEXT step.
    local_grid = None

    if proposed_K is not None:
        offsets = [
            -24,
            -16,
            -8,
            -4,
            -1,
            0,
            1,
            4,
            8,
            16,
            32,
        ]

        local_grid = sorted(
            {
                max(
                    1,
                    int(
                        proposed_K
                        + offset
                    ),
                )
                for offset
                in offsets
            }
        )

        print(
            "Suggested Candidate #5 local-window K grid:"
        )
        print(
            f"  {local_grid}"
        )
        print()

    summary = {
        "candidate_key": (
            candidate[
                "candidate_id"
            ]
        ),
        "candidate_label": (
            CANDIDATE_LABEL
        ),
        "protocol": (
            EXPECTED_PROTOCOL
        ),
        "topology": (
            EXPECTED_TOPOLOGY
        ),
        "r": (
            EXPECTED_R
        ),
        "readout": (
            EXPECTED_READOUT
        ),
        "features": (
            FEATURES
        ),
        "planned_shots_per_setting": (
            PLANNED_SHOTS
        ),
        "alpha": float(
            candidate[
                "alpha"
            ]
        ),
        "dt": float(
            candidate[
                "dt"
            ]
        ),
        "hx": float(
            candidate[
                "hx"
            ]
        ),
        "hy": float(
            candidate[
                "hy"
            ]
        ),
        "J": {
            f"J{i}{j}": float(v)
            for (
                i,
                j,
            ), v
            in candidate[
                "J"
            ].items()
        },
        "initial_states": list(
            INITIAL_STATES.keys()
        ),
        "thresholds": (
            thresholds_df.to_dict(
                orient="records"
            )
        ),
        "contraction_fit": (
            fit
        ),
        "primary_epsilon": (
            0.01
        ),
        "global_trace_washout_steps": (
            trace_tw
        ),
        "global_feature_washout_steps": (
            feature_tw
        ),
        "proposed_local_audit_center_K": (
            proposed_K
        ),
        "suggested_local_K_grid": (
            local_grid
        ),
        "measurement_backaction_included": False,
        "2026_loaded": False,
    }

    summary_path = (
        RESULTS
        / "11_1E1_candidate5_washout_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            db.json_safe(
                summary
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Database persistence.
    # --------------------------------------------------------------
    db.create_all_tables()

    run_uuid = db.create_run(
        script_name=(
            Path(
                __file__
            ).name
        ),
        run_status=(
            "WASHOUT_AUDIT_COMPLETE"
        ),
        started_at_utc=(
            db.utc_now_naive()
        ),
        completed_at_utc=(
            db.utc_now_naive()
        ),
        forecast_dataset_name=(
            "property_damage_next_day_v1"
        ),
        feature_set_name="F4",
        evaluation_split=(
            "trainval_2022_2025"
        ),
        candidate_key=(
            candidate[
                "candidate_id"
            ]
        ),
        selected_by_rules=(
            norm_text(
                manifest_meta.get(
                    "selected_by_rules",
                    "Rule2",
                )
            )
            or "Rule2"
        ),
        protocol=(
            EXPECTED_PROTOCOL
        ),
        topology=(
            EXPECTED_TOPOLOGY
        ),
        window_size=(
            proposed_K
        ),
        trotter_r=(
            EXPECTED_R
        ),
        readout_name=(
            EXPECTED_READOUT
        ),
        primitive_name=(
            "ideal_simulation"
        ),
        measurement_method=(
            "global_trace_distance_washout"
        ),
        alpha=float(
            candidate[
                "alpha"
            ]
        ),
        dt=float(
            candidate[
                "dt"
            ]
        ),
        hx=float(
            candidate[
                "hx"
            ]
        ),
        hy=float(
            candidate[
                "hy"
            ]
        ),
        j_json=(
            summary[
                "J"
            ]
        ),
        manifest_json=(
            manifest_meta
        ),
        shots_per_setting=(
            PLANNED_SHOTS
        ),
        notes=(
            "Candidate #5 global mathematical washout audit. "
            "Five initial memory states, same full chronological F4 "
            "input stream, no intermediate measurement. Readout-space "
            "distance uses the frozen XZinj feature set. No QPU job."
        ),
    )

    db.save_dataframe_artifact(
        run_uuid,
        timewise_path.name,
        timewise,
    )

    db.save_dataframe_artifact(
        run_uuid,
        threshold_path.name,
        thresholds_df,
    )

    db.save_json_artifact(
        run_uuid,
        summary_path.name,
        summary,
    )

    print(
        f"[database] Candidate #5 washout run UUID: "
        f"{run_uuid}"
    )
    print()

    print(
        "Saved:"
    )
    print(
        f"  {timewise_path}"
    )
    print(
        f"  {threshold_path}"
    )
    print(
        f"  {summary_path}"
    )
    print()

    print(
        "STOP HERE. Send me the washout threshold table."
    )
    print(
        "The next step will be the same 365-endpoint local-window "
        "washout validation used for Candidate #4."
    )


if __name__ == "__main__":
    main()
