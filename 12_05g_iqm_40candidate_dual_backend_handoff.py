"""
Week 12.5G1 v2 — Unified-candidate dual-backend IQM hardware handoff audit.

Purpose
-------
Compile the COMPLETE unified frozen logical candidate universe produced by 12.5G0 v3

onto the three retained Rule-5 physical layouts for BOTH IQM backends:

    emerald
    garnet

This is the final dual-backend evidence stage BEFORE selecting one IQM QPU.

What this step does
-------------------
1. Loads the unified H0-H6 CONT/RWP R1-R4 candidate universe from Week 12.5G0 v3.
2. Loads the 3 retained Rule-5 layouts per (backend, topology) from Week 12.5F.
3. Reconstructs a STRUCTURAL QRC circuit using each candidate's:
       topology
       protocol
       replay window W
       Trotter depth r
       readout-setting count
4. Compiles against backend.get_real_target() with:
       optimization_level = 1
       routing_method = "none"
       frozen initial_layout
5. Demands zero SWAPs.
6. Records native CZ count, depth, setting count, and raw IQM Rule-5 properties.
7. Produces backend-level evidence for the NEXT step (12.5H), where exactly
   one IQM backend will be selected.

Important methodological boundary
---------------------------------
This is a STRUCTURAL hardware handoff audit, not a forecasting evaluation.

No model is refit.
No candidate is re-selected.
No 2025/2026 prediction is computed.
No simulator/QPU execution occurs.
No p_depol or synthetic noise parameter is calculated.

The structural circuit intentionally uses generic non-zero angles so Qiskit
cannot eliminate topology edges. Hardware-relevant connectivity and native CZ
resource counts depend on topology/r/W/settings, not on the exact numerical
Hamiltonian coefficients.

Raw IQM physical properties are copied directly from 12.5F:
    T1
    T2_echo
    CZ fidelity
    asymmetric readout error
    PRX fidelity
    PRX duration
    CZ duration

Outputs
-------
results/12_05g1_iqm_candidate_compile_detail.csv
results/12_05g1_iqm_candidate_layout_summary.csv
results/12_05g1_iqm_backend_comparison.csv
results/12_05g1_iqm_backend_topology_comparison.csv
results/12_05g1_iqm_handoff_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from iqm.qiskit_iqm.iqm_provider import IQMProvider

from iqm_account import get_authentication


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

IN_CANDIDATES = RESULTS / "12_05g0_unified_rule1_rule4_candidates.csv"
IN_COVERAGE = RESULTS / "12_05g0_rule_coverage_audit.csv"
IN_TOP3 = RESULTS / "12_05f_iqm_rule5_selected_top3.csv"
IN_DISCOVERY = RESULTS / "12_01_iqm_discovery.json"

OUT_DETAIL = RESULTS / "12_05g1_iqm_candidate_compile_detail.csv"
OUT_LAYOUT = RESULTS / "12_05g1_iqm_candidate_layout_summary.csv"
OUT_BACKEND = RESULTS / "12_05g1_iqm_backend_comparison.csv"
OUT_TOPO = RESULTS / "12_05g1_iqm_backend_topology_comparison.csv"
OUT_MANIFEST = RESULTS / "12_05g1_iqm_handoff_manifest.json"

BACKENDS = ("emerald", "garnet")
TOPOLOGIES = tuple(f"H{i}" for i in range(7))

OPT_LEVEL = 1
SEED_TRANSPILE = 79001

EXPECTED_LAYOUTS_PER_BACKEND_TOPOLOGY = 3

TOPOLOGY_EDGES = {
    "H0": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
    "H1": ((0, 1), (1, 2), (2, 3), (3, 4), (3, 5)),
    "H2": ((0, 1), (1, 2), (2, 3), (2, 4), (4, 5)),
    "H3": ((0, 1), (0, 4), (1, 2), (2, 3), (4, 5)),
    "H4": ((0, 1), (0, 4), (1, 2), (3, 4), (4, 5)),
    "H5": ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (4, 5)),
    "H6": ((0, 1), (0, 4), (1, 5), (2, 3), (2, 5), (3, 4), (4, 5)),
}


# =============================================================================
# HELPERS
# =============================================================================

def finite(x):
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def resolve_protocol(row):
    """Require an explicit protocol from the unified 12.5G0 manifest.

    No protocol inference is allowed in this handoff. If 12.5G0 did not
    produce explicit CONT/RWP, the unified selection stage is incomplete.
    """
    raw = row.get("protocol", None)
    value = str(raw).strip().upper()

    if value not in ("CONT", "RWP"):
        raise ValueError(
            "12.5G1 requires explicit CONT/RWP protocol from the unified "
            f"manifest; got {raw!r} for candidate "
            f"{row.get('candidate_id', '<unknown>')!r}."
        )

    return value, "explicit_unified_manifest"


def normalize_protocol(value):
    """Strict helper for already-resolved protocol strings."""
    value = str(value).strip().upper()

    if value not in ("CONT", "RWP"):
        raise ValueError(
            f"Unsupported protocol {value!r}"
        )

    return value


def normalize_readout(value):
    return str(value).strip()


def candidate_window(row):
    protocol, _ = resolve_protocol(
        row
    )

    if protocol == "CONT":
        return 1

    value = finite(
        row["window"]
    )

    if value is None:
        raise ValueError(
            f"RWP candidate {row['candidate_id']} has no window."
        )

    w = int(round(value))

    if w < 1:
        raise ValueError(
            f"Invalid W={w} for {row['candidate_id']}"
        )

    return w


def candidate_r(row):
    value = finite(
        row["r"]
    )

    if value is None:
        raise ValueError(
            f"Candidate {row['candidate_id']} has no Trotter r."
        )

    r = int(round(value))

    if r < 1:
        raise ValueError(
            f"Invalid r={r} for {row['candidate_id']}"
        )

    return r


def readout_settings(readout):
    """
    Structural measurement groups used by the frozen study.

    XYZ_all uses 3 grouped bases.
    XZ-family readouts use 2 grouped measurement circuits, including
    +YX45 variants (YX45 is accommodated in the second group).
    """
    name = normalize_readout(readout).lower()

    if "xyz" in name:
        return ("X", "Y", "Z")

    return ("A", "B")


def parse_layout_text(raw):
    raw = str(raw).strip()

    if not (
        raw.startswith("[")
        and raw.endswith("]")
    ):
        raise ValueError(
            f"Cannot parse layout {raw!r}"
        )

    vals = [
        x.strip().strip("'\"")
        for x in raw[1:-1].split(",")
        if x.strip()
    ]

    if len(vals) != 6:
        raise ValueError(
            f"Expected 6 physical qubits in {raw!r}"
        )

    return vals


def physical_index_map(backend):
    return {
        str(name): int(i)
        for i, name
        in enumerate(
            backend.physical_qubits
        )
    }


def layout_to_indices(layout_names, backend):
    mapping = physical_index_map(
        backend
    )

    missing = [
        q
        for q in layout_names
        if q not in mapping
    ]

    if missing:
        raise RuntimeError(
            f"Physical labels not found on {backend.name}: {missing}"
        )

    return [
        mapping[q]
        for q in layout_names
    ]


def count_native_twoq(circuit):
    """
    IQM real target normally emits CZ as the entangling operation.
    Keep a generic 2Q counter as an audit in case target naming changes.
    """
    n2q = 0

    for inst in circuit.data:
        try:
            nq = int(
                inst.operation.num_qubits
            )
        except Exception:
            nq = len(
                getattr(
                    inst,
                    "qubits",
                    [],
                )
            )

        if nq == 2:
            n2q += 1

    return n2q


# =============================================================================
# STRUCTURAL QRC CIRCUITS
# =============================================================================

def add_generic_input(qc, replay_step):
    """
    Generic non-zero F4 input encoding.

    Numerical values are intentionally arbitrary but non-zero.
    This audit tests structure/resources only.
    """
    base = (
        0.173
        + 0.011 * (
            replay_step % 7
        )
    )

    for q in range(4):
        angle = (
            base
            * (q + 1)
        )
        qc.ry(
            angle,
            q,
        )


def add_trotter_slice(
    qc,
    logical_edges,
    replay_step,
    trotter_slice,
):
    """
    Generic non-zero Trotter slice with the same structural ingredients
    as the QRC Hamiltonian:
        ZZ on topology edges
        X field on all six qubits
        Y memory field on q4/q5

    Angles vary slightly to prevent accidental cancellation.
    """
    for k, (a, b) in enumerate(
        logical_edges
    ):
        theta = (
            0.137
            + 0.003 * (
                k + 1
            )
            + 0.0007
            * replay_step
            + 0.0003
            * trotter_slice
        )

        qc.rzz(
            theta,
            int(a),
            int(b),
        )

    for q in range(6):
        qc.rx(
            0.211
            + 0.002 * q
            + 0.0005
            * trotter_slice,
            q,
        )

    qc.ry(
        0.089
        + 0.0004
        * replay_step,
        4,
    )

    qc.ry(
        -0.073
        - 0.0004
        * replay_step,
        5,
    )


def add_measurement_basis(
    qc,
    setting,
):
    """
    Structural grouped readout.

    Exact Pauli-feature extraction is NOT evaluated in this step.
    These basis changes reproduce the number of measurement groups while
    keeping each circuit executable.
    """
    if setting == "X":
        for q in range(6):
            qc.h(q)

    elif setting == "Y":
        for q in range(6):
            qc.sdg(q)
            qc.h(q)

    elif setting == "Z":
        pass

    elif setting == "A":
        # X-like group for injection/readout observables.
        for q in (0, 1, 2):
            qc.h(q)

    elif setting == "B":
        # Complementary group; include memory-basis rotations so +YX45
        # families remain structurally represented.
        for q in (4, 5):
            qc.sdg(q)
            qc.h(q)

    else:
        raise ValueError(
            f"Unknown structural setting {setting!r}"
        )


def build_structural_feature_circuit(
    topology,
    protocol,
    window,
    r,
    setting,
):
    qc = QuantumCircuit(
        6,
        6,
    )

    logical_edges = (
        TOPOLOGY_EDGES[
            topology
        ]
    )

    effective_w = (
        1
        if protocol == "CONT"
        else int(window)
    )

    for replay_step in range(
        effective_w
    ):
        add_generic_input(
            qc,
            replay_step,
        )

        for trotter_slice in range(
            int(r)
        ):
            add_trotter_slice(
                qc,
                logical_edges,
                replay_step,
                trotter_slice,
            )

    add_measurement_basis(
        qc,
        setting,
    )

    for q in range(6):
        qc.measure(
            q,
            q,
        )

    return qc


# =============================================================================
# INPUT AUDITS
# =============================================================================

def load_inputs():
    for path in (
        IN_CANDIDATES,
        IN_COVERAGE,
        IN_TOP3,
        IN_DISCOVERY,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    cand = pd.read_csv(
        IN_CANDIDATES
    )

    top3 = pd.read_csv(
        IN_TOP3
    )

    with open(
        IN_DISCOVERY,
        "r",
        encoding="utf-8",
    ) as f:
        discovery = json.load(f)

    if cand.empty:
        raise RuntimeError("Unified candidate universe is empty.")

    if cand["candidate_id"].isna().any():
        raise RuntimeError("Unified candidate manifest contains missing candidate_id values.")

    if cand["candidate_id"].duplicated().any():
        dup = cand.loc[cand["candidate_id"].duplicated(keep=False), "candidate_id"].tolist()
        raise RuntimeError(f"Unified candidate_id values are not unique: {dup}")

    # The preceding unified-selection stage must have passed all 14 cells.
    coverage = pd.read_csv(IN_COVERAGE)
    expected_cells = {(f"H{i}", p) for i in range(7) for p in ("CONT", "RWP")}

    if "pass" not in coverage.columns:
        raise RuntimeError(
            "12.5G0 v3 coverage audit has no 'pass' column."
        )

    pass_mask = (
        coverage["pass"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    observed_cells = {
        (str(r["topology"]), str(r["protocol"]).upper())
        for _, r in coverage.loc[pass_mask].iterrows()
    }

    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise RuntimeError(
            "12.5G0 v3 unified R1-R4 coverage has not passed all 14 cells. "
            f"Missing/failed: {missing}; unexpected: {extra}"
        )

    if "missing_rules" in coverage.columns:
        unresolved = coverage[
            coverage["missing_rules"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ]

        if not unresolved.empty:
            raise RuntimeError(
                "12.5G0 v3 coverage audit still reports missing rules:\n"
                + unresolved.to_string(index=False)
            )

    # Every candidate must now have an explicit protocol.
    protocols = cand["protocol"].astype(str).str.upper()
    bad_protocol = ~protocols.isin(["CONT", "RWP"])
    if bad_protocol.any():
        bad = cand.loc[bad_protocol, ["candidate_id", "topology", "protocol"]]
        raise RuntimeError(
            "Unified candidate manifest contains non-explicit protocol rows:\n"
            + bad.to_string(index=False)
        )
    cand["protocol"] = protocols

    manifest_cells = set(
        zip(
            cand["topology"].astype(str),
            cand["protocol"].astype(str),
        )
    )

    missing_manifest_cells = sorted(
        expected_cells - manifest_cells
    )

    if missing_manifest_cells:
        raise RuntimeError(
            "Unified candidate manifest itself is missing "
            f"topology/protocol cells: {missing_manifest_cells}"
        )

    # Candidate topology audit.
    unknown_topologies = sorted(
        set(
            cand[
                "topology"
            ].astype(str)
        )
        - set(TOPOLOGIES)
    )

    if unknown_topologies:
        raise RuntimeError(
            f"Unknown candidate topologies: {unknown_topologies}"
        )

    # Three layouts per backend/topology.
    counts = (
        top3.groupby(
            [
                "backend",
                "topology",
            ]
        )
        .size()
    )

    for backend in BACKENDS:
        for topology in TOPOLOGIES:
            key = (
                backend,
                topology,
            )

            if int(
                counts.get(
                    key,
                    0,
                )
            ) != EXPECTED_LAYOUTS_PER_BACKEND_TOPOLOGY:
                raise RuntimeError(
                    f"Expected 3 Rule-5 layouts for {key}, "
                    f"found {counts.get(key, 0)}."
                )

    return cand, top3, discovery


# =============================================================================
# COMPILE
# =============================================================================

def make_backend(
    backend_name,
    discovery,
    token,
    server_url,
):
    system = (
        discovery[
            "systems"
        ][backend_name]
    )

    calibration_set_id = (
        system[
            "summary"
        ][
            "calibration_set_id"
        ]
    )

    provider = IQMProvider(
        server_url,
        quantum_computer=backend_name,
        token=token,
    )

    backend = provider.get_backend(
        calibration_set_id=UUID(
            calibration_set_id
        ),
        use_metrics=False,
    )

    real_target = (
        backend.get_real_target()
    )

    return (
        backend,
        real_target,
        calibration_set_id,
    )


def compile_candidate_layout(
    candidate_row,
    physical_row,
    backend,
    real_target,
):
    candidate_id = str(
        candidate_row[
            "candidate_id"
        ]
    )

    topology = str(
        candidate_row[
            "topology"
        ]
    )

    protocol, protocol_source = resolve_protocol(
        candidate_row
    )

    window = candidate_window(
        candidate_row
    )

    r = candidate_r(
        candidate_row
    )

    readout = normalize_readout(
        candidate_row[
            "readout"
        ]
    )

    settings = readout_settings(
        readout
    )

    layout_names = parse_layout_text(
        physical_row[
            "layout"
        ]
    )

    layout_indices = (
        layout_to_indices(
            layout_names,
            backend,
        )
    )

    setting_rows = []

    for setting in settings:
        logical = (
            build_structural_feature_circuit(
                topology=topology,
                protocol=protocol,
                window=window,
                r=r,
                setting=setting,
            )
        )

        compile_error = None
        tqc = None

        try:
            tqc = transpile(
                logical,
                target=real_target,
                initial_layout=layout_indices,
                routing_method="none",
                optimization_level=OPT_LEVEL,
                seed_transpiler=SEED_TRANSPILE,
            )

        except Exception as exc:
            compile_error = repr(
                exc
            )

        if tqc is None:
            setting_rows.append({
                "candidate_id":
                    candidate_id,
                "source_group":
                    candidate_row[
                        "source_group"
                    ],
                "selected_rules":
                    candidate_row[
                        "selected_rules"
                    ],
                "topology":
                    topology,
                "protocol":
                    protocol,
                "protocol_source":
                    protocol_source,
                "window_effective":
                    window,
                "r":
                    r,
                "readout":
                    readout,
                "backend":
                    physical_row[
                        "backend"
                    ],
                "rule5_rank":
                    int(
                        physical_row[
                            "rule5_rank_within_top3"
                        ]
                    ),
                "layout":
                    physical_row[
                        "layout"
                    ],
                "setting":
                    setting,
                "compile_pass":
                    False,
                "compile_error":
                    compile_error,
                "n_cz":
                    np.nan,
                "n_2q":
                    np.nan,
                "n_swap":
                    np.nan,
                "depth":
                    np.nan,
                "size":
                    np.nan,
            })
            continue

        ops = {
            str(k): int(v)
            for k, v
            in tqc.count_ops().items()
        }

        n_swap = int(
            ops.get(
                "swap",
                0,
            )
        )

        setting_rows.append({
            "candidate_id":
                candidate_id,
            "source_group":
                candidate_row[
                    "source_group"
                ],
            "selected_rules":
                candidate_row[
                    "selected_rules"
                ],
            "topology":
                topology,
            "protocol":
                protocol,
            "protocol_source":
                protocol_source,
            "window_effective":
                window,
            "r":
                r,
            "readout":
                readout,
            "backend":
                physical_row[
                    "backend"
                ],
            "rule5_rank":
                int(
                    physical_row[
                        "rule5_rank_within_top3"
                    ]
                ),
            "layout":
                physical_row[
                    "layout"
                ],
            "setting":
                setting,
            "compile_pass":
                True,
            "compile_error":
                None,
            "n_cz":
                int(
                    ops.get(
                        "cz",
                        0,
                    )
                ),
            "n_2q":
                int(
                    count_native_twoq(
                        tqc
                    )
                ),
            "n_swap":
                n_swap,
            "depth":
                int(
                    tqc.depth()
                ),
            "size":
                int(
                    tqc.size()
                ),
            "ops_json":
                json.dumps(
                    ops,
                    sort_keys=True,
                ),
        })

    return setting_rows


# =============================================================================
# SUMMARIES
# =============================================================================

def build_layout_summary(
    detail,
    top3,
):
    keys = [
        "candidate_id",
        "source_group",
        "selected_rules",
        "topology",
        "protocol",
        "protocol_source",
        "window_effective",
        "r",
        "readout",
        "backend",
        "rule5_rank",
        "layout",
    ]

    summary_rows = []

    for key_values, g in detail.groupby(
        keys,
        dropna=False,
        sort=False,
    ):
        row = dict(
            zip(
                keys,
                key_values,
            )
        )

        pass_all = bool(
            g[
                "compile_pass"
            ].astype(bool).all()
        )

        zero_swap_all = bool(
            pass_all
            and (
                pd.to_numeric(
                    g[
                        "n_swap"
                    ],
                    errors="coerce",
                )
                .fillna(999999)
                .eq(0)
                .all()
            )
        )

        row.update({
            "n_settings":
                int(
                    g[
                        "setting"
                    ].nunique()
                ),
            "compile_all_settings_pass":
                pass_all,
            "zero_swap_all_settings":
                zero_swap_all,
            "NCZ_feature_vector":
                (
                    float(
                        pd.to_numeric(
                            g[
                                "n_cz"
                            ],
                            errors="coerce",
                        ).sum()
                    )
                    if pass_all
                    else np.nan
                ),
            "N2q_feature_vector":
                (
                    float(
                        pd.to_numeric(
                            g[
                                "n_2q"
                            ],
                            errors="coerce",
                        ).sum()
                    )
                    if pass_all
                    else np.nan
                ),
            "NSWAP_feature_vector":
                (
                    float(
                        pd.to_numeric(
                            g[
                                "n_swap"
                            ],
                            errors="coerce",
                        ).sum()
                    )
                    if pass_all
                    else np.nan
                ),
            "depth_max_setting":
                (
                    float(
                        pd.to_numeric(
                            g[
                                "depth"
                            ],
                            errors="coerce",
                        ).max()
                    )
                    if pass_all
                    else np.nan
                ),
            "depth_sum_settings":
                (
                    float(
                        pd.to_numeric(
                            g[
                                "depth"
                            ],
                            errors="coerce",
                        ).sum()
                    )
                    if pass_all
                    else np.nan
                ),
            "compile_errors":
                "; ".join(
                    sorted(
                        {
                            str(x)
                            for x in g[
                                "compile_error"
                            ].dropna()
                            if str(x)
                        }
                    )
                ),
        })

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # Attach RAW Rule-5 metrics for the exact physical layout.
    physical_cols = [
        "backend",
        "topology",
        "rule5_rank_within_top3",
        "layout",
        "min_t1_us",
        "mean_t1_us",
        "min_t2_echo_us",
        "mean_t2_echo_us",
        "memory_min_t1_us",
        "memory_min_t2_echo_us",
        "min_cz_fidelity",
        "mean_cz_fidelity",
        "min_readout_fidelity",
        "max_readout_error_0_to_1",
        "max_readout_error_1_to_0",
        "max_asymmetric_readout_error",
        "min_prx_fidelity",
        "mean_prx_fidelity",
        "prx_duration_min_ns",
        "prx_duration_median_ns",
        "prx_duration_max_ns",
        "cz_duration_min_ns",
        "cz_duration_median_ns",
        "cz_duration_max_ns",
        "raw_duration_complete_prx_cz",
    ]

    physical = top3[
        [
            c
            for c in physical_cols
            if c in top3.columns
        ]
    ].copy()

    physical = physical.rename(
        columns={
            "rule5_rank_within_top3":
                "rule5_rank",
        }
    )

    merged = summary.merge(
        physical,
        on=[
            "backend",
            "topology",
            "rule5_rank",
            "layout",
        ],
        how="left",
        validate="many_to_one",
    )

    return merged


def build_topology_comparison(
    layout_summary,
):
    rows = []

    # Use unique physical layouts, not candidate-weighted repetitions,
    # for raw hardware comparison.
    physical_cols = [
        "backend",
        "topology",
        "rule5_rank",
        "layout",
        "min_t1_us",
        "min_t2_echo_us",
        "memory_min_t2_echo_us",
        "min_cz_fidelity",
        "max_asymmetric_readout_error",
        "min_prx_fidelity",
        "prx_duration_median_ns",
        "cz_duration_median_ns",
    ]

    unique_physical = (
        layout_summary[
            physical_cols
        ]
        .drop_duplicates(
            subset=[
                "backend",
                "topology",
                "rule5_rank",
                "layout",
            ]
        )
    )

    compile_by_topology = (
        layout_summary.groupby(
            [
                "backend",
                "topology",
            ],
            as_index=False,
        )
        .agg(
            candidate_layout_rows=(
                "candidate_id",
                "count",
            ),
            compile_pass_rate=(
                "compile_all_settings_pass",
                "mean",
            ),
            zero_swap_rate=(
                "zero_swap_all_settings",
                "mean",
            ),
            median_NCZ=(
                "NCZ_feature_vector",
                "median",
            ),
            max_NCZ=(
                "NCZ_feature_vector",
                "max",
            ),
            median_depth_max_setting=(
                "depth_max_setting",
                "median",
            ),
            max_depth_max_setting=(
                "depth_max_setting",
                "max",
            ),
        )
    )

    hardware_by_topology = (
        unique_physical.groupby(
            [
                "backend",
                "topology",
            ],
            as_index=False,
        )
        .agg(
            retained_layouts=(
                "layout",
                "nunique",
            ),
            worst_selected_min_t1_us=(
                "min_t1_us",
                "min",
            ),
            median_selected_min_t1_us=(
                "min_t1_us",
                "median",
            ),
            worst_selected_min_t2_echo_us=(
                "min_t2_echo_us",
                "min",
            ),
            median_selected_min_t2_echo_us=(
                "min_t2_echo_us",
                "median",
            ),
            worst_selected_memory_min_t2_echo_us=(
                "memory_min_t2_echo_us",
                "min",
            ),
            worst_selected_min_cz_fidelity=(
                "min_cz_fidelity",
                "min",
            ),
            median_selected_min_cz_fidelity=(
                "min_cz_fidelity",
                "median",
            ),
            worst_selected_max_asym_ro_error=(
                "max_asymmetric_readout_error",
                "max",
            ),
            median_selected_max_asym_ro_error=(
                "max_asymmetric_readout_error",
                "median",
            ),
            worst_selected_min_prx_fidelity=(
                "min_prx_fidelity",
                "min",
            ),
            median_prx_duration_ns=(
                "prx_duration_median_ns",
                "median",
            ),
            median_cz_duration_ns=(
                "cz_duration_median_ns",
                "median",
            ),
        )
    )

    return compile_by_topology.merge(
        hardware_by_topology,
        on=[
            "backend",
            "topology",
        ],
        how="outer",
        validate="one_to_one",
    )


def build_backend_comparison(
    layout_summary,
    topology_summary,
):
    rows = []

    for backend in BACKENDS:
        c = layout_summary[
            layout_summary[
                "backend"
            ].eq(backend)
        ].copy()

        t = topology_summary[
            topology_summary[
                "backend"
            ].eq(backend)
        ].copy()

        # Unique retained physical layouts for backend-level raw hardware.
        unique_p = (
            c[
                [
                    "topology",
                    "rule5_rank",
                    "layout",
                    "min_t1_us",
                    "min_t2_echo_us",
                    "memory_min_t2_echo_us",
                    "min_cz_fidelity",
                    "max_asymmetric_readout_error",
                    "min_prx_fidelity",
                    "prx_duration_median_ns",
                    "cz_duration_median_ns",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "topology",
                    "rule5_rank",
                    "layout",
                ]
            )
        )

        rows.append({
            "backend":
                backend,
            "candidate_layout_rows":
                int(len(c)),
            "unique_candidates":
                int(
                    c[
                        "candidate_id"
                    ].nunique()
                ),
            "topologies":
                int(
                    c[
                        "topology"
                    ].nunique()
                ),
            "retained_physical_layouts":
                int(
                    len(
                        unique_p
                    )
                ),
            "compile_pass_rate":
                float(
                    c[
                        "compile_all_settings_pass"
                    ].astype(float).mean()
                ),
            "zero_swap_rate":
                float(
                    c[
                        "zero_swap_all_settings"
                    ].astype(float).mean()
                ),
            "median_candidate_NCZ":
                float(
                    pd.to_numeric(
                        c[
                            "NCZ_feature_vector"
                        ],
                        errors="coerce",
                    ).median()
                ),
            "max_candidate_NCZ":
                float(
                    pd.to_numeric(
                        c[
                            "NCZ_feature_vector"
                        ],
                        errors="coerce",
                    ).max()
                ),
            "median_candidate_depth_max_setting":
                float(
                    pd.to_numeric(
                        c[
                            "depth_max_setting"
                        ],
                        errors="coerce",
                    ).median()
                ),
            "max_candidate_depth_max_setting":
                float(
                    pd.to_numeric(
                        c[
                            "depth_max_setting"
                        ],
                        errors="coerce",
                    ).max()
                ),
            "worst_retained_min_t1_us":
                float(
                    pd.to_numeric(
                        unique_p[
                            "min_t1_us"
                        ],
                        errors="coerce",
                    ).min()
                ),
            "median_retained_min_t1_us":
                float(
                    pd.to_numeric(
                        unique_p[
                            "min_t1_us"
                        ],
                        errors="coerce",
                    ).median()
                ),
            "worst_retained_min_t2_echo_us":
                float(
                    pd.to_numeric(
                        unique_p[
                            "min_t2_echo_us"
                        ],
                        errors="coerce",
                    ).min()
                ),
            "median_retained_min_t2_echo_us":
                float(
                    pd.to_numeric(
                        unique_p[
                            "min_t2_echo_us"
                        ],
                        errors="coerce",
                    ).median()
                ),
            "worst_retained_min_cz_fidelity":
                float(
                    pd.to_numeric(
                        unique_p[
                            "min_cz_fidelity"
                        ],
                        errors="coerce",
                    ).min()
                ),
            "median_retained_min_cz_fidelity":
                float(
                    pd.to_numeric(
                        unique_p[
                            "min_cz_fidelity"
                        ],
                        errors="coerce",
                    ).median()
                ),
            "worst_retained_max_asym_ro_error":
                float(
                    pd.to_numeric(
                        unique_p[
                            "max_asymmetric_readout_error"
                        ],
                        errors="coerce",
                    ).max()
                ),
            "median_retained_max_asym_ro_error":
                float(
                    pd.to_numeric(
                        unique_p[
                            "max_asymmetric_readout_error"
                        ],
                        errors="coerce",
                    ).median()
                ),
            "worst_retained_min_prx_fidelity":
                float(
                    pd.to_numeric(
                        unique_p[
                            "min_prx_fidelity"
                        ],
                        errors="coerce",
                    ).min()
                ),
            "median_raw_prx_duration_ns":
                float(
                    pd.to_numeric(
                        unique_p[
                            "prx_duration_median_ns"
                        ],
                        errors="coerce",
                    ).median()
                ),
            "median_raw_cz_duration_ns":
                float(
                    pd.to_numeric(
                        unique_p[
                            "cz_duration_median_ns"
                        ],
                        errors="coerce",
                    ).median()
                ),
            "all_7_topologies_compile_cleanly":
                bool(
                    len(t) == 7
                    and np.isclose(
                        t[
                            "compile_pass_rate"
                        ].min(),
                        1.0,
                    )
                    and np.isclose(
                        t[
                            "zero_swap_rate"
                        ].min(),
                        1.0,
                    )
                ),
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--backend",
        choices=[
            "all",
            *BACKENDS,
        ],
        default="all",
        help=(
            "Normally leave as all. Single-backend mode is provided only "
            "for debugging; backend selection happens in 12.5H."
        ),
    )

    args = parser.parse_args()

    outputs = (
        OUT_DETAIL,
        OUT_LAYOUT,
        OUT_BACKEND,
        OUT_TOPO,
        OUT_MANIFEST,
    )

    if args.overwrite:
        for path in outputs:
            if path.exists():
                path.unlink()

    candidates, top3, discovery = (
        load_inputs()
    )

    token, server_url, _ = (
        get_authentication()
    )

    backend_names = (
        list(BACKENDS)
        if args.backend == "all"
        else [
            args.backend
        ]
    )

    print("=" * 132)
    print(
        "WEEK 12.5G1 v2 — UNIFIED 47-CANDIDATE DUAL-BACKEND IQM HARDWARE HANDOFF"
    )
    print("=" * 132)
    candidate_count_runtime = int(
        candidates["candidate_id"].nunique()
    )

    print(
        f"Unified frozen logical candidates: {candidate_count_runtime}"
    )
    print(
        "Expected candidate-layout combinations across both backends: "
        f"{candidate_count_runtime * len(backend_names) * EXPECTED_LAYOUTS_PER_BACKEND_TOPOLOGY}"
    )
    print(
        f"Backends: {backend_names}"
    )
    print(
        "Rule-5 layouts: 3 per backend/topology"
    )
    print(
        'Compile: backend.get_real_target(), routing_method="none", optimization_level=1'
    )
    print(
        "No forecasting evaluation. No simulator/QPU jobs. 2026 untouched."
    )
    print()
    print(
        "IMPORTANT: this is a structural compile/resource audit."
    )

    detail_rows = []
    backend_metadata = {}

    for backend_name in backend_names:
        print()
        print("=" * 132)
        print(
            f"BACKEND: {backend_name.upper()}"
        )
        print("=" * 132)

        (
            backend,
            real_target,
            calibration_set_id,
        ) = make_backend(
            backend_name,
            discovery,
            token,
            server_url,
        )

        backend_metadata[
            backend_name
        ] = {
            "calibration_set_id":
                calibration_set_id,
            "physical_qubits":
                len(
                    backend.physical_qubits
                ),
        }

        print(
            f"Calibration set: {calibration_set_id}"
        )
        print(
            f"Physical qubits: {len(backend.physical_qubits)}"
        )

        b_candidates = candidates.copy()

        for idx, candidate in b_candidates.iterrows():
            candidate_id = str(
                candidate[
                    "candidate_id"
                ]
            )
            topology = str(
                candidate[
                    "topology"
                ]
            )

            physical_rows = (
                top3[
                    top3[
                        "backend"
                    ].astype(str).eq(
                        backend_name
                    )
                    & top3[
                        "topology"
                    ].astype(str).eq(
                        topology
                    )
                ]
                .sort_values(
                    "rule5_rank_within_top3"
                )
            )

            resolved_protocol, protocol_source = resolve_protocol(
                candidate
            )

            print(
                f"[{idx + 1:02d}/{len(b_candidates)}] "
                f"{candidate_id} | {topology} | "
                f"{resolved_protocol} [{protocol_source}] | "
                f"W={candidate_window(candidate)} | "
                f"r={candidate_r(candidate)} | "
                f"{candidate['readout']}"
            )

            for _, physical in (
                physical_rows.iterrows()
            ):
                rows = compile_candidate_layout(
                    candidate_row=candidate,
                    physical_row=physical,
                    backend=backend,
                    real_target=real_target,
                )

                detail_rows.extend(
                    rows
                )

                passed = all(
                    bool(r["compile_pass"])
                    for r in rows
                )

                zero_swap = (
                    passed
                    and all(
                        int(r["n_swap"]) == 0
                        for r in rows
                    )
                )

                ncz = (
                    sum(
                        int(r["n_cz"])
                        for r in rows
                    )
                    if passed
                    else None
                )

                print(
                    "    "
                    f"layout#{int(physical['rule5_rank_within_top3'])}: "
                    f"{'PASS' if passed else 'FAIL'} | "
                    f"zeroSWAP={zero_swap} | "
                    f"CZ/feature={ncz}"
                )

    detail = pd.DataFrame(
        detail_rows
    )

    detail.to_csv(
        OUT_DETAIL,
        index=False,
    )

    layout_summary = (
        build_layout_summary(
            detail,
            top3,
        )
    )

    layout_summary.to_csv(
        OUT_LAYOUT,
        index=False,
    )

    topology_summary = (
        build_topology_comparison(
            layout_summary
        )
    )

    topology_summary.to_csv(
        OUT_TOPO,
        index=False,
    )

    backend_summary = (
        build_backend_comparison(
            layout_summary,
            topology_summary,
        )
    )

    backend_summary.to_csv(
        OUT_BACKEND,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Audit totals
    # -------------------------------------------------------------------------
    candidate_count = int(candidates["candidate_id"].nunique())

    if candidate_count != 47:
        raise RuntimeError(
            "12.5G1 v2 expects the passed 12.5G0 v3 unified universe "
            f"with 47 deduplicated candidates; found {candidate_count}. "
            "Do not continue with an older/incomplete manifest."
        )

    expected_rows = (
        candidate_count
        * len(backend_names)
        * EXPECTED_LAYOUTS_PER_BACKEND_TOPOLOGY
    )

    observed_rows = int(
        len(
            layout_summary
        )
    )

    compile_pass_rows = int(
        layout_summary[
            "compile_all_settings_pass"
        ]
        .astype(bool)
        .sum()
    )

    zero_swap_rows = int(
        layout_summary[
            "zero_swap_all_settings"
        ]
        .astype(bool)
        .sum()
    )

    print()
    print("=" * 132)
    print(
        "12.5G1 v2 BACKEND COMPARISON — DESCRIPTIVE ONLY"
    )
    print("=" * 132)

    print(
        backend_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Candidate-layout rows expected: {expected_rows}"
    )
    print(
        f"Candidate-layout rows observed: {observed_rows}"
    )
    print(
        f"All-settings compile PASS:      {compile_pass_rows}/{observed_rows}"
    )
    print(
        f"Zero-SWAP PASS:                 {zero_swap_rows}/{observed_rows}"
    )

    manifest = {
        "stage":
            "12.5G1_v2",
        "purpose":
            (
                "Full unified H0-H6 CONT/RWP dual-backend structural hardware "
                "handoff before selecting one IQM backend."
            ),
        "candidate_universe":
            {
                "source":
                    "results/12_05g0_unified_rule1_rule4_candidates.csv",
                "source_stage":
                    "12.5G0_v3",
                "observed_unique_candidates":
                    candidate_count,
                "expected_current_count":
                    47,
                "required_cells":
                    "H0-H6 x CONT/RWP = 14, each with R1-R4 coverage",
                "coverage_audit_passed":
                    True,
                "protocol_inference":
                    False,
            },
        "backends":
            backend_names,
        "backend_metadata":
            backend_metadata,
        "rule5_layouts_per_backend_topology":
            EXPECTED_LAYOUTS_PER_BACKEND_TOPOLOGY,
        "compile_protocol":
            {
                "target":
                    "backend.get_real_target()",
                "routing_method":
                    "none",
                "optimization_level":
                    OPT_LEVEL,
                "seed_transpiler":
                    SEED_TRANSPILE,
                "structural_only":
                    True,
            },
        "raw_iqm_properties_only":
            True,
        "protocol_resolution": {
            "explicit_protocol_required": True,
            "inference_allowed": False,
            "source": "12.5G0 unified manifest",
            "audit_column": "protocol_source",
        },
        "p_depol_calculated":
            False,
        "synthetic_noise_parameters":
            False,
        "forecasting_data_loaded":
            False,
        "validation_2025_loaded":
            False,
        "test_2026_used":
            False,
        "qpu_jobs":
            0,
        "simulator_jobs":
            0,
        "expected_candidate_layout_rows":
            expected_rows,
        "observed_candidate_layout_rows":
            observed_rows,
        "compile_pass_rows":
            compile_pass_rows,
        "zero_swap_rows":
            zero_swap_rows,
        "backend_selection_performed":
            False,
        "next_step":
            (
                "12.5H: select exactly one IQM backend using the unified "
                "H0-H6 CONT/RWP Rule-5 + strict-compile evidence. Do not "
                "select final five QPU candidates until the backend is frozen."
            ),
    }

    with open(
        OUT_MANIFEST,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("Saved:")
    for path in outputs:
        print(
            f"  {path}"
        )

    print()
    print(
        "STOP HERE. The next step is 12.5H: choose ONE IQM backend "
        "from this unified H0-H6 CONT/RWP comparison before selecting "
        "the final five QPU candidates."
    )


if __name__ == "__main__":
    main()
