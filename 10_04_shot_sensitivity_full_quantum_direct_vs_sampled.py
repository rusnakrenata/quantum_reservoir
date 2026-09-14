#!/usr/bin/env python
"""
10.04 — Shot sensitivity under FULL QUANTUM noise:
        direct expectation values vs finite-shot sampling
==========================================================================

Scientific question
-------------------
For the hardware-aware Trotter candidates r in {1,2,3,4}, how many shots are
needed before finite-shot estimates approach the corresponding FULL-QUANTUM
direct-expectation result?

Two deliberately matched regimes are used:

N5_FULL_QUANTUM_DIRECT
    Backend-calibrated 1Q + 2Q gate noise + T1/T2 thermal relaxation.
    NO readout assignment error.
    NO finite-shot sampling.
    Expectation values are obtained directly with Aer EstimatorV2.

S2_FULL_QUANTUM_SAMPLED
    EXACTLY the same quantum noise model as N5:
        1Q + 2Q gate noise + T1/T2 thermal relaxation.
    NO readout assignment error.
    Pauli expectation values are reconstructed from finite-shot measurement.

This is intentionally NOT S3.  Readout error is switched OFF in both cases so
that the difference

    S2 - N5

isolates finite-shot measurement/sampling under the SAME noisy quantum state.

Trotter depths
--------------
    r in {1,2,3,4}

Shot budgets
------------
    N in {256,512,1024,2048} shots PER measurement setting.

There are two grouped settings, so one forecast feature vector uses:
    2*N circuit shots in total.

Nested-shot design
------------------
For each (r, replicate), the simulator is run ONCE at 2048 shots with
memory=True.  The lower budgets are exact prefixes of that same shot stream:

    256 ⊂ 512 ⊂ 1024 ⊂ 2048.

Thus shot-budget comparisons use common random numbers and are less confounded
by unrelated random seeds.  Three independent 2048-shot replicates are used by
default.

Frozen design
-------------
topology      = H3
protocol      = RWP
window        = 4
alpha         = 0.25
dt            = 1.6
hx, hy, J     = frozen Week-10 reference
readout       = XZinj_dropX3_plus_YX45
physical map  = Week-10.1 Kingston Rule-5 rank-1 layout

For EACH r:
    * ideal features are recomputed,
    * Ridge lambda is selected by TRAINING-ONLY CV,
    * Ridge/scaler are fitted on ideal 2022-2024 features,
    * the fitted readout is frozen for N5 and all S2 shot levels.

We do NOT re-optimize J, hx, hy, alpha, W, topology, readout, or layout.

Calibration
-----------
Kingston is refreshed ONCE at the beginning.  The same calibration-derived
noise model is used for every r and every shot budget in this run.

NO QPU JOBS ARE SUBMITTED.
NO ERROR MITIGATION.
NO READOUT ERROR.
2026 remains untouched.

Required project files
----------------------
10_03b_trotter_depth_r1_to_r5.py
09_04_rwp_common.py
ibm_account.py

results/09_04c_rwp_all_trials.csv
results/10_01_kingston_fresh_top3_embeddings.csv
plus dependencies already required by the 10.03B helper.

Outputs
-------
results/10_04_reference.json
results/10_04_full_quantum_direct.csv
results/10_04_sampled_runs.csv
results/10_04_sampled_summary.csv
results/10_04_feature_metrics.csv
results/10_04_predictions.csv
results/10_04_resource_shot_grid.csv
results/10_04_pareto_front.csv

Full run
--------
python 10_04_shot_sensitivity_full_quantum_direct_vs_sampled.py --sampling-reps 3

Quick plumbing check
--------------------
python 10_04_shot_sensitivity_full_quantum_direct_vs_sampled.py --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from ibm_account import get_service


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

HELPER_FILE = HERE / "10_03b_trotter_depth_r1_to_r5.py"

BACKEND_NAME = "ibm_kingston"

R_VALUES = [1, 2, 3, 4]
SHOT_LEVELS = [256, 512, 1024, 2048]

DEFAULT_REPS = 3
DEFAULT_BATCH = 80
DEFAULT_SIM_SEED = 20260912


# =============================================================================
# IMPORT 10.03B AS A LIBRARY
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required helper not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(
            f"Could not import {path}"
        )

    spec.loader.exec_module(mod)
    return mod


h = load_module(
    HELPER_FILE,
    "qrc_1004_trotter_helper",
)


# =============================================================================
# SMALL HELPERS
# =============================================================================

def rmse(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(
        np.sqrt(
            np.mean(
                (y - p) ** 2
            )
        )
    )


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(
        np.mean(
            np.abs(y - p)
        )
    )


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        len(a) < 2
        or np.std(a) == 0
        or np.std(b) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def json_safe(x):
    if x is None:
        return None

    if isinstance(x, (str, int, float, bool)):
        if (
            isinstance(x, float)
            and not np.isfinite(x)
        ):
            return None
        return x

    if isinstance(x, np.generic):
        return json_safe(
            x.item()
        )

    if isinstance(x, dict):
        return {
            str(k): json_safe(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple, set)):
        return [
            json_safe(v)
            for v in x
        ]

    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            pass

    return str(x)


# =============================================================================
# ONE MAX-SHOT RUN -> ALL NESTED SHOT LEVELS
# =============================================================================

def run_nested_sampled_regime(
    noise_model,
    physical,
    ideal,
    shot_levels,
    batch_size,
    seed,
):
    """
    Run the measured noisy circuits once at max(shot_levels), requesting
    per-shot memory.  Build lower shot budgets from prefixes of the SAME
    shot stream.

    Returns
    -------
    dict[int, np.ndarray]
        shot_count -> feature matrix
    """

    max_shots = int(max(shot_levels))

    simulator = AerSimulator(
        noise_model=noise_model,
        enable_truncation=True,
    )

    circuits, metadata = h.build_bound_measured_circuits(
        physical,
        ideal,
    )

    # counts_by_shot[N][endpoint][setting] = Counter
    counts_by_shot = {
        int(N): {
            int(endpoint): {}
            for endpoint in ideal["val_endpoints"]
        }
        for N in shot_levels
    }

    for start in range(
        0,
        len(circuits),
        int(batch_size),
    ):
        stop = min(
            start + int(batch_size),
            len(circuits),
        )

        sub = circuits[start:stop]
        submeta = metadata[start:stop]

        result = simulator.run(
            sub,
            shots=max_shots,
            memory=True,
            seed_simulator=int(seed + start),
        ).result()

        for local_i, (
            endpoint,
            setting_label,
        ) in enumerate(submeta):
            memory = list(
                result.get_memory(local_i)
            )

            if len(memory) != max_shots:
                raise RuntimeError(
                    "Unexpected memory length: "
                    f"{len(memory)} != {max_shots}"
                )

            for N in shot_levels:
                counts_by_shot[int(N)][
                    int(endpoint)
                ][setting_label] = Counter(
                    memory[: int(N)]
                )

    X_by_shot = {}

    for N in shot_levels:
        rows = []

        for endpoint in ideal["val_endpoints"]:
            pair = counts_by_shot[int(N)][
                int(endpoint)
            ]

            if set(pair) != set(h.SETTINGS):
                raise RuntimeError(
                    f"N={N}: missing setting for "
                    f"endpoint={endpoint}."
                )

            rows.append(
                h.combine_feature_pair(
                    pair["XXXZYX"],
                    pair["ZZZZZZ"],
                )
            )

        X_by_shot[int(N)] = np.asarray(
            rows,
            dtype=float,
        )

    return X_by_shot


# =============================================================================
# SAMPLE-vs-DIRECT METRICS
# =============================================================================

def evaluate_sampled_against_direct(
    r,
    shots,
    replicate,
    X_sampled,
    X_direct,
    direct_pred,
    ideal,
):
    pred = h.common.predict_scaled_ridge(
        ideal["model"],
        ideal["scaler"],
        ideal["keep"],
        X_sampled,
    )

    y = ideal["y_val"]
    ideal_pred = ideal["pred_val_ideal"]

    feature_diff_direct = (
        X_sampled - X_direct
    )

    feature_diff_ideal = (
        X_sampled - ideal["X_val_ideal"]
    )

    row = {
        "r": int(r),
        "shots_per_setting": int(shots),
        "total_circuit_shots_per_feature": int(
            2 * int(shots)
        ),
        "replicate": int(replicate),

        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "bias": float(
            np.mean(pred - y)
        ),

        "direct_full_quantum_rmse": rmse(
            y,
            direct_pred,
        ),

        "delta_rmse_vs_full_quantum_direct": (
            rmse(y, pred)
            - rmse(y, direct_pred)
        ),

        "delta_rmse_vs_same_r_ideal": (
            rmse(y, pred)
            - ideal["ideal_rmse"]
        ),

        "prediction_rmse_vs_full_quantum_direct": rmse(
            direct_pred,
            pred,
        ),

        "prediction_corr_vs_full_quantum_direct": safe_corr(
            direct_pred,
            pred,
        ),

        "prediction_rmse_vs_same_r_ideal": rmse(
            ideal_pred,
            pred,
        ),

        "prediction_corr_vs_same_r_ideal": safe_corr(
            ideal_pred,
            pred,
        ),

        "feature_rmse_vs_full_quantum_direct": float(
            np.sqrt(
                np.mean(
                    feature_diff_direct ** 2
                )
            )
        ),

        "feature_mae_vs_full_quantum_direct": float(
            np.mean(
                np.abs(
                    feature_diff_direct
                )
            )
        ),

        "feature_rmse_vs_same_r_ideal": float(
            np.sqrt(
                np.mean(
                    feature_diff_ideal ** 2
                )
            )
        ),
    }

    feature_rows = []

    for j, feature in enumerate(h.FEATURES):
        d_direct = (
            X_sampled[:, j]
            - X_direct[:, j]
        )

        d_ideal = (
            X_sampled[:, j]
            - ideal["X_val_ideal"][:, j]
        )

        feature_rows.append({
            "r": int(r),
            "shots_per_setting": int(shots),
            "replicate": int(replicate),
            "feature": feature,

            "feature_bias_vs_direct": float(
                np.mean(d_direct)
            ),

            "feature_rmse_vs_direct": float(
                np.sqrt(
                    np.mean(
                        d_direct ** 2
                    )
                )
            ),

            "feature_mae_vs_direct": float(
                np.mean(
                    np.abs(d_direct)
                )
            ),

            "feature_corr_vs_direct": safe_corr(
                X_direct[:, j],
                X_sampled[:, j],
            ),

            "feature_bias_vs_ideal": float(
                np.mean(d_ideal)
            ),

            "feature_rmse_vs_ideal": float(
                np.sqrt(
                    np.mean(
                        d_ideal ** 2
                    )
                )
            ),
        })

    predictions = pd.DataFrame({
        "r": int(r),
        "shots_per_setting": int(shots),
        "replicate": int(replicate),
        "endpoint": ideal["val_endpoints"],
        "y_true": y,
        "prediction_sampled": pred,
        "prediction_full_quantum_direct": direct_pred,
        "prediction_same_r_ideal": ideal_pred,
        "sampled_minus_direct_prediction": (
            pred - direct_pred
        ),
    })

    return row, feature_rows, predictions


# =============================================================================
# SUMMARY / PARETO
# =============================================================================

def summarize_sampled_runs(run_rows):
    df = pd.DataFrame(
        run_rows
    )

    rows = []

    for (
        r,
        shots,
    ), g in df.groupby(
        [
            "r",
            "shots_per_setting",
        ],
        sort=True,
    ):
        n = len(g)

        rmse_sd = (
            float(
                g["rmse"].std(ddof=1)
            )
            if n > 1
            else 0.0
        )

        rows.append({
            "r": int(r),
            "shots_per_setting": int(shots),
            "total_circuit_shots_per_feature": int(
                g[
                    "total_circuit_shots_per_feature"
                ].iloc[0]
            ),
            "n_runs": int(n),

            "rmse_mean": float(
                g["rmse"].mean()
            ),
            "rmse_sd": rmse_sd,
            "rmse_se": (
                rmse_sd / math.sqrt(n)
                if n > 1
                else 0.0
            ),

            "mae_mean": float(
                g["mae"].mean()
            ),
            "mae_sd": (
                float(
                    g["mae"].std(ddof=1)
                )
                if n > 1
                else 0.0
            ),

            "bias_mean": float(
                g["bias"].mean()
            ),
            "bias_sd": (
                float(
                    g["bias"].std(ddof=1)
                )
                if n > 1
                else 0.0
            ),

            "direct_full_quantum_rmse": float(
                g[
                    "direct_full_quantum_rmse"
                ].iloc[0]
            ),

            "delta_rmse_vs_full_quantum_direct_mean": float(
                g[
                    "delta_rmse_vs_full_quantum_direct"
                ].mean()
            ),

            "delta_rmse_vs_same_r_ideal_mean": float(
                g[
                    "delta_rmse_vs_same_r_ideal"
                ].mean()
            ),

            "prediction_rmse_vs_full_quantum_direct_mean": float(
                g[
                    "prediction_rmse_vs_full_quantum_direct"
                ].mean()
            ),

            "prediction_corr_vs_full_quantum_direct_mean": float(
                g[
                    "prediction_corr_vs_full_quantum_direct"
                ].mean()
            ),

            "feature_rmse_vs_full_quantum_direct_mean": float(
                g[
                    "feature_rmse_vs_full_quantum_direct"
                ].mean()
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "r",
                "shots_per_setting",
            ]
        )
        .reset_index(drop=True)
    )


def make_resource_shot_grid(
    sampled_summary,
    resources,
):
    grid = sampled_summary.merge(
        resources,
        on="r",
        how="left",
        validate="many_to_one",
    )

    # Actual number of CZ applications required across both measurement
    # settings for one feature vector, counting shot repetitions.
    grid[
        "cz_applications_per_feature_vector"
    ] = (
        grid["feature_vector_n_cz"]
        * grid["shots_per_setting"]
    ).astype(int)

    return grid


def pareto_mask(
    df,
    criteria,
):
    """
    True = non-dominated.
    All listed criteria are minimized.

    A row is dominated if another row is <= it in every criterion
    and strictly < in at least one.
    """
    values = df[
        criteria
    ].to_numpy(
        dtype=float
    )

    keep = np.ones(
        len(df),
        dtype=bool,
    )

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue

            if (
                np.all(
                    values[j] <= values[i]
                )
                and np.any(
                    values[j] < values[i]
                )
            ):
                keep[i] = False
                break

    return keep


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sampling-reps",
        type=int,
        default=DEFAULT_REPS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
    )

    parser.add_argument(
        "--sim-seed",
        type=int,
        default=DEFAULT_SIM_SEED,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use 30 validation endpoints, r={3,4}, "
            "shots={256,1024}, and one replicate."
        ),
    )

    args = parser.parse_args()

    if args.quick:
        r_values = [3, 4]
        shot_levels = [256, 1024]
        sampling_reps = 1
    else:
        r_values = list(
            R_VALUES
        )
        shot_levels = list(
            SHOT_LEVELS
        )
        sampling_reps = int(
            args.sampling_reps
        )

    max_shots = int(
        max(shot_levels)
    )

    print("=" * 122)
    print("10.04 — FULL-QUANTUM DIRECT EXPECTATION vs FINITE-SHOT SAMPLING")
    print("=" * 122)
    print("NO QPU JOBS ARE SUBMITTED.")
    print("NO ERROR MITIGATION.")
    print("READOUT ERROR: OFF in BOTH regimes.")
    print("2026 remains frozen.")
    print(
        f"r tested: {r_values}"
    )
    print(
        f"shot budgets per setting: {shot_levels}"
    )
    print(
        f"sampling replicates: {sampling_reps}"
    )
    print(
        "Nested design: lower shot budgets are prefixes of "
        f"the same {max_shots}-shot stream."
    )
    print()

    # ------------------------------------------------------------------
    # Frozen reference and layout
    # ------------------------------------------------------------------
    source_candidate, source_row = (
        h.reconstruct_reference_candidate()
    )

    layout, layout_row = (
        h.load_rank1_layout()
    )

    print("FROZEN DESIGN")
    print(
        f"  H3 / RWP W={h.WINDOW} / "
        f"alpha={source_candidate['alpha']} / "
        f"dt={source_candidate['dt']} / "
        f"{h.READOUT}"
    )
    print(
        f"  hx={source_candidate['hx']:.12f}, "
        f"hy={source_candidate['hy']:.12f}"
    )
    print(
        f"  fixed Kingston layout={layout}"
    )
    print()

    # ------------------------------------------------------------------
    # Refresh Kingston ONCE
    # ------------------------------------------------------------------
    service = get_service()

    backend = service.backend(
        BACKEND_NAME,
        use_fractional_gates=False,
    )

    try:
        backend.refresh()
    except Exception:
        pass

    try:
        properties = backend.properties(
            refresh=True
        )
    except TypeError:
        properties = backend.properties()

    calibration_time = getattr(
        properties,
        "last_update_date",
        None,
    )

    # EXACTLY the same full-quantum model for direct and sampled.
    # Readout error is deliberately OFF.
    full_quantum_noise = NoiseModel.from_backend(
        backend,
        gate_error=True,
        thermal_relaxation=True,
        readout_error=False,
    )

    reference = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "backend": BACKEND_NAME,
        "backend_properties_last_update": json_safe(
            calibration_time
        ),

        "layout": layout,

        "r_values": r_values,
        "shot_levels_per_setting": shot_levels,
        "sampling_reps": sampling_reps,

        "direct_regime": {
            "name": "N5_FULL_QUANTUM_DIRECT",
            "gate_error": True,
            "thermal_relaxation": True,
            "readout_error": False,
            "finite_sampling": False,
        },

        "sampled_regime": {
            "name": "S2_FULL_QUANTUM_SAMPLED",
            "gate_error": True,
            "thermal_relaxation": True,
            "readout_error": False,
            "finite_sampling": True,
        },

        "nested_shot_design": (
            "Each replicate is simulated once at max shots. "
            "Lower budgets use prefixes of the same memory stream."
        ),

        "frozen_candidate": {
            "topology": h.TOPOLOGY,
            "protocol": "RWP",
            "window": h.WINDOW,
            "alpha": float(
                source_candidate["alpha"]
            ),
            "dt": float(
                source_candidate["dt"]
            ),
            "hx": float(
                source_candidate["hx"]
            ),
            "hy": float(
                source_candidate["hy"]
            ),
            "J": {
                f"J{i}{j}": float(v)
                for (i, j), v
                in source_candidate["J"].items()
            },
            "readout": h.READOUT,
        },

        "notes": [
            (
                "N5 and S2 use the same calibration-derived "
                "full-quantum noise model."
            ),
            (
                "Readout assignment error is OFF in both, so "
                "S2-N5 isolates finite-shot measurement effects."
            ),
            (
                "Each r gets its own ideal training-only Ridge "
                "fit because r changes the reservoir features."
            ),
            "No mitigation; no QPU jobs; 2026 untouched.",
        ],
    }

    (
        RESULTS / "10_04_reference.json"
    ).write_text(
        json.dumps(
            json_safe(reference),
            indent=2,
        ),
        encoding="utf-8",
    )

    direct_rows = []
    sampled_rows = []
    feature_rows = []
    prediction_parts = []
    resource_rows = []

    # ==================================================================
    # r LOOP
    # ==================================================================
    for r in r_values:
        print()
        print("-" * 122)
        print(f"TROTTER r={r}")
        print("-" * 122)

        candidate = h.clone_candidate(
            source_candidate
        )
        candidate["r"] = int(r)

        # --------------------------------------------------------------
        # Ideal r-specific reservoir/readout
        # --------------------------------------------------------------
        ideal = h.ideal_reference_for_r(
            candidate
        )

        if args.quick:
            idx = np.linspace(
                0,
                len(
                    ideal["val_endpoints"]
                ) - 1,
                30,
                dtype=int,
            )

            ideal = dict(
                ideal
            )

            ideal[
                "val_endpoints"
            ] = ideal[
                "val_endpoints"
            ][idx]

            ideal[
                "X_val_ideal"
            ] = ideal[
                "X_val_ideal"
            ][idx]

            ideal[
                "y_val"
            ] = ideal[
                "y_val"
            ][idx]

            ideal[
                "pred_val_ideal"
            ] = ideal[
                "pred_val_ideal"
            ][idx]

        print(
            f"IDEAL: lambda={ideal['selected_lambda']}, "
            f"CV={ideal['cv_rmse']:.6f}±{ideal['cv_sd']:.6f}, "
            f"2025 RMSE={ideal['ideal_rmse']:.6f}"
        )

        # --------------------------------------------------------------
        # Physical circuits/resources
        # --------------------------------------------------------------
        physical = h.prepare_physical_circuits(
            backend,
            candidate,
            layout,
        )

        resource = dict(
            physical[
                "resource_summary"
            ]
        )

        resource_rows.append(
            resource
        )

        print(
            f"RESOURCES: CZ/feature={resource['feature_vector_n_cz']}, "
            f"depth={resource['max_setting_depth']}, "
            f"max-setting-time={resource['max_setting_duration_us']:.3f} us, "
            f"R_T2={resource['max_setting_time_over_min_t2']:.4f}"
        )

        # --------------------------------------------------------------
        # Mandatory A/B audit
        # --------------------------------------------------------------
        X_B = h.run_direct_regime(
            noise_model=None,
            physical=physical,
            ideal=ideal,
            batch_size=args.batch_size,
        )

        AB_feature_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        X_B
                        - ideal[
                            "X_val_ideal"
                        ]
                    ) ** 2
                )
            )
        )

        print(
            "A/B audit feature RMSE="
            f"{AB_feature_rmse:.3e}"
        )

        if AB_feature_rmse > 1e-6:
            raise RuntimeError(
                f"A/B audit failed for r={r}: "
                f"{AB_feature_rmse:.3e}"
            )

        resource_rows[-1][
            "A_B_feature_rmse_audit"
        ] = AB_feature_rmse

        # --------------------------------------------------------------
        # N5 DIRECT: full quantum noise, exact expectation
        # --------------------------------------------------------------
        print(
            "N5_FULL_QUANTUM_DIRECT "
            "(gate + thermal, no shots, no readout)"
        )

        t0 = time.time()

        X_direct = h.run_direct_regime(
            noise_model=full_quantum_noise,
            physical=physical,
            ideal=ideal,
            batch_size=args.batch_size,
        )

        direct_pred = h.common.predict_scaled_ridge(
            ideal["model"],
            ideal["scaler"],
            ideal["keep"],
            X_direct,
        )

        direct_row = {
            "r": int(r),
            "selected_lambda": float(
                ideal["selected_lambda"]
            ),
            "training_cv_rmse": float(
                ideal["cv_rmse"]
            ),
            "ideal_rmse": float(
                ideal["ideal_rmse"]
            ),

            "rmse": rmse(
                ideal["y_val"],
                direct_pred,
            ),

            "mae": mae(
                ideal["y_val"],
                direct_pred,
            ),

            "bias": float(
                np.mean(
                    direct_pred
                    - ideal["y_val"]
                )
            ),

            "delta_rmse_vs_same_r_ideal": (
                rmse(
                    ideal["y_val"],
                    direct_pred,
                )
                - ideal["ideal_rmse"]
            ),

            "prediction_corr_vs_same_r_ideal": safe_corr(
                ideal[
                    "pred_val_ideal"
                ],
                direct_pred,
            ),

            "feature_rmse_vs_same_r_ideal": float(
                np.sqrt(
                    np.mean(
                        (
                            X_direct
                            - ideal[
                                "X_val_ideal"
                            ]
                        ) ** 2
                    )
                )
            ),
        }

        direct_rows.append(
            direct_row
        )

        print(
            f"  RMSE={direct_row['rmse']:.6f}, "
            f"Δideal={direct_row['delta_rmse_vs_same_r_ideal']:+.6f}, "
            f"time={(time.time()-t0)/60:.2f} min"
        )

        # --------------------------------------------------------------
        # S2 SAMPLED: same full quantum noise, NO readout
        # --------------------------------------------------------------
        for rep in range(
            sampling_reps
        ):
            rep_seed = int(
                args.sim_seed
                + 1000000 * rep
            )

            print(
                f"S2 replicate "
                f"{rep+1}/{sampling_reps}: "
                f"one {max_shots}-shot run -> "
                f"{shot_levels}"
            )

            t0 = time.time()

            X_by_shot = run_nested_sampled_regime(
                noise_model=full_quantum_noise,
                physical=physical,
                ideal=ideal,
                shot_levels=shot_levels,
                batch_size=args.batch_size,
                seed=rep_seed,
            )

            for N in shot_levels:
                (
                    row,
                    feat_rows,
                    pred_df,
                ) = evaluate_sampled_against_direct(
                    r=r,
                    shots=N,
                    replicate=rep + 1,
                    X_sampled=X_by_shot[int(N)],
                    X_direct=X_direct,
                    direct_pred=direct_pred,
                    ideal=ideal,
                )

                sampled_rows.append(
                    row
                )

                feature_rows.extend(
                    feat_rows
                )

                prediction_parts.append(
                    pred_df
                )

                print(
                    f"  N={N:4d}: "
                    f"RMSE={row['rmse']:.6f}, "
                    f"Δdirect="
                    f"{row['delta_rmse_vs_full_quantum_direct']:+.6f}, "
                    f"featureRMSE_vs_direct="
                    f"{row['feature_rmse_vs_full_quantum_direct']:.6f}"
                )

            print(
                f"  replicate runtime="
                f"{(time.time()-t0)/60:.2f} min"
            )

    # ==================================================================
    # SUMMARY
    # ==================================================================
    direct_df = pd.DataFrame(
        direct_rows
    )

    sampled_df = pd.DataFrame(
        sampled_rows
    )

    sampled_summary = summarize_sampled_runs(
        sampled_rows
    )

    feature_df = pd.DataFrame(
        feature_rows
    )

    prediction_df = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    resources = pd.DataFrame(
        resource_rows
    )

    resource_grid = make_resource_shot_grid(
        sampled_summary,
        resources,
    )

    # Pareto across performance + the two explicit resource axes.
    pareto_criteria = [
        "rmse_mean",
        "feature_vector_n_cz",
        "shots_per_setting",
    ]

    resource_grid[
        "pareto_nondominated"
    ] = pareto_mask(
        resource_grid,
        pareto_criteria,
    )

    pareto = (
        resource_grid[
            resource_grid[
                "pareto_nondominated"
            ]
        ]
        .sort_values(
            [
                "rmse_mean",
                "feature_vector_n_cz",
                "shots_per_setting",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ==================================================================
    # SAVE
    # ==================================================================
    direct_df.to_csv(
        RESULTS
        / "10_04_full_quantum_direct.csv",
        index=False,
    )

    sampled_df.to_csv(
        RESULTS
        / "10_04_sampled_runs.csv",
        index=False,
    )

    sampled_summary.to_csv(
        RESULTS
        / "10_04_sampled_summary.csv",
        index=False,
    )

    feature_df.to_csv(
        RESULTS
        / "10_04_feature_metrics.csv",
        index=False,
    )

    prediction_df.to_csv(
        RESULTS
        / "10_04_predictions.csv",
        index=False,
    )

    resource_grid.to_csv(
        RESULTS
        / "10_04_resource_shot_grid.csv",
        index=False,
    )

    pareto.to_csv(
        RESULTS
        / "10_04_pareto_front.csv",
        index=False,
    )

    # ==================================================================
    # CONSOLE
    # ==================================================================
    print()
    print("=" * 122)
    print("10.04 FINAL SUMMARY")
    print("=" * 122)

    print()
    print(
        "N5 FULL-QUANTUM DIRECT "
        "(same noise, exact expectations)"
    )

    print(
        direct_df[
            [
                "r",
                "rmse",
                "mae",
                "bias",
                "delta_rmse_vs_same_r_ideal",
                "prediction_corr_vs_same_r_ideal",
            ]
        ]
        .sort_values(
            "r"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "S2 FULL-QUANTUM SAMPLED "
        "(same noise, readout OFF)"
    )

    display_cols = [
        "r",
        "shots_per_setting",
        "n_runs",
        "rmse_mean",
        "rmse_sd",
        "rmse_se",
        "mae_mean",
        "bias_mean",
        "direct_full_quantum_rmse",
        "delta_rmse_vs_full_quantum_direct_mean",
        "prediction_corr_vs_full_quantum_direct_mean",
        "feature_rmse_vs_full_quantum_direct_mean",
    ]

    print(
        sampled_summary[
            display_cols
        ]
        .sort_values(
            [
                "r",
                "shots_per_setting",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "RESOURCE / SHOT PARETO FRONT "
        "(minimize RMSE, CZ/feature, shots)"
    )

    pareto_cols = [
        "r",
        "shots_per_setting",
        "rmse_mean",
        "rmse_sd",
        "feature_vector_n_cz",
        "max_setting_depth",
        "max_setting_duration_us",
        "cz_applications_per_feature_vector",
    ]

    print(
        pareto[
            pareto_cols
        ].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    for name in [
        "10_04_reference.json",
        "10_04_full_quantum_direct.csv",
        "10_04_sampled_runs.csv",
        "10_04_sampled_summary.csv",
        "10_04_feature_metrics.csv",
        "10_04_predictions.csv",
        "10_04_resource_shot_grid.csv",
        "10_04_pareto_front.csv",
    ]:
        print(
            f"  results/{name}"
        )

    print()
    print("10.04 COMPLETE.")
    print(
        "Stop here. Compare convergence to the direct N5 reference and "
        "the RMSE/CZ/shot Pareto frontier before choosing an operating point."
    )


if __name__ == "__main__":
    main()
