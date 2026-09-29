"""
Week 12.4A — IQM H5/H6 RWP memory closure.

Direct IQM analogue of IBM `09_05b_rwp_memory_gapfill.py`.

Purpose
-------
After the full 12.3G fair Trotter-r reoptimization, evaluate intrinsic delayed-
input memory only on the original/predeclared temporal-memory window grid:

    W in {1, 2, 5, 7, 14, 21, 28}

For every H x W group:
  1. retain the training-CV winner,
  2. retain the hardware-aware winner,
  3. deduplicate if they are the same dynamics,
  4. evaluate memory with XYZ_all over five deterministic shuffled-input probes,
  5. select one Rule-2 memory winner per topology by mean corrected MC/channel,
     breaking ties by lower source training-CV RMSE.

Scientific constraints
----------------------
- 2022-2024: training / memory-probe fitting.
- 2025: diagnostic source RMSE only; never used for selection.
- 2026: untouched / not loaded.
- No QPU jobs.
- Same five probe seeds, delay range, Ridge alpha, and corrected-MC definition as
  IBM 09.05B.
- Exact 12.3G dynamics are reconstructed directly from the corrected canonical
  H5/H6 coupling columns in `12_03g_rwp_trotter_all_trials.csv`.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "12_03d_iqm_rwp_common.py"
TRIAL_FILE = RESULTS / "12_03g_rwp_trotter_all_trials.csv"

BY_SEED_FILE = RESULTS / "12_04a_iqm_rwp_memory_by_seed.csv"
SUMMARY_FILE = RESULTS / "12_04a_iqm_rwp_memory_summary.csv"
WINNER_FILE = RESULTS / "12_04a_iqm_rwp_memory_rule2_winners.csv"
POOL_FILE = RESULTS / "12_04a_iqm_rwp_memory_finalist_pool.csv"

PROBE_SEEDS = [79001, 42, 101, 505, 707]
MEMORY_WINDOWS = [1, 2, 5, 7, 14, 21, 28]
EXPECTED_TOPOLOGIES = ["H5", "H6"]

K_MAX = 20
MC_RIDGE_ALPHA = 1e-6
VAR_TOL = 1e-12

EXPECTED_FULL_TRIAL_ROWS = 56 * 3 * 13 * 5  # 10,920


# =============================================================================
# IMPORT SHARED IQM RWP UTILITIES
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load {path}")

    spec.loader.exec_module(mod)
    return mod


common = load_module(
    COMMON_FILE,
    "qrc_1204a_common",
)


# =============================================================================
# EXACT 12.3G CANDIDATE RECONSTRUCTION
# =============================================================================

def candidate_from_12_03g_row(row: pd.Series | dict) -> dict:
    """
    Reconstruct the exact H5/H6 candidate from a corrected 12.3G row.

    12.3G serializes couplings using canonical names from
    common.EDGE_TO_COLUMN, so no positional H5/H6 repair is allowed here.
    Missing topology-specific couplings are treated as a hard audit failure.
    """
    if isinstance(row, pd.Series):
        row = row.to_dict()

    topology = str(row["topology"])

    if topology not in EXPECTED_TOPOLOGIES:
        raise ValueError(
            f"Unexpected topology {topology!r}; "
            f"expected {EXPECTED_TOPOLOGIES}."
        )

    J = {}

    for edge, col in common.EDGE_TO_COLUMN[topology].items():
        if col not in row or pd.isna(row[col]):
            raise KeyError(
                f"12.3G row for {topology} is missing canonical coupling "
                f"{col} for edge={edge}."
            )
        J[edge] = float(row[col])

    r = int(row["r"])

    if "trotter_r_tested" in row and not pd.isna(row["trotter_r_tested"]):
        tested = int(row["trotter_r_tested"])
        if tested != r:
            raise RuntimeError(
                f"r audit failed: row r={r}, trotter_r_tested={tested}."
            )

    return {
        "topology": topology,
        "alpha": float(row["alpha"]),
        "dt": float(row["dt"]),
        "r": r,
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
        "J": J,
        "candidate_id": (
            f"{topology}_W{int(row['window']):02d}_"
            f"r{r}_{row.get('trotter_search_id', 'unknown')}"
        ),
        "search_id": str(
            row.get(
                "trotter_search_id",
                row.get("search_id", ""),
            )
        ),
        "window": int(row["window"]),
    }


# =============================================================================
# MEMORY-CAPACITY HELPERS — MATCH IBM 09.05B
# =============================================================================

def standardized_ridge_predict(Xtr, ytr, Xva):
    Xtr = np.asarray(Xtr, dtype=float)
    Xva = np.asarray(Xva, dtype=float)
    ytr = np.asarray(ytr, dtype=float)

    mu = np.mean(Xtr, axis=0)
    sd = np.std(Xtr, axis=0, ddof=0)
    keep = sd > VAR_TOL

    if not np.any(keep):
        return np.full(
            len(Xva),
            float(np.mean(ytr)),
        )

    Xtrz = (
        Xtr[:, keep] - mu[keep]
    ) / sd[keep]

    Xvaz = (
        Xva[:, keep] - mu[keep]
    ) / sd[keep]

    model = Ridge(
        alpha=MC_RIDGE_ALPHA,
        fit_intercept=True,
    )

    model.fit(Xtrz, ytr)

    return model.predict(Xvaz)


def safe_corr2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        np.std(a) <= VAR_TOL
        or np.std(b) <= VAR_TOL
    ):
        return 0.0

    r = np.corrcoef(a, b)[0, 1]

    if not np.isfinite(r):
        return 0.0

    return float(r * r)


def build_probe(real_angles, seed):
    """
    Same null-preserving shuffled-input probe construction as IBM 09.05B:
    independently permute each of the four encoded input channels.
    """
    rng = np.random.default_rng(int(seed))

    probe = np.empty_like(real_angles)

    for j in range(real_angles.shape[1]):
        probe[:, j] = real_angles[
            rng.permutation(len(real_angles)),
            j,
        ]

    return probe


def evaluate_rwp_mc(endpoints, X, probe):
    """
    Corrected memory capacity over delays 1..K_MAX.

    For each input channel and delay k:
        MC_raw = corr^2(target delayed input, Ridge prediction)

    A finite-sample null floor 1/(N_val - 1) is subtracted and clipped at zero,
    exactly as in IBM 09.05B.
    """
    null_floor = 1.0 / (
        common.N_VAL - 1.0
    )

    rows = []

    for channel in range(
        probe.shape[1]
    ):
        for delay in range(
            1,
            K_MAX + 1,
        ):
            usable = (
                endpoints >= delay
            )

            e = endpoints[usable]
            Xu = X[usable]

            tr_mask = (
                e < common.N_TRAIN
            )

            va_mask = (
                e >= common.N_TRAIN
            )

            Xtr = Xu[tr_mask]
            Xva = Xu[va_mask]

            ytr = probe[
                e[tr_mask] - delay,
                channel,
            ]

            yva = probe[
                e[va_mask] - delay,
                channel,
            ]

            pred = standardized_ridge_predict(
                Xtr,
                ytr,
                Xva,
            )

            raw = safe_corr2(
                yva,
                pred,
            )

            corrected = max(
                raw - null_floor,
                0.0,
            )

            rows.append(
                {
                    "channel": int(channel),
                    "delay": int(delay),
                    "MC_raw": float(raw),
                    "MC_corrected": float(corrected),
                }
            )

    detail = pd.DataFrame(rows)

    total = float(
        detail["MC_corrected"].sum()
    )

    def dmean(k):
        return float(
            detail.loc[
                detail["delay"] == k,
                "MC_raw",
            ].mean()
        )

    return {
        "MC_total_corrected": total,
        "MC_ch_corrected": total / 4.0,
        "MC_delay1_mean": dmean(1),
        "MC_delay2_mean": dmean(2),
        "MC_delay5_mean": dmean(5),
        "MC_delay10_mean": dmean(10),
        "MC_delay20_mean": dmean(20),
    }


# =============================================================================
# 12.3G FINALIST POOL — DIRECT ANALOGUE OF IBM RULE-2 POOL
# =============================================================================

def finalist_pool(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to the original memory windows and retain:
      - training-CV winner per H x W,
      - hardware-aware winner per H x W.

    If both roles point to the same dynamics, keep that dynamics once.
    Since 12.3G reoptimizes r, exact dynamics identity includes r.
    """
    required_cols = [
        "topology",
        "window",
        "r",
        "trotter_search_id",
        "readout",
        "cv_rmse",
        "cv_rmse_std",
        "validation_rmse",
        "alpha",
        "dt",
        "hx",
        "hy",
    ]

    missing = [
        c for c in required_cols
        if c not in trials.columns
    ]

    if missing:
        raise RuntimeError(
            "12.3G trial table is missing required columns: "
            + ", ".join(missing)
        )

    work = trials[
        trials["topology"].isin(
            EXPECTED_TOPOLOGIES
        )
        &
        trials["window"].isin(
            MEMORY_WINDOWS
        )
    ].copy()

    observed = {
        (str(h), int(w))
        for h, w in (
            work[
                ["topology", "window"]
            ]
            .drop_duplicates()
            .itertuples(
                index=False
            )
        )
    }

    expected = {
        (h, w)
        for h in EXPECTED_TOPOLOGIES
        for w in MEMORY_WINDOWS
    }

    missing_groups = sorted(
        expected - observed
    )

    extra_groups = sorted(
        observed - expected
    )

    if missing_groups:
        raise RuntimeError(
            "12.3G is missing required H x W memory groups: "
            + ", ".join(
                f"{h}/W{w}"
                for h, w in missing_groups
            )
        )

    if extra_groups:
        raise RuntimeError(
            "Unexpected H x W groups survived memory-window filter: "
            + ", ".join(
                f"{h}/W{w}"
                for h, w in extra_groups
            )
        )

    selected = []

    for (H, W), group in work.groupby(
        ["topology", "window"],
        sort=True,
    ):
        # Rule pool member 1: pure training-CV winner.
        cv = (
            group.sort_values(
                [
                    "cv_rmse",
                    "cv_rmse_std",
                    "shot_noise_forecast_sd_proxy_1024",
                    "feature_vector_cz",
                    "n_settings",
                ]
            )
            .iloc[0]
            .copy()
        )

        cv["pool_role"] = (
            "CV_WINNER"
        )

        selected.append(cv)

        # Rule pool member 2: exact shared hardware-aware selector.
        hw, mode = (
            common
            .select_hardware_aware_row(
                group
            )
        )

        hw = hw.copy()

        hw["pool_role"] = (
            "HW_AWARE_WINNER"
        )

        hw[
            "pool_hardware_aware_selection_mode"
        ] = mode

        selected.append(hw)

    pool = pd.DataFrame(selected)

    # Exact 12.3G dynamics identity. Readout is deliberately NOT part of the
    # dynamics identity because memory is measured using XYZ_all below.
    dyn_cols = [
        "topology",
        "window",
        "r",
        "trotter_search_id",
    ]

    deduped = []

    for _, g in pool.groupby(
        dyn_cols,
        dropna=False,
        sort=True,
    ):
        # Match IBM behavior: if CV/HW roles collapse to the same dynamics,
        # use the CV-role row as source metric carrier when available.
        g = g.copy()

        g["role_priority"] = (
            g["pool_role"]
            .map(
                {
                    "CV_WINNER": 0,
                    "HW_AWARE_WINNER": 1,
                }
            )
            .fillna(9)
        )

        representative = (
            g.sort_values(
                [
                    "role_priority",
                    "cv_rmse",
                ]
            )
            .iloc[0]
            .copy()
        )

        representative["pool_role"] = (
            "+"
            .join(
                sorted(
                    set(
                        g[
                            "pool_role"
                        ].astype(str)
                    )
                )
            )
        )

        representative[
            "pool_source_readouts"
        ] = "+".join(
            sorted(
                set(
                    g[
                        "readout"
                    ].astype(str)
                )
            )
        )

        deduped.append(
            representative
        )

    result = (
        pd.DataFrame(deduped)
        .drop(
            columns=["role_priority"],
            errors="ignore",
        )
        .reset_index(drop=True)
    )

    return result


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
        "--probe-seeds",
        default=",".join(
            str(x)
            for x in PROBE_SEEDS
        ),
    )

    args = parser.parse_args()

    seeds = [
        int(x.strip())
        for x in (
            args.probe_seeds
            .split(",")
        )
        if x.strip()
    ]

    if not seeds:
        raise ValueError(
            "At least one probe seed is required."
        )

    if not TRIAL_FILE.exists():
        raise FileNotFoundError(
            TRIAL_FILE
        )

    trials = pd.read_csv(
        TRIAL_FILE
    )

    if len(trials) != EXPECTED_FULL_TRIAL_ROWS:
        raise RuntimeError(
            "12.3G full-run audit failed before memory closure: "
            f"found {len(trials)} rows, "
            f"expected {EXPECTED_FULL_TRIAL_ROWS}."
        )

    pool = finalist_pool(
        trials
    )

    # Save the exact memory finalist universe for auditability.
    pool.to_csv(
        POOL_FILE,
        index=False,
    )

    work_tv, cols, _ = (
        common
        .load_train_validation()
    )

    if args.overwrite:
        for p in [
            BY_SEED_FILE,
            SUMMARY_FILE,
            WINNER_FILE,
        ]:
            if p.exists():
                p.unlink()

    if BY_SEED_FILE.exists():
        rows = (
            pd.read_csv(
                BY_SEED_FILE
            )
            .to_dict(
                "records"
            )
        )
    else:
        rows = []

    completed = {
        (
            str(r["topology"]),
            int(r["window"]),
            int(r["r"]),
            str(r["trotter_search_id"]),
            int(r["probe_seed"]),
        )
        for r in rows
    }

    angle_cache = {}

    print("=" * 120)
    print(
        "WEEK 12.4A — IQM H5/H6 RWP MEMORY CLOSURE"
    )
    print("=" * 120)
    print(
        "Direct analogue of IBM 09.05B."
    )
    print(
        f"Memory windows: {MEMORY_WINDOWS}"
    )
    print(
        "Required H x W groups: "
        f"{len(EXPECTED_TOPOLOGIES) * len(MEMORY_WINDOWS)}"
    )
    print(
        "Unique retained dynamics after CV/HW finalist union: "
        f"{len(pool)}"
    )
    print(
        f"Probe seeds: {seeds}"
    )
    print(
        "Memory readout: XYZ_all"
    )
    print(
        f"Delays: 1..{K_MAX}"
    )
    print(
        "RWP restarts every endpoint; CONT-style initialization washout is not applied."
    )
    print(
        "Selection: memory capacity first; source 2022-2024 CV only as tie-break."
    )
    print(
        "2025 diagnostic only; 2026 untouched / not loaded."
    )
    print(
        "No QPU jobs."
    )
    print()

    for i, row in pool.iterrows():
        candidate = (
            candidate_from_12_03g_row(
                row
            )
        )

        H = candidate["topology"]
        W = int(row["window"])
        r = int(candidate["r"])

        search_id = str(
            row[
                "trotter_search_id"
            ]
        )

        alpha = float(
            candidate["alpha"]
        )

        if alpha not in angle_cache:
            angle_cache[
                alpha
            ] = (
                common.make_angles(
                    work_tv,
                    cols,
                    alpha,
                )
            )

        real_angles = (
            angle_cache[
                alpha
            ]
        )

        for seed in seeds:
            key = (
                H,
                W,
                r,
                search_id,
                int(seed),
            )

            if key in completed:
                continue

            probe = build_probe(
                real_angles,
                seed,
            )

            A_list = (
                common.build_channels(
                    candidate,
                    probe,
                )
            )

            endpoints, master = (
                common
                .rwp_master_feature_bank_from_channels(
                    A_list,
                    W,
                )
            )

            X = (
                common
                .select_features(
                    master,
                    "XYZ_all",
                )
            )

            mc = evaluate_rwp_mc(
                endpoints,
                X,
                probe,
            )

            out = {
                "topology": H,
                "window": W,
                "r": r,
                "trotter_search_id": search_id,
                "pool_role": str(
                    row[
                        "pool_role"
                    ]
                ),
                "pool_source_readouts": str(
                    row.get(
                        "pool_source_readouts",
                        row.get(
                            "readout",
                            "",
                        ),
                    )
                ),
                "probe_seed": int(seed),
                "alpha": float(
                    candidate["alpha"]
                ),
                "dt": float(
                    candidate["dt"]
                ),
                "hx": float(
                    candidate["hx"]
                ),
                "hy": float(
                    candidate["hy"]
                ),
                "source_cv_rmse": float(
                    row[
                        "cv_rmse"
                    ]
                ),
                "source_cv_rmse_std": float(
                    row[
                        "cv_rmse_std"
                    ]
                ),
                "source_validation_rmse": float(
                    row[
                        "validation_rmse"
                    ]
                ),
                "source_shot_noise_forecast_sd_proxy_1024": float(
                    row[
                        "shot_noise_forecast_sd_proxy_1024"
                    ]
                ),
                "source_max_shot_to_train_std_ratio_1024": float(
                    row[
                        "max_shot_to_train_std_ratio_1024"
                    ]
                ),
                "source_all_features_shot_resolvable_1024": bool(
                    row[
                        "all_features_shot_resolvable_1024"
                    ]
                ),
                "source_feature_vector_cz": int(
                    row[
                        "feature_vector_cz"
                    ]
                ),
                "source_n_settings": int(
                    row[
                        "n_settings"
                    ]
                ),
                "2025_used_for_selection": False,
                **common.candidate_to_flat(
                    candidate
                ),
                **mc,
            }

            rows.append(out)
            completed.add(key)

            pd.DataFrame(
                rows
            ).to_csv(
                BY_SEED_FILE,
                index=False,
            )

        print(
            f"[{i+1:02d}/{len(pool)}] "
            f"{H} W={W:02d} r={r} "
            f"{search_id} "
            f"roles={row['pool_role']}"
        )

    by_seed = pd.DataFrame(
        rows
    )

    expected_by_seed_rows = (
        len(pool)
        * len(seeds)
    )

    if (
        len(by_seed)
        != expected_by_seed_rows
    ):
        raise RuntimeError(
            "Memory by-seed row-count audit failed: "
            f"got {len(by_seed)}, "
            f"expected {expected_by_seed_rows}."
        )

    group_cols = [
        "topology",
        "window",
        "r",
        "trotter_search_id",
        "pool_role",
        "pool_source_readouts",
        "alpha",
        "dt",
        "hx",
        "hy",
        "source_cv_rmse",
        "source_cv_rmse_std",
        "source_validation_rmse",
        "source_shot_noise_forecast_sd_proxy_1024",
        "source_max_shot_to_train_std_ratio_1024",
        "source_all_features_shot_resolvable_1024",
        "source_feature_vector_cz",
        "source_n_settings",
    ]

    summary = (
        by_seed
        .groupby(
            group_cols,
            dropna=False,
        )
        .agg(
            n_probe_seeds=(
                "probe_seed",
                "nunique",
            ),
            MC_ch_mean=(
                "MC_ch_corrected",
                "mean",
            ),
            MC_ch_std=(
                "MC_ch_corrected",
                "std",
            ),
            MC_total_mean=(
                "MC_total_corrected",
                "mean",
            ),
            MC_delay1_mean=(
                "MC_delay1_mean",
                "mean",
            ),
            MC_delay2_mean=(
                "MC_delay2_mean",
                "mean",
            ),
            MC_delay5_mean=(
                "MC_delay5_mean",
                "mean",
            ),
            MC_delay10_mean=(
                "MC_delay10_mean",
                "mean",
            ),
            MC_delay20_mean=(
                "MC_delay20_mean",
                "mean",
            ),
        )
        .reset_index()
    )

    summary[
        "2025_used_for_selection"
    ] = False

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    winners = []

    for H, g in summary.groupby(
        "topology",
        sort=True,
    ):
        winner = (
            g.sort_values(
                [
                    "MC_ch_mean",
                    "source_cv_rmse",
                    "source_max_shot_to_train_std_ratio_1024",
                    "source_feature_vector_cz",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True,
                ],
            )
            .iloc[0]
            .copy()
        )

        winner[
            "rule"
        ] = (
            "Rule-2 memory: maximize mean corrected MC/channel "
            "over five fixed probe seeds; lower training-CV RMSE "
            "then lower finite-shot ratio/resource cost as tie-breaks"
        )

        winner[
            "2025_used_for_selection"
        ] = False

        winners.append(
            winner.to_dict()
        )

    winner_df = pd.DataFrame(
        winners
    )

    winner_df.to_csv(
        WINNER_FILE,
        index=False,
    )

    print()
    print("=" * 120)
    print(
        "WEEK 12.4A — RULE-2 RWP MEMORY WINNERS"
    )
    print("=" * 120)

    cols_to_print = [
        "topology",
        "window",
        "r",
        "trotter_search_id",
        "pool_role",
        "MC_ch_mean",
        "MC_ch_std",
        "MC_delay1_mean",
        "MC_delay2_mean",
        "MC_delay5_mean",
        "MC_delay10_mean",
        "MC_delay20_mean",
        "source_cv_rmse",
        "source_max_shot_to_train_std_ratio_1024",
        "source_feature_vector_cz",
    ]

    print(
        winner_df[
            cols_to_print
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:"
    )
    print(
        f"  {POOL_FILE}"
    )
    print(
        f"  {BY_SEED_FILE}"
    )
    print(
        f"  {SUMMARY_FILE}"
    )
    print(
        f"  {WINNER_FILE}"
    )
    print()
    print(
        "No QPU jobs submitted."
    )
    print(
        "2026 untouched."
    )
    print(
        "STOP HERE. Interpret 12.4A before moving to IQM physical compilation/resource closure."
    )


if __name__ == "__main__":
    main()
