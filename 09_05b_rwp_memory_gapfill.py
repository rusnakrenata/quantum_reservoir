
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
TRIAL_FILE = RESULTS / "09_04c_rwp_all_trials.csv"

BY_SEED_FILE = RESULTS / "09_05b_rwp_memory_by_seed.csv"
SUMMARY_FILE = RESULTS / "09_05b_rwp_memory_summary.csv"
WINNER_FILE = RESULTS / "09_05b_rwp_memory_rule2_winners.csv"

PROBE_SEEDS = [79001, 42, 101, 505, 707]
MEMORY_WINDOWS = [1, 2, 5, 7, 14, 21, 28]
EXPECTED_TOPOLOGIES = ["H0", "H1", "H2", "H3"]
K_MAX = 20
MC_RIDGE_ALPHA = 1e-6
VAR_TOL = 1e-12


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_0905b_common")


def reconstruct_stage_c_candidate(row, baselines):
    """
    Reconstruct the exact Stage-C dynamics from the original 9.3c topology
    baseline plus the deterministic Stage-C search_id.

    Why this is necessary:
    09_04c_rwp_all_trials.csv was written incrementally topology-by-topology.
    Because H0/H1/H2/H3 use different semantic coupling columns
    (J34/J35/J24/J04), the first H0 CSV header cannot represent the later
    topology-specific column names. Forecast/CV metrics are valid because they
    were computed before serialization, but the coupling labels in that CSV
    are not a safe source for rebuilding H1-H3 dynamics.

    Stage C itself is deterministic, so the exact candidate can be regenerated
    from:
        topology baseline from 09_03c_forecast_best_135.csv
        + H/W parent identity
        + trial_search_id
    """
    H = str(row["topology"])
    W = int(row["window"])
    search_id = str(row["trial_search_id"])

    if H not in baselines:
        raise KeyError(f"No 9.3c baseline reconstructed for topology {H}.")

    base = common._clone_candidate(baselines[H])
    base["candidate_id"] = f"{H}_W{W:02d}_Aparent"
    base["window"] = W

    candidates = common.stage_c_candidates(base)
    by_id = {str(c["search_id"]): c for c in candidates}

    if search_id not in by_id:
        raise KeyError(
            f"Could not reconstruct Stage-C search_id={search_id!r} "
            f"for {H} W={W:02d}."
        )

    candidate = by_id[search_id]

    # Audit all scalar dynamics that are safely stored in the Stage-C CSV.
    checks = {
        "alpha": float(candidate["alpha"]),
        "dt": float(candidate["dt"]),
        "r": int(candidate["r"]),
        "hx": float(candidate["hx"]),
        "hy": float(candidate["hy"]),
    }
    for key, expected in checks.items():
        observed = float(row[key])
        if key == "r":
            ok = int(observed) == int(expected)
        else:
            ok = np.isclose(observed, expected, rtol=1e-10, atol=1e-12)

        if not ok:
            raise RuntimeError(
                f"Stage-C reconstruction audit failed for {H} W={W:02d} "
                f"{search_id}: {key} CSV={observed}, reconstructed={expected}."
            )

    return candidate


def standardized_ridge_predict(Xtr, ytr, Xva):
    Xtr = np.asarray(Xtr, dtype=float)
    Xva = np.asarray(Xva, dtype=float)
    ytr = np.asarray(ytr, dtype=float)

    mu = np.mean(Xtr, axis=0)
    sd = np.std(Xtr, axis=0, ddof=0)
    keep = sd > VAR_TOL

    if not np.any(keep):
        return np.full(len(Xva), float(np.mean(ytr)))

    Xtrz = (Xtr[:, keep] - mu[keep]) / sd[keep]
    Xvaz = (Xva[:, keep] - mu[keep]) / sd[keep]

    model = Ridge(alpha=MC_RIDGE_ALPHA, fit_intercept=True)
    model.fit(Xtrz, ytr)
    return model.predict(Xvaz)


def safe_corr2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) <= VAR_TOL or np.std(b) <= VAR_TOL:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(r):
        return 0.0
    return float(r * r)


def build_probe(real_angles, seed):
    rng = np.random.default_rng(int(seed))
    probe = np.empty_like(real_angles)
    for j in range(real_angles.shape[1]):
        probe[:, j] = real_angles[rng.permutation(len(real_angles)), j]
    return probe


def evaluate_rwp_mc(endpoints, X, probe):
    null_floor = 1.0 / (common.N_VAL - 1.0)
    rows = []

    for channel in range(probe.shape[1]):
        for delay in range(1, K_MAX + 1):
            usable = endpoints >= delay
            e = endpoints[usable]
            Xu = X[usable]

            tr_mask = e < common.N_TRAIN
            va_mask = e >= common.N_TRAIN

            Xtr = Xu[tr_mask]
            Xva = Xu[va_mask]
            ytr = probe[e[tr_mask] - delay, channel]
            yva = probe[e[va_mask] - delay, channel]

            pred = standardized_ridge_predict(Xtr, ytr, Xva)
            raw = safe_corr2(yva, pred)
            corr = max(raw - null_floor, 0.0)

            rows.append(
                {
                    "channel": channel,
                    "delay": delay,
                    "MC_raw": raw,
                    "MC_corrected": corr,
                }
            )

    detail = pd.DataFrame(rows)
    total = float(detail["MC_corrected"].sum())

    def dmean(k):
        return float(
            detail.loc[detail["delay"] == k, "MC_raw"].mean()
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


def finalist_pool(trials):
    # Rule-2 memory is evaluated only on the original/predeclared RWP
    # temporal protocol set used in Weeks 4-7:
    # W in {1, 2, 5, 7, 14, 21, 28}.
    #
    # Stage-C forecasting was allowed to search every W=1..28, but that
    # larger grid is NOT reused here for the controlled memory comparison.
    trials = trials[
        trials["topology"].isin(EXPECTED_TOPOLOGIES)
        & trials["window"].isin(MEMORY_WINDOWS)
    ].copy()

    observed = {
        (str(h), int(w))
        for h, w in trials[["topology", "window"]].drop_duplicates().itertuples(index=False)
    }
    expected = {
        (h, w)
        for h in EXPECTED_TOPOLOGIES
        for w in MEMORY_WINDOWS
    }

    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    if missing:
        raise RuntimeError(
            "Stage-C table is missing required H x W memory groups: "
            + ", ".join(f"{h}/W{w}" for h, w in missing)
        )
    if extra:
        raise RuntimeError(
            "Unexpected H x W groups survived the memory-grid filter: "
            + ", ".join(f"{h}/W{w}" for h, w in extra)
        )

    selected = []

    for (H, W), group in trials.groupby(["topology", "window"]):
        cv = group.sort_values("cv_rmse").iloc[0].copy()
        cv["pool_role"] = "CV_WINNER"
        selected.append(cv)

        hw, mode = common.select_hardware_aware_row(group)
        hw = hw.copy()
        hw["pool_role"] = "HW_AWARE_WINNER"
        selected.append(hw)

    pool = pd.DataFrame(selected)

    # The exact Stage-C dynamics are identified by
    # (topology, window, trial_search_id).  Do NOT use the serialized J columns
    # from 09_04c_rwp_all_trials.csv here; see reconstruct_stage_c_candidate().
    dyn_cols = ["topology", "window", "trial_search_id"]

    grouped = []
    for _, g in pool.groupby(dyn_cols, dropna=False):
        row = g.iloc[0].copy()
        row["pool_role"] = "+".join(sorted(set(g["pool_role"].astype(str))))
        grouped.append(row)

    return pd.DataFrame(grouped).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--probe-seeds",
        default=",".join(str(x) for x in PROBE_SEEDS),
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.probe_seeds.split(",") if x.strip()]

    if not TRIAL_FILE.exists():
        raise FileNotFoundError(TRIAL_FILE)

    trials = pd.read_csv(TRIAL_FILE)
    pool = finalist_pool(trials)

    # Reconstruct the exact topology baselines from the clean Week-9.3c table.
    baselines = common.load_topology_baselines(EXPECTED_TOPOLOGIES)

    work_tv, cols, y = common.load_train_validation()

    if args.overwrite:
        for p in [BY_SEED_FILE, SUMMARY_FILE, WINNER_FILE]:
            if p.exists():
                p.unlink()

    if BY_SEED_FILE.exists():
        rows = pd.read_csv(BY_SEED_FILE).to_dict("records")
    else:
        rows = []

    completed = {
        (
            str(r["topology"]),
            int(r["window"]),
            str(r["trial_search_id"]),
            int(r["probe_seed"]),
        )
        for r in rows
    }

    angle_cache = {}

    print("=" * 120)
    print("09.05B — RWP MEMORY GAP-FILL ON PREDECLARED WINDOWS")
    print("=" * 120)
    print(f"Memory windows: {MEMORY_WINDOWS}")
    print(f"Required H x W groups: {len(EXPECTED_TOPOLOGIES) * len(MEMORY_WINDOWS)}")
    print(f"Unique retained dynamics after CV/HW finalist union: {len(pool)}")
    print(f"Probe seeds: {seeds}")
    print("Memory readout: XYZ_all")
    print("Rule-2 window grid is fixed independently of the Stage-C W=1..28 forecasting search.")
    print("Stage-C couplings are reconstructed from the clean 9.3c baseline + deterministic search_id.")
    print("The unsafe topology-specific J columns in 09_04c_rwp_all_trials.csv are NOT used to rebuild dynamics.")
    print("Delays: 1..20")
    print("RWP restarts every endpoint; CONT-style initialization washout is not applied.")
    print()

    for i, row in pool.iterrows():
        candidate = reconstruct_stage_c_candidate(row, baselines)
        H = candidate["topology"]
        W = int(row["window"])
        search_id = str(row["trial_search_id"])
        alpha = float(candidate["alpha"])

        if alpha not in angle_cache:
            angle_cache[alpha] = common.make_angles(work_tv, cols, alpha)

        real_angles = angle_cache[alpha]

        for seed in seeds:
            key = (H, W, search_id, int(seed))
            if key in completed:
                continue

            probe = build_probe(real_angles, seed)
            A_list = common.build_channels(candidate, probe)
            endpoints, master = common.rwp_master_feature_bank_from_channels(
                A_list, W
            )

            X = common.select_features(master, "XYZ_all")
            mc = evaluate_rwp_mc(endpoints, X, probe)

            rows.append(
                {
                    "topology": H,
                    "window": W,
                    "trial_search_id": search_id,
                    "pool_role": str(row["pool_role"]),
                    "probe_seed": int(seed),
                    "alpha": candidate["alpha"],
                    "dt": candidate["dt"],
                    "r": candidate["r"],
                    "hx": candidate["hx"],
                    "hy": candidate["hy"],
                    "source_cv_rmse": float(row["cv_rmse"]),
                    "source_validation_rmse": float(row["validation_rmse"]),
                    **common.candidate_to_flat(candidate),
                    **mc,
                }
            )
            completed.add(key)

            pd.DataFrame(rows).to_csv(BY_SEED_FILE, index=False)

        print(
            f"[{i+1:03d}/{len(pool)}] {H} W={W:02d} {search_id}"
        )

    by_seed = pd.DataFrame(rows)

    group_cols = [
        "topology", "window", "trial_search_id", "pool_role",
        "alpha", "dt", "r", "hx", "hy",
        "source_cv_rmse", "source_validation_rmse",
    ]

    summary = (
        by_seed.groupby(group_cols, dropna=False)
        .agg(
            n_probe_seeds=("probe_seed", "nunique"),
            MC_ch_mean=("MC_ch_corrected", "mean"),
            MC_ch_std=("MC_ch_corrected", "std"),
            MC_total_mean=("MC_total_corrected", "mean"),
            MC_delay1_mean=("MC_delay1_mean", "mean"),
            MC_delay2_mean=("MC_delay2_mean", "mean"),
            MC_delay5_mean=("MC_delay5_mean", "mean"),
            MC_delay10_mean=("MC_delay10_mean", "mean"),
            MC_delay20_mean=("MC_delay20_mean", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    winners = []
    for H, g in summary.groupby("topology"):
        winners.append(
            g.sort_values(
                ["MC_ch_mean", "source_cv_rmse"],
                ascending=[False, True],
            ).iloc[0].to_dict()
        )

    pd.DataFrame(winners).to_csv(WINNER_FILE, index=False)

    print()
    print("RULE-2 RWP WINNERS AMONG RETAINED STAGE-C FINALISTS")
    print(
        pd.DataFrame(winners)[
            [
                "topology", "window", "trial_search_id",
                "MC_ch_mean", "MC_ch_std",
                "source_cv_rmse", "source_validation_rmse",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Saved: {BY_SEED_FILE}")
    print(f"Saved: {SUMMARY_FILE}")
    print(f"Saved: {WINNER_FILE}")


if __name__ == "__main__":
    main()
