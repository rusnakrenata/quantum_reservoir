"""
Week 12.5G0 v4 — Unified H0-H6 CONT/RWP Rule-1..Rule-4 selection with strict rule-label parsing.

This repairs the candidate universe so every topology H0..H6 has BOTH:
    CONT
    RWP

and every topology/protocol cell is covered by the same four logical rules:
    R1 accuracy
    R2 intrinsic memory
    R3 readout robustness
    R4 resources

IMPORTANT CONSISTENCY FIX
-------------------------
H5/H6 CONT Week-12.3C originally optimized three readout families.
For cross-topology consistency, this script re-screens the RETAINED 12.3C
dynamics pool with the same five readout families used by the later IBM/IQM
pipeline:

    XZ_injection
    XZinj_dropX3
    XZinj_dropX3_plus_YX45
    XZinj_plus_YX45
    XYZ_all

No new dynamics are invented. The frozen 12.3C dynamics configurations are
reused exactly; only the readout/Ridge layer is re-evaluated.

Selection rules
---------------
R1:
    minimum 2022-2024 chronological CV RMSE.

R2 CONT:
    choose dynamics with maximum MC_ch_mean subject to all five probe seeds
    satisfying Tw(0.01)<=300; then choose the best-CV readout among the same
    five readout families for that frozen dynamics configuration.

R2 RWP:
    preserve the existing frozen replay-memory representatives.

R3:
    1024-shot training-only resolvability hierarchy:
      if any Rmax<=1:
          resolvable set -> CV -> shotSD -> sensitivity -> settings -> N2q
      otherwise:
          Rmax -> shotSD -> CV -> sensitivity -> settings -> N2q

R4:
    one-SE set:
        CV <= CV_best + SD_best/sqrt(5)
    then:
        NSWAP -> N2q -> Nsettings -> structural depth -> CV

For CONT H5/H6 native topology resource screening:
    NSWAP = 0
    N2q(feature vector) = 2*|E|*r*S = 14*r*S
because H5/H6 each contain seven logical ZZ edges.

No QPU jobs.
No simulator execution.
2025 is diagnostic only.
2026 untouched.

Required files
--------------
results/12_05c_iqm_candidate_universe.csv
results/12_03c_forecast_best_by_branch_r.csv
results/12_03c_forecast_best_with_memory.csv
12_03c_iqm_H5_H6_fair_r123_reopt.py

Outputs
-------
results/12_05g0_h5h6_cont_five_readout_screen.csv
results/12_05g0_h5h6_cont_rule_candidates.csv
results/12_05g0_unified_rule1_rule4_candidates.csv
results/12_05g0_rule_coverage_audit.csv
results/12_05g0_manifest.json
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

EXISTING = RESULTS / "12_05c_iqm_candidate_universe.csv"
CONT_FORECAST = RESULTS / "12_03c_forecast_best_by_branch_r.csv"
CONT_MEMORY = RESULTS / "12_03c_forecast_best_with_memory.csv"
CONT_MODULE_PATH = Path("12_03c_iqm_H5_H6_fair_r123_reopt.py")

OUT_SCREEN = RESULTS / "12_05g0_h5h6_cont_five_readout_screen.csv"
OUT_CONT = RESULTS / "12_05g0_h5h6_cont_rule_candidates.csv"
OUT_UNIFIED = RESULTS / "12_05g0_unified_rule1_rule4_candidates.csv"
OUT_AUDIT = RESULTS / "12_05g0_rule_coverage_audit.csv"
OUT_MANIFEST = RESULTS / "12_05g0_manifest.json"

TOPOLOGIES = [f"H{i}" for i in range(7)]
PROTOCOLS = ["CONT", "RWP"]
RULES = ["R1", "R2", "R3", "R4"]
H5H6 = ["H5", "H6"]

N_FOLDS = 5
SHOTS_PROXY = 1024
VAR_TOL = 1e-12

LAMBDA_GRID = np.array(
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0],
    dtype=float,
)

READOUTS = [
    "XZ_injection",
    "XZinj_dropX3",
    "XZinj_dropX3_plus_YX45",
    "XZinj_plus_YX45",
    "XYZ_all",
]

READOUT_FEATURES = {
    "XZ_injection": [
        "X0", "X1", "X2", "X3",
        "Z0", "Z1", "Z2", "Z3",
    ],
    "XZinj_dropX3": [
        "X0", "X1", "X2",
        "Z0", "Z1", "Z2", "Z3",
    ],
    "XZinj_dropX3_plus_YX45": [
        "X0", "X1", "X2",
        "Z0", "Z1", "Z2", "Z3",
        "YX45",
    ],
    "XZinj_plus_YX45": [
        "X0", "X1", "X2", "X3",
        "Z0", "Z1", "Z2", "Z3",
        "YX45",
    ],
    "XYZ_all": [
        "X0", "X1", "X2", "X3", "X4", "X5",
        "Y0", "Y1", "Y2", "Y3", "Y4", "Y5",
        "Z0", "Z1", "Z2", "Z3", "Z4", "Z5",
    ],
}

READOUT_SETTINGS = {
    "XZ_injection": 2,
    "XZinj_dropX3": 2,
    "XZinj_dropX3_plus_YX45": 2,
    "XZinj_plus_YX45": 2,
    "XYZ_all": 3,
}

# Week-12.3 CONT feature-bank compatibility:
# the joint memory observable was historically stored as "YX_45".
# The later IBM/IQM RWP pipeline uses the display alias "YX45".
# Keep the common readout labels, but transparently map the column name.
FEATURE_COLUMN_ALIASES = {
    "YX45": "YX_45",
}

EDGE_COUNTS = {
    "H0": 5, "H1": 5, "H2": 5, "H3": 5, "H4": 5,
    "H5": 7, "H6": 7,
}


def finite(x):
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def rmse(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_scaled_ridge(X, y, ridge_alpha):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    std = np.std(X, axis=0, ddof=0)
    keep = std > VAR_TOL

    if not np.any(keep):
        raise RuntimeError("All selected reservoir features are constant.")

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X[:, keep])

    model = Ridge(
        alpha=float(ridge_alpha),
        fit_intercept=True,
    )
    model.fit(Xz, y)

    return model, scaler, keep


def predict_scaled_ridge(model, scaler, keep, X):
    X = np.asarray(X, dtype=float)
    return model.predict(
        scaler.transform(X[:, keep])
    )


def select_lambda_training_only(X_train, y_train):
    splitter = TimeSeriesSplit(n_splits=N_FOLDS)
    rows = []

    for alpha in LAMBDA_GRID:
        fold_rmse = []

        for fit_idx, cv_idx in splitter.split(X_train):
            model, scaler, keep = fit_scaled_ridge(
                X_train[fit_idx],
                y_train[fit_idx],
                alpha,
            )

            pred = predict_scaled_ridge(
                model,
                scaler,
                keep,
                X_train[cv_idx],
            )

            fold_rmse.append(
                rmse(y_train[cv_idx], pred)
            )

        rows.append({
            "selected_lambda": float(alpha),
            "cv_rmse": float(np.mean(fold_rmse)),
            "cv_rmse_std": float(np.std(fold_rmse, ddof=1)),
        })

    summary = pd.DataFrame(rows).sort_values(
        ["cv_rmse", "selected_lambda"],
        kind="mergesort",
    )

    return summary.iloc[0]


def conditioning_metrics(feature_names, X_train, model, scaler, keep):
    X_train = np.asarray(X_train, dtype=float)
    keep = np.asarray(keep, dtype=bool)

    active_idx = np.flatnonzero(keep)
    coef = np.asarray(model.coef_, dtype=float)

    coef_full = np.zeros(len(feature_names), dtype=float)
    coef_full[active_idx] = coef

    scale_full = np.full(len(feature_names), np.nan, dtype=float)
    scale_full[active_idx] = np.asarray(scaler.scale_, dtype=float)

    details = []

    for j, feature in enumerate(feature_names):
        train_mean = float(np.mean(X_train[:, j]))
        train_std = float(np.std(X_train[:, j], ddof=0))
        active = bool(keep[j])

        pauli_mean = float(np.clip(train_mean, -1.0, 1.0))
        shot_sd = float(
            np.sqrt(
                max(0.0, 1.0 - pauli_mean ** 2)
                / SHOTS_PROXY
            )
        )

        if active:
            sensitivity = float(coef_full[j] / scale_full[j])
            forecast_shot = float(abs(sensitivity) * shot_sd)
        else:
            sensitivity = 0.0
            forecast_shot = 0.0

        ratio = (
            float(shot_sd / train_std)
            if train_std > VAR_TOL
            else float("inf")
        )

        details.append({
            "feature": feature,
            "active": active,
            "train_mean": train_mean,
            "train_std": train_std,
            "raw_prediction_sensitivity": sensitivity,
            "shot_sd_proxy": shot_sd,
            "forecast_shot_sd_proxy": forecast_shot,
            "shot_to_train_std_ratio": ratio,
        })

    df = pd.DataFrame(details)
    active = df[df["active"]].copy()

    if active.empty:
        raise RuntimeError("No active features after scaling.")

    return {
        "max_abs_raw_prediction_sensitivity":
            float(active["raw_prediction_sensitivity"].abs().max()),
        "shot_noise_forecast_sd_proxy_1024":
            float(
                np.sqrt(
                    np.sum(
                        active["forecast_shot_sd_proxy"].to_numpy(float) ** 2
                    )
                )
            ),
        "max_shot_to_train_std_ratio_1024":
            float(active["shot_to_train_std_ratio"].max()),
        "all_features_shot_resolvable_1024":
            bool(
                np.all(
                    active["shot_to_train_std_ratio"].to_numpy(float) <= 1.0
                )
            ),
        "conditioning_details_json":
            json.dumps(details, separators=(",", ":")),
    }


def evaluate_readout(X, y_train, y_val, feature_names):
    X_train = X[: len(y_train)]
    X_val = X[len(y_train): len(y_train) + len(y_val)]

    best = select_lambda_training_only(
        X_train,
        y_train,
    )

    model, scaler, keep = fit_scaled_ridge(
        X_train,
        y_train,
        float(best["selected_lambda"]),
    )

    pred_val = predict_scaled_ridge(
        model,
        scaler,
        keep,
        X_val,
    )

    cond = conditioning_metrics(
        feature_names,
        X_train,
        model,
        scaler,
        keep,
    )

    return {
        "selected_lambda":
            float(best["selected_lambda"]),
        "cv_rmse":
            float(best["cv_rmse"]),
        "cv_rmse_std":
            float(best["cv_rmse_std"]),
        "validation_rmse":
            rmse(y_val, pred_val),
        "validation_bias":
            float(np.mean(pred_val - y_val)),
        **cond,
    }


def build_five_readout_screen(cont, forecast):
    """
    Frozen dynamics pool:
      unique config_id rows retained by 12.3C best-by-branch/r.

    For every frozen dynamics configuration evaluate all five common readouts.
    """
    configs = (
        forecast
        .sort_values(["config_id", "cv_rmse"])
        .drop_duplicates("config_id")
        .reset_index(drop=True)
    )

    y = np.asarray(cont.Y_TARGET, dtype=float)
    n_train = int(cont.N_TRAIN)
    n_val = int(cont.N_VAL)

    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]

    rows = []

    print(
        f"  frozen CONT dynamics configs: {len(configs)} | "
        f"readouts/config: {len(READOUTS)}"
    )

    for idx, cfg in configs.iterrows():
        topology = str(cfg["topology"])

        J = cont.J_from_row(topology, cfg)

        U = cont.trotter_unitary(
            topology=topology,
            hx=float(cfg["hx"]),
            hy=float(cfg["hy"]),
            J=J,
            r=int(cfg["trotter_r"]),
        )

        A_real, _ = cont.qrc.build_input_channels(
            U,
            cont.REAL_ANGLES,
        )

        bank = cont.base.build_feature_bank(A_real)

        for readout in READOUTS:
            features = list(READOUT_FEATURES[readout])

            # Resolve historical CONT column aliases without changing the
            # scientific observable or the public readout label.
            physical_columns = [
                (
                    feature
                    if feature in bank.columns
                    else FEATURE_COLUMN_ALIASES.get(feature, feature)
                )
                for feature in features
            ]

            missing = [
                display_name
                for display_name, physical_name
                in zip(features, physical_columns)
                if physical_name not in bank.columns
            ]

            if missing:
                raise RuntimeError(
                    f"{topology}/{cfg['config_id']} feature bank missing "
                    f"{missing}; available columns include "
                    f"{list(bank.columns)}"
                )

            X = bank[physical_columns].to_numpy(dtype=float)

            metrics = evaluate_readout(
                X,
                y_train,
                y_val,
                features,
            )

            S = READOUT_SETTINGS[readout]
            r = int(cfg["trotter_r"])

            row = cfg.to_dict()

            # Overwrite readout-dependent fields from the uniform five-readout screen.
            row.update({
                "readout": readout,
                "n_features": len(features),
                "n_settings": int(S),
                "feature_vector_cz":
                    int(2 * EDGE_COUNTS[topology] * r * S),
                "n_swap_logical_native": 0,
                **metrics,
            })

            rows.append(row)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(configs):
            print(
                f"  screened dynamics {idx+1}/{len(configs)} "
                f"({len(rows)} readout rows)"
            )

    return pd.DataFrame(rows)


def parse_rule_numbers(value):
    """Parse semantic rule metadata such as R1_accuracy or Rule 2."""
    text = str(value)
    found = set()

    for m in re.finditer(r"(?i)R([1-4])", text):
        found.add(f"R{m.group(1)}")

    for m in re.finditer(r"(?i)Rule\s*([1-4])", text):
        found.add(f"R{m.group(1)}")

    return sorted(found, key=lambda x: int(x[1]))


def parse_rule_numbers_from_candidate_id(value):
    """
    Candidate-ID fallback ONLY.

    This parser is intentionally case-sensitive:
      - historical H0-H4 IDs use uppercase rule tokens such as R1/R2/R3/R4
      - H5/H6 IQM IDs use lowercase r1/r2/r3 for Trotter depth

    Therefore lowercase "_r2_" must never become Rule 2.
    """
    text = str(value)
    found = {
        f"R{m.group(1)}"
        for m in re.finditer(r"R([1-4])", text)
    }
    return sorted(found, key=lambda x: int(x[1]))


def normalize_existing(existing):
    rows = []

    for _, row in existing.iterrows():
        d = row.to_dict()

        topology = str(d.get("topology", "")).strip()
        source_group = str(d.get("source_group", "")).strip()

        protocol = str(d.get("protocol", "")).strip().upper()

        if protocol not in PROTOCOLS:
            # Existing H5/H6 seven-candidate set is the frozen RWP selection.
            if topology in H5H6 and source_group == "H5_H6":
                protocol = "RWP"
            else:
                cid = str(d.get("candidate_id", "")).upper()
                if cid.startswith("CONT_"):
                    protocol = "CONT"
                elif cid.startswith("RWP_"):
                    protocol = "RWP"
                else:
                    raise RuntimeError(
                        f"Cannot resolve protocol for existing row {cid!r}"
                    )

        # Old H5/H6 manifest had RWP only. Any accidental CONT row is discarded
        # and rebuilt below from the consistent five-readout CONT screen.
        if topology in H5H6 and protocol == "CONT":
            continue

        # Parse semantic rule metadata first.
        # Never concatenate candidate_id here: H5/H6 IDs contain lowercase
        # Trotter tokens such as "_r2_" which are NOT Rule 2.
        rules = parse_rule_numbers(
            str(d.get("selected_rules", ""))
        )

        # Historical H0-H4 rows may rely on uppercase rule tokens embedded
        # in candidate_id. Use a strict case-sensitive fallback only.
        if not rules:
            rules = parse_rule_numbers_from_candidate_id(
                d.get("candidate_id", "")
            )

        if not rules:
            raise RuntimeError(
                f"No R1-R4 role found for existing candidate "
                f"{d.get('candidate_id')!r}"
            )

        d["protocol"] = protocol
        d["selected_rules"] = "+".join(rules)
        rows.append(d)

    return pd.DataFrame(rows)


def select_r3(group):
    safe = group[
        group["all_features_shot_resolvable_1024"].astype(bool)
    ].copy()

    if not safe.empty:
        row = (
            safe.sort_values(
                [
                    "cv_rmse",
                    "shot_noise_forecast_sd_proxy_1024",
                    "max_abs_raw_prediction_sensitivity",
                    "n_settings",
                    "feature_vector_cz",
                ],
                ascending=[True, True, True, True, True],
                kind="mergesort",
            )
            .iloc[0]
        )
        mode = "Rmax<=1 -> CV -> shotSD -> sensitivity -> settings -> N2q"
    else:
        row = (
            group.sort_values(
                [
                    "max_shot_to_train_std_ratio_1024",
                    "shot_noise_forecast_sd_proxy_1024",
                    "cv_rmse",
                    "max_abs_raw_prediction_sensitivity",
                    "n_settings",
                    "feature_vector_cz",
                ],
                ascending=[True, True, True, True, True, True],
                kind="mergesort",
            )
            .iloc[0]
        )
        mode = "no Rmax<=1 -> Rmax -> shotSD -> CV -> sensitivity -> settings -> N2q"

    return row, mode


def select_h5h6_cont_rules(screen, memory):
    selected = []

    for topology in H5H6:
        s = screen[screen["topology"].astype(str).eq(topology)].copy()
        m = memory[memory["topology"].astype(str).eq(topology)].copy()

        if s.empty or m.empty:
            raise RuntimeError(f"Missing H5/H6 CONT data for {topology}")

        # R1
        r1 = (
            s.sort_values(
                ["cv_rmse", "selected_lambda"],
                kind="mergesort",
            )
            .iloc[0]
        )
        selected.append(("R1", r1, "minimum chronological CV"))

        # R2: dynamics first, then best common readout on that dynamics.
        valid_mem = m[
            pd.to_numeric(
                m["valid_washout_seeds"],
                errors="coerce",
            ).fillna(0).astype(int).ge(5)
        ].copy()

        if valid_mem.empty:
            raise RuntimeError(
                f"{topology}: no all-five-seed washout-valid memory candidate"
            )

        best_mem_row = (
            valid_mem.sort_values(
                ["MC_ch_mean", "cv_rmse"],
                ascending=[False, True],
                kind="mergesort",
            )
            .iloc[0]
        )

        best_mem_config = str(best_mem_row["config_id"])

        same_dynamics = s[
            s["config_id"].astype(str).eq(best_mem_config)
        ].copy()

        if same_dynamics.empty:
            raise RuntimeError(
                f"{topology}: memory config {best_mem_config} absent from five-readout screen"
            )

        r2 = (
            same_dynamics.sort_values(
                ["cv_rmse", "selected_lambda"],
                kind="mergesort",
            )
            .iloc[0]
            .copy()
        )
        r2["MC_ch_mean"] = float(best_mem_row["MC_ch_mean"])
        r2["MC_ch_std"] = finite(best_mem_row.get("MC_ch_std"))
        r2["Tw_0.01_mean"] = finite(best_mem_row.get("Tw_0.01_mean"))
        r2["valid_washout_seeds"] = int(best_mem_row["valid_washout_seeds"])

        selected.append(
            (
                "R2",
                r2,
                "max MC_ch_mean with all 5 washout-valid seeds; then best-CV common readout",
            )
        )

        # R3
        r3, r3_mode = select_r3(s)
        selected.append(("R3", r3, r3_mode))

        # R4
        threshold = (
            float(r1["cv_rmse"])
            + float(r1["cv_rmse_std"]) / math.sqrt(N_FOLDS)
        )

        one_se = s[
            pd.to_numeric(s["cv_rmse"], errors="coerce").le(threshold)
        ].copy()

        r4 = (
            one_se.sort_values(
                [
                    "n_swap_logical_native",
                    "feature_vector_cz",
                    "n_settings",
                    "trotter_r",
                    "cv_rmse",
                ],
                ascending=[True, True, True, True, True],
                kind="mergesort",
            )
            .iloc[0]
        )

        selected.append(
            (
                "R4",
                r4,
                f"one-SE <= {threshold:.9f}; NSWAP->N2q->settings->depth(r)->CV",
            )
        )

    return selected


def build_cont_manifest(selected, cont):
    grouped = {}

    for rule, row, reason in selected:
        key = (
            str(row["topology"]),
            str(row["config_id"]),
            str(row["readout"]),
        )

        grouped.setdefault(
            key,
            {"row": row, "rules": [], "reasons": []},
        )

        grouped[key]["rules"].append(rule)
        grouped[key]["reasons"].append(f"{rule}:{reason}")

    out = []

    for _, item in grouped.items():
        row = item["row"]
        topology = str(row["topology"])
        rules = sorted(set(item["rules"]), key=lambda x: int(x[1]))

        J = cont.J_from_row(topology, row)

        out.append({
            "candidate_id":
                f"CONT_{topology}_{''.join(rules)}",
            "source_group":
                "H5_H6_CONT_UNIFIED",
            "selected_rules":
                "+".join(rules),
            "topology":
                topology,
            "protocol":
                "CONT",
            "window":
                np.nan,
            "r":
                int(row["trotter_r"]),
            "readout":
                str(row["readout"]),
            "cv_rmse":
                float(row["cv_rmse"]),
            "cv_rmse_std":
                float(row["cv_rmse_std"]),
            "validation_rmse":
                float(row["validation_rmse"]),
            "validation_bias":
                float(row["validation_bias"]),
            "selected_lambda":
                float(row["selected_lambda"]),
            "alpha":
                float(cont.ALPHA),
            "dt":
                float(cont.DT),
            "hx":
                float(row["hx"]),
            "hy":
                float(row["hy"]),
            "J_json":
                json.dumps(
                    {
                        f"J{a}{b}": float(v)
                        for (a, b), v in J.items()
                    },
                    sort_keys=True,
                ),
            "config_id":
                str(row["config_id"]),
            "dynamics_seed_id":
                str(row["dynamics_seed_id"]),
            "n_features":
                int(row["n_features"]),
            "n_settings":
                int(row["n_settings"]),
            "feature_vector_cz":
                int(row["feature_vector_cz"]),
            "MC_ch_mean":
                finite(row.get("MC_ch_mean")),
            "MC_ch_std":
                finite(row.get("MC_ch_std")),
            "Tw_0.01_mean":
                finite(row.get("Tw_0.01_mean")),
            "valid_washout_seeds":
                finite(row.get("valid_washout_seeds")),
            "shot_noise_forecast_sd_proxy_1024":
                finite(row.get("shot_noise_forecast_sd_proxy_1024")),
            "max_shot_to_train_std_ratio_1024":
                finite(row.get("max_shot_to_train_std_ratio_1024")),
            "max_abs_raw_prediction_sensitivity":
                finite(row.get("max_abs_raw_prediction_sensitivity")),
            "selection_notes":
                " | ".join(item["reasons"]),
        })

    return pd.DataFrame(out)


def audit_coverage(unified):
    """
    Strict assignment audit.

    Every H0-H6 x CONT/RWP cell must contain exactly one assignment for
    each R1, R2, R3 and R4.

    Multiple rules may map to the same unique candidate, so a cell may
    contain fewer than four unique rows. But it must still contain exactly
    four rule assignments in total.

    This prevents accidental contamination from lowercase Trotter IDs
    such as "_r2_".
    """
    rows = []
    all_pass = True
    global_assignments = 0

    for topology in TOPOLOGIES:
        for protocol in PROTOCOLS:
            cell = unified[
                unified["topology"].astype(str).eq(topology)
                & unified["protocol"].astype(str).eq(protocol)
            ]

            rule_counts = {rule: 0 for rule in RULES}

            for _, row in cell.iterrows():
                for rule in parse_rule_numbers(
                    row.get("selected_rules", "")
                ):
                    rule_counts[rule] += 1

            assignment_count = int(sum(rule_counts.values()))
            global_assignments += assignment_count

            missing = [
                rule for rule in RULES
                if rule_counts[rule] == 0
            ]
            duplicated = [
                rule for rule in RULES
                if rule_counts[rule] > 1
            ]

            passed = (
                len(cell) > 0
                and assignment_count == 4
                and all(rule_counts[rule] == 1 for rule in RULES)
            )

            rows.append({
                "topology": topology,
                "protocol": protocol,
                "unique_candidates": int(len(cell)),
                "rule_assignments": assignment_count,
                "R1_count": int(rule_counts["R1"]),
                "R2_count": int(rule_counts["R2"]),
                "R3_count": int(rule_counts["R3"]),
                "R4_count": int(rule_counts["R4"]),
                "rule_coverage": "+".join(
                    rule for rule in RULES
                    if rule_counts[rule] > 0
                ),
                "missing_rules": "+".join(missing),
                "duplicated_rules": "+".join(duplicated),
                "pass": bool(passed),
            })

            all_pass = all_pass and passed

    if global_assignments != 56:
        all_pass = False

    return pd.DataFrame(rows), all_pass, global_assignments


def main():
    for p in [EXISTING, CONT_FORECAST, CONT_MEMORY, CONT_MODULE_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    print("=" * 136)
    print("WEEK 12.5G0 v4 — UNIFIED H0-H6 CONT/RWP RULE-1..RULE-4 SELECTION")
    print("=" * 136)
    print("H5/H6 CONT is re-screened with the SAME five readout families.")
    print("No QPU jobs. No simulator jobs. 2026 untouched.")
    print()

    existing = pd.read_csv(EXISTING)
    forecast = pd.read_csv(CONT_FORECAST)
    memory = pd.read_csv(CONT_MEMORY)

    forecast = forecast[
        forecast["topology"].astype(str).isin(H5H6)
    ].copy()

    memory = memory[
        memory["topology"].astype(str).isin(H5H6)
    ].copy()

    cont = load_module(
        CONT_MODULE_PATH,
        "week12_03c_cont_unified",
    )

    print("[1/5] Building uniform five-readout H5/H6 CONT screen...")
    screen = build_five_readout_screen(
        cont,
        forecast,
    )
    screen.to_csv(OUT_SCREEN, index=False)

    print()
    print("[2/5] Applying R1-R4 to H5/H6 CONT...")
    selected = select_h5h6_cont_rules(
        screen,
        memory,
    )

    cont_manifest = build_cont_manifest(
        selected,
        cont,
    )
    cont_manifest.to_csv(OUT_CONT, index=False)

    print()
    print(
        cont_manifest[
            [
                "candidate_id",
                "selected_rules",
                "topology",
                "protocol",
                "r",
                "readout",
                "cv_rmse",
                "feature_vector_cz",
            ]
        ].to_string(index=False)
    )

    print()
    print("[3/5] Normalizing existing H0-H4 CONT/RWP + H5/H6 RWP candidates...")
    existing_norm = normalize_existing(existing)

    print("[4/5] Building unified candidate universe...")
    unified = pd.concat(
        [existing_norm, cont_manifest],
        ignore_index=True,
        sort=False,
    )

    unified["protocol"] = unified["protocol"].astype(str).str.upper()

    unified = (
        unified.sort_values(
            ["topology", "protocol", "candidate_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    unified.to_csv(
        OUT_UNIFIED,
        index=False,
    )

    print("[5/5] Auditing 14 topology/protocol cells for complete R1-R4 coverage...")
    audit, passed, global_rule_assignments = audit_coverage(unified)
    audit.to_csv(OUT_AUDIT, index=False)

    print()
    print(audit.to_string(index=False))

    counts = (
        unified.groupby(["topology", "protocol"])
        .size()
        .rename("unique_candidates")
        .reset_index()
    )

    manifest = {
        "stage": "12.5G0_v4",
        "same_rules_all_H0_H6": True,
        "same_protocols_all_H0_H6": True,
        "same_five_readout_families_for_h5_h6_cont": True,
        "candidate_count": int(len(unified)),
        "coverage_pass": bool(passed),
        "cells_passed": int(audit["pass"].astype(bool).sum()),
        "cells_expected": 14,
        "selection": {
            "R1": "minimum training chronological CV",
            "R2_CONT": "maximum MC/ch with all five Tw<=300, then best-CV common readout",
            "R2_RWP": "frozen replay-memory representative",
            "R3": "1024-shot resolvability hierarchy",
            "R4": "one-SE then resources",
        },
        "h0_h4_reoptimized": False,
        "h5_h6_rwp_reoptimized": False,
        "h5_h6_cont_dynamics_reoptimized": False,
        "h5_h6_cont_readouts_rescreened": READOUTS,
        "historical_feature_aliases": {
            "YX45": "YX_45"
        },
        "rule_label_parsing": {
            "selected_rules": "semantic metadata parsed case-insensitively",
            "candidate_id_fallback": "uppercase R1-R4 only, case-sensitive",
            "lowercase_trotter_r_never_interpreted_as_rule": True
        },
        "strict_rule_assignment_audit": {
            "cells": 14,
            "assignments_per_cell": 4,
            "expected_global_rule_assignments": 56,
            "observed_global_rule_assignments": int(global_rule_assignments),
            "each_rule_exactly_once_per_cell": True
        },
        "validation_2025_used_for_selection": False,
        "test_2026_used": False,
        "qpu_jobs": 0,
        "simulator_jobs": 0,
        "counts_by_cell": counts.to_dict(orient="records"),
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print()
    print("=" * 136)
    print("12.5G0 v4 SUMMARY")
    print("=" * 136)
    print(f"Unified candidates after deduplication: {len(unified)}")
    print(f"Rule assignments before deduplication:  {global_rule_assignments}/56")
    print(f"Complete topology/protocol cells:       {int(audit['pass'].sum())}/14")
    print(f"Strict assignment audit:                {'PASS' if passed else 'FAIL'}")
    print()
    print("Saved:")
    for p in [OUT_SCREEN, OUT_CONT, OUT_UNIFIED, OUT_AUDIT, OUT_MANIFEST]:
        print(f"  {p}")

    if not passed:
        raise RuntimeError(
            "Strict R1-R4 assignment audit failed. Every topology/protocol "
            "cell must contain exactly one R1, one R2, one R3 and one R4 "
            "assignment (56 assignments globally). Do NOT continue."
        )

    print()
    print(
        "PASS. The fixed dual-backend handoff may now consume "
        "12_05g0_unified_rule1_rule4_candidates.csv."
    )


if __name__ == "__main__":
    main()
