"""
Week 12.3D shared RWP utilities.

Direct IQM H5/H6 adaptation of IBM `09_04_rwp_common.py`.
Scientific logic is intentionally kept one-to-one with the IBM branch.
Only topology set, source result file, labels, and hardware-specific comments
are adapted.

2022-2024: training/CV/conditioning
2025: diagnostic only
2026: untouched
"""


from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

QRC_SCRIPT = HERE / "07_02b_memory_pair_isolation.py"
FAIR_FILE = RESULTS / "12_03c_forecast_best_by_branch_r.csv"

N_TRAIN = 1095
N_VAL = 365

DEFAULT_ALPHA = 0.75
DEFAULT_DT = 1.6

DEFAULT_TOPOLOGIES = ["H5", "H6"]
DEFAULT_WINDOWS = list(range(1, 29))

# Same five readout families used in the IBM 09.04 pipeline.
#
# For one-to-one IBM↔IQM symmetry, DROP-X3 variants remain explicit candidate
# families. They are NOT preselected for IQM; training-only conditioning and
# finite-shot resolvability decide whether they are useful.
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

# Human-readable grouped measurement plans used for hardware resource reporting.
READOUT_MEASUREMENT_PLAN = {
    "XZ_injection": "XXXX / ZZZZ",
    "XZinj_dropX3": "XXXZ / ZZZZ",
    "XZinj_dropX3_plus_YX45": "XXXZYX / ZZZZZZ",
    "XZinj_plus_YX45": "XXXXYX / ZZZZZZ",
    "XYZ_all": "XXXXXX / YYYYYY / ZZZZZZ",
}

# Training-only finite-shot proxy.  Keep 1024 shots exactly as in the IBM
# 09.04 pipeline for matched methodology. This is a diagnostic, not an IQM QPU model.
SHOTS_PROXY = 1024

LAMBDA_GRID = np.array(
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0],
    dtype=float,
)

VAR_TOL = 1e-12

TOPOLOGY_EDGES = {
    # Keep the exact edge ordering used in Week 12.3A/12.3C so the
    # Trotter product ordering is unchanged.
    "H5": [
        (3, 0), (0, 4), (2, 1), (1, 5), (3, 2), (0, 1), (4, 5),
    ],
    "H6": [
        (0, 1), (1, 5), (5, 2), (2, 3), (3, 4), (4, 0), (4, 5),
    ],
}

EDGE_TO_COLUMN = {
    "H5": {
        (3, 0): "J03",
        (0, 4): "J04",
        (2, 1): "J12",
        (1, 5): "J15",
        (3, 2): "J23",
        (0, 1): "J01",
        (4, 5): "J45",
    },
    "H6": {
        (0, 1): "J01",
        (1, 5): "J15",
        (5, 2): "J25",
        (2, 3): "J23",
        (3, 4): "J34",
        (4, 0): "J04",
        (4, 5): "J45",
    },
}


def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the Week-9.04 scripts in the project root."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    spec.loader.exec_module(module)
    return module


qrc = import_module_from_path("qrc_week7_for_904", QRC_SCRIPT)


def parse_windows(spec: str | None) -> list[int]:
    if spec is None or spec.strip() == "":
        return DEFAULT_WINDOWS.copy()

    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        a = int(a)
        b = int(b)
        if a < 1 or b < a:
            raise ValueError("Invalid window range.")
        return list(range(a, b + 1))

    vals = sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
    if not vals or min(vals) < 1:
        raise ValueError("Windows must be positive integers.")
    return vals


def parse_topologies() -> list[str]:
    # One-to-one IQM adaptation of IBM 09.04A:
    # the new native topology set is H5/H6.
    return DEFAULT_TOPOLOGIES.copy()


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(a - b)))


def bias(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.mean(b - a))


def load_train_validation():
    work, train, val, cols = qrc.load_data()

    if len(train) != N_TRAIN or len(val) != N_VAL:
        raise RuntimeError(
            f"Expected train/validation={N_TRAIN}/{N_VAL}; "
            f"got {len(train)}/{len(val)}."
        )

    # Deliberately construct only 2022-2025.
    # 2026 remains frozen and is not part of Week 9.04.
    work_tv = pd.concat([train, val], axis=0).reset_index(drop=True)
    y = work_tv[cols["target"]].to_numpy(dtype=float)

    return work_tv, cols, y


def make_angles(work_tv, cols, alpha: float) -> np.ndarray:
    old_alpha = getattr(qrc, "ALPHA", DEFAULT_ALPHA)
    qrc.ALPHA = float(alpha)
    try:
        angles = np.asarray(
            qrc.make_input_angles(work_tv, cols),
            dtype=float,
        )
    finally:
        qrc.ALPHA = old_alpha
    return angles


def load_topology_baselines(
    topologies: Iterable[str],
) -> dict[str, dict]:
    if not FAIR_FILE.exists():
        raise FileNotFoundError(
            f"{FAIR_FILE} not found. Week 12.3C must be complete first."
        )

    df = pd.read_csv(FAIR_FILE)
    out = {}

    for topology in topologies:
        sub = df[df["topology"].astype(str) == topology].copy()
        if len(sub) == 0:
            raise RuntimeError(
                f"No 9.3c candidate found for topology {topology}."
            )

        row = (
            sub.sort_values(["cv_rmse", "validation_rmse"])
            .iloc[0]
        )

        J = {
            edge: float(row[column])
            for edge, column in EDGE_TO_COLUMN[topology].items()
        }

        out[topology] = {
            "source_config_id": str(row["config_id"]),
            "source_candidate_id": str(row["seed_candidate_id"]),
            "topology": topology,
            "alpha": float(
                row["alpha"]
                if "alpha" in row.index and pd.notna(row["alpha"])
                else DEFAULT_ALPHA
            ),
            "dt": float(
                row["dt"]
                if "dt" in row.index and pd.notna(row["dt"])
                else DEFAULT_DT
            ),
            "r": int(row["trotter_r"]),
            "hx": float(row["hx"]),
            "hy": float(row["hy"]),
            "J": J,
            "source_readout": str(row["readout"]),
            "source_lambda": float(row["selected_lambda"]),
            "source_cv_rmse": float(row["cv_rmse"]),
            "source_validation_rmse": float(row["validation_rmse"]),
        }

    return out


# -----------------------------------------------------------------------------
# Quantum dynamics
# -----------------------------------------------------------------------------

I64 = qrc.I64

INJ_OPS = getattr(
    qrc,
    "INJECTION_SINGLE_OPS",
    getattr(qrc, "INJECTION_OPS", None),
)
MEM_OPS = getattr(
    qrc,
    "MEMORY_SINGLE_OPS",
    getattr(qrc, "MEMORY_OPS", None),
)

if INJ_OPS is None or MEM_OPS is None:
    raise AttributeError(
        "Could not resolve injection/memory Pauli operator dictionaries."
    )

ZZ_OPS = {}
for topology, edges in TOPOLOGY_EDGES.items():
    for edge in edges:
        if edge not in ZZ_OPS:
            ZZ_OPS[edge] = qrc.pauli_product_full(
                qrc.Z2, edge[0], qrc.Z2, edge[1]
            )

YX45_MEM = MEM_OPS["Y4"] @ MEM_OPS["X5"]


def pauli_exp(operator, theta):
    return np.cos(theta) * I64 - 1j * np.sin(theta) * operator


def build_trotter_unitary(candidate: dict) -> np.ndarray:
    topology = candidate["topology"]
    r = int(candidate["r"])
    dt = float(candidate["dt"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])
    J = candidate["J"]

    U = I64.copy()

    for _ in range(r):
        for edge in TOPOLOGY_EDGES[topology]:
            theta = float(J[edge]) * dt / r
            U = pauli_exp(ZZ_OPS[edge], theta) @ U

        theta_x = hx * dt / r
        for q in range(qrc.N_QUBITS):
            U = (
                pauli_exp(
                    qrc.FULL_SINGLE_OPS[f"X{q}"],
                    theta_x,
                )
                @ U
            )

        if not np.isclose(hy, 0.0):
            theta_y = hy * dt / r
            for q in [4, 5]:
                U = (
                    pauli_exp(
                        qrc.FULL_SINGLE_OPS[f"Y{q}"],
                        theta_y,
                    )
                    @ U
                )

    return U


def expectation(rho, op) -> float:
    return float(
        np.real_if_close(
            np.trace(rho @ op),
            tol=1000,
        ).real
    )


def reduced_feature_row(rho_i, rho_m) -> dict[str, float]:
    row: dict[str, float] = {}

    for q in range(4):
        for p in ["X", "Y", "Z"]:
            name = f"{p}{q}"
            row[name] = expectation(rho_i, INJ_OPS[name])

    for q in [4, 5]:
        for p in ["X", "Y", "Z"]:
            name = f"{p}{q}"
            row[name] = expectation(rho_m, MEM_OPS[name])

    row["YX45"] = expectation(rho_m, YX45_MEM)
    return row


def build_channels(candidate: dict, angles: np.ndarray):
    U = build_trotter_unitary(candidate)
    A_list, _ = qrc.build_input_channels(U, angles)
    return A_list


def rwp_master_feature_bank_from_channels(
    A_list,
    window: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Return one independent RWP feature vector for every endpoint that has
    `window` available inputs.

    Each endpoint starts from the fixed memory state |00><00|.
    q4,q5 carry memory only *inside* that window.
    The next endpoint restarts from |00><00| again.
    """
    if window < 1:
        raise ValueError("window must be >=1")

    endpoints = np.arange(
        window - 1,
        len(A_list),
        dtype=int,
    )

    rows = []

    for endpoint in endpoints:
        rho_m = qrc.memory_zero_density()
        rho_i = None

        start = int(endpoint) - window + 1

        for idx in range(start, int(endpoint) + 1):
            rho_i, rho_m = qrc.final_reduced_states(
                A_list[idx],
                rho_m,
            )

        rows.append(
            reduced_feature_row(rho_i, rho_m)
        )

    return endpoints, pd.DataFrame(rows)


def select_features(master_df: pd.DataFrame, readout: str) -> np.ndarray:
    cols = READOUT_FEATURES[readout]
    return master_df[cols].to_numpy(dtype=float)


# -----------------------------------------------------------------------------
# Ridge readout: training-only chronology
# -----------------------------------------------------------------------------

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


def select_lambda_training_only(X_train, y_train, n_splits=5):
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []

    for ridge_alpha in LAMBDA_GRID:
        for fold, (fit_idx, cv_idx) in enumerate(
            splitter.split(X_train)
        ):
            model, scaler, keep = fit_scaled_ridge(
                X_train[fit_idx],
                y_train[fit_idx],
                ridge_alpha,
            )

            pred = predict_scaled_ridge(
                model,
                scaler,
                keep,
                X_train[cv_idx],
            )

            fold_rows.append({
                "lambda": float(ridge_alpha),
                "fold": fold + 1,
                "rmse": rmse(y_train[cv_idx], pred),
            })

    folds = pd.DataFrame(fold_rows)

    summary = (
        folds.groupby("lambda", as_index=False)
        .agg(
            cv_rmse_mean=("rmse", "mean"),
            cv_rmse_std=("rmse", "std"),
        )
        .sort_values(["cv_rmse_mean", "lambda"])
        .reset_index(drop=True)
    )

    return float(summary.iloc[0]["lambda"]), folds, summary



def readout_conditioning_metrics(
    feature_names: list[str],
    X_train: np.ndarray,
    model,
    scaler,
    keep: np.ndarray,
    shots_proxy: int = SHOTS_PROXY,
) -> dict:
    """Training-only hardware-robustness diagnostics for a fitted Ridge readout.

    If the readout is

        z_j = (x_j - mu_j) / sigma_j,
        yhat += w_j z_j,

    then the sensitivity to a raw observable perturbation is

        d yhat / d x_j = w_j / sigma_j.

    For a Pauli observable measured with N shots, a finite-shot proxy is

        sigma_P,j ~= sqrt((1 - mean_j^2) / N).

    We also compare this measurement uncertainty with the natural training
    variation of the feature:

        resolvability_ratio_j = sigma_P,j / std_train,j.

    Ratio > 1 means the shot-noise standard deviation is larger than the
    feature's own training standard deviation.  This was the failure mode
    diagnosed for X3 on the real QPU.
    """
    X_train = np.asarray(X_train, dtype=float)
    keep = np.asarray(keep, dtype=bool)

    active_idx = np.flatnonzero(keep)
    coef = np.asarray(model.coef_, dtype=float)

    if len(active_idx) != len(coef):
        raise RuntimeError("Ridge coefficient / active-feature mismatch.")

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
                / float(shots_proxy)
            )
        )

        if active:
            raw_sensitivity = float(coef_full[j] / scale_full[j])
            forecast_shot_sd = float(abs(raw_sensitivity) * shot_sd)
        else:
            raw_sensitivity = 0.0
            forecast_shot_sd = 0.0

        if train_std > VAR_TOL:
            resolvability_ratio = float(shot_sd / train_std)
        else:
            resolvability_ratio = float("inf")

        details.append({
            "feature": feature,
            "active": active,
            "train_mean": train_mean,
            "train_std": train_std,
            "ridge_weight_standardized": float(coef_full[j]),
            "raw_prediction_sensitivity": raw_sensitivity,
            "shot_sd_proxy": shot_sd,
            "forecast_shot_sd_proxy": forecast_shot_sd,
            "shot_to_train_std_ratio": resolvability_ratio,
        })

    detail_df = pd.DataFrame(details)
    active_df = detail_df[detail_df["active"]].copy()

    if len(active_df) == 0:
        raise RuntimeError("No active features after scaling.")

    worst_sens_idx = active_df["raw_prediction_sensitivity"].abs().idxmax()
    worst_res_idx = active_df["shot_to_train_std_ratio"].idxmax()

    total_forecast_shot_sd = float(
        np.sqrt(
            np.sum(
                active_df["forecast_shot_sd_proxy"].to_numpy(dtype=float) ** 2
            )
        )
    )

    return {
        "max_abs_raw_prediction_sensitivity": float(
            active_df["raw_prediction_sensitivity"].abs().max()
        ),
        "worst_sensitivity_feature": str(
            detail_df.loc[worst_sens_idx, "feature"]
        ),
        "shot_noise_forecast_sd_proxy_1024": total_forecast_shot_sd,
        "max_shot_to_train_std_ratio_1024": float(
            active_df["shot_to_train_std_ratio"].max()
        ),
        "worst_resolvability_feature": str(
            detail_df.loc[worst_res_idx, "feature"]
        ),
        "all_features_shot_resolvable_1024": bool(
            np.all(active_df["shot_to_train_std_ratio"].to_numpy(dtype=float) <= 1.0)
        ),
        "min_active_train_feature_std": float(
            active_df["train_std"].min()
        ),
        "conditioning_details_json": json.dumps(
            details,
            separators=(",", ":"),
        ),
    }


def select_hardware_aware_row(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Select one hardware-aware representative without a weighted score.

    Hierarchy:
      1. If any candidate has shot-noise SD <= training SD for every active
         feature at 1024 shots, restrict to that physically interpretable set.
         Then choose lowest training CV RMSE and use robustness/resource metrics
         only as tie-breakers.
      2. If none is fully shot-resolvable, first minimize the worst
         shot-to-training-STD ratio, then forecast-shot-noise proxy, then CV.

    2025 validation is never used for selection.
    """
    work = df.copy()

    safe = work[
        work["all_features_shot_resolvable_1024"].astype(bool)
    ].copy()

    if len(safe) > 0:
        row = (
            safe.sort_values(
                [
                    "cv_rmse",
                    "shot_noise_forecast_sd_proxy_1024",
                    "max_abs_raw_prediction_sensitivity",
                    "n_settings",
                    "feature_vector_cz",
                    "n_features",
                ]
            )
            .iloc[0]
        )
        mode = "shot-resolvable set -> training CV -> robustness/resources"
    else:
        row = (
            work.sort_values(
                [
                    "max_shot_to_train_std_ratio_1024",
                    "shot_noise_forecast_sd_proxy_1024",
                    "cv_rmse",
                    "max_abs_raw_prediction_sensitivity",
                    "n_settings",
                    "feature_vector_cz",
                    "n_features",
                ]
            )
            .iloc[0]
        )
        mode = "no fully shot-resolvable candidate -> resolvability -> shot proxy -> training CV"

    return row, mode


def evaluate_master_bank(
    endpoints: np.ndarray,
    master_df: pd.DataFrame,
    y_all: np.ndarray,
    window: int,
    readout: str,
    n_splits: int = 5,
) -> dict:
    X = select_features(master_df, readout)
    y_endpoint = y_all[endpoints]

    # All training endpoints must lie inside the 2022-2024 region.
    train_mask = endpoints < N_TRAIN
    val_mask = endpoints >= N_TRAIN

    X_train = X[train_mask]
    y_train = y_endpoint[train_mask]

    X_val = X[val_mask]
    y_val = y_endpoint[val_mask]

    val_endpoints = endpoints[val_mask]

    if len(y_val) != N_VAL:
        raise RuntimeError(
            f"W={window}: expected {N_VAL} validation rows, got {len(y_val)}."
        )

    selected_lambda, folds, summary = select_lambda_training_only(
        X_train,
        y_train,
        n_splits=n_splits,
    )

    model, scaler, keep = fit_scaled_ridge(
        X_train,
        y_train,
        selected_lambda,
    )

    pred_val = predict_scaled_ridge(
        model,
        scaler,
        keep,
        X_val,
    )

    conditioning = readout_conditioning_metrics(
        feature_names=READOUT_FEATURES[readout],
        X_train=X_train,
        model=model,
        scaler=scaler,
        keep=keep,
        shots_proxy=SHOTS_PROXY,
    )

    return {
        "window": int(window),
        "readout": readout,
        "measurement_plan": READOUT_MEASUREMENT_PLAN[readout],
        "n_features": len(READOUT_FEATURES[readout]),
        "n_settings": READOUT_SETTINGS[readout],
        "n_train_endpoints": int(len(y_train)),
        "n_validation_endpoints": int(len(y_val)),
        "selected_lambda": float(selected_lambda),
        "cv_rmse": float(summary.iloc[0]["cv_rmse_mean"]),
        "cv_rmse_std": float(summary.iloc[0]["cv_rmse_std"]),
        "validation_rmse": rmse(y_val, pred_val),
        "validation_mae": mae(y_val, pred_val),
        "validation_bias": bias(y_val, pred_val),
        "active_feature_count": int(np.sum(keep)),
        **conditioning,
        "folds": folds,
        "cv_summary": summary,
    }


def resource_metrics(candidate: dict, window: int, readout: str) -> dict:
    r = int(candidate["r"])
    n_edges = len(TOPOLOGY_EDGES[candidate["topology"]])

    # Matched logical resource accounting: one RZZ is counted as 2 CZ-equivalent\n    # entangling gates, exactly as in IBM 09.04. Actual IQM transpiled cost is\n    # measured later in the hardware/resource stage.
    core_cz_per_step = 2 * n_edges * r
    core_cz_window = core_cz_per_step * int(window)

    n_settings = READOUT_SETTINGS[readout]

    return {
        "core_cz_per_step": int(core_cz_per_step),
        "core_cz_per_window": int(core_cz_window),
        "feature_vector_cz": int(core_cz_window * n_settings),
        "logical_reset_count": int(4 * max(window - 1, 0)),
        "n_settings": int(n_settings),
        "n_features": int(len(READOUT_FEATURES[readout])),
    }


def candidate_to_flat(candidate: dict) -> dict:
    row = {
        "topology": candidate["topology"],
        "alpha": float(candidate["alpha"]),
        "dt": float(candidate["dt"]),
        "r": int(candidate["r"]),
        "hx": float(candidate["hx"]),
        "hy": float(candidate["hy"]),
    }

    for edge, value in candidate["J"].items():
        row[f"J{edge[0]}{edge[1]}"] = float(value)

    for key in [
        "source_config_id",
        "source_candidate_id",
        "candidate_id",
        "search_id",
        "parent_id",
    ]:
        if key in candidate:
            row[key] = candidate[key]

    return row


def candidate_from_row(row: pd.Series | dict) -> dict:
    if isinstance(row, pd.Series):
        row = row.to_dict()

    topology = str(row["topology"])
    J = {}

    for edge, col in EDGE_TO_COLUMN[topology].items():
        if col not in row or pd.isna(row[col]):
            raise KeyError(
                f"Missing {col} for topology {topology}."
            )
        J[edge] = float(row[col])

    return {
        "topology": topology,
        "alpha": float(row.get("alpha", DEFAULT_ALPHA)),
        "dt": float(row.get("dt", DEFAULT_DT)),
        "r": int(row["r"] if "r" in row else row["trotter_r"]),
        "hx": float(row["hx"]),
        "hy": float(row["hy"]),
        "J": J,
        "candidate_id": str(
            row.get(
                "candidate_id",
                row.get("search_id", ""),
            )
        ),
    }


def evaluate_candidate_window(
    candidate: dict,
    window: int,
    work_tv: pd.DataFrame,
    cols: dict,
    y_all: np.ndarray,
    readouts: Iterable[str] = READOUTS,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict]:
    angles = make_angles(
        work_tv,
        cols,
        candidate["alpha"],
    )

    A_list = build_channels(
        candidate,
        angles,
    )

    endpoints, master = rwp_master_feature_bank_from_channels(
        A_list,
        window,
    )

    rows = []

    for readout in readouts:
        metrics = evaluate_master_bank(
            endpoints=endpoints,
            master_df=master,
            y_all=y_all,
            window=window,
            readout=readout,
            n_splits=n_splits,
        )

        row = {
            **candidate_to_flat(candidate),
            **{
                k: v
                for k, v in metrics.items()
                if k not in {"folds", "cv_summary"}
            },
            **resource_metrics(candidate, window, readout),
        }

        rows.append(row)

    all_df = pd.DataFrame(rows)

    # Selection uses training-CV as the primary criterion.
    # Complexity breaks numerical ties; 2025 validation is diagnostic only.
    winner = (
        all_df.sort_values(
            [
                "cv_rmse",
                "n_settings",
                "n_features",
                "feature_vector_cz",
                "selected_lambda",
            ]
        )
        .iloc[0]
        .to_dict()
    )

    return all_df, winner


# -----------------------------------------------------------------------------
# Pareto utilities
# -----------------------------------------------------------------------------

def pareto_mask(df: pd.DataFrame, minimize: list[str]) -> np.ndarray:
    values = df[minimize].to_numpy(dtype=float)
    n = len(df)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue

        vi = values[i]

        for j in range(n):
            if i == j:
                continue

            vj = values[j]

            if np.all(vj <= vi) and np.any(vj < vi):
                keep[i] = False
                break

    return keep


# -----------------------------------------------------------------------------
# Equal-budget Stage-C local/coordinate search
# -----------------------------------------------------------------------------

def _clone_candidate(parent: dict) -> dict:
    return {
        **{
            k: v
            for k, v in parent.items()
            if k != "J"
        },
        "J": {
            edge: float(value)
            for edge, value in parent["J"].items()
        },
    }


def stage_c_candidates(parent: dict) -> list[dict]:
    """
    Equal-budget deterministic search around each H,W parent.

    r is deliberately frozen here. Trotter depth is re-optimized in 9.04C2.
    """
    out = []

    def add(tag, updater):
        c = _clone_candidate(parent)
        updater(c)
        c["search_id"] = tag
        c["parent_id"] = parent.get("candidate_id", "")
        out.append(c)

    add("base", lambda c: None)

    # Alpha: hardware-relevant range.
    for alpha in [0.25, 0.50, 0.75, 1.00]:
        if not np.isclose(alpha, parent["alpha"]):
            add(
                f"alpha_{alpha:g}",
                lambda c, a=alpha: c.update(alpha=float(a)),
            )

    # Delta t.
    for dt in [0.8, 1.2, 1.6, 2.0]:
        if not np.isclose(dt, parent["dt"]):
            add(
                f"dt_{dt:g}",
                lambda c, d=dt: c.update(dt=float(d)),
            )

    # hx scale.
    for scale in [0.75, 1.25]:
        add(
            f"hxscale_{scale:g}",
            lambda c, s=scale: c.update(hx=float(c["hx"]) * s),
        )

    # hy: weaker, stronger, off, and sign flip.
    if np.isclose(parent["hy"], 0.0):
        for hy in [-0.3, 0.3]:
            add(
                f"hy_{hy:+g}",
                lambda c, h=hy: c.update(hy=float(h)),
            )
    else:
        for scale in [0.5, 1.5, -1.0]:
            add(
                f"hyscale_{scale:+g}",
                lambda c, s=scale: c.update(hy=float(c["hy"]) * s),
            )
        add("hy_off", lambda c: c.update(hy=0.0))

    # Global J scale.
    for scale in [0.75, 1.25]:
        def update_j(c, s=scale):
            c["J"] = {
                edge: float(value) * s
                for edge, value in c["J"].items()
            }
        add(f"Jglobal_{scale:g}", update_j)

    # Individual-edge local perturbations ±15%.
    for edge in TOPOLOGY_EDGES[parent["topology"]]:
        for scale in [0.85, 1.15]:
            def update_edge(c, e=edge, s=scale):
                c["J"][e] = float(c["J"][e]) * s
            add(
                f"J{edge[0]}{edge[1]}_{scale:g}",
                update_edge,
            )

    # Combined mild parents to expose interactions without a full Cartesian grid.
    for tag, a_scale, hx_scale, j_scale, dt_scale in [
        ("combo_soft", 0.75, 0.90, 0.90, 0.90),
        ("combo_strong", 1.00, 1.10, 1.10, 1.10),
        ("combo_inputstrong_dynsoft", 1.00, 0.90, 0.90, 1.00),
        ("combo_inputsoft_dynstrong", 0.50, 1.10, 1.10, 1.00),
    ]:
        def update_combo(
            c,
            a=a_scale,
            hs=hx_scale,
            js=j_scale,
            ds=dt_scale,
        ):
            c["alpha"] = float(a)
            c["hx"] = float(c["hx"]) * hs
            c["dt"] = float(c["dt"]) * ds
            c["J"] = {
                edge: float(value) * js
                for edge, value in c["J"].items()
            }
            c["hy"] = float(c["hy"]) * hs

        add(tag, update_combo)

    # Deduplicate by numerical parameter signature.
    unique = {}
    for c in out:
        sig = (
            c["topology"],
            round(float(c["alpha"]), 12),
            round(float(c["dt"]), 12),
            int(c["r"]),
            round(float(c["hx"]), 12),
            round(float(c["hy"]), 12),
            tuple(
                round(float(c["J"][e]), 12)
                for e in TOPOLOGY_EDGES[c["topology"]]
            ),
        )
        unique.setdefault(sig, c)

    return list(unique.values())


# -----------------------------------------------------------------------------
# Stage-C2: fair local Trotter search around the Stage-C winner
# -----------------------------------------------------------------------------

def trotter_local_candidates(
    parent: dict,
    r: int,
    seed: int = 42,
    n_random: int = 8,
) -> list[dict]:
    """
    Equal local budget for every r.

    The Stage-C winner is the shared parent. For every r, evaluate:
      - exact same parent Hamiltonian parameters,
      - deterministic ±10% global dynamics variations,
      - n_random small local perturbations with a fixed seed.

    This avoids the unfair 'change r only' comparison that Week 9.3 showed
    can favor the r around which parameters were originally optimized.
    """
    rng = np.random.default_rng(
        seed
        +
        1009 * int(r)
        +
        37 * int(parent.get("window", 0))
    )

    out = []

    def add(tag, c):
        c = _clone_candidate(c)
        c["r"] = int(r)
        c["search_id"] = tag
        c["parent_id"] = parent.get("candidate_id", "")
        out.append(c)

    base = _clone_candidate(parent)
    add("parent_same_H", base)

    for scale in [0.90, 1.10]:
        c = _clone_candidate(parent)
        c["hx"] *= scale
        c["hy"] *= scale
        c["J"] = {
            e: v * scale
            for e, v in c["J"].items()
        }
        add(f"global_dyn_{scale:g}", c)

    for dt_scale in [0.90, 1.10]:
        c = _clone_candidate(parent)
        c["dt"] *= dt_scale
        add(f"dt_{dt_scale:g}", c)

    for k in range(int(n_random)):
        c = _clone_candidate(parent)

        c["hx"] *= float(
            np.clip(
                rng.normal(1.0, 0.08),
                0.80,
                1.20,
            )
        )

        if not np.isclose(c["hy"], 0.0):
            c["hy"] *= float(
                np.clip(
                    rng.normal(1.0, 0.10),
                    0.75,
                    1.25,
                )
            )

        c["dt"] *= float(
            np.clip(
                rng.normal(1.0, 0.06),
                0.85,
                1.15,
            )
        )

        c["J"] = {
            edge: float(value)
            *
            float(
                np.clip(
                    rng.normal(1.0, 0.10),
                    0.75,
                    1.25,
                )
            )
            for edge, value in c["J"].items()
        }

        add(f"local_{k:02d}", c)

    return out


def write_csv_incremental(
    path: Path,
    rows: list[dict],
):
    if not rows:
        return

    df = pd.DataFrame(rows)
    exists = path.exists()

    df.to_csv(
        path,
        mode="a" if exists else "w",
        header=not exists,
        index=False,
    )


def completed_keys(
    path: Path,
    cols: list[str],
) -> set[tuple]:
    if not path.exists():
        return set()

    df = pd.read_csv(path)

    missing = [
        c for c in cols
        if c not in df.columns
    ]

    if missing:
        return set()

    return set(
        tuple(row[c] for c in cols)
        for _, row in df.iterrows()
    )
