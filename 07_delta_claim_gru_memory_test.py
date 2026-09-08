"""
DELTA-CLAIM TEMPORAL-MEMORY DIAGNOSTIC
======================================

Question
--------
Does predicting next-day claim CHANGE

    Delta C_{t+1} = C_{t+1} - C_t

make the insurance task more dependent on temporal history than predicting the
next-day level C_{t+1}?

We repeat the established Week-4 GRU screening as closely as possible:

    Feature sets: F2, F3, F4
    Windows:      1, 2, 5, 7, 14, 21, 28
    Hidden sizes: 1, 2, 4, 8, 16, 32

Frozen GRU settings:
    learning rate = 1e-3
    L2            = 1e-4
    layers        = 1
    batch size    = 64
    max epochs    = 2000
    patience      = 30
    min delta     = 1e-4
    CV folds      = 5
    base seed     = 42

Selection:
    5-fold chronological CV on 2022-2024 only.

Diagnostic:
    2025 validation after model/window/capacity selection.

Untouched:
    2026 test rows are excluded completely.

Important identity
------------------
If the predicted delta is converted back to a count,

    C_hat_{t+1} = C_t + DeltaC_hat_{t+1},

then

    C_{t+1} - C_hat_{t+1}
      = Delta C_{t+1} - DeltaC_hat_{t+1}.

Therefore delta RMSE and reconstructed-count RMSE are mathematically identical.
The purpose of this experiment is NOT to manufacture a different error scale;
it is to test whether the GRU's optimal temporal window changes.

Input
-----
results/03_01_preprocessed_samples.csv

Outputs
-------
results/07_delta_gru_cv_results.csv
results/07_delta_gru_best_per_feature_window.csv
results/07_delta_gru_memory_gain_vs_W1.csv
results/07_delta_gru_validation_metrics.csv
results/07_delta_gru_validation_predictions.csv
results/07_delta_gru_summary.txt
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS = Path("results")
DATA_FILE = RESULTS / "03_01_preprocessed_samples.csv"

FEATURE_SETS = {
    "F2": ["C_t_z", "D_sin", "D_cos"],
    "F3": ["C_t_z", "D_sin", "D_cos", "P_t_z"],
    "F4": [
        "C_t_z",
        "D_sin",
        "D_cos",
        "P_t_z",
        "is_public_holiday_t_plus_1",
    ],
}

WINDOWS = [1, 2, 5, 7, 14, 21, 28]
HIDDEN_SIZES = [1, 2, 4, 8, 16, 32]

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_LAYERS = 1
BATCH_SIZE = 64

MAX_EPOCHS = 2000
PATIENCE = 30
MIN_DELTA = 1e-4

N_CV_SPLITS = 5
BASE_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# =============================================================================
# DATA
# =============================================================================

def find_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "None of the expected columns were found:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\n\nAvailable columns:\n"
        + ", ".join(df.columns)
    )


def load_preprocessed():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Run Week-3 preprocessing first."
        )

    df = pd.read_csv(DATA_FILE)

    input_date_col = find_column(df, ["input_date", "date_t"])
    target_date_col = find_column(df, ["target_date", "date_t_plus_1"])
    split_col = find_column(df, ["split"])

    target_col = find_column(
        df,
        [
            "target_property_damage_claim_count",
            "C_t_plus_1",
            "target",
        ],
    )

    current_claim_col = find_column(
        df,
        [
            "property_damage_claim_count_t",
            "C_t",
            "current_property_damage_claim_count",
        ],
    )

    df[input_date_col] = pd.to_datetime(df[input_date_col])
    df[target_date_col] = pd.to_datetime(df[target_date_col])

    df = df.sort_values(target_date_col).reset_index(drop=True)

    # Exclude 2026/test completely from this diagnostic.
    work = df[df[split_col].isin(["train", "validation"])].copy()
    work = work.sort_values(target_date_col).reset_index(drop=True)

    assert (work[target_date_col] == work[input_date_col] + pd.Timedelta(days=1)).all()

    n_train = int((work[split_col] == "train").sum())
    n_val = int((work[split_col] == "validation").sum())

    assert n_train == 1095, f"Expected 1095 train rows, got {n_train}"
    assert n_val == 365, f"Expected 365 validation rows, got {n_val}"

    # New target.
    work["delta_claim"] = (
        work[target_col].astype(float)
        - work[current_claim_col].astype(float)
    )

    cols = {
        "input_date": input_date_col,
        "target_date": target_date_col,
        "split": split_col,
        "target_count": target_col,
        "current_count": current_claim_col,
    }

    return work, cols


def construct_windows(work, cols, feature_names, W):
    """
    Construct endpoint-aligned temporal windows.

    Training:
      only windows ending in train are used; because the dataset begins on
      2022-01-01, the first W-1 training endpoints are unavailable.

    Validation:
      all 365 2025 targets are retained. Their history may legitimately reach
      backward into late-2024 training observations because those observations
      are known at forecast time. This matches the earlier Week-4 construction.
    """

    X_all = work[feature_names].to_numpy(dtype=np.float32)
    y_delta = work["delta_claim"].to_numpy(dtype=np.float32)
    y_count = work[cols["target_count"]].to_numpy(dtype=np.float32)
    c_current = work[cols["current_count"]].to_numpy(dtype=np.float32)
    split = work[cols["split"]].astype(str).to_numpy()
    target_dates = work[cols["target_date"]].to_numpy()

    train_X, train_y = [], []
    val_X, val_y = [], []
    val_count, val_current, val_dates = [], [], []

    for end in range(len(work)):
        start = end - W + 1
        if start < 0:
            continue

        window = X_all[start : end + 1]
        endpoint_split = split[end]

        if endpoint_split == "train":
            # Prevent any non-train rows inside a training window.
            if np.all(split[start : end + 1] == "train"):
                train_X.append(window)
                train_y.append(y_delta[end])

        elif endpoint_split == "validation":
            val_X.append(window)
            val_y.append(y_delta[end])
            val_count.append(y_count[end])
            val_current.append(c_current[end])
            val_dates.append(target_dates[end])

    X_train = np.asarray(train_X, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.float32)
    X_val = np.asarray(val_X, dtype=np.float32)
    y_val = np.asarray(val_y, dtype=np.float32)

    meta_val = pd.DataFrame({
        "target_date": pd.to_datetime(np.asarray(val_dates)),
        "C_t": np.asarray(val_current, dtype=float),
        "C_t_plus_1": np.asarray(val_count, dtype=float),
        "delta_actual": y_val.astype(float),
    })

    assert X_train.ndim == 3
    assert X_val.ndim == 3
    assert len(X_val) == 365
    assert len(y_val) == 365

    expected_train = 1095 - W + 1
    assert len(X_train) == expected_train, (
        f"W={W}: expected {expected_train} train samples, got {len(X_train)}"
    )

    return X_train, y_train, X_val, y_val, meta_val


# =============================================================================
# MODEL
# =============================================================================

class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.gru(x)
        h_last = hidden[-1]
        return self.readout(h_last).squeeze(-1)


# =============================================================================
# METRICS
# =============================================================================

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    sd = float(np.std(y_true, ddof=0))
    nrmse = rmse / sd if sd > 0 else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "nrmse": nrmse,
    }


# =============================================================================
# TRAINING
# =============================================================================

def make_loader(X, y_scaled):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_scaled, dtype=torch.float32),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def train_early_stopping(
    X_train,
    y_train,
    X_valid,
    y_valid,
    hidden_size,
    seed,
):
    """
    Target scaler is fitted ONLY on the fold-training target.
    Early stopping uses the chronological fold-validation target.
    """

    set_seed(seed)

    y_mu = float(np.mean(y_train))
    y_sd = float(np.std(y_train, ddof=0))
    if y_sd <= 1e-12:
        y_sd = 1.0

    ytr_z = (y_train - y_mu) / y_sd

    model = GRURegressor(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()
    loader = make_loader(X_train, ytr_z)

    X_valid_t = torch.tensor(
        X_valid,
        dtype=torch.float32,
        device=DEVICE,
    )

    best_rmse = np.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_z = model(X_valid_t).cpu().numpy()

        pred = pred_z * y_sd + y_mu
        score = metrics(y_valid, pred)["rmse"]

        if score < best_rmse - MIN_DELTA:
            best_rmse = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Early stopping failed to record a model state.")

    return best_rmse, best_epoch


def chronological_cv(X, y, hidden_size):
    splitter = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    fold_rmse = []
    fold_epochs = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X), start=1):
        rmse, best_epoch = train_early_stopping(
            X[tr_idx],
            y[tr_idx],
            X[va_idx],
            y[va_idx],
            hidden_size=hidden_size,
            seed=BASE_SEED + fold,
        )

        fold_rmse.append(rmse)
        fold_epochs.append(best_epoch)

    return np.asarray(fold_rmse), np.asarray(fold_epochs)


def train_fixed_epochs(X_train, y_train, hidden_size, epochs, seed):
    """
    Train on the complete 2022-2024 training set after architecture selection.
    No 2025 values are used for training or epoch selection.
    """

    set_seed(seed)

    y_mu = float(np.mean(y_train))
    y_sd = float(np.std(y_train, ddof=0))
    if y_sd <= 1e-12:
        y_sd = 1.0

    ytr_z = (y_train - y_mu) / y_sd

    model = GRURegressor(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()
    loader = make_loader(X_train, ytr_z)

    for _ in range(int(epochs)):
        model.train()

        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

    return model, y_mu, y_sd


def predict(model, X, y_mu, y_sd):
    model.eval()

    Xt = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        pred_z = model(Xt).cpu().numpy()

    return pred_z * y_sd + y_mu


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 118)
    print("DELTA CLAIM TEMPORAL-MEMORY TEST - GRU")
    print("=" * 118)
    print()
    print("Target:")
    print("  Delta C_{t+1} = C_{t+1} - C_t")
    print()
    print(f"Feature sets:  {list(FEATURE_SETS)}")
    print(f"Windows:       {WINDOWS}")
    print(f"Hidden sizes:  {HIDDEN_SIZES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"L2:            {WEIGHT_DECAY}")
    print(f"Layers:        {NUM_LAYERS}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Max epochs:    {MAX_EPOCHS}")
    print(f"Patience:      {PATIENCE}")
    print(f"CV folds:      {N_CV_SPLITS}")
    print(f"Device:        {DEVICE}")
    print()
    print("2025 is diagnostic only.")
    print("2026 test set is excluded / untouched.")
    print()

    work, cols = load_preprocessed()

    train_delta = work.loc[
        work[cols["split"]] == "train",
        "delta_claim",
    ].to_numpy(dtype=float)

    val_delta = work.loc[
        work[cols["split"]] == "validation",
        "delta_claim",
    ].to_numpy(dtype=float)

    print("-" * 118)
    print("DELTA TARGET DESCRIPTIVE STATISTICS")
    print("-" * 118)
    print(
        f"Train delta: mean={np.mean(train_delta):.6f}, "
        f"std={np.std(train_delta, ddof=0):.6f}, "
        f"min={np.min(train_delta):.1f}, max={np.max(train_delta):.1f}"
    )
    print(
        f"2025 delta:  mean={np.mean(val_delta):.6f}, "
        f"std={np.std(val_delta, ddof=0):.6f}, "
        f"min={np.min(val_delta):.1f}, max={np.max(val_delta):.1f}"
    )

    # Delta=0 corresponds exactly to persistence C_hat_{t+1}=C_t.
    zero_delta_rmse = float(np.sqrt(np.mean(val_delta**2)))
    print(
        f"Zero-delta / persistence baseline 2025 RMSE = "
        f"{zero_delta_rmse:.6f}"
    )
    print()

    rows = []

    total_configs = (
        len(FEATURE_SETS)
        * len(WINDOWS)
        * len(HIDDEN_SIZES)
    )
    print(f"Expected configurations: {total_configs}")
    print(f"Expected CV fits:        {total_configs * N_CV_SPLITS}")
    print()

    for feature_set, feature_names in FEATURE_SETS.items():
        for W in WINDOWS:
            X_train, y_train, X_val, y_val, _ = construct_windows(
                work,
                cols,
                feature_names,
                W,
            )

            print("-" * 118)
            print(
                f"{feature_set}, W={W} | "
                f"train={X_train.shape} | val={X_val.shape}"
            )

            for H in HIDDEN_SIZES:
                fold_rmse, fold_epochs = chronological_cv(
                    X_train,
                    y_train,
                    hidden_size=H,
                )

                mean_rmse = float(np.mean(fold_rmse))
                std_rmse = float(np.std(fold_rmse, ddof=1))
                median_epoch = int(np.median(fold_epochs))

                rows.append({
                    "feature_set": feature_set,
                    "window": W,
                    "hidden_size": H,
                    "input_dimension": X_train.shape[2],
                    "cv_rmse_mean": mean_rmse,
                    "cv_rmse_std": std_rmse,
                    "median_best_epoch": median_epoch,
                    "fold_rmse": ",".join(
                        f"{x:.8f}" for x in fold_rmse
                    ),
                    "fold_best_epoch": ",".join(
                        str(int(x)) for x in fold_epochs
                    ),
                })

                print(
                    f"  H={H:2d} | "
                    f"CV delta RMSE={mean_rmse:.6f} "
                    f"+/- {std_rmse:.6f} | "
                    f"median epoch={median_epoch}"
                )

    cv = pd.DataFrame(rows).sort_values(
        "cv_rmse_mean"
    ).reset_index(drop=True)

    cv.to_csv(
        RESULTS / "07_delta_gru_cv_results.csv",
        index=False,
    )

    # Best hidden size at each F/W.
    best_fw = (
        cv.sort_values("cv_rmse_mean")
        .groupby(["feature_set", "window"], as_index=False)
        .first()
        .sort_values(["feature_set", "window"])
        .reset_index(drop=True)
    )

    # W=1 reference within each feature set.
    w1 = (
        best_fw[best_fw["window"] == 1][
            ["feature_set", "cv_rmse_mean"]
        ]
        .rename(columns={"cv_rmse_mean": "W1_cv_rmse"})
    )

    memory_gain = best_fw.merge(
        w1,
        on="feature_set",
        how="left",
    )

    memory_gain["memory_gain_vs_W1_pct"] = (
        100.0
        * (
            memory_gain["W1_cv_rmse"]
            - memory_gain["cv_rmse_mean"]
        )
        / memory_gain["W1_cv_rmse"]
    )

    best_fw.to_csv(
        RESULTS / "07_delta_gru_best_per_feature_window.csv",
        index=False,
    )

    memory_gain.to_csv(
        RESULTS / "07_delta_gru_memory_gain_vs_W1.csv",
        index=False,
    )

    print()
    print("=" * 118)
    print("BEST DELTA-GRU CONFIGURATION PER FEATURE SET / WINDOW")
    print("=" * 118)
    print(
        best_fw[
            [
                "feature_set",
                "window",
                "hidden_size",
                "cv_rmse_mean",
                "cv_rmse_std",
                "median_best_epoch",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 118)
    print("MEMORY GAIN RELATIVE TO W=1")
    print("=" * 118)
    print(
        memory_gain[
            [
                "feature_set",
                "window",
                "hidden_size",
                "cv_rmse_mean",
                "memory_gain_vs_W1_pct",
            ]
        ].to_string(index=False)
    )

    # Best configuration per feature set, selected ONLY by train CV.
    best_feature = (
        cv.sort_values("cv_rmse_mean")
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("cv_rmse_mean")
        .reset_index(drop=True)
    )

    print()
    print("=" * 118)
    print("BEST DELTA-GRU CONFIGURATION PER FEATURE SET")
    print("=" * 118)
    print(
        best_feature[
            [
                "feature_set",
                "window",
                "hidden_size",
                "cv_rmse_mean",
                "cv_rmse_std",
                "median_best_epoch",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # 2025 diagnostics for CV-selected winner of each feature set.
    # =========================================================================
    val_rows = []
    pred_parts = []

    for _, chosen in best_feature.iterrows():
        F = str(chosen["feature_set"])
        W = int(chosen["window"])
        H = int(chosen["hidden_size"])
        epochs = int(chosen["median_best_epoch"])

        X_train, y_train, X_val, y_val, meta_val = construct_windows(
            work,
            cols,
            FEATURE_SETS[F],
            W,
        )

        model, y_mu, y_sd = train_fixed_epochs(
            X_train,
            y_train,
            hidden_size=H,
            epochs=epochs,
            seed=BASE_SEED,
        )

        pred_delta = predict(
            model,
            X_val,
            y_mu,
            y_sd,
        )

        delta_metrics = metrics(y_val, pred_delta)

        # Convert delta forecast back to count.
        pred_count = (
            meta_val["C_t"].to_numpy(dtype=float)
            + pred_delta
        )

        actual_count = meta_val[
            "C_t_plus_1"
        ].to_numpy(dtype=float)

        count_metrics = metrics(
            actual_count,
            pred_count,
        )

        equality_error = abs(
            delta_metrics["rmse"]
            - count_metrics["rmse"]
        )

        val_rows.append({
            "feature_set": F,
            "window": W,
            "hidden_size": H,
            "training_epochs": epochs,
            "cv_delta_rmse": float(chosen["cv_rmse_mean"]),
            "validation_delta_mae": delta_metrics["mae"],
            "validation_delta_rmse": delta_metrics["rmse"],
            "validation_delta_nrmse": delta_metrics["nrmse"],
            "validation_delta_bias": delta_metrics["bias"],
            "validation_reconstructed_count_rmse": count_metrics["rmse"],
            "delta_count_rmse_equality_error": equality_error,
            "improvement_vs_zero_delta_persistence_pct":
                100.0
                * (
                    zero_delta_rmse
                    - delta_metrics["rmse"]
                )
                / zero_delta_rmse,
        })

        p = meta_val.copy()
        p["feature_set"] = F
        p["window"] = W
        p["hidden_size"] = H
        p["delta_prediction"] = pred_delta
        p["count_prediction_from_delta"] = pred_count
        p["delta_residual"] = pred_delta - p["delta_actual"]
        p["count_residual"] = (
            pred_count - actual_count
        )
        pred_parts.append(p)

    val_df = pd.DataFrame(val_rows)
    pred_df = pd.concat(pred_parts, ignore_index=True)

    val_df.to_csv(
        RESULTS / "07_delta_gru_validation_metrics.csv",
        index=False,
    )

    pred_df.to_csv(
        RESULTS / "07_delta_gru_validation_predictions.csv",
        index=False,
    )

    print()
    print("=" * 118)
    print("OFFICIAL 2025 DIAGNOSTIC FOR CV-SELECTED CONFIGURATIONS")
    print("=" * 118)
    print(val_df.to_string(index=False))

    # =========================================================================
    # Optional comparison with original claim-level GRU screen.
    # =========================================================================
    original_candidates = [
        RESULTS / "04_04a_gru_screening_cv_results.csv",
        RESULTS / "04_04a_gru_cv_results.csv",
        RESULTS / "04_04a_gru_screening_results.csv",
    ]

    original_path = next(
        (p for p in original_candidates if p.exists()),
        None,
    )

    original_comparison_text = ""

    if original_path is not None:
        try:
            orig = pd.read_csv(original_path)

            fs_col = next(
                c for c in ["feature_set"] if c in orig.columns
            )
            w_col = next(
                c for c in ["window", "window_length"] if c in orig.columns
            )
            rmse_col = next(
                c for c in ["cv_rmse_mean", "cv_rmse"] if c in orig.columns
            )

            orig_best_fw = (
                orig.sort_values(rmse_col)
                .groupby([fs_col, w_col], as_index=False)
                .first()
            )

            orig_best_feature = (
                orig_best_fw.sort_values(rmse_col)
                .groupby(fs_col, as_index=False)
                .first()
            )

            delta_best = best_feature[
                ["feature_set", "window", "cv_rmse_mean"]
            ].rename(
                columns={
                    "window": "delta_best_W",
                    "cv_rmse_mean": "delta_best_cv_rmse",
                }
            )

            orig_best_feature = orig_best_feature[
                [fs_col, w_col, rmse_col]
            ].rename(
                columns={
                    fs_col: "feature_set",
                    w_col: "level_best_W",
                    rmse_col: "level_best_cv_rmse",
                }
            )

            comparison = orig_best_feature.merge(
                delta_best,
                on="feature_set",
                how="outer",
            )

            comparison.to_csv(
                RESULTS / "07_delta_gru_level_vs_delta_optimal_window.csv",
                index=False,
            )

            print()
            print("=" * 118)
            print("ORIGINAL LEVEL TARGET VS DELTA TARGET - OPTIMAL WINDOW")
            print("=" * 118)
            print(f"Original GRU file: {original_path}")
            print(comparison.to_string(index=False))

            original_comparison_text = (
                "\n\nLEVEL VS DELTA OPTIMAL WINDOW:\n"
                + comparison.to_string(index=False)
            )

        except Exception as exc:
            print()
            print(
                "Original-level comparison skipped because the older "
                f"results file could not be parsed: {exc}"
            )

    # =========================================================================
    # Summary file
    # =========================================================================
    with open(
        RESULTS / "07_delta_gru_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write("DELTA-CLAIM GRU TEMPORAL-MEMORY TEST\n")
        fp.write("=" * 100 + "\n\n")
        fp.write(
            "Target: Delta C_{t+1} = C_{t+1} - C_t\n"
        )
        fp.write(
            "Positive memory_gain_vs_W1_pct means a longer window "
            "improves chronological CV RMSE.\n\n"
        )
        fp.write("BEST PER FEATURE/WINDOW:\n")
        fp.write(
            best_fw[
                [
                    "feature_set",
                    "window",
                    "hidden_size",
                    "cv_rmse_mean",
                    "cv_rmse_std",
                ]
            ].to_string(index=False)
        )
        fp.write("\n\nMEMORY GAIN VS W1:\n")
        fp.write(
            memory_gain[
                [
                    "feature_set",
                    "window",
                    "hidden_size",
                    "cv_rmse_mean",
                    "memory_gain_vs_W1_pct",
                ]
            ].to_string(index=False)
        )
        fp.write("\n\n2025 DIAGNOSTIC:\n")
        fp.write(val_df.to_string(index=False))
        fp.write(original_comparison_text)
        fp.write("\n")

    print()
    print("=" * 118)
    print("DECISION RULE")
    print("=" * 118)
    print(
        "Delta is more memory-demanding only if the CV-optimal window "
        "moves systematically above W=1 and longer windows provide a "
        "non-trivial positive memory_gain_vs_W1_pct."
    )
    print(
        "A single isolated longer-window win is weaker evidence than a "
        "consistent pattern across F2/F3/F4."
    )

    print()
    print("Saved:")
    print("  results/07_delta_gru_cv_results.csv")
    print("  results/07_delta_gru_best_per_feature_window.csv")
    print("  results/07_delta_gru_memory_gain_vs_W1.csv")
    print("  results/07_delta_gru_validation_metrics.csv")
    print("  results/07_delta_gru_validation_predictions.csv")
    print("  results/07_delta_gru_summary.txt")
    if original_path is not None:
        print("  results/07_delta_gru_level_vs_delta_optimal_window.csv")


if __name__ == "__main__":
    main()
