from pathlib import Path
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("results")

SCREENING_FILE = (
    RESULTS_DIR /
    "04_04a_gru_screening_cv_results.csv"
)

TOP_N = 4

SEEDS = list(range(42, 52))

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_LAYERS = 1
BATCH_SIZE = 64

DEVICE = torch.device("cpu")

# Existing validation references
RIDGE_F4_RMSE = 4.874844
ESN_BEST_RMSE = 4.856483
RNN_TUNED_RMSE = 5.027691
SEASONAL_NAIVE_RMSE = 6.4057


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:

        torch.use_deterministic_algorithms(
            True
        )

    except Exception:

        pass


# =============================================================================
# TARGET STANDARDIZATION
# =============================================================================

def fit_target_scaler(y_train):

    y_mean = float(
        np.mean(y_train)
    )

    y_std = float(
        np.std(
            y_train,
            ddof=0
        )
    )

    if (
        not np.isfinite(y_std)
        or y_std <= 0
    ):

        raise ValueError(
            "Invalid target standard deviation."
        )

    return y_mean, y_std


def scale_target(
    y,
    y_mean,
    y_std
):

    return (
        (y - y_mean) / y_std
    ).astype(np.float32)


def inverse_scale_target(
    y_scaled,
    y_mean,
    y_std
):

    return (
        y_scaled * y_std
        + y_mean
    )


# =============================================================================
# LOAD TEMPORAL DATA
# =============================================================================

def load_temporal_data(
    feature_set,
    window
):

    path = (
        RESULTS_DIR /
        f"04_01_windows_{feature_set}_W{window:02d}.npz"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    with np.load(
        path,
        allow_pickle=True
    ) as data:

        X_train = np.asarray(
            data["X_train"],
            dtype=np.float32
        )

        y_train = np.asarray(
            data["y_train"],
            dtype=np.float32
        ).reshape(-1)

        X_val = np.asarray(
            data["X_val"],
            dtype=np.float32
        )

        y_val = np.asarray(
            data["y_val"],
            dtype=np.float32
        ).reshape(-1)

    return (
        X_train,
        y_train,
        X_val,
        y_val
    )


# =============================================================================
# GRU
# =============================================================================

class GRURegressor(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=NUM_LAYERS,
            batch_first=True
        )

        self.output = nn.Linear(
            hidden_size,
            1
        )

        self._initialize_weights()

    def _initialize_weights(self):

        weight_ih = (
            self.gru.weight_ih_l0
        )

        weight_hh = (
            self.gru.weight_hh_l0
        )

        # reset, update and candidate gates
        for gate_weight in weight_ih.chunk(
            3,
            dim=0
        ):

            nn.init.xavier_uniform_(
                gate_weight
            )

        for gate_weight in weight_hh.chunk(
            3,
            dim=0
        ):

            nn.init.orthogonal_(
                gate_weight
            )

        nn.init.zeros_(
            self.gru.bias_ih_l0
        )

        nn.init.zeros_(
            self.gru.bias_hh_l0
        )

        nn.init.xavier_uniform_(
            self.output.weight
        )

        nn.init.zeros_(
            self.output.bias
        )

    def forward(self, x):

        output, hidden = self.gru(
            x
        )

        h_last = hidden[-1]

        prediction_scaled = (
            self.output(
                h_last
            )
        )

        return (
            prediction_scaled
            .squeeze(-1)
        )


# =============================================================================
# PARAMETER COUNT
# =============================================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =============================================================================
# TRAIN FIXED NUMBER OF EPOCHS
# =============================================================================

def train_fixed_epochs(
    X_train,
    y_train,
    hidden_size,
    epochs,
    seed
):

    set_seed(seed)

    # -------------------------------------------------------------------------
    # Training-only target scaler
    # -------------------------------------------------------------------------

    y_mean, y_std = (
        fit_target_scaler(
            y_train
        )
    )

    y_scaled = scale_target(
        y_train,
        y_mean,
        y_std
    )

    model = GRURegressor(
        input_size=X_train.shape[2],
        hidden_size=hidden_size
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(

        torch.tensor(
            X_train,
            dtype=torch.float32
        ),

        torch.tensor(
            y_scaled,
            dtype=torch.float32
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            pred_scaled = model(
                X_batch
            )

            loss = criterion(
                pred_scaled,
                y_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

    return (
        model,
        y_mean,
        y_std
    )


# =============================================================================
# PREDICTION
# =============================================================================

def predict(
    model,
    X,
    y_mean,
    y_std
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        pred_scaled = (
            model(
                X_tensor
            )
            .cpu()
            .numpy()
        )

    prediction = (
        inverse_scale_target(
            pred_scaled,
            y_mean,
            y_std
        )
    )

    return prediction


# =============================================================================
# METRICS
# =============================================================================

def metrics(
    y_true,
    y_pred
):

    residual = (
        y_pred - y_true
    )

    mae = float(
        np.mean(
            np.abs(residual)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    bias = float(
        np.mean(
            residual
        )
    )

    nrmse = float(
        rmse /
        np.std(
            y_true,
            ddof=0
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "bias": bias,
        "prediction_mean":
            float(
                np.mean(
                    y_pred
                )
            )
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("STEP 4.4B - TOP-4 GRU SEED ROBUSTNESS")
    print("TARGET STANDARDIZATION: ENABLED")
    print("=" * 100)

    if not SCREENING_FILE.exists():

        raise FileNotFoundError(
            f"Missing screening file: "
            f"{SCREENING_FILE}"
        )

    screening_df = pd.read_csv(
        SCREENING_FILE
    )

    # -------------------------------------------------------------------------
    # Select the four architectures using TRAINING CV ONLY
    # -------------------------------------------------------------------------

    top4 = (
        screening_df
        .sort_values(
            [
                "cv_rmse_mean",
                "parameter_count"
            ]
        )
        .head(TOP_N)
        .reset_index(drop=True)
    )

    print()
    print("Top four configurations selected from 2022-2024 CV only:")
    print()

    print(
        top4[
            [
                "feature_set",
                "window",
                "hidden_size",
                "parameter_count",
                "cv_rmse_mean",
                "cv_rmse_std",
                "median_best_epoch"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "2025 validation is used ONLY for seed-robustness diagnostics."
    )

    print(
        "No architecture or hyperparameter is selected from 2025 here."
    )

    print(
        "2026 test set is NOT used."
    )

    all_rows = []

    # =========================================================================
    # TOP-4 × 10 SEEDS
    # =========================================================================

    for rank, config in top4.iterrows():

        feature_set = str(
            config["feature_set"]
        )

        window = int(
            config["window"]
        )

        hidden_size = int(
            config["hidden_size"]
        )

        epochs = int(
            config[
                "median_best_epoch"
            ]
        )

        (
            X_train,
            y_train,
            X_val,
            y_val
        ) = load_temporal_data(
            feature_set,
            window
        )

        print()
        print("=" * 100)

        print(
            f"CONFIGURATION {rank + 1}"
        )

        print("=" * 100)

        print(
            f"Feature set : {feature_set}"
        )

        print(
            f"Window      : {window}"
        )

        print(
            f"Hidden size : {hidden_size}"
        )

        print(
            f"CV RMSE     : "
            f"{config['cv_rmse_mean']:.6f} "
            f"+/- "
            f"{config['cv_rmse_std']:.6f}"
        )

        print(
            f"Fixed epochs: {epochs}"
        )

        for seed in SEEDS:

            (
                model,
                y_mean,
                y_std
            ) = train_fixed_epochs(
                X_train,
                y_train,
                hidden_size=hidden_size,
                epochs=epochs,
                seed=seed
            )

            prediction = predict(
                model,
                X_val,
                y_mean,
                y_std
            )

            result = metrics(
                y_val,
                prediction
            )

            row = {

                "config_rank":
                    rank + 1,

                "feature_set":
                    feature_set,

                "window":
                    window,

                "hidden_size":
                    hidden_size,

                "parameter_count":
                    int(
                        config[
                            "parameter_count"
                        ]
                    ),

                "cv_rmse_mean":
                    float(
                        config[
                            "cv_rmse_mean"
                        ]
                    ),

                "fixed_epochs":
                    epochs,

                "seed":
                    seed,

                "validation_rmse":
                    result["rmse"],

                "validation_mae":
                    result["mae"],

                "validation_nrmse":
                    result["nrmse"],

                "validation_bias":
                    result["bias"],

                "prediction_mean":
                    result[
                        "prediction_mean"
                    ],

                "target_scaler_mean":
                    y_mean,

                "target_scaler_std":
                    y_std,
            }

            all_rows.append(
                row
            )

            print(
                f"Seed {seed}: "
                f"RMSE="
                f"{result['rmse']:.6f} | "
                f"MAE="
                f"{result['mae']:.6f} | "
                f"Bias="
                f"{result['bias']:+.6f}"
            )

    # =========================================================================
    # SAVE INDIVIDUAL SEED RESULTS
    # =========================================================================

    results_df = pd.DataFrame(
        all_rows
    )

    results_path = (
        RESULTS_DIR /
        "04_04b_gru_top4_seed_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # =========================================================================
    # SUMMARY BY CONFIGURATION
    # =========================================================================

    summary_rows = []

    for (
        config_rank,
        feature_set,
        window,
        hidden_size
    ), group in results_df.groupby(
        [
            "config_rank",
            "feature_set",
            "window",
            "hidden_size"
        ]
    ):

        summary_rows.append({

            "config_rank":
                config_rank,

            "feature_set":
                feature_set,

            "window":
                window,

            "hidden_size":
                hidden_size,

            "parameter_count":
                int(
                    group[
                        "parameter_count"
                    ].iloc[0]
                ),

            "cv_rmse_mean":
                float(
                    group[
                        "cv_rmse_mean"
                    ].iloc[0]
                ),

            "fixed_epochs":
                int(
                    group[
                        "fixed_epochs"
                    ].iloc[0]
                ),

            "seed_rmse_mean":
                group[
                    "validation_rmse"
                ].mean(),

            "seed_rmse_std":
                group[
                    "validation_rmse"
                ].std(
                    ddof=1
                ),

            "seed_rmse_min":
                group[
                    "validation_rmse"
                ].min(),

            "seed_rmse_max":
                group[
                    "validation_rmse"
                ].max(),

            "seed_mae_mean":
                group[
                    "validation_mae"
                ].mean(),

            "seed_bias_mean":
                group[
                    "validation_bias"
                ].mean(),

            "seed_bias_std":
                group[
                    "validation_bias"
                ].std(
                    ddof=1
                ),

            "prediction_mean":
                group[
                    "prediction_mean"
                ].mean()
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # Diagnostic ordering only.
    summary_df = (
        summary_df
        .sort_values(
            "seed_rmse_mean"
        )
        .reset_index(
            drop=True
        )
    )

    summary_path = (
        RESULTS_DIR /
        "04_04b_gru_top4_seed_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # =========================================================================
    # PRINT FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 100)
    print("TOP-4 GRU SEED ROBUSTNESS SUMMARY")
    print("=" * 100)

    print(
        summary_df[
            [
                "config_rank",
                "feature_set",
                "window",
                "hidden_size",
                "parameter_count",
                "seed_rmse_mean",
                "seed_rmse_std",
                "seed_rmse_min",
                "seed_rmse_max",
                "seed_bias_mean"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANT: this table is a robustness diagnostic."
    )

    print(
        "Do not use the 2025 seed-average ranking "
        "to retune or reselect the architecture."
    )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print(results_path)
    print(summary_path)

    print()
    print(
        "2026 TEST SET WAS NOT USED."
    )


if __name__ == "__main__":
    main()