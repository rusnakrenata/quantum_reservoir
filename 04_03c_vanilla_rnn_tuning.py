from pathlib import Path
import copy
import random
import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

STEP43A_CV_FILE = (
    RESULTS_DIR /
    "04_03_vanilla_rnn_cv_results.csv"
)

STEP43A_PREDICTIONS_FILE = (
    RESULTS_DIR /
    "04_03_vanilla_rnn_validation_predictions.csv"
)

# Keep the same hidden-size tuning range used in the original 4.3C.
# Step 4.3A currently selects H=32, and H=64 remains the larger
# RNN-capacity alternative.
HIDDEN_SIZES = [32, 64]

LEARNING_RATES = [
    3e-4,
    1e-3,
    3e-3,
]

WEIGHT_DECAYS = [
    1e-5,
    1e-4,
    1e-3,
]

NUM_LAYERS_LIST = [
    1,
    2,
]

N_SPLITS = 5

BATCH_SIZE = 64

MAX_EPOCHS = 200
PATIENCE = 20
MIN_DELTA = 1e-4

GRID_SEED = 42

DEVICE = torch.device("cpu")


# External reference models
RIDGE_F4_VAL_RMSE = 4.8748
ESN_VAL_RMSE = 4.872332
SEASONAL_NAIVE_VAL_RMSE = 6.4057


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Target standardization
# ============================================================

def fit_target_scaler(y_train):
    """
    Fit target scaling using TRAINING TARGETS ONLY.

        y_scaled = (y - mean) / std

    During chronological CV this function is called separately
    for every fold using only that fold's training targets.
    """

    y_mean = float(
        np.mean(y_train)
    )

    y_std = float(
        np.std(
            y_train,
            ddof=0
        )
    )

    if not np.isfinite(y_mean):
        raise ValueError(
            "Target mean is not finite."
        )

    if (
        not np.isfinite(y_std)
        or y_std <= 0.0
    ):
        raise ValueError(
            "Target standard deviation "
            "must be finite and > 0."
        )

    return y_mean, y_std


def scale_target(
    y,
    y_mean,
    y_std,
):

    return (
        (y - y_mean) / y_std
    ).astype(np.float32)


def inverse_scale_target(
    y_scaled,
    y_mean,
    y_std,
):

    return (
        y_scaled * y_std
        + y_mean
    )


# ============================================================
# Read the corrected Step 4.3A winner
# ============================================================

if not STEP43A_CV_FILE.exists():

    raise FileNotFoundError(
        f"Missing corrected Step 4.3A CV file:\n"
        f"{STEP43A_CV_FILE}"
    )


step43a_cv = pd.read_csv(
    STEP43A_CV_FILE
)

step43a_cv = (
    step43a_cv
    .sort_values("cv_rmse_mean")
    .reset_index(drop=True)
)

step43a_best = step43a_cv.iloc[0]

FEATURE_SET = str(
    step43a_best["feature_set"]
)

WINDOW = int(
    step43a_best["window"]
)

SCREENING_HIDDEN_SIZE = int(
    step43a_best["hidden_size"]
)

ORIGINAL_RNN_CV_RMSE = float(
    step43a_best["cv_rmse_mean"]
)

DATA_FILE = (
    RESULTS_DIR /
    f"04_01_windows_{FEATURE_SET}_W{WINDOW:02d}.npz"
)


# Corrected Step 4.3A official-validation reference
if STEP43A_PREDICTIONS_FILE.exists():

    step43a_pred_df = pd.read_csv(
        STEP43A_PREDICTIONS_FILE
    )

    ORIGINAL_RNN_VAL_RMSE = float(
        np.sqrt(
            mean_squared_error(
                step43a_pred_df["y_true"],
                step43a_pred_df["y_pred"],
            )
        )
    )

else:

    ORIGINAL_RNN_VAL_RMSE = np.nan


# ============================================================
# Load selected temporal tensor
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Missing selected temporal tensor:\n"
        f"{DATA_FILE}"
    )


with np.load(
    DATA_FILE,
    allow_pickle=True,
) as data:

    X_train = np.asarray(
        data["X_train"],
        dtype=np.float32,
    )

    y_train = np.asarray(
        data["y_train"],
        dtype=np.float32,
    ).reshape(-1)

    X_val = np.asarray(
        data["X_val"],
        dtype=np.float32,
    )

    y_val = np.asarray(
        data["y_val"],
        dtype=np.float32,
    ).reshape(-1)


print("=" * 90)
print("STEP 4.3C - FINAL VANILLA RNN HYPERPARAMETER TUNING")
print("TARGET STANDARDIZATION: ENABLED")
print("=" * 90)

print()
print("Architecture inherited from corrected Step 4.3A:")
print("Feature set       :", FEATURE_SET)
print("Window            :", WINDOW)
print("Screening H       :", SCREENING_HIDDEN_SIZE)
print(
    "Screening CV RMSE : "
    f"{ORIGINAL_RNN_CV_RMSE:.6f}"
)

print()
print("Training shape    :", X_train.shape)
print("Validation shape  :", X_val.shape)

full_target_mean, full_target_std = (
    fit_target_scaler(
        y_train
    )
)

print()
print(
    "Full training target mean:",
    f"{full_target_mean:.6f}"
)

print(
    "Full training target std :",
    f"{full_target_std:.6f}"
)

print()
print("Hyperparameter grid:")
print("Hidden sizes  :", HIDDEN_SIZES)
print("Learning rates:", LEARNING_RATES)
print("L2 values     :", WEIGHT_DECAYS)
print("RNN layers    :", NUM_LAYERS_LIST)

n_configs = (
    len(HIDDEN_SIZES)
    * len(LEARNING_RATES)
    * len(WEIGHT_DECAYS)
    * len(NUM_LAYERS_LIST)
)

print()
print("Total configurations:", n_configs)
print("Total CV fits:", n_configs * N_SPLITS)

print()
print(
    "Within every CV fold, the target scaler is fitted "
    "ONLY on that fold's training targets."
)

print(
    "All CV RMSE values are reported after inverse scaling "
    "back to original claim-count units."
)

print()
print("2025 validation is NOT used for hyperparameter selection.")
print("2026 test set is NOT used.")


# ============================================================
# Vanilla RNN
# ============================================================

class VanillaRNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
    ):

        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=0.0,
        )

        self.readout = nn.Linear(
            hidden_size,
            1,
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for layer in range(
            self.rnn.num_layers
        ):

            weight_ih = getattr(
                self.rnn,
                f"weight_ih_l{layer}",
            )

            weight_hh = getattr(
                self.rnn,
                f"weight_hh_l{layer}",
            )

            bias_ih = getattr(
                self.rnn,
                f"bias_ih_l{layer}",
            )

            bias_hh = getattr(
                self.rnn,
                f"bias_hh_l{layer}",
            )

            nn.init.xavier_uniform_(
                weight_ih
            )

            nn.init.orthogonal_(
                weight_hh
            )

            nn.init.zeros_(
                bias_ih
            )

            nn.init.zeros_(
                bias_hh
            )

        nn.init.xavier_uniform_(
            self.readout.weight
        )

        nn.init.zeros_(
            self.readout.bias
        )

    def forward(self, x):

        output, hidden = self.rnn(x)

        last_hidden = output[:, -1, :]

        prediction = self.readout(
            last_hidden
        )

        return prediction.squeeze(-1)


# ============================================================
# Parameter count
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Prediction helpers
# ============================================================

def predict_scaled(
    model,
    X,
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():

        pred_scaled = model(
            X_tensor
        )

    return (
        pred_scaled
        .cpu()
        .numpy()
    )


def predict_original_scale(
    model,
    X,
    y_mean,
    y_std,
):

    pred_scaled = predict_scaled(
        model,
        X,
    )

    return inverse_scale_target(
        pred_scaled,
        y_mean,
        y_std,
    )


# ============================================================
# One fold with early stopping
# ============================================================

def train_with_early_stopping(
    X_train_fold,
    y_train_fold,
    X_valid_fold,
    y_valid_fold,
    hidden_size,
    learning_rate,
    weight_decay,
    num_layers,
    seed,
):

    set_seed(seed)

    # --------------------------------------------------------
    # IMPORTANT:
    # Fit scaler on THIS FOLD'S TRAINING TARGETS ONLY.
    # --------------------------------------------------------

    y_mean, y_std = fit_target_scaler(
        y_train_fold
    )

    y_train_scaled = scale_target(
        y_train_fold,
        y_mean,
        y_std,
    )

    model = VanillaRNN(
        input_size=X_train_fold.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(
            X_train_fold,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train_scaled,
            dtype=torch.float32,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    best_rmse = np.inf
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
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
                max_norm=1.0,
            )

            optimizer.step()

        # ----------------------------------------------------
        # Validate in ORIGINAL claim-count units.
        # ----------------------------------------------------

        validation_pred = (
            predict_original_scale(
                model,
                X_valid_fold,
                y_mean,
                y_std,
            )
        )

        validation_rmse = np.sqrt(
            mean_squared_error(
                y_valid_fold,
                validation_pred,
            )
        )

        if (
            validation_rmse
            < best_rmse - MIN_DELTA
        ):

            best_rmse = (
                validation_rmse
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            break

    if best_state is None:

        raise RuntimeError(
            "No best model state was recorded."
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_rmse,
        best_epoch,
    )


# ============================================================
# Chronological CV
# ============================================================

def chronological_cv(
    hidden_size,
    learning_rate,
    weight_decay,
    num_layers,
):

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    fold_rmses = []
    fold_epochs = []

    for fold, (
        train_idx,
        valid_idx,
    ) in enumerate(
        splitter.split(X_train),
        start=1,
    ):

        X_fold_train = X_train[
            train_idx
        ]

        y_fold_train = y_train[
            train_idx
        ]

        X_fold_valid = X_train[
            valid_idx
        ]

        y_fold_valid = y_train[
            valid_idx
        ]

        _, rmse, best_epoch = (
            train_with_early_stopping(
                X_fold_train,
                y_fold_train,
                X_fold_valid,
                y_fold_valid,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_layers=num_layers,
                seed=GRID_SEED + fold,
            )
        )

        fold_rmses.append(
            rmse
        )

        fold_epochs.append(
            best_epoch
        )

    return (
        np.asarray(
            fold_rmses
        ),
        np.asarray(
            fold_epochs
        ),
    )


# ============================================================
# Grid search
# ============================================================

rows = []

config_number = 0

grid = itertools.product(
    HIDDEN_SIZES,
    LEARNING_RATES,
    WEIGHT_DECAYS,
    NUM_LAYERS_LIST,
)

for (
    hidden_size,
    learning_rate,
    weight_decay,
    num_layers,
) in grid:

    config_number += 1

    fold_rmse, fold_epochs = (
        chronological_cv(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_layers=num_layers,
        )
    )

    mean_rmse = fold_rmse.mean()

    std_rmse = fold_rmse.std(
        ddof=1
    )

    median_epoch = int(
        np.median(
            fold_epochs
        )
    )

    temp_model = VanillaRNN(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    n_parameters = (
        count_parameters(
            temp_model
        )
    )

    print(
        f"[{config_number:02d}/{n_configs}] "
        f"H={hidden_size:>2} | "
        f"LR={learning_rate:.0e} | "
        f"L2={weight_decay:.0e} | "
        f"Layers={num_layers} | "
        f"Params={n_parameters:>6} | "
        f"CV RMSE="
        f"{mean_rmse:.4f} "
        f"+/- {std_rmse:.4f} | "
        f"epochs={median_epoch}"
    )

    rows.append({
        "hidden_size":
            hidden_size,

        "learning_rate":
            learning_rate,

        "weight_decay":
            weight_decay,

        "num_layers":
            num_layers,

        "parameter_count":
            n_parameters,

        "cv_rmse_mean":
            mean_rmse,

        "cv_rmse_std":
            std_rmse,

        "median_best_epoch":
            median_epoch,

        "fold_rmse":
            ",".join(
                f"{x:.6f}"
                for x in fold_rmse
            ),

        "fold_best_epochs":
            ",".join(
                str(int(x))
                for x in fold_epochs
            ),
    })


# ============================================================
# Save complete CV table
# ============================================================

results_df = pd.DataFrame(
    rows
)

results_df = (
    results_df
    .sort_values(
        [
            "cv_rmse_mean",
            "parameter_count",
        ]
    )
    .reset_index(
        drop=True
    )
)

cv_path = (
    RESULTS_DIR /
    "04_03c_vanilla_rnn_tuning_cv.csv"
)

results_df.to_csv(
    cv_path,
    index=False,
)


# ============================================================
# Show top configurations
# ============================================================

print()
print("=" * 90)
print("TOP 10 CV CONFIGURATIONS")
print("=" * 90)

for rank, row in (
    results_df
    .head(10)
    .iterrows()
):

    print(
        f"{rank + 1:>2}. "
        f"H={int(row['hidden_size']):>2} | "
        f"LR={row['learning_rate']:.0e} | "
        f"L2={row['weight_decay']:.0e} | "
        f"Layers={int(row['num_layers'])} | "
        f"Params={int(row['parameter_count'])} | "
        f"RMSE="
        f"{row['cv_rmse_mean']:.6f} "
        f"+/- "
        f"{row['cv_rmse_std']:.6f}"
    )


# ============================================================
# Best configuration
# ============================================================

best = results_df.iloc[0]

BEST_HIDDEN_SIZE = int(
    best["hidden_size"]
)

BEST_LEARNING_RATE = float(
    best["learning_rate"]
)

BEST_WEIGHT_DECAY = float(
    best["weight_decay"]
)

BEST_NUM_LAYERS = int(
    best["num_layers"]
)

BEST_EPOCHS = int(
    best["median_best_epoch"]
)

BEST_PARAMETER_COUNT = int(
    best["parameter_count"]
)


print()
print("=" * 90)
print("BEST TUNED VANILLA RNN - SELECTED USING 2022-2024 CV ONLY")
print("=" * 90)

print(
    "Feature set    :",
    FEATURE_SET
)

print(
    "Window         :",
    WINDOW
)

print(
    "Hidden size    :",
    BEST_HIDDEN_SIZE
)

print(
    "Learning rate  :",
    BEST_LEARNING_RATE
)

print(
    "L2             :",
    BEST_WEIGHT_DECAY
)

print(
    "Layers         :",
    BEST_NUM_LAYERS
)

print(
    "Parameters     :",
    BEST_PARAMETER_COUNT
)

print(
    "CV RMSE        : "
    f"{best['cv_rmse_mean']:.6f} "
    f"+/- "
    f"{best['cv_rmse_std']:.6f}"
)

print(
    "Median epochs  :",
    BEST_EPOCHS
)

cv_improvement = (
    100.0
    *
    (
        ORIGINAL_RNN_CV_RMSE
        - best["cv_rmse_mean"]
    )
    /
    ORIGINAL_RNN_CV_RMSE
)

print(
    "CV improvement vs corrected Step 4.3A:",
    f"{cv_improvement:+.3f}%"
)


# ============================================================
# Train final selected configuration
# on ALL 2022-2024 training data
# ============================================================

def train_fixed_epochs(
    X,
    y,
    hidden_size,
    learning_rate,
    weight_decay,
    num_layers,
    epochs,
    seed,
):

    set_seed(seed)

    # Full-training scaler.
    # 2025 validation is not used.
    y_mean, y_std = fit_target_scaler(
        y
    )

    y_scaled = scale_target(
        y,
        y_mean,
        y_std,
    )

    model = VanillaRNN(
        input_size=X.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_scaled,
            dtype=torch.float32,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    for epoch in range(
        epochs
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
                max_norm=1.0,
            )

            optimizer.step()

    return (
        model,
        y_mean,
        y_std,
    )


(
    final_model,
    final_target_mean,
    final_target_std,
) = train_fixed_epochs(
    X_train,
    y_train,
    hidden_size=BEST_HIDDEN_SIZE,
    learning_rate=BEST_LEARNING_RATE,
    weight_decay=BEST_WEIGHT_DECAY,
    num_layers=BEST_NUM_LAYERS,
    epochs=BEST_EPOCHS,
    seed=GRID_SEED,
)


# ============================================================
# ONE official 2025 validation evaluation
# ============================================================

val_pred = predict_original_scale(
    final_model,
    X_val,
    final_target_mean,
    final_target_std,
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_pred,
    )
)

val_mae = mean_absolute_error(
    y_val,
    val_pred,
)

val_bias = np.mean(
    val_pred - y_val
)

val_nrmse = (
    val_rmse
    /
    np.std(
        y_val,
        ddof=0,
    )
)

prediction_mean = float(
    np.mean(
        val_pred
    )
)

actual_mean = float(
    np.mean(
        y_val
    )
)


# ============================================================
# Improvement helper
# ============================================================

def improvement(
    reference,
    candidate,
):

    if not np.isfinite(reference):
        return np.nan

    return (
        100.0
        *
        (
            reference - candidate
        )
        /
        reference
    )


vs_original_rnn = improvement(
    ORIGINAL_RNN_VAL_RMSE,
    val_rmse,
)

vs_ridge = improvement(
    RIDGE_F4_VAL_RMSE,
    val_rmse,
)

vs_esn = improvement(
    ESN_VAL_RMSE,
    val_rmse,
)

vs_seasonal = improvement(
    SEASONAL_NAIVE_VAL_RMSE,
    val_rmse,
)


# ============================================================
# Final result
# ============================================================

print()
print("=" * 90)
print("OFFICIAL 2025 VALIDATION - TUNED VANILLA RNN")
print("=" * 90)

print(
    f"MAE             : "
    f"{val_mae:.6f}"
)

print(
    f"RMSE            : "
    f"{val_rmse:.6f}"
)

print(
    f"NRMSE           : "
    f"{val_nrmse:.6f}"
)

print(
    f"Bias            : "
    f"{val_bias:+.6f}"
)

print(
    f"Actual mean     : "
    f"{actual_mean:.6f}"
)

print(
    f"Prediction mean : "
    f"{prediction_mean:.6f}"
)

print()
print(
    f"Improvement vs corrected Step 4.3A RNN : "
    f"{vs_original_rnn:+.3f}%"
)

print(
    f"Improvement vs Ridge F4                : "
    f"{vs_ridge:+.3f}%"
)

print(
    f"Improvement vs ESN                     : "
    f"{vs_esn:+.3f}%"
)

print(
    f"Improvement vs seasonal                : "
    f"{vs_seasonal:+.3f}%"
)

print()
print(
    "Positive improvement = tuned RNN is better."
)

print(
    "Negative improvement = tuned RNN is worse."
)


# ============================================================
# Save predictions
# ============================================================

prediction_df = pd.DataFrame({
    "y_true":
        y_val,

    "y_pred":
        val_pred,

    "residual":
        y_val - val_pred,
})

prediction_path = (
    RESULTS_DIR /
    "04_03c_vanilla_rnn_tuned_validation_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)


# ============================================================
# Save summary
# ============================================================

summary_path = (
    RESULTS_DIR /
    "04_03c_vanilla_rnn_tuning_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "STEP 4.3C - FINAL VANILLA RNN TUNING\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Target standardization: YES\n"
    )

    f.write(
        "Within chronological CV, target scaling is fitted "
        "independently on each fold's training targets only.\n"
    )

    f.write(
        "All RMSE values are reported in original "
        "claim-count units after inverse scaling.\n\n"
    )

    f.write(
        "Selected using 2022-2024 chronological CV only\n\n"
    )

    f.write(
        f"Feature set: {FEATURE_SET}\n"
    )

    f.write(
        f"Window: {WINDOW}\n"
    )

    f.write(
        f"Hidden size: "
        f"{BEST_HIDDEN_SIZE}\n"
    )

    f.write(
        f"Learning rate: "
        f"{BEST_LEARNING_RATE}\n"
    )

    f.write(
        f"L2: "
        f"{BEST_WEIGHT_DECAY}\n"
    )

    f.write(
        f"Layers: "
        f"{BEST_NUM_LAYERS}\n"
    )

    f.write(
        f"Parameters: "
        f"{BEST_PARAMETER_COUNT}\n"
    )

    f.write(
        f"Median epochs: "
        f"{BEST_EPOCHS}\n"
    )

    f.write(
        f"CV RMSE: "
        f"{best['cv_rmse_mean']:.6f} "
        f"+/- "
        f"{best['cv_rmse_std']:.6f}\n\n"
    )

    f.write(
        "Final full-training target scaler\n"
    )

    f.write(
        f"Mean: "
        f"{final_target_mean:.6f}\n"
    )

    f.write(
        f"Std: "
        f"{final_target_std:.6f}\n\n"
    )

    f.write(
        "Official 2025 validation\n"
    )

    f.write(
        f"MAE: "
        f"{val_mae:.6f}\n"
    )

    f.write(
        f"RMSE: "
        f"{val_rmse:.6f}\n"
    )

    f.write(
        f"NRMSE: "
        f"{val_nrmse:.6f}\n"
    )

    f.write(
        f"Bias: "
        f"{val_bias:+.6f}\n"
    )

    f.write(
        f"Actual mean: "
        f"{actual_mean:.6f}\n"
    )

    f.write(
        f"Prediction mean: "
        f"{prediction_mean:.6f}\n\n"
    )

    f.write(
        f"Improvement vs corrected Step 4.3A RNN: "
        f"{vs_original_rnn:+.3f}%\n"
    )

    f.write(
        f"Improvement vs Ridge F4: "
        f"{vs_ridge:+.3f}%\n"
    )

    f.write(
        f"Improvement vs ESN: "
        f"{vs_esn:+.3f}%\n"
    )

    f.write(
        f"Improvement vs seasonal naive: "
        f"{vs_seasonal:+.3f}%\n"
    )


print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(cv_path)
print(prediction_path)
print(summary_path)

print()
print("2026 TEST SET WAS NOT USED.")
