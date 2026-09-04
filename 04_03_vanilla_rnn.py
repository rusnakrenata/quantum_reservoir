from pathlib import Path
import copy
import random

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

FEATURE_DIMS = {
    "F2": 3,
    "F3": 4,
    "F4": 5,
}

FEATURE_SETS = ["F2", "F3", "F4"]

WINDOWS = [1, 2, 5, 7, 14, 21, 28]

HIDDEN_SIZES = [8, 16, 32, 64]

N_SPLITS = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 64

MAX_EPOCHS = 200
PATIENCE = 20
MIN_DELTA = 1e-4

GRID_SEED = 42

DEVICE = torch.device("cpu")


# Existing reference models
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
# Temporal-window loader
# ============================================================

def load_temporal_config(feature_set, window):
    """
    Searches results/*.npz for a Step-4.1 temporal tensor with:

        X_train.shape = (N, window, feature_dimension)

    and corresponding:
        y_train
        X_val
        y_val

    This avoids relying on an exact file name.
    """

    expected_dim = FEATURE_DIMS[feature_set]

    required_keys = {
        "X_train",
        "y_train",
        "X_val",
        "y_val",
    }

    matches = []

    for path in RESULTS_DIR.glob("*.npz"):

        try:

            with np.load(path, allow_pickle=True) as data:

                keys = set(data.files)

                if not required_keys.issubset(keys):
                    continue

                X_train = np.asarray(
                    data["X_train"]
                )

                if X_train.ndim != 3:
                    continue

                if X_train.shape[1] != window:
                    continue

                if X_train.shape[2] != expected_dim:
                    continue

                matches.append(path)

        except Exception:
            continue

    if len(matches) == 0:

        raise FileNotFoundError(
            f"\nCould not find temporal tensor for "
            f"{feature_set}, W={window}.\n"
            f"Expected shape: "
            f"(N, {window}, {expected_dim}).\n"
            f"Check the .npz files in results/."
        )

    if len(matches) > 1:

        print(
            f"WARNING: multiple matches for "
            f"{feature_set}, W={window}:"
        )

        for p in matches:
            print("   ", p)

        print("Using:", matches[0])

    path = matches[0]

    with np.load(path, allow_pickle=True) as data:

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

    assert X_train.ndim == 3
    assert X_val.ndim == 3

    assert X_train.shape[0] == len(y_train)
    assert X_val.shape[0] == len(y_val)

    assert np.isfinite(X_train).all()
    assert np.isfinite(y_train).all()
    assert np.isfinite(X_val).all()
    assert np.isfinite(y_val).all()

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        path
    )


# ============================================================
# Target standardization
# ============================================================

def fit_target_scaler(y_train):
    """
    Fit target scaling using TRAINING TARGETS ONLY.

    y_scaled = (y - mean) / std

    This function must always receive only the training
    portion relevant to the current fit.

    In chronological CV:
        scaler is fitted separately inside each fold.

    For final training:
        scaler is fitted using the full training period.

    Validation/test targets are NEVER used to fit this scaler.
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


# ============================================================
# Vanilla RNN
# ============================================================

class VanillaRNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size
    ):

        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )

        self.readout = nn.Linear(
            hidden_size,
            1
        )

        self._initialize_weights()

    def _initialize_weights(self):

        # Input -> hidden
        nn.init.xavier_uniform_(
            self.rnn.weight_ih_l0
        )

        # Hidden -> hidden
        #
        # Orthogonal initialization helps stabilize
        # gradient propagation through time.
        nn.init.orthogonal_(
            self.rnn.weight_hh_l0
        )

        nn.init.zeros_(
            self.rnn.bias_ih_l0
        )

        nn.init.zeros_(
            self.rnn.bias_hh_l0
        )

        nn.init.xavier_uniform_(
            self.readout.weight
        )

        nn.init.zeros_(
            self.readout.bias
        )

    def forward(self, x):

        sequence_output, hidden = self.rnn(
            x
        )

        # sequence_output:
        #
        #   (batch, W, hidden_size)
        #
        # We use the final temporal representation.

        last_hidden = (
            sequence_output[:, -1, :]
        )

        prediction = self.readout(
            last_hidden
        )

        return prediction.squeeze(-1)


# ============================================================
# Raw model prediction
# ============================================================

def predict_scaled(
    model,
    X
):
    """
    Returns predictions in STANDARDIZED target space.
    """

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
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


# ============================================================
# Prediction in original claim-count units
# ============================================================

def predict_original_scale(
    model,
    X,
    y_mean,
    y_std
):
    """
    Model output is standardized.

    Convert it back to original claim-count units:

        y_hat = mean + std * y_hat_scaled
    """

    pred_scaled = predict_scaled(
        model,
        X
    )

    pred = inverse_scale_target(
        pred_scaled,
        y_mean,
        y_std
    )

    return pred


# ============================================================
# Train one model with early stopping
# ============================================================

def train_with_early_stopping(
    X_train,
    y_train,
    X_valid,
    y_valid,
    hidden_size,
    seed,
):

    set_seed(seed)

    # --------------------------------------------------------
    # Target scaling
    #
    # IMPORTANT:
    # Only THIS FOLD'S TRAINING TARGETS are used.
    # --------------------------------------------------------

    y_mean, y_std = fit_target_scaler(
        y_train
    )

    y_train_scaled = scale_target(
        y_train,
        y_mean,
        y_std
    )

    input_size = X_train.shape[2]

    model = VanillaRNN(
        input_size=input_size,
        hidden_size=hidden_size,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(

        torch.tensor(
            X_train,
            dtype=torch.float32
        ),

        torch.tensor(
            y_train_scaled,
            dtype=torch.float32
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
        MAX_EPOCHS + 1
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

            # Protect vanilla RNN against
            # exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        # ----------------------------------------------------
        # Validation
        #
        # Predictions are converted BACK to original
        # claim-count units before computing RMSE.
        # ----------------------------------------------------

        validation_pred = (
            predict_original_scale(
                model,
                X_valid,
                y_mean,
                y_std
            )
        )

        validation_rmse = np.sqrt(
            mean_squared_error(
                y_valid,
                validation_pred
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
            "No best RNN state was recorded."
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_rmse,
        best_epoch,
        y_mean,
        y_std
    )


# ============================================================
# Five-fold chronological CV
# ============================================================

def chronological_cv(
    X,
    y,
    hidden_size,
    seed,
):

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    fold_rmses = []
    fold_epochs = []

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(
        splitter.split(X),
        start=1,
    ):

        X_fold_train = X[
            train_idx
        ]

        y_fold_train = y[
            train_idx
        ]

        X_fold_valid = X[
            valid_idx
        ]

        y_fold_valid = y[
            valid_idx
        ]

        (
            _,
            rmse,
            best_epoch,
            _,
            _
        ) = train_with_early_stopping(
            X_fold_train,
            y_fold_train,
            X_fold_valid,
            y_fold_valid,
            hidden_size=hidden_size,
            seed=seed + fold,
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
# Train on all training data for fixed number of epochs
# ============================================================

def train_fixed_epochs(
    X_train,
    y_train,
    hidden_size,
    epochs,
    seed,
):

    set_seed(seed)

    # --------------------------------------------------------
    # Full training-period target scaler.
    #
    # Validation remains excluded.
    # --------------------------------------------------------

    y_mean, y_std = fit_target_scaler(
        y_train
    )

    y_train_scaled = scale_target(
        y_train,
        y_mean,
        y_std
    )

    model = VanillaRNN(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(

        torch.tensor(
            X_train,
            dtype=torch.float32
        ),

        torch.tensor(
            y_train_scaled,
            dtype=torch.float32
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    for epoch in range(epochs):

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


# ============================================================
# Main grid
# ============================================================

rows = []

print("=" * 80)
print("STEP 4.3A - VANILLA RNN")
print("Chronological architecture screening")
print("TARGET STANDARDIZATION: ENABLED")
print("=" * 80)

print()
print(
    "Target scaling:"
)

print(
    "  y_scaled = "
    "(y - training_mean) / training_std"
)

print(
    "  CV scaler is fitted separately "
    "inside each training fold."
)

print(
    "  Validation target is NEVER used "
    "to fit target scaling."
)

print(
    "  All reported RMSE values remain "
    "in original claim-count units."
)

print()

for feature_set in FEATURE_SETS:

    for window in WINDOWS:

        (
            X_train,
            y_train,
            X_val,
            y_val,
            source_file
        ) = load_temporal_config(
            feature_set,
            window
        )

        print()
        print("-" * 80)

        print(
            f"{feature_set}, W={window}"
        )

        print(
            f"Train shape: "
            f"{X_train.shape}"
        )

        print(
            f"Validation shape: "
            f"{X_val.shape}"
        )

        print(
            f"Source: "
            f"{source_file}"
        )

        print(
            f"Training target mean: "
            f"{np.mean(y_train):.6f}"
        )

        print(
            f"Training target std:  "
            f"{np.std(y_train, ddof=0):.6f}"
        )

        for hidden_size in HIDDEN_SIZES:

            (
                fold_rmse,
                fold_epochs
            ) = chronological_cv(
                X_train,
                y_train,
                hidden_size=hidden_size,
                seed=GRID_SEED,
            )

            mean_rmse = (
                fold_rmse.mean()
            )

            std_rmse = (
                fold_rmse.std(
                    ddof=1
                )
            )

            median_epoch = int(
                np.median(
                    fold_epochs
                )
            )

            print(
                f"{feature_set:>2} | "
                f"W={window:>2} | "
                f"H={hidden_size:>2} | "
                f"CV RMSE="
                f"{mean_rmse:.4f} "
                f"+/- {std_rmse:.4f} | "
                f"epochs={median_epoch}"
            )

            rows.append({

                "feature_set":
                    feature_set,

                "window":
                    window,

                "hidden_size":
                    hidden_size,

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
# Save CV results
# ============================================================

results_df = pd.DataFrame(
    rows
)

results_df = results_df.sort_values(
    "cv_rmse_mean"
).reset_index(
    drop=True
)

csv_path = (
    RESULTS_DIR /
    "04_03_vanilla_rnn_cv_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# Best architecture
# ============================================================

best = results_df.iloc[0]

best_feature_set = (
    best["feature_set"]
)

best_window = int(
    best["window"]
)

best_hidden_size = int(
    best["hidden_size"]
)

best_epochs = int(
    best["median_best_epoch"]
)


print()
print("=" * 80)
print("BEST VANILLA RNN CV CONFIGURATION")
print("=" * 80)

print(
    f"Feature set : "
    f"{best_feature_set}"
)

print(
    f"Window      : "
    f"{best_window}"
)

print(
    f"Hidden size : "
    f"{best_hidden_size}"
)

print(
    f"CV RMSE     : "
    f"{best['cv_rmse_mean']:.6f} "
    f"+/- "
    f"{best['cv_rmse_std']:.6f}"
)

print(
    f"Epochs      : "
    f"{best_epochs}"
)


# ============================================================
# Load best data
# ============================================================

(
    X_train,
    y_train,
    X_val,
    y_val,
    source_file
) = load_temporal_config(
    best_feature_set,
    best_window
)


# ============================================================
# Train once on full training set
# ============================================================

(
    best_model,
    target_mean,
    target_std
) = train_fixed_epochs(
    X_train,
    y_train,
    hidden_size=best_hidden_size,
    epochs=best_epochs,
    seed=GRID_SEED,
)


print()
print("=" * 80)
print("FINAL TRAINING TARGET SCALER")
print("=" * 80)

print(
    f"Training target mean : "
    f"{target_mean:.6f}"
)

print(
    f"Training target std  : "
    f"{target_std:.6f}"
)


# ============================================================
# Official validation evaluation
# ============================================================

val_pred = (
    predict_original_scale(
        best_model,
        X_val,
        target_mean,
        target_std
    )
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_pred
    )
)

val_mae = mean_absolute_error(
    y_val,
    val_pred
)

val_bias = np.mean(
    val_pred - y_val
)

val_std = np.std(
    y_val,
    ddof=0
)

val_nrmse = (
    val_rmse / val_std
)


def improvement(
    reference,
    candidate
):

    return (
        100.0
        * (reference - candidate)
        / reference
    )


vs_ridge = improvement(
    RIDGE_F4_VAL_RMSE,
    val_rmse
)

vs_esn = improvement(
    ESN_VAL_RMSE,
    val_rmse
)

vs_seasonal = improvement(
    SEASONAL_NAIVE_VAL_RMSE,
    val_rmse
)


print()
print("=" * 80)
print("OFFICIAL VALIDATION RESULT")
print("=" * 80)

print(
    f"MAE   : "
    f"{val_mae:.6f}"
)

print(
    f"RMSE  : "
    f"{val_rmse:.6f}"
)

print(
    f"NRMSE : "
    f"{val_nrmse:.6f}"
)

print(
    f"Bias  : "
    f"{val_bias:.6f}"
)

print()

print(
    f"Improvement vs Ridge F4: "
    f"{vs_ridge:+.3f}%"
)

print(
    f"Improvement vs ESN:      "
    f"{vs_esn:+.3f}%"
)

print(
    f"Improvement vs seasonal: "
    f"{vs_seasonal:+.3f}%"
)

print()

print(
    "Positive improvement = "
    "Vanilla RNN is better."
)

print(
    "Negative improvement = "
    "Vanilla RNN is worse."
)


# ============================================================
# Save validation predictions
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
    "04_03_vanilla_rnn_validation_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# Save summary
# ============================================================

summary_path = (
    RESULTS_DIR /
    "04_03_vanilla_rnn_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 4.3A - VANILLA RNN\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Target standardization: YES\n"
    )

    f.write(
        "Target scaler fitted on "
        "training data only.\n"
    )

    f.write(
        "Within chronological CV, "
        "scaler fitted independently "
        "inside every training fold.\n\n"
    )

    f.write(
        f"Best feature set: "
        f"{best_feature_set}\n"
    )

    f.write(
        f"Best window: "
        f"{best_window}\n"
    )

    f.write(
        f"Best hidden size: "
        f"{best_hidden_size}\n"
    )

    f.write(
        f"CV RMSE: "
        f"{best['cv_rmse_mean']:.6f} "
        f"+/- "
        f"{best['cv_rmse_std']:.6f}\n"
    )

    f.write(
        f"Median best epoch: "
        f"{best_epochs}\n\n"
    )

    f.write(
        "Final training target scaler\n"
    )

    f.write(
        f"Mean: "
        f"{target_mean:.6f}\n"
    )

    f.write(
        f"Std: "
        f"{target_std:.6f}\n\n"
    )

    f.write(
        "Official validation\n"
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
        f"{val_bias:.6f}\n\n"
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
print("Saved:")
print(csv_path)
print(prediction_path)
print(summary_path)

print()

print(
    "TEST SET WAS NOT USED."
)