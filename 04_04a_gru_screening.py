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


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_DIMS = {
    "F2": 3,
    "F3": 4,
    "F4": 5,
}

FEATURE_SETS = ["F2", "F3", "F4"]
WINDOWS = [1, 2, 5, 7, 14, 21, 28]

# Deliberately includes very compact GRUs to study capacity.
HIDDEN_SIZES = [1, 2, 4, 8, 16, 32]

N_SPLITS = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_LAYERS = 1

BATCH_SIZE = 64

# High ceiling; early stopping normally terminates much earlier.
MAX_EPOCHS = 2000
PATIENCE = 30
MIN_DELTA = 1e-4

GRID_SEED = 42

DEVICE = torch.device("cpu")


# Existing official-validation references
RIDGE_F4_RMSE = 4.874844
ESN_BEST_RMSE = 4.856483

# Corrected normalized Stage-1 RNN result.
# Replace later with the final normalized tuned-RNN RMSE if desired.
RNN_CORRECTED_STAGE1_RMSE = 5.638507

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
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# =============================================================================
# TEMPORAL-WINDOW LOADER
# =============================================================================

def load_temporal_data(feature_set, window):

    expected_dim = FEATURE_DIMS[feature_set]

    path = (
        RESULTS_DIR /
        f"04_01_windows_{feature_set}_W{window:02d}.npz"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing temporal-window artifact: {path}"
        )

    with np.load(
        path,
        allow_pickle=True
    ) as data:

        required = {
            "X_train",
            "y_train",
            "X_val",
            "y_val",
        }

        if not required.issubset(
            set(data.files)
        ):

            raise KeyError(
                f"{path} does not contain all required keys.\n"
                f"Required: {sorted(required)}\n"
                f"Available: {data.files}"
            )

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

    if X_train.ndim != 3:
        raise ValueError(
            f"X_train must be 3-D, got {X_train.shape}"
        )

    if X_val.ndim != 3:
        raise ValueError(
            f"X_val must be 3-D, got {X_val.shape}"
        )

    if X_train.shape[1] != window:
        raise ValueError(
            f"Unexpected train window length in {path}: "
            f"{X_train.shape[1]} != {window}"
        )

    if X_train.shape[2] != expected_dim:
        raise ValueError(
            f"Unexpected input dimension in {path}: "
            f"{X_train.shape[2]} != {expected_dim}"
        )

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "X_train and y_train lengths do not match."
        )

    if X_val.shape[0] != len(y_val):
        raise ValueError(
            "X_val and y_val lengths do not match."
        )

    if not np.isfinite(X_train).all():
        raise ValueError(
            "X_train contains non-finite values."
        )

    if not np.isfinite(y_train).all():
        raise ValueError(
            "y_train contains non-finite values."
        )

    if not np.isfinite(X_val).all():
        raise ValueError(
            "X_val contains non-finite values."
        )

    if not np.isfinite(y_val).all():
        raise ValueError(
            "y_val contains non-finite values."
        )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        path
    )


# =============================================================================
# TARGET STANDARDIZATION
# =============================================================================

def fit_target_scaler(y_train):
    """
    Fit target scaler on TRAINING TARGETS ONLY.

        y_scaled = (y - mean) / std

    During chronological CV this is called independently
    inside each fold using only the fold-training targets.
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
            "Target std must be finite and > 0."
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
# GRU MODEL
# =============================================================================

class GRURegressor(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_size,
            1
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize each GRU gate block separately.

        PyTorch stores reset/update/candidate gate weights
        stacked along the first dimension.
        """

        for layer in range(
            self.num_layers
        ):

            weight_ih = getattr(
                self.gru,
                f"weight_ih_l{layer}"
            )

            weight_hh = getattr(
                self.gru,
                f"weight_hh_l{layer}"
            )

            bias_ih = getattr(
                self.gru,
                f"bias_ih_l{layer}"
            )

            bias_hh = getattr(
                self.gru,
                f"bias_hh_l{layer}"
            )

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
                bias_ih
            )

            nn.init.zeros_(
                bias_hh
            )

        nn.init.xavier_uniform_(
            self.output.weight
        )

        nn.init.zeros_(
            self.output.bias
        )

    def forward(self, x):

        # x:
        #   (batch, window_length, input_size)

        output, hidden = self.gru(
            x
        )

        # hidden:
        #   (num_layers, batch, hidden_size)
        #
        # Final state of last recurrent layer:
        h_last = hidden[-1]

        prediction_scaled = self.output(
            h_last
        )

        return prediction_scaled.squeeze(-1)


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
# METRICS
# =============================================================================

def regression_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    residual = (
        y_pred - y_true
    )

    mae = np.mean(
        np.abs(residual)
    )

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    bias = np.mean(
        residual
    )

    target_std = np.std(
        y_true,
        ddof=0
    )

    if target_std > 0:
        nrmse = rmse / target_std
    else:
        nrmse = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "nrmse": float(nrmse),
        "bias": float(bias),
    }


# =============================================================================
# PREDICTION
# =============================================================================

def predict_scaled(
    model,
    X
):

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


def predict_original_scale(
    model,
    X,
    y_mean,
    y_std
):

    pred_scaled = predict_scaled(
        model,
        X
    )

    return inverse_scale_target(
        pred_scaled,
        y_mean,
        y_std
    )


# =============================================================================
# ONE CV-FOLD FIT WITH EARLY STOPPING
# =============================================================================

def train_with_early_stopping(
    X_train,
    y_train,
    X_valid,
    y_valid,
    hidden_size,
    seed,
):

    set_seed(seed)

    # -------------------------------------------------------------------------
    # Fold-specific target normalization.
    # IMPORTANT: only fold-training targets are used.
    # -------------------------------------------------------------------------

    y_mean, y_std = fit_target_scaler(
        y_train
    )

    y_train_scaled = scale_target(
        y_train,
        y_mean,
        y_std
    )

    model = GRURegressor(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS
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
            y_train_scaled,
            dtype=torch.float32
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    best_val_rmse = np.inf
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

            # Same protection used in the vanilla RNN.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        # ---------------------------------------------------------------------
        # Validate in ORIGINAL claim-count units.
        # ---------------------------------------------------------------------

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
            < best_val_rmse - MIN_DELTA
        ):

            best_val_rmse = (
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
            "No best GRU model state was recorded."
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        float(best_val_rmse),
        int(best_epoch),
        int(epoch),
    )


# =============================================================================
# FIVE-FOLD CHRONOLOGICAL CV
# =============================================================================

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
    fold_epochs_run = []

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(
        splitter.split(X),
        start=1
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
            epochs_run
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

        fold_epochs_run.append(
            epochs_run
        )

    return (
        np.asarray(
            fold_rmses
        ),
        np.asarray(
            fold_epochs
        ),
        np.asarray(
            fold_epochs_run
        ),
    )


# =============================================================================
# FINAL FIT ON ALL 2022-2024 TRAINING DATA
# =============================================================================

def train_fixed_epochs(
    X_train,
    y_train,
    hidden_size,
    epochs,
    seed,
):

    set_seed(seed)

    y_mean, y_std = fit_target_scaler(
        y_train
    )

    y_train_scaled = scale_target(
        y_train,
        y_mean,
        y_std
    )

    model = GRURegressor(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS
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
            y_train_scaled,
            dtype=torch.float32
        ),
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
# MAIN SCREENING
# =============================================================================

def main():

    print("=" * 100)
    print("WEEK 4 - STEP 4.4A: GRU CHRONOLOGICAL ARCHITECTURE SCREENING")
    print("TARGET STANDARDIZATION: ENABLED")
    print("=" * 100)

    print()
    print("Screening:")
    print(f"  Feature sets:    {FEATURE_SETS}")
    print(f"  Windows:         {WINDOWS}")
    print(f"  Hidden sizes:    {HIDDEN_SIZES}")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  L2:              {WEIGHT_DECAY}")
    print(f"  Layers:          {NUM_LAYERS}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Max epochs:      {MAX_EPOCHS}")
    print(f"  Patience:        {PATIENCE}")
    print(f"  Min delta:       {MIN_DELTA}")
    print(f"  CV folds:        {N_SPLITS}")
    print(f"  Base seed:       {GRID_SEED}")

    expected_configs = (
        len(FEATURE_SETS)
        * len(WINDOWS)
        * len(HIDDEN_SIZES)
    )

    print()
    print(
        f"Expected GRU architectures: "
        f"{expected_configs}"
    )

    print(
        f"Expected chronological CV fits: "
        f"{expected_configs * N_SPLITS}"
    )

    print()
    print(
        "Target scaling is fitted separately "
        "inside every CV training fold."
    )

    print(
        "2025 validation is NOT used for "
        "architecture selection."
    )

    print(
        "2026 TEST SET is NOT loaded / NOT used."
    )

    rows = []

    # =========================================================================
    # ARCHITECTURE SEARCH ON 2022-2024 ONLY
    # =========================================================================

    for feature_set in FEATURE_SETS:

        for window in WINDOWS:

            (
                X_train,
                y_train,
                X_val,
                y_val,
                source_path
            ) = load_temporal_data(
                feature_set,
                window
            )

            print()
            print("-" * 100)

            print(
                f"{feature_set}, W={window}"
            )

            print(
                f"Train shape: "
                f"{X_train.shape}"
            )

            print(
                f"Official validation shape: "
                f"{X_val.shape}"
            )

            print(
                f"Source: "
                f"{source_path}"
            )

            print(
                f"Full training target mean: "
                f"{np.mean(y_train):.6f}"
            )

            print(
                f"Full training target std:  "
                f"{np.std(y_train, ddof=0):.6f}"
            )

            for hidden_size in HIDDEN_SIZES:

                (
                    fold_rmse,
                    fold_epochs,
                    fold_epochs_run
                ) = chronological_cv(
                    X_train,
                    y_train,
                    hidden_size=hidden_size,
                    seed=GRID_SEED,
                )

                mean_rmse = float(
                    fold_rmse.mean()
                )

                std_rmse = float(
                    fold_rmse.std(
                        ddof=1
                    )
                )

                median_epoch = int(
                    np.median(
                        fold_epochs
                    )
                )

                median_epochs_run = int(
                    np.median(
                        fold_epochs_run
                    )
                )

                temp_model = GRURegressor(
                    input_size=X_train.shape[2],
                    hidden_size=hidden_size,
                    num_layers=NUM_LAYERS
                )

                n_parameters = (
                    count_parameters(
                        temp_model
                    )
                )

                print(
                    f"{feature_set:>2} | "
                    f"W={window:>2} | "
                    f"H={hidden_size:>2} | "
                    f"Params={n_parameters:>5} | "
                    f"CV RMSE="
                    f"{mean_rmse:.4f} "
                    f"+/- {std_rmse:.4f} | "
                    f"best epoch={median_epoch} | "
                    f"epochs run={median_epochs_run}"
                )

                if np.any(
                    fold_epochs == MAX_EPOCHS
                ):

                    print(
                        "      WARNING: at least one fold's "
                        "best epoch reached MAX_EPOCHS."
                    )

                rows.append({
                    "feature_set":
                        feature_set,

                    "window":
                        window,

                    "input_size":
                        X_train.shape[2],

                    "hidden_size":
                        hidden_size,

                    "num_layers":
                        NUM_LAYERS,

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "batch_size":
                        BATCH_SIZE,

                    "max_epochs":
                        MAX_EPOCHS,

                    "patience":
                        PATIENCE,

                    "parameter_count":
                        n_parameters,

                    "cv_rmse_mean":
                        mean_rmse,

                    "cv_rmse_std":
                        std_rmse,

                    "median_best_epoch":
                        median_epoch,

                    "median_epochs_run":
                        median_epochs_run,

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

                    "fold_epochs_run":
                        ",".join(
                            str(int(x))
                            for x in fold_epochs_run
                        ),
                })

    # =========================================================================
    # SAVE CV RESULTS
    # =========================================================================

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
        "04_04a_gru_screening_cv_results.csv"
    )

    results_df.to_csv(
        cv_path,
        index=False
    )

    # =========================================================================
    # BEST CONFIGURATION
    # =========================================================================

    best = results_df.iloc[0]

    best_feature_set = str(
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
    print("=" * 100)
    print("TOP 15 GRU CV CONFIGURATIONS")
    print("=" * 100)

    print(
        results_df[
            [
                "feature_set",
                "window",
                "hidden_size",
                "parameter_count",
                "cv_rmse_mean",
                "cv_rmse_std",
                "median_best_epoch",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("BEST GRU CV CONFIGURATION")
    print("=" * 100)

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
        f"Parameters  : "
        f"{int(best['parameter_count'])}"
    )

    print(
        f"CV RMSE     : "
        f"{best['cv_rmse_mean']:.6f} "
        f"+/- "
        f"{best['cv_rmse_std']:.6f}"
    )

    print(
        f"Median epoch: "
        f"{best_epochs}"
    )

    # =========================================================================
    # LOAD BEST ARCHITECTURE DATA
    # =========================================================================

    (
        X_train,
        y_train,
        X_val,
        y_val,
        source_path
    ) = load_temporal_data(
        best_feature_set,
        best_window
    )

    # =========================================================================
    # FINAL TRAINING ON ALL 2022-2024 DATA
    # =========================================================================

    (
        final_model,
        target_mean,
        target_std
    ) = train_fixed_epochs(
        X_train,
        y_train,
        hidden_size=best_hidden_size,
        epochs=best_epochs,
        seed=GRID_SEED,
    )

    # =========================================================================
    # ONE OFFICIAL 2025 VALIDATION EVALUATION
    # =========================================================================

    val_pred = (
        predict_original_scale(
            final_model,
            X_val,
            target_mean,
            target_std
        )
    )

    metrics = regression_metrics(
        y_val,
        val_pred
    )

    def improvement(
        reference,
        candidate
    ):

        return (
            100.0
            * (
                reference - candidate
            )
            / reference
        )

    vs_ridge = improvement(
        RIDGE_F4_RMSE,
        metrics["rmse"]
    )

    vs_esn = improvement(
        ESN_BEST_RMSE,
        metrics["rmse"]
    )

    vs_rnn = improvement(
        RNN_CORRECTED_STAGE1_RMSE,
        metrics["rmse"]
    )

    vs_seasonal = improvement(
        SEASONAL_NAIVE_RMSE,
        metrics["rmse"]
    )

    print()
    print("=" * 100)
    print("OFFICIAL 2025 VALIDATION - SELECTED GRU")
    print("=" * 100)

    print(
        f"Target scaler mean : "
        f"{target_mean:.6f}"
    )

    print(
        f"Target scaler std  : "
        f"{target_std:.6f}"
    )

    print()
    print(
        f"MAE   : "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"RMSE  : "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"NRMSE : "
        f"{metrics['nrmse']:.6f}"
    )

    print(
        f"Bias  : "
        f"{metrics['bias']:+.6f}"
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
        f"Improvement vs corrected Stage-1 RNN: "
        f"{vs_rnn:+.3f}%"
    )

    print(
        f"Improvement vs seasonal naive: "
        f"{vs_seasonal:+.3f}%"
    )

    # =========================================================================
    # SAVE OFFICIAL VALIDATION PREDICTIONS
    # =========================================================================

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
        "04_04a_gru_validation_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    summary_path = (
        RESULTS_DIR /
        "04_04a_gru_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 4.4A - GRU ARCHITECTURE SCREENING\n"
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
            "2025 validation was not used for architecture selection.\n"
        )

        f.write(
            "2026 test set was not used.\n\n"
        )

        f.write(
            f"Feature set: "
            f"{best_feature_set}\n"
        )

        f.write(
            f"Window: "
            f"{best_window}\n"
        )

        f.write(
            f"Hidden size: "
            f"{best_hidden_size}\n"
        )

        f.write(
            f"Parameters: "
            f"{int(best['parameter_count'])}\n"
        )

        f.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )

        f.write(
            f"L2: "
            f"{WEIGHT_DECAY}\n"
        )

        f.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            f"Maximum epochs: "
            f"{MAX_EPOCHS}\n"
        )

        f.write(
            f"Patience: "
            f"{PATIENCE}\n"
        )

        f.write(
            f"Median best epoch: "
            f"{best_epochs}\n"
        )

        f.write(
            f"CV RMSE: "
            f"{best['cv_rmse_mean']:.6f} "
            f"+/- "
            f"{best['cv_rmse_std']:.6f}\n\n"
        )

        f.write(
            "Full-training target scaler\n"
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
            "Official 2025 validation\n"
        )

        f.write(
            f"MAE: "
            f"{metrics['mae']:.6f}\n"
        )

        f.write(
            f"RMSE: "
            f"{metrics['rmse']:.6f}\n"
        )

        f.write(
            f"NRMSE: "
            f"{metrics['nrmse']:.6f}\n"
        )

        f.write(
            f"Bias: "
            f"{metrics['bias']:+.6f}\n"
        )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print(cv_path)
    print(prediction_path)
    print(summary_path)

    print()
    print(
        "2026 TEST SET WAS NOT USED."
    )


if __name__ == "__main__":
    main()
