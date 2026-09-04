from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CV_RESULTS_FILE = (
    RESULTS_DIR /
    "04_03_vanilla_rnn_cv_results.csv"
)

STEP43A_PREDICTIONS_FILE = (
    RESULTS_DIR /
    "04_03_vanilla_rnn_validation_predictions.csv"
)

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 64

# Seed 42 reproduces the Step 4.3A final fit.
# Seeds 43-51 test initialization sensitivity.
SEEDS = list(range(42, 52))

DEVICE = torch.device("cpu")


# ============================================================
# Read winning configuration automatically from Step 4.3A
# ============================================================

if not CV_RESULTS_FILE.exists():

    raise FileNotFoundError(
        f"Missing Step 4.3A CV results:\n"
        f"{CV_RESULTS_FILE}"
    )


cv_df = pd.read_csv(
    CV_RESULTS_FILE
)

cv_df = cv_df.sort_values(
    "cv_rmse_mean"
).reset_index(drop=True)

winner = cv_df.iloc[0]

FEATURE_SET = str(
    winner["feature_set"]
)

WINDOW = int(
    winner["window"]
)

HIDDEN_SIZE = int(
    winner["hidden_size"]
)

# Selected only from training-period chronological CV
N_EPOCHS = int(
    winner["median_best_epoch"]
)

DATA_FILE = (
    RESULTS_DIR /
    f"04_01_windows_{FEATURE_SET}_W{WINDOW:02d}.npz"
)


print("=" * 80)
print("STEP 4.3B - VANILLA RNN DIAGNOSTICS")
print("TARGET STANDARDIZATION: ENABLED")
print("=" * 80)

print()
print("Winner inherited from Step 4.3A:")
print(f"  Feature set : {FEATURE_SET}")
print(f"  Window      : {WINDOW}")
print(f"  Hidden size : {HIDDEN_SIZE}")
print(f"  Epochs      : {N_EPOCHS}")
print(
    f"  CV RMSE     : "
    f"{winner['cv_rmse_mean']:.6f} "
    f"+/- "
    f"{winner['cv_rmse_std']:.6f}"
)

print()
print(
    "No architecture selection is performed "
    "using the 2025 validation period."
)


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
    Fit target scaler using TRAINING TARGETS ONLY.

        y_scaled = (y - mean) / std

    Validation targets are never used.
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
        or y_std <= 0
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


# ============================================================
# Date extraction
# ============================================================

def find_date_array(
    data,
    possible_keys,
    expected_length
):

    for key in possible_keys:

        if key not in data.files:
            continue

        values = np.asarray(
            data[key]
        ).reshape(-1)

        if len(values) != expected_length:
            continue

        try:

            dates = pd.to_datetime(
                values
            )

            if dates.notna().all():

                return (
                    pd.DatetimeIndex(dates),
                    key
                )

        except Exception:
            continue

    return None, None


# ============================================================
# Load temporal data
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Missing temporal tensor:\n"
        f"{DATA_FILE}"
    )


with np.load(
    DATA_FILE,
    allow_pickle=True
) as data:

    print()
    print("=" * 80)
    print("DATA FILE")
    print("=" * 80)

    print(DATA_FILE)

    print()
    print("Available keys:")
    print(data.files)

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

    train_dates, train_date_key = (
        find_date_array(
            data,
            [
                "target_dates_train",
                "target_date_train",
                "dates_train",
                "date_train",
                "y_train_dates",
                "train_dates",
            ],
            len(y_train),
        )
    )

    val_dates, val_date_key = (
        find_date_array(
            data,
            [
                "target_dates_val",
                "target_date_val",
                "dates_val",
                "date_val",
                "y_val_dates",
                "validation_dates",
                "val_dates",
            ],
            len(y_val),
        )
    )


# ============================================================
# Date fallback
# ============================================================

if train_dates is None:

    # The first target for window W occurs W-1 days
    # after the W=1 first target (2022-01-02).

    train_start = (
        pd.Timestamp("2022-01-02")
        + pd.Timedelta(
            days=WINDOW - 1
        )
    )

    train_dates = pd.date_range(
        start=train_start,
        periods=len(y_train),
        freq="D",
    )

    train_date_key = (
        "constructed_from_known_split"
    )


if val_dates is None:

    if len(y_val) == 365:

        val_dates = pd.date_range(
            start="2025-01-01",
            end="2025-12-31",
            freq="D",
        )

        val_date_key = (
            "constructed_from_known_split"
        )

    else:

        raise ValueError(
            "Could not identify validation dates."
        )


print()
print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

print()
print(
    "Training date source:",
    train_date_key
)

print(
    "Validation date source:",
    val_date_key
)

print(
    "Training target dates:",
    train_dates.min().date(),
    "->",
    train_dates.max().date(),
)

print(
    "Validation target dates:",
    val_dates.min().date(),
    "->",
    val_dates.max().date(),
)


# ============================================================
# Full-training target scaler
#
# Used only after the architecture has already been selected.
# 2025 validation is NOT included.
# ============================================================

TARGET_MEAN, TARGET_STD = (
    fit_target_scaler(
        y_train
    )
)


print()
print("=" * 80)
print("TARGET SCALER")
print("=" * 80)

print(
    f"Training target mean : "
    f"{TARGET_MEAN:.6f}"
)

print(
    f"Training target std  : "
    f"{TARGET_STD:.6f}"
)

print(
    "Scaler fitted using training targets only."
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

        nn.init.xavier_uniform_(
            self.rnn.weight_ih_l0
        )

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

        output, hidden = self.rnn(
            x
        )

        last_hidden = (
            output[:, -1, :]
        )

        prediction = self.readout(
            last_hidden
        )

        return prediction.squeeze(-1)


# ============================================================
# Training
# ============================================================

def train_fixed_epochs(
    X_train,
    y_train,
    seed
):

    set_seed(seed)

    # --------------------------------------------------------
    # Target standardization
    # --------------------------------------------------------

    y_mean, y_std = (
        fit_target_scaler(
            y_train
        )
    )

    y_train_scaled = (
        scale_target(
            y_train,
            y_mean,
            y_std
        )
    )

    model = VanillaRNN(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
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

    for epoch in range(
        1,
        N_EPOCHS + 1
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


# ============================================================
# Prediction
# ============================================================

def predict_original_scale(
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

        pred_scaled = model(
            X_tensor
        )

    pred_scaled = (
        pred_scaled
        .cpu()
        .numpy()
    )

    prediction = inverse_scale_target(
        pred_scaled,
        y_mean,
        y_std
    )

    return prediction


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    bias = np.mean(
        y_pred - y_true
    )

    nrmse = (
        rmse /
        np.std(
            y_true,
            ddof=0
        )
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "bias": bias,
    }


# ============================================================
# PART A
# Inspect five CV folds of winning architecture
# ============================================================

print()
print("=" * 80)
print("PART A - WINNING CONFIGURATION: INDIVIDUAL CV FOLDS")
print("=" * 80)


winner_rows = cv_df[
    (cv_df["feature_set"] == FEATURE_SET) &
    (cv_df["window"] == WINDOW) &
    (cv_df["hidden_size"] == HIDDEN_SIZE)
]


if len(winner_rows) != 1:

    raise ValueError(
        "Could not uniquely identify "
        "winning CV row."
    )


winner_row = winner_rows.iloc[0]


fold_rmse = [
    float(x)
    for x in str(
        winner_row["fold_rmse"]
    ).split(",")
]


fold_epochs = [
    int(x)
    for x in str(
        winner_row[
            "fold_best_epochs"
        ]
    ).split(",")
]


splitter = TimeSeriesSplit(
    n_splits=5
)

fold_rows = []


for fold, (
    train_idx,
    valid_idx
) in enumerate(
    splitter.split(X_train),
    start=1
):

    # Reconstruct scaler information for audit only.
    y_fold_train = y_train[
        train_idx
    ]

    fold_target_mean = float(
        np.mean(
            y_fold_train
        )
    )

    fold_target_std = float(
        np.std(
            y_fold_train,
            ddof=0
        )
    )

    row = {

        "fold":
            fold,

        "train_start":
            train_dates[
                train_idx[0]
            ],

        "train_end":
            train_dates[
                train_idx[-1]
            ],

        "validation_start":
            train_dates[
                valid_idx[0]
            ],

        "validation_end":
            train_dates[
                valid_idx[-1]
            ],

        "train_n":
            len(train_idx),

        "validation_n":
            len(valid_idx),

        "target_scaler_mean":
            fold_target_mean,

        "target_scaler_std":
            fold_target_std,

        "rmse":
            fold_rmse[
                fold - 1
            ],

        "best_epoch":
            fold_epochs[
                fold - 1
            ],
    }

    fold_rows.append(
        row
    )


fold_df = pd.DataFrame(
    fold_rows
)


for _, row in fold_df.iterrows():

    print(
        f"Fold {int(row['fold'])}: "
        f"validation "
        f"{row['validation_start'].date()} "
        f"-> "
        f"{row['validation_end'].date()} | "
        f"RMSE={row['rmse']:.6f} | "
        f"best epoch="
        f"{int(row['best_epoch'])} | "
        f"target mean="
        f"{row['target_scaler_mean']:.3f} | "
        f"target std="
        f"{row['target_scaler_std']:.3f}"
    )


print()
print(
    "Mean fold RMSE:",
    f"{np.mean(fold_rmse):.6f}"
)

print(
    "Std fold RMSE:",
    f"{np.std(fold_rmse, ddof=1):.6f}"
)


fold_path = (
    RESULTS_DIR /
    "04_03b_rnn_cv_fold_diagnostics.csv"
)

fold_df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# PART B
# Train target level versus validation target level
# ============================================================

print()
print("=" * 80)
print("PART B - TARGET LEVEL SHIFT")
print("=" * 80)


train_mean = float(
    np.mean(y_train)
)

validation_mean = float(
    np.mean(y_val)
)

absolute_shift = (
    validation_mean -
    train_mean
)

relative_shift = (
    100.0 *
    absolute_shift /
    train_mean
)


print(
    f"Training target mean   : "
    f"{train_mean:.6f}"
)

print(
    f"2025 target mean       : "
    f"{validation_mean:.6f}"
)

print(
    f"Absolute mean shift    : "
    f"{absolute_shift:+.6f}"
)

print(
    f"Relative mean shift    : "
    f"{relative_shift:+.3f}%"
)


# ============================================================
# PART C
# Seed robustness
# ============================================================

print()
print("=" * 80)
print("PART C - 10-SEED ROBUSTNESS")
print("=" * 80)


seed_rows = []
all_predictions = []


for seed in SEEDS:

    (
        model,
        y_mean,
        y_std
    ) = train_fixed_epochs(
        X_train,
        y_train,
        seed=seed,
    )

    prediction = (
        predict_original_scale(
            model,
            X_val,
            y_mean,
            y_std
        )
    )

    metrics = calculate_metrics(
        y_val,
        prediction
    )

    seed_rows.append({

        "seed":
            seed,

        "rmse":
            metrics["rmse"],

        "mae":
            metrics["mae"],

        "nrmse":
            metrics["nrmse"],

        "bias":
            metrics["bias"],

        "prediction_mean":
            float(
                np.mean(
                    prediction
                )
            ),

        "target_scaler_mean":
            y_mean,

        "target_scaler_std":
            y_std,
    })

    all_predictions.append(
        prediction
    )

    print(
        f"Seed {seed}: "
        f"RMSE="
        f"{metrics['rmse']:.6f} | "
        f"MAE="
        f"{metrics['mae']:.6f} | "
        f"Bias="
        f"{metrics['bias']:+.6f}"
    )


seed_df = pd.DataFrame(
    seed_rows
)

all_predictions = np.asarray(
    all_predictions
)


seed_path = (
    RESULTS_DIR /
    "04_03b_rnn_seed_results.csv"
)

seed_df.to_csv(
    seed_path,
    index=False
)


print()
print("-" * 80)

print(
    "RMSE mean +/- std:",
    f"{seed_df['rmse'].mean():.6f} "
    f"+/- "
    f"{seed_df['rmse'].std(ddof=1):.6f}"
)

print(
    "RMSE min:",
    f"{seed_df['rmse'].min():.6f}"
)

print(
    "RMSE max:",
    f"{seed_df['rmse'].max():.6f}"
)

print()

print(
    "Bias mean +/- std:",
    f"{seed_df['bias'].mean():+.6f} "
    f"+/- "
    f"{seed_df['bias'].std(ddof=1):.6f}"
)

print(
    "Prediction mean across seeds:",
    f"{seed_df['prediction_mean'].mean():.6f}"
)


# ============================================================
# Verify seed 42 against corrected Step 4.3A
# ============================================================

seed42_rmse = float(
    seed_df.loc[
        seed_df["seed"] == 42,
        "rmse"
    ].iloc[0]
)


if STEP43A_PREDICTIONS_FILE.exists():

    step43a_df = pd.read_csv(
        STEP43A_PREDICTIONS_FILE
    )

    step43a_rmse = np.sqrt(
        mean_squared_error(
            step43a_df["y_true"],
            step43a_df["y_pred"]
        )
    )

    print()
    print("=" * 80)
    print("SEED 42 REPRODUCTION CHECK")
    print("=" * 80)

    print(
        "Step 4.3B seed 42 RMSE:",
        f"{seed42_rmse:.6f}"
    )

    print(
        "Corrected Step 4.3A RMSE:",
        f"{step43a_rmse:.6f}"
    )

    print(
        "Absolute reproduction difference:",
        f"{abs(seed42_rmse - step43a_rmse):.8f}"
    )

else:

    step43a_rmse = np.nan

    print()
    print(
        "WARNING: corrected Step 4.3A "
        "prediction file was not found."
    )


# ============================================================
# PART D
# Daily prediction uncertainty across seeds
# ============================================================

seed_mean_prediction = np.mean(
    all_predictions,
    axis=0
)

seed_std_prediction = np.std(
    all_predictions,
    axis=0,
    ddof=1
)

seed42_index = SEEDS.index(
    42
)

seed42_prediction = (
    all_predictions[
        seed42_index
    ]
)


daily_df = pd.DataFrame({

    "date":
        val_dates,

    "y_true":
        y_val,

    "seed42_prediction":
        seed42_prediction,

    "mean_seed_prediction":
        seed_mean_prediction,

    "seed_prediction_std":
        seed_std_prediction,

    "seed42_residual":
        y_val -
        seed42_prediction,

    "mean_seed_residual":
        y_val -
        seed_mean_prediction,
})


daily_path = (
    RESULTS_DIR /
    "04_03b_rnn_daily_predictions.csv"
)

daily_df.to_csv(
    daily_path,
    index=False
)


# ============================================================
# PART E
# Monthly validation diagnostics
# ============================================================

print()
print("=" * 80)
print("PART E - MONTHLY 2025 DIAGNOSTICS")
print("=" * 80)


daily_df["month"] = (
    daily_df["date"]
    .dt.to_period("M")
)


monthly_rows = []


for month, group in daily_df.groupby(
    "month"
):

    y_month = (
        group["y_true"]
        .to_numpy()
    )

    pred42 = (
        group[
            "seed42_prediction"
        ]
        .to_numpy()
    )

    pred_mean = (
        group[
            "mean_seed_prediction"
        ]
        .to_numpy()
    )

    metrics42 = calculate_metrics(
        y_month,
        pred42
    )

    metrics_mean = calculate_metrics(
        y_month,
        pred_mean
    )

    monthly_rows.append({

        "month":
            str(month),

        "n":
            len(group),

        "actual_mean":
            np.mean(
                y_month
            ),

        "seed42_prediction_mean":
            np.mean(
                pred42
            ),

        "seed42_rmse":
            metrics42["rmse"],

        "seed42_bias":
            metrics42["bias"],

        "mean_seed_prediction":
            np.mean(
                pred_mean
            ),

        "mean_seed_rmse":
            metrics_mean["rmse"],

        "mean_seed_bias":
            metrics_mean["bias"],

        "mean_prediction_std":
            group[
                "seed_prediction_std"
            ].mean(),
    })


monthly_df = pd.DataFrame(
    monthly_rows
)


for _, row in monthly_df.iterrows():

    print(
        f"{row['month']} | "
        f"Actual mean="
        f"{row['actual_mean']:.3f} | "
        f"Pred mean="
        f"{row['seed42_prediction_mean']:.3f} | "
        f"RMSE="
        f"{row['seed42_rmse']:.3f} | "
        f"Bias="
        f"{row['seed42_bias']:+.3f}"
    )


monthly_path = (
    RESULTS_DIR /
    "04_03b_rnn_monthly_diagnostics.csv"
)

monthly_df.to_csv(
    monthly_path,
    index=False
)


# ============================================================
# Seed-mean prediction diagnostic
#
# This is NOT selected as a forecasting model.
# It is used only to determine whether averaging random
# initializations reduces systematic bias.
# ============================================================

seed_mean_metrics = calculate_metrics(
    y_val,
    seed_mean_prediction
)


print()
print("=" * 80)
print(
    "MEAN PREDICTION ACROSS 10 SEEDS "
    "- DIAGNOSTIC ONLY"
)
print("=" * 80)

print(
    f"RMSE : "
    f"{seed_mean_metrics['rmse']:.6f}"
)

print(
    f"MAE  : "
    f"{seed_mean_metrics['mae']:.6f}"
)

print(
    f"Bias : "
    f"{seed_mean_metrics['bias']:+.6f}"
)


# ============================================================
# Plot 1
# Actual versus predicted monthly mean
# ============================================================

plt.figure(
    figsize=(11, 6)
)

x = np.arange(
    len(monthly_df)
)

plt.plot(
    x,
    monthly_df["actual_mean"],
    marker="o",
    label="Actual"
)

plt.plot(
    x,
    monthly_df[
        "seed42_prediction_mean"
    ],
    marker="o",
    label="RNN seed 42"
)

plt.plot(
    x,
    monthly_df[
        "mean_seed_prediction"
    ],
    marker="o",
    label="Mean prediction across 10 seeds"
)

plt.xticks(
    x,
    monthly_df["month"],
    rotation=45
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Mean daily claim count"
)

plt.title(
    "Vanilla RNN: "
    "2025 Monthly Actual vs Predicted Mean"
)

plt.legend()

plt.tight_layout()


monthly_plot_path = (
    RESULTS_DIR /
    "04_03b_rnn_monthly_actual_vs_predicted.png"
)

plt.savefig(
    monthly_plot_path,
    dpi=200
)

plt.close()


# ============================================================
# Plot 2
# Monthly bias
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.axhline(
    0,
    linewidth=1
)

plt.plot(
    x,
    monthly_df[
        "seed42_bias"
    ],
    marker="o",
    label="Seed 42 bias"
)

plt.plot(
    x,
    monthly_df[
        "mean_seed_bias"
    ],
    marker="o",
    label="Mean-seed bias"
)

plt.xticks(
    x,
    monthly_df["month"],
    rotation=45
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Bias = prediction - actual"
)

plt.title(
    "Vanilla RNN: "
    "2025 Monthly Forecast Bias"
)

plt.legend()

plt.tight_layout()


bias_plot_path = (
    RESULTS_DIR /
    "04_03b_rnn_monthly_bias.png"
)

plt.savefig(
    bias_plot_path,
    dpi=200
)

plt.close()


# ============================================================
# Final summary
# ============================================================

summary_path = (
    RESULTS_DIR /
    "04_03b_rnn_diagnostics_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 4.3B - VANILLA RNN DIAGNOSTICS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Target standardization: YES\n"
    )

    f.write(
        "Scaler fitted on training targets only.\n\n"
    )

    f.write(
        "Selected architecture inherited "
        "from Step 4.3A\n"
    )

    f.write(
        f"Feature set: "
        f"{FEATURE_SET}\n"
    )

    f.write(
        f"Window: "
        f"{WINDOW}\n"
    )

    f.write(
        f"Hidden size: "
        f"{HIDDEN_SIZE}\n"
    )

    f.write(
        f"Epochs: "
        f"{N_EPOCHS}\n"
    )

    f.write(
        f"CV RMSE: "
        f"{winner['cv_rmse_mean']:.6f} "
        f"+/- "
        f"{winner['cv_rmse_std']:.6f}\n\n"
    )

    f.write(
        "Target scaler\n"
    )

    f.write(
        f"Training mean: "
        f"{TARGET_MEAN:.6f}\n"
    )

    f.write(
        f"Training std: "
        f"{TARGET_STD:.6f}\n\n"
    )

    f.write(
        "Target level\n"
    )

    f.write(
        f"Training mean: "
        f"{train_mean:.6f}\n"
    )

    f.write(
        f"2025 mean: "
        f"{validation_mean:.6f}\n"
    )

    f.write(
        f"Relative shift: "
        f"{relative_shift:+.3f}%\n\n"
    )

    f.write(
        "Seed robustness\n"
    )

    f.write(
        f"RMSE mean +/- std: "
        f"{seed_df['rmse'].mean():.6f} "
        f"+/- "
        f"{seed_df['rmse'].std(ddof=1):.6f}\n"
    )

    f.write(
        f"RMSE min: "
        f"{seed_df['rmse'].min():.6f}\n"
    )

    f.write(
        f"RMSE max: "
        f"{seed_df['rmse'].max():.6f}\n"
    )

    f.write(
        f"Bias mean +/- std: "
        f"{seed_df['bias'].mean():+.6f} "
        f"+/- "
        f"{seed_df['bias'].std(ddof=1):.6f}\n\n"
    )

    f.write(
        "Mean-seed prediction diagnostic\n"
    )

    f.write(
        f"RMSE: "
        f"{seed_mean_metrics['rmse']:.6f}\n"
    )

    f.write(
        f"Bias: "
        f"{seed_mean_metrics['bias']:+.6f}\n"
    )

    if np.isfinite(
        step43a_rmse
    ):

        f.write(
            "\nSeed 42 reproduction\n"
        )

        f.write(
            f"Step 4.3A RMSE: "
            f"{step43a_rmse:.6f}\n"
        )

        f.write(
            f"Step 4.3B seed 42 RMSE: "
            f"{seed42_rmse:.6f}\n"
        )

        f.write(
            f"Absolute difference: "
            f"{abs(seed42_rmse - step43a_rmse):.8f}\n"
        )


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(fold_path)
print(seed_path)
print(daily_path)
print(monthly_path)
print(monthly_plot_path)
print(bias_plot_path)
print(summary_path)

print()
print("TEST SET WAS NOT USED.")