from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_SET = "F4"
WINDOW = 5
HIDDEN_SIZE = 1
DATA_FILE = RESULTS_DIR / f"04_01_windows_{FEATURE_SET}_W{WINDOW:02d}.npz"

SEEDS = [42, 101, 202, 505, 707]
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
FIXED_EPOCHS = 675
GRAD_CLIP = 1.0
DEVICE = torch.device("cpu")


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


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing temporal-window file: {DATA_FILE}")

    with np.load(DATA_FILE, allow_pickle=True) as data:
        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.float32).reshape(-1)
        X_val = np.asarray(data["X_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.float32).reshape(-1)

        val_dates = None
        for candidate in [
            "target_dates_val",
            "target_date_val",
            "dates_val",
            "date_val",
            "validation_dates",
            "val_dates",
        ]:
            if candidate in data.files:
                val_dates = np.asarray(data[candidate]).reshape(-1)
                break

    if X_train.shape[1] != WINDOW:
        raise ValueError(f"Expected W={WINDOW}, got {X_train.shape[1]}")
    if X_train.shape[2] != 5:
        raise ValueError(f"Expected F4 input dimension 5, got {X_train.shape[2]}")

    return X_train, y_train, X_val, y_val, val_dates


def fit_target_scaler(y_train):
    mean = float(np.mean(y_train))
    std = float(np.std(y_train, ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid target scaler.")
    return mean, std


def scale_target(y, mean, std):
    return ((y - mean) / std).astype(np.float32)


def inverse_scale_target(y_scaled, mean, std):
    return y_scaled * std + mean


class GateAffine(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weight_input = nn.Parameter(torch.empty(1, input_size))
        self.weight_hidden = nn.Parameter(torch.empty(1, 1))
        self.bias_input = nn.Parameter(torch.zeros(1))
        self.bias_hidden = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.weight_input)
        nn.init.orthogonal_(self.weight_hidden)

    def forward(self, x_t, h_prev):
        return (
            x_t @ self.weight_input.T
            + self.bias_input
            + h_prev @ self.weight_hidden.T
            + self.bias_hidden
        )


class CandidateAffine(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weight_input = nn.Parameter(torch.empty(1, input_size))
        self.weight_hidden = nn.Parameter(torch.empty(1, 1))
        self.bias_input = nn.Parameter(torch.zeros(1))
        self.bias_hidden = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.weight_input)
        nn.init.orthogonal_(self.weight_hidden)

    def forward(self, x_t, h_prev, reset_gate):
        input_term = x_t @ self.weight_input.T + self.bias_input
        recurrent_term = h_prev @ self.weight_hidden.T + self.bias_hidden
        return input_term + reset_gate * recurrent_term


class InspectableGRURegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.reset_affine = GateAffine(input_size)
        self.update_affine = GateAffine(input_size)
        self.candidate_affine = CandidateAffine(input_size)
        self.output = nn.Linear(1, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, return_internal=False):
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, 1, dtype=x.dtype, device=x.device)

        reset_values = []
        update_values = []
        candidate_values = []
        hidden_values = []

        for t in range(x.shape[1]):
            x_t = x[:, t, :]

            r_t = torch.sigmoid(self.reset_affine(x_t, h))
            z_t = torch.sigmoid(self.update_affine(x_t, h))

            n_t = torch.tanh(
                self.candidate_affine(
                    x_t,
                    h,
                    r_t
                )
            )

            h = (1.0 - z_t) * n_t + z_t * h

            if return_internal:
                reset_values.append(r_t)
                update_values.append(z_t)
                candidate_values.append(n_t)
                hidden_values.append(h)

        prediction_scaled = self.output(h).squeeze(-1)

        if not return_internal:
            return prediction_scaled

        return (
            prediction_scaled,
            torch.stack(reset_values, dim=1).squeeze(-1),
            torch.stack(update_values, dim=1).squeeze(-1),
            torch.stack(candidate_values, dim=1).squeeze(-1),
            torch.stack(hidden_values, dim=1).squeeze(-1),
        )


def train_model(X_train, y_train, seed):
    set_seed(seed)

    y_mean, y_std = fit_target_scaler(y_train)
    y_scaled = scale_target(y_train, y_mean, y_std)

    model = InspectableGRURegressor(
        input_size=X_train.shape[2]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_scaled, dtype=torch.float32),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    for _ in range(FIXED_EPOCHS):
        model.train()

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            pred_scaled = model(X_batch)
            loss = criterion(pred_scaled, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
            )

            optimizer.step()

    return model, y_mean, y_std


def inspect_validation(model, X_val, y_val, y_mean, y_std):
    model.eval()

    X_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        (
            pred_scaled,
            reset_values,
            update_values,
            candidate_values,
            hidden_values,
        ) = model(
            X_tensor,
            return_internal=True,
        )

    pred_scaled = pred_scaled.cpu().numpy()
    prediction = inverse_scale_target(
        pred_scaled,
        y_mean,
        y_std,
    )

    reset_values = reset_values.cpu().numpy()
    update_values = update_values.cpu().numpy()
    candidate_values = candidate_values.cpu().numpy()
    hidden_values = hidden_values.cpu().numpy()

    residual = prediction - y_val

    return {
        "prediction": prediction,
        "reset": reset_values,
        "update": update_values,
        "candidate": candidate_values,
        "hidden": hidden_values,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def main():
    print("=" * 110)
    print("STEP 4.4D - LEARNED GRU GATE INSPECTION")
    print("=" * 110)
    print()
    print(f"Architecture : {FEATURE_SET}, W={WINDOW}, H={HIDDEN_SIZE}")
    print(f"Epochs       : {FIXED_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"L2           : {WEIGHT_DECAY}")
    print(f"Seeds        : {SEEDS}")
    print()
    print("Target standardization: ENABLED")
    print("2025 validation: diagnostic inspection only")
    print("2026 TEST SET: NOT USED")

    X_train, y_train, X_val, y_val, val_dates = load_data()

    print()
    print(f"Training shape  : {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(
        f"Training target mean/std: "
        f"{np.mean(y_train):.6f} / "
        f"{np.std(y_train, ddof=0):.6f}"
    )

    long_rows = []
    seed_rows = []

    for seed in SEEDS:
        print()
        print("-" * 110)
        print(f"Seed {seed}")

        model, y_mean, y_std = train_model(
            X_train,
            y_train,
            seed,
        )

        result = inspect_validation(
            model,
            X_val,
            y_val,
            y_mean,
            y_std,
        )

        print(
            f"RMSE={result['rmse']:.6f} | "
            f"MAE={result['mae']:.6f} | "
            f"Bias={result['bias']:+.6f}"
        )
        print(
            f"Overall learned r: "
            f"mean={result['reset'].mean():.4f}, "
            f"std={result['reset'].std(ddof=0):.4f}"
        )
        print(
            f"Overall learned z: "
            f"mean={result['update'].mean():.4f}, "
            f"std={result['update'].std(ddof=0):.4f}"
        )

        for sample_idx in range(X_val.shape[0]):
            target_date = (
                str(val_dates[sample_idx])
                if val_dates is not None
                else ""
            )

            for t in range(WINDOW):
                long_rows.append({
                    "seed": seed,
                    "sample_index": sample_idx,
                    "target_date": target_date,
                    "time_step": t + 1,
                    "reset_gate_r": float(result["reset"][sample_idx, t]),
                    "update_gate_z": float(result["update"][sample_idx, t]),
                    "candidate_n": float(result["candidate"][sample_idx, t]),
                    "hidden_h": float(result["hidden"][sample_idx, t]),
                    "actual_target": float(y_val[sample_idx]),
                    "prediction": float(result["prediction"][sample_idx]),
                })

        r_desc = describe(result["reset"].reshape(-1))
        z_desc = describe(result["update"].reshape(-1))

        seed_rows.append({
            "seed": seed,
            "rmse": result["rmse"],
            "mae": result["mae"],
            "bias": result["bias"],
            "reset_mean": r_desc["mean"],
            "reset_std": r_desc["std"],
            "reset_median": r_desc["median"],
            "reset_q05": r_desc["q05"],
            "reset_q95": r_desc["q95"],
            "update_mean": z_desc["mean"],
            "update_std": z_desc["std"],
            "update_median": z_desc["median"],
            "update_q05": z_desc["q05"],
            "update_q95": z_desc["q95"],
        })

    long_df = pd.DataFrame(long_rows)
    seed_df = pd.DataFrame(seed_rows)

    raw_path = RESULTS_DIR / "04_04d_gru_learned_gate_values.csv"
    seed_path = RESULTS_DIR / "04_04d_gru_learned_gate_summary_by_seed.csv"

    long_df.to_csv(raw_path, index=False)
    seed_df.to_csv(seed_path, index=False)

    aggregate_rows = []

    for time_step, group in long_df.groupby("time_step"):
        r_desc = describe(group["reset_gate_r"].to_numpy())
        z_desc = describe(group["update_gate_z"].to_numpy())

        aggregate_rows.append({
            "time_step": int(time_step),

            "reset_mean": r_desc["mean"],
            "reset_std": r_desc["std"],
            "reset_q05": r_desc["q05"],
            "reset_q25": r_desc["q25"],
            "reset_median": r_desc["median"],
            "reset_q75": r_desc["q75"],
            "reset_q95": r_desc["q95"],

            "update_mean": z_desc["mean"],
            "update_std": z_desc["std"],
            "update_q05": z_desc["q05"],
            "update_q25": z_desc["q25"],
            "update_median": z_desc["median"],
            "update_q75": z_desc["q75"],
            "update_q95": z_desc["q95"],
        })

    aggregate_df = pd.DataFrame(aggregate_rows)

    aggregate_path = RESULTS_DIR / "04_04d_gru_learned_gate_aggregate_by_time.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    overall_r = describe(long_df["reset_gate_r"].to_numpy())
    overall_z = describe(long_df["update_gate_z"].to_numpy())

    summary_path = RESULTS_DIR / "04_04d_gru_learned_gate_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("STEP 4.4D - LEARNED GRU GATE INSPECTION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Architecture: {FEATURE_SET}, W={WINDOW}, H={HIDDEN_SIZE}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Epochs: {FIXED_EPOCHS}\n\n")

        f.write("OVERALL RESET GATE r\n")
        for key, value in overall_r.items():
            f.write(f"{key}: {value:.6f}\n")

        f.write("\nOVERALL UPDATE GATE z\n")
        for key, value in overall_z.items():
            f.write(f"{key}: {value:.6f}\n")

        f.write("\n\nAGGREGATE BY TIME STEP\n")
        f.write(aggregate_df.to_string(index=False))

        f.write("\n\n2026 TEST SET WAS NOT USED.\n")

    print()
    print("=" * 110)
    print("LEARNED GATE SUMMARY BY SEED")
    print("=" * 110)
    print(seed_df.to_string(index=False))

    print()
    print("=" * 110)
    print("LEARNED GATE SUMMARY BY TIME STEP - ALL FIVE SEEDS")
    print("=" * 110)
    print(aggregate_df.to_string(index=False))

    print()
    print("=" * 110)
    print("OVERALL LEARNED GATE DISTRIBUTION")
    print("=" * 110)
    print(
        f"Reset r:  mean={overall_r['mean']:.6f}, "
        f"std={overall_r['std']:.6f}, "
        f"median={overall_r['median']:.6f}, "
        f"q05={overall_r['q05']:.6f}, "
        f"q95={overall_r['q95']:.6f}"
    )
    print(
        f"Update z: mean={overall_z['mean']:.6f}, "
        f"std={overall_z['std']:.6f}, "
        f"median={overall_z['median']:.6f}, "
        f"q05={overall_z['q05']:.6f}, "
        f"q95={overall_z['q95']:.6f}"
    )

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)
    print(raw_path)
    print(seed_path)
    print(aggregate_path)
    print(summary_path)
    print()
    print("2026 TEST SET WAS NOT USED.")


if __name__ == "__main__":
    main()
