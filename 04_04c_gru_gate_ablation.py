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
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)
FEATURE_SET = 'F4'
WINDOW = 5
HIDDEN_SIZE = 1
DATA_FILE = RESULTS_DIR / f'04_01_windows_{FEATURE_SET}_W{WINDOW:02d}.npz'
SEEDS = [42, 101, 202, 505, 707]
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
FIXED_EPOCHS = 675
GRAD_CLIP = 1.0
DEVICE = torch.device('cpu')

VARIANTS = [
    dict(name='learned_r_learned_z', reset='learned', r=None, update='learned', z=None),
    dict(name='fixed_r_0_learned_z', reset='fixed', r=0.0, update='learned', z=None),
    dict(name='fixed_r_05_learned_z', reset='fixed', r=0.5, update='learned', z=None),
    dict(name='fixed_r_1_learned_z', reset='fixed', r=1.0, update='learned', z=None),
    dict(name='learned_r_fixed_z_0', reset='learned', r=None, update='fixed', z=0.0),
    dict(name='learned_r_fixed_z_05', reset='learned', r=None, update='fixed', z=0.5),
    dict(name='learned_r_fixed_z_1', reset='learned', r=None, update='fixed', z=1.0),
]


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
        raise FileNotFoundError(f'Missing file: {DATA_FILE}')
    with np.load(DATA_FILE, allow_pickle=True) as data:
        X_train = np.asarray(data['X_train'], dtype=np.float32)
        y_train = np.asarray(data['y_train'], dtype=np.float32).reshape(-1)
        X_val = np.asarray(data['X_val'], dtype=np.float32)
        y_val = np.asarray(data['y_val'], dtype=np.float32).reshape(-1)
    if X_train.shape[1] != WINDOW or X_train.shape[2] != 5:
        raise ValueError(f'Unexpected tensor shape: {X_train.shape}')
    return X_train, y_train, X_val, y_val


def fit_target_scaler(y):
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid target scaler')
    return mean, std


def scale_target(y, mean, std):
    return ((y - mean) / std).astype(np.float32)


def inverse_target(yz, mean, std):
    return yz * std + mean


class GateAffine(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.w_x = nn.Parameter(torch.empty(1, input_size))
        self.w_h = nn.Parameter(torch.empty(1, 1))
        self.b_x = nn.Parameter(torch.zeros(1))
        self.b_h = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.w_x)
        nn.init.orthogonal_(self.w_h)

    def forward(self, x, h):
        return x @ self.w_x.T + self.b_x + h @ self.w_h.T + self.b_h


class CandidateAffine(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.w_x = nn.Parameter(torch.empty(1, input_size))
        self.w_h = nn.Parameter(torch.empty(1, 1))
        self.b_x = nn.Parameter(torch.zeros(1))
        self.b_h = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.w_x)
        nn.init.orthogonal_(self.w_h)

    def forward(self, x, h, r):
        input_term = x @ self.w_x.T + self.b_x
        recurrent_term = h @ self.w_h.T + self.b_h
        return input_term + r * recurrent_term


class AblationGRU(nn.Module):
    def __init__(self, input_size, variant):
        super().__init__()
        self.variant = variant
        self.reset_affine = GateAffine(input_size) if variant['reset'] == 'learned' else None
        self.update_affine = GateAffine(input_size) if variant['update'] == 'learned' else None
        self.candidate_affine = CandidateAffine(input_size)
        self.output = nn.Linear(1, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def reset_gate(self, x, h):
        if self.variant['reset'] == 'learned':
            return torch.sigmoid(self.reset_affine(x, h))
        return torch.full_like(h, float(self.variant['r']))

    def update_gate(self, x, h):
        if self.variant['update'] == 'learned':
            return torch.sigmoid(self.update_affine(x, h))
        return torch.full_like(h, float(self.variant['z']))

    def forward(self, x, return_gates=False):
        h = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        rs, zs = [], []
        for t in range(x.shape[1]):
            xt = x[:, t, :]
            r = self.reset_gate(xt, h)
            z = self.update_gate(xt, h)
            n = torch.tanh(self.candidate_affine(xt, h, r))
            h = (1.0 - z) * n + z * h
            if return_gates:
                rs.append(r)
                zs.append(z)
        pred = self.output(h).squeeze(-1)
        if return_gates:
            return pred, torch.stack(rs, dim=1).squeeze(-1), torch.stack(zs, dim=1).squeeze(-1)
        return pred


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(X_train, y_train, variant, seed):
    set_seed(seed)
    mean, std = fit_target_scaler(y_train)
    y_scaled = scale_target(y_train, mean, std)
    model = AblationGRU(X_train.shape[2], variant).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    for _ in range(FIXED_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
    return model, mean, std


def evaluate(model, X_val, y_val, mean, std):
    model.eval()
    xt = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred_z, rvals, zvals = model(xt, return_gates=True)
    pred = inverse_target(pred_z.cpu().numpy(), mean, std)
    residual = pred - y_val
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    nrmse = float(rmse / np.std(y_val, ddof=0))
    return {
        'rmse': rmse,
        'mae': mae,
        'nrmse': nrmse,
        'bias': bias,
        'prediction_mean': float(np.mean(pred)),
        'reset_mean': float(rvals.mean().cpu()),
        'reset_std': float(rvals.std(unbiased=False).cpu()),
        'update_mean': float(zvals.mean().cpu()),
        'update_std': float(zvals.std(unbiased=False).cpu()),
    }


def main():
    print('=' * 110)
    print('STEP 4.4C - GRU GATE ABLATION')
    print('=' * 110)
    print(f'Architecture : {FEATURE_SET}, W={WINDOW}, H={HIDDEN_SIZE}')
    print(f'Epochs       : {FIXED_EPOCHS}')
    print(f'Learning rate: {LEARNING_RATE}')
    print(f'L2           : {WEIGHT_DECAY}')
    print(f'Batch size   : {BATCH_SIZE}')
    print(f'Seeds        : {SEEDS}')
    print('Target standardization: ENABLED')
    print('2025 validation: diagnostic only')
    print('2026 TEST SET: NOT USED')

    X_train, y_train, X_val, y_val = load_data()
    print(f'Training shape  : {X_train.shape}')
    print(f'Validation shape: {X_val.shape}')
    print(f'Training target mean/std: {np.mean(y_train):.6f} / {np.std(y_train, ddof=0):.6f}')

    rows = []
    for variant in VARIANTS:
        print('\n' + '=' * 110)
        print(f"VARIANT: {variant['name']}")
        print('=' * 110)
        for seed in SEEDS:
            model, mean, std = train_model(X_train, y_train, variant, seed)
            res = evaluate(model, X_val, y_val, mean, std)
            row = {
                'variant': variant['name'],
                'reset_mode': variant['reset'],
                'reset_value': variant['r'],
                'update_mode': variant['update'],
                'update_value': variant['z'],
                'seed': seed,
                'parameter_count': count_params(model),
                **res,
            }
            rows.append(row)
            print(
                f"Seed {seed:>3} | Params={row['parameter_count']:>2} | "
                f"RMSE={res['rmse']:.6f} | MAE={res['mae']:.6f} | "
                f"Bias={res['bias']:+.6f} | mean r={res['reset_mean']:.3f} | mean z={res['update_mean']:.3f}"
            )

    results = pd.DataFrame(rows)
    result_path = RESULTS_DIR / '04_04c_gru_gate_ablation_seed_results.csv'
    results.to_csv(result_path, index=False)

    baseline = results[results['variant'] == 'learned_r_learned_z'].set_index('seed')['rmse']
    summaries = []
    for name, g in results.groupby('variant', sort=False):
        deltas = [float(row.rmse - baseline.loc[int(row.seed)]) for row in g.itertuples()]
        summaries.append({
            'variant': name,
            'parameter_count': int(g['parameter_count'].iloc[0]),
            'rmse_mean': g['rmse'].mean(),
            'rmse_std': g['rmse'].std(ddof=1),
            'rmse_min': g['rmse'].min(),
            'rmse_max': g['rmse'].max(),
            'mae_mean': g['mae'].mean(),
            'bias_mean': g['bias'].mean(),
            'bias_std': g['bias'].std(ddof=1),
            'reset_mean': g['reset_mean'].mean(),
            'update_mean': g['update_mean'].mean(),
            'paired_rmse_delta_vs_learned': float(np.mean(deltas)),
        })

    summary = pd.DataFrame(summaries)
    summary_path = RESULTS_DIR / '04_04c_gru_gate_ablation_summary.csv'
    summary.to_csv(summary_path, index=False)

    print('\n' + '=' * 110)
    print('GRU GATE ABLATION SUMMARY')
    print('=' * 110)
    print(summary[[
        'variant', 'parameter_count', 'rmse_mean', 'rmse_std', 'bias_mean',
        'reset_mean', 'update_mean', 'paired_rmse_delta_vs_learned'
    ]].to_string(index=False))
    print('\npaired_rmse_delta_vs_learned: positive = worse than learned baseline; negative = better.')
    print('\nSAVED')
    print(result_path)
    print(summary_path)
    print('\n2026 TEST SET WAS NOT USED.')


if __name__ == '__main__':
    main()
