from pathlib import Path
import json
import math
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ============================================================
# Week 9, Step 9.1B.1
# Memory-only Y-Hamiltonian ablation with multi-probe robustness
# ============================================================

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DATA_FILE = RESULTS / "03_01_preprocessed_samples.csv"
J_WINNER_FILE = RESULTS / "08_03b_J_search_topology_winners.csv"

for path in [DATA_FILE, J_WINNER_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# ------------------------------------------------------------
# Frozen QRC parameters
# ------------------------------------------------------------
N_QUBITS = 6
N_INPUT = 4
D_INPUT = 16
D_MEMORY = 4
D_GLOBAL = 64

ALPHA = 0.75
HX = 0.5
DT = 1.6
TROTTER_R = 2

# Y ablation
Y_CONDITIONS = {
    "Y_OFF": 0.0,
    "Y_ON_hy_0p3": 0.3
}

# ------------------------------------------------------------
# Memory benchmark
# ------------------------------------------------------------
# 79001 is the established Week-7/8 probe seed and therefore
# remains the reproduction anchor. The other four seeds are
# additional robustness probes. They affect ONLY the temporal
# permutation of the artificial shuffled F4 benchmark.
PROBE_SEEDS = [79001, 42, 101, 505, 707]
ANCHOR_PROBE_SEED = 79001

K_MAX = 20
MEMORY_DELAYS = list(range(1, K_MAX + 1))
MIN_BURNIN = 100
POST_WASHOUT_MARGIN = 20
WASHOUT_EPS = 0.01
MAX_ACCEPTABLE_WASHOUT = 300
MC_RIDGE_ALPHA = 1e-6
VAR_TOL = 1e-12

# ------------------------------------------------------------
# Forecast benchmark
# ------------------------------------------------------------
FORECAST_RIDGE_ALPHA = 0.01
N_CV_SPLITS = 5

ACTIVE_TOPOLOGIES = ["H0", "H1", "H2", "H3", "H4"]

# Frozen Week-8 single-probe reference (seed 79001)
WEEK8_BASELINE = {
    "H0": {"mc_ch": 0.031152, "mc_total": 0.124610, "washout": 120},
    "H1": {"mc_ch": 0.031264, "mc_total": 0.125054, "washout": 217},
    "H2": {"mc_ch": 1.233298, "mc_total": 4.933193, "washout": 105},
    "H3": {"mc_ch": 1.145214, "mc_total": 4.580857, "washout": 102},
    "H4": {"mc_ch": 0.997772, "mc_total": 3.991090, "washout": 139},
}

# ============================================================
# Input encoding
# ============================================================
def claim_mapping(z):
    return float(np.clip(float(z) / 3.0, -1.0, 1.0))


def policy_mapping(z):
    z = float(z)
    return z / (3.0 + abs(z))


def weekday_mapping(d_sin, d_cos):
    return math.atan2(float(d_sin), float(d_cos)) % (2.0 * math.pi)


def holiday_mapping(h):
    h = int(h)
    if h not in (0, 1):
        raise ValueError(f"Holiday must be 0/1, got {h}")
    return math.pi * h


def encode_f4_row(row):
    return np.array(
        [
            ALPHA * claim_mapping(row["C_t_z"]),
            weekday_mapping(row["D_sin"], row["D_cos"]),
            ALPHA * policy_mapping(row["P_t_z"]),
            holiday_mapping(row["is_public_holiday_t_plus_1"]),
        ],
        dtype=float,
    )


# ============================================================
# Data
# ============================================================
df = pd.read_csv(DATA_FILE)

required_columns = [
    "input_date",
    "target_date",
    "split",
    "C_t_z",
    "D_sin",
    "D_cos",
    "P_t_z",
    "is_public_holiday_t_plus_1",
]
missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise KeyError(f"Missing preprocessing columns: {missing}\nAvailable: {list(df.columns)}")

df["input_date"] = pd.to_datetime(df["input_date"])
df["target_date"] = pd.to_datetime(df["target_date"])
df = df.sort_values("target_date").reset_index(drop=True)


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_target_column(frame):
    normalized = {normalize_name(c): c for c in frame.columns}
    candidates = [
        "target",
        "y",
        "C_t_plus_1",
        "claims_t_plus_1",
        "claim_count_t_plus_1",
        "property_damage_claim_count_t_plus_1",
        "target_claim_count",
        "target_property_damage",
        "target_property_damage_claim_count",
    ]
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            c = normalized[key]
            if pd.api.types.is_numeric_dtype(frame[c]):
                return c

    target_like = [
        c
        for c in frame.columns
        if "target" in normalize_name(c)
        and "date" not in normalize_name(c)
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if len(target_like) == 1:
        return target_like[0]

    raise KeyError(
        "Could not determine next-day property-damage target column.\n"
        f"Available columns: {list(frame.columns)}"
    )


TARGET_COLUMN = find_target_column(df)

split_norm = df["split"].astype(str).str.strip().str.lower()
train_mask_all = split_norm.eq("train")
validation_mask_all = split_norm.isin(["validation", "val"])

work_df = (
    df[train_mask_all | validation_mask_all]
    .copy()
    .reset_index(drop=True)
)

work_split = work_df["split"].astype(str).str.strip().str.lower()
train_mask = work_split.eq("train").to_numpy()
validation_mask = work_split.isin(["validation", "val"]).to_numpy()

N_TRAIN = int(train_mask.sum())
N_VALIDATION = int(validation_mask.sum())
N_TOTAL = len(work_df)

if (N_TRAIN, N_VALIDATION, N_TOTAL) != (1095, 365, 1460):
    raise RuntimeError(
        f"Expected train/validation/total = 1095/365/1460; got "
        f"{N_TRAIN}/{N_VALIDATION}/{N_TOTAL}"
    )

if not np.all(train_mask[:N_TRAIN]) or not np.all(validation_mask[N_TRAIN:]):
    raise RuntimeError("Train/validation chronology is not the expected contiguous split.")

Y_TARGET = work_df[TARGET_COLUMN].to_numpy(dtype=float)
REAL_ANGLES = np.vstack([encode_f4_row(row) for _, row in work_df.iterrows()])

# ============================================================
# Topologies and Week-8 J winners
# ============================================================
TOPOLOGY_EDGES = {
    "H0": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    "H1": [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)],
    "H2": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)],
    "H3": [(2, 3), (1, 2), (0, 1), (0, 4), (4, 5)],
    "H4": [(1, 2), (0, 1), (0, 4), (3, 4), (4, 5)],
}

EDGE_TO_COLUMN = {
    "H0": {(0, 1): "J01", (1, 2): "J12", (2, 3): "J23", (3, 4): "J34", (4, 5): "J45"},
    "H1": {(0, 1): "J01", (1, 2): "J12", (2, 3): "J23", (3, 4): "J34", (3, 5): "J35"},
    "H2": {(0, 1): "J01", (1, 2): "J12", (2, 3): "J23", (2, 4): "J24", (4, 5): "J45"},
    "H3": {(2, 3): "J23", (1, 2): "J12", (0, 1): "J01", (0, 4): "J04", (4, 5): "J45"},
    "H4": {(1, 2): "J12", (0, 1): "J01", (0, 4): "J04", (3, 4): "J34", (4, 5): "J45"},
}

j_df = pd.read_csv(J_WINNER_FILE)
COUPLINGS = {}

for topology in ACTIVE_TOPOLOGIES:
    matches = j_df[j_df["topology"] == topology]
    if len(matches) != 1:
        raise RuntimeError(f"{topology}: expected one Week-8 J winner; found {len(matches)}")

    winner = matches.iloc[0]
    if not np.isclose(float(winner["alpha"]), ALPHA):
        raise RuntimeError(f"{topology}: alpha mismatch")
    if not np.isclose(float(winner["hx"]), HX):
        raise RuntimeError(f"{topology}: hx mismatch")
    if not np.isclose(float(winner["dt"]), DT):
        raise RuntimeError(f"{topology}: dt mismatch")
    if int(winner["Trotter_r"]) != TROTTER_R:
        raise RuntimeError(f"{topology}: Trotter-r mismatch")

    COUPLINGS[topology] = {}
    for edge, column in EDGE_TO_COLUMN[topology].items():
        value = winner[column]
        if pd.isna(value):
            raise RuntimeError(f"{topology}: missing {column}")
        COUPLINGS[topology][edge] = float(value)

# ============================================================
# Operators and unitary
# ============================================================
I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
I64 = np.eye(D_GLOBAL, dtype=complex)


def operator_on_qubit(op, q):
    result = np.array([[1.0 + 0j]], dtype=complex)
    for logical_q in reversed(range(N_QUBITS)):
        result = np.kron(result, op if logical_q == q else I2)
    return result


X_OPS = [operator_on_qubit(X2, q) for q in range(N_QUBITS)]
Y_OPS = [operator_on_qubit(Y2, q) for q in range(N_QUBITS)]
Z_OPS = [operator_on_qubit(Z2, q) for q in range(N_QUBITS)]

ZZ_OPS = {}
for topology in ACTIVE_TOPOLOGIES:
    for edge in TOPOLOGY_EDGES[topology]:
        canonical = tuple(sorted(edge))
        if canonical not in ZZ_OPS:
            ZZ_OPS[canonical] = Z_OPS[canonical[0]] @ Z_OPS[canonical[1]]


def pauli_exponential(op, angle):
    return math.cos(angle) * I64 - 1j * math.sin(angle) * op


def build_unitary(topology, hy):
    U = np.eye(D_GLOBAL, dtype=complex)

    for _ in range(TROTTER_R):
        # ZZ sector
        for edge in TOPOLOGY_EDGES[topology]:
            Jij = COUPLINGS[topology][edge]
            a_zz = Jij * DT / TROTTER_R
            G = pauli_exponential(ZZ_OPS[tuple(sorted(edge))], a_zz)
            U = G @ U

        # X field on all qubits
        a_x = HX * DT / TROTTER_R
        for q in range(N_QUBITS):
            U = pauli_exponential(X_OPS[q], a_x) @ U

        # Optional memory-only Y field on q4,q5
        if abs(hy) > 0.0:
            a_y = hy * DT / TROTTER_R
            for q in (4, 5):
                U = pauli_exponential(Y_OPS[q], a_y) @ U

    error = float(np.linalg.norm(U.conj().T @ U - I64, ord="fro"))
    return U, error

# ============================================================
# Input and memory states
# ============================================================
def ry_zero_state(theta):
    return np.array([math.cos(theta / 2.0), math.sin(theta / 2.0)], dtype=complex)


def input_state_vector(angles):
    q0 = ry_zero_state(angles[0])
    q1 = ry_zero_state(angles[1])
    q2 = ry_zero_state(angles[2])
    q3 = ry_zero_state(angles[3])
    return np.kron(np.kron(np.kron(q3, q2), q1), q0)


REAL_PHI = [input_state_vector(a) for a in REAL_ANGLES]
I4 = np.eye(D_MEMORY, dtype=complex)


def density(vector):
    v = np.asarray(vector, dtype=complex)
    return np.outer(v, v.conj())


KET_00 = np.array([1, 0, 0, 0], dtype=complex)
KET_11 = np.array([0, 0, 0, 1], dtype=complex)
KET_PP = np.ones(4, dtype=complex) / 2.0
KET_BELL = np.array([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], dtype=complex)

RHO_00 = density(KET_00)
INITIAL_MEMORY_STATES = {
    "00": density(KET_00),
    "11": density(KET_11),
    "++": density(KET_PP),
    "BellPhi+": density(KET_BELL),
    "I/4": np.eye(4, dtype=complex) / 4.0,
}

# ============================================================
# Randomized F4 probe
# ============================================================
def build_probe(real_angles, seed):
    rng = np.random.default_rng(seed)
    real_angles = np.asarray(real_angles, dtype=float)
    probe = np.empty_like(real_angles)
    for j in range(real_angles.shape[1]):
        probe[:, j] = real_angles[rng.permutation(len(real_angles)), j]
    return probe


def autocorrelation(x, lag):
    a = np.asarray(x[:-lag], dtype=float)
    b = np.asarray(x[lag:], dtype=float)
    if np.std(a) <= VAR_TOL or np.std(b) <= VAR_TOL:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def probe_audit(real_angles, probe):
    marginal_error = 0.0
    max_abs_ac = 0.0

    for j in range(real_angles.shape[1]):
        marginal_error = max(
            marginal_error,
            float(np.max(np.abs(np.sort(real_angles[:, j]) - np.sort(probe[:, j])))),
        )
        for lag in range(1, K_MAX + 1):
            r = autocorrelation(probe[:, j], lag)
            if np.isfinite(r):
                max_abs_ac = max(max_abs_ac, abs(r))

    return marginal_error, max_abs_ac


PROBE_CACHE = {}
for seed in PROBE_SEEDS:
    angles = build_probe(REAL_ANGLES, seed)
    marginal_error, max_abs_ac = probe_audit(REAL_ANGLES, angles)
    PROBE_CACHE[seed] = {
        "angles": angles,
        "phi": [input_state_vector(a) for a in angles],
        "marginal_error": marginal_error,
        "max_abs_autocorr": max_abs_ac,
    }

# ============================================================
# Recurrent propagation and features
# ============================================================
def isometry_for_input(U, phi):
    embedding = np.kron(I4, np.asarray(phi, dtype=complex).reshape(-1, 1))
    return U @ embedding


def partial_trace_input(rho_global):
    reshaped = rho_global.reshape(D_MEMORY, D_INPUT, D_MEMORY, D_INPUT)
    rho_memory = np.einsum("aibi->ab", reshaped)
    rho_memory = 0.5 * (rho_memory + rho_memory.conj().T)
    rho_memory /= np.trace(rho_memory)
    return rho_memory


BASIS_INDICES = np.arange(D_GLOBAL)


def xyz_expectations(rho_global):
    diagonal = np.real(np.diag(rho_global))
    x_values = np.zeros(N_QUBITS, dtype=float)
    y_values = np.zeros(N_QUBITS, dtype=float)
    z_values = np.zeros(N_QUBITS, dtype=float)

    for q in range(N_QUBITS):
        mask = 1 << q
        flipped = BASIS_INDICES ^ mask
        offdiag = rho_global[BASIS_INDICES, flipped]

        x_values[q] = float(np.real(np.sum(offdiag)))

        bits = (BASIS_INDICES >> q) & 1
        y_coeff = np.where(bits == 0, 1j, -1j)
        y_values[q] = float(np.real(np.sum(offdiag * y_coeff)))

        z_sign = 1.0 - 2.0 * bits
        z_values[q] = float(np.sum(diagonal * z_sign))

    return x_values, y_values, z_values


def simulate_cont(U, phi_sequence):
    n = len(phi_sequence)
    xyz_features = np.zeros((n, 18), dtype=float)
    xzinj_features = np.zeros((n, 8), dtype=float)

    rho_memory = RHO_00.copy()
    max_trace_error = 0.0
    max_hermiticity_error = 0.0
    min_memory_eigenvalue = float("inf")

    for t, phi in enumerate(phi_sequence):
        V = isometry_for_input(U, phi)
        rho_global = V @ rho_memory @ V.conj().T

        x_values, y_values, z_values = xyz_expectations(rho_global)
        xyz_features[t] = np.concatenate([x_values, y_values, z_values])
        xzinj_features[t] = np.concatenate([x_values[:4], z_values[:4]])

        rho_memory = partial_trace_input(rho_global)

        max_trace_error = max(
            max_trace_error,
            float(abs(np.trace(rho_memory) - 1.0)),
        )
        max_hermiticity_error = max(
            max_hermiticity_error,
            float(np.linalg.norm(rho_memory - rho_memory.conj().T, ord="fro")),
        )
        min_memory_eigenvalue = min(
            min_memory_eigenvalue,
            float(np.min(np.linalg.eigvalsh(rho_memory))),
        )

    diagnostics = {
        "max_trace_error": max_trace_error,
        "max_hermiticity_error": max_hermiticity_error,
        "min_memory_eigenvalue": min_memory_eigenvalue,
        "final_memory_purity": float(np.real(np.trace(rho_memory @ rho_memory))),
    }

    return xyz_features, xzinj_features, diagnostics

# ============================================================
# Washout
# ============================================================
def memory_kraus(U, phi):
    V = isometry_for_input(U, phi)
    V_reshaped = V.reshape(D_MEMORY, D_INPUT, D_MEMORY)
    return [V_reshaped[:, i, :] for i in range(D_INPUT)]


def propagate_memory_kraus(rho, kraus_ops):
    output = np.zeros((D_MEMORY, D_MEMORY), dtype=complex)
    for K in kraus_ops:
        output += K @ rho @ K.conj().T
    output = 0.5 * (output + output.conj().T)
    output /= np.trace(output)
    return output


def trace_distance(rho, sigma):
    delta = 0.5 * ((rho - sigma) + (rho - sigma).conj().T)
    vals = np.linalg.eigvalsh(delta)
    return float(0.5 * np.sum(np.abs(vals)))


def max_pairwise_trace_distance(states):
    names = list(states.keys())
    maximum = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            maximum = max(
                maximum,
                trace_distance(states[names[i]], states[names[j]]),
            )
    return maximum


def stable_threshold_crossing(values, threshold):
    arr = np.asarray(values, dtype=float)
    suffix_max = np.maximum.accumulate(arr[::-1])[::-1]
    idx = np.flatnonzero(suffix_max <= threshold)
    return None if len(idx) == 0 else int(idx[0])


def washout_analysis(U, phi_sequence):
    states = {name: rho.copy() for name, rho in INITIAL_MEMORY_STATES.items()}

    # IMPORTANT: match the established Week-7/8 convention.
    # D_curve[0] is the distance AFTER the first driven step;
    # the pre-input initial distance is not inserted.
    distances = []

    for phi in phi_sequence:
        kraus_ops = memory_kraus(U, phi)
        states = {
            name: propagate_memory_kraus(rho, kraus_ops)
            for name, rho in states.items()
        }
        distances.append(max_pairwise_trace_distance(states))

    distances = np.asarray(distances, dtype=float)
    Tw = stable_threshold_crossing(distances, WASHOUT_EPS)
    return Tw, distances

# ============================================================
# Ridge helpers
# ============================================================
def fit_feature_transform(X_train):
    X_train = np.asarray(X_train, dtype=float)
    std = np.std(X_train, axis=0, ddof=0)
    keep = std > VAR_TOL
    if not np.any(keep):
        return None, None, None
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[:, keep])
    return keep, scaler, X_train_scaled


def safe_corr2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) <= VAR_TOL or np.std(b) <= VAR_TOL:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(r):
        return 0.0
    return float(r * r)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# ============================================================
# Memory capacity
# ============================================================
NULL_FLOOR = 1.0 / (N_VALIDATION - 1.0)


def evaluate_memory(X_xyz, probe_angles, burnin):
    X_xyz = np.asarray(X_xyz, dtype=float)
    probe_angles = np.asarray(probe_angles, dtype=float)

    if burnin >= N_TRAIN - K_MAX - 20:
        raise RuntimeError(
            f"Burn-in {burnin} leaves too little training data for MC evaluation."
        )

    rows = []

    for channel in range(probe_angles.shape[1]):
        for delay in MEMORY_DELAYS:
            start = max(int(burnin), delay)
            tr = np.arange(start, N_TRAIN)
            va = np.arange(N_TRAIN, N_TOTAL)

            ytr = probe_angles[tr - delay, channel]
            yva = probe_angles[va - delay, channel]

            keep, scaler, Xtr_z = fit_feature_transform(X_xyz[tr])

            if keep is None:
                pred = np.full(len(va), float(np.mean(ytr)))
                n_features = 0
            else:
                Xva_z = scaler.transform(X_xyz[va][:, keep])
                model = Ridge(alpha=MC_RIDGE_ALPHA, fit_intercept=True)
                model.fit(Xtr_z, ytr)
                pred = model.predict(Xva_z)
                n_features = int(keep.sum())

            mc_raw = safe_corr2(yva, pred)
            mc_corrected = max(mc_raw - NULL_FLOOR, 0.0)

            rows.append(
                {
                    "channel": channel,
                    "delay": delay,
                    "MC_raw": mc_raw,
                    "MC_corrected": mc_corrected,
                    "n_features_retained": n_features,
                }
            )

    detail = pd.DataFrame(rows)
    total_raw = float(detail["MC_raw"].sum())
    total_corrected = float(detail["MC_corrected"].sum())
    n_channels = probe_angles.shape[1]

    def delay_mean(delay, column):
        return float(detail.loc[detail["delay"] == delay, column].mean())

    return {
        "MC_total_raw": total_raw,
        "MC_total_corrected": total_corrected,
        "MC_per_channel_raw": total_raw / n_channels,
        "MC_per_channel_corrected": total_corrected / n_channels,
        "MC_delay1_raw_mean": delay_mean(1, "MC_raw"),
        "MC_delay2_raw_mean": delay_mean(2, "MC_raw"),
        "MC_delay5_raw_mean": delay_mean(5, "MC_raw"),
        "MC_delay10_raw_mean": delay_mean(10, "MC_raw"),
        "MC_delay20_raw_mean": delay_mean(20, "MC_raw"),
        "MC_delay1_corrected_mean": delay_mean(1, "MC_corrected"),
        "MC_delay2_corrected_mean": delay_mean(2, "MC_corrected"),
        "MC_delay5_corrected_mean": delay_mean(5, "MC_corrected"),
        "MC_delay10_corrected_mean": delay_mean(10, "MC_corrected"),
        "MC_delay20_corrected_mean": delay_mean(20, "MC_corrected"),
        "detail": detail,
    }

# ============================================================
# Forecasting on real chronology (no probe seed involved)
# ============================================================
def forecasting_evaluation(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_train_all = X[:N_TRAIN]
    y_train_all = y[:N_TRAIN]
    X_val = X[N_TRAIN:]
    y_val = y[N_TRAIN:]

    splitter = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    fold_rmse = []

    for tr, va in splitter.split(X_train_all):
        keep, scaler, Xtr_z = fit_feature_transform(X_train_all[tr])

        if keep is None:
            pred = np.full(len(va), float(np.mean(y_train_all[tr])))
        else:
            Xva_z = scaler.transform(X_train_all[va][:, keep])
            model = Ridge(alpha=FORECAST_RIDGE_ALPHA, fit_intercept=True)
            model.fit(Xtr_z, y_train_all[tr])
            pred = model.predict(Xva_z)

        fold_rmse.append(rmse(y_train_all[va], pred))

    keep, scaler, Xtr_z = fit_feature_transform(X_train_all)
    if keep is None:
        validation_pred = np.full(len(X_val), float(np.mean(y_train_all)))
        retained = 0
    else:
        Xval_z = scaler.transform(X_val[:, keep])
        final_model = Ridge(alpha=FORECAST_RIDGE_ALPHA, fit_intercept=True)
        final_model.fit(Xtr_z, y_train_all)
        validation_pred = final_model.predict(Xval_z)
        retained = int(keep.sum())

    return {
        "cv_rmse_mean": float(np.mean(fold_rmse)),
        "cv_rmse_std": float(np.std(fold_rmse, ddof=1)),
        "cv_fold_rmse": [float(x) for x in fold_rmse],
        "validation_rmse": rmse(y_val, validation_pred),
        "validation_bias": float(np.mean(validation_pred - y_val)),
        "n_features_retained": retained,
        "n_features_original": int(X.shape[1]),
    }

# ============================================================
# Main experiment
# ============================================================
print("=" * 110)
print("WEEK 9 — STEP 9.1B.1")
print("MEMORY-ONLY Y-HAMILTONIAN ABLATION — 5 PAIRED PROBE SEEDS")
print("=" * 110)
print(f"Target column: {TARGET_COLUMN}")
print(f"Probe seeds: {PROBE_SEEDS}")
print("Primary MC burn-in: B = max(100, Tw(0.01) + 20), computed per topology/Y/seed")
print("Forecasting uses the real chronology once per topology/Y condition; probe seeds do not affect RMSE.")

print("\nProbe audit:")
for seed in PROBE_SEEDS:
    print(
        f"  seed={seed}: marginal_error={PROBE_CACHE[seed]['marginal_error']:.3e}, "
        f"max|AC|_lag1..20={PROBE_CACHE[seed]['max_abs_autocorr']:.9f}"
    )

memory_seed_rows = []
memory_delay_rows = []
forecast_rows = []
diagnostic_rows = []
washout_rows = []
unitary_rows = []

experiment_start = time.time()

for topology in ACTIVE_TOPOLOGIES:
    print("\n" + "-" * 110)
    print(topology)
    print("-" * 110)

    for y_condition, hy in Y_CONDITIONS.items():
        condition_start = time.time()
        U, unitary_error = build_unitary(topology, hy)

        unitary_rows.append(
            {
                "topology": topology,
                "y_condition": y_condition,
                "hy": hy,
                "unitary_error_fro": unitary_error,
            }
        )

        # ----------------------------------------------------
        # Forecast features on the real chronology — once only
        # ----------------------------------------------------
        real_xyz, real_xzinj, real_diag = simulate_cont(U, REAL_PHI)

        for readout, X in [("XZ_injection", real_xzinj), ("XYZ_all", real_xyz)]:
            result = forecasting_evaluation(X, Y_TARGET)
            forecast_rows.append(
                {
                    "topology": topology,
                    "y_condition": y_condition,
                    "hy": hy,
                    "readout": readout,
                    "ridge_alpha": FORECAST_RIDGE_ALPHA,
                    "cv_rmse_mean": result["cv_rmse_mean"],
                    "cv_rmse_std": result["cv_rmse_std"],
                    "validation_rmse": result["validation_rmse"],
                    "validation_bias": result["validation_bias"],
                    "n_features_retained": result["n_features_retained"],
                    "n_features_original": result["n_features_original"],
                    "fold_rmse": json.dumps(result["cv_fold_rmse"]),
                }
            )

        # ----------------------------------------------------
        # Intrinsic memory over five paired probe seeds
        # ----------------------------------------------------
        for probe_seed in PROBE_SEEDS:
            probe_angles = PROBE_CACHE[probe_seed]["angles"]
            probe_phi = PROBE_CACHE[probe_seed]["phi"]

            probe_xyz, _, probe_diag = simulate_cont(U, probe_phi)
            Tw, distances = washout_analysis(U, probe_phi)

            washout_valid = (
                Tw is not None
                and Tw <= MAX_ACCEPTABLE_WASHOUT
            )

            if Tw is None:
                postwash_burnin = None
                mcpost = None
            else:
                postwash_burnin = max(
                    MIN_BURNIN,
                    int(Tw) + POST_WASHOUT_MARGIN,
                )
                mcpost = evaluate_memory(
                    probe_xyz,
                    probe_angles,
                    postwash_burnin,
                )

            # Also retain burn-in-100 as a diagnostic only.
            mc100 = evaluate_memory(
                probe_xyz,
                probe_angles,
                MIN_BURNIN,
            )

            row = {
                "topology": topology,
                "y_condition": y_condition,
                "hy": hy,
                "probe_seed": probe_seed,
                "is_week8_anchor_seed": probe_seed == ANCHOR_PROBE_SEED,
                "probe_marginal_error": PROBE_CACHE[probe_seed]["marginal_error"],
                "probe_max_abs_autocorr_lag1_20": PROBE_CACHE[probe_seed]["max_abs_autocorr"],
                "Tw_0p01": np.nan if Tw is None else int(Tw),
                "washout_valid_le_300": bool(washout_valid),
                "postwash_burnin": np.nan if postwash_burnin is None else int(postwash_burnin),
                "MC_ch_corrected_burnin100": mc100["MC_per_channel_corrected"],
                "MC_total_corrected_burnin100": mc100["MC_total_corrected"],
                "unitary_error_fro": unitary_error,
                "probe_max_trace_error": probe_diag["max_trace_error"],
                "probe_max_hermiticity_error": probe_diag["max_hermiticity_error"],
                "probe_min_memory_eigenvalue": probe_diag["min_memory_eigenvalue"],
            }

            if mcpost is None:
                row.update(
                    {
                        "MC_ch_raw_postwashout": np.nan,
                        "MC_ch_corrected_postwashout": np.nan,
                        "MC_total_raw_postwashout": np.nan,
                        "MC_total_corrected_postwashout": np.nan,
                        "MC1_raw": np.nan,
                        "MC2_raw": np.nan,
                        "MC5_raw": np.nan,
                        "MC10_raw": np.nan,
                        "MC20_raw": np.nan,
                        "MC1_corrected": np.nan,
                        "MC2_corrected": np.nan,
                        "MC5_corrected": np.nan,
                        "MC10_corrected": np.nan,
                        "MC20_corrected": np.nan,
                    }
                )
            else:
                row.update(
                    {
                        "MC_ch_raw_postwashout": mcpost["MC_per_channel_raw"],
                        "MC_ch_corrected_postwashout": mcpost["MC_per_channel_corrected"],
                        "MC_total_raw_postwashout": mcpost["MC_total_raw"],
                        "MC_total_corrected_postwashout": mcpost["MC_total_corrected"],
                        "MC1_raw": mcpost["MC_delay1_raw_mean"],
                        "MC2_raw": mcpost["MC_delay2_raw_mean"],
                        "MC5_raw": mcpost["MC_delay5_raw_mean"],
                        "MC10_raw": mcpost["MC_delay10_raw_mean"],
                        "MC20_raw": mcpost["MC_delay20_raw_mean"],
                        "MC1_corrected": mcpost["MC_delay1_corrected_mean"],
                        "MC2_corrected": mcpost["MC_delay2_corrected_mean"],
                        "MC5_corrected": mcpost["MC_delay5_corrected_mean"],
                        "MC10_corrected": mcpost["MC_delay10_corrected_mean"],
                        "MC20_corrected": mcpost["MC_delay20_corrected_mean"],
                    }
                )

                d = mcpost["detail"].copy()
                d["topology"] = topology
                d["y_condition"] = y_condition
                d["hy"] = hy
                d["probe_seed"] = probe_seed
                d["burnin"] = postwash_burnin
                memory_delay_rows.append(d)

            memory_seed_rows.append(row)

            for step, distance in enumerate(distances):
                washout_rows.append(
                    {
                        "topology": topology,
                        "y_condition": y_condition,
                        "hy": hy,
                        "probe_seed": probe_seed,
                        "step": step,
                        "max_pairwise_trace_distance": float(distance),
                    }
                )

            diagnostic_rows.append(
                {
                    "topology": topology,
                    "y_condition": y_condition,
                    "hy": hy,
                    "probe_seed": probe_seed,
                    "real_max_trace_error": real_diag["max_trace_error"],
                    "real_max_hermiticity_error": real_diag["max_hermiticity_error"],
                    "real_min_memory_eigenvalue": real_diag["min_memory_eigenvalue"],
                    "probe_max_trace_error": probe_diag["max_trace_error"],
                    "probe_max_hermiticity_error": probe_diag["max_hermiticity_error"],
                    "probe_min_memory_eigenvalue": probe_diag["min_memory_eigenvalue"],
                }
            )

        elapsed = time.time() - condition_start
        condition_seed_rows = [
            r
            for r in memory_seed_rows
            if r["topology"] == topology and r["y_condition"] == y_condition
        ]
        seed_values = np.array(
            [r["MC_ch_corrected_postwashout"] for r in condition_seed_rows],
            dtype=float,
        )
        print(
            f"{y_condition:13s} hy={hy:.3f} | "
            f"MC/ch mean={np.nanmean(seed_values):.6f} "
            f"sd={np.nanstd(seed_values, ddof=1):.6f} | "
            f"time={elapsed:.1f}s"
        )

# ============================================================
# Result frames
# ============================================================
memory_seed_df = pd.DataFrame(memory_seed_rows)
forecast_df = pd.DataFrame(forecast_rows)
diagnostics_df = pd.DataFrame(diagnostic_rows)
washout_df = pd.DataFrame(washout_rows)
unitary_df = pd.DataFrame(unitary_rows)

if memory_delay_rows:
    memory_delay_df = pd.concat(memory_delay_rows, ignore_index=True)
else:
    memory_delay_df = pd.DataFrame()

# ============================================================
# Week-8 anchor-seed reproduction audit
# ============================================================
audit_rows = []
for topology in ACTIVE_TOPOLOGIES:
    observed = memory_seed_df[
        (memory_seed_df["topology"] == topology)
        & (memory_seed_df["y_condition"] == "Y_OFF")
        & (memory_seed_df["probe_seed"] == ANCHOR_PROBE_SEED)
    ]
    if len(observed) != 1:
        raise RuntimeError(f"{topology}: missing unique anchor-seed Y_OFF row")

    observed = observed.iloc[0]
    expected = WEEK8_BASELINE[topology]

    audit_rows.append(
        {
            "topology": topology,
            "probe_seed": ANCHOR_PROBE_SEED,
            "expected_mc_ch": expected["mc_ch"],
            "observed_mc_ch": observed["MC_ch_corrected_postwashout"],
            "mc_ch_error": observed["MC_ch_corrected_postwashout"] - expected["mc_ch"],
            "expected_mc_total": expected["mc_total"],
            "observed_mc_total": observed["MC_total_corrected_postwashout"],
            "mc_total_error": observed["MC_total_corrected_postwashout"] - expected["mc_total"],
            "expected_washout": expected["washout"],
            "observed_washout": observed["Tw_0p01"],
            "washout_error": observed["Tw_0p01"] - expected["washout"],
            "postwash_burnin": observed["postwash_burnin"],
        }
    )

audit_df = pd.DataFrame(audit_rows)

# ============================================================
# Multi-seed aggregates
# ============================================================
aggregate_rows = []
for topology in ACTIVE_TOPOLOGIES:
    for y_condition, hy in Y_CONDITIONS.items():
        subset = memory_seed_df[
            (memory_seed_df["topology"] == topology)
            & (memory_seed_df["y_condition"] == y_condition)
        ].copy()

        mc = subset["MC_ch_corrected_postwashout"].to_numpy(dtype=float)
        tw = subset["Tw_0p01"].to_numpy(dtype=float)
        burnin = subset["postwash_burnin"].to_numpy(dtype=float)

        aggregate_rows.append(
            {
                "topology": topology,
                "y_condition": y_condition,
                "hy": hy,
                "n_probe_seeds": len(subset),
                "n_valid_mc": int(np.isfinite(mc).sum()),
                "MC_ch_mean": float(np.nanmean(mc)),
                "MC_ch_std": float(np.nanstd(mc, ddof=1)),
                "MC_ch_median": float(np.nanmedian(mc)),
                "MC_ch_min": float(np.nanmin(mc)),
                "MC_ch_max": float(np.nanmax(mc)),
                "Tw_mean": float(np.nanmean(tw)),
                "Tw_std": float(np.nanstd(tw, ddof=1)),
                "Tw_median": float(np.nanmedian(tw)),
                "burnin_mean": float(np.nanmean(burnin)),
                "burnin_median": float(np.nanmedian(burnin)),
                "valid_washout_seeds": int(subset["washout_valid_le_300"].sum()),
            }
        )

aggregate_df = pd.DataFrame(aggregate_rows)

# ============================================================
# Paired Y effect across the SAME five probe seeds
# ============================================================
paired_rows = []
paired_seed_rows = []

for topology in ACTIVE_TOPOLOGIES:
    off = memory_seed_df[
        (memory_seed_df["topology"] == topology)
        & (memory_seed_df["y_condition"] == "Y_OFF")
    ][["probe_seed", "MC_ch_corrected_postwashout", "Tw_0p01", "postwash_burnin"]].rename(
        columns={
            "MC_ch_corrected_postwashout": "mc_off",
            "Tw_0p01": "tw_off",
            "postwash_burnin": "burnin_off",
        }
    )

    on = memory_seed_df[
        (memory_seed_df["topology"] == topology)
        & (memory_seed_df["y_condition"] == "Y_ON_hy_0p3")
    ][["probe_seed", "MC_ch_corrected_postwashout", "Tw_0p01", "postwash_burnin"]].rename(
        columns={
            "MC_ch_corrected_postwashout": "mc_on",
            "Tw_0p01": "tw_on",
            "postwash_burnin": "burnin_on",
        }
    )

    paired = off.merge(on, on="probe_seed", how="inner").sort_values("probe_seed")
    paired["delta_mc_ch"] = paired["mc_on"] - paired["mc_off"]
    paired["delta_Tw"] = paired["tw_on"] - paired["tw_off"]
    paired["topology"] = topology
    paired_seed_rows.append(paired)

    delta = paired["delta_mc_ch"].to_numpy(dtype=float)

    paired_rows.append(
        {
            "topology": topology,
            "n_pairs": len(paired),
            "delta_mc_ch_mean": float(np.nanmean(delta)),
            "delta_mc_ch_std": float(np.nanstd(delta, ddof=1)),
            "delta_mc_ch_median": float(np.nanmedian(delta)),
            "delta_mc_ch_min": float(np.nanmin(delta)),
            "delta_mc_ch_max": float(np.nanmax(delta)),
            "n_seeds_y_improves_mc": int(np.sum(delta > 0)),
            "n_seeds_y_worsens_mc": int(np.sum(delta < 0)),
            "n_seeds_equal": int(np.sum(np.isclose(delta, 0.0))),
            "mean_relative_change_percent": float(
                np.nanmean(100.0 * delta / paired["mc_off"].to_numpy(dtype=float))
            ),
            "mean_delta_Tw": float(np.nanmean(paired["delta_Tw"].to_numpy(dtype=float))),
        }
    )

paired_effect_df = pd.DataFrame(paired_rows)
paired_seed_df = pd.concat(paired_seed_rows, ignore_index=True)

# ============================================================
# Paired forecasting Y effect (deterministic, no probe seed)
# ============================================================
forecast_compare_rows = []
for topology in ACTIVE_TOPOLOGIES:
    for readout in ["XZ_injection", "XYZ_all"]:
        off = forecast_df[
            (forecast_df["topology"] == topology)
            & (forecast_df["readout"] == readout)
            & (forecast_df["y_condition"] == "Y_OFF")
        ].iloc[0]
        on = forecast_df[
            (forecast_df["topology"] == topology)
            & (forecast_df["readout"] == readout)
            & (forecast_df["y_condition"] == "Y_ON_hy_0p3")
        ].iloc[0]

        forecast_compare_rows.append(
            {
                "topology": topology,
                "readout": readout,
                "cv_rmse_off": off["cv_rmse_mean"],
                "cv_rmse_on": on["cv_rmse_mean"],
                "delta_cv_rmse": on["cv_rmse_mean"] - off["cv_rmse_mean"],
                "cv_change_percent": 100.0
                * (on["cv_rmse_mean"] - off["cv_rmse_mean"])
                / off["cv_rmse_mean"],
                "validation_rmse_off": off["validation_rmse"],
                "validation_rmse_on": on["validation_rmse"],
                "delta_validation_rmse": on["validation_rmse"] - off["validation_rmse"],
                "validation_bias_off": off["validation_bias"],
                "validation_bias_on": on["validation_bias"],
            }
        )

forecast_compare_df = pd.DataFrame(forecast_compare_rows)

# ============================================================
# Save
# ============================================================
memory_seed_df.to_csv(RESULTS / "09_01b_y_memory_seed_results.csv", index=False)
aggregate_df.to_csv(RESULTS / "09_01b_y_memory_aggregate.csv", index=False)
paired_seed_df.to_csv(RESULTS / "09_01b_y_memory_paired_by_seed.csv", index=False)
paired_effect_df.to_csv(RESULTS / "09_01b_y_memory_paired_effect.csv", index=False)
forecast_df.to_csv(RESULTS / "09_01b_y_forecast_summary.csv", index=False)
forecast_compare_df.to_csv(RESULTS / "09_01b_y_forecast_comparison.csv", index=False)
audit_df.to_csv(RESULTS / "09_01b_y_week8_reproduction_audit.csv", index=False)
diagnostics_df.to_csv(RESULTS / "09_01b_y_diagnostics.csv", index=False)
washout_df.to_csv(RESULTS / "09_01b_y_washout_trajectories.csv", index=False)
unitary_df.to_csv(RESULTS / "09_01b_y_unitary_audit.csv", index=False)
if not memory_delay_df.empty:
    memory_delay_df.to_csv(RESULTS / "09_01b_y_memory_by_delay.csv", index=False)

manifest = {
    "step": "Week 9 Step 9.1B.1",
    "experiment": "memory-only Y Hamiltonian ablation with paired multi-probe robustness",
    "active_topologies": ACTIVE_TOPOLOGIES,
    "Y_conditions": Y_CONDITIONS,
    "Y_qubits": [4, 5],
    "alpha": ALPHA,
    "hx": HX,
    "dt": DT,
    "Trotter_r": TROTTER_R,
    "probe_seeds": PROBE_SEEDS,
    "anchor_probe_seed": ANCHOR_PROBE_SEED,
    "probe_seed_role": "controls only independent temporal permutations of the artificial F4 memory probe",
    "postwashout_burnin_rule": "B=max(100,Tw(0.01)+20), separately per topology/Y/probe seed",
    "washout_epsilon": WASHOUT_EPS,
    "max_acceptable_washout": MAX_ACCEPTABLE_WASHOUT,
    "memory_delays": MEMORY_DELAYS,
    "memory_readout": "XYZ_all",
    "memory_ridge_alpha": MC_RIDGE_ALPHA,
    "forecast_readouts": ["XZ_injection", "XYZ_all"],
    "forecast_ridge_alpha": FORECAST_RIDGE_ALPHA,
    "forecast_probe_seed_dependence": False,
    "train_rows": N_TRAIN,
    "validation_rows": N_VALIDATION,
    "test_rows_used": 0,
    "J_reoptimized": False,
    "J_source": str(J_WINNER_FILE),
    "data_source": str(DATA_FILE),
    "target_column": TARGET_COLUMN,
}

with open(RESULTS / "09_01b_y_manifest.json", "w", encoding="utf-8") as fp:
    json.dump(manifest, fp, indent=2)

# ============================================================
# Console summaries
# ============================================================
print("\n" + "=" * 110)
print("WEEK-8 ANCHOR-SEED Y-OFF REPRODUCTION AUDIT — PROBE SEED 79001")
print("=" * 110)
print(audit_df.to_string(index=False))

print("\n" + "=" * 110)
print("MULTI-SEED MEMORY AGGREGATE")
print("=" * 110)
print(
    aggregate_df[
        [
            "topology",
            "y_condition",
            "n_probe_seeds",
            "MC_ch_mean",
            "MC_ch_std",
            "MC_ch_median",
            "MC_ch_min",
            "MC_ch_max",
            "Tw_mean",
            "Tw_std",
            "valid_washout_seeds",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 110)
print("PAIRED Y-ON MINUS Y-OFF MEMORY EFFECT ACROSS THE SAME 5 PROBE SEEDS")
print("=" * 110)
print(paired_effect_df.to_string(index=False))

print("\n" + "=" * 110)
print("FORECAST Y-ON MINUS Y-OFF — REAL CHRONOLOGY, NO PROBE-SEED DEPENDENCE")
print("=" * 110)
print(forecast_compare_df.to_string(index=False))

print("\nSaved:")
for filename in [
    "09_01b_y_memory_seed_results.csv",
    "09_01b_y_memory_aggregate.csv",
    "09_01b_y_memory_paired_by_seed.csv",
    "09_01b_y_memory_paired_effect.csv",
    "09_01b_y_memory_by_delay.csv",
    "09_01b_y_forecast_summary.csv",
    "09_01b_y_forecast_comparison.csv",
    "09_01b_y_week8_reproduction_audit.csv",
    "09_01b_y_diagnostics.csv",
    "09_01b_y_washout_trajectories.csv",
    "09_01b_y_unitary_audit.csv",
    "09_01b_y_manifest.json",
]:
    print(f"  results/{filename}")

print(f"\nTotal runtime: {time.time() - experiment_start:.1f}s")
print("Step 9.1B.1 complete.")
