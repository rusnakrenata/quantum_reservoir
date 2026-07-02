"""
05_baselines.py
---------------
Generates all benchmark chaotic time series, runs classical baseline models
across multiple prediction horizons and (for ESN) multiple random seeds,
and stores every result in the MySQL database via SQLAlchemy.

Baselines
---------
1. Naive          -- x(t+h) = x(t)
2. Linear (Ridge) -- sliding-window ridge regression, direct multi-step
3. ESN-100        -- Echo State Network, 100 nodes  (multiple seeds)
4. ESN-500        -- Echo State Network, 500 nodes  (multiple seeds)

Multi-step strategy: DIRECT forecasting.
    For each horizon h a separate readout W_out is trained.
    The reservoir weights and dynamics are fixed (same for all h).
    This avoids compounding errors of recursive 1-step forecasting.

Seeds: ESN reservoir weights are random; we sweep SEEDS and report
    mean ± std NRMSE so reviewers cannot object to cherry-picking.

Usage
-----
    python 05_baselines.py
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge

from db_config import engine, get_session
from models import Base, QRCRun, QRCDataset, QRCModel, QRCExperiment, QRCResult

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

HORIZONS = [1, 2, 4, 6, 8, 10]
SEEDS    = [42, 123, 456, 789, 2024]    # ESN reservoir seeds
TRAIN, VAL, TEST = 5000, 1000, 2000
WASHOUT  = 100                           # ESN transient steps discarded

# ---------------------------------------------------------------------------
# 0.  Database helpers
# ---------------------------------------------------------------------------

def init_db():
    Base.metadata.create_all(engine)
    print("Database tables ready.\n")



def create_run(session, note=""):
    """
    Insert a QRCRun row and return it.
    Tries to capture the current git short-hash; silently skips if git
    is unavailable (e.g. running outside the repo).
    """
    import subprocess, shutil
    git_hash = None
    if shutil.which("git"):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                git_hash = result.stdout.strip()
        except Exception:
            pass

    run = QRCRun(git_hash=git_hash, note=note)
    session.add(run)
    session.flush()
    print(f"Run #{run.id}  git={git_hash or 'n/a'}  note={note!r}")
    return run


def get_or_create(session, cls, defaults=None, **kw):
    obj = session.query(cls).filter_by(**kw).first()
    if obj:
        return obj, False
    obj = cls(**{**kw, **(defaults or {})})
    session.add(obj)
    session.flush()
    return obj, True


# ---------------------------------------------------------------------------
# 1.  Dataset generators  (return raw numpy arrays)
# ---------------------------------------------------------------------------

def gen_mackey_glass(T, tau=17, beta=0.2, gamma=0.1, n=10,
                     dt=1.0, x0=1.2, washout=500):
    hist = np.full(tau + 1, x0, dtype=float)
    total = T + washout
    out = np.empty(total)
    out[0] = xc = x0
    for i in range(1, total):
        xd = hist[i % (tau + 1)]
        xc = xc + dt * (beta * xd / (1. + xd**n) - gamma * xc)
        hist[i % (tau + 1)] = xc
        out[i] = xc
    return out[washout:]


def gen_narma(T, order=10, alpha=0.3, beta=0.05, gamma=1.5,
              delta=0.1, seed=42):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for t in range(order, T - 1):
        y[t+1] = np.clip(
            alpha*y[t] + beta*y[t]*np.sum(y[t-order+1:t+1])
            + gamma*u[t-order+1]*u[t] + delta, 0, 1)
    return u, y


def gen_lorenz(T_total=100., dt=0.02, x0=(0., 1., 1.05),
               sigma=10., rho=28., beta=8/3, washout_t=10.):
    def rhs(t, s):
        x, y, z = s
        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]
    t_eval = np.arange(0, T_total + washout_t, dt)
    sol = solve_ivp(rhs, (0, T_total + washout_t), x0, method='RK45',
                    t_eval=t_eval, rtol=1e-9, atol=1e-9)
    wi = int(washout_t / dt)
    return sol.y[0, wi:]   # x-component


def gen_henon(T, a=1.4, b=0.3, x0=0., y0=0., washout=500):
    total = T + washout
    x, y = np.empty(total), np.empty(total)
    x[0], y[0] = x0, y0
    for i in range(1, total):
        x[i] = 1. - a*x[i-1]**2 + y[i-1]
        y[i] = b*x[i-1]
    return x[washout:]


# ---------------------------------------------------------------------------
# 2.  Splits helper
# ---------------------------------------------------------------------------

def make_splits(inp, tgt, horizon):
    """
    Build train/val/test dicts for one horizon.

    inp[t] is the model input at time t.
    tgt[t] = series[t + horizon]  (pre-shifted by caller).

    Returns {split: (inp_slice, tgt_slice)}.
    Trims the last `horizon` samples where target would be out of range.
    """
    N = min(len(inp), len(tgt))
    return {
        "train": (inp[:TRAIN],                       tgt[:TRAIN]),
        "val":   (inp[TRAIN:TRAIN+VAL],              tgt[TRAIN:TRAIN+VAL]),
        "test":  (inp[TRAIN+VAL:TRAIN+VAL+TEST],     tgt[TRAIN+VAL:TRAIN+VAL+TEST]),
    }


# ---------------------------------------------------------------------------
# 3.  Metrics
# ---------------------------------------------------------------------------

def nrmse(yt, yp):
    n = min(len(yt), len(yp))
    return float(np.sqrt(np.mean((yp[:n]-yt[:n])**2) / np.var(yt[:n])))

def rmse(yt, yp):
    n = min(len(yt), len(yp))
    return float(np.sqrt(np.mean((yp[:n]-yt[:n])**2)))

def mae(yt, yp):
    n = min(len(yt), len(yp))
    return float(np.mean(np.abs(yp[:n]-yt[:n])))

def all_metrics(yt, yp):
    return {"nrmse": nrmse(yt,yp), "rmse": rmse(yt,yp), "mae": mae(yt,yp)}


# ---------------------------------------------------------------------------
# 4.  Models
# ---------------------------------------------------------------------------

class NaivePredictor:
    """x(t+h) = x(t) for any h."""
    def fit(self, inp, tgt): pass
    def predict(self, inp):  return inp.copy()


class SlidingWindowRidge:
    """
    Ridge regression on W past input values, direct multi-step.

    fit(inp, tgt):
        inp[t] = series[t]
        tgt[t] = series[t+h]   (horizon already baked in by caller)
        Window i = inp[i:i+W]  predicts tgt[i+W-1]
    """
    def __init__(self, window=10, alpha=1e-4):
        self.window = window
        self.alpha  = alpha
        self._clf   = Ridge(alpha=alpha)

    def _feats(self, inp):
        W = self.window
        n = len(inp) - W + 1
        return np.array([inp[i:i+W] for i in range(n)])   # (n, W)

    def fit(self, inp, tgt):
        X    = self._feats(inp)          # (n, W);  window i ends at inp[i+W-1]
        y    = tgt[self.window-1:][:len(X)]   # tgt[i+W-1] for each window i
        self._clf.fit(X, y)

    def predict(self, inp):
        return self._clf.predict(self._feats(inp))

    def predict_aligned(self, inp, tgt):
        yp = self.predict(inp)
        yt = tgt[self.window-1:][:len(yp)]
        return yp, yt


class EchoStateNetwork:
    """
    Echo State Network — direct multi-step, warm-start across splits.

    For each horizon h:
      * The reservoir is driven by inp (same dynamics, horizon-independent).
      * Only the readout W_out is re-trained per horizon.
      * State is carried continuously: train -> val -> test.

    Multiple seeds: instantiate a new ESN per seed so W_res / W_in differ.
    """
    def __init__(self, n_reservoir=200, spectral_radius=0.9, sparsity=0.05,
                 input_scaling=1.0, ridge_alpha=1e-6, seed=42):
        self.n_reservoir     = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity        = sparsity
        self.input_scaling   = input_scaling
        self.ridge_alpha     = ridge_alpha
        self.seed            = seed
        self.x_state = None
        self.W_res   = None
        self.W_in    = None
        self.W_out   = None
        self._train_preds   = None
        self._train_targets = None

    def _build(self, n_in):
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((self.n_reservoir, self.n_reservoir))
        W[rng.random(W.shape) > self.sparsity] = 0.
        sr = np.max(np.abs(np.linalg.eigvals(W)))
        if sr > 1e-10:
            W *= self.spectral_radius / sr
        self.W_res = W
        self.W_in  = rng.uniform(-1, 1, (self.n_reservoir, n_in)) * self.input_scaling

    def _drive(self, inp, x_init=None, washout=0):
        T = len(inp)
        x = np.zeros(self.n_reservoir) if x_init is None else x_init.copy()
        states = np.empty((T, self.n_reservoir))
        for t in range(T):
            x = np.tanh(self.W_res @ x + self.W_in @ inp[t])
            states[t] = x
        self.x_state = x.copy()
        return states[washout:]

    def fit(self, inp, tgt, washout=WASHOUT):
        """
        inp: (T,) raw input series
        tgt: (T,) h-step-ahead targets  (tgt[t] = series[t+h])
        """
        u = inp[:, None]
        self._build(1)
        self.x_state = None
        states = self._drive(u, washout=washout)
        # state[i] corresponds to inp[washout+i]; target is tgt[washout+i]
        y_al = tgt[washout: washout + len(states)]
        clf  = Ridge(alpha=self.ridge_alpha)
        clf.fit(states, y_al)
        self.W_out          = clf
        self._train_preds   = clf.predict(states)
        self._train_targets = y_al

    def predict(self, inp, tgt):
        """Warm-start; returns (y_pred, y_true) aligned arrays."""
        states = self._drive(inp[:, None], x_init=self.x_state, washout=0)
        yp = self.W_out.predict(states)
        yt = tgt[:len(yp)]
        return yp, yt


# ---------------------------------------------------------------------------
# 5.  Dataset catalogue
# ---------------------------------------------------------------------------

def build_datasets():
    """
    Return list of dicts with keys:
        meta     -- dict for QRCDataset columns
        series   -- raw input numpy array  (long enough for all horizons)
        targets  -- dict {h: target_array} for each horizon
                    (or None to use series itself)
        is_narma -- True for NARMA where input != output series
    """
    N = TRAIN + VAL + TEST + max(HORIZONS) + 200   # buffer

    # NARMA generated once; same u, same y for all horizons
    u_narma, y_narma = gen_narma(N, order=10, seed=42)

    catalogue = [
        dict(
            meta=dict(name="mackey_glass_tau17", series_type="dde",
                      parameters={"tau":17,"beta":0.2,"gamma":0.1,"n":10},
                      n_train=TRAIN, n_val=VAL, n_test=TEST,
                      description="Mackey-Glass DDE, weakly chaotic tau=17"),
            series=gen_mackey_glass(N, tau=17),
        ),
        dict(
            meta=dict(name="mackey_glass_tau30", series_type="dde",
                      parameters={"tau":30,"beta":0.2,"gamma":0.1,"n":10},
                      n_train=TRAIN, n_val=VAL, n_test=TEST,
                      description="Mackey-Glass DDE, strongly chaotic tau=30"),
            series=gen_mackey_glass(N, tau=30),
        ),
        dict(
            meta=dict(name="narma10", series_type="synthetic",
                      parameters={"order":10,"alpha":0.3,"beta":0.05,
                                  "gamma":1.5,"delta":0.1},
                      n_train=TRAIN, n_val=VAL, n_test=TEST,
                      description="NARMA-10 synthetic nonlinear memory task"),
            series=u_narma,
            targets=y_narma,    # separate input/output series
        ),
        dict(
            meta=dict(name="lorenz_x", series_type="ode",
                      parameters={"sigma":10.,"rho":28.,
                                  "beta":round(8/3,6),"dt":0.02},
                      n_train=TRAIN, n_val=VAL, n_test=TEST,
                      description="Lorenz x-component, direct multi-step"),
            series=gen_lorenz(T_total=(N*0.02+5.)),
        ),
        dict(
            meta=dict(name="henon_x", series_type="discrete_map",
                      parameters={"a":1.4,"b":0.3},
                      n_train=TRAIN, n_val=VAL, n_test=TEST,
                      description="Henon map x-component, direct multi-step"),
            series=gen_henon(N),
        ),
    ]

    # Pre-compute target arrays for every horizon
    for ds in catalogue:
        s   = ds["series"]
        out = ds.get("targets", s)   # for NARMA: out = y_narma; else = s itself
        ds["horizon_targets"] = {
            h: out[h:]    # out[h:][t] = out[t+h]
            for h in HORIZONS
        }

    return catalogue


# ---------------------------------------------------------------------------
# 6.  Model catalogue
# ---------------------------------------------------------------------------

ESN_COMMON = dict(spectral_radius=0.9, ridge_alpha=1e-6)

MODEL_REGISTRY = [
    dict(name="naive",             category="trivial",
         desc="Predict x(t+h)=x(t), no parameters.",
         hparams={}, is_esn=False,
         factory=lambda seed=None: NaivePredictor()),
    dict(name="linear_ridge_w10",  category="trivial",
         desc="Ridge regression on 10-step sliding window, direct multi-step.",
         hparams={"window":10,"alpha":1e-4}, is_esn=False,
         factory=lambda seed=None: SlidingWindowRidge(window=10, alpha=1e-4)),
    dict(name="esn_n100",          category="classical_rc",
         desc="ESN 100 nodes, sr=0.9, sparsity=0.05.",
         hparams={**ESN_COMMON,"n_reservoir":100,"sparsity":0.05},
         is_esn=True,
         factory=lambda seed: EchoStateNetwork(n_reservoir=100, sparsity=0.05,
                                               seed=seed, **ESN_COMMON)),
    dict(name="esn_n500",          category="classical_rc",
         desc="ESN 500 nodes, sr=0.9, sparsity=0.02.",
         hparams={**ESN_COMMON,"n_reservoir":500,"sparsity":0.02},
         is_esn=True,
         factory=lambda seed: EchoStateNetwork(n_reservoir=500, sparsity=0.02,
                                               seed=seed, **ESN_COMMON)),
]


# ---------------------------------------------------------------------------
# 7.  Runner  (one model, one dataset, one horizon, one seed)
# ---------------------------------------------------------------------------

def run_one(model, splits, is_esn):
    """Returns {split_name: metrics_dict}."""
    X_tr, y_tr = splits["train"]
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_sec = time.perf_counter() - t0

    results = {}

    if is_esn:
        results["train"] = {
            **all_metrics(model._train_targets, model._train_preds),
            "training_time_sec": train_sec,
        }
        for split in ("val", "test"):
            X, y = splits[split]
            yp, yt = model.predict(X, y)
            results[split] = all_metrics(yt, yp)

    elif isinstance(model, SlidingWindowRidge):
        for split, (X, y) in splits.items():
            yp, yt = model.predict_aligned(X, y)
            m = all_metrics(yt, yp)
            if split == "train":
                m["training_time_sec"] = train_sec
            results[split] = m

    else:  # Naive
        for split, (X, y) in splits.items():
            yp = model.predict(X)
            results[split] = all_metrics(y, yp)

    return results


# ---------------------------------------------------------------------------
# 8.  Persist
# ---------------------------------------------------------------------------

def save_experiment(session, run_id, ds_row, model_row, horizon, hparams, split_results):
    exp = QRCExperiment(
        run_id=run_id, dataset_id=ds_row.id, model_id=model_row.id,
        horizon=horizon, hyperparameters=hparams,
    )
    session.add(exp)
    session.flush()
    for split, m in split_results.items():
        session.add(QRCResult(
            experiment_id=exp.id, split=split,
            nrmse=m.get("nrmse"), rmse=m.get("rmse"), mae=m.get("mae"),
            training_time_sec=m.get("training_time_sec"),
        ))
    return exp


# ---------------------------------------------------------------------------
# 9.  Main
# ---------------------------------------------------------------------------

def main():
    init_db()
    session = get_session()
    datasets = build_datasets()

    import sys
    run_note = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        run = create_run(session, note=run_note)
        run_id = run.id

        for ds_def in datasets:
            meta = ds_def["meta"]
            ds_row, created = get_or_create(
                session, QRCDataset,
                defaults={k: v for k, v in meta.items() if k != "name"},
                name=meta["name"],
            )
            tag = "[+]" if created else "[=]"
            print(f"\n{tag} Dataset: {meta['name']}")

            inp_series = ds_def["series"]

            for h in HORIZONS:
                tgt_series = ds_def["horizon_targets"][h]
                splits     = make_splits(inp_series, tgt_series, h)

                print(f"\n  Horizon h={h:2d}  "
                      f"{'Model':22s}  {'NRMSE':>8}  {'±std':>7}  "
                      f"{'RMSE':>9}  {'MAE':>9}")
                print("  " + "-"*70)

                for m_def in MODEL_REGISTRY:
                    model_row, _ = get_or_create(
                        session, QRCModel,
                        defaults={"category": m_def["category"],
                                   "description": m_def["desc"]},
                        name=m_def["name"],
                    )

                    seed_list  = SEEDS if m_def["is_esn"] else [None]
                    test_nrmse = []

                    for seed in seed_list:
                        inst = m_def["factory"](seed=seed)
                        split_res = run_one(inst, splits, m_def["is_esn"])

                        hparams = {**m_def["hparams"]}
                        if seed is not None:
                            hparams["seed"] = seed

                        save_experiment(session, run_id, ds_row, model_row,
                                        h, hparams, split_res)
                        test_nrmse.append(split_res["test"]["nrmse"])

                    mn  = float(np.mean(test_nrmse))
                    std = float(np.std(test_nrmse)) if len(test_nrmse) > 1 else 0.

                    # For display: take last split_res for rmse/mae
                    t = split_res["test"]
                    print(f"  h={h:2d}  {m_def['name']:22s}  "
                          f"{mn:8.5f}  {std:7.5f}  "
                          f"{t['rmse']:9.5f}  {t['mae']:9.5f}")

            print()

        session.commit()
        print("All results committed to database.")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
