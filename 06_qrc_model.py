"""
06_qrc_model.py
---------------
Quantum Reservoir Computing (QRC) with two execution backends:
  * Qiskit Aer   -- local noiseless (or noise-model) simulation
  * IBM Quantum  -- direct execution on a 127-qubit quantum computer

Architecture (Fujii & Nakajima 2017; Martinez-Pena 2021; Yasuda 2025)
----------------------------------------------------------------------
Reservoir   : N-qubit transverse-field Ising model
              H = sum_{i<j} J_ij X_i X_j + h sum_i Z_i
              J_ij ~ Uniform[-J_max, +J_max]; h = 1.0; tau = 4.0

Circuit     : Trotterized gate-based approximation
              RXX(2*J_ij*tau/n_trotter) for each coupled pair (i,j)
              Rx(2*h*tau/n_trotter)     for each qubit  [Rx ~ e^{-i theta/2 X}]

Note: The Hamiltonian has a Z field term (h*Z), but in the circuit
literature the transverse field is often along X.  Here we follow
Martinez-Pena 2021 and use:
  H = sum J_ij X_i X_j + h sum_i X_i
which maps cleanly to RXX + Rx gates without CNOT decomposition.
This is equivalent under a global Hadamard basis change.

Input encoding : Reset qubit 0 each step, then Ry(2*arcsin(sqrt(u_t))) on qubit 0.
                 This implements the partial-trace injection of Fujii 2017:
                 rho_0(t) = (1-u_t)|0><0| + u_t|1><1|

Sliding window : For hardware compatibility the circuit uses the last W
                 input values (fixed circuit depth regardless of T).
                 Simulation uses same circuit for direct comparison.

Virtual nodes  : After the final input injection, evolve for tau/V sub-steps,
                 measuring all N qubits after each.  Features = N*V expectation
                 values <Z_i>(v) = 1 - 2*P(qubit_i = 1 | virtual_node v).

Readout        : Direct multi-step forecasting.  Separate ridge regression
                 readout W_h for each horizon h in HORIZONS.
                 W_h = Y_h X^T (X X^T + alpha I)^{-1}

Hardware notes
--------------
* Circuits are batched (BATCH_SIZE per job) to reduce queue overhead.
* Use ibm_account.py to save credentials before running with --hardware.
* On hardware, T=T_HW training steps are used (instead of T_TRAIN=5000)
  to keep total circuit count manageable (approx T_HW*V circuits/seed).
* Transpilation at optimization_level=3 maps logical to physical qubits
  automatically based on calibration data (lowest error qubits chosen).

Usage
-----
    # Aer simulation (fast, no IBM account needed):
    python 06_qrc_model.py --sim

    # IBM hardware:
    python 06_qrc_model.py --hardware --backend ibm_brisbane

    # IBM noise model on Aer (local, no queue):
    python 06_qrc_model.py --sim --noise ibm_brisbane

    # Custom config:
    python 06_qrc_model.py --sim --n_qubits 7 --n_virtual 10 --note "7q sweep"
"""

import sys
import time
import argparse
import subprocess
import shutil
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge

from db_config import engine, get_session
from models import Base, QRCRun, QRCDataset, QRCModel, QRCExperiment, QRCResult

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

HORIZONS   = [1, 2, 4, 6, 8, 10]
SEEDS      = [42, 123, 456, 789, 2024]
TRAIN      = 5000
VAL        = 1000
TEST       = 2000
T_HW       = 800    # Shorter training series for real hardware (fewer circuits)
T_HW_OLP   = 20     # OLP hardware: circuit depth grows with T, keep very small
WASHOUT    = 100
BATCH_SIZE = 300    # Circuits per IBM job submission


@dataclass
class QRCConfig:
    """All hyperparameters for one QRC run."""
    # Reservoir
    n_qubits:   int   = 5
    n_virtual:  int   = 10      # virtual nodes V
    tau:        float = 4.0     # total evolution time per input step
    J_max:      float = 0.5     # |J_ij| sampled uniformly in [-J_max, J_max]
    h:          float = 1.0     # transverse field strength
    n_trotter:  int   = 4       # Trotter steps per tau (higher = more accurate)

    # Circuit / hardware
    window:     int   = 20      # sliding-window size W (fixed circuit depth)
    n_shots:    int   = 8192    # measurement shots per circuit
    batch_size: int   = BATCH_SIZE

    # Readout
    ridge_alpha: float = 1e-6
    washout:     int   = WASHOUT

    # Reproducibility
    seed:        int   = 42

    # Measurement protocol
    # 'RWP' : Rewinding Protocol -- sliding window, one circuit per time step
    # 'OLP' : Online Protocol    -- one circuit for full series, weak measurements
    protocol:    str   = 'RWP'

    # OLP weak-measurement angle theta_meas in [0, pi/2).
    #   theta_meas = 0   : projective (full collapse, max info, max back-action)
    #   theta_meas -> pi/2 : no measurement (no info, no back-action)
    # Mujal 2023 optimal: ~0.2-0.4 rad (weak enough to preserve memory,
    # strong enough to extract useful features).
    theta_meas:  float = 0.3

    # Physical qubit override (hardware only).  None = let transpiler choose.
    physical_qubits: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# Dataset generators  (same as 05_baselines.py)
# ---------------------------------------------------------------------------

def gen_mackey_glass(T, tau=17, beta=0.2, gamma=0.1, n=10,
                     dt=1.0, x0=1.2, washout=500):
    hist  = np.full(tau + 1, x0, dtype=float)
    total = T + washout
    out   = np.empty(total)
    out[0] = xc = x0
    for t in range(1, total):
        xd = hist[t % (tau + 1)]
        dx = beta * xc / (1 + xc**n) - gamma * xc
        xc = xc + dt * dx
        hist[t % (tau + 1)] = xc
        out[t] = xc
    series = out[washout:]
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn)


def gen_narma10(T, washout=50, seed=0):
    rng = np.random.default_rng(seed)
    u   = rng.uniform(0, 0.5, T + washout + 10)
    y   = np.zeros(T + washout + 10)
    for t in range(10, T + washout + 10):
        y[t] = (0.3 * y[t-1]
                + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-1] * u[t-10]
                + 0.1)
    return u[washout + 10:], y[washout + 10:]


def gen_lorenz(T, dt=0.02, sigma=10, rho=28, beta=8/3,
               x0=(1., 1., 1.05), washout=1000):
    def lorenz(t, s):
        x, y, z = s
        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]
    sol = solve_ivp(lorenz, [0, (T + washout)*dt], x0,
                    dense_output=True, max_step=dt)
    ts  = np.linspace(0, (T + washout)*dt, T + washout)
    xyz = sol.sol(ts).T[washout:]
    mn, mx = xyz[:,0].min(), xyz[:,0].max()
    return (xyz[:,0] - mn) / (mx - mn)


def gen_henon(T, a=1.4, b=0.3, washout=200):
    x, y = 0.1, 0.1
    out  = []
    for _ in range(T + washout):
        x, y = 1 - a*x**2 + y, b*x
        out.append(x)
    series = np.array(out[washout:])
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn)


DATASETS = {
    "MG-17":   lambda: gen_mackey_glass(TRAIN+VAL+TEST, tau=17),
    "MG-30":   lambda: gen_mackey_glass(TRAIN+VAL+TEST, tau=30),
    "NARMA-10":lambda: gen_narma10(TRAIN+VAL+TEST)[1],   # output series
    "Lorenz-x":lambda: gen_lorenz(TRAIN+VAL+TEST),
    "Henon-x": lambda: gen_henon(TRAIN+VAL+TEST),
}


# ---------------------------------------------------------------------------
# Database helpers  (mirrors 05_baselines.py)
# ---------------------------------------------------------------------------

def init_db():
    Base.metadata.create_all(engine)
    print("Database tables ready.")


def create_run(session, note=""):
    git_hash = None
    if shutil.which("git"):
        try:
            r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                git_hash = r.stdout.strip()
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


def nrmse(y_true, y_pred):
    var = np.var(y_true)
    if var < 1e-12:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / var))


def save_result(session, exp, split, y_true, y_pred, elapsed):
    res = QRCResult(
        experiment_id     = exp.id,
        split             = split,
        nrmse             = nrmse(y_true, y_pred),
        rmse              = float(np.sqrt(np.mean((y_true - y_pred)**2))),
        mae               = float(np.mean(np.abs(y_true - y_pred))),
        training_time_sec = elapsed if split == "train" else None,
    )
    session.add(res)
    session.flush()
    return res


# ---------------------------------------------------------------------------
# QRC Circuit Builder
# ---------------------------------------------------------------------------

class QiskitQRC:
    """
    Quantum Reservoir Computer implemented as a Qiskit parameterized circuit.

    Supports two backends:
      * AerSimulator  -- local simulation (noiseless or noise-model)
      * IBMBackend    -- real quantum hardware

    The same circuit code path is used for both; only the backend and
    (optionally) physical_qubits differ.
    """

    def __init__(self, config: QRCConfig, backend, is_hardware: bool = False):
        """
        Parameters
        ----------
        config      : QRCConfig instance
        backend     : AerSimulator or IBMBackend
        is_hardware : set True for real IBM backends to enable transpilation
                      and session-based job submission
        """
        self.cfg         = config
        self.backend     = backend
        self.is_hardware = is_hardware

        # Reservoir Hamiltonian parameters (random, fixed at construction)
        rng       = np.random.default_rng(config.seed)
        N         = config.n_qubits
        self._J   = rng.uniform(-config.J_max, config.J_max, (N, N))
        self._J   = (self._J + self._J.T) / 2   # symmetrize
        np.fill_diagonal(self._J, 0)

        # Coupling pairs: topology-aware for hardware, fully connected for sim.
        # Hardware path queries the device coupling map and picks N qubits with
        # the lowest 2Q gate errors, using ONLY native edges so no SWAPs are
        # ever needed (shallow circuit regardless of N).
        if is_hardware:
            self._init_hardware_topology(backend)
        else:
            self._pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

        # Readout (populated by fit())
        self._W_out: Dict[int, np.ndarray] = {}   # horizon -> weight matrix

    def _init_hardware_topology(self, backend):
        """
        Select the N best physical qubits and restrict coupling pairs to native
        edges ONLY — guaranteeing zero SWAP overhead and the shallowest possible
        Trotterized circuit on the target device.

        Algorithm
        ---------
        1. Read 2Q gate errors from backend.target  (ECR on Heron, CX on Eagle).
        2. Greedy expansion: seed on the qubit with lowest readout error, then
           iteratively add the unselected neighbour whose 2Q gate error is lowest.
        3. Collect all native edges within the selected subgraph.
        4. Store physical indices in self.cfg.physical_qubits so the transpiler
           can map logical -> physical with initial_layout (no routing needed).

        Side effects
        ------------
        Sets self._pairs           : logical (i,j) pairs with native edges
        Sets self.cfg.physical_qubits : selected physical qubit indices
        """
        N = self.cfg.n_qubits

        # ---- 1.  Error tables from backend.target -------------------------
        target     = backend.target
        num_qubits = backend.num_qubits

        # Identify the native 2Q gate (ECR for Heron r2, CX for Eagle r3)
        gate_2q = None
        for name in ('ecr', 'cx', 'cz'):
            if name in target.operation_names:
                gate_2q = name
                break

        edge_error    = {}   # (phys_i, phys_j) -> float
        readout_error = {}   # phys_i -> float

        if gate_2q:
            for qargs, props in target[gate_2q].items():
                if qargs is None or len(qargs) != 2:
                    continue
                q1, q2 = qargs
                err = (props.error
                       if props is not None and props.error is not None
                       else 1.0)
                edge_error[(q1, q2)] = err
                edge_error[(q2, q1)] = err

        for q in range(num_qubits):
            try:
                props = target['measure'][(q,)]
                readout_error[q] = (props.error
                                    if props is not None and props.error is not None
                                    else 1.0)
            except (KeyError, TypeError):
                readout_error[q] = 1.0

        # ---- 2.  Greedy qubit selection -----------------------------------
        if self.cfg.physical_qubits and len(self.cfg.physical_qubits) == N:
            selected = list(self.cfg.physical_qubits)
            print(f"  Using user-specified physical qubits: {selected}")
        else:
            # Seed: qubit with lowest readout error that has at least 1 neighbour
            connected = {q for (qa, _) in edge_error for q in [qa]}
            seed = min(connected, key=lambda q: readout_error.get(q, 1.0))
            selected = [seed]

            while len(selected) < N:
                selected_set = set(selected)
                best_q, best_cost = None, float('inf')
                for q in selected:
                    for (qa, qb), err in edge_error.items():
                        if qa == q and qb not in selected_set:
                            cost = err + 0.5 * readout_error.get(qb, 1.0)
                            if cost < best_cost:
                                best_cost, best_q = cost, qb
                if best_q is None:
                    print(f"  WARNING: connectivity exhausted at {len(selected)} "
                          f"qubits (requested N={N}). Reduce --n_qubits.")
                    break
                selected.append(best_q)

            # Report selection quality
            pairs_in = [(selected[i], selected[i+1])
                        for i in range(len(selected)-1)
                        if (selected[i], selected[i+1]) in edge_error]
            avg_err = (sum(edge_error.get(p, 1.0) for p in pairs_in) / max(len(pairs_in), 1))
            print(f"  Selected physical qubits : {selected}")
            print(f"  Native 2Q gate ({gate_2q})   : avg error {avg_err:.2%}")
            self.cfg.physical_qubits = selected

        # ---- 3.  Native logical pairs -------------------------------------
        selected_set = set(selected)
        phys_to_log  = {p: l for l, p in enumerate(selected)}

        self._pairs = []
        native_found = []
        for pi in selected:
            for pj in selected:
                li, lj = phys_to_log[pi], phys_to_log[pj]
                if li < lj and (pi, pj) in edge_error:
                    self._pairs.append((li, lj))
                    native_found.append((pi, pj))

        n_native   = len(self._pairs)
        n_possible = N * (N - 1) // 2
        print(f"  Coupling pairs (native / fully-connected): "
              f"{n_native} / {n_possible}  -- 0 SWAPs needed")

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _add_trotter_layer(self, qc, tau_step: float):
        """
        Append one first-order Trotter step for H = sum J_ij X_i X_j + h sum X_i.

        Gate decomposition:
            e^{-i J_ij X_i X_j dt} = RXX(2 * J_ij * dt)
            e^{-i h X_i dt}         = Rx(2 * h * dt)
        """
        # Two-qubit XX couplings
        for i, j in self._pairs:
            theta = 2.0 * self._J[i, j] * tau_step
            qc.rxx(theta, i, j)
        # Single-qubit transverse field
        for i in range(self.cfg.n_qubits):
            qc.rx(2.0 * self.cfg.h * tau_step, i)

    def _encode_input(self, qc, qubit: int, u: float):
        """
        Encode scalar input u in [0,1] on `qubit` via Ry rotation.

        State: |psi> = sqrt(1-u)|0> + sqrt(u)|1>
        Rotation: Ry(theta) with theta = 2*arcsin(sqrt(u))
        """
        u_clipped = float(np.clip(u, 1e-9, 1 - 1e-9))
        theta     = 2.0 * np.arcsin(np.sqrt(u_clipped))
        qc.ry(theta, qubit)

    def build_circuit(self, u_window: np.ndarray):
        """
        Build a Qiskit circuit for one time step.

        Parameters
        ----------
        u_window : array of shape (W,) with input values in [0,1].
                   u_window[-1] is the current time step's input.

        Circuit structure
        -----------------
        For k = 0 .. W-2  (all inputs except the last):
            reset(qubit_0)
            Ry(theta_k)(qubit_0)
            Trotter(tau)  [n_trotter steps, each of duration tau/n_trotter]

        For the final input (k = W-1) with V virtual node measurements:
            reset(qubit_0)
            Ry(theta_{W-1})(qubit_0)
            For v = 0 .. V-1:
                Trotter(tau/V)  [n_trotter steps, each of duration tau/(V*n_trotter)]
                measure all N qubits -> classical bits [v*N .. (v+1)*N - 1]

        Returns
        -------
        QuantumCircuit with N*V classical bits.
        """
        from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

        N       = self.cfg.n_qubits
        V       = self.cfg.n_virtual
        W       = len(u_window)
        tau_sub = self.cfg.tau / V
        dt      = tau_sub / self.cfg.n_trotter     # per Trotter step (virtual)
        dt_full = self.cfg.tau / self.cfg.n_trotter  # per Trotter step (full)

        qr = QuantumRegister(N, 'q')
        cr = ClassicalRegister(N * V, 'meas')
        qc = QuantumCircuit(qr, cr)

        # --- Window inputs (all except last): full tau evolution ---
        for k in range(W - 1):
            if k > 0:
                qc.reset(0)            # trace out input qubit (partial-trace injection)
            self._encode_input(qc, 0, u_window[k])
            qc.barrier()
            for _ in range(self.cfg.n_trotter):
                self._add_trotter_layer(qc, dt_full)
            qc.barrier()

        # --- Final input: virtual node measurements ---
        if W > 1:
            qc.reset(0)
        self._encode_input(qc, 0, u_window[-1])
        qc.barrier()

        for v in range(V):
            for _ in range(self.cfg.n_trotter):
                self._add_trotter_layer(qc, dt)
            # Measure all qubits into classical bits [v*N .. (v+1)*N - 1]
            qc.measure(list(range(N)), list(range(v * N, (v + 1) * N)))
            if v < V - 1:
                qc.barrier()

        return qc

    # ------------------------------------------------------------------
    # Running circuits
    # ------------------------------------------------------------------

    def _counts_to_features(self, counts: dict, n_shots: int) -> np.ndarray:
        """
        Convert measurement counts dict to <Z_i>(v) expectation values.

        Returns
        -------
        features : ndarray of shape (N*V,)
            features[v*N + i] = <Z_i> at virtual node v
                              = P(qubit_i=0) - P(qubit_i=1)
                              = 1 - 2 * P(qubit_i=1)

        Qiskit bitstring convention: rightmost character = classical bit 0 = qubit 0.
        """
        N  = self.cfg.n_qubits
        V  = self.cfg.n_virtual
        nf = N * V
        z  = np.zeros(nf)

        for bitstring, count in counts.items():
            # Remove spaces (Qiskit sometimes groups digits)
            bs = bitstring.replace(" ", "")
            # Reverse so index 0 = qubit 0
            bs_rev = bs[::-1]
            p = count / n_shots
            for idx in range(min(nf, len(bs_rev))):
                if bs_rev[idx] == '0':
                    z[idx] += p
                else:
                    z[idx] -= p
        return z

    def _run_circuits_aer(self, circuits) -> np.ndarray:
        """Run a list of circuits on Aer and return feature matrix (T, N*V)."""
        from qiskit import transpile as qk_transpile

        n_shots = self.cfg.n_shots
        all_features = []

        # Aer can handle large batches efficiently
        # Transpile once for the backend
        tc = qk_transpile(circuits, backend=self.backend, optimization_level=0)
        job = self.backend.run(tc, shots=n_shots)
        result = job.result()

        for i in range(len(circuits)):
            counts = result.get_counts(i)
            feat   = self._counts_to_features(counts, n_shots)
            all_features.append(feat)

        return np.array(all_features)   # (T, N*V)

    def _run_circuits_hardware(self, circuits) -> np.ndarray:
        """
        Run a list of circuits on IBM hardware.

        Circuits are submitted in batches of BATCH_SIZE within a single
        Session to minimise queue overhead.
        """
        from qiskit import transpile as qk_transpile
        from qiskit_ibm_runtime import Session, SamplerV2 as Sampler

        n_shots = self.cfg.n_shots
        N       = self.cfg.n_qubits
        V       = self.cfg.n_virtual

        print(f"  Transpiling {len(circuits)} circuits for {self.backend.name} ...",
              end="", flush=True)
        t0 = time.time()

        # All pairs are native edges -> no routing/SWAPs needed.
        # optimization_level=1 is sufficient (gate cancellation only).
        # initial_layout pins logical -> physical qubits chosen by
        # _init_hardware_topology for minimum error.
        tc = qk_transpile(
            circuits,
            backend=self.backend,
            optimization_level=1,
            initial_layout=self.cfg.physical_qubits,
            seed_transpiler=self.cfg.seed,
        )
        print(f" done ({time.time()-t0:.1f}s)")

        all_features = []
        bs = self.cfg.batch_size

        with Session(backend=self.backend) as session:
            sampler = Sampler(mode=session)
            for start in range(0, len(tc), bs):
                chunk = tc[start:start + bs]
                print(f"  Job {start//bs + 1}: circuits {start}..{start+len(chunk)-1} "
                      f"({n_shots} shots each) ...", end="", flush=True)
                t0  = time.time()
                pub = [(c,) for c in chunk]   # SamplerV2 PUB format
                job = sampler.run(pub, shots=n_shots)
                result = job.result()
                print(f" {time.time()-t0:.1f}s", flush=True)

                for pub_res in result:
                    # Access the 'meas' classical register
                    counts = pub_res.data.meas.get_counts()
                    feat   = self._counts_to_features(counts, n_shots)
                    all_features.append(feat)

        return np.array(all_features)   # (T, N*V)

    # ------------------------------------------------------------------
    # Online Protocol (OLP) -- weak measurement, single circuit for all T
    # ------------------------------------------------------------------

    def _build_olp_circuit(self, input_series: np.ndarray):
        """
        Build ONE quantum circuit that processes the ENTIRE input series online.

        Architecture (Mujal 2023, Online Protocol)
        ------------------------------------------
        Qubits 0..N-1     : system (reservoir) -- NEVER fully reset between steps
        Qubits N..2N-1    : ancilla (one per system qubit, for weak measurement)

        For each time step t = 0..T-1:
          1. Reset + encode only qubit 0  (partial-trace injection of u_t)
          2. For each virtual node v = 0..V-1:
               a. Trotterized Ising evolution for tau/V on system qubits
               b. Weak measurement of each system qubit i via ancilla N+i:
                    reset(ancilla_{N+i})
                    Ry(theta_meas, ancilla_{N+i})    # tune strength
                    CX(system_i, ancilla_{N+i})       # entangle
                    measure(ancilla_{N+i}) -> bit[t*V*N + v*N + i]

        Measurement back-action analysis
        ---------------------------------
        With theta_meas = phi:
          <Z_i>  =  -(1 / cos(phi)) * E[2*bit - 1]

        Derivation: after Ry(phi)|0> -> cos(phi/2)|0>+sin(phi/2)|1> on ancilla,
        then CX(system_i, ancilla), tracing out ancilla gives Kraus operators
        M_0 = diag(cos(phi/2), sin(phi/2)), M_1 = diag(sin(phi/2), cos(phi/2)).
        E[2*outcome - 1] = -cos(phi) * <Z_i>, so rescale by -1/cos(phi).

        Returns
        -------
        QuantumCircuit with 2*N qubits and T*V*N classical bits.
        """
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

        T       = len(input_series)
        N       = self.cfg.n_qubits
        V       = self.cfg.n_virtual
        phi     = self.cfg.theta_meas
        tau_sub = self.cfg.tau / V
        dt      = tau_sub / self.cfg.n_trotter

        if abs(np.cos(phi)) < 1e-6:
            raise ValueError(
                f"theta_meas={phi:.4f} ≈ π/2 makes the estimator undefined "
                f"(cos(theta)=0 → division by zero). Choose theta_meas < π/2."
            )

        n_bits = T * V * N
        qr = QuantumRegister(2 * N, 'q')
        cr = ClassicalRegister(n_bits, 'olp')
        qc = QuantumCircuit(qr, cr)

        for t in range(T):
            # -- Input injection on system qubit 0 only --
            qc.reset(0)
            self._encode_input(qc, 0, input_series[t])
            qc.barrier()

            # -- V virtual nodes with weak measurement --
            for v in range(V):
                # Trotterized evolution on system qubits 0..N-1
                for _ in range(self.cfg.n_trotter):
                    self._add_trotter_layer(qc, dt)

                # Weak measurement of each system qubit via its ancilla
                bit_offset = t * V * N + v * N
                for i in range(N):
                    anc = N + i
                    qc.reset(anc)
                    qc.ry(phi, anc)        # tune measurement strength
                    qc.cx(i, anc)          # entangle system with ancilla
                    qc.measure(anc, bit_offset + i)

                if v < V - 1:
                    qc.barrier()

            if t < T - 1:
                qc.barrier()

        return qc

    def _olp_bits_to_features(self, counts: dict, T: int, n_shots: int) -> np.ndarray:
        """
        Convert OLP bitstring counts to feature matrix X of shape (T, N*V).

        Classical bit layout: bit[ t*V*N + v*N + i ] = weak measurement of
        system qubit i at virtual node v, time step t.

        Qiskit convention: rightmost character of bitstring = classical bit 0.

        Feature value:
            X[t, v*N+i] = <Z_i>(t,v)
                        = -(1/cos(theta_meas)) * (1 - 2 * P(bit=1))
        """
        N       = self.cfg.n_qubits
        V       = self.cfg.n_virtual
        phi     = self.cfg.theta_meas
        n_bits  = T * V * N
        p_one   = np.zeros(n_bits)   # P(bit=1) for each position

        for bitstring, count in counts.items():
            bs = bitstring.replace(' ', '')
            p  = count / n_shots
            for idx in range(min(n_bits, len(bs))):
                # Qiskit: bit idx is at position -(idx+1) from right end
                if bs[-(idx + 1)] == '1':
                    p_one[idx] += p

        # Rescale: <Z_i> = -(1/cos(phi)) * (1 - 2*P(1))
        z_exp = -(1.0 / np.cos(phi)) * (1.0 - 2.0 * p_one)

        # Reshape to (T, N*V)
        X = np.empty((T, N * V))
        for t in range(T):
            for v in range(V):
                for i in range(N):
                    X[t, v * N + i] = z_exp[t * V * N + v * N + i]
        return X

    def _collect_features_olp(self, input_series: np.ndarray) -> np.ndarray:
        """
        OLP: build one big circuit for all T steps, run once, extract X.

        For Aer: noiseless; circuit depth scales with T (fine for simulation).
        For hardware: keep T small (T_HW_OLP=20) -- circuit depth is T * O(V*N).
        """
        T = len(input_series)
        N = self.cfg.n_qubits
        V = self.cfg.n_virtual

        print(f"  [OLP] Building 1 circuit: T={T}, N={N}, V={V}, "
              f"theta_meas={self.cfg.theta_meas:.3f} rad "
              f"({np.degrees(self.cfg.theta_meas):.1f}°) ...",
              end="", flush=True)
        t0  = time.time()
        qc  = self._build_olp_circuit(input_series)
        print(f" {time.time()-t0:.1f}s  depth≈{qc.depth()}")

        n_shots = self.cfg.n_shots
        print(f"  [OLP] Running {n_shots} shots ...", end="", flush=True)
        t0 = time.time()

        if self.is_hardware:
            from qiskit import transpile as qk_transpile
            from qiskit_ibm_runtime import Session, SamplerV2 as Sampler

            tc = qk_transpile(
                qc,
                backend=self.backend,
                optimization_level=1,
                initial_layout=self.cfg.physical_qubits,
                seed_transpiler=self.cfg.seed,
            )
            with Session(backend=self.backend) as session:
                sampler = Sampler(mode=session)
                result  = sampler.run([(tc,)], shots=n_shots).result()
            counts = result[0].data.olp.get_counts()
        else:
            from qiskit import transpile as qk_transpile
            tc     = qk_transpile(qc, backend=self.backend, optimization_level=0)
            job    = self.backend.run(tc, shots=n_shots)
            counts = job.result().get_counts(0)

        print(f" {time.time()-t0:.1f}s")

        X = self._olp_bits_to_features(counts, T, n_shots)
        print(f"  [OLP] Feature extraction done.  X.shape={X.shape}")
        return X

    def _collect_features_rwp(self, input_series: np.ndarray) -> np.ndarray:
        """
        RWP: build T separate sliding-window circuits (existing implementation).
        """
        T = len(input_series)
        W = self.cfg.window

        print(f"  [RWP] Building {T} circuits (N={self.cfg.n_qubits}, "
              f"V={self.cfg.n_virtual}, W={W}) ...", end="", flush=True)
        t0       = time.time()
        circuits = []
        for t in range(T):
            start  = max(0, t - W + 1)
            window = np.zeros(W)
            chunk  = input_series[start : t + 1]
            window[-len(chunk):] = chunk
            circuits.append(self.build_circuit(window))
        print(f" {time.time()-t0:.1f}s")

        print(f"  [RWP] Running circuits on backend ...", flush=True)
        t0 = time.time()
        if self.is_hardware:
            X = self._run_circuits_hardware(circuits)
        else:
            X = self._run_circuits_aer(circuits)
        print(f"  [RWP] Feature extraction done ({time.time()-t0:.1f}s)  "
              f"X.shape={X.shape}")
        return X

    def collect_features(self, input_series: np.ndarray) -> np.ndarray:
        """
        Collect the feature matrix using the configured protocol.

        Dispatches to:
          RWP -- _collect_features_rwp()  (one circuit per time step)
          OLP -- _collect_features_olp()  (one circuit for the whole series)

        Returns
        -------
        X : ndarray of shape (T, N*V)
        """
        if self.cfg.protocol == 'OLP':
            return self._collect_features_olp(input_series)
        return self._collect_features_rwp(input_series)

    # ------------------------------------------------------------------
    # Readout training and prediction
    # ------------------------------------------------------------------

    def fit(self, input_series: np.ndarray, targets: np.ndarray, horizon: int):
        """
        Collect features and train ridge regression readout for one horizon.

        Parameters
        ----------
        input_series : 1-D series of length T_train
        targets      : 1-D series; targets[t] = input_series[t + horizon]
        horizon      : integer, stored as key in _W_out

        Returns
        -------
        train_preds : predictions on the training set (length T_train - washout)
        """
        X_all = self.collect_features(input_series)      # (T_train, N*V)
        X     = X_all[self.cfg.washout:]
        y     = targets[self.cfg.washout:]

        clf = Ridge(alpha=self.cfg.ridge_alpha, fit_intercept=True)
        clf.fit(X, y)
        self._W_out[horizon] = clf

        return clf.predict(X), y

    def predict(self, input_series: np.ndarray, horizon: int) -> np.ndarray:
        """
        Collect features and predict using the trained readout.

        Parameters
        ----------
        input_series : 1-D series of length T_pred
        horizon      : must have been used in fit()

        Returns
        -------
        preds : ndarray of shape (T_pred,)
        """
        if horizon not in self._W_out:
            raise ValueError(f"Readout for horizon={horizon} not trained yet.")
        X = self.collect_features(input_series)
        return self._W_out[horizon].predict(X)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(session, run_id, ds_name, series, qrc_config, backend,
                   is_hardware, horizon, seed):
    """
    One experiment: train QRC on train split, evaluate on val and test.

    Parameters
    ----------
    ds_name    : e.g. "MG-17"
    series     : full 1-D normalised time series (length TRAIN+VAL+TEST)
    qrc_config : QRCConfig with seed already set
    horizon    : prediction steps ahead
    """
    T      = len(series)
    t_tr   = (TRAIN if not is_hardware
              else (T_HW_OLP if qrc_config.protocol == 'OLP' else T_HW))
    t_val  = VAL
    t_test = TEST

    # Series splits
    tr_in  = series[:t_tr]
    val_in = series[t_tr : t_tr + t_val]
    te_in  = series[t_tr + t_val : t_tr + t_val + t_test]

    # Build targets: series[t + horizon]
    def make_target(s_in, s_full, offset, h):
        full_len = len(s_in)
        targets  = np.empty(full_len)
        for t in range(full_len):
            idx = offset + t + h
            targets[t] = s_full[idx] if idx < len(s_full) else float("nan")
        return targets

    tr_tgt  = make_target(tr_in,  series, 0,                        horizon)
    val_tgt = make_target(val_in, series, t_tr,                     horizon)
    te_tgt  = make_target(te_in,  series, t_tr + t_val,             horizon)

    # Mask out NaN at the end
    def strip_nan(inp, tgt):
        mask = ~np.isnan(tgt)
        return inp[mask], tgt[mask]

    tr_in, tr_tgt   = strip_nan(tr_in,  tr_tgt)
    val_in, val_tgt = strip_nan(val_in, val_tgt)
    te_in, te_tgt   = strip_nan(te_in,  te_tgt)

    # Register dataset and model in DB
    ds_rec, _ = get_or_create(
        session, QRCDataset, name=ds_name,
        defaults=dict(series_type="chaotic",
                      parameters={},
                      n_train=t_tr, n_val=t_val, n_test=t_test))

    backend_tag = "ibm_hardware" if is_hardware else "qiskit_aer"
    model_name  = (f"QRC-{qrc_config.n_qubits}q-V{qrc_config.n_virtual}"
                   f"-{backend_tag}-seed{seed}")
    model_rec, _ = get_or_create(
        session, QRCModel, name=model_name,
        defaults=dict(category="quantum",
                      description=(
                          f"Trotterized Ising QRC, N={qrc_config.n_qubits} qubits, "
                          f"V={qrc_config.n_virtual} virtual nodes, {backend_tag}"
                      )))

    exp = QRCExperiment(
        run_id          = run_id,
        dataset_id      = ds_rec.id,
        model_id        = model_rec.id,
        horizon         = horizon,
        hyperparameters = {
            **{k: v for k, v in asdict(qrc_config).items()
               if k != "physical_qubits"},
            "backend": backend_tag,
            "t_train": t_tr,
        },
    )
    session.add(exp)
    session.flush()

    # Build and train QRC
    print(f"\n  [{ds_name}] h={horizon} seed={seed}")
    qrc = QiskitQRC(qrc_config, backend, is_hardware=is_hardware)

    t0 = time.time()
    tr_preds, tr_true = qrc.fit(tr_in, tr_tgt, horizon)
    elapsed = time.time() - t0

    # Evaluate on all splits
    val_preds = qrc.predict(val_in, horizon)
    te_preds  = qrc.predict(te_in,  horizon)

    save_result(session, exp, "train", tr_true,  tr_preds, elapsed)
    save_result(session, exp, "val",   val_tgt,  val_preds, None)
    save_result(session, exp, "test",  te_tgt,   te_preds, None)
    session.commit()

    print(f"    train NRMSE={nrmse(tr_true, tr_preds):.4f}  "
          f"val={nrmse(val_tgt, val_preds):.4f}  "
          f"test={nrmse(te_tgt, te_preds):.4f}  "
          f"time={elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="QRC model: Aer or IBM hardware")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sim",      action="store_true",
                      help="Run on Aer simulator (local, no IBM account needed)")
    mode.add_argument("--hardware", action="store_true",
                      help="Run on real IBM quantum hardware")

    p.add_argument("--backend",   default="ibm_brisbane",
                   help="IBM backend name (default: ibm_brisbane); ignored with --sim")
    p.add_argument("--noise",     metavar="BACKEND_NAME", default=None,
                   help="With --sim: load noise model from this real backend")
    p.add_argument("--n_qubits",  type=int,   default=5)
    p.add_argument("--n_virtual", type=int,   default=None,
                   help="Virtual nodes V (default: 10 sim / 5 hardware)")
    p.add_argument("--tau",       type=float, default=4.0)
    p.add_argument("--window",    type=int,   default=None,
                   help="Sliding-window W (default: 20 sim / 10 hardware)")
    p.add_argument("--n_shots",   type=int,   default=None,
                   help="Shots per circuit (default: 8192 sim / 2048 hardware)")
    p.add_argument("--n_trotter", type=int,   default=None,
                   help="Trotter steps per tau (default: 4 sim / 1 hardware)")
    p.add_argument("--ridge_alpha", type=float, default=1e-6)
    p.add_argument("--protocol",  choices=["RWP", "OLP"], default="RWP",
                   help=("RWP (Rewinding, sliding-window, default) or "
                          "OLP (Online weak measurement, one circuit for full series). "
                          "OLP is best for Aer simulation; "
                          "for hardware use --t_hw_olp to limit series length."))
    p.add_argument("--theta_meas", type=float, default=0.3,
                   help=("OLP weak-measurement angle in radians. "
                          "0 = projective (max info, max back-action); "
                          "pi/2 ≈ 1.57 = no measurement. "
                          "Mujal 2023 optimal sweet spot: ~0.2-0.4 (default: 0.3)."))
    p.add_argument("--t_hw_olp",  type=int,   default=T_HW_OLP,
                   help=f"Training series length for OLP on hardware (default: {T_HW_OLP})")
    p.add_argument("--datasets",  nargs="+",
                   default=["MG-17", "MG-30", "NARMA-10", "Lorenz-x", "Henon-x"],
                   help="Datasets to run")
    p.add_argument("--horizons",  nargs="+", type=int, default=HORIZONS)
    p.add_argument("--seeds",     nargs="+", type=int, default=SEEDS)
    p.add_argument("--note",      default="",
                   help="Free-text run label stored in DB")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Hardware-appropriate defaults (shallow circuit = shallow args) ----
    # Simulation: fully connected graph, deep Trotter is fine.
    # Hardware:   native edges only (no SWAPs), n_trotter=1 keeps depth minimal.
    if args.hardware:
        if args.n_trotter is None: args.n_trotter = 1
        if args.n_virtual  is None: args.n_virtual  = 5
        if args.window     is None: args.window     = 10
        if args.n_shots    is None: args.n_shots    = 2048
        if args.protocol == "OLP":
            print(f"[OLP+hardware] T capped at {args.t_hw_olp} steps ")
            print(f"  (circuit depth ∝ T×V×N; current hardware T1/T2 limits depth)")
    else:
        if args.n_trotter is None: args.n_trotter = 4
        if args.n_virtual  is None: args.n_virtual  = 10
        if args.window     is None: args.window     = 20
        if args.n_shots    is None: args.n_shots    = 8192

    # ---- Select backend ----
    from ibm_account import get_aer_backend, get_ibm_backend, get_service

    if args.sim:
        if args.noise:
            print(f"Loading noise model from {args.noise} ...", flush=True)
            hw_backend = get_ibm_backend(args.noise, service=get_service())
            backend    = get_aer_backend(method="density_matrix",
                                         device_backend=hw_backend)
        else:
            backend = get_aer_backend(method="statevector")
        is_hardware = False
    else:  # --hardware
        backend     = get_ibm_backend(args.backend, service=get_service())
        is_hardware = True

    # ---- DB setup ----
    init_db()
    session = get_session()
    run     = create_run(session, note=args.note or
                         ("hw:" + args.backend if is_hardware else "sim"))
    session.commit()

    # ---- Generate all datasets upfront ----
    print("\nGenerating datasets ...", flush=True)
    dataset_series = {}
    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"  WARNING: unknown dataset '{ds_name}', skipping")
            continue
        dataset_series[ds_name] = DATASETS[ds_name]()
        print(f"  {ds_name}: {len(dataset_series[ds_name])} samples")

    # ---- Run experiments ----
    total = (len(dataset_series) * len(args.seeds)
             * len(args.horizons))
    done  = 0

    for ds_name, series in dataset_series.items():
        for seed in args.seeds:
            cfg = QRCConfig(
                n_qubits    = args.n_qubits,
                n_virtual   = args.n_virtual,
                tau         = args.tau,
                window      = args.window,
                n_shots     = args.n_shots,
                n_trotter   = args.n_trotter,
                ridge_alpha = args.ridge_alpha,
                protocol    = args.protocol,
                theta_meas  = args.theta_meas,
                seed        = seed,
            )
            for horizon in args.horizons:
                print(f"\n[{done+1}/{total}] {ds_name}  seed={seed}  h={horizon}",
                      flush=True)
                try:
                    run_experiment(
                        session, run.id, ds_name, series,
                        cfg, backend, is_hardware, horizon, seed,
                    )
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    session.rollback()
                finally:
                    done += 1

    session.close()
    print(f"\nAll done. Run id={run.id}")


if __name__ == "__main__":
    main()



# OLP simulation (recommended for comparing protocols)
#python 06_qrc_model.py --sim --protocol OLP --theta_meas 0.3

# Sweep measurement strength to find the sweet spot
#python 06_qrc_model.py --sim --protocol OLP --theta_meas 0.1 --note "olp strong"
#python 06_qrc_model.py --sim --protocol OLP --theta_meas 0.5 --note "olp weak"