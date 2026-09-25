#!/usr/bin/env python
from __future__ import annotations

"""
11.1E.9C — Candidate #3 logical-vs-transpiled ideal A/B audit (compact-6 fix)

NO QPU jobs are submitted.
2026 is never loaded.

A = frozen logical ideal Candidate #3 features.
B = exact noiseless expectation values after transpilation to the chosen IBM layout.

PASS means the logical feature map survives transpilation/layout mapping to numerical
precision, so the repeated real-QPU collapse is not explained by a logical/transpilation
mismatch.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

from ibm_account import get_service


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DEFAULT_BACKEND = "ibm_kingston"
DEFAULT_LAYOUT = [105, 117, 125, 124, 126, 127]
DEFAULT_BATCH = 80
FEATURES = ["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3"]


def find_direct_script():
    for name in [
        "11_1C_third_candidate_direct_qpu.py",
        "11_1c_third_candidate_direct_qpu.py",
    ]:
        p = HERE / name
        if p.exists():
            return p
    raise FileNotFoundError("11_1C_third_candidate_direct_qpu.py not found.")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


direct = load_module(find_direct_script(), "qrc_candidate3_direct_ab")


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def parse_layout(text):
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 6 or len(set(vals)) != 6:
        raise argparse.ArgumentTypeError(
            "--layout must contain six unique comma-separated qubits"
        )
    return vals


def append_parameterized_step(qc, candidate, angle_params, first_step):
    # EXACTLY mirrors Candidate #3's RWP step, including mid-circuit reset.
    if not first_step:
        for q in range(4):
            qc.reset(q)

    for q in range(4):
        qc.ry(angle_params[q], q)

    r = int(candidate["r"])
    dt = float(candidate["dt"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])
    J = candidate["J"]

    for _ in range(r):
        for i, j in direct.common.TOPOLOGY_EDGES[candidate["topology"]]:
            qc.rzz(2.0 * float(J[(i, j)]) * dt / r, i, j)

        theta_x = 2.0 * hx * dt / r
        for q in range(6):
            qc.rx(theta_x, q)

        if not np.isclose(hy, 0.0):
            theta_y = 2.0 * hy * dt / r
            qc.ry(theta_y, 4)
            qc.ry(theta_y, 5)


def build_parameterized_core(candidate):
    W = int(candidate["window"])
    qreg = QuantumRegister(6, "q")
    qc = QuantumCircuit(qreg, name=f"{candidate['topology']}_W{W}_C3_AB")
    pars = ParameterVector("theta", W * 4)

    for step in range(W):
        p = [pars[step * 4 + q] for q in range(4)]
        append_parameterized_step(qc, candidate, p, first_step=(step == 0))

    return qc, pars


def bind_endpoint(core, pars, angles, endpoint, W):
    start = int(endpoint) - int(W) + 1
    values = {}
    for step, idx in enumerate(range(start, int(endpoint) + 1)):
        for q in range(4):
            values[pars[step * 4 + q]] = float(angles[idx, q])
    return core.assign_parameters(values, inplace=False)


def logical_observable(feature):
    q = int(feature[1])
    axis = feature[0]
    return SparsePauliOp.from_sparse_list(
        [(axis, [q], 1.0)],
        num_qubits=6,
    )



def compact_transpiled_core(physical_core, physical_layout):
    """
    Relabel the six active physical wires of the transpiled IBM circuit to
    compact wires 0..5 so Aer does not attempt to allocate a 156-qubit
    density matrix.

    The mapping is deliberately:
        physical_layout[logical_q] -> compact logical_q

    Thus for the current layout:
        P105 -> q0
        P117 -> q1
        P125 -> q2
        P124 -> q3
        P126 -> q4
        P127 -> q5

    Delays on any wire are omitted because they are exact identities in this
    noiseless A/B audit. Barriers are retained only on active wires.
    Any nontrivial operation on a physical qubit outside the selected six
    aborts the audit.
    """
    phys_to_compact = {
        int(p): logical_q
        for logical_q, p in enumerate(physical_layout)
    }

    compact = QuantumCircuit(
        6,
        name=physical_core.name + "_compact6",
    )
    compact.global_phase = physical_core.global_phase

    skipped_inactive_delays = 0
    skipped_active_delays = 0
    retained_ops = 0

    for ci in physical_core.data:
        op = ci.operation
        qidx = [
            int(physical_core.find_bit(q).index)
            for q in ci.qubits
        ]

        if ci.clbits:
            raise RuntimeError(
                "Unexpected classical bits in the unmeasured transpiled core."
            )

        # Scheduling delays are physically meaningful for noisy timing, but
        # they are exact identities in this ideal equivalence audit.
        if op.name == "delay":
            if any(q in phys_to_compact for q in qidx):
                skipped_active_delays += 1
            else:
                skipped_inactive_delays += 1
            continue

        if op.name == "barrier":
            active = [
                phys_to_compact[q]
                for q in qidx
                if q in phys_to_compact
            ]
            if active:
                compact.barrier(*active)
            continue

        if not qidx:
            # No ordinary gate in this experiment should have zero qargs.
            # Ignore only a true no-op style instruction.
            if op.name in {"id"}:
                continue
            raise RuntimeError(
                f"Unexpected zero-qubit operation after transpilation: {op.name}"
            )

        outside = [q for q in qidx if q not in phys_to_compact]
        if outside:
            raise RuntimeError(
                "Nontrivial transpiled operation touches a qubit outside the "
                f"selected six. operation={op.name}, qargs={qidx}, "
                f"outside={outside}"
            )

        compact_qargs = [
            compact.qubits[phys_to_compact[q]]
            for q in qidx
        ]
        compact.append(op, compact_qargs, [])
        retained_ops += 1

    audit = {
        "physical_to_compact": {
            str(k): int(v)
            for k, v in phys_to_compact.items()
        },
        "retained_non_delay_ops": int(retained_ops),
        "skipped_active_delays": int(skipped_active_delays),
        "skipped_inactive_delays": int(skipped_inactive_delays),
        "compact_depth": int(compact.depth()),
        "compact_size": int(compact.size()),
        "compact_count_ops": {
            str(k): int(v)
            for k, v in compact.count_ops().items()
        },
    }

    return compact, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument(
        "--layout",
        type=parse_layout,
        default=DEFAULT_LAYOUT,
        help="Default: 105,117,125,124,126,127",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("=" * 118)
    print("WEEK 11.1E.9C — CANDIDATE #3 LOGICAL-vs-TRANSPILED IDEAL A/B AUDIT")
    print("=" * 118)
    print("NO QPU JOB WILL BE SUBMITTED.")
    print("2026 remains FROZEN / NOT LOADED.")
    print()

    candidate, meta = direct.load_candidate()
    frozen = direct.build_frozen_readout(candidate, meta)

    if list(direct.FEATURES) != FEATURES:
        raise RuntimeError(f"Unexpected feature order: {direct.FEATURES}")

    print("Frozen Candidate #3:")
    print(f"  candidate={direct.CANDIDATE_KEY}")
    print(f"  topology={candidate['topology']}, W={candidate['window']}, r={candidate['r']}")
    print(f"  alpha={candidate['alpha']}, dt={candidate['dt']}")
    print(f"  hx={candidate['hx']:+.12f}, hy={candidate['hy']:+.12f}")
    print(f"  readout={direct.EXPECTED_READOUT}")
    print(f"  lambda={frozen['lambda']}")
    print(f"  training CV RMSE={frozen['cv_rmse']:.6f}")
    print(f"  ideal 2025 RMSE={frozen['ideal_rmse']:.6f}")
    print()

    all_e = np.asarray(frozen["val_endpoints"], dtype=int)
    if args.quick:
        idx = np.unique(np.linspace(0, len(all_e) - 1, 30, dtype=int))
        endpoints = all_e[idx]
        X_A = np.asarray(frozen["X_val"], dtype=float)[idx]
        y = np.asarray(frozen["y_val"], dtype=float)[idx]
        pred_A = np.asarray(frozen["pred_val"], dtype=float)[idx]
        mode = f"QUICK ({len(endpoints)} endpoints)"
    else:
        endpoints = all_e
        X_A = np.asarray(frozen["X_val"], dtype=float)
        y = np.asarray(frozen["y_val"], dtype=float)
        pred_A = np.asarray(frozen["pred_val"], dtype=float)
        mode = f"FULL 2025 ({len(endpoints)} endpoints)"

    print(f"Audit mode: {mode}")
    print(f"Layout: {args.layout}")
    print()

    service = get_service()
    backend = service.backend(args.backend, use_fractional_gates=False)
    try:
        backend.refresh()
    except Exception:
        pass

    logical_core, pars = build_parameterized_core(candidate)

    physical_core = transpile(
        logical_core,
        backend=backend,
        initial_layout=args.layout,
        routing_method="none",
        optimization_level=int(direct.OPT_LEVEL),
        seed_transpiler=int(direct.SEED_TRANSPILE),
        scheduling_method="alap",
    )

    backend.check_faulty(physical_core)
    ops = {str(k): int(v) for k, v in physical_core.count_ops().items()}

    if ops.get("swap", 0) != 0:
        raise RuntimeError("SWAP detected in strict A/B compile.")

    print("Transpiled core:")
    print(f"  depth={physical_core.depth()}")
    print(f"  size={physical_core.size()}")
    print(f"  CZ={ops.get('cz', 0)}")
    print(f"  reset={ops.get('reset', 0)}")
    print(f"  swap={ops.get('swap', 0)}")
    print()

    # First audit Qiskit's physical observable mapping on the full
    # 156-qubit transpiled circuit.
    physical_mapped_obs = [
        logical_observable(f).apply_layout(physical_core.layout)
        for f in FEATURES
    ]

    print("Physical mapped observables:")
    obs_json = {}
    for f, obs in zip(FEATURES, physical_mapped_obs):
        terms = []
        for pauli, qargs, coeff in obs.to_sparse_list():
            row = {
                "pauli": str(pauli),
                "qargs": [int(q) for q in qargs],
                "coeff_real": float(np.real(coeff)),
                "coeff_imag": float(np.imag(coeff)),
            }
            terms.append(row)
        obs_json[f] = terms
        print(f"  {f}: {terms}")
    print()

    # Aer cannot density-matrix simulate the full 156-qubit backend register.
    # Compact the exact active physical circuit back to six wires, preserving
    # every non-delay gate and reset.
    compact_core, compact_audit = compact_transpiled_core(
        physical_core,
        args.layout,
    )

    print("Compact six-qubit transpiled core:")
    print(
        "  physical -> compact: "
        + json.dumps(compact_audit["physical_to_compact"], sort_keys=True)
    )
    print(f"  depth={compact_core.depth()}")
    print(f"  size={compact_core.size()}")
    print(f"  CZ={compact_core.count_ops().get('cz', 0)}")
    print(f"  reset={compact_core.count_ops().get('reset', 0)}")
    print(f"  swap={compact_core.count_ops().get('swap', 0)}")
    print(
        f"  skipped active scheduling delays="
        f"{compact_audit['skipped_active_delays']}"
    )
    print()

    # Because compact wire k is explicitly the physical wire carrying logical
    # q_k, the compact observables are once again simply Xk/Zk.
    mapped_obs = [
        logical_observable(f)
        for f in FEATURES
    ]

    estimator = AerEstimatorV2(
        options={
            "default_precision": 0.0,
            "backend_options": {
                "method": "density_matrix",
                "enable_truncation": True,
            },
        }
    )

    bound = [
        bind_endpoint(
            compact_core,
            pars,
            frozen["angles"],
            int(e),
            int(candidate["window"]),
        )
        for e in endpoints
    ]

    X_B = np.empty((len(bound), len(FEATURES)), dtype=float)

    for start in range(0, len(bound), args.batch_size):
        stop = min(start + args.batch_size, len(bound))
        pubs = [(bound[i], mapped_obs) for i in range(start, stop)]
        result = estimator.run(pubs, precision=0.0).result()

        if len(result) != len(pubs):
            raise RuntimeError("Estimator result length mismatch.")

        for local_i, pub in enumerate(result):
            evs = np.asarray(pub.data.evs, dtype=float).reshape(-1)
            if evs.size != len(FEATURES):
                raise RuntimeError(f"Unexpected EV shape: {pub.data.evs.shape}")
            X_B[start + local_i, :] = evs

        print(f"  exact Aer batch {start + 1}-{stop} / {len(bound)} complete")

    d = X_B - X_A
    global_rmse = float(np.sqrt(np.mean(d ** 2)))
    global_max = float(np.max(np.abs(d)))

    feature_rows = []
    for j, f in enumerate(FEATURES):
        dd = d[:, j]
        feature_rows.append({
            "feature": f,
            "A_mean": float(np.mean(X_A[:, j])),
            "B_mean": float(np.mean(X_B[:, j])),
            "mean_difference": float(np.mean(dd)),
            "rmse_A_vs_B": float(np.sqrt(np.mean(dd ** 2))),
            "max_abs_difference": float(np.max(np.abs(dd))),
            "corr_A_vs_B": corr(X_A[:, j], X_B[:, j]),
        })
    feature_df = pd.DataFrame(feature_rows)

    pred_B = direct.common.predict_scaled_ridge(
        frozen["model"], frozen["scaler"], frozen["keep"], X_B
    )

    pred_rmse = rmse(pred_A, pred_B)
    pred_max = float(np.max(np.abs(pred_B - pred_A)))
    pred_corr = corr(pred_A, pred_B)

    pass_tol = 1e-8
    passed = global_rmse <= pass_tol and global_max <= 1e-7

    print()
    print("=" * 118)
    print("A/B EQUIVALENCE RESULTS")
    print("=" * 118)
    print(f"Global feature RMSE A vs B:      {global_rmse:.12e}")
    print(f"Global max |feature difference|: {global_max:.12e}")
    print(f"Prediction RMSE A vs B:          {pred_rmse:.12e}")
    print(f"Prediction max |difference|:     {pred_max:.12e}")
    print(f"Prediction correlation A vs B:   {pred_corr:.12f}")
    print(f"A forecast RMSE:                 {rmse(y, pred_A):.9f}")
    print(f"B forecast RMSE:                 {rmse(y, pred_B):.9f}")
    print(f"A/B audit:                       {'PASS' if passed else 'FAIL'}")
    print()
    print(feature_df.to_string(index=False, float_format=lambda x: f"{x:.12e}"))

    endpoint_rows = []
    for i, e in enumerate(endpoints):
        row = {
            "endpoint": int(e),
            "target": float(y[i]),
            "pred_A_logical_ideal": float(pred_A[i]),
            "pred_B_transpiled_ideal": float(pred_B[i]),
            "pred_B_minus_A": float(pred_B[i] - pred_A[i]),
        }
        for j, f in enumerate(FEATURES):
            row[f"{f}_A_logical"] = float(X_A[i, j])
            row[f"{f}_B_transpiled"] = float(X_B[i, j])
            row[f"{f}_B_minus_A"] = float(d[i, j])
        endpoint_rows.append(row)

    endpoint_df = pd.DataFrame(endpoint_rows)

    p1 = RESULTS / "11_1E9C_candidate3_ab_audit_features_predictions.csv"
    p2 = RESULTS / "11_1E9C_candidate3_ab_audit_feature_summary.csv"
    p3 = RESULTS / "11_1E9C_candidate3_ab_audit_summary.json"

    endpoint_df.to_csv(p1, index=False)
    feature_df.to_csv(p2, index=False)

    summary = {
        "candidate_key": direct.CANDIDATE_KEY,
        "backend": args.backend,
        "layout": args.layout,
        "mode": mode,
        "n_endpoints": int(len(endpoints)),
        "global_feature_rmse_A_vs_B": global_rmse,
        "global_max_abs_feature_difference": global_max,
        "prediction_rmse_A_vs_B": pred_rmse,
        "prediction_max_abs_difference": pred_max,
        "prediction_corr_A_vs_B": pred_corr,
        "forecast_rmse_A": rmse(y, pred_A),
        "forecast_rmse_B": rmse(y, pred_B),
        "pass": bool(passed),
        "pass_tolerance_feature_rmse": pass_tol,
        "transpiled_core": {
            "depth": int(physical_core.depth()),
            "size": int(physical_core.size()),
            "count_ops": ops,
        },
        "compact_transpiled_core": compact_audit,
        "mapped_observables": obs_json,
        "2026_loaded": False,
    }
    p3.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("Saved:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")
    print()
    print("2026 remains FROZEN / UNUSED.")

    if not passed:
        raise RuntimeError(
            "A/B audit FAILED. Do not attribute the QPU collapse to hardware yet."
        )


if __name__ == "__main__":
    main()
