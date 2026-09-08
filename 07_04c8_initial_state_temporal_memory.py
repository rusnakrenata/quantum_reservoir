"""
Week 7 - Step 7.4C.8
Does a non-maximally-mixed initialization unlock temporal memory?

For each initial memory state:
    I/4, |00>, |11>, |++>, BellPhi+

compare:

    W1(init):
        reset to the SAME initial state before every current input

    CONT(init):
        start from that state once, then carry memory continuously

Both use:
    - real chronological F4 insurance input
    - same reservoir
    - XZ_injection only
    - no YX45 observable
    - no measurement back-action

Therefore:
    MC_CONT(init) - MC_W1(init)

isolates temporal information carried by the evolving memory state, while
controlling for the nonlinear feature map induced by that same initialization.

Dependency:
    07_04c5_CONT_vs_W1_temporal_memory.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
C5 = HERE / "07_04c5_CONT_vs_W1_temporal_memory.py"

if not C5.exists():
    raise FileNotFoundError(f"Missing {C5.name}. Keep it beside this script.")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


c5 = load_module(C5, "memory74c5")
qrc = c5.qrc
c4 = c5.c4

ALPHA = 0.75
SEED = 42
BURN_IN = 100
K_MAX = 28
SELECTED_DELAYS = [1, 2, 5, 7, 14, 21, 28]

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]

FEATURES_XZ_INJ = [
    "X0", "X1", "X2", "X3",
    "Z0", "Z1", "Z2", "Z3",
]


def ket_density(v):
    v = np.asarray(v, dtype=complex)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def initial_states():
    e00 = np.array([1, 0, 0, 0], dtype=complex)
    e11 = np.array([0, 0, 0, 1], dtype=complex)
    pp = np.array([1, 1, 1, 1], dtype=complex) / 2.0
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2.0)

    return {
        "I4": np.eye(4, dtype=complex) / 4.0,
        "00": ket_density(e00),
        "11": ket_density(e11),
        "pp": ket_density(pp),
        "BellPhi": ket_density(bell),
    }


def trace_distance(rho, sigma):
    delta = np.asarray(rho - sigma, dtype=complex)
    delta = (delta + delta.conj().T) / 2
    vals = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(vals)))


def one_step(A, rho_mem):
    rho_i, rho_m = qrc.final_reduced_states(A, rho_mem)
    feat = qrc.extract_feature_row(A, rho_mem, rho_i, rho_m)
    x = np.array([feat[n] for n in FEATURES_XZ_INJ], dtype=float)
    return rho_m, x


def build_features(A_list, init):
    # W1: same init reset every sample
    X_w1 = []
    for A in A_list:
        _, x = one_step(A, init)
        X_w1.append(x)
    X_w1 = np.vstack(X_w1)

    # CONT: same init once, then carry
    rho = init.copy()
    X_cont = []
    D_to_init = []
    D_to_I4 = []

    I4 = np.eye(4, dtype=complex) / 4.0

    for A in A_list:
        rho, x = one_step(A, rho)
        X_cont.append(x)
        D_to_init.append(trace_distance(rho, init))
        D_to_I4.append(trace_distance(rho, I4))

    return (
        X_w1,
        np.vstack(X_cont),
        np.asarray(D_to_init),
        np.asarray(D_to_I4),
    )


def evaluate_mc(U, X_w1, X_cont, n_train, n_val, init_name):
    rows = []

    for j in range(U.shape[1]):
        for k in range(1, K_MAX + 1):
            train_start = max(BURN_IN, k)

            tr = np.arange(train_start, n_train)
            va = np.arange(n_train, n_train + n_val)

            ytr = U[tr - k, j]
            yva = U[va - k, j]

            res_w1 = c4.train_and_validate(
                X_w1[tr], ytr, X_w1[va], yva
            )
            res_cont = c4.train_and_validate(
                X_cont[tr], ytr, X_cont[va], yva
            )

            mc_w1 = res_w1["MC_validation"]
            mc_cont = res_cont["MC_validation"]

            rows.append({
                "initial_state": init_name,
                "channel": j,
                "delay": k,
                "MC_W1": mc_w1,
                "MC_CONT": mc_cont,
                "Delta_MC_CONT_vs_W1": mc_cont - mc_w1,
                "alpha_W1": res_w1["ridge_alpha"],
                "alpha_CONT": res_cont["ridge_alpha"],
            })

    return pd.DataFrame(rows)


def summarize(detail, trajectory_rows):
    rows = []

    traj = pd.DataFrame(trajectory_rows)

    for init_name in detail["initial_state"].unique():
        s = detail[detail["initial_state"] == init_name]
        tr = traj[traj["initial_state"] == init_name].iloc[0]

        rows.append({
            "initial_state": init_name,
            "sum_MC_W1": float(s["MC_W1"].sum()),
            "sum_MC_CONT": float(s["MC_CONT"].sum()),
            "sum_Delta_MC_CONT_vs_W1": float(
                s["Delta_MC_CONT_vs_W1"].sum()
            ),
            "n_positive_delays": int(
                np.sum(s["Delta_MC_CONT_vs_W1"] > 0)
            ),
            "max_Delta_MC_CONT_vs_W1": float(
                s["Delta_MC_CONT_vs_W1"].max()
            ),
            "feature_RMS_CONT_vs_W1": tr["feature_RMS_CONT_vs_W1"],
            "D_to_I4_t0": tr["D_to_I4_t0"],
            "D_to_I4_t100": tr["D_to_I4_t100"],
            "D_to_I4_t500": tr["D_to_I4_t500"],
            "D_to_I4_final": tr["D_to_I4_final"],
        })

    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 128)
    print("WEEK 7 - STEP 7.4C.8")
    print("NON-MIXED INITIALIZATION TEST: DOES CONT GAIN TEMPORAL MEMORY?")
    print("=" * 128)
    print()
    print("Real chronological F4 input")
    print("XZ_injection only")
    print("No YX45 measurement / no back-action")
    print()
    print("For each initial state:")
    print("  W1   = reset to same initial state before every sample")
    print("  CONT = initialize once, then carry memory")
    print()

    work, train, val, cols = qrc.load_data()
    n_train = len(train)
    n_val = len(val)
    n_steps = n_train + n_val

    Uchron = np.asarray(
        qrc.make_input_angles(work, cols),
        dtype=float,
    )[:n_steps]

    Ures, unitary_err = qrc.build_trotter_unitary(SEED)
    A_list, _ = qrc.build_input_channels(Ures, Uchron)

    print(f"Train={n_train}, validation={n_val}, total={n_steps}")
    print(f"QRC unitarity error={unitary_err:.3e}")
    print()

    detail_parts = []
    trajectory_rows = []

    for name, init in initial_states().items():
        print(f"Running initialization: {name} ...")

        X_w1, X_cont, D_init, D_I4 = build_features(
            A_list,
            init,
        )

        rms = float(np.sqrt(np.mean((X_cont - X_w1) ** 2)))

        def at(arr, idx):
            idx = min(idx, len(arr) - 1)
            return float(arr[idx])

        trajectory_rows.append({
            "initial_state": name,
            "feature_RMS_CONT_vs_W1": rms,
            "D_to_I4_t0": at(D_I4, 0),
            "D_to_I4_t100": at(D_I4, 100),
            "D_to_I4_t500": at(D_I4, 500),
            "D_to_I4_final": at(D_I4, len(D_I4)-1),
            "D_to_initial_final": at(D_init, len(D_init)-1),
        })

        detail_parts.append(
            evaluate_mc(
                Uchron,
                X_w1,
                X_cont,
                n_train,
                n_val,
                name,
            )
        )

    detail = pd.concat(detail_parts, ignore_index=True)
    trajectory = pd.DataFrame(trajectory_rows)
    summary = summarize(detail, trajectory_rows)

    detail.to_csv(
        RESULTS / "07_04c8_initial_state_MC_by_delay.csv",
        index=False,
    )
    trajectory.to_csv(
        RESULTS / "07_04c8_initial_state_trajectory_summary.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS / "07_04c8_initial_state_MC_summary.csv",
        index=False,
    )

    print()
    print("-" * 128)
    print("TRAJECTORY SUMMARY")
    print("-" * 128)
    print(trajectory.to_string(index=False))

    print()
    print("-" * 128)
    print("MEMORY SUMMARY")
    print("-" * 128)
    print(summary.to_string(index=False))

    print()
    print("-" * 128)
    print("SELECTED DELAYS")
    print("-" * 128)

    selected = detail[
        detail["delay"].isin(SELECTED_DELAYS)
    ][
        [
            "initial_state",
            "channel",
            "delay",
            "MC_W1",
            "MC_CONT",
            "Delta_MC_CONT_vs_W1",
        ]
    ]

    print(selected.to_string(index=False))

    print()
    print("-" * 128)
    print("TOP CONT > W1 GAINS")
    print("-" * 128)

    top = detail.sort_values(
        "Delta_MC_CONT_vs_W1",
        ascending=False,
    ).head(25)

    print(
        top[
            [
                "initial_state",
                "channel",
                "delay",
                "MC_W1",
                "MC_CONT",
                "Delta_MC_CONT_vs_W1",
            ]
        ].to_string(index=False)
    )

    with open(
        RESULTS / "07_04c8_initial_state_MC_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "WEEK 7 STEP 7.4C.8 - INITIALIZATION AND TEMPORAL MEMORY\n"
        )
        fp.write("=" * 96 + "\n\n")
        fp.write("TRAJECTORY:\n")
        fp.write(trajectory.to_string(index=False))
        fp.write("\n\nMEMORY:\n")
        fp.write(summary.to_string(index=False))
        fp.write("\n")

    print()
    print("=" * 128)
    print("Saved:")
    print("  results/07_04c8_initial_state_MC_by_delay.csv")
    print("  results/07_04c8_initial_state_trajectory_summary.csv")
    print("  results/07_04c8_initial_state_MC_summary.csv")
    print("  results/07_04c8_initial_state_MC_summary.txt")
    print()
    print("Interpretation:")
    print("  If non-mixed CONT > same-init W1, carried state adds temporal memory.")
    print("  If all initial states still give CONT ≈ W1, initialization is not enough.")


if __name__ == "__main__":
    main()
