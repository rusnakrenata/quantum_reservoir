"""
Week 7 - Step 7.4C.5
CONT versus W1 memoryless-reset test on the real chronological insurance input.

Scientific question
-------------------
Does CONT contain genuine temporal information beyond the nonlinear feature
mapping produced from the CURRENT input alone?

We compare the same XZ_injection observable family under:

    W1_RESET:
        memory is reset to I/4 before EVERY input u(t).
        Therefore x_W1(t) is a nonlinear quantum feature map of u(t), but it
        cannot contain reservoir state carried from t-1, t-2, ...

    CONT:
        memory starts from I/4 once and is then carried continuously.
        Therefore x_CONT(t) may contain information from previous inputs.

Both protocols use:
    - same chronological F4 encoded input sequence
    - same reservoir unitary
    - same alpha=0.75
    - same seed=42
    - same XZ_injection observables
    - no YX45 observable
    - no measurement back-action

Memory readout
--------------
For each channel j and delay k=1..28, a separate Ridge readout reconstructs

    u_j(t-k)

from several feature sets:

    CURRENT        : u(t)
    W1             : x_W1(t)
    CONT           : x_CONT(t)
    CURRENT+W1     : [u(t), x_W1(t)]
    CURRENT+CONT   : [u(t), x_CONT(t)]

Important comparisons
---------------------
1) Pure protocol contrast:

    Delta_MC_CONT_vs_W1
        = MC_CONT - MC_W1

2) Stricter incremental temporal contribution after controlling for both
   current raw input and the nonlinear memoryless W1 map:

    Delta_MC_memory_add
        = MC_CURRENT+CONT - MC_CURRENT+W1

A positive Delta_MC_memory_add is the cleanest evidence here that the
continuous reservoir state contributes delayed-input information beyond a
memoryless quantum transformation of today's input.

Chronology
----------
Original chronological insurance input.
Train = 2022-2024
Validation = 2025
Common analysis burn-in = 100 samples.
2026 remains untouched.

Dependency
----------
Keep beside this script:
    07_04c4_real_chronological_task_memory.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
C4 = HERE / "07_04c4_real_chronological_task_memory.py"

if not C4.exists():
    raise FileNotFoundError(
        f"Missing {C4.name}. Keep it beside this script."
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


c4 = load_module(C4, "memory74c4")
qrc = c4.qrc

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


# =============================================================================
# Feature construction
# =============================================================================

def extract_xzinj(A, rho_before):
    """
    Apply one reservoir input and return:
        rho_memory_out, 8-dimensional XZ_injection feature vector.
    """
    rho_i_out, rho_m_out = qrc.final_reduced_states(A, rho_before)

    feat = qrc.extract_feature_row(
        A,
        rho_before,
        rho_i_out,
        rho_m_out,
    )

    x = np.asarray(
        [feat[name] for name in FEATURES_XZ_INJ],
        dtype=float,
    )

    return rho_m_out, x


def build_w1_features(A_list, reset_state):
    """
    W1 / memoryless control:
    every sample begins from the SAME memory state I/4.

    Thus there is no information transfer from a previous sample.
    """
    rows = []

    for A in A_list:
        _, x = extract_xzinj(A, reset_state)
        rows.append(x)

    return np.vstack(rows)


def build_cont_features(A_list, initial_state):
    """
    CONT:
    initialize once, then carry the memory state forward.
    """
    rho = initial_state.copy()
    rows = []

    for A in A_list:
        rho, x = extract_xzinj(A, rho)
        rows.append(x)

    return np.vstack(rows)


# =============================================================================
# Delayed-input reconstruction
# =============================================================================

def evaluate(U, X_w1, X_cont, n_train, n_val):
    rows = []

    matrices = {
        "CURRENT": U,
        "W1": X_w1,
        "CONT": X_cont,
        "CURRENT_plus_W1": np.hstack([U, X_w1]),
        "CURRENT_plus_CONT": np.hstack([U, X_cont]),
    }

    for j in range(U.shape[1]):
        print(f"  channel {j} ...")

        for k in range(1, K_MAX + 1):
            train_start = max(BURN_IN, k)

            train_t = np.arange(
                train_start,
                n_train,
                dtype=int,
            )
            val_t = np.arange(
                n_train,
                n_train + n_val,
                dtype=int,
            )

            ytr = U[train_t - k, j]
            yva = U[val_t - k, j]

            res = {}

            for model_name, Xall in matrices.items():
                res[model_name] = c4.train_and_validate(
                    Xall[train_t],
                    ytr,
                    Xall[val_t],
                    yva,
                )

            mc_current = res["CURRENT"]["MC_validation"]
            mc_w1 = res["W1"]["MC_validation"]
            mc_cont = res["CONT"]["MC_validation"]
            mc_cur_w1 = res["CURRENT_plus_W1"]["MC_validation"]
            mc_cur_cont = res["CURRENT_plus_CONT"]["MC_validation"]

            rows.append({
                "channel": j,
                "delay": k,

                "MC_current": mc_current,
                "MC_W1": mc_w1,
                "MC_CONT": mc_cont,
                "MC_current_plus_W1": mc_cur_w1,
                "MC_current_plus_CONT": mc_cur_cont,

                "Delta_MC_CONT_vs_W1": mc_cont - mc_w1,

                # Cleanest task-conditioned temporal contribution:
                "Delta_MC_memory_add": mc_cur_cont - mc_cur_w1,

                "Delta_MC_W1_vs_current": mc_w1 - mc_current,
                "Delta_MC_CONT_vs_current": mc_cont - mc_current,

                "alpha_current": res["CURRENT"]["ridge_alpha"],
                "alpha_W1": res["W1"]["ridge_alpha"],
                "alpha_CONT": res["CONT"]["ridge_alpha"],
                "alpha_current_plus_W1": res["CURRENT_plus_W1"]["ridge_alpha"],
                "alpha_current_plus_CONT": res["CURRENT_plus_CONT"]["ridge_alpha"],
            })

    return pd.DataFrame(rows)


def summarize(df):
    rows = []

    for j in sorted(df["channel"].unique()):
        s = df[df["channel"] == j].sort_values("delay")

        rows.append({
            "channel": j,

            "sum_MC_current": float(s["MC_current"].sum()),
            "sum_MC_W1": float(s["MC_W1"].sum()),
            "sum_MC_CONT": float(s["MC_CONT"].sum()),

            "sum_MC_current_plus_W1": float(
                s["MC_current_plus_W1"].sum()
            ),
            "sum_MC_current_plus_CONT": float(
                s["MC_current_plus_CONT"].sum()
            ),

            "sum_Delta_CONT_vs_W1": float(
                s["Delta_MC_CONT_vs_W1"].sum()
            ),
            "sum_Delta_memory_add": float(
                s["Delta_MC_memory_add"].sum()
            ),

            "n_delays_CONT_better_than_W1": int(
                np.sum(s["Delta_MC_CONT_vs_W1"] > 0)
            ),
            "n_delays_memory_add_positive": int(
                np.sum(s["Delta_MC_memory_add"] > 0)
            ),

            "max_Delta_CONT_vs_W1": float(
                s["Delta_MC_CONT_vs_W1"].max()
            ),
            "delay_max_Delta_CONT_vs_W1": int(
                s.loc[
                    s["Delta_MC_CONT_vs_W1"].idxmax(),
                    "delay",
                ]
            ),

            "max_Delta_memory_add": float(
                s["Delta_MC_memory_add"].max()
            ),
            "delay_max_Delta_memory_add": int(
                s.loc[
                    s["Delta_MC_memory_add"].idxmax(),
                    "delay",
                ]
            ),
        })

    rows.append({
        "channel": "ALL",

        "sum_MC_current": float(df["MC_current"].sum()),
        "sum_MC_W1": float(df["MC_W1"].sum()),
        "sum_MC_CONT": float(df["MC_CONT"].sum()),

        "sum_MC_current_plus_W1": float(
            df["MC_current_plus_W1"].sum()
        ),
        "sum_MC_current_plus_CONT": float(
            df["MC_current_plus_CONT"].sum()
        ),

        "sum_Delta_CONT_vs_W1": float(
            df["Delta_MC_CONT_vs_W1"].sum()
        ),
        "sum_Delta_memory_add": float(
            df["Delta_MC_memory_add"].sum()
        ),

        "n_delays_CONT_better_than_W1": int(
            np.sum(df["Delta_MC_CONT_vs_W1"] > 0)
        ),
        "n_delays_memory_add_positive": int(
            np.sum(df["Delta_MC_memory_add"] > 0)
        ),

        "max_Delta_CONT_vs_W1": float(
            df["Delta_MC_CONT_vs_W1"].max()
        ),
        "delay_max_Delta_CONT_vs_W1": np.nan,

        "max_Delta_memory_add": float(
            df["Delta_MC_memory_add"].max()
        ),
        "delay_max_Delta_memory_add": np.nan,
    })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 132)
    print("WEEK 7 - STEP 7.4C.5")
    print("REAL CHRONOLOGICAL TEMPORAL MEMORY: CONT VERSUS W1 RESET")
    print("=" * 132)
    print()
    print("Input:")
    print("  original chronological F4 injection-angle sequence")
    print()
    print("Observable family:")
    print("  XZ_injection only")
    print("  [X0,X1,X2,X3,Z0,Z1,Z2,Z3]")
    print("  NO YX45 observable")
    print("  NO measurement back-action")
    print()
    print("Protocol control:")
    print("  W1   = reset memory to I/4 before every current input")
    print("  CONT = initialize I/4 once, then carry memory continuously")
    print()
    print("Therefore:")
    print("  W1 contains current-input nonlinear quantum features but NO carried history.")
    print("  CONT uses the same quantum map but CAN contain carried history.")
    print()
    print("Primary quantities:")
    print("  Delta_MC_CONT_vs_W1 = MC_CONT - MC_W1")
    print(
        "  Delta_MC_memory_add = "
        "MC_[current+CONT] - MC_[current+W1]"
    )
    print()
    print(f"Delays k=1..{K_MAX}")
    print(f"Common analysis burn-in={BURN_IN}")
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

    reset_state = np.eye(qrc.DIM_M, dtype=complex) / qrc.DIM_M

    print(f"Train={n_train}, validation={n_val}, total={n_steps}")
    print(f"Injection channels={Uchron.shape[1]}")
    print(f"QRC unitarity error={unitary_err:.3e}")
    print()

    print("Building W1 memoryless-reset features ...")
    X_w1 = build_w1_features(
        A_list,
        reset_state,
    )

    print("Building CONT features ...")
    X_cont = build_cont_features(
        A_list,
        reset_state,
    )

    print()
    print(
        "Feature RMS difference CONT vs W1 = "
        f"{np.sqrt(np.mean((X_cont - X_w1)**2)):.8f}"
    )
    print()

    print("Evaluating delayed-input reconstruction ...")
    detail = evaluate(
        Uchron,
        X_w1,
        X_cont,
        n_train,
        n_val,
    )

    summary = summarize(detail)

    detail.to_csv(
        RESULTS / "07_04c5_CONT_vs_W1_memory_by_delay.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS / "07_04c5_CONT_vs_W1_memory_summary.csv",
        index=False,
    )

    print()
    print("-" * 132)
    print("SUMMARY")
    print("-" * 132)
    print(summary.to_string(index=False))

    print()
    print("-" * 132)
    print("SELECTED DELAYS")
    print("-" * 132)

    selected = detail[
        detail["delay"].isin(SELECTED_DELAYS)
    ][
        [
            "channel",
            "delay",
            "MC_current",
            "MC_W1",
            "MC_CONT",
            "MC_current_plus_W1",
            "MC_current_plus_CONT",
            "Delta_MC_CONT_vs_W1",
            "Delta_MC_memory_add",
        ]
    ]

    print(selected.to_string(index=False))

    print()
    print("-" * 132)
    print("TOP CONT > W1 TEMPORAL GAINS")
    print("-" * 132)

    top = detail.sort_values(
        "Delta_MC_CONT_vs_W1",
        ascending=False,
    ).head(20)

    print(
        top[
            [
                "channel",
                "delay",
                "MC_W1",
                "MC_CONT",
                "Delta_MC_CONT_vs_W1",
                "MC_current_plus_W1",
                "MC_current_plus_CONT",
                "Delta_MC_memory_add",
            ]
        ].to_string(index=False)
    )

    with open(
        RESULTS / "07_04c5_CONT_vs_W1_memory_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "WEEK 7 STEP 7.4C.5 - CONT VERSUS W1 TEMPORAL MEMORY\n"
        )
        fp.write("=" * 100 + "\n\n")
        fp.write(
            "W1 resets I/4 before each input; CONT carries memory.\n"
        )
        fp.write(
            "Both use XZ_injection only and the same chronological input.\n\n"
        )
        fp.write(
            "Delta_MC_CONT_vs_W1 = MC_CONT - MC_W1\n"
        )
        fp.write(
            "Delta_MC_memory_add = "
            "MC_current+CONT - MC_current+W1\n\n"
        )
        fp.write(summary.to_string(index=False))
        fp.write("\n")

    print()
    print("=" * 132)
    print("Saved:")
    print("  results/07_04c5_CONT_vs_W1_memory_by_delay.csv")
    print("  results/07_04c5_CONT_vs_W1_memory_summary.csv")
    print("  results/07_04c5_CONT_vs_W1_memory_summary.txt")
    print()
    print("Interpretation:")
    print("  CONT > W1 indicates temporal information carried by reservoir state.")
    print(
        "  Positive Delta_MC_memory_add is the stricter evidence after "
        "controlling for current-input predictability and the W1 nonlinear map."
    )


if __name__ == "__main__":
    main()
