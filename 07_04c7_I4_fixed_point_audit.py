"""
Week 7 - Step 7.4C.7
Audit whether maximally mixed memory I/4 is a fixed point of the QRC memory channel.

Question
--------
For each input u_t, the reservoir induces a memory update

    rho_m(t+1) = E_{u_t}[rho_m(t)].

We test whether

    E_{u_t}[I/4] = I/4

for:
    1) the real chronological insurance inputs,
    2) randomized inputs sampled within the empirical angle ranges,
    3) empirical per-channel min/max corner inputs.

If true for all tested inputs, then initializing CONT at I/4 traps the memory
at I/4 and makes CONT equivalent to a reset-W1 protocol.

Dependency
----------
Keep beside this script:
    07_04c6_memory_influence_audit.py
"""

from __future__ import annotations

import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
C6 = HERE / "07_04c6_memory_influence_audit.py"

if not C6.exists():
    raise FileNotFoundError(
        f"Missing {C6.name}. Keep it beside this script."
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


c6 = load_module(C6, "memory74c6")
qrc = c6.qrc

ALPHA = 0.75
SEED = 42
N_RANDOM = 500
RANDOM_SEED = 74007

qrc.ALPHA = ALPHA
qrc.SEEDS = [SEED]


# =============================================================================
# Helpers
# =============================================================================

def trace_distance(rho, sigma):
    delta = np.asarray(rho - sigma, dtype=complex)
    delta = (delta + delta.conj().T) / 2.0
    eigvals = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def frobenius_distance(rho, sigma):
    return float(np.linalg.norm(rho - sigma, ord="fro"))


def max_abs_entry_diff(rho, sigma):
    return float(np.max(np.abs(rho - sigma)))


def one_memory_update(A, rho_mem):
    _, rho_m_out = qrc.final_reduced_states(A, rho_mem)
    return rho_m_out


def audit_A_list(A_list, label):
    rho_star = np.eye(qrc.DIM_M, dtype=complex) / qrc.DIM_M

    rows = []

    for idx, A in enumerate(A_list):
        rho_out = one_memory_update(A, rho_star)

        rows.append({
            "group": label,
            "index": idx,
            "trace_distance_to_I4": trace_distance(rho_out, rho_star),
            "frobenius_distance_to_I4": frobenius_distance(rho_out, rho_star),
            "max_abs_entry_diff_to_I4": max_abs_entry_diff(rho_out, rho_star),
            "purity_out": float(np.real(np.trace(rho_out @ rho_out))),
            "trace_out": float(np.real(np.trace(rho_out))),
            "hermiticity_error": float(
                np.max(np.abs(rho_out - rho_out.conj().T))
            ),
        })

    return pd.DataFrame(rows)


def summarize(df):
    rows = []

    for group in df["group"].unique():
        s = df[df["group"] == group]

        rows.append({
            "group": group,
            "n_cases": len(s),
            "mean_trace_distance": float(
                s["trace_distance_to_I4"].mean()
            ),
            "median_trace_distance": float(
                s["trace_distance_to_I4"].median()
            ),
            "max_trace_distance": float(
                s["trace_distance_to_I4"].max()
            ),
            "max_frobenius_distance": float(
                s["frobenius_distance_to_I4"].max()
            ),
            "max_abs_entry_diff": float(
                s["max_abs_entry_diff_to_I4"].max()
            ),
            "purity_min": float(s["purity_out"].min()),
            "purity_max": float(s["purity_out"].max()),
            "max_trace_error": float(
                np.max(np.abs(s["trace_out"] - 1.0))
            ),
            "max_hermiticity_error": float(
                s["hermiticity_error"].max()
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    RESULTS.mkdir(exist_ok=True)

    print("=" * 124)
    print("WEEK 7 - STEP 7.4C.7")
    print("AUDIT: IS I/4 A FIXED POINT OF THE MEMORY UPDATE?")
    print("=" * 124)
    print()
    print("Test:")
    print("  rho_in = I/4")
    print("  rho_out = E_u[rho_in]")
    print("  measure D(rho_out, I/4)")
    print()
    print("If D ~ 1e-14 for all tested inputs:")
    print("  I/4 is effectively a fixed point of the implemented memory channel.")
    print()

    work, train, val, cols = qrc.load_data()
    n_steps = len(train) + len(val)

    angles_real = np.asarray(
        qrc.make_input_angles(work, cols),
        dtype=float,
    )[:n_steps]

    Ures, unitary_err = qrc.build_trotter_unitary(SEED)

    print(f"Real chronological steps={n_steps}")
    print(f"Injection channels={angles_real.shape[1]}")
    print(f"QRC unitarity error={unitary_err:.3e}")
    print()

    # -------------------------------------------------------------------------
    # 1. Real chronological inputs
    # -------------------------------------------------------------------------
    print("Testing real chronological inputs ...")
    A_real, _ = qrc.build_input_channels(
        Ures,
        angles_real,
    )
    df_real = audit_A_list(
        A_real,
        "real_chronological",
    )

    # -------------------------------------------------------------------------
    # 2. Random inputs within empirical angle ranges
    # -------------------------------------------------------------------------
    print("Testing randomized inputs within empirical angle ranges ...")

    mins = np.min(angles_real, axis=0)
    maxs = np.max(angles_real, axis=0)

    rng = np.random.default_rng(RANDOM_SEED)
    angles_random = rng.uniform(
        low=mins,
        high=maxs,
        size=(N_RANDOM, angles_real.shape[1]),
    )

    A_random, _ = qrc.build_input_channels(
        Ures,
        angles_random,
    )
    df_random = audit_A_list(
        A_random,
        "random_uniform_empirical_range",
    )

    # -------------------------------------------------------------------------
    # 3. Corners: all empirical min/max combinations
    # -------------------------------------------------------------------------
    print("Testing empirical min/max corners ...")

    corners = np.array(
        [
            [
                mins[j] if bit == 0 else maxs[j]
                for j, bit in enumerate(bits)
            ]
            for bits in product([0, 1], repeat=angles_real.shape[1])
        ],
        dtype=float,
    )

    A_corners, _ = qrc.build_input_channels(
        Ures,
        corners,
    )
    df_corners = audit_A_list(
        A_corners,
        "empirical_minmax_corners",
    )

    # -------------------------------------------------------------------------
    # Combine and summarize
    # -------------------------------------------------------------------------
    detail = pd.concat(
        [df_real, df_random, df_corners],
        ignore_index=True,
    )

    summary = summarize(detail)

    detail.to_csv(
        RESULTS / "07_04c7_I4_fixed_point_detail.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS / "07_04c7_I4_fixed_point_summary.csv",
        index=False,
    )

    print()
    print("-" * 124)
    print("SUMMARY")
    print("-" * 124)
    print(summary.to_string(index=False))

    global_max_D = float(
        detail["trace_distance_to_I4"].max()
    )
    global_max_entry = float(
        detail["max_abs_entry_diff_to_I4"].max()
    )

    print()
    print("=" * 124)
    print("DECISION")
    print("=" * 124)
    print(f"Global max trace distance D(E_u[I/4], I/4) = {global_max_D:.12e}")
    print(f"Global max matrix-entry difference         = {global_max_entry:.12e}")
    print()

    tol = 1e-10

    if global_max_D <= tol:
        verdict = (
            "I/4 is effectively a fixed point of the implemented memory update "
            "for every tested real, randomized, and extreme input. Starting CONT "
            "from I/4 therefore traps the memory at I/4 and explains why CONT and "
            "W1 produced identical XZ_injection features."
        )
    else:
        verdict = (
            "I/4 is NOT a fixed point for all tested inputs. The previous CONT/W1 "
            "equality must therefore be explained by another implementation or "
            "trajectory effect."
        )

    print("VERDICT:")
    print(verdict)

    with open(
        RESULTS / "07_04c7_I4_fixed_point_summary.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "WEEK 7 STEP 7.4C.7 - I/4 FIXED-POINT AUDIT\n"
        )
        fp.write("=" * 90 + "\n\n")
        fp.write(summary.to_string(index=False))
        fp.write("\n\n")
        fp.write(
            f"global max trace distance = {global_max_D:.12e}\n"
        )
        fp.write(
            f"global max entry difference = {global_max_entry:.12e}\n\n"
        )
        fp.write("VERDICT:\n")
        fp.write(verdict + "\n")

    print()
    print("Saved:")
    print("  results/07_04c7_I4_fixed_point_detail.csv")
    print("  results/07_04c7_I4_fixed_point_summary.csv")
    print("  results/07_04c7_I4_fixed_point_summary.txt")


if __name__ == "__main__":
    main()
