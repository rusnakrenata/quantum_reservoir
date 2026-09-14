
from __future__ import annotations

SCRIPT_VERSION = "09.05C-V2-reconstruct-J"

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

from ibm_account import get_service

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

COMMON_FILE = HERE / "09_04_rwp_common.py"
SELECTOR_FILE = HERE / "11_0A_live_embedding_reselection.py"
TRIAL_FILE = RESULTS / "09_04c_rwp_all_trials.csv"
EXPECTED_TOPOLOGIES = ["H0", "H1", "H2", "H3"]

BACKENDS = ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]
N_FOLDS = 5
OPT_LEVEL = 1
SEED_TRANSPILE = 42


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


common = load_module(COMMON_FILE, "qrc_0905c_common")
selector = load_module(SELECTOR_FILE, "qrc_0905c_selector")


def reconstruct_stage_c_candidate(row, baselines):
    """
    Rebuild the exact Stage-C candidate from the clean Week-9.3c topology
    baseline and deterministic Stage-C trial_search_id.

    Do not reconstruct H1/H2/H3 from the topology-specific J columns in
    09_04c_rwp_all_trials.csv, because that CSV was appended using a header
    established by H0 and therefore does not safely preserve J35/J24/J04.
    """
    H = str(row["topology"])
    W = int(row["window"])
    search_id = str(row["trial_search_id"])

    if H not in baselines:
        raise KeyError(f"No clean 9.3c baseline available for topology {H}.")

    base = common._clone_candidate(baselines[H])
    base["candidate_id"] = f"{H}_W{W:02d}_Aparent"
    base["window"] = W

    candidates = common.stage_c_candidates(base)
    by_id = {str(c["search_id"]): c for c in candidates}

    if search_id not in by_id:
        raise KeyError(
            f"Could not reconstruct Stage-C search_id={search_id!r} "
            f"for {H} W={W:02d}."
        )

    candidate = by_id[search_id]

    # Audit scalar dynamics against the Stage-C result row.
    checks = {
        "alpha": float(candidate["alpha"]),
        "dt": float(candidate["dt"]),
        "r": int(candidate["r"]),
        "hx": float(candidate["hx"]),
        "hy": float(candidate["hy"]),
    }
    for key, expected in checks.items():
        observed = float(row[key])
        if key == "r":
            ok = int(observed) == int(expected)
        else:
            ok = np.isclose(observed, expected, rtol=1e-10, atol=1e-12)
        if not ok:
            raise RuntimeError(
                f"Stage-C reconstruction audit failed for {H} W={W:02d} "
                f"{search_id}: {key} CSV={observed}, reconstructed={expected}."
            )

    return candidate


def select_rule4(trials):
    rows = []
    for H, group in trials.groupby("topology"):
        best = group.sort_values("cv_rmse").iloc[0]
        threshold = float(best["cv_rmse"]) + float(best["cv_rmse_std"]) / math.sqrt(N_FOLDS)

        eligible = group[group["cv_rmse"] <= threshold].copy()

        pick = eligible.sort_values(
            [
                "feature_vector_cz",
                "n_settings",
                "logical_reset_count",
                "n_features",
                "cv_rmse",
                "shot_noise_forecast_sd_proxy_1024",
            ]
        ).iloc[0].copy()

        pick["one_se_threshold"] = threshold
        rows.append(pick)

    return pd.DataFrame(rows).reset_index(drop=True)


def append_rwp_step(qc, candidate, angle_row, first_step):
    if not first_step:
        for q in range(4):
            qc.reset(q)

    for q in range(4):
        qc.ry(float(angle_row[q]), q)

    r = int(candidate["r"])
    dt = float(candidate["dt"])
    hx = float(candidate["hx"])
    hy = float(candidate["hy"])
    J = candidate["J"]

    for _ in range(r):
        for i, j in common.TOPOLOGY_EDGES[candidate["topology"]]:
            qc.rzz(
                2.0 * float(J[(i, j)]) * dt / r,
                i,
                j,
            )

        theta_x = 2.0 * hx * dt / r
        for q in range(6):
            qc.rx(theta_x, q)

        if not np.isclose(hy, 0.0):
            theta_y = 2.0 * hy * dt / r
            qc.ry(theta_y, 4)
            qc.ry(theta_y, 5)


def readout_settings(readout):
    if readout == "XZ_injection":
        return [
            ("XXXX", {0:"X",1:"X",2:"X",3:"X"}, [0,1,2,3]),
            ("ZZZZ", {0:"Z",1:"Z",2:"Z",3:"Z"}, [0,1,2,3]),
        ]
    if readout == "XZinj_dropX3":
        return [
            ("XXXZ", {0:"X",1:"X",2:"X",3:"Z"}, [0,1,2,3]),
            ("ZZZZ", {0:"Z",1:"Z",2:"Z",3:"Z"}, [0,1,2,3]),
        ]
    if readout == "XZinj_dropX3_plus_YX45":
        return [
            ("XXXZYX", {0:"X",1:"X",2:"X",3:"Z",4:"Y",5:"X"}, [0,1,2,3,4,5]),
            ("ZZZZZZ", {0:"Z",1:"Z",2:"Z",3:"Z",4:"Z",5:"Z"}, [0,1,2,3,4,5]),
        ]
    if readout == "XZinj_plus_YX45":
        return [
            ("XXXXYX", {0:"X",1:"X",2:"X",3:"X",4:"Y",5:"X"}, [0,1,2,3,4,5]),
            ("ZZZZZZ", {0:"Z",1:"Z",2:"Z",3:"Z",4:"Z",5:"Z"}, [0,1,2,3,4,5]),
        ]
    if readout == "XYZ_all":
        return [
            ("XXXXXX", {q:"X" for q in range(6)}, list(range(6))),
            ("YYYYYY", {q:"Y" for q in range(6)}, list(range(6))),
            ("ZZZZZZ", {q:"Z" for q in range(6)}, list(range(6))),
        ]
    raise ValueError(readout)


def build_rwp_measurement_circuit(candidate, all_angles, endpoint, window, setting):
    label, basis, measured = setting

    qreg = QuantumRegister(6, "q")
    creg = ClassicalRegister(len(measured), "m")
    qc = QuantumCircuit(qreg, creg, name=f"{candidate['topology']}_W{window}_{endpoint}_{label}")

    start = int(endpoint) - int(window) + 1
    if start < 0:
        raise ValueError("Endpoint does not have enough rewind history.")

    for k, idx in enumerate(range(start, int(endpoint) + 1)):
        append_rwp_step(
            qc,
            candidate,
            all_angles[idx],
            first_step=(k == 0),
        )

    for q, axis in basis.items():
        if axis == "X":
            qc.h(q)
        elif axis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif axis == "Z":
            pass
        else:
            raise ValueError(axis)

    for c, q in enumerate(measured):
        qc.measure(q, c)

    return qc


def resource_row(circuit, backend, meta):
    ops = {str(k): int(v) for k, v in circuit.count_ops().items()}
    duration_s = float(circuit.estimate_duration(backend.target, unit="s"))

    n2q = 0
    for gate, count in ops.items():
        try:
            if circuit.num_nonlocal_gates() >= 0:
                pass
        except Exception:
            pass

    # On the current Heron compilation the nonlocal gate is CZ; SWAP is also
    # explicitly recorded if ever introduced.
    n2q = int(ops.get("cz", 0) + ops.get("swap", 0))

    return {
        **meta,
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "n_cz": int(ops.get("cz", 0)),
        "n_swap": int(ops.get("swap", 0)),
        "n_2q_reported": n2q,
        "n_reset": int(ops.get("reset", 0)),
        "n_measure": int(ops.get("measure", 0)),
        "duration_us": duration_s * 1e6,
        "operations": json.dumps(ops, sort_keys=True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="all", choices=["all", *BACKENDS])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--shortlist", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not TRIAL_FILE.exists():
        raise FileNotFoundError(TRIAL_FILE)

    trials = pd.read_csv(TRIAL_FILE)
    reps = select_rule4(trials)

    # Exact H0-H3 Stage-C dynamics are reconstructed from the clean 9.3c
    # topology baselines + deterministic trial_search_id.
    baselines = common.load_topology_baselines(EXPECTED_TOPOLOGIES)

    reps.to_csv(
        RESULTS / "09_05c_rwp_rule4_selected_before_compile.csv",
        index=False,
    )

    work_tv, cols, y = common.load_train_validation()
    service = get_service()

    backend_names = BACKENDS if args.backend == "all" else [args.backend]

    detailed = []
    mapping_rows = []

    print("=" * 120)
    print(f"09.05C — RWP RULE-4 PHYSICAL COMPILATION + FRESH HARDWARE EMBEDDING [{SCRIPT_VERSION}]")
    print("=" * 120)
    print("NO QPU JOBS ARE SUBMITTED.")
    print("Exact H0-H3 Stage-C couplings are reconstructed from the clean 9.3c baseline + trial_search_id.")
    print("The unsafe topology-specific J columns in 09_04c_rwp_all_trials.csv are not used to rebuild dynamics.")
    print()

    for _, row in reps.iterrows():
        candidate = reconstruct_stage_c_candidate(row, baselines)
        H = candidate["topology"]
        W = int(row["window"])
        readout = str(row["readout"])

        angles = common.make_angles(work_tv, cols, float(candidate["alpha"]))

        valid_endpoints = np.arange(common.N_TRAIN, common.N_TRAIN + common.N_VAL)
        sample_idx = np.linspace(
            0, len(valid_endpoints) - 1,
            min(args.samples, len(valid_endpoints)),
            dtype=int,
        )
        endpoints = valid_endpoints[sample_idx]

        for backend_name in backend_names:
            fresh = selector.fresh_hardware_reselection(
                service=service,
                candidate=candidate,
                backend_names=[backend_name],
                shortlist=args.shortlist,
                write_prefix=f"09_05c_{H}_{backend_name}",
                verbose=False,
            )

            chosen = fresh["selected"]
            layout = json.loads(chosen["layout"])

            mapping_rows.append(
                {
                    "topology": H,
                    "window": W,
                    "readout": readout,
                    "backend": backend_name,
                    "layout": chosen["layout"],
                    "physical_C_t": chosen.get("physical_C_t"),
                    "physical_D": chosen.get("physical_D"),
                    "physical_P_t": chosen.get("physical_P_t"),
                    "physical_H": chosen.get("physical_H"),
                    "physical_M1": chosen.get("physical_M1"),
                    "physical_M2": chosen.get("physical_M2"),
                    "compiled_2q_error_max_percent": chosen.get("compiled_2q_error_max_percent"),
                    "compiled_1q_error_max_percent": chosen.get("compiled_1q_error_max_percent"),
                    "max_readout_error_percent": chosen.get("max_readout_error_percent"),
                    "min_t1_us": chosen.get("min_t1_us"),
                    "min_t2_us": chosen.get("min_t2_us"),
                    "compiled_duration_us_core": chosen.get("compiled_duration_us"),
                    "pending_jobs": chosen.get("pending_jobs"),
                }
            )

            backend = service.backend(
                backend_name,
                use_fractional_gates=False,
            )
            try:
                backend.refresh()
            except Exception:
                pass

            for endpoint in endpoints:
                setting_rows = []
                for setting in readout_settings(readout):
                    logical = build_rwp_measurement_circuit(
                        candidate,
                        angles,
                        int(endpoint),
                        W,
                        setting,
                    )

                    isa = transpile(
                        logical,
                        backend=backend,
                        initial_layout=layout,
                        routing_method="none",
                        optimization_level=OPT_LEVEL,
                        seed_transpiler=SEED_TRANSPILE,
                        scheduling_method="alap",
                    )

                    backend.check_faulty(isa)

                    rr = resource_row(
                        isa,
                        backend,
                        {
                            "topology": H,
                            "window": W,
                            "readout": readout,
                            "r": int(candidate["r"]),
                            "backend": backend_name,
                            "endpoint": int(endpoint),
                            "setting": setting[0],
                        },
                    )

                    if rr["n_swap"] != 0:
                        raise RuntimeError(
                            f"{H} {backend_name}: SWAP detected in strict native compile."
                        )

                    detailed.append(rr)
                    setting_rows.append(rr)

            print(
                f"{H} W={W:02d} {readout:28s} {backend_name}: "
                f"layout={layout}"
            )

    detail_df = pd.DataFrame(detailed)
    map_df = pd.DataFrame(mapping_rows)

    detail_df.to_csv(
        RESULTS / "09_05c_rwp_rule4_compile_detail.csv",
        index=False,
    )
    map_df.to_csv(
        RESULTS / "09_05c_rule5_fresh_embedding_metrics.csv",
        index=False,
    )

    # Build feature-vector resources by summing separate settings for each endpoint.
    per_endpoint = (
        detail_df.groupby(
            ["topology", "window", "readout", "r", "backend", "endpoint"],
            dropna=False,
        )
        .agg(
            n_settings=("setting", "nunique"),
            N2q_feature_vector=("n_2q_reported", "sum"),
            NCZ_feature_vector=("n_cz", "sum"),
            NSWAP_feature_vector=("n_swap", "sum"),
            depth_max_setting=("depth", "max"),
            depth_sum_settings=("depth", "sum"),
            Tcircuit_feature_vector_us=("duration_us", "sum"),
            Tcircuit_max_setting_us=("duration_us", "max"),
            resets_sum_settings=("n_reset", "sum"),
        )
        .reset_index()
    )

    summary = (
        per_endpoint.groupby(
            ["topology", "window", "readout", "r", "backend"],
            dropna=False,
        )
        .agg(
            sampled_endpoints=("endpoint", "nunique"),
            n_settings=("n_settings", "median"),
            N2q_median=("N2q_feature_vector", "median"),
            NCZ_median=("NCZ_feature_vector", "median"),
            NSWAP_median=("NSWAP_feature_vector", "median"),
            depth_max_setting_median=("depth_max_setting", "median"),
            depth_sum_settings_median=("depth_sum_settings", "median"),
            Tcircuit_feature_vector_us_median=("Tcircuit_feature_vector_us", "median"),
            Tcircuit_max_setting_us_median=("Tcircuit_max_setting_us", "median"),
            resets_feature_vector_median=("resets_sum_settings", "median"),
        )
        .reset_index()
    )

    per_endpoint.to_csv(
        RESULTS / "09_05c_rwp_rule4_compile_per_endpoint.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS / "09_05c_rwp_rule4_compile_summary.csv",
        index=False,
    )

    print()
    print("PHYSICAL RESOURCE SUMMARY")
    print(summary.to_string(index=False))
    print()
    print("Saved:")
    print("  results/09_05c_rwp_rule4_selected_before_compile.csv")
    print("  results/09_05c_rwp_rule4_compile_detail.csv")
    print("  results/09_05c_rwp_rule4_compile_per_endpoint.csv")
    print("  results/09_05c_rwp_rule4_compile_summary.csv")
    print("  results/09_05c_rule5_fresh_embedding_metrics.csv")


if __name__ == "__main__":
    main()
