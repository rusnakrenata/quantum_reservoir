#!/usr/bin/env python
"""
10.05A — Fresh current Kingston layouts for H0-H3
==================================================

Purpose
-------
The Rule-1..Rule-4 candidate manifest is the reservoir-candidate source.
The historical Rule-5 layouts stored in that manifest/article are NOT used.

Immediately before the long 10.05 noisy-candidate experiment, this script
performs a fresh Kingston hardware reselection for H0, H1, H2 and H3 using
the project's established fresh-reselection helper:

    11_0A_live_embedding_reselection.py

The result is a CURRENT topology-specific Kingston physical layout selected
under the same strict native zero-SWAP / Rule-5 hardware logic already used
in Week 9/10.

Why one layout per topology?
----------------------------
Rule 5 is a physical-topology layer, not a CONT/RWP winner rule.  A topology's
physical region should therefore be shared by all Rule-1..Rule-4 candidates
with that logical topology during one comparison.  This prevents candidate
comparisons from being confounded by giving every candidate a separately
optimized physical region.

The selected layout is subsequently verified again when each r=2/3/4 circuit
is compiled in 10.05.

No QPU jobs are submitted.

Required files
--------------
09_04_rwp_common.py
11_0A_live_embedding_reselection.py
ibm_account.py
results/10_05_rule1_to_rule5_candidate_manifest.csv

Outputs
-------
results/10_05_current_kingston_layouts.csv
results/10_05_current_kingston_layouts.json

Run
---
python 10_05A_refresh_current_kingston_layouts.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ibm_account import get_service


HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

MANIFEST_FILE = RESULTS / "10_05_rule1_to_rule5_candidate_manifest.csv"
COMMON_FILE = HERE / "09_04_rwp_common.py"
SELECTOR_FILE = HERE / "11_0A_live_embedding_reselection.py"

BACKEND_NAME = "ibm_kingston"
TOPOLOGIES = ["H0", "H1", "H2", "H3"]
DEFAULT_SHORTLIST = 80


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load {path}")

    spec.loader.exec_module(mod)
    return mod


def parse_layout(value):
    if isinstance(value, str):
        return [int(x) for x in json.loads(value)]
    return [int(x) for x in value]


def normalized_selected_row(topology, selected):
    layout = parse_layout(selected["layout"])

    return {
        "topology": topology,
        "backend": BACKEND_NAME,
        "layout": json.dumps(layout),
        "physical_C_t": int(layout[0]),
        "physical_D": int(layout[1]),
        "physical_P_t": int(layout[2]),
        "physical_H": int(layout[3]),
        "physical_M1": int(layout[4]),
        "physical_M2": int(layout[5]),
        "compiled_2q_error_max_percent": float(
            selected["compiled_2q_error_max_percent"]
        ),
        "max_readout_error_percent": float(
            selected["max_readout_error_percent"]
        ),
        "compiled_1q_error_max_percent": float(
            selected["compiled_1q_error_max_percent"]
        ),
        "min_t1_us": float(selected["min_t1_us"]),
        "min_t2_us": float(selected["min_t2_us"]),
        "compiled_duration_us_core": float(
            selected["compiled_duration_us"]
        ),
        "compiled_n_cz": int(selected.get("compiled_n_cz", 0)),
        "compiled_n_swap": int(selected.get("compiled_n_swap", 0)),
        "strict_compile_pass": bool(
            selected.get("strict_compile_pass", True)
        ),
        "selection_source": "fresh_current_kingston_reselection",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shortlist",
        type=int,
        default=DEFAULT_SHORTLIST,
    )

    args = parser.parse_args()

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"{MANIFEST_FILE} not found. "
            "Save the 25-candidate Rule-1..Rule-4 manifest into results/ first."
        )

    manifest = pd.read_csv(MANIFEST_FILE)

    found = sorted(set(manifest["topology"].astype(str)))
    expected = sorted(TOPOLOGIES)

    if found != expected:
        raise RuntimeError(
            f"Manifest topology audit failed: found={found}, expected={expected}"
        )

    print("=" * 118)
    print("10.05A — FRESH CURRENT KINGSTON PHYSICAL LAYOUTS")
    print("=" * 118)
    print("Historical Rule-5 layouts from the article/manifest are IGNORED.")
    print("A fresh current Kingston scan is performed now.")
    print("NO QPU JOBS ARE SUBMITTED.")
    print()

    common = load_module(COMMON_FILE, "qrc_1005a_common")
    selector = load_module(SELECTOR_FILE, "qrc_1005a_selector")

    # The clean Week-9 topology baselines provide a representative physical
    # core for strict compilation during the topology-level hardware scan.
    baselines = common.load_topology_baselines(TOPOLOGIES)

    service = get_service()

    rows = []
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": BACKEND_NAME,
        "manifest": str(MANIFEST_FILE),
        "historical_rule5_layouts_used": False,
        "shortlist": int(args.shortlist),
        "topologies": {},
    }

    for topology in TOPOLOGIES:
        candidate = common._clone_candidate(
            baselines[topology]
        )

        print("-" * 118)
        print(f"FRESH KINGSTON SCAN: {topology}")
        print("-" * 118)

        # Keep the topology baseline dynamics only as the strict-compilation
        # reference.  The chosen physical region is then shared by all
        # candidates of this topology in 10.05.
        selector.ALPHA = float(candidate["alpha"])
        selector.DT = float(candidate["dt"])

        fresh = selector.fresh_hardware_reselection(
            service=service,
            candidate=candidate,
            backend_names=[BACKEND_NAME],
            shortlist=int(args.shortlist),
            write_prefix=f"10_05A_{topology}_kingston",
            verbose=True,
        )

        selected = fresh["selected"]
        row = normalized_selected_row(
            topology,
            selected,
        )

        if not row["strict_compile_pass"]:
            raise RuntimeError(
                f"{topology}: selected fresh layout did not pass strict compile."
            )

        if row["compiled_n_swap"] != 0:
            raise RuntimeError(
                f"{topology}: selected fresh layout contains SWAPs."
            )

        rows.append(row)
        audit["topologies"][topology] = row

        print()
        print(
            f"SELECTED {topology}: layout={row['layout']} | "
            f"max2Q={row['compiled_2q_error_max_percent']:.6f}% | "
            f"maxRO={row['max_readout_error_percent']:.6f}% | "
            f"max1Q={row['compiled_1q_error_max_percent']:.6f}% | "
            f"minT2={row['min_t2_us']:.3f} us | "
            f"SWAP={row['compiled_n_swap']}"
        )
        print()

    df = pd.DataFrame(rows).sort_values("topology")

    csv_path = RESULTS / "10_05_current_kingston_layouts.csv"
    json_path = RESULTS / "10_05_current_kingston_layouts.json"

    df.to_csv(csv_path, index=False)

    json_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("=" * 118)
    print("CURRENT KINGSTON LAYOUTS")
    print("=" * 118)
    print(
        df[
            [
                "topology",
                "layout",
                "compiled_2q_error_max_percent",
                "max_readout_error_percent",
                "compiled_1q_error_max_percent",
                "min_t1_us",
                "min_t2_us",
                "compiled_duration_us_core",
                "compiled_n_swap",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print()
    print(
        "Use these fresh layouts in 10.05; do not use the historical "
        "Rule-5 layout column from the candidate manifest."
    )


if __name__ == "__main__":
    main()
