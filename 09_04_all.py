
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(script: str, extra: list[str] | None = None):
    cmd = [
        sys.executable,
        str(HERE / script),
    ]

    if extra:
        cmd.extend(extra)

    print()
    print("=" * 120)
    print("RUNNING:", " ".join(cmd))
    print("=" * 120)

    subprocess.run(
        cmd,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-h4",
        action="store_true",
    )
    parser.add_argument(
        "--windows",
        default="1-28",
    )
    parser.add_argument(
        "--local-budget",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--robustness-perturbations",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    a_extra = [
        "--windows",
        args.windows,
        "--overwrite",
    ]

    if args.include_h4:
        a_extra.append("--include-h4")

    run(
        "09_04a_rwp_transfer_matrix.py",
        a_extra,
    )

    run(
        "09_04b_rwp_pareto_shortlist.py",
    )

    run(
        "09_04c_rwp_hyperparameter_reopt.py",
        ["--overwrite"],
    )

    run(
        "09_04c2_rwp_trotter_reopt.py",
        [
            "--overwrite",
            "--local-budget",
            str(args.local_budget),
        ],
    )

    run(
        "09_04d_rwp_robustness.py",
        [
            "--overwrite",
            "--perturbations",
            str(args.robustness_perturbations),
        ],
    )

    print()
    print("=" * 120)
    print("WEEK 9.04 RWP PIPELINE COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
