from __future__ import annotations

"""
Check the remaining IBM Quantum usage allocation for the PRIMARY account.

Uses:
    ibm_account.py -> get_service()

Usage:
    python 11_check_qpu_remaining_time_FIXED.py

This script DOES NOT submit any QPU job.
"""

import json

from ibm_account import get_service


def main() -> None:
    print("=" * 90)
    print("IBM QUANTUM — CURRENT USAGE / REMAINING QPU TIME")
    print("=" * 90)

    service = get_service()

    try:
        usage = service.usage()
    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve IBM Quantum service usage.\n"
            f"Original error:\n{exc}"
        ) from exc

    if not usage:
        print("IBM returned no usage information for the active instance.")
        return

    print()
    print("Raw IBM usage response:")
    print(json.dumps(usage, indent=2, default=str))

    consumed = usage.get("usage_consumed_seconds")
    limit = usage.get("usage_limit_seconds")
    remaining = usage.get("usage_remaining_seconds")

    # Fallback if IBM gives consumed + limit but not remaining.
    if remaining is None and limit is not None and consumed is not None:
        remaining = max(
            0.0,
            float(limit) - float(consumed),
        )

    print()
    print("-" * 90)
    print("SUMMARY")
    print("-" * 90)

    print(
        f"Instance ID:     "
        f"{usage.get('instance_id', 'unknown')}"
    )
    print(
        f"Plan ID:         "
        f"{usage.get('plan_id', 'unknown')}"
    )

    if consumed is not None:
        consumed = float(consumed)
        print(
            f"Consumed:        "
            f"{consumed:.3f} s "
            f"({consumed / 60.0:.3f} min)"
        )

    if limit is not None:
        limit = float(limit)
        print(
            f"Allocation:      "
            f"{limit:.3f} s "
            f"({limit / 60.0:.3f} min)"
        )

    if remaining is not None:
        remaining = float(remaining)
        print(
            f"REMAINING:       "
            f"{remaining:.3f} s "
            f"({remaining / 60.0:.3f} min)"
        )

    if consumed is not None and limit not in (None, 0):
        percent_used = 100.0 * consumed / limit
        print(
            f"Used:            "
            f"{percent_used:.2f}%"
        )

    period = usage.get("usage_period") or {}

    if period:
        print(
            f"Period start:    "
            f"{period.get('start_time', 'unknown')}"
        )
        print(
            f"Period end:      "
            f"{period.get('end_time', 'unknown')}"
        )

    print(
        f"Limit reached:   "
        f"{usage.get('usage_limit_reached', 'unknown')}"
    )

    print("=" * 90)
    print()
    print(
        "Note: this reports IBM account/instance usage. "
        "It does not submit a QPU job."
    )


if __name__ == "__main__":
    main()
