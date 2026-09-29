#!/usr/bin/env python
from __future__ import annotations

"""
12.6A.2a — Diagnose failed IQM Candidate-1 QPU probe.

NO QPU submission.
Retrieves the already-failed Emerald job and prints/saves the actual IQM
server errors, messages, timeline and compilation calibration ID.

Also updates the existing qrc_qpu_run row with the detailed failure metadata.
"""

import json
from pathlib import Path
from uuid import UUID

from iqm_account import get_client
import db_objects as qpu_db

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

JOB_ID = "01a0edd7-932e-70fe-ad38-726f59169863"
RUN_UUID = "15543bd9-a60b-4266-adec-8b48e7ec1e6b"
BACKEND_NAME = "emerald"

OUT = RESULTS / "12_06a2a_c1_failed_job_diagnostic.json"


def safe(x):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if hasattr(x, "model_dump"):
        try:
            return safe(x.model_dump(mode="json"))
        except Exception:
            try:
                return safe(x.model_dump())
            except Exception:
                pass
    if hasattr(x, "__dict__"):
        try:
            return {
                str(k): safe(v)
                for k, v in vars(x).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass
    return str(x)


def main():
    print("=" * 120)
    print("12.6A.2a — IQM FAILED-JOB DIAGNOSTIC")
    print("=" * 120)
    print("NO QPU SUBMISSION.")
    print(f"Backend : {BACKEND_NAME}")
    print(f"Job ID  : {JOB_ID}")
    print(f"Run UUID: {RUN_UUID}")
    print()

    client = get_client(quantum_computer=BACKEND_NAME)
    job = client.get_job(UUID(JOB_ID))

    status = job.update()
    data = job.data

    compilation = getattr(data, "compilation", None)
    calibration_set_id = (
        str(getattr(compilation, "calibration_set_id", None))
        if compilation is not None
        else None
    )

    payload = {
        "backend": BACKEND_NAME,
        "job_id": JOB_ID,
        "run_uuid": RUN_UUID,
        "status": str(status),
        "errors": safe(getattr(data, "errors", None)),
        "messages": safe(getattr(data, "messages", None)),
        "timeline": safe(getattr(data, "timeline", None)),
        "compilation": safe(compilation),
        "calibration_set_id": calibration_set_id,
        "raw_job_data": safe(data),
        "qpu_resubmitted": False,
    }

    OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("STATUS")
    print("-" * 120)
    print(payload["status"])
    print()

    print("ERRORS")
    print("-" * 120)
    print(json.dumps(payload["errors"], indent=2, ensure_ascii=False))
    print()

    print("MESSAGES")
    print("-" * 120)
    print(json.dumps(payload["messages"], indent=2, ensure_ascii=False))
    print()

    print("TIMELINE")
    print("-" * 120)
    print(json.dumps(payload["timeline"], indent=2, ensure_ascii=False))
    print()

    print("COMPILATION CALIBRATION SET")
    print("-" * 120)
    print(calibration_set_id)
    print()

    try:
        qpu_db.update_run(
            RUN_UUID,
            run_status="FAILED_DIAGNOSED",
            job_status=str(status),
            error_message=json.dumps(
                {
                    "errors": payload["errors"],
                    "messages": payload["messages"],
                },
                ensure_ascii=False,
            ),
            job_metrics_json=payload,
            notes=(
                "IQM Candidate-1 training-only QPU probe failed after submission. "
                "12.6A.2a retrieved detailed IQM server errors/messages/timeline. "
                "No resubmission performed."
            ),
        )
        qpu_db.save_json_artifact(
            RUN_UUID,
            OUT.name,
            payload,
        )
        print("DB diagnostic update: PASS")
    except Exception as exc:
        print(f"DB diagnostic update: WARNING — {exc}")

    print()
    print(f"Saved: {OUT}")
    print("STOP HERE. Send the ERRORS/MESSAGES section before any resubmission.")


if __name__ == "__main__":
    main()
