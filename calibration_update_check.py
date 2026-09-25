from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from ibm_account import get_service

service = get_service()
backend = service.backend("ibm_kingston", use_fractional_gates=False)

local_tz = ZoneInfo("Europe/Bratislava")

# Start from now and walk backwards through calibration snapshots
cursor = datetime.now(timezone.utc)

print("Recent ibm_kingston calibration snapshots")
print("-" * 80)

seen = set()

for i in range(15):
    props = backend.properties(datetime=cursor)

    if props is None:
        break

    ts = props.last_update_date

    # Ensure timezone-aware UTC if necessary
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    key = ts.isoformat()
    if key in seen:
        break
    seen.add(key)

    print(
        f"{i+1:2d}. "
        f"UTC: {ts.astimezone(timezone.utc):%Y-%m-%d %H:%M:%S %Z}   "
        f"Slovakia: {ts.astimezone(local_tz):%Y-%m-%d %H:%M:%S %Z}"
    )

    # Ask for a time immediately before this calibration
    cursor = ts - timedelta(microseconds=1)