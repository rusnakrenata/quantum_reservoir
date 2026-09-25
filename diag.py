from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from ibm_account import get_service

service = get_service()
backend = service.backend("ibm_kingston", use_fractional_gates=False)

tz = ZoneInfo("Europe/Bratislava")

times = [
    datetime(2026, 9, 15, 11, 29, 27, tzinfo=timezone.utc),  # run 22
    datetime(2026, 9, 15, 13, 25, 26, tzinfo=timezone.utc),  # run 26
    datetime(2026, 9, 15, 13, 46, 29, tzinfo=timezone.utc),  # run 27
]

for t in times:
    p = backend.properties(datetime=t)
    print()
    print("RUN UTC:   ", t)
    print("RUN LOCAL: ", t.astimezone(tz))
    print("CAL RAW:   ", p.last_update_date)
    print("CAL LOCAL: ", p.last_update_date.astimezone(tz))