# QRC daily dataset — methodology & caveats

**File:** `qrc_daily_dataset.csv` — 2069 rows, one per calendar day, 2021-01-01 to 2026-08-31.
**Scope:** PZP risk only, passenger cars only (`D_Vehicle[kind]="Osobný automobil"`), package ≠ "5 - AAA". Same filter scope as the earlier monthly QRC dataset.
**Source:** Power BI semantic model `Risks&Claims_2_ACC` (artifact `3704b62f-0c16-449f-ad02-fe415b3674e0`).

## Columns

| Column | Meaning |
|---|---|
| `date` | Calendar day (YYYY-MM-DD) |
| `day_of_week` | ISO day-of-week, 1=Monday..7=Sunday |
| `day_of_week_name` | English weekday name |
| `is_weekend` | 1 if Saturday/Sunday |
| `is_public_holiday` | 1 if a Slovak public holiday (fixed-date + Easter-based, hardcoded 2021-2026) |
| `is_working_day` | 1 if not weekend and not a public holiday |
| `claim_count_total/_maj/_nem/_usly` | Distinct claims occurring that day, by peril (property / injury / lost-profit-MV), from `F_claim`, PZP+passenger+non-AAA scope |
| `policy_live_total`, `policy_age0/age1to3/age4plus`, `share_age0/age1to3/age4plus` | Live policy count and age-bucket breakdown (see caveats) |
| `legal_entity_count`, `share_legal_entity` | Count/share of live policies with `D_Policy[party_type]` = "PO" or "FOP" |
| `vehicle_avg_age_years` | Average vehicle age among live policies |

## How claim counts were computed

Daily distinct-claim counts pulled directly from Power BI via DAX (`GROUPBY`+`CURRENTGROUP()` for correct per-day context transition — an earlier `SUMMARIZE`+`ADDCOLUMNS(...,CALCULATE(...))` attempt silently returned the grand total on every row instead of a per-day count; `GROUPBY` avoided this). Injury/Ušlý-zisk counts use `DISTINCTCOUNT` filtered to the relevant `reserve_type` values; property (`claim_count_maj`) is the residual (total − injury − ušlý zisk), following the same convention used elsewhere in the MTPL report. Pulled in 3 date-range chunks (server enforces a hard ~1000-row cap per query regardless of the requested `maxRows`); chunk boundaries verified to be contiguous with no gaps or overlaps (1000+1000+55 = 2055 days with ≥1 claim, out of 2069 total days). Days with zero claims are absent from the raw pull and filled with 0 here.

## How policy-age / vehicle-age were computed — important caveats

**The original monthly dataset's exact DAX query was not preserved**, so this daily version is a best-effort reconstruction, not a guaranteed match to previously published monthly figures:

- **Liveness/age window**: a policy counts as "live" on day D if `D_Policy[startdate] <= D <= D_Policy[enddate]`. Age bucket (0 / 1–3 / 4+ years) is `DATEDIFF(startdate, D, YEAR)`.
- **Known mismatch**: reconstructing this way for 2026-08-31 gives **225,486 live policies**, vs. **312,895** in the previously published monthly dataset for the same month — a ~28% gap. Root cause not identified (possibly a different liveness definition, a broader vehicle-kind scope in the original figure, or something not visible in the semantic model as queried). The user chose to proceed with this reconstruction rather than block on resolving it — treat the *trend* as informative but the *absolute level* as not necessarily consistent with the earlier monthly file.
- **January sawtooth artifact**: DAX's `DATEDIFF(...,YEAR)` counts calendar-year boundaries crossed, not 365-day anniversaries. So a policy's age bucket advances every January 1st regardless of its actual inception month, producing a visible sawtooth in `age1to3`/`age4plus` each January. This is a modeling convention, not a data error.
- **Daily granularity**: policy-age/vehicle-age figures were only computed as **68 monthly (month-end) snapshots** via Power BI (pulling raw per-policy rows hit the same ~1000-row cap and would have needed ~225 separate calls). Daily values in the CSV are **linearly interpolated** between month-end snapshots. Days before the first snapshot (2021-01-31) or after the last (2026-08-31) are flat-filled from the nearest anchor — not applicable here since the date range starts/ends exactly at anchor points except the first 30 days of January 2021, which are flat-filled from the 2021-01-31 snapshot.

## Legal entity classification

`D_Policy[party_type]` values found: `FO` (individual), `PO` (legal entity), `FOP` (self-employed), and blank. `legal_entity_count` = PO + FOP.
