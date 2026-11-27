"""Pure-Python helpers for the Lakeflow pipeline (src/02_lakeflow_etl.py).

Grid lookups are built from pricing_rules on the driver and joined in Spark, so hours and tiers have a single
source of truth. SQL snippets returned here are plain Spark SQL strings, usable in F.expr and expectations.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from datetime import date, time

from b4l import pricing_rules as pr

OPEN_DAY_TYPES = ("weekday", "weekend_holiday")

BOOKING_STATUSES = ("completed", "upcoming", "cancelled_early", "cancelled_late", "no_show")
BILLED_STATUSES = ("completed", "cancelled_late", "no_show")
BILLED_OR_UPCOMING = BILLED_STATUSES + ("upcoming",)
EVENT_TYPES = ("created", "cancelled", "no_show", "checked_in")

# Events reaching the lakehouse later than this are flagged as late-arriving (ASSUMPTION).
LATE_ARRIVAL_HOURS = 6
# Churn rule: a regular averages >= CHURN_MIN_MONTHLY_PLAYS per 30 days over days 31-120 before the snapshot;
# the label is true when plays in the last 30 days are <= CHURN_DROP_RATIO x that baseline.
CHURN_MIN_MONTHLY_PLAYS = 2.0
CHURN_DROP_RATIO = 0.35
CHURN_BASELINE_DAYS = (31, 120)

# No backslashes: Spark SQL string literals would unescape them before the regex engine sees them.
EMAIL_PATTERN = "^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+([.][A-Za-z0-9-]+)*[.][A-Za-z]{2,}$"
PHONE_PATTERN = "^[+]?[0-9][0-9 ]{8,16}[0-9]$"

HOUR_GRID_DDL = (
    "day_type STRING, weekday0 INT, hour INT, avail_start_minute INT, available_minutes INT, tier STRING"
)
SLOT_GRID_DDL = "day_type STRING, weekday0 INT, start_minute INT, tier STRING, base_price_chf DOUBLE"


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _time(minute_of_day: int) -> time:
    return time(minute_of_day // 60, minute_of_day % 60)


def hour_buckets(day_type: str, weekday0: int) -> list[tuple]:
    """Clock-hour buckets of the rentable window, matching HOUR_GRID_DDL.

    A bucket that starts before opening (weekend 09:00 when courts open at 09:30) only counts the rentable
    minutes, and its tier is taken at the first rentable minute.
    """
    window = pr.rentable_window(day_type)
    if window is None:
        return []
    open_m, close_m = (_minutes(t) for t in window)
    rows = []
    for hour in range(open_m // 60, math.ceil(close_m / 60)):
        seg_start = max(hour * 60, open_m)
        seg_end = min((hour + 1) * 60, close_m)
        tier = pr.tier_for(_time(seg_start), weekday0, day_type)
        rows.append((day_type, weekday0, hour, seg_start, seg_end - seg_start, tier))
    return rows


def hour_grid_rows() -> list[tuple]:
    """Hour buckets for every open day type and weekday (holidays can fall on any weekday)."""
    return [row for dt in OPEN_DAY_TYPES for wd in range(7) for row in hour_buckets(dt, wd)]


def slot_grid_rows() -> list[tuple]:
    """Valid 1-hour booking starts for every open day type and weekday, matching SLOT_GRID_DDL."""
    rows = []
    for dt in OPEN_DAY_TYPES:
        for wd in range(7):
            for start in pr.start_times(dt):
                tier = pr.tier_for(start, wd, dt)
                rows.append((dt, wd, _minutes(start), tier, pr.TIER_PRICE_CHF[tier]))
    return rows


def require_hmac_key(key: str | None) -> str:
    """Reject a missing, short or unresolved ({{secrets/...}}) pseudonymisation key."""
    if not key or key.startswith("{{secrets/") or len(key) < 32:
        raise ValueError("Pseudonymisation key is missing or unresolved; read it with dbutils.secrets.get")
    return key


def hmac_hex(key: str, value: str | None) -> str | None:
    """Pseudonymised player id: hex HMAC-SHA256 of the booking-system id."""
    if value is None:
        return None
    return hmac.new(key.encode("utf-8"), str(value).encode("utf-8"), hashlib.sha256).hexdigest()


# (inclusive upper age, label); anyone older than the last bound is OLDEST_AGE_BAND.
AGE_BANDS = ((17, "<18"), (25, "18-25"), (35, "26-35"), (50, "36-50"), (65, "51-65"))
OLDEST_AGE_BAND = "65+"


def age_band(birth_year: int | None, as_of: date) -> str | None:
    if birth_year is None:
        return None
    age = as_of.year - birth_year
    for upper, label in AGE_BANDS:
        if age <= upper:
            return label
    return OLDEST_AGE_BAND


def age_band_sql(age_expr: str) -> str:
    whens = " ".join(f"WHEN ({age_expr}) <= {upper} THEN '{label}'" for upper, label in AGE_BANDS)
    return f"CASE WHEN ({age_expr}) IS NULL THEN NULL {whens} ELSE '{OLDEST_AGE_BAND}' END"


# (exclusive upper start minute, label) for weekday starts; weekend/holiday days are one band.
TIME_BANDS = ((11 * 60 + 30, "morning"), (13 * 60 + 30, "lunch"), (17 * 60, "afternoon"))
LAST_TIME_BAND = "evening"
WEEKEND_TIME_BAND = "weekend"


def time_band(start_minute: int, day_type: str) -> str:
    if day_type == "weekend_holiday":
        return WEEKEND_TIME_BAND
    for upper, label in TIME_BANDS:
        if start_minute < upper:
            return label
    return LAST_TIME_BAND


def time_band_sql(start_minute_expr: str, day_type_expr: str) -> str:
    whens = " ".join(f"WHEN ({start_minute_expr}) < {upper} THEN '{label}'" for upper, label in TIME_BANDS)
    return (
        f"CASE WHEN ({day_type_expr}) = 'weekend_holiday' THEN '{WEEKEND_TIME_BAND}' "
        f"{whens} ELSE '{LAST_TIME_BAND}' END"
    )


def regex_match_sql(column: str, pattern: str) -> str:
    """Null-safe RLIKE: a missing value counts as invalid."""
    return f"coalesce({column} RLIKE '{pattern}', false)"


def _in_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


# Null-safe so that a row either passes or fails every rule (expectations and quarantine agree).
BOOKING_RULES = {
    "known_court": "is_known_court",
    "has_booker": "booker_uuid IS NOT NULL",
    "positive_duration": "duration_h IS NOT NULL AND duration_h > 0",
    "valid_party_size": "coalesce(party_size IN (2, 4), false)",
    "start_before_end": "start_ts IS NOT NULL AND end_ts IS NOT NULL AND start_ts < end_ts",
    "valid_status": f"coalesce(status IN ({_in_list(BOOKING_STATUSES)}), false)",
}


def violates_any_sql(rules: dict[str, str]) -> str:
    return "NOT (" + " AND ".join(f"({r})" for r in rules.values()) + ")"


def violation_reason_sql(rules: dict[str, str]) -> str:
    """Comma-separated names of the failed rules (concat_ws skips the NULLs of passed rules)."""
    cases = ", ".join(f"CASE WHEN NOT ({r}) THEN '{name}' END" for name, r in rules.items())
    return f"concat_ws(',', {cases})"


def in_list_sql(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({_in_list(values)})"
