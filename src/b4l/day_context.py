"""`get_day_context`: a read-only UC table function giving staff the weather, holidays and events for each day.

It reads the synthetic silver `weather` and `calendar` tables, so it is consistent with the bookings. The demand hints
restate the effects built into src/b4l/demand.py (rain, heat, Aréna events, public and school holidays, Bad'matin).
Each rule has a SQL condition (for the function) and a Python predicate (for tests and the 05 notebook's check).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from b4l import pricing_rules as pr

FUNCTION_NAME = "get_day_context"
MAX_DAYS = 31
HOT_DAY_C = 28.0
WARM_WEEKEND_C = 22.0
SUMMER_HOLIDAYS = "Vacances d'été"
CLOSED_HINT = "Centre closed"
NORMAL_HINT = "Normal day"
HINT_SEPARATOR = " · "

COLUMNS: list[tuple[str, str, str]] = [
    ("date", "DATE", "Day"),
    ("day_name", "STRING", "Monday..Sunday"),
    ("day_type", "STRING", "weekday, weekend_holiday (weekend hours) or closed"),
    ("opening_hours", "STRING", "Rentable hours, e.g. 08:00-22:00; NULL when closed"),
    ("condition", "STRING", "sunny, cloudy, rain or snow"),
    ("temp_mean_c", "DOUBLE", "Mean temperature °C"),
    ("temp_max_c", "DOUBLE", "Max temperature °C"),
    ("precip_mm", "DOUBLE", "Precipitation mm"),
    ("is_rainy", "BOOLEAN", "Rain day (precipitation >= 1 mm)"),
    ("is_forecast", "BOOLEAN", "True for upcoming days (forecast)"),
    ("public_holiday", "STRING", "Vaud public holiday name, or NULL"),
    ("school_holiday", "STRING", "Vaud school holiday period, or NULL"),
    ("arena_event", "BOOLEAN", "Evening event at the neighbouring Vaudoise Aréna"),
    ("demand_hint", "STRING", "What the day usually means for court demand"),
]

Row = Mapping[str, Any]


def _sql_str(s: str) -> str:
    """Spark SQL literal: a doubled quote would concatenate two literals, so escape with backslashes."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _f(row: Row, key: str) -> float | None:
    v = row.get(key)
    return None if v is None else float(v)


def _dry(row: Row) -> bool:
    """Known not rainy (SQL `NOT w.is_rainy` is NULL, so false, when the weather is missing)."""
    v = row.get("is_rainy")
    return v is not None and v == v and not bool(v)


def _rainy(row: Row) -> bool:
    v = row.get("is_rainy")
    return v is not None and v == v and bool(v)


def _weekend(row: Row) -> bool:
    return row["day_type"] == "weekend_holiday"


def _tuesday_weekday(row: Row) -> bool:
    return row["day_type"] == "weekday" and int(row["day_of_week"]) == 2


# (key, hint, SQL condition on calendar c and weather w, Python predicate on the same columns), in display order.
RULES: list[tuple[str, str, str, Callable[[Row], bool]]] = [
    ("public_holiday", "Public holiday: weekend hours, about 10% quieter than a normal weekend",
     "c.is_public_holiday",
     lambda r: bool(r["is_public_holiday"])),
    ("rain_weekend", "Rainy weekend: expect busier courts than usual (about +25%); players book earlier",
     "w.is_rainy AND c.day_type = 'weekend_holiday'",
     lambda r: _rainy(r) and _weekend(r)),
    ("rain_weekday", "Rain: slightly busier than usual (about +5-8%)",
     "w.is_rainy AND c.day_type = 'weekday'",
     lambda r: _rainy(r) and r["day_type"] == "weekday"),
    ("hot", f"Hot day ({HOT_DAY_C:.0f} °C or more): daytime demand drops (about -20 to -25%)",
     f"NOT w.is_rainy AND w.temp_max_c >= {HOT_DAY_C}",
     lambda r: _dry(r) and (_f(r, "temp_max_c") or 0) >= HOT_DAY_C),
    ("warm_weekend", f"Warm weekend ({WARM_WEEKEND_C:.0f} °C or more): slightly quieter (about -8%)",
     f"NOT w.is_rainy AND w.temp_max_c >= {WARM_WEEKEND_C} AND w.temp_max_c < {HOT_DAY_C} "
     "AND c.day_type = 'weekend_holiday'",
     lambda r: _dry(r) and WARM_WEEKEND_C <= (_f(r, "temp_max_c") or 0) < HOT_DAY_C
     and _weekend(r)),
    ("summer_holidays", "Summer school holidays: far fewer corporate, evening, family and club players",
     f"c.school_holiday_name = {_sql_str(SUMMER_HOLIDAYS)}",
     lambda r: r.get("school_holiday_name") == SUMMER_HOLIDAYS),
    ("school_holidays", "School holidays: fewer corporate and club players, more students and families in the daytime",
     f"c.is_school_holiday AND c.school_holiday_name <> {_sql_str(SUMMER_HOLIDAYS)}",
     lambda r: bool(r["is_school_holiday"]) and r.get("school_holiday_name") != SUMMER_HOLIDAYS),
    ("badmatin", "Bad'matin 09:30-11:00 blocks Courts 1-4",
     "c.day_type = 'weekday' AND c.day_of_week = 2 AND NOT c.is_school_holiday",
     lambda r: _tuesday_weekday(r) and not r["is_school_holiday"]),
    ("no_badmatin", "No Bad'matin in school holidays: Courts 1-4 free on Tuesday morning",
     "c.day_type = 'weekday' AND c.day_of_week = 2 AND c.is_school_holiday",
     lambda r: _tuesday_weekday(r) and bool(r["is_school_holiday"])),
    ("arena", "Event at the Vaudoise Aréna: evening demand a little lower (about -7%), parking busier",
     "c.arena_event",
     lambda r: bool(r["arena_event"])),
]


def demand_hint(row: Row) -> str:
    """Python twin of the function's demand_hint column (row: calendar columns plus is_rainy and temp_max_c)."""
    if row["is_closed"]:
        return CLOSED_HINT
    hints = [hint for _, hint, _, pred in RULES if pred(row)]
    return HINT_SEPARATOR.join(hints) or NORMAL_HINT


def opening_hours(day_type: str) -> str | None:
    window = pr.rentable_window(day_type)
    return None if window is None else f"{window[0]:%H:%M}-{window[1]:%H:%M}"


def _ident(*parts: str) -> None:
    for p in parts:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", p):
            raise ValueError(f"Invalid identifier: {p!r}")


def function_name(catalog: str, schema_prefix: str) -> str:
    _ident(catalog, schema_prefix)
    return f"`{catalog}`.`{schema_prefix}_gold`.{FUNCTION_NAME}"


def function_sql(catalog: str, schema_prefix: str) -> str:
    _ident(catalog, schema_prefix)
    silver = f"`{catalog}`.`{schema_prefix}_silver`"
    returns = ",\n    ".join(f"{n} {t} COMMENT {_sql_str(c)}" for n, t, c in COLUMNS)
    hours = "\n".join(f"    WHEN {_sql_str(t)} THEN {_sql_str(opening_hours(t))}"
                      for t in ("weekday", "weekend_holiday"))
    hints = ",\n".join(f"      CASE WHEN {cond} THEN {_sql_str(hint)} END" for _, hint, cond, _ in RULES)
    p = FUNCTION_NAME
    return f"""CREATE OR REPLACE FUNCTION {function_name(catalog, schema_prefix)}(
  start_date DATE COMMENT 'First day',
  n_days INT DEFAULT 7 COMMENT 'Number of days (1-{MAX_DAYS})'
)
RETURNS TABLE (
    {returns}
)
COMMENT 'B4L day outlook (read-only): weather, Vaud public and school holidays, Vaudoise Aréna events and a demand hint per day. Météo, jours fériés et vacances scolaires.'
RETURN
SELECT
  c.date,
  c.day_name,
  c.day_type,
  CASE c.day_type
{hours}
  END AS opening_hours,
  w.condition,
  round(w.temp_mean_c, 1) AS temp_mean_c,
  round(w.temp_max_c, 1) AS temp_max_c,
  round(w.precip_mm, 1) AS precip_mm,
  w.is_rainy,
  w.is_forecast,
  c.holiday_name AS public_holiday,
  c.school_holiday_name AS school_holiday,
  c.arena_event,
  CASE WHEN c.is_closed THEN {_sql_str(CLOSED_HINT)} ELSE coalesce(nullif(concat_ws({_sql_str(HINT_SEPARATOR)},
{hints}
  ), ''), {_sql_str(NORMAL_HINT)}) END AS demand_hint
FROM {silver}.calendar c
LEFT JOIN {silver}.weather w ON w.date = c.date
WHERE c.date >= {p}.start_date
  AND c.date < date_add({p}.start_date, least(greatest(coalesce({p}.n_days, 7), 1), {MAX_DAYS}))"""


def grant_statements(catalog: str, schema_prefix: str, principal: str) -> list[str]:
    """EXECUTE on the function only; USE CATALOG and USE SCHEMA on gold come from the Genie setup grants."""
    if not re.fullmatch(r"[A-Za-z0-9-]+", principal):
        raise ValueError(f"Invalid principal: {principal!r}")
    return [f"GRANT EXECUTE ON FUNCTION {function_name(catalog, schema_prefix)} TO `{principal}`"]
